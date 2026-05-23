//! L2P (T2I-L2P) inference CLI — pure Rust, pixel-space.
//!
//! Pipeline:
//!   1. Load pre-computed text embeddings (`cap_feats` + optional `cap_feats_uncond`).
//!   2. Load L2P safetensors. Translate keys (QKV fusion, ModuleList unwrap,
//!      norm rename, embedder rename) via `weight_loader::load_l2p_safetensors`.
//!   3. Build `L2pDiT::new_resident`.
//!   4. Initialize F32 pixel noise `[1, 3, H, W]`. Cast to BF16 (the DiT
//!      `debug_assert_eq`'s on BF16 input).
//!   5. Run the Euler flow-matching denoise loop (`l2p_euler_step`).
//!      Output is pixel-space `[1, 3, H, W]` BF16 — **no VAE decode**.
//!   6. Convert BF16 → F32, clamp [-1, 1], remap to [0, 255] u8, save PNG.
//!
//! Usage:
//!   cargo run --release --bin l2p_infer -- \
//!       --model /path/to/model-1k-merge.safetensors \
//!       --embeddings /path/to/text_embeddings.safetensors \
//!       --output l2p_output.png

use cudarc::driver::CudaDevice;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result};
use std::time::Instant;

use inference_flame::lora::LoraStack;
use inference_flame::models::l2p::weight_loader::load_l2p_safetensors;
use inference_flame::models::l2p::L2pDiT;
use inference_flame::sampling::l2p_sampling::{
    build_l2p_sigma_schedule, init_l2p_noise, l2p_euler_step,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    embeddings_path: String,
    output_path: String,
    height: usize,
    width: usize,
    steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
    lora_path: Option<String>,
    lora_multiplier: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut embeddings_path = String::new();
    let mut output_path = String::from("l2p_output.png");
    let mut height: usize = 1024;
    let mut width: usize = 1024;
    let mut steps: usize = 30;
    let mut cfg_scale: f32 = 2.0;
    let mut shift: f32 = 3.0;
    let mut seed: u64 = 42;
    let mut lora_path: Option<String> = None;
    let mut lora_multiplier: f32 = 1.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--embeddings" => {
                i += 1;
                embeddings_path = args[i].clone();
            }
            "--output" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--height" => {
                i += 1;
                height = args[i].parse().expect("Invalid height");
            }
            "--width" => {
                i += 1;
                width = args[i].parse().expect("Invalid width");
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("Invalid steps");
            }
            "--cfg" => {
                i += 1;
                cfg_scale = args[i].parse().expect("Invalid cfg");
            }
            "--shift" => {
                i += 1;
                shift = args[i].parse().expect("Invalid shift");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid seed");
            }
            "--lora" => {
                i += 1;
                lora_path = Some(args[i].clone());
            }
            "--lora-multiplier" => {
                i += 1;
                lora_multiplier = args[i].parse().expect("Invalid lora-multiplier");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_path.is_empty() || embeddings_path.is_empty() {
        eprintln!("Usage: l2p_infer --model <path> --embeddings <path> [options]");
        eprintln!("  --model       Path to L2P merged safetensors (model-1k-merge.safetensors)");
        eprintln!(
            "  --embeddings  Path to pre-computed text embeddings (safetensors with cap_feats + optional cap_feats_uncond)"
        );
        eprintln!("  --output      Output PNG path (default: l2p_output.png)");
        eprintln!("  --height      Image height in pixels (default: 1024)");
        eprintln!("  --width       Image width in pixels (default: 1024)");
        eprintln!("  --steps       Number of denoising steps (default: 30)");
        eprintln!("  --cfg         CFG scale (default: 2.0; per L2P README)");
        eprintln!("  --shift       Sigma schedule shift (default: 3.0)");
        eprintln!("  --seed        Random seed (default: 42)");
        eprintln!("  --lora        Optional LoRA safetensors to attach via set_lora");
        eprintln!("  --lora-multiplier  LoRA strength multiplier (default: 1.0)");
        std::process::exit(1);
    }

    Args {
        model_path,
        embeddings_path,
        output_path,
        height,
        width,
        steps,
        cfg_scale,
        shift,
        seed,
        lora_path,
        lora_multiplier,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let t_total = Instant::now();
    println!("=== L2P (T2I-L2P) Inference (Pure Rust, pixel-space) ===\n");

    // Diagnostic toggle: disable autograd recording during inference. Same
    // pattern as `zimage_infer`; not a default-path swap.
    if std::env::var("FLAME_AUTOGRAD_OFF").ok().as_deref() == Some("1") {
        println!("[+] AUTOGRAD DISABLED via FLAME_AUTOGRAD_OFF=1");
        flame_core::AutogradContext::set_enabled(false);
    }

    let args = parse_args();

    let device = CudaDevice::new(0).map_err(|e| {
        Error::InvalidOperation(format!("Failed to create CUDA device: {e:?}"))
    })?;
    println!("[+] CUDA device initialized");

    // Sanity: L2P uses 16×16 patches, so H and W should be multiples of 16
    // (otherwise the U-Net bottleneck nearest-interp fallback fires, which
    // is supported but unusual at inference).
    if args.height % 16 != 0 || args.width % 16 != 0 {
        eprintln!(
            "WARNING: height={} width={} not multiples of 16; U-Net bottleneck will nearest-interp the DiT feat map",
            args.height, args.width
        );
    }
    println!(
        "[+] Image: {}x{} -> Patches: {}x{}",
        args.height,
        args.width,
        args.height / 16,
        args.width / 16,
    );

    // ==================================================================
    // Stage 1: Load text embeddings
    // ==================================================================
    println!("\n--- Stage 1: Text Embeddings ---");
    let emb_tensors = load_file(&args.embeddings_path, &device)?;

    let cap_feats = emb_tensors.get("cap_feats").ok_or_else(|| {
        Error::InvalidInput(
            "Embeddings file must contain 'cap_feats' key (B, seq, 2560)".into(),
        )
    })?;
    let cap_feats = cap_feats.to_dtype(DType::BF16)?;
    println!("    cap_feats shape: {:?}", cap_feats.shape().dims());

    let cap_feats_uncond = match emb_tensors.get("cap_feats_uncond") {
        Some(t) => {
            println!("    cap_feats_uncond shape: {:?}", t.shape().dims());
            Some(t.to_dtype(DType::BF16)?)
        }
        None => {
            if args.cfg_scale > 1.0 {
                eprintln!(
                    "WARNING: cfg={} > 1.0 but embeddings file has no 'cap_feats_uncond'; CFG will be silently disabled",
                    args.cfg_scale
                );
            }
            None
        }
    };
    let cap_feats_uncond_ref = cap_feats_uncond.as_ref();

    // ==================================================================
    // Stage 2: Load L2P weights (translated to internal layout) + build model
    // ==================================================================
    println!("\n--- Stage 2: Load L2P Model ---");
    let t_load = Instant::now();
    let translated = load_l2p_safetensors(std::path::Path::new(&args.model_path), &device)?;
    println!(
        "    Loaded + translated {} tensors in {:.1}s",
        translated.len(),
        t_load.elapsed().as_secs_f32(),
    );

    // Optional LoRA — capture base keys BEFORE moving `translated` into the
    // model so we can match LoRA targets against the resident weight set.
    let base_keys: std::collections::HashSet<String> =
        translated.keys().cloned().collect();

    let mut model = L2pDiT::new_resident(translated, device.clone());

    if let Some(ref lora_path) = args.lora_path {
        println!("\n--- LoRA: {} (mult={:.2}) ---", lora_path, args.lora_multiplier);
        let t_lora = Instant::now();
        let lora_stack = LoraStack::load(
            lora_path,
            &base_keys,
            args.lora_multiplier,
            &device,
        )?;
        println!(
            "    {} target weight(s), loaded in {:.1}s",
            lora_stack.target_count(),
            t_lora.elapsed().as_secs_f32(),
        );
        model.set_lora(Arc::new(lora_stack));
    }

    // ==================================================================
    // Stage 3: Initial noise (F32 per L2P convention → cast to BF16)
    // ==================================================================
    println!("\n--- Stage 3: Init Noise ---");
    println!("[+] Generating initial F32 pixel noise (seed={})", args.seed);
    let x_f32 = init_l2p_noise(args.height, args.width, args.seed, &device)?;
    // The DiT `debug_assert_eq`'s on BF16. F32 → BF16 cast happens here, per
    // PORT_SPEC §"Special / things to watch" #4 (noise gen is F32; model
    // dtype is BF16).
    let x = x_f32.to_dtype(DType::BF16)?;
    drop(x_f32);

    // ==================================================================
    // Stage 4: Denoise
    // ==================================================================
    println!("\n--- Stage 4: Denoise ---");
    let sigmas = build_l2p_sigma_schedule(args.steps, args.shift);
    println!(
        "[+] Sigma schedule: {} steps, shift={}, range [{:.4}, {:.4}]",
        args.steps,
        args.shift,
        sigmas.first().unwrap_or(&0.0),
        sigmas.last().unwrap_or(&0.0),
    );
    println!(
        "[+] CFG: scale={} (uncond={})",
        args.cfg_scale,
        cap_feats_uncond_ref.is_some()
    );

    let t_denoise = Instant::now();
    println!("[+] Starting denoising loop ({} steps)...", args.steps);
    let mut x = x;
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        x = l2p_euler_step(
            &mut model,
            &x,
            sigma,
            sigma_next,
            &cap_feats,
            cap_feats_uncond_ref,
            args.cfg_scale,
        )?;

        println!(
            "  Step {}/{}: sigma={:.4} -> {:.4} ({:.1}s)",
            step + 1,
            args.steps,
            sigma,
            sigma_next,
            t_step.elapsed().as_secs_f32(),
        );
    }
    println!(
        "[+] Denoising complete in {:.1}s",
        t_denoise.elapsed().as_secs_f32()
    );
    println!("    Output pixel shape: {:?}", x.shape().dims());

    // ==================================================================
    // Stage 5: Save PNG
    // ==================================================================
    // L2P output is already pixel-space [-1, 1] BF16 — no VAE decode.
    // Drop the model before we materialize the F32 copy so we keep peak
    // VRAM down (the model is the largest resident on GPU at this point).
    drop(model);
    drop(cap_feats_uncond);

    println!("\n--- Stage 5: Save PNG ---");

    let rgb_f32 = x.to_dtype(DType::F32)?;
    drop(x);
    let data = rgb_f32.to_vec()?;
    let (_, _, out_h, out_w) = {
        let d = rgb_f32.shape().dims();
        (d[0], d[1], d[2], d[3])
    };

    // NCHW float [-1, 1] -> packed RGB u8
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for xp in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + xp;
                let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * out_w + xp) * 3 + c] = val;
            }
        }
    }

    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| Error::InvalidOperation("Failed to create image".into()))?;
    img.save(&args.output_path)
        .map_err(|e| Error::InvalidOperation(format!("Failed to save PNG: {e}")))?;

    println!("[+] Saved: {}", args.output_path);

    println!("\n============================================================");
    println!("Total time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
