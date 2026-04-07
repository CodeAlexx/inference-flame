//! ZImage NextDiT inference CLI — pure Rust.
//!
//! Staged pipeline with memory management:
//!   1. Load text embeddings (pre-computed)
//!   2. Denoise with block-swapped transformer
//!   3. Drop transformer weights
//!   4. Load VAE, decode latents to RGB
//!   5. Save PNG
//!
//! Usage:
//!   cargo run --release --bin zimage_infer -- \
//!       --model /path/to/zimage.safetensors \
//!       --vae /path/to/vae/diffusion_pytorch_model.safetensors \
//!       --embeddings /path/to/text_embeddings.safetensors \
//!       --output /path/to/output.png \
//!       --height 1024 --width 1024

use cudarc::driver::CudaDevice;
use flame_core::serialization::{load_file, load_file_filtered};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

use inference_flame::models::zimage_nextdit::NextDiT;
use inference_flame::sampling::euler::euler_step;
use inference_flame::sampling::schedules::build_sigma_schedule;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    vae_path: String,
    embeddings_path: String,
    output_path: String,
    height: usize,
    width: usize,
    steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut vae_path = String::new();
    let mut embeddings_path = String::new();
    let mut output_path = String::from("output/zimage_output.png");
    let mut height: usize = 1024;
    let mut width: usize = 1024;
    let mut steps: usize = 8;
    let mut cfg_scale: f32 = 0.0;
    let mut shift: f32 = 3.0;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--vae" => {
                i += 1;
                vae_path = args[i].clone();
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
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_path.is_empty() || embeddings_path.is_empty() {
        eprintln!("Usage: zimage_infer --model <path> --embeddings <path> [options]");
        eprintln!("  --model       Path to ZImage safetensors weights");
        eprintln!("  --vae         Path to VAE safetensors (optional, skips decode if omitted)");
        eprintln!("  --embeddings  Path to pre-computed text embeddings (safetensors)");
        eprintln!("  --output      Output path (default: output.png)");
        eprintln!("  --height      Image height in pixels (default: 1024)");
        eprintln!("  --width       Image width in pixels (default: 1024)");
        eprintln!("  --steps       Number of denoising steps (default: 8)");
        eprintln!("  --cfg         Classifier-free guidance scale (default: 0.0 = off)");
        eprintln!("  --shift       Sigma schedule shift (default: 3.0)");
        eprintln!("  --seed        Random seed (default: 42)");
        std::process::exit(1);
    }

    Args {
        model_path,
        vae_path,
        embeddings_path,
        output_path,
        height,
        width,
        steps,
        cfg_scale,
        shift,
        seed,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let t_total = Instant::now();
    println!("=== ZImage NextDiT Inference (Pure Rust) ===\n");

    // A/B test: disable autograd recording to measure its overhead on
    // inference. When FLAME_AUTOGRAD_OFF=1, skip the tape entirely.
    if std::env::var("FLAME_AUTOGRAD_OFF").ok().as_deref() == Some("1") {
        println!("[+] AUTOGRAD DISABLED via FLAME_AUTOGRAD_OFF=1");
        flame_core::AutogradContext::set_enabled(false);
    }

    let args = parse_args();

    // Create CUDA device
    let device = CudaDevice::new(0).map_err(|e| {
        Error::InvalidOperation(format!("Failed to create CUDA device: {e:?}"))
    })?;
    println!("[+] CUDA device initialized");

    // Validate dimensions
    let latent_h = args.height / 8; // VAE downscale factor
    let latent_w = args.width / 8;
    println!(
        "[+] Image: {}x{} -> Latent: {}x{} -> Patches: {}x{}",
        args.height,
        args.width,
        latent_h,
        latent_w,
        latent_h / 2,
        latent_w / 2,
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

    let cap_feats_uncond = emb_tensors.get("cap_feats_uncond").map(|t| {
        println!("    cap_feats_uncond shape: {:?}", t.shape().dims());
        t.clone()
    });
    let cap_feats_uncond_ref = cap_feats_uncond.as_ref();

    // ==================================================================
    // Stage 2: Denoise (all weights resident on GPU)
    // ==================================================================
    println!("\n--- Stage 2: Denoise ---");

    // Load ALL transformer weights into GPU via mmap (12.3GB fits in 24GB).
    // After denoising, we drop everything to make room for VAE.
    println!("[+] Loading all transformer weights to GPU (mmap)...");
    let t_load = Instant::now();
    let all_weights = load_file_filtered(&args.model_path, &device, |_| true)?;
    println!(
        "    Loaded {} tensors in {:.1}s",
        all_weights.len(),
        t_load.elapsed().as_secs_f32(),
    );

    let mut model = NextDiT::new_resident(all_weights, device.clone());

    // Initialize random latents
    println!("[+] Generating initial noise (seed={})", args.seed);
    let x = Tensor::randn(
        Shape::from_dims(&[1, 16, latent_h, latent_w]),
        0.0,
        1.0,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    // Build sigma schedule
    let sigmas = build_sigma_schedule(args.steps, args.shift);
    println!(
        "[+] Sigma schedule: {} steps, shift={}, range [{:.4}, {:.4}]",
        args.steps,
        args.shift,
        sigmas.first().unwrap_or(&0.0),
        sigmas.last().unwrap_or(&0.0),
    );

    // Denoising loop
    let t_denoise = Instant::now();
    println!("[+] Starting denoising loop ({} steps)...", args.steps);
    let mut x = x;
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        x = euler_step(
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
    println!("    Output latent shape: {:?}", x.shape().dims());

    // ==================================================================
    // Stage 3: Drop transformer, load VAE, decode
    // ==================================================================
    // Drop model to free VRAM before loading VAE
    drop(model);
    // Drop embeddings too
    drop(cap_feats_uncond);

    if args.vae_path.is_empty() {
        // No VAE — save raw latents
        println!("\n[+] No --vae specified, saving raw latents");
        let mut output_map = HashMap::new();
        output_map.insert("latents".to_string(), x);
        flame_core::serialization::save_file(&output_map, &args.output_path)?;
        println!("    Saved to: {}", args.output_path);
    } else {
        println!("\n--- Stage 3: VAE Decode ---");
        let t_vae = Instant::now();

        // Z-Image VAE: 16 latent channels, scale=0.3611, shift=0.1159
        let vae = LdmVAEDecoder::from_safetensors(
            &args.vae_path,
            16,
            0.3611,
            0.1159,
            &device,
        )?;

        println!("[+] Decoding latents...");
        let rgb = vae.decode(&x)?;
        println!(
            "    Decoded: {:?} in {:.1}s",
            rgb.shape().dims(),
            t_vae.elapsed().as_secs_f32()
        );

        // Drop VAE to free VRAM
        drop(vae);
        drop(x);

        // ==============================================================
        // Stage 4: Save PNG
        // ==============================================================
        println!("\n--- Stage 4: Save PNG ---");

        let rgb_f32 = rgb.to_dtype(DType::F32)?;
        let data = rgb_f32.to_vec()?;
        let (_, _, out_h, out_w) = {
            let d = rgb_f32.shape().dims();
            (d[0], d[1], d[2], d[3])
        };

        // Convert NCHW float [-1, 1] to packed RGB u8
        let mut pixels = vec![0u8; out_h * out_w * 3];
        for y in 0..out_h {
            for x in 0..out_w {
                for c in 0..3 {
                    let idx = c * out_h * out_w + y * out_w + x;
                    let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                    pixels[(y * out_w + x) * 3 + c] = val;
                }
            }
        }

        let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
            .ok_or_else(|| Error::InvalidOperation("Failed to create image".into()))?;
        img.save(&args.output_path)
            .map_err(|e| Error::InvalidOperation(format!("Failed to save PNG: {e}")))?;

        println!("[+] Saved: {}", args.output_path);
    }

    println!("\n============================================================");
    println!("Total time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
