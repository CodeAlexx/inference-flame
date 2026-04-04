//! ZImage NextDiT inference CLI — pure Rust.
//!
//! Usage:
//!   cargo run --bin zimage_infer -- \
//!       --model /path/to/zimage.safetensors \
//!       --embeddings /path/to/text_embeddings.safetensors \
//!       --output /path/to/output_latents.safetensors \
//!       --height 1024 --width 1024 --steps 30 --cfg 4.0

use cudarc::driver::CudaDevice;
use flame_core::serialization::{load_file, load_file_filtered, save_file};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use inference_flame::models::zimage_nextdit::NextDiT;
use inference_flame::sampling::euler::euler_step;
use inference_flame::sampling::schedules::build_sigma_schedule;

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
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut embeddings_path = String::new();
    let mut output_path = String::from("output_latents.safetensors");
    let mut height: usize = 1024;
    let mut width: usize = 1024;
    let mut steps: usize = 30;
    let mut cfg_scale: f32 = 4.0;
    let mut shift: f32 = 1.0;
    let mut seed: u64 = 42;

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
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Usage: zimage_infer --model <path> --embeddings <path> [options]");
        eprintln!("  --model       Path to ZImage safetensors weights");
        eprintln!("  --embeddings  Path to pre-computed text embeddings (safetensors)");
        eprintln!("  --output      Output latents path (default: output_latents.safetensors)");
        eprintln!("  --height      Image height in pixels (default: 1024)");
        eprintln!("  --width       Image width in pixels (default: 1024)");
        eprintln!("  --steps       Number of denoising steps (default: 30)");
        eprintln!("  --cfg         Classifier-free guidance scale (default: 4.0)");
        eprintln!("  --shift       Sigma schedule shift (default: 1.0)");
        eprintln!("  --seed        Random seed (default: 42)");
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
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    println!("=== ZImage NextDiT Inference (Pure Rust) ===\n");

    let args = parse_args();

    // Create CUDA device
    let device = CudaDevice::new(0).map_err(|e| {
        Error::InvalidOperation(format!("Failed to create CUDA device: {e:?}"))
    })?;
    let device = Arc::new(device);
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

    // Load only small resident weights (embedders, final layer, pad tokens)
    // Block weights (~440MB each) are streamed from disk via mmap on demand
    println!("[+] Loading resident weights from: {}", args.model_path);
    let resident_prefixes = [
        "x_embedder.",
        "cap_embedder.",
        "t_embedder.",
        "final_layer.",
        "x_pad_token",
        "cap_pad_token",
    ];
    let resident = load_file_filtered(&args.model_path, &device, |key| {
        resident_prefixes.iter().any(|p| key.starts_with(p))
    })?;
    println!(
        "    Loaded {} resident tensors (embedders + final layer)",
        resident.len()
    );

    // Print a few key shapes for sanity
    if let Some(t) = resident.get("x_embedder.weight") {
        println!("    x_embedder.weight: {:?}", t.shape().dims());
    }

    let mut model = NextDiT::new(args.model_path.clone(), resident, device.clone());

    // Load text embeddings
    println!(
        "[+] Loading text embeddings from: {}",
        args.embeddings_path
    );
    let emb_tensors = load_file(&args.embeddings_path, &device)?;

    // Expect keys: "cap_feats" (B, seq, 2560) and optionally "cap_feats_uncond"
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
    println!("\n[+] Starting denoising loop ({} steps)...", args.steps);
    let mut x = x;
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        println!(
            "  Step {}/{}: sigma={:.6} -> {:.6}",
            step + 1,
            args.steps,
            sigma,
            sigma_next,
        );

        x = euler_step(
            &mut model,
            &x,
            sigma,
            sigma_next,
            &cap_feats,
            cap_feats_uncond_ref,
            args.cfg_scale,
        )?;
    }

    println!("\n[+] Denoising complete!");
    println!("    Output latent shape: {:?}", x.shape().dims());

    // Save raw latents
    println!("[+] Saving latents to: {}", args.output_path);
    let mut output_map = HashMap::new();
    output_map.insert("latents".to_string(), x);
    save_file(&output_map, &args.output_path)?;

    println!("\n=== Done! ===");
    println!("    To decode: load latents and run through VAE decoder.");

    Ok(())
}
