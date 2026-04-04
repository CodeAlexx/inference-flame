//! SDXL image generation — pure Rust.
//!
//! Usage: sdxl_infer [path_to_cached_embeddings.safetensors]
//!
//! Cached embeddings must contain:
//!   "context": [B, 77, 2048] — CLIP-L + CLIP-G cross-attention hidden states
//!   "y": [B, 2816] — pooled embedding (CLIP-L 768d + CLIP-G 1280d + zeros 768d)
//!
//! Generate cached embeddings with a Python script (CLIP encoders not in Rust yet).

use inference_flame::models::sdxl_unet::SDXLUNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors"; // VAE from combined checkpoint
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_rust.png";

const NUM_STEPS: usize = 30;
const CFG_SCALE: f32 = 7.5;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

// SDXL uses discrete noise schedule (beta_start=0.00085, beta_end=0.012, 1000 steps)
// Returns (sigmas, timesteps) — sigmas for Euler stepping, timesteps for UNet input
fn build_sdxl_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

    // Scaled-linear beta schedule (SDXL default: beta_schedule="scaled_linear")
    let betas: Vec<f64> = (0..num_train_steps)
        .map(|i| {
            let v = beta_start.sqrt()
                + (beta_end.sqrt() - beta_start.sqrt()) * i as f64 / (num_train_steps - 1) as f64;
            v * v
        })
        .collect();

    let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
    let mut prod = 1.0f64;
    for &b in &betas {
        prod *= 1.0 - b;
        alphas_cumprod.push(prod);
    }

    // Leading timestep spacing with steps_offset=1 (SDXL EulerDiscreteScheduler default)
    let step_ratio = num_train_steps / num_steps; // integer division
    let mut ts: Vec<usize> = (0..num_steps).map(|i| i * step_ratio + 1).collect();
    ts.reverse(); // high noise first

    let mut sigmas = Vec::with_capacity(num_steps + 1);
    let mut timesteps = Vec::with_capacity(num_steps);
    for &t in &ts {
        let t = t.min(num_train_steps - 1);
        let alpha = alphas_cumprod[t];
        let sigma = ((1.0 - alpha) / alpha).sqrt();
        sigmas.push(sigma as f32);
        timesteps.push(t as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let emb_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_EMB_PATH.to_string());

    println!("============================================================");
    println!("SDXL — Pure Rust Inference");
    println!("  {}x{}, {} steps, CFG {}, seed {}", WIDTH, HEIGHT, NUM_STEPS, CFG_SCALE, SEED);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(std::path::Path::new(&emb_path), &device)?;
    let context = emb.get("context")
        .ok_or_else(|| anyhow::anyhow!("Missing 'context' in embeddings. Run cache_sdxl_embeddings.py first."))?
        .clone();
    let y = emb.get("y")
        .ok_or_else(|| anyhow::anyhow!("Missing 'y' in embeddings."))?
        .clone();
    let context_uncond = emb.get("context_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'context_uncond' in embeddings."))?
        .clone();
    let y_uncond = emb.get("y_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'y_uncond' in embeddings."))?
        .clone();
    drop(emb);
    println!("  context: {:?}, y: {:?}", context.dims(), y.dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load SDXL UNet
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load SDXL UNet ---");
    let t0 = Instant::now();
    let mut model = SDXLUNet::from_safetensors_all_gpu(MODEL_PATH, &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Noise + denoise (Euler, eps prediction)
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, CFG_SCALE);

    let latent_h = HEIGHT / 8; // SDXL uses /8 downscale
    let latent_w = WIDTH / 8;
    let numel = 4 * latent_h * latent_w; // 4 latent channels

    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut v = Vec::with_capacity(numel);
        for _ in 0..numel / 2 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(r * theta.cos());
            v.push(r * theta.sin());
        }
        v
    };

    let (sigmas, timesteps) = build_sdxl_schedule(NUM_STEPS);
    println!("  sigma_max={:.4}, sigma_min={:.6}", sigmas[0], sigmas[NUM_STEPS - 1]);
    println!("  timestep_max={:.0}, timestep_min={:.0}", timesteps[0], timesteps[NUM_STEPS - 1]);

    // Initialize x = noise * init_noise_sigma (sqrt(sigma_max^2 + 1) per diffusers)
    let init_sigma = (sigmas[0] * sigmas[0] + 1.0).sqrt();
    let mut x = Tensor::from_f32_to_bf16(
        noise_data, Shape::from_dims(&[1, 4, latent_h, latent_w]), device.clone(),
    )?.mul_scalar(init_sigma)?;

    let t0 = Instant::now();
    for i in 0..NUM_STEPS {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Scale input: x_in = x / sqrt(sigma^2 + 1)
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_in = x.mul_scalar(c_in)?;

        // UNet expects discrete timestep (0-999), NOT sigma
        let timestep = Tensor::from_f32_to_bf16(
            vec![timesteps[i]], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Conditional + unconditional predictions (eps-prediction)
        let pred_cond = model.forward(&x_in, &timestep, &context, &y)?;
        let pred_uncond = model.forward(&x_in, &timestep, &context_uncond, &y_uncond)?;

        // CFG
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;

        // Euler step for eps-prediction:
        // derivative = eps (for epsilon prediction, d(x)/d(sigma) = eps)
        // x_next = x + eps * (sigma_next - sigma)
        let dt = sigma_next - sigma;
        x = x.add(&pred.mul_scalar(dt)?)?;

        {
            let pred_f32 = pred.to_dtype(DType::F32)?;
            let data = pred_f32.to_vec()?;
            let mean_abs: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
            let x_f32 = x.to_dtype(DType::F32)?;
            let xd = x_f32.to_vec()?;
            let x_abs: f32 = xd.iter().map(|v| v.abs()).sum::<f32>() / xd.len() as f32;
            println!("  Step {}/{}: t={:.0}, sigma={:.4}, pred_abs={:.4}, x_abs={:.4}, dt={:.4}",
                i + 1, NUM_STEPS, timesteps[i], sigma, mean_abs, x_abs, dt);
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: Save latent for Python VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Save latent ---");
    drop(model);

    let latent_path = OUTPUT_PATH.replace(".png", "_latent.safetensors");
    {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert("latent".to_string(), x.clone());
        flame_core::serialization::save_file(&tensors, std::path::Path::new(&latent_path))?;
        println!("  Saved latent to {}", latent_path);
    }

    // Decode with Python: python decode_sdxl_latent.py
    println!("  Run: python decode_sdxl_latent.py");

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
