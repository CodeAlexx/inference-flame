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

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors"; // VAE embedded in checkpoint
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_rust.png";

const NUM_STEPS: usize = 30;
const CFG_SCALE: f32 = 7.5;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

// SDXL uses discrete noise schedule (beta_start=0.00085, beta_end=0.012, 1000 steps)
fn build_sdxl_sigmas(num_steps: usize) -> Vec<f32> {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

    // Linear beta schedule → cumulative alpha → sigma
    let betas: Vec<f64> = (0..num_train_steps)
        .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (num_train_steps - 1) as f64)
        .collect();

    let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
    let mut prod = 1.0f64;
    for &b in &betas {
        prod *= 1.0 - b;
        alphas_cumprod.push(prod);
    }

    // Pick evenly spaced timesteps
    let step_ratio = num_train_steps as f64 / num_steps as f64;
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        let t = (num_train_steps as f64 - 1.0 - i as f64 * step_ratio).round() as usize;
        let t = t.min(num_train_steps - 1);
        let alpha = alphas_cumprod[t];
        let sigma = ((1.0 - alpha) / alpha).sqrt();
        sigmas.push(sigma as f32);
    }
    sigmas.push(0.0);
    sigmas
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
    let mut model = SDXLUNet::from_safetensors(MODEL_PATH, &device)?;
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

    let sigmas = build_sdxl_sigmas(NUM_STEPS);
    println!("  sigma_max={:.4}, sigma_min={:.6}", sigmas[0], sigmas[NUM_STEPS - 1]);

    // Initialize x = noise * sigma_max
    let mut x = Tensor::from_f32_to_bf16(
        noise_data, Shape::from_dims(&[1, 4, latent_h, latent_w]), device.clone(),
    )?.mul_scalar(sigmas[0])?;

    let t0 = Instant::now();
    for i in 0..NUM_STEPS {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Timestep from sigma: t = sigma_to_timestep(sigma)
        // For EulerDiscreteScheduler: input to UNet is x / sqrt(sigma^2 + 1)
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_in = x.mul_scalar(c_in)?;

        // Timestep value (continuous)
        let timestep = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Conditional prediction
        let pred_cond = model.forward(&x_in, &timestep, &context, &y)?;
        // Unconditional prediction
        let pred_uncond = model.forward(&x_in, &timestep, &context_uncond, &y_uncond)?;

        // CFG
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;

        // Euler step (sigma-based): x = x + (sigma_next - sigma) * (x - pred) / sigma
        let d = x.sub(&pred)?.mul_scalar(1.0 / sigma)?; // noise direction
        let dt = sigma_next - sigma;
        x = x.add(&d.mul_scalar(dt)?)?;

        if i % 10 == 0 || i == NUM_STEPS - 1 {
            println!("  Step {}/{}: sigma={:.4}", i + 1, NUM_STEPS, sigma);
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);

    // SDXL VAE: 4ch latent, scale=0.13025, shift=0.0
    let vae = LdmVAEDecoder::from_safetensors(VAE_PATH, 4, 0.13025, 0.0, &device)?;
    let rgb = vae.decode(&x)?;
    println!("  {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 5: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let d = rgb_f32.dims();
    let (out_h, out_w) = (d[2], d[3]);

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

    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image creation failed"))?
        .save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
