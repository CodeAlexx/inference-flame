//! Anima (Cosmos Predict2) image generation — pure Rust.
//!
//! Usage: anima_infer [path_to_cached_embeddings.safetensors]
//!
//! Cached embeddings contain Qwen3 0.6B hidden states + token IDs.

use inference_flame::models::anima::{Anima, load_resident_weights, count_blocks_from_file};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors";
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/anima_rust.png";

// Cosmos RFLOW sampling with shift=3.0
const NUM_STEPS: usize = 30;
const CFG_SCALE: f32 = 7.0;
const SHIFT: f32 = 3.0;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

/// Cosmos RFLOW sigma schedule.
fn build_cosmos_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    // Linear timesteps from 1→0, then shift
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let emb_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_EMB_PATH.to_string());

    println!("============================================================");
    println!("Anima — Pure Rust Inference");
    println!("  {}x{}, {} steps, CFG {}, seed {}", WIDTH, HEIGHT, NUM_STEPS, CFG_SCALE, SEED);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(std::path::Path::new(&emb_path), &device)?;
    let llm_hidden = emb.get("llm_hidden").ok_or_else(|| anyhow::anyhow!("Missing llm_hidden"))?.clone();
    let token_ids = emb.get("token_ids").ok_or_else(|| anyhow::anyhow!("Missing token_ids"))?.clone();
    let neg_llm_hidden = emb.get("neg_llm_hidden").ok_or_else(|| anyhow::anyhow!("Missing neg_llm_hidden"))?.clone();
    let neg_token_ids = emb.get("neg_token_ids").ok_or_else(|| anyhow::anyhow!("Missing neg_token_ids"))?.clone();
    drop(emb);
    println!("  llm_hidden: {:?}, token_ids: {:?}", llm_hidden.dims(), token_ids.dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load Anima model (block offloading)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Anima model ---");
    let t0 = Instant::now();
    let resident = load_resident_weights(MODEL_PATH, &device)?;
    let num_blocks = count_blocks_from_file(MODEL_PATH, &device)?;
    println!("  {} blocks, {} resident keys", num_blocks, resident.len());
    let mut model = Anima::new(MODEL_PATH.to_string(), resident, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Create noise + denoise
    // ------------------------------------------------------------------
    // Anima latent: [B, T, H/8, W/8, 16] — for images T=1
    let latent_h = HEIGHT / 8;
    let latent_w = WIDTH / 8;
    let t_frames = 1usize;
    let numel = t_frames * latent_h * latent_w * 16;

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

    let timesteps = build_cosmos_schedule(NUM_STEPS, SHIFT);

    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, CFG_SCALE);
    println!("  t[0]={:.4}, t[-2]={:.4}", timesteps[0], timesteps[NUM_STEPS - 1]);

    // x = noise (Cosmos RFLOW initial state)
    let mut x = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, t_frames, latent_h, latent_w, 16]),
        device.clone(),
    )?;

    let t0 = Instant::now();
    for i in 0..NUM_STEPS {
        let t_curr = timesteps[i];
        let t_prev = timesteps[i + 1];

        // Cosmos RFLOW: s = sigma / (sigma + 1)
        let s = t_curr / (t_curr + 1.0);

        // Model input: x * (1 - s)
        let x_in = x.mul_scalar(1.0 - s)?;

        let t_vec = Tensor::from_f32_to_bf16(
            vec![t_curr], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Conditional
        let pred_cond = model.forward(&x_in, &t_vec, &token_ids, &llm_hidden)?;
        // Unconditional
        let pred_uncond = model.forward(&x_in, &t_vec, &neg_token_ids, &neg_llm_hidden)?;

        // CFG
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;

        // Cosmos RFLOW denoised: x_in * (1-s) - pred * s
        let denoised = x_in.mul_scalar(1.0 - s)?.sub(&pred.mul_scalar(s)?)?;

        // Euler step: x = x + (t_prev - t_curr) / t_curr * (denoised - x)
        // Actually for RFLOW: x_next = x + dt * velocity
        // where velocity = (denoised - x) / t_curr
        let dt = t_prev - t_curr;
        let d = x.sub(&denoised)?.mul_scalar(1.0 / t_curr)?;
        x = x.add(&d.mul_scalar(dt)?)?;

        if i % 5 == 0 || i == NUM_STEPS - 1 {
            println!("  Step {}/{}: t={:.4}", i + 1, NUM_STEPS, t_curr);
        }
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);
    println!("  Output: {:?}", x.dims());

    // ------------------------------------------------------------------
    // Stage 4: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);

    // Anima latent is [B, T=1, H, W, 16] → need [B, 16, H, W] for VAE
    // Permute: [1, 1, H, W, 16] → squeeze T → [1, H, W, 16] → [1, 16, H, W]
    let latent = x.reshape(&[1, latent_h, latent_w, 16])?
        .permute(&[0, 3, 1, 2])?;
    println!("  Latent for VAE: {:?}", latent.dims());

    // Load Wan21 VAE - TODO: use wan21_vae.rs once tested
    // For now, try LDM VAE format
    let vae_weights = flame_core::serialization::load_file(
        std::path::Path::new(VAE_PATH), &device
    )?;
    println!("  VAE weights: {} keys", vae_weights.len());

    // Check if this is LDM format or Wan format
    let has_decoder = vae_weights.keys().any(|k| k.starts_with("decoder."));
    if has_decoder {
        println!("  Using LDM VAE decoder");
        let vae = inference_flame::vae::ldm_decoder::LdmVAEDecoder::from_weights(
            vae_weights, 16, 1.0, 0.0, &device
        )?;
        let rgb = vae.decode(&latent)?;
        println!("  Decoded: {:?}", rgb.dims());
        println!("  VAE in {:.1}s", t0.elapsed().as_secs_f32());

        // Save
        let rgb_f32 = rgb.to_dtype(DType::F32)?;
        let data = rgb_f32.to_vec()?;
        let d = rgb_f32.dims();
        let (out_h, out_w) = (d[2], d[3]);

        let mut pixels = vec![0u8; out_h * out_w * 3];
        for y in 0..out_h {
            for px in 0..out_w {
                for c in 0..3 {
                    let idx = c * out_h * out_w + y * out_w + px;
                    let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                    pixels[(y * out_w + px) * 3 + c] = val;
                }
            }
        }

        image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("Image creation failed"))?
            .save(OUTPUT_PATH)?;
    } else {
        println!("  VAE format not recognized — skipping decode");
        println!("  Latent saved but no image output");
    }

    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
