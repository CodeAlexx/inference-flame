//! Anima (Cosmos Predict2) image generation — pure Rust.
//!
//! Usage: anima_infer [path_to_cached_embeddings.safetensors]
//!
//! Cached embeddings contain Qwen3 0.6B hidden states + T5 token IDs.
//! Model is 3.9GB — loads entirely to GPU (no block offloading).

use inference_flame::models::anima::{Anima, load_all_weights};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors";
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/anima_rust.png";

// Rectified flow: linear sigma schedule, no shift at inference
// HF recommends: 1024x1024, CFG 4-5, 30-50 steps
const NUM_STEPS: usize = 30;
const CFG_SCALE: f32 = 4.5;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let emb_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_EMB_PATH.to_string());

    println!("============================================================");
    println!("Anima — Pure Rust Inference (all-on-GPU)");
    println!("  {}x{}, {} steps, CFG {}, seed {}", WIDTH, HEIGHT, NUM_STEPS, CFG_SCALE, SEED);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load pre-computed context (adapter already run in Python)
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load pre-computed context ---");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(std::path::Path::new(&emb_path), &device)?;
    let context_cond = emb.get("context_cond").ok_or_else(|| anyhow::anyhow!("Missing context_cond"))?.clone();
    let context_uncond = emb.get("context_uncond").ok_or_else(|| anyhow::anyhow!("Missing context_uncond"))?.clone();
    drop(emb);
    println!("  context_cond: {:?}", context_cond.dims());
    {
        let ctx_f32 = context_cond.to_dtype(DType::F32)?;
        let data = ctx_f32.to_vec()?;
        let mean_abs: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
        println!("  Context mean_abs: {:.4}", mean_abs);
    }
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load Anima model — ALL weights on GPU (3.9GB fits easily)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Anima model (all-on-GPU) ---");
    let t0 = Instant::now();
    let all_weights = load_all_weights(MODEL_PATH, &device)?;
    println!("  {} weight tensors loaded", all_weights.len());
    let mut model = Anima::new_all_on_gpu(MODEL_PATH.to_string(), all_weights, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Create noise + denoise
    // ------------------------------------------------------------------
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

    // Linear schedule: sigma from 1.0 to 0.0, NO shift at inference
    let sigmas: Vec<f32> = (0..=NUM_STEPS)
        .map(|i| 1.0 - i as f32 / NUM_STEPS as f32)
        .collect();

    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, CFG_SCALE);
    println!("  Latent: [1, {}, {}, {}, 16]", t_frames, latent_h, latent_w);

    let mut x = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, t_frames, latent_h, latent_w, 16]),
        device.clone(),
    )?;

    let t0 = Instant::now();
    for i in 0..NUM_STEPS {
        let step_t = Instant::now();
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Timestep is raw sigma value [0, 1] — NOT multiplied by 1000
        let t_vec = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Conditional + unconditional predictions (context is pre-cached)
        let pred_cond = model.forward_with_context(&x, &t_vec, &context_cond)?;
        let pred_uncond = model.forward_with_context(&x, &t_vec, &context_uncond)?;

        // CFG: pred = uncond + scale * (cond - uncond)
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;

        // Euler step: x = x + dt * pred (rectified flow velocity)
        let dt = sigma_next - sigma;
        x = x.add(&pred.mul_scalar(dt)?)?;

        let step_ms = step_t.elapsed().as_millis();
        if i == 0 || i % 10 == 9 || i == NUM_STEPS - 1 {
            // Debug: check prediction magnitude on step 0
            if i == 0 {
                let pred_f32 = pred.to_dtype(DType::F32)?;
                let data = pred_f32.to_vec()?;
                let mean_abs: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
                println!("  Step {}/{}: sigma={:.4} ({:.0}ms) pred_mean_abs={:.4}",
                    i + 1, NUM_STEPS, sigma, step_ms, mean_abs);
            } else {
                println!("  Step {}/{}: sigma={:.4} ({:.0}ms)", i + 1, NUM_STEPS, sigma, step_ms);
            }
        }
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoise: {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: Save latent for Python VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Save latent ---");
    let t0 = Instant::now();
    drop(model);

    // Anima latent is [B, T=1, H, W, 16] → need [B, 16, T, H, W] for VAE
    let latent = x.permute(&[0, 4, 1, 2, 3])?;
    println!("  Latent for VAE: {:?}", latent.dims());

    // Save latent as safetensors for Python VAE decode
    let latent_path = OUTPUT_PATH.replace(".png", "_latent.safetensors");
    {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert("latent".to_string(), latent.clone());
        flame_core::serialization::save_file(&tensors, std::path::Path::new(&latent_path))?;
        println!("  Saved latent to {}", latent_path);
    }

    // Decode with Python: python decode_anima_latent.py
    println!("  Run: python decode_anima_latent.py {}", latent_path);
    println!("  Stage 4 in {:.1}s", t0.elapsed().as_secs_f32());

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
