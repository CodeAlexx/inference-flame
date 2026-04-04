//! Anima (Cosmos Predict2) image generation — pure Rust.
//!
//! Usage: anima_infer [path_to_cached_embeddings.safetensors]
//!
//! Cached embeddings contain Qwen3 0.6B hidden states + token IDs.
//! Model is 3.9GB — loads entirely to GPU (no block offloading).

use inference_flame::models::anima::{Anima, load_all_weights};
use inference_flame::vae::Wan21VaeDecoder;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors";
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/anima_rust.png";

// Cosmos RFLOW sampling with shift=3.0
const NUM_STEPS: usize = 20;
const CFG_SCALE: f32 = 7.0;
const SHIFT: f32 = 3.0;
const SEED: u64 = 42;
const WIDTH: usize = 512;
const HEIGHT: usize = 512;

/// Cosmos RFLOW sigma schedule with time shifting.
fn build_cosmos_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
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
    println!("Anima — Pure Rust Inference (all-on-GPU)");
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
    // Stage 2: Load Anima model — ALL weights on GPU (3.9GB fits easily)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Anima model (all-on-GPU) ---");
    let t0 = Instant::now();
    let all_weights = load_all_weights(MODEL_PATH, &device)?;
    println!("  {} weight tensors loaded", all_weights.len());
    let mut model = Anima::new_all_on_gpu(MODEL_PATH.to_string(), all_weights, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2b: Pre-compute LLM adapter context (run once, reuse every step)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2b: Encode text context (cached) ---");
    let t0 = Instant::now();
    let context_cond = model.encode_context(&token_ids, &llm_hidden)?;
    let context_uncond = model.encode_context(&neg_token_ids, &neg_llm_hidden)?;
    println!("  Context: {:?}", context_cond.dims());
    println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

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

    let timesteps = build_cosmos_schedule(NUM_STEPS, SHIFT);

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
        let t_curr = timesteps[i];
        let t_prev = timesteps[i + 1];

        let t_vec = Tensor::from_f32_to_bf16(
            vec![t_curr], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Cosmos RFLOW: s = sigma / (sigma + 1)
        let s = t_curr / (t_curr + 1.0);

        // Model input: x * (1 - s)
        let x_in = x.mul_scalar(1.0 - s)?;

        // Conditional + unconditional predictions (context is pre-cached)
        let pred_cond = model.forward_with_context(&x_in, &t_vec, &context_cond)?;
        let pred_uncond = model.forward_with_context(&x_in, &t_vec, &context_uncond)?;

        // CFG: pred = uncond + scale * (cond - uncond)
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;

        // Cosmos RFLOW denoised: x_in * (1-s) - pred * s
        let denoised = x_in.mul_scalar(1.0 - s)?.sub(&pred.mul_scalar(s)?)?;

        // Euler step: x = x + dt/t_curr * (x - denoised)
        let dt = t_prev - t_curr;
        let d = x.sub(&denoised)?.mul_scalar(1.0 / t_curr)?;
        x = x.add(&d.mul_scalar(dt)?)?;

        let step_ms = step_t.elapsed().as_millis();
        if i == 0 || i % 5 == 4 || i == NUM_STEPS - 1 {
            println!("  Step {}/{}: t={:.4} ({:.0}ms)", i + 1, NUM_STEPS, t_curr, step_ms);
        }
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoise: {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode (Wan21 3D causal VAE)
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode (Wan21 VAE) ---");
    let t0 = Instant::now();
    drop(model); // Free DiT VRAM

    // Anima latent is [B, T=1, H, W, 16] → need [B, 16, T, H, W] for Wan VAE
    let latent = x.permute(&[0, 4, 1, 2, 3])?; // [1, 16, 1, H/8, W/8]
    println!("  Latent for VAE: {:?}", latent.dims());

    let vae = Wan21VaeDecoder::load(VAE_PATH, &device)?;
    let rgb = vae.decode(&latent)?;
    println!("  Decoded: {:?}", rgb.dims());
    println!("  VAE in {:.1}s", t0.elapsed().as_secs_f32());

    // Extract first frame: [B, 3, T_out, H, W]
    let rgb_dims = rgb.dims().to_vec();
    let (out_t, out_h, out_w) = (rgb_dims[2], rgb_dims[3], rgb_dims[4]);

    // Take first temporal frame
    let frame = if out_t > 1 {
        rgb.narrow(2, 0, 1)?.reshape(&[1, 3, out_h, out_w])?
    } else {
        rgb.reshape(&[1, 3, out_h, out_w])?
    };

    // Convert to [0, 255] image: output is clamped [-1, 1]
    let frame_f32 = frame.to_dtype(DType::F32)?;
    let data = frame_f32.to_vec()?;

    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for px in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + px;
                let val = ((data[idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                pixels[(y * out_w + px) * 3 + c] = val;
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
