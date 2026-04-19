//! Wan2.1-VACE-14B — Stage 2 (VACE DiT denoise → save latents).
//!
//! Single model (no dual expert). VACE conditioning network produces hints
//! that guide the base DiT's generation.

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::wan_vace_dit::WanVaceDit;

const VACE_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/wan2.1_vace_14b_fp16.safetensors";

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SHIFT: f32 = 5.0;
const VAE_STRIDE: [usize; 3] = [4, 8, 8];
const Z_DIM: usize = 16;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_vace_embeds.safetensors".to_string()
    });
    let out_latents = args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_vace_latents.safetensors".to_string()
    });

    let num_steps: usize = env_usize("WAN_STEPS", 20);
    let cfg_scale: f32 = env_f32("WAN_CFG", 5.0);
    let seed: u64 = env_u64("WAN_SEED", 42);
    let shift: f32 = env_f32("WAN_SHIFT", SHIFT);
    let context_scale: f32 = env_f32("WAN_VACE_SCALE", 1.0);

    println!("=== Wan2.1-VACE-14B — Stage 2 (Rust VACE DiT denoise) ===");
    println!("Embeddings: {}", embeds_path);
    println!("Output:     {}", out_latents);
    println!();

    // Load embeddings
    println!("--- Loading cached embeddings ---");
    let t0 = Instant::now();
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&embeds_path), &device,
    )?;
    let cond = ensure_bf16(tensors.get("cond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'cond'"))?.clone())?;
    let uncond = ensure_bf16(tensors.get("uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'uncond'"))?.clone())?;
    let vace_ctx = ensure_bf16(tensors.get("vace_context")
        .ok_or_else(|| anyhow::anyhow!("Missing 'vace_context'"))?.clone())?;
    let target_h = tensors.get("target_h")
        .ok_or_else(|| anyhow::anyhow!("Missing target_h"))?
        .to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize;
    let target_w = tensors.get("target_w")
        .ok_or_else(|| anyhow::anyhow!("Missing target_w"))?
        .to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize;
    let frame_num = tensors.get("frame_num")
        .ok_or_else(|| anyhow::anyhow!("Missing frame_num"))?
        .to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize;
    drop(tensors);

    println!("  cond:         {:?}", cond.shape().dims());
    println!("  uncond:       {:?}", uncond.shape().dims());
    println!("  vace_context: {:?}", vace_ctx.shape().dims());
    println!("  target: {}x{}, frames={}", target_w, target_h, frame_num);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Geometry
    let lat_f = (frame_num - 1) / VAE_STRIDE[0] + 1;
    let lat_h = target_h / VAE_STRIDE[1];
    let lat_w = target_w / VAE_STRIDE[2];
    let patch_f = lat_f;
    let patch_h = lat_h / 2;
    let patch_w = lat_w / 2;
    let seq_len = patch_f * patch_h * patch_w;

    println!("\nSize:   {}x{}, frames={}, steps={}", target_w, target_h, frame_num, num_steps);
    println!("CFG:    {}, shift={}, vace_scale={}", cfg_scale, shift, context_scale);
    println!("Latent: [{}, {}, {}, {}], seq_len={}", Z_DIM, lat_f, lat_h, lat_w, seq_len);
    println!();

    // Generate noise
    let numel = Z_DIM * lat_f * lat_h * lat_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut v = Vec::with_capacity(numel);
        for _ in 0..numel / 2 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(r * theta.cos());
            v.push(r * theta.sin());
        }
        if numel % 2 == 1 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            v.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }
        v
    };
    let mut latent = Tensor::from_vec(
        noise_data, Shape::from_dims(&[Z_DIM, lat_f, lat_h, lat_w]), device.clone(),
    )?.to_dtype(DType::BF16)?;

    // Sigma schedule
    let sigma_max: f32 = 1.0 - 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let sigma_min: f32 = 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / num_steps as f32;
            sigma_max + t * (sigma_min - sigma_max)
        })
        .collect();
    for s in sigmas.iter_mut() {
        *s = shift * *s / (1.0 + (shift - 1.0) * *s);
    }
    sigmas.push(0.0);
    let timesteps: Vec<f32> = sigmas.iter().map(|s| s * NUM_TRAIN_TIMESTEPS as f32).collect();

    println!("  sigmas[0]={:.4}  sigmas[-2]={:.4}", sigmas[0], sigmas[num_steps - 1]);
    println!();

    // Load VACE model
    println!("--- Loading VACE model ---");
    let t0 = Instant::now();
    let mut dit = WanVaceDit::load(VACE_CHECKPOINT, &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // Denoise loop
    println!("--- Denoise ({} steps, VACE Euler) ---", num_steps);
    let t_denoise = Instant::now();

    for step in 0..num_steps {
        let ts = timesteps[step];
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma;

        let next_latent = {
            let cond_pred = dit.forward(&latent, &vace_ctx, ts, &cond, seq_len, context_scale)?;
            let uncond_pred = dit.forward(&latent, &vace_ctx, ts, &uncond, seq_len, context_scale)?;

            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(cfg_scale)?;
            let noise_pred = uncond_pred.add(&scaled)?;

            let step_delta = noise_pred.mul_scalar(dt)?;
            latent.add(&step_delta)?
        };
        latent = next_latent;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps {
            println!(
                "  step {}/{}  ts={:.1}  sigma={:.4}  ({:.1}s elapsed)",
                step + 1, num_steps, ts, sigma,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }

    println!("  Denoised in {:.1}s ({:.2}s/step)",
        t_denoise.elapsed().as_secs_f32(),
        t_denoise.elapsed().as_secs_f32() / num_steps as f32);

    drop(dit);

    // Save
    if let Some(parent) = std::path::Path::new(&out_latents).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut output = HashMap::new();
    output.insert("latent".to_string(), latent);
    flame_core::serialization::save_file(&output, &out_latents)?;

    println!("\n============================================================");
    println!("LATENTS SAVED: {}", out_latents);
    println!("Total time:    {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    println!("\nNext: python scripts/wan22_decode.py {} <output.mp4>", out_latents);

    Ok(())
}

fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    if t.dtype() == DType::BF16 { Ok(t) } else { Ok(t.to_dtype(DType::BF16)?) }
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
