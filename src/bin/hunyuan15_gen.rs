//! HunyuanVideo 1.5 — Stage 2 (DiT denoise → save latents).
//!
//! 54 double-stream blocks via BlockOffloader, flow-matching Euler scheduler.
//! Text embeddings pre-refined in Python Stage 1.

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::hunyuan15_dit::Hunyuan15Dit;

const DIT_PATH: &str = "/home/alex/.serenity/models/checkpoints/hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model_bf16.safetensors";

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SHIFT: f32 = 7.0;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/hunyuan15_embeds.safetensors".to_string()
    });
    let out_latents = args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/hunyuan15_latents.safetensors".to_string()
    });

    let num_steps: usize = env_usize("HV_STEPS", 20);
    let cfg_scale: f32 = env_f32("HV_CFG", 6.0);
    let seed: u64 = env_u64("HV_SEED", 42);
    let shift: f32 = env_f32("HV_SHIFT", SHIFT);

    println!("=== HunyuanVideo 1.5 — Stage 2 (Rust DiT denoise) ===");

    // Load embeddings
    println!("--- Loading embeddings ---");
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&embeds_path), &device,
    )?;
    let txt_embeds = ensure_bf16(tensors.get("txt_embeds")
        .ok_or_else(|| anyhow::anyhow!("Missing txt_embeds"))?.clone())?;
    let txt_mask = tensors.get("txt_mask")
        .ok_or_else(|| anyhow::anyhow!("Missing txt_mask"))?.clone();
    let target_h = tensors.get("target_h").unwrap()
        .to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize;
    let target_w = tensors.get("target_w").unwrap()
        .to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize;
    let frame_num = tensors.get("frame_num").unwrap()
        .to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize;
    drop(tensors);

    println!("  txt_embeds: {:?}", txt_embeds.shape().dims());
    println!("  target: {}x{}, frames={}", target_w, target_h, frame_num);

    // Geometry — patch_size = [1,1,1], so no spatial reduction
    let c_in = 32usize;
    let tt = frame_num;
    let th = target_h;
    let tw = target_w;

    println!("  Latent: [{}, {}, {}, {}]", c_in, tt, th, tw);
    println!("  Steps={}, CFG={}, shift={}", num_steps, cfg_scale, shift);
    println!();

    // Generate noise [1, 32, T, H, W]
    let numel = c_in * tt * th * tw;
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
    let noise = Tensor::from_vec(
        noise_data, Shape::from_dims(&[1, c_in, tt, th, tw]), device.clone(),
    )?.to_dtype(DType::BF16)?;

    // Build condition input: [1, 65, T, H, W] = concat(noise, zeros_condition, ones_mask)
    // For T2V: condition = zeros, mask = ones (generate everything)
    let condition = Tensor::zeros_dtype(
        Shape::from_dims(&[1, c_in, tt, th, tw]), DType::BF16, device.clone(),
    )?;
    let mask = Tensor::from_vec(
        vec![1.0f32; tt * th * tw],
        Shape::from_dims(&[1, 1, tt, th, tw]),
        device.clone(),
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

    // Load DiT
    println!("\n--- Loading HunyuanVideo 1.5 DiT ---");
    let t0 = Instant::now();
    let mut dit = Hunyuan15Dit::load(DIT_PATH, &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Denoise loop
    println!("\n--- Denoise ({} steps) ---", num_steps);
    let t_denoise = Instant::now();
    let mut latent = noise;

    for step in 0..num_steps {
        let ts = timesteps[step];
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma;

        let next_latent = {
            // Build input: concat [latent, condition, mask] along channel dim
            let model_input = Tensor::cat(&[&latent, &condition, &mask], 1)?; // [1, 65, T, H, W]

            let cond_pred = dit.forward(&model_input, ts, &txt_embeds, &txt_mask, 6016.0)?;
            // For CFG we'd need uncond too — for now just use cond (no CFG)
            // TODO: add negative prompt support
            let noise_pred = cond_pred;

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
