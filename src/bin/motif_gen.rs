//! Motif-Video Stage 2: DiT denoise → save latents.
//!
//! Loads cached T5Gemma2 embeddings + Motif DiT, runs APG denoise loop,
//! saves final video latents to safetensors (decode in Stage 3).

use std::collections::HashMap;
use std::time::Instant;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::motif_video_dit::MotifDit;
use inference_flame::sampling::motif_sampling;

const BASE_DIR: &str = "/home/alex/.serenity/models/checkpoints/motif-video-2b";

// Defaults for a small test run (480x832 × 25 frames, 4× temporal → 13 latent frames).
// Full-res is 736×1280 × 121 frames in the Motif README.
const DEFAULT_HEIGHT: usize = 480;
const DEFAULT_WIDTH: usize = 832;
const DEFAULT_NUM_FRAMES: usize = 25;
const DEFAULT_STEPS: usize = 50;
const DEFAULT_GUIDANCE: f32 = 8.0;
const DEFAULT_SEED: u64 = 42;

// VAE: Wan2.1 (16-channel latents, 8× spatial, 4× temporal)
const VAE_SCALE_SPATIAL: usize = 8;
const VAE_SCALE_TEMPORAL: usize = 4;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned()
        .unwrap_or_else(|| "/home/alex/serenity/output/motif_embeds.safetensors".into());
    let out_path = args.get(2).cloned()
        .unwrap_or_else(|| "/home/alex/serenity/output/motif_latents.safetensors".into());

    let height: usize = env_usize("MOTIF_HEIGHT", DEFAULT_HEIGHT);
    let width: usize = env_usize("MOTIF_WIDTH", DEFAULT_WIDTH);
    let num_frames: usize = env_usize("MOTIF_FRAMES", DEFAULT_NUM_FRAMES);
    let num_steps: usize = env_usize("MOTIF_STEPS", DEFAULT_STEPS);
    let guidance_scale: f32 = env_f32("MOTIF_GUIDANCE", DEFAULT_GUIDANCE);
    let seed: u64 = env_u64("MOTIF_SEED", DEFAULT_SEED);

    println!("=== Motif-Video Stage 2 (DiT denoise) ===");
    println!("  Embeds: {}", embeds_path);
    println!("  Output: {}", out_path);
    println!("  Size: {}x{}, frames={}, steps={}, cfg={}, seed={}",
             width, height, num_frames, num_steps, guidance_scale, seed);

    // --- Load cached embeddings ---
    println!("--- Loading embeddings ---");
    let emb = flame_core::serialization::load_file(std::path::Path::new(&embeds_path), &device)?;
    let cond = emb.get("cond").ok_or_else(|| anyhow::anyhow!("Missing cond"))?.clone();
    let uncond = emb.get("uncond").ok_or_else(|| anyhow::anyhow!("Missing uncond"))?.clone();
    let cond = if cond.dtype() == DType::BF16 { cond } else { cond.to_dtype(DType::BF16)? };
    let uncond = if uncond.dtype() == DType::BF16 { uncond } else { uncond.to_dtype(DType::BF16)? };
    println!("  cond: {:?}, uncond: {:?}", cond.shape().dims(), uncond.shape().dims());

    // --- Load DiT ---
    println!("--- Loading Motif DiT ---");
    let t0 = Instant::now();
    let shard = format!("{}/transformer/diffusion_pytorch_model.safetensors", BASE_DIR);
    let mut dit = MotifDit::load(&[&shard], &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Initialize noise ---
    // Latent shape: [B=1, 16, num_latent_frames, H/8, W/8]
    // num_latent_frames = (num_frames - 1) / 4 + 1 for Wan 2.1 VAE
    let latent_frames = (num_frames - 1) / VAE_SCALE_TEMPORAL + 1;
    let lat_h = height / VAE_SCALE_SPATIAL;
    let lat_w = width / VAE_SCALE_SPATIAL;
    let numel = 16 * latent_frames * lat_h * lat_w;

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut noise_f32 = Vec::with_capacity(numel);
    for _ in 0..numel {
        // Box-Muller
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen();
        noise_f32.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let noise = Tensor::from_vec(
        noise_f32,
        Shape::from_dims(&[1, 16, latent_frames, lat_h, lat_w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    // --- Denoise loop ---
    let sigmas = motif_sampling::get_schedule(num_steps);
    println!("  sigmas[0]={:.4}  sigmas[{}]={:.4}  sigmas[-1]={:.4}",
             sigmas[0], num_steps - 1, sigmas[num_steps - 1], sigmas[num_steps]);

    let apg_cfg = motif_sampling::ApgConfig {
        guidance_scale,
        ..Default::default()
    };
    // Momentum buffer persists across all 50 steps — its `running` is the
    // decaying sum of past `diff`s (matches diffusers `MomentumBuffer` with
    // momentum=0.1).
    let mut momentum = motif_sampling::MomentumBuffer::new(0.1);

    let mut latents = noise;
    let t_steps = Instant::now();
    for i in 0..num_steps {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];
        let dt = sigma_next - sigma;
        // CRITICAL: Motif's DiT expects timestep in [0, num_train_timesteps=1000]
        // range (see `pipeline_motif_video.py:1183` — `t` is from
        // `scheduler.timesteps`, which for FlowMatchEulerDiscreteScheduler is
        // `sigmas * num_train_timesteps`). The previous version passed raw
        // `sigma` (in [0, 1]) directly, 1000× underscaled — the time embedding
        // saw all steps as near-t=0, velocity predictions were near-identity,
        // no denoising happened, VAE decoded pure noise.
        let timestep_value = sigma * 1000.0;
        let timestep = Tensor::from_vec(
            vec![timestep_value],
            Shape::from_dims(&[1]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;

        // For T2V, conditioning channels (16 + 1 mask) are zeros.
        // Concat: [latents, latent_condition=zeros, latent_mask=zeros]
        let latent_cond_shape = [1, 16, latent_frames, lat_h, lat_w];
        let latent_mask_shape = [1, 1, latent_frames, lat_h, lat_w];
        let lat_cond = Tensor::zeros_dtype(
            Shape::from_dims(&latent_cond_shape),
            DType::BF16,
            device.clone(),
        )?;
        let lat_mask = Tensor::zeros_dtype(
            Shape::from_dims(&latent_mask_shape),
            DType::BF16,
            device.clone(),
        )?;
        let full = Tensor::cat(&[&latents, &lat_cond, &lat_mask], 1)?;

        // Cond forward
        let pred_cond = dit.forward(&full, &cond, &timestep, None)?;
        // Uncond forward
        let pred_uncond = dit.forward(&full, &uncond, &timestep, None)?;

        // APG guidance — with MOTIF_PLAIN_CFG=1, bypass APG's projected component
        // and use standard CFG instead: `pred_uncond + scale * diff`. Diagnostic.
        let noise_pred = if std::env::var_os("MOTIF_PLAIN_CFG").is_some() {
            let diff = pred_cond.sub(&pred_uncond)?;
            let scaled = diff.mul_scalar(guidance_scale)?;
            pred_uncond.add(&scaled)?
        } else {
            motif_sampling::apg_guidance(
                &pred_cond, &pred_uncond, &apg_cfg, Some(&mut momentum),
            )?
        };

        // Euler step
        latents = motif_sampling::euler_step(&latents, &noise_pred, dt)?;

        if i == 0 || i == num_steps - 1 || (i + 1) % 10 == 0 {
            println!("  step {}/{}  sigma={:.4}  elapsed={:.1}s",
                     i + 1, num_steps, sigma, t_steps.elapsed().as_secs_f32());
        }
    }
    println!("Denoise complete in {:.1}s", t_steps.elapsed().as_secs_f32());

    drop(dit);

    // --- Save latents ---
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("latents".into(), latents);
    tensors.insert("height".into(), Tensor::from_vec(
        vec![height as f32], Shape::from_dims(&[1]), device.clone())?);
    tensors.insert("width".into(), Tensor::from_vec(
        vec![width as f32], Shape::from_dims(&[1]), device.clone())?);
    tensors.insert("num_frames".into(), Tensor::from_vec(
        vec![num_frames as f32], Shape::from_dims(&[1]), device.clone())?);
    flame_core::serialization::save_file(&tensors, &out_path)?;
    println!("Saved latents to {} ({:.1}s total)", out_path, t_total.elapsed().as_secs_f32());
    Ok(())
}
