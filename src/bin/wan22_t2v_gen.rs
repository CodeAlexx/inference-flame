//! Wan2.2-T2V-A14B — Stage 2 (dual-expert DiT denoise → save latents).
//!
//! Loads cached text embeddings from Stage 1, loads TWO DiT models via
//! BlockOffloader (high_noise + low_noise), runs a flow-matching Euler denoise
//! loop with sigma-shifted timesteps, and saves the final latent for Stage 3
//! VAE decode.
//!
//! ## Dual expert design
//! Wan2.2-A14B uses TWO separate 14B DiT checkpoints:
//!   - `high_noise_model`: used when timestep >= boundary (875)
//!   - `low_noise_model`: used when timestep < boundary (875)
//! Each has its own CFG scale: 4.0 (high) / 3.0 (low).
//!
//! ## Pipeline split (OOM-safe)
//!   Stage 1 (Python): UMT5-XXL text encoder only
//!   Stage 2 (Rust):   Dual DiT via BlockOffloader (one active at a time)
//!   Stage 3 (Python): Wan2.1 VAE decode → MP4
//!
//! ## Scheduler — Euler with flow-matching sigma shift
//! sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
//! where shift = 12.0 (from config).

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::wan22_dit::Wan22Dit;

// Override with WAN22_HIGH_NOISE_PATH / WAN22_LOW_NOISE_PATH env vars. Pair with
// `BLOCKOFF_FP8_PINNED=1` to keep the FP8-scaled variants as FP8 in pinned RAM
// (halves host RAM, ~28 GB → ~14 GB per expert — the only way both fit on a
// 62 GB machine).
const HIGH_NOISE_PATH: &str = "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_high_noise_14b_fp16.safetensors";
const LOW_NOISE_PATH: &str = "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors";

// Wan2.2 T2V-A14B config
const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BOUNDARY: f32 = 0.875;       // boundary fraction
const SHIFT: f32 = 12.0;           // sigma shift
const VAE_STRIDE: [usize; 3] = [4, 8, 8];
const Z_DIM: usize = 16;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_embeds.safetensors".to_string()
    });
    let out_latents = args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_latents.safetensors".to_string()
    });

    // Knobs
    let width: usize = env_usize("WAN_WIDTH", 480);
    let height: usize = env_usize("WAN_HEIGHT", 272);
    let frame_num: usize = env_usize("WAN_FRAMES", 17);  // must be 4n+1
    let num_steps: usize = env_usize("WAN_STEPS", 30);
    let cfg_low: f32 = env_f32("WAN_CFG_LOW", 3.0);
    let cfg_high: f32 = env_f32("WAN_CFG_HIGH", 4.0);
    let seed: u64 = env_u64("WAN_SEED", 42);
    let shift: f32 = env_f32("WAN_SHIFT", SHIFT);

    println!("=== Wan2.2-T2V-A14B — Stage 2 (Rust dual-expert DiT denoise) ===");
    println!("Embeddings: {}", embeds_path);
    println!("Output lat: {}", out_latents);
    println!("Size:       {}x{}, frames={}, steps={}", width, height, frame_num, num_steps);
    println!("CFG:        low={}, high={}, shift={}", cfg_low, cfg_high, shift);
    println!("Seed:       {}", seed);
    println!();

    // ------------------------------------------------------------------
    // Load cached embeddings
    // ------------------------------------------------------------------
    println!("--- Loading cached embeddings ---");
    let t0 = Instant::now();
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&embeds_path),
        &device,
    )?;
    let cond = ensure_bf16(tensors.get("cond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'cond'"))?.clone())?;
    let uncond = ensure_bf16(tensors.get("uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'uncond'"))?.clone())?;
    drop(tensors);
    println!("  cond:   {:?}", cond.shape().dims());
    println!("  uncond: {:?}", uncond.shape().dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Compute latent geometry
    // ------------------------------------------------------------------
    let lat_f = (frame_num - 1) / VAE_STRIDE[0] + 1;
    let lat_h = height / VAE_STRIDE[1];
    let lat_w = width / VAE_STRIDE[2];
    // After patchify with patch_size=(1,2,2):
    let patch_f = lat_f;       // /1
    let patch_h = lat_h / 2;   // /2
    let patch_w = lat_w / 2;   // /2
    let seq_len = patch_f * patch_h * patch_w;
    println!("  Latent: [{}, {}, {}, {}]", Z_DIM, lat_f, lat_h, lat_w);
    println!("  Grid:   ({}, {}, {}), seq_len={}", patch_f, patch_h, patch_w, seq_len);
    println!();

    // ------------------------------------------------------------------
    // Generate noise
    // ------------------------------------------------------------------
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
        noise_data,
        Shape::from_dims(&[Z_DIM, lat_f, lat_h, lat_w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;
    println!("  Noise: [{}, {}, {}, {}]", Z_DIM, lat_f, lat_h, lat_w);

    // ------------------------------------------------------------------
    // Build sigma schedule (Euler flow matching with shift)
    // ------------------------------------------------------------------
    // sigmas = linspace(sigma_max, sigma_min, num_steps+1)[:-1]
    // sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    let sigma_max: f32 = 1.0 - 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let sigma_min: f32 = 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / num_steps as f32;
            sigma_max + t * (sigma_min - sigma_max)
        })
        .collect();
    // Apply shift
    for s in sigmas.iter_mut() {
        *s = shift * *s / (1.0 + (shift - 1.0) * *s);
    }
    // Append terminal
    sigmas.push(0.0);
    // Convert to timesteps (sigma * num_train_timesteps)
    let timesteps: Vec<f32> = sigmas.iter().map(|s| s * NUM_TRAIN_TIMESTEPS as f32).collect();

    let boundary_ts = BOUNDARY * NUM_TRAIN_TIMESTEPS as f32;
    println!("  sigmas[0]={:.4}  sigmas[1]={:.4}  sigmas[-2]={:.4}  sigmas[-1]={:.4}",
        sigmas[0], sigmas[1], sigmas[num_steps - 1], sigmas[num_steps]);
    println!("  timesteps[0]={:.1}  boundary={:.1}", timesteps[0], boundary_ts);
    println!();

    // ------------------------------------------------------------------
    // Load DiT models (one at a time via BlockOffloader)
    // ------------------------------------------------------------------
    // Strategy: load the model needed for the first timestep. When we cross
    // the boundary, drop and reload the other model.
    let first_ts = timesteps[0];
    let first_is_high = first_ts >= boundary_ts;

    println!("--- Loading {} model first ---",
        if first_is_high { "high_noise" } else { "low_noise" });
    let t0 = Instant::now();
    let mut current_is_high = first_is_high;
    let high_path = std::env::var("WAN22_HIGH_NOISE_PATH")
        .unwrap_or_else(|_| HIGH_NOISE_PATH.to_string());
    let low_path = std::env::var("WAN22_LOW_NOISE_PATH")
        .unwrap_or_else(|_| LOW_NOISE_PATH.to_string());
    println!("  high: {high_path}");
    println!("  low:  {low_path}");
    let mut dit = if first_is_high {
        Wan22Dit::load(&high_path, &device)?
    } else {
        Wan22Dit::load(&low_path, &device)?
    };
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Euler denoise loop
    // ------------------------------------------------------------------
    println!("--- Denoise ({} steps, Euler flow-matching) ---", num_steps);
    let t_denoise = Instant::now();

    for step in 0..num_steps {
        let ts = timesteps[step];
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma;

        // Check if we need to switch models
        let need_high = ts >= boundary_ts;
        if need_high != current_is_high {
            println!("  [SWITCH] {} → {} at ts={:.1}",
                if current_is_high { "high" } else { "low" },
                if need_high { "high" } else { "low" },
                ts);
            drop(dit);
            let t_switch = Instant::now();
            dit = if need_high {
                Wan22Dit::load(&high_path, &device)?
            } else {
                Wan22Dit::load(&low_path, &device)?
            };
            current_is_high = need_high;
            println!("  [SWITCH] Loaded in {:.1}s", t_switch.elapsed().as_secs_f32());
        }

        let guide_scale = if need_high { cfg_high } else { cfg_low };

        // Forward passes
        let next_latent = {
            let cond_pred = dit.forward(&latent, ts, &cond, seq_len)?;
            let uncond_pred = dit.forward(&latent, ts, &uncond, seq_len)?;

            // Diagnostic: print prediction stats for first step
            if step == 0 {
                let cp = cond_pred.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let up = uncond_pred.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let cp_mean: f32 = cp.iter().sum::<f32>() / cp.len() as f32;
                let cp_absm: f32 = cp.iter().map(|v| v.abs()).sum::<f32>() / cp.len() as f32;
                let up_mean: f32 = up.iter().sum::<f32>() / up.len() as f32;
                let up_absm: f32 = up.iter().map(|v| v.abs()).sum::<f32>() / up.len() as f32;
                println!("  [PARITY] cond_pred:   mean={:.6}, abs_mean={:.6}, len={}", cp_mean, cp_absm, cp.len());
                println!("  [PARITY] uncond_pred: mean={:.6}, abs_mean={:.6}, len={}", up_mean, up_absm, up.len());
            }

            // CFG: noise_pred = uncond + scale * (cond - uncond)
            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(guide_scale)?;
            let noise_pred = uncond_pred.add(&scaled)?;

            // Euler step on the latent (in [C, F, H, W] space)
            let step_delta = noise_pred.mul_scalar(dt)?;
            latent.add(&step_delta)?
        };
        latent = next_latent;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps {
            println!(
                "  step {}/{}  ts={:.1}  sigma={:.4}  cfg={:.1}  ({:.1}s elapsed)",
                step + 1, num_steps, ts, sigma, guide_scale,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }

    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step)",
        dt_denoise, dt_denoise / num_steps as f32,
    );
    println!();

    drop(dit);
    drop(cond);
    drop(uncond);
    println!("  DiT + cached embeddings evicted");

    // ------------------------------------------------------------------
    // Save latent
    // ------------------------------------------------------------------
    println!("--- Saving latent ---");
    if let Some(parent) = std::path::Path::new(&out_latents).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Save as [C, F, H, W] — the raw latent before VAE decode
    let mut output = HashMap::new();
    output.insert("latent".to_string(), latent);
    flame_core::serialization::save_file(&output, &out_latents)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("LATENTS SAVED: {}", out_latents);
    println!("Total time:    {:.1}s", dt_total);
    println!("============================================================");
    println!();
    println!("Next: python scripts/wan22_decode.py {} <output.mp4>", out_latents);

    let _ = device;
    Ok(())
}

fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        Ok(t.to_dtype(DType::BF16)?)
    }
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
