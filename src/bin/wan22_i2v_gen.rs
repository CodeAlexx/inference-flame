//! Wan2.2-I2V-A14B — Stage 2 (dual-expert DiT denoise for image-to-video).
//!
//! Same architecture as T2V but:
//! - patch_embedding accepts 36 channels (16 noise + 4 mask + 16 VAE image = 36)
//! - `y` (image conditioning) is concatenated with noise before patchify
//! - Different config: boundary=0.900, shift=5.0, cfg=(3.5, 3.5)

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::wan22_dit::Wan22Dit;

const HIGH_NOISE_PATH: &str = "/home/alex/.serenity/models/checkpoints/wan2.2_i2v_high_noise_14b_fp16.safetensors";
const LOW_NOISE_PATH: &str = "/home/alex/.serenity/models/checkpoints/wan2.2_i2v_low_noise_14b_fp16.safetensors";

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BOUNDARY: f32 = 0.900;        // I2V boundary (higher than T2V's 0.875)
const SHIFT: f32 = 5.0;             // I2V shift (lower than T2V's 12.0)
const VAE_STRIDE: [usize; 3] = [4, 8, 8];
const Z_DIM: usize = 16;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_i2v_embeds.safetensors".to_string()
    });
    let out_latents = args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_i2v_latents.safetensors".to_string()
    });

    let num_steps: usize = env_usize("WAN_STEPS", 20);
    let cfg_low: f32 = env_f32("WAN_CFG_LOW", 3.5);
    let cfg_high: f32 = env_f32("WAN_CFG_HIGH", 3.5);
    let seed: u64 = env_u64("WAN_SEED", 42);
    let shift: f32 = env_f32("WAN_SHIFT", SHIFT);

    println!("=== Wan2.2-I2V-A14B — Stage 2 (Rust dual-expert DiT denoise) ===");
    println!("Embeddings: {}", embeds_path);
    println!("Output lat: {}", out_latents);
    println!();

    // ------------------------------------------------------------------
    // Load cached embeddings + image conditioning
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
    let y = ensure_bf16(tensors.get("y")
        .ok_or_else(|| anyhow::anyhow!("Missing 'y' (image conditioning)"))?.clone())?;

    // Read metadata
    let target_h = {
        let t = tensors.get("target_h").ok_or_else(|| anyhow::anyhow!("Missing target_h"))?;
        t.to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize
    };
    let target_w = {
        let t = tensors.get("target_w").ok_or_else(|| anyhow::anyhow!("Missing target_w"))?;
        t.to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize
    };
    let frame_num = {
        let t = tensors.get("frame_num").ok_or_else(|| anyhow::anyhow!("Missing frame_num"))?;
        t.to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as usize
    };
    drop(tensors);

    println!("  cond:   {:?}", cond.shape().dims());
    println!("  uncond: {:?}", uncond.shape().dims());
    println!("  y:      {:?}", y.shape().dims());
    println!("  target: {}x{}, frames={}", target_w, target_h, frame_num);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Compute latent geometry
    // ------------------------------------------------------------------
    let lat_f = (frame_num - 1) / VAE_STRIDE[0] + 1;
    let lat_h = target_h / VAE_STRIDE[1];
    let lat_w = target_w / VAE_STRIDE[2];
    let patch_f = lat_f;
    let patch_h = lat_h / 2;
    let patch_w = lat_w / 2;
    let seq_len = patch_f * patch_h * patch_w;

    println!("Size:       {}x{}, frames={}, steps={}", target_w, target_h, frame_num, num_steps);
    println!("CFG:        low={}, high={}, shift={}", cfg_low, cfg_high, shift);
    println!("Seed:       {}", seed);
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
    let noise = Tensor::from_vec(
        noise_data,
        Shape::from_dims(&[Z_DIM, lat_f, lat_h, lat_w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    // ------------------------------------------------------------------
    // Build sigma schedule
    // ------------------------------------------------------------------
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
    let boundary_ts = BOUNDARY * NUM_TRAIN_TIMESTEPS as f32;

    println!("  sigmas[0]={:.4}  sigmas[-2]={:.4}", sigmas[0], sigmas[num_steps - 1]);
    println!("  timesteps[0]={:.1}  boundary={:.1}", timesteps[0], boundary_ts);
    println!();

    // ------------------------------------------------------------------
    // Load DiT
    // ------------------------------------------------------------------
    let first_is_high = timesteps[0] >= boundary_ts;
    println!("--- Loading {} model first ---",
        if first_is_high { "high_noise" } else { "low_noise" });
    let t0 = Instant::now();
    let mut current_is_high = first_is_high;
    let mut dit = if first_is_high {
        Wan22Dit::load(HIGH_NOISE_PATH, &device)?
    } else {
        Wan22Dit::load(LOW_NOISE_PATH, &device)?
    };
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Euler denoise loop (I2V: concat y with noise before each forward)
    // ------------------------------------------------------------------
    println!("--- Denoise ({} steps, I2V Euler flow-matching) ---", num_steps);
    let t_denoise = Instant::now();
    let mut latent = noise;

    for step in 0..num_steps {
        let ts = timesteps[step];
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma;

        // Switch models if needed
        let need_high = ts >= boundary_ts;
        if need_high != current_is_high {
            println!("  [SWITCH] {} → {} at ts={:.1}",
                if current_is_high { "high" } else { "low" },
                if need_high { "high" } else { "low" }, ts);
            drop(dit);
            let t_switch = Instant::now();
            dit = if need_high {
                Wan22Dit::load(HIGH_NOISE_PATH, &device)?
            } else {
                Wan22Dit::load(LOW_NOISE_PATH, &device)?
            };
            current_is_high = need_high;
            println!("  [SWITCH] Loaded in {:.1}s", t_switch.elapsed().as_secs_f32());
        }

        let guide_scale = if need_high { cfg_high } else { cfg_low };

        // I2V: concatenate [noise, y] along channel dim before forward
        // noise: [16, F_lat, H_lat, W_lat], y: [20, F_lat, H_lat, W_lat]
        // → x_input: [36, F_lat, H_lat, W_lat]
        let next_latent = {
            let x_input = Tensor::cat(&[&latent, &y], 0)?;  // cat along dim 0 (channels)

            let cond_pred = dit.forward_i2v(&x_input, ts, &cond, seq_len)?;
            let x_input_uncond = Tensor::cat(&[&latent, &y], 0)?;
            let uncond_pred = dit.forward_i2v(&x_input_uncond, ts, &uncond, seq_len)?;

            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(guide_scale)?;
            let noise_pred = uncond_pred.add(&scaled)?;

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
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt_denoise, dt_denoise / num_steps as f32);
    println!();

    drop(dit);
    drop(cond);
    drop(uncond);
    drop(y);

    // ------------------------------------------------------------------
    // Save latent (just the 16-channel noise prediction, not the 36-channel concat)
    // ------------------------------------------------------------------
    println!("--- Saving latent ---");
    if let Some(parent) = std::path::Path::new(&out_latents).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut output = HashMap::new();
    output.insert("latent".to_string(), latent);
    flame_core::serialization::save_file(&output, &out_latents)?;

    println!("============================================================");
    println!("LATENTS SAVED: {}", out_latents);
    println!("Total time:    {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    println!();
    println!("Next: python scripts/wan22_decode.py {} <output.mp4>", out_latents);

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
