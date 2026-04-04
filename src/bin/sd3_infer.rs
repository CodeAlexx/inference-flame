//! SD3.5 Large image generation — pure Rust.
//!
//! Usage: sd3_infer [path_to_cached_embeddings.safetensors]
//!
//! Cached embeddings must contain:
//!   "encoder_hidden_states": [B, 154, 4096] — CLIP-L(77) + CLIP-G(77) + T5-XXL(77→pad)
//!   "pooled_projections": [B, 2048] — CLIP-L(768) + CLIP-G(1280)
//!   (uncond variants for CFG)

use inference_flame::models::sd3_mmdit::{SD3MMDiT, load_sd3_resident};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;
use inference_flame::sampling::klein_sampling::euler_denoise;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sd3.5_large.safetensors";
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sd3_embeddings.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sd3_rust.png";

// SD3.5 sampling: flow matching with shift=3.0
const NUM_STEPS: usize = 28;
const CFG_SCALE: f32 = 4.5;
const SHIFT: f32 = 3.0;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

fn build_sd3_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    // SD3 uses shifted linear schedule like Flux
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
    println!("SD3.5 Large — Pure Rust Inference");
    println!("  {}x{}, {} steps, CFG {}, shift {}", WIDTH, HEIGHT, NUM_STEPS, CFG_SCALE, SHIFT);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(std::path::Path::new(&emb_path), &device)?;
    let context = emb.get("encoder_hidden_states")
        .ok_or_else(|| anyhow::anyhow!("Missing 'encoder_hidden_states'"))?.clone();
    let pooled = emb.get("pooled_projections")
        .ok_or_else(|| anyhow::anyhow!("Missing 'pooled_projections'"))?.clone();
    let context_uncond = emb.get("encoder_hidden_states_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'encoder_hidden_states_uncond'"))?.clone();
    let pooled_uncond = emb.get("pooled_projections_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'pooled_projections_uncond'"))?.clone();
    drop(emb);
    println!("  context: {:?}, pooled: {:?}", context.dims(), pooled.dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load SD3.5 MMDiT
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load SD3.5 Large ---");
    let t0 = Instant::now();
    let resident = load_sd3_resident(MODEL_PATH, &device)?;
    let mut model = SD3MMDiT::new(MODEL_PATH.to_string(), resident, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Denoise (flow matching, Euler)
    // ------------------------------------------------------------------
    let latent_h = HEIGHT / 8; // SD3 uses /8 VAE
    let latent_w = WIDTH / 8;
    let numel = 16 * latent_h * latent_w; // 16 latent channels

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

    let noise = Tensor::from_f32_to_bf16(
        noise_data, Shape::from_dims(&[1, 16, latent_h, latent_w]), device.clone(),
    )?;

    let timesteps = build_sd3_schedule(NUM_STEPS, SHIFT);

    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, CFG_SCALE);
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr], Shape::from_dims(&[1]), device.clone(),
            )?;

            let pred_cond = model.forward(x, &t_vec, &context, &pooled)?;
            let pred_uncond = model.forward(x, &t_vec, &context_uncond, &pooled_uncond)?;

            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)
        },
        noise,
        &timesteps,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);

    // SD3 VAE: 16ch latent, scale=1.5305, shift=0.0609
    let vae = LdmVAEDecoder::from_safetensors(MODEL_PATH, 16, 1.5305, 0.0609, &device)?;
    let rgb = vae.decode(&denoised)?;
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
