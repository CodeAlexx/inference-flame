//! LTX-2.3 video generation — pure Rust, with CFG.
//!
//! Pipeline:
//! 1. Load cached Gemma text embeddings (4096-dim, pre-processed)
//! 2. Load LTX-2 transformer via FlameSwap
//! 3. Create noise + sigma schedule (dev model)
//! 4. Denoise with CFG: two forward passes per step (uncond + cond)
//! 5. Save denoised latents → decode with Python VAE

use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::sampling::ltx2_sampling::build_dev_sigma_schedule;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors";
const EMBEDDINGS_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors";
const OUTPUT_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_denoised_latents.safetensors";

const NUM_FRAMES: usize = 9;
const WIDTH: usize = 480;
const HEIGHT: usize = 288;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const GUIDANCE_SCALE: f32 = 4.0; // Official LTX-2 default
const NUM_STEPS: usize = 10;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("LTX-2.3 Video Generation — Pure Rust + CFG");
    println!("============================================================");
    println!("  {}×{}, {} frames, {} steps, cfg={}", WIDTH, HEIGHT, NUM_FRAMES, NUM_STEPS, GUIDANCE_SCALE);

    let device = global_cuda_device();

    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;
    let latent_h = HEIGHT / 32;
    let latent_w = WIDTH / 32;
    let num_tokens = latent_f * latent_h * latent_w;
    println!("  Latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, latent_h, latent_w, num_tokens);

    // Stage 1: Load embeddings (positive + create negative)
    println!("\n--- Stage 1: Load embeddings ---");
    let cached = flame_core::serialization::load_file(
        std::path::Path::new(EMBEDDINGS_PATH), &device,
    )?;
    let text_cond = cached.get("text_hidden")
        .ok_or_else(|| anyhow::anyhow!("Missing text_hidden"))?;
    println!("  Conditional: {:?} {:?}", text_cond.dims(), text_cond.dtype());

    // Negative embedding: proper empty-string encoding from official pipeline
    let neg_path = std::path::Path::new(
        "/home/alex/EriDiffusion/inference-flame/cached_ltx2_negative.safetensors"
    );
    let text_uncond = if neg_path.exists() {
        let neg = flame_core::serialization::load_file(neg_path, &device)?;
        let t = neg.get("text_hidden")
            .ok_or_else(|| anyhow::anyhow!("Missing text_hidden in negative"))?
            .clone();
        println!("  Unconditional: {:?} (official empty-string)", t.dims());
        t
    } else {
        let t = Tensor::zeros_dtype(text_cond.shape().clone(), DType::BF16, device.clone())?;
        println!("  Unconditional: zeros {:?} (fallback)", t.dims());
        t
    };

    // Stage 2: Load transformer
    println!("\n--- Stage 2: Load transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Stage 3: Noise + schedule
    println!("\n--- Stage 3: Prepare noise + sigmas ---");
    let numel = LATENT_CHANNELS * latent_f * latent_h * latent_w;
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
        if numel % 2 == 1 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            v.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }
        v
    };
    let noise = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, LATENT_CHANNELS, latent_f, latent_h, latent_w]),
        device.clone(),
    )?;
    let sigmas = build_dev_sigma_schedule(NUM_STEPS, num_tokens, 0.5, 1.15, 0.0);
    println!("  Noise: {:?}", noise.dims());
    println!("  Sigmas: {:?}", sigmas);

    // Stage 4: Denoise with CFG
    println!("\n--- Stage 4: Denoise ({} steps, CFG={}) ---", NUM_STEPS, GUIDANCE_SCALE);
    let t0 = Instant::now();
    let mut x = noise;

    for step in 0..NUM_STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Forward pass 1: unconditional
        let velocity_uncond = model.forward_video_only(
            &x, &sigma_t, &text_uncond, FRAME_RATE, None,
        )?;

        // Forward pass 2: conditional
        let velocity_cond = model.forward_video_only(
            &x, &sigma_t, text_cond, FRAME_RATE, None,
        )?;

        // CFG: pred = uncond + scale * (cond - uncond)
        let delta = velocity_cond.sub(&velocity_uncond)?;
        let velocity = velocity_uncond.add(&delta.mul_scalar(GUIDANCE_SCALE)?)?;

        // Euler step
        if sigma_next == 0.0 {
            x = x.sub(&velocity.mul_scalar(sigma)?)?;
        } else {
            let dt = sigma_next - sigma;
            x = x.add(&velocity.mul_scalar(dt)?)?;
        }

        let dt_step = t_step.elapsed().as_secs_f32();
        println!("  Step {}/{} sigma={:.4} dt={:.1}s", step + 1, NUM_STEPS, sigma, dt_step);
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.1}s/step)", dt, dt / NUM_STEPS as f32);

    // Stage 5: Save
    println!("\n--- Stage 5: Save latents ---");
    let mut save_map = std::collections::HashMap::new();
    save_map.insert("latents".to_string(), x);
    flame_core::serialization::save_tensors(
        &save_map,
        std::path::Path::new(OUTPUT_PATH),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Saved to {}", OUTPUT_PATH);

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
