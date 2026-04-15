//! LTX-2.3 image-to-video generation — pure Rust, with CFG.
//!
//! Same transformer as T2V, but with latent-space image conditioning:
//! 1. Load pre-encoded image latent (VAE-encoded by Python script)
//! 2. Load cached Gemma text embeddings
//! 3. Blend image latent into noise at frame-0 positions
//! 4. Build conditioning mask (1.0 for frame-0 tokens, 0.0 for rest)
//! 5. Denoise: model sees sigma=0 for conditioned tokens (via mask on timestep)
//! 6. After each step, restore frame-0 latents from init_latents
//! 7. Save denoised latents → decode with Python VAE

use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::sampling::ltx2_sampling::{build_dev_sigma_schedule, LTX2_DISTILLED_SIGMAS};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors";
const OUTPUT_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_i2v_denoised_latents.safetensors";

const NUM_FRAMES: usize = 33;
const WIDTH: usize = 480;
const HEIGHT: usize = 288;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const GUIDANCE_SCALE: f32 = 1.0; // Distilled: no CFG needed
const NUM_STEPS: usize = 8;     // Distilled fixed steps

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let i2v_embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/ltx2_i2v_embeds.safetensors".to_string()
    });
    let output_path = args.get(2).cloned().unwrap_or_else(|| OUTPUT_PATH.to_string());

    println!("============================================================");
    println!("LTX-2.3 Image-to-Video Generation — Pure Rust + CFG");
    println!("============================================================");
    println!("  {}x{}, {} frames, {} steps, cfg={}", WIDTH, HEIGHT, NUM_FRAMES, NUM_STEPS, GUIDANCE_SCALE);

    let device = global_cuda_device();

    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;
    let latent_h = HEIGHT / 32;
    let latent_w = WIDTH / 32;
    let num_tokens = latent_f * latent_h * latent_w;
    println!("  Latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, latent_h, latent_w, num_tokens);

    // Stage 1: Load I2V embeddings (text + image latent)
    println!("\n--- Stage 1: Load I2V embeddings ---");
    let cached = flame_core::serialization::load_file(
        std::path::Path::new(&i2v_embeds_path), &device,
    )?;
    let text_cond = cached.get("text_hidden")
        .ok_or_else(|| anyhow::anyhow!("Missing text_hidden"))?;
    let image_latent = cached.get("image_latent")
        .ok_or_else(|| anyhow::anyhow!("Missing image_latent (VAE-encoded reference image)"))?;
    println!("  Text cond:    {:?} {:?}", text_cond.dims(), text_cond.dtype());
    println!("  Image latent: {:?} {:?}", image_latent.dims(), image_latent.dtype());

    // image_latent should be [1, 128, 1, H_lat, W_lat] (single frame)
    let il_dims = image_latent.shape().dims().to_vec();
    if il_dims.len() != 5 || il_dims[2] != 1 {
        return Err(anyhow::anyhow!(
            "Expected image_latent shape [1, 128, 1, H, W], got {:?}", il_dims
        ));
    }
    let il_h = il_dims[3];
    let il_w = il_dims[4];
    if il_h != latent_h || il_w != latent_w {
        return Err(anyhow::anyhow!(
            "Image latent spatial dims {}x{} don't match target latent {}x{}",
            il_w, il_h, latent_w, latent_h
        ));
    }

    // Unconditional embedding: zeros
    let text_uncond = Tensor::zeros_dtype(text_cond.shape().clone(), DType::BF16, device.clone())?;
    println!("  Unconditional: zeros {:?}", text_uncond.dims());

    // Stage 2: Load transformer
    println!("\n--- Stage 2: Load transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    println!("  Global params loaded in {:.1}s", t0.elapsed().as_secs_f32());
    match model.load_fp8_resident() {
        Ok(()) => println!("  FP8 resident loaded in {:.1}s", t0.elapsed().as_secs_f32()),
        Err(e) => {
            println!("  FP8 resident failed ({e}), falling back to BlockOffloader");
            model.init_offloader()?;
            println!("  BlockOffloader initialized in {:.1}s", t0.elapsed().as_secs_f32());
        }
    }

    // Stage 3: Build noise, blend with image latent, build conditioning mask
    println!("\n--- Stage 3: Prepare noise + I2V conditioning ---");

    // 3a. Expand image latent to all frames (for blending)
    // init_latents: [1, 128, latent_f, H, W] — repeat frame-0 across all frames
    let init_latents = {
        // image_latent is [1, 128, 1, H, W]
        // We need to repeat along dim=2 to get [1, 128, latent_f, H, W]
        let il_bf16 = if image_latent.dtype() != DType::BF16 {
            image_latent.to_dtype(DType::BF16)?
        } else {
            image_latent.clone()
        };
        // Expand by concatenating copies along frame dimension
        let frame_refs: Vec<&Tensor> = (0..latent_f).map(|_| &il_bf16).collect();
        Tensor::cat(&frame_refs, 2)?
    };
    println!("  init_latents (expanded): {:?}", init_latents.dims());

    // 3b. Build conditioning mask [1, 1, latent_f, H, W]
    // Frame 0 = 1.0, rest = 0.0
    let cond_mask_5d = {
        let total = latent_f * latent_h * latent_w;
        let frame0_pixels = latent_h * latent_w;
        let mut mask_data = vec![0.0f32; total];
        // Set frame 0 to 1.0
        for i in 0..frame0_pixels {
            mask_data[i] = 1.0;
        }
        Tensor::from_f32_to_bf16(
            mask_data,
            Shape::from_dims(&[1, 1, latent_f, latent_h, latent_w]),
            device.clone(),
        )?
    };

    // 3c. Generate noise
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

    // 3d. Blend: latents = init_latents * mask + noise * (1 - mask)
    // For 128-channel latents with 1-channel mask, we need to broadcast
    let inv_mask_5d = cond_mask_5d.mul_scalar(-1.0)?.add_scalar(1.0)?;
    let x_init = init_latents.mul(&cond_mask_5d.expand(init_latents.shape().dims())?)?;
    let x_noise = noise.mul(&inv_mask_5d.expand(noise.shape().dims())?)?;
    let blended = x_init.add(&x_noise)?;
    println!("  Blended latents: {:?}", blended.dims());

    // 3e. Build packed conditioning mask for transformer [1, num_tokens]
    // Pack mask from [1, 1, F, H, W] → [1, F*H*W] (patch_size=1 for LTX-2.3)
    let cond_mask_packed = cond_mask_5d.reshape(&[1, latent_f * latent_h * latent_w])?;
    println!("  Conditioning mask (packed): {:?}", cond_mask_packed.dims());

    // 3f. Sigma schedule
    let sigmas = if GUIDANCE_SCALE <= 1.0 {
        LTX2_DISTILLED_SIGMAS.to_vec()
    } else {
        build_dev_sigma_schedule(NUM_STEPS, num_tokens, 0.5, 1.15, 0.0)
    };
    println!("  Noise: {:?}", blended.dims());
    println!("  Sigmas: {:?}", sigmas);

    // Stage 4: Denoise with I2V conditioning
    println!("\n--- Stage 4: Denoise ({} steps, CFG={}, I2V) ---", NUM_STEPS, GUIDANCE_SCALE);
    let t0 = Instant::now();
    let mut x = blended;

    for step in 0..NUM_STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        let velocity = if GUIDANCE_SCALE > 1.0 {
            // CFG: two forward passes with conditioning mask
            let velocity_uncond = model.forward_video_only_i2v(
                &x, &sigma_t, &text_uncond, FRAME_RATE, None, Some(&cond_mask_packed),
            )?;
            let velocity_cond = model.forward_video_only_i2v(
                &x, &sigma_t, text_cond, FRAME_RATE, None, Some(&cond_mask_packed),
            )?;
            let delta = velocity_cond.sub(&velocity_uncond)?;
            velocity_uncond.add(&delta.mul_scalar(GUIDANCE_SCALE)?)?
        } else {
            // No CFG: single forward pass (distilled model)
            model.forward_video_only_i2v(
                &x, &sigma_t, text_cond, FRAME_RATE, None, Some(&cond_mask_packed),
            )?
        };

        // The velocity is [B, C, F, H, W].
        // For I2V: only euler-step on frames 1+, keep frame 0 intact.
        // Split velocity and latents into frame-0 and rest.
        let x_frame0 = x.narrow(2, 0, 1)?;  // [1, C, 1, H, W]
        let x_rest = x.narrow(2, 1, latent_f - 1)?;  // [1, C, F-1, H, W]
        let v_rest = velocity.narrow(2, 1, latent_f - 1)?;  // velocity for non-conditioned frames

        // Euler step on frames 1+ only
        let x_rest_new = if sigma_next == 0.0 {
            x_rest.sub(&v_rest.mul_scalar(sigma)?)?
        } else {
            let dt = sigma_next - sigma;
            x_rest.add(&v_rest.mul_scalar(dt)?)?
        };

        // Re-concatenate: keep original frame 0, use denoised rest
        x = Tensor::cat(&[&x_frame0, &x_rest_new], 2)?;

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
        std::path::Path::new(&output_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Saved to {}", output_path);

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
