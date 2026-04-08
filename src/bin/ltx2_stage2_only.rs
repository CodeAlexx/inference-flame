//! Minimal stage-2-only test: bypass stage 1 and the Rust upsampler entirely
//! by loading Rust's stage 1 latents + Python's upscaled latent from disk.
//!
//! Runs the stage 2 noise injection + 3-step denoise loop through
//! `forward_audio_video` and saves the final latent.
//!
//! This isolates the bug: if the final std is ~1.0 with Python's upscaled
//! input, we know the Rust stage 2 denoise loop works and the only remaining
//! bug is the Rust upsampler itself.

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::models::ltx2_upsampler::LTX2LatentUpsampler;
use inference_flame::sampling::ltx2_sampling::LTX2_STAGE2_DISTILLED_SIGMAS;
use std::collections::HashMap;
use std::time::Instant;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const UPSAMPLER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors";
const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

const LATENT_CHANNELS: usize = 128;
const AUDIO_CHANNELS: usize = 8;
const AUDIO_MEL_BINS: usize = 16;
const NUM_FRAMES: usize = 257;
const TARGET_WIDTH: usize = 512;
const TARGET_HEIGHT: usize = 320;
const FRAME_RATE: f32 = 25.0;
const SEED: u64 = 42;

// Deterministic normal noise — mirrors `make_noise` in ltx2_two_stage.rs so
// both binaries share the exact same RNG semantics.
fn make_noise(_numel: usize, seed: u64, shape: &[usize], device: &std::sync::Arc<flame_core::CudaDevice>) -> anyhow::Result<Tensor> {
    flame_core::rng::set_seed(seed);
    let t = Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

fn stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let v = t.to_dtype(DType::F32)?.to_vec()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| { let d = *x as f64 - mean; d*d }).sum::<f64>() / n;
    let std = var.sqrt();
    println!("  {:<30} shape={:?}  mean={:+.4}  std={:.4}", name, t.shape().dims(), mean, std);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    println!("=== Stage 2 only (isolated) ===");

    // --- Load cached text embeddings ---
    let cache_dir = format!("{OUTPUT_DIR}/embed_cache");
    let vc = flame_core::serialization::load_file(
        std::path::Path::new(&format!("{cache_dir}/video_context.safetensors")), &device)?;
    let ac = flame_core::serialization::load_file(
        std::path::Path::new(&format!("{cache_dir}/audio_context.safetensors")), &device)?;
    let video_context = vc.get("video_context").unwrap().to_dtype(DType::BF16)?;
    let audio_context = ac.get("audio_context").unwrap().to_dtype(DType::BF16)?;
    stats("video_context", &video_context)?;
    stats("audio_context", &audio_context)?;

    // --- Run the Rust upsampler on Rust's stage 1 latent ---
    let rust_s1 = flame_core::serialization::load_file(
        std::path::Path::new(&format!("{OUTPUT_DIR}/ltx2_stage1_video_latents.safetensors")), &device)?;
    let s1_latent = rust_s1.get("latents").unwrap().to_dtype(DType::BF16)?;
    stats("rust_s1 (normalized)", &s1_latent)?;

    // VAE stats come from the distilled checkpoint's
    // `vae.per_channel_statistics.{mean,std}-of-means` — the diffusers VAE
    // `latents_{mean,std}` are different numbers.
    let vae_stats = flame_core::serialization::load_file_filtered(
        std::path::Path::new(MODEL_PATH), &device,
        |k| k == "vae.per_channel_statistics.mean-of-means"
            || k == "vae.per_channel_statistics.std-of-means",
    )?;
    let mean = vae_stats.get("vae.per_channel_statistics.mean-of-means").unwrap();
    let std = vae_stats.get("vae.per_channel_statistics.std-of-means").unwrap();
    let mean_5d = mean.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let std_5d = std.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;

    // un-normalize → upsample → re-normalize, mirroring Python's upsample_video.
    let unnorm = s1_latent.mul(&std_5d)?.add(&mean_5d)?;
    let upsampler = LTX2LatentUpsampler::load(UPSAMPLER_PATH, &device)?;
    let upscaled = upsampler.forward(&unnorm)?;
    let mut video_x = upscaled.sub(&mean_5d)?.div(&std_5d)?;
    drop(upsampler);
    stats("video_x (after Rust upsampler)", &video_x)?;

    // --- Load Rust stage 1 audio latent ---
    let rust_s1_audio = flame_core::serialization::load_file(
        std::path::Path::new(&format!("{OUTPUT_DIR}/ltx2_stage1_audio_latents.safetensors")), &device)?;
    let mut audio_x = rust_s1_audio.get("latents").unwrap().to_dtype(DType::BF16)?;
    stats("audio_x (from Rust stage 1)", &audio_x)?;

    // --- Load transformer ---
    println!("\n--- Load Transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    model.init_swap()?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Stage 2 noise injection ---
    println!("\n--- Stage 2: Refine ---");
    let s2_sigmas = LTX2_STAGE2_DISTILLED_SIGMAS.to_vec();
    let s2_steps = s2_sigmas.len() - 1;
    let noise_scale = s2_sigmas[0];
    println!("  noise_scale = {:.4} ({} steps)", noise_scale, s2_steps);

    let v_dims = video_x.shape().dims().to_vec();
    let s2_numel = v_dims.iter().product::<usize>();
    let noise = make_noise(s2_numel, SEED + 100, &v_dims, &device)?;
    video_x = video_x.mul_scalar(1.0 - noise_scale)?.add(&noise.mul_scalar(noise_scale)?)?;
    drop(noise);
    stats("video_x (after noise injection)", &video_x)?;

    let a_dims = audio_x.shape().dims().to_vec();
    let a_numel = a_dims.iter().product::<usize>();
    let audio_noise = make_noise(a_numel, SEED + 101, &a_dims, &device)?;
    audio_x = audio_x.mul_scalar(1.0 - noise_scale)?.add(&audio_noise.mul_scalar(noise_scale)?)?;
    drop(audio_noise);
    stats("audio_x (after noise injection)", &audio_x)?;

    // --- 3-step denoise loop ---
    for step in 0..s2_steps {
        let sigma = s2_sigmas[step];
        let sigma_next = s2_sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,
        )?;

        let dt = sigma_next - sigma;
        video_x = video_x.add(&video_vel.mul_scalar(dt)?)?;
        audio_x = audio_x.add(&audio_vel.mul_scalar(dt)?)?;

        print!("  step {}/{} sigma={:.4} dt={:.1}s ",
            step + 1, s2_steps, sigma, t_step.elapsed().as_secs_f32());
        stats("after step", &video_x)?;
    }

    println!("\n--- Final ---");
    stats("video_x (final)", &video_x)?;
    stats("audio_x (final)", &audio_x)?;

    let mut out = HashMap::new();
    out.insert("latents".to_string(), video_x);
    flame_core::serialization::save_tensors(
        &out,
        std::path::Path::new(&format!("{OUTPUT_DIR}/rust_stage2_only_video.safetensors")),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    let mut aout = HashMap::new();
    aout.insert("latents".to_string(), audio_x);
    flame_core::serialization::save_tensors(
        &aout,
        std::path::Path::new(&format!("{OUTPUT_DIR}/rust_stage2_only_audio.safetensors")),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;

    println!("\nTotal: {:.1}s", t_total.elapsed().as_secs_f32());
    // Suppress unused constants warnings.
    let _ = (NUM_FRAMES, TARGET_WIDTH, TARGET_HEIGHT, LATENT_CHANNELS, AUDIO_CHANNELS, AUDIO_MEL_BINS);
    Ok(())
}
