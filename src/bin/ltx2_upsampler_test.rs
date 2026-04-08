//! Stand-alone test: run the Rust LTX-2 spatial upsampler on Rust's stage 1
//! latent and save the output to disk so it can be diffed against Python's
//! `python_upscaled_from_rust.safetensors`.
//!
//! Mirrors the exact sequence in `ltx2_two_stage.rs`:
//!   1. un-normalize with VAE per-channel statistics
//!   2. upsample
//!   3. save the upsampled-but-not-yet-renormalized tensor
//!   4. re-normalize
//!   5. save the re-normalized tensor
//!
//! If step 3 already disagrees with Python, the Rust upsampler is the bug.
//! If step 3 matches but step 5 doesn't, the re-normalize step is the bug.

use std::collections::HashMap;

use anyhow::Result;
use flame_core::{global_cuda_device, DType, Tensor};

use inference_flame::models::ltx2_upsampler::LTX2LatentUpsampler;

const LATENT_CHANNELS: usize = 128;
const RUST_S1_VIDEO: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_stage1_video_latents.safetensors";
// Official Lightricks upsampler — the diffusers copy has *different* weights
// (max |diff| ~0.8) so it doesn't work for LTX-2.3.
const UPSAMPLER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors";
// Python's upsample_video() calls `encoder.per_channel_statistics.un_normalize(latent)`
// where `encoder` is built from the DISTILLED checkpoint and reads
// `vae.per_channel_statistics.{mean,std}-of-means`. The diffusers VAE
// `latents_mean`/`latents_std` are *different* numbers (max diff ~0.79) and
// produce a different un-normalized latent, so we load the distilled stats.
const DISTILLED_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const OUT_RAW: &str =
    "/home/alex/EriDiffusion/inference-flame/output/rust_upscaled_from_rust.safetensors";
const OUT_NORM: &str =
    "/home/alex/EriDiffusion/inference-flame/output/rust_upscaled_from_rust_normalized.safetensors";

fn stats(name: &str, t: &Tensor) -> Result<()> {
    let v = t.to_dtype(DType::F32)?.to_vec()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| {
        let d = *x as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let std = var.sqrt();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {:<30} shape={:?}  mean={:+.4}  std={:.4}  min={:+.4}  max={:+.4}",
        name, t.shape().dims(), mean, std, min, max,
    );
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("Loading Rust stage 1 latent...");
    let s1 = flame_core::serialization::load_file(RUST_S1_VIDEO, &device)?;
    let video_x = s1.get("latents").unwrap().to_dtype(DType::BF16)?;
    stats("rust_s1", &video_x)?;

    println!("\nLoading VAE per-channel statistics from distilled checkpoint...");
    let vae_weights = flame_core::serialization::load_file_filtered(
        DISTILLED_PATH, &device,
        |k| k == "vae.per_channel_statistics.mean-of-means"
            || k == "vae.per_channel_statistics.std-of-means",
    )?;
    let latents_mean = vae_weights.get("vae.per_channel_statistics.mean-of-means")
        .ok_or_else(|| anyhow::anyhow!("Missing mean-of-means"))?;
    let latents_std = vae_weights.get("vae.per_channel_statistics.std-of-means")
        .ok_or_else(|| anyhow::anyhow!("Missing std-of-means"))?;
    stats("mean-of-means", latents_mean)?;
    stats("std-of-means", latents_std)?;

    let mean_5d = latents_mean.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let std_5d = latents_std.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;

    println!("\nUn-normalizing (x * std + mean)...");
    let video_unnorm = video_x.mul(&std_5d)?.add(&mean_5d)?;
    stats("rust_s1_unnorm", &video_unnorm)?;

    println!("\nLoading upsampler + forwarding...");
    let upsampler = LTX2LatentUpsampler::load(UPSAMPLER_PATH, &device)?;
    let video_upscaled = upsampler.forward(&video_unnorm)?;
    stats("rust_upscaled_raw", &video_upscaled)?;

    let mut raw = HashMap::new();
    raw.insert("latents".to_string(), video_upscaled.clone());
    flame_core::serialization::save_tensors(
        &raw,
        std::path::Path::new(OUT_RAW),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  saved: {}", OUT_RAW);

    println!("\nRe-normalizing ((x - mean) / std)...");
    let video_renorm = video_upscaled.sub(&mean_5d)?.div(&std_5d)?;
    stats("rust_upscaled_renorm", &video_renorm)?;

    let mut norm = HashMap::new();
    norm.insert("latents".to_string(), video_renorm);
    flame_core::serialization::save_tensors(
        &norm,
        std::path::Path::new(OUT_NORM),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  saved: {}", OUT_NORM);

    println!("\nDONE");
    Ok(())
}
