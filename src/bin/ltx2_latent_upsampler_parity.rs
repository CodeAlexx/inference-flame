//! Parity check for the Rust `LTX2LatentUpsampler` port.
//!
//! Loads `output/ltx2_latent_upsampler_ref.safetensors` (produced by
//! `scripts/ltx2_latent_upsampler_ref.py` — which imports Lightricks's
//! actual `LatentUpsampler` class from `/tmp/ltx-video` and runs it on
//! the on-disk `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` checkpoint),
//! feeds `input_latent` through the Rust upsampler, and compares the
//! result against `output_latent` with cos_sim + max_abs.
//!
//! Pass criterion (BF16 end-to-end, 10-layer conv stack with groupnorm
//! on a 1024-channel feature map):
//!   cos_sim ≥ 0.999, max_abs < 0.05
//!
//! Run:
//!   cargo run --release --bin ltx2_latent_upsampler_parity

use flame_core::{global_cuda_device, DType, Tensor};
use std::path::Path;

use inference_flame::models::ltx2_upsampler::LTX2LatentUpsampler;

const REF_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_latent_upsampler_ref.safetensors";

const CKPT_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors";

fn cos_sim_f64(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    let a_f32 = a.to_dtype(DType::F32)?.to_vec()?;
    let b_f32 = b.to_dtype(DType::F32)?.to_vec()?;
    if a_f32.len() != b_f32.len() {
        return Err(anyhow::anyhow!(
            "cos_sim len mismatch: {} vs {}", a_f32.len(), b_f32.len()));
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a_f32.len() {
        let x = a_f32[i] as f64;
        let y = b_f32[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    Ok(dot / (na.sqrt() * nb.sqrt() + 1e-20))
}

fn max_abs_delta(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a_f32 = a.to_dtype(DType::F32)?.to_vec()?;
    let b_f32 = b.to_dtype(DType::F32)?.to_vec()?;
    if a_f32.len() != b_f32.len() {
        return Err(anyhow::anyhow!(
            "max_abs len mismatch: {} vs {}", a_f32.len(), b_f32.len()));
    }
    let mut m = 0.0f32;
    for i in 0..a_f32.len() {
        let d = (a_f32[i] - b_f32[i]).abs();
        if d > m { m = d; }
    }
    Ok(m)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("=== LTX-2 LatentUpsampler parity ===");
    println!("ref:  {REF_PATH}");
    println!("ckpt: {CKPT_PATH}");

    if !Path::new(REF_PATH).exists() {
        return Err(anyhow::anyhow!(
            "reference file missing — run `scripts/ltx2_latent_upsampler_ref.py` first"
        ));
    }
    if !Path::new(CKPT_PATH).exists() {
        return Err(anyhow::anyhow!(
            "upsampler checkpoint missing at {CKPT_PATH}"
        ));
    }

    let device = global_cuda_device();
    let refs = flame_core::serialization::load_file(Path::new(REF_PATH), &device)?;

    let input = refs.get("input_latent")
        .ok_or_else(|| anyhow::anyhow!("ref missing input_latent"))?
        .to_dtype(DType::BF16)?;
    let want = refs.get("output_latent")
        .ok_or_else(|| anyhow::anyhow!("ref missing output_latent"))?
        .to_dtype(DType::BF16)?;

    println!("input  shape = {:?}", input.shape().dims());
    println!("output shape = {:?}", want.shape().dims());

    println!("\nLoading Rust LTX2LatentUpsampler...");
    let upsampler = LTX2LatentUpsampler::load(CKPT_PATH, &device)?;

    println!("Running Rust forward...");
    let got = upsampler.forward(&input)?;

    assert_eq!(
        got.shape().dims(),
        want.shape().dims(),
        "output shape mismatch: got={:?} want={:?}",
        got.shape().dims(),
        want.shape().dims(),
    );

    let cos = cos_sim_f64(&got, &want)?;
    let max = max_abs_delta(&got, &want)?;

    println!("\n  cos_sim = {cos:.6}");
    println!("  max_abs = {max:.4e}");

    // BF16 end-to-end + cuDNN conv3d + groupnorm in NHWC: expect
    // cos ≥ 0.999 and max_abs in 1e-2 range. Fail hard if we drift below.
    const COS_MIN: f64 = 0.999;
    const MAX_ABS_MAX: f32 = 0.05;

    let ok = cos >= COS_MIN && max <= MAX_ABS_MAX;
    println!(
        "  thresholds: cos ≥ {COS_MIN}, max_abs ≤ {MAX_ABS_MAX}  →  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    if ok {
        println!("\n=== PASS ===");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "parity fail: cos_sim={cos:.6} (≥{COS_MIN})  max_abs={max:.4e} (≤{MAX_ABS_MAX})"
        ))
    }
}
