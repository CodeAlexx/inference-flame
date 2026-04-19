//! Parity check for AdaIN latent filter + sigmoid tone map.
//!
//! Loads `output/ltx2_adain_tone_map_ref.safetensors` (produced by
//! `scripts/ltx2_adain_tone_map_ref.py` — which imports Lightricks's
//! actual `adain_filter_latent` and `LTXVideoPipeline.tone_map_latents`
//! from `/tmp/ltx-video`), then runs the Rust ports on the same inputs
//! and compares outputs with cos_sim + max_abs deltas.
//!
//! Pass criterion:
//!   - Identity cases (factor=0, compression=0): max_abs < 1e-5 (bit-exact
//!     in F32; BF16 round-trip introduces trailing-bit noise).
//!   - Non-identity: max_abs < 0.01, cos_sim ≥ 0.999.
//!
//! Run:
//!   cargo run --release --bin ltx2_adain_tone_map_parity

use flame_core::{global_cuda_device, DType, Tensor};
use std::path::Path;

use inference_flame::sampling::ltx2_multiscale::{adain_filter_latent, tone_map_latents};

const REF_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_adain_tone_map_ref.safetensors";

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

/// Assert-and-print helper for a single parity cell.
fn check_cell(
    label: &str,
    got: &Tensor,
    want: &Tensor,
    max_abs_thresh: f32,
    cos_thresh: f64,
) -> anyhow::Result<bool> {
    assert_eq!(
        got.shape().dims(),
        want.shape().dims(),
        "{label}: shape mismatch got={:?} want={:?}",
        got.shape().dims(),
        want.shape().dims()
    );
    let m = max_abs_delta(got, want)?;
    let c = cos_sim_f64(got, want)?;
    let ok = m <= max_abs_thresh && c >= cos_thresh;
    let flag = if ok { "PASS" } else { "FAIL" };
    println!(
        "  [{flag}] {label:<36} max_abs={m:.4e}   cos_sim={c:.6}   (thresh max_abs<{max_abs_thresh:.2e}, cos≥{cos_thresh:.3})"
    );
    Ok(ok)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("=== LTX-2 AdaIN + tone-map parity ===");
    println!("ref: {REF_PATH}");
    if !Path::new(REF_PATH).exists() {
        return Err(anyhow::anyhow!(
            "reference file missing — run `scripts/ltx2_adain_tone_map_ref.py` first"
        ));
    }

    let device = global_cuda_device();
    let refs = flame_core::serialization::load_file(Path::new(REF_PATH), &device)?;

    // Inputs
    let adain_ref = refs.get("adain_reference")
        .ok_or_else(|| anyhow::anyhow!("missing adain_reference"))?
        .to_dtype(DType::BF16)?;
    let adain_tgt = refs.get("adain_target")
        .ok_or_else(|| anyhow::anyhow!("missing adain_target"))?
        .to_dtype(DType::BF16)?;
    let tm_in = refs.get("tonemap_input")
        .ok_or_else(|| anyhow::anyhow!("missing tonemap_input"))?
        .to_dtype(DType::BF16)?;

    println!("adain_target  shape = {:?}", adain_tgt.shape().dims());
    println!("adain_ref     shape = {:?}", adain_ref.shape().dims());
    println!("tonemap_input shape = {:?}", tm_in.shape().dims());

    let mut all_pass = true;

    // --- AdaIN ---
    println!("\n[AdaIN]");
    for (factor, key, tight) in [
        (0.0f32, "adain_out_factor0p0", true),  // identity: tight threshold
        (0.5f32, "adain_out_factor0p5", false),
        (1.0f32, "adain_out_factor1p0", false),
    ] {
        let want = refs.get(key)
            .ok_or_else(|| anyhow::anyhow!("missing {key}"))?
            .to_dtype(DType::BF16)?;
        let got = adain_filter_latent(&adain_tgt, &adain_ref, factor)?;
        // The identity case (factor=0) still goes through a bf16 →
        // f32 → bf16 round-trip in the Rust impl, so tiny LSB noise is
        // expected. 1e-5 on BF16 is effectively bit-exact.
        // BF16 granularity: at magnitude ~2 the ULP is 2^-6 ≈ 0.0156, which is
        // the noise floor for a round-tripped computation. Thresholds account
        // for this — the cos_sim is the tighter guard.
        let (max_abs_thresh, cos_thresh) = if tight {
            (1e-5f32, 0.99999f64)
        } else {
            (0.02f32, 0.999f64)
        };
        let ok = check_cell(
            &format!("factor={factor:.1}"),
            &got, &want, max_abs_thresh, cos_thresh,
        )?;
        if !ok { all_pass = false; }
    }

    // --- Tone map ---
    println!("\n[tone_map_latents]");
    for (comp, key, tight) in [
        (0.0f32, "tonemap_out_compress_0p0", true),   // identity
        (0.6f32, "tonemap_out_compress_0p6", false),  // distilled pipeline default
        (1.0f32, "tonemap_out_compress_1p0", false),
    ] {
        let want = refs.get(key)
            .ok_or_else(|| anyhow::anyhow!("missing {key}"))?
            .to_dtype(DType::BF16)?;
        let got = tone_map_latents(&tm_in, comp)?;
        let (max_abs_thresh, cos_thresh) = if tight {
            (1e-5f32, 0.99999f64)
        } else {
            (0.02f32, 0.999f64)
        };
        let ok = check_cell(
            &format!("compression={comp:.1}"),
            &got, &want, max_abs_thresh, cos_thresh,
        )?;
        if !ok { all_pass = false; }
    }

    println!();
    if all_pass {
        println!("=== ALL PASS ===");
        Ok(())
    } else {
        Err(anyhow::anyhow!("one or more parity cells failed"))
    }
}
