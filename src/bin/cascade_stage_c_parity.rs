//! Stage C step-0 parity: load reference (x, text_embed, text_pooled, v_cond, v_uncond)
//! from scripts/cascade_stage_c_ref.py and compare against Rust UNet forward.
//!
//! Run:
//!   cargo run --release --bin cascade_stage_c_parity

use std::path::PathBuf;

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::wuerstchen_unet::{WuerstchenUNet, WuerstchenUNetConfig};

const CKPT_ROOT: &str =
    "/home/alex/.cache/huggingface/hub/models--stabilityai--stable-cascade/snapshots/a89f66d459ae653e3b4d4f992a7c3789d0dc4d16";

fn ref_path() -> PathBuf {
    PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/cascade_gen/stage_c_ref_step0.safetensors")
}

fn stage_c_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_c_bf16.safetensors")
}

fn cos_sim(a: &Tensor, b: &Tensor) -> anyhow::Result<(f32, f32, f32)> {
    assert_eq!(a.shape().dims(), b.shape().dims(), "shape mismatch: {:?} vs {:?}", a.shape().dims(), b.shape().dims());
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs_diff = 0.0f32;
    let mut max_abs_ref = 0.0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        let xd = *x as f64;
        let yd = *y as f64;
        dot += xd * yd;
        na += xd * xd;
        nb += yd * yd;
        let d = (x - y).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        let r = y.abs();
        if r > max_abs_ref { max_abs_ref = r; }
    }
    let cos = if na > 0.0 && nb > 0.0 { (dot / (na.sqrt() * nb.sqrt())) as f32 } else { 0.0 };
    let rel = if max_abs_ref > 0.0 { max_abs_diff / max_abs_ref } else { max_abs_diff };
    Ok((cos, max_abs_diff, rel))
}

fn tensor_stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
    let n = v.len();
    let mean: f32 = v.iter().sum::<f32>() / n as f32;
    let max = v.iter().cloned().fold(f32::MIN, f32::max);
    let min = v.iter().cloned().fold(f32::MAX, f32::min);
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    println!("  [{name}] shape={:?} mean={:.4} std={:.4} min={:.4} max={:.4}",
             t.shape().dims(), mean, var.sqrt(), min, max);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::try_init().ok();
    let device = global_cuda_device();

    println!("=== Stage C step-0 parity ===");

    // Load reference.
    let ref_weights = flame_core::serialization::load_file(ref_path(), &device)?;
    let get = |k: &str| -> anyhow::Result<Tensor> {
        ref_weights.get(k).cloned().ok_or_else(|| anyhow::anyhow!("missing ref key: {k}"))
    };

    let x          = get("x")?;
    let text_embed = get("text_embed")?;
    let text_pooled_3d = get("text_pooled")?;  // [1, 1, 1280]
    let neg_embed  = get("neg_embed")?;
    let neg_pooled_3d = get("neg_pooled")?;
    let v_cond_ref = get("v_cond")?;
    let v_uncond_ref = get("v_uncond")?;

    println!();
    println!("Reference inputs:");
    tensor_stats("x", &x)?;
    tensor_stats("text_embed", &text_embed)?;
    tensor_stats("text_pooled_3d", &text_pooled_3d)?;
    tensor_stats("v_cond_ref", &v_cond_ref)?;
    tensor_stats("v_uncond_ref", &v_uncond_ref)?;

    // Our forward() takes pooled as 2D [B, 1280]; squeeze the ref's [B, 1, 1280].
    let text_pooled = text_pooled_3d.squeeze(Some(1))?;
    let neg_pooled  = neg_pooled_3d.squeeze(Some(1))?;

    println!();
    println!("Loading Stage C UNet...");
    let mut unet = WuerstchenUNet::load(
        stage_c_path().to_str().unwrap(),
        WuerstchenUNetConfig::stage_c(),
        &device,
    )?;
    println!("  loaded.");

    println!();
    println!("Running Rust Stage C forward at t=1.0 (cond)...");
    let v_cond_rust = unet.forward(
        &x, 1.0,
        Some(&text_pooled),
        Some(&text_embed),
        None,
    )?;
    tensor_stats("v_cond_rust", &v_cond_rust)?;

    let (cos_c, max_c, rel_c) = cos_sim(&v_cond_rust, &v_cond_ref)?;
    println!();
    println!("  v_cond:   cos_sim={:.6}  max_abs_diff={:.4e}  max_rel={:.4e}", cos_c, max_c, rel_c);

    println!();
    println!("Running Rust Stage C forward at t=1.0 (uncond)...");
    let v_uncond_rust = unet.forward(
        &x, 1.0,
        Some(&neg_pooled),
        Some(&neg_embed),
        None,
    )?;
    tensor_stats("v_uncond_rust", &v_uncond_rust)?;

    let (cos_u, max_u, rel_u) = cos_sim(&v_uncond_rust, &v_uncond_ref)?;
    println!("  v_uncond: cos_sim={:.6}  max_abs_diff={:.4e}  max_rel={:.4e}", cos_u, max_u, rel_u);

    println!();
    println!("=== Summary ===");
    println!("  cond   cos_sim: {:.6} {}", cos_c, if cos_c >= 0.99 { "OK" } else { "FAIL" });
    println!("  uncond cos_sim: {:.6} {}", cos_u, if cos_u >= 0.99 { "OK" } else { "FAIL" });

    let _ = device;
    Ok(())
}
