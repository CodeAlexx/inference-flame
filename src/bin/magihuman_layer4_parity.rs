//! Parity test for ONE shared MagiHuman TransformerLayer (layer 4).
//!
//! Loads the dump from `dump_magihuman_layer4_reference.py` (which holds
//! the input, RoPE, Python output, and dequanted layer weights), runs the
//! Rust port's `SharedTransformerLayer::forward`, and compares.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::SharedTransformerLayer;

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/layer4_smoke.safetensors";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let sum: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).abs()).sum();
    (sum / a.len() as f64) as f32
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("--- Loading fixture ---");
    let fix = flame_core::serialization::load_file(Path::new(FIXTURE), &device)
        .map_err(|e| anyhow!("load: {e}"))?;
    let input = fix.get("input").ok_or_else(|| anyhow!("missing input"))?.clone();
    let rope = fix.get("rope").ok_or_else(|| anyhow!("missing rope"))?.clone();
    let reference = fix.get("output").ok_or_else(|| anyhow!("missing output"))?.clone();
    let seq_len = fix.get("meta.seq_len").unwrap().to_vec_f32().unwrap()[0] as usize;
    println!("  input shape: {:?} dtype: {:?}", input.shape().dims(), input.dtype());
    println!("  rope shape: {:?}", rope.shape().dims());
    println!("  reference shape: {:?}", reference.shape().dims());
    println!("  seq_len: {seq_len}");

    println!("\n--- Building SharedTransformerLayer (layer 4) ---");
    // Construct a fresh weights map from `w.*` keys in the fixture.
    let mut w = std::collections::HashMap::new();
    for (k, v) in fix.iter() {
        if let Some(stripped) = k.strip_prefix("w.") {
            w.insert(stripped.to_string(), v.clone());
        }
    }
    println!("  loaded {} weights from fixture", w.len());
    let layer = SharedTransformerLayer::load(&w, "block.layers.4.")
        .map_err(|e| anyhow!("layer load: {e}"))?;

    println!("\n--- Forward (Rust) ---");
    let input_bf16 = input.to_dtype(DType::BF16)?;
    let rope_f32 = rope.to_dtype(DType::F32)?;
    let out = layer.forward(&input_bf16, &rope_f32)
        .map_err(|e| anyhow!("forward: {e}"))?;
    println!("  output shape: {:?} dtype: {:?}", out.shape().dims(), out.dtype());

    let ref_dims = reference.shape().dims().to_vec();
    let out_dims = out.shape().dims().to_vec();
    if ref_dims != out_dims {
        return Err(anyhow!("shape mismatch: ref {ref_dims:?} vs out {out_dims:?}"));
    }

    let ref_vec = reference.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ref: {e}"))?;
    let out_vec = out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("out: {e}"))?;

    let mxa = max_abs(&out_vec, &ref_vec);
    let mna = mean_abs(&out_vec, &ref_vec);
    let cos = cosine(&out_vec, &ref_vec);

    let abs_ref_max = ref_vec.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("\n--- Parity ---");
    println!("  max_abs : {mxa:.4}  (ref_max_abs={abs_ref_max:.4}, ratio={:.4})", mxa / abs_ref_max);
    println!("  mean_abs: {mna:.4}");
    println!("  cos     : {cos:.6}");
    println!("  out[0..5]: {:?}", &out_vec[..5]);
    println!("  ref[0..5]: {:?}", &ref_vec[..5]);

    if cos < 0.999 {
        return Err(anyhow!("PARITY FAIL: cos {cos} < 0.999"));
    }
    println!("\n  PARITY OK ✓");
    Ok(())
}
