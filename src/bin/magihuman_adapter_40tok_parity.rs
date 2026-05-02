//! Adapter parity for the 40-token full-DiT config (V=24 A=8 T=8).
//!
//! Verifies the Rust adapter produces the same h + rope as the Python
//! reference for the EXACT input used by `full_dit_smoke_gpu.safetensors`.
//! If this fails, the bug is in my adapter for this token-count config;
//! if it passes, the bug is downstream.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::MagiAdapter;

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/adapter_40tok_smoke.safetensors";
const WEIGHTS: &str = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();
    let fix = flame_core::serialization::load_file(Path::new(FIXTURE), &device)?;
    let input = fix.get("input").unwrap().clone();
    let coords = fix.get("coords").unwrap().clone();
    let ref_h = fix.get("h_after_adapter").unwrap().clone();
    let ref_rope = fix.get("rope_after_adapter").unwrap().clone();

    // Load shared weights from the dequanted DiT (same source as full DiT path).
    let shared_prefixes = ["adapter.", "final_"];
    let weights = flame_core::serialization::load_file_filtered(
        Path::new(WEIGHTS), &device,
        |k| shared_prefixes.iter().any(|p| k.starts_with(p)),
    )?;
    let adapter = MagiAdapter::load(&weights).map_err(|e| anyhow!("adapter: {e}"))?;

    let v_count = 24;
    let a_count = 8;
    let t_count = 8;
    let l = v_count + a_count + t_count;
    let video_mask: Vec<bool> = (0..l).map(|i| i < v_count).collect();
    let audio_mask: Vec<bool> = (0..l).map(|i| i >= v_count && i < v_count + a_count).collect();
    let text_mask: Vec<bool> = (0..l).map(|i| i >= v_count + a_count).collect();

    let input_f32 = input.to_dtype(DType::F32)?;
    let h = adapter.embed(&input_f32, &video_mask, &audio_mask, &text_mask)
        .map_err(|e| anyhow!("embed: {e}"))?;
    let rope = adapter.rope_from_coords(&coords)
        .map_err(|e| anyhow!("rope: {e}"))?;

    let ref_h_v = ref_h.to_dtype(DType::F32)?.to_vec_f32()?;
    let h_v = h.to_dtype(DType::F32)?.to_vec_f32()?;
    let h_cos = cosine(&h_v, &ref_h_v);
    let h_mxa = max_abs(&h_v, &ref_h_v);
    let h_abs_ref = ref_h_v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("Adapter embed:  cos={h_cos:.6} max_abs={h_mxa:.6} (ref_max={h_abs_ref:.4} ratio={:.6})", h_mxa / h_abs_ref);

    let ref_rope_v = ref_rope.to_dtype(DType::F32)?.to_vec_f32()?;
    let rope_v = rope.to_dtype(DType::F32)?.to_vec_f32()?;
    let r_cos = cosine(&rope_v, &ref_rope_v);
    let r_mxa = max_abs(&rope_v, &ref_rope_v);
    println!("Adapter rope:   cos={r_cos:.6} max_abs={r_mxa:.6}");

    if h_cos < 0.999 || r_cos < 0.999 {
        return Err(anyhow!("PARITY FAIL"));
    }
    println!("\nPARITY OK ✓");
    Ok(())
}
