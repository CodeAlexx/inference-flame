//! Parity test for the MagiHuman Adapter (embedders + RoPE).

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::MagiAdapter;

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/adapter_smoke.safetensors";

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

    let fix = flame_core::serialization::load_file(Path::new(FIXTURE), &device)
        .map_err(|e| anyhow!("load: {e}"))?;
    let input = fix.get("input").ok_or_else(|| anyhow!("missing input"))?.clone();
    let coords = fix.get("coords").ok_or_else(|| anyhow!("missing coords"))?.clone();
    let video_mask_t = fix.get("video_mask").ok_or_else(|| anyhow!("missing video_mask"))?.clone();
    let audio_mask_t = fix.get("audio_mask").ok_or_else(|| anyhow!("missing audio_mask"))?.clone();
    let text_mask_t = fix.get("text_mask").ok_or_else(|| anyhow!("missing text_mask"))?.clone();
    let ref_out = fix.get("output").ok_or_else(|| anyhow!("missing output"))?.clone();
    let ref_rope = fix.get("rope").ok_or_else(|| anyhow!("missing rope"))?.clone();

    let video_mask: Vec<bool> = video_mask_t.to_vec_f32().unwrap().into_iter().map(|x| x > 0.5).collect();
    let audio_mask: Vec<bool> = audio_mask_t.to_vec_f32().unwrap().into_iter().map(|x| x > 0.5).collect();
    let text_mask: Vec<bool> = text_mask_t.to_vec_f32().unwrap().into_iter().map(|x| x > 0.5).collect();
    println!("V={} A={} T={}",
        video_mask.iter().filter(|b| **b).count(),
        audio_mask.iter().filter(|b| **b).count(),
        text_mask.iter().filter(|b| **b).count(),
    );

    let mut w = std::collections::HashMap::new();
    for (k, v) in fix.iter() {
        if let Some(stripped) = k.strip_prefix("w.") {
            w.insert(stripped.to_string(), v.clone());
        }
    }
    let adapter = MagiAdapter::load(&w).map_err(|e| anyhow!("load adapter: {e}"))?;

    // RoPE
    let rope_out = adapter.rope_from_coords(&coords).map_err(|e| anyhow!("rope: {e}"))?;
    let rope_ref_v = ref_rope.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("rope ref: {e}"))?;
    let rope_out_v = rope_out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("rope out: {e}"))?;
    println!("RoPE: max_abs={:.4} cos={:.6}", max_abs(&rope_out_v, &rope_ref_v), cosine(&rope_out_v, &rope_ref_v));

    // Embed
    let input_f32 = input.to_dtype(DType::F32)?;
    let out = adapter.embed(&input_f32, &video_mask, &audio_mask, &text_mask)
        .map_err(|e| anyhow!("embed: {e}"))?;
    let out_v = out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("out: {e}"))?;
    let ref_v = ref_out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ref: {e}"))?;
    let mxa = max_abs(&out_v, &ref_v);
    let cos = cosine(&out_v, &ref_v);
    let abs_ref = ref_v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("Embed: max_abs={mxa:.4} (ref_max={abs_ref:.4}, ratio={:.6}) cos={cos:.6}", mxa / abs_ref);
    println!("out[0..5]: {:?}", &out_v[..5]);
    println!("ref[0..5]: {:?}", &ref_v[..5]);

    if cos < 0.999 {
        return Err(anyhow!("PARITY FAIL: cos {cos} < 0.999"));
    }
    println!("\nPARITY OK ✓");
    Ok(())
}
