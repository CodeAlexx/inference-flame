//! Parity test for MagiHuman layer 0 (mm_layer + GELU7).

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{MMTransformerLayer, MlpAct};

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/layer0_smoke.safetensors";

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
    let rope = fix.get("rope").ok_or_else(|| anyhow!("missing rope"))?.clone();
    let reference = fix.get("output").ok_or_else(|| anyhow!("missing output"))?.clone();
    let v = fix.get("meta.video_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let a = fix.get("meta.audio_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let t = fix.get("meta.text_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    println!("group_sizes: V={v} A={a} T={t} (total={})", v + a + t);

    let mut w = std::collections::HashMap::new();
    for (k, v) in fix.iter() {
        if let Some(stripped) = k.strip_prefix("w.") {
            w.insert(stripped.to_string(), v.clone());
        }
    }
    println!("loaded {} weights", w.len());

    let layer = MMTransformerLayer::load(&w, "block.layers.0.", MlpAct::GELU7)
        .map_err(|e| anyhow!("layer load: {e}"))?;

    let input_bf16 = input.to_dtype(DType::BF16)?;
    let rope_f32 = rope.to_dtype(DType::F32)?;
    let group_sizes = vec![v, a, t];
    let out = layer.forward(&input_bf16, &rope_f32, &group_sizes)
        .map_err(|e| anyhow!("forward: {e}"))?;

    let ref_vec = reference.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ref: {e}"))?;
    let out_vec = out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("out: {e}"))?;
    let mxa = max_abs(&out_vec, &ref_vec);
    let cos = cosine(&out_vec, &ref_vec);
    let abs_ref_max = ref_vec.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("max_abs={mxa:.4} (ref_max_abs={abs_ref_max:.4}, ratio={:.4}) cos={cos:.6}", mxa / abs_ref_max);
    println!("out[0..5]: {:?}", &out_vec[..5]);
    println!("ref[0..5]: {:?}", &ref_vec[..5]);

    if cos < 0.999 {
        return Err(anyhow!("PARITY FAIL: cos {cos} < 0.999"));
    }
    println!("\nPARITY OK ✓");
    Ok(())
}
