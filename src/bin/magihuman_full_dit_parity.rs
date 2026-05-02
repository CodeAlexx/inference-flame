//! End-to-end MagiHuman DiT parity test using BlockOffloader.
//!
//! Loads the dequanted BF16 weights via BlockOffloader, runs the full
//! 40-layer forward against the Python `full_dit_smoke` reference, and
//! compares.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{MagiHumanDiT, MagiHumanDiTSwapped};

// GPU reference (per-layer streaming) — fair comparison vs Rust on GPU.
// The CPU reference (`full_dit_smoke.safetensors`) is NOT comparable because
// CPU vs GPU PyTorch BF16 diverge at cos < 0.6 even on a single layer.
const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/full_dit_smoke_gpu.safetensors";
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

    println!("--- Loading fixture ---");
    let fix = flame_core::serialization::load_file(Path::new(FIXTURE), &device)
        .map_err(|e| anyhow!("load fixture: {e}"))?;
    let input = fix.get("input").ok_or_else(|| anyhow!("missing input"))?.clone();
    let coords = fix.get("coords").ok_or_else(|| anyhow!("missing coords"))?.clone();
    let reference = fix.get("output").ok_or_else(|| anyhow!("missing output"))?.clone();
    let v = fix.get("meta.video_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let a = fix.get("meta.audio_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let t = fix.get("meta.text_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    println!("  group_sizes: V={v} A={a} T={t} (L={})", v + a + t);

    let mode = std::env::args().nth(1).unwrap_or_else(|| "swapped".to_string());
    let input_f32 = input.to_dtype(DType::F32)?;
    let coords_f32 = coords.to_dtype(DType::F32)?;
    let group_sizes = [v, a, t];
    let t_start = std::time::Instant::now();
    let out = if mode == "in_memory" {
        // Use the in-memory MagiHumanDiT (no offloader). Will OOM at 24 GB —
        // run with `cargo run --bin magihuman_full_dit_parity in_memory` to
        // attempt and see where it fails. Useful for isolating offloader vs
        // accumulator bugs.
        println!("\n--- Loading MagiHumanDiT (in-memory) ---");
        let weights = flame_core::serialization::load_file(Path::new(WEIGHTS), &device)
            .map_err(|e| anyhow!("load weights: {e}"))?;
        println!("loaded {} tensors", weights.len());
        let model = MagiHumanDiT::load(&weights)
            .map_err(|e| anyhow!("model load: {e}"))?;
        println!("\n--- Forward (Rust, in-memory) ---");
        model.forward(&input_f32, &coords_f32, &group_sizes)
            .map_err(|e| anyhow!("forward: {e}"))?
    } else {
        println!("\n--- Loading MagiHumanDiTSwapped ---");
        let mut model = MagiHumanDiTSwapped::load(WEIGHTS, &device)
            .map_err(|e| anyhow!("model load: {e}"))?;
        println!("\n--- Forward (Rust, 40 layers via BlockOffloader) ---");
        model.forward(&input_f32, &coords_f32, &group_sizes)
            .map_err(|e| anyhow!("forward: {e}"))?
    };
    println!("  forward took {:.2}s", t_start.elapsed().as_secs_f32());
    println!("  output shape: {:?} dtype: {:?}", out.shape().dims(), out.dtype());

    let ref_dims = reference.shape().dims().to_vec();
    let out_dims = out.shape().dims().to_vec();
    if ref_dims != out_dims {
        return Err(anyhow!("shape mismatch: ref {ref_dims:?} vs out {out_dims:?}"));
    }

    let ref_vec = reference.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ref: {e}"))?;
    let out_vec = out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("out: {e}"))?;
    let mxa = max_abs(&out_vec, &ref_vec);
    let cos = cosine(&out_vec, &ref_vec);
    let abs_ref = ref_vec.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("\n--- Parity ---");
    println!("  max_abs : {mxa:.4}  (ref_max_abs={abs_ref:.4}, ratio={:.6})", mxa / abs_ref);
    println!("  cos     : {cos:.6}");
    println!("  out[0..5]: {:?}", &out_vec[..5]);
    println!("  ref[0..5]: {:?}", &ref_vec[..5]);
    let n = out_vec.len();
    println!("  out[mid]: {:?}", &out_vec[n / 2..n / 2 + 5]);
    println!("  ref[mid]: {:?}", &ref_vec[n / 2..n / 2 + 5]);

    if cos < 0.999 {
        return Err(anyhow!("PARITY FAIL: cos {cos} < 0.999"));
    }
    println!("\n  PARITY OK ✓");
    Ok(())
}
