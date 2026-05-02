//! GPU-vs-GPU parity for the first N layers of MagiHuman DiT.
//!
//! Loads the layer 0 input fixture (input + rope) and runs the first N
//! transformer layers via BlockOffloader, comparing to a Python GPU
//! reference dumped by `dump_magihuman_first_n_layers_reference.py`.
//!
//! This isolates the chain-of-layers correctness from the CPU-vs-GPU
//! PyTorch BF16 divergence that contaminated the full-DiT CPU reference.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{
    MMTransformerLayer, MlpAct, SharedTransformerLayer, GELU7_LAYERS, MM_LAYERS,
};

const FIXTURE_INPUT: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/h_after_adapter_40tok.safetensors";
const FIXTURE_REF: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/40layers_40tok_smoke.safetensors";
const WEIGHTS: &str = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";
const N_LAYERS: usize = 40;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

struct Facilitator;
impl flame_diffusion::block_offload::BlockFacilitator for Facilitator {
    fn block_count(&self) -> usize { 40 }
    fn classify_key(&self, name: &str) -> Option<usize> {
        name.strip_prefix("block.layers.")?.split('.').next()?.parse().ok()
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    let input_fix = flame_core::serialization::load_file(Path::new(FIXTURE_INPUT), &device)?;
    let ref_fix = flame_core::serialization::load_file(Path::new(FIXTURE_REF), &device)?;
    let input = input_fix.get("input").ok_or_else(|| anyhow!("missing input"))?.clone();
    let rope = input_fix.get("rope").ok_or_else(|| anyhow!("missing rope"))?.clone();
    let v = input_fix.get("meta.video_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let a = input_fix.get("meta.audio_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let t = input_fix.get("meta.text_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let group_sizes = vec![v, a, t];
    let reference = ref_fix.get("hidden_after_layers").ok_or_else(|| anyhow!("missing hidden_after_layers"))?.clone();

    let mut offloader = flame_diffusion::BlockOffloader::load(&[WEIGHTS], &Facilitator, device.clone())
        .map_err(|e| anyhow!("offloader: {e}"))?;

    let rope_b = rope.to_dtype(DType::F32)?;
    let mut h = input.to_dtype(DType::BF16)?;

    println!("running {N_LAYERS} layers via offloader");
    offloader.prefetch_block(0).map_err(|e| anyhow!("prefetch: {e}"))?;
    for i in 0..N_LAYERS {
        let raw = offloader.await_block(i).map_err(|e| anyhow!("await {i}: {e}"))?;
        if i + 1 < 40 {
            offloader.prefetch_block(i + 1).map_err(|e| anyhow!("prefetch {}: {e}", i + 1))?;
        }
        let weights: std::collections::HashMap<String, _> = raw.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let prefix = format!("block.layers.{i}.");
        let is_mm = MM_LAYERS.contains(&i);
        h = if is_mm {
            let act = if GELU7_LAYERS.contains(&i) { MlpAct::GELU7 } else { MlpAct::SwiGLU7 };
            let layer = MMTransformerLayer::load_with_layout(&weights, &prefix, act, true)?;
            layer.forward(&h, &rope_b, &group_sizes)?
        } else {
            let layer = SharedTransformerLayer::load_with_layout(&weights, &prefix, true)?;
            layer.forward(&h, &rope_b)?
        };
        let h_max = h.to_dtype(DType::F32)?.to_vec_f32()?.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        eprintln!("  layer {i} max_abs={h_max:.4}  is_mm={is_mm}");
        drop(weights);
        if i % 4 == 3 {
            flame_core::cuda_alloc_pool::clear_pool_cache();
        }
    }

    let ref_v = reference.to_dtype(DType::F32)?.to_vec_f32()?;
    let out_v = h.to_dtype(DType::F32)?.to_vec_f32()?;
    let cos = cosine(&out_v, &ref_v);
    let mxa = max_abs(&out_v, &ref_v);
    let abs_ref = ref_v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("\nGPU-Rust vs GPU-Python (after {N_LAYERS} layers):");
    println!("  cos = {cos:.6}");
    println!("  max_abs = {mxa:.4}  (ref_max_abs = {abs_ref:.4}, ratio = {:.4})", mxa / abs_ref);
    if cos < 0.999 {
        return Err(anyhow!("PARITY FAIL: cos {cos} < 0.999"));
    }
    println!("\nPARITY OK ✓");
    Ok(())
}
