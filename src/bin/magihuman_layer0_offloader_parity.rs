//! Offloader parity for layer 0 (mm_layer + GELU7).
//!
//! Loads layer 0 via `BlockOffloader` (using the BF16-converted weights),
//! runs forward against the SAME `layer0_smoke` fixture used by
//! `magihuman_layer0_parity` (in-memory). If results differ, the bug is in
//! how my pre_transposed forward consumes the BlockOffloader's auto-
//! transposed weight layout — not in the layer math itself.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{MMTransformerLayer, MlpAct};

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/layer0_smoke.safetensors";
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

    let fix = flame_core::serialization::load_file(Path::new(FIXTURE), &device)?;
    let input = fix.get("input").ok_or_else(|| anyhow!("missing input"))?.clone();
    let rope = fix.get("rope").ok_or_else(|| anyhow!("missing rope"))?.clone();
    let reference = fix.get("output").ok_or_else(|| anyhow!("missing output"))?.clone();
    let v = fix.get("meta.video_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let a = fix.get("meta.audio_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let t = fix.get("meta.text_tokens").unwrap().to_vec_f32().unwrap()[0] as usize;
    let group_sizes = vec![v, a, t];

    println!("loading offloader (layer 0 only)");
    let mut offloader = flame_diffusion::BlockOffloader::load(&[WEIGHTS], &Facilitator, device.clone())
        .map_err(|e| anyhow!("offloader: {e}"))?;
    offloader.prefetch_block(0).map_err(|e| anyhow!("prefetch: {e}"))?;
    let raw = offloader.await_block(0).map_err(|e| anyhow!("await: {e}"))?;

    let weights: std::collections::HashMap<String, _> = raw
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    println!("loaded {} weights for layer 0", weights.len());
    for (k, v) in &weights {
        println!("  {k}: {:?} {:?}", v.shape().dims(), v.dtype());
    }

    let layer = MMTransformerLayer::load_with_layout(
        &weights, "block.layers.0.", MlpAct::GELU7, true,
    ).map_err(|e| anyhow!("layer: {e}"))?;

    let input_bf16 = input.to_dtype(DType::BF16)?;
    let rope_f32 = rope.to_dtype(DType::F32)?;
    let out = layer.forward(&input_bf16, &rope_f32, &group_sizes)
        .map_err(|e| anyhow!("forward: {e}"))?;

    let ref_v = reference.to_dtype(DType::F32)?.to_vec_f32()?;
    let out_v = out.to_dtype(DType::F32)?.to_vec_f32()?;
    let cos = cosine(&out_v, &ref_v);
    let mxa = max_abs(&out_v, &ref_v);
    let abs_ref = ref_v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("\nofflader-loaded layer 0:  cos={cos:.6} max_abs={mxa:.4} (ref_max={abs_ref:.4})");

    if cos < 0.999 {
        println!("\nDIVERGENCE: BlockOffloader integration has a bug");
    } else {
        println!("\nOFFLOADER OK — bug is elsewhere");
    }
    Ok(())
}
