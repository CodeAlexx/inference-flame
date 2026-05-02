//! Bisect cos drop across the 40-token chain to find where Rust diverges
//! from Python reference.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{
    MMTransformerLayer, MlpAct, SharedTransformerLayer, GELU7_LAYERS, MM_LAYERS,
};

const FIXTURE_INPUT: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/h_after_adapter_40tok.safetensors";
const WEIGHTS: &str = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// Compare cos and max-abs both globally and per modality (V=24, A=8, T=8 rows of 5120).
fn report(label: &str, ours: &[f32], refr: &[f32]) {
    const HIDDEN: usize = 5120;
    const V: usize = 24;
    const A: usize = 8;
    const T: usize = 8;
    let v_ours = &ours[0..V * HIDDEN];
    let v_ref = &refr[0..V * HIDDEN];
    let a_ours = &ours[V * HIDDEN..(V + A) * HIDDEN];
    let a_ref = &refr[V * HIDDEN..(V + A) * HIDDEN];
    let t_ours = &ours[(V + A) * HIDDEN..(V + A + T) * HIDDEN];
    let t_ref = &refr[(V + A) * HIDDEN..(V + A + T) * HIDDEN];
    println!(
        "{label}: cos all={:.6} V={:.6} A={:.6} T={:.6}  |  max_abs ours={:.2} ref={:.2}",
        cosine(ours, refr),
        cosine(v_ours, v_ref),
        cosine(a_ours, a_ref),
        cosine(t_ours, t_ref),
        max_abs(ours),
        max_abs(refr),
    );
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
    let input = input_fix.get("input").unwrap().clone();
    let rope = input_fix.get("rope").unwrap().clone();
    let group_sizes = vec![24usize, 8, 8];

    let cut_points = [1usize, 2, 3, 4, 8, 16, 24, 32];
    let mut refs: Vec<(usize, Vec<f32>)> = Vec::new();
    for &n in cut_points.iter() {
        let path = format!("/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/40tok_first_{n}_layers.safetensors");
        let fix = flame_core::serialization::load_file(Path::new(&path), &device)?;
        let h = fix.get("hidden_after_layers").unwrap().to_dtype(DType::F32)?.to_vec_f32()?;
        refs.push((n, h));
    }

    let mut offloader = flame_diffusion::BlockOffloader::load(&[WEIGHTS], &Facilitator, device.clone())
        .map_err(|e| anyhow!("offloader: {e}"))?;

    let rope_b = rope.to_dtype(DType::F32)?;
    let mut h = input.to_dtype(DType::BF16)?;

    offloader.prefetch_block(0).map_err(|e| anyhow!("prefetch: {e}"))?;
    let mut ref_idx = 0;
    for i in 0..40 {
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
        // Compare at cut points
        if ref_idx < cut_points.len() && cut_points[ref_idx] == i + 1 {
            let our_v = h.to_dtype(DType::F32)?.to_vec_f32()?;
            report(&format!("after {:>2} layers", cut_points[ref_idx]), &our_v, &refs[ref_idx].1);
            ref_idx += 1;
        }
        drop(weights);
        flame_core::cuda_alloc_pool::clear_pool_cache();
    }
    Ok(())
}
