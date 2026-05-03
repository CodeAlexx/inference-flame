//! Isolate which layer is buggy.
//!
//! For each layer K in 0..40, feed Python's reference output of layer K-1
//! as input to OUR layer K, compare to Python's reference output of layer K.
//! If cos < 0.999 the bug is in OUR layer K forward. If cos >= 0.999 the
//! per-layer forward is correct and the chain bisect's drift is purely
//! cumulative numerical accumulation.
//!
//! Reference inputs come from the chain bisect's existing fixtures
//! `40tok_first_{1,2,3,4,8,16,24,32}_layers.safetensors`. Layer 0's input
//! comes from `h_after_adapter_40tok.safetensors`. We can only test K
//! values where both K-1 input and K output fixtures exist, so:
//! K = 1 (input=after-1, ref=after-2), 2, 3, 4, 8, 16, 24, 32.
//! Layer 0 is special: input = adapter output, ref = after-1.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{
    MMTransformerLayer, MlpAct, SharedTransformerLayer, GELU7_LAYERS, MM_LAYERS,
};

const ADAPTER_INPUT: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/h_after_adapter_40tok.safetensors";
const WEIGHTS: &str = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

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
    let max_abs = ours.iter().zip(refr).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!(
        "{label}: cos all={:.6} V={:.6} A={:.6} T={:.6}  max_abs_diff={:.2}",
        cosine(ours, refr),
        cosine(v_ours, v_ref),
        cosine(a_ours, a_ref),
        cosine(t_ours, t_ref),
        max_abs,
    );
}

struct Facilitator;
impl flame_diffusion::block_offload::BlockFacilitator for Facilitator {
    fn block_count(&self) -> usize { 40 }
    fn classify_key(&self, name: &str) -> Option<usize> {
        name.strip_prefix("block.layers.")?.split('.').next()?.parse().ok()
    }
}

fn run_one_layer(
    layer_idx: usize,
    input_h_f32: &[f32],
    rope_b: &flame_core::Tensor,
    group_sizes: &[usize],
    offloader: &mut flame_diffusion::BlockOffloader,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Vec<f32>> {
    use flame_core::{Shape, Tensor};
    // Input as F32 → BF16 (same path as chain bisect)
    let l = input_h_f32.len() / 5120;
    let h_f32 = Tensor::from_vec(input_h_f32.to_vec(), Shape::from_dims(&[l, 5120]), device.clone())?;
    let h = h_f32.to_dtype(DType::BF16)?;

    offloader.prefetch_block(layer_idx).map_err(|e| anyhow!("prefetch: {e}"))?;
    let raw = offloader.await_block(layer_idx).map_err(|e| anyhow!("await: {e}"))?;
    let weights: std::collections::HashMap<String, _> =
        raw.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let prefix = format!("block.layers.{layer_idx}.");
    let is_mm = MM_LAYERS.contains(&layer_idx);
    let out = if is_mm {
        let act = if GELU7_LAYERS.contains(&layer_idx) { MlpAct::GELU7 } else { MlpAct::SwiGLU7 };
        let layer = MMTransformerLayer::load_with_layout(&weights, &prefix, act, true)?;
        layer.forward(&h, rope_b, group_sizes)?
    } else {
        let layer = SharedTransformerLayer::load_with_layout(&weights, &prefix, true)?;
        layer.forward(&h, rope_b)?
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    Ok(out.to_dtype(DType::F32)?.to_vec_f32()?)
}

fn load_fixture_h(path: &str, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<Vec<f32>> {
    let fix = flame_core::serialization::load_file(Path::new(path), device)?;
    let h = fix.get("hidden_after_layers")
        .or_else(|| fix.get("input"))
        .ok_or_else(|| anyhow!("no hidden_after_layers or input in {path}"))?;
    Ok(h.to_dtype(DType::F32)?.to_vec_f32()?)
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    // Adapter fixture has rope + input
    let adapter_fix = flame_core::serialization::load_file(Path::new(ADAPTER_INPUT), &device)?;
    let rope = adapter_fix.get("rope").ok_or_else(|| anyhow!("rope missing"))?.clone();
    let rope_b = rope.to_dtype(DType::F32)?;
    let group_sizes = vec![24usize, 8, 8];

    let mut offloader = flame_diffusion::BlockOffloader::load(&[WEIGHTS], &Facilitator, device.clone())
        .map_err(|e| anyhow!("offloader: {e}"))?;

    // Tests: (layer_idx, input_path, ref_path)
    // For layer K, input = ref-after-K, ref = ref-after-(K+1)
    let adapter_input_v = adapter_fix.get("input").unwrap().to_dtype(DType::F32)?.to_vec_f32()?;

    let fix_path = |n: usize| format!("/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/40tok_first_{n}_layers.safetensors");

    println!("=== ISOLATED PER-LAYER PARITY (Python ref → OUR layer K → compare) ===\n");

    // Layer 0: input = adapter output, ref = after-1
    let ref_after_1 = load_fixture_h(&fix_path(1), &device)?;
    let our_l0 = run_one_layer(0, &adapter_input_v, &rope_b, &group_sizes, &mut offloader, &device)?;
    report("layer 0 (adapter→L0)", &our_l0, &ref_after_1);

    // Layer 1..3 and 4, 8, 16, 24, 32
    let layers_to_test: &[(usize, usize)] = &[
        (1, 2),    // layer 1: input=after-1, ref=after-2
        (2, 3),    // layer 2: input=after-2, ref=after-3
        (3, 4),    // layer 3: input=after-3, ref=after-4 (THE CLIFF)
        (4, 8),    // layer 4: input=after-4, ref=after-8 — note 4 layers run
        // Actually we only have one-layer-step tests for K=0..3, then jumps.
    ];

    // For layers 1, 2, 3 we have consecutive references.
    for &(layer_idx, ref_n) in &layers_to_test[..3] {
        let in_path = fix_path(ref_n - 1);
        let ref_path = fix_path(ref_n);
        let in_h = load_fixture_h(&in_path, &device)?;
        let ref_h = load_fixture_h(&ref_path, &device)?;
        let out = run_one_layer(layer_idx, &in_h, &rope_b, &group_sizes, &mut offloader, &device)?;
        report(&format!("layer {layer_idx} (ref-after-{}→L{}→ref-after-{})", ref_n - 1, layer_idx, ref_n), &out, &ref_h);
    }

    // Layer 4 standalone: input=after-4, ref=after-8 isn't a single-layer test.
    // But we can run layer 4 with input=after-4 (Python's layer 3 output) and
    // see what comes out. There's no single-layer-4 reference unfortunately.
    println!("\n(layer 4+ isolation skipped — fixtures jump 4→8, would need layer-4-only ref)");

    Ok(())
}
