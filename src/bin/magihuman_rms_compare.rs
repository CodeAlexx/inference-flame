//! Compare mm_rms_norm_multi (slow F32) vs mm_rms_norm_multi_fused (fast BF16)
//! on identical input + weights. Should be near-bit-equal except for the
//! BF16-input precision floor.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::magihuman_dit::{
    mm_rms_norm_multi, mm_rms_norm_multi_fused, precompute_w_plus_1_bf16_per_modality,
};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}
fn report(label: &str, a: &[f32], b: &[f32]) {
    let max_d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let mean_d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    println!("  {label}: cos={:.6}  max_abs_diff={:.4}  mean_abs_diff={:.4}", cosine(a, b), max_d, mean_d);
}

fn main() -> Result<()> {
    let device = global_cuda_device();
    // Use BlockOffloader to load only layer 2 weights (others get released)
    struct Facilitator;
    impl flame_diffusion::block_offload::BlockFacilitator for Facilitator {
        fn block_count(&self) -> usize { 40 }
        fn classify_key(&self, name: &str) -> Option<usize> {
            name.strip_prefix("block.layers.")?.split('.').next()?.parse().ok()
        }
    }
    let weight_path = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";
    let mut offloader = flame_diffusion::BlockOffloader::load(&[weight_path], &Facilitator, device.clone())
        .map_err(|e| anyhow!("offloader: {e}"))?;
    offloader.prefetch_block(2).map_err(|e| anyhow!("prefetch: {e}"))?;
    let raw = offloader.await_block(2).map_err(|e| anyhow!("await: {e}"))?;
    let w = raw.iter().find(|(k, _)| k.as_str() == "block.layers.2.attention.pre_norm.weight")
        .map(|(_, t)| t.clone())
        .ok_or_else(|| anyhow!("missing pre_norm weight"))?
        .to_dtype(DType::BF16)?;
    println!("weight: shape={:?} dtype={:?}", w.shape().dims(), w.dtype());

    // Build a synthetic input with realistic magnitudes (~max abs 460, like layer 2 input)
    use rand::{SeedableRng, Rng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let l = 40usize;
    let d = 5120usize;
    let total = l * d;
    let mut data: Vec<f32> = (0..total).map(|_| {
        let v: f32 = rng.gen::<f32>() * 2.0 - 1.0;
        v * 50.0  // scale to realistic magnitudes ~50
    }).collect();
    // Inject some big values to test BF16 cliff
    for i in 0..40 {
        data[i * d + 449] = if i < 24 { 256.0 } else if i < 32 { -300.0 } else { 100.0 };
    }
    let input_f32 = Tensor::from_vec(data, Shape::from_dims(&[l, d]), device.clone())?;
    let input_bf16 = input_f32.to_dtype(DType::BF16)?;
    let group_sizes = vec![24usize, 8, 8];

    let w_p1 = precompute_w_plus_1_bf16_per_modality(&w, d)?;

    // Test 1: both with BF16 input (same as old forward path)
    println!("\n=== BF16 input (same as old forward) ===");
    let slow_bf16 = mm_rms_norm_multi(&input_bf16, &w, &group_sizes, 3, 1e-6)?;
    let fast_bf16 = mm_rms_norm_multi_fused(&input_bf16, &w_p1, &group_sizes, 1e-6)?;
    let s = slow_bf16.to_dtype(DType::F32)?.to_vec_f32()?;
    let f = fast_bf16.to_dtype(DType::F32)?.to_vec_f32()?;
    report("slow_BF16in vs fast_BF16in", &s, &f);

    // Test 2: F32 input vs BF16 input through slow path (precision delta from input dtype)
    println!("\n=== Slow path: F32 input vs BF16 input ===");
    let slow_f32 = mm_rms_norm_multi(&input_f32, &w, &group_sizes, 3, 1e-6)?;
    let s_f32_v = slow_f32.to_dtype(DType::F32)?.to_vec_f32()?;
    report("slow_F32in vs slow_BF16in", &s_f32_v, &s);

    // Test 3: Compare F32 input slow path to FAST path (this is what my fix tried to do)
    println!("\n=== F32 slow vs BF16 fast (my fix) ===");
    report("slow_F32in vs fast_BF16in", &s_f32_v, &f);

    // Per-modality breakdown
    println!("\n=== Per-modality (F32-slow vs BF16-fast) ===");
    let v_s = &s_f32_v[0..24*d];
    let v_f = &f[0..24*d];
    let a_s = &s_f32_v[24*d..32*d];
    let a_f = &f[24*d..32*d];
    let t_s = &s_f32_v[32*d..40*d];
    let t_f = &f[32*d..40*d];
    report("V slow_f32 vs fast_bf16", v_s, v_f);
    report("A slow_f32 vs fast_bf16", a_s, a_f);
    report("T slow_f32 vs fast_bf16", t_s, t_f);

    Ok(())
}
