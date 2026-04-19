//! Parity test for `LoraWeights::compute_delta` against a Python
//! reference.
//!
//! Run `scripts/lora_fusion_parity_ref.py` first to populate
//! `output/lora_fusion_ref_ltx2_distilled.safetensors`. This bin loads
//! the same LTX-2 distilled LoRA via `LoraWeights::load`, computes
//! `strength * (B @ A)` for each sample key, and reports cos_sim per
//! key vs the Python `B @ A`.
//!
//! Criterion: cos_sim ≥ 0.999 on every key. The Python ref is FP32;
//! our fused product runs through flame's BF16 matmul, so the residual
//! is expected BF16 rounding.

use std::path::Path;

use flame_core::{global_cuda_device, serialization, DType, Tensor};
use inference_flame::models::lora_loader::LoraWeights;

const LORA: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors";
const REF: &str =
    "/home/alex/EriDiffusion/inference-flame/output/lora_fusion_ref_ltx2_distilled.safetensors";

fn cos_sim(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    if av.len() != bv.len() {
        anyhow::bail!("length mismatch: {} vs {}", av.len(), bv.len());
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    Ok(dot / (na.sqrt() * nb.sqrt()).max(1e-12))
}

fn main() -> anyhow::Result<()> {
    let device = global_cuda_device();
    println!("=== LoRA fusion parity (LTX-2 distilled) ===");

    // Load Python reference
    let refs = serialization::load_file(Path::new(REF), &device)?;
    let strength_t = refs
        .get("__strength__")
        .ok_or_else(|| anyhow::anyhow!("missing __strength__ in ref"))?;
    let strength = strength_t.to_dtype(DType::F32)?.to_vec_f32()?[0];
    println!("  reference strength = {strength:.4}");

    // Load the LoRA via Rust loader
    let lora = LoraWeights::load(LORA, strength, &device)?;
    println!(
        "  loaded {} paired weights, rank={:?}",
        lora.len(),
        lora.rank()
    );
    println!();

    let mut tested = 0usize;
    let mut passed = 0usize;
    let mut worst = ("", 1.0f64);

    for (key, py_delta) in refs.iter() {
        if key.starts_with("__") {
            continue;
        }
        tested += 1;
        let rust_delta = match lora.compute_delta(key)? {
            Some(t) => t,
            None => {
                println!("  {key:<65} [MISSING in Rust LoRA]");
                continue;
            }
        };
        if rust_delta.shape().dims() != py_delta.shape().dims() {
            println!(
                "  {key:<65} SHAPE MISMATCH rust={:?} ref={:?}",
                rust_delta.shape().dims(),
                py_delta.shape().dims()
            );
            continue;
        }
        let c = cos_sim(&rust_delta, py_delta)?;
        let mark = if c >= 0.999 {
            passed += 1;
            "PASS"
        } else {
            "FAIL"
        };
        if c < worst.1 {
            worst = (key.as_str(), c);
        }
        println!(
            "  {mark}  {key:<65} cos_sim={c:.6}  shape={:?}",
            rust_delta.shape().dims()
        );
    }

    println!();
    println!("Result: {passed}/{tested} keys at cos_sim ≥ 0.999");
    if !worst.0.is_empty() {
        println!("Worst key: {}  cos_sim={:.6}", worst.0, worst.1);
    }

    if passed < tested {
        std::process::exit(1);
    }
    Ok(())
}
