//! Parity test for LTX-2 guidance helpers.
//!
//! Loads the reference tensors from Lightricks's reference scripts (run
//! `python scripts/ltx2_stg_mask_ref.py`, `ltx2_cfg_star_ref.py`,
//! `ltx2_stg_rescale_ref.py` first) and compares the Rust implementations
//! element-wise.  Pass criterion: max_abs_diff below
//!   - cfg_star / stg_rescale: 1e-3 on BF16 output, 1e-4 on F32 output
//!   - skip mask: exact (0/1 only)

use flame_core::{global_cuda_device, serialization, DType, Shape, Tensor};
use inference_flame::sampling::ltx2_guidance::{
    build_skip_layer_mask, cfg_star_rescale, stg_rescale,
};
use std::collections::HashMap;
use std::path::Path;

const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

fn load_refs(name: &str, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> anyhow::Result<HashMap<String, Tensor>>
{
    let path = format!("{OUTPUT_DIR}/{name}");
    let p = Path::new(&path);
    if !p.exists() {
        return Err(anyhow::anyhow!(
            "Missing reference file {path}. Run scripts/{name}.py first.",
        ));
    }
    let t = serialization::load_file(p, device)?;
    Ok(t)
}

fn max_abs_diff_f32(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a32 = a.to_dtype(DType::F32)?;
    let b32 = b.to_dtype(DType::F32)?;
    let av = a32.to_vec()?;
    let bv = b32.to_vec()?;
    if av.len() != bv.len() {
        return Err(anyhow::anyhow!(
            "length mismatch: {} vs {}", av.len(), bv.len()));
    }
    let mut m = 0.0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > m { m = d; }
    }
    Ok(m)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("============================================================");
    println!("LTX-2 guidance parity: CFG-star + STG mask + STG std-rescale");
    println!("============================================================");

    let device = global_cuda_device();
    let mut all_pass = true;

    // --- 1. Skip mask ---------------------------------------------------
    println!("\n[1/3] STG skip-layer mask");
    let masks = load_refs("ltx2_stg_mask_ref.safetensors", &device)?;
    // Large: 48 layers, batch=1, num_conds=3, skip=[11,25,35,39], ptb_index=2
    let rust_large = build_skip_layer_mask(48, 1, 3, &[11, 25, 35, 39], 2);
    let rust_large_flat: Vec<f32> = rust_large.into_iter().flatten().collect();
    let rust_large_tensor = Tensor::from_vec(
        rust_large_flat.clone(),
        Shape::from_dims(&[48, 3]),
        device.clone(),
    )?;
    let d_large = max_abs_diff_f32(&rust_large_tensor, masks.get("mask_large")
        .ok_or_else(|| anyhow::anyhow!("mask_large missing"))?)?;
    println!("  large (48x3): max_abs_diff = {d_large:.6}");
    if d_large > 0.0 {
        println!("  FAIL: mask should be exact 0/1");
        all_pass = false;
    }

    // Small: 4 layers, batch=1, num_conds=3, skip=[1,3], ptb_index=2
    let rust_small = build_skip_layer_mask(4, 1, 3, &[1, 3], 2);
    let rust_small_flat: Vec<f32> = rust_small.into_iter().flatten().collect();
    let rust_small_tensor = Tensor::from_vec(
        rust_small_flat,
        Shape::from_dims(&[4, 3]),
        device.clone(),
    )?;
    let d_small = max_abs_diff_f32(&rust_small_tensor, masks.get("mask_small")
        .ok_or_else(|| anyhow::anyhow!("mask_small missing"))?)?;
    println!("  small (4x3):  max_abs_diff = {d_small:.6}");
    if d_small > 0.0 { all_pass = false; }

    // --- 2. CFG-star rescale --------------------------------------------
    println!("\n[2/3] CFG-star rescale");
    let cfg = load_refs("ltx2_cfg_star_ref.safetensors", &device)?;
    let eps_text = cfg.get("eps_text")
        .ok_or_else(|| anyhow::anyhow!("eps_text missing"))?;
    let eps_uncond = cfg.get("eps_uncond")
        .ok_or_else(|| anyhow::anyhow!("eps_uncond missing"))?;
    let ref_rescaled_bf16 = cfg.get("rescaled_bf16")
        .ok_or_else(|| anyhow::anyhow!("rescaled_bf16 missing"))?;
    let ref_rescaled_f32 = cfg.get("rescaled_f32")
        .ok_or_else(|| anyhow::anyhow!("rescaled_f32 missing"))?;

    let rust_rescaled = cfg_star_rescale(eps_text, eps_uncond)?;
    let d_bf16 = max_abs_diff_f32(&rust_rescaled, ref_rescaled_bf16)?;
    let d_f32 = max_abs_diff_f32(&rust_rescaled, ref_rescaled_f32)?;
    println!("  small (BF16 in): max_abs_diff vs ref bf16 = {d_bf16:.6}");
    println!("                   max_abs_diff vs ref f32  = {d_f32:.6}");
    if d_bf16 > 1e-3 { println!("  FAIL: BF16 parity > 1e-3"); all_pass = false; }

    let eps_text_big = cfg.get("eps_text_big")
        .ok_or_else(|| anyhow::anyhow!("eps_text_big missing"))?;
    let eps_uncond_big = cfg.get("eps_uncond_big")
        .ok_or_else(|| anyhow::anyhow!("eps_uncond_big missing"))?;
    let ref_big_bf16 = cfg.get("rescaled_big_bf16")
        .ok_or_else(|| anyhow::anyhow!("rescaled_big_bf16 missing"))?;
    let rust_big = cfg_star_rescale(eps_text_big, eps_uncond_big)?;
    let d_big = max_abs_diff_f32(&rust_big, ref_big_bf16)?;
    println!("  big (BF16 in):   max_abs_diff vs ref bf16 = {d_big:.6}");
    if d_big > 1e-3 { println!("  FAIL: BF16 parity > 1e-3"); all_pass = false; }

    // --- 3. STG std-rescale ---------------------------------------------
    println!("\n[3/3] STG std-rescale");
    let rescale_refs = load_refs("ltx2_stg_rescale_ref.safetensors", &device)?;
    let pos = rescale_refs.get("pos")
        .ok_or_else(|| anyhow::anyhow!("pos missing"))?;
    let guided = rescale_refs.get("guided")
        .ok_or_else(|| anyhow::anyhow!("guided missing"))?;
    let ref_out_bf16 = rescale_refs.get("out_bf16")
        .ok_or_else(|| anyhow::anyhow!("out_bf16 missing"))?;

    let rust_out = stg_rescale(pos, guided, 0.7)?;
    let d_small_rescale = max_abs_diff_f32(&rust_out, ref_out_bf16)?;
    println!("  small (BF16 in): max_abs_diff vs ref bf16 = {d_small_rescale:.6}");
    if d_small_rescale > 1e-3 { println!("  FAIL: > 1e-3"); all_pass = false; }

    let pos_big = rescale_refs.get("pos_big")
        .ok_or_else(|| anyhow::anyhow!("pos_big missing"))?;
    let guided_big = rescale_refs.get("guided_big")
        .ok_or_else(|| anyhow::anyhow!("guided_big missing"))?;
    let ref_big_bf16 = rescale_refs.get("out_big_bf16")
        .ok_or_else(|| anyhow::anyhow!("out_big_bf16 missing"))?;
    let rust_big_rescale = stg_rescale(pos_big, guided_big, 0.7)?;
    let d_big_rescale = max_abs_diff_f32(&rust_big_rescale, ref_big_bf16)?;
    println!("  big (BF16 in):   max_abs_diff vs ref bf16 = {d_big_rescale:.6}");
    if d_big_rescale > 1e-3 { println!("  FAIL: > 1e-3"); all_pass = false; }

    println!("\n============================================================");
    if all_pass {
        println!("ALL PASS");
        Ok(())
    } else {
        Err(anyhow::anyhow!("parity mismatch"))
    }
}
