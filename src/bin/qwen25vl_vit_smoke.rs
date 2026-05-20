//! Smoke test for the Qwen2.5-VL Vision Tower port.
//!
//! 1. Weight-load smoke: filter `vit_model.*` from the Lance 3B Video
//!    checkpoint, count tensors, assert no missing keys vs the expected set.
//! 2. Forward shape-check: synthetic `[16, 1176]` BF16 input + grid_thw
//!    `[[1, 4, 4]]` → expect output shape `[4, 2048]` and no NaN/Inf.

use std::path::PathBuf;
use std::sync::Arc;

use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use inference_flame::models::qwen25vl_vit::{Qwen25VLVisionTower, Qwen25VLVitConfig};

fn expected_keys(cfg: &Qwen25VLVitConfig) -> Vec<String> {
    let mut keys = Vec::new();
    keys.push("vit_model.patch_embed.proj.weight".to_string());
    for i in 0..cfg.num_layers {
        let p = format!("vit_model.blocks.{i}");
        keys.push(format!("{p}.norm1.weight"));
        keys.push(format!("{p}.norm2.weight"));
        keys.push(format!("{p}.attn.qkv.weight"));
        keys.push(format!("{p}.attn.qkv.bias"));
        keys.push(format!("{p}.attn.proj.weight"));
        keys.push(format!("{p}.attn.proj.bias"));
        keys.push(format!("{p}.mlp.gate_proj.weight"));
        keys.push(format!("{p}.mlp.gate_proj.bias"));
        keys.push(format!("{p}.mlp.up_proj.weight"));
        keys.push(format!("{p}.mlp.up_proj.bias"));
        keys.push(format!("{p}.mlp.down_proj.weight"));
        keys.push(format!("{p}.mlp.down_proj.bias"));
    }
    keys.push("vit_model.merger.ln_q.weight".into());
    keys.push("vit_model.merger.mlp.0.weight".into());
    keys.push("vit_model.merger.mlp.0.bias".into());
    keys.push("vit_model.merger.mlp.2.weight".into());
    keys.push("vit_model.merger.mlp.2.bias".into());
    keys
}

fn main() -> Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let device = CudaDevice::new(0).map_err(|e| {
        flame_core::Error::Cuda(format!("CudaDevice::new(0) failed: {e:?}"))
    })?;
    let device = Arc::new(device);

    let ckpt = PathBuf::from("/home/alex/.serenity/models/lance/Lance_3B_Video/model.safetensors");
    println!("=== qwen25vl_vit_smoke ===");
    println!("ckpt: {}", ckpt.display());

    // (1) Weight-load smoke
    let cfg = Qwen25VLVitConfig::default();
    let expected = expected_keys(&cfg);
    println!("Expected vit weight keys: {}", expected.len());

    // Use filtered raw load first so we can introspect what was on disk
    // *before* the tower removes the 5D patch_embed key.
    let raw = flame_core::serialization::load_file_filtered(&ckpt, &device, |k| {
        k.starts_with("vit_model.")
    })?;
    println!("vit_model.* tensors loaded from disk: {}", raw.len());

    // Check no unexpected keys, no missing keys.
    let mut unexpected = Vec::new();
    let expected_set: std::collections::HashSet<&str> =
        expected.iter().map(|s| s.as_str()).collect();
    for k in raw.keys() {
        if !expected_set.contains(k.as_str()) {
            unexpected.push(k.clone());
        }
    }
    let mut missing = Vec::new();
    for k in &expected {
        if !raw.contains_key(k) {
            missing.push(k.clone());
        }
    }
    println!("Unexpected keys: {}", unexpected.len());
    for k in unexpected.iter().take(10) {
        println!("  unexpected: {k}");
    }
    println!("Missing keys: {}", missing.len());
    for k in missing.iter().take(10) {
        println!("  missing: {k}");
    }

    // Print patch_embed weight shape for sanity.
    if let Some(w) = raw.get("vit_model.patch_embed.proj.weight") {
        println!(
            "patch_embed.proj.weight shape: {:?} dtype: {:?}",
            w.shape().dims(),
            w.dtype()
        );
    }

    // Now build the tower from the same raw map (consumes it).
    let tower = Qwen25VLVisionTower::from_weights(raw, cfg.clone(), device.clone())?;
    println!("Tower built. Post-transpose weight count: {}", tower.weights().len());

    // (2) Forward shape-check
    // grid_thw = [[1, 4, 4]] → seq_len = 1*4*4 = 16 patches
    // After 2x2 merge: 16 / 4 = 4 output tokens, out_hidden_size = 2048
    let n_patches = 16usize;
    let flat_dim = cfg.flat_patch_dim(); // 1176
    println!("flat_patch_dim = {flat_dim}");

    // Synthetic input: small-magnitude BF16 noise
    let data: Vec<f32> = (0..n_patches * flat_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let pixel_values = Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[n_patches, flat_dim]),
        device.clone(),
        DType::BF16,
    )?;

    let grid_thw = [[1u32, 4, 4]];
    let out = tower.forward(&pixel_values, &grid_thw)?;
    let out_dims = out.shape().dims().to_vec();
    println!("Output shape: {:?}", out_dims);
    assert_eq!(out_dims, vec![4, 2048], "expected [4, 2048]");

    // NaN / Inf check via host download
    let out_f32 = out.to_dtype(DType::F32)?.to_vec()?;
    let n_nan = out_f32.iter().filter(|v| v.is_nan()).count();
    let n_inf = out_f32.iter().filter(|v| v.is_infinite()).count();
    let mean_abs: f32 = out_f32.iter().map(|v| v.abs()).sum::<f32>() / out_f32.len() as f32;
    let max_abs: f32 = out_f32.iter().fold(0f32, |a, b| a.max(b.abs()));
    println!(
        "Output stats: NaN={n_nan} Inf={n_inf} mean_abs={mean_abs:.4} max_abs={max_abs:.4}"
    );
    assert_eq!(n_nan, 0, "output has NaN entries");
    assert_eq!(n_inf, 0, "output has Inf entries");

    println!("=== SMOKE PASSED ===");
    Ok(())
}
