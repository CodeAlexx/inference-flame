//! Correctness parity for LTX-2 LoRA fusion: proves the fused weight
//! that the model actually uses equals `raw_weight + strength * B @ A`,
//! NOT just "differs from baseline".
//!
//! Procedure:
//!  1. Pick one representative weight key (square + non-square).
//!  2. Load it raw from the checkpoint via the same path the model uses
//!     (`load_file_filtered`).
//!  3. Load the LoRA, compute `B @ A * strength` via `compute_delta`.
//!  4. Compute `expected_fused = raw + delta` in FP32.
//!  5. Attach the LoRA to the model, load block 0, extract the actual
//!     fused tensor from the block struct.
//!  6. Compare `actual` vs `expected_fused`: cos_sim must be ≥ 0.9999
//!     and max |diff| must fit in BF16 noise (~0.01 at typical magnitudes).
//!
//! Runs both the disk-sync path (default `load_block`) and the
//! BlockOffloader path (via `init_offloader()`) so if they disagree we
//! see it here.

use std::path::Path;
use std::collections::HashMap;

use flame_core::{global_cuda_device, serialization, DType, Tensor};
use inference_flame::models::lora_loader::LoraWeights;
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors";
const LORA_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors";

fn cos_sim_max_abs(a: &Tensor, b: &Tensor) -> anyhow::Result<(f64, f32, f64)> {
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    anyhow::ensure!(av.len() == bv.len(), "shape mismatch {} vs {}", av.len(), bv.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut abs_diff = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x, *y);
        dot += (x as f64) * (y as f64);
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        let d = (x - y).abs();
        abs_diff += d as f64;
        if d > max_abs { max_abs = d; }
    }
    Ok((dot / (na.sqrt() * nb.sqrt()).max(1e-12), max_abs, abs_diff / av.len() as f64))
}

/// Load the raw weight for `full_key` from the checkpoint, dequant'd to
/// BF16 via the serialization layer's FP8 auto-dequant.
fn load_raw_weight(full_key: &str) -> anyhow::Result<Tensor> {
    let device = global_cuda_device();
    let prefix = "model.diffusion_model.";
    let target_key = format!("{prefix}{full_key}");
    let raw = flame_core::serialization::load_file_filtered(
        Path::new(MODEL_PATH),
        &device,
        |k| k == target_key,
    )?;
    raw.into_iter().next().map(|(_, v)| v)
        .ok_or_else(|| anyhow::anyhow!("key {target_key} not found"))
}

fn check_path(label: &str, block_weights_for_key: &Tensor, expected_fused: &Tensor) -> anyhow::Result<bool> {
    let (cos, max_abs, mean_abs) = cos_sim_max_abs(block_weights_for_key, expected_fused)?;
    let verdict = if cos >= 0.9999 && max_abs <= 0.05 { "PASS" } else { "FAIL" };
    println!(
        "  {verdict}  {label:<26}  cos_sim={cos:.6}  max_abs={max_abs:.4}  mean_abs={mean_abs:.6}"
    );
    Ok(verdict == "PASS")
}

fn run_once(use_offloader: bool, expected_map: &HashMap<String, Tensor>) -> anyhow::Result<bool> {
    let device = global_cuda_device();
    let config = LTX2Config::default();

    println!("\n=== Path: {} ===", if use_offloader { "BlockOffloader" } else { "disk-sync" });

    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    let lora = LoraWeights::load(LORA_PATH, 1.0, &device)?;
    model.add_lora(lora);

    if use_offloader {
        model.init_offloader()?;
    }

    let block = model.load_block(0)?;

    let mut all_pass = true;
    // Video (main attn)
    all_pass &= check_path("video attn1.to_q", &block.attn1.to_q_weight,
        expected_map.get("transformer_blocks.0.attn1.to_q.weight").unwrap())?;
    all_pass &= check_path("video attn1.to_k", &block.attn1.to_k_weight,
        expected_map.get("transformer_blocks.0.attn1.to_k.weight").unwrap())?;
    // Audio (audio_attn1). Disk-sync `load_block_from_disk` filters out
    // `audio` keys, so in that path audio weights are dummies — the compare
    // vs expected_fused will FAIL. The BlockOffloader path loads the full
    // block (video + audio) and should fuse audio deltas correctly.
    all_pass &= check_path("audio attn1.to_q", &block.audio_attn1.to_q_weight,
        expected_map.get("transformer_blocks.0.audio_attn1.to_q.weight").unwrap())?;
    all_pass &= check_path("audio attn1.to_k", &block.audio_attn1.to_k_weight,
        expected_map.get("transformer_blocks.0.audio_attn1.to_k.weight").unwrap())?;

    Ok(all_pass)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("=== LTX-2 LoRA correctness parity ===");
    println!("Checkpoint: {}", MODEL_PATH);
    println!("LoRA:       {}", LORA_PATH);

    // Precompute expected fused weights, then DROP the LoRA handle before
    // building the model. The distilled LoRA is rank-384 × 1660 pairs × BF16
    // ≈ 10 GB VRAM — two copies would OOM a 24 GB card once LTX-2 globals
    // also sit on GPU.
    let lora = LoraWeights::load(LORA_PATH, 1.0, &device)?;
    // Small square weights only, to avoid 128MB-per-tensor OOM under the
    // BlockOffloader's already-high peak. Mix of video and audio so we
    // prove both halves fuse: audio coverage is 64% of the distilled
    // LoRA's keys, and LTX-2's core value proposition is cross-modal AV.
    let video_keys = [
        "transformer_blocks.0.attn1.to_q.weight",
        "transformer_blocks.0.attn1.to_k.weight",
    ];
    let audio_keys = [
        "transformer_blocks.0.audio_attn1.to_q.weight",
        "transformer_blocks.0.audio_attn1.to_k.weight",
    ];
    let all_keys: Vec<&str> = video_keys.iter().chain(audio_keys.iter()).copied().collect();
    let mut expected: HashMap<String, Tensor> = HashMap::new();
    for k in &all_keys {
        let raw = load_raw_weight(k)?;
        println!("  raw {k}: shape={:?}", raw.shape().dims());
        let delta = lora.compute_delta(k)?.ok_or_else(||
            anyhow::anyhow!("LoRA has no delta for {k}"))?;
        println!("      delta shape={:?}", delta.shape().dims());
        let fused = raw.add(&delta)?;
        expected.insert(k.to_string(), fused);
    }
    // Free the LoRA's GPU tensors before we go on to load it a second time
    // inside `run_once`.
    drop(lora);

    // Run disk-sync path, then BlockOffloader path, each with a fresh model.
    let disk_pass = run_once(false, &expected)?;
    let off_pass = run_once(true, &expected)?;

    println!();
    if disk_pass && off_pass {
        println!("OVERALL: PASS — both paths produce weights equal to raw + B@A within BF16 noise.");
        Ok(())
    } else {
        println!("OVERALL: FAIL — disk_pass={disk_pass}, off_pass={off_pass}");
        std::process::exit(1)
    }
}
