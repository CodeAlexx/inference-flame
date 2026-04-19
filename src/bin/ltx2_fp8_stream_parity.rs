//! Parity test for the FP8-stream BlockOffloader path vs the BF16 streaming
//! reference.
//!
//! For each target key in `transformer_blocks.10` (an FP8-carrying block in
//! the distilled-fp8 file), we load it twice:
//!   * Path A: from the BF16 distilled safetensors via the existing
//!     `BlockOffloader::load` path (no FP8 on disk, identity load).
//!   * Path B: from the FP8 distilled safetensors via
//!     `BlockOffloader::load_fp8_stream`, which pins raw FP8 bytes on host
//!     and GPU-dequants to BF16 with the metadata `weight_scale` sidecar.
//!
//! Both paths apply the same `prepare_weights` transpose, so the BF16
//! tensors we compare are what the model forward actually sees.
//!
//! Expected numerics (per the numpy prototype in the plan):
//!   * cos_sim ≥ 0.999 per key (typical 0.99965)
//!   * max |diff| ≤ 0.10 (FP8 E4M3 has 3 mantissa bits → ~2.2% relative
//!     error; max diff scales with `weight_scale`, typically 0.02–0.05).
//!
//! PASS: every key in block 10 meets both thresholds.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use flame_core::{global_cuda_device, DType, Tensor};
use flame_diffusion::block_offload::BlockFacilitator;
use flame_diffusion::BlockOffloader;

const BF16_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const FP8_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
const KEY_PREFIX: &str = "model.diffusion_model.";
const TARGET_BLOCK: usize = 10;
const COS_FLOOR: f64 = 0.999;
const MAX_ABS_CEIL: f32 = 0.10;

fn cos_sim_max_abs(a: &Tensor, b: &Tensor) -> anyhow::Result<(f64, f32)> {
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    anyhow::ensure!(av.len() == bv.len(), "shape mismatch {} vs {}", av.len(), bv.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x, *y);
        dot += (x as f64) * (y as f64);
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        let d = (x - y).abs();
        if d > max_abs { max_abs = d; }
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    Ok((dot / denom, max_abs))
}

/// Facilitator that routes ONE specific block — we skip everything else so
/// the BlockOffloader doesn't pin the whole 22B checkpoint for a parity
/// check. `num_layers` is set to `TARGET_BLOCK + 1` so index 10 validates.
struct OneBlockFac {
    num_layers: usize,
    target: usize,
}

impl BlockFacilitator for OneBlockFac {
    fn block_count(&self) -> usize { self.num_layers }
    fn classify_key(&self, name: &str) -> Option<usize> {
        let stripped = name.strip_prefix(KEY_PREFIX).unwrap_or(name);
        if !stripped.starts_with("transformer_blocks.") { return None; }
        if stripped.contains("scale_shift_table") { return None; }
        if stripped.ends_with(".weight_scale") || stripped.ends_with(".input_scale") {
            return None;
        }
        let rest = stripped.strip_prefix("transformer_blocks.")?;
        let idx: usize = rest.split('.').next()?.parse().ok()?;
        if idx == self.target { Some(idx) } else { None }
    }
}

fn load_block(
    path: &str,
    target_block: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
    fp8_stream: bool,
) -> anyhow::Result<Arc<HashMap<String, Tensor>>> {
    let fac = OneBlockFac { num_layers: target_block + 1, target: target_block };
    let mut off = if fp8_stream {
        BlockOffloader::load_fp8_stream(&[path], &fac, device.clone())?
    } else {
        BlockOffloader::load(&[path], &fac, device.clone())?
    };
    let arc = off.ensure_block(target_block)
        .map_err(|e| anyhow::anyhow!("ensure_block({target_block}): {e}"))?;
    // Force the device sync so the GPU dequant kernel (FP8 path) has definitely
    // completed before we readback.
    device.synchronize()
        .map_err(|e| anyhow::anyhow!("device sync: {e:?}"))?;
    Ok(arc)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("============================================================");
    println!("LTX-2.3 FP8-stream BlockOffloader parity (block {TARGET_BLOCK})");
    println!("============================================================");
    println!("  BF16 source: {BF16_PATH}");
    println!("  FP8  source: {FP8_PATH}");
    println!("  Thresholds:  cos_sim >= {COS_FLOOR:.4}, max_abs <= {MAX_ABS_CEIL:.3}");
    println!();

    // Sanity: both files must exist before we spin up pinned RAM.
    for p in [BF16_PATH, FP8_PATH] {
        anyhow::ensure!(Path::new(p).exists(), "missing file: {p}");
    }

    let device = global_cuda_device();

    println!("[bf16] Loading block {TARGET_BLOCK} via BlockOffloader::load...");
    let t0 = std::time::Instant::now();
    let bf = load_block(BF16_PATH, TARGET_BLOCK, &device, false)?;
    println!("[bf16] Done in {:.1}s, {} tensors", t0.elapsed().as_secs_f32(), bf.len());

    println!("[fp8 ] Loading block {TARGET_BLOCK} via BlockOffloader::load_fp8_stream...");
    let t1 = std::time::Instant::now();
    let fp = load_block(FP8_PATH, TARGET_BLOCK, &device, true)?;
    println!("[fp8 ] Done in {:.1}s, {} tensors", t1.elapsed().as_secs_f32(), fp.len());
    println!();

    // The two loaders may see different auxiliary tensors (biases stay BF16
    // in both files, but the FP8 file may emit biases as BF16 too). Take the
    // intersection of keys — anything in one but not the other we flag but
    // don't fail on (the model forward sees the union of both, and missing
    // tensors would fail elsewhere).
    let bf_keys: std::collections::BTreeSet<_> = bf.keys().cloned().collect();
    let fp_keys: std::collections::BTreeSet<_> = fp.keys().cloned().collect();
    let common: Vec<_> = bf_keys.intersection(&fp_keys).cloned().collect();
    let only_bf: Vec<_> = bf_keys.difference(&fp_keys).cloned().collect();
    let only_fp: Vec<_> = fp_keys.difference(&bf_keys).cloned().collect();

    if !only_bf.is_empty() {
        println!("  Keys only in BF16 ({}):", only_bf.len());
        for k in only_bf.iter().take(5) { println!("    - {k}"); }
        if only_bf.len() > 5 { println!("    ... ({} more)", only_bf.len() - 5); }
    }
    if !only_fp.is_empty() {
        println!("  Keys only in FP8 ({}):", only_fp.len());
        for k in only_fp.iter().take(5) { println!("    - {k}"); }
        if only_fp.len() > 5 { println!("    ... ({} more)", only_fp.len() - 5); }
    }

    println!("  Common keys: {}", common.len());
    println!();
    println!("  {:<65}  {:>8}  {:>10}  {:>10}", "key", "shape", "cos_sim", "max_abs");
    println!("  {}", "-".repeat(97));

    let mut failed_cos: Vec<(String, f64)> = Vec::new();
    let mut failed_mag: Vec<(String, f32)> = Vec::new();
    let mut worst_cos = 1.0f64;
    let mut worst_max_abs = 0.0f32;

    for key in &common {
        let a = &bf[key];
        let b = &fp[key];
        if a.shape().dims() != b.shape().dims() {
            println!("  {:<65}  SHAPE MISMATCH bf={:?} fp={:?}", key, a.shape().dims(), b.shape().dims());
            failed_cos.push((key.clone(), 0.0));
            continue;
        }
        let (cos, max_abs) = cos_sim_max_abs(a, b)?;
        let shape_s = format!("{:?}", a.shape().dims());
        let tag = if cos < COS_FLOOR || max_abs > MAX_ABS_CEIL { "  FAIL" } else { "" };
        println!("  {:<65}  {:>8}  {:>10.6}  {:>10.4}{}", truncate(key, 65), shape_s, cos, max_abs, tag);
        worst_cos = worst_cos.min(cos);
        worst_max_abs = worst_max_abs.max(max_abs);
        if cos < COS_FLOOR { failed_cos.push((key.clone(), cos)); }
        if max_abs > MAX_ABS_CEIL { failed_mag.push((key.clone(), max_abs)); }
    }

    println!();
    println!("  Worst cos_sim : {worst_cos:.6}");
    println!("  Worst max_abs : {worst_max_abs:.4}");
    println!("  Failed cos_sim: {}", failed_cos.len());
    println!("  Failed max_abs: {}", failed_mag.len());

    if failed_cos.is_empty() && failed_mag.is_empty() {
        println!();
        println!("PASS: all {} common tensors meet cos>={:.4}, max_abs<={:.3}",
            common.len(), COS_FLOOR, MAX_ABS_CEIL);
        Ok(())
    } else {
        for (k, v) in &failed_cos { eprintln!("  cos FAIL {k}: {v:.6}"); }
        for (k, v) in &failed_mag { eprintln!("  mag FAIL {k}: {v:.4}"); }
        anyhow::bail!("FP8-stream parity FAILED: {} cos / {} mag breaches",
            failed_cos.len(), failed_mag.len())
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { return s.to_string(); }
    let mut t = s[s.len() - n..].to_string();
    t.replace_range(..3, "...");
    t
}
