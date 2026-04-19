//! Per-substep parity for Qwen2.5-VL layer 0 vs diffusers.
//!
//! Run `scripts/qwen25vl_layer0_substep_dump.py` first to populate
//! `output/qwen25vl_layer0_substep.safetensors`. This bin loads the
//! same token_ids, calls `Qwen25VLEncoder::layer0_substep_probe`, and
//! reports cos_sim / mean / p99 / max at each substep — in execution
//! order — so the first divergence is obvious.
//!
//! The whole layer 0 is known to be wrong (layer_0_out cos_sim = 0.19
//! as of 2026-04-19). This bin is how we find which substep is the
//! root cause.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use flame_core::{global_cuda_device, serialization, DType, Tensor};
use inference_flame::models::qwen25vl_encoder::Qwen25VLEncoder;

const SNAP: &str = "/home/alex/.serenity/models/checkpoints/qwen-image-2512";
const REF: &str = "/home/alex/EriDiffusion/inference-flame/output/qwen25vl_layer0_substep.safetensors";

/// Ordered list of keys to diff — execution order within layer 0.
const ORDER: &[&str] = &[
    "embed_out",
    "normed_input",
    "q_raw",
    "k_raw",
    "v_raw",
    "q_heads",
    "k_heads",
    "v_heads",
    "q_roped",
    "k_roped",
    "k_repeated",
    "v_repeated",
    "attn_sdpa",
    "attn_merge",
    "attn_o_out",
    "after_attn",
    "normed_post",
    "gate_raw",
    "up_raw",
    "mlp_pre_down",
    "mlp_out",
    "layer_0_out",
];

fn metrics(a: &Tensor, b: &Tensor, label: &str) -> anyhow::Result<()> {
    if a.shape().dims() != b.shape().dims() {
        println!(
            "  {label:<16}  SHAPE MISMATCH  rust={:?}  ref={:?}",
            a.shape().dims(),
            b.shape().dims()
        );
        return Ok(());
    }
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut diffs: Vec<f32> = Vec::with_capacity(av.len());
    for (x, y) in av.iter().zip(bv.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
        let d = (x - y).abs();
        diffs.push(d);
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let mean = (sum_abs / diffs.len() as f64) as f32;
    diffs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p99 = diffs[(diffs.len() as f32 * 0.99) as usize];
    println!(
        "  {label:<16}  cos_sim={cos:.6}  mean={mean:.5}  p99={p99:.4}  max={max_abs:.4}  shape={:?}",
        a.shape().dims()
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = global_cuda_device();
    println!("=== Qwen2.5-VL layer-0 substep parity ===");

    let dump = serialization::load_file(Path::new(REF), &device)?;
    let token_ids_t = dump
        .get("token_ids")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing token_ids in {REF}"))?;
    let token_ids_f32 = token_ids_t.to_dtype(DType::F32)?.to_vec_f32()?;
    let token_ids: Vec<i32> = token_ids_f32.into_iter().map(|f| f as i32).collect();
    println!("  seq_len = {}", token_ids.len());

    println!("--- Loading text encoder shards ---");
    let t_load = Instant::now();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for i in 1..=4 {
        let shard = format!("{SNAP}/text_encoder/model-0000{i}-of-00004.safetensors");
        let w = flame_core::serialization::load_file_filtered(
            Path::new(&shard),
            &device,
            |k| !k.starts_with("visual."),
        )?;
        for (k, v) in w {
            weights.insert(k, v);
        }
    }
    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in keys {
        let t = &weights[&k];
        if t.dtype() != DType::BF16 {
            let bf = t.to_dtype(DType::BF16)?;
            weights.insert(k, bf);
        }
    }
    println!(
        "  loaded {} tensors in {:.1}s",
        weights.len(),
        t_load.elapsed().as_secs_f32()
    );

    let mut cfg = Qwen25VLEncoder::config_from_weights(&weights)?;
    cfg.max_seq_len = token_ids.len();
    println!(
        "  cfg: layers={} hidden={} heads={}/{} head_dim={} rms_eps={} rope_theta={}",
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
        cfg.rope_theta,
    );

    let encoder = Qwen25VLEncoder::new(weights, cfg, device.clone());

    println!("--- Running Rust layer-0 probe ---");
    let t_probe = Instant::now();
    let probe = encoder.layer0_substep_probe(&token_ids)?;
    println!("  probe in {:.2}s", t_probe.elapsed().as_secs_f32());
    println!();

    println!("--- Per-substep parity (execution order) ---");
    for key in ORDER {
        let a = match probe.get(*key) {
            Some(t) => t,
            None => {
                println!("  {key:<16}  [MISSING in Rust probe]");
                continue;
            }
        };
        let b = match dump.get(*key) {
            Some(t) => t,
            None => {
                println!("  {key:<16}  [MISSING in Python dump]");
                continue;
            }
        };
        metrics(a, b, key)?;
    }

    Ok(())
}
