//! Layer-level parity for the Rust Qwen2.5-VL text encoder vs diffusers.
//!
//! Run `scripts/qwen25vl_layer_dump.py` first to generate
//! `output/qwen25vl_layer_dump.safetensors` with the reference
//! embedding, layer-0, layer-1, layer-27, and final-hidden tensors plus
//! the full padded token_ids.
//!
//! This bin loads the same token_ids, runs the Rust encoder, and
//! reports cos_sim / mean / p99 / max at each checkpoint (currently
//! just the final hidden — layer-by-layer probes commented in for
//! when the end-to-end fails).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use flame_core::{global_cuda_device, serialization, DType, Tensor};
use inference_flame::models::qwen25vl_encoder::Qwen25VLEncoder;

const SNAP: &str = "/home/alex/.serenity/models/checkpoints/qwen-image-2512";
const REF: &str = "/home/alex/EriDiffusion/inference-flame/output/qwen25vl_layer_dump.safetensors";

fn metrics(a: &Tensor, b: &Tensor, label: &str) -> anyhow::Result<()> {
    if a.shape().dims() != b.shape().dims() {
        anyhow::bail!(
            "{label}: shape mismatch rust={:?} ref={:?}",
            a.shape().dims(),
            b.shape().dims()
        );
    }
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut diffs: Vec<f32> = Vec::with_capacity(av.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
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
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = diffs[(diffs.len() as f32 * 0.99) as usize];
    println!(
        "  {label:<16}  cos_sim={cos:.9}  mean={mean:.4}  p99={p99:.4}  max={max_abs:.4}"
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = global_cuda_device();
    println!("=== Qwen2.5-VL parity ===");

    // Load reference dump.
    let dump = serialization::load_file(Path::new(REF), &device)?;
    let token_ids_t = dump
        .get("token_ids")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing token_ids"))?;
    let embed_ref = dump
        .get("embed_out")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing embed_out"))?;
    let final_ref = dump
        .get("final_hidden")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing final_hidden"))?;

    let token_ids_f32 = token_ids_t.to_dtype(DType::F32)?.to_vec_f32()?;
    let token_ids: Vec<i32> = token_ids_f32.into_iter().map(|f| f as i32).collect();
    println!("  seq_len = {}", token_ids.len());
    println!();

    // Load encoder.
    println!("--- Loading encoder shards ---");
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
    println!("  loaded {} tensors in {:.1}s", weights.len(), t_load.elapsed().as_secs_f32());

    let mut cfg = Qwen25VLEncoder::config_from_weights(&weights)?;
    cfg.max_seq_len = token_ids.len();
    println!("  cfg: layers={} hidden={} heads={}/{} head_dim={}",
        cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);

    let encoder = Qwen25VLEncoder::new(weights, cfg, device.clone());
    println!();

    // Run the full forward capturing intermediates.
    println!("--- Running Rust forward (with intermediates) ---");
    let t_enc = Instant::now();
    let (embed, layers, final_hidden) = encoder.encode_with_intermediates(&token_ids)?;
    println!(
        "  forward in {:.1}s  final={:?}  num_layers={}",
        t_enc.elapsed().as_secs_f32(),
        final_hidden.shape().dims(),
        layers.len()
    );
    println!();

    // Parity at each checkpoint.
    println!("--- Parity vs diffusers ---");
    metrics(&embed, &embed_ref, "embed")?;
    let layer0_ref = dump
        .get("layer_0_out")
        .ok_or_else(|| anyhow::anyhow!("missing layer_0_out"))?;
    let layer1_ref = dump
        .get("layer_1_out")
        .ok_or_else(|| anyhow::anyhow!("missing layer_1_out"))?;
    let layer27_ref = dump
        .get("layer_27_out")
        .ok_or_else(|| anyhow::anyhow!("missing layer_27_out"))?;
    metrics(&layers[0], layer0_ref, "layer_0")?;
    metrics(&layers[1], layer1_ref, "layer_1")?;
    metrics(&layers[27], layer27_ref, "layer_27")?;
    metrics(&final_hidden, &final_ref, "final_hidden")?;

    // Real-token-only parity: downstream diffusion uses positions < real_len,
    // so full-seq cos_sim gets dominated by masked pad-position junk that
    // differs between impls but never feeds forward. Pull a narrow window.
    let pad_id = 151643i32;
    let real_len = token_ids.iter().position(|&id| id == pad_id).unwrap_or(token_ids.len());
    let hidden = 3584usize;
    let slice_elems = real_len * hidden;

    let slice = |t: &Tensor| -> anyhow::Result<Tensor> {
        let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
        let real_vals = v[..slice_elems].to_vec();
        Ok(Tensor::from_vec(
            real_vals,
            flame_core::Shape::from_dims(&[1, real_len, hidden]),
            device.clone(),
        )?)
    };

    println!();
    println!("--- Real-token-only parity (first {real_len} positions) ---");
    metrics(&slice(&layers[0])?, &slice(layer0_ref)?, "layer_0_real")?;
    metrics(&slice(&layers[27])?, &slice(layer27_ref)?, "layer_27_real")?;
    metrics(&slice(&final_hidden)?, &slice(&final_ref)?, "final_real")?;

    Ok(())
}
