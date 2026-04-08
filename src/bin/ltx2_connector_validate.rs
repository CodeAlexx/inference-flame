//! Validate the Rust port of `Embeddings1DConnector` against a Python
//! reference fixture.
//!
//! The fixture lives at
//!   `/home/alex/EriDiffusion/inference-flame/output/python_connector_io.safetensors`
//! and contains the inputs and outputs of the *video* and *audio* connector
//! captured from a real Python forward (see `python_save_embeddings.py`).
//!
//! Inputs:
//!   - `video_features`           [1, 1024, 4096]   bf16
//!   - `audio_features`           [1, 1024, 2048]   bf16
//!   - `additive_attention_mask`  [1, 1, 1, 1024]   bf16   (real ≥ -9000)
//! Expected outputs:
//!   - `video_encoding`           [1, 1024, 4096]   bf16
//!   - `audio_encoding`           [1, 1024, 2048]   bf16
//!
//! This binary loads the LTX-2.3 22B distilled checkpoint to pull the
//! `video_embeddings_connector.*` and `audio_embeddings_connector.*` weights,
//! reconstructs both connectors via `load_video_embeddings_connector`, runs
//! them on the Python inputs, and reports per-modality mean / std / max-abs-diff
//! against the Python outputs.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType, Tensor};

use inference_flame::models::ltx2_model::{
    load_video_embeddings_connector, LTX2Config,
};

const CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/output/python_connector_io.safetensors";

fn stats(name: &str, t: &Tensor) -> Result<()> {
    let f32 = t.to_dtype(DType::F32)?;
    let v = f32.to_vec()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| {
        let d = *x as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let std = var.sqrt();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {:<28}  shape={:?}  mean={:+.4}  std={:.4}  min={:+.4}  max={:+.4}",
        name, t.shape().dims(), mean, std, min, max,
    );
    Ok(())
}

fn diff(name: &str, rust: &Tensor, py: &Tensor) -> Result<()> {
    let r = rust.to_dtype(DType::F32)?.to_vec()?;
    let p = py.to_dtype(DType::F32)?.to_vec()?;
    if r.len() != p.len() {
        return Err(anyhow!(
            "{}: rust len {} != python len {}",
            name, r.len(), p.len()
        ));
    }
    let n = r.len() as f64;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut p_norm = 0.0f64;
    for (a, b) in r.iter().zip(p.iter()) {
        let d = (*a as f64) - (*b as f64);
        let ad = d.abs();
        if ad > max_abs { max_abs = ad; }
        sum_abs += ad;
        sum_sq += d * d;
        p_norm += (*b as f64) * (*b as f64);
    }
    let mae = sum_abs / n;
    let rmse = (sum_sq / n).sqrt();
    let rel = (sum_sq / p_norm.max(1e-30)).sqrt();
    println!(
        "  {:<28}  max_abs={:.4e}  mae={:.4e}  rmse={:.4e}  rel_l2={:.4e}",
        name, max_abs, mae, rmse, rel,
    );
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("Loading fixture: {}", FIXTURE);
    let fixture = flame_core::serialization::load_file(FIXTURE, &device)?;
    for k in ["video_features", "audio_features", "additive_attention_mask",
              "video_encoding", "audio_encoding"] {
        if !fixture.contains_key(k) {
            return Err(anyhow!("Fixture missing key '{}'", k));
        }
    }
    let video_features = &fixture["video_features"];
    let audio_features = &fixture["audio_features"];
    let attn_mask = &fixture["additive_attention_mask"];
    let video_target = &fixture["video_encoding"];
    let audio_target = &fixture["audio_encoding"];

    println!("\nFixture stats:");
    stats("video_features", video_features)?;
    stats("audio_features", audio_features)?;
    stats("additive_attention_mask", attn_mask)?;
    stats("video_encoding (target)", video_target)?;
    stats("audio_encoding (target)", audio_target)?;

    println!("\nLoading connector weights from checkpoint...");
    let prefix = "model.diffusion_model.";
    let raw = flame_core::serialization::load_file_filtered(
        CHECKPOINT, &device,
        |k| {
            let stripped = k.strip_prefix(prefix).unwrap_or(k);
            stripped.starts_with("video_embeddings_connector.")
                || stripped.starts_with("audio_embeddings_connector.")
        },
    )?;
    println!("  Loaded {} connector tensors", raw.len());
    let weights: HashMap<String, Tensor> = raw.into_iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix(prefix).unwrap_or(&k).to_string();
            (stripped, v)
        })
        .collect();

    let cfg = LTX2Config::default();
    let video_connector = load_video_embeddings_connector(
        &weights, "video_embeddings_connector", cfg.norm_eps, cfg.rope_theta,
    )?;
    let audio_connector = load_video_embeddings_connector(
        &weights, "audio_embeddings_connector", cfg.norm_eps, cfg.rope_theta,
    )?;

    println!("\n=== VIDEO CONNECTOR ===");
    let video_out = video_connector.forward(video_features, Some(attn_mask))?;
    println!("Rust output:");
    stats("rust_video_encoding", &video_out)?;
    println!("Python target:");
    stats("python_video_encoding", video_target)?;
    println!("Diff:");
    diff("video_encoding", &video_out, video_target)?;

    println!("\n=== AUDIO CONNECTOR ===");
    let audio_out = audio_connector.forward(audio_features, Some(attn_mask))?;
    println!("Rust output:");
    stats("rust_audio_encoding", &audio_out)?;
    println!("Python target:");
    stats("python_audio_encoding", audio_target)?;
    println!("Diff:");
    diff("audio_encoding", &audio_out, audio_target)?;

    println!("\nDONE");
    Ok(())
}
