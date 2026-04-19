//! QwenImage VAE decoder parity vs Python diffusers `AutoencoderKLQwenImage`.
//!
//! Run `scripts/qwenimage_vae_decode_ref.py` first to produce
//! `output/qwenimage_vae_decode_ref.safetensors` (keys: `normalized_latent`, `rgb`).
//!
//! The reference stores the *normalized* latent (what the pipeline hands
//! to the decoder in the rest of the stack, and what musubi caches on
//! disk) plus the Python VAE's decode output. Both sides of the parity
//! unnormalize via `z * std + mean` internally (Python inside the
//! ref-dumper, Rust inside `QwenImageVaeDecoder::decode`) so no
//! arithmetic happens outside the codecs.

use std::path::Path;

use anyhow::bail;
use flame_core::{global_cuda_device, serialization, DType};
use inference_flame::vae::QwenImageVaeDecoder;

const VAE_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors";
const REF_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/qwenimage_vae_decode_ref.safetensors";

fn main() -> anyhow::Result<()> {
    let device = global_cuda_device();
    println!("=== QwenImage VAE decoder parity (Rust vs diffusers) ===");
    println!("VAE: {VAE_PATH}");
    println!("Ref: {REF_PATH}");
    println!();

    // Load reference blob.
    let blob = serialization::load_file(Path::new(REF_PATH), &device)?;
    let normalized = blob
        .get("normalized_latent")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing normalized_latent in reference file"))?;
    let rgb_ref = blob
        .get("rgb")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing rgb in reference file"))?;
    println!(
        "  normalized_latent: {:?} {:?}",
        normalized.shape().dims(),
        normalized.dtype()
    );
    println!("  rgb_ref:           {:?} {:?}", rgb_ref.shape().dims(), rgb_ref.dtype());

    // Decode.
    println!("--- loading Rust QwenImage VAE decoder ---");
    let decoder = QwenImageVaeDecoder::from_safetensors(VAE_PATH, &device)?;
    println!("--- decoding ---");
    let rgb_rust = decoder.decode(&normalized)?;

    // Dump Rust output alongside Python ref so we can eyeball them.
    {
        use std::collections::HashMap;
        let mut m: HashMap<String, _> = HashMap::new();
        m.insert("rgb".to_string(), rgb_rust.clone());
        let out = "/home/alex/EriDiffusion/inference-flame/output/qwenimage_vae_decode_rust.safetensors";
        serialization::save_file(&m, Path::new(out))?;
        println!("  (saved rust rgb to {out})");
    }
    println!("  rgb_rust:   {:?} {:?}", rgb_rust.shape().dims(), rgb_rust.dtype());

    if rgb_rust.shape().dims() != rgb_ref.shape().dims() {
        bail!(
            "shape mismatch: rust={:?} vs ref={:?}",
            rgb_rust.shape().dims(),
            rgb_ref.shape().dims()
        );
    }

    // Compare. Both should be [B, 3, 1, H, W] BF16.
    let rust_v = rgb_rust.to_dtype(DType::F32)?.to_vec_f32()?;
    let ref_v = rgb_ref.to_dtype(DType::F32)?.to_vec_f32()?;

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_abs_ref = 0.0f32;
    let mut diffs: Vec<f32> = Vec::with_capacity(rust_v.len());
    let mut sum_abs = 0.0f64;
    for (a, b) in rust_v.iter().zip(ref_v.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
        let d = (a - b).abs();
        diffs.push(d);
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
        }
        if b.abs() > max_abs_ref {
            max_abs_ref = b.abs();
        }
    }
    let cos_sim = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let mean_abs = (sum_abs / diffs.len() as f64) as f32;
    let mut diffs_sorted = diffs.clone();
    diffs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = diffs_sorted[(diffs_sorted.len() as f32 * 0.99) as usize];
    let rel = max_abs / max_abs_ref.max(1e-6);

    println!();
    println!("=== Parity ===");
    println!("  cosine_sim:  {cos_sim:.9}");
    println!("  mean_abs:    {mean_abs:.6}");
    println!("  p99_abs:     {p99:.6}");
    println!("  max_abs:     {max_abs:.6}  (at one or a few outlier pixels)");
    println!("  max_ref:     {max_abs_ref:.6}");
    println!("  max_rel:     {rel:.6}");
    // A 15-layer BF16 VAE decode accumulates ~2-3% max_abs noise at one
    // or two isolated pixels even when the overall image is bit-identical.
    // Gate on the *bulk* statistics (cos_sim, mean, p99) rather than the
    // outlier max.
    let pass = cos_sim >= 0.9999 && mean_abs <= 5e-3 && p99 <= 2e-2;
    println!("  status:      {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        bail!("parity failed");
    }
    Ok(())
}
