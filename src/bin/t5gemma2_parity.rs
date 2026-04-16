//! Per-layer parity check of flame-core T5Gemma2Encoder vs Python reference.
//!
//! Reads a Python dump produced by `scripts/t5_parity_dump.py` (fixed token IDs,
//! same attention mask). Runs the Rust encoder with `encode_with_dump` and
//! prints cosine similarity + max-abs-diff per layer.
//!
//! Expected: cos should be very close to 1.0 at the `embed` layer (pure
//! table lookup + sqrt-hidden scale). Any layer where it drops substantially
//! is where the Rust and Python implementations diverge numerically.

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::t5gemma2_encoder::{T5Gemma2Config, T5Gemma2Encoder};

const BASE: &str = "/home/alex/.serenity/models/checkpoints/motif-video-2b";
const PY_DUMP: &str = "/home/alex/serenity/output/t5_python_dump.safetensors";

// Must EXACTLY match the IDs in scripts/t5_parity_dump.py.
// 18 real tokens + 494 pads = 512 total (exercises sliding-window geometry).
fn token_ids_and_mask() -> (Vec<i32>, Vec<i32>) {
    let real: [i32; 18] = [
        2, 236746, 13935, 102202, 55574, 1343, 496, 2135, 529, 116487,
        657, 14711, 236764, 60420, 15408, 236764, 5111, 5776,
    ];
    const LEN: usize = 512;
    let mut ids = Vec::with_capacity(LEN);
    ids.extend_from_slice(&real);
    ids.resize(LEN, 0);
    let mut mask = vec![1i32; 18];
    mask.resize(LEN, 0);
    (ids, mask)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("=== T5Gemma2 Encoder per-layer parity ===");
    println!("  Reference: {PY_DUMP}");
    let py = flame_core::serialization::load_file(std::path::Path::new(PY_DUMP), &device)?;
    println!("  Loaded {} reference tensors", py.len());

    println!("--- Loading flame-core encoder ---");
    let enc_path = format!("{BASE}/text_encoder/model.safetensors");
    let weights = flame_core::serialization::load_file(std::path::Path::new(&enc_path), &device)?;
    let cfg = T5Gemma2Config::default();
    let encoder = T5Gemma2Encoder::new(weights, cfg, device.clone());

    println!("--- Running flame-core encoder ---");
    let (token_ids, mask) = token_ids_and_mask();
    let (_, dump) = encoder.encode_with_dump(&token_ids, &mask)?;
    println!("  Rust dump has {} tensors", dump.len());

    // Compare order: embed, layer_0..33, final_norm (py) vs final (rust)
    let mut labels: Vec<String> = vec!["embed".into()];
    for i in 0..34 {
        labels.push(format!("layer_{i}"));
    }
    labels.push("final_norm".into());

    println!("--- Per-layer parity ---");
    println!("  {:<12} {:>12} {:>12} {:>10} {:>10}", "label", "cos", "max_abs", "rust_max", "py_max");
    for label in &labels {
        let rust_key = if label == "final_norm" { "final" } else { label.as_str() };
        let (Some(py_t), Some(rust_t)) = (py.get(label), dump.get(rust_key)) else {
            println!("  {:<12} (missing: py={} rust={})", label, py.get(label).is_some(), dump.get(rust_key).is_some());
            continue;
        };
        let py_f = py_t.to_dtype(DType::F32)?.to_vec_f32()?;
        let rust_f = rust_t.to_dtype(DType::F32)?.to_vec_f32()?;
        if py_f.len() != rust_f.len() {
            println!("  {:<12} SHAPE MISMATCH: py={} rust={}", label, py_f.len(), rust_f.len());
            continue;
        }
        let cos = cosine(&py_f, &rust_f);
        let mad = max_abs(&py_f, &rust_f);
        let py_max = py_f.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rust_max = rust_f.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let marker = if cos < 0.99 { " ⚠️" } else if cos < 0.9999 { " ~" } else { "" };
        println!("  {:<12} {:>12.6} {:>12.4} {:>10.2} {:>10.2}{}", label, cos, mad, rust_max, py_max, marker);
    }
    Ok(())
}

#[allow(dead_code)]
fn _silence(_: &Tensor) {}
