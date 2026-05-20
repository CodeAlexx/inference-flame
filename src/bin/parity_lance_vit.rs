//! Per-block parity for the Lance Qwen2.5-VL ViT against Python refs from
//! `ports/lance/parity/gen_refs_lance_vit.py`.
//!
//! Loads the Python-saved `input.safetensors` (pixel_values + grid_thw) so
//! both sides see byte-identical input, runs the Rust ViT with
//! `forward_capture`, and prints per-block cos + max_abs vs the Python
//! captures.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use flame_core::{DType, Tensor};
use inference_flame::models::qwen25vl_vit::{Qwen25VLVisionTower, Qwen25VLVitConfig};

const CAPTURE_LAYERS: &[usize] = &[0, 7, 15, 23, 31];

struct Args {
    model_dir: PathBuf,
    refs_dir: PathBuf,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            model_dir: PathBuf::from("/home/alex/.serenity/models/lance/Lance_3B_Video"),
            refs_dir: PathBuf::from("ports/lance/parity/refs_vit"),
        };
        let argv: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--model-dir" | "--model_dir" => {
                    a.model_dir = PathBuf::from(&argv[i + 1]);
                    i += 2;
                }
                "--refs-dir" | "--refs_dir" => {
                    a.refs_dir = PathBuf::from(&argv[i + 1]);
                    i += 2;
                }
                "-h" | "--help" => {
                    println!(
                        "parity_lance_vit [--model-dir PATH] [--refs-dir PATH]\n  \
                         default model-dir: {}\n  \
                         default refs-dir:  {}",
                        a.model_dir.display(),
                        a.refs_dir.display()
                    );
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(2);
                }
            }
        }
        a
    }
}

fn cos_sim(a: &Tensor, b: &Tensor) -> Result<(f64, f32, f64)> {
    let a_f32 = a.to_dtype(DType::F32)?.to_vec()?;
    let b_f32 = b.to_dtype(DType::F32)?.to_vec()?;
    if a_f32.len() != b_f32.len() {
        anyhow::bail!("cos_sim len mismatch: {} vs {}", a_f32.len(), b_f32.len());
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    for i in 0..a_f32.len() {
        let x = a_f32[i] as f64;
        let y = b_f32[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (a_f32[i] - b_f32[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_sq += (a_f32[i] - b_f32[i]) as f64 * (a_f32[i] - b_f32[i]) as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-20);
    let rms = (sum_sq / a_f32.len() as f64).sqrt();
    Ok((cos, max_abs, rms))
}

fn main() -> Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let device = flame_core::global_cuda_device();
    let args = Args::parse();

    // ---- Load Python-saved input fixture ----
    let input_path = args.refs_dir.join("input.safetensors");
    println!("Loading input fixture: {}", input_path.display());
    let inputs = flame_core::serialization::load_file(&input_path, &device)
        .with_context(|| format!("load {}", input_path.display()))?;
    let pixel_values = inputs
        .get("input.pixel_values")
        .ok_or_else(|| anyhow::anyhow!("missing input.pixel_values in {}", input_path.display()))?
        .clone();
    let grid_thw_t = inputs
        .get("input.grid_thw")
        .ok_or_else(|| anyhow::anyhow!("missing input.grid_thw"))?
        .clone();

    // Reconstruct `&[[u32; 3]]` from grid_thw tensor (Python writes F32).
    let grid_thw_vec: Vec<f32> = grid_thw_t.to_dtype(DType::F32)?.to_vec()?;
    let nrows = grid_thw_t.shape().dims()[0];
    let mut grid_thw: Vec<[u32; 3]> = Vec::with_capacity(nrows);
    for r in 0..nrows {
        let t = grid_thw_vec[r * 3] as u32;
        let h = grid_thw_vec[r * 3 + 1] as u32;
        let w = grid_thw_vec[r * 3 + 2] as u32;
        grid_thw.push([t, h, w]);
    }

    println!(
        "pixel_values: shape={:?} dtype={:?}",
        pixel_values.shape().dims(),
        pixel_values.dtype()
    );
    println!("grid_thw: {grid_thw:?}");

    // Ensure BF16 (Python wrote BF16).
    let pixel_values = if pixel_values.dtype() != DType::BF16 {
        pixel_values.to_dtype(DType::BF16)?
    } else {
        pixel_values
    };

    // ---- Build Rust ViT ----
    let cfg = Qwen25VLVitConfig::default();
    let ckpt = args.model_dir.join("model.safetensors");
    println!("Loading Rust ViT from {}", ckpt.display());
    let raw = flame_core::serialization::load_file_filtered(&ckpt, &device, |k| {
        k.starts_with("vit_model.")
    })?;
    let tower = Qwen25VLVisionTower::from_weights(raw, cfg.clone(), device.clone())?;
    println!("Rust ViT loaded.");

    // ---- Forward with per-block captures ----
    let (post_merger, captures) = tower.forward_capture(&pixel_values, &grid_thw, CAPTURE_LAYERS)?;
    println!("Forward done. Captures: {}", captures.len());

    // ---- Load Python captures ----
    let captures_path = args.refs_dir.join("captures.safetensors");
    let py: HashMap<String, Tensor> =
        flame_core::serialization::load_file(&captures_path, &device)
            .with_context(|| format!("load {}", captures_path.display()))?;
    println!("Python captures loaded: {}", py.len());

    // ---- Compare ----
    let mut keys: Vec<String> = Vec::new();
    keys.push("pre_block_0".into());
    for &li in CAPTURE_LAYERS {
        keys.push(format!("block.{li}"));
    }
    keys.push("post_merger".into());

    println!(
        "\n{:<14} {:<16} {:>14} {:>14} {:>14}",
        "capture", "shape", "cos", "max_abs", "rms"
    );
    println!("{}", "-".repeat(76));

    let mut worst_cos = 1.0f64;
    for name in &keys {
        let rs = captures.get(name).ok_or_else(|| anyhow::anyhow!("missing Rust capture {name}"))?;
        let py_t = py
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing Python capture {name}"))?;
        let (cos, max_abs, rms) = cos_sim(rs, py_t)?;
        if cos < worst_cos {
            worst_cos = cos;
        }
        println!(
            "{:<14} {:<16} {:>14.6} {:>14.4} {:>14.6}",
            name,
            format!("{:?}", rs.shape().dims()),
            cos,
            max_abs,
            rms
        );
    }

    println!("\nworst cos: {worst_cos:.6}");
    let _ = post_merger;

    if worst_cos < 0.999 {
        println!("=== PARITY FAIL (worst cos < 0.999) ===");
        std::process::exit(1);
    } else {
        println!("=== PARITY OK (worst cos ≥ 0.999) ===");
    }
    Ok(())
}
