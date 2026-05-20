//! End-to-end parity: Rust `preprocess::qwen25vl::prepare_image_for_vit` +
//! Rust `Qwen25VLVisionTower::forward_capture` vs Python `VideoTransform` +
//! `Qwen2_5_VisionTransformerPretrainedModel`.
//!
//! Companion Python refs are produced by
//! `ports/lance/parity/gen_refs_lance_vit_real_image.py`.
//!
//! Unlike `parity_lance_vit` (which loads Python's pre-saved pixel_values to
//! isolate the ViT), this bin re-runs the Rust preprocessor on the same real
//! image and feeds *its* output into the Rust ViT. A passing run proves the
//! preprocessor + ViT combo is correct end-to-end, not just each in isolation.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use flame_core::{DType, Tensor};
use inference_flame::models::qwen25vl_vit::{Qwen25VLVisionTower, Qwen25VLVitConfig};
use inference_flame::preprocess::qwen25vl::prepare_image_for_vit;

const CAPTURE_LAYERS: &[usize] = &[0, 7, 15, 23, 31];
const PRE_BLOCK_THRESHOLD: f64 = 0.999;
const POST_MERGER_THRESHOLD: f64 = 0.99;

struct Args {
    model_dir: PathBuf,
    refs_dir: PathBuf,
    image: PathBuf,
    resolution: u32,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            model_dir: PathBuf::from("/home/alex/.serenity/models/lance/Lance_3B_Video"),
            refs_dir: PathBuf::from("ports/lance/parity/refs_vit_real_image"),
            image: PathBuf::new(),
            resolution: 476,
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
                "--image" => {
                    a.image = PathBuf::from(&argv[i + 1]);
                    i += 2;
                }
                "--resolution" => {
                    a.resolution = argv[i + 1].parse().expect("--resolution int");
                    i += 2;
                }
                "-h" | "--help" => {
                    println!(
                        "parity_lance_vit_end_to_end --image PATH \\\n  \
                         [--resolution INT] [--model-dir PATH] [--refs-dir PATH]\n  \
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
        if a.image.as_os_str().is_empty() {
            eprintln!("--image is required");
            std::process::exit(2);
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

    // ---- Run Rust preprocessor ----
    println!(
        "Rust preprocess: image={}, resolution={}",
        args.image.display(),
        args.resolution
    );
    let prepared = prepare_image_for_vit(&args.image, args.resolution, &device)
        .context("prepare_image_for_vit failed")?;
    let pixel_values = prepared.pixel_values.to_dtype(DType::BF16)?;
    let grid_thw_vec = vec![prepared.grid_thw];
    println!(
        "Rust pixel_values: shape={:?} dtype={:?}",
        pixel_values.shape().dims(),
        pixel_values.dtype()
    );
    println!(
        "Rust grid_thw: {:?}, processed HxW: {}x{}",
        prepared.grid_thw, prepared.processed_h, prepared.processed_w
    );

    // ---- Build Rust ViT ----
    let cfg = Qwen25VLVitConfig::default();
    let ckpt = args.model_dir.join("model.safetensors");
    println!("Loading Rust ViT from {}", ckpt.display());
    let raw = flame_core::serialization::load_file_filtered(&ckpt, &device, |k| {
        k.starts_with("vit_model.")
    })?;
    let tower = Qwen25VLVisionTower::from_weights(raw, cfg.clone(), device.clone())?;

    // ---- Forward with per-block captures ----
    let (post_merger, captures) =
        tower.forward_capture(&pixel_values, &grid_thw_vec, CAPTURE_LAYERS)?;
    println!("Forward done. Captures: {}", captures.len());

    // ---- Load Python captures ----
    let captures_path = args.refs_dir.join("captures.safetensors");
    println!("Loading Python refs: {}", captures_path.display());
    let py: HashMap<String, Tensor> =
        flame_core::serialization::load_file(&captures_path, &device)
            .with_context(|| format!("load {}", captures_path.display()))?;
    println!("Python captures: {} entries", py.len());

    // Sanity: grid_thw matches what Python recorded.
    if let Some(py_grid) = py.get("input.grid_thw") {
        let g: Vec<f32> = py_grid.to_dtype(DType::F32)?.to_vec()?;
        let py_arr = [g[0] as u32, g[1] as u32, g[2] as u32];
        if py_arr != prepared.grid_thw {
            println!(
                "WARN: grid_thw mismatch — Python {:?} vs Rust {:?}",
                py_arr, prepared.grid_thw
            );
        } else {
            println!("grid_thw OK ({:?})", py_arr);
        }
    }

    // ---- Compare ----
    let mut keys: Vec<String> = Vec::new();
    keys.push("pre_block_0".into());
    for &li in CAPTURE_LAYERS {
        keys.push(format!("block.{li}"));
    }
    keys.push("post_merger".into());

    println!(
        "\n{:<14} {:<18} {:>14} {:>14} {:>14}",
        "capture", "shape", "cos", "max_abs", "rms"
    );
    println!("{}", "-".repeat(80));

    let mut pre_block_0_cos = f64::NAN;
    let mut post_merger_cos = f64::NAN;
    let mut worst_cos = 1.0f64;
    for name in &keys {
        let rs = captures
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing Rust capture {name}"))?;
        let py_t = py
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing Python capture {name}"))?;
        let (cos, max_abs, rms) = cos_sim(rs, py_t)?;
        if cos < worst_cos {
            worst_cos = cos;
        }
        if name == "pre_block_0" {
            pre_block_0_cos = cos;
        }
        if name == "post_merger" {
            post_merger_cos = cos;
        }
        println!(
            "{:<14} {:<18} {:>14.6} {:>14.4} {:>14.6}",
            name,
            format!("{:?}", rs.shape().dims()),
            cos,
            max_abs,
            rms
        );
    }

    println!("\nworst cos: {worst_cos:.6}");
    println!(
        "pre_block_0 cos = {:.6}  (threshold ≥ {:.3})",
        pre_block_0_cos, PRE_BLOCK_THRESHOLD
    );
    println!(
        "post_merger cos = {:.6}  (threshold ≥ {:.3})",
        post_merger_cos, POST_MERGER_THRESHOLD
    );
    let _ = post_merger;

    let pass = pre_block_0_cos >= PRE_BLOCK_THRESHOLD
        && post_merger_cos >= POST_MERGER_THRESHOLD;
    if pass {
        println!(
            "\n=== END-TO-END PARITY OK (pre_block_0 ≥ {:.3}, post_merger ≥ {:.3}) ===",
            PRE_BLOCK_THRESHOLD, POST_MERGER_THRESHOLD
        );
        Ok(())
    } else {
        println!(
            "\n=== END-TO-END PARITY FAIL (pre_block_0 ≥ {:.3} OR post_merger ≥ {:.3} not met) ===",
            PRE_BLOCK_THRESHOLD, POST_MERGER_THRESHOLD
        );
        std::process::exit(1);
    }
}
