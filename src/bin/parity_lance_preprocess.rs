//! Parity binary for `inference_flame::preprocess::qwen25vl::prepare_image_for_vit`.
//!
//! Loads the Python reference produced by
//! `ports/lance/parity/gen_refs_lance_preprocess.py` and compares against the
//! Rust output for the same image + resolution.

use std::path::PathBuf;

use anyhow::{Context, Result};
use flame_core::{DType, Tensor};
use inference_flame::preprocess::qwen25vl::prepare_image_for_vit;

struct Args {
    image: PathBuf,
    resolution: u32,
    refs_dir: PathBuf,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            image: PathBuf::new(),
            resolution: 476,
            refs_dir: PathBuf::from("ports/lance/parity/refs_preprocess"),
        };
        let argv: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--image" => {
                    a.image = PathBuf::from(&argv[i + 1]);
                    i += 2;
                }
                "--resolution" => {
                    a.resolution = argv[i + 1].parse().expect("--resolution int");
                    i += 2;
                }
                "--refs-dir" | "--refs_dir" => {
                    a.refs_dir = PathBuf::from(&argv[i + 1]);
                    i += 2;
                }
                "-h" | "--help" => {
                    println!(
                        "parity_lance_preprocess --image PATH [--resolution INT] [--refs-dir PATH]"
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

    let refs_path = args.refs_dir.join("preprocess_refs.safetensors");
    println!("Loading Python refs: {}", refs_path.display());
    let refs = flame_core::serialization::load_file(&refs_path, &device)
        .with_context(|| format!("load {}", refs_path.display()))?;

    let py_pixel_values = refs
        .get("pixel_values")
        .ok_or_else(|| anyhow::anyhow!("missing pixel_values in refs"))?
        .clone();
    let py_grid_thw_t = refs
        .get("grid_thw")
        .ok_or_else(|| anyhow::anyhow!("missing grid_thw in refs"))?
        .clone();

    let py_grid_vec: Vec<f32> = py_grid_thw_t.to_dtype(DType::F32)?.to_vec()?;
    let py_grid_thw: [u32; 3] = [
        py_grid_vec[0] as u32,
        py_grid_vec[1] as u32,
        py_grid_vec[2] as u32,
    ];

    println!(
        "Python pixel_values: shape={:?} dtype={:?}",
        py_pixel_values.shape().dims(),
        py_pixel_values.dtype()
    );
    println!("Python grid_thw: {py_grid_thw:?}");

    // ---- Run Rust preprocess ----
    println!(
        "Running Rust prepare_image_for_vit(image={}, resolution={})",
        args.image.display(),
        args.resolution
    );
    let prepared = prepare_image_for_vit(&args.image, args.resolution, &device)
        .context("prepare_image_for_vit failed")?;
    println!(
        "Rust pixel_values: shape={:?} dtype={:?}",
        prepared.pixel_values.shape().dims(),
        prepared.pixel_values.dtype()
    );
    println!(
        "Rust grid_thw: {:?}   processed (HxW): {}x{}",
        prepared.grid_thw, prepared.processed_h, prepared.processed_w
    );

    // ---- Compare pixel_values ----
    let py_dims = py_pixel_values.shape().dims().to_vec();
    let rs_dims = prepared.pixel_values.shape().dims().to_vec();
    let shape_match = py_dims == rs_dims;
    let grid_match = py_grid_thw == prepared.grid_thw;

    println!("\n{:<16} {:>12} {:>14} {:>14}", "capture", "cos", "max_abs", "rms");
    println!("{}", "-".repeat(60));

    let (cos, max_abs, rms) = if shape_match {
        cos_sim(&prepared.pixel_values, &py_pixel_values)?
    } else {
        println!(
            "shape mismatch! Rust {:?} vs Python {:?}  -- skipping numeric cos",
            rs_dims, py_dims
        );
        (f64::NAN, f32::NAN, f64::NAN)
    };

    println!(
        "{:<16} {:>12.6} {:>14.6} {:>14.6}",
        "pixel_values", cos, max_abs, rms
    );
    println!(
        "{:<16} {:>12} {:>14} {:>14}",
        "grid_thw match",
        grid_match,
        "-",
        "-"
    );

    let pass = shape_match && grid_match && cos >= 0.99;
    if pass {
        println!("\n=== PARITY OK (cos >= 0.99, grid_thw + shape match) ===");
        Ok(())
    } else {
        println!("\n=== PARITY FAIL ===");
        std::process::exit(1);
    }
}
