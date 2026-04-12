//! Wan2.2 VAE encoder parity test.
//!
//! Loads the Wan2.2 VAE encoder, encodes a small random video tensor,
//! saves the output latent to safetensors for comparison with Python.
//!
//! Usage:
//!     cargo run --release --bin wan22_vae_encode_parity
//!
//! Set WAN_VAE_DBG=1 for per-layer activation stats.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use flame_core::serialization;

use inference_flame::models::wan::Wan22VaeEncoder;

const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors";
const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

fn tensor_stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let data = t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = data.len();
    let nan = data.iter().filter(|v| v.is_nan()).count();
    let inf = data.iter().filter(|v| v.is_infinite()).count();
    let mean: f32 = data.iter().copied().sum::<f32>() / (n as f32);
    let abs_mean: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / (n as f32);
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &data {
        if v.is_finite() {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
    }
    println!(
        "  [{name}] shape={:?}  n={n}  nan={nan}  inf={inf}  mean={mean:.4}  |mean|={abs_mean:.4}  range=[{lo:.4}, {hi:.4}]",
        t.shape().dims(),
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("=== Wan2.2 VAE Encoder Parity Test ===\n");

    // --- Create random input video [1, 3, 5, 256, 256] in [-1, 1] ---
    // Use a fixed seed for reproducibility by generating from a known pattern.
    let (b, c, t, h, w) = (1, 3, 5, 256, 256);
    let n = b * c * t * h * w;

    // Generate pseudo-random BF16 values in [-1, 1] using a simple LCG
    let mut rng_state: u64 = 42;
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
        vals.push(u * 2.0 - 1.0); // [-1, 1)
    }

    let video = Tensor::from_vec(
        vals,
        Shape::from_dims(&[b, c, t, h, w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;
    tensor_stats("input video", &video)?;

    // Save input for Python comparison
    let input_path = format!("{OUTPUT_DIR}/wan22_vae_encode_input.safetensors");
    std::fs::create_dir_all(OUTPUT_DIR)?;
    let mut input_tensors = HashMap::new();
    input_tensors.insert("video".to_string(), video.to_dtype(DType::F32)?);
    serialization::save_file(&input_tensors, &input_path)?;
    println!("  Saved input to {input_path}\n");

    // --- Load encoder ---
    println!("--- Loading Wan22 VAE Encoder ---");
    let t0 = Instant::now();
    let encoder = Wan22VaeEncoder::load(Path::new(VAE_PATH), &device)?;
    println!("  Loaded in {:.1}s\n", t0.elapsed().as_secs_f32());

    // --- Encode ---
    println!("--- Encoding ---");
    let t1 = Instant::now();
    let latent = encoder.encode(&video)?;
    println!("  Encode: {:.1}s", t1.elapsed().as_secs_f32());
    tensor_stats("output latent", &latent)?;

    // Save output
    let output_path = format!("{OUTPUT_DIR}/wan22_vae_encode_output.safetensors");
    let mut output_tensors = HashMap::new();
    output_tensors.insert("latent".to_string(), latent.to_dtype(DType::F32)?);
    serialization::save_file(&output_tensors, &output_path)?;
    println!("  Saved output to {output_path}\n");

    println!("=== Done ===");
    println!("Run the Python comparison script to verify parity:");
    println!("  python scripts/wan22_vae_encode_parity.py");

    Ok(())
}
