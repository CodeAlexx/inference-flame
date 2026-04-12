//! LDM VAE encoder parity test — encode an image (or random tensor) and save output.
//!
//! Usage:
//!   cargo run --release --bin ldm_vae_encode_parity [vae_path] [latent_channels]
//!
//! Defaults to SDXL VAE (4 latent channels). Compare output with the Python script
//! `scripts/ldm_vae_encode_parity.py` for numerical parity.

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::vae::LdmVAEEncoder;
use std::collections::HashMap;
use std::time::Instant;

const DEFAULT_VAE_PATH: &str =
    "/home/alex/.cache/huggingface/hub/models--stabilityai--sdxl-vae/snapshots/842b3456e87ab5eb43cd1e8f1a828a4151c4d24d/sdxl_vae.safetensors";
const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

fn stats(name: &str, t: &Tensor) {
    let v = t.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v
        .iter()
        .map(|x| {
            let d = *x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for x in &v {
        if *x < mn {
            mn = *x;
        }
        if *x > mx {
            mx = *x;
        }
    }
    println!(
        "  {name:<28} shape={:?} mean={:+.6} std={:.6} min={:.6} max={:.6}",
        t.shape().dims(),
        mean,
        std,
        mn,
        mx
    );
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    let vae_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_VAE_PATH.to_string());
    let latent_ch: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    println!("=== LDM VAE Encoder Parity Test ===");
    println!("  VAE: {vae_path}");
    println!("  Latent channels: {latent_ch}");
    println!();

    // Generate deterministic random input [1, 3, 512, 512]
    // Use a simple linear ramp normalized to [-1, 1] for reproducibility.
    let (h, w) = (512usize, 512usize);
    let n_pixels = 3 * h * w;
    let pixel_data: Vec<f32> = (0..n_pixels)
        .map(|i| (i as f32 / n_pixels as f32) * 2.0 - 1.0)
        .collect();
    let image = Tensor::from_vec(pixel_data, Shape::from_dims(&[1, 3, h, w]), device.clone())?
        .to_dtype(DType::BF16)?;
    stats("input image", &image);

    // Also save the input for Python comparison
    let input_path = format!("{OUTPUT_DIR}/ldm_vae_encode_input.safetensors");
    std::fs::create_dir_all(OUTPUT_DIR)?;
    let mut input_tensors = HashMap::new();
    input_tensors.insert("image".to_string(), image.to_dtype(DType::F32)?);
    flame_core::serialization::save_file(&input_tensors, &input_path)?;
    println!("  Saved input to {input_path}");

    // Load encoder
    let t0 = Instant::now();
    let encoder = LdmVAEEncoder::from_safetensors(&vae_path, latent_ch, &device)?;
    println!("  Encoder loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Encode
    let t1 = Instant::now();
    let latent = encoder.encode(&image)?;
    println!("  Encode time: {:.3}s", t1.elapsed().as_secs_f32());
    stats("output latent", &latent);

    // Save output
    let out_path = format!("{OUTPUT_DIR}/ldm_vae_encode_rust.safetensors");
    let mut out_tensors = HashMap::new();
    out_tensors.insert("latent".to_string(), latent.to_dtype(DType::F32)?);
    flame_core::serialization::save_file(&out_tensors, &out_path)?;
    println!("  Saved output to {out_path}");

    // Also test encode_scaled for SDXL defaults
    if latent_ch == 4 {
        let scaling_factor = 0.13025f32;
        let shift_factor = 0.0f32;
        let scaled = encoder.encode_scaled(&image, scaling_factor, shift_factor)?;
        stats("scaled latent (SDXL)", &scaled);
    }

    println!("\nDone. Compare with scripts/ldm_vae_encode_parity.py");
    Ok(())
}
