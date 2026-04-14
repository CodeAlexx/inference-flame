//! QwenImage VAE encoder parity test.
//!
//! Encodes the same image as Python and compares latents.
//!
//! Usage:
//!   cargo run --release --bin qwenimage_vae_parity

use flame_core::{DType, Shape, Tensor};
use inference_flame::vae::QwenImageVaeEncoder;

const VAE_PATH: &str = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors";
const TEST_IMG: &str = "/home/alex/datasets/boxjana/10.jpg";
const PY_CACHE: &str = "/home/alex/datasets/boxjana_cached/10.safetensors";

fn main() -> anyhow::Result<()> {
    let device = flame_core::global_cuda_device();

    // Load + preprocess image: resize 512x512, [-1, 1], [B, C, T, H, W]
    let img = image::open(TEST_IMG)?.to_rgb8();
    let (tw, th) = (512usize, 512usize);
    let resized = image::imageops::resize(&img, tw as u32, th as u32, image::imageops::FilterType::Lanczos3);

    let mut data = vec![0.0f32; 3 * th * tw];
    for y in 0..th {
        for x in 0..tw {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * th * tw + y * tw + x] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    // [1, 3, 1, 512, 512] — 5D with T=1
    let img_tensor = Tensor::from_f32_to_bf16(
        data, Shape::from_dims(&[1, 3, 1, th, tw]), device.clone(),
    )?;
    println!("Input: {:?}", img_tensor.dims());

    // Encode
    println!("Loading QwenImage VAE encoder...");
    let encoder = QwenImageVaeEncoder::from_safetensors(VAE_PATH, &device)?;
    println!("Encoding...");
    let latent = encoder.encode(&img_tensor)?;
    println!("Rust latent: {:?}", latent.dims());

    // Stats
    let lat_f32 = latent.to_dtype(DType::F32)?.to_vec()?;
    let mean: f32 = lat_f32.iter().sum::<f32>() / lat_f32.len() as f32;
    let std: f32 = (lat_f32.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / lat_f32.len() as f32).sqrt();
    println!("  mean={mean:.4}, std={std:.4}");

    // Compare with Python cache
    let py = flame_core::serialization::load_file(std::path::Path::new(PY_CACHE), &device)?;
    let py_lat = py.get("latent").expect("missing 'latent' in Python cache");
    // Python: [1, 16, 64, 64], Rust: [1, 16, 1, 64, 64] — squeeze T dim
    let rust_lat = latent.squeeze_dim(2)?; // [1, 16, 64, 64]
    println!("Rust squeezed: {:?}, Python: {:?}", rust_lat.dims(), py_lat.dims());

    let r = rust_lat.to_dtype(DType::F32)?.to_vec()?;
    let p = py_lat.to_dtype(DType::F32)?.to_vec()?;
    assert_eq!(r.len(), p.len(), "length mismatch");

    let mut max_abs: f32 = 0.0;
    let mut dot: f64 = 0.0;
    let mut r_sq: f64 = 0.0;
    let mut p_sq: f64 = 0.0;
    for i in 0..r.len() {
        max_abs = max_abs.max((r[i] - p[i]).abs());
        dot += r[i] as f64 * p[i] as f64;
        r_sq += r[i] as f64 * r[i] as f64;
        p_sq += p[i] as f64 * p[i] as f64;
    }
    let cosine = dot / (r_sq.sqrt() * p_sq.sqrt());
    println!("\n=== Parity ===");
    println!("  max_abs_diff: {max_abs:.6}");
    println!("  cosine_sim:   {cosine:.9}");
    println!("  status:       {}", if cosine > 0.9999 { "PASS" } else { "FAIL" });

    Ok(())
}
