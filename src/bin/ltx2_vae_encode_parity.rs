//! LTX-2.3 Video VAE encoder parity test — encode a random video tensor and
//! save output for comparison with the Python reference.
//!
//! Usage:
//!   cargo run --release --bin ltx2_vae_encode_parity
//!
//! Compare output with `scripts/ltx2_vae_encode_parity.py`.

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::vae::LTX2VaeEncoder;
use std::collections::HashMap;
use std::time::Instant;

const CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
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

    println!("=== LTX-2.3 Video VAE Encoder Parity Test ===\n");

    // --- Create deterministic random input ---
    // Use a fixed seed for reproducibility. Single frame for simplicity.
    let seed: u64 = 42;
    let (b, c, t, h, w) = (1, 3, 1, 256, 256);
    println!("Input shape: [{b}, {c}, {t}, {h}, {w}] (BF16, seeded with {seed})");

    // Generate random tensor via host-side RNG then upload.
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let n = b * c * t * h * w;
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let input_f32 = Tensor::from_vec(data, Shape::from_dims(&[b, c, t, h, w]), device.clone())?;
    let input = input_f32.to_dtype(DType::BF16)?;
    stats("input", &input);

    // Save input for Python comparison.
    let input_path = format!("{OUTPUT_DIR}/ltx2_vae_encode_input.safetensors");
    std::fs::create_dir_all(OUTPUT_DIR)?;
    let mut input_map: HashMap<String, Tensor> = HashMap::new();
    input_map.insert("input".to_string(), input.to_dtype(DType::F32)?);
    flame_core::serialization::save_tensors(
        &input_map,
        std::path::Path::new(&input_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Saved input to {input_path}\n");

    // --- Load encoder ---
    let t0 = Instant::now();
    let encoder = LTX2VaeEncoder::from_file(CHECKPOINT, &device)?;
    println!("  Encoder loaded in {:.1}s\n", t0.elapsed().as_secs_f32());

    // --- Encode ---
    let t1 = Instant::now();
    let latent = encoder.encode(&input)?;
    println!("  Encode: {:.1}s", t1.elapsed().as_secs_f32());
    stats("latent (normalized)", &latent);

    // Also encode raw (without per-channel normalization).
    let latent_raw = encoder.encode_raw(&input)?;
    stats("latent (raw)", &latent_raw);

    // --- Save output ---
    let out_path = format!("{OUTPUT_DIR}/ltx2_vae_encode_output.safetensors");
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("latent".to_string(), latent.to_dtype(DType::F32)?);
    out.insert("latent_raw".to_string(), latent_raw.to_dtype(DType::F32)?);
    flame_core::serialization::save_tensors(
        &out,
        std::path::Path::new(&out_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("\n  Saved output to {out_path}");

    println!("\nDone. Run scripts/ltx2_vae_encode_parity.py to compare with Python.");
    Ok(())
}
