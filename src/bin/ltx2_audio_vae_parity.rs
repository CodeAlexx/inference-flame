//! Audio VAE parity binary.
//!
//! Loads the stage-2 audio latent, decodes with the Rust `LTX2AudioVaeDecoder`,
//! saves the mel output for Python parity comparison.

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::vae::LTX2AudioVaeDecoder;
use std::collections::HashMap;
use std::time::Instant;

const CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

fn stats(name: &str, t: &Tensor) {
    let v = t.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| { let d = *x as f64 - mean; d * d }).sum::<f64>() / n;
    let std = var.sqrt();
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for x in &v { if *x < mn { mn = *x; } if *x > mx { mx = *x; } }
    println!("  {name:<28} shape={:?} mean={:+.4} std={:.4} min={:.4} max={:.4}",
        t.shape().dims(), mean, std, mn, mx);
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();
    println!("=== LTX-2.3 Audio VAE parity test ===\n");

    let latent_path = format!("{OUTPUT_DIR}/ltx2_twostage_audio_latents.safetensors");
    let loaded = flame_core::serialization::load_file(std::path::Path::new(&latent_path), &device)?;
    let latent = loaded.get("latents")
        .ok_or_else(|| anyhow::anyhow!("missing 'latents' in {latent_path}"))?
        .to_dtype(DType::BF16)?;
    stats("input latent", &latent);

    let t0 = Instant::now();
    let dec = LTX2AudioVaeDecoder::from_file(CHECKPOINT, &device)?;
    println!("  audio VAE loaded in {:.1}s\n", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let mel = dec.decode(&latent)?;
    println!("  decode: {:.1}s", t1.elapsed().as_secs_f32());
    stats("rust mel", &mel);

    let out_path = format!("{OUTPUT_DIR}/rust_audio_vae_decoded.safetensors");
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("mel".to_string(), mel.to_dtype(DType::F32)?);
    flame_core::serialization::save_tensors(
        &out,
        std::path::Path::new(&out_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("\n  Saved {out_path}");
    Ok(())
}
