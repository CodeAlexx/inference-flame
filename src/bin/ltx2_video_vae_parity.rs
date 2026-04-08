//! Video VAE parity test: decode the Rust stage-2 latent with the new Rust VAE
//! and compare to Python's reference `python_video_decoded.safetensors`.

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::vae::LTX2VaeDecoder;
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

    println!("=== LTX-2.3 Video VAE parity test ===\n");

    // --- Load Rust stage-2 latent (full) and optionally crop for OOM safety ---
    let latent_path = format!("{OUTPUT_DIR}/ltx2_twostage_video_latents.safetensors");
    let loaded = flame_core::serialization::load_file(std::path::Path::new(&latent_path), &device)?;
    let latent_full = loaded.get("latents")
        .ok_or_else(|| anyhow::anyhow!("missing 'latents' in {latent_path}"))?
        .to_dtype(DType::BF16)?;
    stats("input latent (full)", &latent_full);

    // Environment switch: LTX2_VAE_CROP=F,H,W (e.g. "5,4,4") for small parity.
    // Default: use full latent and hope it fits.
    let latent = match std::env::var("LTX2_VAE_CROP").ok() {
        Some(spec) => {
            let parts: Vec<usize> = spec.split(',').filter_map(|s| s.parse().ok()).collect();
            if parts.len() != 3 {
                anyhow::bail!("LTX2_VAE_CROP must be F,H,W");
            }
            let (fc, hc, wc) = (parts[0], parts[1], parts[2]);
            // [1, 128, F, H, W] → narrow each
            let t = latent_full.narrow(2, 0, fc)?
                .narrow(3, 0, hc)?
                .narrow(4, 0, wc)?;
            stats("input latent (cropped)", &t);
            t
        }
        None => latent_full,
    };

    // --- Load VAE ---
    let t0 = Instant::now();
    let vae = LTX2VaeDecoder::from_file(CHECKPOINT, &device)?;
    println!("  VAE loaded in {:.1}s\n", t0.elapsed().as_secs_f32());

    // --- Decode (with optional activation dump) ---
    let t1 = Instant::now();
    let mut activations: HashMap<String, Tensor> = HashMap::new();
    let dump_enabled = std::env::var("FLAME_VAE_DUMP").is_ok();
    let frames = if dump_enabled {
        vae.decode_with_dump(&latent, Some(&mut activations))?
    } else {
        vae.decode(&latent)?
    };
    println!("  decode: {:.1}s", t1.elapsed().as_secs_f32());
    stats("rust frames", &frames);

    if dump_enabled {
        for (k, v) in &activations {
            stats(k, v);
        }
        activations.insert("final_frames".to_string(), frames.to_dtype(DType::F32)?);
        let act_path = format!("{OUTPUT_DIR}/rust_video_activations.safetensors");
        flame_core::serialization::save_tensors(
            &activations,
            std::path::Path::new(&act_path),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        println!("\n  Saved {act_path}");
    }

    // --- Save final frames ---
    let out_path = format!("{OUTPUT_DIR}/rust_video_decoded.safetensors");
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("frames".to_string(), frames.to_dtype(DType::F32)?);
    flame_core::serialization::save_tensors(
        &out,
        std::path::Path::new(&out_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("\n  Saved {out_path}");
    Ok(())
}
