//! Vocoder parity binary — base 16 kHz output (no BWE).

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::vae::LTX2Vocoder;
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
    println!("=== LTX-2.3 Vocoder parity test (base 16 kHz) ===\n");

    // Use the Python reference mel so the vocoder input is identical.
    let mel_path = format!("{OUTPUT_DIR}/python_audio_vae_decoded.safetensors");
    let loaded = flame_core::serialization::load_file(std::path::Path::new(&mel_path), &device)?;
    let mel_full = loaded.get("mel")
        .ok_or_else(|| anyhow::anyhow!("missing 'mel' in {mel_path}"))?
        .to_dtype(DType::BF16)?;
    stats("input mel (full)", &mel_full);
    // Optional crop: LTX2_MEL_CROP=T (keeps first T time frames).
    let mel = match std::env::var("LTX2_MEL_CROP").ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(t) => {
            let cropped = mel_full.narrow(2, 0, t)?;
            stats("input mel (cropped)", &cropped);
            cropped
        }
        None => mel_full,
    };

    let t0 = Instant::now();
    let voc = LTX2Vocoder::from_file(CHECKPOINT, &device, "vocoder")?;
    println!("  vocoder loaded in {:.1}s\n", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let wf = voc.forward(&mel)?;
    println!("  vocoder forward: {:.1}s", t1.elapsed().as_secs_f32());
    stats("rust waveform", &wf);

    let out_path = format!("{OUTPUT_DIR}/rust_vocoder_decoded.safetensors");
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("waveform".to_string(), wf.to_dtype(DType::F32)?);
    flame_core::serialization::save_tensors(
        &out,
        std::path::Path::new(&out_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("\n  Saved {out_path}");
    Ok(())
}
