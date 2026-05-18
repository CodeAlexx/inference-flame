//! Pure-Rust LTX-2.3 audio stack smoke:
//! spectrogram -> Audio VAE encoder -> Audio VAE decoder -> Vocoder+BWE.

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::vae::{LTX2AudioVaeDecoder, LTX2AudioVaeEncoder, LTX2VocoderWithBWE};
use std::collections::HashMap;
use std::time::Instant;

const CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

fn stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
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
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut bad = 0usize;
    for x in &v {
        if !x.is_finite() {
            bad += 1;
        }
        if *x < min {
            min = *x;
        }
        if *x > max {
            max = *x;
        }
    }
    println!(
        "  {name:<24} shape={:?} mean={:+.5} std={:.5} min={:.5} max={:.5} bad={}",
        t.shape().dims(),
        mean,
        var.sqrt(),
        min,
        max,
        bad
    );
    if bad != 0 {
        anyhow::bail!("{name}: found {bad} non-finite values");
    }
    Ok(())
}

fn make_spectrogram() -> Vec<f32> {
    let (b, c, t, f) = (1usize, 2usize, 8usize, 64usize);
    let mut data = Vec::with_capacity(b * c * t * f);
    for bi in 0..b {
        for ch in 0..c {
            for ti in 0..t {
                for fi in 0..f {
                    let phase = (ti as f32 * 0.37) + (fi as f32 * 0.071) + (ch as f32 * 0.53) + bi as f32;
                    let value = -4.0 + 0.65 * phase.sin() + 0.25 * (phase * 0.31).cos();
                    data.push(value);
                }
            }
        }
    }
    data
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let total = Instant::now();
    let device = global_cuda_device();

    println!("=== LTX-2.3 Pure Rust Audio Stack Smoke ===");
    println!("  checkpoint: {CHECKPOINT}");

    let input = Tensor::from_vec(
        make_spectrogram(),
        Shape::from_dims(&[1, 2, 8, 64]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;
    stats("input spectrogram", &input)?;

    let t0 = Instant::now();
    let encoder = LTX2AudioVaeEncoder::from_file(CHECKPOINT, &device)?;
    println!("  encoder loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let latents = encoder.encode_spectrogram(&input)?;
    device.synchronize()?;
    println!("  encode in {:.2}s", t1.elapsed().as_secs_f32());
    stats("audio latents", &latents)?;
    drop(encoder);

    let t2 = Instant::now();
    let decoder = LTX2AudioVaeDecoder::from_file(CHECKPOINT, &device)?;
    println!("  decoder loaded in {:.2}s", t2.elapsed().as_secs_f32());

    let t3 = Instant::now();
    let mel = decoder.decode(&latents)?;
    device.synchronize()?;
    println!("  decode in {:.2}s", t3.elapsed().as_secs_f32());
    stats("decoded mel", &mel)?;
    drop(decoder);

    let t4 = Instant::now();
    let vocoder = LTX2VocoderWithBWE::from_file(CHECKPOINT, &device)?;
    println!("  vocoder+BWE loaded in {:.2}s", t4.elapsed().as_secs_f32());

    let t5 = Instant::now();
    let waveform = vocoder.forward(&mel)?;
    device.synchronize()?;
    println!(
        "  vocoder+BWE in {:.2}s at {} Hz",
        t5.elapsed().as_secs_f32(),
        vocoder.output_sample_rate()
    );
    stats("waveform", &waveform)?;

    std::fs::create_dir_all(OUTPUT_DIR)?;
    let mut out = HashMap::new();
    out.insert("spectrogram".to_string(), input.to_dtype(DType::F32)?);
    out.insert("latents".to_string(), latents.to_dtype(DType::F32)?);
    out.insert("mel".to_string(), mel.to_dtype(DType::F32)?);
    out.insert("waveform".to_string(), waveform.to_dtype(DType::F32)?);
    flame_core::serialization::save_tensors(
        &out,
        std::path::Path::new(&format!("{OUTPUT_DIR}/ltx2_audio_stack_smoke.safetensors")),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;

    println!("  total {:.2}s", total.elapsed().as_secs_f32());
    Ok(())
}
