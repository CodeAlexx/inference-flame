//! Pure-Rust Oobleck VAE round-trip test:
//!   WAV → encoder → audio_lat → decoder → WAV
//!
//! Bypasses the MagiHuman DiT entirely. Output WAV should be intelligible
//! if the encode/decode round-trip preserves speech. If it doesn't, we have
//! a port bug in encoder, decoder, or the bottleneck step.
//!
//! Usage:
//!   audio_roundtrip <input.wav> <output.wav> [--sa-vae <path>]

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, serialization, DType, Tensor};
use hound::{SampleFormat, WavSpec, WavWriter};
use inference_flame::audio::{load_and_encode_audio, TARGET_SAMPLE_RATE, TOTAL_DOWNSAMPLE};
use inference_flame::audio::wav::probe_duration;
use inference_flame::models::sa_audio_vae::OobleckDecoder;
use inference_flame::mux::audio_tensor_to_pcm_i16;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or_else(|| anyhow!(
        "usage: audio_roundtrip <input.wav | --latent <fixture.safetensors[:key]>> <output.wav> [--sa-vae path]"
    ))?;
    let mut sa_vae = PathBuf::from(
        "/home/alex/.serenity/models/vaes/stable_audio_oobleck_vae.safetensors",
    );

    // Two modes: "<input.wav> <output.wav> ..." OR "--latent <path[:key]> <output.wav> ..."
    let (latent_input, output) = if input == "--latent" {
        let lat_arg = args.next().ok_or_else(|| anyhow!("--latent requires <path[:key]>"))?;
        let out = args.next().ok_or_else(|| anyhow!("--latent <path> <output.wav>"))?;
        (Some(lat_arg), out)
    } else {
        let out = args.next().ok_or_else(|| anyhow!("<input.wav> <output.wav>"))?;
        (None, out)
    };
    while let Some(a) = args.next() {
        if a == "--sa-vae" { sa_vae = PathBuf::from(args.next().unwrap()); } else {
            anyhow::bail!("unknown arg: {a}");
        }
    }

    let device = global_cuda_device();

    // === MODE A: load audio_lat directly from a .safetensors fixture ===
    let audio_lat: Tensor = if let Some(lat_arg) = latent_input.as_ref() {
        let (path_str, key) = match lat_arg.rsplit_once(':') {
            Some((p, k)) => (p.to_string(), k.to_string()),
            None => (lat_arg.clone(), "audio_lat".to_string()),
        };
        let path = PathBuf::from(&path_str);
        let tensors = serialization::load_file(&path, &device)?;
        let lat = tensors
            .get(&key)
            .ok_or_else(|| anyhow!("fixture {path_str} missing key '{key}' (have: {:?})", tensors.keys().collect::<Vec<_>>()))?
            .to_dtype(DType::F32)?;
        println!("[in   ] --latent {path_str}:{key}  shape={:?}  dtype={:?}", lat.shape().dims(), lat.dtype());
        // Don't permute here — we'll handle layout below with a check.
        lat
    } else {
        // === MODE B: WAV → encode → audio_lat ===
        let input_path = PathBuf::from(&input);
        let (dur_sec, native_rate, ch) = probe_duration(&input_path)?;
        let target_samples = (dur_sec * TARGET_SAMPLE_RATE as f64).ceil() as usize;
        let num_frames = (target_samples + TOTAL_DOWNSAMPLE - 1) / TOTAL_DOWNSAMPLE;
        println!(
            "[in   ] {input}  dur={dur_sec:.3}s  rate={native_rate}  ch={ch}  → num_frames={num_frames}"
        );
        load_and_encode_audio(&input_path, &sa_vae, num_frames, &device)?
    };
    let lat_dims = audio_lat.shape().dims().to_vec();
    println!("[enc  ] audio_lat={lat_dims:?}  dtype={:?}", audio_lat.dtype());

    // Diagnostic: latent statistics. Healthy SA Open latents are ~N(0,1)-ish per channel.
    let lat_v = audio_lat.to_vec_f32()?;
    let lat_mean: f32 = lat_v.iter().sum::<f32>() / lat_v.len() as f32;
    let lat_var: f32 = lat_v.iter().map(|x| (x - lat_mean).powi(2)).sum::<f32>() / lat_v.len() as f32;
    let lat_max: f32 = lat_v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let n_finite = lat_v.iter().filter(|x| x.is_finite()).count();
    println!(
        "[enc  ] latent stats: mean={lat_mean:+.4}  std={:.4}  |max|={lat_max:.4}  finite={n_finite}/{}",
        lat_var.sqrt(), lat_v.len()
    );

    // === DECODE ===
    // Decoder wants [1, 64, T_lat]. Source could be either [1, T, 64] (audio_lat
    // layout from WAV→encode path / saved post-DiT fixture) or already
    // [1, 64, T]. Detect by checking which axis equals 64.
    let z = {
        let d = audio_lat.shape().dims().to_vec();
        if d.len() == 3 && d[0] == 1 && d[1] == 64 {
            audio_lat.contiguous()? // already [1, 64, T]
        } else if d.len() == 3 && d[0] == 1 && d[2] == 64 {
            audio_lat.permute(&[0, 2, 1])?.contiguous()? // [1, T, 64] → [1, 64, T]
        } else {
            anyhow::bail!("audio_lat shape {d:?} doesn't match [1, T, 64] or [1, 64, T]");
        }
    };
    let decoder = OobleckDecoder::load_default(sa_vae.to_str().unwrap(), &device)
        .map_err(|e| anyhow!("OobleckDecoder load: {e}"))?;
    let t_dec = std::time::Instant::now();
    let audio = decoder
        .decode(&z)
        .map_err(|e| anyhow!("OobleckDecoder decode: {e}"))?;
    println!("[dec  ] decode {} ms  shape={:?}", t_dec.elapsed().as_millis(), audio.shape().dims());
    drop(decoder);

    let dims = audio.shape().dims().to_vec();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != 2 {
        anyhow::bail!("decoder output shape unexpected: {dims:?}");
    }
    let n_samples = dims[2];

    let audio_f32 = audio.to_dtype(DType::F32)?.to_vec_f32()?;
    let out_max: f32 = audio_f32.iter().fold(0f32, |m, x| m.max(x.abs()));
    let out_mean: f32 = audio_f32.iter().sum::<f32>() / audio_f32.len() as f32;
    let out_var: f32 = audio_f32.iter().map(|x| (x - out_mean).powi(2)).sum::<f32>() / audio_f32.len() as f32;
    println!("[dec  ] waveform stats: |max|={out_max:.4}  mean={out_mean:+.4}  rms={:.4}", out_var.sqrt());

    // === WRITE WAV at the model's native rate (51200 Hz stereo s16) ===
    let pcm = audio_tensor_to_pcm_i16(&audio_f32, 2, n_samples, /* channels_first */ true);
    let spec = WavSpec {
        channels: 2,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(&output, spec)
        .map_err(|e| anyhow!("WavWriter::create({output}): {e}"))?;
    for s in &pcm {
        w.write_sample(*s).map_err(|e| anyhow!("write sample: {e}"))?;
    }
    w.finalize().map_err(|e| anyhow!("finalize: {e}"))?;
    println!("[out  ] wrote {output}  ({n_samples} samples × 2ch @ {} Hz, {:.3}s)",
             TARGET_SAMPLE_RATE,
             n_samples as f32 / TARGET_SAMPLE_RATE as f32);

    // Suppress unused-import warning if Tensor goes unused via inference.
    let _ = std::any::type_name::<Tensor>();
    Ok(())
}
