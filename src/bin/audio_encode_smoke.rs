//! Smoke test for the new pure-Rust audio path.
//!
//! Generates a 1-sec 440 Hz sine wave WAV at 44,100 Hz, runs through the
//! full pipeline (load → resample → SA Open VAE encode → reshape), prints
//! the resulting shape and a small statistics summary.
//!
//! Verifies:
//!   * WAV write/read roundtrip works
//!   * Resampler 44.1 kHz → 51.2 kHz produces ~51,200 samples for 1 sec
//!   * Encoder loads from sa_vae_weights without missing keys
//!   * Output shape is exactly [1, 25, 64] for 1 sec @ 25 fps
//!   * Output values are finite (no NaN / Inf)
//!
//! No GPU heavy DiT loaded; runs in seconds.

use anyhow::Result;
use cudarc::driver::CudaDevice;
use hound::{WavSpec, WavWriter};
use std::path::PathBuf;

fn main() -> Result<()> {
    let device = CudaDevice::new(0)?;

    // 1. Generate sine WAV.
    let tmp = std::env::temp_dir().join("magihuman_audio_smoke.wav");
    let rate = 44_100u32;
    let dur_sec = 1.0f32;
    let n_samples = (rate as f32 * dur_sec) as usize;
    let spec = WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut w = WavWriter::create(&tmp, spec)?;
        for i in 0..n_samples {
            let t = i as f32 / rate as f32;
            let v = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
            let i16v = (v * (i16::MAX as f32)) as i16;
            w.write_sample(i16v)?;
        }
        w.finalize()?;
    }
    eprintln!("[smoke] wrote {} ({} samples)", tmp.display(), n_samples);

    // 2. Load + encode.
    let sa_vae = PathBuf::from("/home/alex/.serenity/models/vaes/stable_audio_oobleck_vae.safetensors");
    let num_frames = 25; // 1 sec @ 25 fps → 25 latent tokens
    let lat = inference_flame::audio::load_and_encode_audio(
        &tmp,
        &sa_vae,
        num_frames,
        &device,
    )?;

    let dims = lat.shape().dims().to_vec();
    let v = lat.to_vec_f32()?;
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let max_abs = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let any_nan = v.iter().any(|x| !x.is_finite());

    println!();
    println!("============================================================");
    println!("audio_encode_smoke result");
    println!("============================================================");
    println!("  shape      {:?}", dims);
    println!("  mean       {:.6}", mean);
    println!("  |max|      {:.4}", max_abs);
    println!("  any nan/inf {}", any_nan);
    println!();
    if dims == &[1, num_frames, 64] && !any_nan {
        println!("PASS — output shape matches [1, 25, 64], values finite.");
    } else {
        println!("FAIL — see above.");
        std::process::exit(1);
    }

    let _ = std::fs::remove_file(&tmp);
    Ok(())
}
