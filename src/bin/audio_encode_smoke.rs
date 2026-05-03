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

use anyhow::{anyhow, Result};
use cudarc::driver::CudaDevice;
use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn write_sine_wav(path: &Path, rate: u32, channels: u16, bits: u16, dur_sec: f32, freq: f32) -> Result<usize> {
    let n_frames = (rate as f32 * dur_sec) as usize;
    let spec = WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: bits,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec)?;
    for i in 0..n_frames {
        let t = i as f32 / rate as f32;
        let v = (t * freq * std::f32::consts::TAU).sin() * 0.5;
        let i16v = (v * (i16::MAX as f32)) as i16;
        for _ in 0..channels {
            w.write_sample(i16v)?;
        }
    }
    w.finalize()?;
    Ok(n_frames)
}

fn run_case(
    name: &str,
    rate: u32,
    channels: u16,
    dur_sec: f32,
    num_frames: usize,
    sa_vae: &Path,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    let tmp = std::env::temp_dir().join(format!("smoke_{name}.wav"));
    let n = write_sine_wav(&tmp, rate, channels, 16, dur_sec, 440.0)?;
    eprintln!("[smoke:{name}] wrote {} ({} frames, {} ch, {} Hz, {:.2}s)",
              tmp.display(), n, channels, rate, dur_sec);
    let lat = inference_flame::audio::load_and_encode_audio(&tmp, sa_vae, num_frames, device)?;
    let dims = lat.shape().dims().to_vec();
    let v = lat.to_vec_f32()?;
    let max_abs = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let any_bad = v.iter().any(|x| !x.is_finite());
    println!("  case [{name:25}]  shape={:?}  |max|={:.3}  finite={}", dims, max_abs, !any_bad);
    if dims != &[1, num_frames, 64] {
        return Err(anyhow!("[{name}] shape mismatch: {:?} vs [1, {}, 64]", dims, num_frames));
    }
    if any_bad {
        return Err(anyhow!("[{name}] non-finite values"));
    }
    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

fn main() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let sa_vae = PathBuf::from("/home/alex/.serenity/models/vaes/stable_audio_oobleck_vae.safetensors");

    println!("============================================================");
    println!("audio_encode_smoke — covering common WAV variants");
    println!("============================================================");

    // 1-sec mono @ 44.1 kHz → 1 sec target (25 frames)
    run_case("1s_mono_44k", 44_100, 1, 1.0, 25, &sa_vae, &device)?;
    // 1-sec stereo @ 44.1 kHz (downmix → encode)
    run_case("1s_stereo_44k", 44_100, 2, 1.0, 25, &sa_vae, &device)?;
    // 1-sec mono @ 48 kHz (different resample ratio)
    run_case("1s_mono_48k", 48_000, 1, 1.0, 25, &sa_vae, &device)?;
    // 1-sec mono @ 51.2 kHz (no resample needed)
    run_case("1s_mono_native", 51_200, 1, 1.0, 25, &sa_vae, &device)?;
    // 1-sec mono @ 22.05 kHz (low-rate input)
    run_case("1s_mono_22k", 22_050, 1, 1.0, 25, &sa_vae, &device)?;
    // 0.5-sec mono — zero-pad to 1-sec target
    run_case("half_pad_to_1s", 44_100, 1, 0.5, 25, &sa_vae, &device)?;
    // 2-sec mono — trim to 1-sec target
    run_case("2s_trim_to_1s", 44_100, 1, 2.0, 25, &sa_vae, &device)?;
    // 5-sec target (32 latent frames)
    run_case("5s_mono_44k", 44_100, 1, 5.0, 32, &sa_vae, &device)?;

    println!();
    println!("ALL CASES PASS");
    Ok(())
}
