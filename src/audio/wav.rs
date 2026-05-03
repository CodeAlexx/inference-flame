//! WAV file loader. Produces interleaved f32 samples in `[-1.0, 1.0]` plus
//! the native sample rate and channel count.
//!
//! Supports 16-bit PCM, 24-bit PCM, 32-bit PCM, and 32-bit float WAV — the
//! formats produced by every common DAW / Audacity export. For 8-bit PCM
//! (rare) we still load it but precision is degraded; user should re-export.

use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavReader};
use std::path::Path;

/// Probe a WAV header and return `(duration_seconds, sample_rate, channels)`
/// without decoding any samples. Useful when you need to derive video length
/// from audio length BEFORE running the full load+encode pipeline.
pub fn probe_duration(path: &Path) -> Result<(f64, u32, u16)> {
    let reader = WavReader::open(path)
        .map_err(|e| anyhow!("hound::WavReader::open({}): {e}", path.display()))?;
    let spec = reader.spec();
    let frames_per_channel = reader.duration() as f64; // hound returns frames (per channel)
    let dur_sec = frames_per_channel / spec.sample_rate as f64;
    Ok((dur_sec, spec.sample_rate, spec.channels))
}

/// Returns `(samples, sample_rate, channels)`.
/// `samples` is interleaved across channels (frame-major: `[ch0_t0, ch1_t0, ch0_t1, ch1_t1, ...]`).
pub fn load_wav_f32(path: &Path) -> Result<(Vec<f32>, u32, u16)> {
    let mut reader = WavReader::open(path)
        .map_err(|e| anyhow!("hound::WavReader::open({}): {e}", path.display()))?;
    let spec = reader.spec();
    let bits = spec.bits_per_sample;
    let fmt = spec.sample_format;
    let rate = spec.sample_rate;
    let chans = spec.channels;

    // Convert to f32 in [-1, 1] regardless of source format.
    let samples: Vec<f32> = match (fmt, bits) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .map_err(|e| anyhow!("read f32 samples: {e}"))?,
        (SampleFormat::Int, 16) => {
            let scale = 1.0f32 / (i16::MAX as f32);
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| (v as f32) * scale))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| anyhow!("read i16 samples: {e}"))?
        }
        (SampleFormat::Int, 24) | (SampleFormat::Int, 32) => {
            let max_abs = (1i64 << (bits - 1)) as f32;
            let scale = 1.0f32 / max_abs;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f32) * scale))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| anyhow!("read i32 samples: {e}"))?
        }
        (SampleFormat::Int, 8) => {
            let scale = 1.0f32 / (i8::MAX as f32);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| ((v - 128) as f32) * scale))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| anyhow!("read 8-bit samples: {e}"))?
        }
        (sf, b) => anyhow::bail!("unsupported WAV format: {:?} {}-bit", sf, b),
    };

    Ok((samples, rate, chans))
}
