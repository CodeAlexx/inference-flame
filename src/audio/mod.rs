//! Audio preprocessing for MagiHuman talking-head conditioning.
//!
//! Provides pure-Rust:
//!   * WAV loading (`wav.rs`)
//!   * Sinc resampling to the model's expected 51,200 Hz (`resample.rs`)
//!   * Stable Audio Open 1.0 Oobleck VAE encoder (`oobleck_encoder.rs`)
//!     mirroring `inference/model/sa_audio/sa_audio_module.py:OobleckEncoder`
//!     with the variational bottleneck.
//!
//! End-to-end entry: [`load_and_encode_audio`] reads a WAV, resamples to
//! 51,200 Hz, expands mono to stereo, encodes with the SA Open VAE, and
//! returns the audio latent reshaped to `[1, num_frames, 64]` matching
//! MagiHuman's audio_in_channels = 64.

pub mod oobleck_encoder;
pub mod resample;
pub mod wav;

pub use oobleck_encoder::OobleckVaeEncoder;

use anyhow::{anyhow, Result};
use flame_core::{DType, Shape, Tensor};
use std::path::Path;
use std::sync::Arc;

/// Sample rate the SA Open VAE was trained on. Resample everything to this
/// before encoding. (per inference/pipeline/video_process.py:170)
pub const TARGET_SAMPLE_RATE: u32 = 51_200;

/// Final channel count after the VAE bottleneck split (latent_dim/2).
/// Matches MagiHuman config `audio_in_channels = 64`.
pub const AUDIO_LATENT_CHANNELS: usize = 64;

/// Total downsampling ratio of the SA Open 1.0 encoder
/// (strides [2, 4, 4, 8, 8] → 2 * 4 * 4 * 8 * 8 = 2048).
pub const TOTAL_DOWNSAMPLE: usize = 2048;

/// Load a WAV file, resample, encode through the VAE, return audio_lat
/// shape `[1, num_frames, 64]` ready to drop into MagiHuman's pack_inputs.
///
/// `num_frames` is the expected count from MagiHuman's video schedule
/// (`(seconds * fps - 1) / 4 + 1` matching the latent T dim). The encoder's
/// natural output for `seconds * 51_200` samples is exactly that count when
/// `seconds * fps = num_frames * 4 + 1` (the model's I2V relation).
///
/// If the encoded length doesn't match `num_frames`, we trim or pad with
/// edge replication — matches creator's `merge_overlapping_vae_features` /
/// `final_target_len = ceil(total_samples * latent_to_audio_ratio)` behavior
/// for short clips that fit in one window.
pub fn load_and_encode_audio(
    wav_path: &Path,
    sa_vae_weights: &Path,
    num_frames: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    // 1. Load WAV → mono / stereo f32 samples + native rate.
    let (samples, native_rate, native_channels) = wav::load_wav_f32(wav_path)
        .map_err(|e| anyhow!("WAV load {}: {e}", wav_path.display()))?;
    eprintln!(
        "[audio] loaded {} ({} ch, {} Hz, {:.2} sec)",
        wav_path.display(),
        native_channels,
        native_rate,
        samples.len() as f32 / native_rate as f32 / native_channels as f32,
    );

    // 2. Down-mix to mono (creator does mono → stereo expand later) if input
    //    is multi-channel. Average across channels.
    let mono: Vec<f32> = if native_channels == 1 {
        samples
    } else {
        let n = native_channels as usize;
        let frames = samples.len() / n;
        let mut out = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut sum = 0.0f32;
            for c in 0..n {
                sum += samples[i * n + c];
            }
            out.push(sum / n as f32);
        }
        out
    };

    // 3. Resample to 51,200 Hz if needed.
    let resampled = if native_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        resample::resample_to(&mono, native_rate, TARGET_SAMPLE_RATE)
            .map_err(|e| anyhow!("resample {} → {} Hz: {e}", native_rate, TARGET_SAMPLE_RATE))?
    };
    eprintln!(
        "[audio] after resample: {} samples at {} Hz ({:.2} sec)",
        resampled.len(),
        TARGET_SAMPLE_RATE,
        resampled.len() as f32 / TARGET_SAMPLE_RATE as f32,
    );

    // 4. Length-fit to encoder window: pad-or-trim to `num_frames * 2048`
    //    samples so the encoder's natural output is exactly `num_frames`
    //    latent tokens.
    let target_samples = num_frames * TOTAL_DOWNSAMPLE;
    let fitted: Vec<f32> = if resampled.len() >= target_samples {
        resampled[..target_samples].to_vec()
    } else {
        let mut v = resampled.clone();
        v.resize(target_samples, 0.0);
        v
    };

    // 5. Build audio tensor: stereo by mono replication → [2, T_samples] → [1, 2, T_samples].
    let mut interleaved = Vec::with_capacity(fitted.len() * 2);
    for &s in &fitted {
        interleaved.push(s);
        interleaved.push(s);
    }
    // [T, 2] interleaved layout; we want [1, 2, T] for the Conv1d.
    // Convert: stack both channels into separate buffers.
    let mut chan_data = Vec::with_capacity(2 * fitted.len());
    chan_data.extend_from_slice(&fitted);
    chan_data.extend_from_slice(&fitted);
    let audio_input = Tensor::from_vec(
        chan_data,
        Shape::from_dims(&[1, 2, fitted.len()]),
        device.clone(),
    )?;

    // 6. Load encoder + bottleneck, run forward.
    let encoder = OobleckVaeEncoder::load(sa_vae_weights, device)
        .map_err(|e| anyhow!("OobleckVaeEncoder load {}: {e}", sa_vae_weights.display()))?;
    let audio_input_bf16 = audio_input.to_dtype(DType::BF16)?;
    let pre_bottleneck = encoder.encode(&audio_input_bf16)?;
    // Bottleneck: split last-dim/2 → mean (no sampling at inference; use mean).
    let lat = encoder.bottleneck_mean(&pre_bottleneck)?;
    eprintln!("[audio] encoded latent: {:?} {:?}", lat.shape().dims(), lat.dtype());
    drop(encoder);

    // 7. Reshape from [1, 64, num_frames] (Conv1d output convention) to
    //    [1, num_frames, 64] expected by MagiHuman's pack_inputs.
    let dims = lat.shape().dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != AUDIO_LATENT_CHANNELS as usize {
        anyhow::bail!(
            "audio latent shape mismatch: got {:?}, expected [1, {}, N]",
            dims, AUDIO_LATENT_CHANNELS
        );
    }
    let lat_t = lat.permute(&[0, 2, 1])?.contiguous()?.to_dtype(DType::F32)?;
    eprintln!("[audio] final shape: {:?}", lat_t.shape().dims());
    Ok(lat_t)
}
