//! MP4 muxing via `ffmpeg` subprocess — pure Rust replacement for Python's
//! `ltx_trainer.video_utils.save_video` (which uses PyAV).
//!
//! Mirrors PyAV's encoder options: libx264 CRF 18, yuv420p, AAC stereo audio.
//! Feeds raw RGB24 frames and int16 PCM samples to ffmpeg via temp files
//! (stdin pipe-ing both streams simultaneously needs named pipes; temp files
//! are simpler and the IO cost is negligible for video-scale outputs).
//!
//! Expected inputs:
//!   - frames: `[F, H, W, 3]` in u8, row-major (interleaved RGB)
//!   - audio:  interleaved stereo `[samples * 2]` in i16
//!
//! Uses `which ffmpeg` — ffmpeg binary must be in $PATH.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

/// Normalize a decoded video tensor `[B=1, 3, F, H, W]` in BF16/F32 range
/// `[-1, 1]` (or `[0, 1]`) into `[F, H, W, 3]` u8 frames.
pub fn video_tensor_to_rgb_u8(
    data: &[f32],
    f: usize,
    h: usize,
    w: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; f * h * w * 3];
    // input layout: [3, F, H, W] (channel major)
    for frame in 0..f {
        for hh in 0..h {
            for ww in 0..w {
                for c in 0..3 {
                    let src_idx = c * f * h * w + frame * h * w + hh * w + ww;
                    let val = data[src_idx];
                    // Map [-1, 1] → [0, 255]
                    let mapped = ((val.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
                    let dst_idx = (frame * h * w + hh * w + ww) * 3 + c;
                    out[dst_idx] = mapped;
                }
            }
        }
    }
    out
}

/// Convert a float stereo waveform `[2, samples]` or `[samples, 2]` in
/// range `[-1, 1]` into interleaved s16 PCM `[sample0_L, sample0_R, ...]`.
pub fn audio_tensor_to_pcm_i16(
    data: &[f32],
    n_channels: usize,
    n_samples: usize,
    channels_first: bool,
) -> Vec<i16> {
    assert!(n_channels <= 2, "stereo or mono only");
    let mut out = Vec::with_capacity(n_samples * 2);
    for s in 0..n_samples {
        for c in 0..2 {
            let src_c = c.min(n_channels - 1);
            let idx = if channels_first {
                src_c * n_samples + s
            } else {
                s * n_channels + src_c
            };
            let val = (data[idx].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.push(val);
        }
    }
    out
}

/// Mux RGB frames + stereo PCM into an MP4 via ffmpeg subprocess.
///
/// `frames` — `F * H * W * 3` u8 bytes
/// `audio`  — `samples * 2` i16 samples (interleaved LR)
pub fn write_mp4(
    out_path: &Path,
    frames: &[u8],
    frame_count: usize,
    width: usize,
    height: usize,
    fps: f32,
    audio: &[i16],
    audio_sample_rate: u32,
) -> std::io::Result<()> {
    assert_eq!(frames.len(), frame_count * width * height * 3);
    assert_eq!(audio.len() % 2, 0);

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write the two raw streams to temp files. Using named pipes (mkfifo)
    // would avoid disk IO but is platform-specific and deadlock-prone; for
    // LTX-2.3 outputs (<100 MB raw), tempfile IO is negligible.
    let tmp_dir = std::env::temp_dir();
    let pid = std::process::id();
    let frames_path = tmp_dir.join(format!("ltx2_mux_{pid}_frames.rgb"));
    let audio_path = tmp_dir.join(format!("ltx2_mux_{pid}_audio.s16le"));

    {
        let mut f = File::create(&frames_path)?;
        f.write_all(frames)?;
    }
    {
        let mut f = File::create(&audio_path)?;
        // Convert i16 → little-endian bytes.
        let mut buf = Vec::with_capacity(audio.len() * 2);
        for s in audio {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        f.write_all(&buf)?;
    }

    // ffmpeg invocation mirrors PyAV: libx264 CRF 18, yuv420p, AAC stereo.
    // Audio is resampled to the target_audio_sample_rate inside ffmpeg via
    // `-af aresample=...` — this lets the caller feed 16 kHz vocoder output
    // directly and have ffmpeg upsample to 48 kHz without a separate Rust BWE.
    let target_audio_rate: u32 = 48000;
    let audio_filter = format!("aresample={target_audio_rate}");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("rgb24")
        .arg("-s").arg(format!("{width}x{height}"))
        .arg("-r").arg(format!("{fps}"))
        .arg("-i").arg(&frames_path)
        .arg("-f").arg("s16le")
        .arg("-ar").arg(format!("{audio_sample_rate}"))
        .arg("-ac").arg("2")
        .arg("-i").arg(&audio_path)
        .arg("-c:v").arg("libx264")
        .arg("-crf").arg("18")
        .arg("-pix_fmt").arg("yuv420p")
        .arg("-c:a").arg("aac")
        .arg("-af").arg(&audio_filter)
        .arg("-shortest")
        .arg(out_path)
        .status()?;

    // Clean up temp files regardless of success.
    let _ = std::fs::remove_file(&frames_path);
    let _ = std::fs::remove_file(&audio_path);

    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("ffmpeg exited with status {status}"),
        ));
    }
    Ok(())
}
