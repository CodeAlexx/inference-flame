//! LTX-2.3 audio+video decode — latents → MP4 with audio.
//!
//! Takes the denoised latents saved by `ltx2_generate_av` (or `ltx2_two_stage`)
//! and produces a final MP4:
//!
//!   1. Load video latents from safetensors
//!   2. Video VAE decode → RGB frames
//!   3. Load audio latents from safetensors
//!   4. Audio VAE decode → mel spectrogram
//!   5. BigVGAN vocoder → 16 kHz stereo waveform
//!   6. ffmpeg mux → MP4 (with aresample 16→48 kHz)
//!
//! Audio decode failures are non-fatal: if audio VAE or vocoder fails,
//! the video is still saved as a silent MP4.
//!
//! Env vars:
//!   LTX2_VIDEO_LATENTS — override video latents path
//!   LTX2_AUDIO_LATENTS — override audio latents path
//!   LTX2_OUTPUT        — override output MP4 path
//!   LTX2_CHECKPOINT    — override VAE checkpoint path
//!   LTX2_FPS           — override frame rate (default: 25)
//!   LTX2_VIDEO_ONLY    — set to 1 to skip audio decode entirely
//!   LTX2_FRAMES_DIR    — if set, also save individual PNG frames to this dir

use flame_core::{global_cuda_device, DType};
use inference_flame::mux;
use inference_flame::vae::{LTX2AudioVaeDecoder, LTX2VaeDecoder, LTX2Vocoder};
use std::time::Instant;

const DEFAULT_OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";
const DEFAULT_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const DEFAULT_FPS: f32 = 25.0;
const VOCODER_SAMPLE_RATE: u32 = 16000;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let output_dir = env_or("LTX2_OUTPUT_DIR", DEFAULT_OUTPUT_DIR);
    let video_latents_path = env_or(
        "LTX2_VIDEO_LATENTS",
        &format!("{output_dir}/ltx2_av_video_latents.safetensors"),
    );
    let audio_latents_path = env_or(
        "LTX2_AUDIO_LATENTS",
        &format!("{output_dir}/ltx2_av_audio_latents.safetensors"),
    );
    let output_path = env_or(
        "LTX2_OUTPUT",
        &format!("{output_dir}/ltx2_av_output.mp4"),
    );
    let checkpoint = env_or("LTX2_CHECKPOINT", DEFAULT_CHECKPOINT);
    let fps: f32 = env_or("LTX2_FPS", &format!("{DEFAULT_FPS}"))
        .parse()
        .unwrap_or(DEFAULT_FPS);
    let video_only = std::env::var("LTX2_VIDEO_ONLY").map_or(false, |v| v == "1");
    let frames_dir = std::env::var("LTX2_FRAMES_DIR").ok();

    println!("============================================================");
    println!("LTX-2.3 Audio+Video Decode — Pure Rust");
    println!("============================================================");
    println!("  Video latents: {video_latents_path}");
    println!("  Audio latents: {audio_latents_path}");
    println!("  Checkpoint:    {checkpoint}");
    println!("  Output:        {output_path}");
    println!("  FPS:           {fps}");
    if video_only {
        println!("  Mode:          VIDEO ONLY (audio skipped)");
    }
    if let Some(ref dir) = frames_dir {
        println!("  Frames dir:    {dir}");
    }

    let device = global_cuda_device();

    // ==================================================================
    // Stage 1: Load and decode video latents
    // ==================================================================
    println!("\n--- Stage 1: Video VAE decode ---");
    let t0 = Instant::now();

    println!("  Loading video latents...");
    let video_tensors = flame_core::serialization::load_file_filtered(
        std::path::Path::new(&video_latents_path),
        &device,
        |k| k == "latents",
    )?;
    let video_latents_raw = video_tensors
        .get("latents")
        .ok_or_else(|| anyhow::anyhow!("No 'latents' key in {video_latents_path}"))?;
    let video_latents = if video_latents_raw.dtype() != DType::BF16 {
        video_latents_raw.to_dtype(DType::BF16)?
    } else {
        video_latents_raw.clone()
    };
    let vl_dims = video_latents.shape().dims().to_vec();
    println!(
        "  Video latents: {:?} {:?}",
        vl_dims,
        video_latents.dtype()
    );

    println!("  Loading video VAE...");
    let video_vae = LTX2VaeDecoder::from_file(&checkpoint, &device)?;
    println!("  VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t_decode = Instant::now();
    let video_decoded = video_vae.decode(&video_latents)?;
    println!(
        "  Decoded to {:?} in {:.1}s",
        video_decoded.shape().dims(),
        t_decode.elapsed().as_secs_f32()
    );
    drop(video_vae);

    // [B, 3, F, H, W] → F32 CPU for u8 quantization
    let v_dims = video_decoded.shape().dims().to_vec();
    let (f_out, h_out, w_out) = (v_dims[2], v_dims[3], v_dims[4]);
    let video_f32 = video_decoded.to_dtype(DType::F32)?.to_vec()?;
    drop(video_decoded);
    let rgb_u8 = mux::video_tensor_to_rgb_u8(&video_f32, f_out, h_out, w_out);
    drop(video_f32);
    println!("  RGB24 ready: {f_out} frames, {w_out}x{h_out}");

    // Optionally save individual PNG frames
    if let Some(ref dir) = frames_dir {
        println!("  Saving {f_out} frames to {dir}/...");
        std::fs::create_dir_all(dir)?;
        for frame_idx in 0..f_out {
            let offset = frame_idx * h_out * w_out * 3;
            let frame_bytes = &rgb_u8[offset..offset + h_out * w_out * 3];
            let img = image::RgbImage::from_raw(w_out as u32, h_out as u32, frame_bytes.to_vec())
                .ok_or_else(|| anyhow::anyhow!("Failed to create image for frame {frame_idx}"))?;
            let frame_path = format!("{dir}/frame_{frame_idx:05}.png");
            img.save(&frame_path)?;
        }
        println!("  Frames saved.");
    }

    // ==================================================================
    // Stage 2: Audio decode (non-fatal on failure)
    // ==================================================================
    let audio_result: Option<(Vec<i16>, u32)> = if video_only {
        println!("\n--- Audio decode SKIPPED (LTX2_VIDEO_ONLY=1) ---");
        None
    } else {
        println!("\n--- Stage 2: Audio VAE decode ---");
        match decode_audio(&audio_latents_path, &checkpoint, &device) {
            Ok((pcm, sr)) => {
                println!("  Audio decode complete: {} stereo samples @ {sr}Hz", pcm.len() / 2);
                Some((pcm, sr))
            }
            Err(e) => {
                eprintln!("  WARNING: Audio decode failed: {e}");
                eprintln!("  Continuing with silent video.");
                None
            }
        }
    };

    // ==================================================================
    // Stage 3: MP4 mux
    // ==================================================================
    println!("\n--- Stage 3: MP4 mux → {output_path} ---");
    let t0 = Instant::now();

    match audio_result {
        Some((ref pcm, sr)) => {
            mux::write_mp4(
                std::path::Path::new(&output_path),
                &rgb_u8,
                f_out,
                w_out,
                h_out,
                fps,
                pcm,
                sr,
            )?;
        }
        None => {
            // Silent video: generate empty audio
            let silence: Vec<i16> = vec![0i16; 2]; // minimal stereo silence
            mux::write_mp4(
                std::path::Path::new(&output_path),
                &rgb_u8,
                f_out,
                w_out,
                h_out,
                fps,
                &silence,
                VOCODER_SAMPLE_RATE,
            )?;
        }
    }
    println!("  ffmpeg done in {:.1}s", t0.elapsed().as_secs_f32());

    println!("\n============================================================");
    println!("DONE: {output_path}");
    println!("  Video: {f_out} frames @ {fps}fps ({w_out}x{h_out})");
    if let Some((ref pcm, sr)) = audio_result {
        let duration = pcm.len() as f32 / 2.0 / sr as f32;
        println!("  Audio: {:.2}s stereo @ {sr}Hz", duration);
    } else {
        println!("  Audio: none (silent)");
    }
    println!("  Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}

/// Decode audio latents → mel → waveform → interleaved PCM i16.
///
/// Returns `(pcm_i16, sample_rate)` on success.
fn decode_audio(
    audio_latents_path: &str,
    checkpoint: &str,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(Vec<i16>, u32)> {
    let t0 = Instant::now();

    // Load audio latents
    println!("  Loading audio latents...");
    let audio_tensors = flame_core::serialization::load_file_filtered(
        std::path::Path::new(audio_latents_path),
        device,
        |k| k == "latents",
    )?;
    let audio_latents_raw = audio_tensors
        .get("latents")
        .ok_or_else(|| anyhow::anyhow!("No 'latents' key in {audio_latents_path}"))?;
    let audio_latents = if audio_latents_raw.dtype() != DType::BF16 {
        audio_latents_raw.to_dtype(DType::BF16)?
    } else {
        audio_latents_raw.clone()
    };
    println!(
        "  Audio latents: {:?} {:?}",
        audio_latents.shape().dims(),
        audio_latents.dtype()
    );

    // Audio VAE: latent → mel spectrogram
    println!("  Loading audio VAE...");
    let audio_vae = LTX2AudioVaeDecoder::from_file(checkpoint, device)?;
    println!("  Audio VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t_decode = Instant::now();
    let mel = audio_vae.decode(&audio_latents)?;
    println!(
        "  Mel: {:?} in {:.1}s",
        mel.shape().dims(),
        t_decode.elapsed().as_secs_f32()
    );
    drop(audio_vae);

    // Vocoder: mel → 16 kHz stereo waveform
    println!("  Loading vocoder...");
    let t_voc = Instant::now();
    let vocoder = LTX2Vocoder::from_file(checkpoint, device, "vocoder")?;
    println!("  Vocoder loaded in {:.1}s", t_voc.elapsed().as_secs_f32());

    let t_fwd = Instant::now();
    let waveform = vocoder.forward(&mel)?;
    println!(
        "  Waveform: {:?} in {:.1}s",
        waveform.shape().dims(),
        t_fwd.elapsed().as_secs_f32()
    );
    drop(vocoder);
    drop(mel);

    // [B, 2, samples] → interleaved i16 PCM
    let wf_dims = waveform.shape().dims().to_vec();
    let n_channels = wf_dims[1];
    let n_samples = wf_dims[2];
    let waveform_f32 = waveform.to_dtype(DType::F32)?.to_vec()?;
    drop(waveform);
    let pcm_i16 = mux::audio_tensor_to_pcm_i16(&waveform_f32, n_channels, n_samples, true);
    drop(waveform_f32);

    let duration = n_samples as f32 / VOCODER_SAMPLE_RATE as f32;
    println!(
        "  PCM: {} stereo samples ({:.2}s @ {}Hz)",
        pcm_i16.len() / 2,
        duration,
        VOCODER_SAMPLE_RATE
    );

    Ok((pcm_i16, VOCODER_SAMPLE_RATE))
}
