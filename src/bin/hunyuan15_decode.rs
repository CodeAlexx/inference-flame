//! HunyuanVideo 1.5 — Stage 3 (VAE decode → MP4).
//!
//! Pure-Rust replacement for `inference-flame/scripts/hunyuan15_decode.py`.
//! Loads HunyuanVideo VAE weights, reads the latent saved by Stage 2
//! (`hunyuan15_gen`), runs the Rust 3D causal-VAE decoder, then muxes the
//! resulting frames to an H.264/MP4 via ffmpeg.
//!
//! CLI:
//!     hunyuan15_decode [<latents.safetensors>] [<out.mp4>]
//!
//! Defaults match the Python script's defaults so existing pipelines drop in.

use std::path::PathBuf;
use std::time::Instant;

use flame_core::{global_cuda_device, DType};

use inference_flame::vae::HunyuanVaeDecoder;

const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/hunyuan_video_vae_bf16.safetensors";
const VAE_LATENT_CHANNELS: usize = 16;
const SCALING_FACTOR: f32 = 0.476986;
const SAMPLE_FPS: f32 = 24.0;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let latents_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/hunyuan15_latents.safetensors".to_string()
    });
    let out_path = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/hunyuan15_output.mp4".to_string()
    }));

    println!("=== HunyuanVideo 1.5 — Stage 3 (Rust VAE decode) ===");
    println!("Latents: {latents_path}");
    println!("Output:  {}", out_path.display());
    println!();

    // ------------------------------------------------------------------
    // Load latent (saved by hunyuan15_gen)
    // ------------------------------------------------------------------
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&latents_path),
        &device,
    )?;
    let mut latent = tensors
        .get("latent")
        .ok_or_else(|| anyhow::anyhow!("Missing 'latent' tensor in {latents_path}"))?
        .clone();
    drop(tensors);

    let dims = latent.shape().dims().to_vec();
    println!("  latent: {dims:?}  dtype={:?}", latent.dtype());

    if dims.len() != 5 {
        anyhow::bail!("expected 5D latent [B,C,T,H,W], got {dims:?}");
    }

    // The DiT outputs `out_channels=32` channels. The VAE only consumes 16. Take
    // the first 16 (matches the Python script's behavior).
    if dims[1] > VAE_LATENT_CHANNELS {
        println!(
            "  Trimming latent channels {} → {}",
            dims[1], VAE_LATENT_CHANNELS
        );
        latent = latent.narrow(1, 0, VAE_LATENT_CHANNELS)?;
    }

    // Undo scaling: latent / scaling_factor.
    let latent = if latent.dtype() == DType::BF16 {
        latent.mul_scalar(1.0 / SCALING_FACTOR)?
    } else {
        latent
            .to_dtype(DType::BF16)?
            .mul_scalar(1.0 / SCALING_FACTOR)?
    };

    // ------------------------------------------------------------------
    // Load VAE decoder (pure Rust)
    // ------------------------------------------------------------------
    println!("\n--- Loading HunyuanVideo VAE (Rust) ---");
    let t0 = Instant::now();
    let decoder = HunyuanVaeDecoder::load(VAE_PATH, &device)?;
    println!(
        "  VAE loaded in {:.1}s ({} weight tensors)",
        t0.elapsed().as_secs_f32(),
        decoder.num_weights()
    );

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------
    println!("\n--- VAE decode ---");
    let t0 = Instant::now();
    let decoded = decoder.decode(&latent)?;
    let out_dims = decoded.shape().dims().to_vec();
    println!(
        "  Decoded: {out_dims:?} in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // ------------------------------------------------------------------
    // Convert to RGB u8 frames + mux
    // ------------------------------------------------------------------
    if out_dims.len() != 5 {
        anyhow::bail!("decoder output not 5D: {out_dims:?}");
    }
    let (b, c, f_count, h, w) = (
        out_dims[0],
        out_dims[1],
        out_dims[2],
        out_dims[3],
        out_dims[4],
    );
    if b != 1 {
        anyhow::bail!("expected batch=1, got {b}");
    }
    if c != 3 {
        anyhow::bail!("expected 3 RGB channels, got {c}");
    }

    // mux::video_tensor_to_rgb_u8 wants [3, F, H, W] flat F32 in [-1, 1].
    let video_f32 = decoded
        .squeeze(Some(0))?       // [3, F, H, W]
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    let frames_u8 = inference_flame::mux::video_tensor_to_rgb_u8(&video_f32, f_count, h, w);

    // Silent stereo audio at the same nominal rate as the video runtime so
    // the muxer is happy — the file is video-only by intent.
    let audio_sample_rate: u32 = 16000;
    let total_audio_samples =
        ((f_count as f32 / SAMPLE_FPS) * audio_sample_rate as f32).ceil() as usize;
    let silent_audio = vec![0i16; total_audio_samples * 2];

    println!(
        "\n--- Muxing {} frames @ {} fps to {} ---",
        f_count,
        SAMPLE_FPS,
        out_path.display()
    );
    let t0 = Instant::now();
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    inference_flame::mux::write_mp4(
        &out_path,
        &frames_u8,
        f_count,
        w,
        h,
        SAMPLE_FPS,
        &silent_audio,
        audio_sample_rate,
    )
    .map_err(|e| anyhow::anyhow!("write_mp4: {e}"))?;
    println!("  ffmpeg done in {:.1}s", t0.elapsed().as_secs_f32());

    println!();
    println!("============================================================");
    println!("VIDEO SAVED: {}", out_path.display());
    println!("  {} frames, {}x{}, {} fps", f_count, w, h, SAMPLE_FPS);
    println!("Total time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
