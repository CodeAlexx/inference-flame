//! Motif-Video Stage 3: VAE decode latents → PNG frames (MP4 later).
//!
//! Uses Wan 2.1 VAE decoder (reuses `wan21_vae::Wan21VaeDecoder`).
//! Saves each video frame as a numbered PNG. External ffmpeg can compose MP4.

use std::time::Instant;
use flame_core::{global_cuda_device, DType, Tensor};
use image::{ImageBuffer, Rgb};
use inference_flame::vae::wan21_vae::Wan21VaeDecoder;

const BASE_DIR: &str = "/home/alex/.serenity/models/checkpoints/motif-video-2b";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let latents_path = args.get(1).cloned()
        .unwrap_or_else(|| "/home/alex/EriDiffusion/inference-flame/output/motif_latents.safetensors".into());
    let out_dir = args.get(2).cloned()
        .unwrap_or_else(|| "/home/alex/EriDiffusion/inference-flame/output/motif_frames/".into());
    let vae_path = std::env::var("MOTIF_VAE").unwrap_or_else(|_|
        format!("{}/vae/diffusion_pytorch_model.safetensors", BASE_DIR));

    println!("=== Motif-Video Stage 3 (VAE decode → PNG frames) ===");
    println!("  Latents: {}", latents_path);
    println!("  OutDir:  {}", out_dir);
    println!("  VAE:     {}", vae_path);

    // Load latents
    let tensors = flame_core::serialization::load_file(std::path::Path::new(&latents_path), &device)?;
    let latents = tensors.get("latents")
        .ok_or_else(|| anyhow::anyhow!("Missing 'latents'"))?.clone();
    let height = tensors.get("height").unwrap().to_vec_f32()?[0] as usize;
    let width = tensors.get("width").unwrap().to_vec_f32()?[0] as usize;
    let num_frames = tensors.get("num_frames").unwrap().to_vec_f32()?[0] as usize;
    println!("  Latents shape: {:?}, H={}, W={}, frames={}", latents.shape().dims(), height, width, num_frames);

    let latents = if latents.dtype() == DType::BF16 { latents } else { latents.to_dtype(DType::BF16)? };

    // Load VAE decoder
    println!("--- Loading Wan 2.1 VAE decoder ---");
    let t0 = Instant::now();
    let vae = Wan21VaeDecoder::load(&vae_path, &device)?;
    println!("  VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Decode: [B, 16, T_lat, H_lat, W_lat] → [B, 3, T, H, W]
    println!("--- Decoding ---");
    let t0 = Instant::now();
    let image = vae.decode(&latents)?;
    println!("  Decoded: {:?} in {:.1}s", image.shape().dims(), t0.elapsed().as_secs_f32());

    // image shape: [B=1, 3, T, H, W] BF16
    let dims = image.shape().dims().to_vec();
    let t_out = dims[2];
    let h_out = dims[3];
    let w_out = dims[4];
    println!("  Output frames: {}, {}×{}", t_out, h_out, w_out);

    std::fs::create_dir_all(&out_dir).ok();

    // Pull to CPU as F32 — to_vec_f32 handles BF16 internally.
    let img_flat = image.to_vec_f32()?;
    // Layout: [B=1, 3, T, H, W]; stride for (c, t, y, x) = c*T*H*W + t*H*W + y*W + x
    let chan_stride = t_out * h_out * w_out;

    // Wan VAE output is BGR per channel (motif_vae_decode_bridge.py swaps [2,1,0]).
    // Map VAE channel → PNG byte: VAE 0 (B) → byte 2, VAE 1 (G) → byte 1, VAE 2 (R) → byte 0.
    const VAE_TO_RGB: [usize; 3] = [2, 1, 0];
    for frame_idx in 0..t_out {
        let mut rgb = vec![0u8; h_out * w_out * 3];
        for y in 0..h_out {
            for x in 0..w_out {
                for c in 0..3 {
                    let src_idx = c * chan_stride + frame_idx * h_out * w_out + y * w_out + x;
                    let v = img_flat[src_idx].clamp(-1.0, 1.0);
                    let u = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                    rgb[(y * w_out + x) * 3 + VAE_TO_RGB[c]] = u;
                }
            }
        }
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_vec(w_out as u32, h_out as u32, rgb)
                .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
        let path = format!("{}/frame_{:04}.png", out_dir.trim_end_matches('/'), frame_idx);
        buf.save(&path)?;
    }
    println!("Saved {} PNG frames to {} ({:.1}s total)",
             t_out, out_dir, t_total.elapsed().as_secs_f32());
    println!();
    println!("To assemble MP4 (external ffmpeg):");
    println!("  ffmpeg -framerate 24 -i {}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4",
             out_dir.trim_end_matches('/'));
    Ok(())
}
