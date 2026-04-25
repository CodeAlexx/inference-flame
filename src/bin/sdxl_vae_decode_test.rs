//! Quick test: decode an SDXL latent through LdmVAEDecoder.
//! Lets us swap VAE checkpoints to isolate VAE-decode bugs.

use flame_core::{global_cuda_device, DType, Result, Tensor};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let latent = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/sdxl_rust_latent.safetensors".to_string()
    }));
    let vae = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/.serenity/models/vaes/OfficialStableDiffusion/sdxl_vae.safetensors".to_string()
    }));
    let output = PathBuf::from(args.get(3).cloned().unwrap_or_else(|| {
        "/tmp/sdxl_rust_vae_decode.png".to_string()
    }));

    let device = global_cuda_device();
    println!("latent: {}", latent.display());
    println!("vae:    {}", vae.display());

    let map = flame_core::serialization::load_file(&latent, &device)?;
    let lat = map
        .get("latent")
        .ok_or_else(|| anyhow::anyhow!("missing 'latent'"))?
        .to_dtype(DType::BF16)?;
    println!("latent shape: {:?}", lat.shape().dims());

    let v = LdmVAEDecoder::from_safetensors(&vae.to_string_lossy(), 4, 0.13025, 0.0, &device)?;
    let rgb = v.decode(&lat)?;
    println!("rgb shape: {:?}", rgb.shape().dims());

    let dims = rgb.shape().dims().to_vec();
    let (_b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                pixels[(y * w + x) * 3 + c] = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("PNG buf"))?
        .save(&output)?;
    println!("saved {}", output.display());
    let _: Result<()> = Ok(());
    Ok(())
}
