//! LTX-2 Video VAE decoder test — load checkpoint and decode a dummy latent tensor.
//!
//! Usage: cargo run --bin ltx2_vae_test [--release]
//!
//! This loads the LTX-2 VAE from the standard checkpoint location and
//! decodes a small dummy latent tensor to verify the pipeline works.

use inference_flame::vae::LTX2VaeDecoder;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const VAE_PATH: &str =
    "/home/alex/.serenity/models/vaes/LTX2/LTX2_video_vae_old_bf16.safetensors";

fn main() -> flame_core::Result<()> {
    eprintln!("=== LTX-2 Video VAE Decoder Test ===\n");

    let device = global_cuda_device();
    eprintln!("CUDA device initialized.\n");

    // Load VAE
    let t0 = Instant::now();
    let vae = LTX2VaeDecoder::from_file(VAE_PATH, &device)?;
    eprintln!("VAE loaded in {:.2?}\n", t0.elapsed());

    // Create dummy latent: [1, 128, 2, 2, 2]
    // This is a tiny test tensor:
    //   2 latent frames -> ~9 pixel frames (1 + (2-1)*8 = 9)
    //   2x2 latent spatial -> 64x64 pixel spatial (2*32 = 64)
    let batch = 1;
    let lat_ch = 128;
    let lat_f = 2;
    let lat_h = 2;
    let lat_w = 2;

    eprintln!(
        "Creating dummy latent: [{}, {}, {}, {}, {}]",
        batch, lat_ch, lat_f, lat_h, lat_w
    );

    let latent = Tensor::zeros_dtype(
        Shape::from_dims(&[batch, lat_ch, lat_f, lat_h, lat_w]),
        DType::BF16,
        device.clone(),
    )?;

    eprintln!("Latent shape: {:?}", latent.shape().dims());

    // Decode
    let t1 = Instant::now();
    let decoded = vae.decode(&latent, 0.05, 0.0)?;
    let decode_time = t1.elapsed();

    let out_dims = decoded.shape().dims().to_vec();
    eprintln!(
        "\nDecoded shape: {:?} (expected [1, 3, ~F*8, ~H*32, ~W*32])",
        out_dims
    );
    eprintln!("Decode time: {:.2?}", decode_time);

    // Validate output shape
    assert_eq!(out_dims[0], batch, "Batch size mismatch");
    assert_eq!(out_dims[1], 3, "Output channels should be 3 (RGB)");
    eprintln!("\nOutput dimensions: {}x{}x{} (FxHxW)", out_dims[2], out_dims[3], out_dims[4]);

    eprintln!("\n=== Test PASSED ===");
    Ok(())
}
