//! Klein 9B comparison run — uses PyTorch noise + embeddings for apples-to-apples.
//!
//! Loads pre-saved noise and text embeddings from PyTorch, runs Flame denoise + VAE.
//! Compare output with PyTorch's generate_klein9b.py using same inputs.

use inference_flame::models::klein::KleinTransformer;
use inference_flame::vae::klein_vae::{KleinVaeDecoder, unpatchify_latents};
use inference_flame::sampling::klein_sampling::{build_sigma_schedule, euler_denoise};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const INPUTS_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/pytorch_inputs_9b.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/klein9b_compare.png";

const NUM_STEPS: usize = 35;
const SHIFT: f32 = 2.02;
const CFG_SCALE: f32 = 3.5;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Klein 9B — Comparison (PyTorch inputs, Flame denoise+VAE)");
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load PyTorch inputs (noise + embeddings)
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load PyTorch inputs ---");
    let t0 = Instant::now();
    let inputs = flame_core::serialization::load_file(
        std::path::Path::new(INPUTS_PATH), &device
    )?;
    let noise = inputs.get("noise_packed").expect("missing noise_packed").clone();
    let pos_hidden = inputs.get("pos_hidden").expect("missing pos_hidden").clone();
    let neg_hidden = inputs.get("neg_hidden").expect("missing neg_hidden").clone();
    drop(inputs);
    println!("  noise: {:?}, pos: {:?}, neg: {:?}", noise.dims(), pos_hidden.dims(), neg_hidden.dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Build img_ids and txt_ids
    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;
    let n_img = latent_h * latent_w;
    let txt_seq_len = pos_hidden.dims()[1];

    let mut img_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_data[idx * 4 + 1] = r as f32;
            img_data[idx * 4 + 2] = c as f32;
        }
    }
    let img_ids = Tensor::from_f32_to_bf16(img_data, Shape::from_dims(&[n_img, 4]), device.clone())?;
    let txt_ids = Tensor::zeros_dtype(Shape::from_dims(&[txt_seq_len, 4]), DType::BF16, device.clone())?;

    // Warm the CUDA mempool by allocating ~15GB and freeing it.
    // This mimics the encoder load/drop from klein9b_infer and ensures
    // the pool has enough cached memory for denoise activations + SDPA.
    {
        println!("  Warming CUDA mempool...");
        // 15GB / 2 bytes = 7.5B elements
        let warmup = Tensor::zeros_dtype(
            Shape::from_dims(&[7_500_000_000]),
            DType::BF16, device.clone(),
        )?;
        drop(warmup);
        println!("  Pool warmed (15GB allocated+freed)");
    }

    // ------------------------------------------------------------------
    // Stage 2: Load Klein 9B model
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Klein 9B model ---");
    let t0 = Instant::now();
    let model = KleinTransformer::from_safetensors(MODEL_PATH)?;
    println!("  Config: {:?}", model.config());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Denoise
    // ------------------------------------------------------------------
    let sigmas = build_sigma_schedule(NUM_STEPS, SHIFT);
    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, CFG_SCALE);
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, sigma| {
            let sigma_t = Tensor::from_f32_to_bf16(
                vec![sigma],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;

            let cond = model.forward(x, &pos_hidden, &sigma_t, &img_ids, &txt_ids)?;
            let uncond = model.forward(x, &neg_hidden, &sigma_t, &img_ids, &txt_ids)?;

            let diff = cond.sub(&uncond)?;
            let guided = uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;
            x.sub(&guided.mul_scalar(sigma)?)
        },
        noise,
        &sigmas,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);

    let latents = denoised
        .reshape(&[1, latent_h, latent_w, 128])?
        .permute(&[0, 3, 1, 2])?;
    let latents_32ch = unpatchify_latents(&latents)?;

    let vae_weights = flame_core::serialization::load_file(
        std::path::Path::new(VAE_PATH), &device
    )?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    let rgb = vae.decode(&latents_32ch)?;
    println!("  VAE in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 5: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let (_, _, out_h, out_w) = {
        let d = rgb_f32.dims();
        (d[0], d[1], d[2], d[3])
    };

    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                let val = (data[idx].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0;
                pixels[(y * out_w + x) * 3 + c] = val as u8;
            }
        }
    }

    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
    img.save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
