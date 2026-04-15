//! Klein 4B image generation — pure Rust, no Python.
//!
//! Usage: cargo run --release --bin klein_infer [-- "prompt text"]
//!
//! Pipeline:
//! 1. Tokenize + encode text (Qwen3 4B) OR load cached embeddings
//! 2. Load Klein 4B model
//! 3. Create noise, run Euler denoise loop
//! 4. VAE decode → RGB → PNG

use inference_flame::models::klein::KleinTransformer;
use inference_flame::models::qwen3_encoder::{Qwen3Config, Qwen3Encoder};
use inference_flame::vae::klein_vae::KleinVaeDecoder;
use inference_flame::sampling::klein_sampling::{get_schedule, euler_denoise};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-4b.safetensors";
const ENCODER_PATH: &str = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";
const EMBEDDINGS_PATH: &str = "/home/alex/EriDiffusion/flame-core/inference-test/cached_embeddings.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/klein4b_rust_v2.png";

const DEFAULT_PROMPT: &str = "Beautiful young woman sitting on a park bench in golden hour sunlight, professional model photoshoot for Maxim magazine advertisement, wearing a fitted summer dress, confident relaxed pose, soft bokeh background with green trees and warm light, editorial fashion photography, Canon EOS R5, 85mm f/1.4 lens, natural skin texture, magazine quality retouching, warm color grading";
const DEFAULT_NEGATIVE: &str = "lowres, bad quality, worst quality, bad anatomy, blurry, watermark, simple background, transparent background, sketch, jpeg artifacts, ugly, poorly drawn, censor";
// BFL reference defaults for base models
const NUM_STEPS: usize = 50;
const GUIDANCE: f32 = 4.0;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Klein 4B — Pure Rust Inference (inference-flame)");
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Text encoding (Qwen3 4B) or cached embeddings
    // ------------------------------------------------------------------
    let prompt = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    println!("\n--- Stage 1: Text Encoding ---");
    println!("  Prompt: {}", prompt);
    let t0 = Instant::now();

    const TXT_PAD_LEN: usize = 512;
    // Klein chat template — must match Python pipeline exactly
    const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
    const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    let (pos_hidden, neg_hidden) = {
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Apply Klein chat template, then tokenize
        let pos_formatted = format!("{KLEIN_TEMPLATE_PRE}{prompt}{KLEIN_TEMPLATE_POST}");
        let negative = DEFAULT_NEGATIVE;
        let neg_formatted = format!("{KLEIN_TEMPLATE_PRE}{negative}{KLEIN_TEMPLATE_POST}");

        let pos_enc = tokenizer.encode(pos_formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;
        let neg_enc = tokenizer.encode(neg_formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;

        // Pad to TXT_PAD_LEN with pad_id (causal mask = pad tokens can't affect real tokens)
        let pad_id = 151643i32; // Qwen3 PAD/EOS
        let mut pos_ids: Vec<i32> = pos_enc.get_ids().iter().map(|&id| id as i32).collect();
        let mut neg_ids: Vec<i32> = neg_enc.get_ids().iter().map(|&id| id as i32).collect();
        println!("  Pos tokens: {} (template-wrapped)", pos_ids.len());
        println!("  Neg tokens: {} (template-wrapped)", neg_ids.len());
        pos_ids.resize(TXT_PAD_LEN, pad_id);
        neg_ids.resize(TXT_PAD_LEN, pad_id);

        // Load Qwen3 encoder
        let enc_weights = flame_core::serialization::load_file(
            std::path::Path::new(ENCODER_PATH), &device
        )?;
        let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
        let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());

        let pos_h = encoder.encode(&pos_ids)?;
        let neg_h = encoder.encode(&neg_ids)?;
        println!("  pos: {:?}, neg: {:?}", pos_h.dims(), neg_h.dims());
        println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

        drop(encoder);
        (pos_h, neg_h)
    };
    let txt_seq_len = TXT_PAD_LEN;

    // Build img_ids and txt_ids dynamically for target resolution
    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;
    let n_img = latent_h * latent_w;
    // img_ids: [N_img, 4] where each row = [0, row, col, 0]
    let mut img_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_data[idx * 4 + 1] = r as f32;
            img_data[idx * 4 + 2] = c as f32;
        }
    }
    let img_ids = Tensor::from_f32_to_bf16(img_data, Shape::from_dims(&[n_img, 4]), device.clone())?;
    // txt_ids: [N_txt, 4] — all zeros (text has no spatial position)
    let txt_ids = Tensor::zeros_dtype(Shape::from_dims(&[txt_seq_len, 4]), DType::BF16, device.clone())?;
    println!("  img_ids: {:?}, txt_ids: {:?}", img_ids.dims(), txt_ids.dims());

    // ------------------------------------------------------------------
    // Stage 2: Load Klein 4B model
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Klein 4B model ---");
    let t0 = Instant::now();
    let model = KleinTransformer::from_safetensors(MODEL_PATH)?;
    println!("  Config: {:?}", model.config());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Create noise + sigma schedule
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Prepare noise + sigmas ---");
    // latent_h, latent_w already defined above for img_ids

    // Seeded noise using Box-Muller
    let numel = 128 * latent_h * latent_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut v = Vec::with_capacity(numel);
        for _ in 0..numel / 2 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(r * theta.cos());
            v.push(r * theta.sin());
        }
        if numel % 2 == 1 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            v.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }
        v
    };

    // Upload as BF16: [1, 128, latent_h, latent_w]
    let noise_spatial = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        device.clone(),
    )?;

    // Pack: [1, 128, H, W] → [1, H, W, 128] → [1, H*W, 128]
    let noise = noise_spatial
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, latent_h * latent_w, 128])?;
    println!("  Noise: {:?} (packed)", noise.dims());

    // Dynamic schedule (BFL reference)
    let n_img = latent_h * latent_w;
    let timesteps = get_schedule(NUM_STEPS, n_img);
    println!("  Schedule: {} values, t[0]={:.4}, t[-2]={:.4}",
             timesteps.len(), timesteps[0], timesteps[NUM_STEPS-1]);

    // ------------------------------------------------------------------
    // Stage 4: Denoise (BFL direct velocity + CFG)
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Denoise ({} steps, guidance={}) ---", NUM_STEPS, GUIDANCE);
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;

            let pred_cond = model.forward(x, &pos_hidden, &t_vec, &img_ids, &txt_ids)?;
            let pred_uncond = model.forward(x, &neg_hidden, &t_vec, &img_ids, &txt_ids)?;

            // CFG: uncond + guidance * (cond - uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(GUIDANCE)?)
        },
        noise,
        &timesteps,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);
    println!("  Output: {:?}", denoised.dims());

    // ------------------------------------------------------------------
    // Stage 5: Unpack + VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 5: VAE Decode ---");
    let t0 = Instant::now();

    // Free DiT weights to make room for VAE
    drop(model);
    println!("  DiT weights freed");

    // Unpack: [1, H*W, 128] → [1, H, W, 128] → [1, 128, H, W]
    let latents = denoised
        .reshape(&[1, latent_h, latent_w, 128])?
        .permute(&[0, 3, 1, 2])?;

    // Pass 128ch latents directly — VAE handles inv_normalize + unpatchify internally
    println!("  Latents: {:?}", latents.dims());

    // Load VAE
    println!("  Loading VAE weights...");
    let vae_weights = flame_core::serialization::load_file(
        std::path::Path::new(VAE_PATH), &device
    )?;
    println!("  VAE weights loaded: {} keys", vae_weights.len());
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    println!("  Building VAE decoder...");
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    println!("  VAE decoder built");

    // Decode
    println!("  Decoding...");
    let rgb = vae.decode(&latents)?;
    println!("  Decoded: {:?}", rgb.dims());
    println!("  VAE in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 6: Save PNG
    // ------------------------------------------------------------------
    println!("\n--- Stage 6: Save PNG ---");

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
                let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * out_w + x) * 3 + c] = val;
            }
        }
    }

    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
    img.save(OUTPUT_PATH)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total time: {:.1}s", dt_total);
    println!("============================================================");

    Ok(())
}
