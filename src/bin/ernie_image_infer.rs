//! ERNIE-Image inference — text encode → flow matching denoise → VAE decode → PNG.
//!
//! Usage: cargo run --release --bin ernie_image_infer [-- "prompt text"]
//!
//! Pipeline:
//!   1. Mistral-3 3B text encode → dump from VRAM
//!   2. Load ERNIE-Image DiT via FlameSwap (36 blocks)
//!   3. Flow matching Euler denoise (shift=3.0)
//!   4. VAE decode → PNG

use inference_flame::models::ernie_image::{ErnieImageConfig, ErnieImageSwapped};
use inference_flame::models::mistral3b_encoder::Mistral3bEncoder;
use inference_flame::vae::klein_vae::KleinVaeDecoder;
use inference_flame::sampling::ernie_sampling::{ernie_schedule, ernie_euler_step, sigma_to_timestep};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const TRANSFORMER_DIR: &str = "/home/alex/models/ERNIE-Image/transformer";
const TEXT_ENCODER: &str = "/home/alex/models/ERNIE-Image/text_encoder/model.safetensors";
const TOKENIZER: &str = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json";
const VAE_PATH: &str = "/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors";

const WIDTH: usize = 512;
const HEIGHT: usize = 512;
const STEPS: usize = 28;
const GUIDANCE: f32 = 4.0;
const SEED: u64 = 42;

fn main() {
    env_logger::init();
    if let Err(e) = run() {
        eprintln!("ernie_image_infer failed: {e:?}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "a beautiful sunset over the ocean".to_string());
    let output = std::env::args().nth(2).unwrap_or_else(|| "/home/alex/serenity/output/ernie_image_test.png".to_string());

    let device = global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();

    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;

    println!("=== ERNIE-Image Inference ===");
    println!("  prompt: {prompt:?}");
    println!("  size: {WIDTH}x{HEIGHT}, latent: {latent_w}x{latent_h}");
    println!("  steps: {STEPS}, guidance: {GUIDANCE}, seed: {SEED}");

    // ---------------------------------------------------------------
    // Stage 1: Text Encoding (Mistral-3 3B)
    // ---------------------------------------------------------------
    println!("\n[1/4] Text encoding (Mistral-3 3B)...");
    let t0 = Instant::now();

    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let encoding = tokenizer.encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let text_len = token_ids.len();
    println!("  tokenized: {text_len} tokens");

    let text_embeds = {
        let encoder = Mistral3bEncoder::load(TEXT_ENCODER, &device)?;
        println!("  encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let embeds = encoder.encode(&token_ids, 256)?;
        println!("  encoded: shape={:?}", embeds.shape().dims());
        embeds
    };
    println!("  text encoding done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---------------------------------------------------------------
    // Stage 2: Load Transformer (FlameSwap)
    // ---------------------------------------------------------------
    println!("\n[2/4] Loading ERNIE-Image DiT (FlameSwap)...");
    let t1 = Instant::now();

    let shard_paths = {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(TRANSFORMER_DIR)? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                paths.push(p.to_string_lossy().into_owned());
            }
        }
        paths.sort();
        paths
    };
    let path_refs: Vec<&str> = shard_paths.iter().map(|s| s.as_str()).collect();

    let config = ErnieImageConfig::default();
    let mut model = ErnieImageSwapped::load(&path_refs, config.clone(), &device)?;
    println!("  DiT loaded in {:.1}s", t1.elapsed().as_secs_f32());

    // ---------------------------------------------------------------
    // Stage 3: Denoise
    // ---------------------------------------------------------------
    println!("\n[3/4] Denoising ({STEPS} steps, shift=3.0)...");
    let t2 = Instant::now();

    let sigmas = ernie_schedule(STEPS);

    // Generate initial noise [1, 128, latent_h, latent_w]
    let noise = {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let n = 128 * latent_h * latent_w;
        let mut data = vec![0.0f32; n];
        for v in &mut data {
            let u1: f32 = rand::Rng::gen_range(&mut rng, 1e-7f32..1.0);
            let u2: f32 = rand::Rng::gen_range(&mut rng, 0.0f32..std::f32::consts::TAU);
            *v = (-2.0 * u1.ln()).sqrt() * u2.cos();
        }
        Tensor::from_vec(data, Shape::from_dims(&[1, 128, latent_h, latent_w]), device.clone())?
            .to_dtype(DType::BF16)?
    };

    let mut latent = noise;

    // Text embeds: [1, seq_len, 3072]
    let text_3d = if text_embeds.rank() == 2 {
        text_embeds.unsqueeze(0)?
    } else {
        text_embeds.clone()
    };
    let text_lens = vec![text_len.min(256)];

    // Unconditional embeds for CFG
    let uncond_embeds = Tensor::zeros_dtype(
        text_3d.shape().clone(), DType::BF16, device.clone(),
    )?;
    let uncond_lens = vec![0usize];

    for step in 0..STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

        let pred = if GUIDANCE > 1.0 {
            let pred_cond = model.forward(&latent, &t_tensor, &text_3d, &text_lens)?;
            let pred_uncond = model.forward(&latent, &t_tensor, &uncond_embeds, &uncond_lens)?;
            pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(GUIDANCE)?)?
        } else {
            model.forward(&latent, &t_tensor, &text_3d, &text_lens)?
        };
        latent = ernie_euler_step(&latent, &pred, sigma, sigma_next)?;

        if step % 10 == 0 || step == STEPS - 1 {
            println!("  step {}/{STEPS} sigma={sigma:.4} t={t:.1}", step + 1);
        }
    }
    println!("  denoising done in {:.1}s", t2.elapsed().as_secs_f32());

    // Debug: check latent stats
    {
        let v = latent.to_vec_f32()?;
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let std: f32 = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let nan = v.iter().filter(|x| x.is_nan()).count();
        println!("  latent stats: mean={mean:.4} std={std:.4} nan={nan} shape={:?}", latent.shape().dims());
    }

    // ---------------------------------------------------------------
    // Stage 4: VAE Decode
    // ---------------------------------------------------------------
    println!("\n[4/4] VAE decode...");
    let t3 = Instant::now();

    drop(model);

    let vae_weights = flame_core::serialization::load_file(VAE_PATH, &device)?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    drop(vae_weights);

    let decoded = vae.decode(&latent)?;
    println!("  decoded: shape={:?}", decoded.shape().dims());
    println!("  VAE done in {:.1}s", t3.elapsed().as_secs_f32());

    // Save PNG
    let img_data = decoded.to_vec_f32()?;
    let dims = decoded.shape().dims();
    let (c, h, w) = (dims[1], dims[2], dims[3]);
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3.min(c) {
                let v = img_data[ch * h * w + y * w + x];
                let v = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                rgb[(y * w + x) * 3 + ch] = v;
            }
        }
    }

    std::fs::create_dir_all(std::path::Path::new(&output).parent().unwrap_or(std::path::Path::new(".")))?;
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow::anyhow!("failed to create image"))?;
    img.save(&output)?;

    println!("\n=== Done ===");
    println!("  output: {output}");
    println!("  total: {:.1}s", t0.elapsed().as_secs_f32());

    Ok(())
}
