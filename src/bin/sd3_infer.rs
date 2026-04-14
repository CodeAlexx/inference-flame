//! SD3.5 Large end-to-end inference — pure Rust, flame-core.
//!
//! Full pipeline: prompt → CLIP-L + CLIP-G + T5-XXL → DiT denoise → VAE → PNG.
//! 38 blocks, hidden=2432, no dual attention. ~16.3GB BF16, fits resident on 24GB.
//!
//! Usage: sd3_infer ["prompt text"]

use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::sd3_mmdit::{load_sd3_all_chunked, SD3MMDiT};
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const CLIP_L_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_L_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const CLIP_G_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors";
const CLIP_G_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json";

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const MODEL_PATH: &str = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors";

const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sd3_large_rust.png";

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------

const DEFAULT_PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";

const NUM_STEPS: usize = 28;
const CFG_SCALE: f32 = 4.5;
const SHIFT: f32 = 3.0;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 256;

// SD3 VAE params: 16ch latent, scale=1.5305, shift=0.0609
const VAE_IN_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;
const VAE_SHIFT: f32 = 0.0609;

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Tokenizers
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str, tokenizer_path: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(tokenizer_path) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("clip tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(CLIP_SEQ_LEN);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
        Err(e) => {
            eprintln!("[sd3] CLIP tokenizer failed: {e}; using BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.push(1);
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            eprintln!("[sd3] T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            ids.resize(T5_SEQ_LEN, 0);
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Text encoding pipeline
// ---------------------------------------------------------------------------

fn encode_text_pair(
    prompt: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let empty = "";

    fn load_clip_weights(
        path: &str,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
        let raw = flame_core::serialization::load_file(std::path::Path::new(path), device)?;
        let weights = raw
            .into_iter()
            .map(|(k, v)| {
                let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
                Ok::<_, flame_core::Error>((k, t))
            })
            .collect::<Result<_, _>>()?;
        Ok(weights)
    }

    // ---- CLIP-L ----
    println!("  CLIP-L...");
    let t0 = Instant::now();
    let (clip_l_hidden, clip_l_pooled, clip_l_hidden_u, clip_l_pooled_u) = {
        let weights = load_clip_weights(CLIP_L_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, pc) = clip.encode_sd3(&tokenize_clip(prompt, CLIP_L_TOKENIZER))?;
        let (hu, pu) = clip.encode_sd3(&tokenize_clip(empty, CLIP_L_TOKENIZER))?;
        (hc, pc, hu, pu)
    };
    println!("    cond: {:?}, uncond: {:?} ({:.1}s)",
        clip_l_hidden.dims(), clip_l_hidden_u.dims(), t0.elapsed().as_secs_f32());

    // ---- CLIP-G ----
    println!("  CLIP-G...");
    let t0 = Instant::now();
    let (clip_g_hidden, clip_g_pooled, clip_g_hidden_u, clip_g_pooled_u) = {
        let weights = load_clip_weights(CLIP_G_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::clip_g(), device.clone());
        let (hc, pc) = clip.encode_sd3(&tokenize_clip(prompt, CLIP_G_TOKENIZER))?;
        let (hu, pu) = clip.encode_sd3(&tokenize_clip(empty, CLIP_G_TOKENIZER))?;
        (hc, pc, hu, pu)
    };
    println!("    cond: {:?}, uncond: {:?} ({:.1}s)",
        clip_g_hidden.dims(), clip_g_hidden_u.dims(), t0.elapsed().as_secs_f32());

    // ---- T5-XXL ----
    println!("  T5-XXL...");
    let t0 = Instant::now();
    let (t5_hidden, t5_hidden_u) = {
        let mut t5 = T5Encoder::load(T5_PATH, device)?;
        let hc = t5.encode(&tokenize_t5(prompt))?;
        let hu = t5.encode(&tokenize_t5(empty))?;
        (hc, hu)
    };
    let t5_hidden = t5_hidden.narrow(1, 0, t5_hidden.dims()[1].min(T5_SEQ_LEN))?;
    let t5_hidden_u = t5_hidden_u.narrow(1, 0, t5_hidden_u.dims()[1].min(T5_SEQ_LEN))?;
    println!("    cond: {:?}, uncond: {:?} ({:.1}s)",
        t5_hidden.dims(), t5_hidden_u.dims(), t0.elapsed().as_secs_f32());

    // ---- Combine ----
    let cl_pad = zero_pad_last_dim(&clip_l_hidden, 4096)?;
    let cg_pad = zero_pad_last_dim(&clip_g_hidden, 4096)?;
    let context = Tensor::cat(&[&cl_pad, &cg_pad, &t5_hidden], 1)?;
    let pooled = Tensor::cat(&[&clip_l_pooled, &clip_g_pooled], 1)?;

    let cl_pad_u = zero_pad_last_dim(&clip_l_hidden_u, 4096)?;
    let cg_pad_u = zero_pad_last_dim(&clip_g_hidden_u, 4096)?;
    let context_u = Tensor::cat(&[&cl_pad_u, &cg_pad_u, &t5_hidden_u], 1)?;
    let pooled_u = Tensor::cat(&[&clip_l_pooled_u, &clip_g_pooled_u], 1)?;

    println!("  Combined: cond {:?}, uncond {:?}, pooled {:?}",
        context.dims(), context_u.dims(), pooled.dims());

    Ok((context, pooled, context_u, pooled_u))
}

fn zero_pad_last_dim(x: &Tensor, target_dim: usize) -> anyhow::Result<Tensor> {
    let dims = x.dims();
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    if c >= target_dim {
        return Ok(x.clone());
    }
    let pad = Tensor::zeros_dtype(
        Shape::from_dims(&[b, n, target_dim - c]),
        DType::BF16,
        x.device().clone(),
    )?;
    Ok(Tensor::cat(&[x, &pad], 2)?)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("============================================================");
    println!("SD3.5 Large — Pure Rust End-to-End Inference");
    println!("  Prompt: {:?}", prompt);
    println!("  {}x{}, {} steps, CFG {}, shift {}", WIDTH, HEIGHT, NUM_STEPS, CFG_SCALE, SHIFT);
    println!("============================================================");

    // ------------------------------------------------------------------
    // Stage 1: Text encoding — each encoder loaded once, then freed
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Text Encoding ---");
    let t0 = Instant::now();
    let (context, pooled, context_uncond, pooled_uncond) = encode_text_pair(&prompt, &device)?;
    println!("  Total encode: {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load SD3.5 Large (all resident, loaded block-by-block)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load SD3.5 Large ---");
    let t0 = Instant::now();
    let resident = load_sd3_all_chunked(MODEL_PATH, &device)?;
    println!("  {} total keys", resident.len());
    let mut model = SD3MMDiT::new(MODEL_PATH.to_string(), resident, device.clone());
    println!(
        "  depth={}, hidden={}, heads={}, dual_attn={}",
        model.config.depth,
        model.config.hidden_size,
        model.config.num_heads,
        model.config.use_dual_attention,
    );
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Denoise (flow matching, Euler, CFG)
    // ------------------------------------------------------------------
    let latent_h = HEIGHT / 8;
    let latent_w = WIDTH / 8;
    let numel = VAE_IN_CHANNELS * latent_h * latent_w;

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
    let noise = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    let timesteps = build_schedule(NUM_STEPS, SHIFT);

    println!(
        "\n--- Stage 3: Denoise ({} steps, CFG={}) ---",
        NUM_STEPS, CFG_SCALE
    );
    let t0 = Instant::now();

    let mut step_count = 0usize;
    let denoised = inference_flame::sampling::klein_sampling::euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr * 1000.0],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;
            let pred_cond = model.forward(x, &t_vec, &context, &pooled)?;
            let pred_uncond = model.forward(x, &t_vec, &context_uncond, &pooled_uncond)?;
            let diff = pred_cond.sub(&pred_uncond)?;
            let out = pred_uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;
            if step_count < 3 {
                let stats = out.to_vec_f32()?;
                let mean: f32 = stats.iter().sum::<f32>() / stats.len() as f32;
                let max = stats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = stats.iter().cloned().fold(f32::INFINITY, f32::min);
                let nan_count = stats.iter().filter(|v| v.is_nan()).count();
                println!("  step {} t={:.3}: mean={:.4}, min={:.4}, max={:.4}, nan={}",
                    step_count, t_curr, mean, min, max, nan_count);
            }
            step_count += 1;
            Ok(out)
        },
        noise,
        &timesteps,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // Check denoised latent stats
    {
        let stats = denoised.to_vec_f32()?;
        let mean: f32 = stats.iter().sum::<f32>() / stats.len() as f32;
        let std: f32 = (stats.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / stats.len() as f32).sqrt();
        let max = stats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = stats.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("  Denoised latent: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", mean, std, min, max);
    }

    // Free DiT + embeddings before VAE
    drop(model);
    drop(context);
    drop(pooled);
    drop(context_uncond);
    drop(pooled_uncond);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();

    let vae = LdmVAEDecoder::from_safetensors(MODEL_PATH, VAE_IN_CHANNELS, VAE_SCALE, VAE_SHIFT, &device)?;
    let rgb = vae.decode(&denoised)?;
    drop(denoised);
    drop(vae);
    println!("  {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 5: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec_f32()?;
    let dims = rgb_f32.dims();
    let (out_h, out_w) = (dims[2], dims[3]);

    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                let v = data[idx].clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * out_w + x) * 3 + c] = u;
            }
        }
    }

    if let Some(parent) = std::path::Path::new(OUTPUT_PATH).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image creation failed"))?
        .save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
