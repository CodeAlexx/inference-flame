//! FLUX 1 Dev end-to-end inference — pure Rust, flame-core + BlockOffloader.
//!
//! Pipeline:
//!   1. CLIP-L   → pooled [1, 768]
//!   2. T5-XXL   → hidden [1, 512, 4096]   (BlockOffloader, then DROPPED)
//!   3. FLUX 1   → 20-step Euler denoise via flux1_sampling (BlockOffloader)
//!   4. VAE      → RGB
//!   5. PNG      → /home/alex/EriDiffusion/inference-flame/output/flux1_rust.png
//!
//! Hard rule: NO CFG. FLUX 1 Dev is guidance-distilled, `guidance=3.5` is
//! injected as a model conditioning input by `flux1_denoise`.
//!
//! This binary does NOT touch Klein/flux2 codepaths.

use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::flux1_dit::Flux1DiT;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{
    flux1_denoise, get_schedule, pack_latent, unpack_latent,
};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

// ---------------------------------------------------------------------------
// Paths / knobs
// ---------------------------------------------------------------------------

const CLIP_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const DIT_PATH: &str = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/ae.safetensors";

const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/flux1_rust.png";

const DEFAULT_PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";

const SEED: u64 = 42;
const HEIGHT: usize = 1024;
const WIDTH: usize = 1024;
const NUM_STEPS: usize = 20;
const GUIDANCE: f32 = 3.5;

// FLUX 1 Dev ae_params (util.py: flux-dev)
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 512;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("=== FLUX 1 Dev — pure Rust inference ===");
    println!("Prompt: {:?}", prompt);
    println!("Size:   {}x{}, steps={}, guidance={}", WIDTH, HEIGHT, NUM_STEPS, GUIDANCE);
    println!("Seed:   {}", SEED);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: CLIP-L encode (pooled only)
    // ------------------------------------------------------------------
    println!("--- Stage 1: CLIP-L encode ---");
    let t0 = Instant::now();

    let clip_weights_raw = flame_core::serialization::load_file(
        std::path::Path::new(CLIP_PATH),
        &device,
    )?;
    // CLIP-L safetensors is FP16; flame-core loader upcasts F16 → F32. layer_norm_bf16
    // and other BF16 kernels require BF16 inputs, so cast every weight tensor down.
    let clip_weights: std::collections::HashMap<String, Tensor> = clip_weights_raw
        .into_iter()
        .map(|(k, v)| {
            let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
            Ok::<_, flame_core::Error>((k, t))
        })
        .collect::<Result<_, _>>()?;
    println!("  Loaded {} CLIP weights in {:.1}s", clip_weights.len(), t0.elapsed().as_secs_f32());

    let clip_cfg = ClipConfig::default();
    let clip = ClipEncoder::new(clip_weights, clip_cfg, device.clone());

    let clip_tokens = tokenize_clip(&prompt);
    let (_clip_hidden, clip_pooled) = clip.encode(&clip_tokens)?;
    println!("  pooled: {:?}", clip_pooled.shape().dims());
    println!("  CLIP done in {:.1}s", t0.elapsed().as_secs_f32());
    drop(clip);
    println!();

    // ------------------------------------------------------------------
    // Stage 2: T5-XXL encode — BlockOffloader — then drop
    // ------------------------------------------------------------------
    println!("--- Stage 2: T5-XXL encode ---");
    let t0 = Instant::now();

    let t5_hidden_bf16 = {
        let mut t5 = T5Encoder::load(T5_PATH, &device)?;
        println!("  Loaded T5 in {:.1}s", t0.elapsed().as_secs_f32());

        let t5_tokens = tokenize_t5(&prompt);
        let hidden = t5.encode(&t5_tokens)?;
        println!("  T5 hidden: {:?}", hidden.shape().dims());
        println!("  T5 encode done in {:.1}s", t0.elapsed().as_secs_f32());

        // Scope drop ensures BlockOffloader-backed weights are freed before DiT load.
        hidden
    };
    println!("  T5 weights evicted");
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Load FLUX 1 DiT (BlockOffloader)
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load FLUX 1 DiT (BlockOffloader) ---");
    let t0 = Instant::now();
    let mut dit = Flux1DiT::load(DIT_PATH, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Build noise, pack, run denoise loop
    // ------------------------------------------------------------------
    println!("--- Stage 4: Denoise ({} steps) ---", NUM_STEPS);

    // FLUX 1 VAE has 8x downsample and patchify adds another 2x → 16x effective.
    // BFL get_noise shape: (B, 16, 2*ceil(H/16), 2*ceil(W/16)).
    let latent_h = 2 * ((HEIGHT + 15) / 16);
    let latent_w = 2 * ((WIDTH + 15) / 16);
    println!("  Latent [B,C,H,W] = [1, {}, {}, {}]", AE_IN_CHANNELS, latent_h, latent_w);

    // Seeded Gaussian noise, Box-Muller (deterministic across runs).
    let numel = AE_IN_CHANNELS * latent_h * latent_w;
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
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    // Pack: [1, 16, H, W] → ([1, H*W/4, 64], img_ids[H*W/4, 3])
    let (img_packed, img_ids) = pack_latent(&noise_nchw, &device)?;
    drop(noise_nchw);
    let n_img = img_packed.shape().dims()[1];
    println!("  Packed img: {:?}", img_packed.shape().dims());
    println!("  img_ids:    {:?}", img_ids.shape().dims());

    // txt_ids: zeros [T5_SEQ_LEN, 3]
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )?;

    // Schedule — FLUX 1 linear mu (256→0.5, 4096→1.15)
    let timesteps = get_schedule(NUM_STEPS, n_img, 0.5, 1.15, true);
    println!("  Schedule: {} steps, t[0]={:.4}, t[1]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1, timesteps[0], timesteps[1], timesteps[NUM_STEPS]);
    println!();

    // Denoise — no CFG, single forward per step with guidance_vec injected.
    let t0 = Instant::now();
    let pos_hidden = t5_hidden_bf16;
    let denoised = flux1_denoise(
        |img, t_vec, guidance_vec| {
            dit.forward(
                img,
                &pos_hidden,
                t_vec,
                &img_ids,
                &txt_ids,
                Some(guidance_vec),
                Some(&clip_pooled),
            )
        },
        img_packed,
        &timesteps,
        GUIDANCE,
        &device,
    )?;
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step)",
        dt,
        dt / NUM_STEPS as f32,
    );
    println!("  Output: {:?}", denoised.shape().dims());
    println!();

    // Free DiT before VAE load (DiT BlockOffloader holds ~24GB worth of weights).
    drop(dit);
    println!("  DiT evicted");

    // ------------------------------------------------------------------
    // Stage 5: Unpack + VAE decode
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();

    // Unpack packed latent back to [1, 16, latent_h, latent_w].
    let latent = unpack_latent(&denoised, HEIGHT, WIDTH)?;
    drop(denoised);
    println!("  Unpacked latent: {:?}", latent.shape().dims());

    let vae = LdmVAEDecoder::from_safetensors(
        VAE_PATH,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &device,
    )?;
    println!("  VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let rgb = vae.decode(&latent)?;
    drop(latent);
    drop(vae);
    println!("  Decoded: {:?}", rgb.shape().dims());
    println!("  VAE stage in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 6: Denormalize → PNG
    // ------------------------------------------------------------------
    println!("--- Stage 6: Save PNG ---");
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec_f32()?;
    let dims = rgb_f32.shape().dims().to_vec();
    let (out_c, out_h, out_w) = (dims[1], dims[2], dims[3]);
    assert_eq!(out_c, 3, "VAE decoder must return 3 channels");

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
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to build RgbImage"))?;
    img.save(OUTPUT_PATH)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    let _ = device; // keep alive
    Ok(())
}

// ---------------------------------------------------------------------------
// Tokenization helpers
// ---------------------------------------------------------------------------

/// CLIP-L tokenize, pad to 77 with EOS.
fn tokenize_clip(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(CLIP_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("clip tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(CLIP_SEQ_LEN);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407); // CLIP pad = EOS
            }
            ids
        }
        Err(e) => {
            eprintln!("[flux1_infer] CLIP tokenizer failed: {e}; using BOS/EOS fallback");
            let mut ids = vec![49406i32]; // BOS
            ids.push(49407); // EOS
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
    }
}

/// T5-XXL tokenize, pad to 512 with 0.
fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.push(1); // T5 EOS
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0); // T5 pad
            }
            ids
        }
        Err(e) => {
            eprintln!("[flux1_infer] T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
    }
}
