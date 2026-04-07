//! Chroma end-to-end inference — pure Rust, flame-core + FlameSwap.
//!
//! Pipeline:
//!   1. T5-XXL → hidden [1, 512, 4096]   (FlameSwap, then DROPPED)
//!      Plus a separate uncond pass on an empty prompt for CFG.
//!   2. Chroma DiT → CFG denoise (cond + uncond per step)
//!   3. VAE → RGB
//!   4. PNG → /home/alex/EriDiffusion/inference-flame/output/chroma_rust.png
//!
//! Differences from `flux1_infer.rs`:
//!   - NO CLIP (Chroma is T5-only)
//!   - NO `pooled_projection` / `guidance_in` — modulation comes from
//!     `distilled_guidance_layer` inside the DiT
//!   - Uses real CFG: two forward passes per step combined as
//!     `noise = uncond + scale * (cond - uncond)`
//!
//! ⚠️ This binary is BUILD-COMPLETE but UNTESTED. The user is remote and
//! cannot reboot if something OOMs — DO NOT RUN until they validate.

use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::chroma_dit::ChromaDit;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{
    get_schedule, pack_latent, unpack_latent,
};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

// ---------------------------------------------------------------------------
// Paths / knobs
// ---------------------------------------------------------------------------

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

// Default to the diffusers-format Chroma1-HD shards. Override with CHROMA_DIT_SHARDS.
const CHROMA_DIT_SHARDS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
];

const CHROMA_VAE: &str = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors";

const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/chroma_rust.png";

const DEFAULT_PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";
const DEFAULT_NEGATIVE: &str = "";

const SEED: u64 = 42;
const HEIGHT: usize = 1024;
const WIDTH: usize = 1024;
const NUM_STEPS: usize = 40;
const GUIDANCE_SCALE: f32 = 4.0;

// Chroma uses the same FLUX VAE: 16 latent channels, scale 0.3611, shift 0.1159
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

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
    let negative = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());

    println!("=== Chroma — pure Rust inference ===");
    println!("Prompt:    {:?}", prompt);
    println!("Negative:  {:?}", negative);
    println!("Size:      {}x{}, steps={}, guidance={}", WIDTH, HEIGHT, NUM_STEPS, GUIDANCE_SCALE);
    println!("Seed:      {}", SEED);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: T5-XXL encode (cond + uncond, both DROPPED after)
    // ------------------------------------------------------------------
    println!("--- Stage 1: T5-XXL encode (cond + uncond) ---");
    let t0 = Instant::now();

    let (t5_cond_hidden, t5_uncond_hidden) = {
        let mut t5 = T5Encoder::load(T5_PATH, &device)?;
        println!("  Loaded T5 in {:.1}s", t0.elapsed().as_secs_f32());

        let cond_tokens = tokenize_t5(&prompt);
        let cond_hidden = t5.encode(&cond_tokens)?;
        println!("  cond hidden: {:?}", cond_hidden.shape().dims());

        let uncond_tokens = tokenize_t5(&negative);
        let uncond_hidden = t5.encode(&uncond_tokens)?;
        println!("  uncond hidden: {:?}", uncond_hidden.shape().dims());

        println!("  T5 done in {:.1}s", t0.elapsed().as_secs_f32());
        (cond_hidden, uncond_hidden)
    };
    println!("  T5 weights evicted");
    println!();

    // ------------------------------------------------------------------
    // Stage 2: Load Chroma DiT (FlameSwap)
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load Chroma DiT (FlameSwap) ---");
    let t0 = Instant::now();
    let mut dit = ChromaDit::load(CHROMA_DIT_SHARDS, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Build noise, pack, run CFG denoise loop
    // ------------------------------------------------------------------
    println!("--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, GUIDANCE_SCALE);

    // Same latent geometry as FLUX: VAE 8x downsample + patchify 2x = 16x effective.
    let latent_h = 2 * ((HEIGHT + 15) / 16);
    let latent_w = 2 * ((WIDTH + 15) / 16);
    println!("  Latent [B,C,H,W] = [1, {}, {}, {}]", AE_IN_CHANNELS, latent_h, latent_w);

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

    let (img_packed, img_ids) = pack_latent(&noise_nchw, &device)?;
    drop(noise_nchw);
    let _n_img = img_packed.shape().dims()[1];
    println!("  Packed img: {:?}", img_packed.shape().dims());
    println!("  img_ids:    {:?}", img_ids.shape().dims());

    // txt_ids: zeros [T5_SEQ_LEN, 3]
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )?;

    // Schedule: same FLUX-style flow-match Euler schedule.
    // Chroma is NOT guidance-distilled, so we run real CFG (2 forwards per step).
    let n_img = img_packed.shape().dims()[1];
    let timesteps = get_schedule(NUM_STEPS, n_img, 0.5, 1.15, true);
    println!("  Schedule: {} steps, t[0]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1, timesteps[0], timesteps[NUM_STEPS]);
    println!();

    // CFG denoise loop. The flux1_sampling crate's denoise helpers don't have
    // a CFG version, so we inline the Euler loop here.
    let t0 = Instant::now();
    let mut x = img_packed;
    for step in 0..NUM_STEPS {
        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        let t_vec = Tensor::from_vec(
            vec![t_curr],
            Shape::from_dims(&[1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // cond forward
        let cond_pred = dit.forward(
            &x,
            &t5_cond_hidden,
            &t_vec,
            &img_ids,
            &txt_ids,
        )?;
        // uncond forward
        let uncond_pred = dit.forward(
            &x,
            &t5_uncond_hidden,
            &t_vec,
            &img_ids,
            &txt_ids,
        )?;

        // CFG: noise = uncond + scale * (cond - uncond)
        let diff = cond_pred.sub(&uncond_pred)?;
        let scaled = diff.mul_scalar(GUIDANCE_SCALE)?;
        let pred = uncond_pred.add(&scaled)?;

        // Euler step: x = x + dt * pred
        let step_tensor = pred.mul_scalar(dt)?;
        x = x.add(&step_tensor)?;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == NUM_STEPS {
            log::info!("[Chroma] step {}/{} t={:.4}", step + 1, NUM_STEPS, t_curr);
        }
    }
    let dt_denoise = t0.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step)",
        dt_denoise,
        dt_denoise / NUM_STEPS as f32,
    );
    println!("  Output: {:?}", x.shape().dims());
    println!();

    drop(dit);
    drop(t5_cond_hidden);
    drop(t5_uncond_hidden);
    println!("  DiT evicted");

    // ------------------------------------------------------------------
    // Stage 4: Unpack + VAE decode
    // ------------------------------------------------------------------
    println!("--- Stage 4: VAE decode ---");
    let t0 = Instant::now();

    let latent = unpack_latent(&x, HEIGHT, WIDTH)?;
    drop(x);
    println!("  Unpacked latent: {:?}", latent.shape().dims());

    let vae = LdmVAEDecoder::from_safetensors(
        CHROMA_VAE,
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
    // Stage 5: Denormalize → PNG
    // ------------------------------------------------------------------
    println!("--- Stage 5: Save PNG ---");
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

    let _ = device;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tokenization helpers
// ---------------------------------------------------------------------------

/// T5-XXL tokenize, pad to T5_SEQ_LEN with 0 (pad token).
fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            eprintln!("[chroma_infer] T5 tokenizer load failed: {}", e);
            // Fall back to a dummy tokenization (zeros) so the binary still
            // builds and starts. The user MUST provide a valid tokenizer JSON.
            vec![0i32; T5_SEQ_LEN]
        }
    }
}
