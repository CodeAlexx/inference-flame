//! Klein 9B (base) image generation — pure Rust, BFL reference pipeline.
//!
//! Usage: klein9b_infer ["prompt text"]
//!
//! Pipeline:
//! 1. Qwen3 8B encode → drop encoder (frees ~15GB, warms mempool)
//! 2. Klein 9B all-on-GPU (fits in recycled mempool space)
//! 3. Denoise (50 steps, guidance=4.0)
//! 4. Drop model, VAE decode → PNG
//!
//! Falls back to block offloading if all-on-GPU OOMs.

use inference_flame::models::klein::{KleinTransformer, KleinOffloaded};
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::vae::klein_vae::KleinVaeDecoder;
use inference_flame::sampling::klein_sampling::{get_schedule, euler_denoise};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const ENCODER_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/klein9b_rust.png";

const DEFAULT_PROMPT: &str = "Beautiful young woman sitting on a park bench in golden hour sunlight, professional model photoshoot for Maxim magazine advertisement, wearing a fitted summer dress, confident relaxed pose, soft bokeh background with green trees and warm light, editorial fashion photography, Canon EOS R5, 85mm f/1.4 lens, natural skin texture, magazine quality retouching, warm color grading";
const DEFAULT_NEGATIVE: &str = "lowres, bad quality, worst quality, bad anatomy, blurry, watermark, simple background, transparent background, sketch, jpeg artifacts, ugly, poorly drawn, censor";

const DEFAULT_NUM_STEPS: usize = 50;
const GUIDANCE: f32 = 4.0;

/// Step count, overridable via `KLEIN_STEPS` env var. Used by the
/// `turbo_klein9b_offload` bench to compare klein9b_infer and
/// klein9b_infer_turbo at the same step count. When unset, defaults to
/// `DEFAULT_NUM_STEPS`.
fn resolved_num_steps() -> usize {
    std::env::var("KLEIN_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_NUM_STEPS)
}
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

fn load_sharded_weights(
    dir: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut all_weights = HashMap::new();
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();
    for (i, path) in shard_paths.iter().enumerate() {
        let t0 = Instant::now();
        let shard = flame_core::serialization::load_file(path, device)?;
        println!("    Shard {}/{}: {} keys ({:.1}s)",
            i + 1, shard_paths.len(), shard.len(), t0.elapsed().as_secs_f32());
        all_weights.extend(shard);
    }
    Ok(all_weights)
}

/// Try loading all-on-GPU. If OOM, fall back to offloaded.
enum Model {
    OnGpu(KleinTransformer),
    Offloaded(KleinOffloaded),
}

impl Model {
    fn forward(&self, img: &Tensor, txt: &Tensor, t: &Tensor, img_ids: &Tensor, txt_ids: &Tensor) -> flame_core::Result<Tensor> {
        match self {
            Model::OnGpu(m) => m.forward(img, txt, t, img_ids, txt_ids),
            Model::Offloaded(m) => m.forward(img, txt, t, img_ids, txt_ids),
        }
    }
    fn config_str(&self) -> String {
        match self {
            Model::OnGpu(m) => format!("{:?} (all on GPU)", m.config()),
            Model::Offloaded(m) => format!("{:?} (offloaded)", m.config()),
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let prompt = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let num_steps = resolved_num_steps();

    println!("============================================================");
    println!("Klein 9B Base — Pure Rust");
    println!("  {}x{}, {} steps, guidance {}, seed {}", WIDTH, HEIGHT, num_steps, GUIDANCE, SEED);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Encode text (Qwen3 8B) — then drop (warms mempool with ~15GB)
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Text Encoding (Qwen3 8B) ---");
    println!("  Prompt: {}", &prompt[..prompt.len().min(80)]);
    let t0 = Instant::now();

    let (pos_hidden, neg_hidden) = {
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Tokenizer: {}", e))?;

        let pos_fmt = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let neg_fmt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", DEFAULT_NEGATIVE);

        let pad_id = 151643i32;
        let mut pos_ids: Vec<i32> = tokenizer.encode(pos_fmt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{}", e))?.get_ids().iter().map(|&id| id as i32).collect();
        let mut neg_ids: Vec<i32> = tokenizer.encode(neg_fmt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{}", e))?.get_ids().iter().map(|&id| id as i32).collect();
        println!("  Tokens: {} pos, {} neg", pos_ids.len(), neg_ids.len());
        pos_ids.resize(512, pad_id);
        neg_ids.resize(512, pad_id);

        let enc_weights = load_sharded_weights(ENCODER_DIR, &device)?;
        let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
        let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());

        let pos_h = encoder.encode(&pos_ids)?;
        let neg_h = encoder.encode(&neg_ids)?;
        println!("  Output: {:?}", pos_h.dims());
        drop(encoder); // frees ~15GB — mempool keeps it cached
        (pos_h, neg_h)
    };
    println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load Klein 9B — try all-on-GPU first, fallback to offloaded
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Klein 9B ---");
    let t0 = Instant::now();

    let model = match KleinTransformer::from_safetensors(MODEL_PATH) {
        Ok(m) => {
            println!("  All weights on GPU");
            Model::OnGpu(m)
        }
        Err(e) => {
            println!("  GPU load failed ({:?}), falling back to offloaded...", e);
            Model::Offloaded(KleinOffloaded::from_safetensors(MODEL_PATH)?)
        }
    };
    println!("  {}", model.config_str());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Noise + schedule + denoise
    // ------------------------------------------------------------------
    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;
    let n_img = latent_h * latent_w;
    let txt_seq_len = 512usize;

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
    let noise = Tensor::from_f32_to_bf16(noise_data, Shape::from_dims(&[1, 128, latent_h, latent_w]), device.clone())?
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, n_img, 128])?;

    let timesteps = get_schedule(num_steps, n_img);

    println!("\n--- Stage 3: Denoise ({} steps, guidance={}) ---", num_steps, GUIDANCE);
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(vec![t_curr], Shape::from_dims(&[1]), device.clone())?;
            let pred_cond = model.forward(x, &pos_hidden, &t_vec, &img_ids, &txt_ids)?;
            let pred_uncond = model.forward(x, &neg_hidden, &t_vec, &img_ids, &txt_ids)?;
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(GUIDANCE)?)
        },
        noise,
        &timesteps,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / num_steps as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);

    let latents = denoised.reshape(&[1, latent_h, latent_w, 128])?.permute(&[0, 3, 1, 2])?;

    let vae_weights = flame_core::serialization::load_file(std::path::Path::new(VAE_PATH), &device)?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    let rgb = vae.decode(&latents)?;
    println!("  {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 5: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let d = rgb_f32.dims();
    let (out_h, out_w) = (d[2], d[3]);

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

    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image creation failed"))?
        .save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
