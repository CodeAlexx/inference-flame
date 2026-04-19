//! ERNIE-Image inference — text encode → flow matching denoise → VAE decode → PNG.
//!
//! Usage: ernie_image_infer ["prompt text" [output.png]]
//!
//! Pipeline:
//!   1. Mistral-3 3B text encode → dump from VRAM
//!   2. Load ERNIE-Image DiT (all blocks resident on GPU)
//!   3. Flow matching Euler denoise (shift=3.0, sequential CFG)
//!   4. Drop DiT → VAE decode → PNG

use inference_flame::models::ernie_image::{ErnieImageConfig, ErnieImageModel};
use inference_flame::models::mistral3b_encoder::Mistral3bEncoder;
use inference_flame::vae::klein_vae::KleinVaeDecoder;
use inference_flame::sampling::ernie_sampling::{ernie_schedule, ernie_euler_step, sigma_to_timestep};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const TRANSFORMER_DIR: &str = "/home/alex/models/ERNIE-Image/transformer";
const TEXT_ENCODER: &str = "/home/alex/models/ERNIE-Image/text_encoder/model.safetensors";
const TOKENIZER: &str = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json";
const VAE_PATH: &str = "/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors";

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;
// Baidu's ErnieImagePipeline default (pipeline_ernie_image.py:204). Lower
// step counts produce visibly worse quality — this is SFT not Turbo.
const STEPS: usize = 50;
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
    let output = std::env::args().nth(2).unwrap_or_else(|| "/home/alex/EriDiffusion/inference-flame/output/ernie_cat_1024.png".to_string());

    let device = global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();

    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;

    println!("=== ERNIE-Image 1024x1024 (resident + sequential CFG) ===");
    println!("  prompt: {prompt:?}");
    println!("  size: {WIDTH}x{HEIGHT}, latent: {latent_w}x{latent_h}");
    println!("  steps: {STEPS}, guidance: {GUIDANCE}, seed: {SEED}");

    // ---------------------------------------------------------------
    // Stage 1: Text Encoding (Mistral-3 3B) — then drop encoder
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

    // Diffusers ErnieImagePipeline uses an encoded empty string for the
    // unconditional branch, NOT zeros (pipeline_ernie_image.py:280-298).
    // Passing zeros as uncond_embeds pushes CFG in a wrong direction — it
    // biases toward "anything-not-the-prompt" instead of "generic image",
    // which is visible as stylization drift.
    let empty_encoding = tokenizer.encode("", true)
        .map_err(|e| anyhow::anyhow!("tokenize empty: {e}"))?;
    let empty_ids: Vec<i32> = empty_encoding.get_ids().iter().map(|&id| id as i32).collect();
    let empty_len = empty_ids.len();
    println!("  uncond tokenized: {empty_len} tokens");

    let (text_embeds, uncond_embeds_real) = {
        let encoder = Mistral3bEncoder::load(TEXT_ENCODER, &device)?;
        println!("  encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let embeds = encoder.encode(&token_ids, 256)?;
        println!("  encoded: shape={:?}", embeds.shape().dims());
        let uncond = encoder.encode(&empty_ids, 256)?;
        println!("  uncond encoded: shape={:?}", uncond.shape().dims());
        (embeds, uncond)
        // encoder dropped here — ~7GB freed
    };
    println!("  text encoding done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---------------------------------------------------------------
    // Stage 2: Load Transformer (all resident on GPU — ~15GB)
    // ---------------------------------------------------------------
    println!("\n[2/4] Loading ERNIE-Image DiT (all blocks resident)...");
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

    let mut all_weights = std::collections::HashMap::new();
    for path in &shard_paths {
        let partial = flame_core::serialization::load_file(path, &device)?;
        for (k, v) in partial {
            all_weights.insert(k, v);
        }
    }

    let config = ErnieImageConfig::default();
    let model = ErnieImageModel::load(all_weights, config.clone())?;
    println!("  DiT loaded in {:.1}s ({} blocks on GPU)", t1.elapsed().as_secs_f32(), config.num_layers);

    // ---------------------------------------------------------------
    // Stage 3: Denoise — SEQUENTIAL CFG (one pass at a time)
    // ---------------------------------------------------------------
    println!("\n[3/4] Denoising ({STEPS} steps, sequential CFG)...");
    let t2 = Instant::now();

    let sigmas = ernie_schedule(STEPS);

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

    let text_3d = if text_embeds.rank() == 2 {
        text_embeds.unsqueeze(0)?
    } else {
        text_embeds.clone()
    };
    let text_lens = vec![text_len.min(256)];

    let uncond_3d = if uncond_embeds_real.rank() == 2 {
        uncond_embeds_real.unsqueeze(0)?
    } else {
        uncond_embeds_real.clone()
    };
    let uncond_embeds = uncond_3d;
    let uncond_lens = vec![empty_len.min(256)];

    // Trim text embeddings to real length — ERNIE has no attention mask path,
    // so zero-padded text tokens steal softmax weight from real text and
    // dilute conditioning (~1/256 for uncond, ~32/256 for cond). Reference
    // pipeline pads to max(cond,uncond) = 32 when batching B=2; Rust runs
    // sequentially so each forward gets its own real length with no padding.
    let cond_trim = text_3d.narrow(1, 0, text_lens[0])?;
    let uncond_trim = uncond_embeds.narrow(1, 0, uncond_lens[0])?;
    let cond_lens = vec![text_lens[0]];
    let uncond_trim_lens = vec![uncond_lens[0]];

    for step in 0..STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

        // Sequential CFG: run cond pass, then uncond pass — never both at once.
        // Without periodic pool release, per-forward transients (attention
        // scores ~2–4 GB, rope tables, intermediates) accumulate and OOM by
        // ~step 5. Same pattern as flux1_dit's between-block trim.
        let pred = if GUIDANCE > 1.0 {
            let pred_cond = model.forward(&latent, &t_tensor, &cond_trim, &cond_lens)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            let pred_uncond = model.forward(&latent, &t_tensor, &uncond_trim, &uncond_trim_lens)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(GUIDANCE)?)?
        } else {
            let p = model.forward(&latent, &t_tensor, &cond_trim, &cond_lens)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            p
        };
        latent = ernie_euler_step(&latent, &pred, sigma, sigma_next)?;

        if step % 5 == 0 || step == STEPS - 1 {
            println!("  step {}/{STEPS} sigma={sigma:.4} t={t:.1} ({:.1}s)",
                step + 1, t2.elapsed().as_secs_f32());
        }
    }
    println!("  denoising done in {:.1}s ({:.1}s/step)",
        t2.elapsed().as_secs_f32(),
        t2.elapsed().as_secs_f32() / STEPS as f32);

    {
        let v = latent.to_vec_f32()?;
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let std: f32 = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let nan = v.iter().filter(|x| x.is_nan()).count();
        println!("  latent stats: mean={mean:.4} std={std:.4} nan={nan}");
    }

    // ---------------------------------------------------------------
    // Stage 4: Drop DiT → VAE Decode
    // ---------------------------------------------------------------
    println!("\n[4/4] VAE decode...");
    let t3 = Instant::now();

    drop(model);
    // Return DiT allocations to the driver — the pool caches freed blocks,
    // and without this VAE decode OOMs on the 1024×128×1024×1024 conv2d
    // intermediate even though the DiT has been dropped from Rust-side
    // ownership.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);

    let vae_weights = flame_core::serialization::load_file(VAE_PATH, &device)?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    drop(vae_weights);

    let decoded = vae.decode(&latent)?;
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
