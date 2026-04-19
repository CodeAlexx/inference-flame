//! Chroma — Stage 2: DiT denoise + VAE decode + PNG.
//!
//! Loads pre-cached T5 embeddings (produced by `chroma_encode`), the Chroma
//! DiT (BlockOffloader), and the VAE. Runs the CFG denoise loop and saves a PNG.
//! T5 is NEVER loaded in this binary so we never have T5 + DiT in VRAM at the
//! same time.
//!
//! ## Usage
//! ```bash
//! ./target/release/chroma_encode "your prompt" "" /tmp/cache.safetensors
//! ./target/release/chroma_gen   /tmp/cache.safetensors  /tmp/out.png
//! ```
//!
//! Optional env vars:
//!   `CHROMA_HEIGHT`     default 1024
//!   `CHROMA_WIDTH`      default 1024
//!   `CHROMA_STEPS`      default 40
//!   `CHROMA_GUIDANCE`   default 4.0
//!   `CHROMA_SEED`       default 42
//!   `CHROMA_DIT_SHARDS` colon-separated paths (overrides built-in)
//!   `CHROMA_VAE`        VAE safetensors path (overrides built-in)

use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::chroma_dit::ChromaDit;
use inference_flame::sampling::flux1_sampling::{
    get_schedule, pack_latent, unpack_latent,
};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

const DEFAULT_DIT_SHARDS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
];

const DEFAULT_VAE: &str =
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors";

const DEFAULT_EMBEDDINGS: &str = "/home/alex/EriDiffusion/inference-flame/output/chroma_embeddings.safetensors";
const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/chroma_rust.png";

// FLUX VAE constants — Chroma uses the same VAE.
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;
const T5_SEQ_LEN: usize = 512;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeddings_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_EMBEDDINGS.to_string());
    let output_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUTPUT.to_string());

    // Knobs
    let height: usize = env_usize("CHROMA_HEIGHT", 1024);
    let width: usize = env_usize("CHROMA_WIDTH", 1024);
    let num_steps: usize = env_usize("CHROMA_STEPS", 40);
    let guidance_scale: f32 = env_f32("CHROMA_GUIDANCE", 4.0);
    let seed: u64 = env_u64("CHROMA_SEED", 42);

    // DiT shards (colon-separated for env override)
    let dit_shards_owned: Vec<String> = match std::env::var("CHROMA_DIT_SHARDS") {
        Ok(s) => s.split(':').map(|p| p.to_string()).collect(),
        Err(_) => DEFAULT_DIT_SHARDS.iter().map(|s| s.to_string()).collect(),
    };
    let dit_shards: Vec<&str> = dit_shards_owned.iter().map(|s| s.as_str()).collect();

    let vae_path = std::env::var("CHROMA_VAE").unwrap_or_else(|_| DEFAULT_VAE.to_string());

    println!("=== Chroma — Stage 2 (DiT + VAE + PNG) ===");
    println!("Embeddings: {}", embeddings_path);
    println!("Output:     {}", output_path);
    println!("Size:       {}x{}, steps={}, guidance={}", width, height, num_steps, guidance_scale);
    println!("Seed:       {}", seed);
    println!();

    // ------------------------------------------------------------------
    // Stage A: Load cached T5 embeddings (small — ~4 MB each at 1×512×4096 BF16)
    // ------------------------------------------------------------------
    println!("--- Loading cached embeddings ---");
    let t0 = Instant::now();
    let embeddings = flame_core::serialization::load_file(
        std::path::Path::new(&embeddings_path),
        &device,
    )?;
    let t5_cond = embeddings.get("t5_cond")
        .ok_or_else(|| anyhow::anyhow!("Missing t5_cond in embeddings file"))?
        .clone();
    let t5_uncond = embeddings.get("t5_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing t5_uncond in embeddings file"))?
        .clone();
    drop(embeddings);
    let t5_cond = ensure_bf16(t5_cond)?;
    let t5_uncond = ensure_bf16(t5_uncond)?;
    println!("  cond:   {:?}", t5_cond.shape().dims());
    println!("  uncond: {:?}", t5_uncond.shape().dims());
    // Pre-stack [cond, uncond] into a single B=2 tensor. CFG loop runs
    // a single forward per step instead of two, halving block fetches.
    let t5_batched = Tensor::cat(&[&t5_cond, &t5_uncond], 0)?;
    drop(t5_cond);
    drop(t5_uncond);
    println!("  batched: {:?}", t5_batched.shape().dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage B: Load Chroma DiT (BlockOffloader)
    // ------------------------------------------------------------------
    println!("--- Loading Chroma DiT (BlockOffloader) ---");
    let t0 = Instant::now();
    let mut dit = ChromaDit::load(&dit_shards, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage C: Build noise + ids, run CFG denoise
    // ------------------------------------------------------------------
    println!("--- Denoise ({} steps, CFG={}) ---", num_steps, guidance_scale);

    // FLUX-style latent geometry: VAE 8x + patchify 2x = 16x effective.
    let latent_h = 2 * ((height + 15) / 16);
    let latent_w = 2 * ((width + 15) / 16);
    println!("  Latent [B,C,H,W] = [1, {}, {}, {}]", AE_IN_CHANNELS, latent_h, latent_w);

    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
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
    let n_img = img_packed.shape().dims()[1];
    println!("  Packed img: {:?}", img_packed.shape().dims());
    println!("  img_ids:    {:?}", img_ids.shape().dims());

    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )?;

    // FLUX-style flow-match Euler shift schedule. Chroma is NOT distilled,
    // so we run real CFG (2 forwards per step).
    let timesteps = get_schedule(num_steps, n_img, 0.5, 1.15, true);
    println!("  Schedule: {} steps, t[0]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1, timesteps[0], timesteps[num_steps]);
    println!();

    // CFG Euler loop
    let t_denoise = Instant::now();
    let mut x = img_packed;
    for step in 0..num_steps {
        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        // Scoped block: every per-step temporary drops at the closing `}`
        // BEFORE the next iteration allocates. Prevents VRAM accumulation.
        let next_x = {
            // Batched timestep [t, t] so the approximator and SDPA run in
            // one B=2 pass that serves both cond (row 0) and uncond (row 1).
            let t_vec = Tensor::from_vec(
                vec![t_curr, t_curr],
                Shape::from_dims(&[2]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            let (pooled_temb, pe_cos, pe_sin) =
                dit.precompute_step_cache(&t_vec, &img_ids, &txt_ids)?;

            // Stack x twice to match B=2. Same latent feeds both rows; the
            // split happens in the output based on the T5 side.
            let x_batched = Tensor::cat(&[&x, &x], 0)?;
            let preds = dit.forward_cached(&x_batched, &t5_batched, &pooled_temb, &pe_cos, &pe_sin)?;
            drop(x_batched);

            // preds[0] = cond, preds[1] = uncond (matches t5_batched order).
            let cond_pred = preds.narrow(0, 0, 1)?;
            let uncond_pred = preds.narrow(0, 1, 1)?;

            // noise = uncond + scale * (cond - uncond)
            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(guidance_scale)?;
            let pred = uncond_pred.add(&scaled)?;

            // Euler step: x_next = x + dt * pred
            let step_tensor = pred.mul_scalar(dt)?;
            x.add(&step_tensor)?
        };
        x = next_x;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps {
            println!("  step {}/{}  t={:.4}  ({:.1}s elapsed)",
                step + 1, num_steps, t_curr, t_denoise.elapsed().as_secs_f32());
        }
    }
    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 1 batched B=2 forward/step)",
        dt_denoise,
        dt_denoise / num_steps as f32,
    );
    println!();

    // Drop DiT before VAE load (DiT BlockOffloader holds ~17 GB worth of weights).
    drop(dit);
    drop(t5_batched);
    println!("  DiT + cached embeddings evicted");

    // ------------------------------------------------------------------
    // Stage D: Unpack + VAE decode
    // ------------------------------------------------------------------
    println!("--- VAE decode ---");
    let t0 = Instant::now();

    let latent = unpack_latent(&x, height, width)?;
    drop(x);
    println!("  Unpacked latent: {:?}", latent.shape().dims());

    // Save unpacked latent for cross-tool VAE bisection. Gated on
    // CHROMA_SAVE_LATENT=1 so default runs don't write debug files.
    if std::env::var("CHROMA_SAVE_LATENT").is_ok() {
        use std::collections::HashMap;
        use std::path::Path;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("latent".to_string(), latent.clone_result()?);
        let p = Path::new("/home/alex/EriDiffusion/inference-flame/output/chroma_rust_latent.safetensors");
        flame_core::serialization::save_file(&m, p)?;
        println!("  saved latent -> {}", p.display());
    }

    let vae = LdmVAEDecoder::from_safetensors(
        &vae_path,
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
    println!();

    // ------------------------------------------------------------------
    // Stage E: Denormalize → PNG
    // ------------------------------------------------------------------
    println!("--- Save PNG ---");
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec_f32()?;
    let dims = rgb_f32.shape().dims().to_vec();
    let (out_c, out_h, out_w) = (dims[1], dims[2], dims[3]);
    if out_c != 3 {
        return Err(anyhow::anyhow!("VAE decoder must return 3 channels, got {}", out_c));
    }

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

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to build RgbImage"))?;
    img.save(&output_path)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", output_path);
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    let _ = device;
    Ok(())
}

fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        Ok(t.to_dtype(DType::BF16)?)
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
