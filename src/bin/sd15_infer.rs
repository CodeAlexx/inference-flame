//! Stable Diffusion 1.5 image generation — pure Rust, flame-core.
//!
//! Full pipeline: prompt → CLIP-L → SD 1.5 UNet denoise → VAE → PNG.
//! Native resolution 512×512, 30 steps, CFG 7.5, Euler discrete scheduler.
//!
//! Usage: sd15_infer ["prompt text"]

use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::sd15_unet::SD15UNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const SNAPSHOT: &str = "/home/alex/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14";

fn unet_path() -> String {
    format!("{SNAPSHOT}/unet/diffusion_pytorch_model.safetensors")
}
fn vae_path() -> String {
    format!("{SNAPSHOT}/vae/diffusion_pytorch_model.safetensors")
}
fn clip_l_path() -> String {
    format!("{SNAPSHOT}/text_encoder/model.safetensors")
}
// Reuse the FLUX-shipped CLIP-L tokenizer JSON — CLIP-L BPE is identical for SD 1.5.
const CLIP_L_TOKENIZER: &str =
    "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sd15_sample.png";

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------

const DEFAULT_PROMPT: &str =
    "a photorealistic portrait of an elderly fisherman mending a net, Vermeer lighting";

const DEFAULT_STEPS: usize = 30;
const DEFAULT_CFG: f32 = 7.5;
const SEED: u64 = 42;
const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn cfg_scale() -> f32 {
    std::env::var("SD15_CFG").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_CFG)
}
fn num_steps() -> usize {
    std::env::var("SD15_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_STEPS)
}

const CLIP_SEQ_LEN: usize = 77;

// SD 1.5 VAE: 4 ch latent, scale=0.18215, shift=0.0
const VAE_SCALE: f32 = 0.18215;
const VAE_SHIFT: f32 = 0.0;

// ---------------------------------------------------------------------------
// Scheduler (same as SDXL: scaled-linear β, leading spacing, steps_offset=1)
// ---------------------------------------------------------------------------

fn build_sd15_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

    let betas: Vec<f64> = (0..num_train_steps)
        .map(|i| {
            let v = beta_start.sqrt()
                + (beta_end.sqrt() - beta_start.sqrt()) * i as f64 / (num_train_steps - 1) as f64;
            v * v
        })
        .collect();

    let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
    let mut prod = 1.0f64;
    for &b in &betas {
        prod *= 1.0 - b;
        alphas_cumprod.push(prod);
    }

    let step_ratio = num_train_steps / num_steps;
    let mut ts: Vec<usize> = (0..num_steps).map(|i| i * step_ratio + 1).collect();
    ts.reverse();

    let mut sigmas = Vec::with_capacity(num_steps + 1);
    let mut timesteps = Vec::with_capacity(num_steps);
    for &t in &ts {
        let t = t.min(num_train_steps - 1);
        let alpha = alphas_cumprod[t];
        let sigma = ((1.0 - alpha) / alpha).sqrt();
        sigmas.push(sigma as f32);
        timesteps.push(t as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

// ---------------------------------------------------------------------------
// CLIP-L tokenizer (pad to 77 with eos=49407)
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(CLIP_L_TOKENIZER) {
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
            eprintln!("[sd15] CLIP tokenizer failed: {e}; using BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

fn load_clip_weights(
    path: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    let raw = flame_core::serialization::load_file(std::path::Path::new(path), device)?;
    let mut weights = std::collections::HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
        weights.insert(k, t);
    }
    Ok(weights)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let prompt = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let steps = num_steps();
    let cfg = cfg_scale();

    println!("============================================================");
    println!("SD 1.5 — Pure Rust Inference");
    println!("  {WIDTH}x{HEIGHT}, {steps} steps, CFG {cfg}, seed {SEED}");
    println!("  prompt: {prompt:?}");
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: CLIP-L text encoding (inline, not cached)
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: CLIP-L encode ---");
    let t0 = Instant::now();
    let (context, context_uncond) = {
        let weights = load_clip_weights(&clip_l_path(), &device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, _pooled) = clip.encode(&tokenize_clip(&prompt))?;
        let (hu, _pooled_u) = clip.encode(&tokenize_clip(""))?;
        (hc, hu)
    };
    println!(
        "  cond: {:?}, uncond: {:?} ({:.1}s)",
        context.dims(),
        context_uncond.dims(),
        t0.elapsed().as_secs_f32()
    );

    // ------------------------------------------------------------------
    // Stage 2: Load SD 1.5 UNet
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load SD 1.5 UNet ---");
    let t0 = Instant::now();
    let mut model = SD15UNet::from_safetensors_all_gpu(&unet_path(), &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Noise + denoise (Euler, eps prediction)
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Denoise ({steps} steps, CFG={cfg}) ---");

    let latent_h = HEIGHT / 8;
    let latent_w = WIDTH / 8;
    let numel = 4 * latent_h * latent_w;

    // Box–Muller Gaussian noise (matches sdxl_infer.rs)
    let noise_tensor = {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut data = vec![0.0f32; numel];
        for v in &mut data {
            let u1: f32 = rand::Rng::gen_range(&mut rng, 1e-7f32..1.0);
            let u2: f32 = rand::Rng::gen_range(&mut rng, 0.0f32..std::f32::consts::TAU);
            *v = (-2.0 * u1.ln()).sqrt() * u2.cos();
        }
        Tensor::from_vec(data, Shape::from_dims(&[1, 4, latent_h, latent_w]), device.clone())?
    };
    println!("  Generated noise: {:?}", noise_tensor.shape().dims());

    let (sigmas, timesteps) = build_sd15_schedule(steps);
    println!("  sigma_max={:.4}, sigma_min={:.6}", sigmas[0], sigmas[steps - 1]);
    println!("  t_max={:.0}, t_min={:.0}", timesteps[0], timesteps[steps - 1]);

    // x = noise * init_noise_sigma
    let init_sigma = (sigmas[0] * sigmas[0] + 1.0).sqrt();
    let x = noise_tensor.to_dtype(DType::BF16)?.mul_scalar(init_sigma)?;

    let t0 = Instant::now();
    let mut x_f32 = x.to_dtype(DType::F32)?;

    for i in 0..steps {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_in = x_f32.mul_scalar(c_in)?.to_dtype(DType::BF16)?;

        let timestep = Tensor::from_f32_to_bf16(
            vec![timesteps[i]], Shape::from_dims(&[1]), device.clone(),
        )?;

        let pred_cond = model.forward(&x_in, &timestep, &context)?;
        let pred_uncond = model.forward(&x_in, &timestep, &context_uncond)?;

        // CFG in FP32
        let pred_cond_f32 = pred_cond.to_dtype(DType::F32)?;
        let pred_uncond_f32 = pred_uncond.to_dtype(DType::F32)?;
        let diff = pred_cond_f32.sub(&pred_uncond_f32)?;
        let pred_f32 = pred_uncond_f32.add(&diff.mul_scalar(cfg)?)?;

        // Euler step in FP32
        let dt = sigma_next - sigma;
        x_f32 = x_f32.add(&pred_f32.mul_scalar(dt)?)?;

        if i == 0 || i == steps - 1 {
            let xd = x_f32.to_vec()?;
            let x_abs: f32 = xd.iter().map(|v| v.abs()).sum::<f32>() / xd.len() as f32;
            println!("  Step {}/{}: x_abs={:.4}", i + 1, steps, x_abs);
        }
    }

    let x = x_f32.to_dtype(DType::BF16)?;
    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / steps as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE Decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    drop(model);

    // SD 1.5's VAE safetensors uses the pre-0.14 diffusers attention key
    // names (`query`/`key`/`value`/`proj_attn`), while our LDM remap in
    // ldm_decoder.rs handles only the modern `to_q`/`to_k`/`to_v`/`to_out.0`
    // naming. Pre-rename legacy → modern so the downstream loader succeeds.
    let vae = {
        let raw = flame_core::serialization::load_file(std::path::Path::new(&vae_path()), &device)?;
        let mut w = std::collections::HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            // Only carry decoder + post_quant_conv keys through
            let is_decoder = k.starts_with("decoder.")
                || k.starts_with("first_stage_model.decoder.")
                || k == "post_quant_conv.weight"
                || k == "post_quant_conv.bias"
                || k == "first_stage_model.post_quant_conv.weight"
                || k == "first_stage_model.post_quant_conv.bias";
            if !is_decoder {
                continue;
            }
            // Strip first_stage_model prefix if present
            let k = k.strip_prefix("first_stage_model.").unwrap_or(&k).to_string();
            // Legacy → modern attention key renames
            let k = k
                .replace("attentions.0.query.", "attentions.0.to_q.")
                .replace("attentions.0.key.",   "attentions.0.to_k.")
                .replace("attentions.0.value.", "attentions.0.to_v.")
                .replace("attentions.0.proj_attn.", "attentions.0.to_out.0.");
            let v_bf16 = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
            w.insert(k, v_bf16);
        }
        // Now apply the standard diffusers → LDM decoder remap (defined in
        // ldm_decoder.rs). We can't call it directly (private), so we use
        // the same `from_weights` entry point that `from_safetensors` uses
        // after calling its own remap — but to get the remap we need to
        // route through `from_safetensors`. Alternative: write the remapped
        // map to a temp safetensors file and let `from_safetensors` do its
        // thing. Easier: just save-and-load.
        let tmp_path = "/tmp/sd15_vae_remapped.safetensors";
        flame_core::serialization::save_file(&w, std::path::Path::new(tmp_path))?;
        LdmVAEDecoder::from_safetensors(tmp_path, 4, VAE_SCALE, VAE_SHIFT, &device)?
    };
    let decoded = vae.decode(&x)?;
    let dec_dims = decoded.shape().dims();
    println!("  Decoded: {:?}", dec_dims);

    let img_data = decoded.to_vec_f32()?;
    let (c, h, w) = (dec_dims[1], dec_dims[2], dec_dims[3]);
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for xx in 0..w {
            for ch in 0..3.min(c) {
                let v = img_data[ch * h * w + y * w + xx];
                let v = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                rgb[(y * w + xx) * 3 + ch] = v;
            }
        }
    }
    std::fs::create_dir_all(
        std::path::Path::new(OUTPUT_PATH)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )?;
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow::anyhow!("failed to create image"))?;
    img.save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("IMAGE SAVED: {OUTPUT_PATH}");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
