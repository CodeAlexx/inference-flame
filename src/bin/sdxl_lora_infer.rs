//! SDXL inference with a runtime-applied LoRA — pure Rust.
//!
//! Loads SDXL UNet base, builds SDXLUNet, then attaches a `LoraStack` via
//! `set_lora`. The base weights are NEVER mutated — at every linear
//! chokepoint the model adds `scale * up(down(x))` from any matching LoRA
//! entries to the base output. This matches how ai-toolkit, OneTrainer,
//! and musubi-tuner all apply LoRAs at sampling time.
//!
//! Conv LoRAs (4D weights) are skipped at LoraStack load time — they
//! would need a separate conv runtime path. Text-encoder LoRAs
//! (`lora_te1_*`, `lora_te2_*`) are skipped at LoraStack load time too.
//!
//! Usage:
//!     sdxl_lora_infer \
//!         --base /path/to/sdxl_unet_bf16.safetensors \
//!         --lora /path/to/lora.safetensors \
//!         --vae /path/to/sdxl_vae.safetensors \
//!         --embeddings /path/to/sdxl_embeds.safetensors \
//!         --output /path/to/out.png

use cudarc::driver::CudaDevice;
use flame_core::{
    global_cuda_device, trim_cuda_mempool, DType, Result, Shape, Tensor,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use inference_flame::lora::LoraStack;
use inference_flame::models::sdxl_unet::SDXLUNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

const NUM_TRAIN_STEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

struct Args {
    base: PathBuf,
    lora: PathBuf,
    vae: PathBuf,
    embeddings: PathBuf,
    output: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg_scale: f32,
    seed: u64,
    alpha: Option<f32>,
    rank: Option<usize>,
    multiplier: f32,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        base: PathBuf::from("/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors"),
        lora: PathBuf::new(),
        vae: PathBuf::from(
            "/home/alex/.serenity/models/vaes/OfficialStableDiffusion/sdxl_vae.safetensors",
        ),
        embeddings: PathBuf::new(),
        output: PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/sdxl_lora.png"),
        width: 1024,
        height: 1024,
        steps: 30,
        cfg_scale: 7.5,
        seed: 42,
        alpha: None,
        rank: None,
        multiplier: 1.0,
    };
    let mut i = 1;
    while i < argv.len() {
        let take = |i: &mut usize| -> anyhow::Result<String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| {
                anyhow::anyhow!("missing value for {}", argv[*i - 1])
            })
        };
        match argv[i].as_str() {
            "--base" => a.base = PathBuf::from(take(&mut i)?),
            "--lora" => a.lora = PathBuf::from(take(&mut i)?),
            "--vae" => a.vae = PathBuf::from(take(&mut i)?),
            "--embeddings" | "--embeds" => a.embeddings = PathBuf::from(take(&mut i)?),
            "--output" => a.output = PathBuf::from(take(&mut i)?),
            "--width" => a.width = take(&mut i)?.parse()?,
            "--height" => a.height = take(&mut i)?.parse()?,
            "--steps" => a.steps = take(&mut i)?.parse()?,
            "--cfg" | "--cfg_scale" => a.cfg_scale = take(&mut i)?.parse()?,
            "--seed" => a.seed = take(&mut i)?.parse()?,
            "--alpha" => a.alpha = Some(take(&mut i)?.parse()?),
            "--rank" => a.rank = Some(take(&mut i)?.parse()?),
            "--multiplier" | "--strength" => a.multiplier = take(&mut i)?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "sdxl_lora_infer --base BASE --lora LORA --vae VAE \
--embeddings EMBEDS [--output PNG] [--width W] [--height H] \
[--steps N] [--cfg G] [--seed S] [--alpha A] [--rank R] [--multiplier M]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
        i += 1;
    }
    if a.lora.as_os_str().is_empty() {
        anyhow::bail!("--lora is required");
    }
    if a.embeddings.as_os_str().is_empty() {
        anyhow::bail!("--embeddings is required");
    }
    Ok(a)
}

fn build_sdxl_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let betas: Vec<f64> = (0..NUM_TRAIN_STEPS)
        .map(|i| {
            let v = BETA_START.sqrt()
                + (BETA_END.sqrt() - BETA_START.sqrt()) * i as f64
                    / (NUM_TRAIN_STEPS - 1) as f64;
            v * v
        })
        .collect();
    let mut alphas_cumprod = Vec::with_capacity(NUM_TRAIN_STEPS);
    let mut prod = 1.0f64;
    for &b in &betas {
        prod *= 1.0 - b;
        alphas_cumprod.push(prod);
    }
    let step_ratio = NUM_TRAIN_STEPS / num_steps;
    let mut ts: Vec<usize> = (0..num_steps).map(|i| i * step_ratio + 1).collect();
    ts.reverse();
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    let mut timesteps = Vec::with_capacity(num_steps);
    for &t in &ts {
        let t = t.min(NUM_TRAIN_STEPS - 1);
        let alpha = alphas_cumprod[t];
        let sigma = ((1.0 - alpha) / alpha).sqrt();
        sigmas.push(sigma as f32);
        timesteps.push(t as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

fn save_png(rgb: &Tensor, path: &std::path::Path) -> Result<()> {
    let dims = rgb.shape().dims().to_vec();
    let (h, w) = (dims[2], dims[3]);
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                pixels[(y * w + x) * 3 + c] = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| flame_core::Error::InvalidInput("PNG buffer".into()))?
        .save(path)
        .map_err(|e| flame_core::Error::Io(format!("PNG save: {e}")))
}

fn run(args: Args, device: Arc<CudaDevice>) -> anyhow::Result<()> {
    let t_total = Instant::now();
    let latent_h = args.height / 8;
    let latent_w = args.width / 8;

    println!("=== SDXL LoRA Inference (pure Rust) ===");
    println!("base: {}", args.base.display());
    println!("lora: {}", args.lora.display());
    println!("vae:  {}", args.vae.display());

    // Stage 1: Load embeddings.
    println!("\n--- Stage 1: Load embeddings ---");
    let emb = flame_core::serialization::load_file(&args.embeddings, &device)?;
    let context = emb
        .get("context")
        .ok_or_else(|| anyhow::anyhow!("embeds missing 'context'"))?
        .to_dtype(DType::BF16)?;
    let context_uncond = emb
        .get("context_uncond")
        .ok_or_else(|| anyhow::anyhow!("embeds missing 'context_uncond'"))?
        .to_dtype(DType::BF16)?;
    let y = emb
        .get("y")
        .ok_or_else(|| anyhow::anyhow!("embeds missing 'y'"))?
        .to_dtype(DType::BF16)?;
    let y_uncond = emb
        .get("y_uncond")
        .ok_or_else(|| anyhow::anyhow!("embeds missing 'y_uncond'"))?
        .to_dtype(DType::BF16)?;
    drop(emb);

    // Stage 2: Load base + LoRA, merge, build SDXLUNet.
    println!("\n--- Stage 2: Load base + apply LoRA, build SDXLUNet ---");
    let t_load = Instant::now();
    let all_weights =
        flame_core::serialization::load_file(&args.base, &device)?;
    println!("  base: {} tensors", all_weights.len());

    // Build the runtime LoRA stack BEFORE constructing the model.
    // `LoraStack::load` reads per-module .alpha for every format, so
    // --alpha / --rank CLI args are no longer load-bearing. The kohya
    // SDXL diffusers→LDM rewriter inside LoraStack remaps prefixes when
    // necessary.
    if args.alpha.is_some() || args.rank.is_some() {
        eprintln!(
            "  note: --alpha/--rank are ignored when LoRA file ships per-module .alpha tensors"
        );
    }
    let base_keys: std::collections::HashSet<String> = all_weights.keys().cloned().collect();
    let lora_stack = LoraStack::load(
        args.lora.to_str().expect("lora path utf8"),
        &base_keys,
        args.multiplier,
        &device,
    )?;
    println!(
        "  lora: {} target weight(s), multiplier={:.2}, loaded in {:.1}s",
        lora_stack.target_count(),
        args.multiplier,
        t_load.elapsed().as_secs_f32()
    );
    trim_cuda_mempool(0);

    let mut model = SDXLUNet::from_weights_all_gpu(
        args.base.to_string_lossy().to_string(),
        all_weights,
        device.clone(),
    )?;
    model.set_lora(Arc::new(lora_stack));

    // Stage 3: Sample.
    println!(
        "\n--- Stage 3: Denoise ({} steps, cfg={}) ---",
        args.steps, args.cfg_scale
    );
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let numel = 4 * latent_h * latent_w;
    let mut noise: Vec<f32> = Vec::with_capacity(numel);
    for _ in 0..numel / 2 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        noise.push(r * theta.cos());
        noise.push(r * theta.sin());
    }
    if numel % 2 == 1 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        noise.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let (sigmas, timesteps) = build_sdxl_schedule(args.steps);
    let init_sigma = sigmas[0];
    let scaled: Vec<f32> = noise.iter().map(|v| v * init_sigma).collect();
    let mut x_f32 = Tensor::from_vec(
        scaled,
        Shape::from_dims(&[1, 4, latent_h, latent_w]),
        device.clone(),
    )?;

    let t_denoise = Instant::now();
    for i in 0..args.steps {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_in = x_f32.mul_scalar(c_in)?.to_dtype(DType::BF16)?;
        let timestep = Tensor::from_f32_to_bf16(
            vec![timesteps[i]],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;
        let pred_cond = model.forward(&x_in, &timestep, &context, &y)?;
        let pred = if args.cfg_scale > 1.0 {
            let pred_uncond = model.forward(&x_in, &timestep, &context_uncond, &y_uncond)?;
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(args.cfg_scale)?)?
        } else {
            pred_cond
        };
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let dt = sigma_next - sigma;
        x_f32 = x_f32.add(&pred_f32.mul_scalar(dt)?)?;
    }
    println!(
        "  denoised in {:.1}s ({:.2} s/step)",
        t_denoise.elapsed().as_secs_f32(),
        t_denoise.elapsed().as_secs_f32() / args.steps as f32
    );

    let x = x_f32.to_dtype(DType::BF16)?;

    // Stage 4: VAE decode (drop UNet first).
    println!("\n--- Stage 4: VAE decode ---");
    drop(model);
    trim_cuda_mempool(0);

    let vae = LdmVAEDecoder::from_safetensors(
        &args.vae.to_string_lossy(),
        4,
        0.13025,
        0.0,
        &device,
    )?;
    let t_vae = Instant::now();
    let rgb = vae.decode(&x)?;
    println!(
        "  decoded {:?} in {:.1}s",
        rgb.shape().dims(),
        t_vae.elapsed().as_secs_f32()
    );

    save_png(&rgb, &args.output)?;
    println!("\nIMAGE SAVED: {}", args.output.display());
    println!("Total time: {:.1}s", t_total.elapsed().as_secs_f32());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args()?;
    let device = global_cuda_device();
    run(args, device)
}
