//! SDXL image generation — pure Rust.
//!
//! Usage: sdxl_infer [path_to_cached_embeddings.safetensors]
//!
//! Cached embeddings must contain:
//!   "context": [B, 77, 2048] — CLIP-L + CLIP-G cross-attention hidden states
//!   "y": [B, 2816] — pooled embedding (CLIP-L 768d + CLIP-G 1280d + zeros 768d)
//!
//! Generate cached embeddings with a Python script (CLIP encoders not in Rust yet).

use inference_flame::models::sdxl_unet::SDXLUNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors"; // VAE from combined checkpoint
const DEFAULT_EMB_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_rust.png";

const NUM_STEPS: usize = 30;
const CFG_SCALE: f32 = 7.5;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

// SDXL uses discrete noise schedule (beta_start=0.00085, beta_end=0.012, 1000 steps)
// Returns (sigmas, timesteps) — sigmas for Euler stepping, timesteps for UNet input
fn build_sdxl_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

    // Scaled-linear beta schedule (SDXL default: beta_schedule="scaled_linear")
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

    // Leading timestep spacing with steps_offset=1 (SDXL EulerDiscreteScheduler default)
    let step_ratio = num_train_steps / num_steps; // integer division
    let mut ts: Vec<usize> = (0..num_steps).map(|i| i * step_ratio + 1).collect();
    ts.reverse(); // high noise first

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

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let emb_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_EMB_PATH.to_string());

    println!("============================================================");
    println!("SDXL — Pure Rust Inference");
    println!("  {}x{}, {} steps, CFG {}, seed {}", WIDTH, HEIGHT, NUM_STEPS, CFG_SCALE, SEED);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(std::path::Path::new(&emb_path), &device)?;
    let context = emb.get("context")
        .ok_or_else(|| anyhow::anyhow!("Missing 'context' in embeddings. Run cache_sdxl_embeddings.py first."))?
        .clone();
    let y = emb.get("y")
        .ok_or_else(|| anyhow::anyhow!("Missing 'y' in embeddings."))?
        .clone();
    let context_uncond = emb.get("context_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'context_uncond' in embeddings."))?
        .clone();
    let y_uncond = emb.get("y_uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'y_uncond' in embeddings."))?
        .clone();
    drop(emb);
    println!("  context: {:?}, y: {:?}", context.dims(), y.dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load SDXL UNet
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load SDXL UNet ---");
    let t0 = Instant::now();
    let mut model = SDXLUNet::from_safetensors_all_gpu(MODEL_PATH, &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Noise + denoise (Euler, eps prediction)
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, CFG_SCALE);

    let latent_h = HEIGHT / 8; // SDXL uses /8 downscale
    let latent_w = WIDTH / 8;
    let numel = 4 * latent_h * latent_w; // 4 latent channels

    // Generate initial noise
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

    let (sigmas, timesteps) = build_sdxl_schedule(NUM_STEPS);
    println!("  sigma_max={:.4}, sigma_min={:.6}", sigmas[0], sigmas[NUM_STEPS - 1]);
    println!("  timestep_max={:.0}, timestep_min={:.0}", timesteps[0], timesteps[NUM_STEPS - 1]);

    // Initialize x = noise * init_noise_sigma (sqrt(sigma_max^2 + 1) per diffusers)
    let init_sigma = (sigmas[0] * sigmas[0] + 1.0).sqrt();
    let mut x = noise_tensor.to_dtype(DType::BF16)?.mul_scalar(init_sigma)?;

    // Save initial noise (pre-init_sigma) so a Python reference can start from
    // the bit-identical seed.
    if std::env::var("SDXL_SAVE_INIT_NOISE").is_ok() {
        use std::collections::HashMap;
        use std::path::Path;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("noise".to_string(), noise_tensor.to_dtype(DType::BF16)?);
        let p = Path::new("/home/alex/EriDiffusion/inference-flame/output/sdxl_init_noise.safetensors");
        flame_core::serialization::save_file(&m, p)?;
        println!("  saved init noise -> {}", p.display());
    }

    let t0 = Instant::now();
    // Keep denoising state in FP32 — only convert to BF16 for UNet input
    let mut x_f32 = x.to_dtype(DType::F32)?;
    drop(x);

    for i in 0..NUM_STEPS {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Scale input: x_in = x / sqrt(sigma^2 + 1), convert to BF16 for UNet
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_in = x_f32.mul_scalar(c_in)?.to_dtype(DType::BF16)?;

        // UNet expects discrete timestep (0-999), NOT sigma
        let timestep = Tensor::from_f32_to_bf16(
            vec![timesteps[i]], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Conditional + unconditional predictions (eps-prediction)
        let pred_cond = model.forward(&x_in, &timestep, &context, &y)?;
        let pred_uncond = model.forward(&x_in, &timestep, &context_uncond, &y_uncond)?;

        // Multi-step parity save: save x_in and pred_cond/pred_uncond at
        // several steps so we can track UNet accuracy vs diffusers across
        // the sigma range.
        if [0usize, 5, 15, 25, 29].contains(&i) && std::env::var("SDXL_SAVE_STEPS").is_ok() {
            use std::collections::HashMap;
            use std::path::Path;
            let mut m: HashMap<String, Tensor> = HashMap::new();
            m.insert("x_in".to_string(), x_in.clone_result()?);
            m.insert("pred_cond".to_string(), pred_cond.clone_result()?);
            m.insert("pred_uncond".to_string(), pred_uncond.clone_result()?);
            let p_str = format!(
                "/home/alex/EriDiffusion/inference-flame/output/sdxl_step{i:02}.safetensors"
            );
            let p = Path::new(&p_str);
            flame_core::serialization::save_file(&m, p)?;
            eprintln!("  saved step{i:02} parity data t={} sigma={}", timesteps[i], sigma);
        }

        // Step-0 single-forward parity: save UNet inputs + outputs so a
        // Python reference can feed the exact same tensors into diffusers UNet
        // and diff layer outputs.
        if i == 0 && std::env::var("SDXL_SAVE_STEP0").is_ok() {
            use std::collections::HashMap;
            use std::path::Path;
            let mut m: HashMap<String, Tensor> = HashMap::new();
            m.insert("x_in".to_string(), x_in.clone_result()?);
            m.insert("timestep".to_string(), timestep.clone_result()?);
            m.insert("context".to_string(), context.clone_result()?);
            m.insert("y".to_string(), y.clone_result()?);
            m.insert("context_uncond".to_string(), context_uncond.clone_result()?);
            m.insert("y_uncond".to_string(), y_uncond.clone_result()?);
            m.insert("pred_cond".to_string(), pred_cond.clone_result()?);
            m.insert("pred_uncond".to_string(), pred_uncond.clone_result()?);
            let p = Path::new("/home/alex/EriDiffusion/inference-flame/output/sdxl_step0.safetensors");
            flame_core::serialization::save_file(&m, p)?;
            println!("  saved step0 parity data -> {}", p.display());
        }

        // CFG in FP32
        let pred_cond_f32 = pred_cond.to_dtype(DType::F32)?;
        let pred_uncond_f32 = pred_uncond.to_dtype(DType::F32)?;
        let diff = pred_cond_f32.sub(&pred_uncond_f32)?;
        let pred_f32 = pred_uncond_f32.add(&diff.mul_scalar(CFG_SCALE)?)?;

        // Euler step in FP32
        let dt = sigma_next - sigma;
        x_f32 = x_f32.add(&pred_f32.mul_scalar(dt)?)?;

        if i == 0 || i == NUM_STEPS - 1 {
            let xd = x_f32.to_vec()?;
            let x_abs: f32 = xd.iter().map(|v| v.abs()).sum::<f32>() / xd.len() as f32;
            println!("  Step {}/{}: x_abs={:.4}", i + 1, NUM_STEPS, x_abs);
        }
    }

    // Convert final latent back to BF16 for saving
    let x = x_f32.to_dtype(DType::BF16)?;
    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 4: VAE Decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE Decode ---");
    drop(model);

    // Save post-UNet, pre-VAE latent for cross-tool bisection. When
    // SDXL_SAVE_LATENT=1 is set, writes the BF16 latent to
    // output/sdxl_rust_latent.safetensors so it can be decoded by diffusers
    // for comparison, OR a diffusers-produced latent can be loaded in here.
    if std::env::var("SDXL_SAVE_LATENT").is_ok() {
        use std::collections::HashMap;
        use std::path::Path;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("latent".to_string(), x.clone_result()?);
        let p = Path::new("/home/alex/EriDiffusion/inference-flame/output/sdxl_rust_latent.safetensors");
        flame_core::serialization::save_file(&m, p)?;
        println!("  saved latent -> {}", p.display());
    }

    let _vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = LdmVAEDecoder::from_safetensors(VAE_PATH, 4, 0.13025, 0.0, &device)?;
    let decoded = vae.decode(&x)?;
    let dec_dims = decoded.shape().dims();
    println!("  Decoded: {:?}", dec_dims);

    // Save PNG
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
    std::fs::create_dir_all(std::path::Path::new(OUTPUT_PATH).parent().unwrap_or(std::path::Path::new(".")))?;
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow::anyhow!("failed to create image"))?;
    img.save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("IMAGE SAVED: {OUTPUT_PATH}");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
