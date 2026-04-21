//! SDXL inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `sdxl_infer` for prompt embedding load + UNet forward + VAE
//! decode but adds:
//!   - input image + mask loading via `inference_flame::inpaint::prepare_inpaint`
//!   - per-step LanPaint Langevin inner loop wrapping the standard SDXL
//!     denoise call (replaces the simple Euler step in `sdxl_infer`)
//!   - final pixel-space blend that pastes the input image back where
//!     the mask says "preserve"
//!
//! SDXL is a VARIANCE-EXPLODING ε-prediction model (NOT flow-matching), so
//! `is_flow: false` activates LanPaint's VE branch:
//!   - noise_scaling: x_t = x_0 + σ·noise              (vs flow's σ·noise + (1-σ)·x_0)
//!   - x_t conversion: x_t = x · 1/√(1 + σ²)            (vs flow's x · (√ᾱ + √(1-ᾱ)))
//!   - inner_model receives σ (not tflow) as the "t" input
//!
//! In the VE branch, LanPaint's `score_model` reverses the conversion:
//!   x = x_t · √(1 + σ²)
//! and calls inner_model(x, σ). That `x` is in the variance-exploded space
//! `x_0 + σ·noise`. SDXL's UNet expects `x_unet_in = x / √(σ² + 1)`, so the
//! closure rescales internally.
//!
//! Eps-to-x_0 conversion in VE space:
//!   x_t (VE) = x_0 + σ·noise   ⇒   x_0 = x_t - σ·eps
//!
//! CFG: standard real two-pass: pred = uncond + cfg * (cond - uncond).
//! Text encoders: CLIP-L + CLIP-G (loaded as cached embeddings, like sdxl_infer).
//!
//! CLI:
//!   sdxl_inpaint --embeddings <path> \
//!                --input-image input.png --mask mask.png \
//!                --output-path inpaint_out.png

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::sdxl_unet::SDXLUNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;
use inference_flame::vae::ldm_encoder::LdmVAEEncoder;

// ---------------------------------------------------------------------------
// Paths / knobs (mirror sdxl_infer)
// ---------------------------------------------------------------------------

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors";
const DEFAULT_EMB_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors";

const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/sdxl_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";

const DEFAULT_WIDTH: usize = 1024;
const DEFAULT_HEIGHT: usize = 1024;
const DEFAULT_STEPS: usize = 30;
const DEFAULT_CFG: f32 = 7.5;
const DEFAULT_SEED: u64 = 42;

// SDXL VAE: 4-channel legacy SD VAE, 8x downscale.
const LATENT_CHANNELS: usize = 4;
const VAE_SCALE: usize = 8;
const VAE_SCALE_FACTOR: f32 = 0.13025;
const VAE_SHIFT_FACTOR: f32 = 0.0;

// ---------------------------------------------------------------------------
// Schedule (same as sdxl_infer.rs)
// ---------------------------------------------------------------------------

fn build_sdxl_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

    // Scaled-linear betas (SDXL default).
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

    // Leading spacing with steps_offset=1.
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
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    prompt: String,
    input_image: PathBuf,
    mask: PathBuf,
    output_path: PathBuf,
    embeddings_path: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
}

fn print_help() {
    println!(
        "sdxl_inpaint — SDXL inpainting via LanPaint (variance-exploding eps)\n\
         \n\
         USAGE:\n  \
           sdxl_inpaint --input-image <PATH> --mask <PATH> --embeddings <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n  \
           --embeddings <PATH>      Pre-computed CLIP-L + CLIP-G embeddings safetensors\n  \
                                    (must contain 'context','y','context_uncond','y_uncond')\n\
         \n\
         OPTIONS:\n  \
           --prompt <TEXT>          Informational only; embeddings are pre-computed [default: {prompt}]\n  \
           --output-path <PATH>     Output PNG path [default: {output}]\n  \
           --width <N>              Output width [default: {w}]\n  \
           --height <N>             Output height [default: {h}]\n  \
           --steps <N>              Diffusion steps [default: {steps}]\n  \
           --cfg <F>                Classifier-free guidance [default: {cfg}]\n  \
           --seed <N>               RNG seed [default: {seed}]\n  \
           -h, --help               Print this help",
        prompt = DEFAULT_PROMPT,
        output = DEFAULT_OUTPUT,
        w = DEFAULT_WIDTH,
        h = DEFAULT_HEIGHT,
        steps = DEFAULT_STEPS,
        cfg = DEFAULT_CFG,
        seed = DEFAULT_SEED,
    );
}

fn parse_cli() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().collect();

    let mut prompt: Option<String> = None;
    let mut input_image: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut output_path: PathBuf = PathBuf::from(DEFAULT_OUTPUT);
    let mut embeddings_path: PathBuf = PathBuf::from(DEFAULT_EMB_PATH);
    let mut width: usize = DEFAULT_WIDTH;
    let mut height: usize = DEFAULT_HEIGHT;
    let mut steps: usize = DEFAULT_STEPS;
    let mut cfg: f32 = DEFAULT_CFG;
    let mut seed: u64 = DEFAULT_SEED;

    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--prompt" => prompt = Some(next_arg(&raw, &mut i, "--prompt")?),
            "--input-image" => {
                input_image = Some(PathBuf::from(next_arg(&raw, &mut i, "--input-image")?));
            }
            "--mask" => mask = Some(PathBuf::from(next_arg(&raw, &mut i, "--mask")?)),
            "--output-path" | "--output" => {
                output_path = PathBuf::from(next_arg(&raw, &mut i, "--output-path")?);
            }
            "--embeddings" => {
                embeddings_path = PathBuf::from(next_arg(&raw, &mut i, "--embeddings")?);
            }
            "--width" => {
                width = next_arg(&raw, &mut i, "--width")?
                    .parse()
                    .map_err(|e| anyhow!("--width: {e}"))?;
            }
            "--height" => {
                height = next_arg(&raw, &mut i, "--height")?
                    .parse()
                    .map_err(|e| anyhow!("--height: {e}"))?;
            }
            "--steps" => {
                steps = next_arg(&raw, &mut i, "--steps")?
                    .parse()
                    .map_err(|e| anyhow!("--steps: {e}"))?;
            }
            "--cfg" => {
                cfg = next_arg(&raw, &mut i, "--cfg")?
                    .parse()
                    .map_err(|e| anyhow!("--cfg: {e}"))?;
            }
            "--seed" => {
                seed = next_arg(&raw, &mut i, "--seed")?
                    .parse()
                    .map_err(|e| anyhow!("--seed: {e}"))?;
            }
            other => {
                return Err(anyhow!("unknown argument: {other}. Use --help for usage."));
            }
        }
        i += 1;
    }

    let prompt = prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let input_image =
        input_image.ok_or_else(|| anyhow!("missing required --input-image. Use --help."))?;
    let mask = mask.ok_or_else(|| anyhow!("missing required --mask. Use --help."))?;

    Ok(CliArgs {
        prompt,
        input_image,
        mask,
        output_path,
        embeddings_path,
        width,
        height,
        steps,
        cfg,
        seed,
    })
}

fn next_arg(raw: &[String], i: &mut usize, flag: &str) -> Result<String> {
    *i += 1;
    raw.get(*i)
        .cloned()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();

    let args = parse_cli()?;
    let t_total = Instant::now();
    let device = global_cuda_device();

    let latent_h = args.height / VAE_SCALE;
    let latent_w = args.width / VAE_SCALE;

    println!("=== SDXL — pure Rust INPAINT (LanPaint, VE eps) ===");
    println!("Prompt:       {:?} (informational)", args.prompt);
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
    println!("Embeddings:   {}", args.embeddings_path.display());
    println!(
        "Size: {}x{}, latent: {}x{} ({} ch)",
        args.width, args.height, latent_w, latent_h, LATENT_CHANNELS
    );
    println!(
        "Steps: {}, CFG: {}, seed: {}",
        args.steps, args.cfg, args.seed
    );
    println!();

    // ------------------------------------------------------------------
    // Stage 1: Load cached CLIP-L + CLIP-G embeddings
    // ------------------------------------------------------------------
    println!("--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();

    let emb = flame_core::serialization::load_file(&args.embeddings_path, &device)
        .map_err(|e| anyhow!("load embeddings: {e:?}"))?;
    let context = emb
        .get("context")
        .ok_or_else(|| anyhow!("missing 'context' in embeddings"))?
        .clone();
    let y = emb
        .get("y")
        .ok_or_else(|| anyhow!("missing 'y' in embeddings"))?
        .clone();
    let context_uncond = emb
        .get("context_uncond")
        .ok_or_else(|| anyhow!("missing 'context_uncond' in embeddings"))?
        .clone();
    let y_uncond = emb
        .get("y_uncond")
        .ok_or_else(|| anyhow!("missing 'y_uncond' in embeddings"))?
        .clone();
    drop(emb);
    println!(
        "  context: {:?}, y: {:?}",
        context.dims(),
        y.dims()
    );
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode the input image (BEFORE UNet load to free memory)
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();

    let inputs = {
        let vae_enc = LdmVAEEncoder::from_safetensors(VAE_PATH, LATENT_CHANNELS, &device)?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg_inp = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // SDXL VAE pixel→latent downscale = 8 (4-channel legacy SD VAE).
            vae_scale: VAE_SCALE,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg_inp, device.clone(), |img| {
            // SDXL encode: z = (raw_z - shift) * scale  (shift = 0, scale = 0.13025)
            vae_enc
                .encode_scaled(img, VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR)
                .map_err(|e| anyhow!("vae encode: {e:?}"))
        })?
    };
    println!("  latent_image: {:?}", inputs.latent_image.shape().dims());
    println!("  latent_mask:  {:?}", inputs.latent_mask.shape().dims());
    println!("  pixel_mask:   {:?}", inputs.pixel_mask.shape().dims());
    println!(
        "  Inpaint inputs prepared in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Load SDXL UNet
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load SDXL UNet ---");
    let t0 = Instant::now();
    let model = SDXLUNet::from_safetensors_all_gpu(MODEL_PATH, &device)?;
    println!("  UNet loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Denoise (LanPaint inner + Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, LanPaint VE inner, CFG={}) ---",
        args.steps, args.cfg
    );

    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        LATENT_CHANNELS, latent_h, latent_w
    );

    // Sanity-check VAE encoded shape.
    let li_dims = inputs.latent_image.shape().dims();
    if li_dims[1] != LATENT_CHANNELS || li_dims[2] != latent_h || li_dims[3] != latent_w {
        return Err(anyhow!(
            "latent shape mismatch: VAE gave {:?}, expected [_, {}, {}, {}]",
            li_dims,
            LATENT_CHANNELS,
            latent_h,
            latent_w
        ));
    }

    // Seeded Gaussian noise (Box-Muller, deterministic).
    let numel = LATENT_CHANNELS * latent_h * latent_w;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, LATENT_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    // Initial x at t=sigma_max in VE convention: x = x_0 + sigma_max * noise.
    // LanPaint::run will do this scaling on the known region; for the unknown
    // region we initialize with the same VE-noised expression but with x_0=0
    // (equivalent to sigma_max * noise scaled into LanPaint's internal x_t
    // frame). Easier: start with pure noise scaled by sigma_max, matching the
    // VE convention's "x at t=1 is sigma_max * noise".
    let (sigmas, timesteps) = build_sdxl_schedule(args.steps);
    let sigma_max = sigmas[0];
    let mut x_nchw = noise_nchw.mul_scalar(sigma_max)?;

    println!(
        "  Schedule: {} steps, sigma_max={:.4}, sigma_min={:.6}, t_max={:.0}, t_min={:.0}",
        args.steps,
        sigmas[0],
        sigmas[args.steps - 1],
        timesteps[0],
        timesteps[args.steps - 1]
    );

    // ------------------------------------------------------------------
    // LanPaint config — VE branch (SDXL is eps-prediction, NOT flow).
    // ------------------------------------------------------------------
    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: false, // SDXL is variance-exploding eps-prediction
    };

    // Wrap UNet in RefCell — `forward` takes &mut self.
    let model_cell = RefCell::new(model);
    let context_ref = &context;
    let context_uncond_ref = &context_uncond;
    let y_ref = &y;
    let y_uncond_ref = &y_uncond;
    let cfg_scale = args.cfg;

    // One SDXL UNet forward → eps prediction. Standard real two-pass CFG.
    //
    // `x_ve` here is in the VE space (x_0 + sigma * noise) — that's what
    // LanPaint hands us via score_model's `x = x_t * sqrt(1 + sigma^2)`
    // conversion (and what our outer Euler loop holds in `x_nchw`).
    //
    // SDXL UNet expects pre-conditioned input: x_in = x / sqrt(sigma^2 + 1).
    // Returns eps.
    let sdxl_eps = |x_ve: &Tensor, sigma: f32, t_val: f32| -> Result<Tensor> {
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_f32 = x_ve.to_dtype(DType::F32)?;
        let x_in = x_f32.mul_scalar(c_in)?.to_dtype(DType::BF16)?;

        let timestep = Tensor::from_f32_to_bf16(
            vec![t_val],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        let mut model_borrow = model_cell.borrow_mut();
        let pred_cond = model_borrow.forward(&x_in, &timestep, context_ref, y_ref)?;
        let pred_uncond =
            model_borrow.forward(&x_in, &timestep, context_uncond_ref, y_uncond_ref)?;
        drop(model_borrow);

        // CFG in F32 to dodge BF16 precision loss.
        let pred_cond_f32 = pred_cond.to_dtype(DType::F32)?;
        let pred_uncond_f32 = pred_uncond.to_dtype(DType::F32)?;
        let diff = pred_cond_f32.sub(&pred_uncond_f32)?;
        let pred_f32 = pred_uncond_f32.add(&diff.mul_scalar(cfg_scale)?)?;
        Ok(pred_f32)
    };

    let t0 = Instant::now();
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_val = timesteps[step];

        // -------- Build per-step time scalars for LanPaint (VE branch) --------
        // VE: pass sigma directly. abt = 1 / (1 + sigma^2). tflow ignored.
        let abt_val = 1.0 / (1.0 + sigma * sigma);
        let sigma_scalar = make_b1_bf16(sigma, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        // tflow ignored on VE path but the API still requires a tensor.
        let tflow_scalar = make_b1_bf16(0.0, &device)?;

        // -------- LanPaint inner Langevin loop (VE) --------
        // LanPaint's inner_model contract on VE path: takes (x_VE, sigma_vec)
        // where x_VE = x_t * sqrt(1 + sigma^2) (variance-exploded space).
        // Returns x_0 prediction.
        //
        // For SDXL eps-prediction:
        //   eps = unet(x_in = x_VE * c_in, t)
        //   x_0 = x_VE - sigma * eps
        let advanced_x = {
            let inner_model_fn =
                |x: &Tensor, _sigma_vec: &Tensor| -> flame_core::Result<Tensor> {
                    let eps_f32 = sdxl_eps(x, sigma, t_val).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "sdxl inner eps: {e:?}"
                        ))
                    })?;
                    let x_f32 = x.to_dtype(DType::F32)?;
                    // x_0 = x - sigma * eps  (VE space)
                    let x0 = x_f32.sub(&eps_f32.mul_scalar(sigma)?)?;
                    x0.to_dtype(x.dtype())
                };

            let lanpaint = LanPaint::new(
                LanPaintConfig {
                    n_steps: lanpaint_cfg.n_steps,
                    lambda_: lanpaint_cfg.lambda_,
                    friction: lanpaint_cfg.friction,
                    beta: lanpaint_cfg.beta,
                    step_size: lanpaint_cfg.step_size,
                    is_flow: lanpaint_cfg.is_flow,
                },
                Box::new(inner_model_fn),
            );

            let (_lanpaint_x0, advanced_x) = lanpaint_step(
                &lanpaint,
                &x_nchw,
                &inputs,
                &noise_nchw,
                &sigma_scalar,
                &abt_scalar,
                &tflow_scalar,
            )?;
            drop(lanpaint);
            advanced_x
        };

        // -------- Standard SDXL Euler step on the advanced x --------
        // sdxl_infer.rs computes pred_f32 = uncond + cfg*(cond-uncond), then
        // x_next = x + pred_f32 * dt with dt = sigma_next - sigma. The
        // `pred_f32` there IS the eps prediction itself (Euler eps step).
        //
        // Our `sdxl_eps` returns the same CFG-combined eps in F32, so:
        //   x_next = x + eps * (sigma_next - sigma)
        let eps_f32 = sdxl_eps(&advanced_x, sigma, t_val)?;
        let advanced_f32 = advanced_x.to_dtype(DType::F32)?;
        let dt = sigma_next - sigma;
        let next_x_f32 = advanced_f32.add(&eps_f32.mul_scalar(dt)?)?;
        let next_x = next_x_f32.to_dtype(DType::BF16)?;

        // -------- Post-step mask blend (VE convention) --------
        // VE noisy_image at sigma_next: x_0 + sigma_next * noise.
        // Where latent_mask == 1.0 (preserve), pull toward correctly-noised
        // input latent so next step still sees the correct prior.
        let scaled_noise = noise_nchw.mul_scalar(sigma_next)?;
        let noisy_image = inputs.latent_image.add(&scaled_noise)?;
        let blended = Tensor::where_mask(&inputs.latent_mask, &noisy_image, &next_x)?;

        x_nchw = blended;

        if step == 0 || step + 1 == args.steps || (step + 1) % 5 == 0 {
            println!(
                "  step {}/{}: sigma={:.4} -> {:.4} t={:.0} ({:.1}s)",
                step + 1,
                args.steps,
                sigma,
                sigma_next,
                t_val,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step)",
        dt,
        dt / args.steps as f32
    );
    println!();

    // Free UNet before VAE decoder load.
    drop(model_cell);
    println!("  UNet evicted");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();

    let vae = LdmVAEDecoder::from_safetensors(
        VAE_PATH,
        LATENT_CHANNELS,
        VAE_SCALE_FACTOR,
        VAE_SHIFT_FACTOR,
        &device,
    )?;
    println!("  VAE decoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let rgb = vae.decode(&x_nchw)?;
    drop(x_nchw);
    drop(vae);
    println!("  Decoded: {:?}", rgb.shape().dims());

    // Decoded shape is [1, 3, H, W]; the blend works on [3, H, W].
    let rgb_3chw = rgb.narrow(0, 0, 1)?.reshape(&[3, args.height, args.width])?;
    let rgb_blended = blend_output(&rgb_3chw, &inputs.input_image, &inputs.pixel_mask)?;
    println!("  Blended pixel output: {:?}", rgb_blended.shape().dims());
    println!();

    // ------------------------------------------------------------------
    // Stage 6: Save PNG
    // ------------------------------------------------------------------
    println!("--- Stage 6: Save PNG ---");
    save_chw_f32_to_png(&rgb_blended, args.height, args.width, &args.output_path)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", args.output_path.display());
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    let _ = device;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [1] BF16 GPU tensor from a single f32 value.
fn make_b1_bf16(v: f32, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::from_vec(vec![v], Shape::from_dims(&[1]), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

/// Box-Muller seeded normal noise.
fn box_muller_noise(seed: u64, numel: usize) -> Vec<f32> {
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
}

/// Save a [3, H, W] F32 tensor in [-1, 1] to a PNG.
fn save_chw_f32_to_png(rgb: &Tensor, h: usize, w: usize, path: &PathBuf) -> Result<()> {
    let rgb_f32 = if rgb.dtype() == DType::F32 {
        rgb.clone_result()?
    } else {
        rgb.to_dtype(DType::F32)?
    };
    let data = rgb_f32.to_vec_f32()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                let v = data[idx].clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("Failed to build RgbImage"))?;
    img.save(path)?;
    Ok(())
}
