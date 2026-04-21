//! ERNIE-Image 8B inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `ernie_image_infer` for prompt encoding and VAE decode but adds:
//!   - input image + mask loading via `inference_flame::inpaint::prepare_inpaint`
//!   - per-step LanPaint Langevin inner loop wrapping the standard ERNIE
//!     denoise call (replaces the simple Euler step in `ernie_image_infer`)
//!   - final pixel-space blend that pastes the input image back where
//!     the mask says "preserve"
//!
//! Architecture notes:
//!   - 36-layer single-stream DiT, 4096 hidden, 32 heads, in/out=128 channels.
//!   - Latent grid: [1, 128, H/16, W/16] (Klein VAE outputs packed 128-ch
//!     latents at H/16). For 1024x1024, latent is [1,128,64,64].
//!   - Schedule: flow-matching, exponential time-shift with shift=3.0.
//!   - Velocity convention: `ernie_euler_step` does `x_next = x + dt*pred` with
//!     dt = sigma_next - sigma (negative). So pred IS velocity v, and
//!     x_0 = x - sigma * v (standard flow). Inner-model closure converts.
//!   - CFG: SEQUENTIAL (one pass at a time, free pool between cond and
//!     uncond) per ernie_image_infer; LanPaint also wraps this as two
//!     forward calls inside the closure.
//!   - Text encoder: Mistral-3 3B, 256 max tokens.
//!
//! CLI:
//!   ernie_inpaint --prompt "a sleeping cat" \
//!                 --input-image input.png --mask mask.png \
//!                 --output-path inpaint_out.png

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::ernie_image::{ErnieImageConfig, ErnieImageModel};
use inference_flame::models::mistral3b_encoder::Mistral3bEncoder;
use inference_flame::sampling::ernie_sampling::{ernie_schedule, sigma_to_timestep};
use inference_flame::vae::klein_vae::{KleinVaeDecoder, KleinVaeEncoder};

// ---------------------------------------------------------------------------
// Paths / knobs (mirror ernie_image_infer)
// ---------------------------------------------------------------------------

const TRANSFORMER_DIR: &str = "/home/alex/models/ERNIE-Image/transformer";
const TEXT_ENCODER: &str = "/home/alex/models/ERNIE-Image/text_encoder/model.safetensors";
const TOKENIZER: &str = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json";
const VAE_PATH: &str = "/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors";

const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/ernie_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";

const DEFAULT_WIDTH: usize = 1024;
const DEFAULT_HEIGHT: usize = 1024;
const DEFAULT_STEPS: usize = 50;
const DEFAULT_CFG: f32 = 4.0;
const DEFAULT_SEED: u64 = 42;

// Klein VAE produces packed 128-ch latents at H/16, W/16.
const LATENT_CHANNELS: usize = 128;
const VAE_SCALE: usize = 16;

// Mistral text encoder context length (matches ernie_image_infer).
const TEXT_LEN_MAX: usize = 256;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    prompt: String,
    input_image: PathBuf,
    mask: PathBuf,
    output_path: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
}

fn print_help() {
    println!(
        "ernie_inpaint — ERNIE-Image 8B inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           ernie_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --prompt <TEXT>          Text prompt\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n\
         \n\
         OPTIONS:\n  \
           --output-path <PATH>     Output PNG path [default: {}]\n  \
           --width <N>              Output width [default: {}]\n  \
           --height <N>             Output height [default: {}]\n  \
           --steps <N>              Diffusion steps [default: {}]\n  \
           --cfg <F>                Classifier-free guidance [default: {}]\n  \
           --seed <N>               RNG seed [default: {}]\n  \
           -h, --help               Print this help",
        DEFAULT_OUTPUT, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_SEED
    );
}

fn parse_cli() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().collect();

    let mut prompt: Option<String> = None;
    let mut input_image: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut output_path: PathBuf = PathBuf::from(DEFAULT_OUTPUT);
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
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();

    let latent_h = args.height / VAE_SCALE;
    let latent_w = args.width / VAE_SCALE;

    println!("=== ERNIE-Image 8B — pure Rust INPAINT (LanPaint) ===");
    println!("Prompt:       {:?}", args.prompt);
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
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
    // Stage 1: Mistral-3 3B text encode (then drop encoder ~7GB freed)
    // ------------------------------------------------------------------
    println!("--- Stage 1: Mistral-3 3B text encode ---");
    let t0 = Instant::now();

    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER)
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let text_len = token_ids.len();
    println!("  tokenized: {text_len} tokens");

    // Diffusers ErnieImagePipeline uses encoded empty string for uncond,
    // not zeros (pipeline_ernie_image.py:280-298).
    let empty_encoding = tokenizer
        .encode("", true)
        .map_err(|e| anyhow!("tokenize empty: {e}"))?;
    let empty_ids: Vec<i32> = empty_encoding.get_ids().iter().map(|&id| id as i32).collect();
    let empty_len = empty_ids.len();
    println!("  uncond tokenized: {empty_len} tokens");

    let (text_embeds, uncond_embeds_real) = {
        let encoder = Mistral3bEncoder::load(TEXT_ENCODER, &device)?;
        println!("  encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let embeds = encoder.encode(&token_ids, TEXT_LEN_MAX)?;
        println!("  encoded: shape={:?}", embeds.shape().dims());
        let uncond = encoder.encode(&empty_ids, TEXT_LEN_MAX)?;
        println!("  uncond encoded: shape={:?}", uncond.shape().dims());
        (embeds, uncond)
        // encoder dropped here
    };
    println!("  text encoding done in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode the input image (BEFORE DiT load to free memory).
    //
    // Klein VAE outputs packed [1, 128, H/16, W/16] latents.
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load Klein VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();

    let inputs = {
        let raw = flame_core::serialization::load_file(std::path::Path::new(VAE_PATH), &device)?;
        let vae_device = flame_core::device::Device::from_arc(device.clone());
        let vae_enc = KleinVaeEncoder::load(&raw, &vae_device)?;
        drop(raw);
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg_inp = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // Klein VAE pixel→latent downscale = 16 (8× VAE + 2× patchify).
            vae_scale: VAE_SCALE,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg_inp, device.clone(), |img| {
            // KleinVaeEncoder.encode applies BatchNorm internally — that's the
            // training-cache path matching what the DiT was trained against.
            vae_enc
                .encode(img)
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
    // Stage 3: Load ERNIE-Image DiT (resident on GPU)
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load ERNIE-Image DiT (all blocks resident) ---");
    let t0 = Instant::now();

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
    println!(
        "  DiT loaded in {:.1}s ({} blocks on GPU)",
        t0.elapsed().as_secs_f32(),
        config.num_layers
    );
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Denoise (LanPaint inner + flow-matching Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, LanPaint inner) ---",
        args.steps
    );

    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        LATENT_CHANNELS, latent_h, latent_w
    );

    // Sanity-check VAE encoded shape matches expectation.
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

    // Initial x at t=1: pure noise. LanPaint will replace the known region
    // with the noise-scaled latent_image inside `run()` automatically.
    let mut x_nchw = noise_nchw.clone_result()?;

    // Schedule: ERNIE flow-matching with shift=3.0, exponential time shift.
    let sigmas = ernie_schedule(args.steps);
    println!(
        "  Schedule: {} steps, sigma[0]={:.4}, sigma[1]={:.4}, sigma[-1]={:.4}",
        args.steps,
        sigmas[0],
        sigmas[1],
        sigmas[args.steps]
    );

    // Trim text embeddings to real length — ERNIE has no attention mask,
    // padded tokens steal softmax weight (per ernie_image_infer rationale).
    let text_3d = if text_embeds.rank() == 2 {
        text_embeds.unsqueeze(0)?
    } else {
        text_embeds.clone()
    };
    let uncond_3d = if uncond_embeds_real.rank() == 2 {
        uncond_embeds_real.unsqueeze(0)?
    } else {
        uncond_embeds_real.clone()
    };
    let cond_real_len = text_len.min(TEXT_LEN_MAX);
    let uncond_real_len = empty_len.min(TEXT_LEN_MAX);
    let cond_trim = text_3d.narrow(1, 0, cond_real_len)?;
    let uncond_trim = uncond_3d.narrow(1, 0, uncond_real_len)?;
    let cond_lens = vec![cond_real_len];
    let uncond_lens = vec![uncond_real_len];

    // ------------------------------------------------------------------
    // LanPaint config — flow-matching mode (ERNIE is flow).
    // ------------------------------------------------------------------
    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: true, // ERNIE is flow-matching
    };

    // Wrap model in RefCell — `forward` takes &mut self (ErnieImageModel
    // mutates internal state during BlockOffloader iteration).
    let model_cell = RefCell::new(model);
    let cond_trim_ref = &cond_trim;
    let uncond_trim_ref = &uncond_trim;
    let cond_lens_ref = &cond_lens;
    let uncond_lens_ref = &uncond_lens;
    let cfg_scale = args.cfg;

    // One ERNIE forward with sequential CFG (matches ernie_image_infer).
    // Returns velocity prediction in NCHW form.
    //
    // Note: although ErnieImageModel::forward takes &mut self, RefMut auto-derefs
    // through method dispatch — `mut` keyword on the `let` binding is unused
    // by rustc's borrow checker.
    let ernie_velocity = |x: &Tensor, t_tensor: &Tensor| -> Result<Tensor> {
        let model_borrow = model_cell.borrow_mut();
        let pred = if cfg_scale > 1.0 {
            // Sequential CFG: cond pass → trim pool → uncond pass → trim pool.
            let pred_cond = model_borrow.forward(x, t_tensor, cond_trim_ref, cond_lens_ref)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            let pred_uncond =
                model_borrow.forward(x, t_tensor, uncond_trim_ref, uncond_lens_ref)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(cfg_scale)?)?
        } else {
            let p = model_borrow.forward(x, t_tensor, cond_trim_ref, cond_lens_ref)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            p
        };
        Ok(pred)
    };

    let t0 = Instant::now();
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];

        // -------- Build per-step time scalars for LanPaint --------
        // Flow-matching: sigma == t_flow.
        //   abt = (1 - t)^2 / ((1 - t)^2 + t^2)
        //   ve_sigma = t / (1 - t)
        let flow_t = sigma;
        let abt_val = {
            let one_minus = 1.0 - flow_t;
            let denom = one_minus * one_minus + flow_t * flow_t;
            if denom > 0.0 {
                (one_minus * one_minus) / denom
            } else {
                1.0
            }
        };
        let ve_sigma_val = if (1.0 - flow_t).abs() > 1e-6 {
            flow_t / (1.0 - flow_t)
        } else {
            1.0e6
        };
        let sigma_scalar = make_b1_bf16(ve_sigma_val, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        let tflow_scalar = make_b1_bf16(flow_t, &device)?;

        // Discrete model timestep used by ERNIE: sigma * 1000.
        let t_model_val = sigma_to_timestep(sigma);
        // ErnieImageModel.forward expects timestep as raw F32 (it casts to BF16
        // internally during sinusoidal embed). Match ernie_image_infer.
        let t_tensor = Tensor::from_vec(
            vec![t_model_val],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        // -------- LanPaint inner Langevin loop --------
        // For flow-matching: x_0 = x - t_flow * v(x, t_flow).
        // We use the captured t_tensor for the model call (LanPaint's `t`
        // parameter passed into the closure equals tflow which equals sigma
        // here, so semantically identical — but we pre-build the
        // t_model_val tensor anyway so we don't reallocate per call).
        let advanced_x = {
            let inner_model_fn =
                |x: &Tensor, _t: &Tensor| -> flame_core::Result<Tensor> {
                    let v = ernie_velocity(x, &t_tensor).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "ernie inner velocity: {e:?}"
                        ))
                    })?;
                    // x_0 = x - t_flow * v. Use the per-step `flow_t` scalar.
                    let x_f32 = x.to_dtype(DType::F32)?;
                    let v_f32 = v.to_dtype(DType::F32)?;
                    let x0 = x_f32.sub(&v_f32.mul_scalar(flow_t)?)?;
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

        // -------- Standard ERNIE Euler step on the advanced x --------
        // ernie_euler_step: x_next = x + dt * pred where dt = sigma_next - sigma.
        let v = ernie_velocity(&advanced_x, &t_tensor)?;
        let dt = sigma_next - sigma;
        let next_x = advanced_x.add(&v.mul_scalar(dt)?)?;

        // -------- Post-step mask blend (matches flux_inpaint) --------
        // noisy_image = latent_image * (1 - sigma_next) + noise * sigma_next
        // Where latent_mask == 1.0 (preserve), pull toward correctly-noised
        // input latent so next step still sees the correct prior.
        let nf = sigma_next;
        let one_minus_nf = 1.0 - nf;
        let scaled_image = inputs.latent_image.mul_scalar(one_minus_nf)?;
        let scaled_noise = noise_nchw.mul_scalar(nf)?;
        let noisy_image = scaled_image.add(&scaled_noise)?;
        let blended = Tensor::where_mask(&inputs.latent_mask, &noisy_image, &next_x)?;

        x_nchw = blended;

        if step == 0 || step + 1 == args.steps || (step + 1) % 5 == 0 {
            println!(
                "  step {}/{}: sigma={:.4} -> {:.4} t={:.1} ({:.1}s)",
                step + 1,
                args.steps,
                sigma,
                sigma_next,
                t_model_val,
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

    // Free DiT before VAE decoder load — ERNIE DiT is ~15 GB.
    drop(model_cell);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  DiT evicted");

    // ------------------------------------------------------------------
    // Stage 5: Klein VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();

    let vae = {
        let raw = flame_core::serialization::load_file(std::path::Path::new(VAE_PATH), &device)?;
        let vae_device = flame_core::device::Device::from_arc(device.clone());
        let dec = KleinVaeDecoder::load(&raw, &vae_device)?;
        drop(raw);
        dec
    };
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
