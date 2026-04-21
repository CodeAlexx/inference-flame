//! Stable Diffusion 1.5 inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `sd15_infer` for prompt encoding + UNet forward + VAE decode but
//! adds:
//!   - input image + mask loading via `inference_flame::inpaint::prepare_inpaint`
//!   - per-step LanPaint Langevin inner loop wrapping the standard SD 1.5
//!     denoise call (replaces the simple Euler step in `sd15_infer`)
//!   - final pixel-space blend that pastes the input image back where
//!     the mask says "preserve"
//!
//! SD 1.5 is a VARIANCE-EXPLODING ε-prediction model (NOT flow-matching), so
//! `is_flow: false` activates LanPaint's VE branch. Same conventions as
//! sdxl_inpaint:
//!   - x_t = x_0 + σ·noise
//!   - inner_model receives σ as the "t" input
//!   - eps → x_0:  x_0 = x - σ·eps  (in VE space)
//!
//! Differences vs SDXL inpaint:
//!   - Single CLIP-L encoder (no CLIP-G), encoded inline (no cached embeddings)
//!   - 4-channel SD VAE shared with SDXL but uses scale=0.18215 (SD 1.5 value)
//!   - Default 512×512 (SD 1.5 native resolution), 30 steps, CFG 7.5
//!   - VAE safetensors uses pre-0.14 diffusers attention key naming
//!     (`query`/`key`/`value`/`proj_attn`); rename to `to_q`/`to_k`/`to_v`/`to_out.0`
//!     before LdmVAEDecoder/Encoder will load it.
//!
//! CLI:
//!   sd15_inpaint --prompt "a sleeping cat" \
//!                --input-image input.png --mask mask.png \
//!                --output-path inpaint_out.png

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::sd15_unet::SD15UNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;
use inference_flame::vae::ldm_encoder::LdmVAEEncoder;

// ---------------------------------------------------------------------------
// Paths / knobs (mirror sd15_infer)
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
const CLIP_L_TOKENIZER: &str =
    "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/sd15_inpaint.png";
const DEFAULT_PROMPT: &str =
    "a photorealistic portrait of an elderly fisherman mending a net, Vermeer lighting";

const DEFAULT_WIDTH: usize = 512;
const DEFAULT_HEIGHT: usize = 512;
const DEFAULT_STEPS: usize = 30;
const DEFAULT_CFG: f32 = 7.5;
const DEFAULT_SEED: u64 = 42;

const CLIP_SEQ_LEN: usize = 77;

// SD 1.5 VAE: 4-channel, scale=0.18215, 8x downscale.
const LATENT_CHANNELS: usize = 4;
const VAE_SCALE: usize = 8;
const VAE_SCALE_FACTOR: f32 = 0.18215;
const VAE_SHIFT_FACTOR: f32 = 0.0;

// ---------------------------------------------------------------------------
// Schedule (same as sd15_infer.rs)
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
            eprintln!("[sd15_inpaint] CLIP tokenizer failed: {e}; using BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

fn load_clip_weights(
    path: &str,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let raw = flame_core::serialization::load_file(std::path::Path::new(path), device)?;
    let mut weights = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let t = if v.dtype() == DType::BF16 {
            v
        } else {
            v.to_dtype(DType::BF16)?
        };
        weights.insert(k, t);
    }
    Ok(weights)
}

// ---------------------------------------------------------------------------
// SD 1.5 VAE remap (encoder + decoder).
//
// SD 1.5's VAE safetensors uses pre-0.14 diffusers attention key naming
// (`query`/`key`/`value`/`proj_attn`) inside `attentions.0.*`. The current
// LdmVAEDecoder/Encoder remap only knows about the legacy raw-LDM
// `encoder.mid.attn_1.*` form, so we rename the modern-but-pre-0.14 keys
// in-place, then write a temp safetensors and let `from_safetensors` finish
// the load. This mirrors the workaround in sd15_infer.rs.
// ---------------------------------------------------------------------------

fn remap_sd15_vae_to_modern(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        // Strip first_stage_model. prefix if present.
        let k = k.strip_prefix("first_stage_model.").unwrap_or(&k).to_string();
        // Pre-0.14 → modern attention key renames (decoder + encoder share the
        // same `attentions.0.*` substring).
        let k = k
            .replace("attentions.0.query.", "attentions.0.to_q.")
            .replace("attentions.0.key.", "attentions.0.to_k.")
            .replace("attentions.0.value.", "attentions.0.to_v.")
            .replace("attentions.0.proj_attn.", "attentions.0.to_out.0.");
        let v_bf16 = if v.dtype() == DType::BF16 {
            v
        } else {
            v.to_dtype(DType::BF16)?
        };
        out.insert(k, v_bf16);
    }
    Ok(out)
}

fn write_temp_vae(remapped: &HashMap<String, Tensor>, suffix: &str) -> Result<String> {
    let tmp = format!("/tmp/sd15_vae_{suffix}.safetensors");
    flame_core::serialization::save_file(remapped, std::path::Path::new(&tmp))?;
    Ok(tmp)
}

fn load_sd15_vae_encoder(device: &Arc<CudaDevice>) -> Result<LdmVAEEncoder> {
    // Filter to encoder + quant_conv keys, rename, save temp, load.
    let raw = flame_core::serialization::load_file(std::path::Path::new(&vae_path()), device)?;
    let mut filtered = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let is_encoder = k.starts_with("encoder.")
            || k.starts_with("first_stage_model.encoder.")
            || k == "quant_conv.weight"
            || k == "quant_conv.bias"
            || k == "first_stage_model.quant_conv.weight"
            || k == "first_stage_model.quant_conv.bias";
        if !is_encoder {
            continue;
        }
        filtered.insert(k, v);
    }
    let remapped = remap_sd15_vae_to_modern(filtered)?;
    let tmp = write_temp_vae(&remapped, "encoder")?;
    LdmVAEEncoder::from_safetensors(&tmp, LATENT_CHANNELS, device)
        .map_err(|e| anyhow!("LdmVAEEncoder load: {e:?}"))
}

fn load_sd15_vae_decoder(device: &Arc<CudaDevice>) -> Result<LdmVAEDecoder> {
    let raw = flame_core::serialization::load_file(std::path::Path::new(&vae_path()), device)?;
    let mut filtered = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let is_decoder = k.starts_with("decoder.")
            || k.starts_with("first_stage_model.decoder.")
            || k == "post_quant_conv.weight"
            || k == "post_quant_conv.bias"
            || k == "first_stage_model.post_quant_conv.weight"
            || k == "first_stage_model.post_quant_conv.bias";
        if !is_decoder {
            continue;
        }
        filtered.insert(k, v);
    }
    let remapped = remap_sd15_vae_to_modern(filtered)?;
    let tmp = write_temp_vae(&remapped, "decoder")?;
    LdmVAEDecoder::from_safetensors(
        &tmp,
        LATENT_CHANNELS,
        VAE_SCALE_FACTOR,
        VAE_SHIFT_FACTOR,
        device,
    )
    .map_err(|e| anyhow!("LdmVAEDecoder load: {e:?}"))
}

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
        "sd15_inpaint — SD 1.5 inpainting via LanPaint (variance-exploding eps)\n\
         \n\
         USAGE:\n  \
           sd15_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --prompt <TEXT>          Text prompt\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n\
         \n\
         OPTIONS:\n  \
           --output-path <PATH>     Output PNG path [default: {output}]\n  \
           --width <N>              Output width [default: {w}]\n  \
           --height <N>             Output height [default: {h}]\n  \
           --steps <N>              Diffusion steps [default: {steps}]\n  \
           --cfg <F>                Classifier-free guidance [default: {cfg}]\n  \
           --seed <N>               RNG seed [default: {seed}]\n  \
           -h, --help               Print this help",
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

    let latent_h = args.height / VAE_SCALE;
    let latent_w = args.width / VAE_SCALE;

    println!("=== SD 1.5 — pure Rust INPAINT (LanPaint, VE eps) ===");
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
    // Stage 1: CLIP-L text encoding (inline, drop after)
    // ------------------------------------------------------------------
    println!("--- Stage 1: CLIP-L encode ---");
    let t0 = Instant::now();
    let (context, context_uncond) = {
        let weights = load_clip_weights(&clip_l_path(), &device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, _pooled) = clip.encode(&tokenize_clip(&args.prompt))?;
        let (hu, _pooled_u) = clip.encode(&tokenize_clip(""))?;
        (hc, hu)
    };
    println!(
        "  cond: {:?}, uncond: {:?} ({:.1}s)",
        context.dims(),
        context_uncond.dims(),
        t0.elapsed().as_secs_f32()
    );
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode the input image (BEFORE UNet load)
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();

    let inputs = {
        let vae_enc = load_sd15_vae_encoder(&device)?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg_inp = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            vae_scale: VAE_SCALE,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg_inp, device.clone(), |img| {
            // SD 1.5 encode: z = raw_z * 0.18215 (no shift).
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
    // Stage 3: Load SD 1.5 UNet
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load SD 1.5 UNet ---");
    let t0 = Instant::now();
    let model = SD15UNet::from_safetensors_all_gpu(&unet_path(), &device)?;
    println!("  UNet loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Denoise (LanPaint VE inner + Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, LanPaint VE inner, CFG={}) ---",
        args.steps, args.cfg
    );

    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        LATENT_CHANNELS, latent_h, latent_w
    );

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

    let (sigmas, timesteps) = build_sd15_schedule(args.steps);
    let sigma_max = sigmas[0];
    // Initial x in VE space: sigma_max * noise (equivalent to x_0=0 fully noised).
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
    // LanPaint config — VE branch (SD 1.5 is eps-prediction, NOT flow).
    // ------------------------------------------------------------------
    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: false, // SD 1.5 is variance-exploding eps-prediction
    };

    let model_cell = RefCell::new(model);
    let context_ref = &context;
    let context_uncond_ref = &context_uncond;
    let cfg_scale = args.cfg;

    // One SD 1.5 UNet forward → eps prediction with standard CFG.
    //
    // `x_ve` is in VE space (x_0 + sigma * noise) — what LanPaint hands us
    // post score_model conversion AND what the outer Euler loop holds.
    // SD 1.5 UNet expects pre-conditioned input: x_in = x / sqrt(sigma^2 + 1).
    let sd15_eps = |x_ve: &Tensor, sigma: f32, t_val: f32| -> Result<Tensor> {
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let x_f32 = x_ve.to_dtype(DType::F32)?;
        let x_in = x_f32.mul_scalar(c_in)?.to_dtype(DType::BF16)?;

        let timestep = Tensor::from_f32_to_bf16(
            vec![t_val],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        let mut model_borrow = model_cell.borrow_mut();
        let pred_cond = model_borrow.forward(&x_in, &timestep, context_ref)?;
        let pred_uncond = model_borrow.forward(&x_in, &timestep, context_uncond_ref)?;
        drop(model_borrow);

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

        // VE branch scalars: sigma direct, abt = 1/(1+sigma^2), tflow ignored.
        let abt_val = 1.0 / (1.0 + sigma * sigma);
        let sigma_scalar = make_b1_bf16(sigma, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        let tflow_scalar = make_b1_bf16(0.0, &device)?;

        // -------- LanPaint inner Langevin loop (VE) --------
        // Inner contract: (x_VE, sigma) → x_0 prediction.
        // For SD 1.5 eps:  x_0 = x - sigma * eps  (in VE space)
        let advanced_x = {
            let inner_model_fn =
                |x: &Tensor, _sigma_vec: &Tensor| -> flame_core::Result<Tensor> {
                    let eps_f32 = sd15_eps(x, sigma, t_val).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "sd15 inner eps: {e:?}"
                        ))
                    })?;
                    let x_f32 = x.to_dtype(DType::F32)?;
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

        // -------- Standard SD 1.5 Euler step --------
        // Same as sd15_infer.rs: x_next = x + eps * (sigma_next - sigma).
        let eps_f32 = sd15_eps(&advanced_x, sigma, t_val)?;
        let advanced_f32 = advanced_x.to_dtype(DType::F32)?;
        let dt = sigma_next - sigma;
        let next_x_f32 = advanced_f32.add(&eps_f32.mul_scalar(dt)?)?;
        let next_x = next_x_f32.to_dtype(DType::BF16)?;

        // -------- Post-step mask blend (VE) --------
        // VE noisy_image at sigma_next: x_0 + sigma_next * noise.
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

    drop(model_cell);
    println!("  UNet evicted");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();
    let vae = load_sd15_vae_decoder(&device)?;
    println!("  VAE decoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let rgb = vae.decode(&x_nchw)?;
    drop(x_nchw);
    drop(vae);
    println!("  Decoded: {:?}", rgb.shape().dims());

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
