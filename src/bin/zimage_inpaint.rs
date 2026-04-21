//! Z-Image (NextDiT) inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `zimage_infer` for model load / VAE decode but adds:
//!   - input image + mask loading via `inference_flame::inpaint::prepare_inpaint`
//!   - per-step LanPaint Langevin inner loop wrapping the standard Z-Image
//!     denoise call (replaces the simple Euler step in `zimage_infer`)
//!   - final pixel-space blend that pastes the input image back where
//!     the mask says "preserve"
//!
//! Uses the FLUX-inpaint structure as a template, with these Z-Image
//! adjustments:
//!   - NextDiT.forward takes (x_NCHW, timestep[B], cap_feats) — no pack/unpack,
//!     no FLUX-style img_ids/txt_ids, no distilled guidance scalar
//!   - The model returns NEGATED velocity (see zimage_nextdit.rs final line:
//!     `x_spatial.mul_scalar(-1.0)`). So real velocity v_real = -model_out.
//!     For LanPaint x_0 prediction we need x_0 = x - t * v_real = x + t * model_out.
//!     For the Euler step we keep the same convention as `euler_step`:
//!     x_next = x + model_out * (sigma_next - sigma)
//!   - Schedule: linear t with optional flow-matching shift (build_sigma_schedule).
//!     `sigma == t` in flow-matching, so abt = (1-t)^2/((1-t)^2+t^2),
//!     ve_sigma = t/(1-t).
//!   - Text encoding is NOT done in this bin: like `zimage_infer`, we read
//!     pre-computed embeddings (`cap_feats`, optional `cap_feats_uncond`) from
//!     a safetensors file. Z-Image uses a Mistral-3 family encoder; running it
//!     in-process would require a separate plumbing pass.
//!
//! CLI:
//!   zimage_inpaint --prompt "a sleeping cat" \
//!                  --input-image input.png --mask mask.png \
//!                  --output-path inpaint_out.png \
//!                  --model /path/to/zimage.safetensors \
//!                  --vae   /path/to/vae.safetensors \
//!                  --embeddings /path/to/text_embeddings.safetensors \
//!                  [--turbo]
//!
//! The `--prompt` flag is informational only; encode it offline into the
//! embeddings file with the matching encoder before running this bin.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::serialization::{load_file, load_file_filtered};
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::zimage_nextdit::NextDiT;
use inference_flame::sampling::schedules::build_sigma_schedule;
use inference_flame::vae::{LdmVAEDecoder, LdmVAEEncoder};

// ---------------------------------------------------------------------------
// Defaults — match zimage_infer.rs conventions
// ---------------------------------------------------------------------------

const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/zimage_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";

// Z-Image VAE: same 16-channel LDM-style as FLUX, scale=0.3611, shift=0.1159.
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

const DEFAULT_SEED: u64 = 42;

// Base variant defaults: 28 steps, CFG ~4.0, shift=3.0.
const DEFAULT_STEPS_BASE: usize = 28;
const DEFAULT_CFG_BASE: f32 = 4.0;
// Turbo variant: 8 steps, no CFG.
const DEFAULT_STEPS_TURBO: usize = 8;
const DEFAULT_CFG_TURBO: f32 = 0.0;
const DEFAULT_SHIFT: f32 = 3.0;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    prompt: String,
    negative: String,
    input_image: PathBuf,
    mask: PathBuf,
    output_path: PathBuf,
    model_path: PathBuf,
    vae_path: PathBuf,
    embeddings_path: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    shift: f32,
    seed: u64,
    turbo: bool,
}

fn print_help() {
    println!(
        "zimage_inpaint — Z-Image NextDiT inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           zimage_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> \\\n  \
                          --model <PATH> --vae <PATH> --embeddings <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --prompt <TEXT>          Text prompt (informational; embeddings are pre-computed)\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n  \
           --model <PATH>           Z-Image NextDiT safetensors (file or directory)\n  \
           --vae <PATH>             VAE safetensors\n  \
           --embeddings <PATH>      Pre-computed text embeddings safetensors\n  \
                                    (must contain key 'cap_feats'; optional 'cap_feats_uncond')\n\
         \n\
         OPTIONS:\n  \
           --negative <TEXT>        Negative prompt (informational; embeddings file supplies it)\n  \
           --output-path <PATH>     Output PNG path [default: {default_output}]\n  \
           --width <N>              Output width [default: 1024]\n  \
           --height <N>             Output height [default: 1024]\n  \
           --steps <N>              Diffusion steps [default: {steps_base} base / {steps_turbo} turbo]\n  \
           --cfg <F>                Classifier-free guidance scale\n  \
                                    [default: {cfg_base} base / {cfg_turbo} turbo, requires cap_feats_uncond]\n  \
           --shift <F>              Sigma schedule shift [default: {shift}]\n  \
           --seed <N>               RNG seed [default: {seed}]\n  \
           --turbo                  Use Z-Image Turbo defaults (8 steps, no CFG)\n  \
           -h, --help               Print this help",
        default_output = DEFAULT_OUTPUT,
        steps_base = DEFAULT_STEPS_BASE,
        steps_turbo = DEFAULT_STEPS_TURBO,
        cfg_base = DEFAULT_CFG_BASE,
        cfg_turbo = DEFAULT_CFG_TURBO,
        shift = DEFAULT_SHIFT,
        seed = DEFAULT_SEED,
    );
}

fn parse_cli() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().collect();

    let mut prompt: Option<String> = None;
    let mut negative: String = String::new();
    let mut input_image: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut output_path: PathBuf = PathBuf::from(DEFAULT_OUTPUT);
    let mut model_path: Option<PathBuf> = None;
    let mut vae_path: Option<PathBuf> = None;
    let mut embeddings_path: Option<PathBuf> = None;
    let mut width: usize = 1024;
    let mut height: usize = 1024;
    let mut steps: Option<usize> = None;
    let mut cfg: Option<f32> = None;
    let mut shift: f32 = DEFAULT_SHIFT;
    let mut seed: u64 = DEFAULT_SEED;
    let mut turbo = false;

    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--prompt" => {
                prompt = Some(next_arg(&raw, &mut i, "--prompt")?);
            }
            "--negative" => {
                negative = next_arg(&raw, &mut i, "--negative")?;
            }
            "--input-image" => {
                input_image = Some(PathBuf::from(next_arg(&raw, &mut i, "--input-image")?));
            }
            "--mask" => {
                mask = Some(PathBuf::from(next_arg(&raw, &mut i, "--mask")?));
            }
            "--output-path" | "--output" => {
                output_path = PathBuf::from(next_arg(&raw, &mut i, "--output-path")?);
            }
            "--model" => {
                model_path = Some(PathBuf::from(next_arg(&raw, &mut i, "--model")?));
            }
            "--vae" => {
                vae_path = Some(PathBuf::from(next_arg(&raw, &mut i, "--vae")?));
            }
            "--embeddings" => {
                embeddings_path =
                    Some(PathBuf::from(next_arg(&raw, &mut i, "--embeddings")?));
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
                steps = Some(
                    next_arg(&raw, &mut i, "--steps")?
                        .parse()
                        .map_err(|e| anyhow!("--steps: {e}"))?,
                );
            }
            "--cfg" => {
                cfg = Some(
                    next_arg(&raw, &mut i, "--cfg")?
                        .parse()
                        .map_err(|e| anyhow!("--cfg: {e}"))?,
                );
            }
            "--shift" => {
                shift = next_arg(&raw, &mut i, "--shift")?
                    .parse()
                    .map_err(|e| anyhow!("--shift: {e}"))?;
            }
            "--seed" => {
                seed = next_arg(&raw, &mut i, "--seed")?
                    .parse()
                    .map_err(|e| anyhow!("--seed: {e}"))?;
            }
            "--turbo" => {
                turbo = true;
            }
            other => {
                return Err(anyhow!(
                    "unknown argument: {other}. Use --help for usage."
                ));
            }
        }
        i += 1;
    }

    let prompt = prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let input_image =
        input_image.ok_or_else(|| anyhow!("missing required --input-image. Use --help."))?;
    let mask = mask.ok_or_else(|| anyhow!("missing required --mask. Use --help."))?;
    let model_path =
        model_path.ok_or_else(|| anyhow!("missing required --model. Use --help."))?;
    let vae_path = vae_path.ok_or_else(|| anyhow!("missing required --vae. Use --help."))?;
    let embeddings_path = embeddings_path
        .ok_or_else(|| anyhow!("missing required --embeddings. Use --help."))?;

    let steps = steps.unwrap_or(if turbo { DEFAULT_STEPS_TURBO } else { DEFAULT_STEPS_BASE });
    let cfg = cfg.unwrap_or(if turbo { DEFAULT_CFG_TURBO } else { DEFAULT_CFG_BASE });

    Ok(CliArgs {
        prompt,
        negative,
        input_image,
        mask,
        output_path,
        model_path,
        vae_path,
        embeddings_path,
        width,
        height,
        steps,
        cfg,
        shift,
        seed,
        turbo,
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

    println!("=== Z-Image NextDiT — pure Rust INPAINT (LanPaint) ===");
    println!(
        "Variant:      {}",
        if args.turbo { "Turbo" } else { "Base" }
    );
    println!("Prompt:       {:?}", args.prompt);
    if !args.negative.is_empty() {
        println!("Negative:     {:?}", args.negative);
    }
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
    println!("Model:        {}", args.model_path.display());
    println!("VAE:          {}", args.vae_path.display());
    println!("Embeddings:   {}", args.embeddings_path.display());
    println!(
        "Size: {}x{}, steps={}, cfg={}, shift={}",
        args.width, args.height, args.steps, args.cfg, args.shift
    );
    println!("Seed: {}", args.seed);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: Load pre-computed text embeddings
    // ------------------------------------------------------------------
    println!("--- Stage 1: Load text embeddings ---");
    let t0 = Instant::now();

    let emb_tensors = load_file(&args.embeddings_path, &device)
        .map_err(|e| anyhow!("load embeddings: {e:?}"))?;
    let cap_feats = emb_tensors
        .get("cap_feats")
        .ok_or_else(|| anyhow!("embeddings file must contain 'cap_feats' key"))?
        .to_dtype(DType::BF16)
        .map_err(|e| anyhow!("cap_feats -> bf16: {e:?}"))?;
    println!("  cap_feats shape: {:?}", cap_feats.shape().dims());

    // Optional unconditional embeddings for CFG. If absent, we silently
    // disable CFG even when --cfg > 0 was passed.
    let cap_feats_uncond: Option<Tensor> = match emb_tensors.get("cap_feats_uncond") {
        Some(t) => {
            let bf16 = t
                .to_dtype(DType::BF16)
                .map_err(|e| anyhow!("cap_feats_uncond -> bf16: {e:?}"))?;
            println!("  cap_feats_uncond shape: {:?}", bf16.shape().dims());
            Some(bf16)
        }
        None => {
            if args.cfg > 1.0 {
                println!(
                    "  [warn] --cfg={} requested but embeddings file lacks 'cap_feats_uncond'; \
                     CFG will be disabled.",
                    args.cfg
                );
            }
            None
        }
    };
    let cfg_active = args.cfg > 1.0 && cap_feats_uncond.is_some();
    println!(
        "  CFG: {} (effective scale={})",
        if cfg_active { "enabled" } else { "disabled" },
        if cfg_active { args.cfg } else { 0.0 }
    );
    println!("  Embeddings loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode the input image (BEFORE DiT load to free memory)
    //
    // Z-Image VAE downscale is 8 (latent NCHW = [1, 16, H/8, W/8]).
    // Unlike FLUX, no pack/unpack — NextDiT operates on NCHW latents
    // directly (it patchifies internally with patch_size=2).
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();

    let inputs = {
        let vae_enc =
            LdmVAEEncoder::from_safetensors(args.vae_path.to_str().unwrap(), AE_IN_CHANNELS, &device)
                .map_err(|e| anyhow!("VAE encoder load: {e:?}"))?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg_inp = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // Plain LDM-style latent downscale; NextDiT patchify is internal.
            vae_scale: 8,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg_inp, device.clone(), |img| {
            vae_enc
                .encode_scaled(img, AE_SCALE_FACTOR, AE_SHIFT_FACTOR)
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
    // Stage 3: Build noise + load NextDiT
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load NextDiT (resident) ---");
    let t_load = Instant::now();
    let model_p = std::path::Path::new(&args.model_path);
    let all_weights = if model_p.is_dir() {
        let mut weights = std::collections::HashMap::new();
        let mut entries: Vec<_> = std::fs::read_dir(model_p)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in &entries {
            let partial = load_file(entry.path(), &device)
                .map_err(|e| anyhow!("load shard {}: {e:?}", entry.path().display()))?;
            weights.extend(partial);
        }
        println!("  Loaded {} shards from directory", entries.len());
        weights
    } else {
        load_file_filtered(args.model_path.to_str().unwrap(), &device, |_| true)
            .map_err(|e| anyhow!("load model: {e:?}"))?
    };
    println!(
        "  Loaded {} tensors in {:.1}s",
        all_weights.len(),
        t_load.elapsed().as_secs_f32()
    );
    let dit = NextDiT::new_resident(all_weights, device.clone());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Denoise (LanPaint inner + Z-Image Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, LanPaint inner) ---",
        args.steps
    );

    // Z-Image latent NCHW: [1, 16, H/8, W/8]
    let latent_h = args.height / 8;
    let latent_w = args.width / 8;
    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        AE_IN_CHANNELS, latent_h, latent_w
    );

    // Sanity-check VAE encoded shape matches expectation.
    let li_dims = inputs.latent_image.shape().dims();
    if li_dims[2] != latent_h || li_dims[3] != latent_w {
        return Err(anyhow!(
            "latent shape mismatch: VAE gave {:?}, expected [_, _, {}, {}]",
            li_dims,
            latent_h,
            latent_w
        ));
    }

    // Seeded Gaussian noise via Box-Muller (deterministic).
    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    // Initial x at t=1: pure noise. LanPaint::run replaces the known region
    // with the noise-scaled latent_image internally.
    let mut x_nchw = noise_nchw.clone_result()?;

    // Schedule: same as zimage_infer — linear in [1, 0] with optional shift.
    let sigmas = build_sigma_schedule(args.steps, args.shift);
    println!(
        "  Schedule: {} steps, shift={}, sigma[0]={:.4}, sigma[1]={:.4}, sigma[-1]={:.4}",
        args.steps,
        args.shift,
        sigmas[0],
        sigmas[1],
        sigmas[args.steps]
    );

    // ------------------------------------------------------------------
    // LanPaint config — same defaults as flux_inpaint, flow-matching mode.
    // ------------------------------------------------------------------
    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: true,
    };

    // NextDiT.forward takes &mut self, so wrap in RefCell to allow `Fn` closures.
    let dit_cell = RefCell::new(dit);
    let cap_feats_ref = &cap_feats;
    let cap_feats_uncond_ref = cap_feats_uncond.as_ref();
    let cfg_scale = args.cfg;
    let cfg_active_ref = cfg_active;

    // One Z-Image forward → returns model output (which is -velocity).
    // Applies CFG by composing cond+uncond model outputs the same way
    // `euler_step` does.
    let zimage_model_out = |x: &Tensor, sigma_vec: &Tensor| -> Result<Tensor> {
        let mut dit_borrow = dit_cell.borrow_mut();
        let pred_cond = dit_borrow.forward(x, sigma_vec, cap_feats_ref)?;
        let pred = if cfg_active_ref {
            // Safe to unwrap: cfg_active_ref true implies uncond present.
            let uncond = cap_feats_uncond_ref.unwrap();
            let pred_uncond = dit_borrow.forward(x, sigma_vec, uncond)?;
            // Mirror sampling/euler.rs:
            //   pred = pred_cond + cfg_scale * (pred_cond - pred_uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            let scaled = diff.mul_scalar(cfg_scale)?;
            pred_cond.add(&scaled)?
        } else {
            pred_cond
        };
        Ok(pred)
    };

    let t0 = Instant::now();
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];

        // -------- Build per-step time scalars for LanPaint --------
        // Flow-matching: sigma == t (in [0, 1], with sigma=1 ↔ noise).
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
            // At t=1 sigma -> ∞; clamp. LanPaint tolerates degenerate sigma.
            1.0e6
        };
        let sigma_scalar = make_b1_bf16(ve_sigma_val, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        let tflow_scalar = make_b1_bf16(flow_t, &device)?;
        let t_vec_step = make_b1_bf16(flow_t, &device)?;

        // ------------------------------------------------------------------
        // LanPaint inner Langevin loop.
        //
        // LanPaint expects an `inner_model` closure (x, t) → x_0 prediction.
        // For Z-Image (rectified flow), the model returns -velocity, so:
        //   v_real    = -model_out
        //   x_0       = x - t * v_real = x + t * model_out
        // We compute the math in F32 to dodge BF16 precision loss; LanPaint
        // upcasts internally either way.
        // ------------------------------------------------------------------
        let advanced_x = {
            let inner_model_fn =
                |x: &Tensor, t: &Tensor| -> flame_core::Result<Tensor> {
                    let model_out = zimage_model_out(x, t).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "zimage inner: {e:?}"
                        ))
                    })?;
                    let x_f32 = x.to_dtype(DType::F32)?;
                    let m_f32 = model_out.to_dtype(DType::F32)?;
                    let t_f32 = t.to_dtype(DType::F32)?;
                    let img_dim = x.shape().dims().len();
                    let b = t.shape().dims()[0];
                    let mut tdims = vec![b];
                    tdims.extend(std::iter::repeat(1).take(img_dim - 1));
                    let t_b = t_f32.reshape(&tdims)?;
                    // x_0 = x + t * model_out
                    let x0 = x_f32.add(&t_b.mul(&m_f32)?)?;
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
            // Drop LanPaint (and its captured closure) before reusing the
            // RefCell for the outer Euler step.
            drop(lanpaint);
            advanced_x
        };

        // -------- Standard Z-Image Euler step on the advanced x --------
        // Mirrors `euler_step`: x_next = x + model_out * (sigma_next - sigma).
        // (model_out is -velocity, dt is negative going noise→image, so this
        // is x_next = x - v_real * dt, the standard rectified-flow update.)
        let model_out = zimage_model_out(&advanced_x, &t_vec_step)?;
        let dt = sigma_next - sigma;
        let next_x = advanced_x.add(&model_out.mul_scalar(dt)?)?;

        // -------- Post-step mask blend (matches flux_inpaint convention) --------
        // noisy_image = latent_image * (1 - sigma_next) + noise * sigma_next
        // Where latent_mask == 1.0 (preserve), pull toward the correctly-noised
        // input latent so the next step still sees the correct prior; keep
        // model output where mask == 0.0 (inpaint region).
        let nf = sigma_next;
        let one_minus_nf = 1.0 - nf;
        let scaled_image = inputs.latent_image.mul_scalar(one_minus_nf)?;
        let scaled_noise = noise_nchw.mul_scalar(nf)?;
        let noisy_image = scaled_image.add(&scaled_noise)?;
        let blended = Tensor::where_mask(&inputs.latent_mask, &noisy_image, &next_x)?;

        x_nchw = blended;

        if step == 0 || step + 1 == args.steps || (step + 1) % 5 == 0 {
            println!(
                "  step {}/{}: sigma={:.4} -> {:.4} | dt-step done",
                step + 1,
                args.steps,
                sigma,
                sigma_next
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

    // Free DiT before VAE decoder load.
    drop(dit_cell);
    drop(cap_feats_uncond);
    println!("  DiT evicted");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();

    let vae = LdmVAEDecoder::from_safetensors(
        args.vae_path.to_str().unwrap(),
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &device,
    )
    .map_err(|e| anyhow!("VAE decoder load: {e:?}"))?;
    println!("  VAE decoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let rgb = vae.decode(&x_nchw).map_err(|e| anyhow!("vae decode: {e:?}"))?;
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
