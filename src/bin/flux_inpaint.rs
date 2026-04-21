//! FLUX 1 Dev inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `flux1_infer` for prompt encoding and VAE decode but adds:
//!   - input image + mask loading via `inference_flame::inpaint::prepare_inpaint`
//!   - per-step LanPaint Langevin inner loop wrapping the standard FLUX
//!     denoise call (replaces the simple Euler step in `flux1_denoise`)
//!   - final pixel-space blend that pastes the input image back where
//!     the mask says "preserve"
//!
//! This is the reference inpaint binary; other model bins (SDXL, SD1.5,
//! Cascade, Klein, ...) follow the same structure with model-appropriate
//! `is_flow` / sigma-mapping choices.
//!
//! CLI:
//!   flux_inpaint --prompt "a sleeping cat" \
//!                --input-image input.png --mask mask.png \
//!                --output-path inpaint_out.png

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use cudarc::driver::CudaDevice;
use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{prepare_inpaint, blend_output, lanpaint_step, InpaintConfig};
use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::flux1_dit::Flux1DiT;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{get_schedule, pack_latent, unpack_latent};
use inference_flame::vae::{LdmVAEDecoder, LdmVAEEncoder};

// ---------------------------------------------------------------------------
// Paths / knobs (mirror flux1_infer)
// ---------------------------------------------------------------------------

const CLIP_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const DIT_PATH: &str = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/ae.safetensors";

const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/flux_inpaint.png";

const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";

const SEED: u64 = 42;
const NUM_STEPS: usize = 20;
const GUIDANCE: f32 = 3.5;

// FLUX 1 Dev ae_params (util.py: flux-dev)
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 512;

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
    seed: u64,
    guidance: f32,
}

fn print_help() {
    println!(
        "flux_inpaint — FLUX 1 Dev inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           flux_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --prompt <TEXT>          Text prompt\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n\
         \n\
         OPTIONS:\n  \
           --output-path <PATH>     Output PNG path [default: {}]\n  \
           --width <N>              Output width [default: 1024]\n  \
           --height <N>             Output height [default: 1024]\n  \
           --steps <N>              Diffusion steps [default: {}]\n  \
           --seed <N>               RNG seed [default: {}]\n  \
           --guidance <F>           CFG guidance [default: {}]\n  \
           -h, --help               Print this help",
        DEFAULT_OUTPUT, NUM_STEPS, SEED, GUIDANCE
    );
}

fn parse_cli() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().collect();

    let mut prompt: Option<String> = None;
    let mut input_image: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut output_path: PathBuf = PathBuf::from(DEFAULT_OUTPUT);
    let mut width: usize = 1024;
    let mut height: usize = 1024;
    let mut steps: usize = NUM_STEPS;
    let mut seed: u64 = SEED;
    let mut guidance: f32 = GUIDANCE;

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
            "--input-image" => {
                input_image = Some(PathBuf::from(next_arg(&raw, &mut i, "--input-image")?));
            }
            "--mask" => {
                mask = Some(PathBuf::from(next_arg(&raw, &mut i, "--mask")?));
            }
            "--output-path" => {
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
            "--seed" => {
                seed = next_arg(&raw, &mut i, "--seed")?
                    .parse()
                    .map_err(|e| anyhow!("--seed: {e}"))?;
            }
            "--guidance" => {
                guidance = next_arg(&raw, &mut i, "--guidance")?
                    .parse()
                    .map_err(|e| anyhow!("--guidance: {e}"))?;
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
        seed,
        guidance,
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

    println!("=== FLUX 1 Dev — pure Rust INPAINT (LanPaint) ===");
    println!("Prompt:       {:?}", args.prompt);
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
    println!(
        "Size: {}x{}, steps={}, guidance={}",
        args.width, args.height, args.steps, args.guidance
    );
    println!("Seed: {}", args.seed);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: CLIP-L encode (pooled only) — same as flux1_infer
    // ------------------------------------------------------------------
    println!("--- Stage 1: CLIP-L encode ---");
    let t0 = Instant::now();

    let clip_weights_raw =
        flame_core::serialization::load_file(std::path::Path::new(CLIP_PATH), &device)?;
    let clip_weights: std::collections::HashMap<String, Tensor> = clip_weights_raw
        .into_iter()
        .map(|(k, v)| {
            let t = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)?
            };
            Ok::<_, flame_core::Error>((k, t))
        })
        .collect::<std::result::Result<_, _>>()?;
    println!(
        "  Loaded {} CLIP weights in {:.1}s",
        clip_weights.len(),
        t0.elapsed().as_secs_f32()
    );

    let clip_cfg = ClipConfig::default();
    let clip = ClipEncoder::new(clip_weights, clip_cfg, device.clone());

    let clip_tokens = tokenize_clip(&args.prompt);
    let (_clip_hidden, clip_pooled) = clip.encode(&clip_tokens)?;
    println!("  pooled: {:?}", clip_pooled.shape().dims());
    println!("  CLIP done in {:.1}s", t0.elapsed().as_secs_f32());
    drop(clip);
    println!();

    // ------------------------------------------------------------------
    // Stage 2: T5-XXL encode — same as flux1_infer
    // ------------------------------------------------------------------
    println!("--- Stage 2: T5-XXL encode ---");
    let t0 = Instant::now();

    let t5_hidden_bf16 = {
        let mut t5 = T5Encoder::load(T5_PATH, &device)?;
        println!("  Loaded T5 in {:.1}s", t0.elapsed().as_secs_f32());

        let t5_tokens = tokenize_t5(&args.prompt);
        let hidden = t5.encode(&t5_tokens)?;
        println!("  T5 hidden: {:?}", hidden.shape().dims());
        println!("  T5 encode done in {:.1}s", t0.elapsed().as_secs_f32());
        hidden
    };
    println!("  T5 weights evicted");
    println!();

    // ------------------------------------------------------------------
    // Stage 3: VAE encode the input image (BEFORE DiT load to free memory)
    //
    // The plain VAE downscale factor is 8 (latent is [1, 16, H/8, W/8]).
    // FLUX's pack op then reorganises that to [1, N, 64] tokens at H/16 ×
    // W/16 — but for inpaint mask alignment we work at H/8 × W/8, where
    // LanPaint operates.
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();

    let inputs = {
        let vae_enc = LdmVAEEncoder::from_safetensors(VAE_PATH, AE_IN_CHANNELS, &device)?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // VAE downscale at the *latent NCHW* level. Pack/unpack happens
            // inside the inner_model closure, not here.
            vae_scale: 8,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg, device.clone(), |img| {
            // encode_scaled applies (z - shift) * scale, matching what FLUX's
            // sampling expects in the latent space.
            vae_enc
                .encode_scaled(img, AE_SCALE_FACTOR, AE_SHIFT_FACTOR)
                .map_err(|e| anyhow!("vae encode: {e:?}"))
        })?
    };
    println!(
        "  latent_image: {:?}",
        inputs.latent_image.shape().dims()
    );
    println!("  latent_mask:  {:?}", inputs.latent_mask.shape().dims());
    println!("  pixel_mask:   {:?}", inputs.pixel_mask.shape().dims());
    println!("  Inpaint inputs prepared in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Build noise + load DiT
    // ------------------------------------------------------------------
    println!("--- Stage 4: Load FLUX 1 DiT (BlockOffloader) ---");
    let t0 = Instant::now();
    let dit = Flux1DiT::load(DIT_PATH, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    println!("--- Stage 5: Denoise ({} steps, LanPaint inner) ---", args.steps);

    // FLUX latent NCHW shape: [1, 16, 2*ceil(H/16), 2*ceil(W/16)]
    let latent_h = 2 * ((args.height + 15) / 16);
    let latent_w = 2 * ((args.width + 15) / 16);
    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        AE_IN_CHANNELS, latent_h, latent_w
    );

    // Sanity check: latent_image from VAE must agree.
    let li_dims = inputs.latent_image.shape().dims();
    if li_dims[2] != latent_h || li_dims[3] != latent_w {
        return Err(anyhow!(
            "latent shape mismatch: VAE gave {:?}, expected [_, _, {}, {}]",
            li_dims,
            latent_h,
            latent_w
        ));
    }

    // Seeded Gaussian noise (Box-Muller, deterministic).
    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    // Initial x at t=1: pure noise. LanPaint will replace the known region
    // with the noise-scaled latent_image inside `run()` automatically.
    let mut x_nchw = noise_nchw.clone_result()?;

    // Build the static img_ids/txt_ids once — they don't depend on time step.
    let (_dummy_packed, img_ids) = pack_latent(&noise_nchw, &device)?;
    let n_img = _dummy_packed.shape().dims()[1];
    drop(_dummy_packed);

    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )?;

    // Schedule: same FLUX 1 linear mu (256→0.5, 4096→1.15).
    let timesteps = get_schedule(args.steps, n_img, 0.5, 1.15, true);
    println!(
        "  Schedule: {} steps, t[0]={:.4}, t[1]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1,
        timesteps[0],
        timesteps[1],
        timesteps[args.steps]
    );

    // Guidance vec — distilled CFG, fed as model input.
    let guidance_vec = Tensor::from_vec(
        vec![args.guidance; 1],
        Shape::from_dims(&[1]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let pos_hidden = t5_hidden_bf16;

    // ------------------------------------------------------------------
    // Denoise loop — LanPaint per step, then standard Euler step.
    // ------------------------------------------------------------------
    let t0 = Instant::now();
    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: true, // FLUX is flow-matching
    };

    // References captured by `flux_step`. FLUX1 DiT `forward` takes
    // `&mut self` (it drives the BlockOffloader state), so we pipe `dit`
    // through a RefCell and borrow_mut at call sites.
    let dit_cell = std::cell::RefCell::new(dit);
    let device_ref = &device;
    let img_ids_ref = &img_ids;
    let txt_ids_ref = &txt_ids;
    let pos_hidden_ref = &pos_hidden;
    let clip_pooled_ref = &clip_pooled;
    let guidance_vec_ref = &guidance_vec;
    let target_h = args.height;
    let target_w = args.width;

    // Wrap one FLUX forward: pack NCHW, run DiT, unpack the velocity.
    // Returned tensor is the velocity in NCHW form — the CALLER decides
    // whether to convert to x_0 (LanPaint) or take an Euler step.
    //
    // `&dit_cell` shared ref is OK: we only borrow_mut inside.
    let flux_velocity = |x_nchw_in: &Tensor, t_vec: &Tensor| -> Result<Tensor> {
        let (packed, _) = pack_latent(x_nchw_in, device_ref)?;
        let mut dit_borrow = dit_cell.borrow_mut();
        let v_packed = dit_borrow.forward(
            &packed,
            pos_hidden_ref,
            t_vec,
            img_ids_ref,
            txt_ids_ref,
            Some(guidance_vec_ref),
            Some(clip_pooled_ref),
        )?;
        drop(dit_borrow);
        let v_nchw = unpack_latent(&v_packed, target_h, target_w)?;
        Ok(v_nchw)
    };

    for step in 0..args.steps {
        let t_curr = timesteps[step];
        let t_prev = timesteps[step + 1];

        // -------- Build per-step time scalars for LanPaint --------
        // FLUX flow-matching: sigma == t_curr (the schedule IS sigma).
        //   abt = (1 - t)^2 / ((1 - t)^2 + t^2)
        //   VE_Sigma = t / (1 - t)
        let flow_t = t_curr;
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
            // At t=1, sigma -> ∞. Clamp to a large value; LanPaint guards
            // against degenerate sigma anyway.
            1.0e6
        };

        let sigma_scalar = make_b1_bf16(ve_sigma_val, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        let tflow_scalar = make_b1_bf16(flow_t, &device)?;
        let t_vec_step = make_b1_bf16(flow_t, &device)?;
        let t_curr_local = t_curr;

        // -------- LanPaint inner Langevin loop --------
        // LanPaint's inner_model contract: takes `(x, t_flow)` and returns
        // x_0 PREDICTION (not velocity). For flow-matching models,
        //   x_0 = x - t * v(x, t)
        // so we call FLUX for velocity and convert inside the closure.
        let advanced_x = {
            // `inner_model_fn` borrows `dit_cell` and the captured refs.
            // It's a `Fn` (LanPaint requires Fn, not FnMut) — the RefCell
            // makes that OK despite the inner &mut borrow.
            let inner_model_fn =
                |x: &Tensor, t: &Tensor| -> flame_core::Result<Tensor> {
                    // FLUX's fused_linear3d_native is BF16-only. LanPaint's
                    // internal compute is F32 (autocast), so it hands us F32
                    // tensors. Downcast at the boundary, run FLUX in BF16,
                    // then upcast the velocity result back to F32 for the
                    // x_0 = x - t·v conversion that follows.
                    let x_bf16 = if x.dtype() == DType::BF16 {
                        x.clone()
                    } else {
                        x.to_dtype(DType::BF16)?
                    };
                    let t_bf16 = if t.dtype() == DType::BF16 {
                        t.clone()
                    } else {
                        t.to_dtype(DType::BF16)?
                    };
                    let v = flux_velocity(&x_bf16, &t_bf16).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "flux inner velocity: {e:?}"
                        ))
                    })?;
                    // x_0 = x - t * v in F32 to keep precision.
                    let x_f32 = x.to_dtype(DType::F32)?;
                    let v_f32 = v.to_dtype(DType::F32)?;
                    let t_f32 = t.to_dtype(DType::F32)?;
                    let img_dim = x.shape().dims().len();
                    let b = t.shape().dims()[0];
                    let mut tdims = vec![b];
                    tdims.extend(std::iter::repeat(1).take(img_dim - 1));
                    let t_b = t_f32.reshape(&tdims)?;
                    let x0 = x_f32.sub(&t_b.mul(&v_f32)?)?;
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
            // Drop `lanpaint` (and its captured closure) so we can reuse
            // the RefCell for the post-step Euler call.
            drop(lanpaint);
            let _ = t_curr_local; // quiet warning if unused branch below
            advanced_x
        };

        // -------- Standard FLUX Euler step on the advanced x --------
        // velocity at t_curr; x_{t-1} = x_t + (t_prev - t_curr) * v
        let v_nchw = flux_velocity(&advanced_x, &t_vec_step)?;
        let dt = t_prev - t_curr;
        let next_x = advanced_x.add(&v_nchw.mul_scalar(dt)?)?;

        // -------- Post-step mask blend (matches Wan2GP line 802-806) --------
        // noisy_image = latent_image * (1 - t_prev) + noise * t_prev
        // x = noisy_image * (1 - mask) + mask * x
        // Convention in our latent_mask: 1.0=preserve, 0.0=inpaint
        // → known region pulled toward (correctly noised) latent_image,
        //   unknown region keeps the model's denoised x.
        let nf = t_prev;
        let one_minus_nf = 1.0 - nf;
        let scaled_image = inputs.latent_image.mul_scalar(one_minus_nf)?;
        let scaled_noise = noise_nchw.mul_scalar(nf)?;
        let noisy_image = scaled_image.add(&scaled_noise)?;
        // where_mask(mask, a, b) returns a where mask==1 else b.
        // Want: latent_image-noised on (1-mask) [unknown=0], next_x on mask [known=1]
        //       OR equivalently: next_x on mask=1, noisy_image on mask=0.
        // But conceptually the spec says: keep model output in inpaint
        // region (mask=0), replace with the still-noised latent_image in
        // the known region (mask=1) so the next step still sees the
        // correct prior there.
        let blended = Tensor::where_mask(&inputs.latent_mask, &noisy_image, &next_x)?;

        x_nchw = blended;

        if step == 0 || step + 1 == args.steps || (step + 1) % 5 == 0 {
            println!(
                "  step {}/{} t_curr={:.4} t_prev={:.4} | dt-step done",
                step + 1,
                args.steps,
                t_curr,
                t_prev
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
    println!("  DiT evicted");

    // ------------------------------------------------------------------
    // Stage 6: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 6: VAE decode ---");
    let t0 = Instant::now();

    let vae = LdmVAEDecoder::from_safetensors(
        VAE_PATH,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
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
    // Stage 7: Save PNG
    // ------------------------------------------------------------------
    println!("--- Stage 7: Save PNG ---");
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

// ---------------------------------------------------------------------------
// Tokenization helpers (verbatim from flux1_infer.rs)
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(CLIP_TOKENIZER) {
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
            eprintln!("[flux_inpaint] CLIP tokenizer failed: {e}; using BOS/EOS fallback");
            let mut ids = vec![49406i32];
            ids.push(49407);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
    }
}

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.push(1);
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            eprintln!("[flux_inpaint] T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
    }
}
