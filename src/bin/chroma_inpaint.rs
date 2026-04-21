//! Chroma 8.9B inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `flux_inpaint.rs` with these Chroma-specific changes:
//!   - NO CLIP, NO pooled projection, NO distilled `guidance` scalar
//!     (Chroma's `distilled_guidance_layer` is invoked internally by the DiT).
//!   - REAL two-pass CFG (Chroma is NOT guidance-distilled, see
//!     `chroma_infer.rs`): noise = uncond + cfg * (cond - uncond).
//!   - T5-XXL-only text encoding (cond + uncond on empty negative).
//!   - Same VAE (16ch, scale=0.3611, shift=0.1159) and pack/unpack as FLUX.
//!   - Velocity convention: model output IS velocity. x_0 = x - t * v.
//!
//! CLI:
//!   chroma_inpaint --prompt "..." \
//!                  --input-image input.png --mask mask.png \
//!                  [--output-path inpaint_out.png]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::chroma_dit::ChromaDit;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{get_schedule, pack_latent, unpack_latent};
use inference_flame::vae::{LdmVAEDecoder, LdmVAEEncoder};

// ---------------------------------------------------------------------------
// Paths — match chroma_infer.rs
// ---------------------------------------------------------------------------

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const CHROMA_DIT_SHARDS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
];
const CHROMA_VAE: &str = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors";

const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/chroma_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";
const DEFAULT_NEGATIVE: &str = "";

const DEFAULT_SEED: u64 = 42;
const DEFAULT_STEPS: usize = 40;
const DEFAULT_CFG: f32 = 4.0;

const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;
const T5_SEQ_LEN: usize = 512;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    prompt: String,
    negative: String,
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
        "chroma_inpaint — Chroma 8.9B inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           chroma_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --prompt <TEXT>          Text prompt\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n\
         \n\
         OPTIONS:\n  \
           --negative <TEXT>        Negative prompt [default: empty]\n  \
           --output-path <PATH>     Output PNG path [default: {default_output}]\n  \
           --width <N>              Output width [default: 1024]\n  \
           --height <N>             Output height [default: 1024]\n  \
           --steps <N>              Diffusion steps [default: {steps}]\n  \
           --cfg <F>                Classifier-free guidance scale [default: {cfg}]\n  \
           --seed <N>               RNG seed [default: {seed}]\n  \
           -h, --help               Print this help",
        default_output = DEFAULT_OUTPUT,
        steps = DEFAULT_STEPS,
        cfg = DEFAULT_CFG,
        seed = DEFAULT_SEED,
    );
}

fn parse_cli() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().collect();

    let mut prompt: Option<String> = None;
    let mut negative: String = DEFAULT_NEGATIVE.to_string();
    let mut input_image: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut output_path: PathBuf = PathBuf::from(DEFAULT_OUTPUT);
    let mut width: usize = 1024;
    let mut height: usize = 1024;
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
            "--negative" => negative = next_arg(&raw, &mut i, "--negative")?,
            "--input-image" => {
                input_image = Some(PathBuf::from(next_arg(&raw, &mut i, "--input-image")?))
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
            "--cfg" | "--guidance" => {
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
        negative,
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

    println!("=== Chroma 8.9B — pure Rust INPAINT (LanPaint) ===");
    println!("Prompt:       {:?}", args.prompt);
    println!("Negative:     {:?}", args.negative);
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
    println!(
        "Size: {}x{}, steps={}, cfg={}",
        args.width, args.height, args.steps, args.cfg
    );
    println!("Seed: {}", args.seed);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: T5-XXL encode (cond + uncond), then drop encoder
    // ------------------------------------------------------------------
    println!("--- Stage 1: T5-XXL encode (cond + uncond) ---");
    let t0 = Instant::now();
    let (t5_cond, t5_uncond) = {
        let mut t5 = T5Encoder::load(T5_PATH, &device)?;
        println!("  Loaded T5 in {:.1}s", t0.elapsed().as_secs_f32());
        let cond_tokens = tokenize_t5(&args.prompt);
        let uncond_tokens = tokenize_t5(&args.negative);
        let cond_h = t5.encode(&cond_tokens)?;
        let uncond_h = t5.encode(&uncond_tokens)?;
        println!("  cond hidden:   {:?}", cond_h.shape().dims());
        println!("  uncond hidden: {:?}", uncond_h.shape().dims());
        (cond_h, uncond_h)
    };
    println!("  T5 evicted in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode source image (BEFORE DiT load)
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();
    let inputs = {
        let vae_enc = LdmVAEEncoder::from_safetensors(CHROMA_VAE, AE_IN_CHANNELS, &device)?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let cfg = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // Pack/unpack lives inside the velocity closure; helper aligns
            // mask to the NCHW latent grid (H/8 × W/8).
            vae_scale: 8,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg, device.clone(), |img| {
            vae_enc
                .encode_scaled(img, AE_SCALE_FACTOR, AE_SHIFT_FACTOR)
                .map_err(|e| anyhow!("vae encode: {e:?}"))
        })?
    };
    println!("  latent_image: {:?}", inputs.latent_image.shape().dims());
    println!("  latent_mask:  {:?}", inputs.latent_mask.shape().dims());
    println!("  pixel_mask:   {:?}", inputs.pixel_mask.shape().dims());
    println!("  Inpaint inputs prepared in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Load Chroma DiT (BlockOffloader)
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load Chroma DiT (BlockOffloader) ---");
    let t_load = Instant::now();
    let dit = ChromaDit::load(CHROMA_DIT_SHARDS, &device)?;
    println!("  DiT loaded in {:.1}s", t_load.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Build noise + denoise (LanPaint inner + Chroma Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, CFG={}, LanPaint inner) ---",
        args.steps, args.cfg
    );

    // FLUX-style latent geometry: VAE 8× + pack 2× = 16× effective.
    let latent_h = 2 * ((args.height + 15) / 16);
    let latent_w = 2 * ((args.width + 15) / 16);
    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        AE_IN_CHANNELS, latent_h, latent_w
    );

    // Sanity: VAE must agree on shape.
    let li_dims = inputs.latent_image.shape().dims();
    if li_dims[2] != latent_h || li_dims[3] != latent_w {
        return Err(anyhow!(
            "latent shape mismatch: VAE gave {:?}, expected [_, _, {}, {}]",
            li_dims,
            latent_h,
            latent_w
        ));
    }

    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;
    let mut x_nchw = noise_nchw.clone_result()?;

    // Static img_ids/txt_ids built once.
    let (dummy_packed, img_ids) = pack_latent(&noise_nchw, &device)?;
    let n_img = dummy_packed.shape().dims()[1];
    drop(dummy_packed);
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )?;

    // Same FLUX flow-match schedule, dynamic-mu rectified flow.
    let timesteps = get_schedule(args.steps, n_img, 0.5, 1.15, true);
    println!(
        "  Schedule: {} steps, t[0]={:.4}, t[1]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1,
        timesteps[0],
        timesteps[1],
        timesteps[args.steps]
    );

    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: true,
    };

    // Chroma's forward takes &mut self → wrap in RefCell so the LanPaint Fn
    // closure can borrow_mut.
    let dit_cell = RefCell::new(dit);
    let cond_ref = &t5_cond;
    let uncond_ref = &t5_uncond;
    let img_ids_ref = &img_ids;
    let txt_ids_ref = &txt_ids;
    let cfg_scale = args.cfg;
    let target_h = args.height;
    let target_w = args.width;

    // One Chroma forward — packs NCHW input, runs cond + uncond, real CFG combine,
    // returns NCHW velocity. Two forwards per call (Chroma uses real CFG).
    let chroma_velocity = |x_nchw_in: &Tensor, t_vec: &Tensor| -> Result<Tensor> {
        let (packed, _) = pack_latent(x_nchw_in, &device)?;
        let mut d = dit_cell.borrow_mut();
        let cond_pred = d.forward(&packed, cond_ref, t_vec, img_ids_ref, txt_ids_ref)?;
        let uncond_pred = d.forward(&packed, uncond_ref, t_vec, img_ids_ref, txt_ids_ref)?;
        drop(d);
        // CFG: noise = uncond + cfg * (cond - uncond)
        let diff = cond_pred.sub(&uncond_pred)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        let pred_packed = uncond_pred.add(&scaled)?;
        let v_nchw = unpack_latent(&pred_packed, target_h, target_w)?;
        Ok(v_nchw)
    };

    let t_denoise = Instant::now();
    for step in 0..args.steps {
        let t_curr = timesteps[step];
        let t_prev = timesteps[step + 1];

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
            1.0e6
        };
        let sigma_scalar = make_b1_bf16(ve_sigma_val, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        let tflow_scalar = make_b1_bf16(flow_t, &device)?;
        let t_vec_step = make_b1_bf16(flow_t, &device)?;

        // LanPaint inner Langevin loop.
        // Velocity convention: x_0 = x - t * v (FLUX-style).
        let advanced_x = {
            let inner_model_fn = |x: &Tensor, t: &Tensor| -> flame_core::Result<Tensor> {
                let v = chroma_velocity(x, t).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!("chroma inner: {e:?}"))
                })?;
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

            let (_lp_x0, advanced_x) = lanpaint_step(
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

        // Chroma Euler step: x_next = x + (t_prev - t_curr) * v
        let v_nchw = chroma_velocity(&advanced_x, &t_vec_step)?;
        let dt = t_prev - t_curr;
        let next_x = advanced_x.add(&v_nchw.mul_scalar(dt)?)?;

        let nf = t_prev;
        let one_minus_nf = 1.0 - nf;
        let scaled_image = inputs.latent_image.mul_scalar(one_minus_nf)?;
        let scaled_noise = noise_nchw.mul_scalar(nf)?;
        let noisy_image = scaled_image.add(&scaled_noise)?;
        let blended = Tensor::where_mask(&inputs.latent_mask, &noisy_image, &next_x)?;
        x_nchw = blended;

        if step == 0 || step + 1 == args.steps || (step + 1) % 5 == 0 {
            println!(
                "  step {}/{}  t_curr={:.4}  t_prev={:.4}  ({:.1}s elapsed)",
                step + 1,
                args.steps,
                t_curr,
                t_prev,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step + LanPaint inner)",
        dt_denoise,
        dt_denoise / args.steps as f32
    );
    println!();

    drop(dit_cell);
    drop(t5_cond);
    drop(t5_uncond);
    println!("  DiT + cached embeddings evicted");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        CHROMA_VAE,
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

fn make_b1_bf16(v: f32, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::from_vec(vec![v], Shape::from_dims(&[1]), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

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

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            eprintln!("[chroma_inpaint] T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
    }
}
