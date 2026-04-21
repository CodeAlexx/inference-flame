//! Klein 4B / 9B (Flux 2) inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `flux_inpaint.rs` structure with Klein-specific bits:
//!   - Qwen3 4B (variant=4b) or Qwen3 8B (variant=9b) text encoder, with
//!     Klein chat-template wrapping and 512-token padding.
//!   - Real two-pass CFG (Klein is NOT distilled — see `klein_infer.rs`).
//!   - Klein VAE: 32-channel raw latent → 16× downscale → packed into
//!     128 channels at H/16 × W/16 (the patchify lives inside the VAE).
//!     Latent NCHW shape used in the denoise loop is `[1, 128, H/16, W/16]`.
//!   - Direct velocity: model output IS velocity (no negation, see
//!     `klein_sampling::euler_denoise`). For LanPaint x_0 closure we use
//!     `x_0 = x - t * v` (same as FLUX).
//!   - Schedule: dynamic-mu rectified flow (`klein_sampling::get_schedule`).
//!   - The DiT operates on a packed sequence `[B, H_lat*W_lat, 128]`; we
//!     pack inside the velocity closure and unpack the result so LanPaint
//!     operates on NCHW like the helper expects.
//!
//! CLI:
//!   klein_inpaint --variant 4b|9b --prompt "..." \
//!                 --input-image input.png --mask mask.png \
//!                 [--output-path inpaint_out.png]

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
use inference_flame::models::klein::{KleinOffloaded, KleinTransformer};
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::sampling::klein_sampling::get_schedule;
use inference_flame::vae::klein_vae::{KleinVaeDecoder, KleinVaeEncoder};

// ---------------------------------------------------------------------------
// Paths — match klein_infer / klein9b_infer
// ---------------------------------------------------------------------------

const MODEL_4B: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-4b.safetensors";
const MODEL_9B: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";

const ENCODER_4B_PATH: &str = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors";
const ENCODER_8B_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";

const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";

const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/klein_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";
const DEFAULT_NEGATIVE: &str = "lowres, bad quality, worst quality, bad anatomy, blurry, watermark, simple background, transparent background, sketch, jpeg artifacts, ugly, poorly drawn, censor";

const DEFAULT_SEED: u64 = 42;
const DEFAULT_STEPS: usize = 50;
const DEFAULT_CFG: f32 = 4.0;

const TXT_PAD_LEN: usize = 512;
const QWEN3_PAD_ID: i32 = 151643;
const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

// Klein latent: VAE produces packed [1, 128, H/16, W/16].
const LATENT_CHANNELS: usize = 128;
const VAE_DOWNSCALE: usize = 16;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    K4B,
    K9B,
}

impl Variant {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "4b" => Ok(Variant::K4B),
            "9b" => Ok(Variant::K9B),
            other => Err(anyhow!("--variant expects 4b|9b, got {other}")),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Variant::K4B => "4B",
            Variant::K9B => "9B",
        }
    }
    fn model_path(self) -> &'static str {
        match self {
            Variant::K4B => MODEL_4B,
            Variant::K9B => MODEL_9B,
        }
    }
}

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
    variant: Variant,
}

fn print_help() {
    println!(
        "klein_inpaint — Klein 4B / 9B (Flux 2) inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           klein_inpaint --variant 4b|9b --prompt <TEXT> \\\n  \
                         --input-image <PATH> --mask <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --variant <4b|9b>        Klein variant (4B uses Qwen3 4B encoder, 9B uses Qwen3 8B)\n  \
           --prompt <TEXT>          Text prompt\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n\
         \n\
         OPTIONS:\n  \
           --negative <TEXT>        Negative prompt [default: stock list]\n  \
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
    let mut variant: Option<Variant> = None;

    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--variant" => {
                variant = Some(Variant::parse(&next_arg(&raw, &mut i, "--variant")?)?);
            }
            "--prompt" => prompt = Some(next_arg(&raw, &mut i, "--prompt")?),
            "--negative" => negative = next_arg(&raw, &mut i, "--negative")?,
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
    let variant = variant.ok_or_else(|| anyhow!("missing required --variant 4b|9b"))?;

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
        variant,
    })
}

fn next_arg(raw: &[String], i: &mut usize, flag: &str) -> Result<String> {
    *i += 1;
    raw.get(*i)
        .cloned()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

// ---------------------------------------------------------------------------
// Klein model wrapper — mirrors klein9b_infer's GPU/Offloaded fallback
// ---------------------------------------------------------------------------

enum KleinModel {
    OnGpu(KleinTransformer),
    Offloaded(KleinOffloaded),
}

impl KleinModel {
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        t: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> flame_core::Result<Tensor> {
        match self {
            KleinModel::OnGpu(m) => m.forward(img, txt, t, img_ids, txt_ids),
            KleinModel::Offloaded(m) => m.forward(img, txt, t, img_ids, txt_ids),
        }
    }
}

fn load_qwen3_8b_sharded(
    dir: &str,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let mut all = HashMap::new();
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with("model-") && n.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();
    for p in &shard_paths {
        let shard = flame_core::serialization::load_file(p, device)
            .map_err(|e| anyhow!("load shard {}: {e:?}", p.display()))?;
        all.extend(shard);
    }
    Ok(all)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();

    let args = parse_cli()?;
    let t_total = Instant::now();
    let device = global_cuda_device();

    println!("=== Klein {} — pure Rust INPAINT (LanPaint) ===", args.variant.label());
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

    if args.width % VAE_DOWNSCALE != 0 || args.height % VAE_DOWNSCALE != 0 {
        return Err(anyhow!(
            "Klein requires width/height divisible by {VAE_DOWNSCALE}; got {}x{}",
            args.width,
            args.height
        ));
    }

    // ------------------------------------------------------------------
    // Stage 1: Qwen3 encode (cond + uncond), then drop encoder
    // ------------------------------------------------------------------
    println!("--- Stage 1: Qwen3 encode (cond + uncond) ---");
    let t0 = Instant::now();
    let (pos_hidden, neg_hidden) = {
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow!("Tokenizer load failed: {}", e))?;
        let pos_fmt = format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", args.prompt);
        let neg_fmt = format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", args.negative);
        let pos_enc = tokenizer
            .encode(pos_fmt.as_str(), false)
            .map_err(|e| anyhow!("tokenize: {}", e))?;
        let neg_enc = tokenizer
            .encode(neg_fmt.as_str(), false)
            .map_err(|e| anyhow!("tokenize: {}", e))?;
        let mut pos_ids: Vec<i32> = pos_enc.get_ids().iter().map(|&id| id as i32).collect();
        let mut neg_ids: Vec<i32> = neg_enc.get_ids().iter().map(|&id| id as i32).collect();
        println!("  pos tokens: {}, neg tokens: {}", pos_ids.len(), neg_ids.len());
        pos_ids.resize(TXT_PAD_LEN, QWEN3_PAD_ID);
        neg_ids.resize(TXT_PAD_LEN, QWEN3_PAD_ID);

        let enc_weights = match args.variant {
            Variant::K4B => flame_core::serialization::load_file(
                std::path::Path::new(ENCODER_4B_PATH),
                &device,
            )?,
            Variant::K9B => load_qwen3_8b_sharded(ENCODER_8B_DIR, &device)?,
        };
        let cfg = Qwen3Encoder::config_from_weights(&enc_weights)?;
        let encoder = Qwen3Encoder::new(enc_weights, cfg, device.clone());
        let pos_h = encoder.encode(&pos_ids)?;
        let neg_h = encoder.encode(&neg_ids)?;
        drop(encoder);
        (pos_h, neg_h)
    };
    println!("  pos hidden: {:?}", pos_hidden.dims());
    println!("  neg hidden: {:?}", neg_hidden.dims());
    println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode source image (BEFORE DiT load)
    //   Klein VAE returns packed latent [1, 128, H/16, W/16] AFTER BN.
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();
    let vae_weights = flame_core::serialization::load_file(
        std::path::Path::new(VAE_PATH),
        &device,
    )?;
    println!("  VAE weights loaded ({} keys)", vae_weights.len());
    let vae_device = flame_core::device::Device::from_arc(device.clone());

    let inputs = {
        let vae_enc = KleinVaeEncoder::load(&vae_weights, &vae_device)
            .map_err(|e| anyhow!("VAE encoder load: {e:?}"))?;
        println!("  VAE encoder built in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // Klein patchifies inside the VAE — the latent operates at H/16.
            vae_scale: VAE_DOWNSCALE,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg, device.clone(), |img| {
            // BN-normalized latent matches noise N(0,1), the right input for
            // the LanPaint blend `(1-t)*latent + t*noise`.
            vae_enc
                .encode(img)
                .map_err(|e| anyhow!("klein vae encode: {e:?}"))
        })?
    };
    println!("  latent_image: {:?}", inputs.latent_image.shape().dims());
    println!("  latent_mask:  {:?}", inputs.latent_mask.shape().dims());
    println!("  pixel_mask:   {:?}", inputs.pixel_mask.shape().dims());
    println!("  Inpaint inputs prepared in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Load Klein DiT — try resident, fall back to offloaded
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load Klein {} DiT ---", args.variant.label());
    let t_load = Instant::now();
    let model = match KleinTransformer::from_safetensors(args.variant.model_path()) {
        Ok(m) => {
            println!("  Resident on GPU");
            KleinModel::OnGpu(m)
        }
        Err(e) => {
            println!("  Resident load failed ({:?}); falling back to KleinOffloaded", e);
            KleinModel::Offloaded(KleinOffloaded::from_safetensors(args.variant.model_path())?)
        }
    };
    println!("  Loaded in {:.1}s", t_load.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Build noise + denoise loop (LanPaint inner + Klein Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, CFG={}, LanPaint inner) ---",
        args.steps, args.cfg
    );

    let latent_h = args.height / VAE_DOWNSCALE;
    let latent_w = args.width / VAE_DOWNSCALE;
    let n_img = latent_h * latent_w;
    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]; packed seq len = {}",
        LATENT_CHANNELS, latent_h, latent_w, n_img
    );

    // Sanity: VAE-encoded latent must match.
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

    // img_ids: [N, 4] each row [0, row, col, 0]; txt_ids: zeros [TXT_PAD_LEN, 4]
    let mut img_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_data[idx * 4 + 1] = r as f32;
            img_data[idx * 4 + 2] = c as f32;
        }
    }
    let img_ids = Tensor::from_f32_to_bf16(img_data, Shape::from_dims(&[n_img, 4]), device.clone())?;
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[TXT_PAD_LEN, 4]),
        DType::BF16,
        device.clone(),
    )?;

    // Seeded noise [1, 128, H/16, W/16], BF16.
    let numel = LATENT_CHANNELS * latent_h * latent_w;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, LATENT_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    // Initial x at t=1: pure noise (LanPaint replaces known region inside run()).
    let mut x_nchw = noise_nchw.clone_result()?;

    // Klein BFL schedule: dynamic mu, returns NUM_STEPS+1 values, ts[0]=~1, ts[-1]=0.
    let timesteps = get_schedule(args.steps, n_img);
    println!(
        "  Schedule: {} values, t[0]={:.4}, t[1]={:.4}, t[-1]={:.4}",
        timesteps.len(),
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
        is_flow: true, // Klein is rectified-flow.
    };

    // The model takes &self, but we keep the same RefCell pattern as the
    // FLUX/Z-Image references for symmetry. RefCell::borrow() is enough.
    let model_cell = RefCell::new(model);

    // One Klein forward — packs NCHW input, applies real CFG (cond + uncond),
    // unpacks back to NCHW velocity. CFG formula matches klein_infer.rs:
    //   pred = uncond + cfg * (cond - uncond)
    let pos_hidden_ref = &pos_hidden;
    let neg_hidden_ref = &neg_hidden;
    let img_ids_ref = &img_ids;
    let txt_ids_ref = &txt_ids;
    let cfg_scale = args.cfg;
    let target_h = latent_h;
    let target_w = latent_w;

    let klein_velocity = |x_nchw_in: &Tensor, t_vec: &Tensor| -> Result<Tensor> {
        // Pack: [1, 128, H, W] -> [1, H, W, 128] -> [1, H*W, 128]
        let x_packed = x_nchw_in
            .permute(&[0, 2, 3, 1])?
            .reshape(&[1, target_h * target_w, LATENT_CHANNELS])?;
        let m = model_cell.borrow();
        let pred_cond = m.forward(&x_packed, pos_hidden_ref, t_vec, img_ids_ref, txt_ids_ref)?;
        let pred_uncond =
            m.forward(&x_packed, neg_hidden_ref, t_vec, img_ids_ref, txt_ids_ref)?;
        drop(m);
        // pred_cfg = uncond + cfg * (cond - uncond)
        let diff = pred_cond.sub(&pred_uncond)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        let pred_packed = pred_uncond.add(&scaled)?;
        // Unpack: [1, H*W, 128] -> [1, H, W, 128] -> [1, 128, H, W]
        let v_nchw = pred_packed
            .reshape(&[1, target_h, target_w, LATENT_CHANNELS])?
            .permute(&[0, 3, 1, 2])?;
        Ok(v_nchw)
    };

    let t_denoise = Instant::now();
    for step in 0..args.steps {
        let t_curr = timesteps[step];
        let t_prev = timesteps[step + 1];

        // Flow-matching scalars (sigma == t_curr; abt, ve_sigma derived).
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

        // -------- LanPaint inner Langevin loop --------
        // Klein: model output IS velocity (direct, no negation).
        //   x_0 = x - t * v   (same formula as FLUX).
        let advanced_x = {
            let inner_model_fn = |x: &Tensor, t: &Tensor| -> flame_core::Result<Tensor> {
                let v = klein_velocity(x, t).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!("klein inner: {e:?}"))
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

        // -------- Klein Euler step on the advanced x --------
        // Direct velocity: x_next = x + (t_prev - t_curr) * v
        let v_nchw = klein_velocity(&advanced_x, &t_vec_step)?;
        let dt = t_prev - t_curr;
        let next_x = advanced_x.add(&v_nchw.mul_scalar(dt)?)?;

        // -------- Mask blend so known region tracks the proper noised prior --------
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
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step)",
        dt_denoise,
        dt_denoise / args.steps as f32
    );
    println!();

    // Free DiT before VAE decoder load.
    drop(model_cell);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  DiT evicted; pool trimmed");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();
    let vae_dec = KleinVaeDecoder::load(&vae_weights, &vae_device)
        .map_err(|e| anyhow!("VAE decoder load: {e:?}"))?;
    println!("  VAE decoder built in {:.1}s", t0.elapsed().as_secs_f32());

    let rgb = vae_dec.decode(&x_nchw).map_err(|e| anyhow!("vae decode: {e:?}"))?;
    drop(x_nchw);
    drop(vae_dec);
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
