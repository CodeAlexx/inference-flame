//! SD 3.5 Medium (MMDiT) inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `flux_inpaint.rs` with these SD3-specific bits:
//!   - Triple text encoder load (CLIP-L + CLIP-G + T5-XXL), each one-shot
//!     and freed before DiT load — see `sd3_infer.rs` for the canonical
//!     pattern. Outputs (context, pooled) for cond + uncond.
//!   - Real two-pass CFG: noise = uncond + cfg * (cond - uncond).
//!   - VAE: SD3 16-channel, 8× downscale, scale=1.5305, shift=0.0609.
//!   - Latent NCHW shape [1, 16, H/8, W/8] — no pack/unpack (the MMDiT's
//!     patchify is internal). LanPaint operates directly on NCHW.
//!   - Velocity convention: model output IS velocity (rectified flow).
//!     x_0 = x - t * v.
//!   - Schedule: rectified-flow with shift=3.0 (matches sd3_infer.rs).
//!     Timestep into the model is `t * 1000.0` (sigma in [0,1] → "steps"
//!     in [0, 1000] for the MMDiT timestep_embed).
//!
//! CLI:
//!   sd3_inpaint --prompt "..." --input-image input.png --mask mask.png \
//!               [--output-path inpaint_out.png]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::sd3_mmdit::{load_sd3_all_chunked, SD3MMDiT};
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::vae::{LdmVAEDecoder, LdmVAEEncoder};

// ---------------------------------------------------------------------------
// Paths — match sd3_infer.rs
// ---------------------------------------------------------------------------

const CLIP_L_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_L_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const CLIP_G_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors";
const CLIP_G_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json";

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const MODEL_PATH: &str = "/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors";

const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/sd3_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";
const DEFAULT_NEGATIVE: &str = "";

const DEFAULT_SEED: u64 = 42;
const DEFAULT_STEPS: usize = 28;
const DEFAULT_CFG: f32 = 4.5;
const DEFAULT_SHIFT: f32 = 3.0;

const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 256;

const VAE_IN_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;
const VAE_SHIFT: f32 = 0.0609;

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
    shift: f32,
    seed: u64,
}

fn print_help() {
    println!(
        "sd3_inpaint — SD 3.5 Medium MMDiT inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           sd3_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> [OPTIONS]\n\
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
           --shift <F>              Rectified-flow schedule shift [default: {shift}]\n  \
           --seed <N>               RNG seed [default: {seed}]\n  \
           -h, --help               Print this help",
        default_output = DEFAULT_OUTPUT,
        steps = DEFAULT_STEPS,
        cfg = DEFAULT_CFG,
        shift = DEFAULT_SHIFT,
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
    let mut shift: f32 = DEFAULT_SHIFT;
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
        shift,
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
// Tokenizers — same as sd3_infer.rs
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str, tokenizer_path: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(tokenizer_path) {
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
            eprintln!("[sd3_inpaint] CLIP tokenizer failed: {e}; using BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
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
            eprintln!("[sd3_inpaint] T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            ids.resize(T5_SEQ_LEN, 0);
            ids
        }
    }
}

fn zero_pad_last_dim(x: &Tensor, target_dim: usize) -> Result<Tensor> {
    let dims = x.dims();
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    if c >= target_dim {
        return Ok(x.clone());
    }
    let pad = Tensor::zeros_dtype(
        Shape::from_dims(&[b, n, target_dim - c]),
        DType::BF16,
        x.device().clone(),
    )?;
    Ok(Tensor::cat(&[x, &pad], 2)?)
}

fn encode_text_pair(
    prompt: &str,
    negative: &str,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    fn load_clip_weights(
        path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let raw =
            flame_core::serialization::load_file(std::path::Path::new(path), device)?;
        let weights = raw
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
        Ok(weights)
    }

    // CLIP-L
    println!("  CLIP-L...");
    let t0 = Instant::now();
    let (clip_l_h, clip_l_p, clip_l_h_u, clip_l_p_u) = {
        let weights = load_clip_weights(CLIP_L_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, pc) = clip.encode_sd3(&tokenize_clip(prompt, CLIP_L_TOKENIZER))?;
        let (hu, pu) = clip.encode_sd3(&tokenize_clip(negative, CLIP_L_TOKENIZER))?;
        (hc, pc, hu, pu)
    };
    println!(
        "    cond: {:?}  uncond: {:?}  ({:.1}s)",
        clip_l_h.dims(),
        clip_l_h_u.dims(),
        t0.elapsed().as_secs_f32()
    );

    // CLIP-G
    println!("  CLIP-G...");
    let t0 = Instant::now();
    let (clip_g_h, clip_g_p, clip_g_h_u, clip_g_p_u) = {
        let weights = load_clip_weights(CLIP_G_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::clip_g(), device.clone());
        let (hc, pc) = clip.encode_sd3(&tokenize_clip(prompt, CLIP_G_TOKENIZER))?;
        let (hu, pu) = clip.encode_sd3(&tokenize_clip(negative, CLIP_G_TOKENIZER))?;
        (hc, pc, hu, pu)
    };
    println!(
        "    cond: {:?}  uncond: {:?}  ({:.1}s)",
        clip_g_h.dims(),
        clip_g_h_u.dims(),
        t0.elapsed().as_secs_f32()
    );

    // T5-XXL
    println!("  T5-XXL...");
    let t0 = Instant::now();
    let (t5_h, t5_h_u) = {
        let mut t5 = T5Encoder::load(T5_PATH, device)?;
        let hc = t5.encode(&tokenize_t5(prompt))?;
        let hu = t5.encode(&tokenize_t5(negative))?;
        (hc, hu)
    };
    let t5_h = t5_h.narrow(1, 0, t5_h.dims()[1].min(T5_SEQ_LEN))?;
    let t5_h_u = t5_h_u.narrow(1, 0, t5_h_u.dims()[1].min(T5_SEQ_LEN))?;
    println!(
        "    cond: {:?}  uncond: {:?}  ({:.1}s)",
        t5_h.dims(),
        t5_h_u.dims(),
        t0.elapsed().as_secs_f32()
    );

    // Combine: [CLIP-L padded to 4096] :: [CLIP-G padded to 4096] :: T5
    let cl_pad = zero_pad_last_dim(&clip_l_h, 4096)?;
    let cg_pad = zero_pad_last_dim(&clip_g_h, 4096)?;
    let context = Tensor::cat(&[&cl_pad, &cg_pad, &t5_h], 1)?;
    let pooled = Tensor::cat(&[&clip_l_p, &clip_g_p], 1)?;

    let cl_pad_u = zero_pad_last_dim(&clip_l_h_u, 4096)?;
    let cg_pad_u = zero_pad_last_dim(&clip_g_h_u, 4096)?;
    let context_u = Tensor::cat(&[&cl_pad_u, &cg_pad_u, &t5_h_u], 1)?;
    let pooled_u = Tensor::cat(&[&clip_l_p_u, &clip_g_p_u], 1)?;

    println!(
        "  Combined: cond {:?}  uncond {:?}  pooled {:?}",
        context.dims(),
        context_u.dims(),
        pooled.dims()
    );
    Ok((context, pooled, context_u, pooled_u))
}

fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();

    let args = parse_cli()?;
    let t_total = Instant::now();
    let device = global_cuda_device();

    println!("=== SD 3.5 Medium — pure Rust INPAINT (LanPaint) ===");
    println!("Prompt:       {:?}", args.prompt);
    println!("Negative:     {:?}", args.negative);
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
    println!(
        "Size: {}x{}, steps={}, cfg={}, shift={}",
        args.width, args.height, args.steps, args.cfg, args.shift
    );
    println!("Seed: {}", args.seed);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: Triple text encoding (cond + uncond)
    // ------------------------------------------------------------------
    println!("--- Stage 1: Text Encoding (CLIP-L + CLIP-G + T5-XXL) ---");
    let t0 = Instant::now();
    let (context, pooled, context_uncond, pooled_uncond) =
        encode_text_pair(&args.prompt, &args.negative, &device)?;
    println!("  Total encode: {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode (BEFORE DiT load to keep VRAM in budget)
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();
    let inputs = {
        let vae_enc = LdmVAEEncoder::from_safetensors(MODEL_PATH, VAE_IN_CHANNELS, &device)
            .map_err(|e| anyhow!("VAE encoder load: {e:?}"))?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // SD3 latent NCHW = [1, 16, H/8, W/8]; no pack inside the model.
            vae_scale: 8,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg, device.clone(), |img| {
            vae_enc
                .encode_scaled(img, VAE_SCALE, VAE_SHIFT)
                .map_err(|e| anyhow!("vae encode: {e:?}"))
        })?
    };
    println!("  latent_image: {:?}", inputs.latent_image.shape().dims());
    println!("  latent_mask:  {:?}", inputs.latent_mask.shape().dims());
    println!("  pixel_mask:   {:?}", inputs.pixel_mask.shape().dims());
    println!("  Inpaint inputs prepared in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Load SD 3.5 Medium MMDiT (resident)
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load SD 3.5 Medium ---");
    let t_load = Instant::now();
    let resident = load_sd3_all_chunked(MODEL_PATH, &device)?;
    println!("  {} total weight tensors", resident.len());
    let model = SD3MMDiT::new(MODEL_PATH.to_string(), resident, device.clone());
    println!(
        "  depth={}, hidden={}, heads={}, dual_attn={}",
        model.config.depth,
        model.config.hidden_size,
        model.config.num_heads,
        model.config.use_dual_attention
    );
    println!("  Loaded in {:.1}s", t_load.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Build noise + denoise (LanPaint inner + SD3 Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, CFG={}, LanPaint inner) ---",
        args.steps, args.cfg
    );

    let latent_h = args.height / 8;
    let latent_w = args.width / 8;
    println!(
        "  Latent [B,C,H,W] = [1, {}, {}, {}]",
        VAE_IN_CHANNELS, latent_h, latent_w
    );

    let li_dims = inputs.latent_image.shape().dims();
    if li_dims[1] != VAE_IN_CHANNELS || li_dims[2] != latent_h || li_dims[3] != latent_w {
        return Err(anyhow!(
            "latent shape mismatch: VAE gave {:?}, expected [_, {}, {}, {}]",
            li_dims,
            VAE_IN_CHANNELS,
            latent_h,
            latent_w
        ));
    }

    let numel = VAE_IN_CHANNELS * latent_h * latent_w;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;
    let mut x_nchw = noise_nchw.clone_result()?;

    let timesteps = build_schedule(args.steps, args.shift);
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

    // SD3MMDiT.forward takes &mut self → wrap in RefCell.
    let dit_cell = RefCell::new(model);
    let context_ref = &context;
    let pooled_ref = &pooled;
    let context_u_ref = &context_uncond;
    let pooled_u_ref = &pooled_uncond;
    let cfg_scale = args.cfg;
    let device_ref = &device;

    // One SD3 forward — uses NCHW latent directly; runs cond + uncond and
    // combines via real CFG. The MMDiT's timestep_embed expects values in
    // [0, 1000] (sd3_infer.rs passes `t * 1000.0`); we mirror that.
    let sd3_velocity = |x: &Tensor, t_flow: f32| -> Result<Tensor> {
        let t_vec = Tensor::from_f32_to_bf16(
            vec![t_flow * 1000.0],
            Shape::from_dims(&[1]),
            device_ref.clone(),
        )?;
        let mut m = dit_cell.borrow_mut();
        let pred_cond = m.forward(x, &t_vec, context_ref, pooled_ref)?;
        let pred_uncond = m.forward(x, &t_vec, context_u_ref, pooled_u_ref)?;
        drop(m);
        let diff = pred_cond.sub(&pred_uncond)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        let v = pred_uncond.add(&scaled)?;
        Ok(v)
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

        // -------- LanPaint inner Langevin loop --------
        // Velocity convention: x_0 = x - t * v.
        let advanced_x = {
            let inner_model_fn = |x: &Tensor, t: &Tensor| -> flame_core::Result<Tensor> {
                // LanPaint feeds [B] BF16 t in [0, 1] (flow time); convert
                // scalar back from t for sd3_velocity.
                let t_scalar = t
                    .to_dtype(DType::F32)?
                    .to_vec_f32()
                    .map_err(|e| {
                        flame_core::Error::InvalidOperation(format!("t to vec: {e:?}"))
                    })?
                    .first()
                    .copied()
                    .unwrap_or(flow_t);
                let v = sd3_velocity(x, t_scalar).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!("sd3 inner: {e:?}"))
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

        // SD3 Euler step: x_next = x + (t_prev - t_curr) * v
        let v_nchw = sd3_velocity(&advanced_x, flow_t)?;
        let dt = t_prev - t_curr;
        let next_x = advanced_x.add(&v_nchw.mul_scalar(dt)?)?;

        // Mask blend on known region.
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

    // Free DiT + embeddings before VAE decoder load.
    drop(dit_cell);
    drop(context);
    drop(pooled);
    drop(context_uncond);
    drop(pooled_uncond);
    println!("  DiT + embeddings evicted");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        MODEL_PATH,
        VAE_IN_CHANNELS,
        VAE_SCALE,
        VAE_SHIFT,
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
