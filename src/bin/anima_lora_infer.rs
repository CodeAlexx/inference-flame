//! Anima (Cosmos Predict2) image generation with a runtime-applied LoRA —
//! end-to-end pure Rust. Mirrors `inference_ui/src/worker/anima.rs`.
//!
//! Two-stage encode (Qwen3 → final RMSNorm → mask → Anima LLM adapter →
//! mask), runtime LoRA via `Anima::set_lora`, rectified-flow Euler denoise,
//! Qwen-Image VAE decode → PNG.
//!
//! ## LoRA application
//!
//! Anima base weight keys live under `net.<sub>.weight`. External kohya
//! LoRAs trained against Anima strip that umbrella prefix from their
//! kohya keys (`lora_unet_blocks_{i}_<sub>` not
//! `lora_unet_net_blocks_*`). `inference-flame::lora` registers both
//! variants in its kohya naming table so these LoRAs route to the right
//! base keys without a model-specific mapper.
//!
//! ## Encoding sequence (matches worker/anima.rs::encode_one_prompt)
//!
//!   1. Tokenize prompt with Qwen3-0.6B tokenizer, pad to MAX_SEQ_LEN.
//!   2. Tokenize prompt with T5 tokenizer, pad to MAX_SEQ_LEN.
//!   3. Build Qwen3 + T5 attention masks (1.0 for real tokens, 0.0 for pad).
//!   4. Run Qwen3Encoder.encode → raw_hidden [1, S, 1024]. Encoder skips
//!      its final norm.
//!   5. Apply `model.norm` RMSNorm to raw_hidden — HF's `last_hidden_state`
//!      is post-norm and the adapter was trained against post-norm input.
//!   6. Mask normed_hidden by qwen3 attention mask (zero out pad positions).
//!   7. Build the T5 token-id tensor as F32 (so values up to 32128 are
//!      exact for `embedding_lookup`'s `to_vec().as usize` read).
//!   8. Anima.encode_context(t5_ids, masked_qwen3_hidden) → context.
//!   9. Mask context by T5 attention mask.
//!
//! ## Usage
//!
//!     anima_lora_infer \
//!         --prompt "..." [--negative "..."] \
//!         --lora /path/to/lora.safetensors \
//!         [--multiplier 1.0] \
//!         [--width 1024] [--height 1024] [--steps 30] [--cfg 4.5] [--seed 42]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, trim_cuda_mempool, DType, Result, Shape, Tensor};

use inference_flame::lora::LoraStack;
use inference_flame::models::anima::{load_all_weights, Anima};
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::vae::Wan21VaeDecoder;

const DEFAULT_BASE: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors";
const DEFAULT_QWEN3: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors";
const DEFAULT_VAE: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors";
const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/anima_lora.png";

/// Qwen3 tokenizer.json — fallback chain matches worker/anima.rs.
const QWEN3_TOKENIZER_PATHS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json",
    "/tmp/anima-trainer/configs/qwen3_06b/tokenizer.json",
];

/// T5 tokenizer.json — fallback chain. Worker hardcodes
/// `/tmp/anima-trainer/configs/t5_old/tokenizer.json`; the ComfyUI
/// `t5_tokenizer/` dir ships an equivalent with the same vocab + same
/// token IDs for ASCII text (verified empirically).
const T5_TOKENIZER_PATHS: &[&str] = &[
    "/tmp/anima-trainer/configs/t5_old/tokenizer.json",
    "/home/alex/SwarmUI/dlbackend/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json",
    "/home/alex/serenityflow/comfy/text_encoders/t5_tokenizer/tokenizer.json",
];

const MAX_SEQ_LEN: usize = 256;
const QWEN3_PAD_ID: i32 = 151643;
const T5_PAD_ID: i32 = 0;

struct Args {
    prompt: String,
    negative: String,
    base: PathBuf,
    qwen3: PathBuf,
    vae: PathBuf,
    qwen3_tokenizer: Option<PathBuf>,
    t5_tokenizer: Option<PathBuf>,
    lora: PathBuf,
    output: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    multiplier: f32,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        prompt: String::new(),
        negative: String::new(),
        base: PathBuf::from(DEFAULT_BASE),
        qwen3: PathBuf::from(DEFAULT_QWEN3),
        vae: PathBuf::from(DEFAULT_VAE),
        qwen3_tokenizer: None,
        t5_tokenizer: None,
        lora: PathBuf::new(),
        output: PathBuf::from(DEFAULT_OUTPUT),
        width: 1024,
        height: 1024,
        steps: 30,
        cfg: 4.5,
        seed: 42,
        multiplier: 1.0,
    };
    let mut i = 1;
    while i < argv.len() {
        let take = |i: &mut usize| -> anyhow::Result<String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "missing value for {}",
                    argv.get(*i - 1).cloned().unwrap_or_default()
                )
            })
        };
        match argv[i].as_str() {
            "--prompt" => a.prompt = take(&mut i)?,
            "--negative" => a.negative = take(&mut i)?,
            "--base" => a.base = PathBuf::from(take(&mut i)?),
            "--qwen3" => a.qwen3 = PathBuf::from(take(&mut i)?),
            "--vae" => a.vae = PathBuf::from(take(&mut i)?),
            "--qwen3-tokenizer" => a.qwen3_tokenizer = Some(PathBuf::from(take(&mut i)?)),
            "--t5-tokenizer" => a.t5_tokenizer = Some(PathBuf::from(take(&mut i)?)),
            "--lora" => a.lora = PathBuf::from(take(&mut i)?),
            "--output" | "-o" => a.output = PathBuf::from(take(&mut i)?),
            "--width" => a.width = take(&mut i)?.parse()?,
            "--height" => a.height = take(&mut i)?.parse()?,
            "--steps" => a.steps = take(&mut i)?.parse()?,
            "--cfg" | "--guidance" => a.cfg = take(&mut i)?.parse()?,
            "--seed" => a.seed = take(&mut i)?.parse()?,
            "--multiplier" | "--strength" => a.multiplier = take(&mut i)?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "anima_lora_infer --prompt P [--negative N] --lora LORA \
                     [--base PATH] [--qwen3 PATH] [--vae PATH] \
                     [--qwen3-tokenizer PATH] [--t5-tokenizer PATH] [--output PATH] \
                     [--width W] [--height H] [--steps N] [--cfg G] [--seed S] [--multiplier M]"
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
    if a.prompt.is_empty() {
        anyhow::bail!("--prompt is required");
    }
    Ok(a)
}

fn first_existing(paths: &[&str]) -> Option<PathBuf> {
    paths
        .iter()
        .copied()
        .find(|p| Path::new(p).exists())
        .map(PathBuf::from)
}

fn pad_tokenize(
    tk: &tokenizers::Tokenizer,
    text: &str,
    max_len: usize,
    pad_id: i32,
) -> anyhow::Result<Vec<i32>> {
    let enc = tk
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut out: Vec<i32> = enc.get_ids().iter().take(max_len).map(|&i| i as i32).collect();
    while out.len() < max_len {
        out.push(pad_id);
    }
    Ok(out)
}

/// Apply RMSNorm along the last (hidden) dim using flame_core's BF16 kernel —
/// the `model.norm` step that `Qwen3Encoder::encode` deliberately skips.
fn rms_norm_last_dim(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
    out_2d.reshape(&dims)
}

/// Multiply each row in the seq dim of `x` (`[1, S, D]`) by the per-position
/// scalar in `mask` (`[S]`). Used to zero out padding positions.
fn mask_seq_dim(x: &Tensor, mask: &[f32], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(flame_core::Error::InvalidInput(format!(
            "mask_seq_dim expected [B, S, D], got {dims:?}"
        )));
    }
    let s = dims[1];
    let mask_t = Tensor::from_f32_to_bf16(
        mask.to_vec(),
        Shape::from_dims(&[1, s, 1]),
        device.clone(),
    )?;
    x.mul(&mask_t)
}

fn encode_one_prompt(
    model: &Anima,
    encoder: &Qwen3Encoder,
    final_norm_w: &Tensor,
    qwen3_tk: &tokenizers::Tokenizer,
    t5_tk: &tokenizers::Tokenizer,
    device: &Arc<CudaDevice>,
    prompt: &str,
) -> anyhow::Result<Tensor> {
    let qwen3_ids = pad_tokenize(qwen3_tk, prompt, MAX_SEQ_LEN, QWEN3_PAD_ID)?;
    let t5_ids = pad_tokenize(t5_tk, prompt, MAX_SEQ_LEN, T5_PAD_ID)?;
    let qwen3_mask: Vec<f32> = qwen3_ids
        .iter()
        .map(|&id| if id == QWEN3_PAD_ID { 0.0 } else { 1.0 })
        .collect();
    let t5_mask: Vec<f32> = t5_ids
        .iter()
        .map(|&id| if id == T5_PAD_ID { 0.0 } else { 1.0 })
        .collect();

    let raw_hidden = encoder.encode(&qwen3_ids)?;
    let normed_hidden = rms_norm_last_dim(&raw_hidden, final_norm_w, 1e-6)?;
    let qwen3_hidden = mask_seq_dim(&normed_hidden, &qwen3_mask, device)?;

    let t5_id_f32: Vec<f32> = t5_ids.iter().map(|&i| i as f32).collect();
    let t5_id_tensor =
        Tensor::from_vec(t5_id_f32, Shape::from_dims(&[1, MAX_SEQ_LEN]), device.clone())?;

    let context = model.encode_context(&t5_id_tensor, &qwen3_hidden)?;
    let masked = mask_seq_dim(&context, &t5_mask, device)?;
    Ok(masked)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let args = parse_args()?;
    let device = global_cuda_device();

    println!("============================================================");
    println!("Anima LoRA Inference (pure Rust, end-to-end)");
    println!("  {}x{}  {} steps  cfg {}  seed {}  mult {:.3}",
        args.width, args.height, args.steps, args.cfg, args.seed, args.multiplier);
    println!("  prompt:   {}", args.prompt);
    println!("  negative: {}", if args.negative.is_empty() { "(empty)" } else { &args.negative });
    println!("============================================================");

    // Resolve tokenizer paths.
    let qwen3_tok_path: PathBuf = args
        .qwen3_tokenizer
        .clone()
        .or_else(|| first_existing(QWEN3_TOKENIZER_PATHS))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Qwen3 tokenizer not found. Tried:\n  - {}",
                QWEN3_TOKENIZER_PATHS.join("\n  - ")
            )
        })?;
    let t5_tok_path: PathBuf = args
        .t5_tokenizer
        .clone()
        .or_else(|| first_existing(T5_TOKENIZER_PATHS))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "T5 tokenizer not found. Tried:\n  - {}",
                T5_TOKENIZER_PATHS.join("\n  - ")
            )
        })?;
    println!("  qwen3 tokenizer: {}", qwen3_tok_path.display());
    println!("  t5 tokenizer:    {}", t5_tok_path.display());

    // ------------------------------------------------------------------
    // Stage A: Load Anima DiT all-on-GPU. Adapter weights live here, so
    // load the model BEFORE encoding. ~3.9 GB.
    // ------------------------------------------------------------------
    println!("\n--- Stage A: Load Anima DiT ---");
    let t0 = Instant::now();
    let base_path_str = args.base.to_str().ok_or_else(|| anyhow::anyhow!("base path utf8"))?;
    let all_weights = load_all_weights(base_path_str, &device)?;
    println!("  {} base weight tensors", all_weights.len());
    let base_keys: HashSet<String> = all_weights.keys().cloned().collect();
    let mut model = Anima::new_all_on_gpu(base_path_str.to_string(), all_weights, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage B: Load Qwen3-0.6B encoder + tokenizers, encode both prompts,
    // drop encoder. Mirrors worker/anima.rs::encode_prompts.
    // ------------------------------------------------------------------
    println!("\n--- Stage B: Encode prompts ---");
    let t0 = Instant::now();
    let qwen3_tk = tokenizers::Tokenizer::from_file(&qwen3_tok_path)
        .map_err(|e| anyhow::anyhow!("qwen3 tokenizer load: {e}"))?;
    let t5_tk = tokenizers::Tokenizer::from_file(&t5_tok_path)
        .map_err(|e| anyhow::anyhow!("t5 tokenizer load: {e}"))?;

    let mut weights = flame_core::serialization::load_file(&args.qwen3, &device)?;
    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in keys {
        let t = &weights[&k];
        if t.dtype() != DType::BF16 {
            let bf = t.to_dtype(DType::BF16)?;
            weights.insert(k, bf);
        }
    }
    let mut qcfg = Qwen3Encoder::config_from_weights(&weights)?;
    let last_layer = qcfg.num_layers.checked_sub(1).ok_or_else(|| {
        anyhow::anyhow!("Qwen3 num_layers={} expected >=1", qcfg.num_layers)
    })?;
    qcfg.extract_layers = vec![last_layer];
    println!(
        "  Qwen3 config: layers={} hidden={} extract_layer={}",
        qcfg.num_layers, qcfg.hidden_size, last_layer
    );

    // Pull final norm weight before the encoder takes ownership.
    let final_norm_w = weights
        .get("model.norm.weight")
        .ok_or_else(|| anyhow::anyhow!("missing model.norm.weight in Qwen3 encoder"))?
        .clone();

    let encoder = Qwen3Encoder::new(weights, qcfg, device.clone());

    let context_cond =
        encode_one_prompt(&model, &encoder, &final_norm_w, &qwen3_tk, &t5_tk, &device, &args.prompt)?;
    let context_uncond =
        encode_one_prompt(&model, &encoder, &final_norm_w, &qwen3_tk, &t5_tk, &device, &args.negative)?;
    drop(encoder);
    drop(final_norm_w);
    drop(qwen3_tk);
    drop(t5_tk);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    trim_cuda_mempool(0);
    println!(
        "  context_cond {:?}  context_uncond {:?}  ({:.1}s)",
        context_cond.shape().dims(),
        context_uncond.shape().dims(),
        t0.elapsed().as_secs_f32()
    );

    // ------------------------------------------------------------------
    // Stage C: LoRA stack → set_lora.
    // ------------------------------------------------------------------
    println!("\n--- Stage C: Loading LoRA ---");
    let t0 = Instant::now();
    let lora_path_str = args.lora.to_str().ok_or_else(|| anyhow::anyhow!("lora path utf8"))?;
    let stack = LoraStack::load(lora_path_str, &base_keys, args.multiplier, &device)?;
    println!(
        "  loaded {} target weight(s) in {:.1}s",
        stack.target_count(),
        t0.elapsed().as_secs_f32()
    );
    model.set_lora(Arc::new(stack));

    // ------------------------------------------------------------------
    // Stage D: Build noise + denoise (rectified flow, linear sigmas).
    // ------------------------------------------------------------------
    let latent_h = args.height / 8;
    let latent_w = args.width / 8;
    let t_frames: usize = 1;
    let mut x = Tensor::randn_seeded(
        Shape::from_dims(&[1, t_frames, latent_h, latent_w, 16]),
        0.0,
        1.0,
        args.seed,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let sigmas: Vec<f32> = (0..=args.steps)
        .map(|i| 1.0 - i as f32 / args.steps as f32)
        .collect();

    println!("\n--- Stage D: Denoise ({} steps, cfg={}) ---", args.steps, args.cfg);
    println!("  Latent: [1, {}, {}, {}, 16]", t_frames, latent_h, latent_w);

    let t0 = Instant::now();
    for i in 0..args.steps {
        let step_t = Instant::now();
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let t_vec =
            Tensor::from_f32_to_bf16(vec![sigma], Shape::from_dims(&[1]), device.clone())?;

        let pred_cond = model.forward_with_context(&x, &t_vec, &context_cond)?;
        let pred_uncond = model.forward_with_context(&x, &t_vec, &context_uncond)?;
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(args.cfg)?)?;
        let dt = sigma_next - sigma;
        x = x.add(&pred.mul_scalar(dt)?)?;

        let step_ms = step_t.elapsed().as_millis();
        if i == 0 || (i + 1) % 5 == 0 || i + 1 == args.steps {
            println!(
                "  step {}/{}  sigma={:.4}  ({}ms)",
                i + 1, args.steps, sigma, step_ms
            );
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / args.steps as f32);

    // Drop DiT before VAE for cuDNN-workspace headroom (matches worker).
    drop(model);
    drop(context_cond);
    drop(context_uncond);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    trim_cuda_mempool(0);

    // ------------------------------------------------------------------
    // Stage E: VAE decode (Qwen-Image VAE) — pure Rust, no Python.
    // ------------------------------------------------------------------
    println!("\n--- Stage E: VAE decode ---");
    let t_vae = Instant::now();
    let latent = x.permute(&[0, 4, 1, 2, 3])?.contiguous()?;
    drop(x);
    println!("  VAE input latent: {:?}", latent.shape().dims());
    let vae_path_str = args.vae.to_str().ok_or_else(|| anyhow::anyhow!("vae path utf8"))?;
    let decoder = Wan21VaeDecoder::load(vae_path_str, &device)?;
    // decode_image: single-frame (T=1) path — skips temporal upsample doubling.
    let rgb_5d = decoder.decode_image(&latent)?;
    drop(latent);
    drop(decoder);
    println!(
        "  decoded {:?} in {:.1}s",
        rgb_5d.shape().dims(),
        t_vae.elapsed().as_secs_f32()
    );

    // [1, 3, 1, H, W] -> [1, 3, H, W]
    let rgb = rgb_5d.narrow(2, 0, 1)?.squeeze(Some(2))?;
    let rgb_f32 = rgb.to_dtype(DType::F32)?.to_vec()?;
    let dims = rgb.shape().dims().to_vec();
    let (h_img, w_img) = (dims[2], dims[3]);
    let mut buf = vec![0u8; h_img * w_img * 3];
    for y in 0..h_img {
        for xp in 0..w_img {
            for ch in 0..3 {
                let v = rgb_f32[ch * h_img * w_img + y * w_img + xp];
                let v = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).clamp(0.0, 255.0);
                buf[(y * w_img + xp) * 3 + ch] = v as u8;
            }
        }
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::save_buffer(&args.output, &buf, w_img as u32, h_img as u32, image::ColorType::Rgb8)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", args.output.display());
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    Ok(())
}
