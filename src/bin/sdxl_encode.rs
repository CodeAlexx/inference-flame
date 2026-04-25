//! SDXL prompt encoder — pure Rust replacement for `scripts/sdxl_encode.py`.
//!
//! Tokenizes a (positive, negative) prompt pair through the CLIP tokenizer
//! and runs both CLIP-L (768) and CLIP-G (1280) encoders. Concatenates
//! the per-layer hidden states along the channel dim into the SDXL
//! `context` tensor, and assembles the SDXL `y` pooled vector from
//! CLIP-L's pooler_output + CLIP-G's text_embeds (= text_projection ×
//! pooler_output) + 768 zeros (placeholder for the size_ids time
//! conditioning, matching the reference Python script).
//!
//! Output safetensors keys (BF16):
//!   - `context`         : [1, 77, 2048]
//!   - `context_uncond`  : [1, 77, 2048]
//!   - `y`               : [1, 2816]
//!   - `y_uncond`        : [1, 2816]
//!
//! Usage:
//!     sdxl_encode \
//!         --prompt "a photograph of an astronaut riding a horse" \
//!         --negative "" \
//!         --output /tmp/sdxl_embeds.safetensors
//!
//! Defaults expect the standalone CLIP-L/CLIP-G safetensors at
//! `/home/alex/.serenity/models/text_encoders/clip_{l,g}.safetensors`
//! and the matching `clip_{l,g}.tokenizer.json` files.

use flame_core::{global_cuda_device, DType, Result, Shape, Tensor};
use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    prompt: String,
    negative: String,
    output: PathBuf,
    clip_l: PathBuf,
    clip_g: PathBuf,
    tokenizer_l: PathBuf,
    tokenizer_g: PathBuf,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        prompt: String::new(),
        negative: String::new(),
        output: PathBuf::from("/tmp/sdxl_embeds.safetensors"),
        clip_l: PathBuf::from("/home/alex/.serenity/models/text_encoders/clip_l.safetensors"),
        clip_g: PathBuf::from("/home/alex/.serenity/models/text_encoders/clip_g.safetensors"),
        tokenizer_l: PathBuf::from(
            "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json",
        ),
        tokenizer_g: PathBuf::from(
            "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json",
        ),
    };
    let mut i = 1;
    while i < argv.len() {
        let take = |i: &mut usize| -> anyhow::Result<String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for {}", argv[*i - 1]))
        };
        match argv[i].as_str() {
            "--prompt" => a.prompt = take(&mut i)?,
            "--negative" => a.negative = take(&mut i)?,
            "--output" => a.output = PathBuf::from(take(&mut i)?),
            "--clip-l" => a.clip_l = PathBuf::from(take(&mut i)?),
            "--clip-g" => a.clip_g = PathBuf::from(take(&mut i)?),
            "--tokenizer-l" => a.tokenizer_l = PathBuf::from(take(&mut i)?),
            "--tokenizer-g" => a.tokenizer_g = PathBuf::from(take(&mut i)?),
            "-h" | "--help" => {
                eprintln!(
                    "sdxl_encode --prompt P [--negative N] [--output PATH] \
[--clip-l PATH] [--clip-g PATH] [--tokenizer-l PATH] [--tokenizer-g PATH]"
                );
                std::process::exit(0);
            }
            _ => anyhow::bail!("unknown arg: {}", argv[i]),
        }
        i += 1;
    }
    if a.prompt.is_empty() {
        anyhow::bail!("--prompt is required");
    }
    Ok(a)
}

const PAD_ID: i32 = 49407; // CLIP eos/pad
const MAX_LEN: usize = 77;

fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> anyhow::Result<Vec<i32>> {
    // CLIP tokenization wraps the input with BOS (49406) and EOS (49407);
    // tokenizers crate handles BOS/EOS via the model's special-tokens config.
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&id| id as i32).collect();
    if ids.len() > MAX_LEN {
        ids.truncate(MAX_LEN);
        // Make sure last token is EOS so the encoder finds a real eos for pooling.
        ids[MAX_LEN - 1] = PAD_ID;
    } else {
        ids.resize(MAX_LEN, PAD_ID);
    }
    Ok(ids)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args()?;
    let device = global_cuda_device();
    let t0 = Instant::now();

    eprintln!("=== SDXL Encode ===");
    eprintln!("prompt:   {:?}", args.prompt);
    eprintln!("negative: {:?}", args.negative);

    let tok_l = tokenizers::Tokenizer::from_file(&args.tokenizer_l)
        .map_err(|e| anyhow::anyhow!("load tok_l: {e}"))?;
    let tok_g = tokenizers::Tokenizer::from_file(&args.tokenizer_g)
        .map_err(|e| anyhow::anyhow!("load tok_g: {e}"))?;

    let pos_l_ids = tokenize(&tok_l, &args.prompt)?;
    let neg_l_ids = tokenize(&tok_l, &args.negative)?;
    let pos_g_ids = tokenize(&tok_g, &args.prompt)?;
    let neg_g_ids = tokenize(&tok_g, &args.negative)?;

    // CLIP-L
    eprintln!("\n--- CLIP-L ({}) ---", args.clip_l.display());
    let clip_l_w_raw = flame_core::serialization::load_file(&args.clip_l, &device)?;
    // Standalone clip_l/clip_g safetensors ship F16; ClipEncoder's kernels
    // (layer_norm, matmul) require BF16 so we cast everything once at load.
    let clip_l_w: HashMap<String, Tensor> = clip_l_w_raw
        .into_iter()
        .map(|(k, v)| {
            let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
            Ok((k, t))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let clip_l = ClipEncoder::new(clip_l_w, ClipConfig::default(), device.clone());
    let (pos_l_hidden, pos_l_pool) = clip_l.encode_sdxl(&pos_l_ids)?;
    let (neg_l_hidden, neg_l_pool) = clip_l.encode_sdxl(&neg_l_ids)?;
    eprintln!(
        "  pos hidden {:?}, pooled {:?}",
        pos_l_hidden.shape().dims(),
        pos_l_pool.shape().dims()
    );
    drop(clip_l);
    flame_core::trim_cuda_mempool(0);

    // CLIP-G
    eprintln!("\n--- CLIP-G ({}) ---", args.clip_g.display());
    let clip_g_w_raw = flame_core::serialization::load_file(&args.clip_g, &device)?;
    let mut clip_g_w: HashMap<String, Tensor> = clip_g_w_raw
        .into_iter()
        .map(|(k, v)| {
            let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
            Ok((k, t))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    // text_projection lives outside `text_model.*` so we extract it before
    // building the encoder (encoder strict-loads only `text_model.*` keys).
    let text_proj = clip_g_w
        .remove("text_projection.weight")
        .ok_or_else(|| anyhow::anyhow!("clip_g missing text_projection.weight"))?;
    let clip_g = ClipEncoder::new(clip_g_w, ClipConfig::clip_g(), device.clone());
    let (pos_g_hidden, pos_g_pool_raw) = clip_g.encode_sdxl(&pos_g_ids)?;
    let (neg_g_hidden, neg_g_pool_raw) = clip_g.encode_sdxl(&neg_g_ids)?;
    eprintln!(
        "  pos hidden {:?}, pooled raw {:?}",
        pos_g_hidden.shape().dims(),
        pos_g_pool_raw.shape().dims()
    );

    // CLIP-G text_embeds = pooler_output @ text_projection (HF convention:
    // text_projection.weight is [out, in], applied as Linear with no bias).
    // Result: [1, 1280].
    let proj_t = text_proj.transpose()?.contiguous()?;
    let pos_g_pool = pos_g_pool_raw.matmul(&proj_t)?;
    let neg_g_pool = neg_g_pool_raw.matmul(&proj_t)?;
    drop(clip_g);
    flame_core::trim_cuda_mempool(0);

    // SDXL `context`: cat(CLIP-L hidden [1,77,768], CLIP-G hidden [1,77,1280])
    //   along last dim → [1, 77, 2048].
    let context = Tensor::cat(&[&pos_l_hidden, &pos_g_hidden], 2)?;
    let context_uncond = Tensor::cat(&[&neg_l_hidden, &neg_g_hidden], 2)?;

    // SDXL `y`: cat(CLIP-L pool [1,768], CLIP-G text_embeds [1,1280],
    //   zeros [1,768]) → [1, 2816]. Last 768 is the size/crop time-id
    //   placeholder (matches `scripts/sdxl_encode.py`).
    let zeros_pad = Tensor::zeros_dtype(Shape::from_dims(&[1, 768]), DType::BF16, device.clone())?;
    let y = Tensor::cat(&[&pos_l_pool, &pos_g_pool, &zeros_pad], 1)?;
    let y_uncond = Tensor::cat(&[&neg_l_pool, &neg_g_pool, &zeros_pad], 1)?;

    eprintln!("\ncontext        : {:?}", context.shape().dims());
    eprintln!("context_uncond : {:?}", context_uncond.shape().dims());
    eprintln!("y              : {:?}", y.shape().dims());
    eprintln!("y_uncond       : {:?}", y_uncond.shape().dims());

    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("context".to_string(), context);
    out.insert("context_uncond".to_string(), context_uncond);
    out.insert("y".to_string(), y);
    out.insert("y_uncond".to_string(), y_uncond);

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    flame_core::serialization::save_tensors(
        &out,
        &args.output,
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    eprintln!(
        "\nSAVED: {} ({:.1}s)",
        args.output.display(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
