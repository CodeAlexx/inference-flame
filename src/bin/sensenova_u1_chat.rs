//! SenseNova-U1-8B-MoT — pure-Rust visual QA / chat CLI.
//!
//! Reference: `modeling_neo_chat.py::chat` (line 1732). The pipeline is
//!   1. Build chat-template prompt with `<image>\n{question}`.
//!   2. Replace `<image>` with `<img><IMG_CONTEXT>×L</img>` (L = post-merge
//!      patch count for the input image).
//!   3. Tokenize.
//!   4. Run `extract_feature_und` on the patchified pixels → image features.
//!   5. Splice features in place of the `<IMG_CONTEXT>` slots.
//!   6. Build per-token (t, h, w) RoPE indices via `get_thw_indexes`.
//!   7. Mixed-mode prefix forward → KvCache + last hidden.
//!   8. Greedy autoregressive decode until `<|im_end|>` or max_new_tokens.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use image::imageops::FilterType;
use image::GenericImageView;
use inference_flame::models::sensenova_u1::SenseNovaU1;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

const SYSTEM_MESSAGE_DEFAULT: &str = ""; // chat template default; matches Python.
const IMG_START: &str = "<img>";
const IMG_END: &str = "</img>";
const IMG_CONTEXT: &str = "<IMG_CONTEXT>";

const IMG_START_ID: i32 = 151670;
const IMG_CONTEXT_ID: i32 = 151669;
const IM_END_ID: i32 = 151645;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug)]
struct Args {
    weights_dir: PathBuf,
    image: PathBuf,
    question: String,
    max_new_tokens: usize,
    min_pixels: u32,
    max_pixels: u32,
}

impl Args {
    fn defaults() -> Self {
        Self {
            weights_dir: PathBuf::from("/home/alex/.serenity/models/sensenova_u1"),
            image: PathBuf::new(),
            question: String::new(),
            max_new_tokens: 256,
            // Keep input image at <=1024² by default — VQA is much faster
            // when the prefix is short. 512² is the lower bound from the
            // Python reference (min_pixels=512*512).
            min_pixels: 512 * 512,
            max_pixels: 1024 * 1024,
        }
    }
}

fn parse_args() -> std::result::Result<Args, String> {
    let mut a = Args::defaults();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        let mut next = || -> std::result::Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--weights" | "--model_path" => a.weights_dir = PathBuf::from(next()?),
            "--image" => a.image = PathBuf::from(next()?),
            "--question" | "--prompt" => a.question = next()?,
            "--max_new_tokens" => {
                a.max_new_tokens = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--min_pixels" => {
                a.min_pixels = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--max_pixels" => {
                a.max_pixels = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "-h" | "--help" => {
                eprintln!(
                    "sensenova_u1_chat — VQA / image-conditioned chat\n\
                    Usage: sensenova_u1_chat --image PATH --question STR [options]\n\
                    Options:\n\
                      --weights DIR    weights dir [default /home/alex/.serenity/models/sensenova_u1]\n\
                      --max_new_tokens N  decode budget [default 256]\n\
                      --min_pixels N   smart-resize floor [default 262144]\n\
                      --max_pixels N   smart-resize ceiling [default 1048576]\n"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if a.image.as_os_str().is_empty() {
        return Err("missing --image".into());
    }
    if a.question.is_empty() {
        return Err("missing --question".into());
    }
    Ok(a)
}

// ---------------------------------------------------------------------------
// Tokenizer (Qwen3 ByteLevel-BPE built from vocab.json + merges.txt + added_tokens.json)
// ---------------------------------------------------------------------------

fn build_tokenizer(weights_dir: &Path) -> Result<Tokenizer> {
    let vocab = weights_dir.join("vocab.json");
    let merges = weights_dir.join("merges.txt");
    let added = weights_dir.join("added_tokens.json");
    let bpe = BPE::from_file(
        vocab.to_str().context("vocab path not utf-8")?,
        merges.to_str().context("merges path not utf-8")?,
    )
    .build()
    .map_err(|e| anyhow!("BPE::build failed: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevel::default()));
    let raw = std::fs::read_to_string(&added)
        .with_context(|| format!("read {}", added.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).context("added_tokens.json")?;
    let mut entries: Vec<(String, u64)> = map
        .into_iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    entries.sort_by_key(|(_, id)| *id);
    let base_size = tok.get_vocab_size(false) as u64;
    if let Some((_, first_id)) = entries.first() {
        if *first_id != base_size {
            return Err(anyhow!(
                "added_tokens.json starts at id {first_id} but base vocab size is {base_size}"
            ));
        }
    }
    let added_tokens: Vec<AddedToken> = entries
        .into_iter()
        .map(|(content, _)| AddedToken::from(content, true))
        .collect();
    tok.add_special_tokens(&added_tokens);
    Ok(tok)
}

fn encode_query(tok: &Tokenizer, query: &str) -> Result<Vec<i32>> {
    let enc = tok
        .encode(query, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

fn decode_ids(tok: &Tokenizer, ids: &[i32]) -> Result<String> {
    let u32s: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    tok.decode(&u32s, true).map_err(|e| anyhow!("decode: {e}"))
}

// ---------------------------------------------------------------------------
// Chat template: matches `_build_t2i_query` style for the und path. The
// SYSTEM_MESSAGE_DEFAULT is empty (no system block) because the chat template
// in the Python reference uses an empty default for VQA; T2I/it2i use
// SYSTEM_MESSAGE_FOR_GEN. For single-turn VQA we follow the empty-system path.
// ---------------------------------------------------------------------------

fn build_chat_query(system: &str, user: &str) -> String {
    let mut q = String::new();
    if !system.is_empty() {
        q.push_str("<|im_start|>system\n");
        q.push_str(system);
        q.push_str("<|im_end|>\n");
    }
    q.push_str("<|im_start|>user\n");
    q.push_str(user);
    q.push_str("<|im_end|>\n");
    q.push_str("<|im_start|>assistant\n");
    q
}

// ---------------------------------------------------------------------------
// Image preprocessing (mirror of utils.py::load_image_native).
// ---------------------------------------------------------------------------

/// Smart-resize: round each side to a multiple of `factor` (== patch_size /
/// downsample_ratio = 32) such that total pixels ∈ [min_pixels, max_pixels].
fn smart_resize(h: u32, w: u32, factor: u32, min_pixels: u32, max_pixels: u32) -> (u32, u32) {
    fn round_by(v: f32, f: u32) -> u32 {
        ((v / f as f32).round() as u32).max(1) * f
    }
    fn ceil_by(v: f32, f: u32) -> u32 {
        ((v / f as f32).ceil() as u32).max(1) * f
    }
    fn floor_by(v: f32, f: u32) -> u32 {
        ((v / f as f32).floor() as u32).max(1) * f
    }
    let h_bar = factor.max(round_by(h as f32, factor));
    let w_bar = factor.max(round_by(w as f32, factor));
    if h_bar * w_bar > max_pixels {
        let beta = ((h as f32 * w as f32) / max_pixels as f32).sqrt();
        let h2 = factor.max(floor_by(h as f32 / beta, factor));
        let w2 = factor.max(floor_by(w as f32 / beta, factor));
        (h2, w2)
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (h as f32 * w as f32)).sqrt();
        let h2 = ceil_by(h as f32 * beta, factor);
        let w2 = ceil_by(w as f32 * beta, factor);
        (h2, w2)
    } else {
        (h_bar, w_bar)
    }
}

/// Load + smart-resize + ImageNet-normalize + patchify into `[grid_h*grid_w, 3*16*16]`
/// BF16 device tensor. Returns the tensor plus `(grid_h, grid_w)` post-merge token grid.
fn load_image_native(
    path: &Path,
    patch_size: usize,
    downsample_ratio: f32,
    min_pixels: u32,
    max_pixels: u32,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, usize, usize)> {
    let img = image::open(path).with_context(|| format!("open {}", path.display()))?;
    // RGBA → RGB on white background; otherwise straight to RGB.
    let img = match img.color() {
        image::ColorType::Rgba8 | image::ColorType::Rgba16 => {
            let (w, h) = img.dimensions();
            let mut bg = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
            let rgba = img.to_rgba8();
            for (x, y, p) in rgba.enumerate_pixels() {
                let alpha = p.0[3] as f32 / 255.0;
                let r = (p.0[0] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                let g = (p.0[1] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                let b = (p.0[2] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                bg.put_pixel(x, y, image::Rgb([r, g, b]));
            }
            image::DynamicImage::ImageRgb8(bg)
        }
        _ => image::DynamicImage::ImageRgb8(img.to_rgb8()),
    };

    let factor = (patch_size as f32 / downsample_ratio).round() as u32; // 32
    let (orig_w, orig_h) = img.dimensions();
    let (new_h, new_w) = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels);
    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);
    let resized = resized.to_rgb8();
    let h = new_h as usize;
    let w = new_w as usize;
    if h % patch_size != 0 || w % patch_size != 0 {
        return Err(anyhow!(
            "smart_resize produced {h}x{w} not divisible by patch_size {patch_size}"
        ));
    }
    let grid_h = h / patch_size;
    let grid_w = w / patch_size;

    // Build CHW f32 normalized tensor on host.
    let n = h * w;
    let mut chw = vec![0f32; 3 * n];
    for (i, p) in resized.pixels().enumerate() {
        let r = p.0[0] as f32 / 255.0;
        let g = p.0[1] as f32 / 255.0;
        let b = p.0[2] as f32 / 255.0;
        chw[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        chw[n + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        chw[2 * n + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    // Patchify on host: [C=3, grid_h, ps, grid_w, ps] → permute [grid_h, grid_w, C, ps, ps] → reshape [grid_h*grid_w, C*ps*ps].
    let p = patch_size;
    let cps = 3 * p * p;
    let mut flat = vec![0f32; grid_h * grid_w * cps];
    for gh in 0..grid_h {
        for gw in 0..grid_w {
            for c in 0..3 {
                for py in 0..p {
                    for px in 0..p {
                        let src_y = gh * p + py;
                        let src_x = gw * p + px;
                        let src = c * n + src_y * w + src_x;
                        let dst = ((gh * grid_w + gw) * cps)
                            + (c * p * p)
                            + (py * p)
                            + px;
                        flat[dst] = chw[src];
                    }
                }
            }
        }
    }
    let t = Tensor::from_vec(
        flat,
        Shape::from_dims(&[grid_h * grid_w, cps]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    Ok((t, grid_h, grid_w))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    env_logger::init();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("[sensenova_u1_chat] error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> Result<()> {
    let device = flame_core::global_cuda_device();

    eprintln!("[sensenova_u1_chat] loading tokenizer + weights from {:?}", args.weights_dir);
    let tok = build_tokenizer(&args.weights_dir)?;
    let mut model = SenseNovaU1::load(&args.weights_dir, &device)?;
    let cfg = model.config().clone();

    // 1) Load + preprocess image. `pixels` is [grid_h*grid_w, 3*ps*ps] BF16.
    let (pixels, grid_h, grid_w) = load_image_native(
        &args.image,
        cfg.patch_size,
        cfg.downsample_ratio,
        args.min_pixels,
        args.max_pixels,
        &device,
    )?;
    let merge = cfg.merge_size();
    if grid_h % merge != 0 || grid_w % merge != 0 {
        return Err(anyhow!(
            "grid {grid_h}x{grid_w} not divisible by merge_size {merge}"
        ));
    }
    let token_h = grid_h / merge;
    let token_w = grid_w / merge;
    let l = token_h * token_w;
    eprintln!(
        "[sensenova_u1_chat] image: pixel_grid={}x{}, token_grid={}x{} (L={})",
        grid_h, grid_w, token_h, token_w, l
    );

    // 2) Build query, splice <image> → <img><IMG_CONTEXT>×L</img>.
    let user_msg = if args.question.contains("<image>") {
        args.question.clone()
    } else {
        format!("<image>\n{}", args.question)
    };
    let raw_query = build_chat_query(SYSTEM_MESSAGE_DEFAULT, &user_msg);
    let img_token_block = format!(
        "{}{}{}",
        IMG_START,
        IMG_CONTEXT.repeat(l),
        IMG_END
    );
    let query = raw_query.replacen("<image>", &img_token_block, 1);
    let input_ids = encode_query(&tok, &query)?;
    eprintln!("[sensenova_u1_chat] tokenized prefix length: {}", input_ids.len());

    // Spot-check expected token IDs landed correctly.
    let n_imgctx = input_ids.iter().filter(|&&id| id == IMG_CONTEXT_ID).count();
    if n_imgctx != l {
        return Err(anyhow!(
            "expected {l} <IMG_CONTEXT> tokens after splice, got {n_imgctx} — \
             tokenizer likely splitting/merging differently than expected"
        ));
    }

    // 3) Extract understanding-side image features → [1, L, 4096].
    let t0 = std::time::Instant::now();
    let img_feats = model.extract_feature_und(&pixels, grid_h, grid_w)?;
    eprintln!(
        "[sensenova_u1_chat] extract_feature_und: {:.2}s, shape {:?}",
        t0.elapsed().as_secs_f32(),
        img_feats.shape().dims()
    );

    // 4) Embed input_ids and splice features at IMG_CONTEXT positions.
    let hidden_in = model.embed_with_image_splice(&input_ids, IMG_CONTEXT_ID, &[img_feats])?;

    // 5) Build per-token (t, h, w) RoPE indices + image_mask.
    let (t_idx, h_idx, w_idx, image_mask) =
        model.build_thw_indexes(&input_ids, IMG_CONTEXT_ID, IMG_START_ID, &[(token_h, token_w)])?;

    // 6) Mixed-mode prefix forward.
    let t1 = std::time::Instant::now();
    let (mut cache, last_hidden) = model.forward_mixed_prefix(
        &hidden_in,
        &image_mask,
        &t_idx,
        &h_idx,
        &w_idx,
    )?;
    eprintln!(
        "[sensenova_u1_chat] mixed-mode prefix: {:.2}s (prefix_len={}, next_t={})",
        t1.elapsed().as_secs_f32(),
        cache.prefix_len,
        cache.next_t_index
    );

    // 7) Greedy decode until EOS.
    let t2 = std::time::Instant::now();
    let out_ids = model.decode_autoregressive(
        &mut cache,
        &last_hidden,
        args.max_new_tokens,
        &[IM_END_ID],
    )?;
    eprintln!(
        "[sensenova_u1_chat] decode: {:.2}s ({} tokens)",
        t2.elapsed().as_secs_f32(),
        out_ids.len()
    );

    // Drop the trailing EOS for cleaner output.
    let out_ids: Vec<i32> = if out_ids.last() == Some(&IM_END_ID) {
        out_ids[..out_ids.len() - 1].to_vec()
    } else {
        out_ids
    };
    let answer = decode_ids(&tok, &out_ids)?;
    println!("{}", answer.trim());
    Ok(())
}
