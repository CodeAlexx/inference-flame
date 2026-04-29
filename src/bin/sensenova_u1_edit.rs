//! SenseNova-U1-8B-MoT — pure-Rust image-editing (it2i) CLI.
//!
//! Reference: `modeling_neo_chat.py::it2i_generate` (line 1297). Pipeline:
//!   1. Build cond query: `<image>\n{prompt}` user message + system prompt +
//!      assistant prefix + `<think>\n\n</think>\n\n<img>` (or `<think>\n` if
//!      `--think`). Splice `<image>` → `<img><IMG_CONTEXT>×L_in</img>`.
//!   2. Build uncond query: empty user, append `<img>` (matches T2I uncond).
//!   3. Tokenize both. extract_feature_und on input image → image features.
//!   4. Splice image features into cond input embeddings via embed_with_image_splice.
//!   5. forward_mixed_prefix(cond_embeds, mask, t/h/w idx) → cond_cache.
//!   6. forward_und(uncond_ids) → uncond_cache.
//!   7. Optional --think on cond stream (decode_autoregressive + extend).
//!   8. T2I gen loop with single-CFG combine: v = v_uncond + cfg*(v_cond - v_uncond).
//!   9. Save PNG.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use image::imageops::FilterType;
use image::GenericImageView;
use inference_flame::models::sensenova_u1::{KvCache, SenseNovaU1, TimeOrScale};
use rand::{Rng, SeedableRng};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands and executes ",
    "user intent.\n\nYou support two modes:\n\n",
    "1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. ",
    "Put all reasoning inside the block using plain text. DO NOT include any image tags. ",
    "Keep it reasonable and directly useful for producing the final image.\n\n",
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n",
    "Task Types:\n\nA. Text-to-Image Generation:\n",
    "- Generate a high-quality image based on the user's description.\n",
    "- Ensure visual clarity, semantic consistency, and completeness.\n",
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n",
    "B. Image Editing:\n",
    "- Use the provided image(s) as input or reference for modification or transformation.\n",
    "- The result can be an edited image or a new image based on the reference(s).\n",
    "- Preserve all unspecified attributes unless explicitly changed.\n\n",
    "General Rules:\n",
    "- For any visible text in the image, follow the language specified for the rendered text in ",
    "the user's description, not the language of the prompt. If no language is specified, use the ",
    "user's input language."
);

const IMG_START: &str = "<img>";
const IMG_END: &str = "</img>";
const IMG_CONTEXT: &str = "<IMG_CONTEXT>";

const IMG_START_ID: i32 = 151670;
const IMG_CONTEXT_ID: i32 = 151669;
const THINK_END_ID: i32 = 151668;
const IM_END_ID: i32 = 151645;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug)]
struct Args {
    weights_dir: PathBuf,
    input_image: PathBuf,
    prompt: String,
    output: PathBuf,
    width: u32,
    height: u32,
    cfg_scale: f32,
    timestep_shift: f32,
    num_steps: u32,
    seed: u64,
    in_min_pixels: u32,
    in_max_pixels: u32,
    think_mode: bool,
    max_think_tokens: usize,
}

impl Args {
    fn defaults() -> Self {
        Self {
            weights_dir: PathBuf::from("/home/alex/.serenity/models/sensenova_u1"),
            input_image: PathBuf::new(),
            prompt: String::new(),
            output: PathBuf::from("inference-flame/output/sensenova_u1/edit.png"),
            width: 1024,
            height: 1024,
            cfg_scale: 4.0,
            timestep_shift: 3.0,
            num_steps: 30,
            seed: 42,
            in_min_pixels: 512 * 512,
            in_max_pixels: 1024 * 1024,
            think_mode: false,
            max_think_tokens: 1024,
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
            "--input_image" | "--input" => a.input_image = PathBuf::from(next()?),
            "--prompt" => a.prompt = next()?,
            "--output" | "-o" => a.output = PathBuf::from(next()?),
            "--width" => a.width = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--height" => a.height = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--cfg_scale" | "--cfg" => a.cfg_scale = next()?.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?,
            "--timestep_shift" | "--shift" => a.timestep_shift = next()?.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?,
            "--num_steps" | "--steps" => a.num_steps = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--seed" => a.seed = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--in_min_pixels" => a.in_min_pixels = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--in_max_pixels" => a.in_max_pixels = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--think" => a.think_mode = true,
            "--max_think_tokens" => a.max_think_tokens = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "-h" | "--help" => {
                eprintln!(
                    "sensenova_u1_edit — image editing (it2i)\n\
                    Usage: sensenova_u1_edit --input_image PATH --prompt STR [options]\n"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if a.input_image.as_os_str().is_empty() {
        return Err("missing --input_image".into());
    }
    if a.prompt.is_empty() {
        return Err("missing --prompt".into());
    }
    Ok(a)
}

// ---------------------------------------------------------------------------
// Tokenizer
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
    let added_tokens: Vec<AddedToken> = entries
        .into_iter()
        .map(|(content, _)| AddedToken::from(content, true))
        .collect();
    tok.add_special_tokens(&added_tokens);
    Ok(tok)
}

fn encode_query(tok: &Tokenizer, query: &str) -> Result<Vec<i32>> {
    let enc = tok.encode(query, false).map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

fn build_query(system: &str, user: &str, append: &str) -> String {
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
    q.push_str(append);
    q
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

fn smart_resize(h: u32, w: u32, factor: u32, min_pixels: u32, max_pixels: u32) -> (u32, u32) {
    fn round_by(v: f32, f: u32) -> u32 { ((v / f as f32).round() as u32).max(1) * f }
    fn ceil_by(v: f32, f: u32) -> u32 { ((v / f as f32).ceil() as u32).max(1) * f }
    fn floor_by(v: f32, f: u32) -> u32 { ((v / f as f32).floor() as u32).max(1) * f }
    let h_bar = factor.max(round_by(h as f32, factor));
    let w_bar = factor.max(round_by(w as f32, factor));
    if h_bar * w_bar > max_pixels {
        let beta = ((h as f32 * w as f32) / max_pixels as f32).sqrt();
        (factor.max(floor_by(h as f32 / beta, factor)),
         factor.max(floor_by(w as f32 / beta, factor)))
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (h as f32 * w as f32)).sqrt();
        (ceil_by(h as f32 * beta, factor), ceil_by(w as f32 * beta, factor))
    } else {
        (h_bar, w_bar)
    }
}

fn load_image_native(
    path: &Path,
    patch_size: usize,
    downsample_ratio: f32,
    min_pixels: u32,
    max_pixels: u32,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, usize, usize)> {
    let img = image::open(path).with_context(|| format!("open {}", path.display()))?;
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
    let factor = (patch_size as f32 / downsample_ratio).round() as u32;
    let (orig_w, orig_h) = img.dimensions();
    let (new_h, new_w) = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels);
    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle).to_rgb8();
    let h = new_h as usize;
    let w = new_w as usize;
    let grid_h = h / patch_size;
    let grid_w = w / patch_size;
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
                        let dst = ((gh * grid_w + gw) * cps) + (c * p * p) + (py * p) + px;
                        flat[dst] = chw[src];
                    }
                }
            }
        }
    }
    let t = Tensor::from_vec(flat, Shape::from_dims(&[grid_h * grid_w, cps]), device.clone())?
        .to_dtype(DType::BF16)?;
    Ok((t, grid_h, grid_w))
}

// ---------------------------------------------------------------------------
// Output-image helpers (shared with sensenova_u1_gen).
// ---------------------------------------------------------------------------

fn make_noise_image(seed: u64, shape: &[usize], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let t = Tensor::from_vec(data, Shape::from_dims(shape), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

fn patchify(images: &Tensor, p: usize, channel_first: bool) -> Result<Tensor> {
    let dims = images.shape().dims();
    let (b, h, w) = (dims[0], dims[2], dims[3]);
    let gh = h / p;
    let gw = w / p;
    let x = images.reshape(&[b, 3, gh, p, gw, p])?;
    let x = if channel_first {
        x.permute(&[0, 2, 4, 1, 3, 5])?
    } else {
        x.permute(&[0, 2, 4, 3, 5, 1])?
    };
    Ok(x.reshape(&[b, gh * gw, p * p * 3])?)
}

fn unpatchify(x: &Tensor, p: usize, h: usize, w: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    let b = dims[0];
    let gh = h / p;
    let gw = w / p;
    let x = x.reshape(&[b, gh, gw, p, p, 3])?;
    let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
    Ok(x.reshape(&[b, 3, gh * p, gw * p])?)
}

fn save_png(image: &Tensor, path: &Path) -> Result<()> {
    let img_f32 = image.to_dtype(DType::F32)?;
    let dims = img_f32.shape().dims();
    let (h, w) = (dims[2], dims[3]);
    let data = img_f32.to_vec_f32()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = data[c * h * w + y * w + x];
                pixels[(y * w + x) * 3 + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let buf = image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("PNG buffer build failed"))?;
    buf.save(path).with_context(|| format!("save {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    env_logger::init();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::from(2); }
    };
    if let Err(e) = run(&args) {
        eprintln!("[sensenova_u1_edit] error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> Result<()> {
    let device = flame_core::global_cuda_device();

    eprintln!("[sensenova_u1_edit] loading tokenizer + weights from {:?}", args.weights_dir);
    let tok = build_tokenizer(&args.weights_dir)?;
    let mut model = SenseNovaU1::load(&args.weights_dir, &device)?;
    let cfg = model.config().clone();

    let merge = cfg.merge_size();
    let patch = cfg.patch_size;
    if (args.width as usize) % (patch * merge) != 0
        || (args.height as usize) % (patch * merge) != 0
    {
        return Err(anyhow!(
            "output size {}x{} must be divisible by patch*merge={}",
            args.width, args.height, patch * merge
        ));
    }
    let out_grid_h = args.height as usize / patch;
    let out_grid_w = args.width as usize / patch;
    let out_token_h = out_grid_h / merge;
    let out_token_w = out_grid_w / merge;
    let out_l = out_token_h * out_token_w;
    let b: usize = 1;

    // 1) Preprocess input image.
    let (in_pixels, in_grid_h, in_grid_w) = load_image_native(
        &args.input_image,
        patch, cfg.downsample_ratio,
        args.in_min_pixels, args.in_max_pixels,
        &device,
    )?;
    let in_token_h = in_grid_h / merge;
    let in_token_w = in_grid_w / merge;
    let in_l = in_token_h * in_token_w;
    eprintln!(
        "[sensenova_u1_edit] input image: pixel_grid={}x{}, token_grid={}x{} (L_in={})",
        in_grid_h, in_grid_w, in_token_h, in_token_w, in_l
    );

    // 2) Build cond / uncond queries.
    let cond_append = if args.think_mode { "<think>\n" } else { "<think>\n\n</think>\n\n<img>" };
    let cond_user = if args.prompt.contains("<image>") {
        args.prompt.clone()
    } else {
        format!("<image>\n{}", args.prompt)
    };
    let cond_query_raw = build_query(SYSTEM_MESSAGE_FOR_GEN, &cond_user, cond_append);
    let img_block = format!("{}{}{}", IMG_START, IMG_CONTEXT.repeat(in_l), IMG_END);
    let cond_query = cond_query_raw.replacen("<image>", &img_block, 1);
    let uncond_query = build_query("", "", "<img>");
    let cond_ids = encode_query(&tok, &cond_query)?;
    let uncond_ids = encode_query(&tok, &uncond_query)?;
    eprintln!(
        "[sensenova_u1_edit] cond tokens={} uncond tokens={}{}",
        cond_ids.len(),
        uncond_ids.len(),
        if args.think_mode { "  [think mode]" } else { "" }
    );

    let n_imgctx = cond_ids.iter().filter(|&&id| id == IMG_CONTEXT_ID).count();
    if n_imgctx != in_l {
        return Err(anyhow!(
            "expected {in_l} <IMG_CONTEXT> tokens after splice, got {n_imgctx}"
        ));
    }

    // 3) extract_feature_und + splice into cond embeddings.
    let t0 = std::time::Instant::now();
    let img_feats = model.extract_feature_und(&in_pixels, in_grid_h, in_grid_w)?;
    eprintln!(
        "[sensenova_u1_edit] extract_feature_und: {:.2}s",
        t0.elapsed().as_secs_f32()
    );
    let cond_embeds = model.embed_with_image_splice(&cond_ids, IMG_CONTEXT_ID, &[img_feats])?;
    let (t_idx, h_idx, w_idx, image_mask) =
        model.build_thw_indexes(&cond_ids, IMG_CONTEXT_ID, IMG_START_ID, &[(in_token_h, in_token_w)])?;

    // 4) Mixed-mode prefix forward (cond) + base prefix forward (uncond).
    let t1 = std::time::Instant::now();
    let (mut cond_cache, cond_last_hidden) =
        model.forward_mixed_prefix(&cond_embeds, &image_mask, &t_idx, &h_idx, &w_idx)?;
    let (uncond_cache, _) = model.forward_und(&uncond_ids)?;
    eprintln!(
        "[sensenova_u1_edit] prefix forwards: {:.2}s (cond_prefix_len={}, uncond_prefix_len={})",
        t1.elapsed().as_secs_f32(), cond_cache.prefix_len, uncond_cache.prefix_len
    );

    // 5) Optional --think on cond stream.
    if args.think_mode {
        let t_think = std::time::Instant::now();
        let think_ids = model.decode_autoregressive(
            &mut cond_cache,
            &cond_last_hidden,
            args.max_think_tokens,
            &[THINK_END_ID, IM_END_ID],
        )?;
        let think_text = {
            let u32s: Vec<u32> = think_ids.iter().map(|&i| i as u32).collect();
            tok.decode(&u32s, true).unwrap_or_default()
        };
        eprintln!(
            "[sensenova_u1_edit] think: {} tokens in {:.2}s\n--- think ---\n{}\n---",
            think_ids.len(), t_think.elapsed().as_secs_f32(), think_text
        );
        let append_ids = encode_query(&tok, "\n\n<img>")?;
        let _ = model.extend_cache_with_text_tokens(&mut cond_cache, &append_ids)?;
    }

    // 6) Init noise image at output resolution.
    let noise_scale = model.compute_noise_scale(out_grid_h, out_grid_w);
    eprintln!(
        "[sensenova_u1_edit] output grid={}x{}, tokens={}x{} (L={}), noise_scale={:.4}",
        out_grid_h, out_grid_w, out_token_h, out_token_w, out_l, noise_scale
    );
    let mut img = make_noise_image(args.seed, &[b, 3, args.height as usize, args.width as usize], &device)?;
    img = img.mul_scalar(noise_scale)?;

    let mut t_uniform: Vec<f32> = (0..=args.num_steps as usize)
        .map(|i| i as f32 / args.num_steps as f32).collect();
    t_uniform = model.apply_time_schedule(&t_uniform, out_l, args.timestep_shift);

    // 7) ODE step loop with single-CFG combine.
    let s_norm = noise_scale / cfg.noise_scale_max_value;
    for step in 0..args.num_steps as usize {
        let t = t_uniform[step];
        let t_next = t_uniform[step + 1];
        let step_t0 = std::time::Instant::now();

        let z = patchify(&img, patch * merge, false)?;
        let pixel_values = patchify(&img, patch, true)?;
        let pixel_flat = pixel_values.reshape(&[b * out_grid_h * out_grid_w, 3 * patch * patch])?;

        let mut image_embeds = model.extract_feature_gen(&pixel_flat, out_grid_h, out_grid_w)?;
        let t_vec = vec![t; b * out_l];
        let t_tensor = Tensor::from_vec(t_vec, Shape::from_dims(&[b * out_l]), device.clone())?
            .to_dtype(DType::BF16)?;
        let t_emb = model.time_or_scale_embed(&t_tensor, TimeOrScale::Timestep)?
            .reshape(&[b, out_l, cfg.hidden_size])?;
        let mut additive = t_emb;
        if cfg.add_noise_scale_embedding {
            let s_tensor = Tensor::from_vec(
                vec![s_norm; b * out_l],
                Shape::from_dims(&[b * out_l]), device.clone(),
            )?.to_dtype(DType::BF16)?;
            let s_emb = model.time_or_scale_embed(&s_tensor, TimeOrScale::NoiseScale)?
                .reshape(&[b, out_l, cfg.hidden_size])?;
            additive = additive.add(&s_emb)?;
        }
        image_embeds = image_embeds.add(&additive)?;

        let h_cond = forward_gen_for(&mut model, &image_embeds, cond_cache.next_t_index,
                                      out_token_h, out_token_w, &cond_cache)?;
        let h_uncond = forward_gen_for(&mut model, &image_embeds, uncond_cache.next_t_index,
                                        out_token_h, out_token_w, &uncond_cache)?;
        let x_cond = model.fm_head_forward(&h_cond)?;
        let x_uncond = model.fm_head_forward(&h_uncond)?;
        let denom = (1.0 - t).max(cfg.t_eps);
        let inv = 1.0 / denom;
        let v_cond = x_cond.sub(&z)?.mul_scalar(inv)?;
        let v_uncond = x_uncond.sub(&z)?.mul_scalar(inv)?;
        let v_diff = v_cond.sub(&v_uncond)?;
        let v = v_uncond.add(&v_diff.mul_scalar(args.cfg_scale)?)?;

        let z_next = z.add(&v.mul_scalar(t_next - t)?)?;
        img = unpatchify(&z_next, patch * merge, args.height as usize, args.width as usize)?;

        eprintln!(
            "[sensenova_u1_edit] step {:>3}/{}  t={:.4}→{:.4}  {:.2}s",
            step + 1, args.num_steps, t, t_next, step_t0.elapsed().as_secs_f32()
        );
    }

    let final_img = img.mul_scalar(0.5)?.add_scalar(0.5)?;
    save_png(&final_img, &args.output)?;
    eprintln!("[sensenova_u1_edit] saved → {}", args.output.display());
    Ok(())
}

fn forward_gen_for(
    model: &mut SenseNovaU1, image_embeds: &Tensor, text_len: usize,
    token_h: usize, token_w: usize, cache: &KvCache,
) -> Result<Tensor> {
    Ok(model.forward_gen(image_embeds, text_len, token_h, token_w, cache, None)?)
}
