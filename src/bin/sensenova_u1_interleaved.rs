//! SenseNova-U1-8B-MoT — pure-Rust autonomous interleaved generation CLI.
//!
//! State-machine driver on top of `decode_autoregressive` and the existing
//! T2I image-gen loop. Output is a stream of (text_segment, png_path) pairs.
//!
//! Loop:
//!   1. Decode tokens with EOS={<img> 151670, <|im_end|> 151645}.
//!   2. On `<img>`: dump accumulated text, run T2I gen using the current
//!      cond cache + a one-time uncond cache, save PNG, push `</img>` plus
//!      a single `\n` separator into cond cache, increment img_count.
//!   3. On `<|im_end|>` or `img_count >= max_images`: stop.
//!
//! Reference: closest analog is `interleave_gen_image_only` in
//! modeling_neo_chat.py:626 (scripted), but here we run AUTONOMOUSLY —
//! the model decides when to emit `<img>` rather than following a fixed
//! script. The model card claims the 8B-MoT checkpoint is trained for this.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
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

const IMG_START_ID: i32 = 151670;
const IM_END_ID: i32 = 151645;

#[derive(Debug)]
struct Args {
    weights_dir: PathBuf,
    prompt: String,
    output_dir: PathBuf,
    width: u32,
    height: u32,
    cfg_scale: f32,
    timestep_shift: f32,
    num_steps: u32,
    seed: u64,
    max_text_tokens: usize,
    max_images: usize,
}

impl Args {
    fn defaults() -> Self {
        Self {
            weights_dir: PathBuf::from("/home/alex/.serenity/models/sensenova_u1"),
            prompt: String::new(),
            output_dir: PathBuf::from("inference-flame/output/sensenova_u1/interleaved"),
            width: 512,
            height: 512,
            cfg_scale: 4.0,
            timestep_shift: 3.0,
            num_steps: 30,
            seed: 42,
            max_text_tokens: 256,
            max_images: 4,
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
            argv.get(i).cloned().ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--weights" | "--model_path" => a.weights_dir = PathBuf::from(next()?),
            "--prompt" => a.prompt = next()?,
            "--output_dir" | "-o" => a.output_dir = PathBuf::from(next()?),
            "--width" => a.width = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--height" => a.height = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--cfg_scale" | "--cfg" => a.cfg_scale = next()?.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?,
            "--timestep_shift" | "--shift" => a.timestep_shift = next()?.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?,
            "--num_steps" | "--steps" => a.num_steps = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--seed" => a.seed = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--max_text_tokens" => a.max_text_tokens = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--max_images" => a.max_images = next()?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "-h" | "--help" => {
                eprintln!(
                    "sensenova_u1_interleaved — autonomous interleaved text+image generation\n\
                    Usage: sensenova_u1_interleaved --prompt STR [options]\n\
                    --output_dir DIR    where to write text.txt + img_{{N}}.png [default inference-flame/output/sensenova_u1/interleaved]\n\
                    --max_text_tokens   per-text-segment decode budget [256]\n\
                    --max_images N      stop after this many images [4]\n"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if a.prompt.is_empty() { return Err("missing --prompt".into()); }
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
    ).build().map_err(|e| anyhow!("BPE::build failed: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevel::default()));
    let raw = std::fs::read_to_string(&added).with_context(|| format!("read {}", added.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).context("added_tokens.json")?;
    let mut entries: Vec<(String, u64)> = map.into_iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id))).collect();
    entries.sort_by_key(|(_, id)| *id);
    let added_tokens: Vec<AddedToken> = entries.into_iter()
        .map(|(content, _)| AddedToken::from(content, true)).collect();
    tok.add_special_tokens(&added_tokens);
    Ok(tok)
}

fn encode_query(tok: &Tokenizer, q: &str) -> Result<Vec<i32>> {
    let enc = tok.encode(q, false).map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

fn decode_ids(tok: &Tokenizer, ids: &[i32]) -> String {
    let u32s: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    tok.decode(&u32s, true).unwrap_or_default()
}

fn build_query(system: &str, user: &str, append: &str) -> String {
    let mut q = String::new();
    if !system.is_empty() {
        q.push_str("<|im_start|>system\n"); q.push_str(system); q.push_str("<|im_end|>\n");
    }
    q.push_str("<|im_start|>user\n"); q.push_str(user); q.push_str("<|im_end|>\n");
    q.push_str("<|im_start|>assistant\n"); q.push_str(append);
    q
}

// ---------------------------------------------------------------------------
// Image gen helpers (mirror of sensenova_u1_gen.rs)
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
    let x = if channel_first { x.permute(&[0, 2, 4, 1, 3, 5])? } else { x.permute(&[0, 2, 4, 3, 5, 1])? };
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
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).ok(); }
    let buf = image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("PNG buffer build failed"))?;
    buf.save(path).with_context(|| format!("save {}", path.display()))?;
    Ok(())
}

/// Run a full T2I image-gen step loop using `cond_cache` (the live
/// interleaved-decode cache) and `uncond_cache` (a stable empty-prompt cache).
/// Returns the [1, 3, H, W] BF16 final image in [0, 1] (post-denorm).
#[allow(clippy::too_many_arguments)]
fn run_image_gen_step(
    model: &mut SenseNovaU1,
    device: &Arc<CudaDevice>,
    cond_cache: &KvCache,
    uncond_cache: &KvCache,
    width: usize,
    height: usize,
    cfg_scale: f32,
    timestep_shift: f32,
    num_steps: usize,
    seed: u64,
) -> Result<Tensor> {
    let cfg = model.config().clone();
    let merge = cfg.merge_size();
    let patch = cfg.patch_size;
    if width % (patch * merge) != 0 || height % (patch * merge) != 0 {
        return Err(anyhow!(
            "image size {}x{} must be divisible by patch*merge={}",
            width, height, patch * merge
        ));
    }
    let grid_h = height / patch;
    let grid_w = width / patch;
    let token_h = grid_h / merge;
    let token_w = grid_w / merge;
    let l = token_h * token_w;
    let b: usize = 1;

    let noise_scale = model.compute_noise_scale(grid_h, grid_w);
    let mut img = make_noise_image(seed, &[b, 3, height, width], device)?;
    img = img.mul_scalar(noise_scale)?;

    let mut t_uniform: Vec<f32> = (0..=num_steps).map(|i| i as f32 / num_steps as f32).collect();
    t_uniform = model.apply_time_schedule(&t_uniform, l, timestep_shift);
    let s_norm = noise_scale / cfg.noise_scale_max_value;

    for step in 0..num_steps {
        let t = t_uniform[step];
        let t_next = t_uniform[step + 1];
        let z = patchify(&img, patch * merge, false)?;
        let pixel_values = patchify(&img, patch, true)?;
        let pixel_flat = pixel_values.reshape(&[b * grid_h * grid_w, 3 * patch * patch])?;
        let mut image_embeds = model.extract_feature_gen(&pixel_flat, grid_h, grid_w)?;
        let t_vec = vec![t; b * l];
        let t_tensor = Tensor::from_vec(t_vec, Shape::from_dims(&[b * l]), device.clone())?
            .to_dtype(DType::BF16)?;
        let t_emb = model.time_or_scale_embed(&t_tensor, TimeOrScale::Timestep)?
            .reshape(&[b, l, cfg.hidden_size])?;
        let mut additive = t_emb;
        if cfg.add_noise_scale_embedding {
            let s_tensor = Tensor::from_vec(
                vec![s_norm; b * l],
                Shape::from_dims(&[b * l]), device.clone(),
            )?.to_dtype(DType::BF16)?;
            let s_emb = model.time_or_scale_embed(&s_tensor, TimeOrScale::NoiseScale)?
                .reshape(&[b, l, cfg.hidden_size])?;
            additive = additive.add(&s_emb)?;
        }
        image_embeds = image_embeds.add(&additive)?;

        let h_cond = model.forward_gen(&image_embeds, cond_cache.next_t_index, token_h, token_w, cond_cache, None)?;
        let h_uncond = model.forward_gen(&image_embeds, uncond_cache.next_t_index, token_h, token_w, uncond_cache, None)?;
        let x_cond = model.fm_head_forward(&h_cond)?;
        let x_uncond = model.fm_head_forward(&h_uncond)?;
        let denom = (1.0 - t).max(cfg.t_eps);
        let inv = 1.0 / denom;
        let v_cond = x_cond.sub(&z)?.mul_scalar(inv)?;
        let v_uncond = x_uncond.sub(&z)?.mul_scalar(inv)?;
        let v = v_uncond.add(&v_cond.sub(&v_uncond)?.mul_scalar(cfg_scale)?)?;
        let z_next = z.add(&v.mul_scalar(t_next - t)?)?;
        img = unpatchify(&z_next, patch * merge, height, width)?;
    }
    Ok(img.mul_scalar(0.5)?.add_scalar(0.5)?)
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
        eprintln!("[sensenova_u1_interleaved] error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> Result<()> {
    let device = flame_core::global_cuda_device();
    eprintln!("[sensenova_u1_interleaved] loading tokenizer + weights from {:?}", args.weights_dir);
    let tok = build_tokenizer(&args.weights_dir)?;
    let mut model = SenseNovaU1::load(&args.weights_dir, &device)?;

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    // 1) Build cond + uncond prefixes. Cond carries the system + user message
    // and ends with the assistant role open + a closed think block. The model
    // is then free to emit text and `<img>` tokens autoregressively.
    let cond_query = build_query(SYSTEM_MESSAGE_FOR_GEN, &args.prompt, "<think>\n\n</think>\n\n");
    let uncond_query = build_query("", "", "<img>");
    let cond_ids = encode_query(&tok, &cond_query)?;
    let uncond_ids = encode_query(&tok, &uncond_query)?;
    eprintln!("[sensenova_u1_interleaved] cond tokens={} uncond tokens={}", cond_ids.len(), uncond_ids.len());

    let t0 = std::time::Instant::now();
    let (mut cond_cache, mut last_hidden) = model.forward_und(&cond_ids)?;
    let (uncond_cache, _) = model.forward_und(&uncond_ids)?;
    eprintln!("[sensenova_u1_interleaved] prefix forwards: {:.2}s", t0.elapsed().as_secs_f32());

    // 2) State machine: alternate between text decode and image gen.
    let mut transcript: Vec<i32> = Vec::new(); // accumulated text token ids
    let mut text_path = args.output_dir.clone();
    text_path.push("transcript.txt");
    let mut text_file = std::fs::File::create(&text_path)
        .with_context(|| format!("create {}", text_path.display()))?;
    let mut img_count: usize = 0;
    let mut total_text_emitted: usize = 0;

    loop {
        // Decode until <img>, <|im_end|>, or budget exceeded.
        let t_dec = std::time::Instant::now();
        let new_ids = model.decode_autoregressive(
            &mut cond_cache, &last_hidden,
            args.max_text_tokens, &[IMG_START_ID, IM_END_ID],
        )?;
        eprintln!(
            "[sensenova_u1_interleaved] decode segment: {} tokens in {:.2}s",
            new_ids.len(), t_dec.elapsed().as_secs_f32()
        );

        // Identify whether the segment ended in <img> / <|im_end|> / budget.
        let last = new_ids.last().copied();
        let body: &[i32] = match last {
            Some(IMG_START_ID) | Some(IM_END_ID) => &new_ids[..new_ids.len().saturating_sub(1)],
            _ => &new_ids[..],
        };
        if !body.is_empty() {
            let txt = decode_ids(&tok, body);
            use std::io::Write;
            text_file.write_all(txt.as_bytes()).ok();
            text_file.flush().ok();
            transcript.extend_from_slice(body);
            total_text_emitted += body.len();
            print!("{}", txt);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        match last {
            Some(IM_END_ID) => {
                eprintln!("\n[sensenova_u1_interleaved] hit <|im_end|>, stopping.");
                break;
            }
            Some(IMG_START_ID) => {
                if img_count >= args.max_images {
                    eprintln!("\n[sensenova_u1_interleaved] reached --max_images {}, stopping.", args.max_images);
                    break;
                }
                eprintln!("\n[sensenova_u1_interleaved] <img> emitted at decode pos — running gen step #{}", img_count + 1);
                let img_t0 = std::time::Instant::now();
                let final_img = run_image_gen_step(
                    &mut model, &device,
                    &cond_cache, &uncond_cache,
                    args.width as usize, args.height as usize,
                    args.cfg_scale, args.timestep_shift, args.num_steps as usize,
                    args.seed.wrapping_add(img_count as u64),
                )?;
                let mut out_path = args.output_dir.clone();
                out_path.push(format!("img_{:02}.png", img_count));
                save_png(&final_img, &out_path)?;
                eprintln!(
                    "[sensenova_u1_interleaved] image #{} saved → {} ({:.2}s)",
                    img_count, out_path.display(), img_t0.elapsed().as_secs_f32()
                );
                img_count += 1;
                // Push </img> + a single newline into the cache so decoding
                // resumes after the image. Mirrors the python `<img>...</img>`
                // wrapping on the assistant side.
                let close_ids: Vec<i32> = encode_query(&tok, "</img>\n")?;
                last_hidden = model.extend_cache_with_text_tokens(&mut cond_cache, &close_ids)?;
            }
            _ => {
                // Budget exhausted within a single segment. We stop to avoid
                // runaway decoding; a future caller can raise --max_text_tokens.
                eprintln!("\n[sensenova_u1_interleaved] hit --max_text_tokens, stopping.");
                break;
            }
        }
    }

    eprintln!(
        "[sensenova_u1_interleaved] done: {} text tokens, {} images → {}",
        total_text_emitted, img_count, args.output_dir.display()
    );
    Ok(())
}
