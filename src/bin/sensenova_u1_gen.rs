//! SenseNova-U1-8B-MoT — pure-Rust T2I CLI.
//!
//! Reference: `modeling_neo_chat.py::t2i_generate` (line 1578) +
//! `examples/t2i/inference.py` (parse_args).
//!
//! Defaults match the reference inference script:
//!   --width 2048 --height 2048 --cfg_scale 4.0 --cfg_norm none
//!   --timestep_shift 3.0 --num_steps 50 --seed 42
//!
//! Tokenizer is constructed in-process from `vocab.json` + `merges.txt` +
//! `added_tokens.json` because the SenseNova-U1 weights dir does not ship a
//! unified `tokenizer.json`. We use the HF `tokenizers` crate (already a dep)
//! with a ByteLevel-BPE pipeline matching Qwen3.

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

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    weights_dir: PathBuf,
    prompt: String,
    output: PathBuf,
    width: u32,
    height: u32,
    cfg_scale: f32,
    timestep_shift: f32,
    num_steps: u32,
    seed: u64,
    think_mode: bool,
    max_think_tokens: usize,
}

impl Args {
    fn defaults() -> Self {
        Self {
            weights_dir: PathBuf::from("/home/alex/.serenity/models/sensenova_u1"),
            prompt: String::new(),
            output: PathBuf::from("inference-flame/output/sensenova_u1/sample.png"),
            width: 2048,
            height: 2048,
            cfg_scale: 4.0,
            timestep_shift: 3.0,
            num_steps: 50,
            seed: 42,
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
            "--prompt" => a.prompt = next()?,
            "--output" | "-o" => a.output = PathBuf::from(next()?),
            "--width" => {
                a.width = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--height" => {
                a.height = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--cfg_scale" | "--cfg" => {
                a.cfg_scale = next()?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--timestep_shift" | "--shift" => {
                a.timestep_shift = next()?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--num_steps" | "--steps" => {
                a.num_steps = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--seed" => {
                a.seed = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--think" => a.think_mode = true,
            "--max_think_tokens" => {
                a.max_think_tokens = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if a.prompt.is_empty() {
        return Err("missing required --prompt".into());
    }
    Ok(a)
}

fn print_usage() {
    eprintln!(
        r#"sensenova_u1_gen — SenseNova-U1-8B-MoT T2I

Usage:
  sensenova_u1_gen --prompt "..." [options]

Options:
  --prompt <STR>          (required) text prompt
  --weights <DIR>         path to weights dir   [default: /home/alex/.serenity/models/sensenova_u1]
  --output  <PATH>        output PNG path       [default: inference-flame/output/sensenova_u1/sample.png]
  --width   <N>           output width          [default: 2048]
  --height  <N>           output height         [default: 2048]
  --cfg_scale <F>         CFG guidance scale    [default: 4.0]
  --timestep_shift <F>    exponential shift     [default: 3.0]
  --num_steps <N>         denoise steps         [default: 50]
  --seed <N>              RNG seed              [default: 42]
"#
    );
}

// ---------------------------------------------------------------------------
// Tokenizer construction
// ---------------------------------------------------------------------------

/// Build a Qwen3-style ByteLevel-BPE tokenizer from `vocab.json` + `merges.txt`,
/// then add the 293 special tokens from `added_tokens.json` in ID order so the
/// auto-assigned IDs match the reference (151643..151935, contiguous).
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

    // Read added_tokens.json — content -> id. Sort by id and add in order; the
    // tokenizers crate appends to the vocab so the auto-IDs match exactly when
    // base vocab.json size + added contiguous == declared IDs.
    let raw = std::fs::read_to_string(&added)
        .with_context(|| format!("read {}", added.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).context("added_tokens.json")?;
    let mut entries: Vec<(String, u64)> = map
        .into_iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    entries.sort_by_key(|(_, id)| *id);

    // Sanity: first added token must immediately follow the base vocab.
    let base_size = tok.get_vocab_size(false) as u64;
    if let Some((_, first_id)) = entries.first() {
        if *first_id != base_size {
            return Err(anyhow!(
                "added_tokens.json starts at id {first_id} but base vocab size is {base_size}; \
                 cannot align IDs via append-only AddedToken API"
            ));
        }
    }

    let added_tokens: Vec<AddedToken> = entries
        .into_iter()
        .map(|(content, _)| AddedToken::from(content, true))
        .collect();
    tok.add_special_tokens(&added_tokens);

    // Verify a known token landed at the expected ID.
    let im_start = tok
        .token_to_id("<|im_start|>")
        .ok_or_else(|| anyhow!("<|im_start|> not in tokenizer"))?;
    if im_start != 151644 {
        return Err(anyhow!(
            "<|im_start|> mapped to {im_start}, expected 151644"
        ));
    }
    Ok(tok)
}

/// Build the T2I chat-template prompt. Mirrors `_build_t2i_query`
/// (modeling_neo_chat.py:431) + MPT separator (conversation.py:234).
///
/// Format:
///   <|im_start|>system\n{system}<|im_end|>\n
///   <|im_start|>user\n{user}<|im_end|>\n
///   <|im_start|>assistant\n
///   {append}
///
/// For an empty `system` (CFG-uncond), the system block is omitted.
fn build_t2i_query(system: &str, user: &str, append: &str) -> String {
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

fn encode_query(tok: &Tokenizer, query: &str) -> Result<Vec<i32>> {
    let enc = tok
        .encode(query, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

// ---------------------------------------------------------------------------
// Tensor helpers (patchify / unpatchify / noise / save)
// ---------------------------------------------------------------------------

/// Box-Muller seeded Gaussian noise → BF16 Tensor of shape `[B, 3, H, W]`.
fn make_noise_image(
    seed: u64,
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
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

/// `patchify(images=[B, 3, H, W], p, channel_first)` → `[B, h*w, p*p*3]`.
/// Reference: modeling_neo_chat.py::patchify (line 366).
///
/// `channel_first=true` flattens patches in (C, kH, kW) C-major order so the
/// 768-dim inner axis matches Conv2d weight `[1024, 3, 16, 16]` reshape order.
/// `channel_first=false` flattens in (kH, kW, C) order — used for `z` in the
/// velocity formula (z is what the model predicts on, so the 3072-dim inner
/// axis matches the fm_head 3072 output, which corresponds to a 32×32×3 patch
/// in (kH, kW, C) order).
fn patchify(images: &Tensor, p: usize, channel_first: bool) -> Result<Tensor> {
    let dims = images.shape().dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(anyhow!("patchify expects [B, 3, H, W], got {dims:?}"));
    }
    let (b, h, w) = (dims[0], dims[2], dims[3]);
    if h % p != 0 || w % p != 0 {
        return Err(anyhow!("patchify: H={h} W={w} not divisible by p={p}"));
    }
    let gh = h / p;
    let gw = w / p;
    // Reshape [B, 3, gh, p, gw, p].
    let x = images.reshape(&[b, 3, gh, p, gw, p])?;
    // einsum 'bchpwq->bhwcpq' (channel_first) or 'bchpwq->bhwpqc'.
    // Source axes: 0=B 1=C 2=gh 3=p 4=gw 5=p
    let x = if channel_first {
        // Target: B gh gw C p p   →  axes [0, 2, 4, 1, 3, 5]
        x.permute(&[0, 2, 4, 1, 3, 5])?
    } else {
        // Target: B gh gw p p C   →  axes [0, 2, 4, 3, 5, 1]
        x.permute(&[0, 2, 4, 3, 5, 1])?
    };
    Ok(x.reshape(&[b, gh * gw, p * p * 3])?)
}

/// `unpatchify(x=[B, L, p*p*3], p, h, w)` → `[B, 3, h, w]`. The inner axis
/// flatten order is (kH, kW, C) (channel_first=false during patchify).
fn unpatchify(x: &Tensor, p: usize, h: usize, w: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(anyhow!("unpatchify expects [B, L, D], got {dims:?}"));
    }
    let b = dims[0];
    let gh = h / p;
    let gw = w / p;
    if gh * gw != dims[1] {
        return Err(anyhow!(
            "unpatchify: L={} != gh*gw={}*{}",
            dims[1], gh, gw
        ));
    }
    // einsum 'nhwpqc->nchpwq' inverse.
    // Reshape to [B, gh, gw, p, p, 3]
    let x = x.reshape(&[b, gh, gw, p, p, 3])?;
    // Target: B C gh p gw p   →  axes [0, 5, 1, 3, 2, 4]
    let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
    Ok(x.reshape(&[b, 3, gh * p, gw * p])?)
}

/// Save `[1, 3, H, W]` BF16 in [-1, 1] (after the `*0.5+0.5` denorm we apply
/// before this is called, the values are in [0, 1]) to a PNG.
fn save_png(image: &Tensor, path: &Path) -> Result<()> {
    let img_f32 = image.to_dtype(DType::F32)?;
    let dims = img_f32.shape().dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(anyhow!("save_png expects [1, 3, H, W], got {dims:?}"));
    }
    let (h, w) = (dims[2], dims[3]);
    let data = img_f32.to_vec_f32()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = data[c * h * w + y * w + x];
                let u = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("RgbImage::from_raw failed"))?
        .save(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_t2i(
    model: &mut SenseNovaU1,
    tok: &Tokenizer,
    args: &Args,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    // Clone the config out so the immutable borrow doesn't conflict with the
    // `&mut model` calls into forward_und / forward_gen later.
    let cfg = model.config().clone();
    let merge = cfg.downsample_ratio.recip().round() as usize; // 2
    let patch = cfg.patch_size; // 16
    if (args.width as usize) % (patch * merge) != 0
        || (args.height as usize) % (patch * merge) != 0
    {
        return Err(anyhow!(
            "image size {}x{} must be divisible by patch*merge={}",
            args.width,
            args.height,
            patch * merge
        ));
    }
    let grid_h = args.height as usize / patch;
    let grid_w = args.width as usize / patch;
    let token_h = grid_h / merge;
    let token_w = grid_w / merge;
    let l_tokens = token_h * token_w;
    let b: usize = 1;

    // ---- Tokenize cond/uncond ----
    // The conditional query branches on think_mode:
    //   - non-think: append `<think>\n\n</think>\n\n<img>` so the assistant
    //     emits an empty think block then the <img> sentinel.
    //   - think:     append `<think>\n` only — the model autoregressively fills
    //     the think block, we stop on `</think>`, then push `\n\n<img>` into
    //     the cache. Mirrors the python `t2i_generate(think_mode=True)` path.
    let cond_append = if args.think_mode {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n<img>"
    };
    let cond_query = build_t2i_query(SYSTEM_MESSAGE_FOR_GEN, &args.prompt, cond_append);
    let uncond_query = build_t2i_query("", "", "<img>");
    let cond_ids = encode_query(tok, &cond_query)?;
    let uncond_ids = encode_query(tok, &uncond_query)?;
    eprintln!(
        "[sensenova_u1] cond tokens={}  uncond tokens={}{}",
        cond_ids.len(),
        uncond_ids.len(),
        if args.think_mode { "  [think mode]" } else { "" }
    );

    // ---- Prefix forwards (one-time per generation) ----
    let t0 = std::time::Instant::now();
    let (mut cond_cache, cond_last_hidden) = model.forward_und(&cond_ids)?;
    let (uncond_cache, _) = model.forward_und(&uncond_ids)?;
    eprintln!("[sensenova_u1] prefix forward: {:.2}s", t0.elapsed().as_secs_f32());

    // ---- Optional: think-mode autoregressive generation, then append the
    // literal "\n\n<img>" tokens to the cache. The image-gen loop below works
    // off the EXTENDED cond_cache; uncond_cache is unaffected. ----
    if args.think_mode {
        let t_think = std::time::Instant::now();
        let think_ids = model.decode_autoregressive(
            &mut cond_cache,
            &cond_last_hidden,
            args.max_think_tokens,
            &[151668 /* </think> */, 151645 /* <|im_end|> */],
        )?;
        let think_text = {
            let u32s: Vec<u32> = think_ids.iter().map(|&i| i as u32).collect();
            tok.decode(&u32s, true).unwrap_or_default()
        };
        eprintln!(
            "[sensenova_u1] think: {} tokens in {:.2}s\n--- think ---\n{}\n---",
            think_ids.len(),
            t_think.elapsed().as_secs_f32(),
            think_text
        );
        // Push the literal "\n\n<img>" continuation into the cache.
        let append_str = "\n\n<img>";
        let append_ids = encode_query(tok, append_str)?;
        let _ = model.extend_cache_with_text_tokens(&mut cond_cache, &append_ids)?;
    }

    // ---- Init noise image ----
    let noise_scale = model.compute_noise_scale(grid_h, grid_w);
    eprintln!(
        "[sensenova_u1] grid={}x{}, tokens={}x{} (L={}), noise_scale={:.4}",
        grid_h, grid_w, token_h, token_w, l_tokens, noise_scale
    );
    let mut img = make_noise_image(
        args.seed,
        &[b, 3, args.height as usize, args.width as usize],
        device,
    )?;
    img = img.mul_scalar(noise_scale)?;

    // ---- Build timestep grid ----
    let mut t_uniform: Vec<f32> = (0..=args.num_steps as usize)
        .map(|i| i as f32 / args.num_steps as f32)
        .collect();
    t_uniform = model.apply_time_schedule(&t_uniform, l_tokens, args.timestep_shift);

    // ---- Step loop ----
    let s_norm = noise_scale / cfg.noise_scale_max_value;
    for step in 0..args.num_steps as usize {
        let t = t_uniform[step];
        let t_next = t_uniform[step + 1];
        let step_t0 = std::time::Instant::now();

        // z (target dim) and pixel_values (input to gen embedder)
        let z = patchify(&img, patch * merge, false)?; // [B, L, 3072]
        let pixel_values = patchify(&img, patch, true)?; // [B, grid_h*grid_w, 768]
        let pixel_flat = pixel_values.reshape(&[b * grid_h * grid_w, 3 * patch * patch])?;

        // gen embedder + timestep / noise_scale embedding
        let mut image_embeds = model.extract_feature_gen(&pixel_flat, grid_h, grid_w)?;
        let t_vec = vec![t; b * l_tokens];
        let t_tensor = Tensor::from_vec(
            t_vec,
            Shape::from_dims(&[b * l_tokens]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let t_emb = model
            .time_or_scale_embed(&t_tensor, TimeOrScale::Timestep)?
            .reshape(&[b, l_tokens, cfg.hidden_size])?;
        let mut additive = t_emb;
        if cfg.add_noise_scale_embedding {
            let s_tensor = Tensor::from_vec(
                vec![s_norm; b * l_tokens],
                Shape::from_dims(&[b * l_tokens]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;
            let s_emb = model
                .time_or_scale_embed(&s_tensor, TimeOrScale::NoiseScale)?
                .reshape(&[b, l_tokens, cfg.hidden_size])?;
            additive = additive.add(&s_emb)?;
        }
        image_embeds = image_embeds.add(&additive)?;

        // CFG cond + uncond passes through forward_gen. forward_gen takes
        // &mut self (it drives the offloader), so we sequence the two passes
        // through the same `model` borrow. `text_len` here is the t-axis
        // position to assign to image tokens — equal to `next_t_index` of the
        // KV cache (post-extension if think-mode appended literals).
        let h_cond = forward_gen_for(
            model,
            &image_embeds,
            cond_cache.next_t_index,
            token_h,
            token_w,
            &cond_cache,
        )?;
        let h_uncond = forward_gen_for(
            model,
            &image_embeds,
            uncond_cache.next_t_index,
            token_h,
            token_w,
            &uncond_cache,
        )?;
        let x_cond = model.fm_head_forward(&h_cond)?;
        let x_uncond = model.fm_head_forward(&h_uncond)?;
        let denom = (1.0 - t).max(cfg.t_eps);
        let inv_denom = 1.0 / denom;
        let v_cond = x_cond.sub(&z)?.mul_scalar(inv_denom)?;
        let v_uncond = x_uncond.sub(&z)?.mul_scalar(inv_denom)?;

        // CFG combine (cfg_norm='none' branch).
        let v_diff = v_cond.sub(&v_uncond)?;
        let v = v_uncond.add(&v_diff.mul_scalar(args.cfg_scale)?)?;

        // Euler step on z.
        let z_next = z.add(&v.mul_scalar(t_next - t)?)?;

        // Unpatchify back to image space at p*merge=32.
        img = unpatchify(&z_next, patch * merge, args.height as usize, args.width as usize)?;

        eprintln!(
            "[sensenova_u1] step {:>3}/{}  t={:.4}→{:.4}  {:.2}s",
            step + 1,
            args.num_steps,
            t,
            t_next,
            step_t0.elapsed().as_secs_f32()
        );
    }

    // ---- Denorm ((img * 0.5 + 0.5).clamp(0, 1)) and save ----
    let final_img = img.mul_scalar(0.5)?.add_scalar(0.5)?;
    save_png(&final_img, &args.output)?;
    eprintln!("[sensenova_u1] saved → {}", args.output.display());
    Ok(())
}

/// Thin wrapper because `forward_gen` takes `attn_mask: Option<&Tensor>` and
/// we always pass `None` (no padding in our prefix; full attention with the
/// implicit causal-cross-prefix from the cached prefix tokens).
fn forward_gen_for(
    model: &mut SenseNovaU1,
    image_embeds: &Tensor,
    text_len: usize,
    token_h: usize,
    token_w: usize,
    cache: &KvCache,
) -> Result<Tensor> {
    Ok(model.forward_gen(image_embeds, text_len, token_h, token_w, cache, None)?)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    if let Err(e) = run(&args) {
        eprintln!("[sensenova_u1] FAILED: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> Result<()> {
    let device = flame_core::global_cuda_device();

    eprintln!("[sensenova_u1] loading tokenizer...");
    let tok = build_tokenizer(&args.weights_dir)?;

    eprintln!("[sensenova_u1] loading model from {}", args.weights_dir.display());
    let t0 = std::time::Instant::now();
    let mut model = SenseNovaU1::load(&args.weights_dir, &device)?;
    eprintln!("[sensenova_u1] model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    run_t2i(&mut model, &tok, args, &device)
}
