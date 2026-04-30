//! End-to-end Nucleus-Image text-to-image inference (Phase 6.6).
//!
//! Loads the public `NucleusAI/Nucleus-Image` checkpoint (17 B sparse MoE DiT
//! + Qwen3-VL text encoder + Qwen-Image VAE) and produces a PNG from a text
//! prompt — pure Rust, flame-core, no Python in the loop.
//!
//! Memory plan for a 24 GB GPU:
//! 1. Load Qwen3-VL text encoder (~17 GB BF16) resident, encode prompt(s),
//!    drop encoder.
//! 2. Load Nucleus DiT: ~3.5 GB resident, ~15.3 GB streaming via
//!    `BlockOffloader` from pinned host RAM.
//! 3. Load Qwen-Image VAE decoder (~few hundred MB).
//! 4. Compute per-block text K/V cache (Phase 6.5 path).
//! 5. 50-step flow-matching Euler denoise loop (KV cache reused).
//! 6. Drop DiT, unpack + denormalize latents, VAE decode, save PNG.
//!
//! Reference (mirrored line-for-line):
//!   diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py
//!     :155+    __init__ — registers transformer/scheduler/vae/text_encoder/processor
//!     :178+    _format_prompt — applies the chat template
//!     :187+    encode_prompt — runs Qwen3-VL, takes hidden_states[-8]
//!     :381+    __call__ — sigma schedule + denoise loop + VAE decode
//!
//! See HANDOFF_2026-04-29_NUCLEUS_WEIGHT_AUDIT.md for the per-component audit.

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use inference_flame::models::nucleus_dit::NucleusInferDit;
use inference_flame::models::qwen3_encoder::{Qwen3Config, Qwen3Encoder};
use inference_flame::vae::qwenimage_decoder::QwenImageVaeDecoder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Paths + constants pinned to the public checkpoint.
// ---------------------------------------------------------------------------

const SNAPSHOT: &str = "/home/alex/.cache/huggingface/hub/models--NucleusAI--Nucleus-Image/snapshots/4c5d6738f71bb5ea8dea55f4b0c0fd911e0deefd";

/// Diffusers `DEFAULT_SYSTEM_PROMPT` (pipeline_nucleusmoe_image.py:41).
const SYSTEM_PROMPT: &str = "You are an image generation assistant. Follow the user's prompt literally. Pay careful attention to spatial layout: objects described as on the left must appear on the left, on the right on the right. Match exact object counts and assign colors to the correct objects.";

/// Pipeline `default_max_sequence_length`.
const MAX_SEQ_LEN: usize = 1024;
/// Pipeline `pad_to_multiple_of=8`.
const PAD_TO_MULTIPLE: usize = 8;
/// Qwen3 pad / BOS token id.
const PAD_TOKEN_ID: i32 = 151643;
/// `default_return_index = -8`. With 36 layers, hidden_states is a length-37
/// tuple (embed output + 36 layer outputs). Index -8 = 29 = layer 28 output.
const ENCODER_EXTRACT_LAYER: usize = 28;

const VAE_LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const VAE_LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

const VAE_SCALE_FACTOR: usize = 8; // 2^len(temperal_downsample) where temperal_downsample = [F,T,T] → effectively 8 spatial
const PATCH_SIZE: usize = 2;
const Z_DIM: usize = 16;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    prompt: String,
    seed: u64,
    height: usize,
    width: usize,
    steps: usize,
    /// Pipeline default is 4.0. Set to 1.0 to disable CFG (single forward
    /// per step, halves DiT cost). With CFG: encoder runs twice (cond +
    /// empty uncond), DiT runs twice/step, then `comb = uncond + scale *
    /// (cond - uncond)` followed by per-token CFG-Zero* norm rescale.
    guidance: f32,
    out_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut prompt =
            "An orange tabby cat sitting on a wooden kitchen table, soft natural window light, sharp focus, photorealistic, 4k"
                .to_string();
        let mut seed: u64 = 42;
        let mut height: usize = 512;
        let mut width: usize = 512;
        let mut steps: usize = 30;
        let mut guidance: f32 = 4.0; // pipeline default
        let mut out_dir =
            PathBuf::from("/home/alex/EriDiffusion/inference-flame/output");

        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let key = &argv[i];
            let val = argv
                .get(i + 1)
                .ok_or_else(|| anyhow!("flag {key} missing value"))?;
            match key.as_str() {
                "--prompt" => prompt = val.clone(),
                "--seed" => seed = val.parse()?,
                "--height" => height = val.parse()?,
                "--width" => width = val.parse()?,
                "--steps" => steps = val.parse()?,
                "--guidance" => guidance = val.parse()?,
                "--out-dir" => out_dir = PathBuf::from(val),
                _ => return Err(anyhow!("unknown arg: {key}")),
            }
            i += 2;
        }
        Ok(Args { prompt, seed, height, width, steps, guidance, out_dir })
    }
}

// ---------------------------------------------------------------------------
// Prompt formatting + tokenization
// ---------------------------------------------------------------------------

/// Apply the Qwen chat template by hand. Mirrors what
/// `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
/// produces for a system + user message pair.
fn format_prompt_chat(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn tokenize_prompt(prompt: &str) -> Result<Vec<i32>> {
    let tokenizer_path = format!("{SNAPSHOT}/processor/tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load: {e}"))?;
    let formatted = format_prompt_chat(prompt);
    let enc = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    if ids.len() > MAX_SEQ_LEN {
        ids.truncate(MAX_SEQ_LEN);
    }
    while ids.len() % PAD_TO_MULTIPLE != 0 {
        ids.push(PAD_TOKEN_ID);
    }
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Encoder loading + forward
// ---------------------------------------------------------------------------

/// Load the Qwen3-VL text encoder's text branch from `text_encoder/`,
/// remapping `model.language_model.X` → `model.X` (qwen3_encoder.rs's
/// expected layout) and dropping the vision tower (`model.visual.X`) and
/// `lm_head.weight`.
fn load_qwen3vl_text_branch(
    text_encoder_dir: &Path,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let index_path = text_encoder_dir.join("model.safetensors.index.json");
    let index_text = std::fs::read_to_string(&index_path)
        .with_context(|| format!("read {:?}", index_path))?;
    let index: serde_json::Value = serde_json::from_str(&index_text)?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("weight_map missing in {:?}", index_path))?;
    let mut shard_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    shard_names.sort();

    let mut out: HashMap<String, Tensor> = HashMap::new();
    for shard in &shard_names {
        let path = text_encoder_dir.join(shard);
        // Keep only text-branch keys (drop visual + lm_head).
        let part =
            flame_core::serialization::load_file_filtered(&path, device, |key| {
                key.starts_with("model.language_model.")
            })
            .map_err(|e| anyhow!("load_file_filtered {:?}: {e}", path))?;
        for (k, v) in part {
            // Remap: `model.language_model.X` → `model.X`
            let new_key = k.strip_prefix("model.language_model.")
                .map(|s| format!("model.{s}"))
                .ok_or_else(|| anyhow!("unexpected key {k}"))?;
            out.insert(new_key, v);
        }
    }
    Ok(out)
}

fn encode_prompt(device: &Arc<CudaDevice>, token_ids: &[i32]) -> Result<Tensor> {
    let text_encoder_dir = Path::new(SNAPSHOT).join("text_encoder");
    let weights = load_qwen3vl_text_branch(&text_encoder_dir, device)?;

    // Validate critical keys.
    for k in &["model.embed_tokens.weight", "model.norm.weight"] {
        if !weights.contains_key(*k) {
            return Err(anyhow!("text encoder missing {k}"));
        }
    }

    let mut cfg = Qwen3Config::qwen3_vl_text();
    cfg.extract_layers = vec![ENCODER_EXTRACT_LAYER];

    let encoder = Qwen3Encoder::new(weights, cfg, device.clone());
    let hidden = encoder
        .encode(token_ids)
        .map_err(|e| anyhow!("encoder.encode: {e}"))?;
    // hidden shape: (1, S_txt, 4096) BF16 — pre-final-norm. The Nucleus DiT
    // applies its own `txt_norm` RMSNorm internally, so we hand off the raw
    // layer-28 output. Exactly matches diffusers `prompt_embeds = outputs.hidden_states[-8]`.
    Ok(hidden)
}

// ---------------------------------------------------------------------------
// Latent packing / unpacking (matches diffusers `_pack_latents` / `_unpack_latents`)
// ---------------------------------------------------------------------------

/// `(B, Z, H, W)` -> `(B, S_img, Z * P²)` where `S_img = (H/P) * (W/P)`.
/// Matches diffusers `_pack_latents`:
///   view (B, Z, H/P, P, W/P, P) → permute (0,2,4,1,3,5) → reshape.
fn pack_latents(latents: &Tensor, h_grid: usize, w_grid: usize, patch: usize) -> Result<Tensor> {
    let dims = latents.shape().dims();
    let (b, z) = (dims[0], dims[1]);
    // Drop singleton temporal dim if present (Qwen-Image VAE outputs (B, Z, 1, H, W)).
    let latents = if dims.len() == 5 {
        latents.squeeze(Some(2))?
    } else {
        latents.clone()
    };
    let v = latents.reshape(&[b, z, h_grid, patch, w_grid, patch])?;
    let v = v.permute(&[0, 2, 4, 1, 3, 5])?.contiguous()?;
    Ok(v.reshape(&[b, h_grid * w_grid, z * patch * patch])?)
}

/// `(B, S_img, Z * P²)` -> `(B, Z, 1, H, W)`. Matches diffusers `_unpack_latents`
/// but reads the unpacked dims from `(h_grid, w_grid, patch)`.
fn unpack_latents(
    latents: &Tensor,
    h_grid: usize,
    w_grid: usize,
    patch: usize,
) -> Result<Tensor> {
    let dims = latents.shape().dims();
    let (b, _s, channels) = (dims[0], dims[1], dims[2]);
    let z = channels / (patch * patch);
    let v = latents.reshape(&[b, h_grid, w_grid, z, patch, patch])?;
    let v = v.permute(&[0, 3, 1, 4, 2, 5])?.contiguous()?;
    let h = h_grid * patch;
    let w = w_grid * patch;
    Ok(v.reshape(&[b, z, 1, h, w])?)
}

/// Apply VAE latents denormalization. Diffusers (pipeline_nucleusmoe_image.py:627-635):
/// ```python
/// latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)...   # NOTE: inverse
/// latents = latents / latents_std + latents_mean
/// ```
/// Reassigning `latents_std` to `1 / std` first means `latents / latents_std`
/// is actually `latents * std`. Net formula: `latents * std + mean`.
fn denormalize_latents(latents: &Tensor, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let std = Tensor::from_vec(
        VAE_LATENTS_STD.to_vec(),
        Shape::from_dims(&[1, Z_DIM, 1, 1, 1]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let mean = Tensor::from_vec(
        VAE_LATENTS_MEAN.to_vec(),
        Shape::from_dims(&[1, Z_DIM, 1, 1, 1]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let scaled = latents.mul(&std)?;
    Ok(scaled.add(&mean)?)
}

/// `‖x‖_2` along last dim, keepdim. F32 accumulation for numerical safety,
/// returns BF16. Used by the CFG-Zero* norm rescaling step.
fn norm_along_last(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.mul(&x_f32)?;
    let last = sq.shape().dims().len() - 1;
    let sum_sq = sq.sum_dim_keepdim(last)?;
    let norm = sum_sq.sqrt()?;
    Ok(norm.to_dtype(DType::BF16)?)
}

// ---------------------------------------------------------------------------
// PNG output
// ---------------------------------------------------------------------------

/// `image_tensor`: `[B, 3, 1, H, W]` BF16/F32 in `[-1, 1]`. Saves the first
/// batch element as PNG.
fn save_png(
    image_tensor: &Tensor,
    out_dir: &Path,
    seed: u64,
    steps: usize,
    width: usize,
    height: usize,
    guidance: f32,
) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let dims = image_tensor.shape().dims();
    let (_b, c, _f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    if c != 3 {
        return Err(anyhow!("expected 3 channels, got {c}"));
    }

    let data_f32 = image_tensor.to_vec_f32()?;
    // Layout: (B, 3, 1, H, W) row-major. Skip B index 0, F index 0.
    let plane_size = h * w;
    let mut rgb: Vec<u8> = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let idx = ch * plane_size + y * w + x;
                let v = data_f32[idx];
                // [-1, 1] -> [0, 255]
                let scaled = (v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0;
                rgb.push(scaled.round() as u8);
            }
        }
    }

    let path = out_dir.join(format!(
        "nucleus_seed{seed}_{steps}steps_{width}x{height}_cfg{:.1}.png",
        guidance
    ));
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("RgbImage::from_raw"))?;
    img.save(&path)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse()?;
    env_logger::try_init().ok();

    eprintln!("=== Nucleus-Image T2I (pure Rust) ===");
    eprintln!("prompt: {:?}", args.prompt);
    eprintln!(
        "seed={}  steps={}  size={}x{}  guidance={}",
        args.seed, args.steps, args.width, args.height, args.guidance
    );

    let do_cfg = args.guidance > 1.0;
    eprintln!("CFG: {}", if do_cfg { "on (encoder + DiT each run twice/step)" } else { "off" });

    let device: Arc<CudaDevice> = CudaDevice::new(0)?;

    // ---- Phase 1: tokenize + encode -------------------------------------
    let t0 = Instant::now();
    let token_ids = tokenize_prompt(&args.prompt)?;
    eprintln!("tokens: {}", token_ids.len());

    let prompt_embeds = encode_prompt(&device, &token_ids)?;
    eprintln!(
        "encoded(cond) in {:.1}s; shape = {:?}",
        t0.elapsed().as_secs_f32(),
        prompt_embeds.shape().dims()
    );

    // CFG: encode the empty negative prompt too. Diffusers default behavior:
    //   if do_cfg and not has_neg_prompt: negative_prompt = [""] * batch_size
    let neg_embeds = if do_cfg {
        let t_neg = Instant::now();
        let neg_ids = tokenize_prompt("")?;
        eprintln!("neg tokens: {}", neg_ids.len());
        let e = encode_prompt(&device, &neg_ids)?;
        eprintln!(
            "encoded(uncond) in {:.1}s; shape = {:?}",
            t_neg.elapsed().as_secs_f32(),
            e.shape().dims()
        );
        Some(e)
    } else {
        None
    };

    // Flush the pool so the encoder's ~17 GB BF16 weight cache returns to
    // CUDA before we start allocating the DiT's resident weights and the
    // BlockOffloader's prefetch slots. Without this, OOM hits on the first
    // MoE prefetch.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    eprintln!("[pool] cleared after encoder drop");

    // ---- Phase 2: load DiT + VAE ----------------------------------------
    let t1 = Instant::now();
    let transformer_dir = Path::new(SNAPSHOT).join("transformer");
    let mut dit = NucleusInferDit::load(&transformer_dir, device.clone())
        .map_err(|e| anyhow!("NucleusInferDit::load: {e}"))?;
    eprintln!("DiT loaded in {:.1}s", t1.elapsed().as_secs_f32());

    // ---- Phase 3: latent shape + KV cache + sigma schedule --------------
    let h_lat = args.height / VAE_SCALE_FACTOR; // 512 → 64
    let w_lat = args.width / VAE_SCALE_FACTOR; // 512 → 64
    let h_grid = h_lat / PATCH_SIZE; // 32
    let w_grid = w_lat / PATCH_SIZE; // 32
    let s_img = h_grid * w_grid;
    eprintln!(
        "latent: h_lat={h_lat} w_lat={w_lat} h_grid={h_grid} w_grid={w_grid} S_img={s_img}"
    );

    // Initial Gaussian noise: (B, Z, 1, h_lat, w_lat) in F32, then pack + cast to BF16.
    let latent_unpacked = Tensor::randn_seeded(
        Shape::from_dims(&[1, Z_DIM, h_lat, w_lat]),
        0.0,
        1.0,
        args.seed,
        device.clone(),
    )
    .map_err(|e| anyhow!("randn: {e}"))?;
    let mut latents = pack_latents(&latent_unpacked, h_grid, w_grid, PATCH_SIZE)
        .map_err(|e| anyhow!("pack_latents: {e}"))?
        .to_dtype(DType::BF16)
        .map_err(|e| anyhow!("to_bf16: {e}"))?;

    let img_shapes = (1, h_grid, w_grid);
    let kv_cache = dit
        .compute_kv_cache(&prompt_embeds, img_shapes)
        .map_err(|e| anyhow!("compute_kv_cache(cond): {e}"))?;
    let uncond_kv_cache = if let Some(ref e) = neg_embeds {
        Some(
            dit.compute_kv_cache(e, img_shapes)
                .map_err(|err| anyhow!("compute_kv_cache(uncond): {err}"))?,
        )
    } else {
        None
    };
    eprintln!(
        "KV cache built (cond{})",
        if uncond_kv_cache.is_some() { " + uncond" } else { "" }
    );

    // Sigma schedule: linspace(1.0, 1/N, N) appended with 0.
    let n = args.steps;
    let mut sigmas: Vec<f32> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = i as f32 / (n - 1).max(1) as f32;
        // linspace from 1.0 to 1/N
        let s = 1.0 - t * (1.0 - 1.0 / n as f32);
        sigmas.push(s);
    }
    sigmas.push(0.0);
    eprintln!(
        "sigmas: {:.4}..{:.4}..{:.4}",
        sigmas[0],
        sigmas[n / 2],
        sigmas[n - 1]
    );

    // ---- Phase 4: denoise loop ------------------------------------------
    let t_loop = Instant::now();
    for i in 0..n {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Diffusers passes `timestep / num_train_timesteps = sigma` to the DiT.
        let timestep = Tensor::from_vec(
            vec![sigma],
            Shape::from_dims(&[1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let velocity_cond = dit
            .forward_cached(&latents, &timestep, img_shapes, &kv_cache)
            .map_err(|e| anyhow!("forward_cached(cond) step {i}: {e}"))?;

        // CFG (`guidance > 1`): also run uncond and combine. Pipeline:
        //     comb = uncond + guidance * (cond - uncond)
        //     comb = comb * (||cond|| / ||comb||)        # CFG-Zero* per-token
        //     noise_pred = -comb
        //     latents = latents + (sigma_next - sigma) * noise_pred
        let velocity = if let Some(ref ukv) = uncond_kv_cache {
            let velocity_uncond = dit
                .forward_cached(&latents, &timestep, img_shapes, ukv)
                .map_err(|e| anyhow!("forward_cached(uncond) step {i}: {e}"))?;
            let diff = velocity_cond.sub(&velocity_uncond)?;
            let scaled_diff = diff.mul_scalar(args.guidance)?;
            let comb = velocity_uncond.add(&scaled_diff)?;
            let cond_norm = norm_along_last(&velocity_cond)?;
            let comb_norm = norm_along_last(&comb)?;
            let ratio = cond_norm.div(&comb_norm)?;
            comb.mul(&ratio)?
        } else {
            velocity_cond
        };

        // latents = latents + (sigma - sigma_next) * velocity
        let dt = sigma - sigma_next; // positive for decreasing schedule
        let dv = velocity.mul_scalar(dt)?;
        latents = latents.add(&dv)?;

        eprintln!(
            "step {:>3}/{:<3}  sigma={:.4} → {:.4}   t_loop={:.1}s",
            i + 1,
            n,
            sigma,
            sigma_next,
            t_loop.elapsed().as_secs_f32()
        );
    }
    eprintln!("denoise loop: {:.1}s total", t_loop.elapsed().as_secs_f32());

    // Drop DiT to free its resident weights + offloader pinned RAM before
    // VAE decode runs (VAE upsample blocks need a bit of headroom at 512²).
    drop(dit);
    drop(kv_cache);
    drop(uncond_kv_cache);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    eprintln!("[pool] cleared after DiT drop");

    // ---- Phase 5: load VAE, unpack + denormalize + decode + save --------
    let vae_path = Path::new(SNAPSHOT)
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    let vae = QwenImageVaeDecoder::from_safetensors(
        vae_path.to_str().ok_or_else(|| anyhow!("non-utf8 vae path"))?,
        &device,
    )
    .map_err(|e| anyhow!("VAE load: {e}"))?;
    eprintln!("VAE loaded");

    let unpacked = unpack_latents(&latents, h_grid, w_grid, PATCH_SIZE)
        .map_err(|e| anyhow!("unpack_latents: {e}"))?;
    eprintln!("unpacked latent shape: {:?}", unpacked.shape().dims());

    // NB: `Wan21VaeDecoder::decode_image` (via QwenImageVaeDecoder) DOES the
    // `z * std + mean` denormalization internally — so we must NOT denormalize
    // externally here, even though the diffusers pipeline does. Doing both is
    // a double denorm and was the source of the blurry/oversaturated output
    // before this fix.
    let image = vae.decode(&unpacked).map_err(|e| anyhow!("VAE decode: {e}"))?;
    eprintln!("image shape (pre-save): {:?}", image.shape().dims());

    drop(vae);

    let out_path = save_png(
        &image,
        &args.out_dir,
        args.seed,
        args.steps,
        args.width,
        args.height,
        args.guidance,
    )?;
    eprintln!("saved: {}", out_path.display());

    Ok(())
}
