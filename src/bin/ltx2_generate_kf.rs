//! LTX-2.3 audio+video generation with multi-keyframe conditioning
//! and video extension support.
//!
//! Usage:
//!
//!     cargo run --release --bin ltx2_generate_kf -- \
//!         --prompt "..." \
//!         --cond image0.png:0:1.0 \
//!         --cond image1.png:8:0.8 \
//!         --cond image2.png:16:1.0 \
//!         [--cond-noise-scale 0.15] \
//!         [--cfg 1.0 --stg 0.0 ...]
//!
//! Each `--cond` is `PATH:FRAME_NUMBER:STRENGTH`:
//!   * `PATH` — a PNG/JPG/JPEG image (landing 1 is still-image only — video
//!     clips as keyframes are deferred; the Python side would require a
//!     video decoder which we've kept out of the binary).
//!   * `FRAME_NUMBER` — start frame in the OUTPUT video. Must be 0 or a
//!     multiple of 8 (Lightricks invariant:
//!     `/tmp/ltx-video/ltx_video/pipelines/pipeline_ltx_video.py:1452,1686`).
//!   * `STRENGTH` — 1.0 means hard conditioning, 0.0 means ref has no
//!     effect, in-between means interpolate.
//!
//! The audio stream is always generated — we call the mask-aware variant
//! of `forward_audio_video` so the video mask gates ONLY the video
//! timestep; the audio timestep runs normally.
//!
//! Supported / deferred cases are documented in
//! `src/models/ltx2_conditioning.rs` module docs.

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::feature_extractor;
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::lora_loader::LoraWeights;
use inference_flame::models::ltx2_conditioning::{
    add_image_cond_noise, pack_conditioning_mask_for_transformer, prepare_conditioning,
    ConditioningItem,
};
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::sampling::ltx2_guidance::{cfg_star_rescale, stg_rescale};
use inference_flame::sampling::ltx2_sampling::LTX2_DISTILLED_SIGMAS;
use inference_flame::vae::LTX2VaeEncoder;
use std::collections::HashMap;
use std::time::Instant;

// -- Model paths, matching ltx2_generate_av.rs ------------------------------
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
const LTX_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const VAE_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const GEMMA_ROOT: &str =
    "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const OUTPUT_DIR: &str =
    "/home/alex/EriDiffusion/inference-flame/output";

const DEFAULT_NEGATIVE: &str =
    "worst quality, inconsistent motion, blurry, jittery, distorted";

// -- Sensible defaults; override with --prompt, --width, --height, --frames --
const DEFAULT_PROMPT: &str =
    "A close-up of a person's face turning to camera in a dim room, light flickering.";
const DEFAULT_NUM_FRAMES: usize = 33; // 8*4+1 = 33 frames → 1.32s at 25fps
const DEFAULT_WIDTH: usize = 480;
const DEFAULT_HEIGHT: usize = 288;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const AUDIO_CHANNELS: usize = 8;
const AUDIO_MEL_BINS: usize = 16;
const NUM_STEPS: usize = 8;
const DEFAULT_COND_NOISE_SCALE: f32 = 0.15; // pipeline_ltx_video.py inference.py:365-368

// ===========================================================================
// CLI parsing
// ===========================================================================

/// Parse `--cond PATH:FRAME:STRENGTH` occurrences. Tolerates absolute paths
/// containing `:` via `rsplitn` from the right.
fn collect_cond_args() -> anyhow::Result<Vec<(String, usize, f32)>> {
    let mut out = Vec::new();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let (take_val, val_idx) = if args[i] == "--cond" {
            (true, i + 1)
        } else if args[i].starts_with("--cond=") {
            (false, i)
        } else {
            i += 1;
            continue;
        };
        let raw = if take_val {
            args.get(val_idx)
                .ok_or_else(|| anyhow::anyhow!("--cond requires PATH:FRAME:STRENGTH"))?
                .clone()
        } else {
            args[i]["--cond=".len()..].to_string()
        };
        // Split from right: last ':' is STRENGTH, second-last is FRAME.
        // Paths on Linux can contain ':' though it's unusual.
        let parts: Vec<&str> = raw.rsplitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!("--cond must be PATH:FRAME:STRENGTH, got {raw:?}");
        }
        // rsplitn reverses: parts = [STRENGTH, FRAME, PATH]
        let strength: f32 = parts[0]
            .parse()
            .map_err(|e| anyhow::anyhow!("--cond: bad strength {:?}: {e}", parts[0]))?;
        let frame: usize = parts[1]
            .parse()
            .map_err(|e| anyhow::anyhow!("--cond: bad frame {:?}: {e}", parts[1]))?;
        let path = parts[2].to_string();
        if !(0.0..=1.0).contains(&strength) {
            anyhow::bail!(
                "--cond: strength {strength} out of [0, 1] for path {path:?}"
            );
        }
        if frame != 0 && frame % 8 != 0 {
            anyhow::bail!(
                "--cond: frame_number {frame} must be 0 or a multiple of 8 (path {path:?})"
            );
        }
        out.push((path, frame, strength));
        i = if take_val { val_idx + 1 } else { i + 1 };
    }
    Ok(out)
}

fn get_arg<T: std::str::FromStr>(flag: &str) -> Option<T> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            return args.get(i + 1).and_then(|s| s.parse().ok());
        } else if let Some(val) = args[i].strip_prefix(&format!("{flag}=")) {
            return val.parse().ok();
        }
        i += 1;
    }
    None
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

fn collect_str_arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut out = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            out = args.get(i + 1).cloned();
            i += 2;
        } else if let Some(v) = args[i].strip_prefix(&format!("{flag}=")) {
            out = Some(v.to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

fn parse_stg_blocks() -> Vec<usize> {
    collect_str_arg("--stg-blocks")
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn parse_lora_arg(raw: &str) -> (String, f32) {
    if let Some(idx) = raw.rfind(':') {
        let (path, tail) = raw.split_at(idx);
        let tail = &tail[1..];
        if let Ok(s) = tail.parse::<f32>() {
            return (path.to_string(), s);
        }
    }
    (raw.to_string(), 1.0)
}

fn collect_lora_args() -> Vec<(String, f32)> {
    let mut out = Vec::new();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--lora" {
            if let Some(v) = args.get(i + 1) {
                out.push(parse_lora_arg(v));
                i += 2;
                continue;
            }
        } else if let Some(v) = args[i].strip_prefix("--lora=") {
            out.push(parse_lora_arg(v));
        }
        i += 1;
    }
    out
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Load image → `[1, 3, 1, H, W]` BF16 tensor in `[-1, 1]`.
fn load_image_to_video_tensor(
    path: &str,
    target_h: usize,
    target_w: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("failed to open image {path:?}: {e}"))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let mut data = vec![0.0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * target_h * target_w + y * target_w + x] =
                    (p[c] as f32) / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_f32_to_bf16(
        data,
        Shape::from_dims(&[1, 3, 1, target_h, target_w]),
        device.clone(),
    )?)
}

fn make_noise(
    numel: usize,
    seed: u64,
    shape: &[usize],
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<Tensor> {
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
    Ok(Tensor::from_f32_to_bf16(
        v,
        Shape::from_dims(shape),
        device.clone(),
    )?)
}

fn simple_tokenize(text: &str, max_len: usize) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
    let tok_path = format!("{GEMMA_ROOT}/tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer load ({tok_path}): {e}"))?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode: {e}"))?;
    let raw_ids: &[u32] = encoding.get_ids();
    let ids: Vec<u32> = if raw_ids.len() > max_len {
        eprintln!(
            "  [tokenize] prompt encoded to {} tokens, truncating to {}",
            raw_ids.len(),
            max_len
        );
        raw_ids[..max_len].to_vec()
    } else {
        raw_ids.to_vec()
    };
    let real_len = ids.len();
    let pad = max_len - real_len;
    let mut input_ids: Vec<i32> = vec![0i32; pad];
    input_ids.extend(ids.iter().map(|&id| id as i32));
    let mut attention_mask: Vec<i32> = vec![0i32; pad];
    attention_mask.extend(std::iter::repeat(1i32).take(real_len));
    Ok((input_ids, attention_mask))
}

// ===========================================================================
// Main
// ===========================================================================

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // --- Parse CLI ---
    let cond_specs = collect_cond_args()?;
    let lora_specs = collect_lora_args();
    let prompt: String =
        collect_str_arg("--prompt").unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let width: usize = get_arg("--width").unwrap_or(DEFAULT_WIDTH);
    let height: usize = get_arg("--height").unwrap_or(DEFAULT_HEIGHT);
    let num_frames: usize = get_arg("--frames").unwrap_or(DEFAULT_NUM_FRAMES);
    let cfg_scale: f32 = get_arg("--cfg").unwrap_or(1.0);
    let stg_scale: f32 = get_arg("--stg").unwrap_or(0.0);
    let stg_rescale_factor: f32 = get_arg("--stg-rescale").unwrap_or(0.7);
    let cfg_star = has_flag("--cfg-star-rescale");
    let stg_blocks = parse_stg_blocks();
    let cond_noise_scale: f32 =
        get_arg("--cond-noise-scale").unwrap_or(DEFAULT_COND_NOISE_SCALE);
    let neg_text =
        collect_str_arg("--neg").unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());

    let do_cfg = cfg_scale > 1.0;
    let do_stg = stg_scale > 0.0 && !stg_blocks.is_empty();

    println!("============================================================");
    println!("LTX-2.3 Audio+Video Generation with Multi-Keyframe / Extension");
    println!("============================================================");
    println!(
        "  {}x{}, {} frames @ {}fps, {} steps",
        width, height, num_frames, FRAME_RATE, NUM_STEPS
    );
    println!(
        "  cfg={:.2} ({})  stg={:.2} ({})  cfg_star={}",
        cfg_scale,
        if do_cfg { "on" } else { "off" },
        stg_scale,
        if do_stg { "on" } else { "off" },
        cfg_star
    );
    println!(
        "  --cond items: {} (cond_noise_scale={})",
        cond_specs.len(),
        cond_noise_scale
    );
    for (p, f, s) in &cond_specs {
        println!("    * {p} @ frame {f}, strength {s}");
    }
    if cond_specs.is_empty() {
        println!("  (no conditioning items → behaves like T2V)");
    }

    let device = global_cuda_device();

    // Constraint checks aligned with LTX-Video's shape invariants.
    if (width % 32) != 0 || (height % 32) != 0 {
        anyhow::bail!("width/height must be multiples of 32 (vae_scale_factor=32)");
    }
    if (num_frames - 1) % 8 != 0 {
        anyhow::bail!(
            "num_frames must satisfy (num_frames - 1) % 8 == 0 (e.g. 9, 17, 25, 33, ...)"
        );
    }
    let latent_f = ((num_frames - 1) / 8) + 1;
    let latent_h = height / 32;
    let latent_w = width / 32;
    let num_video_tokens = latent_f * latent_h * latent_w;

    // Audio frames: same formula as ltx2_generate_av.rs.
    let video_duration = num_frames as f32 / FRAME_RATE;
    let audio_frames = (video_duration * 25.0).round() as usize;

    println!(
        "  Video latent: [{}, {}, {}, {}] = {} tokens",
        LATENT_CHANNELS, latent_f, latent_h, latent_w, num_video_tokens
    );
    println!(
        "  Audio latent: [{}, {}, {}]",
        AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS
    );

    // ========================================================================
    // Stage 1: VAE-encode conditioning images BEFORE loading the transformer
    //          so the VAE weights don't stay resident alongside the DiT.
    // ========================================================================
    let cond_items: Vec<ConditioningItem> = if cond_specs.is_empty() {
        Vec::new()
    } else {
        println!("\n--- Stage 1a: VAE-encode conditioning images ---");
        let t0 = Instant::now();
        let vae = LTX2VaeEncoder::from_file(VAE_CHECKPOINT, &device)?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let mut items = Vec::with_capacity(cond_specs.len());
        for (path, frame, strength) in &cond_specs {
            println!("  Loading {path} @ frame {frame}, strength {strength}");
            let video = load_image_to_video_tensor(path, height, width, &device)?;
            // [1, 3, 1, H, W] → [1, 128, 1, H/32, W/32].
            let latent = vae.encode(&video)?;
            let ldims = latent.shape().dims().to_vec();
            println!("    encoded latent: {:?}", ldims);
            if ldims[3] != latent_h || ldims[4] != latent_w {
                anyhow::bail!(
                    "conditioning latent spatial {:?} != target {}x{}",
                    (ldims[3], ldims[4]),
                    latent_h,
                    latent_w
                );
            }
            items.push(ConditioningItem {
                latent,
                frame_number: *frame,
                strength: *strength,
            });
        }
        drop(vae);
        let _ = device.synchronize();
        println!("  VAE encode done in {:.1}s", t0.elapsed().as_secs_f32());
        items
    };

    // ========================================================================
    // Stage 1b: Text encoding (Gemma → FeatureExtractor), as in ltx2_generate_av
    // ========================================================================
    println!("\n--- Stage 1b: Text encoding ---");
    let t0 = Instant::now();
    let mut shards: Vec<String> = Vec::new();
    for i in 1..=5 {
        let path = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
        if std::path::Path::new(&path).exists() {
            shards.push(path);
        }
    }
    let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();

    let (input_ids, attention_mask) = simple_tokenize(&prompt, 256)?;
    let mut encoder = Gemma3Encoder::load(&shard_refs, &device, input_ids.len())?;
    let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;

    // Load feature-extractor projections from the ltx2-dev checkpoint.
    let agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT),
        &device,
        |k| k.starts_with("text_embedding_projection.video_aggregate_embed"),
    )?;
    let agg_w = agg_weights
        .get("text_embedding_projection.video_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing video_aggregate_embed.weight"))?;
    let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");

    let audio_agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT),
        &device,
        |k| k.starts_with("text_embedding_projection.audio_aggregate_embed"),
    )?;
    let audio_agg_w = audio_agg_weights
        .get("text_embedding_projection.audio_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing audio_aggregate_embed.weight"))?;
    let audio_agg_b =
        audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.bias");

    let video_context = feature_extractor::feature_extract_and_project(
        &all_hidden, &mask_out, agg_w, agg_b, 4096,
    )?;
    let audio_context = feature_extractor::feature_extract_and_project(
        &all_hidden,
        &mask_out,
        audio_agg_w,
        audio_agg_b,
        2048,
    )?;
    drop(all_hidden);
    drop(mask_out);

    // Negative prompt encoding if CFG enabled.
    let (neg_video_context, neg_audio_context) = if do_cfg {
        let (nids, nmask) = simple_tokenize(&neg_text, 256)?;
        let (nh, nm) = encoder.encode(&nids, &nmask)?;
        let nv = feature_extractor::feature_extract_and_project(&nh, &nm, agg_w, agg_b, 4096)?;
        let na = feature_extractor::feature_extract_and_project(
            &nh, &nm, audio_agg_w, audio_agg_b, 2048,
        )?;
        drop(nh);
        drop(nm);
        (Some(nv), Some(na))
    } else {
        (None, None)
    };

    drop(encoder);
    drop(agg_weights);
    drop(audio_agg_weights);
    let _ = device.synchronize();
    unsafe {
        extern "C" {
            fn cudaMemPoolTrimTo(pool: *mut std::ffi::c_void, min_bytes: usize) -> i32;
            fn cudaDeviceGetDefaultMemPool(pool: *mut *mut std::ffi::c_void, device: i32) -> i32;
        }
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let _ = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        if !pool.is_null() {
            let _ = cudaMemPoolTrimTo(pool, 0);
        }
    }
    println!(
        "  Text encoding total: {:.1}s (VRAM freed)",
        t0.elapsed().as_secs_f32()
    );

    // ========================================================================
    // Stage 2: Load LTX-2 transformer
    // ========================================================================
    println!("\n--- Stage 2: Load transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    println!(
        "  Global params loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    for (p, s) in &lora_specs {
        println!("  LoRA: {p} @ strength={s:.3}");
        let lora = LoraWeights::load(p, *s, &device)?;
        model.add_lora(lora);
    }
    if !lora_specs.is_empty() {
        model.init_offloader()?;
        println!(
            "  BlockOffloader initialized (LoRA path) in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
    } else {
        match model.load_fp8_resident() {
            Ok(()) => println!(
                "  FP8 resident loaded in {:.1}s",
                t0.elapsed().as_secs_f32()
            ),
            Err(e) => {
                println!("  FP8 resident failed ({e}), falling back to BlockOffloader");
                model.init_offloader()?;
                println!(
                    "  BlockOffloader initialized in {:.1}s",
                    t0.elapsed().as_secs_f32()
                );
            }
        }
    }

    // ========================================================================
    // Stage 3: Prepare noise, run conditioning, build mask
    // ========================================================================
    println!("\n--- Stage 3: Prepare noise + conditioning ---");

    let video_numel = LATENT_CHANNELS * latent_f * latent_h * latent_w;
    let video_noise = make_noise(
        video_numel,
        SEED,
        &[1, LATENT_CHANNELS, latent_f, latent_h, latent_w],
        &device,
    )?;
    let audio_numel = AUDIO_CHANNELS * audio_frames * AUDIO_MEL_BINS;
    let audio_noise = make_noise(
        audio_numel,
        SEED + 1,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS],
        &device,
    )?;

    // prepare_conditioning: start from pure noise (init_latents), merge refs.
    let (merged_latents, cond_mask_5d) = prepare_conditioning(&video_noise, &cond_items)?;
    let cond_mask_packed = if cond_items.is_empty() {
        None
    } else {
        Some(pack_conditioning_mask_for_transformer(&cond_mask_5d)?)
    };

    // Snapshot init_latents for the per-step add_image_cond_noise refresh.
    // This is the value re-seeded at conditioned positions every step;
    // matches Lightricks's `init_latents` parameter
    // (pipeline_ltx_video.py:1151-1159).
    let init_latents_snapshot = merged_latents.clone();

    let sigmas = LTX2_DISTILLED_SIGMAS.to_vec();
    println!("  Merged video latents: {:?}", merged_latents.dims());
    println!("  Audio noise:          {:?}", audio_noise.dims());
    println!(
        "  Conditioning mask:    {} tokens conditioned of {} total",
        // count ones in the packed mask for a quick sanity readout
        cond_mask_packed
            .as_ref()
            .and_then(|t| t.to_dtype(DType::F32).ok().and_then(|x| x.to_vec().ok()))
            .map(|v| v.iter().filter(|&&x| x > 0.5).count())
            .unwrap_or(0),
        num_video_tokens
    );

    let forwards_per_step = 1 + if do_cfg { 1 } else { 0 } + if do_stg { 1 } else { 0 };

    // ========================================================================
    // Stage 4: Denoise
    // ========================================================================
    println!(
        "\n--- Stage 4: Denoise ({} steps, AV joint, {} fwd/step) ---",
        NUM_STEPS, forwards_per_step
    );
    let t0 = Instant::now();
    let mut video_x = merged_latents;
    let mut audio_x = audio_noise;

    for step in 0..NUM_STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        // Re-inject noise on hard-conditioned positions (Lightricks's
        // image_cond_noise_scale). Uses `sigma` (fractional t in [0,1])
        // as the temporal factor `t` in the formula.
        if !cond_items.is_empty() && cond_noise_scale > 0.0 {
            let noise_for_step = make_noise(
                video_numel,
                SEED.wrapping_add(0x5AFE_BEEF).wrapping_add(step as u64),
                &[1, LATENT_CHANNELS, latent_f, latent_h, latent_w],
                &device,
            )?;
            video_x = add_image_cond_noise(
                &init_latents_snapshot,
                &video_x,
                &cond_mask_5d,
                &noise_for_step,
                sigma,
                cond_noise_scale,
            )?;
        }

        let sigma_t =
            Tensor::from_f32_to_bf16(vec![sigma], Shape::from_dims(&[1]), device.clone())?;

        // -- Positive (cond) forward --
        let (video_cond, audio_cond) = model.forward_audio_video_with_mask(
            &video_x,
            &audio_x,
            &sigma_t,
            &video_context,
            &audio_context,
            FRAME_RATE,
            None,
            None,
            None, // no STG on the conditional pass
            cond_mask_packed.as_ref(),
        )?;

        // -- CFG --
        let (video_vel, audio_vel) = if do_cfg {
            let nvc = neg_video_context.as_ref().unwrap();
            let nac = neg_audio_context.as_ref().unwrap();
            let (mut video_uncond, mut audio_uncond) = model.forward_audio_video_with_mask(
                &video_x,
                &audio_x,
                &sigma_t,
                nvc,
                nac,
                FRAME_RATE,
                None,
                None,
                None,
                cond_mask_packed.as_ref(),
            )?;
            if cfg_star {
                video_uncond = cfg_star_rescale(&video_cond, &video_uncond)?;
                audio_uncond = cfg_star_rescale(&audio_cond, &audio_uncond)?;
            }
            let v_guided = video_uncond
                .add(&video_cond.sub(&video_uncond)?.mul_scalar(cfg_scale)?)?;
            let a_guided = audio_uncond
                .add(&audio_cond.sub(&audio_uncond)?.mul_scalar(cfg_scale)?)?;
            (v_guided, a_guided)
        } else {
            (video_cond.clone(), audio_cond.clone())
        };

        // -- STG --
        let (video_vel, audio_vel) = if do_stg {
            let (video_pert, audio_pert) = model.forward_audio_video_with_mask(
                &video_x,
                &audio_x,
                &sigma_t,
                &video_context,
                &audio_context,
                FRAME_RATE,
                None,
                None,
                Some(&stg_blocks),
                cond_mask_packed.as_ref(),
            )?;
            let v_out = video_vel
                .add(&video_cond.sub(&video_pert)?.mul_scalar(stg_scale)?)?;
            let a_out = audio_vel
                .add(&audio_cond.sub(&audio_pert)?.mul_scalar(stg_scale)?)?;
            if stg_rescale_factor != 1.0 {
                let v_resc = stg_rescale(&video_cond, &v_out, stg_rescale_factor)?;
                let a_resc = stg_rescale(&audio_cond, &a_out, stg_rescale_factor)?;
                (v_resc, a_resc)
            } else {
                (v_out, a_out)
            }
        } else {
            (video_vel, audio_vel)
        };

        // Euler step for video + audio.
        if sigma_next == 0.0 {
            video_x = video_x.sub(&video_vel.mul_scalar(sigma)?)?;
            audio_x = audio_x.sub(&audio_vel.mul_scalar(sigma)?)?;
        } else {
            let dt = sigma_next - sigma;
            video_x = video_x.add(&video_vel.mul_scalar(dt)?)?;
            audio_x = audio_x.add(&audio_vel.mul_scalar(dt)?)?;
        }

        let dt_step = t_step.elapsed().as_secs_f32();
        println!(
            "  Step {}/{} sigma={:.4} dt={:.1}s",
            step + 1,
            NUM_STEPS,
            sigma,
            dt_step
        );
    }

    let dt = t0.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.1}s/step)",
        dt,
        dt / NUM_STEPS as f32
    );

    // ========================================================================
    // Stage 5: Save latents
    // ========================================================================
    println!("\n--- Stage 5: Save latents ---");
    let video_path = format!("{OUTPUT_DIR}/ltx2_kf_video_latents.safetensors");
    let audio_path = format!("{OUTPUT_DIR}/ltx2_kf_audio_latents.safetensors");
    let mut video_save = HashMap::new();
    video_save.insert("latents".to_string(), video_x);
    flame_core::serialization::save_tensors(
        &video_save,
        std::path::Path::new(&video_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Video latents: {video_path}");
    let mut audio_save = HashMap::new();
    audio_save.insert("latents".to_string(), audio_x);
    flame_core::serialization::save_tensors(
        &audio_save,
        std::path::Path::new(&audio_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Audio latents: {audio_path}");

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
