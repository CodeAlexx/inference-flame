//! LTX-2.3 video generation — pure Rust, with CFG.
//!
//! Pipeline:
//! 1. Load cached positive text embeddings (4096-dim video_context).
//! 2. If CFG is enabled (`--cfg SCALE` with SCALE > 1):
//!    a. Load Gemma-3 + projection weights
//!    b. Encode the negative prompt to a 4096-dim video_context
//!    c. Drop Gemma to reclaim VRAM for the DiT
//! 3. Load LTX-2 transformer via BlockOffloader / FP8 resident
//! 4. Create noise + sigma schedule (distilled or dev)
//! 5. Denoise; when CFG is on, two forward passes per step (uncond + cond)
//! 6. Save denoised latents → decode with Python VAE
//!
//! Flags:
//!   --lora PATH[:STRENGTH]         attach one or more LoRAs (fusion path)
//!   --cfg SCALE                    override the default guidance scale
//!                                    (default 1.0, distilled)
//!   --neg "TEXT"                   negative prompt text (default:
//!                                    Lightricks canonical). Requires
//!                                    running Gemma-3 in-process.
//!   --neg-embeds PATH              pre-computed neg embedding file
//!                                    (tensor key: any; first value used)
//!                                    — overrides `--neg TEXT`.
//!
//! When CFG is on AND `--neg-embeds` is NOT given, this binary runs the
//! same Gemma-3 + FeatureExtractor pipeline as `ltx2_generate_av` (just
//! the video half — audio context is only needed by the AV binary).

use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;
use inference_flame::models::lora_loader::LoraWeights;
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::sampling::ltx2_sampling::{build_dev_sigma_schedule, LTX2_DISTILLED_SIGMAS};
use flame_core::{global_cuda_device, Shape, Tensor};
use std::time::Instant;

// Use dev-fp8 for testing FP8 resident path
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors";
const LTX_CHECKPOINT_FULL: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const EMBEDDINGS_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors";
const OUTPUT_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_denoised_latents.safetensors";
const GEMMA_ROOT: &str =
    "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";

/// Lightricks-canonical default negative prompt.
///
/// Source: `/tmp/ltx-video/ltx_video/inference.py:351-354` (checked in by
/// the LTX-Video authors). If this is edited, also update
/// `scripts/ltx2_neg_prompt_ref.py` and `src/bin/ltx2_neg_prompt_parity.rs`.
const DEFAULT_NEGATIVE: &str =
    "worst quality, inconsistent motion, blurry, jittery, distorted";

const NUM_FRAMES: usize = 9;
const WIDTH: usize = 480;
const HEIGHT: usize = 288;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const DEFAULT_GUIDANCE_SCALE: f32 = 1.0; // Distilled default: no CFG
const NUM_STEPS: usize = 8;              // Distilled fixed steps

/// Parse a single `--lora` value of the form `PATH[:STRENGTH]`.
/// Default strength is 1.0 when omitted. Supports paths with no colon
/// (e.g. `/foo/bar.safetensors`) as well as `path:0.75`.
fn parse_lora_arg(raw: &str) -> anyhow::Result<(String, f32)> {
    // Find the LAST ':' so paths that happen to contain ':' (rare on Linux)
    // still work when the user supplied a strength.
    if let Some(idx) = raw.rfind(':') {
        let (path, tail) = raw.split_at(idx);
        let tail = &tail[1..]; // skip ':'
        if let Ok(s) = tail.parse::<f32>() {
            return Ok((path.to_string(), s));
        }
    }
    Ok((raw.to_string(), 1.0))
}

/// Collect all `--lora PATH[:STRENGTH]` pairs from `std::env::args`.
fn collect_lora_args() -> anyhow::Result<Vec<(String, f32)>> {
    let mut out = Vec::new();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--lora" {
            let val = args.get(i + 1).ok_or_else(||
                anyhow::anyhow!("--lora requires a PATH[:STRENGTH] argument"))?;
            out.push(parse_lora_arg(val)?);
            i += 2;
        } else if let Some(val) = args[i].strip_prefix("--lora=") {
            out.push(parse_lora_arg(val)?);
            i += 1;
        } else {
            i += 1;
        }
    }
    Ok(out)
}

/// Read a single `--flag VALUE` / `--flag=VALUE`, parse as T, last-wins.
fn get_arg<T: std::str::FromStr>(flag: &str) -> Option<T> {
    let args: Vec<String> = std::env::args().collect();
    let mut out: Option<T> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                out = Some(v);
            }
            i += 2;
        } else if let Some(val) = args[i].strip_prefix(&format!("{flag}=")) {
            if let Ok(v) = val.parse() {
                out = Some(v);
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Read `--neg TEXT` / `--neg=TEXT`, returning the raw string if present.
fn collect_neg_text() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--neg" {
            out = args.get(i + 1).cloned();
            i += 2;
        } else if let Some(val) = args[i].strip_prefix("--neg=") {
            out = Some(val.to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Tokenize via the HF `tokenizers` crate. Matches Python reference:
/// add_special_tokens=True → BOS=2 prepended; right-truncate to max_len;
/// left-pad with id 0; attention_mask 0 for pad, 1 for real.
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
    debug_assert_eq!(input_ids.len(), max_len);
    debug_assert_eq!(attention_mask.len(), max_len);
    Ok((input_ids, attention_mask))
}

/// Trim the CUDA memory pool to free cached blocks — mirrors the AV bin.
fn trim_cuda_pool(device: &std::sync::Arc<flame_core::CudaDevice>) {
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
}

/// Run Gemma-3 → video_aggregate_embed on a single prompt string.
/// Returns a `[1, seq, 4096]` BF16 tensor.
fn encode_video_context(
    text: &str,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<Tensor> {
    let mut shards: Vec<String> = Vec::new();
    for i in 1..=5 {
        let path = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
        if std::path::Path::new(&path).exists() {
            shards.push(path);
        }
    }
    let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();

    let (input_ids, attention_mask) = simple_tokenize(text, 256)?;
    let real_count = attention_mask.iter().filter(|&&m| m != 0).count();
    println!("  tokens: {} (real: {})", input_ids.len(), real_count);

    println!("  Loading Gemma-3...");
    let mut encoder = Gemma3Encoder::load(&shard_refs, device, input_ids.len())?;
    println!("  Gemma forward...");
    let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;

    println!("  Loading video_aggregate_embed...");
    let agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT_FULL),
        device,
        |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
    )?;
    let agg_w = agg_weights.get("text_embedding_projection.video_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing video_aggregate_embed.weight"))?;
    let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");

    println!("  Video feature extractor (→ 4096)...");
    let ctx = feature_extractor::feature_extract_and_project(
        &all_hidden, &mask_out, agg_w, agg_b, 4096,
    )?;

    drop(all_hidden);
    drop(mask_out);
    drop(encoder);
    drop(agg_weights);
    trim_cuda_pool(device);
    Ok(ctx)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // Parse guidance flags early.
    let cfg_scale: f32 = get_arg("--cfg").unwrap_or(DEFAULT_GUIDANCE_SCALE);
    let do_cfg = cfg_scale > 1.0;
    let neg_text = collect_neg_text().unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());
    let neg_embeds_path: Option<String> = get_arg("--neg-embeds");

    println!("============================================================");
    println!("LTX-2.3 Video Generation — Pure Rust + CFG");
    println!("============================================================");
    println!("  {}×{}, {} frames, {} steps, cfg={:.2} ({})",
        WIDTH, HEIGHT, NUM_FRAMES, NUM_STEPS,
        cfg_scale, if do_cfg { "on" } else { "off" });
    if do_cfg {
        if let Some(p) = &neg_embeds_path {
            println!("  neg source: embeds file {p}");
        } else {
            let preview = if neg_text.len() > 120 {
                format!("{}...", &neg_text[..120])
            } else {
                neg_text.clone()
            };
            println!("  neg source: encoded via Gemma-3 from {:?}", preview);
        }
    }

    let device = global_cuda_device();

    // Parse LoRA args before building the model so we can force the
    // BlockOffloader path when any are present.
    let lora_specs = collect_lora_args()?;
    for (p, s) in &lora_specs {
        println!("  LoRA: {} @ strength={:.3}", p, s);
    }

    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;
    let latent_h = HEIGHT / 32;
    let latent_w = WIDTH / 32;
    let num_tokens = latent_f * latent_h * latent_w;
    println!("  Latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, latent_h, latent_w, num_tokens);

    // Stage 1: Load positive embeddings (cached)
    println!("\n--- Stage 1: Load positive embeddings ---");
    let cached = flame_core::serialization::load_file(
        std::path::Path::new(EMBEDDINGS_PATH), &device,
    )?;
    let text_cond = cached.get("text_hidden")
        .ok_or_else(|| anyhow::anyhow!("Missing text_hidden"))?
        .clone();
    println!("  Conditional: {:?} {:?}", text_cond.dims(), text_cond.dtype());
    drop(cached); // release the rest of the cache map

    // Stage 1b: Negative embeddings.
    //   1. `--neg-embeds PATH` — load it; wins over everything.
    //   2. CFG on: encode `--neg TEXT` (or DEFAULT_NEGATIVE) via Gemma-3
    //      + video_aggregate_embed. Drops Gemma before stage 2.
    //   3. CFG off: no negative needed.
    //
    // The previous zeros-fallback produced near-black outputs at low step
    // counts (the "uncond = zeros" failure mode from session-12). We
    // never go back there.
    let text_uncond: Option<Tensor> = if do_cfg {
        let t0 = Instant::now();
        let ctx = if let Some(p) = &neg_embeds_path {
            println!("\n--- Stage 1b: Load cached negative embeddings ---");
            let m = flame_core::serialization::load_file(std::path::Path::new(p), &device)?;
            // Look for standard keys first, then fall back to first tensor.
            let picked = m.get("text_hidden")
                .or_else(|| m.get("video_context_neg"))
                .or_else(|| m.values().next())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no tensors in {p}"))?;
            if picked.dims() != text_cond.dims() {
                return Err(anyhow::anyhow!(
                    "neg embeds shape {:?} != positive shape {:?}",
                    picked.dims(), text_cond.dims()));
            }
            picked
        } else {
            println!("\n--- Stage 1b: Encode negative prompt ---");
            let ctx = encode_video_context(&neg_text, &device)?;
            if ctx.dims() != text_cond.dims() {
                eprintln!("  WARN: encoded neg {:?} vs positive {:?} differ — probably a seq-len mismatch",
                    ctx.dims(), text_cond.dims());
            }
            ctx
        };
        println!("  Unconditional: {:?} ({:.1}s)", ctx.dims(), t0.elapsed().as_secs_f32());
        Some(ctx)
    } else {
        None
    };

    // Stage 2: Load transformer
    println!("\n--- Stage 2: Load transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    println!("  Global params loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Attach LoRAs (fuses into globals now). Must happen BEFORE
    // load_fp8_resident / init_offloader.
    for (path, strength) in &lora_specs {
        let lora = LoraWeights::load(path, *strength, &device)?;
        model.add_lora(lora);
    }

    // When any LoRA is attached, go straight to BlockOffloader — FP8-resident
    // can't re-fuse deltas after dequant. Otherwise, try FP8 resident first.
    if !lora_specs.is_empty() {
        println!("  LoRA attached — skipping FP8 resident, using BlockOffloader");
        model.init_offloader()?;
        println!("  BlockOffloader initialized in {:.1}s", t0.elapsed().as_secs_f32());
    } else {
        match model.load_fp8_resident() {
            Ok(()) => println!("  FP8 resident loaded in {:.1}s", t0.elapsed().as_secs_f32()),
            Err(e) => {
                println!("  FP8 resident failed ({e}), falling back to BlockOffloader");
                model.init_offloader()?;
                println!("  BlockOffloader initialized in {:.1}s", t0.elapsed().as_secs_f32());
            }
        }
    }

    // Stage 3: Noise + schedule
    println!("\n--- Stage 3: Prepare noise + sigmas ---");
    let numel = LATENT_CHANNELS * latent_f * latent_h * latent_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
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
    };
    let noise = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, LATENT_CHANNELS, latent_f, latent_h, latent_w]),
        device.clone(),
    )?;
    let sigmas = if !do_cfg {
        // Distilled: use fixed sigma schedule
        LTX2_DISTILLED_SIGMAS.to_vec()
    } else {
        build_dev_sigma_schedule(NUM_STEPS, num_tokens, 0.5, 1.15, 0.0)
    };
    println!("  Noise: {:?}", noise.dims());
    println!("  Sigmas: {:?}", sigmas);

    // Stage 4: Denoise with CFG
    println!("\n--- Stage 4: Denoise ({} steps, CFG={:.2}) ---", NUM_STEPS, cfg_scale);
    let t0 = Instant::now();
    let mut x = noise;

    for step in 0..NUM_STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        let velocity = if do_cfg {
            // CFG: two forward passes
            let uncond = text_uncond.as_ref().expect("text_uncond must be set when CFG is on");
            let velocity_uncond = model.forward_video_only(
                &x, &sigma_t, uncond, FRAME_RATE, None,
            )?;
            let velocity_cond = model.forward_video_only(
                &x, &sigma_t, &text_cond, FRAME_RATE, None,
            )?;
            let delta = velocity_cond.sub(&velocity_uncond)?;
            velocity_uncond.add(&delta.mul_scalar(cfg_scale)?)?
        } else {
            // No CFG: single forward pass (distilled model)
            model.forward_video_only(&x, &sigma_t, &text_cond, FRAME_RATE, None)?
        };

        // Euler step
        if sigma_next == 0.0 {
            x = x.sub(&velocity.mul_scalar(sigma)?)?;
        } else {
            let dt = sigma_next - sigma;
            x = x.add(&velocity.mul_scalar(dt)?)?;
        }

        let dt_step = t_step.elapsed().as_secs_f32();
        println!("  Step {}/{} sigma={:.4} dt={:.1}s", step + 1, NUM_STEPS, sigma, dt_step);
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.1}s/step)", dt, dt / NUM_STEPS as f32);

    // Stage 5: Save
    println!("\n--- Stage 5: Save latents ---");
    let mut save_map = std::collections::HashMap::new();
    save_map.insert("latents".to_string(), x);
    flame_core::serialization::save_tensors(
        &save_map,
        std::path::Path::new(OUTPUT_PATH),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Saved to {}", OUTPUT_PATH);

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
