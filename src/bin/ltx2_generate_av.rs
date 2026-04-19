//! LTX-2.3 audio+video generation — pure Rust.
//!
//! Pipeline:
//! 1. Run Gemma-3 → FeatureExtractor → video+audio embeddings
//! 2. Load LTX-2 transformer (FP8 resident with audio weights)
//! 3. Create video noise + audio noise
//! 4. Denoise jointly with forward_audio_video, optionally with CFG + STG
//! 5. Save video + audio latents → decode with Python VAE

use inference_flame::models::lora_loader::LoraWeights;
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;
use inference_flame::sampling::ltx2_sampling::LTX2_DISTILLED_SIGMAS;
use inference_flame::sampling::ltx2_guidance::{cfg_star_rescale, stg_rescale};
use flame_core::{global_cuda_device, Shape, Tensor};

/// Parse `--lora PATH[:STRENGTH]`. Default strength is 1.0.
fn parse_lora_arg(raw: &str) -> anyhow::Result<(String, f32)> {
    if let Some(idx) = raw.rfind(':') {
        let (path, tail) = raw.split_at(idx);
        let tail = &tail[1..];
        if let Ok(s) = tail.parse::<f32>() {
            return Ok((path.to_string(), s));
        }
    }
    Ok((raw.to_string(), 1.0))
}

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

/// Read a single `--flag VALUE` or `--flag=VALUE` argument and parse it as T.
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

/// Check if a boolean-style flag (`--flag`) is present.
fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

/// Parse `--stg-blocks 11,25,35,39` into a Vec<usize>. Returns empty on absent.
fn parse_stg_blocks() -> Vec<usize> {
    let raw: Option<String> = get_arg("--stg-blocks");
    let Some(s) = raw else { return Vec::new(); };
    s.split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect()
}

/// Read the negative-prompt text (`--neg "TEXT"` / `--neg=TEXT`).
///
/// `get_arg::<String>` doesn't work for strings containing spaces once you
/// go through `&format!("{flag}=")` — it does, but the stock `.parse::<String>()`
/// always succeeds so this is safe. We keep it as its own helper so the
/// intent reads cleanly at the call site and to allow last-wins semantics.
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
use std::collections::HashMap;
use std::time::Instant;

// Model paths
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";

const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

/// Lightricks-canonical default negative prompt.
///
/// Source: `/tmp/ltx-video/ltx_video/inference.py:351-354` (checked in by
/// the LTX-Video authors). If this is edited, also update
/// `scripts/ltx2_neg_prompt_ref.py` and `src/bin/ltx2_neg_prompt_parity.rs`
/// — the parity bin compares this exact string byte-for-byte.
const DEFAULT_NEGATIVE: &str =
    "worst quality, inconsistent motion, blurry, jittery, distorted";

// Generation params
const PROMPT: &str = "A close-up frames a woman pressed flat against a cold metal locker in a dark storage bay, her face half-lit by a single flickering overhead light. Condensation drips down riveted steel walls as she holds her breath, eyes wide, listening. She whispers to herself in a trembling, barely audible voice, Think. Think. The airlock is two decks down. She swallows hard, closing her eyes as a slow, wet scraping sound passes on the other side of the wall. Her lips move again, voice cracking, It can hear you. It can hear your heartbeat. Stop shaking. The camera drifts slowly from her face down to her hands gripping a makeshift weapon, a sharpened length of pipe wrapped in electrical tape, knuckles white. A distant metallic clang echoes through the bay and her eyes snap open. She exhales in a shuddering whisper, Move now or die here, and pushes off the wall, the camera tracking low behind her as she crouches into the darkness between cargo containers. The ambient hum of the ship s failing life support drones beneath the silence.";
const NUM_FRAMES: usize = 257; // 8*32+1 = 257 frames → 10.28s at 25fps
const WIDTH: usize = 480;
const HEIGHT: usize = 288;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const AUDIO_CHANNELS: usize = 8;
const AUDIO_MEL_BINS: usize = 16;
const NUM_STEPS: usize = 8;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // Parse guidance knobs UP FRONT — stage 1 needs to know whether CFG is
    // on so it can encode the negative prompt in the same pass as the
    // positive (saves loading Gemma twice).
    let cfg_scale: f32 = get_arg("--cfg").unwrap_or(1.0);
    let stg_scale: f32 = get_arg("--stg").unwrap_or(0.0);
    let stg_rescale_factor: f32 = get_arg("--stg-rescale").unwrap_or(0.7);
    let cfg_star = has_flag("--cfg-star-rescale");
    let stg_blocks = parse_stg_blocks();
    let neg_text = collect_neg_text().unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());
    let fp8_stream = has_flag("--fp8-stream");

    let do_cfg = cfg_scale > 1.0;
    let do_stg = stg_scale > 0.0 && !stg_blocks.is_empty();

    println!("============================================================");
    println!("LTX-2.3 Audio+Video Generation — Pure Rust");
    println!("============================================================");
    println!("  {}×{}, {} frames, {} steps", WIDTH, HEIGHT, NUM_FRAMES, NUM_STEPS);
    println!("  cfg={:.2} ({})  stg={:.2} ({})  cfg_star={}",
        cfg_scale, if do_cfg { "on" } else { "off" },
        stg_scale, if do_stg { "on" } else { "off" },
        cfg_star);
    if do_cfg {
        // Truncate the preview — some negatives can be paragraph-length.
        let preview = if neg_text.len() > 120 {
            format!("{}...", &neg_text[..120])
        } else {
            neg_text.clone()
        };
        println!("  neg_prompt = {:?}", preview);
    }

    let device = global_cuda_device();

    // Compute latent dimensions
    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;
    let latent_h = HEIGHT / 32;
    let latent_w = WIDTH / 32;
    let num_video_tokens = latent_f * latent_h * latent_w;

    // Audio latent frames: round((num_frames / frame_rate) * 25.0)
    let video_duration = NUM_FRAMES as f32 / FRAME_RATE;
    let audio_frames = (video_duration * 25.0).round() as usize;
    let num_audio_tokens = audio_frames * AUDIO_MEL_BINS;

    println!("  Video latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, latent_h, latent_w, num_video_tokens);
    println!("  Audio latent: [{}, {}, {}] = {} tokens",
             AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS, num_audio_tokens);
    println!("  Video duration: {:.2}s", video_duration);

    // ========================================
    // Stage 1: Text Encoding (Gemma → FeatureExtractor)
    // ========================================
    println!("\n--- Stage 1: Text Encoding ---");
    let t0 = Instant::now();

    // Find Gemma shards
    let mut shards: Vec<String> = Vec::new();
    for i in 1..=5 {
        let path = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
        if std::path::Path::new(&path).exists() {
            shards.push(path);
        }
    }
    let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();

    // Tokenize (simple: <bos> + text tokens, left-padded to 256)
    println!("  Tokenizing (positive)...");
    let (input_ids, attention_mask) = simple_tokenize(PROMPT, 256)?;
    let real_count = attention_mask.iter().filter(|&&m| m != 0).count();
    println!("  {} tokens ({} real)", input_ids.len(), real_count);

    // Load and run Gemma (positive)
    println!("  Loading Gemma-3...");
    let mut encoder = Gemma3Encoder::load(&shard_refs, &device, input_ids.len())?;
    println!("  Running Gemma forward (positive)...");
    let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;
    println!("  Gemma done: {} hidden states in {:.1}s", all_hidden.len(), t0.elapsed().as_secs_f32());

    // Feature extraction — load projection weights once (reused for neg).
    println!("  Loading aggregate_embed (video, 4096) + (audio, 2048)...");
    let agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT),
        &device,
        |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
    )?;
    let agg_w = agg_weights.get("text_embedding_projection.video_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing video_aggregate_embed.weight"))?;
    let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");

    let audio_agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT),
        &device,
        |key| key.starts_with("text_embedding_projection.audio_aggregate_embed"),
    )?;
    let audio_agg_w = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing audio_aggregate_embed.weight"))?;
    let audio_agg_b = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.bias");

    println!("  Running video feature extractor (→ 4096)...");
    let video_context = feature_extractor::feature_extract_and_project(
        &all_hidden,
        &mask_out,
        agg_w,
        agg_b,
        4096,
    )?;
    println!("  Video context: {:?}", video_context.dims());

    println!("  Running audio feature extractor (→ 2048)...");
    let audio_context = feature_extractor::feature_extract_and_project(
        &all_hidden,
        &mask_out,
        audio_agg_w,
        audio_agg_b,
        2048,
    )?;
    println!("  Audio context: {:?}", audio_context.dims());

    // Drop positive hidden-states / mask before the negative forward to
    // hold peak VRAM steady — Gemma's per-layer [1, 256, 3840] stack is
    // reconstructed when encode() is called again.
    drop(all_hidden);
    drop(mask_out);

    // ----- Negative prompt (only when CFG will be used) -----
    //
    // Lightricks always encodes the negative (pipeline_ltx_video.py:1024-1040)
    // but only USES it in the CFG combination when guidance_scale > 1
    // (pipeline_ltx_video.py:1116). We only spend the cycles when we need
    // them. Both video_context_neg (4096) and audio_context_neg (2048)
    // are produced so the AV forward has a proper uncond pair —
    // "uncond = zeros" for audio silently degrades AV coherence.
    let (neg_video_context_from_text, neg_audio_context_from_text) = if do_cfg {
        println!("  Tokenizing (negative)...");
        let (neg_ids, neg_mask) = simple_tokenize(&neg_text, 256)?;
        let neg_real = neg_mask.iter().filter(|&&m| m != 0).count();
        println!("  {} tokens ({} real)", neg_ids.len(), neg_real);

        println!("  Running Gemma forward (negative)...");
        let (neg_hidden, neg_mask_out) = encoder.encode(&neg_ids, &neg_mask)?;

        println!("  Running video feature extractor on negative (→ 4096)...");
        let v_neg = feature_extractor::feature_extract_and_project(
            &neg_hidden, &neg_mask_out, agg_w, agg_b, 4096,
        )?;
        println!("  Negative video context: {:?}", v_neg.dims());

        println!("  Running audio feature extractor on negative (→ 2048)...");
        let a_neg = feature_extractor::feature_extract_and_project(
            &neg_hidden, &neg_mask_out, audio_agg_w, audio_agg_b, 2048,
        )?;
        println!("  Negative audio context: {:?}", a_neg.dims());

        drop(neg_hidden);
        drop(neg_mask_out);
        (Some(v_neg), Some(a_neg))
    } else {
        (None, None)
    };

    // Free Gemma + feature extraction to reclaim ALL VRAM for DiT
    drop(encoder);
    drop(agg_weights);
    drop(audio_agg_weights);
    // Force CUDA memory pool to release cached allocations
    let _ = device.synchronize();
    // Trim the CUDA memory pool to free cached blocks
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
    println!("  Text encoding total: {:.1}s (VRAM freed)", t0.elapsed().as_secs_f32());

    // ========================================
    // Stage 2: Load LTX-2 Transformer
    // ========================================
    println!("\n--- Stage 2: Load transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    println!("  Global params loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Parse and attach any --lora flags BEFORE picking the forward path.
    // LTX-2 LoRAs patch both video AND audio weights — the distilled
    // LoRA ships 2134 audio keys (64% of deltas) — so we must stay on a
    // path that fuses them.
    let lora_specs = collect_lora_args()?;
    for (p, s) in &lora_specs {
        println!("  LoRA: {} @ strength={:.3}", p, s);
    }
    for (path, strength) in &lora_specs {
        let lora = LoraWeights::load(path, *strength, &device)?;
        model.add_lora(lora);
    }

    // When LoRAs are attached, force BlockOffloader — FP8-resident can't
    // re-fuse deltas after dequant and would silently miss audio LoRAs.
    if fp8_stream {
        // FP8-scaled-mm streaming: keeps raw FP8 bytes pinned on host, GPU
        // dequants to BF16 per block with sidecar `weight_scale` scalars.
        // ~20.5 GB pinned vs ~37 GB for BF16 streaming; fits on a 24 GB card
        // without OOM at 480×288 and higher resolutions. LoRA deltas are
        // fused BF16-post-dequant so audio LoRAs are preserved.
        model.init_offloader_fp8_stream(MODEL_PATH)?;
        println!(
            "  FP8-stream BlockOffloader initialized in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
    } else if !lora_specs.is_empty() {
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
    // Print VRAM after model load
    if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
        println!("  VRAM: {:.1}GB used / {:.1}GB total ({:.1}GB free)",
            (total - free) as f64 / 1e9, total as f64 / 1e9, free as f64 / 1e9);
    }

    // ========================================
    // Stage 3: Create noise
    // ========================================
    println!("\n--- Stage 3: Prepare noise + sigmas ---");

    // Video noise
    let video_numel = LATENT_CHANNELS * latent_f * latent_h * latent_w;
    let video_noise = make_noise(video_numel, SEED, &[1, LATENT_CHANNELS, latent_f, latent_h, latent_w], &device)?;

    // Audio noise
    let audio_numel = AUDIO_CHANNELS * audio_frames * AUDIO_MEL_BINS;
    let audio_noise = make_noise(audio_numel, SEED + 1, &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS], &device)?;

    let sigmas = LTX2_DISTILLED_SIGMAS.to_vec();
    println!("  Video noise: {:?}", video_noise.dims());
    println!("  Audio noise: {:?}", audio_noise.dims());
    println!("  Sigmas: {:?}", sigmas);

    // ========================================
    // Guidance summary (flags already parsed at startup)
    // ========================================
    let forwards_per_step = 1 + if do_cfg { 1 } else { 0 } + if do_stg { 1 } else { 0 };
    println!("\nGuidance: cfg={:.2} ({}) stg={:.2} ({}) blocks={:?} cfg_star={} stg_rescale={:.2}",
        cfg_scale, if do_cfg { "on" } else { "off" },
        stg_scale, if do_stg { "on" } else { "off" },
        stg_blocks, cfg_star, stg_rescale_factor);
    println!("  → {} forward(s) per denoise step", forwards_per_step);

    // Negative-prompt contexts:
    //   1. Prefer `--neg-video-embeddings PATH` / `--neg-audio-embeddings PATH`
    //      (pre-cached tensors in the context dim). Useful for iterating
    //      on LoRAs without re-running Gemma each time.
    //   2. Otherwise use the video/audio contexts encoded from `--neg TEXT`
    //      (or the Lightricks-canonical default) in stage 1.
    //   3. Zeros fallback is explicitly DISABLED — audio-zero-uncond was
    //      the session-12 AV-coherence regression; reaching this branch
    //      with CFG on means the negative encoder silently skipped audio
    //      and is a bug.
    let (neg_video_context, neg_audio_context) = if do_cfg {
        let nvc_path: Option<String> = get_arg("--neg-video-embeddings");
        let nac_path: Option<String> = get_arg("--neg-audio-embeddings");

        let nvc = if let Some(p) = nvc_path {
            println!("  Loading neg video embeddings from {p}");
            let m = flame_core::serialization::load_file(std::path::Path::new(&p), &device)?;
            m.values().next().cloned().ok_or_else(||
                anyhow::anyhow!("no tensors in {p}"))?
        } else {
            neg_video_context_from_text.clone().ok_or_else(||
                anyhow::anyhow!("CFG enabled but no neg video context was encoded — this is a bug"))?
        };
        let nac = if let Some(p) = nac_path {
            println!("  Loading neg audio embeddings from {p}");
            let m = flame_core::serialization::load_file(std::path::Path::new(&p), &device)?;
            m.values().next().cloned().ok_or_else(||
                anyhow::anyhow!("no tensors in {p}"))?
        } else {
            neg_audio_context_from_text.clone().ok_or_else(||
                anyhow::anyhow!("CFG enabled but no neg audio context was encoded — this is a bug"))?
        };
        (Some(nvc), Some(nac))
    } else {
        (None, None)
    };

    // ========================================
    // Stage 4: Denoise (audio + video jointly)
    // ========================================
    println!("\n--- Stage 4: Denoise ({} steps, AV joint, {} fwd/step) ---",
        NUM_STEPS, forwards_per_step);
    let t0 = Instant::now();
    let mut video_x = video_noise;
    let mut audio_x = audio_noise;

    for step in 0..NUM_STEPS {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        // -------- Positive (cond) forward --------
        let (video_cond, audio_cond) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,
        )?;

        // -------- Optional CFG uncond forward --------
        let (video_vel, audio_vel) = if do_cfg {
            let nvc = neg_video_context.as_ref().unwrap();
            let nac = neg_audio_context.as_ref().unwrap();
            let (mut video_uncond, mut audio_uncond) = model.forward_audio_video(
                &video_x, &audio_x, &sigma_t,
                nvc, nac,
                FRAME_RATE,
                None, None,
            )?;
            if cfg_star {
                video_uncond = cfg_star_rescale(&video_cond, &video_uncond)?;
                audio_uncond = cfg_star_rescale(&audio_cond, &audio_uncond)?;
            }
            // noise_pred = uncond + cfg_scale * (cond - uncond)
            let v_guided = video_uncond.add(
                &video_cond.sub(&video_uncond)?.mul_scalar(cfg_scale)?,
            )?;
            let a_guided = audio_uncond.add(
                &audio_cond.sub(&audio_uncond)?.mul_scalar(cfg_scale)?,
            )?;
            (v_guided, a_guided)
        } else {
            (video_cond.clone(), audio_cond.clone())
        };

        // -------- Optional STG perturb forward --------
        let (video_vel, audio_vel) = if do_stg {
            let (video_pert, audio_pert) = model.forward_audio_video_with_stg(
                &video_x, &audio_x, &sigma_t,
                &video_context, &audio_context,
                FRAME_RATE,
                None, None,
                Some(&stg_blocks),
            )?;
            // noise_pred += stg_scale * (cond - perturbed)
            let v_out = video_vel.add(
                &video_cond.sub(&video_pert)?.mul_scalar(stg_scale)?,
            )?;
            let a_out = audio_vel.add(
                &audio_cond.sub(&audio_pert)?.mul_scalar(stg_scale)?,
            )?;
            // STG std-rescale if requested
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

        // Euler step for video
        if sigma_next == 0.0 {
            video_x = video_x.sub(&video_vel.mul_scalar(sigma)?)?;
        } else {
            let dt = sigma_next - sigma;
            video_x = video_x.add(&video_vel.mul_scalar(dt)?)?;
        }

        // Euler step for audio
        if sigma_next == 0.0 {
            audio_x = audio_x.sub(&audio_vel.mul_scalar(sigma)?)?;
        } else {
            let dt = sigma_next - sigma;
            audio_x = audio_x.add(&audio_vel.mul_scalar(dt)?)?;
        }

        let dt_step = t_step.elapsed().as_secs_f32();
        println!("  Step {}/{} sigma={:.4} dt={:.1}s", step + 1, NUM_STEPS, sigma, dt_step);
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.1}s/step)", dt, dt / NUM_STEPS as f32);

    // ========================================
    // Stage 5: Save latents
    // ========================================
    println!("\n--- Stage 5: Save latents ---");
    let video_path = format!("{OUTPUT_DIR}/ltx2_av_video_latents.safetensors");
    let audio_path = format!("{OUTPUT_DIR}/ltx2_av_audio_latents.safetensors");

    let mut video_save = HashMap::new();
    video_save.insert("latents".to_string(), video_x);
    flame_core::serialization::save_tensors(
        &video_save, std::path::Path::new(&video_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Video latents: {}", video_path);

    let mut audio_save = HashMap::new();
    audio_save.insert("latents".to_string(), audio_x);
    flame_core::serialization::save_tensors(
        &audio_save, std::path::Path::new(&audio_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Audio latents: {}", audio_path);

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    println!("\nNext: python decode_latents_av.py");

    Ok(())
}

/// Tokenize a prompt for Gemma-3 using the HuggingFace `tokenizers` crate.
///
/// Matches the Python reference pipeline's behavior:
///   - BOS (id=2) prepended automatically via `add_special_tokens=true`
///   - If encoded length > max_len, right-truncate (truncation_side='right')
///   - If encoded length < max_len, left-pad with pad_id=0 (padding_side='left')
///   - attention_mask: 0 for pad positions, 1 for real tokens
///
/// Verified against `/home/alex/ltx2-refs/bench20/tokens_*.json`:
/// e.g. encode("a cat") → [2, 236746, 5866] (BOS, "a", "▁cat"), matches byte-for-byte.
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

/// Generate Gaussian noise with Box-Muller transform.
fn make_noise(numel: usize, seed: u64, shape: &[usize], device: &std::sync::Arc<flame_core::CudaDevice>) -> anyhow::Result<Tensor> {
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
    Ok(Tensor::from_f32_to_bf16(v, Shape::from_dims(shape), device.clone())?)
}
