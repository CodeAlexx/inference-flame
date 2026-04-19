//! LTX-2.3 multi-scale audio+video generation — pure Rust.
//!
//! This is Lightricks's `LTXMultiScalePipeline` (`pipeline_ltx_video.py:1821-1895`)
//! re-implemented in Rust against our in-tree AV forward + LatentUpsampler
//! + AdaIN. The shape of the pipeline:
//!
//!   1. Gemma-3 → FeatureExtractor → (video_ctx, audio_ctx)   [once]
//!   2. Load LTX-2 transformer (FP8 resident or BlockOffloader fallback)
//!   3. First pass at `(W * d, H * d)` for `--first-pass-steps` steps.
//!      Produces `video_latent_ds` and `audio_latent`.
//!   4. Un-normalize video latent using the distilled VAE's
//!      per-channel-statistics, run through LTX2LatentUpsampler (spatial
//!      2x), re-normalize. Audio is **carried forward unchanged** — its
//!      temporal resolution already covers the full video duration.
//!   5. AdaIN-filter the upsampled latent against the first-pass latent
//!      (factor=1.0). This is the "tone of the low-res guide applied to
//!      the upsampled structure" step from Lightricks.
//!   6. Second pass at `(W * d * 2, H * d * 2)`. We re-noise the
//!      upsampled video latent at the stage-2 starting sigma (mirrors
//!      `skip_initial_inference_steps` by starting partway down the
//!      schedule). Audio is also re-noised to the same sigma so both
//!      streams agree on noise level during the joint forward.
//!   7. Save final video + audio latents.
//!
//! Audio invariant: audio latent is NEVER spatially upsampled. It is
//! re-noised in-place at the stage-2 starting sigma so the AV forward's
//! shared sigma conditioning is consistent across video and audio.
//! Dropping audio entirely during the second pass would desync the two
//! streams and regress AV coherence.
//!
//! Final bilinear resize to exactly (W, H) is deferred to the decode
//! step (`decode_latents_av.py`) which already handles per-frame resize
//! after VAE decode. We save the native latent grid here.

use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::models::ltx2_upsampler::LTX2LatentUpsampler;
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;
use inference_flame::sampling::ltx2_sampling::{LTX2_DISTILLED_SIGMAS, LTX2_STAGE2_DISTILLED_SIGMAS};
use inference_flame::sampling::ltx2_multiscale::adain_filter_latent;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

// Paths — same as `ltx2_two_stage.rs`. If any of these move, update both.
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const GEMMA_ROOT: &str =
    "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const UPSAMPLER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors";
const VAE_STATS_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const OUTPUT_DIR: &str =
    "/home/alex/EriDiffusion/inference-flame/output";

const LATENT_CHANNELS: usize = 128;
const AUDIO_CHANNELS: usize = 8;
const AUDIO_MEL_BINS: usize = 16;
const FRAME_RATE: f32 = 25.0;

const DEFAULT_PROMPT: &str =
    "A close-up frames a woman pressed flat against a cold metal locker in a dark storage bay, \
     her face half-lit by a single flickering overhead light. She whispers to herself in a \
     trembling voice, then pushes off the wall into the darkness.";

/// Parse --flag VALUE or --flag=VALUE.
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

fn collect_prompt() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--prompt" {
            return args.get(i + 1).cloned().unwrap_or_else(|| DEFAULT_PROMPT.to_string());
        } else if let Some(val) = args[i].strip_prefix("--prompt=") {
            return val.to_string();
        }
        i += 1;
    }
    DEFAULT_PROMPT.to_string()
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // --- CLI args ---
    // Target resolution = second-pass output (pre-resize).
    let target_w: usize = get_arg("--width").unwrap_or(512);
    let target_h: usize = get_arg("--height").unwrap_or(320);
    let num_frames: usize = get_arg("--frames").unwrap_or(257);
    let seed: u64 = get_arg("--seed").unwrap_or(42);
    let downscale: f32 = get_arg("--downscale").unwrap_or(0.6666_667);
    let first_pass_steps: Option<usize> = get_arg("--first-pass-steps");
    let second_pass_steps: Option<usize> = get_arg("--second-pass-steps");
    let adain_factor: f32 = get_arg("--adain-factor").unwrap_or(1.0);
    let prompt = collect_prompt();

    // Compute first-pass resolution. Lightricks snaps to
    // `vae_scale_factor` multiples (×32 on the spatial side for LTX-2).
    let x_w = (target_w as f32 * downscale).round() as usize;
    let x_h = (target_h as f32 * downscale).round() as usize;
    let first_w = x_w - (x_w % 32);
    let first_h = x_h - (x_h % 32);
    if first_w == 0 || first_h == 0 {
        return Err(anyhow::anyhow!(
            "downscale={downscale:.3} × target {target_w}×{target_h} → first-pass resolution 0 after rounding"
        ));
    }
    // Second-pass resolution is 2× the first-pass (NOT the target) — matches
    // Lightricks: `kwargs["width"] = downscaled_width * 2`.
    let second_w = first_w * 2;
    let second_h = first_h * 2;

    let latent_f = ((num_frames - 1) / 8) + 1;
    let first_lh = first_h / 32;
    let first_lw = first_w / 32;
    let second_lh = second_h / 32;
    let second_lw = second_w / 32;

    let video_duration = num_frames as f32 / FRAME_RATE;
    let audio_frames = (video_duration * 25.0).round() as usize;

    println!("============================================================");
    println!("LTX-2.3 Multi-Scale AV Generation — Pure Rust");
    println!("============================================================");
    println!("  Target:      {target_w}×{target_h}  (final resize done at decode)");
    println!("  Downscale:   {downscale:.4}");
    println!("  First pass:  {first_w}×{first_h}  latent=[128, {latent_f}, {first_lh}, {first_lw}]");
    println!("  Second pass: {second_w}×{second_h}  latent=[128, {latent_f}, {second_lh}, {second_lw}]");
    println!("  Frames:      {num_frames}  ({video_duration:.2}s at {FRAME_RATE} fps)");
    println!("  Audio:       {audio_frames} latent frames  ({AUDIO_CHANNELS}×T×{AUDIO_MEL_BINS})");
    println!("  Seed:        {seed}");
    println!("  AdaIN factor: {adain_factor}");

    let device = global_cuda_device();

    // ========================================
    // Text Encoding — with on-disk cache.
    // ========================================
    let cache_dir = format!("{OUTPUT_DIR}/embed_cache");
    let video_cache = format!("{cache_dir}/video_context.safetensors");
    let audio_cache = format!("{cache_dir}/audio_context.safetensors");

    let (video_context, audio_context) = if std::path::Path::new(&video_cache).exists()
        && std::path::Path::new(&audio_cache).exists()
        && std::env::var("LTX2_MS_REENCODE").is_err()
    {
        println!("\n--- Text Encoding (cached) ---");
        let vc = flame_core::serialization::load_file(std::path::Path::new(&video_cache), &device)?;
        let ac = flame_core::serialization::load_file(std::path::Path::new(&audio_cache), &device)?;
        let v = vc.get("video_context").unwrap().to_dtype(DType::BF16)?;
        let a = ac.get("audio_context").unwrap().to_dtype(DType::BF16)?;
        println!("  Loaded video_ctx={:?}  audio_ctx={:?}", v.dims(), a.dims());
        (v, a)
    } else {
        println!("\n--- Text Encoding (Gemma-3) ---");
        let t0 = Instant::now();

        let mut shards: Vec<String> = Vec::new();
        for i in 1..=5 {
            let p = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
            if std::path::Path::new(&p).exists() { shards.push(p); }
        }
        let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();

        let (input_ids, attention_mask) = simple_tokenize(&prompt, 256)?;
        let mut encoder = Gemma3Encoder::load(&shard_refs, &device, input_ids.len())?;
        let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;
        println!("  Gemma done: {:.1}s", t0.elapsed().as_secs_f32());

        let agg = flame_core::serialization::load_file_filtered(
            std::path::Path::new(LTX_CHECKPOINT), &device,
            |k| k.starts_with("text_embedding_projection.video_aggregate_embed"),
        )?;
        let agg_w = agg.get("text_embedding_projection.video_aggregate_embed.weight")
            .ok_or_else(|| anyhow::anyhow!("missing video_aggregate_embed.weight"))?;
        let agg_b = agg.get("text_embedding_projection.video_aggregate_embed.bias");
        let video_context = feature_extractor::feature_extract_and_project(
            &all_hidden, &mask_out, agg_w, agg_b, 4096,
        )?;

        let aagg = flame_core::serialization::load_file_filtered(
            std::path::Path::new(LTX_CHECKPOINT), &device,
            |k| k.starts_with("text_embedding_projection.audio_aggregate_embed"),
        )?;
        let aagg_w = aagg.get("text_embedding_projection.audio_aggregate_embed.weight")
            .ok_or_else(|| anyhow::anyhow!("missing audio_aggregate_embed.weight"))?;
        let aagg_b = aagg.get("text_embedding_projection.audio_aggregate_embed.bias");
        let audio_context = feature_extractor::feature_extract_and_project(
            &all_hidden, &mask_out, aagg_w, aagg_b, 2048,
        )?;

        drop(encoder);
        drop(all_hidden);
        drop(mask_out);
        drop(agg);
        drop(aagg);
        let _ = device.synchronize();

        std::fs::create_dir_all(&cache_dir)?;
        let mut vc = HashMap::new();
        vc.insert("video_context".to_string(), video_context.clone());
        flame_core::serialization::save_tensors(
            &vc, std::path::Path::new(&video_cache),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        let mut ac = HashMap::new();
        ac.insert("audio_context".to_string(), audio_context.clone());
        flame_core::serialization::save_tensors(
            &ac, std::path::Path::new(&audio_cache),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());
        (video_context, audio_context)
    };

    // ========================================
    // Load Transformer — BlockOffloader path (works on our 3090 Ti for
    // both passes without re-loading weights).
    // ========================================
    println!("\n--- Load Transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    model.init_offloader()?;
    println!("  BlockOffloader ready in {:.1}s", t0.elapsed().as_secs_f32());

    // ========================================
    // Sigmas: default to the hard-coded distilled schedules. --*-steps
    // overrides with a fresh linear_quadratic schedule of that length
    // (matching Lightricks's dev-mode behavior).
    // ========================================
    let first_sigmas: Vec<f32> = match first_pass_steps {
        Some(n) if n > 0 => {
            use inference_flame::sampling::ltx2_sampling::linear_quadratic_schedule;
            let mut s = linear_quadratic_schedule(n, 0.025);
            s.push(0.0); // terminator for Euler step
            s
        }
        _ => LTX2_DISTILLED_SIGMAS.to_vec(),
    };
    let second_sigmas: Vec<f32> = match second_pass_steps {
        Some(n) if n > 0 => {
            // Shortened schedule starting from first-pass end point. We take
            // the last `n+1` values of a full linear_quadratic schedule of
            // `first_steps + n` so the sigma range is consistent.
            use inference_flame::sampling::ltx2_sampling::linear_quadratic_schedule;
            let total = first_sigmas.len() - 1 + n;
            let full = linear_quadratic_schedule(total, 0.025);
            let mut tail: Vec<f32> = full.iter().skip(total - n).copied().collect();
            tail.push(0.0);
            tail
        }
        _ => LTX2_STAGE2_DISTILLED_SIGMAS.to_vec(),
    };

    let first_steps = first_sigmas.len() - 1;
    let second_steps = second_sigmas.len() - 1;
    println!("\n  First-pass sigmas  ({first_steps} steps): {first_sigmas:?}");
    println!("  Second-pass sigmas ({second_steps} steps): {second_sigmas:?}");

    // ========================================
    // First pass — denoise at low resolution.
    // ========================================
    println!("\n--- Pass 1: Denoise at {first_w}×{first_h} ({first_steps} steps) ---");
    let t0 = Instant::now();

    let first_v_numel = LATENT_CHANNELS * latent_f * first_lh * first_lw;
    let mut video_x = make_noise(
        first_v_numel, seed,
        &[1, LATENT_CHANNELS, latent_f, first_lh, first_lw],
        &device,
    )?;

    let audio_numel = AUDIO_CHANNELS * audio_frames * AUDIO_MEL_BINS;
    let mut audio_x = make_noise(
        audio_numel, seed + 1,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS],
        &device,
    )?;

    for step in 0..first_steps {
        let sigma = first_sigmas[step];
        let sigma_next = first_sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,
        )?;

        // F32-accumulated Euler step (same rationale as ltx2_two_stage.rs).
        let dt = sigma_next - sigma;
        let v_dtype = video_x.dtype();
        let a_dtype = audio_x.dtype();
        video_x = video_x.to_dtype(DType::F32)?
            .add(&video_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(v_dtype)?;
        audio_x = audio_x.to_dtype(DType::F32)?
            .add(&audio_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(a_dtype)?;

        println!("  [P1 {:2}/{}] sigma={:.4} dt={:.1}s",
            step + 1, first_steps, sigma, t_step.elapsed().as_secs_f32());
    }
    println!("  Pass 1 done in {:.1}s", t0.elapsed().as_secs_f32());

    // Keep a copy of the pass-1 video latent as the AdaIN reference.
    let video_first_pass = video_x.clone();

    // ========================================
    // Spatial upsample + AdaIN (video only).
    //
    // Audio is NEVER upsampled — its temporal resolution is fixed by the
    // duration of the clip. It passes through unchanged to the second pass.
    // ========================================
    println!("\n--- Upsample + AdaIN (video) ---");
    let t0 = Instant::now();

    // VAE per-channel stats (distilled checkpoint).
    let vae_stats = flame_core::serialization::load_file_filtered(
        std::path::Path::new(VAE_STATS_PATH), &device,
        |k| k == "vae.per_channel_statistics.mean-of-means"
            || k == "vae.per_channel_statistics.std-of-means",
    )?;
    let lat_mean = vae_stats.get("vae.per_channel_statistics.mean-of-means")
        .ok_or_else(|| anyhow::anyhow!("missing mean-of-means"))?;
    let lat_std = vae_stats.get("vae.per_channel_statistics.std-of-means")
        .ok_or_else(|| anyhow::anyhow!("missing std-of-means"))?;
    let mean_5d = lat_mean.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let std_5d = lat_std.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;

    // un-normalize → upsample → AdaIN (in un-normalized space) → normalize.
    // Lightricks does AdaIN in un-normalized space too (the upsampled and
    // reference latents must share the same statistical space).
    let video_first_unnorm = video_first_pass.mul(&std_5d)?.add(&mean_5d)?;
    let video_x_unnorm = video_x.mul(&std_5d)?.add(&mean_5d)?;
    drop(video_x);

    let upsampler = LTX2LatentUpsampler::load(UPSAMPLER_PATH, &device)?;
    println!("  Upsampler loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let video_upscaled = upsampler.forward(&video_x_unnorm)?;
    drop(upsampler);
    drop(video_x_unnorm);
    println!("  Upscaled: {:?} in {:.1}s",
        video_upscaled.shape().dims(), t0.elapsed().as_secs_f32());

    // AdaIN-filter the upsampled latent against the first-pass latent.
    // Mean/std stats are per-(batch, channel); the spatial mismatch
    // (upscaled is 2× the reference) is fine because stats are scalars.
    println!("  AdaIN filter (factor={adain_factor}, reference shape {:?})...",
        video_first_unnorm.shape().dims());
    let video_adain = adain_filter_latent(
        &video_upscaled,
        &video_first_unnorm,
        adain_factor,
    )?;
    drop(video_upscaled);
    drop(video_first_unnorm);

    // Renormalize back into the model's working space.
    let mut video_x = video_adain.sub(&mean_5d)?.div(&std_5d)?;
    drop(video_adain);
    drop(vae_stats);

    // ========================================
    // Pass 2 — refine at 2× resolution with a truncated sigma schedule.
    //
    // We re-noise both video and audio at `second_sigmas[0]` so the joint
    // AV forward receives a coherent sigma. For video, the "clean"
    // component is the upsampled AdaIN-filtered latent; the Euler trick
    // to start partway down the schedule is `x = (1 - s) * clean + s * eps`.
    // For audio, the "clean" component is the pass-1 audio latent.
    // ========================================
    let s2_start = second_sigmas[0];
    println!("\n--- Pass 2: Refine at {second_w}×{second_h} ({second_steps} steps, start σ={s2_start:.4}) ---");
    let t0 = Instant::now();

    let second_v_numel = LATENT_CHANNELS * latent_f * second_lh * second_lw;
    let v_noise = make_noise(
        second_v_numel, seed + 100,
        &[1, LATENT_CHANNELS, latent_f, second_lh, second_lw],
        &device,
    )?;
    video_x = video_x.mul_scalar(1.0 - s2_start)?
        .add(&v_noise.mul_scalar(s2_start)?)?;
    drop(v_noise);

    let a_noise = make_noise(
        audio_numel, seed + 101,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS],
        &device,
    )?;
    audio_x = audio_x.mul_scalar(1.0 - s2_start)?
        .add(&a_noise.mul_scalar(s2_start)?)?;
    drop(a_noise);

    for step in 0..second_steps {
        let sigma = second_sigmas[step];
        let sigma_next = second_sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,
        )?;

        let dt = sigma_next - sigma;
        let v_dtype = video_x.dtype();
        let a_dtype = audio_x.dtype();
        video_x = video_x.to_dtype(DType::F32)?
            .add(&video_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(v_dtype)?;
        audio_x = audio_x.to_dtype(DType::F32)?
            .add(&audio_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(a_dtype)?;

        println!("  [P2 {:2}/{}] sigma={:.4} dt={:.1}s",
            step + 1, second_steps, sigma, t_step.elapsed().as_secs_f32());
    }
    println!("  Pass 2 done in {:.1}s", t0.elapsed().as_secs_f32());

    // ========================================
    // Save final latents.
    // ========================================
    println!("\n--- Save latents ---");
    let video_out = format!("{OUTPUT_DIR}/ltx2_ms_video_latents.safetensors");
    let audio_out = format!("{OUTPUT_DIR}/ltx2_ms_audio_latents.safetensors");

    let mut vs = HashMap::new();
    vs.insert("latents".to_string(), video_x);
    flame_core::serialization::save_tensors(
        &vs, std::path::Path::new(&video_out),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Video: {video_out}");

    let mut audio_save = HashMap::new();
    audio_save.insert("latents".to_string(), audio_x);
    flame_core::serialization::save_tensors(
        &audio_save, std::path::Path::new(&audio_out),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Audio: {audio_out}");

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    println!("\nNext: python decode_latents_av.py (use ltx2_ms_* paths)");
    Ok(())
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

fn make_noise(
    _numel: usize, seed: u64, dims: &[usize],
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<Tensor> {
    flame_core::rng::set_seed(seed)
        .map_err(|e| anyhow::anyhow!("rng::set_seed: {e}"))?;
    let t = Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}
