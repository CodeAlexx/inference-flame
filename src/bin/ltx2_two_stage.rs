//! LTX-2.3 two-stage audio+video generation — pure Rust.
//!
//! Pipeline:
//! 1. Gemma-3 → FeatureExtractor → video+audio embeddings
//! 2. Load LTX-2 transformer (FP8 resident)
//! 3. Stage 1: Denoise at HALF resolution (8 distilled steps)
//! 4. Spatial upsampler: 2x video latent
//! 5. Stage 2: Refine at FULL resolution (3 steps, stage2 sigmas)
//! 6. Save latents (video + audio)
//!
//! This matches the official two-stage HQ pipeline from Lightricks.

use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::models::ltx2_upsampler::LTX2LatentUpsampler;
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;
use inference_flame::sampling::ltx2_sampling::{LTX2_DISTILLED_SIGMAS, LTX2_STAGE2_DISTILLED_SIGMAS};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

// Model paths
// IMPORTANT: use the BF16 distilled checkpoint, NOT the FP8 distilled.
// Per the LTX-2 README:
//   * `fp8-cast` policy is for BF16 checkpoints (downcasts on the fly)
//   * `fp8-scaled-mm` policy is for FP8 checkpoints (requires tensorrt_llm)
// We don't have tensorrt_llm, and our block offloader can stream BF16 weights
// directly (no per-block dequant needed). Loading the FP8 distilled file
// without scale-aware dequant produces dim/gray output (verified by running
// the official Python pipeline with `fp8_cast` on the FP8 file: same noise;
// same pipeline on the BF16 file produces real content).
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
// IMPORTANT: use the Lightricks official upsampler, not the diffusers copy.
// The diffusers file (`ltx2-diffusers/latent_upsampler/`) uses a *different*
// set of weights (max |diff| ~0.8 on `upsampler.conv.weight`) — likely from
// an older LTX-2 release — and produces completely wrong upscaled latents.
// The official Python DistilledPipeline uses this file by default.
const UPSAMPLER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors";
// Python's `upsample_video()` calls
// `encoder.per_channel_statistics.un_normalize(latent)` where
// `encoder` is built from the DISTILLED checkpoint and reads
// `vae.per_channel_statistics.{mean,std}-of-means`. The diffusers VAE
// `latents_mean`/`latents_std` are *different* numbers (max diff ~0.79) and
// produce a different un-normalized latent — which caused stage 2 to collapse.
// Load the distilled stats directly instead.
const VAE_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";

const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

// Generation params — target resolution (stage 2 output)
const PROMPT: &str = "A close-up frames a woman pressed flat against a cold metal locker in a dark storage bay, her face half-lit by a single flickering overhead light. Condensation drips down riveted steel walls as she holds her breath, eyes wide, listening. She whispers to herself in a trembling, barely audible voice, Think. Think. The airlock is two decks down. She swallows hard, closing her eyes as a slow, wet scraping sound passes on the other side of the wall. Her lips move again, voice cracking, It can hear you. It can hear your heartbeat. Stop shaking. The camera drifts slowly from her face down to her hands gripping a makeshift weapon, a sharpened length of pipe wrapped in electrical tape, knuckles white. A distant metallic clang echoes through the bay and her eyes snap open. She exhales in a shuddering whisper, Move now or die here, and pushes off the wall, the camera tracking low behind her as she crouches into the darkness between cargo containers. The ambient hum of the ship s failing life support drones beneath the silence.";
const NUM_FRAMES: usize = 257; // 8*32+1 = 257 frames → 10.28s at 25fps
const TARGET_WIDTH: usize = 512;
const TARGET_HEIGHT: usize = 320;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const AUDIO_CHANNELS: usize = 8;
const AUDIO_MEL_BINS: usize = 16;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // Stage 1 at half resolution
    let s1_width = TARGET_WIDTH / 2;   // 384
    let s1_height = TARGET_HEIGHT / 2; // 256

    println!("============================================================");
    println!("LTX-2.3 Two-Stage AV Generation — Pure Rust");
    println!("============================================================");
    println!("  Stage 1: {}×{}, {} frames, {} steps", s1_width, s1_height, NUM_FRAMES, LTX2_DISTILLED_SIGMAS.len() - 1);
    println!("  Stage 2: {}×{}, {} frames, {} steps", TARGET_WIDTH, TARGET_HEIGHT, NUM_FRAMES, LTX2_STAGE2_DISTILLED_SIGMAS.len() - 1);

    let device = global_cuda_device();

    // Stage 1 latent dimensions
    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;
    let s1_latent_h = s1_height / 32;
    let s1_latent_w = s1_width / 32;

    // Stage 2 latent dimensions (2x spatial)
    let s2_latent_h = TARGET_HEIGHT / 32;
    let s2_latent_w = TARGET_WIDTH / 32;

    // Audio
    let video_duration = NUM_FRAMES as f32 / FRAME_RATE;
    let audio_frames = (video_duration * 25.0).round() as usize;

    println!("  Stage 1 latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, s1_latent_h, s1_latent_w,
             latent_f * s1_latent_h * s1_latent_w);
    println!("  Stage 2 latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, s2_latent_h, s2_latent_w,
             latent_f * s2_latent_h * s2_latent_w);
    println!("  Audio: {} frames, {:.2}s", audio_frames, video_duration);

    // ========================================
    // Text Encoding — cached to disk
    // ========================================
    let cache_dir = format!("{OUTPUT_DIR}/embed_cache");
    let video_cache = format!("{cache_dir}/video_context.safetensors");
    let audio_cache = format!("{cache_dir}/audio_context.safetensors");

    let (video_context, audio_context) = if std::path::Path::new(&video_cache).exists()
        && std::path::Path::new(&audio_cache).exists()
    {
        println!("\n--- Text Encoding (cached) ---");
        let t0 = Instant::now();
        let vc = flame_core::serialization::load_file(
            std::path::Path::new(&video_cache), &device,
        )?;
        let ac = flame_core::serialization::load_file(
            std::path::Path::new(&audio_cache), &device,
        )?;
        let video_context = vc.get("video_context").unwrap().to_dtype(DType::BF16)?;
        let audio_context = ac.get("audio_context").unwrap().to_dtype(DType::BF16)?;
        println!("  Loaded from cache in {:.1}s", t0.elapsed().as_secs_f32());
        (video_context, audio_context)
    } else {
        println!("\n--- Text Encoding (Gemma-3) ---");
        let t0 = Instant::now();

        let mut shards: Vec<String> = Vec::new();
        for i in 1..=5 {
            let path = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
            if std::path::Path::new(&path).exists() {
                shards.push(path);
            }
        }
        let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();

        let (input_ids, attention_mask) = simple_tokenize(PROMPT, 256)?;
        let real_count = attention_mask.iter().filter(|&&m| m != 0).count();
        println!("  {} tokens ({} real)", input_ids.len(), real_count);

        let mut encoder = Gemma3Encoder::load(&shard_refs, &device, input_ids.len())?;
        let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;
        println!("  Gemma done: {:.1}s", t0.elapsed().as_secs_f32());

        let agg_weights = flame_core::serialization::load_file_filtered(
            std::path::Path::new(LTX_CHECKPOINT), &device,
            |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
        )?;
        let agg_w = agg_weights.get("text_embedding_projection.video_aggregate_embed.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing video_aggregate_embed.weight"))?;
        let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");
        let video_context = feature_extractor::feature_extract_and_project(
            &all_hidden, &mask_out, agg_w, agg_b, 4096,
        )?;

        let audio_agg_weights = flame_core::serialization::load_file_filtered(
            std::path::Path::new(LTX_CHECKPOINT), &device,
            |key| key.starts_with("text_embedding_projection.audio_aggregate_embed"),
        )?;
        let audio_agg_w = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing audio_aggregate_embed.weight"))?;
        let audio_agg_b = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.bias");
        let audio_context = feature_extractor::feature_extract_and_project(
            &all_hidden, &mask_out, audio_agg_w, audio_agg_b, 2048,
        )?;

        // Free Gemma
        drop(encoder);
        drop(all_hidden);
        drop(agg_weights);
        drop(audio_agg_weights);
        drop(mask_out);
        let _ = device.synchronize();
        unsafe {
            extern "C" {
                fn cudaMemPoolTrimTo(pool: *mut std::ffi::c_void, min_bytes: usize) -> i32;
                fn cudaDeviceGetDefaultMemPool(pool: *mut *mut std::ffi::c_void, device: i32) -> i32;
            }
            let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
            let _ = cudaDeviceGetDefaultMemPool(&mut pool, 0);
            if !pool.is_null() { let _ = cudaMemPoolTrimTo(pool, 0); }
        }

        // Cache to disk
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
        println!("  Encoded + cached in {:.1}s", t0.elapsed().as_secs_f32());
        (video_context, audio_context)
    };

    // ========================================
    // Load Transformer
    // ========================================
    println!("\n--- Load Transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;

    model.init_offloader()?;
    println!("  BlockOffloader ready in {:.1}s", t0.elapsed().as_secs_f32());

    // ========================================
    // Stage 1: Denoise at HALF resolution
    // ========================================
    println!("\n--- Stage 1: Denoise at {}×{} ({} steps) ---", s1_width, s1_height, LTX2_DISTILLED_SIGMAS.len() - 1);
    let t0 = Instant::now();

    let s1_video_numel = LATENT_CHANNELS * latent_f * s1_latent_h * s1_latent_w;
    let mut video_x = make_noise(s1_video_numel, SEED,
        &[1, LATENT_CHANNELS, latent_f, s1_latent_h, s1_latent_w], &device)?;

    let audio_numel = AUDIO_CHANNELS * audio_frames * AUDIO_MEL_BINS;
    let mut audio_x = make_noise(audio_numel, SEED + 1,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS], &device)?;

    let sigmas = LTX2_DISTILLED_SIGMAS.to_vec();
    let num_steps = sigmas.len() - 1;

    for step in 0..num_steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        // Model outputs velocity. Convert to X0: denoised = sample - sigma * velocity
        // Then Euler step: velocity = (sample - denoised) / sigma, sample += velocity * dt
        // Which simplifies to: sample += velocity * dt (velocity IS the model output)
        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,  // attention masks: None = pre-projected embeddings
        )?;

        // Dump first-step velocity for diff against Python.
        if step == 0 && std::env::var("LTX2_DUMP_VELOCITY").is_ok() {
            let mut vel_dump = HashMap::new();
            vel_dump.insert("video_vel".to_string(), video_vel.clone());
            vel_dump.insert("audio_vel".to_string(), audio_vel.clone());
            vel_dump.insert("video_x_in".to_string(), video_x.clone());
            vel_dump.insert("audio_x_in".to_string(), audio_x.clone());
            vel_dump.insert("video_context".to_string(), video_context.clone());
            vel_dump.insert("audio_context".to_string(), audio_context.clone());
            vel_dump.insert("sigma".to_string(), sigma_t.clone());
            flame_core::serialization::save_tensors(
                &vel_dump,
                std::path::Path::new("/home/alex/EriDiffusion/inference-flame/output/rust_step0_velocity.safetensors"),
                flame_core::serialization::SerializationFormat::SafeTensors,
            )?;
            println!("  [LTX2_DUMP_VELOCITY] saved step 0 velocity to rust_step0_velocity.safetensors");
            return Ok(());
        }

        // F32 Euler step — mirrors Python's `EulerDiffusionStep.step` which
        // does `(sample.float() + velocity.float() * dt).to(sample.dtype)`.
        // BF16 accumulation drift compounds across 8+3 steps and was the
        // source of the audio "2.4× too hot" symptom.
        let dt = sigma_next - sigma;
        let video_dtype = video_x.dtype();
        let audio_dtype = audio_x.dtype();
        video_x = video_x
            .to_dtype(DType::F32)?
            .add(&video_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(video_dtype)?;
        audio_x = audio_x
            .to_dtype(DType::F32)?
            .add(&audio_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(audio_dtype)?;

        // NaN check
        if let Ok(v) = video_x.to_vec() {
            let nan_count = v.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                println!("  WARNING: {} NaN values in video_x after step {}", nan_count, step + 1);
            }
        }

        println!("  Step {}/{} sigma={:.4} dt={:.1}s",
            step + 1, num_steps, sigma, t_step.elapsed().as_secs_f32());
    }
    println!("  Stage 1 done in {:.1}s ({:.1}s/step)", t0.elapsed().as_secs_f32(),
        t0.elapsed().as_secs_f32() / num_steps as f32);

    // Check latent stats after stage 1
    if let Ok(v) = video_x.to_vec() {
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        let valid: Vec<f32> = v.iter().filter(|x| !x.is_nan()).copied().collect();
        if !valid.is_empty() {
            let mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
            println!("  Stage 1 video latents: mean={:.4} nan={}/{}", mean, nan_count, v.len());
        } else {
            println!("  Stage 1 video latents: ALL NaN ({} values)", v.len());
        }
    }
    if let Ok(v) = audio_x.to_vec() {
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        println!("  Stage 1 audio latents: nan={}/{}", nan_count, v.len());
    }

    // Save stage 1 latents for diff against Python's stage 1.
    {
        let s1_video_path = format!("{OUTPUT_DIR}/ltx2_stage1_video_latents.safetensors");
        let s1_audio_path = format!("{OUTPUT_DIR}/ltx2_stage1_audio_latents.safetensors");
        let mut s1v = HashMap::new();
        s1v.insert("latents".to_string(), video_x.clone());
        flame_core::serialization::save_tensors(
            &s1v, std::path::Path::new(&s1_video_path),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        let mut s1a = HashMap::new();
        s1a.insert("latents".to_string(), audio_x.clone());
        flame_core::serialization::save_tensors(
            &s1a, std::path::Path::new(&s1_audio_path),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        println!("  Stage 1 latents saved to ltx2_stage1_*.safetensors");
    }

    if std::env::var("LTX2_STAGE1_ONLY").is_ok() {
        println!("\nLTX2_STAGE1_ONLY set — exiting after stage 1.");
        return Ok(());
    }

    // ========================================
    // Spatial Upsampler: 2x video latent
    // ========================================
    println!("\n--- Spatial Upsampler (2x) ---");
    let t0 = Instant::now();

    let vae_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(VAE_PATH), &device,
        |key| key == "vae.per_channel_statistics.mean-of-means"
            || key == "vae.per_channel_statistics.std-of-means",
    )?;
    let latents_mean = vae_weights.get("vae.per_channel_statistics.mean-of-means")
        .ok_or_else(|| anyhow::anyhow!("Missing vae.per_channel_statistics.mean-of-means in distilled checkpoint"))?;
    let latents_std = vae_weights.get("vae.per_channel_statistics.std-of-means")
        .ok_or_else(|| anyhow::anyhow!("Missing vae.per_channel_statistics.std-of-means in distilled checkpoint"))?;

    // Un-normalize: x * std + mean
    let mean_5d = latents_mean.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let std_5d = latents_std.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let video_unnorm = video_x.mul(&std_5d)?.add(&mean_5d)?;
    drop(video_x);

    let upsampler = LTX2LatentUpsampler::load(UPSAMPLER_PATH, &device)?;
    println!("  Upsampler loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let video_upscaled = upsampler.forward(&video_unnorm)?;
    drop(upsampler);
    drop(video_unnorm);
    println!("  Upscaled: {:?} in {:.1}s", video_upscaled.shape().dims(), t0.elapsed().as_secs_f32());

    // Debug: dump upscaled latent (post-upsampler, pre re-normalize) so it can
    // be diffed against Python's `output/python_upscaled_from_rust.safetensors`.
    if std::env::var("LTX2_DUMP_UPSCALED").is_ok() {
        let mut up = HashMap::new();
        up.insert("latents".to_string(), video_upscaled.clone());
        flame_core::serialization::save_tensors(
            &up,
            std::path::Path::new("/home/alex/EriDiffusion/inference-flame/output/rust_upscaled_from_rust.safetensors"),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        println!("  [LTX2_DUMP_UPSCALED] saved upscaled latent");
    }

    // Re-normalize: (x - mean) / std
    let s2_mean = latents_mean.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let s2_std = latents_std.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let mut video_x = video_upscaled.sub(&s2_mean)?.div(&s2_std)?;
    drop(vae_weights);

    // ========================================
    // Stage 2: Refine at FULL resolution (BlockOffloader)
    // ========================================
    let s2_sigmas = LTX2_STAGE2_DISTILLED_SIGMAS.to_vec();
    let s2_steps = s2_sigmas.len() - 1;
    println!("\n--- Stage 2: Refine at {}x{} ({} steps) ---", TARGET_WIDTH, TARGET_HEIGHT, s2_steps);
    let t0 = Instant::now();

    // Add noise at stage 2 starting sigma
    let noise_scale = s2_sigmas[0];
    let s2_video_numel = LATENT_CHANNELS * latent_f * s2_latent_h * s2_latent_w;
    let noise = make_noise(s2_video_numel, SEED + 100,
        &[1, LATENT_CHANNELS, latent_f, s2_latent_h, s2_latent_w], &device)?;
    video_x = video_x.mul_scalar(1.0 - noise_scale)?.add(&noise.mul_scalar(noise_scale)?)?;
    drop(noise);

    let audio_noise = make_noise(audio_numel, SEED + 101,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS], &device)?;
    audio_x = audio_x.mul_scalar(1.0 - noise_scale)?.add(&audio_noise.mul_scalar(noise_scale)?)?;
    drop(audio_noise);

    for step in 0..s2_steps {
        let sigma = s2_sigmas[step];
        let sigma_next = s2_sigmas[step + 1];
        let t_step = Instant::now();

        let sigma_t = Tensor::from_f32_to_bf16(
            vec![sigma], Shape::from_dims(&[1]), device.clone(),
        )?;

        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,  // attention masks: None = pre-projected embeddings
        )?;

        // F32 Euler step (see stage 1 loop above for rationale).
        let dt = sigma_next - sigma;
        let video_dtype = video_x.dtype();
        let audio_dtype = audio_x.dtype();
        video_x = video_x
            .to_dtype(DType::F32)?
            .add(&video_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(video_dtype)?;
        audio_x = audio_x
            .to_dtype(DType::F32)?
            .add(&audio_vel.to_dtype(DType::F32)?.mul_scalar(dt)?)?
            .to_dtype(audio_dtype)?;

        // NaN check
        if let Ok(v) = video_x.to_vec() {
            let nan_count = v.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                println!("  WARNING: {} NaN in video_x after stage 2 step {}", nan_count, step + 1);
            }
        }

        println!("  Step {}/{} sigma={:.4} dt={:.1}s",
            step + 1, s2_steps, sigma, t_step.elapsed().as_secs_f32());
    }
    println!("  Stage 2 done in {:.1}s ({:.1}s/step)", t0.elapsed().as_secs_f32(),
        t0.elapsed().as_secs_f32() / s2_steps as f32);

    // Final latent stats
    if let Ok(v) = video_x.to_vec() {
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        let valid: Vec<f32> = v.iter().filter(|x| !x.is_nan()).copied().collect();
        if !valid.is_empty() {
            let mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
            println!("  Final video latents: mean={:.4} std={:.4} nan={}/{}",
                mean, (valid.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / valid.len() as f32).sqrt(),
                nan_count, v.len());
        }
    }

    // ========================================
    // Save latents
    // ========================================
    println!("\n--- Save latents ---");
    let video_path = format!("{OUTPUT_DIR}/ltx2_twostage_video_latents.safetensors");
    let audio_path = format!("{OUTPUT_DIR}/ltx2_twostage_audio_latents.safetensors");

    let mut video_save = HashMap::new();
    video_save.insert("latents".to_string(), video_x);
    flame_core::serialization::save_tensors(
        &video_save, std::path::Path::new(&video_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Video: {}", video_path);

    let mut audio_save = HashMap::new();
    audio_save.insert("latents".to_string(), audio_x);
    flame_core::serialization::save_tensors(
        &audio_save, std::path::Path::new(&audio_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Audio: {}", audio_path);

    println!("\n============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

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
        println!(
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

fn make_noise(_numel: usize, seed: u64, dims: &[usize], device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> anyhow::Result<Tensor> {
    flame_core::rng::set_seed(seed);
    let t = Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}
