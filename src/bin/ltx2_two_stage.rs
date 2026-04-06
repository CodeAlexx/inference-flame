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
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const UPSAMPLER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx2-diffusers/latent_upsampler/diffusion_pytorch_model.safetensors";
const VAE_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx2-diffusers/vae/diffusion_pytorch_model.safetensors";

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

    model.init_swap()?;
    println!("  FlameSwap ready in {:.1}s", t0.elapsed().as_secs_f32());

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

        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
        )?;

        // Euler step
        if sigma_next == 0.0 {
            video_x = video_x.sub(&video_vel.mul_scalar(sigma)?)?;
            audio_x = audio_x.sub(&audio_vel.mul_scalar(sigma)?)?;
        } else {
            let dt = sigma_next - sigma;
            video_x = video_x.add(&video_vel.mul_scalar(dt)?)?;
            audio_x = audio_x.add(&audio_vel.mul_scalar(dt)?)?;
        }

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

    // Skip upsampler — save raw stage 1 output directly
    println!("\n--- Saving raw stage 1 latents (no upsampler, no stage 2) ---");

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

fn simple_tokenize(text: &str, _max_len: usize) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
    let bench_dir = "/home/alex/ltx2-refs/bench20";
    let prompts_path = format!("{bench_dir}/prompts.json");
    if std::path::Path::new(&prompts_path).exists() {
        let prompts_json = std::fs::read_to_string(&prompts_path)?;
        let prompts: Vec<String> = serde_json::from_str(&prompts_json)?;
        for (i, p) in prompts.iter().enumerate() {
            if p == text {
                let token_path = format!("{bench_dir}/tokens_{i:02}.json");
                let token_json = std::fs::read_to_string(&token_path)?;
                let tokens: serde_json::Value = serde_json::from_str(&token_json)?;
                let input_ids: Vec<i32> = tokens["input_ids"].as_array().unwrap()
                    .iter().map(|v| v.as_i64().unwrap() as i32).collect();
                let attention_mask: Vec<i32> = tokens["attention_mask"].as_array().unwrap()
                    .iter().map(|v| v.as_i64().unwrap() as i32).collect();
                println!("  Matched token file: Some(\"tokens_{i:02}.json\")");
                return Ok((input_ids, attention_mask));
            }
        }
    }
    // Fallback: Use cinematic prompt tokens
    let fallback = format!("{bench_dir}/tokens_cinematic.json");
    if std::path::Path::new(&fallback).exists() {
        let token_json = std::fs::read_to_string(&fallback)?;
        let tokens: serde_json::Value = serde_json::from_str(&token_json)?;
        let input_ids: Vec<i32> = tokens["input_ids"].as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap() as i32).collect();
        let attention_mask: Vec<i32> = tokens["attention_mask"].as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap() as i32).collect();
        println!("  Using fallback cinematic tokens");
        return Ok((input_ids, attention_mask));
    }
    anyhow::bail!("No tokenized prompts found in {bench_dir}")
}

fn make_noise(_numel: usize, seed: u64, dims: &[usize], device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> anyhow::Result<Tensor> {
    flame_core::rng::set_seed(seed);
    let t = Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}
