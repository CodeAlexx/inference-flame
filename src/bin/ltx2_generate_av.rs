//! LTX-2.3 audio+video generation — pure Rust.
//!
//! Pipeline:
//! 1. Run Gemma-3 → FeatureExtractor → video+audio embeddings
//! 2. Load LTX-2 transformer (FP8 resident with audio weights)
//! 3. Create video noise + audio noise
//! 4. Denoise jointly with forward_audio_video
//! 5. Save video + audio latents → decode with Python VAE

use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;
use inference_flame::sampling::ltx2_sampling::LTX2_DISTILLED_SIGMAS;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

// Model paths
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";

const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

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

    println!("============================================================");
    println!("LTX-2.3 Audio+Video Generation — Pure Rust");
    println!("============================================================");
    println!("  {}×{}, {} frames, {} steps", WIDTH, HEIGHT, NUM_FRAMES, NUM_STEPS);

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
    println!("  Tokenizing...");
    let (input_ids, attention_mask) = simple_tokenize(PROMPT, 256)?;
    let real_count = attention_mask.iter().filter(|&&m| m != 0).count();
    println!("  {} tokens ({} real)", input_ids.len(), real_count);

    // Load and run Gemma
    println!("  Loading Gemma-3...");
    let mut encoder = Gemma3Encoder::load(&shard_refs, &device, input_ids.len())?;
    println!("  Running Gemma forward...");
    let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;
    println!("  Gemma done: {} hidden states in {:.1}s", all_hidden.len(), t0.elapsed().as_secs_f32());

    // Feature extraction
    println!("  Loading aggregate_embed...");
    let agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT),
        &device,
        |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
    )?;
    let agg_w = agg_weights.get("text_embedding_projection.video_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing video_aggregate_embed.weight"))?;
    let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");

    println!("  Running video feature extractor (→ 4096)...");
    let video_context = feature_extractor::feature_extract_and_project(
        &all_hidden,
        &mask_out,
        agg_w,
        agg_b,
        4096,
    )?;
    println!("  Video context: {:?}", video_context.dims());

    // Audio feature extraction: same Gemma hidden states → audio_aggregate_embed → 2048
    println!("  Loading audio_aggregate_embed...");
    let audio_agg_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(LTX_CHECKPOINT),
        &device,
        |key| key.starts_with("text_embedding_projection.audio_aggregate_embed"),
    )?;
    let audio_agg_w = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing audio_aggregate_embed.weight"))?;
    let audio_agg_b = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.bias");

    println!("  Running audio feature extractor (→ 2048)...");
    let audio_context = feature_extractor::feature_extract_and_project(
        &all_hidden,
        &mask_out,
        audio_agg_w,
        audio_agg_b,
        2048,
    )?;
    println!("  Audio context: {:?}", audio_context.dims());

    // Free Gemma + feature extraction to reclaim ALL VRAM for DiT
    drop(encoder);
    drop(all_hidden);
    drop(agg_weights);
    drop(audio_agg_weights);
    drop(mask_out);
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

    match model.load_fp8_resident() {
        Ok(()) => println!("  FP8 resident loaded in {:.1}s", t0.elapsed().as_secs_f32()),
        Err(e) => {
            println!("  FP8 resident failed ({e}), falling back to FlameSwap");
            model.init_swap()?;
            println!("  FlameSwap initialized in {:.1}s", t0.elapsed().as_secs_f32());
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
    // Stage 4: Denoise (audio + video jointly)
    // ========================================
    println!("\n--- Stage 4: Denoise ({} steps, AV joint) ---", NUM_STEPS);
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

        // Joint AV forward — no CFG for distilled model
        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t,
            &video_context, &audio_context,
            FRAME_RATE,
            None, None,  // attention masks: None = pre-projected embeddings
        )?;

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

/// Simple tokenizer: <bos>(2) + byte-level tokens, left-padded to max_len.
/// This is a placeholder — production should use the HuggingFace tokenizer.
fn simple_tokenize(text: &str, max_len: usize) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
    // For now, load pre-tokenized data if available
    let bench_dir = "/home/alex/ltx2-refs/bench20";

    // Try to find a matching prompt in bench20
    let prompts_path = format!("{bench_dir}/prompts.json");
    if std::path::Path::new(&prompts_path).exists() {
        let prompts_json = std::fs::read_to_string(&prompts_path)?;
        let prompts: Vec<String> = serde_json::from_str(&prompts_json)?;

        for (i, p) in prompts.iter().enumerate() {
            if p == text {
                let token_path = format!("{bench_dir}/tokens_{i:02}.json");
                let token_json = std::fs::read_to_string(&token_path)?;
                let tokens: serde_json::Value = serde_json::from_str(&token_json)?;

                let input_ids: Vec<i32> = tokens["input_ids"]
                    .as_array().unwrap()
                    .iter().map(|v| v.as_i64().unwrap() as i32)
                    .collect();
                let attention_mask: Vec<i32> = tokens["attention_mask"]
                    .as_array().unwrap()
                    .iter().map(|v| v.as_i64().unwrap() as i32)
                    .collect();

                return Ok((input_ids, attention_mask));
            }
        }
    }

    // Scan ALL token files in bench dir for matching prompt
    if let Ok(entries) = std::fs::read_dir(bench_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false)
                && path.file_name().map(|n| n.to_str().unwrap_or("").starts_with("tokens")).unwrap_or(false)
            {
                if let Ok(token_json) = std::fs::read_to_string(&path) {
                    if let Ok(tokens) = serde_json::from_str::<serde_json::Value>(&token_json) {
                        let stored = tokens["prompt"].as_str().unwrap_or("");
                        // Fuzzy match: compare first 60 chars to handle quote differences
                        if stored.len() > 60 && text.len() > 60 && stored[..60] == text[..60] {
                            let input_ids: Vec<i32> = tokens["input_ids"]
                                .as_array().unwrap()
                                .iter().map(|v| v.as_i64().unwrap() as i32)
                                .collect();
                            let attention_mask: Vec<i32> = tokens["attention_mask"]
                                .as_array().unwrap()
                                .iter().map(|v| v.as_i64().unwrap() as i32)
                                .collect();
                            eprintln!("  Matched token file: {:?}", path.file_name());
                            return Ok((input_ids, attention_mask));
                        }
                    }
                }
            }
        }
    }

    let ref_path = "/home/alex/ltx2-refs/gemma3/tokens.json";
    if std::path::Path::new(ref_path).exists() {
        let token_json = std::fs::read_to_string(ref_path)?;
        let tokens: serde_json::Value = serde_json::from_str(&token_json)?;

        let input_ids: Vec<i32> = tokens["input_ids"]
            .as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let attention_mask: Vec<i32> = tokens["attention_mask"]
            .as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap() as i32)
            .collect();

        return Ok((input_ids, attention_mask));
    }

    anyhow::bail!("No tokenized data found for prompt. Run bench_20_prompts.py first.");
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
