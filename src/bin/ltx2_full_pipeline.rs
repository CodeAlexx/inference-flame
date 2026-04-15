//! LTX-2.3 end-to-end pure Rust pipeline: prompt → mp4.
//!
//! Zero Python in process. Wires:
//!   1. Gemma-3 text encode + feature extraction
//!   2. LTX-2.3 transformer stage 1 (half-res, 8 distilled steps)
//!   3. Spatial 2× latent upsampler
//!   4. LTX-2.3 transformer stage 2 (full-res, 3 refinement steps)
//!   5. Video VAE decoder (9-block production, cuDNN conv2d per-kD slice)
//!   6. Audio VAE decoder (HEIGHT causal, 3 up stages)
//!   7. BigVGAN vocoder (16 kHz stereo waveform)
//!   8. ffmpeg mux (RGB24 + s16le → mp4 with aresample 16→48 kHz)
//!
//! NOTE: this duplicates the setup code from `ltx2_two_stage.rs`. They
//! should be factored into a shared helper at some point, but for the
//! first end-to-end milestone the duplication is acceptable.
//!
//! Env vars honoured:
//!   LTX2_SKIP_TRANSFORMER=1 — skip stages 1+2 and reuse cached latents
//!     from `ltx2_twostage_{video,audio}_latents.safetensors` (useful for
//!     iterating on the decoder tail without re-running the 20-min
//!     transformer).

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::feature_extractor;
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use inference_flame::models::ltx2_upsampler::LTX2LatentUpsampler;
use inference_flame::mux;
use inference_flame::sampling::ltx2_sampling::{LTX2_DISTILLED_SIGMAS, LTX2_STAGE2_DISTILLED_SIGMAS};
use inference_flame::vae::{LTX2AudioVaeDecoder, LTX2VaeDecoder, LTX2Vocoder};
use std::collections::HashMap;
use std::time::Instant;

// Model paths (kept in sync with ltx2_two_stage.rs).
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";
const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const UPSAMPLER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors";
const VAE_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";

const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

const PROMPT: &str = "A close-up frames a woman pressed flat against a cold metal locker in a dark storage bay, her face half-lit by a single flickering overhead light. Condensation drips down riveted steel walls as she holds her breath, eyes wide, listening. She whispers to herself in a trembling, barely audible voice, Think. Think. The airlock is two decks down. She swallows hard, closing her eyes as a slow, wet scraping sound passes on the other side of the wall. Her lips move again, voice cracking, It can hear you. It can hear your heartbeat. Stop shaking.";
const NUM_FRAMES: usize = 257;
const TARGET_WIDTH: usize = 512;
const TARGET_HEIGHT: usize = 320;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;
const LATENT_CHANNELS: usize = 128;
const AUDIO_CHANNELS: usize = 8;
const AUDIO_MEL_BINS: usize = 16;
const VOCODER_SAMPLE_RATE: u32 = 16000; // base vocoder output SR

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let skip_transformer = std::env::var("LTX2_SKIP_TRANSFORMER").is_ok();

    let s1_width = TARGET_WIDTH / 2;
    let s1_height = TARGET_HEIGHT / 2;
    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;
    let s1_latent_h = s1_height / 32;
    let s1_latent_w = s1_width / 32;
    let s2_latent_h = TARGET_HEIGHT / 32;
    let s2_latent_w = TARGET_WIDTH / 32;
    let video_duration = NUM_FRAMES as f32 / FRAME_RATE;
    let audio_frames = (video_duration * 25.0).round() as usize;

    println!("============================================================");
    println!("LTX-2.3 Full Pipeline — Pure Rust");
    println!("============================================================");
    println!("  Prompt: {:.80}...", PROMPT);
    println!("  Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT} x {NUM_FRAMES} frames @ {FRAME_RATE}fps");
    println!("  Duration: {:.2}s", video_duration);

    let video_latents_path = format!("{OUTPUT_DIR}/ltx2_twostage_video_latents.safetensors");
    let audio_latents_path = format!("{OUTPUT_DIR}/ltx2_twostage_audio_latents.safetensors");

    // ------------------------------------------------------------------
    // Stages 1–2: transformer (optional — skip with LTX2_SKIP_TRANSFORMER)
    // ------------------------------------------------------------------
    let (video_x, audio_x) = if skip_transformer {
        println!("\n[LTX2_SKIP_TRANSFORMER] loading cached latents from {}", video_latents_path);
        let v = flame_core::serialization::load_file(std::path::Path::new(&video_latents_path), &device)?;
        let a = flame_core::serialization::load_file(std::path::Path::new(&audio_latents_path), &device)?;
        let video_x = v.get("latents").ok_or_else(|| anyhow::anyhow!("missing video latents"))?.to_dtype(DType::BF16)?;
        let audio_x = a.get("latents").ok_or_else(|| anyhow::anyhow!("missing audio latents"))?.to_dtype(DType::BF16)?;
        println!("  Video latents: {:?}", video_x.shape().dims());
        println!("  Audio latents: {:?}", audio_x.shape().dims());
        (video_x, audio_x)
    } else {
        run_transformer_pipeline(
            &device,
            latent_f,
            s1_latent_h,
            s1_latent_w,
            s2_latent_h,
            s2_latent_w,
            audio_frames,
            &video_latents_path,
            &audio_latents_path,
        )?
    };

    // ------------------------------------------------------------------
    // Stage 3: Video VAE decode (latents → [B, 3, F, H, W] in [-1, 1])
    // ------------------------------------------------------------------
    println!("\n--- Video VAE decode ---");
    let t0 = Instant::now();
    let video_vae = LTX2VaeDecoder::from_file(VAE_PATH, &device)?;
    println!("  VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let t_decode = Instant::now();
    let video_frames_tensor = video_vae.decode(&video_x)?;
    println!("  Decoded to {:?} in {:.1}s",
        video_frames_tensor.shape().dims(),
        t_decode.elapsed().as_secs_f32());
    drop(video_vae);
    drop(video_x);

    // [B, 3, F, H, W] → F32 CPU vec for u8 quantization
    let v_dims = video_frames_tensor.shape().dims().to_vec();
    let (_b, _c, f_out, h_out, w_out) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3], v_dims[4]);
    let video_f32 = video_frames_tensor.to_dtype(DType::F32)?.to_vec()?;
    drop(video_frames_tensor);
    let rgb_u8 = mux::video_tensor_to_rgb_u8(&video_f32, f_out, h_out, w_out);
    drop(video_f32);
    println!("  RGB24 ready: {f_out} frames, {w_out}x{h_out}");

    // ------------------------------------------------------------------
    // Stage 4: Audio VAE decode (latent → mel)
    // ------------------------------------------------------------------
    println!("\n--- Audio VAE decode ---");
    let t0 = Instant::now();
    let audio_vae = LTX2AudioVaeDecoder::from_file(VAE_PATH, &device)?;
    println!("  Audio VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let t_decode = Instant::now();
    let mel = audio_vae.decode(&audio_x)?;
    println!("  Mel: {:?} in {:.1}s", mel.shape().dims(), t_decode.elapsed().as_secs_f32());
    drop(audio_vae);
    drop(audio_x);

    // ------------------------------------------------------------------
    // Stage 5: Vocoder (mel → 16 kHz stereo waveform)
    // ------------------------------------------------------------------
    println!("\n--- Vocoder ---");
    let t0 = Instant::now();
    let vocoder = LTX2Vocoder::from_file(VAE_PATH, &device, "vocoder")?;
    println!("  Vocoder loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let t_fwd = Instant::now();
    let waveform = vocoder.forward(&mel)?;
    println!(
        "  Waveform: {:?} in {:.1}s",
        waveform.shape().dims(),
        t_fwd.elapsed().as_secs_f32()
    );
    drop(vocoder);
    drop(mel);

    // [B, 2, samples] → interleaved i16 PCM
    let wf_dims = waveform.shape().dims().to_vec();
    let n_samples = wf_dims[2];
    let n_channels = wf_dims[1]; // 2
    let waveform_f32 = waveform.to_dtype(DType::F32)?.to_vec()?;
    drop(waveform);
    let pcm_i16 = mux::audio_tensor_to_pcm_i16(&waveform_f32, n_channels, n_samples, true);
    drop(waveform_f32);
    println!("  PCM: {} stereo samples", pcm_i16.len() / 2);

    // ------------------------------------------------------------------
    // Stage 6: MP4 mux (ffmpeg, aresample 16→48 kHz internally)
    // ------------------------------------------------------------------
    let out_path = format!("{OUTPUT_DIR}/ltx2_full_pipeline.mp4");
    println!("\n--- MP4 mux → {out_path} ---");
    let t0 = Instant::now();
    mux::write_mp4(
        std::path::Path::new(&out_path),
        &rgb_u8,
        f_out,
        w_out,
        h_out,
        FRAME_RATE,
        &pcm_i16,
        VOCODER_SAMPLE_RATE,
    )?;
    println!("  ffmpeg done in {:.1}s", t0.elapsed().as_secs_f32());

    println!("\n============================================================");
    println!("DONE: {out_path}");
    println!("  Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}

/// Run the Gemma-3 text encoder, the LTX-2.3 transformer stages 1+2, and
/// the spatial upsampler. Returns the final (video_latent, audio_latent)
/// tensors ready for VAE decoding. Mirrors the body of `ltx2_two_stage.rs`.
#[allow(clippy::too_many_arguments)]
fn run_transformer_pipeline(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    latent_f: usize,
    s1_latent_h: usize,
    s1_latent_w: usize,
    s2_latent_h: usize,
    s2_latent_w: usize,
    audio_frames: usize,
    video_latents_path: &str,
    audio_latents_path: &str,
) -> anyhow::Result<(Tensor, Tensor)> {
    // --- Text encoding (Gemma-3) ---
    let cache_dir = format!("{OUTPUT_DIR}/embed_cache");
    let video_cache = format!("{cache_dir}/video_context.safetensors");
    let audio_cache = format!("{cache_dir}/audio_context.safetensors");

    let (video_context, audio_context) = if std::path::Path::new(&video_cache).exists()
        && std::path::Path::new(&audio_cache).exists()
    {
        println!("\n--- Text Encoding (cached) ---");
        let vc = flame_core::serialization::load_file(std::path::Path::new(&video_cache), device)?;
        let ac = flame_core::serialization::load_file(std::path::Path::new(&audio_cache), device)?;
        (
            vc.get("video_context").unwrap().to_dtype(DType::BF16)?,
            ac.get("audio_context").unwrap().to_dtype(DType::BF16)?,
        )
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
        let mut encoder = Gemma3Encoder::load(&shard_refs, device, input_ids.len())?;
        let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;
        println!("  Gemma done: {:.1}s", t0.elapsed().as_secs_f32());

        let agg_weights = flame_core::serialization::load_file_filtered(
            std::path::Path::new(LTX_CHECKPOINT),
            device,
            |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
        )?;
        let agg_w = agg_weights
            .get("text_embedding_projection.video_aggregate_embed.weight")
            .ok_or_else(|| anyhow::anyhow!("missing video_aggregate_embed.weight"))?;
        let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");
        let video_context = feature_extractor::feature_extract_and_project(
            &all_hidden, &mask_out, agg_w, agg_b, 4096,
        )?;

        let audio_agg_weights = flame_core::serialization::load_file_filtered(
            std::path::Path::new(LTX_CHECKPOINT),
            device,
            |key| key.starts_with("text_embedding_projection.audio_aggregate_embed"),
        )?;
        let audio_agg_w = audio_agg_weights
            .get("text_embedding_projection.audio_aggregate_embed.weight")
            .ok_or_else(|| anyhow::anyhow!("missing audio_aggregate_embed.weight"))?;
        let audio_agg_b = audio_agg_weights.get("text_embedding_projection.audio_aggregate_embed.bias");
        let audio_context = feature_extractor::feature_extract_and_project(
            &all_hidden, &mask_out, audio_agg_w, audio_agg_b, 2048,
        )?;

        drop(encoder);
        drop(all_hidden);
        drop(agg_weights);
        drop(audio_agg_weights);
        drop(mask_out);
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
        (video_context, audio_context)
    };

    // --- Transformer (22B, BlockOffloader streamed) ---
    println!("\n--- Load Transformer ---");
    let t0 = Instant::now();
    let config = LTX2Config::default();
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    model.init_offloader()?;
    println!("  BlockOffloader ready in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Stage 1: half resolution, 8 distilled steps ---
    println!("\n--- Stage 1 ({} steps) ---", LTX2_DISTILLED_SIGMAS.len() - 1);
    let t0 = Instant::now();
    let s1_video_numel = LATENT_CHANNELS * latent_f * s1_latent_h * s1_latent_w;
    let mut video_x = make_noise(
        s1_video_numel,
        SEED,
        &[1, LATENT_CHANNELS, latent_f, s1_latent_h, s1_latent_w],
        device,
    )?;
    let audio_numel = AUDIO_CHANNELS * audio_frames * AUDIO_MEL_BINS;
    let mut audio_x = make_noise(
        audio_numel,
        SEED + 1,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS],
        device,
    )?;

    let sigmas = LTX2_DISTILLED_SIGMAS.to_vec();
    for step in 0..sigmas.len() - 1 {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();
        let sigma_t = Tensor::from_f32_to_bf16(vec![sigma], Shape::from_dims(&[1]), device.clone())?;
        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t, &video_context, &audio_context, FRAME_RATE, None, None,
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
        println!("  stage1 step {}/{} sigma={:.4} dt={:.1}s",
            step + 1, sigmas.len() - 1, sigma, t_step.elapsed().as_secs_f32());
    }
    println!("  Stage 1 done in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Spatial upsampler (2×) ---
    println!("\n--- Spatial Upsampler ---");
    let t0 = Instant::now();
    let vae_weights = flame_core::serialization::load_file_filtered(
        std::path::Path::new(VAE_PATH), device,
        |key| key == "vae.per_channel_statistics.mean-of-means"
            || key == "vae.per_channel_statistics.std-of-means",
    )?;
    let latents_mean = vae_weights.get("vae.per_channel_statistics.mean-of-means")
        .ok_or_else(|| anyhow::anyhow!("missing latents_mean"))?;
    let latents_std = vae_weights.get("vae.per_channel_statistics.std-of-means")
        .ok_or_else(|| anyhow::anyhow!("missing latents_std"))?;
    let mean_5d = latents_mean.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let std_5d = latents_std.reshape(&[1, LATENT_CHANNELS, 1, 1, 1])?;
    let video_unnorm = video_x.mul(&std_5d)?.add(&mean_5d)?;
    drop(video_x);

    let upsampler = LTX2LatentUpsampler::load(UPSAMPLER_PATH, device)?;
    let video_upscaled = upsampler.forward(&video_unnorm)?;
    drop(upsampler);
    drop(video_unnorm);
    let video_x_re = video_upscaled.sub(&mean_5d)?.div(&std_5d)?;
    drop(vae_weights);
    println!("  Upsampled in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Stage 2: full resolution, 3 refinement steps ---
    let s2_sigmas = LTX2_STAGE2_DISTILLED_SIGMAS.to_vec();
    let s2_steps = s2_sigmas.len() - 1;
    println!("\n--- Stage 2 ({s2_steps} steps) ---");
    let t0 = Instant::now();
    let noise_scale = s2_sigmas[0];
    let s2_video_numel = LATENT_CHANNELS * latent_f * s2_latent_h * s2_latent_w;
    let noise = make_noise(
        s2_video_numel,
        SEED + 100,
        &[1, LATENT_CHANNELS, latent_f, s2_latent_h, s2_latent_w],
        device,
    )?;
    let mut video_x = video_x_re.mul_scalar(1.0 - noise_scale)?
        .add(&noise.mul_scalar(noise_scale)?)?;
    drop(noise);
    let audio_noise = make_noise(
        audio_numel,
        SEED + 101,
        &[1, AUDIO_CHANNELS, audio_frames, AUDIO_MEL_BINS],
        device,
    )?;
    audio_x = audio_x.mul_scalar(1.0 - noise_scale)?
        .add(&audio_noise.mul_scalar(noise_scale)?)?;
    drop(audio_noise);

    for step in 0..s2_steps {
        let sigma = s2_sigmas[step];
        let sigma_next = s2_sigmas[step + 1];
        let t_step = Instant::now();
        let sigma_t = Tensor::from_f32_to_bf16(vec![sigma], Shape::from_dims(&[1]), device.clone())?;
        let (video_vel, audio_vel) = model.forward_audio_video(
            &video_x, &audio_x, &sigma_t, &video_context, &audio_context, FRAME_RATE, None, None,
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
        println!("  stage2 step {}/{} sigma={:.4} dt={:.1}s",
            step + 1, s2_steps, sigma, t_step.elapsed().as_secs_f32());
    }
    println!("  Stage 2 done in {:.1}s", t0.elapsed().as_secs_f32());

    // Drop the transformer BEFORE returning so the caller can load VAEs
    // without OOM — the 22B BlockOffloader model holds substantial GPU memory.
    drop(model);
    let _ = device.synchronize();

    // Save latents to disk so a follow-up `LTX2_SKIP_TRANSFORMER=1` run
    // can iterate on the decoder tail quickly.
    let mut v = HashMap::new();
    v.insert("latents".to_string(), video_x.clone());
    flame_core::serialization::save_tensors(
        &v, std::path::Path::new(video_latents_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    let mut a = HashMap::new();
    a.insert("latents".to_string(), audio_x.clone());
    flame_core::serialization::save_tensors(
        &a, std::path::Path::new(audio_latents_path),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;

    Ok((video_x, audio_x))
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
    _numel: usize,
    seed: u64,
    dims: &[usize],
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<Tensor> {
    flame_core::rng::set_seed(seed);
    let t = Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}
