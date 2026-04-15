//! ACE-Step music generation — pure Rust, no Python.
//!
//! End-to-end pipeline:
//! 1. Tokenize prompt + lyrics with Qwen3-Embedding-0.6B tokenizer
//! 2. Encode text through Qwen3-Embedding-0.6B (extract last hidden state)
//! 3. Encode lyrics through embed_tokens only (embedding lookup)
//! 4. Load ACE-Step model, build condition (text2music path)
//! 5. Generate noise, run Euler sampler
//! 6. VAE decode latents to waveform
//! 7. Save as WAV (48 kHz stereo)
//!
//! Usage:
//!   cargo run --release --bin acestep_gen -- \
//!     --prompt "pop rock female vocal upbeat" \
//!     --lyrics "La la la / Sing along" \
//!     --duration 30 \
//!     --output output.wav

use flame_core::{global_cuda_device, serialization, CudaDevice, DType, Shape, Tensor};
use inference_flame::models::acestep_condition::AceStepConditionEncoder;
use inference_flame::models::acestep_dit::AceStepDiT;
use inference_flame::sampling::acestep_sampling::acestep_sample;
use inference_flame::vae::OobleckVaeDecoder;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_MODEL_DIR: &str =
    "/home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo";
const LATENT_RATE: usize = 25; // 25 Hz latent rate (48kHz / 1920 downsample)
const SAMPLE_RATE: u32 = 48_000;
const NUM_CHANNELS: u16 = 2; // stereo
const TEXT_MAX_LEN: usize = 256;
const LYRIC_MAX_LEN: usize = 2048;
const _SILENCE_FRAMES: usize = 750; // 30 seconds at 25 Hz

// ---------------------------------------------------------------------------
// CLI arg parsing (minimal, no external crate)
// ---------------------------------------------------------------------------

struct Args {
    prompt: String,
    lyrics: String,
    duration: usize,
    output: PathBuf,
    steps: usize,
    seed: u64,
    model_dir: PathBuf,
    cfg_scale: f32,
    shift: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut prompt = String::from("pop rock female vocal energetic upbeat");
    let mut lyrics = String::new();
    let mut duration: usize = 30;
    let mut output = PathBuf::from("acestep_output.wav");
    let mut steps: usize = 8;
    let mut seed: u64 = 42;
    let mut model_dir = PathBuf::from(DEFAULT_MODEL_DIR);
    let mut cfg_scale: f32 = 1.0; // turbo uses no CFG
    let mut shift: f32 = 1.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--lyrics" => {
                i += 1;
                lyrics = args[i].clone();
            }
            "--duration" => {
                i += 1;
                duration = args[i].parse().expect("invalid --duration");
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("invalid --steps");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("invalid --seed");
            }
            "--model-dir" => {
                i += 1;
                model_dir = PathBuf::from(&args[i]);
            }
            "--cfg-scale" => {
                i += 1;
                cfg_scale = args[i].parse().expect("invalid --cfg-scale");
            }
            "--shift" => {
                i += 1;
                shift = args[i].parse().expect("invalid --shift");
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: acestep_gen [options]\n\
                     Options:\n  \
                       --prompt TEXT       Text prompt (genre/mood/style)\n  \
                       --lyrics TEXT       Song lyrics (use / for line breaks)\n  \
                       --duration SECS     Duration in seconds (default: 30)\n  \
                       --output PATH       Output WAV path (default: acestep_output.wav)\n  \
                       --steps N           Denoising steps (default: 8 for turbo)\n  \
                       --seed N            Random seed (default: 42)\n  \
                       --model-dir PATH    Model checkpoint directory\n  \
                       --cfg-scale F       CFG guidance scale (default: 1.0 = no CFG)\n  \
                       --shift F           Timestep shift (default: 1.0)\n"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        prompt,
        lyrics,
        duration,
        output,
        steps,
        seed,
        model_dir,
        cfg_scale,
        shift,
    }
}

// ---------------------------------------------------------------------------
// Qwen3 text encoder (using existing Qwen3Encoder)
// ---------------------------------------------------------------------------

use inference_flame::models::qwen3_encoder::{Qwen3Config, Qwen3Encoder};

/// Load Qwen3-Embedding-0.6B weights, adding `model.` prefix to match
/// the Qwen3Encoder's expected key format.
fn load_text_encoder(
    encoder_dir: &Path,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<(Qwen3Encoder, HashMap<String, Tensor>)> {
    let safetensors_path = encoder_dir.join("model.safetensors");
    eprintln!("Loading text encoder from: {}", safetensors_path.display());

    let raw_weights = serialization::load_file(&safetensors_path, device)?;
    eprintln!("  {} keys loaded (raw)", raw_weights.len());

    // Add `model.` prefix to all keys to match Qwen3Encoder expectations
    let mut prefixed: HashMap<String, Tensor> = HashMap::with_capacity(raw_weights.len());
    for (key, tensor) in &raw_weights {
        prefixed.insert(format!("model.{key}"), tensor.clone());
    }

    let config = Qwen3Encoder::config_from_weights(&prefixed)?;
    // For ACE-Step: extract the last layer's output (0-indexed = num_layers - 1)
    let last_layer = config.num_layers - 1;
    let config = Qwen3Config {
        extract_layers: vec![last_layer],
        ..config
    };
    eprintln!(
        "  Config: hidden={}, layers={}, heads={}, kv_heads={}, extract=[{}]",
        config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads, last_layer,
    );

    let encoder = Qwen3Encoder::new(prefixed, config, device.clone());

    // Keep raw weights around so we can extract embed_tokens for lyric embedding
    Ok((encoder, raw_weights))
}

/// Encode text through the full Qwen3 model and apply final RMSNorm.
///
/// Returns [1, seq_len, hidden_size] in BF16.
fn encode_text(
    encoder: &Qwen3Encoder,
    token_ids: &[i32],
    norm_weight: &Tensor,
) -> anyhow::Result<Tensor> {
    let hidden = encoder.encode(token_ids)?;
    // Apply final RMSNorm (the Qwen3Encoder extracts before final norm)
    let dims = hidden.shape().dims().to_vec();
    let hidden_size = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let h_2d = hidden.reshape(&[batch, hidden_size])?;
    let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&h_2d, Some(norm_weight), 1e-6)?;
    let result = normed.reshape(&dims)?;
    Ok(result)
}

/// Look up lyric token embeddings from the text encoder's embedding table.
///
/// Returns [1, seq_len, hidden_size] in BF16.
fn embed_lyrics(
    raw_weights: &HashMap<String, Tensor>,
    token_ids: &[i32],
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    let embed_w = raw_weights
        .get("embed_tokens.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing embed_tokens.weight in text encoder"))?;
    let embed_w = embed_w.to_dtype(DType::BF16)?;

    let seq_len = token_ids.len();
    let ids_tensor = Tensor::from_vec(
        token_ids.iter().map(|&id| id as f32).collect(),
        Shape::from_dims(&[seq_len]),
        device.clone(),
    )?
    .to_dtype(DType::I32)?;

    let selected = embed_w.index_select0(&ids_tensor)?;
    let result = selected.unsqueeze(0)?; // [1, seq_len, hidden_size]
    Ok(result)
}

// ---------------------------------------------------------------------------
// WAV writer (minimal, no external crate)
// ---------------------------------------------------------------------------

/// Write a WAV file from interleaved f32 samples.
///
/// `samples`: interleaved stereo samples in [-1, 1] range.
/// `sample_rate`: e.g., 48000.
/// `num_channels`: e.g., 2.
fn write_wav(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> anyhow::Result<()> {
    use std::io::Write;

    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = num_channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;
    let data_size = (samples.len() * bytes_per_sample as usize) as u32;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path)?;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM format
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;

    // Convert f32 samples to i16 and write
    for &s in samples {
        let clamped = s.max(-1.0).min(1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Noise generation (Box-Muller)
// ---------------------------------------------------------------------------

fn generate_noise(
    shape: &[usize],
    seed: u64,
    device: Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    use rand::prelude::*;

    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);

    for _ in 0..numel / 2 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data.push(r * theta.cos());
        data.push(r * theta.sin());
    }
    if numel % 2 == 1 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data.push(r * theta.cos());
    }
    data.truncate(numel);

    let t = Tensor::from_f32_to_bf16(data, Shape::from_dims(shape), device)?;
    Ok(t)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args();
    let t_total = Instant::now();
    let device = global_cuda_device();

    println!("============================================================");
    println!("ACE-Step Music Generation — Pure Rust (inference-flame)");
    println!("============================================================");
    println!("  Prompt:   {}", args.prompt);
    println!("  Lyrics:   {}", if args.lyrics.is_empty() { "(none)" } else { &args.lyrics });
    println!("  Duration: {}s", args.duration);
    println!("  Steps:    {}", args.steps);
    println!("  Seed:     {}", args.seed);
    println!("  CFG:      {}", args.cfg_scale);
    println!("  Output:   {}", args.output.display());
    println!();

    let model_dir = &args.model_dir;
    let encoder_dir = model_dir.join("../Qwen3-Embedding-0.6B");
    let vae_path = model_dir.join("../vae/diffusion_pytorch_model.safetensors");
    let tokenizer_path = encoder_dir.join("tokenizer.json");

    // Compute latent dimensions
    let num_timesteps = args.duration * LATENT_RATE;

    // ------------------------------------------------------------------
    // Stage 1: Tokenize prompt and lyrics
    // ------------------------------------------------------------------
    println!("--- Stage 1: Tokenize ---");
    let t0 = Instant::now();

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e))?;

    let pad_id = 151643i32; // Qwen pad/eos token

    // Tokenize prompt
    let prompt_enc = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Prompt tokenization failed: {}", e))?;
    let mut prompt_ids: Vec<i32> = prompt_enc.get_ids().iter().map(|&id| id as i32).collect();
    let prompt_real_len = prompt_ids.len();
    prompt_ids.resize(TEXT_MAX_LEN, pad_id);
    println!("  Prompt tokens: {} (padded to {})", prompt_real_len, TEXT_MAX_LEN);

    // Build text attention mask
    let mut text_mask_data = vec![0.0f32; TEXT_MAX_LEN];
    for i in 0..prompt_real_len.min(TEXT_MAX_LEN) {
        text_mask_data[i] = 1.0;
    }

    // Tokenize lyrics
    let lyrics_text = if args.lyrics.is_empty() {
        " ".to_string() // at least one token for the encoder
    } else {
        args.lyrics.replace('/', "\n")
    };
    let lyric_enc = tokenizer
        .encode(lyrics_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Lyric tokenization failed: {}", e))?;
    let mut lyric_ids: Vec<i32> = lyric_enc.get_ids().iter().map(|&id| id as i32).collect();
    let lyric_real_len = lyric_ids.len();
    let lyric_padded_len = lyric_ids.len().min(LYRIC_MAX_LEN);
    lyric_ids.truncate(LYRIC_MAX_LEN);
    lyric_ids.resize(LYRIC_MAX_LEN, pad_id);
    println!("  Lyric tokens: {} (padded to {})", lyric_real_len, LYRIC_MAX_LEN);

    // Build lyric attention mask
    let mut lyric_mask_data = vec![0.0f32; LYRIC_MAX_LEN];
    for i in 0..lyric_padded_len.min(LYRIC_MAX_LEN) {
        lyric_mask_data[i] = 1.0;
    }

    println!("  Tokenized in {:.2}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: Text encoding (Qwen3-Embedding-0.6B)
    // ------------------------------------------------------------------
    println!("--- Stage 2: Text Encoding ---");
    let t0 = Instant::now();

    let (text_encoder, raw_encoder_weights) = load_text_encoder(&encoder_dir, &device)?;

    // Get final norm weight for post-extraction normalization
    let norm_weight = raw_encoder_weights
        .get("norm.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing norm.weight in text encoder"))?
        .to_dtype(DType::BF16)?;

    // Encode prompt text (full forward pass + final norm)
    let text_hidden_states = encode_text(&text_encoder, &prompt_ids, &norm_weight)?;
    println!(
        "  text_hidden_states: {:?}",
        text_hidden_states.shape().dims()
    );

    // Embed lyrics (embedding table lookup only, no transformer layers)
    let lyric_hidden_states = embed_lyrics(&raw_encoder_weights, &lyric_ids, &device)?;
    println!(
        "  lyric_hidden_states: {:?}",
        lyric_hidden_states.shape().dims()
    );

    println!("  Encoded in {:.2}s", t0.elapsed().as_secs_f32());

    // Free text encoder
    drop(text_encoder);
    drop(raw_encoder_weights);
    println!("  Text encoder freed.");
    println!();

    // Build mask tensors on device
    let text_attention_mask = Tensor::from_vec(
        text_mask_data,
        Shape::from_dims(&[1, TEXT_MAX_LEN]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let lyric_attention_mask = Tensor::from_vec(
        lyric_mask_data,
        Shape::from_dims(&[1, LYRIC_MAX_LEN]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    // ------------------------------------------------------------------
    // Stage 3: Load ACE-Step model + condition encoder
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load ACE-Step Model ---");
    let t0 = Instant::now();

    // Load all weights from checkpoint
    let model_path = model_dir.join("model.safetensors");
    eprintln!("Loading from: {}", model_path.display());
    let all_weights = serialization::load_file(&model_path, &device)?;
    eprintln!("  {} total keys", all_weights.len());

    // Split weights: encoder.* goes to condition encoder, decoder.* + null_condition_emb to DiT
    let mut encoder_weights: HashMap<String, Tensor> = HashMap::new();
    let mut dit_weights: HashMap<String, Tensor> = HashMap::new();

    for (key, tensor) in all_weights {
        if key.starts_with("encoder.") {
            encoder_weights.insert(key, tensor);
        } else {
            // decoder.* and null_condition_emb
            dit_weights.insert(key, tensor);
        }
    }
    eprintln!(
        "  encoder: {} keys, decoder: {} keys",
        encoder_weights.len(),
        dit_weights.len()
    );

    // Build condition encoder
    let cond_encoder = AceStepConditionEncoder::from_weights(encoder_weights, device.clone());

    // Build DiT
    let mut dit = AceStepDiT::from_weights(dit_weights)?;

    println!("  Model loaded in {:.2}s", t0.elapsed().as_secs_f32());
    println!("  DiT config: {:?}", dit.config());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Prepare condition (text2music path)
    // ------------------------------------------------------------------
    println!("--- Stage 4: Prepare Condition ---");
    let t0 = Instant::now();

    // Load real silence latent (converted from .pt to .safetensors)
    let silence_safetensors = model_dir.join("silence_latent.safetensors");
    let silence_latent = if silence_safetensors.exists() {
        let sl_map = serialization::load_file(&silence_safetensors, &device)?;
        let sl = sl_map.get("silence_latent")
            .ok_or_else(|| anyhow::anyhow!("silence_latent key not found"))?;
        // sl is [1, 64, N] channel-first → permute to [1, N, 64] then narrow to T
        let sl_bf16 = sl.to_dtype(DType::BF16)?;
        let sl_perm = sl_bf16.transpose_dims(1, 2)?; // [1, N, 64]
        sl_perm.narrow(1, 0, num_timesteps)? // [1, T, 64]
    } else {
        eprintln!("  (silence_latent.safetensors not found, using zeros placeholder)");
        Tensor::zeros_dtype(Shape::from_dims(&[1, num_timesteps, 64]), DType::BF16, device.clone())?
    };

    let (encoder_hidden_states, _encoder_attention_mask, context_latents) =
        cond_encoder.prepare_condition(
            &text_hidden_states,
            &text_attention_mask,
            &lyric_hidden_states,
            &lyric_attention_mask,
            &silence_latent,
            num_timesteps,
        )?;

    println!(
        "  encoder_hidden_states: {:?}",
        encoder_hidden_states.shape().dims()
    );
    println!("  context_latents: {:?}", context_latents.shape().dims());
    println!("  Prepared in {:.2}s", t0.elapsed().as_secs_f32());

    // Free condition encoder
    drop(cond_encoder);
    drop(text_hidden_states);
    drop(lyric_hidden_states);
    println!("  Condition encoder freed.");
    println!();

    // ------------------------------------------------------------------
    // Stage 5: Generate noise and run Euler sampler
    // ------------------------------------------------------------------
    println!(
        "--- Stage 5: Denoise ({} steps, CFG={}, shift={}) ---",
        args.steps, args.cfg_scale, args.shift
    );
    let t0 = Instant::now();

    // Noise: [1, T, 64]
    let noise = generate_noise(&[1, num_timesteps, 64], args.seed, device.clone())?;
    println!("  noise: {:?}", noise.shape().dims());

    // Get null condition embedding for CFG (if used)
    let null_emb = if args.cfg_scale > 1.0 {
        Some(dit.null_condition_emb()?.clone())
    } else {
        None
    };

    let latents = acestep_sample(
        &mut dit,
        &noise,
        &encoder_hidden_states,
        &context_latents,
        args.steps,
        args.cfg_scale,
        args.shift,
        null_emb.as_ref(),
    )?;

    println!("  latents: {:?}", latents.shape().dims());
    println!("  Denoised in {:.2}s", t0.elapsed().as_secs_f32());

    // Free DiT
    drop(dit);
    drop(encoder_hidden_states);
    drop(context_latents);
    drop(noise);
    println!("  DiT freed.");
    println!();

    // ------------------------------------------------------------------
    // Stage 6: VAE decode
    // ------------------------------------------------------------------
    println!("--- Stage 6: VAE Decode ---");
    let t0 = Instant::now();

    let vae = OobleckVaeDecoder::from_safetensors(
        vae_path.to_str().unwrap(),
        &device,
    )?;

    // VAE expects [B, 64, T] (channel-first), our latents are [B, T, 64]
    let latents_cf = latents.permute(&[0, 2, 1])?;
    println!("  latents (channel-first): {:?}", latents_cf.shape().dims());

    let waveform = vae.decode(&latents_cf)?;
    println!("  waveform: {:?}", waveform.shape().dims());

    let expected_samples = num_timesteps * 1920; // 1920 = product of upsampling ratios
    println!("  Expected audio samples: {}", expected_samples);
    println!("  VAE decoded in {:.2}s", t0.elapsed().as_secs_f32());

    // Free VAE
    drop(vae);
    drop(latents);
    drop(latents_cf);
    println!("  VAE freed.");
    println!();

    // ------------------------------------------------------------------
    // Stage 7: Save WAV
    // ------------------------------------------------------------------
    println!("--- Stage 7: Save WAV ---");
    let t0 = Instant::now();

    // Waveform is [1, 2, T_audio] in BF16 -> convert to f32 interleaved stereo
    let wav_f32 = waveform.to_dtype(DType::F32)?;
    let wav_dims = wav_f32.shape().dims().to_vec();
    let audio_len = wav_dims[2];

    // Extract channels: [1, 2, T] -> channel 0 and channel 1
    let ch0 = wav_f32.narrow(1, 0, 1)?.reshape(&[audio_len])?;
    let ch1 = wav_f32.narrow(1, 1, 1)?.reshape(&[audio_len])?;

    let ch0_data = ch0.to_vec1()?;
    let ch1_data = ch1.to_vec1()?;

    // Interleave stereo samples: [L0, R0, L1, R1, ...]
    let mut interleaved = Vec::with_capacity(audio_len * 2);
    for i in 0..audio_len {
        interleaved.push(ch0_data[i]);
        interleaved.push(ch1_data[i]);
    }

    write_wav(&args.output, &interleaved, SAMPLE_RATE, NUM_CHANNELS)?;
    let file_size = std::fs::metadata(&args.output)?.len();
    println!(
        "  Saved: {} ({:.1} MB, {:.1}s audio)",
        args.output.display(),
        file_size as f64 / 1_048_576.0,
        audio_len as f64 / SAMPLE_RATE as f64,
    );
    println!("  WAV written in {:.2}s", t0.elapsed().as_secs_f32());
    println!();

    println!(
        "Total time: {:.2}s",
        t_total.elapsed().as_secs_f32()
    );

    Ok(())
}
