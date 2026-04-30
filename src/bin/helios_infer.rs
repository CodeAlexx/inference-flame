//! Helios-Distilled 14B video DiT — pure-Rust T2V inference.
//!
//! Mirrors `diffusers/pipelines/helios/pipeline_helios_pyramid.py`'s
//! `__call__` for the T2V (no image, no video) branch with the
//! Distilled config (is_distilled=true, guidance=1.0 → no CFG, 3-stage
//! pyramid × 2 steps = 6 denoise calls per chunk, autoregressive
//! 33-frame chunks).
//!
//! Weights expected at `~/.cache/huggingface/hub/models--BestWishYsh--Helios-Distilled/snapshots/*/`
//! (override with `HELIOS_SNAPSHOT`):
//!   - `transformer/`        (40-layer DiT, F32 → BF16 at load)
//!   - `text_encoder/`       (UMT5-XXL)
//!   - `tokenizer/tokenizer.json`
//! VAE expected at `~/.cache/huggingface/hub/models--ai-toolkit--wan2.1-vae/snapshots/*/diffusion_pytorch_model.safetensors`
//! (override with `HELIOS_VAE_PATH`).
//!
//! Usage:
//!   cargo run --release --bin helios_infer -- \
//!       --prompt "..." \
//!       [--negative "..."] \
//!       [--width 384] [--height 256] [--frames 33] \
//!       [--seed 42] [--out output/helios_out.mp4]
//!
//! For first-smoke at 33 frames: pass --frames 33 (single chunk, ~3 min on 3090 Ti).
//! Full 132 frames = 4 chunks, ~10-15 min.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::helios_dit::HeliosInferDit;
use inference_flame::models::wan::t5::Umt5Encoder;
use inference_flame::sampling::helios_dmd::{HeliosDMDConfig, HeliosDMDState};
use inference_flame::sampling::helios_pyramid::{
    build_history_indices, calculate_shift, sample_block_noise,
};
use inference_flame::vae::wan21_vae::Wan21VaeDecoder;

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Wan 2.1 VAE temporal stride: 4 latent frames map to (4-1)*4+1 = 13 px frames.
/// For a 9-latent-frame chunk: (9-1)*4+1 = 33 px frames.
const VAE_TEMPORAL_STRIDE: usize = 4;
const VAE_SPATIAL_STRIDE: usize = 8;
const Z_DIM: usize = 16;

/// Default Helios-Distilled config from `transformer/config.json`.
const NUM_LATENT_FRAMES_PER_CHUNK: usize = 9;
/// Distilled: [2, 2, 2] = 6 total denoise calls per chunk.
const PYRAMID_NUM_INFERENCE_STEPS: [usize; 3] = [2, 2, 2];
/// History sizes (sorted descending: long, mid, short_history).
const HISTORY_SIZES: [usize; 3] = [16, 2, 1];

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

struct Args {
    prompt: String,
    negative: String,
    width: usize,
    height: usize,
    num_frames: usize,
    seed: u64,
    output: PathBuf,
    snapshot: PathBuf,
    vae_path: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut prompt: Option<String> = None;
    let mut negative: Option<String> = None;
    let mut width: usize = 384;
    let mut height: usize = 256;
    let mut num_frames: usize = 33;
    let mut seed: u64 = 42;
    let mut output: Option<PathBuf> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let next = || -> Result<&str> {
            argv.get(i + 1)
                .map(|s| s.as_str())
                .ok_or_else(|| anyhow!("missing value for {}", argv[i]))
        };
        match argv[i].as_str() {
            "--prompt" => { prompt = Some(next()?.to_string()); i += 2; }
            "--negative" => { negative = Some(next()?.to_string()); i += 2; }
            "--width" => { width = next()?.parse()?; i += 2; }
            "--height" => { height = next()?.parse()?; i += 2; }
            "--frames" => { num_frames = next()?.parse()?; i += 2; }
            "--seed" => { seed = next()?.parse()?; i += 2; }
            "--out" | "--output" => { output = Some(PathBuf::from(next()?)); i += 2; }
            "--help" | "-h" => {
                eprintln!(
                    "helios_infer\n\n\
                    Usage: helios_infer --prompt TEXT [--negative TEXT] [--width N] \
                    [--height N] [--frames N] [--seed N] [--out PATH]\n\n\
                    Defaults: width=384, height=256, frames=33, seed=42, \
                    out=inference-flame/output/helios_out.mp4\n\n\
                    Env overrides: HELIOS_SNAPSHOT (default: HF cache), \
                    HELIOS_VAE_PATH (default: HF cache)\n"
                );
                std::process::exit(0);
            }
            other => {
                return Err(anyhow!("unknown arg: {}", other));
            }
        }
    }

    let prompt = prompt.ok_or_else(|| anyhow!("--prompt required (use --help)"))?;
    if width % 16 != 0 || height % 16 != 0 {
        return Err(anyhow!(
            "--width and --height must be divisible by 16 (got {width}x{height})"
        ));
    }
    let window_num_frames =
        (NUM_LATENT_FRAMES_PER_CHUNK - 1) * VAE_TEMPORAL_STRIDE + 1; // 33
    if num_frames < window_num_frames {
        return Err(anyhow!(
            "--frames must be at least {window_num_frames} (one chunk); got {num_frames}"
        ));
    }

    let snapshot = std::env::var("HELIOS_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default: HF cache snapshot (resolve glob).
            let pat = "/home/alex/.cache/huggingface/hub/models--BestWishYsh--Helios-Distilled/snapshots";
            std::fs::read_dir(pat)
                .ok()
                .and_then(|mut it| it.next())
                .and_then(|r| r.ok())
                .map(|e| e.path())
                .expect("Helios snapshot not found; set HELIOS_SNAPSHOT")
        });
    // Wan 2.1 VAE — must be in the ORIGINAL Wan format (top-level conv1/conv2,
    // decoder.middle.0/1/2, decoder.upsamples.{N}, decoder.head.{0,2}).
    // Wan21VaeDecoder::load expects this format. The diffusers-style
    // ai-toolkit/wan2.1-vae file uses different key names (decoder.conv_in,
    // decoder.up_blocks.{i}.resnets.{j}, ...) and would need
    // QwenImageVaeDecoder's remap_qwenimage_to_wan21.
    let vae_path = std::env::var("HELIOS_VAE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/alex/.serenity/models/vaes/wan_2.1_vae.safetensors"));

    let output = output
        .unwrap_or_else(|| PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/helios_out.mp4"));

    Ok(Args {
        prompt,
        negative: negative.unwrap_or_default(),
        width,
        height,
        num_frames,
        seed,
        output,
        snapshot,
        vae_path,
    })
}

// -----------------------------------------------------------------------------
// Tokenization (T5TokenizerFast → token ids)
// -----------------------------------------------------------------------------

fn tokenize(tokenizer_path: &Path, text: &str) -> Result<Vec<i32>> {
    use tokenizers::Tokenizer;
    let tok = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load: {e}"))?;
    let enc = tok
        .encode(text, true)
        .map_err(|e| anyhow!("tokenizer encode: {e}"))?;
    Ok(enc.get_ids().iter().map(|&u| u as i32).collect())
}

// -----------------------------------------------------------------------------
// Pyramid downsample / upsample helpers (operate on (B, C, F, H, W))
// -----------------------------------------------------------------------------

/// Pyramid downsample: 2 successive bilinear-by-2 spatial downsamples (each
/// followed by ×2 multiply, mirroring diffusers's `* 2` on each pass).
/// Final shape: (B, C, F, H/4, W/4).
fn pyramid_downsample(x: &Tensor) -> Result<Tensor> {
    use flame_core::cuda_ops::GpuOps;
    let dims = x.shape().dims().to_vec();
    let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let cur = x.permute(&[0, 2, 1, 3, 4]).map_err(|e| anyhow!("{e:?}"))?
        .contiguous().map_err(|e| anyhow!("{e:?}"))?; // (B, F, C, H, W)
    let mut cur = cur.reshape(&[b * f, c, h, w]).map_err(|e| anyhow!("{e:?}"))?;
    let mut cur_h = h;
    let mut cur_w = w;
    for _ in 0..2 {
        cur_h /= 2;
        cur_w /= 2;
        let dn = GpuOps::upsample2d_bilinear(&cur, (cur_h, cur_w), false)
            .map_err(|e| anyhow!("{e:?}"))?;
        cur = dn.mul_scalar(2.0).map_err(|e| anyhow!("{e:?}"))?;
    }
    let out = cur.reshape(&[b, f, c, cur_h, cur_w]).map_err(|e| anyhow!("{e:?}"))?;
    Ok(out.permute(&[0, 2, 1, 3, 4]).map_err(|e| anyhow!("{e:?}"))?
        .contiguous().map_err(|e| anyhow!("{e:?}"))?)
}

/// Nearest-neighbor spatial upsample by 2× (T preserved).
fn nearest_upsample_2x(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let cur = x.permute(&[0, 2, 1, 3, 4]).map_err(|e| anyhow!("{e:?}"))?
        .contiguous().map_err(|e| anyhow!("{e:?}"))?;
    let cur = cur.reshape(&[b * f, c, h, w]).map_err(|e| anyhow!("{e:?}"))?;
    let up = cur.upsample_nearest2d(h * 2, w * 2).map_err(|e| anyhow!("{e:?}"))?;
    let out = up.reshape(&[b, f, c, h * 2, w * 2]).map_err(|e| anyhow!("{e:?}"))?;
    Ok(out.permute(&[0, 2, 1, 3, 4]).map_err(|e| anyhow!("{e:?}"))?
        .contiguous().map_err(|e| anyhow!("{e:?}"))?)
}

/// Initial Gaussian noise for one chunk: (B=1, 16, F=9, H/8, W/8) F32.
fn make_chunk_noise(
    height: usize,
    width: usize,
    seed: u64,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    let h_lat = height / VAE_SPATIAL_STRIDE;
    let w_lat = width / VAE_SPATIAL_STRIDE;
    let shape = Shape::from_dims(&[1, Z_DIM, NUM_LATENT_FRAMES_PER_CHUNK, h_lat, w_lat]);
    let n = Tensor::randn_seeded(shape, 0.0, 1.0, seed, device)?;
    Ok(n)
}

// -----------------------------------------------------------------------------
// Per-chunk denoise loop
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn denoise_one_chunk(
    dit: &mut HeliosInferDit,
    scheduler: &mut HeliosDMDState,
    prompt_embeds: &Tensor,
    initial_noise: Tensor,
    history_short: &Tensor,
    history_mid: &Tensor,
    history_long: &Tensor,
    indices_hidden: &[f32],
    indices_short: &[f32],
    indices_mid: &[f32],
    indices_long: &[f32],
    seed: u64,
    chunk_idx: usize,
) -> Result<Tensor> {
    let device = initial_noise.device().clone();
    let cfg = &dit.config.clone();

    // Pyramid downsample at chunk start (full → /4 spatial).
    let mut latents = pyramid_downsample(&initial_noise)?;

    // Distilled: track start_point_list[stage_idx] for DMD step's
    // `dmd_noisy_tensor` argument.
    let mut start_point_list: Vec<Tensor> = Vec::with_capacity(3);
    start_point_list.push(latents.clone());

    for (stage_idx, &num_steps) in PYRAMID_NUM_INFERENCE_STEPS.iter().enumerate() {
        let t_stage = Instant::now();
        // Compute mu from current pyramid sequence length.
        let dims = latents.shape().dims().to_vec();
        let (_b, _c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let pt = cfg.patch_size[0];
        let ph = cfg.patch_size[1];
        let pw = cfg.patch_size[2];
        let image_seq_len = (f * h * w) / (pt * ph * pw);
        let mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15);
        scheduler.set_timesteps(num_steps, stage_idx, Some(mu), false);
        eprintln!(
            "      stage {}: lat_F={} H={} W={} S_post_patch={} mu={:.4} steps={}",
            stage_idx, f, h, w, image_seq_len, mu, num_steps
        );

        // For stage > 0: nearest upsample + renoise correction.
        if stage_idx > 0 {
            latents = nearest_upsample_2x(&latents)?;
            let dims = latents.shape().dims().to_vec();
            let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
            // Renoise math (pipeline lines 928-945).
            let ori_sigma = 1.0 - scheduler.ori_start_sigmas[stage_idx];
            let gamma = scheduler.config.gamma;
            let alpha = 1.0
                / ((1.0 + 1.0 / gamma).sqrt() * (1.0 - ori_sigma) + ori_sigma);
            let beta = alpha * (1.0 - ori_sigma) / gamma.sqrt();
            // sample_block_noise — we use a chunk-and-stage-derived seed.
            let noise = sample_block_noise(
                b,
                c,
                f,
                h,
                w,
                (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]),
                gamma,
                seed
                    .wrapping_add(chunk_idx as u64 * 100)
                    .wrapping_add(stage_idx as u64),
                device.clone(),
            )?;
            let noise = noise.to_dtype(DType::BF16)?;
            // latents = alpha * latents + beta * noise
            let lat_a = latents.mul_scalar(alpha as f32)?;
            let nse_b = noise.mul_scalar(beta as f32)?;
            latents = lat_a.add(&nse_b)?;
            start_point_list.push(latents.clone());
        }

        // Per-step denoise: forward → DMD step.
        let timesteps_for_stage: Vec<f64> = scheduler.timesteps.clone();
        let dmd_sigmas_for_stage: Vec<f64> = scheduler.sigmas.clone();
        let dmd_timesteps_for_stage: Vec<f64> = scheduler.timesteps.clone();

        for (i, &t) in timesteps_for_stage.iter().enumerate() {
            let latent_input = latents.to_dtype(DType::BF16)?;
            let history_short_bf16 = history_short.to_dtype(DType::BF16)?;
            let history_mid_bf16 = history_mid.to_dtype(DType::BF16)?;
            let history_long_bf16 = history_long.to_dtype(DType::BF16)?;

            // Build a (B,) timestep tensor.
            let timestep_t = Tensor::from_vec(
                vec![t as f32; latents.shape().dims()[0]],
                Shape::from_dims(&[latents.shape().dims()[0]]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            let noise_pred = dit.forward_chunk(
                &latent_input,
                prompt_embeds,
                &timestep_t,
                Some(indices_hidden),
                Some(&history_short_bf16),
                Some(indices_short),
                Some(&history_mid_bf16),
                Some(indices_mid),
                Some(&history_long_bf16),
                Some(indices_long),
            )?;

            // DMD step. dmd_noisy_tensor = start_point_list[stage_idx]
            // (the latents at the START of this stage, before any denoise).
            latents = scheduler.step(
                &noise_pred,
                t,
                &latents,
                &start_point_list[stage_idx],
                &dmd_sigmas_for_stage,
                &dmd_timesteps_for_stage,
                &timesteps_for_stage,
                i,
            )?;
        }
        eprintln!("      stage {} done in {:.1}s", stage_idx, t_stage.elapsed().as_secs_f32());
    }

    Ok(latents)
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();
    let args = parse_args()?;
    let device = global_cuda_device();
    let t_total = Instant::now();

    println!("=== Helios-Distilled T2V (pure Rust) ===");
    println!("  prompt:    {}", args.prompt);
    if !args.negative.is_empty() {
        println!("  negative:  {}", args.negative);
    }
    println!("  size:      {}x{}", args.width, args.height);
    println!("  frames:    {}", args.num_frames);
    println!("  seed:      {}", args.seed);
    println!("  output:    {}", args.output.display());
    println!("  snapshot:  {}", args.snapshot.display());
    println!("  vae_path:  {}", args.vae_path.display());

    // -------------------------------------------------------------------
    // Stage 1: Tokenize + UMT5 encode + drop UMT5
    // -------------------------------------------------------------------
    println!("\n--- Stage 1: UMT5-XXL text encoding ---");
    let t0 = Instant::now();
    let tokenizer_path = args.snapshot.join("tokenizer/tokenizer.json");
    let pos_ids = tokenize(&tokenizer_path, &args.prompt)?;
    println!("  prompt tokens: {} ids", pos_ids.len());

    // Find UMT5 weights (single shard or sharded).
    let te_dir = args.snapshot.join("text_encoder");
    let umt5_path = locate_umt5_weights(&te_dir)?;
    println!("  UMT5 weights: {}", umt5_path.display());

    let mut umt5 = Umt5Encoder::load(&umt5_path, &device)
        .map_err(|e| anyhow!("Umt5Encoder::load: {e}"))?;
    let prompt_embeds = umt5.encode(&pos_ids)?;
    println!(
        "  prompt_embeds shape: {:?}",
        prompt_embeds.shape().dims()
    );
    drop(umt5);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    println!("  stage 1: {:.1}s (UMT5 dropped)", t0.elapsed().as_secs_f32());

    // -------------------------------------------------------------------
    // Stage 2: Load Helios DiT + per-chunk denoise loop
    // -------------------------------------------------------------------
    println!("\n--- Stage 2: Helios DiT (BlockOffloader) + per-chunk denoise ---");
    let t0 = Instant::now();
    let transformer_dir = args.snapshot.join("transformer");
    let mut dit = HeliosInferDit::load(&transformer_dir, device.clone())
        .with_context(|| format!("load HeliosInferDit from {:?}", transformer_dir))?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let scheduler_cfg = HeliosDMDConfig::distilled_default();
    let mut scheduler = HeliosDMDState::new(scheduler_cfg);

    // Latent geometry.
    let h_lat = args.height / VAE_SPATIAL_STRIDE;
    let w_lat = args.width / VAE_SPATIAL_STRIDE;
    let window_num_frames = (NUM_LATENT_FRAMES_PER_CHUNK - 1) * VAE_TEMPORAL_STRIDE + 1; // 33
    let num_latent_chunks = args.num_frames.div_ceil(window_num_frames);
    let num_history_latent_frames: usize = HISTORY_SIZES.iter().sum();
    println!(
        "  latent geom: {}x{} (px), {}x{} (latent), {} chunks of {} frames",
        args.width, args.height, w_lat, h_lat, num_latent_chunks, window_num_frames
    );

    // Build history indices (T2V mode → keep_first_frame=True).
    let (indices_hidden, indices_short, indices_mid, indices_long) =
        build_history_indices(true, &HISTORY_SIZES, NUM_LATENT_FRAMES_PER_CHUNK);
    println!(
        "  indices: hidden={:?} short={:?} mid={:?} long={:?}",
        indices_hidden, indices_short, indices_mid, indices_long
    );

    // history_latents buffer: (1, 16, num_history_latent_frames=19, H/8, W/8) F32 zeros.
    let history_buf_shape = Shape::from_dims(&[
        1,
        Z_DIM,
        num_history_latent_frames,
        h_lat,
        w_lat,
    ]);
    let mut history_latents = Tensor::zeros(history_buf_shape, device.clone())?
        .to_dtype(DType::F32)?;
    let mut total_generated_latent_frames: usize = 0;
    let mut image_latents: Option<Tensor> = None; // T2V: starts None, set to chunk0 first frame.
    let mut all_chunk_latents: Vec<Tensor> = Vec::with_capacity(num_latent_chunks);

    for k in 0..num_latent_chunks {
        let t_chunk = Instant::now();
        let is_first_chunk = k == 0;
        println!("\n  -- chunk {}/{} --", k + 1, num_latent_chunks);

        // Build history splits from history_buf.
        let buf_dims = history_latents.shape().dims().to_vec();
        let buf_t = buf_dims[2];
        let history_window =
            history_latents.narrow(2, buf_t - num_history_latent_frames, num_history_latent_frames)?;
        // Split along T into [long(16), mid(2), short_history(1)] per HISTORY_SIZES.
        let mut cursor = 0usize;
        let history_long = history_window.narrow(2, cursor, HISTORY_SIZES[0])?;
        cursor += HISTORY_SIZES[0];
        let history_mid = history_window.narrow(2, cursor, HISTORY_SIZES[1])?;
        cursor += HISTORY_SIZES[1];
        let history_1x = history_window.narrow(2, cursor, HISTORY_SIZES[2])?;

        // latents_prefix: zeros(1) for first chunk T2V, else image_latents (1 frame).
        // Match history_1x's dtype (F32, since history_latents buffer is F32).
        let latents_prefix = if image_latents.is_none() && is_first_chunk {
            let prefix_shape =
                Shape::from_dims(&[1, Z_DIM, 1, history_1x.shape().dims()[3], history_1x.shape().dims()[4]]);
            Tensor::zeros(prefix_shape, device.clone())?.to_dtype(history_1x.dtype())?
        } else {
            image_latents
                .as_ref()
                .unwrap()
                .clone()
                .to_dtype(history_1x.dtype())?
        };
        let history_short = Tensor::cat(&[&latents_prefix, &history_1x], 2)?;

        // Build initial noise and denoise.
        let chunk_seed = args.seed.wrapping_add(k as u64 * 1000);
        let initial_noise =
            make_chunk_noise(args.height, args.width, chunk_seed, device.clone())?;

        let chunk_latents = denoise_one_chunk(
            &mut dit,
            &mut scheduler,
            &prompt_embeds,
            initial_noise,
            &history_short,
            &history_mid,
            &history_long,
            &indices_hidden,
            &indices_short,
            &indices_mid,
            &indices_long,
            args.seed,
            k,
        )?;

        println!(
            "    chunk_latents shape: {:?} dtype={:?}",
            chunk_latents.shape().dims(),
            chunk_latents.dtype()
        );

        // T2V: capture image_latents from first chunk's first frame.
        if is_first_chunk && image_latents.is_none() {
            image_latents = Some(chunk_latents.narrow(2, 0, 1)?);
        }

        total_generated_latent_frames += chunk_latents.shape().dims()[2];
        // Append to history_latents buffer.
        let chunk_f32 = chunk_latents.to_dtype(DType::F32)?;
        history_latents = Tensor::cat(&[&history_latents, &chunk_f32], 2)?;
        all_chunk_latents.push(chunk_latents);
        println!(
            "    chunk done in {:.1}s (total generated: {} frames)",
            t_chunk.elapsed().as_secs_f32(),
            total_generated_latent_frames
        );
    }

    drop(dit);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    println!("\n  Stage 2 done in {:.1}s", t0.elapsed().as_secs_f32());

    // -------------------------------------------------------------------
    // Stage 3: Wan 2.1 VAE decode each chunk + concat + MP4 mux
    // -------------------------------------------------------------------
    println!("\n--- Stage 3: Wan 2.1 VAE decode + MP4 mux ---");
    let t0 = Instant::now();
    let vae_path_str = args.vae_path.to_str().ok_or_else(|| anyhow!("non-utf8 vae path"))?;
    let vae = Wan21VaeDecoder::load(vae_path_str, &device)
        .map_err(|e| anyhow!("Wan21VaeDecoder::load: {e}"))?;
    println!("  VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut all_frames_u8: Vec<u8> = Vec::new();
    let mut total_pixel_frames: usize = 0;
    for (i, chunk_lat) in all_chunk_latents.iter().enumerate() {
        let t_dec = Instant::now();
        // CRITICAL: Wan21VaeDecoder.decode() does denormalization internally.
        // We pass un-denormalized latents (raw output from the DiT denoise loop).
        let chunk_lat_bf16 = chunk_lat.to_dtype(DType::BF16)?;
        let chunk_video = vae
            .decode(&chunk_lat_bf16)
            .map_err(|e| anyhow!("vae.decode chunk {i}: {e}"))?;
        let dims = chunk_video.shape().dims().to_vec();
        let (_, _, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        println!(
            "    chunk {} decoded in {:.1}s → {} px frames @ {}x{}",
            i,
            t_dec.elapsed().as_secs_f32(),
            t,
            h,
            w
        );
        let flat = chunk_video.to_vec_f32().map_err(|e| anyhow!("to_vec_f32: {e}"))?;
        // Drop the BF16 chunk_video + chunk_lat_bf16 before the GPU pool grows.
        drop(chunk_video);
        drop(chunk_lat_bf16);
        let chunk_u8 = inference_flame::mux::video_tensor_to_rgb_u8(&flat, t, h, w);
        all_frames_u8.extend_from_slice(&chunk_u8);
        total_pixel_frames += t;
        // Flush pool between chunks to avoid fragmentation across multi-chunk runs.
        flame_core::cuda_alloc_pool::clear_pool_cache();
    }

    drop(vae);
    flame_core::cuda_alloc_pool::clear_pool_cache();

    // MP4 mux. Silent stereo audio (zeros) since helios is video-only.
    let fps: f32 = 24.0;
    let audio_sample_rate: u32 = 48000;
    let audio_samples = ((total_pixel_frames as f32 / fps) * audio_sample_rate as f32) as usize;
    let silent_audio = vec![0i16; audio_samples * 2]; // stereo

    println!(
        "\n  Muxing {} frames at {} fps → {}",
        total_pixel_frames,
        fps,
        args.output.display()
    );
    inference_flame::mux::write_mp4(
        &args.output,
        &all_frames_u8,
        total_pixel_frames,
        args.width,
        args.height,
        fps,
        &silent_audio,
        audio_sample_rate,
    )
    .map_err(|e| anyhow!("write_mp4: {e}"))?;

    println!("  Stage 3 done in {:.1}s", t0.elapsed().as_secs_f32());
    println!(
        "\n=== TOTAL: {:.1}s — saved {}",
        t_total.elapsed().as_secs_f32(),
        args.output.display()
    );

    Ok(())
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Find UMT5 weights — Helios ships a sharded checkpoint
/// (`model.safetensors.index.json` + `model-NNNNN-of-NNNNN.safetensors`).
/// `Umt5Encoder::load` takes a single safetensors path. For the smoke
/// path, we'd want a merger; for now, look for either a single
/// `model.safetensors` (small models) or use the index to merge.
fn locate_umt5_weights(te_dir: &Path) -> Result<PathBuf> {
    let single = te_dir.join("model.safetensors");
    if single.exists() {
        return Ok(single);
    }
    let index = te_dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Err(anyhow!(
            "neither {} nor {} found",
            single.display(),
            index.display()
        ));
    }
    // Sharded: Umt5Encoder::load takes a single file. We'd need to merge the
    // shards, but flame's serialization::load_file_filtered handles single
    // files only. As a workaround, use the snapshot symlink and merge via
    // `load_file` from each shard. Defer the actual merge to a helper.
    Err(anyhow!(
        "UMT5 sharded ({} shards needed); merger not yet implemented. \
         Workaround: dump shards to a single `model.safetensors` via:\n\
         `python3 -c 'from safetensors.torch import load_file, save_file; \
         import json, os; idx=json.load(open(\"{}\")); shards=set(idx[\"weight_map\"].values()); \
         w={{}}; \n  [w.update(load_file(os.path.join(\"{}\", s))) for s in shards]; \
         save_file(w, \"{}\")'`",
        index.display(),
        index.display(),
        te_dir.display(),
        single.display()
    ))
}
