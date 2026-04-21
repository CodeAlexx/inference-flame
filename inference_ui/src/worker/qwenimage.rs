//! Qwen-Image-2512 real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3}.rs` structurally — lazy load,
//! drop-to-fit VRAM dance, per-step Started/Progress/Done event emission.
//! Underlying pipeline mirrors `inference-flame/src/bin/qwenimage_gen.rs`
//! for DiT + denoise + VAE and `inference-flame/src/bin/qwenimage_encode.rs`
//! for the in-process Qwen2.5-VL-7B text encoder.
//!
//! ## Architecture
//!
//! Qwen-Image DiT — 60 transformer layers, 3-axis RoPE, split-QKV attention.
//! Loaded via `QwenImageDit::load(&shards, &device)` (BlockOffloader). The
//! 9-shard transformer lives under `.serenity/models/checkpoints/qwen-image-2512/`.
//!
//! ## Text encoder — in-process Qwen2.5-VL-7B
//!
//! The user's typed prompt and negative are tokenized and encoded through
//! `Qwen25VLEncoder` (4-shard BF16 load, ~14 GB). Mirrors
//! `qwenimage_encode.rs` exactly:
//!   1. Wrap prompt in the diffusers system-prompt template
//!   2. Tokenize with the bundled Qwen2.5-VL tokenizer, pad to `max_len`
//!   3. Encode through all 28 layers + final RMSNorm
//!   4. Drop the first 34 tokens (the system prompt) and trim trailing pads
//!
//! Encoder + tokenizer are loaded per-job and dropped before the DiT load
//! to keep us under 24 GB. Total peak during encoding: ~14 GB (encoder) +
//! a few hundred MB of embedding tensors. After drop: ~0. The reloading
//! cost per job is ~15s — acceptable vs. the alternative of OOM on 24 GB.
//!
//! ## CFG
//!
//! Real two-pass (cond + uncond) PLUS Qwen-specific norm rescale:
//!   `comb       = uncond + scale * (cond - uncond)`
//!   `out        = comb * (||cond||_2 / ||comb||_2)`  (per-token, dim=-1)
//!
//! Mirrors `qwenimage_gen.rs::norm_rescale_cfg` exactly. If the rescale
//! computation fails (sum_dim_keepdim or div errors), falls back to raw
//! `comb` — same behavior as the reference bin.
//!
//! ## Schedule
//!
//! `FlowMatchEulerDiscreteScheduler` with dynamic exponential shift +
//! terminal stretch. base_shift=0.5, max_shift=0.9, base_seq=256,
//! max_seq=8192, shift=1.0, shift_terminal=0.02. All inlined here (mirrors
//! the closure body in `qwenimage_gen.rs::main`); no shared helper exists.
//!
//! ## VAE
//!
//! 3D VAE with T=1 axis. Latents are unpacked back to 5D before decode and
//! the T=1 axis is squeezed out at the boundary so the worker's
//! `decoded_to_color_image` (4D NCHW) helper applies. Mirrors lines 461-484
//! in `qwenimage_gen.rs`.
//!
//! ## Latent
//!
//! Raw [1, 1, 16, H/8, W/8], packed [1, seq, 64] where seq = (H/16)(W/16).
//! Pack: diffusers `_pack_latents` —
//!   `[B, C, H/2, 2, W/2, 2] → [B, H/2, W/2, C, 2, 2] → [B, seq, C*4]`.
//! Done on CPU on the f32 noise vector (same as the bin).
//!
//! ## VRAM budget on a 24 GB card
//!
//! Peak moments (no overlap, encoder dropped before DiT loads):
//! - Qwen2.5-VL 7B encoder (BF16):      ~14 GB  (Stage 1 only)
//! - Qwen-Image DiT (BlockOffloader):   ~14-17 GB
//! - QwenImageVaeDecoder:               ~0.4 GB
//! - Activations (1024² latent):        ~2-3 GB
//!
//! The Stage 1 encoder is dropped before Stage 2 loads the DiT, so the
//! two ~14 GB heavyweights never coexist.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::qwen25vl_encoder::Qwen25VLEncoder;
use inference_flame::models::qwenimage_dit::QwenImageDit;
use inference_flame::vae::QwenImageVaeDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors qwenimage_gen.rs / qwenimage_encode.rs).
//
// AGENT-DEFAULT: hardcoded constants. Same rationale as flux.rs / klein.rs.
// ===========================================================================

/// Qwen2.5-VL-7B text encoder snapshot base. Contains the 4 text_encoder
/// shards and the bundled tokenizer.
const QWEN25VL_SNAP: &str = "/home/alex/.serenity/models/checkpoints/qwen-image-2512";

/// Qwen2.5-VL-7B text encoder — 4 sharded safetensors files (BF16).
const TEXT_ENCODER_SHARDS: &[&str] = &[
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder/model-00001-of-00004.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder/model-00002-of-00004.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder/model-00003-of-00004.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder/model-00004-of-00004.safetensors",
];

/// Qwen2.5-VL tokenizer — bundled with the text_encoder snapshot.
const TOKENIZER_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/tokenizer/tokenizer.json";

/// Qwen-Image DiT — 9 sharded safetensors files.
const DIT_SHARDS: &[&str] = &[
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
];

/// Qwen-Image VAE decoder.
const VAE_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors";

// VAE downscale: 2 ** len(temperal_downsample) = 2^3 = 8
const VAE_SCALE_FACTOR: usize = 8;
/// 2× DiT patch — packed seq = (H/8/2)(W/8/2).
const PATCH_SIZE: usize = 2;
const IN_CHANNELS: usize = 16;
const PACKED_CHANNELS: usize = 64; // 16 * 2 * 2

/// Default CFG. qwenimage_gen.rs uses 4.0.
const DEFAULT_CFG: f32 = 4.0;

/// Default step count. qwenimage_gen.rs uses 50.
const DEFAULT_STEPS: u32 = 50;

// Schedule constants from the diffusers pipeline (qwenimage_gen.rs).
const SCHED_BASE_SHIFT: f32 = 0.5;
const SCHED_MAX_SHIFT: f32 = 0.9;
const SCHED_BASE_SEQ: f32 = 256.0;
const SCHED_MAX_SEQ: f32 = 8192.0;
const SCHED_SHIFT_TERMINAL: f32 = 0.02;

// Prompt template constants — mirrors qwenimage_encode.rs and the diffusers
// `pipeline_qwenimage.py::PROMPT_TEMPLATE_ENCODE`.
const PROMPT_PREFIX: &str =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
     texture, quantity, text, spatial relationships of the objects and background:\
     <|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Number of leading tokens to drop (the system prompt). Matches Python
/// constant `PROMPT_TEMPLATE_ENCODE_START_IDX`.
const DROP_IDX: usize = 34;
/// Qwen2.5-VL pad token id.
const PAD_TOKEN: i32 = 151643;
/// Default tokenizer max length — the qwenimage_encode bin uses 1024 (before
/// system-prompt drop). Add DROP_IDX so post-drop usable length is 1024.
const TOKENIZER_MAX_LEN: usize = 1024 + DROP_IDX;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local Qwen-Image resources. Nothing kept resident — encoder, DiT,
/// and VAE are loaded per-job because each exceeds a conservative 10 GB
/// budget and they'd collide on a 24 GB card. The device handle IS kept
/// resident (CUDA context init is expensive).
pub struct QwenImageState {
    pub device: Arc<CudaDevice>,
}

impl QwenImageState {
    pub fn new() -> Result<Self, String> {
        let device =
            CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e:?}"))?;
        Ok(Self { device })
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

pub fn run(
    job: &GenerateJob,
    state: &mut QwenImageState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let steps: u32 = if job.steps == 0 { DEFAULT_STEPS } else { job.steps };
    let cfg_scale: f32 = if job.cfg <= 0.0 { DEFAULT_CFG } else { job.cfg };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: steps,
    });
    ctx.request_repaint();

    match run_inner(job, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending) {
        Ok(image) => {
            let _ = ev_tx.send(WorkerEvent::Done {
                id: job.id,
                image,
                prompt: job.prompt.clone(),
            });
            ctx.request_repaint();
        }
        Err(RunError::Cancelled) => {
            let _ = ev_tx.send(WorkerEvent::Failed {
                id: job.id,
                error: "cancelled".into(),
            });
            ctx.request_repaint();
        }
        Err(RunError::Other(msg)) => {
            log::warn!("Qwen-Image job {} failed: {msg}", job.id);
            let _ = ev_tx.send(WorkerEvent::Failed {
                id: job.id,
                error: msg,
            });
            ctx.request_repaint();
        }
    }
}

// ===========================================================================
// Internals
// ===========================================================================

enum RunError {
    Cancelled,
    Other(String),
}

impl<E: std::fmt::Debug> From<E> for RunError {
    fn from(e: E) -> Self {
        RunError::Other(format!("{e:?}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    job: &GenerateJob,
    steps: u32,
    cfg_scale: f32,
    state: &mut QwenImageState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap so we ALWAYS flush the pool. Same rationale as flux/klein:
    // the encoder is ~14 GB and the DiT is ~14-17 GB; an early Cancelled
    // return would otherwise leave pooled buffers racing the next job's load.
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(job, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending)
    })();
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_inner_body(
    job: &GenerateJob,
    steps: u32,
    cfg_scale: f32,
    state: &mut QwenImageState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. Tokenize + encode prompts through Qwen2.5-VL-7B --------
    // Encoder is ~14 GB; it's loaded, used twice (cond + uncond), then
    // dropped BEFORE the DiT load to stay under 24 GB. Mirrors
    // qwenimage_encode.rs exactly (template + tokenize + encode + trim).
    if !Path::new(TOKENIZER_PATH).exists() {
        return Err(RunError::Other(format!(
            "Qwen2.5-VL tokenizer not found at {TOKENIZER_PATH}"
        )));
    }
    for shard in TEXT_ENCODER_SHARDS {
        if !Path::new(shard).exists() {
            return Err(RunError::Other(format!(
                "Qwen2.5-VL encoder shard not found: {shard}"
            )));
        }
    }
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| RunError::Other(format!("tokenizer load from {TOKENIZER_PATH}: {e}")))?;

    log::info!("Qwen-Image: loading Qwen2.5-VL-7B text encoder ({} shards)", TEXT_ENCODER_SHARDS.len());
    let t_enc_load = Instant::now();
    let mut enc_weights: HashMap<String, Tensor> = HashMap::new();
    for shard in TEXT_ENCODER_SHARDS {
        // Filter out visual.* — we only need the language model (~7B) and
        // skipping the vision tower saves significant VRAM.
        let partial = flame_core::serialization::load_file_filtered(
            Path::new(shard),
            &device,
            |k| !k.starts_with("visual."),
        )
        .map_err(|e| RunError::Other(format!("encoder shard load {shard}: {e:?}")))?;
        enc_weights.extend(partial);
    }
    // Upcast F16/F32 to BF16 if any shard stored a different dtype.
    let keys: Vec<String> = enc_weights.keys().cloned().collect();
    for k in keys {
        let t = &enc_weights[&k];
        if t.dtype() != DType::BF16 {
            let bf = t
                .to_dtype(DType::BF16)
                .map_err(|e| RunError::Other(format!("upcast {k}: {e:?}")))?;
            enc_weights.insert(k, bf);
        }
    }
    let mut cfg = Qwen25VLEncoder::config_from_weights(&enc_weights)
        .map_err(|e| RunError::Other(format!("encoder config: {e:?}")))?;
    cfg.max_seq_len = TOKENIZER_MAX_LEN.max(cfg.max_seq_len);
    let encoder = Qwen25VLEncoder::new(enc_weights, cfg, device.clone());
    log::info!(
        "Qwen-Image: encoder ready in {:.1}s (snapshot {})",
        t_enc_load.elapsed().as_secs_f32(),
        QWEN25VL_SNAP,
    );

    drain_pending(ui_rx, pending)?;

    let t_encode = Instant::now();
    let cond = encode_prompt(&encoder, &tokenizer, &job.prompt)?;
    let uncond = encode_prompt(&encoder, &tokenizer, &job.negative)?;
    log::info!(
        "Qwen-Image: cond {:?}, uncond {:?} in {:.1}s",
        cond.shape().dims(),
        uncond.shape().dims(),
        t_encode.elapsed().as_secs_f32(),
    );

    // Drop encoder + tokenizer before loading the DiT. Encoder holds ~14 GB
    // of BF16 weights that MUST be released or the DiT load OOMs on a 24 GB
    // card.
    drop(encoder);
    drop(tokenizer);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Qwen-Image: encoder dropped, CUDA pool flushed (~14 GB freed)");

    drain_pending(ui_rx, pending)?;

    // -------- 2. Load Qwen-Image DiT (BlockOffloader) --------
    // Base-ComboBox override wins. None → fall back to the 9 hardcoded shards.
    // Single-file override (e.g. a .gguf) turns into a 1-element slice.
    let override_shard: Option<String> = job.path.clone();
    let shard_slice: Vec<&str> = match override_shard.as_deref() {
        Some(p) => vec![p],
        None => DIT_SHARDS.iter().copied().collect(),
    };
    for shard in &shard_slice {
        if !Path::new(shard).exists() {
            return Err(RunError::Other(format!(
                "Qwen-Image DiT shard not found: {shard}"
            )));
        }
    }
    log::info!(
        "Qwen-Image: loading DiT (BlockOffloader, {} shards)",
        shard_slice.len()
    );
    let t0 = Instant::now();
    // GGUF guard: QwenImageDit is BlockOffloader-only. Same reasoning as
    // FLUX/Chroma: BlockOffloader wants mmap'd safetensors, not a pre-loaded
    // HashMap. AGENT-DEFAULT: fail fast with a clear error.
    if shard_slice
        .iter()
        .any(|p| p.to_ascii_lowercase().ends_with(".gguf"))
    {
        return Err(RunError::Other(
            "Qwen-Image DiT GGUF not yet supported: QwenImageDit uses \
             BlockOffloader (streaming pinned-host + dual-GPU-slot) which \
             can't consume a pre-loaded HashMap. Use .safetensors for \
             Qwen-Image for now."
                .to_string(),
        ));
    }
    let mut dit = QwenImageDit::load(&shard_slice, &device)
        .map_err(|e| RunError::Other(format!("DiT load: {e:?}")))?;
    log::info!("Qwen-Image: DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    drain_pending(ui_rx, pending)?;

    // -------- 3. Latent geometry --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    let h_latent_full = height / VAE_SCALE_FACTOR;
    let w_latent_full = width / VAE_SCALE_FACTOR;
    let h_patched = h_latent_full / PATCH_SIZE;
    let w_patched = w_latent_full / PATCH_SIZE;
    let seq_len = h_patched * w_patched;
    if seq_len == 0 {
        return Err(RunError::Other(format!(
            "image too small: {width}x{height}"
        )));
    }
    log::info!(
        "Qwen-Image: latent raw [1, 1, {IN_CHANNELS}, {h_latent_full}, {w_latent_full}], \
         packed [1, {seq_len}, {PACKED_CHANNELS}]"
    );

    // -------- 4. Seeded noise + diffusers _pack_latents (CPU) --------
    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    // Box-Muller noise — must be generated as a flat vec so we can pack on CPU.
    // Mirrors qwenimage_gen.rs lines 233-252.
    let numel = IN_CHANNELS * h_latent_full * w_latent_full;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_u64);
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

    // diffusers _pack_latents: [B, C, H/2, 2, W/2, 2] -> [B, H/2, W/2, C, 2, 2]
    // -> [B, (H/2)(W/2), C*4]. We do it on CPU (mirrors the bin lines 256-277).
    let mut packed = vec![0.0f32; seq_len * PACKED_CHANNELS];
    for c in 0..IN_CHANNELS {
        for hp in 0..h_patched {
            for wp in 0..w_patched {
                for dh in 0..PATCH_SIZE {
                    for dw in 0..PATCH_SIZE {
                        let src_h = hp * PATCH_SIZE + dh;
                        let src_w = wp * PATCH_SIZE + dw;
                        let src_idx = c * h_latent_full * w_latent_full
                            + src_h * w_latent_full
                            + src_w;
                        let dst_seq = hp * w_patched + wp;
                        let dst_chan = c * PATCH_SIZE * PATCH_SIZE + dh * PATCH_SIZE + dw;
                        packed[dst_seq * PACKED_CHANNELS + dst_chan] = noise_data[src_idx];
                    }
                }
            }
        }
    }
    let mut x = Tensor::from_f32_to_bf16(
        packed,
        Shape::from_dims(&[1, seq_len, PACKED_CHANNELS]),
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("packed noise alloc: {e:?}")))?;

    // -------- 5. Sigma schedule (dynamic exponential shift + terminal stretch) --------
    let m = (SCHED_MAX_SHIFT - SCHED_BASE_SHIFT) / (SCHED_MAX_SEQ - SCHED_BASE_SEQ);
    let bb = SCHED_BASE_SHIFT - m * SCHED_BASE_SEQ;
    let mu = (seq_len as f32) * m + bb;
    let exp_mu = mu.exp();

    // 1. Linear sigmas in descending order
    let mut sigmas: Vec<f32> = (0..steps as usize)
        .map(|i| {
            let t = i as f32 / (steps as usize - 1) as f32;
            1.0 - t * (1.0 - 1.0 / steps as f32)
        })
        .collect();
    // 2. Exponential time shift (shift=1.0)
    for s in sigmas.iter_mut() {
        let denom = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / denom;
    }
    // 3. Stretch to terminal
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    if one_minus_last.abs() > 1e-12 {
        let scale = one_minus_last / (1.0 - SCHED_SHIFT_TERMINAL);
        for s in sigmas.iter_mut() {
            let o = 1.0 - *s;
            *s = 1.0 - o / scale;
        }
    }
    // 4. Append terminal sigma = 0
    sigmas.push(0.0);
    log::info!(
        "Qwen-Image: schedule mu={:.3}, sigmas[0]={:.4}, sigmas[-2]={:.4}",
        mu,
        sigmas[0],
        sigmas[steps as usize - 1],
    );

    drain_pending(ui_rx, pending)?;

    // -------- 6. Real two-pass CFG denoise with norm rescale --------
    let frame = 1usize;
    let t_denoise = Instant::now();
    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;

        // Scoped block: temporaries drop before next iteration's allocs.
        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![sigma_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?
            .to_dtype(DType::BF16)
            .map_err(|e| RunError::Other(format!("t_vec -> bf16: {e:?}")))?;

            let cond_pred = dit
                .forward(&x, &cond, &t_vec, (frame, h_patched, w_patched))
                .map_err(|e| RunError::Other(format!("cond fwd step {step}: {e:?}")))?;
            let uncond_pred = dit
                .forward(&x, &uncond, &t_vec, (frame, h_patched, w_patched))
                .map_err(|e| RunError::Other(format!("uncond fwd step {step}: {e:?}")))?;

            // True CFG combine
            let diff = cond_pred
                .sub(&uncond_pred)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            let comb = uncond_pred
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // Qwen-specific norm rescale — fall back to raw comb on failure.
            let noise_pred = match norm_rescale_cfg(&cond_pred, &comb) {
                Ok(t) => t,
                Err(e) => {
                    log::warn!(
                        "Qwen-Image: norm rescale failed at step {step} ({e}); using raw comb"
                    );
                    comb
                }
            };

            // Euler: x_next = x + dt * v
            let step_tensor = noise_pred
                .mul_scalar(dt)
                .map_err(|e| RunError::Other(format!("euler mul: {e:?}")))?;
            x.add(&step_tensor)
                .map_err(|e| RunError::Other(format!("euler add: {e:?}")))?
        };
        x = next_x;

        let step1 = (step + 1) as u32;
        let elapsed = t_denoise.elapsed().as_secs_f32();
        let per_step = elapsed / step1 as f32;
        let eta_secs = ((steps - step1) as f32 * per_step).round().max(0.0) as u32;
        let _ = ev_tx.send(WorkerEvent::Progress {
            id: job.id,
            step: step1,
            total: steps,
            eta_secs,
        });
        ctx.request_repaint();
    }

    // -------- 7. Drop DiT + embeddings before VAE --------
    drop(dit);
    drop(cond);
    drop(uncond);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Qwen-Image: DiT + embeddings dropped before VAE");

    // -------- 8. Unpack + VAE decode (3D VAE, T=1) --------
    // Mirrors qwenimage_gen.rs lines 466-484. Unpack: [1, seq, 64] →
    // [1, 16, 1, H/8, W/8] via diffusers reshape/permute, then decode →
    // [1, 3, 1, H, W]. Squeeze T=1 axis to land at [1, 3, H, W] for the
    // worker's standard color-image converter.
    if !Path::new(VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "Qwen-Image VAE not found at {VAE_PATH}"
        )));
    }
    let h_lat = 2 * (height / (VAE_SCALE_FACTOR * 2));
    let w_lat = 2 * (width / (VAE_SCALE_FACTOR * 2));
    let unpacked = x
        .reshape(&[1, h_lat / 2, w_lat / 2, PACKED_CHANNELS / 4, 2, 2])
        .map_err(|e| RunError::Other(format!("unpack reshape1: {e:?}")))?
        .permute(&[0, 3, 1, 4, 2, 5])
        .map_err(|e| RunError::Other(format!("unpack permute: {e:?}")))?
        .reshape(&[1, PACKED_CHANNELS / 4, 1, h_lat, w_lat])
        .map_err(|e| RunError::Other(format!("unpack reshape2: {e:?}")))?;
    drop(x);

    let t_vae = Instant::now();
    let decoder = QwenImageVaeDecoder::from_safetensors(VAE_PATH, &device)
        .map_err(|e| RunError::Other(format!("VAE load: {e:?}")))?;
    let rgb_5d = decoder
        .decode(&unpacked)
        .map_err(|e| RunError::Other(format!("VAE decode: {e:?}")))?;
    drop(unpacked);
    drop(decoder);
    log::info!("Qwen-Image: VAE decoded in {:.1}s", t_vae.elapsed().as_secs_f32());

    // [1, 3, 1, H, W] → [1, 3, H, W]
    let rgb = rgb_5d
        .narrow(2, 0, 1)
        .map_err(|e| RunError::Other(format!("VAE narrow T: {e:?}")))?
        .squeeze(Some(2))
        .map_err(|e| RunError::Other(format!("VAE squeeze T: {e:?}")))?;
    drop(rgb_5d);

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Prompt encoding helpers — mirrors qwenimage_encode.rs::{tokenize_prompt,
// encode_and_trim} exactly.
// ---------------------------------------------------------------------------

/// Wrap prompt in the diffusers system-prompt template, tokenize, pad with
/// PAD_TOKEN up to TOKENIZER_MAX_LEN, truncate if over. Returns raw i32 ids.
fn tokenize_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<Vec<i32>, RunError> {
    let wrapped = format!("{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}");
    let enc = tokenizer
        .encode(wrapped, false)
        .map_err(|e| RunError::Other(format!("tokenize: {e}")))?;
    let ids = enc.get_ids();
    let mut out: Vec<i32> = ids.iter().take(TOKENIZER_MAX_LEN).map(|&i| i as i32).collect();
    while out.len() < TOKENIZER_MAX_LEN {
        out.push(PAD_TOKEN);
    }
    Ok(out)
}

/// Tokenize + encode + trim system-prompt + trim trailing pads.
/// Returns the `[1, kept_len, 3584]` BF16 hidden state the DiT expects as
/// `encoder_hidden_states`.
fn encode_prompt(
    encoder: &Qwen25VLEncoder,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<Tensor, RunError> {
    let token_ids = tokenize_prompt(tokenizer, prompt)?;
    let real_len = token_ids
        .iter()
        .position(|&id| id == PAD_TOKEN)
        .unwrap_or(token_ids.len());
    if real_len <= DROP_IDX {
        return Err(RunError::Other(format!(
            "prompt produced only {real_len} tokens; expected > {DROP_IDX} after Qwen template"
        )));
    }

    let hidden = encoder
        .encode(&token_ids)
        .map_err(|e| RunError::Other(format!("encode: {e:?}")))?; // [1, seq_len, 3584]

    let kept_len = real_len - DROP_IDX;
    hidden
        .narrow(1, DROP_IDX, kept_len)
        .map_err(|e| RunError::Other(format!("narrow system prompt: {e:?}")))
}

// ---------------------------------------------------------------------------
// Qwen-specific norm rescale (mirrors qwenimage_gen.rs::norm_rescale_cfg)
// ---------------------------------------------------------------------------

fn norm_rescale_cfg(cond: &Tensor, comb: &Tensor) -> Result<Tensor, String> {
    let cond_sq = cond.mul(cond).map_err(|e| format!("cond_sq: {e:?}"))?;
    let comb_sq = comb.mul(comb).map_err(|e| format!("comb_sq: {e:?}"))?;
    let cond_sum = cond_sq
        .sum_dim_keepdim(2)
        .map_err(|e| format!("cond_sum: {e:?}"))?;
    let comb_sum = comb_sq
        .sum_dim_keepdim(2)
        .map_err(|e| format!("comb_sum: {e:?}"))?;
    let cond_norm = cond_sum.sqrt().map_err(|e| format!("cond_norm: {e:?}"))?;
    let comb_norm = comb_sum.sqrt().map_err(|e| format!("comb_norm: {e:?}"))?;
    let ratio = cond_norm
        .div(&comb_norm)
        .map_err(|e| format!("ratio: {e:?}"))?;
    comb.mul(&ratio).map_err(|e| format!("rescale mul: {e:?}"))
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux/klein/sd3)
// ---------------------------------------------------------------------------

fn drain_pending(
    ui_rx: &Receiver<UiMsg>,
    pending: &mut VecDeque<UiMsg>,
) -> Result<(), RunError> {
    while let Ok(msg) = ui_rx.try_recv() {
        match msg {
            UiMsg::Cancel => return Err(RunError::Cancelled),
            UiMsg::Shutdown => {
                pending.push_back(UiMsg::Shutdown);
                return Err(RunError::Cancelled);
            }
            other => pending.push_back(other),
        }
    }
    Ok(())
}

fn decoded_to_color_image(rgb: &Tensor) -> Result<ColorImage, flame_core::Error> {
    let dims = rgb.shape().dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "VAE decode shape {dims:?} != [1, 3, H, W]"
        )));
    }
    let h = dims[2];
    let w = dims[3];
    let f32_t = rgb.to_dtype(DType::F32)?;
    let host = f32_t.to_vec()?;

    let mut pixels = Vec::with_capacity(h * w);
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = ((host[i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            let g = ((host[plane + i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            let b = ((host[2 * plane + i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    Ok(ColorImage {
        size: [w, h],
        pixels,
    })
}
