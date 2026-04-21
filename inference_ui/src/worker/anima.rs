//! Anima (Cosmos Predict2) real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3,qwenimage,ernie}.rs` structurally —
//! lazy load, drop-to-fit VRAM dance, per-step Started/Progress/Done event
//! emission. Underlying pipeline mirrors
//! `inference-flame/src/bin/anima_infer.rs` for DiT + denoise + VAE.
//!
//! ## Architecture
//!
//! Cosmos Predict2 DiT — 28 blocks (`AnimaConfig::default().num_blocks`),
//! hidden=2048, 3D RoPE, patchify via a linear patch_embed. Model is ~3.9 GB
//! and loaded entirely to GPU via `Anima::new_all_on_gpu(...)`. No block
//! offloading.
//!
//! ## Text encoder — in-process Qwen3-0.6B + Anima LLM adapter
//!
//! Mirrors `cache_anima_embeddings.py` exactly. Two tokenizers, two-stage
//! encode:
//!   1. Tokenize prompt with Qwen3-0.6B tokenizer, pad to `MAX_SEQ_LEN`
//!   2. Tokenize same prompt with the T5-old (sentencepiece) tokenizer, pad
//!      to `MAX_SEQ_LEN`
//!   3. Run Qwen3-0.6B encoder on the Qwen3 ids → `[1, S, 1024]` hidden
//!   4. Zero out hidden rows at Qwen3 padding positions
//!   5. Apply final `model.norm` (Qwen3Encoder returns raw layer output)
//!   6. Call `Anima::encode_context(t5_ids, qwen3_hidden)` to run the
//!      cross-attention adapter → `[1, S, 1024]` context
//!   7. Zero out context rows at T5 padding positions
//!
//! Anima DiT must be loaded BEFORE encoding because `encode_context` is a
//! method on `Anima` (the adapter weights live in the DiT checkpoint under
//! `net.llm_adapter.*`). Both encoder + DiT fit comfortably (~5 GB total)
//! on a 24 GB card. Qwen3 encoder + tokenizers are dropped after encoding;
//! Anima DiT stays resident through denoise.
//!
//! ## CFG
//!
//! Real two-pass (cond + uncond). Default cfg=4.5 matches the HF
//! recommendation and the reference bin `anima_infer.rs`. The Anima DiT
//! forward is fast enough (~3.9 GB resident) that the extra cond/uncond
//! pair doesn't threaten VRAM.
//!
//! ## Schedule
//!
//! Linear rectified flow: sigmas in [1, 0], NO shift at inference. Mirrors
//! `anima_infer.rs:90-92`. Timestep fed to the model is raw sigma (NOT
//! multiplied by 1000 — this is different from SD3 / ERNIE).
//!
//! ## VAE
//!
//! Qwen-Image VAE (3D with T=1). Anima ships with `qwen_image_vae.safetensors`
//! under `anima/split_files/vae/`. Latent [1, T=1, H, W, 16]; we permute to
//! [1, 16, T, H, W] before VAE decode.
//!
//! AGENT-DEFAULT: the reference bin `anima_infer.rs` doesn't actually run
//! VAE decode in Rust — it saves the latent for Python decode. We call the
//! Rust `QwenImageVaeDecoder` directly here (same decoder used by the
//! qwenimage worker) and squeeze the T=1 axis before conversion. If this
//! produces wrong colors it's a parity issue with the anima shipping VAE.
//!
//! ### Known caveat: VAE color parity unverified
//!
//! `anima_infer.rs` saves the latent and instructs the user to run
//! `decode_anima_latent.py`. This worker, by contrast, calls
//! `QwenImageVaeDecoder` directly — bypassing the Python decode reference.
//! Shape and permute pipeline is correct (audited against the bin), but
//! whether the QwenImage VAE checkpoint is the right calibration for Anima
//! latents is **unverified by code review**. If the first real run
//! produces wrong colors, this is the first place to look.
//!
//! ## Latent
//!
//! [1, T=1, H/8, W/8, 16] — 5D with the channel axis LAST (different from
//! the other workers). Permuted to [1, 16, T, H, W] for the VAE.
//!
//! ## VRAM budget on a 24 GB card
//!
//! - Qwen3-0.6B encoder (BF16):           ~1.2 GB  (Stage 1 only)
//! - Anima DiT (all resident, BF16):      ~3.9 GB
//! - Qwen-Image VAE decoder:              ~0.4 GB
//! - Activations (1024² latent):          ~2-3 GB
//!
//! Comfortable. Encoder + DiT briefly coexist (~5 GB) during text encoding.
//! Encoder is dropped after both prompts encode; DiT stays resident through
//! denoise; DiT dropped before VAE decode for the usual cuDNN-workspace
//! safety margin.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::anima::{load_all_weights, Anima};
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::vae::QwenImageVaeDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors anima_infer.rs / cache_anima_embeddings.py).
// ===========================================================================

/// Anima DiT — single consolidated safetensors file.
const MODEL_PATH: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors";

/// Qwen-Image VAE (shipped with Anima).
const VAE_PATH: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors";

/// Qwen3-0.6B base text encoder (single safetensors, ~1.2 GB BF16).
const QWEN3_ENCODER_PATH: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors";

/// Qwen3 tokenizer.json. We try the HF cache snapshot first (permanent)
/// and fall back to the anima-trainer config (matches what generated the
/// cached embeddings). Both produce identical token ids for the Qwen3-0.6B
/// vocabulary.
///
/// AGENT-DEFAULT: hardcoded preference list. The HF cache path comes from
/// the auto-downloaded `Qwen/Qwen3-0.6B` snapshot; `/tmp/anima-trainer/` is
/// where the user keeps the trainer checkout (less permanent — can be
/// wiped on reboot).
const QWEN3_TOKENIZER_PATHS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json",
    "/tmp/anima-trainer/configs/qwen3_06b/tokenizer.json",
];

/// T5-old tokenizer.json — bundled with the anima trainer's t5_old config.
/// AGENT-DEFAULT: only known-correct location on disk. Anima's LLM adapter
/// was trained against this exact tokenizer's vocabulary (T5 v1_1 style
/// with sentencepiece); using a different T5 tokenizer would mis-index the
/// adapter's `embed.weight` lookup.
const T5_TOKENIZER_PATH: &str = "/tmp/anima-trainer/configs/t5_old/tokenizer.json";

/// Sequence length for both Qwen3 and T5 tokenization. Mirrors
/// cache_anima_embeddings.py::MAX_SEQ_LEN.
const MAX_SEQ_LEN: usize = 256;

/// Qwen3 pad token id (`<|endoftext|>` = 151643). Same as zimage worker.
/// `Qwen3Encoder` uses this for pad-row detection inside the causal mask.
const QWEN3_PAD_ID: i32 = 151643;

/// T5 pad token id (`<pad>` = 0).
const T5_PAD_ID: i32 = 0;

/// Default CFG. anima_infer.rs uses 4.5 (HF recommendation).
const DEFAULT_CFG: f32 = 4.5;

/// Default step count. anima_infer.rs uses 30.
const DEFAULT_STEPS: u32 = 30;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local Anima resources. Nothing kept resident across jobs — the
/// DiT load is fast enough (~few seconds on warm SSD for 3.9 GB) and
/// keeping it resident would interfere with concurrent Klein/Flux jobs.
pub struct AnimaState {
    pub device: Arc<CudaDevice>,
}

impl AnimaState {
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
    state: &mut AnimaState,
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
            log::warn!("Anima job {} failed: {msg}", job.id);
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
    state: &mut AnimaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap — same rationale as flux/klein. Anima DiT is smaller
    // (3.9 GB) so OOM risk from skipped flush is lower, but the pattern is
    // identical to keep all workers consistent.
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
    state: &mut AnimaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. Load Anima DiT (all weights on GPU) --------
    // Loaded BEFORE encoding because Anima::encode_context is a method on
    // Anima — the adapter weights live in the DiT checkpoint under
    // `net.llm_adapter.*`. ~3.9 GB, comfortably fits alongside the
    // Qwen3-0.6B encoder.
    //
    // Base-ComboBox override wins; `None` falls back to hardcoded MODEL_PATH.
    let model_path: &str = job.path.as_deref().unwrap_or(MODEL_PATH);
    if !Path::new(model_path).exists() {
        return Err(RunError::Other(format!(
            "Anima model not found at {model_path}"
        )));
    }
    log::info!("Anima: loading DiT from {model_path}");
    let t0 = Instant::now();
    let all_weights = if model_path.to_ascii_lowercase().ends_with(".gguf") {
        // GGUF path: loader dequants + strips standard top-level prefixes.
        // Anima accepts the same key set as the safetensors loader.
        log::info!("Anima: loading GGUF from {model_path}");
        inference_flame::gguf::load_file_gguf(Path::new(model_path), device.clone())
            .map_err(|e| RunError::Other(format!("Anima GGUF load: {e:?}")))?
    } else {
        load_all_weights(model_path, &device)
            .map_err(|e| RunError::Other(format!("weight load: {e:?}")))?
    };
    let mut model = Anima::new_all_on_gpu(model_path.to_string(), all_weights, device.clone());
    log::info!("Anima: DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    drain_pending(ui_rx, pending)?;

    // -------- 2. Encode prompts (Qwen3 + LLM adapter) --------
    let (context_cond, context_uncond) =
        encode_prompts(&model, &device, &job.prompt, &job.negative)?;
    log::info!(
        "Anima: context_cond {:?}, context_uncond {:?}",
        context_cond.shape().dims(),
        context_uncond.shape().dims(),
    );

    drain_pending(ui_rx, pending)?;

    // -------- 3. Latent geometry + seeded noise --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    let latent_h = height / 8;
    let latent_w = width / 8;
    if latent_h == 0 || latent_w == 0 {
        return Err(RunError::Other(format!(
            "image too small: {width}x{height}"
        )));
    }
    let t_frames = 1usize;
    log::info!(
        "Anima: latent [1, {t_frames}, {latent_h}, {latent_w}, 16] (8× downscale of {width}x{height})"
    );

    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    // AGENT-DEFAULT: randn_seeded (Box-Muller per docstring). The bin
    // hand-rolls Box-Muller; same statistical distribution, different
    // byte-exact bits.
    let mut x = Tensor::randn_seeded(
        Shape::from_dims(&[1, t_frames, latent_h, latent_w, 16]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?;

    // -------- 4. Schedule — linear, NO shift at inference --------
    let sigmas: Vec<f32> = (0..=steps as usize)
        .map(|i| 1.0 - i as f32 / steps as f32)
        .collect();

    drain_pending(ui_rx, pending)?;

    // -------- 5. Denoise loop, real two-pass CFG --------
    let t_denoise = Instant::now();
    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma;

        // Timestep is raw sigma [0, 1] — NOT multiplied by 1000 (differs
        // from SD3 / ERNIE). Mirrors anima_infer.rs:110.
        let next_x = {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![sigma],
                Shape::from_dims(&[1]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?;

            let pred_cond = model
                .forward_with_context(&x, &t_vec, &context_cond)
                .map_err(|e| RunError::Other(format!("cond fwd step {step}: {e:?}")))?;
            let pred_uncond = model
                .forward_with_context(&x, &t_vec, &context_uncond)
                .map_err(|e| RunError::Other(format!("uncond fwd step {step}: {e:?}")))?;

            let diff = pred_cond
                .sub(&pred_uncond)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            let pred = pred_uncond
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // Euler: x = x + dt * pred  (rectified flow velocity)
            let step_tensor = pred
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

    // -------- 6. Drop DiT before VAE decode --------
    drop(model);
    drop(context_cond);
    drop(context_uncond);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Anima: DiT + context dropped before VAE");

    // -------- 7. Permute latent [B, T, H, W, 16] -> [B, 16, T, H, W] --------
    // Mirrors anima_infer.rs:152.
    let latent = x
        .permute(&[0, 4, 1, 2, 3])
        .map_err(|e| RunError::Other(format!("latent permute: {e:?}")))?;
    drop(x);
    log::info!("Anima: VAE input latent {:?}", latent.shape().dims());

    // -------- 8. VAE decode (Qwen-Image VAE, then squeeze T=1) --------
    if !Path::new(VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "Anima VAE not found at {VAE_PATH}"
        )));
    }
    let t_vae = Instant::now();
    let decoder = QwenImageVaeDecoder::from_safetensors(VAE_PATH, &device)
        .map_err(|e| RunError::Other(format!("VAE load: {e:?}")))?;
    let rgb_5d = decoder
        .decode(&latent)
        .map_err(|e| RunError::Other(format!("VAE decode: {e:?}")))?;
    drop(latent);
    drop(decoder);
    log::info!("Anima: VAE decoded in {:.1}s", t_vae.elapsed().as_secs_f32());

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
// Prompt encoding — Qwen3-0.6B + Anima LLM adapter, mirrors
// cache_anima_embeddings.py exactly.
// ---------------------------------------------------------------------------

/// Tokenize, encode through Qwen3-0.6B, run Anima's LLM adapter, return
/// `(context_cond, context_uncond)` — both `[1, MAX_SEQ_LEN, 1024]` BF16.
///
/// Loads the Qwen3 encoder + both tokenizers once, runs both prompts, then
/// drops everything except the final context tensors.
fn encode_prompts(
    model: &Anima,
    device: &Arc<CudaDevice>,
    prompt: &str,
    negative: &str,
) -> Result<(Tensor, Tensor), RunError> {
    // Load tokenizers — Qwen3 has a fallback chain, T5 is single-source.
    let qwen3_tok_path = QWEN3_TOKENIZER_PATHS
        .iter()
        .copied()
        .find(|p| Path::new(p).exists())
        .ok_or_else(|| {
            RunError::Other(format!(
                "Qwen3-0.6B tokenizer not found. Checked:\n  - {}",
                QWEN3_TOKENIZER_PATHS.join("\n  - ")
            ))
        })?;
    if !Path::new(T5_TOKENIZER_PATH).exists() {
        return Err(RunError::Other(format!(
            "T5-old tokenizer not found at {T5_TOKENIZER_PATH}"
        )));
    }
    let qwen3_tk = tokenizers::Tokenizer::from_file(qwen3_tok_path)
        .map_err(|e| RunError::Other(format!("qwen3 tokenizer load from {qwen3_tok_path}: {e}")))?;
    let t5_tk = tokenizers::Tokenizer::from_file(T5_TOKENIZER_PATH)
        .map_err(|e| RunError::Other(format!("t5 tokenizer load: {e}")))?;

    // Load Qwen3-0.6B encoder
    if !Path::new(QWEN3_ENCODER_PATH).exists() {
        return Err(RunError::Other(format!(
            "Qwen3-0.6B encoder weights not found at {QWEN3_ENCODER_PATH}"
        )));
    }
    log::info!("Anima: loading Qwen3-0.6B text encoder from {QWEN3_ENCODER_PATH}");
    let t0 = Instant::now();
    let mut weights = flame_core::serialization::load_file(Path::new(QWEN3_ENCODER_PATH), device)
        .map_err(|e| RunError::Other(format!("qwen3 encoder load: {e:?}")))?;
    // Upcast non-BF16 shards if any.
    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in keys {
        let t = &weights[&k];
        if t.dtype() != DType::BF16 {
            let bf = t
                .to_dtype(DType::BF16)
                .map_err(|e| RunError::Other(format!("upcast {k}: {e:?}")))?;
            weights.insert(k, bf);
        }
    }

    // Auto-detect config (vocab=151936, hidden=1024, layers=28, heads=16/8).
    // Override extract_layers to grab ONLY the final layer — that's what
    // HF's `last_hidden_state` represents (modulo the final RMSNorm we
    // apply manually below; Qwen3Encoder::encode skips it).
    let mut cfg = Qwen3Encoder::config_from_weights(&weights)
        .map_err(|e| RunError::Other(format!("qwen3 config: {e:?}")))?;
    let last_layer = cfg.num_layers.checked_sub(1).ok_or_else(|| {
        RunError::Other(format!(
            "Qwen3 num_layers={} — expected >=1",
            cfg.num_layers
        ))
    })?;
    cfg.extract_layers = vec![last_layer];

    // Final-norm weight is needed AFTER encoder.encode() to produce the
    // post-norm `last_hidden_state` HF returns. Qwen3Encoder doesn't apply
    // it inside encode(), so we extract it here before the encoder takes
    // ownership of the weights map.
    let final_norm_w = weights
        .get("model.norm.weight")
        .ok_or_else(|| RunError::Other("missing model.norm.weight in Qwen3 encoder".into()))?
        .clone();

    let encoder = Qwen3Encoder::new(weights, cfg, device.clone());
    log::info!(
        "Anima: Qwen3-0.6B encoder ready in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // Encode both prompts
    let context_cond =
        encode_one_prompt(model, &encoder, &final_norm_w, &qwen3_tk, &t5_tk, device, prompt)?;
    let context_uncond =
        encode_one_prompt(model, &encoder, &final_norm_w, &qwen3_tk, &t5_tk, device, negative)?;

    // Drop encoder + tokenizers + final_norm_w before returning.
    drop(encoder);
    drop(final_norm_w);
    drop(qwen3_tk);
    drop(t5_tk);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Anima: Qwen3 encoder dropped, CUDA pool flushed (~1.2 GB freed)");

    Ok((context_cond, context_uncond))
}

/// Tokenize → encode → final norm → adapter → mask. Returns `[1, S, 1024]`.
fn encode_one_prompt(
    model: &Anima,
    encoder: &Qwen3Encoder,
    final_norm_w: &Tensor,
    qwen3_tk: &tokenizers::Tokenizer,
    t5_tk: &tokenizers::Tokenizer,
    device: &Arc<CudaDevice>,
    prompt: &str,
) -> Result<Tensor, RunError> {
    // 1. Tokenize with Qwen3 tokenizer, pad to MAX_SEQ_LEN.
    let qwen3_ids = pad_tokenize(qwen3_tk, prompt, MAX_SEQ_LEN, QWEN3_PAD_ID)?;
    // 2. Tokenize with T5 tokenizer, pad to MAX_SEQ_LEN.
    let t5_ids = pad_tokenize(t5_tk, prompt, MAX_SEQ_LEN, T5_PAD_ID)?;

    // 3. Build attention masks (1 for real tokens, 0 for pad). We use these
    // to zero out padding positions in both the Qwen3 hidden state and the
    // post-adapter context, matching cache_anima_embeddings.py.
    let qwen3_mask: Vec<f32> = qwen3_ids
        .iter()
        .map(|&id| if id == QWEN3_PAD_ID { 0.0 } else { 1.0 })
        .collect();
    let t5_mask: Vec<f32> = t5_ids
        .iter()
        .map(|&id| if id == T5_PAD_ID { 0.0 } else { 1.0 })
        .collect();

    // 4. Encode through Qwen3 → [1, S, 1024] (raw last-layer hidden, no
    // final norm).
    let raw_hidden = encoder
        .encode(&qwen3_ids)
        .map_err(|e| RunError::Other(format!("qwen3 encode: {e:?}")))?;

    // 5. Apply final RMSNorm (model.norm). HF's last_hidden_state is
    // post-norm; the cached embeddings were generated with this applied,
    // so the adapter expects post-norm input.
    let normed_hidden = rms_norm_last_dim(&raw_hidden, final_norm_w, 1e-6)?;

    // 6. Zero out hidden rows at Qwen3 padding positions (matches Python:
    // `nd_encoded_text[~nd_attn_mask.bool()] = 0`).
    let qwen3_hidden = mask_seq_dim(&normed_hidden, &qwen3_mask, device)?;

    // 7. Build T5 token-id tensor for the adapter's embed lookup. Anima's
    // embedding_lookup() reads ids as f32 (representing integers).
    let t5_id_f32: Vec<f32> = t5_ids.iter().map(|&i| i as f32).collect();
    let t5_id_tensor = Tensor::from_vec(
        t5_id_f32,
        Shape::from_dims(&[1, MAX_SEQ_LEN]),
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("t5 id tensor: {e:?}")))?;

    // 8. Run the adapter to produce the final context.
    let context = model
        .encode_context(&t5_id_tensor, &qwen3_hidden)
        .map_err(|e| RunError::Other(format!("encode_context: {e:?}")))?;

    // 9. Zero out context at T5 padding positions (matches Python:
    // `context[~t5_attn_mask.bool()] = 0`).
    let masked = mask_seq_dim(&context, &t5_mask, device)?;
    Ok(masked)
}

/// Tokenize + truncate to `max_len` + right-pad with `pad_id`.
fn pad_tokenize(
    tk: &tokenizers::Tokenizer,
    text: &str,
    max_len: usize,
    pad_id: i32,
) -> Result<Vec<i32>, RunError> {
    let enc = tk
        .encode(text, true)
        .map_err(|e| RunError::Other(format!("tokenize: {e}")))?;
    let mut out: Vec<i32> = enc.get_ids().iter().take(max_len).map(|&i| i as i32).collect();
    while out.len() < max_len {
        out.push(pad_id);
    }
    Ok(out)
}

/// Apply RMSNorm along the last (hidden) dim using flame_core's BF16 kernel.
/// Mirrors what Qwen3Encoder uses internally for `model.norm`.
fn rms_norm_last_dim(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, RunError> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x
        .reshape(&[batch, hidden])
        .map_err(|e| RunError::Other(format!("rms reshape in: {e:?}")))?;
    let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)
        .map_err(|e| RunError::Other(format!("rms_norm_bf16: {e:?}")))?;
    out_2d
        .reshape(&dims)
        .map_err(|e| RunError::Other(format!("rms reshape out: {e:?}")))
}

/// Multiply each row in the seq dim of `x` (`[1, S, D]`) by the per-position
/// scalar in `mask` (`[S]`). Used to zero out padding positions.
fn mask_seq_dim(x: &Tensor, mask: &[f32], device: &Arc<CudaDevice>) -> Result<Tensor, RunError> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(RunError::Other(format!(
            "mask_seq_dim expected [B, S, D], got {dims:?}"
        )));
    }
    let s = dims[1];
    if mask.len() != s {
        return Err(RunError::Other(format!(
            "mask len {} != seq len {s}",
            mask.len()
        )));
    }
    // Build a [1, S, 1] BF16 mask tensor and broadcast-multiply.
    let mask_t = Tensor::from_f32_to_bf16(
        mask.to_vec(),
        Shape::from_dims(&[1, s, 1]),
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("mask alloc: {e:?}")))?;
    x.mul(&mask_t)
        .map_err(|e| RunError::Other(format!("mask mul: {e:?}")))
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux/klein/sd3/qwenimage/ernie)
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
