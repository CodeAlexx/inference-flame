//! Anima (Cosmos Predict2) real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3,qwenimage,ernie}.rs` structurally —
//! lazy load, drop-to-fit VRAM dance, per-step Started/Progress/Done event
//! emission. Underlying pipeline mirrors
//! `inference-flame/src/bin/anima_infer.rs`.
//!
//! ## Architecture
//!
//! Cosmos Predict2 DiT — 28 blocks (`AnimaConfig::default().num_blocks`),
//! hidden=2048, 3D RoPE, patchify via a linear patch_embed. Model is ~3.9 GB
//! and loaded entirely to GPU via `Anima::new_all_on_gpu(...)`. No block
//! offloading.
//!
//! ## Text encoder — pre-computed context
//!
//! Anima's LLM adapter transforms Qwen3 0.6B hidden states into the
//! 1024-dim context consumed by `forward_with_context`. The Python pipeline
//! runs Qwen3 + the adapter's cross-attention and caches the resulting
//! context tensor.
//!
//! AGENT-DEFAULT: this worker mirrors the bin convention exactly. It reads
//! pre-computed `context_cond` + `context_uncond` tensors from a safetensors
//! file at `EMBEDS_PATH`. `job.prompt`/`job.negative` are IGNORED with an
//! UNAMBIGUOUS warning log. The user must pre-compute before clicking
//! Generate; if the file is missing we Fail with a clear error pointing at
//! the bin.
//!
//! The Rust Qwen3 0.6B encoder + adapter forward path exists
//! (`Anima::encode_context`) but no bin has been wired up to use it yet —
//! doing so is out of scope for Batch C. Tracked as future work.
//!
//! ### Known v1 limitation: prompts are NOT typed in the UI
//!
//! Same as the qwenimage worker: typed prompts are ignored. The cached
//! context file decides the result. The warning log says so explicitly. The
//! UI has no per-job warning channel today, so this is the only signal a
//! UI user gets that their typed prompt was discarded. Future work: wire
//! `Anima::encode_context` into this worker and remove the cache.
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
//! produces wrong colors it's a parity issue with the anima shipping VAE
//! that Batch C isn't resolving.
//!
//! ### Known caveat: VAE color parity unverified
//!
//! `anima_infer.rs` saves the latent and instructs the user to run
//! `decode_anima_latent.py`. This worker, by contrast, calls
//! `QwenImageVaeDecoder` directly — bypassing the Python decode reference.
//! Shape and permute pipeline is correct (audited against the bin), but
//! whether the QwenImage VAE checkpoint is the right calibration for Anima
//! latents is **unverified by code review**. If the first real run
//! produces wrong colors, this is the first place to look. Two likely
//! culprits:
//!   1. The shipped `qwen_image_vae.safetensors` differs subtly from the
//!      one the qwenimage worker uses (different scale/shift constants
//!      bundled into the weights).
//!   2. The latent stats (mean/std) Anima emits aren't calibrated to that
//!      VAE's expected input distribution.
//!
//! ## Latent
//!
//! [1, T=1, H/8, W/8, 16] — 5D with the channel axis LAST (different from
//! the other workers). Permuted to [1, 16, T, H, W] for the VAE.
//!
//! ## VRAM budget on a 24 GB card
//!
//! - Cached context tensors:              ~50-200 MB
//! - Anima DiT (all resident, BF16):      ~3.9 GB
//! - Qwen-Image VAE decoder:              ~0.4 GB
//! - Activations (1024² latent):          ~2-3 GB
//!
//! Comfortable. No text-encoder VRAM cost (cached). Drop DiT before VAE
//! decode for the usual cuDNN-workspace safety margin.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::anima::{load_all_weights, Anima};
use inference_flame::vae::QwenImageVaeDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors anima_infer.rs).
// ===========================================================================

/// Anima DiT — single consolidated safetensors file.
const MODEL_PATH: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors";

/// Qwen-Image VAE (shipped with Anima).
const VAE_PATH: &str =
    "/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors";

/// Pre-computed cond + uncond context tensors safetensors file.
///
/// AGENT-DEFAULT: hardcoded path, same as qwenimage.rs. Must contain
/// `context_cond` and `context_uncond` tensors (each [1, S_txt, 1024] BF16 —
/// Anima's post-adapter text context). Generated by the Python pipeline.
const EMBEDS_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors";

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

    if !job.prompt.is_empty() {
        // UNAMBIGUOUS — same rationale as the qwenimage worker. The UI has
        // no per-job warning channel today; the log is the only signal.
        log::warn!(
            "ANIMA: TYPED PROMPT IS IGNORED. Inference will use the CACHED \
             context tensors at {EMBEDS_PATH}. To change the prompt, re-run \
             the Python pre-compute pipeline. (See module-level docstring's \
             'Known v1 limitation' section.)"
        );
    }

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

    // -------- 1. Load pre-computed cond + uncond context --------
    if !Path::new(EMBEDS_PATH).exists() {
        return Err(RunError::Other(format!(
            "Anima context file not found at {EMBEDS_PATH} — \
             pre-compute via the Python pipeline first"
        )));
    }
    log::info!("Anima: loading cached context from {EMBEDS_PATH}");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(Path::new(EMBEDS_PATH), &device)
        .map_err(|e| RunError::Other(format!("context load: {e:?}")))?;
    let context_cond = emb
        .get("context_cond")
        .ok_or_else(|| RunError::Other("missing 'context_cond' in embeddings file".into()))?
        .clone();
    let context_uncond = emb
        .get("context_uncond")
        .ok_or_else(|| RunError::Other("missing 'context_uncond' in embeddings file".into()))?
        .clone();
    drop(emb);
    let context_cond = ensure_bf16(context_cond)?;
    let context_uncond = ensure_bf16(context_uncond)?;
    log::info!(
        "Anima: context_cond {:?}, context_uncond {:?} ({:.1}s)",
        context_cond.shape().dims(),
        context_uncond.shape().dims(),
        t0.elapsed().as_secs_f32(),
    );

    drain_pending(ui_rx, pending)?;

    // -------- 2. Load Anima DiT (all weights on GPU) --------
    if !Path::new(MODEL_PATH).exists() {
        return Err(RunError::Other(format!(
            "Anima model not found at {MODEL_PATH}"
        )));
    }
    log::info!("Anima: loading DiT from {MODEL_PATH}");
    let t0 = Instant::now();
    let all_weights = load_all_weights(MODEL_PATH, &device)
        .map_err(|e| RunError::Other(format!("weight load: {e:?}")))?;
    let mut model = Anima::new_all_on_gpu(MODEL_PATH.to_string(), all_weights, device.clone());
    log::info!("Anima: DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

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

fn ensure_bf16(t: Tensor) -> Result<Tensor, RunError> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        t.to_dtype(DType::BF16)
            .map_err(|e| RunError::Other(format!("ensure_bf16: {e:?}")))
    }
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
