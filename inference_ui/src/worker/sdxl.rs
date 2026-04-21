//! SDXL real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3,qwenimage,ernie,anima}.rs` —
//! lazy load, drop-to-fit VRAM dance, per-step Started/Progress/Done event
//! emission. Underlying pipeline mirrors
//! `inference-flame/src/bin/sdxl_infer.rs`.
//!
//! ## Architecture
//!
//! SDXL UNet — LDM-style UNet2D (BigDiffusers SDXL Base 1.0 architecture).
//! ε-prediction (NOT velocity, NOT flow). Loaded all-on-GPU via
//! `SDXLUNet::from_safetensors_all_gpu(...)` from a pre-extracted BF16
//! safetensors file.
//!
//! ## Text encoders — pre-computed embeddings
//!
//! SDXL uses dual encoders (CLIP-L + CLIP-G). The Rust CLIP-L/G encoders
//! exist (`models::clip_encoder`), but SDXL also needs the CLIP-G
//! `text_projection` matrix and the dual-pool concatenation that the
//! reference bin doesn't bother to wire up (it uses cached embeddings
//! produced by `scripts/cache_sdxl_embeddings.py`).
//!
//! AGENT-DEFAULT: this worker mirrors the bin convention exactly. It reads
//! pre-computed `context`, `y`, `context_uncond`, `y_uncond` tensors from a
//! safetensors file at one of `EMBEDS_PATHS` (first existing wins, mirroring
//! the qwenimage worker's resolve pattern). `job.prompt`/`job.negative` are
//! IGNORED with an UNAMBIGUOUS warning log. The user must pre-compute
//! before clicking Generate; if no candidate file exists we Fail with a
//! clear error pointing at the script.
//!
//! ### Known v1 limitation: prompts are NOT typed in the UI
//!
//! Same as the qwenimage / anima workers: typed prompts are ignored. The
//! cached embeddings file decides the result. The warning log says so
//! explicitly. Future work: wire `ClipEncoder` (CLIP-L + CLIP-G with
//! `text_projection` for CLIP-G's pooled output) into this worker and
//! remove the cache.
//!
//! ## CFG
//!
//! Real two-pass (cond + uncond). `pred = uncond + cfg * (cond - uncond)`.
//! Default cfg=7.5 matches the SDXL reference.
//!
//! ## Schedule — VE (variance-exploding), eps-prediction
//!
//! Scaled-linear β schedule (NOT rectified flow). Build sigmas via
//! `((1 - alpha) / alpha).sqrt()` from `alphas_cumprod`. Leading timestep
//! spacing with `steps_offset=1` per `EulerDiscreteScheduler` defaults.
//! Mirrors `sdxl_infer.rs::build_sdxl_schedule`.
//!
//! ## Euler step formula (VE eps-prediction)
//!
//! Per `sdxl_infer.rs:163-222`:
//! ```text
//!   c_in = 1 / sqrt(sigma^2 + 1)
//!   x_in = x * c_in                              # input scaling
//!   eps = unet(x_in, t, context, y)              # ε prediction (BF16 ok)
//!   eps_cfg = uncond + cfg * (cond - uncond)     # in FP32
//!   dt = sigma_next - sigma
//!   x_next = x + dt * eps_cfg                    # Euler step
//! ```
//! NOT the flow-matching `x = x + dt * v` formula — `eps` here IS the noise
//! direction the EulerDiscrete scheduler advances along, and the `dt` is
//! negative (sigma_next < sigma) so we move toward less noise. Initial x is
//! `noise * sqrt(sigma_max^2 + 1)` (per diffusers `init_noise_sigma`).
//!
//! ## VAE
//!
//! 4-channel SDXL VAE, embedded in the SDXL combined `.safetensors` blob
//! (pulled by key prefix). Decoded via `LdmVAEDecoder::from_safetensors`
//! with scale=0.13025, shift=0.0. 8× downscale.
//!
//! ## Latent
//!
//! NCHW [1, 4, H/8, W/8].
//!
//! ## VRAM budget on a 24 GB card
//!
//! - Cached embeddings: ~10 MB
//! - SDXL UNet (resident BF16): ~5 GB
//! - VAE decoder (load on demand): ~0.3 GB
//! - Activations (1024² latent): ~2-3 GB
//!
//! Comfortable. We drop the UNet before VAE decode (cuDNN conv workspace).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::sdxl_unet::SDXLUNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors sdxl_infer.rs).
//
// AGENT-DEFAULT: hardcoded constants. Same rationale as flux.rs / klein.rs.
// ===========================================================================

/// SDXL UNet — pre-extracted BF16 safetensors with stripped/HWIO-prepared keys.
const MODEL_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors";

/// SDXL VAE — pulled from the combined SDXL Base 1.0 checkpoint by key prefix.
const VAE_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors";

/// Pre-computed cond + uncond text embeddings safetensors file.
///
/// AGENT-DEFAULT: hardcoded paths checked in preference order. Each
/// candidate file must contain "context", "y", "context_uncond", "y_uncond"
/// tensors (CLIP-L + CLIP-G dual-encoder outputs). Generate via:
///   python3 scripts/cache_sdxl_embeddings.py --prompt "..." --negative "..."
///
/// Order:
///   1. bare canonical name (so a future user pre-computing to that
///      filename Just Works — and `scripts/cache_sdxl_embeddings.py` already
///      writes here)
///   2. `.py` infix variant (alternate name a user may have written)
///   3. `.rust` infix variant (what an eventual Rust encoder might write)
///
/// `job.prompt` / `job.negative` are IGNORED — whatever was used to generate
/// the embeddings file is what gets used for inference.
const EMBEDS_PATHS: &[&str] = &[
    "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors",
    "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.py.safetensors",
    "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.rust.safetensors",
];

// SDXL VAE — 4-channel, 8x conv stride. scale=0.13025 (SDXL-specific, NOT
// SD 1.5's 0.18215). shift=0.0.
const VAE_IN_CHANNELS: usize = 4;
const VAE_SCALE: f32 = 0.13025;
const VAE_SHIFT: f32 = 0.0;

/// Default CFG. SDXL reference is 7.5.
const DEFAULT_CFG: f32 = 7.5;

/// Default step count. SDXL reference is 30.
const DEFAULT_STEPS: u32 = 30;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local SDXL resources. Nothing kept resident — embeddings, UNet,
/// and VAE are loaded per-job. The UNet is ~5 GB so reload tax is small
/// (~5s warm SSD) and the VRAM safety dwarfs the load-time win.
pub struct SdxlState {
    pub device: Arc<CudaDevice>,
}

impl SdxlState {
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
    state: &mut SdxlState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let steps: u32 = if job.steps == 0 { DEFAULT_STEPS } else { job.steps };
    let cfg_scale: f32 = if job.cfg <= 0.0 { DEFAULT_CFG } else { job.cfg };

    if !job.prompt.is_empty() {
        // UNAMBIGUOUS — mirrors qwenimage / anima rationale. The UI has no
        // per-job warning channel today; the log is the only signal.
        log::warn!(
            "SDXL: TYPED PROMPT IS IGNORED. Inference will use the CACHED \
             embeddings file. Candidates checked (first existing wins): {:?}. \
             To change the prompt, re-run scripts/cache_sdxl_embeddings.py.",
            EMBEDS_PATHS
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
            log::warn!("SDXL job {} failed: {msg}", job.id);
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
    state: &mut SdxlState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap so we ALWAYS flush the pool on every exit path. Mirrors
    // the comment in flux.rs / klein.rs / sd3.rs: an early Cancelled return
    // otherwise skips the scattered `clear_pool_cache` calls and pooled
    // buffers race the next job's load.
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
    state: &mut SdxlState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. Load cached cond + uncond embeddings --------
    let embeds_path = EMBEDS_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
        .ok_or_else(|| {
            RunError::Other(format!(
                "SDXL embeddings file not found. Checked:\n  - {}\n\
                 Generate one of these via `python3 scripts/cache_sdxl_embeddings.py` first.",
                EMBEDS_PATHS.join("\n  - ")
            ))
        })?;
    log::info!("SDXL: loading cached embeddings from {embeds_path}");
    let t0 = Instant::now();
    let emb = flame_core::serialization::load_file(Path::new(embeds_path), &device)
        .map_err(|e| RunError::Other(format!("embeddings load: {e:?}")))?;
    let context = emb
        .get("context")
        .ok_or_else(|| RunError::Other("missing 'context' in embeddings file".into()))?
        .clone();
    let y = emb
        .get("y")
        .ok_or_else(|| RunError::Other("missing 'y' in embeddings file".into()))?
        .clone();
    let context_uncond = emb
        .get("context_uncond")
        .ok_or_else(|| RunError::Other("missing 'context_uncond' in embeddings file".into()))?
        .clone();
    let y_uncond = emb
        .get("y_uncond")
        .ok_or_else(|| RunError::Other("missing 'y_uncond' in embeddings file".into()))?
        .clone();
    drop(emb);
    let context = ensure_bf16(context)?;
    let y = ensure_bf16(y)?;
    let context_uncond = ensure_bf16(context_uncond)?;
    let y_uncond = ensure_bf16(y_uncond)?;
    log::info!(
        "SDXL: context {:?}, y {:?} ({:.1}s)",
        context.shape().dims(),
        y.shape().dims(),
        t0.elapsed().as_secs_f32(),
    );

    drain_pending(ui_rx, pending)?;

    // -------- 2. Load SDXL UNet (all weights on GPU) --------
    if !Path::new(MODEL_PATH).exists() {
        return Err(RunError::Other(format!(
            "SDXL UNet not found at {MODEL_PATH} \
             (expected pre-extracted BF16 safetensors)"
        )));
    }
    log::info!("SDXL: loading UNet from {MODEL_PATH}");
    let t0 = Instant::now();
    let mut model = SDXLUNet::from_safetensors_all_gpu(MODEL_PATH, &device)
        .map_err(|e| RunError::Other(format!("UNet load: {e:?}")))?;
    log::info!("SDXL: UNet loaded in {:.1}s", t0.elapsed().as_secs_f32());

    drain_pending(ui_rx, pending)?;

    // -------- 3. Build noise + schedule --------
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
    log::info!(
        "SDXL: latent [1, {VAE_IN_CHANNELS}, {latent_h}, {latent_w}] (8x downscale of {width}x{height})"
    );

    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    // AGENT-DEFAULT: Tensor::randn_seeded (Box-Muller per docstring) instead
    // of hand-rolling the loop in sdxl_infer.rs. Same rationale as flux.rs /
    // sd3.rs. Statistical distribution matches; bit-exact bytes don't.
    let noise = Tensor::randn_seeded(
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?;

    let (sigmas, timesteps) = build_sdxl_schedule(steps as usize);
    log::info!(
        "SDXL: schedule sigma_max={:.4} sigma_min={:.6} t_max={:.0} t_min={:.0}",
        sigmas[0],
        sigmas[steps as usize - 1],
        timesteps[0],
        timesteps[steps as usize - 1],
    );

    // x = noise * init_noise_sigma where init_noise_sigma = sqrt(sigma_max^2 + 1)
    // per diffusers EulerDiscreteScheduler. Mirrors sdxl_infer.rs:139-140.
    let init_sigma = (sigmas[0] * sigmas[0] + 1.0).sqrt();
    let x_bf16 = noise
        .to_dtype(DType::BF16)
        .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?
        .mul_scalar(init_sigma)
        .map_err(|e| RunError::Other(format!("init_sigma scale: {e:?}")))?;

    drain_pending(ui_rx, pending)?;

    // -------- 4. Denoise loop with real two-pass CFG --------
    // Mirrors sdxl_infer.rs:155-228. Keep denoising state in FP32; only
    // convert to BF16 for UNet input. CFG combine in FP32 so the small
    // diff between cond/uncond doesn't get rounded away.
    let t_denoise = Instant::now();
    let mut x_f32 = x_bf16
        .to_dtype(DType::F32)
        .map_err(|e| RunError::Other(format!("x -> f32: {e:?}")))?;
    drop(x_bf16);

    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];

        // Scoped block: per-step temporaries drop at `}` BEFORE the next
        // step's allocations. Same defensive pattern as flux/klein/chroma/sd3.
        let next_x = {
            // Scale input: x_in = x / sqrt(sigma^2 + 1), convert to BF16.
            let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
            let x_in = x_f32
                .mul_scalar(c_in)
                .map_err(|e| RunError::Other(format!("c_in scale: {e:?}")))?
                .to_dtype(DType::BF16)
                .map_err(|e| RunError::Other(format!("x_in -> bf16: {e:?}")))?;

            // SDXL UNet expects discrete timestep (0-999), NOT sigma. Mirrors
            // sdxl_infer.rs:168.
            let timestep = Tensor::from_f32_to_bf16(
                vec![timesteps[step]],
                Shape::from_dims(&[1]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("timestep alloc: {e:?}")))?;

            let pred_cond = model
                .forward(&x_in, &timestep, &context, &y)
                .map_err(|e| RunError::Other(format!("cond forward step {step}: {e:?}")))?;
            let pred_uncond = model
                .forward(&x_in, &timestep, &context_uncond, &y_uncond)
                .map_err(|e| RunError::Other(format!("uncond forward step {step}: {e:?}")))?;

            // CFG combine in FP32: pred = uncond + cfg * (cond - uncond).
            // (Conventional formula; see sdxl_infer.rs:217-218.)
            let pred_cond_f32 = pred_cond
                .to_dtype(DType::F32)
                .map_err(|e| RunError::Other(format!("pred_cond -> f32: {e:?}")))?;
            let pred_uncond_f32 = pred_uncond
                .to_dtype(DType::F32)
                .map_err(|e| RunError::Other(format!("pred_uncond -> f32: {e:?}")))?;
            let diff = pred_cond_f32
                .sub(&pred_uncond_f32)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            let pred_f32 = pred_uncond_f32
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // VE eps-prediction Euler step in FP32: x_next = x + (sigma_next - sigma) * eps.
            // sigma_next < sigma so dt is negative; the step moves toward
            // less noise. Mirrors sdxl_infer.rs:221-222.
            let dt = sigma_next - sigma;
            let step_tensor = pred_f32
                .mul_scalar(dt)
                .map_err(|e| RunError::Other(format!("euler mul: {e:?}")))?;
            x_f32
                .add(&step_tensor)
                .map_err(|e| RunError::Other(format!("euler add: {e:?}")))?
        };
        x_f32 = next_x;

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

    // Convert final latent back to BF16 for VAE input.
    let x = x_f32
        .to_dtype(DType::BF16)
        .map_err(|e| RunError::Other(format!("x -> bf16 for VAE: {e:?}")))?;

    // -------- 5. Drop UNet + embeddings before VAE --------
    // Mirrors sdxl_infer.rs:240. The 1024² VAE conv workspace can OOM if
    // the UNet + embed tensors are still resident.
    drop(model);
    drop(context);
    drop(y);
    drop(context_uncond);
    drop(y_uncond);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("SDXL: UNet + embeds dropped, pool flushed before VAE");

    // -------- 6. VAE decode (load + use + drop) --------
    if !Path::new(VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "SDXL VAE source not found at {VAE_PATH}"
        )));
    }
    let t_vae = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        VAE_PATH,
        VAE_IN_CHANNELS,
        VAE_SCALE,
        VAE_SHIFT,
        &device,
    )
    .map_err(|e| RunError::Other(format!("VAE load: {e:?}")))?;
    let rgb = vae
        .decode(&x)
        .map_err(|e| RunError::Other(format!("VAE decode: {e:?}")))?;
    drop(x);
    drop(vae);
    log::info!("SDXL: VAE decode done in {:.1}s", t_vae.elapsed().as_secs_f32());

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Build SDXL's scaled-linear β schedule. Returns (sigmas, timesteps) where:
/// - `sigmas`: `num_steps + 1` values, `sigmas[0]` = sigma_max, `sigmas[-1]` = 0.0
/// - `timesteps`: `num_steps` discrete timesteps the UNet expects (0-999)
///
/// Mirrors `sdxl_infer.rs::build_sdxl_schedule` exactly.
fn build_sdxl_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

    // Scaled-linear beta schedule (SDXL default: beta_schedule="scaled_linear").
    let betas: Vec<f64> = (0..num_train_steps)
        .map(|i| {
            let v = beta_start.sqrt()
                + (beta_end.sqrt() - beta_start.sqrt()) * i as f64
                    / (num_train_steps - 1) as f64;
            v * v
        })
        .collect();

    let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
    let mut prod = 1.0f64;
    for &b in &betas {
        prod *= 1.0 - b;
        alphas_cumprod.push(prod);
    }

    // Leading timestep spacing with steps_offset=1 (SDXL EulerDiscreteScheduler default).
    let step_ratio = num_train_steps / num_steps;
    let mut ts: Vec<usize> = (0..num_steps).map(|i| i * step_ratio + 1).collect();
    ts.reverse(); // high noise first

    let mut sigmas = Vec::with_capacity(num_steps + 1);
    let mut timesteps = Vec::with_capacity(num_steps);
    for &t in &ts {
        let t = t.min(num_train_steps - 1);
        let alpha = alphas_cumprod[t];
        let sigma = ((1.0 - alpha) / alpha).sqrt();
        sigmas.push(sigma as f32);
        timesteps.push(t as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
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
// Cancel + image-conversion helpers (same as flux/klein/sd3/qwenimage/ernie/anima)
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
