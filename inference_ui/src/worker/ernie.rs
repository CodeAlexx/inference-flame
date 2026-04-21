//! ERNIE-Image real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3,qwenimage}.rs` structurally — lazy
//! load, drop-to-fit VRAM dance, per-step Started/Progress/Done event
//! emission. Underlying pipeline mirrors
//! `inference-flame/src/bin/ernie_image_infer.rs`.
//!
//! ## Architecture
//!
//! Single-stream DiT — 36 layers (`ErnieImageConfig::default().num_layers`),
//! patch_size=2. Loaded resident via `ErnieImageModel::load(weights, cfg)`.
//! ~15 GB BF16, fits comfortably on a 24 GB card.
//!
//! ## Text encoder
//!
//! Mistral-3 3B (in-process, Rust port at `models/mistral3b_encoder.rs`).
//! Loaded inside a scope, encodes both cond + uncond, then dropped before
//! the DiT loads. ~7 GB BF16. The empty string "" is encoded as the uncond
//! embedding (NOT zeros — diffusers ErnieImagePipeline does the same; using
//! zeros biases CFG toward "anything-not-the-prompt" instead of "generic
//! image"). Mirrors `ernie_image_infer.rs:73-87`.
//!
//! ## CFG — sequential per-pass with explicit pool clearing
//!
//! ERNIE's per-forward attention scores are 2-4 GB. Running cond and uncond
//! back-to-back without an intermediate `clear_pool_cache` + `trim_cuda_mempool`
//! OOMs by step ~5. We mirror `ernie_image_infer.rs:181-186` exactly:
//!
//! ```text
//!   pred_cond = model.forward(...);
//!   clear_pool_cache(); trim_cuda_mempool(0);
//!   pred_uncond = model.forward(...);
//!   clear_pool_cache(); trim_cuda_mempool(0);
//!   pred = uncond + cfg * (cond - uncond)
//! ```
//!
//! ## Schedule
//!
//! Flow-matching with fixed shift=3.0, exponential time shift. Provided by
//! `sampling::ernie_sampling::ernie_schedule`. `sigma == flow_t` directly;
//! the discrete model timestep is `sigma * 1000` via `sigma_to_timestep`.
//! Velocity is the model output; `ernie_euler_step` does
//! `x_next = x + (sigma_next - sigma) * v`.
//!
//! ## VAE
//!
//! Klein VAE (16-channel, 8× downscale × 2× patch = 16× effective). Same
//! decoder used by the Klein 4B/9B workers but loaded fresh per-job (no
//! cross-worker cache). ~700 MB.
//!
//! ## Latent
//!
//! NCHW [1, 128, H/16, W/16]. The bin uses `latent_h = HEIGHT / 16` directly.
//!
//! ## VRAM budget on a 24 GB card
//!
//! - Mistral-3 3B (load → encode → drop):    ~7 GB peak
//! - ERNIE-Image DiT (resident BF16):        ~15 GB
//! - Klein VAE decoder:                      ~0.7 GB
//! - Activations + per-forward transients:   ~2-4 GB
//! - **Total resident peak (denoise)**:      ~17-20 GB
//!
//! Tight but workable. Sequential CFG + per-pass pool flush is what makes
//! it fit. Mistral is dropped before the DiT loads. DiT is dropped before
//! VAE decode.
//!
//! ## Caching across jobs
//!
//! Nothing kept resident — same as SD3.5 worker. The per-forward pool
//! clearing makes resident caching dangerous (we'd be racing the next job's
//! load), and ERNIE's text encoder is a different one than the other
//! workers so there's no shared cache opportunity.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::ernie_image::{ErnieImageConfig, ErnieImageModel};
use inference_flame::models::mistral3b_encoder::Mistral3bEncoder;
use inference_flame::sampling::ernie_sampling::{
    ernie_euler_step, ernie_schedule, sigma_to_timestep,
};
use inference_flame::vae::klein_vae::KleinVaeDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors ernie_image_infer.rs).
// ===========================================================================

const TRANSFORMER_DIR: &str = "/home/alex/models/ERNIE-Image/transformer";
const TEXT_ENCODER: &str = "/home/alex/models/ERNIE-Image/text_encoder/model.safetensors";
const TOKENIZER: &str = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json";
const VAE_PATH: &str = "/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors";

/// 16× effective downscale (8× VAE conv stride × 2× DiT patch).
const VAE_DOWNSCALE: usize = 16;

/// Mistral-3 max sequence length used by the ERNIE pipeline.
const MAX_TEXT_LEN: usize = 256;

/// Default CFG scale. ernie_image_infer.rs uses 4.0.
const DEFAULT_CFG: f32 = 4.0;

/// Default step count. ernie_image_infer.rs uses 50 (SFT, not Turbo).
const DEFAULT_STEPS: u32 = 50;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local ERNIE resources. Nothing kept resident — see module-level
/// comment for the rationale.
pub struct ErnieState {
    pub device: Arc<CudaDevice>,
}

impl ErnieState {
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
    state: &mut ErnieState,
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
            log::warn!("ERNIE-Image job {} failed: {msg}", job.id);
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
    state: &mut ErnieState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap so we ALWAYS flush the pool. Same rationale as flux/klein.
    // ERNIE's per-forward 2-4 GB transients make this even more important —
    // an early Cancelled return without flush would race the next job.
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
    state: &mut ErnieState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. Mistral-3 3B encode (cond + uncond), then drop --------
    if !Path::new(TOKENIZER).exists() {
        return Err(RunError::Other(format!(
            "ERNIE tokenizer not found at {TOKENIZER}"
        )));
    }
    if !Path::new(TEXT_ENCODER).exists() {
        return Err(RunError::Other(format!(
            "ERNIE text encoder not found at {TEXT_ENCODER}"
        )));
    }
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER)
        .map_err(|e| RunError::Other(format!("tokenizer load: {e}")))?;

    let cond_enc = tokenizer
        .encode(job.prompt.as_str(), true)
        .map_err(|e| RunError::Other(format!("tokenize prompt: {e}")))?;
    let cond_ids: Vec<i32> = cond_enc.get_ids().iter().map(|&id| id as i32).collect();
    let cond_real_len = cond_ids.len();

    // Empty string for uncond, NOT zeros. Mirrors ernie_image_infer.rs:73.
    // (Negative prompt is ignored — diffusers ErnieImagePipeline encodes the
    //  empty string for the uncond branch.)
    let uncond_enc = tokenizer
        .encode("", true)
        .map_err(|e| RunError::Other(format!("tokenize empty: {e}")))?;
    let uncond_ids: Vec<i32> = uncond_enc.get_ids().iter().map(|&id| id as i32).collect();
    let uncond_real_len = uncond_ids.len();

    log::info!(
        "ERNIE: tokenized cond {}, uncond {} (max {MAX_TEXT_LEN})",
        cond_real_len,
        uncond_real_len
    );

    let t0 = Instant::now();
    let (text_embeds, uncond_embeds_real) = {
        log::info!("ERNIE: loading Mistral-3 3B encoder from {TEXT_ENCODER}");
        let encoder = Mistral3bEncoder::load(TEXT_ENCODER, &device)
            .map_err(|e| RunError::Other(format!("Mistral load: {e:?}")))?;
        let cond_emb = encoder
            .encode(&cond_ids, MAX_TEXT_LEN)
            .map_err(|e| RunError::Other(format!("Mistral encode cond: {e:?}")))?;
        let uncond_emb = encoder
            .encode(&uncond_ids, MAX_TEXT_LEN)
            .map_err(|e| RunError::Other(format!("Mistral encode uncond: {e:?}")))?;
        (cond_emb, uncond_emb)
        // encoder dropped here — ~7 GB freed
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "ERNIE: text encode done in {:.1}s, encoder dropped",
        t0.elapsed().as_secs_f32()
    );

    drain_pending(ui_rx, pending)?;

    // -------- 2. Trim text embeddings to real length --------
    // ERNIE has no attention-mask path; padded tokens steal softmax weight
    // from real text. Mirrors ernie_image_infer.rs:165-168.
    let text_3d = if text_embeds.rank() == 2 {
        text_embeds
            .unsqueeze(0)
            .map_err(|e| RunError::Other(format!("cond unsqueeze: {e:?}")))?
    } else {
        text_embeds.clone()
    };
    let uncond_3d = if uncond_embeds_real.rank() == 2 {
        uncond_embeds_real
            .unsqueeze(0)
            .map_err(|e| RunError::Other(format!("uncond unsqueeze: {e:?}")))?
    } else {
        uncond_embeds_real.clone()
    };
    let cond_len_clamped = cond_real_len.min(MAX_TEXT_LEN).max(1);
    let uncond_len_clamped = uncond_real_len.min(MAX_TEXT_LEN).max(1);
    let cond_trim = text_3d
        .narrow(1, 0, cond_len_clamped)
        .map_err(|e| RunError::Other(format!("cond narrow: {e:?}")))?;
    let uncond_trim = uncond_3d
        .narrow(1, 0, uncond_len_clamped)
        .map_err(|e| RunError::Other(format!("uncond narrow: {e:?}")))?;
    let cond_lens = vec![cond_len_clamped];
    let uncond_lens = vec![uncond_len_clamped];

    drain_pending(ui_rx, pending)?;

    // -------- 3. Load ERNIE-Image DiT (all blocks resident) --------
    if !Path::new(TRANSFORMER_DIR).exists() {
        return Err(RunError::Other(format!(
            "ERNIE transformer dir not found at {TRANSFORMER_DIR}"
        )));
    }
    log::info!("ERNIE: loading DiT from {TRANSFORMER_DIR}");
    let t1 = Instant::now();
    let shard_paths = {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(TRANSFORMER_DIR)
            .map_err(|e| RunError::Other(format!("read_dir {TRANSFORMER_DIR}: {e}")))?
        {
            let p = entry
                .map_err(|e| RunError::Other(format!("read_dir entry: {e}")))?
                .path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                paths.push(p.to_string_lossy().into_owned());
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err(RunError::Other(format!(
                "no .safetensors shards in {TRANSFORMER_DIR}"
            )));
        }
        paths
    };

    let mut all_weights = std::collections::HashMap::new();
    for path in &shard_paths {
        let partial = flame_core::serialization::load_file(Path::new(path), &device)
            .map_err(|e| RunError::Other(format!("load shard {path}: {e:?}")))?;
        for (k, v) in partial {
            all_weights.insert(k, v);
        }
    }
    let config = ErnieImageConfig::default();
    let model = ErnieImageModel::load(all_weights, config.clone())
        .map_err(|e| RunError::Other(format!("DiT build: {e:?}")))?;
    log::info!(
        "ERNIE: DiT loaded in {:.1}s ({} blocks resident)",
        t1.elapsed().as_secs_f32(),
        config.num_layers
    );

    drain_pending(ui_rx, pending)?;

    // -------- 4. Latent geometry + seeded noise --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    let latent_h = height / VAE_DOWNSCALE;
    let latent_w = width / VAE_DOWNSCALE;
    if latent_h == 0 || latent_w == 0 {
        return Err(RunError::Other(format!(
            "image too small: {width}x{height} (latent {latent_h}x{latent_w})"
        )));
    }
    log::info!(
        "ERNIE: latent [1, 128, {latent_h}, {latent_w}] (16× downscale of {width}x{height})"
    );

    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    // AGENT-DEFAULT: Tensor::randn_seeded for noise (Box-Muller per docstring).
    let mut latent = Tensor::randn_seeded(
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?;

    // -------- 5. Schedule --------
    let sigmas = ernie_schedule(steps as usize);
    log::info!(
        "ERNIE: schedule {} values, sigma[0]={:.4}, sigma[-1]={:.4}",
        sigmas.len(),
        sigmas[0],
        sigmas[steps as usize]
    );

    drain_pending(ui_rx, pending)?;

    // -------- 6. Sequential CFG denoise loop --------
    let t_denoise = Instant::now();
    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())
            .map_err(|e| RunError::Other(format!("t_tensor alloc: {e:?}")))?;

        // Sequential CFG with explicit pool clears between cond + uncond.
        // Mirrors ernie_image_infer.rs:181-186 EXACTLY — without the
        // intermediate clear, per-forward attention transients (2-4 GB)
        // accumulate and OOM by ~step 5.
        let pred = if cfg_scale > 1.0 {
            let pred_cond = model
                .forward(&latent, &t_tensor, &cond_trim, &cond_lens)
                .map_err(|e| RunError::Other(format!("cond fwd step {step}: {e:?}")))?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            let pred_uncond = model
                .forward(&latent, &t_tensor, &uncond_trim, &uncond_lens)
                .map_err(|e| RunError::Other(format!("uncond fwd step {step}: {e:?}")))?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);

            let diff = pred_cond
                .sub(&pred_uncond)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            pred_uncond
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?
        } else {
            let p = model
                .forward(&latent, &t_tensor, &cond_trim, &cond_lens)
                .map_err(|e| RunError::Other(format!("cond fwd step {step}: {e:?}")))?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            p
        };

        latent = ernie_euler_step(&latent, &pred, sigma, sigma_next)
            .map_err(|e| RunError::Other(format!("euler step {step}: {e:?}")))?;
        drop(pred);

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

    // -------- 7. Drop DiT before VAE decode --------
    drop(model);
    drop(cond_trim);
    drop(uncond_trim);
    drop(text_3d);
    drop(uncond_3d);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("ERNIE: DiT + embeddings dropped before VAE");

    // -------- 8. VAE decode --------
    if !Path::new(VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "ERNIE VAE not found at {VAE_PATH}"
        )));
    }
    let t_vae = Instant::now();
    let vae_weights = flame_core::serialization::load_file(Path::new(VAE_PATH), &device)
        .map_err(|e| RunError::Other(format!("VAE load_file: {e:?}")))?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)
        .map_err(|e| RunError::Other(format!("VAE build: {e:?}")))?;
    drop(vae_weights);

    let rgb = vae
        .decode(&latent)
        .map_err(|e| RunError::Other(format!("VAE decode: {e:?}")))?;
    drop(latent);
    drop(vae);
    log::info!("ERNIE: VAE decoded in {:.1}s", t_vae.elapsed().as_secs_f32());

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux/klein/sd3/qwenimage)
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

/// Convert a [1, 3, H, W] BF16 tensor to ColorImage. ERNIE's Klein VAE
/// produces values in [-1, 1] (same convention as the other workers'
/// VAEs, including `qwenimage_gen.rs`'s post-process).
fn decoded_to_color_image(rgb: &Tensor) -> Result<ColorImage, flame_core::Error> {
    let dims = rgb.shape().dims();
    if dims.len() != 4 || dims[0] != 1 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "VAE decode shape {dims:?} != [1, 3, H, W]"
        )));
    }
    let c = dims[1];
    let h = dims[2];
    let w = dims[3];
    let f32_t = rgb.to_dtype(DType::F32)?;
    let host = f32_t.to_vec()?;

    let mut pixels = Vec::with_capacity(h * w);
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = if c >= 1 {
                ((host[i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
            } else {
                0
            };
            let g = if c >= 2 {
                ((host[plane + i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
            } else {
                r
            };
            let b = if c >= 3 {
                ((host[2 * plane + i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
            } else {
                r
            };
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    Ok(ColorImage {
        size: [w, h],
        pixels,
    })
}
