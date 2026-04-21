//! SD 3.5 Medium real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,zimage}.rs` structurally — lazy load,
//! drop-to-fit VRAM dance, per-step Started/Progress/Done event emission.
//! Underlying pipeline mirrors `inference-flame/src/bin/sd3_medium_infer.rs`.
//!
//! ## Architecture
//!
//! MMDiT — depth=24, hidden=1536, no dual attention (Medium variant). The
//! file uses `SD3MMDiT` from `models::sd3_mmdit` which auto-detects depth
//! from weights.
//!
//! ## Triple text encoder
//!
//! - **CLIP-L** → hidden [1, 77, 768]   + pooled [1, 768]
//! - **CLIP-G** → hidden [1, 77, 1280]  + pooled [1, 1280]
//! - **T5-XXL** → hidden [1, 256, 4096]
//!
//! Combined:
//!   `context = cat([clip_l_hidden→4096, clip_g_hidden→4096, t5_hidden], dim=1)`
//!   `pooled  = cat([clip_l_pooled, clip_g_pooled], dim=1)` → [1, 2048]
//!
//! Encoders are loaded **sequentially** and each is dropped before the next
//! loads. Peak VRAM is one encoder at a time. After all three, only the
//! cond/uncond context + pooled tensors remain (small).
//!
//! ## CFG
//!
//! Real two-pass (cond + uncond). `pred = uncond + cfg * (cond - uncond)`.
//! Default cfg=4.5 from BFL/SAI reference.
//!
//! ### Intentional deviation from `sd3_medium_infer.rs`: negative prompt
//!
//! The reference bin hardcodes `let empty = ""` and uses it for the uncond
//! branch (sd3_medium_infer.rs line 135). This worker instead passes
//! `job.negative` through the triple-encoder so a UI user can type a real
//! negative prompt. This is a **deliberate UX improvement, not a bug**: SD3
//! tuning advice that assumed empty-string uncond won't translate 1:1 to UI
//! runs where the user enters text in the Negative field. If the Negative
//! field is empty, the worker encodes an empty string — matching the bin
//! exactly. AGENT-APPROVED behavior delta.
//!
//! ## Schedule
//!
//! Rectified flow: linear sigmas in [1, 0], then SD3-style shift transform
//! with `shift=3.0`. Same `build_schedule` as `sd3_medium_infer.rs`.
//!
//! ## VAE
//!
//! 16-channel SD3 VAE, embedded in the SD3 .safetensors blob (no separate
//! file). Decoded via `LdmVAEDecoder::from_safetensors(MODEL_PATH, ...)`
//! with scale=1.5305, shift=0.0609. 8× downscale.
//!
//! ## Latent
//!
//! NCHW [1, 16, H/8, W/8]. SD3 patchifies internally — no pack/unpack at
//! the worker level.
//!
//! ## Velocity
//!
//! Model returns velocity. Use `klein_sampling::euler_denoise` which does
//! `x_next = x + dt * v` per step. Closure passes `t * 1000.0` to the model
//! since SD3's MMDiT timestep_embed expects [0, 1000].
//!
//! ## VRAM budget on a 24 GB card
//!
//! Sequential encoder loads keep peak under 10 GB during text encode.
//! - CLIP-L (load → encode → drop):       ~0.15 GB peak
//! - CLIP-G (load → encode → drop):       ~1.4 GB peak
//! - T5-XXL (load → encode → drop):       ~9.4 GB peak
//! - SD3.5 Medium DiT (resident BF16):    ~5 GB
//! - VAE decoder (load on demand):        ~0.3 GB
//! - Activations (1024² latent):          ~2-3 GB
//!
//! Comfortable on a 24 GB card. Medium fits resident; we drop the DiT
//! before VAE decode anyway (cuDNN conv workspace).
//!
//! ## Caching across jobs
//!
//! Nothing is kept resident. Encoders, DiT, and VAE are all loaded per-job
//! and dropped. AGENT-DEFAULT: pay the reload tax (~25-30s warm) for safe
//! VRAM rather than try to keep the DiT or T5 across jobs. The Medium
//! pipeline's bottleneck is denoise, not loads.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::sd3_mmdit::{load_sd3_all, SD3MMDiT};
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors sd3_medium_infer.rs).
//
// AGENT-DEFAULT: hardcoded constants. Same rationale as flux.rs / klein.rs.
// ===========================================================================

const CLIP_L_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_L_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const CLIP_G_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors";
const CLIP_G_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json";

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

/// SD3.5 Medium consolidated checkpoint — DiT + VAE in one blob.
const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors";

const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 256;

// SD3 VAE — 16-channel, 8× conv stride.
const VAE_IN_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;
const VAE_SHIFT: f32 = 0.0609;

/// Default CFG. SAI reference uses 4.5 for SD 3.5 Medium.
const DEFAULT_CFG: f32 = 4.5;

/// Default step count. SAI reference is 28.
const DEFAULT_STEPS: u32 = 28;

/// Rectified-flow shift used by SD3.
const SCHEDULE_SHIFT: f32 = 3.0;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local SD 3.5 resources. Nothing is kept resident — all heavy
/// allocations happen in `run_inner_body` and are dropped before exit.
pub struct Sd3State {
    pub device: Arc<CudaDevice>,
}

impl Sd3State {
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
    state: &mut Sd3State,
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
            log::warn!("SD3.5 Medium job {} failed: {msg}", job.id);
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
    state: &mut Sd3State,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap so we ALWAYS flush the pool on every exit path. Mirrors
    // the comment in flux.rs / klein.rs: an early Cancelled return otherwise
    // skips the scattered `clear_pool_cache` calls and pooled buffers race
    // the next job's encoder load. SD3 isn't as tight as Klein 9B but the
    // pattern is the same.
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
    state: &mut Sd3State,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // Resolve the DiT/VAE checkpoint path ONCE up-front: `job.path` override
    // from the Base ComboBox wins, else the hardcoded MODEL_PATH default.
    // Bind the owned storage here so every downstream borrow (&str) outlives
    // its usage site.
    let model_path_owned: String = job
        .path
        .clone()
        .unwrap_or_else(|| MODEL_PATH.to_string());
    let model_path: &str = model_path_owned.as_str();

    // P1 VAE guard — fire BEFORE the 28-step denoise so the user learns
    // about the GGUF-VAE limitation immediately, not after ~2 minutes of
    // denoising wasted work. SD3's VAE lives inside the same safetensors
    // blob as the DiT. If the user picked a GGUF DiT, the GGUF file almost
    // certainly doesn't ship VAE weights under the expected
    // `first_stage_model.*` keys — the downstream `LdmVAEDecoder::from_safetensors`
    // call will fail. Surface the clearer error early.
    if model_path.to_ascii_lowercase().ends_with(".gguf") {
        return Err(RunError::Other(
            "SD3.5 Medium GGUF VAE decode not yet supported: GGUF DiT files \
             don't ship the embedded SD3 VAE weights. Supply a .safetensors \
             SD3 DiT for now, or wait for separate-VAE support to land."
                .to_string(),
        ));
    }

    // -------- 1. Triple text encoder pipeline (sequential load → drop) --------
    log::info!("SD3.5 Medium: encoding text (CLIP-L + CLIP-G + T5-XXL)");
    let t_enc = Instant::now();
    let (context, pooled, context_uncond, pooled_uncond) =
        encode_text_pair(&job.prompt, &job.negative, &device)?;
    log::info!(
        "SD3.5 Medium: text encode done in {:.1}s, ctx {:?}, pooled {:?}",
        t_enc.elapsed().as_secs_f32(),
        context.shape().dims(),
        pooled.shape().dims(),
    );
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);

    drain_pending(ui_rx, pending)?;

    // -------- 2. Load SD3.5 Medium DiT (resident) --------
    log::info!("SD3.5 Medium: loading DiT from {model_path}");
    if !Path::new(model_path).exists() {
        return Err(RunError::Other(format!(
            "SD3.5 Medium weights not found at {model_path}"
        )));
    }
    let t0 = Instant::now();
    // NB: the early guard above already returns for .gguf so this branch is
    // unreachable today — kept so the pre-extracted resident path stays
    // available if/when the VAE side grows a separate-file load option.
    let resident = if model_path.to_ascii_lowercase().ends_with(".gguf") {
        log::info!("SD3.5 Medium: loading GGUF from {model_path}");
        inference_flame::gguf::load_file_gguf(Path::new(model_path), device.clone())
            .map_err(|e| RunError::Other(format!("SD3 GGUF load: {e:?}")))?
    } else {
        load_sd3_all(model_path, &device)
            .map_err(|e| RunError::Other(format!("SD3 load: {e:?}")))?
    };
    let mut model = SD3MMDiT::new(model_path.to_string(), resident, device.clone());
    log::info!(
        "SD3.5 Medium: DiT loaded in {:.1}s — depth={}, hidden={}, heads={}",
        t0.elapsed().as_secs_f32(),
        model.config.depth,
        model.config.hidden_size,
        model.config.num_heads,
    );

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
        "SD3.5 Medium: latent [1, {VAE_IN_CHANNELS}, {latent_h}, {latent_w}]"
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
    // of hand-rolling the loop in sd3_medium_infer.rs. Same rationale as
    // flux.rs.
    let noise = Tensor::randn_seeded(
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?;

    let timesteps = build_schedule(steps as usize, SCHEDULE_SHIFT);
    log::info!(
        "SD3.5 Medium: schedule {} values, t[0]={:.4}, t[-1]={:.4}, shift={SCHEDULE_SHIFT}",
        timesteps.len(),
        timesteps[0],
        timesteps[steps as usize],
    );

    drain_pending(ui_rx, pending)?;

    // -------- 4. Denoise loop with real two-pass CFG --------
    // Inline Euler (we can't use the `klein_sampling::euler_denoise` helper
    // because it doesn't give per-step hooks for cancel-check + Progress
    // events). Mirrors the closure body in sd3_medium_infer.rs::main.
    let t_denoise = Instant::now();
    let mut x = noise;
    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        // Scoped block: per-step temporaries drop at `}` BEFORE the next
        // step's allocations. Same defensive pattern as flux/klein/chroma.
        let next_x = {
            // SD3 MMDiT timestep_embed expects [0, 1000]. Closure passes
            // t * 1000 — mirrors sd3_medium_infer.rs's closure.
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr * 1000.0],
                Shape::from_dims(&[1]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?;

            let pred_cond = model
                .forward(&x, &t_vec, &context, &pooled)
                .map_err(|e| RunError::Other(format!("cond forward step {step}: {e:?}")))?;
            let pred_uncond = model
                .forward(&x, &t_vec, &context_uncond, &pooled_uncond)
                .map_err(|e| {
                    RunError::Other(format!("uncond forward step {step}: {e:?}"))
                })?;

            let diff = pred_cond
                .sub(&pred_uncond)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            let pred = pred_uncond
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // Velocity Euler step: x_next = x + dt * v.
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

    // -------- 5. Drop DiT + embeddings before VAE --------
    // Mirrors sd3_medium_infer.rs:344. The 1024² VAE conv workspace can
    // OOM if the DiT + cond/uncond context tensors are still resident.
    drop(model);
    drop(context);
    drop(pooled);
    drop(context_uncond);
    drop(pooled_uncond);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("SD3.5 Medium: DiT + context dropped, pool flushed before VAE");

    // -------- 6. VAE decode (load + use + drop) --------
    // SD3's VAE lives inside the same safetensors blob as the DiT. The
    // .gguf-path early guard at the top of this function already rejected
    // that case before encode/denoise, so by the time we get here `model_path`
    // is guaranteed to be a safetensors file containing both DiT + VAE keys.
    let t_vae = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        model_path,
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
    log::info!("SD3.5 Medium: VAE decode done in {:.1}s", t_vae.elapsed().as_secs_f32());

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Triple text encoder pipeline
// ---------------------------------------------------------------------------

/// Encode prompt + negative through CLIP-L → CLIP-G → T5-XXL sequentially.
/// Each encoder is loaded inside a scope, used to encode both cond + uncond,
/// then dropped before the next loads. Mirrors `sd3_medium_infer.rs::encode_text_pair`.
fn encode_text_pair(
    prompt: &str,
    negative: &str,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor, Tensor, Tensor), RunError> {
    fn load_clip_weights(
        path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, Tensor>, RunError> {
        let raw = flame_core::serialization::load_file(Path::new(path), device)
            .map_err(|e| RunError::Other(format!("CLIP load_file: {e:?}")))?;
        let weights = raw
            .into_iter()
            .map(|(k, v)| {
                let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
                Ok::<_, flame_core::Error>((k, t))
            })
            .collect::<Result<_, _>>()
            .map_err(|e| RunError::Other(format!("CLIP cast bf16: {e:?}")))?;
        Ok(weights)
    }

    // ---- CLIP-L (cond + uncond, then drop) ----
    if !Path::new(CLIP_L_PATH).exists() {
        return Err(RunError::Other(format!(
            "CLIP-L weights not found at {CLIP_L_PATH}"
        )));
    }
    log::info!("SD3.5 Medium: CLIP-L encode");
    let t0 = Instant::now();
    let (clip_l_h, clip_l_p, clip_l_h_u, clip_l_p_u) = {
        let weights = load_clip_weights(CLIP_L_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, pc) = clip
            .encode_sd3(&tokenize_clip(prompt, CLIP_L_TOKENIZER))
            .map_err(|e| RunError::Other(format!("CLIP-L cond: {e:?}")))?;
        let (hu, pu) = clip
            .encode_sd3(&tokenize_clip(negative, CLIP_L_TOKENIZER))
            .map_err(|e| RunError::Other(format!("CLIP-L uncond: {e:?}")))?;
        (hc, pc, hu, pu)
        // clip dropped here
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "  CLIP-L: cond {:?}, uncond {:?} ({:.1}s)",
        clip_l_h.shape().dims(),
        clip_l_h_u.shape().dims(),
        t0.elapsed().as_secs_f32()
    );

    // ---- CLIP-G (cond + uncond, then drop) ----
    if !Path::new(CLIP_G_PATH).exists() {
        return Err(RunError::Other(format!(
            "CLIP-G weights not found at {CLIP_G_PATH}"
        )));
    }
    log::info!("SD3.5 Medium: CLIP-G encode");
    let t0 = Instant::now();
    let (clip_g_h, clip_g_p, clip_g_h_u, clip_g_p_u) = {
        let weights = load_clip_weights(CLIP_G_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::clip_g(), device.clone());
        let (hc, pc) = clip
            .encode_sd3(&tokenize_clip(prompt, CLIP_G_TOKENIZER))
            .map_err(|e| RunError::Other(format!("CLIP-G cond: {e:?}")))?;
        let (hu, pu) = clip
            .encode_sd3(&tokenize_clip(negative, CLIP_G_TOKENIZER))
            .map_err(|e| RunError::Other(format!("CLIP-G uncond: {e:?}")))?;
        (hc, pc, hu, pu)
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "  CLIP-G: cond {:?}, uncond {:?} ({:.1}s)",
        clip_g_h.shape().dims(),
        clip_g_h_u.shape().dims(),
        t0.elapsed().as_secs_f32()
    );

    // ---- T5-XXL (cond + uncond, then drop) — biggest VRAM win ----
    if !Path::new(T5_PATH).exists() {
        return Err(RunError::Other(format!(
            "T5 weights not found at {T5_PATH}"
        )));
    }
    log::info!("SD3.5 Medium: T5-XXL encode");
    let t0 = Instant::now();
    let (t5_h_raw, t5_h_u_raw) = {
        let mut t5 = T5Encoder::load(T5_PATH, device)
            .map_err(|e| RunError::Other(format!("T5 load: {e:?}")))?;
        let hc = t5
            .encode(&tokenize_t5(prompt))
            .map_err(|e| RunError::Other(format!("T5 cond: {e:?}")))?;
        let hu = t5
            .encode(&tokenize_t5(negative))
            .map_err(|e| RunError::Other(format!("T5 uncond: {e:?}")))?;
        (hc, hu)
        // t5 dropped here — ~9.4 GB freed
    };
    // T5 encoder pads to its internal max (512); narrow to 256 for SD3.
    let t5_dim = t5_h_raw.shape().dims()[1].min(T5_SEQ_LEN);
    let t5_h = t5_h_raw
        .narrow(1, 0, t5_dim)
        .map_err(|e| RunError::Other(format!("T5 narrow cond: {e:?}")))?;
    let t5_h_u = t5_h_u_raw
        .narrow(1, 0, t5_h_u_raw.shape().dims()[1].min(T5_SEQ_LEN))
        .map_err(|e| RunError::Other(format!("T5 narrow uncond: {e:?}")))?;
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "  T5-XXL: cond {:?}, uncond {:?} ({:.1}s)",
        t5_h.shape().dims(),
        t5_h_u.shape().dims(),
        t0.elapsed().as_secs_f32()
    );

    // ---- Combine: pad CLIP hiddens to 4096 then cat along seq dim ----
    let cl_pad = zero_pad_last_dim(&clip_l_h, 4096)?;
    let cg_pad = zero_pad_last_dim(&clip_g_h, 4096)?;
    let context = Tensor::cat(&[&cl_pad, &cg_pad, &t5_h], 1)
        .map_err(|e| RunError::Other(format!("cat context cond: {e:?}")))?;
    let pooled = Tensor::cat(&[&clip_l_p, &clip_g_p], 1)
        .map_err(|e| RunError::Other(format!("cat pooled cond: {e:?}")))?;

    let cl_pad_u = zero_pad_last_dim(&clip_l_h_u, 4096)?;
    let cg_pad_u = zero_pad_last_dim(&clip_g_h_u, 4096)?;
    let context_u = Tensor::cat(&[&cl_pad_u, &cg_pad_u, &t5_h_u], 1)
        .map_err(|e| RunError::Other(format!("cat context uncond: {e:?}")))?;
    let pooled_u = Tensor::cat(&[&clip_l_p_u, &clip_g_p_u], 1)
        .map_err(|e| RunError::Other(format!("cat pooled uncond: {e:?}")))?;

    Ok((context, pooled, context_u, pooled_u))
}

/// Zero-pad the last dimension of a 3D tensor: [B, N, C] → [B, N, target_dim].
fn zero_pad_last_dim(x: &Tensor, target_dim: usize) -> Result<Tensor, RunError> {
    let dims = x.shape().dims();
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    if c >= target_dim {
        return Ok(x.clone());
    }
    let pad = Tensor::zeros_dtype(
        Shape::from_dims(&[b, n, target_dim - c]),
        DType::BF16,
        x.device().clone(),
    )
    .map_err(|e| RunError::Other(format!("zero_pad alloc: {e:?}")))?;
    Tensor::cat(&[x, &pad], 2).map_err(|e| RunError::Other(format!("zero_pad cat: {e:?}")))
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Linear sigmas in [1, 0] then SD3-style shift transform. Mirrors
/// `sd3_medium_infer.rs::build_schedule`. Returns `num_steps + 1` values.
fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Tokenizers
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str, tokenizer_path: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(tokenizer_path) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("clip tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(CLIP_SEQ_LEN);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
        Err(e) => {
            log::warn!("SD3.5: CLIP tokenizer ({tokenizer_path}) failed: {e}; BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.push(1); // T5 EOS
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            log::warn!("SD3.5: T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            ids.resize(T5_SEQ_LEN, 0);
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux.rs / klein.rs)
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
