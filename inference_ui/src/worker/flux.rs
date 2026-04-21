//! FLUX 1 Dev real-inference path.
//!
//! Mirrors `worker/zimage.rs` structurally — lazy load, drop-to-fit VRAM
//! dance, per-step Started/Progress/Done event emission. Underlying pipeline
//! mirrors `inference-flame/src/bin/flux1_infer.rs`.
//!
//! ## Pipeline
//!
//! 1. CLIP-L encode  → pooled [1, 768]
//! 2. T5-XXL encode  → hidden [1, 512, 4096]      (then dropped — biggest VRAM win)
//! 3. FLUX 1 DiT     → 20-step Euler denoise, distilled (single forward / step)
//! 4. VAE decode     → ColorImage
//!
//! ## CFG behavior
//!
//! FLUX 1 Dev is **guidance-distilled**: `guidance` is a model conditioning
//! input, not classifier-free guidance. One forward per step. The user-facing
//! `cfg` slider is reinterpreted as the distilled-guidance scalar (3.5 stock).
//! `negative` prompt is ignored — there is no uncond pass.
//!
//! ## VRAM budget on a 24 GB card
//!
//! - CLIP-L (resident BF16):                 ~0.15 GB
//! - T5-XXL (resident BF16):                 ~9.4 GB    ← dropped post-encode
//! - FLUX 1 DiT (BlockOffloader):            ~12-14 GB
//! - VAE decoder (resident BF16):            ~0.3 GB
//! - Activations (1024² latent + workspace): ~2-3 GB
//!
//! Resident peak at decode time without dropping T5: ~24 GB → OOMs the cuDNN
//! conv workspace at the last upscale stage. So we drop T5 immediately after
//! cap_feats is captured (mirrors `flux1_infer.rs:127`'s scope-drop), and
//! drop the DiT before VAE decode (mirrors `flux1_infer.rs:244` and
//! `zimage.rs:445`). CLIP-L stays resident — at 150 MB it's free.
//!
//! ## Caching across jobs
//!
//! - **CLIP-L**: load once, keep forever.
//! - **VAE decoder**: load once, keep forever.
//! - **T5-XXL**: dropped after every job. Reload cost ~10s on warm SSD.
//!   AGENT-DEFAULT: pay the reload tax for safe VRAM rather than try to keep
//!   it resident — at 9.4 GB it's the difference between "always works" and
//!   "OOMs at decode time on the 1024² path".
//! - **FLUX DiT**: dropped before VAE decode every job. Reload ~8-15s warm.
//!
//! Per-gen cost target after first run: ~30s (T5 ~10s + DiT 8-15s + denoise
//! 8-15s + VAE 3s).
//!
//! ## Cancel-check policy
//!
//! Same as `worker/zimage.rs`: between denoise steps only. The multi-second
//! T5 + DiT loads are atomic from the UI's POV. Phase 5c can add intermediate
//! "loading T5..." / "loading DiT..." events.

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
use inference_flame::models::flux1_dit::Flux1DiT;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{
    get_schedule, pack_latent, unpack_latent,
};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths.
//
// AGENT-DEFAULT: hardcoded constants. Same rationale as `zimage.rs` —
// inference-flame bins all hardcode (or take CLI args) and the inference_ui
// settings UI doesn't have a model-paths section yet. Keep them clustered
// here so the future config phase has an obvious migration target.
// ===========================================================================

const CLIP_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const DIT_PATH: &str = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/ae.safetensors";

// FLUX 1 Dev VAE knobs (mirrors flux1_infer.rs).
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 512;

/// Default distilled-guidance scalar when the user leaves CFG at the "off"
/// sentinel (≤0). 3.5 is the BFL reference value.
const DEFAULT_GUIDANCE: f32 = 3.5;

/// Default step count when the user leaves it at 0. 20 is the BFL reference.
const DEFAULT_STEPS: u32 = 20;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local FLUX resources. Lives across iterations of the worker thread
/// loop. CLIP-L and VAE are kept resident; T5 and DiT are loaded per-job and
/// dropped to keep peak VRAM under 24 GB.
pub struct FluxState {
    pub device: Arc<CudaDevice>,
    /// Loaded once on first call. ~150 MB — cheap to keep around.
    pub clip: Option<ClipEncoder>,
    /// Loaded once on first call. ~300 MB.
    pub vae: Option<LdmVAEDecoder>,
}

impl FluxState {
    /// Initialize CUDA on device 0. The expensive loads are deferred to the
    /// first job that needs each. CUDA init failure is permanent — the worker
    /// drops the kind back to Mock for the rest of the session if so (handled
    /// in `mock::run`).
    pub fn new() -> Result<Self, String> {
        let device =
            CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e:?}"))?;
        Ok(Self {
            device,
            clip: None,
            vae: None,
        })
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Run one FLUX 1 Dev job to completion (or cancellation). Mirrors
/// `zimage::run`'s event protocol: Started → N×Progress → Done | Failed.
pub fn run(
    job: &GenerateJob,
    state: &mut FluxState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let steps: u32 = if job.steps == 0 { DEFAULT_STEPS } else { job.steps };
    // Reinterpret the cfg slider as distilled-guidance (FLUX 1 Dev is NOT a
    // CFG model). 0/negative → fall back to the BFL default; otherwise use
    // the user value verbatim. AGENT-DEFAULT: silently reinterpret rather
    // than forcing the user to know the FLUX peculiarity. The label still
    // says "CFG" in the UI; if that becomes confusing a later phase can swap
    // the label per ModelKind.
    let guidance: f32 = if job.cfg <= 0.0 { DEFAULT_GUIDANCE } else { job.cfg };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: steps,
    });
    ctx.request_repaint();

    match run_inner(job, steps, guidance, state, ui_rx, ev_tx, ctx, pending) {
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
            log::warn!("FLUX 1 Dev job {} failed: {msg}", job.id);
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
    guidance: f32,
    state: &mut FluxState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Wrap the body in a closure so we can guarantee a final pool flush on
    // EVERY exit path — success, RunError::Cancelled (user clicked Stop
    // mid-job), and RunError::Other. Without this, an early Cancelled return
    // unwinds the stack and drops `dit` / `t5_hidden`, but the
    // `clear_pool_cache` + `trim_cuda_mempool` calls scattered through the
    // happy path are skipped. The next job's T5 load could then race against
    // ~12 GB of pooled buffers that haven't been released back to the CUDA
    // driver, OOMing on a 24 GB card. Single trailing flush handles all
    // cases. (Skeptic Batch A, P1 #2.)
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(job, steps, guidance, state, ui_rx, ev_tx, ctx, pending)
    })();
    // ALWAYS flush, regardless of success/cancel/error. Drops have already
    // run during stack unwind by this point; flush returns the freed pool
    // buffers to the CUDA driver so the next job sees a clean slate.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_inner_body(
    job: &GenerateJob,
    steps: u32,
    guidance: f32,
    state: &mut FluxState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. CLIP-L encode (lazy load, resident) --------
    ensure_clip(state)?;
    let clip = state
        .clip
        .as_ref()
        .ok_or_else(|| RunError::Other("clip None after ensure_clip".into()))?;
    let clip_tokens = tokenize_clip(&job.prompt);
    let (_clip_hidden, clip_pooled) = clip
        .encode(&clip_tokens)
        .map_err(|e| RunError::Other(format!("CLIP encode: {e:?}")))?;
    log::info!("FLUX: CLIP pooled {:?}", clip_pooled.shape().dims());

    drain_pending(ui_rx, pending)?;

    // -------- 2. T5-XXL encode (load → encode → drop) --------
    // Scope drop ensures BlockOffloader-backed weights are freed before the
    // DiT load. Mirrors flux1_infer.rs:127's scope-drop pattern. ~9.4 GB
    // freed, which is what makes the rest of the pipeline fit.
    let t5_hidden = {
        log::info!("FLUX: loading T5-XXL from {T5_PATH}");
        let t0 = Instant::now();
        if !Path::new(T5_PATH).exists() {
            return Err(RunError::Other(format!(
                "T5 weights not found at {T5_PATH}"
            )));
        }
        let mut t5 = T5Encoder::load(T5_PATH, &device)
            .map_err(|e| RunError::Other(format!("T5 load: {e:?}")))?;
        log::info!("FLUX: T5 loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let t5_tokens = tokenize_t5(&job.prompt);
        let h = t5
            .encode(&t5_tokens)
            .map_err(|e| RunError::Other(format!("T5 encode: {e:?}")))?;
        log::info!(
            "FLUX: T5 encode done in {:.1}s, hidden {:?}",
            t0.elapsed().as_secs_f32(),
            h.shape().dims()
        );
        h
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("FLUX: T5 dropped + pool flushed (~9.4 GB freed)");

    drain_pending(ui_rx, pending)?;

    // -------- 3. Load FLUX 1 DiT (BlockOffloader; reloaded each job) --------
    log::info!("FLUX: loading DiT from {DIT_PATH}");
    let t0 = Instant::now();
    if !Path::new(DIT_PATH).exists() {
        return Err(RunError::Other(format!(
            "FLUX DiT weights not found at {DIT_PATH}"
        )));
    }
    let mut dit = Flux1DiT::load(DIT_PATH, &device)
        .map_err(|e| RunError::Other(format!("FLUX DiT load: {e:?}")))?;
    log::info!("FLUX: DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    drain_pending(ui_rx, pending)?;

    // -------- 4. Build noise + pack --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    // FLUX VAE 8× downsample + patchify 2× = 16× effective. Mirrors
    // flux1_infer.rs:146.
    let latent_h = 2 * ((height + 15) / 16);
    let latent_w = 2 * ((width + 15) / 16);
    log::info!(
        "FLUX: latent [B,C,H,W] = [1, {AE_IN_CHANNELS}, {latent_h}, {latent_w}]"
    );

    // Seed handling: -1 sentinel = wall-clock seed. Same convention as
    // zimage::run.
    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    // AGENT-DEFAULT: use Tensor::randn_seeded for noise rather than
    // hand-rolling the Box-Muller loop from flux1_infer.rs. randn_seeded is
    // also Box-Muller (per its docstring) so the result is statistically
    // equivalent; the byte-exact noise tensor differs from the reference bin
    // since the underlying RNG is different. That's fine for a UI worker —
    // we're not asserting parity against any baseline here.
    let noise_nchw = Tensor::randn_seeded(
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?;

    let (img_packed, img_ids) = pack_latent(&noise_nchw, &device)
        .map_err(|e| RunError::Other(format!("pack_latent: {e:?}")))?;
    drop(noise_nchw);
    let n_img = img_packed.shape().dims()[1];

    // txt_ids: zeros [T5_SEQ_LEN, 3] (BF16). Mirrors flux1_infer.rs:184.
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("txt_ids alloc: {e:?}")))?;

    // -------- 5. Build sigma schedule --------
    let timesteps = get_schedule(steps as usize, n_img, 0.5, 1.15, true);
    log::info!(
        "FLUX: schedule {} steps, t[0]={:.4}, t[1]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1,
        timesteps[0],
        timesteps.get(1).copied().unwrap_or(0.0),
        timesteps[steps as usize]
    );

    drain_pending(ui_rx, pending)?;

    // -------- 6. Denoise loop --------
    // We can't use the `flux1_denoise` helper directly because it doesn't
    // give us per-step hooks for cancel-check + Progress events. Inline the
    // Euler loop here, mirroring flux1_sampling.rs::flux1_denoise but with
    // event emission woven in.
    let batch = 1usize;
    let guidance_vec = Tensor::from_vec(
        vec![guidance; batch],
        Shape::from_dims(&[batch]),
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("guidance_vec alloc: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("guidance_vec -> bf16: {e:?}")))?;

    let pos_hidden = t5_hidden;
    let mut x = img_packed;
    let t_start = Instant::now();

    for step in 0..steps as usize {
        // Cancel-check BEFORE each step.
        drain_pending(ui_rx, pending)?;

        let t_curr = timesteps[step];
        let t_prev = timesteps[step + 1];
        let dt = t_prev - t_curr;

        // Per-step temporaries scoped so they drop before the next step's
        // allocations. Same defensive pattern as chroma_infer.rs:198.
        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![t_curr; batch],
                Shape::from_dims(&[batch]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?
            .to_dtype(DType::BF16)
            .map_err(|e| RunError::Other(format!("t_vec -> bf16: {e:?}")))?;

            let pred = dit
                .forward(
                    &x,
                    &pos_hidden,
                    &t_vec,
                    &img_ids,
                    &txt_ids,
                    Some(&guidance_vec),
                    Some(&clip_pooled),
                )
                .map_err(|e| RunError::Other(format!("dit forward step {step}: {e:?}")))?;
            // Euler: x_next = x + dt * pred  (dt is t_prev - t_curr, negative
            // for the standard descending schedule).
            let scaled = pred
                .mul_scalar(dt)
                .map_err(|e| RunError::Other(format!("mul_scalar: {e:?}")))?;
            x.add(&scaled)
                .map_err(|e| RunError::Other(format!("euler add: {e:?}")))?
        };
        x = next_x;

        let step1 = (step + 1) as u32;
        let elapsed = t_start.elapsed().as_secs_f32();
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
    // Mirrors flux1_infer.rs:244 and zimage.rs:445. cuDNN conv workspace at
    // the VAE's last upscale stage needs 1-3 GB; with the DiT BlockOffloader
    // shadow + activations resident, that workspace alloc OOMs the card.
    drop(dit);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("FLUX: DiT dropped + pool flushed before VAE decode");

    // -------- 8. Unpack + VAE decode --------
    let latent = unpack_latent(&x, height, width)
        .map_err(|e| RunError::Other(format!("unpack_latent: {e:?}")))?;
    drop(x);
    log::info!("FLUX: unpacked latent {:?}", latent.shape().dims());

    ensure_vae(state)?;
    let vae = state
        .vae
        .as_ref()
        .ok_or_else(|| RunError::Other("vae None after ensure_vae".into()))?;
    let rgb = vae
        .decode(&latent)
        .map_err(|e| RunError::Other(format!("vae decode: {e:?}")))?;
    drop(latent);

    // -------- 9. Convert to ColorImage --------
    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Lazy loaders (CLIP + VAE — both small enough to keep resident)
// ---------------------------------------------------------------------------

fn ensure_clip(state: &mut FluxState) -> Result<(), RunError> {
    if state.clip.is_some() {
        return Ok(());
    }
    if !Path::new(CLIP_PATH).exists() {
        return Err(RunError::Other(format!(
            "CLIP-L weights not found at {CLIP_PATH}"
        )));
    }
    log::info!("FLUX: loading CLIP-L from {CLIP_PATH}");
    let t0 = Instant::now();

    // Loader upcasts F16 -> F32; BF16 kernels need BF16. Same dance as
    // flux1_infer.rs:90.
    let raw = flame_core::serialization::load_file(Path::new(CLIP_PATH), &state.device)
        .map_err(|e| RunError::Other(format!("CLIP load_file: {e:?}")))?;
    let weights: HashMap<String, Tensor> = raw
        .into_iter()
        .map(|(k, v)| {
            let t = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)?
            };
            Ok::<_, flame_core::Error>((k, t))
        })
        .collect::<Result<_, _>>()
        .map_err(|e| RunError::Other(format!("CLIP cast bf16: {e:?}")))?;
    let cfg = ClipConfig::default();
    let clip = ClipEncoder::new(weights, cfg, state.device.clone());
    log::info!("FLUX: CLIP-L ready in {:.1}s", t0.elapsed().as_secs_f32());
    state.clip = Some(clip);
    Ok(())
}

fn ensure_vae(state: &mut FluxState) -> Result<(), RunError> {
    if state.vae.is_some() {
        return Ok(());
    }
    if !Path::new(VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "FLUX VAE weights not found at {VAE_PATH}"
        )));
    }
    log::info!("FLUX: loading VAE from {VAE_PATH}");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        VAE_PATH,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &state.device,
    )
    .map_err(|e| RunError::Other(format!("VAE load: {e:?}")))?;
    log::info!("FLUX: VAE ready in {:.1}s", t0.elapsed().as_secs_f32());
    state.vae = Some(vae);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tokenization helpers (mirror flux1_infer.rs)
// ---------------------------------------------------------------------------

/// CLIP-L tokenize, pad to 77 with EOS (49407). Mirrors flux1_infer.rs.
fn tokenize_clip(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(CLIP_TOKENIZER) {
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
            log::warn!("FLUX: CLIP tokenizer failed: {e}; using BOS/EOS fallback");
            let mut ids = vec![49406i32]; // BOS
            ids.push(49407);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
    }
}

/// T5-XXL tokenize, pad to 512 with 0. Mirrors flux1_infer.rs.
fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.push(1); // T5 EOS
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0); // T5 pad
            }
            ids
        }
        Err(e) => {
            log::warn!("FLUX: T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as zimage.rs — kept inline rather
// than factored into a shared module since the protocol is simple and the
// types are identical).
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

/// Convert a [1, 3, H, W] BF16 tensor in [-1, 1] to an `egui::ColorImage`.
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

