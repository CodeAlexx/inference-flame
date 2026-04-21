//! Chroma real-inference path.
//!
//! Mirrors `worker/flux.rs` structurally — Chroma is a FLUX-derivative DiT
//! with the same VAE/latent geometry but two key differences:
//!
//!   - **No CLIP, no pooled embed**: Chroma is T5-only.
//!   - **Real CFG**: Chroma is NOT guidance-distilled; we run two forward
//!     passes per step (cond + uncond) and combine them as
//!     `noise = uncond + scale * (cond - uncond)`.
//!
//! Underlying pipeline mirrors `inference-flame/src/bin/chroma_infer.rs`.
//!
//! ## VRAM budget on a 24 GB card
//!
//! - T5-XXL (resident BF16):                 ~9.4 GB  ← dropped post-encode
//! - Chroma DiT (BlockOffloader):            ~14-17 GB
//! - VAE decoder (resident BF16):            ~0.3 GB
//! - Activations (1024² latent + workspace): ~2-3 GB
//!
//! As with FLUX: drop T5 after encoding (here we encode TWICE — cond and
//! uncond — back-to-back, then drop), drop DiT before VAE decode. Same
//! flush-pool dance as `flux.rs` and `zimage.rs`.
//!
//! ## Per-gen cost target
//!
//! ~50-60s after first run: T5 ~10s + DiT ~10s + denoise ~30s (2 forwards/step
//! × 40 steps) + VAE ~3s. Slower than FLUX 1 Dev because of the doubled
//! denoise forwards. AGENT-DEFAULT: still acceptable for a UI worker — if it
//! ends up too painful in practice, a future phase can add the Chroma DiT's
//! `forward_cached` API for ~15% step-time reduction (precompute pe_cos/sin
//! once per step then call the cached forward twice).
//!
//! ## 24 GB OOM risk (P1 #1, hardware-test item)
//!
//! May OOM on 24 GB cards. The reference bin `chroma_infer.rs` ships with a
//! `CHROMA_INFER_FORCE=1` runtime gate because the single-process flow
//! (T5 → drop → DiT load → denoise → drop → VAE) lands very close to the
//! 24 GB ceiling once cuDNN conv workspace at the last VAE upscale stage is
//! factored in. We apply the same `clear_pool_cache` + `trim_cuda_mempool`
//! dance as `flux.rs` and only load the DiT after T5 has been encoded and
//! dropped, but the Chroma DiT (~14-17 GB) is significantly larger than the
//! FLUX DiT (~12-14 GB), so headroom is tighter.
//!
//! If the UI OOMs on a 24 GB card, the next-step mitigations (in order of
//! how invasive they are):
//!   1. Drop the CLIP-L analog (we don't have one — Chroma is T5-only, so
//!      this doesn't apply; noted for symmetry with FLUX).
//!   2. Defer DiT load until the T5 encode is fully drained (already done —
//!      see scope at lines ~192-218).
//!   3. Switch to a two-stage worker analogous to `chroma_encode` +
//!      `chroma_gen` bins: separate processes for encode and denoise so the
//!      OS reclaims T5 pages between stages. This is the path the reference
//!      bin recommends and is the cleanest fix.
//!   4. Wire the DiT's `forward_cached` API and amortize a per-step
//!      pe_cos/sin allocation across the two forwards.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::chroma_dit::ChromaDit;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{get_schedule, pack_latent, unpack_latent};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths.
//
// AGENT-DEFAULT: paths copied verbatim from `chroma_infer.rs`. The DiT lives
// as sharded diffusers safetensors in the HuggingFace cache (Chroma1-HD); the
// reference bin uses these exact paths and they're verified to exist on the
// dev box. Same VAE as FLUX 1 Dev — the path string differs (lives under the
// HF snapshot rather than .serenity/models/vaes/) but the contents are
// equivalent. We use the HF copy here to keep Chroma self-contained.
// ===========================================================================

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const CHROMA_DIT_SHARDS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
];

const CHROMA_VAE: &str = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors";

// Same VAE knobs as FLUX (Chroma uses the FLUX VAE).
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

const T5_SEQ_LEN: usize = 512;

/// Default CFG scale when the user leaves it at 0/negative. 4.0 matches the
/// chroma_infer.rs reference value.
const DEFAULT_CFG: f32 = 4.0;

/// Default step count. 40 matches the chroma_infer.rs reference value
/// (Chroma is slower per-step than FLUX but uses more steps for quality).
const DEFAULT_STEPS: u32 = 40;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local Chroma resources. VAE is kept resident; T5 and DiT are
/// loaded per-job and dropped to keep peak VRAM under 24 GB.
pub struct ChromaState {
    pub device: Arc<CudaDevice>,
    /// Loaded once on first call. ~300 MB.
    pub vae: Option<LdmVAEDecoder>,
}

impl ChromaState {
    pub fn new() -> Result<Self, String> {
        let device =
            CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e:?}"))?;
        Ok(Self { device, vae: None })
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

pub fn run(
    job: &GenerateJob,
    state: &mut ChromaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let steps: u32 = if job.steps == 0 { DEFAULT_STEPS } else { job.steps };
    let cfg: f32 = if job.cfg <= 0.0 { DEFAULT_CFG } else { job.cfg };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: steps,
    });
    ctx.request_repaint();

    match run_inner(job, steps, cfg, state, ui_rx, ev_tx, ctx, pending) {
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
            log::warn!("Chroma job {} failed: {msg}", job.id);
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
    state: &mut ChromaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Wrap the body in a closure so we can guarantee a final pool flush on
    // EVERY exit path — success, RunError::Cancelled (user clicked Stop
    // mid-job), and RunError::Other. Without this, an early Cancelled return
    // unwinds the stack and drops `dit` / `cond_hidden` / `uncond_hidden`,
    // but the `clear_pool_cache` + `trim_cuda_mempool` calls scattered
    // through the happy path are skipped. The next job's T5 load could then
    // race against ~14-17 GB of pooled DiT buffers that haven't been
    // released back to the CUDA driver — guaranteed OOM on a 24 GB card
    // given how tight Chroma already is. Single trailing flush handles all
    // cases. (Skeptic Batch A, P1 #2.)
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(job, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending)
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
    cfg_scale: f32,
    state: &mut ChromaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. T5-XXL encode (cond + uncond) → drop --------
    log::info!("Chroma: loading T5 from {T5_PATH}");
    if !Path::new(T5_PATH).exists() {
        return Err(RunError::Other(format!(
            "T5 weights not found at {T5_PATH}"
        )));
    }

    let (cond_hidden, uncond_hidden) = {
        let t0 = Instant::now();
        let mut t5 = T5Encoder::load(T5_PATH, &device)
            .map_err(|e| RunError::Other(format!("T5 load: {e:?}")))?;
        log::info!("Chroma: T5 loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cond_tokens = tokenize_t5(&job.prompt);
        let cond = t5
            .encode(&cond_tokens)
            .map_err(|e| RunError::Other(format!("T5 cond encode: {e:?}")))?;
        log::info!("Chroma: cond hidden {:?}", cond.shape().dims());

        // For Chroma we always run real CFG (the model is not distilled), so
        // we always need an uncond pass — even if cfg_scale is 1.0. The cost
        // of the second encode is negligible vs. the doubled denoise.
        let uncond_tokens = tokenize_t5(&job.negative);
        let uncond = t5
            .encode(&uncond_tokens)
            .map_err(|e| RunError::Other(format!("T5 uncond encode: {e:?}")))?;
        log::info!("Chroma: uncond hidden {:?}", uncond.shape().dims());

        log::info!("Chroma: T5 done in {:.1}s", t0.elapsed().as_secs_f32());
        (cond, uncond)
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Chroma: T5 dropped + pool flushed (~9.4 GB freed)");

    drain_pending(ui_rx, pending)?;

    // -------- 2. Load Chroma DiT --------
    for path in CHROMA_DIT_SHARDS {
        if !Path::new(path).exists() {
            return Err(RunError::Other(format!(
                "Chroma DiT shard not found: {path}"
            )));
        }
    }
    log::info!("Chroma: loading DiT ({} shards)", CHROMA_DIT_SHARDS.len());
    let t0 = Instant::now();
    let mut dit = ChromaDit::load(CHROMA_DIT_SHARDS, &device)
        .map_err(|e| RunError::Other(format!("Chroma DiT load: {e:?}")))?;
    log::info!("Chroma: DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    drain_pending(ui_rx, pending)?;

    // -------- 3. Build noise + pack --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    let latent_h = 2 * ((height + 15) / 16);
    let latent_w = 2 * ((width + 15) / 16);
    log::info!(
        "Chroma: latent [B,C,H,W] = [1, {AE_IN_CHANNELS}, {latent_h}, {latent_w}]"
    );

    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

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

    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("txt_ids alloc: {e:?}")))?;

    let n_img = img_packed.shape().dims()[1];
    let timesteps = get_schedule(steps as usize, n_img, 0.5, 1.15, true);
    log::info!(
        "Chroma: schedule {} steps, t[0]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1,
        timesteps[0],
        timesteps[steps as usize]
    );

    drain_pending(ui_rx, pending)?;

    // -------- 4. CFG denoise loop (2 forwards/step) --------
    let mut x = img_packed;
    let t_start = Instant::now();
    let batch = 1usize;

    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        // Scoped block: every per-step temporary drops at `}` BEFORE the next
        // iteration's allocations. Mirrors chroma_infer.rs:198 — prevents
        // VRAM accumulation across steps.
        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![t_curr; batch],
                Shape::from_dims(&[batch]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?
            .to_dtype(DType::BF16)
            .map_err(|e| RunError::Other(format!("t_vec -> bf16: {e:?}")))?;

            let cond_pred = dit
                .forward(&x, &cond_hidden, &t_vec, &img_ids, &txt_ids)
                .map_err(|e| RunError::Other(format!("cond forward step {step}: {e:?}")))?;
            let uncond_pred = dit
                .forward(&x, &uncond_hidden, &t_vec, &img_ids, &txt_ids)
                .map_err(|e| {
                    RunError::Other(format!("uncond forward step {step}: {e:?}"))
                })?;

            // CFG: noise = uncond + scale * (cond - uncond)
            let diff = cond_pred
                .sub(&uncond_pred)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            let pred = uncond_pred
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // Euler: x_next = x + dt * pred
            let step_tensor = pred
                .mul_scalar(dt)
                .map_err(|e| RunError::Other(format!("euler mul: {e:?}")))?;
            x.add(&step_tensor)
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

    // -------- 5. Drop DiT before VAE decode --------
    drop(dit);
    drop(cond_hidden);
    drop(uncond_hidden);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Chroma: DiT + T5 hidden dropped, pool flushed before VAE decode");

    // -------- 6. Unpack + VAE decode --------
    let latent = unpack_latent(&x, height, width)
        .map_err(|e| RunError::Other(format!("unpack_latent: {e:?}")))?;
    drop(x);
    log::info!("Chroma: unpacked latent {:?}", latent.shape().dims());

    ensure_vae(state)?;
    let vae = state
        .vae
        .as_ref()
        .ok_or_else(|| RunError::Other("vae None after ensure_vae".into()))?;
    let rgb = vae
        .decode(&latent)
        .map_err(|e| RunError::Other(format!("vae decode: {e:?}")))?;
    drop(latent);

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Lazy loaders
// ---------------------------------------------------------------------------

fn ensure_vae(state: &mut ChromaState) -> Result<(), RunError> {
    if state.vae.is_some() {
        return Ok(());
    }
    if !Path::new(CHROMA_VAE).exists() {
        return Err(RunError::Other(format!(
            "Chroma VAE weights not found at {CHROMA_VAE}"
        )));
    }
    log::info!("Chroma: loading VAE from {CHROMA_VAE}");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        CHROMA_VAE,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &state.device,
    )
    .map_err(|e| RunError::Other(format!("VAE load: {e:?}")))?;
    log::info!("Chroma: VAE ready in {:.1}s", t0.elapsed().as_secs_f32());
    state.vae = Some(vae);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tokenization (mirrors chroma_infer.rs)
// ---------------------------------------------------------------------------

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            log::warn!("Chroma: T5 tokenizer load failed: {e}; zero-pad fallback");
            vec![0i32; T5_SEQ_LEN]
        }
    }
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux.rs / zimage.rs)
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
