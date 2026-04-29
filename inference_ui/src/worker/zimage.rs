//! Z-Image base + turbo real-inference path.
//!
//! Loads the Qwen3 4B encoder once on first call and keeps it resident.
//! Loads the Z-Image NextDiT lazily (heavy — minutes from cold cache) and
//! caches it across calls; same for the VAE decoder. Saves ~minutes per
//! generation after the first.
//!
//! Per-step protocol matches `mock::run_one`: emit Started, then Progress
//! per step (with cancel-check between), then Done with the decoded RGB
//! image, or Failed on error.
//!
//! ## Variants
//!
//! | Variant | Default steps | CFG behavior            | Encoder use         |
//! | ------- | ------------- | ----------------------- | ------------------- |
//! | Base    | 28            | Real 2-pass, cfg ~4.0   | cond + uncond       |
//! | Turbo   | 8             | Disabled (single-pass)  | cond only           |
//!
//! User-facing steps/cfg from the params panel override the defaults; we
//! honor whatever `job.steps`/`job.cfg` is.
//!
//! ## Caching policy
//!
//! - **Encoder**: load once, keep forever. ~8 GB BF16 on GPU. Worth it: a
//!   reload is multi-second on a warm SSD cache, multi-minute on cold.
//! - **VAE decoder**: load once, keep forever. ~300 MB.
//! - **NextDiT**: load once *per variant*. If the user switches base ↔ turbo
//!   between calls, drop the old DiT before loading the new one. Both
//!   variants share encoder + VAE so those don't need to drop.
//!
//! ## Memory budget on a 24 GB card
//!
//! - Qwen3 4B encoder (resident, BF16):       ~8.0 GB
//! - Z-Image NextDiT (resident, BF16):       ~12.3 GB (base; turbo slightly less)
//! - LdmVAEDecoder (resident, BF16):          ~0.3 GB
//! - 1024² latent activations (peak step):    ~1-2 GB
//! - **Total resident peak**:                 ~22-23 GB
//!
//! That's tight on a 24 GB card. If we OOM, the next phase can:
//!   1) drop the encoder after both prompts encode (saves ~8 GB but doubles
//!      latency on the next prompt change),
//!   2) move the encoder to CPU between calls (slower upload, no OOM risk),
//!   3) use NextDiT's BlockOffloader path (one block GPU at a time, ~10×
//!      slower denoise but ~10 GB savings).
//!
//! We document the trade-off here so a future OOM has an obvious next step.
//!
//! ## Cancel-check policy
//!
//! Cancel checks happen between denoise steps only — not inside encoder
//! load, NextDiT load, or VAE decode (those are atomic from the UI's POV).
//! Documented gap: between Started and the first Progress event the UI is
//! "stuck" for the model-load duration. Phase 5c can add an intermediate
//! "loading <model>..." event to fill that gap.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::serialization::{load_file, load_file_filtered};
use flame_core::{DType, Shape, Tensor};

use inference_flame::models::qwen3_encoder::{Qwen3Config, Qwen3Encoder};
use inference_flame::models::zimage_nextdit::NextDiT;
use inference_flame::sampling::euler::euler_step;
use inference_flame::sampling::schedules::build_sigma_schedule;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

use super::{GenerateJob, ModelKind, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths.
//
// AGENT-DEFAULT: hardcoded constants here, not env vars or config UI. The
// inference-flame bins all hardcode (or take CLI args), and the inference_ui
// settings UI doesn't have a model-paths section yet. Keep these in one
// block so the future "model-paths config" phase has an obvious migration
// target. None of these are required to *exist* at compile time — we
// open-and-fail at run time with a useful error message.
// ===========================================================================

/// Z-Image base NextDiT — sharded directory (2 .safetensors files).
const ZIMAGE_BASE_DIT_DIR: &str = "/home/alex/.serenity/models/zimage_base/transformer";

/// Z-Image turbo NextDiT — single consolidated safetensors file.
const ZIMAGE_TURBO_DIT_FILE: &str =
    "/home/alex/.serenity/models/checkpoints/z_image_turbo_bf16.safetensors";

/// Qwen3 4B text encoder — single consolidated safetensors file. Same
/// encoder used by Klein and the Z-Image embedding cache script
/// (cache_zimage_embeddings_v2.py). The model_index.json shipped with
/// zimage_base also points at a Qwen3-family encoder.
const QWEN3_ENCODER_PATH: &str = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors";

/// Qwen3 tokenizer.json. Z-Image base ships its own copy; using that one
/// ensures special tokens line up with what the encoder was trained on.
const TOKENIZER_PATH: &str = "/home/alex/.serenity/models/zimage_base/tokenizer/tokenizer.json";

/// Z-Image VAE — same 16-channel LDM-style as FLUX, scale=0.3611, shift=0.1159.
const ZIMAGE_VAE_PATH: &str =
    "/home/alex/.serenity/models/zimage_base/vae/diffusion_pytorch_model.safetensors";

const VAE_IN_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 0.3611;
const VAE_SHIFT: f32 = 0.1159;
const VAE_DOWNSCALE: usize = 8;

/// Token padding length matching cache_zimage_embeddings_v2.py. After
/// encoding we strip pad tokens so cap_feats has variable length per prompt.
const PAD_TO_LEN: usize = 512;
/// Qwen3 pad token id (`<|endoftext|>` = 151643). Matches qwen3_encoder.rs's
/// hardcoded value used in `build_causal_mask`.
const PAD_ID: i32 = 151643;

/// Sigma schedule shift. Matches zimage_infer.rs default and what training
/// uses for Z-Image base/turbo.
const SIGMA_SHIFT: f32 = 3.0;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local Z-Image resources. Lives across iterations of the worker
/// thread loop; never crosses thread boundaries.
///
/// `Option<...>` everywhere so loads are lazy and per-resource. The encoder
/// and VAE survive variant switches; the DiT is dropped if the variant
/// changes.
pub struct ZImageState {
    pub device: Arc<CudaDevice>,
    /// Loaded once on first encoded prompt. Reused across all subsequent
    /// jobs regardless of variant.
    pub qwen3: Option<Qwen3Encoder>,
    /// Tokenizer kept resident — it's tiny (~few MB) and `from_file` is
    /// non-trivial.
    pub tokenizer: Option<tokenizers::Tokenizer>,
    /// Currently-loaded NextDiT and which variant it represents. When the
    /// user switches between base and turbo we drop this and reload.
    pub dit: Option<NextDiT>,
    pub dit_variant: Option<ZImageVariant>,
    /// VAE decoder. Loaded once, reused across variants.
    pub vae: Option<LdmVAEDecoder>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZImageVariant {
    /// Z-Image base — full quality, 28-step default, real CFG.
    Base,
    /// Z-Image turbo — distilled, 8-step default, no CFG.
    Turbo,
}

impl ZImageVariant {
    fn from_kind(k: ModelKind) -> Option<Self> {
        match k {
            ModelKind::ZImageBase => Some(Self::Base),
            ModelKind::ZImageTurbo => Some(Self::Turbo),
            // Non-Z-Image kinds — outer dispatch routes them elsewhere; we
            // return None as a defensive guard if they ever reach this fn.
            ModelKind::Mock
            | ModelKind::FluxDev
            | ModelKind::Chroma
            | ModelKind::Klein4B
            | ModelKind::Klein9B
            | ModelKind::Sd35
            | ModelKind::QwenImage
            | ModelKind::ErnieImage
            | ModelKind::Anima
            | ModelKind::Sdxl
            | ModelKind::Sd15
            | ModelKind::Cascade
            | ModelKind::SenseNovaU1 => None,
        }
    }

    /// Default step count if the user leaves it untouched. Job's own
    /// `steps` overrides.
    fn default_steps(self) -> u32 {
        match self {
            Self::Base => 28,
            Self::Turbo => 8,
        }
    }

    /// Default CFG scale. Job's own `cfg` overrides; turbo always treats
    /// cfg ≤ 1.0 as "off" so a forgotten 4.0 there still does the right
    /// thing in `euler_step`.
    fn default_cfg(self) -> f32 {
        match self {
            Self::Base => 4.0,
            Self::Turbo => 1.0,
        }
    }

    fn dit_path(self) -> &'static str {
        match self {
            Self::Base => ZIMAGE_BASE_DIT_DIR,
            Self::Turbo => ZIMAGE_TURBO_DIT_FILE,
        }
    }
}

impl ZImageState {
    /// Initialize CUDA on device 0. The expensive loads (encoder, DiT, VAE)
    /// are deferred to the first job that needs each. Failure here is
    /// permanent — the worker drops the kind back to Mock for the rest of
    /// the session if so.
    pub fn new() -> Result<Self, String> {
        let device = CudaDevice::new(0)
            .map_err(|e| format!("CudaDevice::new(0): {e:?}"))?;
        Ok(Self {
            device,
            qwen3: None,
            tokenizer: None,
            dit: None,
            dit_variant: None,
            vae: None,
        })
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Run one Z-Image job to completion (or cancellation). Mirrors
/// `mock::run_one`'s event protocol: Started → N×Progress → Done | Failed.
///
/// `pending` collects non-Cancel/Shutdown messages received during the
/// denoise loop so the outer worker queue can re-apply them after we
/// return — same out-param contract as `run_one`.
pub fn run(
    job: &GenerateJob,
    state: &mut ZImageState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let variant = match ZImageVariant::from_kind(job.model_kind) {
        Some(v) => v,
        None => {
            // Should never happen — outer loop only dispatches here for
            // ZImage* kinds. Defensive: emit Failed and bail.
            let _ = ev_tx.send(WorkerEvent::Failed {
                id: job.id,
                error: "internal: zimage::run called with non-Z-Image kind".into(),
            });
            ctx.request_repaint();
            return;
        }
    };

    // Resolve effective step count + cfg. Job's own values win; if the user
    // left "0 steps" (defensive only — the UI clamps to ≥1), fall back to
    // the variant default.
    let steps: u32 = if job.steps == 0 {
        variant.default_steps()
    } else {
        job.steps
    };
    let cfg_scale: f32 = if job.cfg <= 0.0 {
        variant.default_cfg()
    } else {
        job.cfg
    };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: steps,
    });
    ctx.request_repaint();

    // Heavy lifting in a separate function so we can use `?`-style errors
    // and have one place that turns them into Failed events.
    match run_inner(job, variant, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending) {
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
            log::warn!("Z-Image job {} failed: {msg}", job.id);
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
    variant: ZImageVariant,
    steps: u32,
    cfg_scale: f32,
    state: &mut ZImageState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Wrap the body in a closure so we can guarantee a final pool flush on
    // EVERY exit path — success, RunError::Cancelled (user clicked Stop
    // mid-job), and RunError::Other. Without this, an early Cancelled return
    // unwinds the stack and clears `state.dit` / drops the encoder, but the
    // `clear_pool_cache` + `trim_cuda_mempool` calls scattered through the
    // happy path are skipped. The next job could then race against ~12 GB of
    // pooled DiT buffers that haven't been released back to the CUDA
    // driver, OOMing on a 24 GB card. Single trailing flush handles all
    // cases. (Skeptic Batch A, P1 #2 — same fix as flux.rs / chroma.rs.)
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(job, variant, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending)
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
    variant: ZImageVariant,
    steps: u32,
    cfg_scale: f32,
    state: &mut ZImageState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. Encoder + tokenizer (lazy load, persistent) --------
    ensure_tokenizer(state)?;
    ensure_encoder(state)?;

    // -------- 2. Encode prompts --------
    // Always encode the conditional prompt. For Base with cfg > 1.0 we also
    // encode the negative; Turbo skips it (no second forward pass anyway).
    let cap_feats = encode_prompt(state, &job.prompt)?;
    let want_cfg = matches!(variant, ZImageVariant::Base) && cfg_scale > 1.0;
    let cap_feats_uncond = if want_cfg {
        Some(encode_prompt(state, &job.negative)?)
    } else {
        None
    };

    // Drop Qwen3 encoder to free ~8 GB before loading the DiT (12.3 GB) and
    // running VAE decode (cuDNN conv workspace is 1-2 GB). On a 24 GB card
    // all three resident OOMs the cuDNN conv at decode time. The encoder
    // reloads in ~8s on the next generation — acceptable trade for
    // bullet-proof VAE decode. Mirrors zimage_infer.rs:289 (drop model
    // before VAE, here we drop encoder before DiT — same idea, different
    // hot path because the UI keeps DiT resident across gens).
    state.qwen3 = None;
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Dropped Qwen3 encoder + flushed CUDA pool (~8 GB freed)");

    // Cancel-check between encoding and the multi-minute DiT load. The user
    // might have already given up.
    drain_pending(ui_rx, pending)?;

    // -------- 3. DiT (lazy load, swapped on variant change) --------
    // Thread the Base-ComboBox path override (None → variant.dit_path()).
    ensure_dit(state, variant, job.path.as_deref())?;

    // -------- 4. VAE decoder (lazy load, persistent) --------
    ensure_vae(state)?;

    // -------- 5. Initial latent --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    let latent_h = height / VAE_DOWNSCALE;
    let latent_w = width / VAE_DOWNSCALE;
    if latent_h == 0 || latent_w == 0 {
        return Err(RunError::Other(format!(
            "image too small: {width}×{height} (latent {latent_h}×{latent_w})"
        )));
    }

    // Seed = job.seed; clamp negative ("random / -1") to a wall-clock seed.
    let seed_u64: u64 = if job.seed < 0 {
        // -1 sentinel from the seed picker = "use clock". Mirror what
        // box_muller_noise in zimage_inpaint.rs does for explicit seeds —
        // a fresh u64 each generation.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    let x = Tensor::randn_seeded(
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?;

    // -------- 6. Sigma schedule --------
    let sigmas = build_sigma_schedule(steps as usize, SIGMA_SHIFT);

    // -------- 7. Denoise loop --------
    let dit = state
        .dit
        .as_mut()
        .ok_or_else(|| RunError::Other("dit None after ensure_dit".into()))?;

    let t_start = Instant::now();
    let mut x = x;
    for step in 0..steps as usize {
        // Cancel-check BEFORE each step. Drain non-cancel messages into
        // `pending` so the outer worker loop can re-apply them.
        drain_pending(ui_rx, pending)?;

        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];

        x = euler_step(
            dit,
            &x,
            sigma,
            sigma_next,
            &cap_feats,
            cap_feats_uncond.as_ref(),
            cfg_scale,
        )
        .map_err(|e| RunError::Other(format!("euler step {step}: {e:?}")))?;

        // Per-step Progress event with rolling ETA. Step is 1-indexed in
        // the protocol (matches mock::run_one).
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

    // -------- 8. VAE decode --------
    // Drop the DiT before VAE decode. cuDNN conv at the last upscale stage
    // ([1, H, W, 128] @ [128,128,3,3]) needs a 1-3 GB workspace at 1024².
    // With DiT (12.3 GB) + VAE (0.3 GB) + activations resident, the
    // workspace alloc OOMs on a 24 GB card. Mirrors zimage_infer.rs:289.
    // DiT will reload from disk-cached safetensors on the next gen (~8s
    // for Turbo, ~40s for Base) — acceptable trade for reliable decode.
    state.dit = None;
    state.dit_variant = None;
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("Dropped DiT + flushed CUDA pool before VAE decode");

    let vae = state
        .vae
        .as_ref()
        .ok_or_else(|| RunError::Other("vae None after ensure_vae".into()))?;
    let rgb = vae
        .decode(&x)
        .map_err(|e| RunError::Other(format!("vae decode: {e:?}")))?;

    // -------- 9. Convert to ColorImage --------
    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Lazy loaders
// ---------------------------------------------------------------------------

fn ensure_tokenizer(state: &mut ZImageState) -> Result<(), RunError> {
    if state.tokenizer.is_some() {
        return Ok(());
    }
    if !Path::new(TOKENIZER_PATH).exists() {
        return Err(RunError::Other(format!(
            "Z-Image tokenizer not found at {TOKENIZER_PATH}"
        )));
    }
    let tk = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| RunError::Other(format!("tokenizer load: {e}")))?;
    state.tokenizer = Some(tk);
    Ok(())
}

fn ensure_encoder(state: &mut ZImageState) -> Result<(), RunError> {
    if state.qwen3.is_some() {
        return Ok(());
    }
    if !Path::new(QWEN3_ENCODER_PATH).exists() {
        return Err(RunError::Other(format!(
            "Qwen3 encoder weights not found at {QWEN3_ENCODER_PATH}"
        )));
    }
    log::info!("Loading Qwen3 4B encoder from {QWEN3_ENCODER_PATH}");
    let t0 = Instant::now();
    let weights = load_file(Path::new(QWEN3_ENCODER_PATH), &state.device)
        .map_err(|e| RunError::Other(format!("encoder load_file: {e:?}")))?;
    // Auto-detect config (vocab/hidden/layers) and override extract_layers
    // for the Z-Image case (single layer 26 → [1, seq, 2560]).
    let mut cfg = Qwen3Encoder::config_from_weights(&weights)
        .map_err(|e| RunError::Other(format!("encoder config: {e:?}")))?;
    cfg.extract_layers = Qwen3Config::zimage().extract_layers;
    let enc = Qwen3Encoder::new(weights, cfg, state.device.clone());
    log::info!("Qwen3 encoder ready in {:.1}s", t0.elapsed().as_secs_f32());
    state.qwen3 = Some(enc);
    Ok(())
}

fn ensure_dit(
    state: &mut ZImageState,
    variant: ZImageVariant,
    override_path: Option<&str>,
) -> Result<(), RunError> {
    // Variant change → drop existing DiT before loading the new one.
    if state.dit_variant != Some(variant) {
        state.dit = None;
        state.dit_variant = None;
    }
    if state.dit.is_some() {
        return Ok(());
    }
    // Base-ComboBox override wins; `None` falls back to variant default.
    let path: &str = override_path.unwrap_or(variant.dit_path());
    let p = Path::new(path);
    if !p.exists() {
        return Err(RunError::Other(format!(
            "Z-Image DiT weights not found at {path}"
        )));
    }
    log::info!(
        "Loading Z-Image {:?} DiT from {path}",
        variant
    );
    let t0 = Instant::now();
    let lower = path.to_ascii_lowercase();
    let weights: HashMap<String, Tensor> = if lower.ends_with(".gguf") {
        // GGUF path: Phase 1 loader dequantizes Q-blocks → BF16, uploads, and
        // applies the standard key-name remap. NextDiT::new_resident accepts
        // the same key set as the safetensors loader, so no extra wiring.
        log::info!("Z-Image: loading GGUF from {path}");
        inference_flame::gguf::load_file_gguf(p, state.device.clone())
            .map_err(|e| RunError::Other(format!("GGUF load: {e:?}")))?
    } else if p.is_dir() {
        // Directory of sharded safetensors — iterate, sort, load, merge.
        // Same pattern as zimage_infer.rs.
        let mut weights = HashMap::new();
        let mut entries: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| RunError::Other(format!("read_dir({path}): {e}")))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension().and_then(|s| s.to_str()) == Some("safetensors")
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in &entries {
            let partial = load_file(entry.path(), &state.device)
                .map_err(|e| RunError::Other(format!("load shard {:?}: {e:?}", entry.path())))?;
            weights.extend(partial);
        }
        weights
    } else {
        load_file_filtered(path, &state.device, |_| true)
            .map_err(|e| RunError::Other(format!("dit load_file: {e:?}")))?
    };
    log::info!(
        "Z-Image DiT loaded ({} tensors) in {:.1}s — building model...",
        weights.len(),
        t0.elapsed().as_secs_f32()
    );
    let dit = NextDiT::new_resident(weights, state.device.clone());
    state.dit = Some(dit);
    state.dit_variant = Some(variant);
    log::info!("Z-Image DiT resident in {:.1}s total", t0.elapsed().as_secs_f32());
    Ok(())
}

fn ensure_vae(state: &mut ZImageState) -> Result<(), RunError> {
    if state.vae.is_some() {
        return Ok(());
    }
    if !Path::new(ZIMAGE_VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "Z-Image VAE weights not found at {ZIMAGE_VAE_PATH}"
        )));
    }
    log::info!("Loading Z-Image VAE decoder from {ZIMAGE_VAE_PATH}");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        ZIMAGE_VAE_PATH,
        VAE_IN_CHANNELS,
        VAE_SCALE,
        VAE_SHIFT,
        &state.device,
    )
    .map_err(|e| RunError::Other(format!("vae load: {e:?}")))?;
    log::info!("Z-Image VAE ready in {:.1}s", t0.elapsed().as_secs_f32());
    state.vae = Some(vae);
    Ok(())
}

// ---------------------------------------------------------------------------
// Encode + cancel + image conversion helpers
// ---------------------------------------------------------------------------

/// Tokenize, pad to 512, encode through Qwen3, strip pad tokens. Mirrors
/// what cache_zimage_embeddings_v2.py does so the resulting cap_feats is
/// trained-input-shaped (variable length, no pad junk).
fn encode_prompt(state: &mut ZImageState, prompt: &str) -> Result<Tensor, RunError> {
    let tokenizer = state
        .tokenizer
        .as_ref()
        .ok_or_else(|| RunError::Other("tokenizer None".into()))?;
    let encoder = state
        .qwen3
        .as_ref()
        .ok_or_else(|| RunError::Other("qwen3 None".into()))?;

    // add_special_tokens = true matches the Python script: it lets the
    // tokenizer prepend BOS / wrap the prompt per the model's training.
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| RunError::Other(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();

    // Pad to PAD_TO_LEN. Truncate prompts longer than the pad limit — the
    // encoder's causal_mask + RoPE tables are sized to seq_len, so we have
    // to enforce it. (Real-world prompts almost never go past 512 tokens.)
    let real_len = ids.len().min(PAD_TO_LEN);
    let mut padded: Vec<i32> = ids
        .iter()
        .take(real_len)
        .map(|&id| id as i32)
        .collect();
    padded.resize(PAD_TO_LEN, PAD_ID);

    // Encode → [1, 512, 2560].
    let full = encoder
        .encode(&padded)
        .map_err(|e| RunError::Other(format!("encode: {e:?}")))?;

    // Strip pad tokens: narrow seq dim to real_len. Mirrors
    //   pos_full[0][mask].unsqueeze(0)  (Python)
    // but we don't need a boolean mask because the pad tokens are always at
    // the *end* of the sequence after our explicit pad-fill.
    if real_len == PAD_TO_LEN {
        // Empty negative prompt or full-length prompt — no strip needed.
        return Ok(full);
    }
    // narrow(dim, start, len) → [1, real_len, 2560].
    let dims = full.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(RunError::Other(format!(
            "encoder returned unexpected shape {dims:?}, want [1, seq, hidden]"
        )));
    }
    let stripped = full
        .narrow(1, 0, real_len.max(1))
        .map_err(|e| RunError::Other(format!("narrow seq dim: {e:?}")))?;
    Ok(stripped)
}

/// Drain pending UI messages between denoise steps. Cancel/Shutdown returns
/// `Cancelled`; everything else gets queued for the outer loop to handle.
fn drain_pending(
    ui_rx: &Receiver<UiMsg>,
    pending: &mut VecDeque<UiMsg>,
) -> Result<(), RunError> {
    while let Ok(msg) = ui_rx.try_recv() {
        match msg {
            UiMsg::Cancel => return Err(RunError::Cancelled),
            UiMsg::Shutdown => {
                // Re-route the Shutdown so the outer loop also exits after
                // we return. Same pattern as mock::run_one.
                pending.push_back(UiMsg::Shutdown);
                return Err(RunError::Cancelled);
            }
            other => pending.push_back(other),
        }
    }
    Ok(())
}

/// Convert a [1, 3, H, W] BF16 tensor in [-1, 1] to an `egui::ColorImage`.
/// Mirrors `zimage_infer.rs::save PNG` but produces RGBA pixels for egui
/// instead of writing a PNG.
fn decoded_to_color_image(rgb: &Tensor) -> Result<ColorImage, flame_core::Error> {
    let dims = rgb.shape().dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "VAE decode shape {dims:?} != [1, 3, H, W]"
        )));
    }
    let h = dims[2];
    let w = dims[3];
    // F32 host copy: [3*H*W] in CHW order.
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
