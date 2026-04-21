//! Klein 4B + Klein 9B real-inference path.
//!
//! Mirrors `worker/{flux,chroma,zimage}.rs` structurally — lazy load,
//! drop-to-fit VRAM dance, per-step Started/Progress/Done event emission.
//! Underlying pipelines mirror `inference-flame/src/bin/klein_infer.rs`
//! (Klein 4B) and `inference-flame/src/bin/klein9b_infer.rs` (Klein 9B).
//!
//! ## Klein architecture (Flux 2 DiT)
//!
//! Dual-stream MM-DiT:
//! - **Klein 4B**: 5 double-stream blocks + 20 single-stream blocks
//! - **Klein 9B**: 8 double-stream blocks + 24 single-stream blocks
//! Fused QKV. Model output IS velocity (no negation, no ε→x0 conversion).
//!
//! ## Pipeline (per job)
//!
//! 1. Tokenize prompt + negative w/ Klein chat template, pad to 512.
//! 2. Load Qwen3 encoder (4B for Klein4B, 8B for Klein9B), encode cond +
//!    uncond, then drop encoder + flush pool.
//! 3. Load DiT (resident `KleinTransformer`; for Klein 9B fall back to
//!    `KleinOffloaded` if all-on-GPU OOMs).
//! 4. Build noise [1, 128, H/16, W/16], permute+reshape → packed
//!    [1, H*W, 128]. Build img_ids/txt_ids.
//! 5. Dynamic-mu schedule via `klein_sampling::get_schedule`.
//! 6. Real two-pass CFG denoise (Klein is NOT distilled): cond + uncond,
//!    `pred = uncond + guidance * (cond - uncond)`, Euler step
//!    `x += (t_next - t_curr) * pred`.
//! 7. Drop DiT + flush pool, unpack latent, VAE decode → ColorImage.
//!
//! ## CFG behavior
//!
//! Both variants use real two-pass CFG (cond + uncond). No distilled
//! guidance scalar fed to the DiT (unlike FLUX 1 Dev). Default 4.0 matches
//! the reference bins.
//!
//! ## VRAM budget on a 24 GB card
//!
//! ### Klein 4B
//! - Qwen3 4B encoder (resident BF16):    ~8.0 GB    ← dropped post-encode
//! - Klein 4B DiT (resident BF16):        ~7.2 GB
//! - Klein VAE decoder (resident BF16):   ~0.7 GB
//! - Activations (1024² latent):          ~2-3 GB
//! Resident peak after encoder drop: ~11 GB. Comfortable.
//!
//! ### Klein 9B
//! - Qwen3 8B encoder (sharded, BF16):    ~16 GB     ← dropped post-encode
//! - Klein 9B DiT (resident BF16):        ~17 GB     (all-on-GPU)
//!     OR Klein 9B (offloaded):           ~1-2 GB    (block streaming, ~10× slower)
//! - Klein VAE decoder (resident BF16):   ~0.7 GB
//! - Activations (1024² latent):          ~2-3 GB
//!
//! On a 24 GB card, the all-on-GPU path lands around 19-20 GB after the
//! encoder drop, which fits but not comfortably. We try resident first;
//! on `KleinTransformer::from_safetensors` failure we fall back to
//! `KleinOffloaded`. Mirrors `klein9b_infer.rs`'s try/fallback.
//!
//! ## Caching policy
//!
//! - **VAE decoder**: load once, keep forever. ~700 MB. Same VAE for both
//!   variants.
//! - **Tokenizer**: load once, keep forever (same Qwen3 tokenizer for both).
//! - **Qwen3 encoder**: dropped after every job. The *4B variant* of the
//!   encoder is used for Klein 4B, the *8B variant* for Klein 9B. We cache
//!   neither across jobs — at 8-16 GB they're the difference between
//!   "fits comfortably" and "OOMs the DiT". Reload tax ~10s (4B) / ~25s
//!   (8B sharded) on warm SSD.
//! - **DiT**: dropped after every job. Reload ~5-15s warm.
//! - **DiT variant change**: handled implicitly (DiT is dropped each job).
//!
//! ## Cancel-check policy
//!
//! Same as `worker/{zimage,flux,chroma}.rs`: between denoise steps only.
//! The multi-second encoder + DiT loads are atomic from the UI's POV.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::device::Device;
use flame_core::serialization::load_file;
use flame_core::{DType, Shape, Tensor};

use inference_flame::models::klein::{KleinOffloaded, KleinTransformer};
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::sampling::klein_sampling::get_schedule;
use inference_flame::vae::klein_vae::KleinVaeDecoder;

use super::{GenerateJob, ModelKind, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths.
//
// AGENT-DEFAULT: hardcoded constants. Same rationale as zimage/flux/chroma.
// Paths verified against `klein_infer.rs` / `klein9b_infer.rs` and ls'd on
// disk. The Qwen3 4B encoder lives under .serenity/, the 8B encoder is
// sharded under the HuggingFace cache (matches the reference bin).
// ===========================================================================

/// Klein 4B DiT — single consolidated safetensors file.
const KLEIN_4B_DIT_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-4b.safetensors";

/// Klein 9B DiT — single consolidated safetensors file.
const KLEIN_9B_DIT_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";

/// Qwen3 4B encoder (consolidated safetensors). Used for Klein 4B.
const QWEN3_4B_ENCODER_PATH: &str =
    "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors";

/// Qwen3 8B encoder directory (sharded safetensors). Used for Klein 9B.
/// Mirrors `klein9b_infer.rs`'s `ENCODER_DIR`.
const QWEN3_8B_ENCODER_DIR: &str =
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";

/// Qwen3 tokenizer.json. Same tokenizer file works for 4B and 8B (both
/// share the Qwen3 vocabulary) — taken from the 8B HF snapshot since the
/// 4B safetensors blob doesn't ship a tokenizer alongside it.
const TOKENIZER_PATH: &str =
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";

/// Klein VAE — same decoder for both 4B and 9B variants.
const KLEIN_VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";

/// Klein chat template — must match Python pipeline exactly.
/// Mirrors `klein_infer.rs::KLEIN_TEMPLATE_*`.
const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Token padding length. Matches both klein_infer.rs and klein9b_infer.rs.
const TXT_PAD_LEN: usize = 512;

/// Qwen3 PAD/EOS token id (`<|endoftext|>`). Mirrors klein*_infer.rs.
const PAD_ID: i32 = 151643;

/// Klein VAE 16× downscale (8× VAE conv stride × 2× patchify).
/// Latent geometry: [1, 128, H/16, W/16].
const VAE_DOWNSCALE: usize = 16;

/// Default CFG scale matching klein_infer.rs / klein9b_infer.rs.
const DEFAULT_CFG: f32 = 4.0;

/// Default step count matching klein_infer.rs / klein9b_infer.rs.
const DEFAULT_STEPS: u32 = 50;

// ===========================================================================
// Variants
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KleinVariant {
    /// Klein 4B — 5 double + 20 single, ~7 GB DiT, Qwen3 4B encoder.
    Klein4B,
    /// Klein 9B — 8 double + 24 single, ~17 GB DiT, Qwen3 8B encoder.
    Klein9B,
}

impl KleinVariant {
    fn from_kind(k: ModelKind) -> Option<Self> {
        match k {
            ModelKind::Klein4B => Some(Self::Klein4B),
            ModelKind::Klein9B => Some(Self::Klein9B),
            _ => None,
        }
    }

    fn dit_path(self) -> &'static str {
        match self {
            Self::Klein4B => KLEIN_4B_DIT_PATH,
            Self::Klein9B => KLEIN_9B_DIT_PATH,
        }
    }
}

// ===========================================================================
// State
// ===========================================================================

/// Polymorphic DiT — resident or block-offloaded. Klein 4B always uses
/// `OnGpu`; Klein 9B prefers `OnGpu` and falls back to `Offloaded` on
/// load-time OOM (mirrors `klein9b_infer.rs::Model`). The forward signature
/// is identical, so the dispatch is a thin wrapper.
enum DitInstance {
    OnGpu(KleinTransformer),
    Offloaded(KleinOffloaded),
}

impl DitInstance {
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        t: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> flame_core::Result<Tensor> {
        match self {
            DitInstance::OnGpu(m) => m.forward(img, txt, t, img_ids, txt_ids),
            DitInstance::Offloaded(m) => m.forward(img, txt, t, img_ids, txt_ids),
        }
    }
}

/// Worker-local Klein resources. Lives across iterations of the worker
/// thread loop; never crosses thread boundaries.
///
/// AGENT-DEFAULT: only the VAE + tokenizer are kept resident across jobs.
/// Encoder + DiT are loaded per-job and dropped to keep peak VRAM safe.
/// On a 24 GB card with Klein 9B + Qwen3 8B + VAE all resident we'd land
/// around 34 GB — guaranteed OOM. The reload tax (10-25s for encoder,
/// 5-15s for DiT) is the price of "always works".
pub struct KleinState {
    pub device: Arc<CudaDevice>,
    /// Kept resident — tokenizer is a few MB and `from_file` is non-trivial.
    pub tokenizer: Option<tokenizers::Tokenizer>,
    /// VAE decoder — kept resident (~700 MB). Same decoder for both
    /// variants (the latent format and patchify stride are identical).
    pub vae: Option<KleinVaeDecoder>,
}

impl KleinState {
    pub fn new() -> Result<Self, String> {
        let device =
            CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e:?}"))?;
        Ok(Self {
            device,
            tokenizer: None,
            vae: None,
        })
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Run one Klein job to completion (or cancellation). Mirrors
/// `mock::run_one`'s event protocol: Started → N×Progress → Done | Failed.
pub fn run(
    job: &GenerateJob,
    state: &mut KleinState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let variant = match KleinVariant::from_kind(job.model_kind) {
        Some(v) => v,
        None => {
            let _ = ev_tx.send(WorkerEvent::Failed {
                id: job.id,
                error: "internal: klein::run called with non-Klein kind".into(),
            });
            ctx.request_repaint();
            return;
        }
    };

    let steps: u32 = if job.steps == 0 { DEFAULT_STEPS } else { job.steps };
    let cfg_scale: f32 = if job.cfg <= 0.0 { DEFAULT_CFG } else { job.cfg };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: steps,
    });
    ctx.request_repaint();

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
            log::warn!("Klein {:?} job {} failed: {msg}", variant, job.id);
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
    variant: KleinVariant,
    steps: u32,
    cfg_scale: f32,
    state: &mut KleinState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Wrap the body in a closure so we can guarantee a final pool flush on
    // EVERY exit path — success, RunError::Cancelled (user clicked Stop
    // mid-job), and RunError::Other. Without this, an early Cancelled return
    // unwinds the stack and drops `dit` / `pos_hidden` / `neg_hidden`, but
    // the `clear_pool_cache` + `trim_cuda_mempool` calls scattered through
    // the happy path are skipped. The next job's Qwen3 load could then race
    // against ~17 GB of pooled DiT buffers — guaranteed OOM on a 24 GB card
    // for Klein 9B given how tight it already is. Single trailing flush
    // handles all cases. (Mirrors flux.rs / chroma.rs / zimage.rs.)
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(job, variant, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending)
    })();
    // ALWAYS flush, regardless of success/cancel/error.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_inner_body(
    job: &GenerateJob,
    variant: KleinVariant,
    steps: u32,
    cfg_scale: f32,
    state: &mut KleinState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. Tokenizer (lazy load, persistent) --------
    ensure_tokenizer(state)?;

    // -------- 2. Encode prompt + negative (Qwen3, then drop) --------
    // Mirrors klein_infer.rs lines 57-94 and klein9b_infer.rs lines 101-127:
    // the encoder is loaded inside a scope, used to encode both cond and
    // uncond, then explicitly dropped to free 8-16 GB before the DiT load.
    let (pos_hidden, neg_hidden) = encode_prompts(state, variant, &job.prompt, &job.negative)?;
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "Klein {:?}: Qwen3 encoder dropped + pool flushed",
        variant
    );

    drain_pending(ui_rx, pending)?;

    // -------- 3. Load DiT (resident, fall back to offloaded for 9B) --------
    let dit = ensure_dit(variant)?;

    drain_pending(ui_rx, pending)?;

    // -------- 4. Latent geometry + ids --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    // Klein VAE 16× downscale: 8× conv stride + 2× patchify.
    let latent_h = height / VAE_DOWNSCALE;
    let latent_w = width / VAE_DOWNSCALE;
    if latent_h == 0 || latent_w == 0 {
        return Err(RunError::Other(format!(
            "image too small: {width}x{height} (latent {latent_h}x{latent_w})"
        )));
    }
    let n_img = latent_h * latent_w;
    log::info!(
        "Klein {:?}: latent [B,C,H,W] = [1, 128, {latent_h}, {latent_w}], n_img={n_img}",
        variant,
    );

    // img_ids: [N_img, 4] where each row = [0, row, col, 0].
    // Mirrors klein_infer.rs lines 102-110.
    let mut img_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_data[idx * 4 + 1] = r as f32;
            img_data[idx * 4 + 2] = c as f32;
        }
    }
    let img_ids = Tensor::from_f32_to_bf16(
        img_data,
        Shape::from_dims(&[n_img, 4]),
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("img_ids alloc: {e:?}")))?;

    // txt_ids: [TXT_PAD_LEN, 4] all zeros (text has no spatial position).
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[TXT_PAD_LEN, 4]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("txt_ids alloc: {e:?}")))?;

    // -------- 5. Build noise + pack --------
    // Seed = job.seed; clamp negative ("random / -1") to a wall-clock seed.
    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };

    // AGENT-DEFAULT: use Tensor::randn_seeded for noise rather than
    // hand-rolling the Box-Muller loop from klein_infer.rs. randn_seeded is
    // also Box-Muller (per its docstring) so the result is statistically
    // equivalent; the byte-exact noise tensor differs from the reference bin
    // since the underlying RNG is different. That's fine for a UI worker —
    // we're not asserting parity against any baseline here.
    let noise_nchw = Tensor::randn_seeded(
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?
    .to_dtype(DType::BF16)
    .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?;

    // Pack: [1, 128, H, W] → [1, H, W, 128] → [1, H*W, 128].
    // Mirrors klein_infer.rs lines 160-162.
    let mut x = noise_nchw
        .permute(&[0, 2, 3, 1])
        .map_err(|e| RunError::Other(format!("noise permute: {e:?}")))?
        .reshape(&[1, n_img, 128])
        .map_err(|e| RunError::Other(format!("noise reshape: {e:?}")))?;

    // -------- 6. Sigma schedule (BFL dynamic mu) --------
    let timesteps = get_schedule(steps as usize, n_img);
    log::info!(
        "Klein {:?}: schedule {} values, t[0]={:.4}, t[-1]={:.4}",
        variant,
        timesteps.len(),
        timesteps[0],
        timesteps[steps as usize]
    );

    drain_pending(ui_rx, pending)?;

    // -------- 7. CFG denoise loop (2 forwards/step) --------
    // Klein is NOT distilled, so we always run real two-pass CFG. Mirrors
    // klein_infer.rs lines 177-194 and klein9b_infer.rs lines 196-206.
    let t_start = Instant::now();
    let batch = 1usize;

    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        // Scoped block: every per-step temporary drops at `}` BEFORE the
        // next iteration's allocations. Same defensive pattern as
        // chroma.rs / flux.rs.
        let next_x = {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr; batch],
                Shape::from_dims(&[batch]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?;

            let pred_cond = dit
                .forward(&x, &pos_hidden, &t_vec, &img_ids, &txt_ids)
                .map_err(|e| RunError::Other(format!("cond forward step {step}: {e:?}")))?;
            let pred_uncond = dit
                .forward(&x, &neg_hidden, &t_vec, &img_ids, &txt_ids)
                .map_err(|e| {
                    RunError::Other(format!("uncond forward step {step}: {e:?}"))
                })?;

            // CFG: pred = uncond + cfg * (cond - uncond)
            let diff = pred_cond
                .sub(&pred_uncond)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let scaled = diff
                .mul_scalar(cfg_scale)
                .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
            let pred = pred_uncond
                .add(&scaled)
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // Direct velocity Euler: x_next = x + (t_next - t_curr) * pred
            // (model output IS velocity for Klein/Flux2 — see
            //  klein_sampling.rs::euler_denoise comment).
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

    // -------- 8. Drop DiT before VAE decode --------
    // Mirrors klein_infer.rs:207 / klein9b_infer.rs:216 — and the same
    // pattern as flux.rs / chroma.rs / zimage.rs. cuDNN conv workspace at
    // the VAE's last upscale stages needs 1-3 GB; with the 9B DiT (17 GB)
    // resident on a 24 GB card, that workspace alloc is the most likely
    // OOM site. Drop before decode to keep things safe.
    drop(dit);
    drop(pos_hidden);
    drop(neg_hidden);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "Klein {:?}: DiT + Qwen3 hidden dropped, pool flushed before VAE decode",
        variant
    );

    // -------- 9. Unpack + VAE decode --------
    // Unpack: [1, H*W, 128] → [1, H, W, 128] → [1, 128, H, W].
    // Mirrors klein_infer.rs lines 211-213.
    let latents = x
        .reshape(&[1, latent_h, latent_w, 128])
        .map_err(|e| RunError::Other(format!("unpack reshape: {e:?}")))?
        .permute(&[0, 3, 1, 2])
        .map_err(|e| RunError::Other(format!("unpack permute: {e:?}")))?;
    drop(x);
    log::info!("Klein {:?}: unpacked latent {:?}", variant, latents.shape().dims());

    ensure_vae(state)?;
    let vae = state
        .vae
        .as_ref()
        .ok_or_else(|| RunError::Other("vae None after ensure_vae".into()))?;
    let rgb = vae
        .decode(&latents)
        .map_err(|e| RunError::Other(format!("vae decode: {e:?}")))?;
    drop(latents);

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// Encoder dispatch
// ---------------------------------------------------------------------------

/// Encode cond + uncond prompts through the variant-appropriate Qwen3
/// encoder, then drop the encoder. Returns (pos_hidden, neg_hidden) on the
/// device. Mirrors klein_infer.rs / klein9b_infer.rs Stage 1.
fn encode_prompts(
    state: &KleinState,
    variant: KleinVariant,
    prompt: &str,
    negative: &str,
) -> Result<(Tensor, Tensor), RunError> {
    let tokenizer = state
        .tokenizer
        .as_ref()
        .ok_or_else(|| RunError::Other("tokenizer None".into()))?;

    // Apply Klein chat template, then tokenize.
    let pos_formatted = format!("{KLEIN_TEMPLATE_PRE}{prompt}{KLEIN_TEMPLATE_POST}");
    let neg_formatted = format!("{KLEIN_TEMPLATE_PRE}{negative}{KLEIN_TEMPLATE_POST}");

    // add_special_tokens = false matches klein_infer.rs (the chat template
    // already provides the <|im_start|> / <|im_end|> wrapping).
    let pos_enc = tokenizer
        .encode(pos_formatted.as_str(), false)
        .map_err(|e| RunError::Other(format!("tokenize pos: {e}")))?;
    let neg_enc = tokenizer
        .encode(neg_formatted.as_str(), false)
        .map_err(|e| RunError::Other(format!("tokenize neg: {e}")))?;

    let mut pos_ids: Vec<i32> = pos_enc.get_ids().iter().map(|&id| id as i32).collect();
    let mut neg_ids: Vec<i32> = neg_enc.get_ids().iter().map(|&id| id as i32).collect();
    log::info!(
        "Klein {:?}: pos {} tokens, neg {} tokens (template-wrapped)",
        variant,
        pos_ids.len(),
        neg_ids.len(),
    );
    pos_ids.resize(TXT_PAD_LEN, PAD_ID);
    neg_ids.resize(TXT_PAD_LEN, PAD_ID);

    // Load weights for the variant-appropriate encoder. Klein 4B uses the
    // single-file 4B encoder; Klein 9B uses the sharded 8B encoder dir.
    let t0 = Instant::now();
    let weights = match variant {
        KleinVariant::Klein4B => {
            if !Path::new(QWEN3_4B_ENCODER_PATH).exists() {
                return Err(RunError::Other(format!(
                    "Qwen3 4B encoder weights not found at {QWEN3_4B_ENCODER_PATH}"
                )));
            }
            log::info!(
                "Klein 4B: loading Qwen3 4B encoder from {QWEN3_4B_ENCODER_PATH}"
            );
            load_file(Path::new(QWEN3_4B_ENCODER_PATH), &state.device)
                .map_err(|e| RunError::Other(format!("encoder load_file: {e:?}")))?
        }
        KleinVariant::Klein9B => {
            if !Path::new(QWEN3_8B_ENCODER_DIR).exists() {
                return Err(RunError::Other(format!(
                    "Qwen3 8B encoder dir not found at {QWEN3_8B_ENCODER_DIR}"
                )));
            }
            log::info!(
                "Klein 9B: loading Qwen3 8B encoder shards from {QWEN3_8B_ENCODER_DIR}"
            );
            load_sharded_qwen3(QWEN3_8B_ENCODER_DIR, &state.device)?
        }
    };

    let cfg = Qwen3Encoder::config_from_weights(&weights)
        .map_err(|e| RunError::Other(format!("encoder config: {e:?}")))?;
    let encoder = Qwen3Encoder::new(weights, cfg, state.device.clone());
    log::info!(
        "Klein {:?}: Qwen3 encoder ready in {:.1}s",
        variant,
        t0.elapsed().as_secs_f32()
    );

    let pos_hidden = encoder
        .encode(&pos_ids)
        .map_err(|e| RunError::Other(format!("encode pos: {e:?}")))?;
    let neg_hidden = encoder
        .encode(&neg_ids)
        .map_err(|e| RunError::Other(format!("encode neg: {e:?}")))?;
    log::info!(
        "Klein {:?}: pos hidden {:?}, neg hidden {:?}",
        variant,
        pos_hidden.shape().dims(),
        neg_hidden.shape().dims(),
    );

    drop(encoder); // explicit drop — frees ~8-16 GB
    Ok((pos_hidden, neg_hidden))
}

/// Load all `model-*.safetensors` shards in `dir` and merge into a single
/// HashMap. Mirrors `klein9b_infer.rs::load_sharded_weights`.
fn load_sharded_qwen3(
    dir: &str,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>, RunError> {
    let mut all_weights = HashMap::new();
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| RunError::Other(format!("read_dir({dir}): {e}")))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err(RunError::Other(format!(
            "no model-*.safetensors shards found in {dir}"
        )));
    }
    for (i, path) in shard_paths.iter().enumerate() {
        let t0 = Instant::now();
        let shard = load_file(path, device)
            .map_err(|e| RunError::Other(format!("load shard {:?}: {e:?}", path)))?;
        log::info!(
            "  Qwen3 shard {}/{}: {} keys ({:.1}s)",
            i + 1,
            shard_paths.len(),
            shard.len(),
            t0.elapsed().as_secs_f32()
        );
        all_weights.extend(shard);
    }
    Ok(all_weights)
}

// ---------------------------------------------------------------------------
// DiT loader (resident → offloaded fallback for 9B)
// ---------------------------------------------------------------------------

/// Load the variant-appropriate Klein DiT. For Klein 9B we try the
/// resident `KleinTransformer` first (faster forward) and fall back to
/// `KleinOffloaded` if the resident load returns an OOM-ish error. Mirrors
/// `klein9b_infer.rs::Model::OnGpu / Offloaded` selection. Klein 4B always
/// loads resident — at ~7 GB it fits comfortably even after VAE/encoder
/// considerations.
fn ensure_dit(variant: KleinVariant) -> Result<DitInstance, RunError> {
    let path = variant.dit_path();
    if !Path::new(path).exists() {
        return Err(RunError::Other(format!(
            "Klein {:?} DiT weights not found at {path}",
            variant
        )));
    }
    log::info!("Klein {:?}: loading DiT from {path}", variant);
    let t0 = Instant::now();

    let dit = match variant {
        KleinVariant::Klein4B => {
            // 4B always fits resident.
            let m = KleinTransformer::from_safetensors(path)
                .map_err(|e| RunError::Other(format!("Klein 4B load: {e:?}")))?;
            DitInstance::OnGpu(m)
        }
        KleinVariant::Klein9B => {
            // Try resident first; fall back to offloaded on failure.
            // Mirrors klein9b_infer.rs lines 135-144.
            match KleinTransformer::from_safetensors(path) {
                Ok(m) => {
                    log::info!("Klein 9B: resident load succeeded");
                    DitInstance::OnGpu(m)
                }
                Err(e) => {
                    log::warn!(
                        "Klein 9B: resident load failed ({e:?}), falling back to offloaded"
                    );
                    // Pool flush before retrying — the failed resident attempt
                    // may have left partial allocations behind.
                    flame_core::cuda_alloc_pool::clear_pool_cache();
                    flame_core::device::trim_cuda_mempool(0);
                    let m = KleinOffloaded::from_safetensors(path).map_err(|e| {
                        RunError::Other(format!(
                            "Klein 9B offloaded fallback also failed: {e:?}"
                        ))
                    })?;
                    DitInstance::Offloaded(m)
                }
            }
        }
    };
    log::info!(
        "Klein {:?}: DiT loaded in {:.1}s",
        variant,
        t0.elapsed().as_secs_f32()
    );
    Ok(dit)
}

// ---------------------------------------------------------------------------
// Lazy loaders (tokenizer + VAE — both small enough to keep resident)
// ---------------------------------------------------------------------------

fn ensure_tokenizer(state: &mut KleinState) -> Result<(), RunError> {
    if state.tokenizer.is_some() {
        return Ok(());
    }
    if !Path::new(TOKENIZER_PATH).exists() {
        return Err(RunError::Other(format!(
            "Klein tokenizer not found at {TOKENIZER_PATH}"
        )));
    }
    let tk = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| RunError::Other(format!("tokenizer load: {e}")))?;
    state.tokenizer = Some(tk);
    Ok(())
}

fn ensure_vae(state: &mut KleinState) -> Result<(), RunError> {
    if state.vae.is_some() {
        return Ok(());
    }
    if !Path::new(KLEIN_VAE_PATH).exists() {
        return Err(RunError::Other(format!(
            "Klein VAE weights not found at {KLEIN_VAE_PATH}"
        )));
    }
    log::info!("Klein: loading VAE decoder from {KLEIN_VAE_PATH}");
    let t0 = Instant::now();
    let weights = load_file(Path::new(KLEIN_VAE_PATH), &state.device)
        .map_err(|e| RunError::Other(format!("VAE load_file: {e:?}")))?;
    let vae_device = Device::from_arc(state.device.clone());
    let vae = KleinVaeDecoder::load(&weights, &vae_device)
        .map_err(|e| RunError::Other(format!("VAE build: {e:?}")))?;
    log::info!("Klein: VAE ready in {:.1}s", t0.elapsed().as_secs_f32());
    state.vae = Some(vae);
    Ok(())
}

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux.rs / chroma.rs / zimage.rs)
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
