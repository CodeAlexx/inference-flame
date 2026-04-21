//! SD 1.5 real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3,qwenimage,ernie,anima,sdxl}.rs` —
//! lazy load, drop-to-fit VRAM dance, per-step Started/Progress/Done event
//! emission. Underlying pipeline mirrors
//! `inference-flame/src/bin/sd15_infer.rs`.
//!
//! ## Architecture
//!
//! SD 1.5 UNet — UNet2DConditionModel (~860M params), LDM-style. Loaded
//! all-on-GPU via `SD15UNet::from_safetensors_all_gpu(...)`. The loader
//! handles the diffusers→LDM key remap internally.
//!
//! ## Text encoder — CLIP-L only, encoded inline
//!
//! UNLIKE the qwenimage / anima / sdxl workers which read pre-computed
//! cached embeddings, SD 1.5 uses ONLY CLIP-L (no CLIP-G, no T5) — a
//! single 250 MB encoder we can spin up, encode cond+uncond, and drop in
//! about a second. So this worker matches `sd15_infer.rs` exactly: encode
//! in-process, no embeddings file, the typed prompt is the actual prompt.
//!
//! This is the ONE worker in this batch where `job.prompt` / `job.negative`
//! actually DO get used. AGENT-DEFAULT.
//!
//! ## CFG
//!
//! Real two-pass (cond + uncond). `pred = uncond + cfg * (cond - uncond)`.
//! Default cfg=7.5 matches the SD 1.5 reference.
//!
//! ## Schedule — VE (variance-exploding), eps-prediction
//!
//! SAME scaled-linear β schedule as SDXL (β_start=0.00085, β_end=0.012,
//! 1000 train steps). Build sigmas via `((1 - alpha) / alpha).sqrt()` from
//! `alphas_cumprod`. Leading timestep spacing with `steps_offset=1`.
//! Mirrors `sd15_infer.rs::build_sd15_schedule`.
//!
//! ## Euler step formula (VE eps-prediction)
//!
//! Same as SDXL — eps prediction, NOT velocity. Per `sd15_infer.rs:218-247`:
//! ```text
//!   c_in = 1 / sqrt(sigma^2 + 1)
//!   x_in = x * c_in                              # input scaling
//!   eps = unet(x_in, t, context)                 # ε prediction (BF16 ok)
//!   eps_cfg = uncond + cfg * (cond - uncond)     # in FP32
//!   dt = sigma_next - sigma                      # negative
//!   x_next = x + dt * eps_cfg                    # Euler step
//! ```
//!
//! ## VAE
//!
//! 4-channel SD VAE family. scale=0.18215 (SD 1.5 specific, NOT SDXL's
//! 0.13025). shift=0.0. 8× downscale.
//!
//! ### SD 1.5 attention key naming workaround
//!
//! The shipped SD 1.5 VAE safetensors uses pre-0.14 diffusers attention
//! key names (`query` / `key` / `value` / `proj_attn`), while our LDM remap
//! in `ldm_decoder.rs` only handles modern `to_q` / `to_k` / `to_v` /
//! `to_out.0`. Pre-rename legacy → modern, write a temp safetensors, then
//! call `LdmVAEDecoder::from_safetensors` on that. Mirrors
//! `sd15_infer.rs:262-298` exactly.
//!
//! ## Latent
//!
//! NCHW [1, 4, H/8, W/8]. Default 512×512.
//!
//! ## VRAM budget on a 24 GB card
//!
//! - CLIP-L (load → encode cond+uncond → drop): ~250 MB peak
//! - SD 1.5 UNet (resident BF16): ~1.6 GB
//! - VAE decoder (load on demand): ~0.3 GB
//! - Activations (512² latent): ~0.5 GB
//!
//! Trivial. We drop the UNet before VAE decode anyway for the cuDNN
//! workspace safety margin — it's the same defensive pattern as the
//! larger workers, not a VRAM necessity here.

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
use inference_flame::models::sd15_unet::SD15UNet;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors sd15_infer.rs).
// ===========================================================================

const SNAPSHOT: &str =
    "/home/alex/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14";

fn unet_path() -> String {
    format!("{SNAPSHOT}/unet/diffusion_pytorch_model.safetensors")
}
fn vae_path() -> String {
    format!("{SNAPSHOT}/vae/diffusion_pytorch_model.safetensors")
}
fn clip_l_path() -> String {
    format!("{SNAPSHOT}/text_encoder/model.safetensors")
}
/// Reuse the FLUX-shipped CLIP-L tokenizer JSON — CLIP-L BPE is identical
/// for SD 1.5. Mirrors sd15_infer.rs:32-33.
const CLIP_L_TOKENIZER: &str =
    "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";

/// Temp scratch file the VAE attention-key rename workaround writes to.
/// Re-loaded by `LdmVAEDecoder::from_safetensors`. Lives in `/tmp` because
/// it's deterministic given the source VAE and we don't need to keep it.
///
/// Per-process cache: written ONCE on the first job, reused thereafter.
/// The remap output is purely a function of the source VAE safetensors
/// (whose path is itself a constant), so the file's contents never change
/// across jobs. Skipping the rewrite saves ~167 MB of disk write per job
/// and avoids the tmpfs RAM growth flagged in SKEPTIC_BATCH_D P2.1.
/// First-run check is `Path::exists()` — cheap and correct for the single-
/// process case. (Multi-process races on this path are still possible but
/// are out of scope; would need a per-PID suffix or `tempfile` to fix.)
const VAE_TMP_PATH: &str = "/tmp/inference_ui_sd15_vae_remapped.safetensors";

const CLIP_SEQ_LEN: usize = 77;

// SD 1.5 VAE — 4-channel, 8x conv stride. scale=0.18215 (NOT SDXL's 0.13025).
const VAE_IN_CHANNELS: usize = 4;
const VAE_SCALE: f32 = 0.18215;
const VAE_SHIFT: f32 = 0.0;

/// Default CFG. SD 1.5 reference is 7.5.
const DEFAULT_CFG: f32 = 7.5;

/// Default step count. SD 1.5 reference is 30.
const DEFAULT_STEPS: u32 = 30;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local SD 1.5 resources. Nothing kept resident — encoder, UNet,
/// VAE all loaded per-job. SD 1.5 is small enough that the load tax is
/// trivial (~3-5s warm SSD); keeping anything resident isn't worth the
/// complexity.
pub struct Sd15State {
    pub device: Arc<CudaDevice>,
}

impl Sd15State {
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
    state: &mut Sd15State,
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
            log::warn!("SD 1.5 job {} failed: {msg}", job.id);
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
    state: &mut Sd15State,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap so we ALWAYS flush the pool on every exit path. SD 1.5
    // doesn't really need it (small VRAM footprint) but the pattern is
    // identical to sister workers for consistency.
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
    state: &mut Sd15State,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // -------- 1. CLIP-L encode (cond + uncond, then drop) --------
    if !Path::new(&clip_l_path()).exists() {
        return Err(RunError::Other(format!(
            "CLIP-L weights not found at {}",
            clip_l_path()
        )));
    }
    log::info!("SD 1.5: CLIP-L encode (prompt + negative)");
    let t_enc = Instant::now();
    let (context, context_uncond) = {
        let weights = load_clip_weights(&clip_l_path(), &device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, _pooled) = clip
            .encode(&tokenize_clip(&job.prompt))
            .map_err(|e| RunError::Other(format!("CLIP-L cond: {e:?}")))?;
        let (hu, _pooled_u) = clip
            .encode(&tokenize_clip(&job.negative))
            .map_err(|e| RunError::Other(format!("CLIP-L uncond: {e:?}")))?;
        (hc, hu)
        // clip dropped here — ~250 MB freed
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "SD 1.5: CLIP-L done in {:.1}s — cond {:?}, uncond {:?}",
        t_enc.elapsed().as_secs_f32(),
        context.shape().dims(),
        context_uncond.shape().dims(),
    );

    drain_pending(ui_rx, pending)?;

    // -------- 2. Load SD 1.5 UNet --------
    if !Path::new(&unet_path()).exists() {
        return Err(RunError::Other(format!(
            "SD 1.5 UNet not found at {}",
            unet_path()
        )));
    }
    log::info!("SD 1.5: loading UNet from {}", unet_path());
    let t0 = Instant::now();
    let mut model = SD15UNet::from_safetensors_all_gpu(&unet_path(), &device)
        .map_err(|e| RunError::Other(format!("UNet load: {e:?}")))?;
    log::info!("SD 1.5: UNet loaded in {:.1}s", t0.elapsed().as_secs_f32());

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
        "SD 1.5: latent [1, {VAE_IN_CHANNELS}, {latent_h}, {latent_w}] (8x downscale of {width}x{height})"
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
    // of hand-rolling the loop in sd15_infer.rs. Same rationale as sdxl.rs.
    let noise = Tensor::randn_seeded(
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        0.0,
        1.0,
        seed_u64,
        device.clone(),
    )
    .map_err(|e| RunError::Other(format!("noise gen: {e:?}")))?;

    let (sigmas, timesteps) = build_sd15_schedule(steps as usize);
    log::info!(
        "SD 1.5: schedule sigma_max={:.4} sigma_min={:.6} t_max={:.0} t_min={:.0}",
        sigmas[0],
        sigmas[steps as usize - 1],
        timesteps[0],
        timesteps[steps as usize - 1],
    );

    // x = noise * init_noise_sigma where init_noise_sigma = sqrt(sigma_max^2 + 1).
    // Mirrors sd15_infer.rs:212-213.
    let init_sigma = (sigmas[0] * sigmas[0] + 1.0).sqrt();
    let x_bf16 = noise
        .to_dtype(DType::BF16)
        .map_err(|e| RunError::Other(format!("noise -> bf16: {e:?}")))?
        .mul_scalar(init_sigma)
        .map_err(|e| RunError::Other(format!("init_sigma scale: {e:?}")))?;

    drain_pending(ui_rx, pending)?;

    // -------- 4. Denoise loop with real two-pass CFG --------
    // Mirrors sd15_infer.rs:215-247. FP32 denoising state, BF16 UNet input.
    let t_denoise = Instant::now();
    let mut x_f32 = x_bf16
        .to_dtype(DType::F32)
        .map_err(|e| RunError::Other(format!("x -> f32: {e:?}")))?;
    drop(x_bf16);

    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];

        let next_x = {
            let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
            let x_in = x_f32
                .mul_scalar(c_in)
                .map_err(|e| RunError::Other(format!("c_in scale: {e:?}")))?
                .to_dtype(DType::BF16)
                .map_err(|e| RunError::Other(format!("x_in -> bf16: {e:?}")))?;

            let timestep = Tensor::from_f32_to_bf16(
                vec![timesteps[step]],
                Shape::from_dims(&[1]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("timestep alloc: {e:?}")))?;

            let pred_cond = model
                .forward(&x_in, &timestep, &context)
                .map_err(|e| RunError::Other(format!("cond forward step {step}: {e:?}")))?;
            let pred_uncond = model
                .forward(&x_in, &timestep, &context_uncond)
                .map_err(|e| RunError::Other(format!("uncond forward step {step}: {e:?}")))?;

            // CFG combine in FP32: pred = uncond + cfg * (cond - uncond).
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

            // VE eps Euler step in FP32: x_next = x + (sigma_next - sigma) * eps.
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

    let x = x_f32
        .to_dtype(DType::BF16)
        .map_err(|e| RunError::Other(format!("x -> bf16 for VAE: {e:?}")))?;

    // -------- 5. Drop UNet + embeddings before VAE --------
    drop(model);
    drop(context);
    drop(context_uncond);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!("SD 1.5: UNet + context dropped before VAE");

    // -------- 6. VAE decode (load + use + drop) --------
    if !Path::new(&vae_path()).exists() {
        return Err(RunError::Other(format!(
            "SD 1.5 VAE not found at {}",
            vae_path()
        )));
    }
    let t_vae = Instant::now();
    let vae = load_sd15_vae_with_attention_rename(&vae_path(), &device)?;
    let rgb = vae
        .decode(&x)
        .map_err(|e| RunError::Other(format!("VAE decode: {e:?}")))?;
    drop(x);
    drop(vae);
    log::info!("SD 1.5: VAE decode done in {:.1}s", t_vae.elapsed().as_secs_f32());

    decoded_to_color_image(&rgb).map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// SD 1.5 VAE — pre-rename legacy attention keys, then load
// ---------------------------------------------------------------------------

/// SD 1.5's VAE safetensors uses pre-0.14 diffusers attention key names
/// (`query`/`key`/`value`/`proj_attn`). The LDM remap inside
/// `LdmVAEDecoder::from_safetensors` only handles modern
/// (`to_q`/`to_k`/`to_v`/`to_out.0`). Workaround mirrors
/// `sd15_infer.rs:262-298`: load → filter to decoder + post_quant_conv keys
/// → strip `first_stage_model.` prefix → rename legacy attention keys →
/// write temp safetensors → re-load via `from_safetensors` so its standard
/// remap covers the rest.
fn load_sd15_vae_with_attention_rename(
    src: &str,
    device: &Arc<CudaDevice>,
) -> Result<LdmVAEDecoder, RunError> {
    // Per-process filesystem cache. The remap output is a pure function of
    // the source VAE safetensors path (a constant) — so once the file
    // exists, every subsequent job can reuse it. Saves ~167 MB of disk
    // write and the ~1-2 s safetensors serialization tax per job, and
    // avoids the tmpfs RAM growth from rewriting the same bytes (see
    // SKEPTIC_BATCH_D P2.1).
    if !Path::new(VAE_TMP_PATH).exists() {
        log::info!(
            "SD 1.5: building VAE remap cache at {VAE_TMP_PATH} (one-time per process)"
        );
        let raw = flame_core::serialization::load_file(Path::new(src), device)
            .map_err(|e| RunError::Other(format!("VAE raw load: {e:?}")))?;

        let mut w: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            // Only carry decoder + post_quant_conv keys through.
            let is_decoder = k.starts_with("decoder.")
                || k.starts_with("first_stage_model.decoder.")
                || k == "post_quant_conv.weight"
                || k == "post_quant_conv.bias"
                || k == "first_stage_model.post_quant_conv.weight"
                || k == "first_stage_model.post_quant_conv.bias";
            if !is_decoder {
                continue;
            }
            // Strip `first_stage_model.` prefix if present.
            let k = k.strip_prefix("first_stage_model.").unwrap_or(&k).to_string();
            // Legacy → modern attention key renames. Mirrors sd15_infer.rs:281-284.
            let k = k
                .replace("attentions.0.query.", "attentions.0.to_q.")
                .replace("attentions.0.key.", "attentions.0.to_k.")
                .replace("attentions.0.value.", "attentions.0.to_v.")
                .replace("attentions.0.proj_attn.", "attentions.0.to_out.0.");
            let v_bf16 = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)
                    .map_err(|e| RunError::Other(format!("VAE -> bf16: {e:?}")))?
            };
            w.insert(k, v_bf16);
        }

        flame_core::serialization::save_file(&w, Path::new(VAE_TMP_PATH))
            .map_err(|e| RunError::Other(format!("VAE temp save: {e:?}")))?;
        drop(w);
    } else {
        log::info!("SD 1.5: reusing cached VAE remap at {VAE_TMP_PATH}");
    }
    LdmVAEDecoder::from_safetensors(VAE_TMP_PATH, VAE_IN_CHANNELS, VAE_SCALE, VAE_SHIFT, device)
        .map_err(|e| RunError::Other(format!("VAE remapped load: {e:?}")))
}

// ---------------------------------------------------------------------------
// CLIP-L tokenizer + weight loader
// ---------------------------------------------------------------------------

fn load_clip_weights(
    path: &str,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>, RunError> {
    let raw = flame_core::serialization::load_file(Path::new(path), device)
        .map_err(|e| RunError::Other(format!("CLIP load_file: {e:?}")))?;
    let mut weights = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let t = if v.dtype() == DType::BF16 {
            v
        } else {
            v.to_dtype(DType::BF16)
                .map_err(|e| RunError::Other(format!("CLIP cast bf16: {e:?}")))?
        };
        weights.insert(k, t);
    }
    Ok(weights)
}

fn tokenize_clip(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(CLIP_L_TOKENIZER) {
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
            log::warn!("SD 1.5: CLIP tokenizer failed: {e}; BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Schedule (same as SDXL — scaled-linear β, leading spacing, steps_offset=1)
// ---------------------------------------------------------------------------

fn build_sd15_schedule(num_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let num_train_steps = 1000usize;
    let beta_start: f64 = 0.00085;
    let beta_end: f64 = 0.012;

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

    let step_ratio = num_train_steps / num_steps;
    let mut ts: Vec<usize> = (0..num_steps).map(|i| i * step_ratio + 1).collect();
    ts.reverse();

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

// ---------------------------------------------------------------------------
// Cancel + image-conversion helpers (same as flux/klein/sd3/qwenimage/ernie/anima/sdxl)
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
