//! Stable Cascade (Würstchen v3) real-inference path.
//!
//! Mirrors `worker/{flux,chroma,klein,sd3,qwenimage,ernie,anima,sdxl,sd15}.rs`
//! structurally — lazy load, drop-to-fit VRAM dance, per-step
//! Started/Progress/Done event emission. Underlying pipeline mirrors
//! `inference-flame/src/bin/cascade_infer.rs`.
//!
//! ## Architecture (3 stages)
//!
//! - **Stage C** (prior UNet): `WuerstchenUNet::stage_c()` — denoises a
//!   small [1, 16, 24, 24] latent for `steps_c` iterations. Conditioned on
//!   CLIP-G hidden + pooled. ~7 GB BF16, the largest stage.
//! - **Stage B** (decoder UNet): `WuerstchenUNet::stage_b()` — denoises a
//!   [1, 4, H/4, W/4] latent for `steps_b` iterations. Conditioned on
//!   CLIP-G hidden + pooled AND on the Stage C latent (effnet_cond).
//! - **Stage A** (Paella VQ-GAN): `PaellaVQDecoder` — decodes the Stage B
//!   latent → RGB [1, 3, H, W].
//!
//! ## Text encoder
//!
//! CLIP-ViT-bigG-14 (`ClipConfig::clip_g()` + `encode_cascade`). Single
//! encoder, encoded inline (no embeddings file). Loaded → encode cond+uncond
//! → dropped before Stage C loads. AGENT-DEFAULT: same convention as the
//! sd15 worker — typed prompt is the actual prompt, no cache file. The
//! `job.negative` field flows through as a real negative prompt for both
//! Stage C and Stage B uncond passes.
//!
//! ## CFG
//!
//! Real two-pass on BOTH stages. `pred = uncond + cfg * (cond - uncond)`.
//! Defaults from cascade_infer.rs: `cfg_c=4.0`, `cfg_b=1.1` (near-deterministic).
//!
//! AGENT-DEFAULT: the `job.cfg` slider drives BOTH `cfg_c` and `cfg_b`
//! identically. The reference bin lets them be tuned independently via
//! `--cfg-c` / `--cfg-b` but the UI has only one cfg field. If a future UI
//! split adds a "Stage B CFG" knob, point it at this AGENT-DEFAULT comment.
//! For now, a UI cfg of 4.0 → both stages 4.0; if `job.cfg <= 0` we fall
//! back to the per-stage defaults from the bin.
//!
//! ## Schedule
//!
//! Both stages use `DDPMWuerstchenScheduler::new(num_steps)` — linspace
//! [1.0, 0.0] over `num_steps + 1` values. Different step COUNTS though
//! (stage C and stage B have separate `steps_c` / `steps_b`).
//!
//! ## Step formula (eps-prediction, DDIM-style)
//!
//! Per `cascade_infer.rs` and `ddpm_wuerstchen::step_eps_ddim`: recover x0
//! from `(x_t, eps)` via the alpha schedule, then forward-noise to t_next.
//! NOT a flow Euler step. The bin chose DDIM over the stochastic DDPM step
//! because the latter empirically diverged for the Stage C UNet — so this
//! worker matches that choice unconditionally.
//!
//! ## Latents
//!
//! - Stage C: [1, 16, 24, 24] for 1024×1024 (hardcoded fallback like the
//!   bin — the `((H+31)/32 * 3) / 4` formula is a rough approximation).
//! - Stage B: [1, 4, H/4, W/4]
//! - Final RGB from Paella: [1, 3, H, W] in **[0, 1]** range (NOT [-1, 1]
//!   like the LDM/SD3 VAE families). The image-conversion helper is
//!   different from the sister workers' `decoded_to_color_image` and lives
//!   inline here as `paella_to_color_image`.
//!
//! ## Progress event accounting (two stages, one bar)
//!
//! AGENT-DEFAULT: emit ONE `Started` with `total_steps = steps_c + steps_b`,
//! and Progress events count through both stages monotonically (1..steps_c
//! during Stage C, then steps_c+1..steps_c+steps_b during Stage B). The
//! Stage A VQ-GAN decode is one shot — no Progress for it. Picked over
//! "two Started events" because the UI canvas is a single progress bar
//! and a mid-job reset would look like a regression.
//!
//! ## VRAM budget on a 24 GB card
//!
//! Sequential stage loads keep peak under 10 GB during text encode. The
//! big numbers come from the Stage C UNet at ~7 GB BF16:
//! - CLIP-G (load → encode → drop): ~1.4 GB peak
//! - Stage C UNet (load → denoise → drop): ~7 GB peak
//! - Stage B UNet (load → denoise → drop): ~3 GB peak
//! - Stage A Paella (load → decode → drop): ~0.5 GB peak
//! - Activations (1024² output → 256² Stage B latent): ~2 GB
//!
//! Comfortable. Aggressive sequential drop-and-reload is essential — any
//! two stages held resident together would risk OOM with the conv workspace.
//!
//! ## Defaults
//!
//! From the bin: `steps_c=20`, `steps_b=10`, `cfg_c=4.0`, `cfg_b=1.1`.
//! AGENT-DEFAULT: when `job.steps` is non-zero we split it 2:1 between
//! Stage C and Stage B (`steps_c = ceil(2*steps/3)`, `steps_b = steps - steps_c`)
//! so the user's typed step count maps proportionally. When `job.steps==0`
//! we use the bin defaults exactly.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;

use flame_core::{DType, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::paella_vq::PaellaVQDecoder;
use inference_flame::models::wuerstchen_unet::{WuerstchenUNet, WuerstchenUNetConfig};
use inference_flame::sampling::ddpm_wuerstchen::{randn_bf16, DDPMWuerstchenScheduler};

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Hardcoded weight paths (mirrors cascade_infer.rs).
//
// AGENT-DEFAULT: hardcoded constants pointing at the HF snapshot directory.
// Same rationale as flux.rs / klein.rs.
// ===========================================================================

const CKPT_ROOT: &str =
    "/home/alex/.cache/huggingface/hub/models--stabilityai--stable-cascade/snapshots/a89f66d459ae653e3b4d4f992a7c3789d0dc4d16";

fn clip_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("text_encoder/model.bf16.safetensors")
}
fn clip_tokenizer_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("tokenizer/tokenizer.json")
}
fn stage_c_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_c_bf16.safetensors")
}
fn stage_b_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_b_bf16.safetensors")
}
fn stage_a_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_a.safetensors")
}

const CLIP_SEQ_LEN: usize = 77;

/// Default Stage C step count. cascade_infer.rs uses 20.
const DEFAULT_STEPS_C: u32 = 20;
/// Default Stage B step count. cascade_infer.rs uses 10.
const DEFAULT_STEPS_B: u32 = 10;
/// Default Stage C CFG. cascade_infer.rs uses 4.0.
const DEFAULT_CFG_C: f32 = 4.0;
/// Default Stage B CFG. cascade_infer.rs uses 1.1 (near-deterministic).
const DEFAULT_CFG_B: f32 = 1.1;

// ===========================================================================
// State
// ===========================================================================

/// Worker-local Cascade resources. Nothing kept resident — encoder, all
/// three stages, and Paella all loaded per-job. The drop-and-reload tax
/// across three stages is significant (~10-15s warm SSD) but unavoidable
/// given the per-stage peaks.
pub struct CascadeState {
    pub device: Arc<CudaDevice>,
}

impl CascadeState {
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
    state: &mut CascadeState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    // Split the user's step count 2:1 between Stage C and Stage B if a
    // value was provided; otherwise fall back to the bin defaults exactly.
    let (steps_c, steps_b) = if job.steps == 0 {
        (DEFAULT_STEPS_C, DEFAULT_STEPS_B)
    } else {
        // ceil(2*steps/3) for Stage C, remainder for Stage B. Guarantees
        // each stage gets at least 1 step for any input >=2.
        let total = job.steps.max(2);
        let c = ((2 * total) + 2) / 3; // ceil(2*total/3)
        let b = total.saturating_sub(c).max(1);
        (c.max(1), b)
    };
    let total_steps = steps_c + steps_b;

    // Single CFG slider drives both stages identically (see module docstring).
    let (cfg_c, cfg_b) = if job.cfg <= 0.0 {
        (DEFAULT_CFG_C, DEFAULT_CFG_B)
    } else {
        (job.cfg, job.cfg)
    };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps,
    });
    ctx.request_repaint();

    match run_inner(
        job, steps_c, steps_b, cfg_c, cfg_b, state, ui_rx, ev_tx, ctx, pending,
    ) {
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
            log::warn!("Cascade job {} failed: {msg}", job.id);
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
    steps_c: u32,
    steps_b: u32,
    cfg_c: f32,
    cfg_b: f32,
    state: &mut CascadeState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Closure-wrap so we ALWAYS flush the pool on every exit path. Cascade
    // is the most VRAM-fragmented of all workers — each stage's load and
    // drop leaves pooled buffers around that the next stage's allocations
    // would race. Mandatory pattern.
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(
            job, steps_c, steps_b, cfg_c, cfg_b, state, ui_rx, ev_tx, ctx, pending,
        )
    })();
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_inner_body(
    job: &GenerateJob,
    steps_c: u32,
    steps_b: u32,
    cfg_c: f32,
    cfg_b: f32,
    state: &mut CascadeState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();
    let total_steps = steps_c + steps_b;

    // -------- 0. Geometry --------
    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    // Mirrors cascade_infer.rs:239-245. The exact diffusers compression
    // formula is messy; the bin uses a hardcoded 24 for 1024×1024, falling
    // back to the rough `((H+31)/32 * 3) / 4` approximation otherwise.
    let stage_c_h = if height == 1024 {
        24
    } else {
        (((height + 31) / 32) * 3 / 4).max(1)
    };
    let stage_c_w = if width == 1024 {
        24
    } else {
        (((width + 31) / 32) * 3 / 4).max(1)
    };
    let stage_b_h = (height / 4).max(1);
    let stage_b_w = (width / 4).max(1);
    log::info!(
        "Cascade: Stage C latent [1, 16, {stage_c_h}, {stage_c_w}], \
         Stage B latent [1, 4, {stage_b_h}, {stage_b_w}], \
         output {width}x{height}"
    );

    // Seed flame-core's global RNG so `randn_bf16` (used for both stages'
    // initial noise) is deterministic given job.seed. cascade_infer.rs:298.
    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        job.seed as u64
    };
    flame_core::rng::set_seed(seed_u64)
        .map_err(|e| RunError::Other(format!("rng seed: {e:?}")))?;

    drain_pending(ui_rx, pending)?;

    // -------- 1. CLIP-G encode (cond + uncond, then drop) --------
    if !clip_path().exists() {
        return Err(RunError::Other(format!(
            "Cascade CLIP-G weights not found at {}",
            clip_path().display()
        )));
    }
    log::info!("Cascade: CLIP-G encode (prompt + negative)");
    let t_enc = Instant::now();
    let (pos_hidden, pos_pooled, neg_hidden, neg_pooled) = {
        let raw = flame_core::serialization::load_file(clip_path(), &device)
            .map_err(|e| RunError::Other(format!("CLIP-G load: {e:?}")))?;
        let mut weights = std::collections::HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let t = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)
                    .map_err(|e| RunError::Other(format!("CLIP-G cast bf16: {e:?}")))?
            };
            weights.insert(k, t);
        }
        let clip = ClipEncoder::new(weights, ClipConfig::clip_g(), device.clone());
        let pos_tokens = tokenize_clip(&job.prompt, &clip_tokenizer_path());
        let (ph, pp) = clip
            .encode_cascade(&pos_tokens)
            .map_err(|e| RunError::Other(format!("CLIP-G cond: {e:?}")))?;
        let neg_tokens = tokenize_clip(&job.negative, &clip_tokenizer_path());
        let (nh, np) = clip
            .encode_cascade(&neg_tokens)
            .map_err(|e| RunError::Other(format!("CLIP-G uncond: {e:?}")))?;
        (ph, pp, nh, np)
        // clip dropped here — ~1.4 GB freed
    };
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "Cascade: CLIP-G done in {:.1}s — pos_hidden {:?} pos_pooled {:?}",
        t_enc.elapsed().as_secs_f32(),
        pos_hidden.shape().dims(),
        pos_pooled.shape().dims(),
    );

    drain_pending(ui_rx, pending)?;

    // -------- 2. Stage C denoise --------
    if !stage_c_path().exists() {
        return Err(RunError::Other(format!(
            "Stage C weights not found at {}",
            stage_c_path().display()
        )));
    }
    log::info!("Cascade: loading Stage C from {}", stage_c_path().display());
    let t_stage_c_load = Instant::now();
    let stage_c_latent = {
        let unet_c = load_cascade_unet(
            stage_c_path().to_str().unwrap(),
            WuerstchenUNetConfig::stage_c(),
            &device,
        )
        .map_err(|e| RunError::Other(format!("Stage C load: {e:?}")))?;
        log::info!(
            "Cascade: Stage C loaded in {:.1}s",
            t_stage_c_load.elapsed().as_secs_f32()
        );

        let scheduler = DDPMWuerstchenScheduler::new(steps_c as usize);
        let mut x = randn_bf16(&[1, 16, stage_c_h, stage_c_w], &device)
            .map_err(|e| RunError::Other(format!("Stage C noise: {e:?}")))?;
        let timesteps = scheduler.timesteps.clone();

        // Capture step-timing baseline AFTER load so the first few Progress
        // events don't fold the ~3-7 s Stage C load into per-step. Reset at
        // the Stage B boundary too (see below) so each stage's ETA reflects
        // its own per-step rate, smoothing the transition.
        let t_steps_start = Instant::now();

        for step in 0..steps_c as usize {
            drain_pending(ui_rx, pending)?;
            let r = timesteps[step];
            let r_next = timesteps[step + 1];

            // Per-step temporaries scope so they drop before the next step.
            let next_x = {
                let v_cond = unet_c
                    .forward(&x, r, Some(&pos_pooled), Some(&pos_hidden), None)
                    .map_err(|e| RunError::Other(format!("Stage C cond step {step}: {e:?}")))?;
                let v = if cfg_c > 1.0 {
                    let v_uncond = unet_c
                        .forward(&x, r, Some(&neg_pooled), Some(&neg_hidden), None)
                        .map_err(|e| {
                            RunError::Other(format!("Stage C uncond step {step}: {e:?}"))
                        })?;
                    cfg_combine(&v_cond, &v_uncond, cfg_c)?
                } else {
                    v_cond
                };
                // DDIM-style eps step (matches cascade_infer.rs choice — the
                // stochastic DDPM step empirically diverged for Stage C).
                scheduler
                    .step_eps_ddim(&v, r, r_next, &x)
                    .map_err(|e| RunError::Other(format!("Stage C DDIM step {step}: {e:?}")))?
            };
            x = next_x;

            // Progress: monotonic across both stages — Stage C contributes
            // 1..steps_c, Stage B picks up at steps_c+1. ETA uses the
            // stage-local per-step rate against TOTAL remaining steps —
            // accurate for Stage C; will re-baseline at the Stage B boundary.
            let step_global = (step + 1) as u32;
            let stage_done = (step + 1) as u32;
            let elapsed = t_steps_start.elapsed().as_secs_f32();
            let per_step = elapsed / stage_done as f32;
            let eta_secs =
                ((total_steps - step_global) as f32 * per_step).round().max(0.0) as u32;
            let _ = ev_tx.send(WorkerEvent::Progress {
                id: job.id,
                step: step_global,
                total: total_steps,
                eta_secs,
            });
            ctx.request_repaint();
        }

        drop(unet_c);
        x
    };
    // Flush before Stage B load — Stage C's ~7 GB needs to actually leave
    // the pool before Stage B's ~3 GB lands or we'll fragment.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "Cascade: Stage C done — latent {:?}, pool flushed",
        stage_c_latent.shape().dims()
    );

    drain_pending(ui_rx, pending)?;

    // -------- 3. Stage B denoise --------
    if !stage_b_path().exists() {
        return Err(RunError::Other(format!(
            "Stage B weights not found at {}",
            stage_b_path().display()
        )));
    }
    log::info!("Cascade: loading Stage B from {}", stage_b_path().display());
    let t_stage_b_load = Instant::now();
    let stage_b_latent = {
        let unet_b = load_cascade_unet(
            stage_b_path().to_str().unwrap(),
            WuerstchenUNetConfig::stage_b(),
            &device,
        )
        .map_err(|e| RunError::Other(format!("Stage B load: {e:?}")))?;
        log::info!(
            "Cascade: Stage B loaded in {:.1}s",
            t_stage_b_load.elapsed().as_secs_f32()
        );

        let scheduler = DDPMWuerstchenScheduler::new(steps_b as usize);
        let mut x = randn_bf16(&[1, 4, stage_b_h, stage_b_w], &device)
            .map_err(|e| RunError::Other(format!("Stage B noise: {e:?}")))?;
        let timesteps = scheduler.timesteps.clone();

        // Re-baseline AFTER Stage B load so its ~1-3 s load doesn't fold
        // into the first Progress event's per-step. Stage B's per-step rate
        // differs from Stage C's (different latent size / UNet), so this
        // also gives a more accurate ETA than averaging across stages.
        let t_steps_start = Instant::now();

        for step in 0..steps_b as usize {
            drain_pending(ui_rx, pending)?;
            let r = timesteps[step];
            let r_next = timesteps[step + 1];

            let next_x = {
                let v_cond = unet_b
                    .forward(
                        &x,
                        r,
                        Some(&pos_pooled),
                        Some(&pos_hidden),
                        Some(&stage_c_latent),
                    )
                    .map_err(|e| RunError::Other(format!("Stage B cond step {step}: {e:?}")))?;
                let v = if cfg_b > 1.0 {
                    let v_uncond = unet_b
                        .forward(
                            &x,
                            r,
                            Some(&neg_pooled),
                            Some(&neg_hidden),
                            Some(&stage_c_latent),
                        )
                        .map_err(|e| {
                            RunError::Other(format!("Stage B uncond step {step}: {e:?}"))
                        })?;
                    cfg_combine(&v_cond, &v_uncond, cfg_b)?
                } else {
                    v_cond
                };
                scheduler
                    .step_eps_ddim(&v, r, r_next, &x)
                    .map_err(|e| RunError::Other(format!("Stage B DDIM step {step}: {e:?}")))?
            };
            x = next_x;

            // Progress: Stage B picks up at steps_c+1. ETA uses stage-local
            // per-step rate * remaining_total_steps (which during Stage B is
            // exactly the remaining Stage B steps). Resetting at the boundary
            // smooths the transition vs averaging across the two stages.
            let step_global = steps_c + (step + 1) as u32;
            let stage_done = (step + 1) as u32;
            let elapsed = t_steps_start.elapsed().as_secs_f32();
            let per_step = elapsed / stage_done as f32;
            let eta_secs =
                ((total_steps - step_global) as f32 * per_step).round().max(0.0) as u32;
            let _ = ev_tx.send(WorkerEvent::Progress {
                id: job.id,
                step: step_global,
                total: total_steps,
                eta_secs,
            });
            ctx.request_repaint();
        }

        drop(unet_b);
        x
    };
    // Drop CLIP-G context tensors + Stage C latent before Stage A — none
    // of them are needed for the VQ-GAN decode and Paella's conv workspace
    // wants room.
    drop(pos_hidden);
    drop(pos_pooled);
    drop(neg_hidden);
    drop(neg_pooled);
    let stage_c_latent_dims = stage_c_latent.shape().dims().to_vec();
    drop(stage_c_latent);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    log::info!(
        "Cascade: Stage B done — latent {:?}, Stage C latent {stage_c_latent_dims:?} \
         dropped, pool flushed",
        stage_b_latent.shape().dims(),
    );

    drain_pending(ui_rx, pending)?;

    // -------- 4. Stage A (Paella VQ-GAN) decode --------
    if !stage_a_path().exists() {
        return Err(RunError::Other(format!(
            "Stage A weights not found at {}",
            stage_a_path().display()
        )));
    }
    log::info!("Cascade: loading Stage A (Paella VQ-GAN)");
    let t_vae = Instant::now();
    let rgb = {
        let vae = load_cascade_paella(stage_a_path().to_str().unwrap(), &device)
            .map_err(|e| RunError::Other(format!("Stage A load: {e:?}")))?;
        let img = vae
            .decode(&stage_b_latent)
            .map_err(|e| RunError::Other(format!("Stage A decode: {e:?}")))?;
        drop(vae);
        img
    };
    drop(stage_b_latent);
    log::info!(
        "Cascade: Stage A decode done in {:.1}s — rgb {:?}",
        t_vae.elapsed().as_secs_f32(),
        rgb.shape().dims(),
    );

    // Paella outputs in [0, 1] (NOT [-1, 1] like the LDM/SD3 VAE family).
    paella_to_color_image(&rgb)
        .map_err(|e| RunError::Other(format!("to ColorImage: {e:?}")))
}

// ---------------------------------------------------------------------------
// CFG combine helper (mirrors cascade_infer.rs::cfg_combine)
// ---------------------------------------------------------------------------

/// `pred = uncond + cfg * (cond - uncond)`. Wraps the three Tensor ops in
/// our `RunError` envelope.
fn cfg_combine(v_cond: &Tensor, v_uncond: &Tensor, cfg: f32) -> Result<Tensor, RunError> {
    let diff = v_cond
        .sub(v_uncond)
        .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
    let scaled = diff
        .mul_scalar(cfg)
        .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?;
    v_uncond
        .add(&scaled)
        .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))
}

// ---------------------------------------------------------------------------
// GGUF-or-safetensors dispatch for the three Cascade stages. All three stages
// now accept either a `.gguf` (dequantized + uploaded in one shot by the
// Phase 1 loader) or a `.safetensors` file. Mirrors the per-worker pattern
// in flux/klein/zimage/sdxl/sd15/anima/ernie.
// ---------------------------------------------------------------------------

fn load_cascade_unet(
    path: &str,
    config: WuerstchenUNetConfig,
    device: &Arc<CudaDevice>,
) -> Result<WuerstchenUNet, flame_core::Error> {
    if path.to_ascii_lowercase().ends_with(".gguf") {
        // Cascade GGUF route is currently unavailable: the lib previously
        // exposed `WuerstchenUNet::from_weights(weights, config, device)`
        // (HashMap input) but only `WuerstchenUNet::load(path, config, device)`
        // (safetensors path) remains today. Fail-fast — same pattern as the
        // FLUX / Chroma / SD15 workers' GGUF gates.
        Err(flame_core::Error::InvalidInput(format!(
            "Cascade UNet GGUF not yet supported in the UI: the lib no \
             longer exposes a HashMap-input loader for WuerstchenUNet. \
             Use a .safetensors stage for Cascade. (path={path})"
        )))
    } else {
        WuerstchenUNet::load(path, config, device)
    }
}

fn load_cascade_paella(
    path: &str,
    device: &Arc<CudaDevice>,
) -> Result<PaellaVQDecoder, flame_core::Error> {
    if path.to_ascii_lowercase().ends_with(".gguf") {
        // Same lib drift as load_cascade_unet: `PaellaVQDecoder::from_weights`
        // (HashMap input) was removed; only `PaellaVQDecoder::load(path,
        // device)` (safetensors path) remains. Fail-fast.
        Err(flame_core::Error::InvalidInput(format!(
            "Cascade Paella VQ-GAN GGUF not yet supported in the UI: the \
             lib no longer exposes a HashMap-input loader for \
             PaellaVQDecoder. Use a .safetensors decoder for Cascade. \
             (path={path})"
        )))
    } else {
        PaellaVQDecoder::load(path, device)
    }
}

// ---------------------------------------------------------------------------
// Tokenizer (CLIP BPE, pad to 77 with EOS)
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str, tokenizer_path: &std::path::Path) -> Vec<i32> {
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
            log::warn!("Cascade: tokenizer failed: {e}; BOS/EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Cancel helper (same as sister workers)
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

// ---------------------------------------------------------------------------
// Image conversion — Paella outputs in [0, 1], NOT [-1, 1]
// ---------------------------------------------------------------------------

/// Cascade-specific RGB tensor → ColorImage. Differs from the sister
/// workers' `decoded_to_color_image` (which clamps `(-1, 1)` then maps to
/// `(0, 255)` via `(v+1)*127.5`). Paella's VQ-GAN output is already in
/// `[0, 1]` per `cascade_infer.rs:497-499`.
fn paella_to_color_image(rgb: &Tensor) -> Result<ColorImage, flame_core::Error> {
    let dims = rgb.shape().dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Paella decode shape {dims:?} != [1, 3, H, W]"
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
            let r = (host[i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (host[plane + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (host[2 * plane + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    Ok(ColorImage {
        size: [w, h],
        pixels,
    })
}
