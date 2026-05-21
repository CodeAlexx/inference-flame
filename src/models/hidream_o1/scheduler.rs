//! `FlashFlowMatchEulerDiscreteScheduler` — HiDream-O1's stochastic
//! flow-matching Euler scheduler.
//!
//! Reference: `/home/alex/HiDream-O1-Image/models/flash_scheduler.py`
//!
//! ## Two-mode scheduler
//!
//! - **Dev (28 step)**: hardcoded `DEFAULT_TIMESTEPS` list at
//!   `pipeline.py:25-28`. The pipeline overrides `set_timesteps` output
//!   with this list and recomputes sigmas as `t/1000` (edge case C2/C3).
//!   `s_noise = noise_scale_schedule[step]` (default 7.5 constant for Dev).
//! - **Full (50 step)**: stock diffusers `FlowMatchEulerDiscreteScheduler`
//!   formula — `prev_sample = sample + (sigma_next - sigma) * model_output`.
//!   No noise injection. `pipeline.py:378-379`.
//!
//! ## The HiDream-modified Euler step (flash variant)
//!
//! Per `flash_scheduler.py:340-356`:
//!
//! ```python
//! sigma = self.sigmas[self.step_index]
//! sample = sample.to(torch.float32)
//! denoised = sample - model_output * sigma
//! if self.step_index < self.num_inference_steps:
//!     sigma_next = self.sigmas[self.step_index + 1]
//!     noise = randn(model_output.shape, ...)
//!     if noise_clip_std > 0:
//!         clip_val = noise_clip_std * noise.std().item()
//!         noise = noise.clamp(min=-clip_val, max=clip_val)
//!     sample = sigma_next * noise * s_noise + (1.0 - sigma_next) * denoised
//! ```
//!
//! Edge cases handled:
//! - **C7**: initial latent has std ≈ noise_scale_start (7.5/8.0). Lives
//!   in pipeline.rs (the `randn * noise_scale_start` step).
//! - **C8**: `noise.std()` is the SAMPLE std — computed across all
//!   elements of the freshly drawn noise tensor.
//! - **C9**: per-step noise uses CUDA RNG seeded by `seed + 1` (pipeline.rs).
//! - **C10**: stochastic Euler — re-injects scaled noise every step.
//!   At the last step `sigma_next = 0` so `sample = denoised`.

use std::sync::Arc;

use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};

use crate::sampling::cosmos_unipc::CosmosUniPcMultistepScheduler;

/// Hardcoded 28-step Dev timestep list.
///
/// Source: `/home/alex/HiDream-O1-Image/models/pipeline.py:25-28`.
pub const DEFAULT_TIMESTEPS_DEV: [u32; 28] = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
];

/// Flow-matching Euler scheduler with optional stochastic noise injection.
///
/// **Use `dev_28step()` for the Dev variant** (default in this port).
/// `full_50step()` for the Full variant which uses stock diffusers
/// deterministic Euler.
pub struct FlashFlowMatchEulerDiscreteScheduler {
    /// Always 1000 for Dev/Full per Python.
    pub num_train_timesteps: usize,
    /// Per-step timesteps (descending). Length = `num_inference_steps`.
    /// Stored as f32 (Python casts long → float32 before scheduler step).
    pub timesteps: Vec<f32>,
    /// Per-step sigmas. Length = `num_inference_steps + 1` (the `+1` is
    /// the trailing 0.0 that lets `sigmas[i+1]` work at the last step,
    /// per `flash_scheduler.py:247`).
    pub sigmas: Vec<f32>,
    /// `shift` parameter (Dev: 1.0, Full: 3.0).
    pub shift: f32,
    /// "flash" (stochastic Euler) vs "default" (deterministic Euler).
    pub mode: SchedulerMode,
}

/// Which Euler step variant to use.
///
/// - `Flash`: HiDream Dev — re-injects scaled noise every step. The 28-step
///   variant uses this. Formula: `flash_scheduler.py:340-356`.
/// - `Default`: stock diffusers `FlowMatchEulerDiscreteScheduler.step` —
///   deterministic. The 50-step Full variant uses this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerMode {
    Flash,
    Default,
}

impl FlashFlowMatchEulerDiscreteScheduler {
    /// Dev 28-step constructor.
    ///
    /// Mirrors `pipeline.py:79-93` for `scheduler_name="flash"` with
    /// `timesteps_list=DEFAULT_TIMESTEPS`, `shift=1.0`,
    /// `num_inference_steps=28`. After construction the timesteps array is
    /// the 28 hardcoded values and sigmas = `[t/1000 for t in timesteps] + [0.0]`.
    pub fn dev_28step() -> Self {
        let timesteps: Vec<f32> = DEFAULT_TIMESTEPS_DEV.iter().map(|&t| t as f32).collect();
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| t / 1000.0).collect();
        sigmas.push(0.0);
        Self {
            num_train_timesteps: 1000,
            timesteps,
            sigmas,
            shift: 1.0,
            mode: SchedulerMode::Flash,
        }
    }

    /// Full 50-step constructor.
    ///
    /// Mirrors `pipeline.py:79-93` for `scheduler_name="default"` with
    /// `timesteps_list=None`, `shift=3.0`, `num_inference_steps=50`.
    /// Implements the stock diffusers formula:
    /// - `sigmas = linspace(sigma_to_t(σ_max), sigma_to_t(σ_min), N) / 1000`
    /// - `sigmas = shift * sigmas / (1 + (shift-1) * sigmas)` (per `flash_scheduler.py:228`)
    /// - append trailing 0.0
    pub fn full_50step() -> Self {
        Self::full_n_step(50, 3.0)
    }

    /// Generic Full-mode constructor (deterministic Euler, dynamic sigmas).
    ///
    /// `n` = number of inference steps; `shift` = HiDream-Full default 3.0.
    pub fn full_n_step(n: usize, shift: f32) -> Self {
        // `flash_scheduler.py:99-114` — initial schedule (training):
        //   timesteps_train = linspace(1, 1000, 1000)[::-1]   (descending)
        //   sigmas_train = timesteps_train / 1000             (so σ_min = 1/1000, σ_max = 1.0)
        // Then the stock `set_timesteps` rebuilds for inference (`flash_scheduler.py:215-229`):
        //   timesteps = linspace(sigma_max * 1000, sigma_min * 1000, n)
        //   sigmas = timesteps / 1000
        //   sigmas = shift * sigmas / (1 + (shift-1) * sigmas)
        let sigma_min = 1.0_f32 / 1000.0;
        let sigma_max = 1.0_f32;

        // 1) raw timesteps via linspace.
        let mut timesteps = Vec::with_capacity(n);
        if n == 1 {
            timesteps.push(sigma_max * 1000.0);
        } else {
            let lo = sigma_min * 1000.0;
            let hi = sigma_max * 1000.0;
            for i in 0..n {
                let alpha = (i as f32) / ((n - 1) as f32);
                timesteps.push(hi + (lo - hi) * alpha);
            }
        }

        // 2) sigmas = timesteps / 1000.
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| t / 1000.0).collect();

        // 3) shift transform.
        for s in sigmas.iter_mut() {
            *s = (shift * *s) / (1.0 + (shift - 1.0) * *s);
        }

        // 4) recompute timesteps from shifted sigmas (per `flash_scheduler.py:240`).
        let timesteps_shifted: Vec<f32> = sigmas.iter().map(|&s| s * 1000.0).collect();

        // 5) trailing 0.0.
        sigmas.push(0.0);

        Self {
            num_train_timesteps: 1000,
            timesteps: timesteps_shifted,
            sigmas,
            shift,
            mode: SchedulerMode::Default,
        }
    }

    /// Number of denoising steps.
    pub fn num_inference_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Single step.
    ///
    /// # Arguments
    /// - `model_output`: post-CFG, post-negation velocity. Shape `[B, L, 3072]`.
    ///   Caller is responsible for the `model_output = -v_guided` flip
    ///   (edge case D1/F3 / `pipeline.py:374`).
    /// - `step_index`: index into `self.timesteps` (0-based).
    /// - `sample`: current `z` patches `[B, L, 3072]`.
    /// - `noise_for_step`: pre-drawn noise of shape `model_output.shape`.
    ///   The pipeline draws this with the CUDA-side RNG (edge case C9).
    ///   Pass `None` for the Default (deterministic) mode — it will be ignored.
    /// - `s_noise`: noise scaling factor. For Dev with constant 7.5
    ///   `noise_scale_schedule`, every step gets `s_noise = 7.5`.
    ///   Ignored in Default mode.
    /// - `noise_clip_std`: optional ±k·sample-std clip on the drawn noise.
    ///   Default 2.5 for Dev (`inference.py:35`). Ignored in Default mode.
    ///
    /// # Returns
    /// `prev_sample` — same shape as `sample`, BF16. The pipeline assigns
    /// this to `z` for the next step.
    ///
    /// # Math (Flash mode, `flash_scheduler.py:340-354`)
    /// ```text
    /// denoised   = sample - model_output * sigma                       (FP32)
    /// noise_clip = clamp(noise, -k * std(noise), +k * std(noise))      if k > 0
    /// sample'    = sigma_next * noise_clip * s_noise + (1 - sigma_next) * denoised
    /// ```
    ///
    /// # Math (Default mode, stock diffusers)
    /// ```text
    /// prev_sample = sample + (sigma_next - sigma) * model_output       (FP32)
    /// ```
    pub fn step(
        &self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
        noise_for_step: Option<&Tensor>,
        s_noise: f32,
        noise_clip_std: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        if step_index >= self.timesteps.len() {
            return Err(Error::InvalidOperation(format!(
                "FlashFlowMatchEulerDiscreteScheduler::step: step_index {} >= num_inference_steps {}",
                step_index,
                self.timesteps.len()
            )));
        }
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Upcast to FP32 (per `flash_scheduler.py:338`).
        let sample_f32 = sample.to_dtype(DType::F32)?;
        let model_output_f32 = model_output.to_dtype(DType::F32)?;
        let out_dtype = sample.dtype();

        match self.mode {
            SchedulerMode::Flash => {
                // denoised = sample - model_output * sigma
                let mo_sigma = model_output_f32.mul_scalar(sigma)?;
                let denoised = sample_f32.sub(&mo_sigma)?;

                // noise injection at every step (including last, where sigma_next=0).
                // The Python `if self.step_index < self.num_inference_steps:` is
                // ALWAYS true here because step_index < N before increment.
                let noise = match noise_for_step {
                    Some(n) => n.to_dtype(DType::F32)?,
                    None => {
                        // Caller forgot to pass; emit zero-noise fallback so we
                        // don't silently crash. (The pipeline always passes one.)
                        Tensor::zeros_dtype(
                            denoised.shape().clone(),
                            DType::F32,
                            device.clone(),
                        )?
                    }
                };

                // Optional ±k·sample-std clip (edge case C8). Compute std from
                // the host-side noise vector — small relative to model fwd cost.
                let noise = if noise_clip_std > 0.0 {
                    let host = noise.to_vec_f32()?;
                    let n = host.len() as f32;
                    let mean = host.iter().sum::<f32>() / n;
                    let var = host
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .sum::<f32>()
                        / n.max(1.0);
                    let std = var.sqrt();
                    let clip_val = noise_clip_std * std;
                    // clamp to [-clip_val, clip_val] elementwise.
                    let clamped: Vec<f32> = host
                        .iter()
                        .map(|&x| x.max(-clip_val).min(clip_val))
                        .collect();
                    Tensor::from_vec_dtype(
                        clamped,
                        noise.shape().clone(),
                        device.clone(),
                        DType::F32,
                    )?
                } else {
                    noise
                };

                // sample = sigma_next * noise * s_noise + (1 - sigma_next) * denoised
                let weight_noise = sigma_next * s_noise;
                let weight_den = 1.0 - sigma_next;
                let term_noise = noise.mul_scalar(weight_noise)?;
                let term_den = denoised.mul_scalar(weight_den)?;
                let next_sample = term_noise.add(&term_den)?;

                if out_dtype == DType::F32 {
                    Ok(next_sample)
                } else {
                    next_sample.to_dtype(out_dtype)
                }
            }
            SchedulerMode::Default => {
                // prev_sample = sample + (sigma_next - sigma) * model_output
                let dsigma = sigma_next - sigma;
                let term = model_output_f32.mul_scalar(dsigma)?;
                let next_sample = sample_f32.add(&term)?;
                if out_dtype == DType::F32 {
                    Ok(next_sample)
                } else {
                    next_sample.to_dtype(out_dtype)
                }
            }
        }
    }
}

// ─── Internal helper: shape preservation for `Tensor::from_vec_dtype` ────────

#[allow(dead_code)]
fn _shape_passthrough() -> Shape {
    Shape::from_dims(&[0])
}

// ─────────────────────────────────────────────────────────────────────────────
// HiDreamScheduler — enum dispatch over the available samplers.
// Added 2026-05-21 (Alex's call) to plug `CosmosUniPcMultistepScheduler`
// into HiDream-O1 alongside the reference FlashFlowMatchEulerDiscrete.
//
// **Deviation from upstream:** the HiDream-O1 HF model card does NOT recommend
// UniPC — it specifies `FlowMatchEulerDiscreteScheduler` (Full 50-step) and
// `FlashFlowMatchEulerDiscreteScheduler` (Dev 28-step). Routing through UniPC
// is a deliberate, user-chosen swap. Parity vs reference Python output is NOT
// guaranteed with this path — different multistep math, no stochastic noise
// injection, no `noise_clip_std`. Use when the UniPC convergence rate is
// actually wanted at lower step counts; use the Flash/Default modes for
// reference fidelity.
//
// **Velocity convention:** verified equivalent.
// - Flash:    `denoised = sample - model_output * sigma`         (`model_output = -v_guided`)
// - UniPC:    `x0_pred  = sample - sigma_t * model_output`        (`predict_x0=true`)
// Both produce identical x0 given the same `model_output = -v_guided`,
// so the pipeline's existing sign flip works for both paths unchanged.
// (See `models/hidream_o1/pipeline.rs:540` and
// `sampling/cosmos_unipc.rs:182-192`.)
// ─────────────────────────────────────────────────────────────────────────────

/// Which sampler family the unified scheduler is using.
///
/// `Flash` / `Default` come from the Flash scheduler's own internal mode
/// (`SchedulerMode`); `UniPc` is the new UniPC option.
///
/// The pipeline reads this to decide whether to draw per-step noise:
/// only `Flash` injects stochastic noise. `Default` and `UniPc` are
/// deterministic — calling code MUST NOT pass `noise_for_step` for those.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiDreamSchedulerKind {
    /// `FlashFlowMatchEulerDiscreteScheduler` in `Flash` mode (28-step Dev).
    /// Stochastic — re-injects scaled noise every step.
    FlashStochastic,
    /// `FlashFlowMatchEulerDiscreteScheduler` in `Default` mode (50-step Full).
    /// Deterministic Euler — no noise injection.
    FlowMatchEuler,
    /// `CosmosUniPcMultistepScheduler` (bh2 multistep, predict_x0=true).
    /// Deterministic — no noise injection.
    UniPc,
}

impl HiDreamSchedulerKind {
    /// Whether this scheduler kind needs `noise_for_step` from the caller.
    /// Only `FlashStochastic` does; the others are deterministic.
    pub fn needs_step_noise(self) -> bool {
        matches!(self, HiDreamSchedulerKind::FlashStochastic)
    }
}

/// Unified scheduler enum. Owns one of the underlying scheduler structs;
/// dispatches `step`, `timesteps`, `num_inference_steps` to the active arm.
pub enum HiDreamScheduler {
    /// Flash (stochastic) or Default (deterministic Euler) — same struct,
    /// disambiguated by its internal `mode` field.
    FlashFlowMatch(FlashFlowMatchEulerDiscreteScheduler),
    /// UniPC bh2 multistep. Stateful: `step()` advances internally.
    UniPc(CosmosUniPcMultistepScheduler),
}

impl HiDreamScheduler {
    /// Dev 28-step Flash. Reference parity path.
    pub fn flash_dev_28step() -> Self {
        Self::FlashFlowMatch(FlashFlowMatchEulerDiscreteScheduler::dev_28step())
    }

    /// Full 50-step deterministic Euler. Reference parity path.
    pub fn full_50step() -> Self {
        Self::FlashFlowMatch(FlashFlowMatchEulerDiscreteScheduler::full_50step())
    }

    /// Full N-step deterministic Euler with arbitrary shift. Reference parity.
    pub fn full_n_step(n: usize, shift: f32) -> Self {
        Self::FlashFlowMatch(FlashFlowMatchEulerDiscreteScheduler::full_n_step(n, shift))
    }

    /// **UniPC variant** (deliberate deviation from reference).
    ///
    /// Wires `CosmosUniPcMultistepScheduler` with HiDream-O1's defaults:
    /// `num_train_timesteps = 1000`, `solver_order = 2`, `predict_x0 = true`,
    /// `final_sigmas_type = "zero"`. Caller provides `n_inference_steps` and
    /// `shift` (typically 3.0 for Full / 1.0 for Dev parity, but pick what
    /// converges best at low step counts).
    pub fn unipc(n_inference_steps: usize, shift: f32) -> Self {
        Self::UniPc(CosmosUniPcMultistepScheduler::new(
            /*num_train_timesteps=*/ 1000,
            n_inference_steps,
            shift,
            /*solver_order=*/ 2,
        ))
    }

    /// Which sampler family this is. Pipeline branches on this for
    /// noise-injection decisions and noise_scale_start defaults.
    pub fn kind(&self) -> HiDreamSchedulerKind {
        match self {
            HiDreamScheduler::FlashFlowMatch(s) => match s.mode {
                SchedulerMode::Flash => HiDreamSchedulerKind::FlashStochastic,
                SchedulerMode::Default => HiDreamSchedulerKind::FlowMatchEuler,
            },
            HiDreamScheduler::UniPc(_) => HiDreamSchedulerKind::UniPc,
        }
    }

    /// Total number of inference steps. Pipeline loops `0..num_inference_steps()`.
    pub fn num_inference_steps(&self) -> usize {
        match self {
            HiDreamScheduler::FlashFlowMatch(s) => s.num_inference_steps(),
            HiDreamScheduler::UniPc(s) => s.num_inference_steps,
        }
    }

    /// Timesteps array (descending). Pipeline reads `timesteps[step_idx]`
    /// to compute the `t` value fed to the DiT.
    pub fn timesteps(&self) -> &[f32] {
        match self {
            HiDreamScheduler::FlashFlowMatch(s) => &s.timesteps,
            HiDreamScheduler::UniPc(s) => s.timesteps(),
        }
    }

    /// Sigmas array (length = num_inference_steps + 1; last is 0.0).
    pub fn sigmas(&self) -> &[f32] {
        match self {
            HiDreamScheduler::FlashFlowMatch(s) => &s.sigmas,
            HiDreamScheduler::UniPc(s) => s.sigmas(),
        }
    }

    /// Unified single-step. The `noise_for_step`, `s_noise`, `noise_clip_std`
    /// args are IGNORED for `UniPc` and `FlowMatchEuler` (deterministic).
    /// `step_index` is used by the Flash arm but the UniPC arm tracks its
    /// own `step_index` internally and ignores the caller's value (it must
    /// still be called in monotonic order). If the caller's `step_index`
    /// disagrees with the UniPC internal counter we return an error rather
    /// than silently mis-step.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
        noise_for_step: Option<&Tensor>,
        s_noise: f32,
        noise_clip_std: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        match self {
            HiDreamScheduler::FlashFlowMatch(s) => s.step(
                model_output,
                step_index,
                sample,
                noise_for_step,
                s_noise,
                noise_clip_std,
                device,
            ),
            HiDreamScheduler::UniPc(s) => {
                if s.step_index() != step_index {
                    return Err(Error::InvalidOperation(format!(
                        "HiDreamScheduler::UniPc: caller step_index {} ≠ scheduler internal {} \
                         — UniPC is stateful, callers must invoke step() once per step in order",
                        step_index,
                        s.step_index()
                    )));
                }
                // s_noise / noise_clip_std / noise_for_step are silently
                // ignored — UniPC is deterministic. We don't warn at every
                // step (would be 50 log lines per generation); pipeline
                // should branch on `kind()` and skip drawing noise instead.
                let _ = (noise_for_step, s_noise, noise_clip_std);
                s.step(model_output, sample)
            }
        }
    }
}
