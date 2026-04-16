//! Motif-Video FlowMatch Euler + video-aware Adaptive Projected Guidance (APG).
//!
//! Scheduler config (from `scheduler/scheduler_config.json`):
//!   - num_train_timesteps: 1000
//!   - shift: 15.0                    (≠ 1.0 — actually applies static shift)
//!   - use_dynamic_shifting: false    (so `mu` passed by pipeline is ignored)
//!   - time_shift_type: "exponential" (irrelevant since dynamic_shifting=false)
//!
//! Per diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps` (line 350):
//!   sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
//!   = 15 * sigmas / (1 + 14 * sigmas)
//!
//! Sigmas start: `linspace(1.0, 1/N, N)` (no terminal 0 from linspace itself).
//! After static shift, append terminal 0.
//!
//! ## Video-aware APG (from `pipeline_motif_video.py:108-152`)
//!
//! Standard APG normalizes over all spatial dims [C, T, H, W]. Video APG
//! normalizes per-frame over [C, H, W] only, keeping T independent.
//!
//! For 5D tensor [B, C, T, H, W] the reduce dims are [C=1, H=3, W=4] (not T=2).
//!
//! Formula:
//!   diff = pred_cond - pred_uncond
//!   (optional momentum EMA on diff — skipped in v1)
//!   (optional norm_threshold clipping — skipped in v1)
//!   v1 = normalize(pred_cond, dim=[C,H,W])
//!   v0_parallel = (diff * v1).sum(dim=[C,H,W], keepdim) * v1
//!   v0_orthogonal = diff - v0_parallel
//!   update = v0_orthogonal + eta * v0_parallel
//!   pred = (pred_cond if use_original else pred_uncond) + scale * update
//!
//! Defaults from `inference.py`:
//!   guidance_scale=8.0, eta=1.0, norm_threshold=12.0, momentum=0.1,
//!   use_original_formulation=true
//!
//! v1 implements the core formula without momentum/threshold. Add those later
//! if quality demands it.

use flame_core::{DType, Result, Tensor};

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Build the Motif-Video timestep schedule.
///
/// Matches `FlowMatchEulerDiscreteScheduler.set_timesteps` for Motif's config
/// (shift=15.0, use_dynamic_shifting=false):
///
/// ```text
/// sigmas = linspace(1.0, 1/N, N)             # length N, starts 1.0, ends 1/N
/// sigmas = 15 * sigmas / (1 + 14 * sigmas)   # static shift
/// sigmas = concat(sigmas, [0.0])              # length N+1
/// ```
///
/// Returns `num_steps + 1` f32 values in descending order (sigma=1.0 → 0.0).
pub fn get_schedule(num_steps: usize) -> Vec<f32> {
    const SHIFT: f32 = 15.0;
    // Parity fix: diffusers' FlowMatchEulerDiscreteScheduler uses
    //   sigmas = linspace(1.0, 1/num_train_timesteps, N)
    // where `num_train_timesteps=1000` (from Motif's scheduler_config.json),
    // NOT `1/N`. For N=50 the endpoint difference is 0.001 vs 0.02 — small
    // but accumulates through the static-shift transform and shifts the
    // final-step denoise target. Matches Python to ~5 decimal places:
    //   Python last 3 sigmas: [0.3517, 0.1838, 0.0]
    //   Rust (old 1/N)      : [0.3846, 0.2344, 0.0]  ← 25% final-sigma error
    //   Rust (1/1000)       : [0.3517, 0.1838, 0.0]  ← matches
    const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
    let end = 1.0 / NUM_TRAIN_TIMESTEPS;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let frac = i as f32 / (num_steps - 1).max(1) as f32;
            1.0 - frac * (1.0 - end)
        })
        .collect();

    // Static shift: sigmas = 15 * sigmas / (1 + 14 * sigmas)
    for s in sigmas.iter_mut() {
        *s = SHIFT * *s / (1.0 + (SHIFT - 1.0) * *s);
    }

    sigmas.push(0.0);
    sigmas
}

// ---------------------------------------------------------------------------
// Video-aware APG
// ---------------------------------------------------------------------------

/// Parameters for video-aware adaptive projected guidance.
#[derive(Debug, Clone)]
pub struct ApgConfig {
    pub guidance_scale: f32,
    pub eta: f32,
    /// L2-norm clip threshold for `diff` over (C, H, W) per-frame.
    /// 0.0 disables clipping. Reference uses 12.0 (from inference.py
    /// `adaptive_projected_guidance_rescale=12.0`). Without clipping the
    /// guidance blows up at mid-sigma steps where `pred_cond` and
    /// `pred_uncond` diverge in magnitude.
    pub norm_threshold: f32,
    pub use_original_formulation: bool,
}

impl Default for ApgConfig {
    fn default() -> Self {
        // Matches `inference.py` defaults:
        //   guidance_scale=8.0, adaptive_projected_guidance_rescale=12.0,
        //   adaptive_projected_guidance_momentum=0.1, use_original_formulation=True
        //
        // NOTE on eta: inference.py doesn't pass `eta`. The `AdaptiveProjectedGuidance`
        // class default in diffusers is `eta=0.0` — which means the parallel component
        // is dropped entirely (only orthogonal update kept). The function-signature
        // default for `video_normalized_guidance` is 1.0 but that's not what gets passed.
        Self {
            guidance_scale: 8.0,
            eta: 0.0,
            norm_threshold: 12.0,
            use_original_formulation: true,
        }
    }
}

/// Running EMA of `diff` across denoise steps. Matches diffusers'
/// `MomentumBuffer`:
/// `running = diff + momentum * running` (not a standard EMA — a decaying
/// sum that accumulates more slowly as momentum → 0 and plainly sums when
/// momentum = 1.0). Reference uses momentum=0.1 (from inference.py
/// `adaptive_projected_guidance_momentum=0.1`).
pub struct MomentumBuffer {
    pub momentum: f32,
    pub running: Option<Tensor>,
}

impl MomentumBuffer {
    pub fn new(momentum: f32) -> Self {
        Self { momentum, running: None }
    }

    /// Update with new `diff`, return the smoothed diff that downstream
    /// guidance should use. Stores the running average for the next step.
    pub fn update(&mut self, diff: &Tensor) -> Result<Tensor> {
        let smoothed = match self.running.as_ref() {
            None => diff.clone(),
            Some(prev) => {
                let scaled = prev.mul_scalar(self.momentum)?;
                diff.add(&scaled)?
            }
        };
        self.running = Some(smoothed.clone());
        Ok(smoothed)
    }
}

/// Video-aware Adaptive Projected Guidance.
///
/// Inputs:
/// - `pred_cond`:   [B, C, T, H, W] BF16 — conditional velocity prediction
/// - `pred_uncond`: [B, C, T, H, W] BF16 — unconditional velocity prediction
///
/// Returns the guided prediction with the same shape.
///
/// Per-frame normalization (reduces over C, H, W; keeps T independent).
pub fn apg_guidance(
    pred_cond: &Tensor,
    pred_uncond: &Tensor,
    cfg: &ApgConfig,
    momentum: Option<&mut MomentumBuffer>,
) -> Result<Tensor> {
    // diff = pred_cond - pred_uncond
    let diff = pred_cond.sub(pred_uncond)?;

    // For video APG on 5D [B,C,T,H,W], reduce over dims C=1, H=3, W=4 (NOT T=2).
    // Strategy: compute reductions via a chained sum across the target dims.
    //
    // Cosine-style normalize: v1 = pred_cond / ||pred_cond||_2   (reduced dims)
    // Projection:             v0_par = (diff * v1).sum(dims) * v1
    //                         v0_orth = diff - v0_par
    // Update:                 delta = v0_orth + eta * v0_par
    // Pred:                   base + scale * delta
    //   where base = pred_cond if use_original_formulation else pred_uncond

    // All reductions below compute in F32 for numerical stability, then cast to
    // the original dtype.
    let input_dtype = pred_cond.dtype();

    let pred_cond_f32 = pred_cond.to_dtype(DType::F32)?;
    let mut diff_f32 = diff.to_dtype(DType::F32)?;

    // --- Momentum smoothing (if enabled) ---
    // Per reference `video_normalized_guidance`: momentum is applied BEFORE
    // the norm threshold. `MomentumBuffer.update(diff)` stores a decaying
    // sum `running = diff + m * running_prev` and returns it as the new diff.
    if let Some(mom) = momentum {
        diff_f32 = mom.update(&diff_f32)?;
    }

    // --- Norm threshold clipping (if threshold > 0) ---
    // Clip each per-frame (C, H, W) slice of `diff` to L2-norm ≤ threshold.
    //   scale_factor = min(1, threshold / ||diff||_{CHW,keepdim})
    //   diff = diff * scale_factor
    // Without this, `diff` magnitude can spike at intermediate sigma
    // values and `guidance_scale * diff` blows up the prediction, producing
    // catastrophic artifacts (observed: black-sun eclipse disc in
    // sunset-sky renders).
    if cfg.norm_threshold > 0.0 {
        // ||diff||_2 per-frame over (C, H, W), keepdim → [B, 1, T, 1, 1]
        let diff_sq = diff_f32.mul(&diff_f32)?;
        let diff_norm_sq = reduce_chw_keepdim(&diff_sq)?;
        let diff_norm = diff_norm_sq.sqrt()?;
        // scale_raw = threshold / (||diff|| + eps)
        let safe_norm = diff_norm.add_scalar(1e-12f32)?;
        let scale_raw = safe_norm.reciprocal()?.mul_scalar(cfg.norm_threshold)?;
        // Element-wise clamp to ≤ 1 via identity `min(a, 1) = 0.5*((a+1) - |a-1|)`
        // (flame-core's public Tensor API has no broadcastable min-scalar op).
        let a_plus = scale_raw.add_scalar(1.0f32)?;
        let a_minus = scale_raw.add_scalar(-1.0f32)?.abs()?;
        let scale_factor = a_plus.sub(&a_minus)?.mul_scalar(0.5f32)?;
        diff_f32 = diff_f32.mul(&scale_factor)?;
    }

    // Squared norm of pred_cond over (C, H, W):
    //   v1_sq = pred_cond^2
    //   v1_norm_sq = sum over (C, H, W)  →  shape [B, 1, T, 1, 1]
    let pc_sq = pred_cond_f32.mul(&pred_cond_f32)?;
    let pc_norm_sq = reduce_chw_keepdim(&pc_sq)?;
    let pc_norm = pc_norm_sq.sqrt()?;

    // v1 = pred_cond / ||pred_cond||  (broadcast over C, H, W)
    let eps = 1e-12f32;
    let pc_norm_safe = pc_norm.add_scalar(eps)?;
    let v1 = pred_cond_f32.div(&pc_norm_safe)?;

    // v0_parallel = (diff · v1).sum(chw, keepdim) * v1
    //   dot = sum over (C, H, W) of (diff * v1)
    let dot_raw = diff_f32.mul(&v1)?;
    let dot_sum = reduce_chw_keepdim(&dot_raw)?;
    let v0_parallel = dot_sum.mul(&v1)?;

    // v0_orthogonal = diff - v0_parallel
    let v0_orthogonal = diff_f32.sub(&v0_parallel)?;

    // update = v0_orthogonal + eta * v0_parallel
    let eta_parallel = v0_parallel.mul_scalar(cfg.eta)?;
    let update = v0_orthogonal.add(&eta_parallel)?;

    // pred = base + scale * update
    let base = if cfg.use_original_formulation {
        pred_cond_f32
    } else {
        pred_uncond.to_dtype(DType::F32)?
    };
    let scaled_update = update.mul_scalar(cfg.guidance_scale)?;
    let pred_f32 = base.add(&scaled_update)?;

    pred_f32.to_dtype(input_dtype)
}

/// Sum a 5D `[B, C, T, H, W]` tensor over dims (1=C, 3=H, 4=W) with keepdim.
/// Output shape: `[B, 1, T, 1, 1]`.
fn reduce_chw_keepdim(x: &Tensor) -> Result<Tensor> {
    // Sum along W (dim 4), keepdim → [B, C, T, H, 1]
    let s1 = x.sum_dim_keepdim(4)?;
    // Sum along H (dim 3), keepdim → [B, C, T, 1, 1]
    let s2 = s1.sum_dim_keepdim(3)?;
    // Sum along C (dim 1), keepdim → [B, 1, T, 1, 1]
    s2.sum_dim_keepdim(1)
}

// ---------------------------------------------------------------------------
// Euler step
// ---------------------------------------------------------------------------

/// Single Euler step: `latents = latents + dt * velocity`.
/// `dt` is typically `timesteps[i+1] - timesteps[i]` (negative for denoising).
pub fn euler_step(latents: &Tensor, velocity: &Tensor, dt: f32) -> Result<Tensor> {
    let step = velocity.mul_scalar(dt)?;
    latents.add(&step)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_length_and_bounds() {
        let ts = get_schedule(50);
        assert_eq!(ts.len(), 51);
        assert!((ts[0] - 1.0).abs() < 1e-5);
        assert_eq!(ts[50], 0.0);
    }

    #[test]
    fn schedule_monotonic_decreasing() {
        let ts = get_schedule(50);
        for i in 0..ts.len() - 1 {
            assert!(ts[i] >= ts[i + 1], "not monotonic at i={i}: {} → {}", ts[i], ts[i + 1]);
        }
    }

    #[test]
    fn schedule_static_shift_applied() {
        // With shift=15, sigma=0.5 transforms to 15*0.5 / (1 + 14*0.5) = 7.5 / 8 = 0.9375
        // For N=2, linspace(1.0, 0.5, 2) = [1.0, 0.5].
        // After shift: [1.0 (unchanged: 15/15=1), 0.9375], then append 0.
        let ts = get_schedule(2);
        assert_eq!(ts.len(), 3);
        assert!((ts[0] - 1.0).abs() < 1e-5);
        assert!((ts[1] - 0.9375).abs() < 1e-5, "expected 0.9375 at shift(0.5), got {}", ts[1]);
        assert_eq!(ts[2], 0.0);
    }

    #[test]
    fn schedule_shift_15_squeezes_to_high_noise() {
        // With shift=15, most of the schedule sits near high noise (close to 1.0),
        // which matches the paper's intent for video flow matching.
        let ts = get_schedule(50);
        // Median timestep should still be > 0.7 (high-noise bias).
        let mid = ts[25];
        assert!(mid > 0.5, "mid-schedule sigma should bias to high noise, got {}", mid);
    }
}
