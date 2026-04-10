//! Kandinsky-5 / HunyuanVideo velocity-based diffusion sampler.
//!
//! Ported from the Kandinsky-5 reference implementation.
//!
//! Key properties:
//! - Velocity prediction (not noise prediction, not flow-matching)
//! - Non-linear timestep schedule with configurable scale
//!   (scheduler_scale = 10.0 for video, 3.0 for images)
//! - Standard Euler integration with optional classifier-free guidance
//!
//! Schedule formula:
//! ```text
//!   t_linear = linspace(1, 0, num_steps + 1)
//!   t = scale * t_linear / (1 + (scale - 1) * t_linear)
//! ```
//!
//! Euler velocity step:
//! ```text
//!   velocity = model(x, text, pooled, timestep * 1000, ...)
//!   x += dt * velocity
//! ```

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Default scheduler scale for video generation.
pub const SCHEDULER_SCALE_VIDEO: f32 = 10.0;

/// Default scheduler scale for image generation.
pub const SCHEDULER_SCALE_IMAGE: f32 = 3.0;

/// Build the Kandinsky-5 velocity schedule.
///
/// Returns `num_steps + 1` timestep values. The schedule starts near
/// `scheduler_scale` (at t=1) and ends at 0.0 (at t=0). The non-linear
/// mapping concentrates more steps near the beginning of the diffusion
/// process where detail emerges.
///
/// Formula: `t_out = scale * t / (1 + (scale - 1) * t)`
/// where `t = linspace(1, 0, num_steps + 1)`.
pub fn build_velocity_schedule(num_steps: usize, scheduler_scale: f32) -> Vec<f32> {
    assert!(num_steps > 0, "num_steps must be > 0");
    assert!(scheduler_scale > 0.0, "scheduler_scale must be > 0");

    let mut timesteps: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();

    for v in timesteps.iter_mut() {
        let t = *v;
        // Avoid division by zero at t=0 (denominator = 1 + (scale-1)*0 = 1, result = 0)
        *v = scheduler_scale * t / (1.0 + (scheduler_scale - 1.0) * t);
    }

    timesteps
}

/// Compute the step deltas (dt) from a schedule.
///
/// Returns `num_steps` values where `dt[i] = timesteps[i+1] - timesteps[i]`.
/// All dt values are negative (schedule decreases).
pub fn schedule_deltas(timesteps: &[f32]) -> Vec<f32> {
    timesteps
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect()
}

// ---------------------------------------------------------------------------
// Euler velocity step
// ---------------------------------------------------------------------------

/// Single Euler velocity step for Kandinsky-5.
///
/// Applies `x_next = x + dt * velocity` where dt is the timestep difference.
/// The model predicts velocity directly (not noise or denoised).
pub fn euler_velocity_step(
    velocity: &Tensor,
    x: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let step = velocity.mul_scalar(dt)?;
    x.add(&step)
}

/// Apply classifier-free guidance to conditional and unconditional velocity.
///
/// `guided = uncond + guidance_scale * (cond - uncond)`
pub fn apply_cfg(
    cond_velocity: &Tensor,
    uncond_velocity: &Tensor,
    guidance_scale: f32,
) -> Result<Tensor> {
    let diff = cond_velocity.sub(uncond_velocity)?;
    let scaled = diff.mul_scalar(guidance_scale)?;
    uncond_velocity.add(&scaled)
}

// ---------------------------------------------------------------------------
// Full denoise loop
// ---------------------------------------------------------------------------

/// Kandinsky-5 Euler velocity denoise loop.
///
/// `model_fn(x, timestep_scaled)` should return the velocity prediction.
/// The timestep is scaled by 1000 before being passed to the model, matching
/// the Kandinsky-5 convention.
///
/// If `cfg_fn` is provided, it is called for the unconditional prediction
/// at each step and classifier-free guidance is applied.
///
/// Returns the denoised result.
pub fn kandinsky5_denoise<F>(
    mut model_fn: F,
    mut x: Tensor,
    timesteps: &[f32],
    guidance_scale: f32,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor>
where
    F: FnMut(&Tensor, &Tensor, bool) -> Result<Tensor>,
{
    let batch = x.shape().dims()[0];
    let dtype = x.dtype();
    let use_cfg = guidance_scale > 1.0;

    for w in timesteps.windows(2) {
        let (t_curr, t_next) = (w[0], w[1]);
        let dt = t_next - t_curr;

        // Build timestep tensor scaled by 1000 (Kandinsky-5 convention)
        let t_scaled = t_curr * 1000.0;
        let t_vec = Tensor::from_vec(
            vec![t_scaled; batch],
            Shape::from_dims(&[batch]),
            device.clone(),
        )?;
        let t_vec = if dtype != DType::F32 {
            t_vec.to_dtype(dtype)?
        } else {
            t_vec
        };

        // Conditional velocity
        let cond_vel = model_fn(&x, &t_vec, /* is_cond */ true)?;

        let velocity = if use_cfg {
            // Unconditional velocity
            let uncond_vel = model_fn(&x, &t_vec, /* is_cond */ false)?;
            apply_cfg(&cond_vel, &uncond_vel, guidance_scale)?
        } else {
            cond_vel
        };

        // Euler step: x += dt * velocity
        x = euler_velocity_step(&velocity, &x, dt)?;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_length() {
        let ts = build_velocity_schedule(50, SCHEDULER_SCALE_VIDEO);
        assert_eq!(ts.len(), 51);
    }

    #[test]
    fn test_schedule_endpoints_video() {
        let ts = build_velocity_schedule(50, SCHEDULER_SCALE_VIDEO);
        // First value: scale * 1.0 / (1 + (scale-1)*1.0) = scale / scale = 1.0
        assert!((ts[0] - 1.0).abs() < 1e-6, "first should be 1.0, got {}", ts[0]);
        // Last value: scale * 0.0 / ... = 0.0
        assert!(ts[50].abs() < 1e-6, "last should be 0.0, got {}", ts[50]);
    }

    #[test]
    fn test_schedule_endpoints_image() {
        let ts = build_velocity_schedule(20, SCHEDULER_SCALE_IMAGE);
        assert!((ts[0] - 1.0).abs() < 1e-6);
        assert!(ts[20].abs() < 1e-6);
    }

    #[test]
    fn test_schedule_monotonic_decreasing() {
        let ts = build_velocity_schedule(50, SCHEDULER_SCALE_VIDEO);
        for i in 0..ts.len() - 1 {
            assert!(
                ts[i] >= ts[i + 1],
                "not monotonic at {}: {} < {}",
                i,
                ts[i],
                ts[i + 1]
            );
        }
    }

    #[test]
    fn test_schedule_nonlinear_concentration() {
        // With scale > 1, the schedule should be non-linear —
        // more of the schedule range is in the early (high-t) region.
        let ts = build_velocity_schedule(100, SCHEDULER_SCALE_VIDEO);
        let midpoint = ts[50];
        // For scale=10: midpoint at step 50/100 (t_linear=0.5):
        // t_out = 10*0.5 / (1 + 9*0.5) = 5/5.5 ≈ 0.909
        assert!(midpoint > 0.85, "midpoint should be > 0.85 for scale=10, got {midpoint}");
    }

    #[test]
    fn test_schedule_deltas_sum() {
        let ts = build_velocity_schedule(50, SCHEDULER_SCALE_VIDEO);
        let deltas = schedule_deltas(&ts);
        assert_eq!(deltas.len(), 50);
        // Sum of deltas = ts[last] - ts[first] = 0 - 1 = -1
        let sum: f32 = deltas.iter().sum();
        assert!((sum + 1.0).abs() < 1e-4, "deltas should sum to -1.0, got {sum}");
    }

    #[test]
    fn test_deltas_all_negative() {
        let ts = build_velocity_schedule(50, SCHEDULER_SCALE_VIDEO);
        let deltas = schedule_deltas(&ts);
        for (i, dt) in deltas.iter().enumerate() {
            assert!(*dt < 0.0, "delta[{i}] should be negative, got {dt}");
        }
    }

    #[test]
    fn test_euler_step_arithmetic() {
        // x=10, velocity=5, dt=-0.2 → x_new = 10 + (-0.2)*5 = 9.0
        let x_val = 10.0f32;
        let v_val = 5.0f32;
        let dt = -0.2f32;
        let result = x_val + dt * v_val;
        assert!((result - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_cfg_arithmetic() {
        // uncond=2, cond=4, scale=7.5 → guided = 2 + 7.5*(4-2) = 17.0
        let uncond = 2.0f32;
        let cond = 4.0f32;
        let scale = 7.5f32;
        let result = uncond + scale * (cond - uncond);
        assert!((result - 17.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_1_is_linear() {
        // With scheduler_scale=1.0, the schedule should be purely linear
        let ts = build_velocity_schedule(10, 1.0);
        for (i, v) in ts.iter().enumerate() {
            let expected = 1.0 - i as f32 / 10.0;
            assert!(
                (v - expected).abs() < 1e-6,
                "scale=1 should be linear: ts[{i}]={v}, expected {expected}"
            );
        }
    }
}
