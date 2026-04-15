//! ERNIE-Image sampling — FlowMatchEulerDiscreteScheduler with fixed shift=3.0.
//!
//! Config: num_train_timesteps=1000, shift=3.0, time_shift_type=exponential,
//! use_dynamic_shifting=false.

use flame_core::{Result, Tensor};

const SHIFT: f32 = 3.0;
const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Build sigma schedule for ERNIE-Image with exponential time shift.
///
/// Returns `num_steps + 1` values descending from ~1.0 to 0.0.
pub fn ernie_schedule(num_steps: usize) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let sigma = 1.0 - i as f32 / num_steps as f32; // linspace(1, 0)
        let shifted = if sigma <= 0.0 || sigma >= 1.0 {
            sigma
        } else {
            // exponential shift: shift / (shift + (1/sigma - 1))
            SHIFT / (SHIFT + (1.0 / sigma - 1.0))
        };
        sigmas.push(shifted);
    }
    sigmas
}

// ---------------------------------------------------------------------------
// Timestep conversion
// ---------------------------------------------------------------------------

/// Convert sigma to model timestep: sigma * num_train_timesteps.
pub fn sigma_to_timestep(sigma: f32) -> f32 {
    sigma * NUM_TRAIN_TIMESTEPS
}

// ---------------------------------------------------------------------------
// Euler step
// ---------------------------------------------------------------------------

/// One Euler ODE step for flow-matching velocity prediction.
///
/// `dt = sigma_next - sigma` (negative since sigmas decrease).
/// `x_next = x + dt * pred`
pub fn ernie_euler_step(
    x: &Tensor,
    pred: &Tensor,
    sigma: f32,
    sigma_next: f32,
) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_length() {
        let s = ernie_schedule(30);
        assert_eq!(s.len(), 31);
    }

    #[test]
    fn test_schedule_endpoints() {
        let s = ernie_schedule(30);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
    }

    #[test]
    fn test_schedule_monotonic() {
        let s = ernie_schedule(30);
        for i in 0..s.len() - 1 {
            assert!(s[i] >= s[i + 1], "must decrease: s[{}]={} < s[{}]={}", i, s[i], i + 1, s[i + 1]);
        }
    }

    #[test]
    fn test_shift_midpoint() {
        // At sigma=0.5: shift / (shift + (1/0.5 - 1)) = 3 / (3 + 1) = 0.75
        let s = ernie_schedule(2); // [1.0, shifted(0.5), 0.0]
        assert!((s[1] - 0.75).abs() < 1e-6, "midpoint should be 0.75, got {}", s[1]);
    }

    #[test]
    fn test_sigma_to_timestep() {
        assert!((sigma_to_timestep(1.0) - 1000.0).abs() < 1e-6);
        assert!((sigma_to_timestep(0.5) - 500.0).abs() < 1e-6);
        assert!((sigma_to_timestep(0.0) - 0.0).abs() < 1e-6);
    }
}
