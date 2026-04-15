//! Flow-matching Euler sampler for Wan2.2 TI2V-5B.
//!
//! Matches the scheduler in `wan/modules/model.py` at inference time:
//! - uniform `sigma ∈ [sigma_max, sigma_min]` over `sample_steps` points,
//!   terminal appended as 0.0,
//! - shift applied as `sigma' = shift * sigma / (1 + (shift-1) * sigma)`,
//! - step delta `Δ = sigma_{i+1} - sigma_i`,
//! - Euler step: `latent ← latent + Δ · model(latent, t=sigma*1000, c)`.
//!
//! The Euler solver is deterministic; the only RNG is the initial
//! Gaussian noise (generated in the binary before the first step).

use flame_core::{Result, Tensor};

/// Number of training timesteps in Wan's scheduler.
pub const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// Sigma schedule + corresponding integer-ish timesteps.
pub struct EulerSigmas {
    pub sigmas: Vec<f32>,     // length = num_steps + 1
    pub timesteps: Vec<f32>,  // length = num_steps + 1, = sigmas * 1000
}

/// Build a shifted flow-matching sigma schedule.
///
/// `shift` is `sample_shift` from the config (`5.0` for TI2V-5B).
pub fn shifted_sigma_schedule(num_steps: usize, shift: f32) -> EulerSigmas {
    assert!(num_steps >= 1);
    let sigma_max = 1.0 - 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let sigma_min = 1.0 / NUM_TRAIN_TIMESTEPS as f32;

    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / num_steps as f32;
            sigma_max + t * (sigma_min - sigma_max)
        })
        .collect();
    for s in sigmas.iter_mut() {
        *s = shift * *s / (1.0 + (shift - 1.0) * *s);
    }
    sigmas.push(0.0);

    let timesteps: Vec<f32> = sigmas
        .iter()
        .map(|s| s * NUM_TRAIN_TIMESTEPS as f32)
        .collect();

    EulerSigmas { sigmas, timesteps }
}

/// One Euler step in place:
///   `latent ← latent + (sigma_next - sigma) · noise_pred`
pub fn euler_step(latent: &Tensor, noise_pred: &Tensor, dt: f32) -> Result<Tensor> {
    let delta = noise_pred.mul_scalar(dt)?;
    latent.add(&delta)
}

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`.
pub fn cfg_combine(
    cond: &Tensor,
    uncond: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let diff = cond.sub(uncond)?;
    let scaled = diff.mul_scalar(scale)?;
    uncond.add(&scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_monotonic() {
        let s = shifted_sigma_schedule(50, 5.0);
        assert_eq!(s.sigmas.len(), 51);
        for i in 1..s.sigmas.len() {
            assert!(
                s.sigmas[i] <= s.sigmas[i - 1] + 1e-6,
                "sigmas must be non-increasing: {} -> {}",
                s.sigmas[i - 1],
                s.sigmas[i],
            );
        }
        assert!((s.sigmas.last().unwrap() - 0.0).abs() < 1e-6);
        // First step's shifted sigma must be less than sigma_max (shift > 1 drags it up)
        assert!(s.sigmas[0] > 0.5);
        assert!(s.sigmas[0] < 1.0);
    }

    #[test]
    fn timesteps_track_sigmas() {
        let s = shifted_sigma_schedule(10, 5.0);
        for (sig, ts) in s.sigmas.iter().zip(s.timesteps.iter()) {
            assert!((sig * 1000.0 - ts).abs() < 1e-3);
        }
    }
}
