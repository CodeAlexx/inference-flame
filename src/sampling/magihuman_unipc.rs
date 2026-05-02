//! FlowUniPCMultistepScheduler — distill-mode `step_ddim` shortcut.
//!
//! Port of the path used by MagiHuman distill (`cfg_number=1`):
//!   * Build a 1000-step linear sigma schedule, apply Wan flow shift.
//!   * Subsample to N inference steps via `linspace(0, 1000, N+1)`.
//!   * `step_ddim`: deterministic-on-final-step DDIM with stochastic
//!     resample using `prev_t * randn + (1 - prev_t) * (curr_state - curr_t * v)`.
//!
//! Reference: `inference/pipeline/scheduler_unipc.py` `FlowUniPCMultistepScheduler`.
//! `step` (UniPC) and `step_sde` (SDE) paths are not ported — only `step_ddim`.

use flame_core::{Error, Result, Shape, Tensor};
use std::sync::Arc;

/// Distill scheduler — holds the precomputed sigma table and current step
/// index. Mirror of the reference's `FlowUniPCMultistepScheduler` for the
/// distill `step_ddim` path only.
pub struct FlowUniPcDDim {
    /// Sigma values in F32, length `num_inference_steps + 1`. The last entry
    /// is 0 (clean).
    pub sigmas: Vec<f32>,
    /// Timesteps in [0, 1000), length `num_inference_steps`. Used by the
    /// caller to gate CFG (e.g. `t > 500`).
    pub timesteps: Vec<f32>,
    pub num_train_timesteps: usize,
    pub num_inference_steps: usize,
    pub shift: f32,
}

impl FlowUniPcDDim {
    /// Construct a fresh scheduler. Mirrors `set_timesteps` from reference.
    /// `shift=5.0` and `num_train_timesteps=1000` are MagiHuman distill
    /// defaults.
    pub fn new(num_inference_steps: usize, shift: f32, num_train_timesteps: usize) -> Self {
        let n_train = num_train_timesteps;
        let n_inf = num_inference_steps;
        let shift_f64 = shift as f64;

        // Reference Python (in `__init__`):
        //   alphas = np.linspace(1, 1/num_train, num_train)[::-1]   # ascending [1/N..1]
        //   sigmas = 1.0 - alphas                                    # descending [(N-1)/N..0]
        //   sigmas = shift * sigmas / (1 + (shift-1) * sigmas)      # Wan flow shift
        //   sigma_max = sigmas[0]
        //   sigma_min = sigmas[-1] = 0
        //
        // Reference `set_timesteps(n_inf, shift=5.0)`:
        //   sigmas = np.linspace(sigma_max, sigma_min, n_inf+1)[:-1]   # n_inf descending
        //   sigmas = shift * sigmas / (1 + (shift-1) * sigmas)         # SHIFT APPLIED AGAIN
        //   timesteps = sigmas * num_train_timesteps                   # int timesteps
        //   sigmas = concat([sigmas, [0]])                              # append 0 for final
        // The reference instantiates with default `shift=1.0` in `__init__`
        // (identity), and overrides via `shift=5.0` to `set_timesteps`. So the
        // FULL sigma table is unshifted; only the linspace result here is
        // shifted (single application).
        let sigma_max = (n_train - 1) as f64 / n_train as f64;
        let sigma_min = 0.0_f64;

        // linspace(sigma_max, sigma_min, n_inf+1)[:-1] → n_inf descending values
        let mut sigmas_lin: Vec<f64> = Vec::with_capacity(n_inf);
        for i in 0..n_inf {
            let t = sigma_max + (sigma_min - sigma_max) * (i as f64) / (n_inf as f64);
            sigmas_lin.push(t);
        }
        // Apply shift AGAIN
        let sigmas_shifted: Vec<f64> = sigmas_lin
            .iter()
            .map(|&x| shift_f64 * x / (1.0 + (shift_f64 - 1.0) * x))
            .collect();
        // Timesteps = sigmas * num_train_timesteps, cast to int64 (truncation).
        let timesteps: Vec<f32> = sigmas_shifted
            .iter()
            .map(|&s| (s * (n_train as f64)) as i64 as f32)
            .collect();
        // Final sigma table: shifted sigmas with 0 appended.
        let mut sigmas: Vec<f32> = sigmas_shifted.iter().map(|&x| x as f32).collect();
        sigmas.push(0.0);

        FlowUniPcDDim {
            sigmas,
            timesteps,
            num_train_timesteps: n_train,
            num_inference_steps: n_inf,
            shift,
        }
    }

    /// step_ddim from reference (line 708):
    ///     curr_t = sigmas[t]
    ///     prev_t = sigmas[t + 1]
    ///     noise  = randn_like(curr_state)
    ///     cur_clean = curr_state - curr_t * velocity
    ///     prev_state = prev_t * noise + (1 - prev_t) * cur_clean
    ///
    /// `idx` is the step index `[0, num_inference_steps)`.
    /// `noise` should be a freshly-sampled standard-normal tensor matching
    /// `curr_state.shape()` and `curr_state.dtype()` (caller controls RNG so
    /// the test fixture can replay it).
    pub fn step_ddim(
        &self,
        velocity: &Tensor,
        idx: usize,
        curr_state: &Tensor,
        noise: &Tensor,
    ) -> Result<Tensor> {
        if idx >= self.num_inference_steps {
            return Err(Error::InvalidOperation(format!(
                "step_ddim: idx {idx} out of range (num_inference_steps={})",
                self.num_inference_steps
            )));
        }
        let curr_t = self.sigmas[idx];
        let prev_t = self.sigmas[idx + 1];

        // cur_clean = curr_state - curr_t * velocity
        let v_scaled = velocity.mul_scalar(curr_t)?;
        let cur_clean = curr_state.sub(&v_scaled)?;
        // prev_state = prev_t * noise + (1 - prev_t) * cur_clean
        let term_noise = noise.mul_scalar(prev_t)?;
        let term_clean = cur_clean.mul_scalar(1.0 - prev_t)?;
        term_noise.add(&term_clean)
    }

    /// Convenience: build a noise tensor matching `curr_state` for
    /// step_ddim. Uses flame's host RNG via Tensor::randn for a quick
    /// in-port impl (NOT used by the parity oracle which controls noise
    /// externally).
    pub fn _example_noise(
        shape: &Shape,
        dtype: flame_core::DType,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Tensor> {
        // Just allocate zeros — caller is expected to pass real noise.
        Tensor::zeros_dtype(shape.clone(), dtype, device.clone())
    }
}
