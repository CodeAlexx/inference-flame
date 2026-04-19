//! DDPM Wuerstchen scheduler — continuous-time DDPM used by Stable Cascade.
//!
//! Reference: `diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler`.
//!
//! The scheduler uses a continuous `t in [0, 1]` and a cosine alpha-cumprod schedule:
//!
//! ```text
//! f(t)         = cos((t + s) / (1 + s) * pi/2)^2
//! alpha_cp(t)  = f(t) / f(0),  clamped to [1e-4, 1 - 1e-4]
//! ```
//!
//! `set_timesteps(num_inference_steps)` produces a linearly spaced sequence
//! `[1.0, ..., 0.0]` of length `num_inference_steps + 1`. The denoise loop
//! walks pairs `(t_cur, t_next)`.
//!
//! `step(model_output, t, sample)` follows the diffusers formulation:
//!
//! ```text
//! alpha_cp_t    = _alpha_cumprod(t)
//! alpha_cp_prev = _alpha_cumprod(t_next)
//! alpha         = alpha_cp_t / alpha_cp_prev
//! mu            = sqrt(1/alpha) * (x_t - (1 - alpha) * model_out / sqrt(1 - alpha_cp_t))
//! std           = sqrt((1 - alpha) * (1 - alpha_cp_prev) / (1 - alpha_cp_t))
//! x_{t-1}       = mu + std * N(0,1)    (no noise when t_next == 0)
//! ```
//!
//! The scheduler expects the model to predict `v`-style output in this
//! formulation, matching diffusers' reference.

use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::sync::Arc;

/// Config + state for the Würstchen DDPM scheduler.
pub struct DDPMWuerstchenScheduler {
    /// Number of inference timesteps + 1. `timesteps[0] = 1.0`, `timesteps[-1] = 0.0`.
    pub timesteps: Vec<f32>,
    /// Cosine `s` offset. Default 0.008 (matches diffusers).
    pub s: f32,
    /// `scaler` parameter. Default 1.0 (matches Stable Cascade config).
    pub scaler: f32,
    /// Cached `f(0) = cos(s / (1 + s) * pi/2)^2`.
    pub init_alpha_cumprod: f32,
}

impl DDPMWuerstchenScheduler {
    /// Build a scheduler with `num_inference_steps` equispaced timesteps on
    /// `[1, 0]`. The final scheduler `step()` that targets `t_next == 0` omits
    /// the noise term, matching diffusers.
    pub fn new(num_inference_steps: usize) -> Self {
        let s = 0.008f32;
        let scaler = 1.0f32;
        let init_alpha_cumprod = {
            let x = s / (1.0 + s) * std::f32::consts::FRAC_PI_2;
            let c = x.cos();
            c * c
        };
        // timesteps: linspace(1.0, 0.0, num_inference_steps + 1)
        let n = num_inference_steps + 1;
        let mut timesteps = Vec::with_capacity(n);
        for i in 0..n {
            let v = 1.0 - (i as f32) / (num_inference_steps as f32);
            timesteps.push(v);
        }
        // Ensure exact endpoints (1.0 and 0.0).
        timesteps[0] = 1.0;
        if let Some(last) = timesteps.last_mut() {
            *last = 0.0;
        }
        Self {
            timesteps,
            s,
            scaler,
            init_alpha_cumprod,
        }
    }

    /// The "denoising" timesteps: `timesteps[0..num_inference_steps]`. These
    /// are the `t` values the model is called at (the last one, `0.0`, is the
    /// target and NOT a model-call timestep).
    pub fn model_timesteps(&self) -> &[f32] {
        let n = self.timesteps.len() - 1;
        &self.timesteps[..n]
    }

    /// `t -> alpha_cumprod(t)` (scalar).
    pub fn alpha_cumprod_scalar(&self, t: f32) -> f32 {
        let t = if self.scaler > 1.0 {
            1.0 - (1.0 - t).powf(self.scaler)
        } else if self.scaler < 1.0 {
            t.powf(self.scaler)
        } else {
            t
        };
        let x = (t + self.s) / (1.0 + self.s) * std::f32::consts::FRAC_PI_2;
        let c = x.cos();
        let v = c * c / self.init_alpha_cumprod;
        v.clamp(1e-4, 1.0 - 1e-4)
    }

    /// Euler-like step for eps-prediction (deterministic DDIM-style):
    /// Recover x0 from (x_t, eps), then forward-noise to t_next.
    pub fn step_eps_ddim(
        &self,
        eps: &Tensor,
        t: f32,
        t_next: f32,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let a = self.alpha_cumprod_scalar(t);
        let a_prev = self.alpha_cumprod_scalar(t_next);
        let sqrt_a = a.sqrt();
        let sqrt_1ma = (1.0 - a).sqrt();
        // x0 = (x_t - sqrt(1-a)*eps) / sqrt(a)
        let x0 = sample.sub(&eps.mul_scalar(sqrt_1ma)?)?.mul_scalar(1.0 / sqrt_a)?;
        // x_{t_next} = sqrt(a_prev)*x0 + sqrt(1 - a_prev)*eps
        let out = x0
            .mul_scalar(a_prev.sqrt())?
            .add(&eps.mul_scalar((1.0 - a_prev).sqrt())?)?;
        Ok(out)
    }

    /// Euler-like step for v-prediction: x_next = sqrt(alpha_cp_prev)*x0 + sqrt(1-alpha_cp_prev)*eps
    /// where x0 and eps are recovered from x and v via the continuous-time relations.
    /// (See: "Progressive Distillation for Fast Sampling" Salimans & Ho, Eq 12-14.)
    pub fn step_v_prediction(
        &self,
        v: &Tensor,
        t: f32,
        t_next: f32,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let a = self.alpha_cumprod_scalar(t);
        let a_prev = self.alpha_cumprod_scalar(t_next);
        let sqrt_a = a.sqrt();
        let sqrt_1ma = (1.0 - a).sqrt();
        // v = sqrt(a)*eps - sqrt(1-a)*x0
        // x_t = sqrt(a)*x0 + sqrt(1-a)*eps
        // Solve:
        //   x0  = sqrt(a)*x_t - sqrt(1-a)*v
        //   eps = sqrt(1-a)*x_t + sqrt(a)*v
        let x0 = sample.mul_scalar(sqrt_a)?.sub(&v.mul_scalar(sqrt_1ma)?)?;
        let eps = sample.mul_scalar(sqrt_1ma)?.add(&v.mul_scalar(sqrt_a)?)?;
        // x_{t_next} = sqrt(a_prev)*x0 + sqrt(1-a_prev)*eps
        let out = x0
            .mul_scalar(a_prev.sqrt())?
            .add(&eps.mul_scalar((1.0 - a_prev).sqrt())?)?;
        Ok(out)
    }

    /// Step: compute `x_{t_next}` from `model_output` at `x_t`.
    ///
    /// `t` is the current timestep, `t_next` is the next (smaller) timestep.
    /// If `t_next == 0.0`, no noise is added.
    ///
    /// Implements diffusers' `DDPMWuerstchenScheduler.step()` elementwise.
    pub fn step(
        &self,
        model_output: &Tensor,
        t: f32,
        t_next: f32,
        sample: &Tensor,
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let alpha_cp_t = self.alpha_cumprod_scalar(t);
        let alpha_cp_prev = self.alpha_cumprod_scalar(t_next);
        let alpha = alpha_cp_t / alpha_cp_prev;

        // mu = (1/sqrt(alpha)) * (x - (1 - alpha) * mo / sqrt(1 - alpha_cp_t))
        let one_minus_alpha = 1.0 - alpha;
        let sqrt_one_minus_cp_t = (1.0f32 - alpha_cp_t).sqrt();
        let scale_mo = one_minus_alpha / sqrt_one_minus_cp_t; // scalar
        let inv_sqrt_alpha = 1.0 / alpha.sqrt();

        // term = sample - scale_mo * model_output
        let term = sample.sub(&model_output.mul_scalar(scale_mo)?)?;
        let mu = term.mul_scalar(inv_sqrt_alpha)?;

        if t_next <= 0.0 {
            return Ok(mu);
        }

        // std = sqrt((1 - alpha) * (1 - alpha_cp_prev) / (1 - alpha_cp_t))
        let std_val = (one_minus_alpha * (1.0 - alpha_cp_prev) / (1.0 - alpha_cp_t)).sqrt();

        // If caller provides noise, use that; else return mu only (deterministic).
        // Diffusers always adds stochastic noise here. For faithful inference
        // the caller should pass noise; None is provided only for debugging.
        if let Some(n) = noise {
            let mu_plus = mu.add(&n.mul_scalar(std_val)?)?;
            Ok(mu_plus)
        } else {
            Ok(mu)
        }
    }
}

/// Sample standard normal noise of the given shape on GPU as BF16.
pub fn randn_bf16(shape: &[usize], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, device.clone())?;
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        t.to_dtype(DType::BF16)
    }
}
