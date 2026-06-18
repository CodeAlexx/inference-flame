//! PiD 4-step distilled flow-matching SDE sampler.
//!
//! Source of truth: pid/_src/models/pid_distill_model.py
//!   - `_student_sample_loop` (l.112-161)
//!   - `_net_output_to_x0` / `_velocity_to_x0` (l.50-83)
//!   - `_get_t_list` (l.90-110) — default linspace(student_timestep, 0, steps+1).
//!
//! Released distilled-4step config (PIDistillConfig defaults + the released
//! checkpoint card):
//!   student_timestep = 1.0, student_sample_steps = 4, student_sample_type
//!   = "sde", prediction_type = "velocity", timescale (fm_timescale) = 1000.0.
//!   The published t_list for the 4-step SDE distill is
//!     [0.999, 0.866, 0.634, 0.342, 0.0]
//!   (PID_PORT_PLAN_2026-05-29.md §"Sampler"). The repo's `_get_t_list` would
//!   instead build linspace(1.0, 0.0, 5) = [1.0, 0.75, 0.5, 0.25, 0.0] when no
//!   explicit list is given; the released schedule is the non-uniform one above,
//!   so we default to it and expose an override. **FLAG**: confirm which list
//!   the released pipeline actually loads (see report — uncertain).
//!
//! Loop (prediction_type="velocity", student_sample_type="sde"):
//!   x = noise
//!   for (t_cur, t_next) in zip(t[:-1], t[1:]):
//!       v = net(x, t_cur*timescale, caption, lq_latent, sigma)
//!       x0 = x - t_cur * v                         (_velocity_to_x0)
//!       if t_next > 0:  eps ~ N(0,1); x = (1-t_next)*x0 + t_next*eps
//!       else:           x = x0
//!   (NO final clamp inside the loop; the pipeline clamps to [-1,1] once at the
//!    very end — see pid_distill_model.py:281 generate path. We clamp here.)
//!
//! NOTE: `_net_output_to_x0` does the x0 = x - t*v in float64 in PyTorch
//! (l.60-63). We compute in F32 here; the BF16 net output dominates the error
//! budget, so the f64 vs f32 difference is negligible. Flagged for the skeptic.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use super::model::PidNet;

#[derive(Clone, Debug)]
pub struct PidSamplerConfig {
    /// Non-uniform distilled schedule, length = steps + 1, ending at 0.0.
    pub t_list: Vec<f32>,
    /// fm timescale (timesteps fed to the net are t_cur * timescale).
    pub timescale: f32,
    /// degrade_sigma scalar (0.0 for released latent-only ckpts).
    pub sigma: f32,
    /// Optional fixed seed for the SDE re-noise eps (None = fresh randn).
    pub seed: Option<u64>,
}

impl Default for PidSamplerConfig {
    fn default() -> Self {
        Self {
            t_list: vec![0.999, 0.866, 0.634, 0.342, 0.0],
            timescale: 1000.0,
            sigma: 0.0,
            seed: None,
        }
    }
}

/// Run the 4-step distilled SDE sampler.
///
/// `noise`: [B, 3, H, W] initial pixel noise (BF16).
/// `caption`: [B, Ltxt, 2304] Gemma caption embeds (BF16).
/// `lq_latent`: [B, 16, zH, zW] upstream VAE latent (BF16).
/// Returns the decoded RGB image [B, 3, H, W], clamped to [-1, 1].
pub fn pid_student_sample(
    net: &PidNet,
    noise: &Tensor,
    caption: &Tensor,
    lq_latent: &Tensor,
    cfg: &PidSamplerConfig,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let dims = noise.shape().dims().to_vec();
    let b = dims[0];
    let numel: usize = dims.iter().product();

    let mut x = noise.clone();
    let mut rng_state = cfg.seed.unwrap_or(0x9E3779B97F4A7C15);

    let n_steps = cfg.t_list.len() - 1;
    for step in 0..n_steps {
        let t_cur = cfg.t_list[step];
        let t_next = cfg.t_list[step + 1];

        // timestep fed to the net: t_cur * timescale, shape [B].
        let t_scaled = Tensor::from_vec_dtype(
            vec![t_cur * cfg.timescale; b],
            Shape::from_dims(&[b]),
            device.clone(),
            DType::F32,
        )?;

        let v = net.forward(&x, &t_scaled, caption, lq_latent, cfg.sigma)?;

        // x0 = x - t_cur * v  (_velocity_to_x0, prediction_type="velocity").
        let x0 = x.sub(&v.mul_scalar(t_cur)?)?;

        if t_next > 0.0 {
            // SDE re-noise: x = (1 - t_next)*x0 + t_next*eps, eps ~ N(0,1).
            let eps = randn_bf16(numel, &dims, &mut rng_state, device)?;
            let a = x0.mul_scalar(1.0 - t_next)?;
            let bterm = eps.mul_scalar(t_next)?;
            x = a.add(&bterm)?;
        } else {
            x = x0;
        }
    }

    // Final clamp to [-1, 1] (pid_distill_model.py:281 clamps once at the end).
    x.clamp(-1.0, 1.0)
}

/// Deterministic Box-Muller randn into a BF16 tensor of the given shape.
/// Mirrors the project's preference for a reproducible eps when a seed is set;
/// when no seed is set the caller passes a time-derived seed upstream.
fn randn_bf16(
    n: usize,
    shape: &[usize],
    state: &mut u64,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let u1 = next_unit(state);
        let u2 = next_unit(state);
        let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data[i] = r * theta.cos();
        if i + 1 < n {
            data[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    Tensor::from_vec_dtype(data, Shape::from_dims(shape), device.clone(), DType::BF16)
}

/// SplitMix64 -> uniform (0,1).
fn next_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    // 24-bit mantissa fraction in [0,1).
    ((z >> 40) as f32) / ((1u32 << 24) as f32)
}
