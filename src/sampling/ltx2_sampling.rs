//! LTX-2.3 video sampling — Euler velocity step with distilled sigma schedule.
//!
//! Distilled model uses 8 fixed steps, no CFG (guidance_scale=1.0).
//! Flow matching: denoised = x - model_output * sigma.

use flame_core::{Result, Tensor};

/// Distilled 8-step sigma schedule (hardcoded from Lightricks).
pub const LTX2_DISTILLED_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];

/// Stage 2 refinement sigmas for two-stage pipeline (3 denoise steps).
/// Applied after spatial upsampling to refine the upscaled latent.
pub const LTX2_STAGE2_DISTILLED_SIGMAS: [f32; 4] = [
    0.909375, 0.725, 0.421875, 0.0,
];

/// LinearQuadratic sigma schedule — direct port of Lightricks's
/// `linear_quadratic_schedule` in `ltx_video/schedulers/rf.py`. This is
/// the schedule all 0.9.8-dev yamls specify (`sampler: LinearQuadratic,
/// threshold_noise: 0.025`) and the one every dev-mode LTX-2 gen by
/// Lightricks actually runs. Parity-verified by `bin/ltx2_sigma_parity`
/// against their imported Python function.
///
/// Returns exactly `num_steps` values descending from 1.0 toward a value
/// close to `1.0 - threshold_noise`. Lightricks's original fn appends a
/// trailing `1.0` then reverses via `1.0 - x` then drops the last element
/// (`sigma_schedule[:-1]`); this port does the same and ends up with the
/// same `num_steps`-length list. The caller is responsible for appending
/// a trailing 0.0 if the downstream Euler step expects a terminator.
pub fn linear_quadratic_schedule(num_steps: usize, threshold_noise: f32) -> Vec<f32> {
    if num_steps == 1 {
        return vec![1.0];
    }
    let linear_steps = num_steps / 2;
    let quadratic_steps = num_steps - linear_steps;

    let slope = threshold_noise / linear_steps as f32;
    let threshold_noise_step_diff = linear_steps as f32 - threshold_noise * num_steps as f32;
    let quadratic_coef =
        threshold_noise_step_diff / (linear_steps as f32 * (quadratic_steps as f32).powi(2));
    let linear_coef = threshold_noise / linear_steps as f32
        - 2.0 * threshold_noise_step_diff / (quadratic_steps as f32).powi(2);
    let const_coef = quadratic_coef * (linear_steps as f32).powi(2);

    // Build `linear + quadratic + [1.0]` then reverse via `1.0 - x` then
    // drop the last element — mirrors Lightricks's Python exactly.
    let mut ascending: Vec<f32> = Vec::with_capacity(num_steps + 1);
    for i in 0..linear_steps {
        ascending.push(slope * i as f32);
    }
    for i in linear_steps..num_steps {
        let fi = i as f32;
        ascending.push(quadratic_coef * fi * fi + linear_coef * fi + const_coef);
    }
    ascending.push(1.0);

    let mut descending: Vec<f32> = ascending.into_iter().map(|x| 1.0 - x).collect();
    descending.pop(); // drop the trailing 0 (matches `sigma_schedule[:-1]`)
    descending
}

/// LTX-2 canonical scheduler — direct port of `LTX2Scheduler.execute` from
/// `ltx_core/components/schedulers.py` (the LTX-2 22B 18.88-active-param
/// model's default scheduler). This is what `model_index.json` for the LTX-2
/// pipeline aliases as `FlowMatchEulerDiscreteScheduler` with
/// `use_dynamic_shifting=True, time_shift_type="exponential",
/// base_shift=0.95, max_shift=2.05, shift_terminal=0.1`.
///
/// Returns `num_steps + 1` sigmas in descending order from 1.0 to 0.0.
/// The trailing 0.0 acts as the Euler-step terminator, so callers should
/// NOT push another 0.
///
/// Defaults match upstream `LTX2Scheduler.execute`:
///   max_shift=2.05, base_shift=0.95, stretch=true, terminal=0.1
///
/// `num_tokens` is `prod(latent.shape[2:])` for video, i.e. F * H * W in
/// latent grid units.
///
/// TODO PARITY: not yet covered by `bin/ltx2_sigma_parity` — add a parity
/// test against Python's `LTX2Scheduler` once a Python reference is dumped.
pub fn ltx2_scheduler_sigmas(
    num_steps: usize,
    num_tokens: usize,
    max_shift: f32,
    base_shift: f32,
    stretch: bool,
    terminal: f32,
) -> Vec<f32> {
    // Token-dependent shift (linear in tokens between [BASE_SHIFT_ANCHOR,
    // MAX_SHIFT_ANCHOR] = [1024, 4096]).
    const BASE_ANCHOR: f32 = 1024.0;
    const MAX_ANCHOR: f32 = 4096.0;
    let mm = (max_shift - base_shift) / (MAX_ANCHOR - BASE_ANCHOR);
    let b = base_shift - mm * BASE_ANCHOR;
    let sigma_shift = (num_tokens as f32) * mm + b;

    // sigmas = linspace(1.0, 0.0, steps + 1)
    let n = num_steps + 1;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| 1.0 - (i as f32) / (num_steps as f32))
        .collect();

    // Apply exponential shift: where sigma != 0, sigma = exp(s)/(exp(s)+(1/sigma-1))
    // The Python uses `power=1` so the `(1/t - 1) ** power` is just `(1/t - 1)`.
    let es = sigma_shift.exp();
    for s in sigmas.iter_mut() {
        if *s != 0.0 {
            *s = es / (es + (1.0 / *s - 1.0));
        }
    }

    // Stretch so non-zero tail matches `terminal`.
    if stretch {
        // Find the last non-zero sigma to compute scale_factor.
        let last_nonzero_idx = sigmas
            .iter()
            .rposition(|&s| s != 0.0)
            .unwrap_or(sigmas.len() - 1);
        let last_one_minus_z = 1.0 - sigmas[last_nonzero_idx];
        let scale_factor = last_one_minus_z / (1.0 - terminal);
        if scale_factor != 0.0 {
            for s in sigmas.iter_mut() {
                if *s != 0.0 {
                    let one_minus_z = 1.0 - *s;
                    *s = 1.0 - (one_minus_z / scale_factor);
                }
            }
        }
    }

    sigmas
}

/// Convenience: LTX-2 canonical scheduler with the exact defaults from
/// `models--Lightricks--LTX-2/scheduler/scheduler_config.json`:
/// `base_shift=0.95, max_shift=2.05, stretch=true, terminal=0.1`.
pub fn ltx2_scheduler_default(num_steps: usize, num_tokens: usize) -> Vec<f32> {
    ltx2_scheduler_sigmas(num_steps, num_tokens, 2.05, 0.95, true, 0.1)
}

/// Build token-dependent sigma schedule — **Flux-style exponential shift**,
/// not what Lightricks uses. Kept for experiments and other models;
/// NOT on the LTX-2 dev reference path. Use `linear_quadratic_schedule`
/// for LTX-Video / 0.9.8-dev parity, or `ltx2_scheduler_default` for LTX-2.
pub fn build_dev_sigma_schedule(
    num_steps: usize,
    num_latent_tokens: usize,
    base_shift: f32,
    max_shift: f32,
    terminal: f32,
) -> Vec<f32> {
    // Linear interpolation of shift based on token count
    let m = (max_shift - base_shift) / (4096.0 - 1024.0);
    let b = base_shift - m * 1024.0;
    let shift = (num_latent_tokens as f32 * m + b).clamp(base_shift, max_shift);

    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - (i as f32 / num_steps as f32);
        // Exponential shift: exp(shift) / (exp(shift) + (1/t - 1))
        let shifted = if t <= 0.0 {
            0.0
        } else if t >= 1.0 {
            1.0
        } else {
            let es = shift.exp();
            es / (es + (1.0 / t - 1.0))
        };
        sigmas.push(shifted);
    }

    // Stretch to terminal value
    if terminal > 0.0 && terminal < 1.0 {
        for s in sigmas.iter_mut() {
            if *s > 0.0 {
                *s = *s * (1.0 - terminal) + terminal;
            }
        }
        // Last sigma stays 0
        if let Some(last) = sigmas.last_mut() {
            *last = 0.0;
        }
    }

    sigmas
}

/// Euler velocity sampler for LTX-2.
///
/// `model_fn(x, sigma) -> velocity` returns the model's velocity prediction.
/// Step: `denoised = x - velocity * sigma`, then Euler update.
pub fn euler_denoise_ltx2<F>(
    model_fn: F,
    noise: Tensor,
    sigmas: &[f32],
) -> Result<Tensor>
where
    F: Fn(&Tensor, f32) -> Result<Tensor>,
{
    let mut x = noise;
    let total = sigmas.len() - 1;

    for i in 0..total {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let velocity = model_fn(&x, sigma)?;

        if sigma_next == 0.0 {
            // Final step: return denoised directly
            let denoised = x.sub(&velocity.mul_scalar(sigma)?)?;
            x = denoised;
        } else {
            // Euler step: d = velocity, x += d * dt
            let dt = sigma_next - sigma;
            x = x.add(&velocity.mul_scalar(dt)?)?;
        }

        println!("  Step {}/{} sigma={:.4}", i + 1, total, sigma);
    }

    Ok(x)
}
