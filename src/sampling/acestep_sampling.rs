//! ACE-Step Euler ODE sampler for flow-matching music generation.
//!
//! Implements the denoising loop from modeling_acestep_v15_base.py (lines 1864-1979).
//! Supports both turbo (8 steps, no CFG) and base (30+ steps, CFG) modes.

use crate::models::acestep_dit::AceStepDiT;
use flame_core::{DType, Result, Shape, Tensor};

/// Build the flow-matching timestep schedule.
///
/// Returns `num_steps + 1` values from 1.0 down to 0.0.
/// If `shift != 1.0`, applies time-SNR shifting:
///   `t_shifted = shift * t / (1 + (shift - 1) * t)`
pub fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut timesteps = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - i as f32 / num_steps as f32;
        let t_shifted = if (shift - 1.0).abs() < 1e-6 {
            t
        } else if t <= 0.0 || t >= 1.0 {
            t
        } else {
            shift * t / (1.0 + (shift - 1.0) * t)
        };
        timesteps.push(t_shifted);
    }
    timesteps
}

/// Euler ODE sampler for ACE-Step flow matching.
///
/// # Arguments
/// * `dit` - The ACE-Step DiT model (mutable for KV cache).
/// * `noise` - Initial noise [B, T, 64].
/// * `encoder_hidden_states` - Condition embeddings [B, L, 2048].
/// * `context_latents` - Context latents [B, T, 128] (cat with hidden inside DiT).
/// * `num_steps` - Number of denoising steps (8 for turbo, 30+ for base).
/// * `cfg_scale` - Classifier-free guidance scale (1.0 = no CFG, 7.0 typical for base).
/// * `shift` - Time-SNR shift (1.0 = no shift).
/// * `null_condition_emb` - Null condition embedding [1, 1, 2048] for CFG. Required if cfg_scale > 1.0.
///
/// # Returns
/// Denoised latents [B, T, 64].
pub fn acestep_sample(
    dit: &mut AceStepDiT,
    noise: &Tensor,
    encoder_hidden_states: &Tensor,
    context_latents: &Tensor,
    num_steps: usize,
    cfg_scale: f32,
    shift: f32,
    null_condition_emb: Option<&Tensor>,
) -> Result<Tensor> {
    let device = noise.device().clone();
    let b = noise.shape().dims()[0];
    let use_cfg = cfg_scale > 1.0;

    // Build timestep schedule
    let timesteps = build_schedule(num_steps, shift);

    // Clear KV cache from any previous generation
    dit.clear_cache();

    let mut xt = noise.clone();

    for i in 0..num_steps {
        let t_curr = timesteps[i];
        let t_prev = timesteps[i + 1];

        // Build timestep tensors [B] filled with t_curr
        let t_tensor = Tensor::from_vec_dtype(
            vec![t_curr; if use_cfg { b * 2 } else { b }],
            Shape::from_dims(&[if use_cfg { b * 2 } else { b }]),
            device.clone(),
            DType::BF16,
        )?;

        let vt = if use_cfg {
            let null_emb = null_condition_emb.ok_or_else(|| {
                flame_core::Error::InvalidInput(
                    "null_condition_emb required when cfg_scale > 1.0".into(),
                )
            })?;

            // Double the batch: [cond, uncond]
            let xt_doubled = Tensor::cat(&[&xt, &xt], 0)?;
            let ctx_doubled = Tensor::cat(&[context_latents, context_latents], 0)?;

            // encoder_hidden_states for cond, null_condition_emb for uncond
            let null_expanded = null_emb.broadcast_to(encoder_hidden_states.shape())?;
            let enc_doubled = Tensor::cat(&[encoder_hidden_states, &null_expanded], 0)?;

            // Clear cache on first CFG step since batch size changed
            if i == 0 {
                dit.clear_cache();
            }

            let pred = dit.forward(&xt_doubled, &t_tensor, &t_tensor, &enc_doubled, &ctx_doubled)?;

            // Split: first half = cond, second half = uncond
            let pred_cond = pred.narrow(0, 0, b)?;
            let pred_uncond = pred.narrow(0, b, b)?;

            // CFG: v = uncond + scale * (cond - uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            let scaled = diff.mul_scalar(cfg_scale)?;
            pred_uncond.add(&scaled)?
        } else {
            dit.forward(&xt, &t_tensor, &t_tensor, encoder_hidden_states, context_latents)?
        };

        // Euler step: xt = xt - vt * dt  (where dt = t_curr - t_prev > 0)
        let dt = t_curr - t_prev;
        let step = vt.mul_scalar(dt)?;
        xt = xt.sub(&step)?;
    }

    Ok(xt)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_length() {
        let ts = build_schedule(8, 1.0);
        assert_eq!(ts.len(), 9);
    }

    #[test]
    fn test_schedule_endpoints() {
        let ts = build_schedule(30, 1.0);
        assert!((ts[0] - 1.0).abs() < 1e-6, "first should be 1.0, got {}", ts[0]);
        assert!(ts[30].abs() < 1e-6, "last should be 0.0, got {}", ts[30]);
    }

    #[test]
    fn test_schedule_monotonic() {
        let ts = build_schedule(30, 1.0);
        for i in 0..ts.len() - 1 {
            assert!(
                ts[i] >= ts[i + 1],
                "must decrease: ts[{}]={} < ts[{}]={}",
                i, ts[i], i + 1, ts[i + 1]
            );
        }
    }

    #[test]
    fn test_schedule_no_shift() {
        let ts = build_schedule(4, 1.0);
        // linspace(1.0, 0.0, 5) = [1.0, 0.75, 0.5, 0.25, 0.0]
        let expected = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (a, b) in ts.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "no-shift mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_schedule_with_shift() {
        let ts = build_schedule(4, 2.0);
        assert_eq!(ts.len(), 5);
        assert!((ts[0] - 1.0).abs() < 1e-6);
        assert!(ts[4].abs() < 1e-6);
        // With shift=2.0, interior points should be shifted upward
        // t=0.75 -> 2*0.75 / (1 + 1*0.75) = 1.5/1.75 ≈ 0.857
        let expected_1 = 2.0 * 0.75 / (1.0 + 1.0 * 0.75);
        assert!((ts[1] - expected_1).abs() < 1e-5, "shift check: {} vs {}", ts[1], expected_1);
    }

    #[test]
    fn test_euler_velocity_step() {
        // Direct velocity: x_new = x - v * dt where dt = t_curr - t_prev
        let x = 10.0f32;
        let v = 5.0f32;
        let t_curr = 0.8f32;
        let t_prev = 0.6f32;
        let dt = t_curr - t_prev; // 0.2
        let x_new = x - v * dt;   // 10 - 5*0.2 = 9.0
        assert!((x_new - 9.0).abs() < 1e-6);
    }
}
