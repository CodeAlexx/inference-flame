//! Lens FlowMatch-Euler scheduler helpers.
//!
//! Pure-Rust port of the scheduler bits used by Microsoft Lens
//! (`lens/pipeline.py` lines 38-52 for `compute_empirical_mu`, lines 480-513
//! for the denoise loop). Lens uses
//! `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0,
//!  use_dynamic_shifting=True, time_shift_type="exponential")` driven by an
//! empirical `mu` derived from the image sequence length, with the sigma list
//! built as `linspace(1.0, 1.0/num_inference_steps, num_inference_steps)` —
//! note that Lens uses **N** sigmas (not the usual N+1 schedule-with-zero-tail
//! that Klein/Flux2 build with `get_schedule`). The Diffusers scheduler then
//! applies the `time_shift_type="exponential"` map.
//!
//! Stage A1 of the Lens port (`inference-flame/lens/BUILD_PLAN.md`).

use flame_core::{DType, Result, Tensor};

// ---------------------------------------------------------------------------
// Empirical mu — piecewise linear in (image_seq_len, num_steps)
// ---------------------------------------------------------------------------

/// Compute empirical `mu` from image sequence length and step count.
///
/// Byte-identical to `lens.pipeline.compute_empirical_mu` (lens/pipeline.py:38-52)
/// and to [`crate::sampling::klein_sampling::compute_empirical_mu`] — both
/// piecewise-linear regions and constants are the same calibration; the only
/// reason for this duplicate definition is to keep Lens's scheduler self-
/// contained and let us delete the Klein dependency later if the math drifts.
///
/// All arithmetic is f64 to mirror the Python reference exactly.
pub fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    // Same as klein_sampling::compute_empirical_mu — kept as a copy (not a
    // re-export) so this module stands alone if Lens's calibration diverges.
    let a1: f64 = 8.73809524e-05;
    let b1: f64 = 1.89833333;
    let a2: f64 = 0.00016927;
    let b2: f64 = 0.45666666;

    let seq = image_seq_len as f64;

    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }

    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;

    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

// ---------------------------------------------------------------------------
// Sigma list — Lens-specific (N values, not N+1)
// ---------------------------------------------------------------------------

/// Build Lens's raw sigma list before exponential shift.
///
/// Mirrors `np.linspace(1.0, 1.0/num_inference_steps, num_inference_steps)` in
/// `lens/pipeline.py:484`. Returns **N** values (NOT N+1): starts at 1.0, ends
/// at `1.0 / N`. The trailing zero is added by the scheduler internally; the
/// public Python contract passes exactly these N samples to `set_timesteps`.
///
/// Host-side f32 math is sufficient — `np.linspace` produces f64 then the
/// scheduler casts to f32 anyway; the endpoints `1.0` and `1.0/N` are exact in
/// both. Internal interpolation uses f64 to keep the intermediate values from
/// drifting.
pub fn build_sigmas(num_steps: usize) -> Vec<f32> {
    assert!(num_steps >= 1, "num_steps must be >= 1");
    if num_steps == 1 {
        // linspace(1.0, 1.0, 1) → [1.0]
        return vec![1.0];
    }
    let n = num_steps as f64;
    let start = 1.0_f64;
    let end = 1.0_f64 / n;
    let step = (end - start) / ((n - 1.0) as f64); // negative
    let mut out = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        let v = start + step * (i as f64);
        out.push(v as f32);
    }
    // Force exact endpoints (defensive vs accumulated round-off).
    out[0] = start as f32;
    out[num_steps - 1] = end as f32;
    out
}

// ---------------------------------------------------------------------------
// Exponential time shift — Diffusers FlowMatchEuler `time_shift_type="exponential"`
// ---------------------------------------------------------------------------

/// Apply Diffusers' exponential time shift to each sigma in `sigmas`.
///
/// Formula (per `time_shift_type="exponential"` in
/// `diffusers.schedulers.scheduling_flow_match_euler_discrete`):
///
/// ```text
/// shifted = exp(mu) * s / (exp(mu) * s + (1 - s))
/// ```
///
/// f32 in / f32 out, but the intermediate `(exp(mu)*s) / (exp(mu)*s + 1 - s)`
/// is computed in f64 to match Diffusers' Python implementation (which runs
/// against a torch f64 tensor for dynamic shifting).
///
/// Note: `mu = 0` is the identity (`exp(0) = 1` → `s / (s + 1 - s) = s`).
pub fn apply_exponential_shift(sigmas: &[f32], mu: f64) -> Vec<f32> {
    let exp_mu = mu.exp();
    sigmas
        .iter()
        .map(|&s| {
            let s64 = s as f64;
            let num = exp_mu * s64;
            let den = exp_mu * s64 + (1.0 - s64);
            (num / den) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Euler step — single FlowMatchEuler integration step
// ---------------------------------------------------------------------------

/// One `FlowMatchEulerDiscreteScheduler.step` integration step.
///
/// Computes `x_next = x_curr + (sigma_next - sigma_curr) * noise_pred`,
/// matching the Diffusers FlowMatchEuler implementation used by Lens (which
/// passes `noise_pred` and `sigma_curr=t` and lets the scheduler look up
/// `sigma_next` from its internal sigma list).
///
/// **Dtype contract — diffusers-exact, NOT a clean F32 path.**
///
/// `latents` and `noise_pred` are typically BF16 (the Lens DiT runs BF16
/// end-to-end). Diffusers `FlowMatchEulerDiscreteScheduler.step` does:
///
/// ```python
/// sample = sample.to(torch.float32)            # F32
/// # dt is F32 scalar; model_output is BF16. PyTorch coerces scalar*tensor
/// # to the TENSOR dtype, so dt * model_output stays BF16:
/// prev_sample = sample + dt * model_output     # F32 + BF16 -> F32
/// prev_sample = prev_sample.to(model_output.dtype)  # F32 -> BF16
/// ```
///
/// The subtle bit (and the source of the 2026-05-22 round-1 wrong fix): the
/// `dt * model_output` product is computed in **BF16**, not F32, because
/// PyTorch's `scalar * tensor` returns the tensor's dtype. Only the final
/// add `sample(F32) + delta(BF16)` upcasts. Doing the whole step in F32
/// (the round-1 Fix 2 path) produces a slightly different result because
/// `dt * noise` rounds differently in F32 vs BF16.
///
/// Empirically verified bit-exact (max_abs = 0.0 at every one of 20 steps)
/// against `lens/parity/captures_512/latents_post_step_NN.safetensors`
/// using this exact sequence — see `BUGFIX_TRIAGE_2026-05-22.md` Round 2.
///
/// **Contiguity:** does not call `.contiguous()`. If a future caller hands in
/// a non-contiguous view that breaks the underlying ops, surface that to the
/// model code rather than silently fixing it here (per CONTEXT trap about cat
/// non-contig output — `.contiguous()` sprinkles belong with the producer).
///
/// Math (diffusers FlowMatchEulerDiscreteScheduler.step, bit-exact replication):
/// ```text
/// dt = sigma_next - sigma_curr                     // f32 host
/// delta_bf16 = noise_pred * dt                     // BF16 tensor (dt is host scalar)
/// x_next_f32 = x_curr.to(F32) + delta_bf16         // F32 + BF16 -> F32
/// x_next     = x_next_f32.to(input_dtype)          // back to BF16
/// ```
pub fn euler_step(
    latents: &Tensor,
    noise_pred: &Tensor,
    sigma_curr: f32,
    sigma_next: f32,
) -> Result<Tensor> {
    let dt = sigma_next - sigma_curr;
    let target_dtype = latents.dtype();
    // Step 1: dt * noise in input dtype (BF16). PyTorch's scalar*tensor
    // returns the tensor dtype, so this MUST stay BF16 to match diffusers.
    let delta = noise_pred.mul_scalar(dt)?;
    // Step 2: upcast latents to F32 and add the (still-BF16) delta. The add
    // broadcasts to F32. flame-core's `add` should handle mixed dtype the
    // same way PyTorch does; if it requires same-dtype, this still ends up
    // matching because we cast back to BF16 next anyway.
    let latents_f32 = latents.to_dtype(DType::F32)?;
    // Cast delta to F32 explicitly so flame-core's `add` doesn't need to
    // handle mixed dtype. The values are identical (BF16->F32 is exact).
    let delta_f32 = delta.to_dtype(DType::F32)?;
    let result_f32 = latents_f32.add(&delta_f32)?;
    result_f32.to_dtype(target_dtype)
}

// ---------------------------------------------------------------------------
// Tests (host-only — no CUDA required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mu_small_seq() {
        // image_seq_len = 4096 (1024² / 16 = 64×64 tokens), num_steps = 20 —
        // falls into the linear-in-num_steps branch (seq_len ≤ 4300).
        // Reference value computed from the pipeline.py formula:
        //   a1, b1 = 8.73809524e-05, 1.89833333
        //   a2, b2 = 0.00016927,     0.45666666
        //   m_200 = a2 * 4096 + b2
        //   m_10  = a1 * 4096 + b1
        //   a     = (m_200 - m_10) / 190
        //   b     = m_200 - 200 * a
        //   mu    = a * 20 + b
        //
        // Computed in f64 here (same dtype as the Python ref) to give us a
        // bit-exact gold value rather than a hand-rounded approximation.
        let a1: f64 = 8.73809524e-05;
        let b1: f64 = 1.89833333;
        let a2: f64 = 0.00016927;
        let b2: f64 = 0.45666666;
        let seq: f64 = 4096.0;
        let m_200 = a2 * seq + b2;
        let m_10 = a1 * seq + b1;
        let a = (m_200 - m_10) / 190.0;
        let b = m_200 - 200.0 * a;
        let expected = a * 20.0 + b;

        let mu = compute_empirical_mu(4096, 20);
        assert!(
            (mu - expected).abs() < 1e-12,
            "mu(4096, 20) = {}, expected = {}",
            mu,
            expected
        );
        // Also sanity-check the rough magnitude (≈ 2.198) within 1e-3 as the
        // build-plan asked.
        assert!(
            (mu - 2.198_f64).abs() < 1e-3,
            "mu(4096, 20) = {} not within 1e-3 of ≈ 2.198",
            mu
        );
    }

    #[test]
    fn mu_large_seq() {
        // image_seq_len = 4500 > 4300 → mu = a2 * seq + b2
        // = 0.00016927 * 4500 + 0.45666666
        // = 0.76171500 + 0.45666666
        // = 1.21838166
        let mu = compute_empirical_mu(4500, 20);
        let expected = 0.00016927_f64 * 4500.0 + 0.45666666_f64;
        assert!(
            (mu - expected).abs() < 1e-6,
            "mu(4500, 20) = {}, expected = {}",
            mu,
            expected
        );
    }

    #[test]
    fn sigmas_endpoints() {
        let sigmas = build_sigmas(20);
        assert_eq!(sigmas.len(), 20, "Lens uses N sigmas, not N+1");
        assert!(
            (sigmas[0] - 1.0_f32).abs() < 1e-7,
            "sigmas[0] = {} should be exactly 1.0",
            sigmas[0]
        );
        let expected_last = 1.0_f32 / 20.0_f32; // 0.05
        assert!(
            (sigmas[19] - expected_last).abs() < 1e-7,
            "sigmas[19] = {} should be ≈ {}",
            sigmas[19],
            expected_last
        );
        // Strictly decreasing.
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] > sigmas[i + 1],
                "sigmas should be strictly decreasing: [{}]={} not > [{}]={}",
                i,
                sigmas[i],
                i + 1,
                sigmas[i + 1]
            );
        }
    }

    #[test]
    fn shift_identity_at_mu0() {
        // exp(0) = 1 → shifted = s / (s + (1 - s)) = s / 1 = s
        let out = apply_exponential_shift(&[0.5_f32], 0.0);
        assert_eq!(out.len(), 1);
        assert!(
            (out[0] - 0.5_f32).abs() < 1e-7,
            "mu=0 should be identity, got {}",
            out[0]
        );
        // Spot-check more values are also identity at mu=0.
        let xs = [0.05_f32, 0.25, 0.75, 0.95];
        let ys = apply_exponential_shift(&xs, 0.0);
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert!(
                (x - y).abs() < 1e-7,
                "mu=0 identity broken: in={}, out={}",
                x,
                y
            );
        }
    }

    #[test]
    fn shift_monotonic_positive_mu() {
        // Sanity: exponential shift with mu > 0 must remain strictly monotonic
        // in s (the schedule's ordering is essential).
        let sigmas = build_sigmas(20);
        let shifted = apply_exponential_shift(&sigmas, 2.197);
        assert_eq!(shifted.len(), sigmas.len());
        for i in 0..shifted.len() - 1 {
            assert!(
                shifted[i] > shifted[i + 1],
                "shifted schedule must remain decreasing: [{}]={} not > [{}]={}",
                i,
                shifted[i],
                i + 1,
                shifted[i + 1]
            );
        }
        // Endpoints: shift maps [1.0, 1/N] into (0, 1] and the s=1.0 endpoint
        // is a fixed point (exp_mu*1 / (exp_mu*1 + 0) = 1).
        assert!(
            (shifted[0] - 1.0_f32).abs() < 1e-6,
            "s=1.0 should be a fixed point of the exponential shift, got {}",
            shifted[0]
        );
    }
}
