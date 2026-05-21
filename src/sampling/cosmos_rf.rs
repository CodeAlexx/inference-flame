//! Cosmos-Predict2.5 rectified-flow (flow-matching) sampler primitives.
//!
//! Inference-only port of the sigma schedule + Euler step + CFG combine used
//! by Cosmos's `FlowUniPCMultistepScheduler` configured in flow-matching mode
//! (see `cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py:72-122` for
//! the constructor and `:150-219` for `set_timesteps`).
//!
//! For this port we only need the FlowMatch portions (Euler step + schedule)
//! — the UniPC multistep corrector is not implemented; the Python default is
//! Euler/UniPC, and the default `num_steps` for the 2B inference path is 35
//! (`video2world.py:487`). Default `shift` for V2_2B at inference is 5.0
//! (`text2world_model_rectified_flow.py:502`).
//!
//! This file provides three primitives — no full denoise loop. The binary
//! (chunks 11a/b/c, out of scope here) drives the loop.
//!
//! ## Schedule (`set_timesteps` math, FlowMatch path)
//!
//! Python `FlowUniPCMultistepScheduler.__init__` first builds a full 1000-entry
//! table and stores `self.sigma_max = sigmas[0]`, `self.sigma_min = sigmas[-1]`
//! (`fm_solvers_unipc.py:100-122`). The `__init__`'s default `shift=1.0` is the
//! identity, so the stored `sigma_max = (N-1)/N = 0.999` and `sigma_min = 0.0`.
//! Then `set_timesteps` does:
//!
//! ```text
//!   sigmas_raw = linspace(sigma_max=0.999, sigma_min=0.0, num_steps+1)[:-1]
//!   sigmas = shift * sigmas_raw / (1 + (shift - 1) * sigmas_raw)
//!   sigmas = concat([sigmas, [0.0]])    // final_sigmas_type="zero"
//! ```
//!
//! The endpoint correction matters: at shift=5, n=35, the first Python sigma is
//! `5*0.999 / (1 + 4*0.999) = 0.99980`, NOT 1.0. Mirrors `magihuman_unipc.rs`
//! which gets the same scheduler family right.
//!
//! Notation: `sigmas[0]` is the noisiest (closest to 1.0), `sigmas[-1] = 0.0`
//! is clean. Length = `num_steps + 1`.
//!
//! ## Euler step (FlowMatch convention)
//!
//! Diffusers `FlowMatchEulerDiscrete.step`:
//!
//! ```text
//!   sigma_curr = sigmas[step_idx]
//!   sigma_next = sigmas[step_idx + 1]
//!   x_next     = x_curr + (sigma_next - sigma_curr) * v_pred
//! ```
//!
//! where `v_pred` is the model's velocity output. Sign: `sigma_next < sigma_curr`
//! → dt is negative → we step backwards in noise level. The Cosmos
//! `velocity_field` follows the same convention as diffusers FlowMatch.
//!
//! ## CFG combine
//!
//! Classifier-free guidance: `out = uncond + cfg_scale * (cond - uncond)`.
//! At `cfg_scale=1.0` this collapses to `cond`. At `cfg_scale=0.0` it
//! collapses to `uncond`. Cosmos's `text2world.py:622` default is
//! `guidance=7.0`; the upstream binary forwards this scalar verbatim.

use flame_core::{DType, Error, Result, Tensor};

/// Rectified-flow / FlowMatch sampler primitives — sigma schedule, Euler
/// step, CFG combine. Does NOT implement the full denoise loop (that lives
/// in the inference binary). Mirrors the FlowMatch portions of
/// `FlowUniPCMultistepScheduler`; UniPC corrector is out of scope.
#[derive(Debug, Clone, Copy)]
pub struct RectifiedFlowSampler {
    /// Number of denoising steps (Python `num_inference_steps`).
    pub num_steps: usize,
    /// Classifier-free guidance scale. `1.0` = no guidance, larger = stronger.
    pub cfg_scale: f32,
    /// Time-SNR shift parameter (Python `shift`). Default for V2_2B inference
    /// is 5.0 (`text2world_model_rectified_flow.py:502`). The schedule
    /// transformation is `sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)`.
    pub shift: f32,
    /// Training noise-step count, used to compute `sigma_min = 1 / num_train_timesteps`.
    /// Cosmos default is 1000 (`fm_solvers_unipc.py:74`).
    pub num_train_timesteps: usize,
}

impl RectifiedFlowSampler {
    /// Convenience constructor matching upstream defaults (`num_train_timesteps=1000`).
    pub fn new(num_steps: usize, cfg_scale: f32, shift: f32) -> Self {
        Self {
            num_steps,
            cfg_scale,
            shift,
            num_train_timesteps: 1000,
        }
    }

    /// Pre-compute the sigma schedule. Length = `num_steps + 1`.
    ///
    /// `sigmas[0]` ≈ 1.0 (most noise), `sigmas[num_steps] = 0.0` (clean).
    ///
    /// Math is in f64 for numerical safety, then cast to f32. Python uses
    /// numpy f32 throughout; the small extra precision here is harmless for
    /// the final f32 output and avoids ULP-noise differences from the
    /// linspace endpoint.
    pub fn sigmas(&self) -> Vec<f32> {
        let n = self.num_steps;
        if n == 0 {
            return vec![0.0_f32];
        }
        // Python `__init__` builds the full N-entry table with `__init__`'s
        // default `shift=1.0` (identity), so `self.sigma_max = (N-1)/N` and
        // `self.sigma_min = 0.0`. `set_timesteps` then re-applies the
        // user-supplied `shift` to a fresh linspace over those endpoints.
        // See `magihuman_unipc.rs:55-57` for the matching precedent.
        let n_train = self.num_train_timesteps as f64;
        let sigma_max = (n_train - 1.0) / n_train;
        let sigma_min = 0.0_f64;
        let shift = self.shift as f64;

        // np.linspace(sigma_max, sigma_min, n+1)[:-1] — exactly n entries.
        // linspace endpoints inclusive; we drop the last (== sigma_min).
        let mut sigmas = Vec::<f32>::with_capacity(n + 1);
        let denom = n as f64; // n+1 points → n intervals
        for k in 0..n {
            let raw = sigma_max + (sigma_min - sigma_max) * (k as f64) / denom;
            // Apply time-SNR shift.
            let shifted = shift * raw / (1.0 + (shift - 1.0) * raw);
            sigmas.push(shifted as f32);
        }
        // Append final 0.0 (final_sigmas_type="zero", Python `:194`).
        sigmas.push(0.0_f32);
        sigmas
    }

    /// Single Euler step. Given current `x` at `sigmas[step_idx]` and
    /// `model_output` (velocity prediction), produce x at `sigmas[step_idx + 1]`.
    ///
    /// `x_next = x_curr + (sigma_next - sigma_curr) * model_output`
    ///
    /// `step_idx` must be in `0..num_steps` (the final entry `sigmas[num_steps]
    /// = 0.0` is the destination of the last step, never a step source).
    ///
    /// `x` and `model_output` are typically BF16; the scaled-add is computed
    /// in their native dtype.
    pub fn step(&self, x: &Tensor, model_output: &Tensor, step_idx: usize) -> Result<Tensor> {
        if step_idx >= self.num_steps {
            return Err(Error::InvalidInput(format!(
                "RectifiedFlowSampler::step: step_idx={step_idx} out of range \
                 (num_steps={})", self.num_steps
            )));
        }
        let sigmas = self.sigmas();
        let sigma_curr = sigmas[step_idx];
        let sigma_next = sigmas[step_idx + 1];
        let dt: f32 = sigma_next - sigma_curr;

        // Shape sanity: x and model_output must match.
        if x.shape().dims() != model_output.shape().dims() {
            return Err(Error::InvalidInput(format!(
                "RectifiedFlowSampler::step: x shape {:?} != model_output shape {:?}",
                x.shape().dims(), model_output.shape().dims()
            )));
        }

        let delta = model_output.mul_scalar(dt)?;
        x.add(&delta)
    }

    /// Classifier-free guidance combine:
    ///   `out = uncond + cfg_scale * (cond - uncond)`
    ///
    /// Equivalent to a linear interpolation: at `cfg=0` returns `uncond`, at
    /// `cfg=1` returns `cond`, at `cfg>1` extrapolates past `cond`.
    ///
    /// Both inputs must have identical shape and dtype.
    pub fn cfg_combine(&self, uncond: &Tensor, cond: &Tensor) -> Result<Tensor> {
        if uncond.shape().dims() != cond.shape().dims() {
            return Err(Error::InvalidInput(format!(
                "cfg_combine: uncond shape {:?} != cond shape {:?}",
                uncond.shape().dims(), cond.shape().dims()
            )));
        }
        if uncond.dtype() != cond.dtype() {
            return Err(Error::InvalidInput(format!(
                "cfg_combine: uncond dtype {:?} != cond dtype {:?}",
                uncond.dtype(), cond.dtype()
            )));
        }
        // out = uncond + cfg * (cond - uncond)
        let diff = cond.sub(uncond)?;
        let scaled = diff.mul_scalar(self.cfg_scale)?;
        uncond.add(&scaled)
    }
}

// Compile-time silence for the unused `DType` import when no test cfg sees it.
#[allow(dead_code)]
fn _unused_dtype_import_marker(_: DType) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::Shape;
    use std::sync::Arc;

    fn maybe_cuda_device() -> Option<Arc<cudarc::driver::CudaDevice>> {
        cudarc::driver::CudaDevice::new(0).ok()
    }

    #[test]
    fn sigma_schedule_length_is_num_steps_plus_one() {
        let s = RectifiedFlowSampler::new(35, 7.0, 5.0).sigmas();
        assert_eq!(s.len(), 36);
    }

    #[test]
    fn sigma_schedule_monotone_decreasing_and_ends_at_zero() {
        let s = RectifiedFlowSampler::new(35, 7.0, 5.0).sigmas();
        for i in 1..s.len() {
            assert!(
                s[i] < s[i - 1] || (s[i] == 0.0 && s[i - 1] > 0.0),
                "schedule not monotone-decreasing: s[{}]={} s[{}]={}",
                i - 1, s[i - 1], i, s[i]
            );
        }
        // First sigma is shifted from `(N-1)/N = 0.999` via `shift*x/(1+(shift-1)*x)`
        // with x=0.999, shift=5 → 5*0.999 / (1 + 4*0.999) = 4.995/4.996 ≈ 0.99980.
        let expected_first = 5.0_f64 * 0.999 / (1.0 + 4.0 * 0.999);
        assert!(
            (s[0] as f64 - expected_first).abs() < 1e-5,
            "sigmas[0] = {} != {} (expected from shift=5, raw=0.999)",
            s[0], expected_first
        );
        // Tightness check vs the OLD buggy schedule which had sigmas[0]=1.0.
        assert!(s[0] < 1.0, "sigmas[0] must be strictly less than 1.0, got {}", s[0]);
        assert_eq!(s[s.len() - 1], 0.0);
    }

    /// Independent oracle: construct sigmas using the same formula as
    /// `magihuman_unipc.rs:55-68` (the codebase's known-correct precedent for
    /// the same scheduler family) and assert bit-equality. This is the F1 fix
    /// validation.
    #[test]
    fn sigma_schedule_matches_magihuman_oracle_with_shift_5() {
        let n_inf = 35_usize;
        let n_train = 1000_usize;
        let shift = 5.0_f64;

        // Replicate magihuman_unipc.rs construction.
        let sigma_max = (n_train - 1) as f64 / n_train as f64; // 0.999
        let sigma_min = 0.0_f64;
        let mut sigmas_lin: Vec<f64> = Vec::with_capacity(n_inf);
        for i in 0..n_inf {
            let t = sigma_max + (sigma_min - sigma_max) * (i as f64) / (n_inf as f64);
            sigmas_lin.push(t);
        }
        let mut oracle: Vec<f32> = sigmas_lin.iter()
            .map(|&x| (shift * x / (1.0 + (shift - 1.0) * x)) as f32)
            .collect();
        oracle.push(0.0_f32);

        let rust = RectifiedFlowSampler::new(n_inf, 7.0, shift as f32)
            .sigmas();
        assert_eq!(rust.len(), oracle.len(), "length mismatch");
        for (i, (&r, &o)) in rust.iter().zip(oracle.iter()).enumerate() {
            assert!(
                (r - o).abs() < 1e-7,
                "sigma[{i}] differs: rust={} oracle={}", r, o
            );
        }
    }

    /// numpy.linspace(0.999, 0.0, n+1)[:-1] convention: explicit f64
    /// construction matching numpy step semantics.
    #[test]
    fn sigma_schedule_matches_numpy_linspace_endpoint_convention() {
        let n_inf = 10_usize;
        let n_train = 1000_usize;
        // The unshifted raw values numpy would produce:
        //   linspace(0.999, 0.0, 11)[:-1] = 0.999, 0.8991, 0.7992, ...
        // step = (0.999 - 0.0) / 10 = 0.0999
        let sigma_max = (n_train - 1) as f64 / n_train as f64;
        let mut expected_raw: Vec<f64> = Vec::new();
        for i in 0..n_inf {
            expected_raw.push(sigma_max + (0.0 - sigma_max) * (i as f64) / (n_inf as f64));
        }
        // First raw should equal sigma_max; last raw should equal sigma_max/n.
        assert!((expected_raw[0] - sigma_max).abs() < 1e-12);
        let last_raw_expected = sigma_max - sigma_max * ((n_inf - 1) as f64) / (n_inf as f64);
        assert!((expected_raw[n_inf - 1] - last_raw_expected).abs() < 1e-12);

        // Apply shift=1 (identity) — should yield the raw values back.
        let s_identity = RectifiedFlowSampler::new(n_inf, 7.0, 1.0).sigmas();
        for (i, &r) in expected_raw.iter().enumerate() {
            assert!(
                (s_identity[i] as f64 - r).abs() < 1e-6,
                "shift=1 sigma[{i}] {} != raw {}", s_identity[i], r
            );
        }
        // Final zero appended.
        assert_eq!(s_identity[n_inf], 0.0);
    }

    #[test]
    fn sigma_schedule_shift_above_one_pushes_mass_toward_one() {
        // Larger `shift` should make intermediate sigmas LARGER (more time
        // spent at high-noise levels). Check the median.
        let a = RectifiedFlowSampler::new(20, 7.0, 1.0).sigmas();
        let b = RectifiedFlowSampler::new(20, 7.0, 5.0).sigmas();
        let med = a.len() / 2;
        // shift=1 is identity (sigmas = raw linspace). shift>1 pushes the
        // sigma curve UP at intermediate points (higher noise lingers longer).
        assert!(
            b[med] > a[med],
            "shift>1 should increase median sigma: shift1={} shift5={}",
            a[med], b[med]
        );
    }

    #[test]
    fn cfg_combine_at_scale_one_returns_cond() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let s = RectifiedFlowSampler::new(35, 1.0, 5.0);

        let n: usize = 16;
        let uncond_data: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let cond_data: Vec<f32> = (0..n).map(|i| -0.3 + (i as f32) * 0.07).collect();
        let uncond = Tensor::from_vec(uncond_data, Shape::from_dims(&[1, n]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let cond = Tensor::from_vec(cond_data.clone(), Shape::from_dims(&[1, n]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let out = s.cfg_combine(&uncond, &cond).expect("cfg_combine");
        let out_f32 = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();

        // At cfg=1.0 → out = cond. Tolerate BF16 round-trip noise.
        let mut max_abs_err: f32 = 0.0;
        for (a, b) in out_f32.iter().zip(cond_data.iter()) {
            let e = (a - b).abs();
            if e > max_abs_err { max_abs_err = e; }
        }
        assert!(max_abs_err < 1e-2, "cfg_combine(cfg=1) should equal cond, max_abs_err={max_abs_err}");
    }

    #[test]
    fn cfg_combine_at_scale_zero_returns_uncond() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let s = RectifiedFlowSampler::new(35, 0.0, 5.0);

        let n: usize = 16;
        let uncond_data: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let cond_data: Vec<f32> = (0..n).map(|i| -0.3 + (i as f32) * 0.07).collect();
        let uncond = Tensor::from_vec(uncond_data.clone(), Shape::from_dims(&[1, n]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let cond = Tensor::from_vec(cond_data, Shape::from_dims(&[1, n]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let out = s.cfg_combine(&uncond, &cond).expect("cfg_combine");
        let out_f32 = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();

        let mut max_abs_err: f32 = 0.0;
        for (a, b) in out_f32.iter().zip(uncond_data.iter()) {
            let e = (a - b).abs();
            if e > max_abs_err { max_abs_err = e; }
        }
        assert!(max_abs_err < 1e-2, "cfg_combine(cfg=0) should equal uncond, max_abs_err={max_abs_err}");
    }

    #[test]
    fn euler_step_applies_correct_delta() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let s = RectifiedFlowSampler::new(5, 7.0, 5.0);
        let sigmas = s.sigmas();
        let step_idx = 2_usize;
        let sigma_curr = sigmas[step_idx];
        let sigma_next = sigmas[step_idx + 1];
        let dt = sigma_next - sigma_curr;

        let n: usize = 8;
        let x_data: Vec<f32> = vec![1.0_f32; n];
        let v_data: Vec<f32> = vec![0.5_f32; n];
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[1, n]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let v = Tensor::from_vec(v_data, Shape::from_dims(&[1, n]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let out = s.step(&x, &v, step_idx).expect("step");
        let out_f32 = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();

        // Expected: 1.0 + dt * 0.5
        let expected = 1.0 + dt * 0.5;
        for &v in &out_f32 {
            assert!(
                (v - expected).abs() < 1e-2,
                "euler_step out {} != expected {} (dt={})", v, expected, dt
            );
        }
    }

    #[test]
    fn step_idx_out_of_range_errors() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let s = RectifiedFlowSampler::new(5, 7.0, 5.0);
        let n: usize = 4;
        let x = Tensor::from_vec(
            vec![1.0_f32; n], Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let v = x.clone();
        // step_idx == num_steps is invalid (would read sigmas[num_steps+1]
        // which doesn't exist).
        assert!(s.step(&x, &v, 5).is_err());
        assert!(s.step(&x, &v, 6).is_err());
    }
}
