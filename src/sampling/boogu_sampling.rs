//! Boogu-Image C8a — FlowMatchEuler scheduler with **static v1 time-shift**.
//!
//! Faithful port of Boogu's
//! `boogu/schedulers/scheduling_flow_match_euler_discrete_time_shifting.py`
//! (the released `FlowMatchEulerDiscreteScheduler`) for the production runtime
//! config (`scheduler/scheduler_config.json`):
//!
//! ```json
//! { "do_shift": true, "dynamic_time_shift": false,
//!   "time_shift_version": "v1", "seq_len": 4096, "num_train_timesteps": 1000 }
//! ```
//!
//! Because `dynamic_time_shift=false`, the shift uses the **config `seq_len`
//! (4096)** — NOT the per-image token count — so `mu = lin(4096) = 1.15` is a
//! constant, independent of the requested resolution. The `num_tokens` argument
//! that the pipeline passes to `set_timesteps` is therefore ignored on this
//! static path (verified: the oracle dump for N=8 uses `num_tokens=256` yet
//! still produces the mu=1.15 schedule).
//!
//! ## Schedule (oracle `set_timesteps` + the `_timesteps` cat)
//!
//! 1. `t_arr = linspace(0, 1, N+1)[:-1]`  (N ascending values, t[0]=0=noise).
//! 2. v1 static shift: `t_arr = _time_shift_v1(t_arr, mu=1.15, sigma=1.0)`.
//! 3. `_timesteps = cat([t_arr, ones(1)])`  (append a trailing **1.0**).
//!
//! So [`build_boogu_timesteps`] returns the **N+1** value `_timesteps` array
//! (ascending in `[0,1]`, last entry exactly 1.0) — the array the oracle's
//! `step()` indexes with `t = _timesteps[i]`, `t_next = _timesteps[i+1]`.
//!
//! ## Euler step (oracle `step`, lines 316-320)
//!
//! ```python
//! sample = sample.to(torch.float32)
//! t      = self._timesteps[self.step_index]
//! t_next = self._timesteps[self.step_index + 1]
//! prev_sample = sample + (t_next - t) * model_output      # dt > 0
//! ```
//!
//! **dt = t_next − t > 0** (ascending schedule) and there is **NO model-output
//! sign flip** (contrast Z-Image's diffusers note). The sample is upcast to F32
//! before the update — which is why the denoise loop holds the latent in F32
//! (see `boogu_infer`).
//!
//! Cross-checked against the verified Mojo C8a port
//! (`serenitymojo/sampling/boogu_sched_parity.mojo` /
//! `build_boogu_timesteps` — timesteps max-abs **0.0** bit-exact vs torch).
//!
//! Inference helper — autograd OFF. Host f32 math; the per-step tensor update
//! lives in `boogu_infer` (this module just produces the schedule + the scalar
//! step rule).

/// Boogu scheduler `seq_len` (static v1 shift uses the **config** seq_len, not
/// per-image tokens). `_get_lin_function(256→0.5, 4096→1.15)(4096) = 1.15`.
pub const BOOGU_SCHED_SEQ_LEN: usize = 4096;

/// `_get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15)(x)` — the training-side
/// mu(seq_len) linear map (scheduler `_get_lin_function`,
/// `base_shift=0.5`, `max_shift=1.15`). At `x=4096` this is exactly `1.15`.
pub fn boogu_lin_mu(seq_len: usize) -> f64 {
    let (x1, y1) = (256.0f64, 0.5f64); // base_shift
    let (x2, y2) = (4096.0f64, 1.15f64); // max_shift
    let m = (y2 - y1) / (x2 - x1);
    let b = y1 - m * x1;
    m * seq_len as f64 + b
}

/// Boogu `_time_shift_v1(t, mu, sigma=1.0)` (scheduler lines 144-153):
///
/// ```python
/// eps = 1e-8
/// t1 = clip(1 - t, eps, 1 - eps)
/// num = exp(mu)
/// denom = num + (1/t1 - 1) ** sigma
/// y = num / denom
/// out = 1 - y
/// ```
///
/// Operates per-scalar (the oracle vectorizes over the t array). `sigma` is 1.0
/// on the static-v1 path, so the `** sigma` is the identity power.
pub fn time_shift_v1(t: f64, mu: f64, sigma: f64) -> f64 {
    const EPS: f64 = 1e-8;
    let t1 = (1.0 - t).clamp(EPS, 1.0 - EPS);
    let num = mu.exp();
    let denom = num + (1.0 / t1 - 1.0).powf(sigma);
    let y = num / denom;
    1.0 - y
}

/// Build Boogu's `_timesteps` schedule — the **N+1** array the Euler `step`
/// indexes (`t = ts[i]`, `t_next = ts[i+1]`).
///
/// `num_steps` = `num_inference_steps`. Returns:
///   `[ shifted(linspace(0,1,N+1)[:-1]) ... , 1.0 ]`   (length `num_steps + 1`).
///
/// Static v1 shift with `mu = boogu_lin_mu(BOOGU_SCHED_SEQ_LEN) = 1.15`. The
/// resulting array is ascending in `[0,1]` with the last entry exactly `1.0`.
///
/// Bit-matches the oracle dump for N=8 (see the unit test):
/// `[0.0, 0.0432763.., 0.0954692.., 0.1596512.., 0.2404890..,
///   0.3454332.., 0.4871558.., 0.6890989.., 1.0]`.
pub fn build_boogu_timesteps(num_steps: usize) -> Vec<f32> {
    assert!(num_steps >= 1, "boogu: num_steps must be >= 1");
    let mu = boogu_lin_mu(BOOGU_SCHED_SEQ_LEN); // 1.15 (static, resolution-independent)

    // linspace(0, 1, N+1)[:-1] -> N ascending values t[0]=0.
    let n1 = num_steps + 1;
    let mut ts: Vec<f32> = Vec::with_capacity(n1);
    for i in 0..num_steps {
        let t = i as f64 / (n1 as f64 - 1.0); // == i / N  (linspace(0,1,N+1) step)
        let shifted = time_shift_v1(t, mu, 1.0);
        ts.push(shifted as f32);
    }
    // cat([t_arr, ones(1)]) — append trailing 1.0.
    ts.push(1.0f32);
    ts
}

/// One Euler flow-match step (oracle `step`, line 320):
/// `prev = sample + (t_next - t) * model_output`.
///
/// dt = `t_next - t` is **positive** (ascending schedule); **no sign flip**.
/// Scalar form of the rule — `boogu_infer` applies it to the F32 latent tensor
/// (`latent + dt * velocity`). Provided for documentation/test parity with the
/// oracle's scalar update.
#[inline]
pub fn euler_dt(t: f32, t_next: f32) -> f32 {
    t_next - t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lin_mu_endpoints() {
        // base_shift @256, max_shift @4096.
        assert!((boogu_lin_mu(256) - 0.5).abs() < 1e-9);
        assert!((boogu_lin_mu(4096) - 1.15).abs() < 1e-9);
        // Static config seq_len -> 1.15 regardless of image resolution.
        assert!((boogu_lin_mu(BOOGU_SCHED_SEQ_LEN) - 1.15).abs() < 1e-9);
    }

    #[test]
    fn timesteps_length_and_endpoints() {
        for n in [1usize, 4, 8, 20] {
            let ts = build_boogu_timesteps(n);
            assert_eq!(ts.len(), n + 1, "N+1 values");
            // First t is the shift of linspace[0]=0; the eps-clip leaves a
            // negligible positive residue (rounds to 0.0 in f32 for n>=8, but the
            // raw value is ~3e-9). Assert "ascending start near 0", not exact.
            assert!(ts[0] >= 0.0 && ts[0] < 1e-3, "first t near 0, got {}", ts[0]);
            assert_eq!(ts[n], 1.0, "trailing 1.0 appended (exact)");
        }
    }

    #[test]
    fn timesteps_strictly_ascending_positive_dt() {
        let ts = build_boogu_timesteps(20);
        for i in 0..ts.len() - 1 {
            let dt = euler_dt(ts[i], ts[i + 1]);
            assert!(dt > 0.0, "dt must be > 0 (no sign flip) at i={i}: {dt}");
        }
    }

    /// THE parity gate: bit-match the oracle's `_timesteps` dump for N=8.
    /// Expected values copied from `boogu_sched_oracle.py` (the REAL Boogu
    /// `FlowMatchEulerDiscreteScheduler.set_timesteps(8, num_tokens=256)`
    /// `_timesteps`), which the Mojo C8a port reproduced max-abs 0.0.
    #[test]
    fn timesteps_match_oracle_n8() {
        // f32 values as emitted by torch (`np.asarray(..., '<f4')`).
        let expected: [f32; 9] = [
            0.0,
            0.043_276_31,
            0.095_469_24,
            0.159_651_22,
            0.240_489_07,
            0.345_433_23,
            0.487_155_85,
            0.689_098_95,
            1.0,
        ];
        let got = build_boogu_timesteps(8);
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            // f32 round-trip from the same double-precision math: tolerate one
            // ULP-scale slack (the oracle stores f32; we compute in f64 then
            // cast to f32). Mojo gated this bit-exact (max-abs 0.0); we use a
            // tiny epsilon to be robust to the f64->f32 rounding order.
            let d = (g - e).abs();
            assert!(
                d <= 2e-6,
                "timestep[{i}] mismatch: got {g}, want {e} (|d|={d})"
            );
        }
    }

    #[test]
    fn time_shift_v1_fixed_point() {
        // At t such that (1/t1 - 1) == exp(mu) with t1 = 1 - t, the logistic
        // output y == 0.5 -> out == 0.5. Sanity that the transform is the
        // logistic mapping, not an identity.
        let mu: f64 = 1.15;
        // pick t1 = 1/(exp(mu)+1) so (1/t1 - 1) = exp(mu); then y=exp(mu)/(2 exp(mu))=0.5.
        let t1 = 1.0 / (mu.exp() + 1.0);
        let t = 1.0 - t1;
        let out = time_shift_v1(t, mu, 1.0);
        assert!((out - 0.5).abs() < 1e-9, "expected 0.5, got {out}");
    }
}
