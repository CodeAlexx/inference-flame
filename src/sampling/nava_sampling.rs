//! NAVA scheduler + cross-modal CFG combine — pure-Rust port.
//!
//! Ports the sampling half of NAVA (Base T2AV, no timbre) from:
//!   * `ports/nava/nava_src/models/nava/utils/fm_solvers_unipc.py`
//!     (`FlowUniPCMultistepScheduler`) — the UniPC bh2 multistep flow-match
//!     scheduler.
//!   * `ports/nava/nava_src/pipeline_nava.py:457-552` — the two-instance
//!     (video + audio) lockstep denoise loop + the cross-modal CFG combine.
//!
//! Canonical config (`ports/nava/configs/nava.yaml`):
//!   `scheduler_personalized: true`, `scheduler_unipc: true` →
//!   `FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=5.0,
//!   use_dynamic_shifting=False)` for BOTH video and audio (`shift`,
//!   `shift_audio` both 5).
//!   `video_guidance_scale=3.0`, `audio_guidance_scale=2.0`,
//!   `video_align_guidance_scale=3.0`, `audio_align_guidance_scale=2.0`,
//!   `align_3d_cfg=true`. Timbre (`timbre_cfg=true` in yaml) is DEFERRED to v2.
//!
//! ## Why this is NOT a thin wrapper over `cosmos_unipc.rs`
//!
//! `CosmosUniPcMultistepScheduler` ports the SAME Python class and its
//! predictor/corrector math is identical to NAVA's. BUT its `new()` builds the
//! sigma schedule with a SINGLE flow-shift application (it assumes Python
//! `__init__` got `shift=1.0`, which is true for Cosmos's call site).
//!
//! NAVA constructs the scheduler with `shift=5.0` passed to `__init__`
//! (`pipeline_nava.py:85-92`) AND `use_dynamic_shifting=False`, so the shift is
//! applied TWICE:
//!   1. In `__init__` (`fm_solvers_unipc.py:112-118`) the shift is applied to
//!      the full 1000-step sigma array, giving `sigma_max = shift*0.999/(1+
//!      (shift-1)*0.999)` and `sigma_min = 0`.
//!   2. In `set_timesteps` (`fm_solvers_unipc.py:182-193`) a fresh
//!      `linspace(sigma_max, sigma_min, n+1)[:-1]` is built and the shift is
//!      applied AGAIN.
//!
//! Worked example (shift=5, n=2 steps):
//!   * cosmos single-shift: sigmas = [0.999800, 0.833055, 0]
//!   * NAVA double-shift:   sigmas = [0.999960, 0.833276, 0]
//! They diverge at ~1e-3, enough to fail a per-step parity gate. We therefore
//! reproduce NAVA's exact double-shift schedule here and mirror cosmos's
//! predictor/corrector (same Python source). FOLLOW-UP: the right long-term fix
//! is to give `CosmosUniPcMultistepScheduler` a schedule constructor that takes
//! an explicit `sigma_max` (or an "apply shift in init too" flag) so the
//! multistep math is shared. Left as a gap because this builder owns only
//! `nava_sampling.rs` and may not edit `cosmos_unipc.rs`.

use flame_core::{Error, Result, Tensor};

/// `FlowUniPCMultistepScheduler` configured for NAVA inference.
///
/// bh2 multistep, `solver_order=2`, `predict_x0=true`,
/// `final_sigmas_type="zero"`, `lower_order_final=true`,
/// `disable_corrector=[]`. Stateful: call `step` once per inference step in
/// order; `step_index` advances internally. One instance per modality (video +
/// audio), both with `shift=5.0`.
#[derive(Debug)]
pub struct NavaUniPCScheduler {
    /// Training noise-step count (`num_train_timesteps`, Python `:79`). 1000.
    pub num_train_timesteps: usize,
    /// Number of inference steps (set by `set_timesteps`).
    pub num_inference_steps: usize,
    /// Multistep order (`solver_order`, Python `:80`). 2 for NAVA.
    pub solver_order: usize,
    /// Flow shift parameter (Python `shift`, 5.0 for NAVA video + audio).
    pub shift: f32,
    /// Step indices where the corrector is skipped. Default empty.
    pub disable_corrector: Vec<usize>,
    /// `predict_x0` — true for flow-matching (Python `:87`).
    pub predict_x0: bool,
    /// `lower_order_final` — true (Python `:89`).
    pub lower_order_final: bool,

    /// Sigma schedule. Length = `num_inference_steps + 1`. `sigmas[N] = 0`.
    sigmas: Vec<f32>,
    /// Timesteps. Length = `num_inference_steps`. `timesteps[i] = sigmas[i] * num_train_timesteps`.
    timesteps: Vec<f32>,
    /// Ring buffer of past `convert_model_output` results. Length
    /// `solver_order`. Index `solver_order-1` is most-recent ("m0").
    model_outputs: Vec<Option<Tensor>>,
    /// Multistep warmup counter (Python `lower_order_nums`).
    lower_order_nums: usize,
    /// Sample from the previous step. Used by the corrector.
    last_sample: Option<Tensor>,
    /// Current step index (Python `_step_index`).
    step_index: usize,
    /// Order used at the previous step, consumed by the next corrector
    /// (Python `this_order`).
    this_order: usize,
}

impl NavaUniPCScheduler {
    /// Construct mirroring Python `__init__` (with NAVA's `shift`), without yet
    /// setting the inference timesteps. Call [`set_timesteps`] before [`step`].
    ///
    /// NAVA's call site (`pipeline_nava.py:85-92`) passes `shift` to `__init__`
    /// with `use_dynamic_shifting=False`, so `__init__` applies the shift to the
    /// 1000-step array. We capture the resulting `sigma_max` lazily inside
    /// `set_timesteps`; here we only stash the config.
    pub fn new(num_train_timesteps: usize, shift: f32, solver_order: usize) -> Self {
        Self {
            num_train_timesteps,
            num_inference_steps: 0,
            solver_order,
            shift,
            disable_corrector: Vec::new(),
            predict_x0: true,
            lower_order_final: true,
            sigmas: vec![0.0],
            timesteps: Vec::new(),
            model_outputs: (0..solver_order).map(|_| None).collect(),
            lower_order_nums: 0,
            last_sample: None,
            step_index: 0,
            this_order: 0,
        }
    }

    /// Convenience: NAVA defaults (`num_train_timesteps=1000`, `solver_order=2`,
    /// `shift=5.0`), already `set_timesteps(num_steps)`.
    pub fn nava_default(num_steps: usize, shift: f32) -> Self {
        let mut s = Self::new(1000, shift, 2);
        s.set_timesteps(num_steps, None);
        s
    }

    /// Flow-shift transform `s -> shift*s / (1 + (shift-1)*s)`
    /// (`fm_solvers_unipc.py:114-115` and `:192-193`).
    fn flow_shift(shift: f64, s: f64) -> f64 {
        shift * s / (1.0 + (shift - 1.0) * s)
    }

    /// `__init__` sigma endpoints with the shift already applied
    /// (`fm_solvers_unipc.py:107-118`, `:131-132`).
    ///
    /// `alphas = linspace(1, 1/N, N)[::-1]`, `sigmas = 1 - alphas`, then the
    /// shift. The reversal makes `sigmas[0] = 1 - 1/N` (largest) and
    /// `sigmas[-1] = 1 - 1 = 0`. With the shift applied to both:
    ///   `sigma_max = flow_shift(shift, (N-1)/N)`, `sigma_min = flow_shift(shift, 0) = 0`.
    fn init_sigma_endpoints(&self) -> (f64, f64) {
        let n = self.num_train_timesteps as f64;
        // sigmas[0] = 1 - alphas[0]; after [::-1], alphas[0] = 1/N → sigmas[0] = (N-1)/N.
        let s_hi = (n - 1.0) / n;
        let s_lo = 0.0_f64;
        let shift = self.shift as f64;
        (Self::flow_shift(shift, s_hi), Self::flow_shift(shift, s_lo))
    }

    /// `set_timesteps` (`fm_solvers_unipc.py:159-227`) for the
    /// `use_dynamic_shifting=False`, `final_sigmas_type="zero"` path.
    ///
    /// `shift_override` lets a caller force a different shift for this run
    /// (NAVA does not; both modalities use the construction `shift`). When
    /// `None`, the construction `shift` is used.
    pub fn set_timesteps(&mut self, num_inference_steps: usize, shift_override: Option<f32>) {
        let shift = shift_override.map(|s| s as f64).unwrap_or(self.shift as f64);
        let n_train = self.num_train_timesteps as f64;
        let (sigma_max, sigma_min) = self.init_sigma_endpoints();

        if num_inference_steps == 0 {
            self.num_inference_steps = 0;
            self.sigmas = vec![0.0];
            self.timesteps = Vec::new();
            self.reset_state();
            return;
        }

        // sigmas = linspace(sigma_max, sigma_min, n+1)[:-1]  (Python :183-185)
        let n = num_inference_steps;
        let mut sigmas_lin: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            // linspace over n+1 points then drop the last: step = (max-min)/n.
            let t = sigma_max + (sigma_min - sigma_max) * (i as f64) / (n as f64);
            sigmas_lin.push(t);
        }
        // Apply the flow shift AGAIN (Python :192-193). This is the second
        // application — see module docs on the NAVA double-shift.
        let sigmas_shifted: Vec<f64> = sigmas_lin
            .iter()
            .map(|&x| Self::flow_shift(shift, x))
            .collect();

        // timesteps = sigmas * num_train_timesteps  (Python :205). Python casts
        // to int64; we keep f32 for the scheduler and the model embeds the int.
        self.timesteps = sigmas_shifted.iter().map(|&s| (s * n_train) as f32).collect();

        // sigmas = concat([sigmas, [0]])  (final_sigmas_type="zero", Python :206)
        let mut sigmas: Vec<f32> = sigmas_shifted.iter().map(|&x| x as f32).collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;

        self.num_inference_steps = self.timesteps.len();
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.model_outputs = (0..self.solver_order).map(|_| None).collect();
        self.lower_order_nums = 0;
        self.last_sample = None;
        self.step_index = 0;
        self.this_order = 0;
    }

    pub fn sigmas(&self) -> &[f32] { &self.sigmas }
    pub fn timesteps(&self) -> &[f32] { &self.timesteps }
    pub fn step_index(&self) -> usize { self.step_index }
    pub fn num_inference_steps(&self) -> usize { self.num_inference_steps }

    /// `_sigma_to_alpha_sigma_t` (`fm_solvers_unipc.py:272-273`):
    /// for flow-matching, `alpha = 1 - sigma`.
    fn alpha_from_sigma(sigma: f64) -> f64 { 1.0 - sigma }

    /// `convert_model_output`, flow-matching `predict_x0=True` path
    /// (`fm_solvers_unipc.py:318-321`): `x0_pred = sample - sigma_t * model_output`.
    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if !self.predict_x0 {
            return Err(Error::InvalidOperation(
                "NavaUniPCScheduler: only predict_x0=true is implemented".into(),
            ));
        }
        let sigma_t = self.sigmas[self.step_index];
        let scaled = model_output.mul_scalar(sigma_t)?;
        sample.sub(&scaled)
    }

    /// Solve `R x = b` (f64 Gauss-Jordan, partial pivot). Only used if a future
    /// caller bumps `solver_order >= 3`; at NAVA's `solver_order=2` the
    /// predictor/corrector shortcuts (`rhos=[0.5]`) make this dead.
    fn linsolve_f64(r: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
        let k = b.len();
        if r.len() != k || r.iter().any(|row| row.len() != k) {
            return Err(Error::InvalidOperation("linsolve_f64: shape mismatch".into()));
        }
        let mut aug: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let mut row = r[i].clone();
                row.push(b[i]);
                row
            })
            .collect();
        for col in 0..k {
            let mut piv = col;
            let mut piv_val = aug[col][col].abs();
            for row in (col + 1)..k {
                let v = aug[row][col].abs();
                if v > piv_val { piv_val = v; piv = row; }
            }
            if piv_val < 1e-30 {
                return Err(Error::InvalidOperation("linsolve_f64: singular matrix".into()));
            }
            if piv != col { aug.swap(piv, col); }
            let pivot = aug[col][col];
            for j in col..=k { aug[col][j] /= pivot; }
            for row in 0..k {
                if row == col { continue; }
                let factor = aug[row][col];
                if factor.abs() < 1e-30 { continue; }
                for j in col..=k { aug[row][j] -= factor * aug[col][j]; }
            }
        }
        Ok((0..k).map(|i| aug[i][k]).collect())
    }

    /// bh2 coefficients for predictor (`is_corrector=false`) or corrector.
    /// Mirrors `multistep_uni_p_bh_update` / `multistep_uni_c_bh_update`
    /// (`fm_solvers_unipc.py:405-453` / `:548-596`).
    ///
    /// Returns `(rhos, B_h, alpha_t, sigma_t, sigma_s0, alpha_s0, h_phi_1, rks)`.
    #[allow(clippy::type_complexity)]
    fn compute_bh2_coefficients(
        &self,
        order: usize,
        is_corrector: bool,
    ) -> Result<(Vec<f64>, f64, f64, f64, f64, f64, f64, Vec<f64>)> {
        let (idx_t, idx_s0) = if is_corrector {
            (self.step_index, self.step_index - 1)
        } else {
            (self.step_index + 1, self.step_index)
        };
        let sigma_t = self.sigmas[idx_t] as f64;
        let sigma_s0 = self.sigmas[idx_s0] as f64;
        let alpha_t = Self::alpha_from_sigma(sigma_t);
        let alpha_s0 = Self::alpha_from_sigma(sigma_s0);

        let lg = |v: f64| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY };
        let lambda_t = lg(alpha_t) - lg(sigma_t);
        let lambda_s0 = lg(alpha_s0) - lg(sigma_s0);
        let h = lambda_t - lambda_s0;

        let mut rks: Vec<f64> = Vec::with_capacity(order);
        for i in 1..order {
            let si = if is_corrector {
                (self.step_index as isize) - (i as isize + 1)
            } else {
                (self.step_index as isize) - (i as isize)
            };
            let si = si.max(0) as usize;
            let sigma_si = self.sigmas[si] as f64;
            let alpha_si = Self::alpha_from_sigma(sigma_si);
            let lambda_si = lg(alpha_si) - lg(sigma_si);
            rks.push((lambda_si - lambda_s0) / h);
        }
        rks.push(1.0);

        let hh = if self.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp_m1();
        // bh2: B_h = expm1(hh)  (Python :441-442 / :585)
        let b_h = hh.exp_m1();

        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i: f64 = 1.0;
        let mut b_vec: Vec<f64> = Vec::with_capacity(order);
        for i in 1..=order {
            b_vec.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }

        let mut r_mat: Vec<Vec<f64>> = Vec::with_capacity(order);
        for i in 1..=order {
            r_mat.push(rks.iter().map(|&x| x.powi((i - 1) as i32)).collect());
        }

        let rhos: Vec<f64> = if order == 1 {
            vec![0.0] // placeholder; corrector caller short-circuits to [0.5]
        } else {
            Self::linsolve_f64(&r_mat, &b_vec)?
        };

        Ok((rhos, b_h, alpha_t, sigma_t, sigma_s0, alpha_s0, h_phi_1, rks))
    }

    /// Predictor `multistep_uni_p_bh_update` (`fm_solvers_unipc.py:350-484`).
    fn multistep_uni_p_bh_update(&self, sample: &Tensor, order: usize) -> Result<Tensor> {
        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("predictor: model_outputs[-1] is None".into()))?;

        let (_rhos, b_h, alpha_t, sigma_t, sigma_s0, _alpha_s0, h_phi_1, rks) =
            self.compute_bh2_coefficients(order, false)?;

        let mut d1s_tensors: Vec<Tensor> = Vec::new();
        for i in 1..order {
            let mi = self.model_outputs[self.solver_order - 1 - i]
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation(format!("predictor: model_outputs[-{}] is None", i + 1)))?;
            let rk = rks[i - 1];
            let diff = mi.sub(m0)?;
            d1s_tensors.push(diff.mul_scalar((1.0 / rk) as f32)?);
        }

        if !self.predict_x0 {
            return Err(Error::InvalidOperation("predictor: only predict_x0=true is implemented".into()));
        }
        // x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0   (Python :467)
        let coef_x = (sigma_t / sigma_s0) as f32;
        let coef_m0 = (alpha_t * h_phi_1) as f32;
        let term_x = sample.mul_scalar(coef_x)?;
        let term_m0 = m0.mul_scalar(coef_m0)?;
        let x_t_underscore = term_x.sub(&term_m0)?;

        // pred_res = einsum(rhos_p, D1s); order==2 short-circuits to rhos_p=[0.5]
        // (Python :458-459).
        let pred_res: Option<Tensor> = if d1s_tensors.is_empty() {
            None
        } else if order == 2 {
            Some(d1s_tensors[0].mul_scalar(0.5_f32)?)
        } else {
            return Err(Error::InvalidOperation(format!(
                "predictor: order={} > 2 not implemented (NAVA uses solver_order=2)", order
            )));
        };

        // x_t = x_t_ - alpha_t * B_h * pred_res   (Python :473)
        let coef_pred = (alpha_t * b_h) as f32;
        match pred_res {
            Some(pr) => x_t_underscore.sub(&pr.mul_scalar(coef_pred)?),
            None => Ok(x_t_underscore),
        }
    }

    /// Corrector `multistep_uni_c_bh_update` (`fm_solvers_unipc.py:486-626`).
    fn multistep_uni_c_bh_update(
        &self,
        this_model_output: &Tensor,
        last_sample: &Tensor,
        _this_sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("corrector: model_outputs[-1] is None".into()))?;

        let (rhos, b_h, alpha_t, sigma_t, sigma_s0, _alpha_s0, h_phi_1, rks) =
            self.compute_bh2_coefficients(order, true)?;

        let mut d1s_tensors: Vec<Tensor> = Vec::new();
        for i in 1..order {
            let mi = self.model_outputs[self.solver_order - 1 - i]
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation(format!("corrector: model_outputs[-{}] is None", i + 1)))?;
            let rk = rks[i - 1];
            let diff = mi.sub(m0)?;
            d1s_tensors.push(diff.mul_scalar((1.0 / rk) as f32)?);
        }

        if !self.predict_x0 {
            return Err(Error::InvalidOperation("corrector: only predict_x0=true is implemented".into()));
        }
        // x_t_ = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0  (Python :610)
        let coef_x = (sigma_t / sigma_s0) as f32;
        let coef_m0 = (alpha_t * h_phi_1) as f32;
        let term_x = last_sample.mul_scalar(coef_x)?;
        let term_m0 = m0.mul_scalar(coef_m0)?;
        let x_t_underscore = term_x.sub(&term_m0)?;

        // rhos_c: order==1 → [0.5] (Python :604-605); else full solve.
        let rhos_c: Vec<f64> = if order == 1 {
            vec![0.5]
        } else if order == 2 {
            rhos
        } else {
            return Err(Error::InvalidOperation(format!(
                "corrector: order={} > 2 not implemented (NAVA uses solver_order=2)", order
            )));
        };

        // corr_res = einsum(rhos_c[:-1], D1s)  (Python :612)
        let corr_res: Option<Tensor> = if d1s_tensors.is_empty() {
            None
        } else if order == 2 {
            Some(d1s_tensors[0].mul_scalar(rhos_c[0] as f32)?)
        } else {
            return Err(Error::InvalidOperation(format!("corrector: order={} > 2 not implemented", order)));
        };

        // D1_t = model_t - m0; total = corr_res + rhos_c[-1] * D1_t  (Python :615-616)
        let d1_t = this_model_output.sub(m0)?;
        let rho_last = *rhos_c.last().unwrap() as f32;
        let d1_t_scaled = d1_t.mul_scalar(rho_last)?;
        let total = match corr_res {
            Some(cr) => cr.add(&d1_t_scaled)?,
            None => d1_t_scaled,
        };

        // x_t = x_t_ - alpha_t * B_h * total   (Python :616)
        let coef_total = (alpha_t * b_h) as f32;
        x_t_underscore.sub(&total.mul_scalar(coef_total)?)
    }

    /// Main step. Mirrors Python `step` (`fm_solvers_unipc.py:655-739`).
    /// Returns `prev_sample` (the sample at the next sigma).
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if self.num_inference_steps == 0 {
            return Err(Error::InvalidOperation(
                "NavaUniPCScheduler::step: set_timesteps not called".into(),
            ));
        }
        if self.step_index >= self.num_inference_steps {
            return Err(Error::InvalidOperation(format!(
                "NavaUniPCScheduler::step: step_index {} >= num_inference_steps {}",
                self.step_index, self.num_inference_steps
            )));
        }

        // use_corrector = step_index>0 AND step_index-1 not disabled AND last_sample is Some
        let prev_idx_disabled = self.step_index > 0
            && self.disable_corrector.contains(&(self.step_index - 1));
        let use_corrector =
            self.step_index > 0 && !prev_idx_disabled && self.last_sample.is_some();

        let model_output_convert = self.convert_model_output(model_output, sample)?;

        // Corrector REPLACES `sample` in Python (:698).
        let sample_after_corrector: Tensor = if use_corrector {
            let last = self.last_sample.as_ref().expect("guard").clone();
            self.multistep_uni_c_bh_update(
                &model_output_convert, &last, sample, self.this_order,
            )?
        } else {
            sample.clone()
        };

        // Shift ring buffer left, append new converted output (Python :705-710).
        for i in 0..(self.solver_order.saturating_sub(1)) {
            self.model_outputs[i] = self.model_outputs[i + 1].clone();
        }
        if self.solver_order > 0 {
            self.model_outputs[self.solver_order - 1] = Some(model_output_convert);
        }

        // this_order (Python :712-720).
        let mut this_order = if self.lower_order_final {
            self.solver_order.min(self.timesteps.len() - self.step_index)
        } else {
            self.solver_order
        };
        this_order = this_order.min(self.lower_order_nums + 1);
        if this_order == 0 {
            return Err(Error::InvalidOperation("NavaUniPCScheduler: this_order=0".into()));
        }
        self.this_order = this_order;

        // last_sample = corrected sample (Python :723).
        self.last_sample = Some(sample_after_corrector.clone());

        let prev_sample = self.multistep_uni_p_bh_update(&sample_after_corrector, this_order)?;

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(prev_sample)
    }
}

// ---------------------------------------------------------------------------
// Two-instance lockstep driver
// ---------------------------------------------------------------------------

/// Video + audio schedulers stepped over the SAME timestep list, in lockstep
/// (`pipeline_nava.py:458-464`: `zip(timesteps, timesteps)`, both built with
/// `set_timesteps(num_steps)`).
///
/// At NAVA's config both modalities use `shift=5.0`, so the two timestep lists
/// are identical; we expose `shift_audio` separately to stay faithful to the
/// `shift` / `shift_audio` config split (`pipeline_nava.py:67-68`).
#[derive(Debug)]
pub struct NavaDualScheduler {
    pub video: NavaUniPCScheduler,
    pub audio: NavaUniPCScheduler,
}

impl NavaDualScheduler {
    /// Build both schedulers and call `set_timesteps(num_steps)` on each.
    /// `shift` drives video, `shift_audio` drives audio (both 5.0 for NAVA).
    pub fn new(num_steps: usize, shift: f32, shift_audio: f32) -> Self {
        Self {
            video: NavaUniPCScheduler::nava_default(num_steps, shift),
            audio: NavaUniPCScheduler::nava_default(num_steps, shift_audio),
        }
    }

    /// The shared timestep list the gen loop iterates over. Both modalities are
    /// stepped at each entry (`pipeline_nava.py:464`). Video's list is
    /// canonical (`timesteps = self.sample_scheduler.timesteps`, Python :460).
    pub fn timesteps(&self) -> &[f32] { self.video.timesteps() }

    /// Step both schedulers for one denoise iteration.
    /// `eps_vid`/`eps_audio` are the CFG-combined velocities for this step.
    /// Returns `(latents_vid, latents_audio)` for the next step.
    pub fn step(
        &mut self,
        eps_vid: &Tensor,
        latents_vid: &Tensor,
        eps_audio: &Tensor,
        latents_audio: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let lv = self.video.step(eps_vid, latents_vid)?;
        let la = self.audio.step(eps_audio, latents_audio)?;
        Ok((lv, la))
    }
}

// ---------------------------------------------------------------------------
// Cross-modal CFG
// ---------------------------------------------------------------------------

/// Which CFG pass a model forward corresponds to. The gen bin (built next
/// round) maps each variant to a `WanAVModel::forward` call via
/// [`NavaCfgConfig::pass_args`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavaCfgPass {
    /// Conditional: real text context, no skip-layers, no modality masking.
    /// `predict_eps(masking_modality=False)` (`pipeline_nava.py:473-486`).
    Cond,
    /// Unconditional: zero/neg text context, no masking
    /// (`pipeline_nava.py:487-500`). The pipeline passes `slg_layer=11` but the
    /// live mmdit backbone drops it (no skip) — see `NavaCfgConfig` doc.
    Uncond,
    /// 3D align: real text context, no skip-layers, `masking_modality=True`
    /// (`pipeline_nava.py:502-514`). Only when `align_3d=true`.
    Align,
    // TODO v2: timbre pass — `predict_eps(spk_embs=None, ...)` with the speaker
    // embedding ablated (`pipeline_nava.py:515-529`). Deferred; see
    // `eps_timbre_*` / `effective_timbre` in the pipeline.
}

/// Per-pass arguments the gen bin feeds to
/// `WanAVModel::forward(vid, audio, t, vid_ctx, audio_ctx, skip_layers, masking_modality)`.
///
/// `use_neg_context=true` tells the gen bin to use the zero/negative text
/// context for this pass (the uncond pass). The model forward itself takes the
/// context tensors; this struct only carries the routing decisions the
/// scheduler layer owns.
#[derive(Debug, Clone)]
pub struct NavaPassArgs {
    pub pass: NavaCfgPass,
    /// Layer indices to skip in the DiT stack (SLG). Empty for NAVA's live
    /// mmdit backbone (SLG is a no-op there); plumbing retained for generality.
    pub skip_layers: Vec<usize>,
    /// Whether to mask cross-modal attention (`masking_modality`). Only the
    /// align pass sets this true.
    pub masking_modality: bool,
    /// Whether the gen bin should pass the negative/zero text context instead
    /// of the positive one (uncond pass only).
    pub use_neg_context: bool,
}

/// CFG configuration for NAVA Base T2AV.
///
/// Defaults match `configs/nava.yaml`: video guidance 3.0, audio guidance 2.0,
/// video align 3.0, audio align 2.0, `align_3d=true`.
///
/// `slg_layer` defaults to `None`: in the LIVE NAVA backbone (`WanAVModel`,
/// mmdit single-tower) skip-layer-guidance is a NO-OP. The pipeline passes
/// `slg_layer=11` to `predict_eps` (`pipeline_nava.py:496`), which forwards it
/// to `WanAVModel.forward` (`model_nava.py:452`), but that signature
/// (`model_mm.py:1574-1590`) has no `slg_layer` param — it lands in `**kwargs`
/// and is silently dropped (`grep "slg" model_mm.py` → zero hits). The only
/// `slg_layer` consumer is `FusionModel.forward` (`fusion.py:459`), the DEAD
/// `use_mmdit_model:false` path. So the reference uncond pass runs the FULL
/// 30-block stack with no skip; the port must too.
#[derive(Debug, Clone)]
pub struct NavaCfgConfig {
    pub video_guidance: f32,
    pub audio_guidance: f32,
    pub video_align: f32,
    pub audio_align: f32,
    /// Skip-layer-guidance layer for the uncond pass. `None` for NAVA's live
    /// mmdit backbone (SLG is a no-op there — see struct doc). The plumbing is
    /// retained but defaults to empty so the uncond pass matches the reference.
    pub slg_layer: Option<usize>,
    /// Whether the 3D-align (`masking_modality`) pass is active.
    pub align_3d: bool,
    // TODO v2: timbre_cfg + timbre_align_guidance (deferred).
}

impl Default for NavaCfgConfig {
    fn default() -> Self {
        Self {
            video_guidance: 3.0,
            audio_guidance: 2.0,
            video_align: 3.0,
            audio_align: 2.0,
            slg_layer: None,
            align_3d: true,
        }
    }
}

impl NavaCfgConfig {
    /// The ordered list of forward passes the gen bin must run this step.
    /// `[Cond, Uncond]` for plain CFG; `[Cond, Uncond, Align]` when `align_3d`.
    pub fn passes(&self) -> Vec<NavaCfgPass> {
        let mut v = vec![NavaCfgPass::Cond, NavaCfgPass::Uncond];
        if self.align_3d {
            v.push(NavaCfgPass::Align);
        }
        v
    }

    /// Per-pass routing args for a given pass.
    ///   * Cond:   skip=[],  mask=false, positive context
    ///   * Uncond: skip=[],  mask=false, negative/zero context (slg is a no-op
    ///             in the live mmdit backbone — see `NavaCfgConfig` doc)
    ///   * Align:  skip=[],  mask=true,  positive context
    pub fn pass_args(&self, pass: NavaCfgPass) -> NavaPassArgs {
        match pass {
            NavaCfgPass::Cond => NavaPassArgs {
                pass,
                skip_layers: Vec::new(),
                masking_modality: false,
                use_neg_context: false,
            },
            NavaCfgPass::Uncond => NavaPassArgs {
                pass,
                skip_layers: self.slg_layer.into_iter().collect(),
                masking_modality: false,
                use_neg_context: true,
            },
            NavaCfgPass::Align => NavaPassArgs {
                pass,
                skip_layers: Vec::new(),
                masking_modality: true,
                use_neg_context: false,
            },
        }
    }
}

/// Per-modality CFG pass outputs (the `(eps_vid, eps_audio)` from each forward).
/// `align` is `None` when `align_3d=false`.
#[derive(Debug)]
pub struct NavaCfgInputs<'a> {
    pub cond_vid: &'a Tensor,
    pub cond_audio: &'a Tensor,
    pub uncond_vid: &'a Tensor,
    pub uncond_audio: &'a Tensor,
    pub align_vid: Option<&'a Tensor>,
    pub align_audio: Option<&'a Tensor>,
}

/// Cross-modal CFG combine. Returns the guided `(eps_vid, eps_audio)` fed to
/// the schedulers.
///
/// EXACT formulas (`pipeline_nava.py`):
///   * NOT align_3d (`:537`, `:544`):
///       `eps_vision = eps_uncond_vision + vision_guidance_scale*(eps_cond_vid - eps_uncond_vision)`
///       `eps_audio  = eps_uncond_audio  + audio_guidance_scale *(eps_cond_audio - eps_uncond_audio)`
///   * align_3d (`:539`, `:546`):
///       `eps_vision = eps_cond_vid   + vision_guidance_scale*(eps_cond_vid - eps_uncond_vision)
///                                    + vision_align_guidance_scale*(eps_cond_vid - eps_mmask_cond_vid)`
///       `eps_audio  = eps_cond_audio + audio_guidance_scale *(eps_cond_audio - eps_uncond_audio)
///                                    + audio_align_guidance_scale*(eps_cond_audio - eps_mmask_cond_audio)`
///
/// Note the align path's base term is `eps_cond` (NOT `eps_uncond`) — verified
/// against `pipeline_nava.py:539` and `:546`. This is intentional in NAVA and
/// differs from textbook CFG. Timbre's 4th term is DEFERRED (v2).
pub fn nava_cfg_combine(
    cfg: &NavaCfgConfig,
    inputs: &NavaCfgInputs,
) -> Result<(Tensor, Tensor)> {
    if !cfg.align_3d {
        // Plain CFG: uncond + scale*(cond - uncond)
        let eps_vid = inputs.uncond_vid.add(
            &inputs.cond_vid.sub(inputs.uncond_vid)?.mul_scalar(cfg.video_guidance)?,
        )?;
        let eps_audio = inputs.uncond_audio.add(
            &inputs.cond_audio.sub(inputs.uncond_audio)?.mul_scalar(cfg.audio_guidance)?,
        )?;
        return Ok((eps_vid, eps_audio));
    }

    // align_3d: cond + g*(cond-uncond) + a*(cond-mmask_cond)
    let align_vid = inputs.align_vid.ok_or_else(|| {
        Error::InvalidOperation("nava_cfg_combine: align_3d=true but align_vid is None".into())
    })?;
    let align_audio = inputs.align_audio.ok_or_else(|| {
        Error::InvalidOperation("nava_cfg_combine: align_3d=true but align_audio is None".into())
    })?;

    // video
    let v_guidance = inputs.cond_vid.sub(inputs.uncond_vid)?.mul_scalar(cfg.video_guidance)?;
    let v_align = inputs.cond_vid.sub(align_vid)?.mul_scalar(cfg.video_align)?;
    let eps_vid = inputs.cond_vid.add(&v_guidance)?.add(&v_align)?;

    // audio
    let a_guidance = inputs.cond_audio.sub(inputs.uncond_audio)?.mul_scalar(cfg.audio_guidance)?;
    let a_align = inputs.cond_audio.sub(align_audio)?.mul_scalar(cfg.audio_align)?;
    let eps_audio = inputs.cond_audio.add(&a_guidance)?.add(&a_align)?;

    Ok((eps_vid, eps_audio))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::{DType, Shape};
    use std::sync::Arc;
    #[allow(unused_imports)]
    use cudarc;

    fn maybe_cuda_device() -> Option<Arc<cudarc::driver::CudaDevice>> {
        cudarc::driver::CudaDevice::new(0).ok()
    }

    /// Hand-computed double-shift sigma schedule (shift=5, N=1000):
    ///   sigma_max = 5*0.999/(1+4*0.999) = 4.995/4.996 = 0.9997998...
    /// set_timesteps(2): linspace(sigma_max, 0, 3)[:-1] = [sigma_max, sigma_max/2]
    ///   shift each: flow_shift(5, sigma_max), flow_shift(5, sigma_max/2)
    #[test]
    fn sigma_schedule_double_shift_n2() {
        let sch = NavaUniPCScheduler::nava_default(2, 5.0);
        let s = sch.sigmas();
        assert_eq!(s.len(), 3, "n+1 sigmas with final zero");
        assert_eq!(sch.timesteps().len(), 2);

        // Recompute by hand in f64.
        let n = 1000.0_f64;
        let s_hi = (n - 1.0) / n; // 0.999
        let fs = |x: f64| 5.0 * x / (1.0 + 4.0 * x);
        let sigma_max = fs(s_hi);
        let lin = [sigma_max, sigma_max + (0.0 - sigma_max) * 0.5];
        let expect0 = fs(lin[0]) as f32;
        let expect1 = fs(lin[1]) as f32;

        assert!((s[0] - expect0).abs() < 1e-6, "sigma0 {} vs {}", s[0], expect0);
        assert!((s[1] - expect1).abs() < 1e-6, "sigma1 {} vs {}", s[1], expect1);
        assert_eq!(s[2], 0.0);

        // Confirm it DIFFERS from the single-shift (cosmos) schedule: a
        // single-shift n=2 would give flow_shift over linspace(0.999, 0).
        let single0 = fs(s_hi) as f32; // single shift sigma0 = fs(0.999)
        assert!(
            (s[0] - single0).abs() > 1e-5,
            "double-shift sigma0 {} should differ from single-shift {}",
            s[0], single0
        );

        // timesteps = sigmas[:-1] * 1000
        assert!((sch.timesteps()[0] - s[0] * 1000.0).abs() < 1e-2);
        assert!((sch.timesteps()[1] - s[1] * 1000.0).abs() < 1e-2);
    }

    /// Schedule is strictly decreasing and bounded in (0, 1).
    #[test]
    fn sigma_schedule_monotonic_n3() {
        let sch = NavaUniPCScheduler::nava_default(3, 5.0);
        let s = sch.sigmas();
        assert_eq!(s.len(), 4);
        for w in s.windows(2) {
            assert!(w[0] > w[1], "sigmas must decrease: {} !> {}", w[0], w[1]);
        }
        assert!(s[0] < 1.0 && s[0] > 0.99);
        assert_eq!(*s.last().unwrap(), 0.0);
    }

    #[test]
    fn dual_scheduler_shares_timesteps() {
        let dual = NavaDualScheduler::new(4, 5.0, 5.0);
        assert_eq!(dual.video.timesteps(), dual.audio.timesteps(),
            "equal shift → identical timestep lists");
        assert_eq!(dual.timesteps().len(), 4);
    }

    #[test]
    fn cfg_passes_order() {
        let cfg = NavaCfgConfig::default();
        assert_eq!(cfg.passes(), vec![NavaCfgPass::Cond, NavaCfgPass::Uncond, NavaCfgPass::Align]);

        let plain = NavaCfgConfig { align_3d: false, ..NavaCfgConfig::default() };
        assert_eq!(plain.passes(), vec![NavaCfgPass::Cond, NavaCfgPass::Uncond]);
    }

    #[test]
    fn cfg_pass_args_routing() {
        let cfg = NavaCfgConfig::default();
        let cond = cfg.pass_args(NavaCfgPass::Cond);
        assert!(cond.skip_layers.is_empty() && !cond.masking_modality && !cond.use_neg_context);

        // SLG is a no-op in the live mmdit backbone: the uncond pass must run
        // ALL blocks (skip empty) to match the reference (pipeline passes
        // slg_layer=11 but WanAVModel.forward drops it).
        let uncond = cfg.pass_args(NavaCfgPass::Uncond);
        assert!(uncond.skip_layers.is_empty());
        assert!(!uncond.masking_modality && uncond.use_neg_context);

        let align = cfg.pass_args(NavaCfgPass::Align);
        assert!(align.skip_layers.is_empty() && align.masking_modality && !align.use_neg_context);
    }

    /// CFG combine arithmetic on toy tensors, checked against the exact
    /// pipeline_nava.py formulas.
    #[test]
    fn cfg_combine_align3d_arithmetic() {
        let dev = match maybe_cuda_device() { Some(d) => d, None => return };
        let mk = |vals: &[f32]| Tensor::from_vec(
            vals.to_vec(), Shape::from_dims(&[1, vals.len()]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let cond_v = mk(&[1.0, 2.0, 4.0]);
        let uncond_v = mk(&[0.5, 1.0, 2.0]);
        let align_v = mk(&[0.25, 0.5, 1.0]);
        let cond_a = mk(&[2.0, 4.0, 8.0]);
        let uncond_a = mk(&[1.0, 2.0, 4.0]);
        let align_a = mk(&[0.5, 1.0, 2.0]);

        let cfg = NavaCfgConfig::default(); // vg=3, ag=2, va=3, aa=2, align_3d
        let inputs = NavaCfgInputs {
            cond_vid: &cond_v, cond_audio: &cond_a,
            uncond_vid: &uncond_v, uncond_audio: &uncond_a,
            align_vid: Some(&align_v), align_audio: Some(&align_a),
        };
        let (ev, ea) = nava_cfg_combine(&cfg, &inputs).unwrap();
        let ev = ev.to_vec().unwrap();
        let ea = ea.to_vec().unwrap();

        // video: cond + 3*(cond-uncond) + 3*(cond-align)
        for i in 0..3 {
            let c = [1.0, 2.0, 4.0][i];
            let u = [0.5, 1.0, 2.0][i];
            let al = [0.25, 0.5, 1.0][i];
            let want = c + 3.0 * (c - u) + 3.0 * (c - al);
            assert!((ev[i] - want).abs() < 0.05, "video[{i}] {} vs {}", ev[i], want);
        }
        // audio: cond + 2*(cond-uncond) + 2*(cond-align)
        for i in 0..3 {
            let c = [2.0, 4.0, 8.0][i];
            let u = [1.0, 2.0, 4.0][i];
            let al = [0.5, 1.0, 2.0][i];
            let want = c + 2.0 * (c - u) + 2.0 * (c - al);
            assert!((ea[i] - want).abs() < 0.1, "audio[{i}] {} vs {}", ea[i], want);
        }
    }

    #[test]
    fn cfg_combine_plain_arithmetic() {
        let dev = match maybe_cuda_device() { Some(d) => d, None => return };
        let mk = |vals: &[f32]| Tensor::from_vec(
            vals.to_vec(), Shape::from_dims(&[1, vals.len()]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let cond_v = mk(&[1.0, 2.0]);
        let uncond_v = mk(&[0.5, 1.0]);
        let cond_a = mk(&[2.0, 4.0]);
        let uncond_a = mk(&[1.0, 2.0]);

        let cfg = NavaCfgConfig { align_3d: false, ..NavaCfgConfig::default() };
        let inputs = NavaCfgInputs {
            cond_vid: &cond_v, cond_audio: &cond_a,
            uncond_vid: &uncond_v, uncond_audio: &uncond_a,
            align_vid: None, align_audio: None,
        };
        let (ev, ea) = nava_cfg_combine(&cfg, &inputs).unwrap();
        let ev = ev.to_vec().unwrap();
        let ea = ea.to_vec().unwrap();

        // video: uncond + 3*(cond-uncond)
        for i in 0..2 {
            let c = [1.0, 2.0][i];
            let u = [0.5, 1.0][i];
            let want = u + 3.0 * (c - u);
            assert!((ev[i] - want).abs() < 0.05, "video[{i}] {} vs {}", ev[i], want);
        }
        // audio: uncond + 2*(cond-uncond)
        for i in 0..2 {
            let c = [2.0, 4.0][i];
            let u = [1.0, 2.0][i];
            let want = u + 2.0 * (c - u);
            assert!((ea[i] - want).abs() < 0.05, "audio[{i}] {} vs {}", ea[i], want);
        }
    }

    #[test]
    fn step_advances_and_runs() {
        let dev = match maybe_cuda_device() { Some(d) => d, None => return };
        let mut sch = NavaUniPCScheduler::nava_default(5, 5.0);
        assert_eq!(sch.step_index(), 0);
        let n = 8;
        let x = Tensor::from_vec(
            (0..n).map(|i| 0.1 + i as f32 * 0.01).collect(),
            Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let v = Tensor::from_vec(
            (0..n).map(|i| -0.05 + i as f32 * 0.02).collect(),
            Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let _ = sch.step(&v, &x).expect("step 0");
        assert_eq!(sch.step_index(), 1);
        assert_eq!(sch.this_order, 1, "warmup: first step is order 1");
        let _ = sch.step(&v, &x).expect("step 1");
        assert_eq!(sch.step_index(), 2);
        assert_eq!(sch.this_order, 2, "second step uses full order 2");
    }
}
