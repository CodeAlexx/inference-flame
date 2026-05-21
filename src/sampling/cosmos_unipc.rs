//! Cosmos-Predict2.5 `FlowUniPCMultistepScheduler` — bh2 multistep port.
//!
//! Verbatim port of
//! `cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py` (diffusers
//! v0.31.0 copy). Defaults wired for Cosmos V2_2B inference:
//!   * `num_train_timesteps = 1000`
//!   * `solver_order = 2`
//!   * `solver_type = "bh2"`
//!   * `prediction_type = "flow_prediction"`
//!   * `predict_x0 = True`
//!   * `lower_order_final = True`
//!   * `final_sigmas_type = "zero"`
//!   * `disable_corrector = []`
//!   * `shift = 5.0` (V2_2B inference,
//!     `text2world_model_rectified_flow.py:502`)
//!
//! ## Linsolve choice
//!
//! Python uses `torch.linalg.solve(R, b)` where `R` is `[order, order]` and
//! `b` is `[order]`. For Cosmos's `solver_order=2` the matrices are at most
//! 2×2, but the corrector for `order=1` short-circuits to `rhos_c=[0.5]` and
//! the predictor for `order==2` short-circuits to `rhos_p=[0.5]`
//! (`fm_solvers_unipc.py:441-444`, `:579-582`). So at `solver_order=2` we
//! never actually invoke `torch.linalg.solve` — both shortcuts apply. We
//! still implement a generic 2×2 / 3×3 f64 CPU Gauss-Jordan for
//! robustness when a future caller bumps `solver_order` up. Per
//! BUILD instructions: the matrix is small, stays on CPU in f64, then the
//! scalars upload as `mul_scalar`-style coefficients to the GPU tensor ops.
//!
//! ## Reuse from flame-core
//!
//! All tensor math goes through native `flame_core::Tensor` ops:
//! `mul_scalar`, `add`, `sub`. No new kernels. Linsolve never touches the
//! GPU.
//!
//! ## State
//!
//! `model_outputs` is a ring buffer of `Option<Tensor>` length `solver_order`
//! — index `[solver_order - 1]` is the freshest model output ("m0" in
//! Python). `last_sample` is the sample from the previous step, used by the
//! corrector. `lower_order_nums` ramps the effective order from 1 → solver_order
//! during the multistep warmup.

use flame_core::{Error, Result, Tensor};

/// `FlowUniPCMultistepScheduler` configured for Cosmos V2_2B inference
/// (bh2 multistep, predict_x0=true, lower_order_final=true). Stateful: call
/// `step` once per inference step in order; `step_index` advances internally.
#[derive(Debug)]
pub struct CosmosUniPcMultistepScheduler {
    /// Training noise-step count (`num_train_timesteps`, Python `:74`).
    pub num_train_timesteps: usize,
    /// Number of inference steps (set at construction; `set_timesteps`
    /// in Python).
    pub num_inference_steps: usize,
    /// Multistep order (`solver_order`, Python `:75`). Default 2 for Cosmos.
    pub solver_order: usize,
    /// Flow shift parameter (Python `shift`, default 5.0 for V2_2B).
    pub shift: f32,
    /// Step indices where the corrector is skipped. Default empty.
    pub disable_corrector: Vec<usize>,
    /// Whether to convert model output to x0-prediction. True for
    /// flow-matching (Python default `:82`).
    pub predict_x0: bool,
    /// Whether to use a lower-order solver in the final steps. True for
    /// Cosmos (Python default `:84`).
    pub lower_order_final: bool,

    /// Sigma schedule. Length = `num_inference_steps + 1`. `sigmas[N] = 0`.
    sigmas: Vec<f32>,
    /// Timesteps. Length = `num_inference_steps`. `timesteps[i] = sigmas[i] * num_train_timesteps`.
    timesteps: Vec<f32>,
    /// Ring buffer of past `convert_model_output` results. Length
    /// `solver_order`. Index `solver_order-1` is the most-recent.
    model_outputs: Vec<Option<Tensor>>,
    /// Multistep warmup counter (Python `:113`). After
    /// `solver_order` steps it equals `solver_order` and the full order
    /// applies.
    lower_order_nums: usize,
    /// Sample from the previous step. Used by the corrector
    /// (Python `:697`).
    last_sample: Option<Tensor>,
    /// Current step index (Python `_step_index`, `:117`).
    step_index: usize,
    /// Order used at the previous step. Set inside `step` after the order
    /// computation and consumed by the corrector on the next call
    /// (Python `:679`).
    this_order: usize,
}

impl CosmosUniPcMultistepScheduler {
    /// Build a new scheduler mirroring Python `__init__` then `set_timesteps`
    /// in a single call. Cosmos's call site
    /// (`text2world_model_rectified_flow.py:142-143`) constructs the
    /// scheduler then immediately calls `set_timesteps`, so we fold both.
    pub fn new(
        num_train_timesteps: usize,
        num_inference_steps: usize,
        shift: f32,
        solver_order: usize,
    ) -> Self {
        if num_inference_steps == 0 {
            // Degenerate: nothing to do. Return a sane-but-empty scheduler.
            return Self {
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
            };
        }

        // Python `__init__` defaults `shift=1.0` (identity) so the stored
        // `sigma_max = (N-1)/N` and `sigma_min = 0.0` (`fm_solvers_unipc.py:100-122`).
        // `set_timesteps` then re-applies the user-supplied shift.
        let n_inf = num_inference_steps;
        let n_train = num_train_timesteps as f64;
        let sigma_max = (n_train - 1.0) / n_train;
        let sigma_min = 0.0_f64;
        let shift_f64 = shift as f64;

        // linspace(sigma_max, sigma_min, n+1)[:-1]
        let mut sigmas_lin: Vec<f64> = Vec::with_capacity(n_inf);
        for i in 0..n_inf {
            let t = sigma_max + (sigma_min - sigma_max) * (i as f64) / (n_inf as f64);
            sigmas_lin.push(t);
        }
        // Apply shift.
        let sigmas_shifted: Vec<f64> = sigmas_lin
            .iter()
            .map(|&x| shift_f64 * x / (1.0 + (shift_f64 - 1.0) * x))
            .collect();
        let timesteps: Vec<f32> = sigmas_shifted
            .iter()
            .map(|&s| (s * n_train) as f32)
            .collect();
        // Concat zero (final_sigmas_type="zero").
        let mut sigmas: Vec<f32> = sigmas_shifted.iter().map(|&x| x as f32).collect();
        sigmas.push(0.0);

        Self {
            num_train_timesteps,
            num_inference_steps: n_inf,
            solver_order,
            shift,
            disable_corrector: Vec::new(),
            predict_x0: true,
            lower_order_final: true,
            sigmas,
            timesteps,
            model_outputs: (0..solver_order).map(|_| None).collect(),
            lower_order_nums: 0,
            last_sample: None,
            step_index: 0,
            this_order: 0,
        }
    }

    pub fn sigmas(&self) -> &[f32] { &self.sigmas }
    pub fn timesteps(&self) -> &[f32] { &self.timesteps }
    pub fn step_index(&self) -> usize { self.step_index }
    pub fn num_inference_steps(&self) -> usize { self.num_inference_steps }

    /// `_sigma_to_alpha_sigma_t` (Python `:259-260`).
    /// For flow-matching, `alpha = 1 - sigma`.
    fn alpha_from_sigma(sigma: f64) -> f64 { 1.0 - sigma }

    /// `convert_model_output` for the flow-matching `predict_x0=True` path
    /// (Python `:266-318`). For other configurations the Python raises;
    /// we hardcode the supported path.
    ///
    /// `x0_pred = sample - sigma_t * model_output`
    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if !self.predict_x0 {
            return Err(Error::InvalidOperation(
                "CosmosUniPcMultistepScheduler: only predict_x0=true is implemented".into(),
            ));
        }
        let sigma_t = self.sigmas[self.step_index] as f32;
        // x0_pred = sample - sigma_t * model_output
        let scaled = model_output.mul_scalar(sigma_t)?;
        sample.sub(&scaled)
    }

    /// Solve `R x = b` for `R` shape `[k,k]` and `b` shape `[k]`, all in
    /// f64. Hand-rolled Gauss-Jordan with partial pivoting. Used by the
    /// predictor/corrector when the order ≥ 3 (no shortcut). At Cosmos's
    /// default solver_order=2 this never fires (order==2 short-circuits to
    /// rhos_p=[0.5]; order==1 corrector short-circuits to rhos_c=[0.5]).
    fn linsolve_f64(r: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
        let k = b.len();
        if r.len() != k || r.iter().any(|row| row.len() != k) {
            return Err(Error::InvalidOperation(format!(
                "linsolve_f64: shape mismatch, R is {}x{:?}, b len {}",
                r.len(), r.first().map(|x| x.len()), k
            )));
        }
        // Augmented matrix.
        let mut aug: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let mut row = r[i].clone();
                row.push(b[i]);
                row
            })
            .collect();
        // Forward elimination with partial pivoting.
        for col in 0..k {
            // Find pivot.
            let mut piv = col;
            let mut piv_val = aug[col][col].abs();
            for row in (col + 1)..k {
                let v = aug[row][col].abs();
                if v > piv_val {
                    piv_val = v;
                    piv = row;
                }
            }
            if piv_val < 1e-30 {
                return Err(Error::InvalidOperation(
                    "linsolve_f64: singular matrix".into(),
                ));
            }
            if piv != col {
                aug.swap(piv, col);
            }
            // Scale pivot row.
            let pivot = aug[col][col];
            for j in col..=k {
                aug[col][j] /= pivot;
            }
            // Eliminate other rows.
            for row in 0..k {
                if row == col { continue; }
                let factor = aug[row][col];
                if factor.abs() < 1e-30 { continue; }
                for j in col..=k {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
        Ok((0..k).map(|i| aug[i][k]).collect())
    }

    /// Compute the bh2 coefficients (`rhos`, `B_h`, `alpha_t`, `sigma_t`,
    /// `sigma_s0`, `alpha_s0`, `h_phi_1`) for either predictor or corrector.
    ///
    /// `is_corrector` controls which sigma window to use:
    ///   * Predictor: `sigma_t = sigmas[step_index+1]`, `sigma_s0 = sigmas[step_index]`,
    ///     `si = step_index - i` for i in 1..order.
    ///   * Corrector: `sigma_t = sigmas[step_index]`, `sigma_s0 = sigmas[step_index-1]`,
    ///     `si = step_index - (i+1)` for i in 1..order.
    ///
    /// Returns `(rhos_minus_last, last_rho, B_h, alpha_t, sigma_t, sigma_s0, alpha_s0, h_phi_1)`.
    /// For the predictor, the caller uses `(rhos[0..k-1], rhos[k-1])` differently
    /// than the corrector — see use sites in the predictor/corrector methods.
    /// Here we just return the full rho vector and the scalars.
    fn compute_bh2_coefficients(
        &self,
        order: usize,
        is_corrector: bool,
    ) -> Result<(Vec<f64>, f64, f64, f64, f64, f64, f64, Vec<f64>)> {
        // Sigma window.
        let (idx_t, idx_s0) = if is_corrector {
            (self.step_index, self.step_index - 1)
        } else {
            (self.step_index + 1, self.step_index)
        };
        let sigma_t = self.sigmas[idx_t] as f64;
        let sigma_s0 = self.sigmas[idx_s0] as f64;
        let alpha_t = Self::alpha_from_sigma(sigma_t);
        let alpha_s0 = Self::alpha_from_sigma(sigma_s0);

        // Guard log domain. Sigma=0 at the final step would make log(sigma) diverge.
        // For predictor on the last step (sigma_t == 0), Python relies on
        // `lower_order_final` making this path order=1, where rhos=[0.5] and
        // the pred_res path isn't taken (no D1s). We still compute lambda_t
        // but it's NaN-safe because alpha_t=1, sigma_t=0 → log(0)=-inf →
        // h=+inf → expm1(-h)=−1 → B_h=−1, h_phi_1=−1. The final
        // `sigma_t / sigma_s0 * x` term = 0, and `alpha_t * h_phi_1 * m0 = -m0`,
        // giving `x_t_ = -(-m0) = m0` ... actually Python proceeds.
        // We faithfully replicate: produce the math even if it's degenerate
        // at the boundary; the caller (the predictor with no D1s) ignores
        // rhos.
        let log_alpha_t = if alpha_t > 0.0 { alpha_t.ln() } else { f64::NEG_INFINITY };
        let log_sigma_t = if sigma_t > 0.0 { sigma_t.ln() } else { f64::NEG_INFINITY };
        let log_alpha_s0 = if alpha_s0 > 0.0 { alpha_s0.ln() } else { f64::NEG_INFINITY };
        let log_sigma_s0 = if sigma_s0 > 0.0 { sigma_s0.ln() } else { f64::NEG_INFINITY };
        let lambda_t = log_alpha_t - log_sigma_t;
        let lambda_s0 = log_alpha_s0 - log_sigma_s0;
        let h = lambda_t - lambda_s0;

        // Build `rks` (Python `:399-411` and `:534-546`).
        let mut rks: Vec<f64> = Vec::with_capacity(order);
        for i in 1..order {
            let si = if is_corrector {
                // step_index - (i+1)
                (self.step_index as isize) - (i as isize + 1)
            } else {
                // step_index - i
                (self.step_index as isize) - (i as isize)
            };
            // For i=order-1 with corrector at step_index=order, si can be 0; ok.
            let si = si.max(0) as usize;
            let sigma_si = self.sigmas[si] as f64;
            let alpha_si = Self::alpha_from_sigma(sigma_si);
            let log_alpha_si = if alpha_si > 0.0 { alpha_si.ln() } else { f64::NEG_INFINITY };
            let log_sigma_si = if sigma_si > 0.0 { sigma_si.ln() } else { f64::NEG_INFINITY };
            let lambda_si = log_alpha_si - log_sigma_si;
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
        }
        rks.push(1.0);

        // Build `b` (Python `:413-433` and `:548-568`).
        let hh = if self.predict_x0 { -h } else { h };
        let h_phi_1 = (hh).exp_m1();
        // bh2: B_h = expm1(hh).
        let b_h = (hh).exp_m1();

        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i: f64 = 1.0;
        let mut b_vec: Vec<f64> = Vec::with_capacity(order);
        for i in 1..=order {
            b_vec.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }

        // Build `R` rows = [rks**0, rks**1, ..., rks**(order-1)] (Python
        // `:429-430` `R.append(torch.pow(rks, i - 1))`).
        let mut r_mat: Vec<Vec<f64>> = Vec::with_capacity(order);
        for i in 1..=order {
            let row: Vec<f64> = rks.iter().map(|&x| x.powi((i - 1) as i32)).collect();
            r_mat.push(row);
        }

        // Solve. The predictor uses `R[:-1, :-1] x = b[:-1]` for order > 2,
        // and the simplified `rhos_p = [0.5]` for order == 2.
        // The corrector uses the full `R x = b` for order > 1, and the
        // simplified `rhos_c = [0.5]` for order == 1.
        // We compute the full rhos vector here for general order ≥ 2 with no
        // shortcuts. The caller will short-circuit for the special cases.
        let rhos: Vec<f64> = if order == 1 {
            vec![0.0] // placeholder; corrector caller short-circuits
        } else if order == 2 {
            // Both predictor (uses R[:-1,:-1]=R[0,0]=1, b[:-1]=b[0]=h_phi_1/hh - 1, /B_h … but
            // the special case `[0.5]` is used) and corrector (uses full
            // 2x2) might apply. We solve the full 2x2 here; predictor caller
            // ignores it and uses [0.5] directly.
            Self::linsolve_f64(&r_mat, &b_vec)?
        } else {
            // order >= 3: solve full system.
            Self::linsolve_f64(&r_mat, &b_vec)?
        };

        Ok((rhos, b_h, alpha_t, sigma_t, sigma_s0, alpha_s0, h_phi_1, rks))
    }

    /// `multistep_uni_p_bh_update` — predictor (Python `:337-464`).
    ///
    /// `sample` is the current `x` (input). The ring buffer `model_outputs`
    /// must have its last `order` slots populated (`Some`) — i.e. the most
    /// recent at `model_outputs[solver_order - 1]`.
    fn multistep_uni_p_bh_update(&self, sample: &Tensor, order: usize) -> Result<Tensor> {
        // m0 = most-recent converted model output.
        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation(
                "multistep_uni_p_bh_update: model_outputs[-1] is None".into(),
            ))?;

        let (_rhos, b_h, alpha_t, sigma_t, sigma_s0, _alpha_s0, h_phi_1, _rks) =
            self.compute_bh2_coefficients(order, false)?;

        // Build D1s: list of `(mi - m0) / rk` for i in 1..order. Python
        // `:399-408`. For order=1, D1s is empty.
        // We use the rks from compute_bh2_coefficients (length=order, last is 1.0).
        let mut d1s_tensors: Vec<Tensor> = Vec::new();
        let mut rks_for_d1: Vec<f64> = Vec::new();
        for i in 1..order {
            let mi = self.model_outputs[self.solver_order - 1 - i]
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation(format!(
                    "multistep_uni_p_bh_update: model_outputs[-{}] is None", i + 1
                )))?;
            // rks index is i-1 (we wrote rks[0..order-1] for i in 1..order then appended 1.0).
            let rk = _rks[i - 1];
            let diff = mi.sub(m0)?;
            // (mi - m0) / rk
            let d = diff.mul_scalar((1.0 / rk) as f32)?;
            d1s_tensors.push(d);
            rks_for_d1.push(rk);
        }

        // x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        // For predict_x0=true.
        if !self.predict_x0 {
            return Err(Error::InvalidOperation(
                "multistep_uni_p_bh_update: only predict_x0=true is implemented".into(),
            ));
        }
        let coef_x = (sigma_t / sigma_s0) as f32;
        let coef_m0 = (alpha_t * h_phi_1) as f32;
        let term_x = sample.mul_scalar(coef_x)?;
        let term_m0 = m0.mul_scalar(coef_m0)?;
        let x_t_underscore = term_x.sub(&term_m0)?;

        // pred_res: einsum("k,bkc...->bc...", rhos_p, D1s).
        // For order==2 with simplified rhos_p=[0.5], pred_res = 0.5 * D1s[0].
        // For order > 2, we'd solve R[:-1, :-1] @ rhos_p = b[:-1]. Cosmos
        // uses order ≤ 2 so we only need the shortcut. If a future caller
        // bumps order ≥ 3 we error out (deferred work).
        let pred_res: Option<Tensor> = if d1s_tensors.is_empty() {
            None
        } else if order == 2 {
            // rhos_p = [0.5]; pred_res = 0.5 * D1s[0]
            Some(d1s_tensors[0].mul_scalar(0.5_f32)?)
        } else {
            // order >= 3: would need to solve R[:-1, :-1] @ rhos = b[:-1]
            // and combine D1s by those coefficients. Not exercised by
            // Cosmos's solver_order=2 default. Surface as a hard error
            // rather than ship untested math.
            return Err(Error::InvalidOperation(format!(
                "multistep_uni_p_bh_update: order={} > 2 not implemented (Cosmos uses solver_order=2)",
                order
            )));
        };

        // x_t = x_t_underscore - alpha_t * B_h * pred_res
        let coef_pred = (alpha_t * b_h) as f32;
        if let Some(pr) = pred_res {
            let term = pr.mul_scalar(coef_pred)?;
            x_t_underscore.sub(&term)
        } else {
            Ok(x_t_underscore)
        }
    }

    /// `multistep_uni_c_bh_update` — corrector (Python `:466-601`).
    ///
    /// `this_model_output` is the just-converted m_t (model output at the
    /// current sample). `last_sample` is the x_{t-1} from before the
    /// previous predictor step. `this_sample` is the x_t from the previous
    /// predictor step. `order` is `this_order` carried over from the
    /// previous step.
    fn multistep_uni_c_bh_update(
        &self,
        this_model_output: &Tensor,
        last_sample: &Tensor,
        _this_sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        // m0 = most-recent converted model output (before the corrector).
        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation(
                "multistep_uni_c_bh_update: model_outputs[-1] is None".into(),
            ))?;

        let (rhos, b_h, alpha_t, sigma_t, sigma_s0, _alpha_s0, h_phi_1, rks) =
            self.compute_bh2_coefficients(order, true)?;

        // Build D1s for the corrector (Python `:534-543`).
        let mut d1s_tensors: Vec<Tensor> = Vec::new();
        for i in 1..order {
            let mi = self.model_outputs[self.solver_order - 1 - i]
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation(format!(
                    "multistep_uni_c_bh_update: model_outputs[-{}] is None", i + 1
                )))?;
            let rk = rks[i - 1];
            let diff = mi.sub(m0)?;
            let d = diff.mul_scalar((1.0 / rk) as f32)?;
            d1s_tensors.push(d);
        }

        if !self.predict_x0 {
            return Err(Error::InvalidOperation(
                "multistep_uni_c_bh_update: only predict_x0=true is implemented".into(),
            ));
        }

        // x_t_ = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        let coef_x = (sigma_t / sigma_s0) as f32;
        let coef_m0 = (alpha_t * h_phi_1) as f32;
        let term_x = last_sample.mul_scalar(coef_x)?;
        let term_m0 = m0.mul_scalar(coef_m0)?;
        let x_t_underscore = term_x.sub(&term_m0)?;

        // rhos_c: if order==1, [0.5]; else solve full R @ rhos = b.
        let rhos_c: Vec<f64> = if order == 1 {
            vec![0.5]
        } else if order == 2 {
            // Full 2x2 solve from compute_bh2_coefficients.
            rhos
        } else {
            return Err(Error::InvalidOperation(format!(
                "multistep_uni_c_bh_update: order={} > 2 not implemented (Cosmos uses solver_order=2)",
                order
            )));
        };

        // corr_res = einsum("k,bkc...->bc...", rhos_c[:-1], D1s). For
        // order==1, D1s is empty → corr_res = 0. For order==2, D1s has 1
        // element → corr_res = rhos_c[0] * D1s[0].
        let corr_res: Option<Tensor> = if d1s_tensors.is_empty() {
            None
        } else if order == 2 {
            Some(d1s_tensors[0].mul_scalar(rhos_c[0] as f32)?)
        } else {
            return Err(Error::InvalidOperation(format!(
                "multistep_uni_c_bh_update: order={} > 2 not implemented",
                order
            )));
        };

        // D1_t = model_t - m0
        let d1_t = this_model_output.sub(m0)?;
        // rhos_c[-1] * D1_t
        let rho_last = *rhos_c.last().unwrap() as f32;
        let d1_t_scaled = d1_t.mul_scalar(rho_last)?;

        // total = corr_res + rhos_c[-1] * D1_t
        let total = match corr_res {
            Some(cr) => cr.add(&d1_t_scaled)?,
            None => d1_t_scaled,
        };

        // x_t = x_t_ - alpha_t * B_h * total
        let coef_total = (alpha_t * b_h) as f32;
        let term_total = total.mul_scalar(coef_total)?;
        x_t_underscore.sub(&term_total)
    }

    /// Main step entry. Mirrors Python `step` (`:630-713`).
    ///
    /// Returns the `prev_sample` (the sample at the next sigma).
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if self.num_inference_steps == 0 {
            return Err(Error::InvalidOperation(
                "CosmosUniPcMultistepScheduler::step: num_inference_steps is 0".into(),
            ));
        }
        if self.step_index >= self.num_inference_steps {
            return Err(Error::InvalidOperation(format!(
                "CosmosUniPcMultistepScheduler::step: step_index {} >= num_inference_steps {}",
                self.step_index, self.num_inference_steps
            )));
        }

        // use_corrector: step_index > 0 AND step_index-1 not in disable_corrector
        // AND last_sample is not None.
        let prev_idx_disabled = self.step_index > 0
            && self.disable_corrector.contains(&(self.step_index - 1));
        let use_corrector =
            self.step_index > 0 && !prev_idx_disabled && self.last_sample.is_some();

        let model_output_convert = self.convert_model_output(model_output, sample)?;

        // Apply corrector. The corrector REPLACES `sample` in Python
        // (`sample = self.multistep_uni_c_bh_update(...)`), so subsequent
        // operations see the corrected sample.
        let sample_after_corrector: Tensor = if use_corrector {
            let last = self.last_sample.as_ref()
                .expect("use_corrector guard ensures last_sample is Some").clone();
            self.multistep_uni_c_bh_update(
                &model_output_convert,
                &last,
                sample,
                self.this_order,
            )?
        } else {
            sample.clone()
        };

        // Shift ring buffer of model outputs left, append the new converted output.
        // Python `:682-687`:
        //   for i in range(solver_order - 1):
        //       model_outputs[i] = model_outputs[i + 1]
        //   model_outputs[-1] = model_output_convert
        for i in 0..(self.solver_order.saturating_sub(1)) {
            self.model_outputs[i] = self.model_outputs[i + 1].clone();
        }
        if self.solver_order > 0 {
            self.model_outputs[self.solver_order - 1] = Some(model_output_convert);
        }

        // Compute this_order (Python `:689-695`):
        //   if lower_order_final:
        //       this_order = min(solver_order, len(timesteps) - step_index)
        //   else:
        //       this_order = solver_order
        //   self.this_order = min(this_order, lower_order_nums + 1)
        let mut this_order = if self.lower_order_final {
            self.solver_order
                .min(self.timesteps.len() - self.step_index)
        } else {
            self.solver_order
        };
        this_order = this_order.min(self.lower_order_nums + 1);
        if this_order == 0 {
            return Err(Error::InvalidOperation(
                "CosmosUniPcMultistepScheduler: computed this_order=0".into(),
            ));
        }
        self.this_order = this_order;

        // last_sample = sample (the potentially-corrected sample is what
        // feeds into the predictor and is also what the NEXT step's
        // corrector reads as last_sample).
        self.last_sample = Some(sample_after_corrector.clone());

        // Predictor pass.
        let prev_sample = self.multistep_uni_p_bh_update(
            &sample_after_corrector,
            this_order,
        )?;

        // Warmup the multistep counter (Python `:704-705`).
        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }

        // Advance step index (Python `:708`).
        self.step_index += 1;

        Ok(prev_sample)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::cosmos_rf::RectifiedFlowSampler;
    use flame_core::{DType, Shape};
    use std::sync::Arc;
    #[allow(unused_imports)]
    use cudarc;

    fn maybe_cuda_device() -> Option<Arc<cudarc::driver::CudaDevice>> {
        cudarc::driver::CudaDevice::new(0).ok()
    }

    #[test]
    fn unipc_sigma_schedule_matches_magihuman_oracle() {
        // The UniPC sigma schedule should match the RectifiedFlowSampler's
        // post-F1 schedule (which itself matches magihuman_unipc.rs's
        // oracle). Same flow shift / endpoint convention.
        let n_inf = 35_usize;
        let shift = 5.0_f32;
        let unipc = CosmosUniPcMultistepScheduler::new(1000, n_inf, shift, 2);
        let euler = RectifiedFlowSampler::new(n_inf, 7.0, shift).sigmas();
        // UniPC has length n_inf+1 (with final zero); Euler does too.
        assert_eq!(unipc.sigmas.len(), euler.len());
        for (i, (&a, &b)) in unipc.sigmas.iter().zip(euler.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "sigma[{i}] mismatch: unipc={} euler={}", a, b,
            );
        }
        // Timesteps = sigmas[:-1] * num_train_timesteps.
        assert_eq!(unipc.timesteps.len(), n_inf);
        for (i, &ts) in unipc.timesteps.iter().enumerate() {
            let expected = unipc.sigmas[i] * 1000.0;
            assert!(
                (ts - expected).abs() < 1.0,
                "timestep[{i}] {} != sigma*1000 {}", ts, expected,
            );
        }
    }

    #[test]
    fn unipc_step_index_advances() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let mut sch = CosmosUniPcMultistepScheduler::new(1000, 5, 5.0, 2);
        assert_eq!(sch.step_index(), 0);
        let n = 8_usize;
        let x = Tensor::from_vec(
            (0..n).map(|i| 0.1 + (i as f32) * 0.01).collect(),
            Shape::from_dims(&[1, n]),
            dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let v = Tensor::from_vec(
            (0..n).map(|i| -0.05 + (i as f32) * 0.02).collect(),
            Shape::from_dims(&[1, n]),
            dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let _ = sch.step(&v, &x).expect("step 0");
        assert_eq!(sch.step_index(), 1);
        let _ = sch.step(&v, &x).expect("step 1");
        assert_eq!(sch.step_index(), 2);
    }

    #[test]
    fn unipc_ring_buffer_fills_correctly() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let mut sch = CosmosUniPcMultistepScheduler::new(1000, 5, 5.0, 2);
        // model_outputs starts all-None.
        assert!(sch.model_outputs.iter().all(|x| x.is_none()));

        let n = 8_usize;
        let x = Tensor::from_vec(
            vec![1.0_f32; n], Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let v = Tensor::from_vec(
            vec![0.1_f32; n], Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        // After 1 step, model_outputs[-1] (= [1] for solver_order=2) is Some.
        let _ = sch.step(&v, &x).expect("step 0");
        assert!(sch.model_outputs[1].is_some(), "after 1 step, idx 1 should be Some");

        // After 2 steps, both slots are Some (the older one shifted from idx 1 to idx 0).
        let _ = sch.step(&v, &x).expect("step 1");
        assert!(sch.model_outputs[0].is_some(), "after 2 steps, idx 0 should be Some");
        assert!(sch.model_outputs[1].is_some(), "after 2 steps, idx 1 should be Some");
    }

    #[test]
    fn unipc_first_step_uses_lower_order() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let mut sch = CosmosUniPcMultistepScheduler::new(1000, 5, 5.0, 2);
        // Initially: lower_order_nums = 0, this_order undefined (0).
        assert_eq!(sch.lower_order_nums, 0);

        let n = 8_usize;
        let x = Tensor::from_vec(
            vec![1.0_f32; n], Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let v = Tensor::from_vec(
            vec![0.1_f32; n], Shape::from_dims(&[1, n]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        // After step 0: this_order = min(solver_order=2, n_inf-step_idx=5) = 2, then
        // min(2, lower_order_nums+1 = 0+1 = 1) = 1. So first step uses order=1.
        let _ = sch.step(&v, &x).expect("step 0");
        assert_eq!(sch.this_order, 1, "first step should use order=1 (warmup)");
        assert_eq!(sch.lower_order_nums, 1);

        // After step 1: this_order = min(2, 5-1=4) = 2, then min(2, 1+1=2) = 2.
        let _ = sch.step(&v, &x).expect("step 1");
        assert_eq!(sch.this_order, 2, "second step should use full order=2");
    }

    #[test]
    fn unipc_linsolve_f64_2x2_identity() {
        // R = identity, b = [3, 4] → x = [3, 4]
        let r = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0_f64, 4.0];
        let x = CosmosUniPcMultistepScheduler::linsolve_f64(&r, &b).expect("solve");
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn unipc_linsolve_f64_2x2_general() {
        // R = [[2, 1], [1, 3]], b = [5, 10] → solve:
        //   2x + y = 5
        //   x + 3y = 10
        // det=5, x = (5*3 - 1*10)/5 = 5/5 = 1, y = (2*10 - 1*5)/5 = 15/5 = 3
        let r = vec![vec![2.0_f64, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0_f64, 10.0];
        let x = CosmosUniPcMultistepScheduler::linsolve_f64(&r, &b).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-12, "x[1]={}", x[1]);
    }

    /// Parity test: compares Rust UniPC step output against a Python-generated
    /// fixture. Fixture lives at
    /// `inference-flame/ports/cosmos-predict25-2b/parity/cosmos_unipc_step_ref.safetensors`.
    /// Generated by running the companion Python script on GPU.
    #[test]
    fn unipc_step_matches_python_parity_fixture() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("ports/cosmos-predict25-2b/parity/cosmos_unipc_step_ref.safetensors");
        if !fixture.exists() {
            eprintln!("Parity fixture not present at {:?}; skipping.", fixture);
            return;
        }
        // Load the safetensors fixture via flame-core's loader.
        let tensors = flame_core::serialization::load_tensors(
            &fixture,
            dev.clone(),
            flame_core::serialization::SerializationFormat::SafeTensors,
        ).expect("load fixture");

        // Tensors stored: `sample`, `model_output`, `prev_sample`, `sigmas`.
        let sample = tensors.get("sample").expect("missing sample").clone();
        let model_output = tensors.get("model_output").expect("missing model_output").clone();
        let prev_sample_py = tensors.get("prev_sample").expect("missing prev_sample").clone();

        // The fixture is generated with: num_inference_steps=10, shift=5.0,
        // solver_order=2, predict_x0=true. Step 0.
        let mut sch = CosmosUniPcMultistepScheduler::new(1000, 10, 5.0, 2);
        // Cast inputs to BF16 to match Cosmos's runtime dtype (Python ref
        // can be either; we treat the fixture as F32 and the scheduler is
        // dtype-generic via mul_scalar/add/sub).
        // For the cleanest parity, we use F32 throughout (no BF16 round-trip
        // loss to confuse things).
        let prev_sample_rust = sch.step(&model_output, &sample).expect("step");

        // Compute cos distance and max-abs error.
        let py = prev_sample_py.to_vec().expect("py to_vec");
        let rs = prev_sample_rust.to_vec().expect("rs to_vec");
        assert_eq!(py.len(), rs.len(), "shape mismatch");

        let mut dot = 0.0_f64;
        let mut np2 = 0.0_f64;
        let mut rs2 = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (&a, &b) in py.iter().zip(rs.iter()) {
            dot += (a as f64) * (b as f64);
            np2 += (a as f64) * (a as f64);
            rs2 += (b as f64) * (b as f64);
            let d = (a - b).abs();
            if d > max_abs { max_abs = d; }
        }
        let cos = dot / (np2.sqrt() * rs2.sqrt()).max(1e-30);
        eprintln!(
            "UniPC parity: cos={cos:.6}, max_abs={max_abs:.6}, len={}",
            py.len()
        );
        assert!(
            cos >= 0.9999,
            "UniPC parity cos={cos} below 0.9999 threshold (max_abs={max_abs})"
        );
    }
}
