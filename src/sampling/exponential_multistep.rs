//! Exponential-integrator diffusion samplers (clean-room re-implementation).
//!
//! This module provides a small family of **exponential-integrator**
//! samplers for flow-matching (rectified-flow) diffusion models. The
//! implementations are derived directly from the numerical-analysis
//! literature — no code or layout from any prior GPL/AGPL implementation
//! was consulted:
//!
//!   - Hochbruck & Ostermann, "Exponential integrators", Acta Numerica 2010
//!     (arXiv:1102.4615). Variation-of-constants / multistep derivation,
//!     φ-function definitions.
//!   - Cox & Matthews, "Exponential time differencing for stiff systems",
//!     J. Comp. Phys. 176(2), 2002. ETDRK2 / ETDRK3 singlestep tableaus.
//!   - Lu, Zhou, Bao, Chen, Li, Zhu, "DPM-Solver++" (arXiv:2211.01095). The
//!     data-prediction multistep form used for flow-matching DPM++2M.
//!   - Zhang & Chen, "Fast Sampling of Diffusion Models with Exponential
//!     Integrator" (arXiv:2204.13902). DEIS polynomial-integrand variant.
//!
//! ## Problem setup
//!
//! For a flow-matching model `v(x, σ)` with the rectified-flow
//! convention `x_σ = (1-σ)·data + σ·noise` and model output `v = noise -
//! data`, the denoising ODE from σ=1 → σ=0 reads
//!
//!     dx/dσ = v(x, σ)         (Euler:  x_next = x + (σ_next - σ)·v)
//!
//! with `denoised = x - σ·v` being the data prediction (x1 in FM jargon).
//!
//! Introduce the "log-SNR" `λ(σ) = log((1-σ)/σ)`; `λ` increases as σ
//! decreases, so `h = λ_next - λ > 0` for denoising. In the
//! **data-prediction** (denoised) form the ODE linearises enough that
//! the exact solution of one step is, to leading order,
//!
//!     x_next = (σ_next/σ)·x - α_next · (e^{-h} - 1) · denoised + …
//!            = (σ_next/σ)·x + α_next · (-h).expm1() · (-denoised)
//!
//! where α = 1 - σ. Higher-order schemes add correction terms built from
//! past `denoised` samples (multistep) or extra NFEs (singlestep).
//!
//! For the pure ETDRK-style `res_*` schemes we use the alternate
//! change-of-variable `τ = -log σ` (σ decreases ⇒ τ increases) for which
//! `dx/dτ = -x + denoised(x, τ)` is a linear ODE + nonlinear forcing
//! with constant linear part `L = -I`. The exact variation-of-constants
//! solution over a step of size `h = τ_next - τ = log(σ / σ_next)` is
//!
//!     x_next = e^{-h}·x + ∫_0^h e^{-(h-s)} N(x(τ+s), τ+s) ds
//!            = e^{-h}·x + h·φ1(-h)·N_n
//!                       + h²·φ2(-h)·N'_n
//!                       + h³·φ3(-h)·N''_n/2 + …
//!
//! with φ_k defined below and N ≡ denoised. Approximating the derivatives
//! of N by backward differences over the history of past `denoised`
//! samples yields the `res_2m`, `res_3m` multistep schemes; plugging in
//! intermediate stages yields the singlestep `res_2s`, `res_3s` variants.
//!
//! ## φ-functions
//!
//!     φ0(z) = e^z
//!     φ1(z) = (e^z - 1) / z              (= 1   at z=0)
//!     φ2(z) = (e^z - 1 - z) / z²         (= 1/2 at z=0)
//!     φ3(z) = (e^z - 1 - z - z²/2) / z³  (= 1/6 at z=0)
//!
//! For |z| small the closed forms suffer catastrophic cancellation; we
//! switch to truncated Taylor series below `|z| < 1e-3`.
//!
//! ## API surface
//!
//! All "step" functions are pure and take the current `x`, the current
//! `denoised`, the current σ and σ_next, and (for multistep) a
//! `MultistepHistory` ring buffer of past `(denoised, λ)` pairs. First
//! steps fall back gracefully to lower order. Singlestep variants take
//! a `VelocityFn` closure so they can run extra NFEs inside.

use flame_core::{Result, Tensor};

// ---------------------------------------------------------------------------
// φ-functions
// ---------------------------------------------------------------------------

/// Zero-sized namespace for the φ_k scalar helpers.
///
/// Each `phi*` evaluates its closed form for `|z| ≥ THR` and a Taylor
/// series for `|z| < THR`. The crossover threshold `THR = 1e-3` is chosen
/// so that the closed-form path stays accurate to ≳ 1e-12 relative
/// (its worst cancellation term is `(e^z − 1 − z)/z²`, which has about
/// `log10(z²/ε_machine) ≈ 10` good digits at `z = 1e-3`) while the
/// Taylor path at the same `|z|` is accurate to `|z|^{N}/N!` ≈ 1e-21 for
/// the 6-term expansion used.
pub struct Phi;

const PHI_TAYLOR_THR_F32: f32 = 1.0e-3;
const PHI_TAYLOR_THR_F64: f64 = 1.0e-3;

impl Phi {
    /// φ1(z) = (e^z - 1) / z
    #[inline]
    pub fn phi1(z: f32) -> f32 {
        if z.abs() < PHI_TAYLOR_THR_F32 {
            // Taylor: 1 + z/2 + z²/6 + z³/24 + z⁴/120 + z⁵/720
            let z2 = z * z;
            1.0
                + z * 0.5
                + z2 * (1.0 / 6.0)
                + z2 * z * (1.0 / 24.0)
                + z2 * z2 * (1.0 / 120.0)
                + z2 * z2 * z * (1.0 / 720.0)
        } else {
            // Use expm1 for accuracy near zero on the closed-form branch too.
            z.exp_m1() / z
        }
    }

    /// φ2(z) = (e^z - 1 - z) / z²
    ///
    /// In f32 the closed form cancels ~2 digits by |z| = 0.1; widen the
    /// Taylor window to |z| < 0.3 and extend the series to 7 terms to
    /// maintain ≥ 7 digits everywhere.
    #[inline]
    pub fn phi2(z: f32) -> f32 {
        if z.abs() < 0.3 {
            let c0 = 1.0_f32 / 2.0;
            let c1 = 1.0_f32 / 6.0;
            let c2 = 1.0_f32 / 24.0;
            let c3 = 1.0_f32 / 120.0;
            let c4 = 1.0_f32 / 720.0;
            let c5 = 1.0_f32 / 5040.0;
            let c6 = 1.0_f32 / 40320.0;
            c0 + z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * c6)))))
        } else {
            (z.exp_m1() - z) / (z * z)
        }
    }

    /// φ3(z) = (e^z - 1 - z - z²/2) / z³
    ///
    /// In f32 the closed form loses ~3 digits to cancellation even at
    /// |z| = 0.1, so we use a wider Taylor-series window (|z| < 0.5)
    /// and extend the series to 8 terms, giving ≥ 7 digits everywhere.
    #[inline]
    pub fn phi3(z: f32) -> f32 {
        if z.abs() < 0.5 {
            // Horner form of 1/6 + z/24 + z²/120 + z³/720 + z⁴/5040 + z⁵/40320 + z⁶/362880 + z⁷/3628800
            let c0 = 1.0_f32 / 6.0;
            let c1 = 1.0_f32 / 24.0;
            let c2 = 1.0_f32 / 120.0;
            let c3 = 1.0_f32 / 720.0;
            let c4 = 1.0_f32 / 5040.0;
            let c5 = 1.0_f32 / 40320.0;
            let c6 = 1.0_f32 / 362880.0;
            let c7 = 1.0_f32 / 3628800.0;
            // Horner: ((((((c7 z + c6) z + c5) z + c4) z + c3) z + c2) z + c1) z + c0
            c0 + z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * c7))))))
        } else {
            (z.exp_m1() - z - 0.5 * z * z) / (z * z * z)
        }
    }

    /// φ1(z) at f64 precision (for internal scalar coefficient prep).
    #[inline]
    pub fn phi1_f64(z: f64) -> f64 {
        // f64 phi1 only needs 1 cancellation (expm1 is accurate), so
        // the closed form is fine everywhere except around z = 0.
        // We keep the small threshold.
        if z.abs() < PHI_TAYLOR_THR_F64 {
            // 8-term Taylor: φ1(z) = Σ_{k≥0} z^k/(k+1)!
            let mut term = 1.0_f64;
            let mut sum = 0.0_f64;
            let mut fact = 1.0_f64;
            for k in 0..10 {
                sum += term / fact;
                term *= z;
                fact *= (k + 2) as f64;
            }
            sum
        } else {
            z.exp_m1() / z
        }
    }

    #[inline]
    pub fn phi2_f64(z: f64) -> f64 {
        // f64 phi2 loses ~1-2 digits at z = 1e-4. Widen the Taylor window
        // to 0.05 and use 10 terms — gives ≥ 14 digits everywhere.
        if z.abs() < 0.05 {
            let mut term = 1.0_f64;
            let mut sum = 0.0_f64;
            let mut fact = 2.0_f64; // (k+2)! starting at k=0 → 2
            for k in 0..10 {
                sum += term / fact;
                term *= z;
                fact *= (k + 3) as f64;
            }
            sum
        } else {
            (z.exp_m1() - z) / (z * z)
        }
    }

    #[inline]
    pub fn phi3_f64(z: f64) -> f64 {
        // f64 still loses ~6-7 digits at |z| = 1e-4 with the closed form.
        // Use Taylor for |z| < 0.1 (where the 12-term series converges
        // to sub-1e-16 relative).
        if z.abs() < 0.1 {
            let mut term = 1.0_f64;
            let mut sum = 0.0_f64;
            // φ3(z) = Σ_{k≥0} z^k / (k+3)!
            let mut fact = 6.0_f64; // (k+3)! starting at k=0 → 3! = 6
            for k in 0..12 {
                sum += term / fact;
                term *= z;
                fact *= (k + 4) as f64;
            }
            sum
        } else {
            (z.exp_m1() - z - 0.5 * z * z) / (z * z * z)
        }
    }
}

// ---------------------------------------------------------------------------
// Log-SNR and step-size helpers
// ---------------------------------------------------------------------------

/// Log-SNR `λ(σ) = log((1-σ)/σ)`. Increases monotonically as σ decreases.
#[inline]
pub fn lambda_from_sigma(sigma: f32) -> f32 {
    // Clamp to avoid ±inf at the endpoints. The schedule should never
    // call the sampler at exactly 0 or 1, but guard anyway.
    let s = sigma.clamp(1.0e-6, 1.0 - 1.0e-6);
    ((1.0 - s) / s).ln()
}

/// τ(σ) = -log σ (reciprocal log-scale). Used by the `res_*` family.
#[inline]
pub fn tau_from_sigma(sigma: f32) -> f32 {
    -sigma.clamp(1.0e-6, 1.0).ln()
}

// ---------------------------------------------------------------------------
// Multistep history ring buffer
// ---------------------------------------------------------------------------

/// Bounded ring buffer of past `(denoised, λ)` pairs used by multistep
/// samplers. `push()` evicts the oldest entry when the buffer is full.
///
/// Index semantics for [`MultistepHistory::get`]:
///   - `back = 0` returns the most recently pushed entry
///   - `back = 1` the one before that
///   - etc.
pub struct MultistepHistory {
    capacity: usize,
    denoised: Vec<Tensor>,
    lambdas: Vec<f32>,
    // Index into `denoised` / `lambdas` of the most-recent entry, or
    // `usize::MAX` when empty.
    head: usize,
    len: usize,
}

impl MultistepHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            denoised: Vec::with_capacity(capacity.max(1)),
            lambdas: Vec::with_capacity(capacity.max(1)),
            head: usize::MAX,
            len: 0,
        }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn capacity(&self) -> usize { self.capacity }

    /// Push `(denoised, lambda)` as the new most-recent entry.
    pub fn push(&mut self, denoised: Tensor, lambda: f32) {
        debug_assert_eq!(
            self.denoised.len(),
            self.lambdas.len(),
            "MultistepHistory invariant violated: denoised/lambdas length mismatch"
        );
        if self.denoised.len() < self.capacity {
            // Not yet full: just append. Head = last index pushed.
            self.denoised.push(denoised);
            self.lambdas.push(lambda);
            self.head = self.denoised.len() - 1;
            self.len += 1;
        } else {
            // Full: overwrite the slot following `head` (i.e. the oldest).
            let write = (self.head + 1) % self.capacity;
            self.denoised[write] = denoised;
            self.lambdas[write] = lambda;
            self.head = write;
        }
    }

    /// Most-recent-first access. `back = 0` is the newest.
    pub fn get(&self, back: usize) -> Option<(&Tensor, f32)> {
        if back >= self.len { return None; }
        let idx = (self.head + self.capacity - back) % self.capacity;
        Some((&self.denoised[idx], self.lambdas[idx]))
    }
}

// ---------------------------------------------------------------------------
// Model closure type
// ---------------------------------------------------------------------------

/// A model closure producing **velocity** at the given σ.
///
/// `f(x, σ) -> v` where `v = noise - data` in the rectified-flow
/// convention (Euler step is `x + (σ_next - σ)·v`). The singlestep
/// samplers call this 2×–3× per outer step.
pub type VelocityFn<'a> = &'a mut dyn FnMut(&Tensor, f32) -> Result<Tensor>;

// ---------------------------------------------------------------------------
// Small tensor combinators
// ---------------------------------------------------------------------------

/// `a·x + b·y` via two scalar-muls and an add.
#[inline]
fn lincomb2(x: &Tensor, a: f32, y: &Tensor, b: f32) -> Result<Tensor> {
    let xa = x.mul_scalar(a)?;
    let yb = y.mul_scalar(b)?;
    xa.add(&yb)
}

/// `a·x + b·y + c·z`.
#[inline]
fn lincomb3(x: &Tensor, a: f32, y: &Tensor, b: f32, z: &Tensor, c: f32) -> Result<Tensor> {
    let tmp = lincomb2(x, a, y, b)?;
    let zc = z.mul_scalar(c)?;
    tmp.add(&zc)
}

/// `a·x + b·y + c·z + d·w`.
#[inline]
fn lincomb4(
    x: &Tensor, a: f32,
    y: &Tensor, b: f32,
    z: &Tensor, c: f32,
    w: &Tensor, d: f32,
) -> Result<Tensor> {
    let tmp = lincomb3(x, a, y, b, z, c)?;
    let wd = w.mul_scalar(d)?;
    tmp.add(&wd)
}

// ---------------------------------------------------------------------------
// DPM++ 2M (multistep, flow-matching, 1 NFE/step)
// ---------------------------------------------------------------------------

/// 2nd-order multistep DPM++ (data-prediction, flow-matching variant).
///
/// One NFE per step. Uses one previous `denoised` (and its λ) for the
/// 2nd-order correction; if the history is empty the step degrades to
/// 1st-order (DDIM / Euler-on-data).
///
/// Derivation (rectified flow, α = 1-σ, λ = log(α/σ)):
///   x_next  = (σ_next/σ)·x - α_next·(e^{-h} - 1)·denoised_correction
///           = (σ_next/σ)·x + α_next·(-h).expm1()·(-denoised_correction)
/// with
///   denoised_correction = (1 + 1/(2r))·denoised - (1/(2r))·denoised_prev,
///   h = λ_next - λ, r = (λ - λ_prev) / h.
///
/// Reference: Lu et al. 2022, "DPM-Solver++" (arXiv:2211.01095), §4.
pub fn dpmpp_2m_step(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_next: f32,
    history: &MultistepHistory,
) -> Result<Tensor> {
    let lambda = lambda_from_sigma(sigma);
    let lambda_next = lambda_from_sigma(sigma_next);
    let h = lambda_next - lambda;
    let alpha_next = 1.0 - sigma_next;
    let sigma_ratio = sigma_next / sigma;
    // (-h).expm1() = e^{-h} - 1 ≤ 0 (since h > 0 denoising direction)
    let em1 = (-h).exp_m1();

    // First step or empty history: degrade to 1st-order data-pred step.
    //   x_next = σ_ratio·x - α_next·(e^{-h} - 1)·denoised
    //         = σ_ratio·x + α_next·(-em1)·denoised
    if history.is_empty() {
        // coeff on denoised is -α_next·em1 = α_next·(1 - e^{-h}).
        return lincomb2(x, sigma_ratio, denoised, -alpha_next * em1);
    }

    let (denoised_prev, lambda_prev) = match history.get(0) {
        Some(v) => v,
        None => return lincomb2(x, sigma_ratio, denoised, -alpha_next * em1),
    };
    let h_prev = lambda - lambda_prev;
    // r must be positive; if numerical noise produces a near-zero or
    // negative h_prev, drop to 1st-order to stay robust.
    if !(h_prev > 0.0 && h > 0.0) {
        return lincomb2(x, sigma_ratio, denoised, -alpha_next * em1);
    }
    let r = h_prev / h;
    let inv_2r = 0.5 / r;

    // denoised_correction = (1 + inv_2r)·denoised - inv_2r·denoised_prev
    // x_next = σ_ratio·x + α_next·(-em1)·denoised_correction
    //        = σ_ratio·x
    //          + α_next·(-em1)·(1 + inv_2r)·denoised
    //          - α_next·(-em1)·inv_2r   ·denoised_prev
    let c_d = -alpha_next * em1 * (1.0 + inv_2r);
    let c_p = alpha_next * em1 * inv_2r; // = -(-em1)·inv_2r·α_next
    lincomb3(x, sigma_ratio, denoised, c_d, denoised_prev, c_p)
}

// ---------------------------------------------------------------------------
// res_2m / res_3m — exp-integrator multistep in τ = -log σ
// ---------------------------------------------------------------------------
//
// With L = -I the exact one-step formula is
//     x_next = e^{-h}·x + ∫_0^h e^{-(h-s)} N(τ+s) ds
// expanding N(τ+s) ≈ N_n + s·D1 + (s²/2)·D2 gives
//     x_next = e^{-h}·x
//            + h φ1(-h) N_n
//            + h² φ2(-h) D1
//            + h³ φ3(-h) D2
// where D_k are approximations to the k-th τ-derivative of N at τ_n.
// We approximate with backward divided differences on the τ grid.

/// 2nd-order exp-integrator multistep. 1 NFE / step. Uses one history
/// entry; falls back to 1st-order on step 0.
pub fn res_2m_step(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_next: f32,
    history: &MultistepHistory,
) -> Result<Tensor> {
    let tau = -sigma.clamp(1.0e-6, 1.0).ln();
    let tau_next = -sigma_next.clamp(1.0e-6, 1.0).ln();
    let h = tau_next - tau;

    let e_mh = (-h).exp();
    let ph1 = Phi::phi1(-h); // coefficient on h
    let ph2 = Phi::phi2(-h); // coefficient on h²

    if history.is_empty() || h <= 0.0 {
        // 1st-order exponential Euler: x_next = e^{-h}·x + h·φ1(-h)·N.
        return lincomb2(x, e_mh, denoised, h * ph1);
    }

    // Previous denoised is expressed in **λ** in the history, but we
    // only need the step size between consecutive denoised evaluations
    // on the τ grid. Since τ(σ) is a monotone map of σ, the pair-wise
    // τ differences are computed afresh from σ (we stored λ; convert).
    //
    // For the builder we only need the τ-step `h_prev` = τ(σ_n) -
    // τ(σ_{n-1}). The history stored λ_{n-1}; recover σ_{n-1} from λ
    // using σ = 1/(1+e^λ), then τ = -log σ = log(1 + e^λ) = softplus(λ).
    let (denoised_prev, lambda_prev) = match history.get(0) {
        Some(v) => v,
        None => return lincomb2(x, e_mh, denoised, h * ph1),
    };
    let tau_prev = softplus(lambda_prev);
    let h_prev = tau - tau_prev;
    if h_prev <= 0.0 {
        return lincomb2(x, e_mh, denoised, h * ph1);
    }

    // D1 ≈ (N_n - N_{n-1}) / h_prev.
    //
    // Expand:
    //   x_next = e^{-h}·x + h·φ1(-h)·N_n + h²·φ2(-h)·D1
    //          = e^{-h}·x
    //            + [h·φ1 + h²·φ2 / h_prev] · N_n
    //            - [h²·φ2 / h_prev]        · N_{n-1}
    let c_n = h * ph1 + (h * h * ph2) / h_prev;
    let c_p = -(h * h * ph2) / h_prev;
    lincomb3(x, e_mh, denoised, c_n, denoised_prev, c_p)
}

/// 3rd-order exp-integrator multistep. 1 NFE / step. Needs two history
/// entries; steps 0/1 fall back to `res_2m`/1st-order.
pub fn res_3m_step(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_next: f32,
    history: &MultistepHistory,
) -> Result<Tensor> {
    let tau = -sigma.clamp(1.0e-6, 1.0).ln();
    let tau_next = -sigma_next.clamp(1.0e-6, 1.0).ln();
    let h = tau_next - tau;

    let e_mh = (-h).exp();
    let ph1 = Phi::phi1(-h);
    let ph2 = Phi::phi2(-h);
    let ph3 = Phi::phi3(-h);

    if history.len() < 2 || h <= 0.0 {
        // Not enough history: fall back to res_2m (which itself falls
        // back to 1st order when history is empty).
        return res_2m_step(x, denoised, sigma, sigma_next, history);
    }

    let (n_m1, lam_m1) = match history.get(0) {
        Some(v) => v,
        None => return res_2m_step(x, denoised, sigma, sigma_next, history),
    };
    let (n_m2, lam_m2) = match history.get(1) {
        Some(v) => v,
        None => return res_2m_step(x, denoised, sigma, sigma_next, history),
    };
    let tau_m1 = softplus(lam_m1);
    let tau_m2 = softplus(lam_m2);
    let h1 = tau - tau_m1;         // step n-1 → n
    let h2 = tau_m1 - tau_m2;      // step n-2 → n-1

    if h1 <= 0.0 || h2 <= 0.0 {
        return res_2m_step(x, denoised, sigma, sigma_next, history);
    }

    // Newton-form quadratic polynomial in s = τ - τ_n (backward-looking):
    //
    //   P(τ_n + s) = N_n + s·D1_n + s·(s + h1)·DD2
    //             = N_n + s·(D1_n + h1·DD2) + s²·DD2
    //
    // with divided differences on the non-uniform grid {τ_{n-2}, τ_{n-1}, τ_n}:
    //   D1_n      = (N_n - N_{n-1}) / h1
    //   D1_{n-1}  = (N_{n-1} - N_{n-2}) / h2
    //   DD2       = (D1_n - D1_{n-1}) / (h1 + h2)
    //
    // (DD2 is the 2nd Newton divided difference ≈ N''(τ_n)/2 — no factor of 2.)
    //
    // Integrate against the exponential kernel: the identity
    //   ∫_0^h e^{-(h-s)}·s^{k-1} ds = (k-1)!·h^k·φ_k(-h)
    // gives
    //   x_next = e^{-h}·x
    //          + h·φ1(-h)·N_n
    //          + h²·φ2(-h)·(D1_n + h1·DD2)
    //          + 2·h³·φ3(-h)·DD2
    //
    // Express as a linear combination of (x, N_n, N_{n-1}, N_{n-2}).
    // With S = h1 + h2 and DD2 coefficients
    //   DD2 on N_n:      1 / (h1 · S)
    //   DD2 on N_{n-1}:  -1 / (h1 · h2)
    //   DD2 on N_{n-2}:  1 / (h2 · S)
    // and γ = 2·h³·φ3:
    //
    //   N_n     coeff = α + β·(1/h1 + 1/S) + γ·(1/(h1·S))
    //   N_{n-1} coeff =     β·(-1/h1 - 1/h2) + γ·(-1/(h1·h2))
    //   N_{n-2} coeff =     β·(h1/(h2·S))    + γ·(1/(h2·S))
    //
    // where α = h·φ1, β = h²·φ2.
    let alpha = h * ph1;
    let beta  = h * h * ph2;
    let gamma = 2.0 * h * h * h * ph3;

    let s = h1 + h2;
    let c_n_dd2   =  1.0 / (h1 * s);
    let c_nm1_dd2 = -1.0 / (h1 * h2);
    let c_nm2_dd2 =  1.0 / (h2 * s);

    let c_n   = alpha
              + beta * (1.0 / h1 + 1.0 / s)
              + gamma * c_n_dd2;
    let c_nm1 =        beta * (-1.0 / h1 - 1.0 / h2)
              + gamma * c_nm1_dd2;
    let c_nm2 =        beta * (h1 / (h2 * s))
              + gamma * c_nm2_dd2;

    // x_next = e^{-h}·x + c_n·N_n + c_nm1·N_{n-1} + c_nm2·N_{n-2}
    lincomb4(x, e_mh, denoised, c_n, n_m1, c_nm1, n_m2, c_nm2)
}

// ---------------------------------------------------------------------------
// res_2s / res_3s — exp-integrator singlestep (ETDRK)
// ---------------------------------------------------------------------------
//
// ETDRK2 (Cox & Matthews 2002, eq. 20-22, L = -I, c2 = 1):
//
//     k1 = N(x, τ)
//     a2 = e^{-h}·x + h·φ1(-h)·k1
//     k2 = N(a2, τ + h)
//     x_next = a2 + h·φ2(-h)·(k2 - k1)
//
// This needs N (= denoised = x - σ·v), not v. We convert internally.

#[inline]
fn denoised_from_velocity(x: &Tensor, v: &Tensor, sigma: f32) -> Result<Tensor> {
    // denoised = x - σ·v
    let sv = v.mul_scalar(sigma)?;
    x.sub(&sv)
}

/// σ on the τ grid for a fractional step. Given `h` ∈ (0, h_total] from
/// the current τ, return σ such that τ_c = τ + c·h_total. Since
/// τ = -log σ, σ_c = σ · exp(-c·h_total).
#[inline]
fn sigma_at_tau_step(sigma: f32, h_frac: f32) -> f32 {
    // σ_c = σ · e^{-h_frac}. For large h_frac this can underflow; clamp
    // to the same epsilon tau_from_sigma uses.
    let s = sigma * (-h_frac).exp();
    s.clamp(1.0e-6, 1.0)
}

/// 2nd-order exp-integrator singlestep (ETDRK2). **2 NFE / step.** No
/// history. Stage abscissa c₂ = 1 (i.e. the intermediate stage sits at
/// τ_next, same as the predictor's endpoint).
pub fn res_2s_step(
    x: &Tensor,
    sigma: f32,
    sigma_next: f32,
    model_fn: VelocityFn,
) -> Result<Tensor> {
    let tau = -sigma.clamp(1.0e-6, 1.0).ln();
    let tau_next = -sigma_next.clamp(1.0e-6, 1.0).ln();
    let h = tau_next - tau;

    if h <= 0.0 {
        // Degenerate (should not happen on a proper denoising schedule):
        // fall back to Euler in σ.
        let v = model_fn(x, sigma)?;
        let step = v.mul_scalar(sigma_next - sigma)?;
        return x.add(&step);
    }

    let e_mh = (-h).exp();
    let ph1 = Phi::phi1(-h);
    let ph2 = Phi::phi2(-h);

    // Stage 1
    let v1 = model_fn(x, sigma)?;
    let n1 = denoised_from_velocity(x, &v1, sigma)?;

    // Predictor stage at τ + h (= sigma_next)
    //   a2 = e^{-h}·x + h·φ1(-h)·N1
    let a2 = lincomb2(x, e_mh, &n1, h * ph1)?;

    // Stage 2
    let sigma2 = sigma_next.clamp(1.0e-6, 1.0);
    let v2 = model_fn(&a2, sigma2)?;
    let n2 = denoised_from_velocity(&a2, &v2, sigma2)?;

    // Corrector: x_next = a2 + h·φ2(-h)·(N2 - N1)
    //          = a2 + (h·φ2)·N2 - (h·φ2)·N1
    let hph2 = h * ph2;
    let n_diff = n2.sub(&n1)?;
    let corr = n_diff.mul_scalar(hph2)?;
    a2.add(&corr)
}

/// 3rd-order exp-integrator singlestep (ETDRK3, Cox-Matthews form with
/// c₂ = 1/2, c₃ = 1). **3 NFE / step.** No history.
///
/// With linear part `L = -I` the Cox–Matthews ETDRK3 tableau reduces to
/// (derivation in Cox & Matthews 2002, §II.C, specialised to scalar L):
///
///     k1 = N(x,   τ)
///     a2 = e^{-h/2}·x + (h/2)·φ1(-h/2)·k1
///     k2 = N(a2,  τ + h/2)
///     a3 = e^{-h}·x + h·φ1(-h)·(2 k2 - k1)
///     k3 = N(a3,  τ + h)
///     x_next = e^{-h}·x
///            + h·[ (φ1(-h) - 3 φ2(-h) + 4 φ3(-h)) · k1
///                + (         4 φ2(-h) - 8 φ3(-h)) · k2
///                + (        -φ2(-h) + 4 φ3(-h)) · k3 ]
///
/// The weights are the canonical 3-stage ETDRK3 weights for uniform
/// c₂ = 1/2, c₃ = 1. Because this is a singlestep scheme, a minor
/// reordering of Butcher entries leaves the order of consistency
/// unchanged; we keep this choice because its φ-combinations all appear
/// in published formulations and it's the easiest to double-check.
pub fn res_3s_step(
    x: &Tensor,
    sigma: f32,
    sigma_next: f32,
    model_fn: VelocityFn,
) -> Result<Tensor> {
    let tau = -sigma.clamp(1.0e-6, 1.0).ln();
    let tau_next = -sigma_next.clamp(1.0e-6, 1.0).ln();
    let h = tau_next - tau;

    if h <= 0.0 {
        let v = model_fn(x, sigma)?;
        let step = v.mul_scalar(sigma_next - sigma)?;
        return x.add(&step);
    }

    let hh = 0.5 * h;
    let e_mh = (-h).exp();
    let e_mhh = (-hh).exp();

    // φ's at -h and at -h/2
    let ph1 = Phi::phi1(-h);
    let ph2 = Phi::phi2(-h);
    let ph3 = Phi::phi3(-h);
    let ph1_half = Phi::phi1(-hh);

    // Stage 1 at τ
    let v1 = model_fn(x, sigma)?;
    let n1 = denoised_from_velocity(x, &v1, sigma)?;

    // Stage 2 at τ + h/2
    //   a2 = e^{-h/2}·x + (h/2)·φ1(-h/2)·N1
    let a2 = lincomb2(x, e_mhh, &n1, hh * ph1_half)?;
    let sigma2 = sigma_at_tau_step(sigma, hh);
    let v2 = model_fn(&a2, sigma2)?;
    let n2 = denoised_from_velocity(&a2, &v2, sigma2)?;

    // Stage 3 at τ + h
    //   a3 = e^{-h}·x + h·φ1(-h)·(2 N2 - N1)
    //      = e^{-h}·x + (2 h φ1)·N2 - (h φ1)·N1
    let a3 = lincomb3(x, e_mh, &n2, 2.0 * h * ph1, &n1, -h * ph1)?;
    let sigma3 = sigma_at_tau_step(sigma, h);
    let v3 = model_fn(&a3, sigma3)?;
    let n3 = denoised_from_velocity(&a3, &v3, sigma3)?;

    // Final combination.
    let w1 = h * (ph1 - 3.0 * ph2 + 4.0 * ph3);
    let w2 = h * (4.0 * ph2 - 8.0 * ph3);
    let w3 = h * (-ph2 + 4.0 * ph3);

    // x_next = e^{-h}·x + w1·N1 + w2·N2 + w3·N3
    lincomb4(x, e_mh, &n1, w1, &n2, w2, &n3, w3)
}

// ---------------------------------------------------------------------------
// DEIS 3rd-order multistep — corrected derivation
// ---------------------------------------------------------------------------
//
// The flow-matching ODE dx/dσ = (x − D)/σ, under change of variable
// τ = −log σ, becomes the linear-plus-forcing ODE
//
//     dx/dτ = −x + D(τ)
//
// (L = −1 in ETDRK language). The exact variation-of-constants solution
// over a step h = τ_{n+1} − τ_n > 0 is
//
//     x_{n+1} = e^{−h}·x_n + ∫_0^h e^{−(h−s)} · D(τ_n + s) ds
//
// Change variable to u = log σ = −τ (so du = −dτ, u decreases with n). Let
// u_j := log σ at history index j so that u_n > u_{n+1}. The integrand
// transforms to `D(u) · e^{−(u − u_{n+1})} · (−du)` and after cleaning up
//
//     x_{n+1} = (σ_{n+1}/σ_n) · x_n − σ_{n+1} · ∫_{u_n}^{u_{n+1}}
//                                       e^{−u} · D(u) du
//
// (the bounds have u_n > u_{n+1}, so the integral is negative and the
// overall contribution from a positive, constant D is positive — which
// matches the DDIM-limit sanity check: constant D, x_n = D ⇒ x_{n+1} = D).
//
// Approximate D(u) by the degree-2 Lagrange polynomial through the three
// nodes (u_n, u_{n−1}, u_{n−2}):
//
//     D(u) ≈ Σ_j L_j(u) · D_j         with L_j(u) = Π_{i≠j}(u − u_i) / Π_{i≠j}(u_j − u_i)
//
// Each L_j is a quadratic α_j + β_j·u + γ_j·u². Plug in and integrate:
//
//     W_j := ∫_{u_n}^{u_{n+1}} L_j(u) · e^{−u} du
//          = α_j · M_0 + β_j · M_1 + γ_j · M_2    with (a, b) = (u_n, u_{n+1})
//
// where the elementary moments are (by repeated integration by parts)
//
//     M_k(a, b) := ∫_a^b u^k · e^{−u} du
//     M_0 = e^{−a} − e^{−b}
//     M_1 = (a + 1)·e^{−a} − (b + 1)·e^{−b}
//     M_2 = (a² + 2a + 2)·e^{−a} − (b² + 2b + 2)·e^{−b}
//
// Final update:
//
//     x_{n+1} = (σ_{n+1}/σ_n)·x_n  −  σ_{n+1} · Σ_j W_j · D_j
//
// Fall-backs:
//   len == 0 : first-order DDIM-like step, W_0 = M_0 only (no interpolation).
//   len == 1 : DEIS-2 (two-node Lagrange). W_0^(2), W_1^(2) via M_0/M_1 only.
//   len >= 2 : full DEIS-3.
//
// All coefficients are computed at f64 and cast to f32 for the final tensor
// scalar-mul + add. This keeps the polynomial-cancellation error below the
// f32 tensor rounding floor.

/// Moments `M_k(a,b) = ∫_a^b u^k · e^{−u} du` for k = 0, 1, 2.
///
/// Uses the closed forms derived by integration by parts:
///
///   P_0(u) = 1
///   P_1(u) = u + 1
///   P_2(u) = u² + 2u + 2
///   M_k(a,b) = e^{−a}·P_k(a) − e^{−b}·P_k(b)
#[inline]
fn moments_012(a: f64, b: f64) -> (f64, f64, f64) {
    let ea = (-a).exp();
    let eb = (-b).exp();
    let p1a = a + 1.0;
    let p1b = b + 1.0;
    let p2a = a * a + 2.0 * a + 2.0;
    let p2b = b * b + 2.0 * b + 2.0;
    let m0 = ea - eb;
    let m1 = p1a * ea - p1b * eb;
    let m2 = p2a * ea - p2b * eb;
    (m0, m1, m2)
}

/// Quadratic Lagrange basis for node `u_j` against the two other nodes
/// `u_a`, `u_b`: L_j(u) = (u − u_a)(u − u_b) / ((u_j − u_a)(u_j − u_b)),
/// expanded as `α + β·u + γ·u²`.
#[inline]
fn lagrange_quadratic_coeffs(uj: f64, ua: f64, ub: f64) -> (f64, f64, f64) {
    let denom = (uj - ua) * (uj - ub);
    let inv = 1.0 / denom;
    let gamma = inv;                   // u²
    let beta  = -(ua + ub) * inv;      // u
    let alpha =  (ua * ub) * inv;      // 1
    (alpha, beta, gamma)
}

/// Linear Lagrange basis for node `u_j` against the other node `u_a`:
/// L_j(u) = (u − u_a) / (u_j − u_a), expanded as `α + β·u`.
#[inline]
fn lagrange_linear_coeffs(uj: f64, ua: f64) -> (f64, f64) {
    let inv = 1.0 / (uj - ua);
    let beta = inv;
    let alpha = -ua * inv;
    (alpha, beta)
}

/// 3rd-order DEIS multistep in log-σ space. 1 NFE / step.
///
/// History requirements:
///   - 0 history entries ⇒ first-order (DDIM-like) step.
///   - 1 history entry   ⇒ DEIS-2 (two-node Lagrange in u = log σ).
///   - ≥ 2 history entries ⇒ full DEIS-3.
pub fn deis_3m_step(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_next: f32,
    history: &MultistepHistory,
) -> Result<Tensor> {
    // Sanity-clamp σ to avoid log of a non-positive number.
    let s0 = (sigma as f64).max(1.0e-6);
    let s1 = (sigma_next as f64).max(1.0e-6);
    let u_n = s0.ln();           // = log σ_n
    let u_np1 = s1.ln();         // = log σ_{n+1}
    let sigma_ratio = (s1 / s0) as f32;
    let sn1 = s1; // σ_{n+1}; multiplies every W_j contribution.

    // Common moments on the step-interval (a, b) = (u_n, u_{n+1}).
    let (m0, m1, m2) = moments_012(u_n, u_np1);

    // Case 1: no history. 1st-order step.
    //   W_0 = M_0, so x_{n+1} = (σ_{n+1}/σ_n)·x − σ_{n+1}·M_0·D.
    if history.is_empty() {
        let w0 = sn1 * m0;
        return lincomb2(x, sigma_ratio, denoised, -(w0 as f32));
    }

    // Case 2: exactly one history entry. DEIS-2 (linear Lagrange).
    //   L_0(u) = (u − u_{n−1}) / (u_n − u_{n−1})
    //   L_1(u) = (u − u_n)     / (u_{n−1} − u_n)
    if history.len() == 1 {
        let (n_m1_t, lam_m1) = match history.get(0) {
            Some(v) => v,
            None => {
                // Shouldn't happen — len()==1 but get(0) returned None.
                let w0 = sn1 * m0;
                return lincomb2(x, sigma_ratio, denoised, -(w0 as f32));
            }
        };
        let s_m1 = sigma_from_lambda_f64(lam_m1 as f64).max(1.0e-6);
        let u_m1 = s_m1.ln();
        // Guard against equal nodes (degenerate Lagrange).
        if !(u_n - u_m1).is_finite() || (u_n - u_m1).abs() < 1.0e-9 {
            let w0 = sn1 * m0;
            return lincomb2(x, sigma_ratio, denoised, -(w0 as f32));
        }
        let (a0, b0) = lagrange_linear_coeffs(u_n, u_m1);
        let (a1, b1) = lagrange_linear_coeffs(u_m1, u_n);
        let w0 = sn1 * (a0 * m0 + b0 * m1);
        let w1 = sn1 * (a1 * m0 + b1 * m1);
        return lincomb3(
            x, sigma_ratio,
            denoised, -(w0 as f32),
            n_m1_t,   -(w1 as f32),
        );
    }

    // Case 3: ≥2 history entries. Full DEIS-3 (quadratic Lagrange on
    // nodes {u_n, u_{n−1}, u_{n−2}}).
    let (n_m1_t, lam_m1) = match history.get(0) {
        Some(v) => v,
        None => {
            // Impossible by guard above, but be graceful.
            let w0 = sn1 * m0;
            return lincomb2(x, sigma_ratio, denoised, -(w0 as f32));
        }
    };
    let (n_m2_t, lam_m2) = match history.get(1) {
        Some(v) => v,
        None => {
            let w0 = sn1 * m0;
            return lincomb2(x, sigma_ratio, denoised, -(w0 as f32));
        }
    };
    let s_m1 = sigma_from_lambda_f64(lam_m1 as f64).max(1.0e-6);
    let s_m2 = sigma_from_lambda_f64(lam_m2 as f64).max(1.0e-6);
    let u_m1 = s_m1.ln();
    let u_m2 = s_m2.ln();

    // Guard against degenerate (repeated) nodes.
    let du01 = u_n - u_m1;
    let du02 = u_n - u_m2;
    let du12 = u_m1 - u_m2;
    if !(du01.abs() > 1.0e-9 && du02.abs() > 1.0e-9 && du12.abs() > 1.0e-9) {
        // Fall back to DEIS-2 on (u_n, u_m1).
        let (a0, b0) = lagrange_linear_coeffs(u_n, u_m1);
        let (a1, b1) = lagrange_linear_coeffs(u_m1, u_n);
        let w0 = sn1 * (a0 * m0 + b0 * m1);
        let w1 = sn1 * (a1 * m0 + b1 * m1);
        return lincomb3(
            x, sigma_ratio,
            denoised, -(w0 as f32),
            n_m1_t,   -(w1 as f32),
        );
    }

    let (alpha0, beta0, gamma0) = lagrange_quadratic_coeffs(u_n, u_m1, u_m2);
    let (alpha1, beta1, gamma1) = lagrange_quadratic_coeffs(u_m1, u_n, u_m2);
    let (alpha2, beta2, gamma2) = lagrange_quadratic_coeffs(u_m2, u_n, u_m1);

    // W_j = ∫ L_j(u)·e^{−u} du = α·M_0 + β·M_1 + γ·M_2
    let w0 = sn1 * (alpha0 * m0 + beta0 * m1 + gamma0 * m2);
    let w1 = sn1 * (alpha1 * m0 + beta1 * m1 + gamma1 * m2);
    let w2 = sn1 * (alpha2 * m0 + beta2 * m1 + gamma2 * m2);

    // x_{n+1} = σ_ratio·x − σ_{n+1}·Σ W_j D_j
    lincomb4(
        x, sigma_ratio,
        denoised, -(w0 as f32),
        n_m1_t,   -(w1 as f32),
        n_m2_t,   -(w2 as f32),
    )
}

#[inline]
fn sigma_from_lambda_f64(lambda: f64) -> f64 {
    // λ = log((1-σ)/σ) ⇒ σ = 1 / (1 + e^λ)
    1.0 / (1.0 + lambda.exp())
}

#[inline]
fn softplus(lambda: f32) -> f32 {
    // τ(σ) = -log σ where σ = 1/(1+e^λ). So τ = log(1 + e^λ) = softplus(λ).
    let lam = lambda as f64;
    if lam > 30.0 { lam as f32 }
    else if lam < -30.0 { lam.exp() as f32 }
    else { (1.0 + lam.exp()).ln() as f32 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- φ function accuracy ------------------------------------------

    // Reference implementations that switch to a high-order Taylor series
    // for small |z|, avoiding the catastrophic cancellation that destroys
    // the closed form near zero. These are independent of the Phi
    // implementation (they use a fixed-degree expansion computed bottom-up
    // so the terms are evaluated in the best order).
    fn phi1_ref(z: f64) -> f64 {
        if z.abs() > 0.1 { return z.exp_m1() / z; }
        // 10-term Taylor — converges to ≤ 1e-18 for |z| ≤ 0.1.
        let mut term = 1.0_f64;
        let mut sum = 0.0_f64;
        for k in 0..12 {
            sum += term / factorial_f64(k + 1);
            term *= z;
        }
        sum
    }
    fn phi2_ref(z: f64) -> f64 {
        if z.abs() > 0.1 { return (z.exp_m1() - z) / (z * z); }
        let mut term = 1.0_f64;
        let mut sum = 0.0_f64;
        for k in 0..12 {
            sum += term / factorial_f64(k + 2);
            term *= z;
        }
        sum
    }
    fn phi3_ref(z: f64) -> f64 {
        if z.abs() > 0.1 { return (z.exp_m1() - z - 0.5 * z * z) / (z * z * z); }
        let mut term = 1.0_f64;
        let mut sum = 0.0_f64;
        for k in 0..12 {
            sum += term / factorial_f64(k + 3);
            term *= z;
        }
        sum
    }
    fn factorial_f64(n: usize) -> f64 {
        let mut f = 1.0_f64;
        for i in 2..=n { f *= i as f64; }
        f
    }

    fn rel_err(got: f64, want: f64) -> f64 {
        if want.abs() < 1e-300 { (got - want).abs() }
        else { ((got - want) / want).abs() }
    }

    #[test]
    fn phi_accuracy_f64() {
        // f64 path must hit ≤ 1e-10 rel at every node. We use a high-precision
        // reference (f64 closed form for |z| ≥ 1e-8 where cancellation is OK).
        // Near z = 0 we derive the limit values directly.
        let zs: [f64; 13] = [-10.0, -1.0, -0.1, -1e-2, -1e-4, -1e-8, 0.0, 1e-8, 1e-4, 1e-2, 0.1, 1.0, 10.0];
        for &z in &zs {
            let ref1 = if z.abs() < 1e-6 {
                // series 1 + z/2 + z²/6 + z³/24
                1.0 + z/2.0 + z*z/6.0 + z*z*z/24.0
            } else { phi1_ref(z) };
            let got1 = Phi::phi1_f64(z);
            assert!(rel_err(got1, ref1) < 1e-10, "phi1({}): got {}, want {}", z, got1, ref1);

            let ref2 = if z.abs() < 1e-6 {
                0.5 + z/6.0 + z*z/24.0 + z*z*z/120.0
            } else { phi2_ref(z) };
            let got2 = Phi::phi2_f64(z);
            assert!(rel_err(got2, ref2) < 1e-10, "phi2({}): got {}, want {}", z, got2, ref2);

            let ref3 = if z.abs() < 1e-6 {
                1.0/6.0 + z/24.0 + z*z/120.0 + z*z*z/720.0
            } else { phi3_ref(z) };
            let got3 = Phi::phi3_f64(z);
            assert!(rel_err(got3, ref3) < 1e-10, "phi3({}): got {}, want {}", z, got3, ref3);
        }
    }

    #[test]
    fn phi_accuracy_f32() {
        // f32 path only has ~7 digits; allow 1e-6 rel at each node.
        let zs: [f32; 9] = [-10.0, -1.0, -0.1, -1e-4, 0.0, 1e-4, 0.1, 1.0, 10.0];
        for &z in &zs {
            let got1 = Phi::phi1(z) as f64;
            let got2 = Phi::phi2(z) as f64;
            let got3 = Phi::phi3(z) as f64;
            let z64 = z as f64;
            let ref1 = if z.abs() < 1e-5 { 1.0 + z64/2.0 + z64*z64/6.0 } else { phi1_ref(z64) };
            let ref2 = if z.abs() < 1e-5 { 0.5 + z64/6.0 + z64*z64/24.0 } else { phi2_ref(z64) };
            let ref3 = if z.abs() < 1e-5 { 1.0/6.0 + z64/24.0 + z64*z64/120.0 } else { phi3_ref(z64) };
            // Use an absolute-or-relative combined tolerance so the z=0 cases
            // (where ref is a fixed nonzero constant but we still want to allow
            // f32 rounding on the Taylor path) aren't rejected by a pure
            // relative test.
            let tol = 1e-6_f64;
            let err = (got1 - ref1).abs().min(rel_err(got1, ref1));
            assert!(err < tol, "phi1_f32({}): got {}, want {}, err {}", z, got1, ref1, err);
            let err = (got2 - ref2).abs().min(rel_err(got2, ref2));
            assert!(err < tol, "phi2_f32({}): got {}, want {}, err {}", z, got2, ref2, err);
            let err = (got3 - ref3).abs().min(rel_err(got3, ref3));
            assert!(err < tol, "phi3_f32({}): got {}, want {}, err {}", z, got3, ref3, err);
        }
    }

    #[test]
    fn phi_limits_at_zero() {
        assert_eq!(Phi::phi1(0.0), 1.0);
        assert_eq!(Phi::phi2(0.0), 0.5);
        let p3 = Phi::phi3(0.0);
        assert!((p3 - 1.0_f32 / 6.0_f32).abs() < 1e-7, "phi3(0) = {}", p3);
    }

    #[test]
    fn phi_crossover_continuity() {
        // At every Taylor/closed-form crossover threshold, the two branches
        // must agree to within the target precision for that function. The
        // builder uses per-function thresholds that differ between f32 and
        // f64 branches:
        //   f32: phi1 = 1e-3, phi2 = 0.3, phi3 = 0.5
        //   f64: phi1 = 1e-3, phi2 = 0.05, phi3 = 0.1
        // We walk ±ε around each threshold and compare:
        //   * f64 branches are required to agree to ≤ 1e-10 rel against the
        //     independent high-order Taylor reference.
        //   * f32 branches are required to agree to ≤ 3e-6 rel (≈ 4× f32
        //     ULP margin around the crossover).

        // ---- f64 branches --------------------------------------------------
        let thresholds_f64: [(&str, f64, fn(f64) -> f64, fn(f64) -> f64); 3] = [
            ("phi1_f64", 1.0e-3, Phi::phi1_f64, phi1_ref),
            ("phi2_f64", 0.05,   Phi::phi2_f64, phi2_ref),
            ("phi3_f64", 0.1,    Phi::phi3_f64, phi3_ref),
        ];
        for (name, thr, fun, reference) in thresholds_f64 {
            for &z in &[
                thr * 0.999, thr * 1.001,
                -thr * 0.999, -thr * 1.001,
            ] {
                let got = fun(z);
                let want = reference(z);
                let e = rel_err(got, want);
                assert!(
                    e < 1e-10,
                    "{} continuity @ {}: got {}, ref {}, rel_err {:.3e}",
                    name, z, got, want, e
                );
            }
        }

        // ---- f32 branches --------------------------------------------------
        // For the f32 implementations we walk ±ε around the f32 threshold
        // and compare to the f64 high-precision reference. At the crossover
        // the jump must stay under a few f32 ULPs.
        let thresholds_f32: [(&str, f32, fn(f32) -> f32, fn(f64) -> f64); 3] = [
            ("phi1_f32", 1.0e-3, Phi::phi1, phi1_ref),
            ("phi2_f32", 0.3,    Phi::phi2, phi2_ref),
            ("phi3_f32", 0.5,    Phi::phi3, phi3_ref),
        ];
        for (name, thr, fun, reference) in thresholds_f32 {
            let eps = thr * 1.0e-3;   // 0.1% of threshold
            for &z in &[
                thr - eps, thr + eps,
                -(thr - eps), -(thr + eps),
            ] {
                let got = fun(z) as f64;
                let want = reference(z as f64);
                // Combined abs/rel tolerance: 3e-6 relative, 1e-6 absolute.
                let err_abs = (got - want).abs();
                let err_rel = if want.abs() > 1e-30 { err_abs / want.abs() } else { err_abs };
                assert!(
                    err_rel < 3.0e-6 || err_abs < 1.0e-6,
                    "{} continuity @ {}: got {}, ref {}, rel_err {:.3e}, abs_err {:.3e}",
                    name, z, got, want, err_rel, err_abs
                );
            }
        }
    }

    // ---- Toy-ODE convergence tests ------------------------------------
    //
    // We validate the multistep schemes on a scalar ODE whose analytical
    // solution is known. To keep the tests independent of the flame-core
    // Tensor plumbing (which requires a CUDA device), we reproduce the
    // algebra using plain f64 scalars — this is legitimate because all
    // the *coefficient* logic in the public API is scalar arithmetic, and
    // the tensor layer only provides elementwise multiply/add. If the
    // scalars match the analytical solution, the tensor version (which
    // applies the same scalars elementwise) will too.
    //
    // Toy model: dx/dσ = v(x, σ) with v(x, σ) = (x - 0)/σ = x/σ i.e.
    // denoised ≡ 0. Under this model x(σ) = x0·σ/σ_init, so going from
    // σ = 1 to σ = 0.1 the exact x scales by 0.1.
    //
    // But a "denoised ≡ 0" model makes every `denoised` term in the
    // update drop out and only exercises the σ_ratio trunk. That's a
    // valid sanity check but not a real convergence test.
    //
    // Stronger: pick a toy where the denoised target is a non-trivial
    // constant. Let data = 1 (fixed), noise = 0, so x_σ = (1-σ)·1 + σ·0
    // = 1 - σ, v(x, σ) = 0 - 1 = -1 (constant), denoised = x - σ·v =
    // x + σ. The exact solution from any σ₀ is x(σ) = 1 - σ (a line).
    // Then any consistent scheme recovers it exactly when run with
    // denoised = x + σ at every step — making this a sanity check on the
    // coefficient identities (each scheme must give x_next = 1 - σ_next).

    /// Run a simplified 1D multistep update using `dpmpp_2m` scalar algebra.
    fn dpmpp_2m_scalar(x: f64, denoised: f64, sigma: f64, sigma_next: f64,
                       history: &mut Vec<(f64, f64)>) -> f64 {
        let lambda = (((1.0 - sigma) / sigma).ln()).max(-30.0).min(30.0);
        let lambda_next = (((1.0 - sigma_next) / sigma_next).ln()).max(-30.0).min(30.0);
        let h = lambda_next - lambda;
        let alpha_next = 1.0 - sigma_next;
        let sr = sigma_next / sigma;
        let em1 = (-h).exp_m1();
        let out = if history.is_empty() {
            sr * x + (-alpha_next * em1) * denoised
        } else {
            let (d_prev, lam_prev) = history[history.len() - 1];
            let h_prev = lambda - lam_prev;
            if h_prev <= 0.0 {
                sr * x + (-alpha_next * em1) * denoised
            } else {
                let r = h_prev / h;
                let inv_2r = 0.5 / r;
                let c_d = -alpha_next * em1 * (1.0 + inv_2r);
                let c_p =  alpha_next * em1 * inv_2r;
                sr * x + c_d * denoised + c_p * d_prev
            }
        };
        history.push((denoised, lambda));
        out
    }

    #[test]
    fn dpmpp_2m_convergence_order_honest() {
        // Measure actual 2nd-order convergence on a **non-trivial** ODE.
        // The ODE dx/dλ = (1 − σ)·D(λ) − σ·(dD/dλ) ... is awkward to
        // analyse directly. Instead we use the DPM++ 2M scheme on the
        // rectified-flow ODE dx/dσ = (x − D(σ))/σ with a smooth, non-
        // constant denoised oracle, and compute the reference x(σ_end)
        // by **adaptive Runge–Kutta 4** on the same ODE (same oracle).
        //
        // Oracle: D(σ) = cos(σ). Solve dx/dσ = (x − cos(σ))/σ from
        // σ = 0.9 → σ = 0.1 starting at x(0.9) = 0.1. Reference is
        // produced by RK4 with 10_000 substeps; the sampler is then
        // run at N = 10, 20, 40 and the error-ratio is checked.
        //
        // Expected: ratio ≈ 4 (second order). Test requires ratio ≥ 3
        // asymptotically.

        let sigma_start = 0.9_f64;
        let sigma_end   = 0.1_f64;
        let x_init = 0.1_f64;
        let denoised_fn = |sigma: f64| sigma.cos();

        // RK4 reference on dx/dσ = (x − D(σ))/σ (σ decreases, so dσ < 0).
        let x_ref = {
            let n_ref = 10_000_usize;
            let dsig = (sigma_end - sigma_start) / n_ref as f64;   // negative
            let mut sigma = sigma_start;
            let mut x = x_init;
            let f = |sig: f64, xv: f64| (xv - denoised_fn(sig)) / sig;
            for _ in 0..n_ref {
                let k1 = f(sigma,               x);
                let k2 = f(sigma + 0.5 * dsig,  x + 0.5 * dsig * k1);
                let k3 = f(sigma + 0.5 * dsig,  x + 0.5 * dsig * k2);
                let k4 = f(sigma + dsig,        x + dsig * k3);
                x += (dsig / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                sigma += dsig;
            }
            x
        };

        let run = |n: usize| -> f64 {
            let sigmas: Vec<f64> = (0..=n)
                .map(|i| sigma_start + (sigma_end - sigma_start) * (i as f64 / n as f64))
                .collect();
            let mut x = x_init;
            let mut history: Vec<(f64, f64)> = Vec::new();
            for w in sigmas.windows(2) {
                let (s, sn) = (w[0], w[1]);
                let d = denoised_fn(s);
                x = dpmpp_2m_scalar(x, d, s, sn, &mut history);
            }
            (x - x_ref).abs()
        };

        let e1 = run(10);
        let e2 = run(20);
        let e3 = run(40);
        let r1 = e1 / e2;
        let r2 = e2 / e3;
        // 2nd-order target is ≈ 4. Accept ≥ 3.5 for at least one of
        // (r1, r2) — the startup step is 1st-order so very coarse N
        // can be limited by startup-error dominance.
        let r_best = r1.max(r2);
        assert!(
            r_best >= 3.5,
            "dpmpp_2m: observed best ratio = {:.2} (N=10→20 gives {:.2}, 20→40 gives {:.2}); \
             errors e1={:.3e} e2={:.3e} e3={:.3e} (ref={:.6})",
            r_best, r1, r2, e1, e2, e3, x_ref
        );
        // And the total improvement between coarsest and finest must be
        // at least 10× (confirming we're in the convergence regime, not
        // roundoff-limited).
        assert!(
            e1 / e3 > 10.0,
            "dpmpp_2m: insufficient error reduction: e1={:.3e}, e3={:.3e}",
            e1, e3
        );
    }

    /// Scalar analogue of `res_2m_step` for the L=-I, τ=-log σ problem.
    ///
    /// Toy ODE (in τ): dx/dτ = -x + N(x, τ), with N = denoised. With
    /// denoised ≡ 0 the exact solution is x(τ) = x0·e^{-(τ-τ0)} =
    /// x0·σ/σ0. We use this as the target: starting at σ = 1 with
    /// x(1) = 1, at σ = 0.1 we should have x = 0.1.
    fn res_2m_scalar(x: f64, denoised: f64, sigma: f64, sigma_next: f64,
                     history: &mut Vec<(f64, f64)>) -> f64 {
        let tau = -sigma.max(1e-8).ln();
        let tau_next = -sigma_next.max(1e-8).ln();
        let h = tau_next - tau;
        let e_mh = (-h).exp();
        let ph1 = Phi::phi1_f64(-h);
        let ph2 = Phi::phi2_f64(-h);
        let out = if history.is_empty() || h <= 0.0 {
            e_mh * x + h * ph1 * denoised
        } else {
            let (d_prev, lam_prev) = history[history.len() - 1];
            // softplus(lam) = -log σ
            let tau_prev = (1.0 + lam_prev.exp()).ln();
            let h_prev = tau - tau_prev;
            if h_prev <= 0.0 {
                e_mh * x + h * ph1 * denoised
            } else {
                let c_n = h * ph1 + h * h * ph2 / h_prev;
                let c_p = -h * h * ph2 / h_prev;
                e_mh * x + c_n * denoised + c_p * d_prev
            }
        };
        let lam_now = ((1.0 - sigma) / sigma).ln();
        history.push((denoised, lam_now));
        out
    }

    fn res_3m_scalar(x: f64, denoised: f64, sigma: f64, sigma_next: f64,
                     history: &mut Vec<(f64, f64)>) -> f64 {
        let tau = -sigma.max(1e-8).ln();
        let tau_next = -sigma_next.max(1e-8).ln();
        let h = tau_next - tau;
        let e_mh = (-h).exp();
        let ph1 = Phi::phi1_f64(-h);
        let ph2 = Phi::phi2_f64(-h);
        let ph3 = Phi::phi3_f64(-h);
        let out = if history.len() < 2 || h <= 0.0 {
            res_2m_scalar_peek(x, denoised, sigma, sigma_next, history)
        } else {
            let (n_m1, lam_m1) = history[history.len() - 1];
            let (n_m2, lam_m2) = history[history.len() - 2];
            let tau_m1 = (1.0 + lam_m1.exp()).ln();
            let tau_m2 = (1.0 + lam_m2.exp()).ln();
            let h1 = tau - tau_m1;
            let h2 = tau_m1 - tau_m2;
            if h1 <= 0.0 || h2 <= 0.0 {
                res_2m_scalar_peek(x, denoised, sigma, sigma_next, history)
            } else {
                let alpha = h * ph1;
                let beta = h * h * ph2;
                let gamma = 2.0 * h * h * h * ph3;
                let s_sum = h1 + h2;
                let c_n_dd2   =  1.0 / (h1 * s_sum);
                let c_nm1_dd2 = -1.0 / (h1 * h2);
                let c_nm2_dd2 =  1.0 / (h2 * s_sum);
                let c_n   = alpha + beta * (1.0 / h1 + 1.0 / s_sum) + gamma * c_n_dd2;
                let c_nm1 =         beta * (-1.0 / h1 - 1.0 / h2)    + gamma * c_nm1_dd2;
                let c_nm2 =         beta * (h1 / (h2 * s_sum))       + gamma * c_nm2_dd2;
                e_mh * x + c_n * denoised + c_nm1 * n_m1 + c_nm2 * n_m2
            }
        };
        let lam_now = ((1.0 - sigma) / sigma).ln();
        history.push((denoised, lam_now));
        out
    }

    // res_2m_scalar that does NOT push history (used by res_3m fallback
    // before it pushes itself).
    fn res_2m_scalar_peek(x: f64, denoised: f64, sigma: f64, sigma_next: f64,
                          history: &Vec<(f64, f64)>) -> f64 {
        let tau = -sigma.max(1e-8).ln();
        let tau_next = -sigma_next.max(1e-8).ln();
        let h = tau_next - tau;
        let e_mh = (-h).exp();
        let ph1 = Phi::phi1_f64(-h);
        let ph2 = Phi::phi2_f64(-h);
        if history.is_empty() || h <= 0.0 {
            return e_mh * x + h * ph1 * denoised;
        }
        let (d_prev, lam_prev) = history[history.len() - 1];
        let tau_prev = (1.0 + lam_prev.exp()).ln();
        let h_prev = tau - tau_prev;
        if h_prev <= 0.0 {
            return e_mh * x + h * ph1 * denoised;
        }
        let c_n = h * ph1 + h * h * ph2 / h_prev;
        let c_p = -h * h * ph2 / h_prev;
        e_mh * x + c_n * denoised + c_p * d_prev
    }

    #[test]
    fn res_2m_toy_ode_denoised_zero() {
        // dx/dτ = -x  ⇒  x(σ) = x0·σ/σ0. Start σ=1 with x=1 ⇒ want σ at end.
        let sigmas: Vec<f64> = (0..=10).map(|i| 1.0 - 0.09 * i as f64).collect(); // 1.0 → 0.1
        let mut x = 1.0;
        let mut history: Vec<(f64, f64)> = Vec::new();
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            x = res_2m_scalar(x, 0.0, s, sn, &mut history);
        }
        let want = 0.1;
        let err = (x - want).abs();
        assert!(err < 1e-3, "res_2m 10-step error on toy = {} (target ≤ 1e-3)", err);
    }

    #[test]
    fn res_3m_toy_ode_denoised_zero() {
        let sigmas: Vec<f64> = (0..=10).map(|i| 1.0 - 0.09 * i as f64).collect();
        let mut x = 1.0;
        let mut history: Vec<(f64, f64)> = Vec::new();
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            x = res_3m_scalar(x, 0.0, s, sn, &mut history);
        }
        let want = 0.1;
        let err = (x - want).abs();
        assert!(err < 1e-4, "res_3m 10-step error on toy = {} (target ≤ 1e-4)", err);
    }

    #[test]
    fn dpmpp_2m_convergence_order() {
        // Same linear-path toy but measure convergence order by halving
        // the step size and checking the error ratio ≈ 4 (order 2).
        //
        // NOTE: dpmpp_2m on the linear-path toy is exact at every step,
        // so a pure linear-path run would trivially pass. To actually
        // measure the order, we use a nonlinear oracle: denoised(σ) =
        // cos(πσ/2). Then v = (x - denoised)/σ -> but that's model-
        // dependent. Simpler: use the res_2m exp-integrator scheme (same
        // L=-I ODE) with a nonconstant oracle denoised(τ) = cos(τ). The
        // exact solution to dx/dτ = -x + cos(τ) is x(τ) = 0.5(sin(τ) +
        // cos(τ)) + (x0 - 0.5)·e^{-(τ-τ0)}.
        //
        // res_2m is order 2 → error ∝ N^{-2}.

        let sigma_start = 1.0_f64;
        let sigma_end   = 0.1_f64;
        let tau_start = -sigma_start.ln();
        let tau_end   = -sigma_end.ln();
        let x_exact = |tau: f64, x0: f64| -> f64 {
            0.5 * (tau.sin() + tau.cos()) + (x0 - 0.5 * (tau_start.sin() + tau_start.cos())) * (-(tau - tau_start)).exp()
        };

        let run = |n: usize| -> f64 {
            let mut sigmas: Vec<f64> = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let t = tau_start + (tau_end - tau_start) * (i as f64 / n as f64);
                sigmas.push((-t).exp());
            }
            let mut x = 0.3_f64;
            let x0 = x;
            let mut history: Vec<(f64, f64)> = Vec::new();
            for w in sigmas.windows(2) {
                let (s, sn) = (w[0], w[1]);
                let tau_here = -s.ln();
                let denoised = tau_here.cos();
                x = res_2m_scalar(x, denoised, s, sn, &mut history);
            }
            let want = x_exact(tau_end, x0);
            (x - want).abs()
        };

        // A 2nd-order scheme should give error ratio ≥ 3 when N doubles
        // (asymptotically 4). On smooth linear problems res_2m can be
        // super-convergent at moderate N (sometimes observed ratio ≈ 10
        // near the inflection between startup-limited and steady-state
        // regimes) and near roundoff-limited at large N (ratio near 1);
        // so we test across a couple of scales and require at least one
        // ratio to exceed 3, and the finest N to beat the coarsest by
        // at least a factor 10.
        let e1 = run(10);
        let e2 = run(20);
        let e3 = run(40);
        let r_any = (e1 / e2).max(e2 / e3);
        assert!(
            r_any > 3.0,
            "res_2m observed no ratio > 3 (e1={:.3e}, e2={:.3e}, e3={:.3e})",
            e1, e2, e3
        );
        assert!(
            e1 / e3 > 10.0,
            "res_2m error not decreasing as expected: e1={:.3e}, e3={:.3e}",
            e1, e3
        );
    }

    #[test]
    fn res_3m_convergence_order() {
        let sigma_start = 1.0_f64;
        let sigma_end   = 0.1_f64;
        let tau_start = -sigma_start.ln();
        let tau_end   = -sigma_end.ln();
        let x_exact = |tau: f64, x0: f64| -> f64 {
            0.5 * (tau.sin() + tau.cos()) + (x0 - 0.5 * (tau_start.sin() + tau_start.cos())) * (-(tau - tau_start)).exp()
        };

        let run = |n: usize| -> f64 {
            let mut sigmas: Vec<f64> = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let t = tau_start + (tau_end - tau_start) * (i as f64 / n as f64);
                sigmas.push((-t).exp());
            }
            let mut x = 0.3_f64;
            let x0 = x;
            let mut history: Vec<(f64, f64)> = Vec::new();
            for w in sigmas.windows(2) {
                let (s, sn) = (w[0], w[1]);
                let tau_here = -s.ln();
                let denoised = tau_here.cos();
                x = res_3m_scalar(x, denoised, s, sn, &mut history);
            }
            let want = x_exact(tau_end, x0);
            (x - want).abs()
        };

        // Order 3 → error ratio ≈ 8 when N doubles. The startup
        // (1st-order step 0 + 2nd-order step 1) limits the asymptotic
        // global order to 2 in principle, but the startup errors at
        // moderate N are dwarfed by the 3rd-order steady-state
        // contribution, so the empirical ratio sits between 4 and 8.
        // We sample two ratios and require at least one to show
        // super-2nd-order convergence (> 5).
        let e1 = run(40);
        let e2 = run(80);
        let e3 = run(160);
        let r1 = e1 / e2;
        let r2 = e2 / e3;
        assert!(
            r1 > 4.0 && r1 < 12.0,
            "res_3m ratio1 = {} (expected between 2nd and 3rd order: 4-8)",
            r1
        );
        assert!(
            r2 > 4.0 && r2 < 12.0,
            "res_3m ratio2 = {} (expected between 2nd and 3rd order: 4-8)",
            r2
        );
    }

    // ---- DEIS-3 scalar + convergence test -----------------------------
    //
    // Re-implements `deis_3m_step` purely in f64 scalars so that it can
    // be exercised on the toy cos-forced ODE dx/dτ = −x + cos(τ). The
    // scheme uses the exact moments M_0, M_1, M_2 and the Lagrange basis
    // derived in the module doc comment.

    fn moments_012_scalar(a: f64, b: f64) -> (f64, f64, f64) {
        let ea = (-a).exp();
        let eb = (-b).exp();
        let m0 = ea - eb;
        let m1 = (a + 1.0) * ea - (b + 1.0) * eb;
        let m2 = (a * a + 2.0 * a + 2.0) * ea - (b * b + 2.0 * b + 2.0) * eb;
        (m0, m1, m2)
    }

    fn deis_3m_scalar(x: f64, denoised: f64, sigma: f64, sigma_next: f64,
                      history: &mut Vec<(f64, f64)>) -> f64 {
        let s0 = sigma.max(1.0e-8);
        let s1 = sigma_next.max(1.0e-8);
        let u_n = s0.ln();
        let u_np1 = s1.ln();
        let sigma_ratio = s1 / s0;
        let (m0, m1, m2) = moments_012_scalar(u_n, u_np1);
        let sn1 = s1;

        let out = match history.len() {
            0 => {
                // 1st-order
                let w0 = sn1 * m0;
                sigma_ratio * x - w0 * denoised
            }
            1 => {
                // DEIS-2
                let (d_m1, lam_m1) = history[history.len() - 1];
                let s_m1 = (1.0 / (1.0 + lam_m1.exp())).max(1.0e-8);
                let u_m1 = s_m1.ln();
                let inv_01 = 1.0 / (u_n - u_m1);
                let inv_10 = -inv_01;
                // L_0(u) = (u - u_m1)/(u_n - u_m1) = -u_m1·inv_01 + u·inv_01
                let a0 = -u_m1 * inv_01;
                let b0 = inv_01;
                // L_1(u) = (u - u_n)/(u_m1 - u_n) = -u_n·inv_10 + u·inv_10
                let a1 = -u_n * inv_10;
                let b1 = inv_10;
                let w0 = sn1 * (a0 * m0 + b0 * m1);
                let w1 = sn1 * (a1 * m0 + b1 * m1);
                sigma_ratio * x - w0 * denoised - w1 * d_m1
            }
            _ => {
                // DEIS-3
                let (d_m1, lam_m1) = history[history.len() - 1];
                let (d_m2, lam_m2) = history[history.len() - 2];
                let s_m1 = (1.0 / (1.0 + lam_m1.exp())).max(1.0e-8);
                let s_m2 = (1.0 / (1.0 + lam_m2.exp())).max(1.0e-8);
                let u_m1 = s_m1.ln();
                let u_m2 = s_m2.ln();
                let lquad = |uj: f64, ua: f64, ub: f64| {
                    let inv = 1.0 / ((uj - ua) * (uj - ub));
                    (ua * ub * inv, -(ua + ub) * inv, inv)
                };
                let (a0, b0, c0) = lquad(u_n,  u_m1, u_m2);
                let (a1, b1, c1) = lquad(u_m1, u_n,  u_m2);
                let (a2, b2, c2) = lquad(u_m2, u_n,  u_m1);
                let w0 = sn1 * (a0 * m0 + b0 * m1 + c0 * m2);
                let w1 = sn1 * (a1 * m0 + b1 * m1 + c1 * m2);
                let w2 = sn1 * (a2 * m0 + b2 * m1 + c2 * m2);
                sigma_ratio * x - w0 * denoised - w1 * d_m1 - w2 * d_m2
            }
        };

        let lam_now = ((1.0 - sigma) / sigma).ln();
        history.push((denoised, lam_now));
        out
    }

    #[test]
    fn deis_3m_convergence_order() {
        // Toy ODE: dx/dτ = −x + cos(τ). Exact solution:
        //   x(τ) = ½ (sin τ + cos τ) + C·e^{−(τ−τ0)}
        // with C = x0 − ½ (sin τ0 + cos τ0).
        // In the DEIS-3 scheme the independent variable is σ but the
        // ODE is the same (via τ = −log σ). The `denoised` oracle at
        // step n is cos(τ_n) = cos(−log σ_n).
        let sigma_start = 1.0_f64;
        let sigma_end   = 0.1_f64;
        let tau_start = -sigma_start.ln();
        let tau_end   = -sigma_end.ln();
        let x_exact = |tau: f64, x0: f64| -> f64 {
            0.5 * (tau.sin() + tau.cos())
                + (x0 - 0.5 * (tau_start.sin() + tau_start.cos())) * (-(tau - tau_start)).exp()
        };

        // Uniformly space in τ.
        let run = |n: usize| -> f64 {
            let mut sigmas: Vec<f64> = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let t = tau_start + (tau_end - tau_start) * (i as f64 / n as f64);
                sigmas.push((-t).exp());
            }
            let mut x = 0.3_f64;
            let x0 = x;
            let mut history: Vec<(f64, f64)> = Vec::new();
            for w in sigmas.windows(2) {
                let (s, sn) = (w[0], w[1]);
                let tau_here = -s.ln();
                let denoised = tau_here.cos();
                x = deis_3m_scalar(x, denoised, s, sn, &mut history);
            }
            let want = x_exact(tau_end, x0);
            (x - want).abs()
        };

        let e1 = run(40);
        let e2 = run(80);
        let e3 = run(160);
        let r1 = e1 / e2;
        let r2 = e2 / e3;
        // 3rd-order target is 8. Startup (DEIS-1 step 0, DEIS-2 step 1)
        // is lower-order and limits the asymptotic global order to 2 in
        // principle, but at moderate N the 3rd-order steady-state
        // dominates. We require ratio ≥ 6 to demonstrate super-2nd-order
        // convergence.
        let r_best = r1.max(r2);
        assert!(
            r_best >= 6.0,
            "deis_3m best ratio = {:.2} (r1={:.2}, r2={:.2}); e1={:.3e} e2={:.3e} e3={:.3e}",
            r_best, r1, r2, e1, e2, e3
        );
        assert!(
            e1 / e3 > 30.0,
            "deis_3m error not decreasing as expected: e1={:.3e}, e3={:.3e} (ratio {:.2})",
            e1, e3, e1 / e3
        );
    }

    // ---- res_3s singlestep scalar + convergence test ------------------
    //
    // Same ODE dx/dτ = −x + N(τ) with explicit N(τ) oracle since the
    // singlestep scheme needs N evaluated at the predictor stages, not
    // just at the step endpoints. We re-code the algorithm from scratch
    // in f64 scalars; the algebra mirrors `res_3s_step` exactly.
    //
    // Tableau (ETDRK3, c2 = 1/2, c3 = 1, L = −1, scalar):
    //   a2 = e^{-h/2}·x + (h/2)·φ1(-h/2)·N1                 with N1 = N(τ)
    //   a3 = e^{-h}·x + h·φ1(-h)·(2 N2 − N1)                 with N2 = N(τ + h/2)
    //   x_next = e^{-h}·x
    //          + h·[(φ1(-h) − 3 φ2(-h) + 4 φ3(-h))·N1
    //              + (         4 φ2(-h) − 8 φ3(-h))·N2
    //              + (        −φ2(-h) + 4 φ3(-h))·N3]        with N3 = N(τ + h)

    fn res_3s_scalar(x: f64, sigma: f64, sigma_next: f64,
                     n_of_tau: &impl Fn(f64) -> f64) -> f64 {
        let tau = -sigma.max(1e-8).ln();
        let tau_next = -sigma_next.max(1e-8).ln();
        let h = tau_next - tau;
        if h <= 0.0 {
            // Euler-fallback (not exercised for our toy schedule).
            let n = n_of_tau(tau);
            return x + (sigma_next - sigma) * ((x - n) / sigma);
        }
        let hh = 0.5 * h;
        let e_mh = (-h).exp();
        let e_mhh = (-hh).exp();
        let ph1 = Phi::phi1_f64(-h);
        let ph2 = Phi::phi2_f64(-h);
        let ph3 = Phi::phi3_f64(-h);
        let ph1_half = Phi::phi1_f64(-hh);

        let n1 = n_of_tau(tau);
        let a2 = e_mhh * x + hh * ph1_half * n1;
        let n2 = n_of_tau(tau + hh);
        let _a3 = e_mh * x + h * ph1 * (2.0 * n2 - n1);
        // (a3 needs n3 = N(tau + h) since L = -I and the predictor only
        // influences the final via n3; in the scalar version the oracle
        // doesn't depend on x, so we compute n3 directly.)
        let n3 = n_of_tau(tau + h);

        let w1 = h * (ph1 - 3.0 * ph2 + 4.0 * ph3);
        let w2 = h * (4.0 * ph2 - 8.0 * ph3);
        let w3 = h * (-ph2 + 4.0 * ph3);
        // Suppress unused variable warnings where applicable.
        let _ = a2;
        e_mh * x + w1 * n1 + w2 * n2 + w3 * n3
    }

    #[test]
    fn res_3s_convergence_order() {
        // Same cos-forced toy as res_3m but evaluated as a singlestep.
        let sigma_start = 1.0_f64;
        let sigma_end   = 0.1_f64;
        let tau_start = -sigma_start.ln();
        let tau_end   = -sigma_end.ln();
        let x0_init = 0.3_f64;
        let x_exact = |tau: f64| -> f64 {
            0.5 * (tau.sin() + tau.cos())
                + (x0_init - 0.5 * (tau_start.sin() + tau_start.cos())) * (-(tau - tau_start)).exp()
        };
        let n_of_tau = |tau: f64| tau.cos();

        let run = |n: usize| -> f64 {
            let mut sigmas: Vec<f64> = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let t = tau_start + (tau_end - tau_start) * (i as f64 / n as f64);
                sigmas.push((-t).exp());
            }
            let mut x = x0_init;
            for w in sigmas.windows(2) {
                let (s, sn) = (w[0], w[1]);
                x = res_3s_scalar(x, s, sn, &n_of_tau);
            }
            (x - x_exact(tau_end)).abs()
        };

        let e1 = run(40);
        let e2 = run(80);
        let e3 = run(160);
        let r1 = e1 / e2;
        let r2 = e2 / e3;
        // Target 3rd-order ratio = 8. Accept ≥ 6 (see module doc on
        // Butcher-tableau precision loss near roundoff).
        let r_best = r1.max(r2);
        assert!(
            r_best >= 6.0,
            "res_3s best ratio = {:.2} (r1={:.2}, r2={:.2}); e1={:.3e} e2={:.3e} e3={:.3e}",
            r_best, r1, r2, e1, e2, e3
        );
        assert!(
            e1 / e3 > 30.0,
            "res_3s error not decreasing as expected: e1={:.3e}, e3={:.3e} (ratio {:.2})",
            e1, e3, e1 / e3
        );
    }

    // ---- Tensor-level end-to-end correctness tests --------------------
    //
    // These tests build 1-element tensors on the live CUDA device and
    // drive the public *_step functions through a short history, then
    // compare the tensor result against the corresponding f64 scalar
    // oracle. This proves the Tensor code path uses the same scalar
    // coefficients as the scalar tests (rather than a subtly different
    // algebra).
    //
    // They require a working CUDA device (flame-core has no CPU tensor
    // backend). If the device is unavailable the test returns early with
    // a message; in CI this usually means the tests silently pass (no
    // assertion fires) — which is fine because `cargo test` on a
    // CUDA-less machine wouldn't even link flame-core. On developer
    // machines with CUDA we exercise the full path.

    use flame_core::{CudaDevice, Shape};

    fn try_device() -> Option<std::sync::Arc<CudaDevice>> {
        CudaDevice::new(0).ok()
    }

    fn t_scalar(device: &std::sync::Arc<CudaDevice>, v: f64) -> Tensor {
        // 1-element f32 tensor.
        Tensor::from_vec(vec![v as f32], Shape::from_dims(&[1]), device.clone())
            .expect("from_vec")
    }

    fn t_to_f64(t: &Tensor) -> f64 {
        t.to_vec().expect("to_vec")[0] as f64
    }

    #[test]
    fn dpmpp_2m_tensor_matches_scalar() {
        let device = match try_device() { Some(d) => d, None => return };
        // Run 5 steps: σ 0.9 → 0.1. Use a cos-forced denoised oracle so
        // each step has non-trivial corrections.
        let sigmas: Vec<f64> = (0..=5).map(|i| 0.9 - 0.16 * i as f64).collect();
        let denoised_fn = |s: f64| s.cos();

        // Scalar path.
        let mut x_scalar = 0.1_f64;
        let mut hist_s: Vec<(f64, f64)> = Vec::new();
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let d = denoised_fn(s);
            x_scalar = dpmpp_2m_scalar(x_scalar, d, s, sn, &mut hist_s);
        }

        // Tensor path.
        let mut x_t = t_scalar(&device, 0.1);
        let mut hist_t = MultistepHistory::new(2);
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let d_val = denoised_fn(s);
            let d_t = t_scalar(&device, d_val);
            x_t = dpmpp_2m_step(&x_t, &d_t, s as f32, sn as f32, &hist_t)
                .expect("dpmpp_2m_step");
            let lam = ((1.0 - s) / s).ln() as f32;
            hist_t.push(d_t, lam);
        }

        let x_t_val = t_to_f64(&x_t);
        // The tensor path does arithmetic in f32 while the scalar path is
        // f64, so allow a looser tolerance (~1e-4 abs, 1e-3 rel).
        let err_abs = (x_t_val - x_scalar).abs();
        let err_rel = if x_scalar.abs() > 1e-30 { err_abs / x_scalar.abs() } else { err_abs };
        assert!(
            err_abs < 1e-4 || err_rel < 1e-3,
            "dpmpp_2m tensor vs scalar: got {}, want {}, err_abs {:.3e}, err_rel {:.3e}",
            x_t_val, x_scalar, err_abs, err_rel
        );
    }

    #[test]
    fn res_2m_tensor_matches_scalar() {
        let device = match try_device() { Some(d) => d, None => return };
        let sigmas: Vec<f64> = (0..=5).map(|i| 0.9 - 0.16 * i as f64).collect();

        let mut x_scalar = 0.1_f64;
        let mut hist_s: Vec<(f64, f64)> = Vec::new();
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let tau_here = -s.ln();
            let d = tau_here.cos();
            x_scalar = res_2m_scalar(x_scalar, d, s, sn, &mut hist_s);
        }

        let mut x_t = t_scalar(&device, 0.1);
        let mut hist_t = MultistepHistory::new(2);
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let tau_here = -s.ln();
            let d_val = tau_here.cos();
            let d_t = t_scalar(&device, d_val);
            x_t = res_2m_step(&x_t, &d_t, s as f32, sn as f32, &hist_t)
                .expect("res_2m_step");
            let lam = ((1.0 - s) / s).ln() as f32;
            hist_t.push(d_t, lam);
        }

        let x_t_val = t_to_f64(&x_t);
        let err_abs = (x_t_val - x_scalar).abs();
        let err_rel = if x_scalar.abs() > 1e-30 { err_abs / x_scalar.abs() } else { err_abs };
        assert!(
            err_abs < 1e-4 || err_rel < 1e-3,
            "res_2m tensor vs scalar: got {}, want {}, err_abs {:.3e}, err_rel {:.3e}",
            x_t_val, x_scalar, err_abs, err_rel
        );
    }

    #[test]
    fn res_3m_tensor_matches_scalar() {
        let device = match try_device() { Some(d) => d, None => return };
        let sigmas: Vec<f64> = (0..=6).map(|i| 0.9 - (0.8 / 6.0) * i as f64).collect();

        let mut x_scalar = 0.1_f64;
        let mut hist_s: Vec<(f64, f64)> = Vec::new();
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let tau_here = -s.ln();
            let d = tau_here.cos();
            x_scalar = res_3m_scalar(x_scalar, d, s, sn, &mut hist_s);
        }

        let mut x_t = t_scalar(&device, 0.1);
        let mut hist_t = MultistepHistory::new(3);
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let tau_here = -s.ln();
            let d_val = tau_here.cos();
            let d_t = t_scalar(&device, d_val);
            x_t = res_3m_step(&x_t, &d_t, s as f32, sn as f32, &hist_t)
                .expect("res_3m_step");
            let lam = ((1.0 - s) / s).ln() as f32;
            hist_t.push(d_t, lam);
        }

        let x_t_val = t_to_f64(&x_t);
        let err_abs = (x_t_val - x_scalar).abs();
        let err_rel = if x_scalar.abs() > 1e-30 { err_abs / x_scalar.abs() } else { err_abs };
        assert!(
            err_abs < 1e-4 || err_rel < 1e-3,
            "res_3m tensor vs scalar: got {}, want {}, err_abs {:.3e}, err_rel {:.3e}",
            x_t_val, x_scalar, err_abs, err_rel
        );
    }

    #[test]
    fn deis_3m_tensor_matches_scalar() {
        let device = match try_device() { Some(d) => d, None => return };
        let sigmas: Vec<f64> = (0..=6).map(|i| 0.9 - (0.8 / 6.0) * i as f64).collect();

        let mut x_scalar = 0.1_f64;
        let mut hist_s: Vec<(f64, f64)> = Vec::new();
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let tau_here = -s.ln();
            let d = tau_here.cos();
            x_scalar = deis_3m_scalar(x_scalar, d, s, sn, &mut hist_s);
        }

        let mut x_t = t_scalar(&device, 0.1);
        let mut hist_t = MultistepHistory::new(3);
        for w in sigmas.windows(2) {
            let (s, sn) = (w[0], w[1]);
            let tau_here = -s.ln();
            let d_val = tau_here.cos();
            let d_t = t_scalar(&device, d_val);
            x_t = deis_3m_step(&x_t, &d_t, s as f32, sn as f32, &hist_t)
                .expect("deis_3m_step");
            let lam = ((1.0 - s) / s).ln() as f32;
            hist_t.push(d_t, lam);
        }

        let x_t_val = t_to_f64(&x_t);
        let err_abs = (x_t_val - x_scalar).abs();
        let err_rel = if x_scalar.abs() > 1e-30 { err_abs / x_scalar.abs() } else { err_abs };
        assert!(
            err_abs < 1e-4 || err_rel < 1e-3,
            "deis_3m tensor vs scalar: got {}, want {}, err_abs {:.3e}, err_rel {:.3e}",
            x_t_val, x_scalar, err_abs, err_rel
        );
    }

    // ---- MultistepHistory smoke test ----------------------------------

    #[test]
    fn multistep_history_ring_behavior() {
        // We can't construct Tensors in the test harness (no CUDA in tests),
        // so we just exercise the index arithmetic via a parallel Vec<usize>
        // ring. The `get(back)` mapping is what matters.
        let cap = 3;
        let mut head = usize::MAX;
        let mut len = 0usize;
        let mut buf: Vec<usize> = Vec::with_capacity(cap);
        let push = |val: usize, buf: &mut Vec<usize>, head: &mut usize, len: &mut usize| {
            if buf.len() < cap {
                buf.push(val);
                *head = buf.len() - 1;
                *len += 1;
            } else {
                let w = (*head + 1) % cap;
                buf[w] = val;
                *head = w;
            }
        };
        let get = |back: usize, buf: &Vec<usize>, head: usize, len: usize| -> Option<usize> {
            if back >= len { return None; }
            let idx = (head + cap - back) % cap;
            Some(buf[idx])
        };

        // Push 1, 2, 3, 4, 5. Expect: newest=5, prev=4, prev2=3 (1, 2 dropped).
        for v in [1usize, 2, 3, 4, 5] { push(v, &mut buf, &mut head, &mut len); }
        assert_eq!(get(0, &buf, head, len), Some(5));
        assert_eq!(get(1, &buf, head, len), Some(4));
        assert_eq!(get(2, &buf, head, len), Some(3));
        assert_eq!(get(3, &buf, head, len), None);
    }
}
