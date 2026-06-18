//! Ideogram 4 — logit-normal flow-matching schedule + sampler presets.
//!
//! Mirrors `/home/alex/ideogram4-ref/src/ideogram4/scheduler.py` and
//! `sampler_configs.py` EXACTLY. Host-side f64 math (no GPU): the schedule
//! produces per-step `t` values that drive the Euler flow loop in the
//! (chunk-4b) infer bin.
//!
//! ## LogitNormalSchedule (`scheduler.py:11-26`)
//!
//! ```python
//! def __call__(self, t):
//!   t = t.to(float64)
//!   z = ndtri(t)                              # inverse normal CDF (probit)
//!   y = mean + std * z
//!   t_ = expit(y)                             # logistic sigmoid
//!   t_ = 1 - t_
//!   t_min = 1 / (1 + exp(0.5 * logsnr_max))
//!   t_max = 1 / (1 + exp(0.5 * logsnr_min))
//!   return t_.clamp(t_min, t_max).to(float32)
//! ```
//!
//! with `logsnr_min = -15`, `logsnr_max = 18`. Note `t_min` uses `logsnr_MAX`
//! and `t_max` uses `logsnr_MIN` — the schedule is monotone DECREASING in the
//! logit-normal argument so the larger logsnr gives the smaller t. This is
//! mirrored verbatim; do not "fix" the apparent swap.
//!
//! ## Resolution-aware mean (`scheduler.py:29-39`)
//!
//! `mean = known_mean + 0.5 * ln(H*W / (512*512))`. `known_mean` is the preset
//! `mu`; `std` is the preset `std`.
//!
//! ## Step intervals (`scheduler.py:42-44`)
//!
//! `make_step_intervals(n) = linspace(0, 1, n+1)` (n+1 points).
//!
//! ## Sampler presets (`sampler_configs.py`)
//!
//! `guidance_schedule` is in **loop-INDEX order: index 0 is the LAST sampling
//! step (final polish), index num_steps-1 is the FIRST sampling step.** The
//! Euler loop runs `for i in (num_steps-1 ..= 0)` and reads `gw_per_step[i]`,
//! so index `i` lines up with loop iteration `i`. This convention is preserved
//! exactly — the registry stores the tuples in the same order Python does.
//!
//! ## ndtri / expit
//!
//! `ndtri` (inverse normal CDF / probit) is implemented via the Acklam rational
//! approximation refined by one Halley step (≈ machine precision in f64 over
//! the open interval (0,1)). `expit(y) = 1/(1+exp(-y))`. Both compute in f64,
//! matching the reference's `t.to(float64)`.

/// Logit-normal flow-matching schedule (resolution-aware mean).
///
/// Mirrors `scheduler.py::LogitNormalSchedule`. `eval` maps an interval point
/// `t in (0,1)` to a flow-matching timestep, clamped by the logsnr bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitNormalSchedule {
    /// Logit-normal mean (resolution-adjusted `known_mean + 0.5*ln(HW/512²)`).
    pub mean: f64,
    /// Logit-normal std (preset `std`).
    pub std: f64,
    /// Min log-SNR. Reference default `-15.0`.
    pub logsnr_min: f64,
    /// Max log-SNR. Reference default `18.0`.
    pub logsnr_max: f64,
}

impl LogitNormalSchedule {
    /// Build with the reference default logsnr bounds (`scheduler.py:15-16`).
    pub fn new(mean: f64, std: f64) -> Self {
        Self {
            mean,
            std,
            logsnr_min: -15.0,
            logsnr_max: 18.0,
        }
    }

    /// Evaluate the schedule at interval point `t in (0,1)`.
    ///
    /// Returns the clamped flow-matching timestep as f32 (the reference casts
    /// the result back to float32 at `scheduler.py:26`). All intermediate math
    /// is f64.
    pub fn eval(&self, t: f64) -> f32 {
        let z = ndtri(t);
        let y = self.mean + self.std * z;
        let t_ = 1.0 - expit(y);
        // NOTE: t_min uses logsnr_MAX, t_max uses logsnr_MIN — verbatim from
        // scheduler.py:24-25. Not a typo.
        let t_min = 1.0 / (1.0 + (0.5 * self.logsnr_max).exp());
        let t_max = 1.0 / (1.0 + (0.5 * self.logsnr_min).exp());
        let clamped = t_.clamp(t_min, t_max);
        clamped as f32
    }
}

/// Resolution-aware schedule (`scheduler.py:29-39`).
///
/// `mean = known_mean + 0.5 * ln((h*w) / (known_h*known_w))` with the default
/// known resolution `512×512`. `known_mean` is the preset `mu`; `std` is the
/// preset `std`.
pub fn get_schedule_for_resolution(
    height: usize,
    width: usize,
    known_mean: f64,
    std: f64,
) -> LogitNormalSchedule {
    // Reference fixes known_resolution = (512, 512) at scheduler.py:31.
    const KNOWN_H: usize = 512;
    const KNOWN_W: usize = 512;
    let num_pixels = (height * width) as f64;
    let known_pixels = (KNOWN_H * KNOWN_W) as f64;
    let mean = known_mean + 0.5 * (num_pixels / known_pixels).ln();
    LogitNormalSchedule::new(mean, std)
}

/// `make_step_intervals(n) = linspace(0, 1, n+1)` (`scheduler.py:42-44`).
///
/// Returns `n+1` evenly spaced points from 0.0 to 1.0 inclusive. f32 to match
/// `dtype=torch.float32`.
pub fn make_step_intervals(num_steps: usize) -> Vec<f32> {
    let n = num_steps;
    if n == 0 {
        // linspace(0,1,1) = [0.0] in torch.
        return vec![0.0];
    }
    (0..=n)
        .map(|i| {
            // torch.linspace endpoints are exact (0.0 and 1.0); interior is
            // start + i*(stop-start)/(n) computed in f32-equivalent. Use f64
            // for the interior division then cast, matching torch's internal
            // double-accumulate step.
            if i == 0 {
                0.0f32
            } else if i == n {
                1.0f32
            } else {
                (i as f64 / n as f64) as f32
            }
        })
        .collect()
}

/// A named sampler preset (`sampler_configs.py::SamplerParameters`).
///
/// `guidance_schedule` is in LOOP-INDEX order (index 0 = last/polish step).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerParameters {
    /// Number of sampling steps.
    pub num_steps: usize,
    /// Per-step guidance weights, loop-INDEX order (index 0 = LAST step).
    /// Length must equal `num_steps`.
    pub guidance_schedule: Vec<f32>,
    /// Logit-normal mean (`known_mean` passed to `get_schedule_for_resolution`).
    pub mu: f64,
    /// Logit-normal std.
    pub std: f64,
}

impl SamplerParameters {
    /// Validate the `len(guidance_schedule) == num_steps` invariant
    /// (`scheduler.py:65-70` `__post_init__`).
    pub fn validate(&self) -> Result<(), String> {
        if self.guidance_schedule.len() != self.num_steps {
            return Err(format!(
                "guidance_schedule has length {}, expected num_steps={}",
                self.guidance_schedule.len(),
                self.num_steps
            ));
        }
        Ok(())
    }
}

/// Look up a named preset (`sampler_configs.py::PRESETS`).
///
/// Returns `None` for an unknown name. Known names: `V4_QUALITY_48` (default),
/// `V4_DEFAULT_20`, `V4_TURBO_12`.
pub fn preset(name: &str) -> Option<SamplerParameters> {
    match name {
        "V4_QUALITY_48" => Some(SamplerParameters {
            num_steps: 48,
            // (3,)*3 + (7,)*45
            guidance_schedule: build_gw(&[(3.0, 3), (7.0, 45)]),
            mu: 0.0,
            std: 1.5,
        }),
        "V4_DEFAULT_20" => Some(SamplerParameters {
            num_steps: 20,
            // (3,)*2 + (7,)*18
            guidance_schedule: build_gw(&[(3.0, 2), (7.0, 18)]),
            mu: 0.0,
            std: 1.75,
        }),
        "V4_TURBO_12" => Some(SamplerParameters {
            num_steps: 12,
            // (3,)*1 + (7,)*11
            guidance_schedule: build_gw(&[(3.0, 1), (7.0, 11)]),
            mu: 0.5,
            std: 1.75,
        }),
        _ => None,
    }
}

/// Expand `(value, count)` runs into a flat guidance vector (Python `(v,)*n + ...`).
fn build_gw(runs: &[(f32, usize)]) -> Vec<f32> {
    let mut v = Vec::new();
    for &(val, count) in runs {
        v.extend(std::iter::repeat(val).take(count));
    }
    v
}

/// Logistic sigmoid `expit(y) = 1 / (1 + exp(-y))`. f64.
///
/// `torch.special.expit`. Branchless-stable form to avoid overflow at large
/// `|y|` (matches torch's numerically-stable sigmoid).
#[inline]
pub fn expit(y: f64) -> f64 {
    if y >= 0.0 {
        1.0 / (1.0 + (-y).exp())
    } else {
        let e = y.exp();
        e / (1.0 + e)
    }
}

/// Inverse normal CDF / probit `ndtri(p)` for `p in (0,1)`. f64.
///
/// `torch.special.ndtri`. Acklam's rational approximation (relative error
/// < 1.15e-9 before refinement) followed by ONE Halley step using `erfc`,
/// which brings the result to full f64 accuracy. Returns ±inf at the endpoints
/// `p<=0` / `p>=1` and NaN for out-of-range, matching torch's `ndtri` edge
/// behaviour (ndtri(0)=-inf, ndtri(1)=+inf).
pub fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return if p == 0.0 { f64::NEG_INFINITY } else { f64::NAN };
    }
    if p >= 1.0 {
        return if p == 1.0 { f64::INFINITY } else { f64::NAN };
    }

    // Acklam coefficients.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    // Region boundaries.
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        // Lower tail.
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region.
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail.
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement step: x_{n+1} = x - u/(1 + x*u/2),
    // u = (Phi(x) - p) / phi(x), where Phi is the normal CDF (via erfc) and
    // phi the normal PDF. This removes the ~1e-9 Acklam error.
    let e = 0.5 * erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Complementary error function `erfc(x)` for f64.
///
/// Used only by the `ndtri` Halley step, so it must be accurate enough that the
/// refinement does NOT degrade the ~1e-9 Acklam start (a coarse ~1e-7 erfc
/// nudges `ndtri(0.5)` off zero). This is W. J. Cody's rational Chebyshev
/// approximation (Math. Comp. 1969 / SPECFUN `calerf`), ≈ full f64 accuracy
/// (rel err < 1e-15). `erfc(0) = 1` exactly, so the Halley residual at the
/// median is exactly 0.
fn erfc(x: f64) -> f64 {
    // erfc(-x) = 2 - erfc(x); compute for |x| then reflect.
    let z = x.abs();

    if z < 0.5 {
        // Region 1: erf via rational approx, erfc = 1 - erf.
        const P: [f64; 5] = [
            3.16112374387056560e+00,
            1.13864154151050156e+02,
            3.77485237685302021e+02,
            3.20937758913846947e+03,
            1.85777706184603153e-01,
        ];
        const Q: [f64; 4] = [
            2.36012909523441209e+01,
            2.44024637934444173e+02,
            1.28261652607737228e+03,
            2.84423683343917062e+03,
        ];
        let y = z * z;
        let num = P[4] * y;
        let num = (((num + P[0]) * y + P[1]) * y + P[2]) * y + P[3];
        let den = (((y + Q[0]) * y + Q[1]) * y + Q[2]) * y + Q[3];
        let erf = x * num / den; // erf is odd → use signed x
        return 1.0 - erf;
    }

    let ans = if z < 4.0 {
        // Region 2.
        const P: [f64; 9] = [
            5.64188496988670089e-01,
            8.88314979438837594e+00,
            6.61191906371416295e+01,
            2.98635138197400131e+02,
            8.81952221241769090e+02,
            1.71204761263407058e+03,
            2.05107837782607147e+03,
            1.23033935479799725e+03,
            2.15311535474403846e-08,
        ];
        const Q: [f64; 8] = [
            1.57449261107098347e+01,
            1.17693950891312499e+02,
            5.37181101862009858e+02,
            1.62138957456669019e+03,
            3.29079923573345963e+03,
            4.36261909014324716e+03,
            3.43936767414372164e+03,
            1.23033935480374942e+03,
        ];
        let mut num = P[8] * z;
        let mut den = z;
        for i in 0..7 {
            num = (num + P[i]) * z;
            den = (den + Q[i]) * z;
        }
        num = (num + P[7]) / (den + Q[7]);
        let trunc = (z * 16.0).floor() / 16.0;
        let del = (z - trunc) * (z + trunc);
        (-trunc * trunc).exp() * (-del).exp() * num
    } else {
        // Region 3 (large z): asymptotic.
        const P: [f64; 6] = [
            3.05326634961232344e-01,
            3.60344899949804439e-01,
            1.25781726111229246e-01,
            1.60837851487422766e-02,
            6.58749161529837803e-04,
            1.63153871373020978e-02,
        ];
        const Q: [f64; 5] = [
            2.56852019228982242e+00,
            1.87295284992346047e+00,
            5.27905102951428412e-01,
            6.05183413124413191e-02,
            2.33520497626869185e-03,
        ];
        let y = 1.0 / (z * z);
        let mut num = P[5] * y;
        let mut den = y;
        for i in 0..4 {
            num = (num + P[i]) * y;
            den = (den + Q[i]) * y;
        }
        num = y * (num + P[4]) / (den + Q[4]);
        let sqrt_pi_inv = 0.564189583547756287; // 1/sqrt(pi)
        let num = (sqrt_pi_inv - num) / z;
        let trunc = (z * 16.0).floor() / 16.0;
        let del = (z - trunc) * (z + trunc);
        (-trunc * trunc).exp() * (-del).exp() * num
    };

    if x < 0.0 {
        2.0 - ans
    } else {
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndtri_known_values() {
        // ndtri(0.5) = 0 exactly (median).
        assert!(ndtri(0.5).abs() < 1e-12, "ndtri(0.5) = {}", ndtri(0.5));
        // ndtri(0.975) ≈ 1.959963985 (the 97.5th percentile / z for 95% CI).
        let z = ndtri(0.975);
        assert!(
            (z - 1.959963985).abs() < 1e-7,
            "ndtri(0.975) = {z}, expected ≈ 1.959964"
        );
        // ndtri(0.025) ≈ -1.959963985 (symmetric).
        let zl = ndtri(0.025);
        assert!(
            (zl + 1.959963985).abs() < 1e-7,
            "ndtri(0.025) = {zl}, expected ≈ -1.959964"
        );
        // ndtri(0.84134474606) ≈ 1.0 (one std above mean).
        let z1 = ndtri(0.8413447460685429);
        assert!((z1 - 1.0).abs() < 1e-7, "ndtri(0.8413..) = {z1}, expected ≈ 1");
    }

    #[test]
    fn ndtri_endpoints() {
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri(1.0), f64::INFINITY);
        assert!(ndtri(-0.1).is_nan());
        assert!(ndtri(1.1).is_nan());
    }

    #[test]
    fn ndtri_monotone() {
        // Strictly increasing on (0,1).
        let mut prev = ndtri(0.001);
        for k in 2..1000 {
            let p = k as f64 / 1000.0;
            let v = ndtri(p);
            assert!(v > prev, "ndtri not monotone at p={p}: {v} <= {prev}");
            prev = v;
        }
    }

    #[test]
    fn expit_known_values() {
        assert!((expit(0.0) - 0.5).abs() < 1e-15, "expit(0) = {}", expit(0.0));
        // expit(large) → 1, expit(-large) → 0, no overflow.
        assert!(expit(40.0) > 1.0 - 1e-15);
        assert!(expit(-40.0) < 1e-15);
        // ndtri/expit round-trip: expit is NOT the inverse of ndtri, but
        // expit(ndtri(p)) is the logistic-of-probit; just sanity that it's in
        // (0,1).
        let v = expit(ndtri(0.3));
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn schedule_clamp_bounds() {
        // t_min uses logsnr_max=18, t_max uses logsnr_min=-15.
        let t_min = 1.0 / (1.0 + (0.5 * 18.0_f64).exp());
        let t_max = 1.0 / (1.0 + (0.5 * -15.0_f64).exp());
        assert!(t_min < t_max, "t_min {t_min} should be < t_max {t_max}");
        let s = LogitNormalSchedule::new(0.0, 1.5);
        // At p→1, 1-expit(big) → small → clamps to t_min.
        let hi = s.eval(0.99999);
        // At p→0, 1-expit(very negative) → ~1 → clamps to t_max.
        let lo = s.eval(0.00001);
        assert!(
            (hi as f64) >= t_min - 1e-6 && (hi as f64) <= t_max + 1e-6,
            "eval(0.99999)={hi} out of [{t_min},{t_max}]"
        );
        assert!(
            (lo as f64) >= t_min - 1e-6 && (lo as f64) <= t_max + 1e-6,
            "eval(0.00001)={lo} out of [{t_min},{t_max}]"
        );
    }

    #[test]
    fn schedule_monotone_decreasing() {
        // 1 - expit(mean + std*ndtri(t)) is DECREASING in t (ndtri increasing,
        // expit increasing, 1-expit decreasing). Verify over the interior.
        let s = LogitNormalSchedule::new(0.5, 1.5);
        let mut prev = s.eval(0.01);
        for k in 2..100 {
            let t = k as f64 / 100.0;
            let v = s.eval(t);
            assert!(
                v <= prev + 1e-6,
                "schedule not non-increasing at t={t}: {v} > {prev}"
            );
            prev = v;
        }
    }

    #[test]
    fn resolution_mean_at_known_is_known() {
        // 512×512 → ln(1) = 0 → mean = known_mean.
        let s = get_schedule_for_resolution(512, 512, 0.0, 1.5);
        assert!(s.mean.abs() < 1e-12, "mean at 512² = {}", s.mean);
        assert_eq!(s.std, 1.5);
        // 1024×1024 → 4× pixels → mean = 0 + 0.5*ln(4) = ln(2) ≈ 0.6931.
        let s2 = get_schedule_for_resolution(1024, 1024, 0.0, 1.0);
        assert!(
            (s2.mean - std::f64::consts::LN_2).abs() < 1e-12,
            "mean at 1024² = {}, expected ln(2)",
            s2.mean
        );
    }

    #[test]
    fn step_intervals_linspace() {
        let iv = make_step_intervals(4);
        assert_eq!(iv.len(), 5); // n+1
        assert_eq!(iv[0], 0.0);
        assert_eq!(iv[4], 1.0);
        assert!((iv[1] - 0.25).abs() < 1e-6);
        assert!((iv[2] - 0.5).abs() < 1e-6);
        assert!((iv[3] - 0.75).abs() < 1e-6);
        // 48-step preset → 49 interval points.
        assert_eq!(make_step_intervals(48).len(), 49);
    }

    #[test]
    fn presets_match_python() {
        let q = preset("V4_QUALITY_48").unwrap();
        assert_eq!(q.num_steps, 48);
        assert_eq!(q.guidance_schedule.len(), 48);
        assert_eq!(&q.guidance_schedule[..3], &[3.0, 3.0, 3.0]);
        assert_eq!(q.guidance_schedule[3], 7.0);
        assert_eq!(q.guidance_schedule[47], 7.0);
        assert_eq!(q.mu, 0.0);
        assert_eq!(q.std, 1.5);
        q.validate().unwrap();

        let d = preset("V4_DEFAULT_20").unwrap();
        assert_eq!(d.num_steps, 20);
        assert_eq!(&d.guidance_schedule[..2], &[3.0, 3.0]);
        assert_eq!(d.guidance_schedule[2], 7.0);
        assert_eq!(d.mu, 0.0);
        assert_eq!(d.std, 1.75);
        d.validate().unwrap();

        let t = preset("V4_TURBO_12").unwrap();
        assert_eq!(t.num_steps, 12);
        assert_eq!(t.guidance_schedule[0], 3.0);
        assert_eq!(t.guidance_schedule[1], 7.0);
        assert_eq!(t.guidance_schedule[11], 7.0);
        assert_eq!(t.mu, 0.5);
        assert_eq!(t.std, 1.75);
        t.validate().unwrap();

        assert!(preset("nope").is_none());
    }

    #[test]
    fn preset_validate_rejects_length_mismatch() {
        let bad = SamplerParameters {
            num_steps: 5,
            guidance_schedule: vec![7.0; 4],
            mu: 0.0,
            std: 1.0,
        };
        assert!(bad.validate().is_err());
    }
}
