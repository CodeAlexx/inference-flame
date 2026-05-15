//! Oklab color encoder for AsymFLUX.2.
//!
//! Port of LakonLab's `OklabColorEncoder` (Apache 2.0, MIT-compatible).
//! Replaces a learned VAE with deterministic, invertible color-space math:
//! sRGB ↔ linear RGB ↔ LMS ↔ Oklab, with an affine normalize tuned for the
//! AsymFLUX.2 Klein 9B pixel-space target distribution.
//!
//! Reference: `~/LakonLab/lakonlab/models/architectures/autoencoders/color_encoders.py`
//! Reference config: `~/LakonLab/configs/asymflow/asymflux2_klein_test.py`
//!   `OklabColorEncoder(use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16)`
//!
//! DECISION: CPU f32 only for Step 1. Per-pixel work is trivially batched and
//! the entire encoder/decoder is allocation-free. GPU kernels land when the
//! sampling loop integrates this (Step 5+).
//!
//! DECISION: Input layout is **planar / channel-major** `(3, H*W)`, matching
//! NCHW slicing when N=1. Callers handle batching by feeding one image's
//! buffer at a time. This matches how the rest of inference-flame consumes
//! pixel data.
//!
//! DECISION: 3x3 matrix inverses are computed once at first use via
//! `OnceLock` instead of being hard-coded. Eliminates the maintenance bug
//! where a typed-in inverse silently drifts from its forward matrix.

use std::sync::OnceLock;

/// Linear-RGB → LMS color-space matrix (Björn Ottosson, Oklab paper).
pub const LRGB_TO_LMS: [[f32; 3]; 3] = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
];

/// LMS^(1/3) → Oklab matrix.
pub const LMS_TO_OKLAB: [[f32; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

/// Affine normalization (AsymFLUX.2 Klein 9B target distribution).
pub const AFFINE_MEAN: [f32; 3] = [0.56, 0.0, 0.01];
pub const AFFINE_STD: f32 = 0.16;

fn invert_3x3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let [a, b, c] = m[0];
    let [d, e, f] = m[1];
    let [g, h, i] = m[2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;
    [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

/// Oklab → LMS^(1/3). Lazily computed inverse of `LMS_TO_OKLAB`.
pub fn oklab_to_lms() -> &'static [[f32; 3]; 3] {
    static M: OnceLock<[[f32; 3]; 3]> = OnceLock::new();
    M.get_or_init(|| invert_3x3(LMS_TO_OKLAB))
}

/// LMS → linear RGB. Lazily computed inverse of `LRGB_TO_LMS`.
pub fn lms_to_lrgb() -> &'static [[f32; 3]; 3] {
    static M: OnceLock<[[f32; 3]; 3]> = OnceLock::new();
    M.get_or_init(|| invert_3x3(LRGB_TO_LMS))
}

#[inline]
fn srgb_to_lrgb_one(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn lrgb_to_srgb_one(l: f32) -> f32 {
    // Matches reference `lrgb_to_srgb`: clamp negatives only. The decode path
    // already clamps to [0, 1] before calling us; this guard makes the helper
    // safe to call standalone. NaN propagates (matches PyTorch reference).
    let l = clamp_lo_nan(l, 0.0);
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// `max(x, lo)` that preserves NaN. `f32::max` follows libm `fmaxf`, which
/// returns the non-NaN operand — silently squashing NaN inputs. The PyTorch
/// reference uses `.clamp(min=...)` which propagates NaN; we mirror that so
/// upstream bugs surface instead of being silently zeroed.
#[inline]
fn clamp_lo_nan(x: f32, lo: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x < lo {
        lo
    } else {
        x
    }
}

/// `clamp(x, lo, hi)` that preserves NaN. See `clamp_lo_nan` rationale.
#[inline]
fn clamp_nan(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[inline]
fn matvec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Encode an image from `[-1, 1]` sRGB pixels to normalized Oklab.
///
/// Layout: `pixels` is `(3, H*W)` planar (channel-major). `out` has the same
/// shape. Both must have length divisible by 3 and be equal in length.
pub fn encode_planar(pixels: &[f32], out: &mut [f32]) {
    assert_eq!(
        pixels.len(),
        out.len(),
        "encode_planar: input/output length mismatch ({} vs {})",
        pixels.len(),
        out.len()
    );
    assert!(
        pixels.len() % 3 == 0,
        "encode_planar: buffer length {} is not divisible by 3",
        pixels.len()
    );
    let n = pixels.len() / 3;
    let (r_in, gb) = pixels.split_at(n);
    let (g_in, b_in) = gb.split_at(n);
    let (r_out, gb_out) = out.split_at_mut(n);
    let (g_out, b_out) = gb_out.split_at_mut(n);
    for p in 0..n {
        // [-1, 1] sRGB → [0, 1] sRGB
        let rs = r_in[p] * 0.5 + 0.5;
        let gs = g_in[p] * 0.5 + 0.5;
        let bs = b_in[p] * 0.5 + 0.5;
        // sRGB → linear RGB
        let lr = srgb_to_lrgb_one(rs);
        let lg = srgb_to_lrgb_one(gs);
        let lb = srgb_to_lrgb_one(bs);
        // linear RGB → LMS, then clamp ≥ 0 (matches reference). NaN-preserving.
        let lms = matvec(&LRGB_TO_LMS, [lr, lg, lb]);
        let lms = [
            clamp_lo_nan(lms[0], 0.0),
            clamp_lo_nan(lms[1], 0.0),
            clamp_lo_nan(lms[2], 0.0),
        ];
        // LMS → LMS^(1/3)
        let lms_cbrt = [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()];
        // LMS^(1/3) → Oklab
        let ok = matvec(&LMS_TO_OKLAB, lms_cbrt);
        // Affine normalize
        r_out[p] = (ok[0] - AFFINE_MEAN[0]) / AFFINE_STD;
        g_out[p] = (ok[1] - AFFINE_MEAN[1]) / AFFINE_STD;
        b_out[p] = (ok[2] - AFFINE_MEAN[2]) / AFFINE_STD;
    }
}

/// Decode a normalized Oklab image back to `[-1, 1]` sRGB pixels.
///
/// Layout: `oklab` is `(3, H*W)` planar (channel-major). `out` has the same
/// shape.
pub fn decode_planar(oklab: &[f32], out: &mut [f32]) {
    assert_eq!(
        oklab.len(),
        out.len(),
        "decode_planar: input/output length mismatch ({} vs {})",
        oklab.len(),
        out.len()
    );
    assert!(
        oklab.len() % 3 == 0,
        "decode_planar: buffer length {} is not divisible by 3",
        oklab.len()
    );
    let n = oklab.len() / 3;
    let (l_in, ab) = oklab.split_at(n);
    let (a_in, b_in) = ab.split_at(n);
    let (r_out, gb_out) = out.split_at_mut(n);
    let (g_out, b_out) = gb_out.split_at_mut(n);
    let m_oklab_to_lms = oklab_to_lms();
    let m_lms_to_lrgb = lms_to_lrgb();
    for p in 0..n {
        // Affine denormalize
        let l = l_in[p] * AFFINE_STD + AFFINE_MEAN[0];
        let a = a_in[p] * AFFINE_STD + AFFINE_MEAN[1];
        let bb = b_in[p] * AFFINE_STD + AFFINE_MEAN[2];
        // Oklab → LMS^(1/3) → LMS (cube)
        let lms_cbrt = matvec(m_oklab_to_lms, [l, a, bb]);
        let lms = [
            lms_cbrt[0] * lms_cbrt[0] * lms_cbrt[0],
            lms_cbrt[1] * lms_cbrt[1] * lms_cbrt[1],
            lms_cbrt[2] * lms_cbrt[2] * lms_cbrt[2],
        ];
        // LMS → linear RGB, clamp to [0, 1] (NaN-preserving)
        let lrgb = matvec(m_lms_to_lrgb, lms);
        let lr = clamp_nan(lrgb[0], 0.0, 1.0);
        let lg = clamp_nan(lrgb[1], 0.0, 1.0);
        let lb = clamp_nan(lrgb[2], 0.0, 1.0);
        // linear RGB → sRGB
        let rs = lrgb_to_srgb_one(lr);
        let gs = lrgb_to_srgb_one(lg);
        let bs = lrgb_to_srgb_one(lb);
        // [0, 1] → [-1, 1]
        r_out[p] = rs * 2.0 - 1.0;
        g_out[p] = gs * 2.0 - 1.0;
        b_out[p] = bs * 2.0 - 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn matmul_3x3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        out
    }

    fn assert_identity(m: &[[f32; 3]; 3], tol: f32) {
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(m[i][j], expected, tol),
                    "[{i}][{j}] = {} (expected {expected}, tol {tol})",
                    m[i][j]
                );
            }
        }
    }

    #[test]
    fn lrgb_to_lms_inverse_is_correct() {
        let m = matmul_3x3(&LRGB_TO_LMS, lms_to_lrgb());
        assert_identity(&m, 1e-5);
    }

    #[test]
    fn lms_to_oklab_inverse_is_correct() {
        let m = matmul_3x3(&LMS_TO_OKLAB, oklab_to_lms());
        assert_identity(&m, 1e-5);
    }

    #[test]
    fn srgb_lrgb_roundtrip() {
        for i in 0..=100 {
            let s = i as f32 / 100.0;
            let round = lrgb_to_srgb_one(srgb_to_lrgb_one(s));
            assert!(
                approx_eq(round, s, 1e-5),
                "sRGB roundtrip drifted at s={s}: got {round}"
            );
        }
    }

    #[test]
    fn srgb_lrgb_at_piecewise_boundary() {
        // Lower path: s = 0.04045 → l = 0.04045 / 12.92 = 0.00313085...
        let s = 0.04045f32;
        let l = srgb_to_lrgb_one(s);
        // Upper path at same s: ((0.04045 + 0.055) / 1.055)^2.4
        let upper = ((s + 0.055) / 1.055).powf(2.4);
        assert!(
            approx_eq(l, upper, 1e-4),
            "sRGB piecewise discontinuous at boundary: lower={l} upper={upper}"
        );

        // Reverse: l = 0.0031308 maps via `l * 12.92` to ≈ 0.04045
        let l = 0.0031308f32;
        let s = lrgb_to_srgb_one(l);
        let upper = 1.055 * l.powf(1.0 / 2.4) - 0.055;
        assert!(
            approx_eq(s, upper, 1e-4),
            "lRGB piecewise discontinuous at boundary: lower={s} upper={upper}"
        );
    }

    #[test]
    fn neutral_gray_roundtrip() {
        // Mid-gray in [-1, 1] sRGB is 0.0 (i.e., 0.5 in [0, 1]).
        let pixels = vec![0.0f32; 3];
        let mut oklab = vec![0.0f32; 3];
        let mut decoded = vec![0.0f32; 3];
        encode_planar(&pixels, &mut oklab);
        decode_planar(&oklab, &mut decoded);
        for (a, b) in pixels.iter().zip(decoded.iter()) {
            assert!(approx_eq(*a, *b, 1e-4), "mid-gray drifted: {a} -> {b}");
        }
    }

    #[test]
    fn primary_colors_roundtrip() {
        // Three pixels: pure red, pure green, pure blue in [-1, 1] (i.e. +1.0
        // on one channel, -1.0 on the others).
        // Layout planar (3, 3): R=[+1,-1,-1], G=[-1,+1,-1], B=[-1,-1,+1]
        let pixels = vec![
            1.0, -1.0, -1.0, // R channel
            -1.0, 1.0, -1.0, // G channel
            -1.0, -1.0, 1.0, // B channel
        ];
        let mut oklab = vec![0.0f32; 9];
        let mut decoded = vec![0.0f32; 9];
        encode_planar(&pixels, &mut oklab);
        decode_planar(&oklab, &mut decoded);
        for (i, (a, b)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 5e-4),
                "primary color drift at idx {i}: {a} -> {b}"
            );
        }
    }

    #[test]
    fn black_and_white_roundtrip() {
        // Pure black (-1, -1, -1) and pure white (+1, +1, +1) as two pixels.
        let pixels = vec![
            -1.0, 1.0, // R
            -1.0, 1.0, // G
            -1.0, 1.0, // B
        ];
        let mut oklab = vec![0.0f32; 6];
        let mut decoded = vec![0.0f32; 6];
        encode_planar(&pixels, &mut oklab);
        decode_planar(&oklab, &mut decoded);
        for (i, (a, b)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "black/white drift at idx {i}: {a} -> {b}"
            );
        }
    }

    #[test]
    fn random_image_roundtrip_within_tol() {
        // Deterministic pseudo-random (Lehmer LCG) so the test never flakes.
        let n_pixels = 256;
        let mut pixels = vec![0.0f32; 3 * n_pixels];
        let mut state: u32 = 0xC0FFEE;
        for p in pixels.iter_mut() {
            state = state.wrapping_mul(48271) % 0x7fffffff;
            let u = state as f32 / 0x7fffffff as f32; // [0, 1)
            *p = u * 2.0 - 1.0; // [-1, 1)
        }
        let mut oklab = vec![0.0f32; pixels.len()];
        let mut decoded = vec![0.0f32; pixels.len()];
        encode_planar(&pixels, &mut oklab);
        decode_planar(&oklab, &mut decoded);
        let mut max_err = 0.0f32;
        for (a, b) in pixels.iter().zip(decoded.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-3, "max roundtrip error {max_err} exceeded tol");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn encode_panics_on_length_mismatch() {
        let pixels = vec![0.0f32; 6];
        let mut out = vec![0.0f32; 9];
        encode_planar(&pixels, &mut out);
    }

    #[test]
    #[should_panic(expected = "not divisible by 3")]
    fn encode_panics_on_bad_length() {
        let pixels = vec![0.0f32; 7];
        let mut out = vec![0.0f32; 7];
        encode_planar(&pixels, &mut out);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn decode_panics_on_length_mismatch() {
        let oklab = vec![0.0f32; 6];
        let mut out = vec![0.0f32; 9];
        decode_planar(&oklab, &mut out);
    }

    #[test]
    #[should_panic(expected = "not divisible by 3")]
    fn decode_panics_on_bad_length() {
        let oklab = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        decode_planar(&oklab, &mut out);
    }

    #[test]
    fn empty_input_is_a_noop() {
        let pixels: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        encode_planar(&pixels, &mut out);
        decode_planar(&pixels, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn single_pixel_roundtrip() {
        // Minimal non-empty case: one pixel, planar (3,1).
        let pixels = vec![0.42_f32, -0.17, 0.91];
        let mut oklab = vec![0.0_f32; 3];
        let mut decoded = vec![0.0_f32; 3];
        encode_planar(&pixels, &mut oklab);
        decode_planar(&oklab, &mut decoded);
        for (i, (a, b)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "single-pixel ch {i} drift: {a} -> {b}"
            );
        }
    }

    #[test]
    fn nan_propagates_without_panic() {
        // Documents that NaN inputs flow through to NaN outputs (no implicit
        // sanitization). Callers are contractually responsible for finite
        // inputs in [-1, 1]; this test pins down current behavior so a future
        // "let me add a guard" change is intentional, not silent.
        let pixels = vec![f32::NAN, 0.0, 0.0];
        let mut oklab = vec![0.0_f32; 3];
        encode_planar(&pixels, &mut oklab);
        assert!(oklab[0].is_nan(), "encode swallowed NaN: {}", oklab[0]);

        let pixels = vec![f32::NAN, 0.0, 0.0];
        let mut out = vec![0.0_f32; 3];
        decode_planar(&pixels, &mut out);
        assert!(out[0].is_nan(), "decode swallowed NaN: {}", out[0]);
    }

    #[test]
    fn encode_decode_encode_is_stable() {
        // For an in-gamut sample, going encode -> decode -> encode should
        // produce the same Oklab buffer to f32 ulps (within 5e-4 to absorb
        // sRGB transfer drift). Locks down idempotence of the round trip.
        let pixels = vec![0.3_f32, -0.6, 0.1, -0.2, 0.8, -0.4]; // 2 pixels
        let mut ok1 = vec![0.0_f32; 6];
        let mut pix2 = vec![0.0_f32; 6];
        let mut ok2 = vec![0.0_f32; 6];
        encode_planar(&pixels, &mut ok1);
        decode_planar(&ok1, &mut pix2);
        encode_planar(&pix2, &mut ok2);
        for (i, (a, b)) in ok1.iter().zip(ok2.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 5e-4),
                "Oklab drift after second encode at idx {i}: {a} -> {b}"
            );
        }
    }

    #[test]
    fn out_of_range_input_stays_finite() {
        // Inputs outside [-1, 1] aren't clamped by the encoder. Pin down that
        // the encoder produces finite output rather than NaN/Inf, even if the
        // result is meaningless. Catches accidental gamma overflow.
        let pixels = vec![5.0_f32, -3.0, 2.5];
        let mut out = vec![0.0_f32; 3];
        encode_planar(&pixels, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "out-of-range channel {i} produced non-finite: {v}"
            );
        }
    }

    #[test]
    fn inf_propagates_without_panic() {
        // Companion to nan_propagates_without_panic: Inf inputs flow through
        // to non-finite outputs without panic. Locks behavior.
        let pixels = vec![f32::INFINITY, 0.0, 0.0];
        let mut out = vec![0.0_f32; 3];
        encode_planar(&pixels, &mut out);
        assert!(
            !out[0].is_finite(),
            "encode silently finitized Inf input: {}",
            out[0]
        );

        let oklab = vec![f32::INFINITY, 0.0, 0.0];
        let mut out = vec![0.0_f32; 3];
        decode_planar(&oklab, &mut out);
        // Decode of Inf can produce NaN (via Inf - Inf in the matmul cascade)
        // or non-finite values. The contract is: no panic, and the result is
        // visibly bad (non-finite OR out-of-[-1,1]) rather than silently
        // finitized to a plausible color. Catches a future "let me sanitize
        // Inf to 0" change that would hide upstream bugs.
        let visibly_bad = !out[0].is_finite() || !(-1.0..=1.0).contains(&out[0]);
        assert!(
            visibly_bad,
            "decode silently finitized Inf input to {} (looks like a valid color)",
            out[0]
        );
    }

    #[test]
    fn extreme_negative_oklab_clamps_to_near_black() {
        // Wildly out-of-gamut normalized Oklab (very dark) should decode to
        // something in the [-1, +1] range — clamp prevents Inf/NaN.
        let oklab = vec![-10.0_f32, 0.0, 0.0];
        let mut out = vec![0.0_f32; 3];
        decode_planar(&oklab, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(v),
                "ch {i} out of [-1,1] after extreme negative decode: {v}"
            );
        }
        // L=-10 → de-norm Oklab[0] ≈ -1.04 → cube root path → very dark.
        // Expect channel 0 (R) near -1.
        assert!(out[0] < -0.5, "extreme negative L did not darken: {}", out[0]);
    }

    #[test]
    fn extreme_positive_oklab_clamps_within_range() {
        // Wildly out-of-gamut normalized Oklab (very bright) should decode to
        // something in [-1, +1] thanks to lrgb clamp. We don't assert
        // "near +1" since extreme a/b can saturate one channel only.
        let oklab = vec![10.0_f32, 0.0, 0.0];
        let mut out = vec![0.0_f32; 3];
        decode_planar(&oklab, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(v) && v.is_finite(),
                "ch {i} out of bounds after extreme positive decode: {v}"
            );
        }
    }

    #[test]
    fn per_channel_independence() {
        // Two pixels: each varies one channel only. The decoded output for
        // pixel 0 should not depend on pixel 1's values (and vice versa).
        // Catches any accidental cross-pixel write/read leak.
        let pixels_a = vec![0.3_f32, 0.0, -0.2, 0.0, 0.4, 0.0];
        let pixels_b = vec![0.3_f32, 0.9, -0.2, 0.0, 0.4, -0.7];
        let mut ok_a = vec![0.0_f32; 6];
        let mut ok_b = vec![0.0_f32; 6];
        let mut dec_a = vec![0.0_f32; 6];
        let mut dec_b = vec![0.0_f32; 6];
        encode_planar(&pixels_a, &mut ok_a);
        encode_planar(&pixels_b, &mut ok_b);
        decode_planar(&ok_a, &mut dec_a);
        decode_planar(&ok_b, &mut dec_b);

        // Pixel 0 (index 0, n, 2n) — under planar layout n=2, so indices 0, 2, 4.
        let p0_a = [dec_a[0], dec_a[2], dec_a[4]];
        let p0_b = [dec_b[0], dec_b[2], dec_b[4]];
        for i in 0..3 {
            assert!(
                approx_eq(p0_a[i], p0_b[i], 1e-5),
                "pixel-0 ch{i} drifted between buffers: {} vs {}",
                p0_a[i],
                p0_b[i]
            );
        }
    }

    #[test]
    fn affine_norm_mean_pixel_lands_near_origin() {
        // The dataset mean (sRGB grey ~0.56 in oklab L) should map close to
        // origin in normalized Oklab after the affine.
        // Specifically: feed a single pixel whose Oklab is exactly AFFINE_MEAN
        // by going Oklab → LMS → lRGB → sRGB → encode, then check normalized
        // Oklab ≈ 0.
        let m_oklab_to_lms = oklab_to_lms();
        let m_lms_to_lrgb = lms_to_lrgb();
        let lms_cbrt = matvec(m_oklab_to_lms, AFFINE_MEAN);
        let lms = [
            lms_cbrt[0].powi(3),
            lms_cbrt[1].powi(3),
            lms_cbrt[2].powi(3),
        ];
        let lrgb = matvec(m_lms_to_lrgb, lms);
        let srgb_01 = [
            lrgb_to_srgb_one(lrgb[0].clamp(0.0, 1.0)),
            lrgb_to_srgb_one(lrgb[1].clamp(0.0, 1.0)),
            lrgb_to_srgb_one(lrgb[2].clamp(0.0, 1.0)),
        ];
        let pixels = vec![
            srgb_01[0] * 2.0 - 1.0,
            srgb_01[1] * 2.0 - 1.0,
            srgb_01[2] * 2.0 - 1.0,
        ];
        let mut out = vec![0.0f32; 3];
        encode_planar(&pixels, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.abs() < 5e-3,
                "Oklab channel {i} at affine mean = {v} (expected ≈ 0)"
            );
        }
    }
}
