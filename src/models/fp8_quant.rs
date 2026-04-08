//! BF16 → FP8 (e4m3fn) → BF16 quantization round-trip.
//!
//! Mirrors Python's `QuantizationPolicy.fp8_cast()` from
//! `ltx_core.quantization.fp8_cast`. The Python policy:
//!   1. At weight load time, casts specific Linear weight/bias tensors
//!      from BF16 → torch.float8_e4m3fn (lossy ~6-bit precision per element).
//!   2. At forward time, upcasts FP8 → BF16 in `Fp8CastLinear.forward`.
//!
//! Net effect: the model behaves as if its weights were stored in FP8.
//! The LTX-2.3 distilled checkpoint was distilled with this quantization
//! in mind — using full-precision BF16 weights produces systematically
//! larger activations (the "audio 2.4× too hot" symptom that motivated
//! this module).
//!
//! Rust mirrors this by doing a BF16 → FP8 → BF16 round-trip on the
//! same set of weights at load time. The forward path is unchanged
//! (linear3d still operates on BF16) but the weight values are now
//! FP8-quantized.
//!
//! Python's downcast list (TRANSFORMER_LINEAR_DOWNCAST_MAP):
//!   - .to_q.weight, .to_q.bias
//!   - .to_k.weight, .to_k.bias
//!   - .to_v.weight, .to_v.bias
//!   - .to_out.0.weight, .to_out.0.bias
//!   - ff.net.0.proj.weight, ff.net.0.proj.bias
//!   - ff.net.2.weight, ff.net.2.bias
//!
//! All under the `transformer_blocks.` prefix. This includes ALL six
//! attention modules per block (attn1, attn2, audio_attn1, audio_attn2,
//! audio_to_video_attn, video_to_audio_attn) AND both FFNs (ff, audio_ff).
//! Norms and gate_logits are NOT in the list (they stay BF16).

use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

/// Suffix list for keys that should be FP8-cast. Matches Python's
/// `TRANSFORMER_LINEAR_DOWNCAST_MAP` exactly.
const FP8_CAST_SUFFIXES: &[&str] = &[
    ".to_q.weight", ".to_q.bias",
    ".to_k.weight", ".to_k.bias",
    ".to_v.weight", ".to_v.bias",
    ".to_out.0.weight", ".to_out.0.bias",
    "ff.net.0.proj.weight", "ff.net.0.proj.bias",
    "ff.net.2.weight", "ff.net.2.bias",
];

/// Returns true if `key` (full or stripped of `transformer_blocks.{i}.` prefix)
/// matches one of the FP8-cast suffixes.
pub fn is_fp8_cast_key(key: &str) -> bool {
    FP8_CAST_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// Quantize a single f32 to the nearest representable e4m3fn value, then
/// return as f32. Round-to-nearest, ties-to-even is approximated by simple
/// round-half-up via `(mant + half) >> shift` integer arithmetic.
///
/// e4m3fn format:
///   - 1 sign bit
///   - 4 exponent bits, bias = 7
///   - 3 mantissa bits
///   - No infinity. Only NaN encoding is 0xFF (sign=*, exp=0xF, mant=0x7).
///   - Max representable normal: 0x7E = +448 = 2^8 * (1 + 6/8)
///     (0x7F is the NaN encoding for the positive sign in fn variant)
///   - Min normal:    2^(-6) = 1/64
///   - Min subnormal: 2^(-9) = 1/512
#[inline]
pub fn fp8_e4m3_quantize_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == 0.0 {
        return 0.0_f32.copysign(x);
    }

    let bits = x.to_bits();
    let sign_bit = (bits >> 31) & 1;
    let f32_exp_biased = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7F_FFFF;

    // f32 zero or denormal → fp8 zero (denormal flushed)
    if f32_exp_biased == 0 {
        return 0.0_f32.copysign(x);
    }

    // f32 inf or NaN already handled (NaN above; for inf saturate)
    if f32_exp_biased == 0xFF {
        let max = 448.0_f32;
        return if sign_bit == 1 { -max } else { max };
    }

    let true_exp = f32_exp_biased - 127;

    // Saturate above fp8 max representable (448)
    if true_exp > 8 {
        let max = 448.0_f32;
        return if sign_bit == 1 { -max } else { max };
    }
    // 2^8 * (1 + 6/8) = 448 is the max representable. Anything in [448, +inf)
    // saturates to 448. Round half-to-max for the boundary.
    if true_exp == 8 {
        // f32 mantissa range for true_exp=8 is [0, 2^23-1].
        // fp8 mantissa is 3 bits → quantum at this exponent is 2^5 = 32.
        // mantissa = 7/8 → value = 256 * 1.875 = 480 (overflow, NaN slot)
        // mantissa = 6/8 → value = 256 * 1.75  = 448 (max representable)
        // Round mantissa to nearest 1/8 step.
        let mant_top3 = ((f32_mant + (1u32 << 19)) >> 20) & 0xF; // round to 4 bits then take top 3+overflow
        let final_mant = if mant_top3 >= 7 { 7 } else { mant_top3 } as u32;
        // If final_mant >= 7, saturate to 6 (NaN slot is 7).
        let final_mant = final_mant.min(6);
        let value = (1.0 + final_mant as f32 / 8.0) * 256.0_f32;
        return if sign_bit == 1 { -value } else { value };
    }

    // Subnormal range: true_exp in [-9, -7] (encoded exp = 0, varying mant)
    // Smallest subnormal = 2^-9, smallest normal = 2^-6
    if true_exp < -6 {
        if true_exp < -9 {
            return 0.0_f32.copysign(x);
        }
        // Subnormal: value = (mant/8) * 2^-6 with mant in [0, 7]
        // We need to round x to nearest k * 2^-9 where k in [0, 7]
        let step = 2f32.powi(-9);
        let abs_x = x.abs();
        let q = (abs_x / step).round();
        let q_clamped = q.min(7.0);
        let value = q_clamped * step;
        return if sign_bit == 1 { -value } else { value };
    }

    // Normal range: true_exp in [-6, 7]
    // fp8 mantissa is 3 bits → keep top 3 bits of f32 mantissa with rounding.
    // Round-to-nearest (ties to up — close enough to RTNE for our use).
    let mant_shift = 23 - 3; // 20
    let half = 1u32 << (mant_shift - 1);
    let rounded = (f32_mant + half) >> mant_shift;

    // Mantissa overflow: 0x7 + 1 = 0x8, increment exponent
    let (final_exp, final_mant) = if rounded == 8 {
        (true_exp + 1, 0u32)
    } else {
        (true_exp, rounded)
    };

    // Re-check saturation after potential exponent bump
    if final_exp > 8 || (final_exp == 8 && final_mant > 6) {
        let max = 448.0_f32;
        return if sign_bit == 1 { -max } else { max };
    }

    let value = (1.0 + final_mant as f32 / 8.0) * 2f32.powi(final_exp);
    if sign_bit == 1 { -value } else { value }
}

/// BF16 (as u16 bits) → f32. Just shifts left by 16 bits and reinterprets.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Quantize a BF16 tensor: BF16 → FP8 e4m3 → BF16 round-trip.
/// The result is a new BF16 tensor with values rounded to FP8 e4m3 grid.
///
/// Done on the CPU because there's no flame-core BF16 → FP8 GPU kernel
/// (only the reverse, FP8 → BF16). Bound: each weight is at most ~50 MB,
/// so the total CPU work for all blocks is ~10–30 s at load time.
pub fn quantize_bf16_to_fp8_round_trip(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 {
        return Err(flame_core::Error::InvalidInput(format!(
            "fp8_quant: expected BF16 tensor, got {:?}",
            tensor.dtype()
        )));
    }
    let bf16_data = tensor.to_vec_bf16()?;
    let mut quantized: Vec<f32> = Vec::with_capacity(bf16_data.len());
    for &bits in bf16_data.iter() {
        let f = bf16_to_f32(bits);
        quantized.push(fp8_e4m3_quantize_f32(f));
    }
    let shape = Shape::from_dims(tensor.shape().dims());
    Tensor::from_f32_to_bf16(quantized, shape, tensor.device().clone())
}

/// Walk a `block_weights` HashMap and replace any FP8-cast tensors with
/// their FP8-quantized round-trip. Mirrors Python's `fp8_cast()` policy.
///
/// Skipped silently if the tensor isn't BF16 (e.g. an F32 scale_shift_table
/// from the f32_cache — those are not in the FP8 cast list anyway).
pub fn quantize_block_weights_inplace(
    block_weights: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let keys_to_quantize: Vec<String> = block_weights
        .keys()
        .filter(|k| is_fp8_cast_key(k))
        .cloned()
        .collect();

    for key in keys_to_quantize {
        if let Some(tensor) = block_weights.get(&key) {
            if tensor.dtype() != DType::BF16 {
                continue;
            }
            let quantized = quantize_bf16_to_fp8_round_trip(tensor)?;
            block_weights.insert(key, quantized);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_quantize_zero() {
        assert_eq!(fp8_e4m3_quantize_f32(0.0), 0.0);
        assert_eq!(fp8_e4m3_quantize_f32(-0.0).is_sign_negative(), true);
    }

    #[test]
    fn fp8_quantize_saturation() {
        assert_eq!(fp8_e4m3_quantize_f32(1000.0), 448.0);
        assert_eq!(fp8_e4m3_quantize_f32(-1000.0), -448.0);
        assert_eq!(fp8_e4m3_quantize_f32(448.0), 448.0);
        assert_eq!(fp8_e4m3_quantize_f32(449.0), 448.0);
    }

    #[test]
    fn fp8_quantize_min_normal() {
        // 2^-6 = 0.015625 is the smallest normal
        assert_eq!(fp8_e4m3_quantize_f32(0.015625), 0.015625);
    }

    #[test]
    fn fp8_quantize_subnormal() {
        // 2^-9 ≈ 0.001953
        assert!((fp8_e4m3_quantize_f32(0.002) - 0.001953125).abs() < 1e-6);
    }

    #[test]
    fn is_fp8_cast_matches() {
        assert!(is_fp8_cast_key("transformer_blocks.0.attn1.to_q.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.5.audio_attn1.to_k.bias"));
        assert!(is_fp8_cast_key("transformer_blocks.10.video_to_audio_attn.to_v.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.0.audio_to_video_attn.to_out.0.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.0.ff.net.0.proj.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.0.audio_ff.net.2.bias"));

        // Should NOT match
        assert!(!is_fp8_cast_key("transformer_blocks.0.norm1.weight"));
        assert!(!is_fp8_cast_key("transformer_blocks.0.attn1.q_norm.weight"));
        assert!(!is_fp8_cast_key("transformer_blocks.0.attn1.to_gate_logits.weight"));
        assert!(!is_fp8_cast_key("transformer_blocks.0.scale_shift_table"));
    }
}
