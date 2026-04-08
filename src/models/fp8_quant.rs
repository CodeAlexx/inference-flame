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
//! Rust mirrors this by doing a TRUE BF16 → FP8 bytes → BF16 round-trip:
//!   - CPU: encode each BF16 element as an FP8 e4m3fn byte (round-half-to-even)
//!   - GPU: dequant via the existing `flame_core::ops::fused_inference::
//!     dequant_fp8_to_bf16` kernel (the same kernel used by FlameSwap to
//!     decode FP8-resident weights, so the decode is bit-identical to the
//!     production path)
//!
//! Forward path is unchanged (linear3d still operates on BF16) but the
//! weight values are now FP8-quantized.
//!
//! Python's downcast list (TRANSFORMER_LINEAR_DOWNCAST_MAP):
//!   - .to_q.weight, .to_q.bias
//!   - .to_k.weight, .to_k.bias
//!   - .to_v.weight, .to_v.bias
//!   - .to_out.0.weight, .to_out.0.bias
//!   - ff.net.0.proj.weight, ff.net.0.proj.bias
//!   - ff.net.2.weight, ff.net.2.bias
//!
//! All under the `transformer_blocks.` prefix. Includes ALL six attention
//! modules per block (attn1, attn2, audio_attn1, audio_attn2,
//! audio_to_video_attn, video_to_audio_attn) AND both FFNs (ff, audio_ff).
//! Norms and gate_logits are NOT in the list (they stay BF16).
//!
//! Activated by env var `FLAME_SIMULATE_FP8=1`. Default behavior is no
//! quantization (matches the previous BF16-only Rust path).

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

/// Check if FP8 simulation is requested via env var. Read once and memoized.
pub fn fp8_simulation_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FLAME_SIMULATE_FP8").is_ok())
}

/// BF16 (as u16 bits) → f32. Lossless: shift left by 16 and reinterpret.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → FP8 e4m3fn byte encoding.
///
/// Matches PyTorch's `tensor.to(torch.float8_e4m3fn)` behavior:
///   - Round-to-nearest, ties-to-even on the mantissa
///   - Saturate to ±448 on overflow (no infinity)
///   - NaN encoded as 0xFF (sign-bit can be 0 or 1)
///   - ±0 encoded as 0x00 / 0x80
///
/// e4m3fn format:
///   - 1 sign bit
///   - 4 exponent bits, bias = 7
///   - 3 mantissa bits
///   - Max representable: 1.75 × 2^8 = 448
///   - Min normal:        2^-6  = 1/64
///   - Min subnormal:     2^-9  = 1/512
#[inline]
pub fn f32_to_fp8_e4m3fn_byte(x: f32) -> u8 {
    if x.is_nan() {
        // PyTorch encodes NaN as 0xFF (positive sign + max exp + max mant);
        // some implementations use 0x7F. Either is valid e4m3fn NaN.
        return 0xFF;
    }
    let bits = x.to_bits();
    let sign_bit = ((bits >> 31) & 1) as u8;
    let f32_exp_biased = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7F_FFFF;

    // ±0 (and any f32 subnormal flushes to fp8 zero)
    if f32_exp_biased == 0 {
        return sign_bit << 7;
    }
    // ±inf saturates to ±448
    if f32_exp_biased == 0xFF {
        return (sign_bit << 7) | 0x7E;
    }

    let true_exp = f32_exp_biased - 127;

    // Saturate above fp8 max (1.75 × 2^8 = 448)
    if true_exp > 8 {
        return (sign_bit << 7) | 0x7E;
    }

    // Subnormal range: true_exp <= -7 → fp8 biased exp 0
    //
    // The fp8 e4m3 subnormal grid is k * 2^-9 for k in {0..7}. Any value
    // with abs(x) > 2^-10 (the midpoint between 0 and the smallest non-zero
    // subnormal) should round UP to k=1, not down to 0. The shift formula
    // handles this for all true_exp; we just need to NOT early-bail.
    if true_exp < -6 {
        let mant_with_hidden = (1u32 << 23) | f32_mant;
        // We want k = round((1 + f32_mant/2^23) * 2^(true_exp + 9))
        //            = round(mant_with_hidden * 2^(true_exp + 9 - 23))
        //            = round(mant_with_hidden / 2^(14 - true_exp))
        // For true_exp = -100, the shift is 114 → result is always 0.
        // For true_exp = -10, shift = 24 → in-range bit shift, formula works.
        let total_shift = (14 - true_exp) as u32;
        let k = if total_shift >= 32 {
            // Shift would zero out everything except possibly the rounding
            // bias if the value is exactly at the half-grid boundary; even
            // then, round-to-even gives 0.
            0u32
        } else {
            let lsb = (mant_with_hidden >> total_shift) & 1;
            let half = 1u32 << (total_shift - 1);
            let rounding_bias = half - 1 + lsb;
            // Use saturating add to avoid u32 overflow when bias + mantissa
            // would exceed u32::MAX (only possible at very deep shifts).
            mant_with_hidden.saturating_add(rounding_bias) >> total_shift
        };
        // k can roll over to 8 → becomes the smallest normal (mant=0, exp=1)
        if k >= 8 {
            return (sign_bit << 7) | (1u8 << 3);
        }
        if k == 0 {
            return sign_bit << 7;
        }
        return (sign_bit << 7) | (k as u8);
    }

    // Normal range: true_exp in [-6, 8]
    let biased_exp = (true_exp + 7) as u32; // 1..15

    // Round f32 mantissa (23 bits) to 3 bits, round-half-to-even.
    let lsb = (f32_mant >> 20) & 1;
    let half = 1u32 << 19;
    let rounding_bias = half - 1 + lsb;
    let mut rounded_mant = (f32_mant + rounding_bias) >> 20;
    let mut final_exp = biased_exp;

    // Mantissa overflow (0b111 + 1 = 0b1000): increment exp, reset mant to 0
    if rounded_mant == 8 {
        rounded_mant = 0;
        final_exp += 1;
    }

    // Re-check saturation:
    //   biased_exp = 15 (true_exp = 8) with mant >= 7 would be NaN slot 0x7F.
    //   Saturate to 0x7E (mant = 6 → value 448).
    if final_exp >= 15 && rounded_mant >= 7 {
        return (sign_bit << 7) | 0x7E;
    }
    if final_exp > 15 {
        return (sign_bit << 7) | 0x7E;
    }

    (sign_bit << 7) | ((final_exp as u8) << 3) | (rounded_mant as u8)
}

/// True BF16 → FP8 e4m3fn → BF16 round-trip via the GPU dequant kernel.
///
/// Steps:
///   1. Read tensor's BF16 bits to CPU
///   2. CPU: encode each element as FP8 byte (matching PyTorch's cast)
///   3. Upload FP8 bytes to GPU
///   4. Use the production `dequant_fp8_to_bf16` kernel to decode (same
///      kernel that decodes FP8-resident weights, so bit-identical to
///      what the model would see if loaded from an FP8 checkpoint).
pub fn quantize_bf16_to_fp8_round_trip(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 {
        return Err(flame_core::Error::InvalidInput(format!(
            "fp8_quant: expected BF16 tensor, got {:?}",
            tensor.dtype()
        )));
    }

    // Step 1+2: BF16 → f32 → FP8 byte on CPU
    let bf16_data = tensor.to_vec_bf16()?;
    let mut fp8_bytes: Vec<u8> = Vec::with_capacity(bf16_data.len());
    for &bits in bf16_data.iter() {
        let f = bf16_to_f32(bits);
        fp8_bytes.push(f32_to_fp8_e4m3fn_byte(f));
    }

    // Step 3: upload to GPU as CudaSlice<u8>
    let device = tensor.device().clone();
    let fp8_slice = device
        .htod_sync_copy(&fp8_bytes)
        .map_err(|e| flame_core::Error::Cuda(format!("fp8 htod: {e}")))?;

    // Step 4: GPU dequant to BF16 (scale=1.0, no scaling)
    let shape = Shape::from_dims(tensor.shape().dims());
    flame_core::ops::fused_inference::dequant_fp8_to_bf16(&fp8_slice, 1.0, shape, &device)
}

/// Walk a `block_weights` HashMap and replace any FP8-cast tensors with
/// their FP8-quantized round-trip. Mirrors Python's `fp8_cast()` policy.
///
/// No-op if `FLAME_SIMULATE_FP8` is not set.
pub fn quantize_block_weights_inplace(
    block_weights: &mut HashMap<String, Tensor>,
    block_idx: usize,
) -> Result<()> {
    if !fp8_simulation_enabled() {
        return Ok(());
    }
    let keys_to_quantize: Vec<String> = block_weights
        .keys()
        .filter(|k| is_fp8_cast_key(k))
        .cloned()
        .collect();

    let n = keys_to_quantize.len();
    for key in keys_to_quantize {
        if let Some(tensor) = block_weights.get(&key) {
            if tensor.dtype() != DType::BF16 {
                continue;
            }
            let quantized = quantize_bf16_to_fp8_round_trip(tensor)?;
            block_weights.insert(key, quantized);
        }
    }
    if n > 0 {
        log::info!("[simulate_fp8] Round-tripped {n} tensors for block {block_idx}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_byte_zero() {
        assert_eq!(f32_to_fp8_e4m3fn_byte(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3fn_byte(-0.0), 0x80);
    }

    #[test]
    fn fp8_byte_saturation() {
        // 1000 saturates to +max = 0x7E
        assert_eq!(f32_to_fp8_e4m3fn_byte(1000.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3fn_byte(-1000.0), 0xFE);
        // 448 = 0x7E exactly
        assert_eq!(f32_to_fp8_e4m3fn_byte(448.0), 0x7E);
        // 449 still saturates to 0x7E
        assert_eq!(f32_to_fp8_e4m3fn_byte(449.0), 0x7E);
    }

    #[test]
    fn fp8_byte_min_normal() {
        // 2^-6 = 0.015625 → biased_exp=1, mant=0 → 0x08
        assert_eq!(f32_to_fp8_e4m3fn_byte(0.015625), 0x08);
    }

    #[test]
    fn fp8_byte_one() {
        // 1.0 = 1.0 × 2^0 → biased_exp=7, mant=0 → 0x38
        assert_eq!(f32_to_fp8_e4m3fn_byte(1.0), 0x38);
        assert_eq!(f32_to_fp8_e4m3fn_byte(-1.0), 0xB8);
    }

    #[test]
    fn fp8_byte_subnormal() {
        // 2^-9 ≈ 0.001953 → smallest subnormal = 0x01
        assert_eq!(f32_to_fp8_e4m3fn_byte(0.001953125), 0x01);
        // 2^-7 = 0.0078125 → 4/8 * 2^-6 = 0x04
        assert_eq!(f32_to_fp8_e4m3fn_byte(0.0078125), 0x04);
    }

    #[test]
    fn is_fp8_cast_matches() {
        assert!(is_fp8_cast_key("transformer_blocks.0.attn1.to_q.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.5.audio_attn1.to_k.bias"));
        assert!(is_fp8_cast_key("transformer_blocks.10.video_to_audio_attn.to_v.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.0.audio_to_video_attn.to_out.0.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.0.ff.net.0.proj.weight"));
        assert!(is_fp8_cast_key("transformer_blocks.0.audio_ff.net.2.bias"));

        assert!(!is_fp8_cast_key("transformer_blocks.0.norm1.weight"));
        assert!(!is_fp8_cast_key("transformer_blocks.0.attn1.q_norm.weight"));
        assert!(!is_fp8_cast_key("transformer_blocks.0.attn1.to_gate_logits.weight"));
        assert!(!is_fp8_cast_key("transformer_blocks.0.scale_shift_table"));
    }
}
