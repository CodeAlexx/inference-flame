//! FeatureExtractorV2 for LTX-2 text encoding pipeline.
//!
//! Takes the 49 hidden states from Gemma-3 12B, applies per-token RMSNorm,
//! concatenates along the hidden dim (→ 188160), rescales, and projects
//! through aggregate_embed linear to the connector dimension (4096).
//!
//! ## Pipeline (from ltx_core/text_encoders/gemma/feature_extractor.py):
//! ```python
//! # FeatureExtractorV2.forward:
//! encoded = torch.stack(hidden_states, dim=-1)       # [B, T, D, L=49]
//! normed = norm_and_concat_per_token_rms(encoded, attention_mask)  # [B, T, D*L=188160]
//! normed = normed.to(encoded.dtype)
//! v_dim = self.video_aggregate_embed.out_features     # 4096
//! video = self.video_aggregate_embed(
//!     _rescale_norm(normed, v_dim, self.embedding_dim)  # rescale by sqrt(4096/188160)
//! )
//! ```
//!
//! ⚠️ This module is STANDALONE — it does NOT connect to any inference pipeline.

use flame_core::{DType, Result, Shape, Tensor};

/// Process Gemma-3 hidden states into features ready for the connector.
///
/// This implements:
/// 1. Per-token RMSNorm across hidden states
/// 2. Concatenation (→ 188160-dim)
/// 3. Rescale normalization
/// 4. Linear projection via aggregate_embed (→ 4096-dim)
///
/// ## Arguments
/// - `hidden_states`: Vec of 49 tensors, each [B, seq_len, 3840]
/// - `attention_mask`: [B, seq_len] binary mask (1=real, 0=pad)
/// - `aggregate_embed_weight`: [4096, 188160] or transposed
/// - `aggregate_embed_bias`: [4096] (optional)
/// - `target_dim`: output dim (4096)
///
/// ## Returns
/// [B, seq_len, 4096] BF16 tensor
pub fn feature_extract_and_project(
    hidden_states: &[Tensor],
    attention_mask: &Tensor,
    aggregate_embed_weight: &Tensor,
    aggregate_embed_bias: Option<&Tensor>,
    target_dim: usize,
) -> Result<Tensor> {
    let num_layers = hidden_states.len(); // 49
    let first = &hidden_states[0];
    let dims = first.shape().dims().to_vec();
    let b = dims[0];
    let t = dims[1];
    let d = dims[2]; // 3840
    let embedding_dim = d; // 3840

    log::info!("[FeatureExtractor] Input: {} hidden states, each [{}, {}, {}]", num_layers, b, t, d);

    // 1+2. Per-token RMSNorm and concatenation (fast — no 4D stack)
    //
    // Instead of stacking into [B,T,D,L=49] (huge allocation), we compute
    // variance incrementally across hidden states and normalize in-place.
    // This avoids materializing the 49-tensor stack.
    let normed = norm_and_concat_per_token_rms_fast(hidden_states, attention_mask, b, t, d)?;

    // 3. Rescale: normed * sqrt(target_dim / embedding_dim)
    //
    // ## PyTorch reference (_rescale_norm):
    // ```python
    // return x * math.sqrt(target_dim / source_dim)
    // ```
    let source_dim = embedding_dim; // 3840 (NOT 188160 — source_dim is the original embedding_dim)
    let rescale = ((target_dim as f64) / (source_dim as f64)).sqrt() as f32;
    let rescaled = normed.mul_scalar(rescale)?;

    log::info!("[FeatureExtractor] Rescaled: {:?}, factor={:.4}", rescaled.shape(), rescale);

    // 4. Linear projection: [B, T, 188160] → [B, T, 4096]
    //
    // Uses fused_linear3d (cuBLASLt) instead of generic matmul.
    // Generic matmul falls back to a broken path for k=188160 (~60s vs 5ms).
    let flat_dim = d * num_layers; // 188160
    let agg_w_dims = aggregate_embed_weight.shape().dims().to_vec();

    // Weight must be pre-transposed to [in, out] for fused_linear3d
    let weight_t = if agg_w_dims[0] == target_dim && agg_w_dims[1] == flat_dim {
        // [out, in] → transpose to [in, out]
        flame_core::bf16_elementwise::transpose2d_bf16(aggregate_embed_weight)?
    } else if agg_w_dims[0] == flat_dim && agg_w_dims[1] == target_dim {
        // Already transposed
        aggregate_embed_weight.clone()
    } else {
        return Err(flame_core::Error::InvalidInput(
            format!("Unexpected aggregate_embed weight shape: {:?}, expected [{}, {}] or [{}, {}]",
                agg_w_dims, target_dim, flat_dim, flat_dim, target_dim),
        ));
    };

    // fused_linear3d: [B, T, 188160] x [188160, 4096] + bias → [B, T, 4096]
    // This uses cuBLASLt strided batched GEMM — correct and fast for any K.
    let result = flame_core::ops::fused_inference::fused_linear3d(
        &rescaled,  // [B, T, 188160] — already 3D
        &weight_t,  // [188160, 4096] pre-transposed
        aggregate_embed_bias,
    )?;
    log::info!("[FeatureExtractor] Output: {:?}", result.shape());
    Ok(result)
}

/// Per-token RMSNorm + concatenation — fast version.
///
/// Avoids materializing the [B, T, D, 49] stack. Instead:
/// 1. Compute per-layer variance incrementally: sum(h_i^2) / D for each layer
/// 2. rsqrt(variance + eps) per layer
/// 3. Normalize each hidden state and concatenate into [B, T, D*L]
///
/// This replaces the 49-tensor stack + 48M-element elementwise ops with
/// 49 smaller operations on [B, T, D] tensors.
fn norm_and_concat_per_token_rms_fast(
    hidden_states: &[Tensor],
    attention_mask: &Tensor,
    b: usize,
    t: usize,
    d: usize,
) -> Result<Tensor> {
    let l = hidden_states.len(); // 49
    let flat_dim = d * l; // 188160

    // For each layer, compute variance = mean(h^2, dim=-1, keepdim=True) → [B, T, 1]
    // Then normalize: h_normed = h * rsqrt(var + eps)
    // Concatenate all normalized hidden states into [B, T, D*L]

    // Allocate output: [B*T, D*L] as BF16
    let device = hidden_states[0].device().clone();
    let mut output = Tensor::zeros_dtype(
        Shape::from_dims(&[b * t, flat_dim]), DType::BF16, device.clone(),
    )?;

    for (i, h) in hidden_states.iter().enumerate() {
        // h: [B, T, D] → [B*T, D]
        let h_2d = h.reshape(&[b * t, d])?;

        // variance = mean(h^2, dim=-1) → [B*T]
        let sq = h_2d.mul(&h_2d)?;
        let var = sq.mean_dim(&[1], false)?; // [B*T]

        // inv_std = rsqrt(var + eps) → [B*T]
        let inv_std = var.add_scalar(1e-6)?.rsqrt()?;

        // normed = h * inv_std → [B*T, D]
        let inv_std_bc = inv_std.unsqueeze(1)?.expand(&[b * t, d])?;
        let normed = h_2d.mul(&inv_std_bc)?;

        // Copy into output at offset i*D
        // output[:, i*D : (i+1)*D] = normed
        // Use narrow + copy since we can't do scatter
        // Actually, we need to build the output by parts. Let's collect and cat.
        // This is still cheaper than stacking all 49 into 4D.
        if i == 0 {
            // First: just store the slices for later cat
        }
        // Actually the simplest approach: collect normed slices, cat at the end
        drop(normed); // we'll redo this below
        drop(inv_std_bc);
        drop(inv_std);
        drop(var);
        drop(sq);
        drop(h_2d);
    }
    drop(output);

    // Simpler: normalize each layer, collect, cat along dim 2
    let mut normed_layers = Vec::with_capacity(l);
    for h in hidden_states {
        let h_2d = h.reshape(&[b * t, d])?;
        let sq = h_2d.mul(&h_2d)?;
        let var = sq.mean_dim(&[1], false)?; // [B*T]
        let inv_std = var.add_scalar(1e-6)?.rsqrt()?.unsqueeze(1)?; // [B*T, 1]
        let normed = h_2d.mul(&inv_std.expand(&[b * t, d])?)?;
        normed_layers.push(normed);
    }

    // Cat along dim 1: 49 × [B*T, D] → [B*T, D*49]
    let refs: Vec<&Tensor> = normed_layers.iter().collect();
    let concatenated = Tensor::cat(&refs, 1)?; // [B*T, 188160]

    // Reshape to [B, T, D*L]
    let concatenated = concatenated.reshape(&[b, t, flat_dim])?;

    // Apply attention mask: zero out padded positions
    let mask_3d = attention_mask.unsqueeze(2)?.expand(&[b, t, flat_dim])?;
    concatenated.mul(&mask_3d)
}

/// Per-token RMSNorm normalization for V2 models (original stack-based version).
#[allow(dead_code)]
fn norm_and_concat_per_token_rms(
    encoded: &Tensor,
    attention_mask: &Tensor,
    b: usize,
    t: usize,
    d: usize,
    l: usize,
) -> Result<Tensor> {
    let squared = encoded.mul(encoded)?;
    let variance = squared.mean_dim(&[2], false)?;
    let variance = variance.unsqueeze(2)?;
    let var_eps = variance.add_scalar(1e-6)?;
    let inv_std = var_eps.rsqrt()?;
    let normed = encoded.mul(&inv_std.expand(&[b, t, d, l])?)?;
    let normed = normed.reshape(&[b, t, d * l])?;
    let mask_3d = attention_mask.unsqueeze(2)?.expand(&[b, t, d * l])?;
    normed.mul(&mask_3d)
}

// ---------------------------------------------------------------------------
// Convert attention mask to additive form
// ---------------------------------------------------------------------------

/// Convert binary attention mask [B, seq_len] to additive mask [B, 1, 1, seq_len].
///
/// ## PyTorch reference (convert_to_additive_mask):
/// ```python
/// return (attention_mask.to(torch.int64) - 1).to(dtype).reshape(
///     (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
/// ) * torch.finfo(dtype).max
/// ```
///
/// Real tokens (mask=1) → 0.0, padded tokens (mask=0) → -large_number.
pub fn convert_to_additive_mask(attention_mask: &Tensor) -> Result<Tensor> {
    let dims = attention_mask.shape().dims().to_vec();
    let b = dims[0];
    let seq_len = dims[1];

    // (mask - 1) * large_value
    // mask=1 → 0, mask=0 → -1 * large_value
    let mask_f32 = attention_mask.to_dtype(DType::F32)?;
    let shifted = mask_f32.add_scalar(-1.0)?; // [B, seq] with 0 for real, -1 for pad
    let large_val = f32::MAX * 0.5; // Avoid overflow in BF16
    let additive = shifted.mul_scalar(large_val)?;

    // Reshape to [B, 1, 1, seq_len]
    let additive = additive.reshape(&[b, 1, 1, seq_len])?.to_dtype(DType::BF16)?;
    Ok(additive)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_dim_calculation() {
        let d = 3840;
        let l = 49;
        assert_eq!(d * l, 188160);
    }

    #[test]
    fn test_rescale_factor() {
        let target = 4096.0f64;
        let source = 3840.0f64;
        let factor = (target / source).sqrt();
        assert!((factor - 1.0327955).abs() < 0.001);
    }
}
