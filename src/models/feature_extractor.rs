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

    // 1. Stack hidden states: [B, T, D, L=49]
    //
    // ## PyTorch reference:
    // ```python
    // encoded = torch.stack(hidden_states, dim=-1)
    // ```
    let stacked = Tensor::stack(hidden_states, 3)?; // [B, T, D, L]
    let stacked_dims = stacked.shape().dims().to_vec();
    assert_eq!(stacked_dims, &[b, t, d, num_layers]);

    // 2. Per-token RMSNorm and concatenation
    //
    // ## PyTorch reference (norm_and_concat_per_token_rms):
    // ```python
    // variance = torch.mean(encoded**2, dim=2, keepdim=True)  # [B,T,1,L]
    // normed = encoded * torch.rsqrt(variance + 1e-6)
    // normed = normed.reshape(B, T, D * L)
    // mask_3d = attention_mask.bool().unsqueeze(-1)
    // return torch.where(mask_3d, normed, torch.zeros_like(normed))
    // ```
    let normed = norm_and_concat_per_token_rms(&stacked, attention_mask, b, t, d, num_layers)?;

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
    // ## PyTorch reference:
    // ```python
    // video = self.video_aggregate_embed(rescaled)  # nn.Linear(188160, 4096, bias=True)
    // ```
    let flat_dim = d * num_layers; // 188160
    let agg_w_dims = aggregate_embed_weight.shape().dims().to_vec();

    // Weight may be [4096, 188160] or [188160, 4096] (pre-transposed)
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

    // matmul: [B*T, 188160] x [188160, 4096] → [B*T, 4096]
    let rescaled_2d = rescaled.reshape(&[b * t, flat_dim])?;
    let mut projected = rescaled_2d.matmul(&weight_t)?;

    // Add bias if present
    if let Some(bias) = aggregate_embed_bias {
        projected = projected.add(&bias.unsqueeze(0)?.expand(&[b * t, target_dim])?)?;
    }

    let result = projected.reshape(&[b, t, target_dim])?;
    log::info!("[FeatureExtractor] Output: {:?}", result.shape());
    Ok(result)
}

/// Per-token RMSNorm normalization for V2 models.
///
/// Input: [B, T, D, L] stacked hidden states.
/// Output: [B, T, D*L] normalized tensor with padding zeroed.
///
/// ## PyTorch reference (norm_and_concat_per_token_rms):
/// ```python
/// B, T, D, L = encoded_text.shape
/// variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B,T,1,L]
/// normed = encoded_text * torch.rsqrt(variance + 1e-6)
/// normed = normed.reshape(B, T, D * L)
/// mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
/// return torch.where(mask_3d, normed, torch.zeros_like(normed))
/// ```
fn norm_and_concat_per_token_rms(
    encoded: &Tensor,
    attention_mask: &Tensor,
    b: usize,
    t: usize,
    d: usize,
    l: usize,
) -> Result<Tensor> {
    // Convert to F32 for normalization precision
    let enc_f32 = encoded.to_dtype(DType::F32)?;

    // variance: mean(x^2, dim=2, keepdim=True) → [B, T, 1, L]
    let squared = enc_f32.mul(&enc_f32)?;
    let variance = squared.mean_dim(&[2], false)?; // [B, T, L] after mean over D
    let variance = variance.unsqueeze(2)?; // [B, T, 1, L]

    // rsqrt(variance + 1e-6)
    let eps_tensor = Tensor::from_vec(
        vec![1e-6f32],
        Shape::from_dims(&[1, 1, 1, 1]),
        encoded.device().clone(),
    )?;
    let var_eps = variance.add(&eps_tensor.expand(&variance.shape().dims().to_vec())?)?;
    let inv_std = var_eps.rsqrt()?;

    // normed = encoded * rsqrt(var + eps)
    let normed = enc_f32.mul(&inv_std.expand(&[b, t, d, l])?)?;

    // Reshape to [B, T, D*L]
    let normed = normed.reshape(&[b, t, d * l])?;

    // Apply attention mask: zero out padded positions
    // mask: [B, T] → [B, T, 1]
    let mask_f32 = attention_mask.to_dtype(DType::F32)?;
    let mask_3d = mask_f32.unsqueeze(2)?.expand(&[b, t, d * l])?;
    let normed = normed.mul(&mask_3d)?;

    normed.to_dtype(DType::BF16)
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
