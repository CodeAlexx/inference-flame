//! HiDreamImageSingleTransformerBlock — single-stream forward.
//!
//! Mirrors `transformer_hidream_image.py:34-103`.
//!
//! Block layout (`transformer_hidream_image.py:45-73`):
//!   - `adaLN_modulation = Sequential(SiLU, Linear(dim, 6*dim))`
//!   - `norm1_i` LayerNorm (no affine)
//!   - `attn1` HiDreamAttention(single=True) — only `to_q/to_k/to_v/to_out`
//!     + RMSNorm-Q/K (no _t variants).
//!   - `norm3_i` LayerNorm (no affine)
//!   - `ff_i` MoE FFN (4 routed + shared) — same as double block image stream
//!
//! Modulation chunking — 6 chunks, [B, 1, dim]:
//!   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
//!
//! ## Concat layout (from the main forward)
//! The single block receives `image_tokens` that has already been concatenated
//! with text+llama: `hidden_states = cat([img, initial_encoder, cur_llama_layer], dim=1)`.
//! It then does pure self-attention over the whole concat. The split happens
//! in the outer model forward (`hidden_states[:, :hidden_states_seq_len]`).
//! From the block's POV it's just a sequence with no img/txt split — all one
//! `image_tokens` tensor.
//!
//! ## RoPE
//! `rope` shape `[1, 1, S_total, head_dim/2, 2, 2]` — same RoPE table as
//! the double blocks. The seq dim grows as Llama layers are concatenated;
//! the caller (model.rs) is responsible for re-building / re-using a RoPE
//! table sized for the maximum effective seq length.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::{CudaDevice, Result, Tensor};

use super::moe::{moe_ffn_forward, MoeWeights};

pub(crate) struct SingleBlockCfg {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_routed_experts: usize,
    pub num_activated_experts: usize,
    /// LayerNorm epsilon (1e-6).
    pub eps: f32,
    /// RMSNorm epsilon (1e-5). BUG #6 fix.
    pub eps_rms: f32,
}

fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
    out.reshape(&dims)
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    let normed = layer_norm_no_affine(x, eps)?;
    let one_plus = scale.add_scalar(1.0)?;
    let scaled = normed.mul(&one_plus)?;
    scaled.add(shift)
}

fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 3 && dims[0] > 1 {
        let (b, n, c) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[1, b * n, c])?;
        let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, Some(bias))?;
        let out_c = weight.shape().dims()[0];
        out.reshape(&[b, n, out_c])
    } else {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
    }
}

/// Re-use the double-block apply_rope helper (pair-form matrix multiply).
fn apply_rope_matrix_form(q: &Tensor, k: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    super::double_block::apply_rope_matrix_form(q, k, freqs_cis)
}

/// Single-stream block forward.
///
/// `weights` is the per-block weight map (prefix stripped).
/// `image_tokens`: `[B, N, dim]` where N includes the concatenated text + llama
/// segments per the outer-forward layout.
///
/// LORA-TARGET: every Linear weight is a LoRA candidate:
///   - `adaLN_modulation.1`
///   - `attn1.{to_q, to_k, to_v, to_out}`
///   - `ff_i.{shared_experts, experts.*, gate}` (excluded by default in YAML)
pub(crate) fn forward(
    cfg: &SingleBlockCfg,
    image_tokens: &Tensor,
    adaln_input: &Tensor,
    image_tokens_masks: Option<&Tensor>,
    rope: &Tensor,
    weights: &HashMap<String, Tensor>,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let h = cfg.num_heads;
    let d = cfg.head_dim;
    let dim = cfg.dim;

    let g = |k: &str| -> Result<&Tensor> {
        weights.get(k).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing single-block weight: {k}"))
        })
    };

    // ----- 1. adaLN modulation -------------------------------------------
    let adaln = adaln_input.silu()?;
    let ada_w = g("adaLN_modulation.1.weight")?;
    let ada_b = g("adaLN_modulation.1.bias")?;
    // adaln is [B, dim]; use 2D-compat helper per BUG #1 fix.
    let mod_out = super::model::linear_compat(&adaln, ada_w, Some(ada_b))?;
    let mod_unsq = mod_out.unsqueeze(1)?;
    let mut chunks: Vec<Tensor> = Vec::with_capacity(6);
    for i in 0..6 {
        chunks.push(mod_unsq.narrow(2, i * dim, dim)?);
    }
    let shift_msa = &chunks[0];
    let scale_msa = &chunks[1];
    let gate_msa = &chunks[2];
    let shift_mlp = &chunks[3];
    let scale_mlp = &chunks[4];
    let gate_mlp = &chunks[5];

    // ----- 2. Norm + modulate ---------------------------------------------
    let norm_x = modulate(image_tokens, shift_msa, scale_msa, cfg.eps)?;
    let dims_in = image_tokens.shape().dims().to_vec();
    let (b, n) = (dims_in[0], dims_in[1]);

    // ----- 3. Q/K/V projections -------------------------------------------
    let q_raw = linear_bias(&norm_x, g("attn1.to_q.weight")?, g("attn1.to_q.bias")?)?;
    let k_raw = linear_bias(&norm_x, g("attn1.to_k.weight")?, g("attn1.to_k.bias")?)?;
    let v_raw = linear_bias(&norm_x, g("attn1.to_v.weight")?, g("attn1.to_v.bias")?)?;

    let q = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &q_raw.reshape(&[b * n, h * d])?,
        Some(g("attn1.q_rms_norm.weight")?),
        cfg.eps_rms,
    )?
    .reshape(&[b, n, h, d])?
    .permute(&[0, 2, 1, 3])?;
    let k = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &k_raw.reshape(&[b * n, h * d])?,
        Some(g("attn1.k_rms_norm.weight")?),
        cfg.eps_rms,
    )?
    .reshape(&[b, n, h, d])?
    .permute(&[0, 2, 1, 3])?;
    let v = v_raw.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;

    // ----- 4. Image-token mask on keys (only the image span) --------------
    // The single block sees [img | initial_encoder | llama] concatenated.
    // `image_tokens_masks` in Python (transformer_hidream_image.py:464-468) is
    // expanded to cover the full concat with 1.0s for non-image spans. The
    // outer model passes the already-expanded mask, so it's [B, N_concat] and
    // we apply it uniformly.
    let k = if let Some(mask) = image_tokens_masks {
        let m_b = mask
            .reshape(&[b, 1, n, 1])?
            .expand(&[b, h, n, d])?;
        k.mul(&m_b)?
    } else {
        k
    };

    // ----- 5. RoPE --------------------------------------------------------
    let (q, k) = apply_rope_matrix_form(&q, &k, rope)?;

    // ----- 6. SDPA --------------------------------------------------------
    let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

    // ----- 7. Output projection + gated residual --------------------------
    let attn = linear_bias(&attn, g("attn1.to_out.weight")?, g("attn1.to_out.bias")?)?;
    let gate_attn = gate_msa.mul(&attn)?;
    let x_after_attn = image_tokens.add(&gate_attn)?;

    // ----- 8. FFN (MoE) + gated residual ----------------------------------
    let norm_x_mlp = modulate(&x_after_attn, shift_mlp, scale_mlp, cfg.eps)?;
    let moe = MoeWeights::from_block(weights, "ff_i", cfg.num_routed_experts)?;
    let ff = moe_ffn_forward(&norm_x_mlp, &moe, cfg.num_activated_experts, device)?;
    let gate_ff = gate_mlp.mul(&ff)?;
    let x_out = x_after_attn.add(&gate_ff)?;
    Ok(x_out)
}
