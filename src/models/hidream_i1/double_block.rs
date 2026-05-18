//! HiDreamImageTransformerBlock — double-stream forward (image + text).
//!
//! Mirrors `transformer_hidream_image.py:106-188`.
//!
//! Block layout (per `transformer_hidream_image.py:115-147`):
//!   - `adaLN_modulation = Sequential(SiLU, Linear(dim, 12*dim))`
//!   - `norm1_i` / `norm1_t` LayerNorm (no affine)
//!   - `attn1` HiDreamAttention(single=False) — dual Q/K/V (img: to_q/to_k/to_v,
//!     txt: to_q_t/to_k_t/to_v_t), RMSNorm-Q/K, concat across image+text,
//!     joint SDPA, split outputs to `to_out` (img) and `to_out_t` (txt).
//!   - `norm3_i` / `norm3_t` LayerNorm (no affine)
//!   - `ff_i`  MoE FeedForward (4 routed + shared)
//!   - `ff_t`  dense FeedForward (SwiGLU)
//!
//! Modulation chunking — 12 chunks, [B, 1, dim]:
//!   [0..6]  img: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
//!   [6..12] txt: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
//!
//! `attn_output_i = gate_msa_i * attn_i + img_in`
//! `attn_output_t = gate_msa_t * attn_t + txt_in`
//! `ff_i_out      = gate_mlp_i * MoE(norm_i)`     residual added
//! `ff_t_out      = gate_mlp_t * SwiGLU(norm_t)`  residual added
//!
//! ## RoPE
//! HiDream-I1 uses FLUX-style EmbedND. `rope` shape = `[B*?, 1, S_total, head_dim/2, 2, 2]`
//! per `embeddings.py::rope` (cos/-sin/sin/cos stacked).
//! The attention_processor's `apply_rope` (line 21-26 of `attention_processor.py`)
//! is a matmul form. We implement this directly in `apply_rope_complex_2x2`
//! since the Chroma `bf16_ops::rope_fused_bf16` expects a cos/sin pair format.
//!
//! ## Implementation
//! - The image_tokens_masks path is supported but defaults to None for
//!   square training. See `attention_processor.py:81-83` for the masking
//!   semantics (key_i = key_i * mask.view(B, -1, 1, 1)).
//! - SDPA is called via `flame_core::attention::sdpa(q, k, v, mask)`.
//!   q/k/v shape after concat: `[B, H, S_total, D]`.
//!
//! Note: The Python uses `image_tokens_masks` to zero out keys before
//! attention; we replicate that exactly to avoid changing flame-core's
//! SDPA mask semantics.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::{CudaDevice, DType, Result, Tensor};

use super::moe::{moe_ffn_forward, MoeWeights};

pub(crate) struct DoubleBlockCfg {
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

/// LayerNorm with no affine (matches `nn.LayerNorm(elementwise_affine=False)`).
fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
    out.reshape(&dims)
}

/// Apply RMSNorm with the per-head scale broadcast across heads.
/// `scale` is `[inner_dim]`, `x` is `[B, H, S, D]` with `inner_dim = H*D`.
fn rms_norm_per_head(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    // collapse to [N, inner_dim] = [B*H*S, D] and apply across the inner
    // last-dim. But scale is [H*D] — we need to apply only along the head_dim
    // axis. The Python (`q_rms_norm = nn.RMSNorm(inner_dim)`) is applied to
    // a [B, S, inner_dim] tensor BEFORE the view to [B, S, H, D]. So we
    // mirror by reshaping x back to its pre-view layout.
    // For our internal flow, x already is [B, H, S, D]. Re-collapse:
    //   [B, H, S, D] → permute to [B, S, H*D] → norm → reshape back.
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let x_collapsed = x
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b * s, h * d])?;
    let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_collapsed, Some(scale), eps)?;
    normed
        .reshape(&[b, s, h, d])?
        .permute(&[0, 2, 1, 3])
}

/// FLUX/HiDream apply_rope: `freqs_cis` is the 2x2 matrix form
/// `[..., D/2, 2, 2]` containing `[[cos, -sin], [sin, cos]]`. Apply per-pair.
///
/// Implementation: split q/k along last dim into pairs (D → D/2 pairs of 2),
/// then per pair: `(q0, q1) → (cos*q0 + (-sin)*q1, sin*q0 + cos*q1)`.
/// This is the matrix form: `[q0; q1] := [[cos, -sin], [sin, cos]] @ [q0; q1]`.
///
/// `q`, `k`: `[B, H, S, D]` BF16
/// `freqs_cis`: `[1, 1, S, D/2, 2, 2]` BF16
pub(crate) fn apply_rope_matrix_form(q: &Tensor, k: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    // Decompose freqs_cis into cos/sin components.
    // freqs_cis[..., 0, 0] = cos, freqs_cis[..., 0, 1] = -sin
    // freqs_cis[..., 1, 0] = sin, freqs_cis[..., 1, 1] = cos
    //
    // Slice along last two dims to get cos and -sin, then call into the
    // bf16_ops::rope_fused_bf16 path. That kernel expects `pe_cos` and
    // `pe_sin` separately with the half-split layout (q split into halves
    // and rotated, NOT the pair-interleaved 2x2 matrix form).
    //
    // HiDream uses the PAIR form (q_0, q_1) → (q_0 cos - q_1 sin, q_0 sin + q_1 cos),
    // matching the FLUX BFL math. flame-core has `rope_fused_bf16` for the
    // *half-split* form. The two differ in how pairs are laid out (interleaved
    // vs concatenated halves). For numerical parity with diffusers' HiDream,
    // we implement the pair-form explicitly here.
    //
    // q has shape [B, H, S, D]. Reshape to [B, H, S, D/2, 2], split pair0/pair1.
    let q_dims = q.shape().dims().to_vec();
    let (b, h, s, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let d_half = d / 2;

    let q_pairs = q.reshape(&[b, h, s, d_half, 2])?;
    let q0 = q_pairs.narrow(4, 0, 1)?.squeeze(Some(4))?; // [B, H, S, D/2]
    let q1 = q_pairs.narrow(4, 1, 1)?.squeeze(Some(4))?;
    let k_pairs = k.reshape(&[b, h, s, d_half, 2])?;
    let k0 = k_pairs.narrow(4, 0, 1)?.squeeze(Some(4))?;
    let k1 = k_pairs.narrow(4, 1, 1)?.squeeze(Some(4))?;

    // freqs_cis: [1, 1, S, D/2, 2, 2]. Extract the (0,0) and (1,0) entries
    // → cos, sin. (We don't need (0,1) and (1,1) separately — they're
    // `-sin` and `cos` by construction.)
    let f_dims = freqs_cis.shape().dims().to_vec();
    if f_dims.len() != 6 {
        return Err(flame_core::Error::InvalidInput(format!(
            "apply_rope_matrix_form: freqs_cis must be rank-6, got {f_dims:?}"
        )));
    }
    // [.., 2, 2] → narrow dim 4 (the first 2) into single entries, then dim 5.
    let row0 = freqs_cis.narrow(4, 0, 1)?.squeeze(Some(4))?; // [.., 2]
    let row1 = freqs_cis.narrow(4, 1, 1)?.squeeze(Some(4))?;
    let cos = row0.narrow(4, 0, 1)?.squeeze(Some(4))?; // [1, 1, S, D/2]
    let sin = row1.narrow(4, 0, 1)?.squeeze(Some(4))?;

    // Broadcast cos/sin across batch and heads → [B, H, S, D/2].
    let cos_b = cos.expand(&[b, h, s, d_half])?;
    let sin_b = sin.expand(&[b, h, s, d_half])?;

    // q' = q0*cos - q1*sin (paired)
    // q'' = q0*sin + q1*cos
    let q0_cos = q0.mul(&cos_b)?;
    let q1_sin = q1.mul(&sin_b)?;
    let q0_sin = q0.mul(&sin_b)?;
    let q1_cos = q1.mul(&cos_b)?;
    let q_out0 = q0_cos.sub(&q1_sin)?;
    let q_out1 = q0_sin.add(&q1_cos)?;
    let k0_cos = k0.mul(&cos_b)?;
    let k1_sin = k1.mul(&sin_b)?;
    let k0_sin = k0.mul(&sin_b)?;
    let k1_cos = k1.mul(&cos_b)?;
    let k_out0 = k0_cos.sub(&k1_sin)?;
    let k_out1 = k0_sin.add(&k1_cos)?;

    // Recombine into pairs and reshape back.
    // Need [B, H, S, D/2, 2] from two [B, H, S, D/2] tensors stacked along a new last dim.
    let q_out0_u = q_out0.unsqueeze(4)?;
    let q_out1_u = q_out1.unsqueeze(4)?;
    let q_paired = Tensor::cat(&[&q_out0_u, &q_out1_u], 4)?;
    let q_final = q_paired.reshape(&[b, h, s, d])?;
    let k_out0_u = k_out0.unsqueeze(4)?;
    let k_out1_u = k_out1.unsqueeze(4)?;
    let k_paired = Tensor::cat(&[&k_out0_u, &k_out1_u], 4)?;
    let k_final = k_paired.reshape(&[b, h, s, d])?;

    Ok((q_final, k_final))
}

/// Modulate: `out = LayerNorm(x) * (1 + scale) + shift`.
/// `shift`, `scale` are `[B, 1, dim]` (broadcastable over seq).
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    let normed = layer_norm_no_affine(x, eps)?;
    let one_plus = scale.add_scalar(1.0)?;
    let scaled = normed.mul(&one_plus)?;
    scaled.add(shift)
}

/// Linear with bias.
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

fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 3 && dims[0] > 1 {
        let (b, n, c) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[1, b * n, c])?;
        let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, None)?;
        let out_c = weight.shape().dims()[0];
        out.reshape(&[b, n, out_c])
    } else {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
    }
}

/// Dense SwiGLU FeedForward (for `ff_t` — text stream only).
fn dense_swiglu_ffn(x: &Tensor, w1: &Tensor, w2: &Tensor, w3: &Tensor) -> Result<Tensor> {
    let h1 = linear_nobias(x, w1)?.silu()?;
    let h3 = linear_nobias(x, w3)?;
    let gated = h1.mul(&h3)?;
    linear_nobias(&gated, w2)
}

/// Double-stream block forward.
///
/// `weights` is the (untransposed) per-block weight map for
/// `double_stream_blocks.{block_idx}.block.*` — the prefix is already stripped.
///
/// `adaln_input`: `[B, dim]` BF16. The block's adaLN_modulation projects to
/// `[B, 12*dim]`, unsqueezes to `[B, 1, 12*dim]`, and chunks into 12 modulation
/// vectors `[B, 1, dim]` each.
///
/// `img`: `[B, N_img, dim]`, `txt`: `[B, N_txt, dim]`.
/// `image_tokens_masks`: optional `[B, N_img]` mask (1.0 valid, 0.0 padding).
/// `rope`: `[1, 1, S_total, head_dim/2, 2, 2]` BF16 (full S = N_img + N_txt).
///
/// Returns `(img', txt')`.
///
/// LORA-TARGET: every Linear weight is a LoRA candidate:
///   - `adaLN_modulation.1` (the only Linear in the modulation sequence)
///   - `attn1.{to_q, to_k, to_v, to_out, to_q_t, to_k_t, to_v_t, to_out_t}`
///   - `ff_i.{shared_experts, experts.*, gate}` (excluded from LoRA per YAML default)
///   - `ff_t.{w1, w2, w3}`
pub(crate) fn forward(
    cfg: &DoubleBlockCfg,
    img: &Tensor,
    txt: &Tensor,
    adaln_input: &Tensor,
    image_tokens_masks: Option<&Tensor>,
    rope: &Tensor,
    weights: &HashMap<String, Tensor>,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let h = cfg.num_heads;
    let d = cfg.head_dim;
    let dim = cfg.dim;

    let g = |k: &str| -> Result<&Tensor> {
        weights.get(k).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing double-block weight: {k}"))
        })
    };

    // ----- 1. adaLN modulation: SiLU(adaln) → Linear(dim → 12*dim) ---------
    let adaln = adaln_input.silu()?;
    let ada_w = g("adaLN_modulation.1.weight")?;
    let ada_b = g("adaLN_modulation.1.bias")?;
    // adaln is [B, dim]; fused_linear3d_native rejects 2D, so use 2D-compat helper.
    let mod_out = super::model::linear_compat(&adaln, ada_w, Some(ada_b))?;
    // mod_out shape [B, 12*dim]; unsqueeze to [B, 1, 12*dim] then chunk to 12 [B, 1, dim].
    let mod_unsq = mod_out.unsqueeze(1)?;
    // Slice into 12 chunks along dim=2.
    let mut chunks: Vec<Tensor> = Vec::with_capacity(12);
    for i in 0..12 {
        chunks.push(mod_unsq.narrow(2, i * dim, dim)?);
    }
    let img_shift_msa = &chunks[0];
    let img_scale_msa = &chunks[1];
    let img_gate_msa = &chunks[2];
    let img_shift_mlp = &chunks[3];
    let img_scale_mlp = &chunks[4];
    let img_gate_mlp = &chunks[5];
    let txt_shift_msa = &chunks[6];
    let txt_scale_msa = &chunks[7];
    let txt_gate_msa = &chunks[8];
    let txt_shift_mlp = &chunks[9];
    let txt_scale_mlp = &chunks[10];
    let txt_gate_mlp = &chunks[11];

    // ----- 2. Norm + modulate (MSA inputs) ---------------------------------
    let norm_img = modulate(img, img_shift_msa, img_scale_msa, cfg.eps)?;
    let norm_txt = modulate(txt, txt_shift_msa, txt_scale_msa, cfg.eps)?;

    // ----- 3. Q/K/V projections (img and txt have SEPARATE Linears) --------
    let q_i_raw = linear_bias(&norm_img, g("attn1.to_q.weight")?, g("attn1.to_q.bias")?)?;
    let k_i_raw = linear_bias(&norm_img, g("attn1.to_k.weight")?, g("attn1.to_k.bias")?)?;
    let v_i_raw = linear_bias(&norm_img, g("attn1.to_v.weight")?, g("attn1.to_v.bias")?)?;
    let q_t_raw = linear_bias(&norm_txt, g("attn1.to_q_t.weight")?, g("attn1.to_q_t.bias")?)?;
    let k_t_raw = linear_bias(&norm_txt, g("attn1.to_k_t.weight")?, g("attn1.to_k_t.bias")?)?;
    let v_t_raw = linear_bias(&norm_txt, g("attn1.to_v_t.weight")?, g("attn1.to_v_t.bias")?)?;

    let img_dims = img.shape().dims().to_vec();
    let (b, n_img) = (img_dims[0], img_dims[1]);
    let n_txt = txt.shape().dims()[1];

    // RMSNorm on (img) Q/K, then reshape+permute to [B, H, S, D].
    // Python: q_rms_norm(to_q(x))  -- norm over inner_dim, then view+permute.
    // We follow the python ordering exactly: RMSNorm BEFORE the head reshape.
    let q_i = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &q_i_raw.reshape(&[b * n_img, h * d])?,
        Some(g("attn1.q_rms_norm.weight")?),
        cfg.eps_rms,
    )?
    .reshape(&[b, n_img, h, d])?
    .permute(&[0, 2, 1, 3])?;
    let k_i = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &k_i_raw.reshape(&[b * n_img, h * d])?,
        Some(g("attn1.k_rms_norm.weight")?),
        cfg.eps_rms,
    )?
    .reshape(&[b, n_img, h, d])?
    .permute(&[0, 2, 1, 3])?;
    let v_i = v_i_raw.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;

    let q_t = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &q_t_raw.reshape(&[b * n_txt, h * d])?,
        Some(g("attn1.q_rms_norm_t.weight")?),
        cfg.eps_rms,
    )?
    .reshape(&[b, n_txt, h, d])?
    .permute(&[0, 2, 1, 3])?;
    let k_t = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &k_t_raw.reshape(&[b * n_txt, h * d])?,
        Some(g("attn1.k_rms_norm_t.weight")?),
        cfg.eps_rms,
    )?
    .reshape(&[b, n_txt, h, d])?
    .permute(&[0, 2, 1, 3])?;
    let v_t = v_t_raw.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;
    let _ = rms_norm_per_head; // exported helper; kept for readability.

    // ----- 4. Apply image-token masks to keys (Python attention_processor:81) ----
    let k_i = if let Some(mask) = image_tokens_masks {
        // mask: [B, N_img]; need to broadcast to [B, H, N_img, D] for elementwise mul.
        let m_b = mask
            .reshape(&[b, 1, n_img, 1])?
            .expand(&[b, h, n_img, d])?;
        k_i.mul(&m_b)?
    } else {
        k_i
    };

    // ----- 5. Concat img+txt along seq dim → [B, H, N_img + N_txt, D] ------
    // Python concats in order [img, txt] (attention_processor.py:95-97).
    let q = Tensor::cat(&[&q_i, &q_t], 2)?;
    let k = Tensor::cat(&[&k_i, &k_t], 2)?;
    let v = Tensor::cat(&[&v_i, &v_t], 2)?;

    // ----- 6. Apply RoPE (pair-form matrix multiply per attention_processor.py:21) --
    let (q, k) = apply_rope_matrix_form(&q, &k, rope)?;

    // ----- 7. SDPA -----
    let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;

    // ----- 8. Split back into img / txt portions, permute to [B, S, H*D] ----
    // attn shape: [B, H, N_img + N_txt, D]. Narrow along dim=2.
    let attn_i = attn
        .narrow(2, 0, n_img)?
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b, n_img, h * d])?;
    let attn_t = attn
        .narrow(2, n_img, n_txt)?
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b, n_txt, h * d])?;

    // ----- 9. Output projections (separate img and txt heads) --------------
    let attn_i = linear_bias(&attn_i, g("attn1.to_out.weight")?, g("attn1.to_out.bias")?)?;
    let attn_t = linear_bias(&attn_t, g("attn1.to_out_t.weight")?, g("attn1.to_out_t.bias")?)?;

    // ----- 10. Gated residual (MSA) ----------------------------------------
    let gate_attn_i = img_gate_msa.mul(&attn_i)?;
    let img_after_attn = img.add(&gate_attn_i)?;
    let gate_attn_t = txt_gate_msa.mul(&attn_t)?;
    let txt_after_attn = txt.add(&gate_attn_t)?;

    // ----- 11. FFN block ---------------------------------------------------
    // Image: MoE FFN; Text: dense SwiGLU.
    let norm_img_mlp = modulate(&img_after_attn, img_shift_mlp, img_scale_mlp, cfg.eps)?;
    let norm_txt_mlp = modulate(&txt_after_attn, txt_shift_mlp, txt_scale_mlp, cfg.eps)?;

    let moe = MoeWeights::from_block(weights, "ff_i", cfg.num_routed_experts)?;
    let ff_img = moe_ffn_forward(&norm_img_mlp, &moe, cfg.num_activated_experts, device)?;
    let ff_txt = dense_swiglu_ffn(
        &norm_txt_mlp,
        g("ff_t.w1.weight")?,
        g("ff_t.w2.weight")?,
        g("ff_t.w3.weight")?,
    )?;

    let gate_ff_i = img_gate_mlp.mul(&ff_img)?;
    let img_out = img_after_attn.add(&gate_ff_i)?;
    let gate_ff_t = txt_gate_mlp.mul(&ff_txt)?;
    let txt_out = txt_after_attn.add(&gate_ff_t)?;

    // silence unused on the float helper variables
    let _ = (n_img, n_txt, dim);
    let _ = DType::BF16;
    Ok((img_out, txt_out))
}
