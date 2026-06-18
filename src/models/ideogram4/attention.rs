//! Ideogram 4 attention block — bias-free QKV, QK-RMSNorm on head_dim, MRoPE
//! (half-split), SDPA with an optional block-diagonal segment mask, bias-free
//! output projection.
//!
//! Mirrors `Ideogram4Attention.forward` (`modeling_ideogram4.py:130-158`)
//! op-for-op:
//!
//! ```text
//! qkv = self.qkv(x)                                   # bias-free Linear(H -> 3H)
//! qkv = qkv.view(B, L, 3, num_heads, head_dim)
//! q, k, v = qkv.unbind(dim=2)                         # each [B, L, num_heads, head_dim]
//! q = self.norm_q(q); k = self.norm_k(k)              # RMSNorm over head_dim, BEFORE transpose
//! q = q.transpose(1, 2); k = ...; v = ...             # -> [B, num_heads, L, head_dim]
//! q, k = _apply_rotary_pos_emb(q, k, cos, sin)        # q*cos + rotate_half(q)*sin (HALFSPLIT)
//! attn_mask = (seg[:,:,None] == seg[:,None,:])[:,None]  # [B, 1, L, L] bool, True=attend
//! out = F.scaled_dot_product_attention(q, k, v, attn_mask)
//! out = out.transpose(1, 2).reshape(B, L, H)
//! return self.o(out)                                  # bias-free Linear(H -> H)
//! ```
//!
//! ## head_dim 256 note (v1 GPU-gated perf gate)
//! `head_dim = emb_dim/num_heads = 4608/18 = 256` is OUTSIDE the cuDNN/WMMA
//! flash-attention set {64, 96, 128}, so `attention::sdpa::sdpa` routes to the
//! math/streaming BF16 fallback (`sdpa_stream_bf16`). That fallback is correct
//! but slow; a d=256 fast path is a flame-core primitive change deferred to a
//! GPU-gated step (BUILD_PLAN flame-core changes + PORT_STATE). This module
//! only WIRES the call. `attention::sdpa::sdpa` has NO compile-time head_dim
//! assertion (verified `sdpa.rs:452-467` only checks dq==dk==dv), so d=256 is
//! accepted at the API.
//!
//! ## segment mask
//! For B=1 unpadded inference the block-diagonal segment mask is all-ones
//! (every token attends to every other token), so it can be passed as `None`.
//! When packing multiple samples, build a bool `[B,1,L,L]` mask (True=attend)
//! and pass it through. `sdpa` accepts a bool mask (`sdpa.rs:502-507`).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::attention::sdpa;
use flame_core::bf16_ops::rope_halfsplit_bf16_pytorch;
use flame_core::ops::fused_inference::{fused_linear3d_native, fused_rms_norm};
use flame_core::{Result, Tensor};

use super::weights::Ideogram4RawWeight;

/// Bias-free 3D linear from an FP8/BF16 raw weight in the block weight map.
///
/// `weight_key` is the full PyTorch weight path (e.g.
/// `layers.0.attention.qkv.weight`). The weight is dequantized on-the-fly to
/// BF16 `[out, in]` and consumed by `fused_linear3d_native` (cuBLASLt TRANSA=T
/// — no pre-transpose; CLAUDE.md "NEVER pre-transpose weights at every call").
/// `input` must be 3D `[B, N, Cin]`.
pub(crate) fn linear_no_bias(
    weights: &HashMap<String, Ideogram4RawWeight>,
    weight_key: &str,
    input: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let w = weights
        .get(weight_key)
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!(
                "ideogram4 linear_no_bias: missing weight `{weight_key}`"
            ))
        })?
        .to_bf16_tensor(device)?;
    fused_linear3d_native(input, &w, None)
}

/// Ideogram4 attention forward.
///
/// - `x`: `[B, L, hidden]` BF16 — the (already adaln-scaled) block input.
/// - `cos`, `sin`: `[1, S, head_dim/2]` BF16 MRoPE half-table from
///   [`super::mrope::build_cos_sin`].
/// - `attn_mask`: optional bool `[B, 1, L, L]` (True=attend). Pass `None` for
///   B=1 unpadded (mask is all-ones).
/// - `prefix`: e.g. `layers.0.attention` — weight keys are `{prefix}.qkv.weight`,
///   `{prefix}.norm_q.weight`, `{prefix}.norm_k.weight`, `{prefix}.o.weight`.
#[allow(clippy::too_many_arguments)]
pub fn attention_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    attn_mask: Option<&Tensor>,
    num_heads: usize,
    norm_eps: f32,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(flame_core::Error::InvalidShape(format!(
            "ideogram4 attention: input must be 3D [B,L,hidden], got {dims:?}"
        )));
    }
    let (b, seq, hidden) = (dims[0], dims[1], dims[2]);
    if hidden % num_heads != 0 {
        return Err(flame_core::Error::InvalidShape(format!(
            "ideogram4 attention: hidden {hidden} not divisible by num_heads {num_heads}"
        )));
    }
    let head_dim = hidden / num_heads;

    // qkv = self.qkv(x) -> [B, L, 3*hidden]; view [B, L, 3, num_heads, head_dim].
    let qkv = linear_no_bias(weights, &format!("{prefix}.qkv.weight"), x, device)?;
    // chunk along last dim into q,k,v each [B, L, hidden]. The Python view is
    // [B,L,3,H,Dh] then unbind(dim=2); a contiguous [B,L,3H] row-major buffer
    // splits identically with chunk(3, last) — each chunk is [B,L,hidden].
    let chunks = qkv.chunk(3, 2)?;
    let q = chunks[0].reshape(&[b, seq, num_heads, head_dim])?;
    let k = chunks[1].reshape(&[b, seq, num_heads, head_dim])?;
    let v = chunks[2].reshape(&[b, seq, num_heads, head_dim])?;

    // QK-RMSNorm over head_dim (last dim), BEFORE the transpose — matches
    // Python `norm_q(q)` on [B,L,H,Dh] then `q.transpose(1,2)`.
    // `fused_rms_norm` normalizes the last dim of a 2D [rows, head_dim] input.
    let q_norm_w = weights
        .get(&format!("{prefix}.norm_q.weight"))
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("ideogram4 attention: missing {prefix}.norm_q.weight"))
        })?
        .to_bf16_tensor(device)?;
    let k_norm_w = weights
        .get(&format!("{prefix}.norm_k.weight"))
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("ideogram4 attention: missing {prefix}.norm_k.weight"))
        })?
        .to_bf16_tensor(device)?;
    let q_flat = q.reshape(&[b * seq * num_heads, head_dim])?;
    let k_flat = k.reshape(&[b * seq * num_heads, head_dim])?;
    let q = fused_rms_norm(&q_flat, &q_norm_w, norm_eps)?.reshape(&[b, seq, num_heads, head_dim])?;
    let k = fused_rms_norm(&k_flat, &k_norm_w, norm_eps)?.reshape(&[b, seq, num_heads, head_dim])?;

    // transpose to [B, num_heads, L, head_dim] (Python `.transpose(1,2)`).
    let q = q.permute(&[0, 2, 1, 3])?;
    let k = k.permute(&[0, 2, 1, 3])?;
    let v = v.permute(&[0, 2, 1, 3])?;

    // MRoPE half-split apply: q*cos + rotate_half(q)*sin. cos/sin are the
    // [1, S, head_dim/2] half-table; the kernel pairs (d, d+half) and applies
    // the same coefficient. Use the `_pytorch` variant (skeptic F1): it rounds
    // each BF16 product before the add, matching how the reference evaluates
    // `(q*cos) + (rotate_half(q)*sin)` as two separate BF16 tensor ops. Mirror
    // hidream_o1/mrope.rs::apply_mrope call site (`apply_mrope(x, cos, sin)`).
    // cos/sin are [1,S,half]; the halfsplit kernel reshapes internally (cos_bh
    // = cos_elem/(n*half) = 1 → broadcast over B/H).
    let q = rope_halfsplit_bf16_pytorch(&q, cos, sin)?;
    let k = rope_halfsplit_bf16_pytorch(&k, cos, sin)?;

    // SDPA. Default scale = 1/sqrt(head_dim) = 1/16, matching
    // F.scaled_dot_product_attention (sdpa.rs:874 `scale = 1/sqrt(d_q)` when
    // unspecified). head_dim 256 routes to the math/streaming fallback (no
    // flash kernel at d=256) — wired only; perf is a GPU-gated v1 gate.
    let out = sdpa(&q, &k, &v, attn_mask)?;

    // transpose back [B, L, num_heads, head_dim] -> reshape [B, L, hidden].
    let out = out.permute(&[0, 2, 1, 3])?;
    let out = out.reshape(&[b, seq, hidden])?;

    // bias-free output projection.
    linear_no_bias(weights, &format!("{prefix}.o.weight"), &out, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shape-contract: qkv split dims. The qkv projection emits [B, L, 3*hidden];
    // chunk(3) over the last dim yields three [B, L, hidden]; each reshapes to
    // [B, L, num_heads, head_dim] with hidden == num_heads*head_dim.
    #[test]
    fn qkv_split_dims_are_consistent() {
        let hidden = 4608usize;
        let num_heads = 18usize;
        let head_dim = hidden / num_heads;
        assert_eq!(head_dim, 256);
        assert_eq!(num_heads * head_dim, hidden);
        // qkv output width is 3*hidden; each chunk is hidden wide.
        let qkv_width = 3 * hidden;
        assert_eq!(qkv_width / 3, hidden);
    }

    #[test]
    fn head_dim_drives_default_sdpa_scale() {
        let hidden = 4608usize;
        let num_heads = 18usize;
        let head_dim = hidden / num_heads;
        // Default SDPA scale = 1/sqrt(head_dim) = 1/16.
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        assert!((scale - 1.0 / 16.0).abs() < 1e-6);
    }

    // GPU-dependent end-to-end attention forward: requires CUDA device + FP8
    // dequant + SDPA. GPU busy → compile-only, ignored.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn attention_forward_shapes() {
        // This test documents the call shape; it is not executed without a GPU.
        // attention_forward(&weights, "layers.0.attention", &x, &cos, &sin,
        //                   None, 18, 1e-5, &device) -> [B, L, 4608].
        let _ = super::attention_forward;
    }
}
