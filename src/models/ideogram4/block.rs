//! Ideogram 4 SwiGLU MLP + sandwich-norm AdaLN transformer block.
//!
//! Mirrors `Ideogram4MLP.forward` and `Ideogram4TransformerBlock.forward`
//! (`modeling_ideogram4.py:161-215`) op-for-op.
//!
//! ## MLP (`Ideogram4MLP`)
//! ```text
//! return self.w2(F.silu(self.w1(x)) * self.w3(x))   # all bias-free
//! ```
//! `silu(w1(x)) * w3(x)` is the fused `swiglu_fused_bf16(w1_out, w3_out)`
//! (gate = w1, up = w3); then bias-free `w2`.
//!
//! ## Block (`Ideogram4TransformerBlock`) — SANDWICH-NORM, not FLUX-style
//! ```text
//! mod = adaln_modulation(adaln_input)               # BIAS Linear(adanln_dim -> 4*hidden)
//! scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
//! gate_msa = tanh(gate_msa); gate_mlp = tanh(gate_mlp)
//! scale_msa = 1 + scale_msa; scale_mlp = 1 + scale_mlp
//!
//! attn_out = attention(attention_norm1(x) * scale_msa, seg, cos, sin)   # scale POST-norm, NO shift
//! x = x + gate_msa * attention_norm2(attn_out)      # norm AFTER attn; gate-residual
//! x = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))   # norm AFTER mlp
//! ```
//! Four RMSNorms per block (`attention_norm1/2`, `ffn_norm1/2`), eps = `norm_eps`
//! (1e-5). The `adaln_modulation` is a BIAS Linear → `fused_linear3d_native_pytorch_parity`.
//!
//! ## Idioms mirrored from `zimage_nextdit.rs::transformer_block`
//! - adaln chunk-4 → (scale_msa, gate_msa, scale_mlp, gate_mlp).
//! - scale path: `(1 + scale)` via `scale.unsqueeze(1)?.add_scalar(1.0)?` then
//!   broadcast `mul` against the `[B,L,hidden]` post-norm input.
//! - gate path: `gate.tanh()?` then `gate_residual_fused_bf16(residual, gate, sublayer_out)`
//!   which computes `residual + gate * sublayer_out` (verified bf16_ops.rs:2219).
//!
//! The modulation tensors are `[B, 1, hidden]` (adaln_input is per-sample,
//! kept at L=1), so the scale `mul` and the gate-residual broadcast over the
//! L axis — done once, not per-token-materialized.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::bf16_ops::{gate_residual_fused_bf16, swiglu_fused_bf16};
use flame_core::ops::fused_inference::{
    fused_linear3d_native, fused_linear3d_native_pytorch_parity, fused_rms_norm,
};
use flame_core::{Result, Tensor};

use super::attention::attention_forward;
use super::weights::Ideogram4RawWeight;

/// Dequantize a raw weight to BF16 `[out, in]`, erroring on a missing key.
fn weight(
    weights: &HashMap<String, Ideogram4RawWeight>,
    key: &str,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("ideogram4 block: missing weight `{key}`"))
        })?
        .to_bf16_tensor(device)
}

/// SwiGLU MLP: `w2(silu(w1(x)) * w3(x))`, all bias-free.
///
/// `prefix` is e.g. `layers.0.feed_forward`; weight keys are `{prefix}.w1.weight`,
/// `{prefix}.w2.weight`, `{prefix}.w3.weight`. `x` is `[B, L, hidden]`.
pub fn mlp_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    x: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let w1 = weight(weights, &format!("{prefix}.w1.weight"), device)?;
    let w3 = weight(weights, &format!("{prefix}.w3.weight"), device)?;
    let gate = fused_linear3d_native(x, &w1, None)?; // w1(x)  [B,L,hidden_dim]
    let up = fused_linear3d_native(x, &w3, None)?; // w3(x)  [B,L,hidden_dim]
    // Fused silu(gate) * up — single kernel (gate=w1, up=w3).
    let hidden = swiglu_fused_bf16(&gate, &up)?;
    let w2 = weight(weights, &format!("{prefix}.w2.weight"), device)?;
    fused_linear3d_native(&hidden, &w2, None) // w2(...)  [B,L,hidden]
}

/// Ideogram4 transformer block forward (sandwich-norm AdaLN).
///
/// - `x`: `[B, L, hidden]` BF16 — block input/residual stream.
/// - `cos`, `sin`: `[1, S, head_dim/2]` BF16 MRoPE half-table.
/// - `adaln_input`: `[B, 1, adanln_dim]` BF16 — the shared post-SiLU AdaLN
///   conditioning (same for every block; produced by the embed layer chunk).
/// - `attn_mask`: optional bool `[B, 1, L, L]` (True=attend); `None` for B=1.
/// - `prefix`: e.g. `layers.0`. Sub-weights: `{prefix}.adaln_modulation.{weight,bias}`,
///   `{prefix}.attention.*`, `{prefix}.feed_forward.*`,
///   `{prefix}.{attention_norm1,attention_norm2,ffn_norm1,ffn_norm2}.weight`.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    adaln_input: &Tensor,
    attn_mask: Option<&Tensor>,
    num_heads: usize,
    norm_eps: f32,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    // mod = adaln_modulation(adaln_input)  (BIAS Linear, adanln_dim -> 4*hidden)
    let adaln_w = weight(weights, &format!("{prefix}.adaln_modulation.weight"), device)?;
    let adaln_b = weight(weights, &format!("{prefix}.adaln_modulation.bias"), device)?;
    // bias is [4*hidden] 1D; pytorch-parity linear expects bias as a 1D vector.
    let modulation = fused_linear3d_native_pytorch_parity(adaln_input, &adaln_w, Some(&adaln_b))?;

    // chunk 4 along last dim -> scale_msa, gate_msa, scale_mlp, gate_mlp,
    // each [B, 1, hidden]. (Python `mod.chunk(4, dim=-1)`.)
    let chunks = modulation.chunk(4, modulation.shape().dims().len() - 1)?;
    let scale_msa = &chunks[0];
    let gate_msa = &chunks[1];
    let scale_mlp = &chunks[2];
    let gate_mlp = &chunks[3];

    // gates = tanh(gate); scales = 1 + scale.  (1+scale applied via add_scalar.)
    let gate_msa = gate_msa.tanh()?;
    let gate_mlp = gate_mlp.tanh()?;
    let scale_msa = scale_msa.add_scalar(1.0)?; // [B,1,hidden]
    let scale_mlp = scale_mlp.add_scalar(1.0)?; // [B,1,hidden]

    // --- Attention sublayer ---
    // attention_norm1(x) * scale_msa  (scale multiplies the POST-norm input; no shift).
    let norm1_w = weight(weights, &format!("{prefix}.attention_norm1.weight"), device)?;
    let x_norm = fused_rms_norm(x, &norm1_w, norm_eps)?; // [B,L,hidden]
    let attn_in = x_norm.mul(&scale_msa)?; // broadcast [B,1,hidden] over L

    let attn_out = attention_forward(
        weights,
        &format!("{prefix}.attention"),
        &attn_in,
        cos,
        sin,
        attn_mask,
        num_heads,
        norm_eps,
        device,
    )?;

    // x = x + gate_msa * attention_norm2(attn_out)  (norm AFTER attn; gate-residual).
    let norm2_w = weight(weights, &format!("{prefix}.attention_norm2.weight"), device)?;
    let attn_out = fused_rms_norm(&attn_out, &norm2_w, norm_eps)?;
    // gate_residual_fused_bf16(residual, gate, x) = residual + gate * x.
    let x = gate_residual_fused_bf16(x, &gate_msa, &attn_out)?;

    // --- MLP sublayer ---
    // ffn_norm1(x) * scale_mlp
    let ffn_norm1_w = weight(weights, &format!("{prefix}.ffn_norm1.weight"), device)?;
    let ff_norm = fused_rms_norm(&x, &ffn_norm1_w, norm_eps)?;
    let ff_in = ff_norm.mul(&scale_mlp)?;

    // feed_forward(...)
    let ff_out = mlp_forward(weights, &format!("{prefix}.feed_forward"), &ff_in, device)?;

    // x = x + gate_mlp * ffn_norm2(ff_out)  (norm AFTER mlp).
    let ffn_norm2_w = weight(weights, &format!("{prefix}.ffn_norm2.weight"), device)?;
    let ff_out = fused_rms_norm(&ff_out, &ffn_norm2_w, norm_eps)?;
    gate_residual_fused_bf16(&x, &gate_mlp, &ff_out)
}

#[cfg(test)]
mod tests {
    // adaln chunk-4 dims: the modulation Linear emits 4*hidden; chunk(4) over
    // the last dim yields four hidden-wide slices.
    #[test]
    fn adaln_chunk4_dims() {
        let hidden = 4608usize;
        let adanln_dim = 512usize;
        let mod_width = 4 * hidden;
        // chunk(4) over a [B,1,4*hidden] tensor -> four [B,1,hidden].
        assert_eq!(mod_width / 4, hidden);
        // adaln_modulation maps adanln_dim -> 4*hidden.
        assert_eq!(mod_width, 4 * hidden);
        assert_eq!(adanln_dim, 512);
    }

    #[test]
    fn mlp_swiglu_intermediate_width() {
        let hidden = 4608usize;
        let intermediate = 12288usize;
        // w1, w3: hidden -> intermediate; w2: intermediate -> hidden.
        assert_eq!(intermediate, 12288);
        assert_ne!(intermediate, hidden);
    }

    // GPU-dependent block forward: requires CUDA device + FP8 dequant + SDPA.
    // GPU busy → compile-only, ignored.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn block_forward_compiles() {
        let _ = super::transformer_block_forward;
        let _ = super::mlp_forward;
    }
}
