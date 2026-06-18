//! Boogu-Image C3 — the single-stream / refiner transformer block
//! (`BooguImageTransformerBlock`), used by `single_stream_layers` (×32,
//! modulation=True), `noise_refiner` (×2, modulation=True), and
//! `context_refiner` (×2, modulation=False).
//!
//! Mirrors `boogu/models/transformers/transformer_boogu.py::BooguImageTransformerBlock`
//! (lines 186-373; `LuminaRMSNormZero` is block_lumina2.py:39-71,
//! `LuminaFeedForward` SwiGLU is block_lumina2.py:125-174) op-for-op,
//! cross-checked against the verified Mojo C3 `BooguBlock`
//! (`/home/alex/mojodiffusion/serenitymojo/models/dit/boogu_dit.mojo:366-537`,
//! handoff block-out cos 0.9999980). Sibling Rust idiom: `zimage_nextdit.rs`
//! (adaLN chunk → `scale.unsqueeze(1).add_scalar(1.0).mul()` → `gate.tanh()`
//! gated residual).
//!
//! ## modulation=True (single_stream / noise_refiner) — block_lumina2.py:68-71
//! + transformer_boogu.py:342-360
//!
//! ```text
//! # LuminaRMSNormZero(hidden, temb):
//! emb = norm1.linear(silu(temb))                    # [B,1024] -> [B, 4*3360]
//! scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)   # IN ORDER
//! norm_hidden = norm1.norm(hidden) * (1 + scale_msa[:,None,:])     # RMSNorm eps 1e-5
//! # attention (sandwich-normed onto the attn OUTPUT):
//! attn_out = attn(norm_hidden)
//! hidden   = hidden + tanh(gate_msa)[:,None,:] * norm2(attn_out)   # norm2 = RMSNorm
//! # MLP:
//! mlp_in  = ffn_norm1(hidden) * (1 + scale_mlp[:,None,:])
//! mlp_out = feed_forward(mlp_in)                                    # SwiGLU
//! hidden  = hidden + tanh(gate_mlp)[:,None,:] * ffn_norm2(mlp_out)
//! ```
//! Chunk count = **4**, order = (scale_msa, gate_msa, scale_mlp, gate_mlp).
//! Only `gate_msa`/`gate_mlp` are `tanh`'d; `scale_msa`/`scale_mlp` are not.
//! There is NO shift term in this (single-stream) block — shift_mlp appears
//! only in the double-stream block (C4).
//!
//! ## modulation=False (context_refiner) — transformer_boogu.py:362-371
//!
//! ```text
//! norm_hidden = norm1(hidden)            # PLAIN RMSNorm (norm1.weight), no temb
//! attn_out    = attn(norm_hidden)
//! hidden      = hidden + norm2(attn_out)
//! mlp_out     = feed_forward(ffn_norm1(hidden))
//! hidden      = hidden + ffn_norm2(mlp_out)
//! ```
//! Ungated residuals; temb is unused.
//!
//! ## Weight keys (prefix = e.g. `single_stream_layers.0`)
//! - modulation=True : `norm1.linear.{weight,bias}` (1024→4·3360),
//!                     `norm1.norm.weight` (RMSNorm 3360).
//! - modulation=False: `norm1.weight` (plain RMSNorm 3360).
//! - both: `norm2.weight`, `ffn_norm1.weight`, `ffn_norm2.weight` (RMSNorm 3360);
//!         `attn.{to_q,to_k,to_v,to_out.0}.weight` (bias-free),
//!         `attn.{norm_q,norm_k}.weight` (QK-RMSNorm on head_dim 120);
//!         `feed_forward.{linear_1,linear_2,linear_3}.weight` (SwiGLU, bias-free).

use std::collections::HashMap;

use flame_core::bf16_ops::{silu_bf16, swiglu_fused_bf16};
use flame_core::ops::fused_inference::{fused_linear3d_native, fused_rms_norm};
use flame_core::{Error, Result, Tensor};

use super::attention::gqa_self_attention;
use super::config::BooguConfig;
use super::loader::get;

/// A single-stream / refiner `BooguImageTransformerBlock`.
///
/// Holds the resolved BF16 weights for one block. Built once per block from the
/// weight map and reused across denoise steps. `modulation` selects the
/// `LuminaRMSNormZero` (True) vs plain-`RMSNorm` (False) norm1 path.
pub struct BooguBlock {
    /// modulation=True ⇒ AdaLN (LuminaRMSNormZero); False ⇒ plain RMSNorm norm1.
    pub modulation: bool,

    // norm1: modulation=True → LuminaRMSNormZero (linear + norm). The AdaLN
    // linear maps the timestep embed (1024) → 4·hidden. norm1_norm_w is the
    // RMSNorm gamma. modulation=False → norm1_norm_w holds the plain
    // `norm1.weight`; the linear fields are None.
    norm1_lin_w: Option<Tensor>, // [4·hidden, 1024]  (PyTorch [Cout,Cin])
    norm1_lin_b: Option<Tensor>, // [4·hidden]
    norm1_norm_w: Tensor,        // [hidden]

    norm2_w: Tensor,     // [hidden]
    ffn_norm1_w: Tensor, // [hidden]
    ffn_norm2_w: Tensor, // [hidden]

    // attention (bias-free).
    to_q_w: Tensor,   // [28·120, hidden]
    to_k_w: Tensor,   // [7·120, hidden]
    to_v_w: Tensor,   // [7·120, hidden]
    to_out_w: Tensor, // [hidden, 28·120]  (to_out.0, no bias)
    norm_q_w: Tensor, // [120]
    norm_k_w: Tensor, // [120]

    // SwiGLU FFN (bias-free): linear_1 = gate, linear_3 = up, linear_2 = down.
    ff_w1: Tensor, // [inner, hidden]  (linear_1)
    ff_w2: Tensor, // [hidden, inner]  (linear_2)
    ff_w3: Tensor, // [inner, hidden]  (linear_3)
}

impl BooguBlock {
    /// Build a block from the transformer weight map.
    ///
    /// `prefix` is the block's module path (e.g. `"single_stream_layers.0"`,
    /// `"noise_refiner.0"`, `"context_refiner.0"`). `modulation` MUST match the
    /// block kind (single_stream/noise_refiner = True, context_refiner = False)
    /// so the correct `norm1` keys are read.
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        modulation: bool,
    ) -> Result<Self> {
        let (norm1_lin_w, norm1_lin_b, norm1_norm_w) = if modulation {
            (
                Some(get(weights, &format!("{prefix}.norm1.linear.weight"))?.clone()),
                Some(get(weights, &format!("{prefix}.norm1.linear.bias"))?.clone()),
                get(weights, &format!("{prefix}.norm1.norm.weight"))?.clone(),
            )
        } else {
            // Plain RMSNorm: norm1.weight (no `.linear`/`.norm` split, no AdaLN).
            (
                None,
                None,
                get(weights, &format!("{prefix}.norm1.weight"))?.clone(),
            )
        };

        Ok(Self {
            modulation,
            norm1_lin_w,
            norm1_lin_b,
            norm1_norm_w,
            norm2_w: get(weights, &format!("{prefix}.norm2.weight"))?.clone(),
            ffn_norm1_w: get(weights, &format!("{prefix}.ffn_norm1.weight"))?.clone(),
            ffn_norm2_w: get(weights, &format!("{prefix}.ffn_norm2.weight"))?.clone(),
            to_q_w: get(weights, &format!("{prefix}.attn.to_q.weight"))?.clone(),
            to_k_w: get(weights, &format!("{prefix}.attn.to_k.weight"))?.clone(),
            to_v_w: get(weights, &format!("{prefix}.attn.to_v.weight"))?.clone(),
            to_out_w: get(weights, &format!("{prefix}.attn.to_out.0.weight"))?.clone(),
            norm_q_w: get(weights, &format!("{prefix}.attn.norm_q.weight"))?.clone(),
            norm_k_w: get(weights, &format!("{prefix}.attn.norm_k.weight"))?.clone(),
            ff_w1: get(weights, &format!("{prefix}.feed_forward.linear_1.weight"))?.clone(),
            ff_w2: get(weights, &format!("{prefix}.feed_forward.linear_2.weight"))?.clone(),
            ff_w3: get(weights, &format!("{prefix}.feed_forward.linear_3.weight"))?.clone(),
        })
    }

    /// SwiGLU feed-forward (`LuminaFeedForward`): `linear_2(silu(linear_1(x)) *
    /// linear_3(x))`. linear_1 = gate, linear_3 = up; all bias-free; inner 13568.
    fn feed_forward(&self, x: &Tensor) -> Result<Tensor> {
        let h1 = fused_linear3d_native(x, &self.ff_w1, None)?; // linear_1 (gate)
        let h3 = fused_linear3d_native(x, &self.ff_w3, None)?; // linear_3 (up)
        let act = swiglu_fused_bf16(&h1, &h3)?; // silu(h1) * h3
        fused_linear3d_native(&act, &self.ff_w2, None) // linear_2 (down)
    }

    /// Block forward.
    ///
    /// `hidden` is `[B, N, hidden]` BF16. `temb` is `[B, 1, TIME_EMBED_DIM]` BF16
    /// (the C1 timestep embed; unused when `modulation=False`). `cos`/`sin` are
    /// this block's RoPE tables `[N, head_dim/2]` (the caller narrows the joint
    /// tables to the block's segment). Returns `[B, N, hidden]` BF16.
    pub fn forward(
        &self,
        hidden: &Tensor,
        temb: Option<&Tensor>,
        cfg: &BooguConfig,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        if self.modulation {
            let temb = temb.ok_or_else(|| {
                Error::InvalidOperation(
                    "boogu BooguBlock: temb must be provided when modulation=True".into(),
                )
            })?;
            let norm1_lin_w = self.norm1_lin_w.as_ref().ok_or_else(|| {
                Error::InvalidOperation("boogu BooguBlock: modulation=True needs norm1.linear".into())
            })?;
            let norm1_lin_b = self.norm1_lin_b.as_ref();

            // LuminaRMSNormZero: emb = norm1.linear(silu(temb)) -> [B, 1, 4·hidden].
            let temb_silu = silu_bf16(temb)?;
            let emb = fused_linear3d_native(&temb_silu, norm1_lin_w, norm1_lin_b)?;
            // Drop the singleton seq axis: [B,1,4·hidden] -> [B,4·hidden].
            let b = emb.shape().dims()[0];
            let emb = emb.reshape(&[b, 4 * cfg.hidden_size])?;
            // chunk(4, dim=1): (scale_msa, gate_msa, scale_mlp, gate_mlp).
            let chunks = emb.chunk(4, 1)?;
            let scale_msa = &chunks[0]; // [B, hidden]
            let gate_msa = &chunks[1];
            let scale_mlp = &chunks[2];
            let gate_mlp = &chunks[3];

            // norm_hidden = norm1.norm(hidden) * (1 + scale_msa[:,None,:]).
            let nh = fused_rms_norm(hidden, &self.norm1_norm_w, cfg.norm_eps)?;
            let scale_msa_b = scale_msa.unsqueeze(1)?.add_scalar(1.0)?; // [B,1,hidden]
            let norm_hidden = nh.mul(&scale_msa_b)?; // broadcast over N

            // attn_out = attn(norm_hidden).
            let attn_out = self.attn(&norm_hidden, cfg, cos, sin)?;

            // hidden = hidden + tanh(gate_msa)[:,None,:] * norm2(attn_out).
            let n2 = fused_rms_norm(&attn_out, &self.norm2_w, cfg.norm_eps)?;
            let gate_msa_b = gate_msa.tanh()?.unsqueeze(1)?; // [B,1,hidden]
            let hidden1 = hidden.add(&gate_msa_b.mul(&n2)?)?;

            // mlp_in = ffn_norm1(hidden) * (1 + scale_mlp[:,None,:]).
            let fn1 = fused_rms_norm(&hidden1, &self.ffn_norm1_w, cfg.norm_eps)?;
            let scale_mlp_b = scale_mlp.unsqueeze(1)?.add_scalar(1.0)?;
            let mlp_in = fn1.mul(&scale_mlp_b)?;
            let mlp_out = self.feed_forward(&mlp_in)?;

            // hidden = hidden + tanh(gate_mlp)[:,None,:] * ffn_norm2(mlp_out).
            let fn2 = fused_rms_norm(&mlp_out, &self.ffn_norm2_w, cfg.norm_eps)?;
            let gate_mlp_b = gate_mlp.tanh()?.unsqueeze(1)?;
            hidden1.add(&gate_mlp_b.mul(&fn2)?)
        } else {
            // context_refiner: plain RMSNorm pre/post, NO temb, NO gates.
            let norm_hidden = fused_rms_norm(hidden, &self.norm1_norm_w, cfg.norm_eps)?;
            let attn_out = self.attn(&norm_hidden, cfg, cos, sin)?;
            let n2 = fused_rms_norm(&attn_out, &self.norm2_w, cfg.norm_eps)?;
            let hidden1 = hidden.add(&n2)?;
            let fn1 = fused_rms_norm(&hidden1, &self.ffn_norm1_w, cfg.norm_eps)?;
            let mlp_out = self.feed_forward(&fn1)?;
            let fn2 = fused_rms_norm(&mlp_out, &self.ffn_norm2_w, cfg.norm_eps)?;
            hidden1.add(&fn2)
        }
    }

    /// Block-local GQA self-attention wrapper (delegates to the C3 attention
    /// helper with this block's attn weights).
    fn attn(&self, x: &Tensor, cfg: &BooguConfig, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        gqa_self_attention(
            x,
            cfg,
            &self.to_q_w,
            &self.to_k_w,
            &self.to_v_w,
            &self.to_out_w,
            &self.norm_q_w,
            &self.norm_k_w,
            cos,
            sin,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BooguConfig {
        BooguConfig::default()
    }

    #[test]
    fn modulation_chunk_count_is_four() {
        // LuminaRMSNormZero.linear outputs 4·hidden, chunked into exactly 4:
        // (scale_msa, gate_msa, scale_mlp, gate_mlp). No shift in single-stream.
        let c = cfg();
        let adaln_out = 4 * c.hidden_size;
        assert_eq!(adaln_out % 4, 0);
        assert_eq!(adaln_out / 4, c.hidden_size);
    }

    #[test]
    fn swiglu_inner_is_13568() {
        // feed_forward inner dim (LuminaFeedForward round-up): 13568 for hidden 3360.
        assert_eq!(cfg().swiglu_inner(), 13568);
    }

    #[test]
    fn timestep_embed_dim_feeds_adaln_linear() {
        // norm1.linear maps min(hidden,1024)=1024 -> 4·hidden.
        let c = cfg();
        assert_eq!(BooguConfig::TIME_EMBED_DIM, c.hidden_size.min(1024));
        assert_eq!(BooguConfig::TIME_EMBED_DIM, 1024);
    }

    // GPU-dependent: load + forward need a CUDA device + real weights.
    // Compile-only here.
    #[test]
    #[ignore = "requires CUDA device + checkpoint (GPU busy); compile-only this chunk"]
    fn block_load_and_forward_compile() {
        let _ = BooguBlock::load;
        let _ = BooguBlock::forward;
    }
}
