//! Gemma-4 decoder layer.
//!
//! Reference: `modeling_gemma4.py` lines 1126-1413 (TextAttention +
//! TextDecoderLayer). Each layer is:
//!
//! ```text
//! residual = x
//! x = input_layernorm(x)                       // RMSNorm
//! x = self_attention(x, kv_cache[layer])       // GQA + RoPE (dual theta)
//! x = post_attention_layernorm(x)              // RMSNorm
//! x = residual + x
//! residual = x
//! x = pre_feedforward_layernorm(x)             // RMSNorm
//! x = mlp(x)                                   // SwiGLU(gelu_pytorch_tanh)
//! x = post_feedforward_layernorm(x)            // RMSNorm
//! x = residual + x
//! ```
//!
//! Note Gemma-4's four-RMSNorm-per-layer pattern (pre+post on each
//! sublayer) is the same as Gemma-3 — different from vanilla Llama's
//! two RMSNorms.

use crate::models::gemma4::kv_cache::Gemma4LayerCache;
use crate::models::gemma4::{Gemma4Config, LayerType};
use flame_core::{Result, Tensor};

// AGENT-DEFAULT: Concrete primitive choices for this layer
// ---------------------------------------------------------
//   - Linear:   flame_core::ops::fused_inference::fused_linear3d_native
//               (weight kept in native PyTorch [Cout,Cin] layout; cuBLASLt
//               transposes inside the GEMM. No pre-transpose at every call,
//               per flame-core TENET 1 perf landmines.)
//   - RMSNorm:  flame_core::cuda_ops_bf16::rms_norm_bf16(x, Some(&w), eps)
//               (canonical BF16 RMSNorm; Gemma-4 uses plain Llama-style
//               `norm(x) * weight`, NOT Gemma-3's `* (1+weight)`. The
//               kernel reduces in F32 internally so PyTorch parity is
//               within BF16 tolerance.)
//   - RoPE:     flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
//               (halfsplit layout matches HF Gemma-4 `apply_rotary_pos_emb`:
//               cos = cat(freqs, freqs), and rotate_half does the front-back
//               swap. cos/sin tables are `[1,1,N,head_dim/2]` BF16.)
//   - Partial RoPE for Full layers: handled at the table level by the
//     caller (model.rs builds proportional inv_freq with zeros for the
//     non-rotated trailing dims → cos=1, sin=0 there → identity rotation).
//     No kernel-level partial handling needed in this file.
//   - GELU:     Tensor::gelu() routes to `gelu_bf16_contig_direct` which
//               uses the tanh-approx form — that IS `gelu_pytorch_tanh`,
//               so `gate.gelu().mul(&up)` is the correct SwiGLU.
//   - SDPA:     flame_core::attention::sdpa(q, k, v, Some(&mask)) or None
//               for decode. head_dim=256 means cuDNN's BF16 SDPA does not
//               apply; we land in the BF16 fallback path. GQA is NOT native
//               — we repeat KV from num_kv_heads to num_q_heads before the
//               call.
//
// API CHANGE (AGENT-DEFAULT): forward() now takes `&Gemma4Config` so the
// layer knows num_q_heads / num_kv_heads / head_dim / rms_norm_eps without
// having to infer them from tensor shapes. Builder 1's model.rs will pass
// it when invoking each layer.

/// One Gemma-4 decoder block. Owns four RMSNorm weights, Q/K/V/O
/// projections, gate/up/down MLP projections. The block_offload
/// integration brings these into GPU memory on demand at forward.
pub struct Gemma4DecoderLayer {
    /// Layer index (0..60). Determines whether this layer uses sliding
    /// or full attention via `cfg.layer_types[layer_idx]`.
    pub layer_idx: usize,
    /// Cached layer-type bit so we don't re-index the config vector.
    pub layer_type: LayerType,
    /// Per-layer head_dim × rotary span; see `Gemma4Config::partial_rotary_factor_full`.
    /// Sliding layers rotate all `head_dim` dims; full layers rotate
    /// only the first `(head_dim * 0.25).floor() * 2` dims (the rest
    /// of Q/K passes through unchanged).
    pub rotary_dim: usize,

    // ── Weights (loaded on demand via BlockOffloader) ────────────────
    /// `input_layernorm.weight` — RMSNorm on attention input. `[hidden_size]`.
    pub input_norm_w: Tensor,
    /// `post_attention_layernorm.weight`. `[hidden_size]`.
    pub post_attention_norm_w: Tensor,
    /// `pre_feedforward_layernorm.weight`. `[hidden_size]`.
    pub pre_ffn_norm_w: Tensor,
    /// `post_feedforward_layernorm.weight`. `[hidden_size]`.
    pub post_ffn_norm_w: Tensor,

    /// Q projection `[num_q_heads * head_dim, hidden_size]` = `[8192, 5376]` for 31B.
    pub q_proj_w: Tensor,
    /// K projection `[num_kv_heads * head_dim, hidden_size]` = `[4096, 5376]`.
    pub k_proj_w: Tensor,
    /// V projection `[num_kv_heads * head_dim, hidden_size]` = `[4096, 5376]`.
    /// (config has `attention_k_eq_v=true` but the safetensors still ship K and V
    /// separately at 31B; the flag is for tied-weight variants. Confirm at load.)
    pub v_proj_w: Tensor,
    /// O projection `[hidden_size, num_q_heads * head_dim]` = `[5376, 8192]`.
    pub o_proj_w: Tensor,

    /// SwiGLU gate projection `[intermediate_size, hidden_size]` = `[21504, 5376]`.
    pub gate_proj_w: Tensor,
    /// SwiGLU up projection `[intermediate_size, hidden_size]`.
    pub up_proj_w: Tensor,
    /// Down projection `[hidden_size, intermediate_size]` = `[5376, 21504]`.
    pub down_proj_w: Tensor,

    // ── Q/K/V per-head RMSNorm + per-layer scalar (added per coordinator
    //    URGENT correction 2026-05-21 after Builder 1 re-verified the
    //    Gemma4TextAttention class). Norms have weight `[head_dim]`=`[256]`;
    //    v_norm is scale-free (with_scale=False in Python) so the loader
    //    may place a zero/ones placeholder there — we treat it as
    //    "no weight" by ignoring this field's value and passing None to
    //    rms_norm_bf16 in the forward. ────────────────────────────────
    /// `self_attn.q_norm.weight` — `[head_dim]` BF16. Always applied.
    pub q_norm_w: Tensor,
    /// `self_attn.k_norm.weight` — `[head_dim]` BF16. Applied unless this
    /// is a KV-sharing layer (`is_kv_shared_layer=True`). 31B's text
    /// decoder is dense with `num_kv_shared_layers=0`, so always applied
    /// in the current target.
    pub k_norm_w: Tensor,
    /// `self_attn.v_norm.weight` — `[head_dim]` BF16. Per Python source
    /// `v_norm = Gemma4RMSNorm(head_dim, with_scale=False)`, this is a
    /// scale-free RMSNorm. The forward treats v_norm as weightless
    /// (passes `None` to `rms_norm_bf16`) regardless of what the loader
    /// stashed in this slot — the field exists only so the weight loader
    /// doesn't have an orphan tensor to drop. If a future variant ships
    /// `with_scale=True` for v_norm, switch the forward to
    /// `Some(&self.v_norm_w)`.
    pub v_norm_w: Tensor,
    /// `layer_scalar` buffer — shape `[1]` BF16. Applied via a scalar
    /// multiply at the very end of the layer forward (after the FFN
    /// residual add). Gemma-4's per-layer learned-scale-of-output knob.
    pub layer_scalar: Tensor,
}

impl Gemma4DecoderLayer {
    /// Run one decoder layer.
    ///
    /// `x`: `[B, S, hidden_size]` BF16 (for prefill) or `[B, 1, hidden_size]` BF16 (decode).
    /// `kv_cache`: layer's mutable KV cache. Updated in place with the
    /// new K/V columns from this forward.
    /// `rope_cos`, `rope_sin`: layer-appropriate RoPE tables for THIS
    /// layer's rotary_dim (sliding → full head_dim; full → partial).
    /// `attention_mask`: optional precomputed causal/sliding mask.
    /// During decode (`S=1`) we don't need a mask — single-query
    /// attention attends to all cached keys freely.
    pub fn forward(
        &self,
        x: &Tensor,
        cfg: &Gemma4Config,
        kv_cache: &mut Gemma4LayerCache,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Reference: modeling_gemma4.py Gemma4TextDecoderLayer.forward
        // (lines 1325-1413, dense-no-MoE path) and Gemma4TextAttention.forward
        // (lines 1185-1246, branch where is_kv_shared_layer=False and
        // v_proj is not None — which is the case for 31B dense).
        //
        // CORRECTION 2026-05-21 (coordinator URGENT): the 31B variant DOES
        // ship q_norm, k_norm, v_norm and a per-layer `layer_scalar` buffer.
        // q_norm/k_norm/v_norm are per-head RMSNorms (`[head_dim]` weights)
        // applied to the reshape `[B, H, S, head_dim]` form BEFORE RoPE.
        // v_norm uses with_scale=False in Python → we pass None for its
        // weight regardless of what the loader put in `v_norm_w` (see field
        // docstring). `layer_scalar` is a `[1]` BF16 buffer multiplied into
        // the final hidden output AFTER the FFN residual add.

        let h_q = cfg.num_attention_heads;
        let h_kv = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        let n_rep = h_q / h_kv; // GQA repeat factor (2 for 31B)

        let in_dims = x.shape().dims().to_vec();
        if in_dims.len() != 3 {
            return Err(flame_core::Error::InvalidInput(format!(
                "Gemma4DecoderLayer::forward: x must be [B, S, hidden], got {:?}",
                in_dims
            )));
        }
        let b = in_dims[0];
        let s = in_dims[1];

        // ── Self-attention sub-block ────────────────────────────────────
        let residual = x.clone();

        // 1. input_layernorm (RMSNorm; Gemma-4 = plain Llama-style * w)
        let x1 = flame_core::cuda_ops_bf16::rms_norm_bf16(
            x,
            Some(&self.input_norm_w),
            cfg.rms_norm_eps,
        )?;

        // 2. Q/K/V projections via cuBLASLt fused linear3d (no pre-transpose;
        //    weight is the safetensors-native [Cout, Cin] layout).
        let q = flame_core::ops::fused_inference::fused_linear3d_native(
            &x1, &self.q_proj_w, None,
        )?;
        let k_new = flame_core::ops::fused_inference::fused_linear3d_native(
            &x1, &self.k_proj_w, None,
        )?;
        let v_new = flame_core::ops::fused_inference::fused_linear3d_native(
            &x1, &self.v_proj_w, None,
        )?;

        // 3. Reshape to multi-head: [B, S, h*d] -> [B, h, S, d]
        //    Q has h_q heads, K/V have h_kv heads (GQA).
        let q = q
            .reshape(&[b, s, h_q, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;
        let k_new = k_new
            .reshape(&[b, s, h_kv, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;
        let v_new = v_new
            .reshape(&[b, s, h_kv, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;

        // 3b. Per-head Q/K/V RMSNorm (Gemma-4-specific; Python lines 1200,
        //     1216, 1220). RMSNorm reduces along the LAST dim only, so the
        //     `[B,H,S,d]` ordering vs Python's `[B,S,H,d]` ordering doesn't
        //     matter — we just need to flatten everything except the last
        //     dim before calling the BF16 RMSNorm primitive, then reshape
        //     back. v_norm has `with_scale=False` in Python so we pass
        //     `None` as the weight even though `self.v_norm_w` exists.
        let q = per_head_rms_norm(&q, Some(&self.q_norm_w), cfg.rms_norm_eps)?;
        let k_new = per_head_rms_norm(&k_new, Some(&self.k_norm_w), cfg.rms_norm_eps)?;
        let v_new = per_head_rms_norm(&v_new, None, cfg.rms_norm_eps)?;

        // 4. Apply RoPE to Q and K. For Sliding layers the cos/sin tables
        //    cover the full head_dim (theta=10K). For Full layers the
        //    proportional-RoPE tables have the trailing (head_dim/2 - rotated)
        //    positions zeroed in inv_freq, so cos=1/sin=0 there → identity
        //    on the non-rotated dims. No partial-rotary kernel needed.
        //    V is NOT rotated (apply_rotary_pos_emb is only called on Q, K).
        //
        //    cos/sin layout expected by rope_halfsplit_bf16:
        //      [1, 1, S, head_dim/2] BF16 (matches HF emb=cat(freqs,freqs)
        //      halved by the kernel internally).
        let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, rope_cos, rope_sin)?;
        let k_rotated = flame_core::bf16_ops::rope_halfsplit_bf16(&k_new, rope_cos, rope_sin)?;

        // Silence unused-field warning on layer_type while making intent
        // explicit. The layer_type/rotary_dim choice is encoded in the
        // cos/sin tables passed in by the caller; this layer's body is
        // identical for Sliding and Full.
        let _layer_type = self.layer_type;
        let _rotary_dim = self.rotary_dim;
        // Reference attention_k_eq_v so the field's intent is visible.
        // The branch is currently disabled (see field docstring); separate
        // v_proj is used unconditionally for the 31B dense decoder.
        let _ = cfg.attention_k_eq_v;

        // 5. Append rotated K (and unrotated-by-RoPE V) to the per-layer cache.
        //    PREFILL: writes S new columns. DECODE: writes 1 new column.
        kv_cache.append(&k_rotated, &v_new)?;
        // 6. Fetch the active K/V slice (full layers: all valid; sliding:
        //    only last `window` columns, ring-buffer-rolled if overflowed).
        let (k_full, v_full) = kv_cache.fetch()?;

        // 7. GQA: SDPA expects matching head counts on Q and K/V, so repeat
        //    KV from h_kv to h_q heads (factor n_rep, 2 for 31B).
        let k_full = repeat_kv(&k_full, n_rep)?;
        let v_full = repeat_kv(&v_full, n_rep)?;

        // 8. Scaled dot-product attention. flame_core::attention::sdpa applies
        //    `1/sqrt(head_dim)` scaling internally. head_dim=256 means cuDNN's
        //    BF16 SDPA does NOT apply (only 64/96/128); we land in the BF16
        //    fallback. Per the prompt we use the explicit-mask path even for
        //    Sliding-causal-during-prefill, NOT sdpa_prefix_causal_full
        //    (which is documented as broken under multi-layer cascade today).
        //
        //    During decode (S=1) the caller passes mask=None: a single new
        //    query attends to all in-window cached keys with no causal
        //    restriction. During prefill the caller passes a precomputed
        //    [1, 1, S, K_seq] BF16 keep-mask (1.0 = attend, 0.0 = masked).
        let attn = flame_core::attention::sdpa(&q, &k_full, &v_full, attention_mask)?;

        // 9. Merge heads: [B, h_q, S, d] -> [B, S, h_q*d]
        let attn = attn
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, s, h_q * d])?;

        // 10. O projection
        let attn = flame_core::ops::fused_inference::fused_linear3d_native(
            &attn,
            &self.o_proj_w,
            None,
        )?;

        // 11. post_attention_layernorm + residual.
        //     (Gemma-4 norms AFTER each sublayer, then adds residual —
        //     the four-RMSNorm-per-layer pattern.)
        let attn = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &attn,
            Some(&self.post_attention_norm_w),
            cfg.rms_norm_eps,
        )?;
        let x = residual.add(&attn)?;

        // ── Feed-forward sub-block ──────────────────────────────────────
        let residual = x.clone();

        // 12. pre_feedforward_layernorm
        let x2 = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &x,
            Some(&self.pre_ffn_norm_w),
            cfg.rms_norm_eps,
        )?;

        // 13. SwiGLU with GELU-tanh (NOT SiLU). Tensor::gelu() routes to
        //     bf16_ops::gelu_bf16_contig_direct, which uses the tanh
        //     approximation — that is exactly `gelu_pytorch_tanh`.
        let gate = flame_core::ops::fused_inference::fused_linear3d_native(
            &x2,
            &self.gate_proj_w,
            None,
        )?;
        let up = flame_core::ops::fused_inference::fused_linear3d_native(
            &x2,
            &self.up_proj_w,
            None,
        )?;
        let ff = gate.gelu()?.mul(&up)?;

        // 14. Down projection
        let ff = flame_core::ops::fused_inference::fused_linear3d_native(
            &ff,
            &self.down_proj_w,
            None,
        )?;

        // 15. post_feedforward_layernorm + residual
        let ff = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &ff,
            Some(&self.post_ffn_norm_w),
            cfg.rms_norm_eps,
        )?;
        let out = residual.add(&ff)?;

        // 16. Per-layer learned scalar (Gemma-4 `layer_scalar` buffer,
        //     shape `[1]`). Python line 1410: `hidden_states *= layer_scalar`.
        //     AGENT-DEFAULT: read the scalar host-side once per layer call
        //     (single-float DtoH, ~µs) and use `mul_scalar` to avoid
        //     materializing a broadcast `[B,S,H]` buffer of the same value.
        //     A future optimization could cache the f32 in the struct at
        //     load time, but `layer_scalar` is a buffer (not Parameter) so
        //     in theory it could be model-config-modified between calls;
        //     do the safe thing per layer for now.
        let scalar_host = self.layer_scalar.to_dtype(flame_core::DType::F32)?.to_vec_f32()?;
        let scalar = scalar_host
            .first()
            .copied()
            .ok_or_else(|| flame_core::Error::InvalidInput(
                "Gemma4DecoderLayer: layer_scalar buffer was empty".into(),
            ))?;
        if (scalar - 1.0).abs() < f32::EPSILON {
            // Fast path: 31B ships `layer_scalar = [1.0]` (it's a ones-init
            // buffer); skip the no-op multiply.
            Ok(out)
        } else {
            out.mul_scalar(scalar)
        }
    }

    /// Build the RoPE rotation dimension for this layer.
    /// Sliding layers rotate the full `head_dim`; full layers rotate
    /// `floor(head_dim * partial_rotary_factor) * 2` (matching the
    /// PyTorch convention where partial means "rotate the first
    /// half-pair-count of dims").
    pub fn rotary_dim_for(cfg: &Gemma4Config, layer_idx: usize) -> usize {
        match cfg.layer_types[layer_idx] {
            LayerType::Sliding => cfg.head_dim,
            LayerType::Full => {
                let pairs = ((cfg.head_dim as f32) * cfg.partial_rotary_factor_full / 2.0)
                    .floor() as usize;
                pairs * 2
            }
        }
    }
}

// ───────────────────────── helpers ─────────────────────────────────────
//
// Note on GELU: `gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) *
// (x + 0.044715 * x^3)))`. In flame-core, `Tensor::gelu()` routes to
// `bf16_ops::gelu_bf16_contig_direct`, which the kernel source
// (`bf16_ops.rs::CUDA_GELU` lines 32-58) confirms uses the tanh
// approximation (constant `0.7978845608 = sqrt(2/π)`, calls `tanhf`).
// The forward calls `gate.gelu()?.mul(&up)?` — this IS the
// `gelu_pytorch_tanh` SwiGLU. The exact-erf variant is a separate
// `gelu_exact_bf16_contig_direct` that we do NOT use for Gemma-4.

/// Per-head RMSNorm for an `[B, H, S, head_dim]` Q/K/V tensor.
///
/// flame-core's `cuda_ops_bf16::rms_norm_bf16` normalizes along the LAST
/// dim regardless of the rank of the input — but it expects the weight
/// to broadcast over all non-last dims. To keep both contracts crisp,
/// we flatten to `[B*H*S, head_dim]`, call the canonical RMSNorm, then
/// reshape back. The norm value at each (b, h, s, :) row is identical
/// whether you do this flatten-first approach or apply norm on the
/// 4-D tensor directly — RMS reduction is purely over `head_dim`.
///
/// `weight=None` ⇒ scale-free RMSNorm (matches Gemma-4's
/// `v_norm = Gemma4RMSNorm(..., with_scale=False)`).
fn per_head_rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidInput(format!(
            "per_head_rms_norm: expected [B,H,S,head_dim], got {:?}",
            dims
        )));
    }
    let head_dim = dims[3];
    let flat: usize = dims[..3].iter().product();
    let x2 = x.reshape(&[flat, head_dim])?;
    let n = flame_core::cuda_ops_bf16::rms_norm_bf16(&x2, weight, eps)?;
    n.reshape(&dims)
}

/// Grouped-query-attention KV broadcast: `[B, H_kv, S, D]` → `[B, H_kv*n_rep, S, D]`.
///
/// SDPA in flame-core requires Q and K/V to have matching head counts.
/// For Gemma-4-31B (Q heads 32, KV heads 16) we duplicate every KV
/// head `n_rep = 2` times to match Q. Implementation mirrors the
/// existing `gemma3_encoder.rs::repeat_kv` (stack + reshape) — a future
/// flame-core primitive could fuse this into the SDPA call, but for
/// now the stack-reshape is cheap (BF16 contiguous copy).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidInput(format!(
            "repeat_kv: expected [B,H_kv,S,D], got {:?}",
            dims
        )));
    }
    let b = dims[0];
    let h_kv = dims[1];
    let s = dims[2];
    let d = dims[3];

    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    let stacked = Tensor::stack(&copies, 2)?; // [B, H_kv, n_rep, S, D]
    stacked.reshape(&[b, h_kv * n_rep, s, d])
}
