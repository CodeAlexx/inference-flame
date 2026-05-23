//! GPT-OSS decoder layer (Lens M2 Stage D5a).
//!
//! Single-layer assembly for the microsoft/Lens text encoder. Combines:
//!
//! - YaRN-scaled RoPE (`gpt_oss_rope::GptOssRotaryEmbedding`, D2)
//! - Sliding-window causal mask (`flame_core::attention::sliding_window_causal_keep_mask`, D3)
//! - SDPA with attention sinks (`flame_core::sdpa::forward_with_sinks`, D3)
//! - Token-choice top-K routing (`flame_core::ops::token_choice_routing`, D4)
//! - Grouped MoE expert FFN (gate_up @ down) via `grouped_mm_bf16`
//! - GPT-OSS-specific clamped GELU-style activation: `gated = (up + 1) * gate * sigmoid(alpha * gate)`
//!
//! **Scope**: One [`GptOssLayer`] with synthetic-weight smoke tests.
//! 24-layer stack assembly, weight loading, real-checkpoint parity, and
//! the lens_infer.rs wiring all happen in stages D5b / D6 / D7.
//!
//! ## Reference (transformers)
//!
//! `transformers/models/gpt_oss/modeling_gpt_oss.py`:
//!   - `GptOssRMSNorm` (line 47-64) → `flame_core::cuda_ops_bf16::rms_norm_bf16`
//!   - `GptOssAttention.forward` (line 300-343)
//!   - `eager_attention_forward` (line 241-270) → `forward_with_sinks`
//!   - `GptOssTopKRouter.forward` (line 152-158) → `token_choice_route(..., TopKSoftmax)`
//!   - `GptOssExperts.forward` (line 81-140), specifically the inference branch
//!     starting at line 127 which does the bmm-over-all-experts repeat pattern.
//!     We use the **looped expert** semantics (the for-loop branch at line 108) —
//!     same math, but our `grouped_mm_bf16` is the kernel equivalent.
//!   - `GptOssDecoderLayer.forward` (line 357-388): pre-norm, residual → norm →
//!     self-attn → residual → norm → MLP → residual.

use crate::models::gpt_oss_rope::GptOssRotaryEmbedding;

use flame_core::{
    attention::sliding_window_causal_keep_mask,
    bf16_ops::rope_halfsplit_bf16,
    bf16_clamp::clamp_bf16,
    cuda_ops_bf16::rms_norm_bf16,
    ops::{
        fused_inference::{dequant_mxfp4_to_bf16, fused_linear3d_native},
        grouped_mm::grouped_mm_bf16,
        token_choice_routing::{
            permute_tokens_for_token_choice, token_choice_route, ScoreMode,
        },
        fused_gated_scatter_add::fused_gated_scatter_add_bf16,
    },
    sdpa::forward_with_sinks,
    CudaDevice, DType, Error, Result, Shape, Tensor,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Layer attention type. Alternates per layer in the Lens GPT-OSS encoder.
///
/// Reference: `config.layer_types` in `microsoft/Lens/text_encoder/config.json`
/// — a 24-entry array of `"sliding_attention"` / `"full_attention"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Sliding-window causal attention (window=128 for Lens).
    Sliding,
    /// Full causal attention.
    Full,
}

impl LayerType {
    /// Parse the transformers JSON value into a `LayerType`.
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "sliding_attention" => Ok(LayerType::Sliding),
            "full_attention" => Ok(LayerType::Full),
            other => Err(Error::InvalidInput(format!(
                "GPT-OSS layer_type must be 'sliding_attention' or 'full_attention', got {other:?}"
            ))),
        }
    }
}

/// Full GPT-OSS encoder configuration.
///
/// All defaults match `microsoft/Lens/text_encoder/config.json`.
///
/// This struct is the single source of truth — the rotary-embedding module
/// (`gpt_oss_rope`) re-exports it for backwards compatibility but the canonical
/// home is here.
#[derive(Debug, Clone)]
pub struct GptOssConfig {
    // ---- transformer shape ----
    pub hidden_size: usize,        // 2880
    pub num_hidden_layers: usize,  // 24
    pub num_attention_heads: usize, // 64
    pub num_key_value_heads: usize, // 8 (GQA group ratio 8)
    pub head_dim: usize,           // 64
    pub intermediate_size: usize,  // 2880 (per-expert FFN hidden)
    pub vocab_size: usize,         // 201_088

    // ---- MoE ----
    pub num_local_experts: usize, // 32
    pub experts_per_token: usize, // 4 (top-K)

    // ---- attention windowing ----
    pub layer_types: Vec<LayerType>, // length = num_hidden_layers
    pub sliding_window: usize,       // 128

    // ---- norms / activations ----
    pub rms_norm_eps: f32, // 1e-5
    pub swiglu_limit: f32, // 7.0
    pub swiglu_alpha: f32, // 1.702 (GELU-exact constant)

    // ---- biases ----
    pub attention_bias: bool, // true — Q/K/V/O all have biases

    // ---- RoPE (YaRN) ----
    pub rope_theta: f64,              // 150_000
    pub rope_factor: f64,             // 32.0
    pub rope_beta_fast: f64,          // 32.0
    pub rope_beta_slow: f64,          // 1.0
    pub rope_original_max_pos: usize, // 4096
    pub rope_truncate: bool,          // false for Lens

    // ---- Lens-specific consumer ----
    /// 0-indexed decoder layers whose post-residual hidden states feed into
    /// the LensDiT cross-attention. For Lens: `[5, 11, 17, 23]`.
    pub selected_layer_index: Vec<usize>,
}

impl GptOssConfig {
    /// Default = `microsoft/Lens` text encoder.
    ///
    /// The `layer_types` field is initialised from the canonical 24-entry
    /// pattern in `microsoft/Lens/text_encoder/config.json`. Per the source:
    /// `["sliding_attention", "full_attention", "sliding_attention", ...]` —
    /// i.e. odd layers are full, even layers are sliding (0-indexed). This
    /// matches the GPT-OSS-20B published config.
    pub fn lens_default() -> Self {
        let layer_types = (0..24)
            .map(|i| {
                if i % 2 == 0 {
                    LayerType::Sliding
                } else {
                    LayerType::Full
                }
            })
            .collect();
        Self {
            hidden_size: 2880,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 64,
            intermediate_size: 2880,
            vocab_size: 201_088,
            num_local_experts: 32,
            experts_per_token: 4,
            layer_types,
            sliding_window: 128,
            rms_norm_eps: 1e-5,
            swiglu_limit: 7.0,
            swiglu_alpha: 1.702,
            attention_bias: true,
            rope_theta: 150_000.0,
            rope_factor: 32.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_original_max_pos: 4096,
            rope_truncate: false,
            selected_layer_index: vec![5, 11, 17, 23],
        }
    }
}

impl Default for GptOssConfig {
    fn default() -> Self {
        Self::lens_default()
    }
}

// ---------------------------------------------------------------------------
// Pre-built attention mask cache (cos/sin live in GptOssRotaryEmbedding)
// ---------------------------------------------------------------------------

/// Mask kit built once per forward, reused across all layers.
///
/// The caller (or the stack — D5b) builds this once for the current `seq_len`
/// and hands it down to each `GptOssLayer::forward`, which picks the right
/// mask based on its layer type.
#[derive(Debug)]
pub struct GptOssMaskKit {
    pub full_mask: Tensor,    // [1, 1, S, S] BF16 keep-mask
    pub sliding_mask: Tensor, // [1, 1, S, S] BF16 keep-mask
}

impl GptOssMaskKit {
    /// Build both masks for a given sequence length and sliding window.
    pub fn build(
        seq_len: usize,
        sliding_window: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // `causal_keep_mask` in flame-core is private; per the D3 report,
        // `sliding_window_causal_keep_mask` with `window_size >= seq_len`
        // produces a pure lower-triangular causal mask (each row q attends to
        // [0..=q]). We use that as the "full" mask. Tested in flame-core's
        // `sliding_window_pure_causal_at_short_seq`.
        let full_mask =
            sliding_window_causal_keep_mask(seq_len, seq_len, device, DType::BF16)?;
        let sliding_mask =
            sliding_window_causal_keep_mask(seq_len, sliding_window, device, DType::BF16)?;
        Ok(Self {
            full_mask,
            sliding_mask,
        })
    }
}

// ---------------------------------------------------------------------------
// GptOssAttention
// ---------------------------------------------------------------------------

/// Self-attention block (GQA with attention sinks + sliding window).
///
/// Reference: `transformers/models/gpt_oss/modeling_gpt_oss.py::GptOssAttention`.
///
/// All linear weights are stored in **PyTorch layout** `[Cout, Cin]` (no
/// pre-transpose). Multiplication uses `fused_linear3d_native` which performs
/// the `TRANSA=T` inside cuBLASLt.
pub struct GptOssAttention {
    /// Q projection weight: `[num_heads * head_dim, hidden] = [4096, 2880]`.
    pub q_weight: Tensor,
    pub q_bias: Tensor, // `[num_heads * head_dim]`
    /// K projection: `[num_kv_heads * head_dim, hidden] = [512, 2880]`.
    pub k_weight: Tensor,
    pub k_bias: Tensor, // `[num_kv_heads * head_dim]`
    /// V projection: same shape as K.
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    /// O projection: `[hidden, num_heads * head_dim] = [2880, 4096]`.
    pub o_weight: Tensor,
    pub o_bias: Tensor, // `[hidden]`
    /// Per-head attention sink logits. Shape `[num_heads]`, BF16.
    pub sinks: Tensor,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub layer_type: LayerType,
    /// `1 / sqrt(head_dim)` — kept for documentation, not used (SDPA handles scaling).
    pub scaling: f32,
}

impl GptOssAttention {
    /// Run one self-attention block.
    ///
    /// Shapes:
    /// - `x`:        `[B, S, hidden]` BF16
    /// - `cos`, `sin`: `[S, head_dim/2]` BF16 (or any shape reshapeable to
    ///   `[1, 1, S, head_dim/2]` for the half-split kernel)
    /// - `full_mask`, `sliding_mask`: `[1, 1, S, S]` BF16 keep-masks.
    ///
    /// Output: `[B, S, hidden]` BF16.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        full_mask: Option<&Tensor>,
        sliding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let xd = x.shape().dims().to_vec();
        if xd.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "GptOssAttention.forward: x must be [B, S, hidden], got {xd:?}"
            )));
        }
        let (b, s, hidden) = (xd[0], xd[1], xd[2]);
        let h = self.num_heads;
        let h_kv = self.num_kv_heads;
        let d = self.head_dim;
        let n_rep = h / h_kv;
        if h % h_kv != 0 {
            return Err(Error::InvalidInput(format!(
                "GptOssAttention: num_heads={h} must be divisible by num_kv_heads={h_kv}"
            )));
        }

        // ---- Q / K / V projections (BF16 cuBLASLt w/ bias fused) ----
        // Weight layout: `[Cout, Cin]` (PyTorch native) → fused_linear3d_native.
        let q = fused_linear3d_native(x, &self.q_weight, Some(&self.q_bias))?;
        let k = fused_linear3d_native(x, &self.k_weight, Some(&self.k_bias))?;
        let v = fused_linear3d_native(x, &self.v_weight, Some(&self.v_bias))?;

        // ---- Reshape to [B, H, S, D] ----
        let q = q.reshape(&[b, s, h, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;
        let k = k.reshape(&[b, s, h_kv, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;
        let v = v.reshape(&[b, s, h_kv, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;

        // ---- RoPE (halfsplit) on Q and K only ----
        // rope_halfsplit_bf16 expects cos/sin reshapeable to `[cos_bh, S, D/2]`.
        // We pass `[1, 1, S, half]` → cos_bh = 1, broadcasts across BH.
        let half = d / 2;
        let cos_4d = cos.reshape(&[1, 1, s, half])?;
        let sin_4d = sin.reshape(&[1, 1, s, half])?;
        let q = rope_halfsplit_bf16(&q, &cos_4d, &sin_4d)?
            .reshape(&[b, h, s, d])?;
        let k = rope_halfsplit_bf16(&k, &cos_4d, &sin_4d)?
            .reshape(&[b, h_kv, s, d])?;

        // ---- GQA repeat-KV (forward_with_sinks requires Hq == Hk == Hv) ----
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // ---- Pick the right mask ----
        let mask: Option<&Tensor> = match self.layer_type {
            LayerType::Sliding => sliding_mask,
            LayerType::Full => full_mask,
        };

        // ---- SDPA with attention sinks ----
        // forward_with_sinks output is `[B, H, S, D]` in q.dtype() (BF16).
        let attn = forward_with_sinks(&q, &k, &v, mask, &self.sinks)
            .map_err(|e| Error::InvalidOperation(format!("forward_with_sinks: {e:?}")))?;

        // ---- Re-assemble [B, S, H*D] ----
        let attn = attn.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[b, s, h * d])?;

        // ---- O projection ----
        let _ = hidden; // hidden == h*d; sanity-checked implicitly by o_weight shape
        fused_linear3d_native(&attn, &self.o_weight, Some(&self.o_bias))
    }
}

/// GQA expand: `[B, Hkv, S, D]` → `[B, Hkv*n_rep, S, D]` by repeat.
///
/// Mirrors `repeat_kv` in transformers and qwen3_encoder. Returns a
/// contiguous tensor.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return x.clone_result();
    }
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "repeat_kv: expected 4D [B,Hkv,S,D], got {dims:?}"
        )));
    }
    let (b, h_kv, n, d) = (dims[0], dims[1], dims[2], dims[3]);
    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    // Stack along axis-2 → [B, Hkv, n_rep, S, D], then reshape.
    let stacked = Tensor::stack(&copies, 2)?;
    stacked.reshape(&[b, h_kv * n_rep, n, d])
}

// ---------------------------------------------------------------------------
// GptOssMoE — token-choice top-K MoE FFN
// ---------------------------------------------------------------------------

/// Token-choice MoE block. Mirrors `GptOssExperts` + `GptOssTopKRouter`.
///
/// **Layout choice for the per-row bias**: transformers stores per-expert
/// biases as `[E, 2*inter]` (`gate_up_proj_bias`) and `[E, hidden]`
/// (`down_proj_bias`). After permuting tokens into expert-major order, each
/// permuted row needs its expert's bias added. We materialize an
/// `[T*K, 2*inter]` (resp. `[T*K, hidden]`) per-row bias tensor via
/// `index_select0` on the per-expert bias table indexed by the
/// `expert_id_for_row` host vector built from `plan.offsets`.
///
/// **Activation** (the GPT-OSS twist):
/// ```text
/// gate, up = gate_up[..., ::2], gate_up[..., 1::2]   # interleaved split
/// gate     = gate.clamp(max=limit)                    # one-sided clamp
/// up       = up.clamp(-limit, limit)                  # symmetric clamp
/// glu      = gate * sigmoid(alpha * gate)             # GELU-exact style (alpha=1.702)
/// out      = (up + 1) * glu                           # GPT-OSS gated-output
/// ```
/// Source: `transformers/models/gpt_oss/modeling_gpt_oss.py` lines 117-122 and 131-135.
pub struct GptOssMoE {
    /// Router weight: `[num_local_experts, hidden]` BF16 (PyTorch layout).
    pub router_weight: Tensor,
    /// Router bias: `[num_local_experts]` BF16.
    pub router_bias: Tensor,

    /// Per-expert gate_up bias: `[E, 2 * inter]` BF16.
    pub gate_up_bias: Tensor,
    /// Per-expert down bias: `[E, hidden]` BF16.
    pub down_bias: Tensor,

    /// Expert FFN weight storage. Either resident BF16 (tests, small models)
    /// or packed MXFP4 with on-the-fly dequant (real Lens checkpoint, fits
    /// on 24 GB).
    pub experts: ExpertWeightStorage,

    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,

    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
}

/// How the per-expert gate_up and down matrices are stored.
///
/// - **`Bf16Resident`**: dequantized BF16 tensors live on GPU. Used by tests
///   and small models. Total memory for Lens: 38 GB — won't fit on 24 GB.
/// - **`Mxfp4Packed`**: packed MXFP4 + E8M0 scales live on GPU as raw `u8`
///   buffers (~10 GB total for Lens). Dequantization happens per layer at
///   forward time into a transient BF16 tensor that's freed after the
///   layer's grouped_mm returns.
pub enum ExpertWeightStorage {
    /// Resident BF16 weights. Used by tests and small synthetic-weight setups.
    Bf16Resident {
        /// `[E, hidden, 2 * inter]` BF16.
        gate_up_weight: Tensor,
        /// `[E, inter, hidden]` BF16.
        down_weight: Tensor,
    },
    /// Packed MXFP4 + scales. Dequantizes to BF16 at forward time.
    Mxfp4Packed {
        /// Raw gate_up MXFP4 blocks: shape `[E, 2*inter, hidden/32, 16]` u8.
        gate_up_blocks: Mxfp4Pair,
        /// Raw down MXFP4 blocks: shape `[E, hidden, inter/32, 16]` u8.
        down_blocks: Mxfp4Pair,
    },
}

/// One MXFP4 weight tensor: packed-4-bit blocks + E8M0 per-block scales.
///
/// Stored as `CudaSlice<u8>` (no `Tensor` wrapper) because flame-core's
/// `TensorStorage` doesn't have a U8 variant. The `dequant_mxfp4_to_bf16`
/// kernel takes raw `CudaSlice<u8>` ptrs anyway.
pub struct Mxfp4Pair {
    /// Block data: `rows_total * 16` bytes.
    pub blocks: cudarc::driver::CudaSlice<u8>,
    /// E8M0 scales: `rows_total` bytes.
    pub scales: cudarc::driver::CudaSlice<u8>,
    /// Logical (post-dequant, post-transpose) shape. For gate_up:
    /// `[E, hidden, 2*inter]`. For down: `[E, inter, hidden]`. This is the
    /// shape that goes into `grouped_mm_bf16`.
    pub dequant_shape: Vec<usize>,
    /// Pre-transpose shape (matches transformers' storage layout):
    ///   gate_up_blocks: `[E, 2*inter, hidden]`
    ///   down_blocks:    `[E, hidden, inter]`
    /// Needed because `dequant_mxfp4_to_bf16` writes in this layout, and we
    /// `permute(&[0, 2, 1])` after to get `dequant_shape`.
    pub pre_transpose_shape: Vec<usize>,
}

impl GptOssMoE {
    /// Run one MoE FFN block.
    ///
    /// Input: `[B, S, hidden]` BF16. Output: `[B, S, hidden]` BF16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xd = x.shape().dims().to_vec();
        if xd.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "GptOssMoE.forward: x must be [B, S, hidden], got {xd:?}"
            )));
        }
        let (b, s, hidden) = (xd[0], xd[1], xd[2]);
        if hidden != self.hidden_size {
            return Err(Error::InvalidInput(format!(
                "GptOssMoE.forward: x hidden={hidden} != config.hidden_size={}",
                self.hidden_size
            )));
        }
        let t = b * s;
        let inter = self.intermediate_size;
        let two_inter = 2 * inter;
        let e = self.num_experts;
        let k = self.top_k;

        // ---- 1. Flatten to (T, hidden) ----
        let x2 = x.reshape(&[t, hidden])?;

        // ---- 2. Router: linear(hidden -> E) with bias → (T, E) BF16 ----
        // Use the same fused_linear3d_native path via a [1, T, hidden] reshape.
        let x3 = x2.reshape(&[1, t, hidden])?;
        let router_logits =
            fused_linear3d_native(&x3, &self.router_weight, Some(&self.router_bias))?;
        let router_logits = router_logits.reshape(&[t, e])?;

        // ---- 3. Top-K routing (GPT-OSS: TopKSoftmax) ----
        let plan = token_choice_route(&router_logits, k, ScoreMode::TopKSoftmax)?;

        // ---- 4. Permute tokens to expert-major (T*K, hidden) ----
        let permuted_x = permute_tokens_for_token_choice(&x2, &plan)?;
        let t_picks = t * k;
        debug_assert_eq!(permuted_x.shape().dims(), &[t_picks, hidden]);

        // Compute t_max = max(offsets[e+1] - offsets[e]) for grouped_mm sizing.
        let t_max = plan
            .offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        if t_max == 0 {
            // No tokens routed anywhere — pathological case; just return zeros
            // so the residual add is a no-op. (Shouldn't happen with k >= 1.)
            return Tensor::zeros_dtype(Shape::from_dims(&[b, s, hidden]), DType::BF16, x.device().clone());
        }

        // Expert-major offsets for grouped_mm_bf16 (drops the leading 0).
        let gmm_offsets = plan.offsets_for_grouped_mm();
        debug_assert_eq!(gmm_offsets.len(), e);

        // Build per-row expert-id vector → I32 device tensor for index_select0.
        let expert_id_for_row = build_expert_id_for_row(&plan.offsets, t_picks);
        let expert_id_tensor = i32_device_tensor(&expert_id_for_row, x.device())?;

        // ---- 5. Grouped GEMM: gate_up projection → (T*K, 2*inter) ----
        // grouped_mm_bf16 expects w: (E, K=hidden, N=2*inter). With
        // `Bf16Resident` we have it pre-materialized; with `Mxfp4Packed` we
        // dequantize-and-transpose just before the GEMM. The transient BF16
        // tensor lives until the end of this layer's MoE call.
        let gate_up_w_bf16 = materialize_bf16_weight(
            &self.experts,
            ExpertWeightSlot::GateUp,
            x.device(),
        )?;
        let gate_up = grouped_mm_bf16(&permuted_x, &gate_up_w_bf16, &gmm_offsets, t_max)?;
        debug_assert_eq!(gate_up.shape().dims(), &[t_picks, two_inter]);
        // Free the dequantized weight before allocating the next one (peak
        // memory savings: ~530 MiB for the down dequant later).
        drop(gate_up_w_bf16);

        // ---- 6. Add per-expert gate_up bias (broadcast via index_select) ----
        // bias[e] is `[2*inter]`; gather into `[T*K, 2*inter]` and add.
        // `gate_up_bias` is already `[E, 2*inter]`, so `index_select0` is
        // direct.
        let bias_per_row = self.gate_up_bias.index_select0(&expert_id_tensor)?;
        let gate_up = gate_up.add(&bias_per_row)?;

        // ---- 7. Interleaved split: gate = [..., ::2], up = [..., 1::2] ----
        // Reshape `[T*K, 2*inter]` → `[T*K, inter, 2]` so the innermost dim
        // holds each (g_i, u_i) pair. Then narrow + contiguous.
        let gu_pairs = gate_up.reshape(&[t_picks, inter, 2])?;
        let gate = gu_pairs.narrow(2, 0, 1)?.contiguous()?.reshape(&[t_picks, inter])?;
        let up = gu_pairs.narrow(2, 1, 1)?.contiguous()?.reshape(&[t_picks, inter])?;

        // ---- 8. Clamp + activation ----
        // gate.clamp(min=None, max=limit) — one-sided. We use the symmetric
        // clamp with a large lower bound; the BF16 representable range easily
        // exceeds anything that could survive a clamp at +limit.
        let gate = clamp_bf16(&gate, f32::NEG_INFINITY, self.swiglu_limit)?;
        let up = clamp_bf16(&up, -self.swiglu_limit, self.swiglu_limit)?;

        // glu = gate * sigmoid(alpha * gate)
        let sig_in = gate.mul_scalar(self.swiglu_alpha)?;
        let sig = sig_in.sigmoid()?;
        let glu = gate.mul(&sig)?;
        // out = (up + 1) * glu
        let up_plus = up.add_scalar(1.0)?;
        let activated = up_plus.mul(&glu)?;
        debug_assert_eq!(activated.shape().dims(), &[t_picks, inter]);

        // ---- 9. Grouped GEMM: down projection → (T*K, hidden) ----
        let down_w_bf16 = materialize_bf16_weight(
            &self.experts,
            ExpertWeightSlot::Down,
            x.device(),
        )?;
        let down_out = grouped_mm_bf16(&activated, &down_w_bf16, &gmm_offsets, t_max)?;
        debug_assert_eq!(down_out.shape().dims(), &[t_picks, hidden]);
        drop(down_w_bf16);

        // ---- 10. Add per-expert down bias ----
        let down_bias_per_row = self.down_bias.index_select0(&expert_id_tensor)?;
        let down_out = down_out.add(&down_bias_per_row)?;

        // ---- 11. Weighted scatter-add unpermute → (T, hidden) F32 → BF16 ----
        let gating = Tensor::from_vec_dtype(
            plan.expert_weights_flat.clone(),
            Shape::from_dims(&[t_picks]),
            x.device().clone(),
            DType::F32,
        )?;
        let mut accum = Tensor::zeros_dtype(
            Shape::from_dims(&[t, hidden]),
            DType::F32,
            x.device().clone(),
        )?;
        fused_gated_scatter_add_bf16(&down_out, &gating, &plan.permuted_token_indices, &mut accum)?;
        let out = accum.to_dtype(DType::BF16)?;
        out.reshape(&[b, s, hidden])
    }
}

/// Identifies which expert weight to materialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertWeightSlot {
    GateUp,
    Down,
}

/// Materialize a BF16 expert weight ready for `grouped_mm_bf16`.
///
/// - For `Bf16Resident`, returns a clone of the stored tensor (no copy of
///   the storage — `Tensor::clone` is a refcount bump).
/// - For `Mxfp4Packed`, runs `dequant_mxfp4_to_bf16` against the slot's
///   `(blocks, scales)` pair, then `permute(&[0, 2, 1])` + `.contiguous()`
///   to match the `(E, K, N)` layout expected by `grouped_mm_bf16`. Allocates
///   a fresh BF16 tensor each call; the caller is expected to drop it after
///   the GEMM so peak memory stays bounded.
fn materialize_bf16_weight(
    experts: &ExpertWeightStorage,
    slot: ExpertWeightSlot,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    match (experts, slot) {
        (ExpertWeightStorage::Bf16Resident { gate_up_weight, .. }, ExpertWeightSlot::GateUp) => {
            gate_up_weight.clone_result()
        }
        (ExpertWeightStorage::Bf16Resident { down_weight, .. }, ExpertWeightSlot::Down) => {
            down_weight.clone_result()
        }
        (ExpertWeightStorage::Mxfp4Packed { gate_up_blocks, .. }, ExpertWeightSlot::GateUp) => {
            dequant_mxfp4_pair(gate_up_blocks, device)
        }
        (ExpertWeightStorage::Mxfp4Packed { down_blocks, .. }, ExpertWeightSlot::Down) => {
            dequant_mxfp4_pair(down_blocks, device)
        }
    }
}

/// Dequant an `Mxfp4Pair` to BF16 and transpose to the `(E, K, N)` GEMM layout.
fn dequant_mxfp4_pair(pair: &Mxfp4Pair, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let dequantized = dequant_mxfp4_to_bf16(
        &pair.blocks,
        &pair.scales,
        Shape::from_dims(&pair.pre_transpose_shape),
        device,
    )?;
    // [E, R, K] → [E, K, R]
    dequantized.permute(&[0, 2, 1])?.contiguous()
}

/// Build the per-row expert-id vector. For row `r` in expert-major order,
/// `result[r] = e` where `offsets[e] <= r < offsets[e+1]`.
fn build_expert_id_for_row(offsets: &[i32], t_picks: usize) -> Vec<i32> {
    let mut out = vec![0i32; t_picks];
    for (e, win) in offsets.windows(2).enumerate() {
        let start = win[0] as usize;
        let end = win[1] as usize;
        for v in &mut out[start..end] {
            *v = e as i32;
        }
    }
    out
}

/// Upload a host `Vec<i32>` as a device I32 tensor of shape `[len]`.
///
/// Mirrors the F32-roundtrip-then-cast pattern used in
/// `flame_core::ops::token_choice_routing` (since `from_vec` is F32-only).
/// Indices in `0..num_experts` are well below `2^24`, so the round-trip is exact.
fn i32_device_tensor(host: &[i32], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let len = host.len();
    let as_f32: Vec<f32> = host.iter().map(|&v| v as f32).collect();
    let t = Tensor::from_vec(as_f32, Shape::from_dims(&[len]), device.clone())?;
    t.to_dtype(DType::I32)
}

// ---------------------------------------------------------------------------
// GptOssLayer
// ---------------------------------------------------------------------------

/// One full GPT-OSS decoder layer.
///
/// Pre-norm architecture (mirrors `GptOssDecoderLayer.forward`):
/// ```text
/// r = x + attn(rms_norm(x, pre_norm))
/// r = r + mlp(rms_norm(r, post_norm))
/// ```
pub struct GptOssLayer {
    /// `[hidden]` BF16 — RMSNorm scale for the attention input.
    pub pre_norm: Tensor,
    /// `[hidden]` BF16 — RMSNorm scale for the MLP input.
    pub post_norm: Tensor,
    pub attn: GptOssAttention,
    pub mlp: GptOssMoE,
    pub rms_norm_eps: f32,
}

impl GptOssLayer {
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        full_mask: Option<&Tensor>,
        sliding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // ---- Pre-norm + self-attention + residual ----
        let normed = apply_rms_norm(x, &self.pre_norm, self.rms_norm_eps)?;
        let attn_out = self.attn.forward(&normed, cos, sin, full_mask, sliding_mask)?;
        let h = x.add(&attn_out)?;

        // ---- Post-attention norm + MLP + residual ----
        let normed2 = apply_rms_norm(&h, &self.post_norm, self.rms_norm_eps)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        h.add(&mlp_out)
    }
}

/// Apply RMSNorm with a per-channel weight along the last dim.
///
/// `x`: `[B, S, hidden]` BF16. `weight`: `[hidden]` BF16. Returns `[B, S, hidden]` BF16.
///
/// `cuda_ops_bf16::rms_norm_bf16` operates on `[..., hidden]` (any leading
/// dims); we flatten to 2D for the call and reshape back to be defensive.
fn apply_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let last = *dims.last().ok_or_else(|| {
        Error::InvalidInput("apply_rms_norm: tensor has no dims".into())
    })?;
    let lead: usize = dims[..dims.len() - 1].iter().product();
    let x2 = x.reshape(&[lead, last])?;
    let y2 = rms_norm_bf16(&x2, Some(weight), eps)?;
    y2.reshape(&dims)
}

// ---------------------------------------------------------------------------
// GptOssEncoder — 24-layer stack + token embedding + weight loader
// ---------------------------------------------------------------------------

/// Full GPT-OSS encoder for Lens.
///
/// Owns the token embedding table, a 24-layer stack of `GptOssLayer`, and a
/// shared `GptOssRotaryEmbedding`. The encoder operates in "early-exit" mode:
/// the forward only processes layers `0..=max(selected_layer_index)` and
/// captures the post-residual hidden state at each `selected_layer_index`
/// for cross-attention into the Lens DiT. Layers above the max selected
/// index are never constructed (saves memory + load time).
///
/// **Memory strategy for D5b (Option γ — MXFP4 resident on GPU):**
///
/// 24 × 32 × 2 expert weights, each ~50 MiB BF16 (gate_up) or ~16 MiB BF16
/// (down), would total ~38 GiB if all dequantized at load time. Won't fit on
/// 24 GiB. Instead, we store the **packed MXFP4** weights on GPU (~10 GiB
/// total: blocks are 1/4 the size of BF16, scales are 1/32) and dequantize
/// per-layer at forward time into a reusable BF16 scratch tensor. This
/// trades wall-time (~1536 dequant launches per forward) for memory.
///
/// **D6 will replace this with a BlockOffloader-backed streaming variant**
/// — packed MXFP4 in pinned host RAM, async H2D + dequant overlapped with
/// the previous layer's compute. The MoE forward interface stays the same.
pub struct GptOssEncoder {
    pub config: GptOssConfig,
    /// Token embedding table: `[vocab_size, hidden]` BF16.
    pub embed_tokens: Tensor,
    /// Layer stack, length `= max(selected_layer_index) + 1`.
    pub layers: Vec<GptOssLayer>,
    /// Rotary-embedding cache (built lazily by `freqs_for`).
    pub rope: GptOssRotaryEmbedding,
    /// 0-indexed layers to capture, sorted ascending. Default `[5, 11, 17, 23]`.
    pub selected_layers: Vec<usize>,
    /// CUDA device handle.
    pub device: Arc<CudaDevice>,
}

impl GptOssEncoder {
    /// Construct an empty encoder (zeroed weights) sized for the config.
    ///
    /// Only allocates the layers up to `max(selected_layer_index)` inclusive
    /// since later layers' outputs are never consumed by Lens.
    ///
    /// All tensors are device-resident, BF16, but value-undefined (zero) — call
    /// `load_from_directory` to populate from a real checkpoint.
    pub fn new(config: GptOssConfig, device: &Arc<CudaDevice>) -> Result<Self> {
        let selected_layers = {
            let mut v = config.selected_layer_index.clone();
            v.sort_unstable();
            v.dedup();
            v
        };
        let max_layer = *selected_layers.iter().max().ok_or_else(|| {
            Error::InvalidInput(
                "GptOssEncoder::new: selected_layer_index must be non-empty".into(),
            )
        })?;
        if max_layer >= config.num_hidden_layers {
            return Err(Error::InvalidInput(format!(
                "GptOssEncoder::new: max selected layer {max_layer} >= num_hidden_layers {}",
                config.num_hidden_layers
            )));
        }

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let h_q = config.num_attention_heads * config.head_dim;
        let h_kv = config.num_key_value_heads * config.head_dim;
        let e = config.num_local_experts;

        // Embedding table.
        let embed_tokens = Tensor::zeros_dtype(
            Shape::from_dims(&[config.vocab_size, hidden]),
            DType::BF16,
            device.clone(),
        )?;

        // Per-layer empty tensors.
        let mut layers: Vec<GptOssLayer> = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
            let layer_type = config.layer_types.get(i).copied().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "GptOssEncoder::new: layer_types missing entry for layer {i}"
                ))
            })?;

            let attn = GptOssAttention {
                q_weight: Tensor::zeros_dtype(
                    Shape::from_dims(&[h_q, hidden]),
                    DType::BF16,
                    device.clone(),
                )?,
                q_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[h_q]),
                    DType::BF16,
                    device.clone(),
                )?,
                k_weight: Tensor::zeros_dtype(
                    Shape::from_dims(&[h_kv, hidden]),
                    DType::BF16,
                    device.clone(),
                )?,
                k_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[h_kv]),
                    DType::BF16,
                    device.clone(),
                )?,
                v_weight: Tensor::zeros_dtype(
                    Shape::from_dims(&[h_kv, hidden]),
                    DType::BF16,
                    device.clone(),
                )?,
                v_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[h_kv]),
                    DType::BF16,
                    device.clone(),
                )?,
                o_weight: Tensor::zeros_dtype(
                    Shape::from_dims(&[hidden, h_q]),
                    DType::BF16,
                    device.clone(),
                )?,
                o_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[hidden]),
                    DType::BF16,
                    device.clone(),
                )?,
                sinks: Tensor::zeros_dtype(
                    Shape::from_dims(&[config.num_attention_heads]),
                    DType::BF16,
                    device.clone(),
                )?,
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                layer_type,
                scaling: (config.head_dim as f32).powf(-0.5),
            };

            // Note: do NOT pre-allocate the full `[E, hidden, 2*inter]` BF16
            // expert weights — that would burn ~38 GB on Lens before any
            // weights are loaded. We use 1-element BF16 placeholders that
            // `load_from_directory` swaps out for `Mxfp4Packed`. Tests that
            // construct an encoder without calling `load_from_directory`
            // must replace these placeholders explicitly (see
            // `encoder_forward_smoke_shape`).
            let placeholder_gate_up = Tensor::zeros_dtype(
                Shape::from_dims(&[1, 1, 1]),
                DType::BF16,
                device.clone(),
            )?;
            let placeholder_down = Tensor::zeros_dtype(
                Shape::from_dims(&[1, 1, 1]),
                DType::BF16,
                device.clone(),
            )?;
            let mlp = GptOssMoE {
                router_weight: Tensor::zeros_dtype(
                    Shape::from_dims(&[e, hidden]),
                    DType::BF16,
                    device.clone(),
                )?,
                router_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[e]),
                    DType::BF16,
                    device.clone(),
                )?,
                gate_up_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[e, 2 * inter]),
                    DType::BF16,
                    device.clone(),
                )?,
                down_bias: Tensor::zeros_dtype(
                    Shape::from_dims(&[e, hidden]),
                    DType::BF16,
                    device.clone(),
                )?,
                experts: ExpertWeightStorage::Bf16Resident {
                    gate_up_weight: placeholder_gate_up,
                    down_weight: placeholder_down,
                },
                num_experts: e,
                top_k: config.experts_per_token,
                hidden_size: hidden,
                intermediate_size: inter,
                swiglu_limit: config.swiglu_limit,
                swiglu_alpha: config.swiglu_alpha,
            };

            let pre_norm = Tensor::zeros_dtype(
                Shape::from_dims(&[hidden]),
                DType::BF16,
                device.clone(),
            )?;
            let post_norm = Tensor::zeros_dtype(
                Shape::from_dims(&[hidden]),
                DType::BF16,
                device.clone(),
            )?;

            layers.push(GptOssLayer {
                pre_norm,
                post_norm,
                attn,
                mlp,
                rms_norm_eps: config.rms_norm_eps,
            });
        }

        let rope = GptOssRotaryEmbedding::new(&config)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            rope,
            selected_layers,
            device: device.clone(),
        })
    }

    /// Load weights from the `microsoft/Lens/text_encoder/` directory.
    ///
    /// The directory must contain `model-0000{1,2,3}-of-00003.safetensors`.
    ///
    /// **Streaming strategy:** For each shard we open the mmap once, walk its
    /// tensor index, and route each tensor to the right destination field.
    /// MXFP4 expert weights are uploaded as raw `u8` to GPU, dequantized to
    /// BF16 via `dequant_mxfp4_to_bf16`, then transposed to match
    /// `grouped_mm_bf16`'s expected `(E, K, N)` layout. The intermediate
    /// raw-byte upload is freed when the loop iteration ends.
    ///
    /// **Skipped tensors** (with reason logged):
    /// - `lm_head.weight` — Lens early-exits at layer 23, never used
    /// - `model.norm.weight` — same reason (final norm is post-layer 23)
    /// - `model.layers.{i}.*` for `i > max(selected_layers)` — never reached
    pub fn load_from_directory(
        &mut self,
        dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<LoaderStats> {
        let shard_paths: Vec<std::path::PathBuf> = (1..=3)
            .map(|i| dir.join(format!("model-0000{i}-of-00003.safetensors")))
            .collect();
        for p in &shard_paths {
            if !p.exists() {
                return Err(Error::InvalidInput(format!(
                    "GptOssEncoder::load_from_directory: missing shard {}",
                    p.display()
                )));
            }
        }

        let max_layer = *self.selected_layers.iter().max().unwrap();
        let mut stats = LoaderStats::default();

        for path in &shard_paths {
            let mmap = eri_safetensors::MmapFile::open_path(path).map_err(|e| {
                Error::Io(format!(
                    "eri-safetensors open '{}': {e}",
                    path.display()
                ))
            })?;
            // Snapshot the keys+dtypes so we can mutate `self` without borrowing the mmap.
            let entries: Vec<(String, String, Vec<usize>)> = mmap
                .tensors
                .iter()
                .map(|(k, v)| (k.clone(), v.dtype.clone(), v.shape.clone()))
                .collect();
            for (key, dtype, shape) in entries {
                let routed = self.route_and_load_tensor(&mmap, &key, &dtype, &shape, max_layer, device)?;
                match routed {
                    LoadOutcome::Consumed => stats.consumed += 1,
                    LoadOutcome::Skipped(reason) => {
                        stats.skipped += 1;
                        if stats.skipped <= 8 {
                            log::debug!("[GptOssEncoder] skip '{key}': {reason}");
                        }
                    }
                    LoadOutcome::Mxfp4Dequant => {
                        stats.mxfp4_dequants += 1;
                        stats.consumed += 1;
                    }
                }
            }
        }

        // Per-layer post-load sanity: ensure pre_norm got loaded (would be all-zero otherwise).
        for (i, layer) in self.layers.iter().enumerate() {
            if !tensor_has_any_finite(&layer.pre_norm)? {
                return Err(Error::InvalidInput(format!(
                    "GptOssEncoder: layer {i}.pre_norm is all-zero after load — checkpoint missing 'model.layers.{i}.input_layernorm.weight'?"
                )));
            }
        }

        log::info!(
            "[GptOssEncoder] loaded: {} tensors consumed ({} MXFP4 dequants), {} skipped, {} layers active",
            stats.consumed,
            stats.mxfp4_dequants,
            stats.skipped,
            self.layers.len()
        );
        Ok(stats)
    }

    /// Encode a batch of token IDs into hidden states at each selected layer.
    ///
    /// Inputs:
    /// - `input_ids`: `[B, S]` I32 (token IDs in `[0, vocab_size)`)
    /// - `attention_mask`: `[B, S]` I32 (1 = valid, 0 = padded). Currently
    ///   only used for shape validation — padding handling is done at the
    ///   caller's mask boundary; sequences with right-padding produce correct
    ///   captured hidden states for the non-padded prefix.
    ///
    /// Returns a `Vec<Tensor>` of length `selected_layers.len()`, each
    /// `[B, S, hidden]` BF16. The order matches `self.selected_layers`.
    pub fn encode(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let id_dims = input_ids.shape().dims().to_vec();
        if id_dims.len() != 2 {
            return Err(Error::InvalidInput(format!(
                "GptOssEncoder::encode: input_ids must be [B, S], got {id_dims:?}"
            )));
        }
        let (b, s) = (id_dims[0], id_dims[1]);
        let mask_dims = attention_mask.shape().dims().to_vec();
        if mask_dims != vec![b, s] {
            return Err(Error::InvalidInput(format!(
                "GptOssEncoder::encode: attention_mask shape {mask_dims:?} != input_ids shape {id_dims:?}"
            )));
        }
        if input_ids.dtype() != DType::I32 {
            return Err(Error::InvalidInput(format!(
                "GptOssEncoder::encode: input_ids must be I32, got {:?}",
                input_ids.dtype()
            )));
        }

        // 1. Token embedding: flatten to [B*S], gather rows, reshape back.
        let flat_ids = input_ids.reshape(&[b * s])?;
        let gathered = self.embed_tokens.index_select0(&flat_ids)?;
        let mut x = gathered.reshape(&[b, s, self.config.hidden_size])?;

        // 2. RoPE tables (host cache reused across calls).
        let (cos, sin) = self.rope.freqs_for(s, &self.device)?;

        // 3. Build masks once (full + sliding).
        let kit = GptOssMaskKit::build(s, self.config.sliding_window, &self.device)?;

        // 4. Layer loop with capture.
        let max_idx = *self.selected_layers.iter().max().unwrap();
        let mut captured: Vec<Option<Tensor>> = vec![None; self.selected_layers.len()];

        for i in 0..=max_idx {
            let layer = &self.layers[i];
            x = layer.forward(
                &x,
                &cos,
                &sin,
                Some(&kit.full_mask),
                Some(&kit.sliding_mask),
            )?;
            if let Some(pos) = self.selected_layers.iter().position(|&idx| idx == i) {
                captured[pos] = Some(x.clone_result()?);
            }
        }

        captured
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "GptOssEncoder::encode: capture slot {i} (layer {}) was not populated",
                        self.selected_layers[i]
                    ))
                })
            })
            .collect()
    }

    /// Route a single tensor from the safetensors mmap to its destination.
    fn route_and_load_tensor(
        &mut self,
        mmap: &eri_safetensors::MmapFile,
        key: &str,
        dtype: &str,
        shape: &[usize],
        max_layer: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<LoadOutcome> {
        // 1. Cheap structural skips.
        if key == "lm_head.weight" {
            return Ok(LoadOutcome::Skipped("Lens early-exits — lm_head unused"));
        }
        if key == "model.norm.weight" {
            return Ok(LoadOutcome::Skipped(
                "Lens uses pre-final hidden states — final norm unused",
            ));
        }
        if key == "model.embed_tokens.weight" {
            self.embed_tokens =
                load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
            return Ok(LoadOutcome::Consumed);
        }

        // 2. Layer-scoped routing.
        let layer_idx = match parse_layer_idx(key) {
            Some(i) => i,
            None => {
                return Ok(LoadOutcome::Skipped("not a layer-scoped tensor"));
            }
        };
        if layer_idx > max_layer {
            return Ok(LoadOutcome::Skipped("layer past early-exit threshold"));
        }
        let suffix = &key[format!("model.layers.{layer_idx}.").len()..];

        // Slice out the destination layer.
        let layer = &mut self.layers[layer_idx];

        match suffix {
            "input_layernorm.weight" => {
                layer.pre_norm = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "post_attention_layernorm.weight" => {
                layer.post_norm = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.q_proj.weight" => {
                layer.attn.q_weight = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.q_proj.bias" => {
                layer.attn.q_bias = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.k_proj.weight" => {
                layer.attn.k_weight = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.k_proj.bias" => {
                layer.attn.k_bias = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.v_proj.weight" => {
                layer.attn.v_weight = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.v_proj.bias" => {
                layer.attn.v_bias = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.o_proj.weight" => {
                layer.attn.o_weight = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.o_proj.bias" => {
                layer.attn.o_bias = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "self_attn.sinks" => {
                layer.attn.sinks = load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "mlp.router.weight" => {
                layer.mlp.router_weight =
                    load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "mlp.router.bias" => {
                layer.mlp.router_bias =
                    load_bf16_tensor_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "mlp.experts.gate_up_proj_bias" => {
                // Stored as F32 in the checkpoint; cast to BF16 to match GptOssMoE field.
                layer.mlp.gate_up_bias =
                    load_f32_then_bf16_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "mlp.experts.down_proj_bias" => {
                layer.mlp.down_bias =
                    load_f32_then_bf16_from_mmap(mmap, key, dtype, shape, device)?;
                Ok(LoadOutcome::Consumed)
            }
            "mlp.experts.gate_up_proj_blocks" => {
                let scales_key =
                    format!("model.layers.{layer_idx}.mlp.experts.gate_up_proj_scales");
                let scales_info = mmap.tensors.get(&scales_key).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "GptOssEncoder: gate_up_proj_blocks at layer {layer_idx} but scales companion '{scales_key}' missing from same shard"
                    ))
                })?;
                let pair = upload_mxfp4_pair(
                    mmap,
                    key,
                    &scales_key,
                    shape,
                    &scales_info.shape,
                    device,
                    Mxfp4Slot::GateUp,
                )?;
                set_expert_slot(&mut layer.mlp.experts, Mxfp4Slot::GateUp, pair, device)?;
                Ok(LoadOutcome::Mxfp4Dequant)
            }
            "mlp.experts.gate_up_proj_scales" => {
                Ok(LoadOutcome::Skipped("paired-companion to _blocks"))
            }
            "mlp.experts.down_proj_blocks" => {
                let scales_key =
                    format!("model.layers.{layer_idx}.mlp.experts.down_proj_scales");
                let scales_info = mmap.tensors.get(&scales_key).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "GptOssEncoder: down_proj_blocks at layer {layer_idx} but scales companion '{scales_key}' missing from same shard"
                    ))
                })?;
                let pair = upload_mxfp4_pair(
                    mmap,
                    key,
                    &scales_key,
                    shape,
                    &scales_info.shape,
                    device,
                    Mxfp4Slot::Down,
                )?;
                set_expert_slot(&mut layer.mlp.experts, Mxfp4Slot::Down, pair, device)?;
                Ok(LoadOutcome::Mxfp4Dequant)
            }
            "mlp.experts.down_proj_scales" => {
                Ok(LoadOutcome::Skipped("paired-companion to _blocks"))
            }
            other => Ok(LoadOutcome::Skipped(match other {
                _ => "unrecognized layer-scoped tensor",
            })),
        }
    }
}

/// Statistics from `GptOssEncoder::load_from_directory`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoaderStats {
    pub consumed: usize,
    pub skipped: usize,
    pub mxfp4_dequants: usize,
}

enum LoadOutcome {
    Consumed,
    Skipped(&'static str),
    Mxfp4Dequant,
}

/// Parse a layer index out of `"model.layers.{i}.*"`.
fn parse_layer_idx(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse().ok()
}

/// Load a BF16 tensor from an mmap region into device memory.
///
/// Mirrors the `decode_tensors_from_mmap` BF16 path in
/// `flame_core::serialization`. Only BF16 is supported here — other dtypes
/// are rejected (the encoder loader is type-strict by design).
fn load_bf16_tensor_from_mmap(
    mmap: &eri_safetensors::MmapFile,
    key: &str,
    dtype: &str,
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    if dtype != "BF16" {
        return Err(Error::InvalidInput(format!(
            "GptOssEncoder: expected BF16 for '{key}', got {dtype}"
        )));
    }
    let bytes = mmap.tensor_bytes(key).ok_or_else(|| {
        Error::InvalidInput(format!("GptOssEncoder: mmap missing '{key}'"))
    })?;
    let num_elems = bytes.len() / 2;
    let expected: usize = shape.iter().product();
    if num_elems != expected {
        return Err(Error::InvalidInput(format!(
            "GptOssEncoder: '{key}' byte len {}/2={} != shape elems {expected}",
            bytes.len(),
            num_elems
        )));
    }
    let mut bf16_u16 = vec![0u16; num_elems];
    for (value, chunk) in bf16_u16.iter_mut().zip(bytes.chunks_exact(2)) {
        *value = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
    let mut tensor = Tensor::zeros_dtype(
        Shape::from_dims(shape),
        DType::BF16,
        device.clone(),
    )?;
    tensor.copy_from_bf16_slice(&bf16_u16)?;
    Ok(tensor)
}

/// Load an F32 tensor from mmap and cast to BF16. Used for MoE biases which
/// are stored as F32 in the checkpoint but match BF16 fields in our struct.
fn load_f32_then_bf16_from_mmap(
    mmap: &eri_safetensors::MmapFile,
    key: &str,
    dtype: &str,
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    if dtype != "F32" {
        return Err(Error::InvalidInput(format!(
            "GptOssEncoder: expected F32 for '{key}', got {dtype}"
        )));
    }
    let bytes = mmap.tensor_bytes(key).ok_or_else(|| {
        Error::InvalidInput(format!("GptOssEncoder: mmap missing '{key}'"))
    })?;
    let num_elems = bytes.len() / 4;
    let expected: usize = shape.iter().product();
    if num_elems != expected {
        return Err(Error::InvalidInput(format!(
            "GptOssEncoder: '{key}' byte len {}/4={} != shape elems {expected}",
            bytes.len(),
            num_elems
        )));
    }
    let mut data = vec![0f32; num_elems];
    for (v, chunk) in data.iter_mut().zip(bytes.chunks_exact(4)) {
        *v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let f32_tensor =
        Tensor::from_vec(data, Shape::from_dims(shape), device.clone())?;
    f32_tensor.to_dtype(DType::BF16)
}

/// Dequantize an MXFP4 pair (`_blocks`, `_scales`) and transpose the inner
/// pair of expert-weight axes to match `grouped_mm_bf16`'s expected layout.
///
/// Reference: `transformers/integrations/mxfp4.py::convert_moe_packed_tensors`
/// returns `out.transpose(1, 2).contiguous()`. The Rust forward mirrors that
/// transpose lazily (per-layer, at forward time) since storing all 24×2
/// dequantized BF16 matrices would burn 38 GB. Here at load time we upload
/// the packed bytes to GPU and remember the pre-/post-transpose shapes.
///
///   gate_up_blocks: `[E, 2*inter, hidden/32, 16]` u8
///     → dequant `[E, 2*inter, hidden]` BF16 → transpose → `[E, hidden, 2*inter]`
///   down_blocks:    `[E, hidden, inter/32, 16]`  u8
///     → dequant `[E, hidden, inter]`     BF16 → transpose → `[E, inter, hidden]`
fn upload_mxfp4_pair(
    mmap: &eri_safetensors::MmapFile,
    blocks_key: &str,
    scales_key: &str,
    blocks_shape: &[usize], // [E, R, G, 16]
    scales_shape: &[usize], // [E, R, G]
    device: &Arc<CudaDevice>,
    _slot: Mxfp4Slot,
) -> Result<Mxfp4Pair> {
    if blocks_shape.len() != 4 || blocks_shape[3] != 16 {
        return Err(Error::InvalidInput(format!(
            "MXFP4 blocks expected [E, R, G, 16], got {blocks_shape:?}"
        )));
    }
    if scales_shape.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "MXFP4 scales expected [E, R, G], got {scales_shape:?}"
        )));
    }
    if blocks_shape[..3] != *scales_shape {
        return Err(Error::InvalidInput(format!(
            "MXFP4 blocks {:?} prefix != scales {:?}",
            blocks_shape, scales_shape
        )));
    }
    let e = blocks_shape[0];
    let r = blocks_shape[1];
    let g = blocks_shape[2];
    let pre_transpose = vec![e, r, g * 32];
    let dequant_shape = vec![e, g * 32, r];

    let blocks_bytes = mmap.tensor_bytes(blocks_key).ok_or_else(|| {
        Error::InvalidInput(format!("MXFP4: mmap missing blocks '{blocks_key}'"))
    })?;
    let scales_bytes = mmap.tensor_bytes(scales_key).ok_or_else(|| {
        Error::InvalidInput(format!("MXFP4: mmap missing scales '{scales_key}'"))
    })?;
    let expected_blocks_bytes: usize = blocks_shape.iter().product();
    let expected_scales_bytes: usize = scales_shape.iter().product();
    if blocks_bytes.len() != expected_blocks_bytes {
        return Err(Error::InvalidInput(format!(
            "MXFP4 blocks '{blocks_key}': bytes={} expected={}",
            blocks_bytes.len(),
            expected_blocks_bytes
        )));
    }
    if scales_bytes.len() != expected_scales_bytes {
        return Err(Error::InvalidInput(format!(
            "MXFP4 scales '{scales_key}': bytes={} expected={}",
            scales_bytes.len(),
            expected_scales_bytes
        )));
    }

    // Upload raw bytes to device. These stay resident for the encoder's lifetime
    // (~10 GB total across all layers).
    let blocks_dev: cudarc::driver::CudaSlice<u8> = device
        .htod_copy(blocks_bytes.to_vec())
        .map_err(|e| Error::CudaDriver(format!("MXFP4 blocks htod: {e:?}")))?;
    let scales_dev: cudarc::driver::CudaSlice<u8> = device
        .htod_copy(scales_bytes.to_vec())
        .map_err(|e| Error::CudaDriver(format!("MXFP4 scales htod: {e:?}")))?;

    Ok(Mxfp4Pair {
        blocks: blocks_dev,
        scales: scales_dev,
        dequant_shape,
        pre_transpose_shape: pre_transpose,
    })
}

/// Which MoE expert slot an Mxfp4Pair belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mxfp4Slot {
    GateUp,
    Down,
}

/// Install a freshly-loaded `Mxfp4Pair` into the `experts` storage. Transitions
/// `Bf16Resident` → `Mxfp4Packed` on first MXFP4 arrival; subsequent calls
/// just overwrite the corresponding slot.
///
/// **Invariant**: the loader is expected to set BOTH `gate_up` and `down` for a
/// given layer; partial state (one slot still placeholder) would crash the
/// forward when it tries to dequantize the empty slot. The loader walks both
/// shards before forward, and Lens stores both in the same shard per layer,
/// so this is safe in practice.
fn set_expert_slot(
    experts: &mut ExpertWeightStorage,
    slot: Mxfp4Slot,
    new_pair: Mxfp4Pair,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    // `std::mem::replace` to take ownership of the current state, then write
    // back the updated variant.
    let prev = std::mem::replace(experts, placeholder_bf16_storage(device)?);
    let next = match (prev, slot) {
        (ExpertWeightStorage::Bf16Resident { .. }, Mxfp4Slot::GateUp) => {
            ExpertWeightStorage::Mxfp4Packed {
                gate_up_blocks: new_pair,
                down_blocks: empty_mxfp4_pair_placeholder(device)?,
            }
        }
        (ExpertWeightStorage::Bf16Resident { .. }, Mxfp4Slot::Down) => {
            ExpertWeightStorage::Mxfp4Packed {
                gate_up_blocks: empty_mxfp4_pair_placeholder(device)?,
                down_blocks: new_pair,
            }
        }
        (
            ExpertWeightStorage::Mxfp4Packed {
                gate_up_blocks: _,
                down_blocks,
            },
            Mxfp4Slot::GateUp,
        ) => ExpertWeightStorage::Mxfp4Packed {
            gate_up_blocks: new_pair,
            down_blocks,
        },
        (
            ExpertWeightStorage::Mxfp4Packed {
                gate_up_blocks,
                down_blocks: _,
            },
            Mxfp4Slot::Down,
        ) => ExpertWeightStorage::Mxfp4Packed {
            gate_up_blocks,
            down_blocks: new_pair,
        },
    };
    *experts = next;
    Ok(())
}

/// Build a 1-element BF16 placeholder used as a transient stand-in during
/// `std::mem::replace` in `set_expert_slot`. Replaced before the function
/// returns; never observed by `forward`.
fn placeholder_bf16_storage(device: &Arc<CudaDevice>) -> Result<ExpertWeightStorage> {
    let g = Tensor::zeros_dtype(
        Shape::from_dims(&[1, 1, 1]),
        DType::BF16,
        device.clone(),
    )?;
    let d = Tensor::zeros_dtype(
        Shape::from_dims(&[1, 1, 1]),
        DType::BF16,
        device.clone(),
    )?;
    Ok(ExpertWeightStorage::Bf16Resident {
        gate_up_weight: g,
        down_weight: d,
    })
}

/// Empty `Mxfp4Pair` used as a transient placeholder. Crashes if dequantized
/// (the kernel will error on the 0-byte scales). Always overwritten before
/// any forward call when the loader runs against a real Lens checkpoint.
fn empty_mxfp4_pair_placeholder(device: &Arc<CudaDevice>) -> Result<Mxfp4Pair> {
    // 1 block worth (32 elements) so the kernel won't reject the zero-len case.
    use cudarc::driver::DeviceSlice;
    let blocks: cudarc::driver::CudaSlice<u8> = device
        .htod_copy(vec![0u8; 16])
        .map_err(|e| Error::CudaDriver(format!("placeholder blocks htod: {e:?}")))?;
    let scales: cudarc::driver::CudaSlice<u8> = device
        .htod_copy(vec![127u8])
        .map_err(|e| Error::CudaDriver(format!("placeholder scales htod: {e:?}")))?;
    let _ = blocks.len(); // suppress unused-import lint
    Ok(Mxfp4Pair {
        blocks,
        scales,
        dequant_shape: vec![1, 1, 32],
        pre_transpose_shape: vec![1, 1, 32],
    })
}

/// Cheap "is this BF16 tensor populated?" check used as a post-load sanity gate.
/// Reads up to the first 4 BF16 values from the tensor (after `to_dtype` it
/// would be a roundtrip, so use `to_dtype(F32)` and look at the host copy).
fn tensor_has_any_finite(t: &Tensor) -> Result<bool> {
    if t.shape().elem_count() == 0 {
        return Ok(false);
    }
    let head = t.to_dtype(DType::F32)?.to_vec()?;
    let n = head.len().min(8);
    Ok(head[..n].iter().any(|v| v.is_finite() && *v != 0.0))
}

/// Convenience: `HashMap`-API for the loader stats. Unused by core code,
/// kept for test ergonomics.
impl LoaderStats {
    pub fn as_map(&self) -> HashMap<&'static str, usize> {
        let mut m = HashMap::new();
        m.insert("consumed", self.consumed);
        m.insert("skipped", self.skipped);
        m.insert("mxfp4_dequants", self.mxfp4_dequants);
        m
    }
}

// ---------------------------------------------------------------------------
// Tests (synthetic random weights — no real-checkpoint parity yet)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    /// Build a fresh CUDA device handle for unit tests. Returns `None` if no
    /// CUDA device is available (CI without GPU).
    fn dev() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok()
    }

    #[test]
    fn config_defaults_match_lens_spec() {
        let c = GptOssConfig::default();
        assert_eq!(c.hidden_size, 2880);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.num_attention_heads, 64);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.intermediate_size, 2880);
        assert_eq!(c.vocab_size, 201_088);
        assert_eq!(c.num_local_experts, 32);
        assert_eq!(c.experts_per_token, 4);
        assert_eq!(c.sliding_window, 128);
        assert_eq!(c.swiglu_limit, 7.0);
        assert_eq!(c.swiglu_alpha, 1.702);
        assert!(c.attention_bias);
        assert_eq!(c.rope_theta, 150_000.0);
        assert_eq!(c.rope_factor, 32.0);
        assert_eq!(c.rope_beta_fast, 32.0);
        assert_eq!(c.rope_beta_slow, 1.0);
        assert_eq!(c.rope_original_max_pos, 4096);
        assert!(!c.rope_truncate);
        assert_eq!(c.selected_layer_index, vec![5, 11, 17, 23]);
        assert_eq!(c.layer_types.len(), 24);
        // Spot-check alternation.
        assert_eq!(c.layer_types[0], LayerType::Sliding);
        assert_eq!(c.layer_types[1], LayerType::Full);
        assert_eq!(c.layer_types[22], LayerType::Sliding);
        assert_eq!(c.layer_types[23], LayerType::Full);
    }

    #[test]
    fn layer_type_from_str_round_trip() {
        assert_eq!(LayerType::from_str("sliding_attention").unwrap(), LayerType::Sliding);
        assert_eq!(LayerType::from_str("full_attention").unwrap(), LayerType::Full);
        assert!(LayerType::from_str("global_attention").is_err());
    }

    #[test]
    fn build_expert_id_for_row_correctness() {
        // 3 experts, picks: e0=2 rows, e1=0 rows, e2=3 rows  → offsets [0,2,2,5]
        let offsets = vec![0i32, 2, 2, 5];
        let got = build_expert_id_for_row(&offsets, 5);
        assert_eq!(got, vec![0, 0, 2, 2, 2]);
    }

    #[test]
    fn parse_layer_idx_happy_path() {
        assert_eq!(
            parse_layer_idx("model.layers.0.input_layernorm.weight"),
            Some(0)
        );
        assert_eq!(
            parse_layer_idx("model.layers.23.self_attn.sinks"),
            Some(23)
        );
        assert_eq!(
            parse_layer_idx("model.layers.7.mlp.experts.gate_up_proj_blocks"),
            Some(7)
        );
        // Non-layer keys should not match.
        assert_eq!(parse_layer_idx("model.embed_tokens.weight"), None);
        assert_eq!(parse_layer_idx("lm_head.weight"), None);
        assert_eq!(parse_layer_idx("model.norm.weight"), None);
    }

    #[test]
    fn parse_layer_idx_rejects_garbage() {
        assert_eq!(parse_layer_idx("model.layers."), None);
        assert_eq!(parse_layer_idx("model.layers.abc.foo"), None);
        assert_eq!(parse_layer_idx("something_else"), None);
    }

    #[test]
    fn loader_stats_default_zero() {
        let s = LoaderStats::default();
        assert_eq!(s.consumed, 0);
        assert_eq!(s.skipped, 0);
        assert_eq!(s.mxfp4_dequants, 0);
        let m = s.as_map();
        assert_eq!(m["consumed"], 0);
        assert_eq!(m["mxfp4_dequants"], 0);
    }

    // --- GPU-touching tests below: gated on a live CUDA device ---

    fn make_attn(
        cfg: &GptOssConfig,
        layer_type: LayerType,
        device: &Arc<CudaDevice>,
    ) -> Result<GptOssAttention> {
        let hidden = cfg.hidden_size;
        let h = cfg.num_attention_heads;
        let h_kv = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        let q_out = h * d;
        let kv_out = h_kv * d;

        // Tiny std to keep activations within BF16 dynamic range.
        let std = 0.02f32;
        let q_w = Tensor::randn(Shape::from_dims(&[q_out, hidden]), 0.0, std, device.clone())?
            .to_dtype(DType::BF16)?;
        let k_w = Tensor::randn(Shape::from_dims(&[kv_out, hidden]), 0.0, std, device.clone())?
            .to_dtype(DType::BF16)?;
        let v_w = Tensor::randn(Shape::from_dims(&[kv_out, hidden]), 0.0, std, device.clone())?
            .to_dtype(DType::BF16)?;
        let o_w = Tensor::randn(Shape::from_dims(&[hidden, q_out]), 0.0, std, device.clone())?
            .to_dtype(DType::BF16)?;
        let q_b = Tensor::zeros_dtype(Shape::from_dims(&[q_out]), DType::BF16, device.clone())?;
        let k_b = Tensor::zeros_dtype(Shape::from_dims(&[kv_out]), DType::BF16, device.clone())?;
        let v_b = Tensor::zeros_dtype(Shape::from_dims(&[kv_out]), DType::BF16, device.clone())?;
        let o_b = Tensor::zeros_dtype(Shape::from_dims(&[hidden]), DType::BF16, device.clone())?;
        let sinks =
            Tensor::randn(Shape::from_dims(&[h]), 0.0, 0.1, device.clone())?.to_dtype(DType::BF16)?;
        Ok(GptOssAttention {
            q_weight: q_w,
            q_bias: q_b,
            k_weight: k_w,
            k_bias: k_b,
            v_weight: v_w,
            v_bias: v_b,
            o_weight: o_w,
            o_bias: o_b,
            sinks,
            num_heads: h,
            num_kv_heads: h_kv,
            head_dim: d,
            layer_type,
            scaling: (d as f32).powf(-0.5),
        })
    }

    fn make_moe(cfg: &GptOssConfig, device: &Arc<CudaDevice>) -> Result<GptOssMoE> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let e = cfg.num_local_experts;
        let std = 0.02f32;
        let router_w =
            Tensor::randn(Shape::from_dims(&[e, hidden]), 0.0, std, device.clone())?.to_dtype(DType::BF16)?;
        let router_b =
            Tensor::zeros_dtype(Shape::from_dims(&[e]), DType::BF16, device.clone())?;
        let gate_up_w = Tensor::randn(
            Shape::from_dims(&[e, hidden, 2 * inter]),
            0.0,
            std,
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let gate_up_b = Tensor::zeros_dtype(
            Shape::from_dims(&[e, 2 * inter]),
            DType::BF16,
            device.clone(),
        )?;
        let down_w = Tensor::randn(
            Shape::from_dims(&[e, inter, hidden]),
            0.0,
            std,
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let down_b = Tensor::zeros_dtype(
            Shape::from_dims(&[e, hidden]),
            DType::BF16,
            device.clone(),
        )?;
        Ok(GptOssMoE {
            router_weight: router_w,
            router_bias: router_b,
            gate_up_bias: gate_up_b,
            down_bias: down_b,
            experts: ExpertWeightStorage::Bf16Resident {
                gate_up_weight: gate_up_w,
                down_weight: down_w,
            },
            num_experts: e,
            top_k: cfg.experts_per_token,
            hidden_size: hidden,
            intermediate_size: inter,
            swiglu_limit: cfg.swiglu_limit,
            swiglu_alpha: cfg.swiglu_alpha,
        })
    }

    /// Tiny config for fast smoke tests. Same architectural shape as Lens but
    /// scaled way down to keep weight allocation small and tests fast.
    fn tiny_cfg() -> GptOssConfig {
        let mut c = GptOssConfig::default();
        c.hidden_size = 64;
        c.num_attention_heads = 4;
        c.num_key_value_heads = 2;
        c.head_dim = 16;
        c.intermediate_size = 64;
        c.num_local_experts = 4;
        c.experts_per_token = 2;
        c.sliding_window = 4;
        // Keep layer_types short — only need length matching what we test.
        c.num_hidden_layers = 2;
        c.layer_types = vec![LayerType::Sliding, LayerType::Full];
        c.selected_layer_index = vec![1];
        c
    }

    #[test]
    fn attention_forward_shape_full_layer() {
        let Some(device) = dev() else {
            return;
        };
        let cfg = tiny_cfg();
        let attn = make_attn(&cfg, LayerType::Full, &device).expect("make attn");
        let b = 1;
        let s = 8;
        let x = Tensor::randn(
            Shape::from_dims(&[b, s, cfg.hidden_size]),
            0.0,
            1.0,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        // Build RoPE tables.
        let rope_cfg = cfg.clone();
        let mut rope = GptOssRotaryEmbedding::new(&rope_cfg).expect("rope");
        let (cos, sin) = rope.freqs_for(s, &device).expect("freqs");
        let kit = GptOssMaskKit::build(s, cfg.sliding_window, &device).expect("masks");
        let out = attn
            .forward(&x, &cos, &sin, Some(&kit.full_mask), Some(&kit.sliding_mask))
            .expect("attn forward");
        assert_eq!(out.shape().dims(), &[b, s, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn attention_forward_shape_sliding_layer() {
        let Some(device) = dev() else {
            return;
        };
        let cfg = tiny_cfg();
        let attn = make_attn(&cfg, LayerType::Sliding, &device).expect("make attn");
        let b = 1;
        let s = 16;
        let x = Tensor::randn(
            Shape::from_dims(&[b, s, cfg.hidden_size]),
            0.0,
            1.0,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let mut rope = GptOssRotaryEmbedding::new(&cfg).expect("rope");
        let (cos, sin) = rope.freqs_for(s, &device).expect("freqs");
        let kit = GptOssMaskKit::build(s, cfg.sliding_window, &device).expect("masks");
        let out = attn
            .forward(&x, &cos, &sin, Some(&kit.full_mask), Some(&kit.sliding_mask))
            .expect("attn forward");
        assert_eq!(out.shape().dims(), &[b, s, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn moe_forward_shape() {
        let Some(device) = dev() else {
            return;
        };
        let cfg = tiny_cfg();
        let moe = make_moe(&cfg, &device).expect("make moe");
        let b = 1;
        let s = 8;
        let x = Tensor::randn(
            Shape::from_dims(&[b, s, cfg.hidden_size]),
            0.0,
            1.0,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = moe.forward(&x).expect("moe forward");
        assert_eq!(out.shape().dims(), &[b, s, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn layer_forward_shape() {
        let Some(device) = dev() else {
            return;
        };
        let cfg = tiny_cfg();
        let attn = make_attn(&cfg, LayerType::Sliding, &device).expect("make attn");
        let moe = make_moe(&cfg, &device).expect("make moe");
        let pre_norm =
            Tensor::ones_dtype(Shape::from_dims(&[cfg.hidden_size]), DType::BF16, device.clone())
                .expect("pre_norm");
        let post_norm =
            Tensor::ones_dtype(Shape::from_dims(&[cfg.hidden_size]), DType::BF16, device.clone())
                .expect("post_norm");
        let layer = GptOssLayer {
            pre_norm,
            post_norm,
            attn,
            mlp: moe,
            rms_norm_eps: cfg.rms_norm_eps,
        };
        let b = 1;
        let s = 8;
        let x = Tensor::randn(
            Shape::from_dims(&[b, s, cfg.hidden_size]),
            0.0,
            1.0,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let mut rope = GptOssRotaryEmbedding::new(&cfg).expect("rope");
        let (cos, sin) = rope.freqs_for(s, &device).expect("freqs");
        let kit = GptOssMaskKit::build(s, cfg.sliding_window, &device).expect("masks");
        let out = layer
            .forward(&x, &cos, &sin, Some(&kit.full_mask), Some(&kit.sliding_mask))
            .expect("layer forward");
        assert_eq!(out.shape().dims(), &[b, s, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    /// End-to-end smoke test: build a tiny encoder, write synthetic weights into
    /// the empty-shape tensors, run encode, check captured hidden states' shape.
    ///
    /// This is the closest we can get to a parity test without loading the real
    /// 20B checkpoint. It exercises:
    ///   - `GptOssEncoder::new` (allocator paths for all 2 tiny layers)
    ///   - `encode()` orchestration (embed → mask kit → layer loop → capture)
    ///   - capture-position correctness for selected_layers
    #[test]
    fn encoder_forward_smoke_shape() {
        let Some(device) = dev() else {
            return;
        };
        let cfg = tiny_cfg();
        // tiny_cfg has 2 layers, select layer 1 only (last).
        let mut enc = GptOssEncoder::new(cfg.clone(), &device).expect("encoder build");

        // Replace embed_tokens with a randn tensor so embedding lookup isn't all-zero.
        enc.embed_tokens = Tensor::randn(
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            0.0,
            0.02,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        // Replace each layer's weights with non-zero values so the forward
        // produces finite (not all-zero) output.
        for layer in &mut enc.layers {
            layer.pre_norm = Tensor::ones_dtype(
                Shape::from_dims(&[cfg.hidden_size]),
                DType::BF16,
                device.clone(),
            )
            .unwrap();
            layer.post_norm = Tensor::ones_dtype(
                Shape::from_dims(&[cfg.hidden_size]),
                DType::BF16,
                device.clone(),
            )
            .unwrap();
            let std = 0.02f32;
            let h_q = cfg.num_attention_heads * cfg.head_dim;
            let h_kv = cfg.num_key_value_heads * cfg.head_dim;
            layer.attn.q_weight =
                Tensor::randn(Shape::from_dims(&[h_q, cfg.hidden_size]), 0.0, std, device.clone())
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
            layer.attn.k_weight =
                Tensor::randn(Shape::from_dims(&[h_kv, cfg.hidden_size]), 0.0, std, device.clone())
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
            layer.attn.v_weight =
                Tensor::randn(Shape::from_dims(&[h_kv, cfg.hidden_size]), 0.0, std, device.clone())
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
            layer.attn.o_weight =
                Tensor::randn(Shape::from_dims(&[cfg.hidden_size, h_q]), 0.0, std, device.clone())
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
            layer.attn.sinks =
                Tensor::randn(Shape::from_dims(&[cfg.num_attention_heads]), 0.0, 0.1, device.clone())
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
            layer.mlp.router_weight = Tensor::randn(
                Shape::from_dims(&[cfg.num_local_experts, cfg.hidden_size]),
                0.0,
                std,
                device.clone(),
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            let gate_up_weight = Tensor::randn(
                Shape::from_dims(&[cfg.num_local_experts, cfg.hidden_size, 2 * cfg.intermediate_size]),
                0.0,
                std,
                device.clone(),
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            let down_weight = Tensor::randn(
                Shape::from_dims(&[cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size]),
                0.0,
                std,
                device.clone(),
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            layer.mlp.experts = ExpertWeightStorage::Bf16Resident {
                gate_up_weight,
                down_weight,
            };
        }

        let b = 1;
        let s = 8;
        // Small input ids; vocab_size for tiny_cfg is the Lens default (201088).
        let ids_data: Vec<f32> = (0..b * s).map(|i| (i % cfg.vocab_size) as f32).collect();
        let input_ids = Tensor::from_vec(ids_data, Shape::from_dims(&[b, s]), device.clone())
            .unwrap()
            .to_dtype(DType::I32)
            .unwrap();
        let mask_data: Vec<f32> = vec![1.0; b * s];
        let attention_mask =
            Tensor::from_vec(mask_data, Shape::from_dims(&[b, s]), device.clone())
                .unwrap()
                .to_dtype(DType::I32)
                .unwrap();

        let captures = enc
            .encode(&input_ids, &attention_mask)
            .expect("encode smoke");
        assert_eq!(captures.len(), 1, "tiny_cfg selects 1 layer");
        assert_eq!(captures[0].shape().dims(), &[b, s, cfg.hidden_size]);
        assert_eq!(captures[0].dtype(), DType::BF16);
    }

    #[test]
    fn encoder_new_allocates_max_layer_plus_one() {
        let Some(device) = dev() else {
            return;
        };
        let mut cfg = tiny_cfg();
        // Pick a non-last layer to confirm we allocate `max+1` layers, not all.
        cfg.num_hidden_layers = 4;
        cfg.layer_types = vec![
            LayerType::Sliding,
            LayerType::Full,
            LayerType::Sliding,
            LayerType::Full,
        ];
        cfg.selected_layer_index = vec![1]; // only need layers 0,1
        let enc = GptOssEncoder::new(cfg.clone(), &device).expect("encoder build");
        assert_eq!(
            enc.layers.len(),
            2,
            "max(selected)=1 → allocate 2 layers (0 and 1)"
        );
        assert_eq!(enc.selected_layers, vec![1]);
    }
}
