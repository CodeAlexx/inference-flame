//! T5Gemma2 text-only encoder for Motif-Video-2B inference — pure flame-core.
//!
//! ## Scope
//!
//! This module implements the **text encoder** of `T5Gemma2Model` (model_type
//! `t5gemma2_encoder`), the conditioning network used by Motif-Video-2B. It is
//! standalone: no decoder, no vision tower, no multi-modal projector.
//!
//! ## Architecture (from `transformers.models.t5gemma2.modeling_t5gemma2`)
//!
//! - 34 encoder layers, hidden_size=2560
//! - 8 attention heads, 4 KV heads (GQA 2:1), head_dim=256 (explicit in config,
//!   NOT 128 — that's what makes `query_pre_attn_scalar=256` equivalent to
//!   default 1/sqrt(head_dim) scaling)
//! - intermediate_size=10240, activation = `gelu_pytorch_tanh`
//! - `attention_bias=false`, `rms_norm_eps=1e-6`
//! - vocab_size=262144, `tie_word_embeddings=true` (we don't use unembed here)
//! - `query_pre_attn_scalar=256` → attention scale = 1/16 = 1/sqrt(head_dim=256)
//!
//! ### Alternating attention (layer_types, 34 entries)
//!
//! - 29 × `sliding_attention`: bidirectional sliding window, RoPE theta=10000
//! - 5 × `full_attention` at indices {5, 11, 17, 23, 29}: bidirectional (no
//!   window), RoPE theta=1_000_000 with linear factor 8 (attention_scaling=1/8)
//!
//! The pattern is 5 sliding + 1 full repeating.
//!
//! ### Per-layer block (`T5Gemma2EncoderLayer.forward`)
//!
//! ```text
//! residual = x
//! x = pre_self_attn_layernorm(x)           # 4 norms per layer — one before
//! x = self_attn(x, position_embeddings,    #   and one AFTER attention, then
//!               attention_mask)             #   one before and one after MLP
//! x = post_self_attn_layernorm(x)
//! x = residual + x
//!
//! residual = x
//! x = pre_feedforward_layernorm(x)
//! x = mlp(x)  # gelu_tanh(gate(x)) * up(x) -> down(...)
//! x = post_feedforward_layernorm(x)
//! x = residual + x
//! ```
//!
//! RMSNorm follows the Gemma formulation: `x * (1 + weight)` in FP32, cast to BF16.
//! Per-head RMSNorm on Q and K (`head_dim` sized weight each) before RoPE.
//!
//! ### Attention mask — BIDIRECTIONAL
//!
//! Despite `use_bidirectional_attention=false` in the config (which controls the
//! decoder self-attention), `T5Gemma2TextEncoder.forward` unconditionally calls
//! `create_bidirectional_mask`. Full-attention layers are fully bidirectional
//! (padding-only mask); sliding layers use a symmetric window of
//! `left = (sw + 1) / 2` and `right = sw / 2 + 1` (for sw=1024: 512 + 513 = 1025
//! reachable positions centered on q).
//!
//! ### Embedding
//!
//! `T5Gemma2TextScaledWordEmbedding`: plain lookup then `* sqrt(hidden_size)`.
//! The `eoi_embedding` row is a special token embedding injected into the
//! embedding table; we keep just the standard `weight` path for text-only input.
//!
//! ## Weight key layout (actual safetensors — verified against HF header)
//!
//! Flat `encoder.` prefix (no `text_model.` intermediate like the PyTorch
//! module hierarchy would imply):
//!
//! - `encoder.embed_tokens.weight` (`[262144, 2560]`)
//! - `encoder.embed_tokens.eoi_embedding` (`[2560]`) — unused here
//! - `encoder.layers.{i}.pre_self_attn_layernorm.weight` (`[2560]`)
//! - `encoder.layers.{i}.post_self_attn_layernorm.weight` (`[2560]`)
//! - `encoder.layers.{i}.pre_feedforward_layernorm.weight` (`[2560]`)
//! - `encoder.layers.{i}.post_feedforward_layernorm.weight` (`[2560]`)
//! - `encoder.layers.{i}.self_attn.q_proj.weight` (`[2048, 2560]`) — 8 heads × 256
//! - `encoder.layers.{i}.self_attn.k_proj.weight` (`[1024, 2560]`) — 4 kv × 256
//! - `encoder.layers.{i}.self_attn.v_proj.weight` (`[1024, 2560]`)
//! - `encoder.layers.{i}.self_attn.o_proj.weight` (`[2560, 2048]`)
//! - `encoder.layers.{i}.self_attn.q_norm.weight` (`[256]`) — per-head norm
//! - `encoder.layers.{i}.self_attn.k_norm.weight` (`[256]`)
//! - `encoder.layers.{i}.mlp.gate_proj.weight` (`[10240, 2560]`)
//! - `encoder.layers.{i}.mlp.up_proj.weight` (`[10240, 2560]`)
//! - `encoder.layers.{i}.mlp.down_proj.weight` (`[2560, 10240]`)
//! - `encoder.norm.weight` (`[2560]`)
//!
//! All weights are BF16. F32 is used only for RoPE angles and RMSNorm internals.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Attention type for a given layer index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Bidirectional sliding window (theta=10000, no RoPE scaling).
    Sliding,
    /// Full bidirectional attention (theta=1_000_000, linear scale factor=8).
    Full,
}

/// T5Gemma2 text encoder config.
#[derive(Debug, Clone)]
pub struct T5Gemma2Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub rope_theta_sliding: f64,
    pub rope_theta_full: f64,
    /// Linear-RoPE divisor for full-attention layers (`attention_scaling = 1/factor`).
    pub full_rope_factor: f32,
    /// Per-layer attention type. Must have `num_layers` entries.
    pub layer_types: Vec<LayerType>,
    /// Query pre-attention scalar. Scale = 1/sqrt(query_pre_attn_scalar).
    pub query_pre_attn_scalar: f32,
}

impl Default for T5Gemma2Config {
    /// Motif-Video-2B defaults, matching
    /// `/home/alex/.serenity/models/checkpoints/motif-video-2b/text_encoder/config.json`
    /// (field `encoder.text_config`).
    fn default() -> Self {
        // Pattern: 5 sliding + 1 full, repeating. Full indices: 5, 11, 17, 23, 29.
        let mut layer_types = Vec::with_capacity(34);
        for i in 0..34 {
            if (i + 1) % 6 == 0 {
                layer_types.push(LayerType::Full);
            } else {
                layer_types.push(LayerType::Sliding);
            }
        }
        Self {
            hidden_size: 2560,
            num_layers: 34,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            intermediate_size: 10240,
            rms_norm_eps: 1e-6,
            sliding_window: 1024,
            rope_theta_sliding: 10_000.0,
            rope_theta_full: 1_000_000.0,
            full_rope_factor: 8.0,
            layer_types,
            query_pre_attn_scalar: 256.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// T5Gemma2 text-only encoder.
///
/// Holds a flat weight map and the device. 2D projection weights are
/// pre-transposed at construction time so each forward pass is a plain matmul
/// (consistent with the pattern used by `qwen3_encoder` / `qwen25vl_encoder`).
pub struct T5Gemma2Encoder {
    weights: HashMap<String, Tensor>,
    config: T5Gemma2Config,
    device: Arc<CudaDevice>,
}

impl T5Gemma2Encoder {
    /// Construct from pre-loaded BF16 weights on device.
    ///
    /// All projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
    /// `gate_proj`, `up_proj`, `down_proj`) are pre-transposed so they can be
    /// consumed directly by `matmul`.
    pub fn new(
        weights: HashMap<String, Tensor>,
        config: T5Gemma2Config,
        device: Arc<CudaDevice>,
    ) -> Self {
        debug_assert_eq!(
            config.layer_types.len(),
            config.num_layers,
            "layer_types length must equal num_layers"
        );

        // Pre-transpose 2D projection weights from [out, in] → [in, out].
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            if !is_projection_key(&key) {
                continue;
            }
            let w = &weights[&key];
            if w.shape().dims().len() != 2 {
                continue;
            }
            match flame_core::bf16_elementwise::transpose2d_bf16(w) {
                Ok(wt) => {
                    weights.insert(key, wt);
                }
                Err(e) => {
                    log::warn!("[T5Gemma2] transpose failed for {key}: {e}");
                }
            }
        }

        Self {
            weights,
            config,
            device,
        }
    }

    /// Auto-derive a config from the weight map, validating shapes.
    ///
    /// Returns an error if required keys are missing or shapes disagree with
    /// the T5Gemma2 text-encoder defaults used by Motif-Video-2B.
    pub fn config_from_weights(weights: &HashMap<String, Tensor>) -> Result<T5Gemma2Config> {
        let embed_key = "encoder.embed_tokens.weight";
        let embed = weights.get(embed_key).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Missing {embed_key}. First 10 keys: {:?}",
                weights.keys().take(10).collect::<Vec<_>>()
            ))
        })?;
        let embed_dims = embed.shape().dims();
        if embed_dims.len() != 2 {
            return Err(Error::InvalidShape(format!(
                "{embed_key} must be 2D, got {:?}",
                embed_dims
            )));
        }
        let hidden_size = embed_dims[1];

        // Count layers.
        let mut num_layers = 0usize;
        while weights.contains_key(&format!("encoder.layers.{num_layers}.self_attn.q_proj.weight")) {
            num_layers += 1;
        }
        if num_layers == 0 {
            return Err(Error::InvalidInput(
                "No encoder.layers.*.self_attn.q_proj.weight keys found".into(),
            ));
        }

        // Derive shapes from layer 0.
        let q_w = weights
            .get("encoder.layers.0.self_attn.q_proj.weight")
            .ok_or_else(|| Error::InvalidInput("layer 0 q_proj missing".into()))?;
        let k_w = weights
            .get("encoder.layers.0.self_attn.k_proj.weight")
            .ok_or_else(|| Error::InvalidInput("layer 0 k_proj missing".into()))?;
        let q_norm = weights
            .get("encoder.layers.0.self_attn.q_norm.weight")
            .ok_or_else(|| Error::InvalidInput("layer 0 q_norm missing".into()))?;
        let gate_w = weights
            .get("encoder.layers.0.mlp.gate_proj.weight")
            .ok_or_else(|| Error::InvalidInput("layer 0 gate_proj missing".into()))?;

        let head_dim = q_norm.shape().dims()[0];
        let q_out = q_w.shape().dims()[0];
        let k_out = k_w.shape().dims()[0];
        if q_out % head_dim != 0 || k_out % head_dim != 0 {
            return Err(Error::InvalidShape(format!(
                "q/k proj out dims ({q_out}/{k_out}) not divisible by head_dim={head_dim}"
            )));
        }
        let num_heads = q_out / head_dim;
        let num_kv_heads = k_out / head_dim;
        let intermediate_size = gate_w.shape().dims()[0];

        let cfg = T5Gemma2Config::default();

        if hidden_size != cfg.hidden_size {
            return Err(Error::InvalidShape(format!(
                "hidden_size mismatch: expected {}, got {}",
                cfg.hidden_size, hidden_size
            )));
        }
        if num_layers != cfg.num_layers {
            return Err(Error::InvalidShape(format!(
                "num_layers mismatch: expected {}, got {}",
                cfg.num_layers, num_layers
            )));
        }
        if head_dim != cfg.head_dim
            || num_heads != cfg.num_heads
            || num_kv_heads != cfg.num_kv_heads
            || intermediate_size != cfg.intermediate_size
        {
            return Err(Error::InvalidShape(format!(
                "shape mismatch: got hidden={hidden_size} layers={num_layers} heads={num_heads} kv={num_kv_heads} head_dim={head_dim} ff={intermediate_size}; \
                 expected hidden={} layers={} heads={} kv={} head_dim={} ff={}",
                cfg.hidden_size,
                cfg.num_layers,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.intermediate_size
            )));
        }

        Ok(cfg)
    }

    // -----------------------------------------------------------------------
    // Weight accessor
    // -----------------------------------------------------------------------

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight key: {key}")))
    }

    // -----------------------------------------------------------------------
    // Gemma RMSNorm: F32 normalize, multiply by (1 + weight), cast back to BF16.
    //
    // This is the Gemma formulation (see PR 29402); different from standard
    // RMSNorm which just multiplies by `weight`.
    // -----------------------------------------------------------------------

    fn gemma_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;

        let normed_f32 = flame_core::cuda_ops_bf16::rms_norm_bf16_to_f32(&x_2d, eps)?;
        let weight_f32 = weight.to_dtype(DType::F32)?;
        let one_plus_w = weight_f32.add_scalar(1.0)?;
        let result_f32 =
            normed_f32.mul(&one_plus_w.unsqueeze(0)?.expand(&[batch, hidden])?)?;
        let result = result_f32.to_dtype(DType::BF16)?;
        result.reshape(&dims)
    }

    // -----------------------------------------------------------------------
    // Linear (weight pre-transposed to [in, out])
    // -----------------------------------------------------------------------

    fn linear_3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(weight_t)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    // -----------------------------------------------------------------------
    // GQA expand: repeat KV heads to match Q head count.
    // -----------------------------------------------------------------------

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let b = dims[0];
        let h_kv = dims[1];
        let n = dims[2];
        let d = dims[3];
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    // -----------------------------------------------------------------------
    // RoPE half-split (HF rotate_half convention)
    // -----------------------------------------------------------------------

    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
    }

    // -----------------------------------------------------------------------
    // Mask construction — BIDIRECTIONAL per actual T5Gemma2 text encoder.
    //
    // Semantics: 1.0 = attend, 0.0 = masked (flame sdpa interprets (1 - mask)
    // as the additive -inf penalty).
    //
    // - Full: padding-only mask. A token at position q can attend to any
    //   position k such that `attention_mask[k] != 0`. Pad rows are allowed to
    //   attend to real columns too (matches HF behavior and prevents NaN from
    //   a fully-masked row).
    // - Sliding: padding AND symmetric window. For sliding_window=1024,
    //   `left = (1024+1)/2 = 512`, `right = 1024/2 + 1 = 513`.
    //   A token q attends to k iff `attention_mask[k] != 0` AND
    //   `-right < (q - k) < left` (i.e. `k > q - left` AND `k < q + right`),
    //   matching `sliding_window_mask_function(is_causal=False)`.
    // -----------------------------------------------------------------------

    fn build_attn_mask(
        attention_mask: &[i32],
        sliding_window: Option<usize>,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let seq = attention_mask.len();
        let mut data = vec![0.0f32; seq * seq];

        let (left, right) = match sliding_window {
            Some(sw) => ((sw + 1) / 2, sw / 2 + 1),
            None => (usize::MAX, usize::MAX),
        };

        for q in 0..seq {
            for k in 0..seq {
                if attention_mask[k] == 0 {
                    continue;
                }
                let in_window = match sliding_window {
                    None => true,
                    Some(_) => {
                        // left window: q - k < left AND q >= k (dist >= 0)
                        // right window: k - q < right AND k > q
                        let left_ok = k <= q && (q - k) < left;
                        let right_ok = k > q && (k - q) < right;
                        left_ok || right_ok
                    }
                };
                if in_window {
                    data[q * seq + k] = 1.0;
                }
            }
        }
        let _ = (left, right); // silence unused when sliding_window is None

        let mask = Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq, seq]),
            device.clone(),
        )?;
        mask.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Embedding lookup with sqrt(hidden_size) scale
    // -----------------------------------------------------------------------

    fn embed_tokens(&self, token_ids: &[i32]) -> Result<Tensor> {
        let embed_w = self.w("encoder.embed_tokens.weight")?;
        let seq = token_ids.len();

        let ids = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;

        let selected = embed_w.index_select0(&ids)?;
        let scale = (self.config.hidden_size as f32).sqrt();
        let scaled = selected.mul_scalar(scale)?;
        scaled.unsqueeze(0) // [1, seq, hidden]
    }

    // -----------------------------------------------------------------------
    // Single encoder layer
    // -----------------------------------------------------------------------

    fn layer_forward(
        &self,
        layer_idx: usize,
        hidden: &Tensor,
        cos_sliding: &Tensor,
        sin_sliding: &Tensor,
        cos_full: &Tensor,
        sin_full: &Tensor,
        sliding_mask: &Tensor,
        full_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let is_full = cfg.layer_types[layer_idx] == LayerType::Full;
        let prefix = format!("encoder.layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        // --- Self-attention block ---

        // 1. pre_self_attn_layernorm
        let pre_norm_w = self.w(&format!("{prefix}.pre_self_attn_layernorm.weight"))?;
        let x = Self::gemma_rms_norm(hidden, pre_norm_w, cfg.rms_norm_eps)?;

        // 2. Q, K, V projections
        let q = Self::linear_3d(&x, self.w(&format!("{prefix}.self_attn.q_proj.weight"))?)?;
        let k = Self::linear_3d(&x, self.w(&format!("{prefix}.self_attn.k_proj.weight"))?)?;
        let v = Self::linear_3d(&x, self.w(&format!("{prefix}.self_attn.v_proj.weight"))?)?;

        // Reshape to [B, heads, N, head_dim]
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // 3. Per-head Q/K RMSNorm (Gemma formulation).
        //    Flatten to [B*heads*N, head_dim] and apply norm using shared kernel.
        let q_norm_w = self.w(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm_w = self.w(&format!("{prefix}.self_attn.k_norm.weight"))?;

        let q = {
            let q_flat = q.reshape(&[b * h * n, d])?;
            let normed = Self::gemma_rms_norm(&q_flat, q_norm_w, cfg.rms_norm_eps)?;
            normed.reshape(&[b, h, n, d])?
        };
        let k = {
            let k_flat = k.reshape(&[b * h_kv * n, d])?;
            let normed = Self::gemma_rms_norm(&k_flat, k_norm_w, cfg.rms_norm_eps)?;
            normed.reshape(&[b, h_kv, n, d])?
        };

        // 4. RoPE — choose table based on layer type, narrow to actual seq_len.
        let (cos_tbl, sin_tbl) = if is_full {
            (cos_full, sin_full)
        } else {
            (cos_sliding, sin_sliding)
        };
        let cos = cos_tbl.narrow(2, 0, n)?;
        let sin = sin_tbl.narrow(2, 0, n)?;
        let q = Self::apply_rope(&q, &cos, &sin)?;
        let k = Self::apply_rope(&k, &cos, &sin)?;

        // 5. GQA expand K, V.
        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        // 6. SDPA. flame_sdpa internally scales by 1/sqrt(head_dim).
        //    For T5Gemma2, the intended scale is 1/sqrt(query_pre_attn_scalar).
        //    Since head_dim == query_pre_attn_scalar == 256, these coincide.
        debug_assert_eq!(
            cfg.head_dim as f32, cfg.query_pre_attn_scalar,
            "sdpa scale assumes head_dim == query_pre_attn_scalar"
        );
        let mask = if is_full { full_mask } else { sliding_mask };
        let attn = flame_sdpa(&q, &k, &v, Some(mask))?;

        // [B, H, N, D] → [B, N, H*D]
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        // 7. o_proj
        let attn = Self::linear_3d(&attn, self.w(&format!("{prefix}.self_attn.o_proj.weight"))?)?;

        // 8. post_self_attn_layernorm + residual
        let post_attn_w =
            self.w(&format!("{prefix}.post_self_attn_layernorm.weight"))?;
        let attn = Self::gemma_rms_norm(&attn, post_attn_w, cfg.rms_norm_eps)?;
        let hidden = hidden.add(&attn)?;

        // --- MLP block ---

        // 9. pre_feedforward_layernorm
        let pre_ff_w = self.w(&format!("{prefix}.pre_feedforward_layernorm.weight"))?;
        let y = Self::gemma_rms_norm(&hidden, pre_ff_w, cfg.rms_norm_eps)?;

        // 10. MLP: gelu_tanh(gate(y)) * up(y) → down
        let gate = Self::linear_3d(&y, self.w(&format!("{prefix}.mlp.gate_proj.weight"))?)?;
        let up = Self::linear_3d(&y, self.w(&format!("{prefix}.mlp.up_proj.weight"))?)?;
        // Tensor::gelu in flame-core is the tanh approximation — matches
        // `gelu_pytorch_tanh` exactly.
        let mlp = gate.gelu()?.mul(&up)?;
        let mlp = Self::linear_3d(&mlp, self.w(&format!("{prefix}.mlp.down_proj.weight"))?)?;

        // 11. post_feedforward_layernorm + residual
        let post_ff_w = self.w(&format!("{prefix}.post_feedforward_layernorm.weight"))?;
        let mlp = Self::gemma_rms_norm(&mlp, post_ff_w, cfg.rms_norm_eps)?;
        hidden.add(&mlp)
    }

    // -----------------------------------------------------------------------
    // Full forward
    // -----------------------------------------------------------------------

    /// Encode a sequence of token ids. Returns `[1, seq_len, hidden_size]`
    /// after the final RMSNorm.
    ///
    /// All tokens are treated as real (attention_mask = all ones). For masked
    /// encoding, use `encode_with_mask`.
    pub fn encode(&self, token_ids: &[i32]) -> Result<Tensor> {
        let attn_mask = vec![1i32; token_ids.len()];
        self.encode_with_mask(token_ids, &attn_mask)
    }

    /// Encode with an explicit attention mask (1 = real token, 0 = pad).
    ///
    /// `token_ids.len()` must equal `attention_mask.len()`.
    pub fn encode_with_mask(
        &self,
        token_ids: &[i32],
        attention_mask: &[i32],
    ) -> Result<Tensor> {
        if token_ids.len() != attention_mask.len() {
            return Err(Error::InvalidInput(format!(
                "token_ids.len()={} but attention_mask.len()={}",
                token_ids.len(),
                attention_mask.len()
            )));
        }
        let cfg = &self.config;
        let seq = token_ids.len();

        // Embed + sqrt(hidden_size) scale.
        let mut hidden = self.embed_tokens(token_ids)?;

        // Build two RoPE tables (sliding theta + full theta with linear scale).
        let (cos_sliding, sin_sliding) = build_rope_table(
            seq,
            cfg.head_dim,
            cfg.rope_theta_sliding,
            1.0, // default RoPE, no scaling
            &self.device,
        )?;
        let (cos_full, sin_full) = build_rope_table(
            seq,
            cfg.head_dim,
            cfg.rope_theta_full,
            cfg.full_rope_factor as f64, // linear: divide positions by factor
            &self.device,
        )?;

        // Two attention masks.
        let sliding_mask =
            Self::build_attn_mask(attention_mask, Some(cfg.sliding_window), &self.device)?;
        let full_mask = Self::build_attn_mask(attention_mask, None, &self.device)?;

        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(
                i,
                &hidden,
                &cos_sliding,
                &sin_sliding,
                &cos_full,
                &sin_full,
                &sliding_mask,
                &full_mask,
            )?;
        }

        // Final RMSNorm (Gemma formulation).
        let final_norm_w = self.w("encoder.norm.weight")?;
        Self::gemma_rms_norm(&hidden, final_norm_w, cfg.rms_norm_eps)
    }

    /// Parity-test version: runs the full encoder and captures per-layer
    /// intermediates as a HashMap for external comparison. Tags: `"embed"`
    /// after the embedding scale, `"layer_{i}"` after each of the 34 layers,
    /// and `"final"` after the trailing RMSNorm.
    pub fn encode_with_dump(
        &self,
        token_ids: &[i32],
        attention_mask: &[i32],
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        if token_ids.len() != attention_mask.len() {
            return Err(Error::InvalidInput(format!(
                "token_ids.len()={} but attention_mask.len()={}",
                token_ids.len(),
                attention_mask.len()
            )));
        }
        let cfg = &self.config;
        let seq = token_ids.len();
        let mut dump = std::collections::HashMap::new();

        let mut hidden = self.embed_tokens(token_ids)?;
        dump.insert("embed".to_string(), hidden.clone());

        let (cos_sliding, sin_sliding) = build_rope_table(
            seq, cfg.head_dim, cfg.rope_theta_sliding, 1.0, &self.device,
        )?;
        let (cos_full, sin_full) = build_rope_table(
            seq, cfg.head_dim, cfg.rope_theta_full, cfg.full_rope_factor as f64, &self.device,
        )?;
        let sliding_mask = Self::build_attn_mask(attention_mask, Some(cfg.sliding_window), &self.device)?;
        let full_mask = Self::build_attn_mask(attention_mask, None, &self.device)?;

        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(
                i, &hidden, &cos_sliding, &sin_sliding, &cos_full, &sin_full,
                &sliding_mask, &full_mask,
            )?;
            dump.insert(format!("layer_{i}"), hidden.clone());
        }

        let final_norm_w = self.w("encoder.norm.weight")?;
        let final_out = Self::gemma_rms_norm(&hidden, final_norm_w, cfg.rms_norm_eps)?;
        dump.insert("final".to_string(), final_out.clone());
        Ok((final_out, dump))
    }

    /// Config accessor.
    pub fn config(&self) -> &T5Gemma2Config {
        &self.config
    }

    /// Output hidden size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Device accessor.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build RoPE cos/sin tables.
///
/// Shape: `[1, 1, seq_len, head_dim/2]` BF16 — matches
/// `bf16_ops::rope_halfsplit_bf16` expectations.
///
/// Follows T5Gemma2's "linear" RoPE rule for full layers, per
/// `transformers.modeling_rope_utils._compute_linear_scaling_rope_parameters`:
/// ```
/// inv_freq = 1 / theta^(2i/dim)
/// inv_freq /= factor          # linear scaling applied to INV_FREQ
/// angle = pos * inv_freq       # angle naturally shrinks by 1/factor
/// attention_factor = 1.0       # NO extra scaling of cos/sin
/// ```
///
/// Previous version of this function multiplied `cos/sin` by `1/factor`
/// instead — that shrunk RoPE rotation amplitudes to ~13% for factor=8,
/// effectively disabling RoPE on the 5 full-attention layers. It took
/// ~20 cos points off the layer-5 parity (0.99992 → 0.99919) and
/// compounded through layers 11/17/23/29, ending at 0.975 vs Python's
/// final output.
///
/// For sliding layers, `factor = 1.0` → no change (trivial division).
fn build_rope_table(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    scaling_factor: f64,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;

    let mut cos_data = vec![0.0f32; seq_len * half];
    let mut sin_data = vec![0.0f32; seq_len * half];

    for pos in 0..seq_len {
        for i in 0..half {
            // `inv_freq /= factor` per Python — divide the frequency, NOT the amplitude.
            let inv_freq = (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) / scaling_factor;
            let angle = (pos as f64) * inv_freq;
            cos_data[pos * half + i] = angle.cos() as f32;
            sin_data[pos * half + i] = angle.sin() as f32;
        }
    }

    let cos = Tensor::from_vec(
        cos_data,
        Shape::from_dims(&[1, 1, seq_len, half]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(
        sin_data,
        Shape::from_dims(&[1, 1, seq_len, half]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Returns true if the weight key is a projection weight that benefits from
/// pre-transpose. Excludes RMSNorm weights, embedding, and 1D biases.
fn is_projection_key(key: &str) -> bool {
    if !key.ends_with(".weight") {
        return false;
    }
    if key.contains("layernorm") || key.contains("_norm.") {
        return false;
    }
    if key.contains("embed_tokens") {
        return false;
    }
    if key == "encoder.norm.weight" {
        return false;
    }
    // q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj
    key.contains(".self_attn.")
        || key.contains(".mlp.gate_proj")
        || key.contains(".mlp.up_proj")
        || key.contains(".mlp.down_proj")
}

/// Enumerate all required weight keys for a T5Gemma2 text encoder with the
/// given layer count. Useful for validation / filtered loads.
pub fn expected_weight_keys(num_layers: usize) -> Vec<String> {
    let mut keys = vec!["encoder.embed_tokens.weight".to_string()];
    for i in 0..num_layers {
        let p = format!("encoder.layers.{i}");
        keys.extend([
            format!("{p}.pre_self_attn_layernorm.weight"),
            format!("{p}.post_self_attn_layernorm.weight"),
            format!("{p}.pre_feedforward_layernorm.weight"),
            format!("{p}.post_feedforward_layernorm.weight"),
            format!("{p}.self_attn.q_proj.weight"),
            format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.v_proj.weight"),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.self_attn.q_norm.weight"),
            format!("{p}.self_attn.k_norm.weight"),
            format!("{p}.mlp.gate_proj.weight"),
            format!("{p}.mlp.up_proj.weight"),
            format!("{p}.mlp.down_proj.weight"),
        ]);
    }
    keys.push("encoder.norm.weight".to_string());
    keys
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = T5Gemma2Config::default();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_layers, 34);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 10240);
        assert_eq!(cfg.layer_types.len(), 34);
    }

    #[test]
    fn test_layer_type_pattern() {
        let cfg = T5Gemma2Config::default();
        // Expected full-attention layers: 5, 11, 17, 23, 29.
        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, lt)| (*lt == LayerType::Full).then_some(i))
            .collect();
        assert_eq!(full_indices, vec![5, 11, 17, 23, 29]);

        // 29 sliding + 5 full = 34.
        let sliding = cfg
            .layer_types
            .iter()
            .filter(|lt| **lt == LayerType::Sliding)
            .count();
        assert_eq!(sliding, 29);
    }

    #[test]
    fn test_expected_weight_keys_count() {
        // 1 (embed) + 34 * 13 (per-layer) + 1 (final norm) = 444
        let keys = expected_weight_keys(34);
        assert_eq!(keys.len(), 1 + 34 * 13 + 1);
    }

    #[test]
    fn test_sdpa_scale_equivalence() {
        // Critical invariant: sdpa applies 1/sqrt(head_dim), and T5Gemma2
        // intends 1/sqrt(query_pre_attn_scalar). These must be equal.
        let cfg = T5Gemma2Config::default();
        assert_eq!(cfg.head_dim as f32, cfg.query_pre_attn_scalar);
    }

    #[test]
    fn test_is_projection_key() {
        assert!(is_projection_key(
            "encoder.layers.5.self_attn.q_proj.weight"
        ));
        assert!(is_projection_key(
            "encoder.layers.0.mlp.gate_proj.weight"
        ));
        assert!(is_projection_key(
            "encoder.layers.33.mlp.down_proj.weight"
        ));
        assert!(!is_projection_key("encoder.embed_tokens.weight"));
        assert!(!is_projection_key("encoder.norm.weight"));
        assert!(!is_projection_key(
            "encoder.layers.0.pre_self_attn_layernorm.weight"
        ));
        assert!(!is_projection_key(
            "encoder.layers.0.self_attn.q_norm.weight"
        ));
    }

    #[test]
    fn test_sliding_window_params() {
        // For sliding_window = 1024, left window = 512 and right = 513.
        // Total reachable = 512 + 513 = 1025 (q itself is in the left window:
        // dist = 0, q - k = 0 < 512 ✓).
        let sw = 1024usize;
        let left = (sw + 1) / 2;
        let right = sw / 2 + 1;
        assert_eq!(left, 512);
        assert_eq!(right, 513);
        assert_eq!(left + right, 1025);
    }
}
