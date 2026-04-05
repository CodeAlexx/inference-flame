//! Gemma-3 12B text encoder for LTX-2 — pure flame-core implementation.
//!
//! Loads the Gemma-3 12B language model and runs forward-only inference,
//! returning ALL hidden states (49 total: embedding + 48 layers) for use
//! by LTX-2's FeatureExtractorV2 + Connector pipeline.
//!
//! ## Architecture (from HuggingFace Gemma3ForConditionalGeneration)
//!
//! - 48 transformer layers, hidden_size=3840
//! - head_dim=256, 16 Q heads, 8 KV heads (GQA 2:1)
//! - Sliding window pattern: every 6th layer is full attention (rest: window=1024)
//! - Embedding: lookup * sqrt(hidden_size)
//! - RMSNorm: output * (1.0 + weight) — Gemma-specific formulation
//! - 4 norms per layer: input_layernorm, post_attention_layernorm,
//!   pre_feedforward_layernorm, post_feedforward_layernorm
//! - Dual RoPE: global (theta=1M, linear scale 8x) and local (theta=10K)
//! - QK norm (per-head RMSNorm) before RoPE
//! - MLP: gelu_tanh(gate_proj(x)) * up_proj(x) → down_proj
//! - Attention scaling: 1/sqrt(query_pre_attn_scalar) = 1/sqrt(256) = 1/16
//!
//! ## Weight key format (from Gemma3ForConditionalGeneration safetensors):
//!   language_model.model.embed_tokens.weight
//!   language_model.model.layers.{i}.input_layernorm.weight
//!   language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   language_model.model.layers.{i}.self_attn.{q,k}_norm.weight
//!   language_model.model.layers.{i}.post_attention_layernorm.weight
//!   language_model.model.layers.{i}.pre_feedforward_layernorm.weight
//!   language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
//!   language_model.model.layers.{i}.post_feedforward_layernorm.weight
//!   language_model.model.norm.weight
//!
//! ⚠️ This module is STANDALONE — it does NOT connect to any inference pipeline.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_swap::FlameSwap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Gemma-3 12B config, matching GEMMA3_CONFIG_FOR_LTX in the official code.
#[derive(Debug, Clone)]
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    /// Global RoPE theta (for full-attention layers).
    pub rope_theta_global: f64,
    /// Local RoPE theta (for sliding-attention layers).
    pub rope_theta_local: f64,
    /// Linear RoPE scaling factor for global rope.
    pub rope_scaling_factor: f64,
    /// Sliding window size for local attention layers.
    pub sliding_window: usize,
    /// Every Nth layer is full attention (pattern=6 means layers 5,11,17,...).
    pub sliding_window_pattern: usize,
    /// Attention scaling: 1/sqrt(query_pre_attn_scalar).
    pub query_pre_attn_scalar: f64,
}

impl Default for Gemma3Config {
    fn default() -> Self {
        Self {
            vocab_size: 262208,
            hidden_size: 3840,
            num_layers: 48,
            intermediate_size: 15360,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta_global: 1_000_000.0,
            rope_theta_local: 10_000.0,
            rope_scaling_factor: 8.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 256.0,
        }
    }
}

impl Gemma3Config {
    /// Returns true if layer `i` uses full attention (global RoPE).
    /// Pattern: layer (i+1) % pattern == 0 → full attention.
    /// Layers 5,11,17,23,29,35,41,47 for pattern=6.
    fn is_full_attention(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.sliding_window_pattern == 0
    }
}

// ---------------------------------------------------------------------------
// Gemma3 Encoder (with FlameSwap for block-level offloading)
// ---------------------------------------------------------------------------

/// Gemma-3 12B text encoder with FlameSwap for layer-by-layer GPU offloading.
///
/// Shared weights (embeddings, final norm) are kept on GPU.
/// Transformer layers are swapped in/out via FlameSwap.
pub struct Gemma3Encoder {
    /// Shared weights on GPU: embed_tokens, final norm.
    shared: HashMap<String, Tensor>,
    /// FlameSwap for the 48 transformer layers.
    swap: FlameSwap,
    config: Gemma3Config,
    device: Arc<CudaDevice>,
    /// Pre-computed global RoPE (cos, sin) for max sequence length.
    /// Shape: [1, seq_len, head_dim] each.
    rope_global_cos: Tensor,
    rope_global_sin: Tensor,
    /// Pre-computed local RoPE.
    rope_local_cos: Tensor,
    rope_local_sin: Tensor,
}

impl Gemma3Encoder {
    /// Load Gemma-3 from safetensors files using FlameSwap.
    ///
    /// `safetensors_paths`: paths to the sharded safetensors files.
    /// `max_seq_len`: maximum sequence length for RoPE precomputation.
    pub fn load(
        safetensors_paths: &[&str],
        device: &Arc<CudaDevice>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let config = Gemma3Config::default();

        // Strip "language_model.model." prefix from weight keys for cleaner access.
        // FlameSwap block detection: "layers.N.xxx" → block N.
        let prefix = "language_model.model.";

        let swap = FlameSwap::load(
            safetensors_paths,
            device,
            |name| {
                // Only process keys under our prefix
                let stripped = name.strip_prefix(prefix)?;
                let rest = stripped.strip_prefix("layers.")?;
                rest.split('.').next()?.parse().ok()
            },
        ).map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap load: {e}")))?;

        // Load shared (non-block) weights onto GPU.
        // These are embed_tokens and final norm — NOT in FlameSwap blocks.
        // Scan each safetensors file for keys matching our shared weight patterns.
        let shared_keys_needed = [
            ("embed_tokens.weight", format!("{prefix}embed_tokens.weight")),
            ("norm.weight", format!("{prefix}norm.weight")),
        ];

        let mut shared = HashMap::new();
        for path in safetensors_paths {
            let file_weights = load_file_filtered(
                Path::new(path),
                device,
                |key| shared_keys_needed.iter().any(|(_, full)| key == full.as_str()),
            )?;
            for (short_key, full_key) in &shared_keys_needed {
                if let Some(t) = file_weights.get(full_key.as_str()) {
                    shared.insert(short_key.to_string(), t.clone());
                    log::info!("[Gemma3] Loaded shared weight '{}': {:?}", short_key, t.shape());
                }
            }
        }

        // Verify all shared weights were found
        for (short_key, full_key) in &shared_keys_needed {
            if !shared.contains_key(*short_key) {
                return Err(flame_core::Error::InvalidInput(
                    format!("Missing shared weight: {full_key}"),
                ));
            }
        }

        // Pre-compute RoPE tables
        // Global RoPE: theta=1M with linear scaling factor 8
        // Effective inv_freq is divided by scaling_factor
        let (rope_global_cos, rope_global_sin) = build_rope_table(
            max_seq_len,
            config.head_dim,
            config.rope_theta_global,
            config.rope_scaling_factor, // attention_scaling = 1/factor for linear
            device,
        )?;

        // Local RoPE: theta=10K, no scaling
        let (rope_local_cos, rope_local_sin) = build_rope_table(
            max_seq_len,
            config.head_dim,
            config.rope_theta_local,
            1.0, // no scaling for local
            device,
        )?;

        log::info!("[Gemma3] Loaded: {} blocks via FlameSwap, {} shared weights",
            swap.num_blocks(), shared.len());
        log::info!("[Gemma3] RoPE: global cos {:?}, local cos {:?}",
            rope_global_cos.shape(), rope_local_cos.shape());

        Ok(Self {
            shared,
            swap,
            config,
            device: device.clone(),
            rope_global_cos,
            rope_global_sin,
            rope_local_cos,
            rope_local_sin,
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Gemma3 RMSNorm: output = rms_norm(x) * (1 + weight)
    ///
    /// Different from standard RMSNorm which uses just `weight`.
    /// See: https://github.com/huggingface/transformers/pull/29402
    fn gemma3_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        // Standard RMSNorm: x * rsqrt(mean(x^2) + eps)
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let x_2d = x.reshape(&[batch, hidden])?;
        // flame_core rms_norm applies: norm(x) * weight
        // But Gemma3 needs: norm(x) * (1 + weight)
        // So we do: norm(x) without weight, then multiply by (1 + weight)
        let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, None, eps)?;

        // (1 + weight): weight is [hidden], broadcast over batch dim
        let one_plus_w = weight.add_scalar(1.0)?;
        let result = normed.mul(&one_plus_w.unsqueeze(0)?.expand(&[batch, hidden])?)?;
        result.reshape(&dims)
    }

    /// Linear projection: [B, N, C] x [in, out]^T -> [B, N, out].
    /// Weight shape from checkpoint: [out_features, in_features] — needs transpose.
    fn linear_3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];

        // weight: [out, in] → transpose to [in, out]
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;

        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(&wt)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    /// Linear with pre-transposed weight: [B, N, C] x [in, out] -> [B, N, out].
    fn linear_3d_t(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];

        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(weight_t)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    /// Apply half-split RoPE to a tensor [B, H, N, D].
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
    }

    /// Repeat KV heads to match Q head count for GQA.
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
    // Build masks
    // -----------------------------------------------------------------------

    /// Build causal mask [1, 1, seq, seq].
    /// For full attention: standard causal mask.
    /// For sliding attention: causal + sliding window.
    fn build_causal_mask(
        seq_len: usize,
        real_len: usize,
        sliding_window: Option<usize>,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        let min_val = f32::MIN; // Large negative for masked positions

        for i in 0..seq_len {
            for j in 0..seq_len {
                let is_causal = j <= i;
                let is_real = j < real_len;
                let in_window = sliding_window
                    .map(|w| i.saturating_sub(w) <= j)
                    .unwrap_or(true);

                if is_causal && is_real && in_window {
                    data[i * seq_len + j] = 0.0; // attend
                } else {
                    data[i * seq_len + j] = min_val; // mask out
                }
            }
        }

        let mask = Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?;
        mask.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Single transformer layer forward
    // -----------------------------------------------------------------------

    /// Execute one Gemma3 decoder layer using weights from FlameSwap.
    ///
    /// ## PyTorch reference (Gemma3DecoderLayer.forward):
    /// ```python
    /// residual = hidden_states
    /// hidden_states = self.input_layernorm(hidden_states)
    /// # select rope based on layer type
    /// position_embeddings = position_embeddings_local if is_sliding else position_embeddings_global
    /// hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask, ...)
    /// hidden_states = self.post_attention_layernorm(hidden_states)
    /// hidden_states = residual + hidden_states
    /// residual = hidden_states
    /// hidden_states = self.pre_feedforward_layernorm(hidden_states)
    /// hidden_states = self.mlp(hidden_states)
    /// hidden_states = self.post_feedforward_layernorm(hidden_states)
    /// hidden_states = residual + hidden_states
    /// ```
    fn layer_forward(
        &self,
        hidden: &Tensor,
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        full_mask: &Tensor,
        sliding_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let is_full = cfg.is_full_attention(layer_idx);
        let prefix = format!("layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        // --- Self-attention ---

        // 1. input_layernorm
        let norm_w = weights.get(&format!("{prefix}.input_layernorm.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.input_layernorm.weight")))?;
        let normed = Self::gemma3_rms_norm(hidden, norm_w, cfg.rms_norm_eps)?;

        // 2. QKV projections
        let q_w = weights.get(&format!("{prefix}.self_attn.q_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.self_attn.q_proj.weight")))?;
        let k_w = weights.get(&format!("{prefix}.self_attn.k_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.self_attn.k_proj.weight")))?;
        let v_w = weights.get(&format!("{prefix}.self_attn.v_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.self_attn.v_proj.weight")))?;

        let q = Self::linear_3d(&normed, q_w)?;
        let k = Self::linear_3d(&normed, k_w)?;
        let v = Self::linear_3d(&normed, v_w)?;

        // Reshape to [B, heads, N, head_dim]
        let mut q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let mut k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // 3. QK norm (per-head RMSNorm with Gemma3 formulation)
        let q_norm_w = weights.get(&format!("{prefix}.self_attn.q_norm.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.self_attn.q_norm.weight")))?;
        let k_norm_w = weights.get(&format!("{prefix}.self_attn.k_norm.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.self_attn.k_norm.weight")))?;

        // q: [B, H, N, D] → flatten heads for per-head norm → reshape back
        let q_flat = q.reshape(&[b * h * n, d])?;
        let q_normed = Self::gemma3_rms_norm_2d(&q_flat, q_norm_w, cfg.rms_norm_eps)?;
        q = q_normed.reshape(&[b, h, n, d])?;

        let k_flat = k.reshape(&[b * h_kv * n, d])?;
        let k_normed = Self::gemma3_rms_norm_2d(&k_flat, k_norm_w, cfg.rms_norm_eps)?;
        k = k_normed.reshape(&[b, h_kv, n, d])?;

        // 4. Apply RoPE (select based on layer type)
        let (cos, sin) = if is_full {
            (&self.rope_global_cos, &self.rope_global_sin)
        } else {
            (&self.rope_local_cos, &self.rope_local_sin)
        };
        // Narrow to actual seq_len
        let cos = cos.narrow(1, 0, n)?;
        let sin = sin.narrow(1, 0, n)?;
        let q = Self::apply_rope(&q, &cos, &sin)?;
        let k = Self::apply_rope(&k, &cos, &sin)?;

        // 5. GQA expand
        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        // 6. Scaled dot-product attention
        // Gemma3 scaling: 1/sqrt(query_pre_attn_scalar) = 1/sqrt(256) = 1/16
        // SDPA handles scaling internally via q * scale, but flame_sdpa might not.
        // We scale Q before passing to SDPA.
        let attn_scale = (cfg.query_pre_attn_scalar as f32).powf(-0.5);
        let q_scaled = q.mul_scalar(attn_scale)?;

        let mask = if is_full { full_mask } else { sliding_mask };
        let attn_out = flame_sdpa(&q_scaled, &k, &v, Some(mask))?;

        // [B, H, N, D] → [B, N, H*D]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        // 7. Output projection
        let o_w = weights.get(&format!("{prefix}.self_attn.o_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.self_attn.o_proj.weight")))?;
        let attn_out = Self::linear_3d(&attn_out, o_w)?;

        // 8. post_attention_layernorm + residual
        let post_attn_norm_w = weights.get(&format!("{prefix}.post_attention_layernorm.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.post_attention_layernorm.weight")))?;
        let attn_out = Self::gemma3_rms_norm(&attn_out, post_attn_norm_w, cfg.rms_norm_eps)?;
        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---

        // 9. pre_feedforward_layernorm
        let pre_ff_norm_w = weights.get(&format!("{prefix}.pre_feedforward_layernorm.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.pre_feedforward_layernorm.weight")))?;
        let normed2 = Self::gemma3_rms_norm(&hidden, pre_ff_norm_w, cfg.rms_norm_eps)?;

        // 10. MLP: gelu_tanh(gate) * up → down
        let gate_w = weights.get(&format!("{prefix}.mlp.gate_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.mlp.gate_proj.weight")))?;
        let up_w = weights.get(&format!("{prefix}.mlp.up_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.mlp.up_proj.weight")))?;
        let down_w = weights.get(&format!("{prefix}.mlp.down_proj.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.mlp.down_proj.weight")))?;

        let gate = Self::linear_3d(&normed2, gate_w)?;
        let up = Self::linear_3d(&normed2, up_w)?;
        let mlp_out = gate.gelu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, down_w)?;

        // 11. post_feedforward_layernorm + residual
        let post_ff_norm_w = weights.get(&format!("{prefix}.post_feedforward_layernorm.weight"))
            .ok_or_else(|| flame_core::Error::InvalidInput(
                format!("Missing {prefix}.post_feedforward_layernorm.weight")))?;
        let mlp_out = Self::gemma3_rms_norm(&mlp_out, post_ff_norm_w, cfg.rms_norm_eps)?;
        hidden.add(&mlp_out)
    }

    /// Gemma3 RMSNorm for 2D tensors (already [batch, hidden]).
    fn gemma3_rms_norm_2d(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = dims[1];
        let batch = dims[0];

        let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(x, None, eps)?;
        let one_plus_w = weight.add_scalar(1.0)?;
        normed.mul(&one_plus_w.unsqueeze(0)?.expand(&[batch, hidden])?)
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    /// Embed tokens and scale by sqrt(hidden_size).
    ///
    /// ## PyTorch reference:
    /// ```python
    /// class Gemma3TextScaledWordEmbedding(nn.Embedding):
    ///     def forward(self, input_ids):
    ///         return super().forward(input_ids) * self.embed_scale
    /// ```
    fn embed_tokens(&self, token_ids: &[i32]) -> Result<Tensor> {
        let embed_w = self.shared.get("embed_tokens.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput(
                "Missing embed_tokens.weight".into()))?;
        let seq_len = token_ids.len();

        let ids_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;

        let selected = embed_w.index_select0(&ids_tensor)?;

        // Scale by sqrt(hidden_size) = sqrt(3840) ≈ 61.97
        let scale = (self.config.hidden_size as f32).sqrt();
        let scaled = selected.mul_scalar(scale)?;
        scaled.unsqueeze(0) // [1, seq_len, hidden_size]
    }

    // -----------------------------------------------------------------------
    // Full forward pass
    // -----------------------------------------------------------------------

    /// Run Gemma-3 forward pass, returning ALL 49 hidden states.
    ///
    /// Returns: Vec of 49 tensors, each [1, seq_len, 3840].
    /// Index 0 = embedding output, 1..=48 = layer outputs.
    /// The last element (index 48) has the final norm applied.
    ///
    /// Also returns the attention mask as a [1, seq_len] tensor.
    pub fn encode(&mut self, token_ids: &[i32], attention_mask: &[i32]) -> Result<(Vec<Tensor>, Tensor)> {
        let cfg = self.config.clone();
        let seq_len = token_ids.len();

        // Detect real token count from attention mask
        let real_len = attention_mask.iter().filter(|&&m| m != 0).count();

        log::info!("[Gemma3] Encoding: seq_len={}, real_tokens={}", seq_len, real_len);

        // 1. Embed tokens (with scaling)
        let mut hidden = self.embed_tokens(token_ids)?;

        // 2. Build attention masks (full + sliding)
        let full_mask = Self::build_causal_mask(seq_len, real_len, None, &self.device)?;
        let sliding_mask = Self::build_causal_mask(
            seq_len, real_len, Some(cfg.sliding_window), &self.device,
        )?;

        // 3. Collect all hidden states
        let mut all_hidden_states = Vec::with_capacity(cfg.num_layers + 1);
        all_hidden_states.push(hidden.clone()); // embedding output (index 0)

        // 4. Forward through 48 layers with FlameSwap
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..cfg.num_layers {
            // Prefetch next block
            if i + 1 < cfg.num_layers {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            // Wait for current block's weights and strip key prefix
            let raw_weights = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await_block: {e}")))?;

            // FlameSwap returns keys with full safetensors names, e.g.
            // "language_model.model.layers.5.self_attn.q_proj.weight"
            // Strip prefix to get "layers.5.self_attn.q_proj.weight"
            let weights: HashMap<String, Tensor> = raw_weights.into_iter()
                .map(|(k, v)| {
                    let stripped = k.strip_prefix("language_model.model.")
                        .unwrap_or(&k)
                        .to_string();
                    (stripped, v)
                })
                .collect();

            hidden = self.layer_forward(&hidden, &weights, i, &full_mask, &sliding_mask)?;

            all_hidden_states.push(hidden.clone());

            if i % 12 == 0 || i == cfg.num_layers - 1 {
                log::info!("[Gemma3] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }

        // 5. Apply final norm to the last hidden state
        let final_norm_w = self.shared.get("norm.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput(
                "Missing norm.weight".into()))?;
        let final_hidden = Self::gemma3_rms_norm(
            all_hidden_states.last().unwrap(),
            final_norm_w,
            cfg.rms_norm_eps,
        )?;
        // Replace the last hidden state with the normed version
        let last_idx = all_hidden_states.len() - 1;
        all_hidden_states[last_idx] = final_hidden;

        // Return attention mask as [1, seq_len] tensor
        let mask_tensor = Tensor::from_vec(
            attention_mask.iter().map(|&m| m as f32).collect(),
            Shape::from_dims(&[1, seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        log::info!("[Gemma3] Encoding complete: {} hidden states, each {:?}",
            all_hidden_states.len(), all_hidden_states[0].shape());

        Ok((all_hidden_states, mask_tensor))
    }

    /// Get number of hidden states returned (49 for Gemma-3 12B).
    pub fn num_hidden_states(&self) -> usize {
        self.config.num_layers + 1
    }

    /// Get hidden size (3840 for Gemma-3 12B).
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get config reference.
    pub fn config(&self) -> &Gemma3Config {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// RoPE table construction
// ---------------------------------------------------------------------------

/// Build RoPE cos/sin tables [1, max_seq_len, head_dim].
///
/// ## PyTorch reference (Gemma3RotaryEmbedding.forward):
/// ```python
/// inv_freq_expanded = self.inv_freq[None, :, None].float()
/// position_ids_expanded = position_ids[:, None, :].float()
/// freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
/// emb = torch.cat((freqs, freqs), dim=-1)
/// cos = emb.cos() * self.attention_scaling
/// sin = emb.sin() * self.attention_scaling
/// ```
///
/// For "linear" rope_type: inv_freq = 1/(theta^(i/dim)) and attention_scaling = 1/factor.
fn build_rope_table(
    max_seq_len: usize,
    head_dim: usize,
    theta: f64,
    scaling_factor: f64,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let attention_scaling = 1.0 / scaling_factor;

    // inv_freq: [half] = 1 / theta^(2i/dim)
    let mut inv_freq = vec![0.0f64; half];
    for i in 0..half {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
    }

    // freqs: [max_seq_len, half] = pos * inv_freq
    let mut cos_data = vec![0.0f32; max_seq_len * head_dim];
    let mut sin_data = vec![0.0f32; max_seq_len * head_dim];

    for pos in 0..max_seq_len {
        for i in 0..half {
            let angle = (pos as f64) * inv_freq[i];
            let c = (angle.cos() * attention_scaling) as f32;
            let s = (angle.sin() * attention_scaling) as f32;
            // cat((freqs, freqs), dim=-1): duplicate across full head_dim
            cos_data[pos * head_dim + i] = c;
            cos_data[pos * head_dim + half + i] = c;
            sin_data[pos * head_dim + i] = s;
            sin_data[pos * head_dim + half + i] = s;
        }
    }

    let cos = Tensor::from_vec(
        cos_data,
        Shape::from_dims(&[1, max_seq_len, head_dim]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    let sin = Tensor::from_vec(
        sin_data,
        Shape::from_dims(&[1, max_seq_len, head_dim]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    Ok((cos, sin))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Gemma3Config::default();
        assert_eq!(cfg.vocab_size, 262208);
        assert_eq!(cfg.hidden_size, 3840);
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 15360);
    }

    #[test]
    fn test_sliding_window_pattern() {
        let cfg = Gemma3Config::default();
        // Pattern=6: layers 5, 11, 17, 23, 29, 35, 41, 47 are full attention
        assert!(!cfg.is_full_attention(0));
        assert!(!cfg.is_full_attention(4));
        assert!(cfg.is_full_attention(5));
        assert!(!cfg.is_full_attention(6));
        assert!(cfg.is_full_attention(11));
        assert!(cfg.is_full_attention(17));
        assert!(cfg.is_full_attention(47));
    }

    #[test]
    fn test_total_hidden_states() {
        let cfg = Gemma3Config::default();
        // embedding + 48 layers = 49 hidden states
        assert_eq!(cfg.num_layers + 1, 49);
    }

    #[test]
    fn test_flat_dim() {
        let cfg = Gemma3Config::default();
        // 49 hidden states * 3840 = 188160
        assert_eq!((cfg.num_layers + 1) * cfg.hidden_size, 188160);
    }
}
