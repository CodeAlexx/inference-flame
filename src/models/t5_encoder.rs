//! T5-XXL encoder for FLUX 1 — pure flame-core implementation.
//!
//! Produces [B, 512, 4096] hidden states from text input.
//!
//! Architecture (T5 v1.1 XXL encoder-only):
//!   - 24 layers, d_model=4096, 64 heads, d_kv=64
//!   - Gated-GELU FFN: gelu(wi_0(x)) * wi_1(x) → wo
//!   - T5 LayerNorm (RMSNorm variant: no bias, no mean subtraction)
//!   - Relative position bias (computed from layer 0, shared across all layers)
//!   - NO position embeddings (positions encoded via relative bias)
//!   - NO biases on attention or FFN projections
//!
//! Weight key format:
//!   encoder.embed_tokens.weight                              [32128, 4096]
//!   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight [4096, 4096]
//!   encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight [32, 64]
//!   encoder.block.{i}.layer.0.layer_norm.weight              [4096]
//!   encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight     [10240, 4096]  (gate)
//!   encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight     [10240, 4096]  (up)
//!   encoder.block.{i}.layer.1.DenseReluDense.wo.weight       [4096, 10240]  (down)
//!   encoder.block.{i}.layer.1.layer_norm.weight              [4096]
//!   encoder.final_layer_norm.weight                          [4096]
//!
//! ⚠️ STANDALONE — does NOT connect to any inference pipeline.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_swap::FlameSwap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub d_kv: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_eps: f32,
    pub max_seq_len: usize,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            num_layers: 24,
            d_ff: 10240,
            num_heads: 64,
            d_kv: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            max_seq_len: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// T5 Encoder with FlameSwap
// ---------------------------------------------------------------------------

/// T5-XXL encoder. Uses FlameSwap for 24 transformer layers (~18GB BF16).
pub struct T5Encoder {
    shared: HashMap<String, Tensor>,
    swap: FlameSwap,
    config: T5Config,
    device: Arc<CudaDevice>,
}

impl T5Encoder {
    /// Load T5-XXL from safetensors.
    pub fn load(
        safetensors_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = T5Config::default();

        // FlameSwap: "encoder.block.N.xxx" → block N
        let swap = FlameSwap::load(
            &[safetensors_path],
            device,
            |name| {
                let rest = name.strip_prefix("encoder.block.")?;
                rest.split('.').next()?.parse().ok()
            },
        ).map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap T5: {e}")))?;

        // Load shared weights
        let shared_keys = ["encoder.embed_tokens.weight", "encoder.final_layer_norm.weight", "shared.weight"];
        let mut shared = HashMap::new();
        let file_weights = load_file_filtered(
            Path::new(safetensors_path),
            device,
            |key| shared_keys.contains(&key),
        )?;
        for &sk in &shared_keys {
            if let Some(t) = file_weights.get(sk) {
                shared.insert(sk.to_string(), t.clone());
            }
        }

        // Use shared.weight as embed_tokens if embed_tokens not found
        if !shared.contains_key("encoder.embed_tokens.weight") {
            if let Some(sw) = shared.get("shared.weight") {
                shared.insert("encoder.embed_tokens.weight".to_string(), sw.clone());
            }
        }

        log::info!("[T5] Loaded: {} blocks via FlameSwap, {} shared weights",
            swap.num_blocks(), shared.len());

        Ok(Self { shared, swap, config, device: device.clone() })
    }

    /// T5 LayerNorm (RMSNorm variant): norm(x) * weight
    /// T5 uses standard RMSNorm without the Gemma (1+weight) formulation.
    fn t5_layer_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out.reshape(&dims)
    }

    /// Linear without bias: [B, N, C] x [out, in]^T → [B, N, out]
    fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        let shape = x.shape().dims().to_vec();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x_2d = x.reshape(&[b * n, c])?;
        let out = x_2d.matmul(&wt)?;
        let out_dim = out.shape().dims()[1];
        out.reshape(&[b, n, out_dim])
    }

    /// Compute T5 relative position bias.
    ///
    /// ## Reference:
    /// ```python
    /// # Bucket positions using logarithmic binning
    /// # relative_attention_bias.weight: [num_buckets, num_heads] = [32, 64]
    /// # Output: [1, num_heads, seq_len, seq_len]
    /// ```
    fn compute_relative_bias(
        bias_weight: &Tensor, // [32, 64]
        seq_len: usize,
        num_buckets: usize,
        max_distance: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let num_heads = bias_weight.shape().dims()[1];

        // Compute relative positions: context_position - memory_position
        // For encoder (bidirectional), use half buckets for positive, half for negative
        let half_buckets = num_buckets / 2;
        let mut bias_data = vec![0i32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let relative_position = j as i64 - i as i64;
                let bucket = t5_relative_position_bucket(
                    relative_position,
                    true, // bidirectional
                    half_buckets,
                    max_distance,
                );
                bias_data[i * seq_len + j] = bucket as i32;
            }
        }

        // Gather from bias_weight: [seq*seq] indices into [32, 64]
        let indices = Tensor::from_vec(
            bias_data.iter().map(|&i| i as f32).collect(),
            Shape::from_dims(&[seq_len * seq_len]),
            device.clone(),
        )?.to_dtype(DType::I32)?;

        // bias_weight: [32, 64] → gather rows → [seq*seq, 64]
        let gathered = bias_weight.index_select0(&indices)?; // [seq*seq, num_heads]

        // Reshape to [seq, seq, heads] → permute to [1, heads, seq, seq]
        let reshaped = gathered.reshape(&[seq_len, seq_len, num_heads])?;
        let permuted = reshaped.permute(&[2, 0, 1])?; // [heads, seq, seq]
        permuted.unsqueeze(0)?.to_dtype(DType::BF16) // [1, heads, seq, seq]
    }

    /// Single T5 encoder layer.
    ///
    /// ## PyTorch reference:
    /// ```python
    /// # Self-attention
    /// normed = layer_norm(hidden)
    /// q, k, v = attn_q(normed), attn_k(normed), attn_v(normed)
    /// attn_out = self_attention(q, k, v, position_bias)
    /// hidden = hidden + attn_out
    /// # FFN (gated-gelu)
    /// normed = layer_norm(hidden)
    /// gate = gelu(wi_0(normed))
    /// up = wi_1(normed)
    /// hidden = hidden + wo(gate * up)
    /// ```
    fn layer_forward(
        &self,
        hidden: &Tensor,
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        position_bias: &Tensor, // [1, heads, seq, seq]
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.d_kv;
        let prefix = format!("encoder.block.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing T5 weight: {key}"))
            })
        };

        // --- Self-attention ---
        let normed = Self::t5_layer_norm(hidden, w("layer.0.layer_norm.weight")?, cfg.layer_norm_eps)?;

        let q = Self::linear(&normed, w("layer.0.SelfAttention.q.weight")?)?;
        let k = Self::linear(&normed, w("layer.0.SelfAttention.k.weight")?)?;
        let v = Self::linear(&normed, w("layer.0.SelfAttention.v.weight")?)?;

        // Reshape: [B, N, heads*d_kv] → [B, heads, N, d_kv]
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;

        // Attention with position bias added to scores
        // flame_sdpa adds mask to scores; position_bias acts as the "mask" here
        let attn_out = flame_sdpa(&q, &k, &v, Some(position_bias))?;

        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        let attn_out = Self::linear(&attn_out, w("layer.0.SelfAttention.o.weight")?)?;
        let hidden = hidden.add(&attn_out)?;

        // --- Gated-GELU FFN ---
        let normed2 = Self::t5_layer_norm(&hidden, w("layer.1.layer_norm.weight")?, cfg.layer_norm_eps)?;

        let gate = Self::linear(&normed2, w("layer.1.DenseReluDense.wi_0.weight")?)?;
        let up = Self::linear(&normed2, w("layer.1.DenseReluDense.wi_1.weight")?)?;
        let ffn_out = gate.gelu()?.mul(&up)?;
        let ffn_out = Self::linear(&ffn_out, w("layer.1.DenseReluDense.wo.weight")?)?;

        hidden.add(&ffn_out)
    }

    /// Encode text → [1, seq_len, 4096] hidden states.
    pub fn encode(&mut self, token_ids: &[i32]) -> Result<Tensor> {
        let cfg = self.config.clone();
        let seq_len = token_ids.len();

        log::info!("[T5] Encoding: seq_len={}", seq_len);

        // 1. Token embeddings (no position embeddings — T5 uses relative bias)
        let embed_w = self.shared.get("encoder.embed_tokens.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing embed_tokens".into()))?;
        let ids = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let mut hidden = embed_w.index_select0(&ids)?.unsqueeze(0)?; // [1, seq, 4096]

        // 2. Compute relative position bias from layer 0
        // Prefetch block 0 to get the bias weight
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
        let raw_block0 = self.swap.await_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
        let block0: HashMap<String, Tensor> = raw_block0.into_iter()
            .map(|(k, v)| (k, v))
            .collect();

        let bias_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight";
        let bias_weight = block0.get(bias_key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing {bias_key}")))?;

        let position_bias = Self::compute_relative_bias(
            bias_weight,
            seq_len,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
            &self.device,
        )?;

        // 3. Forward through layer 0 (already loaded)
        hidden = self.layer_forward(&hidden, &block0, 0, &position_bias)?;
        log::info!("[T5] Layer 1/{} done", cfg.num_layers);

        // 4. Forward through remaining layers
        for i in 1..cfg.num_layers {
            self.swap.prefetch(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;

            // Prefetch next
            if i + 1 < cfg.num_layers {
                // Can't prefetch next until await_block is done — FlameSwap constraint
            }

            let weights: HashMap<String, Tensor> = raw.into_iter().collect();
            hidden = self.layer_forward(&hidden, &weights, i, &position_bias)?;

            if (i + 1) % 6 == 0 || i == cfg.num_layers - 1 {
                log::info!("[T5] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }

        // 5. Final layer norm
        let final_norm_w = self.shared.get("encoder.final_layer_norm.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer_norm".into()))?;
        hidden = Self::t5_layer_norm(&hidden, final_norm_w, cfg.layer_norm_eps)?;

        log::info!("[T5] Output: {:?}", hidden.shape());
        Ok(hidden)
    }

    pub fn config(&self) -> &T5Config {
        &self.config
    }
}

/// T5 relative position bucket computation.
/// Matches `transformers.models.t5.modeling_t5._relative_position_bucket`.
fn t5_relative_position_bucket(
    relative_position: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut ret = 0usize;
    let mut n = -(relative_position);

    if bidirectional {
        let num_buckets = num_buckets;
        if relative_position > 0 {
            ret += num_buckets;
            n = -n;
        }
    } else if n < 0 {
        n = 0;
    }

    let n = n as usize;
    let half_buckets = num_buckets / 2;
    let is_small = n < half_buckets;

    if is_small {
        ret += n;
    } else {
        let val = ((n as f64 / half_buckets as f64).ln()
            / (max_distance as f64 / half_buckets as f64).ln()
            * (half_buckets - 1) as f64) as usize;
        ret += half_buckets + val.min(half_buckets - 1);
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_config() {
        let cfg = T5Config::default();
        assert_eq!(cfg.d_model, 4096);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads * cfg.d_kv, 4096);
        assert_eq!(cfg.d_ff, 10240);
    }

    #[test]
    fn test_relative_position_bucket() {
        // Same position → bucket 0
        assert_eq!(t5_relative_position_bucket(0, true, 16, 128), 0);
        // Small positive
        assert_eq!(t5_relative_position_bucket(1, true, 16, 128), 16 + 1);
        // Small negative
        assert_eq!(t5_relative_position_bucket(-1, true, 16, 128), 1);
    }
}
