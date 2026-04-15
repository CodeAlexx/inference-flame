//! Mistral Small 3.1 24B text encoder for FLUX 2 — pure flame-core with BlockOffloader.
//!
//! Loads Mistral-Small-3.1-24B from BF16 safetensors and extracts hidden states
//! at layers [10, 20, 30] (1-indexed), stacking them along the hidden dimension
//! to produce FLUX 2's text conditioning:
//!   output shape = [batch, seq_len, 3 * 5120] = [batch, 512, 15360]
//!
//! Architecture per layer:
//!   input -> RMSNorm -> QKV proj -> RoPE -> GQA attention -> residual
//!         -> RMSNorm -> gate+up -> SiLU(gate)*up -> down -> residual
//!
//! This is almost identical to Qwen3Encoder but:
//!   - No QK norm
//!   - hidden_size=5120, 40 layers, head_dim=128, 32 Q heads, 8 KV heads
//!   - rope_theta=1e9, rms_norm_eps=1e-5
//!   - Weight prefix: "language_model.model." (stripped on load)
//!   - 24B params → BlockOffloader required (won't fit in 24GB VRAM)
//!
//! ## Weight key format (from FLUX.2-dev text_encoder safetensors):
//!   language_model.model.embed_tokens.weight          [131072, 5120]
//!   language_model.model.layers.{i}.input_layernorm.weight         [5120]
//!   language_model.model.layers.{i}.self_attn.q_proj.weight        [4096, 5120]
//!   language_model.model.layers.{i}.self_attn.k_proj.weight        [1024, 5120]
//!   language_model.model.layers.{i}.self_attn.v_proj.weight        [1024, 5120]
//!   language_model.model.layers.{i}.self_attn.o_proj.weight        [5120, 4096]
//!   language_model.model.layers.{i}.post_attention_layernorm.weight [5120]
//!   language_model.model.layers.{i}.mlp.gate_proj.weight           [32768, 5120]
//!   language_model.model.layers.{i}.mlp.up_proj.weight             [32768, 5120]
//!   language_model.model.layers.{i}.mlp.down_proj.weight           [5120, 32768]
//!   language_model.model.norm.weight                   [5120]
//!
//! ⚠️ This module is STANDALONE — does NOT connect to any inference pipeline.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_diffusion::block_offload::BlockFacilitator;
use flame_diffusion::BlockOffloader;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BlockFacilitator for Mistral:
//   language_model.model.layers.{i}.* → block i
// ---------------------------------------------------------------------------

struct MistralFacilitator {
    num_layers: usize,
}

impl BlockFacilitator for MistralFacilitator {
    fn block_count(&self) -> usize { self.num_layers }
    fn classify_key(&self, key: &str) -> Option<usize> {
        let stripped = key.strip_prefix(KEY_PREFIX)?;
        let rest = stripped.strip_prefix("layers.")?;
        rest.split('.').next()?.parse().ok()
    }
}

/// FLUX 2 extraction layers (0-indexed layer indices).
/// Musubi reference: OUTPUT_LAYERS_MISTRAL = [10, 20, 30] (1-indexed hidden_states).
/// hidden_states[0] = embedding, hidden_states[k] = output of layer k-1.
/// So layers 10, 20, 30 (1-indexed) = layer indices 9, 19, 29 (0-indexed).
const FLUX2_EXTRACT_LAYERS: [usize; 3] = [9, 19, 29];

/// Max sequence length for FLUX 2 text conditioning.
const FLUX2_MAX_LENGTH: usize = 512;

/// Weight key prefix in the FLUX 2 text encoder safetensors.
const KEY_PREFIX: &str = "language_model.model.";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Mistral Small 3.1 24B config, matching the FLUX 2 text encoder checkpoint.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub extract_layers: Vec<usize>,
    pub max_seq_len: usize,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 5120,
            num_layers: 40,
            intermediate_size: 32768,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000_000.0,
            extract_layers: FLUX2_EXTRACT_LAYERS.to_vec(),
            max_seq_len: FLUX2_MAX_LENGTH,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Mistral Small 3.1 24B encoder with BlockOffloader for layer-by-layer GPU offloading.
///
/// Shared weights (embed_tokens, final norm) kept on GPU.
/// 40 transformer layers swapped in/out via BlockOffloader.
pub struct MistralEncoder {
    /// Shared weights on GPU: embed_tokens, final norm.
    shared: HashMap<String, Tensor>,
    /// BlockOffloader for the 40 transformer layers.
    offloader: BlockOffloader,
    config: MistralConfig,
    device: Arc<CudaDevice>,
}

impl MistralEncoder {
    /// Load Mistral from sharded BF16 safetensors using BlockOffloader.
    ///
    /// `safetensors_paths`: paths to the text_encoder shards.
    pub fn load(
        safetensors_paths: &[&str],
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = MistralConfig::default();

        // BlockOffloader: classify keys into blocks by layer index.
        // Keys like "language_model.model.layers.5.self_attn.q_proj.weight" → block 5.
        let facilitator = MistralFacilitator { num_layers: config.num_layers };
        let offloader = BlockOffloader::load(
            safetensors_paths,
            &facilitator,
            device.clone(),
        ).map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader load: {e}")))?;

        // Load shared (non-block) weights: embed_tokens and final norm.
        let shared_targets = [
            ("embed_tokens.weight", format!("{KEY_PREFIX}embed_tokens.weight")),
            ("norm.weight", format!("{KEY_PREFIX}norm.weight")),
        ];

        let mut shared = HashMap::new();
        for path in safetensors_paths {
            let file_weights = load_file_filtered(
                Path::new(path),
                device,
                |key| shared_targets.iter().any(|(_, full)| key == full.as_str()),
            )?;
            for (short_key, full_key) in &shared_targets {
                if let Some(t) = file_weights.get(full_key.as_str()) {
                    shared.insert(short_key.to_string(), t.clone());
                    log::info!("[Mistral] Loaded shared '{}': {:?}", short_key, t.shape());
                }
            }
        }

        for (short_key, full_key) in &shared_targets {
            if !shared.contains_key(*short_key) {
                return Err(flame_core::Error::InvalidInput(
                    format!("Missing shared weight: {full_key}"),
                ));
            }
        }

        log::info!("[Mistral] Loaded: {} blocks via BlockOffloader, {} shared weights",
            offloader.block_count(), shared.len());

        Ok(Self { shared, offloader, config, device: device.clone() })
    }

    // -----------------------------------------------------------------------
    // Helpers (same as Qwen3Encoder — standard LLM building blocks)
    // -----------------------------------------------------------------------

    /// Matmul for [B, N, C] x [C, out] -> [B, N, out].
    /// Weight is [out, in] from checkpoint — transpose inline.
    fn linear_3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        let shape = x.shape().dims().to_vec();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(&wt)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    /// Standard RMSNorm: norm(x) * weight.
    fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out_2d.reshape(&dims)
    }

    /// Build 1D RoPE cos/sin tables [1, 1, seq_len, head_dim/2].
    ///
    /// ## PyTorch reference (standard HF RoPE):
    /// ```python
    /// inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    /// t = torch.arange(seq_len)
    /// freqs = torch.outer(t, inv_freq)
    /// cos = freqs.cos()
    /// sin = freqs.sin()
    /// ```
    fn build_rope_1d(
        seq_len: usize,
        head_dim: usize,
        theta: f64,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let half = head_dim / 2;

        let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;
        let freq_idx = Tensor::arange(0.0, head_dim as f32, 2.0, device.clone())?;
        let log_theta = (theta as f32).ln();
        let scale = -log_theta / head_dim as f32;
        let log_freqs = freq_idx.mul_scalar(scale)?.exp()?;

        let pos_col = pos.reshape(&[seq_len, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.matmul(&freq_row)?;

        let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    /// Apply half-split RoPE: fused CUDA kernel.
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
    }

    /// Repeat KV heads to match Q head count for GQA.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let (b, h_kv, n, d) = (dims[0], dims[1], dims[2], dims[3]);
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    /// Build causal + padding mask [1, 1, seq, seq].
    fn build_causal_mask(
        seq_len: usize,
        real_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i && j < real_len {
                    data[i * seq_len + j] = 1.0;
                }
            }
        }
        Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Single transformer layer
    // -----------------------------------------------------------------------

    /// Execute one Mistral decoder layer using weights from BlockOffloader.
    ///
    /// ## PyTorch reference (standard Mistral decoder layer):
    /// ```python
    /// residual = hidden_states
    /// hidden_states = self.input_layernorm(hidden_states)
    /// hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
    /// hidden_states = residual + hidden_states
    /// residual = hidden_states
    /// hidden_states = self.post_attention_layernorm(hidden_states)
    /// hidden_states = self.mlp(hidden_states)
    /// hidden_states = residual + hidden_states
    /// ```
    fn layer_forward(
        &self,
        hidden: &Tensor,
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let prefix = format!("layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        // Helper to get a weight, with clear error
        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing weight: {key}"))
            })
        };

        // --- Self-attention ---

        let normed = Self::rms_norm(hidden, w("input_layernorm.weight")?, cfg.rms_norm_eps)?;

        let q = Self::linear_3d(&normed, w("self_attn.q_proj.weight")?)?;
        let k = Self::linear_3d(&normed, w("self_attn.k_proj.weight")?)?;
        let v = Self::linear_3d(&normed, w("self_attn.v_proj.weight")?)?;

        // [B, N, heads*dim] → [B, heads, N, dim]
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // No QK norm in Mistral (unlike Qwen3)

        // RoPE
        let q = Self::apply_rope(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope(&k, pe_cos, pe_sin)?;

        // GQA expand
        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        // SDPA
        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?;

        // [B, H, N, D] → [B, N, H*D]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        // Output projection
        let attn_out = Self::linear_3d(&attn_out, w("self_attn.o_proj.weight")?)?;
        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---

        let normed2 = Self::rms_norm(&hidden, w("post_attention_layernorm.weight")?, cfg.rms_norm_eps)?;

        let gate = Self::linear_3d(&normed2, w("mlp.gate_proj.weight")?)?;
        let up = Self::linear_3d(&normed2, w("mlp.up_proj.weight")?)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, w("mlp.down_proj.weight")?)?;

        hidden.add(&mlp_out)
    }

    // -----------------------------------------------------------------------
    // Full forward pass
    // -----------------------------------------------------------------------

    /// Run Mistral forward pass and return stacked hidden states for FLUX 2.
    ///
    /// ## Musubi reference (Mistral3Embedder.forward):
    /// ```python
    /// output = self.mistral3(input_ids=input_ids, attention_mask=attention_mask,
    ///                        output_hidden_states=True, use_cache=False)
    /// out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
    /// return rearrange(out, "b c l d -> b l (c d)")
    /// ```
    ///
    /// OUTPUT_LAYERS_MISTRAL = [10, 20, 30] (1-indexed hidden_states).
    /// hidden_states[0] = embedding, [k] = output of layer k-1.
    /// So we extract after layers 9, 19, 29 (0-indexed).
    ///
    /// Returns: [1, seq_len, 3 * 5120] = [1, 512, 15360] BF16.
    pub fn encode(
        &mut self,
        token_ids: &[i32],
        attention_mask: &[i32],
    ) -> Result<Tensor> {
        let cfg = self.config.clone();
        let seq_len = token_ids.len();

        let real_len = attention_mask.iter().filter(|&&m| m != 0).count();
        log::info!("[Mistral] Encoding: seq_len={}, real_tokens={}", seq_len, real_len);

        // 1. Embed tokens (no scaling — unlike Gemma)
        let embed_w = self.shared.get("embed_tokens.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing embed_tokens.weight".into()))?;
        let ids_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let mut hidden = embed_w.index_select0(&ids_tensor)?.unsqueeze(0)?;

        // 2. Build RoPE and causal mask
        let (pe_cos, pe_sin) = Self::build_rope_1d(
            seq_len, cfg.head_dim, cfg.rope_theta, &self.device,
        )?;
        let attn_mask = Self::build_causal_mask(seq_len, real_len, &self.device)?;

        // 3. Forward through 40 layers with BlockOffloader, collecting extract layers
        let mut collected: Vec<(usize, Tensor)> = Vec::new();

        self.offloader.prefetch_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..cfg.num_layers {
            if i + 1 < cfg.num_layers {
                self.offloader.prefetch_block(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            let raw_weights = self.offloader.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await_block: {e}")))?;

            // Strip "language_model.model." prefix from BlockOffloader keys
            let weights: HashMap<String, Tensor> = raw_weights.iter()
                .map(|(k, v)| {
                    let stripped = k.strip_prefix(KEY_PREFIX).unwrap_or(k).to_string();
                    (stripped, v.clone())
                })
                .collect();

            hidden = self.layer_forward(&hidden, &weights, i, &pe_cos, &pe_sin, &attn_mask)?;

            // Collect if this is an extract layer
            if cfg.extract_layers.contains(&i) {
                collected.push((i, hidden.clone()));
                log::info!("[Mistral] Captured layer {} output", i);
            }

            if i % 10 == 0 || i == cfg.num_layers - 1 {
                log::info!("[Mistral] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }

        // 4. Stack extracted hidden states: [B, 3, seq_len, hidden] → [B, seq_len, 3*hidden]
        //
        // Musubi: out = torch.stack([...], dim=1) → rearrange("b c l d -> b l (c d)")
        if collected.len() != cfg.extract_layers.len() {
            return Err(flame_core::Error::InvalidInput(format!(
                "Expected {} extract layers, got {}",
                cfg.extract_layers.len(), collected.len()
            )));
        }

        let tensors: Vec<Tensor> = collected.into_iter().map(|(_, t)| t).collect();
        let stacked = Tensor::stack(&tensors, 1)?; // [B, 3, seq_len, hidden]
        let dims = stacked.shape().dims().to_vec();
        let b = dims[0];
        let c = dims[1]; // 3
        let l = dims[2]; // seq_len
        let d = dims[3]; // hidden_size

        // "b c l d -> b l (c d)"
        let result = stacked.permute(&[0, 2, 1, 3])?.reshape(&[b, l, c * d])?;

        log::info!("[Mistral] Output: {:?} (expected [{}, {}, {}])",
            result.shape(), b, l, c * d);

        Ok(result)
    }

    /// Get config reference.
    pub fn config(&self) -> &MistralConfig {
        &self.config
    }

    /// Expected output dim: 3 * 5120 = 15360.
    pub fn output_dim(&self) -> usize {
        self.config.extract_layers.len() * self.config.hidden_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = MistralConfig::default();
        assert_eq!(cfg.vocab_size, 131072);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 32768);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.rope_theta, 1e9);
    }

    #[test]
    fn test_extract_layers() {
        let cfg = MistralConfig::default();
        // 0-indexed: 9, 19, 29 → 1-indexed hidden_states: 10, 20, 30
        assert_eq!(cfg.extract_layers, vec![9, 19, 29]);
    }

    #[test]
    fn test_output_dim() {
        let cfg = MistralConfig::default();
        assert_eq!(cfg.extract_layers.len() * cfg.hidden_size, 15360);
    }

    #[test]
    fn test_gqa_ratio() {
        let cfg = MistralConfig::default();
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 4); // 4 Q heads per KV head
    }

    #[test]
    fn test_q_proj_dim() {
        let cfg = MistralConfig::default();
        // q_proj: [num_heads * head_dim, hidden_size] = [4096, 5120]
        assert_eq!(cfg.num_heads * cfg.head_dim, 4096);
        // k_proj: [num_kv_heads * head_dim, hidden_size] = [1024, 5120]
        assert_eq!(cfg.num_kv_heads * cfg.head_dim, 1024);
    }
}
