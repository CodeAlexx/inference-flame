//! ACE-Step condition encoder for music generation inference.
//!
//! Encodes text, lyrics, and timbre into a packed condition sequence for
//! cross-attention in the DiT decoder. All sub-components use the same
//! encoder layer pattern (Qwen3-style bidirectional transformer):
//!   - RMSNorm -> self-attention (bidirectional, no causal mask) -> residual
//!   - RMSNorm -> SwiGLU MLP -> residual
//!   - GQA: 16 heads, 8 KV heads, head_dim=128, RoPE, QK-norm
//!
//! Sub-components:
//!   1. Text projector: Linear(1024 -> 2048, no bias) — simple matmul
//!   2. Lyric encoder: embed_tokens + 8 encoder layers + final RMSNorm
//!   3. Timbre encoder: embed_tokens + 4 encoder layers + final RMSNorm
//!      (for text2music: outputs zeros — no reference audio)
//!   4. pack_sequences: left-pack valid tokens from two sequences
//!   5. prepare_condition: orchestrates all sub-components
//!
//! Weight prefixes:
//!   - encoder.text_projector.*
//!   - encoder.lyric_encoder.*
//!   - encoder.timbre_encoder.*

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the ACE-Step condition encoder sub-components.
/// Matches the encoder-specific fields from the ACE-Step config.json.
#[derive(Debug, Clone)]
pub struct AceStepConditionConfig {
    /// Encoder hidden size (2048 for xl-base).
    pub hidden_size: usize,
    /// Encoder intermediate (MLP) size (6144 for xl-base).
    pub intermediate_size: usize,
    /// Number of attention heads (16 for xl-base).
    pub num_heads: usize,
    /// Number of KV heads for GQA (8 for xl-base).
    pub num_kv_heads: usize,
    /// Head dimension (128).
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE theta.
    pub rope_theta: f64,
    /// Input dim for text projector (1024 — from text encoder output).
    pub text_hidden_dim: usize,
    /// Input dim for timbre encoder (64 — acoustic hidden dim).
    pub timbre_hidden_dim: usize,
    /// Number of lyric encoder layers (8).
    pub num_lyric_layers: usize,
    /// Number of timbre encoder layers (4).
    pub num_timbre_layers: usize,
}

impl Default for AceStepConditionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 6144,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            text_hidden_dim: 1024,
            timbre_hidden_dim: 64,
            num_lyric_layers: 8,
            num_timbre_layers: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// ACE-Step condition encoder.
///
/// Holds all encoder weights as a flat HashMap and runs the forward pass
/// by indexing into it. Follows the same pattern as `Qwen3Encoder`.
pub struct AceStepConditionEncoder {
    weights: HashMap<String, Tensor>,
    config: AceStepConditionConfig,
    device: Arc<CudaDevice>,
}

impl AceStepConditionEncoder {
    /// Create from pre-loaded weight tensors.
    ///
    /// All tensors must already be on the target CUDA device and in BF16.
    /// 2D weight matrices are pre-transposed for efficient matmul.
    pub fn new(
        weights: HashMap<String, Tensor>,
        config: AceStepConditionConfig,
        device: Arc<CudaDevice>,
    ) -> Self {
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            // Pre-transpose 2D weight matrices [out, in] -> [in, out]
            // Skip norm weights and 1D parameters.
            if key.ends_with(".weight")
                && !key.contains("norm")
                && !key.contains("special_token")
            {
                let w = &weights[key];
                if w.shape().dims().len() == 2 {
                    if let Ok(wt) = flame_core::bf16_elementwise::transpose2d_bf16(w) {
                        weights.insert(key.clone(), wt);
                    }
                }
            }
        }
        Self {
            weights,
            config,
            device,
        }
    }

    /// Build from a safetensors weight map, filtering for encoder.* keys.
    ///
    /// Caller should load safetensors and pass only keys starting with
    /// "encoder." (text_projector, lyric_encoder, timbre_encoder).
    pub fn from_weights(
        weights: HashMap<String, Tensor>,
        device: Arc<CudaDevice>,
    ) -> Self {
        Self::new(weights, AceStepConditionConfig::default(), device)
    }

    /// Get a reference to a weight tensor.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight key: {key}"))
        })
    }

    /// Get the config.
    pub fn config(&self) -> &AceStepConditionConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Linear projection helper
    // -----------------------------------------------------------------------

    /// Matmul for [B, N, C] x [C, out] -> [B, N, out].
    /// Weight is already pre-transposed at construction time.
    fn linear_3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        if shape.len() == 2 {
            return x.matmul(weight_t);
        }
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(weight_t)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    // -----------------------------------------------------------------------
    // RMSNorm helper
    // -----------------------------------------------------------------------

    /// Apply RMSNorm: reshape to 2D, normalize, reshape back.
    fn rms_norm_apply(
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out_2d.reshape(&dims)
    }

    // -----------------------------------------------------------------------
    // RoPE — 1D bidirectional
    // -----------------------------------------------------------------------

    /// Build 1D RoPE cos/sin tables.
    ///
    /// Returns (cos, sin), each shaped `[1, 1, seq_len, head_dim / 2]` in BF16.
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

    /// Apply rotary position embeddings (half-split convention).
    fn apply_rope_single(
        x: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<Tensor> {
        flame_core::bf16_ops::rope_halfsplit_bf16(x, pe_cos, pe_sin)
    }

    // -----------------------------------------------------------------------
    // GQA head repeat
    // -----------------------------------------------------------------------

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
    // Bidirectional attention mask
    // -----------------------------------------------------------------------

    /// Build a bidirectional attention mask from a padding mask [B, L].
    ///
    /// Returns [B, 1, L, L] in BF16 where mask[b,0,i,j] = 1.0 if
    /// padding_mask[b,j] == 1 (token j is real), else 0.0.
    /// All real tokens can attend to all other real tokens (no causal constraint).
    fn build_bidirectional_mask(
        padding_mask: &Tensor,
        seq_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        // padding_mask: [B, L] with 1=real, 0=pad
        // We want [B, 1, L, L] where col j is masked out if padding_mask[b,j]==0
        // Expand: [B, L] -> [B, 1, 1, L] and broadcast over query dim
        let mask_4d = padding_mask
            .unsqueeze(1)? // [B, 1, L]
            .unsqueeze(2)?; // [B, 1, 1, L]
        // Broadcast to [B, 1, L, L] by expanding query dimension
        let ones_q = Tensor::ones_dtype(
            Shape::from_dims(&[1, 1, seq_len, 1]),
            DType::BF16,
            device.clone(),
        )?;
        mask_4d.to_dtype(DType::BF16)?.mul(&ones_q)
    }

    // -----------------------------------------------------------------------
    // Shared encoder layer forward
    // -----------------------------------------------------------------------

    /// Execute one bidirectional encoder layer.
    ///
    /// Shared between lyric_encoder and timbre_encoder — only the weight prefix
    /// differs. Architecture: RMSNorm -> GQA self-attn -> residual -> RMSNorm
    /// -> SwiGLU MLP -> residual.
    fn encoder_layer_forward(
        &self,
        prefix: &str,
        layer_idx: usize,
        hidden: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;

        let lp = format!("{prefix}.layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        // --- Self-attention ---

        let norm_w = self.w(&format!("{lp}.input_layernorm.weight"))?;
        let normed = Self::rms_norm_apply(hidden, norm_w, cfg.rms_norm_eps)?;

        let q_w = self.w(&format!("{lp}.self_attn.q_proj.weight"))?;
        let k_w = self.w(&format!("{lp}.self_attn.k_proj.weight"))?;
        let v_w = self.w(&format!("{lp}.self_attn.v_proj.weight"))?;

        let q = Self::linear_3d(&normed, q_w)?;
        let k = Self::linear_3d(&normed, k_w)?;
        let v = Self::linear_3d(&normed, v_w)?;

        let mut q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let mut k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // QK norm (per-head RMSNorm)
        let q_norm_key = format!("{lp}.self_attn.q_norm.weight");
        let k_norm_key = format!("{lp}.self_attn.k_norm.weight");
        if let (Some(q_norm_w), Some(k_norm_w)) =
            (self.weights.get(&q_norm_key), self.weights.get(&k_norm_key))
        {
            let q_flat = q.reshape(&[b * h * n, d])?;
            let q_normed =
                flame_core::cuda_ops_bf16::rms_norm_bf16(&q_flat, Some(q_norm_w), cfg.rms_norm_eps)?;
            q = q_normed.reshape(&[b, h, n, d])?;

            let k_flat = k.reshape(&[b * h_kv * n, d])?;
            let k_normed =
                flame_core::cuda_ops_bf16::rms_norm_bf16(&k_flat, Some(k_norm_w), cfg.rms_norm_eps)?;
            k = k_normed.reshape(&[b, h_kv, n, d])?;
        }

        // RoPE (bidirectional — same positional encoding, just no causal mask)
        let q = Self::apply_rope_single(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope_single(&k, pe_cos, pe_sin)?;

        // GQA: repeat KV heads
        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        // Attention (bidirectional mask — no causal constraint)
        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?;

        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        let o_w = self.w(&format!("{lp}.self_attn.o_proj.weight"))?;
        let attn_out = Self::linear_3d(&attn_out, o_w)?;

        let hidden = hidden.add(&attn_out)?;

        // --- MLP (SwiGLU) ---

        let post_norm_w = self.w(&format!("{lp}.post_attention_layernorm.weight"))?;
        let normed2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps)?;

        let gate_w = self.w(&format!("{lp}.mlp.gate_proj.weight"))?;
        let up_w = self.w(&format!("{lp}.mlp.up_proj.weight"))?;
        let down_w = self.w(&format!("{lp}.mlp.down_proj.weight"))?;

        let gate = Self::linear_3d(&normed2, gate_w)?;
        let up = Self::linear_3d(&normed2, up_w)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, down_w)?;

        hidden.add(&mlp_out)
    }

    // -----------------------------------------------------------------------
    // Text projector
    // -----------------------------------------------------------------------

    /// Project text hidden states: [B, L, 1024] -> [B, L, 2048].
    fn text_project(&self, text_hidden_states: &Tensor) -> Result<Tensor> {
        let w = self.w("encoder.text_projector.weight")?;
        Self::linear_3d(text_hidden_states, w)
    }

    // -----------------------------------------------------------------------
    // Lyric encoder
    // -----------------------------------------------------------------------

    /// Encode lyrics: [B, L, 1024] -> [B, L, 2048].
    ///
    /// embed_tokens -> 8 bidirectional encoder layers -> final RMSNorm.
    fn lyric_encode(
        &self,
        lyric_hidden_states: &Tensor,
        lyric_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let prefix = "encoder.lyric_encoder";

        // Project: [B, L, 1024] -> [B, L, 2048]
        let embed_w = self.w(&format!("{prefix}.embed_tokens.weight"))?;
        let mut hidden = Self::linear_3d(lyric_hidden_states, embed_w)?;

        let seq_len = hidden.shape().dims()[1];

        // Build RoPE and attention mask
        let (pe_cos, pe_sin) =
            Self::build_rope_1d(seq_len, cfg.head_dim, cfg.rope_theta, &self.device)?;
        let attn_mask = Self::build_bidirectional_mask(lyric_mask, seq_len, &self.device)?;

        // Pass through encoder layers
        for i in 0..cfg.num_lyric_layers {
            hidden = self.encoder_layer_forward(prefix, i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;
        }

        // Final norm
        let norm_w = self.w(&format!("{prefix}.norm.weight"))?;
        Self::rms_norm_apply(&hidden, norm_w, cfg.rms_norm_eps)
    }

    // -----------------------------------------------------------------------
    // Timbre encoder (text2music path: returns zeros)
    // -----------------------------------------------------------------------

    /// Encode timbre from reference audio.
    ///
    /// For text2music (no reference audio), call `timbre_encode_silence` instead.
    ///
    /// Input: [N, T, 64] packed acoustic features
    /// Output: (timbre_embs [B, max_count, 2048], mask [B, max_count])
    #[allow(dead_code)]
    fn timbre_encode(
        &self,
        refer_audio: &Tensor,
        _order_mask: &Tensor,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let prefix = "encoder.timbre_encoder";

        // Project: [N, T, 64] -> [N, T, 2048]
        let embed_w = self.w(&format!("{prefix}.embed_tokens.weight"))?;
        let mut hidden = Self::linear_3d(refer_audio, embed_w)?;

        let seq_len = hidden.shape().dims()[1];

        let (pe_cos, pe_sin) =
            Self::build_rope_1d(seq_len, cfg.head_dim, cfg.rope_theta, &self.device)?;

        // For timbre encoder, build an all-ones mask (no padding in packed seqs)
        let n = hidden.shape().dims()[0];
        let ones_mask = Tensor::ones_dtype(
            Shape::from_dims(&[n, seq_len]),
            DType::BF16,
            self.device.clone(),
        )?;
        let attn_mask = Self::build_bidirectional_mask(&ones_mask, seq_len, &self.device)?;

        for i in 0..cfg.num_timbre_layers {
            hidden = self.encoder_layer_forward(prefix, i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;
        }

        let norm_w = self.w(&format!("{prefix}.norm.weight"))?;
        hidden = Self::rms_norm_apply(&hidden, norm_w, cfg.rms_norm_eps)?;

        // Extract position 0 as timbre embedding: [N, T, D] -> [N, D]
        let _timbre_emb = hidden.narrow(1, 0, 1)?.squeeze(Some(1))?;

        // For text2music with batch_size, return zeros (simplified path).
        // Full unpack_timbre_embeddings logic is needed for cover/timbre mode.
        let d = cfg.hidden_size;
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, 1, d]),
            DType::BF16,
            self.device.clone(),
        )?;
        let mask = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, 1]),
            DType::BF16,
            self.device.clone(),
        )?;
        Ok((zeros, mask))
    }

    /// Text2music path: timbre encoder produces zeros (no reference audio).
    ///
    /// Returns (zeros [B, 1, 2048], zeros_mask [B, 1]).
    fn timbre_encode_silence(
        &self,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let d = self.config.hidden_size;
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, 1, d]),
            DType::BF16,
            self.device.clone(),
        )?;
        let mask = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, 1]),
            DType::BF16,
            self.device.clone(),
        )?;
        Ok((zeros, mask))
    }

    // -----------------------------------------------------------------------
    // pack_sequences
    // -----------------------------------------------------------------------

    /// Pack two sequences by concatenating and sorting valid tokens first.
    ///
    /// Concatenates hidden1 and hidden2 along dim=1, then reorders so that
    /// positions with mask=1 come before mask=0 (left-packing).
    ///
    /// Returns (packed_hidden [B, L1+L2, D], new_mask [B, L1+L2] in BF16).
    fn pack_sequences(
        hidden1: &Tensor,
        hidden2: &Tensor,
        mask1: &Tensor,
        mask2: &Tensor,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let h1_refs: Vec<&Tensor> = vec![hidden1, hidden2];
        let hidden_cat = Tensor::cat(&h1_refs, 1)?; // [B, L, D]
        let m_refs: Vec<&Tensor> = vec![mask1, mask2];
        let mask_cat = Tensor::cat(&m_refs, 1)?; // [B, L]

        let dims = hidden_cat.shape().dims().to_vec();
        let b = dims[0];
        let l = dims[1];
        let _d = dims[2];

        // For inference with batch_size=1 or small batches, we can do the
        // sort+gather on CPU since it's just index manipulation.
        // For now, implement a CPU-side sort and gather approach.

        // Convert mask to f32 on CPU for sorting
        let mask_f32 = mask_cat.to_dtype(DType::F32)?;
        let mask_data = mask_f32.to_vec2()?; // [B, L]

        // Build sort indices per batch element (descending by mask value, stable)
        let mut all_indices: Vec<f32> = Vec::with_capacity(b * l);
        let mut all_lengths: Vec<f32> = Vec::with_capacity(b);

        for bi in 0..b {
            let row: &Vec<f32> = &mask_data[bi];
            let mut indices: Vec<usize> = (0..l).collect();
            // Stable sort descending: mask=1 before mask=0
            indices.sort_by(|&a, &b_idx| {
                row[b_idx].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            all_indices.extend(indices.iter().map(|&idx| idx as f32));

            // Count valid tokens
            let count: f32 = row.iter().sum();
            all_lengths.push(count);
        }

        // Build sort_idx tensor [B, L] as I32
        let sort_idx_f32 = Tensor::from_vec(
            all_indices,
            Shape::from_dims(&[b, l]),
            device.clone(),
        )?;
        let sort_idx = sort_idx_f32.to_dtype(DType::I32)?;

        // Gather hidden_cat along dim=1 using sort indices.
        // gather_rows only supports dim=0, so do the gather on CPU —
        // this is a small tensor (condition sequence, ~2-4K tokens) and
        // runs only once per generation.
        let hidden_f32 = hidden_cat.to_dtype(DType::F32)?;
        let hidden_cpu: Vec<f32> = hidden_f32.to_vec()?;
        let d = _d;
        let mut packed_data = vec![0.0f32; b * l * d];
        let idx_data: Vec<f32> = sort_idx_f32.to_vec()?;
        for bi in 0..b {
            for li in 0..l {
                let src_idx = idx_data[bi * l + li] as usize;
                let dst_off = (bi * l + li) * d;
                let src_off = (bi * l + src_idx) * d;
                packed_data[dst_off..dst_off + d]
                    .copy_from_slice(&hidden_cpu[src_off..src_off + d]);
            }
        }
        let hidden_packed = Tensor::from_vec(
            packed_data,
            Shape::from_dims(&[b, l, d]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Build new_mask: arange(L) < lengths[b]
        let mut new_mask_data = vec![0.0f32; b * l];
        for bi in 0..b {
            let len = all_lengths[bi] as usize;
            for j in 0..len.min(l) {
                new_mask_data[bi * l + j] = 1.0;
            }
        }
        let new_mask = Tensor::from_vec(
            new_mask_data,
            Shape::from_dims(&[b, l]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;

        Ok((hidden_packed, new_mask))
    }

    // -----------------------------------------------------------------------
    // prepare_condition — main entry point
    // -----------------------------------------------------------------------

    /// Prepare condition inputs for the DiT decoder.
    ///
    /// This is the text2music path (no reference audio / cover).
    ///
    /// # Arguments
    /// - `text_hidden_states`: [B, L_text, 1024] — text encoder output
    /// - `text_attention_mask`: [B, L_text] — 1=real, 0=pad (BF16)
    /// - `lyric_hidden_states`: [B, L_lyric, 1024] — lyric embeddings
    /// - `lyric_attention_mask`: [B, L_lyric] — 1=real, 0=pad (BF16)
    /// - `silence_latent`: [B, T, 64] — silence acoustic latent
    /// - `num_timesteps`: T — number of acoustic timesteps
    ///
    /// # Returns
    /// `(encoder_hidden_states, encoder_attention_mask, context_latents)`:
    /// - encoder_hidden_states: [B, L_packed, 2048]
    /// - encoder_attention_mask: [B, L_packed] (BF16)
    /// - context_latents: [B, T, 128] (cat of silence + ones)
    pub fn prepare_condition(
        &self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        silence_latent: &Tensor,
        num_timesteps: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let batch_size = text_hidden_states.shape().dims()[0];

        // 1. Project text: [B, L_text, 1024] -> [B, L_text, 2048]
        let text_projected = self.text_project(text_hidden_states)?;

        // 2. Encode lyrics: [B, L_lyric, 1024] -> [B, L_lyric, 2048]
        let lyric_encoded = self.lyric_encode(lyric_hidden_states, lyric_attention_mask)?;

        // 3. Timbre: for text2music, just zeros
        let (timbre_embs, timbre_mask) = self.timbre_encode_silence(batch_size)?;

        // 4. Pack sequences: first pack(lyrics, timbre), then pack(result, text)
        let (packed1, packed1_mask) = Self::pack_sequences(
            &lyric_encoded,
            &timbre_embs,
            lyric_attention_mask,
            &timbre_mask,
            &self.device,
        )?;
        let (encoder_hidden_states, encoder_attention_mask) = Self::pack_sequences(
            &packed1,
            &text_projected,
            &packed1_mask,
            text_attention_mask,
            &self.device,
        )?;

        // 5. Build context_latents: cat(silence[B,T,64], ones[B,T,64]) -> [B, T, 128]
        let chunk_masks = Tensor::ones_dtype(
            Shape::from_dims(&[batch_size, num_timesteps, 64]),
            DType::BF16,
            self.device.clone(),
        )?;
        let ctx_refs: Vec<&Tensor> = vec![silence_latent, &chunk_masks];
        let context_latents = Tensor::cat(&ctx_refs, 2)?; // [B, T, 128]

        Ok((encoder_hidden_states, encoder_attention_mask, context_latents))
    }

    /// Get read-only access to weights (for debugging).
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Weight key listing (for validation)
// ---------------------------------------------------------------------------

/// Return all expected weight keys for the condition encoder.
pub fn expected_weight_keys(config: &AceStepConditionConfig) -> Vec<String> {
    let mut keys = Vec::new();

    // Text projector (no bias)
    keys.push("encoder.text_projector.weight".to_string());

    // Lyric encoder
    let lyric_prefix = "encoder.lyric_encoder";
    keys.push(format!("{lyric_prefix}.embed_tokens.weight"));
    keys.push(format!("{lyric_prefix}.norm.weight"));
    for i in 0..config.num_lyric_layers {
        let lp = format!("{lyric_prefix}.layers.{i}");
        keys.extend([
            format!("{lp}.self_attn.q_proj.weight"),
            format!("{lp}.self_attn.k_proj.weight"),
            format!("{lp}.self_attn.v_proj.weight"),
            format!("{lp}.self_attn.o_proj.weight"),
            format!("{lp}.self_attn.q_norm.weight"),
            format!("{lp}.self_attn.k_norm.weight"),
            format!("{lp}.mlp.gate_proj.weight"),
            format!("{lp}.mlp.up_proj.weight"),
            format!("{lp}.mlp.down_proj.weight"),
            format!("{lp}.input_layernorm.weight"),
            format!("{lp}.post_attention_layernorm.weight"),
        ]);
    }

    // Timbre encoder
    let timbre_prefix = "encoder.timbre_encoder";
    keys.push(format!("{timbre_prefix}.embed_tokens.weight"));
    keys.push(format!("{timbre_prefix}.special_token"));
    keys.push(format!("{timbre_prefix}.norm.weight"));
    for i in 0..config.num_timbre_layers {
        let lp = format!("{timbre_prefix}.layers.{i}");
        keys.extend([
            format!("{lp}.self_attn.q_proj.weight"),
            format!("{lp}.self_attn.k_proj.weight"),
            format!("{lp}.self_attn.v_proj.weight"),
            format!("{lp}.self_attn.o_proj.weight"),
            format!("{lp}.self_attn.q_norm.weight"),
            format!("{lp}.self_attn.k_norm.weight"),
            format!("{lp}.mlp.gate_proj.weight"),
            format!("{lp}.mlp.up_proj.weight"),
            format!("{lp}.mlp.down_proj.weight"),
            format!("{lp}.input_layernorm.weight"),
            format!("{lp}.post_attention_layernorm.weight"),
        ]);
    }

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
        let cfg = AceStepConditionConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.text_hidden_dim, 1024);
        assert_eq!(cfg.timbre_hidden_dim, 64);
        assert_eq!(cfg.num_lyric_layers, 8);
        assert_eq!(cfg.num_timbre_layers, 4);
    }

    #[test]
    fn test_expected_weight_keys_count() {
        let cfg = AceStepConditionConfig::default();
        let keys = expected_weight_keys(&cfg);
        // text_projector: 1
        // lyric_encoder: 1 (embed) + 1 (norm) + 8 * 11 (layers) = 90
        // timbre_encoder: 1 (embed) + 1 (special_token) + 1 (norm) + 4 * 11 (layers) = 47
        // Total: 1 + 90 + 47 = 138
        assert_eq!(keys.len(), 1 + (2 + 8 * 11) + (3 + 4 * 11));
    }

    #[test]
    fn test_gqa_head_ratio() {
        let cfg = AceStepConditionConfig::default();
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 2);
    }
}
