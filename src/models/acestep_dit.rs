//! ACE-Step DiT decoder — inference-only forward pass.
//!
//! Ported from the training model (flame-diffusion/acestep-trainer/src/model.rs)
//! with all LoRA, autograd, and training-specific code removed.
//!
//! Architecture: 24-layer DiT with self-attn + cross-attn, AdaLN modulation,
//!   GQA (16 heads, 8 KV heads), RoPE, SiLU-gated MLP (SwiGLU).
//!   Patch embed via Conv1d(in=192, out=2048, k=2, s=2).
//!
//! Weight key patterns (from safetensors, decoder.* prefix):
//!   decoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   decoder.layers.{i}.cross_attn.{q,k,v,o}_proj.weight
//!   decoder.layers.{i}.self_attn.{q,k}_norm.weight
//!   decoder.layers.{i}.cross_attn.{q,k}_norm.weight
//!   decoder.layers.{i}.{self_attn_norm,cross_attn_norm,mlp_norm}.weight
//!   decoder.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
//!   decoder.layers.{i}.scale_shift_table  [1, 6, 2048]
//!   decoder.time_embed{,_r}.{linear_1,linear_2,time_proj}.{weight,bias}
//!   decoder.{proj_in.1,proj_out.1}.{weight,bias}  (Conv1d / ConvTranspose1d)
//!   decoder.{condition_embedder,norm_out}.weight
//!   decoder.condition_embedder.bias
//!   decoder.scale_shift_table  [1, 2, 2048]
//!   null_condition_emb  [1, 1, 2048]

use flame_core::serialization;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// ACE-Step DiT decoder configuration, auto-detected from weights.
#[derive(Debug, Clone)]
pub struct AceStepDiTConfig {
    pub hidden_size: usize,       // 2048 (turbo) or 2560 (xl-base)
    pub num_heads: usize,         // 16 or 32
    pub num_kv_heads: usize,      // 8 or 32
    pub head_dim: usize,          // 128
    pub intermediate_size: usize, // 6144 or larger
    pub num_layers: usize,        // 24 or 32
    pub in_channels: usize,       // 192
    pub acoustic_dim: usize,      // 64
    pub patch_size: usize,        // 2
    pub rope_theta: f32,          // 1_000_000
    pub rms_norm_eps: f32,        // 1e-6
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// ACE-Step DiT decoder for inference (no LoRA, no autograd).
pub struct AceStepDiT {
    weights: HashMap<String, Tensor>,
    config: AceStepDiTConfig,
    /// Optional KV cache for cross-attention (populated on first step, reused).
    /// Each entry is (K, V) for one layer.
    cross_kv_cache: Option<Vec<(Tensor, Tensor)>>,
}

impl AceStepDiT {
    /// Load from a safetensors checkpoint, keeping only `decoder.*` keys.
    pub fn from_safetensors(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let all_weights = serialization::load_file(path, device)?;
        Self::from_weights(all_weights)
    }

    /// Build from a pre-loaded weight map. Only `decoder.*` and
    /// `null_condition_emb` keys are used.
    pub fn from_weights(mut weights: HashMap<String, Tensor>) -> Result<Self> {
        // Auto-detect config from weight shapes
        let hidden_size = {
            let w = weights.get("decoder.condition_embedder.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.condition_embedder.weight".into()))?;
            w.shape().dims()[0]
        };

        let num_heads = {
            let q = weights.get("decoder.layers.0.self_attn.q_proj.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.layers.0.self_attn.q_proj.weight".into()))?;
            q.shape().dims()[0] / 128
        };

        let num_kv_heads = {
            let k = weights.get("decoder.layers.0.self_attn.k_proj.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.layers.0.self_attn.k_proj.weight".into()))?;
            k.shape().dims()[0] / 128
        };

        let intermediate_size = {
            let g = weights.get("decoder.layers.0.mlp.gate_proj.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.layers.0.mlp.gate_proj.weight".into()))?;
            g.shape().dims()[0]
        };

        let in_channels = {
            let p = weights.get("decoder.proj_in.1.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.proj_in.1.weight".into()))?;
            // Conv1d weight: [out_channels, in_channels, kernel_size]
            p.shape().dims()[1]
        };

        let acoustic_dim = {
            let p = weights.get("decoder.proj_out.1.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.proj_out.1.weight".into()))?;
            // ConvTranspose1d weight: [in_channels, out_channels, kernel_size]
            p.shape().dims()[1]
        };

        let mut num_layers = 0;
        while weights.contains_key(&format!("decoder.layers.{num_layers}.self_attn.q_proj.weight")) {
            num_layers += 1;
        }

        let config = AceStepDiTConfig {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim: 128,
            intermediate_size,
            num_layers,
            in_channels,
            acoustic_dim,
            patch_size: 2,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };

        log::info!(
            "ACE-Step DiT config: hidden={} heads={} kv_heads={} layers={} mlp={} in_ch={} acoustic={}",
            config.hidden_size, config.num_heads, config.num_kv_heads,
            config.num_layers, config.intermediate_size, config.in_channels,
            config.acoustic_dim,
        );

        // Convert all decoder weights to BF16, pre-transpose 2D weight matrices
        let decoder_keys: Vec<String> = weights.keys()
            .filter(|k| k.starts_with("decoder."))
            .cloned()
            .collect();
        for key in decoder_keys {
            let tensor = weights.remove(&key).unwrap();
            let tensor = tensor.to_dtype(DType::BF16)?;
            let dims = tensor.shape().dims().to_vec();
            // Pre-transpose 2D weight matrices for fused linear
            // (Conv1d weights are 3D -- skip those)
            let stored = if key.ends_with(".weight") && dims.len() == 2 {
                tensor.transpose()?
            } else {
                tensor
            };
            weights.insert(key, stored);
        }

        // Ensure null_condition_emb is BF16
        if let Some(nce) = weights.remove("null_condition_emb") {
            weights.insert("null_condition_emb".to_string(), nce.to_dtype(DType::BF16)?);
        }

        Ok(Self {
            weights,
            config,
            cross_kv_cache: None,
        })
    }

    /// Get the model config.
    pub fn config(&self) -> &AceStepDiTConfig {
        &self.config
    }

    /// Get a reference to the null condition embedding [1, 1, hidden_size].
    pub fn null_condition_emb(&self) -> Result<&Tensor> {
        self.weights.get("null_condition_emb")
            .ok_or_else(|| Error::InvalidInput("Missing null_condition_emb".into()))
    }

    /// Clear the cross-attention KV cache (call between generations).
    pub fn clear_cache(&mut self) {
        self.cross_kv_cache = None;
    }

    // -----------------------------------------------------------------------
    // Weight accessor
    // -----------------------------------------------------------------------

    #[inline]
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key)
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight: {key}")))
    }

    // -----------------------------------------------------------------------
    // Timestep embedding
    // Reference: modeling_acestep_v15_base.py TimestepEmbedding
    // -----------------------------------------------------------------------

    /// Compute timestep embedding + 6-way modulation projection.
    /// Returns (temb [B, hidden], timestep_proj [B, 6, hidden]).
    fn timestep_embedding_full(
        &self,
        t: &Tensor,
        prefix: &str,
    ) -> Result<(Tensor, Tensor)> {
        let device = t.device();
        let b = t.shape().dims()[0];
        let h = self.config.hidden_size;

        // Sinusoidal embedding: dim=256, scale=1000
        let t_scaled = t.mul_scalar(1000.0)?;
        let half = 128; // 256 / 2
        let freqs_data: Vec<f32> = (0..half)
            .map(|i| (-(10000.0f32.ln()) * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs_data, Shape::from_dims(&[1, half]), device.clone())?;

        let t_2d = t_scaled.to_dtype(DType::F32)?.reshape(&[b, 1])?;
        let args = t_2d
            .broadcast_to(&Shape::from_dims(&[b, half]))?
            .mul(&freqs.broadcast_to(&Shape::from_dims(&[b, half]))?)?;
        let embedding = Tensor::cat(&[&args.cos()?, &args.sin()?], 1)?
            .to_dtype(DType::BF16)?;

        // MLP: linear_1 -> silu -> linear_2
        let emb_3d = embedding.reshape(&[b, 1, 256])?;
        let temb = linear3d_bias(
            &emb_3d,
            self.w(&format!("{prefix}.linear_1.weight"))?,
            self.w(&format!("{prefix}.linear_1.bias"))?,
        )?;
        let temb = temb.silu()?;
        let temb = linear3d_bias(
            &temb,
            self.w(&format!("{prefix}.linear_2.weight"))?,
            self.w(&format!("{prefix}.linear_2.bias"))?,
        )?;

        // time_proj: silu -> linear -> reshape to [B, 6, H]
        let proj = linear3d_bias(
            &temb.silu()?,
            self.w(&format!("{prefix}.time_proj.weight"))?,
            self.w(&format!("{prefix}.time_proj.bias"))?,
        )?;
        let timestep_proj = proj.reshape(&[b, 6, h])?;

        // temb: [B, 1, H] -> [B, H]
        let temb = temb.reshape(&[b, h])?;

        Ok((temb, timestep_proj))
    }

    // -----------------------------------------------------------------------
    // RoPE
    // -----------------------------------------------------------------------

    /// Compute RoPE cos/sin tensors for a given sequence length.
    /// Returns (cos, sin) each [seq_len, head_dim/2] in BF16.
    fn compute_rope(
        &self,
        seq_len: usize,
        device: Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let hd = self.config.head_dim;
        let half = hd / 2;
        let theta = self.config.rope_theta;

        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / hd as f32))
            .collect();
        let inv_freq_t = Tensor::from_vec(inv_freq, Shape::from_dims(&[1, half]), device.clone())?;

        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_t = Tensor::from_vec(positions, Shape::from_dims(&[seq_len, 1]), device.clone())?;

        let angles = pos_t.matmul(&inv_freq_t)?;
        let cos = angles.cos()?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.to_dtype(DType::BF16)?;

        Ok((cos, sin))
    }

    // -----------------------------------------------------------------------
    // Self-attention (GQA + RoPE + QK norm, no LoRA)
    // -----------------------------------------------------------------------

    fn self_attention_forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer_idx}.self_attn");
        let b = hidden_states.shape().dims()[0];
        let s = hidden_states.shape().dims()[1];
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;

        // Q, K, V projections (base weights only)
        let q = linear3d(hidden_states, self.w(&format!("{prefix}.q_proj.weight"))?)?;
        let k = linear3d(hidden_states, self.w(&format!("{prefix}.k_proj.weight"))?)?;
        let v = linear3d(hidden_states, self.w(&format!("{prefix}.v_proj.weight"))?)?;

        // Reshape to [B, S, heads, head_dim]
        let q = q.reshape(&[b, s, nh, hd])?;
        let k = k.reshape(&[b, s, nkv, hd])?;
        let v = v.reshape(&[b, s, nkv, hd])?;

        // QK norm (RMSNorm on head_dim)
        let q = rms_norm_per_head(&q, self.w(&format!("{prefix}.q_norm.weight"))?, self.config.rms_norm_eps)?;
        let k = rms_norm_per_head(&k, self.w(&format!("{prefix}.k_norm.weight"))?, self.config.rms_norm_eps)?;

        // Transpose to [B, heads, S, head_dim]
        let q = q.transpose_dims(1, 2)?;
        let k = k.transpose_dims(1, 2)?;
        let v = v.transpose_dims(1, 2)?;

        // Apply RoPE
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat KV heads to match Q heads
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        // Scaled dot-product attention
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Transpose back: [B, heads, S, hd] -> [B, S, heads*hd]
        let out = attn_out.transpose_dims(1, 2)?.reshape(&[b, s, nh * hd])?;

        // Output projection
        linear3d(&out, self.w(&format!("{prefix}.o_proj.weight"))?)
    }

    // -----------------------------------------------------------------------
    // Cross-attention (GQA + QK norm, no RoPE, optional KV cache)
    // -----------------------------------------------------------------------

    fn cross_attention_forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer_idx}.cross_attn");
        let b = hidden_states.shape().dims()[0];
        let s_q = hidden_states.shape().dims()[1];
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;

        // Q from hidden_states
        let q = linear3d(hidden_states, self.w(&format!("{prefix}.q_proj.weight"))?)?;
        let q = q.reshape(&[b, s_q, nh, hd])?;
        let q = rms_norm_per_head(&q, self.w(&format!("{prefix}.q_norm.weight"))?, self.config.rms_norm_eps)?;
        let q = q.transpose_dims(1, 2)?;

        // K, V: use cache if available, otherwise compute and cache
        let (k, v) = if let Some(ref cache) = self.cross_kv_cache {
            // Reuse cached K, V (encoder_hidden_states doesn't change between steps)
            (cache[layer_idx].0.clone(), cache[layer_idx].1.clone())
        } else {
            // Compute K, V from encoder_hidden_states
            let s_kv = encoder_hidden_states.shape().dims()[1];
            let k_raw = linear3d(encoder_hidden_states, self.w(&format!("{prefix}.k_proj.weight"))?)?;
            let v_raw = linear3d(encoder_hidden_states, self.w(&format!("{prefix}.v_proj.weight"))?)?;

            let k_raw = k_raw.reshape(&[b, s_kv, nkv, hd])?;
            let v_raw = v_raw.reshape(&[b, s_kv, nkv, hd])?;

            let k_normed = rms_norm_per_head(&k_raw, self.w(&format!("{prefix}.k_norm.weight"))?, self.config.rms_norm_eps)?;

            let k_out = k_normed.transpose_dims(1, 2)?;
            let v_out = v_raw.transpose_dims(1, 2)?;

            (k_out, v_out)
        };

        // GQA
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        // SDPA (no RoPE for cross-attention)
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Reshape back
        let out = attn_out.transpose_dims(1, 2)?.reshape(&[b, s_q, nh * hd])?;

        // Output projection
        linear3d(&out, self.w(&format!("{prefix}.o_proj.weight"))?)
    }

    // -----------------------------------------------------------------------
    // Single DiT layer
    // Reference: modeling_acestep_v15_base.py AceStepDiTLayer.forward
    // -----------------------------------------------------------------------

    fn dit_layer_forward(
        &self,
        hidden_states: &Tensor,
        timestep_proj: &Tensor, // [B, 6, H]
        encoder_hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer_idx}");
        let h = self.config.hidden_size;
        let b = hidden_states.shape().dims()[0];
        let device = hidden_states.device();

        // Extract 6-way modulation: scale_shift_table + timestep_proj
        let sst = self.w(&format!("{prefix}.scale_shift_table"))?; // [1, 6, H]
        let modulation = sst.broadcast_to(&Shape::from_dims(&[b, 6, h]))?.add(timestep_proj)?;

        let shift_msa = modulation.narrow(1, 0, 1)?; // [B, 1, H]
        let scale_msa = modulation.narrow(1, 1, 1)?;
        let gate_msa = modulation.narrow(1, 2, 1)?;
        let c_shift = modulation.narrow(1, 3, 1)?;
        let c_scale = modulation.narrow(1, 4, 1)?;
        let c_gate = modulation.narrow(1, 5, 1)?;

        // --- Step 1: Self-attention with AdaLN ---
        let x_normed = rms_norm(
            hidden_states,
            self.w(&format!("{prefix}.self_attn_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let ones_s = Tensor::ones_dtype(scale_msa.shape().clone(), DType::BF16, device.clone())?;
        let norm_hs = x_normed.mul(&ones_s.add(&scale_msa)?)?.add(&shift_msa)?;

        let attn_out = self.self_attention_forward(&norm_hs, cos, sin, layer_idx)?;

        // Gated residual: x = x + attn_out * gate
        let x = hidden_states.add(&attn_out.mul(&gate_msa)?)?;

        // --- Step 2: Cross-attention ---
        let cross_normed = rms_norm(
            &x,
            self.w(&format!("{prefix}.cross_attn_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let cross_out = self.cross_attention_forward(
            &cross_normed, encoder_hidden_states, layer_idx,
        )?;
        let x = x.add(&cross_out)?;

        // --- Step 3: MLP with AdaLN ---
        let mlp_normed = rms_norm(
            &x,
            self.w(&format!("{prefix}.mlp_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let ones_c = Tensor::ones_dtype(c_scale.shape().clone(), DType::BF16, device.clone())?;
        let mlp_in = mlp_normed.mul(&ones_c.add(&c_scale)?)?.add(&c_shift)?;

        // SwiGLU MLP: gate_proj(x).silu() * up_proj(x) -> down_proj
        let gate = linear3d(&mlp_in, self.w(&format!("{prefix}.mlp.gate_proj.weight"))?)?.silu()?;
        let up = linear3d(&mlp_in, self.w(&format!("{prefix}.mlp.up_proj.weight"))?)?;
        let mlp_out = linear3d(
            &gate.mul(&up)?,
            self.w(&format!("{prefix}.mlp.down_proj.weight"))?,
        )?;

        // Gated residual: x = x + mlp_out * c_gate
        let x = x.add(&mlp_out.mul(&c_gate)?)?;

        Ok(x)
    }

    // -----------------------------------------------------------------------
    // Full decoder forward pass (inference)
    // Reference: modeling_acestep_v15_base.py AceStepDiTModel.forward
    // -----------------------------------------------------------------------

    /// Run the decoder forward pass for inference.
    ///
    /// # Arguments
    /// * `hidden_states` - Noised latents x_t [B, T, acoustic_dim=64]
    /// * `timestep` - Timestep values t [B]
    /// * `timestep_r` - Timestep r values [B] (= t for inference)
    /// * `encoder_hidden_states` - Condition encoder output [B, L, hidden_size]
    /// * `context_latents` - Source context [B, T, 128] (cat with hidden_states inside to make in_channels=192)
    ///
    /// # Returns
    /// Predicted velocity [B, T, acoustic_dim]
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        encoder_hidden_states: &Tensor,
        context_latents: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let b = hidden_states.shape().dims()[0];
        let original_seq_len = hidden_states.shape().dims()[1];
        let h = self.config.hidden_size;

        // --- Timestep embeddings ---
        // temb = temb_t + temb_r; proj = proj_t + proj_r
        let (temb_t, proj_t) = self.timestep_embedding_full(timestep, "decoder.time_embed")?;
        let t_minus_r = timestep.sub(timestep_r)?;
        let (temb_r, proj_r) = self.timestep_embedding_full(&t_minus_r, "decoder.time_embed_r")?;
        let temb = temb_t.add(&temb_r)?;
        let timestep_proj = proj_t.add(&proj_r)?; // [B, 6, hidden]

        // --- Concatenate context_latents with hidden_states ---
        // cat along last dim: context_latents [B, T, in_channels] + hidden_states [B, T, acoustic_dim]
        let last_dim = context_latents.shape().dims().len() - 1;
        let x = Tensor::cat(&[context_latents, hidden_states], last_dim)?;

        // Pad to patch_size multiple
        let seq_len = x.shape().dims()[1];
        let ps = self.config.patch_size;
        let pad_length = if seq_len % ps != 0 { ps - (seq_len % ps) } else { 0 };
        let x = if pad_length > 0 {
            let pad_shape = Shape::from_dims(&[b, pad_length, x.shape().dims()[2]]);
            let pad = Tensor::zeros_dtype(pad_shape, DType::BF16, device.clone())?;
            Tensor::cat(&[&x, &pad], 1)?
        } else {
            x
        };

        // --- proj_in: Conv1d(in_channels+acoustic_dim, hidden, kernel=2, stride=2) ---
        let x = conv1d_forward(
            &x,
            self.w("decoder.proj_in.1.weight")?,
            self.w("decoder.proj_in.1.bias")?,
            self.config.patch_size,
        )?;

        // --- condition_embedder: Linear + bias ---
        let encoder_hs = linear3d(
            encoder_hidden_states,
            self.w("decoder.condition_embedder.weight")?,
        )?;
        let cond_bias = self.w("decoder.condition_embedder.bias")?
            .reshape(&[1, 1, h])?
            .broadcast_to(encoder_hs.shape())?;
        let encoder_hs = encoder_hs.add(&cond_bias)?;

        // --- RoPE position embeddings ---
        let patched_seq_len = x.shape().dims()[1];
        let (cos, sin) = self.compute_rope(patched_seq_len, device.clone())?;

        // --- Populate cross-attention KV cache on first step ---
        if self.cross_kv_cache.is_none() {
            let mut cache = Vec::with_capacity(self.config.num_layers);
            let s_kv = encoder_hs.shape().dims()[1];
            let nkv = self.config.num_kv_heads;
            let hd = self.config.head_dim;
            for layer_idx in 0..self.config.num_layers {
                let prefix = format!("decoder.layers.{layer_idx}.cross_attn");
                let k_raw = linear3d(&encoder_hs, self.w(&format!("{prefix}.k_proj.weight"))?)?;
                let v_raw = linear3d(&encoder_hs, self.w(&format!("{prefix}.v_proj.weight"))?)?;
                let k_raw = k_raw.reshape(&[b, s_kv, nkv, hd])?;
                let v_raw = v_raw.reshape(&[b, s_kv, nkv, hd])?;
                let k_normed = rms_norm_per_head(
                    &k_raw,
                    self.w(&format!("{prefix}.k_norm.weight"))?,
                    self.config.rms_norm_eps,
                )?;
                let k_out = k_normed.transpose_dims(1, 2)?;
                let v_out = v_raw.transpose_dims(1, 2)?;
                cache.push((k_out, v_out));
            }
            self.cross_kv_cache = Some(cache);
        }

        // --- Process through transformer layers ---
        let mut x = x;
        for layer_idx in 0..self.config.num_layers {
            x = self.dit_layer_forward(
                &x,
                &timestep_proj,
                &encoder_hs,
                &cos,
                &sin,
                layer_idx,
            )?;
        }

        // --- Output: AdaLN + proj_out ---
        // shift, scale = (scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        let out_sst = self.w("decoder.scale_shift_table")?; // [1, 2, hidden]
        let temb_unsqueeze = temb.reshape(&[b, 1, h])?;
        let sst_plus_temb = out_sst
            .broadcast_to(&Shape::from_dims(&[b, 2, h]))?
            .add(&temb_unsqueeze)?;
        let shift = sst_plus_temb.narrow(1, 0, 1)?;
        let scale = sst_plus_temb.narrow(1, 1, 1)?;

        // norm_out -> adaptive: norm(x) * (1 + scale) + shift
        let x_normed = rms_norm(&x, self.w("decoder.norm_out.weight")?, self.config.rms_norm_eps)?;
        let ones = Tensor::ones_dtype(scale.shape().clone(), DType::BF16, device.clone())?;
        let x = x_normed.mul(&ones.add(&scale)?)?.add(&shift)?;

        // proj_out: ConvTranspose1d(hidden -> acoustic_dim, kernel=2, stride=2)
        let x = conv_transpose1d_forward(
            &x,
            self.w("decoder.proj_out.1.weight")?,
            self.w("decoder.proj_out.1.bias")?,
            self.config.patch_size,
        )?;

        // Crop to original sequence length
        let x = if x.shape().dims()[1] > original_seq_len {
            x.narrow(1, 0, original_seq_len)?
        } else {
            x
        };

        Ok(x)
    }
}

// ===========================================================================
// Helper functions
// ===========================================================================

/// 3D linear: [B, S, in] @ W_transposed[in, out] -> [B, S, out]
/// Expects W already transposed (from weight loading).
fn linear3d(input: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    input.matmul(weight_t)
}

/// 3D linear with bias.
fn linear3d_bias(input: &Tensor, weight_t: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let out = input.matmul(weight_t)?;
    let bias_expanded = bias
        .reshape(&[1, 1, bias.shape().elem_count()])?
        .broadcast_to(out.shape())?;
    out.add(&bias_expanded)
}

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.mul(&x_f32)?;
    let last_dim = sq.shape().dims().len() - 1;
    let mean_sq = sq.mean_dim(&[last_dim], true)?;
    let rsqrt = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = x_f32.mul(&rsqrt)?;
    let w = weight
        .to_dtype(DType::F32)?
        .reshape(&[1, 1, weight.shape().elem_count()])?;
    normed.mul(&w)?.to_dtype(DType::BF16)
}

/// RMSNorm per head: x is [B, S, heads, hd], weight is [hd]
fn rms_norm_per_head(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, heads, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * s * heads, hd])?;
    let flat_f32 = flat.to_dtype(DType::F32)?;
    let sq = flat_f32.mul(&flat_f32)?;
    let last_dim = sq.shape().dims().len() - 1;
    let mean_sq = sq.mean_dim(&[last_dim], true)?;
    let rsqrt = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = flat_f32.mul(&rsqrt)?;
    let w = weight
        .to_dtype(DType::F32)?
        .reshape(&[1, weight.shape().elem_count()])?;
    let result = normed.mul(&w)?.to_dtype(DType::BF16)?;
    result.reshape(&[b, s, heads, hd])
}

/// Apply RoPE to tensor [B, heads, S, hd].
/// cos/sin: [S, hd/2].
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (_b, _heads, s, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let half = hd / 2;

    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;

    // cos/sin: [S, half] -> broadcast to [1, 1, S, half]
    let cos_b = cos.narrow(0, 0, s)?.reshape(&[1, 1, s, half])?;
    let sin_b = sin.narrow(0, 0, s)?.reshape(&[1, 1, s, half])?;

    let r1 = x1.mul(&cos_b)?.sub(&x2.mul(&sin_b)?)?;
    let r2 = x1.mul(&sin_b)?.add(&x2.mul(&cos_b)?)?;

    Tensor::cat(&[&r1, &r2], 3)
}

/// Repeat KV heads for GQA: [B, kv_heads, S, hd] -> [B, q_heads, S, hd]
fn repeat_kv(kv: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv.clone());
    }
    let dims = kv.shape().dims().to_vec();
    let (b, kv_heads, s, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let expanded = kv
        .reshape(&[b, kv_heads, 1, s, hd])?
        .broadcast_to(&Shape::from_dims(&[b, kv_heads, n_rep, s, hd]))?;
    expanded.reshape(&[b, kv_heads * n_rep, s, hd])
}

/// Conv1d forward as reshape+matmul (kernel_size == stride).
/// Weight: [C_out, C_in, kernel_size], Bias: [C_out]
fn conv1d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let (b, t, c_in) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.shape().dims().to_vec();
    let (c_out, _c_in_k, k) = (w_dims[0], w_dims[1], w_dims[2]);
    debug_assert_eq!(stride, k, "conv1d_forward requires stride == kernel_size");

    let t_out = t / k;
    let x = input.reshape(&[b, t_out, k * c_in])?;

    // Input reshape gives [B, T/K, K*C_in] with ordering: k*C_in + c_in
    // PyTorch Conv1d weight: [C_out, C_in, K] -> permute to [C_out, K, C_in] -> reshape -> [C_out, K*C_in]
    // This aligns weight indexing with the input's k-major ordering.
    let w_perm = weight.transpose_dims(1, 2)?; // [C_out, K, C_in]
    let w_2d = w_perm.reshape(&[c_out, k * c_in])?.transpose()?; // [K*C_in, C_out]
    let out = x.matmul(&w_2d)?;

    let bias_expanded = bias
        .reshape(&[1, 1, c_out])?
        .broadcast_to(out.shape())?;
    out.add(&bias_expanded)
}

/// ConvTranspose1d forward as matmul+reshape (kernel_size == stride).
/// Weight: [C_in, C_out, kernel_size], Bias: [C_out]
fn conv_transpose1d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let (b, t, c_in) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.shape().dims().to_vec();
    let (_c_in_w, c_out, k) = (w_dims[0], w_dims[1], w_dims[2]);
    debug_assert_eq!(stride, k, "conv_transpose1d_forward requires stride == kernel_size");

    // PyTorch ConvTranspose1d weight: [C_in, C_out, K]
    // Output reshape [B, T*K, C_out] expects ordering: k*C_out + c_out
    // Permute weight to [C_in, K, C_out] -> reshape to [C_in, K*C_out]
    let w_perm = weight.transpose_dims(1, 2)?; // [C_in, K, C_out]
    let w_2d = w_perm.reshape(&[c_in, k * c_out])?;
    let out = input.matmul(&w_2d)?; // [B, T, K*C_out]
    let out = out.reshape(&[b, t * k, c_out])?;

    let bias_expanded = bias
        .reshape(&[1, 1, c_out])?
        .broadcast_to(out.shape())?;
    out.add(&bias_expanded)
}
