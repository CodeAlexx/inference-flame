//! ZImage NextDiT model — Lumina2/NextDiT with joint attention, 3D RoPE, SwiGLU FFN.
//!
//! Architecture:
//! - 30 main layers, 2 noise refiners, 2 context refiners
//! - dim=3840, 30 heads, head_dim=128
//! - Qwen3 4B text (cap_feat_dim=2560)
//! - z_image modulation: per-layer adaLN with tanh gates, min_mod=256
//! - Patchify 2x2 -> Linear(64, 3840), NOT Conv2d
//! - Model returns negated velocity: -img

use flame_core::attention::sdpa;
use flame_core::layer_norm::layer_norm;
use flame_core::norm::RMSNorm;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::offload::BlockLoader;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct NextDiTConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub num_noise_refiner: usize,
    pub num_context_refiner: usize,
    pub cap_feat_dim: usize,
    pub mlp_hidden: usize,
    pub min_mod: usize,
    pub t_embedder_hidden: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
    pub time_scale: f32,
    pub pad_tokens_multiple: usize,
}

impl Default for NextDiTConfig {
    fn default() -> Self {
        Self {
            dim: 3840,
            num_heads: 30,
            head_dim: 128,
            num_layers: 30,
            num_noise_refiner: 2,
            num_context_refiner: 2,
            cap_feat_dim: 2560,
            mlp_hidden: 10240,
            min_mod: 256,
            t_embedder_hidden: 1024,
            patch_size: 2,
            in_channels: 16,
            axes_dims_rope: [32, 48, 48],
            rope_theta: 256.0,
            time_scale: 1000.0,
            pad_tokens_multiple: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight-backed model -- stores resident weights + streams blocks from disk
// ---------------------------------------------------------------------------

pub struct NextDiT {
    pub config: NextDiTConfig,
    /// Small weights that stay on GPU permanently (embedders, final layer, pad tokens)
    resident: HashMap<String, Tensor>,
    /// Block loader for on-demand weight streaming via mmap
    loader: BlockLoader,
    device: Arc<cudarc::driver::CudaDevice>,
}

impl NextDiT {
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let loader = BlockLoader::new(model_path, device.clone());
        Self {
            config: NextDiTConfig::default(),
            resident,
            loader,
            device,
        }
    }

    /// Load a block's weights from disk (mmap) into GPU, replacing any previous block.
    pub fn load_block(&mut self, prefix: &str) -> Result<()> {
        self.loader.load_block(prefix)
    }

    /// Drop current block weights to free VRAM.
    pub fn unload_block(&mut self) {
        self.loader.unload_block();
    }

    /// Get a weight tensor by key -- checks block_cache first, then resident.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.loader.get(key, &self.resident)
    }

    // -- Linear helpers (matmul + optional bias) -----------------------------

    /// x @ weight.T  (weight shape: [out, in], no bias)
    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let x_dims = x.shape().dims().to_vec();
        let in_features = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let out_features = weight.shape().dims()[0];

        let x_2d = x.reshape(&[batch, in_features])?;
        let wt = transpose_2d(weight)?;
        let out_2d = x_2d.matmul(&wt)?;

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        out_2d.reshape(&out_shape)
    }

    /// x @ weight.T + bias
    fn linear_with_bias(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let out = self.linear_no_bias(x, weight_key)?;
        let bias = self.w(bias_key)?;
        let out_dims = out.shape().dims().to_vec();
        let batch: usize = out_dims[..out_dims.len() - 1].iter().product();
        let out_feat = *out_dims.last().unwrap();
        let bias_1d = bias.reshape(&[1, out_feat])?;
        let out_2d = out.reshape(&[batch, out_feat])?;
        let result_2d = out_2d.add(&bias_1d)?;
        result_2d.reshape(&out_dims)
    }

    // -- RMSNorm (functional, using weight from HashMap) ---------------------

    fn rms_norm(&self, x: &Tensor, weight_key: &str, eps: f32) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let norm_dim = weight.shape().dims()[0];
        let mut norm = RMSNorm::new(vec![norm_dim], eps, true, self.device.clone())?;
        norm.copy_weight_from(weight)?;
        norm.forward(x)
    }

    // -- SwiGLU FFN ----------------------------------------------------------

    fn swiglu(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        // w2(silu(w1(x)) * w3(x))
        let w1_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w1.weight"))?;
        let w3_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w3.weight"))?;
        let gate = w1_out.silu()?;
        let hidden = gate.mul(&w3_out)?;
        self.linear_no_bias(&hidden, &format!("{prefix}.feed_forward.w2.weight"))
    }

    // -- Attention -----------------------------------------------------------

    fn joint_attention(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        prefix: &str,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Fused QKV projection (no bias)
        let qkv = self.linear_no_bias(x, &format!("{prefix}.attention.qkv.weight"))?;
        let chunks = qkv.chunk(3, 2)?; // split on last dim
        let q = chunks[0].reshape(&[b, seq, num_heads, head_dim])?;
        let k = chunks[1].reshape(&[b, seq, num_heads, head_dim])?;
        let v = chunks[2].reshape(&[b, seq, num_heads, head_dim])?;

        // QK RMSNorm (per-head, applied on head_dim dimension)
        let q = self.rms_norm_per_head(&q, &format!("{prefix}.attention.q_norm.weight"))?;
        let k = self.rms_norm_per_head(&k, &format!("{prefix}.attention.k_norm.weight"))?;

        // Apply 3D RoPE
        let q = apply_rope_real(&q, rope_cos, rope_sin)?;
        let k = apply_rope_real(&k, rope_cos, rope_sin)?;

        // Transpose to [B, H, S, D] for SDPA
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let out = sdpa(&q, &k, &v, None)?;

        // Back to [B, S, H*D]
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq, num_heads * head_dim])?;

        // Output projection
        self.linear_no_bias(&out, &format!("{prefix}.attention.out.weight"))
    }

    /// Apply RMSNorm per head: input [B, S, H, D], norm weight [D]
    fn rms_norm_per_head(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);

        // Flatten to [..., D] for RMSNorm
        let flat = x.reshape(&[b * s * h, d])?;
        let mut norm = RMSNorm::new(vec![d], 1e-6, true, self.device.clone())?;
        norm.copy_weight_from(weight)?;
        let normed = norm.forward(&flat)?;
        normed.reshape(&[b, s, h, d])
    }

    // -- Transformer block ---------------------------------------------------

    fn transformer_block(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        t_cond: Option<&Tensor>,
        prefix: &str,
    ) -> Result<Tensor> {
        let has_adaln = t_cond.is_some()
            && self
                .loader
                .cache_contains(&format!("{prefix}.adaLN_modulation.0.weight"));

        // Compute modulation if conditioned
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if has_adaln {
            let t_cond = t_cond.unwrap();
            let mod_out = self.linear_with_bias(
                t_cond,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &format!("{prefix}.adaLN_modulation.0.bias"),
            )?;
            let chunks = mod_out.chunk(4, mod_out.shape().dims().len() - 1)?;
            (
                Some(chunks[0].clone()),
                Some(chunks[1].clone()),
                Some(chunks[2].clone()),
                Some(chunks[3].clone()),
            )
        } else {
            (None, None, None, None)
        };

        // Attention: pre-norm -> modulate -> attention -> post-norm -> gate + residual
        let mut x_norm =
            self.rms_norm(x, &format!("{prefix}.attention_norm1.weight"), 1e-6)?;

        if let Some(ref scale) = scale_msa {
            let ones = Tensor::from_vec_dtype(
                vec![1.0f32],
                Shape::from_dims(&[1, 1, 1]),
                self.device.clone(),
                DType::BF16,
            )?;
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = ones.add(&scale_unsq)?;
            x_norm = x_norm.mul(&factor)?;
        }

        let attn_out = self.joint_attention(&x_norm, rope_cos, rope_sin, prefix)?;
        let attn_out =
            self.rms_norm(&attn_out, &format!("{prefix}.attention_norm2.weight"), 1e-6)?;

        let mut x_out = if let Some(ref gate) = gate_msa {
            let g = gate.tanh()?.unsqueeze(1)?;
            let gated = g.mul(&attn_out)?;
            x.add(&gated)?
        } else {
            x.add(&attn_out)?
        };

        // FFN: pre-norm -> modulate -> FFN -> post-norm -> gate + residual
        let mut ff_norm =
            self.rms_norm(&x_out, &format!("{prefix}.ffn_norm1.weight"), 1e-6)?;

        if let Some(ref scale) = scale_mlp {
            let ones = Tensor::from_vec_dtype(
                vec![1.0f32],
                Shape::from_dims(&[1, 1, 1]),
                self.device.clone(),
                DType::BF16,
            )?;
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = ones.add(&scale_unsq)?;
            ff_norm = ff_norm.mul(&factor)?;
        }

        let ff_out = self.swiglu(&ff_norm, prefix)?;
        let ff_out =
            self.rms_norm(&ff_out, &format!("{prefix}.ffn_norm2.weight"), 1e-6)?;

        x_out = if let Some(ref gate) = gate_mlp {
            let g = gate.tanh()?.unsqueeze(1)?;
            let gated = g.mul(&ff_out)?;
            x_out.add(&gated)?
        } else {
            x_out.add(&ff_out)?
        };

        Ok(x_out)
    }

    // -- Timestep embedder ---------------------------------------------------

    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        let freq_dim = self.config.min_mod; // 256
        let half = freq_dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_vec()?;
        let batch = t_data.len();

        let mut emb_data = vec![0.0f32; batch * freq_dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                emb_data[b * freq_dim + i] = angle.cos();
                emb_data[b * freq_dim + half + i] = angle.sin();
            }
        }

        let emb = Tensor::from_vec_dtype(
            emb_data,
            Shape::from_dims(&[batch, freq_dim]),
            self.device.clone(),
            DType::BF16,
        )?;

        // MLP: Linear(256, 1024) -> SiLU -> Linear(1024, 256)
        let h =
            self.linear_with_bias(&emb, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    // -- Caption embedder ----------------------------------------------------

    fn caption_embed(&self, cap_feats: &Tensor) -> Result<Tensor> {
        // RMSNorm(2560) -> Linear(2560, 3840)
        let normed = self.rms_norm(cap_feats, "cap_embedder.0.weight", 1e-6)?;
        if self.resident.contains_key("cap_embedder.1.bias") {
            self.linear_with_bias(&normed, "cap_embedder.1.weight", "cap_embedder.1.bias")
        } else {
            self.linear_no_bias(&normed, "cap_embedder.1.weight")
        }
    }

    // -- Patchify / Unpatchify -----------------------------------------------

    fn patchify(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let p = self.config.patch_size;
        let ph = h / p;
        let pw = w / p;

        let x = x.reshape(&[b, c, ph, p, pw, p])?;
        let x = x.permute(&[0, 2, 4, 3, 5, 1])?;
        let x = x.reshape(&[b, ph * pw, p * p * c])?;
        Ok((x, ph, pw))
    }

    fn unpatchify(&self, x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let p = self.config.patch_size;
        let c = self.config.in_channels;

        let x = x.reshape(&[b, ph, pw, p, p, c])?;
        let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
        x.reshape(&[b, c, ph * p, pw * p])
    }

    // -- Pad tokens ----------------------------------------------------------

    fn pad_to_multiple(
        &self,
        tokens: &Tensor,
        pad_token_key: &str,
        multiple: usize,
    ) -> Result<(Tensor, usize)> {
        let seq_len = tokens.shape().dims()[1];
        let pad_len = (multiple - (seq_len % multiple)) % multiple;
        if pad_len == 0 {
            return Ok((tokens.clone(), 0));
        }

        let b = tokens.shape().dims()[0];
        let dim = tokens.shape().dims()[2];
        let pad_token = self.w(pad_token_key)?; // shape [1, dim]

        let pad_single = pad_token.reshape(&[1, 1, dim])?;
        let mut pad_parts = Vec::new();
        for _ in 0..pad_len {
            pad_parts.push(&pad_single);
        }
        let pad_seq = if pad_len == 1 {
            pad_single.clone()
        } else {
            let refs: Vec<&Tensor> = pad_parts.into_iter().collect();
            Tensor::cat(&refs, 1)?
        };

        let mut batch_parts: Vec<Tensor> = Vec::new();
        for _ in 0..b {
            batch_parts.push(pad_seq.clone());
        }
        let batch_refs: Vec<&Tensor> = batch_parts.iter().collect();
        let pad_batch = if b == 1 {
            pad_seq
        } else {
            Tensor::cat(&batch_refs, 0)?
        };

        let result = Tensor::cat(&[tokens, &pad_batch], 1)?;
        Ok((result, pad_len))
    }

    // -- Full forward pass ---------------------------------------------------

    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
    ) -> Result<Tensor> {
        let pad_mult = self.config.pad_tokens_multiple;

        // Invert timestep and scale
        let t_data = timestep.to_vec()?;
        let inv_data: Vec<f32> = t_data
            .iter()
            .map(|v| (1.0 - v) * self.config.time_scale)
            .collect();
        let t_scaled = Tensor::from_vec_dtype(
            inv_data,
            timestep.shape().clone(),
            self.device.clone(),
            DType::BF16,
        )?;
        let t_cond = self.timestep_embed(&t_scaled)?;

        // Patchify and embed image
        let (x_patches, ph, pw) = self.patchify(x)?;
        let x_emb = if self.resident.contains_key("x_embedder.bias") {
            self.linear_with_bias(&x_patches, "x_embedder.weight", "x_embedder.bias")?
        } else {
            self.linear_no_bias(&x_patches, "x_embedder.weight")?
        };
        let img_len = x_emb.shape().dims()[1];

        // Embed captions
        let c = self.caption_embed(cap_feats)?;

        // Pad caption to multiple of pad_tokens_multiple
        let (c, _cap_pad_len) = self.pad_to_multiple(&c, "cap_pad_token", pad_mult)?;
        let cap_len = c.shape().dims()[1];

        // Pad image to multiple of pad_tokens_multiple
        let (x_emb, img_pad_len) = self.pad_to_multiple(&x_emb, "x_pad_token", pad_mult)?;

        // Build position IDs and RoPE
        let (rope_cos_full, rope_sin_full) =
            self.build_3d_rope(cap_len, ph, pw, img_pad_len)?;

        // Split RoPE for caption and image portions
        let rope_cos_cap = rope_cos_full.narrow(0, 0, cap_len)?;
        let rope_sin_cap = rope_sin_full.narrow(0, 0, cap_len)?;
        let rope_cos_img = rope_cos_full.narrow(0, cap_len, x_emb.shape().dims()[1])?;
        let rope_sin_img = rope_sin_full.narrow(0, cap_len, x_emb.shape().dims()[1])?;

        // Context refiner: text self-attention (unconditioned)
        let mut c = c;
        for i in 0..self.config.num_context_refiner {
            let prefix = format!("context_refiner.{i}");
            self.load_block(&prefix)?;
            c = self.transformer_block(&c, &rope_cos_cap, &rope_sin_cap, None, &prefix)?;
            self.unload_block();
        }

        // Noise refiner: image-only self-attention (conditioned)
        let mut x_emb = x_emb;
        for i in 0..self.config.num_noise_refiner {
            let prefix = format!("noise_refiner.{i}");
            self.load_block(&prefix)?;
            x_emb = self.transformer_block(
                &x_emb,
                &rope_cos_img,
                &rope_sin_img,
                Some(&t_cond),
                &prefix,
            )?;
            self.unload_block();
        }

        // Concatenate text + image for main layers
        let mut xc = Tensor::cat(&[&c, &x_emb], 1)?;

        // Main transformer layers -- stream each block from disk
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{i}");
            println!("  Layer {i}/{}", self.config.num_layers);
            self.load_block(&prefix)?;
            xc = self.transformer_block(
                &xc,
                &rope_cos_full,
                &rope_sin_full,
                Some(&t_cond),
                &prefix,
            )?;
            self.unload_block();
        }

        // Extract image tokens (skip text, remove padding)
        let x_out = xc.narrow(1, cap_len, img_len)?;

        // Final layer: LayerNorm(no affine) -> adaLN scale -> Linear
        let x_final = self.final_layer(&x_out, &t_cond)?;

        // Unpatchify
        let x_spatial = self.unpatchify(&x_final, ph, pw)?;

        // Negate (ZImage convention: return negated velocity)
        x_spatial.mul_scalar(-1.0)
    }

    // -- Final layer ---------------------------------------------------------

    fn final_layer(&self, x: &Tensor, t_cond: &Tensor) -> Result<Tensor> {
        let dim = self.config.dim;

        // LayerNorm without affine
        let x_norm = layer_norm(x, &[dim], None, None, 1e-6)?;

        // adaLN modulation: SiLU -> Linear
        let t_silu = t_cond.silu()?;
        let scale = self.linear_with_bias(
            &t_silu,
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
        )?;
        let scale_unsq = scale.unsqueeze(1)?;

        // x = x_norm * (1 + scale)
        let ones = Tensor::from_vec_dtype(
            vec![1.0f32],
            Shape::from_dims(&[1, 1, 1]),
            self.device.clone(),
            DType::BF16,
        )?;
        let factor = ones.add(&scale_unsq)?;
        let x_modulated = x_norm.mul(&factor)?;

        // Final linear projection
        self.linear_with_bias(
            &x_modulated,
            "final_layer.linear.weight",
            "final_layer.linear.bias",
        )
    }

    // -- 3D RoPE -------------------------------------------------------------

    fn build_3d_rope(
        &self,
        cap_len: usize,
        ph: usize,
        pw: usize,
        img_pad_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let axes_dims = self.config.axes_dims_rope;
        let theta = self.config.rope_theta;
        let img_seq = ph * pw + img_pad_len;
        let total_seq = cap_len + img_seq;
        let half_head_dim = self.config.head_dim / 2; // 64

        // Build position IDs: (total_seq, 3) for [t, h, w]
        let mut pos_ids = vec![[0.0f32; 3]; total_seq];

        // Caption positions
        for i in 0..cap_len {
            pos_ids[i] = [(i + 1) as f32, 0.0, 0.0];
        }

        // Image positions
        for ih in 0..ph {
            for iw in 0..pw {
                let idx = cap_len + ih * pw + iw;
                pos_ids[idx] = [(cap_len + 1) as f32, ih as f32, iw as f32];
            }
        }
        // Padding positions remain [0, 0, 0]

        // Build cos/sin for each axis, concatenate
        let mut cos_data = vec![0.0f32; total_seq * half_head_dim];
        let mut sin_data = vec![0.0f32; total_seq * half_head_dim];

        let mut offset = 0;
        for (axis_idx, &axis_dim) in axes_dims.iter().enumerate() {
            let half_axis = axis_dim / 2;

            let mut freqs = vec![0.0f32; half_axis];
            for i in 0..half_axis {
                freqs[i] = 1.0 / theta.powf(i as f32 / half_axis as f32);
            }

            for seq_idx in 0..total_seq {
                let pos = pos_ids[seq_idx][axis_idx];
                for (freq_idx, &freq) in freqs.iter().enumerate() {
                    let angle = pos * freq;
                    cos_data[seq_idx * half_head_dim + offset + freq_idx] = angle.cos();
                    sin_data[seq_idx * half_head_dim + offset + freq_idx] = angle.sin();
                }
            }
            offset += half_axis;
        }

        let cos_tensor = Tensor::from_vec_dtype(
            cos_data,
            Shape::from_dims(&[total_seq, half_head_dim]),
            self.device.clone(),
            DType::BF16,
        )?;
        let sin_tensor = Tensor::from_vec_dtype(
            sin_data,
            Shape::from_dims(&[total_seq, half_head_dim]),
            self.device.clone(),
            DType::BF16,
        )?;

        Ok((cos_tensor, sin_tensor))
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Transpose a 2D tensor [M, N] -> [N, M]
fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    t.permute(&[1, 0])
}

/// Apply RoPE using real-valued even/odd interleaving.
///
/// x shape: [B, S, H, D]  (D = head_dim = 128)
/// rope_cos, rope_sin shape: [S, D/2]  (D/2 = 64)
fn apply_rope_real(
    x: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half_d = d / 2;

    // Reshape x to [..., D/2, 2] to split even/odd
    let x_pairs = x.reshape(&[b, s, h, half_d, 2])?;
    let x_even = x_pairs.narrow(4, 0, 1)?.squeeze(Some(4))?; // [B, S, H, D/2]
    let x_odd = x_pairs.narrow(4, 1, 1)?.squeeze(Some(4))?; // [B, S, H, D/2]

    // Broadcast rope: [S, D/2] -> [1, S, 1, D/2]
    let cos = rope_cos.reshape(&[1, s, 1, half_d])?;
    let sin = rope_sin.reshape(&[1, s, 1, half_d])?;

    // x_even' = x_even * cos - x_odd * sin
    let new_even = x_even.mul(&cos)?.sub(&x_odd.mul(&sin)?)?;
    // x_odd' = x_even * sin + x_odd * cos
    let new_odd = x_even.mul(&sin)?.add(&x_odd.mul(&cos)?)?;

    // Interleave back: stack on last dim then flatten
    let new_even_exp = new_even.unsqueeze(4)?; // [B, S, H, D/2, 1]
    let new_odd_exp = new_odd.unsqueeze(4)?; // [B, S, H, D/2, 1]
    let stacked = Tensor::cat(&[&new_even_exp, &new_odd_exp], 4)?; // [B, S, H, D/2, 2]
    stacked.reshape(&[b, s, h, d])
}
