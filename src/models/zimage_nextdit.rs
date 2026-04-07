//! ZImage NextDiT model — fused-kernel inference path.
//!
//! Architecture:
//! - 30 main layers, 2 noise refiners, 2 context refiners
//! - dim=3840, 30 heads, head_dim=128
//! - Qwen3 4B text (cap_feat_dim=2560)
//! - adaLN with tanh gates, min_mod=256
//! - Patchify 2x2 -> Linear(64, 3840)
//! - Model returns negated velocity: -img
//!
//! Uses flame-core fused kernels: fused_rms_norm, rope_fused_bf16,
//! swiglu_fused_bf16, gate_residual_fused_bf16.

use flame_core::attention::sdpa;
use flame_core::bf16_ops::{gate_residual_fused_bf16, rope_fused_bf16, swiglu_fused_bf16};
use flame_core::layer_norm::layer_norm;
use flame_core::ops::fused_inference::fused_rms_norm;
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
// Model
// ---------------------------------------------------------------------------

pub struct NextDiT {
    pub config: NextDiTConfig,
    resident: HashMap<String, Tensor>,
    loader: Option<BlockLoader>,
    device: Arc<cudarc::driver::CudaDevice>,
}

impl NextDiT {
    /// Block-swap mode: blocks streamed from disk via mmap.
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let loader = Some(BlockLoader::new(model_path, device.clone()));
        Self { config: NextDiTConfig::default(), resident, loader, device }
    }

    /// All-resident mode: every weight on GPU, no disk I/O.
    /// Pre-transposes all 2D weight matrices [out, in] -> [in, out] IN PLACE
    /// so matmul never has to transpose during forward.
    pub fn new_resident(
        mut weights: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let keys: Vec<String> = weights.keys()
            .filter(|k| k.ends_with(".weight"))
            .cloned()
            .collect();
        let mut transposed = 0;
        for key in &keys {
            let is_2d = weights.get(key).map(|t| t.shape().dims().len() == 2).unwrap_or(false);
            if is_2d {
                if let Some(t) = weights.remove(key) {
                    if let Ok(tt) = t.permute(&[1, 0]) {
                        weights.insert(key.clone(), tt);
                        transposed += 1;
                    } else {
                        weights.insert(key.clone(), t);
                    }
                }
            }
        }
        println!("    Pre-transposed {transposed} weight matrices");
        Self { config: NextDiTConfig::default(), resident: weights, loader: None, device }
    }

    fn load_block(&mut self, prefix: &str) -> Result<()> {
        if let Some(ref mut loader) = self.loader {
            loader.load_block(prefix)
        } else {
            Ok(())
        }
    }

    fn unload_block(&mut self) {
        if let Some(ref mut loader) = self.loader {
            loader.unload_block();
        }
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        if let Some(ref loader) = self.loader {
            loader.get(key, &self.resident)
        } else {
            self.resident.get(key).ok_or_else(|| {
                Error::InvalidInput(format!("Missing weight key: {key}"))
            })
        }
    }

    fn has_key(&self, key: &str) -> bool {
        if let Some(ref loader) = self.loader {
            loader.cache_contains(key)
        } else {
            self.resident.contains_key(key)
        }
    }

    // -- Linear helpers -------------------------------------------------------

    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let w_dims = weight.shape().dims();
        let x_dims = x.shape().dims().to_vec();
        let in_features = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();

        // In resident mode weights are pre-transposed [in, out], just matmul.
        // In block-swap mode weights are original [out, in], need transpose.
        let (wt, out_features) = if self.loader.is_none() {
            // Pre-transposed: [in, out]
            (weight, w_dims[1])
        } else {
            // Original: [out, in] -> transpose
            return {
                let out_features = w_dims[0];
                let x_2d = x.reshape(&[batch, in_features])?;
                let wt = weight.permute(&[1, 0])?;
                let out_2d = x_2d.matmul(&wt)?;
                let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
                out_shape.push(out_features);
                out_2d.reshape(&out_shape)
            };
        };

        let x_2d = x.reshape(&[batch, in_features])?;
        let out_2d = x_2d.matmul(wt)?;

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        out_2d.reshape(&out_shape)
    }

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

    // -- SwiGLU FFN (fused kernel) -------------------------------------------

    fn swiglu(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w1_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w1.weight"))?;
        let w3_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w3.weight"))?;
        // Fused silu(w1) * w3 — single kernel
        let hidden = swiglu_fused_bf16(&w1_out, &w3_out)?;
        self.linear_no_bias(&hidden, &format!("{prefix}.feed_forward.w2.weight"))
    }

    // -- Attention (fused RoPE + SDPA) ---------------------------------------

    fn joint_attention(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        prefix: &str,
    ) -> Result<Tensor> {
        // Sub-profiling: fires only on layer 0 when ZIMAGE_BLOCK_PROF=1.
        let prof = (prefix.ends_with(".0") || prefix.ends_with("layers.0"))
            && std::env::var("ZIMAGE_BLOCK_PROF").ok().as_deref() == Some("1");
        let dev = x.device().clone();
        let mark = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = dev.synchronize();
                eprintln!("[ZATTN]   {:<18} {:>6}ms", label, t.elapsed().as_millis());
            }
        };

        let dims = x.shape().dims().to_vec();
        let (b, seq, num_heads, head_dim) = (dims[0], dims[1], self.config.num_heads, self.config.head_dim);

        let t = std::time::Instant::now();
        let qkv = self.linear_no_bias(x, &format!("{prefix}.attention.qkv.weight"))?;
        let chunks = qkv.chunk(3, 2)?;
        let q = chunks[0].reshape(&[b, seq, num_heads, head_dim])?;
        let k = chunks[1].reshape(&[b, seq, num_heads, head_dim])?;
        let v = chunks[2].reshape(&[b, seq, num_heads, head_dim])?;
        mark("a.qkv_proj+chunk", t);

        let t = std::time::Instant::now();
        let q_w = self.w(&format!("{prefix}.attention.q_norm.weight"))?;
        let k_w = self.w(&format!("{prefix}.attention.k_norm.weight"))?;
        let q_flat = q.reshape(&[b * seq * num_heads, head_dim])?;
        let k_flat = k.reshape(&[b * seq * num_heads, head_dim])?;
        let q = fused_rms_norm(&q_flat, q_w, 1e-6)?.reshape(&[b, seq, num_heads, head_dim])?;
        let k = fused_rms_norm(&k_flat, k_w, 1e-6)?.reshape(&[b, seq, num_heads, head_dim])?;
        mark("b.qk_rmsnorm", t);

        let t = std::time::Instant::now();
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;
        mark("c.permute_qkv", t);

        let t = std::time::Instant::now();
        let half_d = head_dim / 2;
        let cos = rope_cos.reshape(&[1, 1, seq, half_d])?;
        let sin = rope_sin.reshape(&[1, 1, seq, half_d])?;
        let q = rope_fused_bf16(&q, &cos, &sin)?;
        let k = rope_fused_bf16(&k, &cos, &sin)?;
        mark("d.rope", t);

        let t = std::time::Instant::now();
        let out = sdpa(&q, &k, &v, None)?;
        mark("e.sdpa", t);

        let t = std::time::Instant::now();
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq, num_heads * head_dim])?;
        mark("f.permute_out", t);

        let t = std::time::Instant::now();
        let r = self.linear_no_bias(&out, &format!("{prefix}.attention.out.weight"));
        mark("g.out_proj", t);
        r
    }

    // -- Transformer block (fused kernels) -----------------------------------

    fn transformer_block(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        t_cond: Option<&Tensor>,
        prefix: &str,
    ) -> Result<Tensor> {
        // Section profiler: fires when ZIMAGE_BLOCK_PROF=1 and only on block 0.
        let prof = prefix.ends_with(".0") || prefix.ends_with("layers.0");
        let prof = prof && std::env::var("ZIMAGE_BLOCK_PROF").ok().as_deref() == Some("1");
        let dev = x.device().clone();
        let mark = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = dev.synchronize();
                eprintln!("[ZBPROF] {:<22} {:>6}ms", label, t.elapsed().as_millis());
            }
        };
        let t_block = std::time::Instant::now();

        let t = std::time::Instant::now();
        let adaln_key = format!("{prefix}.adaLN_modulation.0.weight");
        let has_adaln = t_cond.is_some() && self.has_key(&adaln_key);

        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if has_adaln {
            let t_cond = t_cond.unwrap();
            let mod_out = self.linear_with_bias(
                t_cond,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &format!("{prefix}.adaLN_modulation.0.bias"),
            )?;
            let chunks = mod_out.chunk(4, mod_out.shape().dims().len() - 1)?;
            (Some(chunks[0].clone()), Some(chunks[1].clone()),
             Some(chunks[2].clone()), Some(chunks[3].clone()))
        } else {
            (None, None, None, None)
        };
        mark("1.adaln_mod", t);

        // --- Attention branch ---
        let t = std::time::Instant::now();
        let norm1_w = self.w(&format!("{prefix}.attention_norm1.weight"))?;
        let x_norm = fused_rms_norm(x, norm1_w, 1e-6)?;
        let x_norm = if let Some(ref scale) = scale_msa {
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = scale_unsq.add_scalar(1.0)?;
            x_norm.mul(&factor)?
        } else {
            x_norm
        };
        mark("2.rms1+scale_msa", t);

        let t = std::time::Instant::now();
        let attn_out = self.joint_attention(&x_norm, rope_cos, rope_sin, prefix)?;
        mark("3.joint_attention", t);

        let t = std::time::Instant::now();
        let norm2_w = self.w(&format!("{prefix}.attention_norm2.weight"))?;
        let attn_out = fused_rms_norm(&attn_out, norm2_w, 1e-6)?;
        let x_out = if let Some(ref gate) = gate_msa {
            let g = gate.tanh()?;
            gate_residual_fused_bf16(x, &g, &attn_out)?
        } else {
            x.add(&attn_out)?
        };
        mark("4.rms2+gate1", t);

        // --- FFN branch ---
        let t = std::time::Instant::now();
        let ffn_norm1_w = self.w(&format!("{prefix}.ffn_norm1.weight"))?;
        let ff_norm = fused_rms_norm(&x_out, ffn_norm1_w, 1e-6)?;
        let ff_norm = if let Some(ref scale) = scale_mlp {
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = scale_unsq.add_scalar(1.0)?;
            ff_norm.mul(&factor)?
        } else {
            ff_norm
        };
        mark("5.rms3+scale_mlp", t);

        let t = std::time::Instant::now();
        let ff_out = self.swiglu(&ff_norm, prefix)?;
        mark("6.swiglu", t);

        let t = std::time::Instant::now();
        let ffn_norm2_w = self.w(&format!("{prefix}.ffn_norm2.weight"))?;
        let ff_out = fused_rms_norm(&ff_out, ffn_norm2_w, 1e-6)?;
        let x_out = if let Some(ref gate) = gate_mlp {
            let g = gate.tanh()?;
            gate_residual_fused_bf16(&x_out, &g, &ff_out)?
        } else {
            x_out.add(&ff_out)?
        };
        mark("7.rms4+gate2", t);

        if prof {
            let _ = dev.synchronize();
            eprintln!("[ZBPROF] TOTAL                 {:>6}ms", t_block.elapsed().as_millis());
        }

        Ok(x_out)
    }

    // -- Timestep embedder ---------------------------------------------------

    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        let freq_dim = self.config.min_mod;
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

        let h = self.linear_with_bias(&emb, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    // -- Caption embedder ----------------------------------------------------

    fn caption_embed(&self, cap_feats: &Tensor) -> Result<Tensor> {
        let norm_w = self.w("cap_embedder.0.weight")?;
        let normed = fused_rms_norm(cap_feats, norm_w, 1e-6)?;
        if self.has_key("cap_embedder.1.bias") {
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
        let (ph, pw) = (h / p, w / p);

        let x = x.reshape(&[b, c, ph, p, pw, p])?;
        let x = x.permute(&[0, 2, 4, 3, 5, 1])?;
        x.reshape(&[b, ph * pw, p * p * c]).map(|t| (t, ph, pw))
    }

    fn unpatchify(&self, x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        let b = x.shape().dims()[0];
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
        let pad_token = self.w(pad_token_key)?;

        // Build padding: [1, pad_len, dim] then expand to [B, pad_len, dim]
        let pad_single = pad_token.reshape(&[1, 1, dim])?;
        let pad_expanded = pad_single.expand(&[1, pad_len, dim])?;
        let pad_batch = if b > 1 {
            pad_expanded.expand(&[b, pad_len, dim])?
        } else {
            pad_expanded
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
        let inv_data: Vec<f32> = t_data.iter().map(|v| (1.0 - v) * self.config.time_scale).collect();
        let t_scaled = Tensor::from_vec_dtype(
            inv_data, timestep.shape().clone(), self.device.clone(), DType::BF16,
        )?;
        let t_cond = self.timestep_embed(&t_scaled)?;

        // Patchify and embed image
        let (x_patches, ph, pw) = self.patchify(x)?;
        let x_emb = if self.has_key("x_embedder.bias") {
            self.linear_with_bias(&x_patches, "x_embedder.weight", "x_embedder.bias")?
        } else {
            self.linear_no_bias(&x_patches, "x_embedder.weight")?
        };
        let img_len = x_emb.shape().dims()[1];

        // Embed captions
        let c = self.caption_embed(cap_feats)?;

        // Pad caption and image to multiple of 32
        let (c, _) = self.pad_to_multiple(&c, "cap_pad_token", pad_mult)?;
        let cap_len = c.shape().dims()[1];
        let (x_emb, img_pad_len) = self.pad_to_multiple(&x_emb, "x_pad_token", pad_mult)?;

        // Build 3D RoPE
        let (rope_cos_full, rope_sin_full) = self.build_3d_rope(cap_len, ph, pw, img_pad_len)?;

        // Split RoPE for caption and image portions
        let rope_cos_cap = rope_cos_full.narrow(0, 0, cap_len)?;
        let rope_sin_cap = rope_sin_full.narrow(0, 0, cap_len)?;
        let img_seq = x_emb.shape().dims()[1];
        let rope_cos_img = rope_cos_full.narrow(0, cap_len, img_seq)?;
        let rope_sin_img = rope_sin_full.narrow(0, cap_len, img_seq)?;

        // Context refiner: text self-attention (unconditioned)
        let mut c = c;
        for i in 0..self.config.num_context_refiner {
            let prefix = format!("context_refiner.{i}");
            self.load_block(&prefix)?;
            c = self.transformer_block(&c, &rope_cos_cap, &rope_sin_cap, None, &prefix)?;
            self.unload_block();
        }

        // Noise refiner: image self-attention (conditioned)
        let mut x_emb = x_emb;
        for i in 0..self.config.num_noise_refiner {
            let prefix = format!("noise_refiner.{i}");
            self.load_block(&prefix)?;
            x_emb = self.transformer_block(&x_emb, &rope_cos_img, &rope_sin_img, Some(&t_cond), &prefix)?;
            self.unload_block();
        }

        // Concatenate text + image for main layers
        let mut xc = Tensor::cat(&[&c, &x_emb], 1)?;

        // Main transformer layers
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{i}");
            self.load_block(&prefix)?;
            xc = self.transformer_block(&xc, &rope_cos_full, &rope_sin_full, Some(&t_cond), &prefix)?;
            self.unload_block();
        }

        // Extract image tokens (skip text, remove padding)
        let x_out = xc.narrow(1, cap_len, img_len)?;

        // Final layer
        let x_final = self.final_layer(&x_out, &t_cond)?;

        // Unpatchify + negate (ZImage convention)
        let x_spatial = self.unpatchify(&x_final, ph, pw)?;
        x_spatial.mul_scalar(-1.0)
    }

    // -- Final layer ---------------------------------------------------------

    fn final_layer(&self, x: &Tensor, t_cond: &Tensor) -> Result<Tensor> {
        let dim = self.config.dim;

        // LayerNorm without affine
        let x_norm = layer_norm(x, &[dim], None, None, 1e-6)?;

        // adaLN: SiLU -> Linear -> (1 + scale) * x_norm
        let t_silu = t_cond.silu()?;
        let scale = self.linear_with_bias(
            &t_silu,
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
        )?;
        let factor = scale.unsqueeze(1)?.add_scalar(1.0)?;
        let x_modulated = x_norm.mul(&factor)?;

        self.linear_with_bias(&x_modulated, "final_layer.linear.weight", "final_layer.linear.bias")
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
        let half_head_dim = self.config.head_dim / 2;

        let mut pos_ids = vec![[0.0f32; 3]; total_seq];

        // Caption positions: (1, 0, 0), (2, 0, 0), ...
        for i in 0..cap_len {
            pos_ids[i] = [(i + 1) as f32, 0.0, 0.0];
        }

        // Image positions: (cap_len+1, ih, iw)
        for ih in 0..ph {
            for iw in 0..pw {
                pos_ids[cap_len + ih * pw + iw] = [(cap_len + 1) as f32, ih as f32, iw as f32];
            }
        }

        // Build cos/sin — layout [total_seq, half_head_dim] with axes concatenated
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
            cos_data, Shape::from_dims(&[total_seq, half_head_dim]),
            self.device.clone(), DType::BF16,
        )?;
        let sin_tensor = Tensor::from_vec_dtype(
            sin_data, Shape::from_dims(&[total_seq, half_head_dim]),
            self.device.clone(), DType::BF16,
        )?;

        Ok((cos_tensor, sin_tensor))
    }
}
