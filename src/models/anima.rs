//! Anima (Cosmos Predict2) MiniTrainDIT + LLM Adapter inference module.
//!
//! Architecture:
//! - MiniTrainDIT backbone: 28 blocks, model_channels=2048, 16 heads, head_dim=128
//! - Patch embed: Linear(68, 2048) — 17ch (16 latent + 1 padding mask) * 2*2 patch
//! - Per-block: self-attn + cross-attn + GELU MLP(2048->8192->2048), all no bias
//! - AdaLN with LoRA: SiLU -> Linear(2048->256) -> Linear(256->6144=3*2048)
//! - QK norm: RMSNorm(128) per-head
//! - Cross-attn: Q from image (2048), K/V from text context (1024->2048)
//! - 3D RoPE: rotation-matrix format on self-attention only
//! - Timestep embed: sinusoidal(2048) -> Linear(2048->2048, bias) -> SiLU -> Linear(2048->6144, no bias)
//! - LLM Adapter: 6 blocks, dim=1024, 16 heads, head_dim=64, 1D RoPE
//! - Data layout: [B, T, H, W, D] — 5D throughout
//!
//! Weight keys prefixed with `net.`

use flame_core::attention::sdpa;
use flame_core::layer_norm::layer_norm;
use flame_core::norm::RMSNorm;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::offload::BlockLoader;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct AnimaConfig {
    /// MiniTrainDIT backbone
    pub model_channels: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_blocks: usize,
    pub mlp_hidden: usize,
    pub patch_size_hw: usize,
    pub patch_size_t: usize,
    pub in_channels: usize, // 16 latent channels
    pub adaln_lora_dim: usize,
    /// LLM Adapter
    pub adapter_dim: usize,
    pub adapter_heads: usize,
    pub adapter_head_dim: usize,
    pub adapter_blocks: usize,
    pub adapter_mlp_hidden: usize,
    pub adapter_vocab_size: usize,
    /// RoPE
    pub rope_theta: f32,
}

impl Default for AnimaConfig {
    fn default() -> Self {
        Self {
            model_channels: 2048,
            num_heads: 16,
            head_dim: 128,
            num_blocks: 28,
            mlp_hidden: 8192,
            patch_size_hw: 2,
            patch_size_t: 1,
            in_channels: 16,
            adaln_lora_dim: 256,
            adapter_dim: 1024,
            adapter_heads: 16,
            adapter_head_dim: 64,
            adapter_blocks: 6,
            adapter_mlp_hidden: 4096,
            adapter_vocab_size: 32128,
            rope_theta: 10000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight-backed model -- stores resident weights + streams blocks from disk
// ---------------------------------------------------------------------------

pub struct Anima {
    pub config: AnimaConfig,
    /// Small weights that stay on GPU permanently
    /// (x_embedder, t_embedder, t_embedding_norm, final_layer, llm_adapter)
    resident: HashMap<String, Tensor>,
    /// Block loader for on-demand weight streaming via mmap
    loader: BlockLoader,
    device: Arc<cudarc::driver::CudaDevice>,
}

impl Anima {
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let loader = BlockLoader::new(model_path, device.clone());
        Self {
            config: AnimaConfig::default(),
            resident,
            loader,
            device,
        }
    }

    /// Load a block's weights from disk (mmap) into GPU.
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

    // -- Linear helpers ------------------------------------------------------

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

    /// Apply RMSNorm per head: input [B, S, H, D], norm weight [D]
    fn rms_norm_per_head(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);

        let flat = x.reshape(&[b * s * h, d])?;
        let mut norm = RMSNorm::new(vec![d], 1e-6, true, self.device.clone())?;
        norm.copy_weight_from(weight)?;
        let normed = norm.forward(&flat)?;
        normed.reshape(&[b, s, h, d])
    }

    // ========================================================================
    // Timestep embedder
    // ========================================================================

    /// Prepare timestep conditioning.
    /// Returns (t_cond [B, 2048], base_adaln [B, 6144]):
    /// - t_cond: RMSNormed hidden state, input to per-block adaln_modulation
    /// - base_adaln: 6144-dim base modulation added to each sub-block's output
    ///
    /// Flow: sinusoidal(2048) -> Linear(2048,2048,bias) -> SiLU -> hidden(2048)
    ///       hidden -> Linear(2048,6144,no bias) -> base_adaln
    ///       hidden -> RMSNorm -> t_cond
    fn prepare_timestep(&self, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let dim = self.config.model_channels;
        let half = dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_vec()?;
        let batch = t_data.len();

        let mut emb_data = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                emb_data[b * dim + i] = angle.cos();
                emb_data[b * dim + half + i] = angle.sin();
            }
        }

        let emb = Tensor::from_vec_dtype(
            emb_data,
            Shape::from_dims(&[batch, dim]),
            self.device.clone(),
            DType::BF16,
        )?;

        // hidden = SiLU(Linear(emb))  [B, 2048] — no bias (use_adaln_lora=True)
        let hidden = self.linear_no_bias(
            &emb,
            "net.t_embedder.1.linear_1.weight",
        )?;
        let hidden = hidden.silu()?;

        // base_adaln = Linear(hidden) [B, 6144] — no bias
        let base_adaln = self.linear_no_bias(&hidden, "net.t_embedder.1.linear_2.weight")?;

        // t_cond = RMSNorm(hidden) [B, 2048] — input to per-block adaln_modulation
        let t_cond = self.rms_norm(&hidden, "net.t_embedding_norm.weight", 1e-6)?;

        Ok((t_cond, base_adaln))
    }

    // ========================================================================
    // AdaLN-LoRA modulation
    // ========================================================================

    /// Compute AdaLN modulation for a sub-block.
    /// adaln_modulation: SiLU(t_cond) @ W1 [2048->256] @ W2 [256->6144] + base_adaln
    /// -> chunk(3) -> (shift, scale, gate) each [B, 2048]
    fn adaln_modulation(
        &self,
        t_cond: &Tensor,
        base_adaln: &Tensor,
        prefix: &str, // e.g. "net.blocks.0.adaln_modulation_self_attn"
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let t_silu = t_cond.silu()?;
        let h = self.linear_no_bias(&t_silu, &format!("{prefix}.1.weight"))?; // [B, 256]
        let mod_out = self.linear_no_bias(&h, &format!("{prefix}.2.weight"))?; // [B, 6144]

        // Add base timestep modulation
        let mod_out = mod_out.add(base_adaln)?; // [B, 6144]

        // chunk(3) -> shift, scale, gate each [B, 2048]
        let dim = self.config.model_channels;
        let shift = mod_out.narrow(1, 0, dim)?;
        let scale = mod_out.narrow(1, dim, dim)?;
        let gate = mod_out.narrow(1, 2 * dim, dim)?;

        Ok((shift, scale, gate))
    }

    /// Compute final-layer AdaLN modulation (only 2 outputs: shift+scale, or scale+shift).
    /// Final layer: SiLU -> Linear(2048->256) -> Linear(256->4096=2*2048) -> chunk(2)
    fn final_adaln_modulation(
        &self,
        t_cond: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let t_silu = t_cond.silu()?;
        let h = self.linear_no_bias(&t_silu, "net.final_layer.adaln_modulation.1.weight")?;
        let mod_out = self.linear_no_bias(&h, "net.final_layer.adaln_modulation.2.weight")?;

        let dim = self.config.model_channels;
        let shift = mod_out.narrow(1, 0, dim)?;
        let scale = mod_out.narrow(1, dim, dim)?;

        Ok((shift, scale))
    }

    /// Apply adaLN: LayerNorm(no affine) -> scale * x + shift
    fn apply_adaln(
        &self,
        x: &Tensor,
        shift: &Tensor,
        scale: &Tensor,
    ) -> Result<Tensor> {
        let dim = *x.shape().dims().last().unwrap();
        // LayerNorm without affine parameters
        let x_norm = layer_norm(x, &[dim], None, None, 1e-6)?;

        // Broadcast shift/scale from [B, D] to [B, 1, D] or [B, 1, 1, D] etc.
        let x_rank = x.shape().dims().len();
        let mut s_shift = shift.clone();
        let mut s_scale = scale.clone();
        // shift/scale are [B, D], x might be [B, S, D]
        for _ in 0..(x_rank - 2) {
            s_shift = s_shift.unsqueeze(1)?;
            s_scale = s_scale.unsqueeze(1)?;
        }

        // x_out = x_norm * (1 + scale) + shift
        let ones = Tensor::from_vec_dtype(
            vec![1.0f32],
            Shape::from_dims(&[1, 1]),
            self.device.clone(),
            DType::BF16,
        )?;
        let factor = ones.add(&s_scale)?;
        let modulated = x_norm.mul(&factor)?;
        modulated.add(&s_shift)
    }

    // ========================================================================
    // Self-attention (with 3D RoPE rotation-matrix format)
    // ========================================================================

    fn self_attention(
        &self,
        x: &Tensor,
        rope_rot: &Tensor, // [S, D/2, 2, 2] rotation matrices
        prefix: &str,      // e.g. "net.blocks.0.self_attn"
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Q, K, V projections (all no bias)
        let q = self.linear_no_bias(x, &format!("{prefix}.q_proj.weight"))?;
        let k = self.linear_no_bias(x, &format!("{prefix}.k_proj.weight"))?;
        let v = self.linear_no_bias(x, &format!("{prefix}.v_proj.weight"))?;

        // Reshape to [B, S, H, D]
        let q = q.reshape(&[b, seq, num_heads, head_dim])?;
        let k = k.reshape(&[b, seq, num_heads, head_dim])?;
        let v = v.reshape(&[b, seq, num_heads, head_dim])?;

        // QK RMSNorm per-head
        let q = self.rms_norm_per_head(&q, &format!("{prefix}.q_norm.weight"))?;
        let k = self.rms_norm_per_head(&k, &format!("{prefix}.k_norm.weight"))?;

        // Apply 3D RoPE (rotation-matrix format)
        let q = apply_rope_rotation_matrix(&q, rope_rot)?;
        let k = apply_rope_rotation_matrix(&k, rope_rot)?;

        // Transpose to [B, H, S, D] for SDPA
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let out = sdpa(&q, &k, &v, None)?;

        // Back to [B, S, H*D]
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq, num_heads * head_dim])?;

        // Output projection (no bias)
        self.linear_no_bias(&out, &format!("{prefix}.output_proj.weight"))
    }

    // ========================================================================
    // Cross-attention (Q from image 2048, K/V from text 1024->2048, NO RoPE)
    // ========================================================================

    fn cross_attention(
        &self,
        x: &Tensor,       // [B, S_img, 2048]
        context: &Tensor,  // [B, S_txt, 1024]
        prefix: &str,      // e.g. "net.blocks.0.cross_attn"
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let seq_img = dims[1];
        let seq_txt = context.shape().dims()[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Q from image (2048 -> 2048)
        let q = self.linear_no_bias(x, &format!("{prefix}.q_proj.weight"))?;
        // K, V from context (1024 -> 2048)
        let k = self.linear_no_bias(context, &format!("{prefix}.k_proj.weight"))?;
        let v = self.linear_no_bias(context, &format!("{prefix}.v_proj.weight"))?;

        // Reshape to [B, S, H, D]
        let q = q.reshape(&[b, seq_img, num_heads, head_dim])?;
        let k = k.reshape(&[b, seq_txt, num_heads, head_dim])?;
        let v = v.reshape(&[b, seq_txt, num_heads, head_dim])?;

        // QK RMSNorm per-head
        let q = self.rms_norm_per_head(&q, &format!("{prefix}.q_norm.weight"))?;
        let k = self.rms_norm_per_head(&k, &format!("{prefix}.k_norm.weight"))?;

        // NO RoPE on cross-attention

        // Transpose to [B, H, S, D]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // SDPA
        let out = sdpa(&q, &k, &v, None)?;

        // Back to [B, S_img, H*D]
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq_img, num_heads * head_dim])?;

        // Output projection (no bias)
        self.linear_no_bias(&out, &format!("{prefix}.output_proj.weight"))
    }

    // ========================================================================
    // GELU MLP (no bias)
    // ========================================================================

    fn mlp(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        // layer1: Linear(2048 -> 8192, no bias) -> GELU -> layer2: Linear(8192 -> 2048, no bias)
        let h = self.linear_no_bias(x, &format!("{prefix}.layer1.weight"))?;
        let h = h.gelu()?;
        self.linear_no_bias(&h, &format!("{prefix}.layer2.weight"))
    }

    // ========================================================================
    // Transformer block (self-attn + cross-attn + MLP with AdaLN-LoRA)
    // ========================================================================

    fn transformer_block(
        &self,
        x: &Tensor,
        context: &Tensor,
        t_cond: &Tensor,
        base_adaln: &Tensor,
        rope_rot: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("net.blocks.{block_idx}");

        // --- Self-attention ---
        let (shift_sa, scale_sa, gate_sa) = self.adaln_modulation(
            t_cond,
            base_adaln,
            &format!("{prefix}.adaln_modulation_self_attn"),
        )?;
        // LayerNorm(no affine) is applied inside apply_adaln
        let x_mod = self.apply_adaln(x, &shift_sa, &scale_sa)?;
        let attn_out = self.self_attention(&x_mod, rope_rot, &format!("{prefix}.self_attn"))?;
        // Gate and residual
        let gate_sa_unsq = gate_sa.unsqueeze(1)?; // [B, 1, D]
        let x = x.add(&attn_out.mul(&gate_sa_unsq)?)?;

        // --- Cross-attention ---
        let (shift_ca, scale_ca, gate_ca) = self.adaln_modulation(
            t_cond,
            base_adaln,
            &format!("{prefix}.adaln_modulation_cross_attn"),
        )?;
        let x_mod = self.apply_adaln(&x, &shift_ca, &scale_ca)?;
        let cross_out =
            self.cross_attention(&x_mod, context, &format!("{prefix}.cross_attn"))?;
        let gate_ca_unsq = gate_ca.unsqueeze(1)?;
        let x = x.add(&cross_out.mul(&gate_ca_unsq)?)?;

        // --- MLP ---
        let (shift_mlp, scale_mlp, gate_mlp) = self.adaln_modulation(
            t_cond,
            base_adaln,
            &format!("{prefix}.adaln_modulation_mlp"),
        )?;
        let x_mod = self.apply_adaln(&x, &shift_mlp, &scale_mlp)?;
        let mlp_out = self.mlp(&x_mod, &format!("{prefix}.mlp"))?;
        let gate_mlp_unsq = gate_mlp.unsqueeze(1)?;
        let x = x.add(&mlp_out.mul(&gate_mlp_unsq)?)?;

        Ok(x)
    }

    // ========================================================================
    // Final layer
    // ========================================================================

    fn final_layer(&self, x: &Tensor, t_cond: &Tensor) -> Result<Tensor> {
        let (shift, scale) = self.final_adaln_modulation(t_cond)?;
        let x_mod = self.apply_adaln(x, &shift, &scale)?;
        // Linear(2048 -> 64, no bias) — unpatchify output
        self.linear_no_bias(&x_mod, "net.final_layer.linear.weight")
    }

    // ========================================================================
    // Patchify / Unpatchify (5D: [B, T, H, W, C])
    // ========================================================================

    /// Patchify: [B, T, H, W, C] -> [B, T*pH*pW, patch_dim]
    /// patch_dim = (C+1) * pH * pW * pT = 17 * 2 * 2 * 1 = 68
    /// The +1 is the padding mask channel (zeros).
    fn patchify(&self, x: &Tensor) -> Result<(Tensor, usize, usize, usize)> {
        let dims = x.shape().dims().to_vec();
        // x: [B, T, H, W, C] where C = in_channels (16)
        let (b, t, h, w, c) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let ph = self.config.patch_size_hw;
        let _pt = self.config.patch_size_t;
        let nh = h / ph;
        let nw = w / ph;

        // Append padding mask channel (zeros) -> [B, T, H, W, C+1]
        let mask = Tensor::zeros_dtype(
            Shape::from_dims(&[b, t, h, w, 1]),
            DType::BF16,
            self.device.clone(),
        )?;
        let x_padded = Tensor::cat(&[x, &mask], 4)?; // [B, T, H, W, 17]
        let c_pad = c + 1; // 17

        // Reshape to extract patches: [B, T, nH, pH, nW, pW, C+1]
        let x_r = x_padded.reshape(&[b, t, nh, ph, nw, ph, c_pad])?;
        // Permute to [B, T, nH, nW, pH, pW, C+1]
        let x_p = x_r.permute(&[0, 1, 2, 4, 3, 5, 6])?;
        // Flatten patches: [B, T*nH*nW, pH*pW*(C+1)] = [B, T*nH*nW, 68]
        let num_patches = t * nh * nw;
        let patch_dim = ph * ph * c_pad; // 2*2*17 = 68
        let x_flat = x_p.reshape(&[b, num_patches, patch_dim])?;

        Ok((x_flat, t, nh, nw))
    }

    /// Unpatchify: [B, T*nH*nW, out_dim] -> [B, T, H, W, C]
    /// out_dim = pH * pW * pT * C = 2*2*1*16 = 64
    fn unpatchify(
        &self,
        x: &Tensor,
        t: usize,
        nh: usize,
        nw: usize,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let ph = self.config.patch_size_hw;
        let c = self.config.in_channels;

        // [B, T*nH*nW, pH*pW*C] -> [B, T, nH, nW, pH, pW, C]
        let x_r = x.reshape(&[b, t, nh, nw, ph, ph, c])?;
        // Permute to [B, T, nH, pH, nW, pW, C]
        let x_p = x_r.permute(&[0, 1, 2, 4, 3, 5, 6])?;
        // Flatten: [B, T, H, W, C]
        x_p.reshape(&[b, t, nh * ph, nw * ph, c])
    }

    // ========================================================================
    // LLM Adapter
    // ========================================================================

    /// LLM Adapter: transforms Qwen3 hidden states via adapter blocks.
    /// Input: token_ids [B, S_txt] (T5 tokenizer vocab)
    /// llm_hidden: [B, S_txt, D_llm] from Qwen3
    /// Output: [B, S_txt, 1024]
    fn llm_adapter(
        &self,
        token_ids: &Tensor,  // [B, S_txt] integer token ids
        llm_hidden: &Tensor, // [B, S_txt, D_llm] Qwen3 hidden states
    ) -> Result<Tensor> {
        let prefix = "net.llm_adapter";
        let b = token_ids.shape().dims()[0];
        let seq_len = token_ids.shape().dims()[1];
        let dim = self.config.adapter_dim; // 1024
        let num_heads = self.config.adapter_heads; // 16
        let head_dim = self.config.adapter_head_dim; // 64

        // Embedding lookup
        let embed_weight = self.w(&format!("{prefix}.embed.weight"))?;
        let x = embedding_lookup(embed_weight, token_ids, dim, &self.device)?; // [B, S, 1024]

        // Build 1D RoPE for adapter self-attention
        let (rope_cos, rope_sin) =
            build_1d_rope(seq_len, head_dim, self.config.rope_theta, &self.device)?;

        let mut x = x;

        for j in 0..self.config.adapter_blocks {
            let bp = format!("{prefix}.blocks.{j}");

            // --- Self-attention with 1D RoPE ---
            let x_norm = self.rms_norm(&x, &format!("{bp}.norm_self_attn.weight"), 1e-6)?;

            let q = self.linear_no_bias(&x_norm, &format!("{bp}.self_attn.q_proj.weight"))?;
            let k = self.linear_no_bias(&x_norm, &format!("{bp}.self_attn.k_proj.weight"))?;
            let v = self.linear_no_bias(&x_norm, &format!("{bp}.self_attn.v_proj.weight"))?;

            let q = q.reshape(&[b, seq_len, num_heads, head_dim])?;
            let k = k.reshape(&[b, seq_len, num_heads, head_dim])?;
            let v = v.reshape(&[b, seq_len, num_heads, head_dim])?;

            let q = self.rms_norm_per_head(&q, &format!("{bp}.self_attn.q_norm.weight"))?;
            let k = self.rms_norm_per_head(&k, &format!("{bp}.self_attn.k_norm.weight"))?;

            // Apply 1D RoPE (cos/sin rotate_half)
            let q = apply_rope_cossin(&q, &rope_cos, &rope_sin)?;
            let k = apply_rope_cossin(&k, &rope_cos, &rope_sin)?;

            let q = q.permute(&[0, 2, 1, 3])?;
            let k = k.permute(&[0, 2, 1, 3])?;
            let v = v.permute(&[0, 2, 1, 3])?;

            let attn_out = sdpa(&q, &k, &v, None)?;
            let attn_out = attn_out.permute(&[0, 2, 1, 3])?;
            let attn_out = attn_out.reshape(&[b, seq_len, num_heads * head_dim])?;
            let attn_out =
                self.linear_no_bias(&attn_out, &format!("{bp}.self_attn.o_proj.weight"))?;
            x = x.add(&attn_out)?;

            // --- Cross-attention (Q from adapter, K/V from Qwen3 hidden states) ---
            let x_norm = self.rms_norm(&x, &format!("{bp}.norm_cross_attn.weight"), 1e-6)?;

            let q =
                self.linear_no_bias(&x_norm, &format!("{bp}.cross_attn.q_proj.weight"))?;
            let k =
                self.linear_no_bias(llm_hidden, &format!("{bp}.cross_attn.k_proj.weight"))?;
            let v =
                self.linear_no_bias(llm_hidden, &format!("{bp}.cross_attn.v_proj.weight"))?;

            let seq_llm = llm_hidden.shape().dims()[1];
            let q = q.reshape(&[b, seq_len, num_heads, head_dim])?;
            let k = k.reshape(&[b, seq_llm, num_heads, head_dim])?;
            let v = v.reshape(&[b, seq_llm, num_heads, head_dim])?;

            let q =
                self.rms_norm_per_head(&q, &format!("{bp}.cross_attn.q_norm.weight"))?;
            let k =
                self.rms_norm_per_head(&k, &format!("{bp}.cross_attn.k_norm.weight"))?;

            // Cross-attention RoPE: Q gets target positions, K gets context positions
            let (q_cos, q_sin) =
                build_1d_rope(seq_len, head_dim, self.config.rope_theta, &self.device)?;
            let (k_cos, k_sin) =
                build_1d_rope(seq_llm, head_dim, self.config.rope_theta, &self.device)?;
            let q = apply_rope_cossin(&q, &q_cos, &q_sin)?;
            let k = apply_rope_cossin(&k, &k_cos, &k_sin)?;

            let q = q.permute(&[0, 2, 1, 3])?;
            let k = k.permute(&[0, 2, 1, 3])?;
            let v = v.permute(&[0, 2, 1, 3])?;

            let cross_out = sdpa(&q, &k, &v, None)?;
            let cross_out = cross_out.permute(&[0, 2, 1, 3])?;
            let cross_out = cross_out.reshape(&[b, seq_len, num_heads * head_dim])?;
            let cross_out = self.linear_no_bias(
                &cross_out,
                &format!("{bp}.cross_attn.o_proj.weight"),
            )?;
            x = x.add(&cross_out)?;

            // --- MLP (with bias) ---
            let x_norm = self.rms_norm(&x, &format!("{bp}.norm_mlp.weight"), 1e-6)?;
            let h = self.linear_with_bias(
                &x_norm,
                &format!("{bp}.mlp.0.weight"),
                &format!("{bp}.mlp.0.bias"),
            )?;
            let h = h.gelu()?;
            let mlp_out = self.linear_with_bias(
                &h,
                &format!("{bp}.mlp.2.weight"),
                &format!("{bp}.mlp.2.bias"),
            )?;
            x = x.add(&mlp_out)?;
        }

        // Final out_proj + norm
        let x = self.linear_no_bias(&x, &format!("{prefix}.out_proj.weight"))?;
        let x = self.rms_norm(&x, &format!("{prefix}.norm.weight"), 1e-6)?;

        Ok(x)
    }

    // ========================================================================
    // Patch embedding
    // ========================================================================

    fn patch_embed(&self, patches: &Tensor) -> Result<Tensor> {
        // Linear(68, 2048, no bias)
        self.linear_no_bias(patches, "net.x_embedder.proj.1.weight")
    }

    // ========================================================================
    // Full forward pass
    // ========================================================================

    /// Full MiniTrainDIT forward.
    ///
    /// Arguments:
    /// - `x`: latent [B, T, H, W, C] where C=16
    /// - `timestep`: [B] float timestep values
    /// - `token_ids`: [B, S_txt] integer token IDs for LLM adapter embedding
    /// - `llm_hidden`: [B, S_txt, D_llm] Qwen3 hidden states for LLM adapter cross-attn
    ///
    /// Returns: predicted noise/velocity [B, T, H, W, C]
    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        token_ids: &Tensor,
        llm_hidden: &Tensor,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let t_frames = x_dims[1];

        // 1. Prepare timestep conditioning
        let (t_cond, base_adaln) = self.prepare_timestep(timestep)?;

        // 2. Run LLM adapter to get text context [B, S_txt, 1024]
        let context = self.llm_adapter(token_ids, llm_hidden)?;

        // 3. Patchify: [B, T, H, W, 16] -> [B, N_patches, 68]
        let (patches, _t, nh, nw) = self.patchify(x)?;

        // 4. Patch embed: [B, N_patches, 68] -> [B, N_patches, 2048]
        let x_emb = self.patch_embed(&patches)?;

        // 5. Build 3D RoPE rotation matrices
        let rope_rot = build_3d_rope_rotation_matrix(
            t_frames,
            nh,
            nw,
            self.config.head_dim,
            &self.device,
        )?;

        // 6. Stream through 28 transformer blocks
        let mut x_hidden = x_emb;
        for i in 0..self.config.num_blocks {
            let prefix = format!("net.blocks.{i}");
            println!("  Block {i}/{}", self.config.num_blocks);
            self.load_block(&prefix)?;
            x_hidden = self.transformer_block(
                &x_hidden,
                &context,
                &t_cond,
                &base_adaln,
                &rope_rot,
                i,
            )?;
            self.unload_block();
        }

        // 7. Final layer
        let x_out = self.final_layer(&x_hidden, &t_cond)?;

        // 8. Unpatchify: [B, N_patches, 64] -> [B, T, H, W, 16]
        self.unpatchify(&x_out, t_frames, nh, nw)
    }
}

// ===========================================================================
// Standalone helpers
// ===========================================================================

/// Transpose a 2D tensor [M, N] -> [N, M]
fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    t.permute(&[1, 0])
}

/// Embedding lookup: select rows from weight matrix by integer indices.
/// weight: [vocab_size, dim], indices: [B, S] (stored as f32 but representing ints)
/// Returns: [B, S, dim]
fn embedding_lookup(
    weight: &Tensor,
    indices: &Tensor,
    dim: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let idx_dims = indices.shape().dims().to_vec();
    let b = idx_dims[0];
    let s = idx_dims[1];

    // Read indices as f32 and convert to usize
    let idx_flat = indices.reshape(&[b * s])?;
    let idx_data = idx_flat.to_vec()?;

    // Read weight matrix
    let vocab = weight.shape().dims()[0];
    let weight_flat = weight.reshape(&[vocab * dim])?;
    let weight_data = weight_flat.to_vec()?;

    // Gather
    let mut out_data = vec![0.0f32; b * s * dim];
    for i in 0..(b * s) {
        let idx = idx_data[i] as usize;
        if idx < vocab {
            let src_offset = idx * dim;
            let dst_offset = i * dim;
            out_data[dst_offset..dst_offset + dim]
                .copy_from_slice(&weight_data[src_offset..src_offset + dim]);
        }
    }

    Tensor::from_vec_dtype(
        out_data,
        Shape::from_dims(&[b, s, dim]),
        device.clone(),
        DType::BF16,
    )
}

/// Build 1D RoPE cos/sin for the LLM adapter.
/// Returns (cos, sin) each [S, D/2] for standard rotate_half application.
fn build_1d_rope(
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half_d = head_dim / 2;

    let mut cos_data = vec![0.0f32; seq_len * half_d];
    let mut sin_data = vec![0.0f32; seq_len * half_d];

    for pos in 0..seq_len {
        for i in 0..half_d {
            let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
            let angle = (pos as f32) * freq;
            cos_data[pos * half_d + i] = angle.cos();
            sin_data[pos * half_d + i] = angle.sin();
        }
    }

    let cos = Tensor::from_vec_dtype(
        cos_data,
        Shape::from_dims(&[seq_len, half_d]),
        device.clone(),
        DType::BF16,
    )?;
    let sin = Tensor::from_vec_dtype(
        sin_data,
        Shape::from_dims(&[seq_len, half_d]),
        device.clone(),
        DType::BF16,
    )?;

    Ok((cos, sin))
}

/// Apply 1D RoPE using standard cos/sin rotate_half.
/// x: [B, S, H, D], rope_cos/rope_sin: [S, D/2]
fn apply_rope_cossin(
    x: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half_d = d / 2;

    // Split into first half and second half
    let x1 = x.narrow(3, 0, half_d)?; // [B, S, H, D/2]
    let x2 = x.narrow(3, half_d, half_d)?; // [B, S, H, D/2]

    // Broadcast rope: [S, D/2] -> [1, S, 1, D/2]
    let cos = rope_cos.reshape(&[1, s, 1, half_d])?;
    let sin = rope_sin.reshape(&[1, s, 1, half_d])?;

    // rotate_half: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    let new_x1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
    let new_x2 = x1.mul(&sin)?.add(&x2.mul(&cos)?)?;

    // Concatenate back along last dim
    let new_x1_exp = new_x1.unsqueeze(4)?; // [B, S, H, D/2, 1]
    let new_x2_exp = new_x2.unsqueeze(4)?;
    let stacked = Tensor::cat(&[&new_x1_exp, &new_x2_exp], 4)?; // [B, S, H, D/2, 2]
    stacked.reshape(&[b, s, h, d])
}

/// Build 3D RoPE rotation matrices for MiniTrainDIT self-attention.
///
/// For each position (t, h, w) we compute a [D/2, 2, 2] rotation matrix.
/// The D/2=64 frequency bins are split across the 3 axes proportionally.
/// Returns: [T*nH*nW, D/2, 2, 2]
fn build_3d_rope_rotation_matrix(
    t_frames: usize,
    nh: usize,
    nw: usize,
    head_dim: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let half_d = head_dim / 2; // 64
    let total_seq = t_frames * nh * nw;

    // Cosmos 3D RoPE dimension split (matches position_embedding.py:81-83):
    //   dim = head_dim = 128 (full, not half)
    //   dim_h = dim // 6 * 2 = 42
    //   dim_w = dim_h = 42
    //   dim_t = dim - 2*dim_h = 44
    // Each axis uses dim_x/2 frequency bins:
    //   t: 22 bins, h: 21 bins, w: 21 bins → total 64 = half_d ✓
    let full_d = half_d * 2; // 128
    let dim_h: usize = full_d / 6 * 2; // 42
    let dim_w: usize = dim_h;           // 42
    let dim_t: usize = full_d - 2 * dim_h; // 44
    let bins_t = dim_t / 2; // 22
    let bins_h = dim_h / 2; // 21
    let bins_w = dim_w / 2; // 21
    assert_eq!(bins_t + bins_h + bins_w, half_d);

    // NTK-scaled theta per axis (matches position_embedding.py:96-129):
    // h/w extrapolation_ratio = 4.0 (for 16ch model), t = 1.0
    let base_theta: f64 = 10000.0;
    let h_extrapolation_ratio: f64 = 4.0;
    let w_extrapolation_ratio: f64 = 4.0;
    let t_extrapolation_ratio: f64 = 1.0;
    let h_ntk = h_extrapolation_ratio.powf(dim_h as f64 / (dim_h as f64 - 2.0));
    let w_ntk = w_extrapolation_ratio.powf(dim_w as f64 / (dim_w as f64 - 2.0));
    let t_ntk = t_extrapolation_ratio.powf(dim_t as f64 / (dim_t as f64 - 2.0));
    let theta_h = (base_theta * h_ntk) as f32;
    let theta_w = (base_theta * w_ntk) as f32;
    let theta_t = (base_theta * t_ntk) as f32;

    // Frequency exponents: arange(0, dim_x, 2)[:dim_x//2] / dim_x
    // Then freq = 1.0 / (theta ^ exponent)
    let freqs_t: Vec<f32> = (0..bins_t)
        .map(|i| 1.0 / theta_t.powf((2 * i) as f32 / dim_t as f32))
        .collect();
    let freqs_h: Vec<f32> = (0..bins_h)
        .map(|i| 1.0 / theta_h.powf((2 * i) as f32 / dim_h as f32))
        .collect();
    let freqs_w: Vec<f32> = (0..bins_w)
        .map(|i| 1.0 / theta_w.powf((2 * i) as f32 / dim_w as f32))
        .collect();

    // Build rotation matrices: [total_seq, half_d, 2, 2]
    // Each 2x2 = [[cos, -sin], [sin, cos]]
    let mut rot_data = vec![0.0f32; total_seq * half_d * 4];

    for tf in 0..t_frames {
        for ih in 0..nh {
            for iw in 0..nw {
                let seq_idx = tf * nh * nw + ih * nw + iw;
                let base = seq_idx * half_d * 4;

                let mut dim_offset = 0;

                // Temporal frequencies (22 bins)
                for (fi, &freq) in freqs_t.iter().enumerate() {
                    let angle = (tf as f32) * freq;
                    let c = angle.cos();
                    let s = angle.sin();
                    let off = base + (dim_offset + fi) * 4;
                    rot_data[off] = c;     // [0,0]
                    rot_data[off + 1] = -s; // [0,1]
                    rot_data[off + 2] = s;  // [1,0]
                    rot_data[off + 3] = c;  // [1,1]
                }
                dim_offset += bins_t;

                // Height frequencies (21 bins)
                for (fi, &freq) in freqs_h.iter().enumerate() {
                    let angle = (ih as f32) * freq;
                    let c = angle.cos();
                    let s = angle.sin();
                    let off = base + (dim_offset + fi) * 4;
                    rot_data[off] = c;
                    rot_data[off + 1] = -s;
                    rot_data[off + 2] = s;
                    rot_data[off + 3] = c;
                }
                dim_offset += bins_h;

                // Width frequencies (21 bins)
                for (fi, &freq) in freqs_w.iter().enumerate() {
                    let angle = (iw as f32) * freq;
                    let c = angle.cos();
                    let s = angle.sin();
                    let off = base + (dim_offset + fi) * 4;
                    rot_data[off] = c;
                    rot_data[off + 1] = -s;
                    rot_data[off + 2] = s;
                    rot_data[off + 3] = c;
                }
            }
        }
    }

    Tensor::from_vec_dtype(
        rot_data,
        Shape::from_dims(&[total_seq, half_d, 2, 2]),
        device.clone(),
        DType::BF16,
    )
}

/// Apply 3D RoPE using rotation-matrix format.
///
/// x: [B, S, H, D] where D=head_dim=128
/// rope_rot: [S, D/2, 2, 2] rotation matrices
///
/// For each head dimension pair (2i, 2i+1), apply the 2x2 rotation:
///   [x_even', x_odd'] = [x_even, x_odd] @ [[cos, sin], [-sin, cos]]
/// which is equivalent to:
///   x_even' = x_even * cos - x_odd * sin
///   x_odd'  = x_even * sin + x_odd * cos
fn apply_rope_rotation_matrix(
    x: &Tensor,
    rope_rot: &Tensor, // [S, D/2, 2, 2]
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half_d = d / 2;

    // Reshape x to [B, S, H, D/2, 2]
    let x_pairs = x.reshape(&[b, s, h, half_d, 2])?;
    let x_even = x_pairs.narrow(4, 0, 1)?.squeeze(Some(4))?; // [B, S, H, D/2]
    let x_odd = x_pairs.narrow(4, 1, 1)?.squeeze(Some(4))?;  // [B, S, H, D/2]

    // Extract cos and sin from rotation matrices
    // rope_rot: [S, D/2, 2, 2]
    // cos = rope_rot[:, :, 0, 0], sin = rope_rot[:, :, 1, 0]
    let rot_flat = rope_rot.reshape(&[s, half_d, 4])?;
    let cos = rot_flat.narrow(2, 0, 1)?.squeeze(Some(2))?; // [S, D/2] (element [0,0] = cos)
    let sin = rot_flat.narrow(2, 2, 1)?.squeeze(Some(2))?; // [S, D/2] (element [1,0] = sin)

    // Broadcast: [S, D/2] -> [1, S, 1, D/2]
    let cos = cos.reshape(&[1, s, 1, half_d])?;
    let sin = sin.reshape(&[1, s, 1, half_d])?;

    // Apply rotation
    let new_even = x_even.mul(&cos)?.sub(&x_odd.mul(&sin)?)?;
    let new_odd = x_even.mul(&sin)?.add(&x_odd.mul(&cos)?)?;

    // Interleave back
    let new_even_exp = new_even.unsqueeze(4)?; // [B, S, H, D/2, 1]
    let new_odd_exp = new_odd.unsqueeze(4)?;
    let stacked = Tensor::cat(&[&new_even_exp, &new_odd_exp], 4)?; // [B, S, H, D/2, 2]
    stacked.reshape(&[b, s, h, d])
}

// ===========================================================================
// Resident weight loader
// ===========================================================================

/// Prefixes that should stay resident on GPU (not streamed per-block).
const ANIMA_RESIDENT_PREFIXES: &[&str] = &[
    "net.x_embedder.",
    "net.t_embedder.",
    "net.t_embedding_norm.",
    "net.final_layer.",
    "net.llm_adapter.",
];

/// Load resident weights (everything except net.blocks.{i}) from a safetensors file.
pub fn load_resident_weights(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    load_file_filtered(model_path, device, |key| {
        ANIMA_RESIDENT_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
    })
}

/// Count backbone blocks by probing weight keys.
pub fn count_blocks_from_file(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<usize> {
    let mut num_blocks = 0;
    loop {
        let prefix = format!("net.blocks.{num_blocks}.");
        let probe = load_file_filtered(model_path, device, |key| key.starts_with(&prefix))?;
        if probe.is_empty() {
            break;
        }
        num_blocks += 1;
    }
    Ok(num_blocks)
}
