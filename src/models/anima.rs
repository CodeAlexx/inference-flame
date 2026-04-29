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
use flame_core::bf16_ops::modulate_pre_fused_bf16;
use flame_core::cuda_ops_bf16;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::offload::BlockLoader;
use crate::lora::LoraStack;

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
    /// When true, all weights are in `resident` — skip load_block/unload_block
    all_on_gpu: bool,
    device: Arc<cudarc::driver::CudaDevice>,
    /// Optional runtime LoRA stack. Applied at the `linear_no_bias`
    /// chokepoint after every base matmul. Base weights are never
    /// mutated.
    lora: Option<Arc<LoraStack>>,
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
            all_on_gpu: false,
            device,
            lora: None,
        }
    }

    /// Create with ALL weights already on GPU (no block offloading).
    /// Use when model fits in VRAM (e.g. Anima at 3.9GB on 24GB GPU).
    pub fn new_all_on_gpu(
        model_path: String,
        all_weights: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let loader = BlockLoader::new(model_path, device.clone());
        Self {
            config: AnimaConfig::default(),
            resident: all_weights,
            loader,
            all_on_gpu: true,
            device,
            lora: None,
        }
    }

    /// Attach a runtime LoRA stack. Subsequent forwards add
    /// `scale * up(down(x))` from any matching LoRA entries to the
    /// base output of every linear in the model. Base weights are
    /// not mutated.
    pub fn set_lora(&mut self, lora: Arc<LoraStack>) {
        self.lora = Some(lora);
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
        // weight is [out, in], need [in, out] for x @ weight_t
        let wt = weight.permute(&[1, 0])?;
        let out_2d = x_2d.matmul(&wt)?;

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        let out = out_2d.reshape(&out_shape)?;

        // Apply LoRA at the chokepoint. `LoraStack::apply` returns
        // `out` unchanged when `weight_key` has no registered entry,
        // so this stays inert for non-target linears (RMSNorm scales
        // are loaded via `self.w` directly, not this function).
        match &self.lora {
            Some(stack) => stack.apply(weight_key, x, out),
            None => Ok(out),
        }
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

    // -- RMSNorm (using fused CUDA kernel) -----------------------------------

    fn rms_norm(&self, x: &Tensor, weight_key: &str, eps: f32) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let x_dims = x.shape().dims().to_vec();
        let last_dim = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let flat = x.reshape(&[batch, last_dim])?;
        let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(weight), eps)?;
        normed.reshape(&x_dims)
    }

    /// Apply RMSNorm per head: input [B, S, H, D], norm weight [D]
    fn rms_norm_per_head(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * s * h, d])?;
        let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(weight), 1e-6)?;
        normed.reshape(&[b, s, h, d])
    }

    /// Apply RMSNorm per head: input [B, H, S, D], norm weight [D]
    fn rms_norm_per_head_bhsd(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * h * s, d])?;
        let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(weight), 1e-6)?;
        normed.reshape(&[b, h, s, d])
    }

    // ========================================================================
    // Timestep embedder
    // ========================================================================

    /// Prepare timestep conditioning.
    /// Returns (t_cond [B, 2048], base_adaln [B, 6144]):
    /// - t_cond: RMSNormed hidden state, input to per-block adaln_modulation
    /// - base_adaln: 6144-dim base modulation added to each sub-block's output
    ///
    /// Flow: sinusoidal(2048) -> Linear(2048,2048,no bias) -> SiLU -> hidden(2048)
    ///       hidden -> Linear(2048,6144,no bias) -> base_adaln
    ///       sinusoidal -> RMSNorm -> t_cond  (NOT hidden — adaln_lora returns raw input)
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

        // t_cond = RMSNorm(sinusoidal) [B, 2048] — input to per-block adaln_modulation
        // IMPORTANT: With use_adaln_lora=True, TimestepEmbedding returns the ORIGINAL
        // sinusoidal embedding (not the SiLU'd hidden), then t_embedding_norm is applied.
        let t_cond = self.rms_norm(&emb, "net.t_embedding_norm.weight", 1e-6)?;

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
        base_adaln: &Tensor, // [B, 6144] — take first 4096 elements
    ) -> Result<(Tensor, Tensor)> {
        let t_silu = t_cond.silu()?;
        let h = self.linear_no_bias(&t_silu, "net.final_layer.adaln_modulation.1.weight")?;
        let mod_out = self.linear_no_bias(&h, "net.final_layer.adaln_modulation.2.weight")?; // [B, 4096]

        // Add base_adaln[:, :2*D] (first 4096 of 6144)
        let dim = self.config.model_channels;
        let adaln_slice = base_adaln.narrow(1, 0, 2 * dim)?; // [B, 4096]
        let mod_out = mod_out.add(&adaln_slice)?;

        let shift = mod_out.narrow(1, 0, dim)?;
        let scale = mod_out.narrow(1, dim, dim)?;

        Ok((shift, scale))
    }

    /// Apply adaLN: (1 + scale) * LayerNorm(x) + shift — fused kernel.
    /// shift, scale: [B, D], x: [B, S, D]
    fn apply_adaln(
        &self,
        x: &Tensor,
        shift: &Tensor,
        scale: &Tensor,
    ) -> Result<Tensor> {
        modulate_pre_fused_bf16(x, shift, scale, 1e-6)
    }

    // ========================================================================
    // Self-attention (with 3D RoPE rotation-matrix format)
    // ========================================================================

    fn self_attention(
        &self,
        x: &Tensor,
        rope_cos: &Tensor, // [1, 1, S, D/2]
        rope_sin: &Tensor, // [1, 1, S, D/2]
        prefix: &str,
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

        // Reshape [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        let q = q.reshape(&[b, seq, num_heads, head_dim])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq, num_heads, head_dim])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq, num_heads, head_dim])?.permute(&[0, 2, 1, 3])?;

        // QK RMSNorm per-head ([B, H, S, D] format)
        let q = self.rms_norm_per_head_bhsd(&q, &format!("{prefix}.q_norm.weight"))?;
        let k = self.rms_norm_per_head_bhsd(&k, &format!("{prefix}.k_norm.weight"))?;

        // Apply 3D RoPE using half-split kernel (matches standalone trainer's interleaved=False)
        let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, rope_cos, rope_sin)?;
        let k = flame_core::bf16_ops::rope_halfsplit_bf16(&k, rope_cos, rope_sin)?;

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
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("net.blocks.{block_idx}");

        // FP32 residual stream (model has large values ~200+, BF16 loses precision)
        let mut x_f32 = x.to_dtype(DType::F32)?;

        // --- Self-attention ---
        let (shift_sa, scale_sa, gate_sa) = self.adaln_modulation(
            t_cond,
            base_adaln,
            &format!("{prefix}.adaln_modulation_self_attn"),
        )?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;
        let x_mod = self.apply_adaln(&x_bf16, &shift_sa, &scale_sa)?;
        let attn_out = self.self_attention(&x_mod, rope_cos, rope_sin, &format!("{prefix}.self_attn"))?;
        let gate_sa_unsq = gate_sa.unsqueeze(1)?;
        x_f32 = x_f32.add(&attn_out.to_dtype(DType::F32)?.mul(&gate_sa_unsq.to_dtype(DType::F32)?)?)?;

        // --- Cross-attention ---
        let (shift_ca, scale_ca, gate_ca) = self.adaln_modulation(
            t_cond,
            base_adaln,
            &format!("{prefix}.adaln_modulation_cross_attn"),
        )?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;
        let x_mod = self.apply_adaln(&x_bf16, &shift_ca, &scale_ca)?;
        let cross_out =
            self.cross_attention(&x_mod, context, &format!("{prefix}.cross_attn"))?;
        let gate_ca_unsq = gate_ca.unsqueeze(1)?;
        x_f32 = x_f32.add(&cross_out.to_dtype(DType::F32)?.mul(&gate_ca_unsq.to_dtype(DType::F32)?)?)?;

        // --- MLP ---
        let (shift_mlp, scale_mlp, gate_mlp) = self.adaln_modulation(
            t_cond,
            base_adaln,
            &format!("{prefix}.adaln_modulation_mlp"),
        )?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;
        let x_mod = self.apply_adaln(&x_bf16, &shift_mlp, &scale_mlp)?;
        let mlp_out = self.mlp(&x_mod, &format!("{prefix}.mlp"))?;
        let gate_mlp_unsq = gate_mlp.unsqueeze(1)?;
        x_f32 = x_f32.add(&mlp_out.to_dtype(DType::F32)?.mul(&gate_mlp_unsq.to_dtype(DType::F32)?)?)?;

        x_f32.to_dtype(DType::BF16)
    }

    // ========================================================================
    // Final layer
    // ========================================================================

    fn final_layer(&self, x: &Tensor, t_cond: &Tensor, base_adaln: &Tensor) -> Result<Tensor> {
        let (shift, scale) = self.final_adaln_modulation(t_cond, base_adaln)?;
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
        // Permute to [B, T, nH, nW, C+1, pH, pW] — channels OUTERMOST in patch dim
        // Matches Python einops: "b c (t r) (h m) (w n) -> b t h w (c r m n)"
        // where c varies slowest and n (spatial width) varies fastest
        let x_p = x_r.permute(&[0, 1, 2, 4, 6, 3, 5])?;
        // Flatten patches: [B, T*nH*nW, (C+1)*pH*pW] = [B, T*nH*nW, 68]
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

    /// Run LLM adapter once to produce text context. Cache the result
    /// and pass it to `forward_with_context` to avoid recomputing every step.
    pub fn encode_context(
        &self,
        token_ids: &Tensor,
        llm_hidden: &Tensor,
    ) -> Result<Tensor> {
        self.llm_adapter(token_ids, llm_hidden)
    }

    /// Full MiniTrainDIT forward with pre-computed context.
    ///
    /// Arguments:
    /// - `x`: latent [B, T, H, W, C] where C=16
    /// - `timestep`: [B] float timestep values
    /// - `context`: [B, S_txt, 1024] pre-computed from `encode_context`
    ///
    /// Returns: predicted noise/velocity [B, T, H, W, C]
    pub fn forward_with_context(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let t_frames = x_dims[1];

        // 1. Prepare timestep conditioning
        let (t_cond, base_adaln) = self.prepare_timestep(timestep)?;

        // 2. Patchify: [B, T, H, W, 16] -> [B, N_patches, 68]
        let (patches, _t, nh, nw) = self.patchify(x)?;

        // 3. Patch embed: [B, N_patches, 68] -> [B, N_patches, 2048]
        let x_emb = self.patch_embed(&patches)?;

        // 4. Build 3D RoPE cos/sin tables
        let (rope_cos, rope_sin) = build_3d_rope_cossin(
            t_frames,
            nh,
            nw,
            self.config.head_dim,
            &self.device,
        )?;

        // 5. Run through 28 transformer blocks
        let mut x_hidden = x_emb;
        for i in 0..self.config.num_blocks {
            if !self.all_on_gpu {
                let prefix = format!("net.blocks.{i}");
                self.load_block(&prefix)?;
            }
            x_hidden = self.transformer_block(
                &x_hidden,
                context,
                &t_cond,
                &base_adaln,
                &rope_cos,
                &rope_sin,
                i,
            )?;
            if !self.all_on_gpu {
                self.unload_block();
            }
        }

        // 6. Final layer
        let x_out = self.final_layer(&x_hidden, &t_cond, &base_adaln)?;

        // 7. Unpatchify: [B, N_patches, 64] -> [B, T, H, W, 16]
        self.unpatchify(&x_out, t_frames, nh, nw)
    }

    /// Full MiniTrainDIT forward (convenience — runs adapter internally).
    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        token_ids: &Tensor,
        llm_hidden: &Tensor,
    ) -> Result<Tensor> {
        let context = self.encode_context(token_ids, llm_hidden)?;
        self.forward_with_context(x, timestep, &context)
    }
}

// ===========================================================================
// Standalone helpers
// ===========================================================================

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

    // Half-split rotate: out[:D/2] = x1*cos - x2*sin, out[D/2:] = x2*cos + x1*sin
    let new_x1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
    let new_x2 = x2.mul(&cos)?.add(&x1.mul(&sin)?)?;

    // Concatenate halves back along last dim (NOT interleaved)
    let flat1 = new_x1.reshape(&[b * s * h, half_d])?;
    let flat2 = new_x2.reshape(&[b * s * h, half_d])?;
    let result = Tensor::cat(&[&flat1, &flat2], 1)?;
    result.reshape(&[b, s, h, d])
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

/// Build 3D RoPE cos/sin tables for the fused kernel.
/// Returns (cos, sin) each [1, 1, S, D/2] where S = T*nH*nW.
/// Uses the same frequency computation as build_3d_rope_rotation_matrix.
fn build_3d_rope_cossin(
    t_frames: usize,
    nh: usize,
    nw: usize,
    head_dim: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half_d = head_dim / 2;
    let total_seq = t_frames * nh * nw;

    let full_d = half_d * 2;
    let dim_h: usize = full_d / 6 * 2;
    let dim_w: usize = dim_h;
    let dim_t: usize = full_d - 2 * dim_h;
    let bins_t = dim_t / 2;
    let bins_h = dim_h / 2;
    let bins_w = dim_w / 2;

    // Extrapolation ratios: 4.0 for in_channels=16 (padding mask is a concat, not in_channels)
    let base_theta: f64 = 10000.0;
    let h_ntk = 4.0f64.powf(dim_h as f64 / (dim_h as f64 - 2.0));
    let w_ntk = 4.0f64.powf(dim_w as f64 / (dim_w as f64 - 2.0));
    let t_ntk = 1.0f64.powf(dim_t as f64 / (dim_t as f64 - 2.0));
    let theta_h = (base_theta * h_ntk) as f32;
    let theta_w = (base_theta * w_ntk) as f32;
    let theta_t = (base_theta * t_ntk) as f32;

    let freqs_t: Vec<f32> = (0..bins_t)
        .map(|i| 1.0 / theta_t.powf((2 * i) as f32 / dim_t as f32))
        .collect();
    let freqs_h: Vec<f32> = (0..bins_h)
        .map(|i| 1.0 / theta_h.powf((2 * i) as f32 / dim_h as f32))
        .collect();
    let freqs_w: Vec<f32> = (0..bins_w)
        .map(|i| 1.0 / theta_w.powf((2 * i) as f32 / dim_w as f32))
        .collect();

    // Build angle values for each position: [T_angles, H_angles, W_angles]
    // Then DOUBLE them: [angles, angles] to produce head_dim-sized freq tensor
    // This is how Cosmos/Anima does it — apply_rotary_pos_emb uses half-split,
    // computing cos(freqs) and sin(freqs) where freqs has shape [S, head_dim].
    // The first half and second half of freqs are identical angle values.
    let mut cos_data = vec![0.0f32; total_seq * half_d * 2]; // doubled: head_dim
    let mut sin_data = vec![0.0f32; total_seq * half_d * 2];

    for tf in 0..t_frames {
        for ih in 0..nh {
            for iw in 0..nw {
                let seq_idx = tf * nh * nw + ih * nw + iw;
                // Two copies: [0..half_d) and [half_d..head_dim)
                for copy in 0..2 {
                    let base = seq_idx * (half_d * 2) + copy * half_d;
                    let mut dim_offset = 0;

                    for (fi, &freq) in freqs_t.iter().enumerate() {
                        let angle = (tf as f32) * freq;
                        cos_data[base + dim_offset + fi] = angle.cos();
                        sin_data[base + dim_offset + fi] = angle.sin();
                    }
                    dim_offset += bins_t;

                    for (fi, &freq) in freqs_h.iter().enumerate() {
                        let angle = (ih as f32) * freq;
                        cos_data[base + dim_offset + fi] = angle.cos();
                        sin_data[base + dim_offset + fi] = angle.sin();
                    }
                    dim_offset += bins_h;

                    for (fi, &freq) in freqs_w.iter().enumerate() {
                        let angle = (iw as f32) * freq;
                        cos_data[base + dim_offset + fi] = angle.cos();
                        sin_data[base + dim_offset + fi] = angle.sin();
                    }
                }
            }
        }
    }

    // rope_halfsplit_bf16 expects cos/sin shape [1, 1, S, D/2] where D = head_dim
    // But our cos_data is [S, head_dim] = [S, 2*half_d]
    // rope_halfsplit_bf16 expects D/2 = head_dim/2 = half_d
    // So we pass [1, 1, S, half_d] — the kernel will apply half-split on head_dim
    // Wait — rope_halfsplit_bf16 takes [B,H,N,D] input and [1,1,N,D/2] cos/sin
    // With D=head_dim=128, D/2=64=half_d. The cos/sin should contain the
    // angle values for the first half of head_dim.
    //
    // Actually, the Cosmos code builds freqs as [S, head_dim] with doubled angles,
    // then in apply_rotary_pos_emb_base it does cos(freqs) for ALL head_dim.
    // The half-split then uses cos[:half] and sin[:half] for the rotation.
    // Since the two halves are identical, we just need [S, half_d] cos/sin.
    //
    // rope_halfsplit_bf16 already handles this correctly:
    // out[0..D/2] = x[0..D/2]*cos - x[D/2..D]*sin
    // out[D/2..D] = x[D/2..D]*cos + x[0..D/2]*sin
    // So we just need cos/sin of the angle values (half_d each).

    // Take first half of the doubled data (they're identical)
    let cos_half: Vec<f32> = cos_data.chunks(half_d * 2)
        .flat_map(|chunk| chunk[..half_d].iter().copied())
        .collect();
    let sin_half: Vec<f32> = sin_data.chunks(half_d * 2)
        .flat_map(|chunk| chunk[..half_d].iter().copied())
        .collect();

    let cos = Tensor::from_vec_dtype(
        cos_half,
        Shape::from_dims(&[1, 1, total_seq, half_d]),
        device.clone(),
        DType::BF16,
    )?;
    let sin = Tensor::from_vec_dtype(
        sin_half,
        Shape::from_dims(&[1, 1, total_seq, half_d]),
        device.clone(),
        DType::BF16,
    )?;

    Ok((cos, sin))
}

/// Apply 3D RoPE using pre-computed cos/sin tables.
///
/// x: [B, S, H, D] where D=head_dim=128
/// rope_cos, rope_sin: [S, D/2] pre-computed from build_3d_rope_cossin
///
/// Uses interleaved pairs: (x[2i], x[2i+1]) rotated by (cos[i], sin[i]).
fn apply_rope_interleaved(
    x: &Tensor,
    rope_cos: &Tensor, // [S, D/2]
    rope_sin: &Tensor, // [S, D/2]
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half_d = d / 2;

    // Use half-split approach: first half and second half of head_dim
    // Reshape to [B*S*H, D], split into halves, apply rotation, concat back
    let flat = x.reshape(&[b * s * h, d])?;
    let x1 = flat.narrow(1, 0, half_d)?;      // [BSH, D/2] — first half
    let x2 = flat.narrow(1, half_d, half_d)?;  // [BSH, D/2] — second half

    // Broadcast cos/sin: [S, D/2] -> [B*S*H, D/2]
    // Each of the B*H copies of a position shares the same cos/sin
    let cos_flat = rope_cos.reshape(&[1, s, 1, half_d])?;
    let sin_flat = rope_sin.reshape(&[1, s, 1, half_d])?;
    // Broadcast over B and H by reshaping x to [B, S, H, D/2]
    let x1_4d = x1.reshape(&[b, s, h, half_d])?;
    let x2_4d = x2.reshape(&[b, s, h, half_d])?;

    // rotate_half: new_x1 = x1*cos - x2*sin, new_x2 = x1*sin + x2*cos
    let new_x1 = x1_4d.mul(&cos_flat)?.sub(&x2_4d.mul(&sin_flat)?)?;
    let new_x2 = x1_4d.mul(&sin_flat)?.add(&x2_4d.mul(&cos_flat)?)?;

    // Concat halves back: [B, S, H, D/2] + [B, S, H, D/2] -> [B, S, H, D]
    let new_x1_flat = new_x1.reshape(&[b * s * h, half_d])?;
    let new_x2_flat = new_x2.reshape(&[b * s * h, half_d])?;
    let result = Tensor::cat(&[&new_x1_flat, &new_x2_flat], 1)?;
    result.reshape(&[b, s, h, d])
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

/// Load ALL weights (resident + blocks) to GPU. For small models that fit in VRAM.
pub fn load_all_weights(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    flame_core::serialization::load_file(std::path::Path::new(model_path), device)
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
