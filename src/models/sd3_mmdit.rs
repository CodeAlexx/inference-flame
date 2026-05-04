//! SD3 MMDiT (Multi-Modal Diffusion Transformer) — pure Rust, key-exact.
//!
//! Matches the Stability AI SD3/SD3.5 checkpoints after stripping
//! the `model.diffusion_model.` prefix.
//!
//! Architecture:
//! - Joint transformer blocks with dual-stream (x_block for image, context_block for text)
//! - Per-block adaLN: conditioning vector `c` (timestep + pooled text) -> SiLU -> Linear -> chunk
//! - LAST block's context_block is pre_only: only 2 mods (shift, scale), no proj, no MLP
//! - LayerNorm(elementwise_affine=False) for pre-norm, LayerNorm(affine=True) for QK norm
//! - GELU(tanh) MLP with 4x expansion
//! - Patch embedding: Conv2d(in_channels, hidden, kernel_size=patch_size, stride=patch_size)
//! - Learned position embedding (cropped to actual patch grid)
//! - No RoPE
//!
//! Weight keys (non-block):
//!   pos_embed                                [1, num_patches, hidden]
//!   x_embedder.proj.weight/bias              [hidden, in_ch, patch, patch]
//!   t_embedder.mlp.{0,2}.weight/bias
//!   y_embedder.mlp.{0,2}.weight/bias
//!   context_embedder.weight/bias
//!   final_layer.adaLN_modulation.1.weight/bias
//!   final_layer.linear.weight/bias
//!
//! Joint block keys:
//!   joint_blocks.{i}.{x_block,context_block}.adaLN_modulation.1.weight/bias
//!   joint_blocks.{i}.{x_block,context_block}.attn.qkv.weight/bias
//!   joint_blocks.{i}.{x_block,context_block}.attn.proj.weight/bias  (NOT context pre_only)
//!   joint_blocks.{i}.{x_block,context_block}.attn.ln_q.weight/bias
//!   joint_blocks.{i}.{x_block,context_block}.attn.ln_k.weight/bias
//!   joint_blocks.{i}.{x_block,context_block}.mlp.fc1.weight/bias    (NOT context pre_only)
//!   joint_blocks.{i}.{x_block,context_block}.mlp.fc2.weight/bias    (NOT context pre_only)

use flame_core::attention::sdpa;
use flame_core::conv::Conv2d;
use flame_core::layer_norm::layer_norm;
use flame_core::norm::rms_norm;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::offload::BlockLoader;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct SD3Config {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f32,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_dim: usize,
    pub pooled_dim: usize,
    pub timestep_dim: usize,
    pub pos_embed_max_size: usize,
    pub num_patches: usize,
    pub qk_norm_has_bias: bool,
    /// SD3.5 Medium uses dual attention (attn + attn2 in x_block).
    /// Detected from the adaLN output size: 9*hidden → dual, 6*hidden → single.
    pub use_dual_attention: bool,
}

impl SD3Config {
    /// Auto-detect config from resident weights.
    pub fn from_weights(resident: &HashMap<String, Tensor>) -> Self {
        // hidden_size from x_embedder.proj.weight shape[0]
        let proj_w = resident.get("x_embedder.proj.weight").expect("missing x_embedder.proj.weight");
        let proj_dims = proj_w.shape().dims().to_vec();
        let hidden_size = proj_dims[0];
        let in_channels = proj_dims[1];
        let patch_size = proj_dims[2]; // kernel_size = stride = patch_size

        // num_heads = hidden_size / 64
        let head_dim = 64;
        let num_heads = hidden_size / head_dim;

        // out_channels from final_layer.linear.weight shape[0] / (patch_size^2)
        let final_w = resident.get("final_layer.linear.weight").expect("missing final_layer.linear.weight");
        let total_out = final_w.shape().dims()[0];
        let out_channels = total_out / (patch_size * patch_size);

        // context_dim from context_embedder.weight shape[1]
        let ctx_w = resident.get("context_embedder.weight").expect("missing context_embedder.weight");
        let context_dim = ctx_w.shape().dims()[1];

        // pooled_dim from y_embedder.mlp.0.weight shape[1]
        let y_w = resident.get("y_embedder.mlp.0.weight").expect("missing y_embedder.mlp.0.weight");
        let pooled_dim = y_w.shape().dims()[1];

        // timestep_dim from t_embedder.mlp.0.weight shape[1]
        let t_w = resident.get("t_embedder.mlp.0.weight").expect("missing t_embedder.mlp.0.weight");
        let timestep_dim = t_w.shape().dims()[1];

        // num_patches from pos_embed shape[1]
        let pos = resident.get("pos_embed").expect("missing pos_embed");
        let num_patches = pos.shape().dims()[1];
        let pos_embed_max_size = (num_patches as f64).sqrt().round() as usize;

        // Count depth by scanning keys for joint_blocks.{i}.x_block.adaLN_modulation.1.weight
        let mut depth = 0;
        loop {
            let key = format!("joint_blocks.{depth}.x_block.adaLN_modulation.1.weight");
            // Check resident — but blocks are streamed, so we need to count from keys
            // that might exist in the safetensors file. Since we only have resident here,
            // we infer depth from hidden_size = 64 * depth (SD3 convention).
            // However, this doesn't hold for all models. For safety, we also accept
            // depth = hidden_size / head_dim, which is the number of heads.
            // SD3: num_heads = depth. If that fails, fall back to file scan.
            if resident.contains_key(&key) {
                depth += 1;
            } else {
                break;
            }
        }
        // If we couldn't count from resident (blocks not loaded), use hidden_size / 64
        if depth == 0 {
            depth = hidden_size / head_dim;
        }

        // QK norm: check if ln_q has bias (loaded from block, not resident)
        // Default to true (LayerNorm with bias) — SD3 uses LN for QK norm
        let qk_norm_has_bias = true;

        // Dual attention: detect from adaLN output size.
        // 9*hidden → SD3.5 Medium (dual attention), 6*hidden → SD3/SD3.5 Large (single).
        let ada_key = "joint_blocks.0.x_block.adaLN_modulation.1.weight";
        let use_dual_attention = if let Some(ada_w) = resident.get(ada_key) {
            let ada_out = ada_w.shape().dims()[0];
            ada_out / hidden_size == 9
        } else {
            false
        };

        Self {
            depth,
            hidden_size,
            num_heads,
            head_dim,
            mlp_ratio: 4.0,
            patch_size,
            in_channels,
            out_channels,
            context_dim,
            pooled_dim,
            timestep_dim,
            pos_embed_max_size,
            num_patches,
            qk_norm_has_bias,
            use_dual_attention,
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct SD3MMDiT {
    pub config: SD3Config,
    /// Small weights that stay on GPU permanently (embedders, pos_embed, final layer).
    resident: HashMap<String, Tensor>,
    /// Block loader for on-demand weight streaming via mmap.
    loader: BlockLoader,
    device: Arc<cudarc::driver::CudaDevice>,
    /// Optional runtime LoRA stack — applied at each `linear_no_bias` call.
    /// Base weights are never mutated.
    lora: Option<Arc<crate::lora::LoraStack>>,
}

impl SD3MMDiT {
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let config = SD3Config::from_weights(&resident);
        let loader = BlockLoader::new(model_path, device.clone());
        Self {
            config,
            resident,
            loader,
            device,
            lora: None,
        }
    }

    /// Create with a key prefix for the safetensors file (e.g. "model.diffusion_model.").
    /// The BlockLoader will search for prefixed keys and strip the prefix on load.
    pub fn new_with_prefix(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
        key_prefix: &str,
    ) -> Self {
        let config = SD3Config::from_weights(&resident);
        let loader = BlockLoader::new_with_prefix(model_path, device.clone(), key_prefix);
        Self {
            config,
            resident,
            loader,
            device,
            lora: None,
        }
    }

    /// Load a joint block's weights from disk into GPU.
    pub fn load_block(&mut self, prefix: &str) -> Result<()> {
        self.loader.load_block(prefix)
    }

    /// Drop current block weights to free VRAM.
    pub fn unload_block(&mut self) {
        self.loader.unload_block();
    }

    /// Get a weight tensor by key — checks block cache first, then resident.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.loader.get(key, &self.resident)
    }

    // -- Linear helpers -------------------------------------------------------

    /// Attach a runtime LoRA stack. Subsequent `linear_no_bias` (and via it,
    /// `linear_with_bias`) calls will add `scale * up(down(x))` from any
    /// matching LoRA entries to the base output. Base weights are not mutated.
    pub fn set_lora(&mut self, lora: Arc<crate::lora::LoraStack>) {
        self.lora = Some(lora);
    }

    /// x @ weight.T (weight shape: [out, in], no bias)
    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let x_dims = x.shape().dims().to_vec();
        let in_features = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let out_features = weight.shape().dims()[0];

        let x_2d = x.reshape(&[batch, in_features])?;
        // Compute x @ weight.T via cuBLASLt with TRANSB=T directly.
        // The previous `permute([1,0]).contiguous().matmul(...)` path produced
        // numerically wrong output for SD3.5 Large's [14592, 2432] adaLN
        // weight, causing block-0 hidden state to explode ~5000× → uniform
        // color final image. matmul_bf16_trans avoids the permute+materialize.
        let out_2d = flame_core::ops::gemm_bf16::matmul_bf16_trans(
            &x_2d, weight, false, true,
        )?;

        let out_2d = match self.lora {
            Some(ref lora) => lora.apply(weight_key, &x_2d, out_2d)?,
            None => out_2d,
        };

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

    // -- LayerNorm helpers ----------------------------------------------------

    /// LayerNorm without affine (elementwise_affine=False).
    fn layer_norm_no_affine(&self, x: &Tensor) -> Result<Tensor> {
        let dim = self.config.hidden_size;
        layer_norm(x, &[dim], None, None, 1e-6)
    }

    /// LayerNorm with affine parameters (for QK norm, per head_dim).
    fn layer_norm_affine(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dim = weight.shape().dims()[0];
        let bias = if self.config.qk_norm_has_bias {
            Some(self.w(bias_key)?)
        } else {
            None
        };
        layer_norm(x, &[dim], Some(weight), bias, 1e-6)
    }

    // -- Modulation helper ----------------------------------------------------

    /// adaLN modulation: x * (1 + scale) + shift
    /// shift, scale: [B, D] — unsqueeze to [B, 1, D] for broadcast over [B, N, D]
    fn modulate(&self, x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let ones = Tensor::from_vec_dtype(
            vec![1.0f32],
            Shape::from_dims(&[1, 1, 1]),
            self.device.clone(),
            DType::BF16,
        )?;
        let scale_unsq = scale.unsqueeze(1)?;
        let shift_unsq = shift.unsqueeze(1)?;
        let factor = ones.add(&scale_unsq)?;
        let scaled = x.mul(&factor)?;
        scaled.add(&shift_unsq)
    }

    // -- Timestep embedding ---------------------------------------------------

    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        let freq_dim = self.config.timestep_dim; // 256
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

        // MLP: Linear(256, hidden) -> SiLU -> Linear(hidden, hidden)
        let h = self.linear_with_bias(&emb, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    // -- Pooled text embedding ------------------------------------------------

    fn pooled_embed(&self, y: &Tensor) -> Result<Tensor> {
        // MLP: Linear(pooled_dim, hidden) -> SiLU -> Linear(hidden, hidden)
        let h = self.linear_with_bias(y, "y_embedder.mlp.0.weight", "y_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "y_embedder.mlp.2.weight", "y_embedder.mlp.2.bias")
    }

    // -- Patch embed (Conv2d) -------------------------------------------------

    fn patch_embed(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, H, W] (NCHW)
        // Conv2d(in_channels, hidden_size, kernel=patch_size, stride=patch_size, bias=True)
        let w = self.w("x_embedder.proj.weight")?;
        let b = self.w("x_embedder.proj.bias")?;
        let p = self.config.patch_size;
        let in_ch = self.config.in_channels;
        let out_ch = self.config.hidden_size;

        let mut conv = Conv2d::new_with_bias(in_ch, out_ch, p, p, 0, self.device.clone(), true)?;
        conv.copy_weight_from(w)?;
        conv.copy_bias_from(b)?;

        // Conv2d expects NCHW, output is NCHW [B, hidden, pH, pW]
        let out = conv.forward(x)?;

        // Flatten spatial to sequence: [B, hidden, pH, pW] -> [B, pH*pW, hidden]
        let out_dims = out.shape().dims().to_vec();
        let (batch, ch, ph, pw) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
        let out = out.reshape(&[batch, ch, ph * pw])?;
        // [B, C, N] -> [B, N, C]
        out.permute(&[0, 2, 1])
    }

    // -- Cropped position embedding -------------------------------------------

    fn cropped_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let p = self.config.patch_size;
        let ph = (h + 1) / p;
        let pw = (w + 1) / p;
        let max_s = self.config.pos_embed_max_size;

        let top = (max_s - ph) / 2;
        let left = (max_s - pw) / 2;

        // pos_embed: [1, max_s*max_s, hidden] -> reshape to [1, max_s, max_s, hidden]
        let pos = self.w("pos_embed")?;
        let hidden = self.config.hidden_size;
        let spatial = pos.reshape(&[1, max_s, max_s, hidden])?;

        // Crop rows [top..top+ph], cols [left..left+pw]
        let row_crop = spatial.narrow(1, top, ph)?;
        let crop = row_crop.narrow(2, left, pw)?;

        // Back to [1, ph*pw, hidden]
        crop.reshape(&[1, ph * pw, hidden])
    }

    // -- Unpatchify -----------------------------------------------------------

    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let c = self.config.out_channels;
        let p = self.config.patch_size;
        let ph = (h + 1) / p;
        let pw = (w + 1) / p;
        let b = x.shape().dims()[0];

        // x: [B, N, P*P*C] -> [B, pH, pW, P, P, C]
        let x = x.reshape(&[b, ph, pw, p, p, c])?;
        // einsum "nhwpqc->nchpwq" -> [B, C, pH, P, pW, P]
        let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
        x.reshape(&[b, c, ph * p, pw * p])
    }

    // -- Pre-attention for one stream (DismantledBlock) -----------------------

    /// Compute Q, K, V for one stream with QK norm.
    /// Returns (q, k, v) each as [B, H, N, D].
    fn pre_attention_qkv(
        &self,
        x: &Tensor,
        prefix: &str, // e.g. "joint_blocks.0.x_block"
        attn_name: &str, // "attn" or "attn2"
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let b = x.shape().dims()[0];
        let n = x.shape().dims()[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // QKV: [B, N, 3*hidden]
        let qkv = self.linear_with_bias(
            x,
            &format!("{prefix}.{attn_name}.qkv.weight"),
            &format!("{prefix}.{attn_name}.qkv.bias"),
        )?;

        // Reshape to [B, N, 3, H, D] then permute to [3, B, H, N, D]
        let qkv = qkv.reshape(&[b, n, 3, num_heads, head_dim])?;
        let qkv = qkv.permute(&[2, 0, 3, 1, 4])?;

        // Split into q, k, v: each [B, H, N, D]
        // narrow on dim 0 (the '3' dimension)
        let q = qkv.narrow(0, 0, 1)?.squeeze(Some(0))?;
        let k = qkv.narrow(0, 1, 1)?.squeeze(Some(0))?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(Some(0))?;

        // QK norm (LayerNorm with affine, per head_dim)
        // q: [B, H, N, D] -> reshape to [B*H*N, D] for layer_norm, then back
        let q = self.qk_norm_4d(&q, &format!("{prefix}.{attn_name}.ln_q"))?;
        let k = self.qk_norm_4d(&k, &format!("{prefix}.{attn_name}.ln_k"))?;

        Ok((q, k, v))
    }

    /// Apply QK LayerNorm on a [B, H, N, D] tensor.
    fn qk_norm_4d(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, h, n, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * h * n, d])?;
        let weight = self.w(&format!("{prefix}.weight"))?;
        let dim = weight.shape().dims()[0];
        // SD3.5 (medium + large) uses RMSNorm for ln_q/ln_k per ComfyUI MMDiT
        // (qk_norm="rms"). The previous LayerNorm path was incorrect.
        let normed = rms_norm(&flat, &[dim], Some(weight), 1e-6)?;
        normed.reshape(&[b, h, n, d])
    }

    // -- Joint block forward --------------------------------------------------

    fn joint_block(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        block_idx: usize,
        is_last: bool,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let prefix = format!("joint_blocks.{block_idx}");
        let x_prefix = format!("{prefix}.x_block");
        let ctx_prefix = format!("{prefix}.context_block");
        let hidden = self.config.hidden_size;

        // ------ Context stream pre-attention ------
        let ctx_mods = {
            let c_silu = c.silu()?;
            self.linear_with_bias(
                &c_silu,
                &format!("{ctx_prefix}.adaLN_modulation.1.weight"),
                &format!("{ctx_prefix}.adaLN_modulation.1.bias"),
            )?
        };

        let (ctx_q, ctx_k, ctx_v, ctx_intermediates) = if is_last {
            // pre_only: 2 mods (shift_msa, scale_msa)
            let chunks = ctx_mods.chunk(2, ctx_mods.shape().dims().len() - 1)?;
            let shift_msa = &chunks[0];
            let scale_msa = &chunks[1];

            let ctx_norm = self.layer_norm_no_affine(context)?;
            let ctx_mod = self.modulate(&ctx_norm, shift_msa, scale_msa)?;
            let (q, k, v) = self.pre_attention_qkv(&ctx_mod, &ctx_prefix, "attn")?;
            (q, k, v, None)
        } else {
            // 6 mods
            let chunks = ctx_mods.chunk(6, ctx_mods.shape().dims().len() - 1)?;
            let shift_msa = &chunks[0];
            let scale_msa = &chunks[1];
            let gate_msa = &chunks[2];
            let shift_mlp = &chunks[3];
            let scale_mlp = &chunks[4];
            let gate_mlp = &chunks[5];

            let ctx_norm = self.layer_norm_no_affine(context)?;
            let ctx_mod = self.modulate(&ctx_norm, shift_msa, scale_msa)?;
            let (q, k, v) = self.pre_attention_qkv(&ctx_mod, &ctx_prefix, "attn")?;
            (
                q,
                k,
                v,
                Some((
                    context.clone(),
                    gate_msa.clone(),
                    shift_mlp.clone(),
                    scale_mlp.clone(),
                    gate_mlp.clone(),
                )),
            )
        };

        // ------ X stream pre-attention ------
        let x_mods = {
            let c_silu = c.silu()?;
            self.linear_with_bias(
                &c_silu,
                &format!("{x_prefix}.adaLN_modulation.1.weight"),
                &format!("{x_prefix}.adaLN_modulation.1.bias"),
            )?
        };

        // Per-block dual attention detection: check adaLN output size.
        // 9*hidden → dual (blocks 0-12 in SD3.5 Medium), 6*hidden → single.
        let ada_out_dim = x_mods.dims()[x_mods.dims().len() - 1];
        let block_has_dual = ada_out_dim / hidden == 9;
        let num_mods = if block_has_dual { 9 } else { 6 };
        let x_chunks = x_mods.chunk(num_mods, x_mods.shape().dims().len() - 1)?;
        let x_shift_msa = &x_chunks[0];
        let x_scale_msa = &x_chunks[1];
        let x_gate_msa = &x_chunks[2];
        let x_shift_mlp = &x_chunks[3];
        let x_scale_mlp = &x_chunks[4];
        let x_gate_mlp = &x_chunks[5];

        let x_norm = self.layer_norm_no_affine(x)?;
        let x_mod = self.modulate(&x_norm, x_shift_msa, x_scale_msa)?;
        let (x_q, x_k, x_v) = self.pre_attention_qkv(&x_mod, &x_prefix, "attn")?;

        // For dual attention: also prepare the second modulated input
        let x_mod2 = if block_has_dual {
            let x_shift_msa2 = &x_chunks[6];
            let x_scale_msa2 = &x_chunks[7];
            // norm is shared: norm_hidden_states2 = norm * (1 + scale2) + shift2
            Some(self.modulate(&x_norm, x_shift_msa2, x_scale_msa2)?)
        } else {
            None
        };
        let x_gate_msa2 = if block_has_dual {
            Some(x_chunks[8].clone())
        } else {
            None
        };

        // ------ Joint attention: concatenate context + x along sequence dim ------
        let q = Tensor::cat(&[&ctx_q, &x_q], 2)?;
        let k = Tensor::cat(&[&ctx_k, &x_k], 2)?;
        let v = Tensor::cat(&[&ctx_v, &x_v], 2)?;

        let attn_out = sdpa(&q, &k, &v, None)?;

        // Split back into context and x portions
        let n_ctx = ctx_q.shape().dims()[2];
        let n_x = x_q.shape().dims()[2];
        let batch = attn_out.shape().dims()[0];

        let ctx_attn = attn_out.narrow(2, 0, n_ctx)?;
        let x_attn = attn_out.narrow(2, n_ctx, n_x)?;

        // Reshape from [B, H, N, D] -> [B, N, H*D]
        let ctx_attn = ctx_attn.permute(&[0, 2, 1, 3])?;
        let ctx_attn = ctx_attn.reshape(&[batch, n_ctx, hidden])?;
        let x_attn = x_attn.permute(&[0, 2, 1, 3])?;
        let x_attn = x_attn.reshape(&[batch, n_x, hidden])?;

        // ------ Context post-attention ------
        let context_out = if let Some((ctx_residual, gate_msa, shift_mlp, scale_mlp, gate_mlp)) =
            ctx_intermediates
        {
            // Proj
            let ctx_proj = self.linear_with_bias(
                &ctx_attn,
                &format!("{ctx_prefix}.attn.proj.weight"),
                &format!("{ctx_prefix}.attn.proj.bias"),
            )?;
            // gate_msa * proj + residual
            let gated = gate_msa.unsqueeze(1)?.mul(&ctx_proj)?;
            let ctx_out = ctx_residual.add(&gated)?;

            // MLP
            let ctx_norm2 = self.layer_norm_no_affine(&ctx_out)?;
            let ctx_mlp_in = self.modulate(&ctx_norm2, &shift_mlp, &scale_mlp)?;
            let ctx_mlp = self.gelu_mlp(&ctx_mlp_in, &ctx_prefix)?;
            let ctx_gated = gate_mlp.unsqueeze(1)?.mul(&ctx_mlp)?;
            Some(ctx_out.add(&ctx_gated)?)
        } else {
            // pre_only: no post-attention for context
            None
        };

        // ------ X post-attention ------
        let x_proj = self.linear_with_bias(
            &x_attn,
            &format!("{x_prefix}.attn.proj.weight"),
            &format!("{x_prefix}.attn.proj.bias"),
        )?;
        let x_gated = x_gate_msa.unsqueeze(1)?.mul(&x_proj)?;
        let mut x_out = x.add(&x_gated)?;

        // ------ X dual attention (attn2, SD3.5 Medium only) ------
        // Second self-attention: x-stream only, no context concatenation.
        // Python: attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
        //         hidden_states = hidden_states + gate_msa2 * attn_output2
        if let (Some(x_mod2), Some(gate_msa2)) = (x_mod2, x_gate_msa2) {
            let (q2, k2, v2) = self.pre_attention_qkv(&x_mod2, &x_prefix, "attn2")?;
            let attn2_out = sdpa(&q2, &k2, &v2, None)?;
            let attn2_out = attn2_out.permute(&[0, 2, 1, 3])?;
            let attn2_out = attn2_out.reshape(&[batch, n_x, hidden])?;
            let attn2_proj = self.linear_with_bias(
                &attn2_out,
                &format!("{x_prefix}.attn2.proj.weight"),
                &format!("{x_prefix}.attn2.proj.bias"),
            )?;
            let attn2_gated = gate_msa2.unsqueeze(1)?.mul(&attn2_proj)?;
            x_out = x_out.add(&attn2_gated)?;
        }

        // X MLP
        let x_norm2 = self.layer_norm_no_affine(&x_out)?;
        let x_mlp_in = self.modulate(&x_norm2, x_shift_mlp, x_scale_mlp)?;
        let x_mlp = self.gelu_mlp(&x_mlp_in, &x_prefix)?;
        let x_mlp_gated = x_gate_mlp.unsqueeze(1)?.mul(&x_mlp)?;
        let x_out = x_out.add(&x_mlp_gated)?;

        Ok((context_out, x_out))
    }

    // -- GELU MLP (fc1 -> GELU(tanh) -> fc2) ----------------------------------

    fn gelu_mlp(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = self.linear_with_bias(
            x,
            &format!("{prefix}.mlp.fc1.weight"),
            &format!("{prefix}.mlp.fc1.bias"),
        )?;
        let h = h.gelu()?;
        self.linear_with_bias(
            &h,
            &format!("{prefix}.mlp.fc2.weight"),
            &format!("{prefix}.mlp.fc2.bias"),
        )
    }

    // -- Final layer ----------------------------------------------------------

    fn final_layer(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let x_norm = self.layer_norm_no_affine(x)?;

        // adaLN: SiLU(c) -> Linear -> chunk(2) -> (shift, scale)
        let c_silu = c.silu()?;
        let mods = self.linear_with_bias(
            &c_silu,
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
        )?;
        let chunks = mods.chunk(2, mods.shape().dims().len() - 1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];

        let x_mod = self.modulate(&x_norm, shift, scale)?;

        // Final linear projection
        self.linear_with_bias(
            &x_mod,
            "final_layer.linear.weight",
            "final_layer.linear.bias",
        )
    }

    // -- Full forward pass ----------------------------------------------------

    /// Forward pass.
    ///
    /// Args:
    ///   x: Latent image tensor [B, C, H, W] (NCHW, BF16)
    ///   timestep: Timestep tensor [B] (f32 values, e.g. 0..1000)
    ///   encoder_hidden_states: Text hidden states [B, N_ctx, context_dim] (BF16)
    ///   pooled_projections: Pooled text embedding [B, pooled_dim] (BF16)
    ///
    /// Returns:
    ///   Noise prediction [B, C, H, W]
    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        pooled_projections: &Tensor,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (h, w) = (x_dims[2], x_dims[3]);

        // Patch embed + position embed
        let x_tokens = self.patch_embed(x)?;
        let pos_embed = self.cropped_pos_embed(h, w)?;
        let mut x_tokens = x_tokens.add(&pos_embed)?;

        // Conditioning vector: timestep + pooled text
        let t_emb = self.timestep_embed(timestep)?;
        let y_emb = self.pooled_embed(pooled_projections)?;
        let c = t_emb.add(&y_emb)?;

        // Context embedding
        let mut ctx_tokens = self.linear_with_bias(
            encoder_hidden_states,
            "context_embedder.weight",
            "context_embedder.bias",
        )?;

        // Joint blocks — stream each from disk
        let depth = self.config.depth;
        for i in 0..depth {
            let is_last = i == depth - 1;
            let prefix = format!("joint_blocks.{i}");
            println!("  SD3 block {i}/{depth}");
            self.load_block(&prefix)?;

            let (new_ctx, new_x) =
                self.joint_block(&ctx_tokens, &x_tokens, &c, i, is_last)?;

            x_tokens = new_x;
            if let Some(ctx) = new_ctx {
                ctx_tokens = ctx;
            }
            // If pre_only (last block), ctx_tokens is stale but won't be used again

            self.unload_block();
        }

        // Final layer
        let x_out = self.final_layer(&x_tokens, &c)?;

        // Unpatchify to spatial
        let x_spatial = self.unpatchify(&x_out, h, w)?;

        // Crop to original size (in case padding was added by conv)
        let out_dims = x_spatial.shape().dims().to_vec();
        if out_dims[2] != h || out_dims[3] != w {
            let cropped = x_spatial.narrow(2, 0, h)?;
            cropped.narrow(3, 0, w)
        } else {
            Ok(x_spatial)
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Transpose a 2D tensor [M, N] -> [N, M]
fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    t.permute(&[1, 0])
}

// ---------------------------------------------------------------------------
// Resident weight loading
// ---------------------------------------------------------------------------

/// Load resident (non-block) weights for SD3 MMDiT from a safetensors file.
///
/// Resident weights are small enough to keep on GPU permanently:
/// - pos_embed, x_embedder.*, t_embedder.*, y_embedder.*, context_embedder.*
/// - final_layer.*
///
/// Handles optional `model.diffusion_model.` prefix in combined checkpoints.
pub fn load_sd3_resident(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let resident_suffixes = [
        "pos_embed",
        "x_embedder.",
        "t_embedder.",
        "y_embedder.",
        "context_embedder.",
        "final_layer.",
    ];
    let prefix = "model.diffusion_model.";

    let raw = load_file_filtered(model_path, device, |key| {
        let stripped = key.strip_prefix(prefix).unwrap_or(key);
        resident_suffixes
            .iter()
            .any(|p| stripped.starts_with(p) || stripped == "pos_embed")
    })?;

    // Strip prefix and cast to BF16
    let mut weights = HashMap::with_capacity(raw.len());
    for (key, val) in raw {
        let k = key.strip_prefix(prefix).unwrap_or(&key).to_string();
        let val = if val.dtype() != DType::BF16 { val.to_dtype(DType::BF16)? } else { val };
        weights.insert(k, val);
    }
    Ok(weights)
}

/// Load ALL weights (resident + blocks) for models that fit in VRAM
/// (e.g. SD3.5 Medium at ~5GB). Strips `model.diffusion_model.` prefix.
pub fn load_sd3_all(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let prefix = "model.diffusion_model.";
    let mut weights = load_file_filtered(model_path, device, |key| {
        key.starts_with(prefix) || key == "pos_embed"
    })?;
    // Strip prefix and cast to BF16
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        let stripped = key.strip_prefix(prefix).unwrap_or(&key).to_string();
        let t = weights.remove(&key).unwrap();
        let t = if t.dtype() != DType::BF16 { t.to_dtype(DType::BF16)? } else { t };
        weights.insert(stripped, t);
    }
    Ok(weights)
}

/// Load ALL DiT weights block-by-block to avoid peak memory from F32 intermediates.
/// Loads resident weights first, then each block prefix one at a time, casting to BF16
/// immediately and accumulating into a single HashMap.
pub fn load_sd3_all_chunked(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let prefix = "model.diffusion_model.";

    // 1. Load resident (non-block) weights
    let mut all = load_sd3_resident(model_path, device)?;
    println!("  Resident: {} keys", all.len());

    // 2. Detect depth from resident config
    let config = SD3Config::from_weights(&all);
    let depth = config.depth;

    // 3. Load each block one at a time, cast to BF16, accumulate
    for i in 0..depth {
        let block_prefix = format!("{prefix}joint_blocks.{i}.");
        let block = load_file_filtered(model_path, device, |key| {
            key.starts_with(&block_prefix)
        })?;
        for (key, val) in block {
            let k = key.strip_prefix(prefix).unwrap_or(&key).to_string();
            let val = if val.dtype() != DType::BF16 { val.to_dtype(DType::BF16)? } else { val };
            all.insert(k, val);
        }
        if (i + 1) % 10 == 0 || i == depth - 1 {
            println!("  Loaded block {}/{}", i + 1, depth);
        }
    }

    Ok(all)
}
