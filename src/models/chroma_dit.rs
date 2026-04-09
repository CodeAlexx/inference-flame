//! Chroma DiT — pure Rust, flame-core + FlameSwap.
//!
//! Chroma is a FLUX.1-schnell derivative (8.9B params, Apache 2.0, by lodestones).
//! Same architecture as FLUX (19 double + 38 single, dim 3072, 24 heads, head_dim
//! 128) with three surgical changes:
//!
//!   1. **No CLIP / no pooled vector / no guidance_in**. T5-XXL only.
//!   2. **Per-block `img_mod` / `txt_mod` / `modulation` removed**. All
//!      modulation parameters come from a single shared `distilled_guidance_layer`
//!      (the "approximator"), which runs once per step on the timestep and emits
//!      a `[B, mod_index, inner_dim]` tensor that gets sliced per-block.
//!   3. **Separate Q/K/V projections**. Where FLUX uses `img_attn.qkv`, Chroma
//!      uses `transformer_blocks.{i}.attn.{to_q,to_k,to_v}` and
//!      `attn.{add_q_proj,add_k_proj,add_v_proj}` for the txt stream.
//!
//! ## Diffusers reference
//! `diffusers/src/diffusers/models/transformers/transformer_chroma.py`
//!
//! ## Checkpoint key layout (`Chroma1-HD.safetensors`)
//!
//! Top-level (shared):
//!   x_embedder.{weight,bias}                       [3072, 64] / [3072]
//!   context_embedder.{weight,bias}                 [3072, 4096] / [3072]
//!   proj_out.{weight,bias}                         [64, 3072] / [64]
//!   distilled_guidance_layer.in_proj.{weight,bias} [5120, 64] / [5120]
//!   distilled_guidance_layer.layers.{0..4}.linear_1.{weight,bias}  [5120, 5120]
//!   distilled_guidance_layer.layers.{0..4}.linear_2.{weight,bias}  [5120, 5120]
//!   distilled_guidance_layer.norms.{0..4}.weight                   [5120]
//!   distilled_guidance_layer.out_proj.{weight,bias}                [3072, 5120]
//!
//! Per double block `transformer_blocks.{i}`:
//!   attn.{to_q,to_k,to_v}.{weight,bias}            [3072, 3072]   img Q/K/V
//!   attn.{add_q_proj,add_k_proj,add_v_proj}.{weight,bias}        txt Q/K/V
//!   attn.{norm_q,norm_k,norm_added_q,norm_added_k}.weight        [128]
//!   attn.to_out.0.{weight,bias}                    [3072, 3072]   img out
//!   attn.to_add_out.{weight,bias}                  [3072, 3072]   txt out
//!   ff.net.{0.proj,2}.{weight,bias}                                img FFN (GELU)
//!   ff_context.net.{0.proj,2}.{weight,bias}                       txt FFN
//!   (NO img_mod, NO txt_mod — modulation comes from distilled_guidance_layer)
//!   (NO norm1, NO norm2 — LayerNorm without affine)
//!
//! Per single block `single_transformer_blocks.{i}`:
//!   attn.{to_q,to_k,to_v}.{weight,bias}            [3072, 3072]
//!   attn.{norm_q,norm_k}.weight                    [128]
//!   proj_mlp.{weight,bias}                         [12288, 3072]
//!   proj_out.{weight,bias}                         [3072, 15360]   (3072+12288)
//!   (NO modulation, NO norm)
//!
//! ## Modulation slicing
//! `distilled_guidance_layer` outputs `[B, mod_index_length, inner_dim]` where
//!   mod_index_length = 3*num_single + 2*6*num_double + 2
//!                    = 3*38 + 12*19 + 2
//!                    = 114 + 228 + 2 = 344
//!
//! Slicing matches the diffusers reference (transformer_chroma.py:560-570):
//!   - For double block i: img_mod = pooled_temb[:, 114+6i : 114+6i+6]
//!                         txt_mod = pooled_temb[:, 114+114+6i : 114+114+6i+6]
//!   - For single block i: mod = pooled_temb[:, 3i : 3i+3]
//!   - For norm_out:       mod = pooled_temb[:, -2:]
//!
//! ⚠️ STANDALONE — does NOT touch any other model.

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
pub struct ChromaConfig {
    pub num_double_blocks: usize,    // 19
    pub num_single_blocks: usize,    // 38
    pub inner_dim: usize,            // 3072
    pub num_heads: usize,            // 24
    pub head_dim: usize,             // 128
    pub in_channels: usize,          // 64
    pub joint_attention_dim: usize,  // 4096 (T5-XXL hidden)
    pub mlp_ratio: f32,              // 4.0
    pub axes_dims_rope: [usize; 3],  // (16, 56, 56)
    pub rope_theta: f64,             // 10000.0
    pub approximator_in_channels: usize,   // 64 (4 * 16 = time_proj + guidance_proj + 32-dim mod_proj)
    pub approximator_hidden_dim: usize,    // 5120
    pub approximator_num_layers: usize,    // 5
    pub eps: f32,                          // 1e-6
}

impl Default for ChromaConfig {
    fn default() -> Self {
        Self {
            num_double_blocks: 19,
            num_single_blocks: 38,
            inner_dim: 3072,
            num_heads: 24,
            head_dim: 128,
            in_channels: 64,
            joint_attention_dim: 4096,
            mlp_ratio: 4.0,
            axes_dims_rope: [16, 56, 56],
            rope_theta: 10000.0,
            approximator_in_channels: 64,
            approximator_hidden_dim: 5120,
            approximator_num_layers: 5,
            eps: 1e-6,
        }
    }
}

impl ChromaConfig {
    /// Number of slices in the distilled_guidance_layer output:
    ///   3 per single block + 2*6 per double block + 2 (norm_out)
    pub fn mod_index_length(&self) -> usize {
        3 * self.num_single_blocks + 2 * 6 * self.num_double_blocks + 2
    }
}

// ---------------------------------------------------------------------------
// ChromaDit struct
// ---------------------------------------------------------------------------

/// Chroma DiT transformer.
///
/// Double blocks (`transformer_blocks.{i}`) and single blocks
/// (`single_transformer_blocks.{i}`) are managed by FlameSwap.
/// Shared weights (x_embedder, context_embedder, proj_out, distilled_guidance_layer)
/// stay GPU-resident.
pub struct ChromaDit {
    shared: HashMap<String, Tensor>,
    swap: FlameSwap,
    config: ChromaConfig,
    device: Arc<CudaDevice>,

    /// Cached `mod_proj` buffer used by the time_text_embed step. Computed
    /// once at load (no trainable params), shape `[mod_index_length, 32]` BF16.
    mod_proj_buf: Tensor,
}

impl ChromaDit {
    /// Load Chroma from a single-file safetensors checkpoint
    /// (e.g. `Chroma1-HD.safetensors`) or a sharded diffusers checkpoint.
    pub fn load(
        checkpoint_paths: &[&str],
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = ChromaConfig::default();
        let num_double = config.num_double_blocks;

        // FlameSwap classifies blocks by name:
        //   transformer_blocks.{i}        → block i
        //   single_transformer_blocks.{i} → block (num_double + i)
        let swap = FlameSwap::load(
            checkpoint_paths,
            device,
            |name| {
                if let Some(rest) = name.strip_prefix("transformer_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(idx);
                }
                if let Some(rest) = name.strip_prefix("single_transformer_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(num_double + idx);
                }
                None
            },
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap Chroma: {e}")))?;

        // Shared weights (everything that's NOT inside a block).
        let shared_prefixes = [
            "x_embedder.",
            "context_embedder.",
            "proj_out.",
            "distilled_guidance_layer.",
        ];
        let mut shared_weights = HashMap::new();
        for path in checkpoint_paths {
            let part = load_file_filtered(Path::new(path), device, |key| {
                shared_prefixes.iter().any(|p| key.starts_with(p))
            })?;
            for (k, v) in part {
                shared_weights.insert(k, v);
            }
        }

        log::info!(
            "[Chroma] Loaded: {} blocks via FlameSwap, {} shared weights",
            swap.num_blocks(),
            shared_weights.len()
        );

        // Pre-compute the mod_proj buffer used by time_text_embed.
        // Reference (transformer_chroma.py:159-165):
        //   mod_proj = get_timestep_embedding(
        //       torch.arange(out_dim) * 1000, 2 * num_channels, ...)
        // where out_dim = mod_index_length, num_channels = approximator_in_channels // 4 = 16,
        // so 2 * num_channels = 32. Output shape: [out_dim, 32].
        let mod_index_length = config.mod_index_length();
        let mod_proj_channels = 2 * (config.approximator_in_channels / 4);
        let mod_proj_buf = build_mod_proj_buffer(
            mod_index_length,
            mod_proj_channels,
            device,
        )?;

        Ok(Self {
            shared: shared_weights,
            swap,
            config,
            device: device.clone(),
            mod_proj_buf,
        })
    }

    pub fn config(&self) -> &ChromaConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Helpers (mirror of flux1_dit helpers)
    // -----------------------------------------------------------------------

    /// Flatten [B, N, C] → [1, B*N, C], call fused_linear3d, reshape back.
    /// `fused_linear3d_bf16_native` uses cuBLASLt strided batched GEMM with
    /// BATCH_COUNT=1 on the weight and BATCH_COUNT=B on input/output, which
    /// cuBLASLt treats as invalid when B>1 (error 7). Linear is a per-position
    /// op so flattening is mathematically identical and works for any B.
    fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() == 3 && dims[0] > 1 {
            let (b, n, c) = (dims[0], dims[1], dims[2]);
            let flat = x.reshape(&[1, b * n, c])?;
            let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, Some(bias))?;
            let out_c = weight.shape().dims()[0];
            out.reshape(&[b, n, out_c])
        } else {
            flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
        }
    }

    fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() == 3 && dims[0] > 1 {
            let (b, n, c) = (dims[0], dims[1], dims[2]);
            let flat = x.reshape(&[1, b * n, c])?;
            let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, None)?;
            let out_c = weight.shape().dims()[0];
            out.reshape(&[b, n, out_c])
        } else {
            flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
        }
    }

    fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
        out.reshape(&dims)
    }

    /// LayerNorm with no affine — used by Chroma's normalization paths
    /// (norm1, norm2 in double block; norm in single block; norm_out).
    fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
        out.reshape(&dims)
    }

    /// Modulate: out = LayerNorm(x) * (1 + scale) + shift
    /// `scale` and `shift` are `[B, dim]`, broadcast over the seq dim.
    ///
    /// Fused path: `flame_core::bf16_ops::modulate_pre_fused_bf16` does
    /// LayerNorm → (1 + scale) → shift in a single kernel. Replaces the
    /// 4-kernel unfused version (layer_norm → add_scalar → mul → add).
    fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::modulate_pre_fused_bf16(x, shift, scale, 1e-6)
    }

    /// Sinusoidal `Timesteps` embedding (FLUX/diffusers style:
    /// `flip_sin_to_cos=True`, `downscale_freq_shift=0`). Returns `[B, dim]` BF16.
    fn timesteps_embedding(t: &Tensor, dim: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let in_dtype = t.dtype();
        let t_f32 = t.to_dtype(DType::F32)?;
        let half = dim / 2;
        let max_period = 10000.0f64;
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), device.clone())?;
        let t_col = t_f32.unsqueeze(1)?;
        let args = t_col.matmul(&freqs)?; // [B, half] F32
        // FLUX/Chroma use flip_sin_to_cos=True → concat(cos, sin)
        let cos = args.cos()?;
        let sin = args.sin()?;
        let emb = Tensor::cat(&[&cos, &sin], 1)?;
        emb.to_dtype(in_dtype)
    }

    // -----------------------------------------------------------------------
    // distilled_guidance_layer (the "approximator")
    // -----------------------------------------------------------------------
    //
    //   in_proj: Linear(in_dim, hidden_dim)
    //   for i in 0..n_layers:
    //       x = x + linear_2(silu(linear_1(rms_norm(x))))   (PixArtAlphaTextProjection)
    //   out_proj: Linear(hidden_dim, out_dim)
    //
    // The norms are RMSNorm with weight only.

    /// Run the approximator on a `[B, mod_index_length, in_channels]` input,
    /// returning `[B, mod_index_length, inner_dim]`.
    fn approximator_forward(&self, x: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        // in_proj
        let in_w = self.shared.get("distilled_guidance_layer.in_proj.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing distilled_guidance_layer.in_proj.weight".into()))?;
        let in_b = self.shared.get("distilled_guidance_layer.in_proj.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing distilled_guidance_layer.in_proj.bias".into()))?;
        let mut x = Self::linear_bias(x, in_w, in_b)?;

        // Residual block stack
        for i in 0..cfg.approximator_num_layers {
            let norm_w = self.shared.get(&format!("distilled_guidance_layer.norms.{i}.weight"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing distilled_guidance_layer.norms.{i}.weight")))?;
            let l1_w = self.shared.get(&format!("distilled_guidance_layer.layers.{i}.linear_1.weight"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing layer {i}.linear_1.weight")))?;
            let l1_b = self.shared.get(&format!("distilled_guidance_layer.layers.{i}.linear_1.bias"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing layer {i}.linear_1.bias")))?;
            let l2_w = self.shared.get(&format!("distilled_guidance_layer.layers.{i}.linear_2.weight"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing layer {i}.linear_2.weight")))?;
            let l2_b = self.shared.get(&format!("distilled_guidance_layer.layers.{i}.linear_2.bias"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing layer {i}.linear_2.bias")))?;

            let n = Self::rms_norm(&x, norm_w, 1e-6)?;
            let h = Self::linear_bias(&n, l1_w, l1_b)?;
            let h = h.silu()?;
            let h = Self::linear_bias(&h, l2_w, l2_b)?;
            x = x.add(&h)?;
        }

        // out_proj
        let out_w = self.shared.get("distilled_guidance_layer.out_proj.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing distilled_guidance_layer.out_proj.weight".into()))?;
        let out_b = self.shared.get("distilled_guidance_layer.out_proj.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing distilled_guidance_layer.out_proj.bias".into()))?;
        Self::linear_bias(&x, out_w, out_b)
    }

    /// Build the input vector to the approximator from the timestep, mirroring
    /// `ChromaCombinedTimestepTextProjEmbeddings.forward`. Output:
    /// `[B, mod_index_length, approximator_in_channels]` BF16.
    fn build_approximator_input(&self, timestep: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let device = &self.device;
        let batch_size = timestep.shape().dims()[0];
        let mod_index_length = cfg.mod_index_length();
        let num_channels = cfg.approximator_in_channels / 4; // 16

        // timesteps_proj: [B, 2*num_channels] BF16
        let t_scaled = timestep.to_dtype(DType::F32)?.mul_scalar(1000.0)?
            .to_dtype(DType::BF16)?;
        let timesteps_proj = Self::timesteps_embedding(&t_scaled, 2 * num_channels, device)?;

        // guidance_proj: same kernel applied to a zero scalar (Chroma always passes 0).
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size]),
            DType::BF16,
            device.clone(),
        )?;
        let guidance_proj = Self::timesteps_embedding(&zeros, 2 * num_channels, device)?;

        // Concat along the channel dim: [B, 4*num_channels]
        let timestep_guidance = Tensor::cat(&[&timesteps_proj, &guidance_proj], 1)?;
        // Broadcast over mod_index_length: [B, 1, 4*num_channels] → [B, L, 4*num_channels]
        let tg_3d = timestep_guidance.unsqueeze(1)?;
        let tg_expanded = tg_3d.expand(&[batch_size, mod_index_length, 4 * num_channels])?;

        // mod_proj_buf: [L, 2*num_channels] → [1, L, 2*num_channels] → expand
        let mod_proj_3d = self.mod_proj_buf.unsqueeze(0)?;
        let mod_proj_exp = mod_proj_3d.expand(&[batch_size, mod_index_length, 2 * num_channels])?;

        // Final input_vec: [B, L, 4*num_channels + 2*num_channels = 6*num_channels = approximator_in_channels]
        // Wait — approximator_in_channels = 64 = 4*16 = 4*num_channels. But the
        // diffusers source concats timestep_guidance (4*num_channels) with mod_proj
        // (2*num_channels), giving 6*num_channels = 96. That doesn't match
        // approximator_in_channels = 64.
        //
        // Re-reading transformer_chroma.py more carefully:
        //   num_channels = approximator_num_channels // 4   # 16
        //   timesteps_proj has num_channels=16 channels (not 2*16)
        //   guidance_proj has num_channels=16 channels
        //   mod_proj has 2*num_channels=32 channels
        //   timestep_guidance = cat(timesteps_proj, guidance_proj, dim=1) = 32
        //   input_vec = cat(timestep_guidance, mod_proj, dim=-1) = 32 + 32 = 64 ✓
        //
        // So timesteps_embedding should produce `num_channels` (16) outputs, NOT
        // 2*num_channels. The line above is wrong — fix it.
        //
        // (This comment intentionally left as a paper-trail; the actual code below
        // uses `num_channels`, not `2*num_channels`.)
        Tensor::cat(&[&tg_expanded, &mod_proj_exp], 2)
    }

    /// Re-do build_approximator_input with the correct channel counts. Wraps the
    /// staging above; returns `[B, L, approximator_in_channels]`.
    fn approximator_input(&self, timestep: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let device = &self.device;
        let batch_size = timestep.shape().dims()[0];
        let mod_index_length = cfg.mod_index_length();
        let num_channels = cfg.approximator_in_channels / 4; // 16

        // timesteps_proj: [B, num_channels]
        let t_scaled = timestep.to_dtype(DType::F32)?.mul_scalar(1000.0)?
            .to_dtype(DType::BF16)?;
        let timesteps_proj = Self::timesteps_embedding(&t_scaled, num_channels, device)?;
        // guidance_proj: [B, num_channels] from zeros
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size]),
            DType::BF16,
            device.clone(),
        )?;
        let guidance_proj = Self::timesteps_embedding(&zeros, num_channels, device)?;

        // [B, 2*num_channels]
        let tg = Tensor::cat(&[&timesteps_proj, &guidance_proj], 1)?;
        // [B, 1, 2*num_channels] → [B, L, 2*num_channels]
        let tg_exp = tg.unsqueeze(1)?
            .expand(&[batch_size, mod_index_length, 2 * num_channels])?;

        // mod_proj_buf is [L, 2*num_channels] → [1, L, 2*num_channels] → expand
        let mp = self.mod_proj_buf.unsqueeze(0)?
            .expand(&[batch_size, mod_index_length, 2 * num_channels])?;

        // [B, L, 4*num_channels = approximator_in_channels]
        Tensor::cat(&[&tg_exp, &mp], 2)
    }

    // -----------------------------------------------------------------------
    // RoPE (3-axis, identical to FLUX)
    // -----------------------------------------------------------------------

    pub fn build_rope_2d(
        img_ids: &Tensor,
        txt_ids: &Tensor,
        config: &ChromaConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
        let all_ids_f32 = all_ids.to_dtype(DType::F32)?;

        let mut cos_parts: Vec<Tensor> = Vec::with_capacity(config.axes_dims_rope.len());
        let mut sin_parts: Vec<Tensor> = Vec::with_capacity(config.axes_dims_rope.len());

        for (axis, &axis_dim) in config.axes_dims_rope.iter().enumerate() {
            assert!(axis_dim % 2 == 0, "rope axis dim must be even");
            let half = axis_dim / 2;

            let omega_data: Vec<f32> = (0..half)
                .map(|i| {
                    let scale = (2 * i) as f64 / axis_dim as f64;
                    (1.0 / config.rope_theta.powf(scale)) as f32
                })
                .collect();
            let omega = Tensor::from_vec(omega_data, Shape::from_dims(&[1, half]), device.clone())?;

            let pos = all_ids_f32.narrow(1, axis, 1)?.squeeze(Some(1))?;
            let pos_col = pos.unsqueeze(1)?;
            let angles = pos_col.matmul(&omega)?;

            cos_parts.push(angles.cos()?);
            sin_parts.push(angles.sin()?);
        }

        let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
        let cos_full = Tensor::cat(&cos_refs, 1)?;
        let sin_full = Tensor::cat(&sin_refs, 1)?;

        let pe_cos = cos_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let pe_sin = sin_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((pe_cos, pe_sin))
    }

    fn apply_rope_complex(q: &Tensor, k: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_out = flame_core::bf16_ops::rope_fused_bf16(q, pe_cos, pe_sin)?;
        let k_out = flame_core::bf16_ops::rope_fused_bf16(k, pe_cos, pe_sin)?;
        Ok((q_out, k_out))
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Forward pass.
    ///
    /// ## Arguments
    /// - `img`: packed image latents `[B, N_img, in_channels]`
    /// - `txt`: T5 hidden states `[B, N_txt, joint_attention_dim]`
    /// - `timesteps`: `[B]` sigmas in `[0, 1]`
    /// - `img_ids`: `[N_img, 3]`
    /// - `txt_ids`: `[N_txt, 3]`
    /// Precompute per-step tensors that are constant across the CFG loop
    /// (cond and uncond forwards share the same `timesteps`/`img_ids`/`txt_ids`).
    ///
    /// Returns `(pooled_temb, pe_cos, pe_sin)`. Pass to `forward_cached` to
    /// skip ~10 approximator matmuls and 1 RoPE table build on the second
    /// CFG forward.
    pub fn precompute_step_cache(
        &self,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let approx_in = self.approximator_input(timesteps)?;
        let pooled_temb = self.approximator_forward(&approx_in)?;
        let (pe_cos, pe_sin) = Self::build_rope_2d(img_ids, txt_ids, &self.config, &self.device)?;
        Ok((pooled_temb, pe_cos, pe_sin))
    }

    pub fn forward(
        &mut self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<Tensor> {
        // Back-compat wrapper: compute step cache inline then delegate.
        let (pooled_temb, pe_cos, pe_sin) =
            self.precompute_step_cache(timesteps, img_ids, txt_ids)?;
        self.forward_cached(img, txt, &pooled_temb, &pe_cos, &pe_sin)
    }

    /// Forward with externally-computed per-step cache. Use in CFG loops:
    /// call `precompute_step_cache` once per step, then `forward_cached`
    /// twice (cond + uncond) reusing the same cache.
    pub fn forward_cached(
        &mut self,
        img: &Tensor,
        txt: &Tensor,
        pooled_temb: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = self.config.clone();
        let _dim = cfg.inner_dim;
        let img_len = img.shape().dims()[1];
        let _txt_len = txt.shape().dims()[1];

        // --- Input projections ---
        let img_in_w = self.shared.get("x_embedder.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing x_embedder.weight".into()))?;
        let img_in_b = self.shared.get("x_embedder.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing x_embedder.bias".into()))?;
        let mut img = Self::linear_bias(img, img_in_w, img_in_b)?;

        let ctx_w = self.shared.get("context_embedder.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing context_embedder.weight".into()))?;
        let ctx_b = self.shared.get("context_embedder.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing context_embedder.bias".into()))?;
        let mut txt = Self::linear_bias(txt, ctx_w, ctx_b)?;

        // --- Block indexing constants (layout from diffusers transformer_chroma.py) ---
        //   single block i:  [3*i : 3*i + 3]                          for i in 0..38
        //   double img i:    [114 + 6*i : 114 + 6*i + 6]               for i in 0..19
        //   double txt i:    [114 + 114 + 6*i : 114 + 114 + 6*i + 6]   for i in 0..19
        //   norm_out:        [-2:]
        let img_offset_dbl = 3 * cfg.num_single_blocks;     // 114
        let txt_offset_dbl = img_offset_dbl + 6 * cfg.num_double_blocks; // 228

        // --- Double blocks ---
        let total_blocks = cfg.num_double_blocks + cfg.num_single_blocks;
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..cfg.num_double_blocks {
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            // Slice modulation params for this block.
            // pooled_temb is [B, L, inner_dim].
            let img_mod = pooled_temb.narrow(1, img_offset_dbl + 6 * i, 6)?; // [B, 6, dim]
            let txt_mod = pooled_temb.narrow(1, txt_offset_dbl + 6 * i, 6)?; // [B, 6, dim]

            let (new_img, new_txt) = self.double_block_forward(
                &img, &txt, &img_mod, &txt_mod, pe_cos, pe_sin, &raw, i,
            )?;
            img = new_img;
            txt = new_txt;
            if i % 5 == 0 || i == cfg.num_double_blocks - 1 {
                log::info!("[Chroma] Double block {}/{}", i + 1, cfg.num_double_blocks);
            }
        }

        // --- Concat for single blocks: [B, N_txt + N_img, dim] ---
        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len_after = txt.shape().dims()[1];

        for i in 0..cfg.num_single_blocks {
            let block_idx = cfg.num_double_blocks + i;
            let raw = self.swap.await_block(block_idx)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if block_idx + 1 < total_blocks {
                self.swap.prefetch(block_idx + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            let single_mod = pooled_temb.narrow(1, 3 * i, 3)?; // [B, 3, dim]
            x = self.single_block_forward(&x, &single_mod, pe_cos, pe_sin, &raw, i)?;

            if i % 10 == 0 || i == cfg.num_single_blocks - 1 {
                log::info!("[Chroma] Single block {}/{}", i + 1, cfg.num_single_blocks);
            }
        }

        // --- Extract image portion (drop the txt prefix) and apply norm_out ---
        let img_out = x.narrow(1, txt_len_after, img_len)?;
        // norm_out: ChromaAdaLayerNormContinuousPruned, no affine, no params.
        // Last 2 entries of pooled_temb provide shift, scale.
        let mod_out = pooled_temb.narrow(1, cfg.mod_index_length() - 2, 2)?; // [B, 2, dim]
        let shift = mod_out.narrow(1, 0, 1)?.squeeze(Some(1))?; // [B, dim]
        let scale = mod_out.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let normed = Self::layer_norm_no_affine(&img_out, 1e-6)?;
        let one_plus = scale.add_scalar(1.0)?;
        let scaled = normed.mul(&one_plus.unsqueeze(1)?)?;
        let modulated = scaled.add(&shift.unsqueeze(1)?)?;

        // proj_out
        let proj_w = self.shared.get("proj_out.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing proj_out.weight".into()))?;
        let proj_b = self.shared.get("proj_out.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing proj_out.bias".into()))?;
        Self::linear_bias(&modulated, proj_w, proj_b)
    }

    // -----------------------------------------------------------------------
    // Block forwards
    // -----------------------------------------------------------------------

    /// Double block forward. `img_mod` and `txt_mod` are `[B, 6, dim]` slices
    /// of the approximator output. The 6 entries map to:
    ///   [0]=shift_msa  [1]=scale_msa  [2]=gate_msa
    ///   [3]=shift_mlp  [4]=scale_mlp  [5]=gate_mlp
    /// (matching the diffusers `ChromaAdaLayerNormZeroPruned.forward` chunking.)
    fn double_block_forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &Tensor,
        txt_mod: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let prefix = format!("transformer_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let dims_img = img.shape().dims().to_vec();
        let dims_txt = txt.shape().dims().to_vec();
        let (b, n_img) = (dims_img[0], dims_img[1]);
        let n_txt = dims_txt[1];

        // ── Slice modulation params ──
        let img_shift_msa = img_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
        let img_scale_msa = img_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let img_gate_msa  = img_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;
        let img_shift_mlp = img_mod.narrow(1, 3, 1)?.squeeze(Some(1))?;
        let img_scale_mlp = img_mod.narrow(1, 4, 1)?.squeeze(Some(1))?;
        let img_gate_mlp  = img_mod.narrow(1, 5, 1)?.squeeze(Some(1))?;

        let txt_shift_msa = txt_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
        let txt_scale_msa = txt_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let txt_gate_msa  = txt_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;
        let txt_shift_mlp = txt_mod.narrow(1, 3, 1)?.squeeze(Some(1))?;
        let txt_scale_mlp = txt_mod.narrow(1, 4, 1)?.squeeze(Some(1))?;
        let txt_gate_mlp  = txt_mod.narrow(1, 5, 1)?.squeeze(Some(1))?;

        // ── 1. Modulate img and txt (norm + scale + shift) ──
        let img_normed = Self::modulate_pre(img, &img_shift_msa, &img_scale_msa)?;
        let txt_normed = Self::modulate_pre(txt, &txt_shift_msa, &txt_scale_msa)?;

        // ── 2. Separate Q/K/V projections ──
        // img stream: attn.to_q / to_k / to_v
        let img_q = Self::linear_bias(&img_normed, w("attn.to_q.weight")?, w("attn.to_q.bias")?)?;
        let img_k = Self::linear_bias(&img_normed, w("attn.to_k.weight")?, w("attn.to_k.bias")?)?;
        let img_v = Self::linear_bias(&img_normed, w("attn.to_v.weight")?, w("attn.to_v.bias")?)?;

        // txt stream: attn.add_q_proj / add_k_proj / add_v_proj
        let txt_q = Self::linear_bias(&txt_normed, w("attn.add_q_proj.weight")?, w("attn.add_q_proj.bias")?)?;
        let txt_k = Self::linear_bias(&txt_normed, w("attn.add_k_proj.weight")?, w("attn.add_k_proj.bias")?)?;
        let txt_v = Self::linear_bias(&txt_normed, w("attn.add_v_proj.weight")?, w("attn.add_v_proj.bias")?)?;

        // ── 3. Reshape to [B, N, H, D] then permute to [B, H, N, D] ──
        let img_q = img_q.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;
        let img_k = img_k.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;
        let img_v = img_v.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;
        let txt_q = txt_q.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;
        let txt_k = txt_k.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;
        let txt_v = txt_v.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;

        // ── 4. Q/K RMSNorm ──
        let img_q = Self::rms_norm(&img_q, w("attn.norm_q.weight")?, 1e-6)?;
        let img_k = Self::rms_norm(&img_k, w("attn.norm_k.weight")?, 1e-6)?;
        let txt_q = Self::rms_norm(&txt_q, w("attn.norm_added_q.weight")?, 1e-6)?;
        let txt_k = Self::rms_norm(&txt_k, w("attn.norm_added_k.weight")?, 1e-6)?;

        // ── 5. Concat txt + img along seq dim and apply RoPE ──
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;
        let (q, k) = Self::apply_rope_complex(&q, &k, pe_cos, pe_sin)?;

        // ── 6. SDPA ──
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // ── 7. Split back, permute, project out ──
        let txt_attn = attn_out.narrow(2, 0, n_txt)?;
        let img_attn = attn_out.narrow(2, n_txt, n_img)?;
        let img_attn = img_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img, h * d])?;
        let txt_attn = txt_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_txt, h * d])?;
        let img_attn = Self::linear_bias(&img_attn, w("attn.to_out.0.weight")?, w("attn.to_out.0.bias")?)?;
        let txt_attn = Self::linear_bias(&txt_attn, w("attn.to_add_out.weight")?, w("attn.to_add_out.bias")?)?;

        // ── 8. Gated residual (using gate_msa) — fused ──
        let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, &img_gate_msa, &img_attn)?;
        let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, &txt_gate_msa, &txt_attn)?;

        // ── 9. FFN: norm + (1+scale_mlp) + shift_mlp → linear → gelu → linear → gate ──
        let img_mlp_in = Self::modulate_pre(&img, &img_shift_mlp, &img_scale_mlp)?;
        let txt_mlp_in = Self::modulate_pre(&txt, &txt_shift_mlp, &txt_scale_mlp)?;

        let img_mlp = Self::linear_bias(&img_mlp_in, w("ff.net.0.proj.weight")?, w("ff.net.0.proj.bias")?)?;
        let img_mlp = img_mlp.gelu()?;
        let img_mlp = Self::linear_bias(&img_mlp, w("ff.net.2.weight")?, w("ff.net.2.bias")?)?;

        let txt_mlp = Self::linear_bias(&txt_mlp_in, w("ff_context.net.0.proj.weight")?, w("ff_context.net.0.proj.bias")?)?;
        let txt_mlp = txt_mlp.gelu()?;
        let txt_mlp = Self::linear_bias(&txt_mlp, w("ff_context.net.2.weight")?, w("ff_context.net.2.bias")?)?;

        let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, &img_gate_mlp, &img_mlp)?;
        let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, &txt_gate_mlp, &txt_mlp)?;

        Ok((img, txt))
    }

    /// Single block forward. `single_mod` is `[B, 3, dim]` containing
    /// (shift_msa, scale_msa, gate_msa).
    fn single_block_forward(
        &self,
        x: &Tensor,
        single_mod: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let dim = cfg.inner_dim;
        let mlp_hidden = (dim as f32 * cfg.mlp_ratio) as usize;
        let prefix = format!("single_transformer_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let dims = x.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // ── Slice modulation params ──
        let shift_msa = single_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
        let scale_msa = single_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let gate      = single_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;

        // ── 1. Modulate ──
        let x_normed = Self::modulate_pre(x, &shift_msa, &scale_msa)?;

        // ── 2. Q/K/V projections (separate, not fused) ──
        let q = Self::linear_bias(&x_normed, w("attn.to_q.weight")?, w("attn.to_q.bias")?)?;
        let k = Self::linear_bias(&x_normed, w("attn.to_k.weight")?, w("attn.to_k.bias")?)?;
        let v = Self::linear_bias(&x_normed, w("attn.to_v.weight")?, w("attn.to_v.bias")?)?;

        // ── 3. Reshape + permute ──
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;

        // ── 4. Q/K RMSNorm ──
        let q = Self::rms_norm(&q, w("attn.norm_q.weight")?, 1e-6)?;
        let k = Self::rms_norm(&k, w("attn.norm_k.weight")?, 1e-6)?;

        // ── 5. RoPE ──
        let (q, k) = Self::apply_rope_complex(&q, &k, pe_cos, pe_sin)?;

        // ── 6. SDPA ──
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        // ── 7. MLP path: proj_mlp → GELU ──
        let mlp_out = Self::linear_bias(&x_normed, w("proj_mlp.weight")?, w("proj_mlp.bias")?)?;
        let mlp_out = mlp_out.gelu()?;

        // ── 8. Concat attn + mlp → proj_out → gated residual ──
        let cat = Tensor::cat(&[&attn_out, &mlp_out], 2)?;
        let out = Self::linear_bias(&cat, w("proj_out.weight")?, w("proj_out.bias")?)?;

        let _ = mlp_hidden; // silence unused warning if mlp_ratio path differs
        flame_core::bf16_ops::gate_residual_fused_bf16(x, &gate, &out)
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Build the `mod_proj` buffer used by `ChromaCombinedTimestepTextProjEmbeddings`.
///
/// Reference (transformer_chroma.py:159-165):
///   self.register_buffer(
///       "mod_proj",
///       get_timestep_embedding(
///           torch.arange(out_dim) * 1000, 2 * num_channels,
///           flip_sin_to_cos=True, downscale_freq_shift=0
///       ),
///       persistent=False,
///   )
///
/// `get_timestep_embedding(t, dim, ...)` produces sinusoidal embeddings of
/// shape `[len(t), dim]`. With `flip_sin_to_cos=True` the order is
/// `cat([cos, sin], dim=-1)`. We compute on CPU and copy to GPU as BF16.
fn build_mod_proj_buffer(
    out_dim: usize,
    num_channels: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let half = num_channels / 2;
    let max_period = 10000.0f64;
    // freqs[i] = exp(-ln(max_period) * i / half)  — same as FLUX timestep_embedding
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
        .collect();

    let mut data = Vec::with_capacity(out_dim * num_channels);
    for idx in 0..out_dim {
        let t = (idx as f64) * 1000.0;
        // First half: cos(t * freq)
        for &f in &freqs {
            data.push((t as f32 * f).cos());
        }
        // Second half: sin(t * freq)
        for &f in &freqs {
            data.push((t as f32 * f).sin());
        }
    }

    let buf = Tensor::from_vec(
        data,
        Shape::from_dims(&[out_dim, num_channels]),
        device.clone(),
    )?;
    buf.to_dtype(DType::BF16)
}
