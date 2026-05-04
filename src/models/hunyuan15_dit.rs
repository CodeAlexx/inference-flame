//! HunyuanVideo 1.5 DiT — pure Rust, flame-core + BlockOffloader.
//!
//! Port of `HunyuanVideo_1_5_DiffusionTransformer` from the HunyuanVideo-1.5 repo.
//!
//! ## Architecture (480p T2V)
//! - **54 double-stream blocks** (no single-stream blocks)
//! - hidden_size = 2048, heads = 16, head_dim = 128
//! - in_channels = 32, out_channels = 32, patch_size = (1, 1, 1)
//! - With concat_condition: img_in takes 65 channels (32*2+1)
//! - rope_dim_list = [16, 56, 56], theta = 256
//! - text_dim = 3584 (Qwen2.5-VL), ByT5 mapper, SigLIP vision
//! - SingleTokenRefiner (2-block self-attn text preprocessing)
//!
//! ## Weight key patterns
//! ```
//! double_blocks.{0-53}.img_mod.linear.{weight,bias}    [12288, 2048]
//! double_blocks.{0-53}.img_attn_{q,k,v}.{weight,bias}  [2048, 2048]
//! double_blocks.{0-53}.img_attn_{q,k}_norm.weight       [128]
//! double_blocks.{0-53}.img_attn_proj.{weight,bias}      [2048, 2048]
//! double_blocks.{0-53}.img_mlp.fc{1,2}.{weight,bias}
//! double_blocks.{0-53}.txt_mod/attn/mlp.*  (mirror of img)
//! ```

use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_diffusion::block_offload::BlockFacilitator;
use flame_diffusion::BlockOffloader;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BlockFacilitator for HunyuanVideo: `double_blocks.{i}.*` → block i
// ---------------------------------------------------------------------------

struct Hunyuan15Facilitator {
    num_blocks: usize,
}

impl BlockFacilitator for Hunyuan15Facilitator {
    fn block_count(&self) -> usize { self.num_blocks }
    fn classify_key(&self, key: &str) -> Option<usize> {
        let rest = key.strip_prefix("double_blocks.")?;
        rest.split('.').next()?.parse().ok()
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Hunyuan15Config {
    pub num_double_blocks: usize,  // 54
    pub hidden_size: usize,        // 2048
    pub num_heads: usize,          // 16
    pub head_dim: usize,           // 128
    pub in_channels: usize,        // 32
    pub out_channels: usize,       // 32
    pub patch_size: [usize; 3],    // [1, 1, 1]
    pub rope_dim_list: [usize; 3], // [16, 56, 56]
    pub rope_theta: f64,           // 256.0
    pub mlp_ratio: f32,            // 4.0
    pub text_dim: usize,           // 3584
    pub eps: f32,                  // 1e-6
}

impl Default for Hunyuan15Config {
    fn default() -> Self {
        Self {
            num_double_blocks: 54,
            hidden_size: 2048,
            num_heads: 16,
            head_dim: 128,
            in_channels: 32,
            out_channels: 32,
            patch_size: [1, 1, 1],
            rope_dim_list: [16, 56, 56],
            rope_theta: 256.0,
            mlp_ratio: 4.0,
            text_dim: 3584,
            eps: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Hunyuan15Dit
// ---------------------------------------------------------------------------

pub struct Hunyuan15Dit {
    shared: HashMap<String, Tensor>,
    offloader: BlockOffloader,
    config: Hunyuan15Config,
    device: Arc<CudaDevice>,
}

impl Hunyuan15Dit {
    pub fn load(
        checkpoint_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = Hunyuan15Config::default();

        let facilitator = Hunyuan15Facilitator { num_blocks: config.num_double_blocks };
        let offloader = BlockOffloader::load(
            &[checkpoint_path],
            &facilitator,
            device.clone(),
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader HunyuanVideo: {e}")))?;

        // Shared weights
        let shared_prefixes = [
            "img_in.", "time_in.", "txt_in.", "byt5_in.", "vision_in.",
            "final_layer.", "cond_type_embedding.",
        ];
        let part = load_file_filtered(Path::new(checkpoint_path), device, |key| {
            shared_prefixes.iter().any(|p| key.starts_with(p))
        })?;
        let shared: HashMap<String, Tensor> = part.into_iter().map(|(k, v)| {
            let v_bf16 = if v.dtype() != DType::BF16 {
                v.to_dtype(DType::BF16).unwrap_or(v)
            } else { v };
            (k, v_bf16)
        }).collect();

        log::info!(
            "[HunyuanVideo] Loaded: {} blocks via BlockOffloader, {} shared weights",
            offloader.block_count(), shared.len()
        );

        Ok(Self { shared, offloader, config, device: device.clone() })
    }

    pub fn config(&self) -> &Hunyuan15Config { &self.config }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
    }

    fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
        out.reshape(&dims)
    }

    fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
        out.reshape(&dims)
    }

    fn layer_norm_affine(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, Some(w), Some(b), eps)?;
        out.reshape(&dims)
    }

    /// Sinusoidal timestep embedding (same as FLUX/HunyuanVideo TimestepEmbedder).
    /// Input: [B] timesteps. Output: [B, 256] F32.
    fn timestep_embedding(&self, timestep: f32) -> Result<Tensor> {
        let dim = 256usize; // freq_dim
        let half = dim / 2;
        let mut data = vec![0.0f32; dim];
        let pos = timestep as f64;
        for i in 0..half {
            let freq = (-(i as f64) / half as f64 * 10000.0f64.ln()).exp();
            let angle = pos * freq;
            data[i] = angle.cos() as f32;
            data[half + i] = angle.sin() as f32;
        }
        Tensor::from_vec(data, Shape::from_dims(&[1, dim]), self.device.clone())
    }

    /// Build 3-axis RoPE (cos, sin) tables.
    /// Returns (freqs_cos, freqs_sin) each [seq_len, head_dim/2].
    /// Uses `use_real=True` mode (separate cos/sin, not complex).
    fn build_rope(
        &self,
        tt: usize, th: usize, tw: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let dims = &cfg.rope_dim_list; // [16, 56, 56]
        let theta = cfg.rope_theta;
        let seq_len = tt * th * tw;
        let total_half: usize = dims.iter().sum::<usize>() / 2; // 64

        let mut cos_data = vec![0.0f32; seq_len * total_half];
        let mut sin_data = vec![0.0f32; seq_len * total_half];

        // Per-axis frequency tables
        let freq_tables: Vec<Vec<f32>> = dims.iter().map(|&d| {
            let half = d / 2;
            (0..half).map(|i| {
                (1.0 / theta.powf(2.0 * i as f64 / d as f64)) as f32
            }).collect()
        }).collect();

        for ti in 0..tt {
            for hi in 0..th {
                for wi in 0..tw {
                    let token_idx = ti * th * tw + hi * tw + wi;
                    let positions = [ti as f32, hi as f32, wi as f32];
                    let mut offset = 0usize;

                    for (axis_i, freqs) in freq_tables.iter().enumerate() {
                        let half = dims[axis_i] / 2;
                        let pos = positions[axis_i];
                        for i in 0..half {
                            let angle = pos * freqs[i];
                            cos_data[token_idx * total_half + offset + i] = angle.cos();
                            sin_data[token_idx * total_half + offset + i] = angle.sin();
                        }
                        offset += half;
                    }
                }
            }
        }

        let cos_t = Tensor::from_vec(cos_data, Shape::from_dims(&[seq_len, total_half]), self.device.clone())?;
        let sin_t = Tensor::from_vec(sin_data, Shape::from_dims(&[seq_len, total_half]), self.device.clone())?;
        Ok((cos_t, sin_t))
    }

    /// Apply rotary embedding (HunyuanVideo half-split / `rotate_half` convention).
    ///
    /// x: `[B, S, H, D]` BF16 — caller passes the full image stream pre-permute.
    ///   S must equal `img_len` (text tokens are handled outside, with no RoPE).
    /// freqs_cos / freqs_sin: `[S_img, D/2]` BF16.
    ///
    /// Internally permutes to `[B, H, N, D]`, calls the GPU
    /// `rope_halfsplit_bf16` kernel, and permutes back to `[B, S, H, D]`.
    /// Eliminates the host-CPU F32 round-trip that previously dominated cost
    /// per double-stream block.
    fn apply_rope(
        x: &Tensor,
        freqs_cos: &Tensor,
        freqs_sin: &Tensor,
        img_len: usize,
        _device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
        debug_assert_eq!(s, img_len, "apply_rope: S must equal img_len (text gets no RoPE)");
        let half = d / 2;

        // Reshape cos/sin to [1, 1, N, half] as the kernel expects (cos_bh=1).
        let cos = freqs_cos.reshape(&[1, 1, s, half])?;
        let sin = freqs_sin.reshape(&[1, 1, s, half])?;

        // [B, S, H, D] → [B, H, S, D] (kernel layout).
        // Use .contiguous() to force a fresh dense tensor — the rope kernel
        // expects flat row-major and the BF16 reshape inside it would error
        // on a permuted view.
        let x_bhnd = x.permute(&[0, 2, 1, 3])?.contiguous()?;

        // Fused half-split RoPE on GPU.
        let out_bhnd = flame_core::bf16_ops::rope_halfsplit_bf16(&x_bhnd, &cos, &sin)?;

        // [B, H, S, D] → [B, S, H, D]
        out_bhnd.permute(&[0, 2, 1, 3])?.contiguous()
    }

    /// gelu_tanh: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    /// Matches PyTorch `F.gelu(approximate='tanh')` and HunyuanVideo's MLP
    /// activation. Uses existing flame-core BF16 elementwise ops; no new kernel.
    fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
        // c = sqrt(2 / pi) ≈ 0.7978845608028654
        const C: f32 = 0.7978845608028654;
        const K: f32 = 0.044715;
        let x_sq = x.mul(x)?;
        let x_cu = x_sq.mul(x)?;
        // inner = c * (x + k * x^3)
        let inner = x.add(&x_cu.mul_scalar(K)?)?.mul_scalar(C)?;
        let t = inner.tanh()?;
        // 0.5 * x * (1 + tanh(...))
        let gate = t.add_scalar(1.0)?.mul_scalar(0.5)?;
        x.mul(&gate)
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Forward pass.
    ///
    /// Arguments:
    /// - `hidden_states`: [B, C_in, T, H, W] video latent (already concatenated with condition+mask = 65ch)
    /// - `timestep`: scalar timestep
    /// - `txt_embeds`: [B, L_txt, hidden_size] — already refined text embeddings (from Python Stage 1)
    /// - `txt_mask`: [B, L_txt] — attention mask for text
    /// - `guidance`: guidance scale value (default 6016.0)
    ///
    /// Returns: [B, C_out, T, H, W] prediction
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,  // [1, 65, T, H, W]
        timestep: f32,
        txt_embeds: &Tensor,     // [1, L_txt, 2048] — pre-refined
        txt_mask: &Tensor,       // [1, L_txt]
        guidance: f32,
    ) -> Result<Tensor> {
        let cfg = self.config.clone();
        let dim = cfg.hidden_size;
        let nh = cfg.num_heads;
        let hd = cfg.head_dim;

        let hs_dims = hidden_states.shape().dims().to_vec();
        let (bs, _c_in, ot, oh, ow) = (hs_dims[0], hs_dims[1], hs_dims[2], hs_dims[3], hs_dims[4]);
        let tt = ot / cfg.patch_size[0];
        let th = oh / cfg.patch_size[1];
        let tw = ow / cfg.patch_size[2];
        let img_seq_len = tt * th * tw;

        // ── img_in: Conv3d(65, 2048, k=1, s=1) as linear ──
        let img_in_w = self.shared.get("img_in.proj.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.proj.weight".into()))?;
        let img_in_b = self.shared.get("img_in.proj.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.proj.bias".into()))?;

        // Reshape [1, 65, T, H, W] → [1, T*H*W, 65], apply linear → [1, T*H*W, 2048]
        let c_in = hs_dims[1];
        let hs_flat = hidden_states.reshape(&[bs, c_in, img_seq_len])?
            .permute(&[0, 2, 1])?; // [1, seq, 65]
        let img_in_w_flat = img_in_w.reshape(&[dim, c_in])?;
        let mut img = Self::linear_bias(&hs_flat, &img_in_w_flat, img_in_b)?;
        // img: [1, img_seq_len, 2048]

        // ── time_in: TimestepEmbedder → vec ──
        let t_emb = self.timestep_embedding(timestep)?; // [1, 256]
        let t_w0 = self.shared.get("time_in.mlp.0.weight").unwrap();
        let t_b0 = self.shared.get("time_in.mlp.0.bias").unwrap();
        let t_w2 = self.shared.get("time_in.mlp.2.weight").unwrap();
        let t_b2 = self.shared.get("time_in.mlp.2.bias").unwrap();
        let t_emb_bf16 = t_emb.to_dtype(DType::BF16)?;
        let vec = Self::linear_bias(&t_emb_bf16.unsqueeze(0)?, t_w0, t_b0)?;
        let vec = vec.silu()?;
        let vec = Self::linear_bias(&vec, t_w2, t_b2)?.squeeze(Some(0))?; // [1, 2048]

        // ── txt already refined (done in Python) ──
        let txt = txt_embeds.clone(); // [1, L_txt, 2048]
        let txt_len = txt.shape().dims()[1];

        // ── RoPE ──
        let (freqs_cos, freqs_sin) = self.build_rope(tt, th, tw)?;

        // ── Double-stream blocks ──
        let total_blocks = cfg.num_double_blocks;
        self.offloader.prefetch_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        let mut txt_stream = txt;

        for i in 0..total_blocks {
            let raw = self.offloader.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.offloader.prefetch_block(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            let (new_img, new_txt) = self.double_block_forward(
                &img, &txt_stream, &vec, &freqs_cos, &freqs_sin, &raw, i,
            )?;
            img = new_img;
            txt_stream = new_txt;

            if i % 10 == 0 || i == total_blocks - 1 {
                log::info!("[HunyuanVideo] Block {}/{}", i + 1, total_blocks);
            }
        }

        // ── Final layer: AdaLN + Linear ──
        let fl_mod_w = self.shared.get("final_layer.adaLN_modulation.1.weight").unwrap();
        let fl_mod_b = self.shared.get("final_layer.adaLN_modulation.1.bias").unwrap();
        let fl_lin_w = self.shared.get("final_layer.linear.weight").unwrap();
        let fl_lin_b = self.shared.get("final_layer.linear.bias").unwrap();

        // vec → SiLU → Linear → chunk(2) → (shift, scale)
        let vec_3d = vec.unsqueeze(0)?; // [1, 1, 2048]
        let fl_mods = Self::linear_bias(&vec_3d.silu()?, fl_mod_w, fl_mod_b)?.squeeze(Some(0))?;
        // fl_mods: [1, 4096] → chunk → shift [1, 2048], scale [1, 2048]
        let shift = fl_mods.narrow(1, 0, dim)?;
        let scale = fl_mods.narrow(1, dim, dim)?;

        let img_normed = Self::layer_norm_no_affine(&img, cfg.eps)?;
        let one_plus = scale.add_scalar(1.0)?.unsqueeze(1)?;
        let img_mod = img_normed.mul(&one_plus)?.add(&shift.unsqueeze(1)?)?;
        let out = Self::linear_bias(&img_mod, fl_lin_w, fl_lin_b)?;
        // out: [1, img_seq_len, 32]

        // ── Unpatchify: [1, T*H*W, 32] → [1, 32, T, H, W] ──
        let out_perm = out.permute(&[0, 2, 1])?; // [1, 32, T*H*W]
        let result = out_perm.reshape(&[bs, cfg.out_channels, tt, th, tw])?;
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Double-stream block
    // -----------------------------------------------------------------------

    fn double_block_forward(
        &self,
        img: &Tensor,       // [1, img_seq, dim]
        txt: &Tensor,       // [1, txt_seq, dim]
        vec: &Tensor,       // [1, dim]
        freqs_cos: &Tensor, // [img_seq, dim/2]
        freqs_sin: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let dim = cfg.hidden_size;
        let nh = cfg.num_heads;
        let hd = cfg.head_dim;
        let prefix = format!("double_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let img_len = img.shape().dims()[1];
        let txt_len = txt.shape().dims()[1];

        // ── img_mod: SiLU → Linear → chunk(6) ──
        let vec_3d = vec.unsqueeze(0)?; // [1, 1, dim]
        let img_mods = Self::linear_bias(&vec_3d.silu()?, w("img_mod.linear.weight")?, w("img_mod.linear.bias")?)?
            .squeeze(Some(0))?; // [1, 6*dim]
        let txt_mods = Self::linear_bias(&vec_3d.silu()?, w("txt_mod.linear.weight")?, w("txt_mod.linear.bias")?)?
            .squeeze(Some(0))?;

        // chunk(6): shift1, scale1, gate1, shift2, scale2, gate2
        let img_s1 = img_mods.narrow(1, 0, dim)?;
        let img_sc1 = img_mods.narrow(1, dim, dim)?;
        let img_g1 = img_mods.narrow(1, 2*dim, dim)?;
        let img_s2 = img_mods.narrow(1, 3*dim, dim)?;
        let img_sc2 = img_mods.narrow(1, 4*dim, dim)?;
        let img_g2 = img_mods.narrow(1, 5*dim, dim)?;

        let txt_s1 = txt_mods.narrow(1, 0, dim)?;
        let txt_sc1 = txt_mods.narrow(1, dim, dim)?;
        let txt_g1 = txt_mods.narrow(1, 2*dim, dim)?;
        let txt_s2 = txt_mods.narrow(1, 3*dim, dim)?;
        let txt_sc2 = txt_mods.narrow(1, 4*dim, dim)?;
        let txt_g2 = txt_mods.narrow(1, 5*dim, dim)?;

        // ── img norm + modulate ──
        let img_normed = Self::layer_norm_no_affine(img, cfg.eps)?;
        let img_modulated = img_normed.mul(&img_sc1.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&img_s1.unsqueeze(1)?)?;

        // ── img Q/K/V ──
        let img_q = Self::linear_bias(&img_modulated, w("img_attn_q.weight")?, w("img_attn_q.bias")?)?;
        let img_k = Self::linear_bias(&img_modulated, w("img_attn_k.weight")?, w("img_attn_k.bias")?)?;
        let img_v = Self::linear_bias(&img_modulated, w("img_attn_v.weight")?, w("img_attn_v.bias")?)?;

        let img_q = img_q.reshape(&[1, img_len, nh, hd])?;
        let img_k = img_k.reshape(&[1, img_len, nh, hd])?;
        let img_v = img_v.reshape(&[1, img_len, nh, hd])?;

        // QK RMSNorm (per-head)
        let img_q = Self::rms_norm(&img_q, w("img_attn_q_norm.weight")?, cfg.eps)?;
        let img_k = Self::rms_norm(&img_k, w("img_attn_k_norm.weight")?, cfg.eps)?;

        // Apply RoPE to image Q/K
        let img_q = Self::apply_rope(&img_q, freqs_cos, freqs_sin, img_len, &self.device)?;
        let img_k = Self::apply_rope(&img_k, freqs_cos, freqs_sin, img_len, &self.device)?;

        // ── txt norm + modulate ──
        let txt_normed = Self::layer_norm_no_affine(txt, cfg.eps)?;
        let txt_modulated = txt_normed.mul(&txt_sc1.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&txt_s1.unsqueeze(1)?)?;

        let txt_q = Self::linear_bias(&txt_modulated, w("txt_attn_q.weight")?, w("txt_attn_q.bias")?)?;
        let txt_k = Self::linear_bias(&txt_modulated, w("txt_attn_k.weight")?, w("txt_attn_k.bias")?)?;
        let txt_v = Self::linear_bias(&txt_modulated, w("txt_attn_v.weight")?, w("txt_attn_v.bias")?)?;

        let txt_q = txt_q.reshape(&[1, txt_len, nh, hd])?;
        let txt_k = txt_k.reshape(&[1, txt_len, nh, hd])?;
        let txt_v = txt_v.reshape(&[1, txt_len, nh, hd])?;

        let txt_q = Self::rms_norm(&txt_q, w("txt_attn_q_norm.weight")?, cfg.eps)?;
        let txt_k = Self::rms_norm(&txt_k, w("txt_attn_k_norm.weight")?, cfg.eps)?;
        // No RoPE on text tokens

        // ── Joint attention: concat [img, txt] → SDPA → split ──
        let q = Tensor::cat(&[
            &img_q.permute(&[0, 2, 1, 3])?,
            &txt_q.permute(&[0, 2, 1, 3])?,
        ], 2)?; // [1, H, img+txt, D]
        let k = Tensor::cat(&[
            &img_k.permute(&[0, 2, 1, 3])?,
            &txt_k.permute(&[0, 2, 1, 3])?,
        ], 2)?;
        let v = Tensor::cat(&[
            &img_v.permute(&[0, 2, 1, 3])?,
            &txt_v.permute(&[0, 2, 1, 3])?,
        ], 2)?;

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        let img_attn = attn_out.narrow(2, 0, img_len)?
            .permute(&[0, 2, 1, 3])?.reshape(&[1, img_len, nh * hd])?;
        let txt_attn = attn_out.narrow(2, img_len, txt_len)?
            .permute(&[0, 2, 1, 3])?.reshape(&[1, txt_len, nh * hd])?;

        // ── img gated residual + MLP ──
        let img_attn_proj = Self::linear_bias(&img_attn, w("img_attn_proj.weight")?, w("img_attn_proj.bias")?)?;
        let img = img.add(&img_attn_proj.mul(&img_g1.unsqueeze(1)?)?)?;

        let img_normed2 = Self::layer_norm_no_affine(&img, cfg.eps)?;
        let img_mlp_in = img_normed2.mul(&img_sc2.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&img_s2.unsqueeze(1)?)?;
        let img_mlp = Self::linear_bias(&img_mlp_in, w("img_mlp.fc1.weight")?, w("img_mlp.fc1.bias")?)?;
        let img_mlp = Self::gelu_tanh(&img_mlp)?;
        let img_mlp = Self::linear_bias(&img_mlp, w("img_mlp.fc2.weight")?, w("img_mlp.fc2.bias")?)?;
        let img = img.add(&img_mlp.mul(&img_g2.unsqueeze(1)?)?)?;

        // ── txt gated residual + MLP ──
        let txt_attn_proj = Self::linear_bias(&txt_attn, w("txt_attn_proj.weight")?, w("txt_attn_proj.bias")?)?;
        let txt = txt.add(&txt_attn_proj.mul(&txt_g1.unsqueeze(1)?)?)?;

        let txt_normed2 = Self::layer_norm_no_affine(&txt, cfg.eps)?;
        let txt_mlp_in = txt_normed2.mul(&txt_sc2.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&txt_s2.unsqueeze(1)?)?;
        let txt_mlp = Self::linear_bias(&txt_mlp_in, w("txt_mlp.fc1.weight")?, w("txt_mlp.fc1.bias")?)?;
        let txt_mlp = Self::gelu_tanh(&txt_mlp)?;
        let txt_mlp = Self::linear_bias(&txt_mlp, w("txt_mlp.fc2.weight")?, w("txt_mlp.fc2.bias")?)?;
        let txt = txt.add(&txt_mlp.mul(&txt_g2.unsqueeze(1)?)?)?;

        Ok((img, txt))
    }
}
