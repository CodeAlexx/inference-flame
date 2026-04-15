//! ERNIE Image — Baidu's single-stream DiT for image generation.
//!
//! Ported from `diffusers.models.transformers.transformer_ernie_image`
//! (Apache-2.0). Weights not yet public — architecture verified against
//! the diffusers source on `main` (2026-04-12).
//!
//! ## Architecture
//!
//! - Single-stream DiT: img + text tokens concatenated, joint attention
//! - Shared AdaLN: one modulation computed once, broadcast to all blocks
//! - SwiGLU FFN: `down(up(x) * gelu(gate(x)))`, no bias
//! - QK RMSNorm with learned scale (elementwise_affine=True)
//! - Non-interleaved RoPE (half-split: `[-x2, x1]`), theta=256
//! - Patch embed: Conv2d (kernel=patch_size, stride=patch_size, bias=True)
//!
//! ## Default config
//!
//! ```text
//! hidden_size=4096, heads=32, head_dim=128, layers=36
//! in_channels=128, out_channels=128, patch_size=1
//! text_in_dim=3072, ffn_hidden=12288
//! rope_theta=256, rope_axes_dim=(32, 48, 48)
//! ```

use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ErnieImageConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub ffn_hidden_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub text_in_dim: usize,
    pub rope_theta: f64,
    pub rope_axes_dim: [usize; 3],
    pub eps: f32,
    pub qk_norm: bool,
}

impl Default for ErnieImageConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 32,
            head_dim: 128,
            num_layers: 36,
            ffn_hidden_size: 12288,
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            text_in_dim: 3072,
            rope_theta: 256.0,
            rope_axes_dim: [32, 48, 48],
            eps: 1e-6,
            qk_norm: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE — 3-axis, non-interleaved (half-split)
// ---------------------------------------------------------------------------

/// Build per-axis frequencies: `pos / theta^(2k/dim)`.
/// Returns `[..., dim/2]` angles (NOT cos/sin yet — applied in attention).
fn rope_freqs(pos: &[f32], dim: usize, theta: f64, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
    let half = dim / 2;
    let mut data = vec![0.0f32; pos.len() * half];
    for (p_idx, &p) in pos.iter().enumerate() {
        for k in 0..half {
            let scale = 2.0 * k as f64 / dim as f64;
            let omega = 1.0 / theta.powf(scale);
            data[p_idx * half + k] = (p as f64 * omega) as f32;
        }
    }
    Tensor::from_vec(data, Shape::from_dims(&[pos.len(), half]), device.clone())
}

/// Compute 3-axis RoPE for image + text tokens.
///
/// Image IDs: `[B, N_img, 3]` where cols are `(text_offset, row, col)`.
/// Text IDs: `[B, N_txt, 3]` where cols are `(seq_idx, 0, 0)`.
/// Combined: `[B, S, 3]` → per-axis frequencies → cat → `[B, S, 1, head_dim]`.
///
/// Output format matches diffusers: angles with doubled layout
/// `[θ0, θ0, θ1, θ1, ...]` for non-interleaved rotate_half.
pub fn compute_rope_embeddings(
    height_patches: usize,
    width_patches: usize,
    text_len: usize,
    config: &ErnieImageConfig,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let n_img = height_patches * width_patches;
    let total = n_img + text_len;
    let [ax0, ax1, ax2] = config.rope_axes_dim;

    // Image position IDs: (text_len, row, col) — text_len offset on axis 0
    let mut ids_ax0 = Vec::with_capacity(total);
    let mut ids_ax1 = Vec::with_capacity(total);
    let mut ids_ax2 = Vec::with_capacity(total);

    // Image tokens
    for r in 0..height_patches {
        for c in 0..width_patches {
            ids_ax0.push(text_len as f32); // text_offset
            ids_ax1.push(r as f32);
            ids_ax2.push(c as f32);
        }
    }
    // Text tokens
    for t in 0..text_len {
        ids_ax0.push(t as f32);
        ids_ax1.push(0.0);
        ids_ax2.push(0.0);
    }

    let freq0 = rope_freqs(&ids_ax0, ax0, config.rope_theta, device)?; // [S, ax0/2]
    let freq1 = rope_freqs(&ids_ax1, ax1, config.rope_theta, device)?; // [S, ax1/2]
    let freq2 = rope_freqs(&ids_ax2, ax2, config.rope_theta, device)?; // [S, ax2/2]

    // Cat along last dim: [S, head_dim/2]
    let freqs = Tensor::cat(&[&freq0, &freq1, &freq2], 1)?; // [S, (ax0+ax1+ax2)/2]

    // Double to [S, head_dim]: [θ0, θ0, θ1, θ1, ...] for non-interleaved rotate_half.
    // Python: torch.stack([emb, emb], dim=-1).reshape(...) interleaves pairs.
    let freqs_unsq = freqs.unsqueeze(1)?; // [S, 1, head_dim/2]
    // Stack on new last dim then reshape to interleave: [S, 1, head_dim/2, 2] → [S, 1, head_dim]
    let stacked = Tensor::stack(&[freqs_unsq.clone(), freqs_unsq], 3)?; // [S, 1, head_dim/2, 2]
    let dims = stacked.shape().dims().to_vec();
    let doubled = stacked.reshape(&[dims[0], dims[1], dims[2] * dims[3]])?; // [S, 1, head_dim]

    // Add batch dim: [1, S, 1, head_dim]
    doubled.unsqueeze(0)?.to_dtype(DType::F32)
}

/// Apply non-interleaved RoPE (half-split rotate: `[-x2, x1]`).
/// `x`: `[B, S, H, D]`, `freqs`: `[1, S, 1, D]` (angles).
pub fn apply_rotary_emb(x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let rot_dim = freqs.shape().dims()[3];
    let x_dim = x.shape().dims()[3];

    let (x_rot, x_pass) = if rot_dim < x_dim {
        (x.narrow(3, 0, rot_dim)?, Some(x.narrow(3, rot_dim, x_dim - rot_dim)?))
    } else {
        (x.clone(), None)
    };

    let cos = freqs.cos()?.to_dtype(x.dtype())?;
    let sin = freqs.sin()?.to_dtype(x.dtype())?;

    // Non-interleaved rotate_half: [-x2, x1]
    let half = rot_dim / 2;
    let x1 = x_rot.narrow(3, 0, half)?;
    let x2 = x_rot.narrow(3, half, half)?;
    let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

    let out_rot = x_rot.mul(&cos)?.add(&x_rotated.mul(&sin)?)?;

    if let Some(pass) = x_pass {
        Tensor::cat(&[&out_rot, &pass], 3)
    } else {
        Ok(out_rot)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct ErnieImageModel {
    pub config: ErnieImageConfig,

    // Patch embedding (Conv2d as linear for patch_size=1)
    pub patch_proj_weight: Tensor, // [hidden, in_ch, P, P]
    pub patch_proj_bias: Tensor,   // [hidden]

    // Text projection (optional — only when text_in_dim != hidden_size)
    pub text_proj_weight: Option<Tensor>, // [hidden, text_in_dim]

    // Time embedding
    pub time_embedding_linear1_weight: Tensor, // [hidden, hidden]
    pub time_embedding_linear1_bias: Tensor,
    pub time_embedding_linear2_weight: Tensor,
    pub time_embedding_linear2_bias: Tensor,

    // Shared AdaLN modulation: SiLU → Linear(hidden, 6*hidden)
    pub adaln_mod_weight: Tensor, // [6*hidden, hidden]
    pub adaln_mod_bias: Tensor,

    // Transformer blocks
    pub blocks: Vec<ErnieImageBlock>,

    // Final norm + linear
    pub final_norm_linear_weight: Tensor, // [2*hidden, hidden]
    pub final_norm_linear_bias: Tensor,
    pub final_linear_weight: Tensor, // [P*P*out_ch, hidden]
    pub final_linear_bias: Tensor,

    device: Arc<cudarc::driver::CudaDevice>,
}

pub struct ErnieImageBlock {
    // Self-attention
    pub sa_norm_weight: Tensor,  // RMSNorm [hidden]
    pub attn_q_weight: Tensor,   // [hidden, hidden]
    pub attn_k_weight: Tensor,
    pub attn_v_weight: Tensor,
    pub attn_out_weight: Tensor,
    pub norm_q_weight: Tensor,   // RMSNorm [head_dim]
    pub norm_k_weight: Tensor,

    // FFN (SwiGLU)
    pub mlp_norm_weight: Tensor, // RMSNorm [hidden]
    pub gate_proj_weight: Tensor, // [ffn_hidden, hidden]
    pub up_proj_weight: Tensor,   // [ffn_hidden, hidden]
    pub down_proj_weight: Tensor, // [hidden, ffn_hidden]
}

impl ErnieImageModel {
    pub fn load(weights: &HashMap<String, Tensor>, config: ErnieImageConfig) -> Result<Self> {
        let device = flame_core::global_cuda_device();

        let get = |key: &str| -> Result<Tensor> {
            weights.get(key).cloned()
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {key}")))
        };

        let text_proj_weight = if config.text_in_dim != config.hidden_size {
            Some(get("text_proj.weight")?)
        } else { None };

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("layers.{i}");
            blocks.push(ErnieImageBlock {
                sa_norm_weight: get(&format!("{p}.adaLN_sa_ln.weight"))?,
                attn_q_weight: get(&format!("{p}.self_attention.to_q.weight"))?,
                attn_k_weight: get(&format!("{p}.self_attention.to_k.weight"))?,
                attn_v_weight: get(&format!("{p}.self_attention.to_v.weight"))?,
                attn_out_weight: get(&format!("{p}.self_attention.to_out.0.weight"))?,
                norm_q_weight: get(&format!("{p}.self_attention.norm_q.weight"))?,
                norm_k_weight: get(&format!("{p}.self_attention.norm_k.weight"))?,
                mlp_norm_weight: get(&format!("{p}.adaLN_mlp_ln.weight"))?,
                gate_proj_weight: get(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj_weight: get(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj_weight: get(&format!("{p}.mlp.linear_fc2.weight"))?,
            });
        }

        Ok(Self {
            patch_proj_weight: get("x_embedder.proj.weight")?,
            patch_proj_bias: get("x_embedder.proj.bias")?,
            text_proj_weight,
            time_embedding_linear1_weight: get("time_embedding.linear_1.weight")?,
            time_embedding_linear1_bias: get("time_embedding.linear_1.bias")?,
            time_embedding_linear2_weight: get("time_embedding.linear_2.weight")?,
            time_embedding_linear2_bias: get("time_embedding.linear_2.bias")?,
            adaln_mod_weight: get("adaLN_modulation.1.weight")?,
            adaln_mod_bias: get("adaLN_modulation.1.bias")?,
            blocks,
            final_norm_linear_weight: get("final_norm.linear.weight")?,
            final_norm_linear_bias: get("final_norm.linear.bias")?,
            final_linear_weight: get("final_linear.weight")?,
            final_linear_bias: get("final_linear.bias")?,
            config,
            device,
        })
    }

    /// Forward pass.
    ///
    /// `image`: `[B, C, H, W]` BF16 latent
    /// `timestep`: `[B]` F32
    /// `text_embeds`: `[B, T, text_dim]` BF16
    /// `text_lens`: `[B]` i64 — valid text lengths per sample
    ///
    /// Returns `[B, out_channels, H, W]` BF16.
    pub fn forward(
        &self,
        image: &Tensor,
        timestep: &Tensor,
        text_embeds: &Tensor,
        text_lens: &[usize],
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = image.shape().dims();
        let (b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let p = cfg.patch_size;
        let (hp, wp) = (h / p, w / p);
        let n_img = hp * wp;
        let t_max = text_embeds.shape().dims()[1];

        // 1. Patch embed: Conv2d → [B, hidden, Hp, Wp] → [B, Hp*Wp, hidden]
        let img_tokens = self.patch_embed(image)?; // [B, N_img, hidden]

        // 2. Text projection
        let txt_tokens = if let Some(ref w) = self.text_proj_weight {
            let wt = w.transpose()?;
            text_embeds.matmul(&wt)?
        } else {
            text_embeds.clone()
        };

        // 3. Concatenate: [B, N_img + T_max, hidden]
        let x = Tensor::cat(&[&img_tokens, &txt_tokens], 1)?;
        let s_total = n_img + t_max;

        // 4. RoPE
        let rope_emb = compute_rope_embeddings(hp, wp, t_max, cfg, &self.device)?;

        // 5. Attention mask: True for valid tokens
        let mut mask_data = vec![0u8; b * s_total];
        for bi in 0..b {
            // All image tokens valid
            for j in 0..n_img {
                mask_data[bi * s_total + j] = 1;
            }
            // Text tokens up to text_lens[bi]
            let tl = if bi < text_lens.len() { text_lens[bi] } else { t_max };
            for j in 0..tl.min(t_max) {
                mask_data[bi * s_total + n_img + j] = 1;
            }
        }
        // For now: no mask (all valid) — SDPA with None mask
        // Full mask support needs bool tensor → SDPA mask path

        // 6. Time embedding
        let t_emb = self.time_embed(timestep)?; // [B, hidden]
        let c = t_emb.clone();

        // 7. Shared AdaLN modulation: SiLU → Linear → 6 chunks
        let mod_out = c.silu()?;
        let mod_wt = self.adaln_mod_weight.transpose()?;
        let mod_out = mod_out.matmul(&mod_wt)?.add(&self.adaln_mod_bias)?;
        let chunks = mod_out.chunk(6, 1)?; // each [B, hidden]
        // Expand to [B, S, hidden] for broadcast
        let expand = |t: &Tensor| -> Result<Tensor> {
            t.unsqueeze(1)?.expand(&[b, s_total, cfg.hidden_size])
        };
        let shift_msa = expand(&chunks[0])?;
        let scale_msa = expand(&chunks[1])?;
        let gate_msa = expand(&chunks[2])?;
        let shift_mlp = expand(&chunks[3])?;
        let scale_mlp = expand(&chunks[4])?;
        let gate_mlp = expand(&chunks[5])?;

        // 8. Transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = self.block_forward(
                &x, &rope_emb,
                &shift_msa, &scale_msa, &gate_msa,
                &shift_mlp, &scale_mlp, &gate_mlp,
                block,
            )?;
        }

        // 9. Final norm: LayerNorm(x) * (1 + scale) + shift
        let final_mod_wt = self.final_norm_linear_weight.transpose()?;
        let final_mod = c.matmul(&final_mod_wt)?.add(&self.final_norm_linear_bias)?;
        let final_chunks = final_mod.chunk(2, 1)?;
        let f_scale = final_chunks[0].unsqueeze(1)?;
        let f_shift = final_chunks[1].unsqueeze(1)?;

        let x_norm = flame_core::layer_norm::layer_norm(&x, &[cfg.hidden_size], None, None, cfg.eps)?;
        let x_out = x_norm.mul(&f_scale.add_scalar(1.0)?)?.add(&f_shift)?;

        // 10. Final linear: [B, S, hidden] → [B, S, P*P*out_ch]
        let fl_wt = self.final_linear_weight.transpose()?;
        let patches = x_out.matmul(&fl_wt)?.add(&self.final_linear_bias)?;

        // 11. Extract image tokens and unpatchify
        let img_patches = patches.narrow(1, 0, n_img)?; // [B, N_img, P*P*out_ch]
        self.unpatchify(&img_patches, hp, wp)
    }

    fn patch_embed(&self, image: &Tensor) -> Result<Tensor> {
        let dims = image.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let p = self.config.patch_size;
        let (hp, wp) = (h / p, w / p);

        if p == 1 {
            // Conv2d with kernel=1 is just a linear on channel dim
            // [B, C, H, W] → [B, H*W, C] → matmul → [B, H*W, hidden]
            let x = image.reshape(&[b, c, h * w])?.permute(&[0, 2, 1])?;
            let w_flat = self.patch_proj_weight.reshape(&[self.config.hidden_size, c])?;
            let wt = w_flat.transpose()?;
            x.matmul(&wt)?.add(&self.patch_proj_bias)
        } else {
            // General Conv2d patch embedding
            // [B, C, H, W] → reshape to patches → linear
            let x = image.reshape(&[b, c, hp, p, wp, p])?;
            let x = x.permute(&[0, 2, 4, 1, 3, 5])?; // [B, Hp, Wp, C, P, P]
            let patch_dim = c * p * p;
            let x = x.reshape(&[b, hp * wp, patch_dim])?;
            let w_flat = self.patch_proj_weight.reshape(&[self.config.hidden_size, patch_dim])?;
            let wt = w_flat.transpose()?;
            x.matmul(&wt)?.add(&self.patch_proj_bias)
        }
    }

    fn time_embed(&self, timestep: &Tensor) -> Result<Tensor> {
        let dim = self.config.hidden_size;
        let half = dim / 2;

        // Sinusoidal embedding (flip_sin_to_cos=False, downscale_freq_shift=0)
        let t = timestep.to_dtype(DType::F32)?;
        let mut freqs = Vec::with_capacity(half);
        for i in 0..half {
            freqs.push((-((i as f64) / half as f64) * (10000.0f64).ln()).exp() as f32);
        }
        let freq_t = Tensor::from_vec(freqs, Shape::from_dims(&[1, half]), self.device.clone())?;
        let t_2d = t.unsqueeze(1)?;
        let args = t_2d.mul(&freq_t)?;
        let sin_part = args.sin()?;
        let cos_part = args.cos()?;
        let emb = Tensor::cat(&[&sin_part, &cos_part], 1)?.to_dtype(DType::BF16)?;

        // MLP: Linear → SiLU → Linear
        let w1t = self.time_embedding_linear1_weight.transpose()?;
        let h = emb.matmul(&w1t)?.add(&self.time_embedding_linear1_bias)?;
        let h = h.silu()?;
        let w2t = self.time_embedding_linear2_weight.transpose()?;
        h.matmul(&w2t)?.add(&self.time_embedding_linear2_bias)
    }

    fn block_forward(
        &self,
        x: &Tensor,
        rope_emb: &Tensor,
        shift_msa: &Tensor, scale_msa: &Tensor, gate_msa: &Tensor,
        shift_mlp: &Tensor, scale_mlp: &Tensor, gate_mlp: &Tensor,
        block: &ErnieImageBlock,
    ) -> Result<Tensor> {
        let cfg = &self.config;

        // Self-attention with AdaLN
        let residual = x.clone();
        let normed = flame_core::norm::rms_norm(x, &[cfg.hidden_size], Some(&block.sa_norm_weight), cfg.eps)?;
        let modulated = normed.mul(&scale_msa.add_scalar(1.0)?)?.add(shift_msa)?;

        let attn_out = self.attention(&modulated, rope_emb, block)?;
        let x = residual.add(&gate_msa.mul(&attn_out)?)?;

        // FFN with AdaLN
        let residual = x.clone();
        let normed = flame_core::norm::rms_norm(&x, &[cfg.hidden_size], Some(&block.mlp_norm_weight), cfg.eps)?;
        let modulated = normed.mul(&scale_mlp.add_scalar(1.0)?)?.add(shift_mlp)?;

        let ffn_out = self.ffn(&modulated, block)?;
        residual.add(&gate_mlp.mul(&ffn_out)?)
    }

    fn attention(&self, x: &Tensor, rope_emb: &Tensor, block: &ErnieImageBlock) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = x.shape().dims();
        let (b, s, _) = (dims[0], dims[1], dims[2]);

        // Q/K/V — no bias
        let qt = block.attn_q_weight.transpose()?;
        let kt = block.attn_k_weight.transpose()?;
        let vt = block.attn_v_weight.transpose()?;
        let q = x.matmul(&qt)?;
        let k = x.matmul(&kt)?;
        let v = x.matmul(&vt)?;

        // Reshape to [B, S, H, D]
        let q = q.reshape(&[b, s, cfg.num_heads, cfg.head_dim])?;
        let k = k.reshape(&[b, s, cfg.num_heads, cfg.head_dim])?;
        let v = v.reshape(&[b, s, cfg.num_heads, cfg.head_dim])?;

        // QK RMSNorm (per head)
        let q = per_head_rms_norm(&q, &block.norm_q_weight, cfg.num_heads, cfg.head_dim, cfg.eps)?;
        let k = per_head_rms_norm(&k, &block.norm_k_weight, cfg.num_heads, cfg.head_dim, cfg.eps)?;

        // RoPE
        let q = apply_rotary_emb(&q, rope_emb)?;
        let k = apply_rotary_emb(&k, rope_emb)?;

        // SDPA: [B, H, S, D]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, s, cfg.hidden_size])?;

        // Output projection — no bias
        let ot = block.attn_out_weight.transpose()?;
        attn_out.matmul(&ot)
    }

    fn ffn(&self, x: &Tensor, block: &ErnieImageBlock) -> Result<Tensor> {
        // SwiGLU: down(up(x) * gelu(gate(x)))
        let gt = block.gate_proj_weight.transpose()?;
        let ut = block.up_proj_weight.transpose()?;
        let dt = block.down_proj_weight.transpose()?;

        let gate = x.matmul(&gt)?.gelu()?;
        let up = x.matmul(&ut)?;
        let activated = up.mul(&gate)?;
        activated.matmul(&dt)
    }

    fn unpatchify(&self, patches: &Tensor, hp: usize, wp: usize) -> Result<Tensor> {
        let dims = patches.shape().dims();
        let b = dims[0];
        let p = self.config.patch_size;
        let out_ch = self.config.out_channels;

        // [B, Hp*Wp, P*P*out_ch] → [B, Hp, Wp, P, P, out_ch] → [B, out_ch, Hp*P, Wp*P]
        let x = patches.reshape(&[b, hp, wp, p, p, out_ch])?;
        let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
        x.reshape(&[b, out_ch, hp * p, wp * p])
    }
}

// ---------------------------------------------------------------------------
// FlameSwap-backed model for 24GB inference
// ---------------------------------------------------------------------------

/// ERNIE-Image with FlameSwap block offloading. Shared weights on GPU,
/// block weights loaded one-at-a-time from CPU via FlameSwap.
pub struct ErnieImageSwapped {
    pub config: ErnieImageConfig,
    shared: HashMap<String, Tensor>,
    swap: flame_swap::FlameSwap,
    device: Arc<cudarc::driver::CudaDevice>,
}

impl ErnieImageSwapped {
    /// Load from safetensors shard(s). Block weights go to FlameSwap,
    /// shared weights stay on GPU.
    pub fn load(paths: &[&str], config: ErnieImageConfig, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let swap = flame_swap::FlameSwap::load(paths, device, |name| {
            if let Some(rest) = name.strip_prefix("layers.") {
                rest.split('.').next()?.parse::<usize>().ok()
            } else {
                None
            }
        }).map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap ERNIE: {e}")))?;

        let shared_prefixes = [
            "x_embedder.", "text_proj.", "time_embedding.", "time_proj.",
            "adaLN_modulation.", "final_norm.", "final_linear.", "pos_embed.",
        ];
        let mut shared = HashMap::new();
        for path in paths {
            let partial = flame_core::serialization::load_file_filtered(
                std::path::Path::new(path), device,
                |key| shared_prefixes.iter().any(|p| key.starts_with(p)),
            )?;
            for (k, v) in partial {
                let v_bf16 = if v.dtype() != DType::BF16 { v.to_dtype(DType::BF16).unwrap_or(v) } else { v };
                shared.insert(k, v_bf16);
            }
        }

        log::info!(
            "[ERNIE-Image] Loaded: {} blocks via FlameSwap, {} shared weights",
            config.num_layers, shared.len(),
        );

        Ok(Self { config, shared, swap, device: device.clone() })
    }

    fn get(&self, key: &str) -> Result<&Tensor> {
        self.shared.get(key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing shared: {key}")))
    }

    /// Forward pass with FlameSwap block iteration.
    pub fn forward(
        &mut self,
        image: &Tensor,
        timestep: &Tensor,
        text_embeds: &Tensor,
        text_lens: &[usize],
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = image.shape().dims();
        let (b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let p = cfg.patch_size;
        let (hp, wp) = (h / p, w / p);
        let n_img = hp * wp;
        let t_max = text_embeds.shape().dims()[1];

        // 1. Patch embed
        let patch_w = self.get("x_embedder.proj.weight")?;
        let patch_b = self.get("x_embedder.proj.bias")?;
        let img_tokens = {
            let x = image.reshape(&[b, dims[1], h * w])?.permute(&[0, 2, 1])?;
            let w_flat = patch_w.reshape(&[cfg.hidden_size, cfg.in_channels])?;
            let wt = w_flat.transpose()?;
            x.matmul(&wt)?.add(patch_b)?
        };

        // 2. Text projection
        let txt_tokens = if let Ok(tp) = self.get("text_proj.weight") {
            let wt = tp.transpose()?;
            text_embeds.matmul(&wt)?
        } else {
            text_embeds.clone()
        };

        // 3. Concatenate
        let mut x = Tensor::cat(&[&img_tokens, &txt_tokens], 1)?;

        // 4. RoPE
        let rope_emb = compute_rope_embeddings(hp, wp, t_max, cfg, &self.device)?;

        // 5. Time embedding + AdaLN modulation
        let te_w1 = self.get("time_embedding.linear_1.weight")?;
        let te_b1 = self.get("time_embedding.linear_1.bias")?;
        let te_w2 = self.get("time_embedding.linear_2.weight")?;
        let te_b2 = self.get("time_embedding.linear_2.bias")?;

        let t_emb = {
            let dim = cfg.hidden_size;
            let half = dim / 2;
            let t = timestep.to_dtype(DType::F32)?;
            let mut freqs = Vec::with_capacity(half);
            for i in 0..half {
                freqs.push((-((i as f64) / half as f64) * (10000.0f64).ln()).exp() as f32);
            }
            let freq_t = Tensor::from_vec(freqs, Shape::from_dims(&[1, half]), self.device.clone())?;
            let args = t.unsqueeze(1)?.mul(&freq_t)?;
            let emb = Tensor::cat(&[&args.sin()?, &args.cos()?], 1)?.to_dtype(DType::BF16)?;
            let h = emb.matmul(&te_w1.transpose()?)?.add(te_b1)?;
            h.silu()?.matmul(&te_w2.transpose()?)?.add(te_b2)?
        };
        let c = t_emb.clone();

        let adaln_w = self.get("adaLN_modulation.1.weight")?;
        let adaln_b = self.get("adaLN_modulation.1.bias")?;
        let mod_out = c.silu()?.matmul(&adaln_w.transpose()?)?.add(adaln_b)?;
        let chunks = mod_out.chunk(6, 1)?;
        let s_total = n_img + t_max;
        let expand = |t: &Tensor| -> Result<Tensor> {
            t.unsqueeze(1)?.expand(&[b, s_total, cfg.hidden_size])
        };
        let shift_msa = expand(&chunks[0])?;
        let scale_msa = expand(&chunks[1])?;
        let gate_msa = expand(&chunks[2])?;
        let shift_mlp = expand(&chunks[3])?;
        let scale_mlp = expand(&chunks[4])?;
        let gate_mlp = expand(&chunks[5])?;

        // 6. Block loop with FlameSwap
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..cfg.num_layers {
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < cfg.num_layers {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            x = block_forward_from_map(
                &x, &rope_emb,
                &shift_msa, &scale_msa, &gate_msa,
                &shift_mlp, &scale_mlp, &gate_mlp,
                &raw, cfg, &self.device,
            )?;

            if i % 6 == 0 || i == cfg.num_layers - 1 {
                log::info!("[ERNIE] Block {}/{}", i + 1, cfg.num_layers);
            }
        }

        // 7. Final norm + linear
        let fn_w = self.get("final_norm.linear.weight")?;
        let fn_b = self.get("final_norm.linear.bias")?;
        let fl_w = self.get("final_linear.weight")?;
        let fl_b = self.get("final_linear.bias")?;

        let final_mod = c.matmul(&fn_w.transpose()?)?.add(fn_b)?;
        let final_chunks = final_mod.chunk(2, 1)?;
        let f_scale = final_chunks[0].unsqueeze(1)?;
        let f_shift = final_chunks[1].unsqueeze(1)?;

        let x_norm = flame_core::layer_norm::layer_norm(&x, &[cfg.hidden_size], None, None, cfg.eps)?;
        let x_out = x_norm.mul(&f_scale.add_scalar(1.0)?)?.add(&f_shift)?;

        let patches = x_out.matmul(&fl_w.transpose()?)?.add(fl_b)?;
        let img_patches = patches.narrow(1, 0, n_img)?;

        // Unpatchify
        let out_ch = cfg.out_channels;
        let x_r = img_patches.reshape(&[b, hp, wp, p, p, out_ch])?;
        let x_r = x_r.permute(&[0, 5, 1, 3, 2, 4])?;
        x_r.reshape(&[b, out_ch, hp * p, wp * p])
    }
}

/// Run one block's forward using a weight HashMap (from FlameSwap).
fn block_forward_from_map(
    x: &Tensor,
    rope_emb: &Tensor,
    shift_msa: &Tensor, scale_msa: &Tensor, gate_msa: &Tensor,
    shift_mlp: &Tensor, scale_mlp: &Tensor, gate_mlp: &Tensor,
    weights: &HashMap<String, Tensor>,
    cfg: &ErnieImageConfig,
    _device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let get = |suffix: &str| -> Result<&Tensor> {
        // FlameSwap keys have the full `layers.{i}.` prefix
        // Find the key that ends with the suffix
        weights.values().next(); // just to use weights
        for (k, v) in weights {
            if k.ends_with(suffix) {
                return Ok(v);
            }
        }
        Err(flame_core::Error::InvalidInput(format!("missing block weight: *{suffix}")))
    };

    // Self-attention with AdaLN
    let residual = x.clone();
    let sa_norm_w = get("adaLN_sa_ln.weight")?;
    let normed = flame_core::norm::rms_norm(x, &[cfg.hidden_size], Some(sa_norm_w), cfg.eps)?;
    let modulated = normed.mul(&scale_msa.add_scalar(1.0)?)?.add(shift_msa)?;

    let dims = modulated.shape().dims();
    let (b, s, _) = (dims[0], dims[1], dims[2]);

    let q = modulated.matmul(&get("self_attention.to_q.weight")?.transpose()?)?;
    let k = modulated.matmul(&get("self_attention.to_k.weight")?.transpose()?)?;
    let v = modulated.matmul(&get("self_attention.to_v.weight")?.transpose()?)?;

    let q = q.reshape(&[b, s, cfg.num_heads, cfg.head_dim])?;
    let k = k.reshape(&[b, s, cfg.num_heads, cfg.head_dim])?;
    let v = v.reshape(&[b, s, cfg.num_heads, cfg.head_dim])?;

    let q = per_head_rms_norm(&q, get("self_attention.norm_q.weight")?, cfg.num_heads, cfg.head_dim, cfg.eps)?;
    let k = per_head_rms_norm(&k, get("self_attention.norm_k.weight")?, cfg.num_heads, cfg.head_dim, cfg.eps)?;

    let q = apply_rotary_emb(&q, rope_emb)?;
    let k = apply_rotary_emb(&k, rope_emb)?;

    let q = q.permute(&[0, 2, 1, 3])?;
    let k = k.permute(&[0, 2, 1, 3])?;
    let v = v.permute(&[0, 2, 1, 3])?;

    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, s, cfg.hidden_size])?;
    let attn_out = attn_out.matmul(&get("self_attention.to_out.0.weight")?.transpose()?)?;

    let x = residual.add(&gate_msa.mul(&attn_out)?)?;

    // FFN with AdaLN
    let residual = x.clone();
    let mlp_norm_w = get("adaLN_mlp_ln.weight")?;
    let normed = flame_core::norm::rms_norm(&x, &[cfg.hidden_size], Some(mlp_norm_w), cfg.eps)?;
    let modulated = normed.mul(&scale_mlp.add_scalar(1.0)?)?.add(shift_mlp)?;

    let gate = modulated.matmul(&get("mlp.gate_proj.weight")?.transpose()?)?.gelu()?;
    let up = modulated.matmul(&get("mlp.up_proj.weight")?.transpose()?)?;
    let ffn_out = up.mul(&gate)?.matmul(&get("mlp.linear_fc2.weight")?.transpose()?)?;

    residual.add(&gate_mlp.mul(&ffn_out)?)
}

fn per_head_rms_norm(x: &Tensor, weight: &Tensor, _num_heads: usize, head_dim: usize, eps: f32) -> Result<Tensor> {
    // x: [B, S, H, D] — normalize each head vector independently
    let dims = x.shape().dims().to_vec();
    let total: usize = dims.iter().product();
    let n_vecs = total / head_dim;
    let flat = x.reshape(&[n_vecs, head_dim])?;
    let normed = flame_core::norm::rms_norm(&flat.unsqueeze(0)?, &[head_dim], Some(weight), eps)?;
    normed.reshape(&dims)
}
