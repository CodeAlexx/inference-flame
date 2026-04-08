//! Qwen-Image DiT — pure Rust, flame-core + FlameSwap.
//!
//! Port of `QwenImageTransformer2DModel` from diffusers
//! (`src/diffusers/models/transformers/transformer_qwenimage.py`).
//!
//! ## Architecture
//! - **60 identical double-stream transformer blocks** (no single blocks)
//! - inner_dim = 24 × 128 = **3072**
//! - joint_attention_dim = **3584** (Qwen2.5-VL-7B hidden)
//! - in_channels = 64 (16 raw VAE channels × 2×2 patchify)
//! - out_channels = 16 (raw VAE channels)
//! - patch_size = 2
//! - axes_dims_rope = (16, 56, 56) → 3-axis RoPE over (frame, height, width)
//! - mlp_ratio = 4.0 (GELU-approximate FFN)
//!
//! ## Key differences from FLUX / Chroma
//! 1. **Per-block modulation** like FLUX (img_mod + txt_mod linears inside each
//!    block) — unlike Chroma which pulls from a shared `distilled_guidance_layer`.
//! 2. **All blocks are double-stream** — no single-stream trailing blocks.
//! 3. **Separate Q/K/V projections** per stream (`to_q`/`to_k`/`to_v` for img,
//!    `add_q_proj`/`add_k_proj`/`add_v_proj` for txt) — matches Chroma, differs
//!    from FLUX's fused `qkv` linear.
//! 4. **`txt_norm` RMSNorm before `txt_in`** — unique Qwen preprocessing.
//! 5. **`time_text_embed` is a simple MLP** (Linear → SiLU → Linear) with
//!    sinusoidal input (`Timesteps(num_channels=256, flip_sin_to_cos=True,
//!    scale=1000, downscale_freq_shift=0)`).
//! 6. **`norm_out` is AdaLayerNormContinuous**: `SiLU(temb) → Linear(dim, 2*dim) →
//!    chunk(shift, scale) → LayerNorm(x) * (1+scale) + shift`.
//! 7. **3-axis RoPE** over video shapes `(frame, height, width)` with
//!    `scale_rope=True` (symmetric ±half for h and w), computed on the fly
//!    from image shape instead of explicit img_ids.
//! 8. **True CFG** (not guidance-distilled) — the pipeline runs cond + uncond
//!    forwards per step. Inside this Rust struct, that's handled by the caller.
//!
//! ## Weight keys (confirmed from `Qwen/Qwen-Image-2512`)
//! ```
//! img_in.{weight,bias}                            [3072, 64]
//! txt_norm.weight                                 [3584]           (RMSNorm)
//! txt_in.{weight,bias}                            [3072, 3584]
//! time_text_embed.timestep_embedder.linear_1.{weight,bias}  [3072, 256]
//! time_text_embed.timestep_embedder.linear_2.{weight,bias}  [3072, 3072]
//! norm_out.linear.{weight,bias}                   [6144, 3072]     (outputs 2*dim for shift+scale)
//! proj_out.{weight,bias}                          [64, 3072]
//!
//! transformer_blocks.{i}.img_mod.1.{weight,bias}  [18432, 3072]    (6*dim, .0 is SiLU, .1 is Linear)
//! transformer_blocks.{i}.txt_mod.1.{weight,bias}  [18432, 3072]
//! transformer_blocks.{i}.attn.to_q.{weight,bias}  [3072, 3072]
//! transformer_blocks.{i}.attn.to_k.{weight,bias}
//! transformer_blocks.{i}.attn.to_v.{weight,bias}
//! transformer_blocks.{i}.attn.add_q_proj.{weight,bias}
//! transformer_blocks.{i}.attn.add_k_proj.{weight,bias}
//! transformer_blocks.{i}.attn.add_v_proj.{weight,bias}
//! transformer_blocks.{i}.attn.to_out.0.{weight,bias}
//! transformer_blocks.{i}.attn.to_add_out.{weight,bias}
//! transformer_blocks.{i}.attn.norm_q.weight          [128]        (RMSNorm on head_dim)
//! transformer_blocks.{i}.attn.norm_k.weight          [128]
//! transformer_blocks.{i}.attn.norm_added_q.weight    [128]
//! transformer_blocks.{i}.attn.norm_added_k.weight    [128]
//! transformer_blocks.{i}.img_mlp.net.0.proj.{weight,bias}  [12288, 3072]  (4x FFN)
//! transformer_blocks.{i}.img_mlp.net.2.{weight,bias}       [3072, 12288]
//! transformer_blocks.{i}.txt_mlp.net.0.proj.{weight,bias}
//! transformer_blocks.{i}.txt_mlp.net.2.{weight,bias}
//! ```
//!
//! ⚠️ This port is BUILD-COMPLETE but UNTESTED. The user is remote and
//! cannot reboot if something OOMs — DO NOT RUN until they validate.

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
pub struct QwenImageConfig {
    pub num_layers: usize,         // 60
    pub inner_dim: usize,          // 3072
    pub num_heads: usize,          // 24
    pub head_dim: usize,           // 128
    pub in_channels: usize,        // 64
    pub out_channels: usize,       // 16
    pub patch_size: usize,         // 2
    pub joint_attention_dim: usize, // 3584 (Qwen2.5-VL-7B hidden)
    pub mlp_ratio: f32,            // 4.0
    pub axes_dims_rope: [usize; 3], // (16, 56, 56)
    pub rope_theta: f64,           // 10000.0
    pub timestep_dim: usize,       // 256 (sinusoidal input to time_text_embed)
    pub eps: f32,                  // 1e-6
}

impl Default for QwenImageConfig {
    fn default() -> Self {
        Self {
            num_layers: 60,
            inner_dim: 3072,
            num_heads: 24,
            head_dim: 128,
            in_channels: 64,
            out_channels: 16,
            patch_size: 2,
            joint_attention_dim: 3584,
            mlp_ratio: 4.0,
            axes_dims_rope: [16, 56, 56],
            rope_theta: 10000.0,
            timestep_dim: 256,
            eps: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// QwenImageDit struct
// ---------------------------------------------------------------------------

pub struct QwenImageDit {
    shared: HashMap<String, Tensor>,
    swap: FlameSwap,
    config: QwenImageConfig,
    device: Arc<CudaDevice>,
}

impl QwenImageDit {
    /// Load Qwen-Image DiT from the diffusers-format sharded safetensors.
    pub fn load(
        checkpoint_paths: &[&str],
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = QwenImageConfig::default();

        // FlameSwap classifies blocks by the `transformer_blocks.{i}` prefix.
        let swap = FlameSwap::load(
            checkpoint_paths,
            device,
            |name| {
                if let Some(rest) = name.strip_prefix("transformer_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(idx);
                }
                None
            },
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap QwenImage: {e}")))?;

        // Shared weights: everything that's not in a transformer block.
        let shared_prefixes = [
            "img_in.",
            "txt_norm.",
            "txt_in.",
            "time_text_embed.",
            "norm_out.",
            "proj_out.",
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
            "[QwenImage] Loaded: {} blocks via FlameSwap, {} shared weights",
            swap.num_blocks(),
            shared_weights.len()
        );

        Ok(Self {
            shared: shared_weights,
            swap,
            config,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &QwenImageConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
    }

    fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
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

    /// Sinusoidal `Timesteps` matching diffusers:
    /// `Timesteps(num_channels=256, flip_sin_to_cos=True, scale=1000, downscale_freq_shift=0)`.
    ///
    /// Reference (transformer_qwenimage.py:43-94 `get_timestep_embedding`):
    /// ```python
    /// exponent = -log(max_period) * arange(half) / (half - downscale_freq_shift)
    /// emb = exp(exponent) * timesteps[:, None] * scale
    /// emb = cat(sin(emb), cos(emb))
    /// if flip_sin_to_cos: emb = cat(emb[:, half:], emb[:, :half])  # → cat(cos, sin)
    /// ```
    fn time_proj(&self, timestep: &Tensor) -> Result<Tensor> {
        let in_dtype = timestep.dtype();
        let dim = self.config.timestep_dim; // 256
        let half = dim / 2;
        let max_period = 10000.0f64;
        // downscale_freq_shift = 0 → (half - 0) = half
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(
            freq_data,
            Shape::from_dims(&[1, half]),
            self.device.clone(),
        )?;

        // scale = 1000
        let t_f32 = timestep.to_dtype(DType::F32)?.mul_scalar(1000.0)?;
        let args = t_f32.unsqueeze(1)?.matmul(&freqs)?; // [B, half]

        let sin_part = args.sin()?;
        let cos_part = args.cos()?;
        // flip_sin_to_cos=True → final order is (cos, sin). Diffusers builds
        // (sin, cos) and then index-swaps the halves; we skip the intermediate
        // and directly cat (cos, sin). The numerical result is byte-identical
        // because cat is a memcpy and sin/cos are computed independently.
        let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;
        emb.to_dtype(in_dtype)
    }

    /// `QwenTimestepProjEmbeddings.forward`: sinusoidal embed → Linear → SiLU → Linear.
    /// The diffusers `TimestepEmbedding` is a 2-Linear MLP with SiLU in between.
    fn time_text_embed(&self, timestep: &Tensor) -> Result<Tensor> {
        let proj = self.time_proj(timestep)?;

        let l1_w = self.shared.get("time_text_embed.timestep_embedder.linear_1.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_text_embed.timestep_embedder.linear_1.weight".into()))?;
        let l1_b = self.shared.get("time_text_embed.timestep_embedder.linear_1.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_text_embed.timestep_embedder.linear_1.bias".into()))?;
        let l2_w = self.shared.get("time_text_embed.timestep_embedder.linear_2.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_text_embed.timestep_embedder.linear_2.weight".into()))?;
        let l2_b = self.shared.get("time_text_embed.timestep_embedder.linear_2.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_text_embed.timestep_embedder.linear_2.bias".into()))?;

        // Linear expects [B, N, C] for fused_linear3d_native. Our proj is [B, C].
        // Unsqueeze to [B, 1, C], run, squeeze back.
        let x = proj.unsqueeze(1)?;
        let h = Self::linear_bias(&x, l1_w, l1_b)?;
        let h = h.silu()?;
        let out = Self::linear_bias(&h, l2_w, l2_b)?;
        out.squeeze(Some(1))
    }

    /// Build the RoPE (cos, sin) interleaved tables for joint (txt, img) attention.
    ///
    /// Qwen uses 3-axis RoPE over `(frame, height, width)` with `axes_dim = (16, 56, 56)`
    /// and `scale_rope=True` (symmetric ±half for h and w). For a single-frame image
    /// at (h, w) latent patch-grid, frame=1.
    ///
    /// The result shape is `[1, 1, N_txt+N_img, sum(axes_dim)/2]` in the FLUX
    /// interleaved-pair format that `rope_fused_bf16` consumes (same kernel that
    /// FLUX, Klein, and LTX use).
    ///
    /// Reference: `QwenEmbedRope._compute_video_freqs` (transformer_qwenimage.py:306-328).
    pub fn build_rope_tables(
        frame: usize,
        height: usize,
        width: usize,
        txt_seq_len: usize,
        config: &QwenImageConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        // Build per-axis angle tables for a wide range, then slice per-axis.
        //
        // We take a simpler approach than the diffusers version: compute
        // positions directly for each token instead of precomputing a giant
        // [4096, total_dim] table and slicing.
        //
        // Layout: tokens are (txt[0..L_txt], img[0..F*H*W]) along the seq dim.
        //
        // For each token we emit one angle per axis, concatenated into a flat
        // per-axis vector of length (axis_dim / 2), then concatenated across
        // axes into a final per-token vector of length sum(axis_dim)/2. Then
        // all tokens are stacked to form [N, sum/2].

        let axes = &config.axes_dims_rope;
        let total_half: usize = axes.iter().map(|d| d / 2).sum();

        let n_img = frame * height * width;
        let n_total = txt_seq_len + n_img;

        // We'll build everything in FP32 and cast at the end.
        let mut angle_data = vec![0.0f32; n_total * total_half];

        // Per-axis frequency tables.
        let theta = config.rope_theta;
        let freq_tables: Vec<Vec<f32>> = axes
            .iter()
            .map(|&axis_dim| {
                let half = axis_dim / 2;
                (0..half)
                    .map(|i| {
                        let scale = (2 * i) as f64 / axis_dim as f64;
                        (1.0 / theta.powf(scale)) as f32
                    })
                    .collect()
            })
            .collect();

        // Text tokens: Qwen uses a forward position range starting from
        // `max_vid_index`, which with scale_rope=True equals max(h/2, w/2).
        // The text gets positions [max_vid_index, max_vid_index + txt_seq_len).
        let max_vid_index = (height / 2).max(width / 2).max(1);
        for t in 0..txt_seq_len {
            let pos = (max_vid_index + t) as f32;
            let mut offset = 0usize;
            for (axis_i, freqs) in freq_tables.iter().enumerate() {
                let half = axes[axis_i] / 2;
                for i in 0..half {
                    angle_data[t * total_half + offset + i] = pos * freqs[i];
                }
                offset += half;
            }
        }

        // Image tokens: 3-axis positions over (frame, h, w), with
        // scale_rope=True the h and w axes use symmetric positions
        // in [-h/2, h/2) and [-w/2, w/2).
        for f_idx in 0..frame {
            for h_idx in 0..height {
                for w_idx in 0..width {
                    let token_idx = txt_seq_len + f_idx * height * width + h_idx * width + w_idx;
                    let f_pos = f_idx as f32;
                    // scale_rope=True: symmetric around center.
                    let h_pos = h_idx as f32 - (height as f32) / 2.0;
                    let w_pos = w_idx as f32 - (width as f32) / 2.0;

                    let positions = [f_pos, h_pos, w_pos];
                    let mut offset = 0usize;
                    for (axis_i, freqs) in freq_tables.iter().enumerate() {
                        let half = axes[axis_i] / 2;
                        let pos = positions[axis_i];
                        for i in 0..half {
                            angle_data[token_idx * total_half + offset + i] = pos * freqs[i];
                        }
                        offset += half;
                    }
                }
            }
        }

        let angles = Tensor::from_vec(
            angle_data,
            Shape::from_dims(&[n_total, total_half]),
            device.clone(),
        )?;
        let cos_full = angles.cos()?;
        let sin_full = angles.sin()?;

        // [1, 1, N, total_half] BF16 — shape consumed by rope_fused_bf16.
        let pe_cos = cos_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let pe_sin = sin_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((pe_cos, pe_sin))
    }

    fn apply_rope_complex(q: &Tensor, k: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_out = flame_core::bf16_ops::rope_fused_bf16(q, pe_cos, pe_sin)?;
        let k_out = flame_core::bf16_ops::rope_fused_bf16(k, pe_cos, pe_sin)?;
        Ok((q_out, k_out))
    }

    /// Multi-region RoPE table for Qwen-Image-Edit (or any pipeline that
    /// concatenates multiple image regions in the seq dim).
    ///
    /// `regions` is the list of `(frame, height, width)` patch grids in the
    /// SAME order they appear in the latent seq. For Edit:
    ///   regions[0] = (1, h_target / 2, w_target / 2)   — the noisy target
    ///   regions[1] = (1, h_ref    / 2, w_ref    / 2)   — the clean reference
    ///
    /// The text positions start at `max_vid_index = max over all regions of
    /// max(h/2, w/2)` (with `scale_rope=True`), matching `QwenEmbedRope.forward`
    /// in the diffusers reference (`transformer_qwenimage.py:286-304`).
    ///
    /// Each region's frame axis uses `frame_offset = idx`, so the second
    /// region's frame index starts at 1 (and so on for additional regions).
    /// This matches the diffusers `_compute_video_freqs(frame, height, width, idx)`
    /// behavior at line 317: `freqs_pos[0][idx : idx + frame]`.
    ///
    /// Token order in the returned table is `(txt[0..L_txt], region0[0..F0*H0*W0], region1[...], ...)`.
    pub fn build_rope_tables_multi(
        regions: &[(usize, usize, usize)],
        txt_seq_len: usize,
        config: &QwenImageConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let axes = &config.axes_dims_rope;
        let total_half: usize = axes.iter().map(|d| d / 2).sum();

        let n_img: usize = regions.iter().map(|(f, h, w)| f * h * w).sum();
        let n_total = txt_seq_len + n_img;

        let mut angle_data = vec![0.0f32; n_total * total_half];

        // Per-axis frequency tables (same as the single-region path).
        let theta = config.rope_theta;
        let freq_tables: Vec<Vec<f32>> = axes
            .iter()
            .map(|&axis_dim| {
                let half = axis_dim / 2;
                (0..half)
                    .map(|i| {
                        let scale = (2 * i) as f64 / axis_dim as f64;
                        (1.0 / theta.powf(scale)) as f32
                    })
                    .collect()
            })
            .collect();

        // max_vid_index is computed across ALL regions (so the text positions
        // sit beyond the largest image position). With scale_rope=True the
        // image positions are symmetric around 0 with magnitude up to h/2.
        let max_vid_index = regions
            .iter()
            .map(|(_f, h, w)| (h / 2).max(w / 2))
            .max()
            .unwrap_or(0)
            .max(1);

        // Text tokens come first in the seq.
        for t in 0..txt_seq_len {
            let pos = (max_vid_index + t) as f32;
            let mut offset = 0usize;
            for (axis_i, freqs) in freq_tables.iter().enumerate() {
                let half = axes[axis_i] / 2;
                for i in 0..half {
                    angle_data[t * total_half + offset + i] = pos * freqs[i];
                }
                offset += half;
            }
        }

        // Image tokens, region by region. Each region's frame positions are
        // offset by the region index (matching diffusers `idx : idx + frame`).
        let mut token_offset = txt_seq_len;
        for (region_idx, &(frame, height, width)) in regions.iter().enumerate() {
            // Symmetric centering matches Python's
            //   torch.cat([freqs_neg[-(H - H//2):], freqs_pos[:H//2]])
            // which yields positions [-(H - H//2), ..., -1, 0, 1, ..., H//2 - 1].
            // Equivalent integer formula: pos = i - (H - H//2). Computing in
            // i64 then casting to f32 avoids the half-integer drift that
            // `i as f32 - H as f32 / 2.0` introduces for odd H.
            let h_neg = (height - height / 2) as i64;
            let w_neg = (width - width / 2) as i64;
            for f_idx in 0..frame {
                for h_idx in 0..height {
                    for w_idx in 0..width {
                        let token_idx = token_offset
                            + f_idx * height * width
                            + h_idx * width
                            + w_idx;
                        // Frame position is offset by region index — see
                        // `_compute_video_freqs` line 317 in the Python ref.
                        let f_pos = (region_idx + f_idx) as f32;
                        let h_pos = (h_idx as i64 - h_neg) as f32;
                        let w_pos = (w_idx as i64 - w_neg) as f32;

                        let positions = [f_pos, h_pos, w_pos];
                        let mut offset = 0usize;
                        for (axis_i, freqs) in freq_tables.iter().enumerate() {
                            let half = axes[axis_i] / 2;
                            let pos = positions[axis_i];
                            for i in 0..half {
                                angle_data[token_idx * total_half + offset + i] = pos * freqs[i];
                            }
                            offset += half;
                        }
                    }
                }
            }
            token_offset += frame * height * width;
        }

        let angles = Tensor::from_vec(
            angle_data,
            Shape::from_dims(&[n_total, total_half]),
            device.clone(),
        )?;
        let cos_full = angles.cos()?;
        let sin_full = angles.sin()?;

        let pe_cos = cos_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let pe_sin = sin_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((pe_cos, pe_sin))
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Forward pass.
    ///
    /// Arguments:
    /// - `hidden_states`: packed image latents `[B, N_img, in_channels=64]`
    /// - `encoder_hidden_states`: Qwen2.5-VL hidden states `[B, N_txt, 3584]`
    ///   (already extracted, system-prompt tokens already dropped by Stage 1)
    /// - `timestep`: `[B]` scalar timesteps in `[0, 1]`
    /// - `img_shape`: `(frame, height, width)` — for a single-frame 1024²
    ///   image with `vae_scale=8` and `patch_size=2` the latent grid is
    ///   `(1, 64, 64)`.
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_shape: (usize, usize, usize),
    ) -> Result<Tensor> {
        let cfg = self.config.clone();
        let img_len = hidden_states.shape().dims()[1];
        let txt_len = encoder_hidden_states.shape().dims()[1];

        // ── img_in ──
        let img_in_w = self.shared.get("img_in.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.weight".into()))?;
        let img_in_b = self.shared.get("img_in.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.bias".into()))?;
        let mut img = Self::linear_bias(hidden_states, img_in_w, img_in_b)?;

        // ── txt_norm (RMSNorm) then txt_in ──
        let txt_norm_w = self.shared.get("txt_norm.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_norm.weight".into()))?;
        let txt_in_w = self.shared.get("txt_in.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.weight".into()))?;
        let txt_in_b = self.shared.get("txt_in.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.bias".into()))?;
        let txt_normed = Self::rms_norm(encoder_hidden_states, txt_norm_w, 1e-6)?;
        let mut txt = Self::linear_bias(&txt_normed, txt_in_w, txt_in_b)?;

        // ── time_text_embed ──
        let temb = self.time_text_embed(timestep)?; // [B, inner_dim]

        // ── RoPE tables for joint [txt, img] attention ──
        let (frame, h_latent, w_latent) = img_shape;
        let (pe_cos, pe_sin) = Self::build_rope_tables(
            frame,
            h_latent,
            w_latent,
            txt_len,
            &cfg,
            &self.device,
        )?;

        // ── Transformer blocks ──
        let total_blocks = cfg.num_layers;
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..total_blocks {
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            let (new_img, new_txt) =
                self.block_forward(&img, &txt, &temb, &pe_cos, &pe_sin, &raw, i)?;
            img = new_img;
            txt = new_txt;

            if i % 10 == 0 || i == total_blocks - 1 {
                log::info!("[QwenImage] Block {}/{}", i + 1, total_blocks);
            }
        }

        // ── norm_out: AdaLayerNormContinuous ──
        //   shift, scale = chunk(norm_out.linear(SiLU(temb)), 2)
        //   out = LayerNorm(x) * (1 + scale)[:, None] + shift[:, None]
        let norm_out_w = self.shared.get("norm_out.linear.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing norm_out.linear.weight".into()))?;
        let norm_out_b = self.shared.get("norm_out.linear.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing norm_out.linear.bias".into()))?;
        let temb_silu = temb.silu()?;
        let mods = Self::linear_bias(&temb_silu.unsqueeze(1)?, norm_out_w, norm_out_b)?
            .squeeze(Some(1))?;
        // mods: [B, 2*dim]. Shift first, then scale (per diffusers convention).
        let shift = mods.narrow(1, 0, cfg.inner_dim)?;
        let scale = mods.narrow(1, cfg.inner_dim, cfg.inner_dim)?;
        let normed = Self::layer_norm_no_affine(&img, 1e-6)?;
        let one_plus = scale.add_scalar(1.0)?;
        let scaled = normed.mul(&one_plus.unsqueeze(1)?)?;
        let modulated = scaled.add(&shift.unsqueeze(1)?)?;

        // ── proj_out ──
        let proj_w = self.shared.get("proj_out.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing proj_out.weight".into()))?;
        let proj_b = self.shared.get("proj_out.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing proj_out.bias".into()))?;
        let out = Self::linear_bias(&modulated, proj_w, proj_b)?;
        let _ = img_len; // keep variable
        Ok(out)
    }

    /// Forward pass for **Qwen-Image-Edit** (image editing variant).
    ///
    /// Same DiT model as the T2I `forward`, but accepts a **concatenated**
    /// image latent containing the noisy target tokens followed by the clean
    /// reference image tokens, plus a multi-region RoPE table that gives each
    /// region its own frame position offset.
    ///
    /// Reference: `pipeline_qwenimage_edit.py:806-825` —
    /// ```python
    /// latent_model_input = torch.cat([latents, image_latents], dim=1)
    /// noise_pred = self.transformer(
    ///     hidden_states=latent_model_input,
    ///     img_shapes=[(1, h//16, w//16), (1, ref_h//16, ref_w//16)],
    ///     ...
    /// )[0]
    /// noise_pred = noise_pred[:, :latents.size(1)]   # ← strip the reference portion
    /// ```
    ///
    /// Caller responsibility:
    /// 1. Concatenate `[noisy_target, image_latents]` along the seq dim before
    ///    passing as `hidden_states`.
    /// 2. Pass `image_regions` matching the regions in the same order:
    ///    `&[(1, h_target/2, w_target/2), (1, h_ref/2, w_ref/2)]`.
    /// 3. After this returns, slice the noise_pred to keep only the FIRST
    ///    `target_seq_len` tokens (drop the reference portion before the
    ///    Euler step).
    pub fn forward_edit(
        &mut self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        image_regions: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        let cfg = self.config.clone();
        let img_len = hidden_states.shape().dims()[1];
        let txt_len = encoder_hidden_states.shape().dims()[1];

        // Sanity: the input seq must equal the sum over all region patches.
        let expected_img_len: usize = image_regions
            .iter()
            .map(|(f, h, w)| f * h * w)
            .sum();
        if expected_img_len != img_len {
            return Err(flame_core::Error::InvalidInput(format!(
                "forward_edit: hidden_states seq_len {} != sum(image_regions) {} (regions={:?})",
                img_len, expected_img_len, image_regions
            )));
        }

        // ── img_in ──
        let img_in_w = self.shared.get("img_in.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.weight".into()))?;
        let img_in_b = self.shared.get("img_in.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.bias".into()))?;
        let mut img = Self::linear_bias(hidden_states, img_in_w, img_in_b)?;

        // ── txt_norm (RMSNorm) then txt_in ──
        let txt_norm_w = self.shared.get("txt_norm.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_norm.weight".into()))?;
        let txt_in_w = self.shared.get("txt_in.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.weight".into()))?;
        let txt_in_b = self.shared.get("txt_in.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.bias".into()))?;
        let txt_normed = Self::rms_norm(encoder_hidden_states, txt_norm_w, 1e-6)?;
        let mut txt = Self::linear_bias(&txt_normed, txt_in_w, txt_in_b)?;

        // ── time_text_embed ──
        let temb = self.time_text_embed(timestep)?;

        // ── Multi-region RoPE tables ──
        let (pe_cos, pe_sin) = Self::build_rope_tables_multi(
            image_regions,
            txt_len,
            &cfg,
            &self.device,
        )?;

        // ── Transformer blocks (same as T2I forward) ──
        let total_blocks = cfg.num_layers;
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..total_blocks {
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            let (new_img, new_txt) =
                self.block_forward(&img, &txt, &temb, &pe_cos, &pe_sin, &raw, i)?;
            img = new_img;
            txt = new_txt;

            if i % 10 == 0 || i == total_blocks - 1 {
                log::info!("[QwenImageEdit] Block {}/{}", i + 1, total_blocks);
            }
        }

        // ── norm_out + proj_out (same as T2I) ──
        let norm_out_w = self.shared.get("norm_out.linear.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing norm_out.linear.weight".into()))?;
        let norm_out_b = self.shared.get("norm_out.linear.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing norm_out.linear.bias".into()))?;
        let temb_silu = temb.silu()?;
        let mods = Self::linear_bias(&temb_silu.unsqueeze(1)?, norm_out_w, norm_out_b)?
            .squeeze(Some(1))?;
        let shift = mods.narrow(1, 0, cfg.inner_dim)?;
        let scale = mods.narrow(1, cfg.inner_dim, cfg.inner_dim)?;
        let normed = Self::layer_norm_no_affine(&img, 1e-6)?;
        let one_plus = scale.add_scalar(1.0)?;
        let scaled = normed.mul(&one_plus.unsqueeze(1)?)?;
        let modulated = scaled.add(&shift.unsqueeze(1)?)?;

        let proj_w = self.shared.get("proj_out.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing proj_out.weight".into()))?;
        let proj_b = self.shared.get("proj_out.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing proj_out.bias".into()))?;
        // Returns the FULL prediction over (target + reference) tokens. The
        // caller is responsible for slicing `[:, :target_seq_len, :]` before
        // the Euler step. (See pipeline_qwenimage_edit.py:825.)
        Self::linear_bias(&modulated, proj_w, proj_b)
    }

    // -----------------------------------------------------------------------
    // Block forward
    // -----------------------------------------------------------------------

    fn block_forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let dim = cfg.inner_dim;
        let prefix = format!("transformer_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let b = img.shape().dims()[0];
        let n_img = img.shape().dims()[1];
        let n_txt = txt.shape().dims()[1];

        // ── img_mod(temb) and txt_mod(temb) ──
        // nn.Sequential(SiLU, Linear(dim, 6*dim))
        //
        // `temb` arrives as [B, dim]. The Linear expects 3D input; unsqueeze to
        // [B, 1, dim] and squeeze back.
        let temb_silu = temb.silu()?;
        let img_mod_w = w("img_mod.1.weight")?;
        let img_mod_b = w("img_mod.1.bias")?;
        let txt_mod_w = w("txt_mod.1.weight")?;
        let txt_mod_b = w("txt_mod.1.bias")?;
        let img_mods = Self::linear_bias(&temb_silu.unsqueeze(1)?, img_mod_w, img_mod_b)?
            .squeeze(Some(1))?;
        let txt_mods = Self::linear_bias(&temb_silu.unsqueeze(1)?, txt_mod_w, txt_mod_b)?
            .squeeze(Some(1))?;
        // Split each into two halves (norm1, norm2), each 3*dim:
        //   [shift, scale, gate] for norm1 and then [shift, scale, gate] for norm2.
        //
        // Reference (transformer_qwenimage.py:677-679):
        //   img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # [B, 3*dim]
        //   txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        let img_mod1 = img_mods.narrow(1, 0, 3 * dim)?;
        let img_mod2 = img_mods.narrow(1, 3 * dim, 3 * dim)?;
        let txt_mod1 = txt_mods.narrow(1, 0, 3 * dim)?;
        let txt_mod2 = txt_mods.narrow(1, 3 * dim, 3 * dim)?;

        let img_shift1 = img_mod1.narrow(1, 0, dim)?;
        let img_scale1 = img_mod1.narrow(1, dim, dim)?;
        let img_gate1  = img_mod1.narrow(1, 2 * dim, dim)?;
        let img_shift2 = img_mod2.narrow(1, 0, dim)?;
        let img_scale2 = img_mod2.narrow(1, dim, dim)?;
        let img_gate2  = img_mod2.narrow(1, 2 * dim, dim)?;

        let txt_shift1 = txt_mod1.narrow(1, 0, dim)?;
        let txt_scale1 = txt_mod1.narrow(1, dim, dim)?;
        let txt_gate1  = txt_mod1.narrow(1, 2 * dim, dim)?;
        let txt_shift2 = txt_mod2.narrow(1, 0, dim)?;
        let txt_scale2 = txt_mod2.narrow(1, dim, dim)?;
        let txt_gate2  = txt_mod2.narrow(1, 2 * dim, dim)?;

        // ── norm1 + modulate for both streams ──
        //   norm(x) * (1 + scale)[:, None] + shift[:, None]
        let img_normed = Self::layer_norm_no_affine(img, 1e-6)?;
        let img_modulated = img_normed
            .mul(&img_scale1.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&img_shift1.unsqueeze(1)?)?;

        let txt_normed = Self::layer_norm_no_affine(txt, 1e-6)?;
        let txt_modulated = txt_normed
            .mul(&txt_scale1.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&txt_shift1.unsqueeze(1)?)?;

        // ── Separate Q/K/V projections per stream ──
        let img_q = Self::linear_bias(&img_modulated, w("attn.to_q.weight")?, w("attn.to_q.bias")?)?;
        let img_k = Self::linear_bias(&img_modulated, w("attn.to_k.weight")?, w("attn.to_k.bias")?)?;
        let img_v = Self::linear_bias(&img_modulated, w("attn.to_v.weight")?, w("attn.to_v.bias")?)?;

        let txt_q = Self::linear_bias(&txt_modulated, w("attn.add_q_proj.weight")?, w("attn.add_q_proj.bias")?)?;
        let txt_k = Self::linear_bias(&txt_modulated, w("attn.add_k_proj.weight")?, w("attn.add_k_proj.bias")?)?;
        let txt_v = Self::linear_bias(&txt_modulated, w("attn.add_v_proj.weight")?, w("attn.add_v_proj.bias")?)?;

        // ── Reshape to [B, H, N, D] ──
        let img_q = img_q.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;
        let img_k = img_k.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;
        let img_v = img_v.reshape(&[b, n_img, h, d])?.permute(&[0, 2, 1, 3])?;
        let txt_q = txt_q.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;
        let txt_k = txt_k.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;
        let txt_v = txt_v.reshape(&[b, n_txt, h, d])?.permute(&[0, 2, 1, 3])?;

        // ── QK RMSNorm ──
        let img_q = Self::rms_norm(&img_q, w("attn.norm_q.weight")?, 1e-6)?;
        let img_k = Self::rms_norm(&img_k, w("attn.norm_k.weight")?, 1e-6)?;
        let txt_q = Self::rms_norm(&txt_q, w("attn.norm_added_q.weight")?, 1e-6)?;
        let txt_k = Self::rms_norm(&txt_k, w("attn.norm_added_k.weight")?, 1e-6)?;

        // ── Concat txt + img (same order as diffusers) and apply RoPE ──
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;
        let (q, k) = Self::apply_rope_complex(&q, &k, pe_cos, pe_sin)?;

        // ── Joint SDPA ──
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // ── Split, permute, project out ──
        let txt_attn = attn_out.narrow(2, 0, n_txt)?;
        let img_attn = attn_out.narrow(2, n_txt, n_img)?;
        let img_attn = img_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img, h * d])?;
        let txt_attn = txt_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_txt, h * d])?;

        let img_attn = Self::linear_bias(&img_attn, w("attn.to_out.0.weight")?, w("attn.to_out.0.bias")?)?;
        let txt_attn = Self::linear_bias(&txt_attn, w("attn.to_add_out.weight")?, w("attn.to_add_out.bias")?)?;

        // ── Gated residual (using gate1) ──
        let img = img.add(&img_gate1.unsqueeze(1)?.mul(&img_attn)?)?;
        let txt = txt.add(&txt_gate1.unsqueeze(1)?.mul(&txt_attn)?)?;

        // ── FFN path for img ──
        let img_normed2 = Self::layer_norm_no_affine(&img, 1e-6)?;
        let img_mlp_in = img_normed2
            .mul(&img_scale2.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&img_shift2.unsqueeze(1)?)?;
        let img_mlp = Self::linear_bias(&img_mlp_in, w("img_mlp.net.0.proj.weight")?, w("img_mlp.net.0.proj.bias")?)?;
        let img_mlp = img_mlp.gelu()?;
        let img_mlp = Self::linear_bias(&img_mlp, w("img_mlp.net.2.weight")?, w("img_mlp.net.2.bias")?)?;
        let img = img.add(&img_gate2.unsqueeze(1)?.mul(&img_mlp)?)?;

        // ── FFN path for txt ──
        let txt_normed2 = Self::layer_norm_no_affine(&txt, 1e-6)?;
        let txt_mlp_in = txt_normed2
            .mul(&txt_scale2.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&txt_shift2.unsqueeze(1)?)?;
        let txt_mlp = Self::linear_bias(&txt_mlp_in, w("txt_mlp.net.0.proj.weight")?, w("txt_mlp.net.0.proj.bias")?)?;
        let txt_mlp = txt_mlp.gelu()?;
        let txt_mlp = Self::linear_bias(&txt_mlp, w("txt_mlp.net.2.weight")?, w("txt_mlp.net.2.bias")?)?;
        let txt = txt.add(&txt_gate2.unsqueeze(1)?.mul(&txt_mlp)?)?;

        let _ = Self::linear_nobias; // silence unused
        Ok((img, txt))
    }
}
