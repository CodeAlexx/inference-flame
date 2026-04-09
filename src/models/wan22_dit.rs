//! Wan2.2 DiT — pure Rust, flame-core + FlameSwap.
//!
//! Port of `WanModel` from `wan/modules/model.py` in the Wan2.2 repo.
//!
//! ## Architecture (A14B T2V variant)
//! - **40 WanAttentionBlocks** with separate self-attn + cross-attn (NOT joint)
//! - dim = 5120, ffn_dim = 13824, num_heads = 40, head_dim = 128
//! - in_channels = 16, out_channels = 16, patch_size = (1, 2, 2)
//! - freq_dim = 256 (sinusoidal timestep embedding)
//! - text_dim = 4096 (UMT5-XXL hidden), text_len = 512
//! - qk_norm = True, cross_attn_norm = True (LayerNorm with affine on cross input)
//!
//! ## Key differences from FLUX / QwenImage
//! 1. **Separate self-attn + cross-attn** per block (not joint attention).
//! 2. **Per-token timestep**: `t` expanded to `[B, seq_len]`, sinusoidal embed
//!    per token → time MLP → `[B, seq_len, 6, dim]` modulation per block.
//! 3. **Additive block modulation**: each block has a learnable `[1, 6, dim]`
//!    parameter ADDED to the time projection before chunking into 6 modulations.
//! 4. **Complex-number 3-axis RoPE**: `view_as_complex → multiply → view_as_real`
//!    with axes split `[d-4*(d//6), 2*(d//6), 2*(d//6)]` = `[44, 42, 42]`.
//! 5. **Dual expert**: two separate DiT models (high_noise + low_noise),
//!    switched at `boundary = 0.875 * 1000 = 875` timesteps.
//!
//! ## Weight keys (from `wan2.2_t2v_*_14b_fp16.safetensors`)
//! ```
//! patch_embedding.{weight,bias}                     [5120, 16, 1, 2, 2] / [5120]
//! text_embedding.0.{weight,bias}                    [5120, 4096] / [5120]
//! text_embedding.2.{weight,bias}                    [5120, 5120] / [5120]
//! time_embedding.0.{weight,bias}                    [5120, 256] / [5120]
//! time_embedding.2.{weight,bias}                    [5120, 5120] / [5120]
//! time_projection.1.{weight,bias}                   [30720, 5120] / [30720]
//! head.head.{weight,bias}                           [64, 5120] / [64]
//! head.modulation                                   [1, 2, 5120]
//!
//! blocks.{i}.modulation                             [1, 6, 5120]
//! blocks.{i}.self_attn.{q,k,v,o}.{weight,bias}     [5120, 5120] / [5120]
//! blocks.{i}.self_attn.norm_{q,k}.weight            [5120]
//! blocks.{i}.cross_attn.{q,k,v,o}.{weight,bias}    [5120, 5120] / [5120]
//! blocks.{i}.cross_attn.norm_{q,k}.weight           [5120]
//! blocks.{i}.norm3.{weight,bias}                    [5120] / [5120]  (cross_attn_norm)
//! blocks.{i}.ffn.0.{weight,bias}                    [13824, 5120] / [13824]
//! blocks.{i}.ffn.2.{weight,bias}                    [5120, 13824] / [5120]
//! ```

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
pub struct Wan22Config {
    pub num_layers: usize,         // 40
    pub dim: usize,                // 5120
    pub ffn_dim: usize,            // 13824
    pub num_heads: usize,          // 40
    pub head_dim: usize,           // 128
    pub in_channels: usize,        // 16
    pub out_channels: usize,       // 16
    pub patch_size: [usize; 3],    // (1, 2, 2)
    pub freq_dim: usize,           // 256
    pub text_dim: usize,           // 4096
    pub text_len: usize,           // 512
    pub eps: f32,                  // 1e-6
    pub rope_theta: f64,           // 10000.0
}

impl Default for Wan22Config {
    fn default() -> Self {
        Self {
            num_layers: 40,
            dim: 5120,
            ffn_dim: 13824,
            num_heads: 40,
            head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            patch_size: [1, 2, 2],
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Wan22Dit struct
// ---------------------------------------------------------------------------

pub struct Wan22Dit {
    shared: HashMap<String, Tensor>,
    swap: FlameSwap,
    config: Wan22Config,
    device: Arc<CudaDevice>,
    /// Precomputed complex RoPE frequencies [1024, head_dim/2] as complex pairs.
    /// Split into 3 axes: [1024, ax0], [1024, ax1], [1024, ax2].
    rope_freqs_cos: Vec<Vec<f32>>,  // [3][max_seq * axis_half]
    rope_freqs_sin: Vec<Vec<f32>>,
    rope_axes: [usize; 3],          // [44, 42, 42] half-dims per axis
}

impl Wan22Dit {
    /// Load Wan2.2 DiT from a single-file safetensors checkpoint.
    pub fn load(
        checkpoint_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = Wan22Config::default();

        // FlameSwap classifies blocks by the `blocks.{i}` prefix.
        let swap = FlameSwap::load(
            &[checkpoint_path],
            device,
            |name| {
                if let Some(rest) = name.strip_prefix("blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(idx);
                }
                None
            },
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap Wan22: {e}")))?;

        // Shared weights: everything not in a block.
        let shared_prefixes = [
            "patch_embedding.",
            "text_embedding.",
            "time_embedding.",
            "time_projection.",
            "head.",
        ];
        let part = load_file_filtered(Path::new(checkpoint_path), device, |key| {
            shared_prefixes.iter().any(|p| key.starts_with(p))
        })?;
        // Convert all shared weights to BF16 (source may be F16 loaded as F32).
        let shared: HashMap<String, Tensor> = part.into_iter().map(|(k, v)| {
            let v_bf16 = if v.dtype() != DType::BF16 {
                v.to_dtype(DType::BF16).unwrap_or(v)
            } else { v };
            (k, v_bf16)
        }).collect();

        log::info!(
            "[Wan22] Loaded: {} blocks via FlameSwap, {} shared weights",
            swap.num_blocks(),
            shared.len()
        );

        // Precompute RoPE frequency tables (complex representation).
        // Axes split: d=128, d//6=21, so axes = [128 - 4*21, 2*21, 2*21] = [44, 42, 42]
        let d = config.head_dim;
        let d6 = d / 6;
        let axes = [d - 4 * d6, 2 * d6, 2 * d6];
        let max_seq: usize = 1024;
        let theta = config.rope_theta;

        let mut rope_cos = Vec::new();
        let mut rope_sin = Vec::new();
        for &axis_dim in &axes {
            let half = axis_dim / 2;
            let mut cos_data = vec![0.0f32; max_seq * half];
            let mut sin_data = vec![0.0f32; max_seq * half];
            for pos in 0..max_seq {
                for i in 0..half {
                    let freq = 1.0 / theta.powf(2.0 * i as f64 / axis_dim as f64);
                    let angle = pos as f64 * freq;
                    cos_data[pos * half + i] = angle.cos() as f32;
                    sin_data[pos * half + i] = angle.sin() as f32;
                }
            }
            rope_cos.push(cos_data);
            rope_sin.push(sin_data);
        }

        Ok(Self {
            shared,
            swap,
            config,
            device: device.clone(),
            rope_freqs_cos: rope_cos,
            rope_freqs_sin: rope_sin,
            rope_axes: axes,
        })
    }

    pub fn config(&self) -> &Wan22Config {
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

    fn layer_norm_affine(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, Some(weight), Some(bias), eps)?;
        out.reshape(&dims)
    }

    /// Sinusoidal timestep embedding matching Wan's `sinusoidal_embedding_1d`.
    ///
    /// Reference (model.py:14-24):
    /// ```python
    /// half = dim // 2
    /// sinusoid = outer(position, pow(10000, -arange(half)/half))
    /// x = cat(cos(sinusoid), sin(sinusoid))
    /// ```
    ///
    /// Input: `timesteps` [N] in FP32 (integer timesteps 0-1000).
    /// Output: [N, freq_dim] in FP32.
    fn sinusoidal_embedding(&self, timesteps: &[f32]) -> Result<Tensor> {
        let dim = self.config.freq_dim;
        let half = dim / 2;
        let n = timesteps.len();

        let mut data = vec![0.0f32; n * dim];
        for (t_idx, &pos) in timesteps.iter().enumerate() {
            let pos = pos as f64;
            for i in 0..half {
                let freq = 10000.0f64.powf(-(i as f64) / half as f64);
                let angle = pos * freq;
                data[t_idx * dim + i] = angle.cos() as f32;           // cos first half
                data[t_idx * dim + half + i] = angle.sin() as f32;    // sin second half
            }
        }

        Tensor::from_vec(data, Shape::from_dims(&[n, dim]), self.device.clone())
    }

    /// Apply complex-number 3-axis RoPE to Q or K tensor.
    ///
    /// Input: [B, seq, num_heads, head_dim] in BF16
    /// grid_sizes: (F, H, W) — the latent grid after patchifying
    ///
    /// Wan's RoPE operates on pairs of values as complex numbers:
    ///   (a + bi)(cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
    ///
    /// The head_dim is split into 3 axes: [44, 42, 42].
    /// For each axis, positions [0..axis_size) index into the precomputed freq table.
    /// Axis 0 = frame positions, Axis 1 = height, Axis 2 = width.
    fn apply_rope(
        &self,
        x: &Tensor,
        grid_sizes: (usize, usize, usize),
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];
        let nh = dims[2];
        let hd = dims[3];
        let (f, h, w) = grid_sizes;
        let img_seq = f * h * w;

        // Convert to F32 for precision
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_data = x_f32.to_vec1::<f32>()?;

        let total = b * seq * nh * hd;
        let mut out_data = vec![0.0f32; total];
        // Copy input (padding tokens stay as-is)
        out_data.copy_from_slice(&x_data);

        // Only apply RoPE to the first img_seq tokens (rest are padding)
        let axes = &self.rope_axes;
        let axes_half: Vec<usize> = axes.iter().map(|a| a / 2).collect();

        for bi in 0..b {
            for si in 0..img_seq.min(seq) {
                // Decompose token index into (frame, height, width)
                let fi = si / (h * w);
                let hi = (si % (h * w)) / w;
                let wi = si % w;
                let positions = [fi, hi, wi];

                for ni in 0..nh {
                    let base = bi * seq * nh * hd + si * nh * hd + ni * hd;
                    let mut dim_offset = 0usize;

                    for (axis_i, &ax_half) in axes_half.iter().enumerate() {
                        let pos = positions[axis_i];
                        let cos_table = &self.rope_freqs_cos[axis_i];
                        let sin_table = &self.rope_freqs_sin[axis_i];

                        for i in 0..ax_half {
                            let cos_val = cos_table[pos * ax_half + i];
                            let sin_val = sin_table[pos * ax_half + i];
                            let re_idx = base + dim_offset + 2 * i;
                            let im_idx = re_idx + 1;
                            let a = out_data[re_idx];
                            let b_val = out_data[im_idx];
                            // Complex multiply: (a+bi)(cos+i*sin)
                            out_data[re_idx] = a * cos_val - b_val * sin_val;
                            out_data[im_idx] = a * sin_val + b_val * cos_val;
                        }
                        dim_offset += axes[axis_i];
                    }
                }
            }
        }

        let out = Tensor::from_vec(
            out_data,
            Shape::from_dims(&dims),
            self.device.clone(),
        )?;
        out.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Forward pass through the Wan2.2 DiT.
    ///
    /// Arguments:
    /// - `x`: list of video latents, each [C_in=16, F, H, W] (unpacked)
    /// - `timestep`: scalar timestep (integer, 0-1000)
    /// - `context`: text embeddings [1, L_text, 4096] BF16
    /// - `seq_len`: padded sequence length for the latent
    ///
    /// Returns: [C_out=16, F, H, W] denoised prediction (single sample).
    pub fn forward(
        &mut self,
        x: &Tensor,            // [C, F, H, W]
        timestep: f32,         // scalar timestep
        context: &Tensor,      // [1, L, 4096]
        seq_len: usize,
    ) -> Result<Tensor> {
        let cfg = self.config.clone();

        // ── Patch embedding: Conv3d(16, 5120, k=(1,2,2), s=(1,2,2)) ──
        // We implement this as a manual reshape + linear since flame-core
        // doesn't have Conv3d. Input [C=16, F, H, W], output [1, F*(H/2)*(W/2), 5120].
        let x_dims = x.shape().dims().to_vec();
        let (c_in, f_in, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let h_out = h_in / cfg.patch_size[1];
        let w_out = w_in / cfg.patch_size[2];
        let f_out = f_in / cfg.patch_size[0];
        let n_patches = f_out * h_out * w_out;
        let grid_sizes = (f_out, h_out, w_out);

        // Patchify: reshape [C, F, H, W] → [F, H/2, W/2, C*1*2*2=64] → [n_patches, 64]
        let patch_dim = c_in * cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]; // 16*1*2*2=64
        let patched = self.patchify(x, f_in, h_in, w_in)?;
        // patched: [n_patches, patch_dim=64]

        // Conv3d as linear: [1, n_patches, 64] → [1, n_patches, 5120]
        let pe_w = self.shared.get("patch_embedding.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing patch_embedding.weight".into()))?;
        let pe_b = self.shared.get("patch_embedding.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing patch_embedding.bias".into()))?;
        // pe_w is [5120, 16, 1, 2, 2] — flatten to [5120, 64] for linear
        let pe_w_flat = pe_w.reshape(&[cfg.dim, patch_dim])?;
        let patched_3d = patched.unsqueeze(0)?; // [1, n_patches, 64]
        let mut img = Self::linear_bias(&patched_3d, &pe_w_flat, pe_b)?;

        // Pad to seq_len
        assert!(n_patches <= seq_len, "n_patches {} > seq_len {}", n_patches, seq_len);
        if n_patches < seq_len {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[1, seq_len - n_patches, cfg.dim]),
                DType::BF16,
                self.device.clone(),
            )?;
            img = Tensor::cat(&[&img, &pad], 1)?;
        }
        // img: [1, seq_len, 5120]

        // ── Time embedding ──
        // Per-token: expand timestep to [seq_len], sinusoidal embed, MLP
        let t_vals = vec![timestep; seq_len];
        let sin_emb = self.sinusoidal_embedding(&t_vals)?; // [seq_len, 256] F32
        let sin_emb_bf16 = sin_emb.to_dtype(DType::BF16)?.unsqueeze(0)?; // [1, seq_len, 256]

        let te_w0 = self.shared.get("time_embedding.0.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_embedding.0.weight".into()))?;
        let te_b0 = self.shared.get("time_embedding.0.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_embedding.0.bias".into()))?;
        let te_w2 = self.shared.get("time_embedding.2.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_embedding.2.weight".into()))?;
        let te_b2 = self.shared.get("time_embedding.2.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_embedding.2.bias".into()))?;

        // time_embedding: Linear(256, 5120) → SiLU → Linear(5120, 5120)
        let e = Self::linear_bias(&sin_emb_bf16, te_w0, te_b0)?;
        let e = e.silu()?;
        let e = Self::linear_bias(&e, te_w2, te_b2)?;
        // e: [1, seq_len, 5120]

        // time_projection: SiLU → Linear(5120, 30720) → reshape to [1, seq_len, 6, 5120]
        let tp_w = self.shared.get("time_projection.1.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_projection.1.weight".into()))?;
        let tp_b = self.shared.get("time_projection.1.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing time_projection.1.bias".into()))?;
        let e_silu = e.silu()?;
        let e0_flat = Self::linear_bias(&e_silu, tp_w, tp_b)?; // [1, seq_len, 30720]
        let e0 = e0_flat.reshape(&[1, seq_len, 6, cfg.dim])?; // [1, seq_len, 6, 5120]
        // e0 must stay in FP32 for modulation (per Python: assert e0.dtype == torch.float32)
        let e0 = e0.to_dtype(DType::F32)?;

        // ── Text embedding ──
        // context: [1, L, 4096] → text_embedding MLP → [1, text_len, 5120]
        let txt_w0 = self.shared.get("text_embedding.0.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing text_embedding.0.weight".into()))?;
        let txt_b0 = self.shared.get("text_embedding.0.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing text_embedding.0.bias".into()))?;
        let txt_w2 = self.shared.get("text_embedding.2.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing text_embedding.2.weight".into()))?;
        let txt_b2 = self.shared.get("text_embedding.2.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing text_embedding.2.bias".into()))?;

        // Pad context to text_len
        let ctx_len = context.shape().dims()[1];
        let ctx_padded = if ctx_len < cfg.text_len {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[1, cfg.text_len - ctx_len, cfg.text_dim]),
                context.dtype(),
                self.device.clone(),
            )?;
            Tensor::cat(&[context, &pad], 1)?
        } else {
            context.narrow(1, 0, cfg.text_len)?
        };

        let txt = Self::linear_bias(&ctx_padded, txt_w0, txt_b0)?;
        let txt = txt.gelu()?;
        let txt = Self::linear_bias(&txt, txt_w2, txt_b2)?;
        // txt: [1, text_len, 5120]

        // ── Transformer blocks ──
        let total_blocks = cfg.num_layers;
        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        let seq_lens_val = n_patches;

        for i in 0..total_blocks {
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            img = self.block_forward(
                &img, &e0, &e, &txt, seq_lens_val, grid_sizes, &raw, i,
            )?;

            if i % 10 == 0 || i == total_blocks - 1 {
                log::info!("[Wan22] Block {}/{}", i + 1, total_blocks);
            }
        }

        // ── Head: norm + modulate + linear ──
        let head_mod = self.shared.get("head.modulation")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing head.modulation".into()))?;
        let head_w = self.shared.get("head.head.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing head.head.weight".into()))?;
        let head_b = self.shared.get("head.head.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing head.head.bias".into()))?;

        // head modulation: (head.modulation + e.unsqueeze(2)).chunk(2)
        // head.modulation: [1, 2, 5120], e: [1, seq_len, 5120]
        // → broadcast add: [1, seq_len, 2, 5120] → chunk → shift, scale
        let head_mod_f32 = head_mod.to_dtype(DType::F32)?; // [1, 2, 5120]
        let e_f32 = e.to_dtype(DType::F32)?; // [1, seq_len, 5120]
        // e.unsqueeze(2): [1, seq_len, 1, 5120]
        // head_mod: [1, 1, 2, 5120] (broadcast over seq)
        // sum: [1, seq_len, 2, 5120]
        // chunk(2, dim=2) → two [1, seq_len, 1, 5120] → squeeze → [1, seq_len, 5120]
        let e_expanded = e_f32.unsqueeze(2)?; // [1, seq_len, 1, 5120]
        let head_mod_expanded = head_mod_f32.unsqueeze(0)?; // [1, 1, 2, 5120] — wait, it's already [1, 2, 5120]

        // Actually: head.modulation is [1, 2, dim]. We need to add it to e.unsqueeze(2) which is [1, seq, 1, dim].
        // The Python does: e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        // self.modulation: [1, 2, dim] → unsqueeze(0): [1, 1, 2, dim]
        // e: [1, seq, dim] → unsqueeze(2): [1, seq, 1, dim]
        // broadcast sum: [1, seq, 2, dim]
        // chunk(2, dim=2): two [1, seq, 1, dim]
        //
        // For the head, e is the time embedding (before projection), shape [1, seq_len, 5120].
        // We compute this manually in FP32.
        let head_mod_4d = head_mod_f32.unsqueeze(0)?; // [1, 1, 2, 5120] — but it's [1, 2, 5120], so unsqueeze at 0 gives [1, 1, 2, 5120]
        // Wait — head_mod is already [1, 2, 5120], the batch dim is already there.
        // unsqueeze(0) gives [1, 1, 2, 5120]. Then broadcast with [1, seq, 1, 5120]
        // gives [1, seq, 2, 5120].

        // Since we can't do fancy broadcast in flame-core, compute the two modulation chunks
        // as shift (index 0) and scale (index 1) from head.modulation, each broadcast-added to e.
        let head_mod_data = head_mod_f32.to_vec1::<f32>()?; // [1 * 2 * 5120]
        let dim = cfg.dim;
        let head_shift: Vec<f32> = head_mod_data[..dim].to_vec();     // modulation[:, 0, :]
        let head_scale: Vec<f32> = head_mod_data[dim..2*dim].to_vec(); // modulation[:, 1, :]

        // x = head(norm(x) * (1 + scale) + shift)  where shift/scale each get e added
        // Doing this token-by-token in a flat compute
        let img_normed_bf16 = Self::layer_norm_no_affine(&img, cfg.eps)?;
        let normed_data = img_normed_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let e_data = e_f32.to_vec1::<f32>()?;

        let mut head_input = vec![0.0f32; seq_len * dim];
        for s in 0..seq_len {
            for d in 0..dim {
                let e_val = e_data[s * dim + d];
                let shift = head_shift[d] + e_val;
                let scale = head_scale[d] + e_val;
                head_input[s * dim + d] = normed_data[s * dim + d] * (1.0 + scale) + shift;
            }
        }

        let head_in = Tensor::from_vec(
            head_input,
            Shape::from_dims(&[1, seq_len, dim]),
            self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        let out = Self::linear_bias(&head_in, head_w, head_b)?; // [1, seq_len, 64]

        // ── Unpatchify ──
        // Take first n_patches tokens, reshape to [F, H, W, p0, p1, p2, C_out]
        // then einsum 'fhwpqrc->cfphqwr' → [C, F*p0, H*p1, W*p2]
        let out_trimmed = out.narrow(1, 0, n_patches)?; // [1, n_patches, 64]
        let unpacked = self.unpatchify(&out_trimmed, grid_sizes)?;
        Ok(unpacked)
    }

    /// Patchify: [C, F, H, W] → [F*(H/2)*(W/2), C*1*2*2]
    fn patchify(&self, x: &Tensor, f: usize, h: usize, w: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let c = x.shape().dims()[0]; // read from tensor, not config (I2V has 36, T2V has 16)
        let (pf, ph, pw) = (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]);
        let fo = f / pf;
        let ho = h / ph;
        let wo = w / pw;
        let patch_dim = c * pf * ph * pw;
        let n_patches = fo * ho * wo;

        let x_data = x.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut out = vec![0.0f32; n_patches * patch_dim];

        for fi in 0..fo {
            for hi in 0..ho {
                for wi in 0..wo {
                    let patch_idx = fi * ho * wo + hi * wo + wi;
                    for pfi in 0..pf {
                        for phi in 0..ph {
                            for pwi in 0..pw {
                                for ci in 0..c {
                                    let src_f = fi * pf + pfi;
                                    let src_h = hi * ph + phi;
                                    let src_w = wi * pw + pwi;
                                    let src_idx = ci * f * h * w + src_f * h * w + src_h * w + src_w;
                                    let dst_ch = ci * pf * ph * pw + pfi * ph * pw + phi * pw + pwi;
                                    out[patch_idx * patch_dim + dst_ch] = x_data[src_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_f32_to_bf16(out, Shape::from_dims(&[n_patches, patch_dim]), self.device.clone())
    }

    /// Unpatchify: [1, n_patches, C_out*prod(patch_size)] → [C_out, F, H, W]
    fn unpatchify(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let cfg = &self.config;
        let c = cfg.out_channels;
        let (pf, ph, pw) = (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]);
        let (fo, ho, wo) = grid;
        let patch_dim = c * pf * ph * pw;

        let x_data = x.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let out_f = fo * pf;
        let out_h = ho * ph;
        let out_w = wo * pw;
        let mut out = vec![0.0f32; c * out_f * out_h * out_w];

        // Python: u = u[:prod(v)].view(*v, *patch_size, c)
        //         u = einsum('fhwpqrc->cfphqwr', u)
        for fi in 0..fo {
            for hi in 0..ho {
                for wi in 0..wo {
                    let patch_idx = fi * ho * wo + hi * wo + wi;
                    for pfi in 0..pf {
                        for phi in 0..ph {
                            for pwi in 0..pw {
                                for ci in 0..c {
                                    let src_ch = pfi * ph * pw * c + phi * pw * c + pwi * c + ci;
                                    let val = x_data[patch_idx * patch_dim + src_ch];
                                    let dst_f = fi * pf + pfi;
                                    let dst_h = hi * ph + phi;
                                    let dst_w = wi * pw + pwi;
                                    let dst_idx = ci * out_f * out_h * out_w + dst_f * out_h * out_w + dst_h * out_w + dst_w;
                                    out[dst_idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_f32_to_bf16(out, Shape::from_dims(&[c, out_f, out_h, out_w]), self.device.clone())
    }

    /// I2V forward pass. Input `x` is `[36, F, H, W]` (noise + y already concatenated).
    /// Output is still `[16, F, H, W]` (the head outputs out_channels=16).
    /// Internally identical to `forward` — the patchify and patch_embedding handle
    /// the larger input channels automatically.
    pub fn forward_i2v(
        &mut self,
        x: &Tensor,
        timestep: f32,
        context: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.forward(x, timestep, context, seq_len)
    }

    // -----------------------------------------------------------------------
    // Block forward
    // -----------------------------------------------------------------------

    fn block_forward(
        &self,
        x: &Tensor,            // [1, seq_len, dim]
        e0: &Tensor,           // [1, seq_len, 6, dim] FP32 — time projection
        e: &Tensor,            // [1, seq_len, dim] — time embedding (for head, kept around)
        context: &Tensor,      // [1, text_len, dim]
        seq_len_actual: usize, // actual (non-padded) sequence length
        grid_sizes: (usize, usize, usize),
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let nh = cfg.num_heads;
        let hd = cfg.head_dim;
        let prefix = format!("blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        // ── Block modulation ──
        // self.modulation: [1, 6, dim] + e0 → chunk(6)
        // e0: [1, seq_len, 6, dim] FP32
        let block_mod = w("modulation")?;
        let block_mod_f32 = block_mod.to_dtype(DType::F32)?; // [1, 6, dim]

        // Compute modulated e: (block_mod.unsqueeze(0) + e0).chunk(6, dim=2)
        // block_mod: [1, 6, dim] → unsqueeze(0) → [1, 1, 6, dim]
        // e0: [1, seq_len, 6, dim]
        // broadcast sum: [1, seq_len, 6, dim]
        // We need to add block_mod to each token's 6 modulation vectors.
        // Then split into 6 separate [1, seq_len, dim] tensors.

        // Get block_mod data [6 * dim]
        let bm_data = block_mod_f32.to_vec1::<f32>()?;
        let e0_data = e0.to_vec1::<f32>()?;
        let sl = x.shape().dims()[1]; // padded seq_len

        // We need 6 modulation tensors, each [1, sl, dim]
        let mut mods = vec![vec![0.0f32; sl * dim]; 6];
        for s in 0..sl {
            for m in 0..6 {
                for d in 0..dim {
                    mods[m][s * dim + d] = bm_data[m * dim + d] + e0_data[s * 6 * dim + m * dim + d];
                }
            }
        }

        // Create mod tensors on GPU
        let make_mod = |idx: usize| -> Result<Tensor> {
            Tensor::from_vec(
                mods[idx].clone(),
                Shape::from_dims(&[1, sl, dim]),
                self.device.clone(),
            )
        };
        // Modulation order: shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn
        // But Python calls them e[0]..e[5], used as:
        //   self_attn: norm(x) * (1 + e[1]) + e[0], gate e[2]
        //   ffn:       norm(x) * (1 + e[4]) + e[3], gate e[5]

        // ── Self-attention ──
        // mod_input = norm1(x) * (1 + e[1]) + e[0]   (in FP32 for modulation)
        // LayerNorm needs BF16 input, then convert to F32 for modulation math.
        let x_normed_bf16 = Self::layer_norm_no_affine(x, cfg.eps)?;
        let x_normed_f32 = x_normed_bf16.to_dtype(DType::F32)?;
        let normed_data = x_normed_f32.to_vec1::<f32>()?;
        let x_f32 = x.to_dtype(DType::F32)?;

        let mut sa_input_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                sa_input_data[idx] = normed_data[idx] * (1.0 + mods[1][idx]) + mods[0][idx];
            }
        }
        let sa_input = Tensor::from_vec(
            sa_input_data,
            Shape::from_dims(&[1, sl, dim]),
            self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Q, K, V projections
        let q = Self::linear_bias(&sa_input, w("self_attn.q.weight")?, w("self_attn.q.bias")?)?;
        let k = Self::linear_bias(&sa_input, w("self_attn.k.weight")?, w("self_attn.k.bias")?)?;
        let v = Self::linear_bias(&sa_input, w("self_attn.v.weight")?, w("self_attn.v.bias")?)?;

        // QK RMSNorm (full-dim, not per-head)
        let q = Self::rms_norm(&q, w("self_attn.norm_q.weight")?, cfg.eps)?;
        let k = Self::rms_norm(&k, w("self_attn.norm_k.weight")?, cfg.eps)?;

        // Reshape to [1, seq, nh, hd]
        let q = q.reshape(&[1, sl, nh, hd])?;
        let k = k.reshape(&[1, sl, nh, hd])?;
        let v = v.reshape(&[1, sl, nh, hd])?;

        // Apply 3-axis RoPE
        let q = self.apply_rope(&q, grid_sizes)?;
        let k = self.apply_rope(&k, grid_sizes)?;

        // SDPA: transpose to [B, H, S, D]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Reshape back: [1, nh, sl, hd] → [1, sl, nh*hd]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[1, sl, nh * hd])?;

        // Output projection
        let sa_out = Self::linear_bias(&attn_out, w("self_attn.o.weight")?, w("self_attn.o.bias")?)?;

        // Gated residual (gate in FP32)
        let sa_out_f32 = sa_out.to_dtype(DType::F32)?;
        let sa_out_data = sa_out_f32.to_vec1::<f32>()?;
        let x_data = x_f32.to_vec1::<f32>()?;
        let mut x_new_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                x_new_data[idx] = x_data[idx] + sa_out_data[idx] * mods[2][idx];
            }
        }
        let x_after_sa = Tensor::from_vec(
            x_new_data,
            Shape::from_dims(&[1, sl, dim]),
            self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        // ── Cross-attention ──
        // norm3 (LayerNorm with affine for cross_attn_norm)
        let ca_input = if let (Ok(n3w), Ok(n3b)) = (w("norm3.weight"), w("norm3.bias")) {
            Self::layer_norm_affine(&x_after_sa, n3w, n3b, cfg.eps)?
        } else {
            x_after_sa.clone()
        };

        let ca_q = Self::linear_bias(&ca_input, w("cross_attn.q.weight")?, w("cross_attn.q.bias")?)?;
        let ca_k = Self::linear_bias(context, w("cross_attn.k.weight")?, w("cross_attn.k.bias")?)?;
        let ca_v = Self::linear_bias(context, w("cross_attn.v.weight")?, w("cross_attn.v.bias")?)?;

        // QK RMSNorm
        let ca_q = Self::rms_norm(&ca_q, w("cross_attn.norm_q.weight")?, cfg.eps)?;
        let ca_k = Self::rms_norm(&ca_k, w("cross_attn.norm_k.weight")?, cfg.eps)?;

        let ca_q = ca_q.reshape(&[1, sl, nh, hd])?.permute(&[0, 2, 1, 3])?;
        let ca_k_len = context.shape().dims()[1];
        let ca_k = ca_k.reshape(&[1, ca_k_len, nh, hd])?.permute(&[0, 2, 1, 3])?;
        let ca_v = ca_v.reshape(&[1, ca_k_len, nh, hd])?.permute(&[0, 2, 1, 3])?;

        let ca_out = flame_core::attention::sdpa(&ca_q, &ca_k, &ca_v, None)?;
        let ca_out = ca_out.permute(&[0, 2, 1, 3])?.reshape(&[1, sl, nh * hd])?;
        let ca_out = Self::linear_bias(&ca_out, w("cross_attn.o.weight")?, w("cross_attn.o.bias")?)?;

        // Residual (no gate for cross-attn)
        let x_after_ca = x_after_sa.add(&ca_out)?;

        // ── FFN ──
        // mod_input = norm2(x) * (1 + e[4]) + e[3], gated by e[5]
        let x_ca_normed_bf16 = Self::layer_norm_no_affine(&x_after_ca, cfg.eps)?;
        let x_ca_f32 = x_after_ca.to_dtype(DType::F32)?;
        let normed_data = x_ca_normed_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        let mut ffn_input_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                ffn_input_data[idx] = normed_data[idx] * (1.0 + mods[4][idx]) + mods[3][idx];
            }
        }
        let ffn_input = Tensor::from_vec(
            ffn_input_data,
            Shape::from_dims(&[1, sl, dim]),
            self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        let ffn_h = Self::linear_bias(&ffn_input, w("ffn.0.weight")?, w("ffn.0.bias")?)?;
        let ffn_h = ffn_h.gelu()?;
        let ffn_out = Self::linear_bias(&ffn_h, w("ffn.2.weight")?, w("ffn.2.bias")?)?;

        // Gated residual (FP32)
        let ffn_out_f32 = ffn_out.to_dtype(DType::F32)?;
        let ffn_data = ffn_out_f32.to_vec1::<f32>()?;
        let x_ca_data = x_ca_f32.to_vec1::<f32>()?;
        let mut x_final_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                x_final_data[idx] = x_ca_data[idx] + ffn_data[idx] * mods[5][idx];
            }
        }
        let x_final = Tensor::from_vec(
            x_final_data,
            Shape::from_dims(&[1, sl, dim]),
            self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        let _ = Self::linear_nobias; // silence unused
        Ok(x_final)
    }

    // -----------------------------------------------------------------------
    // Public accessors for WanVaceDit
    // -----------------------------------------------------------------------

    pub fn patchify_public(&self, x: &Tensor, f: usize, h: usize, w: usize) -> Result<Tensor> {
        self.patchify(x, f, h, w)
    }

    pub fn unpatchify_public(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        self.unpatchify(x, grid)
    }

    pub fn shared_weight(&self, key: &str) -> Result<&Tensor> {
        self.shared.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing shared weight: {key}"))
        })
    }

    pub fn linear_bias_pub(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        Self::linear_bias(x, weight, bias)
    }

    /// Compute time + text embeddings (shared between base and VACE paths).
    /// Returns (e, e0, txt) where:
    /// - e: [1, seq_len, dim] BF16 — time embedding
    /// - e0: [1, seq_len, 6, dim] F32 — time projection
    /// - txt: [1, text_len, dim] BF16 — text embedding
    pub fn compute_embeddings(
        &self,
        timestep: f32,
        context: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;

        // Time embedding
        let t_vals = vec![timestep; seq_len];
        let sin_emb = self.sinusoidal_embedding(&t_vals)?;
        let sin_emb_bf16 = sin_emb.to_dtype(DType::BF16)?.unsqueeze(0)?;

        let te_w0 = self.shared.get("time_embedding.0.weight").unwrap();
        let te_b0 = self.shared.get("time_embedding.0.bias").unwrap();
        let te_w2 = self.shared.get("time_embedding.2.weight").unwrap();
        let te_b2 = self.shared.get("time_embedding.2.bias").unwrap();
        let e = Self::linear_bias(&sin_emb_bf16, te_w0, te_b0)?;
        let e = e.silu()?;
        let e = Self::linear_bias(&e, te_w2, te_b2)?;

        let tp_w = self.shared.get("time_projection.1.weight").unwrap();
        let tp_b = self.shared.get("time_projection.1.bias").unwrap();
        let e_silu = e.silu()?;
        let e0_flat = Self::linear_bias(&e_silu, tp_w, tp_b)?;
        let e0 = e0_flat.reshape(&[1, seq_len, 6, cfg.dim])?.to_dtype(DType::F32)?;

        // Text embedding
        let txt_w0 = self.shared.get("text_embedding.0.weight").unwrap();
        let txt_b0 = self.shared.get("text_embedding.0.bias").unwrap();
        let txt_w2 = self.shared.get("text_embedding.2.weight").unwrap();
        let txt_b2 = self.shared.get("text_embedding.2.bias").unwrap();
        let ctx_len = context.shape().dims()[1];
        let ctx_padded = if ctx_len < cfg.text_len {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[1, cfg.text_len - ctx_len, cfg.text_dim]),
                context.dtype(), self.device.clone(),
            )?;
            Tensor::cat(&[context, &pad], 1)?
        } else {
            context.narrow(1, 0, cfg.text_len)?
        };
        let txt = Self::linear_bias(&ctx_padded, txt_w0, txt_b0)?;
        let txt = txt.gelu()?;
        let txt = Self::linear_bias(&txt, txt_w2, txt_b2)?;

        Ok((e, e0, txt))
    }

    /// Run a single block forward with configurable prefix (for VACE reuse).
    pub fn block_forward_pub(
        &self,
        x: &Tensor,
        e0: &Tensor,
        e: &Tensor,
        context: &Tensor,
        seq_len_actual: usize,
        grid_sizes: (usize, usize, usize),
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
        prefix_base: &str,  // "blocks" or "vace_blocks"
    ) -> Result<Tensor> {
        // Delegate to block_forward but with custom prefix
        self.block_forward_with_prefix(x, e0, e, context, seq_len_actual, grid_sizes, weights, block_idx, prefix_base)
    }

    /// FlameSwap accessors for VACE
    pub fn swap_prefetch(&mut self, idx: usize) -> Result<()> {
        self.swap.prefetch(idx)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))
    }

    pub fn swap_await(&mut self, idx: usize) -> Result<HashMap<String, Tensor>> {
        self.swap.await_block(idx)
            .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))
    }

    /// Head forward (norm_out + proj_out)
    pub fn head_forward(&self, img: &Tensor, e: &Tensor, seq_len: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let dim = cfg.dim;

        let head_mod = self.shared.get("head.modulation").unwrap();
        let head_w = self.shared.get("head.head.weight").unwrap();
        let head_b = self.shared.get("head.head.bias").unwrap();

        let head_mod_f32 = head_mod.to_dtype(DType::F32)?;
        let e_f32 = e.to_dtype(DType::F32)?;
        let bm_data = head_mod_f32.to_vec1::<f32>()?;
        let head_shift: Vec<f32> = bm_data[..dim].to_vec();
        let head_scale: Vec<f32> = bm_data[dim..2*dim].to_vec();

        let img_normed_bf16 = Self::layer_norm_no_affine(img, cfg.eps)?;
        let normed_data = img_normed_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let e_data = e_f32.to_vec1::<f32>()?;

        let mut head_input = vec![0.0f32; seq_len * dim];
        for s in 0..seq_len {
            for d in 0..dim {
                let e_val = e_data[s * dim + d];
                let shift = head_shift[d] + e_val;
                let scale = head_scale[d] + e_val;
                head_input[s * dim + d] = normed_data[s * dim + d] * (1.0 + scale) + shift;
            }
        }

        let head_in = Tensor::from_vec(
            head_input, Shape::from_dims(&[1, seq_len, dim]), self.device.clone(),
        )?.to_dtype(DType::BF16)?;
        Self::linear_bias(&head_in, head_w, head_b)
    }

    /// Block forward with configurable key prefix
    fn block_forward_with_prefix(
        &self,
        x: &Tensor,
        e0: &Tensor,
        _e: &Tensor,
        context: &Tensor,
        _seq_len_actual: usize,
        grid_sizes: (usize, usize, usize),
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
        prefix_base: &str,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let nh = cfg.num_heads;
        let hd = cfg.head_dim;
        let prefix = format!("{prefix_base}.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let sl = x.shape().dims()[1];

        // Block modulation
        let block_mod = w("modulation")?;
        let block_mod_f32 = block_mod.to_dtype(DType::F32)?;
        let bm_data = block_mod_f32.to_vec1::<f32>()?;
        let e0_data = e0.to_vec1::<f32>()?;

        let mut mods = vec![vec![0.0f32; sl * dim]; 6];
        for s in 0..sl {
            for m in 0..6 {
                for d in 0..dim {
                    mods[m][s * dim + d] = bm_data[m * dim + d] + e0_data[s * 6 * dim + m * dim + d];
                }
            }
        }

        // Self-attention
        let x_normed_bf16 = Self::layer_norm_no_affine(x, cfg.eps)?;
        let x_normed_f32 = x_normed_bf16.to_dtype(DType::F32)?;
        let normed_data = x_normed_f32.to_vec1::<f32>()?;
        let x_f32 = x.to_dtype(DType::F32)?;

        let mut sa_input_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                sa_input_data[idx] = normed_data[idx] * (1.0 + mods[1][idx]) + mods[0][idx];
            }
        }
        let sa_input = Tensor::from_vec(
            sa_input_data, Shape::from_dims(&[1, sl, dim]), self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        let q = Self::linear_bias(&sa_input, w("self_attn.q.weight")?, w("self_attn.q.bias")?)?;
        let k = Self::linear_bias(&sa_input, w("self_attn.k.weight")?, w("self_attn.k.bias")?)?;
        let v = Self::linear_bias(&sa_input, w("self_attn.v.weight")?, w("self_attn.v.bias")?)?;
        let q = Self::rms_norm(&q, w("self_attn.norm_q.weight")?, cfg.eps)?;
        let k = Self::rms_norm(&k, w("self_attn.norm_k.weight")?, cfg.eps)?;
        let q = q.reshape(&[1, sl, nh, hd])?;
        let k = k.reshape(&[1, sl, nh, hd])?;
        let v = v.reshape(&[1, sl, nh, hd])?;
        let q = self.apply_rope(&q, grid_sizes)?;
        let k = self.apply_rope(&k, grid_sizes)?;
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[1, sl, nh * hd])?;
        let sa_out = Self::linear_bias(&attn_out, w("self_attn.o.weight")?, w("self_attn.o.bias")?)?;

        let sa_out_f32 = sa_out.to_dtype(DType::F32)?;
        let sa_out_data = sa_out_f32.to_vec1::<f32>()?;
        let x_data = x_f32.to_vec1::<f32>()?;
        let mut x_new_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                x_new_data[idx] = x_data[idx] + sa_out_data[idx] * mods[2][idx];
            }
        }
        let x_after_sa = Tensor::from_vec(
            x_new_data, Shape::from_dims(&[1, sl, dim]), self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Cross-attention
        let ca_input = if let (Ok(n3w), Ok(n3b)) = (w("norm3.weight"), w("norm3.bias")) {
            Self::layer_norm_affine(&x_after_sa, n3w, n3b, cfg.eps)?
        } else {
            x_after_sa.clone()
        };
        let ca_q = Self::linear_bias(&ca_input, w("cross_attn.q.weight")?, w("cross_attn.q.bias")?)?;
        let ca_k = Self::linear_bias(context, w("cross_attn.k.weight")?, w("cross_attn.k.bias")?)?;
        let ca_v = Self::linear_bias(context, w("cross_attn.v.weight")?, w("cross_attn.v.bias")?)?;
        let ca_q = Self::rms_norm(&ca_q, w("cross_attn.norm_q.weight")?, cfg.eps)?;
        let ca_k = Self::rms_norm(&ca_k, w("cross_attn.norm_k.weight")?, cfg.eps)?;
        let ca_k_len = context.shape().dims()[1];
        let ca_q = ca_q.reshape(&[1, sl, nh, hd])?.permute(&[0, 2, 1, 3])?;
        let ca_k = ca_k.reshape(&[1, ca_k_len, nh, hd])?.permute(&[0, 2, 1, 3])?;
        let ca_v = ca_v.reshape(&[1, ca_k_len, nh, hd])?.permute(&[0, 2, 1, 3])?;
        let ca_out = flame_core::attention::sdpa(&ca_q, &ca_k, &ca_v, None)?;
        let ca_out = ca_out.permute(&[0, 2, 1, 3])?.reshape(&[1, sl, nh * hd])?;
        let ca_out = Self::linear_bias(&ca_out, w("cross_attn.o.weight")?, w("cross_attn.o.bias")?)?;
        let x_after_ca = x_after_sa.add(&ca_out)?;

        // FFN
        let x_ca_normed_bf16 = Self::layer_norm_no_affine(&x_after_ca, cfg.eps)?;
        let x_ca_f32 = x_after_ca.to_dtype(DType::F32)?;
        let normed_data = x_ca_normed_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut ffn_input_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                ffn_input_data[idx] = normed_data[idx] * (1.0 + mods[4][idx]) + mods[3][idx];
            }
        }
        let ffn_input = Tensor::from_vec(
            ffn_input_data, Shape::from_dims(&[1, sl, dim]), self.device.clone(),
        )?.to_dtype(DType::BF16)?;
        let ffn_h = Self::linear_bias(&ffn_input, w("ffn.0.weight")?, w("ffn.0.bias")?)?;
        let ffn_h = ffn_h.gelu()?;
        let ffn_out = Self::linear_bias(&ffn_h, w("ffn.2.weight")?, w("ffn.2.bias")?)?;

        let ffn_out_f32 = ffn_out.to_dtype(DType::F32)?;
        let ffn_data = ffn_out_f32.to_vec1::<f32>()?;
        let x_ca_data = x_ca_f32.to_vec1::<f32>()?;
        let mut x_final_data = vec![0.0f32; sl * dim];
        for s in 0..sl {
            for d in 0..dim {
                let idx = s * dim + d;
                x_final_data[idx] = x_ca_data[idx] + ffn_data[idx] * mods[5][idx];
            }
        }
        Tensor::from_vec(
            x_final_data, Shape::from_dims(&[1, sl, dim]), self.device.clone(),
        )?.to_dtype(DType::BF16)
    }
}
