//! Lens — Microsoft Lens text-to-image MM-DiT (Stage A2 + A3 skeleton).
//!
//! Source: `/home/alex/vendor-refs/Lens/lens/transformer.py`
//! HF: microsoft/Lens, microsoft/Lens-Base, microsoft/Lens-Turbo (identical arch)
//!
//! This module currently provides:
//! - `LensDiTConfig` matching `DEFAULT_TRANSFORMER_CONFIG` in `lens/pipeline.py`
//! - `LensDiTBlock` struct skeleton (weight tensors only, no forward)
//! - `LensTransformer2DModel` struct skeleton (weights + RoPE table builder)
//! - `LensEmbedRope`: host-side 3-axis complex-polar RoPE table (Stage A3),
//!   converted to `(cos, sin)` BF16 GPU tables for `rope_fused_bf16`.
//!
//! `forward` and `load_weights` are stubs — Stage A4 wires the block forward,
//! Stage A5 wires weight loading.

use flame_core::linear::Linear;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaDevice;

// ---------------------------------------------------------------------------
// AUTOGRAD NOTE (Skeptic F1)
// ---------------------------------------------------------------------------
// `Linear::new_zeroed` records params with `requires_grad=true`. For inference,
// the binary that drives this module is expected to call
// `flame_core::autograd::AutogradContext::set_enabled(false)` in `main()` —
// see e.g. `inference-flame/src/bin/hidream_o1_infer.rs:172`,
// `zimage_infer.rs:152`, `asymflux2_klein9b_infer.rs:713`. Without it the
// 48-block DiT × 20-step denoise loop will retain a full autograd tape across
// steps and OOM (same trap as HiDream-O1 multi-step OOM, 2026-05-09). The
// forward path below does NOT itself disable autograd; that is the bin's
// responsibility. This note is the in-code reminder requested in
// SKEPTIC_FINDINGS_2026-05-22.md F1.

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture constants for the Lens DiT. Defaults mirror the
/// `DEFAULT_TRANSFORMER_CONFIG` dict in `lens/pipeline.py`.
#[derive(Debug, Clone)]
pub struct LensDiTConfig {
    /// `patch_size` — 2x2 patch over the latent (post-VAE-decode).
    pub patch_size: usize,
    /// `in_channels` — 128 (FLUX.2 VAE latent ch × 4 from 2x2 patchify).
    pub in_channels: usize,
    /// `out_channels` — 32 (final un-patchify gives 32 latent ch).
    pub out_channels: usize,
    /// `num_layers` — 48 dual-stream blocks.
    pub num_layers: usize,
    /// `attention_head_dim` — 64.
    pub attention_head_dim: usize,
    /// `num_attention_heads` — 24.
    pub num_attention_heads: usize,
    /// `inner_dim` — 1536 = 24 * 64.
    pub inner_dim: usize,
    /// `enc_hidden_dim` — 2880 (GPT-OSS hidden size per selected layer).
    pub enc_hidden_dim: usize,
    /// `axes_dims_rope` — (frame, H, W) dims, sums to attention_head_dim.
    pub axes_dims_rope: [usize; 3],
    /// `gate_mlp` — true → SwiGLU MLP; false → GELU FeedForward (dead in Lens).
    pub gate_mlp: bool,
    /// `rms_norm` — true → RMSNorm per stream; false → LayerNorm (dead in Lens).
    pub rms_norm: bool,
    /// `multi_layer_encoder_feature` — true → 4 GPT-OSS layers stacked.
    pub multi_layer_encoder_feature: bool,
    /// Indices of GPT-OSS hidden states fed in via `encoder_hidden_states`.
    pub selected_layer_index: [usize; 4],
}

impl Default for LensDiTConfig {
    fn default() -> Self {
        Self {
            patch_size: 2,
            in_channels: 128,
            out_channels: 32,
            num_layers: 48,
            attention_head_dim: 64,
            num_attention_heads: 24,
            inner_dim: 1536,
            enc_hidden_dim: 2880,
            axes_dims_rope: [8, 28, 28],
            gate_mlp: true,
            rms_norm: true,
            multi_layer_encoder_feature: true,
            selected_layer_index: [5, 11, 17, 23],
        }
    }
}

impl LensDiTConfig {
    /// SwiGLU hidden dim: `dim * 8 / 3` per `GateMLP(dim, int(dim/3*8))` in
    /// `transformer.py:313`. For inner_dim=1536 → 4096 exactly.
    pub fn mlp_hidden(&self) -> usize {
        (self.inner_dim * 8) / 3
    }

    /// Joined text input dim into `txt_in`: 4 × 2880 = 11520 when
    /// `multi_layer_encoder_feature=True`.
    pub fn txt_in_dim(&self) -> usize {
        if self.multi_layer_encoder_feature {
            self.enc_hidden_dim * self.selected_layer_index.len()
        } else {
            self.enc_hidden_dim
        }
    }
}

// ---------------------------------------------------------------------------
// LensEmbedRope — 3-axis axial complex-polar RoPE (Stage A3)
// ---------------------------------------------------------------------------
//
// Mirrors `LensEmbedRope` in `transformer.py:100-185`. The Python module
// pre-builds two complex tables of shape `[4096, 32]` (32 = sum(axes_dim)/2):
//
//   pos_index = arange(4096)
//   neg_index = arange(4096).flip(0) * -1 - 1     # [-4096, ..., -1]
//   pos_freqs = cat([_rope_params(pos_index, d, theta) for d in axes_dim], 1)
//   neg_freqs = cat([_rope_params(neg_index, d, theta) for d in axes_dim], 1)
//
// where `_rope_params(idx, d, theta) = polar(1, outer(idx, theta^(-arange(0,d,2)/d)))`.
//
// `_compute_video_freqs(frame=1, h, w, scale_rope=True)`:
//   * split pos/neg by [4, 14, 14] along dim=1
//   * frame axis: pos[0][0:1] → broadcast [1, h, w, 4] → [h*w, 4]
//   * H axis (scale_rope=True): cat(neg[1][-(h-h/2):], pos[1][:h/2], 0)
//                                  → broadcast [1, h, w, 14] → [h*w, 14]
//   * W axis: cat(neg[2][-(w-w/2):], pos[2][:w/2], 0)
//                                  → broadcast [1, 1, w, 14] → [h*w, 14]
//   * concat along last → [h*w, 32]
//
// `forward(...)` returns (vid_freqs, txt_freqs) where
//   txt_freqs = pos_freqs[max_vid_idx : max_vid_idx + S_txt]
// and `max_vid_idx = max(h/2, w/2)` when scale_rope=True.
//
// Layout match: `apply_rotary_emb_lens` does
//   `view_as_complex(x.reshape(..., -1, 2)) * freqs_cis`
// → consecutive pairs (x[..., 2k], x[..., 2k+1]) form one complex number.
// That is the **interleaved-pair** layout, which is what
// `flame_core::bf16_ops::rope_fused_bf16` consumes. So we just store cos and
// sin at table-build time and feed them straight to the fused kernel later.

const ROPE_TABLE_ROWS: usize = 4096;

/// Cached GPU-resident `(cos, sin)` for the image stream of a given `(h, w)`.
struct CachedImgFreqs {
    cos: Tensor,
    sin: Tensor,
}

/// Cached GPU-resident `(cos, sin)` for the text stream at a given
/// `(max_vid_index, s_txt)` slice of the pos table.
struct CachedTxtFreqs {
    cos: Tensor,
    sin: Tensor,
}

/// 3-axis axial RoPE following Lens's complex-polar formulation, but stored
/// host-side as separate `cos` and `sin` `f32` matrices so the GPU upload
/// drops the imaginary-vs-real bookkeeping. The per-`(h, w)` GPU tensors are
/// cached on first use behind a `Mutex` so `freqs_for` is cheap on hot paths.
pub struct LensEmbedRope {
    theta: f64,
    axes_dim: [usize; 3],
    scale_rope: bool,
    /// Half of `sum(axes_dim)` — number of complex columns. For Lens this is 32.
    half_dim: usize,
    /// Per-axis half widths `[dim/2 for dim in axes_dim]` — for Lens `[4, 14, 14]`.
    half_axes: [usize; 3],
    /// Row-major `[4096, half_dim]` host F32 tables.
    pos_cos_host: Vec<f32>,
    pos_sin_host: Vec<f32>,
    neg_cos_host: Vec<f32>,
    neg_sin_host: Vec<f32>,
    /// GPU cache for image freqs keyed by `(h, w)`.
    img_cache: Mutex<HashMap<(usize, usize), Arc<CachedImgFreqs>>>,
    /// GPU cache for text freqs keyed by `(max_vid_index, s_txt)`.
    txt_cache: Mutex<HashMap<(usize, usize), Arc<CachedTxtFreqs>>>,
}

impl LensEmbedRope {
    /// Build the complex-polar tables on the host. No GPU allocations here.
    pub fn new(theta: f64, axes_dim: [usize; 3], scale_rope: bool) -> Result<Self> {
        for d in &axes_dim {
            if *d % 2 != 0 {
                return Err(Error::InvalidInput(format!(
                    "LensEmbedRope: each axes_dim entry must be even, got {axes_dim:?}"
                )));
            }
        }
        let half_axes = [axes_dim[0] / 2, axes_dim[1] / 2, axes_dim[2] / 2];
        let half_dim: usize = half_axes.iter().sum();

        let rows = ROPE_TABLE_ROWS;
        let mut pos_cos = vec![0f32; rows * half_dim];
        let mut pos_sin = vec![0f32; rows * half_dim];
        let mut neg_cos = vec![0f32; rows * half_dim];
        let mut neg_sin = vec![0f32; rows * half_dim];

        // For each axis d, compute the column-block [..., half_axes[axis]].
        let mut col_offset = 0usize;
        for (axis_idx, &d) in axes_dim.iter().enumerate() {
            let half = half_axes[axis_idx];
            // base[k] = theta^(-(2k)/d) for k in 0..half
            // i.e. 1.0 / theta^(arange(0, d, 2) / d) — Python `_rope_params`.
            let mut base = vec![0f64; half];
            for k in 0..half {
                let exponent = (2 * k) as f64 / d as f64;
                base[k] = 1.0 / theta.powf(exponent);
            }

            for row in 0..rows {
                // pos_index[row] = row
                let pos_n = row as f64;
                // neg_index[row] = (rows - 1 - row) * -1 - 1
                //                = -(rows - row)
                // So neg_index[0] = -rows = -4096; neg_index[4095] = -1. ✓
                let neg_n = -(rows as f64 - row as f64);

                for k in 0..half {
                    let arg_pos = pos_n * base[k];
                    let arg_neg = neg_n * base[k];
                    let dst = row * half_dim + col_offset + k;
                    pos_cos[dst] = arg_pos.cos() as f32;
                    pos_sin[dst] = arg_pos.sin() as f32;
                    neg_cos[dst] = arg_neg.cos() as f32;
                    neg_sin[dst] = arg_neg.sin() as f32;
                }
            }
            col_offset += half;
        }
        debug_assert_eq!(col_offset, half_dim);

        Ok(Self {
            theta,
            axes_dim,
            scale_rope,
            half_dim,
            half_axes,
            pos_cos_host: pos_cos,
            pos_sin_host: pos_sin,
            neg_cos_host: neg_cos,
            neg_sin_host: neg_sin,
            img_cache: Mutex::new(HashMap::new()),
            txt_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn theta(&self) -> f64 {
        self.theta
    }
    pub fn axes_dim(&self) -> [usize; 3] {
        self.axes_dim
    }
    pub fn scale_rope(&self) -> bool {
        self.scale_rope
    }
    pub fn half_dim(&self) -> usize {
        self.half_dim
    }

    /// Build the per-(h, w, s_txt) `(img_cos, img_sin, txt_cos, txt_sin)` BF16
    /// tensors on `device`. Per-shape results are cached.
    ///
    /// Shapes (all BF16):
    ///   * `img_*`: `[h*w, half_dim]`   (frame=1, so `seq_lens = h*w`)
    ///   * `txt_*`: `[s_txt, half_dim]`
    ///
    /// When `scale_rope=true`, the text rows start at `max(h/2, w/2)` of the
    /// pos table — matching `_compute_video_freqs` + the `pos_freqs[max:max+S]`
    /// slice in `LensEmbedRope.forward`.
    pub fn freqs_for(
        &self,
        h: usize,
        w: usize,
        s_txt: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let img = self.img_freqs(h, w, device)?;
        let max_vid_index = if self.scale_rope {
            std::cmp::max(h / 2, w / 2)
        } else {
            std::cmp::max(h, w)
        };
        let txt = self.txt_freqs(max_vid_index, s_txt, device)?;
        Ok((img.cos.clone(), img.sin.clone(), txt.cos.clone(), txt.sin.clone()))
    }

    fn img_freqs(
        &self,
        h: usize,
        w: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Arc<CachedImgFreqs>> {
        {
            let cache = self.img_cache.lock().unwrap();
            if let Some(c) = cache.get(&(h, w)) {
                return Ok(Arc::clone(c));
            }
        }

        let seq = h * w;
        let half_axes = self.half_axes;
        let half_dim = self.half_dim;
        let (frame_half, h_half, w_half) = (half_axes[0], half_axes[1], half_axes[2]);

        // Build [seq, half_dim] cos and sin on the host.
        let mut cos_host = vec![0f32; seq * half_dim];
        let mut sin_host = vec![0f32; seq * half_dim];

        // Frame axis: pos_freqs[0][0:1] — single row of width `frame_half`.
        // i.e. row 0 of pos table, cols [0 .. frame_half).
        // Broadcast across every (yy, xx).
        // Source rows in pos_*_host: row 0, cols 0..frame_half.
        // Destination cols within [0, frame_half) at every seq position.
        for yy in 0..h {
            for xx in 0..w {
                let dst_row = (yy * w + xx) * half_dim;
                for k in 0..frame_half {
                    let src = 0 * half_dim + 0 + k; // row=0, col=k
                    cos_host[dst_row + k] = self.pos_cos_host[src];
                    sin_host[dst_row + k] = self.pos_sin_host[src];
                }
            }
        }

        // Helper: source row index in pos_*_host / neg_*_host within column-block
        // for axis 1 (H) or axis 2 (W). The column block starts at `axis_col_offset`
        // (= frame_half for H, frame_half+h_half for W) of the table.
        // Each axis contributes `axis_half` columns.

        let h_col_offset = frame_half;
        let w_col_offset = frame_half + h_half;

        // Height axis rows: scale_rope=true → cat(neg[1][-(h-h/2):], pos[1][:h/2])
        //                                       len = h
        // scale_rope=false → pos[1][:h]
        // We materialize a length-h list of (cos, sin) row-vectors for axis H.
        let h_lo = h / 2;
        let h_hi = h - h_lo; // rows taken from negative table

        let mut height_cos = vec![0f32; h * h_half];
        let mut height_sin = vec![0f32; h * h_half];
        if self.scale_rope {
            // First `h_hi` rows come from neg_freqs[1][-h_hi:]
            // i.e. rows (ROPE_TABLE_ROWS - h_hi)..ROPE_TABLE_ROWS in neg table,
            // cols [h_col_offset .. h_col_offset + h_half).
            for i in 0..h_hi {
                let src_row = ROPE_TABLE_ROWS - h_hi + i;
                for k in 0..h_half {
                    let src = src_row * half_dim + h_col_offset + k;
                    height_cos[i * h_half + k] = self.neg_cos_host[src];
                    height_sin[i * h_half + k] = self.neg_sin_host[src];
                }
            }
            // Next `h_lo` rows come from pos_freqs[1][:h_lo].
            for i in 0..h_lo {
                let dst_i = h_hi + i;
                let src_row = i;
                for k in 0..h_half {
                    let src = src_row * half_dim + h_col_offset + k;
                    height_cos[dst_i * h_half + k] = self.pos_cos_host[src];
                    height_sin[dst_i * h_half + k] = self.pos_sin_host[src];
                }
            }
        } else {
            for i in 0..h {
                let src_row = i;
                for k in 0..h_half {
                    let src = src_row * half_dim + h_col_offset + k;
                    height_cos[i * h_half + k] = self.pos_cos_host[src];
                    height_sin[i * h_half + k] = self.pos_sin_host[src];
                }
            }
        }

        // Width axis rows — same construction with w/h_lo replaced by w/w_lo.
        let w_lo = w / 2;
        let w_hi = w - w_lo;
        let mut width_cos = vec![0f32; w * w_half];
        let mut width_sin = vec![0f32; w * w_half];
        if self.scale_rope {
            for i in 0..w_hi {
                let src_row = ROPE_TABLE_ROWS - w_hi + i;
                for k in 0..w_half {
                    let src = src_row * half_dim + w_col_offset + k;
                    width_cos[i * w_half + k] = self.neg_cos_host[src];
                    width_sin[i * w_half + k] = self.neg_sin_host[src];
                }
            }
            for i in 0..w_lo {
                let dst_i = w_hi + i;
                let src_row = i;
                for k in 0..w_half {
                    let src = src_row * half_dim + w_col_offset + k;
                    width_cos[dst_i * w_half + k] = self.pos_cos_host[src];
                    width_sin[dst_i * w_half + k] = self.pos_sin_host[src];
                }
            }
        } else {
            for i in 0..w {
                let src_row = i;
                for k in 0..w_half {
                    let src = src_row * half_dim + w_col_offset + k;
                    width_cos[i * w_half + k] = self.pos_cos_host[src];
                    width_sin[i * w_half + k] = self.pos_sin_host[src];
                }
            }
        }

        // Fill destination rows: for each (yy, xx), columns are
        //   [frame block | H[yy] | W[xx]]
        for yy in 0..h {
            for xx in 0..w {
                let dst_row = (yy * w + xx) * half_dim;
                // H block at cols [frame_half .. frame_half + h_half).
                for k in 0..h_half {
                    cos_host[dst_row + h_col_offset + k] = height_cos[yy * h_half + k];
                    sin_host[dst_row + h_col_offset + k] = height_sin[yy * h_half + k];
                }
                // W block at cols [frame_half + h_half .. half_dim).
                for k in 0..w_half {
                    cos_host[dst_row + w_col_offset + k] = width_cos[xx * w_half + k];
                    sin_host[dst_row + w_col_offset + k] = width_sin[xx * w_half + k];
                }
            }
        }

        let shape = Shape::from_dims(&[seq, half_dim]);
        let cos_f32 = Tensor::from_vec(cos_host, shape.clone(), device.clone())?;
        let sin_f32 = Tensor::from_vec(sin_host, shape, device.clone())?;
        let cos_bf16 = cos_f32.to_dtype(DType::BF16)?;
        let sin_bf16 = sin_f32.to_dtype(DType::BF16)?;

        let cached = Arc::new(CachedImgFreqs {
            cos: cos_bf16,
            sin: sin_bf16,
        });
        let mut cache = self.img_cache.lock().unwrap();
        cache.insert((h, w), Arc::clone(&cached));
        Ok(cached)
    }

    fn txt_freqs(
        &self,
        max_vid_index: usize,
        s_txt: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Arc<CachedTxtFreqs>> {
        let key = (max_vid_index, s_txt);
        {
            let cache = self.txt_cache.lock().unwrap();
            if let Some(c) = cache.get(&key) {
                return Ok(Arc::clone(c));
            }
        }

        if max_vid_index + s_txt > ROPE_TABLE_ROWS {
            return Err(Error::InvalidInput(format!(
                "LensEmbedRope txt slice [{max_vid_index}, {}) exceeds table rows {ROPE_TABLE_ROWS}",
                max_vid_index + s_txt
            )));
        }

        let half_dim = self.half_dim;
        let mut cos_host = vec![0f32; s_txt * half_dim];
        let mut sin_host = vec![0f32; s_txt * half_dim];
        for i in 0..s_txt {
            let src_row = max_vid_index + i;
            for k in 0..half_dim {
                let src = src_row * half_dim + k;
                cos_host[i * half_dim + k] = self.pos_cos_host[src];
                sin_host[i * half_dim + k] = self.pos_sin_host[src];
            }
        }

        let shape = Shape::from_dims(&[s_txt, half_dim]);
        let cos_f32 = Tensor::from_vec(cos_host, shape.clone(), device.clone())?;
        let sin_f32 = Tensor::from_vec(sin_host, shape, device.clone())?;
        let cos_bf16 = cos_f32.to_dtype(DType::BF16)?;
        let sin_bf16 = sin_f32.to_dtype(DType::BF16)?;

        let cached = Arc::new(CachedTxtFreqs {
            cos: cos_bf16,
            sin: sin_bf16,
        });
        let mut cache = self.txt_cache.lock().unwrap();
        cache.insert(key, Arc::clone(&cached));
        Ok(cached)
    }
}

// ---------------------------------------------------------------------------
// LensDiTBlock — weight container (Stage A2)
// ---------------------------------------------------------------------------
//
// Mirrors `LensTransformerBlock` (transformer.py:289-325) with:
//   * `gate_mlp=True` (SwiGLU branch is the only live path)
//   * `rms_norm=True` (LayerNorm branch is dead)
//   * `eps=1e-6` for the block norms
//   * `eps=1e-5` for the QK head norms (LensJointAttention default)
//
// Modulation in python uses `nn.Sequential(SiLU, Linear(dim, 6*dim))`, which
// stores the Linear under key `img_mod.1` (index 1 of the Sequential). We
// keep the same naming convention for `load_weights`.

pub struct LensDiTBlock {
    // Joint attention: fused QKV per stream → 3·inner_dim outputs each.
    pub img_qkv: Linear, // 1536 → 4608, bias=true
    pub txt_qkv: Linear, // 1536 → 4608, bias=true

    // Q/K RMSNorm per head (over head_dim=64). LensJointAttention uses
    // eps=1e-5 by default.
    pub norm_q: Tensor,        // [head_dim] BF16
    pub norm_k: Tensor,        // [head_dim] BF16
    pub norm_added_q: Tensor,  // [head_dim] BF16
    pub norm_added_k: Tensor,  // [head_dim] BF16

    // Output projections.
    // Python keys: `attn.to_out.0` (img), `attn.to_add_out` (txt).
    pub img_out: Linear, // 1536 → 1536, bias=true
    pub txt_out: Linear, // 1536 → 1536, bias=true

    // Modulation: Linear(dim → 6·dim). Stored under `img_mod.1` / `txt_mod.1`.
    pub img_mod: Linear, // 1536 → 9216, bias=true
    pub txt_mod: Linear, // 1536 → 9216, bias=true

    // RMSNorm scales for block norms (eps=1e-6 at use site).
    pub img_norm1: Tensor, // [inner_dim]
    pub img_norm2: Tensor, // [inner_dim]
    pub txt_norm1: Tensor, // [inner_dim]
    pub txt_norm2: Tensor, // [inner_dim]

    // SwiGLU MLP per stream. `w2(silu(w1(x)) * w3(x))`, all biases off.
    // Hidden = inner_dim * 8 / 3 = 4096 for inner_dim=1536.
    pub img_mlp_w1: Linear, // 1536 → 4096, bias=false
    pub img_mlp_w2: Linear, // 4096 → 1536, bias=false
    pub img_mlp_w3: Linear, // 1536 → 4096, bias=false
    pub txt_mlp_w1: Linear,
    pub txt_mlp_w2: Linear,
    pub txt_mlp_w3: Linear,
}

impl LensDiTBlock {
    /// Allocate zeroed BF16 weight tensors. No CUDA kernel launches —
    /// `Linear::new_zeroed` just allocates the storage; weights are filled by
    /// `LensTransformer2DModel::load_weights` in Stage A5.
    pub fn new_zeroed(cfg: &LensDiTConfig, device: &Arc<CudaDevice>) -> Result<Self> {
        let dim = cfg.inner_dim;
        let head_dim = cfg.attention_head_dim;
        let qkv_out = 3 * dim;
        let mod_out = 6 * dim;
        let mlp_hidden = cfg.mlp_hidden();

        // QKV per stream — bias=true.
        let img_qkv = Linear::new_zeroed(dim, qkv_out, true, device)?;
        let txt_qkv = Linear::new_zeroed(dim, qkv_out, true, device)?;

        // QK head norms — RMSNorm weight only.
        let norm_shape = Shape::from_dims(&[head_dim]);
        let norm_q = Tensor::zeros_dtype(norm_shape.clone(), DType::BF16, device.clone())?;
        let norm_k = Tensor::zeros_dtype(norm_shape.clone(), DType::BF16, device.clone())?;
        let norm_added_q = Tensor::zeros_dtype(norm_shape.clone(), DType::BF16, device.clone())?;
        let norm_added_k = Tensor::zeros_dtype(norm_shape, DType::BF16, device.clone())?;

        // Output projections — bias=true.
        let img_out = Linear::new_zeroed(dim, dim, true, device)?;
        let txt_out = Linear::new_zeroed(dim, dim, true, device)?;

        // Modulation — bias=true.
        let img_mod = Linear::new_zeroed(dim, mod_out, true, device)?;
        let txt_mod = Linear::new_zeroed(dim, mod_out, true, device)?;

        // Block RMSNorm scales (eps=1e-6 applied at use site).
        let block_norm_shape = Shape::from_dims(&[dim]);
        let img_norm1 = Tensor::zeros_dtype(block_norm_shape.clone(), DType::BF16, device.clone())?;
        let img_norm2 = Tensor::zeros_dtype(block_norm_shape.clone(), DType::BF16, device.clone())?;
        let txt_norm1 = Tensor::zeros_dtype(block_norm_shape.clone(), DType::BF16, device.clone())?;
        let txt_norm2 = Tensor::zeros_dtype(block_norm_shape, DType::BF16, device.clone())?;

        // SwiGLU MLP per stream — bias=false everywhere.
        let img_mlp_w1 = Linear::new_zeroed(dim, mlp_hidden, false, device)?;
        let img_mlp_w2 = Linear::new_zeroed(mlp_hidden, dim, false, device)?;
        let img_mlp_w3 = Linear::new_zeroed(dim, mlp_hidden, false, device)?;
        let txt_mlp_w1 = Linear::new_zeroed(dim, mlp_hidden, false, device)?;
        let txt_mlp_w2 = Linear::new_zeroed(mlp_hidden, dim, false, device)?;
        let txt_mlp_w3 = Linear::new_zeroed(dim, mlp_hidden, false, device)?;

        Ok(Self {
            img_qkv,
            txt_qkv,
            norm_q,
            norm_k,
            norm_added_q,
            norm_added_k,
            img_out,
            txt_out,
            img_mod,
            txt_mod,
            img_norm1,
            img_norm2,
            txt_norm1,
            txt_norm2,
            img_mlp_w1,
            img_mlp_w2,
            img_mlp_w3,
            txt_mlp_w1,
            txt_mlp_w2,
            txt_mlp_w3,
        })
    }

    /// One dual-stream LensTransformerBlock forward.
    ///
    /// Mirrors `LensTransformerBlock.forward` in `transformer.py:332-362` with
    /// `rms_norm=True` and `gate_mlp=True` (the only path Lens config exercises;
    /// the LayerNorm/FeedForward branches are dead in the live model).
    ///
    /// Returns `(encoder_hidden_states, hidden_states)` in the same order as
    /// the Python signature.
    ///
    /// Shapes (all BF16):
    ///   * `hidden_states`:         `[B, S_img, inner_dim=1536]`
    ///   * `encoder_hidden_states`: `[B, S_txt, 1536]` (S_txt may be 0)
    ///   * `temb`:                  `[B, 1536]`
    ///   * `img_cos`, `img_sin`:    `[S_img, head_dim/2=32]`
    ///   * `txt_cos`, `txt_sin`:    `[S_txt, 32]` (zero-len allowed)
    ///   * `attention_mask`:        `[B, 1, 1, S_img+S_txt]` BF16 KEEP-MASK
    ///                              (1.0 = valid, 0.0 = padded). NOTE: flame-core
    ///                              `sdpa::forward` interprets the mask as
    ///                              `(1-mask)*-inf` (binary keep) — NOT additive.
    ///                              Stage A5 builds the mask in this form from
    ///                              the Python-style bool encoder mask.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        attention_mask: &Tensor,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        // ---- 1) Modulation projection: silu(temb) → linear → split 6 ways ----
        // Python: img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        //         shift, scale, gate = mod.chunk(3, dim=-1)
        // self.img_mod is `nn.Sequential(SiLU, Linear)` so the linear sees silu(temb).
        let temb_act = temb.silu()?;
        let img_mod_full = self.img_mod.forward(&temb_act)?; // [B, 6*dim]
        let txt_mod_full = self.txt_mod.forward(&temb_act)?; // [B, 6*dim]
        let img_halves = img_mod_full.chunk(2, 1)?; // 2 × [B, 3*dim]
        let txt_halves = txt_mod_full.chunk(2, 1)?;
        let img_m1 = img_halves[0].chunk(3, 1)?; // 3 × [B, dim]
        let img_m2 = img_halves[1].chunk(3, 1)?;
        let txt_m1 = txt_halves[0].chunk(3, 1)?;
        let txt_m2 = txt_halves[1].chunk(3, 1)?;
        let (img_shift1, img_scale1, img_gate1) = (&img_m1[0], &img_m1[1], &img_m1[2]);
        let (img_shift2, img_scale2, img_gate2) = (&img_m2[0], &img_m2[1], &img_m2[2]);
        let (txt_shift1, txt_scale1, txt_gate1) = (&txt_m1[0], &txt_m1[1], &txt_m1[2]);
        let (txt_shift2, txt_scale2, txt_gate2) = (&txt_m2[0], &txt_m2[1], &txt_m2[2]);

        // ---- 2) First norm + modulate ----
        // img_norm1 / txt_norm1 are RMSNorm scales (eps=1e-6).
        let img_n1 = flame_core::cuda_ops_bf16::rms_norm_bf16(
            hidden_states,
            Some(&self.img_norm1),
            1e-6,
        )?;
        let txt_n1 = flame_core::cuda_ops_bf16::rms_norm_bf16(
            encoder_hidden_states,
            Some(&self.txt_norm1),
            1e-6,
        )?;
        let img_modulated = modulate(&img_n1, img_shift1, img_scale1)?;
        let txt_modulated = modulate(&txt_n1, txt_shift1, txt_scale1)?;

        // ---- 3) Joint attention ----
        let (img_attn, txt_attn) = self.joint_attention(
            &img_modulated,
            &txt_modulated,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            attention_mask,
            num_heads,
            head_dim,
        )?;

        // ---- 4) Gate + residual (gate broadcast over seq dim) ----
        let img_gate1_b = img_gate1.unsqueeze(1)?; // [B, 1, dim]
        let txt_gate1_b = txt_gate1.unsqueeze(1)?;
        let hidden_states = hidden_states.add(&img_gate1_b.mul(&img_attn)?)?;
        let encoder_hidden_states = encoder_hidden_states.add(&txt_gate1_b.mul(&txt_attn)?)?;

        // ---- 5) Second norm + modulate + SwiGLU MLP + gate + residual ----
        let img_n2 = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &hidden_states,
            Some(&self.img_norm2),
            1e-6,
        )?;
        let img_modulated2 = modulate(&img_n2, img_shift2, img_scale2)?;
        let img_mlp_out = swiglu_mlp(
            &img_modulated2,
            &self.img_mlp_w1,
            &self.img_mlp_w2,
            &self.img_mlp_w3,
        )?;
        let img_gate2_b = img_gate2.unsqueeze(1)?;
        let hidden_states = hidden_states.add(&img_gate2_b.mul(&img_mlp_out)?)?;

        let txt_n2 = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &encoder_hidden_states,
            Some(&self.txt_norm2),
            1e-6,
        )?;
        let txt_modulated2 = modulate(&txt_n2, txt_shift2, txt_scale2)?;
        let txt_mlp_out = swiglu_mlp(
            &txt_modulated2,
            &self.txt_mlp_w1,
            &self.txt_mlp_w2,
            &self.txt_mlp_w3,
        )?;
        let txt_gate2_b = txt_gate2.unsqueeze(1)?;
        let encoder_hidden_states = encoder_hidden_states.add(&txt_gate2_b.mul(&txt_mlp_out)?)?;

        // Python returns (encoder_hidden_states, hidden_states).
        Ok((encoder_hidden_states, hidden_states))
    }

    /// Joint image+text attention with per-stream QK RMSNorm + RoPE.
    ///
    /// Mirrors `LensJointAttention.forward` (`transformer.py:221-281`).
    ///
    /// - `img_modulated`: `[B, S_img, dim]`
    /// - `txt_modulated`: `[B, S_txt, dim]` (S_txt may be 0)
    /// - `*_cos`, `*_sin`: `[S, head_dim/2]` BF16 RoPE tables
    /// - `attention_mask`: BF16 keep-mask `[B, 1, 1, S_img+S_txt]`
    /// - `num_heads`, `head_dim`: e.g. 24, 64 for Lens
    ///
    /// Returns `(img_out, txt_out)`, each `[B, S, dim]`.
    #[allow(clippy::too_many_arguments)]
    fn joint_attention(
        &self,
        img_modulated: &Tensor,
        txt_modulated: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        attention_mask: &Tensor,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let img_dims = img_modulated.shape().dims().to_vec();
        let txt_dims = txt_modulated.shape().dims().to_vec();
        if img_dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "joint_attention: img must be 3D, got {img_dims:?}"
            )));
        }
        if txt_dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "joint_attention: txt must be 3D, got {txt_dims:?}"
            )));
        }
        let (b, s_img) = (img_dims[0], img_dims[1]);
        let s_txt = txt_dims[1];
        let half = head_dim / 2;

        // ---- QKV projections per stream ----
        let img_qkv = self.img_qkv.forward(img_modulated)?; // [B, S_img, 3*H*D]
        let txt_qkv = self.txt_qkv.forward(txt_modulated)?; // [B, S_txt, 3*H*D]

        // qkv_split_permute_bf16 returns 3 tensors of shape [B, H, S, D].
        let (img_q, img_k, img_v) =
            flame_core::bf16_ops::qkv_split_permute_bf16(&img_qkv, num_heads, head_dim)?;
        let (txt_q, txt_k, txt_v) = if s_txt > 0 {
            let (q, k, v) =
                flame_core::bf16_ops::qkv_split_permute_bf16(&txt_qkv, num_heads, head_dim)?;
            (Some(q), Some(k), Some(v))
        } else {
            (None, None, None)
        };

        // ---- QK RMSNorm per-head (eps=1e-5 from LensJointAttention default) ----
        // rms_norm_bf16 normalizes over the last dim (head_dim=64) — matches Python.
        let img_q =
            flame_core::cuda_ops_bf16::rms_norm_bf16(&img_q, Some(&self.norm_q), 1e-5)?;
        let img_k =
            flame_core::cuda_ops_bf16::rms_norm_bf16(&img_k, Some(&self.norm_k), 1e-5)?;
        let (txt_q, txt_k, txt_v) = if s_txt > 0 {
            let tq = flame_core::cuda_ops_bf16::rms_norm_bf16(
                txt_q.as_ref().unwrap(),
                Some(&self.norm_added_q),
                1e-5,
            )?;
            let tk = flame_core::cuda_ops_bf16::rms_norm_bf16(
                txt_k.as_ref().unwrap(),
                Some(&self.norm_added_k),
                1e-5,
            )?;
            (Some(tq), Some(tk), txt_v)
        } else {
            (None, None, None)
        };

        // ---- RoPE apply ----
        // rope_fused_bf16 expects cos/sin shape [1, 1, S, half]. Our tables are
        // [S, half], so reshape.
        let img_cos_b = img_cos.reshape(&[1, 1, s_img, half])?;
        let img_sin_b = img_sin.reshape(&[1, 1, s_img, half])?;
        let img_q = flame_core::bf16_ops::rope_fused_bf16(&img_q, &img_cos_b, &img_sin_b)?;
        let img_k = flame_core::bf16_ops::rope_fused_bf16(&img_k, &img_cos_b, &img_sin_b)?;

        // F4 guard: skip RoPE on empty text. txt_cos shape[0] is 0 in that case.
        let (txt_q, txt_k) = if s_txt > 0 {
            let txt_cos_b = txt_cos.reshape(&[1, 1, s_txt, half])?;
            let txt_sin_b = txt_sin.reshape(&[1, 1, s_txt, half])?;
            let tq = flame_core::bf16_ops::rope_fused_bf16(
                txt_q.as_ref().unwrap(),
                &txt_cos_b,
                &txt_sin_b,
            )?;
            let tk = flame_core::bf16_ops::rope_fused_bf16(
                txt_k.as_ref().unwrap(),
                &txt_cos_b,
                &txt_sin_b,
            )?;
            (Some(tq), Some(tk))
        } else {
            (None, None)
        };

        // ---- Joint sequence: cat along seq dim (dim=2 of [B,H,S,D]). ----
        // Python order is [img, txt] (transformer.py:264) — image first.
        // `Tensor::cat` materializes non-contig inputs and returns a fresh
        // contiguous tensor (see flame-core tensor_ops_extended.rs:323-334), so
        // no explicit `.contiguous()` is needed afterward.
        let (q, k, v) = if s_txt > 0 {
            let q = Tensor::cat(&[&img_q, txt_q.as_ref().unwrap()], 2)?;
            let k = Tensor::cat(&[&img_k, txt_k.as_ref().unwrap()], 2)?;
            let v = Tensor::cat(&[&img_v, txt_v.as_ref().unwrap()], 2)?;
            (q, k, v)
        } else {
            (img_q, img_k, img_v)
        };

        // ---- SDPA with broadcast mask [B,1,1,S_total] ----
        // flame_core::sdpa::forward interprets `mask` as a binary keep-mask
        // applied as `(1-mask)*-inf` to logits before softmax (sdpa.rs:1771-1777).
        // Stage A5 builds the mask in this convention.
        let attn_out = flame_core::sdpa::forward(&q, &k, &v, Some(attention_mask))
            .map_err(|e| Error::InvalidOperation(format!("sdpa: {e}")))?;

        // attn_out is [B, H, S_total, D]. Transpose to [B, S_total, H, D]
        // then reshape to [B, S_total, H*D] = [B, S_total, dim].
        let attn_t = attn_out.permute(&[0, 2, 1, 3])?;
        let s_total = s_img + s_txt;
        let dim = num_heads * head_dim;
        let attn_reshaped = attn_t.contiguous()?.reshape(&[b, s_total, dim])?;
        // FLAME-CORE: `permute(0,2,1,3)` followed by `reshape` requires contiguous
        // intermediate (reshape can't fuse a non-contig view in general). The
        // `.contiguous()` here is the standard pattern in z-image / klein after
        // an SDPA-out transpose; no producer-side fix possible without API change.

        // ---- Split image vs text along seq dim ----
        // Use `narrow_owning` because the two outputs flow into independent
        // Linear forwards downstream and the joined `attn_reshaped` should be
        // free to drop (CONTEXT trap: narrow holds parent alive).
        let img_out = attn_reshaped.narrow_owning(1, 0, s_img)?;
        let img_out = self.img_out.forward(&img_out)?;
        let txt_out = if s_txt > 0 {
            let t = attn_reshaped.narrow_owning(1, s_img, s_txt)?;
            self.txt_out.forward(&t)?
        } else {
            // Preserve the [B, 0, dim] shape so callers can still add/multiply
            // by zero-len residuals without a shape mismatch.
            Tensor::zeros_dtype(
                Shape::from_dims(&[b, 0, dim]),
                DType::BF16,
                img_modulated.device().clone(),
            )?
        };

        Ok((img_out, txt_out))
    }
}

// ---------------------------------------------------------------------------
// Block-private helpers (modulate, SwiGLU MLP, timestep embedding, AdaLN-out)
// ---------------------------------------------------------------------------

/// `x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)`. BF16 in/out.
///
/// Shapes: `x: [B, S, D]`, `shift/scale: [B, D]`. Broadcasts over seq dim.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one_plus_scale = scale.add_scalar(1.0)?.unsqueeze(1)?; // [B, 1, D]
    let scaled = x.mul(&one_plus_scale)?;
    scaled.add(&shift.unsqueeze(1)?)
}

/// SwiGLU MLP: `w2( silu(w1(x)) * w3(x) )`. All biases off in Lens GateMLP.
fn swiglu_mlp(x: &Tensor, w1: &Linear, w2: &Linear, w3: &Linear) -> Result<Tensor> {
    let gate = w1.forward(x)?;
    let up = w3.forward(x)?;
    let activated = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?;
    w2.forward(&activated)
}

/// Sinusoidal timestep embedding (diffusers `get_timestep_embedding`).
///
/// Mirrors `transformer.py:30-53` with `flip_sin_to_cos=True`,
/// `downscale_freq_shift=0`, `scale=1000`, `max_period=10000`.
///
/// `timesteps`: `[B]` in any float dtype.
/// Returns `[B, embedding_dim]` in `out_dtype`.
fn timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    out_dtype: DType,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    // Python (with flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000):
    //   half = embedding_dim // 2
    //   exponent = -log(max_period) * arange(half) / half
    //   emb = exp(exponent)
    //   args = t[:, None].float() * emb[None, :] * scale
    //   emb = cat([sin(args), cos(args)], -1)
    //   if flip_sin_to_cos: emb = cat([emb[:, half:], emb[:, :half]], -1)
    //                                       (= cos, sin)
    let half = embedding_dim / 2;
    let max_period = 10000.0f64;
    let scale = 1000.0f32;

    // Frequencies in F32 (Python casts emb back to t.dtype but the F32 path is
    // intermediate; we follow Flux1 idiom: do everything in F32, downcast once).
    let freq_data: Vec<f32> = (0..half)
        .map(|k| (-max_period.ln() * k as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), device.clone())?;

    // t in F32, scaled by `scale=1000`.
    let t_f32 = timesteps.to_dtype(DType::F32)?.mul_scalar(scale)?;
    let t_col = t_f32.unsqueeze(1)?; // [B, 1]
    let args = t_col.matmul(&freqs)?; // [B, half] F32

    let sin = args.sin()?;
    let cos = args.cos()?;
    // flip_sin_to_cos=True → final layout is [cos, sin] (cos in the FIRST half).
    let emb = Tensor::cat(&[&cos, &sin], 1)?;

    if embedding_dim % 2 == 1 {
        return Err(Error::InvalidInput(
            "timestep_embedding: odd embedding_dim not supported".into(),
        ));
    }

    emb.to_dtype(out_dtype)
}

/// AdaLayerNormContinuous (`elementwise_affine=False`, eps=1e-6).
///
/// Mirrors `diffusers.models.normalization.AdaLayerNormContinuous`:
///   shift, scale = linear(silu(temb)).chunk(2, dim=-1)
///   x = layer_norm(x, normalized_shape, None, None, eps)
///   return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
///
/// `x`: `[B, S, dim]` BF16. `temb`: `[B, dim]` BF16.
fn ada_layer_norm_continuous(
    x: &Tensor,
    temb: &Tensor,
    norm_out_linear: &Linear,
    dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let temb_act = temb.silu()?;
    let mod_params = norm_out_linear.forward(&temb_act)?; // [B, 2*dim]
    let chunks = mod_params.chunk(2, 1)?; // 2 × [B, dim]
    // diffusers AdaLayerNormContinuous (`normalization.py:337`):
    //   `scale, shift = torch.chunk(emb, 2, dim=1)`
    // — SCALE IS FIRST. Same convention used by FLUX1 final layer.
    let scale = &chunks[0];
    let shift = &chunks[1];
    // layer_norm signature is (input, normalized_shape, weight, bias, eps).
    let normed = flame_core::layer_norm::layer_norm(x, &[dim], None, None, eps)?;
    let one_plus_scale = scale.add_scalar(1.0)?.unsqueeze(1)?;
    normed.mul(&one_plus_scale)?.add(&shift.unsqueeze(1)?)
}

// ---------------------------------------------------------------------------
// LensTransformer2DModel — top-level (Stage A2)
// ---------------------------------------------------------------------------

pub struct LensTransformer2DModel {
    pub config: LensDiTConfig,

    // Patch projection: 128 → 1536, bias=true.
    pub img_in: Linear,

    // Text projection: 4 * 2880 → 1536, bias=true.
    pub txt_in: Linear,

    // Per-selected-layer text RMSNorm (eps=1e-5), len = 4.
    pub txt_norm: Vec<Tensor>, // each [enc_hidden_dim=2880] BF16

    // Timestep embedding MLP: Linear(256, 1536), SiLU, Linear(1536, 1536) — both biased.
    pub time_embed_linear_1: Linear,
    pub time_embed_linear_2: Linear,

    // AdaLayerNormContinuous final modulation: Linear(1536 → 2·1536) for (shift, scale).
    pub norm_out_linear: Linear,

    // Final projection: 1536 → 128 (patch² * out_channels = 4 * 32).
    pub proj_out: Linear,

    // 48 dual-stream blocks.
    pub blocks: Vec<LensDiTBlock>,

    // RoPE table builder + cache. Eagerly constructed so first forward is
    // not stalled by table init.
    pub rope: LensEmbedRope,
}

impl LensTransformer2DModel {
    /// Allocate zeroed BF16 weight tensors and build the RoPE host tables.
    /// Weights are filled by `load_weights` in Stage A5.
    pub fn new(config: LensDiTConfig, device: &Arc<CudaDevice>) -> Result<Self> {
        let dim = config.inner_dim;
        let head_dim = config.attention_head_dim;
        let enc_dim = config.enc_hidden_dim;
        let timestep_dim = 256; // `Timesteps(num_channels=256, ...)` in Python.

        let img_in = Linear::new_zeroed(config.in_channels, dim, true, device)?;
        let txt_in = Linear::new_zeroed(config.txt_in_dim(), dim, true, device)?;

        let txt_norm_shape = Shape::from_dims(&[enc_dim]);
        let mut txt_norm = Vec::with_capacity(config.selected_layer_index.len());
        for _ in 0..config.selected_layer_index.len() {
            txt_norm.push(Tensor::zeros_dtype(
                txt_norm_shape.clone(),
                DType::BF16,
                device.clone(),
            )?);
        }

        let time_embed_linear_1 = Linear::new_zeroed(timestep_dim, dim, true, device)?;
        let time_embed_linear_2 = Linear::new_zeroed(dim, dim, true, device)?;

        // AdaLayerNormContinuous: Linear(emb_dim, 2*hidden_dim) — bias=true,
        // outputs (shift, scale).
        let norm_out_linear = Linear::new_zeroed(dim, 2 * dim, true, device)?;

        let proj_out_dim = config.patch_size * config.patch_size * config.out_channels;
        let proj_out = Linear::new_zeroed(dim, proj_out_dim, true, device)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(LensDiTBlock::new_zeroed(&config, device)?);
        }

        // sanity: axes_dims_rope sum must equal head_dim
        let axes_sum: usize = config.axes_dims_rope.iter().sum();
        if axes_sum != head_dim {
            return Err(Error::InvalidInput(format!(
                "axes_dims_rope sum ({axes_sum}) must equal attention_head_dim ({head_dim})"
            )));
        }
        let rope = LensEmbedRope::new(10000.0, config.axes_dims_rope, true)?;

        Ok(Self {
            config,
            img_in,
            txt_in,
            txt_norm,
            time_embed_linear_1,
            time_embed_linear_2,
            norm_out_linear,
            proj_out,
            blocks,
            rope,
        })
    }

    /// Load BF16 weights from a `key → Tensor` map.
    ///
    /// Expected key prefixes (per `BUILD_PLAN.md` Weight mapping):
    ///   * `img_in.{weight,bias}`
    ///   * `txt_in.{weight,bias}`
    ///   * `txt_norm.{0..3}.weight`
    ///   * `time_text_embed.timestep_embedder.linear_{1,2}.{weight,bias}`
    ///   * `norm_out.linear.{weight,bias}`
    ///   * `proj_out.{weight,bias}`
    ///   * Per block `i` in `0..num_layers`:
    ///       - `transformer_blocks.{i}.attn.img_qkv.{weight,bias}`
    ///       - `transformer_blocks.{i}.attn.txt_qkv.{weight,bias}`
    ///       - `transformer_blocks.{i}.attn.norm_q.weight`
    ///       - `transformer_blocks.{i}.attn.norm_k.weight`
    ///       - `transformer_blocks.{i}.attn.norm_added_q.weight`
    ///       - `transformer_blocks.{i}.attn.norm_added_k.weight`
    ///       - `transformer_blocks.{i}.attn.to_out.0.{weight,bias}`
    ///       - `transformer_blocks.{i}.attn.to_add_out.{weight,bias}`
    ///       - `transformer_blocks.{i}.img_mod.1.{weight,bias}`
    ///       - `transformer_blocks.{i}.txt_mod.1.{weight,bias}`
    ///       - `transformer_blocks.{i}.{img,txt}_norm{1,2}.weight`
    ///       - `transformer_blocks.{i}.{img,txt}_mlp.w{1,2,3}.weight`
    ///
    /// Strict: returns an error if any expected key is missing OR any
    /// provided key is left unconsumed. The error message lists both lists
    /// with counts so debugging upstream conversion bugs is straightforward.
    ///
    /// Post-copy each parameter has `requires_grad=false`. Combined with
    /// `AutogradContext::set_enabled(false)` at the bin's main() (see the
    /// module-level AUTOGRAD NOTE), this prevents the 48-block × N-step
    /// autograd tape that would otherwise OOM.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        use std::collections::HashSet;

        let mut consumed: HashSet<String> = HashSet::new();
        let mut missing: Vec<String> = Vec::new();

        // Helper closures (capturing `weights`, `consumed`, `missing` via &mut).
        //
        // `copy_linear_weight`: look up `key` in `weights`, then
        // `linear.copy_weight_from`. Track consumption + missing.
        // `copy_linear_bias`: same for bias.
        // `copy_norm_scale`: look up `key`, shape-check against `dst.shape()`,
        // dtype-cast to BF16 if needed, replace `*dst`. `requires_grad=false`.
        fn copy_linear_weight(
            linear: &mut Linear,
            key: &str,
            weights: &HashMap<String, Tensor>,
            consumed: &mut HashSet<String>,
            missing: &mut Vec<String>,
        ) -> Result<()> {
            if let Some(src) = weights.get(key) {
                linear.copy_weight_from(src)?;
                // Strip the requires_grad flag that copy_weight_from preserves
                // from the destination (which was true on `new_zeroed`).
                linear.weight = linear.weight.clone().requires_grad_(false);
                consumed.insert(key.to_string());
            } else {
                missing.push(key.to_string());
            }
            Ok(())
        }
        fn copy_linear_bias(
            linear: &mut Linear,
            key: &str,
            weights: &HashMap<String, Tensor>,
            consumed: &mut HashSet<String>,
            missing: &mut Vec<String>,
        ) -> Result<()> {
            if let Some(src) = weights.get(key) {
                linear.copy_bias_from(src)?;
                if let Some(b) = linear.bias.take() {
                    linear.bias = Some(b.requires_grad_(false));
                }
                consumed.insert(key.to_string());
            } else {
                missing.push(key.to_string());
            }
            Ok(())
        }
        fn copy_norm_scale(
            dst: &mut Tensor,
            key: &str,
            weights: &HashMap<String, Tensor>,
            consumed: &mut HashSet<String>,
            missing: &mut Vec<String>,
        ) -> Result<()> {
            if let Some(src) = weights.get(key) {
                if src.shape().dims() != dst.shape().dims() {
                    return Err(Error::InvalidInput(format!(
                        "shape mismatch for '{}': expected {:?}, got {:?}",
                        key,
                        dst.shape().dims(),
                        src.shape().dims()
                    )));
                }
                let cast = if src.dtype() != DType::BF16 {
                    src.to_dtype(DType::BF16)?
                } else {
                    src.clone()
                };
                *dst = cast.requires_grad_(false);
                consumed.insert(key.to_string());
            } else {
                missing.push(key.to_string());
            }
            Ok(())
        }

        // ----- Top-level tensors -----
        copy_linear_weight(&mut self.img_in, "img_in.weight", weights, &mut consumed, &mut missing)?;
        copy_linear_bias(&mut self.img_in, "img_in.bias", weights, &mut consumed, &mut missing)?;
        copy_linear_weight(&mut self.txt_in, "txt_in.weight", weights, &mut consumed, &mut missing)?;
        copy_linear_bias(&mut self.txt_in, "txt_in.bias", weights, &mut consumed, &mut missing)?;

        // txt_norm.0..3.weight — 4 RMSNorm scales of shape [enc_hidden_dim=2880].
        for i in 0..self.config.selected_layer_index.len() {
            let key = format!("txt_norm.{}.weight", i);
            copy_norm_scale(&mut self.txt_norm[i], &key, weights, &mut consumed, &mut missing)?;
        }

        copy_linear_weight(
            &mut self.time_embed_linear_1,
            "time_text_embed.timestep_embedder.linear_1.weight",
            weights,
            &mut consumed,
            &mut missing,
        )?;
        copy_linear_bias(
            &mut self.time_embed_linear_1,
            "time_text_embed.timestep_embedder.linear_1.bias",
            weights,
            &mut consumed,
            &mut missing,
        )?;
        copy_linear_weight(
            &mut self.time_embed_linear_2,
            "time_text_embed.timestep_embedder.linear_2.weight",
            weights,
            &mut consumed,
            &mut missing,
        )?;
        copy_linear_bias(
            &mut self.time_embed_linear_2,
            "time_text_embed.timestep_embedder.linear_2.bias",
            weights,
            &mut consumed,
            &mut missing,
        )?;

        copy_linear_weight(
            &mut self.norm_out_linear,
            "norm_out.linear.weight",
            weights,
            &mut consumed,
            &mut missing,
        )?;
        copy_linear_bias(
            &mut self.norm_out_linear,
            "norm_out.linear.bias",
            weights,
            &mut consumed,
            &mut missing,
        )?;

        copy_linear_weight(&mut self.proj_out, "proj_out.weight", weights, &mut consumed, &mut missing)?;
        copy_linear_bias(&mut self.proj_out, "proj_out.bias", weights, &mut consumed, &mut missing)?;

        // ----- Per-block tensors (48 blocks for default config) -----
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("transformer_blocks.{}.", i);

            // Attention QKV — fused Linears, bias=true.
            copy_linear_weight(&mut block.img_qkv, &format!("{p}attn.img_qkv.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_bias(&mut block.img_qkv, &format!("{p}attn.img_qkv.bias"), weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.txt_qkv, &format!("{p}attn.txt_qkv.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_bias(&mut block.txt_qkv, &format!("{p}attn.txt_qkv.bias"), weights, &mut consumed, &mut missing)?;

            // Per-head RMSNorm scales (shape [head_dim=64]).
            copy_norm_scale(&mut block.norm_q,        &format!("{p}attn.norm_q.weight"),        weights, &mut consumed, &mut missing)?;
            copy_norm_scale(&mut block.norm_k,        &format!("{p}attn.norm_k.weight"),        weights, &mut consumed, &mut missing)?;
            copy_norm_scale(&mut block.norm_added_q,  &format!("{p}attn.norm_added_q.weight"),  weights, &mut consumed, &mut missing)?;
            copy_norm_scale(&mut block.norm_added_k,  &format!("{p}attn.norm_added_k.weight"),  weights, &mut consumed, &mut missing)?;

            // Output projections — to_out.0 (img) + to_add_out (txt).
            copy_linear_weight(&mut block.img_out, &format!("{p}attn.to_out.0.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_bias  (&mut block.img_out, &format!("{p}attn.to_out.0.bias"),   weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.txt_out, &format!("{p}attn.to_add_out.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_bias  (&mut block.txt_out, &format!("{p}attn.to_add_out.bias"),   weights, &mut consumed, &mut missing)?;

            // Modulation Linears (Sequential[SiLU, Linear]; weight is under `.1.`).
            copy_linear_weight(&mut block.img_mod, &format!("{p}img_mod.1.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_bias  (&mut block.img_mod, &format!("{p}img_mod.1.bias"),   weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.txt_mod, &format!("{p}txt_mod.1.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_bias  (&mut block.txt_mod, &format!("{p}txt_mod.1.bias"),   weights, &mut consumed, &mut missing)?;

            // Block RMSNorm scales [inner_dim=1536].
            copy_norm_scale(&mut block.img_norm1, &format!("{p}img_norm1.weight"), weights, &mut consumed, &mut missing)?;
            copy_norm_scale(&mut block.img_norm2, &format!("{p}img_norm2.weight"), weights, &mut consumed, &mut missing)?;
            copy_norm_scale(&mut block.txt_norm1, &format!("{p}txt_norm1.weight"), weights, &mut consumed, &mut missing)?;
            copy_norm_scale(&mut block.txt_norm2, &format!("{p}txt_norm2.weight"), weights, &mut consumed, &mut missing)?;

            // SwiGLU MLP — bias=false for all three Linears, per BUILD_PLAN.
            copy_linear_weight(&mut block.img_mlp_w1, &format!("{p}img_mlp.w1.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.img_mlp_w2, &format!("{p}img_mlp.w2.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.img_mlp_w3, &format!("{p}img_mlp.w3.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.txt_mlp_w1, &format!("{p}txt_mlp.w1.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.txt_mlp_w2, &format!("{p}txt_mlp.w2.weight"), weights, &mut consumed, &mut missing)?;
            copy_linear_weight(&mut block.txt_mlp_w3, &format!("{p}txt_mlp.w3.weight"), weights, &mut consumed, &mut missing)?;
        }

        // ----- Strictness: enumerate missing-expected and unmatched-extra -----
        let mut extra: Vec<String> = weights
            .keys()
            .filter(|k| !consumed.contains(*k))
            .cloned()
            .collect();
        extra.sort();
        missing.sort();
        if !missing.is_empty() || !extra.is_empty() {
            // Truncate each list for readability while keeping count.
            let trim = |xs: &Vec<String>, n: usize| -> String {
                let sample: Vec<String> = xs.iter().take(n).cloned().collect();
                let mut s = sample.join(", ");
                if xs.len() > n {
                    s.push_str(&format!(", ... ({} total)", xs.len()));
                }
                s
            };
            return Err(Error::InvalidInput(format!(
                "Lens load_weights mismatch: {} missing (expected but absent), \
                 {} extra (present but not consumed).\n\
                 missing[{}]: [{}]\n\
                 extra  [{}]: [{}]",
                missing.len(),
                extra.len(),
                missing.len(),
                trim(&missing, 8),
                extra.len(),
                trim(&extra, 8),
            )));
        }

        Ok(())
    }

    /// Single denoising-step forward pass.
    ///
    /// Inputs (all BF16 except mask):
    ///   * `hidden_states`: `[B, S_img, 128]`
    ///   * `encoder_hidden_states`: 4 tensors, each `[B, S_txt, 2880]`
    ///   * `encoder_hidden_states_mask`: `[B, S_txt]` (bool)
    ///   * `timestep`: `[B]` in `[0, 1]`
    ///   * `img_shapes`: `(frame=1, h_lat, w_lat)` — derives `S_img = h_lat * w_lat`
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &[Tensor],
        encoder_hidden_states_mask: &Tensor,
        timestep: &Tensor,
        img_shapes: (usize, usize, usize),
    ) -> Result<Tensor> {
        // ----- Input shape validation (mirrors transformer.py:460-503) -----
        let hs_dims = hidden_states.shape().dims().to_vec();
        if hs_dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "hidden_states must be rank-3 [B, S_img, in_channels], got {hs_dims:?}"
            )));
        }
        let (bsz, img_len, in_ch) = (hs_dims[0], hs_dims[1], hs_dims[2]);
        if in_ch != self.config.in_channels {
            return Err(Error::InvalidInput(format!(
                "hidden_states channels {in_ch} != config.in_channels {}",
                self.config.in_channels
            )));
        }

        let (frame, h_lat, w_lat) = img_shapes;
        if frame != 1 {
            return Err(Error::InvalidInput(format!(
                "Lens DiT supports frame=1 only, got frame={frame}"
            )));
        }
        if h_lat * w_lat != img_len {
            return Err(Error::InvalidInput(format!(
                "img_shapes={img_shapes:?} (h*w = {}) does not match img_len={img_len}",
                h_lat * w_lat
            )));
        }

        // Multi-layer encoder feature check.
        if self.config.multi_layer_encoder_feature {
            if encoder_hidden_states.len() != self.config.selected_layer_index.len() {
                return Err(Error::InvalidInput(format!(
                    "Expected {} text feature layers, got {}",
                    self.config.selected_layer_index.len(),
                    encoder_hidden_states.len()
                )));
            }
            let layer0_dims = encoder_hidden_states[0].shape().dims().to_vec();
            if layer0_dims.len() != 3 {
                return Err(Error::InvalidInput(format!(
                    "encoder_hidden_states[0] must be rank-3, got {layer0_dims:?}"
                )));
            }
            let txt_seq_len = layer0_dims[1];
            for (i, feat) in encoder_hidden_states.iter().enumerate() {
                let d = feat.shape().dims();
                if d.len() != 3 || d[0] != bsz || d[1] != txt_seq_len || d[2] != self.config.enc_hidden_dim {
                    return Err(Error::InvalidInput(format!(
                        "encoder_hidden_states[{i}] expected [B={bsz}, S_txt={txt_seq_len}, {}], got {d:?}",
                        self.config.enc_hidden_dim
                    )));
                }
            }
            let mask_dims = encoder_hidden_states_mask.shape().dims();
            if mask_dims != &[bsz, txt_seq_len][..] {
                return Err(Error::InvalidInput(format!(
                    "encoder_hidden_states_mask shape {mask_dims:?} != [{bsz}, {txt_seq_len}]"
                )));
            }
        } else {
            // Single-layer path is dead in default Lens config.
            return Err(Error::InvalidInput(
                "single-layer encoder feature path is not implemented; Lens uses multi_layer_encoder_feature=true".into(),
            ));
        }

        // Timestep shape check.
        let t_dims = timestep.shape().dims();
        if t_dims != &[bsz][..] {
            return Err(Error::InvalidInput(format!(
                "timestep shape {t_dims:?} != [B={bsz}]"
            )));
        }

        // Skeptic F3 — mask dtype validation. Python coerces to bool before
        // building the additive mask; our Rust path consumes the mask as a
        // BF16 keep-mask `[B, S_txt]` with 1.0 = valid, 0.0 = padded (so we
        // can broadcast through `Tensor::cat` and reach `sdpa::forward`'s
        // keep-mask convention). Accept BF16/F32; reject anything else.
        match encoder_hidden_states_mask.dtype() {
            DType::BF16 | DType::F32 => {}
            other => {
                return Err(Error::InvalidInput(format!(
                    "encoder_hidden_states_mask dtype {other:?} unsupported; \
                     expected BF16 or F32 keep-mask (1.0 valid, 0.0 padded). \
                     Convert from bool at the caller (Python-style additive masks \
                     are NOT accepted — flame-core's sdpa applies (1-mask)*-inf)."
                )));
            }
        }

        let txt_seq_len = encoder_hidden_states[0].shape().dims()[1];
        let device = hidden_states.device();
        let dim = self.config.inner_dim;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;

        // ---- Build joint attention mask [B, 1, 1, S_img + S_txt] ----
        // Image positions always valid (1.0); text positions follow `mask`.
        // Result is a BF16 keep-mask consumed by `sdpa::forward`.
        let mask_bf16 = if encoder_hidden_states_mask.dtype() == DType::BF16 {
            encoder_hidden_states_mask.clone()
        } else {
            encoder_hidden_states_mask.to_dtype(DType::BF16)?
        };
        // [B, S_txt] → [B, 1, 1, S_txt]
        let txt_mask_4d = mask_bf16
            .reshape(&[bsz, 1, 1, txt_seq_len])?;
        // Build a [B, 1, 1, S_img] of ones via from_vec + broadcast.
        let img_ones_data: Vec<f32> = vec![1.0; img_len];
        let img_ones = Tensor::from_vec(
            img_ones_data,
            Shape::from_dims(&[1, 1, 1, img_len]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
        .broadcast_to(&Shape::from_dims(&[bsz, 1, 1, img_len]))?;
        // F4 application: if txt_seq_len == 0 the cat below would produce
        // exactly the img_ones tensor; we let `Tensor::cat` handle the
        // degenerate case (it materializes inputs first).
        let attention_mask = if txt_seq_len == 0 {
            img_ones.contiguous()?
        } else {
            Tensor::cat(&[&img_ones, &txt_mask_4d], 3)?
        };

        // ---- img_in projection: [B, S_img, 128] → [B, S_img, 1536] ----
        let mut h = self.img_in.forward(hidden_states)?;

        // ---- Text feature stack: per-layer RMSNorm + cat + project ----
        // multi_layer_encoder_feature=True path only (asserted above).
        let mut normed_layers: Vec<Tensor> = Vec::with_capacity(encoder_hidden_states.len());
        for (i, feat) in encoder_hidden_states.iter().enumerate() {
            let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(
                feat,
                Some(&self.txt_norm[i]),
                1e-5,
            )?;
            normed_layers.push(normed);
        }
        let normed_refs: Vec<&Tensor> = normed_layers.iter().collect();
        let txt_cat = Tensor::cat(&normed_refs, 2)?; // [B, S_txt, 4*2880=11520]
        let mut e = self.txt_in.forward(&txt_cat)?;     // [B, S_txt, 1536]

        // ---- Timestep embedding ----
        //   proj = get_timestep_embedding(t, 256, flip_sin_to_cos=True, ds=0, scale=1000)
        //   temb = linear_2(silu(linear_1(proj.to(bf16))))
        let t_proj = timestep_embedding(timestep, 256, DType::BF16, device)?;
        let t_h1 = self.time_embed_linear_1.forward(&t_proj)?;
        let t_h1 = t_h1.silu()?;
        let temb = self.time_embed_linear_2.forward(&t_h1)?; // [B, 1536]

        // ---- RoPE tables ----
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.freqs_for(h_lat, w_lat, txt_seq_len, device)?;

        // ---- Block loop: 48 dual-stream LensTransformerBlocks ----
        for block in &self.blocks {
            let (new_e, new_h) = block.forward(
                &h,
                &e,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                &attention_mask,
                num_heads,
                head_dim,
            )?;
            h = new_h;
            e = new_e;
        }

        // ---- AdaLayerNormContinuous final norm + proj_out ----
        let h_final =
            ada_layer_norm_continuous(&h, &temb, &self.norm_out_linear, dim, 1e-6)?;
        // proj_out: 1536 → 128 (patch² * out_channels = 4 * 32)
        let out = self.proj_out.forward(&h_final)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_invariants() {
        let cfg = LensDiTConfig::default();
        assert_eq!(cfg.num_layers, 48);
        let axes_sum: usize = cfg.axes_dims_rope.iter().sum();
        assert_eq!(axes_sum, cfg.attention_head_dim);
        assert_eq!(cfg.inner_dim, cfg.attention_head_dim * cfg.num_attention_heads);
        // 2880 * 4 = 11520
        assert_eq!(cfg.txt_in_dim(), 11520);
        assert_eq!(cfg.mlp_hidden(), 4096);
        assert!(cfg.gate_mlp);
        assert!(cfg.rms_norm);
        assert!(cfg.multi_layer_encoder_feature);
    }

    #[test]
    fn rope_table_shape() {
        let rope = LensEmbedRope::new(10000.0, [8, 28, 28], true).expect("rope build");
        assert_eq!(rope.half_dim(), 32);
        assert_eq!(rope.pos_cos_host.len(), ROPE_TABLE_ROWS * 32);
        assert_eq!(rope.pos_sin_host.len(), ROPE_TABLE_ROWS * 32);
        assert_eq!(rope.neg_cos_host.len(), ROPE_TABLE_ROWS * 32);
        assert_eq!(rope.neg_sin_host.len(), ROPE_TABLE_ROWS * 32);
    }

    #[test]
    fn rope_table_first_row() {
        // pos_index[0] = 0 → every freq · 0 = 0 → cos=1, sin=0 for every column.
        let rope = LensEmbedRope::new(10000.0, [8, 28, 28], true).expect("rope build");
        for k in 0..32 {
            let c = rope.pos_cos_host[k];
            let s = rope.pos_sin_host[k];
            assert!((c - 1.0).abs() < 1e-6, "pos_cos[0, {k}] = {c}, expected 1.0");
            assert!(s.abs() < 1e-6, "pos_sin[0, {k}] = {s}, expected 0.0");
        }
    }

    #[test]
    fn rope_table_neg_index() {
        // neg_index[4095] = -1. Axis 0 (dim=8) column 0 has base = theta^(0/8) = 1.0,
        // so the freq multiplier is 1.0 → arg = -1.0.
        let rope = LensEmbedRope::new(10000.0, [8, 28, 28], true).expect("rope build");
        let last_row_off = (ROPE_TABLE_ROWS - 1) * 32;
        let c = rope.neg_cos_host[last_row_off + 0];
        let s = rope.neg_sin_host[last_row_off + 0];
        let want_c = (-1.0f64).cos() as f32;
        let want_s = (-1.0f64).sin() as f32;
        assert!(
            (c - want_c).abs() < 1e-6,
            "neg_cos[4095, 0] = {c}, expected {want_c}"
        );
        assert!(
            (s - want_s).abs() < 1e-6,
            "neg_sin[4095, 0] = {s}, expected {want_s}"
        );
    }
}
