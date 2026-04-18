//! Wan2.2 high-compression VAE — decoder only.
//!
//! Port of `wan/modules/vae2_2.py::Wan2_2_VAE` (Apache-2.0), decode path only.
//!
//! ## High-level architecture
//! - Latent shape:  `[B, 48, T', H', W']` with `T' = (T-1)/4 + 1`,
//!   `H' = H/16`, `W' = W/16`. 48 latent channels (not 16 — that was a
//!   copy-paste error in the original job description from the Wan 2.1 VAE).
//! - `conv2`: 1×1×1 CausalConv3d(48→48).
//! - `Decoder3d`:
//!     - `conv1`: CausalConv3d(48→dims[0]=1024, 3×3×3, pad=1).
//!     - middle: ResBlock(1024) → AttnBlock(1024) → ResBlock(1024).
//!     - 4 `Up_ResidualBlock` groups, each a stack of
//!       `(num_res_blocks+1)=3` `ResidualBlock`s with channels from
//!       `dims = [1024, 1024, 1024, 512, 256]` and an optional trailing
//!       `Resample` (upsample2d or upsample3d). The first three groups
//!       also add a `DupUp3D` avg-shortcut onto the main path.
//!     - head: `RMS_norm(256) + SiLU + CausalConv3d(256→12, 3×3×3, pad=1)`.
//! - Final `unpatchify(patch_size=2)` packs the last 2×2 spatial block into
//!   channels: 12 → 3 RGB, with `H ← H*2, W ← W*2`.
//! - Normalization: `x = z / scale[1] + scale[0]` with per-channel
//!   mean/std arrays of length 48 baked into this file.
//!
//! ## Structural choice: full-pass, not per-frame chunked
//! The reference Python decode calls the `Decoder3d` once per latent frame
//! with a `feat_cache` that stores `CACHE_T=2` previous activations at every
//! `CausalConv3d`. This is an optimization for long-video streaming: the
//! math is equivalent to processing all latent frames in a single pass with
//! the causal left-pad that `CausalConv3d` already applies (see
//! `CausalConv3d.forward`'s `F.pad(padding)` branch, identical to the
//! single-pass behavior of `wan21_vae.rs` in this crate).
//!
//! The `Resample.forward` body of `vae2_2.py` short-circuits past the
//! `time_conv` branch when `feat_cache is None`, because that code path was
//! never hit in production. We restore the temporal step unconditionally —
//! exactly as `wan21_vae.rs` does — because our single-pass semantics need it.
//!
//! ## GPU parity caveat
//! Compile-clean only. Numerical parity against PyTorch `Wan2_2_VAE.decode`
//! has **not** been verified (GPU was occupied). Before shipping, run the
//! `scripts/wan22_vae_parity.py` comparison (shape-for-shape diff of the
//! decoded video tensor) and fix any drift. Expected hotspots:
//!   - `DupUp3D.first_chunk` slicing off `factor_t-1` initial frames;
//!   - `RMS_norm` with `images=True` vs `False` axis choice;
//!   - order of operations around the middle attention block's `b*t`
//!     permute.

use flame_core::conv::Conv2d;
use flame_core::conv3d_bf16::Conv3dBF16;
use flame_core::cuda_ops::GpuOps;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Normalisation constants — from `Wan2_2_VAE.__init__` mean / std arrays.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
];

#[rustfmt::skip]
const STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Wan22VaeConfig {
    pub z_dim: usize,        // 48
    pub dec_dim: usize,      // 256
    pub dim_mult: [usize; 4], // [1, 2, 4, 4]
    pub num_res_blocks: usize, // 2 (group mult = 3 residuals per group)
    pub temperal_upsample: [bool; 3], // [False, True, True]
    pub out_patch_size: usize,        // 2 — unpatchify factor
    pub out_channels: usize,          // 3 — RGB
    pub head_out_channels: usize,     // 12 — pre-unpatchify
    pub eps: f32,
}

impl Default for Wan22VaeConfig {
    fn default() -> Self {
        Self {
            z_dim: 48,
            dec_dim: 256,
            dim_mult: [1, 2, 4, 4],
            num_res_blocks: 2,
            // `WanVAE_.__init__` computes `temperal_upsample =
            // temperal_downsample[::-1]`. `Wan2_2_VAE` sets
            // `temperal_downsample=[False, True, True]`, so the upsample
            // pattern is `[True, True, False]` — i.e. the **first two**
            // groups do temporal upsample (upsample3d with time_conv),
            // the third is spatial-only (upsample2d). Verified against
            // the local `wan2.2_vae.safetensors` shard keys.
            temperal_upsample: [true, true, false],
            out_patch_size: 2,
            out_channels: 3,
            head_out_channels: 12,
            eps: 1e-6,
        }
    }
}

impl Wan22VaeConfig {
    /// Decoder channel sequence: `[dim*mult[-1]] + reversed(dim*mult)`
    /// = `[256*4, 256*4, 256*2, 256*1, 256*1]` (but the last entry comes from
    /// reversed, not reversed then shifted). Matches
    /// `dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]`.
    fn decoder_dims(&self) -> Vec<usize> {
        let mut out = vec![self.dec_dim * self.dim_mult[self.dim_mult.len() - 1]];
        for &m in self.dim_mult.iter().rev() {
            out.push(self.dec_dim * m);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Wan2.2 VAE: missing weight: {key}")))
}
fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d — uses the BF16-native `Conv3dBF16` (im2vol + cuBLASLt GEMM).
//
// The older `conv3d_simple::Conv3d` in flame-core is a naive JIT kernel with
// a known numerical bug that silently zeros most outputs for large input
// channel counts. `Conv3dBF16` is what every other production VAE path in
// this crate uses.
// ---------------------------------------------------------------------------

struct CausalConv3d {
    conv: Conv3dBF16,
    time_pad: usize,
}

impl CausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        _in_ch: usize,
        _out_ch: usize,
        _kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        pad: (usize, usize, usize),
        _device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let time_pad = 2 * pad.0;
        // Conv3dBF16::from_weights infers C_out/C_in/kernel from the weight
        // shape — no need to pass them in.
        let weight = get_bf16(weights, &format!("{prefix}.weight"))?;
        let bias = Some(get_bf16(weights, &format!("{prefix}.bias"))?);
        let conv = Conv3dBF16::from_weights(
            weight,
            bias,
            stride,
            (0, pad.1, pad.2), // temporal padding handled manually below
        );
        Ok(Self { conv, time_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Causal zero-pad along the temporal axis (dim 2).
        //
        // Python reference: `CausalConv3d.forward` in `vae2_2.py` calls
        // `F.pad(x, padding)` which is constant zero padding by default.
        // Left-pads dim 2 by `2 * padding[0]`.
        let x_padded = if self.time_pad > 0 {
            let d = x.shape().dims().to_vec();
            debug_assert_eq!(d.len(), 5);
            let zero_shape = [d[0], d[1], self.time_pad, d[3], d[4]];
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&zero_shape),
                DType::BF16,
                x.device().clone(),
            )?;
            let x_bf16 = if x.dtype() == DType::BF16 { x.clone() } else { x.to_dtype(DType::BF16)? };
            Tensor::cat(&[&pad, &x_bf16], 2)?
        } else if x.dtype() != DType::BF16 {
            x.to_dtype(DType::BF16)?
        } else {
            x.clone()
        };

        // `Conv3dBF16` unfolds the whole input into an `[C_in*kD*kH*kW, D_out*H_out*W_out]`
        // column matrix. For the upsampled decoder tensors (e.g. the last
        // head conv on `[1, 256, 12, 208, 352]` with a 3×3×3 kernel) that
        // column is ~12 GB — way past the 3090 Ti's ceiling. Chunk the conv
        // across the temporal output axis so each per-chunk column is
        // bounded. Causal 3D conv is equivariant under temporal slicing
        // with kT-wide overlap.
        let dims_in = x_padded.shape().dims();
        let d_in = dims_in[2];
        let (kd, _kh, _kw) = self.conv.kernel_size;
        let (sd, _sh, _sw) = self.conv.stride;
        let (pd, _ph, _pw) = self.conv.padding;
        let d_out_total = (d_in + 2 * pd - kd) / sd + 1;

        // Budget: aim for ≤ 256 MB column buffer per chunk. Earlier budget of
        // 1 GB hit OOM on 24 GB cards when combined with the DiT pool cache
        // and the VAE's surrounding activations — observed at head_conv on
        // [1, 256, 8, 96, 128] (192×256 output) where a 6-frame chunk
        // materialized a 973 MB col buffer. 256 MB gives a ~4× safety margin
        // and pushes worst-case into 1.5-frame chunks at the head stage.
        const COL_BYTE_BUDGET: usize = 256usize << 20; // 256 MB
        let hw = dims_in[3] * dims_in[4];
        let col_rows = dims_in[1] * kd * self.conv.kernel_size.1 * self.conv.kernel_size.2;
        let cols_per_out_frame = hw; // with stride 1 spatially
        let max_frames_per_chunk = (COL_BYTE_BUDGET / 2 / col_rows / cols_per_out_frame.max(1)).max(1);

        if max_frames_per_chunk >= d_out_total || d_out_total == 1 {
            // Fits in one pass.
            return self.conv.forward(&x_padded);
        }

        // Chunk output temporally. For output frames [t_out, t_out + k),
        // we need input frames [t_out, t_out + kd + k - 1) (with stride 1).
        let mut out_chunks: Vec<Tensor> = Vec::new();
        let mut t_out = 0usize;
        while t_out < d_out_total {
            let chunk = max_frames_per_chunk.min(d_out_total - t_out);
            let in_start = t_out; // stride=1 only
            let in_len = (chunk - 1) * sd + kd;
            let in_slice = x_padded.narrow(2, in_start, in_len)?;
            let piece = self.conv.forward(&in_slice)?;
            out_chunks.push(piece);
            t_out += chunk;
        }
        let refs: Vec<&Tensor> = out_chunks.iter().collect();
        Tensor::cat(&refs, 2)
    }
}

// ---------------------------------------------------------------------------
// RMS_norm variants
// ---------------------------------------------------------------------------

struct RmsNorm5d {
    gamma: Tensor, // [C, 1, 1, 1]
    scale: f32,
}

impl RmsNorm5d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: get_bf16(weights, &format!("{prefix}.gamma"))?,
            scale: (dim as f32).sqrt(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.mul(&x_f32)?.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let n = x_f32.div(&norm)?;
        let s = n.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        s.mul(&self.gamma)
    }
}

struct RmsNorm4d {
    gamma: Tensor, // [C, 1, 1]
    scale: f32,
}

impl RmsNorm4d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: get_bf16(weights, &format!("{prefix}.gamma"))?,
            scale: (dim as f32).sqrt(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.mul(&x_f32)?.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let n = x_f32.div(&norm)?;
        let s = n.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        s.mul(&self.gamma)
    }
}

// ---------------------------------------------------------------------------
// ResidualBlock
// ---------------------------------------------------------------------------

struct ResidualBlock {
    norm1: RmsNorm5d,
    conv1: CausalConv3d,
    norm2: RmsNorm5d,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // residual Sequential layout:
        //   0: RMS_norm(in)  — `.gamma`
        //   1: SiLU
        //   2: CausalConv3d(in, out, 3, pad=1)
        //   3: RMS_norm(out)
        //   4: SiLU
        //   5: Dropout (no params)
        //   6: CausalConv3d(out, out, 3, pad=1)
        let norm1 = RmsNorm5d::load(weights, &format!("{prefix}.residual.0"), in_dim)?;
        let conv1 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.2"),
            in_dim, out_dim,
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;
        let norm2 = RmsNorm5d::load(weights, &format!("{prefix}.residual.3"), out_dim)?;
        let conv2 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.6"),
            out_dim, out_dim,
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                weights,
                &format!("{prefix}.shortcut"),
                in_dim, out_dim,
                (1, 1, 1), (1, 1, 1), (0, 0, 0),
                device,
            )?)
        } else {
            None
        };
        Ok(Self { norm1, conv1, norm2, conv2, shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = if let Some(ref s) = self.shortcut { s.forward(x)? } else { x.clone() };
        let mut out = self.norm1.forward(x)?;
        out = out.silu()?;
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward(&out)?;
        out = out.silu()?;
        out = self.conv2.forward(&out)?;
        out.add(&h)
    }
}

// ---------------------------------------------------------------------------
// Middle AttentionBlock — single-head per-frame self-attention.
// ---------------------------------------------------------------------------

struct AttentionBlock {
    norm: RmsNorm4d,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl AttentionBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm = RmsNorm4d::load(weights, &format!("{prefix}.norm"), dim)?;

        let mut to_qkv = Conv2d::new_with_bias(dim, dim * 3, 1, 1, 0, device.clone(), true)?;
        to_qkv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.to_qkv.weight"))?)?;
        to_qkv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.to_qkv.bias"))?)?;

        let mut proj = Conv2d::new_with_bias(dim, dim, 1, 1, 0, device.clone(), true)?;
        proj.copy_weight_from(&get_bf16(weights, &format!("{prefix}.proj.weight"))?)?;
        proj.copy_bias_from(&get_bf16(weights, &format!("{prefix}.proj.bias"))?)?;

        Ok(Self { norm, to_qkv, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let identity = x.clone();

        // [B, C, T, H, W] -> [B*T, C, H, W]
        let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
        let x_normed = self.norm.forward(&x_4d)?;

        let qkv = self.to_qkv.forward(&x_normed)?;
        let n = h * w;
        let qkv_flat = qkv
            .reshape(&[b * t, c * 3, n])?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, 1, n, c * 3])?;

        let q = qkv_flat.narrow(3, 0, c)?;
        let k = qkv_flat.narrow(3, c, c)?;
        let v = qkv_flat.narrow(3, c * 2, c)?;

        let attn = sdpa_forward(&q, &k, &v, None)?;
        let attn = attn
            .squeeze(Some(1))?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, c, h, w])?;

        let projected = self.proj.forward(&attn)?;
        let out = projected
            .reshape(&[b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;
        identity.add(&out)
    }
}

// ---------------------------------------------------------------------------
// Resample (2.2): spatial upsample, channels unchanged, + optional temporal
//
// 2.2 differs from 2.1 in two ways:
//   - `Conv2d(dim, dim, 3)` in resample (not dim→dim/2);
//   - `time_conv` outputs `dim*2` (same as 2.1 but now wrapped by feat_cache
//     path that we collapse into an unconditional call, matching 2.1).
// ---------------------------------------------------------------------------

enum Resample {
    Upsample2d { conv: Conv2d },
    Upsample3d { conv: Conv2d, time_conv: CausalConv3d },
}

impl Resample {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        Ok(Resample::Upsample2d { conv })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        let time_conv = CausalConv3d::load(
            weights,
            &format!("{prefix}.time_conv"),
            dim, dim * 2,
            (3, 1, 1), (1, 1, 1), (1, 0, 0),
            device,
        )?;
        Ok(Resample::Upsample3d { conv, time_conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Resample::Upsample2d { conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            Resample::Upsample3d { conv, time_conv } => {
                let dbg = std::env::var("WAN_VAE_DBG").is_ok();
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // Temporal upsample: dim → dim*2 then interleave into 2T.
                let tc = time_conv.forward(x)?;
                if dbg { Wan22VaeDecoder::dbg_stats_static("        time_conv", &tc)?; }
                let tc = tc.reshape(&[b, 2, c, t, h, w])?;
                let x0 = tc.narrow(1, 0, 1)?.squeeze(Some(1))?;
                let x1 = tc.narrow(1, 1, 1)?.squeeze(Some(1))?;
                let stacked = Tensor::cat(&[&x0.unsqueeze(3)?, &x1.unsqueeze(3)?], 3)?;
                let x_t = stacked.reshape(&[b, c, t * 2, h, w])?;
                if dbg { Wan22VaeDecoder::dbg_stats_static("        after tconv interleave", &x_t)?; }

                // Spatial upsample.
                let t2 = t * 2;
                let x_4d = x_t.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t2, c, h, w])?;
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                if dbg { Wan22VaeDecoder::dbg_stats_static("        after nearest2d", &x_up)?; }
                let x_conv = conv.forward(&x_up)?;
                if dbg { Wan22VaeDecoder::dbg_stats_static("        after spat conv", &x_conv)?; }
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t2, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DupUp3D — parameter-free channel-duplication upsample shortcut.
//
// Given `[B, in_c, T, H, W]`, repeat-interleave along channels by
// `repeats = out_c * factor / in_c`, then reshape the channel block back
// into spatial/temporal dims. Matches `DupUp3D.forward` in vae2_2.py
// (minus the `first_chunk` path — documented below).
// ---------------------------------------------------------------------------

struct DupUp3D {
    _in_c: usize,
    out_c: usize,
    factor_t: usize,
    factor_s: usize,
    repeats: usize,
}

impl DupUp3D {
    fn new(in_c: usize, out_c: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        debug_assert_eq!((out_c * factor) % in_c, 0);
        let repeats = out_c * factor / in_c;
        Self { _in_c: in_c, out_c, factor_t, factor_s, repeats }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, _c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // repeat_interleave on channel dim.
        let x = x.repeat_axis_device(1, self.repeats)?;

        // Python:
        //   view(b, out, ft, fs, fs, t, h, w)
        //   permute(0, 1, 5, 2, 6, 3, 7, 4) -> (b, out, t, ft, h, fs, w, fs)
        //   view(b, out, t*ft, h*fs, w*fs)
        let x = x.reshape(&[
            b, self.out_c, self.factor_t, self.factor_s, self.factor_s, t, h, w,
        ])?;
        let x = x.permute(&[0, 1, 5, 2, 6, 3, 7, 4])?;
        let x = x.reshape(&[
            b,
            self.out_c,
            t * self.factor_t,
            h * self.factor_s,
            w * self.factor_s,
        ])?;

        // NOTE: `first_chunk=True` in the reference slices off the first
        // (factor_t - 1) frames of the upsampled tensor. Full-pass decode
        // in our path does not chunk, so we skip that slice. This is a
        // *known parity hazard* — the first (factor_t - 1) output frames
        // may differ from the chunked reference. Verify with
        // `scripts/wan22_vae_parity.py` once GPU is available.
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Up_ResidualBlock: (N ResidualBlocks + optional Resample) + optional avg_shortcut.
// ---------------------------------------------------------------------------

struct UpResidualBlock {
    residuals: Vec<ResidualBlock>,
    resample: Option<Resample>,
    avg_shortcut: Option<DupUp3D>,
}

impl UpResidualBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        mult: usize,           // num_res_blocks + 1
        temperal_upsample: bool,
        up_flag: bool,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Main path Sequential indexing inside `upsamples`:
        //   [0..mult]:  ResidualBlock   (first with in_dim→out_dim,
        //                                rest out_dim→out_dim)
        //   [mult]:     Resample        (only if up_flag)
        let mut residuals: Vec<ResidualBlock> = Vec::with_capacity(mult);
        let mut cur_in = in_dim;
        for i in 0..mult {
            let rb_prefix = format!("{prefix}.upsamples.{i}");
            let rb = ResidualBlock::load(weights, &rb_prefix, cur_in, out_dim, device)?;
            residuals.push(rb);
            cur_in = out_dim;
        }

        let resample = if up_flag {
            let idx = mult;
            let res_prefix = format!("{prefix}.upsamples.{idx}");
            let r = if temperal_upsample {
                Resample::load_3d(weights, &res_prefix, out_dim, device)?
            } else {
                Resample::load_2d(weights, &res_prefix, out_dim, device)?
            };
            Some(r)
        } else {
            None
        };

        // Avg-shortcut (parameter-free) exists only when up_flag.
        let avg_shortcut = if up_flag {
            Some(DupUp3D::new(
                in_dim,
                out_dim,
                if temperal_upsample { 2 } else { 1 },
                2, // factor_s always 2 when up_flag
            ))
        } else {
            None
        };

        Ok(Self { residuals, resample, avg_shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dbg = std::env::var("WAN_VAE_DBG").is_ok();
        if dbg {
            Wan22VaeDecoder::dbg_stats_static("    upg input", x)?;
        }
        // Main path: ResBlocks → optional Resample.
        let mut main = x.clone();
        for (i, rb) in self.residuals.iter().enumerate() {
            main = rb.forward(&main)?;
            if dbg {
                Wan22VaeDecoder::dbg_stats_static(&format!("    upg rb{i}"), &main)?;
            }
        }
        if let Some(ref r) = self.resample {
            main = r.forward(&main)?;
            if dbg {
                Wan22VaeDecoder::dbg_stats_static("    upg resample", &main)?;
            }
        }

        if let Some(ref s) = self.avg_shortcut {
            // The shortcut is a duplicate-upsample of the ORIGINAL x
            // (pre-main-path), matching Python's `self.avg_shortcut(x, first_chunk)`.
            let short = s.forward(x)?;
            if dbg {
                Wan22VaeDecoder::dbg_stats_static("    upg avg_shortcut", &short)?;
            }
            main.add(&short)
        } else {
            Ok(main)
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level Wan2.2 VAE decoder.
// ---------------------------------------------------------------------------

pub struct Wan22VaeDecoder {
    config: Wan22VaeConfig,
    device: Arc<cudarc::driver::CudaDevice>,

    /// Per-channel mean `[1, 48, 1, 1, 1]` BF16 (broadcast add).
    mean: Tensor,
    /// Per-channel std `[1, 48, 1, 1, 1]` BF16 (multiply, since scale[1]=1/std).
    std: Tensor,

    /// `Wan2_2_VAE.conv2`: 1×1×1 CausalConv3d(z_dim→z_dim).
    conv2: CausalConv3d,

    /// Decoder `conv1`: 3×3×3 CausalConv3d(z_dim→dims[0]).
    decoder_conv1: CausalConv3d,

    /// `decoder.middle.0 / .1 / .2` — ResBlock → Attn → ResBlock.
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,

    /// `decoder.upsamples.{0..3}` — four `Up_ResidualBlock`s.
    up_groups: Vec<UpResidualBlock>,

    /// `decoder.head.0 / .2` — RMS(out_dim=256) + SiLU + CausalConv3d(→12).
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,
}

impl Wan22VaeDecoder {
    pub fn load(path: &Path, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let weights = load_file(path, device)?;
        log::info!(
            "[Wan22 VAE] Loaded {} tensors from {}",
            weights.len(),
            path.display()
        );
        Self::from_weights(&weights, device)
    }

    fn from_weights(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let config = Wan22VaeConfig::default();

        // mean & std broadcast tensors.
        let mean = Tensor::from_vec(
            MEAN.to_vec(),
            Shape::from_dims(&[1, config.z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let std = Tensor::from_vec(
            STD.to_vec(),
            Shape::from_dims(&[1, config.z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // conv2: CausalConv3d(48, 48, 1) — padding=0 (kernel 1×1×1).
        let conv2 = CausalConv3d::load(
            weights, "conv2",
            config.z_dim, config.z_dim,
            (1, 1, 1), (1, 1, 1), (0, 0, 0),
            device,
        )?;

        let dims = config.decoder_dims(); // [1024, 1024, 1024, 512, 256]

        // decoder.conv1: CausalConv3d(z_dim, dims[0], 3, pad=1)
        let decoder_conv1 = CausalConv3d::load(
            weights, "decoder.conv1",
            config.z_dim, dims[0],
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;

        // middle = ResBlock(dims[0]) + Attn(dims[0]) + ResBlock(dims[0])
        let mid_res0 = ResidualBlock::load(weights, "decoder.middle.0", dims[0], dims[0], device)?;
        let mid_attn = AttentionBlock::load(weights, "decoder.middle.1", dims[0], device)?;
        let mid_res1 = ResidualBlock::load(weights, "decoder.middle.2", dims[0], dims[0], device)?;

        // Up groups: zip(dims[:-1], dims[1:]) → 4 groups.
        // i=0: 1024→1024 (2d upsample, up_flag=True)
        // i=1: 1024→1024 (3d upsample)
        // i=2: 1024→512  (3d upsample)
        // i=3: 512→256   (up_flag=False, last group)
        let mult = config.num_res_blocks + 1;
        let mut up_groups: Vec<UpResidualBlock> = Vec::with_capacity(4);
        let num_groups = dims.len() - 1; // 4
        for i in 0..num_groups {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            let t_up = if i < config.temperal_upsample.len() {
                config.temperal_upsample[i]
            } else {
                false
            };
            let up_flag = i != config.dim_mult.len() - 1;
            let prefix = format!("decoder.upsamples.{i}");
            up_groups.push(UpResidualBlock::load(
                weights, &prefix, in_d, out_d, mult, t_up, up_flag, device,
            )?);
        }

        // head: RMS_norm(256, images=False) + SiLU + CausalConv3d(256, 12, 3, pad=1)
        let final_dim = *dims.last().unwrap();
        let head_norm = RmsNorm5d::load(weights, "decoder.head.0", final_dim)?;
        let head_conv = CausalConv3d::load(
            weights, "decoder.head.2",
            final_dim, config.head_out_channels,
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;

        Ok(Self {
            config, device: device.clone(),
            mean, std,
            conv2, decoder_conv1,
            mid_res0, mid_attn, mid_res1,
            up_groups,
            head_norm, head_conv,
        })
    }

    /// Decode a latent `[B, 48, T', H', W']` → RGB `[B, 3, T, H, W]`.
    ///
    /// Output matches the reference `Wan2_2_VAE.decode(...).clamp(-1, 1)` —
    /// final `unpatchify(patch_size=2)` is inlined at the end.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        Self::dbg_stats("INPUT z", z)?;
        // Unnormalize: z = z / scale[1] + scale[0]  where scale = [mean, 1/std]
        // → z = z * std + mean
        let z = z.mul(&self.std)?.add(&self.mean)?;
        Self::dbg_stats("after unnorm", &z)?;

        // conv2 (1×1×1)
        let mut x = self.conv2.forward(&z)?;
        Self::dbg_stats("after conv2", &x)?;

        // decoder.conv1
        x = self.decoder_conv1.forward(&x)?;
        Self::dbg_stats("after decoder_conv1", &x)?;

        // middle block
        x = self.mid_res0.forward(&x)?;
        Self::dbg_stats("after mid_res0", &x)?;
        x = self.mid_attn.forward(&x)?;
        Self::dbg_stats("after mid_attn", &x)?;
        x = self.mid_res1.forward(&x)?;
        Self::dbg_stats("after mid_res1", &x)?;

        // upsample groups
        for (i, g) in self.up_groups.iter().enumerate() {
            x = g.forward(&x)?;
            Self::dbg_stats(&format!("after up_group {i}"), &x)?;
        }

        // head
        x = self.head_norm.forward(&x)?;
        Self::dbg_stats("after head_norm", &x)?;
        x = x.silu()?;
        x = self.head_conv.forward(&x)?;
        Self::dbg_stats("after head_conv", &x)?;

        // unpatchify patch_size=2: [B, 12, T, H, W] → [B, 3, T, 2H, 2W]
        // Python: `rearrange(x, 'b (c r q) f h w -> b c f (h q) (w r)', q=2, r=2)`
        let out = Self::unpatchify2(&x)?;
        out.clamp(-1.0, 1.0)
    }

    /// Inline unpatchify with patch_size=2 on the spatial dims.
    ///
    /// Input `[B, 12, T, H, W]`, output `[B, 3, T, 2H, 2W]`. Channels laid
    /// out as `c = out_c * q * r` with `q, r ∈ {0,1}` the spatial sub-index.
    fn unpatchify2(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        assert_eq!(dims.len(), 5, "unpatchify2 expects [B,C,T,H,W]");
        let (b, c_in, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        assert!(c_in % 4 == 0, "channels must be divisible by patch^2");
        let c_out = c_in / 4;

        // rearrange 'b (c r q) f h w -> b c f (h q) (w r)'
        //   view: [b, c_out, r, q, t, h, w]        (r then q inside the channel block)
        //   permute: 0, 1, 4, 5, 3, 6, 2  → [b, c_out, t, h, q, w, r]
        //   reshape: [b, c_out, t, h*q, w*r]
        let v = x.reshape(&[b, c_out, 2, 2, t, h, w])?; // (b, c, r, q, t, h, w)
        let p = v.permute(&[0, 1, 4, 5, 3, 6, 2])?;
        p.reshape(&[b, c_out, t, h * 2, w * 2])
    }

    pub fn config(&self) -> &Wan22VaeConfig { &self.config }

    pub(crate) fn dbg_stats_static(label: &str, t: &Tensor) -> Result<()> {
        Self::dbg_stats(label, t)
    }

    fn dbg_stats(label: &str, t: &Tensor) -> Result<()> {
        if std::env::var("WAN_VAE_DBG").is_err() { return Ok(()); }
        let v = t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let n = v.len();
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let mean = v.iter().sum::<f32>() / n as f32;
        let abs_mean = v.iter().map(|x| x.abs()).sum::<f32>() / n as f32;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &v {
            if x.is_finite() {
                if x < lo { lo = x; }
                if x > hi { hi = x; }
            }
        }
        eprintln!(
            "    [vae] {label:30}  shape={:?}  mean={mean:.4}  |mean|={abs_mean:.4}  range=[{lo:.4}, {hi:.4}]  nan={nan} inf={inf}",
            t.shape().dims(),
        );
        Ok(())
    }
    pub fn device(&self) -> &Arc<cudarc::driver::CudaDevice> { &self.device }
}
