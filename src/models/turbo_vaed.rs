//! TurboVAED — pure-Rust port of MagiHuman's distill-mode video decoder.
//!
//! Decode-only student VAE for the GAIR/daVinci-MagiHuman pipeline. Maps a
//! `[1, 48, T_lat, H_lat, W_lat]` BF16 latent into a
//! `[1, 3, ((T_lat-1)*4)+1, H_lat*16, W_lat*16]` BF16 video clip in
//! pixel-space, value range roughly `[-1, 1]`.
//!
//! Architecture (from `inference/model/turbo_vaed/turbo_vaed_module.py` +
//! `TurboV3-Wan22-TinyShallow_7_7.json`):
//!     latent_channels = 48 (Wan 2.2 z_dim)
//!     decoder_block_out_channels = (64, 128, 256, 512)  # encoder dir
//!         reversed for decoder = (512, 256, 128, 64)
//!     decoder_layers_per_block = (2, 2, 2, 3, 3)        # encoder dir
//!         reversed = (3, 3, 2, 2, 2)  # mid + 4 up_blocks
//!     decoder_spatio_temporal_scaling = (false, true, true, true)
//!         reversed = (true, true, true, false)          # 3 upsamplers
//!     decoder_spatio_only = (false, true, false, false)
//!         reversed = (false, false, true, false)        # up2 = upsample2d, others 3d
//!     patch_size = 2, use_unpatchify = true             # final 2× spatial via unpatchify
//!     spatial_compression_ratio = 16, temporal_compression_ratio = 4
//!     first_chunk_size = 7, step_size = 7               # sliding window
//!
//! Differences from `vae/wan21_vae.rs` (all important):
//! 1. CausalConv3d here is **symmetric** replicate-pad on both temporal sides
//!    (`(kT-1)/2` left and right). Wan 2.1 only pads left.
//! 2. RMSNorm here is **variance-style** (`x.pow(2).mean(dim=1)`) and has
//!    NO learnable weight (`elementwise_affine=False` per config).
//! 3. ResnetBlock has an `is_upsampler_modified` flag: when true the FIRST
//!    nonlinearity becomes ReLU instead of SiLU. Set on every resnet that
//!    follows an upsampler.
//! 4. Each up_block's upsampler honors an `is_first_chunk` flag — it drops
//!    the first temporal frame after time_conv when set, so chunked decode
//!    can keep a 1-latent-frame overlap.

use flame_core::conv::Conv2d;
use flame_core::conv3d_simple::Conv3d;
use flame_core::cuda_ops::GpuOps;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ===========================================================================
// Hardcoded latent normalization constants
// ===========================================================================
//
// Per-channel mean and std for the 48-dim Wan 2.2 latent space. These are
// HARDCODED in the Python module (`turbo_vaed_module.py` lines 803-911) —
// the corresponding `latents_mean` / `latents_std` tensors in the
// shipped checkpoint are placeholders (all zeros / ones). The Python
// `__init__` overwrites them with these literals, so we must do the same.

const TURBO_VAED_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
];

const TURBO_VAED_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

// ===========================================================================
// Helpers
// ===========================================================================

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("TurboVAED: missing weight: {key}")))
}

// ===========================================================================
// TurboCausalConv3d — symmetric replicate-pad on both temporal sides
// ===========================================================================

struct TurboCausalConv3d {
    conv: Conv3d,
    time_pad: usize, // (kT - 1) / 2 — applied on BOTH left and right
}

impl TurboCausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Symmetric temporal pad: (kT - 1) / 2 frames replicated on each side.
        // Spatial pad is symmetric and folded into Conv3d's own padding param.
        let time_pad = (kernel.0.saturating_sub(1)) / 2;
        let height_pad = kernel.1 / 2;
        let width_pad = kernel.2 / 2;
        let mut conv = Conv3d::new(
            in_ch,
            out_ch,
            kernel,
            Some(stride),
            Some((0, height_pad, width_pad)), // no temporal pad here — manual replicate below
            None,
            None,
            true,
            device.clone(),
        )?;
        conv.weight = get(weights, &format!("{prefix}.weight"))?.to_dtype(DType::F32)?;
        conv.bias_tensor = Some(get(weights, &format!("{prefix}.bias"))?.to_dtype(DType::F32)?);
        Ok(Self { conv, time_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_padded = if self.time_pad > 0 {
            let first_frame = x.narrow(2, 0, 1)?.contiguous()?;
            let last_idx = x.shape().dims()[2] - 1;
            let last_frame = x.narrow(2, last_idx, 1)?.contiguous()?;
            let pad_left = first_frame.repeat_axis_device(2, self.time_pad)?;
            let pad_right = last_frame.repeat_axis_device(2, self.time_pad)?;
            Tensor::cat(&[&pad_left, x, &pad_right], 2)?.contiguous()?
        } else {
            x.clone()
        };
        // Conv3d kernel uses F32 internally and assumes contiguous BCDHW.
        let is_bf16 = x_padded.dtype() == DType::BF16;
        let input = if is_bf16 {
            x_padded.to_dtype(DType::F32)?
        } else {
            x_padded
        };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }
}

// ===========================================================================
// TurboRmsNorm — variance-style, no learnable weight
// ===========================================================================
//
// Python:
//     variance = x.to(F32).pow(2).mean(dim=1, keepdim=True)
//     x = x * rsqrt(variance + eps)
//     return x.to(input_dtype)        # weight is None when elementwise_affine=False

fn turbo_rms_norm_5d(x: &Tensor, eps: f32) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.mul(&x_f32)?.mean_dim(&[1], true)?;
    // rsqrt(variance + eps) = 1 / sqrt(variance + eps)
    let denom = variance.add_scalar(eps)?.sqrt()?;
    let normed = x_f32.div(&denom)?;
    if in_dtype == DType::F32 {
        Ok(normed)
    } else {
        normed.to_dtype(in_dtype)
    }
}

// ===========================================================================
// TurboResnetBlock3d
// ===========================================================================

struct TurboResnetBlock3d {
    conv1: TurboCausalConv3d,
    conv2: TurboCausalConv3d,
    // optional channel-changing shortcut (1×1×1 conv when in_ch != out_ch)
    conv_shortcut: Option<TurboCausalConv3d>,
    rms_eps: f32,                 // 1e-8 for norm1/norm2 — matches Python
    rms_eps_shortcut: f32,        // resnet_eps when shortcut path
    is_upsampler_modified: bool,  // swap first activation to ReLU
}

impl TurboResnetBlock3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        is_upsampler_modified: bool,
        rms_eps_shortcut: f32,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Python: norm1=RMSNorm(in,1e-8), conv1, norm2=RMSNorm(out,1e-8), conv2,
        //         norm3+conv_shortcut iff in!=out (norm3 uses resnet_eps)
        let conv1 = TurboCausalConv3d::load(
            weights,
            &format!("{prefix}.conv1.conv"),
            in_ch,
            out_ch,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;
        let conv2 = TurboCausalConv3d::load(
            weights,
            &format!("{prefix}.conv2.conv"),
            out_ch,
            out_ch,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;
        let conv_shortcut = if in_ch != out_ch {
            Some(TurboCausalConv3d::load(
                weights,
                &format!("{prefix}.conv_shortcut.conv"),
                in_ch,
                out_ch,
                (1, 1, 1),
                (1, 1, 1),
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            conv_shortcut,
            rms_eps: 1e-8,
            rms_eps_shortcut,
            is_upsampler_modified,
        })
    }

    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // Main path: norm1 → act_first → conv1 → norm2 → silu → conv2
        let h = turbo_rms_norm_5d(inputs, self.rms_eps)?;
        let h = if self.is_upsampler_modified {
            h.relu()?
        } else {
            h.silu()?
        };
        let h = self.conv1.forward(&h)?;
        let h = turbo_rms_norm_5d(&h, self.rms_eps)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;

        // Shortcut path
        let shortcut = if let Some(ref sc) = self.conv_shortcut {
            // Python: norm3(inputs) → conv_shortcut(...). norm3 uses resnet_eps.
            let s = turbo_rms_norm_5d(inputs, self.rms_eps_shortcut)?;
            sc.forward(&s)?
        } else {
            inputs.clone()
        };

        h.add(&shortcut)
    }
}

// ===========================================================================
// WanResample — upsample2d or upsample3d
// ===========================================================================
//
// upsample2d: WanUpsample(nearest-exact, ×2) → Conv2d(dim, out, 3, pad=1)
// upsample3d: time_conv(dim, dim*2, (3,1,1)) → split-and-double-T
//             → optional drop-first-frame-if-first-chunk
//             → WanUpsample(nearest-exact, ×2) → Conv2d(dim, out, 3, pad=1)
//
// In TurboVAED's config, every upsampler keeps `out = dim` (no channel
// halving) — channel changes happen in the up_block's conv_in resnet first.

enum WanResample {
    Upsample2d {
        conv: Conv2d,
        out_dim: usize,
    },
    Upsample3d {
        conv: Conv2d,
        time_conv: TurboCausalConv3d,
        out_dim: usize,
    },
}

impl WanResample {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        upsample_out_dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // resample = Sequential(WanUpsample, Conv2d). Conv2d at index .resample.1
        let mut conv = Conv2d::new_with_bias(dim, upsample_out_dim, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get(weights, &format!("{prefix}.resample.1.weight"))?.to_dtype(DType::BF16)?)?;
        conv.copy_bias_from(&get(weights, &format!("{prefix}.resample.1.bias"))?.to_dtype(DType::BF16)?)?;
        Ok(Self::Upsample2d { conv, out_dim: upsample_out_dim })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        upsample_out_dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, upsample_out_dim, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get(weights, &format!("{prefix}.resample.1.weight"))?.to_dtype(DType::BF16)?)?;
        conv.copy_bias_from(&get(weights, &format!("{prefix}.resample.1.bias"))?.to_dtype(DType::BF16)?)?;
        let time_conv = TurboCausalConv3d::load(
            weights,
            &format!("{prefix}.time_conv.conv"),
            dim,
            dim * 2,
            (3, 1, 1),
            (1, 1, 1),
            device,
        )?;
        Ok(Self::Upsample3d { conv, time_conv, out_dim: upsample_out_dim })
    }

    fn forward(&self, x: &Tensor, is_first_chunk: bool) -> Result<Tensor> {
        // x: [B, C, T, H, W]
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        let (x_t, t_out) = match self {
            WanResample::Upsample2d { .. } => (x.clone(), t),
            WanResample::Upsample3d { time_conv, .. } => {
                // time_conv: [B, C, T, H, W] → [B, 2C, T, H, W]
                let tc_out = time_conv.forward(x)?;
                // Reshape (2, C) outer-then-inner along channel:
                //   [B, 2*C, T, H, W] -> [B, 2, C, T, H, W]
                // Then interleave T-axis: [B, C, T*2, H, W] with order
                //   t0_part0, t0_part1, t1_part0, t1_part1, ...
                let tc_out = tc_out.reshape(&[b, 2, c, t, h, w])?;
                let x0 = tc_out.narrow(1, 0, 1)?.squeeze(Some(1))?; // [B, C, T, H, W]
                let x1 = tc_out.narrow(1, 1, 1)?.squeeze(Some(1))?;
                // Interleave on T: cat along new axis 3 then collapse
                let stacked = Tensor::cat(
                    &[&x0.unsqueeze(3)?, &x1.unsqueeze(3)?],
                    3,
                )?; // [B, C, T, 2, H, W]
                let doubled = stacked.reshape(&[b, c, t * 2, h, w])?;
                // First chunk: drop the very first temporal frame so the
                // overlap with the previous (non-existent) chunk is zero.
                if is_first_chunk {
                    let kept = doubled.narrow(2, 1, t * 2 - 1)?;
                    (kept, t * 2 - 1)
                } else {
                    (doubled, t * 2)
                }
            }
        };

        // Spatial upsample (nearest-exact ≡ nearest at integer factor=2):
        //   permute [B, C, T_out, H, W] → [B, T_out, C, H, W] → [B*T_out, C, H, W]
        //   upsample 2× → conv → permute back
        let x_4d = x_t
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(&[b * t_out, c, h, w])?;
        let x_f32 = x_4d.to_dtype(DType::F32)?;
        let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
        let x_up = x_up.to_dtype(DType::BF16)?;
        let conv_ref = match self {
            WanResample::Upsample2d { conv, .. } => conv,
            WanResample::Upsample3d { conv, .. } => conv,
        };
        let out_dim = match self {
            WanResample::Upsample2d { out_dim, .. } => *out_dim,
            WanResample::Upsample3d { out_dim, .. } => *out_dim,
        };
        let x_conv = conv_ref.forward(&x_up)?;
        x_conv
            .reshape(&[b, t_out, out_dim, h * 2, w * 2])?
            .permute(&[0, 2, 1, 3, 4])
    }
}

// ===========================================================================
// TurboMidBlock3d
// ===========================================================================

struct TurboMidBlock3d {
    resnets: Vec<TurboResnetBlock3d>,
}

impl TurboMidBlock3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        num_layers: usize,
        resnet_eps: f32,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(TurboResnetBlock3d::load(
                weights,
                &format!("{prefix}.resnets.{i}"),
                in_ch,
                in_ch,
                false, // mid-block resnets are NOT upsampler-modified
                resnet_eps,
                device,
            )?);
        }
        Ok(Self { resnets })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

// ===========================================================================
// TurboUpBlock3d
// ===========================================================================

struct TurboUpBlock3d {
    conv_in: Option<TurboResnetBlock3d>,
    upsampler: Option<WanResample>,
    resnets: Vec<TurboResnetBlock3d>,
}

impl TurboUpBlock3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        num_layers: usize,
        spatio_temporal_scale: bool,
        spatio_only: bool,
        resnet_eps: f32,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Python order: optional conv_in (resnet for channel change), then
        // optional upsamplers (single WanResample), then resnets.
        let conv_in = if in_ch != out_ch {
            Some(TurboResnetBlock3d::load(
                weights,
                &format!("{prefix}.conv_in"),
                in_ch,
                out_ch,
                false, // conv_in resnet itself is not upsampler-modified
                resnet_eps,
                device,
            )?)
        } else {
            None
        };

        let upsampler = if spatio_temporal_scale {
            // upsample_out_dim defaults to out_channels (per TurboVAED config).
            let pref = format!("{prefix}.upsamplers.0");
            Some(if spatio_only {
                WanResample::load_2d(weights, &pref, out_ch, out_ch, device)?
            } else {
                WanResample::load_3d(weights, &pref, out_ch, out_ch, device)?
            })
        } else {
            None
        };

        let resnet_modified = upsampler.is_some(); // is_upsampler_modified=spatio_temporal_scale
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(TurboResnetBlock3d::load(
                weights,
                &format!("{prefix}.resnets.{i}"),
                out_ch,
                out_ch,
                resnet_modified,
                resnet_eps,
                device,
            )?);
        }

        Ok(Self {
            conv_in,
            upsampler,
            resnets,
        })
    }

    fn forward(&self, x: &Tensor, is_first_chunk: bool) -> Result<Tensor> {
        let mut h = if let Some(ref ci) = self.conv_in {
            ci.forward(x)?
        } else {
            x.clone()
        };
        if let Some(ref up) = self.upsampler {
            h = up.forward(&h, is_first_chunk)?;
        }
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

// ===========================================================================
// unpatchify — reverse spatial patchification
// ===========================================================================
//
// Python:
//   x: [B, C*p*p, T, H, W] → reshape [B, C, p, p, T, H, W]
//      → permute [0, 1, 4, 5, 3, 6, 2]
//      → [B, C, T, H, p, W, p] → reshape [B, C, T, H*p, W*p]

fn unpatchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    if dims.len() != 5 {
        return Err(Error::InvalidOperation(format!(
            "unpatchify: expected 5d input, got {dims:?}"
        )));
    }
    let (b, c_patches, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let pp = patch_size * patch_size;
    if c_patches % pp != 0 {
        return Err(Error::InvalidOperation(format!(
            "unpatchify: channel {c_patches} not divisible by patch_size² = {pp}"
        )));
    }
    let c = c_patches / pp;

    let reshaped = x.reshape(&[b, c, patch_size, patch_size, t, h, w])?;
    // Python: [b, c, p, p, t, h, w] → permute [0,1,4,5,3,6,2]
    //   axis 0 (b) → 0
    //   axis 1 (c) → 1
    //   axis 4 (t) → 2
    //   axis 5 (h) → 3
    //   axis 3 (p2) → 4   (p_w)
    //   axis 6 (w) → 5
    //   axis 2 (p1) → 6   (p_h)
    let perm = reshaped.permute(&[0, 1, 4, 5, 3, 6, 2])?.contiguous()?;
    perm.reshape(&[b, c, t, h * patch_size, w * patch_size])
}

// ===========================================================================
// TurboDecoder3d
// ===========================================================================

struct TurboDecoder3d {
    conv_in: TurboCausalConv3d,
    mid_block: TurboMidBlock3d,
    up_blocks: Vec<TurboUpBlock3d>,
    conv_out: TurboCausalConv3d,
    use_unpatchify: bool,
    patch_size: usize,
}

impl TurboDecoder3d {
    fn load(
        weights: &Weights,
        cfg: &TurboVaedConfig,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Reverse encoder-direction config to decoder direction.
        let block_out: Vec<usize> = cfg.decoder_block_out_channels.iter().rev().copied().collect();
        let layers_per: Vec<usize> = cfg.decoder_layers_per_block.iter().rev().copied().collect();
        let st_scale: Vec<bool> = cfg.decoder_spatio_temporal_scaling.iter().rev().copied().collect();
        let sp_only: Vec<bool> = cfg.decoder_spatio_only.iter().rev().copied().collect();

        let top_dim = block_out[0]; // 512 for the shipped checkpoint

        let conv_in = TurboCausalConv3d::load(
            weights,
            "decoder.conv_in.conv",
            cfg.latent_channels,
            top_dim,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;

        let mid_block = TurboMidBlock3d::load(
            weights,
            "decoder.mid_block",
            top_dim,
            layers_per[0],
            cfg.resnet_norm_eps,
            device,
        )?;

        // Build 4 up_blocks. Channel transition: top_dim → block_out[1] → ...
        let mut up_blocks = Vec::with_capacity(block_out.len());
        let mut input_ch = top_dim;
        for i in 0..block_out.len() {
            let out_ch = block_out[i];
            up_blocks.push(TurboUpBlock3d::load(
                weights,
                &format!("decoder.up_blocks.{i}"),
                input_ch,
                out_ch,
                layers_per[i + 1],
                st_scale[i],
                sp_only[i],
                cfg.resnet_norm_eps,
                device,
            )?);
            input_ch = out_ch;
        }

        // conv_out: input_ch → out_channels * patch_size * patch_size when use_unpatchify,
        //           else → out_channels.
        let conv_out_ch = if cfg.use_unpatchify && cfg.patch_size >= 2 {
            cfg.out_channels * cfg.patch_size * cfg.patch_size
        } else {
            cfg.out_channels
        };
        let conv_out = TurboCausalConv3d::load(
            weights,
            "decoder.conv_out.conv",
            input_ch,
            conv_out_ch,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_out,
            use_unpatchify: cfg.use_unpatchify,
            patch_size: cfg.patch_size,
        })
    }

    fn forward(&self, z: &Tensor, is_first_chunk: bool) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;
        h = self.mid_block.forward(&h)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h, is_first_chunk)?;
        }

        // Tail: when use_unpatchify, the per-pixel norm_up_1+upsampler2d_1
        // path is skipped. Reference still does an inline RMSNorm + SiLU
        // before conv_out. (See lines 738-743 in turbo_vaed_module.py.)
        h = turbo_rms_norm_5d(&h, 1e-8)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;

        if self.use_unpatchify {
            unpatchify(&h, self.patch_size)
        } else {
            // Non-unpatchify path is not used by the shipped checkpoint;
            // would need TurboVAEDConv2dSplitUpsampler if ever needed.
            Err(Error::InvalidOperation(
                "TurboVAED: use_unpatchify=false branch not implemented".into(),
            ))
        }
    }

    /// Bisect helper: returns the same forward path as `forward` but yields
    /// every intermediate tensor along the way for parity comparison.
    /// Order matches the Python reference dump in `scripts/.../bisect.safetensors`.
    pub fn forward_with_dumps(
        &self,
        z: &Tensor,
        is_first_chunk: bool,
    ) -> Result<Vec<(String, Tensor)>> {
        let mut dumps = Vec::new();
        let h = self.conv_in.forward(z)?;
        dumps.push(("after_conv_in".to_string(), h.clone()));
        let h = self.mid_block.forward(&h)?;
        dumps.push(("after_mid_block".to_string(), h.clone()));
        let mut h = h;
        for (i, ub) in self.up_blocks.iter().enumerate() {
            h = ub.forward(&h, is_first_chunk)?;
            dumps.push((format!("after_up{i}"), h.clone()));
        }
        let h_n = turbo_rms_norm_5d(&h, 1e-8)?;
        dumps.push(("after_inline_norm".to_string(), h_n.clone()));
        let h_s = h_n.silu()?;
        dumps.push(("after_silu".to_string(), h_s.clone()));
        let h_c = self.conv_out.forward(&h_s)?;
        dumps.push(("after_conv_out".to_string(), h_c.clone()));
        if self.use_unpatchify {
            let h_u = unpatchify(&h_c, self.patch_size)?;
            dumps.push(("after_unpatchify".to_string(), h_u));
        }
        Ok(dumps)
    }
}

// ===========================================================================
// TurboVaedConfig
// ===========================================================================

pub struct TurboVaedConfig {
    pub out_channels: usize,
    pub latent_channels: usize,
    pub decoder_block_out_channels: Vec<usize>,
    pub decoder_layers_per_block: Vec<usize>,
    pub decoder_spatio_temporal_scaling: Vec<bool>,
    pub decoder_spatio_only: Vec<bool>,
    pub patch_size: usize,
    pub resnet_norm_eps: f32,
    pub first_chunk_size: usize,
    pub step_size: usize,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
    pub use_unpatchify: bool,
}

impl TurboVaedConfig {
    /// Defaults matching `TurboV3-Wan22-TinyShallow_7_7.json`.
    pub fn shipped_default() -> Self {
        Self {
            out_channels: 3,
            latent_channels: 48,
            decoder_block_out_channels: vec![64, 128, 256, 512],
            decoder_layers_per_block: vec![2, 2, 2, 3, 3],
            decoder_spatio_temporal_scaling: vec![false, true, true, true],
            decoder_spatio_only: vec![false, true, false, false],
            patch_size: 2,
            resnet_norm_eps: 1e-6,
            first_chunk_size: 7,
            step_size: 7,
            spatial_compression_ratio: 16,
            temporal_compression_ratio: 4,
            use_unpatchify: true,
        }
    }
}

// ===========================================================================
// TurboVAED — top level with sliding-window decode
// ===========================================================================

pub struct TurboVAED {
    decoder: TurboDecoder3d,
    mean: Tensor, // [1, z_dim, 1, 1, 1] BF16
    std: Tensor,  // [1, z_dim, 1, 1, 1] BF16
    z_dim: usize,
    first_chunk_size: usize,
    step_size: usize,
    temporal_compression_ratio: usize,
}

impl TurboVAED {
    /// Load from a pre-converted safetensors file (see
    /// `scripts/convert_magihuman_turbo_vaed_to_safetensors.py`). The file
    /// must contain `decoder.*` weights plus `latents_mean` and `latents_std`.
    pub fn load(
        path: &str,
        cfg: &TurboVaedConfig,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let weights = load_file(path, device)?;
        println!(
            "[TurboVAED] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        Self::from_weights(&weights, cfg, device)
    }

    pub fn from_weights(
        weights: &Weights,
        cfg: &TurboVaedConfig,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let decoder = TurboDecoder3d::load(weights, cfg, device)?;

        // Per-channel latent normalization (z = z * std + mean before decode).
        // The checkpoint's `latents_mean` / `latents_std` are placeholders;
        // the real values come from hardcoded Python constants. We construct
        // them in F32 and keep F32 to mirror the reference's mixed-precision
        // path (z_BF16 / scale[1]_F32 → F32 result → cast back to BF16).
        if cfg.latent_channels != TURBO_VAED_MEAN.len() {
            return Err(Error::InvalidOperation(format!(
                "TurboVAED: latent_channels={} but hardcoded mean/std have {} entries",
                cfg.latent_channels,
                TURBO_VAED_MEAN.len()
            )));
        }
        let mean = Tensor::from_vec(
            TURBO_VAED_MEAN.to_vec(),
            Shape::from_dims(&[1, cfg.latent_channels, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let std = Tensor::from_vec(
            TURBO_VAED_STD.to_vec(),
            Shape::from_dims(&[1, cfg.latent_channels, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        Ok(Self {
            decoder,
            mean,
            std,
            z_dim: cfg.latent_channels,
            first_chunk_size: cfg.first_chunk_size,
            step_size: cfg.step_size,
            temporal_compression_ratio: cfg.temporal_compression_ratio,
        })
    }

    /// Decode a `[B, z_dim, T_lat, H_lat, W_lat]` latent into a
    /// `[B, 3, ((T_lat-1) * temporal_ratio) + 1, H_lat * 16, W_lat * 16]`
    /// pixel-space video (BF16, range ≈ [-1, 1]).
    ///
    /// Internally denormalizes the latent (`z * std + mean`) and then runs
    /// the sliding-window decode logic from MagiHuman's reference.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // 1. Denormalize: z = z * std + mean. Both std and mean are [1, C, 1, 1, 1].
        let z_norm = z.mul(&self.std)?.add(&self.mean)?;
        self.sliding_window_decode(&z_norm)
    }

    /// Bisect helper for parity tests: returns latent denorm result + every
    /// intermediate decoder dump. Caller is responsible for slicing the
    /// latent appropriately for is_first_chunk semantics.
    pub fn forward_with_dumps(
        &self,
        z: &Tensor,
        is_first_chunk: bool,
    ) -> Result<Vec<(String, Tensor)>> {
        let z_norm = z.mul(&self.std)?.add(&self.mean)?;
        let mut out = vec![("z_norm".to_string(), z_norm.clone())];
        let inner = self.decoder.forward_with_dumps(&z_norm, is_first_chunk)?;
        out.extend(inner);
        Ok(out)
    }

    fn sliding_window_decode(&self, z: &Tensor) -> Result<Tensor> {
        let dims = z.shape().dims().to_vec();
        let num_frames_orig = dims[2];
        let first_chunk = self.first_chunk_size;
        let step = self.step_size;
        let overlap_pixel = self.temporal_compression_ratio; // 1 latent frame × ratio = pixel overlap

        // Pad temporal axis so num_frames satisfies first_chunk + n*step.
        let mut num_frames = num_frames_orig;
        let mut num_padding_frames = 0usize;
        if num_frames < first_chunk {
            num_padding_frames = first_chunk - num_frames;
        } else if (num_frames - first_chunk) % step != 0 {
            num_padding_frames = step - (num_frames - first_chunk) % step;
        }
        let z_padded = if num_padding_frames > 0 {
            let last = z.narrow(2, num_frames - 1, 1)?;
            let pad = last.repeat_axis_device(2, num_padding_frames)?;
            num_frames += num_padding_frames;
            Tensor::cat(&[z, &pad], 2)?
        } else {
            z.clone()
        };

        // 2. Decode with overlapping windows.
        let mut out_chunks: Vec<Tensor> = Vec::new();

        if num_frames == first_chunk {
            // Single chunk: decode directly with first-chunk semantics.
            eprintln!("[turbovaed] single-chunk path: in_t={}", first_chunk);
            let out = self.decoder.forward(&z_padded, true)?;
            out_chunks.push(out);
        } else {
            eprintln!(
                "[turbovaed] multi-chunk path: total_in_t={num_frames} first_chunk={first_chunk} step={step} overlap_pixel={overlap_pixel}"
            );
            // First chunk: input frames [0..first_chunk + 1) (= first_chunk+1
            // latent frames), then trim right `overlap_pixel` pixel frames.
            let first_in = z_padded.narrow(2, 0, first_chunk + 1)?;
            eprintln!("[turbovaed] first chunk in shape: {:?}", first_in.shape().dims());
            let first_out = self.decoder.forward(&first_in, true)?;
            eprintln!("[turbovaed] first chunk out shape: {:?}", first_out.shape().dims());
            let first_t = first_out.shape().dims()[2];
            if first_t <= overlap_pixel {
                return Err(Error::InvalidOperation(format!(
                    "TurboVAED: first chunk produced {first_t} frames, expected > overlap {overlap_pixel}"
                )));
            }
            let first_out_trimmed = first_out.narrow(2, 0, first_t - overlap_pixel)?;
            out_chunks.push(first_out_trimmed);

            // Middle / last chunks.
            let mut i = first_chunk;
            while i < num_frames {
                let is_last_chunk = i + step == num_frames;
                let left = i - 1;
                let right = if is_last_chunk { i + step } else { i + step + 1 };
                let in_len = right - left;

                let chunk_in = z_padded.narrow(2, left, in_len)?;
                eprintln!(
                    "[turbovaed] chunk i={i} (last={is_last_chunk}) in shape={:?}",
                    chunk_in.shape().dims()
                );
                let chunk_out = self.decoder.forward(&chunk_in, false)?;
                eprintln!("[turbovaed] chunk i={i} out shape={:?}", chunk_out.shape().dims());
                let chunk_t = chunk_out.shape().dims()[2];
                let trimmed = if is_last_chunk {
                    chunk_out.narrow(2, overlap_pixel, chunk_t - overlap_pixel)?
                } else {
                    chunk_out.narrow(
                        2,
                        overlap_pixel,
                        chunk_t - 2 * overlap_pixel,
                    )?
                };
                out_chunks.push(trimmed);

                i += step;
            }
        }

        // 3. Concatenate chunks along temporal axis.
        let chunk_refs: Vec<&Tensor> = out_chunks.iter().collect();
        let mut out = Tensor::cat(&chunk_refs, 2)?;

        // 4. Trim padded pixel frames if we padded the latent input.
        if num_padding_frames > 0 {
            let pixel_pad = num_padding_frames * self.temporal_compression_ratio;
            let total_t = out.shape().dims()[2];
            if pixel_pad >= total_t {
                return Err(Error::InvalidOperation(format!(
                    "TurboVAED: pixel pad {pixel_pad} ≥ output frames {total_t}"
                )));
            }
            out = out.narrow(2, 0, total_t - pixel_pad)?;
        }

        // Suppress unused warnings.
        let _ = self.z_dim;
        Ok(out)
    }
}
