//! Wan 2.1 3D Video VAE Encoder — pure flame_core.
//!
//! Encode-only CausalConv3d VAE matching the Wan 2.1 checkpoint.
//! Input: `[B, 3, T, H, W]` RGB video/images (BF16, in [-1, 1]).
//! Output: `[B, 16, T, H/8, W/8]` normalized latents (BF16).
//!
//! Architecture (dim=96, z_dim=16, dim_mult=[1,2,4,4]):
//!   encoder.conv1: CausalConv3d(3, 96, 3x3x3)
//!   encoder.downsamples: 11 flat blocks (8 ResBlocks + 3 Resamples)
//!     0,1: ResBlock(96->96), no shortcut
//!     2:   Resample downsample2d(96)
//!     3:   ResBlock(96->192), with shortcut
//!     4:   ResBlock(192->192), no shortcut
//!     5:   Resample downsample3d(192), with time_conv
//!     6:   ResBlock(192->384), with shortcut
//!     7:   ResBlock(384->384), no shortcut
//!     8:   Resample downsample3d(384), with time_conv
//!     9,10: ResBlock(384->384), no shortcut
//!   encoder.middle: ResBlock(384) + AttentionBlock(384) + ResBlock(384)
//!   encoder.head: RMS_norm(384) + SiLU + CausalConv3d(384, 32, 3x3x3)
//!   conv1: CausalConv3d(32, 32, 1x1x1) (top-level)
//!   -> chunk first 16 channels (mu), normalize
//!
//! Weight key format:
//!   encoder.conv1.weight/bias
//!   encoder.downsamples.{n}.residual.{0,3}.gamma  (RMS_norm)
//!   encoder.downsamples.{n}.residual.{2,6}.weight/bias  (CausalConv3d)
//!   encoder.downsamples.{n}.shortcut.weight/bias  (when in_dim != out_dim)
//!   encoder.downsamples.{n}.resample.1.weight/bias  (Conv2d in Resample)
//!   encoder.downsamples.{n}.time_conv.weight/bias  (CausalConv3d in downsample3d)
//!   encoder.middle.{0,2}.residual.*
//!   encoder.middle.1.norm.gamma, to_qkv.weight/bias, proj.weight/bias
//!   encoder.head.0.gamma, encoder.head.2.weight/bias
//!   conv1.weight/bias (top-level 1x1 conv)

use flame_core::conv::Conv2d;
use flame_core::conv3d_simple::Conv3d;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Per-channel normalization constants (same as decoder wan21_vae.rs)
// ---------------------------------------------------------------------------

const MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
];

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Wan21 VAE encoder: missing weight: {key}")))
}

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d — uses Conv3d (F32 internally), same as wan21_vae.rs decoder.
// ---------------------------------------------------------------------------

struct CausalConv3d {
    conv: Conv3d,
    time_pad: usize,
}

impl CausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        pad: (usize, usize, usize),
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let time_pad = 2 * pad.0;
        let mut conv = Conv3d::new(
            in_ch,
            out_ch,
            kernel,
            Some(stride),
            Some((0, pad.1, pad.2)),
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
            let first_frame = x.narrow(2, 0, 1)?;
            let repeated = first_frame.repeat_axis_device(2, self.time_pad)?;
            Tensor::cat(&[&repeated, x], 2)?
        } else {
            x.clone()
        };
        let is_bf16 = x_padded.dtype() == DType::BF16;
        let input = if is_bf16 { x_padded.to_dtype(DType::F32)? } else { x_padded };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }

    /// Forward using raw conv (no causal padding). Used for downsample3d
    /// time_conv where we manually provide the frames.
    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let is_bf16 = x.dtype() == DType::BF16;
        let input = if is_bf16 { x.to_dtype(DType::F32)? } else { x.clone() };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }
}

// ---------------------------------------------------------------------------
// RMS_norm (channel_first=True, images=False -> gamma shape [dim, 1, 1, 1])
// ---------------------------------------------------------------------------

struct RmsNorm5d {
    gamma: Tensor,
    scale: f32,
}

impl RmsNorm5d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        let gamma = get_bf16(weights, &format!("{prefix}.gamma"))?;
        Ok(Self {
            gamma,
            scale: (dim as f32).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_sq = x_f32.mul(&x_f32)?;
        let sum_sq = x_sq.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let normalized = x_f32.div(&norm)?;
        let scaled = normalized.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        scaled.mul(&self.gamma)
    }
}

// ---------------------------------------------------------------------------
// RMS_norm for 2D (channel_first=True, images=True -> gamma shape [dim, 1, 1])
// ---------------------------------------------------------------------------

struct RmsNorm4d {
    gamma: Tensor,
    scale: f32,
}

impl RmsNorm4d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        let gamma = get_bf16(weights, &format!("{prefix}.gamma"))?;
        Ok(Self {
            gamma,
            scale: (dim as f32).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_sq = x_f32.mul(&x_f32)?;
        let sum_sq = x_sq.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let normalized = x_f32.div(&norm)?;
        let scaled = normalized.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        scaled.mul(&self.gamma)
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
        let norm1 = RmsNorm5d::load(weights, &format!("{prefix}.residual.0"), in_dim)?;
        let conv1 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.2"),
            in_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;
        let norm2 = RmsNorm5d::load(weights, &format!("{prefix}.residual.3"), out_dim)?;
        let conv2 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.6"),
            out_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                weights,
                &format!("{prefix}.shortcut"),
                in_dim,
                out_dim,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = if let Some(ref s) = self.shortcut {
            s.forward(x)?
        } else {
            x.clone()
        };
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
// AttentionBlock — per-frame 2D self-attention, single head
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
        let identity = x;

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

        let attn_out = sdpa_forward(&q, &k, &v, None)?;

        let attn_out = attn_out
            .squeeze(Some(1))?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, c, h, w])?;

        let out = self.proj.forward(&attn_out)?;

        let out = out
            .reshape(&[b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;

        identity.add(&out)
    }
}

// ---------------------------------------------------------------------------
// DownResample — downsample2d or downsample3d
// ---------------------------------------------------------------------------

enum DownResample {
    Downsample2d { conv: Conv2d },
    Downsample3d { conv: Conv2d, time_conv: CausalConv3d },
}

impl DownResample {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // resample = [ZeroPad2d(0,1,0,1), Conv2d(dim, dim, 3, stride=2)]
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 2, 0, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        Ok(DownResample::Downsample2d { conv })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 2, 0, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;

        // time_conv: CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1), padding=(0,0,0))
        let time_conv = CausalConv3d::load(
            weights,
            &format!("{prefix}.time_conv"),
            dim,
            dim,
            (3, 1, 1),
            (2, 1, 1),
            (0, 0, 0),
            device,
        )?;

        Ok(DownResample::Downsample3d { conv, time_conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DownResample::Downsample2d { conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            DownResample::Downsample3d { conv, time_conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // Spatial downsample first
                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                let x_5d = x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])?;

                // Temporal downsample — match Python's chunked cache behavior:
                //   Frame 0: pass through without time_conv
                //   Frames 1+: apply time_conv to all frames (no zero padding)
                let first_frame = x_5d.narrow(2, 0, 1)?;

                if t == 1 {
                    Ok(first_frame)
                } else {
                    // Apply time_conv raw (no causal padding) to all t frames.
                    // time_conv kernel=(3,1,1), stride=(2,1,1), padding=(0,0,0).
                    // Input: t frames -> output: (t-3)/2 + 1 = (t-1)/2 frames
                    let rest_frames = time_conv.forward_raw(&x_5d)?;
                    Tensor::cat(&[&first_frame, &rest_frames], 2)
                }
            }
        }
    }

    /// ZeroPad2d(0, 1, 0, 1) — pad right by 1, pad bottom by 1.
    fn zero_pad2d_right_bottom(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        let col_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, h, 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        let x_wpad = Tensor::cat(&[x, &col_pad], 3)?;

        let row_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, 1, w + 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        Tensor::cat(&[&x_wpad, &row_pad], 2)
    }
}

// ---------------------------------------------------------------------------
// Encoder block — either a ResidualBlock or a DownResample
// ---------------------------------------------------------------------------

enum EncoderBlock {
    Res(ResidualBlock),
    Downsample(DownResample),
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            EncoderBlock::Res(r) => r.forward(x),
            EncoderBlock::Downsample(d) => d.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Wan21VaeEncoder
// ---------------------------------------------------------------------------

pub struct Wan21VaeEncoder {
    /// Per-channel mean `[1, 16, 1, 1, 1]` BF16.
    mean: Tensor,
    /// Per-channel 1/std `[1, 16, 1, 1, 1]` BF16.
    inv_std: Tensor,

    /// `encoder.conv1`: CausalConv3d(3, 96, 3x3x3, pad=1).
    encoder_conv1: CausalConv3d,

    /// 11 flat downsample blocks (ResBlocks + Resamples).
    downsamples: Vec<EncoderBlock>,

    /// Middle: ResBlock(384) + AttentionBlock(384) + ResBlock(384).
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,

    /// Head: RMS_norm(384) + SiLU + CausalConv3d(384, 32, 3x3x3).
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,

    /// Top-level conv1: CausalConv3d(32, 32, 1x1x1).
    conv1: CausalConv3d,
}

impl Wan21VaeEncoder {
    /// Load encoder from safetensors file.
    pub fn load(path: &str, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let weights = load_file(path, device)?;
        println!(
            "[Wan21 VAE Encoder] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        Self::from_weights(&weights, device)
    }

    fn from_weights(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let z_dim: usize = 16;

        // Build mean and inv_std tensors for normalization
        let mean = Tensor::from_vec(
            MEAN.to_vec(),
            Shape::from_dims(&[1, z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let inv_std_vals: Vec<f32> = STD.iter().map(|s| 1.0 / s).collect();
        let inv_std = Tensor::from_vec(
            inv_std_vals,
            Shape::from_dims(&[1, z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // encoder.conv1: CausalConv3d(3, 96, 3, pad=1)
        let encoder_conv1 = CausalConv3d::load(
            weights, "encoder.conv1", 3, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;

        // Build flat downsample blocks.
        //
        // Python Encoder3d with dim=96, dim_mult=[1,2,4,4], num_res_blocks=2,
        // temperal_downsample=[False, True, True]:
        //   dims = [96, 96, 192, 384, 384]
        //   For each (in_dim, out_dim) pair with index i:
        //     2 ResidualBlocks (first: in_dim->out_dim, second: out_dim->out_dim)
        //     If i != 3 (last): Resample block
        //
        // Flat block indices:
        //   0: ResBlock(96, 96)      1: ResBlock(96, 96)      2: Resample downsample2d(96)
        //   3: ResBlock(96, 192)     4: ResBlock(192, 192)    5: Resample downsample3d(192)
        //   6: ResBlock(192, 384)    7: ResBlock(384, 384)    8: Resample downsample3d(384)
        //   9: ResBlock(384, 384)   10: ResBlock(384, 384)
        let dim_mult: [usize; 4] = [1, 2, 4, 4];
        let enc_dim: usize = 96;
        let dims: Vec<usize> = {
            let mut d = vec![enc_dim];
            for &m in &dim_mult {
                d.push(enc_dim * m);
            }
            d
        }; // [96, 96, 192, 384, 384]
        let temperal_downsample = [false, true, true];

        let mut blocks: Vec<EncoderBlock> = Vec::new();
        let mut idx = 0usize;

        for i in 0..4 {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];

            // First ResBlock: in_dim -> out_dim
            blocks.push(EncoderBlock::Res(ResidualBlock::load(
                weights,
                &format!("encoder.downsamples.{idx}"),
                in_dim,
                out_dim,
                device,
            )?));
            idx += 1;

            // Second ResBlock: out_dim -> out_dim
            blocks.push(EncoderBlock::Res(ResidualBlock::load(
                weights,
                &format!("encoder.downsamples.{idx}"),
                out_dim,
                out_dim,
                device,
            )?));
            idx += 1;

            // Resample (not for last group)
            if i != dim_mult.len() - 1 {
                let t_down = temperal_downsample[i];
                if t_down {
                    blocks.push(EncoderBlock::Downsample(DownResample::load_3d(
                        weights,
                        &format!("encoder.downsamples.{idx}"),
                        out_dim,
                        device,
                    )?));
                } else {
                    blocks.push(EncoderBlock::Downsample(DownResample::load_2d(
                        weights,
                        &format!("encoder.downsamples.{idx}"),
                        out_dim,
                        device,
                    )?));
                }
                idx += 1;
            }
        }

        // Middle block: top_dim = 384
        let top_dim = *dims.last().unwrap(); // 384
        let mid_res0 = ResidualBlock::load(weights, "encoder.middle.0", top_dim, top_dim, device)?;
        let mid_attn = AttentionBlock::load(weights, "encoder.middle.1", top_dim, device)?;
        let mid_res1 = ResidualBlock::load(weights, "encoder.middle.2", top_dim, top_dim, device)?;

        // Head: RMS_norm(384) + SiLU + CausalConv3d(384, 32, 3, pad=1)
        let head_norm = RmsNorm5d::load(weights, "encoder.head.0", top_dim)?;
        let head_conv = CausalConv3d::load(
            weights, "encoder.head.2", top_dim, z_dim * 2, (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;

        // Top-level conv1: CausalConv3d(32, 32, 1, pad=0)
        let conv1 = CausalConv3d::load(
            weights, "conv1", z_dim * 2, z_dim * 2, (1, 1, 1), (1, 1, 1), (0, 0, 0), device,
        )?;

        Ok(Self {
            mean,
            inv_std,
            encoder_conv1,
            downsamples: blocks,
            mid_res0,
            mid_attn,
            mid_res1,
            head_norm,
            head_conv,
            conv1,
        })
    }

    /// Encode video/image to normalized latents.
    ///
    /// Input: `[B, 3, T, H, W]` BF16 RGB in `[-1, 1]`.
    /// Output: `[B, 16, T', H/8, W/8]` BF16 normalized latents.
    ///
    /// No patchify — Wan 2.1 processes raw RGB directly.
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let dbg = std::env::var("WAN_VAE_DBG").is_ok();

        // 1. encoder.conv1
        let mut x = self.encoder_conv1.forward(video)?;
        if dbg { Self::dbg_stats("after encoder_conv1", &x); }

        // 2. Downsample blocks
        for (i, block) in self.downsamples.iter().enumerate() {
            x = block.forward(&x)?;
            if dbg { Self::dbg_stats(&format!("after downsample[{i}]"), &x); }
        }

        // 3. Middle block
        x = self.mid_res0.forward(&x)?;
        if dbg { Self::dbg_stats("after mid_res0", &x); }
        x = self.mid_attn.forward(&x)?;
        if dbg { Self::dbg_stats("after mid_attn", &x); }
        x = self.mid_res1.forward(&x)?;
        if dbg { Self::dbg_stats("after mid_res1", &x); }

        // 4. Head: RMS_norm + SiLU + CausalConv3d
        x = self.head_norm.forward(&x)?;
        if dbg { Self::dbg_stats("after head_norm", &x); }
        x = x.silu()?;
        if dbg { Self::dbg_stats("after head_silu", &x); }
        x = self.head_conv.forward(&x)?;
        if dbg { Self::dbg_stats("after head_conv", &x); }

        // 5. Top-level conv1 (1x1) -> chunk into mu/log_var
        let out = self.conv1.forward(&x)?;
        if dbg { Self::dbg_stats("after conv1 (top-level)", &out); }

        // Chunk: first 16 channels = mu, discard log_var
        let mu = out.narrow(1, 0, 16)?;
        if dbg { Self::dbg_stats("mu (before normalize)", &mu); }

        // 6. Normalize: z = (mu - mean) * inv_std = (mu - mean) / std
        //    Compute in F32 for numerical stability.
        let mu_f32 = mu.to_dtype(DType::F32)?;
        let mean_f32 = self.mean.to_dtype(DType::F32)?;
        let inv_std_f32 = self.inv_std.to_dtype(DType::F32)?;
        let z = mu_f32.sub(&mean_f32)?.mul(&inv_std_f32)?.to_dtype(DType::BF16)?;

        Ok(z)
    }

    fn dbg_stats(name: &str, t: &Tensor) {
        if let Ok(data) = t.to_dtype(DType::F32).and_then(|t| t.to_vec1::<f32>()) {
            let n = data.len();
            let mean: f32 = data.iter().copied().sum::<f32>() / n as f32;
            let abs_mean: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / n as f32;
            let lo = data.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
            let hi = data.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
            let nan = data.iter().filter(|v| v.is_nan()).count();
            eprintln!(
                "  [DBG {name}] shape={:?} mean={mean:.4} |mean|={abs_mean:.4} range=[{lo:.4}, {hi:.4}] nan={nan}",
                t.shape().dims()
            );
        }
    }
}
