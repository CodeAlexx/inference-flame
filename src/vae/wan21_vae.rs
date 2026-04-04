//! Wan 2.1 3D Video VAE Decoder — pure flame_core.
//!
//! Decode-only CausalConv3d VAE matching the Wan 2.1 checkpoint.
//! Input: `[B, 16, T, H, W]` latents (BF16, normalized).
//! Output: `[B, 3, T*4, H*8, W*8]` RGB video frames (BF16, clamped to [-1, 1]).
//!
//! Architecture (dim=96, z_dim=16, dim_mult=[1,2,4,4]):
//!   conv2: CausalConv3d(16, 16, 1x1x1)
//!   conv1: CausalConv3d(16, 384, 3x3x3)
//!   middle: ResBlock(384,384) + AttentionBlock(384) + ResBlock(384,384)
//!   upsamples: 15 blocks (ResBlocks + Resample upsample2d/3d)
//!   head: RMS_norm(96) + SiLU + CausalConv3d(96, 3, 3x3x3)
//!
//! Weight key format:
//!   conv2.weight/bias
//!   decoder.conv1.weight/bias
//!   decoder.middle.{0,2}.residual.{0,3}.gamma  (RMS_norm)
//!   decoder.middle.{0,2}.residual.{2,6}.weight/bias  (CausalConv3d)
//!   decoder.middle.1.norm.gamma, to_qkv.weight/bias, proj.weight/bias
//!   decoder.upsamples.{n}.residual.{0,3}.gamma
//!   decoder.upsamples.{n}.residual.{2,6}.weight/bias
//!   decoder.upsamples.{n}.shortcut.weight/bias  (when in_dim != out_dim)
//!   decoder.upsamples.{n}.resample.1.weight/bias  (Conv2d in Resample)
//!   decoder.upsamples.{n}.time_conv.weight/bias  (CausalConv3d in upsample3d)
//!   decoder.head.0.gamma, decoder.head.2.weight/bias

use flame_core::conv::Conv2d;
use flame_core::conv3d_simple::Conv3d;
use flame_core::cuda_ops::GpuOps;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Per-channel normalization constants (from Python WAN_VAE_MEAN / WAN_VAE_STD)
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
        .ok_or_else(|| Error::InvalidOperation(format!("Wan VAE: missing weight: {key}")))
}

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d
// ---------------------------------------------------------------------------

struct CausalConv3d {
    conv: Conv3d,
    time_pad: usize, // = 2 * padding[0], i.e. left-only temporal pad
    // spatial padding is handled by Conv3d itself
}

impl CausalConv3d {
    /// Load from weights. `kernel` and `pad` are the original (kT, kH, kW) and (pT, pH, pW).
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
        // Causal: temporal pad = 2 * pad_t on left only, spatial = pad_h/pad_w symmetric
        let time_pad = 2 * pad.0;
        let mut conv = Conv3d::new(
            in_ch,
            out_ch,
            kernel,
            Some(stride),
            Some((0, pad.1, pad.2)), // no temporal pad in Conv3d — we prepend manually
            None,
            None,
            true,
            device.clone(),
        )?;
        conv.weight = get_bf16(weights, &format!("{prefix}.weight"))?;
        conv.bias_tensor = Some(get_bf16(weights, &format!("{prefix}.bias"))?);
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
        self.conv.forward(&x_padded)
    }
}

// ---------------------------------------------------------------------------
// RMS_norm (channel_first=True, images=False -> gamma shape [dim, 1, 1, 1])
// ---------------------------------------------------------------------------

struct RmsNorm5d {
    gamma: Tensor, // [C, 1, 1, 1]
    scale: f32,    // dim^0.5
}

impl RmsNorm5d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        let gamma = get_bf16(weights, &format!("{prefix}.gamma"))?;
        Ok(Self {
            gamma,
            scale: (dim as f32).sqrt(),
        })
    }

    /// F.normalize(x, dim=1) * scale * gamma
    /// normalize(x, dim=1) = x / ||x||_2 along channel dim
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T, H, W]
        let x_f32 = x.to_dtype(DType::F32)?;
        // L2 norm along dim=1: sqrt(sum(x^2, dim=1, keepdim=True))
        let x_sq = x_f32.mul(&x_f32)?;
        let sum_sq = x_sq.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let normalized = x_f32.div(&norm)?;
        let scaled = normalized.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        // gamma: [C, 1, 1, 1] — broadcast multiply over [B, C, T, H, W]
        scaled.mul(&self.gamma)
    }
}

// ---------------------------------------------------------------------------
// RMS_norm for 2D (channel_first=True, images=True -> gamma shape [dim, 1, 1])
// ---------------------------------------------------------------------------

struct RmsNorm4d {
    gamma: Tensor, // [C, 1, 1]
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

    /// Same as RmsNorm5d but for [B*T, C, H, W] tensors
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
        // residual: [RMS_norm(0), SiLU(1), CausalConv3d(2), RMS_norm(3), SiLU(4), Dropout(5), CausalConv3d(6)]
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
    to_qkv: Conv2d, // Conv2d(dim, dim*3, 1)
    proj: Conv2d,    // Conv2d(dim, dim, 1)
    _dim: usize,
}

impl AttentionBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm = RmsNorm4d::load(weights, &format!("{prefix}.norm"), dim)?;

        let mut to_qkv = Conv2d::new(dim, dim * 3, 1, 1, 0, device.clone())?;
        to_qkv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.to_qkv.weight"))?)?;
        to_qkv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.to_qkv.bias"))?)?;

        let mut proj = Conv2d::new(dim, dim, 1, 1, 0, device.clone())?;
        proj.copy_weight_from(&get_bf16(weights, &format!("{prefix}.proj.weight"))?)?;
        proj.copy_bias_from(&get_bf16(weights, &format!("{prefix}.proj.bias"))?)?;

        Ok(Self {
            norm,
            to_qkv,
            proj,
            _dim: dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T, H, W]
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let identity = x;

        // Reshape to [B*T, C, H, W]
        let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;

        // RMS norm (4D)
        let x_normed = self.norm.forward(&x_4d)?;

        // to_qkv: [B*T, C*3, H, W]
        let qkv = self.to_qkv.forward(&x_normed)?;

        // Reshape: [B*T, C*3, H, W] -> [B*T, 1, H*W, C*3] -> chunk into Q, K, V
        let n = h * w;
        let qkv_flat = qkv
            .reshape(&[b * t, c * 3, n])?
            .permute(&[0, 2, 1])? // [B*T, H*W, C*3]
            .reshape(&[b * t, 1, n, c * 3])?;

        // Split into 3 chunks along last dim
        let q = qkv_flat.narrow(3, 0, c)?;
        let k = qkv_flat.narrow(3, c, c)?;
        let v = qkv_flat.narrow(3, c * 2, c)?;

        // SDPA: [B*T, 1, N, C]
        let attn_out = sdpa_forward(&q, &k, &v, None)?;

        // Reshape back: [B*T, 1, N, C] -> [B*T, C, H, W]
        let attn_out = attn_out
            .squeeze(Some(1))? // [B*T, N, C]
            .permute(&[0, 2, 1])? // [B*T, C, N]
            .reshape(&[b * t, c, h, w])?;

        // Project
        let out = self.proj.forward(&attn_out)?;

        // Reshape to [B, C, T, H, W]
        let out = out
            .reshape(&[b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;

        // Residual
        identity.add(&out)
    }
}

// ---------------------------------------------------------------------------
// Resample — upsample2d or upsample3d
// ---------------------------------------------------------------------------

enum UpsampleBlock {
    /// upsample2d: nearest 2x + Conv2d(dim, dim//2, 3, pad=1)
    Upsample2d {
        conv: Conv2d,
    },
    /// upsample3d: time_conv (CausalConv3d for temporal 2x) + nearest 2x + Conv2d
    Upsample3d {
        conv: Conv2d,
        time_conv: CausalConv3d,
        _in_dim: usize,
    },
}

impl UpsampleBlock {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // resample.0 = Upsample (no weights), resample.1 = Conv2d(dim, dim//2, 3, pad=1)
        let mut conv = Conv2d::new(dim, dim / 2, 3, 1, 1, device.clone())?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        Ok(UpsampleBlock::Upsample2d { conv })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new(dim, dim / 2, 3, 1, 1, device.clone())?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;

        // time_conv: CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
        let time_conv = CausalConv3d::load(
            weights,
            &format!("{prefix}.time_conv"),
            dim,
            dim * 2,
            (3, 1, 1),
            (1, 1, 1),
            (1, 0, 0),
            device,
        )?;

        Ok(UpsampleBlock::Upsample3d {
            conv,
            time_conv,
            _in_dim: dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            UpsampleBlock::Upsample2d { conv } => {
                // x: [B, C, T, H, W] -> reshape to [B*T, C, H, W], upsample 2x, conv, reshape back
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_up = GpuOps::upsample2d_nearest(&x_4d, (h * 2, w * 2))?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            UpsampleBlock::Upsample3d {
                conv, time_conv, ..
            } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // Temporal upsample via time_conv: output is [B, dim*2, T, H, W]
                let tc_out = time_conv.forward(x)?;
                // Reshape [B, 2, C, T, H, W] then interleave: stack dim 3 -> [B, C, T*2, H, W]
                let tc_out = tc_out.reshape(&[b, 2, c, t, h, w])?;
                // x0 = tc_out[:, 0, :, :, :, :] and x1 = tc_out[:, 1, :, :, :, :]
                let x0 = tc_out.narrow(1, 0, 1)?.squeeze(Some(1))?; // [B, C, T, H, W]
                let x1 = tc_out.narrow(1, 1, 1)?.squeeze(Some(1))?; // [B, C, T, H, W]
                // Interleave along T: [B, C, T*2, H, W]
                // Stack along dim 3: [B, C, T, 2, H, W] then reshape
                let stacked = Tensor::cat(&[
                    &x0.unsqueeze(3)?,
                    &x1.unsqueeze(3)?,
                ], 3)?; // [B, C, T, 2, H, W]
                let x_t = stacked.reshape(&[b, c, t * 2, h, w])?;

                // Spatial upsample: [B*T*2, C, H, W] -> nearest 2x -> Conv2d -> [B, C/2, T*2, H*2, W*2]
                let t2 = t * 2;
                let x_4d = x_t.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t2, c, h, w])?;
                let x_up = GpuOps::upsample2d_nearest(&x_4d, (h * 2, w * 2))?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t2, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder block — either a ResidualBlock or an UpsampleBlock
// ---------------------------------------------------------------------------

enum DecoderBlock {
    Res(ResidualBlock),
    Upsample(UpsampleBlock),
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DecoderBlock::Res(r) => r.forward(x),
            DecoderBlock::Upsample(u) => u.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Wan21VaeDecoder
// ---------------------------------------------------------------------------

pub struct Wan21VaeDecoder {
    // Unnormalization constants (per-channel, [16])
    mean: Tensor,
    inv_std: Tensor,

    // conv2: 1x1 conv on latent
    conv2: CausalConv3d,
    // conv1: z_dim -> top_dim
    conv1: CausalConv3d,
    // middle: ResBlock + Attn + ResBlock
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    // upsamples: flat list of 15 decoder blocks
    upsamples: Vec<DecoderBlock>,
    // head: RMS_norm + SiLU + CausalConv3d
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,
}

impl Wan21VaeDecoder {
    /// Load decoder from safetensors file.
    pub fn load(path: &str, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let weights = load_file(path, device)?;
        println!(
            "[Wan21 VAE] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        Self::from_weights(&weights, device)
    }

    fn from_weights(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let cuda_device: &Arc<cudarc::driver::CudaDevice> = device;

        // Build mean and inv_std tensors for unnormalization
        let mean = Tensor::from_vec(
            MEAN.to_vec(),
            Shape::from_dims(&[1, 16, 1, 1, 1]),
            cuda_device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let inv_std: Vec<f32> = STD.iter().map(|s| 1.0 / s).collect();
        let inv_std = Tensor::from_vec(
            inv_std,
            Shape::from_dims(&[1, 16, 1, 1, 1]),
            cuda_device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // conv2: CausalConv3d(16, 16, 1x1x1) — note: padding=0 for 1x1x1
        let conv2 = CausalConv3d::load(
            weights, "conv2", 16, 16, (1, 1, 1), (1, 1, 1), (0, 0, 0), device,
        )?;

        // conv1: CausalConv3d(16, 384, 3x3x3, padding=1)
        let conv1 = CausalConv3d::load(
            weights,
            "decoder.conv1",
            16,
            384,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;

        // Middle block
        let mid_res0 =
            ResidualBlock::load(weights, "decoder.middle.0", 384, 384, device)?;
        let mid_attn =
            AttentionBlock::load(weights, "decoder.middle.1", 384, device)?;
        let mid_res1 =
            ResidualBlock::load(weights, "decoder.middle.2", 384, 384, device)?;

        // Build upsamples — 15 blocks matching Python Decoder3d.__init__
        // dims = [384, 384, 384, 192, 96], temperal_upsample = [False, True, True]
        //
        // i=0: 3x ResBlock(384,384) + Resample(384, upsample2d)
        // i=1: ResBlock(192,384) + 2x ResBlock(384,384) + Resample(384, upsample3d)
        //      (in_dim halved for i=1,2,3)
        // i=2: 3x ResBlock(192,192) + Resample(192, upsample3d)
        //      (in_dim halved)
        // i=3: ResBlock(96,96) + 2x ResBlock(96,96)
        //      (in_dim halved, no resample for last group)

        let mut blocks: Vec<DecoderBlock> = Vec::new();

        // Block structure with (index, in_ch, out_ch) from Python trace:
        let block_spec = [
            // i=0: upsamples 0,1,2
            (384, 384),
            (384, 384),
            (384, 384),
            // Resample(384, upsample2d) at index 3

            // i=1: upsamples 4,5,6 (in_dim halved: 384//2=192 for first)
            (192, 384),
            (384, 384),
            (384, 384),
            // Resample(384, upsample3d) at index 7

            // i=2: upsamples 8,9,10 (in_dim halved: 384//2=192)
            (192, 192),
            (192, 192),
            (192, 192),
            // Resample(192, upsample3d) at index 11

            // i=3: upsamples 12,13,14 (in_dim halved: 192//2=96)
            (96, 96),
            (96, 96),
            (96, 96),
        ];

        // We interleave ResBlocks and Resample blocks in flat order
        let mut idx = 0;
        for &(in_ch, out_ch) in &block_spec {
            blocks.push(DecoderBlock::Res(ResidualBlock::load(
                weights,
                &format!("decoder.upsamples.{idx}"),
                in_ch,
                out_ch,
                device,
            )?));
            idx += 1;

            // Insert Resample blocks after specific positions
            // After block 2 (idx=3): upsample2d(384)
            // After block 6 (idx=7): upsample3d(384)
            // After block 10 (idx=11): upsample3d(192)
            if idx == 3 {
                blocks.push(DecoderBlock::Upsample(UpsampleBlock::load_2d(
                    weights,
                    &format!("decoder.upsamples.{idx}"),
                    384,
                    device,
                )?));
                idx += 1;
            } else if idx == 7 {
                blocks.push(DecoderBlock::Upsample(UpsampleBlock::load_3d(
                    weights,
                    &format!("decoder.upsamples.{idx}"),
                    384,
                    device,
                )?));
                idx += 1;
            } else if idx == 11 {
                blocks.push(DecoderBlock::Upsample(UpsampleBlock::load_3d(
                    weights,
                    &format!("decoder.upsamples.{idx}"),
                    192,
                    device,
                )?));
                idx += 1;
            }
        }

        // Head: RMS_norm(96, images=False) + SiLU + CausalConv3d(96, 3, 3, pad=1)
        let head_norm = RmsNorm5d::load(weights, "decoder.head.0", 96)?;
        let head_conv = CausalConv3d::load(
            weights,
            "decoder.head.2",
            96,
            3,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;

        Ok(Self {
            mean,
            inv_std,
            conv2,
            conv1,
            mid_res0,
            mid_attn,
            mid_res1,
            upsamples: blocks,
            head_norm,
            head_conv,
        })
    }

    /// Decode latents to video.
    ///
    /// Input: `[B, 16, T, H, W]` BF16 latents (normalized).
    /// Output: `[B, 3, T*4, H*8, W*8]` BF16 RGB video frames clamped to [-1, 1].
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Unnormalize: z = z / (1/std) + mean = z * std + mean
        // But we store inv_std = 1/std, so: z = z / inv_std + mean
        // Python: z = z / scale[1] + scale[0]  where scale = [mean, 1/std]
        let z = z.div(&self.inv_std)?.add(&self.mean)?;

        // conv2: 1x1 conv
        let mut x = self.conv2.forward(&z)?;

        // conv1: z_dim -> 384
        x = self.conv1.forward(&x)?;

        // Middle: ResBlock + Attn + ResBlock
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;

        // Upsamples
        for block in &self.upsamples {
            x = block.forward(&x)?;
        }

        // Head: RMS_norm + SiLU + CausalConv3d
        x = self.head_norm.forward(&x)?;
        x = x.silu()?;
        x = self.head_conv.forward(&x)?;

        // Clamp to [-1, 1]
        x.clamp(-1.0, 1.0)
    }
}
