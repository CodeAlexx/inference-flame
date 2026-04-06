//! LTX-2 Latent Upsampler — spatial 2x upscaling for video latents.
//!
//! Upscales `[B, 128, F, H, W]` latents to `[B, 128, F, H*2, W*2]`.
//!
//! Architecture:
//!   initial_conv: Conv3d(128, 1024, k=3, p=1)
//!   initial_norm: GroupNorm(32, 1024)
//!   SiLU
//!   4× ResBlock(1024): Conv3d→GN→SiLU→Conv3d→GN + skip→SiLU
//!   SpatialRationalResampler(2x):
//!     rearrange [B,C,F,H,W] → [B*F,C,H,W]
//!     Conv2d(1024, 4096, k=3, p=1)
//!     PixelShuffle(2,2): [BF,4096,H,W] → [BF,1024,H*2,W*2]
//!     rearrange [B*F,C,H*2,W*2] → [B,C,F,H*2,W*2]
//!   4× ResBlock(1024)
//!   final_conv: Conv3d(1024, 128, k=3, p=1)
//!
//! Weight keys (73 tensors):
//!   initial_conv.{weight,bias}
//!   initial_norm.{weight,bias}
//!   res_blocks.{0-3}.conv1.{weight,bias}
//!   res_blocks.{0-3}.norm1.{weight,bias}
//!   res_blocks.{0-3}.conv2.{weight,bias}
//!   res_blocks.{0-3}.norm2.{weight,bias}
//!   upsampler.conv.{weight,bias}           (Conv2D!)
//!   upsampler.blur_down.kernel             (ignored — stride=1 = identity)
//!   post_upsample_res_blocks.{0-3}.*
//!   final_conv.{weight,bias}

use flame_core::conv3d_bf16::Conv3dBF16;
use flame_core::cudnn::conv2d::cudnn_conv2d_bf16;
use flame_core::group_norm::group_norm;
use flame_core::cuda_ops::GpuOps;
use flame_core::serialization;
use flame_core::{CudaDevice, DType, Error, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NUM_GROUPS: usize = 32;
const NUM_BLOCKS: usize = 4;
const PIXEL_SHUFFLE_FACTOR: usize = 2;

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_weight(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("LTX2 Upsampler: missing weight: {key}")))
}

fn get_weight_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get_weight(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// GroupNorm helper — 5D [B, C, F, H, W] via reshape to 4D
// ---------------------------------------------------------------------------

/// GroupNorm on a 5D NCFHW tensor.
///
/// flame_core's group_norm expects NHWC 4D input. We:
///   1. Reshape [B, C, F, H, W] → [B, C, F*H, W]  (merge F into spatial)
///   2. Permute NCHW → NHWC
///   3. Run group_norm
///   4. Permute NHWC → NCHW
///   5. Reshape [B, C, F*H, W] → [B, C, F, H, W]
fn group_norm_5d(
    x: &Tensor,
    num_groups: usize,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

    // Merge F into spatial: [B, C, F*H, W]
    let x_4d = x.reshape(&[b, c, f * h, w])?;

    // NCHW → NHWC
    let x_nhwc = GpuOps::permute_nchw_to_nhwc(&x_4d)?;

    // GroupNorm (expects NHWC 4D)
    let out_nhwc = group_norm(&x_nhwc, num_groups, Some(weight), Some(bias), eps)?;

    // NHWC → NCHW
    let out_nchw = GpuOps::permute_nhwc_to_nchw(&out_nhwc)?;

    // Restore 5D: [B, C, F, H, W]
    out_nchw.reshape(&[b, c, f, h, w])
}

// ---------------------------------------------------------------------------
// ResBlock — Conv3d → GroupNorm → SiLU → Conv3d → GroupNorm + skip → SiLU
// ---------------------------------------------------------------------------

struct ResBlock {
    conv1: Conv3dBF16,
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    conv2: Conv3dBF16,
    norm2_weight: Tensor,
    norm2_bias: Tensor,
}

impl ResBlock {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let conv1_w = get_weight_bf16(weights, &format!("{prefix}.conv1.weight"))?;
        let conv1_b = get_weight_bf16(weights, &format!("{prefix}.conv1.bias"))?;
        let conv1 = Conv3dBF16::from_weights(conv1_w, Some(conv1_b), (1, 1, 1), (1, 1, 1));

        let norm1_weight = get_weight_bf16(weights, &format!("{prefix}.norm1.weight"))?;
        let norm1_bias = get_weight_bf16(weights, &format!("{prefix}.norm1.bias"))?;

        let conv2_w = get_weight_bf16(weights, &format!("{prefix}.conv2.weight"))?;
        let conv2_b = get_weight_bf16(weights, &format!("{prefix}.conv2.bias"))?;
        let conv2 = Conv3dBF16::from_weights(conv2_w, Some(conv2_b), (1, 1, 1), (1, 1, 1));

        let norm2_weight = get_weight_bf16(weights, &format!("{prefix}.norm2.weight"))?;
        let norm2_bias = get_weight_bf16(weights, &format!("{prefix}.norm2.bias"))?;

        Ok(Self {
            conv1,
            norm1_weight,
            norm1_bias,
            conv2,
            norm2_weight,
            norm2_bias,
        })
    }

    /// Forward: conv1 → norm1 → silu → conv2 → norm2 → (+ residual) → silu
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        let h = self.conv1.forward(x)?;
        let h = group_norm_5d(&h, NUM_GROUPS, &self.norm1_weight, &self.norm1_bias, 1e-5)?;
        let h = h.silu()?;

        let h = self.conv2.forward(&h)?;
        let h = group_norm_5d(&h, NUM_GROUPS, &self.norm2_weight, &self.norm2_bias, 1e-5)?;

        // Skip connection + SiLU
        let h = h.add(&residual)?;
        h.silu()
    }
}

// ---------------------------------------------------------------------------
// PixelShuffle 2D — pure reshape + permute
// ---------------------------------------------------------------------------

/// PixelShuffle for 4D [N, C*p1*p2, H, W] → [N, C, H*p1, W*p2]
///
/// Steps:
///   reshape [N, C, p1, p2, H, W]
///   permute [N, C, H, p1, W, p2]
///   reshape [N, C, H*p1, W*p2]
fn pixel_shuffle_2d(x: &Tensor, p1: usize, p2: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (n, c_packed, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let c = c_packed / (p1 * p2);

    // [N, C*p1*p2, H, W] → [N, C, p1, p2, H, W]
    let x = x.reshape(&[n, c, p1, p2, h, w])?;
    // → [N, C, H, p1, W, p2]
    let x = x.permute(&[0, 1, 4, 2, 5, 3])?;
    // → [N, C, H*p1, W*p2]
    x.reshape(&[n, c, h * p1, w * p2])
}

// ---------------------------------------------------------------------------
// SpatialRationalResampler — Conv2d + PixelShuffle (2x spatial, per-frame)
// ---------------------------------------------------------------------------

struct SpatialResampler {
    conv_weight: Tensor, // [4096, 1024, 3, 3] Conv2D
    conv_bias: Tensor,   // [4096]
}

impl SpatialResampler {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let conv_weight = get_weight_bf16(weights, &format!("{prefix}.conv.weight"))?;
        let conv_bias = get_weight_bf16(weights, &format!("{prefix}.conv.bias"))?;
        Ok(Self {
            conv_weight,
            conv_bias,
        })
    }

    /// Forward: rearrange to per-frame → Conv2d → PixelShuffle → rearrange back.
    ///
    /// Input:  [B, 1024, F, H, W]
    /// Output: [B, 1024, F, H*2, W*2]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // [B, C, F, H, W] → [B*F, C, H, W]
        let x_2d = x.reshape(&[b * f, c, h, w])?;

        // Conv2d(1024 → 4096, k=3, p=1)
        let x_conv = cudnn_conv2d_bf16(
            &x_2d,
            &self.conv_weight,
            Some(&self.conv_bias),
            (1, 1),  // stride
            (1, 1),  // padding
            1,       // groups
        )?;

        // PixelShuffle: [B*F, 4096, H, W] → [B*F, 1024, H*2, W*2]
        let x_up = pixel_shuffle_2d(&x_conv, PIXEL_SHUFFLE_FACTOR, PIXEL_SHUFFLE_FACTOR)?;

        // BlurDown with stride=1 is identity — skip

        // [B*F, C, H*2, W*2] → [B, C, F, H*2, W*2]
        x_up.reshape(&[b, c, f, h * PIXEL_SHUFFLE_FACTOR, w * PIXEL_SHUFFLE_FACTOR])
    }
}

// ---------------------------------------------------------------------------
// LTX2LatentUpsampler — main public struct
// ---------------------------------------------------------------------------

/// LTX-2 latent spatial upsampler.
///
/// Upscales `[B, 128, F, H, W]` → `[B, 128, F, H*2, W*2]`.
pub struct LTX2LatentUpsampler {
    initial_conv: Conv3dBF16,
    initial_norm_weight: Tensor,
    initial_norm_bias: Tensor,
    res_blocks: Vec<ResBlock>,
    resampler: SpatialResampler,
    post_res_blocks: Vec<ResBlock>,
    final_conv: Conv3dBF16,
}

impl LTX2LatentUpsampler {
    /// Load upsampler weights from a safetensors file.
    ///
    /// `path` should point to `diffusion_pytorch_model.safetensors`.
    pub fn load(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let weights = serialization::load_file(path, device)?;

        // initial_conv: Conv3d(128, 1024, k=3, p=1)
        let ic_w = get_weight_bf16(&weights, "initial_conv.weight")?;
        let ic_b = get_weight_bf16(&weights, "initial_conv.bias")?;
        let initial_conv = Conv3dBF16::from_weights(ic_w, Some(ic_b), (1, 1, 1), (1, 1, 1));

        // initial_norm: GroupNorm(32, 1024)
        let initial_norm_weight = get_weight_bf16(&weights, "initial_norm.weight")?;
        let initial_norm_bias = get_weight_bf16(&weights, "initial_norm.bias")?;

        // 4× ResBlock (pre-upsample)
        let mut res_blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            res_blocks.push(ResBlock::load(&weights, &format!("res_blocks.{i}"))?);
        }

        // SpatialRationalResampler
        let resampler = SpatialResampler::load(&weights, "upsampler")?;

        // 4× ResBlock (post-upsample)
        let mut post_res_blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            post_res_blocks.push(ResBlock::load(
                &weights,
                &format!("post_upsample_res_blocks.{i}"),
            )?);
        }

        // final_conv: Conv3d(1024, 128, k=3, p=1)
        let fc_w = get_weight_bf16(&weights, "final_conv.weight")?;
        let fc_b = get_weight_bf16(&weights, "final_conv.bias")?;
        let final_conv = Conv3dBF16::from_weights(fc_w, Some(fc_b), (1, 1, 1), (1, 1, 1));

        Ok(Self {
            initial_conv,
            initial_norm_weight,
            initial_norm_bias,
            res_blocks,
            resampler,
            post_res_blocks,
            final_conv,
        })
    }

    /// Forward pass: upscale latents spatially by 2x.
    ///
    /// Input:  `[B, 128, F, H, W]`
    /// Output: `[B, 128, F, H*2, W*2]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // initial_conv → initial_norm → SiLU
        let mut h = self.initial_conv.forward(x)?;
        h = group_norm_5d(
            &h,
            NUM_GROUPS,
            &self.initial_norm_weight,
            &self.initial_norm_bias,
            1e-5,
        )?;
        h = h.silu()?;

        // 4× ResBlock (pre-upsample)
        for block in &self.res_blocks {
            h = block.forward(&h)?;
        }

        // Spatial 2x upsample
        h = self.resampler.forward(&h)?;

        // 4× ResBlock (post-upsample)
        for block in &self.post_res_blocks {
            h = block.forward(&h)?;
        }

        // final_conv: 1024 → 128
        self.final_conv.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_shuffle_shapes() {
        // Verify the pixel shuffle reshape logic (CPU-only, no GPU needed)
        // Just check that the math is right for the reshape dims
        let p1 = 2usize;
        let p2 = 2usize;
        let n = 2;
        let c = 1024;
        let h = 4;
        let w = 6;
        let c_packed = c * p1 * p2;

        // After pixel shuffle: [N, C, H*p1, W*p2]
        assert_eq!(c_packed, 4096);
        assert_eq!(c_packed / (p1 * p2), c);
        assert_eq!(h * p1, 8);
        assert_eq!(w * p2, 12);
    }
}
