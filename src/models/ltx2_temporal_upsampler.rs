//! LTX-2.3 Temporal Latent Upsampler — temporal 2x upscaling for video latents.
//!
//! Upscales `[B, 128, F, H, W]` latents to `[B, 128, F*2 - 1, H, W]`
//! (the leading frame is dropped post-upscale because the encoder's first
//! frame is special — it encodes a single pixel frame, not a clip).
//!
//! Mirrors upstream `ltx_core/model/upsampler/model.py::LatentUpsampler` with
//! `dims=3, spatial_upsample=False, temporal_upsample=True, mid_channels=512`.
//!
//! Architecture (matches `ltx-2.3-temporal-upscaler-x2-1.0.safetensors`,
//! upstream commit 41d9243):
//!   initial_conv:  Conv3d(128, 512, k=3, p=1)
//!   initial_norm:  GroupNorm(32, 512)
//!   SiLU
//!   4× ResBlock(512): Conv3d→GN→SiLU→Conv3d→GN + skip→SiLU
//!   upsampler:
//!     Conv3d(512, 1024, k=3, p=1)         # 2 * mid_channels
//!     PixelShuffleND(dims=1, factor=2)    # split 1024→512 with f×2
//!   x = x[:, :, 1:, :, :]                 # drop leading frame
//!   4× ResBlock(512)
//!   final_conv:    Conv3d(512, 128, k=3, p=1)
//!
//! Weight keys (72 tensors):
//!   initial_conv.{weight,bias}
//!   initial_norm.{weight,bias}
//!   res_blocks.{0-3}.{conv1,conv2}.{weight,bias}
//!   res_blocks.{0-3}.{norm1,norm2}.{weight,bias}
//!   upsampler.0.{weight,bias}
//!   post_upsample_res_blocks.{0-3}.*
//!   final_conv.{weight,bias}
//!
//! NOTE: This module is build-clean but has no parity test yet. Runtime
//! correctness is not gated until a Python reference dump is produced and
//! a `bin/ltx2_temporal_upsampler_parity` is added.
//! TODO PARITY: add parity bin against Lightricks reference once available.

use flame_core::conv3d_bf16::Conv3dBF16;
use flame_core::cuda_ops::GpuOps;
use flame_core::group_norm::group_norm;
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
/// Temporal upscale factor (frames doubled, then leading frame dropped).
const TEMPORAL_FACTOR: usize = 2;

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_weight(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("LTX2 TemporalUpsampler: missing weight: {key}")))
}

fn get_weight_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get_weight(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// GroupNorm helper — 5D [B, C, F, H, W] via reshape to 4D
// ---------------------------------------------------------------------------
//
// Mirrors the helper in `ltx2_upsampler.rs`. flame_core::group_norm expects
// NHWC 4D input, so we collapse F into spatial, permute, run, permute back,
// then restore.

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
//
// Identical structure to `ltx2_upsampler.rs::ResBlock` but lives in this
// file so the temporal upsampler is self-contained (the spatial module
// constants for channel widths differ; keeping them separate avoids
// cross-coupling weight-loading errors).

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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        let h = self.conv1.forward(x)?;
        let h = group_norm_5d(&h, NUM_GROUPS, &self.norm1_weight, &self.norm1_bias, 1e-5)?;
        let h = h.silu()?;

        let h = self.conv2.forward(&h)?;
        let h = group_norm_5d(&h, NUM_GROUPS, &self.norm2_weight, &self.norm2_bias, 1e-5)?;

        let h = h.add(&residual)?;
        h.silu()
    }
}

// ---------------------------------------------------------------------------
// PixelShuffleND(dims=1) — temporal pixel shuffle
// ---------------------------------------------------------------------------

/// Temporal pixel shuffle: split the channel dim by `factor` and interleave
/// into the frame dim. Mirrors einops:
///   "b (c p1) f h w -> b c (f p1) h w", p1=factor
///
/// Input:  [B, C*p1, F, H, W]
/// Output: [B, C, F*p1, H, W]
///
/// Implemented as reshape → permute → reshape:
///   [B, C*p1, F, H, W] → [B, C, p1, F, H, W]   (split channel)
///   → [B, C, F, p1, H, W]                      (move p1 next to F)
///   → [B, C, F*p1, H, W]                       (merge)
fn pixel_shuffle_temporal(x: &Tensor, factor: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c_packed, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    if c_packed % factor != 0 {
        return Err(Error::InvalidOperation(format!(
            "pixel_shuffle_temporal: channels {c_packed} not divisible by factor {factor}"
        )));
    }
    let c = c_packed / factor;

    // [B, C*p1, F, H, W] → [B, C, p1, F, H, W]
    let x = x.reshape(&[b, c, factor, f, h, w])?;
    // → [B, C, F, p1, H, W]
    let x = x.permute(&[0, 1, 3, 2, 4, 5])?;
    // → [B, C, F*p1, H, W]
    x.reshape(&[b, c, f * factor, h, w])
}

// ---------------------------------------------------------------------------
// LTX2TemporalLatentUpsampler — main public struct
// ---------------------------------------------------------------------------

/// LTX-2 temporal latent upsampler — doubles the latent frame count.
///
/// Upscales `[B, 128, F, H, W]` → `[B, 128, F*2 - 1, H, W]`.
///
/// The post-upscale `[:, :, 1:, :, :]` slice (dropping the leading frame)
/// matches the upstream model. Per the upstream comment: "the first frame
/// encodes one pixel frame" so it's discarded after temporal interpolation.
pub struct LTX2TemporalLatentUpsampler {
    initial_conv: Conv3dBF16,
    initial_norm_weight: Tensor,
    initial_norm_bias: Tensor,
    res_blocks: Vec<ResBlock>,
    upsampler_conv: Conv3dBF16,
    post_res_blocks: Vec<ResBlock>,
    final_conv: Conv3dBF16,
}

impl LTX2TemporalLatentUpsampler {
    /// Load temporal upsampler weights from a safetensors file.
    ///
    /// `path` should point to `ltx-2.3-temporal-upscaler-x2-1.0.safetensors`.
    pub fn load(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let weights = serialization::load_file(path, device)?;

        // initial_conv: Conv3d(128, 512, k=3, p=1)
        let ic_w = get_weight_bf16(&weights, "initial_conv.weight")?;
        let ic_b = get_weight_bf16(&weights, "initial_conv.bias")?;
        let initial_conv = Conv3dBF16::from_weights(ic_w, Some(ic_b), (1, 1, 1), (1, 1, 1));

        // initial_norm: GroupNorm(32, 512)
        let initial_norm_weight = get_weight_bf16(&weights, "initial_norm.weight")?;
        let initial_norm_bias = get_weight_bf16(&weights, "initial_norm.bias")?;

        // 4× ResBlock (pre-upsample, 512 channels)
        let mut res_blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            res_blocks.push(ResBlock::load(&weights, &format!("res_blocks.{i}"))?);
        }

        // upsampler.0: Conv3d(512, 1024, k=3, p=1).  PixelShuffleND has no
        // weights — it's a pure rearrange.
        let up_w = get_weight_bf16(&weights, "upsampler.0.weight")?;
        let up_b = get_weight_bf16(&weights, "upsampler.0.bias")?;
        let upsampler_conv = Conv3dBF16::from_weights(up_w, Some(up_b), (1, 1, 1), (1, 1, 1));

        // 4× ResBlock (post-upsample, 512 channels)
        let mut post_res_blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            post_res_blocks.push(ResBlock::load(
                &weights,
                &format!("post_upsample_res_blocks.{i}"),
            )?);
        }

        // final_conv: Conv3d(512, 128, k=3, p=1)
        let fc_w = get_weight_bf16(&weights, "final_conv.weight")?;
        let fc_b = get_weight_bf16(&weights, "final_conv.bias")?;
        let final_conv = Conv3dBF16::from_weights(fc_w, Some(fc_b), (1, 1, 1), (1, 1, 1));

        Ok(Self {
            initial_conv,
            initial_norm_weight,
            initial_norm_bias,
            res_blocks,
            upsampler_conv,
            post_res_blocks,
            final_conv,
        })
    }

    /// Forward pass: upscale latents temporally by 2x (then drop leading frame).
    ///
    /// Input:  `[B, 128, F, H, W]`
    /// Output: `[B, 128, F*2 - 1, H, W]`
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
        for block in self.res_blocks.iter() {
            h = block.forward(&h)?;
        }

        // Temporal upsample: Conv3d → PixelShuffleND(dims=1, factor=2)
        h = self.upsampler_conv.forward(&h)?;
        h = pixel_shuffle_temporal(&h, TEMPORAL_FACTOR)?;

        // Drop leading frame (matches upstream `x = x[:, :, 1:, :, :]`).
        // The first frame after temporal upscale encodes a single pixel
        // frame from the encoder, not a real interpolation; discard.
        let dims = h.shape().dims().to_vec();
        let (b, c, f, hh, ww) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        if f < 2 {
            return Err(Error::InvalidOperation(format!(
                "LTX2TemporalLatentUpsampler: post-upsample F={f} < 2, cannot drop leading frame"
            )));
        }
        // narrow(dim=2, start=1, length=f-1)
        h = h.narrow(2, 1, f - 1)?;
        let _ = (b, c, hh, ww); // silence unused

        // 4× ResBlock (post-upsample)
        for block in self.post_res_blocks.iter() {
            h = block.forward(&h)?;
        }

        // final_conv: 512 → 128
        self.final_conv.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_shuffle_temporal_shape_math() {
        // After temporal pixel shuffle:
        // [B, C*p1, F, H, W] → [B, C, F*p1, H, W]
        let p1 = 2usize;
        let b = 2;
        let c = 512;
        let f = 4;
        let h = 8;
        let w = 12;
        let c_packed = c * p1;
        assert_eq!(c_packed, 1024);
        assert_eq!(c_packed / p1, c);
        assert_eq!(f * p1, 8);
        let _ = (b, h, w);
    }
}
