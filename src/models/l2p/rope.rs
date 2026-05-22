//! L2P 3-axis RoPE — copy of `NextDiT::build_3d_rope`.
//!
//! Layout matches Z-Image's `RopeEmbedder`:
//! - axes_dims `[32, 48, 48]` (T, H, W). Sum = 128 = head_dim. Each
//!   axis contributes `axis_dim/2` frequencies → 16 + 24 + 24 = 64 =
//!   half_head_dim. The cos/sin buffers are laid out as
//!   `[total_seq, half_head_dim=64]` with axes packed contiguously
//!   along the last dim in axis order (T, then H, then W).
//! - theta = 256.0, complex-multiply form (interleaved-pair pairs
//!   later consumed by `rope_fused_bf16` with `RopeLayout::Interleaved`).
//! - Position id grid: caption at `(1+i, 0, 0)` for `i in 0..cap_len`,
//!   image at `(cap_len+1, ih, iw)` for `ih in 0..ph, iw in 0..pw`.
//!
//! Math identical to `inference_flame::models::zimage_nextdit::NextDiT::build_3d_rope`
//! (commit at the time of writing) — `i / half_axis` where `i in 0..half_axis`
//! matches the Python reference's `arange(0, d, 2) / d` because
//! `2k/d == k/(d/2)`.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use super::dit::L2pDiTConfig;

/// Build the `(cos, sin)` BF16 RoPE tables for an L2P forward pass.
///
/// Arguments mirror `NextDiT::build_3d_rope`:
/// - `cap_len`: caption token count (already padded to
///   `pad_tokens_multiple`).
/// - `ph, pw`: image patch-grid extents
///   (`image_H / patch_size`, `image_W / patch_size`).
/// - `img_pad_len`: image-token padding count to round the joint
///   sequence to `pad_tokens_multiple`.
///
/// Output shape: `[cap_len + ph*pw + img_pad_len, head_dim/2]` BF16,
/// for both `cos` and `sin`.
pub fn build_3d_rope(
    device: &Arc<cudarc::driver::CudaDevice>,
    config: &L2pDiTConfig,
    cap_len: usize,
    ph: usize,
    pw: usize,
    img_pad_len: usize,
) -> Result<(Tensor, Tensor)> {
    let axes_dims = config.axes_dims_rope;
    let theta = config.rope_theta;
    let img_seq = ph * pw + img_pad_len;
    let total_seq = cap_len + img_seq;
    let half_head_dim = config.head_dim / 2;

    let mut pos_ids = vec![[0.0f32; 3]; total_seq];

    // Caption positions: (1, 0, 0), (2, 0, 0), ...
    for i in 0..cap_len {
        pos_ids[i] = [(i + 1) as f32, 0.0, 0.0];
    }

    // Image positions: (cap_len+1, ih, iw)
    for ih in 0..ph {
        for iw in 0..pw {
            pos_ids[cap_len + ih * pw + iw] = [(cap_len + 1) as f32, ih as f32, iw as f32];
        }
    }

    // Build cos/sin — layout [total_seq, half_head_dim] with axes concatenated
    let mut cos_data = vec![0.0f32; total_seq * half_head_dim];
    let mut sin_data = vec![0.0f32; total_seq * half_head_dim];

    let mut offset = 0;
    for (axis_idx, &axis_dim) in axes_dims.iter().enumerate() {
        let half_axis = axis_dim / 2;
        let mut freqs = vec![0.0f32; half_axis];
        for i in 0..half_axis {
            freqs[i] = 1.0 / theta.powf(i as f32 / half_axis as f32);
        }

        for seq_idx in 0..total_seq {
            let pos = pos_ids[seq_idx][axis_idx];
            for (freq_idx, &freq) in freqs.iter().enumerate() {
                let angle = pos * freq;
                cos_data[seq_idx * half_head_dim + offset + freq_idx] = angle.cos();
                sin_data[seq_idx * half_head_dim + offset + freq_idx] = angle.sin();
            }
        }
        offset += half_axis;
    }

    let cos_tensor = Tensor::from_vec_dtype(
        cos_data,
        Shape::from_dims(&[total_seq, half_head_dim]),
        device.clone(),
        DType::BF16,
    )?;
    let sin_tensor = Tensor::from_vec_dtype(
        sin_data,
        Shape::from_dims(&[total_seq, half_head_dim]),
        device.clone(),
        DType::BF16,
    )?;

    Ok((cos_tensor, sin_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::global_cuda_device;

    /// At L2P's 1024² inference reference shape:
    /// - cap_len = 320 (a multiple of pad_tokens_multiple=32)
    /// - ph = pw = 64 (1024 / 16)
    /// - img_pad_len = 0 (64*64 = 4096 is already a multiple of 32)
    /// the cos/sin tables should be `[320 + 64*64, head_dim/2]` = `[4416, 64]`,
    /// both BF16.
    #[test]
    fn rope_shape_and_dtype_at_1024_reference() {
        let device = global_cuda_device();
        let config = L2pDiTConfig::default();

        let cap_len = 320usize;
        let ph = 64usize;
        let pw = 64usize;
        let img_pad_len = 0usize;

        let (cos, sin) = build_3d_rope(&device, &config, cap_len, ph, pw, img_pad_len)
            .expect("build_3d_rope must succeed at L2P reference shape");

        let expected_seq = cap_len + ph * pw + img_pad_len; // 320 + 4096 = 4416
        let expected_half = config.head_dim / 2; // 64

        assert_eq!(cos.shape().dims(), &[expected_seq, expected_half]);
        assert_eq!(sin.shape().dims(), &[expected_seq, expected_half]);
        assert_eq!(cos.dtype(), DType::BF16);
        assert_eq!(sin.dtype(), DType::BF16);
    }
}
