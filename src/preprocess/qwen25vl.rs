//! Lance `image_edit` preprocessing for the Qwen2.5-VL ViT path.
//!
//! Pipeline (mirror of `data/transforms.py::VideoTransform` with the
//! `vit_video_transform_args` dict from `data/datasets_custom/validation_dataset.py:147`):
//!
//! 1. Load image (RGB, white-background composite for RGBA).
//! 2. [`BucketResize`] with `max_area = resolution^2`, aspect ratios
//!    `["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]`, and **stride = 16**
//!    (the default in `NaResize`, since `vit_video_transform_args` does not
//!    override `stride_spatial`).
//! 3. [`divisible_crop`] with factor 28 (`vit_patch_size * spatial_merge`).
//! 4. CLIP-normalize to `[3, H, W]` F32 CHW.
//! 5. Add a `T` axis and duplicate the frame so `T == 2`. This matches
//!    `validation_dataset.py:274` (`video_tensor.repeat(1, 2, 1, 1)` when
//!    `element_dtype == "image"`), which exists because the encoder's
//!    `temporal_patch_size = 2` requires `T % 2 == 0`.
//! 6. [`patchify_video_with_merge`] (port of `data/data_utils.py:108`) to
//!    `[N_patches, 1176]` F32.
//! 7. Cast to BF16 — that is the `pixel_values` consumed by
//!    `Qwen25VLVisionTower::forward`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use image::imageops::FilterType;

use crate::preprocess::bucket_resize::{divisible_crop, BucketResize};
use crate::preprocess::common::{
    chw_normalize_from_rgb, load_image_rgb_white_bg, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD,
};

pub const VIT_PATCH_SIZE: u32 = 14;
pub const VIT_TEMPORAL_PATCH_SIZE: u32 = 2;
pub const VIT_SPATIAL_MERGE_SIZE: u32 = 2;
/// `vit_patch_size * spatial_merge_size` — the divisibility required by
/// Qwen2.5-VL post-merge tokens.
pub const VIT_DIVISIBLE_CROP: u32 = 28;
/// Default spatial stride used by `BucketResize` for the ViT path
/// (`NaResize.kwargs.get("stride", 16)`).
pub const VIT_BUCKET_STRIDE: u32 = 16;
pub const DEFAULT_ASPECT_RATIOS: &[&str] = &["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];

/// Output of [`prepare_image_for_vit`].
pub struct PreparedVitImage {
    /// `[N_patches_pre_merge, 1176]` BF16. `N = grid_t * grid_h * grid_w`.
    pub pixel_values: Tensor,
    /// `[grid_t, grid_h, grid_w]` — patch units before spatial merge.
    /// For a single image, `grid_t = 1`, `grid_h = H / 14`, `grid_w = W / 14`.
    pub grid_thw: [u32; 3],
    /// Pre-patchify spatial dims (post bucket-resize, post divisible-crop).
    pub processed_h: u32,
    pub processed_w: u32,
}

/// Lance `image_edit` ViT preprocessor. Loads an image and produces the
/// `pixel_values` + `grid_thw` consumed by
/// [`crate::models::qwen25vl_vit::Qwen25VLVisionTower::forward`].
pub fn prepare_image_for_vit(
    path: &Path,
    resolution: u32,
    device: &Arc<CudaDevice>,
) -> Result<PreparedVitImage> {
    let img = load_image_rgb_white_bg(path)?;

    let bucket = BucketResize::new(
        DEFAULT_ASPECT_RATIOS,
        resolution
            .checked_mul(resolution)
            .ok_or_else(|| anyhow!("resolution^2 overflowed u32: {resolution}"))?,
        VIT_BUCKET_STRIDE,
        FilterType::CatmullRom,
    )?;
    let resized = bucket.apply(&img);
    let cropped = divisible_crop(&resized, VIT_DIVISIBLE_CROP);

    let (w, h) = (cropped.width(), cropped.height());
    if h % VIT_PATCH_SIZE != 0 || w % VIT_PATCH_SIZE != 0 {
        return Err(anyhow!(
            "post divisible_crop dims ({w}x{h}) not divisible by patch {VIT_PATCH_SIZE}"
        ));
    }

    // CLIP-normalized CHW F32 view: length = 3 * h * w.
    let chw = chw_normalize_from_rgb(&cropped, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD);

    // Temporal duplicate: build [3, 2, H, W] F32 with frame 0 == frame 1.
    // Source data is already in [C, H, W] / CHW layout; we want [C, T, H, W]
    // with T=2 and T-axis as the inner contiguous "frame index" right after C.
    // The CTHW layout means storage order is C, T, H, W with T fastest among
    // (T,H,W) outer. Linearly, for fixed C and (h,w), the two T entries sit
    // at offsets [c*T*H*W + 0*H*W + h*W + w] and [c*T*H*W + 1*H*W + h*W + w].
    // We just copy the H*W block per (C,T) twice from the single source.
    let hw = (h as usize) * (w as usize);
    let cthw_len = 3usize * 2 * hw;
    let mut cthw = vec![0f32; cthw_len];
    for c in 0..3usize {
        let src = &chw[c * hw..(c + 1) * hw];
        let dst_base = c * 2 * hw;
        cthw[dst_base..dst_base + hw].copy_from_slice(src);
        cthw[dst_base + hw..dst_base + 2 * hw].copy_from_slice(src);
    }

    let cthw_tensor = Tensor::from_vec_dtype(
        cthw,
        Shape::from_dims(&[3, 2, h as usize, w as usize]),
        device.clone(),
        DType::F32,
    )
    .context("build CTHW F32 tensor")?;

    let patches_f32 = patchify_video_with_merge(
        &cthw_tensor,
        VIT_PATCH_SIZE,
        VIT_TEMPORAL_PATCH_SIZE,
        VIT_SPATIAL_MERGE_SIZE,
    )?;

    let pixel_values = patches_f32.to_dtype(DType::BF16).context("cast to BF16")?;

    let grid_thw = [1u32, h / VIT_PATCH_SIZE, w / VIT_PATCH_SIZE];

    let expected_n = (grid_thw[0] * grid_thw[1] * grid_thw[2]) as usize;
    let got_n = pixel_values.shape().dims()[0];
    if expected_n != got_n {
        return Err(anyhow!(
            "patch count mismatch: grid_thw={:?} implies {}, got {}",
            grid_thw,
            expected_n,
            got_n,
        ));
    }

    Ok(PreparedVitImage {
        pixel_values,
        grid_thw,
        processed_h: h,
        processed_w: w,
    })
}

/// Pure-tensor port of `data/data_utils.py:108::patchify_video_with_merge`.
///
/// Input: `[C, T, H, W]` F32 (host-materialized for the reshuffle). Output:
/// `[gt * gh * gw, C * tp * p * p]` F32, where `gt = T/tp`, `gh = H/p`,
/// `gw = W/p`.
///
/// Python reference:
/// ```text
/// video = rearrange(video, "C T H W -> T C H W")
/// T, C, H, W = video.shape
/// video = video.reshape(gt, tp, C, gh//ms, ms, p, gw//ms, ms, p)
/// video = video.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
/// patches = video.reshape(gt*gh*gw, C*tp*p*p)
/// ```
///
/// Implementation note: `flame_core::Tensor::permute` caps at 8 dimensions
/// (the 9-axis permute above is rejected by `permute_generic`). The reshuffle
/// is therefore performed on the host F32 buffer, which is the only stage
/// where this is acceptable — input volumes for ViT preprocessing are small
/// (e.g. 476² × 3 × 2 F32 ≈ 5.5 MB).
pub fn patchify_video_with_merge(
    cthw: &Tensor,
    spatial_patch_size: u32,
    temporal_patch_size: u32,
    merge_size: u32,
) -> Result<Tensor> {
    let dims = cthw.shape().dims();
    if dims.len() != 4 {
        return Err(anyhow!(
            "patchify_video_with_merge expects [C,T,H,W], got {:?}",
            dims
        ));
    }
    let (c, t, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let p = spatial_patch_size as usize;
    let tp = temporal_patch_size as usize;
    let ms = merge_size as usize;

    if t % tp != 0 || h % p != 0 || w % p != 0 {
        return Err(anyhow!(
            "patchify: T={t},H={h},W={w} not divisible by tp={tp},p={p}"
        ));
    }
    let gt = t / tp;
    let gh = h / p;
    let gw = w / p;
    if gh % ms != 0 || gw % ms != 0 {
        return Err(anyhow!(
            "patchify: grid ({gh}x{gw}) not divisible by merge_size {ms}"
        ));
    }

    // F32 input materializes via to_vec for the 9-axis permute below. Tensor::to_vec
    // always returns F32 (auto-casting BF16); for F32 input this is a plain D2H copy.
    let src = cthw.to_vec().context("materialize CTHW to host F32")?;

    // Source layout: [C, T, H, W] row-major. Strides:
    //   c -> T*H*W, t -> H*W, h -> W, w -> 1
    let stride_c = t * h * w;
    let stride_t = h * w;
    let stride_h = w;

    let out_rows = gt * gh * gw;
    let out_cols = c * tp * p * p;
    let mut out = vec![0f32; out_rows * out_cols];

    // Output layout (from the permute order `0,3,6,4,7,2,1,5,8` on
    // `[gt, tp, C, gh/ms, ms, p, gw/ms, ms, p]`):
    //
    //   axes:        0,    3,    6,    4,    7,    2,    1,    5,    8
    //   semantics:   gt,  gh/ms, gw/ms, ms,   ms,   C,    tp,   p,    p
    //
    // Then reshape collapses to `(gt*gh*gw)` rows × `(C*tp*p*p)` cols.
    // The leading 5 axes pack into the row index in this order; the
    // trailing 4 axes pack into the column index.
    //
    // Mapping back to source (C,T,H,W):
    //   t_src = a_gt * tp + a_tp
    //   h_src = (a_gh_ms * ms + a_ms_h) * p + a_p_h
    //   w_src = (a_gw_ms * ms + a_ms_w) * p + a_p_w
    //   c_src = a_c
    for a_gt in 0..gt {
        for a_gh_ms in 0..(gh / ms) {
            for a_gw_ms in 0..(gw / ms) {
                for a_ms_h in 0..ms {
                    for a_ms_w in 0..ms {
                        // row index from outer 5 axes.
                        let row = (((a_gt * (gh / ms) + a_gh_ms) * (gw / ms) + a_gw_ms) * ms
                            + a_ms_h)
                            * ms
                            + a_ms_w;
                        let row_base = row * out_cols;
                        for a_c in 0..c {
                            for a_tp in 0..tp {
                                let t_src = a_gt * tp + a_tp;
                                let h_base = (a_gh_ms * ms + a_ms_h) * p;
                                let w_base = (a_gw_ms * ms + a_ms_w) * p;
                                let src_chan = a_c * stride_c + t_src * stride_t;
                                let col_block_base =
                                    ((a_c * tp + a_tp) * p) * p; // axes 2,1 then 5,8
                                for a_p_h in 0..p {
                                    let src_row = src_chan + (h_base + a_p_h) * stride_h;
                                    let dst_row =
                                        row_base + col_block_base + a_p_h * p;
                                    for a_p_w in 0..p {
                                        out[dst_row + a_p_w] =
                                            src[src_row + w_base + a_p_w];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_t = Tensor::from_vec_dtype(
        out,
        Shape::from_dims(&[out_rows, out_cols]),
        cthw.device().clone(),
        DType::F32,
    )
    .context("build patchified F32 tensor")?;
    Ok(out_t)
}
