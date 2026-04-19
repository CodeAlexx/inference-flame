//! LTX-2.3 multi-keyframe / video-extension conditioning helpers.
//!
//! Pipeline-level utilities for building the conditioning mask and merged
//! latents that `LTX2StreamingModel::forward_video_only_i2v` already
//! consumes. Operates in **latent space** — callers VAE-encode their
//! reference frames/clips upstream and pass us `ConditioningItem`s with
//! pre-computed latents.
//!
//! # Scope (landing 1)
//!
//! We match the subset of
//! `/tmp/ltx-video/ltx_video/pipelines/pipeline_ltx_video.py::prepare_conditioning`
//! (:1383-1587) that does NOT need the "extra conditioning tokens" path.
//! That means:
//!
//! | Case | `frame_number` | `f_l` (latent frames of item) | What we do |
//! |------|----------------|-------------------------------|------------|
//! | I2V / frame-0 | `== 0`     | any                           | centered spatial lerp on the target grid — Python :1466-1487 |
//! | Multi-frame mid-clip | `> 0` | `> 2`                    | `_handle_non_first_conditioning_sequence` with `prefix_latents_mode="drop"` — Python :1687-1697 then :1714-1716 |
//! | Single-frame mid-keyframe | `> 0` | `== 1`                | **Pragmatic grid-lerp fallback** (NOT Lightricks-default). See note below. |
//!
//! ## Pragmatic single-frame mid-keyframe note
//!
//! Python's `prepare_conditioning` (pipeline_ltx_video.py:1505-1541) puts
//! single-frame mid-keyframes onto the "extra tokens" path — they become
//! tokens prepended to the sequence with their own pixel coordinates,
//! bypassing the target grid. That requires the transformer forward to
//! accept an arbitrary sequence length plus explicit pixel coords, which
//! is a non-trivial refactor of
//! `LTX2StreamingModel::forward_audio_video_with_stg`.
//!
//! Instead we lerp directly into the target latent grid at
//! `latent_frame = frame_number // 8` and mark those positions in the
//! conditioning mask with `strength`. The model sees the reference frame
//! at its intended temporal position; the cost is losing Lightricks's
//! spatial-position flexibility (media_x/y) for single-frame
//! mid-keyframes — always OK for the multi-keyframe use case, where the
//! reference fills the frame.
//!
//! Multi-frame `prefix_latents_mode="concat"` (Python default at
//! :1717-1719) is similarly deferred — we use "drop" instead, which
//! discards the 2 boundary latent frames that Lightricks would otherwise
//! inject as extra tokens.
//!
//! # Parity
//!
//! Numerics match `scripts/ltx2_conditioning_mask_ref.py` exactly for
//! the supported cases; see `src/bin/ltx2_conditioning_parity.rs`.

use flame_core::{DType, Result, Shape, Tensor};

/// Number of "prefix" latent frames at the boundary of a non-first
/// conditioning sequence, matching Lightricks's
/// `_handle_non_first_conditioning_sequence(num_prefix_latent_frames=2)`
/// (pipeline_ltx_video.py:1659).
pub const NUM_PREFIX_LATENT_FRAMES: usize = 2;

/// A single conditioning item, in **latent** space. Callers encode their
/// pixel-space reference with the LTX-2 VAE and hand us the result.
///
/// Shape conventions (mirroring LTX-Video):
/// - `latent`: `[B, 128, f_l, H_lat, W_lat]`, post-VAE normalized
/// - `frame_number`: start frame in the generated video pixel sequence,
///   must be `% 8 == 0` unless it's 0.
/// - `strength`: `1.0` = hard conditioning (timestep forced to 0 for
///   these tokens in the inner forward), `0.0` = no effect,
///   `(0, 1)` = lerp blend.
#[derive(Debug, Clone)]
pub struct ConditioningItem {
    pub latent: Tensor,
    pub frame_number: usize,
    pub strength: f32,
}

/// Merge `conditioning_items` into `init_latents` and build a
/// per-latent-cell conditioning mask.
///
/// # Inputs
/// - `init_latents`: `[B, 128, F_lat, H_lat, W_lat]` target grid
///   (typically pure noise before denoising).
/// - `items`: list of [`ConditioningItem`]s.
///
/// # Outputs
/// - `merged_latents`: `[B, 128, F_lat, H_lat, W_lat]` — `init_latents`
///   with the item latents lerped in at their spatial/temporal
///   positions (strength `[0,1]`).
/// - `conditioning_mask_5d`: `[B, 1, F_lat, H_lat, W_lat]`, dtype `F32`,
///   value `strength` at conditioned cells and `0.0` elsewhere.
///
/// Patchify the mask to the per-token form the transformer expects with
/// [`pack_conditioning_mask_for_transformer`].
///
/// See module docs for which Python paths this covers.
pub fn prepare_conditioning(
    init_latents: &Tensor,
    items: &[ConditioningItem],
) -> Result<(Tensor, Tensor)> {
    let dims = init_latents.shape().dims().to_vec();
    if dims.len() != 5 {
        return Err(flame_core::Error::InvalidInput(format!(
            "init_latents must be 5D [B, C, F, H, W], got {:?}",
            dims
        )));
    }
    let (batch, channels, f_lat, h_lat, w_lat) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let device = init_latents.device().clone();
    let dtype = init_latents.dtype();

    // Early exit: no conditioning → identity + zero mask.
    if items.is_empty() {
        let zero_mask = Tensor::zeros_dtype(
            Shape::from_dims(&[batch, 1, f_lat, h_lat, w_lat]),
            DType::F32,
            device,
        )?;
        return Ok((init_latents.clone(), zero_mask));
    }

    let mut merged = init_latents.clone();
    let mut mask = Tensor::zeros_dtype(
        Shape::from_dims(&[batch, 1, f_lat, h_lat, w_lat]),
        DType::F32,
        device.clone(),
    )?;

    for (i, item) in items.iter().enumerate() {
        let ldims = item.latent.shape().dims().to_vec();
        if ldims.len() != 5 {
            return Err(flame_core::Error::InvalidInput(format!(
                "conditioning item {i}: latent must be 5D, got {:?}",
                ldims
            )));
        }
        if ldims[0] != batch || ldims[1] != channels {
            return Err(flame_core::Error::InvalidInput(format!(
                "conditioning item {i}: batch/channels mismatch: item={:?} grid={:?}",
                ldims, dims
            )));
        }
        if ldims[3] > h_lat || ldims[4] > w_lat {
            return Err(flame_core::Error::InvalidInput(format!(
                "conditioning item {i}: item spatial {:?} exceeds grid {}x{}",
                (ldims[3], ldims[4]),
                h_lat,
                w_lat
            )));
        }
        if !(item.strength >= 0.0 && item.strength <= 1.0) {
            return Err(flame_core::Error::InvalidInput(format!(
                "conditioning item {i}: strength {} out of [0,1]",
                item.strength
            )));
        }
        // Match item's dtype to init_latents (caller may have encoded at F32).
        let item_lat = if item.latent.dtype() != dtype {
            item.latent.to_dtype(dtype)?
        } else {
            item.latent.clone()
        };
        let f_l = ldims[2];
        let h_l = ldims[3];
        let w_l = ldims[4];

        if item.frame_number == 0 {
            // Case (A): frame-0, centered placement. Matches
            // pipeline_ltx_video.py:1466-1487 with strip_latent_border=False,
            // media_x=None, media_y=None.
            let y0 = (h_lat - h_l) / 2;
            let x0 = (w_lat - w_l) / 2;
            if f_l > f_lat {
                return Err(flame_core::Error::InvalidInput(format!(
                    "conditioning item {i}: f_l={} exceeds grid F={}",
                    f_l, f_lat
                )));
            }
            merged = lerp_slab_5d(
                &merged, &item_lat, /*f_start=*/ 0, f_l, y0, h_l, x0, w_l, item.strength,
            )?;
            mask = set_mask_slab_5d(
                &mask, item.strength, /*f_start=*/ 0, f_l, y0, h_l, x0, w_l,
            )?;
        } else {
            // frame_number > 0
            if item.frame_number % 8 != 0 {
                return Err(flame_core::Error::InvalidInput(format!(
                    "conditioning item {i}: frame_number {} must be a multiple of 8 (or 0)",
                    item.frame_number
                )));
            }

            if f_l > NUM_PREFIX_LATENT_FRAMES {
                // Case (B): multi-frame clip, "drop" prefix. Matches
                // pipeline_ltx_video.py:1687-1697 main-body branch then the
                // prefix_latents_mode="drop" branch at :1714-1716 (we do NOT
                // fall through to the extra-tokens path).
                let f_l_start = item.frame_number / 8 + NUM_PREFIX_LATENT_FRAMES;
                let body_len = f_l - NUM_PREFIX_LATENT_FRAMES;
                if f_l_start + body_len > f_lat {
                    return Err(flame_core::Error::InvalidInput(format!(
                        "conditioning item {i}: target latent frames {}..{} out of range (F_lat={})",
                        f_l_start, f_l_start + body_len, f_lat
                    )));
                }
                let body = item_lat.narrow(2, NUM_PREFIX_LATENT_FRAMES, body_len)?;
                // Full spatial: h_l must match the grid for mid-sequence.
                // (Python `_handle_non_first_conditioning_sequence` does not
                // touch H/W.)
                if h_l != h_lat || w_l != w_lat {
                    return Err(flame_core::Error::InvalidInput(format!(
                        "conditioning item {i}: mid-sequence item must cover the full HxW \
                         grid ({}x{}), got {}x{}",
                        h_lat, w_lat, h_l, w_l
                    )));
                }
                merged = lerp_slab_5d(
                    &merged, &body, f_l_start, body_len, 0, h_lat, 0, w_lat, item.strength,
                )?;
                mask = set_mask_slab_5d(
                    &mask, item.strength, f_l_start, body_len, 0, h_lat, 0, w_lat,
                )?;
            } else if f_l == 1 {
                // Case (C): single-frame mid-keyframe, PRAGMATIC grid-lerp.
                // See module docs — NOT Lightricks-default; Python would
                // extra-token this at :1505-1541.
                let f_l_start = item.frame_number / 8;
                if f_l_start >= f_lat {
                    return Err(flame_core::Error::InvalidInput(format!(
                        "conditioning item {i}: latent frame {} out of range (F_lat={})",
                        f_l_start, f_lat
                    )));
                }
                if h_l != h_lat || w_l != w_lat {
                    return Err(flame_core::Error::InvalidInput(format!(
                        "conditioning item {i}: mid-sequence item must cover the full HxW \
                         grid ({}x{}), got {}x{}",
                        h_lat, w_lat, h_l, w_l
                    )));
                }
                merged = lerp_slab_5d(
                    &merged, &item_lat, f_l_start, 1, 0, h_lat, 0, w_lat, item.strength,
                )?;
                mask = set_mask_slab_5d(
                    &mask, item.strength, f_l_start, 1, 0, h_lat, 0, w_lat,
                )?;
            } else {
                // f_l in (1, NUM_PREFIX_LATENT_FRAMES] — would need extra tokens.
                return Err(flame_core::Error::InvalidInput(format!(
                    "conditioning item {i}: non-first-sequence with f_l={} not supported \
                     (would require extra-tokens path; deferred).",
                    f_l
                )));
            }
        }
    }

    Ok((merged, mask))
}

/// In-place `lerp`-equivalent on a 5D slab. Returns a new tensor. Matches
/// `torch.lerp(dst_slab, src_slab, strength)` used throughout Lightricks.
///
/// `lerp(a, b, t) = a + t*(b-a) = (1-t)*a + t*b`.
#[allow(clippy::too_many_arguments)]
fn lerp_slab_5d(
    dst: &Tensor,
    src_slab: &Tensor,
    f_start: usize,
    f_len: usize,
    y0: usize,
    h_len: usize,
    x0: usize,
    w_len: usize,
    strength: f32,
) -> Result<Tensor> {
    // Concatenate: [0:f_start], lerped_slab, [f_start+f_len:F] along dim 2;
    // within the lerped slab, along dim 3: [0:y0], lerped_row, [y0+h_len:H];
    // within the lerped row, along dim 4: [0:x0], lerped_core, [x0+w_len:W].
    //
    // The slab is small relative to the grid in the I2V path, but for video
    // extension it can be the full grid — which simplifies to a single concat
    // chain, and for the full-spatial case we skip the spatial splits entirely.
    let dims = dst.shape().dims().to_vec();
    let (_, _, f_total, h_total, w_total) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

    // Full spatial fast-path: src covers the entire HxW, so we can skip the
    // inner splits and just do the temporal concat.
    let full_spatial = y0 == 0 && h_len == h_total && x0 == 0 && w_len == w_total;

    // Build the lerped slab [B, C, f_len, h_len, w_len].
    let dst_slab = if full_spatial {
        dst.narrow(2, f_start, f_len)?
    } else {
        dst.narrow(2, f_start, f_len)?
            .narrow(3, y0, h_len)?
            .narrow(4, x0, w_len)?
    };
    // lerped = dst_slab + strength * (src_slab - dst_slab)
    let diff = src_slab.sub(&dst_slab)?;
    let scaled = diff.mul_scalar(strength)?;
    let lerped = dst_slab.add(&scaled)?;

    // If full HxW, rebuild along dim 2 only.
    if full_spatial {
        let mut pieces: Vec<Tensor> = Vec::with_capacity(3);
        if f_start > 0 {
            pieces.push(dst.narrow(2, 0, f_start)?);
        }
        pieces.push(lerped);
        let tail = f_start + f_len;
        if tail < f_total {
            pieces.push(dst.narrow(2, tail, f_total - tail)?);
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        return Tensor::cat(&refs, 2);
    }

    // Otherwise need to rebuild the temporal slab with spatial splits.
    // Step 1: rebuild along dim 4 (width) within the slab.
    let slab_full_f = dst.narrow(2, f_start, f_len)?;
    // We need to replace [y0:y0+h_len, x0:x0+w_len] in slab_full_f.
    // Assemble slab row-by-row across dim 3.
    let mut rows: Vec<Tensor> = Vec::with_capacity(3);
    if y0 > 0 {
        rows.push(slab_full_f.narrow(3, 0, y0)?);
    }
    // Middle row strip with width splits.
    let row_slab = slab_full_f.narrow(3, y0, h_len)?;
    let mut cols: Vec<Tensor> = Vec::with_capacity(3);
    if x0 > 0 {
        cols.push(row_slab.narrow(4, 0, x0)?);
    }
    cols.push(lerped);
    let tail_x = x0 + w_len;
    if tail_x < w_total {
        cols.push(row_slab.narrow(4, tail_x, w_total - tail_x)?);
    }
    let col_refs: Vec<&Tensor> = cols.iter().collect();
    rows.push(Tensor::cat(&col_refs, 4)?);
    let tail_y = y0 + h_len;
    if tail_y < h_total {
        rows.push(slab_full_f.narrow(3, tail_y, h_total - tail_y)?);
    }
    let row_refs: Vec<&Tensor> = rows.iter().collect();
    let new_slab = Tensor::cat(&row_refs, 3)?;

    // Step 2: rebuild along dim 2 (temporal).
    let mut pieces: Vec<Tensor> = Vec::with_capacity(3);
    if f_start > 0 {
        pieces.push(dst.narrow(2, 0, f_start)?);
    }
    pieces.push(new_slab);
    let tail_f = f_start + f_len;
    if tail_f < f_total {
        pieces.push(dst.narrow(2, tail_f, f_total - tail_f)?);
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 2)
}

/// Set `mask[f_start:f_start+f_len, y0:y0+h_len, x0:x0+w_len] = strength`
/// on the `[B, 1, F, H, W]` mask. Returns a new tensor. Mask is F32.
#[allow(clippy::too_many_arguments)]
fn set_mask_slab_5d(
    mask: &Tensor,
    strength: f32,
    f_start: usize,
    f_len: usize,
    y0: usize,
    h_len: usize,
    x0: usize,
    w_len: usize,
) -> Result<Tensor> {
    let dims = mask.shape().dims().to_vec();
    let (batch, _, f_total, h_total, w_total) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let device = mask.device().clone();

    // Build the replacement slab = constant `strength`.
    let slab_shape = Shape::from_dims(&[batch, 1, f_len, h_len, w_len]);
    let slab = if strength == 0.0 {
        Tensor::zeros_dtype(slab_shape, DType::F32, device)?
    } else if strength == 1.0 {
        Tensor::ones_dtype(slab_shape, DType::F32, device)?
    } else {
        // ones * strength
        Tensor::ones_dtype(slab_shape, DType::F32, device)?.mul_scalar(strength)?
    };

    // Reuse the same 5D splice as lerp_slab_5d's mask form.
    let full_spatial = y0 == 0 && h_len == h_total && x0 == 0 && w_len == w_total;
    if full_spatial {
        let mut pieces: Vec<Tensor> = Vec::with_capacity(3);
        if f_start > 0 {
            pieces.push(mask.narrow(2, 0, f_start)?);
        }
        pieces.push(slab);
        let tail = f_start + f_len;
        if tail < f_total {
            pieces.push(mask.narrow(2, tail, f_total - tail)?);
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        return Tensor::cat(&refs, 2);
    }

    let slab_full_f = mask.narrow(2, f_start, f_len)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(3);
    if y0 > 0 {
        rows.push(slab_full_f.narrow(3, 0, y0)?);
    }
    let row_slab = slab_full_f.narrow(3, y0, h_len)?;
    let mut cols: Vec<Tensor> = Vec::with_capacity(3);
    if x0 > 0 {
        cols.push(row_slab.narrow(4, 0, x0)?);
    }
    cols.push(slab);
    let tail_x = x0 + w_len;
    if tail_x < w_total {
        cols.push(row_slab.narrow(4, tail_x, w_total - tail_x)?);
    }
    let col_refs: Vec<&Tensor> = cols.iter().collect();
    rows.push(Tensor::cat(&col_refs, 4)?);
    let tail_y = y0 + h_len;
    if tail_y < h_total {
        rows.push(slab_full_f.narrow(3, tail_y, h_total - tail_y)?);
    }
    let row_refs: Vec<&Tensor> = rows.iter().collect();
    let new_slab = Tensor::cat(&row_refs, 3)?;

    let mut pieces: Vec<Tensor> = Vec::with_capacity(3);
    if f_start > 0 {
        pieces.push(mask.narrow(2, 0, f_start)?);
    }
    pieces.push(new_slab);
    let tail_f = f_start + f_len;
    if tail_f < f_total {
        pieces.push(mask.narrow(2, tail_f, f_total - tail_f)?);
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 2)
}

/// Reshape `[B, 1, F, H, W]` conditioning mask to `[B, F*H*W]` — the
/// per-token form `LTX2StreamingModel::forward_video_only_i2v` expects.
/// Matches Lightricks's patchifier with `patch_size=1` (inference.py:242),
/// which for LTX-2 reduces to a plain squeeze + reshape.
pub fn pack_conditioning_mask_for_transformer(mask_5d: &Tensor) -> Result<Tensor> {
    let dims = mask_5d.shape().dims().to_vec();
    if dims.len() != 5 || dims[1] != 1 {
        return Err(flame_core::Error::InvalidInput(format!(
            "mask_5d must be [B, 1, F, H, W], got {:?}",
            dims
        )));
    }
    let (batch, _, f_lat, h_lat, w_lat) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    mask_5d.reshape(&[batch, f_lat * h_lat * w_lat])
}

/// Add timestep-dependent noise to hard-conditioning latents.
///
/// Port of `LTXVideoPipeline.add_noise_to_image_conditioning_latents`
/// (pipeline_ltx_video.py:596-620):
///
/// ```text
/// need_to_noise = conditioning_mask > 1.0 - eps   # [B, 1, F, H, W]
/// noised = init_latents + noise_scale * noise * t^2
/// latents = where(need_to_noise, noised, latents)
/// ```
///
/// Caller is responsible for:
/// - Providing `noise` at `latents.shape` and dtype.
/// - Supplying a stored `init_latents` — the snapshot of merged latents
///   **before** any denoising steps, so the re-injected conditioning
///   stays anchored at the reference.
///
/// `eps` defaults to `1e-6`.
pub fn add_image_cond_noise(
    init_latents: &Tensor,
    latents: &Tensor,
    conditioning_mask_5d: &Tensor,
    noise: &Tensor,
    t: f32,
    noise_scale: f32,
) -> Result<Tensor> {
    let eps = 1e-6f32;
    let dtype = latents.dtype();

    let dims = conditioning_mask_5d.shape().dims().to_vec();
    if dims.len() != 5 || dims[1] != 1 {
        return Err(flame_core::Error::InvalidInput(format!(
            "conditioning_mask_5d must be [B, 1, F, H, W], got {:?}",
            dims
        )));
    }

    // Broadcast mask to latents.shape (F32 → same as `latents` ultimately).
    let broadcast_shape = latents.shape().dims().to_vec();
    let mask_b = conditioning_mask_5d.expand(&broadcast_shape)?;

    // Build a hard 0/1 binary mask: `mask >= 1 - eps`.
    //
    // Since our mask values are exactly in {0.0, strength} where strength ∈
    // [0, 1], computing `relu(mask - (1-eps)) / eps` yields values in
    // {0, (strength-(1-eps))/eps} — which is 0 for strength<1-eps and
    // approximately 1 for strength=1. flame-core's `where_mask` treats the
    // mask as a blend factor (a*m + b*(1-m)), so we then clamp to [0,1] to
    // guarantee we never extrapolate. `clamp_bf16` supports BF16; for F32 we
    // take `min(1, relu(...))` by subtracting 1 and re-ReLU-ing.
    let thresholded = mask_b.add_scalar(-(1.0 - eps))?;
    let positive = thresholded.relu()?; // 0 or ~eps for strength=1
    let scaled = positive.mul_scalar(1.0 / eps)?; // ~0 or ~1
    // Saturate to [0, 1]: x - relu(x - 1) == min(x, 1) for x>=0.
    let minus_one = scaled.add_scalar(-1.0)?;
    let overshoot = minus_one.relu()?;
    let binary_f32 = scaled.sub(&overshoot)?;

    // Cast binary mask into the latent dtype.
    let binary = if binary_f32.dtype() != dtype {
        binary_f32.to_dtype(dtype)?
    } else {
        binary_f32
    };

    // Noised reference: init + noise_scale * noise * t^2
    let t_sq = t * t;
    let scale = noise_scale * t_sq;
    let noise_scaled = noise.mul_scalar(scale)?;
    let noised = init_latents.add(&noise_scaled)?;
    let noised = if noised.dtype() != dtype {
        noised.to_dtype(dtype)?
    } else {
        noised
    };

    // where_mask(mask, a, b): a * mask + b * (1 - mask).
    Tensor::where_mask(&binary, &noised, latents)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::global_cuda_device;

    fn grid_f32(data: Vec<f32>, dims: &[usize]) -> Tensor {
        Tensor::from_vec(data, Shape::from_dims(dims), global_cuda_device()).unwrap()
    }

    #[test]
    fn empty_items_is_identity() {
        let d = global_cuda_device();
        let init = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 128, 3, 2, 2]),
            DType::F32,
            d.clone(),
        )
        .unwrap();
        let (merged, mask) = prepare_conditioning(&init, &[]).unwrap();
        assert_eq!(merged.shape().dims(), init.shape().dims());
        assert_eq!(mask.shape().dims(), &[1, 1, 3, 2, 2]);
        // mask must be all zeros
        let v = mask.to_vec().unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn frame0_strength1_hard_replaces_target_frame() {
        let d = global_cuda_device();
        let init = grid_f32(vec![0.0f32; 1 * 2 * 3 * 2 * 2], &[1, 2, 3, 2, 2]);
        let item = grid_f32(
            (0..(1 * 2 * 1 * 2 * 2)).map(|i| i as f32 + 1.0).collect(),
            &[1, 2, 1, 2, 2],
        );
        let (merged, mask) = prepare_conditioning(
            &init,
            &[ConditioningItem {
                latent: item.clone(),
                frame_number: 0,
                strength: 1.0,
            }],
        )
        .unwrap();
        // Frame 0 should equal item exactly (strength=1 lerp from 0).
        let m = merged.to_vec().unwrap();
        let it = item.to_vec().unwrap();
        // Merged is [B, C, F, H, W] = [1, 2, 3, 2, 2]; frame 0 covers
        // indices for (f=0). Channel stride = F*H*W = 12. Frame stride = 4.
        for c in 0..2 {
            for hw in 0..4 {
                let merged_idx = c * 12 + 0 * 4 + hw;
                let item_idx = c * 4 + hw;
                assert!((m[merged_idx] - it[item_idx]).abs() < 1e-6);
            }
        }
        // Mask shape [1, 1, 3, 2, 2] = 12 elements. Frame 0 (first 4) should be 1, rest 0.
        let mk = mask.to_vec().unwrap();
        for i in 0..4 {
            assert_eq!(mk[i], 1.0);
        }
        for i in 4..12 {
            assert_eq!(mk[i], 0.0);
        }
    }
}
