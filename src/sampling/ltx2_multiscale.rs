//! LTX-2 multi-scale helpers: AdaIN latent filtering + sigmoid tone mapping.
//!
//! Direct ports of two pure tensor-math helpers from Lightricks's
//! `ltx_video/pipelines/pipeline_ltx_video.py`:
//!
//! - `adain_filter_latent`  (:1790-1818) — per-channel mean+std match of a
//!   target tensor to a reference tensor with a lerp factor.
//! - `tone_map_latents`     (:1749-1787) — sigmoid-based amplitude
//!   compression applied element-wise before VAE decode.
//!
//! Both are parity-checked against Lightricks's actual Python functions
//! via `scripts/ltx2_adain_tone_map_ref.py` → `output/ltx2_adain_tone_map_ref.safetensors`,
//! read back by `bin/ltx2_adain_tone_map_parity.rs`.
//!
//! Shape: both functions operate on 5D NCFHW tensors (latent video layout).
//! AdaIN's statistics are reduced over (F, H, W) per (B, C) — i.e. one
//! mean and one std per channel per batch element. This matches Python's
//! `torch.std_mean(reference_latents[i, c], dim=None)` which collapses the
//! full [F, H, W] slab to scalars.

use flame_core::{DType, Error, Result, Tensor};

/// AdaIN filter latent — mean/std match `latents` to `reference_latents`
/// per (B, C) channel, blended with `factor`:
///
/// ```text
/// matched_bc   = ((latents_bc - mean(latents_bc)) / std(latents_bc))
///                * std(ref_bc) + mean(ref_bc)
/// output_bc    = latents_bc + factor * (matched_bc - latents_bc)
///              = lerp(latents_bc, matched_bc, factor)
/// ```
///
/// Statistics are computed in F32 for stability, then cast back to the
/// input dtype. Lightricks's Python reference uses `torch.std_mean` which
/// defaults to unbiased (N-1) std — we mirror that.
///
/// Shape contract: `latents` and `reference_latents` must both be 5D and
/// share (B, C). The spatial/temporal sizes may differ (they typically do
/// in the multi-scale pipeline, where `reference` is low-res and
/// `latents` is upsampled). Per-channel stats are over ALL non-(B,C) axes.
pub fn adain_filter_latent(
    latents: &Tensor,
    reference_latents: &Tensor,
    factor: f32,
) -> Result<Tensor> {
    let ldims = latents.shape().dims().to_vec();
    let rdims = reference_latents.shape().dims().to_vec();
    if ldims.len() != 5 || rdims.len() != 5 {
        return Err(Error::InvalidOperation(format!(
            "adain_filter_latent expects 5D NCFHW tensors, got latents={:?} ref={:?}",
            ldims, rdims
        )));
    }
    if ldims[0] != rdims[0] || ldims[1] != rdims[1] {
        return Err(Error::InvalidOperation(format!(
            "adain_filter_latent (B, C) mismatch: latents={:?} vs ref={:?}",
            ldims, rdims
        )));
    }

    let (b, c, lf, lh, lw) = (ldims[0], ldims[1], ldims[2], ldims[3], ldims[4]);
    let (rf, rh, rw) = (rdims[2], rdims[3], rdims[4]);

    let orig_dtype = latents.dtype();

    // Compute stats in F32 for numeric stability. Torch's default std is
    // unbiased (N-1); match that.
    let l32 = latents.to_dtype(DType::F32)?;
    let r32 = reference_latents.to_dtype(DType::F32)?;

    let l_numel = (lf * lh * lw) as f32;
    let r_numel = (rf * rh * rw) as f32;

    // mean over axes (2, 3, 4), keepdim → [B, C, 1, 1, 1]
    let l_mean = l32.mean_dim(&[2, 3, 4], true)?;
    let r_mean = r32.mean_dim(&[2, 3, 4], true)?;

    // Unbiased variance: sum((x - mean)^2) / (N - 1)
    let l_centered = l32.sub(&l_mean)?;
    let r_centered = r32.sub(&r_mean)?;

    let l_var_sum = l_centered.square()?.sum_dims(&[2, 3, 4])?; // [B, C]
    let r_var_sum = r_centered.square()?.sum_dims(&[2, 3, 4])?;

    let l_std_unbiased = l_var_sum
        .mul_scalar(1.0 / (l_numel - 1.0).max(1.0))?
        .sqrt()?
        .reshape(&[b, c, 1, 1, 1])?;
    let r_std_unbiased = r_var_sum
        .mul_scalar(1.0 / (r_numel - 1.0).max(1.0))?
        .sqrt()?
        .reshape(&[b, c, 1, 1, 1])?;

    // matched = (l - l_mean) / l_std * r_std + r_mean.
    // Guard against div-by-zero when a channel is fully uniform (std=0).
    // Torch silently produces NaN in that case; for parity we add a tiny
    // epsilon — the reference never has zero-std channels in practice.
    let l_std_safe = l_std_unbiased.add_scalar(1e-20)?;
    let normalized = l_centered.div(&l_std_safe)?; // [B, C, F, H, W]
    let matched = normalized.mul(&r_std_unbiased)?.add(&r_mean)?;

    // lerp(latents, matched, factor) = latents + factor * (matched - latents)
    let diff = matched.sub(&l32)?;
    let out = l32.add(&diff.mul_scalar(factor)?)?;

    out.to_dtype(orig_dtype)
}

/// Tone-map latents — sigmoid-based amplitude compression.
///
/// Mirrors `LTXVideoPipeline.tone_map_latents`:
///
/// ```text
/// cs          = compression * 0.75
/// scales      = 1 - 0.8 * cs * sigmoid(4 * cs * (|x| - 1))
/// out         = x * scales
/// ```
///
/// `compression=0.0` is identity (by construction: the scalar multiplier
/// in front of sigmoid is zero, so `scales=1`). `compression=1.0` gives
/// the strongest compression (up to a factor of `1 - 0.6`). Used by the
/// distilled pipelines with compression=0.6 on the pre-decode latent.
pub fn tone_map_latents(latents: &Tensor, compression: f32) -> Result<Tensor> {
    if !(0.0..=1.0).contains(&compression) {
        return Err(Error::InvalidOperation(format!(
            "tone_map_latents compression must be in [0, 1], got {compression}"
        )));
    }

    let orig_dtype = latents.dtype();
    let cs = compression * 0.75;

    if compression == 0.0 {
        // Identity — scales is exactly 1 everywhere. Clone to keep the
        // contract that we always return a fresh tensor.
        return latents.clone_result();
    }

    // Compute in F32 for stability of the sigmoid term.
    let x = latents.to_dtype(DType::F32)?;

    // sigmoid_term = sigmoid(4 * cs * (|x| - 1))
    let abs_x = x.abs()?;
    let arg = abs_x.add_scalar(-1.0)?.mul_scalar(4.0 * cs)?;
    let sig = arg.sigmoid()?;

    // scales = 1 - 0.8 * cs * sigmoid_term
    let scales = sig.mul_scalar(-0.8 * cs)?.add_scalar(1.0)?;

    let filtered = x.mul(&scales)?;
    filtered.to_dtype(orig_dtype)
}
