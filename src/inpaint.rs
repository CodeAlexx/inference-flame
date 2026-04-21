//! Inpainting helpers for inference-flame: image+mask loading, latent/pixel
//! mask preparation, LanPaint per-step wrapping, and final pixel-space blend.
//!
//! This module is reference-implementation glue: model-specific bins
//! (`flux_inpaint`, etc.) call into it to avoid duplicating image/mask
//! munging code. It does NOT decide which model or which schedule to use —
//! the caller wires that, and only delegates the LanPaint step itself.
//!
//! Mask convention (matches `lanpaint-flame`):
//!   1.0 = preserve / known region
//!   0.0 = inpaint / unknown region
//!
//! Latent mask shape mirrors `latent_image` shape (`[B, C, H_lat, W_lat]`):
//! the single-channel pixel mask is replicated along the channel axis.
//!
//! Pixel mask shape is `[H, W]` at the original (output) image resolution
//! and is used only for the final RGB blend after VAE decode.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use image::imageops::FilterType;

use lanpaint_flame::LanPaint;

/// Bundled inpaint inputs ready to feed into a denoise loop.
///
/// `latent_image` and `latent_mask` are at latent resolution and consumed by
/// `lanpaint_step` on every step. `pixel_mask` and `input_image` are at
/// output (pixel) resolution and consumed by `blend_output` exactly once
/// after VAE decode.
pub struct InpaintInputs {
    /// Latent of the input image, shape `[B, C, H_lat, W_lat]` BF16.
    pub latent_image: Tensor,
    /// Latent-space mask, same shape as `latent_image`, values in {0.0, 1.0}.
    /// Convention: 1.0 = preserve (known region), 0.0 = inpaint (unknown).
    pub latent_mask: Tensor,
    /// Pixel-space mask at original image resolution, `[H, W]` f32 in {0, 1}.
    /// Used for the final pixel-space output blend.
    pub pixel_mask: Tensor,
    /// Original input image as RGB `[3, H, W]` f32 in `[-1, 1]`.
    /// Used for the final pixel-space output blend.
    pub input_image: Tensor,
}

/// Static configuration: input paths and output geometry.
pub struct InpaintConfig {
    /// Path to the input image (PNG/JPG/WEBP).
    pub image_path: PathBuf,
    /// Path to the mask (PNG/JPG; grayscale or RGB; threshold at >0.5 = inpaint).
    pub mask_path: PathBuf,
    /// VAE downscale factor at the LATENT level the model operates on.
    /// FLUX after pack uses `vae_scale = 16` (8× VAE + 2× patchify); plain
    /// LDM-style latents use `vae_scale = 8`.
    pub vae_scale: usize,
    /// Output (target) pixel resolution. Image and mask are resized to this.
    pub width: usize,
    pub height: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load image+mask from disk, resize to `(width, height)`, encode the image
/// with the supplied VAE-encode closure, threshold the mask, downscale it by
/// `vae_scale` (nearest-neighbor for crisp edges), and broadcast it across
/// the latent channels.
///
/// `vae_encode`: takes RGB `[1, 3, H, W]` BF16 in `[-1, 1]` and returns the
/// model's latent `[1, C, H_lat, W_lat]` BF16. The closure is called exactly
/// once.
///
/// Mask threshold: pixel value `> 0.5` → inpaint (0.0), `<= 0.5` → preserve
/// (1.0). Caller controls the convention by inverting the input mask if
/// needed.
pub fn prepare_inpaint(
    cfg: &InpaintConfig,
    device: Arc<CudaDevice>,
    vae_encode: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<InpaintInputs> {
    if cfg.width == 0 || cfg.height == 0 {
        return Err(anyhow!(
            "InpaintConfig: width/height must be > 0, got {}x{}",
            cfg.width,
            cfg.height
        ));
    }
    if cfg.vae_scale == 0 {
        return Err(anyhow!("InpaintConfig: vae_scale must be > 0"));
    }
    if cfg.width % cfg.vae_scale != 0 || cfg.height % cfg.vae_scale != 0 {
        return Err(anyhow!(
            "InpaintConfig: width ({}) and height ({}) must be divisible by vae_scale ({})",
            cfg.width,
            cfg.height,
            cfg.vae_scale
        ));
    }

    // ----- Image: load → resize Lanczos3 → BF16 NCHW in [-1, 1] -----
    let input_image_chw = load_image_chw(&cfg.image_path, cfg.width, cfg.height)?;
    let input_image_bf16 = upload_image_bf16(&input_image_chw, cfg.height, cfg.width, &device)?;
    // Keep an f32 [3, H, W] copy for the final pixel blend (lives on GPU).
    let input_image_f32 = upload_image_f32_chw(&input_image_chw, cfg.height, cfg.width, &device)?;

    // ----- VAE encode → latent [1, C, H/8 (typically), W/8] -----
    let latent_image = vae_encode(&input_image_bf16)?;
    let latent_dims = latent_image.shape().dims().to_vec();
    if latent_dims.len() != 4 {
        return Err(anyhow!(
            "vae_encode must return 4D [B, C, H, W]; got {:?}",
            latent_dims
        ));
    }
    let (lb, lc, lh, lw) = (latent_dims[0], latent_dims[1], latent_dims[2], latent_dims[3]);

    // ----- Mask: load grayscale → resize for pixel mask + latent mask -----
    let pixel_mask =
        load_mask_pixel(&cfg.mask_path, cfg.width, cfg.height, &device)?;

    // Latent-mask resolution. We want the mask aligned with the LATENT shape
    // the LanPaint sampler operates on. The supplied closure returned a
    // latent of shape [B, C, lh, lw]; that's the source of truth — derive
    // the mask shape from it, not from `vae_scale * pixel`. (For FLUX on
    // 1024×1024, the unpacked latent is 128×128 and `vae_scale=8` is what
    // the caller passes for the *pixel→latent* downscale. The packed-token
    // grid at 64×64 only matters for the model's pack/unpack, not here.)
    let latent_mask = load_mask_latent(&cfg.mask_path, lh, lw, lb, lc, &device)?;

    Ok(InpaintInputs {
        latent_image,
        latent_mask,
        pixel_mask,
        input_image: input_image_f32,
    })
}

/// Final pixel-space blend after VAE decode:
///
/// ```text
/// output[pixel_mask == 1.0]  = input_image      (preserve known)
/// output[pixel_mask == 0.0]  = decoded_image    (inpainted)
/// ```
///
/// Returns RGB `[3, H, W]` f32 in `[-1, 1]`.
///
/// All inputs must be on the same device. `pixel_mask` is `[H, W]`; it is
/// broadcast across channels via `where_mask`.
pub fn blend_output(
    decoded_image: &Tensor,
    input_image: &Tensor,
    pixel_mask: &Tensor,
) -> Result<Tensor> {
    let dec_dims = decoded_image.shape().dims();
    let inp_dims = input_image.shape().dims();
    let m_dims = pixel_mask.shape().dims();

    if dec_dims != inp_dims {
        return Err(anyhow!(
            "blend_output: decoded {:?} and input {:?} shapes differ",
            dec_dims,
            inp_dims
        ));
    }
    if dec_dims.len() != 3 {
        return Err(anyhow!(
            "blend_output: expected [3, H, W], got {:?}",
            dec_dims
        ));
    }
    let (c, h, w) = (dec_dims[0], dec_dims[1], dec_dims[2]);
    if c != 3 {
        return Err(anyhow!("blend_output: expected 3 channels, got {}", c));
    }
    if m_dims.len() != 2 || m_dims[0] != h || m_dims[1] != w {
        return Err(anyhow!(
            "blend_output: pixel_mask must be [{}, {}], got {:?}",
            h,
            w,
            m_dims
        ));
    }

    // Promote everything to F32 in [-1, 1] for an arithmetic blend (avoids a
    // dependency on `where_mask` broadcasting from [H,W] to [3,H,W], which
    // would need explicit replication).
    let dec_f32 = ensure_f32(decoded_image)?;
    let inp_f32 = ensure_f32(input_image)?;
    let mask_f32 = ensure_f32(pixel_mask)?;

    // Replicate mask to [3, H, W] manually so the blend is shape-exact.
    let mask_chw = mask_f32.reshape(&[1, h, w])?.broadcast_to(&Shape::from_dims(&[c, h, w]))?;

    // out = mask * input + (1 - mask) * decoded
    let one = Tensor::ones_dtype(
        mask_chw.shape().clone(),
        mask_chw.dtype(),
        mask_chw.device().clone(),
    )
    .map_err(|e| anyhow!("ones_dtype: {e:?}"))?;
    let one_minus = one.sub(&mask_chw).map_err(|e| anyhow!("sub: {e:?}"))?;
    let kept = mask_chw.mul(&inp_f32).map_err(|e| anyhow!("mul kept: {e:?}"))?;
    let painted = one_minus
        .mul(&dec_f32)
        .map_err(|e| anyhow!("mul painted: {e:?}"))?;
    kept.add(&painted).map_err(|e| anyhow!("add blend: {e:?}"))
}

/// Wrap one outer denoise step with LanPaint's Langevin inner loop.
///
/// The caller MUST thread the returned `advanced_x` back as `x` on the next
/// call — see `LanPaint::run` docs for why.
///
/// `inputs.latent_image` and `inputs.latent_mask` must match `x`'s shape
/// (typically `[B, C, H_lat, W_lat]` BF16). `noise` is the fixed seed noise
/// also at that shape. `sigma`, `abt`, `tflow` are `[B]` BF16 scalars
/// computed by the caller from the current step's `t_curr` (see
/// `flux_inpaint.rs` for the flow-matching mapping).
pub fn lanpaint_step<'a>(
    lanpaint: &LanPaint<'a>,
    x: &Tensor,
    inputs: &InpaintInputs,
    noise: &Tensor,
    sigma: &Tensor,
    abt: &Tensor,
    tflow: &Tensor,
) -> Result<(Tensor, Tensor)> {
    lanpaint
        .run(
            x,
            &inputs.latent_image,
            noise,
            sigma,
            abt,
            tflow,
            &inputs.latent_mask,
        )
        .map_err(|e| anyhow!("LanPaint::run failed: {e:?}"))
}

// ---------------------------------------------------------------------------
// Internal: image / mask loading helpers
// ---------------------------------------------------------------------------

/// Load an image, resize to `(target_w, target_h)` Lanczos3, return CHW
/// f32 in `[-1, 1]`. Stays on CPU.
fn load_image_chw(path: &std::path::Path, target_w: usize, target_h: usize) -> Result<Vec<f32>> {
    let img = image::io::Reader::open(path)
        .map_err(|e| anyhow!("open {}: {e}", path.display()))?
        .with_guessed_format()
        .map_err(|e| anyhow!("guess format {}: {e}", path.display()))?
        .decode()
        .map_err(|e| anyhow!("decode {}: {e}", path.display()))?
        .to_rgb8();

    // Resize directly to target (caller already chose dimensions).
    let resized = if (img.width(), img.height()) == (target_w as u32, target_h as u32) {
        img
    } else {
        image::imageops::resize(
            &img,
            target_w as u32,
            target_h as u32,
            FilterType::Lanczos3,
        )
    };

    let mut data = vec![0.0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = (p.0[c] as f32) / 127.5 - 1.0;
                data[c * target_h * target_w + y * target_w + x] = v;
            }
        }
    }
    Ok(data)
}

/// Upload CHW f32 -> [1, 3, H, W] BF16 GPU tensor.
fn upload_image_bf16(
    chw: &[f32],
    h: usize,
    w: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    Tensor::from_f32_to_bf16(
        chw.to_vec(),
        Shape::from_dims(&[1, 3, h, w]),
        device.clone(),
    )
    .map_err(|e| anyhow!("upload BF16 image: {e:?}"))
}

/// Upload CHW f32 -> [3, H, W] F32 GPU tensor (for the pixel blend).
fn upload_image_f32_chw(
    chw: &[f32],
    h: usize,
    w: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    Tensor::from_vec(
        chw.to_vec(),
        Shape::from_dims(&[3, h, w]),
        device.clone(),
    )
    .map_err(|e| anyhow!("upload F32 image: {e:?}"))
}

/// Load mask, resize to `(target_w, target_h)` Lanczos3, threshold at >0.5
/// → 0.0 (inpaint), else 1.0 (preserve). Returns `[H, W]` F32 GPU tensor.
fn load_mask_pixel(
    path: &std::path::Path,
    target_w: usize,
    target_h: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let m = image::io::Reader::open(path)
        .map_err(|e| anyhow!("open mask {}: {e}", path.display()))?
        .with_guessed_format()
        .map_err(|e| anyhow!("guess mask format {}: {e}", path.display()))?
        .decode()
        .map_err(|e| anyhow!("decode mask {}: {e}", path.display()))?
        .to_luma8();

    let resized = if (m.width(), m.height()) == (target_w as u32, target_h as u32) {
        m
    } else {
        image::imageops::resize(
            &m,
            target_w as u32,
            target_h as u32,
            FilterType::Lanczos3,
        )
    };

    let mut data = vec![0.0f32; target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let v = resized.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            // White (>0.5) = inpaint = 0.0. Black (<=0.5) = preserve = 1.0.
            data[y * target_w + x] = if v > 0.5 { 0.0 } else { 1.0 };
        }
    }
    Tensor::from_vec(
        data,
        Shape::from_dims(&[target_h, target_w]),
        device.clone(),
    )
    .map_err(|e| anyhow!("upload pixel mask: {e:?}"))
}

/// Load mask, resize nearest-neighbor to `(latent_w, latent_h)` for crisp
/// edges, threshold, broadcast across `[B, C, H_lat, W_lat]`.
fn load_mask_latent(
    path: &std::path::Path,
    latent_h: usize,
    latent_w: usize,
    batch: usize,
    channels: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let m = image::io::Reader::open(path)
        .map_err(|e| anyhow!("open mask {}: {e}", path.display()))?
        .with_guessed_format()
        .map_err(|e| anyhow!("guess mask format {}: {e}", path.display()))?
        .decode()
        .map_err(|e| anyhow!("decode mask {}: {e}", path.display()))?
        .to_luma8();

    // Nearest-neighbor avoids softening mask boundaries — important for
    // sharp inpaint edges. Resize directly to LATENT resolution.
    let resized = if (m.width(), m.height()) == (latent_w as u32, latent_h as u32) {
        m
    } else {
        image::imageops::resize(
            &m,
            latent_w as u32,
            latent_h as u32,
            FilterType::Nearest,
        )
    };

    let mut data = vec![0.0f32; batch * channels * latent_h * latent_w];
    for y in 0..latent_h {
        for x in 0..latent_w {
            let v = resized.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            // White (>0.5) = inpaint = 0.0. Black (<=0.5) = preserve = 1.0.
            let bit = if v > 0.5 { 0.0 } else { 1.0 };
            // Replicate across batch+channels.
            for b in 0..batch {
                for c in 0..channels {
                    let idx = ((b * channels + c) * latent_h + y) * latent_w + x;
                    data[idx] = bit;
                }
            }
        }
    }

    Tensor::from_f32_to_bf16(
        data,
        Shape::from_dims(&[batch, channels, latent_h, latent_w]),
        device.clone(),
    )
    .map_err(|e| anyhow!("upload latent mask: {e:?}"))
}

/// Cast to F32 if not already. Cheap when already F32.
fn ensure_f32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        t.clone_result().map_err(|e| anyhow!("clone f32: {e:?}"))
    } else {
        t.to_dtype(DType::F32)
            .map_err(|e| anyhow!("cast to f32: {e:?}"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These exercise the pure-tensor blend logic, not the full pipeline. The
// VAE-encode integration is left to bin-level smoke runs because it requires
// real model weights.

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::global_cuda_device;

    /// Build a [3, H, W] F32 tensor from row-major channel-first data.
    fn img_chw(data: Vec<f32>, h: usize, w: usize) -> Tensor {
        let dev = global_cuda_device();
        Tensor::from_vec(data, Shape::from_dims(&[3, h, w]), dev).unwrap()
    }

    fn mask_hw(data: Vec<f32>, h: usize, w: usize) -> Tensor {
        let dev = global_cuda_device();
        Tensor::from_vec(data, Shape::from_dims(&[h, w]), dev).unwrap()
    }

    /// 4×4 checkerboard mask: known pixels (mask=1) keep `input_image`,
    /// unknown pixels (mask=0) keep `decoded_image`. Hand-verify by
    /// reconstructing the expected per-pixel value.
    #[test]
    fn blend_output_checker_4x4() {
        let h = 4;
        let w = 4;
        // Input image: every channel is +1.0 everywhere.
        let inp = img_chw(vec![1.0f32; 3 * h * w], h, w);
        // Decoded image: every channel is -1.0 everywhere.
        let dec = img_chw(vec![-1.0f32; 3 * h * w], h, w);
        // Mask: 1.0 on (y+x) even, 0.0 elsewhere.
        let mut m = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                m[y * w + x] = if (y + x) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let mask = mask_hw(m, h, w);

        let out = blend_output(&dec, &inp, &mask).expect("blend_output");
        let out_f32 = out.to_vec_f32().expect("to_vec_f32");
        assert_eq!(out_f32.len(), 3 * h * w);

        // Build expected: per-pixel = if mask==1 then +1 (input) else -1 (decoded).
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let idx = c * h * w + y * w + x;
                    let want = if (y + x) % 2 == 0 { 1.0f32 } else { -1.0f32 };
                    assert!(
                        (out_f32[idx] - want).abs() < 1e-5,
                        "blend mismatch at c={c} y={y} x={x}: got {} want {}",
                        out_f32[idx],
                        want
                    );
                }
            }
        }
    }

    /// Mask all-zero → output = decoded everywhere (full inpaint).
    #[test]
    fn blend_output_all_zero_mask_returns_decoded() {
        let h = 2;
        let w = 2;
        let inp = img_chw(vec![0.5f32; 3 * h * w], h, w);
        let dec = img_chw(vec![-0.7f32; 3 * h * w], h, w);
        let mask = mask_hw(vec![0.0f32; h * w], h, w);

        let out = blend_output(&dec, &inp, &mask).expect("blend_output");
        let out_f32 = out.to_vec_f32().expect("to_vec_f32");
        for &v in &out_f32 {
            assert!((v - (-0.7)).abs() < 1e-5, "expected -0.7, got {}", v);
        }
    }

    /// Mask all-one → output = input everywhere (no inpaint).
    #[test]
    fn blend_output_all_one_mask_returns_input() {
        let h = 2;
        let w = 2;
        let inp = img_chw(vec![0.3f32; 3 * h * w], h, w);
        let dec = img_chw(vec![-0.9f32; 3 * h * w], h, w);
        let mask = mask_hw(vec![1.0f32; h * w], h, w);

        let out = blend_output(&dec, &inp, &mask).expect("blend_output");
        let out_f32 = out.to_vec_f32().expect("to_vec_f32");
        for &v in &out_f32 {
            assert!((v - 0.3).abs() < 1e-5, "expected 0.3, got {}", v);
        }
    }

    /// Shape mismatch is rejected.
    #[test]
    fn blend_output_rejects_mismatched_shapes() {
        let inp = img_chw(vec![0.0f32; 3 * 4 * 4], 4, 4);
        let dec = img_chw(vec![0.0f32; 3 * 2 * 2], 2, 2);
        let mask = mask_hw(vec![0.0f32; 4 * 4], 4, 4);
        assert!(blend_output(&dec, &inp, &mask).is_err());
    }
}
