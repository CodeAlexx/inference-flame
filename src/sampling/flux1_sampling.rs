//! FLUX 1 Dev sampling helpers — verbatim port from BFL.
//!
//! Source of truth: `black-forest-labs-flux/src/flux/sampling.py`
//!
//! This module is ONLY for FLUX 1 (Dev/Schnell). It does NOT touch Klein /
//! flux2 codepaths. FLUX 1 Dev is a guidance-distilled model: the denoise loop
//! is a plain Euler pass with `guidance_vec = ones * guidance` fed to the DiT
//! as an additional conditioning input — there is NO classifier-free guidance
//! (no two forwards per step).

use flame_core::{DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Time-shift schedule
// ---------------------------------------------------------------------------

/// BFL `time_shift` — flux/sampling.py:277-278
/// ```python
/// def time_shift(mu: float, sigma: float, t: Tensor):
///     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
/// ```
pub fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let em = mu.exp();
    em / (em + (1.0 / t - 1.0).powf(sigma))
}

/// BFL `get_lin_function` with FLUX 1 defaults — flux/sampling.py:281-286
/// ```python
/// def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
///     m = (y2 - y1) / (x2 - x1)
///     b = y1 - m * x1
///     return lambda x: m * x + b
/// ```
/// Returns the mu value for a given image sequence length. FLUX 1 defaults:
/// base_shift = 0.5 @ x1=256, max_shift = 1.15 @ x2=4096.
pub fn flux1_mu(image_seq_len: usize) -> f64 {
    let (x1, y1) = (256.0f64, 0.5f64); // base_shift
    let (x2, y2) = (4096.0f64, 1.15f64); // max_shift
    let m = (y2 - y1) / (x2 - x1);
    let b = y1 - m * x1;
    m * image_seq_len as f64 + b
}

/// BFL `get_schedule` — flux/sampling.py:289-305
/// ```python
/// def get_schedule(num_steps, image_seq_len,
///                  base_shift: float = 0.5,
///                  max_shift: float = 1.15,
///                  shift: bool = True) -> list[float]:
///     timesteps = torch.linspace(1, 0, num_steps + 1)
///     if shift:
///         mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
///         timesteps = time_shift(mu, 1.0, timesteps)
///     return timesteps.tolist()
/// ```
///
/// Returns `num_steps + 1` timesteps, `t[0] ≈ 1.0`, `t[-1] == 0.0`.
pub fn get_schedule(
    num_steps: usize,
    image_seq_len: usize,
    _base_shift: f64,
    _max_shift: f64,
    shift: bool,
) -> Vec<f32> {
    // linspace(1, 0, num_steps + 1)
    let mut ts: Vec<f64> = (0..=num_steps)
        .map(|i| 1.0 - (i as f64) / (num_steps as f64))
        .collect();

    if shift {
        let mu = flux1_mu(image_seq_len);
        for t in ts.iter_mut() {
            // Avoid division by zero at endpoints; t=0 stays 0, t=1 stays 1.
            if *t > 0.0 && *t < 1.0 {
                *t = time_shift(mu, 1.0, *t);
            }
        }
    }

    ts.into_iter().map(|v| v as f32).collect()
}

// ---------------------------------------------------------------------------
// Pack / unpack latents — BFL `prepare` / `unpack`
// ---------------------------------------------------------------------------

/// BFL `prepare` packing — flux/sampling.py:41-48
/// ```python
/// img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
/// img_ids = torch.zeros(h // 2, w // 2, 3)
/// img_ids[..., 1] += torch.arange(h // 2)[:, None]
/// img_ids[..., 2] += torch.arange(w // 2)[None, :]
/// img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
/// ```
///
/// Input:  `[B, C, H, W]` latent tensor, H and W MUST be even.
/// Output: `(packed, img_ids)` where
///   - `packed`  has shape `[B, (H/2)*(W/2), C*4]`
///   - `img_ids` has shape `[(H/2)*(W/2), 3]`, dtype BF16, with axis 0 = 0,
///     axis 1 = h_idx, axis 2 = w_idx. Flattened row-major over (h2, w2)
///     to match the `b (h w) c` rearrange order.
pub fn pack_latent(
    z: &Tensor,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let d = z.shape().dims().to_vec();
    if d.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "pack_latent expects 4D [B,C,H,W], got {:?}",
            d
        )));
    }
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    if h % 2 != 0 || w % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "pack_latent: H and W must be even, got H={} W={}",
            h, w
        )));
    }
    let (h2, w2) = (h / 2, w / 2);

    // "b c (h ph) (w pw) -> b (h w) (c ph pw)" with ph=pw=2
    //
    // Walk through the einops axes:
    //   reshape  [B, C, h2, 2, w2, 2]    (split H→(h2,ph) and W→(w2,pw))
    //   permute  [B, h2, w2, C, 2, 2]    (bring h2,w2 out front; keep c before ph,pw)
    //   reshape  [B, h2*w2, C*2*2]
    let t = z.reshape(&[b, c, h2, 2, w2, 2])?;
    let t = t.permute(&[0, 2, 4, 1, 3, 5])?;
    let packed = t.reshape(&[b, h2 * w2, c * 4])?;

    // Build img_ids: [h2*w2, 3] with (0, r, c) entries
    let n = h2 * w2;
    let mut data = vec![0.0f32; n * 3];
    for r in 0..h2 {
        for col in 0..w2 {
            let idx = r * w2 + col;
            data[idx * 3] = 0.0; // axis 0
            data[idx * 3 + 1] = r as f32; // axis 1 = h_idx
            data[idx * 3 + 2] = col as f32; // axis 2 = w_idx
        }
    }
    let img_ids = Tensor::from_f32_to_bf16(data, Shape::from_dims(&[n, 3]), device.clone())?;

    Ok((packed, img_ids))
}

/// BFL `unpack` — flux/sampling.py:356-364
/// ```python
/// def unpack(x: Tensor, height: int, width: int) -> Tensor:
///     return rearrange(
///         x,
///         "b (h w) (c ph pw) -> b c (h ph) (w pw)",
///         h=math.ceil(height / 16),
///         w=math.ceil(width / 16),
///         ph=2,
///         pw=2,
///     )
/// ```
///
/// `packed` has shape `[B, h2*w2, C*4]`. `target_height`/`target_width` are
/// the requested **pixel** dimensions: the latent is then `2*ceil(H/16) x
/// 2*ceil(W/16)` which means h2 = ceil(H/16), w2 = ceil(W/16).
/// Returns `[B, C, 2*h2, 2*w2]`.
pub fn unpack_latent(packed: &Tensor, target_height: usize, target_width: usize) -> Result<Tensor> {
    let d = packed.shape().dims().to_vec();
    if d.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "unpack_latent expects 3D [B,N,C*4], got {:?}",
            d
        )));
    }
    let b = d[0];
    let n = d[1];
    let c4 = d[2];
    if c4 % 4 != 0 {
        return Err(Error::InvalidInput(format!(
            "unpack_latent: last dim must be C*4, got {}",
            c4
        )));
    }
    let c = c4 / 4;
    let h2 = (target_height + 15) / 16;
    let w2 = (target_width + 15) / 16;
    if h2 * w2 != n {
        return Err(Error::InvalidInput(format!(
            "unpack_latent: h2*w2 ({}*{}={}) does not match N ({})",
            h2,
            w2,
            h2 * w2,
            n
        )));
    }

    // Inverse of pack:
    //   reshape [B, h2, w2, C, 2, 2]
    //   permute [B, C, h2, 2, w2, 2]       (dims 0,3,1,4,2,5)
    //   reshape [B, C, h2*2, w2*2]
    let t = packed.reshape(&[b, h2, w2, c, 2, 2])?;
    let t = t.permute(&[0, 3, 1, 4, 2, 5])?;
    let out = t.reshape(&[b, c, h2 * 2, w2 * 2])?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Denoise loop — plain Euler with guidance vector input (no CFG)
// ---------------------------------------------------------------------------

/// BFL `denoise` — flux/sampling.py:308-353 (simplified, no kontext/conditioning)
/// ```python
/// def denoise(model, img, img_ids, txt, txt_ids, vec, timesteps, guidance=4.0, ...):
///     guidance_vec = torch.full((img.shape[0],), guidance, device=..., dtype=img.dtype)
///     for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
///         t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=...)
///         pred = model(
///             img=img, img_ids=img_ids,
///             txt=txt, txt_ids=txt_ids,
///             y=vec,
///             timesteps=t_vec,
///             guidance=guidance_vec,
///         )
///         img = img + (t_prev - t_curr) * pred
///     return img
/// ```
///
/// FLUX 1 Dev is guidance-distilled — the `guidance` scalar is a model input,
/// NOT used for classifier-free guidance. Only one forward per step.
///
/// `model_fn` closure: `(img, t_vec, guidance_vec) -> velocity pred`.
pub fn flux1_denoise<F>(
    mut model_fn: F,
    mut img: Tensor,
    timesteps: &[f32],
    guidance: f32,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor>
where
    F: FnMut(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
{
    let batch = img.shape().dims()[0];
    let dtype = img.dtype();

    // guidance_vec = full((B,), guidance, dtype=img.dtype)
    let guidance_vec = Tensor::from_vec(
        vec![guidance; batch],
        Shape::from_dims(&[batch]),
        device.clone(),
    )?;
    let guidance_vec = if dtype != DType::F32 {
        guidance_vec.to_dtype(dtype)?
    } else {
        guidance_vec
    };

    for w in timesteps.windows(2) {
        let (t_curr, t_prev) = (w[0], w[1]);
        // t_vec = full((B,), t_curr, dtype=img.dtype)
        let t_vec = Tensor::from_vec(
            vec![t_curr; batch],
            Shape::from_dims(&[batch]),
            device.clone(),
        )?;
        let t_vec = if dtype != DType::F32 {
            t_vec.to_dtype(dtype)?
        } else {
            t_vec
        };

        let pred = model_fn(&img, &t_vec, &guidance_vec)?;
        // img = img + (t_prev - t_curr) * pred
        img = img.add(&pred.mul_scalar(t_prev - t_curr)?)?;
    }

    Ok(img)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_length_and_endpoints() {
        let ts = get_schedule(20, 4096, 0.5, 1.15, true);
        assert_eq!(ts.len(), 21);
        assert!((ts[0] - 1.0).abs() < 1e-5);
        assert!(ts[20].abs() < 1e-5);
    }

    #[test]
    fn schedule_monotonic_decreasing() {
        let ts = get_schedule(20, 4096, 0.5, 1.15, true);
        for i in 0..ts.len() - 1 {
            assert!(ts[i] >= ts[i + 1], "not monotonic at {}", i);
        }
    }

    #[test]
    fn flux1_mu_1024_image() {
        // 1024x1024 → 128x128 latent → 64x64 packed = 4096 tokens → mu = 1.15
        let mu = flux1_mu(4096);
        assert!((mu - 1.15).abs() < 1e-6, "expected 1.15, got {}", mu);
    }

    #[test]
    fn flux1_mu_256_ref() {
        // image_seq_len = 256 → mu = 0.5
        let mu = flux1_mu(256);
        assert!((mu - 0.5).abs() < 1e-6, "expected 0.5, got {}", mu);
    }

    #[test]
    fn time_shift_fixed_point_half() {
        // When t = exp(mu) / (exp(mu) + 1) we should get back t (sigma=1)
        let mu = 1.15;
        let em = (mu as f64).exp();
        let t = em / (em + 1.0);
        let shifted = time_shift(mu, 1.0, t);
        assert!((shifted - t).abs() < 1e-10);
    }
}
