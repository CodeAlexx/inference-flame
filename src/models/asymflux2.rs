//! AsymFLUX.2 — AsymFlow velocity reconstruction layer.
//!
//! Port of LakonLab's `AsymFlowMixin` (Apache 2.0) from
//! `~/LakonLab/lakonlab/models/architectures/asymflow/common.py`.
//!
//! What this is: a stateless math layer that wraps a base Klein 9B
//! transformer's `proj_out` to produce a full-rank flow-matching velocity
//! from the transformer's "asymmetric" velocity output `u_a`. The wrapper
//! also handles per-step timestep/scale calibration and the input rescale
//! by `k` that the transformer's `x_embedder` expects.
//!
//! This is **model-specific composition of flame-core primitives**, not a
//! new primitive. Per `flame-core/TENETS.md` (tenet 1 + tenet 5), the math
//! lives here in `inference-flame/src/models/`, not in flame-core.
//!
//! DECISION: F32 throughout (input cast, all math, output cast). The
//! reference disables autocast for the entire velocity reconstruction with
//! `torch.autocast(..., enabled=False)`. This is the *algorithm's*
//! precision requirement, not a missing-BF16-kernel fallback — see
//! `flame-core/CLAUDE.md` "NEVER use F32 fallbacks in inference code." The
//! caller passes BF16 in, gets BF16 back; F32 is internal.
//!
//! DECISION: Calibration scalars (`s`, `k`, `cal_timestep`, `sigma`) are
//! plain f32 host values, not (B,1,1) tensors. inference-flame runs CFG by
//! issuing two separate forward passes (B=1 each), so per-batch broadcast
//! isn't needed. The Python (B,1,1) shape is for training where B>1.

use flame_core::{DType, Error, Result, Tensor};

/// Per-step calibration values for one AsymFlow inference step.
///
/// All fields are host-side f32 scalars. With batch size 1 (the inference
/// case) and the Klein 9B config `num_timesteps = 1`, the per-batch
/// broadcast that the reference applies collapses to a scalar multiply.
#[derive(Debug, Clone, Copy)]
pub struct AsymFlowCalibration {
    /// `scale_buffer` value (host scalar). Reference uses 1.0 by default;
    /// loaded from the adapter safetensors at runtime.
    pub s: f32,
    /// Calibration coefficient `k = 1 / (s + (1 - s) * sigma)`. Multiplied
    /// into the input before `x_embedder` and used as a coefficient in the
    /// velocity reconstruction.
    pub k: f32,
    /// `cal_timestep = timestep * k`. The transformer's
    /// `time_guidance_embed` consumes `cal_timestep * 1000`.
    pub cal_timestep: f32,
    /// `sigma = timestep / num_timesteps`. With `num_timesteps = 1` (Klein
    /// 9B AsymFLUX.2 config) this equals `timestep`.
    pub sigma: f32,
}

/// NaN-preserving lower clamp. `f32::max` follows libm `fmaxf` and
/// silently returns the non-NaN operand — squashing upstream NaN inputs.
/// The PyTorch reference uses `.clamp(min=...)` which propagates NaN; we
/// mirror that so upstream bugs (e.g. a corrupted timestep schedule)
/// surface instead of being silently zeroed.
#[inline]
fn clamp_sigma_nan_preserving(sigma: f32, sigma_min: f32) -> f32 {
    if sigma.is_nan() {
        sigma
    } else if sigma < sigma_min {
        sigma_min
    } else {
        sigma
    }
}

/// Compute the AsymFlow calibration for a single inference timestep.
///
/// `timestep` is in `[0, 1]` (the flow-matching sampler's sigma; the
/// reference pipeline calls `sigma = t / 1000` after scaling its `t` from
/// `[0, 1000]`). `num_timesteps` is the transformer's config field
/// (`1` for AsymFLUX.2 Klein 9B). `scale_buffer` is the host-scalar value
/// read from the adapter safetensors' `scale_buffer` tensor.
///
/// **Caller preconditions:**
/// - `num_timesteps > 0` (otherwise `sigma = ±Inf`).
/// - `scale_buffer > 0` OR `timestep > 0` (otherwise `k = 1/0 = Inf` →
///   `cal_timestep = 0 * Inf = NaN`).
///
/// This function does not validate inputs; bad calibrations propagate as
/// non-finite values into the velocity reconstruction so the upstream bug
/// is visible.
pub fn compute_calibration(
    timestep: f32,
    scale_buffer: f32,
    num_timesteps: f32,
) -> AsymFlowCalibration {
    let sigma = timestep / num_timesteps;
    let k = 1.0 / (scale_buffer + (1.0 - scale_buffer) * sigma);
    AsymFlowCalibration {
        s: scale_buffer,
        k,
        cal_timestep: timestep * k,
        sigma,
    }
}

/// Orthogonal decomposition of `state` along the column space of `P`.
///
/// Returns `(subspace, complement)` where:
/// - `subspace = state @ P @ P^T`
/// - `complement = state - subspace`
///
/// Shapes: `state` is `(B, hw, D)` F32; `P` is `(D, R)` F32 with `R <= D`;
/// `p_t` is `(R, D)` F32 — caller-materialized `P^T` (see note below).
/// Output tensors are both `(B, hw, D)` F32. `P` is treated as
/// orthonormal-column-wise (the Procrustes projection from the adapter
/// safetensors); the wrapper does not orthonormalize at runtime.
///
/// **Why `p_t` is a parameter, not computed inside:** flame-core's
/// `launch_gemm` reads storage via `try_as_slice_f32` and ignores custom
/// strides, so a `.transpose()` view would silently feed un-transposed
/// memory to cuBLAS. We materialize `P^T` once at the caller via
/// `p.transpose()?.contiguous()?` and reuse it across both decompositions
/// inside `asymflow_velocity` (twice — once for `u_a`, once for `x_t`).
fn orthogonal_decomposition(
    state: &Tensor,
    p: &Tensor,
    p_t: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // state @ P  →  (B, hw, R)
    let proj = state.matmul(p)?;
    // (B, hw, R) @ (R, D)  →  (B, hw, D)
    let subspace = proj.matmul(p_t)?;
    // state - subspace
    let complement = state.sub(&subspace)?;
    Ok((subspace, complement))
}

/// AsymFlow velocity reconstruction.
///
/// Combines the transformer's asymmetric velocity `u_a_packed` with the
/// per-step packed pixel tensor `x_t_packed` and the precomputed
/// calibration to produce the full flow-matching velocity `u`. All math
/// runs in F32 regardless of input dtype; the output is returned in F32
/// (callers cast back to their working dtype, matching the reference's
/// `output_packed.to(hidden_states.dtype)`).
///
/// Inputs:
/// - `u_a_packed`: `(B, hw, D)` — transformer output after `proj_out`,
///   any dtype. **Must be contiguous**: `launch_gemm` reads storage
///   directly and ignores custom strides, so a stride-view here would
///   silently compute on the un-permuted buffer.
/// - `x_t_packed`: `(B, hw, D)` — packed pre-`x_embedder` pixel tensor
///   (NOT the `* k`-scaled version), any dtype. Same contiguous
///   requirement.
/// - `calibration`: from [`compute_calibration`].
/// - `p`: `(D, R)` projection buffer, any dtype. Same contiguous
///   requirement.
/// - `sigma_min`: lower clamp for `sigma` in the divisor. AsymFLUX.2 Klein
///   inference uses `1e-4` (from the transformer's config `sigma_min`
///   field).
///
/// Output: F32 tensor with shape `(B, hw, D)`.
pub fn asymflow_velocity(
    u_a_packed: &Tensor,
    x_t_packed: &Tensor,
    calibration: &AsymFlowCalibration,
    p: &Tensor,
    sigma_min: f32,
) -> Result<Tensor> {
    // Cast everything to F32. `to_dtype` is a no-op when the dtype already
    // matches (returns a clone with the same storage view).
    let u_a = if u_a_packed.dtype() == DType::F32 {
        u_a_packed.clone()
    } else {
        u_a_packed.to_dtype(DType::F32)?
    };
    let x_t = if x_t_packed.dtype() == DType::F32 {
        x_t_packed.clone()
    } else {
        x_t_packed.to_dtype(DType::F32)?
    };
    let p_f32 = if p.dtype() == DType::F32 {
        p.clone()
    } else {
        p.to_dtype(DType::F32)?
    };

    // Materialize P^T once for both decompositions. See
    // `orthogonal_decomposition` doc for why this can't live inside the
    // helper (stride-view vs. cuBLAS storage-direct read).
    let p_t = p_f32.transpose()?.contiguous()?;

    // Orthogonal decomposition of both tensors.
    let (u_a_sub, u_a_comp) = orthogonal_decomposition(&u_a, &p_f32, &p_t)?;
    let (x_t_sub, x_t_comp) = orthogonal_decomposition(&x_t, &p_f32, &p_t)?;

    let sk = calibration.s * calibration.k;
    let sigma_clamped = clamp_sigma_nan_preserving(calibration.sigma, sigma_min);
    let inv_sigma = 1.0 / sigma_clamped;

    // Low-rank subspace velocity:
    //   u_subspace = sk * u_a_sub + (1 - sk) / sigma_clamped * x_t_sub
    let term1 = u_a_sub.mul_scalar(sk)?;
    let coef2 = (1.0 - sk) * inv_sigma;
    let term2 = x_t_sub.mul_scalar(coef2)?;
    let u_sub = term1.add(&term2)?;

    // Orthogonal complement velocity:
    //   u_complement = (x_t_comp + s * u_a_comp) / sigma_clamped
    let s_u_a_comp = u_a_comp.mul_scalar(calibration.s)?;
    let comp_sum = x_t_comp.add(&s_u_a_comp)?;
    let u_comp = comp_sum.mul_scalar(inv_sigma)?;

    // Full velocity.
    u_sub.add(&u_comp)
}

// =====================================================================
// Step 6 — Adapter weight loading.
// =====================================================================

/// Extract the AsymFlow `proj_buffer` and `scale_buffer` from an already-
/// loaded safetensors weight map.
///
/// `proj_buffer` is expected at shape `(in_channels * patch_size², base_rank)`,
/// which for AsymFLUX.2 Klein 9B is `(3 * 16² = 768, 128)`. Stored as F32
/// per the reference (`common.py::init_asymflow_buffers`).
///
/// `scale_buffer` is a scalar (zero-dim or 1-element F32 tensor). Default
/// value at init is `1.0`; the trained adapter overrides this.
///
/// We tolerate three key spellings to be robust to packaging variations:
/// - `proj_buffer` / `scale_buffer` (LakonLab native, matches the
///   `register_buffer` names in `common.py:23-24`)
/// - `transformer.proj_buffer` / `transformer.scale_buffer` (if the
///   adapter ships under a `transformer` prefix)
/// - `model.proj_buffer` / `model.scale_buffer` (diffusers prefix)
///
/// Returns `(proj_buffer F32 tensor, scale_buffer f32 host scalar)`.
pub fn extract_asymflow_buffers(
    weights: &std::collections::HashMap<String, Tensor>,
) -> Result<(Tensor, f32)> {
    let proj_keys = [
        "proj_buffer",
        "transformer.proj_buffer",
        "model.proj_buffer",
    ];
    let scale_keys = [
        "scale_buffer",
        "transformer.scale_buffer",
        "model.scale_buffer",
    ];

    let proj = proj_keys
        .iter()
        .find_map(|k| weights.get(*k))
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "extract_asymflow_buffers: no proj_buffer tensor found (tried {})",
                proj_keys.join(", ")
            ))
        })?;
    let scale = scale_keys
        .iter()
        .find_map(|k| weights.get(*k))
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "extract_asymflow_buffers: no scale_buffer tensor found (tried {})",
                scale_keys.join(", ")
            ))
        })?;

    // proj_buffer: cast to F32, ensure 2D, return as contiguous.
    let proj_f32 = if proj.dtype() == DType::F32 {
        proj.clone()
    } else {
        proj.to_dtype(DType::F32)?
    };
    if proj_f32.shape().dims().len() != 2 {
        return Err(Error::InvalidInput(format!(
            "extract_asymflow_buffers: proj_buffer must be rank-2, got shape {:?}",
            proj_f32.shape().dims()
        )));
    }
    let proj_contig = if proj_f32.is_contiguous() {
        proj_f32
    } else {
        proj_f32.contiguous()?
    };

    // scale_buffer: collapse to host f32. Handles 0-dim and 1-element tensors.
    let scale_f32 = if scale.dtype() == DType::F32 {
        scale.clone()
    } else {
        scale.to_dtype(DType::F32)?
    };
    let scale_val = match scale_f32.shape().dims() {
        [] => scale_f32.item()?,
        [1] => scale_f32.item()?,
        dims => {
            return Err(Error::InvalidInput(format!(
                "extract_asymflow_buffers: scale_buffer must be scalar (0-dim) or 1-element, got shape {:?}",
                dims
            )));
        }
    };

    Ok((proj_contig, scale_val))
}

// =====================================================================
// Step 5 — FlowEulerODE scheduler + clamp-denoised round-trip.
// =====================================================================

/// Compute the sqrt-type dynamic shift used by `FlowEulerODEScheduler`
/// when `use_dynamic_shifting=True, dynamic_shifting_type='sqrt'`.
///
/// AsymFLUX.2 Klein 9B uses these constants (per the CC plan and matched
/// against `lakonlab/models/diffusions/schedulers/flow_euler_ode.py`):
/// - `base_shift = 17.0`, `max_shift = 34.0`
/// - `base_seq_len = 1024² = 1_048_576`, `max_seq_len = 2048² = 4_194_304`
/// - `seq_len = H * W` in **pixel space** (not patch space)
///
/// ```text
/// m = (max_shift - base_shift) / (sqrt(max_seq_len) - sqrt(base_seq_len))
/// shift = (sqrt(seq_len) - sqrt(base_seq_len)) * m + base_shift
/// ```
pub fn dynamic_shift_sqrt(
    seq_len: f32,
    base_shift: f32,
    max_shift: f32,
    base_seq_len: f32,
    max_seq_len: f32,
) -> f32 {
    let sqrt_seq = seq_len.sqrt();
    let sqrt_base = base_seq_len.sqrt();
    let sqrt_max = max_seq_len.sqrt();
    let m = (max_shift - base_shift) / (sqrt_max - sqrt_base);
    (sqrt_seq - sqrt_base) * m + base_shift
}

/// AsymFLUX.2 Klein 9B default dynamic shift constants.
pub const KLEIN_BASE_SHIFT: f32 = 17.0;
pub const KLEIN_MAX_SHIFT: f32 = 34.0;
pub const KLEIN_BASE_SEQ_LEN: f32 = 1024.0 * 1024.0;
pub const KLEIN_MAX_SEQ_LEN: f32 = 2048.0 * 2048.0;

/// Convenience wrapper: dynamic shift with AsymFLUX.2 Klein 9B defaults.
pub fn klein_dynamic_shift(h: usize, w: usize) -> f32 {
    dynamic_shift_sqrt(
        (h * w) as f32,
        KLEIN_BASE_SHIFT,
        KLEIN_MAX_SHIFT,
        KLEIN_BASE_SEQ_LEN,
        KLEIN_MAX_SEQ_LEN,
    )
}

/// Compute the flow-matching sigma schedule for inference.
///
/// `num_inference_steps` sigma values are returned, plus one trailing
/// `0.0` — totalling `num_inference_steps + 1` entries. Step `i` uses
/// `sigmas[i]` as the current noise level and `sigmas[i + 1]` as the
/// target.
///
/// Reference:
/// ```text
/// raw = linspace(1.0, 0.0, num_inference_steps, endpoint=False)
/// sigmas = shift * raw / (1 + (shift - 1) * raw)
/// # then concat a trailing 0.0
/// ```
///
/// `shift` is the dynamic-shift coefficient from [`klein_dynamic_shift`]
/// (or the static `shift` from config when not using dynamic shifting).
pub fn compute_sigma_schedule(num_inference_steps: usize, shift: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(num_inference_steps + 1);
    // `linspace(1.0, 0.0, N, endpoint=False)` = `1 - i / N` for `i in 0..N`.
    for i in 0..num_inference_steps {
        let raw = 1.0 - (i as f32) / (num_inference_steps as f32);
        let s = shift * raw / (1.0 + (shift - 1.0) * raw);
        sigmas.push(s);
    }
    sigmas.push(0.0);
    sigmas
}

/// Convert the sigma schedule to the timestep schedule the transformer
/// consumes. The reference scales by `num_train_timesteps = 1000` and
/// then divides by 1000 again inside the pipeline (`_t = t / 1000`) — we
/// short-circuit and pass `sigma` directly as the timestep (since
/// `num_timesteps = 1` in AsymFLUX.2 Klein, the calibration's
/// `sigma = timestep / num_timesteps` becomes identity).
pub fn timesteps_from_sigmas(sigmas: &[f32]) -> Vec<f32> {
    // For Klein AsymFLUX.2 with num_timesteps=1, timestep ≡ sigma in [0,1].
    // We drop the trailing 0.0 boundary value because the loop only
    // iterates N times (the boundary is the post-final state).
    sigmas[..sigmas.len().saturating_sub(1)].to_vec()
}

/// One Euler-ODE step: `x_t_next = x_t + model_output * (sigma_next - sigma_cur)`.
///
/// `model_output` is the velocity prediction `u` from
/// [`asymflow_velocity`] (after CFG mixing). `sigma_cur` and `sigma_next`
/// come from the schedule. `dt = sigma_next - sigma_cur` is negative
/// (sigma decreases over inference) so each step moves toward `x_0`.
///
/// All math runs at the dtype of `x_t`. The reference upcasts to F32 for
/// the step; here we follow the same pattern — the caller passes
/// model_output already in F32 (it's the AsymFlow output), and x_t is F32
/// per the pipeline's initial noise allocation.
pub fn euler_step(
    x_t: &Tensor,
    model_output: &Tensor,
    sigma_cur: f32,
    sigma_next: f32,
) -> Result<Tensor> {
    if x_t.shape().dims() != model_output.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "euler_step: x_t shape {:?} != model_output shape {:?}",
            x_t.shape().dims(),
            model_output.shape().dims()
        )));
    }
    let dt = sigma_next - sigma_cur;
    let delta = model_output.mul_scalar(dt)?;
    x_t.add(&delta)
}

/// Clamp the denoised prediction by round-tripping through Oklab encode
/// → decode → clamp(-1, 1) → encode, then recovering the model output.
///
/// This is the `clamp_denoised=True` branch from
/// `pipeline_pixelflux2_klein.py:275-279`:
/// ```text
/// denoised = x_t - model_output * sigma
/// image = oklab_decode(denoised).clamp(-1, 1)
/// denoised = oklab_encode(image)
/// model_output = (x_t - denoised) / max(sigma, 1e-4)
/// ```
///
/// Forces all per-step predictions to lie inside the sRGB-representable
/// gamut, keeping the trajectory stable.
///
/// **Performance note**: This routes through host memory because Oklab is
/// a CPU implementation (Step 1). At 960×1280 with 38 steps that's
/// ~140 MB of device-↔-host traffic. Acceptable for v0; a GPU Oklab
/// kernel can replace this later without changing the interface.
pub fn clamp_denoised_oklab(
    x_t: &Tensor,
    model_output: &Tensor,
    sigma: f32,
    sigma_min: f32,
) -> Result<Tensor> {
    if x_t.shape().dims() != model_output.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "clamp_denoised_oklab: shape mismatch {:?} vs {:?}",
            x_t.shape().dims(),
            model_output.shape().dims()
        )));
    }
    let dims = x_t.shape().dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::InvalidInput(format!(
            "clamp_denoised_oklab expects (B, 3, H, W), got {:?}",
            dims
        )));
    }
    let (b, _, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    // denoised = x_t - model_output * sigma  (F32)
    let scaled = model_output.mul_scalar(sigma)?;
    let denoised = x_t.sub(&scaled)?;

    // Host round-trip per batch slice. We pull, decode, clamp, encode, push.
    let denoised_host = denoised.to_vec()?;
    let plane = 3 * h * w;
    let mut clipped = vec![0.0_f32; denoised_host.len()];
    let mut tmp = vec![0.0_f32; plane];
    for bi in 0..b {
        let lo = bi * plane;
        let hi = lo + plane;
        // Oklab → sRGB pixels in [-1, 1]
        crate::vae::oklab::decode_planar(&denoised_host[lo..hi], &mut tmp);
        // Clamp NaN-preservingly to [-1, 1]
        for v in tmp.iter_mut() {
            if !v.is_nan() {
                if *v < -1.0 {
                    *v = -1.0;
                } else if *v > 1.0 {
                    *v = 1.0;
                }
            }
        }
        // sRGB → Oklab
        crate::vae::oklab::encode_planar(&tmp, &mut clipped[lo..hi]);
    }

    // denoised_clipped → Tensor
    let denoised_clipped = Tensor::from_vec(
        clipped,
        flame_core::Shape::from_dims(dims),
        x_t.device().clone(),
    )?;

    // model_output_new = (x_t - denoised_clipped) / max(sigma, sigma_min)
    let diff = x_t.sub(&denoised_clipped)?;
    let denom = if sigma.is_nan() {
        sigma
    } else if sigma < sigma_min {
        sigma_min
    } else {
        sigma
    };
    diff.mul_scalar(1.0 / denom)
}

/// Orthogonal classifier-free guidance bias.
///
/// Port of LakonLab's `gaussian_flow.guidance_jit` (Apache 2.0). Returns
/// the **bias tensor** to add to `pos`: the caller does `output = pos +
/// guidance_bias(...)`.
///
/// Algorithm:
/// ```text
/// bias  = (pos - neg) * (guidance_scale - 1)
/// coef  = mean(bias * parallel_dir) / max(mean(parallel_dir^2), 1e-6)
/// bias  = bias - coef * parallel_dir * orthogonal_strength
/// ```
///
/// The mean is taken over every dim except batch (`keepdim=True` in the
/// reference). For B=1 inference — the only case AsymFLUX.2 hits, since
/// CFG runs as two sequential single-batch forward passes — this collapses
/// to a single host-side scalar, pulled via `.item()` and pushed back as
/// `mul_scalar`. The roundtrip is one host sync per CFG step; acceptable
/// at the inference cadence.
///
/// Set `guidance_scale = 1.0` to short-circuit: returns a zero-filled
/// tensor with the same dtype and shape as `pos` without launching any
/// kernels beyond the allocation.
///
/// All math runs in F32 internally; the output is returned in `pos`'s
/// dtype so the caller can `pos.add(&bias)` without a cast.
pub fn guidance_bias(
    pos: &Tensor,
    neg: &Tensor,
    guidance_scale: f32,
    orthogonal_strength: f32,
    parallel_dir: &Tensor,
) -> Result<Tensor> {
    if pos.shape().dims() != neg.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "guidance_bias: pos shape {:?} != neg shape {:?}",
            pos.shape().dims(),
            neg.shape().dims()
        )));
    }
    if pos.shape().dims() != parallel_dir.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "guidance_bias: pos shape {:?} != parallel_dir shape {:?}",
            pos.shape().dims(),
            parallel_dir.shape().dims()
        )));
    }
    let target_dtype = pos.dtype();

    // Short-circuit: guidance_scale == 1.0 means no guidance — bias is 0.
    if guidance_scale == 1.0 {
        return Tensor::zeros_dtype(pos.shape().clone(), target_dtype, pos.device().clone());
    }

    // F32 cast for stable math.
    let pos_f = if pos.dtype() == DType::F32 {
        pos.clone()
    } else {
        pos.to_dtype(DType::F32)?
    };
    let neg_f = if neg.dtype() == DType::F32 {
        neg.clone()
    } else {
        neg.to_dtype(DType::F32)?
    };
    let par_f = if parallel_dir.dtype() == DType::F32 {
        parallel_dir.clone()
    } else {
        parallel_dir.to_dtype(DType::F32)?
    };

    // bias = (pos - neg) * (g - 1)
    let diff = pos_f.sub(&neg_f)?;
    let bias = diff.mul_scalar(guidance_scale - 1.0)?;

    if orthogonal_strength == 0.0 {
        // Skip the orthogonal projection — caller wants plain CFG.
        return if target_dtype == DType::F32 {
            Ok(bias)
        } else {
            bias.to_dtype(target_dtype)
        };
    }

    // Compute scalar coefficients.
    let bias_dot_par = bias.mul(&par_f)?.mean()?.item()?;
    let par_dot_par = par_f.mul(&par_f)?.mean()?.item()?;
    // Lower-clamp denominator to 1e-6 (matches reference `.clamp(min=1e-6)`),
    // NaN-preserving to surface upstream issues.
    let denom = if par_dot_par.is_nan() {
        par_dot_par
    } else if par_dot_par < 1e-6 {
        1e-6
    } else {
        par_dot_par
    };
    let coef = bias_dot_par / denom * orthogonal_strength;

    // bias = bias - coef * parallel_dir
    let scaled_par = par_f.mul_scalar(coef)?;
    let out = bias.sub(&scaled_par)?;

    if target_dtype == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(target_dtype)
    }
}

/// Reorganize `(B, C, H, W)` pixels into packed `(B, C * p^2, H/p, W/p)`
/// patches.
///
/// The packing rule matches the reference `AsymFlux2Transformer2DModel.patchify`
/// (Apache 2.0): for each output channel index `k = c * p^2 + ph * p + pw`,
/// the value is `x[b, c, ph + y * p, pw + x * p]` where `(y, x) = (H/p,
/// W/p)` is the output spatial position. Equivalent to PyTorch:
///
/// ```python
/// x.reshape(B, C, H/p, p, W/p, p).permute(0, 1, 3, 5, 2, 4)
///  .reshape(B, C*p*p, H/p, W/p)
/// ```
///
/// `H` and `W` must be exact multiples of `patch_size`. Output is
/// **contiguous** — the trailing `reshape` after `permute` auto-
/// materializes via flame-core's reshape implementation.
pub fn patchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "patchify expects rank-4 (B,C,H,W), got rank {}",
            dims.len()
        )));
    }
    if patch_size == 0 {
        return Err(Error::InvalidInput(
            "patchify: patch_size must be > 0".into(),
        ));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if h % patch_size != 0 || w % patch_size != 0 {
        return Err(Error::InvalidInput(format!(
            "patchify: H={h} and W={w} must both be multiples of patch_size={patch_size}"
        )));
    }
    let h_p = h / patch_size;
    let w_p = w / patch_size;
    // (B, C, H, W) → (B, C, H/p, p, W/p, p)
    let x = x.reshape(&[b, c, h_p, patch_size, w_p, patch_size])?;
    // permute(0, 1, 3, 5, 2, 4) → (B, C, p, p, H/p, W/p)
    let x = x.permute(&[0, 1, 3, 5, 2, 4])?;
    // Pack channels: (B, C*p*p, H/p, W/p). `reshape` materializes the view.
    x.reshape(&[b, c * patch_size * patch_size, h_p, w_p])
}

/// Inverse of [`patchify`]. `(B, C * p^2, H/p, W/p)` → `(B, C, H, W)`.
///
/// `packed_channels` must be divisible by `patch_size * patch_size`. The
/// caller passes the **original** `patch_size`; the function recovers `C`
/// as `packed_channels / p^2`.
pub fn unpatchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "unpatchify expects rank-4, got rank {}",
            dims.len()
        )));
    }
    if patch_size == 0 {
        return Err(Error::InvalidInput(
            "unpatchify: patch_size must be > 0".into(),
        ));
    }
    let (b, c_packed, h_p, w_p) = (dims[0], dims[1], dims[2], dims[3]);
    let p_sq = patch_size * patch_size;
    if c_packed % p_sq != 0 {
        return Err(Error::InvalidInput(format!(
            "unpatchify: packed channel count {c_packed} not divisible by patch_size^2={p_sq}"
        )));
    }
    let c = c_packed / p_sq;
    // (B, C*p*p, H/p, W/p) → (B, C, p, p, H/p, W/p)
    let x = x.reshape(&[b, c, patch_size, patch_size, h_p, w_p])?;
    // permute(0, 1, 4, 2, 5, 3) → (B, C, H/p, p, W/p, p)
    let x = x.permute(&[0, 1, 4, 2, 5, 3])?;
    // (B, C, H, W)
    x.reshape(&[b, c, h_p * patch_size, w_p * patch_size])
}

/// Reshape `(B, C, H, W)` to packed-sequence `(B, H*W, C)` layout for the
/// transformer.
///
/// Output is **contiguous** — needed because downstream `matmul` reads
/// storage directly (`launch_gemm` → `try_as_slice_f32`) and would
/// silently use the un-permuted memory if fed a stride view.
pub fn pack(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "pack expects rank-4 (B,C,H,W), got rank {}",
            dims.len()
        )));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    // (B, C, H, W) → (B, C, H*W)
    let x = x.reshape(&[b, c, h * w])?;
    // permute(0, 2, 1) → (B, H*W, C)
    let x = x.permute(&[0, 2, 1])?;
    // Materialize so the next matmul / mul_scalar sees row-major storage.
    x.contiguous()
}

/// Inverse of [`pack`]: `(B, N, C)` with `N = H * W` → `(B, C, H, W)`.
pub fn unpack(x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "unpack expects rank-3 (B,N,C), got rank {}",
            dims.len()
        )));
    }
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    if n != h * w {
        return Err(Error::InvalidInput(format!(
            "unpack: N={n} does not match H*W={}",
            h * w
        )));
    }
    // permute(0, 2, 1) → (B, C, N). `reshape` materializes the view.
    let x = x.permute(&[0, 2, 1])?;
    x.reshape(&[b, c, h, w])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn calibration_at_s_eq_1_is_identity() {
        // With s = 1: k = 1 / (1 + 0 * sigma) = 1, cal_timestep = timestep.
        let c = compute_calibration(0.42, 1.0, 1.0);
        assert!(approx_eq(c.k, 1.0, 1e-7));
        assert!(approx_eq(c.cal_timestep, 0.42, 1e-7));
        assert!(approx_eq(c.sigma, 0.42, 1e-7));
        assert!(approx_eq(c.s, 1.0, 1e-7));
    }

    #[test]
    fn calibration_at_sigma_eq_0_gives_k_eq_1_over_s() {
        // sigma=0 → k = 1 / s.
        let c = compute_calibration(0.0, 0.5, 1.0);
        assert!(approx_eq(c.k, 2.0, 1e-6));
        assert!(approx_eq(c.cal_timestep, 0.0, 1e-7));
        assert!(approx_eq(c.sigma, 0.0, 1e-7));
    }

    #[test]
    fn calibration_at_sigma_eq_1_gives_k_eq_1() {
        // sigma=1 → k = 1 / (s + 1 - s) = 1.
        let c = compute_calibration(1.0, 0.7, 1.0);
        assert!(approx_eq(c.k, 1.0, 1e-7));
        assert!(approx_eq(c.cal_timestep, 1.0, 1e-7));
    }

    #[test]
    fn calibration_matches_reference_formula() {
        // Hand-verified: s=0.8, timestep=0.3, num_timesteps=1
        //   sigma = 0.3
        //   k = 1 / (0.8 + 0.2 * 0.3) = 1 / 0.86 ≈ 1.16279
        //   cal_timestep = 0.3 * k ≈ 0.34884
        let c = compute_calibration(0.3, 0.8, 1.0);
        assert!(approx_eq(c.sigma, 0.3, 1e-7));
        assert!(approx_eq(c.k, 1.0 / 0.86, 1e-6));
        assert!(approx_eq(c.cal_timestep, 0.3 / 0.86, 1e-6));
    }

    #[test]
    fn calibration_scale_zero_timestep_zero_yields_non_finite() {
        // s=0, timestep=0 → sigma=0, k = 1/(0 + 0) = Inf, cal_timestep = 0*Inf = NaN.
        // Document this as a caller-bug surfacing through non-finite values.
        let c = compute_calibration(0.0, 0.0, 1.0);
        assert!(c.k.is_infinite(), "expected k=Inf, got {}", c.k);
        assert!(c.cal_timestep.is_nan(), "expected NaN cal_timestep, got {}", c.cal_timestep);
    }

    #[test]
    fn calibration_num_timesteps_zero_produces_inf_sigma() {
        // num_timesteps=0 → divide by zero → Inf sigma. Caller bug surfacing.
        let c = compute_calibration(1.0, 0.5, 0.0);
        assert!(c.sigma.is_infinite(), "expected Inf sigma, got {}", c.sigma);
    }

    #[test]
    fn sigma_clamp_at_boundary_returns_sigma_min() {
        // sigma exactly equal to sigma_min should land in the >= branch
        // (returns sigma itself, not sigma_min). Test boundary stability.
        let out = clamp_sigma_nan_preserving(1e-4, 1e-4);
        assert!(approx_eq(out, 1e-4, 0.0), "boundary drifted: {out}");
    }

    #[test]
    fn sigma_clamp_below_min_returns_min() {
        let out = clamp_sigma_nan_preserving(1e-9, 1e-4);
        assert!(approx_eq(out, 1e-4, 0.0), "below-min clamp broken: {out}");
    }

    #[test]
    fn sigma_clamp_above_min_returns_sigma() {
        let out = clamp_sigma_nan_preserving(0.5, 1e-4);
        assert!(approx_eq(out, 0.5, 0.0), "above-min passthrough broken: {out}");
    }

    #[test]
    fn sigma_clamp_preserves_nan() {
        // Critical: a NaN sigma must propagate, not be squashed to
        // sigma_min. This catches future "let me sanitize" changes that
        // would silently hide upstream bugs.
        let out = clamp_sigma_nan_preserving(f32::NAN, 1e-4);
        assert!(out.is_nan(), "sigma NaN got swallowed to {out}");
    }

    #[test]
    fn sigma_clamp_with_negative_sigma_min_passes_through() {
        // Caller bug surface: sigma_min < 0 should still clamp correctly,
        // not panic. (sigma=-0.5 < sigma_min=-0.1 → return sigma_min.)
        let out = clamp_sigma_nan_preserving(-0.5, -0.1);
        assert!(approx_eq(out, -0.1, 0.0), "negative sigma_min path broke: {out}");
    }

    #[test]
    fn sigma_clamp_with_inf_sigma_returns_inf() {
        let out = clamp_sigma_nan_preserving(f32::INFINITY, 1e-4);
        assert!(out.is_infinite() && out > 0.0, "Inf sigma squashed: {out}");
    }

    #[test]
    fn calibration_at_s_eq_0_gives_k_eq_one_over_sigma() {
        // Degenerate case: s=0 makes sigma the only nonzero term in the
        // denominator. k = 1 / sigma. At sigma=0.5 → k=2. cal_timestep =
        // 0.5 * 2 = 1.0. Lock down the algebra so a future refactor of
        // `k = 1 / (s + (1 - s) * sigma)` to e.g. precomputed coefficients
        // can't silently drift here.
        let c = compute_calibration(0.5, 0.0, 1.0);
        assert!(approx_eq(c.k, 2.0, 1e-6), "s=0 k path broke: {}", c.k);
        assert!(approx_eq(c.cal_timestep, 1.0, 1e-6), "s=0 cal_timestep broke: {}", c.cal_timestep);
    }

    #[test]
    fn num_timesteps_scales_sigma() {
        // With num_timesteps = 1000 (hypothetical), sigma = timestep / 1000.
        let c = compute_calibration(500.0, 1.0, 1000.0);
        assert!(approx_eq(c.sigma, 0.5, 1e-6));
        assert!(approx_eq(c.k, 1.0, 1e-7));
    }

    // ---------------------------------------------------------------------
    // GPU-bound patchify / pack / unpack / unpatchify tests.
    //
    // Each test gates on `CudaDevice::new(0)` and prints a skip message
    // rather than failing, matching the pattern in `src/lycoris.rs::tests`.
    // ---------------------------------------------------------------------

    use flame_core::Shape;
    use std::sync::Arc;

    fn maybe_cuda_device() -> Option<Arc<cudarc::driver::CudaDevice>> {
        cudarc::driver::CudaDevice::new(0).ok()
    }

    /// Generate a deterministic (Lehmer LCG) F32 tensor with values in
    /// `[-1, 1)`. Removes test flakiness from RNG seeds.
    fn deterministic_tensor(
        shape: &[usize],
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Tensor {
        let numel: usize = shape.iter().product();
        let mut state: u32 = 0xC0FFEE_u32.wrapping_add(numel as u32);
        let mut data = vec![0.0f32; numel];
        for slot in data.iter_mut() {
            state = state.wrapping_mul(48271) % 0x7fff_ffff;
            let u = state as f32 / 0x7fff_ffff as f32;
            *slot = u * 2.0 - 1.0;
        }
        Tensor::from_vec(data, Shape::from_dims(shape), device).expect("from_vec")
    }

    #[test]
    fn patchify_produces_correct_output_shape() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // (B=1, C=3, H=32, W=32) with p=16 → (1, 3*16*16=768, 2, 2)
        let x = deterministic_tensor(&[1, 3, 32, 32], dev);
        let out = patchify(&x, 16).expect("patchify");
        assert_eq!(out.shape().dims(), &[1, 768, 2, 2]);
    }

    #[test]
    fn unpatchify_inverts_patchify_shape() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[2, 3, 64, 48], dev);
        let p = patchify(&x, 16).expect("patchify");
        assert_eq!(p.shape().dims(), &[2, 768, 4, 3]);
        let r = unpatchify(&p, 16).expect("unpatchify");
        assert_eq!(r.shape().dims(), &[2, 3, 64, 48]);
    }

    #[test]
    fn patchify_unpatchify_roundtrip_is_identity() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 32, 32], dev);
        let original = x.to_vec().expect("to_vec");
        let p = patchify(&x, 16).expect("patchify");
        let r = unpatchify(&p, 16).expect("unpatchify");
        let recovered = r.to_vec().expect("to_vec");
        assert_eq!(original.len(), recovered.len());
        for (i, (a, b)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 0.0),
                "patchify roundtrip drift at idx {i}: {a} -> {b}"
            );
        }
    }

    #[test]
    fn pack_produces_correct_output_shape() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 768, 4, 4], dev);
        let out = pack(&x).expect("pack");
        assert_eq!(out.shape().dims(), &[1, 16, 768]);
    }

    #[test]
    fn pack_unpack_roundtrip_is_identity() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 768, 4, 4], dev);
        let original = x.to_vec().expect("to_vec");
        let p = pack(&x).expect("pack");
        assert_eq!(p.shape().dims(), &[1, 16, 768]);
        let r = unpack(&p, 4, 4).expect("unpack");
        assert_eq!(r.shape().dims(), &[1, 768, 4, 4]);
        let recovered = r.to_vec().expect("to_vec");
        for (i, (a, b)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 0.0),
                "pack/unpack drift at idx {i}: {a} -> {b}"
            );
        }
    }

    #[test]
    fn full_chain_roundtrip_is_identity() {
        // pixels → patchify → pack → unpack → unpatchify → original.
        // Locks the whole patch-and-pack story end-to-end.
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 32, 32], dev);
        let original = x.to_vec().expect("to_vec");

        let p = patchify(&x, 16).expect("patchify");
        let pk = pack(&p).expect("pack");
        // After pack, shape is (1, 4, 768).
        assert_eq!(pk.shape().dims(), &[1, 4, 768]);
        let up = unpack(&pk, 2, 2).expect("unpack");
        let r = unpatchify(&up, 16).expect("unpatchify");
        let recovered = r.to_vec().expect("to_vec");

        for (i, (a, b)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 0.0),
                "full chain drift at idx {i}: {a} -> {b}"
            );
        }
    }

    #[test]
    fn patchify_rejects_non_multiple_of_patch_size() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // H=17 is not a multiple of 16.
        let x = deterministic_tensor(&[1, 3, 17, 32], dev);
        let err = patchify(&x, 16).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("must both be multiples"),
            "wrong error: {msg}"
        );
    }

    #[test]
    fn patchify_rejects_rank_other_than_4() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[3, 32, 32], dev);
        let err = patchify(&x, 16).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("rank-4"), "wrong error: {msg}");
    }

    #[test]
    fn patchify_rejects_zero_patch_size() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 16, 16], dev);
        let err = patchify(&x, 0).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("patch_size must be > 0"), "wrong error: {msg}");
    }

    #[test]
    fn unpatchify_rejects_non_divisible_packed_channels() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // 700 is not divisible by 256 (=16*16).
        let x = deterministic_tensor(&[1, 700, 2, 2], dev);
        let err = unpatchify(&x, 16).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not divisible"), "wrong error: {msg}");
    }

    #[test]
    fn unpack_rejects_n_mismatch() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // N=15 ≠ H*W=16.
        let x = deterministic_tensor(&[1, 15, 768], dev);
        let err = unpack(&x, 4, 4).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("does not match H*W"), "wrong error: {msg}");
    }

    #[test]
    fn patchify_at_exact_patch_size_yields_one_patch() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // H = W = patch_size = 16 → exactly one patch.
        let x = deterministic_tensor(&[1, 3, 16, 16], dev);
        let out = patchify(&x, 16).expect("patchify");
        assert_eq!(out.shape().dims(), &[1, 768, 1, 1]);
    }

    #[test]
    fn patchify_non_square_h_and_w_works() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // 960 × 1280 — the model card's default resolution. p=16 → (60, 80).
        // Run a smaller scaled version: 48 × 64 with p=16 → (3, 4).
        let x = deterministic_tensor(&[1, 3, 48, 64], dev);
        let p = patchify(&x, 16).expect("patchify");
        assert_eq!(p.shape().dims(), &[1, 768, 3, 4]);
        let original = x.to_vec().expect("to_vec");
        let r = unpatchify(&p, 16).expect("unpatchify");
        let recovered = r.to_vec().expect("to_vec");
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 0.0));
        }
    }

    #[test]
    fn dynamic_shift_at_base_seq_len_equals_base_shift() {
        // sqrt-type interpolation at the base anchor returns base_shift.
        let s = dynamic_shift_sqrt(
            KLEIN_BASE_SEQ_LEN,
            KLEIN_BASE_SHIFT,
            KLEIN_MAX_SHIFT,
            KLEIN_BASE_SEQ_LEN,
            KLEIN_MAX_SEQ_LEN,
        );
        assert!(approx_eq(s, KLEIN_BASE_SHIFT, 1e-4), "base anchor: {s}");
    }

    #[test]
    fn dynamic_shift_at_max_seq_len_equals_max_shift() {
        let s = dynamic_shift_sqrt(
            KLEIN_MAX_SEQ_LEN,
            KLEIN_BASE_SHIFT,
            KLEIN_MAX_SHIFT,
            KLEIN_BASE_SEQ_LEN,
            KLEIN_MAX_SEQ_LEN,
        );
        assert!(approx_eq(s, KLEIN_MAX_SHIFT, 1e-4), "max anchor: {s}");
    }

    #[test]
    fn klein_dynamic_shift_960x1280_in_range() {
        // The model card's default size is 960x1280 → seq_len = 1_228_800.
        // sqrt(1_228_800) ≈ 1108.5; sqrt(base=1024²)=1024; sqrt(max=2048²)=2048.
        // m = (34 - 17) / (2048 - 1024) ≈ 0.0166.
        // shift = (1108.5 - 1024) * 0.0166 + 17 ≈ 18.4.
        let s = klein_dynamic_shift(1280, 960);
        assert!(
            s > KLEIN_BASE_SHIFT && s < KLEIN_MAX_SHIFT,
            "960x1280 shift {s} not in ({}, {})",
            KLEIN_BASE_SHIFT,
            KLEIN_MAX_SHIFT
        );
    }

    #[test]
    fn sigma_schedule_length_and_endpoints() {
        let n = 38;
        let sigmas = compute_sigma_schedule(n, 17.0);
        assert_eq!(sigmas.len(), n + 1, "schedule should have N+1 entries");
        // First raw is 1.0, post-shift sigma = shift*1/(1+(shift-1)*1) = 1.
        assert!(approx_eq(sigmas[0], 1.0, 1e-5), "first sigma: {}", sigmas[0]);
        // Trailing entry is 0.0 (the boundary).
        assert!(approx_eq(sigmas[n], 0.0, 0.0), "trailing sigma: {}", sigmas[n]);
        // Monotonically non-increasing.
        for i in 1..sigmas.len() {
            assert!(
                sigmas[i] <= sigmas[i - 1] + 1e-6,
                "non-monotone at {i}: {} > {}",
                sigmas[i],
                sigmas[i - 1]
            );
        }
    }

    #[test]
    fn sigma_schedule_one_step_endpoint_false() {
        // With N=1, endpoint=False linspace gives [1.0] only. Schedule
        // length = 2 ([1.0, 0.0]).
        let sigmas = compute_sigma_schedule(1, 17.0);
        assert_eq!(sigmas.len(), 2);
        assert!(approx_eq(sigmas[0], 1.0, 1e-5));
        assert!(approx_eq(sigmas[1], 0.0, 0.0));
    }

    #[test]
    fn timesteps_from_sigmas_drops_trailing() {
        let sigmas = vec![1.0_f32, 0.7, 0.4, 0.0];
        let ts = timesteps_from_sigmas(&sigmas);
        assert_eq!(ts.len(), 3);
        assert_eq!(ts, vec![1.0, 0.7, 0.4]);
    }

    #[test]
    fn euler_step_shape_correctness() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 8, 8], dev.clone());
        let v = deterministic_tensor(&[1, 3, 8, 8], dev);
        let out = euler_step(&x, &v, 1.0, 0.95).expect("euler_step");
        assert_eq!(out.shape().dims(), &[1, 3, 8, 8]);
    }

    #[test]
    fn euler_step_with_zero_velocity_returns_input() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 8, 8], dev.clone());
        let v = Tensor::zeros(Shape::from_dims(&[1, 3, 8, 8]), dev).unwrap();
        let out = euler_step(&x, &v, 1.0, 0.5).expect("euler_step");
        let in_data = x.to_vec().unwrap();
        let out_data = out.to_vec().unwrap();
        for (a, b) in in_data.iter().zip(out_data.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn euler_step_with_dt_zero_returns_input() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 4, 4], dev.clone());
        let v = deterministic_tensor(&[1, 3, 4, 4], dev);
        // sigma_next == sigma_cur → dt = 0 → out = x.
        let out = euler_step(&x, &v, 0.5, 0.5).expect("euler_step");
        let in_data = x.to_vec().unwrap();
        let out_data = out.to_vec().unwrap();
        for (a, b) in in_data.iter().zip(out_data.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn euler_step_rejects_shape_mismatch() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 8, 8], dev.clone());
        let v = deterministic_tensor(&[1, 3, 4, 4], dev);
        let err = euler_step(&x, &v, 1.0, 0.5).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("shape"), "wrong error: {msg}");
    }

    #[test]
    fn clamp_denoised_oklab_shape_preserved() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 3, 16, 16], dev.clone());
        let v = deterministic_tensor(&[1, 3, 16, 16], dev);
        let out = clamp_denoised_oklab(&x, &v, 0.5, 1e-4).expect("clamp_denoised");
        assert_eq!(out.shape().dims(), &[1, 3, 16, 16]);
        let vals = out.to_vec().unwrap();
        for (i, val) in vals.iter().enumerate() {
            assert!(val.is_finite(), "non-finite at {i}: {val}");
        }
    }

    #[test]
    fn clamp_denoised_oklab_rejects_non_3channel() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 4, 8, 8], dev.clone());
        let v = deterministic_tensor(&[1, 4, 8, 8], dev);
        let err = clamp_denoised_oklab(&x, &v, 0.5, 1e-4).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("(B, 3, H, W)"), "wrong error: {msg}");
    }

    #[test]
    fn guidance_scale_one_returns_zero_bias() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let pos = deterministic_tensor(&[1, 3, 8, 8], dev.clone());
        let neg = deterministic_tensor(&[1, 3, 8, 8], dev.clone());
        let par = deterministic_tensor(&[1, 3, 8, 8], dev);
        let bias = guidance_bias(&pos, &neg, 1.0, 1.0, &par).expect("guidance_bias");
        let v = bias.to_vec().expect("to_vec");
        assert!(
            v.iter().all(|x| *x == 0.0),
            "guidance_scale=1 should return all-zero bias, got nonzero values"
        );
    }

    #[test]
    fn guidance_bias_shape_matches_pos() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let pos = deterministic_tensor(&[1, 3, 16, 16], dev.clone());
        let neg = deterministic_tensor(&[1, 3, 16, 16], dev.clone());
        let par = deterministic_tensor(&[1, 3, 16, 16], dev);
        let bias = guidance_bias(&pos, &neg, 4.0, 1.0, &par).expect("guidance_bias");
        assert_eq!(bias.shape().dims(), &[1, 3, 16, 16]);
    }

    #[test]
    fn guidance_orthogonal_zero_skips_projection() {
        // With orthogonal_strength = 0, the function returns plain CFG bias
        // = (pos - neg) * (g - 1). Verify against a hand computation.
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let pos_data = vec![1.0_f32, 2.0, 3.0];
        let neg_data = vec![0.5_f32, 1.5, 2.0];
        let par_data = vec![0.0_f32; 3];
        let pos = Tensor::from_vec(
            pos_data.clone(),
            Shape::from_dims(&[1, 3]),
            dev.clone(),
        )
        .unwrap();
        let neg = Tensor::from_vec(neg_data.clone(), Shape::from_dims(&[1, 3]), dev.clone())
            .unwrap();
        let par = Tensor::from_vec(par_data, Shape::from_dims(&[1, 3]), dev).unwrap();
        let g = 4.0_f32;
        let bias = guidance_bias(&pos, &neg, g, 0.0, &par).expect("guidance_bias");
        let v = bias.to_vec().unwrap();
        for i in 0..3 {
            let expected = (pos_data[i] - neg_data[i]) * (g - 1.0);
            assert!(
                approx_eq(v[i], expected, 1e-5),
                "orthogonal=0 plain CFG drift at {i}: {} vs {}",
                v[i],
                expected
            );
        }
    }

    #[test]
    fn guidance_bias_with_parallel_dir_zero_does_not_explode() {
        // parallel_dir = 0 → denominator = 0, clamped to 1e-6. Coef numerator
        // = mean(bias * 0) = 0, so coef = 0/1e-6 = 0. bias unchanged. Verify
        // no Inf/NaN leaks.
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let pos = deterministic_tensor(&[1, 3, 4, 4], dev.clone());
        let neg = deterministic_tensor(&[1, 3, 4, 4], dev.clone());
        let par = Tensor::zeros(Shape::from_dims(&[1, 3, 4, 4]), dev).unwrap();
        let bias = guidance_bias(&pos, &neg, 4.0, 1.0, &par).expect("guidance_bias");
        let v = bias.to_vec().unwrap();
        for (i, x) in v.iter().enumerate() {
            assert!(x.is_finite(), "non-finite bias at {i}: {x}");
        }
    }

    #[test]
    fn guidance_bias_rejects_shape_mismatch() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let pos = deterministic_tensor(&[1, 3, 8, 8], dev.clone());
        let neg = deterministic_tensor(&[1, 3, 4, 4], dev.clone());
        let par = deterministic_tensor(&[1, 3, 8, 8], dev);
        let err = guidance_bias(&pos, &neg, 4.0, 1.0, &par).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("shape"), "wrong error: {msg}");
    }

    #[test]
    fn pack_output_is_contiguous() {
        // Critical: pack's caller (asymflow_velocity, x_embedder matmul)
        // depends on the contiguous output. If a future refactor drops
        // the trailing .contiguous(), every downstream matmul silently
        // computes on the un-permuted buffer.
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = deterministic_tensor(&[1, 768, 4, 4], dev);
        let p = pack(&x).expect("pack");
        assert!(
            p.is_contiguous(),
            "pack output must be contiguous for matmul-correctness"
        );
    }
}
