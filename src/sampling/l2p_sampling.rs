//! L2P flow-matching sampler — Z-Image-style Euler + CFG, pixel-space.
//!
//! Differences from `inference_flame::sampling::euler::euler_step`:
//!   - Operates on `L2pDiT` (not `NextDiT`).
//!   - Input/output tensor is pixel-space `[B, 3, H, W]` BF16 (NOT
//!     `[B, 16, H/8, W/8]` latents).
//!   - F32 noise initializer — the reference pipeline forces F32 noise per
//!     PORT_SPEC §"Special / things to watch" #4. Caller must cast to BF16
//!     before the first `l2p_euler_step` call.
//!
//! Sign / timestep conventions are owned by `L2pDiT::forward_inner`:
//!   - The pipeline passes the flow-matching normalized timestep `v ∈ [0,1]`
//!     **as-is**. The DiT internally remaps to `(1 - v) * time_scale`.
//!   - The DiT internally negates the U-Net output before returning. The
//!     sampler MUST NOT re-negate.
//!
//! Both contracts are documented at the call sites below.

use crate::models::l2p::L2pDiT;
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

/// Build the L2P FlowMatch sigma schedule.
///
/// Identical math to `klein_sampling::build_sigma_schedule`. The L2P
/// reference pipeline uses DiffSynth's "Z-Image" FlowMatchScheduler
/// preset with `shift = 3.0`; the CLI default mirrors that.
pub fn build_l2p_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    crate::sampling::klein_sampling::build_sigma_schedule(num_steps, shift)
}

/// Generate initial pixel-space noise on CPU (Box–Muller, seeded
/// `StdRng`) and upload to GPU as **F32**.
///
/// Shape: `[1, 3, height, width]`, F32. Per PORT_SPEC §"Special / things
/// to watch" #4, the L2P reference pipeline forces noise to F32 even
/// though the DiT is BF16. The caller is responsible for the F32→BF16
/// cast before calling `l2p_euler_step` (see the `debug_assert_eq` in
/// that function and in `L2pDiT::forward_inner`).
pub fn init_l2p_noise(
    height: usize,
    width: usize,
    seed: u64,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let numel = 3 * height * width;
    let data = crate::sampling::klein_sampling::box_muller_noise(numel, seed);
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 3, height, width]),
        device.clone(),
        DType::F32,
    )
}

/// One Euler step for L2P with optional classifier-free guidance.
///
/// **Dtype contract.** `x` must be BF16 — `L2pDiT::forward_inner`
/// `debug_assert_eq`'s on this. `init_l2p_noise` returns F32 per the
/// reference; the caller must cast (`.to_dtype(DType::BF16)?`) before the
/// first call.
///
/// **Sign / timestep contract.** `sigma` is the flow-matching normalized
/// `v ∈ [0, 1]` from the schedule. `L2pDiT::forward_inner` handles the
/// L2P sign + timestep inversion internally:
///   - maps `sigma → (1 - sigma) * time_scale` before `timestep_embed`
///   - negates the U-Net output before returning
/// Do NOT re-invert the timestep here. Do NOT re-negate the prediction
/// here.
///
/// **CFG.** Mirrors `inference_flame::sampling::euler::euler_step`:
///     `pred = pred_cond + cfg_scale * (pred_cond - pred_uncond)`
/// (Mathematically equivalent to the standard
/// `uncond + cfg * (cond - uncond)`; expressed in the same form Z-Image
/// uses so the two pipelines stay paired.)
pub fn l2p_euler_step(
    model: &mut L2pDiT,
    x: &Tensor,
    sigma: f32,
    sigma_next: f32,
    cap_feats: &Tensor,
    cap_feats_uncond: Option<&Tensor>,
    cfg_scale: f32,
) -> Result<Tensor> {
    debug_assert_eq!(
        x.dtype(),
        DType::BF16,
        "l2p_euler_step expects BF16 x — caller must cast F32 noise to BF16"
    );

    let device = x.device().clone();
    let b = x.shape().dims()[0];

    // Timestep tensor: pass `sigma` (normalized) as-is. The DiT inverts
    // and scales internally.
    let sigma_tensor = Tensor::from_vec_dtype(
        vec![sigma; b],
        Shape::from_dims(&[b]),
        device,
        DType::BF16,
    )?;

    // Conditional prediction. forward_inner already applies the
    // pipeline-level sign flip; do NOT negate here.
    let pred_cond = model.forward(x, &sigma_tensor, cap_feats)?;

    let pred = if let (Some(uncond_feats), true) = (cap_feats_uncond, cfg_scale > 1.0) {
        let pred_uncond = model.forward(x, &sigma_tensor, uncond_feats)?;
        // pred = pred_cond + cfg_scale * (pred_cond - pred_uncond)
        let diff = pred_cond.sub(&pred_uncond)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        pred_cond.add(&scaled)?
    } else {
        pred_cond
    };

    // Euler step: x_next = x + (sigma_next - sigma) * pred
    let dsigma = sigma_next - sigma;
    let step = pred.mul_scalar(dsigma)?;
    x.add(&step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_endpoints_match_klein() {
        let s_l2p = build_l2p_sigma_schedule(30, 3.0);
        let s_klein = crate::sampling::klein_sampling::build_sigma_schedule(30, 3.0);
        assert_eq!(s_l2p.len(), s_klein.len());
        assert_eq!(s_l2p.len(), 31);
        for (a, b) in s_l2p.iter().zip(s_klein.iter()) {
            assert!((a - b).abs() < 1e-7, "{a} vs {b}");
        }
        assert_eq!(*s_l2p.last().unwrap(), 0.0);
    }
}
