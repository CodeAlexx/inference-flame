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
/// Matches DiffSynth's `FlowMatchScheduler.set_timesteps` (FLUX-shift form)
/// at `reference/diffsynth/diffusion/flow_match.py:103-118`:
///
/// ```python
/// sigmas = linspace(1, 0, num_steps + 1)[:-1]
/// sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
/// sigmas = cat([sigmas, zeros(1)])
/// ```
///
/// **NOT the Klein/Qwen-Image `exp(μ)/(exp(μ)+(1/t-1))` curve** — those
/// schedulers diverge from L2P's by 2-4× at low sigma per audit F1
/// (`MATH_AUDIT_2026-05-22.md`). Klein's schedule is for the Qwen-Image
/// preset; L2P specifically uses the FLUX-shift form.
pub fn build_l2p_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    // sigmas[i] = shift * t / (1 + (shift - 1) * t)  for t = 1 - i/num_steps
    // i in 0..num_steps gives t = 1.0, (n-1)/n, ..., 1/n.
    // Then append 0.0 for the final endpoint (matches the Python `cat`).
    let shift = shift as f64;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = 1.0 - (i as f64) / (num_steps as f64);
            let s = shift * t / (1.0 + (shift - 1.0) * t);
            s as f32
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// Training-time sigma table — matches Python `FlowMatchSFTLoss` at
/// `reference/diffsynth/diffusion/loss.py:6-13` with `num_inference_steps=500`
/// (the value `train_L2P.py:89` passes in `inputs_shared`).
///
/// Python does:
/// ```python
/// pipe.scheduler.set_timesteps(500)              # FLUX-shift sigmas
/// timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (1,))
/// timestep    = pipe.scheduler.timesteps[timestep_id]   # in [0, 1000]
/// # add_noise then looks up sigma = self.sigmas[argmin(timesteps - timestep)]
/// ```
/// i.e. uniformly pick an index in `[0, 500)`, take the shift-warped sigma.
///
/// Returns the precomputed `500`-entry (or whatever `num_train_steps`) table
/// of `shift·s/(1 + (shift-1)·s)` values. The caller `rng.gen_range`s an
/// index and indexes this table per step. Builds once, reuses for every step
/// in the training loop.
///
/// The endpoint `0.0` that `build_l2p_sigma_schedule` appends for inference
/// is NOT in this table — training never samples a degenerate `sigma=0`
/// (Python's `randint(0, 500)` is exclusive on the upper bound, so it
/// picks from indices that map to the unshifted sigma values).
pub fn build_l2p_training_sigma_table(num_train_steps: usize, shift: f32) -> Vec<f32> {
    let shift = shift as f64;
    (0..num_train_steps)
        .map(|i| {
            let t = 1.0 - (i as f64) / (num_train_steps as f64);
            let s = shift * t / (1.0 + (shift - 1.0) * t);
            s as f32
        })
        .collect()
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
        // CFG: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        // Matches Python `base_pipeline_L2P.py:353`:
        //     noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        // The earlier formulation `pred_cond + cfg*(pred_cond - pred_uncond)`
        // is a different curve — at cfg=2 it produced effective CFG ≈ 4
        // (audit F2, `MATH_AUDIT_2026-05-22.md`).
        let diff = pred_cond.sub(&pred_uncond)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        pred_uncond.add(&scaled)?
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
    fn schedule_matches_python_flux_shift() {
        // Reference values from Python: shift=3, num_steps=4.
        //   t = [1.0, 0.75, 0.5, 0.25]
        //   sigma = shift*t/(1+(shift-1)*t)
        //         = 3*t/(1+2*t)
        // t=1.0   -> 3/3   = 1.0
        // t=0.75  -> 2.25/2.5  = 0.9
        // t=0.5   -> 1.5/2.0   = 0.75
        // t=0.25  -> 0.75/1.5  = 0.5
        // appended endpoint: 0.0
        let s = build_l2p_sigma_schedule(4, 3.0);
        assert_eq!(s.len(), 5);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        for (a, b) in s.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "schedule: got {a}, expected {b}");
        }
    }
}
