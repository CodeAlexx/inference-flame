//! Euler sampler for flow-matching diffusion models.

use crate::models::zimage_nextdit::NextDiT;
use flame_core::{DType, Result, Shape, Tensor};

/// Single Euler step for flow-matching inference.
///
/// `model` predicts velocity v at the given sigma, then:
///   x_next = x + v * (sigma_next - sigma)
///
/// If `cap_feats_uncond` is provided and `cfg_scale > 1.0`, applies
/// classifier-free guidance:
///   v = v_uncond + cfg_scale * (v_cond - v_uncond)
pub fn euler_step(
    model: &mut NextDiT,
    x: &Tensor,
    sigma: f32,
    sigma_next: f32,
    cap_feats: &Tensor,
    cap_feats_uncond: Option<&Tensor>,
    cfg_scale: f32,
) -> Result<Tensor> {
    let device = x.device().clone();
    let b = x.shape().dims()[0];

    // Build timestep tensor (sigma value)
    let sigma_tensor = Tensor::from_vec_dtype(
        vec![sigma; b],
        Shape::from_dims(&[b]),
        device,
        DType::BF16,
    )?;

    // Model prediction (conditional)
    let pred_cond = model.forward(x, &sigma_tensor, cap_feats)?;

    let pred = if let Some(uncond_feats) = cap_feats_uncond {
        if cfg_scale > 1.0 {
            // Unconditional prediction
            let pred_uncond = model.forward(x, &sigma_tensor, uncond_feats)?;
            // CFG: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            let scaled = diff.mul_scalar(cfg_scale)?;
            pred_uncond.add(&scaled)?
        } else {
            pred_cond
        }
    } else {
        pred_cond
    };

    // Euler step: x_next = x + v * dt where dt = sigma_next - sigma
    let dt = sigma_next - sigma;
    let step = pred.mul_scalar(dt)?;
    x.add(&step)
}
