//! LTX-2/LTX-Video guidance helpers: CFG-star rescale, STG skip-layer mask,
//! and STG std-rescale. All math lifted verbatim from Lightricks's reference
//! pipeline (`ltx_video/pipelines/pipeline_ltx_video.py`, see
//! `LTX_FEATURE_PARITY.md` §4 for line numbers).
//!
//! Parity scripts in `scripts/`:
//!   - `ltx2_stg_mask_ref.py`   → `output/ltx2_stg_mask_ref.safetensors`
//!   - `ltx2_cfg_star_ref.py`   → `output/ltx2_cfg_star_ref.safetensors`
//!   - `ltx2_stg_rescale_ref.py` → `output/ltx2_stg_rescale_ref.safetensors`
//!
//! The parity checks live in `bin/ltx2_guidance_parity.rs`.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

/// STG skip-layer strategies, 1:1 with Lightricks's
/// `ltx_video/utils/skip_layer_strategy.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipLayerStrategy {
    /// `AttentionSkip`: on masked blocks, replace attention output with the
    /// pre-attention `hidden_states` (attention.py:1071-1077).  **Not
    /// implemented in Rust yet** — requires hooking pre-attention input
    /// through the block forward.
    AttentionSkip,
    /// `AttentionValues` (default in Lightricks dev configs): on masked
    /// blocks, replace attention output with the V projection
    /// (attention.py:1078-1084). This is what STG-AV actually does.
    AttentionValues,
    /// `Residual`: keep attention output, but drop residual on masked
    /// blocks (attention.py:1097-1110). **Not implemented** — requires
    /// routing the `residual` tensor through the attention call, which
    /// LTX-2's `LTX2Attention.forward` doesn't currently expose.
    Residual,
    /// `TransformerBlock`: blend post-block state back to pre-block state
    /// (attention.py:312-319). **Not implemented** — requires hooking
    /// pre-block input through the block forward.
    TransformerBlock,
}

impl SkipLayerStrategy {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            // Accept all the aliases Lightricks uses (inference.py:548-557).
            "stg_av" | "attention_values" => Some(Self::AttentionValues),
            "stg_as" | "attention_skip" => Some(Self::AttentionSkip),
            "stg_r" | "residual" => Some(Self::Residual),
            "stg_t" | "transformer_block" => Some(Self::TransformerBlock),
            _ => None,
        }
    }
}

/// Build the STG skip-layer mask — direct port of
/// `Transformer3DModel.create_skip_layer_mask` (transformer3d.py:173-188).
///
/// The returned flat `Vec<Vec<f32>>` has shape `[num_layers][batch_size*num_conds]`.
/// Rows corresponding to layers in `skip_block_list` get a 0 at every slot
/// `ptb_index, ptb_index + num_conds, …` (the perturb slots across the
/// duplicated batch).  All other entries are 1.
///
/// We keep this on the CPU because it's tiny (48×3 = 144 floats for 13B)
/// and the downstream consumer only needs the per-layer scalar at the
/// current block.
pub fn build_skip_layer_mask(
    num_layers: usize,
    batch_size: usize,
    num_conds: usize,
    skip_block_list: &[usize],
    ptb_index: usize,
) -> Vec<Vec<f32>> {
    let row_len = batch_size * num_conds;
    let mut mask: Vec<Vec<f32>> = (0..num_layers).map(|_| vec![1.0f32; row_len]).collect();
    for &block_idx in skip_block_list {
        if block_idx >= num_layers {
            continue;
        }
        let mut j = ptb_index;
        while j < row_len {
            mask[block_idx][j] = 0.0;
            j += num_conds;
        }
    }
    mask
}

/// CFG-star rescale: α = ⟨ε_text, ε_uncond⟩ / (‖ε_uncond‖² + 1e-8), then
/// ε_uncond ← α · ε_uncond.  Verbatim from pipeline_ltx_video.py:1227-1240.
///
/// Math runs in F32 for numerical stability (the reduction accumulates
/// over ~10⁶ elements on a full video latent); the output is cast back
/// to the input dtype.  `eps_text` and `eps_uncond` must share shape.
pub fn cfg_star_rescale(eps_text: &Tensor, eps_uncond: &Tensor) -> Result<Tensor> {
    let dims = eps_text.shape().dims().to_vec();
    if dims != eps_uncond.shape().dims() {
        return Err(flame_core::Error::InvalidInput(format!(
            "cfg_star_rescale shape mismatch: text={:?} uncond={:?}",
            dims, eps_uncond.shape().dims()
        )));
    }
    let batch = dims[0];
    let flat_len: usize = dims.iter().skip(1).product();

    let text_f32 = eps_text.to_dtype(DType::F32)?;
    let uncond_f32 = eps_uncond.to_dtype(DType::F32)?;
    let text_flat = text_f32.reshape(&[batch, flat_len])?;
    let uncond_flat = uncond_f32.reshape(&[batch, flat_len])?;

    // Compute alpha per-batch on CPU — single scalar per batch item.
    // Pulls the two flats back and reduces in f64 for parity with the
    // Python reference, which runs at torch.float32 but wide sums.
    let text_vec = text_flat.to_vec()?;
    let uncond_vec = uncond_flat.to_vec()?;
    debug_assert_eq!(text_vec.len(), batch * flat_len);

    let mut alphas: Vec<f32> = Vec::with_capacity(batch);
    for b in 0..batch {
        let t = &text_vec[b * flat_len..(b + 1) * flat_len];
        let u = &uncond_vec[b * flat_len..(b + 1) * flat_len];
        let mut dot = 0.0f64;
        let mut nsq = 0.0f64;
        for i in 0..flat_len {
            let ti = t[i] as f64;
            let ui = u[i] as f64;
            dot += ti * ui;
            nsq += ui * ui;
        }
        let alpha = (dot / (nsq + 1e-8)) as f32;
        alphas.push(alpha);
    }

    // Build alpha tensor of shape [batch, 1, 1, …] matching uncond rank.
    let mut alpha_shape = vec![batch];
    alpha_shape.extend(std::iter::repeat(1usize).take(dims.len() - 1));
    let alpha_tensor = Tensor::from_vec(
        alphas,
        Shape::from_dims(&alpha_shape),
        uncond_f32.device().clone(),
    )?;

    // Multiply in F32 then cast back to input dtype.
    let out_f32 = uncond_f32.mul(&alpha_tensor)?;
    out_f32.to_dtype(eps_uncond.dtype())
}

/// STG std-rescale: matches the std of `guided` back toward `pos`.
/// Verbatim from pipeline_ltx_video.py:1251-1262.
///
///   factor = std(pos) / std(guided)
///   factor = rescaling_scale * factor + (1 - rescaling_scale)
///   out    = guided * factor
///
/// std is computed per-batch over ALL remaining dims (view(B, -1).std).
/// Lightricks's torch `.std` uses Bessel's correction (unbiased=True) by
/// default, so we match that (divide by n-1, not n).
pub fn stg_rescale(pos: &Tensor, guided: &Tensor, rescaling_scale: f32) -> Result<Tensor> {
    let dims = guided.shape().dims().to_vec();
    if dims != pos.shape().dims() {
        return Err(flame_core::Error::InvalidInput(format!(
            "stg_rescale shape mismatch: pos={:?} guided={:?}",
            pos.shape().dims(), dims
        )));
    }
    let batch = dims[0];
    let flat_len: usize = dims.iter().skip(1).product();
    if flat_len < 2 {
        return Err(flame_core::Error::InvalidInput(
            "stg_rescale requires at least 2 elements per batch for unbiased std".into(),
        ));
    }

    let pos_f32 = pos.to_dtype(DType::F32)?;
    let guided_f32 = guided.to_dtype(DType::F32)?;
    let pos_flat_v = pos_f32.reshape(&[batch, flat_len])?.to_vec()?;
    let guided_flat_v = guided_f32.reshape(&[batch, flat_len])?.to_vec()?;

    let mut factors: Vec<f32> = Vec::with_capacity(batch);
    for b in 0..batch {
        let p = &pos_flat_v[b * flat_len..(b + 1) * flat_len];
        let g = &guided_flat_v[b * flat_len..(b + 1) * flat_len];
        let pos_std = unbiased_std_f64(p);
        let guided_std = unbiased_std_f64(g);
        let f = (pos_std / guided_std) as f32;
        let f = rescaling_scale * f + (1.0 - rescaling_scale);
        factors.push(f);
    }

    let mut factor_shape = vec![batch];
    factor_shape.extend(std::iter::repeat(1usize).take(dims.len() - 1));
    let factor_tensor = Tensor::from_vec(
        factors,
        Shape::from_dims(&factor_shape),
        guided_f32.device().clone(),
    )?;
    let out_f32 = guided_f32.mul(&factor_tensor)?;
    out_f32.to_dtype(guided.dtype())
}

fn unbiased_std_f64(v: &[f32]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| { let d = *x as f64 - mean; d * d }).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// Small helper for binaries: build the per-layer scalar mask column for a
/// given `(batch, num_conds, ptb_index)` by reshaping the output of
/// `build_skip_layer_mask`. Returns a `[num_layers]` tensor of ones/zeros
/// on the given device — suitable to pass as `skip_mask_1d` to block
/// forwards that only operate on a single conditioning slot.
pub fn single_cond_mask_from_skip_list(
    num_layers: usize,
    skip_block_list: &[usize],
    device: Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    let mut v = vec![1.0f32; num_layers];
    for &i in skip_block_list {
        if i < num_layers {
            v[i] = 0.0;
        }
    }
    Tensor::from_vec(v, Shape::from_dims(&[num_layers]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_mask_matches_lightricks_small() {
        // num_layers=4, batch=1, num_conds=3, skip=[1,3], ptb_index=2
        // Expected (per `ltx2_stg_mask_ref.py` small case):
        //   [[1,1,1],[1,1,0],[1,1,1],[1,1,0]]
        let mask = build_skip_layer_mask(4, 1, 3, &[1, 3], 2);
        assert_eq!(mask.len(), 4);
        assert_eq!(mask[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(mask[1], vec![1.0, 1.0, 0.0]);
        assert_eq!(mask[2], vec![1.0, 1.0, 1.0]);
        assert_eq!(mask[3], vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn build_mask_batch_duplicates() {
        // batch=2, num_conds=3, ptb_index=2 → zeros at indices 2 and 5.
        let mask = build_skip_layer_mask(2, 2, 3, &[0], 2);
        assert_eq!(mask[0], vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(mask[1], vec![1.0; 6]);
    }

    #[test]
    fn empty_skip_list_all_ones() {
        let mask = build_skip_layer_mask(3, 1, 2, &[], 1);
        for row in &mask {
            for v in row {
                assert_eq!(*v, 1.0);
            }
        }
    }
}
