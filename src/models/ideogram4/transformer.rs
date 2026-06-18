//! Ideogram 4 top-level transformer forward — the block stack + embeds + final.
//!
//! Mirrors `Ideogram4Transformer.forward`
//! (`/home/alex/ideogram4-ref/src/ideogram4/modeling_ideogram4.py:311-379`)
//! op-for-op. This COMPLETES the DiT module (chunks 1-3): config + FP8 weights
//! (chunk 1), MRoPE/attention/block (chunks 1-2), embeds + final + this stack
//! (chunk 3).
//!
//! ## Reference forward sequence (lines 335-379)
//! ```text
//! x = x.to(param_dtype); t = t.to(param_dtype); llm_features = llm_features.to(param_dtype)
//! indicator = indicator.long()
//! llm_token_mask    = (indicator == LLM_TOKEN_INDICATOR).to(dtype).unsqueeze(-1)   # [B,L,1]
//! output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(dtype).unsqueeze(-1)
//! llm_features = llm_features * llm_token_mask
//! x = x * output_image_mask
//! x = input_proj(x) * output_image_mask
//! t_cond = t_embedding(t); if t.dim()==1: t_cond = t_cond.unsqueeze(1)            # [B,1,emb]
//! adaln_input = silu(adaln_proj(t_cond))                                          # [B,1,512]
//! llm_features = llm_cond_norm(llm_features)
//! llm_features = llm_cond_proj(llm_features) * llm_token_mask
//! h = x + llm_features
//! image_indicator_embedding = embed_image_indicator((indicator==OUTPUT_IMAGE).long())
//! h = h + image_indicator_embedding
//! cos, sin = rotary_emb(position_ids); cos=cos.to(h.dtype); sin=sin.to(h.dtype)
//! for layer in layers: h = layer(h, segment_ids, cos, sin, adaln_input)
//! out = final_layer(h, c=adaln_input)
//! return out.to(float32)
//! ```
//!
//! ## Input contract (B=1 inference; host-built positions/indicator)
//! Consistent with [`super::mrope::build_cos_sin`] (host `&[u32]` columns) and
//! the attention mask contract (optional bool `[B,1,L,L]`), this forward takes
//! the position columns + indicator as host slices and the segment mask as an
//! optional pre-built tensor (the caller / chunk-4 assembles position_ids with
//! the IMAGE_POSITION_OFFSET=65536 stamping and the segment mask). For B=1
//! unpadded the segment mask is all-ones → pass `None`.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{DType, Error, Result, Shape, Tensor};

use super::block::transformer_block_forward;
use super::embed::{
    adaln_input_from_t_cond, embed_scalar_t, image_indicator_embedding, input_proj_forward,
    llm_cond_forward,
};
use super::final_layer::final_layer_forward;
use super::mrope::build_cos_sin;
use super::weights::Ideogram4RawWeight;
use super::Ideogram4Config;

/// Token role indicator: an LLM (text) conditioning token.
/// (`constants.py:4` `LLM_TOKEN_INDICATOR = 3`.)
pub const LLM_TOKEN_INDICATOR: i32 = 3;
/// Token role indicator: an output-image latent token.
/// (`constants.py:3` `OUTPUT_IMAGE_INDICATOR = 2`.)
pub const OUTPUT_IMAGE_INDICATOR: i32 = 2;

/// Build a `[B, L, 1]` BF16 mask = `(indicator == target)` as 0.0/1.0.
///
/// Mirrors `(indicator == X).to(dtype).unsqueeze(-1)` (lines 346-347). The
/// trailing `1` makes it broadcast-multiply against `[B, L, C]` over the
/// channel axis.
fn role_mask(indicator: &[i32], target: i32, b: usize, l: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let vals: Vec<f32> = indicator
        .iter()
        .map(|&i| if i == target { 1.0f32 } else { 0.0f32 })
        .collect();
    Tensor::from_vec_dtype(vals, Shape::from_dims(&[b, l, 1]), device.clone(), DType::BF16)
}

/// Full Ideogram4 transformer forward — velocity prediction.
///
/// Inputs (all already on `device`):
/// - `x`: `[B, L, in_channels=128]` BF16 — noise/image-latent tokens.
/// - `t_vals`: one flow-matching time scalar per sample (`(B,)`), in `[0,1]`.
/// - `llm_features`: `[B, L, llm_features_dim=53248]` BF16 — Qwen3-VL features.
/// - `t_pos`, `h_pos`, `w_pos`: per-token MRoPE position columns, each length
///   `L` (= the three columns of `position_ids[(B=1),L,3]`).
/// - `segment_mask`: optional bool `[B,1,L,L]` (True=attend); `None` for B=1
///   unpadded (all tokens attend).
/// - `indicator`: per-token role, length `B*L` (`LLM_TOKEN_INDICATOR` or
///   `OUTPUT_IMAGE_INDICATOR`).
///
/// Returns `[B, L, in_channels]` **F32** velocity (only OUTPUT_IMAGE positions
/// are meaningful — the reference returns `out.to(float32)`).
///
/// Weight access mirrors the chunk-2 block idiom: per-block prefix
/// `layers.{i}` and top-level keys `input_proj.*`, `llm_cond_norm.weight`,
/// `llm_cond_proj.*`, `t_embedding.mlp_{in,out}.*`, `adaln_proj.*`,
/// `embed_image_indicator.weight`, `final_layer.*`.
#[allow(clippy::too_many_arguments)]
pub fn transformer_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    config: &Ideogram4Config,
    x: &Tensor,
    t_vals: &[f32],
    llm_features: &Tensor,
    t_pos: &[u32],
    h_pos: &[u32],
    w_pos: &[u32],
    segment_mask: Option<&Tensor>,
    indicator: &[i32],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let xdims = x.shape().dims().to_vec();
    if xdims.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "ideogram4 transformer: x must be 3D [B,L,in_channels], got {xdims:?}"
        )));
    }
    let (b, l, in_channels) = (xdims[0], xdims[1], xdims[2]);
    if in_channels != config.in_channels {
        return Err(Error::InvalidShape(format!(
            "ideogram4 transformer: x in_channels {in_channels} != config.in_channels {}",
            config.in_channels
        )));
    }
    if t_vals.len() != b {
        return Err(Error::InvalidOperation(format!(
            "ideogram4 transformer: t_vals len {} != batch {b}",
            t_vals.len()
        )));
    }
    if indicator.len() != b * l {
        return Err(Error::InvalidOperation(format!(
            "ideogram4 transformer: indicator len {} != b*l ({b}*{l})",
            indicator.len()
        )));
    }
    if t_pos.len() != l || h_pos.len() != l || w_pos.len() != l {
        return Err(Error::InvalidOperation(format!(
            "ideogram4 transformer: position columns must each be length L={l}, got T={} H={} W={}",
            t_pos.len(),
            h_pos.len(),
            w_pos.len()
        )));
    }

    // Cast inputs to BF16 (param dtype). x/llm_features are passed BF16 already
    // by the caller; cast defensively (zero-copy if already BF16).
    let x = if x.dtype() == DType::BF16 { x.clone() } else { x.to_dtype(DType::BF16)? };
    let llm_features = if llm_features.dtype() == DType::BF16 {
        llm_features.clone()
    } else {
        llm_features.to_dtype(DType::BF16)?
    };

    // Role masks [B,L,1].
    let llm_token_mask = role_mask(indicator, LLM_TOKEN_INDICATOR, b, l, device)?;
    let output_image_mask = role_mask(indicator, OUTPUT_IMAGE_INDICATOR, b, l, device)?;

    // llm_features = llm_features * llm_token_mask
    let llm_features = llm_features.mul(&llm_token_mask)?;
    // x = x * output_image_mask
    let x = x.mul(&output_image_mask)?;
    // x = input_proj(x) * output_image_mask   (input_proj: 128 -> emb_dim)
    let x = input_proj_forward(weights, "input_proj", &x, device)?;
    let x = x.mul(&output_image_mask)?;

    // t_cond = t_embedding(t); unsqueeze(1) for t.dim()==1 -> [B,1,emb_dim].
    // embed_scalar_t returns the already-unsqueezed [B,1,emb_dim] tensor.
    let t_cond = embed_scalar_t(
        weights,
        "t_embedding",
        t_vals,
        0.0, // input_range = (0.0, 1.0)
        1.0,
        config.emb_dim,
        device,
    )?;
    // adaln_input = silu(adaln_proj(t_cond))  -> [B,1,adanln_dim]
    let adaln_input = adaln_input_from_t_cond(weights, "adaln_proj", &t_cond, device)?;

    // llm_features = llm_cond_norm(llm_features); llm_cond_proj(...) * llm_token_mask
    let llm_features = llm_cond_forward(
        weights,
        "llm_cond_norm",
        "llm_cond_proj",
        &llm_features,
        device,
    )?;
    let llm_features = llm_features.mul(&llm_token_mask)?;

    // h = x + llm_features
    let mut h = x.add(&llm_features)?;

    // image_indicator_embedding = embed_image_indicator((indicator==OUTPUT_IMAGE).long())
    // index = 1 where OUTPUT_IMAGE, else 0.
    let indicator_index: Vec<i32> = indicator
        .iter()
        .map(|&i| if i == OUTPUT_IMAGE_INDICATOR { 1 } else { 0 })
        .collect();
    let img_ind_emb = image_indicator_embedding(
        weights,
        "embed_image_indicator.weight",
        &indicator_index,
        b,
        l,
        device,
    )?;
    // h = h + image_indicator_embedding
    h = h.add(&img_ind_emb)?;

    // cos, sin = rotary_emb(position_ids).to(h.dtype)  — half-table [1,S,head_dim/2] BF16.
    let head_dim = config.head_dim();
    let (cos, sin) = build_cos_sin(
        t_pos,
        h_pos,
        w_pos,
        head_dim,
        config.rope_theta,
        config.mrope_section_arr(),
        device,
    )?;

    // Block stack: 34 layers, sandwich-norm AdaLN.
    let norm_eps = config.norm_eps as f32;
    for i in 0..config.num_layers {
        let prefix = format!("layers.{i}");
        h = transformer_block_forward(
            weights,
            &prefix,
            &h,
            &cos,
            &sin,
            &adaln_input,
            segment_mask,
            config.num_heads,
            norm_eps,
            device,
        )?;
    }

    // out = final_layer(h, c=adaln_input); return out.to(float32).
    let out = final_layer_forward(weights, "final_layer", &h, &adaln_input, device)?;
    out.to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indicator_constants_match_reference() {
        // constants.py: OUTPUT_IMAGE_INDICATOR = 2, LLM_TOKEN_INDICATOR = 3.
        assert_eq!(OUTPUT_IMAGE_INDICATOR, 2);
        assert_eq!(LLM_TOKEN_INDICATOR, 3);
    }

    #[test]
    fn config_drives_layer_count_34() {
        let c = Ideogram4Config::default();
        assert_eq!(c.num_layers, 34);
    }

    // GPU-dependent end-to-end forward: requires CUDA device + FP8 dequant +
    // SDPA + all kernels. GPU busy → compile-only, ignored.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn transformer_forward_compiles() {
        let _ = super::transformer_forward;
    }
}
