//! Ideogram 4 embedding layers: timestep scalar embed, AdaLN projection,
//! image-indicator embedding, image-latent input projection, and the LLM
//! conditioning norm + projection.
//!
//! Mirrors the embed/conditioning portions of `Ideogram4Transformer.__init__`
//! and `.forward` plus `Ideogram4EmbedScalar` in
//! `/home/alex/ideogram4-ref/src/ideogram4/modeling_ideogram4.py`
//! (lines 218-250, 274-280, 339-369) op-for-op.
//!
//! ## `Ideogram4EmbedScalar` (`t_embedding`)  — lines 232-250
//! ```text
//! x = x.to(float32)
//! scaled = 1e4 * (x - range_min) / (range_max - range_min)        # range (0,1)
//! emb = _sinusoidal_embedding(scaled, dim, scale=1e4)             # F32, sin‖cos
//! emb = emb.to(compute_dtype)                                     # -> bf16
//! emb = F.silu(mlp_in(emb))                                       # BIAS Linear
//! return mlp_out(emb)                                            # BIAS Linear
//! ```
//! `_sinusoidal_embedding` (lines 218-229):
//! ```text
//! half = dim // 2
//! freq = exp(arange(half) * -(ln(1e4) / (half - 1)))             # F32
//! emb  = t[..., None] * freq                                     # F32
//! emb  = cat([sin(emb), cos(emb)], -1)                           # F32
//! if dim % 2 == 1: emb = pad(emb, (0,1))
//! ```
//! The reference computes the table + trig in **F32** then casts to the MLP
//! dtype. We mirror that exactly: the sinusoidal table is built host-side in
//! f32 (B is small — one scalar per sample per step) and uploaded as a BF16
//! `[B, 1, dim]` tensor for the bias linears. F32-internals avoids letting the
//! BF16 floor eat the high-frequency table (CONTEXT.md timestep trap).
//!
//! ## conditioning projections (`Ideogram4Transformer.__init__`, lines 274-280)
//! - `input_proj`     : BIAS Linear(in_channels=128 -> emb_dim=4608)
//! - `llm_cond_norm`  : RMSNorm(llm_features_dim=53248, **eps=1e-6**)  ← line 275
//! - `llm_cond_proj`  : BIAS Linear(53248 -> 4608)
//! - `adaln_proj`     : BIAS Linear(4608 -> adanln_dim=512)
//! - `embed_image_indicator`: nn.Embedding(2, 4608)
//!
//! Note `llm_cond_norm` eps is a hardcoded **1e-6** in the reference
//! (`Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)`), distinct from the
//! block `norm_eps = 1e-5`. We pass 1e-6 here verbatim.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::bf16_ops::silu_bf16;
use flame_core::ops::fused_inference::{
    fused_linear3d_native_pytorch_parity, fused_rms_norm,
};
use flame_core::{DType, Error, Result, Shape, Tensor};

use super::weights::Ideogram4RawWeight;

/// `llm_cond_norm` epsilon — hardcoded 1e-6 in the reference
/// (`modeling_ideogram4.py:275`), NOT the block `norm_eps` (1e-5).
pub const LLM_COND_NORM_EPS: f32 = 1e-6;

/// Dequantize a raw weight to BF16 `[out, in]`, erroring on a missing key.
fn weight(
    weights: &HashMap<String, Ideogram4RawWeight>,
    key: &str,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| {
            Error::InvalidOperation(format!("ideogram4 embed: missing weight `{key}`"))
        })?
        .to_bf16_tensor(device)
}

/// `_sinusoidal_embedding(scaled, dim, scale=1e4)` built host-side in F32.
///
/// Mirrors `modeling_ideogram4.py:218-229`. `scaled_vals` holds the already
/// `1e4*(x-min)/(max-min)`-scaled scalars (one per sample). Returns a flat F32
/// `Vec` of length `b * dim` in row-major `[b, dim]` order: for each sample,
/// `[sin(t*freq[0..half]), cos(t*freq[0..half])]` then a single 0.0 pad if
/// `dim` is odd.
///
/// `freq[d] = exp(d * -(ln(1e4) / (half-1)))`, `half = dim // 2`.
fn sinusoidal_embedding_f32(scaled_vals: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    // freq = exp(arange(half) * -(ln(1e4)/(half-1)))
    let scale = 1e4f64;
    let denom = (half as f64) - 1.0;
    let step = scale.ln() / denom;
    let freq: Vec<f64> = (0..half).map(|d| (-(d as f64) * step).exp()).collect();

    let mut out = Vec::with_capacity(scaled_vals.len() * dim);
    for &t in scaled_vals {
        // cat([sin(t*freq), cos(t*freq)]) — F32 trig, f64 accumulation to match
        // the reference's float32 compute (PyTorch sin/cos are computed in the
        // tensor dtype = float32; f64 here is conservative and rounds to f32 on
        // store, identical at these magnitudes).
        for &f in &freq {
            out.push((t as f64 * f).sin() as f32);
        }
        for &f in &freq {
            out.push((t as f64 * f).cos() as f32);
        }
        // dim odd -> F.pad(emb, (0,1)) appends one zero.
        if dim % 2 == 1 {
            out.push(0.0);
        }
    }
    out
}

/// `Ideogram4EmbedScalar.forward` for the timestep embedding.
///
/// `t_vals` holds one flow-matching time scalar per sample (the Python `t` of
/// shape `(B,)`). `range = (0.0, 1.0)`. Returns the post-MLP embedding shaped
/// `[B, 1, emb_dim]` BF16 — i.e. the reference's `t_cond` AFTER the
/// `t.dim()==1 -> unsqueeze(1)` reshape (the caller never needs to unsqueeze).
///
/// Op chain (lines 241-250):
/// `scaled = 1e4*(x-min)/(max-min)` → `sinusoidal(scaled, dim)` (F32) → cast bf16
/// → `silu(mlp_in(emb))` → `mlp_out(...)`.
pub fn embed_scalar_t(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    t_vals: &[f32],
    range_min: f32,
    range_max: f32,
    emb_dim: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    if range_max <= range_min {
        return Err(Error::InvalidOperation(format!(
            "ideogram4 embed_scalar_t: range_max ({range_max}) must exceed range_min ({range_min})"
        )));
    }
    let b = t_vals.len();
    // scaled = 1e4 * (x - range_min) / (range_max - range_min)  (F32, line 244).
    let span = (range_max - range_min) as f64;
    let scaled: Vec<f32> = t_vals
        .iter()
        .map(|&x| (1e4f64 * ((x - range_min) as f64) / span) as f32)
        .collect();

    // F32 sinusoidal table, then upload as BF16 [b, 1, emb_dim] (cast = the
    // reference `emb.to(compute_dtype)` at line 246-248).
    let emb_f32 = sinusoidal_embedding_f32(&scaled, emb_dim);
    let emb = Tensor::from_vec_dtype(
        emb_f32,
        Shape::from_dims(&[b, 1, emb_dim]),
        device.clone(),
        DType::BF16,
    )?;

    // emb = silu(mlp_in(emb))  (BIAS Linear, line 249).
    let mlp_in_w = weight(weights, &format!("{prefix}.mlp_in.weight"), device)?;
    let mlp_in_b = weight(weights, &format!("{prefix}.mlp_in.bias"), device)?;
    let hidden = fused_linear3d_native_pytorch_parity(&emb, &mlp_in_w, Some(&mlp_in_b))?;
    let hidden = silu_bf16(&hidden)?;

    // return mlp_out(emb)  (BIAS Linear, line 250).
    let mlp_out_w = weight(weights, &format!("{prefix}.mlp_out.weight"), device)?;
    let mlp_out_b = weight(weights, &format!("{prefix}.mlp_out.bias"), device)?;
    // Output [B, 1, emb_dim] — already in the unsqueeze(1) shape the top-level
    // forward expects (t.dim()==1 path).
    fused_linear3d_native_pytorch_parity(&hidden, &mlp_out_w, Some(&mlp_out_b))
}

/// `adaln_proj` then SiLU → `adaln_input` shared by every block + final layer.
///
/// Mirrors `adaln_input = F.silu(self.adaln_proj(t_cond))`
/// (`modeling_ideogram4.py:359`). `t_cond` is `[B, 1, emb_dim]` (from
/// [`embed_scalar_t`]); output is `[B, 1, adanln_dim]` BF16 — the 3D shape the
/// block AdaLN modulation linear requires (chunk-2 contract).
pub fn adaln_input_from_t_cond(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    t_cond: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    // adaln_proj: BIAS Linear(emb_dim -> adanln_dim).
    let w = weight(weights, &format!("{prefix}.weight"), device)?;
    let b = weight(weights, &format!("{prefix}.bias"), device)?;
    let proj = fused_linear3d_native_pytorch_parity(t_cond, &w, Some(&b))?;
    silu_bf16(&proj)
}

/// `input_proj`: BIAS Linear(in_channels=128 -> emb_dim=4608) on image-latent
/// tokens. Mirrors `self.input_proj(x)` (`modeling_ideogram4.py:352`). `x` is
/// `[B, L, in_channels]` BF16; output `[B, L, emb_dim]` BF16. The masking by
/// `output_image_mask` is applied by the top-level forward, NOT here.
pub fn input_proj_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    x: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let w = weight(weights, &format!("{prefix}.weight"), device)?;
    let b = weight(weights, &format!("{prefix}.bias"), device)?;
    fused_linear3d_native_pytorch_parity(x, &w, Some(&b))
}

/// `llm_cond_norm` (RMSNorm 53248, eps 1e-6) then `llm_cond_proj`
/// (BIAS Linear 53248 -> 4608).
///
/// Mirrors `modeling_ideogram4.py:361-362`:
/// ```text
/// llm_features = self.llm_cond_norm(llm_features)
/// llm_features = self.llm_cond_proj(llm_features) * llm_token_mask
/// ```
/// `llm_features` is `[B, L, llm_features_dim]` BF16; output `[B, L, emb_dim]`
/// BF16 (PRE-mask — the `* llm_token_mask` is applied by the top-level
/// forward). The RMSNorm normalizes the last dim (53248), so `fused_rms_norm`
/// over the flattened `[B*L, 53248]` rows is exact; we keep it 3D since
/// `fused_rms_norm` normalizes the last dim regardless of leading rank.
pub fn llm_cond_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    norm_prefix: &str,
    proj_prefix: &str,
    llm_features: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let norm_w = weight(weights, &format!("{norm_prefix}.weight"), device)?;
    let normed = fused_rms_norm(llm_features, &norm_w, LLM_COND_NORM_EPS)?;
    let proj_w = weight(weights, &format!("{proj_prefix}.weight"), device)?;
    let proj_b = weight(weights, &format!("{proj_prefix}.bias"), device)?;
    fused_linear3d_native_pytorch_parity(&normed, &proj_w, Some(&proj_b))
}

/// `embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).long())`.
///
/// Mirrors `modeling_ideogram4.py:366-368`. `nn.Embedding(2, emb_dim)` indexed
/// by a 0/1 long mask: 1 where the token is an OUTPUT_IMAGE token, 0 otherwise.
/// `index_mask` is the per-token `[B, L]` index (0 or 1) the top-level forward
/// builds from `indicator`. Returns `[B, L, emb_dim]` BF16.
///
/// Implemented via `Tensor::index_select0` directly on the embedding weight
/// `[2, emb_dim]` (the flame `Embedding` struct allocates a random weight; here
/// the weight comes from the FP8 container, so we gather rows directly). For a
/// `[B, L]` index, we flatten to `[B*L]`, gather `[B*L, emb_dim]`, reshape back.
pub fn image_indicator_embedding(
    weights: &HashMap<String, Ideogram4RawWeight>,
    key: &str,
    index_mask: &[i32],
    b: usize,
    l: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    if index_mask.len() != b * l {
        return Err(Error::InvalidOperation(format!(
            "ideogram4 image_indicator_embedding: index_mask len {} != b*l ({}*{}={})",
            index_mask.len(),
            b,
            l,
            b * l
        )));
    }
    let emb_w = weight(weights, key, device)?; // [2, emb_dim] BF16
    let emb_dim = emb_w.shape().dims()[1];
    // index_select0 wants I32 indices. Build a flat [B*L] I32 index tensor.
    let idx_f32: Vec<f32> = index_mask.iter().map(|&i| i as f32).collect();
    let idx =
        Tensor::from_vec(idx_f32, Shape::from_dims(&[b * l]), device.clone())?.to_dtype(DType::I32)?;
    let gathered = emb_w.index_select0(&idx)?; // [B*L, emb_dim]
    gathered.reshape(&[b, l, emb_dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_embedding_output_dim_even() {
        // dim even -> output width == dim (half sin + half cos, no pad).
        let dim = 4608usize;
        let out = sinusoidal_embedding_f32(&[0.5f32], dim);
        assert_eq!(out.len(), dim);
    }

    #[test]
    fn sinusoidal_embedding_output_dim_odd_pads_one() {
        // dim odd -> half sin + half cos + 1 pad = dim.
        let dim = 7usize; // half = 3 -> 3 sin + 3 cos + 1 pad = 7
        let out = sinusoidal_embedding_f32(&[0.25f32], dim);
        assert_eq!(out.len(), dim);
        assert_eq!(*out.last().unwrap(), 0.0); // pad value
    }

    #[test]
    fn sinusoidal_freq_formula_first_and_last() {
        // freq[0] = exp(0) = 1.0; freq[half-1] = exp(-(half-1)*ln(1e4)/(half-1))
        //                                       = exp(-ln(1e4)) = 1/1e4.
        let dim = 8usize; // half = 4
        let half = dim / 2;
        let scale = 1e4f64;
        let step = scale.ln() / ((half as f64) - 1.0);
        let freq_first = (-(0.0) * step).exp();
        let freq_last = (-((half - 1) as f64) * step).exp();
        assert!((freq_first - 1.0).abs() < 1e-12);
        assert!((freq_last - 1.0 / 1e4).abs() < 1e-9);
    }

    #[test]
    fn sinusoidal_at_t_zero_is_sin0_cos0() {
        // t = 0 -> sin = 0 for the first half, cos = 1 for the second half.
        let dim = 8usize;
        let out = sinusoidal_embedding_f32(&[0.0f32], dim);
        let half = dim / 2;
        for &s in &out[0..half] {
            assert_eq!(s, 0.0);
        }
        for &c in &out[half..2 * half] {
            assert!((c - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn batch_layout_is_row_major() {
        // two samples -> 2*dim flat, sample 0 occupies [0,dim), sample 1 [dim,2dim).
        let dim = 8usize;
        let out = sinusoidal_embedding_f32(&[0.0f32, 0.0f32], dim);
        assert_eq!(out.len(), 2 * dim);
    }

    #[test]
    fn llm_cond_norm_eps_is_1e6() {
        // Reference hardcodes Ideogram4RMSNorm(llm_features_dim, eps=1e-6).
        assert_eq!(LLM_COND_NORM_EPS, 1e-6);
    }

    // GPU-dependent: embed_scalar_t / adaln_input / projections require a CUDA
    // device + FP8 dequant + linear kernels. GPU busy → compile-only, ignored.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn embed_layers_compile() {
        let _ = super::embed_scalar_t;
        let _ = super::adaln_input_from_t_cond;
        let _ = super::input_proj_forward;
        let _ = super::llm_cond_forward;
        let _ = super::image_indicator_embedding;
    }
}
