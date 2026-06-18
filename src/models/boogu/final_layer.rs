//! Boogu-Image C5 — the final layer: `norm_out` (`LuminaLayerNormContinuous`,
//! affine-free LayerNorm) + channel-FASTEST unpatchify.
//!
//! Mirrors `boogu/models/transformers/block_lumina2.py::LuminaLayerNormContinuous.forward`
//! (lines 109-122) and the unpatchify `rearrange` in
//! `transformer_boogu.py::BooguImageTransformer2DModel.forward` (lines 1583-1591)
//! op-for-op, cross-checked against the verified Mojo C5
//! (`/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_c5_parity.mojo`,
//! handoff parity row C5: norm_out cos 0.9999943, unpatchify round-trip max_abs
//! 0.0 = bit-exact inverse).
//!
//! ## norm_out (`LuminaLayerNormContinuous`, constructed at transformer_boogu.py:937-944)
//!
//! ```text
//! # elementwise_affine=False (the LayerNorm has NO gamma/beta), eps=1e-6,
//! # conditioning_embedding_dim=1024, embedding_dim=3360, out_dim=64, bias=True.
//! emb   = norm_out.linear_1(silu(temb))          # BIAS Linear 1024 -> 3360
//! scale = emb
//! x     = LayerNorm(x, eps=1e-6, affine=False) * (1 + scale)[:, None, :]   # broadcast over seq
//! x     = norm_out.linear_2(x)                   # BIAS Linear 3360 -> 64
//! ```
//! The modulation `scale` is a per-sample `[B, 3360]` vector broadcast across the
//! sequence (`[:, None, :]`), exactly the `scale.unsqueeze(1).add_scalar(1.0)`
//! idiom the C3/C4 blocks already use. The LayerNorm is **affine-free** (no
//! weight/bias tensors in the checkpoint) and eps is **1e-6** — NOT the block
//! `norm_eps` of 1e-5 (see [`BooguConfig::NORM_OUT_EPS`]).
//!
//! ## unpatchify — channel-FASTEST `(p1 p2 c)` (transformer_boogu.py:1584-1591)
//!
//! ```text
//! img_output = rearrange(img_tokens, "(h w) (p1 p2 c) -> c (h p1) (w p2)",
//!                        h=Hl/p, w=Wl/p, p1=p, p2=p)
//! ```
//! In the patch group `(p1 p2 c)`, einops makes the LAST named dim the
//! FASTEST-varying, so the 64-wide patch vector is laid out
//! `[p1=0,p2=0,c=0..15, p1=0,p2=1,c=0..15, p1=1,p2=0,c=0..15, p1=1,p2=1,c=0..15]`
//! — i.e. **c innermost (fastest), then p2, then p1**. This is the SAME pack the
//! patchify in `flat_and_pad_to_seq` produces
//! (`"c (h p1) (w p2) -> (h w) (p1 p2 c)"`, line 1153).
//!
//! flame-core's `bf16_elementwise::unpatchify_bf16` kernel computes the source
//! index as `d = p_h*(p*C) + p_w*C + c` (bf16_elementwise.rs:384) — i.e. the
//! decode is `c = d % C`, `p_w = (d/C) % p`, `p_h = d/(p*C)`, so `c` is the
//! fastest-varying and `(p_h, p_w, c) == (p1, p2, c)`. **That is EXACTLY the
//! reference's channel-FASTEST `(p1 p2 c)` pack** (and its `patchify_bf16` is the
//! matching inverse, also channel-fastest — verified bit-exact round-trip in the
//! host test below). So we reuse `unpatchify_bf16` directly; NO hand-rolled
//! reshape is needed.
//!
//! Note flame-core ALSO has `ops::layout::unpatchify`, which is the standard
//! channel-SLOWEST DiT variant (`(c p1 p2)`) — we deliberately do NOT use it
//! (using it would scramble channels, the cos≈0.99 / high-max_abs fingerprint).
//!
//! ## Weight keys (prefix `norm_out`)
//! - `norm_out.linear_1.{weight,bias}` (1024 → 3360).
//! - `norm_out.linear_2.{weight,bias}` (3360 → 64).
//! - (No LayerNorm gamma/beta — `elementwise_affine=False`.)

use std::collections::HashMap;

use flame_core::bf16_ops::silu_bf16;
use flame_core::bf16_elementwise::unpatchify_bf16;
use flame_core::layer_norm::layer_norm;
use flame_core::ops::fused_inference::fused_linear3d_native;
use flame_core::{Error, Result, Tensor};

use super::config::BooguConfig;
use super::loader::get;

/// Weight-key prefix for the final-layer module.
const NORM_OUT_LINEAR_1: &str = "norm_out.linear_1";
const NORM_OUT_LINEAR_2: &str = "norm_out.linear_2";

/// `norm_out` (`LuminaLayerNormContinuous`, affine-free LayerNorm, temb-modulated).
///
/// `hidden` is the post-single-stream joint sequence `[B, S, hidden]` BF16; `temb`
/// is the C1 timestep embed `[B, 1, TIME_EMBED_DIM(1024)]` BF16. Returns the
/// projected patch tokens `[B, S, 64]` BF16 (= `patch² · out_channels`).
///
/// Op chain (block_lumina2.py:115-120):
/// `emb = linear_1(silu(temb))` (BIAS 1024→3360) →
/// `LayerNorm(hidden, eps=1e-6, affine=False) * (1 + emb)[:,None,:]` →
/// `linear_2(.)` (BIAS 3360→64).
pub fn norm_out(
    weights: &HashMap<String, Tensor>,
    cfg: &BooguConfig,
    hidden: &Tensor,
    temb: &Tensor,
) -> Result<Tensor> {
    let hd = hidden.shape().dims();
    if hd.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "boogu norm_out: hidden must be [B,S,hidden], got {hd:?}"
        )));
    }
    let (b, _s, hidden_dim) = (hd[0], hd[1], hd[2]);
    if hidden_dim != cfg.hidden_size {
        return Err(Error::InvalidOperation(format!(
            "boogu norm_out: hidden dim {hidden_dim} != cfg.hidden_size {}",
            cfg.hidden_size
        )));
    }

    // emb = linear_1(silu(temb)) -> [B, 1, hidden]. silu BEFORE the linear, per
    // LuminaLayerNormContinuous.forward (block_lumina2.py:115).
    let lin1_w = get(weights, &format!("{NORM_OUT_LINEAR_1}.weight"))?;
    let lin1_b = get(weights, &format!("{NORM_OUT_LINEAR_1}.bias"))?;
    let temb_silu = silu_bf16(temb)?;
    let emb = fused_linear3d_native(&temb_silu, lin1_w, Some(lin1_b))?; // [B,1,hidden]
    // scale broadcast: (1 + scale)[:, None, :]. Drop the singleton seq axis then
    // re-add it so the `+1` and broadcast match the C3/C4 modulation idiom.
    let scale = emb.reshape(&[b, cfg.hidden_size])?;
    let scale_b = scale.unsqueeze(1)?.add_scalar(1.0)?; // [B,1,hidden]

    // LayerNorm over the last dim (hidden), affine-free (weight=None, bias=None),
    // eps = NORM_OUT_EPS (1e-6, NOT the block 1e-5).
    let normed = layer_norm(hidden, &[cfg.hidden_size], None, None, BooguConfig::NORM_OUT_EPS)?;
    // x = normed * (1 + scale)[:, None, :]  (broadcast over the sequence axis).
    let modulated = normed.mul(&scale_b)?;

    // x = linear_2(x) -> [B, S, 64]  (BIAS Linear 3360 -> patch²·out_channels).
    let lin2_w = get(weights, &format!("{NORM_OUT_LINEAR_2}.weight"))?;
    let lin2_b = get(weights, &format!("{NORM_OUT_LINEAR_2}.bias"))?;
    fused_linear3d_native(&modulated, lin2_w, Some(lin2_b))
}

/// Channel-FASTEST unpatchify `(p1 p2 c) -> c (h p1) (w p2)`.
///
/// `patch_tokens` is `[B, Li, patch²·out_channels]` BF16 (the norm_out output
/// rows for the IMAGE tokens only — the caller extracts `[cap_len:seq]` BEFORE
/// calling this). `h_tok`/`w_tok` are the latent patch-grid dims (`Hl/p`, `Wl/p`)
/// so `Li == h_tok*w_tok`. Returns `[B, out_channels, Hl, Wl]` BF16.
///
/// Delegates to `flame_core::bf16_elementwise::unpatchify_bf16`, whose kernel is
/// the channel-FASTEST `(p1 p2 c)` variant (see module docs) — the matching
/// inverse of the reference patchify. We do NOT use `ops::layout::unpatchify`
/// (channel-slowest).
pub fn unpatchify(
    patch_tokens: &Tensor,
    cfg: &BooguConfig,
    h_tok: usize,
    w_tok: usize,
) -> Result<Tensor> {
    let pd = patch_tokens.shape().dims();
    if pd.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "boogu unpatchify: patch_tokens must be [B,Li,patch_dim], got {pd:?}"
        )));
    }
    let li = pd[1];
    let patch_dim = pd[2];
    let expected_patch_dim = cfg.patch_size * cfg.patch_size * cfg.out_channels; // 64
    if patch_dim != expected_patch_dim {
        return Err(Error::InvalidOperation(format!(
            "boogu unpatchify: patch_dim {patch_dim} != patch²·out_channels {expected_patch_dim}"
        )));
    }
    if li != h_tok * w_tok {
        return Err(Error::InvalidOperation(format!(
            "boogu unpatchify: token count {li} != h_tok*w_tok ({h_tok}*{w_tok})"
        )));
    }
    // bf16_elementwise::unpatchify_bf16(x, ph, pw, patch_size, in_channels)
    // -> [B, C, ph*p, pw*p]; here `in_channels` = out_channels (= 16).
    unpatchify_bf16(patch_tokens, h_tok, w_tok, cfg.patch_size, cfg.out_channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::bf16_elementwise::patchify_bf16;
    use flame_core::{DType, Shape};

    fn cfg() -> BooguConfig {
        BooguConfig::default()
    }

    #[test]
    fn norm_out_eps_is_1e_6_not_1e_5() {
        // The single biggest C5 trap: norm_out LayerNorm eps is 1e-6, distinct
        // from the block RMSNorm eps of 1e-5.
        let c = cfg();
        assert_eq!(BooguConfig::NORM_OUT_EPS, 1e-6);
        assert_ne!(BooguConfig::NORM_OUT_EPS, c.norm_eps);
    }

    #[test]
    fn out_patch_dim_is_64() {
        // norm_out.linear_2 maps hidden(3360) -> patch²·out_channels = 2·2·16 = 64.
        let c = cfg();
        assert_eq!(c.patch_size * c.patch_size * c.out_channels, 64);
    }

    /// Channel-FASTEST round-trip: patchify∘unpatchify == identity (the Mojo C5
    /// gate's bit-exact inverse check). Confirms flame-core's `patchify_bf16` /
    /// `unpatchify_bf16` pair are BOTH the `(p1 p2 c)` channel-fastest variant
    /// the reference uses — i.e. they are mutual inverses.
    #[test]
    #[ignore = "requires CUDA device; run with --ignored"]
    fn unpatchify_channel_fastest_round_trip_is_identity() {
        // `CudaDevice::new` already returns `Arc<CudaDevice>`.
        let device = cudarc::driver::CudaDevice::new(0).unwrap();
        let c = cfg();
        let p = c.patch_size; // 2
        let ch = c.out_channels; // 16
        let (h_tok, w_tok) = (4usize, 5usize); // non-square to catch transpose bugs
        let (hl, wl) = (h_tok * p, w_tok * p);
        let b = 1usize;

        // Build a deterministic [B, C, Hl, Wl] BF16 image.
        let n = b * ch * hl * wl;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 1.7).collect();
        let img = Tensor::from_vec_dtype(
            data.clone(),
            Shape::from_dims(&[b, ch, hl, wl]),
            device.clone(),
            DType::BF16,
        )
        .unwrap();

        // patchify -> [B, h_tok*w_tok, 64] (channel-fastest pack).
        let (patches, ph, pw) = patchify_bf16(&img, p).unwrap();
        assert_eq!(ph, h_tok);
        assert_eq!(pw, w_tok);
        assert_eq!(patches.shape().dims(), &[b, h_tok * w_tok, p * p * ch]);

        // unpatchify back -> [B, C, Hl, Wl]; must equal the original (bit-exact,
        // since both ops are pure gathers of the same BF16 values).
        let back = super::unpatchify(&patches, &c, h_tok, w_tok).unwrap();
        assert_eq!(back.shape().dims(), &[b, ch, hl, wl]);

        let orig = img.to_vec_f32().unwrap();
        let rt = back.to_vec_f32().unwrap();
        assert_eq!(orig.len(), rt.len());
        let mut max_abs = 0.0f32;
        for i in 0..orig.len() {
            max_abs = max_abs.max((orig[i] - rt[i]).abs());
        }
        assert_eq!(max_abs, 0.0, "channel-fastest round-trip not bit-exact: max_abs={max_abs}");
    }

    #[test]
    #[ignore = "requires CUDA device + checkpoint (GPU busy); compile-only this chunk"]
    fn norm_out_compiles() {
        let _ = super::norm_out;
        let _ = super::unpatchify;
    }
}
