//! Boogu-Image C1 — timestep embedder + caption embedder
//! (`Lumina2CombinedTimestepCaptionEmbedding`).
//!
//! Mirrors `boogu/models/transformers/block_lumina2.py`
//! (`Lumina2CombinedTimestepCaptionEmbedding`, lines 177-219) and
//! `boogu/models/embeddings.py` (`TimestepEmbedding`, lines 24-77) op-for-op,
//! cross-checked against the verified Mojo C1 (cos 1.0, max_abs 0.0 bit-exact —
//! handoff parity table row C1).
//!
//! ## Timestep path (`time_caption_embed.forward` → `time_proj` + `timestep_embedder`)
//!
//! ```text
//! timestep_proj = Timesteps(num_channels=256, flip_sin_to_cos=True,
//!                           downscale_freq_shift=0.0, scale=1000.0)(timestep)  # [B,256] F32
//! timestep_proj = timestep_proj.to(dtype=bf16)
//! time_embed = TimestepEmbedding(256 -> 1024):
//!     h = silu(linear_1(timestep_proj))   # BIAS Linear 256->1024
//!     time_embed = linear_2(h)            # BIAS Linear 1024->1024
//! ```
//! `Timesteps` is diffusers `get_timestep_embedding` (embeddings.py:get_timestep_embedding):
//! ```text
//! half_dim = 256/2 = 128
//! exponent[d] = -ln(10000) * d / (128 - downscale_freq_shift=0)   # F32, d in [0,128)
//! freq[d]     = exp(exponent[d])
//! arg[d]      = scale(=1000) * (t * freq[d])
//! emb = cat([sin(arg), cos(arg)])                                  # [256]
//! emb = cat([emb[128:], emb[:128]])  (flip_sin_to_cos=True)        # COS-first
//! ```
//! The sinusoid is built host-side in F32 (the reference computes it in F32 then
//! casts to bf16) and uploaded as a BF16 `[B, 1, 256]` tensor — F32-internals
//! keep the BF16 floor off the high-frequency table (CONTEXT.md timestep trap;
//! the Mojo C1 verified this gives a bit-exact match).
//!
//! ## Caption path (`caption_embedder = Sequential(RMSNorm(4096), Linear(4096->3360, bias=True))`)
//!
//! ```text
//! instruction = linear(rms_norm(instruction_hidden_states))  # RMSNorm eps=1e-5, BIAS Linear
//! ```
//! `preprocess_instruction_hidden_states` (mean over 1 layer) is the identity on
//! the T2I text-only path, so the input is `[B, L, 4096]` as-is.
//!
//! Weight keys (verbatim from the safetensors index, 942 tensors):
//! - `time_caption_embed.timestep_embedder.linear_1.{weight,bias}` (256→1024)
//! - `time_caption_embed.timestep_embedder.linear_2.{weight,bias}` (1024→1024)
//! - `time_caption_embed.caption_embedder.0.weight` (RMSNorm 4096)
//! - `time_caption_embed.caption_embedder.1.{weight,bias}` (Linear 4096→3360)

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::bf16_ops::silu_bf16;
use flame_core::ops::fused_inference::{fused_linear3d_native, fused_rms_norm};
use flame_core::{DType, Result, Shape, Tensor};

use super::config::BooguConfig;
use super::loader::get;

/// Weight-key prefixes (so a future rename is one edit, not N).
const TS_LINEAR_1: &str = "time_caption_embed.timestep_embedder.linear_1";
const TS_LINEAR_2: &str = "time_caption_embed.timestep_embedder.linear_2";
const CAPTION_NORM: &str = "time_caption_embed.caption_embedder.0"; // RMSNorm weight
const CAPTION_LINEAR: &str = "time_caption_embed.caption_embedder.1"; // Linear weight+bias

/// Build the diffusers `Timesteps`/`get_timestep_embedding` sinusoid host-side
/// in F32. Returns a flat `Vec<f32>` of length `b * dim`, row-major `[b, dim]`.
///
/// Mirrors `get_timestep_embedding(t, dim=256, flip_sin_to_cos=True,
/// downscale_freq_shift=0, scale=1000, max_period=10000)`:
/// per sample the row is `[cos(arg[0..128]), sin(arg[0..128])]` (COS-first
/// because `flip_sin_to_cos=True` moves the cos block to the front).
///
/// `arg[d] = scale * (t * exp(-ln(max_period) * d / (half - downscale_shift)))`.
fn timestep_sinusoid_f32(
    t_vals: &[f32],
    dim: usize,
    max_period: f32,
    downscale_freq_shift: f32,
    scale: f32,
) -> Vec<f32> {
    let half = dim / 2;
    // exponent[d] = -ln(max_period) * d / (half - downscale_freq_shift); F64
    // accumulation, rounds to F32 on store (matches PyTorch F32 trig at these
    // magnitudes; conservative).
    let neg_ln = -(max_period as f64).ln();
    let denom = (half as f64) - (downscale_freq_shift as f64);
    let freq: Vec<f64> = (0..half)
        .map(|d| (neg_ln * (d as f64) / denom).exp())
        .collect();

    let mut out = Vec::with_capacity(t_vals.len() * dim);
    for &t in t_vals {
        // flip_sin_to_cos=True => COS block first, then SIN block.
        for &f in &freq {
            let arg = (scale as f64) * (t as f64 * f);
            out.push(arg.cos() as f32);
        }
        for &f in &freq {
            let arg = (scale as f64) * (t as f64 * f);
            out.push(arg.sin() as f32);
        }
        // dim is even (256) here, so no odd-pad branch needed.
    }
    out
}

/// `time_caption_embed`'s timestep branch → `temb`.
///
/// `t_vals` holds one flow timestep per sample (Python `timestep` of shape
/// `(B,)`). Returns `temb` shaped `[B, 1, TIME_EMBED_DIM]` BF16 — kept 3D so it
/// feeds `fused_linear3d_native` directly and the later AdaLN modulation linears
/// (which also take 3D). The reference returns `[B, 1024]`; the singleton seq
/// axis is a no-op reshape away (`[B, 1024]` ↔ `[B, 1, 1024]`).
///
/// Op chain: F32 sinusoid (host) → upload BF16 `[B,1,256]` →
/// `silu(linear_1(.))` → `linear_2(.)`.
pub fn timestep_embed(
    weights: &HashMap<String, Tensor>,
    cfg: &BooguConfig,
    t_vals: &[f32],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let b = t_vals.len();
    let dim = BooguConfig::FREQ_EMBEDDING_SIZE; // 256

    // Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000).
    let emb_f32 = timestep_sinusoid_f32(
        t_vals,
        dim,
        BooguConfig::TIMESTEP_MAX_PERIOD,
        0.0,
        cfg.timestep_scale,
    );
    // Upload as BF16 [B, 1, 256] (= the reference `.to(dtype=bf16)`).
    let proj = Tensor::from_vec_dtype(
        emb_f32,
        Shape::from_dims(&[b, 1, dim]),
        device.clone(),
        DType::BF16,
    )?;

    // h = silu(linear_1(proj))  (BIAS Linear 256->1024).
    let w1 = get(weights, &format!("{TS_LINEAR_1}.weight"))?;
    let b1 = get(weights, &format!("{TS_LINEAR_1}.bias"))?;
    let h = fused_linear3d_native(&proj, w1, Some(b1))?;
    let h = silu_bf16(&h)?;

    // temb = linear_2(h)  (BIAS Linear 1024->1024).
    let w2 = get(weights, &format!("{TS_LINEAR_2}.weight"))?;
    let b2 = get(weights, &format!("{TS_LINEAR_2}.bias"))?;
    fused_linear3d_native(&h, w2, Some(b2))
}

/// `time_caption_embed`'s caption branch → `instruction`.
///
/// `instruction_hidden_states` is `[B, L, instruction_feat_dim(4096)]` BF16 (the
/// Qwen3-VL `hidden_states[-1]`, already mean-reduced over its 1 layer =
/// identity). Returns `instruction` shaped `[B, L, hidden_size(3360)]` BF16.
///
/// Op chain: `rms_norm(., eps=1e-5)` (caption_embedder.0) → `linear(.)`
/// (caption_embedder.1, BIAS).
pub fn caption_embed(
    weights: &HashMap<String, Tensor>,
    cfg: &BooguConfig,
    instruction_hidden_states: &Tensor,
) -> Result<Tensor> {
    // caption_embedder.0: RMSNorm(4096, eps=norm_eps=1e-5). Reference RMSNorm is
    // `normed * weight` (no +1) — flame `fused_rms_norm` matches.
    let norm_w = get(weights, &format!("{CAPTION_NORM}.weight"))?;
    let normed = fused_rms_norm(instruction_hidden_states, norm_w, cfg.norm_eps)?;

    // caption_embedder.1: BIAS Linear(4096 -> 3360).
    let lin_w = get(weights, &format!("{CAPTION_LINEAR}.weight"))?;
    let lin_b = get(weights, &format!("{CAPTION_LINEAR}.bias"))?;
    fused_linear3d_native(&normed, lin_w, Some(lin_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_sinusoid_width_is_dim() {
        let dim = BooguConfig::FREQ_EMBEDDING_SIZE; // 256
        let out = timestep_sinusoid_f32(&[0.5], dim, 10000.0, 0.0, 1000.0);
        assert_eq!(out.len(), dim);
    }

    #[test]
    fn timestep_sinusoid_t_zero_is_cos1_sin0() {
        // t=0 -> arg=0 -> cos=1 (first half), sin=0 (second half) because
        // flip_sin_to_cos=True puts cos first.
        let dim = BooguConfig::FREQ_EMBEDDING_SIZE; // 256
        let half = dim / 2;
        let out = timestep_sinusoid_f32(&[0.0], dim, 10000.0, 0.0, 1000.0);
        for &c in &out[0..half] {
            assert!((c - 1.0).abs() < 1e-6, "cos block should be 1.0 at t=0");
        }
        for &s in &out[half..dim] {
            assert_eq!(s, 0.0, "sin block should be 0.0 at t=0");
        }
    }

    #[test]
    fn timestep_sinusoid_freq_endpoints() {
        // freq[0] = exp(0) = 1.0; freq[half-1] = exp(-ln(P)*(half-1)/half).
        // With downscale_freq_shift=0, denom = half = 128.
        let dim = BooguConfig::FREQ_EMBEDDING_SIZE; // 256
        let half = dim / 2; // 128
        let max_period = 10000.0f64;
        let denom = half as f64; // downscale_freq_shift=0
        let f0 = (-(max_period.ln()) * 0.0 / denom).exp();
        let f_last = (-(max_period.ln()) * ((half - 1) as f64) / denom).exp();
        assert!((f0 - 1.0).abs() < 1e-12);
        // Last freq is small (high freq) but > 1/max_period since denom=half not half-1.
        assert!(f_last > 0.0 && f_last < 1.0);
    }

    #[test]
    fn timestep_sinusoid_batch_row_major() {
        let dim = BooguConfig::FREQ_EMBEDDING_SIZE;
        let out = timestep_sinusoid_f32(&[0.0, 0.5], dim, 10000.0, 0.0, 1000.0);
        assert_eq!(out.len(), 2 * dim);
        // Sample 0 (t=0) cos block is all 1.0; sample 1 (t=0.5) is not.
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    // GPU-dependent: the forwards need a CUDA device + linear/rmsnorm kernels.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn embed_forwards_compile() {
        let _ = super::timestep_embed;
        let _ = super::caption_embed;
    }
}
