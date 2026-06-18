//! NAVA (= Ovi joint audio-video MM-DiT) — transformer blocks, heads, and the
//! top-level `WanAVModel` forward.
//!
//! Built on the foundation in [`super::nava_av`] (config, embeddings, RoPE
//! tables, `apply_audio_rope_1d`). Mirrors `wan22_dit.rs` for the single-block
//! template and `ltx2_model.rs` for the dual-stream topology.
//!
//! ## Reference (read line-by-line, do not infer)
//!   - `ports/nava/nava_src/models/nava/modules/model_mm.py`
//!     * `WanAttentionBlock.forward` (~1016-1098) — SINGLE/fusion block.
//!     * `WanDoubleStreamAttentionBlock.forward` (~829-953) — DOUBLE/alignment block.
//!     * `WanDoubleStreamSelfAttention.forward` (~339-421) — joint self-attn.
//!     * `WanT2VCrossAttention` (~570-596) + `WanT2VDoubleStreamCrossAttention`
//!       (~598-675) — text cross-attn (QK-norm, NO rope).
//!     * `Head.forward` (~1118-1128), `unpatchify` (~1693-1719).
//!     * `WanAVModel.forward` (~1574-1691).
//!   - `ports/nava/nava_src/models/nava/modules/model.py`
//!     * `rope_apply_3d` (~72-100), `ModulationAdd` (~748-754),
//!       `WanRMSNorm` / `WanLayerNorm` (~203-232).
//!
//! ## Joint-attention simplification (BUILD_PLAN, verified by 3 research agents)
//! At B=1 unpadded, the argsort valid-token packing is the IDENTITY permutation
//! (every token is valid, gather/scatter are no-ops). So joint self-attention =
//!   `cat([vid, audio]) → rope each → SDPA → split at max_seq_len_vid`.
//! No argsort / gather / scatter is built (`model_mm.py:339,493`).
//!
//! ## Dtype discipline (PORT_SPEC / BUILD_PLAN: HIGH risk)
//! ALL modulation / gate / e0 math is BF16 — NAVA runs the AdaLN chain under
//! `amp.autocast('cuda', dtype=torch.bfloat16)` and asserts `e.dtype==bfloat16`.
//! Do NOT upgrade to F32 (fails parity — see SKEPTIC_FINDINGS F-checks). The
//! wan22 block does this on CPU/F32; we deliberately diverge and stay BF16/GPU.
//!
//! ## QKV kept SPLIT
//! `self_attn.{q,k,v,o}` (+ `..._audio` on double blocks) are 3 separate
//! `fused_linear3d_native` calls — matches the checkpoint layout. No fused QKV.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};

use super::nava_av::{apply_audio_rope_1d, NavaAVConfig, NavaRopeTables, AUDIO_ROPE_DIM};

// ===========================================================================
// Small shared helpers
// ===========================================================================

fn w<'a>(weights: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidInput(format!("nava: missing weight: {key}")))
}

fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
}

/// `WanLayerNorm` (no affine): LN over last dim, eps. BF16 in/out.
fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
    out.reshape(&dims)
}

/// `WanLayerNorm` (affine — only `norm3` / cross_attn_norm). BF16 in/out.
fn layer_norm_affine(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, Some(weight), Some(bias), eps)?;
    out.reshape(&dims)
}

/// `WanRMSNorm` QK-norm (full-dim, applied pre head-reshape). BF16.
fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
    out.reshape(&dims)
}

/// FFN: `Linear(dim,ffn_dim) → GELU(tanh) → Linear(ffn_dim,dim)`. BF16.
/// `Tensor::gelu()` is the tanh approximation (BUILD_PLAN gate), matching
/// `nn.GELU(approximate='tanh')`.
fn ffn(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let h = linear_bias(x, w1, b1)?;
    let h = h.gelu()?;
    linear_bias(&h, w2, b2)
}

/// AdaLN modulate: `norm(x) * (1 + scale) + shift`, all BF16. Wraps
/// `fused_modulate` (`x*(1+scale)+shift`). Shapes must match (`[1,seq,dim]`).
fn modulate(norm_x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    flame_core::ops::fused_inference::fused_modulate(norm_x, scale, shift)
}

/// Gated residual: `x + gate * y`, all BF16. Wraps `fused_residual_gate`.
fn gated_residual(x: &Tensor, y: &Tensor, gate: &Tensor) -> Result<Tensor> {
    flame_core::ops::fused_inference::fused_residual_gate(x, y, gate)
}

/// Compute the six BF16 modulation tensors for a stream.
///
/// `block_mod`: `modulation.modulation` param `[1, 6, dim]` BF16.
/// `e0`: `[1, seq, 6, dim]` BF16 (per-token time projection for this stream).
/// Returns `[e0..e5]`, each `[1, seq, dim]` BF16, matching
/// `(modulation(e).chunk(6, dim=2))` with `.squeeze(2)`.
///
/// `ModulationAdd.forward` = `modulation.bfloat16() + e.bfloat16()`; the block
/// then `.chunk(6, dim=2)`. The param broadcasts over the seq axis.
fn block_modulation(block_mod: &Tensor, e0: &Tensor) -> Result<[Tensor; 6]> {
    let dims = e0.shape().dims();
    debug_assert_eq!(dims.len(), 4, "e0 must be [1, seq, 6, dim]");
    let (b, seq, six, dim) = (dims[0], dims[1], dims[2], dims[3]);
    debug_assert_eq!(six, 6);
    // block_mod [1,6,dim] -> [1,1,6,dim] to broadcast over seq.
    let bm = block_mod.reshape(&[1, 1, six, dim])?;
    // Broadcast add (TensorIterator slow path handles the broadcast). BF16.
    let e_mod = e0.add(&bm)?; // [1, seq, 6, dim]
    let pick = |i: usize| -> Result<Tensor> {
        e_mod
            .narrow(2, i, 1)? // [1, seq, 1, dim]
            .reshape(&[b, seq, dim])?
            .contiguous() // dense for the fused elementwise kernels
    };
    Ok([pick(0)?, pick(1)?, pick(2)?, pick(3)?, pick(4)?, pick(5)?])
}

// ===========================================================================
// Video 3D RoPE apply (fused interleaved kernel)
// ===========================================================================
//
// `rope_apply_3d` (model.py:72-100) rotates the first `seq=f*h*w` tokens with
// per-token freqs `cat([t_freq[f], h_freq[h], w_freq[w]])` (interleaved, 64
// complex = 128 real), passing through any padding tail. At B=1 unpadded the
// whole video block is rotated.
//
// We assemble a per-token cos/sin table `[1,1,seq,head_dim/2]` by gathering the
// 3-axis F32 freq tables per (f,h,w), cast to BF16, then call the interleaved
// `rope_fused_bf16` kernel — same convention as the audio wrapper. This avoids
// a per-element CPU rope loop and keeps the rotation on-GPU/fused.

/// Build a `[1, 1, seq, head_dim/2]` BF16 cos and sin table for the video grid.
fn video_rope_cos_sin(
    tables: &NavaRopeTables,
    grid: (usize, usize, usize),
    head_dim: usize,
    device: Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let (f, h, ww) = grid;
    let seq = f * h * ww;
    let half = head_dim / 2; // 64
    let (at, ah, aw) = tables.video_axes; // (44, 42, 42)
    let (ath, ahh, awh) = (at / 2, ah / 2, aw / 2); // (22, 21, 21)
    debug_assert_eq!(ath + ahh + awh, half);

    let t_cos = tables.video_cos[0].to_vec1::<f32>()?;
    let t_sin = tables.video_sin[0].to_vec1::<f32>()?;
    let h_cos = tables.video_cos[1].to_vec1::<f32>()?;
    let h_sin = tables.video_sin[1].to_vec1::<f32>()?;
    let w_cos = tables.video_cos[2].to_vec1::<f32>()?;
    let w_sin = tables.video_sin[2].to_vec1::<f32>()?;

    let mut cos_data = vec![0.0f32; seq * half];
    let mut sin_data = vec![0.0f32; seq * half];
    for si in 0..seq {
        let fi = si / (h * ww);
        let hi = (si % (h * ww)) / ww;
        let wi = si % ww;
        let base = si * half;
        // axis 0: frame (first `ath` complex pairs)
        for i in 0..ath {
            cos_data[base + i] = t_cos[fi * ath + i];
            sin_data[base + i] = t_sin[fi * ath + i];
        }
        // axis 1: height
        for i in 0..ahh {
            cos_data[base + ath + i] = h_cos[hi * ahh + i];
            sin_data[base + ath + i] = h_sin[hi * ahh + i];
        }
        // axis 2: width
        for i in 0..awh {
            cos_data[base + ath + ahh + i] = w_cos[wi * awh + i];
            sin_data[base + ath + ahh + i] = w_sin[wi * awh + i];
        }
    }
    let cos = Tensor::from_vec(cos_data, Shape::from_dims(&[1, 1, seq, half]), device.clone())?
        .to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(sin_data, Shape::from_dims(&[1, 1, seq, half]), device)?
        .to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Apply video 3D interleaved RoPE to a `[1, H, seq, head_dim]` Q/K tensor.
fn apply_video_rope_3d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    flame_core::bf16_ops::rope_fused_bf16(x, cos, sin)
}

/// Slice an audio rope F32 table to `seq` and shape `[1,1,seq,AUDIO_ROPE_DIM/2]` BF16.
fn audio_rope_cos_sin(tables: &NavaRopeTables, seq: usize) -> Result<(Tensor, Tensor)> {
    let half = AUDIO_ROPE_DIM / 2;
    let cos = tables
        .audio_cos
        .narrow(0, 0, seq)?
        .reshape(&[1, 1, seq, half])?
        .to_dtype(DType::BF16)?;
    let sin = tables
        .audio_sin
        .narrow(0, 0, seq)?
        .reshape(&[1, 1, seq, half])?
        .to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

// ===========================================================================
// Attention primitives
// ===========================================================================

/// Reshape `[1, S, dim]` → SDPA layout `[1, H, S, D]`.
fn to_sdpa_layout(x: &Tensor, nh: usize, hd: usize) -> Result<Tensor> {
    let s = x.shape().dims()[1];
    // [1, S, dim] -> [1, S, H, D] -> [1, H, S, D]
    x.reshape(&[1, s, nh, hd])?.permute(&[0, 2, 1, 3])
}

/// Reshape SDPA output `[1, H, S, D]` → `[1, S, dim]`.
fn from_sdpa_layout(x: &Tensor, nh: usize, hd: usize) -> Result<Tensor> {
    let s = x.shape().dims()[2];
    x.permute(&[0, 2, 1, 3])?.reshape(&[1, s, nh * hd])
}

/// Text cross-attention (`WanT2VCrossAttention`): Q from x, K/V from text
/// context, QK-norm on q/k, NO rope, unmasked at B=1. Returns `o(attn)`.
///
/// `prefix` = e.g. `"single_blocks.3.cross_attn"` or
/// `"double_blocks.2.cross_attn"` with a `q/k/v/o` (+ `..._audio`) suffix.
#[allow(clippy::too_many_arguments)]
fn cross_attn(
    x_norm: &Tensor, // [1, Sx, dim]  (already norm3'd)
    context: &Tensor, // [1, Sc, dim]
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str, // "" for vid stream, "_audio" for the audio stream
    cfg: &NavaAVConfig,
) -> Result<Tensor> {
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let qn = format!("{prefix}.q{suffix}");
    let kn = format!("{prefix}.k{suffix}");
    let vn = format!("{prefix}.v{suffix}");
    let on = format!("{prefix}.o{suffix}");
    let nqn = format!("{prefix}.norm_q{suffix}.weight");
    let nkn = format!("{prefix}.norm_k{suffix}.weight");

    let q = linear_bias(x_norm, w(weights, &format!("{qn}.weight"))?, w(weights, &format!("{qn}.bias"))?)?;
    let k = linear_bias(context, w(weights, &format!("{kn}.weight"))?, w(weights, &format!("{kn}.bias"))?)?;
    let v = linear_bias(context, w(weights, &format!("{vn}.weight"))?, w(weights, &format!("{vn}.bias"))?)?;

    let q = rms_norm(&q, w(weights, &nqn)?, cfg.eps)?;
    let k = rms_norm(&k, w(weights, &nkn)?, cfg.eps)?;

    let q = to_sdpa_layout(&q, nh, hd)?;
    let k = to_sdpa_layout(&k, nh, hd)?;
    let v = to_sdpa_layout(&v, nh, hd)?;
    let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn = from_sdpa_layout(&attn, nh, hd)?;
    linear_bias(&attn, w(weights, &format!("{on}.weight"))?, w(weights, &format!("{on}.bias"))?)
}

/// Compute Q/K/V for a self-attn stream: linear → QK-norm → `[1,S,dim]`.
/// Returns `(q, k, v)` each still `[1, S, dim]` (head-reshape happens at SDPA).
fn self_attn_qkv(
    x_norm: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str, // e.g. "single_blocks.3.self_attn"
    suffix: &str, // "" or "_audio"
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let qn = format!("{prefix}.q{suffix}");
    let kn = format!("{prefix}.k{suffix}");
    let vn = format!("{prefix}.v{suffix}");
    let nqn = format!("{prefix}.norm_q{suffix}.weight");
    let nkn = format!("{prefix}.norm_k{suffix}.weight");

    let q = linear_bias(x_norm, w(weights, &format!("{qn}.weight"))?, w(weights, &format!("{qn}.bias"))?)?;
    let k = linear_bias(x_norm, w(weights, &format!("{kn}.weight"))?, w(weights, &format!("{kn}.bias"))?)?;
    let v = linear_bias(x_norm, w(weights, &format!("{vn}.weight"))?, w(weights, &format!("{vn}.bias"))?)?;
    let q = rms_norm(&q, w(weights, &nqn)?, eps)?;
    let k = rms_norm(&k, w(weights, &nkn)?, eps)?;
    Ok((q, k, v))
}

// ===========================================================================
// Per-stream RoPE context (cos/sin tables for one forward, B=1)
// ===========================================================================

/// Precomputed per-forward RoPE cos/sin for a video stream + audio stream.
/// Built once in `WanAVModel::forward` and shared by every block.
pub struct RopeCtx {
    /// Video `[1,1,seq_vid,head_dim/2]` BF16 cos/sin (None if no video).
    pub vid: Option<(Tensor, Tensor)>,
    /// Audio `[1,1,seq_aud,AUDIO_ROPE_DIM/2]` BF16 cos/sin (None if no audio).
    pub aud: Option<(Tensor, Tensor)>,
}

/// Apply RoPE to a `[1,S,dim]` Q (or K), splitting into the vid/audio segments
/// at `seq_vid`. Returns `[1,S,dim]` with each segment rotated by its own
/// convention (video 3D, audio 1D-partial). Used for the joint self-attn cat
/// where Q/K are `[1, seq_vid + seq_aud, dim]`.
fn apply_joint_rope(
    qk: &Tensor, // [1, S, dim]
    seq_vid: usize,
    seq_aud: usize,
    nh: usize,
    hd: usize,
    rope: &RopeCtx,
) -> Result<Tensor> {
    // Reshape to [1, H, S, D] so the fused rope kernel can rotate per-head.
    let qk_h = to_sdpa_layout(qk, nh, hd)?; // [1, H, S, D]
    let mut segs: Vec<Tensor> = Vec::new();
    if seq_vid > 0 {
        let v = qk_h.narrow(2, 0, seq_vid)?.contiguous()?;
        let (c, s) = rope.vid.as_ref().ok_or_else(|| {
            Error::InvalidInput("apply_joint_rope: video rope missing".into())
        })?;
        segs.push(apply_video_rope_3d(&v, c, s)?);
    }
    if seq_aud > 0 {
        let a = qk_h.narrow(2, seq_vid, seq_aud)?.contiguous()?;
        let (c, s) = rope.aud.as_ref().ok_or_else(|| {
            Error::InvalidInput("apply_joint_rope: audio rope missing".into())
        })?;
        segs.push(apply_audio_rope_1d(&a, c, s)?);
    }
    let rotated = if segs.len() == 1 {
        segs.pop().unwrap()
    } else {
        Tensor::cat(&segs.iter().collect::<Vec<_>>(), 2)? // cat along seq
    };
    // back to [1, S, dim]
    from_sdpa_layout(&rotated, nh, hd)
}

// ===========================================================================
// NavaSingleBlock (fusion ×20)
// ===========================================================================

/// One fusion block (`WanAttentionBlock`). Shared `self_attn.{q,k,v,o}`,
/// `norm_q/norm_k`, `norm1/2/3`, `ffn`, single `modulation` — no `_audio`.
///
/// Forward (`model_mm.py:1042-1098`, `split_av_qk_norm_modulation=False`):
///   - `e = (modulation + cat([e0_vid, e0_aud])).chunk(6)` (BF16).
///   - `y = self_attn(norm1(x)*(1+e[1]) + e[0], use_joint_attention=(not masking_modality))`.
///   - `x = x + y * e[2]`.
///   - `x = x + cross_attn(norm3(x), text_ctx)`.
///   - `y = ffn(norm2(x)*(1+e[4]) + e[3])`.
///   - `x = x + y * e[5]`.
/// All gate/modulation BF16.
///
/// `masking_modality=true` (the align pass) ⇒ `use_joint_attention=false` ⇒
/// the shared `self_attn` splits the cat Q/K/V at `seq_vid` and self-attends
/// video and audio SEPARATELY (no cross-modal coupling), then re-cats — exactly
/// the reference `WanSelfAttention.forward` else-branch (`model_mm.py:545-560`).
/// Q/K/V projection and the `o` projection stay SHARED across both segments
/// (single `self_attn`, unlike the double block's split projections).
#[allow(clippy::too_many_arguments)]
pub fn single_block_forward(
    x: &Tensor, // [1, seq_vid + seq_aud, dim] BF16
    e0: &Tensor, // [1, seq, 6, dim] BF16 (cat of vid|audio e0)
    context: &Tensor, // [1, text_len, dim] BF16 text context
    seq_vid: usize,
    seq_aud: usize,
    rope: &RopeCtx,
    masking_modality: bool,
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    cfg: &NavaAVConfig,
) -> Result<Tensor> {
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let prefix = format!("single_blocks.{block_idx}");

    // Modulation (single shared `modulation` over the whole cat seq).
    let e = block_modulation(w(weights, &format!("{prefix}.modulation.modulation"))?, e0)?;

    // ── Self-attention ──
    // Shared q/k/v over the full cat seq; rope each segment in its own
    // convention. Joint (masking off) = one SDPA over the cat; separate
    // (masking on) = per-modality SDPA, no cross-modal coupling.
    let x_n1 = layer_norm_no_affine(x, cfg.eps)?;
    let sa_in = modulate(&x_n1, &e[1], &e[0])?;
    let (q, k, v) = self_attn_qkv(&sa_in, weights, &format!("{prefix}.self_attn"), "", cfg.eps)?;
    let q = apply_joint_rope(&q, seq_vid, seq_aud, nh, hd, rope)?;
    let k = apply_joint_rope(&k, seq_vid, seq_aud, nh, hd, rope)?;
    let q = to_sdpa_layout(&q, nh, hd)?; // [1,H,Sv+Sa,D]
    let k = to_sdpa_layout(&k, nh, hd)?;
    let v = to_sdpa_layout(&v, nh, hd)?;
    let attn = if masking_modality {
        // Separate self-attention per stream (no cross-modal coupling).
        let qv = q.narrow(2, 0, seq_vid)?.contiguous()?;
        let kv = k.narrow(2, 0, seq_vid)?.contiguous()?;
        let vv = v.narrow(2, 0, seq_vid)?.contiguous()?;
        let qa = q.narrow(2, seq_vid, seq_aud)?.contiguous()?;
        let ka = k.narrow(2, seq_vid, seq_aud)?.contiguous()?;
        let va = v.narrow(2, seq_vid, seq_aud)?.contiguous()?;
        let av = flame_core::attention::sdpa(&qv, &kv, &vv, None)?;
        let aa = flame_core::attention::sdpa(&qa, &ka, &va, None)?;
        Tensor::cat(&[&av, &aa], 2)? // [1,H,Sv+Sa,D]
    } else {
        flame_core::attention::sdpa(&q, &k, &v, None)?
    };
    let attn = from_sdpa_layout(&attn, nh, hd)?;
    let sa_out = linear_bias(
        &attn,
        w(weights, &format!("{prefix}.self_attn.o.weight"))?,
        w(weights, &format!("{prefix}.self_attn.o.bias"))?,
    )?;
    let x = gated_residual(x, &sa_out, &e[2])?;

    // ── Cross-attention (text) ──
    let x_n3 = layer_norm_affine(
        &x,
        w(weights, &format!("{prefix}.norm3.weight"))?,
        w(weights, &format!("{prefix}.norm3.bias"))?,
        cfg.eps,
    )?;
    let ca_out = cross_attn(&x_n3, context, weights, &format!("{prefix}.cross_attn"), "", cfg)?;
    let x = x.add(&ca_out)?;

    // ── FFN ──
    let x_n2 = layer_norm_no_affine(&x, cfg.eps)?;
    let ffn_in = modulate(&x_n2, &e[4], &e[3])?;
    let y = ffn(
        &ffn_in,
        w(weights, &format!("{prefix}.ffn.0.weight"))?,
        w(weights, &format!("{prefix}.ffn.0.bias"))?,
        w(weights, &format!("{prefix}.ffn.2.weight"))?,
        w(weights, &format!("{prefix}.ffn.2.bias"))?,
    )?;
    gated_residual(&x, &y, &e[5])
}

// ===========================================================================
// NavaDoubleBlock (alignment ×10)
// ===========================================================================

/// One alignment block (`WanDoubleStreamAttentionBlock`, `no_split_norm_ffn=true`).
///
/// Separate vid/audio self-attn QKV (`self_attn.{q,k,v,o}` + `..._audio`) and
/// separate `modulation` / `modulation_audio`; SHARED `norm1/2/3` + `ffn`.
/// Dual-stream text cross-attn (`cross_attn.{q,k,v,o}` + `..._audio`).
///
/// Forward (`model_mm.py:890-953`):
///   - vid: `e_vid = modulation(e0_vid).chunk(6)`; aud: `e_audio = modulation_audio(e0_aud).chunk(6)`.
///   - `x_vid_norm = norm1(x_vid)*(1+e_vid[1]) + e_vid[0]` (shared norm1 for audio too).
///   - self_attn: compute vid Q/K/V and audio Q/K/V separately, rope each, cat
///     along seq, ONE joint SDPA (when `use_joint_attention`), split, separate `o`/`o_audio`.
///   - `x_vid = x_vid + y_vid_attn * e_vid[2]`, likewise audio.
///   - cross_attn_ffn_doublestream: per-stream text-CA (shared norm3) then shared FFN
///     gated by e_vid[5] / e_audio[5].
///   - `x = cat([x_vid, x_audio], dim=1)`.
///
/// `masking_modality=true` ⇒ `use_joint_attention=false` ⇒ vid and audio
/// self-attend SEPARATELY (no cross-modal coupling).
#[allow(clippy::too_many_arguments)]
pub fn double_block_forward(
    x: &Tensor, // [1, seq_vid + seq_aud, dim] BF16
    e0_vid: &Tensor, // [1, seq_vid, 6, dim] BF16
    e0_aud: &Tensor, // [1, seq_aud, 6, dim] BF16
    context: &Tensor, // [1, text_len, dim] BF16
    seq_vid: usize,
    seq_aud: usize,
    rope: &RopeCtx,
    masking_modality: bool,
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    cfg: &NavaAVConfig,
) -> Result<Tensor> {
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let prefix = format!("double_blocks.{block_idx}");

    // x split into vid / audio segments.
    let x_vid = x.narrow(1, 0, seq_vid)?.contiguous()?;
    let x_aud = x.narrow(1, seq_vid, seq_aud)?.contiguous()?;

    // Separate modulation per stream.
    let e_vid = block_modulation(w(weights, &format!("{prefix}.modulation.modulation"))?, e0_vid)?;
    let e_aud = block_modulation(
        w(weights, &format!("{prefix}.modulation_audio.modulation"))?,
        e0_aud,
    )?;

    // ── Self-attention ──
    // norm1 is SHARED (no_split_norm_ffn=true). Modulate each stream by its own e.
    let xv_n1 = layer_norm_no_affine(&x_vid, cfg.eps)?;
    let xa_n1 = layer_norm_no_affine(&x_aud, cfg.eps)?;
    let sa_in_vid = modulate(&xv_n1, &e_vid[1], &e_vid[0])?;
    let sa_in_aud = modulate(&xa_n1, &e_aud[1], &e_aud[0])?;

    let (qv, kv, vv) = self_attn_qkv(&sa_in_vid, weights, &format!("{prefix}.self_attn"), "", cfg.eps)?;
    let (qa, ka, va) =
        self_attn_qkv(&sa_in_aud, weights, &format!("{prefix}.self_attn"), "_audio", cfg.eps)?;

    // RoPE each stream in its own convention.
    let (vc, vs) = rope.vid.as_ref().ok_or_else(|| {
        Error::InvalidInput("double_block: video rope missing".into())
    })?;
    let (ac, as_) = rope.aud.as_ref().ok_or_else(|| {
        Error::InvalidInput("double_block: audio rope missing".into())
    })?;
    let qv_h = apply_video_rope_3d(&to_sdpa_layout(&qv, nh, hd)?, vc, vs)?; // [1,H,Sv,D]
    let kv_h = apply_video_rope_3d(&to_sdpa_layout(&kv, nh, hd)?, vc, vs)?;
    let vv_h = to_sdpa_layout(&vv, nh, hd)?;
    let qa_h = apply_audio_rope_1d(&to_sdpa_layout(&qa, nh, hd)?, ac, as_)?; // [1,H,Sa,D]
    let ka_h = apply_audio_rope_1d(&to_sdpa_layout(&ka, nh, hd)?, ac, as_)?;
    let va_h = to_sdpa_layout(&va, nh, hd)?;

    let (y_vid_attn, y_aud_attn) = if masking_modality {
        // Separate self-attention per stream (no cross-modal coupling).
        let av = flame_core::attention::sdpa(&qv_h, &kv_h, &vv_h, None)?;
        let aa = flame_core::attention::sdpa(&qa_h, &ka_h, &va_h, None)?;
        (from_sdpa_layout(&av, nh, hd)?, from_sdpa_layout(&aa, nh, hd)?)
    } else {
        // Joint: cat along seq (dim 2 = S in [1,H,S,D]), one SDPA, split.
        let q = Tensor::cat(&[&qv_h, &qa_h], 2)?;
        let k = Tensor::cat(&[&kv_h, &ka_h], 2)?;
        let v = Tensor::cat(&[&vv_h, &va_h], 2)?;
        let attn = flame_core::attention::sdpa(&q, &k, &v, None)?; // [1,H,Sv+Sa,D]
        let av = attn.narrow(2, 0, seq_vid)?.contiguous()?;
        let aa = attn.narrow(2, seq_vid, seq_aud)?.contiguous()?;
        (from_sdpa_layout(&av, nh, hd)?, from_sdpa_layout(&aa, nh, hd)?)
    };

    // Separate output projection.
    let yv = linear_bias(
        &y_vid_attn,
        w(weights, &format!("{prefix}.self_attn.o.weight"))?,
        w(weights, &format!("{prefix}.self_attn.o.bias"))?,
    )?;
    let ya = linear_bias(
        &y_aud_attn,
        w(weights, &format!("{prefix}.self_attn.o_audio.weight"))?,
        w(weights, &format!("{prefix}.self_attn.o_audio.bias"))?,
    )?;
    let x_vid = gated_residual(&x_vid, &yv, &e_vid[2])?;
    let x_aud = gated_residual(&x_aud, &ya, &e_aud[2])?;

    // ── Cross-attention + FFN (per stream, shared norm3/norm2/ffn) ──
    // Video stream.
    let xv_n3 = layer_norm_affine(
        &x_vid,
        w(weights, &format!("{prefix}.norm3.weight"))?,
        w(weights, &format!("{prefix}.norm3.bias"))?,
        cfg.eps,
    )?;
    let ca_vid = cross_attn(&xv_n3, context, weights, &format!("{prefix}.cross_attn"), "", cfg)?;
    let x_vid = x_vid.add(&ca_vid)?;
    let xv_n2 = layer_norm_no_affine(&x_vid, cfg.eps)?;
    let ffn_in_vid = modulate(&xv_n2, &e_vid[4], &e_vid[3])?;
    let yv_ffn = ffn(
        &ffn_in_vid,
        w(weights, &format!("{prefix}.ffn.0.weight"))?,
        w(weights, &format!("{prefix}.ffn.0.bias"))?,
        w(weights, &format!("{prefix}.ffn.2.weight"))?,
        w(weights, &format!("{prefix}.ffn.2.bias"))?,
    )?;
    let x_vid = gated_residual(&x_vid, &yv_ffn, &e_vid[5])?;

    // Audio stream.
    let xa_n3 = layer_norm_affine(
        &x_aud,
        w(weights, &format!("{prefix}.norm3.weight"))?,
        w(weights, &format!("{prefix}.norm3.bias"))?,
        cfg.eps,
    )?;
    let ca_aud = cross_attn(&xa_n3, context, weights, &format!("{prefix}.cross_attn"), "_audio", cfg)?;
    let x_aud = x_aud.add(&ca_aud)?;
    let xa_n2 = layer_norm_no_affine(&x_aud, cfg.eps)?;
    let ffn_in_aud = modulate(&xa_n2, &e_aud[4], &e_aud[3])?;
    let ya_ffn = ffn(
        &ffn_in_aud,
        w(weights, &format!("{prefix}.ffn.0.weight"))?,
        w(weights, &format!("{prefix}.ffn.0.bias"))?,
        w(weights, &format!("{prefix}.ffn.2.weight"))?,
        w(weights, &format!("{prefix}.ffn.2.bias"))?,
    )?;
    let x_aud = gated_residual(&x_aud, &ya_ffn, &e_aud[5])?;

    // Recombine cat([x_vid, x_audio], dim=1).
    Tensor::cat(&[&x_vid, &x_aud], 1)
}

// ===========================================================================
// Heads + unpatchify
// ===========================================================================

/// `Head.forward` (model_mm.py:1118-1128): 2-param modulation `[1,2,dim]` + Linear.
///
///   `e = (modulation.bf16().unsqueeze(0) + e_te.unsqueeze(2)).chunk(2, dim=2)`
///   `x = head(norm(x) * (1 + e[1]) + e[0])`
///
/// `e_te`: the time-embedding `e` for THIS stream, `[1, seq, dim]` BF16 (NOT e0).
/// `mod_param`: `head.modulation` (or `head_audio.modulation`) `[1, 2, dim]` BF16.
/// Returns `[1, seq, prod(patch)*out_dim]` BF16.
pub fn head_forward(
    x: &Tensor, // [1, seq, dim] BF16
    e_te: &Tensor, // [1, seq, dim] BF16
    mod_param: &Tensor, // [1, 2, dim] BF16
    head_w: &Tensor,
    head_b: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = e_te.shape().dims();
    let (b, seq, dim) = (dims[0], dims[1], dims[2]);
    // modulation [1,2,dim] -> [1,1,2,dim]; e_te [1,seq,dim] -> [1,seq,1,dim]
    let mp = mod_param.reshape(&[1, 1, 2, dim])?;
    let e_un = e_te.reshape(&[b, seq, 1, dim])?;
    let e_mod = e_un.add(&mp)?; // broadcast -> [1, seq, 2, dim]
    let shift = e_mod.narrow(2, 0, 1)?.reshape(&[b, seq, dim])?.contiguous()?;
    let scale = e_mod.narrow(2, 1, 1)?.reshape(&[b, seq, dim])?.contiguous()?;

    let x_norm = layer_norm_no_affine(x, eps)?;
    let modulated = modulate(&x_norm, &scale, &shift)?;
    linear_bias(&modulated, head_w, head_b)
}

/// Unpatchify video (`unpatchify`, model_mm.py:1693-1719).
/// `x`: `[1, seq, prod(patch)*c]` BF16 (seq = f*h*w grid tokens).
/// Returns `[c, F, H, W]` BF16 via einsum `fhwpqrc->cfphqwr`.
pub fn unpatchify_video(
    x: &Tensor,
    grid: (usize, usize, usize),
    patch: [usize; 3],
    c: usize,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let (fo, ho, wo) = grid;
    let (pf, ph, pw) = (patch[0], patch[1], patch[2]);
    let patch_dim = c * pf * ph * pw;
    let seq = fo * ho * wo;

    let x_data = x
        .reshape(&[seq, patch_dim])?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let out_f = fo * pf;
    let out_h = ho * ph;
    let out_w = wo * pw;
    let mut out = vec![0.0f32; c * out_f * out_h * out_w];
    // u.view(f,h,w,pf,ph,pw,c) ; einsum fhwpqrc->cfphqwr
    for fi in 0..fo {
        for hi in 0..ho {
            for wi in 0..wo {
                let patch_idx = fi * ho * wo + hi * wo + wi;
                for pfi in 0..pf {
                    for phi in 0..ph {
                        for pwi in 0..pw {
                            for ci in 0..c {
                                // src layout: [..pf, ph, pw, c] within the patch
                                let src_ch = pfi * ph * pw * c + phi * pw * c + pwi * c + ci;
                                let val = x_data[patch_idx * patch_dim + src_ch];
                                let dst_f = fi * pf + pfi;
                                let dst_h = hi * ph + phi;
                                let dst_w = wi * pw + pwi;
                                let dst_idx = ci * out_f * out_h * out_w
                                    + dst_f * out_h * out_w
                                    + dst_h * out_w
                                    + dst_w;
                                out[dst_idx] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, Shape::from_dims(&[c, out_f, out_h, out_w]), device)?.to_dtype(DType::BF16)
}

/// Unpatchify audio (trivial: patch_size=[1], so the head output IS the latent).
/// `x`: `[1, seq, audio_out_dim]` → `[seq, audio_out_dim]` (slice to grid length).
pub fn unpatchify_audio(x: &Tensor, seq_aud: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    let out_dim = dims[2];
    x.reshape(&[dims[1], out_dim])?.narrow(0, 0, seq_aud)
}

// ===========================================================================
// Top-level WanAVModel forward
// ===========================================================================

/// The NAVA single-tower joint-AV MM-DiT (`WanAVModel`, mmdit path).
///
/// Holds config + RoPE freq tables + the flat weight map (loaded directly into
/// `backbone.*`-stripped keys: `double_blocks.*`, `single_blocks.*`,
/// `patch_embedding*`, `text_embedding*`, `time_embedding*`, `time_projection*`,
/// `head*`). The loader (out of scope this round) populates `weights`.
pub struct WanAVModel {
    pub config: NavaAVConfig,
    pub tables: NavaRopeTables,
    pub weights: HashMap<String, Tensor>,
    pub device: Arc<CudaDevice>,
}

impl WanAVModel {
    pub fn new(
        config: NavaAVConfig,
        weights: HashMap<String, Tensor>,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let tables = NavaRopeTables::new(&config, 1024, device.clone())?;
        Ok(Self { config, tables, weights, device })
    }

    /// Build the per-stream `(e, e0)` time embeddings for B=1, all tokens at the
    /// same timestep `t` (T2AV base scope — `first_frame_is_clean=false`).
    /// `e`: `[1, seq, dim]`, `e0`: `[1, seq, 6, dim]`, both BF16.
    fn stream_time_embed(&self, t: f32, seq: usize) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let timesteps = vec![t; seq];
        super::nava_av::time_embedding(
            cfg,
            &timesteps,
            1,
            seq,
            self.device.clone(),
            w(&self.weights, "time_embedding.0.weight")?,
            w(&self.weights, "time_embedding.0.bias")?,
            w(&self.weights, "time_embedding.2.weight")?,
            w(&self.weights, "time_embedding.2.bias")?,
            w(&self.weights, "time_projection.1.weight")?,
            w(&self.weights, "time_projection.1.bias")?,
        )
    }

    /// Embed a `[L, text_dim]` (or `[1, L, text_dim]`) text context to
    /// `[1, text_len, dim]` BF16, right-padding/truncating to `text_len`.
    fn embed_context(&self, ctx: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = ctx.shape().dims();
        // Normalize to [1, L, text_dim].
        let ctx3 = match dims.len() {
            2 => ctx.reshape(&[1, dims[0], dims[1]])?,
            3 => ctx.clone(),
            _ => return Err(Error::InvalidInput(format!("context must be 2D/3D, got {dims:?}"))),
        };
        let l = ctx3.shape().dims()[1];
        let td = ctx3.shape().dims()[2];
        // Pad/truncate to text_len along L.
        let ctx_padded = if l == cfg.text_len {
            ctx3
        } else if l > cfg.text_len {
            ctx3.narrow(1, 0, cfg.text_len)?.contiguous()?
        } else {
            let pad = bf16_zeros_like(&[1, cfg.text_len - l, td], self.device.clone())?;
            Tensor::cat(&[&ctx3, &pad], 1)?
        };
        super::nava_av::text_embedding(
            &ctx_padded,
            w(&self.weights, "text_embedding.0.weight")?,
            w(&self.weights, "text_embedding.0.bias")?,
            w(&self.weights, "text_embedding.2.weight")?,
            w(&self.weights, "text_embedding.2.bias")?,
        )
    }

    /// Full forward.
    ///
    /// `vid`: video latent `[C=48, F, H, W]` BF16.
    /// `audio`: audio latent `[L, 128]` BF16 (channel-last, as `ChannelLastConv1d` expects).
    /// `t`: scalar diffusion timestep (B=1).
    /// `vid_ctx` / `audio_ctx`: text context `[L_text, 4096]` BF16 per stream.
    /// `skip_layers`: block indices to skip (slg — index space is double then single,
    ///   i.e. 0..num_double_layers are double, the rest single, matching the python
    ///   `enumerate` over `double_blocks` then `single_blocks`).
    /// `masking_modality`: align_3d pass — disables cross-modal joint self-attn in
    ///   BOTH double AND single blocks (the python `use_joint_attention=(not
    ///   masking_modality)` is passed to every block's self-attn; `model_mm.py:913`
    ///   double, `:1083` single).
    ///
    /// Returns `(eps_vid [48,F,H,W], eps_audio [L,128])`, both BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        vid: &Tensor,
        audio: &Tensor,
        t: f32,
        vid_ctx: &Tensor,
        audio_ctx: &Tensor,
        skip_layers: &[usize],
        masking_modality: bool,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;

        // ── Patch embed both streams ──
        let (x_vid, grid) = super::nava_av::video_patch_embed(
            cfg,
            vid,
            w(&self.weights, "patch_embedding.weight")?,
            w(&self.weights, "patch_embedding.bias")?,
        )?; // x_vid [1, seq_vid, dim]
        let seq_vid = x_vid.shape().dims()[1];

        // audio [L, 128] -> [1, L, 128]
        let adims = audio.shape().dims();
        let audio_in = match adims.len() {
            2 => audio.reshape(&[1, adims[0], adims[1]])?,
            3 => audio.clone(),
            _ => return Err(Error::InvalidInput(format!("audio must be 2D/3D, got {adims:?}"))),
        };
        let x_aud = super::nava_av::audio_patch_embed(
            &audio_in,
            w(&self.weights, "patch_embedding_audio.0.weight")?,
            w(&self.weights, "patch_embedding_audio.0.bias")?,
            w(&self.weights, "patch_embedding_audio.2.w1.weight")?,
            w(&self.weights, "patch_embedding_audio.2.w2.weight")?,
            w(&self.weights, "patch_embedding_audio.2.w3.weight")?,
        )?; // [1, seq_aud, dim]
        let seq_aud = x_aud.shape().dims()[1];

        // ── Per-stream time embeddings (separate e0; do NOT upgrade to F32) ──
        let (e_vid, e0_vid) = self.stream_time_embed(t, seq_vid)?;
        let (e_aud, e0_aud) = self.stream_time_embed(t, seq_aud)?;

        // ── Text context. python picks vid context (spk_embed=None base path). ──
        let _ = audio_ctx; // base T2AV: both streams cross-attend the video text context
        let context = self.embed_context(vid_ctx)?;

        // ── RoPE cos/sin built once for this forward ──
        let rope = RopeCtx {
            vid: Some(video_rope_cos_sin(&self.tables, grid, cfg.head_dim, self.device.clone())?),
            aud: Some(audio_rope_cos_sin(&self.tables, seq_aud)?),
        };

        // cat([x_vid, x_audio], dim=1)
        let mut x = Tensor::cat(&[&x_vid, &x_aud], 1)?;

        // ── Double blocks (alignment ×N) ──
        for i in 0..cfg.num_double_layers {
            if skip_layers.contains(&i) {
                continue;
            }
            x = double_block_forward(
                &x, &e0_vid, &e0_aud, &context, seq_vid, seq_aud, &rope, masking_modality,
                &self.weights, i, cfg,
            )?;
        }

        // ── Single blocks (fusion ×N) ── e0 = cat([e0_vid, e0_aud]) along seq.
        let e0_cat = Tensor::cat(&[&e0_vid, &e0_aud], 1)?;
        for i in 0..cfg.num_single_layers {
            // slg index space: double blocks first, then single (matches the
            // python single enumerate continuing the conceptual block index).
            let global_idx = cfg.num_double_layers + i;
            if skip_layers.contains(&global_idx) {
                continue;
            }
            x = single_block_forward(
                &x, &e0_cat, &context, seq_vid, seq_aud, &rope, masking_modality,
                &self.weights, i, cfg,
            )?;
        }

        // ── Split + heads ──
        let xv = x.narrow(1, 0, seq_vid)?.contiguous()?;
        let xa = x.narrow(1, seq_vid, seq_aud)?.contiguous()?;

        let head_vid = head_forward(
            &xv,
            &e_vid,
            w(&self.weights, "head.modulation")?,
            w(&self.weights, "head.head.weight")?,
            w(&self.weights, "head.head.bias")?,
            cfg.eps,
        )?;
        let head_aud = head_forward(
            &xa,
            &e_aud,
            w(&self.weights, "head_audio.modulation")?,
            w(&self.weights, "head_audio.head.weight")?,
            w(&self.weights, "head_audio.head.bias")?,
            cfg.eps,
        )?;

        let eps_vid = unpatchify_video(&head_vid, grid, cfg.patch_size, cfg.vid_out_dim, self.device.clone())?;
        let eps_aud = unpatchify_audio(&head_aud, seq_aud)?;
        Ok((eps_vid, eps_aud))
    }
}

fn bf16_zeros_like(shape: &[usize], device: Arc<CudaDevice>) -> Result<Tensor> {
    Tensor::from_vec(vec![0.0f32; shape.iter().product()], Shape::from_dims(shape), device)?
        .to_dtype(DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::Device;

    fn dev() -> Arc<CudaDevice> {
        Device::cuda(0).unwrap().cuda_device_arc()
    }

    fn bf16_zeros(shape: &[usize], device: Arc<CudaDevice>) -> Tensor {
        Tensor::from_vec(
            vec![0.0f32; shape.iter().product()],
            Shape::from_dims(shape),
            device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
    }

    fn bf16_ones(shape: &[usize], device: Arc<CudaDevice>) -> Tensor {
        Tensor::from_vec(
            vec![1.0f32; shape.iter().product()],
            Shape::from_dims(shape),
            device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
    }

    /// Synthetic weight set for a single/double block (zeros where it keeps the
    /// math finite, ones for the QK/LN scale params). Shapes only — this is a
    /// shape/contract test, not a numeric-parity test.
    fn synth_block_weights(prefix: &str, double: bool, cfg: &NavaAVConfig, d: Arc<CudaDevice>) -> HashMap<String, Tensor> {
        let dim = cfg.dim;
        let ffn_dim = cfg.ffn_dim;
        let mut wts = HashMap::new();
        let lin = |o: usize, i: usize, dd: Arc<CudaDevice>| bf16_zeros(&[o, i], dd);
        let bias = |o: usize, dd: Arc<CudaDevice>| bf16_zeros(&[o], dd);

        // modulation params
        wts.insert(format!("{prefix}.modulation.modulation"), bf16_zeros(&[1, 6, dim], d.clone()));
        if double {
            wts.insert(format!("{prefix}.modulation_audio.modulation"), bf16_zeros(&[1, 6, dim], d.clone()));
        }
        // self-attn (+ _audio for double)
        let mut sa = |suffix: &str, wts: &mut HashMap<String, Tensor>, d: Arc<CudaDevice>| {
            for proj in ["q", "k", "v", "o"] {
                wts.insert(format!("{prefix}.self_attn.{proj}{suffix}.weight"), lin(dim, dim, d.clone()));
                wts.insert(format!("{prefix}.self_attn.{proj}{suffix}.bias"), bias(dim, d.clone()));
            }
            wts.insert(format!("{prefix}.self_attn.norm_q{suffix}.weight"), bf16_ones(&[dim], d.clone()));
            wts.insert(format!("{prefix}.self_attn.norm_k{suffix}.weight"), bf16_ones(&[dim], d.clone()));
        };
        sa("", &mut wts, d.clone());
        if double {
            sa("_audio", &mut wts, d.clone());
        }
        // cross-attn (+ _audio for double)
        let mut ca = |suffix: &str, wts: &mut HashMap<String, Tensor>, d: Arc<CudaDevice>| {
            for proj in ["q", "k", "v", "o"] {
                wts.insert(format!("{prefix}.cross_attn.{proj}{suffix}.weight"), lin(dim, dim, d.clone()));
                wts.insert(format!("{prefix}.cross_attn.{proj}{suffix}.bias"), bias(dim, d.clone()));
            }
            wts.insert(format!("{prefix}.cross_attn.norm_q{suffix}.weight"), bf16_ones(&[dim], d.clone()));
            wts.insert(format!("{prefix}.cross_attn.norm_k{suffix}.weight"), bf16_ones(&[dim], d.clone()));
        };
        ca("", &mut wts, d.clone());
        if double {
            ca("_audio", &mut wts, d.clone());
        }
        // norms
        wts.insert(format!("{prefix}.norm3.weight"), bf16_ones(&[dim], d.clone()));
        wts.insert(format!("{prefix}.norm3.bias"), bf16_zeros(&[dim], d.clone()));
        // ffn
        wts.insert(format!("{prefix}.ffn.0.weight"), lin(ffn_dim, dim, d.clone()));
        wts.insert(format!("{prefix}.ffn.0.bias"), bias(ffn_dim, d.clone()));
        wts.insert(format!("{prefix}.ffn.2.weight"), lin(dim, ffn_dim, d.clone()));
        wts.insert(format!("{prefix}.ffn.2.bias"), bias(dim, d.clone()));
        wts
    }

    fn make_rope(cfg: &NavaAVConfig, grid: (usize, usize, usize), seq_aud: usize, d: Arc<CudaDevice>) -> RopeCtx {
        let tables = NavaRopeTables::new(cfg, 1024, d.clone()).unwrap();
        let vid = video_rope_cos_sin(&tables, grid, cfg.head_dim, d.clone()).unwrap();
        let aud = audio_rope_cos_sin(&tables, seq_aud).unwrap();
        RopeCtx { vid: Some(vid), aud: Some(aud) }
    }

    #[test]
    fn single_block_output_shape() {
        let d = dev();
        let cfg = NavaAVConfig::default();
        // tiny grid: f=1,h=2,w=2 -> seq_vid=4; seq_aud=3.
        let grid = (1usize, 2usize, 2usize);
        let seq_vid = grid.0 * grid.1 * grid.2;
        let seq_aud = 3usize;
        let seq = seq_vid + seq_aud;
        let dim = cfg.dim;

        let x = bf16_zeros(&[1, seq, dim], d.clone());
        let e0 = bf16_zeros(&[1, seq, 6, dim], d.clone());
        let ctx = bf16_zeros(&[1, cfg.text_len, dim], d.clone());
        let rope = make_rope(&cfg, grid, seq_aud, d.clone());
        let wts = synth_block_weights("single_blocks.0", false, &cfg, d.clone());

        // Joint path (masking_modality=false).
        let out = single_block_forward(
            &x, &e0, &ctx, seq_vid, seq_aud, &rope, false, &wts, 0, &cfg,
        )
        .unwrap();
        assert_eq!(out.shape().dims(), &[1, seq, dim]);

        // Separate path (masking_modality=true) — per-modality self-attn, same
        // output contract [1, Sv+Sa, dim].
        let out_m = single_block_forward(
            &x, &e0, &ctx, seq_vid, seq_aud, &rope, true, &wts, 0, &cfg,
        )
        .unwrap();
        assert_eq!(out_m.shape().dims(), &[1, seq, dim]);
    }

    #[test]
    fn double_block_split_point() {
        let d = dev();
        let cfg = NavaAVConfig::default();
        let grid = (1usize, 2usize, 2usize);
        let seq_vid = grid.0 * grid.1 * grid.2; // 4
        let seq_aud = 3usize;
        let seq = seq_vid + seq_aud;
        let dim = cfg.dim;

        let x = bf16_zeros(&[1, seq, dim], d.clone());
        let e0_vid = bf16_zeros(&[1, seq_vid, 6, dim], d.clone());
        let e0_aud = bf16_zeros(&[1, seq_aud, 6, dim], d.clone());
        let ctx = bf16_zeros(&[1, cfg.text_len, dim], d.clone());
        let rope = make_rope(&cfg, grid, seq_aud, d.clone());
        let wts = synth_block_weights("double_blocks.0", true, &cfg, d.clone());

        // Joint path (masking_modality=false).
        let out = double_block_forward(
            &x, &e0_vid, &e0_aud, &ctx, seq_vid, seq_aud, &rope, false, &wts, 0, &cfg,
        )
        .unwrap();
        assert_eq!(out.shape().dims(), &[1, seq, dim], "double block must preserve [1, Sv+Sa, dim]");

        // Separate path (masking_modality=true) — same output contract.
        let out_m = double_block_forward(
            &x, &e0_vid, &e0_aud, &ctx, seq_vid, seq_aud, &rope, true, &wts, 0, &cfg,
        )
        .unwrap();
        assert_eq!(out_m.shape().dims(), &[1, seq, dim]);
    }

    #[test]
    fn unpatchify_video_shape_contract() {
        let d = dev();
        let cfg = NavaAVConfig::default();
        let grid = (2usize, 3usize, 4usize); // f,h,w grid tokens
        let patch = cfg.patch_size; // (1,2,2)
        let c = cfg.vid_out_dim; // 48
        let seq = grid.0 * grid.1 * grid.2;
        let patch_dim = c * patch[0] * patch[1] * patch[2];
        let x = bf16_zeros(&[1, seq, patch_dim], d.clone());
        let out = unpatchify_video(&x, grid, patch, c, d.clone()).unwrap();
        assert_eq!(
            out.shape().dims(),
            &[c, grid.0 * patch[0], grid.1 * patch[1], grid.2 * patch[2]]
        );
    }

    #[test]
    fn unpatchify_audio_shape_contract() {
        let d = dev();
        let cfg = NavaAVConfig::default();
        let seq_full = 8usize;
        let seq_aud = 5usize; // grid length (right-padding removed)
        let x = bf16_zeros(&[1, seq_full, cfg.audio_out_dim], d.clone());
        let out = unpatchify_audio(&x, seq_aud).unwrap();
        assert_eq!(out.shape().dims(), &[seq_aud, cfg.audio_out_dim]);
    }

    #[test]
    fn head_output_shape() {
        let d = dev();
        let cfg = NavaAVConfig::default();
        let seq = 4usize;
        let dim = cfg.dim;
        let c = cfg.vid_out_dim;
        let out_dim = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2] * c;

        let x = bf16_zeros(&[1, seq, dim], d.clone());
        let e_te = bf16_zeros(&[1, seq, dim], d.clone());
        let mod_param = bf16_zeros(&[1, 2, dim], d.clone());
        let head_w = bf16_zeros(&[out_dim, dim], d.clone());
        let head_b = bf16_zeros(&[out_dim], d.clone());

        let out = head_forward(&x, &e_te, &mod_param, &head_w, &head_b, cfg.eps).unwrap();
        assert_eq!(out.shape().dims(), &[1, seq, out_dim]);
    }
}
