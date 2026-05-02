//! daVinci-MagiHuman SR (super-resolution) DiT — pure-Rust port.
//!
//! Architecture overview vs base distill (`magihuman_dit.rs`):
//!   * Same 40-layer stack, same `mm_layers` / `gelu7_layers`, same hidden=5120,
//!     head_dim=128, GQA factor 5, SwiGLU7 / GELU7 activations, attn gating.
//!   * Same adapter, RoPE (ElementWiseFourierEmbed), final video/audio heads.
//!   * **DIFFERENT block-layer weight layout**: SR has NO fused `linear_qkv`.
//!     Every layer stores `linear_q`, `linear_k`, `linear_v`, `linear_g`,
//!     `linear_proj` as separate weights. MM layers further split each of those
//!     into per-modality variants (`linear_q_video`, `linear_q_audio`,
//!     `linear_q_text`, etc.). Same for MLP `up_gate_proj` and `down_proj`.
//!
//! Norm tensor shapes are identical between base and SR:
//!   * Shared layers: `pre_norm.weight` `[hidden]`, `q_norm.weight`/`k_norm.weight`
//!     `[head_dim]`.
//!   * MM layers: `pre_norm.weight` `[hidden * 3]`, `q_norm.weight`/`k_norm.weight`
//!     `[head_dim * 3]`.
//!
//! MLP shapes are identical to base:
//!   * GELU7 layers (0..3 in mm_layers): `up_gate_proj` out=20480 (4×hidden, no
//!     gated doubling), `down_proj` in=20480.
//!   * SwiGLU7 layers (others): `up_gate_proj` out=27304 (2×13652 gated),
//!     `down_proj` in=13652.
//!
//! This file imports helper kernels from `magihuman_dit` (norm, RoPE, sdpa,
//! activations, splice_rows, mm_linear pieces) and adds two SR-specific
//! per-layer forward functions:
//!   * `sr_shared_layer_forward` — 4 split-linear matmuls for Q/K/V/G + 1 for
//!     proj; identical norm/rope/sdpa/gating/MLP path to base shared layer.
//!   * `sr_mm_layer_forward` — per-modality split matmuls (3 per linear, 4
//!     linears, plus per-modality proj/up_gate/down) with identical attention
//!     shape afterwards.
//!
//! Wiring: `MagiHumanSrDiTSwapped` parallels `MagiHumanDiTSwapped` — same
//! BlockOffloader, same adapter/final-head loading, same forward signature.

use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use super::magihuman_dit::{
    self as base,
    MagiAdapter, MlpAct,
    HIDDEN_SIZE, HEAD_DIM, NUM_HEADS_Q, NUM_HEADS_KV,
    VIDEO_IN_CHANNELS, AUDIO_IN_CHANNELS,
    MM_LAYERS, GELU7_LAYERS, NUM_LAYERS,
};

type Weights = HashMap<String, Tensor>;

// Mirror constants from base for readability.
const ROPE_DIM: usize = (HEAD_DIM / 8) * 2 * 3;        // 96
const REPEAT_KV: usize = NUM_HEADS_Q / NUM_HEADS_KV;   // 5

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("MagiHumanSrDiT: missing weight: {key}")))
}

// ===========================================================================
// Helpers — split-linear forms of `mm_linear` for the SR weight layout.
// ===========================================================================

/// Single-modality matmul where `w` is `[out, in]` (or `[in, out]` if the
/// offloader has already pre-transposed). Returns `x @ w.T` in PyTorch terms.
fn linear_split(x: &Tensor, w: &Tensor, pre_transposed: bool) -> Result<Tensor> {
    if pre_transposed {
        // `w` is `[in, out]`; matmul is `x @ w` directly.
        x.contiguous()?.matmul(&w.contiguous()?)
    } else {
        base::matmul_with_w_t(x, w)
    }
}

/// Per-modality matmul where each modality has its own weight tensor (each
/// `[out, in]` or `[in, out]` if pre-transposed). Mirrors the body of
/// `Attention._apply_expert_linear` from Python:
/// ```python
/// for w_i, n_i in zip([w_video, w_audio, w_text], group_sizes):
///     out[start:start+n_i] = w_i(x[start:start+n_i])
/// ```
fn linear_split_per_modality(
    x: &Tensor,
    w_video: &Tensor,
    w_audio: &Tensor,
    w_text: &Tensor,
    group_sizes: &[usize],
    pre_transposed: bool,
) -> Result<Tensor> {
    if group_sizes.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "linear_split_per_modality expects 3 group sizes, got {}",
            group_sizes.len()
        )));
    }
    let weights = [w_video, w_audio, w_text];
    let mut offset = 0;
    let mut pieces: Vec<Tensor> = Vec::with_capacity(3);
    for i in 0..3 {
        let n = group_sizes[i];
        if n == 0 {
            offset += n;
            continue;
        }
        let chunk_x = x.narrow(0, offset, n)?.contiguous()?;
        let chunk_out = linear_split(&chunk_x, weights[i], pre_transposed)?;
        pieces.push(chunk_out);
        offset += n;
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Cast and reshape a [1, L, ROPE_DIM] F32 tensor to [1, 1, L, ROPE_DIM/2] BF16
/// for sin and cos halves. Same logic as in base; duplicated to avoid the cost
/// of repeating it per layer when called via the offloader loop.
fn split_rope(rope_b: &Tensor, l: usize) -> Result<(Tensor, Tensor)> {
    let half = ROPE_DIM / 2;
    let sin_emb = rope_b
        .narrow(2, 0, half)?
        .to_dtype(DType::BF16)?
        .reshape(&[1, 1, l, half])?;
    let cos_emb = rope_b
        .narrow(2, half, half)?
        .to_dtype(DType::BF16)?
        .reshape(&[1, 1, l, half])?;
    Ok((sin_emb, cos_emb))
}

// ===========================================================================
// Per-layer forward — shared (num_modality=1, layers 4..35)
// ===========================================================================
//
// Reads SPLIT linears: linear_q, linear_k, linear_v, linear_g, linear_proj.
// Same downstream path as `magihuman_dit::shared_layer_forward`.

#[allow(clippy::too_many_arguments)]
fn sr_shared_layer_forward(
    hidden_states: &Tensor,
    cos_emb: &Tensor,
    sin_emb: &Tensor,
    weights: &Weights,
    prefix: &str,
    cache: &SrLayerNormCache,
    pre_transposed: bool,
) -> Result<Tensor> {
    let l = hidden_states.shape().dims()[0];

    // SR shared linears — separate weights for each of Q/K/V/G/proj.
    let linear_q = get(weights, &format!("{prefix}attention.linear_q.weight"))?;
    let linear_k = get(weights, &format!("{prefix}attention.linear_k.weight"))?;
    let linear_v = get(weights, &format!("{prefix}attention.linear_v.weight"))?;
    let linear_g = get(weights, &format!("{prefix}attention.linear_g.weight"))?;
    let linear_proj = get(weights, &format!("{prefix}attention.linear_proj.weight"))?;
    let mlp_up_gate = get(weights, &format!("{prefix}mlp.up_gate_proj.weight"))?;
    let mlp_down = get(weights, &format!("{prefix}mlp.down_proj.weight"))?;

    let attn_pre_norm_p1 = cache.attn_pre_norm_p1.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_shared_layer_forward: cache missing attn_pre_norm_p1".into())
    })?;
    let attn_q_norm_p1 = cache.attn_q_norm_p1.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_shared_layer_forward: cache missing attn_q_norm_p1".into())
    })?;
    let attn_k_norm_p1 = cache.attn_k_norm_p1.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_shared_layer_forward: cache missing attn_k_norm_p1".into())
    })?;
    let mlp_pre_norm_p1 = cache.mlp_pre_norm_p1.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_shared_layer_forward: cache missing mlp_pre_norm_p1".into())
    })?;

    let hidden_bf16 = if hidden_states.dtype() == DType::BF16 {
        hidden_states.clone()
    } else {
        hidden_states.to_dtype(DType::BF16)?
    };

    // Pre-norm.
    let h = base::mm_rms_norm_single_fused(&hidden_bf16, attn_pre_norm_p1, 1e-6)?;

    // Q/K/V/G — four separate matmuls.
    let q = linear_split(&h, &linear_q, pre_transposed)?;
    let k = linear_split(&h, &linear_k, pre_transposed)?;
    let v = linear_split(&h, &linear_v, pre_transposed)?;
    let g = linear_split(&h, &linear_g, pre_transposed)?;

    let q = q.reshape(&[l, NUM_HEADS_Q, HEAD_DIM])?;
    let k = k.reshape(&[l, NUM_HEADS_KV, HEAD_DIM])?;
    let v = v.reshape(&[l, NUM_HEADS_KV, HEAD_DIM])?;
    let g = g.reshape(&[l, NUM_HEADS_Q, 1])?;

    let q = base::mm_rms_norm_single_fused(&q, attn_q_norm_p1, 1e-6)?;
    let k = base::mm_rms_norm_single_fused(&k, attn_k_norm_p1, 1e-6)?;

    let q_h_pre = q.unsqueeze(0)?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let k_h_pre = k.unsqueeze(0)?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let v_h     = v.unsqueeze(0)?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let q_h = base::rope_partial_halfsplit(&q_h_pre, cos_emb, sin_emb)?;
    let k_h = base::rope_partial_halfsplit(&k_h_pre, cos_emb, sin_emb)?;

    let k_h = base::repeat_interleave_dim1(&k_h, REPEAT_KV)?;
    let v_h = base::repeat_interleave_dim1(&v_h, REPEAT_KV)?;
    let attn_out = base::sdpa(&q_h, &k_h, &v_h)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?.squeeze(Some(0))?.contiguous()?;

    // Per-head sigmoid gating in F32 (matches base precision).
    let attn_f32 = attn_out.to_dtype(DType::F32)?;
    let gate = g.to_dtype(DType::F32)?.sigmoid()?;
    let attn_gated = attn_f32.mul(&gate)?;
    let attn_flat = attn_gated
        .reshape(&[l, NUM_HEADS_Q * HEAD_DIM])?
        .to_dtype(DType::BF16)?;
    let attn_out_proj = linear_split(&attn_flat, &linear_proj, pre_transposed)?;

    let h_after_attn = hidden_bf16.add(&attn_out_proj)?;

    // ----- MLP (SwiGLU7 — shared layers in SR are not in gelu7_layers) -----
    let h_mlp_in = base::mm_rms_norm_single_fused(&h_after_attn, mlp_pre_norm_p1, 1e-6)?;
    let up = linear_split(&h_mlp_in, &mlp_up_gate, pre_transposed)?
        .to_dtype(DType::F32)?;
    let activated = base::swiglu7(&up)?.to_dtype(DType::BF16)?;
    let mlp_out = linear_split(&activated, &mlp_down, pre_transposed)?;

    h_after_attn.add(&mlp_out)?.to_dtype(DType::F32)
}

// ===========================================================================
// Per-layer forward — MM (num_modality=3, layers 0..3 GELU7 / 36..39 SwiGLU7)
// ===========================================================================
//
// Reads PER-MODALITY split linears: linear_q_video/audio/text, linear_k_*, etc.
// Computes Q/K/V/G via per-modality `linear_split_per_modality`, concatenated
// along token dim. Identical downstream attention path to base mm_layer_forward
// (norm, RoPE, GQA, SDPA, sigmoid gating, proj), and the MLP runs through the
// same per-modality up_gate / down with the same SwiGLU7 / GELU7 activation.

#[allow(clippy::too_many_arguments)]
fn sr_mm_layer_forward(
    hidden_states: &Tensor,
    cos_emb: &Tensor,
    sin_emb: &Tensor,
    group_sizes: &[usize],
    weights: &Weights,
    prefix: &str,
    activation: MlpAct,
    cache: &SrLayerNormCache,
    pre_transposed: bool,
) -> Result<Tensor> {
    let l = hidden_states.shape().dims()[0];

    // Per-modality Q/K/V/G/proj.
    let linear_q_v = get(weights, &format!("{prefix}attention.linear_q_video.weight"))?;
    let linear_q_a = get(weights, &format!("{prefix}attention.linear_q_audio.weight"))?;
    let linear_q_t = get(weights, &format!("{prefix}attention.linear_q_text.weight"))?;
    let linear_k_v = get(weights, &format!("{prefix}attention.linear_k_video.weight"))?;
    let linear_k_a = get(weights, &format!("{prefix}attention.linear_k_audio.weight"))?;
    let linear_k_t = get(weights, &format!("{prefix}attention.linear_k_text.weight"))?;
    let linear_v_v = get(weights, &format!("{prefix}attention.linear_v_video.weight"))?;
    let linear_v_a = get(weights, &format!("{prefix}attention.linear_v_audio.weight"))?;
    let linear_v_t = get(weights, &format!("{prefix}attention.linear_v_text.weight"))?;
    let linear_g_v = get(weights, &format!("{prefix}attention.linear_g_video.weight"))?;
    let linear_g_a = get(weights, &format!("{prefix}attention.linear_g_audio.weight"))?;
    let linear_g_t = get(weights, &format!("{prefix}attention.linear_g_text.weight"))?;
    let linear_proj_v = get(weights, &format!("{prefix}attention.linear_proj_video.weight"))?;
    let linear_proj_a = get(weights, &format!("{prefix}attention.linear_proj_audio.weight"))?;
    let linear_proj_t = get(weights, &format!("{prefix}attention.linear_proj_text.weight"))?;
    let mlp_up_v = get(weights, &format!("{prefix}mlp.up_gate_proj_video.weight"))?;
    let mlp_up_a = get(weights, &format!("{prefix}mlp.up_gate_proj_audio.weight"))?;
    let mlp_up_t = get(weights, &format!("{prefix}mlp.up_gate_proj_text.weight"))?;
    let mlp_down_v = get(weights, &format!("{prefix}mlp.down_proj_video.weight"))?;
    let mlp_down_a = get(weights, &format!("{prefix}mlp.down_proj_audio.weight"))?;
    let mlp_down_t = get(weights, &format!("{prefix}mlp.down_proj_text.weight"))?;

    let attn_pre_norm_p1 = cache.attn_pre_norm_p1_mm.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_mm_layer_forward: cache missing attn_pre_norm_p1_mm".into())
    })?;
    let attn_q_norm_p1 = cache.attn_q_norm_p1_mm.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_mm_layer_forward: cache missing attn_q_norm_p1_mm".into())
    })?;
    let attn_k_norm_p1 = cache.attn_k_norm_p1_mm.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_mm_layer_forward: cache missing attn_k_norm_p1_mm".into())
    })?;
    let mlp_pre_norm_p1 = cache.mlp_pre_norm_p1_mm.as_ref().ok_or_else(|| {
        Error::InvalidOperation("sr_mm_layer_forward: cache missing mlp_pre_norm_p1_mm".into())
    })?;

    let hidden_bf16 = if hidden_states.dtype() == DType::BF16 {
        hidden_states.clone()
    } else {
        hidden_states.to_dtype(DType::BF16)?
    };

    // ----- Attention -----
    let h = base::mm_rms_norm_multi_fused(&hidden_bf16, attn_pre_norm_p1, group_sizes, 1e-6)?;

    // Per-modality Q/K/V/G via `_apply_expert_linear`-style narrowed matmuls.
    let q = linear_split_per_modality(
        &h, &linear_q_v, &linear_q_a, &linear_q_t, group_sizes, pre_transposed
    )?;
    let k = linear_split_per_modality(
        &h, &linear_k_v, &linear_k_a, &linear_k_t, group_sizes, pre_transposed
    )?;
    let v = linear_split_per_modality(
        &h, &linear_v_v, &linear_v_a, &linear_v_t, group_sizes, pre_transposed
    )?;
    let g = linear_split_per_modality(
        &h, &linear_g_v, &linear_g_a, &linear_g_t, group_sizes, pre_transposed
    )?;

    let q = q.reshape(&[l, NUM_HEADS_Q, HEAD_DIM])?;
    let k = k.reshape(&[l, NUM_HEADS_KV, HEAD_DIM])?;
    let v = v.reshape(&[l, NUM_HEADS_KV, HEAD_DIM])?;
    let g = g.reshape(&[l, NUM_HEADS_Q, 1])?;

    let q = base::mm_rms_norm_multi_fused(&q, attn_q_norm_p1, group_sizes, 1e-6)?;
    let k = base::mm_rms_norm_multi_fused(&k, attn_k_norm_p1, group_sizes, 1e-6)?;

    let q_h_pre = q.unsqueeze(0)?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let k_h_pre = k.unsqueeze(0)?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let v_h     = v.unsqueeze(0)?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let q_h = base::rope_partial_halfsplit(&q_h_pre, cos_emb, sin_emb)?;
    let k_h = base::rope_partial_halfsplit(&k_h_pre, cos_emb, sin_emb)?;

    let k_h = base::repeat_interleave_dim1(&k_h, REPEAT_KV)?;
    let v_h = base::repeat_interleave_dim1(&v_h, REPEAT_KV)?;
    let attn_out = base::sdpa(&q_h, &k_h, &v_h)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?.squeeze(Some(0))?.contiguous()?;

    let attn_f32 = attn_out.to_dtype(DType::F32)?;
    let gate = g.to_dtype(DType::F32)?.sigmoid()?;
    let attn_gated = attn_f32.mul(&gate)?;
    let attn_flat = attn_gated
        .reshape(&[l, NUM_HEADS_Q * HEAD_DIM])?
        .to_dtype(DType::BF16)?;
    let attn_out_proj = linear_split_per_modality(
        &attn_flat, &linear_proj_v, &linear_proj_a, &linear_proj_t,
        group_sizes, pre_transposed,
    )?;

    let h_after_attn = hidden_bf16.add(&attn_out_proj)?;

    // ----- MLP -----
    let h_mlp_in = base::mm_rms_norm_multi_fused(&h_after_attn, mlp_pre_norm_p1, group_sizes, 1e-6)?;
    let up = linear_split_per_modality(
        &h_mlp_in, &mlp_up_v, &mlp_up_a, &mlp_up_t, group_sizes, pre_transposed,
    )?
    .to_dtype(DType::F32)?;
    let activated = match activation {
        MlpAct::SwiGLU7 => base::swiglu7(&up)?,
        MlpAct::GELU7 => base::gelu7(&up)?,
    }
    .to_dtype(DType::BF16)?;
    let mlp_out = linear_split_per_modality(
        &activated, &mlp_down_v, &mlp_down_a, &mlp_down_t, group_sizes, pre_transposed,
    )?;

    h_after_attn.add(&mlp_out)?.to_dtype(DType::F32)
}

// ===========================================================================
// Per-layer norm cache (mirrors `LayerNormCache` in base).
// ===========================================================================

struct SrLayerNormCache {
    // MM-only: 3 per-modality (weight + 1) BF16 chunks.
    attn_pre_norm_p1_mm: Option<[Tensor; 3]>,
    attn_q_norm_p1_mm: Option<[Tensor; 3]>,
    attn_k_norm_p1_mm: Option<[Tensor; 3]>,
    mlp_pre_norm_p1_mm: Option<[Tensor; 3]>,
    // Shared-only: single (weight + 1) BF16 tensor.
    attn_pre_norm_p1: Option<Tensor>,
    attn_q_norm_p1: Option<Tensor>,
    attn_k_norm_p1: Option<Tensor>,
    mlp_pre_norm_p1: Option<Tensor>,
}

// ===========================================================================
// Top-level SR DiT — BlockOffloader-backed for 24 GB GPUs.
// ===========================================================================

struct MagiHumanSrFacilitator;
impl flame_diffusion::block_offload::BlockFacilitator for MagiHumanSrFacilitator {
    fn block_count(&self) -> usize { NUM_LAYERS }
    fn classify_key(&self, name: &str) -> Option<usize> {
        name.strip_prefix("block.layers.")?.split('.').next()?.parse().ok()
    }
}

pub struct MagiHumanSrDiTSwapped {
    pub adapter: MagiAdapter,
    final_norm_video: Tensor,
    final_norm_audio: Tensor,
    final_linear_video: Tensor,
    final_linear_audio: Tensor,
    offloader: flame_diffusion::BlockOffloader,
    norm_cache: Vec<SrLayerNormCache>,
}

impl MagiHumanSrDiTSwapped {
    /// Load from a single dequantized BF16 safetensors with the SR1080 weight
    /// layout (per-modality split linears). Block weights go to pinned CPU;
    /// `adapter.*` and `final_*` stay on GPU.
    pub fn load(path: &str, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let facilitator = MagiHumanSrFacilitator;
        let offloader = flame_diffusion::BlockOffloader::load(
            &[path], &facilitator, device.clone()
        ).map_err(|e| Error::InvalidOperation(format!("BlockOffloader: {e}")))?;

        // Load adapter + final heads + per-layer norm weights eagerly.
        let shared_prefixes = ["adapter.", "final_"];
        let norm_suffixes = [
            ".attention.pre_norm.weight",
            ".attention.q_norm.weight",
            ".attention.k_norm.weight",
            ".mlp.pre_norm.weight",
        ];
        let is_norm_key = |k: &str| {
            k.starts_with("block.layers.")
                && norm_suffixes.iter().any(|sfx| k.ends_with(sfx))
        };
        let partial = flame_core::serialization::load_file_filtered(
            std::path::Path::new(path), device,
            |k| shared_prefixes.iter().any(|p| k.starts_with(p)) || is_norm_key(k),
        )?;

        let adapter = MagiAdapter::load(&partial)?;
        let to_f32 = |t: Tensor| -> Result<Tensor> { t.to_dtype(DType::F32) };
        let final_norm_video = to_f32(get(&partial, "final_norm_video.weight")?)?;
        let final_norm_audio = to_f32(get(&partial, "final_norm_audio.weight")?)?;
        let final_linear_video = to_f32(get(&partial, "final_linear_video.weight")?)?;
        let final_linear_audio = to_f32(get(&partial, "final_linear_audio.weight")?)?;

        // Per-layer norm cache. MM layers store `(weight + 1)` BF16 in 3 chunks,
        // shared layers in a single tensor.
        let mut norm_cache: Vec<SrLayerNormCache> = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let prefix = format!("block.layers.{i}.");
            let to_bf16 = |t: Tensor| -> Result<Tensor> { t.to_dtype(DType::BF16) };
            let attn_pre = to_bf16(get(&partial, &format!("{prefix}attention.pre_norm.weight"))?)?;
            let attn_q = to_bf16(get(&partial, &format!("{prefix}attention.q_norm.weight"))?)?;
            let attn_k = to_bf16(get(&partial, &format!("{prefix}attention.k_norm.weight"))?)?;
            let mlp_pre = to_bf16(get(&partial, &format!("{prefix}mlp.pre_norm.weight"))?)?;
            let cache = if MM_LAYERS.contains(&i) {
                SrLayerNormCache {
                    attn_pre_norm_p1_mm: Some(base::precompute_w_plus_1_bf16_per_modality(&attn_pre, HIDDEN_SIZE)?),
                    attn_q_norm_p1_mm:   Some(base::precompute_w_plus_1_bf16_per_modality(&attn_q, HEAD_DIM)?),
                    attn_k_norm_p1_mm:   Some(base::precompute_w_plus_1_bf16_per_modality(&attn_k, HEAD_DIM)?),
                    mlp_pre_norm_p1_mm:  Some(base::precompute_w_plus_1_bf16_per_modality(&mlp_pre, HIDDEN_SIZE)?),
                    attn_pre_norm_p1: None, attn_q_norm_p1: None, attn_k_norm_p1: None, mlp_pre_norm_p1: None,
                }
            } else {
                SrLayerNormCache {
                    attn_pre_norm_p1: Some(base::precompute_w_plus_1_bf16(&attn_pre)?),
                    attn_q_norm_p1:   Some(base::precompute_w_plus_1_bf16(&attn_q)?),
                    attn_k_norm_p1:   Some(base::precompute_w_plus_1_bf16(&attn_k)?),
                    mlp_pre_norm_p1:  Some(base::precompute_w_plus_1_bf16(&mlp_pre)?),
                    attn_pre_norm_p1_mm: None, attn_q_norm_p1_mm: None, attn_k_norm_p1_mm: None, mlp_pre_norm_p1_mm: None,
                }
            };
            norm_cache.push(cache);
        }

        println!(
            "[MagiHuman SR DiT] loaded: {} blocks ({:.2} GB pinned), shared on GPU",
            offloader.block_count(),
            offloader.pinned_bytes() as f64 / (1u64 << 30) as f64,
        );

        Ok(Self {
            adapter,
            final_norm_video,
            final_norm_audio,
            final_linear_video,
            final_linear_audio,
            offloader,
            norm_cache,
        })
    }

    /// Forward — same contract as `MagiHumanDiTSwapped::forward`.
    pub fn forward(
        &mut self,
        x: &Tensor,
        coords: &Tensor,
        group_sizes: &[usize; 3],
    ) -> Result<Tensor> {
        let v_count = group_sizes[0];
        let a_count = group_sizes[1];
        let t_count = group_sizes[2];
        let l = v_count + a_count + t_count;

        let video_mask: Vec<bool> = (0..l).map(|i| i < v_count).collect();
        let audio_mask: Vec<bool> = (0..l).map(|i| i >= v_count && i < v_count + a_count).collect();
        let text_mask: Vec<bool> = (0..l).map(|i| i >= v_count + a_count).collect();

        // Adapter
        let mut h = self.adapter.embed(x, &video_mask, &audio_mask, &text_mask)?;
        let rope = self.adapter.rope_from_coords(coords)?;
        let rope_b = rope.unsqueeze(0)?;
        h = h.to_dtype(DType::BF16)?;

        let group_sizes_vec = group_sizes.to_vec();
        self.offloader.prefetch_block(0)
            .map_err(|e| Error::InvalidOperation(format!("prefetch 0: {e}")))?;
        let prof = std::env::var("MAGI_PROFILE").ok().as_deref() == Some("1");
        let device = h.device().clone();

        let l = h.shape().dims()[0];
        let (sin_emb, cos_emb) = split_rope(&rope_b, l)?;

        for i in 0..NUM_LAYERS {
            let layer_t0 = std::time::Instant::now();

            let t = std::time::Instant::now();
            let raw = self.offloader.await_block(i)
                .map_err(|e| Error::InvalidOperation(format!("await {i}: {e}")))?;
            let t_await = t.elapsed().as_millis();

            let t = std::time::Instant::now();
            if i + 1 < NUM_LAYERS {
                self.offloader.prefetch_block(i + 1)
                    .map_err(|e| Error::InvalidOperation(format!("prefetch {}: {e}", i + 1)))?;
            }
            let t_prefetch = t.elapsed().as_millis();

            let prefix = format!("block.layers.{i}.");
            let is_mm = MM_LAYERS.contains(&i);
            let t = std::time::Instant::now();
            h = if is_mm {
                let act = if GELU7_LAYERS.contains(&i) { MlpAct::GELU7 } else { MlpAct::SwiGLU7 };
                let cache = &self.norm_cache[i];
                sr_mm_layer_forward(&h, &cos_emb, &sin_emb, &group_sizes_vec, &raw, &prefix, act, cache, true)?
            } else {
                let cache = &self.norm_cache[i];
                sr_shared_layer_forward(&h, &cos_emb, &sin_emb, &raw, &prefix, cache, true)?
            };
            if prof {
                device.synchronize().ok();
            }
            let t_fwd = t.elapsed().as_millis();

            let t = std::time::Instant::now();
            if i % 4 == 3 {
                flame_core::cuda_alloc_pool::clear_pool_cache();
            }
            let t_pool = t.elapsed().as_millis();

            let elapsed = layer_t0.elapsed().as_millis();
            if prof {
                eprintln!(
                    "[sr layer {i}] {elapsed} ms  is_mm={is_mm}  await={t_await}  prefetch={t_prefetch}  fwd={t_fwd}  pool={t_pool}",
                );
            } else {
                eprintln!("[sr layer {i}] {elapsed} ms  is_mm={is_mm}");
            }
        }

        // Final heads — identical to base.
        let h_f32 = h.to_dtype(DType::F32)?;
        let device = h_f32.device().clone();
        let mut out = Tensor::zeros_dtype(
            Shape::from_dims(&[l, VIDEO_IN_CHANNELS.max(AUDIO_IN_CHANNELS)]),
            DType::F32,
            device.clone(),
        )?;
        if v_count > 0 {
            let xv = h_f32.narrow(0, 0, v_count)?;
            let xv = base::mm_rms_norm_single(&xv, &self.final_norm_video, 1e-6)?;
            let proj_v = base::matmul_with_w_t(&xv, &self.final_linear_video)?;
            out = base::splice_rows(&out, &proj_v, 0)?;
        }
        if a_count > 0 {
            let xa = h_f32.narrow(0, v_count, a_count)?;
            let xa = base::mm_rms_norm_single(&xa, &self.final_norm_audio, 1e-6)?;
            let proj_a = base::matmul_with_w_t(&xa, &self.final_linear_audio)?;
            let zeros_pad = Tensor::zeros_dtype(
                Shape::from_dims(&[a_count, VIDEO_IN_CHANNELS - AUDIO_IN_CHANNELS]),
                DType::F32, device.clone(),
            )?;
            let proj_a_padded = Tensor::cat(&[&proj_a, &zeros_pad], 1)?;
            out = base::splice_rows(&out, &proj_a_padded, v_count)?;
        }
        let _ = t_count;
        Ok(out)
    }
}
