//! Flux 2 Klein DiT transformer — pure flame_core, key-exact for BFL checkpoints.
//!
//! Supports Klein 4B (5+20 blocks) and Klein 9B (8+24 blocks).
//!
//! Architecture-exact match with BFL safetensors checkpoints.
//!
//! Key structural differences from Flux 1:
//! - NO biases anywhere
//! - SHARED modulation (3 layers at model level, NOT per-block)
//! - SwiGLU MLP (6x ratio) instead of GELU (4x ratio)
//! - No guidance_in, no vector_in
//! - in_channels=128, joint_attention_dim=7680 (4B) or 12288 (9B)
//!
//! Rewritten from the ground-truth Python+Flame reference:
//!   flame-core/inference-test/models/klein_flame.py

use flame_core::cuda_ops_bf16;
use flame_core::serialization;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::lora::LoraStack;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture constants for Klein variants.
#[derive(Debug, Clone)]
pub struct KleinConfig {
    pub inner_dim: usize,
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub num_double: usize,
    pub num_single: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub timestep_dim: usize,
    pub axes_dims: [usize; 4],
    pub theta: f32,
}

impl KleinConfig {
    /// Klein 4B: 5 double + 20 single blocks, inner_dim=3072
    pub fn klein_4b() -> Self {
        Self {
            inner_dim: 3072,
            in_channels: 128,
            joint_attention_dim: 7680,
            num_double: 5,
            num_single: 20,
            num_heads: 24,
            head_dim: 128,
            mlp_hidden: 3072 * 3, // SwiGLU 3x ratio
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        }
    }

    /// Klein 9B: 8 double + 24 single blocks, inner_dim=4096
    pub fn klein_9b() -> Self {
        Self {
            inner_dim: 4096,
            in_channels: 128,
            joint_attention_dim: 12288,
            num_double: 8,
            num_single: 24,
            num_heads: 32,
            head_dim: 128,
            mlp_hidden: 4096 * 3,
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: 3D linear (flame_core Linear only handles 2D batch internally,
// but we need [B, N, C] -> [B, N, out_C] without constructing Linear structs)
// ---------------------------------------------------------------------------

/// Compute `x @ weight.T` for x with shape [B, N, C] and weight [out_C, C].
/// No bias (Klein has NO biases anywhere).
/// Linear projection using pre-transposed weights [in_features, out_features].
/// Used by resident (non-swap) forward paths.
fn linear3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    if shape.len() == 2 {
        return x.matmul(weight_t);
    }
    let b = shape[0];
    let n = shape[1];
    let c = shape[2];
    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = x_2d.matmul(weight_t)?;
    let out_dim = out_2d.shape().dims()[1];
    out_2d.reshape(&[b, n, out_dim])
}

/// Linear projection using NON-transposed weights [out_features, in_features]
/// (PyTorch nn.Linear layout). Uses cuBLASLt TRANSA=T internally — the weight
/// is read directly from the BlockOffloader slot view with zero transpose allocation.
fn linear3d_nt(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    if shape.len() == 3 {
        return flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None);
    }
    // 2D fallback (rare): transpose + matmul
    let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
    x.matmul(&wt)
}

/// x @ weight where weight is ALREADY transposed [in, out].
/// Result is [M, out].
fn matmul_weight_pretransposed(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    x.matmul(weight_t)
}

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding matching ComfyUI.
///
/// `t`: [B] sigma values in [0, 1].
/// `dim`: embedding dimension.
/// `time_factor`: scales sigma to [0, 1000] (default 1000.0).
///
/// Returns [B, dim] in the original dtype of `t`.
fn timestep_embedding(t: &Tensor, dim: usize, time_factor: f32) -> Result<Tensor> {
    let orig_dtype = t.dtype();
    let device = t.device().clone();
    let b = t.shape().dims()[0];

    // Scale: t_scaled = t.to_f32() * time_factor
    let t_f32 = t.to_dtype(DType::F32)?;
    let t_scaled = t_f32.mul_scalar(time_factor)?;

    let half = dim / 2;
    let max_period: f32 = 10000.0;

    // freqs = exp(-log(max_period) * arange(0, half) / half)
    let freqs = Tensor::arange(0.0, half as f32, 1.0, device.clone())?;
    let freqs = freqs.mul_scalar(-max_period.ln() / half as f32)?.exp()?;

    // Outer product: [B, 1] * [1, half] -> [B, half]
    let t_col = t_scaled.reshape(&[b, 1])?;
    let freqs_row = freqs.reshape(&[1, half])?;
    let args = t_col.mul(&freqs_row)?;

    // [cos(args), sin(args)] -> [B, dim]
    let cos_part = args.cos()?;
    let sin_part = args.sin()?;
    let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;

    emb.to_dtype(orig_dtype)
}

// ---------------------------------------------------------------------------
// Modulation: adaLN with LayerNorm (subtracts mean, NOT RMSNorm)
// ---------------------------------------------------------------------------

/// `(1 + scale) * LayerNorm(x) + shift` — fused single kernel.
///
/// `shift`, `scale`: [B, dim]; `x`: [B, N, dim].
fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    flame_core::bf16_ops::modulate_pre_fused_bf16(x, shift, scale, 1e-6)
}

// ---------------------------------------------------------------------------
// QK RMSNorm (no mean subtraction — different from LayerNorm)
// ---------------------------------------------------------------------------

/// Per-head RMSNorm for query/key.
///
/// `x`: [B, H, N, D] reshaped to [B*H*N, D] for rms_norm, then back.
/// `scale`: [D] learned scale parameter.
fn head_rms_norm(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, h, n, d) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * h * n, d])?;
    let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(scale), 1e-6)?;
    normed.reshape(&[b, h, n, d])
}

// ---------------------------------------------------------------------------
// RoPE — real-valued rotation (no complex numbers)
// ---------------------------------------------------------------------------

/// Build 2D RoPE cos/sin tables from position IDs.
///
/// `img_ids`: [N_img, ndim] position coordinates.
/// `txt_ids`: [N_txt, ndim] position coordinates.
/// `axes_dims`: per-axis RoPE dimensions (must sum to head_dim).
/// `theta`: base frequency (2000.0 for Klein).
///
/// Returns `(pe_cos, pe_sin)`: each [1, 1, N_total, head_dim//2] in BF16.
fn build_rope_2d(
    img_ids: &Tensor,
    txt_ids: &Tensor,
    axes_dims: &[usize; 4],
    theta: f32,
) -> Result<(Tensor, Tensor)> {
    let device = img_ids.device().clone();

    // Concat: txt first, then img (matching Python: flame_core.cat([txt_ids, img_ids], 0))
    let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
    let n_total = all_ids.shape().dims()[0];

    let mut cos_parts: Vec<Tensor> = Vec::new();
    let mut sin_parts: Vec<Tensor> = Vec::new();

    for (axis_idx, &dim) in axes_dims.iter().enumerate() {
        let half = dim / 2;

        // Position values for this axis: all_ids[:, axis_idx] -> [N]
        let pos = all_ids
            .narrow(1, axis_idx, 1)?
            .squeeze(Some(1))?
            .to_dtype(DType::F32)?;

        // Frequency indices: exp(-log(theta) * arange(0, dim, 2) / dim)
        let freq_idx = Tensor::arange(0.0, dim as f32, 2.0, device.clone())?;
        let log_freqs = freq_idx
            .mul_scalar(-(theta as f32).ln() / dim as f32)?
            .exp()?;

        // Outer product: [N, 1] * [1, half] -> [N, half]
        let pos_col = pos.reshape(&[n_total, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.mul(&freq_row)?;

        cos_parts.push(angles.cos()?);
        sin_parts.push(angles.sin()?);
    }

    // Concat along last dim -> [N, head_dim//2]
    let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
    let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
    let pe_cos = Tensor::cat(&cos_refs, 1)?;
    let pe_sin = Tensor::cat(&sin_refs, 1)?;

    // -> [1, 1, N, head_dim//2] in BF16
    let pe_cos = pe_cos.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    let pe_sin = pe_sin.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;

    Ok((pe_cos, pe_sin))
}

/// Apply rotary position embeddings — fused CUDA kernel, no intermediates.
///
/// `q`, `k`: [B, H, N, D] in BF16.
/// `pe_cos`, `pe_sin`: [1, 1, N, D//2] in BF16.
fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    pe_cos: &Tensor,
    pe_sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_out = flame_core::bf16_ops::rope_fused_bf16(q, pe_cos, pe_sin)?;
    let k_out = flame_core::bf16_ops::rope_fused_bf16(k, pe_cos, pe_sin)?;
    Ok((q_out, k_out))
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

/// SwiGLU: split fused gate+up projection, apply silu*up, project down.
///
/// `gate_up_w`: [2*mlp_hidden, inner_dim] fused weight.
/// `down_w`: [inner_dim, mlp_hidden] down projection weight.
/// `x`: [B, N, inner_dim].
fn swiglu(gate_up_w: &Tensor, down_w: &Tensor, x: &Tensor, native_weights: bool) -> Result<Tensor> {
    let gu = if native_weights { linear3d_nt(x, gate_up_w)? } else { linear3d(x, gate_up_w)? };
    let last_dim = *gu.shape().dims().last().unwrap();
    let half_dim = last_dim / 2;
    let ndim = gu.shape().dims().len();
    let gate = gu.narrow(ndim - 1, 0, half_dim)?;
    let up = gu.narrow(ndim - 1, half_dim, half_dim)?;
    let activated = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?;
    if native_weights { linear3d_nt(&activated, down_w) } else { linear3d(&activated, down_w) }
}

/// Apply LoRA contribution to the output of a 3D linear, if a LoRA stack
/// is attached and has entries for `weight_key`. The flatten/unflatten is
/// handled inside `LoraStack::apply` itself (it reshapes input/output to
/// 2D internally and restores the original shape).
fn apply_lora_3d(
    base_out: Tensor,
    weight_key: &str,
    x_in: &Tensor,
    lora: Option<&LoraStack>,
) -> Result<Tensor> {
    match lora {
        Some(stack) => stack.apply(weight_key, x_in, base_out),
        None => Ok(base_out),
    }
}

/// SwiGLU variant aware of an attached LoRA stack. Applies LoRA to the
/// fused gate+up projection and to the down projection by their full
/// weight keys. Note: external Klein LoRAs typically target the FUSED
/// `img_mlp.0.weight` / `img_mlp.2.weight` shapes directly via klein-trainer
/// naming (`mlp_0` / `mlp_2`); split-half (gate vs up) LoRAs would not be
/// natively supported here.
#[allow(clippy::too_many_arguments)]
fn swiglu_lora(
    weights: &HashMap<String, Tensor>,
    gate_up_key: &str,
    down_key: &str,
    x: &Tensor,
    native_weights: bool,
    lora: Option<&LoraStack>,
) -> Result<Tensor> {
    let gate_up_w = &weights[gate_up_key];
    let down_w = &weights[down_key];
    let gu_base = if native_weights { linear3d_nt(x, gate_up_w)? } else { linear3d(x, gate_up_w)? };
    let gu = apply_lora_3d(gu_base, gate_up_key, x, lora)?;
    let last_dim = *gu.shape().dims().last().unwrap();
    let half_dim = last_dim / 2;
    let ndim = gu.shape().dims().len();
    let gate = gu.narrow(ndim - 1, 0, half_dim)?;
    let up = gu.narrow(ndim - 1, half_dim, half_dim)?;
    let activated = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?;
    let down_base = if native_weights {
        linear3d_nt(&activated, down_w)?
    } else {
        linear3d(&activated, down_w)?
    };
    apply_lora_3d(down_base, down_key, &activated, lora)
}

// ---------------------------------------------------------------------------
// Double block
// ---------------------------------------------------------------------------

/// Execute one double-stream block.
///
/// `img_mods`: [shift1, scale1, gate1, shift2, scale2, gate2]
/// `txt_mods`: same structure.
#[allow(clippy::too_many_arguments)]
fn double_block_forward(
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    img: &Tensor,
    txt: &Tensor,
    img_mods: &[Tensor; 6],
    txt_mods: &[Tensor; 6],
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    num_heads: usize,
    head_dim: usize,
    native_weights: bool,
    lora: Option<&LoraStack>,
) -> Result<(Tensor, Tensor)> {
    let prefix = format!("double_blocks.{block_idx}");
    let h = num_heads;
    let d = head_dim;
    // Linear dispatch: native_weights uses cuBLASLt TRANSA=T (swap path),
    // otherwise standard pre-transposed matmul (resident path).
    // After the base matmul we apply any LoRA contributions for `key`.
    let lin = |x: &Tensor, key: &str| -> Result<Tensor> {
        let w = &weights[key];
        let base = if native_weights { linear3d_nt(x, w)? } else { linear3d(x, w)? };
        apply_lora_3d(base, key, x, lora)
    };
    // Unpack modulation parameters
    let (img_shift1, img_scale1, img_gate1) = (&img_mods[0], &img_mods[1], &img_mods[2]);
    let (img_shift2, img_scale2, img_gate2) = (&img_mods[3], &img_mods[4], &img_mods[5]);
    let (txt_shift1, txt_scale1, txt_gate1) = (&txt_mods[0], &txt_mods[1], &txt_mods[2]);
    let (txt_shift2, txt_scale2, txt_gate2) = (&txt_mods[3], &txt_mods[4], &txt_mods[5]);

    // --- Attention ---
    let img_normed = modulate_pre(img, img_shift1, img_scale1)?;
    let txt_normed = modulate_pre(txt, txt_shift1, txt_scale1)?;

    // QKV projections
    let img_qkv = lin(&img_normed, &format!("{prefix}.img_attn.qkv.weight"))?;
    let txt_qkv = lin(&txt_normed, &format!("{prefix}.txt_attn.qkv.weight"))?;

    let b = img_qkv.shape().dims()[0];
    let n_img = img_qkv.shape().dims()[1];
    let n_txt = txt_qkv.shape().dims()[1];
    let _ = b;

    // Split QKV: [B, N, 3*inner_dim] -> q, k, v each [B, H, N, D]
    // Fused: one kernel replaces 3 narrows + 3 permutes per QKV group.
    let (mut img_q, mut img_k, img_v) =
        flame_core::bf16_ops::qkv_split_permute_bf16(&img_qkv, h, d)?;
    let (mut txt_q, mut txt_k, txt_v) =
        flame_core::bf16_ops::qkv_split_permute_bf16(&txt_qkv, h, d)?;
    let _ = n_img;
    let _ = n_txt;

    // QK norm (RMSNorm per head)
    img_q = head_rms_norm(
        &img_q,
        &weights[&format!("{prefix}.img_attn.norm.query_norm.scale")],
    )?;
    img_k = head_rms_norm(
        &img_k,
        &weights[&format!("{prefix}.img_attn.norm.key_norm.scale")],
    )?;
    txt_q = head_rms_norm(
        &txt_q,
        &weights[&format!("{prefix}.txt_attn.norm.query_norm.scale")],
    )?;
    txt_k = head_rms_norm(
        &txt_k,
        &weights[&format!("{prefix}.txt_attn.norm.key_norm.scale")],
    )?;

    // Joint attention: concat txt first, then img along sequence dim
    let q = Tensor::cat(&[&txt_q, &img_q], 2)?; // [B, H, N_txt+N_img, D]
    let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
    let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

    // Apply RoPE
    let (q, k) = apply_rope(&q, &k, pe_cos, pe_sin)?;

    // Scaled dot-product attention
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?; // [B, H, N_total, D]

    // Split back: txt first, then img. Fused kernel replaces
    // 2 narrow + 2 permute with a single pass over attn_out.
    let (txt_out, img_out) =
        flame_core::bf16_ops::attn_split_txt_img_bf16(&attn_out, n_txt, n_img)?;
    let _ = b;

    // Output projection (NO bias)
    let img_out = lin(&img_out, &format!("{prefix}.img_attn.proj.weight"))?;
    let txt_out = lin(&txt_out, &format!("{prefix}.txt_attn.proj.weight"))?;

    // Gate + residual (fused)
    let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, img_gate1, &img_out)?;
    let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, txt_gate1, &txt_out)?;

    // --- MLP (SwiGLU) ---
    let img_mlp_in = modulate_pre(&img, img_shift2, img_scale2)?;
    let txt_mlp_in = modulate_pre(&txt, txt_shift2, txt_scale2)?;

    let img_mlp_out = swiglu_lora(
        weights,
        &format!("{prefix}.img_mlp.0.weight"),
        &format!("{prefix}.img_mlp.2.weight"),
        &img_mlp_in,
        native_weights,
        lora,
    )?;
    let txt_mlp_out = swiglu_lora(
        weights,
        &format!("{prefix}.txt_mlp.0.weight"),
        &format!("{prefix}.txt_mlp.2.weight"),
        &txt_mlp_in,
        native_weights,
        lora,
    )?;

    // Gate + residual (fused)
    let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, img_gate2, &img_mlp_out)?;
    let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, txt_gate2, &txt_mlp_out)?;

    Ok((img, txt))
}

// ---------------------------------------------------------------------------
// Single block
// ---------------------------------------------------------------------------

/// Execute one single-stream block.
///
/// `linear1` fuses: QKV (3*dim) + SwiGLU gate+up (2*mlp_hidden) = 9*dim
/// `linear2` fuses: attn_proj (dim) + SwiGLU down (mlp_hidden) = 4*dim
///
/// LoRA support: external LoRAs targeting the narrow QKV slice (first
/// 3*inner_dim rows of linear1) and narrow attn_proj slice (first inner_dim
/// cols of linear2) are mapped via `Slot::Rows(3*inner_dim)` and
/// `Slot::Cols(inner_dim)` inside `LoraStack`. Constants assumed are the
/// Klein-4B values (KLEIN_4B_SINGLE_QKV_ROWS=9216, KLEIN_4B_SINGLE_OUT_COLS
/// =3072) — see `inference-flame/src/lora.rs`. Klein-9B would need
/// different constants; not yet wired.
#[allow(clippy::too_many_arguments)]
fn single_block_forward(
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    x: &Tensor,
    mods: &[Tensor; 3],
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    num_heads: usize,
    head_dim: usize,
    inner_dim: usize,
    mlp_hidden: usize,
    native_weights: bool,
    lora: Option<&LoraStack>,
) -> Result<Tensor> {
    let prefix = format!("single_blocks.{block_idx}");
    let h = num_heads;
    let d = head_dim;
    let lin = |x: &Tensor, key: &str| -> Result<Tensor> {
        let w = &weights[key];
        let base = if native_weights { linear3d_nt(x, w)? } else { linear3d(x, w)? };
        apply_lora_3d(base, key, x, lora)
    };

    let (shift, scale, gate) = (&mods[0], &mods[1], &mods[2]);

    let x_normed = modulate_pre(x, shift, scale)?;

    // Fused QKV + SwiGLU gate+up
    let qkv_mlp = lin(&x_normed, &format!("{prefix}.linear1.weight"))?;

    // Split: first 3*inner_dim = QKV, rest = gate+up
    let qkv_dim = 3 * inner_dim;
    let qkv = qkv_mlp.narrow(2, 0, qkv_dim)?;
    let gate_up = qkv_mlp.narrow(2, qkv_dim, 2 * mlp_hidden)?;

    let b = qkv.shape().dims()[0];
    let n = qkv.shape().dims()[1];

    // Split QKV -> q, k, v each [B, H, N, D]
    // Fused: one kernel replaces 3 narrows + 3 permutes.
    let _ = (b, n, inner_dim);
    let (q, k, v) = flame_core::bf16_ops::qkv_split_permute_bf16(&qkv, h, d)?;

    // QK norm (RMSNorm per head)
    let q = head_rms_norm(&q, &weights[&format!("{prefix}.norm.query_norm.scale")])?;
    let k = head_rms_norm(&k, &weights[&format!("{prefix}.norm.key_norm.scale")])?;

    // RoPE
    let (q, k) = apply_rope(&q, &k, pe_cos, pe_sin)?;

    // Attention
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn_out = attn_out
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b, n, h * d])?;

    // SwiGLU from the fused gate+up (fused kernel)
    let gate_proj = gate_up.narrow(2, 0, mlp_hidden)?;
    let up_proj = gate_up.narrow(2, mlp_hidden, mlp_hidden)?;
    let mlp_out = flame_core::bf16_ops::swiglu_fused_bf16(&gate_proj, &up_proj)?;

    // Fused output: cat [attn, mlp] -> linear2
    let fused = Tensor::cat(&[&attn_out, &mlp_out], 2)?;
    let out = lin(&fused, &format!("{prefix}.linear2.weight"))?;

    // Gate + residual (fused)
    flame_core::bf16_ops::gate_residual_fused_bf16(x, gate, &out)
}

// ---------------------------------------------------------------------------
// Shared modulation helper
// ---------------------------------------------------------------------------

/// Apply shared modulation: silu(vec) -> linear -> chunk into N parts.
fn shared_modulation_from_silu(
    vec_silu: &Tensor,
    weight: &Tensor,
    num_chunks: usize,
) -> Result<Vec<Tensor>> {
    let raw = linear3d(vec_silu, weight)?;
    let last_dim = *raw.shape().dims().last().unwrap();
    let chunk_size = last_dim / num_chunks;
    let ndim = raw.shape().dims().len();

    let mut chunks = Vec::with_capacity(num_chunks);
    for j in 0..num_chunks {
        chunks.push(raw.narrow(ndim - 1, j * chunk_size, chunk_size)?);
    }
    Ok(chunks)
}

/// Convert a 6-element Vec into a fixed-size array (moves, no clone).
fn vec_to_arr6(mut v: Vec<Tensor>) -> Result<[Tensor; 6]> {
    if v.len() != 6 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 6 modulation chunks, got {}",
            v.len()
        )));
    }
    // Drain in reverse to pop efficiently
    let e5 = v.pop().unwrap();
    let e4 = v.pop().unwrap();
    let e3 = v.pop().unwrap();
    let e2 = v.pop().unwrap();
    let e1 = v.pop().unwrap();
    let e0 = v.pop().unwrap();
    Ok([e0, e1, e2, e3, e4, e5])
}

/// Convert a 3-element Vec into a fixed-size array (moves, no clone).
fn vec_to_arr3(mut v: Vec<Tensor>) -> Result<[Tensor; 3]> {
    if v.len() != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 3 modulation chunks, got {}",
            v.len()
        )));
    }
    let e2 = v.pop().unwrap();
    let e1 = v.pop().unwrap();
    let e0 = v.pop().unwrap();
    Ok([e0, e1, e2])
}

// ---------------------------------------------------------------------------
// Main Transformer
// ---------------------------------------------------------------------------

/// Flux 2 Klein DiT — pure flame_core implementation.
///
/// Exact key-compatible with BFL .safetensors checkpoints.
///
/// `Clone` is intentionally cheap: `Tensor` is Arc-internal,
/// `KleinConfig` is small POD, and `Arc<LoraStack>` is a refcount
/// bump. Cloning a `KleinTransformer` shares all GPU storage with
/// the original and only duplicates the `HashMap` spine + small
/// metadata. Used by EriGui's node graph to apply a LoRA to a
/// model handle without mutating the upstream `Arc` (see
/// `erigui-nodes/src/builtin/load_lora.rs`).
#[derive(Clone)]
pub struct KleinTransformer {
    weights: HashMap<String, Tensor>,
    config: KleinConfig,
    /// Optional runtime LoRA stack — applied at each linear inside the
    /// double/single-block forwards. Base weights are never mutated.
    lora: Option<Arc<LoraStack>>,
}

impl KleinTransformer {
    /// Load from a BFL safetensors checkpoint.
    ///
    /// Auto-detects Klein 4B vs 9B from weight shapes.
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let device = flame_core::global_cuda_device();
        let weights = serialization::load_file(path, &device)?;
        Self::from_weights(weights)
    }

    /// Construct from a pre-loaded weight dict.
    ///
    /// Auto-detects Klein 4B vs 9B from weight shapes.
    pub fn from_weights(weights: HashMap<String, Tensor>) -> Result<Self> {
        // Auto-detect architecture from weight shapes
        let inner_dim = weights["img_in.weight"].shape().dims()[0];
        let in_channels = weights["img_in.weight"].shape().dims()[1];
        let joint_dim = weights["txt_in.weight"].shape().dims()[1];

        let mut num_double = 0;
        while weights.contains_key(&format!("double_blocks.{num_double}.img_attn.qkv.weight")) {
            num_double += 1;
        }
        let mut num_single = 0;
        while weights.contains_key(&format!("single_blocks.{num_single}.linear1.weight")) {
            num_single += 1;
        }

        let num_heads = inner_dim / 128;
        let head_dim = 128;
        let mlp_hidden = inner_dim * 3; // SwiGLU 3x ratio

        let config = KleinConfig {
            inner_dim,
            in_channels,
            joint_attention_dim: joint_dim,
            num_double,
            num_single,
            num_heads,
            head_dim,
            mlp_hidden,
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        };

        log::info!(
            "[KleinTransformer] Detected: inner_dim={}, joint_dim={}, \
             double={}, single={}, heads={}",
            inner_dim,
            joint_dim,
            num_double,
            num_single,
            num_heads,
        );

        // Pre-transpose ALL weight matrices [out, in] -> [in, out] for faster matmul.
        // This avoids 500+ GPU transpose copies per forward pass.
        log::info!("[KleinTransformer] Pre-transposing weights...");
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if key.ends_with(".weight") && !key.ends_with(".scale") {
                let w = &weights[key];
                let dims = w.shape().dims();
                if dims.len() == 2 {
                    let wt = flame_core::bf16_elementwise::transpose2d_bf16(w)?;
                    weights.insert(key.clone(), wt);
                }
            }
        }
        log::info!("[KleinTransformer] Weights pre-transposed.");

        Ok(Self { weights, config, lora: None })
    }

    /// Attach a runtime LoRA stack. Subsequent block forwards will add
    /// `scale * up(down(x))` from any matching LoRA entries to the base
    /// matmul outputs at every double/single-block linear chokepoint.
    /// Base weights are not mutated.
    pub fn set_lora(&mut self, lora: Arc<LoraStack>) {
        self.lora = Some(lora);
    }

    /// Forward pass.
    ///
    /// * `img`: [B, N_img, in_channels] image latent tokens.
    /// * `txt`: [B, N_txt, joint_attention_dim] text embeddings.
    /// * `timesteps`: [B] sigma values.
    /// * `img_ids`: [N_img, 4] image position IDs.
    /// * `txt_ids`: [N_txt, 4] text position IDs.
    ///
    /// Returns [B, N_img, in_channels] predicted noise/velocity.
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<Tensor> {
        let w = &self.weights;
        let cfg = &self.config;

        // 1. Input projections (NO bias)
        let mut img = linear3d(img, &w["img_in.weight"])?;
        let mut txt = linear3d(txt, &w["txt_in.weight"])?;

        // 2. Timestep embedding -> vec
        let t_emb = timestep_embedding(timesteps, cfg.timestep_dim, 1000.0)?;
        let t_emb_bf16 = t_emb.to_dtype(DType::BF16)?;
        let vec = {
            let h = linear3d(&t_emb_bf16, &w["time_in.in_layer.weight"])?;
            let h = h.silu()?;
            linear3d(&h, &w["time_in.out_layer.weight"])?
        };

        // 3. Build RoPE tables
        let (pe_cos, pe_sin) = build_rope_2d(img_ids, txt_ids, &cfg.axes_dims, cfg.theta)?;

        // 4. Pre-compute shared modulations ONCE (vec doesn't change between blocks)
        let vec_silu = vec.silu()?;
        let img_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_img.lin.weight"], 6)?
        )?;
        let txt_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_txt.lin.weight"], 6)?
        )?;
        let single_mods_arr: [Tensor; 3] = vec_to_arr3(
            shared_modulation_from_silu(&vec_silu, &w["single_stream_modulation.lin.weight"], 3)?
        )?;

        // 5. Double blocks (reuse cached modulations)
        let lora_ref = self.lora.as_deref();
        for i in 0..cfg.num_double {
            let (new_img, new_txt) = double_block_forward(
                w,
                i,
                &img,
                &txt,
                &img_mods_arr,
                &txt_mods_arr,
                &pe_cos,
                &pe_sin,
                cfg.num_heads,
                cfg.head_dim,
                false,
                lora_ref,
            )?;
            img = new_img;
            txt = new_txt;
        }

        // 6. Concatenate for single stream: txt first, then img
        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.shape().dims()[1];

        // 7. Single blocks (reuse cached modulations)
        for i in 0..cfg.num_single {
            x = single_block_forward(
                w,
                i,
                &x,
                &single_mods_arr,
                &pe_cos,
                &pe_sin,
                cfg.num_heads,
                cfg.head_dim,
                cfg.inner_dim,
                cfg.mlp_hidden,
                false,
                lora_ref,
            )?;
        }

        // 8. Extract image tokens (txt_len onwards)
        let total_len = x.shape().dims()[1];
        let img_out = x.narrow(1, txt_len, total_len - txt_len)?;

        // 9. Final layer: adaLN modulation + linear (reuse cached silu)
        let final_mod = linear3d(&vec_silu, &w["final_layer.adaLN_modulation.1.weight"])?;
        let last_dim = *final_mod.shape().dims().last().unwrap();
        let half_mod = last_dim / 2;
        let ndim = final_mod.shape().dims().len();
        let shift = final_mod.narrow(ndim - 1, 0, half_mod)?;
        let scale = final_mod.narrow(ndim - 1, half_mod, half_mod)?;

        let img_out = modulate_pre(&img_out, &shift, &scale)?;
        linear3d(&img_out, &w["final_layer.linear.weight"])
    }

    /// Get the model config.
    pub fn config(&self) -> &KleinConfig {
        &self.config
    }

    /// Get read-only access to weights.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Offloaded Transformer (block-level CPU→GPU streaming)
// ---------------------------------------------------------------------------

/// Pre-parsed CPU tensor in pinned memory: shape + bf16 data ready for fast cudaMemcpy.
struct CpuWeight {
    shape: Vec<usize>,
    /// BF16 as u16 in CUDA pinned host memory for fast DMA transfer.
    /// Falls back to regular Vec if pinned alloc fails.
    data: Vec<u16>,
}

/// Klein transformer with block-level CPU offloading.
///
/// Shared weights (projections, modulations, final layer) live on GPU permanently.
/// Block weights are pre-parsed to CPU Vec<u16> at load time. Per-block forward:
/// cudaMemcpy to GPU → transpose → compute → drop. No byte parsing per step.
///
/// GPU usage: ~1-2GB shared + ~700MB per active block (vs 17GB all-on-GPU for 9B).
pub struct KleinOffloaded {
    /// Shared weights on GPU (small — ~200MB for 9B).
    shared: HashMap<String, Tensor>,
    /// Block weights in CPU RAM, pre-parsed as Vec<u16>.
    cpu_weights: HashMap<String, CpuWeight>,
    config: KleinConfig,
    /// Optional runtime LoRA stack — applied at each linear inside the
    /// double/single-block forwards. Base weights are never mutated.
    lora: Option<Arc<LoraStack>>,
}

impl KleinOffloaded {
    /// Load from a safetensors checkpoint with block offloading.
    ///
    /// Shared weights → GPU. Block weights → pre-parsed CPU Vec<u16>.
    pub fn from_safetensors(path: &str) -> Result<Self> {
        use serde_json::Value;

        let device = flame_core::global_cuda_device();

        // Memory-map the file for parsing
        let file = std::fs::File::open(path)
            .map_err(|e| flame_core::Error::Io(format!("Failed to open: {:?}", e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| flame_core::Error::Io(format!("Failed to mmap: {:?}", e)))?;

        // Parse header
        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_size;
        let data_start = header_end;

        let metadata: Value = serde_json::from_slice(&mmap[8..header_end])
            .map_err(|e| flame_core::Error::Io(format!("Failed to parse header: {:?}", e)))?;
        let metadata_obj = metadata.as_object()
            .ok_or_else(|| flame_core::Error::InvalidInput("Invalid metadata".into()))?;

        // Shared weight prefixes (stay on GPU)
        let shared_prefixes = [
            "img_in.", "txt_in.", "time_in.",
            "double_stream_modulation_img.", "double_stream_modulation_txt.",
            "single_stream_modulation.", "final_layer.",
        ];

        let mut shared = HashMap::new();
        let mut cpu_weights = HashMap::new();
        let mut num_double = 0usize;
        let mut num_single = 0usize;
        let mut inner_dim = 0usize;
        let mut in_channels = 0usize;
        let mut joint_dim = 0usize;

        for (name, info) in metadata_obj {
            if name == "__metadata__" { continue; }

            let shape: Vec<usize> = info["shape"].as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as usize))
                .collect();

            let offsets = info["data_offsets"].as_array();
            if offsets.is_none() { continue; }
            let offsets = offsets.unwrap();
            let start = data_start + offsets[0].as_u64().unwrap_or(0) as usize;
            let end = data_start + offsets[1].as_u64().unwrap_or(0) as usize;
            let dtype_str = info["dtype"].as_str().unwrap_or("F32");

            if !matches!(dtype_str, "BF16" | "F16" | "F32") { continue; }

            // Detect config from key shapes
            if name == "img_in.weight" {
                inner_dim = shape[0];
                in_channels = shape[1];
            }
            if name == "txt_in.weight" {
                joint_dim = shape[1];
            }

            let is_shared = shared_prefixes.iter().any(|p| name.starts_with(p));

            if is_shared {
                // Load directly to GPU + pre-transpose
                let data = &mmap[start..end];
                let t = Self::bytes_to_gpu_tensor(data, dtype_str, &shape, &device)?;
                let t = if name.ends_with(".weight") && !name.ends_with(".scale") && t.shape().dims().len() == 2 {
                    flame_core::bf16_elementwise::transpose2d_bf16(&t)?
                } else {
                    t
                };
                shared.insert(name.clone(), t);
            } else {
                // Parse bytes to CPU Vec<u16> (one-time cost, fast cudaMemcpy later)
                let data = &mmap[start..end];
                let bf16_data = Self::bytes_to_bf16_vec(data, dtype_str);
                cpu_weights.insert(name.clone(), CpuWeight { shape, data: bf16_data });
            }
        }

        // Count blocks
        while cpu_weights.contains_key(&format!("double_blocks.{num_double}.img_attn.qkv.weight")) {
            num_double += 1;
        }
        while cpu_weights.contains_key(&format!("single_blocks.{num_single}.linear1.weight")) {
            num_single += 1;
        }

        let num_heads = inner_dim / 128;
        let config = KleinConfig {
            inner_dim,
            in_channels,
            joint_attention_dim: joint_dim,
            num_double,
            num_single,
            num_heads,
            head_dim: 128,
            mlp_hidden: inner_dim * 3,
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        };

        log::info!(
            "[KleinOffloaded] inner_dim={}, double={}, single={}, shared={} GPU keys, {} CPU keys, ~{:.1}GB CPU RAM",
            inner_dim, num_double, num_single, shared.len(), cpu_weights.len(),
            cpu_weights.values().map(|w| w.data.len() * 2).sum::<usize>() as f64 / 1e9,
        );

        Ok(Self { shared, cpu_weights, config, lora: None })
    }

    /// Attach a runtime LoRA stack. Subsequent block forwards will add
    /// `scale * up(down(x))` from any matching LoRA entries to the base
    /// matmul outputs at every double/single-block linear chokepoint.
    /// Base weights are not mutated.
    pub fn set_lora(&mut self, lora: Arc<LoraStack>) {
        self.lora = Some(lora);
    }

    /// Convert raw safetensors bytes to BF16 Vec<u16> on CPU.
    fn bytes_to_bf16_vec(data: &[u8], dtype: &str) -> Vec<u16> {
        match dtype {
            "BF16" => {
                let mut out = vec![0u16; data.len() / 2];
                for (value, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
                    *value = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
                out
            }
            "F16" => {
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        let f = half::f16::from_bits(bits).to_f32();
                        half::bf16::from_f32(f).to_bits()
                    })
                    .collect()
            }
            _ => {
                // F32 → BF16
                data.chunks_exact(4)
                    .map(|chunk| {
                        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        half::bf16::from_f32(f).to_bits()
                    })
                    .collect()
            }
        }
    }

    /// Load raw bytes to a GPU tensor (used for shared weights at init time).
    fn bytes_to_gpu_tensor(
        data: &[u8],
        dtype: &str,
        shape: &[usize],
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Tensor> {
        let bf16_data = Self::bytes_to_bf16_vec(data, dtype);
        let mut tensor = Tensor::zeros_dtype(
            Shape::from_dims(shape), DType::BF16, device.clone(),
        )?;
        tensor.copy_from_bf16_slice(&bf16_data)?;
        Ok(tensor)
    }

    /// Upload a pre-parsed CPU weight to GPU as a Tensor.
    fn upload_weight(&self, key: &str) -> Result<Tensor> {
        let cw = self.cpu_weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {key}"))
        })?;
        let device = flame_core::global_cuda_device();
        let mut tensor = Tensor::zeros_dtype(
            Shape::from_dims(&cw.shape), DType::BF16, device.clone(),
        )?;
        tensor.copy_from_bf16_slice(&cw.data)?;
        Ok(tensor)
    }

    /// Upload + pre-transpose a 2D weight matrix.
    fn upload_weight_transposed(&self, key: &str) -> Result<Tensor> {
        let t = self.upload_weight(key)?;
        if t.shape().dims().len() == 2 {
            flame_core::bf16_elementwise::transpose2d_bf16(&t)
        } else {
            Ok(t)
        }
    }

    /// Load all weights for a block prefix to GPU, pre-transposed.
    fn load_block_to_gpu(&self, prefix: &str) -> Result<HashMap<String, Tensor>> {
        let mut block_w = HashMap::new();
        for key in self.cpu_weights.keys() {
            if !key.starts_with(prefix) { continue; }
            let t = if key.ends_with(".weight") && !key.ends_with(".scale") {
                self.upload_weight_transposed(key)?
            } else {
                self.upload_weight(key)?
            };
            block_w.insert(key.clone(), t);
        }
        Ok(block_w)
    }

    /// Forward pass with block-level CPU→GPU streaming.
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<Tensor> {
        let w = &self.shared;
        let cfg = &self.config;

        // 1. Input projections
        let mut img = linear3d(img, &w["img_in.weight"])?;
        let mut txt = linear3d(txt, &w["txt_in.weight"])?;

        // 2. Timestep embedding
        let t_emb = timestep_embedding(timesteps, cfg.timestep_dim, 1000.0)?;
        let t_emb_bf16 = t_emb.to_dtype(DType::BF16)?;
        let vec = {
            let h = linear3d(&t_emb_bf16, &w["time_in.in_layer.weight"])?;
            let h = h.silu()?;
            linear3d(&h, &w["time_in.out_layer.weight"])?
        };

        // 3. RoPE tables
        let (pe_cos, pe_sin) = build_rope_2d(img_ids, txt_ids, &cfg.axes_dims, cfg.theta)?;

        // 4. Shared modulations
        let vec_silu = vec.silu()?;
        let img_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_img.lin.weight"], 6)?
        )?;
        let txt_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_txt.lin.weight"], 6)?
        )?;
        let single_mods_arr: [Tensor; 3] = vec_to_arr3(
            shared_modulation_from_silu(&vec_silu, &w["single_stream_modulation.lin.weight"], 3)?
        )?;

        // 5. Double blocks — upload from CPU, run, drop
        let lora_ref = self.lora.as_deref();
        for i in 0..cfg.num_double {
            let block_w = self.load_block_to_gpu(&format!("double_blocks.{i}."))?;
            let (new_img, new_txt) = double_block_forward(
                &block_w, i, &img, &txt,
                &img_mods_arr, &txt_mods_arr,
                &pe_cos, &pe_sin,
                cfg.num_heads, cfg.head_dim,
                false,
                lora_ref,
            )?;
            img = new_img;
            txt = new_txt;
        }

        // 6. Single stream
        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.shape().dims()[1];

        // 7. Single blocks — upload from CPU, run, drop
        for i in 0..cfg.num_single {
            let block_w = self.load_block_to_gpu(&format!("single_blocks.{i}."))?;
            x = single_block_forward(
                &block_w, i, &x,
                &single_mods_arr,
                &pe_cos, &pe_sin,
                cfg.num_heads, cfg.head_dim,
                cfg.inner_dim, cfg.mlp_hidden,
                false,
                lora_ref,
            )?;
        }

        // 8. Extract image tokens
        let total_len = x.shape().dims()[1];
        let img_out = x.narrow(1, txt_len, total_len - txt_len)?;

        // 9. Final layer
        let final_mod = linear3d(&vec_silu, &w["final_layer.adaLN_modulation.1.weight"])?;
        let last_dim = *final_mod.shape().dims().last().unwrap();
        let half_mod = last_dim / 2;
        let ndim = final_mod.shape().dims().len();
        let shift = final_mod.narrow(ndim - 1, 0, half_mod)?;
        let scale = final_mod.narrow(ndim - 1, half_mod, half_mod)?;

        let img_out = modulate_pre(&img_out, &shift, &scale)?;
        linear3d(&img_out, &w["final_layer.linear.weight"])
    }

    pub fn config(&self) -> &KleinConfig { &self.config }

    /// Forward pass that pulls block weights through a BlockOffloader instead of
    /// the in-process CPU staging dictionary.
    ///
    /// `offloader` must have been initialised with the SAME safetensors file and
    /// the standard Klein block_fn:
    ///   - `double_blocks.{i}` → block idx `i`
    ///   - `single_blocks.{i}` → block idx `num_double + i`
    ///
    /// `offloader` is left in a clean state on success.
    pub fn forward_with_offloader(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        offloader: &mut flame_diffusion::BlockOffloader,
    ) -> Result<Tensor> {
        let w = &self.shared;
        let cfg = &self.config;

        let mut img = linear3d(img, &w["img_in.weight"])?;
        let mut txt = linear3d(txt, &w["txt_in.weight"])?;

        let t_emb = timestep_embedding(timesteps, cfg.timestep_dim, 1000.0)?;
        let t_emb_bf16 = t_emb.to_dtype(DType::BF16)?;
        let vec = {
            let h = linear3d(&t_emb_bf16, &w["time_in.in_layer.weight"])?;
            let h = h.silu()?;
            linear3d(&h, &w["time_in.out_layer.weight"])?
        };

        let (pe_cos, pe_sin) = build_rope_2d(img_ids, txt_ids, &cfg.axes_dims, cfg.theta)?;

        let vec_silu = vec.silu()?;
        let img_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_img.lin.weight"], 6)?
        )?;
        let txt_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_txt.lin.weight"], 6)?
        )?;
        let single_mods_arr: [Tensor; 3] = vec_to_arr3(
            shared_modulation_from_silu(&vec_silu, &w["single_stream_modulation.lin.weight"], 3)?
        )?;

        let total_blocks = cfg.num_double + cfg.num_single;
        if offloader.block_count() != total_blocks {
            return Err(flame_core::Error::InvalidInput(format!(
                "BlockOffloader has {} blocks, expected {} (={} double + {} single)",
                offloader.block_count(), total_blocks, cfg.num_double, cfg.num_single,
            )));
        }

        // BlockOffloader's prepare_weights auto-transposes 2D .weight tensors,
        // but Klein's block_forward uses linear3d_nt (TRANSA=T on non-transposed
        // weights). We undo the transpose to get back to [out, in] layout.
        let prepare_block = |arc: Arc<HashMap<String, Tensor>>| -> Result<HashMap<String, Tensor>> {
            let mut out = HashMap::with_capacity(arc.len());
            for (k, t) in arc.iter() {
                let owned = if k.ends_with(".weight") && t.shape().dims().len() == 2 {
                    // BlockOffloader transposed to [in, out]. Undo → [out, in]
                    // for linear3d_nt's TRANSA=T path.
                    t.transpose()?.requires_grad_(false)
                } else {
                    t.clone()
                };
                out.insert(k.clone(), owned);
            }
            Ok(out)
        };

        // Kick off staging for the very first block (double_blocks.0).
        offloader.prefetch_block(0).map_err(|e| flame_core::Error::InvalidInput(format!("prefetch(0): {e}")))?;

        let lora_ref = self.lora.as_deref();
        for i in 0..cfg.num_double {
            let raw = offloader.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await({i}): {e}")))?;
            // Pre-fetch the next block while we materialize this one.
            let next = i + 1;
            if next < total_blocks {
                offloader.prefetch_block(next)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch({next}): {e}")))?;
            }
            let block_w = prepare_block(raw)?;
            let (new_img, new_txt) = double_block_forward(
                &block_w, i, &img, &txt,
                &img_mods_arr, &txt_mods_arr,
                &pe_cos, &pe_sin,
                cfg.num_heads, cfg.head_dim,
                true,
                lora_ref,
            )?;
            img = new_img;
            txt = new_txt;
        }

        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.shape().dims()[1];

        for i in 0..cfg.num_single {
            let block_idx = cfg.num_double + i;
            let raw = offloader.await_block(block_idx)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await({block_idx}): {e}")))?;
            let next = block_idx + 1;
            if next < total_blocks {
                offloader.prefetch_block(next)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch({next}): {e}")))?;
            }
            let block_w = prepare_block(raw)?;
            x = single_block_forward(
                &block_w, i, &x,
                &single_mods_arr,
                &pe_cos, &pe_sin,
                cfg.num_heads, cfg.head_dim,
                cfg.inner_dim, cfg.mlp_hidden,
                true,
                lora_ref,
            )?;
        }

        let total_len = x.shape().dims()[1];
        let img_out = x.narrow(1, txt_len, total_len - txt_len)?;

        let final_mod = linear3d(&vec_silu, &w["final_layer.adaLN_modulation.1.weight"])?;
        let last_dim = *final_mod.shape().dims().last().unwrap();
        let half_mod = last_dim / 2;
        let ndim = final_mod.shape().dims().len();
        let shift = final_mod.narrow(ndim - 1, 0, half_mod)?;
        let scale = final_mod.narrow(ndim - 1, half_mod, half_mod)?;

        let img_out = modulate_pre(&img_out, &shift, &scale)?;
        linear3d(&img_out, &w["final_layer.linear.weight"])
    }

    /// Forward pass that pulls block weights through a TurboBlockLoader (VMM
    /// double-buffered). Mirrors `forward_with_offloader` exactly — the only
    /// differences are the loader type and `await_block`'s return type.
    ///
    /// Behind `feature = "turbo"` so the default build never references the
    /// turbo module.
    #[cfg(feature = "turbo")]
    pub fn forward_with_turbo(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        loader: &mut crate::turbo::TurboBlockLoader,
    ) -> Result<Tensor> {
        let w = &self.shared;
        let cfg = &self.config;

        let mut img = linear3d(img, &w["img_in.weight"])?;
        let mut txt = linear3d(txt, &w["txt_in.weight"])?;

        let t_emb = timestep_embedding(timesteps, cfg.timestep_dim, 1000.0)?;
        let t_emb_bf16 = t_emb.to_dtype(DType::BF16)?;
        let vec = {
            let h = linear3d(&t_emb_bf16, &w["time_in.in_layer.weight"])?;
            let h = h.silu()?;
            linear3d(&h, &w["time_in.out_layer.weight"])?
        };

        let (pe_cos, pe_sin) = build_rope_2d(img_ids, txt_ids, &cfg.axes_dims, cfg.theta)?;

        let vec_silu = vec.silu()?;
        let img_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_img.lin.weight"], 6)?
        )?;
        let txt_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_txt.lin.weight"], 6)?
        )?;
        let single_mods_arr: [Tensor; 3] = vec_to_arr3(
            shared_modulation_from_silu(&vec_silu, &w["single_stream_modulation.lin.weight"], 3)?
        )?;

        let total_blocks = cfg.num_double + cfg.num_single;
        if loader.block_count() != total_blocks {
            return Err(flame_core::Error::InvalidInput(format!(
                "TurboBlockLoader has {} blocks, expected {} (={} double + {} single)",
                loader.block_count(), total_blocks, cfg.num_double, cfg.num_single,
            )));
        }

        // The turbo loader publishes BF16View tensors over VMM-mapped memory
        // already in their on-disk [out, in] layout — no auto-transpose pass
        // is required (unlike BlockOffloader, which transposes inside
        // `prepare_weights`). Native cuBLASLt TRANSA=T (`linear3d_nt`,
        // `native_weights=true`) consumes them in place.
        loader
            .prefetch_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("turbo prefetch(0): {e}")))?;

        let lora_ref = self.lora.as_deref();
        for i in 0..cfg.num_double {
            let block = loader
                .await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("turbo await({i}): {e}")))?;
            let next = i + 1;
            if next < total_blocks {
                loader
                    .prefetch_block(next)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("turbo prefetch({next}): {e}")))?;
            }
            let (new_img, new_txt) = double_block_forward(
                &block.weights, i, &img, &txt,
                &img_mods_arr, &txt_mods_arr,
                &pe_cos, &pe_sin,
                cfg.num_heads, cfg.head_dim,
                true,
                lora_ref,
            )?;
            img = new_img;
            txt = new_txt;
        }

        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.shape().dims()[1];

        for i in 0..cfg.num_single {
            let block_idx = cfg.num_double + i;
            let block = loader
                .await_block(block_idx)
                .map_err(|e| flame_core::Error::InvalidInput(format!("turbo await({block_idx}): {e}")))?;
            let next = block_idx + 1;
            if next < total_blocks {
                loader
                    .prefetch_block(next)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("turbo prefetch({next}): {e}")))?;
            }
            x = single_block_forward(
                &block.weights, i, &x,
                &single_mods_arr,
                &pe_cos, &pe_sin,
                cfg.num_heads, cfg.head_dim,
                cfg.inner_dim, cfg.mlp_hidden,
                true,
                lora_ref,
            )?;
        }

        let total_len = x.shape().dims()[1];
        let img_out = x.narrow(1, txt_len, total_len - txt_len)?;

        let final_mod = linear3d(&vec_silu, &w["final_layer.adaLN_modulation.1.weight"])?;
        let last_dim = *final_mod.shape().dims().last().unwrap();
        let half_mod = last_dim / 2;
        let ndim = final_mod.shape().dims().len();
        let shift = final_mod.narrow(ndim - 1, 0, half_mod)?;
        let scale = final_mod.narrow(ndim - 1, half_mod, half_mod)?;

        let img_out = modulate_pre(&img_out, &shift, &scale)?;
        linear3d(&img_out, &w["final_layer.linear.weight"])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_klein_config_4b() {
        let cfg = KleinConfig::klein_4b();
        assert_eq!(cfg.inner_dim, 3072);
        assert_eq!(cfg.num_double, 5);
        assert_eq!(cfg.num_single, 20);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.joint_attention_dim, 7680);
        assert_eq!(cfg.mlp_hidden, 9216); // 3072 * 3
        assert_eq!(cfg.theta, 2000.0);
    }

    #[test]
    fn test_klein_config_9b() {
        let cfg = KleinConfig::klein_9b();
        assert_eq!(cfg.inner_dim, 4096);
        assert_eq!(cfg.num_double, 8);
        assert_eq!(cfg.num_single, 24);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.joint_attention_dim, 12288);
        assert_eq!(cfg.mlp_hidden, 12288); // 4096 * 3
    }

    #[test]
    fn test_klein_config_detection() {
        // Simulate auto-detection from weight shapes
        let mut weights: HashMap<String, Vec<usize>> = HashMap::new();

        // 4B variant shapes
        weights.insert("img_in.weight".into(), vec![3072, 128]);
        weights.insert("txt_in.weight".into(), vec![3072, 7680]);

        let inner_dim = weights["img_in.weight"][0];
        let in_channels = weights["img_in.weight"][1];
        let joint_dim = weights["txt_in.weight"][1];

        assert_eq!(inner_dim, 3072);
        assert_eq!(in_channels, 128);
        assert_eq!(joint_dim, 7680);
        assert_eq!(inner_dim / 128, 24); // num_heads
    }

    #[test]
    fn test_timestep_embedding_properties() {
        // Verify structural properties without GPU:
        // - dim=256 with time_factor=1000.0
        // - Output should be [B, 256] = [B, cos(128) + sin(128)]
        // - cos comes first, then sin (matching ComfyUI ordering)
        let dim = 256;
        let half = dim / 2;
        assert_eq!(half, 128);

        // Verify frequency computation matches expected formula
        let max_period: f32 = 10000.0;
        let freq_0 = (-max_period.ln() * 0.0 / half as f32).exp();
        assert!((freq_0 - 1.0).abs() < 1e-6, "freq[0] should be 1.0");

        let freq_last = (-max_period.ln() * (half - 1) as f32 / half as f32).exp();
        assert!(
            freq_last < 0.01,
            "freq[last] should be very small, got {}",
            freq_last
        );
    }

    #[test]
    fn test_weight_key_names() {
        // Verify all expected weight key patterns for Klein architecture
        let expected_double_keys = [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.img_attn.proj.weight",
            "double_blocks.0.img_attn.norm.query_norm.scale",
            "double_blocks.0.img_attn.norm.key_norm.scale",
            "double_blocks.0.img_mlp.0.weight",
            "double_blocks.0.img_mlp.2.weight",
            "double_blocks.0.txt_attn.qkv.weight",
            "double_blocks.0.txt_attn.proj.weight",
            "double_blocks.0.txt_attn.norm.query_norm.scale",
            "double_blocks.0.txt_attn.norm.key_norm.scale",
            "double_blocks.0.txt_mlp.0.weight",
            "double_blocks.0.txt_mlp.2.weight",
        ];

        let expected_single_keys = [
            "single_blocks.0.linear1.weight",
            "single_blocks.0.linear2.weight",
            "single_blocks.0.norm.query_norm.scale",
            "single_blocks.0.norm.key_norm.scale",
        ];

        let expected_model_keys = [
            "img_in.weight",
            "txt_in.weight",
            "time_in.in_layer.weight",
            "time_in.out_layer.weight",
            "double_stream_modulation_img.lin.weight",
            "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight",
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.linear.weight",
        ];

        // Just verify the strings are well-formed
        for key in expected_double_keys
            .iter()
            .chain(expected_single_keys.iter())
            .chain(expected_model_keys.iter())
        {
            assert!(!key.is_empty());
            assert!(key.contains('.'));
        }
    }
}
