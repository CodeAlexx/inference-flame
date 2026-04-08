//! LTX-2.3 (Lightricks 18.88B) Video+Audio Transformer — pure flame_core.
//!
//! Architecture matches `diffusers.LTX2VideoTransformer3DModel` exactly.
//! Key-exact for Lightricks/LTX-2 safetensors checkpoints.
//!
//! This is a dual-stream (video + audio) transformer with:
//! - Video stream: inner_dim = num_heads * head_dim (default 32 * 128 = 4096)
//! - Audio stream: audio_inner_dim = audio_heads * audio_head_dim (default 32 * 64 = 2048)
//! - 48 transformer blocks with self-attn, cross-attn, a2v/v2a cross-attn, FFN
//! - RoPE positional embeddings (3D for video, 1D for audio)
//! - AdaLN-Single timestep conditioning with 6 mod params per block
//! - GELU-approximate FeedForward (4x expansion)
//! - RMSNorm QK normalization across heads
//!
//! At 18.88B parameters, this model REQUIRES block-level CPU offloading
//! (Stagehand-style) for inference on 24GB GPUs.
//!
//! Weight key format (diffusers):
//!   proj_in.{weight,bias}
//!   audio_proj_in.{weight,bias}
//!   caption_projection.linear_{1,2}.{weight,bias}
//!   audio_caption_projection.linear_{1,2}.{weight,bias}
//!   time_embed.emb.timestep_embedder.linear_{1,2}.{weight,bias}
//!   time_embed.linear.{weight,bias}
//!   audio_time_embed.emb.timestep_embedder.linear_{1,2}.{weight,bias}
//!   audio_time_embed.linear.{weight,bias}
//!   av_cross_attn_video_scale_shift.{emb,linear}.* (similarly for other cross attn mods)
//!   scale_shift_table, audio_scale_shift_table
//!   transformer_blocks.{i}.norm1.weight (RMSNorm, no affine by default)
//!   transformer_blocks.{i}.attn1.{to_q,to_k,to_v}.{weight,bias}
//!   transformer_blocks.{i}.attn1.{norm_q,norm_k}.weight
//!   transformer_blocks.{i}.attn1.to_out.0.{weight,bias}
//!   transformer_blocks.{i}.ff.net.0.proj.{weight,bias}   (GELU-approximate)
//!   transformer_blocks.{i}.ff.net.2.{weight,bias}        (output projection)
//!   transformer_blocks.{i}.scale_shift_table              [6, dim]
//!   ... (audio_* mirrors for each block)
//!   norm_out.{weight,bias} (LayerNorm, no affine)
//!   proj_out.{weight,bias}
//!   audio_norm_out.{weight,bias}
//!   audio_proj_out.{weight,bias}

use flame_core::ops::fused_inference::{
    fused_modulate, fused_residual_gate, fused_rms_norm, fused_rms_norm_modulate,
};
use flame_core::serialization;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture config for LTX-2 variants.
#[derive(Debug, Clone)]
pub struct LTX2Config {
    // Video stream
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,

    // Audio stream
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_patch_size: usize,
    pub audio_patch_size_t: usize,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_scale_factor: usize,
    pub audio_sampling_rate: usize,
    pub audio_hop_length: usize,

    // Shared
    pub num_layers: usize,
    pub caption_channels: usize,
    pub norm_eps: f32,
    pub attention_bias: bool,

    // RoPE
    pub vae_scale_factors: [usize; 3],
    pub pos_embed_max_pos: usize,
    pub base_height: usize,
    pub base_width: usize,
    pub rope_theta: f64,
    pub causal_offset: usize,
    pub timestep_scale_multiplier: f64,
    pub cross_attn_timestep_scale_multiplier: f64,
}

impl Default for LTX2Config {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 32,
            attention_head_dim: 128,
            cross_attention_dim: 4096,

            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_patch_size: 1,
            audio_patch_size_t: 1,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_cross_attention_dim: 2048,
            audio_scale_factor: 4,
            audio_sampling_rate: 16000,
            audio_hop_length: 160,

            num_layers: 48,
            caption_channels: 3840,
            norm_eps: 1e-6,
            attention_bias: true,

            vae_scale_factors: [8, 32, 32],
            pos_embed_max_pos: 20,
            base_height: 2048,
            base_width: 2048,
            rope_theta: 10000.0,
            causal_offset: 1,
            timestep_scale_multiplier: 1000.0,
            cross_attn_timestep_scale_multiplier: 1000.0,
        }
    }
}

impl LTX2Config {
    /// Video inner dimension: num_heads * head_dim
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio inner dimension: audio_heads * audio_head_dim
    pub fn audio_inner_dim(&self) -> usize {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    /// FFN hidden dimension (4x expansion, GELU-approximate)
    pub fn ffn_hidden(&self) -> usize {
        self.inner_dim() * 4
    }

    /// Audio FFN hidden dimension
    pub fn audio_ffn_hidden(&self) -> usize {
        self.audio_inner_dim() * 4
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute `x @ weight.T + bias` for x=[B, N, Cin], weight=[Cout, Cin] (PyTorch
/// `nn.Linear` layout), bias=[Cout]. Routes to `fused_linear3d_native` which
/// uses cuBLASLt with TRANSA=T (handles the transpose inside the GEMM, no
/// per-call transpose kernel). 2D inputs are unsqueezed to 3D and squeezed
/// back; that path is rare (only `AdaLayerNormSingle` style ops).
fn linear3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    if shape.len() == 3 {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, bias)
    } else if shape.len() == 2 {
        let x3 = x.unsqueeze(0)?;
        let out3 = flame_core::ops::fused_inference::fused_linear3d_native(&x3, weight, bias)?;
        out3.squeeze_dim(0)
    } else {
        Err(flame_core::Error::InvalidShape(format!(
            "linear3d expects rank 2 or 3, got {:?}",
            shape
        )))
    }
}

/// Identity — kept as a stub so existing callers compile.  In the old path
/// this transposed weights from `[Cout, Cin]` to `[Cin, Cout]` so a plain
/// matmul could skip a runtime transpose.  The new `linear3d` uses
/// `fused_linear3d_native` which absorbs the transpose into the cuBLASLt GEMM
/// itself, so weights stay in the native PyTorch `[Cout, Cin]` layout.
///
/// This MUST stay a no-op: with FlameSwap v2 the swapper returns non-owning
/// views into pre-allocated GPU staging buffers, and calling `.transpose()`
/// on each weight per block per step was burning ~6 seconds per block.
fn pre_transpose_weight(w: &Tensor) -> Result<Tensor> {
    Ok(w.clone())
}

/// Compute `x @ weight^T` for 2D tensors. Wraps `linear3d` (which routes
/// through `fused_linear3d_native`).  `weight` is in PyTorch `[Cout, Cin]`
/// layout, NOT pre-transposed.
fn matmul_weight_t(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let x3 = x.unsqueeze(0)?;
    let out3 = flame_core::ops::fused_inference::fused_linear3d_native(&x3, weight, None)?;
    out3.squeeze_dim(0)
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

/// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

/// RMSNorm: x / sqrt(mean(x^2) + eps), optionally scaled by weight.
///
/// REVERTED to the slow F32 manual path for diagnosis: the BF16 fast kernel
/// (`flame_core::cuda_ops_bf16::rms_norm_bf16`) was producing slightly
/// different values from PyTorch's `torch.nn.functional.rms_norm`, which
/// compounds to gray output across 48 blocks.  Performance regressed to the
/// pre-1f32f84 baseline; if the F32 path produces real video, we know the
/// fast kernel is the culprit and we can either fix the kernel or keep the
/// F32 path.
fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let last_dim = dims[dims.len() - 1];

    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.mul(&x_f32)?;
    let mean_sq = x_sq.mean_along_dims(&[dims.len() - 1], true)?;

    let mut bc_shape = dims.clone();
    bc_shape[dims.len() - 1] = 1;
    let mean_sq = mean_sq.reshape(&bc_shape)?;

    let rsqrt = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = x_f32.mul(&rsqrt)?.to_dtype(DType::BF16)?;

    if let Some(w) = weight {
        let w_bc = w.reshape(&vec![1; dims.len() - 1].into_iter().chain(std::iter::once(last_dim)).collect::<Vec<_>>())?;
        normed.mul(&w_bc)
    } else {
        Ok(normed)
    }
}

/// LayerNorm without affine parameters: (x - mean) / sqrt(var + eps)
fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let rank = dims.len();

    let x_f32 = x.to_dtype(DType::F32)?;

    // Construct broadcast shape: [..., 1]
    let mut bc_shape = dims.clone();
    bc_shape[rank - 1] = 1;

    let mean = x_f32.mean_along_dims(&[rank - 1], true)?.reshape(&bc_shape)?;
    let centered = x_f32.sub(&mean)?;
    let var = centered.mul(&centered)?.mean_along_dims(&[rank - 1], true)?.reshape(&bc_shape)?;
    let rstd = var.add_scalar(eps)?.rsqrt()?;
    centered.mul(&rstd)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// Sinusoidal Timestep Embedding
// ---------------------------------------------------------------------------

/// Get sinusoidal timestep embedding, matching diffusers' `get_timestep_embedding`
/// with flip_sin_to_cos=True, downscale_freq_shift=0.
fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let half_dim = dim / 2;
    let device = timesteps.device().clone();

    // Build frequencies: exp(-i * log(10000) / half_dim)
    let mut freq_data = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        freq_data[i] = (-(i as f32) * (10000f32.ln()) / (half_dim as f32)).exp();
    }
    let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half_dim]), device)?;

    // timesteps: [B] -> [B, 1]
    let t = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;

    // [B, 1] * [1, half_dim] -> [B, half_dim]
    let args = t.mul(&freqs)?;

    let cos_part = args.cos()?.to_dtype(DType::BF16)?;
    let sin_part = args.sin()?.to_dtype(DType::BF16)?;

    // flip_sin_to_cos=True -> [cos, sin]
    Tensor::cat(&[&cos_part, &sin_part], 1)
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (3D Video / 1D Audio)
// ---------------------------------------------------------------------------

/// Compute split-RoPE frequencies for video (3D) or audio (1D) coordinates.
///
/// Returns (cos_freqs, sin_freqs) shaped [B, num_heads, N, head_dim/2]
/// suitable for apply_rotary_emb (split variant matching official LTX-2).
///
/// `coords` shape: [B, num_dims, num_patches, 2] where last dim is [start, end)
pub fn compute_rope_frequencies(
    coords: &Tensor,
    dim: usize,
    max_positions: &[f64],
    theta: f64,
    num_heads: usize,
) -> Result<(Tensor, Tensor)> {
    let device = coords.device().clone();
    let cdims = coords.shape().dims().to_vec();
    let batch_size = cdims[0];
    let num_pos_dims = cdims[1];
    let num_patches = cdims[2];

    // Midpoint of [start, end) boundaries
    let coords_f32 = coords.to_dtype(DType::F32)?;
    let starts = coords_f32.narrow(3, 0, 1)?;
    let ends = coords_f32.narrow(3, 1, 1)?;
    let midpoints = starts.add(&ends)?.mul_scalar(0.5)?.squeeze_dim(3)?; // [B, D, P]

    // Fractional positions: grid[d] = midpoints[:, d, :] / max_positions[d]
    let mut grid_expanded = Vec::with_capacity(num_pos_dims);
    for d in 0..num_pos_dims {
        let dim_slice = midpoints.narrow(1, d, 1)?.squeeze_dim(1)?; // [B, P]
        let normed = dim_slice.mul_scalar(1.0 / max_positions[d] as f32)?;
        grid_expanded.push(normed.unsqueeze(2)?); // [B, P, 1]
    }
    let grid = Tensor::cat(&grid_expanded.iter().collect::<Vec<_>>(), 2)?; // [B, P, num_pos_dims]

    // Frequency vector: theta^linspace(0, 1, dim // (num_pos_dims * 2)) * pi / 2
    let num_rope_elems = num_pos_dims * 2;
    let freq_count = dim / num_rope_elems;
    let mut freq_data = Vec::with_capacity(freq_count);
    for i in 0..freq_count {
        let t = i as f64 / (freq_count as f64 - 1.0).max(1.0);
        freq_data.push((theta.powf(t) * PI / 2.0) as f32);
    }
    let freqs_vec = Tensor::from_vec(
        freq_data, Shape::from_dims(&[1, 1, 1, freq_count]), device.clone(),
    )?;

    // angles = (grid * 2 - 1) * freqs → [B, P, num_pos_dims, freq_count]
    let grid_4d = grid.unsqueeze(3)?;
    let scaled = grid_4d.mul_scalar(2.0)?.add_scalar(-1.0)?;
    let angles = scaled.mul(&freqs_vec.expand(&[batch_size, num_patches, num_pos_dims, freq_count])?)?;

    // Transpose dims and flatten: [B, P, freq_count, num_pos_dims] → [B, P, freq_count * num_pos_dims]
    let angles_t = angles.permute(&[0, 1, 3, 2])?;
    let rope_freqs = freq_count * num_pos_dims;
    let angles_flat = angles_t.reshape(&[batch_size, num_patches, rope_freqs])?;

    // Split RoPE: cos/sin without repeat_interleave, pad to dim/2, reshape to [B, H, N, D_head/2]
    let cos_raw = angles_flat.cos()?.to_dtype(DType::BF16)?;
    let sin_raw = angles_flat.sin()?.to_dtype(DType::BF16)?;

    let half_dim = dim / 2;
    let (cos_out, sin_out) = if rope_freqs < half_dim {
        // Pad with ones (cos) / zeros (sin) at the FRONT (matching official)
        let pad_size = half_dim - rope_freqs;
        let cos_pad = Tensor::ones_dtype(
            Shape::from_dims(&[batch_size, num_patches, pad_size]),
            DType::BF16, device.clone(),
        )?;
        let sin_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, num_patches, pad_size]),
            DType::BF16, device.clone(),
        )?;
        (Tensor::cat(&[&cos_pad, &cos_raw], 2)?,
         Tensor::cat(&[&sin_pad, &sin_raw], 2)?)
    } else {
        (cos_raw, sin_raw)
    };

    // Reshape to per-head: [B, N, H, D_head/2] → [B, H, N, D_head/2]
    let head_rope_dim = half_dim / num_heads;
    let cos_heads = cos_out.reshape(&[batch_size, num_patches, num_heads, head_rope_dim])?
        .permute(&[0, 2, 1, 3])?;  // [B, H, N, D_head/2]
    let sin_heads = sin_out.reshape(&[batch_size, num_patches, num_heads, head_rope_dim])?
        .permute(&[0, 2, 1, 3])?;

    Ok((cos_heads, sin_heads))
}

/// Repeat each element along the last dimension `n` times.
/// [B, S, D] -> [B, S, D*n]
fn repeat_interleave_last(x: &Tensor, n: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let rank = dims.len();
    let last = dims[rank - 1];

    // Unsqueeze last dim: [..., D, 1], expand to [..., D, n], flatten
    let expanded = x.unsqueeze(rank)?; // [..., D, 1]
    let mut target_shape = dims.clone();
    target_shape.push(n);
    let expanded = expanded.expand(&target_shape)?;

    // Flatten last two dims
    let mut out_shape = dims[..rank - 1].to_vec();
    out_shape.push(last * n);
    expanded.reshape(&out_shape)
}

/// Apply interleaved rotary embedding to query/key tensor.
/// x: [B, S, inner_dim], freqs: (cos [B, S, inner_dim], sin [B, S, inner_dim])
/// Apply split rotary embeddings (LTX-2 style).
///
/// `x`: [B, H, N, D_head] or [B, N, D] (auto-reshaped if cos is 4D)
/// `cos_freqs`, `sin_freqs`: [B, H, N, D_head/2] (BF16 from `compute_rope_frequencies`)
///
/// Split RoPE splits head_dim into two halves and cross-rotates:
///   first_half_out  = first_half * cos - second_half * sin
///   second_half_out = second_half * cos + first_half * sin
///
/// Stays in BF16 throughout.  The previous version round-tripped through
/// FP32 (4 casts in + 1 cast out + 6 elementwise FP32 ops = ~10 launches and
/// ~10 allocations per call) which was burning 200–400 ms per attention.
/// Cast-free BF16 is ~10× faster on the per-head LTX-2 RoPE shapes; using
/// `flame_core::bf16_ops::rope_halfsplit_bf16` directly would be even faster
/// but its kernel assumes cos/sin are broadcast across heads, which LTX-2
/// does not.
pub fn apply_rotary_emb(x: &Tensor, cos_freqs: &Tensor, sin_freqs: &Tensor) -> Result<Tensor> {
    let x_dims = x.shape().dims().to_vec();
    let cos_dims = cos_freqs.shape().dims().to_vec();

    // If x is 3D [B, N, D] but cos is 4D [B, H, N, D_head/2], reshape x
    let (x4d, needs_reshape) = if x_dims.len() == 3 && cos_dims.len() == 4 {
        let (b, h, t, _half_d) = (cos_dims[0], cos_dims[1], cos_dims[2], cos_dims[3]);
        let d_head = x_dims[2] / h;
        let reshaped = x.reshape(&[b, t, h, d_head])?.permute(&[0, 2, 1, 3])?; // [B, H, N, D_head]
        (reshaped, true)
    } else {
        (x.clone(), false)
    };

    let xd = x4d.shape().dims().to_vec();
    let (b, h, n, d_head) = (xd[0], xd[1], xd[2], xd[3]);
    let half = d_head / 2;

    // RoPE rotation in FP32 for numerical stability (BF16 path produced NaN).
    // The narrow() calls materialize contiguous BF16 halves first; cast each
    // to FP32 just before the rotation math, then cast the result back to BF16.
    let first_half = x4d.narrow(3, 0, half)?;       // [B, H, N, half] BF16
    let second_half = x4d.narrow(3, half, half)?;    // [B, H, N, half] BF16

    let cos_f32 = cos_freqs.to_dtype(DType::F32)?;
    let sin_f32 = sin_freqs.to_dtype(DType::F32)?;
    let first_f32 = first_half.to_dtype(DType::F32)?;
    let second_f32 = second_half.to_dtype(DType::F32)?;

    // Split RoPE rotation:
    //   first_out  = first * cos - second * sin
    //   second_out = second * cos + first * sin
    let first_out = first_f32.mul(&cos_f32)?.sub(&second_f32.mul(&sin_f32)?)?;
    let second_out = second_f32.mul(&cos_f32)?.add(&first_f32.mul(&sin_f32)?)?;

    // Concatenate halves back and cast: [B, H, N, D_head] BF16
    let out = Tensor::cat(&[&first_out, &second_out], 3)?.to_dtype(DType::BF16)?;

    if needs_reshape {
        // [B, H, N, D_head] → [B, N, H*D_head]
        let (b, h, n, d) = (xd[0], xd[1], xd[2], xd[3]);
        out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])
    } else {
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Sub-modules
// ---------------------------------------------------------------------------

/// Timestep embedding MLP: sinusoidal -> linear -> SiLU -> linear
pub struct TimestepEmbedderMLP {
    pub linear_1_weight: Tensor,
    pub linear_1_bias: Tensor,
    pub linear_2_weight: Tensor,
    pub linear_2_bias: Tensor,
}

impl TimestepEmbedderMLP {
    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        // Sinusoidal embedding (256 channels)
        let emb = timestep_embedding(timesteps, 256)?;

        // MLP: linear -> silu -> linear
        let h = linear3d(&emb, &self.linear_1_weight, Some(&self.linear_1_bias))?;
        let h = silu(&h)?;
        linear3d(&h, &self.linear_2_weight, Some(&self.linear_2_bias))
    }
}

/// AdaLN-Single: timestep_embedding -> silu -> linear (produces num_mod_params * dim modulation values)
pub struct AdaLayerNormSingle {
    /// TimestepEmbedderMLP inside `emb`
    pub emb: TimestepEmbedderMLP,
    /// Modulation projection: linear(dim -> num_mod_params * dim)
    pub linear_weight: Tensor,
    pub linear_bias: Tensor,
    pub num_mod_params: usize,
}

impl AdaLayerNormSingle {
    /// Forward: returns (mod_params, embedded_timestep)
    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded = self.emb.forward(timestep)?;
        let h = silu(&embedded)?;
        let mod_params = linear3d(&h, &self.linear_weight, Some(&self.linear_bias))?;
        Ok((mod_params, embedded))
    }
}

/// Caption projection (PixArtAlphaTextProjection): linear -> GELU(tanh) -> linear
pub struct CaptionProjection {
    pub linear_1_weight: Tensor,
    pub linear_1_bias: Tensor,
    pub linear_2_weight: Tensor,
    pub linear_2_bias: Tensor,
}

impl CaptionProjection {
    fn forward(&self, caption: &Tensor) -> Result<Tensor> {
        let h = linear3d(caption, &self.linear_1_weight, Some(&self.linear_1_bias))?;
        let h = gelu_approximate(&h)?;
        linear3d(&h, &self.linear_2_weight, Some(&self.linear_2_bias))
    }
}

/// GELU-approximate FeedForward: GELU(linear(x)) -> linear -> output
pub struct FeedForward {
    /// net.0.proj: linear -> gelu
    pub gelu_proj_weight: Tensor,
    pub gelu_proj_bias: Tensor,
    /// net.2: output projection
    pub out_weight: Tensor,
    pub out_bias: Tensor,
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = linear3d(x, &self.gelu_proj_weight, Some(&self.gelu_proj_bias))?;
        let h = gelu_approximate(&h)?;
        linear3d(&h, &self.out_weight, Some(&self.out_bias))
    }
}

/// LTX2 Attention block: Q/K/V projection, RMSNorm on Q/K, optional RoPE, SDPA, output projection.
pub struct LTX2Attention {
    pub to_q_weight: Tensor,
    pub to_q_bias: Tensor,
    pub to_k_weight: Tensor,
    pub to_k_bias: Tensor,
    pub to_v_weight: Tensor,
    pub to_v_bias: Tensor,
    pub norm_q_weight: Tensor,  // RMSNorm across heads
    pub norm_k_weight: Tensor,
    pub to_out_weight: Tensor,  // to_out.0
    pub to_out_bias: Tensor,
    // Optional per-head gating (ComfyUI LTX-2.3 checkpoint)
    pub to_gate_logits_weight: Option<Tensor>,  // [num_heads, query_dim]
    pub to_gate_logits_bias: Option<Tensor>,    // [num_heads]
    pub num_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
}

impl LTX2Attention {
    /// Forward with optional separate query/key RoPE.
    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        query_rope: Option<(&Tensor, &Tensor)>,
        key_rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let prof = std::env::var("LTX2_ATTN_PROF").is_ok();
        let dev = hidden_states.device().clone();
        let mut t = std::time::Instant::now();
        let mut tick = |name: &'static str, t: &mut std::time::Instant, out: &mut Vec<(&'static str, u128)>| {
            if prof { let _ = dev.synchronize(); }
            out.push((name, t.elapsed().as_millis()));
            *t = std::time::Instant::now();
        };
        let mut phases: Vec<(&'static str, u128)> = Vec::with_capacity(8);

        let kv_input = encoder_hidden_states.unwrap_or(hidden_states);

        let q = linear3d(hidden_states, &self.to_q_weight, Some(&self.to_q_bias))?;
        let k = linear3d(kv_input, &self.to_k_weight, Some(&self.to_k_bias))?;
        let v = linear3d(kv_input, &self.to_v_weight, Some(&self.to_v_bias))?;
        tick("qkv", &mut t, &mut phases);

        // Fused RMSNorm across heads on Q and K
        let q = rms_norm(&q, Some(&self.norm_q_weight), self.eps)?;
        let k = rms_norm(&k, Some(&self.norm_k_weight), self.eps)?;
        tick("qknorm", &mut t, &mut phases);

        // Apply rotary embeddings
        let q = if let Some((cos, sin)) = query_rope {
            apply_rotary_emb(&q, cos, sin)?
        } else {
            q
        };
        let k = if let Some((cos, sin)) = key_rope.or(query_rope) {
            apply_rotary_emb(&k, cos, sin)?
        } else {
            k
        };
        tick("rope", &mut t, &mut phases);

        // Reshape to [B, num_heads, S, head_dim] for SDPA
        let q_dims = q.shape().dims().to_vec();
        let (b, s_q) = (q_dims[0], q_dims[1]);
        let k_dims = k.shape().dims().to_vec();
        let s_kv = k_dims[1];

        let q = q.reshape(&[b, s_q, self.num_heads, self.head_dim])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, s_kv, self.num_heads, self.head_dim])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, s_kv, self.num_heads, self.head_dim])?.permute(&[0, 2, 1, 3])?;
        tick("permute_in", &mut t, &mut phases);

        // Scaled dot-product attention
        let attn_out = flame_core::sdpa::forward(&q, &k, &v, attention_mask)?;
        tick("sdpa", &mut t, &mut phases);

        // Reshape back: [B, heads, S, head_dim] -> [B, S, inner_dim]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?; // [B, S, H, D]
        let inner_dim = self.num_heads * self.head_dim;

        // Apply per-head gating if present (ComfyUI LTX-2.3)
        let attn_out = if let (Some(gate_w), Some(gate_b)) =
            (&self.to_gate_logits_weight, &self.to_gate_logits_bias)
        {
            // gate_logits = hidden_states @ gate_w.T + gate_b → [B, S, num_heads]
            let gate_logits = linear3d(hidden_states, gate_w, Some(gate_b))?;
            // gates = 2.0 * sigmoid(gate_logits) → [B, S, num_heads]
            let gates = gate_logits.sigmoid()?.mul_scalar(2.0)?;
            // gates: [B, S, H] → [B, S, H, 1] for broadcast with attn_out [B, S, H, D]
            let gates = gates.unsqueeze(3)?;
            attn_out.mul(&gates)?.reshape(&[b, s_q, inner_dim])?
        } else {
            attn_out.reshape(&[b, s_q, inner_dim])?
        };

        // Output projection
        let out = linear3d(&attn_out, &self.to_out_weight, Some(&self.to_out_bias))?;
        tick("post", &mut t, &mut phases);
        if prof {
            let parts: Vec<String> = phases.iter().map(|(n, ms)| format!("{}={}", n, ms)).collect();
            log::info!("[ATTN-PROF d={} sq={}] {}", self.head_dim, s_q, parts.join(" "));
        }
        Ok(out)
    }
}

/// A single LTX2 transformer block: dual-stream (video + audio) with
/// self-attention, cross-attention, a2v/v2a cross-attention, and FFN.
pub struct LTX2TransformerBlock {
    // Video self-attention
    pub norm1_weight: Option<Tensor>,  // RMSNorm (may be None if no elementwise_affine)
    pub attn1: LTX2Attention,

    // Audio self-attention
    pub audio_norm1_weight: Option<Tensor>,
    pub audio_attn1: LTX2Attention,

    // Video cross-attention (with text)
    pub norm2_weight: Option<Tensor>,
    pub attn2: LTX2Attention,

    // Audio cross-attention (with text)
    pub audio_norm2_weight: Option<Tensor>,
    pub audio_attn2: LTX2Attention,

    // Audio-to-Video cross-attention: Q=video, KV=audio
    pub audio_to_video_norm_weight: Option<Tensor>,
    pub audio_to_video_attn: LTX2Attention,

    // Video-to-Audio cross-attention: Q=audio, KV=video
    pub video_to_audio_norm_weight: Option<Tensor>,
    pub video_to_audio_attn: LTX2Attention,

    // Video FFN
    pub norm3_weight: Option<Tensor>,
    pub ff: FeedForward,

    // Audio FFN
    pub audio_norm3_weight: Option<Tensor>,
    pub audio_ff: FeedForward,

    // AdaLN-Zero modulation tables
    pub scale_shift_table: Tensor,                    // [6, dim] or [9, dim] for video
    pub audio_scale_shift_table: Tensor,              // [6, audio_dim]
    pub video_a2v_cross_attn_scale_shift_table: Tensor, // [5, dim]
    pub audio_a2v_cross_attn_scale_shift_table: Tensor, // [5, audio_dim]

    // Cross-attention adaln modulation (ComfyUI LTX-2.3)
    pub prompt_scale_shift_table: Option<Tensor>,     // [2, dim] for video CA context KV modulation
    pub audio_prompt_scale_shift_table: Option<Tensor>, // [2, audio_dim] for audio CA context KV modulation

    pub eps: f32,
}

impl LTX2TransformerBlock {
    /// Video-only forward: self-attn -> cross-attn (text) -> FFN.
    /// Skips all audio paths, A2V/V2A cross-attention.
    ///
    /// `prompt_timestep`: Optional [B, seq, 2*dim] from prompt_adaln_single (ComfyUI 9-param path).
    pub fn forward_video_only(
        &self,
        hidden_states: &Tensor,        // [B, N, dim]
        encoder_hidden_states: &Tensor, // [B, seq, dim]
        temb: &Tensor,                 // [B, N, num_params*dim]
        video_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
        prompt_timestep: Option<&Tensor>, // [B, seq, 2*dim] for cross-attn KV modulation
    ) -> Result<Tensor> {
        let b = hidden_states.shape().dims()[0];
        let dim = hidden_states.shape().dims()[2];
        let num_ada_params = self.scale_shift_table.shape().dims()[0];

        // Extract first 6 AdaLN-Zero params: [shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff]
        let t0 = std::time::Instant::now();
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.compute_ada_params_6(&self.scale_shift_table, temb, b, dim)?;

        // 1. Self-Attention with AdaLN-Zero (fused: rms_norm+modulate → 1 kernel, residual+gate → 1 kernel)
        let mod_h = if let Some(w) = self.norm1_weight.as_ref() {
            fused_rms_norm_modulate(hidden_states, w, &scale_msa, &shift_msa, self.eps)?
        } else {
            let norm_h = rms_norm(hidden_states, None, self.eps)?;
            fused_modulate(&norm_h, &scale_msa, &shift_msa)?
        };
        let t_adaln = t0.elapsed().as_millis();
        let attn_out = self.attn1.forward(&mod_h, None, None, video_rotary_emb, None)?;
        let t_sa = t0.elapsed().as_millis() - t_adaln;
        let mut hs = fused_residual_gate(hidden_states, &attn_out, &gate_msa)?;

        // 2. Cross-Attention (text) — with or without adaln modulation
        if num_ada_params >= 9 {
            // 9-param path: indices 6-8 from scale_shift_table + temb for query modulation
            let (shift_ca_q, scale_ca_q, gate_ca) =
                self.compute_ada_params_ca(&self.scale_shift_table, temb, b, dim)?;

            // Modulate query: fused rms_norm + modulate → 1 kernel
            let attn_input = if let Some(w) = self.norm2_weight.as_ref() {
                fused_rms_norm_modulate(&hs, w, &scale_ca_q, &shift_ca_q, self.eps)?
            } else {
                let norm_h2 = rms_norm(&hs, None, self.eps)?;
                fused_modulate(&norm_h2, &scale_ca_q, &shift_ca_q)?
            };

            // Modulate context (KV) using prompt_scale_shift_table + prompt_timestep
            let modulated_context = if let (Some(psst), Some(pt)) =
                (&self.prompt_scale_shift_table, prompt_timestep)
            {
                // psst: [2, dim] → [1, 1, 2, dim]
                let psst_bc = psst.unsqueeze(0)?.unsqueeze(0)?;
                // pt: [B, seq, 2*dim] → [B, seq, 2, dim]
                let pt_dims = pt.shape().dims().to_vec();
                let seq_len = pt_dims[1];
                let pt_4d = pt.reshape(&[b, seq_len, 2, dim])?;
                // Combined: [B, seq, 2, dim]
                let combined = psst_bc.add(&pt_4d)?.to_dtype(DType::BF16)?;
                let shift_kv = combined.narrow(2, 0, 1)?.squeeze_dim(2)?; // [B, seq, dim]
                let scale_kv = combined.narrow(2, 1, 1)?.squeeze_dim(2)?; // [B, seq, dim]
                // context * (1 + scale_kv) + shift_kv — fused modulate
                fused_modulate(encoder_hidden_states, &scale_kv, &shift_kv)?
            } else {
                encoder_hidden_states.clone()
            };

            let ca_out = self.attn2.forward(
                &attn_input, Some(&modulated_context), encoder_attention_mask, None, None,
            )?;
            hs = fused_residual_gate(&hs, &ca_out, &gate_ca)?;
        } else {
            // Legacy 6-param path: no cross-attn modulation
            let norm_h2 = rms_norm(&hs, self.norm2_weight.as_ref(), self.eps)?;
            let ca_out = self.attn2.forward(
                &norm_h2, Some(encoder_hidden_states), encoder_attention_mask, None, None,
            )?;
            hs = hs.add(&ca_out)?;
        }

        let t_ca = t0.elapsed().as_millis() - t_adaln - t_sa;

        // 3. FeedForward with AdaLN-Zero (fused: rms_norm+modulate → 1 kernel, residual+gate → 1 kernel)
        let mod_ff = if let Some(w) = self.norm3_weight.as_ref() {
            fused_rms_norm_modulate(&hs, w, &scale_mlp, &shift_mlp, self.eps)?
        } else {
            let norm_ff = rms_norm(&hs, None, self.eps)?;
            fused_modulate(&norm_ff, &scale_mlp, &shift_mlp)?
        };
        let ff_out = self.ff.forward(&mod_ff)?;
        hs = fused_residual_gate(&hs, &ff_out, &gate_mlp)?;
        let t_ff = t0.elapsed().as_millis() - t_adaln - t_sa - t_ca;
        log::info!("[PERF] adaln={}ms sa={}ms ca={}ms ff={}ms total={}ms",
            t_adaln, t_sa, t_ca, t_ff, t0.elapsed().as_millis());

        Ok(hs)
    }

    /// Forward pass for one block. Returns (video_hidden, audio_hidden).
    ///
    /// `video_prompt_timestep` / `audio_prompt_timestep`: optional `[B, seq, 2*dim]`
    /// tensors produced by `prompt_adaln_single` / `audio_prompt_adaln_single`.
    /// When present (and the matching `*_prompt_scale_shift_table` is loaded),
    /// the cross-attention KV (text encoder hidden states) is modulated as
    /// `context * (1 + scale_kv) + shift_kv` BEFORE the attention call. This
    /// matches the Lightricks reference (transformer.py:380-398) and fixes the
    /// "dark output / std=0.33" bug documented in LTX2_AV_BUGS.md.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        audio_hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: &Tensor,
        temb: &Tensor,                    // [B, 1, 6*dim]
        temb_audio: &Tensor,              // [B, 1, 6*audio_dim]
        temb_ca_scale_shift: &Tensor,     // [B, 1, 4*dim]
        temb_ca_audio_scale_shift: &Tensor, // [B, 1, 4*audio_dim]
        temb_ca_gate: &Tensor,            // [B, 1, dim]
        temb_ca_audio_gate: &Tensor,      // [B, 1, audio_dim]
        video_rotary_emb: Option<(&Tensor, &Tensor)>,
        audio_rotary_emb: Option<(&Tensor, &Tensor)>,
        ca_video_rotary_emb: Option<(&Tensor, &Tensor)>,
        ca_audio_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
        video_prompt_timestep: Option<&Tensor>, // [B, text_seq, 2*video_dim]
        audio_prompt_timestep: Option<&Tensor>, // [B, audio_text_seq, 2*audio_dim]
    ) -> Result<(Tensor, Tensor)> {
        let b = hidden_states.shape().dims()[0];
        let video_dim = hidden_states.shape().dims()[2];
        let audio_dim = audio_hidden_states.shape().dims()[2];

        // Profiling — gated by LTX2_FWD_PROF env var. When set, syncs the
        // device after each phase and stores per-phase millis. Logged at the
        // end of the function.  Used to identify the slow phase inside the
        // 6-second-per-block AV forward.
        let prof = std::env::var("LTX2_FWD_PROF").is_ok();
        let dev = hidden_states.device().clone();
        let mut t = std::time::Instant::now();
        let mut take = |label: &mut Vec<(&'static str, u128)>, name: &'static str, start: &mut std::time::Instant| {
            if prof { let _ = dev.synchronize(); }
            let ms = start.elapsed().as_millis();
            label.push((name, ms));
            *start = std::time::Instant::now();
        };
        let mut phases: Vec<(&'static str, u128)> = Vec::with_capacity(8);

        // Tensor dump scaffolding for python_block0_forward.py diff. Fires once
        // per process and only when LTX2_DUMP_BLOCK0=1. Outputs go to
        // rust_block0_intermediates.safetensors. Each save is gated by a static
        // OnceLock so subsequent block forwards don't overwrite.
        let dump_intermediates = std::env::var("LTX2_DUMP_BLOCK0").is_ok();
        let mut intermediate_dump: HashMap<String, Tensor> = HashMap::new();

        if dump_intermediates {
            intermediate_dump.insert("dump_input_hs".into(), hidden_states.clone());
            intermediate_dump.insert("dump_input_ahs".into(), audio_hidden_states.clone());
        }

        // ---- 1. Video Self-Attention with AdaLN-Zero ----
        // Compute 6 modulation params from scale_shift_table + temb
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.compute_ada_params_6(&self.scale_shift_table, temb, b, video_dim)?;

        if dump_intermediates {
            intermediate_dump.insert("dump_shift_msa".into(), shift_msa.clone());
            intermediate_dump.insert("dump_scale_msa".into(), scale_msa.clone());
            intermediate_dump.insert("dump_gate_msa".into(), gate_msa.clone());
        }

        // Fused: rms_norm + (1+scale)*x + shift in one kernel.  Falls back
        // to separate calls when no affine weight is present.
        let mod_h = if let Some(w) = self.norm1_weight.as_ref() {
            fused_rms_norm_modulate(hidden_states, w, &scale_msa, &shift_msa, self.eps)?
        } else {
            let norm_h = rms_norm(hidden_states, None, self.eps)?;
            fused_modulate(&norm_h, &scale_msa, &shift_msa)?
        };
        if dump_intermediates {
            intermediate_dump.insert("dump_mod_h".into(), mod_h.clone());
        }
        let attn_out = self.attn1.forward(&mod_h, None, None, video_rotary_emb, None)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_vsa_attn_out".into(), attn_out.clone());
        }
        // Fused: x + attn_out * gate in one kernel.
        let mut hidden_states = fused_residual_gate(hidden_states, &attn_out, &gate_msa)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_vsa".into(), hidden_states.clone());
        }
        take(&mut phases, "vsa", &mut t);

        // ---- Audio Self-Attention with AdaLN-Zero ----
        let (a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp) =
            self.compute_ada_params_6(&self.audio_scale_shift_table, temb_audio, b, audio_dim)?;
        let mod_a = if let Some(w) = self.audio_norm1_weight.as_ref() {
            fused_rms_norm_modulate(audio_hidden_states, w, &a_scale_msa, &a_shift_msa, self.eps)?
        } else {
            let norm_a = rms_norm(audio_hidden_states, None, self.eps)?;
            fused_modulate(&norm_a, &a_scale_msa, &a_shift_msa)?
        };
        let attn_a_out = self.audio_attn1.forward(&mod_a, None, None, audio_rotary_emb, None)?;
        let mut audio_hidden_states = fused_residual_gate(audio_hidden_states, &attn_a_out, &a_gate_msa)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_asa".into(), audio_hidden_states.clone());
        }
        take(&mut phases, "asa", &mut t);

        // ---- 2. Video/Audio Cross-Attention with text (AdaLN modulated) ----
        // Extract cross-attention modulation params (indices 6-8 of [9, dim] table)
        let (v_shift_ca, v_scale_ca, v_gate_ca) =
            self.compute_ada_params_ca(&self.scale_shift_table, temb, b, video_dim)?;
        let mod_h2 = if let Some(w) = self.norm2_weight.as_ref() {
            fused_rms_norm_modulate(&hidden_states, w, &v_scale_ca, &v_shift_ca, self.eps)?
        } else {
            let norm_h2 = rms_norm(&hidden_states, None, self.eps)?;
            fused_modulate(&norm_h2, &v_scale_ca, &v_shift_ca)?
        };

        // KV (text) context modulation — see LTX2_AV_BUGS.md Bug 2.
        // Mirrors `apply_cross_attention_adaln` in the Lightricks reference.
        let modulated_v_context = if let (Some(psst), Some(pt)) =
            (&self.prompt_scale_shift_table, video_prompt_timestep)
        {
            // psst: [2, video_dim] → [1, 1, 2, video_dim]
            let psst_bc = psst.unsqueeze(0)?.unsqueeze(0)?;
            // pt:   [B, seq, 2*video_dim] → [B, seq, 2, video_dim]
            let seq_len = pt.shape().dims()[1];
            let pt_4d = pt.reshape(&[b, seq_len, 2, video_dim])?;
            let combined = psst_bc.add(&pt_4d)?.to_dtype(DType::BF16)?;
            let shift_kv = combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
            let scale_kv = combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
            // context * (1 + scale_kv) + shift_kv
            fused_modulate(encoder_hidden_states, &scale_kv, &shift_kv)?
        } else {
            encoder_hidden_states.clone()
        };

        let ca_out = self.attn2.forward(&mod_h2, Some(&modulated_v_context), encoder_attention_mask, None, None)?;
        hidden_states = fused_residual_gate(&hidden_states, &ca_out, &v_gate_ca)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_vca".into(), hidden_states.clone());
        }
        drop(modulated_v_context);
        take(&mut phases, "vca", &mut t);

        let (a_shift_ca, a_scale_ca, a_gate_ca) =
            self.compute_ada_params_ca(&self.audio_scale_shift_table, temb_audio, b, audio_dim)?;
        let mod_a2 = if let Some(w) = self.audio_norm2_weight.as_ref() {
            fused_rms_norm_modulate(&audio_hidden_states, w, &a_scale_ca, &a_shift_ca, self.eps)?
        } else {
            let norm_a2 = rms_norm(&audio_hidden_states, None, self.eps)?;
            fused_modulate(&norm_a2, &a_scale_ca, &a_shift_ca)?
        };

        // Same KV modulation for the audio cross-attention.
        let modulated_a_context = if let (Some(apsst), Some(apt)) =
            (&self.audio_prompt_scale_shift_table, audio_prompt_timestep)
        {
            let apsst_bc = apsst.unsqueeze(0)?.unsqueeze(0)?;
            let seq_len = apt.shape().dims()[1];
            let apt_4d = apt.reshape(&[b, seq_len, 2, audio_dim])?;
            let combined = apsst_bc.add(&apt_4d)?.to_dtype(DType::BF16)?;
            let shift_kv = combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
            let scale_kv = combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
            fused_modulate(audio_encoder_hidden_states, &scale_kv, &shift_kv)?
        } else {
            audio_encoder_hidden_states.clone()
        };

        let ca_a_out = self.audio_attn2.forward(&mod_a2, Some(&modulated_a_context), audio_encoder_attention_mask, None, None)?;
        audio_hidden_states = fused_residual_gate(&audio_hidden_states, &ca_a_out, &a_gate_ca)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_aca".into(), audio_hidden_states.clone());
        }
        drop(modulated_a_context);
        take(&mut phases, "aca", &mut t);


        // ---- 3. Audio-to-Video / Video-to-Audio Cross-Attention ----
        // The norm is shared between A2V and V2A on each side; modulation
        // happens via fused_modulate per branch.
        let norm_a2v = rms_norm(&hidden_states, self.audio_to_video_norm_weight.as_ref(), self.eps)?;
        let norm_v2a = rms_norm(&audio_hidden_states, self.video_to_audio_norm_weight.as_ref(), self.eps)?;
        take(&mut phases, "av_norm", &mut t);

        // Compute per-layer cross-attention modulation
        let (a2v_gate, v2a_gate, video_a2v_mod, video_v2a_mod, audio_a2v_mod, audio_v2a_mod) =
            self.compute_cross_attn_params(
                temb_ca_scale_shift, temb_ca_audio_scale_shift,
                temb_ca_gate, temb_ca_audio_gate,
                b, video_dim, audio_dim,
            )?;
        take(&mut phases, "av_params", &mut t);

        // A2V: Q=video (modulated by video_a2v scale/shift), KV=audio (modulated by audio_a2v).
        //
        // NOTE: video_a2v_mod / audio_a2v_mod scale/shift have shape `[B, 1, dim]`
        // (broadcast across the token dimension), so we CANNOT use `fused_modulate`
        // here — that kernel is plain elementwise and doesn't broadcast, it would
        // read garbage past the end of scale/shift and produce NaN.  Use the
        // broadcast-aware manual `mul + add` path.
        let mod_video_a2v = norm_a2v
            .mul(&video_a2v_mod.0.add_scalar(1.0)?)?
            .add(&video_a2v_mod.1)?;
        let mod_audio_a2v = norm_v2a
            .mul(&audio_a2v_mod.0.add_scalar(1.0)?)?
            .add(&audio_a2v_mod.1)?;
        take(&mut phases, "a2v_mod", &mut t);
        let a2v_out = self.audio_to_video_attn.forward(
            &mod_video_a2v, Some(&mod_audio_a2v), None,
            ca_video_rotary_emb, ca_audio_rotary_emb,
        )?;
        take(&mut phases, "a2v_attn", &mut t);
        // a2v_gate is [B, 1, dim] broadcast — fused_residual_gate doesn't
        // broadcast, so use manual mul+add (same as A2V/V2A modulation above).
        hidden_states = hidden_states.add(&a2v_out.mul(&a2v_gate)?)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_a2v".into(), hidden_states.clone());
        }
        drop(mod_video_a2v);
        drop(mod_audio_a2v);

        // V2A: Q=audio (modulated by audio_v2a), KV=video (modulated by video_v2a)
        let mod_video_v2a = norm_a2v
            .mul(&video_v2a_mod.0.add_scalar(1.0)?)?
            .add(&video_v2a_mod.1)?;
        let mod_audio_v2a = norm_v2a
            .mul(&audio_v2a_mod.0.add_scalar(1.0)?)?
            .add(&audio_v2a_mod.1)?;
        take(&mut phases, "v2a_mod", &mut t);
        let v2a_out = self.video_to_audio_attn.forward(
            &mod_audio_v2a, Some(&mod_video_v2a), None,
            ca_audio_rotary_emb, ca_video_rotary_emb,
        )?;
        take(&mut phases, "v2a_attn", &mut t);
        // v2a_gate is [B, 1, dim] broadcast — manual residual.
        audio_hidden_states = audio_hidden_states.add(&v2a_out.mul(&v2a_gate)?)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_v2a".into(), audio_hidden_states.clone());
        }
        take(&mut phases, "av_tail", &mut t);

        // ---- 4. FeedForward ----
        let mod_ff = if let Some(w) = self.norm3_weight.as_ref() {
            fused_rms_norm_modulate(&hidden_states, w, &scale_mlp, &shift_mlp, self.eps)?
        } else {
            let norm_ff = rms_norm(&hidden_states, None, self.eps)?;
            fused_modulate(&norm_ff, &scale_mlp, &shift_mlp)?
        };
        let ff_out = self.ff.forward(&mod_ff)?;
        hidden_states = fused_residual_gate(&hidden_states, &ff_out, &gate_mlp)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_vff".into(), hidden_states.clone());
        }
        take(&mut phases, "vff", &mut t);

        let mod_aff = if let Some(w) = self.audio_norm3_weight.as_ref() {
            fused_rms_norm_modulate(&audio_hidden_states, w, &a_scale_mlp, &a_shift_mlp, self.eps)?
        } else {
            let norm_aff = rms_norm(&audio_hidden_states, None, self.eps)?;
            fused_modulate(&norm_aff, &a_scale_mlp, &a_shift_mlp)?
        };
        let aff_out = self.audio_ff.forward(&mod_aff)?;
        audio_hidden_states = fused_residual_gate(&audio_hidden_states, &aff_out, &a_gate_mlp)?;
        if dump_intermediates {
            intermediate_dump.insert("dump_after_aff".into(), audio_hidden_states.clone());
        }
        take(&mut phases, "aff", &mut t);

        if prof {
            let parts: Vec<String> = phases.iter().map(|(n, ms)| format!("{}={}ms", n, ms)).collect();
            log::info!("[FWD-PROF] {}", parts.join(" "));
        }

        if dump_intermediates && !intermediate_dump.is_empty() {
            // Save once per process. The OnceLock ensures only the first call
            // (block 0 of step 0) writes the file.
            static SAVED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if SAVED.set(()).is_ok() {
                let _ = dev.synchronize();
                let _ = flame_core::serialization::save_tensors(
                    &intermediate_dump,
                    std::path::Path::new("/home/alex/EriDiffusion/inference-flame/output/rust_block0_intermediates.safetensors"),
                    flame_core::serialization::SerializationFormat::SafeTensors,
                );
                log::info!("[LTX2 DUMP] saved {} intermediate tensors", intermediate_dump.len());
            }
        }

        Ok((hidden_states, audio_hidden_states))
    }

    /// Compute first 6 AdaLN-Zero modulation parameters from scale_shift_table + temb.
    /// Works with both [6, dim] and [9, dim] tables (only uses indices 0-5).
    /// Returns (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
    ///
    /// Performance: the scale_shift_table is loaded as F32 (small, ~150 KB) by
    /// the F32 cache.  The OLD path did `(F32 table + BF16 temb) -> F32 ada
    /// [B, N, 6, dim] (~128 MB) -> to_dtype(BF16) -> [B, N, 6, dim] BF16
    /// (~64 MB)`, which is ~190 MB allocated per call * 4 calls/block * 48
    /// blocks * 8 steps ≈ 290 GB of cudaMallocAsync churn per stage 1.
    /// Per-call cost: ~250–500 ms because the cudarc mempool can't recycle
    /// fast enough on a 24 GB card.
    ///
    /// New path: cast the small table to BF16 (one tiny alloc), then do the
    /// add in BF16.  Single ~64 MB BF16 allocation per call instead of two
    /// large allocs and a round-trip cast.  ~10× faster on the same shapes.
    fn compute_ada_params_6(
        &self,
        table: &Tensor,  // [6+, dim] possibly F32 from f32_cache
        temb: &Tensor,   // [B, N, num_params*dim] BF16
        batch_size: usize,
        dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let num_tokens = temb.shape().dims()[1];

        // Cast table to BF16 if needed (small alloc on a [9, dim] tensor, cheap).
        let table_bf16;
        let table_ref = if table.dtype() == DType::BF16 {
            table
        } else {
            table_bf16 = table.to_dtype(DType::BF16)?;
            &table_bf16
        };

        // Extract first 6 rows from table
        let table_6 = table_ref.narrow(0, 0, 6)?; // [6, dim]
        let table_bc = table_6.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 6, dim]

        // Extract first 6*dim from temb
        let temb_6 = temb.narrow(2, 0, 6 * dim)?; // [B, N, 6*dim]
        let temb_4d = temb_6.reshape(&[batch_size, num_tokens, 6, dim])?;
        // Single BF16 alloc, no F32 round-trip.
        let ada = table_bc.add(&temb_4d)?; // [B, N, 6, dim] BF16

        let shift_msa = ada.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale_msa = ada.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let gate_msa = ada.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let shift_mlp = ada.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let scale_mlp = ada.narrow(2, 4, 1)?.squeeze_dim(2)?;
        let gate_mlp = ada.narrow(2, 5, 1)?.squeeze_dim(2)?;

        Ok((shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }

    /// Compute cross-attention adaln params from indices 6-8 of a [9, dim] table.
    /// Returns (shift_ca_q, scale_ca_q, gate_ca).
    /// Same BF16-throughout treatment as `compute_ada_params_6`.
    fn compute_ada_params_ca(
        &self,
        table: &Tensor,  // [9, dim] possibly F32
        temb: &Tensor,   // [B, N, 9*dim] BF16
        batch_size: usize,
        dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let num_tokens = temb.shape().dims()[1];

        let table_bf16;
        let table_ref = if table.dtype() == DType::BF16 {
            table
        } else {
            table_bf16 = table.to_dtype(DType::BF16)?;
            &table_bf16
        };

        // Extract rows 6-8 from table
        let table_ca = table_ref.narrow(0, 6, 3)?; // [3, dim]
        let table_bc = table_ca.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 3, dim]

        // Extract last 3*dim from temb
        let temb_ca = temb.narrow(2, 6 * dim, 3 * dim)?; // [B, N, 3*dim]
        let temb_4d = temb_ca.reshape(&[batch_size, num_tokens, 3, dim])?;
        let ada = table_bc.add(&temb_4d)?; // [B, N, 3, dim] BF16

        let shift_ca_q = ada.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale_ca_q = ada.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let gate_ca = ada.narrow(2, 2, 1)?.squeeze_dim(2)?;

        Ok((shift_ca_q, scale_ca_q, gate_ca))
    }

    /// Compute cross-attention modulation parameters from per-layer tables + global temb.
    /// Returns (a2v_gate, v2a_gate, (video_a2v_scale, video_a2v_shift),
    ///          (video_v2a_scale, video_v2a_shift), (audio_a2v_scale, audio_a2v_shift),
    ///          (audio_v2a_scale, audio_v2a_shift))
    #[allow(clippy::type_complexity)]
    fn compute_cross_attn_params(
        &self,
        temb_ca_ss: &Tensor,           // [B, 1, 4*video_dim]
        temb_ca_audio_ss: &Tensor,     // [B, 1, 4*audio_dim]
        temb_ca_gate: &Tensor,         // [B, 1, video_dim]
        temb_ca_audio_gate: &Tensor,   // [B, 1, audio_dim]
        b: usize,
        video_dim: usize,
        audio_dim: usize,
    ) -> Result<(
        Tensor,                  // a2v_gate
        Tensor,                  // v2a_gate
        (Tensor, Tensor),       // (video_a2v_scale, video_a2v_shift)
        (Tensor, Tensor),       // (video_v2a_scale, video_v2a_shift)
        (Tensor, Tensor),       // (audio_a2v_scale, audio_a2v_shift)
        (Tensor, Tensor),       // (audio_v2a_scale, audio_v2a_shift)
    )> {
        // Video per-layer: first 4 rows = scale/shift, 5th row = gate contribution.
        //
        // Performance: BOTH operands of `add` MUST be the same rank. The
        // OLD path mixed 3D `[1, 4, dim]` (from `unsqueeze(0)` on `[4, dim]`)
        // with 4D `[B, 1, 4, dim]` (from temb reshape).  flame-core's
        // mixed-rank broadcast falls back to a slow generic path that took
        // ~600 ms per add * 4 adds = ~2.5 SECONDS per call * 48 blocks * 8
        // steps ≈ 16 minutes per stage 1.  Aligning to 4D up front routes
        // through the fast same-rank broadcast.
        //
        // Also pre-cast tables to BF16 once (small allocs, ~1 KB each).
        let v_full_table = if self.video_a2v_cross_attn_scale_shift_table.dtype() == DType::BF16 {
            self.video_a2v_cross_attn_scale_shift_table.clone()
        } else {
            self.video_a2v_cross_attn_scale_shift_table.to_dtype(DType::BF16)?
        };
        let a_full_table = if self.audio_a2v_cross_attn_scale_shift_table.dtype() == DType::BF16 {
            self.audio_a2v_cross_attn_scale_shift_table.clone()
        } else {
            self.audio_a2v_cross_attn_scale_shift_table.to_dtype(DType::BF16)?
        };

        let v_table_ss = v_full_table.narrow(0, 0, 4)?; // [4, dim]
        let v_table_gate = v_full_table.narrow(0, 4, 1)?; // [1, dim]
        let a_table_ss = a_full_table.narrow(0, 0, 4)?;
        let a_table_gate = a_full_table.narrow(0, 4, 1)?;

        // Combine global + per-layer video cross-attn scale/shift.  Rank-aligned.
        let v_ss_bc = v_table_ss.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 4, dim] (4D)
        let temb_ss_4d = temb_ca_ss.reshape(&[b, 1, 4, video_dim])?; // [B, 1, 4, dim] (4D)
        let v_combined = v_ss_bc.add(&temb_ss_4d)?;

        let video_a2v_scale = v_combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let video_a2v_shift = v_combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let video_v2a_scale = v_combined.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let video_v2a_shift = v_combined.narrow(2, 3, 1)?.squeeze_dim(2)?;

        // a2v gate: per_layer + global.  Same rank-alignment fix.
        let v_gate_bc = v_table_gate.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 1, dim] (4D)
        let temb_gate_4d = temb_ca_gate.reshape(&[b, 1, 1, video_dim])?; // [B, 1, 1, dim] (4D)
        let a2v_gate = v_gate_bc.add(&temb_gate_4d)?.squeeze_dim(2)?;

        // Audio per-layer
        let a_ss_bc = a_table_ss.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 4, dim]
        let temb_a_ss_4d = temb_ca_audio_ss.reshape(&[b, 1, 4, audio_dim])?;
        let a_combined = a_ss_bc.add(&temb_a_ss_4d)?;

        let audio_a2v_scale = a_combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let audio_a2v_shift = a_combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let audio_v2a_scale = a_combined.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let audio_v2a_shift = a_combined.narrow(2, 3, 1)?.squeeze_dim(2)?;

        let a_gate_bc = a_table_gate.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 1, dim]
        let temb_a_gate_4d = temb_ca_audio_gate.reshape(&[b, 1, 1, audio_dim])?;
        let v2a_gate = a_gate_bc.add(&temb_a_gate_4d)?.squeeze_dim(2)?;

        Ok((
            a2v_gate, v2a_gate,
            (video_a2v_scale, video_a2v_shift),
            (video_v2a_scale, video_v2a_shift),
            (audio_a2v_scale, audio_a2v_shift),
            (audio_v2a_scale, audio_v2a_shift),
        ))
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// LTX-2.3 Video+Audio Transformer (18.88B parameters).
///
/// This is the main model struct. Blocks are stored as a Vec for
/// Stagehand-style block-level CPU offloading during inference.
pub struct LTX2Model {
    pub config: LTX2Config,

    // Patchification
    proj_in_weight: Tensor,
    proj_in_bias: Tensor,
    audio_proj_in_weight: Tensor,
    audio_proj_in_bias: Tensor,

    // Caption projections
    caption_projection: CaptionProjection,
    audio_caption_projection: CaptionProjection,

    // Timestep embeddings and modulation
    time_embed: AdaLayerNormSingle,
    audio_time_embed: AdaLayerNormSingle,

    // Cross-attention global modulation
    av_cross_attn_video_scale_shift: AdaLayerNormSingle,
    av_cross_attn_audio_scale_shift: AdaLayerNormSingle,
    av_cross_attn_video_a2v_gate: AdaLayerNormSingle,
    av_cross_attn_audio_v2a_gate: AdaLayerNormSingle,

    // Output scale/shift tables
    scale_shift_table: Tensor,       // [2, inner_dim]
    audio_scale_shift_table: Tensor, // [2, audio_inner_dim]

    // Transformer blocks (stored separately for block-level offloading)
    pub blocks: Vec<LTX2TransformerBlock>,

    // Output layers
    // norm_out: LayerNorm (no affine)
    proj_out_weight: Tensor,
    proj_out_bias: Tensor,
    // audio_norm_out: LayerNorm (no affine)
    audio_proj_out_weight: Tensor,
    audio_proj_out_bias: Tensor,
}

impl LTX2Model {
    /// Total number of transformer blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - [B, num_video_tokens, in_channels] patchified video latents
    /// * `audio_hidden_states` - [B, num_audio_tokens, audio_in_channels] patchified audio latents
    /// * `encoder_hidden_states` - [B, text_seq_len, caption_channels] text embeddings for video
    /// * `audio_encoder_hidden_states` - [B, text_seq_len, caption_channels] text embeddings for audio
    /// * `timestep` - [B, num_video_tokens] scaled timestep
    /// * `audio_timestep` - [B, num_audio_tokens] scaled timestep (or same as timestep)
    /// * `video_coords` - [B, 3, num_video_tokens, 2] RoPE coordinate bounds
    /// * `audio_coords` - [B, 1, num_audio_tokens, 2] RoPE coordinate bounds
    /// * `encoder_attention_mask` - optional [B, text_seq_len]
    /// * `audio_encoder_attention_mask` - optional [B, text_seq_len]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        audio_hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        audio_timestep: &Tensor,
        video_coords: &Tensor,
        audio_coords: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = hidden_states.shape().dims()[0];
        let inner_dim = self.config.inner_dim();
        let audio_inner_dim = self.config.audio_inner_dim();

        // 1. Compute RoPE positional embeddings
        let video_max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let audio_max_pos = [self.config.pos_embed_max_pos as f64];

        let (v_cos, v_sin) = compute_rope_frequencies(
            video_coords, inner_dim, &video_max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;
        let (a_cos, a_sin) = compute_rope_frequencies(
            audio_coords, audio_inner_dim, &audio_max_pos,
            self.config.rope_theta, self.config.audio_num_attention_heads,
        )?;

        // Cross-attention RoPE: temporal only (first dim of video coords)
        let video_temporal_coords = video_coords.narrow(1, 0, 1)?; // [B, 1, P, 2]
        let ca_audio_dim = self.config.audio_cross_attention_dim;
        let ca_video_max_pos = [self.config.pos_embed_max_pos as f64];

        let (ca_v_cos, ca_v_sin) = compute_rope_frequencies(
            &video_temporal_coords, ca_audio_dim, &ca_video_max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;
        let audio_temporal_coords = audio_coords.narrow(1, 0, 1)?;
        let (ca_a_cos, ca_a_sin) = compute_rope_frequencies(
            &audio_temporal_coords, ca_audio_dim, &ca_video_max_pos,
            self.config.rope_theta, self.config.audio_num_attention_heads,
        )?;

        // 2. Convert encoder attention masks to bias
        let enc_mask = encoder_attention_mask.map(|m| -> Result<Tensor> {
            let m_f = m.to_dtype(DType::BF16)?;
            let one = Tensor::ones_dtype(m_f.shape().clone(), DType::BF16, m_f.device().clone())?;
            let inverted = one.sub(&m_f)?;
            let biased = inverted.mul_scalar(-10000.0)?;
            biased.unsqueeze(1)
        }).transpose()?;

        let audio_enc_mask = audio_encoder_attention_mask.map(|m| -> Result<Tensor> {
            let m_f = m.to_dtype(DType::BF16)?;
            let one = Tensor::ones_dtype(m_f.shape().clone(), DType::BF16, m_f.device().clone())?;
            let inverted = one.sub(&m_f)?;
            let biased = inverted.mul_scalar(-10000.0)?;
            biased.unsqueeze(1)
        }).transpose()?;

        // 3. Patchify input projections
        let mut hs = linear3d(hidden_states, &self.proj_in_weight, Some(&self.proj_in_bias))?;
        let mut ahs = linear3d(audio_hidden_states, &self.audio_proj_in_weight, Some(&self.audio_proj_in_bias))?;

        // 4. Timestep embeddings and global modulation params
        let ts_flat = timestep.reshape(&[batch_size * timestep.shape().dims()[1]])?;
        let (temb, embedded_ts) = self.time_embed.forward(&ts_flat)?;
        let temb = temb.reshape(&[batch_size, 1, temb.shape().dims()[temb.shape().rank() - 1]])?;
        let embedded_ts = embedded_ts.reshape(&[batch_size, 1, embedded_ts.shape().dims()[embedded_ts.shape().rank() - 1]])?;

        let ats_flat = audio_timestep.reshape(&[batch_size * audio_timestep.shape().dims()[1]])?;
        let (temb_audio, audio_embedded_ts) = self.audio_time_embed.forward(&ats_flat)?;
        let temb_audio = temb_audio.reshape(&[batch_size, 1, temb_audio.shape().dims()[temb_audio.shape().rank() - 1]])?;
        let audio_embedded_ts = audio_embedded_ts.reshape(&[batch_size, 1, audio_embedded_ts.shape().dims()[audio_embedded_ts.shape().rank() - 1]])?;

        // 4.2 Global cross-attention modulation params
        let cross_gate_scale = (self.config.cross_attn_timestep_scale_multiplier
            / self.config.timestep_scale_multiplier) as f32;

        let (v_ca_ss, _) = self.av_cross_attn_video_scale_shift.forward(&ts_flat)?;
        let v_ca_ss = v_ca_ss.reshape(&[batch_size, 1, v_ca_ss.shape().dims()[v_ca_ss.shape().rank() - 1]])?;

        let scaled_ts = ts_flat.mul_scalar(cross_gate_scale)?;
        let (v_ca_gate, _) = self.av_cross_attn_video_a2v_gate.forward(&scaled_ts)?;
        let v_ca_gate = v_ca_gate.reshape(&[batch_size, 1, v_ca_gate.shape().dims()[v_ca_gate.shape().rank() - 1]])?;

        let (a_ca_ss, _) = self.av_cross_attn_audio_scale_shift.forward(&ats_flat)?;
        let a_ca_ss = a_ca_ss.reshape(&[batch_size, 1, a_ca_ss.shape().dims()[a_ca_ss.shape().rank() - 1]])?;

        let scaled_ats = ats_flat.mul_scalar(cross_gate_scale)?;
        let (a_ca_gate, _) = self.av_cross_attn_audio_v2a_gate.forward(&scaled_ats)?;
        let a_ca_gate = a_ca_gate.reshape(&[batch_size, 1, a_ca_gate.shape().dims()[a_ca_gate.shape().rank() - 1]])?;

        // 5. Prepare prompt embeddings
        let enc_hs = self.caption_projection.forward(encoder_hidden_states)?;
        let enc_hs = enc_hs.reshape(&[batch_size, enc_hs.shape().dims()[1], inner_dim])?;

        let audio_enc_hs = self.audio_caption_projection.forward(audio_encoder_hidden_states)?;
        let audio_enc_hs = audio_enc_hs.reshape(&[batch_size, audio_enc_hs.shape().dims()[1], audio_inner_dim])?;

        // 6. Run transformer blocks
        // Legacy LTX2Model forward — does NOT carry the prompt_adaln_single
        // path (that's only in LTX2StreamingModel::forward_audio_video).
        // Pass None for the new prompt_timestep params so this path is
        // unchanged in behavior; it'll fall back to the unmodulated KV path
        // inside the block.
        for block in &self.blocks {
            let (new_hs, new_ahs) = block.forward(
                &hs, &ahs,
                &enc_hs, &audio_enc_hs,
                &temb, &temb_audio,
                &v_ca_ss, &a_ca_ss,
                &v_ca_gate, &a_ca_gate,
                Some((&v_cos, &v_sin)),
                Some((&a_cos, &a_sin)),
                Some((&ca_v_cos, &ca_v_sin)),
                Some((&ca_a_cos, &ca_a_sin)),
                enc_mask.as_ref(),
                audio_enc_mask.as_ref(),
                None, None,
            )?;
            hs = new_hs;
            ahs = new_ahs;
        }

        // 7. Output layers
        // Video: norm -> scale/shift -> proj_out
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 2, dim]
        let embedded_ts_4d = embedded_ts.unsqueeze(2)?; // [B, 1, 1, dim]
        let final_ss = shift_scale.add(&embedded_ts_4d)?;
        let shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let normed = layer_norm_no_affine(&hs, 1e-6)?;
        let output = normed.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        let output = linear3d(&output, &self.proj_out_weight, Some(&self.proj_out_bias))?;

        // Audio: norm -> scale/shift -> proj_out
        let a_shift_scale = self.audio_scale_shift_table.unsqueeze(0)?.unsqueeze(0)?;
        let a_embedded_ts_4d = audio_embedded_ts.unsqueeze(2)?;
        let a_final_ss = a_shift_scale.add(&a_embedded_ts_4d)?;
        let a_shift = a_final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let a_scale = a_final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let a_normed = layer_norm_no_affine(&ahs, 1e-6)?;
        let a_output = a_normed.mul(&a_scale.add_scalar(1.0)?)?.add(&a_shift)?;
        let audio_output = linear3d(&a_output, &self.audio_proj_out_weight, Some(&self.audio_proj_out_bias))?;

        Ok((output, audio_output))
    }
}

// ---------------------------------------------------------------------------
// Weight Loading
// ---------------------------------------------------------------------------

/// Load an AdaLayerNormSingle from a weight map with the given prefix.
fn load_ada_layer_norm_single(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    num_mod_params: usize,
) -> Result<AdaLayerNormSingle> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(AdaLayerNormSingle {
        emb: TimestepEmbedderMLP {
            linear_1_weight: pre_transpose_weight(&get(&format!("{prefix}.emb.timestep_embedder.linear_1.weight"))?)?,
            linear_1_bias: get(&format!("{prefix}.emb.timestep_embedder.linear_1.bias"))?,
            linear_2_weight: pre_transpose_weight(&get(&format!("{prefix}.emb.timestep_embedder.linear_2.weight"))?)?,
            linear_2_bias: get(&format!("{prefix}.emb.timestep_embedder.linear_2.bias"))?,
        },
        linear_weight: pre_transpose_weight(&get(&format!("{prefix}.linear.weight"))?)?,
        linear_bias: get(&format!("{prefix}.linear.bias"))?,
        num_mod_params,
    })
}

/// Load a CaptionProjection from a weight map.
fn load_caption_projection(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<CaptionProjection> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(CaptionProjection {
        linear_1_weight: pre_transpose_weight(&get(&format!("{prefix}.linear_1.weight"))?)?,
        linear_1_bias: get(&format!("{prefix}.linear_1.bias"))?,
        linear_2_weight: pre_transpose_weight(&get(&format!("{prefix}.linear_2.weight"))?)?,
        linear_2_bias: get(&format!("{prefix}.linear_2.bias"))?,
    })
}

/// Load a FeedForward from a weight map.
fn load_feed_forward(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<FeedForward> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(FeedForward {
        gelu_proj_weight: pre_transpose_weight(&get(&format!("{prefix}.net.0.proj.weight"))?)?,
        gelu_proj_bias: get(&format!("{prefix}.net.0.proj.bias"))?,
        out_weight: pre_transpose_weight(&get(&format!("{prefix}.net.2.weight"))?)?,
        out_bias: get(&format!("{prefix}.net.2.bias"))?,
    })
}

/// Load an LTX2Attention from a weight map.
fn load_attention(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<LTX2Attention> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    let gate_weight = weights.get(&format!("{prefix}.to_gate_logits.weight"))
        .map(|w| pre_transpose_weight(w))
        .transpose()?;

    Ok(LTX2Attention {
        to_q_weight: pre_transpose_weight(&get(&format!("{prefix}.to_q.weight"))?)?,
        to_q_bias: get(&format!("{prefix}.to_q.bias"))?,
        to_k_weight: pre_transpose_weight(&get(&format!("{prefix}.to_k.weight"))?)?,
        to_k_bias: get(&format!("{prefix}.to_k.bias"))?,
        to_v_weight: pre_transpose_weight(&get(&format!("{prefix}.to_v.weight"))?)?,
        to_v_bias: get(&format!("{prefix}.to_v.bias"))?,
        norm_q_weight: get(&format!("{prefix}.norm_q.weight"))
            .or_else(|_| get(&format!("{prefix}.q_norm.weight")))?,
        norm_k_weight: get(&format!("{prefix}.norm_k.weight"))
            .or_else(|_| get(&format!("{prefix}.k_norm.weight")))?,
        to_out_weight: pre_transpose_weight(&get(&format!("{prefix}.to_out.0.weight"))?)?,
        to_out_bias: get(&format!("{prefix}.to_out.0.bias"))?,
        to_gate_logits_weight: gate_weight,
        to_gate_logits_bias: weights.get(&format!("{prefix}.to_gate_logits.bias")).cloned(),
        num_heads,
        head_dim,
        eps,
    })
}

/// Load a single transformer block.
fn load_transformer_block(
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    config: &LTX2Config,
) -> Result<LTX2TransformerBlock> {
    let prefix = format!("transformer_blocks.{block_idx}");
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };
    let get_opt = |key: &str| -> Option<Tensor> {
        weights.get(key).cloned()
    };

    let eps = config.norm_eps;
    let num_heads = config.num_attention_heads;
    let head_dim = config.attention_head_dim;
    let audio_heads = config.audio_num_attention_heads;
    let audio_head_dim = config.audio_attention_head_dim;

    Ok(LTX2TransformerBlock {
        // Video self-attention
        norm1_weight: get_opt(&format!("{prefix}.norm1.weight")),
        attn1: load_attention(weights, &format!("{prefix}.attn1"), num_heads, head_dim, eps)?,

        // Audio self-attention
        audio_norm1_weight: get_opt(&format!("{prefix}.audio_norm1.weight")),
        audio_attn1: load_attention(weights, &format!("{prefix}.audio_attn1"), audio_heads, audio_head_dim, eps)?,

        // Video cross-attention
        norm2_weight: get_opt(&format!("{prefix}.norm2.weight")),
        attn2: load_attention(weights, &format!("{prefix}.attn2"), num_heads, head_dim, eps)?,

        // Audio cross-attention
        audio_norm2_weight: get_opt(&format!("{prefix}.audio_norm2.weight")),
        audio_attn2: load_attention(weights, &format!("{prefix}.audio_attn2"), audio_heads, audio_head_dim, eps)?,

        // A2V cross-attention
        audio_to_video_norm_weight: get_opt(&format!("{prefix}.audio_to_video_norm.weight")),
        audio_to_video_attn: load_attention(weights, &format!("{prefix}.audio_to_video_attn"), audio_heads, audio_head_dim, eps)?,

        // V2A cross-attention
        video_to_audio_norm_weight: get_opt(&format!("{prefix}.video_to_audio_norm.weight")),
        video_to_audio_attn: load_attention(weights, &format!("{prefix}.video_to_audio_attn"), audio_heads, audio_head_dim, eps)?,

        // Video FFN
        norm3_weight: get_opt(&format!("{prefix}.norm3.weight")),
        ff: load_feed_forward(weights, &format!("{prefix}.ff"))?,

        // Audio FFN
        audio_norm3_weight: get_opt(&format!("{prefix}.audio_norm3.weight")),
        audio_ff: load_feed_forward(weights, &format!("{prefix}.audio_ff"))?,

        // Modulation tables
        scale_shift_table: get(&format!("{prefix}.scale_shift_table"))?,
        audio_scale_shift_table: get(&format!("{prefix}.audio_scale_shift_table"))?,
        video_a2v_cross_attn_scale_shift_table: get(&format!("{prefix}.video_a2v_cross_attn_scale_shift_table"))?,
        audio_a2v_cross_attn_scale_shift_table: get(&format!("{prefix}.audio_a2v_cross_attn_scale_shift_table"))?,

        prompt_scale_shift_table: get_opt(&format!("{prefix}.prompt_scale_shift_table")),
        audio_prompt_scale_shift_table: get_opt(&format!("{prefix}.audio_prompt_scale_shift_table")),

        eps,
    })
}

/// Load the full LTX-2 model from a safetensors weight map.
///
/// The weight map should contain all keys matching the diffusers
/// `LTX2VideoTransformer3DModel` state dict format.
///
/// # Block-Level Offloading
///
/// After loading, individual blocks can be moved to CPU and loaded
/// back on-demand using the `blocks` Vec. This is essential for
/// fitting the 18.88B model in 24GB VRAM.
pub fn load_ltx2_model(
    weights: &HashMap<String, Tensor>,
    config: &LTX2Config,
) -> Result<LTX2Model> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    // Load all transformer blocks
    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        blocks.push(load_transformer_block(weights, i, config)?);
    }

    Ok(LTX2Model {
        config: config.clone(),

        proj_in_weight: pre_transpose_weight(&get("proj_in.weight")?)?,
        proj_in_bias: get("proj_in.bias")?,
        audio_proj_in_weight: pre_transpose_weight(&get("audio_proj_in.weight")?)?,
        audio_proj_in_bias: get("audio_proj_in.bias")?,

        caption_projection: load_caption_projection(weights, "caption_projection")?,
        audio_caption_projection: load_caption_projection(weights, "audio_caption_projection")?,

        time_embed: load_ada_layer_norm_single(weights, "time_embed", 6)?,
        audio_time_embed: load_ada_layer_norm_single(weights, "audio_time_embed", 6)?,

        av_cross_attn_video_scale_shift: load_ada_layer_norm_single(weights, "av_cross_attn_video_scale_shift", 4)?,
        av_cross_attn_audio_scale_shift: load_ada_layer_norm_single(weights, "av_cross_attn_audio_scale_shift", 4)?,
        av_cross_attn_video_a2v_gate: load_ada_layer_norm_single(weights, "av_cross_attn_video_a2v_gate", 1)?,
        av_cross_attn_audio_v2a_gate: load_ada_layer_norm_single(weights, "av_cross_attn_audio_v2a_gate", 1)?,

        scale_shift_table: get("scale_shift_table")?,
        audio_scale_shift_table: get("audio_scale_shift_table")?,

        blocks,

        proj_out_weight: pre_transpose_weight(&get("proj_out.weight")?)?,
        proj_out_bias: get("proj_out.bias")?,
        audio_proj_out_weight: pre_transpose_weight(&get("audio_proj_out.weight")?)?,
        audio_proj_out_bias: get("audio_proj_out.bias")?,
    })
}

// ---------------------------------------------------------------------------
// Block-level offloading API (Stagehand-style)
// ---------------------------------------------------------------------------

impl LTX2Model {
    /// Number of parameters in a single transformer block (approximate).
    /// Useful for VRAM budget calculations.
    pub fn block_param_count(&self) -> usize {
        let d = self.config.inner_dim();
        let ad = self.config.audio_inner_dim();
        let ffn = self.config.ffn_hidden();
        let affn = self.config.audio_ffn_hidden();

        // Per-block: 6 attention modules + 2 FFN + modulation tables
        // Each attention: Q/K/V/Out projections + 2 norms
        let video_self_attn = 4 * d * d + 4 * d + 2 * d;  // Q,K,V,Out weights+biases + QK norms
        let audio_self_attn = 4 * ad * ad + 4 * ad + 2 * ad;
        let video_cross_attn = video_self_attn;  // same structure
        let audio_cross_attn = audio_self_attn;
        let a2v_attn = 2 * d * ad + 2 * ad * ad + (d + ad) * 2;  // mixed dims
        let v2a_attn = a2v_attn;
        let video_ffn = d * ffn + ffn + ffn * d + d;
        let audio_ffn = ad * affn + affn + affn * ad + ad;
        let tables = 6 * d + 6 * ad + 5 * d + 5 * ad;

        video_self_attn + audio_self_attn + video_cross_attn + audio_cross_attn
            + a2v_attn + v2a_attn + video_ffn + audio_ffn + tables
    }

    /// Estimated VRAM for a single block in BF16 (bytes).
    pub fn block_vram_bytes(&self) -> usize {
        self.block_param_count() * 2  // BF16 = 2 bytes per param
    }

    /// Estimated total model VRAM in BF16 (bytes).
    pub fn total_vram_bytes(&self) -> usize {
        self.block_vram_bytes() * self.config.num_layers
            + self.non_block_param_count() * 2
    }

    /// Non-block parameter count (input/output projections, embeddings, etc.)
    fn non_block_param_count(&self) -> usize {
        let d = self.config.inner_dim();
        let ad = self.config.audio_inner_dim();
        let cc = self.config.caption_channels;

        // proj_in, proj_out, audio variants
        let proj = 2 * (self.config.in_channels * d + d)
            + 2 * (self.config.audio_in_channels * ad + ad);

        // caption projections (2x: linear1 + linear2 for video and audio)
        let caption = 2 * (cc * d + d + d * d + d)
            + 2 * (cc * ad + ad + ad * ad + ad);

        // time embeddings (6 AdaLayerNormSingle modules)
        // Each: timestep_embedder(256->dim, dim->dim) + silu + linear(dim->n*dim)
        let time_emb = |dim: usize, n: usize| -> usize {
            (256 * dim + dim + dim * dim + dim) + (dim * n * dim + n * dim)
        };
        let te = time_emb(d, 6) + time_emb(ad, 6)
            + time_emb(d, 4) + time_emb(ad, 4)
            + time_emb(d, 1) + time_emb(ad, 1);

        // Output tables
        let tables = 2 * d + 2 * ad;

        proj + caption + te + tables
    }
}

// ---------------------------------------------------------------------------
// Video-Only Forward + Loader
// ---------------------------------------------------------------------------

impl LTX2Model {
    /// Video-only forward pass. Skips all audio paths.
    ///
    /// # Arguments
    /// * `x` - Spatial video latents [B, C, F, H, W] in BF16
    /// * `timestep` - Sigma values [B] in [0, 1] (NOT pre-scaled)
    /// * `context` - Text embeddings [B, seq_len, caption_channels] from Gemma
    /// * `frame_rate` - Video frame rate (default 25.0), affects temporal RoPE scaling
    ///
    /// # Returns
    /// Model velocity output [B, C, F, H, W] in BF16.
    pub fn forward_video_only(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        frame_rate: f32,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (batch_size, channels, num_frames, height, width) =
            (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
        let inner_dim = self.config.inner_dim();
        let num_tokens = num_frames * height * width;

        // 1. Patchify: [B, C, F, H, W] → [B, F*H*W, C]
        // For patch_size=1 this is just a reshape
        let x_flat = x.reshape(&[batch_size, channels, num_tokens])?
            .permute(&[0, 2, 1])?; // [B, N, C]

        // 2. Build coordinate grid [B, 3, N, 2] for RoPE
        let device = x.device().clone();
        let vae_sf = &self.config.vae_scale_factors;
        let coords = build_video_coords(
            batch_size, num_frames, height, width,
            vae_sf, self.config.causal_offset, frame_rate, device.clone(),
        )?;

        // 3. Compute RoPE
        let max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let (v_cos, v_sin) = compute_rope_frequencies(
            &coords, inner_dim, &max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;

        // 4. Input projection: [B, N, C] → [B, N, inner_dim]
        let mut hs = linear3d(&x_flat, &self.proj_in_weight, Some(&self.proj_in_bias))?;

        // 5. Timestep conditioning
        // Expand sigma to per-token: [B] → [B, N]
        let ts_expanded = timestep.unsqueeze(1)?
            .expand(&[batch_size, num_tokens])?;
        let ts_scaled = ts_expanded.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
        let ts_flat = ts_scaled.reshape(&[batch_size * num_tokens])?;
        let (v_timestep, v_embedded) = self.time_embed.forward(&ts_flat)?;
        // Reshape: [B*N, 6*dim] → [B, N, 6*dim]
        let v_timestep = v_timestep.reshape(&[batch_size, num_tokens, 6 * inner_dim])?;
        let v_embedded = v_embedded.reshape(&[batch_size, num_tokens, inner_dim])?;

        // 6. Caption projection: [B, seq, 3840] → [B, seq, inner_dim]
        let enc_hs = self.caption_projection.forward(context)?;

        // 7. Run transformer blocks (video-only)
        for block in &self.blocks {
            hs = block.forward_video_only(
                &hs, &enc_hs, &v_timestep,
                Some((&v_cos, &v_sin)),
                None, // encoder_attention_mask
                None, // prompt_timestep (legacy path)
            )?;
        }

        // 8. Output: norm → scale/shift → proj_out
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 2, dim]
        let emb_4d = v_embedded.unsqueeze(2)?; // [B, N, 1, dim]
        let final_ss = shift_scale.add(&emb_4d)?;
        let shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let normed = layer_norm_no_affine(&hs, self.config.norm_eps)?;
        let output = normed.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        let output = linear3d(&output, &self.proj_out_weight, Some(&self.proj_out_bias))?;

        // 9. Unpatchify: [B, N, C] → [B, C, F, H, W]
        let output = output.permute(&[0, 2, 1])?; // [B, C, N]
        output.reshape(&[batch_size, channels, num_frames, height, width])
    }
}

/// Build 3D video coordinate grid for RoPE.
///
/// Returns [B, 3, N, 2] tensor where N = F*H*W, last dim is [start, end).
/// Coordinates are in pixel space (scaled by VAE factors) with causal fix
/// and frame_rate scaling applied.
fn build_video_coords(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    vae_scale_factors: &[usize; 3],
    causal_offset: usize,
    frame_rate: f32,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    let num_tokens = num_frames * height * width;

    // Build meshgrid: (f, h, w) for each token
    let mut coords_data = vec![0.0f32; batch_size * 3 * num_tokens * 2];

    for b in 0..batch_size {
        for f in 0..num_frames {
            for h in 0..height {
                for w in 0..width {
                    let token_idx = f * height * width + h * width + w;
                    let base = b * 3 * num_tokens * 2;

                    // Latent coords: [start, end) = [idx, idx+1]
                    // Pixel coords = latent * vae_scale_factor
                    let f_start = (f * vae_scale_factors[0]) as f32;
                    let f_end = ((f + 1) * vae_scale_factors[0]) as f32;
                    let h_start = (h * vae_scale_factors[1]) as f32;
                    let h_end = ((h + 1) * vae_scale_factors[1]) as f32;
                    let w_start = (w * vae_scale_factors[2]) as f32;
                    let w_end = ((w + 1) * vae_scale_factors[2]) as f32;

                    // Causal fix: temporal coords = (coords + causal_offset - vae_t).clamp(min=0)
                    let vae_t = vae_scale_factors[0] as f32;
                    let f_start_causal = (f_start + causal_offset as f32 - vae_t).max(0.0);
                    let f_end_causal = (f_end + causal_offset as f32 - vae_t).max(0.0);

                    // Frame-rate scaling on temporal dim
                    let f_start_scaled = f_start_causal / frame_rate;
                    let f_end_scaled = f_end_causal / frame_rate;

                    // dim 0 = temporal
                    coords_data[base + 0 * num_tokens * 2 + token_idx * 2] = f_start_scaled;
                    coords_data[base + 0 * num_tokens * 2 + token_idx * 2 + 1] = f_end_scaled;
                    // dim 1 = height
                    coords_data[base + 1 * num_tokens * 2 + token_idx * 2] = h_start;
                    coords_data[base + 1 * num_tokens * 2 + token_idx * 2 + 1] = h_end;
                    // dim 2 = width
                    coords_data[base + 2 * num_tokens * 2 + token_idx * 2] = w_start;
                    coords_data[base + 2 * num_tokens * 2 + token_idx * 2 + 1] = w_end;
                }
            }
        }
    }

    Tensor::from_vec_dtype(
        coords_data,
        Shape::from_dims(&[batch_size, 3, num_tokens, 2]),
        device,
        DType::F32,
    )
}

/// Build audio coordinate grid for RoPE. Audio has temporal-only coords.
/// Returns [B, 1, num_audio_tokens, 2] with (start, end) pairs for each token.
fn build_audio_coords(
    batch_size: usize,
    audio_frames: usize,
    _audio_freq: usize,
    audio_scale_factor: usize,
    vae_temporal_scale: usize,
    causal_offset: usize,
    frame_rate: f32,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    // Audio has 1 token per temporal frame (mel bins are flattened into channels).
    let num_tokens = audio_frames;
    let mut coords_data = vec![0.0f32; batch_size * 1 * num_tokens * 2];

    for b in 0..batch_size {
        for t in 0..audio_frames {
            let base = b * 1 * num_tokens * 2;

            // Audio temporal coords: scaled by audio_scale_factor * vae_temporal_scale
            let scale = (audio_scale_factor * vae_temporal_scale) as f32;
            let t_start = t as f32 * scale;
            let t_end = (t + 1) as f32 * scale;

            // Causal fix
            let vae_t = vae_temporal_scale as f32;
            let t_start_causal = (t_start + causal_offset as f32 - vae_t).max(0.0);
            let t_end_causal = (t_end + causal_offset as f32 - vae_t).max(0.0);

            // Frame-rate scaling
            let t_start_scaled = t_start_causal / frame_rate;
            let t_end_scaled = t_end_causal / frame_rate;

            coords_data[base + t * 2] = t_start_scaled;
            coords_data[base + t * 2 + 1] = t_end_scaled;
        }
    }

    Tensor::from_vec_dtype(
        coords_data,
        Shape::from_dims(&[batch_size, 1, num_tokens, 2]),
        device,
        DType::F32,
    )
}

// ---------------------------------------------------------------------------
// VideoEmbeddingsConnector (ComfyUI LTX-2.3)
// ---------------------------------------------------------------------------

/// A single connector block: self-attention + FFN with residual connections.
struct ConnectorBlock {
    attn: LTX2Attention,
    ff: FeedForward,
    norm1_weight: Option<Tensor>,
    norm2_weight: Option<Tensor>,
    eps: f32,
}

impl ConnectorBlock {
    fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        // Self-attention with residual
        let norm_x = rms_norm(x, self.norm1_weight.as_ref(), self.eps)?;
        let attn_out = self.attn.forward(&norm_x, None, None, rope, None)?;
        let x = x.add(&attn_out)?;
        // FFN with residual
        let norm_x = rms_norm(&x, self.norm2_weight.as_ref(), self.eps)?;
        let ff_out = self.ff.forward(&norm_x)?;
        x.add(&ff_out)
    }
}

/// VideoEmbeddingsConnector: small transformer that processes text embeddings.
/// Replaces `caption_projection` in ComfyUI LTX-2.3 checkpoints.
pub struct VideoEmbeddingsConnector {
    blocks: Vec<ConnectorBlock>,
    learnable_registers: Option<Tensor>,  // [num_registers, inner_dim]
    num_registers: usize,
    inner_dim: usize,
    num_heads: usize,
    _head_dim: usize,
    eps: f32,
    rope_theta: f64,
    /// `connector_positional_embedding_max_pos` from the checkpoint config
    /// (LTX-2.3 22B uses 4096). Default for LTX-2 prior to 2.3 was 1.
    rope_max_pos: u32,
}

impl VideoEmbeddingsConnector {
    /// Forward: process text embeddings through connector blocks with 1D RoPE.
    ///
    /// `x`: [B, seq, inner_dim] left-padded text embeddings (Gemma is causal,
    ///      so real tokens sit at the right end of the sequence).
    /// `mask`: Optional additive attention mask `[B, 1, 1, seq]`. Real
    ///      positions are 0, padded positions are <= -9000.
    ///
    /// Mirrors `Embeddings1DConnector.forward` in
    /// `ltx_core/text_encoders/gemma/embeddings_connector.py`:
    ///   1. Compress real tokens to the front of `seq`, fill the rest with
    ///      tiled `learnable_registers`.
    ///   2. Compute 1D RoPE with `position = i`, `max_pos = [1]`.
    ///   3. Run N transformer blocks (rms_norm / self-attn + residual /
    ///      rms_norm / FFN + residual).
    ///   4. Final unweighted RMSNorm.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (batch_size, seq_len, _dim) = (dims[0], dims[1], dims[2]);
        let device = x.device().clone();

        // 1. Replace padded positions with learnable registers (compress real
        //    tokens to the front + tile registers to fill the rest).
        let mut x = if let Some(registers) = &self.learnable_registers {
            // Determine number of real tokens N from the additive mask if
            // provided. With no mask we treat the whole sequence as real.
            let num_real: usize = if let Some(m) = mask {
                // Extract last dim of the mask to a flat F32 tensor and count
                // positions whose value is >= -9000 (real).
                let m_dims = m.shape().dims().to_vec();
                let last = m_dims[m_dims.len() - 1];
                // Take batch 0 only — left-padding is identical across batch
                // for our pipeline (always batch_size=1 in inference).
                let m_b0 = if m_dims.len() == 4 {
                    m.narrow(0, 0, 1)?.squeeze_dim(0)?.squeeze_dim(0)?.squeeze_dim(0)?
                } else if m_dims.len() == 3 {
                    m.narrow(0, 0, 1)?.squeeze_dim(0)?.squeeze_dim(0)?
                } else {
                    m.narrow(0, 0, 1)?.squeeze_dim(0)?
                };
                let m_f32 = m_b0.to_dtype(DType::F32)?;
                let cpu = m_f32.to_vec()?;
                debug_assert_eq!(cpu.len(), last);
                cpu.iter().filter(|v| **v >= -9000.0).count()
            } else {
                seq_len
            };

            // For batch_size > 1 we'd need per-batch handling. For our
            // inference pipeline batch is always 1, so this simple path is
            // sufficient.
            assert_eq!(
                batch_size, 1,
                "VideoEmbeddingsConnector currently supports batch_size=1 only"
            );

            // Real tokens live at the right end of the sequence (left-padded).
            // [1, num_real, dim]
            let real_tokens = x.narrow(1, seq_len - num_real, num_real)?;

            // Tile registers to [seq, dim] then slice the [num_real..seq] tail
            // (Python uses tile then masked-blend; this gives the same
            // register indices as the Python algorithm).
            let registers_dim = registers.shape().dims()[1];
            assert_eq!(registers_dim, self.inner_dim);
            let n_reg = self.num_registers;
            let registers_seq = if seq_len <= n_reg {
                registers.narrow(0, 0, seq_len)?
            } else {
                assert_eq!(
                    seq_len % n_reg,
                    0,
                    "seq_len {} must be divisible by num_learnable_registers {}",
                    seq_len, n_reg
                );
                let n_copies = seq_len / n_reg;
                let parts: Vec<Tensor> = (0..n_copies).map(|_| registers.clone()).collect();
                let refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&refs, 0)?
            };

            let registers_tail = if num_real < seq_len {
                Some(registers_seq.narrow(0, num_real, seq_len - num_real)?
                    .unsqueeze(0)?  // [1, seq - num_real, dim]
                    .to_dtype(x.dtype())?)
            } else {
                None
            };

            if let Some(tail) = registers_tail {
                Tensor::cat(&[&real_tokens, &tail], 1)?
            } else {
                real_tokens
            }
        } else {
            x.clone()
        };

        // 2. Compute 1D RoPE for connector. Python uses `position = i` with
        //    `max_pos = connector_positional_embedding_max_pos` from the
        //    checkpoint config (LTX-2.3 22B uses [4096]; the Python class
        //    default of [1] is overridden). The Rust `compute_rope_frequencies`
        //    expects [start, end) coord pairs and takes their midpoint, so we
        //    pass `(i, i)` to make the midpoint exactly `i`.
        let coords_data: Vec<f32> = (0..batch_size)
            .flat_map(|_| {
                (0..seq_len).flat_map(|i| vec![i as f32, i as f32])
            })
            .collect();
        let coords = Tensor::from_vec_dtype(
            coords_data,
            Shape::from_dims(&[batch_size, 1, seq_len, 2]),
            device.clone(),
            DType::F32,
        )?;
        let max_pos = [self.rope_max_pos as f64];
        let (cos_freqs, sin_freqs) = compute_rope_frequencies(
            &coords, self.inner_dim, &max_pos,
            self.rope_theta, self.num_heads,
        )?;

        // 3. Run connector blocks. After the register replacement step the
        //    full sequence is "real" content (Python sets the mask to all
        //    zeros after this), so we pass `None` as the attention mask.
        for block in self.blocks.iter() {
            x = block.forward(&x, Some((&cos_freqs, &sin_freqs)))?;
        }

        // 4. Final unweighted RMSNorm (matches `rms_norm(hidden_states)`
        //    at the bottom of Embeddings1DConnector.forward).
        rms_norm(&x, None, self.eps)
    }
}

/// Load a VideoEmbeddingsConnector from checkpoint weights.
/// Auto-detects number of blocks from keys.
pub fn load_video_embeddings_connector(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    eps: f32,
    rope_theta: f64,
) -> Result<VideoEmbeddingsConnector> {
    // Auto-detect number of blocks by scanning keys.
    // Checkpoint may use "transformer_1d_blocks" (official) or "blocks" (alt).
    let mut max_block = None;
    let mut blocks_key = "transformer_1d_blocks";
    for key in weights.keys() {
        for pattern in &["transformer_1d_blocks", "blocks"] {
            if let Some(rest) = key.strip_prefix(&format!("{prefix}.{pattern}.")) {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        blocks_key = pattern;
                        max_block = Some(max_block.map_or(idx, |m: usize| m.max(idx)));
                    }
                }
            }
        }
    }
    let num_blocks = max_block.map(|m| m + 1).unwrap_or(0);
    if num_blocks == 0 {
        return Err(flame_core::Error::InvalidInput(
            format!("No connector blocks found with prefix '{prefix}'"),
        ));
    }
    let blk = blocks_key; // lifetime helper

    // Detect inner_dim and num_heads from first attention weight
    let first_q = weights.get(&format!("{prefix}.{blk}.0.attn1.to_q.weight"))
        .or_else(|| weights.get(&format!("{prefix}.{blk}.0.attn.to_q.weight")))
        .ok_or_else(|| flame_core::Error::InvalidInput(
            format!("Cannot find connector attention Q weight at '{prefix}.{blk}.0'"),
        ))?;
    let q_dims = first_q.shape().dims().to_vec();
    let inner_dim = q_dims[0]; // [out_features, in_features]

    // Detect num_heads from to_gate_logits if present, otherwise default head_dim=128
    let gate_logits_w = weights.get(&format!("{prefix}.{blk}.0.attn1.to_gate_logits.weight"))
        .or_else(|| weights.get(&format!("{prefix}.{blk}.0.attn.to_gate_logits.weight")));
    let (num_heads, head_dim) = if let Some(gl) = gate_logits_w {
        // to_gate_logits.weight shape: [num_heads, query_dim]
        let nh = gl.shape().dims()[0];
        (nh, inner_dim / nh)
    } else {
        let hd = 128; // default head_dim for LTX-2
        (inner_dim / hd, hd)
    };

    log::info!("[LTX2] Connector: {} blocks (key='{}'), inner_dim={}, heads={}, head_dim={}",
        num_blocks, blk, inner_dim, num_heads, head_dim);

    // Load blocks
    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        // Try "attn1" first (official format), then "attn"
        let attn_prefix = if weights.contains_key(&format!("{prefix}.{blk}.{i}.attn1.to_q.weight")) {
            format!("{prefix}.{blk}.{i}.attn1")
        } else {
            format!("{prefix}.{blk}.{i}.attn")
        };
        let ff_prefix = format!("{prefix}.{blk}.{i}.ff");

        let get_opt = |key: &str| -> Option<Tensor> { weights.get(key).cloned() };

        blocks.push(ConnectorBlock {
            attn: load_attention(weights, &attn_prefix, num_heads, head_dim, eps)?,
            ff: load_feed_forward(weights, &ff_prefix)?,
            norm1_weight: get_opt(&format!("{prefix}.{blk}.{i}.norm1.weight")),
            norm2_weight: get_opt(&format!("{prefix}.{blk}.{i}.norm2.weight")),
            eps,
        });
    }

    // Learnable registers
    let learnable_registers = weights.get(&format!("{prefix}.learnable_registers")).cloned();
    let num_registers = learnable_registers.as_ref()
        .map(|t| t.shape().dims()[0])
        .unwrap_or(128);

    Ok(VideoEmbeddingsConnector {
        blocks,
        learnable_registers,
        num_registers,
        inner_dim,
        num_heads,
        _head_dim: head_dim,
        eps,
        rope_theta,
        // LTX-2.3 22B distilled checkpoint sets max_pos=[4096]. We hard-code
        // it here because we don't currently parse the safetensors metadata
        // config; for older LTX-2 variants this would need to come from cfg.
        rope_max_pos: 4096,
    })
}

// ---------------------------------------------------------------------------
// LTX2StreamingModel
// ---------------------------------------------------------------------------

/// LTX-2 model with only global params loaded (blocks streamed from disk).
///
/// This struct holds the non-block parameters on GPU (~400MB) and loads
/// each of the 48 blocks from the safetensors file on demand during
/// forward pass. This is essential for fitting 33GB of video-only
/// weights into 24GB VRAM.
pub struct LTX2StreamingModel {
    pub config: LTX2Config,
    pub checkpoint_path: String,

    // Video global params on GPU
    pub proj_in_weight: Tensor,
    pub proj_in_bias: Tensor,
    pub caption_projection: Option<CaptionProjection>,
    pub connector: Option<VideoEmbeddingsConnector>,  // ComfyUI: replaces caption_projection (video)
    pub audio_connector: Option<VideoEmbeddingsConnector>,  // ComfyUI: same struct, audio dims
    pub prompt_adaln_single: Option<AdaLayerNormSingle>,  // ComfyUI: 2-param for cross-attn context (video)
    pub audio_prompt_adaln_single: Option<AdaLayerNormSingle>,  // ComfyUI: 2-param for cross-attn context (audio)
    // Text embedding projection: packed Gemma (188160-dim) → connector dim (4096 video / 2048 audio)
    pub aggregate_embed_weight: Option<Tensor>,  // [4096, 188160] (pre-transposed)
    pub aggregate_embed_bias: Option<Tensor>,    // [4096]
    pub audio_aggregate_embed_weight: Option<Tensor>,  // [2048, 188160] (pre-transposed)
    pub audio_aggregate_embed_bias: Option<Tensor>,    // [2048]
    pub time_embed: AdaLayerNormSingle,
    pub scale_shift_table: Tensor,  // [2, inner_dim]
    pub proj_out_weight: Tensor,
    pub proj_out_bias: Tensor,

    // Audio global params on GPU (None = audio not loaded)
    pub audio_proj_in_weight: Option<Tensor>,
    pub audio_proj_in_bias: Option<Tensor>,
    pub audio_caption_projection: Option<CaptionProjection>,
    pub audio_time_embed: Option<AdaLayerNormSingle>,
    pub audio_scale_shift_table: Option<Tensor>,  // [2, audio_inner_dim]
    pub audio_proj_out_weight: Option<Tensor>,
    pub audio_proj_out_bias: Option<Tensor>,
    // AV cross-attention global modulation
    pub av_cross_attn_video_scale_shift: Option<AdaLayerNormSingle>,
    pub av_cross_attn_audio_scale_shift: Option<AdaLayerNormSingle>,
    pub av_cross_attn_video_a2v_gate: Option<AdaLayerNormSingle>,
    pub av_cross_attn_audio_v2a_gate: Option<AdaLayerNormSingle>,

    // Block weight cache: pre-parsed block weights stored as raw tensors on GPU.
    // Populated by preload_blocks(). When present, load_block() reads from cache
    // instead of re-parsing the 43GB checkpoint for every denoise step.
    block_cache: Vec<Option<HashMap<String, Tensor>>>,

    /// Async block swapper (FlameSwap). When initialized via `init_swap()`,
    /// replaces block_cache and load_block_from_disk with async pinned transfers.
    pub swap: Option<flame_swap::FlameSwap>,
    /// Detected key prefix for stripping from FlameSwap output keys.
    pub key_prefix: String,
    /// Pre-cached F32 tensors per block (scale_shift_table etc.)
    /// Loaded once at init_swap to avoid BF16 precision loss.
    pub f32_cache: Vec<HashMap<String, Tensor>>,
    /// FP8-resident blocks: all weights on GPU, dequant per-block during forward.
    fp8_blocks: Vec<super::fp8_resident::ResidentBlock>,
}

impl LTX2StreamingModel {
    /// Load global (non-block) params from checkpoint. Blocks are NOT loaded.
    ///
    /// Auto-detects key prefix (`model.diffusion_model.` for ComfyUI format)
    /// and strips it. Supports both diffusers and ComfyUI checkpoints.
    pub fn load_globals(path: &str, config: &LTX2Config) -> Result<Self> {
        let device = flame_core::global_cuda_device();
        log::info!("[LTX2] Loading global params (non-block)...");

        // Detect key prefix by checking for known global keys
        let prefix = detect_key_prefix(path)?;
        log::info!("[LTX2] Detected key prefix: '{}'", prefix);

        let globals = flame_core::serialization::load_file_filtered(
            path, &device,
            |key| {
                // Only load diffusion model keys (skip VAE, vocoder, audio VAE)
                if !key.starts_with(&prefix) {
                    // Also load text_embedding_projection (video only)
                    if key.starts_with("text_embedding_projection.video") {
                        return true;
                    }
                    return false;
                }
                let k = key.strip_prefix(&prefix).unwrap_or(key);
                // Skip per-block weights (loaded later by FP8 resident or FlameSwap)
                if k.starts_with("transformer_blocks.") { return false; }
                // Load both video and audio globals (incl. video_embeddings_connector,
                // audio_embeddings_connector, and text_embedding_projection.* aggregate_embed
                // weights — needed to run text encoding fully in Rust without
                // pre-computed Python embeddings).
                true
            },
        )?;

        // Remap keys: strip prefix
        let globals: HashMap<String, Tensor> = globals.into_iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix(&prefix).unwrap_or(&k).to_string();
                (stripped, v)
            })
            .collect();

        let get = |key: &str| -> Result<Tensor> {
            globals.get(key).cloned()
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing: {}", key)))
        };

        log::info!("[LTX2] Global params loaded: {} keys", globals.len());

        // Support both key naming conventions
        let get_alt = |k1: &str, k2: &str| -> Result<Tensor> {
            get(k1).or_else(|_| get(k2))
        };

        // proj_in / patchify_proj
        let proj_in_w = get_alt("proj_in.weight", "patchify_proj.weight")?;
        let proj_in_b = get_alt("proj_in.bias", "patchify_proj.bias")?;

        // caption_projection — try multiple key patterns
        let caption_proj = load_caption_projection(&globals, "caption_projection")
            .or_else(|_| load_caption_projection(&globals, "text_embedding_projection.video"))
            .ok();

        // video_embeddings_connector (ComfyUI LTX-2.3) — replaces caption_projection
        let connector = match load_video_embeddings_connector(
            &globals, "video_embeddings_connector", config.norm_eps, config.rope_theta,
        ) {
            Ok(c) => Some(c),
            Err(e) => { log::warn!("[LTX2] Video connector load failed: {:?}", e); None }
        };
        // audio_embeddings_connector — same module structure as video, but with
        // audio dims (typically 16 heads × 128 = 2048 inner). Auto-detected from
        // weights so it works for any LTX-2 variant.
        let audio_connector = match load_video_embeddings_connector(
            &globals, "audio_embeddings_connector", config.norm_eps, config.rope_theta,
        ) {
            Ok(c) => Some(c),
            Err(e) => { log::warn!("[LTX2] Audio connector load failed: {:?}", e); None }
        };

        if caption_proj.is_none() && connector.is_none() {
            log::warn!("[LTX2] No video connector or caption_projection loaded (OK if using 4096-dim cached embeddings)");
        }
        if connector.is_some() {
            log::info!("[LTX2] Using VideoEmbeddingsConnector (ComfyUI format)");
        } else {
            log::info!("[LTX2] Using CaptionProjection (diffusers format)");
        }
        if audio_connector.is_some() {
            log::info!("[LTX2] Using AudioEmbeddingsConnector (ComfyUI format)");
        }

        // prompt_adaln_single (ComfyUI LTX-2.3): 2-param conditioning for cross-attn context (video)
        let prompt_adaln = load_ada_layer_norm_single(&globals, "prompt_adaln_single", 2).ok();
        if prompt_adaln.is_some() {
            log::info!("[LTX2] Found prompt_adaln_single (video cross-attn context modulation)");
        }
        // audio_prompt_adaln_single: same idea but for the audio cross-attention KV.
        // Without this, audio CA context is timestep-independent and Bug 1+3 from
        // LTX2_AV_BUGS.md fire (dark output, std=0.33).
        let audio_prompt_adaln = load_ada_layer_norm_single(&globals, "audio_prompt_adaln_single", 2).ok();
        if audio_prompt_adaln.is_some() {
            log::info!("[LTX2] Found audio_prompt_adaln_single (audio cross-attn context modulation)");
        }

        // video_aggregate_embed: projects packed Gemma embeddings (188160-dim) → 4096
        let agg_w = globals.get("text_embedding_projection.video_aggregate_embed.weight").cloned();
        let agg_b = globals.get("text_embedding_projection.video_aggregate_embed.bias").cloned();
        if agg_w.is_some() {
            let w_shape = agg_w.as_ref().unwrap().shape().dims().to_vec();
            log::info!("[LTX2] Found video_aggregate_embed: {:?}", w_shape);
        }
        // audio_aggregate_embed: projects packed Gemma embeddings (188160-dim) → 2048
        let audio_agg_w = globals.get("text_embedding_projection.audio_aggregate_embed.weight").cloned();
        let audio_agg_b = globals.get("text_embedding_projection.audio_aggregate_embed.bias").cloned();
        if audio_agg_w.is_some() {
            let w_shape = audio_agg_w.as_ref().unwrap().shape().dims().to_vec();
            log::info!("[LTX2] Found audio_aggregate_embed: {:?}", w_shape);
        }

        // time_embed / adaln_single (may output 6 or 9 params)
        let time_emb = load_ada_layer_norm_single(&globals, "time_embed", 6)
            .or_else(|_| load_ada_layer_norm_single(&globals, "adaln_single", 9))?;

        let agg_w_t = agg_w.as_ref().map(|w| pre_transpose_weight(w)).transpose()?;
        let audio_agg_w_t = audio_agg_w.as_ref().map(|w| pre_transpose_weight(w)).transpose()?;

        // Audio globals (optional — None if keys not found)
        // Key names vary: audio_proj_in / audio_patchify_proj, audio_time_embed / audio_adaln_single
        let audio_proj_in_w = globals.get("audio_proj_in.weight")
            .or_else(|| globals.get("audio_patchify_proj.weight"))
            .map(|w| pre_transpose_weight(w)).transpose()?;
        let audio_proj_in_b = globals.get("audio_proj_in.bias")
            .or_else(|| globals.get("audio_patchify_proj.bias"))
            .cloned();
        let audio_caption_proj = load_caption_projection(&globals, "audio_caption_projection").ok();
        let audio_time_emb = load_ada_layer_norm_single(&globals, "audio_time_embed", 6)
            .or_else(|_| load_ada_layer_norm_single(&globals, "audio_adaln_single", 9))
            .ok();
        let audio_sst = globals.get("audio_scale_shift_table").cloned();
        let audio_proj_out_w = globals.get("audio_proj_out.weight")
            .map(|w| pre_transpose_weight(w)).transpose()?;
        let audio_proj_out_b = globals.get("audio_proj_out.bias").cloned();

        // AV cross-attention global modulation
        let av_v_ss = load_ada_layer_norm_single(&globals, "av_cross_attn_video_scale_shift", 4)
            .or_else(|_| load_ada_layer_norm_single(&globals, "av_ca_video_scale_shift_adaln_single", 4))
            .ok();
        let av_a_ss = load_ada_layer_norm_single(&globals, "av_cross_attn_audio_scale_shift", 4)
            .or_else(|_| load_ada_layer_norm_single(&globals, "av_ca_audio_scale_shift_adaln_single", 4))
            .ok();
        let av_v_gate = load_ada_layer_norm_single(&globals, "av_cross_attn_video_a2v_gate", 1)
            .or_else(|_| load_ada_layer_norm_single(&globals, "av_ca_video_a2v_gate_adaln_single", 1))
            .or_else(|_| load_ada_layer_norm_single(&globals, "av_ca_a2v_gate_adaln_single", 1))
            .ok();
        let av_a_gate = load_ada_layer_norm_single(&globals, "av_cross_attn_audio_v2a_gate", 1)
            .or_else(|_| load_ada_layer_norm_single(&globals, "av_ca_audio_v2a_gate_adaln_single", 1))
            .or_else(|_| load_ada_layer_norm_single(&globals, "av_ca_v2a_gate_adaln_single", 1))
            .ok();

        let has_audio = audio_proj_in_w.is_some();
        if has_audio {
            log::info!("[LTX2] Audio globals loaded (audio_proj_in, audio_time_embed, etc.)");
        } else {
            log::info!("[LTX2] No audio globals found (video-only mode)");
        }

        Ok(Self {
            config: config.clone(),
            checkpoint_path: path.to_string(),
            proj_in_weight: pre_transpose_weight(&proj_in_w)?,
            proj_in_bias: proj_in_b,
            caption_projection: caption_proj,
            connector,
            audio_connector,
            prompt_adaln_single: prompt_adaln,
            audio_prompt_adaln_single: audio_prompt_adaln,
            aggregate_embed_weight: agg_w_t,
            aggregate_embed_bias: agg_b,
            audio_aggregate_embed_weight: audio_agg_w_t,
            audio_aggregate_embed_bias: audio_agg_b,
            time_embed: time_emb,
            scale_shift_table: get("scale_shift_table")?,
            proj_out_weight: pre_transpose_weight(&get("proj_out.weight")?)?,
            proj_out_bias: get("proj_out.bias")?,
            // Audio globals
            audio_proj_in_weight: audio_proj_in_w,
            audio_proj_in_bias: audio_proj_in_b,
            audio_caption_projection: audio_caption_proj,
            audio_time_embed: audio_time_emb,
            audio_scale_shift_table: audio_sst,
            audio_proj_out_weight: audio_proj_out_w,
            audio_proj_out_bias: audio_proj_out_b,
            av_cross_attn_video_scale_shift: av_v_ss,
            av_cross_attn_audio_scale_shift: av_a_ss,
            av_cross_attn_video_a2v_gate: av_v_gate,
            av_cross_attn_audio_v2a_gate: av_a_gate,
            block_cache: Vec::new(),
            swap: None,
            key_prefix: prefix,
            f32_cache: Vec::new(),
            fp8_blocks: Vec::new(),
        })
    }

    /// Load F32 tensors (scale_shift_table etc.) for a block from disk.
    /// These are kept in F32 to avoid BF16 truncation that causes divergence.
    fn load_block_f32_tensors(&self, block_idx: usize) -> Result<HashMap<String, Tensor>> {
        let device = flame_core::global_cuda_device();
        let key_prefix = &self.key_prefix;
        let pfx = format!("{key_prefix}transformer_blocks.{block_idx}.");

        let f32_tensors = flame_core::serialization::load_file_filtered(
            &self.checkpoint_path, &device,
            |key| key.starts_with(&pfx) && key.contains("scale_shift_table"),
        )?;

        // Strip prefix
        Ok(f32_tensors.into_iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix(key_prefix).unwrap_or(&k).to_string();
                (stripped, v)
            })
            .collect())
    }

    /// Load all block weights to GPU as raw FP8/BF16 bytes.
    /// ~12-16GB on GPU. No disk I/O during inference.
    pub fn load_fp8_resident(&mut self) -> Result<()> {
        use super::fp8_resident::{RawWeight, ResidentBlock};
        use cudarc::driver::DevicePtr;

        let device = flame_core::global_cuda_device();
        let t0 = std::time::Instant::now();
        let prefix = self.key_prefix.clone();
        let num_layers = self.config.num_layers;

        log::info!("[LTX2] Loading FP8 resident ({} blocks)...", num_layers);

        // Parse checkpoint header
        let file = std::fs::File::open(&self.checkpoint_path)
            .map_err(|e| flame_core::Error::Io(format!("open: {e}")))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| flame_core::Error::Io(format!("mmap: {e}")))?;
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])
            .map_err(|e| flame_core::Error::Io(format!("header parse: {e}")))?;
        let data_start = 8 + header_len;
        let meta = header.as_object().ok_or_else(||
            flame_core::Error::InvalidInput("Invalid header".into()))?;

        // Build scale map
        let mut scale_map: HashMap<String, f32> = HashMap::new();
        for (k, v) in meta {
            if k.ends_with("_scale") && v["shape"].as_array().map(|a| a.is_empty()).unwrap_or(false) {
                let offsets = v["data_offsets"].as_array().unwrap();
                let s = data_start + offsets[0].as_u64().unwrap() as usize;
                let scale = f32::from_le_bytes([mmap[s], mmap[s+1], mmap[s+2], mmap[s+3]]);
                let target = k[..k.len()-6].to_string();
                scale_map.insert(target, scale);
            }
        }

        let mut blocks = Vec::with_capacity(num_layers);
        let mut total_gpu_bytes = 0usize;

        for block_idx in 0..num_layers {
            let block_prefix = format!("{prefix}transformer_blocks.{block_idx}.");
            let mut weights = HashMap::new();
            let mut f32_tensors = HashMap::new();

            for (k, v) in meta {
                if k == "__metadata__" || !k.starts_with(&block_prefix) { continue; }
                if k.ends_with("_scale") || k.ends_with("input_scale") { continue; }

                // Skip only LARGE BF16 2D audio weight matrices from boundary blocks
                // (these are loaded per-block from disk during forward to avoid OOM).
                // Keep biases (1D), norms (1D), and F32 scale_shift_tables for all blocks.
                let stripped_key = k.strip_prefix(&block_prefix).unwrap_or(k);
                let is_audio_key = stripped_key.starts_with("audio_")
                    || stripped_key.starts_with("video_to_audio")
                    || stripped_key.starts_with("audio_to_video")
                    || stripped_key.contains("scale_shift_table_a2v")
                    || stripped_key.starts_with("audio_prompt_scale_shift");
                let dtype_str = v["dtype"].as_str().unwrap_or("?");
                let empty_arr = vec![];
                let shape_arr = v["shape"].as_array().unwrap_or(&empty_arr);
                let is_large_2d = shape_arr.len() == 2
                    && shape_arr.iter().all(|s| s.as_u64().unwrap_or(0) > 1);
                // Only skip large 2D BF16 audio weights (boundary block matrices).
                // Keep biases, norms, F32 tables, and FP8 weights.
                if is_audio_key && dtype_str == "BF16" && is_large_2d {
                    continue;
                }

                let dtype = v["dtype"].as_str().unwrap_or("?");
                let shape: Vec<usize> = v["shape"].as_array().unwrap_or(&vec![])
                    .iter().filter_map(|s| s.as_u64().map(|u| u as usize)).collect();
                let numel: usize = shape.iter().product();
                let offsets = v["data_offsets"].as_array().unwrap();
                let start = data_start + offsets[0].as_u64().unwrap() as usize;
                let end = data_start + offsets[1].as_u64().unwrap() as usize;
                let stripped = k.strip_prefix(&prefix).unwrap_or(k).to_string();

                match dtype {
                    "F8_E4M3" => {
                        let scale = scale_map.get(k).copied().unwrap_or(1.0);
                        let bytes = &mmap[start..end];
                        let gpu: cudarc::driver::CudaSlice<u8> = device.htod_copy(bytes.to_vec())
                            .map_err(|e| flame_core::Error::Cuda(format!("htod: {:?}", e)))?;
                        total_gpu_bytes += bytes.len();
                        weights.insert(stripped, RawWeight::FP8 {
                            data: gpu, shape, numel, scale,
                        });
                    }
                    "BF16" => {
                        // Load as BF16 Tensor, pre-transpose weight matrices
                        let bf16_data: Vec<u16> = mmap[start..end].chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect();
                        let mut tensor = Tensor::zeros_dtype(
                            Shape::from_dims(&shape), DType::BF16, device.clone(),
                        )?;
                        tensor.copy_from_bf16_slice(&bf16_data)?;
                        total_gpu_bytes += bf16_data.len() * 2;
                        // Pre-transpose 2D weight matrices at load time (once)
                        // so forward pass never calls transpose()
                        let tensor = if super::fp8_resident::needs_transpose(&stripped, &shape) {
                            tensor.transpose()?
                        } else {
                            tensor
                        };
                        weights.insert(stripped, RawWeight::BF16 { tensor });
                    }
                    "F32" => {
                        // Load F32 as proper Tensor (scale_shift_table etc.)
                        let data = &mmap[start..end];
                        let f32_data: Vec<f32> = data.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let tensor = Tensor::from_vec(
                            f32_data, Shape::from_dims(&shape), device.clone(),
                        )?;
                        f32_tensors.insert(stripped, tensor);
                    }
                    _ => {} // skip unsupported
                }
            }

            blocks.push(ResidentBlock { weights, f32_tensors });

            if (block_idx + 1) % 12 == 0 || block_idx == 0 || block_idx + 1 == num_layers {
                log::info!("[LTX2] Loaded block {}/{}, {:.1}GB GPU so far",
                    block_idx + 1, num_layers, total_gpu_bytes as f64 / 1e9);
            }
        }

        log::info!("[LTX2] FP8 resident ready: {} blocks, {:.2}GB GPU, {:.1}s",
            blocks.len(), total_gpu_bytes as f64 / 1e9, t0.elapsed().as_secs_f32());

        self.fp8_blocks = blocks;
        Ok(())
    }

    /// Drop FP8 resident weights to free GPU memory.
    /// Call before switching to FlameSwap for a higher-resolution stage.
    pub fn drop_fp8_resident(&mut self) {
        self.fp8_blocks.clear();
        log::info!("[LTX2] FP8 resident weights dropped");
    }

    /// Initialize FlameSwap for async double-buffered block streaming.
    /// Call after `load_globals()`. Replaces sync `load_block_from_disk`.
    pub fn init_swap(&mut self) -> Result<()> {
        let device = flame_core::global_cuda_device();
        let t0 = std::time::Instant::now();
        let prefix = self.key_prefix.clone();
        let num_layers = self.config.num_layers;

        log::info!("[LTX2] Initializing FlameSwap for {} blocks (prefix='{}')", num_layers, prefix);

        let swap = flame_swap::FlameSwap::load(
            &[&self.checkpoint_path],
            &device,
            |name| {
                let stripped = name.strip_prefix(&prefix).unwrap_or(name);
                if !stripped.starts_with("transformer_blocks.") { return None; }
                // Load ALL block weights including audio, A2V, V2A
                // Skip scale_shift_table (F32) — loaded separately to preserve precision
                if stripped.contains("scale_shift_table") { return None; }
                let rest = stripped.strip_prefix("transformer_blocks.")?;
                rest.split('.').next()?.parse().ok()
            },
        ).map_err(|e| flame_core::Error::Io(format!("FlameSwap load failed: {e}")))?;

        log::info!("[LTX2] FlameSwap ready: {} blocks, ~{:.2}GB pinned, {:.1}s",
            swap.num_blocks(), swap.pinned_bytes() as f64 / 1e9, t0.elapsed().as_secs_f32());

        // Pre-cache F32 tensors (scale_shift_table etc.) — one-time load, ~1.7MB total
        log::info!("[LTX2] Caching F32 block tensors...");
        let mut f32_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let pfx = format!("{prefix}transformer_blocks.{i}.");
            let f32_tensors = flame_core::serialization::load_file_filtered(
                &self.checkpoint_path, &device,
                |key| key.starts_with(&pfx) && key.contains("scale_shift_table"),
            )?;
            let stripped: HashMap<String, Tensor> = f32_tensors.into_iter()
                .map(|(k, v)| {
                    let s = k.strip_prefix(&prefix).unwrap_or(&k).to_string();
                    (s, v)
                })
                .collect();
            f32_cache.push(stripped);
        }
        log::info!("[LTX2] F32 cache: {} blocks, {:.1}s total init",
            f32_cache.len(), t0.elapsed().as_secs_f32());

        self.swap = Some(swap);
        self.f32_cache = f32_cache;
        Ok(())
    }

    /// Pre-load all block weights by reading each block once from disk.
    /// Subsequent load_block() calls use cached weights instead of re-parsing
    /// the 43GB checkpoint. Loads one block at a time to GPU to avoid OOM,
    /// caches the GPU tensors, then the next block reuses the same slot.
    ///
    /// Actually: since all 48 blocks don't fit in VRAM (~26GB > 24GB),
    /// we pre-load them one at a time and cache. On subsequent denoise steps,
    /// load_block reconstructs from cached maps (no disk re-read).
    pub fn preload_blocks(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        log::info!("[LTX2] Pre-loading all {} block weight maps...", self.config.num_layers);

        let mut cache: Vec<Option<HashMap<String, Tensor>>> = Vec::with_capacity(self.config.num_layers);
        for i in 0..self.config.num_layers {
            let block_weights = self.load_block_from_disk(i)?;
            cache.push(Some(block_weights));
            if (i + 1) % 12 == 0 {
                log::info!("[LTX2] Cached block weights {}/{}", i + 1, self.config.num_layers);
            }
        }

        self.block_cache = cache;
        log::info!("[LTX2] Block cache ready ({} blocks, {:.1}s total)",
            self.config.num_layers, t0.elapsed().as_secs_f32());
        Ok(())
    }

    /// Load raw block weight map from disk (slow path).
    fn load_block_from_disk(&self, block_idx: usize) -> Result<HashMap<String, Tensor>> {
        let device = flame_core::global_cuda_device();
        let key_prefix = detect_key_prefix(&self.checkpoint_path)?;
        let prefix = format!("{key_prefix}transformer_blocks.{block_idx}.");
        let raw_weights = flame_core::serialization::load_file_filtered(
            &self.checkpoint_path, &device,
            |key| key.starts_with(&prefix) && !key.contains("audio"),
        )?;
        Ok(raw_weights.into_iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix(&key_prefix).unwrap_or(&k).to_string();
                (stripped, v)
            })
            .collect())
    }

    /// Load a single block's video-only weights.
    /// Uses pre-loaded cache if available, otherwise reads from disk.
    pub fn load_block(&self, block_idx: usize) -> Result<LTX2TransformerBlock> {
        let block_weights = if block_idx < self.block_cache.len() {
            if let Some(ref cached) = self.block_cache[block_idx] {
                cached.clone()
            } else {
                self.load_block_from_disk(block_idx)?
            }
        } else {
            self.load_block_from_disk(block_idx)?
        };
        Self::load_block_from_weights_static(&self.config, block_idx, block_weights)
    }

    /// Build an LTX2TransformerBlock from a weight HashMap. Public for validation.
    /// Build block with optional F32 override tensors (borrowed, no clone).
    fn load_block_from_weights_with_overrides(
        config: &LTX2Config,
        block_idx: usize,
        mut block_weights: HashMap<String, Tensor>,
        f32_overrides: Option<&HashMap<String, Tensor>>,
    ) -> Result<LTX2TransformerBlock> {
        // Merge F32 overrides by borrowing — clone only the Tensor wrapper (cheap Arc if shared_storage,
        // or unavoidable deep copy without it — but F32 tensors are tiny: ~36KB each)
        if let Some(overrides) = f32_overrides {
            for (k, v) in overrides {
                block_weights.insert(k.clone(), v.clone());
            }
        }
        Self::load_block_from_weights_static(config, block_idx, block_weights)
    }

    pub fn load_block_from_weights_static(
        config: &LTX2Config,
        block_idx: usize,
        block_weights: HashMap<String, Tensor>,
    ) -> Result<LTX2TransformerBlock> {
        let device = flame_core::global_cuda_device();

        let eps = config.norm_eps;
        let num_heads = config.num_attention_heads;
        let head_dim = config.attention_head_dim;
        let ad = config.audio_inner_dim();

        let get_opt = |key: &str| -> Option<Tensor> { block_weights.get(key).cloned() };

        // Dummy tensors for audio fields
        let dummy_1d = |sz: usize| Tensor::zeros_dtype(
            Shape::from_dims(&[sz]), DType::BF16, device.clone(),
        );
        let dummy_2d = |r: usize, c: usize| Tensor::zeros_dtype(
            Shape::from_dims(&[r, c]), DType::BF16, device.clone(),
        );

        // Dummy weights are pre-transposed: [in, out] layout to match matmul_weight_t expectations
        let dummy_attn = || -> Result<LTX2Attention> {
            Ok(LTX2Attention {
                to_q_weight: dummy_2d(ad, ad)?, to_q_bias: dummy_1d(ad)?,
                to_k_weight: dummy_2d(ad, ad)?, to_k_bias: dummy_1d(ad)?,
                to_v_weight: dummy_2d(ad, ad)?, to_v_bias: dummy_1d(ad)?,
                norm_q_weight: dummy_1d(ad)?, norm_k_weight: dummy_1d(ad)?,
                to_out_weight: dummy_2d(ad, ad)?, to_out_bias: dummy_1d(ad)?,
                to_gate_logits_weight: None, to_gate_logits_bias: None,
                num_heads: config.audio_num_attention_heads,
                head_dim: config.audio_attention_head_dim, eps,
            })
        };

        let dummy_ff = || -> Result<FeedForward> {
            let affn = config.audio_ffn_hidden();
            Ok(FeedForward {
                gelu_proj_weight: dummy_2d(ad, affn)?,     // pre-transposed: [in=ad, out=affn]
                gelu_proj_bias: dummy_1d(affn)?,
                out_weight: dummy_2d(affn, ad)?,           // pre-transposed: [in=affn, out=ad]
                out_bias: dummy_1d(ad)?,
            })
        };

        let pfx = format!("transformer_blocks.{block_idx}");
        let a_heads = config.audio_num_attention_heads;
        let a_head_dim = config.audio_attention_head_dim;

        // Try loading audio weights; fall back to dummies if not present
        let has_audio = block_weights.contains_key(&format!("{pfx}.audio_attn1.to_q.weight"));

        let (audio_n1, audio_a1) = if has_audio {
            (get_opt(&format!("{pfx}.audio_norm1.weight")),
             load_attention(&block_weights, &format!("{pfx}.audio_attn1"), a_heads, a_head_dim, eps)?)
        } else { (None, dummy_attn()?) };

        let (audio_n2, audio_a2) = if has_audio {
            (get_opt(&format!("{pfx}.audio_norm2.weight")),
             load_attention(&block_weights, &format!("{pfx}.audio_attn2"), a_heads, a_head_dim, eps)?)
        } else { (None, dummy_attn()?) };

        // A2V/V2A use audio dimensions (cross_attention_dim = 2048)
        let (a2v_norm, a2v_attn) = if has_audio {
            (get_opt(&format!("{pfx}.audio_to_video_norm.weight")),
             load_attention(&block_weights, &format!("{pfx}.audio_to_video_attn"), a_heads, a_head_dim, eps)?)
        } else { (None, dummy_attn()?) };

        let (v2a_norm, v2a_attn) = if has_audio {
            (get_opt(&format!("{pfx}.video_to_audio_norm.weight")),
             load_attention(&block_weights, &format!("{pfx}.video_to_audio_attn"), a_heads, a_head_dim, eps)?)
        } else { (None, dummy_attn()?) };

        let (audio_n3, audio_ffn) = if has_audio {
            (get_opt(&format!("{pfx}.audio_norm3.weight")),
             load_feed_forward(&block_weights, &format!("{pfx}.audio_ff"))?)
        } else { (None, dummy_ff()?) };

        Ok(LTX2TransformerBlock {
            norm1_weight: get_opt(&format!("{pfx}.norm1.weight")),
            attn1: load_attention(&block_weights, &format!("{pfx}.attn1"), num_heads, head_dim, eps)?,
            audio_norm1_weight: audio_n1,
            audio_attn1: audio_a1,
            norm2_weight: get_opt(&format!("{pfx}.norm2.weight")),
            attn2: load_attention(&block_weights, &format!("{pfx}.attn2"), num_heads, head_dim, eps)?,
            audio_norm2_weight: audio_n2,
            audio_attn2: audio_a2,
            audio_to_video_norm_weight: a2v_norm,
            audio_to_video_attn: a2v_attn,
            video_to_audio_norm_weight: v2a_norm,
            video_to_audio_attn: v2a_attn,
            norm3_weight: get_opt(&format!("{pfx}.norm3.weight")),
            ff: load_feed_forward(&block_weights, &format!("{pfx}.ff"))?,
            audio_norm3_weight: audio_n3,
            audio_ff: audio_ffn,
            scale_shift_table: block_weights.get(&format!("{pfx}.scale_shift_table"))
                .cloned()
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing: {pfx}.scale_shift_table")))?,
            audio_scale_shift_table: block_weights.get(&format!("{pfx}.audio_scale_shift_table"))
                .cloned()
                .unwrap_or(dummy_2d(9, ad)?),
            video_a2v_cross_attn_scale_shift_table: block_weights.get(&format!("{pfx}.video_a2v_cross_attn_scale_shift_table"))
                .or_else(|| block_weights.get(&format!("{pfx}.scale_shift_table_a2v_ca_video")))
                .cloned()
                .unwrap_or(dummy_2d(5, config.inner_dim())?),
            audio_a2v_cross_attn_scale_shift_table: block_weights.get(&format!("{pfx}.audio_a2v_cross_attn_scale_shift_table"))
                .or_else(|| block_weights.get(&format!("{pfx}.scale_shift_table_a2v_ca_audio")))
                .cloned()
                .unwrap_or(dummy_2d(5, ad)?),
            prompt_scale_shift_table: get_opt(&format!("{pfx}.prompt_scale_shift_table")),
            audio_prompt_scale_shift_table: get_opt(&format!("{pfx}.audio_prompt_scale_shift_table")),
            eps,
        })
    }

    /// Like `load_block_from_weights_static` but assumes all 2D weight
    /// matrices are **already pre-transposed** to [in, out] layout.
    /// Skips all `pre_transpose_weight` calls → zero GPU transposes.
    pub fn load_block_from_weights_pretransposed(
        config: &LTX2Config,
        block_idx: usize,
        block_weights: HashMap<String, Tensor>,
    ) -> Result<LTX2TransformerBlock> {
        let device = flame_core::global_cuda_device();

        let eps = config.norm_eps;
        let num_heads = config.num_attention_heads;
        let head_dim = config.attention_head_dim;
        let ad = config.audio_inner_dim();

        let get_opt = |key: &str| -> Option<Tensor> { block_weights.get(key).cloned() };

        let dummy_1d = |sz: usize| Tensor::zeros_dtype(
            Shape::from_dims(&[sz]), DType::BF16, device.clone(),
        );
        let dummy_2d = |r: usize, c: usize| Tensor::zeros_dtype(
            Shape::from_dims(&[r, c]), DType::BF16, device.clone(),
        );

        let dummy_attn = || -> Result<LTX2Attention> {
            Ok(LTX2Attention {
                to_q_weight: dummy_2d(ad, ad)?, to_q_bias: dummy_1d(ad)?,
                to_k_weight: dummy_2d(ad, ad)?, to_k_bias: dummy_1d(ad)?,
                to_v_weight: dummy_2d(ad, ad)?, to_v_bias: dummy_1d(ad)?,
                norm_q_weight: dummy_1d(ad)?, norm_k_weight: dummy_1d(ad)?,
                to_out_weight: dummy_2d(ad, ad)?, to_out_bias: dummy_1d(ad)?,
                to_gate_logits_weight: None, to_gate_logits_bias: None,
                num_heads: config.audio_num_attention_heads,
                head_dim: config.audio_attention_head_dim, eps,
            })
        };

        let dummy_ff = || -> Result<FeedForward> {
            let affn = config.audio_ffn_hidden();
            Ok(FeedForward {
                gelu_proj_weight: dummy_2d(ad, affn)?,
                gelu_proj_bias: dummy_1d(affn)?,
                out_weight: dummy_2d(affn, ad)?,
                out_bias: dummy_1d(ad)?,
            })
        };

        let pfx = format!("transformer_blocks.{block_idx}");

        // Attention/FF loaders that skip pre_transpose_weight
        let load_attn_pt = |prefix: &str| -> Result<LTX2Attention> {
            let get = |key: &str| -> Result<Tensor> {
                block_weights.get(key).cloned().ok_or_else(||
                    flame_core::Error::InvalidInput(format!("Missing weight: {}", key)))
            };
            let gate_weight = block_weights.get(&format!("{prefix}.to_gate_logits.weight")).cloned();
            Ok(LTX2Attention {
                to_q_weight: get(&format!("{prefix}.to_q.weight"))?,
                to_q_bias: get(&format!("{prefix}.to_q.bias"))?,
                to_k_weight: get(&format!("{prefix}.to_k.weight"))?,
                to_k_bias: get(&format!("{prefix}.to_k.bias"))?,
                to_v_weight: get(&format!("{prefix}.to_v.weight"))?,
                to_v_bias: get(&format!("{prefix}.to_v.bias"))?,
                norm_q_weight: get(&format!("{prefix}.norm_q.weight"))
                    .or_else(|_| get(&format!("{prefix}.q_norm.weight")))?,
                norm_k_weight: get(&format!("{prefix}.norm_k.weight"))
                    .or_else(|_| get(&format!("{prefix}.k_norm.weight")))?,
                to_out_weight: get(&format!("{prefix}.to_out.0.weight"))?,
                to_out_bias: get(&format!("{prefix}.to_out.0.bias"))?,
                to_gate_logits_weight: gate_weight,
                to_gate_logits_bias: block_weights.get(&format!("{prefix}.to_gate_logits.bias")).cloned(),
                num_heads,
                head_dim,
                eps,
            })
        };

        let load_ff_pt = |prefix: &str| -> Result<FeedForward> {
            let get = |key: &str| -> Result<Tensor> {
                block_weights.get(key).cloned().ok_or_else(||
                    flame_core::Error::InvalidInput(format!("Missing weight: {}", key)))
            };
            Ok(FeedForward {
                gelu_proj_weight: get(&format!("{prefix}.net.0.proj.weight"))?,
                gelu_proj_bias: get(&format!("{prefix}.net.0.proj.bias"))?,
                out_weight: get(&format!("{prefix}.net.2.weight"))?,
                out_bias: get(&format!("{prefix}.net.2.bias"))?,
            })
        };

        // Audio attention loader (uses audio dimensions)
        let a_heads = config.audio_num_attention_heads;
        let a_head_dim = config.audio_attention_head_dim;
        let load_audio_attn_pt = |prefix: &str| -> Result<LTX2Attention> {
            let get = |key: &str| -> Result<Tensor> {
                block_weights.get(key).cloned().ok_or_else(||
                    flame_core::Error::InvalidInput(format!("Missing weight: {}", key)))
            };
            let gate_weight = block_weights.get(&format!("{prefix}.to_gate_logits.weight")).cloned();
            Ok(LTX2Attention {
                to_q_weight: get(&format!("{prefix}.to_q.weight"))?,
                to_q_bias: get(&format!("{prefix}.to_q.bias"))?,
                to_k_weight: get(&format!("{prefix}.to_k.weight"))?,
                to_k_bias: get(&format!("{prefix}.to_k.bias"))?,
                to_v_weight: get(&format!("{prefix}.to_v.weight"))?,
                to_v_bias: get(&format!("{prefix}.to_v.bias"))?,
                norm_q_weight: get(&format!("{prefix}.norm_q.weight"))
                    .or_else(|_| get(&format!("{prefix}.q_norm.weight")))?,
                norm_k_weight: get(&format!("{prefix}.norm_k.weight"))
                    .or_else(|_| get(&format!("{prefix}.k_norm.weight")))?,
                to_out_weight: get(&format!("{prefix}.to_out.0.weight"))?,
                to_out_bias: get(&format!("{prefix}.to_out.0.bias"))?,
                to_gate_logits_weight: gate_weight,
                to_gate_logits_bias: block_weights.get(&format!("{prefix}.to_gate_logits.bias")).cloned(),
                num_heads: a_heads,
                head_dim: a_head_dim,
                eps,
            })
        };

        let load_audio_ff_pt = |prefix: &str| -> Result<FeedForward> {
            let get = |key: &str| -> Result<Tensor> {
                block_weights.get(key).cloned().ok_or_else(||
                    flame_core::Error::InvalidInput(format!("Missing weight: {}", key)))
            };
            Ok(FeedForward {
                gelu_proj_weight: get(&format!("{prefix}.net.0.proj.weight"))?,
                gelu_proj_bias: get(&format!("{prefix}.net.0.proj.bias"))?,
                out_weight: get(&format!("{prefix}.net.2.weight"))?,
                out_bias: get(&format!("{prefix}.net.2.bias"))?,
            })
        };

        // Try loading audio weights; fall back to dummies if not present
        let has_audio = block_weights.contains_key(&format!("{pfx}.audio_attn1.to_q.weight"));

        let (audio_n1, audio_a1) = if has_audio {
            (get_opt(&format!("{pfx}.audio_norm1.weight")),
             load_audio_attn_pt(&format!("{pfx}.audio_attn1"))?)
        } else {
            (None, dummy_attn()?)
        };
        let (audio_n2, audio_a2) = if has_audio {
            (get_opt(&format!("{pfx}.audio_norm2.weight")),
             load_audio_attn_pt(&format!("{pfx}.audio_attn2"))?)
        } else {
            (None, dummy_attn()?)
        };
        // A2V/V2A cross-attention use cross_attention_dim (2048), not video dim
        // A2V: Q from video→2048, K/V from audio→2048, out→video 4096
        // V2A: Q from audio→2048, K/V from video→2048, out→audio 2048
        // Both use audio_num_heads (32) and audio_head_dim (64)
        let (a2v_norm, a2v_attn) = if has_audio {
            (get_opt(&format!("{pfx}.audio_to_video_norm.weight")),
             load_audio_attn_pt(&format!("{pfx}.audio_to_video_attn"))?)
        } else {
            (None, dummy_attn()?)
        };
        let (v2a_norm, v2a_attn) = if has_audio {
            (get_opt(&format!("{pfx}.video_to_audio_norm.weight")),
             load_audio_attn_pt(&format!("{pfx}.video_to_audio_attn"))?)
        } else {
            (None, dummy_attn()?)
        };
        let (audio_n3, audio_ffn) = if has_audio {
            (get_opt(&format!("{pfx}.audio_norm3.weight")),
             load_audio_ff_pt(&format!("{pfx}.audio_ff"))?)
        } else {
            (None, dummy_ff()?)
        };

        Ok(LTX2TransformerBlock {
            norm1_weight: get_opt(&format!("{pfx}.norm1.weight")),
            attn1: load_attn_pt(&format!("{pfx}.attn1"))?,
            audio_norm1_weight: audio_n1,
            audio_attn1: audio_a1,
            norm2_weight: get_opt(&format!("{pfx}.norm2.weight")),
            attn2: load_attn_pt(&format!("{pfx}.attn2"))?,
            audio_norm2_weight: audio_n2,
            audio_attn2: audio_a2,
            audio_to_video_norm_weight: a2v_norm,
            audio_to_video_attn: a2v_attn,
            video_to_audio_norm_weight: v2a_norm,
            video_to_audio_attn: v2a_attn,
            norm3_weight: get_opt(&format!("{pfx}.norm3.weight")),
            ff: load_ff_pt(&format!("{pfx}.ff"))?,
            audio_norm3_weight: audio_n3,
            audio_ff: audio_ffn,
            scale_shift_table: block_weights.get(&format!("{pfx}.scale_shift_table"))
                .cloned()
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing: {pfx}.scale_shift_table")))?,
            audio_scale_shift_table: block_weights.get(&format!("{pfx}.audio_scale_shift_table"))
                .cloned()
                .unwrap_or(dummy_2d(9, ad)?),
            video_a2v_cross_attn_scale_shift_table: block_weights.get(&format!("{pfx}.video_a2v_cross_attn_scale_shift_table"))
                .or_else(|| block_weights.get(&format!("{pfx}.scale_shift_table_a2v_ca_video")))
                .cloned()
                .unwrap_or(dummy_2d(5, config.inner_dim())?),
            audio_a2v_cross_attn_scale_shift_table: block_weights.get(&format!("{pfx}.audio_a2v_cross_attn_scale_shift_table"))
                .or_else(|| block_weights.get(&format!("{pfx}.scale_shift_table_a2v_ca_audio")))
                .cloned()
                .unwrap_or(dummy_2d(5, ad)?),
            prompt_scale_shift_table: get_opt(&format!("{pfx}.prompt_scale_shift_table")),
            audio_prompt_scale_shift_table: get_opt(&format!("{pfx}.audio_prompt_scale_shift_table")),
            eps,
        })
    }

    /// Video-only forward with block streaming from disk.
    ///
    /// Each block is loaded from the safetensors checkpoint, executed,
    /// and immediately dropped to free GPU memory. Only 1 block + global
    /// params are on GPU at any time (~1GB total).
    ///
    /// Supports both diffusers format (CaptionProjection, 6-param adaln)
    /// and ComfyUI format (VideoEmbeddingsConnector, 9-param adaln, prompt_adaln_single).
    pub fn forward_video_only(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        frame_rate: f32,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (batch_size, channels, num_frames, height, width) =
            (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
        let inner_dim = self.config.inner_dim();
        let num_tokens = num_frames * height * width;

        // 1. Patchify: [B, C, F, H, W] -> [B, F*H*W, C]
        let x_flat = x.reshape(&[batch_size, channels, num_tokens])?
            .permute(&[0, 2, 1])?;

        // 2. Build coordinate grid for RoPE
        let device = x.device().clone();
        let vae_sf = &self.config.vae_scale_factors;
        let coords = build_video_coords(
            batch_size, num_frames, height, width,
            vae_sf, self.config.causal_offset, frame_rate, device.clone(),
        )?;

        // 3. Compute RoPE
        let max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let (v_cos, v_sin) = compute_rope_frequencies(
            &coords, inner_dim, &max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;

        // 4. Input projection
        let mut hs = linear3d(&x_flat, &self.proj_in_weight, Some(&self.proj_in_bias))?;

        // 5. Timestep conditioning
        let num_mod_params = self.time_embed.num_mod_params;
        log::info!("[LTX2] timestep dtype={:?}, x dtype={:?}", timestep.dtype(), x.dtype());
        let ts_expanded = timestep.unsqueeze(1)?.expand(&[batch_size, num_tokens])?;
        let ts_scaled = ts_expanded.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
        let ts_flat = ts_scaled.reshape(&[batch_size * num_tokens])?;
        log::info!("[LTX2] ts_flat dtype={:?}", ts_flat.dtype());
        let (v_timestep, v_embedded) = self.time_embed.forward(&ts_flat)?;
        log::info!("[LTX2] v_timestep dtype={:?}, v_embedded dtype={:?}", v_timestep.dtype(), v_embedded.dtype());
        let v_timestep = v_timestep.reshape(&[batch_size, num_tokens, num_mod_params * inner_dim])?;
        let v_embedded = v_embedded.reshape(&[batch_size, num_tokens, inner_dim])?;

        // 6. Text embedding: skip aggregate_embed + connector if already 4096-dim
        let context_dim = context.shape().dims()[2];
        let enc_hs = if context_dim == inner_dim {
            log::info!("[LTX2] Pre-processed 4096-dim embeddings, skipping connector");
            context.clone()
        } else {
            let context_projected = if let (Some(w), Some(b)) =
                (&self.aggregate_embed_weight, &self.aggregate_embed_bias)
            {
                let rescale = ((inner_dim as f64) / (context_dim as f64)).sqrt() as f32;
                linear3d(&context.mul_scalar(rescale)?, w, Some(b))?
            } else {
                context.clone()
            };
            if let Some(connector) = &self.connector {
                connector.forward(&context_projected, encoder_attention_mask)?
            } else if let Some(caption_proj) = &self.caption_projection {
                caption_proj.forward(&context_projected)?
            } else {
                return Err(flame_core::Error::InvalidInput(
                    "No text projection available".into(),
                ));
            }
        };

        // 6b. Compute prompt_timestep from prompt_adaln_single (ComfyUI 9-param path)
        log::info!("[LTX2] enc_hs dtype={:?} shape={:?}", enc_hs.dtype(), enc_hs.dims());
        let prompt_timestep = if let Some(prompt_adaln) = &self.prompt_adaln_single {
            let text_seq_len = enc_hs.shape().dims()[1];
            // Use same timestep but for text positions
            let prompt_ts = timestep.unsqueeze(1)?.expand(&[batch_size, text_seq_len])?;
            let prompt_ts_scaled = prompt_ts.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
            let prompt_ts_flat = prompt_ts_scaled.reshape(&[batch_size * text_seq_len])?;
            let (prompt_mod, _) = prompt_adaln.forward(&prompt_ts_flat)?;
            log::info!("[LTX2] prompt_mod dtype={:?} shape={:?}", prompt_mod.dtype(), prompt_mod.dims());
            // prompt_mod: [B*seq, 2*dim] -> [B, seq, 2*dim]
            let prompt_mod = prompt_mod.reshape(&[batch_size, text_seq_len, 2 * inner_dim])?;
            Some(prompt_mod)
        } else {
            None
        };

        // 7. Stream blocks
        let num_layers = self.config.num_layers;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;
        let config_clone = self.config.clone();
        let checkpoint_path = self.checkpoint_path.clone();

        if !self.fp8_blocks.is_empty() {
            // FP8 resident path: dequant each block on-the-fly, no disk I/O.
            // Uses persistent BF16 buffers to eliminate per-block GPU allocation.
            log::info!("[LTX2] FP8 resident forward ({} blocks, persistent bufs)", num_layers);
            // Use inner block for buffer sizing (boundary blocks 0-3,47 may have fewer FP8 weights)
            let buf_idx = if num_layers > 5 { 5 } else { 0 };
            let persistent = super::fp8_resident::PersistentBlockBuf::new(
                &self.fp8_blocks[buf_idx], &device,
            )?;
            for i in 0..num_layers {
                let t_block = std::time::Instant::now();
                let block_weights = self.fp8_blocks[i].to_bf16_block_reuse(&persistent, &device)?;
                let t_dequant = t_block.elapsed().as_millis();
                let block = Self::load_block_from_weights_pretransposed(&config_clone, i, block_weights)?;
                let t_load = t_block.elapsed().as_millis();
                hs = block.forward_video_only(
                    &hs, &enc_hs, &v_timestep,
                    Some((&v_cos, &v_sin)),
                    None,
                    prompt_timestep.as_ref(),
                )?;
                drop(block);
                let t_total = t_block.elapsed().as_millis();
                if (i + 1) % 12 == 0 || i + 1 == num_layers || i == 0 {
                    log::info!("[LTX2] Block {}/{}: dequant={}ms, build={}ms, forward={}ms, total={}ms",
                        i + 1, num_layers, t_dequant, t_load - t_dequant, t_total - t_load, t_total);
                }
            }
        } else if self.swap.is_some() {
            // Async path: FlameSwap prefetch/await
            let swap = self.swap.as_mut().unwrap();
            let key_prefix = &self.key_prefix;
            log::info!("[LTX2] Block streaming via FlameSwap ({} blocks)", num_layers);

            swap.prefetch(0)
                .map_err(|e| flame_core::Error::Io(format!("prefetch: {e}")))?;
            for i in 0..num_layers {
                let t_block = std::time::Instant::now();
                let raw_weights = swap.await_block(i)
                    .map_err(|e| flame_core::Error::Io(format!("await_block: {e}")))?;
                if i + 1 < num_layers {
                    swap.prefetch(i + 1)
                        .map_err(|e| flame_core::Error::Io(format!("prefetch: {e}")))?;
                }
                // Strip key prefix
                let mut block_weights: HashMap<String, Tensor> = raw_weights.into_iter()
                    .map(|(k, v)| {
                        let stripped = k.strip_prefix(key_prefix).unwrap_or(&k).to_string();
                        (stripped, v)
                    })
                    .collect();
                // Merge F32 tensors from cache
                let n_f32 = if i < self.f32_cache.len() {
                    let cache = &self.f32_cache[i];
                    for (k, v) in cache {
                        block_weights.insert(k.clone(), v.clone());
                    }
                    cache.len()
                } else { 0 };
                if i == 0 {
                    eprintln!("[DEBUG] Block 0: {} swap + {} f32 = {} total. Has SST: {}",
                        block_weights.len() - n_f32, n_f32, block_weights.len(),
                        block_weights.contains_key("transformer_blocks.0.scale_shift_table"));
                }
                let t_load = t_block.elapsed().as_millis();
                let block = Self::load_block_from_weights_static(&config_clone, i, block_weights)?;
                hs = block.forward_video_only(
                    &hs, &enc_hs, &v_timestep,
                    Some((&v_cos, &v_sin)),
                    None,
                    prompt_timestep.as_ref(),
                )?;
                drop(block); // Free transposed weights immediately
                let t_total = t_block.elapsed().as_millis();
                if (i + 1) % 12 == 0 || i + 1 == num_layers || i == 0 {
                    log::info!("[LTX2] Block {}/{}: load={}ms, forward={}ms, total={}ms",
                        i + 1, num_layers, t_load, t_total - t_load, t_total);
                }
            }
        } else {
            // Sync fallback: load from disk
            log::info!("[LTX2] Block streaming from disk ({} blocks)", num_layers);
            for i in 0..num_layers {
                let t_block = std::time::Instant::now();
                let block = self.load_block(i)?;
                let t_load = t_block.elapsed().as_millis();
                hs = block.forward_video_only(
                    &hs, &enc_hs, &v_timestep,
                    Some((&v_cos, &v_sin)),
                    None,
                    prompt_timestep.as_ref(),
                )?;
                if i < 3 || i == num_layers - 1 {
                    if let Ok(hd) = hs.to_vec() {
                        let mean = hd.iter().sum::<f32>() / hd.len().max(1) as f32;
                        log::info!("[SYNC] Block {} hs: mean={:.6} first=[{:.4},{:.4},{:.4}]",
                            i, mean, hd.get(0).unwrap_or(&0.0), hd.get(1).unwrap_or(&0.0), hd.get(2).unwrap_or(&0.0));
                    }
                }
                drop(block);
                let t_total = t_block.elapsed().as_millis();
                if (i + 1) % 12 == 0 || i + 1 == num_layers || i == 0 {
                    log::info!("[LTX2] Block {}/{}: load={}ms, forward={}ms, total={}ms",
                        i + 1, num_layers, t_load, t_total - t_load, t_total);
                }
            }
        }

        // 8. Output
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?;
        let emb_4d = v_embedded.unsqueeze(2)?;
        let final_ss = shift_scale.add(&emb_4d)?.to_dtype(DType::BF16)?;
        let shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let normed = layer_norm_no_affine(&hs, self.config.norm_eps)?;
        let output = normed.mul(&scale.add_scalar(1.0)?.to_dtype(DType::BF16)?)?.add(&shift)?;
        let output = linear3d(&output, &self.proj_out_weight, Some(&self.proj_out_bias))?;

        // 9. Unpatchify
        let output = output.permute(&[0, 2, 1])?;
        output.reshape(&[batch_size, channels, num_frames, height, width])
    }

    /// Audio+video forward pass with FP8 resident block streaming.
    /// Returns (video_output, audio_output) velocity tensors.
    ///
    /// `audio_x`: [B, audio_channels, audio_frames] (NOT 5D — audio has no spatial dims)
    /// `audio_context`: [B, seq_len, caption_channels] text embeddings for audio
    pub fn forward_audio_video(
        &mut self,
        x: &Tensor,              // [B, C, F, H, W] video latent
        audio_x: &Tensor,        // [B, audio_C, audio_T, audio_F] audio latent
        timestep: &Tensor,       // [B] sigma
        context: &Tensor,        // [B, seq, caption_channels] text for video
        audio_context: &Tensor,  // [B, seq, caption_channels] text for audio
        frame_rate: f32,
        encoder_attention_mask: Option<&Tensor>,        // [B, 1, 1, seq] additive mask (video text)
        audio_encoder_attention_mask: Option<&Tensor>,  // [B, 1, 1, seq] additive mask (audio text)
    ) -> Result<(Tensor, Tensor)> {
        // Verify audio globals are loaded
        let audio_proj_in_w = self.audio_proj_in_weight.as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("Audio globals not loaded".into()))?;
        let audio_proj_in_b = self.audio_proj_in_bias.as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing audio_proj_in_bias".into()))?;
        let audio_time_emb = self.audio_time_embed.as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing audio_time_embed".into()))?;
        // audio_caption_projection is optional — if audio context is already
        // at audio_inner_dim (2048), it can be used directly.
        let audio_sst = self.audio_scale_shift_table.as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing audio_scale_shift_table".into()))?;
        let audio_proj_out_w = self.audio_proj_out_weight.as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing audio_proj_out_weight".into()))?;
        let audio_proj_out_b = self.audio_proj_out_bias.as_ref();

        let x_dims = x.shape().dims().to_vec();
        let (batch_size, channels, num_frames, height, width) =
            (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
        let inner_dim = self.config.inner_dim();
        let audio_inner_dim = self.config.audio_inner_dim();
        let num_video_tokens = num_frames * height * width;

        let audio_dims = audio_x.shape().dims().to_vec();
        let audio_channels = audio_dims[1];  // 8
        let audio_frames = audio_dims[2];     // T
        let audio_freq = audio_dims[3];       // 16 (mel bins)
        let num_audio_tokens = audio_frames;  // Each frame is one token (C*F flattened)

        let device = x.device().clone();

        // 1. Patchify video: [B, C, F, H, W] → [B, F*H*W, C]
        let v_flat = x.reshape(&[batch_size, channels, num_video_tokens])?
            .permute(&[0, 2, 1])?;

        // Patchify audio: [B, C, T, F] → [B, T, C*F]
        // Audio VAE outputs [B, 8, T, 16]. The model expects C*F=128 input channels.
        let audio_in_features = audio_channels * audio_freq; // 8 * 16 = 128
        let a_flat = audio_x.permute(&[0, 2, 1, 3])?  // [B, T, C, F]
            .reshape(&[batch_size, audio_frames, audio_in_features])?; // [B, T, 128]

        // 2. Build RoPE coords
        let vae_sf = &self.config.vae_scale_factors;
        let video_coords = build_video_coords(
            batch_size, num_frames, height, width,
            vae_sf, self.config.causal_offset, frame_rate, device.clone(),
        )?;

        // Audio coords: temporal only [B, 1, audio_tokens, 2]
        let audio_coords = build_audio_coords(
            batch_size, audio_frames, audio_freq,
            self.config.audio_scale_factor, vae_sf[0],
            self.config.causal_offset, frame_rate, device.clone(),
        )?;

        // 3. Compute RoPE
        let video_max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let audio_max_pos = [self.config.pos_embed_max_pos as f64];

        let (v_cos, v_sin) = compute_rope_frequencies(
            &video_coords, inner_dim, &video_max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;
        let (a_cos, a_sin) = compute_rope_frequencies(
            &audio_coords, audio_inner_dim, &audio_max_pos,
            self.config.rope_theta, self.config.audio_num_attention_heads,
        )?;

        // Cross-attention RoPE (temporal only)
        let video_temporal_coords = video_coords.narrow(1, 0, 1)?;
        let ca_audio_dim = self.config.audio_cross_attention_dim;
        let ca_max_pos = [self.config.pos_embed_max_pos as f64];
        let (ca_v_cos, ca_v_sin) = compute_rope_frequencies(
            &video_temporal_coords, ca_audio_dim, &ca_max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;
        let audio_temporal_coords = audio_coords.narrow(1, 0, 1)?;
        let (ca_a_cos, ca_a_sin) = compute_rope_frequencies(
            &audio_temporal_coords, ca_audio_dim, &ca_max_pos,
            self.config.rope_theta, self.config.audio_num_attention_heads,
        )?;

        // 4. Input projections
        let mut hs = linear3d(&v_flat, &self.proj_in_weight, Some(&self.proj_in_bias))?;
        let mut ahs = linear3d(&a_flat, audio_proj_in_w, Some(audio_proj_in_b))?;

        // Pre-block audio dump for LTX2_DUMP_AUDIO (appended to the output
        // dump once both phases run).
        let ahs_after_projin = if std::env::var("LTX2_DUMP_AUDIO").is_ok() {
            Some(ahs.clone())
        } else {
            None
        };
        let a_flat_input = if std::env::var("LTX2_DUMP_AUDIO").is_ok() {
            Some(a_flat.clone())
        } else {
            None
        };

        // 5. Timestep embeddings
        let num_mod_params = self.time_embed.num_mod_params;
        let ts_expanded = timestep.unsqueeze(1)?.expand(&[batch_size, num_video_tokens])?;
        let ts_scaled = ts_expanded.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
        let ts_flat = ts_scaled.reshape(&[batch_size * num_video_tokens])?;
        let (v_timestep, v_embedded) = self.time_embed.forward(&ts_flat)?;
        let v_timestep = v_timestep.reshape(&[batch_size, num_video_tokens, num_mod_params * inner_dim])?;
        let v_embedded = v_embedded.reshape(&[batch_size, num_video_tokens, inner_dim])?;

        // Audio timestep
        let audio_num_mod = audio_time_emb.num_mod_params;
        let ats_expanded = timestep.unsqueeze(1)?.expand(&[batch_size, num_audio_tokens])?;
        let ats_scaled = ats_expanded.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
        let ats_flat = ats_scaled.reshape(&[batch_size * num_audio_tokens])?;
        let (a_timestep, a_embedded) = audio_time_emb.forward(&ats_flat)?;
        let a_timestep = a_timestep.reshape(&[batch_size, num_audio_tokens, audio_num_mod * audio_inner_dim])?;
        let a_embedded = a_embedded.reshape(&[batch_size, num_audio_tokens, audio_inner_dim])?;

        // 6. AV cross-attention global modulation
        let cross_gate_scale = (self.config.cross_attn_timestep_scale_multiplier
            / self.config.timestep_scale_multiplier) as f32;

        let v_ca_ss = if let Some(ref adaln) = self.av_cross_attn_video_scale_shift {
            let (ss, _) = adaln.forward(&ts_flat.narrow(0, 0, batch_size)?)?;
            ss.reshape(&[batch_size, 1, ss.shape().dims()[ss.shape().rank() - 1]])?
        } else {
            Tensor::zeros_dtype(Shape::from_dims(&[batch_size, 1, 4 * inner_dim]), DType::BF16, device.clone())?
        };

        let v_ca_gate = if let Some(ref adaln) = self.av_cross_attn_video_a2v_gate {
            let scaled_ts = ts_flat.narrow(0, 0, batch_size)?.mul_scalar(cross_gate_scale)?;
            let (g, _) = adaln.forward(&scaled_ts)?;
            g.reshape(&[batch_size, 1, g.shape().dims()[g.shape().rank() - 1]])?
        } else {
            Tensor::zeros_dtype(Shape::from_dims(&[batch_size, 1, inner_dim]), DType::BF16, device.clone())?
        };

        let a_ca_ss = if let Some(ref adaln) = self.av_cross_attn_audio_scale_shift {
            let (ss, _) = adaln.forward(&ats_flat.narrow(0, 0, batch_size)?)?;
            ss.reshape(&[batch_size, 1, ss.shape().dims()[ss.shape().rank() - 1]])?
        } else {
            Tensor::zeros_dtype(Shape::from_dims(&[batch_size, 1, 4 * audio_inner_dim]), DType::BF16, device.clone())?
        };

        let a_ca_gate = if let Some(ref adaln) = self.av_cross_attn_audio_v2a_gate {
            let scaled_ats = ats_flat.narrow(0, 0, batch_size)?.mul_scalar(cross_gate_scale)?;
            let (g, _) = adaln.forward(&scaled_ats)?;
            g.reshape(&[batch_size, 1, g.shape().dims()[g.shape().rank() - 1]])?
        } else {
            Tensor::zeros_dtype(Shape::from_dims(&[batch_size, 1, audio_inner_dim]), DType::BF16, device.clone())?
        };

        // 7. Text embeddings.
        // Three accepted input formats per modality:
        //   (a) already inner_dim: pre-processed context, used directly.
        //   (b) packed Gemma 188160-dim: project via aggregate_embed, then
        //       run through embeddings_connector (Q-Former).
        //   (c) feature_extractor 4096/2048 features: skip aggregate_embed
        //       and feed directly into the connector.
        let video_context_dim = context.shape().dims()[2];
        let enc_hs = if video_context_dim == inner_dim && self.connector.is_none() {
            context.clone()
        } else if let Some(connector) = &self.connector {
            let projected = if video_context_dim == inner_dim {
                context.clone()
            } else if let (Some(w), Some(b)) = (
                &self.aggregate_embed_weight,
                &self.aggregate_embed_bias,
            ) {
                let rescale = ((inner_dim as f64) / (video_context_dim as f64)).sqrt() as f32;
                linear3d(&context.mul_scalar(rescale)?, w, Some(b))?
            } else {
                return Err(flame_core::Error::InvalidInput(format!(
                    "Video context dim {} != inner_dim {} and no aggregate_embed loaded",
                    video_context_dim, inner_dim,
                )));
            };
            connector.forward(&projected, encoder_attention_mask)?
        } else if let Some(ref cap_proj) = self.caption_projection {
            cap_proj.forward(context)?
        } else {
            return Err(flame_core::Error::InvalidInput("No video text projection".into()));
        };

        let audio_context_dim = audio_context.shape().dims()[2];
        let audio_enc_hs = if audio_context_dim == audio_inner_dim && self.audio_connector.is_none() {
            // Already projected to audio_inner_dim (e.g. from feature extractor or cache)
            audio_context.clone()
        } else if let Some(connector) = &self.audio_connector {
            let projected = if audio_context_dim == audio_inner_dim {
                audio_context.clone()
            } else if let (Some(w), Some(b)) = (
                &self.audio_aggregate_embed_weight,
                &self.audio_aggregate_embed_bias,
            ) {
                let rescale = ((audio_inner_dim as f64) / (audio_context_dim as f64)).sqrt() as f32;
                linear3d(&audio_context.mul_scalar(rescale)?, w, Some(b))?
            } else {
                return Err(flame_core::Error::InvalidInput(format!(
                    "Audio context dim {} != audio_inner_dim {} and no audio_aggregate_embed loaded",
                    audio_context_dim, audio_inner_dim,
                )));
            };
            connector.forward(&projected, audio_encoder_attention_mask)?
        } else if let Some(ref cap_proj) = self.audio_caption_projection {
            cap_proj.forward(audio_context)?
        } else {
            return Err(flame_core::Error::InvalidInput(format!(
                "Audio context dim {} != audio_inner_dim {}, and no audio path available",
                audio_context_dim, audio_inner_dim)));
        };

        // 7b. Compute prompt_timestep tensors for cross-attention KV modulation
        // (LTX2_AV_BUGS.md Bug 1 + 3). These are produced by the prompt_adaln_single
        // / audio_prompt_adaln_single AdaLayerNormSingle layers and threaded into
        // each block forward, where they're combined with the per-block
        // prompt_scale_shift_table to modulate the text encoder hidden states
        // before cross-attention.
        let video_prompt_ts: Option<Tensor> = if let Some(ref padaln) = self.prompt_adaln_single {
            let text_seq_len = enc_hs.shape().dims()[1];
            let prompt_ts = timestep.unsqueeze(1)?.expand(&[batch_size, text_seq_len])?;
            let prompt_ts_scaled = prompt_ts.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
            let prompt_ts_flat = prompt_ts_scaled.reshape(&[batch_size * text_seq_len])?;
            let (prompt_mod, _) = padaln.forward(&prompt_ts_flat)?;
            // prompt_mod is [B*seq, 2*inner_dim] — reshape to [B, seq, 2*inner_dim]
            Some(prompt_mod.reshape(&[batch_size, text_seq_len, 2 * inner_dim])?)
        } else {
            None
        };
        let audio_prompt_ts: Option<Tensor> = if let Some(ref apadaln) = self.audio_prompt_adaln_single {
            let text_seq_len = audio_enc_hs.shape().dims()[1];
            let prompt_ts = timestep.unsqueeze(1)?.expand(&[batch_size, text_seq_len])?;
            let prompt_ts_scaled = prompt_ts.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
            let prompt_ts_flat = prompt_ts_scaled.reshape(&[batch_size * text_seq_len])?;
            let (prompt_mod, _) = apadaln.forward(&prompt_ts_flat)?;
            Some(prompt_mod.reshape(&[batch_size, text_seq_len, 2 * audio_inner_dim])?)
        } else {
            None
        };
        if video_prompt_ts.is_some() && audio_prompt_ts.is_none() {
            log::warn!("[LTX2 AV] video prompt_timestep loaded but audio_prompt_adaln_single missing — audio CA will be unmodulated");
        }

        // 8. Stream blocks (FP8 resident or FlameSwap)
        let num_layers = self.config.num_layers;
        let config_clone = self.config.clone();

        if !self.fp8_blocks.is_empty() {
            // FP8 resident path: video weights on GPU, audio weights loaded per-block from disk
            log::info!("[LTX2] AV FP8 resident forward ({} blocks, audio from disk)", num_layers);
            // Use inner block (not boundary) for buffer sizing — inner blocks have FP8 audio
            let buf_idx = if num_layers > 5 { 5 } else { 0 };
            let persistent = super::fp8_resident::PersistentBlockBuf::new(
                &self.fp8_blocks[buf_idx], &device,
            )?;

            for i in 0..num_layers {
                let t_block = std::time::Instant::now();
                let mut block_weights = self.fp8_blocks[i].to_bf16_block_reuse(&persistent, &device)?;

                // Boundary blocks (0-3, 47) have BF16 audio/A2V/V2A weights that
                // aren't in fp8_blocks (skipped to avoid OOM). Load from disk.
                let has_audio = block_weights.contains_key(
                    &format!("transformer_blocks.{i}.audio_attn1.to_q.weight"));
                if !has_audio {
                    let key_prefix = detect_key_prefix(&self.checkpoint_path)?;
                    let pfx = format!("{key_prefix}transformer_blocks.{i}.");
                    let audio_weights = flame_core::serialization::load_file_filtered(
                        &self.checkpoint_path, &device,
                        |key| {
                            if !key.starts_with(&pfx) { return false; }
                            let stripped = key.strip_prefix(&pfx).unwrap_or(key);
                            stripped.starts_with("audio_")
                                || stripped.starts_with("video_to_audio")
                                || stripped.starts_with("audio_to_video")
                                || stripped.contains("scale_shift_table")
                        },
                    )?;
                    for (k, v) in audio_weights {
                        let stripped = k.strip_prefix(&key_prefix).unwrap_or(&k).to_string();
                        // Pre-transpose 2D weight matrices
                        let shape = v.shape().dims().to_vec();
                        let v = if super::fp8_resident::needs_transpose(&stripped, &shape) {
                            v.transpose()?
                        } else {
                            v
                        };
                        block_weights.insert(stripped, v);
                    }
                    log::info!("[LTX2] Block {}: loaded BF16 audio weights from disk", i);
                }

                let block = Self::load_block_from_weights_pretransposed(&config_clone, i, block_weights)?;
                let (new_hs, new_ahs) = block.forward(
                    &hs, &ahs,
                    &enc_hs, &audio_enc_hs,
                    &v_timestep, &a_timestep,
                    &v_ca_ss, &a_ca_ss,
                    &v_ca_gate, &a_ca_gate,
                    Some((&v_cos, &v_sin)),
                    Some((&a_cos, &a_sin)),
                    Some((&ca_v_cos, &ca_v_sin)),
                    Some((&ca_a_cos, &ca_a_sin)),
                    None, None,
                    video_prompt_ts.as_ref(),
                    audio_prompt_ts.as_ref(),
                )?;
                hs = new_hs;
                ahs = new_ahs;
                drop(block);
                if (i + 1) % 12 == 0 || i + 1 == num_layers || i < 4 || i == num_layers - 1 {
                    log::info!("[LTX2] AV Block {}/{}: {}ms",
                        i + 1, num_layers, t_block.elapsed().as_millis());
                }
            }
        } else if self.swap.is_some() {
            // FlameSwap path
            let swap = self.swap.as_mut().unwrap();
            let key_prefix = &self.key_prefix;
            log::info!("[LTX2] AV FlameSwap forward ({} blocks)", num_layers);

            swap.prefetch(0)
                .map_err(|e| flame_core::Error::Io(format!("prefetch: {e}")))?;
            let prof = std::env::var("LTX2_BLOCK_PROF").is_ok();
            for i in 0..num_layers {
                let t_block = std::time::Instant::now();
                let raw_weights = swap.await_block(i)
                    .map_err(|e| flame_core::Error::Io(format!("await_block: {e}")))?;
                if prof { let _ = device.synchronize(); }
                let t_after_await = t_block.elapsed().as_millis();

                if i + 1 < num_layers {
                    swap.prefetch(i + 1)
                        .map_err(|e| flame_core::Error::Io(format!("prefetch: {e}")))?;
                }
                let mut block_weights: HashMap<String, Tensor> = raw_weights.into_iter()
                    .map(|(k, v)| {
                        let stripped = k.strip_prefix(key_prefix).unwrap_or(&k).to_string();
                        (stripped, v)
                    })
                    .collect();
                // Merge F32 tensors from cache
                if i < self.f32_cache.len() {
                    for (k, v) in &self.f32_cache[i] {
                        block_weights.insert(k.clone(), v.clone());
                    }
                }
                let block = Self::load_block_from_weights_static(&config_clone, i, block_weights)?;
                if prof { let _ = device.synchronize(); }
                let t_after_build = t_block.elapsed().as_millis();

                // Dump block 0 inputs ONCE per process for python_block0_forward.py to diff against.
                if i == 0 && std::env::var("LTX2_DUMP_BLOCK0").is_ok() {
                    static DUMPED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                    if DUMPED.set(()).is_ok() {
                        let _ = device.synchronize();
                        let mut dump: HashMap<String, Tensor> = HashMap::new();
                        dump.insert("hs".into(), hs.clone());
                        dump.insert("ahs".into(), ahs.clone());
                        dump.insert("enc_hs".into(), enc_hs.clone());
                        dump.insert("audio_enc_hs".into(), audio_enc_hs.clone());
                        dump.insert("v_timestep".into(), v_timestep.clone());
                        dump.insert("a_timestep".into(), a_timestep.clone());
                        dump.insert("v_ca_ss".into(), v_ca_ss.clone());
                        dump.insert("a_ca_ss".into(), a_ca_ss.clone());
                        dump.insert("v_ca_gate".into(), v_ca_gate.clone());
                        dump.insert("a_ca_gate".into(), a_ca_gate.clone());
                        dump.insert("v_cos".into(), v_cos.clone());
                        dump.insert("v_sin".into(), v_sin.clone());
                        dump.insert("a_cos".into(), a_cos.clone());
                        dump.insert("a_sin".into(), a_sin.clone());
                        dump.insert("ca_v_cos".into(), ca_v_cos.clone());
                        dump.insert("ca_v_sin".into(), ca_v_sin.clone());
                        dump.insert("ca_a_cos".into(), ca_a_cos.clone());
                        dump.insert("ca_a_sin".into(), ca_a_sin.clone());
                        if let Some(ref t) = video_prompt_ts {
                            dump.insert("video_prompt_ts".into(), t.clone());
                        }
                        if let Some(ref t) = audio_prompt_ts {
                            dump.insert("audio_prompt_ts".into(), t.clone());
                        }
                        flame_core::serialization::save_tensors(
                            &dump,
                            std::path::Path::new("/home/alex/EriDiffusion/inference-flame/output/rust_block0_dump.safetensors"),
                            flame_core::serialization::SerializationFormat::SafeTensors,
                        )?;
                        log::info!("[LTX2 DUMP] Saved block 0 inputs to rust_block0_dump.safetensors");
                    }
                }

                let (new_hs, new_ahs) = block.forward(
                    &hs, &ahs,
                    &enc_hs, &audio_enc_hs,
                    &v_timestep, &a_timestep,
                    &v_ca_ss, &a_ca_ss,
                    &v_ca_gate, &a_ca_gate,
                    Some((&v_cos, &v_sin)),
                    Some((&a_cos, &a_sin)),
                    Some((&ca_v_cos, &ca_v_sin)),
                    Some((&ca_a_cos, &ca_a_sin)),
                    None, None,
                    video_prompt_ts.as_ref(),
                    audio_prompt_ts.as_ref(),
                )?;
                if prof { let _ = device.synchronize(); }
                let t_after_forward = t_block.elapsed().as_millis();

                // Per-block audio/video stats for LTX2_DUMP_AUDIO — reveals
                // where the audio stream std explodes across the 48 blocks.
                if std::env::var("LTX2_DUMP_AUDIO").is_ok() {
                    let _ = device.synchronize();
                    let stats = |t: &Tensor| -> Option<(f64, f64, f32, f32)> {
                        t.to_dtype(DType::F32).and_then(|x| x.to_vec()).ok().map(|v| {
                            let n = v.len() as f64;
                            let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
                            let var = v.iter().map(|x| { let d = *x as f64 - mean; d*d }).sum::<f64>() / n;
                            let std = var.sqrt();
                            let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            (mean, std, mn, mx)
                        })
                    };
                    if let (Some(a), Some(v)) = (stats(&new_ahs), stats(&new_hs)) {
                        log::info!("[BLK {:>2}] VIDEO mean={:+.4} std={:.4} min={:+.4} max={:+.4}  ||  AUDIO mean={:+.4} std={:.4} min={:+.4} max={:+.4}",
                            i, v.0, v.1, v.2, v.3, a.0, a.1, a.2, a.3);
                    }
                }

                // Dump block 0 outputs (the last thing fired in the OnceLock above
                // already saved inputs; we save outputs to a separate file).
                if i == 0 && std::env::var("LTX2_DUMP_BLOCK0").is_ok() {
                    static OUT_DUMPED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                    if OUT_DUMPED.set(()).is_ok() {
                        let _ = device.synchronize();
                        let mut out: HashMap<String, Tensor> = HashMap::new();
                        out.insert("hs_out".into(), new_hs.clone());
                        out.insert("ahs_out".into(), new_ahs.clone());
                        flame_core::serialization::save_tensors(
                            &out,
                            std::path::Path::new("/home/alex/EriDiffusion/inference-flame/output/rust_block0_output.safetensors"),
                            flame_core::serialization::SerializationFormat::SafeTensors,
                        )?;
                        log::info!("[LTX2 DUMP] Saved block 0 outputs. Exiting after this block to keep dump fast.");
                        return Err(flame_core::Error::Io("LTX2_DUMP_BLOCK0 early exit".into()));
                    }
                }

                hs = new_hs;
                ahs = new_ahs;
                drop(block);
                if prof || (i + 1) % 12 == 0 || i + 1 == num_layers || i == 0 {
                    log::info!("[LTX2] AV Block {}/{}: total={}ms (await={}ms, build={}ms, forward={}ms)",
                        i + 1, num_layers,
                        t_block.elapsed().as_millis(),
                        t_after_await,
                        t_after_build - t_after_await,
                        t_after_forward - t_after_build);
                }
            }
        } else {
            return Err(flame_core::Error::InvalidInput(
                "forward_audio_video requires FP8 resident or FlameSwap".into()));
        }

        // 9. Video output
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?;
        let emb_4d = v_embedded.unsqueeze(2)?;
        let final_ss = shift_scale.add(&emb_4d)?.to_dtype(DType::BF16)?;
        let v_shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let v_scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let v_normed = layer_norm_no_affine(&hs, self.config.norm_eps)?;
        let v_output = v_normed.mul(&v_scale.add_scalar(1.0)?.to_dtype(DType::BF16)?)?.add(&v_shift)?;
        let v_output = linear3d(&v_output, &self.proj_out_weight, Some(&self.proj_out_bias))?;
        let v_output = v_output.permute(&[0, 2, 1])?;
        let v_output = v_output.reshape(&[batch_size, channels, num_frames, height, width])?;

        // 10. Audio output
        let a_ss = audio_sst.unsqueeze(0)?.unsqueeze(0)?;
        let a_emb_4d = a_embedded.unsqueeze(2)?;
        let a_final_ss = a_ss.add(&a_emb_4d)?.to_dtype(DType::BF16)?;
        let a_shift = a_final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let a_scale = a_final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let a_normed = layer_norm_no_affine(&ahs, self.config.norm_eps)?;
        let a_modulated = a_normed.mul(&a_scale.add_scalar(1.0)?.to_dtype(DType::BF16)?)?.add(&a_shift)?;
        let a_linout = linear3d(&a_modulated, audio_proj_out_w, audio_proj_out_b)?;
        // Unpatchify audio: [B, T, C*F] → [B, C, T, F]
        let a_output = a_linout.reshape(&[batch_size, audio_frames, audio_channels, audio_freq])?; // [B, T, C, F]
        let a_output = a_output.permute(&[0, 2, 1, 3])?; // [B, C, T, F]

        // LTX2_DUMP_AUDIO=1 — audio path sanity dump. Fires once per process
        // and saves intermediates to rust_audio_dump.safetensors so they can
        // be diffed against python_audio_dump.safetensors.
        if std::env::var("LTX2_DUMP_AUDIO").is_ok() {
            static AUDIO_DUMPED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if AUDIO_DUMPED.set(()).is_ok() {
                let _ = device.synchronize();
                let mut dump: HashMap<String, Tensor> = HashMap::new();
                if let Some(t) = a_flat_input { dump.insert("a_flat_input".into(), t); }
                if let Some(t) = ahs_after_projin { dump.insert("ahs_after_projin".into(), t); }
                dump.insert("ahs_final".into(), ahs.clone());
                dump.insert("a_timestep".into(), a_timestep.clone());
                dump.insert("a_embedded".into(), a_embedded.clone());
                dump.insert("audio_sst".into(), audio_sst.clone());
                dump.insert("a_shift".into(), a_shift.clone());
                dump.insert("a_scale".into(), a_scale.clone());
                dump.insert("a_normed".into(), a_normed.clone());
                dump.insert("a_modulated".into(), a_modulated.clone());
                dump.insert("a_linout".into(), a_linout.clone());
                dump.insert("a_output_final".into(), a_output.clone());
                flame_core::serialization::save_tensors(
                    &dump,
                    std::path::Path::new("/home/alex/EriDiffusion/inference-flame/output/rust_audio_dump.safetensors"),
                    flame_core::serialization::SerializationFormat::SafeTensors,
                )?;
                log::info!("[LTX2 AUDIO DUMP] saved {} tensors", dump.len());
            }
        }

        Ok((v_output, a_output))
    }
}

/// Detect key prefix in a safetensors checkpoint.
///
/// Returns `"model.diffusion_model."` for ComfyUI format, `""` for diffusers.
fn detect_key_prefix(path: &str) -> Result<String> {
    use serde_json::Value;
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| flame_core::Error::Io(format!("Failed to open: {}", e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| flame_core::Error::Io(format!("mmap: {}", e)))?;

    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let metadata: Value = serde_json::from_slice(&mmap[8..8 + header_size])
        .map_err(|e| flame_core::Error::Io(format!("json: {}", e)))?;

    if let Some(obj) = metadata.as_object() {
        for key in obj.keys() {
            if key.starts_with("model.diffusion_model.") {
                return Ok("model.diffusion_model.".to_string());
            }
            if key.starts_with("transformer_blocks.")
                || key == "proj_in.weight"
                || key == "patchify_proj.weight"
                || key == "adaln_single.linear.weight"
            {
                return Ok(String::new());
            }
        }
    }
    Ok(String::new())
}

/// Load LTX-2 model with video-only weights (skip all audio keys).
///
/// This loads ~16.5B params instead of 22B, saving ~13GB of VRAM.
/// Audio fields are initialized with zero-sized dummy tensors.
pub fn load_ltx2_model_video_only(
    weights: &HashMap<String, Tensor>,
    config: &LTX2Config,
) -> Result<LTX2Model> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    let device = weights.values().next()
        .map(|t| t.device().clone())
        .ok_or_else(|| flame_core::Error::InvalidInput("Empty weights".into()))?;

    // Create zero-filled dummy tensors for audio fields
    let dummy_1d = |size: usize| -> Result<Tensor> {
        Tensor::zeros_dtype(Shape::from_dims(&[size]), DType::BF16, device.clone())
    };
    let dummy_2d = |r: usize, c: usize| -> Result<Tensor> {
        Tensor::zeros_dtype(Shape::from_dims(&[r, c]), DType::BF16, device.clone())
    };

    let ad = config.audio_inner_dim();
    let ahd = config.audio_attention_head_dim;
    let anh = config.audio_num_attention_heads;

    let dummy_attention = |_prefix: &str| -> Result<LTX2Attention> {
        Ok(LTX2Attention {
            to_q_weight: dummy_2d(ad, ad)?,
            to_q_bias: dummy_1d(ad)?,
            to_k_weight: dummy_2d(ad, ad)?,
            to_k_bias: dummy_1d(ad)?,
            to_v_weight: dummy_2d(ad, ad)?,
            to_v_bias: dummy_1d(ad)?,
            norm_q_weight: dummy_1d(ad)?,
            norm_k_weight: dummy_1d(ad)?,
            to_out_weight: dummy_2d(ad, ad)?,
            to_out_bias: dummy_1d(ad)?,
            to_gate_logits_weight: None,
            to_gate_logits_bias: None,
            num_heads: anh,
            head_dim: ahd,
            eps: config.norm_eps,
        })
    };

    // Dummy weights use pre-transposed layout: [in_features, out_features]
    let dummy_ff = || -> Result<FeedForward> {
        let affn = config.audio_ffn_hidden();
        Ok(FeedForward {
            gelu_proj_weight: dummy_2d(ad, affn)?,      // pre-transposed: [in=ad, out=affn]
            gelu_proj_bias: dummy_1d(affn)?,
            out_weight: dummy_2d(affn, ad)?,             // pre-transposed: [in=affn, out=ad]
            out_bias: dummy_1d(ad)?,
        })
    };

    let dummy_adaln = || -> Result<AdaLayerNormSingle> {
        Ok(AdaLayerNormSingle {
            emb: TimestepEmbedderMLP {
                linear_1_weight: dummy_2d(256, ad)?,     // pre-transposed: [in=256, out=ad]
                linear_1_bias: dummy_1d(ad)?,
                linear_2_weight: dummy_2d(ad, ad)?,
                linear_2_bias: dummy_1d(ad)?,
            },
            linear_weight: dummy_2d(ad, ad)?,
            linear_bias: dummy_1d(ad)?,
            num_mod_params: 1,
        })
    };

    let dummy_caption = || -> Result<CaptionProjection> {
        Ok(CaptionProjection {
            linear_1_weight: dummy_2d(config.caption_channels, ad)?,  // pre-transposed
            linear_1_bias: dummy_1d(ad)?,
            linear_2_weight: dummy_2d(ad, ad)?,
            linear_2_bias: dummy_1d(ad)?,
        })
    };

    // Load video-only transformer blocks
    let eps = config.norm_eps;
    let num_heads = config.num_attention_heads;
    let head_dim = config.attention_head_dim;

    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("transformer_blocks.{i}");
        let get_opt = |key: &str| -> Option<Tensor> { weights.get(key).cloned() };

        blocks.push(LTX2TransformerBlock {
            // Video self-attention (real weights)
            norm1_weight: get_opt(&format!("{prefix}.norm1.weight")),
            attn1: load_attention(weights, &format!("{prefix}.attn1"), num_heads, head_dim, eps)?,

            // Audio self-attention (dummy)
            audio_norm1_weight: None,
            audio_attn1: dummy_attention("")?,

            // Video cross-attention (real weights)
            norm2_weight: get_opt(&format!("{prefix}.norm2.weight")),
            attn2: load_attention(weights, &format!("{prefix}.attn2"), num_heads, head_dim, eps)?,

            // Audio cross-attention (dummy)
            audio_norm2_weight: None,
            audio_attn2: dummy_attention("")?,

            // A2V/V2A cross-attention (dummy)
            audio_to_video_norm_weight: None,
            audio_to_video_attn: dummy_attention("")?,
            video_to_audio_norm_weight: None,
            video_to_audio_attn: dummy_attention("")?,

            // Video FFN (real weights)
            norm3_weight: get_opt(&format!("{prefix}.norm3.weight")),
            ff: load_feed_forward(weights, &format!("{prefix}.ff"))?,

            // Audio FFN (dummy)
            audio_norm3_weight: None,
            audio_ff: dummy_ff()?,

            // Video modulation table (real), audio (dummy)
            scale_shift_table: get(&format!("{prefix}.scale_shift_table"))?,
            audio_scale_shift_table: dummy_2d(6, ad)?,
            video_a2v_cross_attn_scale_shift_table: dummy_2d(5, config.inner_dim())?,
            audio_a2v_cross_attn_scale_shift_table: dummy_2d(5, ad)?,

            prompt_scale_shift_table: get_opt(&format!("{prefix}.prompt_scale_shift_table")),
            audio_prompt_scale_shift_table: get_opt(&format!("{prefix}.audio_prompt_scale_shift_table")),

            eps,
        });

        if (i + 1) % 12 == 0 {
            log::info!("[LTX2] Loaded block {}/{}", i + 1, config.num_layers);
        }
    }

    log::info!("[LTX2] All {} blocks loaded (video-only)", config.num_layers);

    Ok(LTX2Model {
        config: config.clone(),

        proj_in_weight: pre_transpose_weight(&get("proj_in.weight")?)?,
        proj_in_bias: get("proj_in.bias")?,
        audio_proj_in_weight: dummy_2d(config.audio_in_channels, ad)?,  // pre-transposed
        audio_proj_in_bias: dummy_1d(ad)?,

        caption_projection: load_caption_projection(weights, "caption_projection")?,
        audio_caption_projection: dummy_caption()?,

        time_embed: load_ada_layer_norm_single(weights, "time_embed", 6)?,
        audio_time_embed: dummy_adaln()?,

        av_cross_attn_video_scale_shift: dummy_adaln()?,
        av_cross_attn_audio_scale_shift: dummy_adaln()?,
        av_cross_attn_video_a2v_gate: dummy_adaln()?,
        av_cross_attn_audio_v2a_gate: dummy_adaln()?,

        scale_shift_table: get("scale_shift_table")?,
        audio_scale_shift_table: dummy_2d(2, ad)?,

        blocks,

        proj_out_weight: pre_transpose_weight(&get("proj_out.weight")?)?,
        proj_out_bias: get("proj_out.bias")?,
        audio_proj_out_weight: dummy_2d(ad, config.audio_out_channels)?,  // pre-transposed
        audio_proj_out_bias: dummy_1d(config.audio_out_channels)?,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = LTX2Config::default();
        assert_eq!(cfg.inner_dim(), 4096);          // 32 * 128
        assert_eq!(cfg.audio_inner_dim(), 2048);     // 32 * 64
        assert_eq!(cfg.ffn_hidden(), 16384);          // 4096 * 4
        assert_eq!(cfg.audio_ffn_hidden(), 8192);     // 2048 * 4
        assert_eq!(cfg.num_layers, 48);
    }

    #[test]
    fn test_vram_estimate() {
        let cfg = LTX2Config::default();
        // Rough sanity check: 18.88B params * 2 bytes = ~37.76 GB
        // Our estimate won't be exact but should be in the right ballpark
        let model_estimate_gb = {
            let d = cfg.inner_dim();
            let ad = cfg.audio_inner_dim();
            // Very rough: each block has ~6 attention modules + 2 FFN
            let per_block = 6 * (4 * d * d) + 6 * (4 * ad * ad) + 2 * (d * d * 4 * 2) + 2 * (ad * ad * 4 * 2);
            let total_params = per_block * cfg.num_layers;
            (total_params * 2) as f64 / (1024.0 * 1024.0 * 1024.0)
        };
        // Should be roughly 30-40 GB in BF16
        assert!(model_estimate_gb > 10.0, "Model should be > 10 GB, got {:.1} GB", model_estimate_gb);
        assert!(model_estimate_gb < 80.0, "Model should be < 80 GB, got {:.1} GB", model_estimate_gb);
    }
}
