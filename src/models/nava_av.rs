//! NAVA (= Ovi joint audio-video MM-DiT) — foundation module.
//!
//! Scope of THIS file (foundation only — NOT the transformer blocks):
//!   - [`NavaAVConfig`] mirroring `configs/model/dit/NAVA_6B.json`.
//!   - Video Conv3d patch-embed (k=stride=(1,2,2), 48→3072) — mirrors `wan22_dit.rs`.
//!   - Audio Conv1d patch-embed: `ChannelLastConv1d(k=7,p=3,bias) + SiLU + ConvMLP`.
//!   - Text embedding (4096→dim→dim, GELU-tanh), time embedding
//!     (sinusoidal→dim→dim, SiLU) + time_projection (SiLU→dim→6·dim).
//!   - RoPE freq tables: video 3-axis [44,42,42] interleaved (no temporal scaling),
//!     audio 1-axis 22-complex (44 real) interleaved with `temporal_rope_scaling_factor=0.24`.
//!   - Audio 1D interleaved-PARTIAL rope-apply wrapper (rotates first 44 of 128 dims).
//!
//! The double/single transformer blocks, joint-attention, heads, loader and gen
//! bin are OUT OF SCOPE this round (next round).
//!
//! ## Reference (read line-by-line, do not infer)
//!   - `ports/nava/nava_src/models/nava/modules/model_mm.py`
//!     (`WanAVModel.__init__` ~1289-1305, `set_rope_params` ~1392-1407,
//!     `sinusoidal_embedding_1d` ~24, time/text embed ~1457-1513).
//!   - `ports/nava/nava_src/models/nava/modules/model.py`
//!     (`rope_params` ~38-45 torch.polar complex, `rope_apply_1d` ~47-70
//!     INTERLEAVED PARTIAL, `ChannelLastConv1d` ~137-143, `ConvMLP` ~146-191).
//!   - `configs/model/dit/NAVA_6B.json` — live config values.
//!
//! ## Dtype discipline (PORT_SPEC / BUILD_PLAN: HIGH risk)
//!   - Model runs under BF16 autocast. Embedding / modulation math stays BF16
//!     (do NOT upgrade to F32 — fails parity).
//!   - RoPE freq tables built in **F32** (precision floor). NAVA computes freqs
//!     in F64 then casts to BF16 inside `rope_apply_*`; F32 tables are the
//!     parity-safe staging dtype, converted to BF16 only at the fused-kernel call.
//!
//! ## RoPE convention (PORT_SPEC #1 skeptic-bait)
//!   Audio 1D RoPE is **INTERLEAVED** (`rope_fused_bf16`: `out[2i]=x[2i]·cos[i]−x[2i+1]·sin[i]`),
//!   NOT halfsplit. It rotates only the first `AUDIO_ROPE_DIM=44` real dims (= 22
//!   complex pairs) of `head_dim=128`, passing through the remaining 84.
//!   Getting this wrong silently corrupts audio.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use flame_core::CudaDevice;

use super::wan::rope3d::axis_split;

/// RoPE base frequency (theta) — matches NAVA `rope_params(theta=10000)`.
const ROPE_THETA: f64 = 10000.0;

/// Number of real dims rotated by the audio 1D partial RoPE.
///
/// From `set_rope_params`: `freqs_audio = rope_params(1024, d - 4*(d//6))`.
/// With `d = head_dim = 128`: `d//6 = 21`, `4*(d//6) = 84`, so the rotated
/// dim = `128 - 84 = 44` real = 22 complex pairs. The remaining `128 - 44 = 84`
/// real dims pass through untouched.
pub const AUDIO_ROPE_DIM: usize = 44;

/// Config for the NAVA single-tower joint-AV MM-DiT (`WanAVModel`).
///
/// Mirrors `configs/model/dit/NAVA_6B.json`. Scope = Base T2AV.
/// `add_spk_emb` fields are present for naming completeness but SpkToken is
/// NOT built this round.
#[derive(Debug, Clone)]
pub struct NavaAVConfig {
    pub patch_size: [usize; 3],   // (1, 2, 2) video
    pub model_type: String,       // "ti2v"
    pub dim: usize,               // 3072
    pub ffn_dim: usize,           // 14336
    pub freq_dim: usize,          // 256 (sinusoidal timestep)
    pub num_heads: usize,         // 24
    pub head_dim: usize,          // 128 (dim / num_heads) — BOTH video AND audio
    pub num_layers: usize,        // 30
    pub num_double_layers: usize, // 10 (alignment)
    pub num_single_layers: usize, // 20 (fusion)
    pub vid_in_dim: usize,        // 48
    pub vid_out_dim: usize,       // 48
    pub audio_in_dim: usize,      // 128
    pub audio_out_dim: usize,     // 128
    pub text_dim: usize,          // 4096 (umt5-xxl hidden) — text_embedding input
    pub text_len: usize,          // 512
    pub qk_norm: bool,            // true
    pub cross_attn_norm: bool,    // true
    pub no_split_norm_ffn: bool,  // true (double blocks share norm/ffn)
    pub add_spk_emb: bool,        // true (SpkToken — NOT built this round)
    pub eps: f32,                 // 1e-6
    pub temporal_rope_scaling_factor: f32, // 0.24 (audio RoPE only)
}

impl Default for NavaAVConfig {
    /// NAVA_6B.json defaults.
    fn default() -> Self {
        let dim = 3072;
        let num_heads = 24;
        Self {
            patch_size: [1, 2, 2],
            model_type: "ti2v".to_string(),
            dim,
            ffn_dim: 14336,
            freq_dim: 256,
            num_heads,
            head_dim: dim / num_heads, // 128
            num_layers: 30,
            num_double_layers: 10,
            num_single_layers: 20,
            vid_in_dim: 48,
            vid_out_dim: 48,
            audio_in_dim: 128,
            audio_out_dim: 128,
            // umt5-xxl encoder hidden size; `nn.Linear(text_dim, dim)` in text_embedding.
            text_dim: 4096,
            text_len: 512,
            qk_norm: true,
            cross_attn_norm: true,
            no_split_norm_ffn: true,
            add_spk_emb: true,
            eps: 1e-6,
            temporal_rope_scaling_factor: 0.24,
        }
    }
}

impl NavaAVConfig {
    /// ConvMLP hidden dim used by the audio patch-embed.
    ///
    /// From `ConvMLP.__init__` (model.py:171-172) with `hidden_dim = dim*4`:
    /// ```text
    /// hidden_dim = int(2 * (dim*4) / 3)
    /// hidden_dim = 256 * ceil(hidden_dim / 256)
    /// ```
    pub fn audio_convmlp_hidden(&self) -> usize {
        let multiple_of = 256usize;
        let h = (2 * (self.dim * 4)) / 3; // int() truncates toward zero; all positive
        multiple_of * h.div_ceil(multiple_of)
    }
}

// ===========================================================================
// Linear helpers (mirror wan22_dit::linear_bias / linear_nobias)
// ===========================================================================
// fused_linear3d_native uses cuBLASLt TRANSA=T and takes native PyTorch
// [Cout, Cin] weights — zero transpose allocation.

fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
}

fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
}

// ===========================================================================
// Video patch-embed (Conv3d k=stride=(1,2,2), 48→3072) — mirrors wan22_dit.rs
// ===========================================================================
//
// Conv3d with kernel == stride == patch_size is a per-patch linear: gather each
// (1,2,2) cube into a `patch_dim = C·1·2·2` vector, then `Linear(patch_dim, dim)`.

/// Patchify a `[C, F, H, W]` latent into `[n_patches, patch_dim]`.
///
/// `patch_dim = c * pt * ph * pw` with patch = (pt, ph, pw).
/// Token order is (frame, height, width) row-major — matching
/// `u.flatten(2).transpose(1,2)` in `prepare_transformer_block_kwargs`.
pub fn patchify_video(
    x: &Tensor,
    c: usize,
    f: usize,
    h: usize,
    w: usize,
    patch: [usize; 3],
) -> Result<Tensor> {
    let (pt, ph, pw) = (patch[0], patch[1], patch[2]);
    let (f_out, h_out, w_out) = (f / pt, h / ph, w / pw);
    let patch_dim = c * pt * ph * pw;

    // [C, F, H, W] -> [C, f_out, pt, h_out, ph, w_out, pw]
    let x = x.reshape(&[c, f_out, pt, h_out, ph, w_out, pw])?;
    // -> [f_out, h_out, w_out, C, pt, ph, pw] so the gathered cube is contiguous
    //    per patch and channel-major within the cube (matches Conv3d weight layout
    //    [Cout, Cin, pt, ph, pw] flattened to [Cout, Cin*pt*ph*pw]).
    let x = x.permute(&[1, 3, 5, 0, 2, 4, 6])?.contiguous()?;
    // -> [n_patches, patch_dim]
    x.reshape(&[f_out * h_out * w_out, patch_dim])
}

/// Run the video Conv3d patch-embed as a linear over patchified tokens.
///
/// `x`: latent `[C, F, H, W]` BF16.
/// `pe_w`: `patch_embedding.weight` `[dim, Cin, pt, ph, pw]` BF16.
/// `pe_b`: `patch_embedding.bias` `[dim]` BF16.
/// Returns `[1, n_patches, dim]` BF16 plus the `(f_out, h_out, w_out)` grid.
pub fn video_patch_embed(
    cfg: &NavaAVConfig,
    x: &Tensor,
    pe_w: &Tensor,
    pe_b: &Tensor,
) -> Result<(Tensor, (usize, usize, usize))> {
    let dims = x.shape().dims();
    let (c, f, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let patch = cfg.patch_size;
    let (f_out, h_out, w_out) = (f / patch[0], h / patch[1], w / patch[2]);
    let patch_dim = c * patch[0] * patch[1] * patch[2];

    let patched = patchify_video(x, c, f, h, w, patch)?; // [n_patches, patch_dim]
    let pe_w_flat = pe_w.reshape(&[cfg.dim, patch_dim])?;
    let patched_3d = patched.unsqueeze(0)?; // [1, n_patches, patch_dim]
    let img = linear_bias(&patched_3d, &pe_w_flat, pe_b)?; // [1, n_patches, dim]
    Ok((img, (f_out, h_out, w_out)))
}

// ===========================================================================
// Audio patch-embed (NEW — assembled from existing ops)
// ===========================================================================
//
// patch_embedding_audio = Sequential(
//     ChannelLastConv1d(audio_in_dim -> dim, k=7, p=3, bias=True),
//     SiLU(),
//     ConvMLP(dim, dim*4, k=7, p=3),       # w2(silu(w1(x)) * w3(x)), bias=False
// )
//
// ChannelLastConv1d.forward: permute [B,L,C]->[B,C,L]; Conv1d; permute back.

/// `ChannelLastConv1d`: `[B, L, Cin] -> [B, L, Cout]`.
///
/// `w`: `[Cout, Cin, K]` BF16. `bias`: optional `[Cout]` BF16.
/// Uses `flame_core::conv1d::conv1d` (cuDNN). The `permute`s mirror
/// `ChannelLastConv1d.forward` exactly.
fn channel_last_conv1d(
    x: &Tensor, // [B, L, Cin]
    w: &Tensor, // [Cout, Cin, K]
    bias: Option<&Tensor>,
    padding: usize,
) -> Result<Tensor> {
    // [B, L, Cin] -> [B, Cin, L]. conv1d requires a 3D [B,C,L] contiguous input;
    // cudnn conv (k>1) reads the buffer densely, so a permuted view is unsafe.
    let x_cl = x.permute(&[0, 2, 1])?.contiguous()?;
    let out = flame_core::conv1d::conv1d(&x_cl, w, bias, 1, padding, 1, 1)?; // [B, Cout, L]
    // [B, Cout, L] -> [B, L, Cout]
    out.permute(&[0, 2, 1])?.contiguous()
}

/// `ConvMLP.forward`: `w2(silu(w1(x)) * w3(x))`, all k=7 p=3 bias=False.
///
/// `w1`,`w3`: `[hidden, dim, 7]`. `w2`: `[dim, hidden, 7]`. All BF16, channel-last.
/// The `silu(w1(x)) * w3(x)` middle is computed by `swiglu_fused_bf16(gate=w1x, up=w3x)`
/// (= `silu(gate) * up`), matching NAVA's `F.silu(self.w1(x)) * self.w3(x)`.
fn audio_conv_mlp(
    x: &Tensor, // [B, L, dim]
    w1: &Tensor,
    w2: &Tensor,
    w3: &Tensor,
) -> Result<Tensor> {
    let gate = channel_last_conv1d(x, w1, None, 3)?; // silu side: [B, L, hidden]
    let up = channel_last_conv1d(x, w3, None, 3)?; // [B, L, hidden]
    let mid = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?; // silu(gate)*up
    channel_last_conv1d(&mid, w2, None, 3) // [B, L, dim]
}

/// Audio patch-embed over a raw audio-latent stream.
///
/// `x`: `[B, L, audio_in_dim]` BF16 (channel-last, as fed to `ChannelLastConv1d`).
/// Weight names (PyTorch `nn.Sequential`):
///   `patch_embedding_audio.0.{weight,bias}` — the input k=7 conv (bias=True).
///   `patch_embedding_audio.2.w{1,2,3}.weight` — the ConvMLP convs (bias=False).
/// Returns `[B, L, dim]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn audio_patch_embed(
    x: &Tensor,
    conv0_w: &Tensor,
    conv0_b: &Tensor,
    mlp_w1: &Tensor,
    mlp_w2: &Tensor,
    mlp_w3: &Tensor,
) -> Result<Tensor> {
    let x = channel_last_conv1d(x, conv0_w, Some(conv0_b), 3)?; // [B, L, dim]
    let x = x.silu()?; // nn.SiLU()
    audio_conv_mlp(&x, mlp_w1, mlp_w2, mlp_w3)
}

// ===========================================================================
// Timestep / text embeddings
// ===========================================================================

/// `sinusoidal_embedding_1d(dim, position)` — matches model_mm.py:24-34.
///
/// ```text
/// half = dim // 2
/// sinusoid = outer(position, pow(10000, -arange(half)/half))   # F64
/// x = cat(cos(sinusoid), sin(sinusoid), dim=1)                 # [N, dim]
/// ```
/// Computed in F64 then returned F32 (caller casts to BF16). cos first half,
/// sin second half — NOT interleaved.
pub fn sinusoidal_embedding_1d(
    dim: usize,
    timesteps: &[f32],
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    assert!(dim % 2 == 0, "sinusoidal_embedding_1d: dim must be even");
    let half = dim / 2;
    let n = timesteps.len();
    let mut data = vec![0.0f32; n * dim];
    for (t_idx, &pos) in timesteps.iter().enumerate() {
        let pos = pos as f64;
        for i in 0..half {
            let freq = ROPE_THETA.powf(-(i as f64) / half as f64);
            let angle = pos * freq;
            data[t_idx * dim + i] = angle.cos() as f32; // cos first half
            data[t_idx * dim + half + i] = angle.sin() as f32; // sin second half
        }
    }
    Tensor::from_vec(data, Shape::from_dims(&[n, dim]), device)
}

/// `time_embedding` + `time_projection` (model_mm.py:1470-1473).
///
/// `time_embedding`: `Linear(freq_dim, dim) -> SiLU -> Linear(dim, dim)`.
/// `time_projection`: `SiLU -> Linear(dim, 6·dim)`, reshaped `[B, seq, 6, dim]`.
///
/// All BF16 (NAVA does this under `autocast(bfloat16)`; `assert e.dtype ==
/// bfloat16 and e0.dtype == bfloat16`). Returns `(e, e0)`:
///   `e`:  `[B, seq, dim]`
///   `e0`: `[B, seq, 6, dim]`
///
/// Weight names: `time_embedding.{0,2}.{weight,bias}`, `time_projection.1.{weight,bias}`.
#[allow(clippy::too_many_arguments)]
pub fn time_embedding(
    cfg: &NavaAVConfig,
    timesteps: &[f32], // per-token timesteps, flattened length = b*seq
    b: usize,
    seq: usize,
    device: Arc<CudaDevice>,
    te_w0: &Tensor,
    te_b0: &Tensor,
    te_w2: &Tensor,
    te_b2: &Tensor,
    tp_w: &Tensor,
    tp_b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    debug_assert_eq!(timesteps.len(), b * seq);
    // sinusoidal -> [b*seq, freq_dim] F32 -> reshape [b, seq, freq_dim] BF16
    let sin = sinusoidal_embedding_1d(cfg.freq_dim, timesteps, device)?;
    let sin = sin
        .reshape(&[b, seq, cfg.freq_dim])?
        .to_dtype(DType::BF16)?;

    // time_embedding: Linear -> SiLU -> Linear
    let e = linear_bias(&sin, te_w0, te_b0)?;
    let e = e.silu()?;
    let e = linear_bias(&e, te_w2, te_b2)?; // [b, seq, dim]

    // time_projection: SiLU -> Linear(dim, 6*dim) -> [b, seq, 6, dim]
    let e_silu = e.silu()?;
    let e0_flat = linear_bias(&e_silu, tp_w, tp_b)?; // [b, seq, 6*dim]
    let e0 = e0_flat.reshape(&[b, seq, 6, cfg.dim])?;
    Ok((e, e0))
}

/// `text_embedding` (model_mm.py:1299-1301 / 1508-1513).
///
/// `Linear(text_dim, dim) -> GELU(approximate='tanh') -> Linear(dim, dim)`.
/// `Tensor::gelu()` is the tanh approximation (`gelu.cu:24`
/// `0.7978845608·(x + 0.044715·x³)`), matching NAVA's `nn.GELU(approximate='tanh')`.
/// Context is padded/truncated to `text_len` by the caller. BF16.
///
/// `context`: `[B, L, text_dim]` BF16. Returns `[B, L, dim]` BF16.
/// Weight names: `text_embedding.{0,2}.{weight,bias}`.
pub fn text_embedding(
    context: &Tensor,
    txt_w0: &Tensor,
    txt_b0: &Tensor,
    txt_w2: &Tensor,
    txt_b2: &Tensor,
) -> Result<Tensor> {
    let x = linear_bias(context, txt_w0, txt_b0)?;
    let x = x.gelu()?; // tanh-approx
    linear_bias(&x, txt_w2, txt_b2)
}

// ===========================================================================
// RoPE freq tables (F32) + audio 1D interleaved-PARTIAL apply wrapper
// ===========================================================================
//
// NAVA `rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0)`:
//   freqs = (1/theta^(arange(0,dim,2)/dim)) * freqs_scaling          # [dim/2]
//   freqs = outer(arange(max_seq_len), freqs)                        # [L, dim/2]
//   freqs = polar(1, freqs)                                          # complex
// The complex `polar(1, angle)` = cos(angle) + i·sin(angle); the interleaved
// kernel consumes the cos/sin tables of `angle = pos * freq` directly.

/// Build interleaved RoPE cos/sin tables for a single axis.
///
/// Returns `(cos, sin)` each `[max_seq_len, dim/2]` in **F32** (precision floor).
/// `freqs_scaling` multiplies the per-pair frequency (audio uses 0.24; video uses 1.0).
fn rope_axis_table(
    max_seq_len: usize,
    dim: usize,
    freqs_scaling: f64,
    device: Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    assert!(dim % 2 == 0, "rope_axis_table: dim must be even");
    let half = dim / 2;
    let mut cos_data = vec![0.0f32; max_seq_len * half];
    let mut sin_data = vec![0.0f32; max_seq_len * half];
    for pos in 0..max_seq_len {
        for i in 0..half {
            // freq = 1 / theta^(2i/dim), scaled. (arange(0,dim,2)[i] = 2i)
            let freq = (1.0 / ROPE_THETA.powf((2 * i) as f64 / dim as f64)) * freqs_scaling;
            let angle = pos as f64 * freq;
            cos_data[pos * half + i] = angle.cos() as f32;
            sin_data[pos * half + i] = angle.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos_data, Shape::from_dims(&[max_seq_len, half]), device.clone())?;
    let sin = Tensor::from_vec(sin_data, Shape::from_dims(&[max_seq_len, half]), device)?;
    Ok((cos, sin))
}

/// Per-axis F32 RoPE freq tables for NAVA.
///
/// Video: 3 axes [44, 42, 42] (no temporal scaling), each `[max_seq_len, axis/2]`.
/// Audio: 1 axis of `AUDIO_ROPE_DIM=44` real (= 22 complex) with `temporal_rope_scaling_factor`.
///
/// Tables are F32; the apply wrappers cast the gathered cos/sin to BF16 right
/// before the fused kernel (NAVA computes freqs in F64 then casts to BF16).
pub struct NavaRopeTables {
    /// Video per-axis (t, h, w) cos tables, each `[max_seq_len, axis_half]` F32.
    pub video_cos: [Tensor; 3],
    pub video_sin: [Tensor; 3],
    pub video_axes: (usize, usize, usize), // (44, 42, 42)
    /// Audio cos/sin, `[max_seq_len, AUDIO_ROPE_DIM/2]` F32.
    pub audio_cos: Tensor,
    pub audio_sin: Tensor,
}

impl NavaRopeTables {
    /// Build all RoPE tables.
    ///
    /// `max_seq_len` matches NAVA's `rope_params(1024, ...)` default (1024).
    pub fn new(cfg: &NavaAVConfig, max_seq_len: usize, device: Arc<CudaDevice>) -> Result<Self> {
        let (at, ah, aw) = axis_split(cfg.head_dim); // (44, 42, 42)
        debug_assert_eq!(at + ah + aw, cfg.head_dim);

        // Video: NO temporal scaling (freqs_scaling=1.0).
        let (vt_c, vt_s) = rope_axis_table(max_seq_len, at, 1.0, device.clone())?;
        let (vh_c, vh_s) = rope_axis_table(max_seq_len, ah, 1.0, device.clone())?;
        let (vw_c, vw_s) = rope_axis_table(max_seq_len, aw, 1.0, device.clone())?;

        // Audio: rope_params(1024, 44) WITH temporal_rope_scaling_factor.
        debug_assert_eq!(at, AUDIO_ROPE_DIM, "audio rope dim must equal video t-axis (both 44)");
        let (a_c, a_s) = rope_axis_table(
            max_seq_len,
            AUDIO_ROPE_DIM,
            cfg.temporal_rope_scaling_factor as f64,
            device,
        )?;

        Ok(Self {
            video_cos: [vt_c, vh_c, vw_c],
            video_sin: [vt_s, vh_s, vw_s],
            video_axes: (at, ah, aw),
            audio_cos: a_c,
            audio_sin: a_s,
        })
    }
}

/// Apply audio 1D interleaved-PARTIAL RoPE to a Q/K tensor.
///
/// `x`: `[B, H, S, head_dim]` BF16. `cos`,`sin`: `[1, 1, S, AUDIO_ROPE_DIM/2]` BF16
/// (caller slices the F32 table to the live seq length and casts to BF16).
///
/// Mirrors the magihuman `rope_partial_halfsplit` STRUCTURE — slice the first
/// `AUDIO_ROPE_DIM` channels, rotate them, cat with the passthrough tail — BUT
/// uses the INTERLEAVED `rope_fused_bf16` kernel (`out[2i]=x[2i]·cos[i]−x[2i+1]·sin[i]`),
/// NOT halfsplit. This matches NAVA `rope_apply_1d`, which rotates the first
/// `c_rope=22` complex pairs (= 44 real dims) and passes through the rest.
///
/// The `.contiguous()` on the rotated slice is REQUIRED before the fused kernel
/// (it reads the BF16 buffer densely). This is the documented magihuman-pattern
/// requirement, NOT a stride-bug workaround.
pub fn apply_audio_rope_1d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "apply_audio_rope_1d expects [B,H,S,D], got {:?}",
            dims
        )));
    }
    let head_dim = dims[3];
    if AUDIO_ROPE_DIM > head_dim {
        return Err(flame_core::Error::InvalidOperation(format!(
            "AUDIO_ROPE_DIM={} > head_dim={}",
            AUDIO_ROPE_DIM, head_dim
        )));
    }
    if AUDIO_ROPE_DIM == head_dim {
        // Full-rotation fast path — no cat needed.
        return flame_core::bf16_ops::rope_fused_bf16(x, cos, sin);
    }
    // Slice into rotated (first AUDIO_ROPE_DIM) + passthrough (rest).
    // .contiguous() required: the fused kernel reads the slice densely.
    let x_rot = x.narrow(3, 0, AUDIO_ROPE_DIM)?.contiguous()?;
    let x_pass = x.narrow(3, AUDIO_ROPE_DIM, head_dim - AUDIO_ROPE_DIM)?;
    let rotated = flame_core::bf16_ops::rope_fused_bf16(&x_rot, cos, sin)?;
    Tensor::cat(&[&rotated, &x_pass], 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::Device;

    fn dev() -> Arc<CudaDevice> {
        Device::cuda(0).unwrap().cuda_device_arc()
    }

    #[test]
    fn config_defaults_match_nava_6b() {
        let cfg = NavaAVConfig::default();
        assert_eq!(cfg.dim, 3072);
        assert_eq!(cfg.ffn_dim, 14336);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.head_dim, 128); // dim / num_heads
        assert_eq!(cfg.num_double_layers, 10);
        assert_eq!(cfg.num_single_layers, 20);
        assert_eq!(cfg.num_double_layers + cfg.num_single_layers, cfg.num_layers);
        assert!(cfg.qk_norm && cfg.cross_attn_norm && cfg.no_split_norm_ffn);
        assert_eq!(cfg.temporal_rope_scaling_factor, 0.24);
        assert_eq!(cfg.text_len, 512);
        // ConvMLP hidden: int(2*(3072*4)/3)=8192 -> ceil to 256-multiple = 8192.
        assert_eq!(cfg.audio_convmlp_hidden(), 8192);
    }

    #[test]
    fn rope_tables_have_correct_dims() {
        let cfg = NavaAVConfig::default();
        let max_len = 1024;
        let tables = NavaRopeTables::new(&cfg, max_len, dev()).unwrap();

        // Video axes split = (44, 42, 42), summing to head_dim=128.
        assert_eq!(tables.video_axes, (44, 42, 42));
        assert_eq!(tables.video_axes.0 + tables.video_axes.1 + tables.video_axes.2, 128);

        // Each video axis table is [max_len, axis/2].
        assert_eq!(tables.video_cos[0].shape().dims(), &[max_len, 22]); // 44/2
        assert_eq!(tables.video_cos[1].shape().dims(), &[max_len, 21]); // 42/2
        assert_eq!(tables.video_cos[2].shape().dims(), &[max_len, 21]);
        assert_eq!(tables.video_sin[0].shape().dims(), &[max_len, 22]);

        // Audio table is [max_len, AUDIO_ROPE_DIM/2] = [max_len, 22].
        assert_eq!(tables.audio_cos.shape().dims(), &[max_len, AUDIO_ROPE_DIM / 2]);
        assert_eq!(tables.audio_sin.shape().dims(), &[max_len, 22]);

        // Tables built in F32 (precision floor).
        assert_eq!(tables.audio_cos.dtype(), DType::F32);
        assert_eq!(tables.video_cos[0].dtype(), DType::F32);
    }

    #[test]
    fn audio_rope_preserves_shape_and_passthrough() {
        let device = dev();
        let cfg = NavaAVConfig::default();
        let (b, h, s, d) = (1usize, cfg.num_heads, 16usize, cfg.head_dim); // d=128

        // x = ones so passthrough tail stays exactly 1.0 (RoPE only touches first 44).
        let x = Tensor::from_vec(
            vec![1.0f32; b * h * s * d],
            Shape::from_dims(&[b, h, s, d]),
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        // cos/sin sliced to seq=s, shaped [1,1,s,AUDIO_ROPE_DIM/2], BF16.
        let tables = NavaRopeTables::new(&cfg, 1024, device).unwrap();
        let cos = tables
            .audio_cos
            .narrow(0, 0, s)
            .unwrap()
            .reshape(&[1, 1, s, AUDIO_ROPE_DIM / 2])
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let sin = tables
            .audio_sin
            .narrow(0, 0, s)
            .unwrap()
            .reshape(&[1, 1, s, AUDIO_ROPE_DIM / 2])
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let out = apply_audio_rope_1d(&x, &cos, &sin).unwrap();

        // Shape [.., 128] preserved.
        assert_eq!(out.shape().dims(), &[b, h, s, d]);

        // Passthrough tail (dims 44..128) must be untouched (still 1.0).
        let tail = out
            .narrow(3, AUDIO_ROPE_DIM, d - AUDIO_ROPE_DIM)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for v in &tail {
            assert!((v - 1.0).abs() < 1e-2, "passthrough tail changed: {}", v);
        }

        // At pos=0, angle=0 -> cos=1, sin=0, so the rotated head is identity too.
        // For pos>0 the rotated region generally differs from 1.0; assert that at
        // least one rotated element moved off 1.0 (proves rotation actually ran).
        let rot = out
            .narrow(2, 1, s - 1)
            .unwrap() // skip pos 0
            .narrow(3, 0, AUDIO_ROPE_DIM)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let moved = rot.iter().any(|v| (v - 1.0).abs() > 1e-2);
        assert!(moved, "rotated region never changed — RoPE did not run");
    }
}
