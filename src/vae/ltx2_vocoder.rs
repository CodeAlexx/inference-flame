//! LTX-2.3 BigVGAN vocoder — pure flame-core, production checkpoint parity.
//!
//! Matches `ltx_core.model.audio_vae.vocoder.Vocoder` for the LTX-2.3 22B
//! distilled checkpoint:
//!
//!   conv_pre:  Conv1d(128 → 1536, k=7, pad=3)
//!   ups[0..6]: ConvTranspose1d with strides [5, 2, 2, 2, 2, 2],
//!              kernels [11, 4, 4, 4, 4, 4], padding=(k-s)//2
//!              channel flow: 1536 → 768 → 384 → 192 → 96 → 48 → 24
//!   resblocks[0..18]: 6 stages × 3 AMPBlock1 (dilations 1, 3, 5)
//!                     Average of 3 outputs per stage.
//!   act_post:  Activation1d(SnakeBeta(24))
//!   conv_post: Conv1d(24 → 2, k=7, pad=3, bias=True)
//!   tanh final
//!
//! Base vocoder output: stereo 16 kHz (LTX-2.3 `input_sampling_rate`).
//! Final 48 kHz BWE path is NOT yet wired — this module returns 16 kHz which
//! the caller can upsample via linear/sinc as a fallback.
//!
//! AMPBlock1 structure (`ltx2_audio_vae.resnet.py`):
//!   for i in 0..3:
//!     xt = Activation1d(x, alpha_i, beta_i)         # Snake + anti-alias
//!     xt = Conv1d(xt, convs1[i].weight, dilation=D[i])
//!     xt = Activation1d(xt, alpha2_i, beta2_i)
//!     xt = Conv1d(xt, convs2[i].weight, dilation=1)
//!     x += xt
//!
//! Activation1d (kaiser ratio=2 anti-alias):
//!   x = replicate_pad(x, (5, 5))
//!   x = 2 * conv_transpose1d(x, upsample.filter [1,1,12] broadcast to C groups,
//!                            stride=2)
//!   x = x[..., 15 : -15]      # pad_left=15, pad_right=15
//!   x = SnakeBeta(x, alpha, beta)  # per-channel
//!   x = replicate_pad(x, (5, 6))
//!   x = conv1d(x, downsample.lowpass.filter [1,1,12] broadcast to C groups,
//!              stride=2)
//!
//! SnakeBeta: `x + (1/(exp(beta) + eps)) * sin²(exp(alpha) * x)`
//! alpha, beta are both per-channel and log-scale.

use flame_core::conv1d::{conv1d, conv1d_grouped};
use flame_core::{serialization, CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

const CONV_PRE_KERNEL: usize = 7;
const CONV_POST_KERNEL: usize = 7;

// Production config (matches checkpoint inspection):
const UPSAMPLE_RATES: [usize; 6] = [5, 2, 2, 2, 2, 2];
const UPSAMPLE_KERNEL_SIZES: [usize; 6] = [11, 4, 4, 4, 4, 4];
const INITIAL_CHANNELS: usize = 1536;
const NUM_KERNELS_PER_STAGE: usize = 3; // AMPBlock1 count per upsample stage
/// Per-resblock kernel size (Python's `resblock_kernel_sizes`).
const RESBLOCK_KERNEL_SIZES: [usize; 3] = [3, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];

/// Matches Python's `get_padding(kernel_size, dilation)` in vocoder.py.
#[inline]
fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

// Activation1d kaiser ratio=2 parameters:
//   kernel_size = 12
//   up:    pad = kernel_size/ratio - 1 = 5
//          pad_left  = 5*2 + (12-2)//2 = 15
//          pad_right = 5*2 + (12-2+1)//2 = 15
//   down:  pad_left  = 5, pad_right = 6 (LowPassFilter1d kernel=12 even)
const ACT_UP_REPLICATE_PAD: usize = 5;
const ACT_UP_SLICE_LEFT: usize = 15;
const ACT_UP_SLICE_RIGHT: usize = 15;
const ACT_DOWN_PAD_LEFT: usize = 5;
const ACT_DOWN_PAD_RIGHT: usize = 6;
const ACT_RATIO: usize = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidOperation(format!("Vocoder: missing weight: {key}")))?
        .to_dtype(DType::BF16)
}

/// Replicate-pad along the last (length) axis of `[B, C, L]`.
///
/// Uses `Tensor::expand` (broadcast view, no kernel launch) for the border
/// replicas so the whole thing becomes `narrow + expand + narrow + expand
/// + cat` — down from 5 full ops to 1 kernel-launching op (the cat).
fn replicate_pad1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    let d = x.shape().dims();
    let (b, c, l) = (d[0], d[1], d[2]);
    if pad_left == 0 && pad_right == 0 {
        return Ok(x.clone());
    }
    // Build the left/right border tensors as broadcast views when possible.
    let left_view;
    let right_view;
    let mut refs: Vec<&Tensor> = Vec::with_capacity(3);
    if pad_left > 0 {
        left_view = x.narrow(2, 0, 1)?.expand(&[b, c, pad_left])?;
        refs.push(&left_view);
    } else {
        left_view = Tensor::zeros_dtype(Shape::from_dims(&[0]), x.dtype(), x.device().clone())?;
    }
    refs.push(x);
    if pad_right > 0 {
        right_view = x.narrow(2, l - 1, 1)?.expand(&[b, c, pad_right])?;
        refs.push(&right_view);
    } else {
        right_view = Tensor::zeros_dtype(Shape::from_dims(&[0]), x.dtype(), x.device().clone())?;
    }
    let out = Tensor::cat(&refs, 2)?;
    // `left_view` / `right_view` are kept alive until `cat` returns.
    let _ = (&left_view, &right_view);
    Ok(out)
}

/// Zero-pad along the last axis of `[B, C, L]` with left/right zero lengths.
fn zero_pad1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    x.pad1d(pad_left, pad_right)
}

/// Insert `(stride - 1)` zeros between each element on the last axis.
/// Input `[B, C, L]` → output `[B, C, L*stride - (stride - 1)]`.
/// Implementation: reshape → pad with zeros to the right → reshape → trim.
fn zero_insert1d(x: &Tensor, stride: usize) -> Result<Tensor> {
    if stride <= 1 {
        return Ok(x.clone());
    }
    let d = x.shape().dims();
    let (b, c, l) = (d[0], d[1], d[2]);

    // Build [B, C, L, stride] by concat [x_unsq, zeros, zeros, ...] along dim 3.
    // Actually simpler: reshape x to [B, C, L, 1] then cat with a zero tensor
    // of shape [B, C, L, stride-1].
    let x_unsq = x.reshape(&[b, c, l, 1])?;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[b, c, l, stride - 1]),
        x.dtype(),
        x.device().clone(),
    )?;
    let cat = Tensor::cat(&[&x_unsq, &zeros], 3)?; // [B, C, L, stride]
    let interleaved = cat.reshape(&[b, c, l * stride])?;
    // Trim trailing (stride-1) zeros — output length = (L-1)*stride + 1
    let out_len = (l - 1) * stride + 1;
    interleaved.narrow(2, 0, out_len)
}

/// SnakeBeta activation: `x + (1/(exp(beta) + eps)) * sin²(exp(alpha) * x)`
///
/// `alpha_exp` and `inv_beta_exp_eps` are precomputed at load time as
/// broadcast-ready `[1, C, 1]` tensors so the hot path becomes
/// `sin²(a·x) * inv_b + x` with four kernel launches instead of nine.
fn snake_beta_fast(x: &Tensor, alpha_exp: &Tensor, inv_beta_exp_eps: &Tensor) -> Result<Tensor> {
    let ax = x.mul(alpha_exp)?;
    let sin_ax = ax.sin()?;
    let sin_sq = sin_ax.mul(&sin_ax)?;
    let scaled = sin_sq.mul(inv_beta_exp_eps)?;
    x.add(&scaled)
}

/// Grouped Conv1d with a shared [1,1,K] filter broadcast to `[C, 1, K]`.
/// Used for both Activation1d's upsample (via ConvTranspose1d) and downsample.
fn grouped_filter_broadcast(filter_1_1_k: &Tensor, channels: usize) -> Result<Tensor> {
    // filter: [1, 1, K] → [C, 1, K]
    let k = filter_1_1_k.shape().dims()[2];
    filter_1_1_k.expand(&[channels, 1, k])
}

/// Insert `(stride - 1)` zeros between each element on the last axis.
/// Matches `flame_core::conv1d::zero_insert_last_axis` but inlined here so
/// the hot path can stay entirely in vocoder code without round-tripping
/// through `conv_transpose1d`.
fn zero_insert_len(x: &Tensor, stride: usize) -> Result<Tensor> {
    if stride <= 1 {
        return Ok(x.clone());
    }
    let d = x.shape().dims();
    let (b, c, l) = (d[0], d[1], d[2]);
    let x4 = x.reshape(&[b, c, l, 1])?;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[b, c, l, stride - 1]),
        x.dtype(),
        x.device().clone(),
    )?;
    let cat = Tensor::cat(&[&x4, &zeros], 3)?;
    let flat = cat.reshape(&[b, c, l * stride])?;
    flat.narrow(2, 0, (l - 1) * stride + 1)
}

/// Flip the last axis of a 3D tensor via narrow + cat.
fn flip_last_axis(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    let k = dims[dims.len() - 1];
    if k <= 1 {
        return Ok(x.clone());
    }
    let mut parts: Vec<Tensor> = Vec::with_capacity(k);
    for i in (0..k).rev() {
        parts.push(x.narrow(dims.len() - 1, i, 1)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dims.len() - 1)
}

/// Precompute a ConvTranspose1d weight `[C_in, C_out/g, K]` into the
/// regular Conv1d layout `[C_out, C_in/g, K]`. This is the expensive part
/// of `conv_transpose1d` that we want to do ONCE at load time instead of
/// every forward. For groups=1 (full vocoder ups), this is a flip + global
/// transpose of the C_in and C_out axes.
fn precompute_conv_transpose_weight(
    weight: &Tensor,
    groups: usize,
) -> Result<Tensor> {
    let dims = weight.shape().dims();
    let (c_in, c_out_per_group, k) = (dims[0], dims[1], dims[2]);
    let c_in_per_group = c_in / groups;
    let c_out = c_out_per_group * groups;

    let flipped = flip_last_axis(weight)?;
    let grouped = flipped.reshape(&[groups, c_in_per_group, c_out_per_group, k])?;
    let permuted = grouped.permute(&[0, 2, 1, 3])?;
    permuted.reshape(&[c_out, c_in_per_group, k])
}

/// Activation1d: upsample 2× (Kaiser sinc) → SnakeBeta → downsample 2× (Kaiser sinc).
///
/// CRITICAL optimization: every op in this function is depthwise (per-channel).
/// At C=768 (vocoder stage 0), running everything at `groups=C` makes cuDNN's
/// grouped conv take ~4 ms and makes flame-core's `cat` loop per-channel at
/// ~4 ms too. Reshaping the input to `[B*C, 1, L]` up front collapses the
/// channel dim into the batch dim, lets cuDNN use the fast `groups=1` path
/// and lets `cat` process a single outer dim — both drop from ~4 ms to
/// ~150 µs per call. On a 16-mel-frame input this saves ~20 seconds of
/// wall time.
///
/// The per-channel SnakeBeta parameters are reshaped to `[C, 1, 1]` so they
/// broadcast elementwise against the `[C, 1, L']` tensors.
fn activation1d(
    x: &Tensor,
    alpha_exp_bc1: &Tensor, // shape [C, 1, 1] (depthwise layout)
    inv_beta_exp_eps_bc1: &Tensor, // shape [C, 1, 1]
    up_filter_1_1_k: &Tensor, // shape [1, 1, 12] — single-channel filter
    down_filter_1_1_k: &Tensor, // shape [1, 1, 12]
) -> Result<Tensor> {
    let in_dims = x.shape().dims().to_vec();
    let (b, c, l) = (in_dims[0], in_dims[1], in_dims[2]);

    // Fold channels into batch: [B, C, L] → [B*C, 1, L].
    let x_bc = x.reshape(&[b * c, 1, l])?;

    // --- Upsample (ConvTranspose1d stride=2 kernel=12, groups=1 after fold) ---
    let x_pad = replicate_pad1d(&x_bc, ACT_UP_REPLICATE_PAD, ACT_UP_REPLICATE_PAD)?;
    let x_zi = zero_insert_len(&x_pad, ACT_RATIO)?;
    // Conv1d side padding for stride=1 equivalent: dilation*(K-1) - padding = 1*11 - 0 = 11.
    let x_padded = x_zi.pad1d(11, 11)?;
    let y = conv1d(&x_padded, up_filter_1_1_k, None, 1, 0, 1, 1)?;
    let y = y.mul_scalar(ACT_RATIO as f32)?;
    // Slice [pad_left : -pad_right]
    let y_len = y.shape().dims()[2];
    let y = y.narrow(2, ACT_UP_SLICE_LEFT, y_len - ACT_UP_SLICE_LEFT - ACT_UP_SLICE_RIGHT)?;

    // --- Snake (4 ops, broadcast per-channel via the [C, 1, 1] shape) ---
    // y has shape [B*C, 1, L'], alpha_exp_bc1 has shape [C, 1, 1].
    // For B=1, B*C == C so they broadcast directly.
    // For B>1, alpha would need to be tiled; we assume B=1 (LTX-2.3 vocoder).
    debug_assert_eq!(b, 1, "depthwise activation1d assumes B=1");
    let y = snake_beta_fast(&y, alpha_exp_bc1, inv_beta_exp_eps_bc1)?;

    // --- Downsample (regular conv1d, stride=2) ---
    let y_pad = replicate_pad1d(&y, ACT_DOWN_PAD_LEFT, ACT_DOWN_PAD_RIGHT)?;
    let out = conv1d(&y_pad, down_filter_1_1_k, None, ACT_RATIO, 0, 1, 1)?;

    // Unfold back to [B, C, L_out].
    let l_out = out.shape().dims()[2];
    out.reshape(&[b, c, l_out])
}

// ---------------------------------------------------------------------------
// Resblock filter/param cache — broadcast filters once to save work.
// ---------------------------------------------------------------------------

struct ActParams {
    /// Precomputed `exp(alpha)` reshaped to `[C, 1, 1]` so it broadcasts
    /// against the folded `[B*C=C, 1, L]` hot-path tensors.
    alpha_exp_bc1: Tensor,
    /// Precomputed `1 / (exp(beta) + eps)` in the same `[C, 1, 1]` layout.
    inv_beta_exp_eps_bc1: Tensor,
    /// Kaiser sinc up-filter in single-channel `[1, 1, K]` layout. Symmetric,
    /// so the flip that a ConvTranspose1d would do is a no-op.
    up_filter_1_1_k: Tensor,
    /// Kaiser sinc down-filter in single-channel `[1, 1, K]` layout.
    down_filter_1_1_k: Tensor,
}

impl ActParams {
    fn load(weights: &Weights, prefix: &str, _channels: usize) -> Result<Self> {
        // `_channels` is kept for API symmetry with the previous broadcast
        // path; we now read C directly from the alpha tensor shape.
        let alpha = get(weights, &format!("{prefix}.act.alpha"))?;
        let beta = get(weights, &format!("{prefix}.act.beta"))?;
        let up_raw = get(weights, &format!("{prefix}.upsample.filter"))?;
        let down_raw = get(weights, &format!("{prefix}.downsample.lowpass.filter"))?;

        let c = alpha.shape().dims()[0];
        // [C] → [C, 1, 1] so it broadcasts against [B*C, 1, L] (B=1).
        let alpha_exp_bc1 = alpha.reshape(&[c, 1, 1])?.exp()?;
        let beta_exp_bc1 = beta.reshape(&[c, 1, 1])?.exp()?;
        let inv_beta_exp_eps_bc1 = beta_exp_bc1.add_scalar(1e-9)?.reciprocal()?;

        // The checkpoint stores both filters as [1, 1, K]; use them as-is.
        Ok(Self {
            alpha_exp_bc1,
            inv_beta_exp_eps_bc1,
            up_filter_1_1_k: up_raw,
            down_filter_1_1_k: down_raw,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        activation1d(
            x,
            &self.alpha_exp_bc1,
            &self.inv_beta_exp_eps_bc1,
            &self.up_filter_1_1_k,
            &self.down_filter_1_1_k,
        )
    }
}

struct AmpBlock1 {
    channels: usize,
    /// Kernel size shared by all 6 Conv1d layers in this block.
    kernel_size: usize,
    convs1_w: [Tensor; 3],
    convs1_b: [Tensor; 3],
    convs2_w: [Tensor; 3],
    convs2_b: [Tensor; 3],
    acts1: [ActParams; 3],
    acts2: [ActParams; 3],
}

impl AmpBlock1 {
    fn load(weights: &Weights, prefix: &str, channels: usize, kernel_size: usize) -> Result<Self> {
        let mut c1_w = Vec::with_capacity(3);
        let mut c1_b = Vec::with_capacity(3);
        let mut c2_w = Vec::with_capacity(3);
        let mut c2_b = Vec::with_capacity(3);
        for i in 0..3 {
            c1_w.push(get(weights, &format!("{prefix}.convs1.{i}.weight"))?);
            c1_b.push(get(weights, &format!("{prefix}.convs1.{i}.bias"))?);
            c2_w.push(get(weights, &format!("{prefix}.convs2.{i}.weight"))?);
            c2_b.push(get(weights, &format!("{prefix}.convs2.{i}.bias"))?);
        }
        let acts1 = [
            ActParams::load(weights, &format!("{prefix}.acts1.0"), channels)?,
            ActParams::load(weights, &format!("{prefix}.acts1.1"), channels)?,
            ActParams::load(weights, &format!("{prefix}.acts1.2"), channels)?,
        ];
        let acts2 = [
            ActParams::load(weights, &format!("{prefix}.acts2.0"), channels)?,
            ActParams::load(weights, &format!("{prefix}.acts2.1"), channels)?,
            ActParams::load(weights, &format!("{prefix}.acts2.2"), channels)?,
        ];
        Ok(Self {
            channels,
            kernel_size,
            convs1_w: [c1_w[0].clone(), c1_w[1].clone(), c1_w[2].clone()],
            convs1_b: [c1_b[0].clone(), c1_b[1].clone(), c1_b[2].clone()],
            convs2_w: [c2_w[0].clone(), c2_w[1].clone(), c2_w[2].clone()],
            convs2_b: [c2_b[0].clone(), c2_b[1].clone(), c2_b[2].clone()],
            acts1,
            acts2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let profile_deep = std::env::var("FLAME_VOCODER_DEEP").is_ok();
        let mut x = x.clone();
        let k = self.kernel_size;
        let mut t_a1 = 0.0_f32;
        let mut t_c1 = 0.0_f32;
        let mut t_a2 = 0.0_f32;
        let mut t_c2 = 0.0_f32;
        let mut t_add = 0.0_f32;
        let device = x.device().clone();
        let sync = || {
            if profile_deep {
                let _ = device.synchronize();
            }
        };
        let now = || std::time::Instant::now();

        for i in 0..3 {
            let d = DILATIONS[i];
            let pad1 = get_padding(k, d);

            let t = now();
            let xt = self.acts1[i].apply(&x)?;
            sync();
            t_a1 += t.elapsed().as_secs_f32();

            let t = now();
            let xt = conv1d(&xt, &self.convs1_w[i], Some(&self.convs1_b[i]), 1, pad1, d, 1)?;
            sync();
            t_c1 += t.elapsed().as_secs_f32();

            let t = now();
            let xt = self.acts2[i].apply(&xt)?;
            sync();
            t_a2 += t.elapsed().as_secs_f32();

            let t = now();
            let pad2 = get_padding(k, 1);
            let xt = conv1d(&xt, &self.convs2_w[i], Some(&self.convs2_b[i]), 1, pad2, 1, 1)?;
            sync();
            t_c2 += t.elapsed().as_secs_f32();

            let t = now();
            x = x.add(&xt)?;
            sync();
            t_add += t.elapsed().as_secs_f32();
        }

        if profile_deep {
            eprintln!(
                "  [amp k={k}] a1={t_a1:.3} c1={t_c1:.3} a2={t_a2:.3} c2={t_c2:.3} add={t_add:.3}"
            );
        }
        let _ = self.channels;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Vocoder
// ---------------------------------------------------------------------------

pub struct LTX2Vocoder {
    conv_pre_w: Tensor,
    conv_pre_b: Tensor,
    /// Pre-transformed ConvTranspose1d → Conv1d weights for each ups stage.
    /// Shape `[C_out, C_in, K]` after `precompute_conv_transpose_weight`.
    /// Stored alongside the bias and the stride/kernel metadata.
    ups: Vec<(Tensor, Tensor)>, // (preconv_weight, bias)
    resblocks: Vec<AmpBlock1>,
    act_post: ActParams,
    conv_post_w: Tensor,
    conv_post_b: Option<Tensor>,
    channels_per_stage: Vec<usize>, // after each ups[i]
    final_channels: usize,
}

impl LTX2Vocoder {
    pub fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let conv_pre_w = get(weights, &format!("{prefix}.conv_pre.weight"))?;
        let conv_pre_b = get(weights, &format!("{prefix}.conv_pre.bias"))?;

        let mut ups = Vec::with_capacity(6);
        let mut channels_per_stage = Vec::with_capacity(6);
        for i in 0..UPSAMPLE_RATES.len() {
            let w_raw = get(weights, &format!("{prefix}.ups.{i}.weight"))?;
            let b = get(weights, &format!("{prefix}.ups.{i}.bias"))?;
            // ConvTranspose1d weight shape: [C_in, C_out, K]
            let c_out = w_raw.shape().dims()[1];
            channels_per_stage.push(c_out);
            // Pre-transform once at load time — this is the most expensive
            // op in the old hot path (flip + permute on a 26 MB weight).
            let w_preconv = precompute_conv_transpose_weight(&w_raw, 1)?;
            ups.push((w_preconv, b));
        }

        let final_channels = *channels_per_stage.last().unwrap();

        let mut resblocks = Vec::with_capacity(18);
        for stage_i in 0..UPSAMPLE_RATES.len() {
            let ch = channels_per_stage[stage_i];
            for kernel_i in 0..NUM_KERNELS_PER_STAGE {
                let block_idx = stage_i * NUM_KERNELS_PER_STAGE + kernel_i;
                let k = RESBLOCK_KERNEL_SIZES[kernel_i];
                resblocks.push(AmpBlock1::load(
                    weights,
                    &format!("{prefix}.resblocks.{block_idx}"),
                    ch,
                    k,
                )?);
            }
        }

        let act_post = ActParams::load(weights, &format!("{prefix}.act_post"), final_channels)?;

        let conv_post_w = get(weights, &format!("{prefix}.conv_post.weight"))?;
        let conv_post_b = weights
            .get(&format!("{prefix}.conv_post.bias"))
            .map(|t| t.to_dtype(DType::BF16))
            .transpose()?;

        Ok(Self {
            conv_pre_w,
            conv_pre_b,
            ups,
            resblocks,
            act_post,
            conv_post_w,
            conv_post_b,
            channels_per_stage,
            final_channels,
        })
    }

    /// Forward: mel `[B, 2, T_mel, F_mel]` → waveform `[B, 2, T_out]` at the
    /// inner `input_sampling_rate` (16 kHz for production).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Set FLAME_VOCODER_PROFILE=1 for per-phase timing.
        let profile = std::env::var("FLAME_VOCODER_PROFILE").is_ok();
        let device = mel.device().clone();
        let sync = || {
            if profile {
                let _ = device.synchronize();
            }
        };
        let now = || std::time::Instant::now();

        // Normalize to [B, 2*F_mel, T_mel]
        let d = mel.shape().dims();
        assert_eq!(d[1], 2, "stereo mel expected at dim 1");
        let (b, _s, t, f) = (d[0], d[1], d[2], d[3]);
        // [B, 2, T, F] → [B, 2, F, T] → [B, 2*F, T]
        let x = mel.permute(&[0, 1, 3, 2])?.reshape(&[b, 2 * f, t])?;

        let t0 = now();
        // conv_pre: k=7, pad=3
        let mut x = conv1d(
            &x,
            &self.conv_pre_w,
            Some(&self.conv_pre_b),
            1,
            CONV_PRE_KERNEL / 2,
            1,
            1,
        )?;
        sync();
        if profile {
            eprintln!("[voc] conv_pre:    {:.3}s", t0.elapsed().as_secs_f32());
        }

        for i in 0..UPSAMPLE_RATES.len() {
            let stride = UPSAMPLE_RATES[i];
            let k = UPSAMPLE_KERNEL_SIZES[i];
            let padding = (k - stride) / 2;
            let (w_preconv, bi) = &self.ups[i];

            let t_up = now();
            // ConvTranspose1d via cached preconv weight: zero-insert → pad →
            // regular cuDNN conv1d. The weight transform that used to run
            // here on every call is now done once at load time.
            let x_zi = zero_insert_len(&x, stride)?;
            let side_pad = (k - 1) - padding; // dilation = 1
            let x_padded = x_zi.pad1d(side_pad, side_pad)?;
            x = conv1d(&x_padded, w_preconv, Some(bi), 1, 0, 1, 1)?;
            sync();
            let t_up_done = t_up.elapsed().as_secs_f32();

            // 3 resblocks averaged
            let start = i * NUM_KERNELS_PER_STAGE;
            let t_rb = now();
            let mut sum: Option<Tensor> = None;
            for j in 0..NUM_KERNELS_PER_STAGE {
                let out = self.resblocks[start + j].forward(&x)?;
                sum = Some(match sum {
                    None => out,
                    Some(s) => s.add(&out)?,
                });
            }
            x = sum.unwrap().mul_scalar(1.0 / NUM_KERNELS_PER_STAGE as f32)?;
            sync();
            let t_rb_done = t_rb.elapsed().as_secs_f32();

            if profile {
                let l = x.shape().dims()[2];
                eprintln!("[voc] stage {i} (L={l}): ups={t_up_done:.3}s rblocks={t_rb_done:.3}s");
            }
        }

        // act_post + conv_post + tanh
        let t_post = now();
        x = self.act_post.apply(&x)?;
        sync();
        let t_ap = t_post.elapsed().as_secs_f32();
        let t_cp = now();
        x = conv1d(
            &x,
            &self.conv_post_w,
            self.conv_post_b.as_ref(),
            1,
            CONV_POST_KERNEL / 2,
            1,
            1,
        )?;
        let out = x.tanh()?;
        sync();
        if profile {
            eprintln!("[voc] act_post:   {t_ap:.3}s");
            eprintln!("[voc] conv_post+tanh: {:.3}s", t_cp.elapsed().as_secs_f32());
        }
        Ok(out)
    }

    /// Load from a full LTX-2.3 safetensors checkpoint. Uses filtered
    /// GPU load for only `vocoder.{vocoder,bwe_generator}.*` keys.
    pub fn from_file(path: &str, device: &Arc<CudaDevice>, inner_prefix: &str) -> Result<Self> {
        eprintln!("Loading LTX-2.3 Vocoder ({inner_prefix}) from: {path}");
        let full_prefix = format!("vocoder.{inner_prefix}.");
        let raw = serialization::load_file_filtered(
            std::path::Path::new(path),
            device,
            |k| k.starts_with(&full_prefix),
        )?;
        let mut normalized: Weights = HashMap::new();
        let strip = "vocoder.";
        for (key, value) in raw {
            let short = key.strip_prefix(strip).unwrap_or(&key).to_string();
            normalized.insert(short, value);
        }
        eprintln!("  {} vocoder keys loaded", normalized.len());
        Self::load(&normalized, inner_prefix)
    }

    pub fn final_channels(&self) -> usize {
        self.final_channels
    }
}

// Silence unused warnings on const tables we only index.
const _: usize = UPSAMPLE_RATES[0] + UPSAMPLE_KERNEL_SIZES[0] + INITIAL_CHANNELS;
