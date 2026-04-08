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

use flame_core::conv1d::{conv1d, conv1d_grouped, conv_transpose1d};
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
fn replicate_pad1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    let d = x.shape().dims();
    let (b, c, l) = (d[0], d[1], d[2]);
    if pad_left == 0 && pad_right == 0 {
        return Ok(x.clone());
    }
    let mut parts: Vec<Tensor> = Vec::new();
    if pad_left > 0 {
        let left = x.narrow(2, 0, 1)?.repeat_axis_device(2, pad_left)?;
        parts.push(left);
    }
    parts.push(x.clone());
    if pad_right > 0 {
        let right = x.narrow(2, l - 1, 1)?.repeat_axis_device(2, pad_right)?;
        parts.push(right);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let out = Tensor::cat(&refs, 2)?;
    let _ = b;
    let _ = c;
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
fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    // alpha, beta: [C] → [1, C, 1]
    let c = alpha.shape().dims()[0];
    let a = alpha.reshape(&[1, c, 1])?.exp()?;
    let b = beta.reshape(&[1, c, 1])?.exp()?;
    let eps = 1e-9f32;
    // sin(a * x)
    let ax = x.mul(&a)?;
    let sin_ax = ax.sin()?;
    let sin_sq = sin_ax.mul(&sin_ax)?;
    // 1 / (b + eps)
    let inv_b = b.add_scalar(eps)?.reciprocal()?;
    let scaled = sin_sq.mul(&inv_b)?;
    x.add(&scaled)
}

/// Grouped Conv1d with a shared [1,1,K] filter broadcast to `[C, 1, K]`.
/// Used for both Activation1d's upsample (via ConvTranspose1d) and downsample.
fn grouped_filter_broadcast(filter_1_1_k: &Tensor, channels: usize) -> Result<Tensor> {
    // filter: [1, 1, K] → [C, 1, K]
    let k = filter_1_1_k.shape().dims()[2];
    filter_1_1_k.expand(&[channels, 1, k])
}

/// Activation1d: upsample 2× (Kaiser sinc) → SnakeBeta → downsample 2× (Kaiser sinc).
fn activation1d(
    x: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    up_filter_broadcast: &Tensor,
    down_filter_broadcast: &Tensor,
) -> Result<Tensor> {
    let c = x.shape().dims()[1];

    // --- Upsample ---
    let x_pad = replicate_pad1d(x, ACT_UP_REPLICATE_PAD, ACT_UP_REPLICATE_PAD)?;
    // Native flame-core conv_transpose1d (groups=c, stride=2, padding=0).
    let y = conv_transpose1d(
        &x_pad,
        up_filter_broadcast,
        None,
        ACT_RATIO, // stride
        0,         // padding
        0,         // output_padding
        c,         // groups
    )?;
    let y = y.mul_scalar(ACT_RATIO as f32)?;
    // Slice [pad_left : -pad_right]
    let y_len = y.shape().dims()[2];
    let y = y.narrow(2, ACT_UP_SLICE_LEFT, y_len - ACT_UP_SLICE_LEFT - ACT_UP_SLICE_RIGHT)?;

    // --- Snake ---
    let y = snake_beta(&y, alpha, beta)?;

    // --- Downsample ---
    let y_pad = replicate_pad1d(&y, ACT_DOWN_PAD_LEFT, ACT_DOWN_PAD_RIGHT)?;
    // grouped conv1d with stride=2
    conv1d_grouped(&y_pad, down_filter_broadcast, ACT_RATIO, 0, c)
}

// ---------------------------------------------------------------------------
// Resblock filter/param cache — broadcast filters once to save work.
// ---------------------------------------------------------------------------

struct ActParams {
    alpha: Tensor,
    beta: Tensor,
    up_filter: Tensor,   // broadcast to [C, 1, 12]
    down_filter: Tensor, // broadcast to [C, 1, 12]
}

impl ActParams {
    fn load(weights: &Weights, prefix: &str, channels: usize) -> Result<Self> {
        let alpha = get(weights, &format!("{prefix}.act.alpha"))?;
        let beta = get(weights, &format!("{prefix}.act.beta"))?;
        let up_raw = get(weights, &format!("{prefix}.upsample.filter"))?;
        let down_raw = get(weights, &format!("{prefix}.downsample.lowpass.filter"))?;
        Ok(Self {
            alpha,
            beta,
            up_filter: grouped_filter_broadcast(&up_raw, channels)?,
            down_filter: grouped_filter_broadcast(&down_raw, channels)?,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        activation1d(
            x,
            &self.alpha,
            &self.beta,
            &self.up_filter,
            &self.down_filter,
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
        let mut x = x.clone();
        let k = self.kernel_size;
        for i in 0..3 {
            let d = DILATIONS[i];
            let pad1 = get_padding(k, d);
            // a1
            let xt = self.acts1[i].apply(&x)?;
            // conv1: kernel `k`, dilation `d`, padding `d*(k-1)/2` preserves length.
            let xt = conv1d(&xt, &self.convs1_w[i], Some(&self.convs1_b[i]), 1, pad1, d, 1)?;
            // a2
            let xt = self.acts2[i].apply(&xt)?;
            // conv2 with kernel k, dilation=1, padding=(k-1)/2
            let pad2 = get_padding(k, 1);
            let xt = conv1d(&xt, &self.convs2_w[i], Some(&self.convs2_b[i]), 1, pad2, 1, 1)?;
            x = x.add(&xt)?;
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
    ups: Vec<(Tensor, Tensor)>, // (weight [C_in, C_out, K], bias [C_out])
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
            let w = get(weights, &format!("{prefix}.ups.{i}.weight"))?;
            let b = get(weights, &format!("{prefix}.ups.{i}.bias"))?;
            // ConvTranspose1d weight shape: [C_in, C_out, K]
            let c_out = w.shape().dims()[1];
            channels_per_stage.push(c_out);
            ups.push((w, b));
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
        // Normalize to [B, 2*F_mel, T_mel]
        let d = mel.shape().dims();
        assert_eq!(d[1], 2, "stereo mel expected at dim 1");
        let (b, _s, t, f) = (d[0], d[1], d[2], d[3]);
        // [B, 2, T, F] → [B, 2, F, T] → [B, 2*F, T]
        let x = mel.permute(&[0, 1, 3, 2])?.reshape(&[b, 2 * f, t])?;

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

        for i in 0..UPSAMPLE_RATES.len() {
            let stride = UPSAMPLE_RATES[i];
            let k = UPSAMPLE_KERNEL_SIZES[i];
            let padding = (k - stride) / 2;
            let (w, bi) = &self.ups[i];
            x = conv_transpose1d(&x, w, Some(bi), stride, padding, 0, 1)?;

            // 3 resblocks averaged
            let start = i * NUM_KERNELS_PER_STAGE;
            let mut sum: Option<Tensor> = None;
            for j in 0..NUM_KERNELS_PER_STAGE {
                let out = self.resblocks[start + j].forward(&x)?;
                sum = Some(match sum {
                    None => out,
                    Some(s) => s.add(&out)?,
                });
            }
            x = sum.unwrap().mul_scalar(1.0 / NUM_KERNELS_PER_STAGE as f32)?;
        }

        // act_post + conv_post + tanh
        x = self.act_post.apply(&x)?;
        x = conv1d(
            &x,
            &self.conv_post_w,
            self.conv_post_b.as_ref(),
            1,
            CONV_POST_KERNEL / 2,
            1,
            1,
        )?;
        x.tanh()
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
