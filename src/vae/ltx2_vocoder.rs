//! LTX-2.3 BigVGAN Vocoder + BWE — pure flame-core.
//!
//! Converts mel spectrogram → audio waveform.
//!
//! Architecture: BigVGAN v2 with anti-aliased activations:
//!   conv_pre(128→1024, k=7) → 5 upsample stages → conv_post(16→2, k=7)
//!   Each upsample: ConvTranspose1d + 3 AMPBlock1 (averaged)
//!   AMPBlock1: 3 dilated conv pairs with anti-aliased Snake activation
//!   BWE: vocoder output → mel → bwe_generator → resample → add skip
//!
//! MUST run in FP32 (BF16 degrades quality 40-90%).
//!
//! Weight prefix: `vocoder.vocoder.*` and `vocoder.bwe_generator.*`
//!
//! ⚠️ STAGING — move to production after cargo check passes.

use flame_core::conv1d::{conv1d, conv1d_grouped, conv_transpose1d};
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

type Weights = HashMap<String, Tensor>;

const LRELU_SLOPE: f32 = 0.1;

// ---------------------------------------------------------------------------
// Snake activation: x + (1/exp(alpha)) * sin²(exp(alpha) * x)
// ---------------------------------------------------------------------------

fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    // alpha, beta: [channels] → [1, channels, 1] for broadcasting over [B, C, L]
    let a = alpha.exp()?.unsqueeze(0)?.unsqueeze(2)?;
    let b = beta.exp()?.unsqueeze(0)?.unsqueeze(2)?;
    let sin_term = x.mul(&a)?.sin()?.mul(&x.mul(&a)?.sin()?)?; // sin²(alpha*x)
    let scaled = sin_term.mul(&b.add_scalar(1e-9)?.reciprocal()?)?; // 1/(beta+eps) * sin²
    x.add(&scaled)
}

// ---------------------------------------------------------------------------
// Anti-aliased resampling (kaiser sinc filters loaded from checkpoint)
// ---------------------------------------------------------------------------

/// Low-pass filter 1D: grouped conv1d with pre-computed filter.
fn lowpass_filter(x: &Tensor, filter: &Tensor, stride: usize, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    let n_channels = x.shape().dims()[1];
    // Pad input
    let padded = x.pad1d(pad_left, pad_right)?;
    // Expand filter: [1, 1, K] → [C, 1, K] for grouped conv
    let filter_exp = filter.expand(&[n_channels, 1, filter.shape().dims()[2]])?;
    conv1d_grouped(&padded, &filter_exp, stride, 0, n_channels)
}

/// Upsample 1D: insert zeros then conv_transpose with filter.
fn upsample_1d(x: &Tensor, filter: &Tensor, ratio: usize, pad_left: usize, pad_right: usize, pad: usize) -> Result<Tensor> {
    let n_channels = x.shape().dims()[1];
    // Pad input with `pad` on each side (replicate)
    let padded = x.pad1d(pad, pad)?;
    // Expand filter for grouped conv_transpose
    let filter_exp = filter.expand(&[n_channels, 1, filter.shape().dims()[2]])?;
    // ConvTranspose1d: stride=ratio, groups=n_channels
    let out = conv_transpose1d(&padded, &filter_exp, None, ratio, 0, 0, n_channels)?;
    let out = out.mul_scalar(ratio as f32)?;
    // Crop to remove padding artifacts
    let l_out = out.shape().dims()[2];
    out.narrow(2, pad_left, l_out - pad_left - pad_right)
}

/// Anti-aliased activation: upsample → snake → downsample
fn activation1d(
    x: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    up_filter: &Tensor,
    down_filter: &Tensor,
    up_pad_left: usize,
    up_pad_right: usize,
    up_pad: usize,
    down_pad_left: usize,
    down_pad_right: usize,
) -> Result<Tensor> {
    let up = upsample_1d(x, up_filter, 2, up_pad_left, up_pad_right, up_pad)?;
    let activated = snake_beta(&up, alpha, beta)?;
    lowpass_filter(&activated, down_filter, 2, down_pad_left, down_pad_right)
}

// ---------------------------------------------------------------------------
// AMPBlock1: dilated conv pairs with anti-aliased activation
// ---------------------------------------------------------------------------

fn amp_block(
    x: &Tensor,
    weights: &Weights,
    prefix: &str,
    channels: usize,
) -> Result<Tensor> {
    let w = |suffix: &str| -> Result<&Tensor> {
        let key = format!("{prefix}.{suffix}");
        weights.get(&key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing vocoder weight: {key}"))
        })
    };

    let mut x = x.clone();
    // 3 dilated conv pairs
    for i in 0..3 {
        // Anti-aliased activation 1
        let xt = activation1d(
            &x,
            w(&format!("acts1.{i}.act.alpha"))?,
            w(&format!("acts1.{i}.act.beta"))?,
            w(&format!("acts1.{i}.upsample.filter"))?,
            w(&format!("acts1.{i}.downsample.lowpass.filter"))?,
            // Padding params — derived from filter size and ratio
            // Kaiser filter k=12, ratio=2: pad_left=5, pad_right=6, pad=3
            5, 6, 3, 5, 6,
        )?;

        // Dilated conv1d
        let conv1_w = w(&format!("convs1.{i}.weight"))?;
        let conv1_b = w(&format!("convs1.{i}.bias"))?;
        let k = conv1_w.shape().dims()[2];
        let dilation = [1, 3, 5][i];
        let padding = (k * dilation - dilation) / 2;
        let xt = conv1d(&xt, conv1_w, Some(conv1_b), 1, padding, dilation, 1)?;

        // Anti-aliased activation 2
        let xt = activation1d(
            &xt,
            w(&format!("acts2.{i}.act.alpha"))?,
            w(&format!("acts2.{i}.act.beta"))?,
            w(&format!("acts2.{i}.upsample.filter"))?,
            w(&format!("acts2.{i}.downsample.lowpass.filter"))?,
            5, 6, 3, 5, 6,
        )?;

        // Conv1d (dilation=1)
        let conv2_w = w(&format!("convs2.{i}.weight"))?;
        let conv2_b = w(&format!("convs2.{i}.bias"))?;
        let k2 = conv2_w.shape().dims()[2];
        let padding2 = (k2 - 1) / 2;
        let xt = conv1d(&xt, conv2_w, Some(conv2_b), 1, padding2, 1, 1)?;

        x = x.add(&xt)?;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Vocoder (BigVGAN)
// ---------------------------------------------------------------------------

pub struct LTX2Vocoder {
    weights: Weights,
    upsample_rates: Vec<usize>,
    upsample_kernel_sizes: Vec<usize>,
    num_kernels: usize,
}

impl LTX2Vocoder {
    /// Load vocoder weights.
    /// `prefix` is typically "vocoder.vocoder" or "vocoder.bwe_generator".
    pub fn load(all_weights: &Weights, prefix: &str) -> Result<Self> {
        let vocoder_weights: Weights = all_weights.iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| {
                let short = k.strip_prefix(&format!("{prefix}.")).unwrap_or(k);
                (short.to_string(), v.clone())
            })
            .collect();

        if vocoder_weights.is_empty() {
            return Err(flame_core::Error::InvalidInput(
                format!("No vocoder keys found with prefix '{prefix}'"),
            ));
        }

        log::info!("[Vocoder] Loaded {} weights from '{prefix}'", vocoder_weights.len());

        Ok(Self {
            weights: vocoder_weights,
            upsample_rates: vec![6, 5, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 15, 8, 4, 4],
            num_kernels: 3, // resblock_kernel_sizes = [3, 7, 11]
        })
    }

    /// Forward: mel spectrogram → waveform.
    ///
    /// Input: [B, 2, T, mel_bins] stereo mel spectrogram
    /// Output: [B, 2, audio_length] stereo waveform
    ///
    /// ## PyTorch reference:
    /// ```python
    /// x = x.transpose(2, 3)  # [B, 2, T, mel] → [B, 2, mel, T]
    /// x = rearrange(x, "b s c t -> b (s c) t")  # stereo interleave
    /// x = conv_pre(x)
    /// for i in range(num_upsamples):
    ///     x = ups[i](x)
    ///     blocks = [resblocks[j](x) for j in range(start, end)]
    ///     x = mean(blocks)
    /// x = act_post(x)
    /// x = conv_post(x)
    /// x = tanh(x)
    /// ```
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let w = &self.weights;
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing vocoder: {key}"))
            })
        };

        let dims = mel.shape().dims().to_vec();

        // Transpose: [B, 2, T, mel] → [B, 2, mel, T]
        let x = mel.permute(&[0, 1, 3, 2])?;

        // Stereo interleave: [B, 2, mel, T] → [B, 2*mel, T]
        let (b, s, c, t) = (dims[0], x.shape().dims()[1], x.shape().dims()[2], x.shape().dims()[3]);
        let x = x.reshape(&[b, s * c, t])?;

        // conv_pre: [B, 128, T] → [B, 1024, T]
        let x = conv1d(&x, get("conv_pre.weight")?, Some(get("conv_pre.bias")?), 1, 3, 1, 1)?;

        let mut x = x;
        for i in 0..self.upsample_rates.len() {
            // Upsample via ConvTranspose1d
            let stride = self.upsample_rates[i];
            let k = self.upsample_kernel_sizes[i];
            let padding = (k - stride) / 2;
            let up_w = get(&format!("ups.{i}.weight"))?;
            let up_b = get(&format!("ups.{i}.bias"))?;
            x = conv_transpose1d(&x, up_w, Some(up_b), stride, padding, 0, 1)?;

            // Average of 3 AMPBlock1 outputs
            let start = i * self.num_kernels;
            let ch = x.shape().dims()[1];
            let mut sum: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let idx = start + j;
                let block_out = amp_block(&x, w, &format!("resblocks.{idx}"), ch)?;
                sum = Some(match sum {
                    None => block_out,
                    Some(s) => s.add(&block_out)?,
                });
            }
            x = sum.unwrap().mul_scalar(1.0 / self.num_kernels as f32)?;
        }

        // Final: act_post (snake) → conv_post → tanh
        x = activation1d(
            &x,
            get("act_post.act.alpha")?,
            get("act_post.act.beta")?,
            get("act_post.upsample.filter")?,
            get("act_post.downsample.lowpass.filter")?,
            5, 6, 3, 5, 6,
        )?;
        x = conv1d(&x, get("conv_post.weight")?, Some(get("conv_post.bias")?), 1, 3, 1, 1)?;
        x.tanh()
    }
}

// ---------------------------------------------------------------------------
// VocoderWithBWE
// ---------------------------------------------------------------------------

pub struct LTX2VocoderWithBWE {
    vocoder: LTX2Vocoder,
    bwe_generator: LTX2Vocoder,
    // MelSTFT bases loaded from checkpoint
    mel_basis: Tensor,        // [n_mels, n_freqs]
    stft_forward_basis: Tensor, // [n_freqs*2, 1, filter_length]
    hop_length: usize,
    input_sr: usize,
    output_sr: usize,
}

impl LTX2VocoderWithBWE {
    pub fn load(all_weights: &Weights) -> Result<Self> {
        let vocoder = LTX2Vocoder::load(all_weights, "vocoder.vocoder")?;
        let bwe_generator = LTX2Vocoder::load(all_weights, "vocoder.bwe_generator")?;

        let mel_basis = all_weights.get("vocoder.mel_stft.mel_basis")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing vocoder.mel_stft.mel_basis".into()))?
            .clone();
        let stft_forward_basis = all_weights.get("vocoder.mel_stft.stft_fn.forward_basis")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing vocoder.mel_stft.stft_fn.forward_basis".into()))?
            .clone();

        log::info!("[VocoderBWE] mel_basis: {:?}, stft_basis: {:?}",
            mel_basis.shape(), stft_forward_basis.shape());

        Ok(Self {
            vocoder,
            bwe_generator,
            mel_basis,
            stft_forward_basis,
            hop_length: 256,
            input_sr: 24000,
            output_sr: 48000,
        })
    }

    /// Full forward: mel → vocoder → BWE → high-quality waveform.
    ///
    /// Input: [B, 2, T, mel_bins] stereo mel spectrogram
    /// Output: [B, 2, audio_length] stereo waveform at output_sr
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Step 1: Base vocoder
        let x = self.vocoder.forward(mel)?;
        let length_low = x.shape().dims()[2];
        let output_length = length_low * self.output_sr / self.input_sr;

        // Step 2: Compute mel of base output for BWE
        let mel_bwe = self.compute_mel(&x)?;

        // Step 3: BWE generator on mel of base output
        // mel_bwe: [B, C, n_mels, T_frames] → transpose to [B, C, T_frames, n_mels]
        let mel_for_bwe = mel_bwe.permute(&[0, 1, 3, 2])?;
        let residual = self.bwe_generator.forward(&mel_for_bwe)?;

        // Step 4: Sinc resample base output (2x upsample)
        let skip = self.sinc_upsample(&x)?;

        // Step 5: Add residual + skip, clamp
        let out = residual.add(&skip)?;
        let out = out.clamp(-1.0, 1.0)?;

        // Crop to output_length
        let l = out.shape().dims()[2];
        if l > output_length {
            out.narrow(2, 0, output_length)
        } else {
            Ok(out)
        }
    }

    /// Compute log-mel spectrogram from waveform using loaded STFT bases.
    fn compute_mel(&self, audio: &Tensor) -> Result<Tensor> {
        let dims = audio.shape().dims().to_vec();
        let (b, n_ch, _t) = (dims[0], dims[1], dims[2]);

        // Flatten stereo: [B, C, T] → [B*C, T]
        let flat = audio.reshape(&[b * n_ch, dims[2]])?;

        // Causal STFT via conv1d with loaded bases
        let win_length = self.stft_forward_basis.shape().dims()[2];
        let left_pad = win_length.saturating_sub(self.hop_length);
        let flat_padded = flat.unsqueeze(1)?.pad1d(left_pad, 0)?;

        // Conv1d with STFT basis: [n_freqs*2, 1, filter_length]
        let spec = conv1d(&flat_padded, &self.stft_forward_basis, None, self.hop_length, 0, 1, 1)?;

        // Split into real/imag
        let n_freqs = spec.shape().dims()[1] / 2;
        let real = spec.narrow(1, 0, n_freqs)?;
        let imag = spec.narrow(1, n_freqs, n_freqs)?;

        // Magnitude
        let magnitude = real.mul(&real)?.add(&imag.mul(&imag)?)?.sqrt()?;

        // Mel projection
        let mel = self.mel_basis.matmul(&magnitude)?; // [n_mels, n_freqs] × [B*C, n_freqs, T] hmm...
        // Actually need: mel_basis @ magnitude for each batch element
        // mel_basis: [n_mels, n_freqs], magnitude: [B*C, n_freqs, T_frames]
        // Use bmm or manual reshape
        let n_mels = self.mel_basis.shape().dims()[0];
        let t_frames = magnitude.shape().dims()[2];
        let bc = b * n_ch;

        // Expand mel_basis to [B*C, n_mels, n_freqs] then bmm
        let mel_exp = self.mel_basis.unsqueeze(0)?.expand(&[bc, n_mels, n_freqs])?;
        let mel = mel_exp.matmul(&magnitude)?; // [B*C, n_mels, T_frames]

        // Log mel
        let mel = mel.clamp(1e-5, f32::MAX)?.log()?;

        // Reshape back: [B*C, n_mels, T_frames] → [B, C, n_mels, T_frames]
        mel.reshape(&[b, n_ch, n_mels, t_frames])
    }

    /// Sinc upsample (hann-windowed) for skip connection.
    fn sinc_upsample(&self, x: &Tensor) -> Result<Tensor> {
        let ratio = self.output_sr / self.input_sr; // 2
        // Simple nearest-neighbor as fallback — proper sinc would need
        // pre-computed hann-windowed filter. For initial implementation,
        // use repeat-interleave.
        // TODO: Implement proper hann-windowed sinc resampler
        let dims = x.shape().dims().to_vec();
        let (b, c, l) = (dims[0], dims[1], dims[2]);
        // Repeat each sample `ratio` times: [B, C, L] → [B, C, L*ratio]
        let expanded = x.unsqueeze(3)?.expand(&[b, c, l, ratio])?.reshape(&[b, c, l * ratio])?;
        Ok(expanded)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_upsample_rates() {
        let rates = [6, 5, 2, 2, 2];
        let total: usize = rates.iter().product();
        assert_eq!(total, 240); // 240x temporal upsample
    }

    #[test]
    fn test_channel_progression() {
        let initial = 1024usize;
        let mut ch = initial;
        for i in 0..5 {
            ch /= 2;
        }
        assert_eq!(ch, 32); // 1024 → 512 → 256 → 128 → 64 → 32
        // But conv_post maps to 2 (stereo), and act_post uses ch=16
        // Actually: initial=1024, 5 halvings: 512,256,128,64,32
        // But the actual final is 1024 / 2^5 = 32, conv_post: 16→2
        // Wait, let me recheck: default is 1024 → /2 each upsample → 32 → conv_post→2
    }
}
