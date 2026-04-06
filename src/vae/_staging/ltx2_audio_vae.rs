//! LTX-2.3 Audio VAE Decoder — pure flame-core.
//!
//! Decodes audio latents [B, 8, T, mel_bins] → mel spectrogram [B, 2, T', mel_bins].
//!
//! Architecture: standard 2D LDM-style VAE decoder with:
//!   - Causal conv2d (pad width axis only)
//!   - GroupNorm normalization
//!   - SiLU activation
//!   - Mid block: 2 ResNet blocks + self-attention (Conv2d 1x1)
//!   - 4 up stages with ResNet blocks + upsample convs
//!   - Per-channel statistics denormalization
//!
//! Weight key prefix: `audio_vae.decoder.*`
//!
//! ⚠️ STAGING — move to production after cargo check passes.

use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

type Weights = HashMap<String, Tensor>;

const LATENT_DOWNSAMPLE_FACTOR: usize = 4;

// ---------------------------------------------------------------------------
// Per-channel statistics (denormalize latents before decode)
// ---------------------------------------------------------------------------

/// Denormalize audio latents using stored mean/std statistics.
///
/// ## PyTorch reference:
/// ```python
/// def un_normalize(self, x):
///     return x * self.channel_std + self.channel_mean
/// ```
fn denormalize(
    x: &Tensor,
    mean: &Tensor,  // [channels]
    std: &Tensor,   // [channels]
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let c = dims[1];
    // Reshape mean/std to [1, C, 1, 1] for broadcasting
    let mean_4d = mean.reshape(&[1, c, 1, 1])?;
    let std_4d = std.reshape(&[1, c, 1, 1])?;
    x.mul(&std_4d.expand(&dims)?)?.add(&mean_4d.expand(&dims)?)
}

// ---------------------------------------------------------------------------
// Causal Conv2d helper
// ---------------------------------------------------------------------------

/// Causal 2D convolution: pad width (frequency) axis only.
/// For k=3: pad_left=1, pad_right=1 on width axis; no height padding.
///
/// Uses flame-core's Conv2d infrastructure.
fn causal_conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    let k_h = weight.shape().dims()[2];
    let k_w = weight.shape().dims()[3];
    let pad_h = (k_h - 1) / 2; // symmetric padding on height (time)
    let pad_w = (k_w - 1) / 2; // symmetric padding on width (freq)

    // Use flame-core conv2d with padding
    // For now, use manual padding + conv with no padding
    // TODO: Use flame_core::conv::Conv2d when available with proper padding

    // Pad: [pad_w_left, pad_w_right, pad_h_top, pad_h_bottom]
    let padded = x.pad(&[pad_w, pad_w, pad_h, pad_h])?;

    // Conv2d with no padding (already padded manually)
    flame_core::conv::conv2d_forward(&padded, weight, bias, stride, 0)
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

fn group_norm_32(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    flame_core::group_norm::group_norm(x, 32, Some(weight), Some(bias), 1e-6)
}

// ---------------------------------------------------------------------------
// ResNet Block (2D)
// ---------------------------------------------------------------------------

/// Audio VAE ResNet block.
///
/// ## PyTorch reference:
/// ```python
/// h = self.norm1(x)
/// h = silu(h)
/// h = self.conv1(h)
/// h = self.norm2(h)
/// h = silu(h)
/// h = self.conv2(h)
/// if in_ch != out_ch: x = self.nin_shortcut(x)
/// return x + h
/// ```
fn resnet_block(
    x: &Tensor,
    weights: &Weights,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
) -> Result<Tensor> {
    let w = |suffix: &str| -> Result<&Tensor> {
        let key = format!("{prefix}.{suffix}");
        weights.get(&key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing audio VAE weight: {key}"))
        })
    };

    let h = group_norm_32(x, w("norm1.weight")?, w("norm1.bias")?)?;
    let h = h.silu()?;
    let h = causal_conv2d(&h, w("conv1.conv.weight")?, Some(w("conv1.conv.bias")?), 1)?;
    let h = group_norm_32(&h, w("norm2.weight")?, w("norm2.bias")?)?;
    let h = h.silu()?;
    let h = causal_conv2d(&h, w("conv2.conv.weight")?, Some(w("conv2.conv.bias")?), 1)?;

    let x = if in_ch != out_ch {
        causal_conv2d(x, w("nin_shortcut.conv.weight")?, Some(w("nin_shortcut.conv.bias")?), 1)?
    } else {
        x.clone()
    };

    x.add(&h)
}

// ---------------------------------------------------------------------------
// Self-Attention (Conv2d 1x1 QKV)
// ---------------------------------------------------------------------------

/// Self-attention block using Conv2d 1x1 for Q/K/V projections.
///
/// ## PyTorch reference:
/// ```python
/// h = norm(x)
/// q = q_conv(h)  # 1x1 conv
/// k = k_conv(h)
/// v = v_conv(h)
/// # Reshape to [B, C, N] → attention → reshape back
/// attn = softmax(q^T @ k / sqrt(C)) @ v
/// h = proj_out(attn)
/// return x + h
/// ```
fn self_attention(
    x: &Tensor,
    weights: &Weights,
    prefix: &str,
) -> Result<Tensor> {
    let w = |suffix: &str| -> Result<&Tensor> {
        let key = format!("{prefix}.{suffix}");
        weights.get(&key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing: {key}"))
        })
    };

    let dims = x.shape().dims().to_vec();
    let (b, c, h_dim, w_dim) = (dims[0], dims[1], dims[2], dims[3]);
    let n = h_dim * w_dim;

    let normed = group_norm_32(x, w("norm.weight")?, w("norm.bias")?)?;

    // QKV via 1x1 conv (equivalent to linear projection per spatial position)
    let q = causal_conv2d(&normed, w("q.weight")?, Some(w("q.bias")?), 1)?;
    let k = causal_conv2d(&normed, w("k.weight")?, Some(w("k.bias")?), 1)?;
    let v = causal_conv2d(&normed, w("v.weight")?, Some(w("v.bias")?), 1)?;

    // Reshape to [B, C, N] for attention
    let q = q.reshape(&[b, c, n])?;
    let k = k.reshape(&[b, c, n])?;
    let v = v.reshape(&[b, c, n])?;

    // scores = q^T @ k / sqrt(C)
    let q_t = q.permute(&[0, 2, 1])?; // [B, N, C]
    let scores = q_t.matmul(&k)?; // [B, N, N]
    let scale = (c as f32).powf(-0.5);
    let scores = scores.mul_scalar(scale)?;
    let attn = scores.softmax(-1)?;

    // out = v @ attn^T → [B, C, N]
    let v_out = v.matmul(&attn.permute(&[0, 2, 1])?)?; // [B, C, N]
    let v_out = v_out.reshape(&[b, c, h_dim, w_dim])?;

    // Output projection
    let out = causal_conv2d(&v_out, w("proj_out.weight")?, Some(w("proj_out.bias")?), 1)?;

    x.add(&out)
}

// ---------------------------------------------------------------------------
// Mid block
// ---------------------------------------------------------------------------

fn mid_block(x: &Tensor, weights: &Weights, prefix: &str, ch: usize) -> Result<Tensor> {
    let x = resnet_block(x, weights, &format!("{prefix}.block_1"), ch, ch)?;
    let x = self_attention(&x, weights, &format!("{prefix}.attn_1"))?;
    resnet_block(&x, weights, &format!("{prefix}.block_2"), ch, ch)
}

// ---------------------------------------------------------------------------
// Audio VAE Decoder
// ---------------------------------------------------------------------------

pub struct AudioVaeDecoder {
    weights: Weights,
    ch_mult: Vec<usize>,
    base_ch: usize,
    num_res_blocks: usize,
}

impl AudioVaeDecoder {
    /// Load audio VAE decoder weights from checkpoint.
    ///
    /// `weights` should contain keys with prefix `audio_vae.decoder.*`.
    pub fn load(weights: &Weights) -> Result<Self> {
        // Filter to audio_vae keys only
        let audio_weights: Weights = weights.iter()
            .filter(|(k, _)| k.starts_with("audio_vae."))
            .map(|(k, v)| {
                // Strip "audio_vae." prefix
                let short = k.strip_prefix("audio_vae.").unwrap_or(k);
                (short.to_string(), v.clone())
            })
            .collect();

        if audio_weights.is_empty() {
            return Err(flame_core::Error::InvalidInput(
                "No audio_vae keys found in weights".into(),
            ));
        }

        log::info!("[AudioVAE] Loaded {} weights", audio_weights.len());

        Ok(Self {
            weights: audio_weights,
            ch_mult: vec![1, 2, 4, 4], // Standard LDM channel multipliers
            base_ch: 128,
            num_res_blocks: 2,
        })
    }

    /// Decode audio latent → mel spectrogram.
    ///
    /// Input: [B, 8, T, mel_bins] latent
    /// Output: [B, 2, T', mel_bins] mel spectrogram
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let w = &self.weights;

        // 1. Denormalize latents
        let mean = w.get("per_channel_statistics.mean-of-means")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing per_channel_statistics.mean-of-means".into()))?;
        let std = w.get("per_channel_statistics.std-of-means")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing per_channel_statistics.std-of-means".into()))?;
        let x = denormalize(latent, mean, std)?;

        // 2. conv_in: 8 → 512
        let x = causal_conv2d(
            &x,
            w.get("decoder.conv_in.conv.weight").ok_or_else(|| flame_core::Error::InvalidInput("Missing decoder.conv_in".into()))?,
            w.get("decoder.conv_in.conv.bias"),
            1,
        )?;

        // 3. Mid block (512 channels)
        let block_in = self.base_ch * self.ch_mult[self.ch_mult.len() - 1]; // 128 * 4 = 512
        let x = mid_block(&x, w, "decoder.mid", block_in)?;

        // 4. Up blocks (reversed)
        let mut x = x;
        let num_resolutions = self.ch_mult.len();
        for level in (0..num_resolutions).rev() {
            let ch = self.base_ch * self.ch_mult[level];
            let ch_prev = if level == num_resolutions - 1 {
                block_in
            } else {
                self.base_ch * self.ch_mult[level + 1]
            };

            for block_idx in 0..=self.num_res_blocks {
                let in_ch = if block_idx == 0 { ch_prev } else { ch };
                let prefix = format!("decoder.up.{level}.block.{block_idx}");
                x = resnet_block(&x, w, &prefix, in_ch, ch)?;
            }

            if level != 0 {
                let up_w = w.get(&format!("decoder.up.{level}.upsample.conv.weight"))
                    .ok_or_else(|| flame_core::Error::InvalidInput(
                        format!("Missing decoder.up.{level}.upsample.conv.weight")))?;
                let up_b = w.get(&format!("decoder.up.{level}.upsample.conv.bias"));
                // Upsample: nearest-neighbor 2x then conv
                let dims = x.shape().dims().to_vec();
                let (b, c, h, ww) = (dims[0], dims[1], dims[2], dims[3]);
                x = x.upsample_nearest2d(h * 2, ww * 2)?;
                x = causal_conv2d(&x, up_w, up_b, 1)?;
            }
        }

        // 5. Final: norm → silu → conv_out
        let norm_w = w.get("decoder.norm_out.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing decoder.norm_out.weight".into()))?;
        let norm_b = w.get("decoder.norm_out.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing decoder.norm_out.bias".into()))?;
        let x = group_norm_32(&x, norm_w, norm_b)?;
        let x = x.silu()?;
        let x = causal_conv2d(
            &x,
            w.get("decoder.conv_out.conv.weight").ok_or_else(|| flame_core::Error::InvalidInput("Missing decoder.conv_out".into()))?,
            w.get("decoder.conv_out.conv.bias"),
            1,
        )?;

        log::info!("[AudioVAE] Decoded: {:?}", x.shape());
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ch_mult() {
        let ch_mult = vec![1, 2, 4, 4];
        let base = 128;
        assert_eq!(base * ch_mult[3], 512); // top = 512
        assert_eq!(base * ch_mult[0], 128); // bottom = 128
    }
}
