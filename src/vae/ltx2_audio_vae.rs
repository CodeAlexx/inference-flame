//! LTX-2.3 Audio VAE decoder — pure flame-core, production checkpoint parity.
//!
//! Matches `AudioDecoder` from `ltx_core.model.audio_vae.audio_vae` for the
//! LTX-2.3 22B {dev,distilled} checkpoints:
//!
//!   input latent: [B, 8, T_lat, F_lat]  (normalized)
//!   1. patchify denormalize:  [B, 8, T, F] → [B, T, 128] via rearrange "b c t f -> b t (c f)"
//!                             then un_normalize with per_channel_statistics (128-dim!)
//!                             then rearrange back to [B, 8, T, F]
//!   2. conv_in: CausalConv2d(8 → 512, k=3, causality_axis=HEIGHT)
//!   3. mid:
//!        block_1: ResnetBlock(512, 512)
//!        attn_1: Identity   (production checkpoint has NO mid attention)
//!        block_2: ResnetBlock(512, 512)
//!   4. up blocks (forward order = reversed iteration over up[0..3]):
//!        up[2]: 3 ResnetBlocks all 512 → 512, then upsample(512→512, nearest ×2 + conv + drop first frame)
//!        up[1]: 3 ResnetBlocks (512→256, 256→256, 256→256), then upsample(256→256)
//!        up[0]: 3 ResnetBlocks (256→128, 128→128, 128→128), NO upsample
//!   5. norm_out: PixelNorm (no weights)
//!   6. conv_out: CausalConv2d(128 → 2, k=3)   — stereo output
//!
//! Causality is on HEIGHT (dim 2, time). Frequency (dim 3) is padded symmetrically.
//! Causal padding is ZERO padding (unlike video VAE first-frame replicate).
//!
//! ResnetBlock:
//!   h = norm1(x)                       # PixelNorm (no weights)
//!   h = silu(h)
//!   h = conv1(h, causal)
//!   h = norm2(h)
//!   h = silu(h)
//!   h = conv2(h, causal)
//!   if in != out: x = nin_shortcut(x)  # 1×1 conv
//!   return x + h

use flame_core::serialization;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

const LATENT_CH: usize = 8;
const PATCHED_CH: usize = 128; // 8 latent × 16 mel_bins
const PIXEL_NORM_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidOperation(format!("AudioVAE: missing weight: {key}")))?
        .to_dtype(DType::BF16)
}

fn has_key(weights: &Weights, key: &str) -> bool {
    weights.contains_key(key)
}

// ---------------------------------------------------------------------------
// 2D zero padding (asymmetric per axis).
// `left`, `right`, `top`, `bottom` in F.pad order (W left/right, H top/bottom).
// Input is [B, C, H, W].
// ---------------------------------------------------------------------------

fn pad2d_zero(x: &Tensor, left: usize, right: usize, top: usize, bottom: usize) -> Result<Tensor> {
    let d = x.shape().dims();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    let new_h = h + top + bottom;
    let new_w = w + left + right;

    // Fast path: no padding
    if left == 0 && right == 0 && top == 0 && bottom == 0 {
        return Ok(x.clone());
    }

    // Build horizontally padded rows: [B, C, H, W + left + right]
    let mut with_w = if left > 0 || right > 0 {
        let mut parts: Vec<Tensor> = Vec::new();
        if left > 0 {
            parts.push(Tensor::zeros_dtype(
                Shape::from_dims(&[b, c, h, left]),
                x.dtype(),
                x.device().clone(),
            )?);
        }
        parts.push(x.clone());
        if right > 0 {
            parts.push(Tensor::zeros_dtype(
                Shape::from_dims(&[b, c, h, right]),
                x.dtype(),
                x.device().clone(),
            )?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 3)?
    } else {
        x.clone()
    };

    // Now pad rows: [B, C, H + top + bottom, new_w]
    if top > 0 || bottom > 0 {
        let mut parts: Vec<Tensor> = Vec::new();
        if top > 0 {
            parts.push(Tensor::zeros_dtype(
                Shape::from_dims(&[b, c, top, new_w]),
                x.dtype(),
                x.device().clone(),
            )?);
        }
        parts.push(with_w);
        if bottom > 0 {
            parts.push(Tensor::zeros_dtype(
                Shape::from_dims(&[b, c, bottom, new_w]),
                x.dtype(),
                x.device().clone(),
            )?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        with_w = Tensor::cat(&refs, 2)?;
    }

    let _ = new_h;
    Ok(with_w)
}

// ---------------------------------------------------------------------------
// CausalConv2d — zero-pads time axis on the "top" (dim 2), symmetric on W.
// ---------------------------------------------------------------------------

struct CausalConv2d {
    weight: Tensor,
    bias: Tensor,
    kernel: (usize, usize),
}

impl CausalConv2d {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let weight = get_bf16(weights, &format!("{prefix}.weight"))?;
        let bias = get_bf16(weights, &format!("{prefix}.bias"))?;
        let dims = weight.shape().dims().to_vec();
        Ok(Self {
            weight,
            bias,
            kernel: (dims[2], dims[3]),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let kh = self.kernel.0;
        let kw = self.kernel.1;
        let pad_h = kh - 1; // all on top (causal time)
        let pad_w = kw - 1; // symmetric split below
        let pad_w_left = pad_w / 2;
        let pad_w_right = pad_w - pad_w_left;

        let padded = pad2d_zero(x, pad_w_left, pad_w_right, pad_h, 0)?;
        // conv with padding=0 (we already padded manually)
        flame_core::cudnn::cudnn_conv2d_bf16(
            &padded,
            &self.weight,
            Some(&self.bias),
            (1, 1), // stride
            (0, 0), // padding
            (1, 1), // dilation
            1,      // groups
        )
    }
}

// ---------------------------------------------------------------------------
// PixelNorm — x / sqrt(mean(x^2, dim=1, keepdim=True) + eps), no weights.
// ---------------------------------------------------------------------------

fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.mul(&x_f32)?;
    let mean_sq = x_sq.mean_along_dims(&[1], true)?;
    let denom = mean_sq.add_scalar(PIXEL_NORM_EPS)?.rsqrt()?;
    x_f32.mul(&denom)?.to_dtype(x.dtype())
}

// ---------------------------------------------------------------------------
// ResnetBlock — norm → silu → conv → norm → silu → conv + skip (nin if needed)
// ---------------------------------------------------------------------------

struct ResnetBlock {
    conv1: CausalConv2d,
    conv2: CausalConv2d,
    nin_shortcut: Option<CausalConv2d>,
}

impl ResnetBlock {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let conv1 = CausalConv2d::load(weights, &format!("{prefix}.conv1.conv"))?;
        let conv2 = CausalConv2d::load(weights, &format!("{prefix}.conv2.conv"))?;
        let nin_key = format!("{prefix}.nin_shortcut.conv.weight");
        let nin_shortcut = if has_key(weights, &nin_key) {
            Some(CausalConv2d::load(weights, &format!("{prefix}.nin_shortcut.conv"))?)
        } else {
            None
        };
        Ok(Self { conv1, conv2, nin_shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = pixel_norm(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = pixel_norm(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;

        let skip = match &self.nin_shortcut {
            Some(nin) => nin.forward(x)?,
            None => x.clone(),
        };
        skip.add(&h)
    }
}

// ---------------------------------------------------------------------------
// Mid block — block_1 → (Identity) → block_2
// ---------------------------------------------------------------------------

struct MidBlock {
    block_1: ResnetBlock,
    block_2: ResnetBlock,
}

impl MidBlock {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            block_1: ResnetBlock::load(weights, &format!("{prefix}.block_1"))?,
            block_2: ResnetBlock::load(weights, &format!("{prefix}.block_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.block_1.forward(x)?;
        // attn_1 = Identity in production (no op)
        self.block_2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Upsample — nearest 2× → causal conv2d → drop first frame on time axis.
// ---------------------------------------------------------------------------

struct Upsample {
    conv: CausalConv2d,
}

impl Upsample {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv: CausalConv2d::load(weights, &format!("{prefix}.conv.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims();
        let (h, w) = (d[2], d[3]);
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        let conv_out = self.conv.forward(&up)?;
        // Drop the first frame on the causal (HEIGHT) axis.
        let new_h = conv_out.shape().dims()[2];
        conv_out.narrow(2, 1, new_h - 1)
    }
}

// ---------------------------------------------------------------------------
// UpStage — list of ResnetBlocks + optional Upsample
// ---------------------------------------------------------------------------

struct UpStage {
    blocks: Vec<ResnetBlock>,
    upsample: Option<Upsample>,
}

impl UpStage {
    fn load(weights: &Weights, prefix: &str, n_blocks: usize, has_upsample: bool) -> Result<Self> {
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(ResnetBlock::load(weights, &format!("{prefix}.block.{i}"))?);
        }
        let upsample = if has_upsample {
            Some(Upsample::load(weights, &format!("{prefix}.upsample"))?)
        } else {
            None
        };
        Ok(Self { blocks, upsample })
    }

    fn forward(&self, mut x: Tensor) -> Result<Tensor> {
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(up) = &self.upsample {
            x = up.forward(&x)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// PerChannelStatistics — denormalize on the patched representation
// ---------------------------------------------------------------------------

struct PerChannelStatistics {
    mean: Tensor, // [PATCHED_CH]
    std: Tensor,  // [PATCHED_CH]
}

impl PerChannelStatistics {
    fn load(weights: &Weights) -> Result<Self> {
        Ok(Self {
            mean: get_bf16(weights, "per_channel_statistics.mean-of-means")?,
            std: get_bf16(weights, "per_channel_statistics.std-of-means")?,
        })
    }

    /// `x`: `[B, 8, T, 16]`. Returns same shape, denormalized via:
    ///   rearrange "b c t f -> b t (c f)"  (so channel dim becomes 128)
    ///   x_128 * std + mean
    ///   rearrange back.
    fn un_normalize(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims();
        let (b, c, t, f) = (d[0], d[1], d[2], d[3]);
        let cf = c * f;

        // [B, C, T, F] → [B, T, C, F] → [B, T, C*F]
        let flat = x.permute(&[0, 2, 1, 3])?.reshape(&[b, t, cf])?;
        let std = self.std.reshape(&[1, 1, cf])?;
        let mean = self.mean.reshape(&[1, 1, cf])?;
        let denorm = flat.mul(&std)?.add(&mean)?;
        // [B, T, C*F] → [B, T, C, F] → [B, C, T, F]
        denorm.reshape(&[b, t, c, f])?.permute(&[0, 2, 1, 3])
    }
}

// ---------------------------------------------------------------------------
// Public decoder
// ---------------------------------------------------------------------------

pub struct LTX2AudioVaeDecoder {
    stats: PerChannelStatistics,
    conv_in: CausalConv2d,
    mid: MidBlock,
    /// Stages stored in ascending order (up[0], up[1], up[2]); forward
    /// iterates in REVERSE to match Python's `_run_upsampling_path`.
    ups: Vec<UpStage>,
    conv_out: CausalConv2d,
}

impl LTX2AudioVaeDecoder {
    pub fn load(weights: &Weights) -> Result<Self> {
        let stats = PerChannelStatistics::load(weights)?;
        let conv_in = CausalConv2d::load(weights, "decoder.conv_in.conv")?;
        let mid = MidBlock::load(weights, "decoder.mid")?;

        // 3 up stages; up[0] has no upsample, up[1]/up[2] do.
        let ups = vec![
            UpStage::load(weights, "decoder.up.0", 3, false)?,
            UpStage::load(weights, "decoder.up.1", 3, true)?,
            UpStage::load(weights, "decoder.up.2", 3, true)?,
        ];

        let conv_out = CausalConv2d::load(weights, "decoder.conv_out.conv")?;

        Ok(Self { stats, conv_in, mid, ups, conv_out })
    }

    /// Decode normalized audio latent `[B, 8, T_lat, F_lat]` → mel spectrogram
    /// `[B, 2, T_out, F_out]` where `T_out ≈ T_lat * 4`, `F_out = F_lat * 4`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // 1. Denormalize via patchify → un_normalize → unpatchify
        let h = self.stats.un_normalize(latent)?;

        // 2. conv_in: 8 → 512
        let mut h = self.conv_in.forward(&h)?;

        // 3. mid block
        h = self.mid.forward(&h)?;

        // 4. Up stages in REVERSE order: up[2] → up[1] → up[0]
        for stage in self.ups.iter().rev() {
            h = stage.forward(h)?;
        }

        // 5. norm_out → silu → conv_out
        h = pixel_norm(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }

    /// Load from a full LTX-2.3 safetensors checkpoint. Strips the
    /// `audio_vae.` prefix and only pulls decoder + stats keys into GPU.
    pub fn from_file(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        eprintln!("Loading LTX-2.3 Audio VAE from: {path}");
        let raw = serialization::load_file_filtered(
            std::path::Path::new(path),
            device,
            |k| k.starts_with("audio_vae.decoder.") || k.starts_with("audio_vae.per_channel_statistics."),
        )?;

        let mut normalized: Weights = HashMap::new();
        for (key, value) in raw {
            let stripped = key.strip_prefix("audio_vae.").unwrap_or(&key).to_string();
            normalized.insert(stripped, value);
        }

        let dec_count = normalized.keys().filter(|k| k.starts_with("decoder.")).count();
        let stat_count = normalized
            .keys()
            .filter(|k| k.starts_with("per_channel_statistics."))
            .count();
        eprintln!("  {dec_count} audio decoder keys, {stat_count} statistics keys");

        Self::load(&normalized)
    }
}

// Suppress unused const warning if compile excludes patched_ch usage.
const _: usize = LATENT_CH + PATCHED_CH;
