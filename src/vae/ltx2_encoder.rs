//! LTX-2.3 Video VAE Encoder — pure flame-core, production checkpoint parity.
//!
//! Mirror of the decoder in `ltx2_vae.rs`. Matches the VideoEncoder module from
//! `ltx_core.model.video_vae.video_vae` as loaded from
//! `ltx-2.3-22b-{dev,distilled}.safetensors`:
//!
//!   patchify:    [B, 3, T, H, W] → [B, 48, T, H/4, W/4]  (pixel-shuffle inverse, p=4)
//!   conv_in:     CausalConv3d(48, 128, k=3)
//!   down_blocks.0  MidBlock   channels=128  n_res=4
//!   down_blocks.1  SpaceToDepth stride=(1,2,2) group_size=2   conv 128→64   → 256 ch   H/2 W/2
//!   down_blocks.2  MidBlock   channels=256  n_res=6
//!   down_blocks.3  SpaceToDepth stride=(2,1,1) group_size=1   conv 256→256  → 512 ch   T/2
//!   down_blocks.4  MidBlock   channels=512  n_res=4
//!   down_blocks.5  SpaceToDepth stride=(2,2,2) group_size=4   conv 512→128  → 1024 ch  T/2 H/2 W/2
//!   down_blocks.6  MidBlock   channels=1024 n_res=2
//!   down_blocks.7  SpaceToDepth stride=(2,2,2) group_size=8   conv 1024→128 → 1024 ch  T/2 H/2 W/2
//!   down_blocks.8  MidBlock   channels=1024 n_res=2
//!   norm_out:    PerChannelRMSNorm (no weights, RMS over dim=1, eps=1e-8)
//!   conv_act:    SiLU
//!   conv_out:    CausalConv3d(1024, 129, k=3)
//!   expand:      last channel → repeat 127 times, concat → 256 = 2×128 channels
//!   take mean:   first 128 channels (deterministic)
//!   normalize:   (z - mean_of_means) / std_of_means  (per_channel_statistics)
//!
//! Input:  `[B, 3, T, H, W]` video in `[-1, 1]`.
//! Output: `[B, 128, T/8, H/32, W/32]` normalized latents.

use flame_core::serialization;
use flame_core::{CudaDevice, DType, Error, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

const PATCH_SIZE: usize = 4;
const LATENT_CH: usize = 128;
const IN_CH: usize = 3;
const PIXEL_NORM_EPS: f32 = 1e-6;
/// PerChannelRMSNorm in the encoder output uses a tighter eps than the
/// block-level PixelNorm.
const ENCODER_NORM_OUT_EPS: f32 = 1e-8;

// ---------------------------------------------------------------------------
// Block schedule — mirrors the decoder in reverse, production LTX-2.3.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum BlockSpec {
    /// Mid block: `n_res` ResnetBlock3D at `channels` channels.
    Mid { channels: usize, n_res: usize },
    /// SpaceToDepth downsample: causal conv → space_to_depth rearrange, with
    /// grouped-average residual connection.
    SpaceToDepth {
        in_ch: usize,
        out_ch: usize,
        stride: (usize, usize, usize),
    },
}

/// Encoder block schedule — exact mirror of `DECODER_BLOCKS` in ltx2_vae.rs.
const ENCODER_BLOCKS: &[BlockSpec] = &[
    // Mirror of decoder block 8: Mid 128ch ×4
    BlockSpec::Mid { channels: 128, n_res: 4 },
    // Mirror of decoder block 7: DepthToSpace stride=(1,2,2) red=2
    // Encoder: SpaceToDepth stride=(1,2,2), in=128, out=256
    BlockSpec::SpaceToDepth { in_ch: 128, out_ch: 256, stride: (1, 2, 2) },
    // Mirror of decoder block 6: Mid 256ch ×6
    BlockSpec::Mid { channels: 256, n_res: 6 },
    // Mirror of decoder block 5: DepthToSpace stride=(2,1,1) red=2
    // Encoder: SpaceToDepth stride=(2,1,1), in=256, out=512
    BlockSpec::SpaceToDepth { in_ch: 256, out_ch: 512, stride: (2, 1, 1) },
    // Mirror of decoder block 4: Mid 512ch ×4
    BlockSpec::Mid { channels: 512, n_res: 4 },
    // Mirror of decoder block 3: DepthToSpace stride=(2,2,2) red=1
    // Encoder: SpaceToDepth stride=(2,2,2), in=512, out=1024
    BlockSpec::SpaceToDepth { in_ch: 512, out_ch: 1024, stride: (2, 2, 2) },
    // Mirror of decoder block 2: Mid 1024ch (was 512) ×2
    BlockSpec::Mid { channels: 1024, n_res: 2 },
    // Mirror of decoder block 1: DepthToSpace stride=(2,2,2) red=2
    // Encoder: SpaceToDepth stride=(2,2,2), in=1024, out=1024
    BlockSpec::SpaceToDepth { in_ch: 1024, out_ch: 1024, stride: (2, 2, 2) },
    // Mirror of decoder block 0: Mid 1024ch ×2
    BlockSpec::Mid { channels: 1024, n_res: 2 },
];

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidOperation(format!("LTX2 VAE Encoder: missing weight: {key}")))?
        .to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d — 3D conv with first-frame replicate padding on time dim.
// Duplicated from ltx2_vae.rs (decoder) since it's not pub.
// ---------------------------------------------------------------------------

struct CausalConv3d {
    weight: Tensor,
    bias: Tensor,
    #[allow(dead_code)]
    in_ch: usize,
    #[allow(dead_code)]
    out_ch: usize,
    kernel: (usize, usize, usize),
}

impl CausalConv3d {
    fn load(weights: &Weights, prefix: &str, in_ch: usize, out_ch: usize, k: usize) -> Result<Self> {
        let weight = get_bf16(weights, &format!("{prefix}.weight"))?;
        let bias = get_bf16(weights, &format!("{prefix}.bias"))?;
        Ok(Self {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel: (k, k, k),
        })
    }

    /// Causal forward: replicate-pad the first frame `(kT-1)` times on the left
    /// of the time axis (matching the encoder's `causal=True` mode).
    fn forward_causal(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims()[2];
        if d == 0 {
            return Ok(x.clone());
        }

        // Causal: pad only on the left side of time.
        let time_pad = self.kernel.0 - 1;
        let x_padded = if time_pad > 0 {
            let first = x.narrow(2, 0, 1)?;
            let first_rep = first.repeat_axis_device(2, time_pad)?;
            Tensor::cat(&[&first_rep, x], 2)?
        } else {
            x.clone()
        };

        let h_pad = self.kernel.1 / 2;
        let w_pad = self.kernel.2 / 2;
        let (b, c_in, d_pad, h, w) = {
            let dd = x_padded.shape().dims();
            (dd[0], dd[1], dd[2], dd[3], dd[4])
        };
        let kd = self.kernel.0;
        let d_out = d_pad - kd + 1;

        let c_out = self.weight.shape().dims()[0];
        let kh = self.kernel.1;
        let kw = self.kernel.2;
        let mut kd_slices: Vec<Tensor> = Vec::with_capacity(kd);
        for i in 0..kd {
            let slice_5d = self.weight.narrow(2, i, 1)?;
            let slice_4d = slice_5d.reshape(&[c_out, c_in, kh, kw])?;
            kd_slices.push(slice_4d);
        }

        let mut out_frames: Vec<Tensor> = Vec::with_capacity(d_out);
        for od in 0..d_out {
            let mut frame_f32: Option<Tensor> = None;
            for kd_idx in 0..kd {
                let in_d = od + kd_idx;
                let input_4d = x_padded
                    .narrow(2, in_d, 1)?
                    .reshape(&[b, c_in, h, w])?;
                let weight_4d = &kd_slices[kd_idx];
                let conv_out = flame_core::cudnn::cudnn_conv2d_bf16(
                    &input_4d,
                    weight_4d,
                    None,
                    (1, 1),
                    (h_pad, w_pad),
                    (1, 1),
                    1,
                )?;
                let conv_out_f32 = conv_out.to_dtype(DType::F32)?;
                frame_f32 = Some(match frame_f32 {
                    None => conv_out_f32,
                    Some(acc) => acc.add(&conv_out_f32)?,
                });
            }
            let summed = frame_f32.unwrap();
            let summed_bf16 = summed.to_dtype(DType::BF16)?;
            let h_out = summed_bf16.shape().dims()[2];
            let w_out = summed_bf16.shape().dims()[3];
            out_frames.push(summed_bf16.reshape(&[b, c_out, 1, h_out, w_out])?);
        }

        let refs: Vec<&Tensor> = out_frames.iter().collect();
        let mut out = Tensor::cat(&refs, 2)?;

        let bias_5d = self.bias.reshape(&[1, c_out, 1, 1, 1])?;
        out = out.add(&bias_5d)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PixelNorm — x / sqrt(mean(x^2, dim=1, keepdim=True) + eps). No weights.
// ---------------------------------------------------------------------------

fn pixel_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.mul(&x_f32)?;
    let mean_sq = x_sq.mean_along_dims(&[1], true)?;
    let denom = mean_sq.add_scalar(eps)?.rsqrt()?;
    x_f32.mul(&denom)?.to_dtype(x.dtype())
}

// ---------------------------------------------------------------------------
// ResnetBlock3D — PixelNorm → SiLU → CausalConv3d → PixelNorm → SiLU →
// CausalConv3d + residual skip. Encoder version uses causal=True.
// ---------------------------------------------------------------------------

struct ResnetBlock3D {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl ResnetBlock3D {
    fn load(weights: &Weights, prefix: &str, channels: usize) -> Result<Self> {
        let conv1 = CausalConv3d::load(weights, &format!("{prefix}.conv1.conv"), channels, channels, 3)?;
        let conv2 = CausalConv3d::load(weights, &format!("{prefix}.conv2.conv"), channels, channels, 3)?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = pixel_norm(x, PIXEL_NORM_EPS)?;
        let h = h.silu()?;
        let h = self.conv1.forward_causal(&h)?;
        let h = pixel_norm(&h, PIXEL_NORM_EPS)?;
        let h = h.silu()?;
        let h = self.conv2.forward_causal(&h)?;
        x.add(&h)
    }
}

// ---------------------------------------------------------------------------
// UNetMidBlock3D — sequence of ResnetBlock3D at fixed channel count.
// ---------------------------------------------------------------------------

struct UNetMidBlock3D {
    res_blocks: Vec<ResnetBlock3D>,
}

impl UNetMidBlock3D {
    fn load(weights: &Weights, prefix: &str, channels: usize, n_res: usize) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(n_res);
        for i in 0..n_res {
            res_blocks.push(ResnetBlock3D::load(
                weights,
                &format!("{prefix}.res_blocks.{i}"),
                channels,
            )?);
        }
        Ok(Self { res_blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for rb in &self.res_blocks {
            out = rb.forward(&out)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// SpaceToDepthDownsample — inverse of DepthToSpaceUpsample.
//
// Forward:
//   1. Prepend stride[0]-1 copies of the first frame (causal temporal padding)
//   2. Compute residual: space_to_depth(input) → group average
//   3. Compute main: causal_conv3d(input) → space_to_depth
//   4. output = main + residual
//
// This matches the Python LTXVideoDownsampler3d.
// ---------------------------------------------------------------------------

struct SpaceToDepthDownsample {
    conv: CausalConv3d,
    stride: (usize, usize, usize),
    group_size: usize,
}

impl SpaceToDepthDownsample {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        stride: (usize, usize, usize),
    ) -> Result<Self> {
        let prod = stride.0 * stride.1 * stride.2;
        let group_size = (in_ch * prod) / out_ch;
        let conv_out_ch = out_ch / prod;
        let conv = CausalConv3d::load(weights, &format!("{prefix}.conv.conv"), in_ch, conv_out_ch, 3)?;
        Ok(Self {
            conv,
            stride,
            group_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims();
        let t = d[2];
        let (st, _sh, _sw) = self.stride;

        // 1. Prepend first frames for causal temporal padding.
        let x_padded = if st > 1 {
            let first = x.narrow(2, 0, 1)?;
            let first_rep = first.repeat_axis_device(2, st - 1)?;
            Tensor::cat(&[&first_rep, x], 2)?
        } else {
            x.clone()
        };

        // 2. Compute residual: space_to_depth(x_padded) → group average.
        let residual = space_to_depth(&x_padded, self.stride)?;
        let residual = if self.group_size > 1 {
            // residual has shape [B, C*prod, T', H', W']
            // unflatten channel dim into [B, out_ch, group_size, T', H', W']
            // then mean over group_size dim
            let rd = residual.shape().dims();
            let (rb, rc, rt, rh, rw) = (rd[0], rd[1], rd[2], rd[3], rd[4]);
            let n_groups = rc / self.group_size;
            let reshaped = residual.reshape(&[rb, n_groups, self.group_size, rt, rh, rw])?;
            reshaped.mean_along_dims(&[2], false)?
        } else {
            residual
        };

        // 3. Compute main: conv → space_to_depth.
        let conv_out = self.conv.forward_causal(&x_padded)?;
        let main = space_to_depth(&conv_out, self.stride)?;

        // 4. Output = main + residual.
        main.add(&residual)
    }
}

/// `[B, C, D, H, W] → [B, C*p1*p2*p3, D/p1, H/p2, W/p3]`
///
/// Inverse of `depth_to_space` in ltx2_vae.rs.
fn space_to_depth(x: &Tensor, stride: (usize, usize, usize)) -> Result<Tensor> {
    let d = x.shape().dims();
    let (b, c, f, h, w) = (d[0], d[1], d[2], d[3], d[4]);
    let (p1, p2, p3) = stride;
    let f2 = f / p1;
    let h2 = h / p2;
    let w2 = w / p3;

    // [B, C, F, H, W] → [B, C, F/p1, p1, H/p2, p2, W/p3, p3]
    let y = x.reshape(&[b, c, f2, p1, h2, p2, w2, p3])?;
    // → [B, C, p1, p2, p3, F/p1, H/p2, W/p3]
    let y = y.permute(&[0, 1, 3, 5, 7, 2, 4, 6])?;
    y.reshape(&[b, c * p1 * p2 * p3, f2, h2, w2])
}

// ---------------------------------------------------------------------------
// Patchify — [B, 3, T, H, W] → [B, 48, T, H/4, W/4]
//
// Pixel-shuffle inverse with patch_size=4 on spatial dims.
// Matches the Python encoder's patchify:
//   reshape(B, C, T/pt, pt, H/p, p, W/p, p)
//   permute(0, 1, 3, 7, 5, 2, 4, 6)
//   flatten(1, 4)
// With pt=1 this simplifies to:
//   reshape(B, 3, T, 1, H/4, 4, W/4, 4)
//   permute(0, 1, 3, 7, 5, 2, 4, 6) → (B, 3, 1, 4, 4, T, H/4, W/4)
//   flatten(1,4) → (B, 48, T, H/4, W/4)
// ---------------------------------------------------------------------------

fn patchify(x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims();
    let (b, c, t, h, w) = (d[0], d[1], d[2], d[3], d[4]);
    let p = PATCH_SIZE;
    let hp = h / p;
    let wp = w / p;

    // [B, C, T, H, W] → [B, C, T, 1, H/p, p, W/p, p]
    let y = x.reshape(&[b, c, t, 1, hp, p, wp, p])?;
    // permute: (0, 1, 3, 7, 5, 2, 4, 6)
    // → [B, C, 1, p, p, T, H/p, W/p]
    let y = y.permute(&[0, 1, 3, 7, 5, 2, 4, 6])?;
    // flatten dims 1..4: [B, C*1*p*p, T, H/p, W/p] = [B, 48, T, H/p, W/p]
    y.reshape(&[b, c * p * p, t, hp, wp])
}

// ---------------------------------------------------------------------------
// Encoder block enum
// ---------------------------------------------------------------------------

enum EncoderBlock {
    Mid(UNetMidBlock3D),
    SpaceToDepth(SpaceToDepthDownsample),
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            EncoderBlock::Mid(b) => b.forward(x),
            EncoderBlock::SpaceToDepth(b) => b.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// PerChannelStatistics — normalize latents for encoding: (x - mean) / std.
// ---------------------------------------------------------------------------

struct PerChannelStatistics {
    std_of_means: Tensor,
    mean_of_means: Tensor,
}

impl PerChannelStatistics {
    fn load(weights: &Weights) -> Result<Self> {
        Ok(Self {
            std_of_means: get_bf16(weights, "per_channel_statistics.std-of-means")?,
            mean_of_means: get_bf16(weights, "per_channel_statistics.mean-of-means")?,
        })
    }

    /// Normalize latents for the training pipeline: (x - mean) / std.
    /// Inverse of the decoder's `un_normalize`.
    fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let std = self.std_of_means.reshape(&[1, LATENT_CH, 1, 1, 1])?;
        let mean = self.mean_of_means.reshape(&[1, LATENT_CH, 1, 1, 1])?;
        // (x - mean) / std = x/std - mean/std
        // Compute in F32 for numerical stability.
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean_f32 = mean.to_dtype(DType::F32)?;
        let std_f32 = std.to_dtype(DType::F32)?;
        let centered = x_f32.sub(&mean_f32)?;
        let normalized = centered.div(&std_f32)?;
        normalized.to_dtype(x.dtype())
    }
}

// ---------------------------------------------------------------------------
// Public encoder
// ---------------------------------------------------------------------------

pub struct LTX2VaeEncoder {
    conv_in: CausalConv3d,
    blocks: Vec<EncoderBlock>,
    conv_out: CausalConv3d,
    stats: PerChannelStatistics,
}

impl LTX2VaeEncoder {
    pub fn load(weights: &Weights) -> Result<Self> {
        let patched_in = IN_CH * PATCH_SIZE * PATCH_SIZE; // 48
        let conv_in = CausalConv3d::load(weights, "encoder.conv_in.conv", patched_in, 128, 3)?;

        let mut blocks = Vec::with_capacity(ENCODER_BLOCKS.len());
        for (i, spec) in ENCODER_BLOCKS.iter().enumerate() {
            match *spec {
                BlockSpec::Mid { channels, n_res } => {
                    blocks.push(EncoderBlock::Mid(UNetMidBlock3D::load(
                        weights,
                        &format!("encoder.down_blocks.{i}"),
                        channels,
                        n_res,
                    )?));
                }
                BlockSpec::SpaceToDepth { in_ch, out_ch, stride } => {
                    blocks.push(EncoderBlock::SpaceToDepth(SpaceToDepthDownsample::load(
                        weights,
                        &format!("encoder.down_blocks.{i}"),
                        in_ch,
                        out_ch,
                        stride,
                    )?));
                }
            }
        }

        // conv_out: 1024 → 129 (128 + 1 expansion channel)
        let conv_out = CausalConv3d::load(weights, "encoder.conv_out.conv", 1024, LATENT_CH + 1, 3)?;

        let stats = PerChannelStatistics::load(weights)?;

        Ok(Self {
            conv_in,
            blocks,
            conv_out,
            stats,
        })
    }

    /// Print debug stats for a tensor (shape, mean, std, min, max).
    fn debug_stats(label: &str, t: &Tensor) {
        let dims = t.shape().dims().to_vec();
        let v = t.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let n = v.len() as f64;
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
        let var = v.iter().map(|x| { let d = *x as f64 - mean; d * d }).sum::<f64>() / n;
        let std = var.sqrt();
        let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
        for x in &v {
            let xf = *x as f64;
            if xf < mn { mn = xf; }
            if xf > mx { mx = xf; }
        }
        eprintln!(
            "  {label} {:?}  mean={:+.6} std={:.6} min={:.6} max={:.6}",
            dims, mean, std, mn, mx
        );
    }

    /// Encode video `[B, 3, T, H, W]` in `[-1, 1]` to normalized latents
    /// `[B, 128, T', H', W']` where T'=T/8, H'=H/32, W'=W/32.
    ///
    /// Uses the deterministic mean (no sampling) and applies per-channel
    /// normalization — the right choice for caching training latents.
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        // 1. Patchify: [B, 3, T, H, W] → [B, 48, T, H/4, W/4]
        let mut h = patchify(video)?;
        Self::debug_stats("after patchify   ", &h);

        // 2. conv_in
        h = self.conv_in.forward_causal(&h)?;
        Self::debug_stats("after conv_in    ", &h);

        // 3. Down blocks
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h)?;
            Self::debug_stats(&format!("after block {i}    "), &h);
        }

        // 4. norm_out + SiLU
        h = pixel_norm(&h, ENCODER_NORM_OUT_EPS)?;
        h = h.silu()?;

        // 5. conv_out: → [B, 129, T', H', W']
        h = self.conv_out.forward_causal(&h)?;
        Self::debug_stats("after conv_out   ", &h);

        // 6. Expand last channel: concat last ch repeated 127 times → 256 ch
        let n_ch = h.shape().dims()[1]; // 129
        let last_ch = h.narrow(1, n_ch - 1, 1)?; // [B, 1, T', H', W']
        let repeated = last_ch.repeat_axis_device(1, n_ch - 2)?; // [B, 127, ...]
        h = Tensor::cat(&[&h, &repeated], 1)?; // [B, 256, ...]

        // 7. Take deterministic mean (first 128 channels)
        let mu = h.narrow(1, 0, LATENT_CH)?;

        // 8. Per-channel normalization
        self.stats.normalize(&mu)
    }

    /// Encode WITHOUT per-channel normalization — returns raw latents
    /// `[B, 128, T', H', W']`.
    ///
    /// Use this for edit/reference conditioning where the model expects
    /// un-normalized latents.
    pub fn encode_raw(&self, video: &Tensor) -> Result<Tensor> {
        let mut h = patchify(video)?;
        h = self.conv_in.forward_causal(&h)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        h = pixel_norm(&h, ENCODER_NORM_OUT_EPS)?;
        h = h.silu()?;
        h = self.conv_out.forward_causal(&h)?;

        let n_ch = h.shape().dims()[1];
        let last_ch = h.narrow(1, n_ch - 1, 1)?;
        let repeated = last_ch.repeat_axis_device(1, n_ch - 2)?;
        h = Tensor::cat(&[&h, &repeated], 1)?;

        h.narrow(1, 0, LATENT_CH)
    }

    /// Load from a safetensors file. Accepts both `vae.encoder.*` (production)
    /// and `encoder.*` (standalone VAE file) key prefixes. Uses filtered
    /// loading so only encoder+stats tensors hit GPU memory.
    pub fn from_file(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        eprintln!("Loading LTX-2.3 Video VAE Encoder from: {path}");
        let raw = serialization::load_file_filtered(
            std::path::Path::new(path),
            device,
            |k| {
                k.starts_with("vae.encoder.")
                    || k.starts_with("vae.per_channel_statistics.")
                    || k.starts_with("encoder.")
                    || k.starts_with("per_channel_statistics.")
                    || k.starts_with("video_vae.encoder.")
                    || k.starts_with("video_vae.per_channel_statistics.")
                    || k.starts_with("first_stage_model.encoder.")
                    || k.starts_with("first_stage_model.per_channel_statistics.")
            },
        )?;

        let mut normalized: Weights = HashMap::new();
        for (key, value) in raw {
            let mut k = key.clone();
            for prefix in &["vae.", "video_vae.", "first_stage_model."] {
                if k.starts_with(prefix) {
                    k = k[prefix.len()..].to_string();
                    break;
                }
            }
            if k.starts_with("encoder.") || k.starts_with("per_channel_statistics.") {
                normalized.insert(k, value);
            }
        }

        let encoder_count = normalized.keys().filter(|k| k.starts_with("encoder.")).count();
        let stats_count = normalized
            .keys()
            .filter(|k| k.starts_with("per_channel_statistics."))
            .count();
        eprintln!("  {encoder_count} encoder keys, {stats_count} statistics keys");

        Self::load(&normalized)
    }
}
