//! LTX-2.3 Video VAE Decoder — pure flame-core, production checkpoint parity.
//!
//! Matches the VideoDecoder module from `ltx_core.model.video_vae.video_vae`
//! as loaded from `ltx-2.3-22b-{dev,distilled}.safetensors`:
//!
//!   conv_in (CausalConv3d 128 → 1024, k=3)
//!   up_blocks.0  MidBlock   channels=1024 n_res=2
//!   up_blocks.1  DepthToSpace stride=(2,2,2) red=2   → 512 ch   F*2 H*2 W*2
//!   up_blocks.2  MidBlock   channels=512  n_res=2
//!   up_blocks.3  DepthToSpace stride=(2,2,2) red=1   → 512 ch   F*2 H*2 W*2
//!   up_blocks.4  MidBlock   channels=512  n_res=4
//!   up_blocks.5  DepthToSpace stride=(2,1,1) red=2   → 256 ch   F*2
//!   up_blocks.6  MidBlock   channels=256  n_res=6
//!   up_blocks.7  DepthToSpace stride=(1,2,2) red=2   → 128 ch   H*2 W*2
//!   up_blocks.8  MidBlock   channels=128  n_res=4
//!   conv_norm_out: PixelNorm (no weights, RMS over dim=1)
//!   conv_act: SiLU
//!   conv_out (CausalConv3d 128 → 48, k=3)   — 48 = 3 * patch_size^2
//!   unpatchify → (B, 3, F_out, H_out, W_out)
//!
//! No timestep conditioning. No scale_shift_table. No noise injection.
//! ResnetBlock3D:  norm1 → silu → conv1 → norm2 → silu → conv2 + skip
//! DepthToSpaceUpsample:  conv → depth_to_space rearrange; if temporal stride=2 drop first frame
//!
//! Input latent: normalized `[B, 128, F_lat, H_lat, W_lat]`.
//! `per_channel_statistics.{mean,std}-of-means` applied first (x * std + mean).
//!
//! Output: `[B, 3, 1 + (F_lat-1)*8, H_lat*32, W_lat*32]` in [-1, 1].

use flame_core::serialization;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

const PATCH_SIZE: usize = 4;
const LATENT_CH: usize = 128;
const OUT_CH: usize = 3;
const PIXEL_NORM_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Block schedule — matches the production LTX-2.3 checkpoint exactly.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum BlockSpec {
    /// Mid block: `n_res` ResnetBlock3D at `channels` channels.
    Mid { channels: usize, n_res: usize },
    /// DepthToSpace upsample: conv → rearrange, no residual path.
    /// `in_ch` is the channel count entering the conv; `out_ch = stride_prod * in_ch / reduction`.
    DepthToSpace {
        in_ch: usize,
        stride: (usize, usize, usize),
        reduction: usize,
    },
}

const DECODER_BLOCKS: &[BlockSpec] = &[
    BlockSpec::Mid { channels: 1024, n_res: 2 },
    BlockSpec::DepthToSpace { in_ch: 1024, stride: (2, 2, 2), reduction: 2 },
    BlockSpec::Mid { channels: 512, n_res: 2 },
    BlockSpec::DepthToSpace { in_ch: 512,  stride: (2, 2, 2), reduction: 1 },
    BlockSpec::Mid { channels: 512, n_res: 4 },
    BlockSpec::DepthToSpace { in_ch: 512,  stride: (2, 1, 1), reduction: 2 },
    BlockSpec::Mid { channels: 256, n_res: 6 },
    BlockSpec::DepthToSpace { in_ch: 256,  stride: (1, 2, 2), reduction: 2 },
    BlockSpec::Mid { channels: 128, n_res: 4 },
];

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidOperation(format!("LTX2 VAE: missing weight: {key}")))?
        .to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d — 3D conv with first-frame replicate padding on time dim.
// ---------------------------------------------------------------------------

struct CausalConv3d {
    weight: Tensor,
    bias: Tensor,
    in_ch: usize,
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

    /// Input: `[B, C, D, H, W]`. Production LTX-2.3 VAE uses **non-causal**
    /// padding (`vae.causal = False` in the checkpoint): replicate the first
    /// and last frames `(kT-1)/2` times on each side of the time axis. The
    /// spatial dimensions get standard symmetric padding via the conv kernel.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims()[2];
        if d == 0 {
            return Ok(x.clone());
        }

        // Non-causal replicate-pad on time: (kT-1)/2 on each side.
        let half_pad = (self.kernel.0 - 1) / 2;
        let x_padded = if half_pad > 0 {
            let first = x.narrow(2, 0, 1)?;
            let last = x.narrow(2, d - 1, 1)?;
            let first_rep = first.repeat_axis_device(2, half_pad)?;
            let last_rep = last.repeat_axis_device(2, half_pad)?;
            Tensor::cat(&[&first_rep, x, &last_rep], 2)?
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

        // Weight is `[C_out, C_in, kD, kH, kW]`. Slice per-kD and treat each
        // slice as a Conv2d weight `[C_out, C_in, kH, kW]`.
        let c_out = self.weight.shape().dims()[0];
        let kh = self.kernel.1;
        let kw = self.kernel.2;
        // Use narrow on the kD axis of the weight tensor.
        let mut kd_slices: Vec<Tensor> = Vec::with_capacity(kd);
        for i in 0..kd {
            let slice_5d = self.weight.narrow(2, i, 1)?;
            // [C_out, C_in, 1, kH, kW] → [C_out, C_in, kH, kW]
            let slice_4d = slice_5d.reshape(&[c_out, c_in, kh, kw])?;
            kd_slices.push(slice_4d);
        }

        // For each output time step, accumulate over the kD slices.
        let zero_bias = Tensor::zeros_dtype(
            Shape::from_dims(&[c_out]),
            DType::BF16,
            x.device().clone(),
        )?;

        let mut out_frames: Vec<Tensor> = Vec::with_capacity(d_out);
        for od in 0..d_out {
            // Collect per-kd contributions
            let mut frame_f32: Option<Tensor> = None;
            for kd_idx in 0..kd {
                let in_d = od + kd_idx;
                // [B, C_in, H, W] slice at time step `in_d`
                let input_4d = x_padded.narrow(2, in_d, 1)?
                    .reshape(&[b, c_in, h, w])?;
                let weight_4d = &kd_slices[kd_idx];
                let conv_out = flame_core::cudnn::cudnn_conv2d_bf16(
                    &input_4d,
                    weight_4d,
                    None,
                    (1, 1),          // stride
                    (h_pad, w_pad),  // padding
                    (1, 1),          // dilation
                    1,               // groups
                )?;
                let conv_out_f32 = conv_out.to_dtype(DType::F32)?;
                frame_f32 = Some(match frame_f32 {
                    None => conv_out_f32,
                    Some(acc) => acc.add(&conv_out_f32)?,
                });
            }
            // Add bias once per output time step.
            let summed = frame_f32.unwrap();
            let summed_bf16 = summed.to_dtype(DType::BF16)?;
            // [B, C_out, H', W'] — unsqueeze time dim back
            let h_out = summed_bf16.shape().dims()[2];
            let w_out = summed_bf16.shape().dims()[3];
            out_frames.push(summed_bf16.reshape(&[b, c_out, 1, h_out, w_out])?);
            let _ = &zero_bias;
        }

        let refs: Vec<&Tensor> = out_frames.iter().collect();
        let mut out = Tensor::cat(&refs, 2)?;

        // Apply bias: broadcast [C_out] → [1, C_out, 1, 1, 1] and add.
        let bias_5d = self.bias.reshape(&[1, c_out, 1, 1, 1])?;
        out = out.add(&bias_5d)?;
        let _ = self.in_ch;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PixelNorm — x / sqrt(mean(x^2, dim=1, keepdim=True) + eps). No weights.
// ---------------------------------------------------------------------------

fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.mul(&x_f32)?;
    let mean_sq = x_sq.mean_along_dims(&[1], true)?;
    let denom = mean_sq.add_scalar(PIXEL_NORM_EPS)?.rsqrt()?;
    x_f32.mul(&denom)?.to_dtype(x.dtype())
}

// ---------------------------------------------------------------------------
// ResnetBlock3D — PixelNorm (no weights) + SiLU + CausalConv3d, residual skip.
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
        let h = pixel_norm(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = pixel_norm(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        x.add(&h)
    }
}

// ---------------------------------------------------------------------------
// UNetMidBlock3D — sequence of ResnetBlock3D at a fixed channel count.
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
// DepthToSpaceUpsample — conv → rearrange (no residual path in production).
// ---------------------------------------------------------------------------

struct DepthToSpaceUpsample {
    conv: CausalConv3d,
    stride: (usize, usize, usize),
}

impl DepthToSpaceUpsample {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        stride: (usize, usize, usize),
        reduction: usize,
    ) -> Result<Self> {
        let prod = stride.0 * stride.1 * stride.2;
        let conv_out_ch = prod * in_ch / reduction;
        let conv = CausalConv3d::load(weights, &format!("{prefix}.conv.conv"), in_ch, conv_out_ch, 3)?;
        Ok(Self { conv, stride })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        let y = depth_to_space(&y, self.stride)?;
        // Drop first frame introduced by temporal causal duplication.
        if self.stride.0 == 2 {
            let d = y.shape().dims()[2];
            if d > 0 {
                return y.narrow(2, 1, d - 1);
            }
        }
        Ok(y)
    }
}

/// `[B, C*p1*p2*p3, D, H, W] → [B, C, D*p1, H*p2, W*p3]`
fn depth_to_space(x: &Tensor, stride: (usize, usize, usize)) -> Result<Tensor> {
    let d = x.shape().dims();
    let (b, c_total, f, h, w) = (d[0], d[1], d[2], d[3], d[4]);
    let (p1, p2, p3) = stride;
    let c = c_total / (p1 * p2 * p3);

    let y = x.reshape(&[b, c, p1, p2, p3, f, h, w])?;
    // [B, C, p1, p2, p3, F, H, W] → [B, C, F, p1, H, p2, W, p3]
    let y = y.permute(&[0, 1, 5, 2, 6, 3, 7, 4])?;
    y.reshape(&[b, c, f * p1, h * p2, w * p3])
}

// ---------------------------------------------------------------------------
// Decoder block enum
// ---------------------------------------------------------------------------

enum DecoderBlock {
    Mid(UNetMidBlock3D),
    DepthToSpace(DepthToSpaceUpsample),
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DecoderBlock::Mid(b) => b.forward(x),
            DecoderBlock::DepthToSpace(b) => b.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// PerChannelStatistics — un-normalize latents: x * std + mean.
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

    fn un_normalize(&self, x: &Tensor) -> Result<Tensor> {
        let std = self.std_of_means.reshape(&[1, LATENT_CH, 1, 1, 1])?;
        let mean = self.mean_of_means.reshape(&[1, LATENT_CH, 1, 1, 1])?;
        x.mul(&std)?.add(&mean)
    }
}

// ---------------------------------------------------------------------------
// Unpatchify — `[B, 3*p*p, F, H, W] → [B, 3, F, H*p, W*p]` with patch_size=4.
// ---------------------------------------------------------------------------

fn unpatchify(x: &Tensor) -> Result<Tensor> {
    let p = PATCH_SIZE;
    let d = x.shape().dims();
    let (b, c_total, f, h, w) = (d[0], d[1], d[2], d[3], d[4]);
    let c = c_total / (p * p);

    // Python: `"b (c p r q) f h w -> b c (f p) (h q) (w r)"` with p=1, q=4, r=4.
    // Channel splits c-major, then r, then q:
    //   [B, c*r*q, F, H, W] → [B, c, r, q, F, H, W]
    //        dim 2 = r, dim 3 = q
    // Target: [B, c, F, H, q, W, r] so reshape gives H*q across dim3,
    // W*r across dim4 — matching the einops layout.
    let y = x.reshape(&[b, c, p, p, f, h, w])?;
    let y = y.permute(&[0, 1, 4, 5, 3, 6, 2])?; // [B, c, F, H, q, W, r]
    y.reshape(&[b, c, f, h * p, w * p])
}

// ---------------------------------------------------------------------------
// Public decoder
// ---------------------------------------------------------------------------

pub struct LTX2VaeDecoder {
    conv_in: CausalConv3d,
    blocks: Vec<DecoderBlock>,
    conv_out: CausalConv3d,
    stats: PerChannelStatistics,
}

impl LTX2VaeDecoder {
    pub fn load(weights: &Weights) -> Result<Self> {
        let conv_in = CausalConv3d::load(weights, "decoder.conv_in.conv", LATENT_CH, 1024, 3)?;

        let mut blocks = Vec::with_capacity(DECODER_BLOCKS.len());
        for (i, spec) in DECODER_BLOCKS.iter().enumerate() {
            match *spec {
                BlockSpec::Mid { channels, n_res } => {
                    blocks.push(DecoderBlock::Mid(UNetMidBlock3D::load(
                        weights,
                        &format!("decoder.up_blocks.{i}"),
                        channels,
                        n_res,
                    )?));
                }
                BlockSpec::DepthToSpace { in_ch, stride, reduction } => {
                    blocks.push(DecoderBlock::DepthToSpace(DepthToSpaceUpsample::load(
                        weights,
                        &format!("decoder.up_blocks.{i}"),
                        in_ch,
                        stride,
                        reduction,
                    )?));
                }
            }
        }

        let out_ch = OUT_CH * PATCH_SIZE * PATCH_SIZE; // 48
        let conv_out = CausalConv3d::load(weights, "decoder.conv_out.conv", 128, out_ch, 3)?;

        let stats = PerChannelStatistics::load(weights)?;

        Ok(Self {
            conv_in,
            blocks,
            conv_out,
            stats,
        })
    }

    /// Decode normalized latents `[B, 128, F_lat, H_lat, W_lat]` → video
    /// `[B, 3, 1 + (F_lat-1)*8, H_lat*32, W_lat*32]` in approximately [-1, 1].
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        self.decode_with_dump(x, None)
    }

    /// Decode with optional activation capture at each stage, for Python
    /// parity bisect. When `dump` is `Some(map)`, inserts:
    ///   after_conv_in, after_up_block_{i}, after_conv_norm_out, after_conv_out
    /// as F32 CPU copies.
    pub fn decode_with_dump(
        &self,
        x: &Tensor,
        mut dump: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let x = self.stats.un_normalize(x)?;
        if let Some(d) = dump.as_deref_mut() {
            d.insert("after_un_normalize".to_string(), x.to_dtype(DType::F32)?);
        }
        let mut h = self.conv_in.forward(&x)?;
        if let Some(d) = dump.as_deref_mut() {
            d.insert("after_conv_in".to_string(), h.to_dtype(DType::F32)?);
        }

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h)?;
            if let Some(d) = dump.as_deref_mut() {
                d.insert(format!("after_up_block_{i}"), h.to_dtype(DType::F32)?);
            }
        }

        h = pixel_norm(&h)?;
        if let Some(d) = dump.as_deref_mut() {
            d.insert("after_conv_norm_out".to_string(), h.to_dtype(DType::F32)?);
        }
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;
        if let Some(d) = dump.as_deref_mut() {
            d.insert("after_conv_out".to_string(), h.to_dtype(DType::F32)?);
        }

        unpatchify(&h)
    }

    /// Legacy 3-arg signature for compatibility with `ltx2_vae_test.rs`.
    #[deprecated(note = "LTX-2.3 production VAE has no timestep conditioning; use `decode` instead")]
    pub fn decode_legacy(&self, x: &Tensor, _timestep: f32, _noise_scale: f32) -> Result<Tensor> {
        self.decode(x)
    }

    /// Load from a safetensors file. Accepts both `vae.decoder.*` (production)
    /// and `decoder.*` (standalone VAE file) key prefixes. Uses filtered
    /// loading so only VAE tensors hit GPU memory — essential when the file
    /// is the full 44 GB LTX-2.3 checkpoint.
    pub fn from_file(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        eprintln!("Loading LTX-2.3 Video VAE from: {path}");
        let raw = serialization::load_file_filtered(
            std::path::Path::new(path),
            device,
            |k| {
                k.starts_with("vae.decoder.")
                    || k.starts_with("vae.per_channel_statistics.")
                    || k.starts_with("decoder.")
                    || k.starts_with("per_channel_statistics.")
                    || k.starts_with("video_vae.decoder.")
                    || k.starts_with("video_vae.per_channel_statistics.")
                    || k.starts_with("first_stage_model.decoder.")
                    || k.starts_with("first_stage_model.per_channel_statistics.")
            },
        )?;

        let mut normalized: Weights = HashMap::new();
        for (key, value) in raw {
            let mut k = key.clone();
            // Strip outer prefix if present.
            for prefix in &["vae.", "video_vae.", "first_stage_model."] {
                if k.starts_with(prefix) {
                    k = k[prefix.len()..].to_string();
                    break;
                }
            }
            if k.starts_with("decoder.") || k.starts_with("per_channel_statistics.") {
                normalized.insert(k, value);
            }
        }

        let decoder_count = normalized.keys().filter(|k| k.starts_with("decoder.")).count();
        let stats_count = normalized
            .keys()
            .filter(|k| k.starts_with("per_channel_statistics."))
            .count();
        eprintln!("  {decoder_count} decoder keys, {stats_count} statistics keys");

        Self::load(&normalized)
    }
}

pub use self::LTX2VaeDecoder as LTX2VideoVAE;
