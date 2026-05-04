//! HunyuanVideo (1.0/1.5) 3D causal VAE decoder — pure flame_core port of
//! diffusers' `AutoencoderKLHunyuanVideo`.
//!
//! Decode-only path: latents `[B, 16, T, H, W]` → pixels `[B, 3, T_out, H_out, W_out]`.
//! Spatial compression 8×, temporal compression 4×.
//!
//! Architecture (block_out_channels=[128, 256, 512, 512], layers_per_block=2):
//!   conv_in: HunyuanCausalConv3d(16, 512, 3, stride=1)
//!   mid_block: ResBlock(512) + Attention(512) + ResBlock(512)
//!   up_blocks (reversed iteration over [128,256,512,512] gives [512,512,256,128]):
//!     up_blocks.0: 3× ResBlock(512→512) + Upsample(spatial-only 2×)        i=0
//!     up_blocks.1: 3× ResBlock(512→512) + Upsample(temporal+spatial 2×)    i=1
//!     up_blocks.2: 3× ResBlock(512→256) + Upsample(temporal+spatial 2×)    i=2
//!     up_blocks.3: 3× ResBlock(256→128) — no upsample (final block)         i=3
//!   conv_norm_out: GroupNorm(32, 128)
//!   conv_act: SiLU
//!   conv_out: HunyuanCausalConv3d(128, 3, 3)
//!
//! Weight key format (diffusers AutoencoderKLHunyuanVideo, verified against
//! /home/alex/.serenity/models/vaes/hunyuan_video_vae_bf16.safetensors):
//!   decoder.conv_in.conv.{weight,bias}                 — note `.conv.` from HunyuanCausalConv3d
//!   decoder.mid_block.resnets.{0,1}.{norm{1,2},conv{1,2}.conv}.{weight,bias}
//!   decoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.{weight,bias}
//!   decoder.up_blocks.{n}.resnets.{m}.{norm{1,2},conv{1,2}.conv,conv_shortcut.conv}.{weight,bias}
//!   decoder.up_blocks.{n}.upsamplers.0.conv.conv.{weight,bias}
//!   decoder.conv_norm_out.{weight,bias}
//!   decoder.conv_out.conv.{weight,bias}
//!
//! NOTE on parity:
//! - Block-by-block parity vs the reference Python decoder is NOT yet verified.
//!   Each subsystem is structurally faithful to the diffusers source but
//!   end-to-end cos similarity must be measured before trusting output.

use flame_core::conv3d_simple::Conv3d;
use flame_core::cuda_ops::GpuOps;
use flame_core::group_norm::GroupNorm;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HunyuanVaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub scaling_factor: f32,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    /// Spatial compression ratio (8 for HunyuanVideo).
    pub spatial_compression_ratio: usize,
    /// Temporal compression ratio (4 for HunyuanVideo).
    pub temporal_compression_ratio: usize,
}

impl Default for HunyuanVaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 16,
            out_channels: 3,
            scaling_factor: 0.476986,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            spatial_compression_ratio: 8,
            temporal_compression_ratio: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight helpers
// ---------------------------------------------------------------------------

type Weights = HashMap<String, Tensor>;

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("HunyuanVAE: missing weight: {key}")))
}

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

fn get_f32(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::F32)
}

fn has_key(weights: &Weights, key: &str) -> bool {
    weights.contains_key(key)
}

// ---------------------------------------------------------------------------
// HunyuanCausalConv3d
//
// diffusers `HunyuanVideoCausalConv3d`:
//   pad with F.pad(x, (pW,pW,pH,pH,kT-1,0), mode='replicate')
//   then Conv3d(...padding=0...)
//
// In flame-core terms: Conv3d kernel handles symmetric spatial padding via its
// `padding` argument; temporal pad-left is replicate of first frame, kT-1 copies.
// ---------------------------------------------------------------------------

struct HunyuanCausalConv3d {
    conv: Conv3d,
    time_pad: usize, // = kT - 1 left-only frames
}

impl HunyuanCausalConv3d {
    /// Load with diffusers `.conv.{weight,bias}` suffix.
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // Diffusers padding (1,1,1,1,kT-1,0) on (W,W,H,H,T_left,T_right) means:
        // - spatial pad = kH//2 each side, kW//2 each side (symmetric)
        // - temporal pad = kT-1 left only
        let p_h = kernel.1 / 2;
        let p_w = kernel.2 / 2;
        let time_pad = kernel.0.saturating_sub(1);
        let mut conv = Conv3d::new(
            in_ch,
            out_ch,
            kernel,
            Some(stride),
            Some((0, p_h, p_w)), // no temporal pad here, we prepend manually
            None,
            None,
            true,
            device.clone(),
        )?;
        // Conv3d kernel is F32-internal; stash weights as F32.
        conv.weight = get_f32(weights, &format!("{prefix}.conv.weight"))?;
        conv.bias_tensor = Some(get_f32(weights, &format!("{prefix}.conv.bias"))?);
        Ok(Self { conv, time_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_padded = if self.time_pad > 0 {
            // Replicate pad: prepend `time_pad` copies of frame 0.
            let first = x.narrow(2, 0, 1)?;
            let pad = first.repeat_axis_device(2, self.time_pad)?;
            Tensor::cat(&[&pad, x], 2)?
        } else {
            x.clone()
        };
        let is_bf16 = x_padded.dtype() == DType::BF16;
        let input = if is_bf16 {
            x_padded.to_dtype(DType::F32)?
        } else {
            x_padded
        };
        let out = self.conv.forward(&input)?;
        if is_bf16 {
            out.to_dtype(DType::BF16)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// GroupNorm wrapper for 5D tensors [B, C, T, H, W].
//
// flame_core's GroupNorm operates on 4D NCHW BF16. We fold T into spatial:
// [B, C, T, H, W] → [B, C, T*H, W] → GN → [B, C, T, H, W].
// This is mathematically equivalent because GroupNorm reduces over all spatial
// elements per (batch, group) — the layout of those spatial elements doesn't
// affect the result.
// ---------------------------------------------------------------------------

fn group_norm_5d(
    x: &Tensor,
    num_groups: usize,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 5 {
        return Err(Error::InvalidInput(format!(
            "group_norm_5d: expected 5D, got {:?}",
            dims
        )));
    }
    let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    // BF16 path required by GroupNorm forward_nchw.
    let x4 = x.reshape(&[b, c, t * h, w])?;
    let x4_bf16 = if x4.dtype() == DType::BF16 {
        x4
    } else {
        x4.to_dtype(DType::BF16)?
    };
    let gn = GroupNorm {
        num_groups,
        num_channels: c,
        eps,
        affine: true,
        weight: Some(weight.clone()),
        bias: Some(bias.clone()),
    };
    let y4 = gn.forward_nchw(&x4_bf16)?;
    y4.reshape(&[b, c, t, h, w])
}

// ---------------------------------------------------------------------------
// HunyuanResnetBlock3D (diffusers HunyuanVideoResnetBlockCausal3D)
//
//   x_in -> [GN, SiLU, CausalConv3d(3)] -> [GN, SiLU, CausalConv3d(3)]
//        \                                                          /
//         + (conv_shortcut if in_ch != out_ch)                     /
//
// Weight keys (under prefix):
//   norm1.weight, norm1.bias            — GroupNorm(num_groups=32)
//   conv1.conv.weight, conv1.conv.bias  — HunyuanCausalConv3d(in→out, k=3)
//   norm2.weight, norm2.bias            — GroupNorm(num_groups=32)
//   conv2.conv.weight, conv2.conv.bias  — HunyuanCausalConv3d(out→out, k=3)
//   conv_shortcut.conv.{weight,bias}    — HunyuanCausalConv3d(in→out, k=1) [optional]
// ---------------------------------------------------------------------------

struct HunyuanResnetBlock3D {
    norm1_w: Tensor,
    norm1_b: Tensor,
    conv1: HunyuanCausalConv3d,
    norm2_w: Tensor,
    norm2_b: Tensor,
    conv2: HunyuanCausalConv3d,
    shortcut: Option<HunyuanCausalConv3d>,
    in_ch: usize,
    out_ch: usize,
    num_groups: usize,
    eps: f32,
}

impl HunyuanResnetBlock3D {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        num_groups: usize,
        eps: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let norm1_w = get_bf16(weights, &format!("{prefix}.norm1.weight"))?;
        let norm1_b = get_bf16(weights, &format!("{prefix}.norm1.bias"))?;
        let conv1 = HunyuanCausalConv3d::load(
            weights,
            &format!("{prefix}.conv1"),
            in_ch,
            out_ch,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;
        let norm2_w = get_bf16(weights, &format!("{prefix}.norm2.weight"))?;
        let norm2_b = get_bf16(weights, &format!("{prefix}.norm2.bias"))?;
        let conv2 = HunyuanCausalConv3d::load(
            weights,
            &format!("{prefix}.conv2"),
            out_ch,
            out_ch,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;
        let shortcut = if in_ch != out_ch {
            Some(HunyuanCausalConv3d::load(
                weights,
                &format!("{prefix}.conv_shortcut"),
                in_ch,
                out_ch,
                (1, 1, 1),
                (1, 1, 1),
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w,
            norm1_b,
            conv1,
            norm2_w,
            norm2_b,
            conv2,
            shortcut,
            in_ch,
            out_ch,
            num_groups,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Diffusers prepends a .contiguous() call; flame's GroupNorm path
        // reads in NCHW so we just hand it x.
        let residual = if let Some(s) = &self.shortcut {
            s.forward(x)?
        } else {
            x.clone()
        };
        let h = group_norm_5d(x, self.num_groups, &self.norm1_w, &self.norm1_b, self.eps)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = group_norm_5d(&h, self.num_groups, &self.norm2_w, &self.norm2_b, self.eps)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        h.add(&residual)
    }

    #[allow(dead_code)]
    fn channels(&self) -> (usize, usize) {
        (self.in_ch, self.out_ch)
    }
}

// ---------------------------------------------------------------------------
// HunyuanMidBlockAttention
//
// In diffusers, mid_block uses a single `Attention(_from_deprecated_attn_block=True)`
// with `residual_connection=True`. The attention is a one-head spatial-temporal
// self-attention over (T*H*W) tokens, with a CAUSAL mask in the temporal axis
// (frame i can only attend to frames 0..=i).
//
// Weight keys (under prefix `decoder.mid_block.attentions.0`):
//   group_norm.weight, group_norm.bias  — GroupNorm(num_groups=32, channels=in_channels)
//   to_q.weight, to_q.bias              — Linear(in_channels, in_channels)
//   to_k.weight, to_k.bias              — Linear(in_channels, in_channels)
//   to_v.weight, to_v.bias              — Linear(in_channels, in_channels)
//   to_out.0.weight, to_out.0.bias      — Linear(in_channels, in_channels)
//
// Forward, given [B, C, T, H, W]:
//   1. residual = x
//   2. h = group_norm(x)  (handled by Attention's `group_norm`)
//   3. permute h to [B, T*H*W, C]
//   4. q,k,v = to_q(h), to_k(h), to_v(h)  → reshape to [B, 1, T*H*W, C]   (one head)
//   5. attn = SDPA(q,k,v, mask=causal_temporal)
//   6. out = to_out(attn) reshape back to [B, C, T, H, W]
//   7. return out + residual
//
// TODO PARITY: build the causal attention mask. diffusers' helper
// `prepare_causal_attention_mask(num_frames, height*width, ...)` produces a
// `[seq_len, seq_len]` -inf/0 mask where token i can attend to all j with
// j_frame ≤ i_frame. Currently we approximate with the un-masked SDPA path
// because flame-core's sdpa_forward does not yet expose a frame-causal
// helper. For inference this typically still produces a usable image, but
// matches diffusers exactly only when num_frames == 1.
// ---------------------------------------------------------------------------

struct HunyuanMidBlockAttention {
    gn_w: Tensor,
    gn_b: Tensor,
    qw: Tensor,
    qb: Tensor,
    kw: Tensor,
    kb: Tensor,
    vw: Tensor,
    vb: Tensor,
    pw: Tensor,
    pb: Tensor,
    channels: usize,
    num_groups: usize,
    eps: f32,
}

impl HunyuanMidBlockAttention {
    fn load(
        weights: &Weights,
        prefix: &str,
        channels: usize,
        num_groups: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            gn_w: get_bf16(weights, &format!("{prefix}.group_norm.weight"))?,
            gn_b: get_bf16(weights, &format!("{prefix}.group_norm.bias"))?,
            qw: get_bf16(weights, &format!("{prefix}.to_q.weight"))?,
            qb: get_bf16(weights, &format!("{prefix}.to_q.bias"))?,
            kw: get_bf16(weights, &format!("{prefix}.to_k.weight"))?,
            kb: get_bf16(weights, &format!("{prefix}.to_k.bias"))?,
            vw: get_bf16(weights, &format!("{prefix}.to_v.weight"))?,
            vb: get_bf16(weights, &format!("{prefix}.to_v.bias"))?,
            pw: get_bf16(weights, &format!("{prefix}.to_out.0.weight"))?,
            pb: get_bf16(weights, &format!("{prefix}.to_out.0.bias"))?,
            channels,
            num_groups,
            eps,
        })
    }

    /// Linear with bias: y = x @ w.T + b
    fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        // x: [..., in], w: [out, in], b: [out]
        let dims = x.shape().dims().to_vec();
        let last = *dims.last().unwrap();
        let total: usize = dims.iter().take(dims.len() - 1).product();
        let x2 = x.reshape(&[total, last])?;
        let wt = w.permute(&[1, 0])?.contiguous()?; // [in, out]
        let y = x2.matmul(&wt)?.add(b)?;
        let mut out_dims = dims.clone();
        *out_dims.last_mut().unwrap() = w.shape().dims()[0];
        y.reshape(&out_dims)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        debug_assert_eq!(c, self.channels);

        let residual = x.clone();

        // GroupNorm in NCHW (folded T into spatial).
        let h_gn = group_norm_5d(x, self.num_groups, &self.gn_w, &self.gn_b, self.eps)?;

        // diffusers: permute(0, 2, 3, 4, 1).flatten(1, 3)  → [B, T*H*W, C]
        let n = t * h * w;
        let tokens = h_gn
            .permute(&[0, 2, 3, 4, 1])?
            .reshape(&[b, n, c])?
            .contiguous()?;

        // Q/K/V projections.
        let q = Self::linear(&tokens, &self.qw, &self.qb)?;
        let k = Self::linear(&tokens, &self.kw, &self.kb)?;
        let v = Self::linear(&tokens, &self.vw, &self.vb)?;

        // Single-head attention: reshape to [B, 1, N, C].
        let q4 = q.reshape(&[b, 1, n, c])?;
        let k4 = k.reshape(&[b, 1, n, c])?;
        let v4 = v.reshape(&[b, 1, n, c])?;

        // TODO PARITY: causal frame-mask not yet wired (see module note).
        let attn = sdpa_forward(&q4, &k4, &v4, None)
            .map_err(|e| Error::InvalidOperation(format!("HunyuanVAE mid attn SDPA: {e}")))?;

        // [B, 1, N, C] → [B, N, C]
        let attn = attn.reshape(&[b, n, c])?;

        // to_out projection.
        let out = Self::linear(&attn, &self.pw, &self.pb)?;

        // Reshape back: [B, N=T*H*W, C] → [B, T, H, W, C] → [B, C, T, H, W]
        let out = out
            .reshape(&[b, t, h, w, c])?
            .permute(&[0, 4, 1, 2, 3])?
            .contiguous()?;

        out.add(&residual)
    }
}

// ---------------------------------------------------------------------------
// HunyuanMidBlock3D
//
//   resnets[0] → attention → resnets[1]
// ---------------------------------------------------------------------------

struct HunyuanMidBlock3D {
    res0: HunyuanResnetBlock3D,
    attn: HunyuanMidBlockAttention,
    res1: HunyuanResnetBlock3D,
}

impl HunyuanMidBlock3D {
    fn load(
        weights: &Weights,
        prefix: &str,
        channels: usize,
        num_groups: usize,
        eps: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let res0 = HunyuanResnetBlock3D::load(
            weights,
            &format!("{prefix}.resnets.0"),
            channels,
            channels,
            num_groups,
            eps,
            device,
        )?;
        let attn = HunyuanMidBlockAttention::load(
            weights,
            &format!("{prefix}.attentions.0"),
            channels,
            num_groups,
            eps,
        )?;
        let res1 = HunyuanResnetBlock3D::load(
            weights,
            &format!("{prefix}.resnets.1"),
            channels,
            channels,
            num_groups,
            eps,
            device,
        )?;
        Ok(Self { res0, attn, res1 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.res0.forward(x)?;
        let h = self.attn.forward(&h)?;
        self.res1.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// HunyuanUpsampleCausal3D
//
// diffusers HunyuanVideoUpsampleCausal3D:
//   1. split frames (1, T-1)
//   2. first_frame: nearest-interpolate spatial only (H,W) -> (H*s_h, W*s_w)
//   3. other_frames: nearest-interpolate (T-1, H, W) -> ((T-1)*s_t, H*s_h, W*s_w)
//   4. cat back → [B, C, 1*s_h..._or_(T-1)*s_t, s_h*H, s_w*W]
//   5. CausalConv3d(in→out, 3, stride=1)
//
// upsample_factor is (s_t, s_h, s_w). For HunyuanVideo:
//   up_blocks.0 → (1, 2, 2)   (spatial only)
//   up_blocks.1 → (2, 2, 2)
//   up_blocks.2 → (2, 2, 2)
//   up_blocks.3 → no upsampler
// ---------------------------------------------------------------------------

struct HunyuanUpsampleCausal3D {
    conv: HunyuanCausalConv3d,
    upsample_factor: (usize, usize, usize),
}

impl HunyuanUpsampleCausal3D {
    fn load(
        weights: &Weights,
        prefix: &str,
        channels: usize,
        upsample_factor: (usize, usize, usize),
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let conv = HunyuanCausalConv3d::load(
            weights,
            &format!("{prefix}.conv"),
            channels,
            channels,
            (3, 3, 3),
            (1, 1, 1),
            device,
        )?;
        Ok(Self {
            conv,
            upsample_factor,
        })
    }

    /// Nearest-neighbor temporal repeat-interleave by factor s_t (stride 1).
    fn temporal_nearest(x: &Tensor, s_t: usize) -> Result<Tensor> {
        if s_t == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        // [B,C,T,H,W] → [B,C,T,1,H,W] → repeat along axis=3 by s_t → reshape
        let x_un = x.reshape(&[b, c, t, 1, h, w])?;
        let x_rep = x_un.repeat_axis_device(3, s_t)?;
        x_rep.reshape(&[b, c, t * s_t, h, w])
    }

    /// Per-frame spatial nearest 2× via flame-core's `upsample2d_nearest` (F32 path).
    fn spatial_nearest(x: &Tensor, s_h: usize, s_w: usize) -> Result<Tensor> {
        if s_h == 1 && s_w == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        // [B,C,T,H,W] → [B,T,C,H,W] → [B*T, C, H, W]
        let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
        let x_f32 = x_4d.to_dtype(DType::F32)?;
        let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * s_h, w * s_w))?;
        let x_up = x_up.to_dtype(DType::BF16)?;
        // [B*T, C, H*s_h, W*s_w] → [B,T,C,H*s_h,W*s_w] → [B,C,T,H*s_h,W*s_w]
        x_up
            .reshape(&[b, t, c, h * s_h, w * s_w])?
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (s_t, s_h, s_w) = self.upsample_factor;
        let t = dims[2];

        // Split (1, T-1) along temporal axis.
        let first = x.narrow(2, 0, 1)?;
        // first_frame: interpolate spatial only, no temporal scale.
        let first_up = Self::spatial_nearest(&first, s_h, s_w)?;

        let h_cat = if t > 1 {
            let others = x.narrow(2, 1, t - 1)?;
            // others: temporal interp first, then spatial.
            let others_t = Self::temporal_nearest(&others, s_t)?;
            let others_full = Self::spatial_nearest(&others_t, s_h, s_w)?;
            Tensor::cat(&[&first_up, &others_full], 2)?
        } else {
            first_up
        };

        self.conv.forward(&h_cat)
    }
}

// ---------------------------------------------------------------------------
// HunyuanUpBlock3D
//
//   resnets[0..N] → optional upsampler
//
// N = layers_per_block + 1 = 3 for HunyuanVideo (because decoder gets one extra
// block per stage relative to encoder).
// ---------------------------------------------------------------------------

struct HunyuanUpBlock3D {
    resnets: Vec<HunyuanResnetBlock3D>,
    upsampler: Option<HunyuanUpsampleCausal3D>,
}

impl HunyuanUpBlock3D {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,           // = layers_per_block + 1
        num_groups: usize,
        eps: f32,
        upsample_factor: Option<(usize, usize, usize)>,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(HunyuanResnetBlock3D::load(
                weights,
                &format!("{prefix}.resnets.{i}"),
                in_ch,
                out_channels,
                num_groups,
                eps,
                device,
            )?);
        }
        let upsampler = match upsample_factor {
            Some(f) => Some(HunyuanUpsampleCausal3D::load(
                weights,
                &format!("{prefix}.upsamplers.0"),
                out_channels,
                f,
                device,
            )?),
            None => None,
        };
        Ok(Self { resnets, upsampler })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// HunyuanVaeDecoder
// ---------------------------------------------------------------------------

pub struct HunyuanVaeDecoder {
    config: HunyuanVaeConfig,
    device: Arc<CudaDevice>,

    conv_in: HunyuanCausalConv3d,
    mid_block: HunyuanMidBlock3D,
    up_blocks: Vec<HunyuanUpBlock3D>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: HunyuanCausalConv3d,

    // Optional post_quant_conv (1×1×1) when present in the checkpoint. Several
    // diffusers VAE checkpoints store a `post_quant_conv` that maps z to z'
    // pre-decoder. The HunyuanVideo VAE stores it at the top level (not under
    // `decoder.`). We apply it just before `conv_in` if found.
    //
    // TODO PARITY: confirm whether this checkpoint expects the latent to first
    // pass through `post_quant_conv`. The Python diffusers script
    // `inference-flame/scripts/hunyuan15_decode.py` does NOT call it explicitly
    // because `vae.decode()` wraps it. Mirror the same behavior here.
    post_quant_conv_w: Option<Tensor>,
    post_quant_conv_b: Option<Tensor>,

    /// Raw weight bag, retained for debugging key lookups.
    weights: Weights,
}

impl HunyuanVaeDecoder {
    pub fn load(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let weights = load_file(Path::new(path), device)?;
        log::info!(
            "[HunyuanVAE] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        Self::from_weights(weights, HunyuanVaeConfig::default(), device.clone())
    }

    pub fn load_with_config(
        path: &str,
        config: HunyuanVaeConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let weights = load_file(Path::new(path), device)?;
        Self::from_weights(weights, config, device.clone())
    }

    fn from_weights(
        weights: Weights,
        config: HunyuanVaeConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let eps = 1e-6;
        let groups = config.norm_num_groups;

        // conv_in: latent_channels → block_out_channels[-1]
        let top_ch = *config.block_out_channels.last().unwrap();
        let conv_in = HunyuanCausalConv3d::load(
            &weights,
            "decoder.conv_in",
            config.latent_channels,
            top_ch,
            (3, 3, 3),
            (1, 1, 1),
            &device,
        )?;

        let mid_block = HunyuanMidBlock3D::load(
            &weights,
            "decoder.mid_block",
            top_ch,
            groups,
            eps,
            &device,
        )?;

        // up_blocks: iterate reversed(block_out_channels) i.e. [512,512,256,128].
        // For HunyuanVideo: temporal compression = 4, spatial compression = 8.
        //   num_spatial_upsample_layers = log2(8) = 3
        //   num_time_upsample_layers    = log2(4) = 2
        //   For i in 0..4:
        //     add_spatial = (i < 3)
        //     add_time    = (i >= len(block_out)-1-num_time AND not is_final)
        //                 = (i >= 1 AND i != 3)
        //   →  i=0: spatial only           → factor=(1,2,2)
        //      i=1: temporal+spatial       → factor=(2,2,2)
        //      i=2: temporal+spatial       → factor=(2,2,2)
        //      i=3: no upsample            → no upsampler
        let block_out = &config.block_out_channels;
        let n = block_out.len();
        let reversed: Vec<usize> = block_out.iter().rev().copied().collect();
        let num_spatial = (config.spatial_compression_ratio as f32).log2().round() as usize;
        let num_time = (config.temporal_compression_ratio as f32).log2().round() as usize;

        let mut up_blocks = Vec::with_capacity(n);
        let mut prev_out = reversed[0]; // 512
        for i in 0..n {
            let out_ch = reversed[i];
            let is_final = i == n - 1;
            let add_spatial = i < num_spatial;
            let add_time = (i >= n - 1 - num_time) && !is_final;
            let upsample_factor = if add_spatial || add_time {
                let s_t = if add_time { 2 } else { 1 };
                let s_h = if add_spatial { 2 } else { 1 };
                let s_w = if add_spatial { 2 } else { 1 };
                Some((s_t, s_h, s_w))
            } else {
                None
            };
            let block = HunyuanUpBlock3D::load(
                &weights,
                &format!("decoder.up_blocks.{i}"),
                prev_out,
                out_ch,
                config.layers_per_block + 1,
                groups,
                eps,
                upsample_factor,
                &device,
            )?;
            up_blocks.push(block);
            prev_out = out_ch;
        }

        let bottom_ch = block_out[0];
        let norm_out_w = get_bf16(&weights, "decoder.conv_norm_out.weight")?;
        let norm_out_b = get_bf16(&weights, "decoder.conv_norm_out.bias")?;

        let conv_out = HunyuanCausalConv3d::load(
            &weights,
            "decoder.conv_out",
            bottom_ch,
            config.out_channels,
            (3, 3, 3),
            (1, 1, 1),
            &device,
        )?;

        let post_quant_conv_w = if has_key(&weights, "post_quant_conv.weight") {
            Some(get_f32(&weights, "post_quant_conv.weight")?)
        } else {
            None
        };
        let post_quant_conv_b = if has_key(&weights, "post_quant_conv.bias") {
            Some(get_f32(&weights, "post_quant_conv.bias")?)
        } else {
            None
        };

        Ok(Self {
            config,
            device,
            conv_in,
            mid_block,
            up_blocks,
            norm_out_w,
            norm_out_b,
            conv_out,
            post_quant_conv_w,
            post_quant_conv_b,
            weights,
        })
    }

    pub fn config(&self) -> &HunyuanVaeConfig {
        &self.config
    }
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }
    pub fn weight_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.weights.keys().map(|s| s.as_str()).collect();
        keys.sort();
        keys
    }
    pub fn get_weight(&self, key: &str) -> Result<Tensor> {
        get(&self.weights, key)
    }
    pub fn has_weight(&self, key: &str) -> bool {
        has_key(&self.weights, key)
    }

    /// Apply optional post_quant_conv (1×1×1 conv). Implemented as a per-channel
    /// linear since kernel is 1: out[b,c_o,t,h,w] = sum_c W[c_o,c] * x[b,c,t,h,w] + b[c_o].
    fn apply_post_quant_conv(&self, x: &Tensor) -> Result<Tensor> {
        let (Some(w), Some(b)) = (&self.post_quant_conv_w, &self.post_quant_conv_b) else {
            return Ok(x.clone());
        };
        // w: [c_out, c_in, 1, 1, 1] → [c_out, c_in]
        let wd = w.shape().dims();
        let c_out = wd[0];
        let c_in = wd[1];
        let w2 = w.reshape(&[c_out, c_in])?;
        // x: [B, C_in, T, H, W] → [B, T, H, W, C_in] → [B*T*H*W, C_in]
        let dims = x.shape().dims().to_vec();
        let (bb, _c, t, h, ww) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let n = bb * t * h * ww;
        let x_perm = x
            .permute(&[0, 2, 3, 4, 1])?
            .reshape(&[n, c_in])?
            .to_dtype(DType::F32)?;
        let wt = w2.permute(&[1, 0])?.contiguous()?; // [c_in, c_out]
        let y = x_perm.matmul(&wt)?.add(b)?;
        let y = y
            .reshape(&[bb, t, h, ww, c_out])?
            .permute(&[0, 4, 1, 2, 3])?
            .contiguous()?;
        if x.dtype() == DType::BF16 {
            y.to_dtype(DType::BF16)
        } else {
            Ok(y)
        }
    }

    /// Decode latents to pixels.
    ///
    /// Input:  `[B, 16, T, H, W]` BF16 latents (already at output dtype; the
    ///         caller is expected to have applied the scaling factor).
    /// Output: `[B, 3, T_out, H_out, W_out]` BF16 pixels in roughly [-1, 1].
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "HunyuanVAE decode expects 5D [B,C,T,H,W], got {:?}",
                dims
            )));
        }
        let c = dims[1];
        if c != self.config.latent_channels {
            return Err(Error::InvalidInput(format!(
                "HunyuanVAE: expected {} latent channels, got {}",
                self.config.latent_channels, c
            )));
        }

        // Optional post_quant_conv (matches diffusers .decode()).
        let h = self.apply_post_quant_conv(latents)?;

        // conv_in: [B, 16, T, H, W] → [B, 512, T, H, W]
        let mut h = self.conv_in.forward(&h)?;

        // mid_block: ResBlock + Attention + ResBlock at top channels.
        h = self.mid_block.forward(&h)?;

        // up_blocks (4 stages).
        for (idx, block) in self.up_blocks.iter().enumerate() {
            h = block.forward(&h)?;
            log::debug!(
                "[HunyuanVAE] up_blocks.{idx} out shape={:?}",
                h.shape().dims()
            );
        }

        // conv_norm_out + SiLU + conv_out.
        let h = group_norm_5d(
            &h,
            self.config.norm_num_groups,
            &self.norm_out_w,
            &self.norm_out_b,
            1e-6,
        )?;
        let h = h.silu()?;
        let out = self.conv_out.forward(&h)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = HunyuanVaeConfig::default();
        assert_eq!(cfg.latent_channels, 16);
        assert_eq!(cfg.out_channels, 3);
        assert!((cfg.scaling_factor - 0.476986).abs() < 1e-5);
        assert_eq!(cfg.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(cfg.layers_per_block, 2);
        assert_eq!(cfg.norm_num_groups, 32);
        assert_eq!(cfg.spatial_compression_ratio, 8);
        assert_eq!(cfg.temporal_compression_ratio, 4);
    }

    #[test]
    fn test_velocity_schedule_sanity() {
        let cfg = HunyuanVaeConfig::default();
        let inv_scale = 1.0 / cfg.scaling_factor;
        assert!(inv_scale > 2.0 && inv_scale < 2.2, "inv_scale = {inv_scale}");
    }
}
