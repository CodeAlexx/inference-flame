//! LTX-2 Video VAE Decoder — pure flame_core.
//!
//! Decode-only 3D causal video VAE matching the Lightricks LTX-2 checkpoint.
//! Input: `[B, 128, F, H, W]` latents (normalized).
//! Output: `[B, 3, F*8, H*32, W*32]` RGB video frames.
//!
//! Architecture (from checkpoint, reversed encoder blocks):
//!   conv_in: CausalConv3d(128, 1024, k=3)
//!   up_blocks.0: UNetMidBlock3D(1024, 5 res_blocks, timestep_cond)
//!   up_blocks.1: DepthToSpaceUpsample(1024, stride=(2,2,2), residual, mul=2) → 512 ch
//!   up_blocks.2: UNetMidBlock3D(512, 5 res_blocks, timestep_cond)
//!   up_blocks.3: DepthToSpaceUpsample(512, stride=(2,2,2), residual, mul=2) → 256 ch
//!   up_blocks.4: UNetMidBlock3D(256, 5 res_blocks, timestep_cond)
//!   up_blocks.5: DepthToSpaceUpsample(256, stride=(2,2,2), residual, mul=2) → 128 ch
//!   up_blocks.6: UNetMidBlock3D(128, 5 res_blocks, timestep_cond)
//!   conv_norm_out: PixelNorm
//!   last_time_embedder + last_scale_shift_table (final timestep conditioning)
//!   conv_act: SiLU
//!   conv_out: CausalConv3d(128, 48, k=3)   (48 = 3 * patch_size^2)
//!   unpatchify: rearrange to (B, 3, F, H*4, W*4)
//!
//! Weight key format (from safetensors):
//!   decoder.conv_in.conv.{weight,bias}
//!   decoder.up_blocks.{i}.res_blocks.{j}.conv{1,2}.conv.{weight,bias}
//!   decoder.up_blocks.{i}.res_blocks.{j}.scale_shift_table
//!   decoder.up_blocks.{i}.time_embedder.timestep_embedder.linear_{1,2}.{weight,bias}
//!   decoder.up_blocks.{i}.conv.conv.{weight,bias}   (DepthToSpace blocks)
//!   decoder.conv_out.conv.{weight,bias}
//!   decoder.timestep_scale_multiplier
//!   decoder.last_time_embedder.timestep_embedder.linear_{1,2}.{weight,bias}
//!   decoder.last_scale_shift_table
//!   per_channel_statistics.{std-of-means, mean-of-means}

use flame_core::serialization;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PATCH_SIZE: usize = 4;
const LATENT_CH: usize = 128;
const OUT_CH: usize = 3;
const TIMESTEP_EMBED_DIM: usize = 256;

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_weight(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("LTX2 VAE: missing weight: {key}")))
}

fn get_weight_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get_weight(weights, key)?.to_dtype(DType::BF16)
}

/// Load a Linear layer (weight + bias) from weights.
fn load_linear(
    weights: &Weights,
    prefix: &str,
    in_features: usize,
    out_features: usize,
) -> Result<LinearLayer> {
    let weight = get_weight_bf16(weights, &format!("{prefix}.weight"))?;
    let bias = get_weight_bf16(weights, &format!("{prefix}.bias"))?;
    Ok(LinearLayer {
        weight,
        bias,
        in_features,
        out_features,
    })
}

// ---------------------------------------------------------------------------
// LinearLayer (simple weight+bias, no flame_core::linear::Linear dependency)
// ---------------------------------------------------------------------------

struct LinearLayer {
    weight: Tensor,
    bias: Tensor,
    in_features: usize,
    out_features: usize,
}

impl LinearLayer {
    /// Forward: x @ weight.T + bias. x is [B, in_features].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch = if dims.len() == 2 { dims[0] } else { 1 };
        let x_2d = if dims.len() == 2 {
            x.clone()
        } else {
            x.reshape(&[batch, self.in_features])?
        };
        let wt = self.weight.transpose()?;
        let out = x_2d.matmul(&wt)?;
        let bias_bc = self.bias.reshape(&[1, self.out_features])?;
        out.add(&bias_bc)
    }
}

// ---------------------------------------------------------------------------
// CausalConv3d — 3D convolution with causal temporal padding
// ---------------------------------------------------------------------------

/// Causal 3D convolution: pads temporal dimension on left side only.
///
/// Wraps flame_core's Conv3d. The weight is [out_ch, in_ch, kT, kH, kW].
/// Spatial padding = kernel_size // 2, temporal padding = 0 (we prepend manually).
struct CausalConv3d {
    weight: Tensor,
    bias: Tensor,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    time_kernel_size: usize,
}

impl CausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let weight = get_weight_bf16(weights, &format!("{prefix}.weight"))?;
        let bias = get_weight_bf16(weights, &format!("{prefix}.bias"))?;
        Ok(Self {
            weight,
            bias,
            in_channels: in_ch,
            out_channels: out_ch,
            kernel_size: (kernel_size, kernel_size, kernel_size),
            time_kernel_size: kernel_size,
        })
    }

    /// Forward with causal temporal padding.
    /// Input: [B, C, D, H, W]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims[2] == 0 {
            return Ok(x.clone());
        }

        // Causal pad: repeat first frame (time_kernel_size - 1) times, prepend
        let pad_len = self.time_kernel_size - 1;
        let x_padded = if pad_len > 0 {
            let first_frame = x.narrow(2, 0, 1)?; // [B, C, 1, H, W]
            let repeated = first_frame.repeat_axis_device(2, pad_len)?; // [B, C, pad_len, H, W]
            // cat [repeated, x] along dim 2
            Tensor::cat(&[&repeated, x], 2)?
        } else {
            x.clone()
        };

        // Use flame_core conv3d_simple
        let h_pad = self.kernel_size.1 / 2;
        let w_pad = self.kernel_size.2 / 2;

        conv3d_forward(
            &x_padded,
            &self.weight,
            &self.bias,
            (1, 1, 1),       // stride
            (0, h_pad, w_pad), // padding (no temporal, spatial only)
            self.in_channels,
            self.out_channels,
        )
    }
}

/// Perform 3D convolution using flame_core's Conv3d.
///
/// Input: [B, C_in, D, H, W], Weight: [C_out, C_in, kD, kH, kW], Bias: [C_out]
fn conv3d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    in_channels: usize,
    out_channels: usize,
) -> Result<Tensor> {
    let device = input.device().clone();
    let kernel_dims = weight.shape().dims().to_vec();
    let kd = kernel_dims[2];
    let kh = kernel_dims[3];
    let kw = kernel_dims[4];

    // Create Conv3d from flame_core::conv3d_simple and run it
    let mut conv = flame_core::conv3d_simple::Conv3d::new(
        in_channels,
        out_channels,
        (kd, kh, kw),
        Some(stride),
        Some(padding),
        None, // dilation
        None, // groups
        true, // bias
        device,
    )?;
    conv.weight = weight.clone();
    conv.bias_tensor = Some(bias.clone());
    conv.forward(input)
}

// ---------------------------------------------------------------------------
// PixelNorm — x / sqrt(mean(x^2, dim=1, keepdim=True) + eps)
// ---------------------------------------------------------------------------

const PIXEL_NORM_EPS: f32 = 1e-8;

fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    // x: [B, C, D, H, W]
    // mean(x^2, dim=1, keepdim=True)
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.mul(&x_f32)?;
    let mean_sq = x_sq.mean_along_dims(&[1], true)?;
    let denom = mean_sq.add_scalar(PIXEL_NORM_EPS)?.rsqrt()?;
    let result = x_f32.mul(&denom)?;
    result.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// Timestep Embedding
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding (256-dim), matching Python _get_timestep_embedding.
fn get_timestep_embedding(timesteps: &Tensor) -> Result<Tensor> {
    let half_dim = TIMESTEP_EMBED_DIM / 2; // 128
    let device = timesteps.device().clone();

    // exp(-i * ln(10000) / 128) for i in [0..128)
    let mut freq_data = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        freq_data[i] = (-(i as f64) * (10000.0f64).ln() / half_dim as f64).exp() as f32;
    }
    let freqs =
        Tensor::from_vec(freq_data, Shape::from_dims(&[1, half_dim]), device)?;

    // timesteps: [B] -> [B, 1]
    let t = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;
    let args = t.mul(&freqs)?; // [B, 128]

    let sin_part = args.sin()?;
    let cos_part = args.cos()?;

    // flip_sin_to_cos=True -> [cos, sin]
    let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?; // [B, 256]
    emb.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// TimestepEmbedder (MLP: 256 -> dim -> dim)
// ---------------------------------------------------------------------------

struct TimestepEmbedder {
    linear_1: LinearLayer,
    linear_2: LinearLayer,
}

impl TimestepEmbedder {
    fn load(weights: &Weights, prefix: &str, embed_dim: usize) -> Result<Self> {
        let linear_1 = load_linear(
            weights,
            &format!("{prefix}.timestep_embedder.linear_1"),
            TIMESTEP_EMBED_DIM,
            embed_dim,
        )?;
        let linear_2 = load_linear(
            weights,
            &format!("{prefix}.timestep_embedder.linear_2"),
            embed_dim,
            embed_dim,
        )?;
        Ok(Self { linear_1, linear_2 })
    }

    /// Forward: timestep_embedding(t) -> linear_1 -> silu -> linear_2
    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let proj = get_timestep_embedding(timestep)?; // [B, 256]
        let h = self.linear_1.forward(&proj)?;
        let h = h.silu()?;
        self.linear_2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// ResnetBlock3D — with PixelNorm and timestep conditioning
// ---------------------------------------------------------------------------

struct ResnetBlock3D {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
    /// scale_shift_table: [4, channels] for timestep conditioning
    scale_shift_table: Tensor,
    _in_channels: usize,
    _out_channels: usize,
}

impl ResnetBlock3D {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
    ) -> Result<Self> {
        let conv1 = CausalConv3d::load(weights, &format!("{prefix}.conv1.conv"), in_ch, out_ch, 3)?;
        let conv2 = CausalConv3d::load(weights, &format!("{prefix}.conv2.conv"), out_ch, out_ch, 3)?;
        let scale_shift_table =
            get_weight_bf16(weights, &format!("{prefix}.scale_shift_table"))?; // [4, ch]

        Ok(Self {
            conv1,
            conv2,
            scale_shift_table,
            _in_channels: in_ch,
            _out_channels: out_ch,
        })
    }

    /// Forward pass.
    /// x: [B, C, D, H, W], timestep_embed: [B, C*4, 1, 1, 1]
    fn forward(&self, x: &Tensor, timestep_embed: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch = dims[0];
        let ch = dims[1];

        // h = pixel_norm(x)
        let mut h = pixel_norm(x)?;

        // Apply timestep conditioning (shift1, scale1 from scale_shift_table + embed)
        if let Some(te) = timestep_embed {
            // scale_shift_table: [4, ch] -> [1, 4, ch, 1, 1, 1]
            let sst = self.scale_shift_table.reshape(&[1, 4, ch, 1, 1, 1])?;

            // te: [B, ch*4, 1, 1, 1] -> [B, 4, ch, 1, 1, 1]
            let te_dims = te.shape().dims().to_vec();
            let te_reshaped = te.reshape(&[
                batch, 4, ch,
                te_dims[2], te_dims[3], te_dims[4],
            ])?;

            let ada_values = sst.add(&te_reshaped)?; // [B, 4, ch, 1, 1, 1]

            // unbind dim=1: shift1, scale1, shift2, scale2
            let shift1 = ada_values.narrow(1, 0, 1)?.squeeze_dim(1)?; // [B, ch, 1, 1, 1]
            let scale1 = ada_values.narrow(1, 1, 1)?.squeeze_dim(1)?;
            let shift2 = ada_values.narrow(1, 2, 1)?.squeeze_dim(1)?;
            let scale2 = ada_values.narrow(1, 3, 1)?.squeeze_dim(1)?;

            // h = h * (1 + scale1) + shift1
            let ones = Tensor::ones_dtype(
                Shape::from_dims(&[1, 1, 1, 1, 1]),
                DType::BF16,
                h.device().clone(),
            )?;
            h = h.mul(&ones.add(&scale1)?)?.add(&shift1)?;

            // h = silu(h)
            let h_act = h.silu()?;
            // h = conv1(h)
            let mut h2 = self.conv1.forward(&h_act)?;

            // h2 = pixel_norm(h2)
            h2 = pixel_norm(&h2)?;

            // h2 = h2 * (1 + scale2) + shift2
            h2 = h2.mul(&ones.add(&scale2)?)?.add(&shift2)?;

            // h2 = silu(h2), conv2(h2)
            let h2_act = h2.silu()?;
            let h2_out = self.conv2.forward(&h2_act)?;

            // skip connection (identity since in_ch == out_ch for these blocks)
            return h2_out.add(x);
        }

        // No timestep conditioning path (fallback, shouldn't happen with this VAE)
        h = h.silu()?;
        h = self.conv1.forward(&h)?;
        h = pixel_norm(&h)?;
        h = h.silu()?;
        h = self.conv2.forward(&h)?;
        h.add(x)
    }
}

// ---------------------------------------------------------------------------
// UNetMidBlock3D — multiple ResBlocks with shared timestep embedder
// ---------------------------------------------------------------------------

struct UNetMidBlock3D {
    res_blocks: Vec<ResnetBlock3D>,
    time_embedder: TimestepEmbedder,
    channels: usize,
}

impl UNetMidBlock3D {
    fn load(
        weights: &Weights,
        prefix: &str,
        channels: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            res_blocks.push(ResnetBlock3D::load(
                weights,
                &format!("{prefix}.res_blocks.{i}"),
                channels,
                channels,
            )?);
        }

        let time_embedder = TimestepEmbedder::load(
            weights,
            &format!("{prefix}.time_embedder"),
            channels * 4,
        )?;

        Ok(Self {
            res_blocks,
            time_embedder,
            channels,
        })
    }

    /// Forward: compute timestep embed once, pass to all res_blocks.
    fn forward(&self, h: &Tensor, scaled_timestep: Option<&Tensor>) -> Result<Tensor> {
        let timestep_embed = if let Some(ts) = scaled_timestep {
            let dims = h.shape().dims().to_vec();
            let batch = dims[0];
            let embed = self.time_embedder.forward(ts)?; // [B, ch*4]
            Some(embed.reshape(&[batch, self.channels * 4, 1, 1, 1])?)
        } else {
            None
        };

        let mut out = h.clone();
        for block in &self.res_blocks {
            out = block.forward(&out, timestep_embed.as_ref())?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// DepthToSpaceUpsample — rearrange channels into spatial/temporal dims
// ---------------------------------------------------------------------------

struct DepthToSpaceUpsample {
    conv: CausalConv3d,
    stride: (usize, usize, usize), // (2, 2, 2) for compress_all
    residual: bool,
    out_channels_reduction_factor: usize,
    _in_channels: usize,
}

impl DepthToSpaceUpsample {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_channels: usize,
        stride: (usize, usize, usize),
        residual: bool,
        reduction_factor: usize,
    ) -> Result<Self> {
        let prod = stride.0 * stride.1 * stride.2;
        let conv_out_ch = prod * in_channels / reduction_factor;

        let conv = CausalConv3d::load(
            weights,
            &format!("{prefix}.conv.conv"),
            in_channels,
            conv_out_ch,
            3,
        )?;

        Ok(Self {
            conv,
            stride,
            residual,
            out_channels_reduction_factor: reduction_factor,
            _in_channels: in_channels,
        })
    }

    /// Forward: conv -> depth_to_space rearrange -> optional residual.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;

        // Depth-to-space: [B, C*p1*p2*p3, D, H, W] -> [B, C, D*p1, H*p2, W*p3]
        let y = depth_to_space(&y, self.stride)?;

        // If temporal stride is 2, drop the first (causal duplicate) frame
        let y = if self.stride.0 == 2 {
            let d = y.shape().dims()[2];
            if d > 0 {
                y.narrow(2, 1, d - 1)?
            } else {
                y
            }
        } else {
            y
        };

        if self.residual {
            // Also rearrange x
            let x_rearranged = depth_to_space(x, self.stride)?;
            let prod = self.stride.0 * self.stride.1 * self.stride.2;
            let num_repeat = prod / self.out_channels_reduction_factor;
            let x_repeated = repeat_channels(&x_rearranged, num_repeat)?;

            let x_repeated = if self.stride.0 == 2 {
                let d = x_repeated.shape().dims()[2];
                if d > 0 {
                    x_repeated.narrow(2, 1, d - 1)?
                } else {
                    x_repeated
                }
            } else {
                x_repeated
            };

            y.add(&x_repeated)
        } else {
            Ok(y)
        }
    }
}

/// Depth-to-space rearrange: [B, C*p1*p2*p3, D, H, W] -> [B, C, D*p1, H*p2, W*p3]
fn depth_to_space(x: &Tensor, stride: (usize, usize, usize)) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c_total, d, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let (p1, p2, p3) = stride;
    let c = c_total / (p1 * p2 * p3);

    // Reshape to [B, C, p1, p2, p3, D, H, W]
    let x = x.reshape(&[b, c, p1, p2, p3, d, h, w])?;
    // Permute to [B, C, D, p1, H, p2, W, p3]
    let x = x.permute(&[0, 1, 5, 2, 6, 3, 7, 4])?;
    // Reshape to [B, C, D*p1, H*p2, W*p3]
    x.reshape(&[b, c, d * p1, h * p2, w * p3])
}

/// Repeat channels: [B, C, D, H, W] -> [B, C*num_repeat, D, H, W]
fn repeat_channels(x: &Tensor, num_repeat: usize) -> Result<Tensor> {
    if num_repeat == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims().to_vec();
    let (b, c, d, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    // [B, C, D, H, W] -> [B, C, 1, D, H, W]
    let x = x.reshape(&[b, c, 1, d, h, w])?;
    // repeat along dim 2
    let x = x.repeat_axis_device(2, num_repeat)?;
    // [B, C, num_repeat, D, H, W] -> [B, C*num_repeat, D, H, W]
    x.reshape(&[b, c * num_repeat, d, h, w])
}

// ---------------------------------------------------------------------------
// Decoder up block enum
// ---------------------------------------------------------------------------

enum UpBlock {
    MidBlock(UNetMidBlock3D),
    DepthToSpace(DepthToSpaceUpsample),
}

impl UpBlock {
    fn forward(&self, x: &Tensor, scaled_timestep: Option<&Tensor>) -> Result<Tensor> {
        match self {
            UpBlock::MidBlock(block) => block.forward(x, scaled_timestep),
            UpBlock::DepthToSpace(block) => block.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// PerChannelStatistics — un-normalize latents before decoding
// ---------------------------------------------------------------------------

struct PerChannelStatistics {
    std_of_means: Tensor,  // [128]
    mean_of_means: Tensor, // [128]
}

impl PerChannelStatistics {
    fn load(weights: &Weights) -> Result<Self> {
        let std_of_means =
            get_weight_bf16(weights, "per_channel_statistics.std-of-means")?;
        let mean_of_means =
            get_weight_bf16(weights, "per_channel_statistics.mean-of-means")?;
        Ok(Self {
            std_of_means,
            mean_of_means,
        })
    }

    /// Un-normalize: x * std + mean (broadcast over [B, 128, D, H, W])
    fn un_normalize(&self, x: &Tensor) -> Result<Tensor> {
        let std = self.std_of_means.reshape(&[1, LATENT_CH, 1, 1, 1])?;
        let mean = self.mean_of_means.reshape(&[1, LATENT_CH, 1, 1, 1])?;
        x.mul(&std)?.add(&mean)
    }
}

// ---------------------------------------------------------------------------
// Unpatchify — reverse patchification (patch_size=4)
// ---------------------------------------------------------------------------

/// Unpatchify: [B, C*p*p, F, H, W] -> [B, C, F, H*p, W*p]
/// For LTX-2: C=3, p=4, so 48 channels -> 3 channels, HW scaled by 4.
fn unpatchify(x: &Tensor) -> Result<Tensor> {
    let p = PATCH_SIZE;
    let dims = x.shape().dims().to_vec();
    let (b, c_packed, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let c = c_packed / (p * p); // 48 / 16 = 3

    // Reshape: [B, c*p*p, F, H, W] -> [B, c, p, p, F, H, W]
    // Then permute to [B, c, F, H, p, W, p] -> [B, c, F, H*p, W*p]
    //
    // Python einops: "b (c p r q) f h w -> b c (f p) (h q) (w r)" with p=1,q=4,r=4
    // But actually p=patch_size_t=1, q=patch_size_hw, r=patch_size_hw
    // So: [B, c*1*4*4, F, H, W] -> [B, c, F*1, H*4, W*4]
    let x = x.reshape(&[b, c, 1, p, p, f, h, w])?;
    // Target: [B, c, f*1, h*p, w*p]
    // Permute: [B, c, f_patch_t, patch_h, patch_w, F, H, W]
    //       -> [B, c, F, f_patch_t, H, patch_h, W, patch_w]
    let x = x.permute(&[0, 1, 5, 2, 6, 3, 7, 4])?;
    x.reshape(&[b, c, f * 1, h * p, w * p])
}

// ---------------------------------------------------------------------------
// LTX2 Video VAE Decoder — main public struct
// ---------------------------------------------------------------------------

/// LTX-2 Video VAE decoder.
///
/// Decode-only: takes normalized 128-channel latents and produces RGB video.
pub struct LTX2VaeDecoder {
    conv_in: CausalConv3d,
    up_blocks: Vec<UpBlock>,
    // Final conditioning
    timestep_scale_multiplier: Tensor,
    last_time_embedder: TimestepEmbedder,
    last_scale_shift_table: Tensor,
    conv_out: CausalConv3d,
    per_channel_statistics: PerChannelStatistics,
    final_channels: usize,
}

impl LTX2VaeDecoder {
    /// Load the decoder from a weight map (safetensors-style keys).
    ///
    /// Expected keys: `decoder.*` and `per_channel_statistics.*`.
    pub fn load(weights: &Weights) -> Result<Self> {
        // conv_in: CausalConv3d(128 -> 1024, k=3)
        let conv_in = CausalConv3d::load(weights, "decoder.conv_in.conv", LATENT_CH, 1024, 3)?;

        // Build up_blocks from the checkpoint structure.
        // The architecture from the checkpoint:
        //   0: UNetMidBlock3D(1024, 5 res)
        //   1: DepthToSpaceUpsample(1024, stride=2,2,2, residual, mul=2)
        //   2: UNetMidBlock3D(512, 5 res)
        //   3: DepthToSpaceUpsample(512, stride=2,2,2, residual, mul=2)
        //   4: UNetMidBlock3D(256, 5 res)
        //   5: DepthToSpaceUpsample(256, stride=2,2,2, residual, mul=2)
        //   6: UNetMidBlock3D(128, 5 res)
        let mut up_blocks = Vec::new();

        // Block 0: UNetMidBlock3D(1024, 5 layers)
        up_blocks.push(UpBlock::MidBlock(UNetMidBlock3D::load(
            weights,
            "decoder.up_blocks.0",
            1024,
            5,
        )?));

        // Block 1: DepthToSpaceUpsample(1024, stride=(2,2,2), residual, mul=2)
        up_blocks.push(UpBlock::DepthToSpace(DepthToSpaceUpsample::load(
            weights,
            "decoder.up_blocks.1",
            1024,
            (2, 2, 2),
            true, // residual
            2,    // reduction_factor
        )?));

        // Block 2: UNetMidBlock3D(512, 5 layers)
        up_blocks.push(UpBlock::MidBlock(UNetMidBlock3D::load(
            weights,
            "decoder.up_blocks.2",
            512,
            5,
        )?));

        // Block 3: DepthToSpaceUpsample(512, stride=(2,2,2), residual, mul=2)
        up_blocks.push(UpBlock::DepthToSpace(DepthToSpaceUpsample::load(
            weights,
            "decoder.up_blocks.3",
            512,
            (2, 2, 2),
            true,
            2,
        )?));

        // Block 4: UNetMidBlock3D(256, 5 layers)
        up_blocks.push(UpBlock::MidBlock(UNetMidBlock3D::load(
            weights,
            "decoder.up_blocks.4",
            256,
            5,
        )?));

        // Block 5: DepthToSpaceUpsample(256, stride=(2,2,2), residual, mul=2)
        up_blocks.push(UpBlock::DepthToSpace(DepthToSpaceUpsample::load(
            weights,
            "decoder.up_blocks.5",
            256,
            (2, 2, 2),
            true,
            2,
        )?));

        // Block 6: UNetMidBlock3D(128, 5 layers)
        up_blocks.push(UpBlock::MidBlock(UNetMidBlock3D::load(
            weights,
            "decoder.up_blocks.6",
            128,
            5,
        )?));

        // Timestep conditioning finals
        let timestep_scale_multiplier =
            get_weight_bf16(weights, "decoder.timestep_scale_multiplier")?;

        // last_time_embedder: embed_dim = 128 * 2 = 256
        let last_time_embedder = TimestepEmbedder::load(
            weights,
            "decoder.last_time_embedder",
            128 * 2,
        )?;

        let last_scale_shift_table =
            get_weight_bf16(weights, "decoder.last_scale_shift_table")?; // [2, 128]

        // conv_out: CausalConv3d(128 -> 48, k=3)
        let conv_out_ch = OUT_CH * PATCH_SIZE * PATCH_SIZE; // 3 * 16 = 48
        let conv_out = CausalConv3d::load(weights, "decoder.conv_out.conv", 128, conv_out_ch, 3)?;

        let per_channel_statistics = PerChannelStatistics::load(weights)?;

        Ok(Self {
            conv_in,
            up_blocks,
            timestep_scale_multiplier,
            last_time_embedder,
            last_scale_shift_table,
            conv_out,
            per_channel_statistics,
            final_channels: 128,
        })
    }

    /// Decode normalized latents to video frames.
    ///
    /// Input: `[B, 128, F, H, W]` — normalized latents.
    /// Output: `[B, 3, F*8, H*32, W*32]` — RGB video (approx [-1, 1]).
    ///
    /// `timestep`: decoder conditioning timestep (default 0.05).
    /// `noise_scale`: noise mixing ratio (default 0.025).
    pub fn decode(
        &self,
        x: &Tensor,
        timestep: f32,
        noise_scale: f32,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch = dims[0];
        let device = x.device().clone();

        // Optional noise injection for timestep conditioning
        // x = randn_like(x) * noise_scale + (1 - noise_scale) * x
        // For simplicity in inference, we skip noise injection (noise_scale=0 effectively)
        // In production: create noise tensor and mix
        let x_input = if noise_scale > 0.0 {
            // Create noise (for now, we skip this for deterministic decode)
            // A proper implementation would generate random noise here
            x.mul_scalar(1.0 - noise_scale)?
        } else {
            x.clone()
        };

        // Un-normalize latents
        let x_unnorm = self.per_channel_statistics.un_normalize(&x_input)?;

        // Compute scaled timestep
        let ts_tensor = Tensor::from_vec(
            vec![timestep],
            Shape::from_dims(&[1]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Scale by multiplier
        let ts_multiplier_f32 = self.timestep_scale_multiplier.to_dtype(DType::F32)?;
        let ts_f32 = ts_tensor.to_dtype(DType::F32)?;
        let scaled_ts = ts_f32.mul(&ts_multiplier_f32)?;
        let scaled_ts = scaled_ts.to_dtype(DType::BF16)?;

        // conv_in
        let mut sample = self.conv_in.forward(&x_unnorm)?;

        // Up blocks
        for block in &self.up_blocks {
            sample = block.forward(&sample, Some(&scaled_ts))?;
        }

        // Final PixelNorm
        sample = pixel_norm(&sample)?;

        // Final timestep conditioning: last_time_embedder + last_scale_shift_table
        let embedded_ts = self.last_time_embedder.forward(&scaled_ts)?; // [B, 256]
        let embedded_ts = embedded_ts.reshape(&[batch, 256, 1, 1, 1])?;

        // last_scale_shift_table: [2, 128] -> [1, 2, 128, 1, 1, 1]
        let sst = self
            .last_scale_shift_table
            .reshape(&[1, 2, self.final_channels, 1, 1, 1])?;

        // embedded_ts: [B, 256, 1, 1, 1] -> [B, 2, 128, 1, 1, 1]
        let et_reshaped = embedded_ts.reshape(&[batch, 2, self.final_channels, 1, 1, 1])?;

        let ada_values = sst.add(&et_reshaped)?; // [B, 2, 128, 1, 1, 1]
        let shift = ada_values.narrow(1, 0, 1)?.squeeze_dim(1)?; // [B, 128, 1, 1, 1]
        let scale = ada_values.narrow(1, 1, 1)?.squeeze_dim(1)?;

        // sample = sample * (1 + scale) + shift
        let ones = Tensor::ones_dtype(
            Shape::from_dims(&[1, 1, 1, 1, 1]),
            DType::BF16,
            device.clone(),
        )?;
        sample = sample.mul(&ones.add(&scale)?)?.add(&shift)?;

        // SiLU activation
        sample = sample.silu()?;

        // conv_out: CausalConv3d(128 -> 48)
        sample = self.conv_out.forward(&sample)?;

        // Unpatchify: [B, 48, F, H, W] -> [B, 3, F, H*4, W*4]
        sample = unpatchify(&sample)?;

        Ok(sample)
    }

    /// Load from a safetensors file path.
    pub fn from_file(
        path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        eprintln!("Loading LTX-2 VAE from: {path}");
        let weights = serialization::load_file(path, device)?;

        // Normalize keys: strip optional "vae." or "video_vae." prefix
        let mut normalized: Weights = HashMap::new();
        for (key, value) in weights {
            let mut k = key.clone();
            for prefix in &["vae.", "video_vae.", "first_stage_model."] {
                if k.starts_with(prefix) {
                    k = k[prefix.len()..].to_string();
                    break;
                }
            }
            // Some exports use "resnets" instead of "res_blocks"
            if k.starts_with("decoder.") && k.contains(".resnets.") {
                k = k.replace(".resnets.", ".res_blocks.");
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
        eprintln!(
            "  Found {} decoder keys, {} statistics keys",
            decoder_count, stats_count
        );

        Self::load(&normalized)
    }
}

// ---------------------------------------------------------------------------
// Public exports
// ---------------------------------------------------------------------------

pub use self::LTX2VaeDecoder as LTX2VideoVAE;
