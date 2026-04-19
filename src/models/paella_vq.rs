//! Paella VQGAN decoder — pure Rust, used as Stage A of Stable Cascade.
//!
//! This is NOT an `AutoencoderKL`. It's a Paella-style MixingResidualBlock
//! decoder. Reference: `diffusers.pipelines.wuerstchen.modeling_paella_vq_model.PaellaVQModel`.
//!
//! ## Architecture (decoder-only; we don't encode):
//!
//! ```text
//! latent [B, 4, H, W]  (Stage B output)
//!   up_blocks.0      : Conv2d(4 -> 384, 1x1)                -> [B, 384, H, W]
//!   up_blocks.1..12  : 12 x MixingResidualBlock(c=384)
//!   up_blocks.13     : ConvTranspose2d(384->192, k=4, s=2)  -> [B, 192, 2H, 2W]
//!   up_blocks.14     : 1 x MixingResidualBlock(c=192)
//!   out_block.0      : Conv2d(192 -> 12, 1x1)
//!   out_block.1 (PixelShuffle 2)                            -> [B, 3, 4H, 4W]
//! ```
//!
//! For the 1024x1024 target: Stage B produces a latent of [1, 4, 256, 256],
//! and the decoder outputs [1, 3, 1024, 1024] (4x upsample from the VQ).
//!
//! ## MixingResidualBlock
//!
//! ```text
//! mods = gammas    # [6] learnable scalars
//! x_temp = LN(x permute NHWC).permute NCHW * (1 + mods[0]) + mods[1]
//! x     += depthwise(x_temp) * mods[2]
//! x_temp = LN(x permute NHWC).permute NCHW * (1 + mods[3]) + mods[4]
//! x     += channelwise(x_temp permute NHWC).permute NCHW * mods[5]
//! ```
//!
//! `depthwise` = `nn.Sequential(ReplicationPad2d(1), Conv2d(c, c, k=3, groups=c))`.
//! We approximate ReplicationPad2d with zero-padding (Conv2d padding=1),
//! which introduces edge artifacts but is good enough for the first end-to-end image.

use flame_core::conv::{Conv2d, Conv2dConfig};
use flame_core::ops::fused_inference::fused_linear3d_native;
use flame_core::serialization::load_file;
use flame_core::{cuda_ops_bf16, CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Default Paella VQ scale factor from config (multiply latent by this before decode).
pub const PAELLA_VQ_SCALE_FACTOR: f32 = 0.3764;

fn get<'a>(w: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
    w.get(key)
        .ok_or_else(|| Error::InvalidInput(format!("paella_vq: missing key {:?}", key)))
}

/// Convert a raw weight to BF16 if not already.
fn to_bf16(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t.clone())
    } else {
        t.to_dtype(DType::BF16)
    }
}

/// Linear: flatten leading dims, cuBLASLt matmul, restore shape. Matches
/// `wuerstchen_blocks::linear_fwd`.
fn linear_fwd(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let cin = *dims.last().unwrap();
    let cout = w.shape().dims()[0];
    let leading: usize = dims[..dims.len() - 1].iter().product();
    let x3 = x.reshape(&[1, leading, cin])?;
    let y3 = fused_linear3d_native(&x3, w, b)?;
    let mut out = dims.clone();
    *out.last_mut().unwrap() = cout;
    y3.reshape(&out)
}

/// LayerNorm on NHWC tensor over the last (channel) dim, no affine.
fn ln_nhwc(x: &Tensor) -> Result<Tensor> {
    cuda_ops_bf16::layer_norm_bf16(x, None, None, 1e-6)
}

/// One MixingResidualBlock (decoder-compatible).
pub struct MixingResidualBlock {
    pub depthwise: Conv2d,
    /// `channelwise.0.weight` [embed_dim, inp_channels], `channelwise.0.bias` [embed_dim]
    pub cw0_w: Tensor,
    pub cw0_b: Tensor,
    /// `channelwise.2.weight` [inp_channels, embed_dim], `channelwise.2.bias` [inp_channels]
    pub cw2_w: Tensor,
    pub cw2_b: Tensor,
    /// 6 learnable scalars, BF16 on GPU.
    pub gammas: Tensor, // [6]
    /// Host copy of gammas for scalar-multiplication shortcuts.
    pub gammas_host: [f32; 6],
    pub c: usize,
}

impl MixingResidualBlock {
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        c: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // Depthwise: Conv2d(c, c, k=3, padding=1, groups=c). The checkpoint stores
        // this as `depthwise.1.weight` because in PyTorch it's Sequential(ReplicationPad2d(1), Conv2d(...)).
        let dw_w = to_bf16(get(weights, &format!("{prefix}depthwise.1.weight"))?)?;
        let dw_b = to_bf16(get(weights, &format!("{prefix}depthwise.1.bias"))?)?;
        let cfg = Conv2dConfig {
            in_channels: c,
            out_channels: c,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            groups: c,
        };
        let mut depthwise = Conv2d::from_config_with_bias(cfg, device.clone(), true)?;
        depthwise.copy_weight_from(&dw_w)?;
        depthwise.copy_bias_from(&dw_b)?;

        let cw0_w = to_bf16(get(weights, &format!("{prefix}channelwise.0.weight"))?)?;
        let cw0_b = to_bf16(get(weights, &format!("{prefix}channelwise.0.bias"))?)?;
        let cw2_w = to_bf16(get(weights, &format!("{prefix}channelwise.2.weight"))?)?;
        let cw2_b = to_bf16(get(weights, &format!("{prefix}channelwise.2.bias"))?)?;

        let gammas_t = to_bf16(get(weights, &format!("{prefix}gammas"))?)?;
        let gammas_host_f = gammas_t.to_dtype(DType::F32)?.to_vec_f32()?;
        let mut gh = [0.0f32; 6];
        for (i, v) in gammas_host_f.iter().take(6).enumerate() {
            gh[i] = *v;
        }

        Ok(Self {
            depthwise,
            cw0_w,
            cw0_b,
            cw2_w,
            cw2_b,
            gammas: gammas_t,
            gammas_host: gh,
            c,
        })
    }

    /// Forward: `x`: `[B, C, H, W]` BF16 NCHW.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mods = self.gammas_host;

        // First branch: LN(x_NCHW->NHWC) * (1+mods[0]) + mods[1] -> depthwise -> add * mods[2]
        let x_nhwc = x.permute(&[0, 2, 3, 1])?;
        let ln1 = ln_nhwc(&x_nhwc)?; // NHWC
        let ln1_scaled = ln1.mul_scalar(1.0 + mods[0])?.add_scalar(mods[1])?;
        let ln1_nchw = ln1_scaled.permute(&[0, 3, 1, 2])?;
        let dw = self.depthwise.forward(&ln1_nchw)?;
        let x = x.add(&dw.mul_scalar(mods[2])?)?;

        // Second branch: LN(x) * (1+mods[3]) + mods[4] -> channelwise MLP -> add * mods[5]
        let x_nhwc2 = x.permute(&[0, 2, 3, 1])?;
        let ln2 = ln_nhwc(&x_nhwc2)?;
        let ln2_scaled = ln2.mul_scalar(1.0 + mods[3])?.add_scalar(mods[4])?;
        // MLP runs in NHWC directly (no permute back needed for linear).
        let h1 = linear_fwd(&ln2_scaled, &self.cw0_w, Some(&self.cw0_b))?;
        let h1g = cuda_ops_bf16::gelu_bf16(&h1)?;
        let h2 = linear_fwd(&h1g, &self.cw2_w, Some(&self.cw2_b))?;
        let h2_nchw = h2.permute(&[0, 3, 1, 2])?;

        x.add(&h2_nchw.mul_scalar(mods[5])?)
    }
}

/// Paella VQ decoder (decoder-only — no encoder, no vquantizer).
pub struct PaellaVQDecoder {
    pub in_conv: Conv2d,       // up_blocks.0: 4 -> 384, 1x1
    pub mrbs_384: Vec<MixingResidualBlock>, // 12 of them
    /// up_blocks.13: ConvTranspose2d(384 -> 192, k=4, s=2, p=1).
    /// Uses flame-core's real conv_transpose2d_forward (commit d99c8e9).
    pub up_conv: flame_core::upsampling::ConvTranspose2d,
    pub mrb_192: MixingResidualBlock,
    pub out_conv: Conv2d,      // out_block.0: 192 -> 12, 1x1
}

impl PaellaVQDecoder {
    pub fn load(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let raw = load_file(Path::new(path), device)?;
        // Convert everything to BF16 at load time.
        let mut w: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, v) in raw.into_iter() {
            let bf = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)?
            };
            w.insert(k, bf);
        }

        // up_blocks.0: Conv2d(4, 384, 1x1) — wrapped in nn.Sequential hence the ".0.0"
        let in_conv = {
            let ww = get(&w, "up_blocks.0.0.weight")?.clone();
            let bw = get(&w, "up_blocks.0.0.bias")?.clone();
            let cfg = Conv2dConfig {
                in_channels: 4,
                out_channels: 384,
                kernel_size: (1, 1),
                stride: (1, 1),
                padding: (0, 0),
                groups: 1,
            };
            let mut c = Conv2d::from_config_with_bias(cfg, device.clone(), true)?;
            c.copy_weight_from(&ww)?;
            c.copy_bias_from(&bw)?;
            c
        };

        // 12 MRB at c=384
        let mut mrbs_384 = Vec::with_capacity(12);
        for i in 1..=12 {
            let prefix = format!("up_blocks.{}.", i);
            mrbs_384.push(MixingResidualBlock::from_weights(&w, &prefix, 384, device)?);
        }

        // up_blocks.13: ConvTranspose2d(384 -> 192, k=4, s=2, p=1).
        // Uses the real flame-core conv_transpose2d_forward (landed in
        // flame-core d99c8e9), replacing the prior "nearest-upsample + Conv3x3
        // with averaged kernel-subblocks" surrogate that blurred the final
        // VAE decode.
        let up_conv_w = get(&w, "up_blocks.13.weight")?.clone(); // [384, 192, 4, 4]
        let up_conv_b = get(&w, "up_blocks.13.bias")?.clone();   // [192]
        let up_conv = {
            let cfg = flame_core::upsampling::ConvTranspose2dConfig {
                in_channels: 384,
                out_channels: 192,
                kernel_size: (4, 4),
                stride: (2, 2),
                padding: (1, 1),
                output_padding: (0, 0),
                groups: 1,
                bias: true,
                dilation: (1, 1),
            };
            let mut ct = flame_core::upsampling::ConvTranspose2d::new(cfg, device.clone())?;
            // Weight layout for ConvTranspose2d matches the checkpoint: [in, out, kh, kw].
            ct.weight = up_conv_w;
            ct.bias = Some(up_conv_b);
            ct
        };

        // up_blocks.14: 1 MRB at c=192
        let mrb_192 = MixingResidualBlock::from_weights(&w, "up_blocks.14.", 192, device)?;

        // out_block.0: Conv2d(192 -> 12, 1x1)
        let out_conv = {
            let ww = get(&w, "out_block.0.weight")?.clone();
            let bw = get(&w, "out_block.0.bias")?.clone();
            let cfg = Conv2dConfig {
                in_channels: 192,
                out_channels: 12,
                kernel_size: (1, 1),
                stride: (1, 1),
                padding: (0, 0),
                groups: 1,
            };
            let mut c = Conv2d::from_config_with_bias(cfg, device.clone(), true)?;
            c.copy_weight_from(&ww)?;
            c.copy_bias_from(&bw)?;
            c
        };

        Ok(Self {
            in_conv,
            mrbs_384,
            up_conv,
            mrb_192,
            out_conv,
        })
    }

    /// Decode a latent `[B, 4, H, W]` into RGB `[B, 3, 4H, 4W]` (roughly in [-1,1]).
    ///
    /// Stable Cascade's decode divides the latent by the scale factor first.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // 1. Pre-scale: multiply by scale_factor (NOT divide). Diffusers:
        //    `latents = scale_factor * latents` then `vqgan.decode(latents)`.
        let x = latent.mul_scalar(PAELLA_VQ_SCALE_FACTOR)?;

        // 2. in_conv
        let mut x = self.in_conv.forward(&x)?;

        // 3. 12 MRB at c=384
        for b in self.mrbs_384.iter() {
            x = b.forward(&x)?;
        }

        // 4. ConvTranspose2d(384 -> 192, k=4, s=2, p=1) — real op (flame-core d99c8e9).
        x = self.up_conv.forward(&x)?;

        // 5. 1 MRB at c=192
        x = self.mrb_192.forward(&x)?;

        // 6. out_conv: [B, 12, H, W]
        x = self.out_conv.forward(&x)?;

        // 7. PixelShuffle(2): [B, 12, h, w] -> [B, 3, 2h, 2w]
        //    Standard PyTorch pixel_shuffle: reshape to [B, 3, 2, 2, h, w]
        //    then permute to [B, 3, h, 2, w, 2] then reshape to [B, 3, 2h, 2w].
        let dims = x.shape().dims().to_vec();
        let (b, cc, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let upscale = 2;
        debug_assert_eq!(cc, 3 * upscale * upscale);
        let x = x
            .reshape(&[b, 3, upscale, upscale, h, w])?
            .permute(&[0, 1, 4, 2, 5, 3])?
            .reshape(&[b, 3, h * upscale, w * upscale])?;

        Ok(x)
    }
}
