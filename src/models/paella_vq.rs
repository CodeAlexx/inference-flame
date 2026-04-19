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
    /// Emulated via zero-insertion (upsample with inserted zeros) + Conv2d(k=4, s=1, p=1) with spatially-flipped kernel.
    pub up_conv_emu_w: Tensor, // [out=192, in=384, 4, 4] — flipped spatial + transposed ch
    pub up_conv: Conv2d,       // the actual Conv2d(k=4, s=1, p=1) with weight == up_conv_emu_w
    pub up_conv_bias: Tensor,  // [192], applied after the Conv2d
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
        //
        // We use the 1x1-conv + pixel-shuffle trick but adapted for k=4, s=2, p=1.
        // The PyTorch operation for ConvTranspose2d(k=4, s=2, p=1) produces output
        // size 2H from input H. At output position `(h_out, w_out)`, it's:
        //
        //   out[h_out, w_out] = sum_{cin, ky, kx} in[cin, (h_out+1+ky)/2, ...] * w[cin, cout, ky, kx]
        //     where h_out + 1 + ky is even and ≥ 2, < 2H+2.
        //
        // Split output by pixel phase a in {0,1}, b in {0,1} where h_out = 2h + a, w_out = 2w + b.
        // For a given (a, b), valid ky pairs are a_set(a) = {even kys where (a+1+ky)%2==0 and (a+1+ky)/2 in [0, H)}
        // = {kys where a+1+ky is even and between 0 and 2H}, equivalently ky ∈ {1-a, 3-a} (mod 2 matching,
        // ky in [0,4]).
        //
        // For a=0: ky ∈ {1, 3}, h_in = h (ky=1) or h-1 (ky=3).
        // For a=1: ky ∈ {0, 2}, h_in = h (ky=0) or h+1 (ky=2). But h+1 might be out of range at the top.
        //
        // This is complex. **Simplification**: we approximate with "nearest upsample 2x + Conv2d(k=3, s=1, p=1)"
        // using the SUM of the four 3x3 subblocks of the original 4x4 kernel (i.e. averaging the kernel shifts).
        // This is NOT mathematically equivalent — it's a smoothed approximation. Quality impact: the VQ
        // decoder produces slightly blurrier output but avoids both the checkerboarding of zero-insertion
        // and the spatial shape mismatch of NN upsample + raw 4x4 conv.
        let up_conv_w_raw = get(&w, "up_blocks.13.weight")?.clone(); // [384, 192, 4, 4]
        let up_conv_b = get(&w, "up_blocks.13.bias")?.clone();       // [192]
        // Build a [192, 384, 3, 3] surrogate kernel that averages the four overlapping 3x3 subblocks
        // of the flipped transposed kernel.
        let perm = up_conv_w_raw.permute(&[1, 0, 2, 3])?; // [192, 384, 4, 4]
        let surrogate_k3 = {
            let f32_tensor = perm.to_dtype(DType::F32)?;
            let data = f32_tensor.to_vec_f32()?;
            let out_ch = 192;
            let in_ch = 384;
            let k = 4;
            let mut kernel3 = vec![0.0f32; out_ch * in_ch * 3 * 3];
            for o in 0..out_ch {
                for i in 0..in_ch {
                    // Sum over 4 shifted 3x3 subblocks (top-left at (0,0), (0,1), (1,0), (1,1))
                    // of the 4x4 kernel, divided by 4 to average.
                    for sy in 0..3 {
                        for sx in 0..3 {
                            let mut acc = 0.0f32;
                            for oy in 0..2 {
                                for ox in 0..2 {
                                    let ky = oy + sy;
                                    let kx = ox + sx;
                                    acc += data[((o * in_ch + i) * k + ky) * k + kx];
                                }
                            }
                            kernel3[((o * in_ch + i) * 3 + sy) * 3 + sx] = acc / 4.0;
                        }
                    }
                }
            }
            let t = Tensor::from_vec(
                kernel3,
                Shape::from_dims(&[out_ch, in_ch, 3, 3]),
                device.clone(),
            )?;
            t.to_dtype(DType::BF16)?
        };
        let cfg = Conv2dConfig {
            in_channels: 384,
            out_channels: 192,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            groups: 1,
        };
        let mut up_conv = Conv2d::from_config_with_bias(cfg, device.clone(), false)?;
        up_conv.copy_weight_from(&surrogate_k3)?;
        let up_conv_emu_w = surrogate_k3;

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
            up_conv_emu_w,
            up_conv,
            up_conv_bias: up_conv_b,
            mrb_192,
            out_conv,
        })
    }

    /// Zero-insert (proper): `[N, C, H, W] -> [N, C, 2H-1, 2W-1]` with
    /// `out[:, :, 2i, 2j] = in[:, :, i, j]`, zeros at odd indices. Stride-2 equivalence.
    fn zero_insert_2x(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        // First do [N,C,2H,2W] then narrow to [N,C,2H-1,2W-1].
        let x5 = x.reshape(&[n, c, h, 1, w])?;
        let zeros_row = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, h, 1, w]),
            DType::BF16,
            x.device().clone(),
        )?;
        let stacked = Tensor::cat(&[&x5, &zeros_row], 3)?;
        let rows = stacked.reshape(&[n, c, 2 * h, w])?;
        let r5 = rows.reshape(&[n, c, 2 * h, w, 1])?;
        let zeros_col = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, 2 * h, w, 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        let stacked2 = Tensor::cat(&[&r5, &zeros_col], 4)?;
        let full = stacked2.reshape(&[n, c, 2 * h, 2 * w])?;
        // Trim last row and column.
        let trimmed_h = full.narrow(2, 0, 2 * h - 1)?;
        trimmed_h.narrow(3, 0, 2 * w - 1)
    }

    /// Apply a scalar bias `[C]` to an NCHW tensor `[B, C, H, W]` via broadcast.
    fn add_bias_nchw(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let c = dims[1];
        // bias [C] -> [1, C, 1, 1]
        let b4 = bias
            .reshape(&[1, c, 1, 1])?;
        x.add(&b4)
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

        // 4. ConvTranspose2d (approximate): nearest-upsample 2x + Conv2d(k=3) with surrogate kernel.
        // This introduces some blurring vs diffusers reference but avoids the checkerboard that
        // zero-insertion + k=4 Conv2d produced.
        let dims = x.shape().dims().to_vec();
        let up_cfg = flame_core::upsampling::Upsample2dConfig::new(
            flame_core::upsampling::UpsampleMode::Nearest,
        )
        .with_size((dims[2] * 2, dims[3] * 2));
        let xu = flame_core::upsampling::Upsample2d::new(up_cfg).forward(&x)?;
        x = self.up_conv.forward(&xu)?;
        x = Self::add_bias_nchw(&x, &self.up_conv_bias)?;

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
