//! Stable Cascade UNet (Stage B + Stage C) — pure Rust, flame-core.
//!
//! Shares the five building blocks from `wuerstchen_blocks.rs`:
//! - `SDCascadeResBlock`, `SDCascadeTimestepBlock`, `SDCascadeAttnBlock`
//! - `CascadeChannelLayerNorm`
//!
//! Reference: `diffusers.models.unets.unet_stable_cascade.StableCascadeUNet`.
//!
//! ## Block-sequence per level (observed from checkpoints)
//!
//! Stage C (`stage_c_bf16.safetensors`):
//! - `block_out_channels = [2048, 2048]`
//! - `down_num_layers_per_block = [8, 24]`, `up_num_layers_per_block = [24, 8]`
//! - `num_attention_heads = [32, 32]`
//! - `timestep_conditioning_type = ["sca", "crp"]` — 3 mappers per TSBlock
//! - Block pattern per layer: `[Res, TS, Attn]`
//! - Downscaler (`down_downscalers.1`): UpDownBlock2d-down (bilinear 0.5 + 1x1 conv)
//! - Upscaler  (`up_upscalers.0`):      UpDownBlock2d-up   (bilinear 2.0 + 1x1 conv)
//!
//! Stage B (`stage_b_bf16.safetensors`):
//! - `block_out_channels = [320, 640, 1280, 1280]`
//! - `down_num_layers_per_block = [2, 6, 28, 6]`, `up_num_layers_per_block = [6, 28, 6, 2]`
//! - `num_attention_heads = [-, -, 20, 20]` (only levels 2/3 have attention)
//! - `timestep_conditioning_type = ["sca"]` — 2 mappers per TSBlock
//! - Block pattern per layer: `[Res, TS]` for levels 0/1; `[Res, TS, Attn]` for levels 2/3
//! - Downscaler: plain `Conv2d(k=2, s=2)` (no LN-wrapper required for weight loading)
//! - Upscaler: plain `ConvTranspose2d(k=2, s=2)`
//! - Has `effnet_mapper` (16→1280→320) and `pixels_mapper` (3→1280→320) conditioning inputs
//! - Has `up_repeat_mappers` (1x1 conv) at certain levels — 2 at levels 0/1 of up_blocks, 1 at 2/3
//!
//! ## Conditioning for `forward`
//!
//! Stage C forward expects:
//! - `sample`:            [B, 16, H, W]
//! - `timestep_ratio`:    scalar `r in [0, 1]` (broadcast)
//! - `clip_text_pooled`:  [B, 1280] — from CLIP-G `text_projection`
//! - `clip_text`:         [B, 77, 1280] — CLIP-G last_hidden_state (optional)
//! - `clip_image`:        None (text-only inference)
//! - `sca`, `crp`:        None (zeros)
//!
//! Stage B forward expects:
//! - `sample`:           [B, 4, H, W]
//! - `timestep_ratio`:   scalar
//! - `clip_text` alone:  [B, 77, 1280] — note Stage B uses `clip_mapper(clip_txt)` and *does not*
//!   use the pooled vector as conditioning; it uses `effnet` (Stage C latent) instead.
//! - `effnet`:           Stage C output [B, 16, h_c, w_c] — upsampled inside the UNet
//! - `pixels`:           zeros [B, 3, 8, 8]
//! - `sca`:              None (zeros)

use crate::models::wuerstchen_blocks::{
    CascadeChannelLayerNorm, SDCascadeAttnBlock, SDCascadeResBlock, SDCascadeTimestepBlock,
};
use flame_core::conv::{Conv2d, Conv2dConfig};
use flame_core::ops::fused_inference::fused_linear3d_native;
use flame_core::serialization::load_file;
use flame_core::upsampling::{Upsample2d, Upsample2dConfig, UpsampleMode};
use flame_core::{cuda_ops_bf16, CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Block-type sum
// ---------------------------------------------------------------------------

pub enum Block {
    Res(SDCascadeResBlock),
    TS(SDCascadeTimestepBlock),
    Attn(SDCascadeAttnBlock),
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WuerstchenUNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    /// Patch size: Stage C uses 1, Stage B uses 2 (PixelUnshuffle at in, PixelShuffle at out).
    pub patch_size: usize,
    /// Timestep ratio embedding dim (sinusoidal). Always 64 in Cascade.
    pub timestep_ratio_embedding_dim: usize,
    pub block_out_channels: Vec<usize>,
    /// Per-level `[Res, TS]` or `[Res, TS, Attn]`.
    pub block_pattern: Vec<Vec<char>>, // 'R','T','A'
    pub down_num_layers_per_block: Vec<usize>,
    pub up_num_layers_per_block: Vec<usize>,
    pub num_attention_heads: Vec<usize>,
    /// Conditioning conds list for TSBlocks — ("sca","crp") or ("sca",).
    pub timestep_conditioning_type: Vec<String>,
    /// Shared dim for all attention kv-mapping.
    pub conditioning_dim: usize,
    /// Pooled CLIP text embed dim (always 1280).
    pub clip_text_pooled_in_channels: usize,
    /// clip_text hidden dim (1280 for CLIP-G).
    pub clip_text_in_channels: Option<usize>,
    pub clip_image_in_channels: Option<usize>,
    pub clip_seq: usize,
    /// Optional effnet conditioning (Stage B takes Stage C latent here).
    pub effnet_in_channels: Option<usize>,
    /// Optional pixels conditioning (Stage B uses zeros).
    pub pixels_mapper_in_channels: Option<usize>,
    /// If Some, use the Stage-B style single `clip_mapper` that maps
    /// CLIP last_hidden to [clip_seq * c_cond] before attention.
    /// If None, use the Stage-C style with `clip_txt_pooled_mapper` +
    /// `clip_txt_mapper` (+ optional `clip_img_mapper`).
    pub use_clip_mapper: bool,
    /// Downscaler style per level (index 0 is skipped).
    pub downscaler_style: DownscalerStyle,
    /// Upscaler style per level.
    pub upscaler_style: UpscalerStyle,
    /// Whether `up_repeat_mappers` is present — Stage B yes, Stage C no.
    pub has_up_repeat_mappers: bool,
    pub up_repeat_mapper_counts: Vec<usize>, // per-level count of 1x1 convs in `up_repeat_mappers.L.*`
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DownscalerStyle {
    /// Stage B: `Conv2d(k=2, s=2)`. Key: `down_downscalers.L.1.weight/bias`
    Conv2D,
    /// Stage C: UpDownBlock2d (optionally bilinear 0.5) + 1x1 conv.
    /// The `switch_level` config flag gates the bilinear half-resolution
    /// interpolation. In the official Stable Cascade prior checkpoint
    /// `switch_level = [False]`, so the interp block is `nn.Identity()`:
    /// the downscaler is effectively just a 1x1 channel-mapping conv
    /// wrapped in a channels-first LN.
    /// Key (conv): `down_downscalers.L.1.blocks.0.weight/bias`
    BilinearPlusConv1x1 { do_interp: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpscalerStyle {
    /// Stage B: `ConvTranspose2d(k=2, s=2)`. Key: `up_upscalers.L.1.weight/bias`
    ConvTranspose2D,
    /// Stage C: UpDownBlock2d (optionally bilinear 2.0) + 1x1 conv.
    /// Same gating rule as `DownscalerStyle::BilinearPlusConv1x1`.
    /// Key (conv): `up_upscalers.L.1.blocks.1.weight/bias`
    BilinearPlusConv1x1 { do_interp: bool },
}

impl WuerstchenUNetConfig {
    pub fn stage_c() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            patch_size: 1,
            timestep_ratio_embedding_dim: 64,
            block_out_channels: vec![2048, 2048],
            block_pattern: vec![
                vec!['R', 'T', 'A'],
                vec!['R', 'T', 'A'],
            ],
            down_num_layers_per_block: vec![8, 24],
            up_num_layers_per_block: vec![24, 8],
            num_attention_heads: vec![32, 32],
            timestep_conditioning_type: vec!["sca".into(), "crp".into()],
            conditioning_dim: 2048,
            clip_text_pooled_in_channels: 1280,
            clip_text_in_channels: Some(1280),
            clip_image_in_channels: Some(768), // unused at inference (no img cond)
            clip_seq: 4,
            effnet_in_channels: None,
            pixels_mapper_in_channels: None,
            use_clip_mapper: false,
            // Stage C checkpoint ships with `switch_level = [False]` → the
            // bilinear interp blocks inside down/up-scalers are `Identity()`.
            // The 1x1 mapping conv is still applied. Net effect: Stage C
            // stays at the same spatial resolution (24×24 at 1024px) across
            // both levels.
            downscaler_style: DownscalerStyle::BilinearPlusConv1x1 { do_interp: false },
            upscaler_style: UpscalerStyle::BilinearPlusConv1x1 { do_interp: false },
            has_up_repeat_mappers: false,
            up_repeat_mapper_counts: vec![0, 0],
        }
    }

    pub fn stage_b() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            patch_size: 2,
            timestep_ratio_embedding_dim: 64,
            block_out_channels: vec![320, 640, 1280, 1280],
            block_pattern: vec![
                vec!['R', 'T'],
                vec!['R', 'T'],
                vec!['R', 'T', 'A'],
                vec!['R', 'T', 'A'],
            ],
            // down order: shallow → deep; up order: deep → shallow.
            down_num_layers_per_block: vec![2, 6, 28, 6],
            up_num_layers_per_block: vec![6, 28, 6, 2],
            num_attention_heads: vec![1, 1, 20, 20], // first two unused
            timestep_conditioning_type: vec!["sca".into()],
            conditioning_dim: 1280,
            clip_text_pooled_in_channels: 1280,
            clip_text_in_channels: Some(1280),
            clip_image_in_channels: None,
            clip_seq: 4,
            effnet_in_channels: Some(16),
            pixels_mapper_in_channels: Some(3),
            use_clip_mapper: true,
            downscaler_style: DownscalerStyle::Conv2D,
            upscaler_style: UpscalerStyle::ConvTranspose2D,
            has_up_repeat_mappers: true,
            // From the checkpoint: level 0/1 have 2 each, level 2/3 have 1 each.
            up_repeat_mapper_counts: vec![2, 2, 1, 1],
        }
    }
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

fn to_bf16(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t.clone())
    } else {
        t.to_dtype(DType::BF16)
    }
}

fn get<'a>(w: &'a HashMap<String, Tensor>, k: &str) -> Result<&'a Tensor> {
    w.get(k).ok_or_else(|| {
        Error::InvalidInput(format!("wuerstchen_unet: missing key {:?}", k))
    })
}

/// PyTorch-style PixelUnshuffle(r): [N, C, H, W] → [N, C*r*r, H/r, W/r].
/// Implemented as two reshapes + one permute to land contiguous.
fn pixel_unshuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if h % r != 0 || w % r != 0 {
        return Err(Error::InvalidInput(format!(
            "pixel_unshuffle: H={h}, W={w} not divisible by r={r}"
        )));
    }
    // [N, C, H/r, r, W/r, r] -> [N, C, r, r, H/r, W/r] -> [N, C*r*r, H/r, W/r]
    let a = x.reshape(&[n, c, h / r, r, w / r, r])?;
    let b = a.permute(&[0, 1, 3, 5, 2, 4])?;
    b.reshape(&[n, c * r * r, h / r, w / r])
}

/// PyTorch-style PixelShuffle(r): [N, C*r*r, H, W] → [N, C, H*r, W*r].
fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (n, cc, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let r2 = r * r;
    if cc % r2 != 0 {
        return Err(Error::InvalidInput(format!(
            "pixel_shuffle: C={cc} not divisible by r^2={r2}"
        )));
    }
    let c = cc / r2;
    // [N, C, r, r, H, W] -> [N, C, H, r, W, r] -> [N, C, H*r, W*r]
    let a = x.reshape(&[n, c, r, r, h, w])?;
    let b = a.permute(&[0, 1, 4, 2, 5, 3])?;
    b.reshape(&[n, c, h * r, w * r])
}

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

/// Conv2d 1x1 with bias, loaded directly.
fn make_conv1x1(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    device: &Arc<CudaDevice>,
) -> Result<Conv2d> {
    let w = to_bf16(get(weights, &format!("{prefix}weight"))?)?;
    let b = to_bf16(get(weights, &format!("{prefix}bias"))?)?;
    let cfg = Conv2dConfig {
        in_channels: in_ch,
        out_channels: out_ch,
        kernel_size: (1, 1),
        stride: (1, 1),
        padding: (0, 0),
        groups: 1,
    };
    let mut c = Conv2d::from_config_with_bias(cfg, device.clone(), true)?;
    c.copy_weight_from(&w)?;
    c.copy_bias_from(&b)?;
    Ok(c)
}

/// Sinusoidal timestep ratio embedding with dim/2 sin + dim/2 cos.
fn timestep_ratio_embedding(r: f32, dim: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let max_positions = 10000.0f32;
    let r_scaled = r * max_positions;
    let half = dim / 2;
    // emb[i] = exp(-(ln max / (half-1)) * i)
    let log_max = (max_positions).ln();
    let mut data = vec![0f32; dim];
    let denom = (half - 1).max(1) as f32;
    for i in 0..half {
        let freq = (-(log_max / denom) * i as f32).exp();
        let ang = r_scaled * freq;
        data[i] = ang.sin();
        data[i + half] = ang.cos();
    }
    // Return as BF16 [1, dim].
    let t = Tensor::from_vec(data, Shape::from_dims(&[1, dim]), device.clone())?;
    t.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// Downscaler / Upscaler abstractions
// ---------------------------------------------------------------------------

pub enum Downscaler {
    /// Stage B style: Conv2d(k=2, s=2).
    Conv2D(Conv2d),
    /// Stage C style: no-affine LN(C) → UpDownBlock2d(mode="down").
    ///
    /// `UpDownBlock2d` for mode="down" applies `[conv1x1, interpolation]`
    /// (conv first, then interp — reversed from mode="up"!). When
    /// `switch_level[i] == False` the interpolation is `nn.Identity`, so
    /// `do_interp=false` skips the 0.5x resize entirely. Stage C's shipped
    /// checkpoint uses `switch_level=[False]`: no spatial change, just a
    /// channel-mapping 1x1 conv.
    BilinearPlusConv1x1 { conv: Conv2d, do_interp: bool },
}

pub enum Upscaler {
    /// ConvTranspose2d(k=2, s=2, p=0) emulated via a 1x1 Conv2d + PixelShuffle(2).
    /// Stored as a 1x1 Conv2d with C_out = orig_out_channels * 4 (the 2x2 kernel
    /// positions are flattened into the output-channel axis of the 1x1 conv).
    /// Bias is the original bias (applied AFTER pixel-shuffle, broadcast over out_ch).
    ConvTranspose2x2 { conv: Conv2d, bias: Tensor, out_ch: usize },
    /// Stage C style: no-affine LN(C) → UpDownBlock2d(mode="up").
    ///
    /// For mode="up" the reference applies `[interpolation, conv1x1]`
    /// (interp first, then conv). When `do_interp=false` the interp is
    /// skipped (Stage C's `switch_level=[False]` disables it).
    BilinearPlusConv1x1 { conv: Conv2d, do_interp: bool },
}

// ---------------------------------------------------------------------------
// UNet
// ---------------------------------------------------------------------------

pub struct WuerstchenUNet {
    pub config: WuerstchenUNetConfig,

    // Input path.
    pub embedding_conv: Conv2d,
    pub embedding_ln_c: usize, // channel count for no-affine LN

    // Conditioning mappers.
    /// Stage C: Linear(1280 -> 2048*4). Present only for Stage C.
    pub clip_txt_pooled_mapper: Option<(Tensor, Tensor)>, // (weight, bias)
    /// Stage C: Linear(1280 -> 2048). Present only if use_clip_mapper == false.
    pub clip_txt_mapper: Option<(Tensor, Tensor)>,
    /// Stage C: Linear(768 -> 2048*4). We keep it loaded for completeness but do not use it.
    pub clip_img_mapper: Option<(Tensor, Tensor)>,
    /// Stage B: Linear(1280 -> 1280*4). If present, used instead of the pooled mapper.
    pub clip_mapper: Option<(Tensor, Tensor)>,
    /// Stage B effnet_mapper sequence (16 -> 1280 -> 320).
    pub effnet_mapper: Option<EffnetMapper>,
    /// Stage B pixels_mapper sequence (3 -> 1280 -> 320).
    pub pixels_mapper: Option<EffnetMapper>,

    // Body.
    pub down_downscalers: Vec<Option<Downscaler>>, // level 0 is always None
    pub down_blocks: Vec<Vec<Block>>,              // flat list per level
    pub up_blocks: Vec<Vec<Block>>,
    pub up_upscalers: Vec<Option<Upscaler>>,       // last level None
    pub up_repeat_mappers: Vec<Vec<Conv2d>>,       // len = num_levels; inner may be empty

    // Output.
    pub clf_conv: Conv2d,
    pub clf_ln_c: usize,
}

/// Stage-B effnet_mapper / pixels_mapper:
///   Conv2d(in, 4*c_out, 1x1) -> GELU -> Conv2d(4*c_out, c_out, 1x1) -> LN no-affine over channels.
pub struct EffnetMapper {
    pub conv0: Conv2d,
    pub conv2: Conv2d,
    pub c_out: usize,
}

impl EffnetMapper {
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        hidden: usize,
        out_ch: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let conv0 = make_conv1x1(weights, &format!("{prefix}0."), in_ch, hidden, device)?;
        let conv2 = make_conv1x1(weights, &format!("{prefix}2."), hidden, out_ch, device)?;
        Ok(Self { conv0, conv2, c_out: out_ch })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv0.forward(x)?;
        let hg = cuda_ops_bf16::gelu_bf16(&h)?;
        let y = self.conv2.forward(&hg)?;
        // LN no-affine over channel axis.
        let ln = CascadeChannelLayerNorm::no_affine(self.c_out, 1e-6);
        ln.forward(&y)
    }
}

impl WuerstchenUNet {
    pub fn load(
        path: &str,
        config: WuerstchenUNetConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // Load all weights as BF16.
        let raw = load_file(Path::new(path), device)?;
        let mut w: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let bf = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
            w.insert(k, bf);
        }

        let c0 = config.block_out_channels[0];
        // embedding.1: Conv2d(in_channels * patch_size^2, c0, 1x1)
        let emb_in = config.in_channels * config.patch_size * config.patch_size;
        let embedding_conv = make_conv1x1(&w, "embedding.1.", emb_in, c0, device)?;

        let clip_txt_pooled_mapper = if w.contains_key("clip_txt_pooled_mapper.weight") {
            let we = to_bf16(get(&w, "clip_txt_pooled_mapper.weight")?)?;
            let bi = to_bf16(get(&w, "clip_txt_pooled_mapper.bias")?)?;
            Some((we, bi))
        } else {
            None
        };
        let clip_txt_mapper = if !config.use_clip_mapper && w.contains_key("clip_txt_mapper.weight") {
            let we = to_bf16(get(&w, "clip_txt_mapper.weight")?)?;
            let bi = to_bf16(get(&w, "clip_txt_mapper.bias")?)?;
            Some((we, bi))
        } else {
            None
        };
        let clip_img_mapper = if w.contains_key("clip_img_mapper.weight") {
            let we = to_bf16(get(&w, "clip_img_mapper.weight")?)?;
            let bi = to_bf16(get(&w, "clip_img_mapper.bias")?)?;
            Some((we, bi))
        } else {
            None
        };
        let clip_mapper = if config.use_clip_mapper && w.contains_key("clip_mapper.weight") {
            let we = to_bf16(get(&w, "clip_mapper.weight")?)?;
            let bi = to_bf16(get(&w, "clip_mapper.bias")?)?;
            Some((we, bi))
        } else {
            None
        };

        let effnet_mapper = if let Some(in_ch) = config.effnet_in_channels {
            Some(EffnetMapper::from_weights(&w, "effnet_mapper.", in_ch, c0 * 4, c0, device)?)
        } else {
            None
        };
        let pixels_mapper = if let Some(in_ch) = config.pixels_mapper_in_channels {
            Some(EffnetMapper::from_weights(&w, "pixels_mapper.", in_ch, c0 * 4, c0, device)?)
        } else {
            None
        };

        let num_levels = config.block_out_channels.len();

        // Downscalers: level 0 is Identity, level 1..N-1 are the style-specific op.
        let mut down_downscalers: Vec<Option<Downscaler>> = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            if level == 0 {
                down_downscalers.push(None);
            } else {
                let in_ch = config.block_out_channels[level - 1];
                let out_ch = config.block_out_channels[level];
                let ds = match config.downscaler_style {
                    DownscalerStyle::Conv2D => {
                        let w_ = to_bf16(get(&w, &format!("down_downscalers.{level}.1.weight"))?)?;
                        let b_ = to_bf16(get(&w, &format!("down_downscalers.{level}.1.bias"))?)?;
                        let cfg = Conv2dConfig {
                            in_channels: in_ch,
                            out_channels: out_ch,
                            kernel_size: (2, 2),
                            stride: (2, 2),
                            padding: (0, 0),
                            groups: 1,
                        };
                        let mut c = Conv2d::from_config_with_bias(cfg, device.clone(), true)?;
                        c.copy_weight_from(&w_)?;
                        c.copy_bias_from(&b_)?;
                        Downscaler::Conv2D(c)
                    }
                    DownscalerStyle::BilinearPlusConv1x1 { do_interp } => {
                        // Key: down_downscalers.{level}.1.blocks.0.weight/bias — 1x1 Conv2d(in, out)
                        let c = make_conv1x1(
                            &w,
                            &format!("down_downscalers.{level}.1.blocks.0."),
                            in_ch,
                            out_ch,
                            device,
                        )?;
                        Downscaler::BilinearPlusConv1x1 { conv: c, do_interp }
                    }
                };
                down_downscalers.push(Some(ds));
            }
        }

        // Upscalers: there are num_levels entries, but the LAST processed level has no upscaler
        // (we iterate up_blocks from highest-index level down to 0). Key indices in checkpoint:
        //   Stage C: up_upscalers.0.* (between up level 0 processing and level 1)
        //   Stage B: up_upscalers.0.*, up_upscalers.1.*, up_upscalers.2.* (3 entries for 4 levels)
        let mut up_upscalers: Vec<Option<Upscaler>> = Vec::with_capacity(num_levels);
        // up_upscalers has `num_levels - 1` entries. We store them indexed 0..num_levels-1
        // and push a final None (the last level has no upscaler).
        for k in 0..num_levels {
            if k < num_levels - 1 {
                // Between the k-th up-level processing and the (k+1)-th: upscaler
                // goes from channels at up-level k (reversed block index) to the
                // next resolution. up-level k maps to reversed block index (num_levels-1 - k).
                // After processing that level, we upscale to the previous level's channels.
                let rev_k = num_levels - 1 - k;
                let in_ch = config.block_out_channels[rev_k];
                let out_ch = config.block_out_channels[rev_k - 1];
                let us = match config.upscaler_style {
                    UpscalerStyle::ConvTranspose2D => {
                        let w_raw = to_bf16(get(&w, &format!("up_upscalers.{k}.1.weight"))?)?;
                        let b_ = to_bf16(get(&w, &format!("up_upscalers.{k}.1.bias"))?)?;
                        // Reinterpret ConvTranspose2d(in, out, k=2, s=2, p=0) as
                        // Conv2d(in, out*4, k=1) + PixelShuffle(2).
                        //
                        // Weight layout reindex:
                        //   ct_weight[in, out, ky, kx] -> conv1x1_weight[out*4 + ky*2 + kx, in, 1, 1]
                        let r = 2usize;
                        // ct_weight shape: [in_ch, out_ch, r, r]
                        // Permute to [out_ch, r, r, in_ch] -> reshape [out_ch*r*r, in_ch, 1, 1]
                        let ct_dims = w_raw.shape().dims();
                        debug_assert_eq!(ct_dims, &[in_ch, out_ch, r, r]);
                        let permuted = w_raw.permute(&[1, 2, 3, 0])?; // [out, r, r, in]
                        let conv_w =
                            permuted.reshape(&[out_ch * r * r, in_ch, 1, 1])?;
                        let cfg = Conv2dConfig {
                            in_channels: in_ch,
                            out_channels: out_ch * r * r,
                            kernel_size: (1, 1),
                            stride: (1, 1),
                            padding: (0, 0),
                            groups: 1,
                        };
                        let mut c = Conv2d::from_config_with_bias(cfg, device.clone(), false)?;
                        c.copy_weight_from(&conv_w)?;
                        Upscaler::ConvTranspose2x2 {
                            conv: c,
                            bias: b_,
                            out_ch,
                        }
                    }
                    UpscalerStyle::BilinearPlusConv1x1 { do_interp } => {
                        let c = make_conv1x1(
                            &w,
                            &format!("up_upscalers.{k}.1.blocks.1."),
                            in_ch,
                            out_ch,
                            device,
                        )?;
                        Upscaler::BilinearPlusConv1x1 { conv: c, do_interp }
                    }
                };
                up_upscalers.push(Some(us));
            } else {
                up_upscalers.push(None);
            }
        }

        // Build down/up blocks per level.
        let mut down_blocks: Vec<Vec<Block>> = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let block_count =
                config.down_num_layers_per_block[level] * config.block_pattern[level].len();
            let mut lvl = Vec::with_capacity(block_count);
            let c = config.block_out_channels[level];
            let n_heads = config.num_attention_heads[level];
            for i in 0..block_count {
                let type_ch = config.block_pattern[level][i % config.block_pattern[level].len()];
                let prefix = format!("down_blocks.{level}.{i}.");
                let blk = load_block(
                    type_ch,
                    &w,
                    &prefix,
                    c,
                    /*c_skip*/ 0,
                    /*c_cond*/ config.conditioning_dim,
                    /*n_heads*/ n_heads,
                    config.timestep_ratio_embedding_dim,
                    &config.timestep_conditioning_type,
                    device,
                )?;
                lvl.push(blk);
            }
            down_blocks.push(lvl);
        }

        let mut up_blocks: Vec<Vec<Block>> = Vec::with_capacity(num_levels);
        for up_idx in 0..num_levels {
            // up_blocks[up_idx] corresponds to processing the reversed level:
            //   reversed_level = num_levels - 1 - up_idx
            // The channel count / block pattern / n_heads all come from the REVERSED level,
            // but the `up_num_layers_per_block` vector is indexed in up-traversal order
            // (first element = first up level = deepest = highest channel).
            let rev_level = num_levels - 1 - up_idx;
            let block_count =
                config.up_num_layers_per_block[up_idx] * config.block_pattern[rev_level].len();
            let mut lvl = Vec::with_capacity(block_count);
            let c = config.block_out_channels[rev_level];
            let n_heads = config.num_attention_heads[rev_level];
            for i in 0..block_count {
                let pat = &config.block_pattern[rev_level];
                let type_ch = pat[i % pat.len()];
                // The first ResBlock at the start of this up-level (i.e. i==0) has a
                // concat skip if we're not at the deepest level (up_idx==0). c_skip == c.
                let c_skip = if type_ch == 'R' && i == 0 && up_idx > 0 {
                    c
                } else {
                    0
                };
                let prefix = format!("up_blocks.{up_idx}.{i}.");
                let blk = load_block(
                    type_ch,
                    &w,
                    &prefix,
                    c,
                    c_skip,
                    config.conditioning_dim,
                    n_heads,
                    config.timestep_ratio_embedding_dim,
                    &config.timestep_conditioning_type,
                    device,
                )?;
                lvl.push(blk);
            }
            up_blocks.push(lvl);
        }

        // up_repeat_mappers: only Stage B.
        let mut up_repeat_mappers: Vec<Vec<Conv2d>> = Vec::with_capacity(num_levels);
        if config.has_up_repeat_mappers {
            for up_idx in 0..num_levels {
                let rev_level = num_levels - 1 - up_idx;
                let c = config.block_out_channels[rev_level];
                let count = config.up_repeat_mapper_counts[up_idx];
                let mut v = Vec::with_capacity(count);
                for k in 0..count {
                    let conv = make_conv1x1(
                        &w,
                        &format!("up_repeat_mappers.{up_idx}.{k}."),
                        c,
                        c,
                        device,
                    )?;
                    v.push(conv);
                }
                up_repeat_mappers.push(v);
            }
        } else {
            for _ in 0..num_levels {
                up_repeat_mappers.push(Vec::new());
            }
        }

        // clf.1: Conv2d(c0, out_channels * patch_size^2, 1x1)
        let clf_out = config.out_channels * config.patch_size * config.patch_size;
        let clf_conv = make_conv1x1(&w, "clf.1.", c0, clf_out, device)?;

        // Drop raw weight cache now that it's been consumed.
        let _ = w;

        Ok(Self {
            embedding_ln_c: c0,
            clf_ln_c: c0,
            config,
            embedding_conv,
            clip_txt_pooled_mapper,
            clip_txt_mapper,
            clip_img_mapper,
            clip_mapper,
            effnet_mapper,
            pixels_mapper,
            down_downscalers,
            down_blocks,
            up_blocks,
            up_upscalers,
            up_repeat_mappers,
            clf_conv,
        })
    }

    /// Build the `clip` conditioning tensor `[B, Sk, c_cond]` from available inputs.
    ///
    /// Mirrors `StableCascadeUNet.get_clip_embeddings` in diffusers:
    /// - Stage C: concat([clip_txt_mapper(clip_text),
    ///                    clip_txt_pooled_mapper(pooled).view(B, 4, c_cond),
    ///                    clip_img_mapper(clip_img).view(B, 4, c_cond)],
    ///                   dim=1) → [B, 85, c_cond] → clip_norm (LN).
    /// - Stage B: clip_mapper(pooled).view(B, clip_seq, c_cond) → LN.
    ///
    /// Note: Stage C *always* enters the concat branch at inference because
    /// the prior pipeline passes `clip_img = zeros[B, 1, 768]` when no image
    /// conditioning is provided — `clip_img is not None` is then satisfied.
    /// The Rust caller should follow the same convention.
    fn build_clip_cond(
        &self,
        clip_text_pooled: Option<&Tensor>,
        clip_text: Option<&Tensor>,
        clip_img: Option<&Tensor>,
    ) -> Result<Tensor> {
        let cfg = &self.config;

        let clip = if cfg.use_clip_mapper {
            // Stage B: clip = clip_mapper(clip_text_pooled).view(B, clip_seq, c_cond)
            // Stage B's clip_mapper is applied to the pooled embedding (not hidden).
            let (we, bi) = self
                .clip_mapper
                .as_ref()
                .ok_or_else(|| Error::InvalidInput("Stage B requires clip_mapper".into()))?;
            let pool = clip_text_pooled.ok_or_else(|| {
                Error::InvalidInput("Stage B requires clip_text_pooled".into())
            })?;
            // pool [B, C_in] -> reshape [B, 1, C_in]
            let p3 = pool.unsqueeze(1)?;
            let y = linear_fwd(&p3, we, Some(bi))?; // [B, 1, clip_seq * c_cond]
            let b = y.shape().dims()[0];
            let flat = y.shape().dims()[2];
            let c_cond = cfg.conditioning_dim;
            debug_assert_eq!(flat, cfg.clip_seq * c_cond);
            y.reshape(&[b, cfg.clip_seq, c_cond])?
        } else {
            // Stage C: build from pooled (+ text hidden + image pool).
            let pool = clip_text_pooled.ok_or_else(|| {
                Error::InvalidInput("Stage C requires clip_text_pooled".into())
            })?;
            let (we, bi) = self
                .clip_txt_pooled_mapper
                .as_ref()
                .ok_or_else(|| Error::InvalidInput("Stage C: clip_txt_pooled_mapper missing".into()))?;

            // clip_txt_pool: the pooled embedding is passed as [B, 1280] here.
            // Diffusers receives [B, 1, 1280] and does
            //     mapper(x).view(B, 1*clip_seq, c_cond)  →  [B, 4, c_cond]
            // Our 2D pool produces the same result when we unsqueeze to 3D first.
            let p3 = pool.unsqueeze(1)?; // [B, 1, 1280]
            let yp = linear_fwd(&p3, we, Some(bi))?; // [B, 1, clip_seq * c_cond]
            let b = yp.shape().dims()[0];
            let c_cond = cfg.conditioning_dim;
            let clip_txt_pool = yp.reshape(&[b, cfg.clip_seq, c_cond])?; // [B, 4, c_cond]

            // Build the optional text-hidden and image branches.
            //
            // Diffusers gates the concat on `clip_txt is not None and clip_img is not None`.
            // The prior pipeline ALWAYS passes non-None for both (clip_img=zeros when no
            // image is provided), so we take the concat branch whenever both are available.
            // Our API exposes `clip_img` as Option<&Tensor>; when the caller supplies
            // None we fall back to zeros of shape [B, 1, clip_image_in_channels] so
            // this block still exercises `clip_img_mapper` (its bias + weights touch the
            // zero vector → a non-trivial contribution that the attention sees).
            if let (Some((txt_w, txt_b)), Some(txt)) =
                (self.clip_txt_mapper.as_ref(), clip_text)
            {
                // clip_txt_mapper: Linear(1280 -> 2048) on [B, 77, 1280] → [B, 77, c_cond]
                let clip_txt_mapped = linear_fwd(txt, txt_w, Some(txt_b))?;

                // clip_img_mapper: always run with zeros when caller didn't pass an image.
                let img_pieces: Tensor = if let (Some((img_w, img_b)), Some(img_in_ch)) =
                    (self.clip_img_mapper.as_ref(), cfg.clip_image_in_channels)
                {
                    let img_src: Tensor = if let Some(img) = clip_img {
                        // If caller passed 2D [B, C], unsqueeze to [B, 1, C].
                        let dims = img.shape().dims();
                        if dims.len() == 2 { img.unsqueeze(1)? } else { img.clone() }
                    } else {
                        // [B, 1, clip_image_in_channels] zeros — mirrors pipeline default.
                        Tensor::zeros_dtype(
                            Shape::from_dims(&[b, 1, img_in_ch]),
                            DType::BF16,
                            txt_w.device().clone(),
                        )?
                    };
                    // Linear(768 -> clip_seq * c_cond = 2048 * 4) on [B, 1, 768] → [B, 1, 8192]
                    let yi = linear_fwd(&img_src, img_w, Some(img_b))?;
                    let img_b_dim = yi.shape().dims()[0];
                    let img_s = yi.shape().dims()[1]; // = 1
                    yi.reshape(&[img_b_dim, img_s * cfg.clip_seq, c_cond])? // [B, 4, c_cond]
                } else {
                    // No img mapper loaded — still match diffusers' shape by padding with zeros
                    // so the attention sees an 85-token KV. (In practice Stage C always has
                    // clip_img_mapper; this is just defensive.)
                    Tensor::zeros_dtype(
                        Shape::from_dims(&[b, cfg.clip_seq, c_cond]),
                        DType::BF16,
                        txt_w.device().clone(),
                    )?
                };

                // concat([clip_txt, clip_txt_pool, clip_img], dim=1)
                Tensor::cat(&[&clip_txt_mapped, &clip_txt_pool, &img_pieces], 1)?
            } else {
                clip_txt_pool
            }
        };

        // Apply LN over channel axis (no affine, eps=1e-6) — `self.clip_norm`.
        let last = clip.shape().dims().len() - 1;
        debug_assert_eq!(last, 2); // [B, Sk, C]
        cuda_ops_bf16::layer_norm_bf16(&clip, None, None, 1e-6)
    }

    /// Apply the input `embedding` block: PixelUnshuffle(p) → Conv2d(in*p^2, c0, 1x1) → LN-noaffine.
    fn apply_embedding(&self, sample: &Tensor) -> Result<Tensor> {
        let p = self.config.patch_size;
        let x = if p == 1 {
            sample.clone()
        } else {
            pixel_unshuffle(sample, p)?
        };
        let x = self.embedding_conv.forward(&x)?;
        let ln = CascadeChannelLayerNorm::no_affine(self.embedding_ln_c, 1e-6);
        ln.forward(&x)
    }

    /// Apply the output `clf` block: LN-noaffine → Conv2d(c0, out*p^2, 1x1) → PixelShuffle(p).
    fn apply_clf(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.config.patch_size;
        let ln = CascadeChannelLayerNorm::no_affine(self.clf_ln_c, 1e-6);
        let xn = ln.forward(x)?;
        let y = self.clf_conv.forward(&xn)?;
        if p == 1 {
            Ok(y)
        } else {
            pixel_shuffle(&y, p)
        }
    }

    /// Build `r_embed` = concat(sinusoidal(r), sinusoidal(r_sca), sinusoidal(r_crp))
    /// where any missing conditioning is replaced with 0. Returns `[1, (1+num_conds) * emb_dim]`.
    fn build_r_embed(&self, r: f32, sca: Option<f32>, crp: Option<f32>, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let emb_dim = self.config.timestep_ratio_embedding_dim;
        let mut embs = vec![timestep_ratio_embedding(r, emb_dim, device)?];
        for cname in self.config.timestep_conditioning_type.iter() {
            let v = match cname.as_str() {
                "sca" => sca.unwrap_or(0.0),
                "crp" => crp.unwrap_or(0.0),
                _ => 0.0,
            };
            embs.push(timestep_ratio_embedding(v, emb_dim, device)?);
        }
        let refs: Vec<&Tensor> = embs.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Stage-C or Stage-B forward.
    ///
    /// `clip_text_pooled`:   [1, 1280]   (required for Stage C; Stage B uses this too via clip_mapper)
    /// `clip_text_hidden`:   [1, 77, 1280] (optional, Stage C uses the full hidden)
    /// `effnet_cond`:        [1, 16, H_c, W_c] (Stage B only — Stage C's output latent)
    pub fn forward(
        &self,
        sample: &Tensor,
        r: f32,
        clip_text_pooled: Option<&Tensor>,
        clip_text_hidden: Option<&Tensor>,
        effnet_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_dumps(sample, r, clip_text_pooled, clip_text_hidden, effnet_cond, None)
    }

    /// Same as `forward`, but also appends per-stage intermediate activations
    /// into `dumps` (keyed by a short string) for parity debugging.
    pub fn forward_with_dumps(
        &self,
        sample: &Tensor,
        r: f32,
        clip_text_pooled: Option<&Tensor>,
        clip_text_hidden: Option<&Tensor>,
        effnet_cond: Option<&Tensor>,
        mut dumps: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        macro_rules! dump {
            ($k:expr, $t:expr) => {
                if let Some(ref mut d) = dumps {
                    d.insert($k.to_string(), ($t).clone());
                }
            };
        }
        let device = sample.device();

        // 1. r embed.
        let r_embed = self.build_r_embed(r, None, None, device)?;

        // 2. CLIP conditioning.
        let clip_cond = self.build_clip_cond(clip_text_pooled, clip_text_hidden, None)?;

        // 3. Initial embedding (includes PixelUnshuffle for Stage B).
        let mut x = self.apply_embedding(sample)?;
        let x_dims = x.shape().dims().to_vec();
        let (h, w) = (x_dims[2], x_dims[3]);

        // 4. effnet_mapper (Stage B only). Diffusers uses bilinear here.
        if let (Some(em), Some(effnet)) = (self.effnet_mapper.as_ref(), effnet_cond) {
            let cfg = Upsample2dConfig::new(UpsampleMode::Bilinear).with_size((h, w));
            let up = Upsample2d::new(cfg).forward(effnet)?;
            let mapped = em.forward(&up)?;
            x = x.add(&mapped)?;
        }

        // 5. pixels_mapper (Stage B only): apply to zeros [B,3,8,8] and upsample.
        if let Some(pm) = self.pixels_mapper.as_ref() {
            let pix = Tensor::zeros_dtype(
                Shape::from_dims(&[1, 3, 8, 8]),
                DType::BF16,
                device.clone(),
            )?;
            let mapped = pm.forward(&pix)?;
            let cfg = Upsample2dConfig::new(UpsampleMode::Bilinear).with_size((h, w));
            let up = Upsample2d::new(cfg).forward(&mapped)?;
            x = x.add(&up)?;
        }
        dump!("emb", x);

        // 6. Down path. Collect each level's output for skip-connections.
        let mut level_outputs: Vec<Tensor> = Vec::with_capacity(self.config.block_out_channels.len());
        for level in 0..self.config.block_out_channels.len() {
            if let Some(ds) = &self.down_downscalers[level] {
                x = apply_downscaler(ds, &x, level)?;
            }
            dump!(format!("down_{level}_after_ds"), x);
            for blk in &self.down_blocks[level] {
                x = forward_block(blk, &x, &r_embed, &clip_cond)?;
            }
            dump!(format!("down_{level}_out"), x);
            level_outputs.insert(0, x.clone()); // deepest first
        }

        // 7. Up path — start at the deepest level.
        //
        // Replays the diffusers reference loop structure exactly:
        //   for i, (up_block, upscaler, repmap) in enumerate(...):
        //       for j in range(len(repmap) + 1):
        //           for k, block in enumerate(up_block):
        //               skip = level_outputs[i] if k == 0 and i > 0 else None
        //               ... apply block ...
        //           if j < len(repmap):
        //               x = repmap[j](x)
        //       x = upscaler(x)
        //
        // Crucially, the c_skip ResBlock at k==0 is called on EVERY outer-j
        // iteration (skip is provided on every pass), and the 1×1 repeat
        // mapper convs run BETWEEN block-group passes, not after.
        let mut x = level_outputs[0].clone();
        for up_idx in 0..self.config.block_out_channels.len() {
            let repmap_count = self.up_repeat_mappers[up_idx].len();
            for j in 0..(repmap_count + 1) {
                for (k, blk) in self.up_blocks[up_idx].iter().enumerate() {
                    match blk {
                        Block::Res(r_) => {
                            // Skip concat at k == 0 of a non-first up level — on every j.
                            let skip = if up_idx > 0 && k == 0 {
                                Some(&level_outputs[up_idx])
                            } else {
                                None
                            };
                            let mut y_in = x.clone();
                            // Guard: if the skip spatial differs (shouldn't normally, since the
                            // upscaler already matched), interp x to skip's size.
                            if let Some(s) = skip {
                                let sd = s.shape().dims();
                                let xd = y_in.shape().dims();
                                if xd[2] != sd[2] || xd[3] != sd[3] {
                                    let cfg = Upsample2dConfig::new(UpsampleMode::Nearest)
                                        .with_size((sd[2], sd[3]));
                                    y_in = Upsample2d::new(cfg).forward(&y_in)?;
                                }
                            }
                            x = r_.forward(&y_in, skip)?;
                        }
                        Block::TS(t) => {
                            x = t.forward(&x, &r_embed)?;
                        }
                        Block::Attn(a) => {
                            x = a.forward(&x, &clip_cond)?;
                        }
                    }
                }
                // Apply the j-th repeat mapper (if any) — runs AFTER each of the first
                // `repmap_count` block-group passes, BEFORE the next one.
                if j < repmap_count {
                    x = self.up_repeat_mappers[up_idx][j].forward(&x)?;
                }
            }
            dump!(format!("up_{up_idx}_before_us"), x);

            if let Some(us) = &self.up_upscalers[up_idx] {
                x = apply_upscaler(us, &x, device)?;
            }
            dump!(format!("up_{up_idx}_after_us"), x);
        }

        // 8. Output clf.
        let y = self.apply_clf(&x)?;
        dump!("out", y);
        Ok(y)
    }
}

fn apply_downscaler(ds: &Downscaler, x: &Tensor, level: usize) -> Result<Tensor> {
    let _ = level;
    match ds {
        Downscaler::Conv2D(c) => {
            // Stage B downscalers are wrapped in `nn.Sequential(LN_noaffine, Conv2d(k=2,s=2))`.
            // The LN normalizes over channels before the strided conv.
            let in_ch = x.shape().dims()[1];
            let ln = CascadeChannelLayerNorm::no_affine(in_ch, 1e-6);
            let xn = ln.forward(x)?;
            c.forward(&xn)
        }
        Downscaler::BilinearPlusConv1x1 { conv, do_interp } => {
            // Pre-LN no-affine over channels — the Stage C downscaler wrapper is
            // `nn.Sequential(LN_noaffine, UpDownBlock2d)`.
            let in_ch = x.shape().dims()[1];
            let ln = CascadeChannelLayerNorm::no_affine(in_ch, 1e-6);
            let xn = ln.forward(x)?;
            // UpDownBlock2d(mode="down") order is [conv1x1, interp] — conv FIRST.
            let y = conv.forward(&xn)?;
            if *do_interp {
                // UpDownBlock2d-down: bilinear 0.5.
                let dims = y.shape().dims().to_vec();
                let new_h = dims[2] / 2;
                let new_w = dims[3] / 2;
                let cfg = Upsample2dConfig::new(UpsampleMode::Bilinear).with_size((new_h, new_w));
                Upsample2d::new(cfg).forward(&y)
            } else {
                Ok(y)
            }
        }
    }
}

fn apply_upscaler(us: &Upscaler, x: &Tensor, _device: &Arc<CudaDevice>) -> Result<Tensor> {
    match us {
        Upscaler::ConvTranspose2x2 { conv, bias, out_ch } => {
            // Stage B upscalers are wrapped in `nn.Sequential(LN_noaffine, ConvTranspose2d(k=2,s=2))`.
            // Apply the pre-LN, then the emulated ConvTranspose (1x1 Conv2d + PixelShuffle + bias).
            let in_ch = x.shape().dims()[1];
            let ln = CascadeChannelLayerNorm::no_affine(in_ch, 1e-6);
            let xn = ln.forward(x)?;
            let y1 = conv.forward(&xn)?; // [N, out_ch*4, H, W]
            let y2 = pixel_shuffle(&y1, 2)?; // [N, out_ch, 2H, 2W]
            let b4 = bias.reshape(&[1, *out_ch, 1, 1])?;
            y2.add(&b4)
        }
        Upscaler::BilinearPlusConv1x1 { conv, do_interp } => {
            // Pre-LN no-affine over channels — same wrapper as Stage C downscaler.
            let in_ch = x.shape().dims()[1];
            let ln = CascadeChannelLayerNorm::no_affine(in_ch, 1e-6);
            let xn = ln.forward(x)?;
            // UpDownBlock2d(mode="up") order is [interp, conv1x1] — interp FIRST.
            let up = if *do_interp {
                let dims = xn.shape().dims().to_vec();
                let new_h = dims[2] * 2;
                let new_w = dims[3] * 2;
                let cfg = Upsample2dConfig::new(UpsampleMode::Bilinear).with_size((new_h, new_w));
                Upsample2d::new(cfg).forward(&xn)?
            } else {
                xn
            };
            conv.forward(&up)
        }
    }
}

fn forward_block(
    blk: &Block,
    x: &Tensor,
    r_embed: &Tensor,
    clip_cond: &Tensor,
) -> Result<Tensor> {
    match blk {
        Block::Res(r_) => r_.forward(x, None),
        Block::TS(t) => t.forward(x, r_embed),
        Block::Attn(a) => a.forward(x, clip_cond),
    }
}

/// Load one block (R/T/A) from the weight map at `prefix`.
fn load_block(
    type_ch: char,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    c: usize,
    c_skip: usize,
    c_cond: usize,
    n_heads: usize,
    c_timestep: usize,
    conds: &[String],
    device: &Arc<CudaDevice>,
) -> Result<Block> {
    match type_ch {
        'R' => {
            let b = SDCascadeResBlock::from_weights(weights, prefix, c, c_skip, 3, device)?;
            Ok(Block::Res(b))
        }
        'T' => {
            let cs: Vec<&str> = conds.iter().map(|s| s.as_str()).collect();
            let b = SDCascadeTimestepBlock::from_weights(weights, prefix, c, c_timestep, &cs)?;
            Ok(Block::TS(b))
        }
        'A' => {
            // self_attn=True throughout in Cascade config.
            let b = SDCascadeAttnBlock::from_weights(weights, prefix, c, c_cond, n_heads, true)?;
            Ok(Block::Attn(b))
        }
        _ => Err(Error::InvalidInput(format!("unknown block type {type_ch}"))),
    }
}
