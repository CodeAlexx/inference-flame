//! L2P MicroDiffusionModel — 4-stage U-Net pixel-space head.
//!
//! Replaces `FinalLayer + unpatchify` in the standard DiT recipe.
//! Takes the noisy pixel input + the DiT's per-token feature map
//! reshaped to `[B, dim=3840, H/16, W/16]`, emits a pixel-space
//! prediction `[B, 3, H, W]`.
//!
//! Topology (from `diffsynth/models/z_image_dit_L2P.py::MicroDiffusionModel`):
//! - Enc{1..4}: Conv3×3 (c→c') + SiLU, MaxPool2×2 between stages
//! - Bottleneck: cat([p4_out, feat_map_aligned], dim=1) → Conv1×1 (512+3840→512) + SiLU
//!   (feat_map is nearest-interp'd to p4_out's H/W if not already aligned;
//!    at 1024² with H/W multiples of 16 this branch does NOT fire)
//! - Dec{4..1}: Upsample(nearest×2) + Conv3×3 (`up{i}`), then cat skip,
//!   then Conv3×3 + SiLU (`dec{i}`)
//! - out_conv: Conv1×1 (64→3)
//!
//! ## Layout convention
//!
//! All tensors stay NCHW for Conv2d / Upsample2d. flame-core's
//! `MaxPool2d::forward` is NHWC-only, so we permute NCHW→NHWC immediately
//! before pool and NHWC→NCHW immediately after. This is a flame-core
//! quirk, not a bug in this module.
//!
//! ## Contiguity contract
//!
//! Every `Tensor::cat` is followed by `.contiguous()` (5 sites in
//! `forward`). This is the documented kernel contract from CONTEXT.md
//! "Known traps": Conv2d reads garbage from non-contig cat output. Until
//! flame-core fixes the producer the workaround is mandatory.

use flame_core::conv::{Conv2d, Conv2dConfig};
use flame_core::pooling::{MaxPool2d, MaxPool2dConfig};
use flame_core::upsampling::{Upsample2d, Upsample2dConfig, UpsampleMode};
use flame_core::{Error, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// L2P U-Net pixel head.
///
/// Constructed by `MicroDiffusionModel::new` from a fully-resident
/// weight HashMap; expects every conv layer's `weight` and `bias`
/// already on GPU under the canonical "internal" key namespace (see
/// chunk-4 weight loader for the safetensors→internal mapping; for
/// chunk 3 we just read the internal keys directly).
pub struct MicroDiffusionModel {
    // Encoder convs (Conv3×3, in→out: 3→64, 64→128, 128→256, 256→512)
    enc1: Conv2d,
    enc2: Conv2d,
    enc3: Conv2d,
    enc4: Conv2d,
    // Bottleneck (Conv1×1, 512+3840=4352 → 512)
    bottleneck: Conv2d,
    // Up convs — the Conv inside up{i} = Sequential[Upsample, Conv3×3].
    // Channel counts: 64→64, 128→128, 256→256, 512→512 (channel-preserving).
    up1_conv: Conv2d,
    up2_conv: Conv2d,
    up3_conv: Conv2d,
    up4_conv: Conv2d,
    // Decoder convs — the Conv inside dec{i} = Sequential[Conv3×3, SiLU].
    // Each takes 2c→c (post-skip-cat):
    //   dec4: 512+512 → 256
    //   dec3: 256+256 → 128
    //   dec2: 128+128 → 64
    //   dec1: 64+64   → 64
    dec1: Conv2d,
    dec2: Conv2d,
    dec3: Conv2d,
    dec4: Conv2d,
    // Output projection (Conv1×1, 64 → 3)
    out_conv: Conv2d,
    // Pooling / upsampling layers (stateless).
    maxpool: MaxPool2d,
    upsample2x: Upsample2d,
}

impl MicroDiffusionModel {
    /// Construct from a fully-resident weight HashMap.
    ///
    /// Reads 14 conv layers' `weight`+`bias` by safetensors-key
    /// convention from the "L2P internal key" namespace:
    ///
    /// - `local_decoder.enc{1..4}.conv.weight/bias`  (Conv3×3, channels 3→64, 64→128, 128→256, 256→512)
    /// - `local_decoder.bottleneck.conv.weight/bias` (Conv1×1, 512+3840=4352 → 512)
    /// - `local_decoder.up{1..4}.conv.weight/bias`   (Conv3×3, 64→64, 128→128, 256→256, 512→512 — the conv inside up{i} Sequential[Upsample, Conv])
    /// - `local_decoder.dec{1..4}.conv.weight/bias`  (Conv3×3; channels per stage:
    ///     `dec4: 1024→256, dec3: 512→128, dec2: 256→64, dec1: 128→64`.
    ///     Note dec{i} is the post-skip-cat Conv: input channels are
    ///     `up{i}_out + enc{i}_out = 2 * enc{i}_out`.)
    /// - `local_decoder.out_conv.weight/bias`        (Conv1×1, 64 → 3)
    pub fn new(
        weights: &HashMap<String, Tensor>,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Helper: load one Conv2d layer (random-init + copy_weight/bias)
        // by key prefix + config.
        let load_conv = |prefix: &str, cfg: Conv2dConfig| -> Result<Conv2d> {
            let w_key = format!("{prefix}.weight");
            let b_key = format!("{prefix}.bias");
            let w = weights.get(&w_key).ok_or_else(|| {
                Error::InvalidInput(format!("Missing weight: {w_key}"))
            })?;
            let b = weights.get(&b_key).ok_or_else(|| {
                Error::InvalidInput(format!("Missing bias:   {b_key}"))
            })?;
            let mut conv = Conv2d::from_config_with_bias(cfg, device.clone(), true)?;
            conv.copy_weight_from(w)?;
            conv.copy_bias_from(b)?;
            Ok(conv)
        };

        // Conv3×3 (padding=1) helper
        let c3 = |inc: usize, outc: usize| Conv2dConfig {
            in_channels: inc,
            out_channels: outc,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            groups: 1,
        };
        // Conv1×1 (padding=0) helper
        let c1 = |inc: usize, outc: usize| Conv2dConfig {
            in_channels: inc,
            out_channels: outc,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            groups: 1,
        };

        let enc1 = load_conv("local_decoder.enc1.conv", c3(3, 64))?;
        let enc2 = load_conv("local_decoder.enc2.conv", c3(64, 128))?;
        let enc3 = load_conv("local_decoder.enc3.conv", c3(128, 256))?;
        let enc4 = load_conv("local_decoder.enc4.conv", c3(256, 512))?;

        // Bottleneck takes cat([p4_out (512), feat_map (3840)], dim=1) → 512.
        let bottleneck = load_conv("local_decoder.bottleneck.conv", c1(512 + 3840, 512))?;

        // up{i} convs — channel-preserving Conv3×3.
        let up4_conv = load_conv("local_decoder.up4.conv", c3(512, 512))?;
        let up3_conv = load_conv("local_decoder.up3.conv", c3(256, 256))?;
        let up2_conv = load_conv("local_decoder.up2.conv", c3(128, 128))?;
        let up1_conv = load_conv("local_decoder.up1.conv", c3(64, 64))?;

        // dec{i} convs — post-skip-cat 2c→c.
        let dec4 = load_conv("local_decoder.dec4.conv", c3(512 + 512, 256))?;
        let dec3 = load_conv("local_decoder.dec3.conv", c3(256 + 256, 128))?;
        let dec2 = load_conv("local_decoder.dec2.conv", c3(128 + 128, 64))?;
        let dec1 = load_conv("local_decoder.dec1.conv", c3(64 + 64, 64))?;

        let out_conv = load_conv("local_decoder.out_conv", c1(64, 3))?;

        let maxpool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
        let upsample2x = Upsample2d::new(
            Upsample2dConfig::new(UpsampleMode::Nearest).with_scale_factor((2.0, 2.0)),
        );

        Ok(Self {
            enc1,
            enc2,
            enc3,
            enc4,
            bottleneck,
            up1_conv,
            up2_conv,
            up3_conv,
            up4_conv,
            dec1,
            dec2,
            dec3,
            dec4,
            out_conv,
            maxpool,
            upsample2x,
        })
    }

    /// Forward pass.
    ///
    /// - `noisy_input`: `[B, 3, H, W]` BF16. **Input must be BF16.** The
    ///   chunk-4 pipeline wrapper is responsible for casting F32 noise
    ///   to BF16 via `.to_dtype(DType::BF16)?` before invoking this
    ///   method. Per PORT_SPEC §"Special / things to watch" #4 the L2P
    ///   pipeline supplies F32 noise; flame-core's `Conv2d::forward`
    ///   errors on non-BF16 inputs (`conv.rs:330-334`) so the cast
    ///   cannot be silently absorbed.
    /// - `feat_map`: `[B, 3840, fh, fw]` BF16 (DiT image tokens reshaped —
    ///   typically `fh=H/16, fw=W/16`).
    ///
    /// Returns `[B, 3, H, W]` BF16. If H/W aren't multiples of 16, the
    /// bottleneck nearest-interps `feat_map` to match the encoder's
    /// H/16-resolution output. At 1024² inference (H/W multiples of 16)
    /// this branch does NOT fire.
    pub fn forward(&self, noisy_input: &Tensor, feat_map: &Tensor) -> Result<Tensor> {
        // Dtype contract: chunk-4 pipeline must pre-cast F32 noise to BF16.
        // Conv2d::forward errors on non-BF16; surface the contract here.
        debug_assert_eq!(
            noisy_input.dtype(),
            flame_core::DType::BF16,
            "MicroDiffusionModel::forward expects BF16 noisy_input — chunk-4 pipeline must cast F32→BF16 before calling"
        );
        debug_assert_eq!(
            feat_map.dtype(),
            flame_core::DType::BF16,
            "MicroDiffusionModel::forward expects BF16 feat_map"
        );

        // -- Encoder (NCHW; MaxPool2d needs NHWC, so permute around it) ----

        // Stage 1: enc1 + pool1 → [B, 64, H/2, W/2]
        let enc1_out = self.enc1.forward(noisy_input)?.silu()?;
        let p1_out = self.pool_nchw(&enc1_out)?;

        // Stage 2: enc2 + pool2 → [B, 128, H/4, W/4]
        let enc2_out = self.enc2.forward(&p1_out)?.silu()?;
        let p2_out = self.pool_nchw(&enc2_out)?;

        // Stage 3: enc3 + pool3 → [B, 256, H/8, W/8]
        let enc3_out = self.enc3.forward(&p2_out)?.silu()?;
        let p3_out = self.pool_nchw(&enc3_out)?;

        // Stage 4: enc4 + pool4 → [B, 512, H/16, W/16]
        let enc4_out = self.enc4.forward(&p3_out)?.silu()?;
        let p4_out = self.pool_nchw(&enc4_out)?;

        // -- Bottleneck ---------------------------------------------------
        // Align feat_map to p4_out spatial shape if they differ. At 1024²
        // with H/W multiples of 16, feat_map already matches p4_out and
        // the branch does NOT fire.
        let p4_dims = p4_out.shape().dims().to_vec();
        let feat_dims = feat_map.shape().dims();
        let feat_aligned = if feat_dims[2] != p4_dims[2] || feat_dims[3] != p4_dims[3] {
            // Build a one-shot Upsample2d with explicit target size
            // (NOT scale_factor) — defensive rescue path.
            let resize = Upsample2d::new(
                Upsample2dConfig::new(UpsampleMode::Nearest)
                    .with_size((p4_dims[2], p4_dims[3])),
            );
            resize.forward(feat_map)?
        } else {
            feat_map.clone()
        };
        // Cat #1: bottleneck input. Conv1×1 reads it — must be contig.
        let cat = Tensor::cat(&[&p4_out, &feat_aligned], 1)?.contiguous()?;
        let bottleneck_out = self.bottleneck.forward(&cat)?.silu()?;

        // -- Decoder stage 4: up4 + cat enc4 + dec4 → [B, 256, H/8, W/8] --
        let up4_out = self.upsample2x.forward(&bottleneck_out)?;
        let up4_out = self.up4_conv.forward(&up4_out)?;
        // Cat #2: skip connection. Conv3×3 reads it — must be contig.
        let cat4 = Tensor::cat(&[&up4_out, &enc4_out], 1)?.contiguous()?;
        let dec4_out = self.dec4.forward(&cat4)?.silu()?;

        // -- Decoder stage 3: up3 + cat enc3 + dec3 → [B, 128, H/4, W/4] --
        let up3_out = self.upsample2x.forward(&dec4_out)?;
        let up3_out = self.up3_conv.forward(&up3_out)?;
        // Cat #3: skip connection.
        let cat3 = Tensor::cat(&[&up3_out, &enc3_out], 1)?.contiguous()?;
        let dec3_out = self.dec3.forward(&cat3)?.silu()?;

        // -- Decoder stage 2: up2 + cat enc2 + dec2 → [B, 64, H/2, W/2] --
        let up2_out = self.upsample2x.forward(&dec3_out)?;
        let up2_out = self.up2_conv.forward(&up2_out)?;
        // Cat #4: skip connection.
        let cat2 = Tensor::cat(&[&up2_out, &enc2_out], 1)?.contiguous()?;
        let dec2_out = self.dec2.forward(&cat2)?.silu()?;

        // -- Decoder stage 1: up1 + cat enc1 + dec1 → [B, 64, H, W] ------
        let up1_out = self.upsample2x.forward(&dec2_out)?;
        let up1_out = self.up1_conv.forward(&up1_out)?;
        // Cat #5: skip connection.
        let cat1 = Tensor::cat(&[&up1_out, &enc1_out], 1)?.contiguous()?;
        let dec1_out = self.dec1.forward(&cat1)?.silu()?;

        // -- Output projection: out_conv (1×1) → [B, 3, H, W] ------------
        self.out_conv.forward(&dec1_out)
    }

    /// NCHW MaxPool2d helper. flame-core's `MaxPool2d::forward` requires
    /// NHWC BF16, so we permute in and out.
    ///
    /// Both `.contiguous()` calls below are mandatory; each maps to a
    /// distinct kernel contract documented in CONTEXT.md "Known traps"
    /// (cuDNN/Conv2d reads garbage from non-contig / permute-view inputs
    /// because `as_device_ptr_bf16` returns the base storage pointer and
    /// ignores `custom_strides`):
    ///
    /// - **Input side** (NCHW → NHWC): `MaxPool2d::forward` asserts NHWC
    ///   BF16 input via `assert_nhwc_bf16_public`. The pool's first
    ///   internal step is `permute_nhwc_to_nchw` followed by
    ///   `to_dtype(F32)` which would materialize the view today — but
    ///   the kernel contract says "physically NHWC", not "logically
    ///   NHWC over NCHW bytes". Explicit `.contiguous()` matches the
    ///   contract and is robust against future changes to pool's
    ///   internal layout handling.
    /// - **Output side** (NHWC → NCHW): downstream `Conv2d::forward`
    ///   feeds the result into cuDNN, which builds an `NCHW` descriptor
    ///   and reads the **base device pointer** — `custom_strides` from a
    ///   bare permute-view are silently ignored, producing the classic
    ///   "looks like garbage but no error" channel-scramble failure.
    ///   See `flame-core/src/tensor.rs:719-728` for the pointer
    ///   accessor and CONTEXT.md "Tensor::cat non-contig" — same
    ///   hazard, same fix.
    fn pool_nchw(&self, x_nchw: &Tensor) -> Result<Tensor> {
        // NCHW → NHWC for MaxPool's NHWC-BF16 input contract.
        let x_nhwc = x_nchw.permute(&[0, 2, 3, 1])?.contiguous()?;
        let (y_nhwc, _indices) = self.maxpool.forward(&x_nhwc)?;
        // NHWC → NCHW for the next Conv2d (cuDNN reads base pointer,
        // ignores custom_strides — mandatory materialization).
        y_nhwc.permute(&[0, 3, 1, 2])?.contiguous()
    }
}
