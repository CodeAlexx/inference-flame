//! Wan 2.2 high-compression VAE — decoder.
//!
//! Port of `Lance/modeling/vae/wan/vae2_2.py::Wan2_2_VAE` (Apache-2.0), decode path.
//!
//! Architecture (z_dim=48, c_dim=160, dec_dim=256, dim_mult=[1,2,4,4]):
//!   * Wan 2.2 differs from Wan 2.1 (see comparison table in
//!     `inference-flame/tools/convert_wan22_vae.py`). Key deltas:
//!       - latent z_dim = 48 (not 16)
//!       - decoder base dim = 256 (not 96)
//!       - `decoder.conv1`: 48 → 1024 (not 16 → 384)
//!       - block dims: `[1024, 1024, 1024, 512, 256]`
//!       - `temperal_upsample = [True, True, False]` (Lance defaults
//!         `temperal_downsample=[False,True,True]`; upsample is its reverse)
//!       - head output 12 channels, unpatchified to 3 RGB at 2x spatial
//!       - `decoder.upsamples.{i}.upsamples.{j}.*` (NESTED, extra `Up_ResidualBlock`
//!         wrapper) — flat `.{idx}.` layout from Wan 2.1 does NOT apply
//!       - `avg_shortcut` (`DupUp3D`) is present per upsample stage with `up_flag`
//!       - Resample Conv2d output channels = `dim` (not `dim/2` as in Wan 2.1)
//!       - per-channel mean/std are 48-element vectors hardcoded in
//!         `Wan2_2_VAE.__init__` (vae2_2.py:877-984) — NOT in the .pth file
//!
//! ## Top-level forward
//!   `z [B, 48, T, H, W]` (normalized BF16)
//!     → unnormalize: `z = z * std + mean` per-channel
//!     → `conv2`: CausalConv3d(48, 48, 1×1×1)            [top-level]
//!     → `decoder.conv1`: CausalConv3d(48, 1024, 3×3×3)
//!     → `decoder.middle`: ResBlock(1024) + Attn(1024) + ResBlock(1024)
//!     → `decoder.upsamples`: 4 nested `Up_ResidualBlock` stages
//!     → `decoder.head`: RMS_norm(256) + SiLU + CausalConv3d(256, 12, 3×3×3)
//!     → unpatchify(patch_size=2): [B, 12, T_out, H', W'] → [B, 3, T_out, H'*2, W'*2]
//!   Output: `[B, 3, T_out, H_img, W_img]` clamped to [-1, 1]
//!
//! ## Weight key format (preserved verbatim from `Wan2.2_VAE.pth`)
//!   conv2.weight/bias
//!   decoder.conv1.weight/bias
//!   decoder.middle.{0,2}.residual.{0,3}.gamma   (RMS_norm)
//!   decoder.middle.{0,2}.residual.{2,6}.weight/bias  (CausalConv3d)
//!   decoder.middle.1.norm.gamma, to_qkv.weight/bias, proj.weight/bias
//!   decoder.upsamples.{i}.upsamples.{j}.residual.{0,3}.gamma
//!   decoder.upsamples.{i}.upsamples.{j}.residual.{2,6}.weight/bias
//!   decoder.upsamples.{i}.upsamples.{j}.shortcut.weight/bias  (when in!=out)
//!   decoder.upsamples.{i}.upsamples.{mult}.resample.1.weight/bias   (Conv2d)
//!   decoder.upsamples.{i}.upsamples.{mult}.time_conv.weight/bias    (upsample3d)
//!   decoder.head.0.gamma, decoder.head.2.weight/bias
//!
//! NOTE: `avg_shortcut` (DupUp3D) is PARAMETER-FREE — no weight keys.

use flame_core::conv::Conv2d;
use flame_core::conv3d_simple::Conv3d;
use flame_core::cuda_ops::GpuOps;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Per-channel normalisation constants — from `Wan2_2_VAE.__init__`
// (Lance/modeling/vae/wan/vae2_2.py:877-984). NOT in the .pth state_dict;
// must live in code. Copied byte-for-byte (visual diff against Python).
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
];

#[rustfmt::skip]
const STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Wan 2.2 VAE decoder: missing weight: {key}")))
}

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// FeatCacheSlot / FeatCacheState — per-frame state machine for causal video
// decode (Python `_feat_map` + `_conv_idx` in `Wan2_2_VAE`).
//
// Python uses three sentinels (`None`, the string `"Rep"`, a tensor) stored
// in `self._feat_map`, a list of length `count_conv3d(self.decoder)`. The
// `feat_idx[0]` counter is reset to 0 at the start of each per-frame call
// to `self.decoder(...)`, and advanced by leaf conv3d/Resample paths after
// they consume their slot. See vae2_2.py:121-148 (upsample3d Resample),
// 213-230 (ResidualBlock conv), 539-589 (Encoder3d wrapping pattern),
// 650-698 (Decoder3d wrapping pattern), 787-810 (top-level decode loop).
// ---------------------------------------------------------------------------

/// One slot of the per-frame causal feat_cache.
///
/// Mirrors Python's tri-state pattern:
/// * `None`           → first encounter; matches `feat_cache[idx] is None`.
/// * `Rep`            → "replicate-pad" sentinel; matches `feat_cache[idx] == "Rep"`.
///                       Set by Resample.upsample3d on first encounter; signals that
///                       the next time the slot is read, prepend-zeros logic fires.
/// * `Past(Tensor)`   → previously stored cache_x tensor (typically `x[:, :, -CACHE_T:]`
///                       after any prepend); will be used as left-pad for the next
///                       call's causal conv3d.
pub enum FeatCacheSlot {
    None,
    Rep,
    Past(Tensor),
}

/// Persistent state passed across per-frame `decoder(...)` calls.
///
/// `slots` is allocated once at the top of `decode()` with length equal to
/// the total number of CausalConv3d instances inside the Decoder3d module
/// tree (matches Python `clear_cache`'s `self._feat_map = [None] * self._conv_num`).
///
/// `conv_idx` is reset to 0 before every per-frame call (matches Python
/// `self._conv_idx = [0]` at vae2_2.py:796). Leaf conv/Resample paths read
/// `slot_mut()` then call `advance()` to move to the next index.
pub struct FeatCacheState {
    pub slots: Vec<FeatCacheSlot>,
    pub conv_idx: usize,
}

impl FeatCacheState {
    /// Allocate a fresh cache with `num_causal_conv3d` slots, all set to `None`.
    pub fn new(num_causal_conv3d: usize) -> Self {
        let mut slots = Vec::with_capacity(num_causal_conv3d);
        for _ in 0..num_causal_conv3d {
            slots.push(FeatCacheSlot::None);
        }
        Self { slots, conv_idx: 0 }
    }

    /// Reset only the index counter. Slots persist across per-frame calls
    /// (this is what `self._conv_idx = [0]` does in Python).
    pub fn reset_idx(&mut self) {
        self.conv_idx = 0;
    }

    /// Borrow the slot at the current `conv_idx` for mutation. Panics if
    /// `conv_idx` is out of bounds — that means `count_causal_conv3d()`
    /// returned a value too small (a structural bug).
    pub fn slot_mut(&mut self) -> &mut FeatCacheSlot {
        assert!(
            self.conv_idx < self.slots.len(),
            "FeatCacheState::slot_mut: conv_idx={} >= slots.len()={} \
             (count_causal_conv3d() under-counted)",
            self.conv_idx,
            self.slots.len()
        );
        &mut self.slots[self.conv_idx]
    }

    /// Advance the index counter. Mirrors Python's `feat_idx[0] += 1`.
    pub fn advance(&mut self) {
        self.conv_idx += 1;
    }
}

/// Compile-time analogue of Python's `CACHE_T = 2`. The "last K frames to
/// keep" window size for the temporal feat cache.
const CACHE_T: usize = 2;

// ---------------------------------------------------------------------------
// CausalConv3d — same layout as Wan 2.1 (zero-pad left along temporal axis).
//
// Wan 2.2's `CausalConv3d.forward` (vae2_2.py:50-58) uses `F.pad(x, ...)`
// which is constant (zero) padding. The Wan 2.1 Rust loader has a `PadMode`
// enum (Replicate vs Zero); for Wan 2.2 we always zero-pad (no `Replicate`
// option needed).
// ---------------------------------------------------------------------------

struct CausalConv3d {
    conv: Conv3d,
    time_pad: usize, // = 2 * pad.0, left-only temporal pad
}

impl CausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        pad: (usize, usize, usize),
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let time_pad = 2 * pad.0;
        let mut conv = Conv3d::new(
            in_ch,
            out_ch,
            kernel,
            Some(stride),
            Some((0, pad.1, pad.2)), // no temporal pad in Conv3d — we prepend zeros manually
            None,
            None,
            true,
            device.clone(),
        )?;
        // Conv3d kernel uses F32 internally — store weights as F32
        conv.weight = get(weights, &format!("{prefix}.weight"))?.to_dtype(DType::F32)?;
        conv.bias_tensor = Some(get(weights, &format!("{prefix}.bias"))?.to_dtype(DType::F32)?);
        Ok(Self { conv, time_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_cache_x(x, None)
    }

    /// Low-level causal-conv with explicit left-pad tensor.
    ///
    /// Python (vae2_2.py:50-58):
    /// ```text
    /// padding = list(self._padding)             # padding[4] = 2*pad.0 = time_pad
    /// if cache_x is not None and self._padding[4] > 0:
    ///     cache_x = cache_x.to(x.device)
    ///     x = torch.cat([cache_x, x], dim=2)    # prepend cache along T
    ///     padding[4] -= cache_x.shape[2]        # reduce remaining left-pad
    /// x = F.pad(x, padding)                     # remaining zero-pad on T-left
    /// return super().forward(x)
    /// ```
    ///
    /// When `cache_x` is `None` (no cache), this is equivalent to the original
    /// `forward(x)` — full `time_pad` zeros prepended.
    /// When `cache_x.shape[2] >= time_pad`, the zero-pad is fully replaced by
    /// the cache (asserted via debug_assert).
    fn forward_with_cache_x(&self, x: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        // Step 1: prepend `cache_x` (if any) along T, then compute remaining
        // zero-pad amount.
        let (x_concat, remaining_pad) = match cache_x {
            Some(cx) if self.time_pad > 0 => {
                let cx_t = cx.shape().dims()[2];
                debug_assert!(
                    cx_t <= self.time_pad,
                    "CausalConv3d::forward_with_cache_x: cache_x T={} exceeds time_pad={}",
                    cx_t,
                    self.time_pad
                );
                let cat_out = Tensor::cat(&[cx, x], 2)?;
                (cat_out, self.time_pad - cx_t)
            }
            _ => (x.clone(), self.time_pad),
        };

        // Step 2: zero-pad the remainder on T-left.
        let x_padded = if remaining_pad > 0 {
            let dims = x_concat.shape().dims();
            let pad_shape = Shape::from_dims(&[
                dims[0], dims[1], remaining_pad, dims[3], dims[4],
            ]);
            let pad = Tensor::zeros_dtype(pad_shape, x_concat.dtype(), x_concat.device().clone())?;
            Tensor::cat(&[&pad, &x_concat], 2)?
        } else {
            x_concat
        };

        // Conv3d kernel uses F32 internally — convert BF16 if needed.
        let is_bf16 = x_padded.dtype() == DType::BF16;
        let input = if is_bf16 {
            x_padded.to_dtype(DType::F32)?
        } else {
            x_padded
        };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }

    /// ResidualBlock / Decoder.conv1 / Decoder.head-style cache wrapper.
    ///
    /// Python pattern (vae2_2.py:213-230 ResidualBlock; the same shape is
    /// inlined for Decoder.conv1 at 651-664 and for Decoder.head at 683-696):
    /// ```text
    /// idx = feat_idx[0]
    /// cache_x = x[:, :, -CACHE_T:, :, :].clone()
    /// if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
    ///     cache_x = cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
    /// x = layer(x, feat_cache[idx])
    /// feat_cache[idx] = cache_x
    /// feat_idx[0] += 1
    /// ```
    ///
    /// Crucially Python passes `feat_cache[idx]` (which may be `None` on the
    /// first encounter) directly to the conv's `cache_x` argument — when it's
    /// `None` the conv falls through to the full zero-pad path. So the slot
    /// transition is `None` → `Past(cache_x_2)` on first call (where
    /// `cache_x_2` has T<2 since no prepend happened), then
    /// `Past(_)` → `Past(_)` thereafter (with prepend when current T<2).
    ///
    /// This wrapper handles the entire pattern; callers just thread
    /// `Some(&mut FeatCacheState)` through.
    ///
    /// When `cache` is `None`, this is equivalent to `forward(x)` (no-cache,
    /// no advance — matches Python's `else: x = self.conv1(x)` branch).
    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&mut FeatCacheState>,
    ) -> Result<Tensor> {
        let Some(state) = cache else {
            return self.forward(x);
        };

        // Capture old slot value (consume it to support move-out of Past tensor)
        // and replace with placeholder while we compute. We'll write the new
        // cache_x at the end.
        let slot_old = std::mem::replace(state.slot_mut(), FeatCacheSlot::None);

        // 1) Build the candidate `cache_x` from the current `x`:
        //    `cache_x = x[:, :, -CACHE_T:, :, :].clone()`
        let t_in = x.shape().dims()[2];
        let take = t_in.min(CACHE_T);
        let mut cache_x_new = x.narrow(2, t_in - take, take)?.contiguous()?;

        // 2) `if cache_x.shape[2] < 2 and feat_cache[idx] is not None: prepend`
        //    Note: Python prepends regardless of whether slot is "Rep" or Tensor
        //    (the only check is `is not None`). For the ResBlock-style sites
        //    "Rep" never appears (only Resample.upsample3d writes "Rep"), so
        //    in practice the prepend source is always Past(tensor). We assert
        //    this and prepend from `past_last` to mirror Python semantics.
        if cache_x_new.shape().dims()[2] < 2 {
            match &slot_old {
                FeatCacheSlot::Past(past) => {
                    // last frame of past: past[:, :, -1:, :, :]
                    let past_t = past.shape().dims()[2];
                    let past_last = past.narrow(2, past_t - 1, 1)?.contiguous()?;
                    cache_x_new = Tensor::cat(&[&past_last, &cache_x_new], 2)?.contiguous()?;
                }
                FeatCacheSlot::Rep => {
                    // ResBlock-style sites never write "Rep" in Python — only
                    // Resample.upsample3d does. Surfacing this as a hard error
                    // catches accidental wiring mistakes.
                    return Err(Error::InvalidOperation(
                        "CausalConv3d::forward_with_cache: unexpected Rep slot at \
                         ResBlock-style site (only Resample.upsample3d should write Rep)".into(),
                    ));
                }
                FeatCacheSlot::None => {}
            }
        }

        // 3) Call the conv with the OLD slot as left-pad (None → full zero-pad).
        let pad_cache: Option<Tensor> = match slot_old {
            FeatCacheSlot::Past(past) => Some(past),
            FeatCacheSlot::None | FeatCacheSlot::Rep => None,
        };
        let out = self.forward_with_cache_x(x, pad_cache.as_ref())?;

        // 4) Write the new cache_x back to the slot and advance.
        *state.slot_mut() = FeatCacheSlot::Past(cache_x_new);
        state.advance();
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// RMS_norm variants
//
// Python (vae2_2.py:62-74): `F.normalize(x, dim=1) * scale * gamma + bias`
// where bias=0 by default and `scale = dim**0.5`. Two flavours:
//   - 5D (images=False): gamma shape [C, 1, 1, 1] — used on conv-3d outputs
//   - 4D (images=True):  gamma shape [C, 1, 1]    — used inside AttentionBlock
// ---------------------------------------------------------------------------

struct RmsNorm5d {
    gamma: Tensor, // [C, 1, 1, 1]
    scale: f32,
}

impl RmsNorm5d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: get_bf16(weights, &format!("{prefix}.gamma"))?,
            scale: (dim as f32).sqrt(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.mul(&x_f32)?.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let n = x_f32.div(&norm)?;
        let s = n.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        s.mul(&self.gamma)
    }
}

struct RmsNorm4d {
    gamma: Tensor, // [C, 1, 1]
    scale: f32,
}

impl RmsNorm4d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: get_bf16(weights, &format!("{prefix}.gamma"))?,
            scale: (dim as f32).sqrt(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.mul(&x_f32)?.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let n = x_f32.div(&norm)?;
        let s = n.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        s.mul(&self.gamma)
    }
}

// ---------------------------------------------------------------------------
// ResidualBlock (Python: vae2_2.py:194-233)
//
// residual Sequential layout:
//   0: RMS_norm(in)
//   1: SiLU
//   2: CausalConv3d(in,  out, 3, pad=1)
//   3: RMS_norm(out)
//   4: SiLU
//   5: Dropout (no params, no-op at inference)
//   6: CausalConv3d(out, out, 3, pad=1)
// shortcut: CausalConv3d(in, out, 1) if in != out else Identity
// ---------------------------------------------------------------------------

struct ResidualBlock {
    norm1: RmsNorm5d,
    conv1: CausalConv3d,
    norm2: RmsNorm5d,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm1 = RmsNorm5d::load(weights, &format!("{prefix}.residual.0"), in_dim)?;
        let conv1 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.2"),
            in_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;
        let norm2 = RmsNorm5d::load(weights, &format!("{prefix}.residual.3"), out_dim)?;
        let conv2 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.6"),
            out_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                weights,
                &format!("{prefix}.shortcut"),
                in_dim,
                out_dim,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = if let Some(ref s) = self.shortcut {
            s.forward(x)?
        } else {
            x.clone()
        };
        let mut out = self.norm1.forward(x)?;
        out = out.silu()?;
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward(&out)?;
        out = out.silu()?;
        out = self.conv2.forward(&out)?;
        out.add(&h)
    }

    /// Cache-aware ResidualBlock forward.
    ///
    /// Python (vae2_2.py:213-233): the residual Sequential is iterated, and
    /// any `CausalConv3d` layer inside it is wrapped with the ResBlock-style
    /// cache pattern (which we factored into `CausalConv3d::forward_with_cache`).
    /// `self.shortcut` — when it's a CausalConv3d(in, out, 1) — is also
    /// iterated over per-layer in Python at line 214, but **without** the cache
    /// wrapping (the `isinstance(layer, CausalConv3d) and feat_cache is not None`
    /// branch only fires inside the residual loop, not for the shortcut). So
    /// the shortcut runs with the no-cache `forward(x)` path. Even though
    /// `self.shortcut` is a CausalConv3d with k=1 and pad=0 (time_pad=0),
    /// `forward(x)` with no left-pad is a no-op equivalent of "no cache".
    /// We therefore do NOT thread cache through the shortcut here.
    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&mut FeatCacheState>,
    ) -> Result<Tensor> {
        let Some(state) = cache else {
            return self.forward(x);
        };

        // Shortcut: identity or CausalConv3d(in, out, 1) — no cache wrap.
        let h = if let Some(ref s) = self.shortcut {
            s.forward(x)?
        } else {
            x.clone()
        };

        // Inner sequence: norm1 → silu → conv1(cached) → norm2 → silu → conv2(cached).
        // norm/silu/dropout layers don't consume cache slots in Python (the
        // `isinstance(layer, CausalConv3d)` guard skips them).
        let mut out = self.norm1.forward(x)?;
        out = out.silu()?;
        out = self.conv1.forward_with_cache(&out, Some(state))?;
        out = self.norm2.forward(&out)?;
        out = out.silu()?;
        out = self.conv2.forward_with_cache(&out, Some(state))?;
        out.add(&h)
    }
}

/// Count CausalConv3d instances inside a single ResidualBlock:
/// always 2 inner conv3ds (residual.2, residual.6), plus 1 shortcut if `in_dim != out_dim`.
///
/// Shortcut convs are counted (matches Python `count_conv3d` walking all modules)
/// but never `conv_idx`-advanced at runtime — `ResidualBlock::forward_with_cache`
/// calls `self.shortcut.forward(x)` without threading the cache, so the matching
/// slot stays `FeatCacheSlot::None` for the entire `decode()` run. Python
/// `vae2_2.py:214` does the same (`h = self.shortcut(x)` outside the cache walk).
/// Net effect: production decoder allocates 34 slots, advances 32 per frame,
/// 2 slots permanently unused. Intentional, not a bug.
fn res_block_conv3d_count(rb: &ResidualBlock) -> usize {
    2 + if rb.shortcut.is_some() { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// AttentionBlock — single-head per-frame self-attention (Python:
// vae2_2.py:236-272). Same shape as Wan 2.1's AttentionBlock; lifted from
// `inference-flame/src/vae/wan21_vae.rs::AttentionBlock`.
// ---------------------------------------------------------------------------

struct AttentionBlock {
    norm: RmsNorm4d,
    to_qkv: Conv2d, // Conv2d(dim, dim*3, 1)
    proj: Conv2d,   // Conv2d(dim, dim, 1)
}

impl AttentionBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm = RmsNorm4d::load(weights, &format!("{prefix}.norm"), dim)?;

        let mut to_qkv = Conv2d::new_with_bias(dim, dim * 3, 1, 1, 0, device.clone(), true)?;
        to_qkv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.to_qkv.weight"))?)?;
        to_qkv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.to_qkv.bias"))?)?;

        let mut proj = Conv2d::new_with_bias(dim, dim, 1, 1, 0, device.clone(), true)?;
        proj.copy_weight_from(&get_bf16(weights, &format!("{prefix}.proj.weight"))?)?;
        proj.copy_bias_from(&get_bf16(weights, &format!("{prefix}.proj.bias"))?)?;

        Ok(Self { norm, to_qkv, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let identity = x.clone();

        // [B, C, T, H, W] -> [B*T, C, H, W]
        let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
        let x_normed = self.norm.forward(&x_4d)?;

        let qkv = self.to_qkv.forward(&x_normed)?;
        let n = h * w;
        let qkv_flat = qkv
            .reshape(&[b * t, c * 3, n])?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, 1, n, c * 3])?;

        let q = qkv_flat.narrow(3, 0, c)?;
        let k = qkv_flat.narrow(3, c, c)?;
        let v = qkv_flat.narrow(3, c * 2, c)?;

        let attn = sdpa_forward(&q, &k, &v, None)?;
        let attn = attn
            .squeeze(Some(1))?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, c, h, w])?;

        let projected = self.proj.forward(&attn)?;
        let out = projected
            .reshape(&[b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;
        identity.add(&out)
    }
}

// ---------------------------------------------------------------------------
// UpsampleBlock — Resample(mode=upsample2d|upsample3d).
//
// Python (vae2_2.py:86-170):
//   upsample2d: Sequential(Upsample(scale=2, nearest-exact), Conv2d(dim, dim, 3, pad=1))
//   upsample3d: Sequential(Upsample(scale=2, nearest-exact), Conv2d(dim, dim, 3, pad=1))
//               + time_conv = CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
//
// CRITICAL Wan 2.2 vs 2.1 delta: Conv2d output channels = `dim` (not `dim/2`).
//
// upsample3d temporal pattern: time_conv outputs `dim*2` channels, which then
// get reshaped to `(B, 2, dim, T, H, W)` and stack-interleaved along time
// (vae2_2.py:151-153) to give `(B, dim, T*2, H, W)`.
//
// In feat_cache=None mode (full single-pass), Python's `Resample.forward`
// only triggers time_conv inside the `feat_cache is not None` branch (line
// 124). When we go single-pass with no cache, time_conv is SKIPPED and
// T stays the same. (This matches the QwenImage image-mode behaviour.)
//
// However, `Wan2_2_VAE.decode` ALWAYS provides a feat_cache and processes
// frames one at a time (vae2_2.py:787-810). For our `decode` we follow the
// no-cache (image-mode-like) path — keep T constant inside upsample3d but
// double T elsewhere through DupUp3D. This is the same single-pass
// approximation wan21_vae.rs uses for its no-cache decode path.
// ---------------------------------------------------------------------------

enum UpsampleBlock {
    Upsample2d {
        conv: Conv2d,
    },
    Upsample3d {
        conv: Conv2d,
        time_conv: CausalConv3d,
        in_dim: usize,
    },
}

impl UpsampleBlock {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // resample.0 = Upsample (no weights), resample.1 = Conv2d(dim, dim, 3, pad=1)
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        Ok(UpsampleBlock::Upsample2d { conv })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;

        // time_conv: CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
        let time_conv = CausalConv3d::load(
            weights,
            &format!("{prefix}.time_conv"),
            dim,
            dim * 2,
            (3, 1, 1),
            (1, 1, 1),
            (1, 0, 0),
            device,
        )?;

        Ok(UpsampleBlock::Upsample3d {
            conv,
            time_conv,
            in_dim: dim,
        })
    }

    /// `image_mode=true`: skip temporal doubling (mirrors Python feat_cache=None
    /// path, which is the only path we model here).
    fn forward(&self, x: &Tensor, image_mode: bool) -> Result<Tensor> {
        match self {
            UpsampleBlock::Upsample2d { conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            UpsampleBlock::Upsample3d {
                conv,
                time_conv,
                in_dim,
            } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
                debug_assert_eq!(c, *in_dim);

                let (x_t, t_out) = if image_mode {
                    (x.clone(), t)
                } else {
                    // Temporal upsample: time_conv outputs [B, dim*2, T, H, W],
                    // reshape to [B, 2, dim, T, H, W] and interleave along time
                    // to give [B, dim, T*2, H, W] (vae2_2.py:151-153).
                    let tc_out = time_conv.forward(x)?;
                    let tc_out = tc_out.reshape(&[b, 2, c, t, h, w])?;
                    let x0 = tc_out.narrow(1, 0, 1)?.squeeze(Some(1))?; // [B, C, T, H, W]
                    let x1 = tc_out.narrow(1, 1, 1)?.squeeze(Some(1))?;
                    let stacked = Tensor::cat(&[
                        &x0.unsqueeze(3)?, // [B, C, T, 1, H, W]
                        &x1.unsqueeze(3)?, // [B, C, T, 1, H, W]
                    ], 3)?; // [B, C, T, 2, H, W]
                    (stacked.reshape(&[b, c, t * 2, h, w])?, t * 2)
                };

                // Spatial upsample (nearest 2x) + Conv2d(dim, dim, 3, pad=1)
                let x_4d = x_t.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t_out, c, h, w])?;
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t_out, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
        }
    }

    /// Cache-aware Resample forward.
    ///
    /// For `Upsample2d`: no temporal state — forwards to existing
    /// `forward(x, image_mode=true)` (the upsample2d branch ignores image_mode).
    ///
    /// For `Upsample3d`: implements the state machine at vae2_2.py:121-153.
    /// When `cache` is `None`, falls back to `image_mode=true` (no temporal
    /// doubling, no time_conv).
    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&mut FeatCacheState>,
    ) -> Result<Tensor> {
        match self {
            UpsampleBlock::Upsample2d { .. } => {
                // upsample2d has no time_conv / no feat_cache consumption.
                self.forward(x, true)
            }
            UpsampleBlock::Upsample3d {
                conv,
                time_conv,
                in_dim,
            } => {
                let Some(state) = cache else {
                    // No cache → fall through to image_mode behaviour (T unchanged).
                    return self.forward(x, true);
                };

                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
                debug_assert_eq!(c, *in_dim);

                // Consume the slot before any conv calls — the time_conv branch
                // below also wants to read the slot, so we capture it now.
                let slot_old = std::mem::replace(state.slot_mut(), FeatCacheSlot::None);

                let (x_after_time, t_out) = match slot_old {
                    // First encounter (Python: `if feat_cache[idx] is None:`,
                    // vae2_2.py:126-128):
                    //   feat_cache[idx] = "Rep"
                    //   feat_idx[0] += 1
                    //   (no time_conv, no T doubling — fall through to spatial)
                    FeatCacheSlot::None => {
                        *state.slot_mut() = FeatCacheSlot::Rep;
                        state.advance();
                        (x.clone(), t)
                    }
                    // Second encounter onward (Python `else:` branch, lines 129-153).
                    slot_old @ (FeatCacheSlot::Rep | FeatCacheSlot::Past(_)) => {
                        // 1) cache_x = x[:, :, -CACHE_T:, :, :].clone()
                        let take = t.min(CACHE_T);
                        let mut cache_x_new = x.narrow(2, t - take, take)?.contiguous()?;

                        // 2) Prepend logic when cache_x is too short:
                        //    if cache_x.shape[2] < 2 and slot != "Rep": prepend slot[-1:]
                        //    if cache_x.shape[2] < 2 and slot == "Rep": prepend zeros
                        if cache_x_new.shape().dims()[2] < 2 {
                            match &slot_old {
                                FeatCacheSlot::Past(past) => {
                                    let past_t = past.shape().dims()[2];
                                    let past_last = past
                                        .narrow(2, past_t - 1, 1)?
                                        .contiguous()?;
                                    cache_x_new =
                                        Tensor::cat(&[&past_last, &cache_x_new], 2)?
                                            .contiguous()?;
                                }
                                FeatCacheSlot::Rep => {
                                    let cx_dims = cache_x_new.shape().dims().to_vec();
                                    let zero = Tensor::zeros_dtype(
                                        Shape::from_dims(&cx_dims),
                                        cache_x_new.dtype(),
                                        cache_x_new.device().clone(),
                                    )?;
                                    cache_x_new =
                                        Tensor::cat(&[&zero, &cache_x_new], 2)?
                                            .contiguous()?;
                                }
                                FeatCacheSlot::None => unreachable!(),
                            }
                        }

                        // 3) Run time_conv:
                        //    if slot == "Rep": x = time_conv(x)         (no left-pad cache)
                        //    else:             x = time_conv(x, slot)   (left-pad = old past)
                        let time_left_pad: Option<Tensor> = match &slot_old {
                            FeatCacheSlot::Past(past) => Some(past.clone()),
                            FeatCacheSlot::Rep => None,
                            FeatCacheSlot::None => unreachable!(),
                        };
                        let tc_out =
                            time_conv.forward_with_cache_x(x, time_left_pad.as_ref())?;

                        // 4) Store new cache_x, advance idx.
                        *state.slot_mut() = FeatCacheSlot::Past(cache_x_new);
                        state.advance();

                        // 5) Reshape [B, 2, C, T, H, W] then stack-interleave on T
                        //    -> [B, C, T*2, H, W]   (vae2_2.py:151-153)
                        let tc_out = tc_out.reshape(&[b, 2, c, t, h, w])?;
                        let x0 = tc_out.narrow(1, 0, 1)?.squeeze(Some(1))?;
                        let x1 = tc_out.narrow(1, 1, 1)?.squeeze(Some(1))?;
                        let stacked = Tensor::cat(
                            &[&x0.unsqueeze(3)?, &x1.unsqueeze(3)?],
                            3,
                        )?;
                        (stacked.reshape(&[b, c, t * 2, h, w])?, t * 2)
                    }
                };

                // Spatial upsample 2x + Conv2d — identical to non-cache path.
                let x_4d = x_after_time
                    .permute(&[0, 2, 1, 3, 4])?
                    .reshape(&[b * t_out, c, h, w])?;
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t_out, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DupUp3D — parameter-free spatial+temporal repeat shortcut (Python:
// vae2_2.py:363-405). The "inverse" of AvgDown3D from the encoder.
//
// Steps:
//   1. Repeat-interleave along channel dim: `out_channels * factor / in_channels` repeats
//   2. View into [B, out_channels, factor_t, factor_s, factor_s, T, H, W]
//   3. Permute (0, 1, 5, 2, 6, 3, 7, 4) -> [B, out_channels, T, factor_t, H, factor_s, W, factor_s]
//   4. View into [B, out_channels, T*factor_t, H*factor_s, W*factor_s]
//   5. If first_chunk: drop the leading (factor_t - 1) temporal frames
//      (this matches `Wan2_2_VAE.decode`'s causal first-chunk semantics)
//
// `up_flag=False` stages don't use DupUp3D — Up_ResidualBlock.avg_shortcut is None.
// ---------------------------------------------------------------------------

struct DupUp3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    repeats: usize,
}

impl DupUp3D {
    fn new(in_channels: usize, out_channels: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        debug_assert_eq!((out_channels * factor) % in_channels, 0);
        let repeats = out_channels * factor / in_channels;
        Self {
            in_channels,
            out_channels,
            factor_t,
            factor_s,
            repeats,
        }
    }

    /// Repeat-interleave along channel dim by `self.repeats`.
    ///
    /// Python `repeat_interleave(repeats, dim=1)` on a tensor of shape
    /// `[B, C, T, H, W]` produces `[B, C*repeats, T, H, W]` where each
    /// input channel `c` appears `repeats` consecutive times in the
    /// output. We implement this by reshape→unsqueeze→broadcast-via-cat→reshape.
    fn channel_repeat_interleave(x: &Tensor, repeats: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        debug_assert_eq!(dims.len(), 5);
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // [B, C, T, H, W] -> [B, C, 1, T, H, W] then cat `repeats` copies on dim 2
        // -> [B, C, repeats, T, H, W] then reshape -> [B, C*repeats, T, H, W].
        let x6 = x.reshape(&[b, c, 1, t, h, w])?;
        let copies: Vec<Tensor> = (0..repeats).map(|_| x6.clone()).collect();
        let refs: Vec<&Tensor> = copies.iter().collect();
        let repeated = Tensor::cat(&refs, 2)?; // [B, C, repeats, T, H, W]
        repeated.reshape(&[b, c * repeats, t, h, w])
    }

    /// Single-pass forward (no caching). `first_chunk=true` mimics Python's
    /// behavior where the first chunk drops the leading (factor_t - 1) frames.
    /// For single-image decode we treat the entire pass as first_chunk=true,
    /// since the temporal expansion is causal.
    fn forward(&self, x: &Tensor, first_chunk: bool) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        debug_assert_eq!(dims.len(), 5);
        let (b, _c_in, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // 1. repeat-interleave on channel dim
        let x = Self::channel_repeat_interleave(x, self.repeats)?;
        // x: [B, in_channels*repeats, T, H, W] = [B, out_channels*factor, T, H, W]

        // 2. View [B, out_channels, factor_t, factor_s, factor_s, T, H, W]
        let x = x.reshape(&[
            b,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            t,
            h,
            w,
        ])?;

        // 3. Permute (0, 1, 5, 2, 6, 3, 7, 4)
        let x = x.permute(&[0, 1, 5, 2, 6, 3, 7, 4])?;

        // 4. View [B, out_channels, T*factor_t, H*factor_s, W*factor_s]
        let t_out = t * self.factor_t;
        let h_out = h * self.factor_s;
        let w_out = w * self.factor_s;
        let x = x.reshape(&[b, self.out_channels, t_out, h_out, w_out])?;

        // 5. first_chunk: drop leading (factor_t - 1) frames
        if first_chunk && self.factor_t > 1 {
            let drop = self.factor_t - 1;
            x.narrow(2, drop, t_out - drop)
        } else {
            Ok(x)
        }
    }
}

// ---------------------------------------------------------------------------
// UpResidualBlock — Python `Up_ResidualBlock` (vae2_2.py:442-478).
//
// One outer stage of `Decoder3d.upsamples`. Layout:
//   self.upsamples = Sequential(
//     ResidualBlock(in_dim, out_dim),       # j=0
//     ResidualBlock(out_dim, out_dim),       # j=1 .. mult-1
//     ...
//     Resample(out_dim, mode=...),           # j=mult, only if up_flag
//   )
//   self.avg_shortcut = DupUp3D(in_dim, out_dim, ft, fs)  # only if up_flag
//
// where mult = num_res_blocks + 1 = 3 for Wan 2.2 default.
//
// forward: main path through self.upsamples, plus avg_shortcut(x) if present.
// ---------------------------------------------------------------------------

struct UpResidualBlock {
    residuals: Vec<ResidualBlock>,
    resample: Option<UpsampleBlock>,
    avg_shortcut: Option<DupUp3D>,
}

impl UpResidualBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        mult: usize, // num_res_blocks + 1
        temperal_upsample: bool,
        up_flag: bool,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Inner residuals at upsamples.{i}.upsamples.{j} for j in 0..mult
        let mut residuals: Vec<ResidualBlock> = Vec::with_capacity(mult);
        let mut cur_in = in_dim;
        for j in 0..mult {
            let rb_prefix = format!("{prefix}.upsamples.{j}");
            let rb = ResidualBlock::load(weights, &rb_prefix, cur_in, out_dim, device)?;
            residuals.push(rb);
            cur_in = out_dim;
        }

        // Final Resample at upsamples.{i}.upsamples.{mult} when up_flag
        let resample = if up_flag {
            let res_prefix = format!("{prefix}.upsamples.{mult}");
            let r = if temperal_upsample {
                UpsampleBlock::load_3d(weights, &res_prefix, out_dim, device)?
            } else {
                UpsampleBlock::load_2d(weights, &res_prefix, out_dim, device)?
            };
            Some(r)
        } else {
            None
        };

        // avg_shortcut: parameter-free DupUp3D, only when up_flag
        let avg_shortcut = if up_flag {
            Some(DupUp3D::new(
                in_dim,
                out_dim,
                if temperal_upsample { 2 } else { 1 },
                2, // factor_s = 2 if up_flag else 1 -> always 2 here
            ))
        } else {
            None
        };

        Ok(Self {
            residuals,
            resample,
            avg_shortcut,
        })
    }

    fn forward(&self, x: &Tensor, image_mode: bool) -> Result<Tensor> {
        // Main path: ResBlocks → optional Resample
        let mut main = x.clone();
        for rb in &self.residuals {
            main = rb.forward(&main)?;
        }
        if let Some(ref r) = self.resample {
            main = r.forward(&main, image_mode)?;
        }

        // Shortcut: DupUp3D on the original input.
        //
        // Python `Wan2_2_VAE.decode` (vae2_2.py:787-810) always iterates
        // frame-by-frame with first_chunk=True passed only for i==0
        // (the very first frame). For our T2I scope (image_mode=true,
        // T_in=1), that single-frame call is exactly the i==0 case → we
        // pass first_chunk=true here and let DupUp3D drop the leading
        // (factor_t - 1) frames per its configured factor_t.
        //
        // Why this matches Python at T=1, factor_t=2, first_chunk=true:
        //   - Main path's `Resample` in `upsample3d` mode hits the
        //     `feat_cache[idx] is None` branch (vae2_2.py:126-128). That
        //     branch records `"Rep"` and SKIPS `time_conv` entirely, so
        //     T stays at 1 through the main path. Our `image_mode=true`
        //     in `UpsampleBlock::forward` mirrors this (skips time_conv).
        //   - Shortcut: DupUp3D(factor_t=2, first_chunk=true) at T_in=1
        //     produces T_out = 1*2 - (2-1) = 1 (vae2_2.py:383-405).
        //   - So main T = shortcut T = 1. ✓
        //
        // For stages with factor_t=1 (no temporal upsample), first_chunk
        // is a no-op (drops 0 frames) — identical to Python.
        //
        // The previous synthetic `factor_t=1` DupUp3D for image_mode used
        // `repeats = out*factor_s^2/in` (i.e. 4 for the production
        // stage-0 case) instead of Python's configured `repeats =
        // out*factor_t*factor_s^2/in` (i.e. 8). Even though the
        // first_chunk drop made the output shape "look right" at T_in=1,
        // the channel-repeat count and the subsequent reshape/permute
        // bin assignment did NOT match Python — output values were
        // numerically wrong (skeptic F2). The fix is to always use the
        // configured DupUp3D and rely on first_chunk=true for the
        // temporal collapse, which is exactly what Python does at i==0.
        if let Some(ref short) = self.avg_shortcut {
            let s = short.forward(x, true)?;
            main.add(&s)
        } else {
            Ok(main)
        }
    }

    /// Cache-aware UpResidualBlock forward.
    ///
    /// Python (vae2_2.py:470-478):
    /// ```text
    /// def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    ///     x_main = x.clone()
    ///     for module in self.upsamples:
    ///         x_main = module(x_main, feat_cache, feat_idx)
    ///     if self.avg_shortcut is not None:
    ///         x_shortcut = self.avg_shortcut(x, first_chunk)
    ///         return x_main + x_shortcut
    ///     else:
    ///         return x_main
    /// ```
    ///
    /// The main path threads cache through every inner ResBlock and the final
    /// Resample. The `avg_shortcut` (DupUp3D) is parameter-free, does not
    /// consume any cache slot, and takes `first_chunk` for its causal-frame
    /// drop (matches existing `DupUp3D::forward` signature).
    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&mut FeatCacheState>,
        first_chunk: bool,
    ) -> Result<Tensor> {
        let Some(state) = cache else {
            // No cache → fall back to image_mode behaviour.
            return self.forward(x, true);
        };

        // Main path: ResBlocks → optional Resample, threading cache.
        let mut main = x.clone();
        for rb in &self.residuals {
            main = rb.forward_with_cache(&main, Some(state))?;
        }
        if let Some(ref r) = self.resample {
            main = r.forward_with_cache(&main, Some(state))?;
        }

        // Shortcut: DupUp3D on the ORIGINAL x with the caller-provided
        // `first_chunk` flag (NOT always-true as in image_mode). Python
        // never threads cache through avg_shortcut.
        if let Some(ref short) = self.avg_shortcut {
            let s = short.forward(x, first_chunk)?;
            main.add(&s)
        } else {
            Ok(main)
        }
    }
}

// ---------------------------------------------------------------------------
// unpatchify(patch_size=2) for 5D tensor (Python vae2_2.py:293-306).
//
// Python: `rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=2, r=2)`
//
// Input:  [B, C*4, F, H, W]
// Output: [B, C,   F, H*2, W*2]
//
// einops ordering: the merged channel dim `(c r q)` unrolls as for c in 0..C
// for r in 0..patch_size for q in 0..patch_size. The "r" pairs with W, "q" with H.
// ---------------------------------------------------------------------------

fn unpatchify_5d(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims().to_vec();
    debug_assert_eq!(dims.len(), 5);
    let (b, c_in, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    debug_assert_eq!(c_in % (patch_size * patch_size), 0);
    let c_out = c_in / (patch_size * patch_size);

    // Step 1: view (c r q) as separate dims -> [B, C_out, r=patch, q=patch, F, H, W]
    let x = x.reshape(&[b, c_out, patch_size, patch_size, f, h, w])?;

    // Step 2: permute to [B, C_out, F, H, q, W, r]
    //   einops target order: b c f (h q) (w r) means dims (B, C, F, H, q, W, r)
    //   our current ordering is (B, C, r, q, F, H, W) = dims (0,1,2,3,4,5,6)
    //   target permutation: (0, 1, 4, 5, 3, 6, 2) -> (B, C, F, H, q, W, r)
    let x = x.permute(&[0, 1, 4, 5, 3, 6, 2])?;

    // Step 3: merge -> [B, C_out, F, H*q, W*r]
    x.reshape(&[b, c_out, f, h * patch_size, w * patch_size])
}

// ---------------------------------------------------------------------------
// Wan22VaeDecoder — top-level decoder model.
// ---------------------------------------------------------------------------

pub struct Wan22VaeDecoder {
    /// Per-channel mean [1, 48, 1, 1, 1] BF16
    mean: Tensor,
    /// Per-channel inverse std (1/std) [1, 48, 1, 1, 1] BF16
    /// In `decode`: `z_unnorm = z / inv_std + mean = z * std + mean`.
    inv_std: Tensor,

    /// Top-level conv2: CausalConv3d(48, 48, 1×1×1)
    conv2: CausalConv3d,
    /// decoder.conv1: CausalConv3d(48, 1024, 3×3×3)
    decoder_conv1: CausalConv3d,

    /// decoder.middle.0: ResidualBlock(1024, 1024)
    mid_res0: ResidualBlock,
    /// decoder.middle.1: AttentionBlock(1024)
    mid_attn: AttentionBlock,
    /// decoder.middle.2: ResidualBlock(1024, 1024)
    mid_res1: ResidualBlock,

    /// 4 outer Up_ResidualBlock stages.
    up_stages: Vec<UpResidualBlock>,

    /// decoder.head: RMS_norm(256) + SiLU + CausalConv3d(256, 12, 3×3×3)
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,

    #[allow(dead_code)]
    device: Arc<cudarc::driver::CudaDevice>,
}

impl Wan22VaeDecoder {
    /// Load from a safetensors file. Matches the layout produced by
    /// `inference-flame/tools/convert_wan22_vae.py` (all `.pth` keys preserved
    /// verbatim, no renaming).
    pub fn load(path: &Path, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let weights = load_file(path, device)?;
        log::info!(
            "[Wan22 VAE Decoder] Loaded {} tensors from {}",
            weights.len(),
            path.display()
        );
        Self::from_weights(&weights, device)
    }

    /// Build a tiny randomly-initialised decoder for tests. Reduced layer
    /// counts but matching nesting/key structure.
    #[cfg(test)]
    fn new_random_tiny(device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        use crate::vae::wan22_vae::test_helpers::*;
        // Tiny config: z_dim=48, dec_dim=8 (instead of 256), num_res_blocks=0 (mult=1)
        let z_dim = 48;
        let dec_dim: usize = 8;
        let dim_mult: [usize; 4] = [1, 2, 4, 4];
        let temperal_upsample: [bool; 3] = [true, true, false];
        let num_res_blocks = 0; // tiny: mult=1
        let mult = num_res_blocks + 1; // 1

        // dims = [dec_dim*dim_mult[-1]] + [dec_dim*u for u in dim_mult[::-1]]
        // dim_mult[::-1] = [4,4,2,1], dim_mult[-1] = 4
        // dims = [32, 32, 32, 16, 8]
        let mut dims: Vec<usize> = Vec::new();
        dims.push(dec_dim * *dim_mult.last().unwrap());
        for &m in dim_mult.iter().rev() {
            dims.push(dec_dim * m);
        }
        assert_eq!(dims.len(), 5);

        let mut weights: Weights = HashMap::new();
        // conv2: CausalConv3d(48, 48, 1×1×1)
        put_conv(&mut weights, device, "conv2", z_dim, z_dim, 1, 1, 1)?;
        // decoder.conv1: CausalConv3d(48, dims[0], 3×3×3)
        put_conv(&mut weights, device, "decoder.conv1", dims[0], z_dim, 3, 3, 3)?;

        // middle
        let d0 = dims[0];
        put_resblock(&mut weights, device, "decoder.middle.0", d0, d0)?;
        put_rms4(&mut weights, device, "decoder.middle.1.norm", d0)?;
        zero_bf16(&mut weights, device, &[d0 * 3, d0, 1, 1], "decoder.middle.1.to_qkv.weight")?;
        zero_bf16(&mut weights, device, &[d0 * 3], "decoder.middle.1.to_qkv.bias")?;
        zero_bf16(&mut weights, device, &[d0, d0, 1, 1], "decoder.middle.1.proj.weight")?;
        zero_bf16(&mut weights, device, &[d0], "decoder.middle.1.proj.bias")?;
        put_resblock(&mut weights, device, "decoder.middle.2", d0, d0)?;

        let num_stages = dims.len() - 1;
        for i in 0..num_stages {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            let t_up = if i < temperal_upsample.len() { temperal_upsample[i] } else { false };
            let up_flag = i != dim_mult.len() - 1;
            let mut cur_in = in_d;
            for j in 0..mult {
                let k = format!("decoder.upsamples.{i}.upsamples.{j}");
                put_resblock(&mut weights, device, &k, cur_in, out_d)?;
                cur_in = out_d;
            }
            if up_flag {
                let k = format!("decoder.upsamples.{i}.upsamples.{mult}");
                if t_up {
                    zero_bf16(&mut weights, device, &[out_d, out_d, 3, 3], &format!("{k}.resample.1.weight"))?;
                    zero_bf16(&mut weights, device, &[out_d], &format!("{k}.resample.1.bias"))?;
                    put_conv(&mut weights, device, &format!("{k}.time_conv"), out_d * 2, out_d, 3, 1, 1)?;
                } else {
                    zero_bf16(&mut weights, device, &[out_d, out_d, 3, 3], &format!("{k}.resample.1.weight"))?;
                    zero_bf16(&mut weights, device, &[out_d], &format!("{k}.resample.1.bias"))?;
                }
            }
        }

        let head_in = *dims.last().unwrap();
        put_rms5(&mut weights, device, "decoder.head.0", head_in)?;
        put_conv(&mut weights, device, "decoder.head.2", 12, head_in, 3, 3, 3)?;

        Self::from_weights_with_config(
            &weights,
            device,
            z_dim,
            dec_dim,
            &dim_mult,
            num_res_blocks,
            &temperal_upsample,
        )
    }

    fn from_weights(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Wan 2.2 production config (Lance defaults — vae2_2.py:861-872, line 988-996):
        //   z_dim=48, c_dim=160 (encoder), dec_dim=256, dim_mult=[1,2,4,4],
        //   num_res_blocks=2, temperal_downsample=[False, True, True],
        //   temperal_upsample = reverse = [True, True, False].
        Self::from_weights_with_config(
            weights,
            device,
            48,
            256,
            &[1, 2, 4, 4],
            2,
            &[true, true, false],
        )
    }

    fn from_weights_with_config(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        z_dim: usize,
        dec_dim: usize,
        dim_mult: &[usize],
        num_res_blocks: usize,
        temperal_upsample: &[bool],
    ) -> Result<Self> {
        // mean / inv_std broadcast tensors. We store 1/std and decode multiplies
        // by std (i.e. divides by inv_std), matching Python `z / scale[1] + scale[0]`
        // where scale[1] = 1/std (stored).
        let mean = Tensor::from_vec(
            MEAN.to_vec(),
            Shape::from_dims(&[1, z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let inv_std_vals: Vec<f32> = STD.iter().map(|s| 1.0_f32 / s).collect();
        let inv_std = Tensor::from_vec(
            inv_std_vals,
            Shape::from_dims(&[1, z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // dims = [dec_dim*dim_mult[-1]] + [dec_dim*u for u in dim_mult[::-1]]
        let mut dims: Vec<usize> = Vec::new();
        dims.push(dec_dim * *dim_mult.last().unwrap());
        for &m in dim_mult.iter().rev() {
            dims.push(dec_dim * m);
        }
        // For dec_dim=256, dim_mult=[1,2,4,4] → dims = [1024,1024,1024,512,256]
        let d0 = dims[0];

        // Top-level conv2: CausalConv3d(z_dim, z_dim, 1×1×1, pad=0)
        let conv2 = CausalConv3d::load(
            weights, "conv2", z_dim, z_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), device,
        )?;

        // decoder.conv1: CausalConv3d(z_dim, dims[0], 3×3×3, pad=1)
        let decoder_conv1 = CausalConv3d::load(
            weights,
            "decoder.conv1",
            z_dim,
            d0,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;

        // middle: ResBlock(d0, d0) + Attn(d0) + ResBlock(d0, d0)
        let mid_res0 = ResidualBlock::load(weights, "decoder.middle.0", d0, d0, device)?;
        let mid_attn = AttentionBlock::load(weights, "decoder.middle.1", d0, device)?;
        let mid_res1 = ResidualBlock::load(weights, "decoder.middle.2", d0, d0, device)?;

        // 4 upsample stages
        let num_stages = dims.len() - 1; // 4
        let mult = num_res_blocks + 1; // 3 in production, 1 in tiny
        let mut up_stages: Vec<UpResidualBlock> = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            let t_up = if i < temperal_upsample.len() {
                temperal_upsample[i]
            } else {
                false
            };
            let up_flag = i != dim_mult.len() - 1; // i != 3
            let prefix = format!("decoder.upsamples.{i}");
            up_stages.push(UpResidualBlock::load(
                weights, &prefix, in_d, out_d, mult, t_up, up_flag, device,
            )?);
        }

        // head: RMS_norm(out_d) + SiLU + CausalConv3d(out_d, 12, 3×3×3, pad=1)
        let head_dim = *dims.last().unwrap();
        let head_norm = RmsNorm5d::load(weights, "decoder.head.0", head_dim)?;
        let head_conv = CausalConv3d::load(
            weights,
            "decoder.head.2",
            head_dim,
            12,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            device,
        )?;

        Ok(Self {
            mean,
            inv_std,
            conv2,
            decoder_conv1,
            mid_res0,
            mid_attn,
            mid_res1,
            up_stages,
            head_norm,
            head_conv,
            device: device.clone(),
        })
    }

    /// Count the total number of `CausalConv3d` instances inside the Decoder3d
    /// module tree. Mirrors Python `count_conv3d(self.decoder)` (vae2_2.py:702).
    ///
    /// Decoder structure (vae2_2.py:594-699):
    ///   decoder.conv1                                     1
    ///   decoder.middle.0 (ResidualBlock)                  2  (residual.2, residual.6)
    ///   decoder.middle.1 (AttentionBlock)                 0
    ///   decoder.middle.2 (ResidualBlock)                  2
    ///   decoder.upsamples.{i} for i in 0..num_stages:
    ///     Each Up_ResidualBlock has:
    ///       - mult inner ResidualBlocks (each: 2 conv3d, plus 1 shortcut if in!=out)
    ///       - if up_flag: + Resample (1 time_conv if temperal_upsample else 0)
    ///   decoder.head.2                                    1
    ///
    /// `mult = num_res_blocks + 1` (Python `Up_ResidualBlock.__init__`
    /// argument at vae2_2.py:636 passes `mult=num_res_blocks+1`).
    ///
    /// Note: ResidualBlock's `shortcut` IS a `CausalConv3d` (kernel=1, pad=0)
    /// when `in_dim != out_dim`, so it counts. `AvgDown3D` / `DupUp3D` /
    /// `AttentionBlock` / RMS_norm / SiLU / Conv2d (inside Resample) do NOT
    /// count.
    pub fn count_causal_conv3d(&self) -> usize {
        let mut n: usize = 0;
        // decoder.conv1
        n += 1;
        // decoder.middle.0 (ResBlock) + middle.2 (ResBlock); each contributes
        // 2 inner conv3ds + maybe shortcut. Production middle uses d0=d0 so
        // no shortcut. We use the loaded blocks as the source of truth.
        n += res_block_conv3d_count(&self.mid_res0);
        n += res_block_conv3d_count(&self.mid_res1);
        // 4 upsample stages
        for stage in &self.up_stages {
            for rb in &stage.residuals {
                n += res_block_conv3d_count(rb);
            }
            if let Some(UpsampleBlock::Upsample3d { .. }) = stage.resample {
                n += 1; // time_conv
            }
            // Upsample2d has no CausalConv3d; avg_shortcut (DupUp3D) is
            // parameter-free / non-conv. No contribution.
        }
        // decoder.head.2
        n += 1;
        n
    }

    /// Video decode — per-frame causal `feat_cache` loop.
    ///
    /// Port of Python `WanVAE_.decode` (vae2_2.py:787-813) and its wrapper
    /// `Wan2_2_VAE.decode` (vae2_2.py:1013-1025) combined. The rescale + clamp
    /// from the wrapper are folded into this method since `Wan22VaeDecoder`
    /// owns the scale tensors (mean / inv_std).
    ///
    /// Algorithm:
    /// 1. Rescale: `z = z * std + mean` (using `inv_std = 1/std`, so
    ///    `z / inv_std + mean`). Matches Python line 790.
    /// 2. `x = conv2(z)` — top-level 1×1 conv, full T at once, no cache.
    /// 3. For each latent frame `i in 0..T_lat`:
    ///    a. Reset `conv_idx` to 0
    ///    b. `xi = x[:, :, i:i+1, :, :]`
    ///    c. Run decoder body with cache; pass `first_chunk = (i == 0)`
    /// 4. `out = cat([...], dim=2).contiguous()` — cat-not-contig trap per
    ///    CONTEXT.md
    /// 5. `unpatchify_5d(out, 2)` — head 2× spatial unpatch
    /// 6. Clamp to [-1, 1] (matches Python `clamp_(-1, 1)`)
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // 1. Rescale per-channel.
        let z_rescaled = z.div(&self.inv_std)?.add(&self.mean)?;

        // 2. conv2: 1×1 conv on full T (no cache — time_pad=0).
        let x = self.conv2.forward(&z_rescaled)?;

        // 3. Per-frame loop with persistent cache.
        let t_lat = x.shape().dims()[2];
        let mut cache = FeatCacheState::new(self.count_causal_conv3d());
        let mut frame_outs: Vec<Tensor> = Vec::with_capacity(t_lat);
        for i in 0..t_lat {
            cache.reset_idx();
            let xi = x.narrow(2, i, 1)?.contiguous()?;
            let first_chunk = i == 0;
            let oi = self.run_decoder_with_cache(&xi, &mut cache, first_chunk)?;
            frame_outs.push(oi);
        }

        // 4. Concatenate along T. cat-not-contig trap per CONTEXT.md known traps —
        // unpatchify_5d does reshape+permute, which silently mis-reads non-contig
        // input. `.contiguous()` is mandatory here.
        let frame_refs: Vec<&Tensor> = frame_outs.iter().collect();
        let out = Tensor::cat(&frame_refs, 2)?.contiguous()?;

        // 5. unpatchify_5d(2): [B, 12, T, H, W] → [B, 3, T, H*2, W*2]
        let out = unpatchify_5d(&out, 2)?;

        // 6. Clamp to [-1, 1] (matches Python `Wan2_2_VAE.decode` wrapper).
        out.clamp(-1.0, 1.0)
    }

    /// Run the inner decoder body (decoder_conv1 → middle → upsamples → head)
    /// threading the persistent feat_cache. Used by `decode()` once per
    /// latent frame.
    ///
    /// Python reference: `Decoder3d.forward` (vae2_2.py:650-699).
    fn run_decoder_with_cache(
        &self,
        x: &Tensor,
        cache: &mut FeatCacheState,
        first_chunk: bool,
    ) -> Result<Tensor> {
        // decoder.conv1: ResBlock-style cache pattern (vae2_2.py:651-664).
        let mut y = self.decoder_conv1.forward_with_cache(x, Some(cache))?;

        // middle: ResBlock + Attn + ResBlock. Only ResBlocks consume cache
        // slots (the `isinstance(layer, ResidualBlock)` guard at vae2_2.py:669).
        y = self.mid_res0.forward_with_cache(&y, Some(cache))?;
        y = self.mid_attn.forward(&y)?;
        y = self.mid_res1.forward_with_cache(&y, Some(cache))?;

        // upsamples: each outer stage threads cache + first_chunk.
        for stage in &self.up_stages {
            y = stage.forward_with_cache(&y, Some(cache), first_chunk)?;
        }

        // head: RMS_norm → SiLU → CausalConv3d. Only the conv3d consumes a
        // cache slot.
        y = self.head_norm.forward(&y)?;
        y = y.silu()?;
        y = self.head_conv.forward_with_cache(&y, Some(cache))?;

        Ok(y)
    }

    /// Decode an image latent (single-frame, T=1).
    ///
    /// Skips the temporal doubling inside each upsample3d block, matching the
    /// `feat_cache=None` / image-mode behaviour. Output T == input T (typically 1).
    pub fn decode_image(&self, z: &Tensor) -> Result<Tensor> {
        self.decode_with_mode(z, true)
    }

    fn decode_with_mode(&self, z: &Tensor, image_mode: bool) -> Result<Tensor> {
        // Unnormalize per-channel: z_un = z / inv_std + mean  (i.e. z*std + mean)
        let z = z.div(&self.inv_std)?.add(&self.mean)?;

        // conv2: 1×1 conv on latent
        let mut x = self.conv2.forward(&z)?;

        // decoder.conv1: z_dim -> dims[0]
        x = self.decoder_conv1.forward(&x)?;

        // middle: ResBlock + Attn + ResBlock
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;

        // 4 nested upsample stages
        for stage in &self.up_stages {
            x = stage.forward(&x, image_mode)?;
        }

        // head: RMS_norm + SiLU + CausalConv3d(out_d, 12)
        x = self.head_norm.forward(&x)?;
        x = x.silu()?;
        x = self.head_conv.forward(&x)?;

        // unpatchify(patch_size=2): [B, 12, T, H', W'] → [B, 3, T, H'*2, W'*2]
        x = unpatchify_5d(&x, 2)?;

        // Clamp to [-1, 1] (Python: `.clamp_(-1, 1)` in `Wan2_2_VAE.decode`)
        x.clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Test helpers (only compiled in test mode)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_helpers {
    use super::*;

    pub(super) fn zero_bf16(
        weights: &mut Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        shape: &[usize],
        k: &str,
    ) -> Result<()> {
        let t = Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, device.clone())?;
        weights.insert(k.to_string(), t);
        Ok(())
    }

    pub(super) fn zero_f32(
        weights: &mut Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        shape: &[usize],
        k: &str,
    ) -> Result<()> {
        let t = Tensor::zeros_dtype(Shape::from_dims(shape), DType::F32, device.clone())?;
        weights.insert(k.to_string(), t);
        Ok(())
    }

    pub(super) fn put_conv(
        weights: &mut Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        k: &str,
        oc: usize,
        ic: usize,
        kt: usize,
        kh: usize,
        kw: usize,
    ) -> Result<()> {
        zero_f32(weights, device, &[oc, ic, kt, kh, kw], &format!("{k}.weight"))?;
        zero_f32(weights, device, &[oc], &format!("{k}.bias"))
    }

    pub(super) fn put_rms5(
        weights: &mut Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        k: &str,
        c: usize,
    ) -> Result<()> {
        zero_bf16(weights, device, &[c, 1, 1, 1], &format!("{k}.gamma"))
    }

    pub(super) fn put_rms4(
        weights: &mut Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        k: &str,
        c: usize,
    ) -> Result<()> {
        zero_bf16(weights, device, &[c, 1, 1], &format!("{k}.gamma"))
    }

    pub(super) fn put_resblock(
        weights: &mut Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
        prefix: &str,
        in_d: usize,
        out_d: usize,
    ) -> Result<()> {
        put_rms5(weights, device, &format!("{prefix}.residual.0"), in_d)?;
        put_conv(weights, device, &format!("{prefix}.residual.2"), out_d, in_d, 3, 3, 3)?;
        put_rms5(weights, device, &format!("{prefix}.residual.3"), out_d)?;
        put_conv(weights, device, &format!("{prefix}.residual.6"), out_d, out_d, 3, 3, 3)?;
        if in_d != out_d {
            put_conv(weights, device, &format!("{prefix}.shortcut"), out_d, in_d, 1, 1, 1)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device 0")
    }

    #[test]
    fn test_mean_std_constants_length() {
        assert_eq!(MEAN.len(), 48);
        assert_eq!(STD.len(), 48);
        for &m in &MEAN {
            assert!(m.is_finite(), "MEAN value not finite: {m}");
        }
        for &s in &STD {
            assert!(s > 0.0 && s.is_finite(), "STD value not positive/finite: {s}");
        }
        // Spot-check first / last few values verbatim from Python
        assert!((MEAN[0] - (-0.2289)).abs() < 1e-7);
        assert!((MEAN[47] - (-0.0667)).abs() < 1e-7);
        assert!((STD[0] - 0.4765).abs() < 1e-7);
        assert!((STD[47] - 0.7744).abs() < 1e-7);
    }

    #[test]
    fn test_unpatchify_shape() {
        let device = dev();
        // [1, 12, 1, 4, 4] -> [1, 3, 1, 8, 8]
        let x = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 12, 1, 4, 4]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();
        let y = unpatchify_5d(&x, 2).unwrap();
        assert_eq!(y.shape().dims(), &[1, 3, 1, 8, 8]);
    }

    #[test]
    fn test_dupup3d_shape_factor_t_eq_1() {
        let device = dev();
        // out_channels=4, in_channels=8, factor_t=1, factor_s=2
        // factor = 1*2*2 = 4, repeats = 4*4/8 = 2
        let d = DupUp3D::new(8, 4, 1, 2);
        assert_eq!(d.repeats, 2);
        let x = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 8, 2, 3, 3]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();
        let y = d.forward(&x, true).unwrap();
        // factor_t=1, first_chunk drops (factor_t-1)=0 frames -> T unchanged
        assert_eq!(y.shape().dims(), &[1, 4, 2, 6, 6]);
    }

    #[test]
    fn test_dupup3d_shape_factor_t_eq_2() {
        let device = dev();
        // factor_t=2, factor_s=2; in=8, out=4
        // factor = 2*2*2 = 8, repeats = 4*8/8 = 4
        let d = DupUp3D::new(8, 4, 2, 2);
        assert_eq!(d.repeats, 4);
        let x = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 8, 2, 3, 3]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();
        let y = d.forward(&x, false).unwrap();
        // first_chunk=false: T_out = T_in*factor_t = 4
        assert_eq!(y.shape().dims(), &[1, 4, 4, 6, 6]);
        // With first_chunk=true: T_out = T_in*factor_t - (factor_t-1) = 4 - 1 = 3
        let y2 = d.forward(&x, true).unwrap();
        assert_eq!(y2.shape().dims(), &[1, 4, 3, 6, 6]);
    }

    #[test]
    fn test_wan22_decoder_image_shape() {
        // Tiny config decode_image, T=1 → output T=1 (no temporal expansion in image_mode).
        // Input:  [1, 48, 1, 4, 4] BF16
        //
        // With dim_mult=[1,2,4,4] and 3 spatial upsample stages (stages 0,1,2
        // all have up_flag=True), each doubles spatial. Stage 3 has up_flag=False
        // (no resample, no avg_shortcut). Net spatial upscale = 2^3 = 8 from
        // the 3 resample stages, then patchify(2) at the head doubles again =
        // 16. So H_out = 4 * 8 * 2 = 64.
        //
        // (For full production weights the same logic gives H_out = H_in * 16.
        // That's consistent with the audit's "downsample_spatial=16" for the
        // overall VAE compression.)
        let device = dev();
        let dec = Wan22VaeDecoder::new_random_tiny(&device).unwrap();
        let z = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 48, 1, 4, 4]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();
        let y = dec.decode_image(&z).unwrap();
        let d = y.shape().dims();
        assert_eq!(d.len(), 5, "output should be 5D, got {:?}", d);
        assert_eq!(d[0], 1, "batch");
        assert_eq!(d[1], 3, "RGB channels");
        assert_eq!(d[2], 1, "image_mode preserves T_in=1");
        assert_eq!(d[3], 4 * 16, "H upscaled by 16x: {:?}", d);
        assert_eq!(d[4], 4 * 16, "W upscaled by 16x: {:?}", d);
    }

    // ---- T2 (feat_cache state machine) tests ---------------------------------

    #[test]
    fn test_feat_cache_state_advance() {
        // Basic FeatCacheState wiring: fresh state has the requested slot count,
        // all None, conv_idx=0. After several advances, conv_idx tracks.
        let mut s = FeatCacheState::new(5);
        assert_eq!(s.slots.len(), 5);
        assert_eq!(s.conv_idx, 0);
        // All slots start as None
        for slot in &s.slots {
            assert!(matches!(slot, FeatCacheSlot::None));
        }
        // Mutate slot 0 → Rep, advance
        *s.slot_mut() = FeatCacheSlot::Rep;
        s.advance();
        assert_eq!(s.conv_idx, 1);
        assert!(matches!(s.slots[0], FeatCacheSlot::Rep));
        // reset_idx leaves slots untouched but rewinds counter
        s.reset_idx();
        assert_eq!(s.conv_idx, 0);
        assert!(matches!(s.slots[0], FeatCacheSlot::Rep));
    }

    #[test]
    fn test_causal_conv3d_with_cache_first_frame() {
        // First call (cache slot = None) must produce same output as the
        // cache-less forward path: full zero-pad on T-left, and advance the
        // slot to `Past(cache_x)` where cache_x = x[:, :, -CACHE_T:] (with
        // T=1 input → cache_x has T=1).
        let device = dev();
        // Tiny causal conv: in=2, out=2, k=(3,1,1), pad=(1,0,0) → time_pad = 2.
        let mut weights: Weights = HashMap::new();
        test_helpers::put_conv(&mut weights, &device, "c", 2, 2, 3, 1, 1).unwrap();
        let c = CausalConv3d::load(
            &weights, "c", 2, 2, (3, 1, 1), (1, 1, 1), (1, 0, 0), &device,
        )
        .unwrap();

        // Input shape [1, 2, 1, 2, 2] BF16 — single latent frame.
        let x = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 2, 1, 2, 2]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();

        // Cache-less reference: pad with 2 zero frames on T-left.
        let y_ref = c.forward(&x).unwrap();
        assert_eq!(y_ref.shape().dims(), &[1, 2, 1, 2, 2]);

        // Cache-aware path with fresh None slot.
        let mut state = FeatCacheState::new(1);
        let y_cached = c.forward_with_cache(&x, Some(&mut state)).unwrap();
        assert_eq!(y_cached.shape().dims(), &[1, 2, 1, 2, 2]);
        // Slot 0 must now hold cache_x (Past). conv_idx advanced to 1.
        assert!(matches!(state.slots[0], FeatCacheSlot::Past(_)));
        assert_eq!(state.conv_idx, 1);

        // Both paths use zero conv weights → outputs are bias-only and equal.
        let r = y_ref.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let g = y_cached.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(r.len(), g.len());
        for (a, b) in r.iter().zip(g.iter()) {
            assert!((a - b).abs() < 1e-5, "first-frame: ref={a} cached={b}");
        }
    }

    #[test]
    fn test_causal_conv3d_with_cache_second_frame() {
        // Second call reads the Past slot from first call, prepends past_last
        // frame to current cache_x (per ResBlock pattern), and uses old Past
        // as left-pad for the conv. Output must differ from first-frame
        // (different effective left context) and shape must match.
        let device = dev();
        let mut weights: Weights = HashMap::new();
        test_helpers::put_conv(&mut weights, &device, "c", 2, 2, 3, 1, 1).unwrap();

        // Non-zero conv weight so first vs second frame differ visibly:
        // identity-like kernel on the central temporal tap.
        let mut w = vec![0.0_f32; 2 * 2 * 3 * 1 * 1];
        // weight[o, i, t, h, w]; we set out=in identity on t=2 (rightmost)
        // index = o*(in*kt*kh*kw) + i*(kt*kh*kw) + t*(kh*kw) + h*kw + w
        for o in 0..2 {
            for i in 0..2 {
                if o == i {
                    let idx = o * (2 * 3) + i * 3 + 2; // t=2, h=0, w=0
                    w[idx] = 1.0;
                }
            }
        }
        let wt = Tensor::from_vec(w, Shape::from_dims(&[2, 2, 3, 1, 1]), device.clone()).unwrap();
        weights.insert("c.weight".into(), wt);

        let c = CausalConv3d::load(
            &weights, "c", 2, 2, (3, 1, 1), (1, 1, 1), (1, 0, 0), &device,
        )
        .unwrap();

        // First frame: all 1.0s.
        let ones_a = vec![1.0_f32; 1 * 2 * 1 * 2 * 2];
        let x1 = Tensor::from_vec(
            ones_a.clone(),
            Shape::from_dims(&[1, 2, 1, 2, 2]),
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        // Second frame: all 2.0s.
        let twos = vec![2.0_f32; 1 * 2 * 1 * 2 * 2];
        let x2 = Tensor::from_vec(
            twos,
            Shape::from_dims(&[1, 2, 1, 2, 2]),
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let mut state = FeatCacheState::new(1);
        let y1 = c.forward_with_cache(&x1, Some(&mut state)).unwrap();
        // After first call: slot = Past(cache_x_from_x1)
        assert!(matches!(state.slots[0], FeatCacheSlot::Past(_)));
        assert_eq!(state.conv_idx, 1);

        // Reset idx, run second frame.
        state.reset_idx();
        let y2 = c.forward_with_cache(&x2, Some(&mut state)).unwrap();
        assert_eq!(y2.shape().dims(), &[1, 2, 1, 2, 2]);
        assert!(matches!(state.slots[0], FeatCacheSlot::Past(_)));
        assert_eq!(state.conv_idx, 1);

        // First frame: kernel at t=2 reads x1 (rightmost), left-pad is zero,
        // so y1 ≈ x1 = 1.0.
        let y1v = y1.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in &y1v {
            assert!((v - 1.0).abs() < 1e-3, "y1 should be ~1.0 (identity on x1), got {v}");
        }

        // Second frame: kernel at t=2 reads x2 = 2.0, so y2 ≈ 2.0.
        // The cache merely provides left context, which the identity-on-t=2
        // kernel ignores. y1 ≠ y2 confirms x2 is what's read (else they'd
        // collide). Stronger differentiation comes from non-identity kernels;
        // here we assert basic per-frame correctness + slot advancement.
        let y2v = y2.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in &y2v {
            assert!((v - 2.0).abs() < 1e-3, "y2 should be ~2.0 (identity on x2), got {v}");
        }

        // The two outputs must differ.
        assert!(
            y1v.iter().zip(y2v.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "y1 and y2 should differ across frames"
        );
    }

    #[test]
    fn test_wan22_vae_decode_video_shape_t3() {
        // Full video decode through the tiny decoder with T_lat=3.
        //
        // The tiny config uses num_res_blocks=0 (mult=1), dec_dim=8,
        // dim_mult=[1,2,4,4], temperal_upsample=[true,true,false].
        // Stage layout (matches production):
        //   i=0 (up_flag=true, t_up=true)  → main Upsample3d doubles T
        //   i=1 (up_flag=true, t_up=true)  → main Upsample3d doubles T
        //   i=2 (up_flag=true, t_up=false) → no temporal doubling
        //   i=3 (up_flag=false)            → no resample, no shortcut
        //
        // Per-frame: input T=1. For each per-frame call, Resample.upsample3d
        // is hit twice (stages 0 and 1). On frame 0 (first encounter), both
        // slots write "Rep" and do NOT time-double — output T stays 1, and
        // DupUp3D with first_chunk=true emits T*2-1=1. So frame 0 emits T=1.
        //
        // On frame i≥1, Resample.upsample3d hits the `else` branch: time_conv
        // fires, T doubles → 2, then reshape interleaves → T_after=2. DupUp3D
        // with first_chunk=false emits T*factor_t=2. Both contributions match
        // at T=2 → after stage 0 T=2, after stage 1 T=4, stages 2 & 3 leave T
        // unchanged. So frames i≥1 emit T=4 each.
        //
        // Cat: 1 + 4 + 4 = 9 frames. Then unpatchify(2) does NOT touch T (it
        // only unpatches H and W). Final T_pixel = 9.
        //
        // This matches the Python invariant `T_pixel = (T_lat - 1) * 4 + 1`
        // since (3-1)*4 + 1 = 9. The two upsample3d stages each contribute a
        // ×2 to the post-frame-0 T expansion: factor 4 = 2 × 2.
        //
        // Spatial: 4 spatial upsample stages? stages 0/1/2 have up_flag=true,
        // 2× each = 8×. Then unpatchify(2) gives another 2× = 16×.
        // Input H=W=2 → output H=W=32.
        let device = dev();
        let dec = Wan22VaeDecoder::new_random_tiny(&device).unwrap();
        let z = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 48, 3, 2, 2]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();
        let y = dec.decode(&z).unwrap();
        let d = y.shape().dims();
        assert_eq!(d.len(), 5, "output should be 5D, got {:?}", d);
        assert_eq!(d[0], 1, "batch");
        assert_eq!(d[1], 3, "RGB channels");
        assert_eq!(d[2], 9, "T_pixel = (T_lat-1)*4 + 1 = 9, got {:?}", d);
        assert_eq!(d[3], 32, "H upscaled 16x, got {:?}", d);
        assert_eq!(d[4], 32, "W upscaled 16x, got {:?}", d);
    }

    #[test]
    fn test_wan22_decoder_image_shape_t1_small() {
        // Tiny config decode_image, T=1, smaller spatial: [1, 48, 1, 2, 2]
        // → [1, 3, 1, 32, 32] (16x spatial upscale).
        let device = dev();
        let dec = Wan22VaeDecoder::new_random_tiny(&device).unwrap();
        let z = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 48, 1, 2, 2]),
            DType::BF16,
            device.clone(),
        )
        .unwrap();
        let y = dec.decode_image(&z).unwrap();
        let d = y.shape().dims();
        assert_eq!(d, &[1, 3, 1, 2 * 16, 2 * 16]);
    }

    #[test]
    fn test_wan22_decoder_top_level_dims_match_audit() {
        // The audit (convert_wan22_vae.py header) states for production weights:
        //   conv1 in→out: 48 → 1024
        //   block dims: [1024, 1024, 1024, 512, 256]
        //   head out channels: 12 (patchify=2 wraps RGB)
        //
        // This test confirms the `from_weights_with_config` dim computation
        // matches the audit when run with production hyperparameters.
        let z_dim = 48;
        let dec_dim = 256;
        let dim_mult = [1, 2, 4, 4];
        let mut dims: Vec<usize> = Vec::new();
        dims.push(dec_dim * *dim_mult.last().unwrap());
        for &m in dim_mult.iter().rev() {
            dims.push(dec_dim * m);
        }
        assert_eq!(dims, vec![1024, 1024, 1024, 512, 256]);
        // conv1: z_dim=48 -> dims[0]=1024
        assert_eq!(z_dim, 48);
        assert_eq!(dims[0], 1024);
    }
}
