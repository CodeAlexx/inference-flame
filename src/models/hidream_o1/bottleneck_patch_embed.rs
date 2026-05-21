//! `BottleneckPatchEmbed` — image patch embedder with a low-rank bottleneck.
//!
//! Reference: `qwen3_vl_transformers.py:944-960`.
//!
//! ```python
//! class BottleneckPatchEmbed(nn.Module):
//!     def __init__(self, config, patch_size=16, in_chans=3,
//!                  pca_dim=768, embed_dim=768, bias=True):
//!         self.proj1 = nn.Linear(patch_size*patch_size*in_chans, pca_dim, bias=False)
//!         self.proj2 = nn.Linear(pca_dim, embed_dim, bias=bias)
//!     def forward(self, x):
//!         return self.proj2(self.proj1(x))
//! ```
//!
//! For HiDream-O1: `patch_size=32`, `in_chans=3`, `pca_dim=hidden_size/4=1024`,
//! `embed_dim=hidden_size=4096`, `bias=True` on `proj2` only
//! (`qwen3_vl_transformers.py:1042`).

use std::sync::Arc;

use flame_core::nn::Linear;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

use super::HiDreamO1Config;
use super::lora::LoraRegistry;

/// Two-stage patch embedder: `Linear(P*P*C → bottleneck) → Linear(bottleneck → hidden)`.
///
/// `proj1` has **no bias**; `proj2` has bias.
pub struct BottleneckPatchEmbed {
    /// Linear from `P*P*in_channels` → `bottleneck_dim`. No bias.
    pub proj1: Linear,
    /// Linear from `bottleneck_dim` → `hidden_size`. Has bias.
    pub proj2: Linear,
    /// Per-side patch size (32 for HiDream-O1).
    pub patch_size: usize,
    /// Input channel count (3 = RGB).
    pub in_channels: usize,
}

impl BottleneckPatchEmbed {
    /// Instantiate with random Xavier init. Weights are loaded later via
    /// `Linear::copy_weight_from` / `copy_bias_from` from the safetensors.
    pub fn new(config: &HiDreamO1Config, device: &Arc<CudaDevice>) -> Result<Self> {
        let in_dim = config.patch_size * config.patch_size * config.patch_in_channels;
        let proj1 = Linear::new_zeroed(in_dim, config.bottleneck_dim, /*bias=*/ false, device)?;
        let proj2 = Linear::new_zeroed(config.bottleneck_dim, config.hidden_size, /*bias=*/ true, device)?;
        Ok(Self {
            proj1,
            proj2,
            patch_size: config.patch_size,
            in_channels: config.patch_in_channels,
        })
    }

    /// Forward: `[B, L, P*P*C] → [B, L, hidden_size]`.
    /// Mirrors `qwen3_vl_transformers.py:958-959`.
    pub fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        self.forward_lora(patches, None)
    }

    /// LoRA-aware forward for edv2-reference O1's non-decoder target set.
    pub fn forward_lora(&self, patches: &Tensor, lora: Option<&LoraRegistry>) -> Result<Tensor> {
        let h = match lora.and_then(|r| r.get_global("x_embedder.proj1")) {
            Some(adapter) => {
                let a = adapter.a_tensor()?;
                let b = adapter.b_tensor()?;
                flame_core::ops::fused_inference::fused_linear3d_native_lora(
                    patches,
                    &self.proj1.weight,
                    self.proj1.bias.as_ref(),
                    Some(&a),
                    Some(&b),
                    adapter.scale,
                )?
            }
            None => flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
                patches,
                &self.proj1.weight,
                self.proj1.bias.as_ref(),
            )?,
        };
        match lora.and_then(|r| r.get_global("x_embedder.proj2")) {
            Some(adapter) => {
                let a = adapter.a_tensor()?;
                let b = adapter.b_tensor()?;
                flame_core::ops::fused_inference::fused_linear3d_native_lora(
                    &h,
                    &self.proj2.weight,
                    self.proj2.bias.as_ref(),
                    Some(&a),
                    Some(&b),
                    adapter.scale,
                )
            }
            None => flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
                &h,
                &self.proj2.weight,
                self.proj2.bias.as_ref(),
            ),
        }
    }

    /// Patchify an image tensor `[B, C, H, W]` into
    /// `[B, (H/P)*(W/P), P*P*C]` using einops-style
    /// `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=P, pw=P)`.
    ///
    /// `H` and `W` must be divisible by `patch_size`.
    pub fn patchify(image: &Tensor, patch_size: usize) -> Result<Tensor> {
        let dims = image.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "BottleneckPatchEmbed::patchify expected [B,C,H,W], got {:?}",
                dims
            )));
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if h % patch_size != 0 || w % patch_size != 0 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "BottleneckPatchEmbed::patchify: H={} W={} must be divisible by P={}",
                h, w, patch_size
            )));
        }
        let hp = h / patch_size;
        let wp = w / patch_size;
        // [B, C, H, W] → [B, C, hp, ph, wp, pw]
        let x = image.reshape(&[b, c, hp, patch_size, wp, patch_size])?;
        // → [B, hp, wp, C, ph, pw]
        let x = x.permute(&[0, 2, 4, 1, 3, 5])?;
        // → [B, hp*wp, C*ph*pw]
        x.reshape(&[b, hp * wp, c * patch_size * patch_size])
    }

    /// Inverse of `patchify`. `[B, hp*wp, P*P*C] → [B, C, hp*P, wp*P]`.
    /// Used by `FinalLayer`'s output to reconstruct the predicted image.
    pub fn unpatchify(
        patches: &Tensor,
        hp: usize,
        wp: usize,
        patch_size: usize,
    ) -> Result<Tensor> {
        let dims = patches.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "BottleneckPatchEmbed::unpatchify expected [B,L,P*P*C], got {:?}",
                dims
            )));
        }
        let (b, l, last) = (dims[0], dims[1], dims[2]);
        if l != hp * wp {
            return Err(flame_core::Error::InvalidOperation(format!(
                "unpatchify: L={} != hp*wp={}*{}",
                l, hp, wp
            )));
        }
        if last % (patch_size * patch_size) != 0 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "unpatchify: last dim {} not divisible by P*P={}",
                last,
                patch_size * patch_size
            )));
        }
        let c = last / (patch_size * patch_size);
        // [B, hp*wp, C*P*P] → [B, hp, wp, C, P, P]
        let x = patches.reshape(&[b, hp, wp, c, patch_size, patch_size])?;
        // → [B, C, hp, P, wp, P]
        let x = x.permute(&[0, 3, 1, 4, 2, 5])?;
        // → [B, C, hp*P, wp*P]
        x.reshape(&[b, c, hp * patch_size, wp * patch_size])
    }
}

// Silence unused-import warnings (DType/Shape used by readers when extending).
#[allow(dead_code)]
const _DTYPE_HINT: DType = DType::BF16;
#[allow(dead_code)]
fn _shape_hint() -> Shape {
    Shape::from_dims(&[0])
}
