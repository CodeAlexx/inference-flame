//! QwenImage VAE Decoder — pure Rust, flame-core.
//!
//! Architecturally identical to the Wan 2.1 VAE decoder (base_dim=96,
//! z_dim=16, dim_mult=[1,2,4,4], matching `latents_mean` / `latents_std`)
//! but with one semantic difference: `QwenImageCausalConv3d.forward` uses
//! `F.pad(..., mode='constant')` — zero padding in the left temporal axis
//! — where Wan 2.1's CausalConv3d repeats the first frame.
//!
//! We share the whole Wan21VaeDecoder implementation and just pass the
//! `PadMode::Zero` variant through to every `CausalConv3d` the decoder
//! owns.
//!
//! The on-disk safetensors uses diffusers naming
//! (`decoder.up_blocks.{i}.resnets.{j}`, etc.); `remap_key` below rewrites
//! to the Wan 2.1 internal layout expected by `Wan21VaeDecoder::from_weights`.
//!
//! Usage:
//!   let decoder = QwenImageVaeDecoder::from_safetensors(path, &device)?;
//!   let image = decoder.decode(&latents)?;  // [B,16,1,H/8,W/8] -> [B,3,1,H,W]

use crate::vae::wan21_vae::{PadMode, Wan21VaeDecoder};
use flame_core::{Result, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub struct QwenImageVaeDecoder {
    inner: Wan21VaeDecoder,
}

impl QwenImageVaeDecoder {
    /// Load from a QwenImage VAE safetensors file (diffusers format).
    pub fn from_safetensors(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let raw = flame_core::serialization::load_file(Path::new(path), device)?;
        let remapped = remap_qwenimage_to_wan21(raw)?;
        let inner = Wan21VaeDecoder::from_weights_with_pad(&remapped, PadMode::Zero, device)?;
        Ok(Self { inner })
    }

    /// Decode latents to image.
    ///
    /// Input: `[B, 16, 1, H/8, W/8]` BF16 normalized latents (normalized by
    /// the same `latents_mean` / `latents_std` constants that Wan 2.1 VAE
    /// uses — the QwenImage diffusers config ships those values verbatim).
    /// Output: `[B, 3, 1, H, W]` BF16 RGB in `[-1, 1]`.
    ///
    /// Calls the underlying Wan 2.1 decoder's `decode_image` path —
    /// `image_mode=true` skips the temporal doubling inside every
    /// upsample3d block, matching diffusers's `feat_cache=None` behavior.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.inner.decode_image(latents)
    }
}

// ---- key remapping (diffusers decoder → wan21 internal) ----

fn remap_qwenimage_to_wan21(qwen: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut wan = HashMap::with_capacity(qwen.len());
    for (key, tensor) in qwen {
        // Skip encoder-side keys.
        if key.starts_with("encoder") || key.starts_with("quant_conv") {
            continue;
        }
        if let Some(wk) = remap_key(&key) {
            wan.insert(wk, tensor);
        } else {
            log::warn!("[QwenImage VAE decoder] unmapped key: {key}");
        }
    }
    Ok(wan)
}

/// Translate diffusers decoder keys to Wan21 internal format.
///
/// Diffusers layout (from HF checkpoint):
///   decoder.conv_in.{weight,bias}                  → decoder.conv1.{...}
///   decoder.mid_block.resnets.{0,1}                → decoder.middle.{0,2}
///   decoder.mid_block.attentions.0                 → decoder.middle.1
///   decoder.up_blocks.{i}.resnets.{j}              → decoder.upsamples.{flat_idx}
///   decoder.up_blocks.{i}.upsamplers.0             → decoder.upsamples.{flat_idx}
///   decoder.norm_out.gamma                         → decoder.head.0.gamma
///   decoder.conv_out.{weight,bias}                 → decoder.head.2.{...}
///   post_quant_conv.{weight,bias}                  → conv2.{...}
///
/// Wan21 upsample index layout (15 blocks total, 3 res + 1 resample per group, 4 groups, last has no resample):
///   group 0 (up_blocks.0): resnets 0,1,2 at indices 0,1,2 + upsamplers.0 at index 3
///   group 1 (up_blocks.1): resnets 0,1,2 at indices 4,5,6 + upsamplers.0 at index 7
///   group 2 (up_blocks.2): resnets 0,1,2 at indices 8,9,10 + upsamplers.0 at index 11
///   group 3 (up_blocks.3): resnets 0,1,2 at indices 12,13,14 (no upsampler)
fn remap_key(key: &str) -> Option<String> {
    if let Some(s) = key.strip_prefix("decoder.conv_in") {
        return Some(format!("decoder.conv1{s}"));
    }
    if let Some(s) = key.strip_prefix("decoder.conv_out") {
        return Some(format!("decoder.head.2{s}"));
    }
    if let Some(s) = key.strip_prefix("decoder.norm_out") {
        return Some(format!("decoder.head.0{s}"));
    }
    if let Some(rest) = key.strip_prefix("decoder.mid_block.resnets.") {
        let dot = rest.find('.')?;
        let idx: usize = rest[..dot].parse().ok()?;
        // resnets 0 → middle.0, resnets 1 → middle.2
        return Some(format!(
            "decoder.middle.{}.{}",
            idx * 2,
            remap_block_field(&rest[dot + 1..])?
        ));
    }
    if let Some(rest) = key.strip_prefix("decoder.mid_block.attentions.0.") {
        return Some(format!("decoder.middle.1.{rest}"));
    }
    if let Some(rest) = key.strip_prefix("decoder.up_blocks.") {
        // Parse: {group}.{resnets|upsamplers}.{j}.{field}
        let dot1 = rest.find('.')?;
        let group: usize = rest[..dot1].parse().ok()?;
        let tail = &rest[dot1 + 1..];

        if let Some(rest2) = tail.strip_prefix("resnets.") {
            let dot2 = rest2.find('.')?;
            let j: usize = rest2[..dot2].parse().ok()?;
            let flat_idx = group * 4 + j;
            return Some(format!(
                "decoder.upsamples.{}.{}",
                flat_idx,
                remap_block_field(&rest2[dot2 + 1..])?
            ));
        }
        if let Some(rest2) = tail.strip_prefix("upsamplers.0.") {
            // The upsampler sits at index (group*4 + 3)
            let flat_idx = group * 4 + 3;
            return Some(format!("decoder.upsamples.{flat_idx}.{rest2}"));
        }
    }
    if let Some(s) = key.strip_prefix("post_quant_conv") {
        return Some(format!("conv2{s}"));
    }
    None
}

fn remap_block_field(field: &str) -> Option<String> {
    if let Some(r) = field.strip_prefix("norm1.") {
        return Some(format!("residual.0.{r}"));
    }
    if let Some(r) = field.strip_prefix("conv1.") {
        return Some(format!("residual.2.{r}"));
    }
    if let Some(r) = field.strip_prefix("norm2.") {
        return Some(format!("residual.3.{r}"));
    }
    if let Some(r) = field.strip_prefix("conv2.") {
        return Some(format!("residual.6.{r}"));
    }
    if let Some(r) = field.strip_prefix("conv_shortcut.") {
        return Some(format!("shortcut.{r}"));
    }
    if field.starts_with("resample.") || field.starts_with("time_conv.") {
        return Some(field.to_string());
    }
    None
}
