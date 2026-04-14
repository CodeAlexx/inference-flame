//! QwenImage VAE Encoder — pure Rust, flame-core.
//!
//! Same architecture as Wan 2.1 VAE (base_dim=96, z_dim=16, dim_mult=[1,2,4,4])
//! but with **zero-padded** causal convolutions (Wan repeats the first frame).
//!
//! Uses diffusers weight key format (encoder.down_blocks.{i}.conv1, etc).
//! Remaps to Wan21 internal format, then patches the CausalConv3d padding mode.
//!
//! Usage:
//!   let encoder = QwenImageVaeEncoder::from_safetensors(path, &device)?;
//!   let latents = encoder.encode(&image_bf16)?;  // [B,3,1,H,W] -> [B,16,1,H/8,W/8]

use crate::vae::wan21_encoder::Wan21VaeEncoder;
use flame_core::{DType, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub struct QwenImageVaeEncoder {
    inner: Wan21VaeEncoder,
}

impl QwenImageVaeEncoder {
    /// Load from a QwenImage VAE safetensors file (diffusers format).
    pub fn from_safetensors(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let raw = flame_core::serialization::load_file(Path::new(path), device)?;
        let remapped = remap_qwenimage_to_wan21(raw)?;
        let inner = Wan21VaeEncoder::from_weights_zero_pad(&remapped, device)?;
        Ok(Self { inner })
    }

    /// Encode image to normalized latents.
    ///
    /// Input: `[B, 3, T, H, W]` BF16 RGB in `[-1, 1]`.
    /// Output: `[B, 16, T', H/8, W/8]` BF16 normalized latents.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        self.inner.encode(image)
    }
}

// ---- key remapping (diffusers -> wan21 internal) ----

fn remap_qwenimage_to_wan21(qwen: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut wan = HashMap::with_capacity(qwen.len());
    for (key, tensor) in qwen {
        if key.starts_with("decoder") || key.starts_with("post_quant_conv") {
            continue;
        }
        if let Some(wk) = remap_key(&key) {
            wan.insert(wk, tensor);
        } else {
            log::warn!("[QwenImage VAE] unmapped key: {key}");
        }
    }
    Ok(wan)
}

fn remap_key(key: &str) -> Option<String> {
    if let Some(s) = key.strip_prefix("encoder.conv_in") {
        return Some(format!("encoder.conv1{s}"));
    }
    if let Some(rest) = key.strip_prefix("encoder.down_blocks.") {
        let dot = rest.find('.')?;
        let (idx, field) = (&rest[..dot], &rest[dot + 1..]);
        return Some(format!("encoder.downsamples.{idx}.{}", remap_block_field(field)?));
    }
    if let Some(rest) = key.strip_prefix("encoder.mid_block.resnets.") {
        let dot = rest.find('.')?;
        let idx: usize = rest[..dot].parse().ok()?;
        return Some(format!("encoder.middle.{}.{}", idx * 2, remap_block_field(&rest[dot + 1..])?));
    }
    if let Some(rest) = key.strip_prefix("encoder.mid_block.attentions.0.") {
        return Some(format!("encoder.middle.1.{rest}"));
    }
    if let Some(s) = key.strip_prefix("encoder.norm_out") {
        return Some(format!("encoder.head.0{s}"));
    }
    if let Some(s) = key.strip_prefix("encoder.conv_out") {
        return Some(format!("encoder.head.2{s}"));
    }
    if let Some(s) = key.strip_prefix("quant_conv") {
        return Some(format!("conv1{s}"));
    }
    None
}

fn remap_block_field(field: &str) -> Option<String> {
    if let Some(r) = field.strip_prefix("norm1.") { return Some(format!("residual.0.{r}")); }
    if let Some(r) = field.strip_prefix("conv1.") { return Some(format!("residual.2.{r}")); }
    if let Some(r) = field.strip_prefix("norm2.") { return Some(format!("residual.3.{r}")); }
    if let Some(r) = field.strip_prefix("conv2.") { return Some(format!("residual.6.{r}")); }
    if let Some(r) = field.strip_prefix("conv_shortcut.") { return Some(format!("shortcut.{r}")); }
    if field.starts_with("resample.") || field.starts_with("time_conv.") {
        return Some(field.to_string());
    }
    None
}
