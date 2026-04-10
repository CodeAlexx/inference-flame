//! HunyuanVideo 3D causal VAE decoder for Kandinsky-5 — pure flame_core.
//!
//! Decode-only path: latents `[B, 16, T, H, W]` → pixels `[B, 3, T_out, H_out, W_out]`.
//! Spatial compression 8x, temporal compression varies by model config.
//!
//! Architecture (latent_ch=16, block_out_channels=[128, 256, 512, 512]):
//!   post_quant_conv: Conv3d(16, 16, 1x1x1)
//!   conv_in: Conv3d(16, 512, 3x3x3, padding=1)
//!   mid_block:
//!     resnets.0: ResBlock3d(512, 512)
//!     attentions.0: Attention(512)
//!     resnets.1: ResBlock3d(512, 512)
//!   up_blocks (reversed block_out_channels, 4 stages):
//!     up_blocks.0: 3x ResBlock3d(512→512) + Upsample3d
//!     up_blocks.1: 3x ResBlock3d(512→512) + Upsample3d
//!     up_blocks.2: 3x ResBlock3d(512→256) + Upsample3d
//!     up_blocks.3: 3x ResBlock3d(256→128) — NO upsample (last)
//!   conv_norm_out: GroupNorm(32, 128)
//!   conv_out: Conv3d(128, 3, 3x3x3, padding=1)
//!
//! Weight key format (diffusers-style):
//!   decoder.post_quant_conv.weight/bias
//!   decoder.conv_in.weight/bias
//!   decoder.mid_block.resnets.{0,1}.norm{1,2}.weight/bias
//!   decoder.mid_block.resnets.{0,1}.conv{1,2}.weight/bias
//!   decoder.mid_block.attentions.0.group_norm.weight/bias
//!   decoder.mid_block.attentions.0.to_q.weight/bias
//!   decoder.mid_block.attentions.0.to_k.weight/bias
//!   decoder.mid_block.attentions.0.to_v.weight/bias
//!   decoder.mid_block.attentions.0.to_out.0.weight/bias
//!   decoder.up_blocks.{n}.resnets.{m}.norm{1,2}.weight/bias
//!   decoder.up_blocks.{n}.resnets.{m}.conv{1,2}.weight/bias
//!   decoder.up_blocks.{n}.resnets.{m}.conv_shortcut.weight/bias  (when in_ch != out_ch)
//!   decoder.up_blocks.{n}.upsamplers.0.conv.weight/bias
//!   decoder.conv_norm_out.weight/bias
//!   decoder.conv_out.weight/bias
//!
//! NOTE: Full 3D causal convolution forward pass is pending — this loads weights
//! and provides the decode interface with a placeholder output. The causal padding
//! logic (temporal left-only pad, no future frame leakage) requires careful porting
//! from the HunyuanVideo reference.

use flame_core::serialization::load_file;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the HunyuanVideo 3D causal VAE.
#[derive(Debug, Clone)]
pub struct HunyuanVaeConfig {
    /// Number of latent channels (input to decoder).
    pub latent_channels: usize,
    /// Number of output channels (RGB = 3).
    pub out_channels: usize,
    /// Latent scaling factor applied before decoding.
    pub scaling_factor: f32,
    /// Channel multipliers per decoder stage (reversed from encoder).
    pub block_out_channels: Vec<usize>,
    /// Number of ResBlocks per stage (actual count = layers_per_block + 1).
    pub layers_per_block: usize,
    /// GroupNorm group count.
    pub norm_num_groups: usize,
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

/// Check if a weight key exists (for optional layers like conv_shortcut).
fn has_key(weights: &Weights, key: &str) -> bool {
    weights.contains_key(key)
}

// ---------------------------------------------------------------------------
// HunyuanVaeDecoder
// ---------------------------------------------------------------------------

/// HunyuanVideo 3D causal VAE decoder for Kandinsky-5.
///
/// Decodes latents `[B, latent_ch, T, H, W]` to pixels `[B, 3, T_out, H_out, W_out]`.
/// Spatial compression is 8x. Temporal compression varies by model.
///
/// Currently loads all decoder weights into a flat HashMap and provides a
/// placeholder decode path. The full 3D causal conv chain will be implemented
/// once the weight format is validated against a reference checkpoint.
pub struct HunyuanVaeDecoder {
    weights: Weights,
    config: HunyuanVaeConfig,
    device: Arc<CudaDevice>,
}

impl HunyuanVaeDecoder {
    /// Load decoder weights from a safetensors file.
    pub fn load(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let weights = load_file(Path::new(path), device)?;
        log::info!("[HunyuanVAE] Loaded {} weight tensors from {}", weights.len(), path);

        // Validate critical weights exist
        let critical_keys = [
            "decoder.conv_in.weight",
            "decoder.conv_out.weight",
            "decoder.conv_norm_out.weight",
        ];
        for key in &critical_keys {
            if !weights.contains_key(*key) {
                log::warn!("[HunyuanVAE] Missing expected key: {key}");
            }
        }

        Ok(Self {
            weights,
            config: HunyuanVaeConfig::default(),
            device: device.clone(),
        })
    }

    /// Load with a custom config.
    pub fn load_with_config(
        path: &str,
        config: HunyuanVaeConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let weights = load_file(Path::new(path), device)?;
        log::info!("[HunyuanVAE] Loaded {} weight tensors from {}", weights.len(), path);
        Ok(Self {
            weights,
            config,
            device: device.clone(),
        })
    }

    /// Access the config.
    pub fn config(&self) -> &HunyuanVaeConfig {
        &self.config
    }

    /// Number of loaded weight tensors.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// List all loaded weight keys (useful for debugging key format).
    pub fn weight_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.weights.keys().map(|s| s.as_str()).collect();
        keys.sort();
        keys
    }

    /// Retrieve a specific weight tensor by key.
    pub fn get_weight(&self, key: &str) -> Result<Tensor> {
        get(&self.weights, key)
    }

    /// Check if a weight key exists.
    pub fn has_weight(&self, key: &str) -> bool {
        has_key(&self.weights, key)
    }

    /// Decode latents to pixels.
    ///
    /// Input:  `[B, 16, T, H/8, W/8]` BF16 latents.
    /// Output: `[B, 3, T_out, H_out, W_out]` BF16 pixels.
    ///
    /// The full decoder chain is:
    /// 1. Scale latents by `1 / scaling_factor`
    /// 2. `post_quant_conv`: Conv3d(16, 16, 1) — identity-ish projection
    /// 3. `conv_in`: Conv3d(16, 512, 3, pad=1)
    /// 4. `mid_block`: ResBlock(512) + Attention(512) + ResBlock(512)
    /// 5. `up_blocks`: 4 stages of ResBlocks + CausalUpsample3d
    /// 6. `conv_norm_out`: GroupNorm(32, 128) + SiLU
    /// 7. `conv_out`: Conv3d(128, 3, 3, pad=1)
    ///
    /// TODO: Implement the full 3D causal conv chain. Currently returns
    /// correctly-shaped zeros for compilation and integration testing.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "HunyuanVAE decode expects 5D [B,C,T,H,W], got {:?}",
                dims
            )));
        }
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        if c != self.config.latent_channels {
            return Err(Error::InvalidInput(format!(
                "HunyuanVAE: expected {} latent channels, got {}",
                self.config.latent_channels, c
            )));
        }

        // Scale latents: x = latents / scaling_factor
        let _x = latents.mul_scalar(1.0 / self.config.scaling_factor)?;

        // TODO: Full 3D causal conv decoder chain:
        //   post_quant_conv → conv_in → mid_block → up_blocks → norm → conv_out
        //
        // The causal convolutions require temporal left-only padding:
        //   For kernel_t=3, pad_t=1: prepend 2 copies of the first frame,
        //   then apply Conv3d with zero temporal padding.
        //
        // For now, return correctly-shaped placeholder for compilation.
        let out_t = t; // temporal upsampling factor depends on model config
        let out_h = h * 8;
        let out_w = w * 8;

        log::info!(
            "[HunyuanVAE] decode: [{b}, {c}, {t}, {h}, {w}] -> [{b}, 3, {out_t}, {out_h}, {out_w}] (placeholder)"
        );

        Tensor::zeros_dtype(
            Shape::from_dims(&[b, 3, out_t, out_h, out_w]),
            DType::BF16,
            self.device.clone(),
        )
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
    }

    #[test]
    fn test_velocity_schedule_sanity() {
        // Verify the scaling_factor reciprocal is reasonable
        let cfg = HunyuanVaeConfig::default();
        let inv_scale = 1.0 / cfg.scaling_factor;
        assert!(inv_scale > 2.0 && inv_scale < 2.2, "inv_scale = {inv_scale}");
    }
}
