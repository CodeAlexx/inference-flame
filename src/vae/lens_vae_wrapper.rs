//! Lens-specific VAE wrapper around the existing FLUX.2 decoder (`klein_vae`).
//!
//! Mirrors `lens.pipeline.LensPipeline._decode` (vendor-refs/Lens/lens/pipeline.py:367-388):
//!
//! ```text
//!   1. rearrange "b (h w) (c p1 p2) -> b c (h p1) (w p2)"  // [B, S, 128] -> [B, 32, 2h, 2w]
//!   2. _patchify_latents                                    // [B, 32, 2h, 2w] -> [B, 128, h, w]
//!   3. x = x * std + mean                                   // inverse BatchNorm (rewritten from
//!                                                              the Python `x / scale - shift`)
//!   4. _unpatchify_latents                                   // [B, 128, h, w] -> [B, 32, 2h, 2w]
//!   5. self.vae.decode(x).sample                            // -> [B, 3, H, W] in [-1, 1]
//! ```
//!
//! Implementation note. `klein_vae::KleinVaeDecoder::decode` already does steps 3, 4, 5
//! internally when given a `[B, 128, h, w]` tensor (inverse BN on 128 channels then
//! `unpatchify_latents` then the diffusers decoder). So this wrapper only ports steps 1+2,
//! then hands off to the existing decoder.
//!
//! The eps used by `klein_vae` was historically `1e-5` (copied from a Python upstream bug
//! in `pipeline_ernie_image.py:367`); the Lens config and the encoder side of the same
//! file both say `1e-4`. As of this port the decoder also uses `1e-4`. The two VAE files
//! on disk (`Models/vaes/flux2-vae.safetensors` used by Klein and
//! `microsoft_lens/vae/diffusion_pytorch_model.safetensors` used by Lens) are
//! md5-identical, so the fix applies cleanly to both consumers.

use std::collections::HashMap;

use flame_core::device::Device;
use flame_core::{Error, Result, Tensor};

use crate::vae::klein_vae::{patchify_latents, KleinVaeDecoder};

/// Lens VAE wrapper.
///
/// Owns a `KleinVaeDecoder` and adds the pipeline's pre-decode rearrange + patchify
/// so callers can pass the DiT output (`[B, S, 128]`) directly.
pub struct LensVaeWrapper {
    decoder: KleinVaeDecoder,
}

impl LensVaeWrapper {
    /// Load decoder weights from a HashMap (diffusers-format keys under `vae/`).
    pub fn load(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let decoder = KleinVaeDecoder::load(weights, device)?;
        Ok(Self { decoder })
    }

    /// Full pipeline-equivalent decode.
    ///
    /// `dit_output`: `[B, S, 128]` BF16, the DiT's `proj_out` output where `S = h*w`.
    /// `latent_h`, `latent_w`: the post-patchify lattice (so `S == latent_h * latent_w`).
    ///
    /// Returns `[B, 3, latent_h * 16, latent_w * 16]` BF16 in `[-1, 1]`. The PIL/RGB
    /// uint8 conversion lives in the inference bin, not here.
    pub fn decode(
        &self,
        dit_output: &Tensor,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Tensor> {
        let dims = dit_output.shape().dims();
        if dims.len() != 3 {
            return Err(Error::InvalidOperation(format!(
                "LensVaeWrapper::decode expects [B, S, 128], got rank {}",
                dims.len()
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        if d != 128 {
            return Err(Error::InvalidOperation(format!(
                "LensVaeWrapper::decode expects last-dim 128, got {d}"
            )));
        }
        if s != latent_h * latent_w {
            return Err(Error::InvalidOperation(format!(
                "LensVaeWrapper::decode: S={s} != latent_h*latent_w = {}*{} = {}",
                latent_h,
                latent_w,
                latent_h * latent_w
            )));
        }

        // Step 1: rearrange "b (h w) (c p1 p2) -> b c (h p1) (w p2)" with c=32, p1=p2=2.
        // Semantically: treat input as [B, h, w, c=32, p1=2, p2=2], output as [B, c, h*p1, w*p2].
        //   - view [B, S, 128] -> [B, h, w, 32, 2, 2]   (rebuild the 6 dims)
        //   - permute (B, h, w, c, p1, p2) -> (B, c, h, p1, w, p2): axes [0, 3, 1, 4, 2, 5]
        //   - reshape -> [B, 32, 2h, 2w]
        let x = dit_output.reshape(&[b, latent_h, latent_w, 32, 2, 2])?;
        let x = x.permute(&[0, 3, 1, 4, 2, 5])?;
        let x = x.reshape(&[b, 32, latent_h * 2, latent_w * 2])?;

        // Step 2: patchify "b c h w -> b (c p1 p2) (h/2) (w/2)" -> [B, 128, h, w].
        // Reuse the helper that the encoder side already uses; this guarantees the
        // patchify/unpatchify pair (the encoder's patchify + klein_vae::decode's
        // internal unpatchify_latents) is exactly inverse on the BN-channel layout.
        let x = patchify_latents(&x)?;

        // Steps 3+4+5: klein_vae::decode does:
        //   - inverse BatchNorm: z * sqrt(running_var + eps) + running_mean   (eps = 1e-4)
        //   - unpatchify_latents: [B, 128, h, w] -> [B, 32, 2h, 2w]
        //   - post_quant_conv + conv_in + mid_block + up_blocks + conv_norm_out + conv_out
        // Output: [B, 3, latent_h*16, latent_w*16] in [-1, 1].
        self.decoder.decode(&x)
    }

    /// Direct access to the inner decoder (for callers that already have
    /// `[B, 128, h, w]` latents and don't need the rearrange/patchify dance).
    pub fn inner(&self) -> &KleinVaeDecoder {
        &self.decoder
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::serialization::load_file;
    use std::path::Path;

    /// CUDA-gated load probe: verify the Lens VAE safetensors loads cleanly via klein_vae.
    /// Run with: `cargo test --release -p inference-flame --lib vae::lens_vae_wrapper -- --ignored`.
    #[test]
    #[ignore]
    fn load_lens_vae_safetensors() {
        let path = "/home/alex/.serenity/models/microsoft_lens/vae/diffusion_pytorch_model.safetensors";
        let device = Device::cuda(0).expect("CUDA device 0");
        let cuda_arc = device.cuda_device_arc();
        let weights = load_file(Path::new(path), &cuda_arc).expect("load safetensors");
        let total_keys = weights.len();
        println!("Loaded {} tensors from {}", total_keys, path);

        let wrapper = LensVaeWrapper::load(&weights, &device).expect("LensVaeWrapper::load");
        let _ = wrapper.inner();
        println!(
            "LensVaeWrapper::load OK (decoder ready). Total keys in file: {}",
            total_keys
        );
    }
}
