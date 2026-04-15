//! Wan2.2 TI2V-5B transformer — thin wrapper over [`Wan22Dit`].
//!
//! The TI2V-5B DiT is a single-expert variant of the same `WanModel` class
//! that A14B uses. Every block — self-attention, cross-attention, FFN,
//! AdaLN time modulation, 3-axis RoPE, patch embedding, head — is
//! structurally identical. Only the dimensions differ:
//!
//! | Param           | A14B   | TI2V-5B |
//! | --------------- | -----: | ------: |
//! | dim             | 5120   | 3072    |
//! | num_layers      | 40     | 30      |
//! | num_heads       | 24×... | 24      |
//! | head_dim        | 128    | 128     |
//! | ffn_dim         | 13824  | 14336   |
//! | patch_size      | (1,2,2)| (1,2,2) |
//! | in/out channels | 16     | 16      |
//! | text_len / _dim | 512/4096 | 512/4096 |
//!
//! Because all per-tensor shapes in `Wan22Dit` are derived from `config.*`
//! and read dynamically from the loaded weights, reusing it for 5B is as
//! simple as passing the right [`Wan22Config`].
//!
//! ## Checkpoint format
//! TI2V-5B ships as 7 safetensors shards with a `diffusion_pytorch_model.
//! safetensors.index.json` manifest. We resolve shard paths via
//! [`ShardIndex`], filter to the DiT namespace (anything that starts with
//! `blocks.`, `patch_embedding.`, `text_embedding.`, `time_embedding.`,
//! `time_projection.`, or `head.`), and hand the resulting paths to
//! `Wan22Dit::load_with_config`, which already knows how to merge them via
//! `BlockOffloader` and `load_file_filtered`.

use flame_core::{CudaDevice, Result, Tensor};
use std::path::Path;
use std::sync::Arc;

use super::super::wan22_dit::{Wan22Config, Wan22Dit};
use super::shard_loader::ShardIndex;

/// Wan2.2 TI2V-5B static configuration (from `wan/configs/wan_ti2v_5B.py`).
#[derive(Debug, Clone)]
pub struct WanConfig {
    pub num_layers: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: [usize; 3],
    pub freq_dim: usize,
    pub text_dim: usize,
    pub text_len: usize,
    pub eps: f32,
    pub rope_theta: f64,
    // T2V sampling defaults (carried for reference; actual sampler reads its own).
    pub sample_fps: u32,
    pub sample_shift: f32,
    pub sample_steps: usize,
    pub sample_guide_scale: f32,
}

impl WanConfig {
    /// TI2V-5B official config.
    ///
    /// Note on `in_channels`/`out_channels`: the Wan2.2 VAE for TI2V-5B
    /// outputs **48-channel** latents (`z_dim=48` in `vae2_2.py::WanVAE_`),
    /// not 16. The DiT's `patch_embedding` takes 48 input channels — this
    /// is also what `Wan2_2TI2V` assumes when it computes
    /// `target_shape = (self.vae.model.z_dim, ...)`. The original job
    /// description had 16 copied from the Wan 2.1 VAE; correcting here.
    pub fn ti2v_5b() -> Self {
        Self {
            num_layers: 30,
            dim: 3072,
            ffn_dim: 14336,
            num_heads: 24,
            head_dim: 128, // 24 * 128 == 3072
            in_channels: 48,
            out_channels: 48,
            patch_size: [1, 2, 2],
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            rope_theta: 10000.0,
            sample_fps: 24,
            sample_shift: 5.0,
            sample_steps: 50,
            sample_guide_scale: 5.0,
        }
    }

    /// T2V-14B official config (Wan 2.2).
    ///
    /// Uses Wan 2.1 VAE (16-channel, stride 4×8×8) — NOT the 48-channel
    /// Wan 2.2 VAE used by TI2V-5B.
    pub fn t2v_14b() -> Self {
        Self {
            num_layers: 40,
            dim: 5120,
            ffn_dim: 13824,
            num_heads: 40,
            head_dim: 128, // 40 * 128 == 5120
            in_channels: 16,
            out_channels: 16,
            patch_size: [1, 2, 2],
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            rope_theta: 10000.0,
            sample_fps: 24,
            sample_shift: 12.0,
            sample_steps: 40,
            sample_guide_scale: 5.0,
        }
    }

    /// I2V-14B official config (Wan 2.2).
    pub fn i2v_14b() -> Self {
        let mut c = Self::t2v_14b();
        c.in_channels = 32; // 16 video + 16 image conditioning
        c.sample_shift = 12.0;
        c
    }
}

impl From<WanConfig> for Wan22Config {
    fn from(c: WanConfig) -> Self {
        Wan22Config {
            num_layers: c.num_layers,
            dim: c.dim,
            ffn_dim: c.ffn_dim,
            num_heads: c.num_heads,
            head_dim: c.head_dim,
            in_channels: c.in_channels,
            out_channels: c.out_channels,
            patch_size: c.patch_size,
            freq_dim: c.freq_dim,
            text_dim: c.text_dim,
            text_len: c.text_len,
            eps: c.eps,
            rope_theta: c.rope_theta,
        }
    }
}

/// TI2V-5B DiT wrapper. Delegates forward to [`Wan22Dit`].
pub struct WanTransformer {
    dit: Wan22Dit,
    config: WanConfig,
}

impl WanTransformer {
    /// Load the 5B DiT from a checkpoint directory containing a sharded
    /// `diffusion_pytorch_model` + index.json.
    pub fn load_ti2v_5b(
        ckpt_dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = WanConfig::ti2v_5b();
        let wan22 = Wan22Config::from(config.clone());

        // Resolve shards from the index.json.
        let index = ShardIndex::load_default(ckpt_dir)?;

        // DiT keys live under these prefixes.
        let is_dit = |k: &str| {
            k.starts_with("blocks.")
                || k.starts_with("patch_embedding.")
                || k.starts_with("text_embedding.")
                || k.starts_with("time_embedding.")
                || k.starts_with("time_projection.")
                || k.starts_with("head.")
        };
        let shard_paths = index.shards_for(is_dit);
        if shard_paths.is_empty() {
            return Err(flame_core::Error::InvalidInput(format!(
                "No DiT shards found in {}",
                ckpt_dir.display()
            )));
        }
        log::info!(
            "[WanTransformer] Loading TI2V-5B DiT from {} shards",
            shard_paths.len()
        );

        // Wan22Dit::load_with_config takes &[&str].
        let path_strings: Vec<String> = shard_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let path_refs: Vec<&str> = path_strings.iter().map(|s| s.as_str()).collect();

        let dit = Wan22Dit::load_with_config(&path_refs, wan22, device)?;
        Ok(Self { dit, config })
    }

    pub fn config(&self) -> &WanConfig {
        &self.config
    }

    /// Forward a latent through the DiT. Mirrors `Wan22Dit::forward`.
    ///
    /// * `x` — latent `[in_channels, F, H, W]` (post-patchify grid is
    ///   `(F/1, H/2, W/2)` — see [`WanConfig::patch_size`]).
    /// * `timestep` — scalar in `[0, 1000]` (diffusers-style).
    /// * `context` — UMT5 hidden states `[1, L<=512, 4096]` BF16.
    /// * `seq_len` — padded patch-sequence length (must be `>= F*H/2*W/2`).
    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: f32,
        context: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.dit.forward(x, timestep, context, seq_len)
    }
}
