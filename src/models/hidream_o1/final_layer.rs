//! `FinalLayer` — final projection from hidden states back to image patches.
//!
//! Reference: `qwen3_vl_transformers.py:962-976`.
//!
//! ```python
//! class FinalLayer(nn.Module):
//!     def __init__(self, config, hidden_size, patch_size, out_channels):
//!         self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
//!     def forward(self, x, adaln_input=None):
//!         x = self.linear(x)
//!         return x
//! ```
//!
//! **Note (vs. DiT-style final layers)**: the Python signature includes an
//! `adaln_input` argument, but the body **ignores it** — there is no adaLN
//! modulation on the final projection in HiDream-O1. The `adaln_input`
//! parameter is a vestige (likely a copied DiT skeleton). Verified by
//! reading lines 974-976 directly: `forward` does only `self.linear(x)`.
//!
//! Conditioning happens upstream via the `<|tms_token|>` embedding
//! replacement (`qwen3_vl_transformers.py:1443-1452`), not inside `FinalLayer`.
//! Phase 2b will not need to wire any modulation here.

use std::sync::Arc;

use flame_core::nn::Linear;
use flame_core::{CudaDevice, Result, Tensor};

use super::HiDreamO1Config;

/// Final projection: `[B, L, hidden_size] → [B, L, P*P*C]`.
///
/// In the full HiDream-O1 forward only the L positions corresponding to
/// `vinput_mask == True` are kept (`qwen3_vl_transformers.py:1525-1526` /
/// `pipeline.py:325-328`). This module is unaware of that mask — it
/// projects every hidden state, and the caller does the gather.
pub struct FinalLayer {
    pub linear: Linear,
    pub patch_size: usize,
    pub out_channels: usize,
}

impl FinalLayer {
    /// Instantiate. Output features = `patch_size² * out_channels` (3072 for 32²×3).
    pub fn new(config: &HiDreamO1Config, device: &Arc<CudaDevice>) -> Result<Self> {
        let out_dim = config.patch_size * config.patch_size * config.patch_in_channels;
        let linear = Linear::new(config.hidden_size, out_dim, /*bias=*/ true, device)?;
        Ok(Self {
            linear,
            patch_size: config.patch_size,
            out_channels: config.patch_in_channels,
        })
    }

    /// Forward. Input `[B, L, H] → [B, L, P*P*C]`.
    ///
    /// `_cond` is accepted but ignored — it's reserved for a hypothetical
    /// adaLN extension (the Python signature has it, but the body ignores
    /// it, see `qwen3_vl_transformers.py:974-976`). Pass `None` from
    /// Phase 2b's pipeline; revisit only if a future HiDream variant
    /// actually wires modulation here.
    pub fn forward(&self, hidden_states: &Tensor, _cond: Option<&Tensor>) -> Result<Tensor> {
        self.linear.forward(hidden_states)
    }
}
