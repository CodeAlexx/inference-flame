//! BlockOffloader facilitator for HiDream-I1 weights.
//!
//! HiDream-I1 stores blocks under two name prefixes:
//!   - `double_stream_blocks.{i}.block.*`  → block index = `i`
//!   - `single_stream_blocks.{i}.block.*`  → block index = `num_double + i`
//!
//! Everything else (`t_embedder.*`, `p_embedder.*`, `x_embedder.*`,
//! `pe_embedder.*`, `caption_projection.{i}.linear.*`, `final_layer.*`)
//! stays GPU-resident and is loaded via [`load_file_filtered`] in the
//! caller (mirrors the Chroma loader pattern).
//!
//! Reference: `transformer_hidream_image.py:268-302` for the canonical
//! HF diffusers safetensors key layout.

use flame_diffusion::block_offload::BlockFacilitator;

/// Per-block name classifier for HiDream-I1 checkpoints.
pub struct HiDreamI1Facilitator {
    pub num_double: usize,
    pub total_blocks: usize,
}

impl BlockFacilitator for HiDreamI1Facilitator {
    fn block_count(&self) -> usize {
        self.total_blocks
    }
    fn classify_key(&self, key: &str) -> Option<usize> {
        if let Some(rest) = key.strip_prefix("double_stream_blocks.") {
            let idx: usize = rest.split('.').next()?.parse().ok()?;
            return Some(idx);
        }
        if let Some(rest) = key.strip_prefix("single_stream_blocks.") {
            let idx: usize = rest.split('.').next()?.parse().ok()?;
            return Some(self.num_double + idx);
        }
        None
    }
}

/// Shared-weight prefix list for `load_file_filtered`. Anything not under
/// a block prefix lives in CPU-loaded shared state.
pub const SHARED_PREFIXES: [&str; 5] = [
    "t_embedder.",
    "p_embedder.",
    "x_embedder.",
    "caption_projection.",
    "final_layer.",
];

/// Returns true if `key` should be loaded as a shared weight (not via the
/// BlockOffloader).
pub fn is_shared_key(key: &str) -> bool {
    SHARED_PREFIXES.iter().any(|p| key.starts_with(p))
}
