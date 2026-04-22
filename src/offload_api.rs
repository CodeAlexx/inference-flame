//! `OffloaderApi` — common surface for any block-loader the model's per-block
//! forward loop is generic over.
//!
//! This trait + the `BlockOffloader` impl are always compiled (no
//! `#[cfg(feature = "turbo")]` gate). The TurboBlockLoader impl lives in
//! `src/turbo/api.rs` and IS turbo-gated by being inside `src/turbo/`.
//!
//! The trait surface is intentionally minimal — exactly the four methods both
//! `BlockOffloader` (flame-diffusion) and `TurboBlockLoader` (Phase 1) already
//! expose. No `evict()`, no `set_priority()`, no `stats()`. New models adopt
//! turbo by parameterising their block-loop over `&mut impl OffloaderApi` and
//! adding a thin `forward_with_turbo` wrapper.
//!
//! Pollution contract: trait + foreign-type (`BlockOffloader`) impl live
//! entirely in inference-flame. The orphan rule allows this because
//! `OffloaderApi` is local. `git diff flame-core/ flame-diffusion/` stays
//! empty.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use flame_core::Tensor;
use flame_diffusion::BlockOffloader;

/// Common surface that BlockOffloader and TurboBlockLoader both expose,
/// so a model's block-loop can be generic in which loader it consumes.
pub trait OffloaderApi {
    /// Smart-pointer-like handle returned by await_block. Derefs to a
    /// HashMap so loop bodies can write `block.get(name)` uniformly.
    type Block: Deref<Target = HashMap<String, Tensor>>;

    fn prefetch_block(&mut self, idx: usize) -> anyhow::Result<()>;
    fn await_block(&mut self, idx: usize) -> anyhow::Result<Self::Block>;
    fn block_count(&self) -> usize;
    fn pinned_bytes(&self) -> usize;
}

/// Wrapper: BlockOffloader returns Arc<HashMap<...>>; Arc<HashMap> already
/// Derefs to HashMap, but we wrap to give the trait Block type a single
/// concrete shape across both impls.
pub struct OffloaderBlock(pub Arc<HashMap<String, Tensor>>);

impl Deref for OffloaderBlock {
    type Target = HashMap<String, Tensor>;
    fn deref(&self) -> &HashMap<String, Tensor> { &self.0 }
}

impl OffloaderApi for BlockOffloader {
    type Block = OffloaderBlock;
    fn prefetch_block(&mut self, idx: usize) -> anyhow::Result<()> {
        BlockOffloader::prefetch_block(self, idx)
    }
    fn await_block(&mut self, idx: usize) -> anyhow::Result<Self::Block> {
        BlockOffloader::await_block(self, idx).map(OffloaderBlock)
    }
    fn block_count(&self) -> usize { BlockOffloader::block_count(self) }
    fn pinned_bytes(&self) -> usize { BlockOffloader::pinned_bytes(self) }
}
