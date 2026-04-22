//! `OffloaderApi` impl for `TurboBlockLoader`.
//!
//! Lives under `src/turbo/` so it only compiles when the `turbo` feature is
//! active. The trait itself is always compiled (see `src/offload_api.rs`).

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use flame_core::Tensor;

use crate::offload_api::OffloaderApi;
use crate::turbo::{TurboBlock, TurboBlockLoader};

/// Wrapper preserving `Arc<TurboBlock>` refcount semantics (Phase 1 safety:
/// Arc keeps the slot's ResidentHandle alive until reader drops) while
/// providing the same `Deref<Target = HashMap>` shape as `OffloaderBlock`.
pub struct TurboAwaited(pub Arc<TurboBlock>);

impl Deref for TurboAwaited {
    type Target = HashMap<String, Tensor>;
    fn deref(&self) -> &HashMap<String, Tensor> { &self.0.weights }
}

impl OffloaderApi for TurboBlockLoader {
    type Block = TurboAwaited;
    fn prefetch_block(&mut self, idx: usize) -> anyhow::Result<()> {
        TurboBlockLoader::prefetch_block(self, idx)
            .map_err(|e| anyhow::anyhow!("turbo prefetch_block: {e}"))
    }
    fn await_block(&mut self, idx: usize) -> anyhow::Result<Self::Block> {
        TurboBlockLoader::await_block(self, idx)
            .map(TurboAwaited)
            .map_err(|e| anyhow::anyhow!("turbo await_block: {e}"))
    }
    fn block_count(&self) -> usize { TurboBlockLoader::block_count(self) }
    fn pinned_bytes(&self) -> usize { TurboBlockLoader::pinned_bytes(self) }
}
