//! Port of stagehand-vmm v0.2.0 — CUDA Virtual Memory Management slabs with
//! event-gated eviction and lock-free hot-path residency. Source mirror:
//! /home/alex/stagehand-vmm/src/.
//!
//! Deviations from upstream are documented in `PORT_NOTES.md`.

#[allow(non_snake_case)]
pub mod cuda_ffi;
pub mod error;
pub mod slab;
pub mod allocator;
pub mod eviction;
pub mod prefetch;
pub mod handle;

pub use allocator::{AllocatorStats, RegionStats, SlabAllocator, SlabStats};
pub use error::VmmError;
pub use handle::ResidentHandle;
pub use slab::{RegionId, RegionState, SlabId};
