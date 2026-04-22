//! Turbo Flame Phase 1: VMM-backed double-buffered block loader for Klein 9B
//! inference.  Feature-gated; off by default.
//!
//! Surface:
//!   * [`vmm`]            — port of stagehand-vmm v0.2.0.
//!   * [`VmmArena`]       — Klein-sized SlabAllocator wrapper.
//!   * [`TurboBlock`]     — `Arc<ResidentHandle>` + BF16View tensors.
//!   * [`TurboBlockLoader`] — `prefetch_block` / `await_block` / `block_count`
//!     / `pinned_bytes`.

pub mod vmm;
pub mod arena;
pub mod block;
pub mod loader;

pub use arena::VmmArena;
pub use block::TurboBlock;
pub use loader::TurboBlockLoader;
pub use vmm::{ResidentHandle, SlabAllocator, VmmError};
