//! TurboBlock — the post-`await_block` aggregate of a VMM-resident slot plus a
//! `HashMap<String, Tensor>` of BF16View tensors that point into the slot.
//!
//! The Tensor `BF16View` storage is a non-owning pointer; lifetime is tied to
//! the `Arc<ResidentHandle>` held in this struct. The handle's Drop chain
//! records an event on the consumer's compute stream and only then decrements
//! the slot's refcount, so eviction must wait for kernels to finish before it
//! can unmap the underlying physical memory.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::Tensor;

use crate::turbo::vmm::ResidentHandle;

pub struct TurboBlock {
    pub handle: Arc<ResidentHandle>,
    pub weights: HashMap<String, Tensor>,
}
