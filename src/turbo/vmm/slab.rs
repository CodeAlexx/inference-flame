use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;

use crate::turbo::vmm::cuda_ffi::{CUdeviceptr, CUevent, CUmemGenericAllocationHandle};

pub type SlabId = usize;
pub type RegionId = usize;

const STATE_UNMAPPED: u8 = 0;
const STATE_MAPPED_EMPTY: u8 = 1;
const STATE_RESIDENT: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionState {
    Unmapped = STATE_UNMAPPED,
    MappedEmpty = STATE_MAPPED_EMPTY,
    Resident = STATE_RESIDENT,
}

impl RegionState {
    pub fn from_u8(v: u8) -> Self {
        match v {
            STATE_UNMAPPED => RegionState::Unmapped,
            STATE_MAPPED_EMPTY => RegionState::MappedEmpty,
            STATE_RESIDENT => RegionState::Resident,
            _ => RegionState::Unmapped,
        }
    }
}

pub struct Slab {
    pub(crate) base_ptr: CUdeviceptr,
    pub(crate) total_size: usize,
    pub(crate) regions: Vec<Region>,
    pub(crate) priority: u32,
    pub(crate) watermark: std::sync::atomic::AtomicUsize,
}

/// Mutable region state protected by a per-region Mutex.
/// The fast path never touches this — only cold path, eviction, and handle Drop.
pub(crate) struct RegionMutable {
    pub(crate) phys_handle: Option<CUmemGenericAllocationHandle>,
    pub(crate) last_access: u64,
    pub(crate) last_use_event: Option<CUevent>,
    pub(crate) prefetch_event: Option<CUevent>,
}

pub struct Region {
    pub(crate) offset: usize,
    pub(crate) size: usize,
    /// Atomic state for lock-free fast-path reads.
    pub(crate) state: AtomicU8,
    pub(crate) refcount: AtomicU32,
    /// Mutable fields behind per-region Mutex. Never touched on the fast path.
    pub(crate) mutable: Mutex<RegionMutable>,
}

impl Region {
    pub fn new(offset: usize, size: usize) -> Self {
        Self {
            offset,
            size,
            state: AtomicU8::new(STATE_UNMAPPED),
            refcount: AtomicU32::new(0),
            mutable: Mutex::new(RegionMutable {
                phys_handle: None,
                last_access: 0,
                last_use_event: None,
                prefetch_event: None,
            }),
        }
    }

    #[inline]
    pub fn load_state(&self) -> RegionState {
        RegionState::from_u8(self.state.load(Ordering::Acquire))
    }

    #[inline]
    pub fn store_state(&self, s: RegionState) {
        self.state.store(s as u8, Ordering::Release);
    }
}

// SAFETY: CUevent and CUmemGenericAllocationHandle are opaque CUDA driver handles.
// The per-region Mutex protects concurrent access to mutable fields.
// Atomic fields (state, refcount, watermark) are inherently thread-safe.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}
unsafe impl Send for Slab {}
unsafe impl Sync for Slab {}
unsafe impl Send for RegionMutable {}
