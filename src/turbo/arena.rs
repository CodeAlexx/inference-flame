//! Klein 9B-sized VMM arena: thin wrapper around SlabAllocator that pre-sizes
//! the virtual reservation and physical pool for a double-buffered block
//! loader.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaStream};

use crate::turbo::vmm::{SlabAllocator, VmmError};

/// Reserved virtual address range. 32 GB is enough for any current Klein
/// variant (9B max ~17 GB total weights, doubled gives slack for alignment).
const VIRTUAL_RESERVE_BYTES: usize = 32 * 1024 * 1024 * 1024;

/// Physical memory pool the allocator may map at any one time. 4 GB covers two
/// ~2 GB block slots (Klein 9B's largest single-stream block is ~750 MB, so
/// even after granularity rounding two slots fit comfortably in 4 GB).
const PHYSICAL_POOL_BYTES: usize = 4 * 1024 * 1024 * 1024;

pub struct VmmArena {
    pub allocator: SlabAllocator,
    pub copy_stream: Arc<CudaStream>,
    pub device: Arc<CudaDevice>,
}

impl VmmArena {
    /// Construct a VMM arena for Klein 9B. Returns `VmmError::Unsupported`
    /// cleanly when the GPU lacks CUDA Virtual Memory Management.
    pub fn new_for_klein9b(
        device: Arc<CudaDevice>,
        copy_stream: Arc<CudaStream>,
    ) -> Result<Self, VmmError> {
        let device_ordinal = device.ordinal() as i32;
        let allocator = SlabAllocator::new(device_ordinal, Some(PHYSICAL_POOL_BYTES))?;
        Ok(Self { allocator, copy_stream, device })
    }

    pub fn virtual_reserve_bytes() -> usize { VIRTUAL_RESERVE_BYTES }
    pub fn physical_pool_bytes() -> usize { PHYSICAL_POOL_BYTES }
}
