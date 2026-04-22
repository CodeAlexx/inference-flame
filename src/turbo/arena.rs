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

    /// Construct a VMM arena for Chroma (8.9B, FLUX-derivative: 19 double +
    /// 38 single blocks, dim 3072). Block byte size is comparable to Klein 9B
    /// (same per-block parameter count magnitude since inner_dim is smaller
    /// but there are separate Q/K/V projections), so the same 4 GB physical
    /// pool / 32 GB virtual reserve comfortably fits two slots.
    pub fn new_for_chroma(
        device: Arc<CudaDevice>,
        copy_stream: Arc<CudaStream>,
    ) -> Result<Self, VmmError> {
        let device_ordinal = device.ordinal() as i32;
        let allocator = SlabAllocator::new(device_ordinal, Some(PHYSICAL_POOL_BYTES))?;
        Ok(Self { allocator, copy_stream, device })
    }

    /// Construct a VMM arena for Qwen-Image-Edit (60-layer DiT, dim 3072).
    /// Per-block size is comparable to the others (within single-block
    /// Klein/Chroma magnitudes); the 4 GB physical pool / 32 GB virtual
    /// reserve is sufficient for two slots after granularity rounding.
    pub fn new_for_qwen_image_edit(
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
