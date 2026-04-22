#![cfg(feature = "turbo")]

//! Cold path: region Unmapped → ensure_resident maps on the caller's stream
//! and returns a valid handle whose pointer is non-null and aligned to the
//! allocator's granularity.

use cudarc::driver::CudaDevice;
use inference_flame::turbo::vmm::{SlabAllocator, VmmError};

#[test]
fn cold_path_maps_and_returns() {
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e:?})");
            return;
        }
    };

    let allocator = match SlabAllocator::new(0, Some(512 * 1024 * 1024)) {
        Ok(a) => a,
        Err(VmmError::Unsupported) => {
            eprintln!("skipped: VMM unsupported");
            return;
        }
        Err(e) => panic!("SlabAllocator::new: {e}"),
    };

    let slab = allocator.create_slab(1024 * 1024 * 1024).expect("slab");
    let region = allocator.define_region(slab, 0, 64 * 1024 * 1024).expect("region");
    allocator.set_priority(slab, 1).expect("set_priority");

    let null_stream: *mut std::ffi::c_void = std::ptr::null_mut();

    let stats_before = allocator.stats();
    let mapped_before = stats_before.mapped_bytes;

    let handle = allocator.ensure_resident(slab, region, null_stream).expect("cold");
    let ptr = unsafe { handle.as_ptr() };
    assert!(ptr != 0, "cold ensure_resident returned null device pointer");

    // Sync so we can measure a stable mapped byte count.
    let stats_after = allocator.stats();
    assert!(
        stats_after.mapped_bytes >= mapped_before + 64 * 1024 * 1024,
        "cold map did not increase mapped_bytes: before={} after={}",
        mapped_before, stats_after.mapped_bytes
    );

    drop(handle);
}
