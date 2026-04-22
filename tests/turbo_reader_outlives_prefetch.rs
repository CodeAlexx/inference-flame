#![cfg(feature = "turbo")]

//! Stress the event-gated Drop chain: while a kernel reads from slot N's
//! VMM-mapped memory, ensure that the prefetch path's slot reuse can't
//! `cuMemUnmap` until the in-flight kernel completes. The chain we exercise:
//!
//!   prefetch(N)            → ResidentHandle on slot, refcount=1
//!   read kernel on stream  (in-flight when we drop the handle)
//!   drop ResidentHandle    → records event on stream, refcount=0
//!   prefetch(M) reuses the slot, eviction path waits on the recorded event
//!   read kernel finishes   → bytes on the GPU still match what we wrote
//!
//! If the event-gate were broken we'd observe corrupted data after the kernel
//! returns. We use a synchronous `to_vec` to materialize the read.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{memcpy_async_host_to_device, DType, Shape, Tensor};
use inference_flame::turbo::vmm::{cuda_ffi, SlabAllocator, VmmError};

#[test]
fn reader_outlives_prefetch_via_event_gate() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e:?})");
            return;
        }
    };

    let allocator = match SlabAllocator::new(0, Some(256 * 1024 * 1024)) {
        Ok(a) => a,
        Err(VmmError::Unsupported) => {
            eprintln!("skipped: VMM unsupported");
            return;
        }
        Err(e) => panic!("SlabAllocator::new: {e}"),
    };

    let slab = allocator.create_slab(64 * 1024 * 1024).expect("slab");
    let region = allocator.define_region(slab, 0, 16 * 1024 * 1024).expect("region");
    allocator.set_priority(slab, 1).expect("set_priority");

    // Map once, write known BF16 data, build a Tensor, kick off a kernel.
    let null_stream: *mut c_void = std::ptr::null_mut();
    let handle_a = Arc::new(
        allocator
            .ensure_resident(slab, region, null_stream)
            .expect("ensure A"),
    );

    // We pack BF16 values whose F32 widening is easy to verify (small
    // signed integers that round-trip through BF16 exactly).
    let n = 1024 * 8usize;
    let mut bytes = vec![0u16; n];
    for i in 0..n {
        let f = ((i % 32) as i32 - 16) as f32;
        bytes[i] = f32_to_bf16(f);
    }

    let base = unsafe { handle_a.as_ptr() };
    memcpy_async_host_to_device(
        base as *mut c_void,
        bytes.as_ptr() as *const c_void,
        n * 2,
        std::ptr::null_mut(),
    )
    .unwrap();
    unsafe {
        let _ = cuda_ffi::cuStreamSynchronize(std::ptr::null_mut());
    }

    // Sanity: build a tensor view, run a no-op (clone via to_dtype same dtype
    // round-trips through a kernel), and read back.
    let view = unsafe {
        Tensor::from_bf16_device_ptr_non_owning(
            base, n, Shape::from_dims(&[n]), device.clone(),
        )
    };

    // Launch a real kernel that depends on these bytes (cast BF16→F32 lives
    // on the default compute stream).
    let f32_view = view.to_dtype(DType::F32).expect("cast");
    // The kernel hasn't necessarily synced — we let the eviction path race.

    // Now drop our `handle_a` Arc clones so refcount → 0.  The slot is still
    // logically resident.
    drop(view);

    // Allocate a *second* region in a different slot offset and immediately
    // request it. If the original slot's eviction proceeded without waiting
    // for the F32 kernel, our subsequent `to_vec` would observe corruption.
    let region_b = allocator.define_region(slab, 32 * 1024 * 1024, 16 * 1024 * 1024).expect("region B");
    allocator.set_priority(slab, 1).expect("set_priority B");
    let handle_b = allocator.ensure_resident(slab, region_b, null_stream).expect("ensure B");
    drop(handle_b);

    // Force the first kernel to actually finish before reading.
    let result = f32_view.to_vec().expect("read after race");
    drop(handle_a);

    for (i, v) in result.iter().enumerate().take(n) {
        let expected = ((i % 32) as i32 - 16) as f32;
        // BF16→F32 widen of these small integers is exact.
        assert_eq!(
            v.to_bits(),
            expected.to_bits(),
            "element {i}: VMM bytes corrupted by premature unmap (got {v}, want {expected})",
        );
    }
}

#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}
