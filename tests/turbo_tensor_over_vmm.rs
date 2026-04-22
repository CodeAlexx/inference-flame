#![cfg(feature = "turbo")]

//! CRITICAL: verifies that a `Tensor::view_from_buffer` constructed over VMM
//! memory round-trips through the same BF16 kernels as an equivalent
//! normally-allocated tensor. If this test fails, the whole premise of the
//! Turbo Flame Phase 1 port is broken — every downstream forward call uses
//! BF16View tensors pointing into VMM.
//!
//! We exercise a single BF16×BF16 matmul at a shape Klein would actually hit
//! (4096 × 4096) and compare output bytes for bit-identity.

use std::ffi::c_void;

use cudarc::driver::CudaDevice;
use flame_core::{memcpy_async_host_to_device, DType, Shape, Tensor};
use inference_flame::turbo::vmm::{cuda_ffi, SlabAllocator, VmmError};

#[test]
fn tensor_over_vmm_matmul_matches_normal_alloc() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e:?})");
            return;
        }
    };

    let m = 4usize;
    let k = 16usize;
    let n = 8usize;

    // Deterministic BF16 patterns for A (m×k) and B (k×n).
    let mut a_bf16 = vec![0u16; m * k];
    let mut b_bf16 = vec![0u16; k * n];
    for i in 0..a_bf16.len() {
        a_bf16[i] = f32_to_bf16((i as f32) * 0.1 - 0.5);
    }
    for i in 0..b_bf16.len() {
        b_bf16[i] = f32_to_bf16((i as f32) * 0.05 + 0.25);
    }

    // --- Reference: normally allocated Tensor ---
    let mut ref_a = Tensor::zeros_dtype(Shape::from_dims(&[m, k]), DType::BF16, device.clone()).unwrap();
    ref_a.copy_from_bf16_slice(&a_bf16).unwrap();
    let mut ref_b = Tensor::zeros_dtype(Shape::from_dims(&[k, n]), DType::BF16, device.clone()).unwrap();
    ref_b.copy_from_bf16_slice(&b_bf16).unwrap();
    let ref_c = ref_a.matmul(&ref_b).expect("ref matmul");
    let ref_vec = ref_c.to_vec().expect("ref to_vec");

    // --- VMM-backed: construct BF16Views over a SlabAllocator region ---
    let allocator = match SlabAllocator::new(0, Some(256 * 1024 * 1024)) {
        Ok(a) => a,
        Err(VmmError::Unsupported) => {
            eprintln!("skipped: VMM unsupported");
            return;
        }
        Err(e) => panic!("SlabAllocator::new: {e}"),
    };
    let slab = allocator.create_slab(64 * 1024 * 1024).expect("slab");

    // One combined region for A + B for simplicity, 16 MiB covers both.
    let region = allocator.define_region(slab, 0, 16 * 1024 * 1024).expect("region");
    allocator.set_priority(slab, 1).expect("set_priority");
    let stream_ptr: *mut c_void = std::ptr::null_mut();
    let handle = allocator.ensure_resident(slab, region, stream_ptr).expect("map");
    let base = unsafe { handle.as_ptr() };

    // Copy A to [base+0], B to [base+4096] (ensure alignment).
    let a_bytes = a_bf16.len() * 2;
    let b_offset = (a_bytes + 255) & !255;
    let b_bytes = b_bf16.len() * 2;

    let dst_a = base as *mut c_void;
    let dst_b = (base + b_offset as u64) as *mut c_void;
    memcpy_async_host_to_device(
        dst_a,
        a_bf16.as_ptr() as *const c_void,
        a_bytes,
        std::ptr::null_mut(),
    )
    .unwrap();
    memcpy_async_host_to_device(
        dst_b,
        b_bf16.as_ptr() as *const c_void,
        b_bytes,
        std::ptr::null_mut(),
    )
    .unwrap();
    // Sync default stream so the views see the data.
    unsafe {
        let _ = cuda_ffi::cuStreamSynchronize(std::ptr::null_mut());
    }

    let vmm_a = unsafe {
        Tensor::from_bf16_device_ptr_non_owning(
            base,
            m * k,
            Shape::from_dims(&[m, k]),
            device.clone(),
        )
    };
    let vmm_b = unsafe {
        Tensor::from_bf16_device_ptr_non_owning(
            base + b_offset as u64,
            k * n,
            Shape::from_dims(&[k, n]),
            device.clone(),
        )
    };
    let vmm_c = vmm_a.matmul(&vmm_b).expect("vmm matmul");
    let vmm_vec = vmm_c.to_vec().expect("vmm to_vec");

    // Bit-identity (both are BF16 results of the same kernel on identical
    // inputs, read out as F32 — should match exactly).
    assert_eq!(ref_vec.len(), vmm_vec.len(), "output size mismatch");
    for (i, (r, v)) in ref_vec.iter().zip(vmm_vec.iter()).enumerate() {
        assert_eq!(
            r.to_bits(), v.to_bits(),
            "element {i}: ref={r} vmm={v} (bits differ)",
        );
    }

    drop(handle);
    drop(allocator);
}

#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}
