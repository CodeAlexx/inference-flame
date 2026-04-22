use std::sync::atomic::Ordering;

use crate::turbo::vmm::cuda_ffi;
use crate::turbo::vmm::error::VmmError;
use crate::turbo::vmm::slab::{RegionState, Slab};

/// Evict resident regions until at least `needed_bytes` can be mapped without
/// exceeding `ceiling`. Uses CAS on region state + refcount double-check for
/// lock-free safety against concurrent ensure_resident fast path.
///
/// Victims are chosen by (slab priority ASC, last_access ASC).
/// Only regions with refcount == 0 AND state == Resident are eligible.
///
/// Called under write lock on slabs (for &mut access to region mutable fields).
pub fn evict_for_space(
    slabs: &mut [Option<Slab>],
    needed_bytes: usize,
    current_mapped: usize,
    ceiling: usize,
) -> Result<usize, VmmError> {
    let must_free = (current_mapped + needed_bytes).saturating_sub(ceiling);
    if must_free == 0 {
        return Ok(0);
    }

    let mut candidates: Vec<(usize, usize, u32, u64)> = Vec::new();
    for (slab_idx, slab_opt) in slabs.iter().enumerate() {
        if let Some(slab) = slab_opt {
            for (region_idx, region) in slab.regions.iter().enumerate() {
                if region.load_state() == RegionState::Resident
                    && region.refcount.load(Ordering::Acquire) == 0
                {
                    let la = region.mutable.lock().map(|rm| rm.last_access).unwrap_or(0);
                    candidates.push((slab_idx, region_idx, slab.priority, la));
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.2.cmp(&b.2).then(a.3.cmp(&b.3)));

    let mut freed: usize = 0;

    for (slab_idx, region_idx, _, _) in &candidates {
        let slab = slabs[*slab_idx].as_mut().unwrap();
        let region = &mut slab.regions[*region_idx];

        // CAS-based eviction protocol:
        // Step 1: Check refcount == 0
        if region.refcount.load(Ordering::Acquire) != 0 {
            continue;
        }

        // Step 2: CAS state from Resident to Unmapped.
        // This "claims" the region for eviction. The fast path will see
        // state != Resident and fall through to the cold path.
        let prev = region.state.compare_exchange(
            RegionState::Resident as u8,
            RegionState::Unmapped as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if prev.is_err() {
            continue;
        }

        // Step 3: Double-check refcount AFTER CAS.
        // If ensure_resident's fast path raced us:
        //   - It incremented refcount BEFORE rechecking state
        //   - It will see state == Unmapped, decrement refcount, and go to cold path
        // So if refcount is still 0, no one has a handle.
        if region.refcount.load(Ordering::Acquire) != 0 {
            // Race lost: restore state so the handle holder sees Resident
            region.state.store(RegionState::Resident as u8, Ordering::Release);
            continue;
        }

        let mut rm = region.mutable.lock().unwrap();

        // Wait for kernels using the reader's recorded compute-stream event
        // before tearing down the physical mapping. This is the load-bearing
        // event-gated drop chain — do not remove.
        if let Some(event) = rm.last_use_event.take() {
            // SAFETY: event from cuEventCreate/cuEventRecord in handle Drop.
            unsafe {
                let _ = cuda_ffi::cuEventSynchronize(event);
                let _ = cuda_ffi::cuEventDestroy_v2(event);
            }
        }

        if let Some(event) = rm.prefetch_event.take() {
            // SAFETY: event from prefetch worker.
            unsafe {
                let _ = cuda_ffi::cuEventSynchronize(event);
                let _ = cuda_ffi::cuEventDestroy_v2(event);
            }
        }

        let phys = match rm.phys_handle.take() {
            Some(h) => h,
            None => continue,
        };
        drop(rm);

        let unmap_ptr = slab.base_ptr + region.offset as u64;
        let size = region.size;

        // SAFETY: unmap_ptr/size from successful cuMemMap. State is Unmapped,
        // refcount is 0, events synchronized. We hold the write lock.
        unsafe {
            let unmap_result = cuda_ffi::cuMemUnmap(unmap_ptr, size);
            if unmap_result != cuda_ffi::CUDA_SUCCESS {
                let mut rm = region.mutable.lock().unwrap();
                rm.phys_handle = Some(phys);
                region.state.store(RegionState::Resident as u8, Ordering::Release);
                continue;
            }
            let _ = cuda_ffi::cuMemRelease(phys);
        }

        let current_wm = slab.watermark.load(Ordering::Acquire);
        slab.watermark.store(current_wm.min(*region_idx), Ordering::Release);

        freed += size;
        if freed >= must_free {
            return Ok(freed);
        }
    }

    if freed >= must_free {
        Ok(freed)
    } else {
        Err(VmmError::NoEvictableRegions)
    }
}
