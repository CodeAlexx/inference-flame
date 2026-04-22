use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::turbo::vmm::cuda_ffi::{self, CUcontext, CUdevice, CUmemAllocationProp, CUstream};
use crate::turbo::vmm::error::VmmError;
use crate::turbo::vmm::eviction;
use crate::turbo::vmm::handle::ResidentHandle;
use crate::turbo::vmm::prefetch::{PrefetchRequest, PrefetchWorker};
use crate::turbo::vmm::slab::{Region, RegionId, RegionState, Slab, SlabId};

/// Cold-path mutable state protected by Mutex.
pub(crate) struct ColdState {
    pub(crate) mapped_bytes: usize,
    pub(crate) access_clock: u64,
}

/// Shared allocator state. Held behind Arc so handles and the prefetch
/// worker can reference it independently of the SlabAllocator lifetime.
pub struct AllocatorInner {
    #[allow(dead_code)]
    pub(crate) device: CUdevice,
    #[allow(dead_code)]
    pub(crate) context: CUcontext,
    pub(crate) device_ordinal: i32,
    pub(crate) granularity: usize,
    pub(crate) alloc_prop: CUmemAllocationProp,
    pub(crate) vram_ceiling: AtomicUsize,

    /// Slab vector. Read lock for fast path (concurrent readers OK).
    /// Write lock for structural changes (create_slab, define_region, destroy_slab).
    pub(crate) slabs: RwLock<Vec<Option<Slab>>>,

    /// Cold-path serialization (mapped_bytes accounting, access clock).
    pub(crate) cold: Mutex<ColdState>,

    pub(crate) prefetch_worker: Mutex<Option<PrefetchWorker>>,
}

// SAFETY: CUdevice is i32, CUcontext is an opaque driver pointer.
// CUDA driver API is thread-safe. All mutable state is behind RwLock/Mutex.
unsafe impl Send for AllocatorInner {}
unsafe impl Sync for AllocatorInner {}

#[derive(Debug, Clone)]
pub struct SlabStats {
    pub priority: u32,
    pub watermark: usize,
    pub regions: Vec<RegionStats>,
}

#[derive(Debug, Clone)]
pub struct RegionStats {
    pub state: RegionState,
    pub refcount: u32,
    pub offset: usize,
    pub size: usize,
    pub last_access: u64,
}

#[derive(Debug, Clone)]
pub struct AllocatorStats {
    pub total_slabs: usize,
    pub total_regions: usize,
    pub mapped_bytes: usize,
    pub vram_ceiling: usize,
    pub granularity: usize,
    pub slabs: Vec<Option<SlabStats>>,
}

/// Central VMM manager. One per GPU device.
///
/// All public methods take `&self` — concurrent access is safe.
/// The fast path (already-Resident regions) uses only a RwLock read lock
/// and atomic operations. The cold path (mapping, eviction) uses a Mutex.
pub struct SlabAllocator {
    pub(crate) inner: Arc<AllocatorInner>,
}

impl SlabAllocator {
    pub fn new(device_ordinal: i32, ceiling_override: Option<usize>) -> Result<Self, VmmError> {
        let mut device: CUdevice = 0;
        // SAFETY: cuDeviceGet writes to a valid local i32 pointer.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuDeviceGet(&mut device, device_ordinal))?;
        }

        // Probe VMM support before reserving anything. Returns
        // VmmError::Unsupported on devices/drivers that lack the feature so
        // callers can route to a non-turbo fallback cleanly.
        let mut vmm_supported: i32 = 0;
        unsafe {
            cuda_ffi::check(cuda_ffi::cuDeviceGetAttribute(
                &mut vmm_supported,
                cuda_ffi::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device,
            ))?;
        }
        if vmm_supported == 0 {
            return Err(VmmError::Unsupported);
        }

        let mut context: CUcontext = std::ptr::null_mut();
        // SAFETY: cuCtxGetCurrent writes to a valid local pointer.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuCtxGetCurrent(&mut context))?;
        }

        let mut total_mem: usize = 0;
        // SAFETY: cuDeviceTotalMem_v2 writes to a valid local usize pointer.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuDeviceTotalMem_v2(&mut total_mem, device))?;
        }

        let alloc_prop = CUmemAllocationProp::pinned_on_device(device_ordinal);

        let mut granularity: usize = 0;
        // SAFETY: cuMemGetAllocationGranularity writes to a valid local pointer.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuMemGetAllocationGranularity(
                &mut granularity,
                &alloc_prop,
                cuda_ffi::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ))?;
        }

        let two_gb = 2 * 1024 * 1024 * 1024_usize;
        let fifteen_pct = total_mem * 15 / 100;
        let reserve = std::cmp::max(two_gb, fifteen_pct);
        let default_ceiling = total_mem.saturating_sub(reserve);
        let vram_ceiling = ceiling_override.unwrap_or(default_ceiling);

        let inner = Arc::new(AllocatorInner {
            device,
            context,
            device_ordinal,
            granularity,
            alloc_prop,
            vram_ceiling: AtomicUsize::new(vram_ceiling),
            slabs: RwLock::new(Vec::new()),
            cold: Mutex::new(ColdState {
                mapped_bytes: 0,
                access_clock: 0,
            }),
            prefetch_worker: Mutex::new(None),
        });

        // Create prefetch worker
        let worker = PrefetchWorker::new(Arc::clone(&inner), device_ordinal);
        *inner.prefetch_worker.lock().unwrap() = Some(worker);

        Ok(Self { inner })
    }

    /// Reserve a virtual address range and create a slab.
    pub fn create_slab(&self, total_size: usize) -> Result<SlabId, VmmError> {
        let gran = self.inner.granularity;
        let rounded = round_up(total_size, gran);

        let mut base_ptr: cuda_ffi::CUdeviceptr = 0;
        // SAFETY: cuMemAddressReserve writes to base_ptr. addr=0 lets driver choose.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuMemAddressReserve(
                &mut base_ptr, rounded, gran, 0, 0,
            ))?;
        }

        let slab = Slab {
            base_ptr,
            total_size: rounded,
            regions: Vec::new(),
            priority: 0,
            watermark: std::sync::atomic::AtomicUsize::new(0),
        };

        let mut slabs = self.inner.slabs.write().map_err(|_| VmmError::InvalidSlab)?;

        for (idx, slot) in slabs.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(slab);
                return Ok(idx);
            }
        }

        let id = slabs.len();
        slabs.push(Some(slab));
        Ok(id)
    }

    /// Define a region within a slab.
    pub fn define_region(
        &self,
        slab_id: SlabId,
        offset: usize,
        size: usize,
    ) -> Result<RegionId, VmmError> {
        let gran = self.inner.granularity;
        let rounded_size = round_up(size, gran);

        let mut slabs = self.inner.slabs.write().map_err(|_| VmmError::InvalidSlab)?;
        let slab = slabs
            .get_mut(slab_id)
            .and_then(|s| s.as_mut())
            .ok_or(VmmError::InvalidSlab)?;

        let region_id = slab.regions.len();
        slab.regions.push(Region::new(offset, rounded_size));

        if slab.watermark.load(Ordering::Acquire) == 0 {
            slab.watermark.store(slab.regions.len(), Ordering::Release);
        }

        Ok(region_id)
    }

    /// Ensure a region is resident. Returns an RAII handle.
    ///
    /// **Fast path** (region already Resident): RwLock read + atomic ops only.
    /// No Mutex, no CUDA calls — nanoseconds.
    ///
    /// **Cold path** (need to map): Takes Mutex, potentially evicts, calls CUDA.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn ensure_resident(
        &self,
        slab_id: SlabId,
        region_id: RegionId,
        stream: CUstream,
    ) -> Result<ResidentHandle, VmmError> {
        // === FAST PATH: read lock + atomics only ===
        {
            let slabs = self.inner.slabs.read().map_err(|_| VmmError::InvalidSlab)?;
            let slab = slabs
                .get(slab_id)
                .and_then(|s| s.as_ref())
                .ok_or(VmmError::InvalidSlab)?;
            let region = slab.regions.get(region_id).ok_or(VmmError::InvalidRegion)?;

            let watermark = slab.watermark.load(Ordering::Acquire);
            if region_id >= watermark {
                return Err(VmmError::Watermarked);
            }

            if region.state.load(Ordering::Acquire) == RegionState::Resident as u8 {
                // Weight is here. Increment refcount atomically.
                region.refcount.fetch_add(1, Ordering::AcqRel);

                // Double-check state hasn't changed. Eviction uses CAS on state
                // and then checks refcount, so our increment blocks eviction.
                // But we must verify state is still Resident AFTER incrementing.
                if region.state.load(Ordering::Acquire) == RegionState::Resident as u8 {
                    let ptr = slab.base_ptr + region.offset as u64;
                    if let Ok(mut cold) = self.inner.cold.try_lock() {
                        cold.access_clock += 1;
                        if let Ok(mut rm) = region.mutable.try_lock() {
                            rm.last_access = cold.access_clock;
                        }
                    }
                    drop(slabs);
                    return Ok(ResidentHandle::new(
                        Arc::clone(&self.inner),
                        slab_id,
                        region_id,
                        ptr,
                        stream,
                    ));
                }

                // Race lost — eviction CAS'd state between our check and increment.
                // Undo refcount and fall through to cold path.
                region.refcount.fetch_sub(1, Ordering::AcqRel);
            }
        }

        // === COLD PATH: Mutex + CUDA calls ===
        self.ensure_resident_cold(slab_id, region_id, stream)
    }

    fn ensure_resident_cold(
        &self,
        slab_id: SlabId,
        region_id: RegionId,
        stream: CUstream,
    ) -> Result<ResidentHandle, VmmError> {
        let mut cold = self.inner.cold.lock().map_err(|_| VmmError::InvalidSlab)?;
        let slabs = self.inner.slabs.read().map_err(|_| VmmError::InvalidSlab)?;

        let slab = slabs.get(slab_id).and_then(|s| s.as_ref()).ok_or(VmmError::InvalidSlab)?;
        let region = slab.regions.get(region_id).ok_or(VmmError::InvalidRegion)?;

        let watermark = slab.watermark.load(Ordering::Acquire);
        if region_id >= watermark {
            return Err(VmmError::Watermarked);
        }

        cold.access_clock += 1;
        let clock = cold.access_clock;

        let state = region.load_state();
        match state {
            RegionState::Resident => {
                // Another thread mapped it while we were waiting for the Mutex
                region.refcount.fetch_add(1, Ordering::AcqRel);
                if let Ok(mut rm) = region.mutable.lock() {
                    rm.last_access = clock;
                }
                let ptr = slab.base_ptr + region.offset as u64;
                Ok(ResidentHandle::new(
                    Arc::clone(&self.inner), slab_id, region_id, ptr, stream,
                ))
            }
            RegionState::MappedEmpty => {
                region.store_state(RegionState::Resident);
                region.refcount.fetch_add(1, Ordering::AcqRel);
                if let Ok(mut rm) = region.mutable.lock() {
                    rm.last_access = clock;
                }
                let ptr = slab.base_ptr + region.offset as u64;
                Ok(ResidentHandle::new(
                    Arc::clone(&self.inner), slab_id, region_id, ptr, stream,
                ))
            }
            RegionState::Unmapped => {
                // Check if prefetch is inflight
                {
                    let rm = region.mutable.lock().map_err(|_| VmmError::InvalidRegion)?;
                    if let Some(pf_event) = rm.prefetch_event {
                        // SAFETY: event from prefetch worker, stream from caller.
                        unsafe {
                            let _ = cuda_ffi::cuStreamWaitEvent(stream, pf_event, 0);
                        }
                    }
                }
                // Re-check state after potential prefetch completion
                if region.load_state() == RegionState::Resident {
                    region.refcount.fetch_add(1, Ordering::AcqRel);
                    let mut rm = region.mutable.lock().map_err(|_| VmmError::InvalidRegion)?;
                    rm.last_access = clock;
                    if let Some(pf_ev) = rm.prefetch_event.take() {
                        // SAFETY: prefetch event is complete.
                        unsafe { let _ = cuda_ffi::cuEventDestroy_v2(pf_ev); }
                    }
                    let ptr = slab.base_ptr + region.offset as u64;
                    return Ok(ResidentHandle::new(
                        Arc::clone(&self.inner), slab_id, region_id, ptr, stream,
                    ));
                }
                // Clean up stale prefetch event
                {
                    let mut rm = region.mutable.lock().map_err(|_| VmmError::InvalidRegion)?;
                    if let Some(pf_ev) = rm.prefetch_event.take() {
                        // SAFETY: prefetch event is stale.
                        unsafe { let _ = cuda_ffi::cuEventDestroy_v2(pf_ev); }
                    }
                }

                let needed = region.size;
                let base_ptr = slab.base_ptr;
                let offset = region.offset;
                let ceiling = self.inner.vram_ceiling.load(Ordering::Acquire);

                // Eviction if needed
                if cold.mapped_bytes + needed > ceiling {
                    drop(slabs);
                    let mut slabs_w = self.inner.slabs.write().map_err(|_| VmmError::InvalidSlab)?;
                    let result = eviction::evict_for_space(
                        &mut slabs_w, needed, cold.mapped_bytes, ceiling,
                    );
                    match result {
                        Ok(freed) => {
                            cold.mapped_bytes = cold.mapped_bytes.saturating_sub(freed);
                        }
                        Err(_) => {
                            if let Some(s) = slabs_w.get(slab_id).and_then(|s| s.as_ref()) {
                                let wm = s.watermark.load(Ordering::Acquire);
                                if region_id < wm {
                                    s.watermark.store(region_id, Ordering::Release);
                                }
                            }
                            return Err(VmmError::Watermarked);
                        }
                    }
                    drop(slabs_w);
                    let slabs = self.inner.slabs.read().map_err(|_| VmmError::InvalidSlab)?;
                    return self.do_map_region(
                        &slabs, &mut cold, slab_id, region_id, base_ptr, offset, needed, clock, stream,
                    );
                }

                self.do_map_region(
                    &slabs, &mut cold, slab_id, region_id, base_ptr, offset, needed, clock, stream,
                )
            }
        }
    }

    /// Map physical memory for a region. Called under cold Mutex + slabs read lock.
    #[allow(clippy::too_many_arguments)]
    fn do_map_region(
        &self,
        slabs: &[Option<Slab>],
        cold: &mut ColdState,
        slab_id: SlabId,
        region_id: RegionId,
        base_ptr: cuda_ffi::CUdeviceptr,
        offset: usize,
        needed: usize,
        clock: u64,
        stream: CUstream,
    ) -> Result<ResidentHandle, VmmError> {
        let alloc_prop = self.inner.alloc_prop;
        let device_ordinal = self.inner.device_ordinal;

        let mut phys_handle: cuda_ffi::CUmemGenericAllocationHandle = 0;
        // SAFETY: alloc_prop has valid device info, needed is granularity-aligned.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuMemCreate(
                &mut phys_handle, needed, &alloc_prop, 0,
            ))?;
        }

        let map_ptr = base_ptr + offset as u64;

        // SAFETY: map_ptr is within the VA range reserved for this slab.
        let map_result = unsafe {
            cuda_ffi::cuMemMap(map_ptr, needed, 0, phys_handle, 0)
        };
        if map_result != cuda_ffi::CUDA_SUCCESS {
            // SAFETY: cleanup on map failure.
            unsafe { let _ = cuda_ffi::cuMemRelease(phys_handle); }
            return Err(VmmError::CudaError(map_result));
        }

        let access_desc = cuda_ffi::CUmemAccessDesc::readwrite_on_device(device_ordinal);
        // SAFETY: map_ptr/needed define the just-mapped range.
        let access_result = unsafe {
            cuda_ffi::cuMemSetAccess(map_ptr, needed, &access_desc, 1)
        };
        if access_result != cuda_ffi::CUDA_SUCCESS {
            // SAFETY: rollback.
            unsafe {
                let _ = cuda_ffi::cuMemUnmap(map_ptr, needed);
                let _ = cuda_ffi::cuMemRelease(phys_handle);
            }
            return Err(VmmError::CudaError(access_result));
        }

        let slab = slabs.get(slab_id).and_then(|s| s.as_ref()).ok_or(VmmError::InvalidSlab)?;
        let region = slab.regions.get(region_id).ok_or(VmmError::InvalidRegion)?;
        {
            let mut rm = region.mutable.lock().map_err(|_| VmmError::InvalidRegion)?;
            rm.phys_handle = Some(phys_handle);
            rm.last_access = clock;
        }
        region.store_state(RegionState::Resident);
        region.refcount.fetch_add(1, Ordering::AcqRel);
        cold.mapped_bytes += needed;

        Ok(ResidentHandle::new(
            Arc::clone(&self.inner), slab_id, region_id, map_ptr, stream,
        ))
    }

    pub fn prefetch(&self, slab_id: SlabId, region_id: RegionId) {
        let should_enqueue = {
            let slabs = match self.inner.slabs.read() {
                Ok(s) => s,
                Err(_) => return,
            };
            slabs.get(slab_id)
                .and_then(|s| s.as_ref())
                .map(|slab| {
                    let wm = slab.watermark.load(Ordering::Acquire);
                    if region_id >= wm { return false; }
                    match slab.regions.get(region_id) {
                        Some(r) => {
                            if r.load_state() == RegionState::Resident { return false; }
                            if let Ok(rm) = r.mutable.try_lock() {
                                rm.prefetch_event.is_none()
                            } else {
                                false
                            }
                        }
                        None => false,
                    }
                })
                .unwrap_or(false)
        };

        if should_enqueue {
            if let Ok(pw) = self.inner.prefetch_worker.lock() {
                if let Some(ref worker) = *pw {
                    worker.enqueue(PrefetchRequest { slab: slab_id, region: region_id });
                }
            }
        }
    }

    pub fn set_priority(&self, slab_id: SlabId, priority: u32) -> Result<(), VmmError> {
        let mut slabs = self.inner.slabs.write().map_err(|_| VmmError::InvalidSlab)?;

        {
            let slab = slabs.get_mut(slab_id).and_then(|s| s.as_mut())
                .ok_or(VmmError::InvalidSlab)?;
            slab.priority = priority;
        }

        let mut max_priority = 0u32;
        let mut max_slab_id = None;
        for (idx, slot) in slabs.iter().enumerate() {
            if let Some(s) = slot {
                if max_slab_id.is_none() || s.priority > max_priority {
                    max_priority = s.priority;
                    max_slab_id = Some(idx);
                }
            }
        }

        if let Some(id) = max_slab_id {
            if let Some(s) = &slabs[id] {
                let num_regions = s.regions.len();
                s.watermark.store(num_regions, Ordering::Release);
            }
        }

        Ok(())
    }

    pub fn set_vram_ceiling(&self, ceiling_bytes: usize) {
        self.inner.vram_ceiling.store(ceiling_bytes, Ordering::Release);
    }

    /// Insert a `cuStreamWaitEvent` on `stream` against the region's last
    /// recorded reader event, if any. This lets a writer (H2D copy) hold off
    /// until prior compute-stream readers have finished — required when a
    /// slot is being reused for a fresh prefetch while the previously-mapped
    /// region's `last_use_event` is still meaningful.
    ///
    /// No-op if the region has no recorded event.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn wait_for_last_use_event(
        &self,
        slab_id: SlabId,
        region_id: RegionId,
        stream: CUstream,
    ) -> Result<(), VmmError> {
        let slabs = self.inner.slabs.read().map_err(|_| VmmError::InvalidSlab)?;
        let slab = slabs.get(slab_id).and_then(|s| s.as_ref()).ok_or(VmmError::InvalidSlab)?;
        let region = slab.regions.get(region_id).ok_or(VmmError::InvalidRegion)?;
        let rm = region.mutable.lock().map_err(|_| VmmError::InvalidRegion)?;
        if let Some(event) = rm.last_use_event {
            // SAFETY: event was recorded by ResidentHandle::Drop; stream is a
            // caller-supplied CUstream. cuStreamWaitEvent is a GPU-side gate
            // and does not block the host.
            let r = unsafe { cuda_ffi::cuStreamWaitEvent(stream, event, 0) };
            if r != cuda_ffi::CUDA_SUCCESS {
                return Err(VmmError::CudaError(r));
            }
        }
        Ok(())
    }

    pub fn destroy_slab(&self, slab_id: SlabId) -> Result<(), VmmError> {
        let mut cold = self.inner.cold.lock().map_err(|_| VmmError::InvalidSlab)?;
        let mut slabs = self.inner.slabs.write().map_err(|_| VmmError::InvalidSlab)?;

        let mut slab = slabs.get_mut(slab_id).and_then(|s| s.take())
            .ok_or(VmmError::InvalidSlab)?;

        for region in &slab.regions {
            if region.refcount.load(Ordering::Acquire) > 0 {
                slabs[slab_id] = Some(slab);
                return Err(VmmError::SlabNotEmpty);
            }
        }

        let mut bytes_freed = 0usize;
        for region in &mut slab.regions {
            let state = region.load_state();
            if state == RegionState::Resident || state == RegionState::MappedEmpty {
                let unmap_ptr = slab.base_ptr + region.offset as u64;
                // SAFETY: unmap_ptr/size from successful cuMemMap. Refcounts are 0.
                unsafe { let _ = cuda_ffi::cuMemUnmap(unmap_ptr, region.size); }
                let mut rm = region.mutable.lock().unwrap();
                if let Some(phys) = rm.phys_handle.take() {
                    // SAFETY: phys from cuMemCreate.
                    unsafe { let _ = cuda_ffi::cuMemRelease(phys); }
                }
                bytes_freed += region.size;
            }

            let mut rm = region.mutable.lock().unwrap();
            if let Some(event) = rm.last_use_event.take() {
                // SAFETY: event from cuEventCreate.
                unsafe { let _ = cuda_ffi::cuEventDestroy_v2(event); }
            }
            if let Some(event) = rm.prefetch_event.take() {
                // SAFETY: event from cuEventCreate.
                unsafe { let _ = cuda_ffi::cuEventDestroy_v2(event); }
            }
        }

        cold.mapped_bytes = cold.mapped_bytes.saturating_sub(bytes_freed);

        // SAFETY: base_ptr/total_size from cuMemAddressReserve.
        unsafe {
            cuda_ffi::check(cuda_ffi::cuMemAddressFree(slab.base_ptr, slab.total_size))?;
        }

        Ok(())
    }

    /// Snapshot is approximate (Relaxed loads of atomic counters).
    /// Do not drive eviction or watermark decisions from this.
    pub fn stats(&self) -> AllocatorStats {
        let cold = self.inner.cold.lock().ok();
        let slabs = match self.inner.slabs.read() {
            Ok(s) => s,
            Err(_) => {
                return AllocatorStats {
                    total_slabs: 0, total_regions: 0, mapped_bytes: 0,
                    vram_ceiling: 0, granularity: self.inner.granularity, slabs: Vec::new(),
                };
            }
        };

        let mut total_slabs = 0;
        let mut total_regions = 0;
        let mut slab_stats = Vec::new();

        for slot in slabs.iter() {
            match slot {
                Some(slab) => {
                    total_slabs += 1;
                    total_regions += slab.regions.len();
                    let regions: Vec<RegionStats> = slab.regions.iter().map(|r| {
                        let la = r.mutable.lock().map(|rm| rm.last_access).unwrap_or(0);
                        RegionStats {
                            state: r.load_state(),
                            refcount: r.refcount.load(Ordering::Relaxed),
                            offset: r.offset,
                            size: r.size,
                            last_access: la,
                        }
                    }).collect();
                    slab_stats.push(Some(SlabStats {
                        priority: slab.priority,
                        watermark: slab.watermark.load(Ordering::Relaxed),
                        regions,
                    }));
                }
                None => slab_stats.push(None),
            }
        }

        AllocatorStats {
            total_slabs,
            total_regions,
            mapped_bytes: cold.map(|c| c.mapped_bytes).unwrap_or(0),
            vram_ceiling: self.inner.vram_ceiling.load(Ordering::Relaxed),
            granularity: self.inner.granularity,
            slabs: slab_stats,
        }
    }

    pub fn granularity(&self) -> usize {
        self.inner.granularity
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        if let Ok(mut pw) = self.inner.prefetch_worker.lock() {
            if let Some(mut worker) = pw.take() {
                worker.shutdown();
            }
        }
    }
}

impl Drop for AllocatorInner {
    fn drop(&mut self) {
        if let Ok(mut pw) = self.prefetch_worker.lock() {
            if let Some(mut worker) = pw.take() {
                worker.shutdown();
            }
        }

        if let Ok(mut slabs) = self.slabs.write() {
            for slot in slabs.iter_mut() {
                if let Some(slab) = slot.take() {
                    for region in &slab.regions {
                        let state = region.load_state();
                        if state == RegionState::Resident || state == RegionState::MappedEmpty {
                            let unmap_ptr = slab.base_ptr + region.offset as u64;
                            // SAFETY: best-effort cleanup. All handles are dropped
                            // (Arc refcount is 0), so no one references this memory.
                            unsafe { let _ = cuda_ffi::cuMemUnmap(unmap_ptr, region.size); }
                            if let Ok(rm) = region.mutable.lock() {
                                if let Some(phys) = rm.phys_handle {
                                    unsafe { let _ = cuda_ffi::cuMemRelease(phys); }
                                }
                            }
                        }
                        if let Ok(rm) = region.mutable.lock() {
                            if let Some(event) = rm.last_use_event {
                                unsafe { let _ = cuda_ffi::cuEventDestroy_v2(event); }
                            }
                            if let Some(event) = rm.prefetch_event {
                                unsafe { let _ = cuda_ffi::cuEventDestroy_v2(event); }
                            }
                        }
                    }
                    // SAFETY: freeing VA reservation. No handles exist.
                    unsafe { let _ = cuda_ffi::cuMemAddressFree(slab.base_ptr, slab.total_size); }
                }
            }
        }
    }
}

fn round_up(value: usize, align: usize) -> usize {
    debug_assert!(align > 0);
    value.div_ceil(align) * align
}
