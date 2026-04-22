use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::turbo::vmm::allocator::AllocatorInner;
use crate::turbo::vmm::cuda_ffi::{self, CU_EVENT_DISABLE_TIMING};
use crate::turbo::vmm::slab::{RegionId, RegionState, SlabId};

pub struct PrefetchRequest {
    pub slab: SlabId,
    pub region: RegionId,
}

/// MPSC injector. The original stagehand-vmm port uses
/// `crossbeam_deque::Injector` here; we substitute a `Mutex<VecDeque>` to
/// avoid pulling a new dependency into inference-flame. The prefetch worker
/// is a single consumer and the queue depth is bounded by the number of
/// regions, so contention is negligible.
struct Injector<T> {
    queue: Mutex<VecDeque<T>>,
}

impl<T> Injector<T> {
    fn new() -> Self {
        Self { queue: Mutex::new(VecDeque::new()) }
    }

    fn push(&self, item: T) {
        if let Ok(mut q) = self.queue.lock() {
            q.push_back(item);
        }
    }

    fn pop(&self) -> Option<T> {
        self.queue.lock().ok().and_then(|mut q| q.pop_front())
    }
}

pub struct PrefetchWorker {
    pub(crate) injector: Arc<Injector<PrefetchRequest>>,
    pub(crate) thread: Option<std::thread::JoinHandle<()>>,
    pub(crate) shutdown: Arc<AtomicBool>,
    pub(crate) thread_handle: Arc<std::thread::Thread>,
}

impl PrefetchWorker {
    pub fn new(inner: Arc<AllocatorInner>, device_ordinal: i32) -> Self {
        let injector = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let inj_clone = Arc::clone(&injector);
        let shut_clone = Arc::clone(&shutdown);
        let inner_clone = Arc::clone(&inner);

        let thread = std::thread::Builder::new()
            .name("inference-flame-vmm-prefetch".into())
            .spawn(move || {
                prefetch_loop(inj_clone, shut_clone, inner_clone, device_ordinal);
            })
            .expect("failed to spawn prefetch thread");

        let thread_handle = Arc::new(thread.thread().clone());

        PrefetchWorker {
            injector,
            thread: Some(thread),
            shutdown,
            thread_handle,
        }
    }

    pub fn enqueue(&self, request: PrefetchRequest) {
        self.injector.push(request);
        self.thread_handle.unpark();
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.thread_handle.unpark();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PrefetchWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.thread_handle.unpark();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn prefetch_loop(
    injector: Arc<Injector<PrefetchRequest>>,
    shutdown: Arc<AtomicBool>,
    inner: Arc<AllocatorInner>,
    device_ordinal: i32,
) {
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        match injector.pop() {
            Some(req) => {
                handle_prefetch(&inner, &req, device_ordinal);
            }
            None => {
                std::thread::park();
            }
        }
    }
}

fn handle_prefetch(
    inner: &Arc<AllocatorInner>,
    req: &PrefetchRequest,
    device_ordinal: i32,
) {
    let mut cold = match inner.cold.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let slabs = match inner.slabs.read() {
        Ok(s) => s,
        Err(_) => return,
    };

    let (base_ptr, region_size) = {
        let slab = match slabs.get(req.slab).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return,
        };
        let watermark = slab.watermark.load(Ordering::Acquire);
        if req.region >= watermark { return; }
        let region = match slab.regions.get(req.region) {
            Some(r) => r,
            None => return,
        };
        if region.load_state() == RegionState::Resident { return; }
        if let Ok(rm) = region.mutable.try_lock() {
            if rm.prefetch_event.is_some() { return; }
        } else {
            return;
        }
        (slab.base_ptr, region.size)
    };

    let needed = region_size;
    let ceiling = inner.vram_ceiling.load(Ordering::Acquire);

    if cold.mapped_bytes + needed > ceiling {
        drop(slabs);
        let mut slabs_w = match inner.slabs.write() {
            Ok(s) => s,
            Err(_) => return,
        };
        let result = crate::turbo::vmm::eviction::evict_for_space(
            &mut slabs_w, needed, cold.mapped_bytes, ceiling,
        );
        match result {
            Ok(freed) => cold.mapped_bytes = cold.mapped_bytes.saturating_sub(freed),
            Err(_) => return,
        }
        drop(slabs_w);
        let slabs = match inner.slabs.read() {
            Ok(s) => s,
            Err(_) => return,
        };
        do_prefetch_map(inner, &slabs, &mut cold, req, base_ptr, needed, device_ordinal);
        return;
    }

    do_prefetch_map(inner, &slabs, &mut cold, req, base_ptr, needed, device_ordinal);
}

fn do_prefetch_map(
    inner: &AllocatorInner,
    slabs: &[Option<crate::turbo::vmm::slab::Slab>],
    cold: &mut crate::turbo::vmm::allocator::ColdState,
    req: &PrefetchRequest,
    base_ptr: crate::turbo::vmm::cuda_ffi::CUdeviceptr,
    needed: usize,
    device_ordinal: i32,
) {
    let region_offset = {
        let slab = match slabs.get(req.slab).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return,
        };
        let region = match slab.regions.get(req.region) {
            Some(r) => r,
            None => return,
        };
        if region.load_state() == RegionState::Resident { return; }
        if let Ok(rm) = region.mutable.try_lock() {
            if rm.prefetch_event.is_some() { return; }
        } else {
            return;
        }
        region.offset
    };

    let alloc_prop = inner.alloc_prop;
    let mut phys_handle: cuda_ffi::CUmemGenericAllocationHandle = 0;
    let map_ptr = base_ptr + region_offset as u64;

    // SAFETY: alloc_prop has valid device info, needed is granularity-aligned.
    let result = unsafe { cuda_ffi::cuMemCreate(&mut phys_handle, needed, &alloc_prop, 0) };
    if result != cuda_ffi::CUDA_SUCCESS { return; }

    // SAFETY: map_ptr is within the VA range reserved for this slab.
    let result = unsafe { cuda_ffi::cuMemMap(map_ptr, needed, 0, phys_handle, 0) };
    if result != cuda_ffi::CUDA_SUCCESS {
        unsafe { let _ = cuda_ffi::cuMemRelease(phys_handle); }
        return;
    }

    let access_desc = cuda_ffi::CUmemAccessDesc::readwrite_on_device(device_ordinal);
    // SAFETY: map_ptr/needed define the just-mapped range.
    let result = unsafe { cuda_ffi::cuMemSetAccess(map_ptr, needed, &access_desc, 1) };
    if result != cuda_ffi::CUDA_SUCCESS {
        unsafe {
            let _ = cuda_ffi::cuMemUnmap(map_ptr, needed);
            let _ = cuda_ffi::cuMemRelease(phys_handle);
        }
        return;
    }

    let mut event: cuda_ffi::CUevent = std::ptr::null_mut();
    // SAFETY: cuEventCreate writes to a valid local pointer.
    let ev_ok = unsafe {
        cuda_ffi::cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING)
    } == cuda_ffi::CUDA_SUCCESS;

    if ev_ok {
        // SAFETY: event just created, null stream = default stream.
        unsafe { let _ = cuda_ffi::cuEventRecord(event, std::ptr::null_mut()); }
    }

    let updated = if let Some(slab) = slabs.get(req.slab).and_then(|s| s.as_ref()) {
        if let Some(region) = slab.regions.get(req.region) {
            if let Ok(mut rm) = region.mutable.lock() {
                rm.phys_handle = Some(phys_handle);
                if ev_ok { rm.prefetch_event = Some(event); }
                region.store_state(RegionState::Resident);
                true
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    if updated {
        cold.mapped_bytes += needed;
    } else {
        unsafe {
            let _ = cuda_ffi::cuMemUnmap(map_ptr, needed);
            let _ = cuda_ffi::cuMemRelease(phys_handle);
        }
        if ev_ok {
            unsafe { let _ = cuda_ffi::cuEventDestroy_v2(event); }
        }
    }
}
