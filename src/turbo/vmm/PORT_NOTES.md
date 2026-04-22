# PORT_NOTES — stagehand-vmm v0.2.0 → inference-flame::turbo::vmm

Source: `/home/alex/stagehand-vmm/src/` (commit-pinned mirror at the time of port).

The 8 ported files (`cuda_ffi.rs`, `error.rs`, `slab.rs`, `allocator.rs`, `eviction.rs`,
`prefetch.rs`, `handle.rs`, `lib.rs` → `mod.rs`) are byte-for-byte mirrors except
for the deviations below. `dlpack.rs` and `python.rs` were dropped; this crate
has no Python interop story.

## 1. Module path rename

Every `crate::cuda_ffi`, `crate::slab`, `crate::handle`, etc. became
`crate::turbo::vmm::cuda_ffi`, …. Mechanical find-and-replace; no semantics.

The `lib.rs` file from upstream became `mod.rs` (the inference-flame submodule
entry point) and lost the `#[pymodule]` `stagehand_vmm` registration plus
`pub use python::*` / `pub use dlpack::*` re-exports that referenced dropped
files.

## 2. PyO3 dependency removed

`error.rs` upstream defined `impl From<VmmError> for pyo3::PyErr`. Removed —
inference-flame has no PyO3 dependency. All other code paths use only
`Result<T, VmmError>`.

## 3. New `VmmError::Unsupported` variant

Added a `VmmError::Unsupported` variant so callers can distinguish "device
doesn't support VMM" from a generic CUDA driver error. Wired into
`SlabAllocator::new`: after `cuDeviceGet`, the constructor calls
`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)`
and returns `VmmError::Unsupported` if the result is 0. Upstream skipped this
check because PySlabAllocator was created behind a Python-driven init path
that already gated on supported devices; here the binary needs to detect lack
of support cleanly to log and exit.

A new FFI declaration `cuDeviceGetAttribute` plus the constant
`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102` was added to
`cuda_ffi.rs`. Constant value matches CUDA 12 driver header `cuda.h`.

## 4. `cuStreamSynchronize` FFI added

Added the `cuStreamSynchronize` extern declaration to `cuda_ffi.rs`. Upstream
did not declare it because Python-side `torch.cuda.synchronize()` covered
those cases. The Rust loader needs to await the H2D event from the caller
side without going through cudarc's `result::stream::synchronize` (which
would need a `CudaStream` wrapper we don't always have at the FFI boundary).

## 5. crossbeam-deque replaced by std VecDeque + Mutex

Upstream `prefetch.rs` used `crossbeam_deque::Injector` for the prefetch work
queue. inference-flame has no crossbeam dep and the spec forbids new deps,
so we substituted a `Mutex<VecDeque<PrefetchRequest>>`. The worker is a
single consumer; queue depth is bounded by region count (≤ a few dozen for
Klein 9B). Lock contention is negligible at this scale.

`std::thread::park` / `Thread::unpark` continue to handle the
sleep/wake protocol — upstream already used those, only the queue type
changed.

## 6. Thread name string

The prefetch worker thread name went from `stagehand-vmm-prefetch` to
`inference-flame-vmm-prefetch`. Pure cosmetics; visible in `top`/`gdb`.

## 7. `unsafe impl Sync for ResidentHandle`

`handle.rs` upstream declared only `unsafe impl Send`. Since `await_block`
returns `Arc<TurboBlock>` which holds the handle, and Arc<T> requires
T: Sync, we added `unsafe impl Sync for ResidentHandle`. The CUDA driver API
is documented thread-safe for distinct handle operations, and the
ResidentHandle's only mutable state is the underlying region's atomic
refcount + mutex-protected event slot — both are already Sync-safe. No race
possible from concurrent handle reads (ptr/stream/IDs are immutable).

## 8. No behavior change to hot path or event chain

The fast-path `ensure_resident` still uses RwLock-read + atomic load on
state + atomic increment on refcount + atomic re-check of state, exactly as
upstream. The event-gated Drop in `ResidentHandle::drop` still records the
event on the *consumer's compute stream* (`self.stream`) and stores it
before decrementing refcount, and the eviction path still
`cuEventSynchronize`s `last_use_event` before `cuMemUnmap`. All comments
flagging this as load-bearing were preserved verbatim.
