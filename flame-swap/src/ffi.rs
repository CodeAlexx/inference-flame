//! CUDA stream and event FFI — the pieces flame-core is missing.
//!
//! flame-core has `cudaStreamCreate` and `flame_cuda_memcpy_async` but no
//! stream synchronization or event primitives.  We add them here as thin
//! wrappers so the rest of the crate stays safe.

use std::ffi::c_void;
use std::fmt;

// ---------------------------------------------------------------------------
// Raw FFI
// ---------------------------------------------------------------------------

extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
}

// From flame-core cuda/ffi.rs — re-declared here so we don't depend on
// flame-core exposing it publicly.  Identical signature.
extern "C" {
    pub(crate) fn flame_cuda_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: i32, // 1=H2D, 2=D2H, 3=D2D
        stream: *mut c_void,
    ) -> i32;
}

const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;
const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CudaError(pub i32);

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {}", self.0)
    }
}

impl std::error::Error for CudaError {}

#[inline]
fn check(code: i32) -> Result<(), CudaError> {
    if code == 0 { Ok(()) } else { Err(CudaError(code)) }
}

// ---------------------------------------------------------------------------
// Stream — RAII wrapper
// ---------------------------------------------------------------------------

pub struct Stream {
    raw: *mut c_void,
}

// SAFETY: Stream is a handle to a GPU resource.  We ensure single-owner
// semantics via the RAII wrapper (no Clone).  Sends across threads are safe
// because CUDA streams are thread-safe.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a non-blocking stream (won't serialize with the default stream).
    pub fn new() -> Result<Self, CudaError> {
        let mut raw = std::ptr::null_mut();
        unsafe { check(cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING))? };
        Ok(Self { raw })
    }

    /// Block the host until all work on this stream completes.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { check(cudaStreamSynchronize(self.raw)) }
    }

    /// Make this stream wait for an event recorded on another stream.
    pub fn wait_event(&self, event: &Event) -> Result<(), CudaError> {
        unsafe { check(cudaStreamWaitEvent(self.raw, event.raw, 0)) }
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.raw); }
    }
}

// ---------------------------------------------------------------------------
// Event — RAII wrapper
// ---------------------------------------------------------------------------

pub struct Event {
    raw: *mut c_void,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new() -> Result<Self, CudaError> {
        let mut raw = std::ptr::null_mut();
        unsafe { check(cudaEventCreateWithFlags(&mut raw, CUDA_EVENT_DISABLE_TIMING))? };
        Ok(Self { raw })
    }

    /// Record this event on a stream.  Any subsequent `stream.wait_event()`
    /// will block that stream until work up to this point completes.
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        unsafe { check(cudaEventRecord(self.raw, stream.as_raw())) }
    }

    /// Record on the default (null) stream.
    pub fn record_default(&self) -> Result<(), CudaError> {
        unsafe { check(cudaEventRecord(self.raw, std::ptr::null_mut())) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.raw); }
    }
}

// ---------------------------------------------------------------------------
// Async memcpy helper
// ---------------------------------------------------------------------------

/// Asynchronous host-to-device copy on the given stream.
///
/// # Safety
/// `dst` must be a valid device pointer with room for `bytes`.
/// `src` must be a valid **pinned** host pointer (page-locked memory).
pub unsafe fn async_h2d(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check(flame_cuda_memcpy_async(dst, src, bytes, 1, stream.as_raw()))
}
