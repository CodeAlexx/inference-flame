use crate::turbo::vmm::error::VmmError;

// --- Type aliases matching CUDA driver API ---

pub type CUdeviceptr = u64;
pub type CUdevice = i32;
pub type CUcontext = *mut std::ffi::c_void;
pub type CUstream = *mut std::ffi::c_void;
pub type CUevent = *mut std::ffi::c_void;
pub type CUresult = u32;
pub type CUmemGenericAllocationHandle = u64;

pub const CUDA_SUCCESS: CUresult = 0;

pub const CU_MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
pub const CU_MEM_LOCATION_TYPE_DEVICE: u32 = 1;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: u32 = 3;
pub const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: u32 = 1;

pub const CU_EVENT_DEFAULT: u32 = 0;
pub const CU_EVENT_DISABLE_TIMING: u32 = 2;

pub const CUDA_ERROR_NOT_READY: CUresult = 600;

// CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED — added so that
// ports outside the Python-binding layer can probe device support before
// reserving any VA range. See CUDA 12 driver header `cuda.h`.
pub const CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: i32 = 102;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemLocation {
    pub type_: u32,
    pub id: i32,
}

/// FFI-compatible allocFlags sub-struct within CUmemAllocationProp.
/// Layout from cuda.h: { u8 compressionType, u8 gpuDirectRDMACapable, u16 usage, u8 reserved[4] }
/// Total: 8 bytes, no padding needed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationPropAllocFlags {
    pub compressionType: u8,
    pub gpuDirectRDMACapable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

/// Matches CUDA driver API's CUmemAllocationProp struct layout exactly.
///
/// From cuda.h (CUDA 12.x):
///   - type: CUmemAllocationType (enum = u32)           offset 0
///   - requestedHandleTypes: CUmemAllocationHandleType   offset 4
///   - location: CUmemLocation { type: u32, id: i32 }   offset 8
///   - win32HandleMetaData: void* (8 bytes on x86_64)    offset 16
///   - allocFlags: { u8, u8, u16, u8[4] } = 8 bytes     offset 24
///   - (implicit padding to align reserved[])            offset 32
///
/// VERIFIED: Rust sizeof = 32, matches C sizeof(CUmemAllocationProp) = 32
/// on CUDA 12.4 (no trailing reserved[] in this version).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationProp {
    pub type_: u32,
    pub requestedHandleTypes: u32,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut std::ffi::c_void,
    pub allocFlags: CUmemAllocationPropAllocFlags,
}

const _: () = assert!(std::mem::size_of::<CUmemAllocationProp>() == 32);

// SAFETY: CUmemAllocationProp contains a *mut c_void (win32HandleMetaData)
// which is always null on Linux. The struct is only passed by-pointer to
// CUDA driver functions and never dereferenced from multiple threads
// simultaneously — it's copied into local variables before use.
unsafe impl Send for CUmemAllocationProp {}
unsafe impl Sync for CUmemAllocationProp {}

impl CUmemAllocationProp {
    pub fn pinned_on_device(device_ordinal: i32) -> Self {
        Self {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: 0,
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_ordinal,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: CUmemAllocationPropAllocFlags {
                compressionType: 0,
                gpuDirectRDMACapable: 0,
                usage: 0,
                reserved: [0; 4],
            },
        }
    }
}

/// Matches CUDA driver API's CUmemAccessDesc struct layout.
/// From cuda.h: { CUmemLocation location (8 bytes), CUmemAccess_flags flags (u32, 4 bytes) }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: u32,
}

impl CUmemAccessDesc {
    pub fn readwrite_on_device(device_ordinal: i32) -> Self {
        Self {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_ordinal,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        }
    }
}

// --- FFI bindings to CUDA driver API (libcuda.so) ---

// SAFETY: These are raw FFI declarations for the CUDA driver API.
// All functions are called only through our `check()` wrapper which
// converts error codes. Callers are responsible for providing valid
// handles and pointers obtained from prior successful CUDA calls.
#[link(name = "cuda")]
extern "C" {
    pub fn cuMemAddressReserve(
        ptr: *mut CUdeviceptr,
        size: usize,
        alignment: usize,
        addr: CUdeviceptr,
        flags: u64,
    ) -> CUresult;

    pub fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult;

    pub fn cuMemCreate(
        handle: *mut CUmemGenericAllocationHandle,
        size: usize,
        prop: *const CUmemAllocationProp,
        flags: u64,
    ) -> CUresult;

    pub fn cuMemMap(
        ptr: CUdeviceptr,
        size: usize,
        offset: usize,
        handle: CUmemGenericAllocationHandle,
        flags: u64,
    ) -> CUresult;

    pub fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult;

    pub fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult;

    pub fn cuMemSetAccess(
        ptr: CUdeviceptr,
        size: usize,
        desc: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult;

    pub fn cuMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const CUmemAllocationProp,
        option: u32,
    ) -> CUresult;

    pub fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const std::ffi::c_void,
        size: usize,
        stream: CUstream,
    ) -> CUresult;

    // Events
    pub fn cuEventCreate(event: *mut CUevent, flags: u32) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;
    pub fn cuEventQuery(event: CUevent) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;

    // Stream
    pub fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: u32) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    // Device
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, device: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(value: *mut i32, attrib: i32, device: CUdevice) -> CUresult;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
}

/// Convert a CUDA result code into a Result, mapping CUDA_SUCCESS to Ok(()).
#[inline]
pub fn check(result: CUresult) -> Result<(), VmmError> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(VmmError::CudaError(result))
    }
}
