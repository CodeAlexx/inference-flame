use std::fmt;

#[derive(Debug)]
pub enum VmmError {
    CudaError(u32),
    Watermarked,
    NoEvictableRegions,
    InvalidSlab,
    InvalidRegion,
    SlabNotEmpty,
    Unsupported,
}

impl fmt::Display for VmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmmError::CudaError(code) => write!(f, "CUDA driver error: {code}"),
            VmmError::Watermarked => write!(f, "region is above watermark, use fallback"),
            VmmError::NoEvictableRegions => {
                write!(f, "no evictable regions (all have active refcounts)")
            }
            VmmError::InvalidSlab => write!(f, "invalid slab ID"),
            VmmError::InvalidRegion => write!(f, "invalid region ID"),
            VmmError::SlabNotEmpty => write!(f, "slab has active refcounts, cannot destroy"),
            VmmError::Unsupported => write!(f, "CUDA Virtual Memory Management not supported on this device"),
        }
    }
}

impl std::error::Error for VmmError {}
