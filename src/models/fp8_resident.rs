//! FP8 resident model: keep all block weights on GPU as raw bytes.
//! Dequant each block to BF16 on-the-fly during forward pass.
//!
//! Memory budget for LTX-2.3 22B FP8:
//!   FP8 blocks (2-45): ~9GB raw
//!   BF16 blocks (0,1,46,47): ~3GB raw
//!   F32 tables: ~19MB
//!   Globals: ~4GB
//!   Per-block BF16 dequant: ~580MB temporary
//!   Total: ~16-17GB → fits on 24GB

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use flame_core::{DType, Result, Shape, Tensor};

/// One weight tensor stored as raw GPU bytes (FP8 or BF16).
pub struct RawWeight {
    /// Raw bytes on GPU: FP8 = u8, BF16 = u8 (but 2 bytes per element)
    pub data: CudaSlice<u8>,
    pub shape: Vec<usize>,
    pub numel: usize,
    /// FP8 scale (1.0 for BF16 tensors)
    pub scale: f32,
    /// True if FP8 E4M3, false if BF16
    pub is_fp8: bool,
}

impl RawWeight {
    /// Dequant to BF16 Tensor on GPU.
    /// For BF16: zero-copy wrap. For FP8: GPU kernel dequant.
    pub fn to_bf16_tensor(&self, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let shape = Shape::from_dims(&self.shape);
        if self.is_fp8 {
            // GPU-side FP8 → BF16 dequant
            flame_core::ops::fused_inference::dequant_fp8_to_bf16(
                // Need to cast CudaSlice<u8> — it's already u8
                &self.data, self.scale, shape, device,
            )
        } else {
            // BF16: reinterpret u8 as u16
            // The data is already BF16 bytes on GPU — wrap as CudaSlice<u16>
            let bf16_ptr = *self.data.device_ptr() as *const u16;
            // SAFETY: data contains BF16 bytes, numel u16 elements
            let bf16_slice = unsafe {
                // We can't easily convert CudaSlice<u8> to CudaSlice<u16>
                // So we copy into a new u16 allocation
                let out: CudaSlice<u16> = device.alloc(self.numel)?;
                let bytes = self.numel * 2;
                cudarc::driver::result::memcpy_dtod_async(
                    *out.device_ptr(),
                    *self.data.device_ptr() as u64,
                    bytes,
                    std::ptr::null_mut(), // default stream
                ).map_err(|e| flame_core::Error::Cuda(format!("dtod copy: {:?}", e)))?;
                out
            };
            Ok(Tensor::from_bf16_slice_gpu(bf16_slice, shape, Arc::clone(device)))
        }
    }
}

/// All block weights for one transformer block, stored as raw GPU bytes.
pub struct ResidentBlock {
    pub weights: HashMap<String, RawWeight>,
    /// F32 tensors (scale_shift_table etc.) — kept as proper Tensors
    pub f32_tensors: HashMap<String, Tensor>,
}

impl ResidentBlock {
    /// Convert all weights to BF16 Tensors for one forward pass.
    /// Returns a HashMap compatible with load_block_from_weights_static.
    pub fn to_bf16_block(&self, device: &Arc<CudaDevice>) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::with_capacity(self.weights.len() + self.f32_tensors.len());

        for (name, raw) in &self.weights {
            result.insert(name.clone(), raw.to_bf16_tensor(device)?);
        }

        // F32 tensors go in directly (no conversion needed)
        for (name, tensor) in &self.f32_tensors {
            result.insert(name.clone(), tensor.clone());
        }

        Ok(result)
    }
}
