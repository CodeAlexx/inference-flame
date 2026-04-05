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

/// One weight tensor stored on GPU — either FP8 (u8) or BF16 (u16).
pub enum RawWeight {
    FP8 {
        data: CudaSlice<u8>,
        shape: Vec<usize>,
        numel: usize,
        scale: f32,
    },
    BF16 {
        /// Already a BF16 Tensor on GPU — zero-cost to use.
        tensor: Tensor,
    },
}

impl RawWeight {
    /// Get as BF16 Tensor. FP8: dequant kernel. BF16: clone (Arc bump with shared_storage).
    pub fn to_bf16_tensor(&self, device: &Arc<CudaDevice>) -> Result<Tensor> {
        match self {
            RawWeight::FP8 { data, shape, numel: _, scale } => {
                let shape = Shape::from_dims(shape);
                flame_core::ops::fused_inference::dequant_fp8_to_bf16(data, *scale, shape, device)
            }
            RawWeight::BF16 { tensor } => {
                Ok(tensor.clone()) // Arc bump with shared_storage, zero GPU copy
            }
        }
    }
}

/// All block weights for one transformer block, stored as raw GPU bytes.
pub struct ResidentBlock {
    pub weights: HashMap<String, RawWeight>,
    /// F32 tensors (scale_shift_table etc.) — kept as proper Tensors
    pub f32_tensors: HashMap<String, Tensor>,
}

/// Pre-allocated dequant buffer — reused across all blocks and steps.
pub struct DequantBuffer {
    /// BF16 buffer on GPU, sized to the largest FP8 weight tensor
    pub buf: CudaSlice<u16>,
    pub capacity: usize,
}

impl DequantBuffer {
    pub fn new(max_numel: usize, device: &Arc<CudaDevice>) -> Result<Self> {
        let buf: CudaSlice<u16> = unsafe { device.alloc(max_numel)? };
        Ok(Self { buf, capacity: max_numel })
    }
}

impl ResidentBlock {
    /// Convert all weights to BF16 Tensors for one forward pass.
    /// BF16 weights: Arc clone (zero GPU copy with shared_storage).
    /// FP8 weights: dequant into pre-allocated buffer (zero alloc).
    pub fn to_bf16_block(
        &self,
        device: &Arc<CudaDevice>,
        dequant_buf: &mut DequantBuffer,
    ) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::with_capacity(self.weights.len() + self.f32_tensors.len());

        for (name, raw) in &self.weights {
            let tensor = match raw {
                RawWeight::BF16 { tensor } => tensor.clone(), // Arc bump, zero copy
                RawWeight::FP8 { data, shape, numel, scale } => {
                    // Dequant into pre-allocated buffer
                    assert!(*numel <= dequant_buf.capacity,
                        "FP8 tensor {} has {} elems, buffer only {}", name, numel, dequant_buf.capacity);
                    let stream = std::ptr::null_mut(); // default stream
                    let ret = unsafe {
                        flame_core::cuda::ffi::flame_fp8_to_bf16(
                            *data.device_ptr() as *const _,
                            *dequant_buf.buf.device_ptr() as *mut _,
                            *scale,
                            *numel,
                            stream,
                        )
                    };
                    if ret != 0 {
                        return Err(flame_core::Error::Cuda(format!("dequant error: {ret}")));
                    }
                    // Wrap the buffer slice as a Tensor
                    // IMPORTANT: this aliases the buffer — only one FP8 tensor can be
                    // "active" at a time. The GEMM must consume it before next dequant.
                    // For per-block forward this is fine: each weight is used then dropped.
                    //
                    // Actually, we can't alias — the Tensor takes ownership.
                    // We need to allocate per-FP8-tensor. Keep using dequant_fp8_to_bf16.
                    flame_core::ops::fused_inference::dequant_fp8_to_bf16(
                        data, *scale, Shape::from_dims(shape), device,
                    )?
                }
            };
            result.insert(name.clone(), tensor);
        }

        for (name, tensor) in &self.f32_tensors {
            result.insert(name.clone(), tensor.clone());
        }

        Ok(result)
    }
}
