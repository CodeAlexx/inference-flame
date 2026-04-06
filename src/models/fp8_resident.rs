//! FP8 resident model: keep all block weights on GPU as raw bytes.
//! Dequant each block to BF16 on-the-fly during forward pass.
//! Weights are pre-transposed so the forward pass does zero transposes.
//!
//! Memory budget for LTX-2.3 22B FP8:
//!   FP8 blocks (2-45): ~9GB raw
//!   BF16 blocks (0,1,46,47): ~3GB raw (pre-transposed at load)
//!   F32 tables: ~19MB
//!   Globals: ~4GB
//!   Per-block BF16 dequant: ~580MB temporary
//!   Total: ~16-17GB → fits on 24GB

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use flame_core::{DType, Result, Shape, Tensor};

/// Returns true if this key is a 2D weight matrix that needs transposing
/// from [out_features, in_features] → [in_features, out_features].
/// Biases (1D), norms (1D), and scale_shift_tables (F32) don't need it.
pub fn needs_transpose(key: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && key.ends_with(".weight")
        && !key.contains("norm_q") && !key.contains("norm_k")
        && !key.contains("q_norm") && !key.contains("k_norm")
        && !key.contains("scale_shift")
        && !key.contains("norm1") && !key.contains("norm2") && !key.contains("norm3")
}

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
    /// **All 2D weight matrices are pre-transposed** to [in, out] layout
    /// so the forward pass can skip 576 GPU transpose kernels per step.
    ///
    /// BF16 weights: already pre-transposed at load time → Arc clone (zero copy).
    /// FP8 weights: dequant + transpose (two kernels, but only once per step).
    pub fn to_bf16_block(
        &self,
        device: &Arc<CudaDevice>,
        _dequant_buf: &mut DequantBuffer,
    ) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::with_capacity(self.weights.len() + self.f32_tensors.len());

        for (name, raw) in &self.weights {
            let tensor = match raw {
                RawWeight::BF16 { tensor } => tensor.clone(), // Already pre-transposed at load
                RawWeight::FP8 { data, shape, numel: _, scale } => {
                    // Dequant to BF16
                    let t = flame_core::ops::fused_inference::dequant_fp8_to_bf16(
                        data, *scale, Shape::from_dims(shape), device,
                    )?;
                    // Transpose 2D weight matrices: [out, in] → [in, out]
                    if needs_transpose(name, shape) {
                        t.transpose()?
                    } else {
                        t
                    }
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
