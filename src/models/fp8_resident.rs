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

/// Pre-allocated BF16 weight buffer. Single GPU allocation reused for every
/// weight in every block across all denoise steps. Zero new GPU allocations
/// during forward pass.
pub struct PersistentBlockBuf {
    /// Single shared BF16 buffer on GPU, sized to the largest FP8 weight.
    pub buf_ptr: cudarc::driver::CudaSlice<u16>,
    pub capacity: usize, // max elements
    pub bufs: HashMap<String, Tensor>, // unused, kept for compatibility
}

impl PersistentBlockBuf {
    /// Create persistent buffer sized to hold ALL FP8 weights in one block
    /// simultaneously (they must all be alive during forward pass).
    pub fn new(block: &ResidentBlock, device: &Arc<CudaDevice>) -> Result<Self> {
        let mut total_numel = 0usize;
        for (_name, raw) in &block.weights {
            if let RawWeight::FP8 { shape, .. } = raw {
                let numel: usize = shape.iter().product();
                total_numel += numel;
            }
        }

        if total_numel == 0 { total_numel = 1; } // avoid zero alloc
        let buf_ptr: CudaSlice<u16> = unsafe { device.alloc(total_numel)? };
        log::info!("[PersistentBlockBuf] Allocated {:.1}MB shared buffer ({} elements, all FP8 weights)",
            total_numel as f64 * 2.0 / 1e6, total_numel);
        Ok(Self { buf_ptr, capacity: total_numel, bufs: HashMap::new() })
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
                    // Dequant to BF16 (allocates new tensor each time)
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

    /// Convert FP8 weights to BF16 using the persistent shared buffer.
    /// **True zero-alloc**: all FP8 dequant writes into the persistent buffer,
    /// then we create non-owning view tensors via `view_from_buffer`.
    ///
    /// Each FP8 weight gets a view into the shared buffer at sequential offsets.
    /// The buffer is large enough for ALL FP8 weights in one block simultaneously
    /// (they're consumed during forward, so we need them all alive at once).
    ///
    /// BF16 weights: Arc clone (zero copy).
    /// F32 tensors: Arc clone (zero copy).
    pub fn to_bf16_block_reuse(
        &self,
        persistent: &PersistentBlockBuf,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::with_capacity(self.weights.len() + self.f32_tensors.len());

        // Compute total FP8 elements needed to see if they fit in the shared buf
        let mut fp8_entries: Vec<(&String, &CudaSlice<u8>, &[usize], f32, bool)> = Vec::new();
        let mut total_needed = 0usize;
        for (name, raw) in &self.weights {
            if let RawWeight::FP8 { data, shape, scale, .. } = raw {
                let transpose = needs_transpose(name, shape);
                let numel: usize = shape.iter().product();
                fp8_entries.push((name, data, shape, *scale, transpose));
                total_needed += numel;
            }
        }

        let use_shared = total_needed <= persistent.capacity;
        if !use_shared {
            log::warn!(
                "[fp8_resident] FP8 total {} > buf capacity {}, falling back to per-weight alloc",
                total_needed, persistent.capacity
            );
        }

        let buf_base_ptr = *persistent.buf_ptr.device_ptr() as *mut u16;
        let mut offset = 0usize;

        // Process FP8 weights
        for (name, fp8_data, shape, scale, transpose) in &fp8_entries {
            let numel: usize = shape.iter().product();

            let tensor = if use_shared {
                // Create a view into the shared buffer at current offset
                let view_ptr = unsafe { buf_base_ptr.add(offset) };
                let (view_shape, out_shape_dims) = if *transpose {
                    (Shape::from_dims(&[shape[1], shape[0]]), vec![shape[1], shape[0]])
                } else {
                    (Shape::from_dims(shape), shape.to_vec())
                };
                let view = unsafe {
                    Tensor::view_from_buffer(view_ptr, view_shape, device.clone())
                };

                if *transpose {
                    flame_core::ops::fused_inference::dequant_fp8_transpose_into(
                        fp8_data, *scale, &view, shape[0], shape[1],
                    )?;
                } else {
                    flame_core::ops::fused_inference::dequant_fp8_to_bf16_into(
                        fp8_data, *scale, &view,
                    )?;
                }
                offset += numel;
                view
            } else {
                // Fallback: allocate per-weight
                if *transpose {
                    let out_shape = Shape::from_dims(&[shape[1], shape[0]]);
                    let output = Tensor::zeros_dtype(out_shape, DType::BF16, device.clone())?;
                    flame_core::ops::fused_inference::dequant_fp8_transpose_into(
                        fp8_data, *scale, &output, shape[0], shape[1],
                    )?;
                    output
                } else {
                    flame_core::ops::fused_inference::dequant_fp8_to_bf16(
                        fp8_data, *scale, Shape::from_dims(shape), device,
                    )?
                }
            };
            result.insert((*name).clone(), tensor);
        }

        // BF16 weights: zero-copy Arc clone
        for (name, raw) in &self.weights {
            if let RawWeight::BF16 { tensor } = raw {
                result.insert(name.clone(), tensor.clone());
            }
        }

        // F32 tensors: zero-copy Arc clone
        for (name, tensor) in &self.f32_tensors {
            result.insert(name.clone(), tensor.clone());
        }

        Ok(result)
    }
}
