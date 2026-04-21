//! Tier A GGUF weight loader.
//!
//! Loads any stable-diffusion.cpp / diffusers GGUF file, dequantizes every
//! tensor on CPU to f32, converts to BF16, and uploads to a CUDA device.
//! Returns a `HashMap<String, Tensor>` drop-in compatible with the
//! existing safetensors loader in `flame_core::serialization::load_file`.
//!
//! # Design
//!
//! - **Streaming per-tensor**: dequant→upload→free one tensor at a time.
//!   A FLUX Q4_K_M file is ~6 GB on disk; an f32 intermediate of the full
//!   expanded tensor graph would peak at ~24 GB host RAM. Per-tensor
//!   streaming caps peak host RAM at ~max-single-tensor size.
//! - **CPU dequant**: one-shot cost at load time. GPU dequant (keeping
//!   tensors quantized in VRAM) is Tier B / Phase 2 and lives elsewhere.
//! - **BF16 output**: matches the rest of the inference pipeline's
//!   assumed weight dtype (see `feature = "bf16_u16"` wiring).
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//! use flame_core::CudaDevice;
//! use inference_flame::gguf::load_file_gguf;
//!
//! let device: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
//! let weights = load_file_gguf(Path::new("flux1-dev-Q4_K_M.gguf"), device).unwrap();
//! // `weights` is a HashMap<String, Tensor> with BF16 tensors and
//! // the `model.diffusion_model.` / `transformer.` prefix stripped.
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};

pub mod dequant;
pub mod reader;
pub mod remap;

pub use reader::{GgufHeader, GgufQuantType, GgufTensorInfo};
pub use remap::{default_rename, rename_keys};

/// Read just the header + tensor table, no tensor data.
///
/// Useful for introspection and tests without the cost of dequantizing
/// every weight in a multi-GB file.
pub fn read_header(path: &Path) -> Result<GgufHeader> {
    let (header, _mmap) = reader::parse_header_from_file(path)?;
    Ok(header)
}

/// Load a GGUF file into a `HashMap<String, Tensor>` of BF16 tensors.
///
/// Keys are passed through [`default_rename`] which strips the
/// `model.diffusion_model.` / `transformer.` / `first_stage_model.`
/// top-level prefix. Callers that need the raw GGUF key names should use
/// [`load_file_gguf_raw`].
pub fn load_file_gguf(
    path: &Path,
    device: Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let raw = load_file_gguf_raw(path, device)?;
    Ok(rename_keys(raw, default_rename))
}

/// Load a GGUF file without any key renaming.
///
/// Tensors are still dequantized to BF16 and uploaded to `device`. Only
/// the final remap step is skipped.
pub fn load_file_gguf_raw(
    path: &Path,
    device: Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let (header, mmap) = reader::parse_header_from_file(path)
        .with_context(|| format!("parsing GGUF header {}", path.display()))?;

    let mut out = HashMap::with_capacity(header.tensors.len());

    for info in &header.tensors {
        let n_elems = info.n_elements() as usize;

        // Empty-tensor guard: a tensor with any zero-sized dim (n_elems = 0)
        // cannot be dequantized by the block-chunked kernels below — they
        // assume at least one block. Real diffusion GGUFs don't ship empty
        // tensors, but a malformed file shouldn't crash the whole loader.
        // We bail here rather than returning a 0-size Tensor because the
        // downstream `Tensor::from_f32_to_bf16` semantics on a 0-element
        // Vec are not specified in flame-core.
        if n_elems == 0 {
            return Err(anyhow!(
                "GGUF: tensor {} has zero elements (dims {:?}); refusing to load",
                info.name,
                info.dims
            ));
        }

        let byte_size = info.byte_size.ok_or_else(|| {
            anyhow!(
                "GGUF: tensor {} uses unsupported ggml_type {:?} \
                 (or n_elements {} not divisible by the type's block size)",
                info.name,
                info.quant,
                n_elems
            )
        })?;

        let start = info.data_offset as usize;
        let end = start + byte_size as usize;
        if end > mmap.len() {
            return Err(anyhow!(
                "GGUF: tensor {} data range [{},{}) exceeds file size {}",
                info.name,
                start,
                end,
                mmap.len()
            ));
        }

        let bytes = &mmap[start..end];

        // Dequant → f32 on CPU.
        let data_f32 = dequant::dequantize_to_f32(info.quant, bytes, n_elems)
            .with_context(|| {
                format!(
                    "dequantizing tensor {} ({} elems, {} bytes, type {:?})",
                    info.name, n_elems, byte_size, info.quant
                )
            })?;

        // Upload as BF16. This mirrors the F16/F32 paths in
        // flame_core::serialization and keeps peak host RAM bounded:
        // `data_f32` is moved into the upload and dropped at function
        // boundary before the next tensor is dequantized.
        let tensor = Tensor::from_f32_to_bf16(
            data_f32,
            Shape::from_dims(&info.dims),
            device.clone(),
        )
        .map_err(|e| {
            anyhow!(
                "GGUF: uploading tensor {} to device: {:?}",
                info.name,
                e
            )
        })?;

        out.insert(info.name.clone(), tensor);
    }

    Ok(out)
}
