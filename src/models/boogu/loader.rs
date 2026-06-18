//! Boogu-Image — sharded-safetensors → BF16 weight map (minimal but real).
//!
//! Mirrors the sibling `ideogram4/loader.rs` idiom (sharded-index merge +
//! `eri_safetensors::MmapFile` raw-byte read) but WITHOUT the FP8 machinery:
//! Boogu ships **bf16 throughout, no quantization** (PORT_SPEC §Dependencies,
//! handoff "no quant"). So every tensor lands as a plain BF16 [`Tensor`] keyed
//! by its PyTorch module path (`time_caption_embed.timestep_embedder.linear_1.weight`,
//! `x_embedder.weight`, …).
//!
//! ## Scope (this chunk)
//!
//! Enough to load the embedder + a transformer shard for C1/C2 probes. The
//! sharded-index resolution + mmap read covers the FULL transformer (942 bf16
//! tensors) so later chunks reuse this loader unchanged — the only "minimal"
//! part is that no mllm/VAE remapping lives here yet (those are C7/C8, per
//! BUILD_PLAN).
//!
//! ## Reader
//!
//! Uses `eri_safetensors::MmapFile` — the SAME mmap reader
//! `flame_core::serialization::load_file_filtered` and `ideogram4/loader.rs`
//! build on. No new Cargo dep. We read raw bytes ourselves (rather than
//! `serialization::load_file`) only to keep dtype handling explicit + uniform
//! with the ideogram4 sibling; Boogu has no FP8 so a future switch to
//! `serialization::load_file_filtered` would also work.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{DType, Error, Result, Shape, Tensor};

/// Resolve the shard file paths for a component, mirroring diffusers'
/// `_load_indexed_or_single_state_dict` (and `ideogram4/loader.rs`).
///
/// `subfolder_dir` is the absolute path to e.g. `<repo>/transformer`. Returns
/// the safetensors files to read (the merged-shard set in deterministic sorted
/// order, or the single-file fallback).
///
/// Pure host logic (file existence + JSON parse): unit-testable without a GPU.
pub fn resolve_shard_paths(subfolder_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = subfolder_dir.join("diffusion_pytorch_model.safetensors.index.json");
    let single_path = subfolder_dir.join("diffusion_pytorch_model.safetensors");

    if index_path.is_file() {
        let bytes = std::fs::read(&index_path)
            .map_err(|e| Error::Io(format!("read index '{}': {e}", index_path.display())))?;
        let index: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Io(format!("parse index '{}': {e}", index_path.display())))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "index '{}' missing object field 'weight_map'",
                    index_path.display()
                ))
            })?;
        // sorted(set(weight_map.values())) — deterministic shard order.
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shard_names.sort();
        shard_names.dedup();
        if shard_names.is_empty() {
            return Err(Error::InvalidInput(format!(
                "index '{}' has empty weight_map",
                index_path.display()
            )));
        }
        Ok(shard_names
            .into_iter()
            .map(|s| subfolder_dir.join(s))
            .collect())
    } else if single_path.is_file() {
        Ok(vec![single_path])
    } else {
        Err(Error::InvalidInput(format!(
            "no checkpoint in '{}' (looked for {} and {})",
            subfolder_dir.display(),
            index_path.display(),
            single_path.display()
        )))
    }
}

/// Decode raw safetensors bytes to a BF16 [`Tensor`].
///
/// Boogu transformer tensors are all BF16; F16/F32 paths are kept for
/// robustness (cast to BF16, the model compute dtype), matching the
/// ideogram4 sibling's non-FP8 handling. Integer/bool/unknown dtypes are
/// rejected (the caller's `_` arm skips them by name, but a raw call should
/// fail loud rather than silently produce garbage).
fn decode_to_bf16(
    dtype_str: &str,
    bytes: &[u8],
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let bf16_u16: Vec<u16> = match dtype_str {
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::bf16::from_f32(half::f16::from_bits(bits).to_f32()).to_bits()
            })
            .collect(),
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| {
                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                half::bf16::from_f32(f).to_bits()
            })
            .collect(),
        other => {
            return Err(Error::InvalidInput(format!(
                "boogu loader: unsupported tensor dtype '{other}' (expected BF16/F16/F32)"
            )));
        }
    };
    let mut tensor = Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, device.clone())?;
    tensor.copy_from_bf16_slice(&bf16_u16)?;
    Ok(tensor)
}

/// Load one component's sharded safetensors into a BF16 weight map.
///
/// `repo_path` is the local checkpoint root
/// (`/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base`); `subfolder` is e.g.
/// `"transformer"`. Keys are the verbatim PyTorch module paths from the
/// safetensors index (`x_embedder.weight`,
/// `time_caption_embed.caption_embedder.1.bias`, …) — **no renames**.
///
/// **Requires a CUDA device** (uploads bytes via `copy_from_bf16_slice`).
pub fn load_component(
    repo_path: &Path,
    subfolder: &str,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let subfolder_dir = repo_path.join(subfolder);
    let shard_paths = resolve_shard_paths(&subfolder_dir)?;

    let mut weights: HashMap<String, Tensor> = HashMap::new();

    for shard_path in &shard_paths {
        let mmap = eri_safetensors::MmapFile::open_path(shard_path).map_err(|e| {
            Error::Io(format!(
                "eri-safetensors mmap open '{}': {e}",
                shard_path.display()
            ))
        })?;

        for (name, tref) in &mmap.tensors {
            let bytes = mmap.tensor_bytes(name).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "mmap missing tensor bytes for '{name}' in {}",
                    shard_path.display()
                ))
            })?;
            let shape = tref.shape.clone();
            match tref.dtype.as_str() {
                "BF16" | "F16" | "F32" => {
                    let tensor = decode_to_bf16(tref.dtype.as_str(), bytes, &shape, device)?;
                    weights.insert(name.clone(), tensor);
                }
                // Integer / bool tensors (none expected in the Boogu transformer
                // — RoPE tables are recomputed, not stored) are skipped.
                _ => {}
            }
        }
    }

    if weights.is_empty() {
        return Err(Error::InvalidInput(format!(
            "boogu loader: loaded zero weights from '{}/{subfolder}'",
            repo_path.display()
        )));
    }

    Ok(weights)
}

/// Convenience: fetch a weight by key, erroring (not panicking) on a miss.
///
/// Used by the embedder/block forwards so a missing/renamed key fails loud with
/// the exact key name rather than a silent zero.
pub fn get<'a>(weights: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidOperation(format!("boogu: missing weight `{key}`")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_json(dir: &Path, name: &str, body: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    fn touch(dir: &Path, name: &str) {
        std::fs::File::create(dir.join(name)).unwrap();
    }

    #[test]
    fn resolve_sharded_index_dedups_and_sorts() {
        let tmp = std::env::temp_dir().join(format!("boogu_loader_sharded_{}", std::process::id()));
        let sub = tmp.join("transformer");
        std::fs::create_dir_all(&sub).unwrap();
        write_json(
            &sub,
            "diffusion_pytorch_model.safetensors.index.json",
            r#"{"weight_map":{"a.weight":"model-00003-of-00003.safetensors",
                "b.weight":"model-00001-of-00003.safetensors",
                "c.weight":"model-00002-of-00003.safetensors",
                "d.weight":"model-00001-of-00003.safetensors"}}"#,
        );
        let paths = resolve_shard_paths(&sub).unwrap();
        assert_eq!(paths.len(), 3); // dedup'd from 4 entries
        assert!(paths[0].ends_with("model-00001-of-00003.safetensors"));
        assert!(paths[1].ends_with("model-00002-of-00003.safetensors"));
        assert!(paths[2].ends_with("model-00003-of-00003.safetensors"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_single_file_fallback() {
        let tmp = std::env::temp_dir().join(format!("boogu_loader_single_{}", std::process::id()));
        let sub = tmp.join("transformer");
        std::fs::create_dir_all(&sub).unwrap();
        touch(&sub, "diffusion_pytorch_model.safetensors");
        let paths = resolve_shard_paths(&sub).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("diffusion_pytorch_model.safetensors"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_missing_errors() {
        let tmp = std::env::temp_dir().join(format!("boogu_loader_missing_{}", std::process::id()));
        let sub = tmp.join("transformer");
        std::fs::create_dir_all(&sub).unwrap();
        assert!(resolve_shard_paths(&sub).is_err());
        std::fs::remove_dir_all(&tmp).ok();
    }

    // GPU + real-checkpoint dependent: load_component uploads to device.
    #[test]
    #[ignore = "requires CUDA device + checkpoint (GPU busy); compile-only this chunk"]
    fn load_transformer_smoke() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let repo = Path::new("/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base");
        let w = load_component(repo, "transformer", &device).unwrap();
        // 942 transformer tensors per the safetensors index (handoff).
        assert!(!w.is_empty());
        assert!(get(&w, "x_embedder.weight").is_ok());
        assert!(get(&w, "time_caption_embed.timestep_embedder.linear_1.weight").is_ok());
    }
}
