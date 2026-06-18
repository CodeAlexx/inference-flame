//! Ideogram 4 — transformer weight loader (FP8-resident, sharded safetensors).
//!
//! Mirrors `pipeline_ideogram4.py` (`_load_sharded_state_dict`,
//! `_load_indexed_or_single_state_dict`, `_build_transformer`) +
//! `quantized_loading.py` (`is_fp8_state_dict`, the weight-only FP8 path) for a
//! LOCAL filesystem checkpoint (the runtime is pure Rust — no hf_hub download).
//!
//! ## What it produces
//!
//! `load_transformer(repo_path, subfolder, device, config)` →
//! `HashMap<String, Ideogram4RawWeight>` keyed by PyTorch module path
//! (`layers.{i}.attention.qkv.weight`, `input_proj.weight`, …). FP8 entries
//! keep raw e4m3 bytes on GPU + the per-output-row `[out]` f32 scale (uploaded
//! BF16 inside [`Ideogram4RawWeight::fp8_from_parts`]); everything else (norms,
//! biases, embeds, non-quantized linears) is loaded BF16.
//!
//! Two callers, same `config`, distinct weights:
//!   - `subfolder = "transformer"` (conditional branch)
//!   - `subfolder = "unconditional_transformer"` (unconditional branch)
//!
//! ## FP8 detection (`quantized_loading.py:157-161`)
//!
//! `is_fp8_state_dict` = any key ends with `.weight_scale` OR any tensor dtype
//! is float8. Here we detect per-tensor: an `F8_E4M3` dtype entry is FP8, and
//! its `<name>.weight_scale` sibling (f32 `[out]`) carries the per-row scale.
//! `.weight_scale` keys are CONSUMED (folded into the FP8 entry), never emitted
//! as standalone weights.
//!
//! ## Sharded index merge (`_load_sharded_state_dict`)
//!
//! `{subfolder}/diffusion_pytorch_model.safetensors.index.json` → `weight_map`
//! (key → shard filename). Shards resolve relative to the index's directory.
//! If the index is absent, fall back to the single
//! `{subfolder}/diffusion_pytorch_model.safetensors`
//! (`_load_indexed_or_single_state_dict`). The pure shard-list resolution lives
//! in [`resolve_shard_paths`] (host-testable).
//!
//! ## Reader
//!
//! Uses `eri_safetensors::MmapFile` (the SAME mmap reader
//! `flame_core::serialization::load_file_filtered` builds on) to get raw bytes
//! + dtype + shape per tensor. We do NOT use `serialization::load_file`: that
//! path dequantizes FP8 with a per-TENSOR scalar scale, which is wrong for
//! Ideogram-4's per-ROW scale — so we read raw bytes and build the per-row
//! container ourselves. No new Cargo dep (`eri-safetensors` is already a direct
//! dependency of `inference-flame`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use flame_core::{DType, Error, Result, Shape, Tensor};

use crate::models::ideogram4::weights::Ideogram4RawWeight;
use crate::models::ideogram4::Ideogram4Config;

/// safetensors dtype string for weight-only e4m3 FP8 (matches
/// `flame_core::serialization` line 529 + `eri_safetensors` dtype passthrough).
const FP8_DTYPE: &str = "F8_E4M3";
/// Per-output-row scale key suffix. The scale for FP8 weight `<name>.weight` is
/// `<name>.weight_scale` — i.e. `_scale` APPENDED to the full weight key (the
/// checkpoint keys are e.g. `layers.0.attention.o.weight` +
/// `layers.0.attention.o.weight_scale`), NOT `<name>.weight.weight_scale`.
/// (`quantized_loading.py:22`; verified against the shipped ideogram-4-fp8
/// transformer index.)
const FP8_SCALE_SUFFIX: &str = "_scale";

/// Resolve the shard file paths for a component, mirroring
/// `_load_indexed_or_single_state_dict`.
///
/// `subfolder_dir` is the absolute path to e.g. `<repo>/transformer`. Returns
/// the list of safetensors files to read (the merged-shard set, or the single
/// file fallback) in deterministic sorted order.
///
/// Pure host logic (file existence + JSON parse): unit-testable without GPU.
pub fn resolve_shard_paths(subfolder_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = subfolder_dir.join("diffusion_pytorch_model.safetensors.index.json");
    let single_path = subfolder_dir.join("diffusion_pytorch_model.safetensors");

    if index_path.is_file() {
        let bytes = std::fs::read(&index_path).map_err(|e| {
            Error::Io(format!("read index '{}': {e}", index_path.display()))
        })?;
        let index: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            Error::Io(format!("parse index '{}': {e}", index_path.display()))
        })?;
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
        // Shards resolve relative to the index's directory (= subfolder_dir).
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

/// Load one Ideogram-4 transformer's weights into a raw-weight map.
///
/// `repo_path` is the local checkpoint root; `subfolder` is `"transformer"`
/// (conditional) or `"unconditional_transformer"` (unconditional). `config` is
/// the shared `Ideogram4Config` (used only for sanity — the shapes come from
/// the file).
///
/// FP8 weights → `Ideogram4RawWeight::Fp8` (raw bytes + per-row scale).
/// Everything else floating → `Ideogram4RawWeight::Bf16`. `.weight_scale`
/// entries are consumed into their FP8 sibling and not emitted.
///
/// **Requires a CUDA device** (uploads bytes via `htod_copy`). GPU-gated.
pub fn load_transformer(
    repo_path: &Path,
    subfolder: &str,
    device: &Arc<CudaDevice>,
    _config: &Ideogram4Config,
) -> Result<HashMap<String, Ideogram4RawWeight>> {
    let subfolder_dir = repo_path.join(subfolder);
    let shard_paths = resolve_shard_paths(&subfolder_dir)?;

    let mut weights: HashMap<String, Ideogram4RawWeight> = HashMap::new();

    for shard_path in &shard_paths {
        let mmap = eri_safetensors::MmapFile::open_path(shard_path).map_err(|e| {
            Error::Io(format!(
                "eri-safetensors mmap open '{}': {e}",
                shard_path.display()
            ))
        })?;

        // First pass: index the per-row scale tensors in this shard so an FP8
        // weight can resolve its `<name>.weight_scale` sibling regardless of
        // map iteration order.
        let scale_keys: Vec<String> = mmap
            .tensors
            .keys()
            .filter(|k| k.ends_with(FP8_SCALE_SUFFIX))
            .cloned()
            .collect();

        for (name, tref) in &mmap.tensors {
            // Skip scale companions — they are folded into their FP8 weight.
            if name.ends_with(FP8_SCALE_SUFFIX) {
                continue;
            }
            let bytes = mmap.tensor_bytes(name).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "mmap missing tensor bytes for '{name}' in {}",
                    shard_path.display()
                ))
            })?;
            let shape = tref.shape.clone();

            match tref.dtype.as_str() {
                FP8_DTYPE => {
                    // FP8 weight: raw e4m3 bytes + per-row scale sibling.
                    let scale_key = format!("{name}{FP8_SCALE_SUFFIX}");
                    if !scale_keys.iter().any(|k| k == &scale_key) {
                        return Err(Error::InvalidInput(format!(
                            "FP8 weight '{name}' has no '{scale_key}' per-row scale in {}",
                            shard_path.display()
                        )));
                    }
                    let scale_bytes = mmap.tensor_bytes(&scale_key).ok_or_else(|| {
                        Error::InvalidInput(format!("mmap missing scale bytes '{scale_key}'"))
                    })?;
                    // Scale is f32 [out]. Decode little-endian.
                    let scale_f32: Vec<f32> = scale_bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();

                    let gpu: CudaSlice<u8> = device
                        .htod_copy(bytes.to_vec())
                        .map_err(|e| Error::Cuda(format!("htod fp8 '{name}': {e:?}")))?;

                    let w = Ideogram4RawWeight::fp8_from_parts(gpu, shape, scale_f32, device)?;
                    weights.insert(name.clone(), w);
                }
                "BF16" => {
                    let bf16_u16: Vec<u16> = bytes
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let mut tensor =
                        Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
                    tensor.copy_from_bf16_slice(&bf16_u16)?;
                    weights.insert(name.clone(), Ideogram4RawWeight::Bf16 { tensor });
                }
                "F16" => {
                    // Convert F16 → BF16 (model compute dtype is BF16).
                    let bf16_u16: Vec<u16> = bytes
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            half::bf16::from_f32(half::f16::from_bits(bits).to_f32()).to_bits()
                        })
                        .collect();
                    let mut tensor =
                        Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
                    tensor.copy_from_bf16_slice(&bf16_u16)?;
                    weights.insert(name.clone(), Ideogram4RawWeight::Bf16 { tensor });
                }
                "F32" => {
                    // Non-quantized f32 params (rare here) → cast to BF16 to
                    // match the reference's `.to(compute_dtype=bf16)` for every
                    // non-FP8 floating tensor (load_fp8_state_dict line 265-266).
                    let bf16_u16: Vec<u16> = bytes
                        .chunks_exact(4)
                        .map(|c| {
                            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            half::bf16::from_f32(f).to_bits()
                        })
                        .collect();
                    let mut tensor =
                        Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
                    tensor.copy_from_bf16_slice(&bf16_u16)?;
                    weights.insert(name.clone(), Ideogram4RawWeight::Bf16 { tensor });
                }
                // Integer / bool / unsupported dtypes (e.g. computed rotary
                // caches the reference recomputes) are skipped — the DiT builds
                // its own MRoPE table.
                _ => {}
            }
        }
    }

    if weights.is_empty() {
        return Err(Error::InvalidInput(format!(
            "loaded zero weights from '{}/{subfolder}'",
            repo_path.display()
        )));
    }

    Ok(weights)
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
    fn resolve_sharded_index() {
        let tmp = std::env::temp_dir().join(format!("idg4_loader_sharded_{}", std::process::id()));
        let sub = tmp.join("transformer");
        std::fs::create_dir_all(&sub).unwrap();
        write_json(
            &sub,
            "diffusion_pytorch_model.safetensors.index.json",
            r#"{"weight_map":{"a.weight":"model-00002-of-00002.safetensors",
                "b.weight":"model-00001-of-00002.safetensors",
                "c.weight":"model-00001-of-00002.safetensors"}}"#,
        );
        let paths = resolve_shard_paths(&sub).unwrap();
        // dedup + sorted → 2 shards, shard 1 before shard 2.
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("model-00001-of-00002.safetensors"));
        assert!(paths[1].ends_with("model-00002-of-00002.safetensors"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_single_file_fallback() {
        let tmp = std::env::temp_dir().join(format!("idg4_loader_single_{}", std::process::id()));
        let sub = tmp.join("unconditional_transformer");
        std::fs::create_dir_all(&sub).unwrap();
        // No index, just the single file.
        touch(&sub, "diffusion_pytorch_model.safetensors");
        let paths = resolve_shard_paths(&sub).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("diffusion_pytorch_model.safetensors"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_missing_errors() {
        let tmp = std::env::temp_dir().join(format!("idg4_loader_missing_{}", std::process::id()));
        let sub = tmp.join("transformer");
        std::fs::create_dir_all(&sub).unwrap();
        // Neither index nor single file.
        assert!(resolve_shard_paths(&sub).is_err());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_index_takes_priority_over_single() {
        // When both exist, the index path is used (matches the reference: try
        // index first, fall back only on EntryNotFound).
        let tmp = std::env::temp_dir().join(format!("idg4_loader_both_{}", std::process::id()));
        let sub = tmp.join("transformer");
        std::fs::create_dir_all(&sub).unwrap();
        touch(&sub, "diffusion_pytorch_model.safetensors");
        write_json(
            &sub,
            "diffusion_pytorch_model.safetensors.index.json",
            r#"{"weight_map":{"x.weight":"shard-A.safetensors"}}"#,
        );
        let paths = resolve_shard_paths(&sub).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("shard-A.safetensors"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    // GPU-dependent: load_transformer uploads bytes to device.
    #[test]
    #[ignore = "requires CUDA device + a real checkpoint (GPU busy); compile-only this chunk"]
    fn load_transformer_smoke() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let cfg = Ideogram4Config::default();
        let repo = Path::new("/home/alex/.serenity/models/ideogram-4-fp8");
        let w = load_transformer(repo, "transformer", &device, &cfg).unwrap();
        assert!(!w.is_empty());
    }
}
