//! LoRA weight loader and fusion for LTX-2.3 inference pipeline.
//!
//! Loads LoRA safetensors files (distilled, camera control, IC LoRA) and fuses
//! their deltas into model weights via `weight_new = weight_orig + strength * (B @ A)`.
//!
//! LoRA key format:
//!   `diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight`  [rank, in]
//!   `diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight`  [out, rank]
//!
//! Base key extraction: strip `diffusion_model.` prefix and `.lora_{A,B}.weight` suffix.

use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// A loaded LoRA with paired A/B matrices ready for fusion.
pub struct LoraWeights {
    /// Map: base_key -> (lora_A [rank, in_features], lora_B [out_features, rank])
    pub deltas: HashMap<String, (Tensor, Tensor)>,
    /// Fusion strength multiplier
    pub strength: f32,
    /// Safetensors __metadata__ (e.g. "reference_downscale_factor" for IC LoRAs)
    pub metadata: HashMap<String, String>,
}

impl LoraWeights {
    /// Load a LoRA from a safetensors file.
    ///
    /// Parses the header to extract `__metadata__`, then loads all `lora_A`/`lora_B`
    /// pairs as BF16 tensors on the given CUDA device. Keys are remapped to base
    /// model keys by stripping `diffusion_model.` prefix and `.lora_{A,B}.weight` suffix.
    pub fn load(path: &str, strength: f32, device: &Arc<CudaDevice>) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| flame_core::Error::Io(format!("open {path}: {e}")))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| flame_core::Error::Io(format!("mmap {path}: {e}")))?;

        if mmap.len() < 8 {
            return Err(flame_core::Error::Io("File too small for safetensors".into()));
        }

        // Parse safetensors header
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        let data_start = header_end;

        let header: serde_json::Value = serde_json::from_slice(&mmap[8..header_end])
            .map_err(|e| flame_core::Error::Io(format!("header parse: {e}")))?;
        let meta = header.as_object().ok_or_else(||
            flame_core::Error::InvalidInput("Invalid safetensors header".into()))?;

        // Extract __metadata__
        let metadata = extract_metadata(meta);

        // First pass: collect all lora_A and lora_B entries by base key
        let mut a_entries: HashMap<String, (&serde_json::Value, String)> = HashMap::new();
        let mut b_entries: HashMap<String, (&serde_json::Value, String)> = HashMap::new();

        for (key, info) in meta {
            if key == "__metadata__" {
                continue;
            }

            if let Some(base_key) = extract_base_key(key) {
                if key.contains(".lora_A.") || key.contains(".lora_a.") {
                    a_entries.insert(base_key, (info, key.clone()));
                } else if key.contains(".lora_B.") || key.contains(".lora_b.") {
                    b_entries.insert(base_key, (info, key.clone()));
                }
            }
        }

        // Second pass: load paired A/B tensors
        let mut deltas = HashMap::with_capacity(a_entries.len());
        let mut loaded = 0usize;

        for (base_key, (a_info, _a_key)) in &a_entries {
            let (b_info, _b_key) = match b_entries.get(base_key) {
                Some(entry) => entry,
                None => continue, // orphaned lora_A without lora_B — skip
            };

            let tensor_a = load_bf16_tensor(a_info, &mmap, data_start, device)?;
            let tensor_b = load_bf16_tensor(b_info, &mmap, data_start, device)?;

            deltas.insert(base_key.clone(), (tensor_a, tensor_b));
            loaded += 1;
        }

        log::info!(
            "[LoRA] Loaded {} paired weights from {}, strength={:.2}",
            loaded, path, strength,
        );

        Ok(Self { deltas, strength, metadata })
    }

    /// Compute the fused delta for a single weight key: `strength * (B @ A)`.
    ///
    /// Returns `None` if this LoRA doesn't have weights for the given key.
    /// The result shape is [out_features, in_features], matching the original weight.
    pub fn compute_delta(&self, base_key: &str) -> Result<Option<Tensor>> {
        let (lora_a, lora_b) = match self.deltas.get(base_key) {
            Some(pair) => pair,
            None => return Ok(None),
        };

        // B [out, rank] @ A [rank, in] -> [out, in]
        let product = lora_b.matmul(lora_a)?;

        if (self.strength - 1.0).abs() < 1e-6 {
            Ok(Some(product))
        } else {
            Ok(Some(product.mul_scalar(self.strength)?))
        }
    }

    /// Get a metadata value (e.g. "reference_downscale_factor" for IC LoRAs).
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Number of paired A/B weight entries.
    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    /// Whether this LoRA has no weights.
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Infer LoRA rank from the first weight pair.
    /// Returns None if there are no weights.
    pub fn rank(&self) -> Option<usize> {
        self.deltas.values().next().map(|(a, _b)| {
            // lora_A shape is [rank, in_features]
            a.shape().dims()[0]
        })
    }
}

/// Fuse one or more LoRAs into a weight tensor.
///
/// Computes: `weight + sum(strength_i * (B_i @ A_i))` for each LoRA that
/// affects the given key. Returns the original weight unchanged if no LoRA
/// has a delta for this key.
pub fn fuse_loras(weight: &Tensor, key: &str, loras: &[&LoraWeights]) -> Result<Tensor> {
    let mut result = weight.clone();
    let mut fused_any = false;

    for lora in loras {
        if let Some(delta) = lora.compute_delta(key)? {
            result = result.add(&delta)?;
            fused_any = true;
        }
    }

    if fused_any {
        log::debug!("[LoRA] Fused {} LoRA(s) into {}", loras.len(), key);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract the base model key from a LoRA key.
///
/// `diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight`
/// -> `transformer_blocks.0.attn1.to_q.weight`
///
/// Steps:
/// 1. Strip `diffusion_model.` prefix (if present)
/// 2. Strip `.lora_A.weight` or `.lora_B.weight` suffix -> get the param path
/// 3. Append `.weight` to get the base weight key
fn extract_base_key(lora_key: &str) -> Option<String> {
    // Must contain .lora_A. or .lora_B. (case-insensitive check)
    let is_a = lora_key.contains(".lora_A.") || lora_key.contains(".lora_a.");
    let is_b = lora_key.contains(".lora_B.") || lora_key.contains(".lora_b.");
    if !is_a && !is_b {
        return None;
    }

    let mut key = lora_key.to_string();

    // Strip prefix
    if let Some(rest) = key.strip_prefix("diffusion_model.") {
        key = rest.to_string();
    }

    // Strip .lora_{A,B}.weight suffix to get the param path
    // e.g. "transformer_blocks.0.attn1.to_q.lora_A.weight" -> "transformer_blocks.0.attn1.to_q"
    for suffix in &[".lora_A.weight", ".lora_B.weight", ".lora_a.weight", ".lora_b.weight"] {
        if let Some(prefix) = key.strip_suffix(suffix) {
            return Some(format!("{prefix}.weight"));
        }
    }

    None
}

/// Extract __metadata__ from safetensors header as a String->String map.
fn extract_metadata(header: &serde_json::Map<String, serde_json::Value>) -> HashMap<String, String> {
    let mut result = HashMap::new();
    if let Some(serde_json::Value::Object(meta)) = header.get("__metadata__") {
        for (k, v) in meta {
            if let serde_json::Value::String(s) = v {
                result.insert(k.clone(), s.clone());
            }
        }
    }
    result
}

/// Load a single tensor from mmap'd safetensors data as BF16 on GPU.
fn load_bf16_tensor(
    info: &serde_json::Value,
    mmap: &memmap2::Mmap,
    data_start: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let shape: Vec<usize> = info["shape"]
        .as_array()
        .ok_or_else(|| flame_core::Error::InvalidInput("Missing shape".into()))?
        .iter()
        .filter_map(|v| v.as_u64().map(|u| u as usize))
        .collect();

    let offsets = info["data_offsets"]
        .as_array()
        .ok_or_else(|| flame_core::Error::InvalidInput("Missing data_offsets".into()))?;
    let start = data_start
        + offsets[0]
            .as_u64()
            .ok_or_else(|| flame_core::Error::InvalidInput("invalid start offset".into()))?
            as usize;
    let end = data_start
        + offsets[1]
            .as_u64()
            .ok_or_else(|| flame_core::Error::InvalidInput("invalid end offset".into()))?
            as usize;

    let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
    let data = &mmap[start..end];

    match dtype_str {
        "BF16" => {
            let bf16_u16: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let mut tensor =
                Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
            tensor.copy_from_bf16_slice(&bf16_u16)?;
            Ok(tensor)
        }
        "F32" => {
            // Upconvert: load F32, store as BF16 for consistent matmul dtype
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let bf16_u16: Vec<u16> = f32_data
                .iter()
                .map(|&f| half::bf16::from_f32(f).to_bits())
                .collect();
            let mut tensor =
                Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
            tensor.copy_from_bf16_slice(&bf16_u16)?;
            Ok(tensor)
        }
        "F16" => {
            // Convert F16 -> BF16
            let bf16_u16: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let f = half::f16::from_bits(bits).to_f32();
                    half::bf16::from_f32(f).to_bits()
                })
                .collect();
            let mut tensor =
                Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
            tensor.copy_from_bf16_slice(&bf16_u16)?;
            Ok(tensor)
        }
        other => Err(flame_core::Error::InvalidInput(format!(
            "Unsupported LoRA dtype: {other} (expected BF16, F32, or F16)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_base_key() {
        // Standard LTX-2 LoRA key
        assert_eq!(
            extract_base_key(
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"
            ),
            Some("transformer_blocks.0.attn1.to_q.weight".to_string())
        );
        assert_eq!(
            extract_base_key(
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight"
            ),
            Some("transformer_blocks.0.attn1.to_q.weight".to_string())
        );

        // Without diffusion_model prefix
        assert_eq!(
            extract_base_key("transformer_blocks.5.ff.net.0.proj.lora_A.weight"),
            Some("transformer_blocks.5.ff.net.0.proj.weight".to_string())
        );

        // Not a LoRA key
        assert_eq!(
            extract_base_key("transformer_blocks.0.attn1.to_q.weight"),
            None
        );
    }

    #[test]
    fn test_extract_base_key_pairs_match() {
        // A and B keys for the same weight must produce the same base key
        let key_a = "diffusion_model.transformer_blocks.12.attn2.to_v.lora_A.weight";
        let key_b = "diffusion_model.transformer_blocks.12.attn2.to_v.lora_B.weight";
        let base_a = extract_base_key(key_a).unwrap();
        let base_b = extract_base_key(key_b).unwrap();
        assert_eq!(base_a, base_b);
        assert_eq!(base_a, "transformer_blocks.12.attn2.to_v.weight");
    }
}
