//! Block-level weight offloading — load/unload transformer blocks from disk via mmap.
//!
//! Provides a generic `BlockLoader` that uses `load_file_filtered` to load only
//! the weights for a single block prefix at a time, freeing VRAM between blocks.

use flame_core::serialization::load_file_filtered;
use flame_core::{Error, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Generic block loader that streams transformer block weights from a safetensors
/// file on disk via mmap. Only one block's weights are kept on GPU at a time.
pub struct BlockLoader {
    /// Path to the safetensors model file.
    model_path: String,
    /// CUDA device to load tensors onto.
    device: Arc<cudarc::driver::CudaDevice>,
    /// Currently loaded block weights (keyed by stripped weight name).
    cache: HashMap<String, Tensor>,
    /// Optional key prefix in the safetensors file (e.g. "model.diffusion_model.").
    /// Stripped when loading so internal lookups use unprefixed keys.
    key_prefix: String,
}

impl BlockLoader {
    /// Create a new block loader for the given safetensors file.
    pub fn new(model_path: String, device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self {
            model_path,
            device,
            cache: HashMap::new(),
            key_prefix: String::new(),
        }
    }

    /// Create a block loader that strips a key prefix from loaded weights.
    /// E.g. `key_prefix = "model.diffusion_model."` strips that from all keys.
    pub fn new_with_prefix(
        model_path: String,
        device: Arc<cudarc::driver::CudaDevice>,
        key_prefix: &str,
    ) -> Self {
        Self {
            model_path,
            device,
            cache: HashMap::new(),
            key_prefix: key_prefix.to_string(),
        }
    }

    /// Load all weights whose key starts with `prefix.` into GPU memory.
    /// Any previously cached block is dropped first to free VRAM.
    /// If a key_prefix is set, searches for `{key_prefix}{prefix}.*` in the file
    /// and strips the key_prefix from cached keys.
    pub fn load_block(&mut self, prefix: &str) -> Result<()> {
        self.cache.clear();

        let file_prefix = format!("{}{prefix}.", self.key_prefix);
        let block_weights = load_file_filtered(&self.model_path, &self.device, |key| {
            key.starts_with(&file_prefix)
        })?;

        // Strip the key_prefix so internal lookups use unprefixed keys
        let kp = &self.key_prefix;
        let mut stripped = HashMap::with_capacity(block_weights.len());
        for (key, val) in block_weights {
            let k = key.strip_prefix(kp).unwrap_or(&key).to_string();
            let val = if val.dtype() == flame_core::DType::BF16 {
                val
            } else {
                val.to_dtype(flame_core::DType::BF16)?
            };
            stripped.insert(k, val);
        }

        println!(
            "    [offload] Loaded {} tensors for {prefix}",
            stripped.len()
        );
        self.cache = stripped;
        Ok(())
    }

    /// Drop all cached block weights to free VRAM.
    pub fn unload_block(&mut self) {
        self.cache.clear();
    }

    /// Look up a weight tensor by key. Checks the block cache first, then falls
    /// back to the provided resident weights map.
    pub fn get<'a>(
        &'a self,
        key: &str,
        resident: &'a HashMap<String, Tensor>,
    ) -> Result<&'a Tensor> {
        self.cache
            .get(key)
            .or_else(|| resident.get(key))
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight key: {key}")))
    }

    /// Direct access to the block cache (no resident fallback).
    pub fn cache(&self) -> &HashMap<String, Tensor> {
        &self.cache
    }

    /// Check whether a key exists in the block cache.
    pub fn cache_contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    /// Path to the underlying safetensors file.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// The CUDA device tensors are loaded onto.
    pub fn device(&self) -> &Arc<cudarc::driver::CudaDevice> {
        &self.device
    }
}
