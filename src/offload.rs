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
    /// Currently loaded block weights (keyed by full weight name).
    cache: HashMap<String, Tensor>,
}

impl BlockLoader {
    /// Create a new block loader for the given safetensors file.
    pub fn new(model_path: String, device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self {
            model_path,
            device,
            cache: HashMap::new(),
        }
    }

    /// Load all weights whose key starts with `prefix.` into GPU memory.
    /// Any previously cached block is dropped first to free VRAM.
    pub fn load_block(&mut self, prefix: &str) -> Result<()> {
        self.cache.clear();

        let prefix_dot = format!("{prefix}.");
        let block_weights = load_file_filtered(&self.model_path, &self.device, |key| {
            key.starts_with(&prefix_dot)
        })?;

        println!(
            "    [offload] Loaded {} tensors for {prefix}",
            block_weights.len()
        );
        self.cache = block_weights;
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
