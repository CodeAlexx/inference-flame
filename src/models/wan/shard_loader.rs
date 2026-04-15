//! Sharded safetensors index loader.
//!
//! Reads `diffusion_pytorch_model.safetensors.index.json` from a checkpoint
//! directory and returns the set of shard files that contain any weights
//! matching the caller's key filter. The returned list is sorted for
//! deterministic load order.
//!
//! The index JSON has the form:
//! ```json
//! {
//!   "metadata": { "total_size": 10215714816 },
//!   "weight_map": {
//!     "blocks.0.self_attn.q.weight": "diffusion_pytorch_model-00001-of-00007.safetensors",
//!     ...
//!   }
//! }
//! ```

use flame_core::{Error, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Parsed safetensors shard index.
pub struct ShardIndex {
    pub base_dir: PathBuf,
    /// (key, shard_filename) pairs.
    pub entries: Vec<(String, String)>,
}

impl ShardIndex {
    /// Load an index from `ckpt_dir/diffusion_pytorch_model.safetensors.index.json`.
    pub fn load_default(ckpt_dir: &Path) -> Result<Self> {
        Self::load_named(ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")
    }

    pub fn load_named(ckpt_dir: &Path, index_name: &str) -> Result<Self> {
        let path = ckpt_dir.join(index_name);
        let bytes = std::fs::read(&path).map_err(|e| {
            Error::InvalidInput(format!("Reading shard index {}: {e}", path.display()))
        })?;
        let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            Error::InvalidInput(format!("Parsing shard index {}: {e}", path.display()))
        })?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::InvalidInput(format!("{}: no weight_map", path.display())))?;

        let mut entries: Vec<(String, String)> = Vec::with_capacity(weight_map.len());
        for (k, v) in weight_map {
            let shard = v
                .as_str()
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "{}: non-string shard for key {k}",
                        path.display()
                    ))
                })?
                .to_string();
            entries.push((k.clone(), shard));
        }
        Ok(Self {
            base_dir: ckpt_dir.to_path_buf(),
            entries,
        })
    }

    /// All unique shard filenames in ascending order.
    pub fn all_shards(&self) -> Vec<PathBuf> {
        let mut set: BTreeSet<&str> = BTreeSet::new();
        for (_, s) in &self.entries {
            set.insert(s.as_str());
        }
        set.into_iter().map(|s| self.base_dir.join(s)).collect()
    }

    /// Shards that contain at least one key matching `pred`.
    pub fn shards_for<F: Fn(&str) -> bool>(&self, pred: F) -> Vec<PathBuf> {
        let mut set: BTreeSet<&str> = BTreeSet::new();
        for (k, s) in &self.entries {
            if pred(k) {
                set.insert(s.as_str());
            }
        }
        set.into_iter().map(|s| self.base_dir.join(s)).collect()
    }

    /// Sanity check: total number of distinct keys.
    pub fn num_keys(&self) -> usize {
        self.entries.len()
    }
}
