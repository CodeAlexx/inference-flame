//! KV cache for Gemma-4 autoregressive decode.
//!
//! Per-layer (K, V) BF16 tensors that grow each decode step. Two
//! variants per cache reflecting Gemma-4's mixed attention pattern:
//!
//! - **Full**: keep all K, V columns since the layer's first step. K
//!   shape `[B, num_kv_heads, seq, head_dim]`, grown along dim=2.
//! - **Sliding**: keep only the last `window_size` columns. Implemented
//!   as a grow-then-trim buffer (we hold up to `sliding_window` columns
//!   in logical order; new appends past the cap drop the oldest column).
//!
//! ## Allocation strategy — divergence from the skeleton docs
//!
//! AGENT-DEFAULT (Builder 3, 2026-05-21): The original skeleton called
//! for a fixed-capacity scatter-write buffer (pre-allocate `[B, H, max_seq, D]`
//! once, write into slices on each step). flame-core today has no
//! scatter-into-tensor-slice primitive; the public surface for
//! growing K/V across decode steps is `Tensor::cat`. We follow the
//! pattern already in production at
//! `inference-flame/src/models/sensenova_u1.rs:1603` — `cat` to grow,
//! re-bind the cache slot to the result. This:
//!
//! - Avoids inventing a new flame-core primitive (TENET 5: don't push
//!   a kernel-shaped fix into a model file).
//! - Matches the sensenova_u1 8B-MoT decode path, which has been
//!   smoke-tested at 1024² / 2048².
//! - Pays the same per-step alloc-and-copy cost that path pays today.
//!
//! For sliding layers, after `append` we trim the cache to the most
//! recent `sliding_window` columns via `narrow_owning(2, …)`. The
//! `ring_head` field is therefore unused in this impl — kept in the
//! struct so the API matches a future ring-buffer rewrite when
//! flame-core ships an in-place scatter primitive.
//!
//! ## Memory
//!
//! At 6K decode (system ~3K + user ~0.2K + gen ~3K), per-layer cost:
//! `2 (K,V) × 16 kv_heads × 6000 × 256 head_dim × 2 B = ~98 MB`
//! for a full layer, capped at `2 × 16 × 1024 × 256 × 2 B = ~17 MB`
//! for a sliding layer. The cat-grow strategy peaks at the full size
//! during the cat call but releases the old tensor immediately after.

use crate::models::gemma4::{Gemma4Config, LayerType};
use flame_core::{DType, Result, Shape, Tensor};

/// Per-layer KV cache slot. Lives for one inference invocation; freed
/// when the LM call returns.
pub struct Gemma4LayerCache {
    /// `[B, num_kv_heads, valid_len, head_dim]` BF16 — grows via `cat` on `append`.
    /// During decode the tensor's dim-2 length equals `valid_len`; we
    /// trim on overflow for sliding layers.
    pub k: Tensor,
    /// Same shape as `k`.
    pub v: Tensor,
    /// Layer type drives the lookup semantics (full vs sliding window).
    pub layer_type: LayerType,
    /// Sliding-window width for `LayerType::Sliding`. Ignored for full.
    pub sliding_window: usize,
    /// Number of valid columns currently held. Grows by `S` per `append`;
    /// for sliding layers, capped at `sliding_window`.
    pub valid_len: usize,
    /// Reserved for a future fixed-capacity ring-buffer impl. Unused by
    /// the current cat-grow strategy. Kept so callers / Builder-2 layer
    /// code doesn't need to change when we revisit the allocation strategy.
    pub ring_head: usize,
}

impl Gemma4LayerCache {
    /// Allocate a fresh layer cache.
    ///
    /// Per the cat-grow strategy documented at module level, we start
    /// with a zero-length tensor on dim=2 and let `append` materialize
    /// the storage. The `_max_seq` argument is retained from the
    /// skeleton API for forward-compat with a fixed-capacity rewrite.
    pub fn new(
        cfg: &Gemma4Config,
        layer_idx: usize,
        batch: usize,
        _max_seq: usize,
        device: &flame_core::Device,
    ) -> Result<Self> {
        let layer_type = cfg.layer_types[layer_idx];
        let dev_arc = device.cuda_device_arc();
        // Start with a zero-length tensor along the seq dim. Subsequent
        // `append` calls grow this via cat. We use `zeros_dtype` rather
        // than `empty_dtype` because cat reads from this on the very
        // first append (no-op if `dim=2` length is 0, but `cat` does
        // shape validation up front).
        let init_shape = Shape::from_dims(&[batch, cfg.num_key_value_heads, 0, cfg.head_dim]);
        let k = Tensor::zeros_dtype(init_shape.clone(), DType::BF16, dev_arc.clone())?;
        let v = Tensor::zeros_dtype(init_shape, DType::BF16, dev_arc)?;
        Ok(Self {
            k,
            v,
            layer_type,
            sliding_window: cfg.sliding_window,
            valid_len: 0,
            ring_head: 0,
        })
    }

    /// Append one decode step's K and V to the cache.
    ///
    /// `new_k`, `new_v`: shape `[B, num_kv_heads, S_new, head_dim]`. For
    /// prefill, `S_new = prompt_len`; for decode, `S_new = 1`. After
    /// this call `self.k` / `self.v` hold the updated (and possibly
    /// front-trimmed for sliding) sequence.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let new_dims = new_k.shape().dims().to_vec();
        if new_dims.len() != 4 {
            return Err(flame_core::Error::InvalidInput(format!(
                "Gemma4LayerCache::append: new_k must be 4D [B,H,S,D], got {new_dims:?}"
            )));
        }
        let s_new = new_dims[2];

        // Concatenate along the sequence dim. On the first call (valid_len==0)
        // this is effectively the new tensor; on subsequent calls we grow.
        let k_grown = if self.valid_len == 0 {
            new_k.clone()
        } else {
            Tensor::cat(&[&self.k, new_k], 2)?
        };
        let v_grown = if self.valid_len == 0 {
            new_v.clone()
        } else {
            Tensor::cat(&[&self.v, new_v], 2)?
        };

        self.valid_len += s_new;

        // Sliding layers: trim the front so we keep only the latest `sliding_window` cols.
        // AGENT-DEFAULT: trim via narrow_owning. This materializes a contiguous
        // tensor; without it, attention would see a strided view (slow path).
        if self.layer_type == LayerType::Sliding && self.valid_len > self.sliding_window {
            let total = self.valid_len;
            let start = total - self.sliding_window;
            let len = self.sliding_window;
            self.k = k_grown.narrow_owning(2, start, len)?;
            self.v = v_grown.narrow_owning(2, start, len)?;
            self.valid_len = self.sliding_window;
        } else {
            self.k = k_grown;
            self.v = v_grown;
        }
        Ok(())
    }

    /// Fetch the active K, V for attention.
    ///
    /// Under the cat-grow strategy, `self.k` / `self.v` are already
    /// stored in logical order with `valid_len` along dim=2 — no roll
    /// needed. We return clones (cheap — Arc-shared storage).
    pub fn fetch(&self) -> Result<(Tensor, Tensor)> {
        Ok((self.k.clone(), self.v.clone()))
    }
}

/// Top-level KV cache: one slot per decoder layer.
pub struct Gemma4KvCache {
    pub layers: Vec<Gemma4LayerCache>,
}

impl Gemma4KvCache {
    pub fn new(
        cfg: &Gemma4Config,
        batch: usize,
        max_seq: usize,
        device: &flame_core::Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma4LayerCache::new(cfg, i, batch, max_seq, device)?);
        }
        Ok(Self { layers })
    }
}
