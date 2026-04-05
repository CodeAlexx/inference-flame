//! # flame-swap
//!
//! Double-buffered async block swapper for FLAME inference.
//!
//! Loads transformer block weights into pinned CPU memory at init, then
//! overlaps GPU compute with async H2D transfers using two CUDA streams.
//! No scheduling, no watermarks — just prefetch-next-while-current-runs.
//!
//! ## Example
//!
//! ```ignore
//! use flame_swap::FlameSwap;
//!
//! // Load model — only block weights go into pinned memory.
//! // Shared weights (embeddings, final norm) are loaded separately.
//! let mut swap = FlameSwap::load(
//!     &["model-00001.safetensors", "model-00002.safetensors"],
//!     &device,
//!     |name| {
//!         // "transformer_blocks.7.attn.qkv.weight" → Some(7)
//!         let rest = name.strip_prefix("transformer_blocks.")?;
//!         let idx_str = rest.split('.').next()?;
//!         idx_str.parse().ok()
//!     },
//! )?;
//!
//! // Kick off first prefetch
//! swap.prefetch(0)?;
//!
//! for i in 0..swap.num_blocks() {
//!     // Overlap: start loading next block while current one computes
//!     if i + 1 < swap.num_blocks() {
//!         swap.prefetch(i + 1)?;
//!     }
//!
//!     // Wait for current block's weights to land on GPU
//!     let weights = swap.await_block(i)?;
//!
//!     // Run the block (weights are a HashMap<String, Tensor>)
//!     x = block_forward(&x, &weights, ...);
//!
//!     // `weights` dropped here → GPU memory returns to CUDA mempool
//! }
//! ```

mod ffi;
mod swap;

pub use swap::FlameSwap;
