//! HiDream-I1 DiT — pure-Rust port (M2 of the I1 training stack).
//!
//! HiDream-I1 is a FLUX-derived double + single stream DiT with three
//! I1-specific changes vs FLUX/Chroma:
//!   1. **MoE FFN** (shared expert + top-k routed SwiGLU experts) inside
//!      every block instead of a dense MLP.
//!   2. **Four text encoders** — CLIP-L (pooled), CLIP-G (pooled, concatenated
//!      to make a 2048-d `pooled_embeds`), T5-XXL (sequence), Llama-3.1-8B
//!      (per-layer hidden states fed block-by-block).
//!   3. **adaLN modulation** computed from `t_embedder(t) + p_embedder(pooled)`
//!      (single block emits 6 chunks, double block emits 12).
//!
//! ## File map
//! - [`model::HiDreamI1Dit`] — top-level DiT wired through [`BlockOffloader`].
//! - [`double_block`] / [`single_block`] — per-block forwards.
//! - [`moe`] — MoEGate + MOEFeedForwardSwiGLU (shared + N routed experts).
//! - [`final_layer`] — adaLN + Linear projection back to patch space.
//! - [`weight_loader`] — BlockFacilitator + shared-weight loader.
//! - [`text_encoder_cat`] — helpers that pack the 4-encoder outputs into the
//!   list format the DiT expects.
//!
//! ## Architecture references
//! - `edv2-reference/extensions_built_in/diffusion_models/hidream/src/models/transformers/transformer_hidream_image.py`
//! - `edv2-reference/.../hidream/src/models/{embeddings,attention,attention_processor,moe}.py`
//! - `edv2-reference/.../hidream/src/pipelines/hidream_image/pipeline_hidream_image.py`
//!
//! ## Scheduler
//! The flow-matching Euler scheduler is shared with hidream_o1 in concept,
//! but I1 inference uses the diffusers-stock variant (no stochastic noise
//! re-injection) plus an optional UniPC. For now, reuse
//! [`crate::models::hidream_o1::scheduler::FlashFlowMatchEulerDiscreteScheduler`]'s
//! `Default` mode (`full_n_step`) which already implements the deterministic
//! flow-match step that training and stock inference use.

pub mod model;
pub mod double_block;
pub mod single_block;
pub mod moe;
pub mod weight_loader;
pub mod text_encoder_cat;
pub mod final_layer;

pub use model::{HiDreamI1Config, HiDreamI1Dit};
pub use moe::{moe_ffn_forward, MoeWeights};
pub use text_encoder_cat::{HiDreamEncoderInputs, OwnedEncoderInputs};
pub use weight_loader::HiDreamI1Facilitator;
