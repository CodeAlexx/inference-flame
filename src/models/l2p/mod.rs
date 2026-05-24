//! L2P (T2I-L2P, Tencent Youtu) — pure-Rust inference port.
//!
//! L2P is the **Z-Image-Turbo DiT body + 16×16 pixel-space patchify +
//! MicroDiffusionModel U-Net head**, with the original `FinalLayer +
//! unpatchify` removed. Output is direct pixels, not VAE latents.
//!
//! Per the anima-cosmos rule, the DiT body is copied as patterns from
//! `inference_flame::models::zimage_nextdit` rather than cross-imported.
//! See `PORT_SPEC.md` and `BUILD_PLAN.md` in this directory for the full
//! architecture description, weight key map, and per-module build order.
//!
//! Wave 1 chunk 1 (this commit): config + RoPE + DiT struct scaffold.
//! No forward pass, no weight loader, no pipeline yet — those land in
//! later chunks per `BUILD_PLAN.md`.

pub mod block_trap;
pub mod dit;
pub mod local_decoder;
pub mod rope;
pub mod weight_loader;

pub use dit::{L2pDiT, L2pDiTConfig};
pub use local_decoder::MicroDiffusionModel;
pub use rope::build_3d_rope;
pub use weight_loader::load_l2p_safetensors;
