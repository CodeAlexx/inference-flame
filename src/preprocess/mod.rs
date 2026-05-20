//! Shared image preprocessing utilities for multimodal model ports.
//!
//! Layer split:
//! - `common`: scaffolding shared across all multimodal ports — image load,
//!   RGBA→RGB composite, CHW-normalize, resize wrappers, smart_resize.
//! - Future model-specific submodules sit on top.

pub mod bucket_resize;
pub mod common;
pub mod qwen25vl;
