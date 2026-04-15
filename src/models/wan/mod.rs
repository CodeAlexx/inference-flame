//! Wan2.2 TI2V-5B — pure Rust T2V inference module.
//!
//! This module provides a text-to-video pipeline for the Wan2.2 TI2V-5B
//! checkpoint, targeting RTX 3090 (24GB) with sequential stage loading.
//!
//! ## Layout
//! - [`transformer`] — WanTransformer: TI2V-5B config + sharded loader,
//!   delegates block forward to the existing `Wan22Dit` implementation
//!   (identical block architecture, different dimensions).
//! - [`vae`] — Wan2.2 VAE decoder only (4×16×16 high-compression causal).
//! - [`t5`] — UMT5-XXL encoder matching Wan's native weight key format.
//! - [`rope3d`] — 3-axis RoPE frequency construction (lives in Wan22Dit
//!   already; this file documents the axis split for TI2V-5B head_dim=128).
//! - [`shard_loader`] — reads
//!   `diffusion_pytorch_model.safetensors.index.json` and returns the
//!   list of shard paths that contain any DiT weights.
//! - [`sampler`] — 50-step Euler flow-matching solver with shift=5.0.
//!
//! ## Pipeline (sequential, one component resident at a time)
//! 1. Load UMT5-XXL → encode prompt + negative → drop encoder.
//! 2. Load DiT shards via BlockOffloader → run Euler loop (50 steps × 2 CFG).
//! 3. Load VAE → decode latent → write MP4 via `mux::save_mp4_video_only`.
//!
//! ## Algorithmic sources
//! Port of the official Wan2.2 repo (Apache-2.0):
//!   - `wan/configs/wan_ti2v_5B.py` — config constants
//!   - `wan/modules/model.py::WanModel` — transformer (same block as A14B)
//!   - `wan/modules/vae2_2.py::Wan2_2_VAE` — high-compression VAE
//!   - `wan/modules/t5.py::umt5_xxl` — text encoder
//! `diffusers.models.WanTransformer3DModel` / `AutoencoderKLWan` were used
//! as a cross-reference for ordering and activation choices.

pub mod rope3d;
pub mod sampler;
pub mod shard_loader;
pub mod t5;
pub mod transformer;
pub mod vae;
pub mod encoder;

pub use sampler::{euler_step, shifted_sigma_schedule, EulerSigmas};
pub use shard_loader::ShardIndex;
pub use t5::{Umt5Config, Umt5Encoder};
pub use transformer::{WanConfig, WanTransformer};
pub use vae::{Wan22VaeConfig, Wan22VaeDecoder};
pub use encoder::Wan22VaeEncoder;
