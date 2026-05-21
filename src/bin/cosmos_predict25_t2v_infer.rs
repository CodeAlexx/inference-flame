//! `cosmos_predict25_t2v_infer` — text-to-video inference for
//! Cosmos-Predict2.5-2B.
//!
//! Strict 24 GB-safe stage discipline (text encoder → DiT → VAE decoder,
//! each loaded then dropped before the next). See
//! `cosmos_predict25_common.rs` for shared helpers and the CLI surface.
//!
//! Worked example (480p, 35-step UniPC, default post-trained variant):
//!
//! ```text
//! RUST_LOG=info COSMOS_REASON1_PATH=/path/to/Cosmos-Reason1-7B \
//!     COSMOS_REASON1_TOKENIZER=/path/to/Cosmos-Reason1-7B/tokenizer.json \
//!     WAN21_VAE_COSMOS_SAFETENSORS=/path/to/wan21_vae.safetensors \
//!     cargo run --release --bin cosmos_predict25_t2v_infer -- \
//!         --prompt "A cat running across a field at sunset" \
//!         --negative-prompt "blurry, low quality" \
//!         --num-frames 81 --num-steps 35 --cfg 7.0 --seed 42 \
//!         --resolution 480p --variant post-trained \
//!         --sampler unipc \
//!         --dit-path /path/to/cosmos_predict25_2b_dit.safetensors \
//!         --output-dir ./output
//! ```

#[path = "cosmos_predict25_common.rs"]
mod common;

fn main() -> anyhow::Result<()> {
    common::run(common::Mode::T2V)
}
