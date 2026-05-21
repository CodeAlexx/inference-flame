//! `cosmos_predict25_v2v_infer` — video-conditioned video inference for
//! Cosmos-Predict2.5-2B.
//!
//! Pipeline:
//! - Stage 1: Cosmos-Reason1-7B text encode (prompt + negative).
//! - Stage 2: Wan21 VAE encode the last `4*(num_latent_cond-1) + 1` pixel
//!   frames of the input video, padded with the last frame to `num_frames`.
//!   `num_latent_conditional_frames` is 1 or 2 (Python `:211-212`); default 2.
//! - Stage 3: DiT denoise with conditioning latent mixing (same convention as i2v).
//! - Stages 4-5: VAE decode + mp4 mux.
//!
//! Requires `ffmpeg` and `ffprobe` on PATH (used to extract pixel frames from
//! the input video).
//!
//! Worked example (480p, 35-step UniPC, 2 latent conditional frames):
//!
//! ```text
//! RUST_LOG=info COSMOS_REASON1_PATH=/path/to/Cosmos-Reason1-7B \
//!     COSMOS_REASON1_TOKENIZER=/path/to/Cosmos-Reason1-7B/tokenizer.json \
//!     WAN21_VAE_COSMOS_SAFETENSORS=/path/to/wan21_vae.safetensors \
//!     cargo run --release --bin cosmos_predict25_v2v_infer -- \
//!         --prompt "The dog now leaps over the fence" \
//!         --input-video ./input_clip.mp4 \
//!         --num-latent-conditional-frames 2 \
//!         --num-frames 81 --num-steps 35 --cfg 7.0 --seed 42 \
//!         --resolution 480p --variant post-trained --sampler unipc \
//!         --dit-path /path/to/cosmos_predict25_2b_dit.safetensors \
//!         --output-dir ./output
//! ```

#[path = "cosmos_predict25_common.rs"]
mod common;

fn main() -> anyhow::Result<()> {
    common::run(common::Mode::V2V)
}
