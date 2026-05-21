//! `cosmos_predict25_i2v_infer` — image-conditioned video inference for
//! Cosmos-Predict2.5-2B.
//!
//! Pipeline:
//! - Stage 1: Cosmos-Reason1-7B text encode (prompt + negative).
//! - Stage 2: Wan21 VAE encode the input image as a single-latent-frame
//!   conditioning signal (`num_latent_conditional_frames = 1`). Python's
//!   `read_and_process_image` pads frames 1..num_frames with zeros; the VAE
//!   compresses this so latent-frame 0 carries the image and the rest are
//!   masked out during denoise.
//! - Stage 3: DiT denoise. Each step mixes `gt_lat * mask + xt * (1 - mask)`
//!   into the input latent BEFORE the forward pass, then replaces the
//!   velocity on conditional positions with the analytical `noise - gt_lat`
//!   (Python `:130-135`).
//! - Stages 4-5: VAE decode + mp4 mux.
//!
//! Worked example (480p, 35-step UniPC):
//!
//! ```text
//! RUST_LOG=info COSMOS_REASON1_PATH=/path/to/Cosmos-Reason1-7B \
//!     COSMOS_REASON1_TOKENIZER=/path/to/Cosmos-Reason1-7B/tokenizer.json \
//!     WAN21_VAE_COSMOS_SAFETENSORS=/path/to/wan21_vae.safetensors \
//!     cargo run --release --bin cosmos_predict25_i2v_infer -- \
//!         --prompt "Continue the scene: the cat jumps onto the fence" \
//!         --input-image ./input.png \
//!         --num-frames 81 --num-steps 35 --cfg 7.0 --seed 42 \
//!         --resolution 480p --variant post-trained --sampler unipc \
//!         --dit-path /path/to/cosmos_predict25_2b_dit.safetensors \
//!         --output-dir ./output
//! ```

#[path = "cosmos_predict25_common.rs"]
mod common;

fn main() -> anyhow::Result<()> {
    common::run(common::Mode::I2V)
}
