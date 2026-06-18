//! PiD (Pixel Diffusion Decoder) — pure-Rust port of NVIDIA's PiD super-res
//! pixel-diffusion decoder.
//!
//! Source of truth: github.com/nv-tlabs/PiD (Apache-2.0), PyTorch reference
//! read in full from the local clone at `/tmp/PiD_repo` during the port:
//!   - `pid/_src/networks/pixeldit_official.py` (PixDiT_T2I, MMDiTBlockT2I,
//!     PiTBlock, RotaryAttention, RMSNorm, FeedForward, RoPE precompute)
//!   - `pid/_src/networks/pid_net.py`           (PidNet — LQ injection subclass)
//!   - `pid/_src/networks/lq_projection_2d.py`  (LQProjection2D + sigma gate)
//!   - `pid/_src/models/pid_distill_model.py`   (4-step distilled SDE sampler)
//!
//! Weights: HF `nvidia/PiD`, e.g.
//! `checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.safetensors`.
//! Released SD3 ckpt config (verified against the 456-key safetensors header):
//!   hidden=1536, num_groups=24 (head_dim=64), patch_depth=14,
//!   pixel_hidden_size=16, pixel_depth=2, pixel_attn_hidden_size=1152,
//!   pixel_num_groups=16 (pixel head_dim=72), patch_size=16, txt_embed_dim=2304,
//!   txt_max_length=300, rope_mode="ntk_aware", rope_ref_h=rope_ref_w=1024
//!   (=> rope_ref_grid = 1024/16 = 64), use_text_rope=True, text_rope_theta=1e4,
//!   lq_latent_channels=16, lq_hidden_dim=512, lq_num_res_blocks=4,
//!   lq_interval=2 (=> 7 output_heads + 7 gate_modules), in_channels=3,
//!   enable_ed=False, latent-only LQ branch (in_channels for LQ image branch = 0).
//!
//! See `model.rs` for the network forward and `sampler.rs` for the 4-step loop.

pub mod model;
pub mod sampler;

pub use model::{load_pid_resident, PidConfig, PidNet};
pub use sampler::{pid_student_sample, PidSamplerConfig};
