//! HiDream-O1-Image — pure-Rust port (Phase 2a foundational primitives).
//!
//! HiDream-O1 is **Qwen3-VL 8B with three new heads bolted on** for
//! one-pass image diffusion:
//!
//! - `BottleneckPatchEmbed` — `Linear(32*32*3 → 1024) → Linear(1024 → 4096)`
//!   (`qwen3_vl_transformers.py:944-960`)
//! - `TimestepEmbedder` — sinusoidal(256) → Linear → SiLU → Linear → 4096
//!   (`qwen3_vl_transformers.py:980-1019`)
//! - `FinalLayer` — `Linear(4096 → 32*32*3)` (`qwen3_vl_transformers.py:962-976`)
//!
//! The MRoPE in `mrope.rs` is the **interleaved** variant from
//! `qwen3_vl_transformers.py:315-330` — frequencies are stride-3 woven
//! across the T/H/W axes, NOT chunked-then-concatenated. See edge case
//! #1 in `/tmp/hidream_scope_bugs.md`.
//!
//! Phase 2a scope: just the four primitive modules + a Config struct.
//! The Qwen3-VL text spine, attention mask builder, scheduler, and
//! pipeline live in later phases.

pub mod bottleneck_patch_embed;
pub mod decoder;
pub mod final_layer;
pub mod model;
pub mod mrope;
pub mod pipeline;
pub mod scheduler;
pub mod timestep_embedder;
pub mod weight_loader;

pub use bottleneck_patch_embed::BottleneckPatchEmbed;
pub use decoder::HiDreamDecoderLayer;
pub use final_layer::FinalLayer;
pub use model::HiDreamO1Model;
pub use weight_loader::HiDreamO1WeightLoader;
pub use mrope::{
    apply_interleaved_mrope, apply_mrope, build_mrope_positions, interleaved_mrope_cos_sin,
    MRopePositions,
};
pub use pipeline::{
    find_closest_resolution, HiDreamO1Pipeline, MRopePositionsOwned, PREDEFINED_RESOLUTIONS,
};
pub use scheduler::{
    FlashFlowMatchEulerDiscreteScheduler, SchedulerMode, DEFAULT_TIMESTEPS_DEV,
};
pub use timestep_embedder::TimestepEmbedder;

/// Configuration for HiDream-O1-Image.
///
/// Default values via `Self::dev_8b()` correspond to the
/// HiDream-ai/HiDream-O1-Image-Dev variant (28-step flash scheduler).
/// All numeric values are sourced from
/// `qwen3_vl_transformers.py` (per-line citations beside each field).
#[derive(Clone, Debug)]
pub struct HiDreamO1Config {
    /// Qwen3-VL text hidden size (`hidden_size` from `Qwen3VLTextConfig`).
    /// 4096 for the 8B variant.
    pub hidden_size: usize,
    /// Number of decoder layers (`num_hidden_layers`). 36 for 8B.
    pub num_layers: usize,
    /// Number of attention heads (`num_attention_heads`). 32 for 8B.
    pub num_attention_heads: usize,
    /// Number of KV heads for GQA (`num_key_value_heads`). 8 for 8B → 32:8 = 4:1.
    pub num_kv_heads: usize,
    /// Per-head dimension. 128 for 8B (`hidden_size // num_attention_heads` fallback,
    /// `qwen3_vl_transformers.py:412`).
    pub head_dim: usize,
    /// MLP intermediate size (`intermediate_size`). 12288 for 8B.
    pub intermediate_size: usize,
    /// RoPE θ (`rope_theta`). 5_000_000.0 for Qwen3-VL.
    pub rope_theta: f32,
    /// MRoPE T/H/W section split (`mrope_section`). [24, 20, 20] for Qwen3-VL.
    /// Sums to 64 = `head_dim / 2`.
    /// Reference: `qwen3_vl_transformers.py:313`.
    pub mrope_section: [usize; 3],
    /// Vocab size (`vocab_size`). ~152K for Qwen3-VL.
    pub vocab_size: usize,
    /// RMSNorm epsilon (`rms_norm_eps`). 1e-6 default.
    pub rms_norm_eps: f32,
    /// Whether attention QKV/O have biases (`attention_bias`). Per the Qwen3 family
    /// 8B+ models this is typically False; verify against `config.json`.
    pub attention_bias: bool,

    // ─── HiDream-O1 specific ─────────────────────────────────────────
    /// Image patch size (P). 32 for HiDream-O1
    /// (`qwen3_vl_transformers.py:1036`).
    pub patch_size: usize,
    /// Image input channels (RGB). 3 (`qwen3_vl_transformers.py:1037`).
    pub patch_in_channels: usize,
    /// `BottleneckPatchEmbed` middle dimension. `hidden_size / 4` per
    /// `qwen3_vl_transformers.py:1039` → 1024 for 8B.
    pub bottleneck_dim: usize,
    /// `<|tms_token|>` token id. Hardcoded at
    /// `qwen3_vl_transformers.py:1047` to 151673; **must be validated
    /// against the real tokenizer at load time** (edge case #3 in
    /// `/tmp/hidream_scope_bugs.md`).
    pub tms_token_id: u32,
    /// `<|image_pad|>` / `image_token_id` (the L gen-image slot id).
    /// Read from tokenizer.
    pub image_token_id: u32,
    /// `<|video_pad|>` / `video_token_id`. Currently unused for T2I but kept
    /// so the position-id builder matches Python signatures.
    pub video_token_id: u32,
    /// `<|vision_start|>` / `vision_start_token_id`. Read from tokenizer.
    pub vision_start_token_id: u32,
    /// RoPE position-id anchor for the **first** generation image.
    /// 4096 per `utils.py:86` and the spec.
    pub fix_point: usize,
    /// Frequency-embedding dimension fed into `TimestepEmbedder`'s sinusoid.
    /// 256 per `qwen3_vl_transformers.py:984`.
    pub timestep_freq_dim: usize,
}

impl HiDreamO1Config {
    /// HiDream-O1-Dev (28-step) defaults aligned with the Qwen3-VL 8B
    /// architecture. The vocab/token-id fields are placeholders and
    /// **must be replaced** with values read from the actual tokenizer
    /// at load time. See edge case #3 in `/tmp/hidream_scope_bugs.md`.
    pub fn dev_8b() -> Self {
        let hidden_size = 4096;
        Self {
            hidden_size,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            // 151_936 per HiDream-O1-Image-Dev-weights/config.json:34
            // (`text_config.vocab_size`); the same value the Qwen3-VL
            // tokenizer rounds-up. Phase 2a originally guessed 152_064.
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            attention_bias: false,

            patch_size: 32,
            patch_in_channels: 3,
            bottleneck_dim: hidden_size / 4, // = 1024
            // Placeholders — verify against tokenizer.json at load time.
            tms_token_id: 151_673,
            image_token_id: 151_655,
            video_token_id: 151_656,
            vision_start_token_id: 151_652,
            fix_point: 4096,
            timestep_freq_dim: 256,
        }
    }
}
