//! Boogu-Image-0.1-Base — transformer architecture config.
//!
//! Mirrors `transformer/config.json`
//! (`/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer/config.json`,
//! `_class_name = "BooguImageTransformer2DModel"`) field-for-field, cross-checked
//! against the verified Mojo port handoff
//! (`/home/alex/mojodiffusion/serenitymojo/docs/BOOGU_PORT_HANDOFF.md`,
//! "Transformer config" section) and PORT_SPEC.md.
//!
//! Boogu is a two-stage flow-matching DiT (Lumina2/OmniGen2 lineage):
//! **8 double-stream** joint-attention blocks then **32 single-stream** blocks
//! on the fused sequence, plus **2 context-refiner** (modulation=False) +
//! **2 noise-refiner** (modulation=True) blocks. Velocity prediction, Euler
//! flow-matching; the single transformer is run twice for CFG (NOT a
//! dual-transformer like Ideogram-4). bf16 throughout — NO quantization.
//!
//! This chunk (C1/C2 foundation) only needs the embedder + RoPE constants, but
//! the full config is encoded here so later chunks have a single source of
//! truth.

/// Boogu-Image transformer architecture constants.
///
/// Mirrors `transformer/config.json` exactly. `out_channels` in the JSON is
/// `null`, which the reference resolves to `in_channels` (16) — encoded here as
/// the resolved value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BooguConfig {
    /// Transformer hidden size. (`hidden_size = 3360`)
    pub hidden_size: usize,
    /// Total transformer layers (8 double-stream → 32 single-stream).
    /// (`num_layers = 40`)
    pub num_layers: usize,
    /// Double-stream (joint-attention) block count. (`num_double_stream_layers = 8`)
    pub num_double_stream_layers: usize,
    /// Refiner block count PER refiner kind: context_refiner ×2, noise_refiner
    /// ×2. (`num_refiner_layers = 2`)
    pub num_refiner_layers: usize,
    /// Attention query heads. (`num_attention_heads = 28`)
    pub num_attention_heads: usize,
    /// Attention key/value heads (GQA 4:1, repeat_kv ×4). (`num_kv_heads = 7`)
    pub num_kv_heads: usize,
    /// Per-head dimension. `hidden_size / num_attention_heads = 3360/28 = 120`.
    /// NOTE: 120 ∉ {64,96,128} so the fused cuDNN SDPA path pads to 128 (later
    /// chunk concern; recorded here for completeness).
    pub head_dim: usize,
    /// Per-axis RoPE rotary dims. (`axes_dim_rope = [40, 40, 40]`, sum = 120 =
    /// head_dim). Each axis contributes `40/2 = 20` complex freqs; the 3 axes
    /// concat to `head_dim/2 = 60` cos/sin table columns.
    pub axes_dim_rope: [usize; 3],
    /// Max position-id table length per axis. (`axes_lens = [2048, 1664, 1664]`)
    pub axes_lens: [usize; 3],
    /// Latent channels in / out. (`in_channels = 16`; JSON `out_channels = null`
    /// → resolved to `in_channels`.)
    pub in_channels: usize,
    /// Resolved output channels (= `in_channels`, since JSON `out_channels` is
    /// null).
    pub out_channels: usize,
    /// Patchify factor. (`patch_size = 2`) → x_embedder in = `patch²·in_channels
    /// = 2·2·16 = 64`.
    pub patch_size: usize,
    /// SwiGLU inner-dim rounding multiple. (`multiple_of = 256`) → SwiGLU inner
    /// = 13568 for hidden_size 3360.
    pub multiple_of: usize,
    /// Block RMSNorm epsilon. (`norm_eps = 1e-5`). NOTE: `norm_out`
    /// (LayerNormContinuous) uses a SEPARATE eps of **1e-6** — see
    /// [`BooguConfig::NORM_OUT_EPS`].
    pub norm_eps: f32,
    /// RoPE base θ. (`theta = 10000`).
    pub theta: f32,
    /// Timestep pre-scale fed into the sinusoid (`Timesteps(scale=...)`).
    /// (`timestep_scale = 1000.0`).
    pub timestep_scale: f32,
    /// Instruction (caption) feature dim before the caption embedder.
    /// (`instruction_feature_configs.instruction_feat_dim = 4096`;
    /// `num_instruction_feature_layers = 1`, `reduce = mean` → identity).
    pub instruction_feat_dim: usize,
    /// Prompt-tuning is DISABLED (`prompt_tuning_configs.use_prompt_tuning =
    /// false`) → the PromptEmbedding path is skipped entirely.
    pub use_prompt_tuning: bool,
}

impl BooguConfig {
    /// `x_embedder` input width: `patch_size² · in_channels` (= 64).
    pub const X_EMBEDDER_IN: usize = 64;
    /// Sinusoidal timestep frequency-embedding size fed to TimestepEmbedder
    /// (`frequency_embedding_size = 256`, the diffusers `Timesteps`
    /// `num_channels`).
    pub const FREQ_EMBEDDING_SIZE: usize = 256;
    /// TimestepEmbedder hidden/output width: `min(hidden_size, 1024) = 1024`
    /// (block_lumina2.py:196).
    pub const TIME_EMBED_DIM: usize = 1024;
    /// `Timesteps` `max_period` (diffusers default). Distinct from RoPE θ.
    pub const TIMESTEP_MAX_PERIOD: f32 = 10000.0;
    /// `norm_out` (LuminaLayerNormContinuous) eps — **1e-6**, NOT the block
    /// `norm_eps` (1e-5). Recorded for the final-layer chunk.
    pub const NORM_OUT_EPS: f32 = 1e-6;

    /// GQA repeat factor: `num_attention_heads / num_kv_heads` (= 4).
    #[inline]
    pub fn gqa_repeat(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads
    }

    /// `head_dim / 2` — the cos/sin RoPE table column count (= 60).
    #[inline]
    pub fn rope_half(&self) -> usize {
        self.head_dim / 2
    }

    /// SwiGLU inner dim: `multiple_of * ceil(inner / multiple_of)` where the
    /// block passes `inner = 4 * hidden_size` to `LuminaFeedForward`
    /// (transformer_boogu.py:232 `inner_dim=4 * dim`). `LuminaFeedForward` does
    /// NOT apply a `2/3` scaling (unlike a stock Llama SwiGLU) — it only applies
    /// `ffn_dim_multiplier` (null here) then rounds up to `multiple_of`
    /// (block_lumina2.py:151-153). So for hidden 3360:
    /// `4*3360 = 13440 → round_up(13440, 256) = 13568` (the shipped value).
    #[inline]
    pub fn swiglu_inner(&self) -> usize {
        let inner = 4 * self.hidden_size;
        self.multiple_of * inner.div_ceil(self.multiple_of)
    }

    /// Number of single-stream (post-fusion) blocks: `num_layers -
    /// num_double_stream_layers` (= 32).
    #[inline]
    pub fn num_single_stream_layers(&self) -> usize {
        self.num_layers - self.num_double_stream_layers
    }
}

impl Default for BooguConfig {
    /// The shipped `Boogu-Image-0.1-Base` transformer config.
    fn default() -> Self {
        Self {
            hidden_size: 3360,
            num_layers: 40,
            num_double_stream_layers: 8,
            num_refiner_layers: 2,
            num_attention_heads: 28,
            num_kv_heads: 7,
            head_dim: 120,
            axes_dim_rope: [40, 40, 40],
            axes_lens: [2048, 1664, 1664],
            in_channels: 16,
            out_channels: 16,
            patch_size: 2,
            multiple_of: 256,
            norm_eps: 1e-5,
            theta: 10000.0,
            timestep_scale: 1000.0,
            instruction_feat_dim: 4096,
            use_prompt_tuning: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_config_json() {
        let c = BooguConfig::default();
        assert_eq!(c.hidden_size, 3360);
        assert_eq!(c.num_layers, 40);
        assert_eq!(c.num_double_stream_layers, 8);
        assert_eq!(c.num_refiner_layers, 2);
        assert_eq!(c.num_attention_heads, 28);
        assert_eq!(c.num_kv_heads, 7);
        assert_eq!(c.head_dim, 120);
        assert_eq!(c.axes_dim_rope, [40, 40, 40]);
        assert_eq!(c.axes_lens, [2048, 1664, 1664]);
        assert_eq!(c.in_channels, 16);
        assert_eq!(c.out_channels, 16); // JSON null → in_channels
        assert_eq!(c.patch_size, 2);
        assert_eq!(c.multiple_of, 256);
        assert_eq!(c.norm_eps, 1e-5);
        assert_eq!(c.theta, 10000.0);
        assert_eq!(c.timestep_scale, 1000.0);
        assert_eq!(c.instruction_feat_dim, 4096);
        assert!(!c.use_prompt_tuning);
    }

    #[test]
    fn derived_constants() {
        let c = BooguConfig::default();
        // head_dim = hidden_size / num_attention_heads.
        assert_eq!(c.head_dim, c.hidden_size / c.num_attention_heads);
        // axes_dim_rope sums to head_dim (120).
        assert_eq!(c.axes_dim_rope.iter().sum::<usize>(), c.head_dim);
        // rope_half = head_dim/2 = 60 = sum of per-axis halves (20*3).
        assert_eq!(c.rope_half(), 60);
        assert_eq!(
            c.axes_dim_rope.iter().map(|&d| d / 2).sum::<usize>(),
            c.rope_half()
        );
        // GQA 4:1.
        assert_eq!(c.gqa_repeat(), 4);
        // x_embedder in = patch² · in_channels.
        assert_eq!(
            BooguConfig::X_EMBEDDER_IN,
            c.patch_size * c.patch_size * c.in_channels
        );
        // single-stream block count = 40 - 8 = 32.
        assert_eq!(c.num_single_stream_layers(), 32);
        // SwiGLU inner = 13568 (shipped value for hidden 3360).
        assert_eq!(c.swiglu_inner(), 13568);
    }

    #[test]
    fn eps_constants_are_distinct() {
        let c = BooguConfig::default();
        // Block norms use 1e-5; norm_out uses 1e-6.
        assert_eq!(c.norm_eps, 1e-5);
        assert_eq!(BooguConfig::NORM_OUT_EPS, 1e-6);
        assert_ne!(c.norm_eps, BooguConfig::NORM_OUT_EPS);
    }

    #[test]
    fn timestep_embed_constants() {
        // Sinusoid width 256, embedder width min(hidden,1024)=1024.
        assert_eq!(BooguConfig::FREQ_EMBEDDING_SIZE, 256);
        assert_eq!(BooguConfig::TIME_EMBED_DIM, 1024);
        assert_eq!(
            BooguConfig::TIME_EMBED_DIM,
            BooguConfig::default().hidden_size.min(1024)
        );
    }
}
