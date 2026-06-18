//! Ideogram 4 — pure-Rust inference port (foundational chunk).
//!
//! Ideogram 4 is a **single-stream flow-matching DiT** (~9.3B per transformer,
//! two of them: conditional + unconditional for asymmetric CFG). Text
//! (Qwen3-VL hidden states) and image-latent tokens are concatenated into one
//! sequence and modulated per-block by sandwich-norm AdaLN from the timestep
//! embedding. Velocity prediction; Euler flow-matching.
//!
//! Reference impl: `/home/alex/ideogram4-ref/src/ideogram4/modeling_ideogram4.py`
//! @ `1f586aa`. The dataclass `Ideogram4Config` (lines 24-41) is mirrored here
//! verbatim.
//!
//! ## This chunk (build-order steps 1-2, minus GPU SDPA micro-verify)
//!
//! 1. [`Ideogram4Config`] — architecture constants.
//! 2. [`weights`] — FP8-resident weight container with **per-output-row** fp32
//!    scale (Ideogram-4 FP8 is weight-only e4m3 + per-row `.weight_scale`).
//! 3. [`mrope`] — 3D interleaved MRoPE cos/sin builder (section (24,20,20),
//!    head_dim 256). Cos/sin only; the q*cos + rotate_half(q)*sin application
//!    lives in the attention module (a later chunk).
//!
//! NOT built here: attention, transformer blocks, embeds, final layer,
//! scheduler, VAE, weight loader, bins.

pub mod attention;
pub mod block;
pub mod embed;
pub mod final_layer;
pub mod inputs;
pub mod loader;
pub mod mrope;
pub mod scheduler;
pub mod transformer;
pub mod weights;

pub use transformer::transformer_forward;
pub use weights::Ideogram4RawWeight;

/// Ideogram4 transformer architecture constants.
///
/// Mirrors the Python `@dataclass Ideogram4Config`
/// (`modeling_ideogram4.py:23-41`) field-for-field. The defaults are the
/// shipped `ideogram-ai/ideogram-4-fp8` config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ideogram4Config {
    /// Transformer hidden size. (`emb_dim = 4608`)
    pub emb_dim: usize,
    /// Number of transformer blocks. (`num_layers = 34`)
    pub num_layers: usize,
    /// Attention heads. (`num_heads = 18`) → `head_dim = emb_dim/num_heads = 256`.
    pub num_heads: usize,
    /// SwiGLU intermediate size. (`intermediate_size = 12288`)
    pub intermediate_size: usize,
    /// AdaLN conditioning width. (`adanln_dim = 512`)
    pub adanln_dim: usize,
    /// Latent dim after patchify: ae_channels(32) * patch_size²(4) = 128.
    pub in_channels: usize,
    /// Concatenated Qwen3-VL hidden dim: 4096 × 13 taps = 53248.
    pub llm_features_dim: usize,
    /// RoPE base θ. (`rope_theta = 5_000_000`)
    pub rope_theta: u32,
    /// MRoPE T/H/W section sizes. (`mrope_section = (24, 20, 20)`)
    /// NOTE: these sum to 64, which is HALF of `head_dim/2 = 128` — slots
    /// `64..127` are T-only. See [`mrope`] for the interleave semantics.
    pub mrope_section: (usize, usize, usize),
    /// RMSNorm epsilon for block norms. (`norm_eps = 1e-5`)
    pub norm_eps: f64,
}

impl Default for Ideogram4Config {
    /// The shipped `ideogram-ai/ideogram-4-fp8` config.
    fn default() -> Self {
        // QWEN3_VL_ACTIVATION_LAYERS has 13 entries; llm_features_dim = 4096*13.
        const QWEN3_VL_TAPS: usize = 13;
        Self {
            emb_dim: 4608,
            num_layers: 34,
            num_heads: 18,
            intermediate_size: 12288,
            adanln_dim: 512,
            in_channels: 128,
            llm_features_dim: 4096 * QWEN3_VL_TAPS, // 53248
            rope_theta: 5_000_000,
            mrope_section: (24, 20, 20),
            norm_eps: 1e-5,
        }
    }
}

impl Ideogram4Config {
    /// Per-head dimension. `emb_dim / num_heads = 4608 / 18 = 256`.
    ///
    /// This is the attention head_dim that drives MRoPE (head_dim/2 = 128)
    /// and the HIGH-risk d=256 SDPA path (outside cuDNN/WMMA flash {64,96,128}).
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.emb_dim / self.num_heads
    }

    /// `mrope_section` as a `[usize; 3]` for the MRoPE builder.
    #[inline]
    pub fn mrope_section_arr(&self) -> [usize; 3] {
        [
            self.mrope_section.0,
            self.mrope_section.1,
            self.mrope_section.2,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_python_dataclass() {
        let c = Ideogram4Config::default();
        assert_eq!(c.emb_dim, 4608);
        assert_eq!(c.num_layers, 34);
        assert_eq!(c.num_heads, 18);
        assert_eq!(c.intermediate_size, 12288);
        assert_eq!(c.adanln_dim, 512);
        assert_eq!(c.in_channels, 128);
        assert_eq!(c.llm_features_dim, 53248);
        assert_eq!(c.rope_theta, 5_000_000);
        assert_eq!(c.mrope_section, (24, 20, 20));
        assert_eq!(c.norm_eps, 1e-5);
    }

    #[test]
    fn head_dim_is_256() {
        let c = Ideogram4Config::default();
        // 4608 / 18 = 256; head_dim/2 = 128 (the MRoPE half-table width).
        assert_eq!(c.head_dim(), 256);
        assert_eq!(c.head_dim() % 2, 0);
    }

    #[test]
    fn mrope_section_sums_to_half_head_dim_minus_t_only_tail() {
        let c = Ideogram4Config::default();
        let half = c.head_dim() / 2; // 128
        let sec_sum: usize = c.mrope_section_arr().iter().sum(); // 64
        // Section sum (64) is strictly LESS than head_dim/2 (128). The
        // remaining 64 slots (64..127) are T-only. This is the key
        // adaptation vs hidream_o1 (where section sum == head_dim/2).
        assert_eq!(sec_sum, 64);
        assert!(sec_sum < half);
        assert_eq!(half - sec_sum, 64); // 64 T-only tail slots
    }
}
