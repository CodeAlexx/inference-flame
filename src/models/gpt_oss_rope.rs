//! YaRN-scaled RoPE table builder for the GPT-OSS encoder (Lens M2, Stage D2).
//!
//! Bit-exact port of `transformers/modeling_rope_utils.py::_compute_yarn_parameters`
//! + `models/gpt_oss/modeling_gpt_oss.py::GptOssRotaryEmbedding.forward`.
//!
//! Pipeline (host-side only — no new CUDA kernels):
//!   1. Compute `inv_freq[head_dim/2]` with YaRN frequency stretching:
//!      - `pos_freqs[k] = theta^(2k / head_dim)`
//!      - `inv_freq_extrap[k] = 1 / pos_freqs[k]`           (unscaled, high-freq)
//!      - `inv_freq_interp[k] = 1 / (factor * pos_freqs[k])` (1/factor scaled, low-freq)
//!      - `low, high` = `find_correction_range(beta_fast, beta_slow, ...)` → ramp endpoints
//!         **in dimension-index space** (not wavelength space). Note transformers passes
//!         (beta_fast, beta_slow) to `find_correction_range(low_rot, high_rot, ...)` —
//!         despite the parameter names "low_rot/high_rot", `low` corresponds to
//!         `beta_fast` (so `low ≈ 8.09` for our config). This is a transformers quirk.
//!      - `ramp[k] = clamp((k - low) / (high - low), 0, 1)` for k in 0..head_dim/2
//!      - `extrap_factor[k] = 1 - ramp[k]`  ← weight on the EXTRAPOLATION branch
//!      - `inv_freq[k] = interp * (1 - extrap_factor) + extrap * extrap_factor`
//!         which equals `interp * ramp + extrap * (1 - ramp)`. At k=0 (ramp=0 typ),
//!         extrap wins → unscaled high-freq. At k=half-1 (ramp=1 typ), interp wins →
//!         /factor low-freq. Verified vs transformers reference.
//!   2. `attention_scaling = 0.1 * ln(factor) + 1.0` (mscale). For factor=32: ≈ 1.34657.
//!   3. For each position `p ∈ 0..seq_len`:
//!         `freq[p, k] = p * inv_freq[k]`
//!         `cos_table[p, k] = cos(freq) * attention_scaling`
//!         `sin_table[p, k] = sin(freq) * attention_scaling`
//!   4. Upload as BF16 GPU tensors. Layout matches what `rope_halfsplit_bf16`
//!      expects: `[1, 1, seq_len, head_dim/2]` (caller can reshape).
//!
//! Apply step (NOT in this module): GPT-OSS uses **halfsplit** layout
//! (`torch.chunk(x, 2, dim=-1)`), so the encoder forward will call
//! `flame_core::bf16_ops::rope_halfsplit_bf16`, not `rope_fused_bf16`. The
//! tables produced here are layout-agnostic for half_dim columns — only the
//! apply kernel differs.

use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::sync::Arc;

// The full encoder config lives in `gpt_oss_encoder.rs` (Stage D5a). We import
// it here so the rotary builder reads the canonical fields. Earlier
// chunk-1 versions kept a minimal in-module `GptOssConfig` carrying only the
// RoPE fields; that has been merged into the full encoder config to avoid two
// types with the same name in the workspace.
pub use crate::models::gpt_oss_encoder::GptOssConfig;

// ---------------------------------------------------------------------------
// GptOssRotaryEmbedding — YaRN-shifted RoPE table generator.
// ---------------------------------------------------------------------------

/// Host-side YaRN RoPE table generator.
///
/// Builds and caches `(cos, sin)` BF16 tables for arbitrary sequence lengths.
/// All non-trivial math is host F32; only the final upload + cast hits the GPU.
pub struct GptOssRotaryEmbedding {
    head_dim: usize,
    theta: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    orig_max_pos: usize,
    truncate: bool,
    /// `inv_freq[k]` for k in 0..head_dim/2, after YaRN scaling. F64 to mirror
    /// transformers which keeps intermediate math in F32 but the symbolic
    /// derivation is exact in F64.
    inv_freq: Vec<f64>,
    /// Attention scaling (`mscale`) applied to cos/sin.
    attention_scaling: f64,
    /// Cached host F32 cos/sin tables (`[max_seq_len_cached, head_dim/2]`).
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    max_seq_len_cached: usize,
}

impl GptOssRotaryEmbedding {
    /// Build the rotary embedding from a config. Only the inv_freq vector is
    /// computed eagerly; the per-position cos/sin tables are built lazily by
    /// `freqs_for`.
    pub fn new(config: &GptOssConfig) -> Result<Self> {
        if config.head_dim % 2 != 0 {
            return Err(flame_core::Error::InvalidInput(format!(
                "GptOssRotaryEmbedding: head_dim must be even, got {}",
                config.head_dim
            )));
        }
        if config.rope_factor <= 0.0 {
            return Err(flame_core::Error::InvalidInput(format!(
                "GptOssRotaryEmbedding: rope_factor must be > 0, got {}",
                config.rope_factor
            )));
        }
        let (inv_freq, attention_scaling) = compute_yarn_inv_freq(config);
        Ok(Self {
            head_dim: config.head_dim,
            theta: config.rope_theta,
            factor: config.rope_factor,
            beta_fast: config.rope_beta_fast,
            beta_slow: config.rope_beta_slow,
            orig_max_pos: config.rope_original_max_pos,
            truncate: config.rope_truncate,
            inv_freq,
            attention_scaling,
            cos_table: Vec::new(),
            sin_table: Vec::new(),
            max_seq_len_cached: 0,
        })
    }

    /// YaRN-scaled inverse frequencies (length `head_dim / 2`).
    pub fn inv_freq(&self) -> &[f64] {
        &self.inv_freq
    }

    /// Attention temperature scaling (`mscale = 0.1 * ln(factor) + 1`).
    pub fn attention_scaling(&self) -> f64 {
        self.attention_scaling
    }

    /// Build (or fetch from cache) the `(cos, sin)` BF16 tensors for a sequence
    /// of length `seq_len`. Both tensors have shape `[seq_len, head_dim/2]`,
    /// BF16 dtype, on `device`.
    ///
    /// Caller is expected to broadcast/reshape to match the apply kernel —
    /// e.g. `cos.unsqueeze(0).unsqueeze(0)` for `rope_halfsplit_bf16` which
    /// expects `[1, 1, N, half]` (or `[cos_bh, N, half]` if per-head).
    pub fn freqs_for(
        &mut self,
        seq_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        if seq_len == 0 {
            return Err(flame_core::Error::InvalidInput(
                "GptOssRotaryEmbedding::freqs_for: seq_len must be > 0".into(),
            ));
        }
        // Grow the host cache if needed. We never shrink — the cache is small
        // (`seq_len * head_dim/2 * 4` bytes; e.g. 4096*32*4 = 512 KiB).
        if seq_len > self.max_seq_len_cached {
            self.rebuild_host_tables(seq_len);
        }

        let half = self.head_dim / 2;
        let n = seq_len;
        // Slice the active region. Row stride = half (cached at exact width).
        let cos_slice: Vec<f32> = self.cos_table[..n * half].to_vec();
        let sin_slice: Vec<f32> = self.sin_table[..n * half].to_vec();

        let cos_f32 = Tensor::from_vec(cos_slice, Shape::from_dims(&[n, half]), device.clone())?;
        let sin_f32 = Tensor::from_vec(sin_slice, Shape::from_dims(&[n, half]), device.clone())?;
        let cos = cos_f32.to_dtype(DType::BF16)?;
        let sin = sin_f32.to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    /// (Re)compute the host cos/sin cache for at least `seq_len` rows.
    fn rebuild_host_tables(&mut self, seq_len: usize) {
        let half = self.head_dim / 2;
        let mut cos_tbl = vec![0.0f32; seq_len * half];
        let mut sin_tbl = vec![0.0f32; seq_len * half];
        let mscale = self.attention_scaling;
        for p in 0..seq_len {
            let pos = p as f64;
            let row = p * half;
            for k in 0..half {
                let angle = pos * self.inv_freq[k];
                let (s, c) = angle.sin_cos();
                cos_tbl[row + k] = (c * mscale) as f32;
                sin_tbl[row + k] = (s * mscale) as f32;
            }
        }
        self.cos_table = cos_tbl;
        self.sin_table = sin_tbl;
        self.max_seq_len_cached = seq_len;
    }

    /// Diagnostic: read a single `(cos, sin)` pair from the host cache without
    /// uploading to GPU. Triggers a rebuild if needed. Used by unit tests.
    pub fn host_freq_at(&mut self, pos: usize, k: usize) -> (f32, f32) {
        let half = self.head_dim / 2;
        assert!(k < half, "k out of range");
        if pos >= self.max_seq_len_cached {
            self.rebuild_host_tables(pos + 1);
        }
        let row = pos * half;
        (self.cos_table[row + k], self.sin_table[row + k])
    }

    // Field accessors for tests / debugging.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn theta(&self) -> f64 {
        self.theta
    }
    pub fn factor(&self) -> f64 {
        self.factor
    }
    pub fn beta_fast(&self) -> f64 {
        self.beta_fast
    }
    pub fn beta_slow(&self) -> f64 {
        self.beta_slow
    }
    pub fn orig_max_pos(&self) -> usize {
        self.orig_max_pos
    }
    pub fn truncate(&self) -> bool {
        self.truncate
    }
}

// ---------------------------------------------------------------------------
// YaRN inv_freq computation (pure host math, no GPU).
// ---------------------------------------------------------------------------

/// `find_correction_dim` — inverse dim-formula from the YaRN paper.
///
/// For a given number of rotations `num_rot` to fit in `max_pos` positions,
/// returns the dimension index whose wavelength gives exactly `num_rot`
/// rotations. transformers `modeling_rope_utils.py::find_correction_dim`.
fn find_correction_dim(num_rot: f64, dim: usize, base: f64, max_pos: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (dim as f64 * (max_pos / (num_rot * two_pi)).ln()) / (2.0 * base.ln())
}

/// transformers `find_correction_range`. Note the **argument order is
/// (beta_fast, beta_slow)** even though the parameter names are
/// `(low_rot, high_rot)` — the resulting `low` corresponds to beta_fast
/// (more rotations → smaller wavelength → smaller dim index).
fn find_correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: usize,
    base: f64,
    max_pos: f64,
    truncate: bool,
) -> (f64, f64) {
    let mut low = find_correction_dim(beta_fast, dim, base, max_pos);
    let mut high = find_correction_dim(beta_slow, dim, base, max_pos);
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let high = high.min((dim as f64) - 1.0);
    (low, high)
}

/// Per-dimension ramp `[0, 1]` of length `half = dim/2`. Mirrors
/// `linear_ramp_factor` in transformers, including the `+0.001` singularity
/// guard when `min == max`.
fn linear_ramp_factor(min: f64, max: f64, half: usize) -> Vec<f64> {
    let max = if min == max { max + 0.001 } else { max };
    let mut out = vec![0.0f64; half];
    let denom = max - min;
    for i in 0..half {
        let v = (i as f64 - min) / denom;
        out[i] = v.clamp(0.0, 1.0);
    }
    out
}

/// `get_mscale(scale)` — attention temperature scaling factor.
/// Returns 1.0 if `scale <= 1`. Mirrors transformers exactly.
fn get_mscale(scale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * scale.ln() + 1.0
    }
}

/// Compute YaRN-scaled inverse frequencies and the attention scaling.
///
/// Returns `(inv_freq[half], attention_scaling)`. Bit-exact port of
/// `transformers/modeling_rope_utils.py::_compute_yarn_parameters`
/// for the GPT-OSS config (no `mscale`/`mscale_all_dim`, no `attention_factor`
/// override → `attention_scaling = get_mscale(factor)`).
fn compute_yarn_inv_freq(config: &GptOssConfig) -> (Vec<f64>, f64) {
    let dim = config.head_dim; // partial_rotary_factor = 1.0 in GPT-OSS
    let half = dim / 2;
    let theta = config.rope_theta;
    let factor = config.rope_factor;
    let orig_max = config.rope_original_max_pos as f64;

    // pos_freqs[k] = theta ** (2k / dim), so inv_freq_extrap[k] = 1 / pos_freqs[k]
    let mut inv_freq_extrap = vec![0.0f64; half];
    let mut inv_freq_interp = vec![0.0f64; half];
    for k in 0..half {
        let exponent = (2 * k) as f64 / dim as f64;
        let pos_freq = theta.powf(exponent);
        inv_freq_extrap[k] = 1.0 / pos_freq;
        inv_freq_interp[k] = 1.0 / (factor * pos_freq);
    }

    // Find correction range in dim-index space.
    let (low, high) = find_correction_range(
        config.rope_beta_fast,
        config.rope_beta_slow,
        dim,
        theta,
        orig_max,
        config.rope_truncate,
    );

    // ramp[k] for k in 0..half (NOT 0..dim — see transformers line 363:
    // `linear_ramp_factor(low, high, dim // 2)`).
    let ramp = linear_ramp_factor(low, high, half);

    // extrap_factor = 1 - ramp; inv_freq = interp*(1-extrap) + extrap*extrap
    // == interp*ramp + extrap*(1-ramp).
    let mut inv_freq = vec![0.0f64; half];
    for k in 0..half {
        let extrap_factor = 1.0 - ramp[k];
        inv_freq[k] =
            inv_freq_interp[k] * (1.0 - extrap_factor) + inv_freq_extrap[k] * extrap_factor;
    }

    let attention_scaling = get_mscale(factor);
    (inv_freq, attention_scaling)
}

// ---------------------------------------------------------------------------
// Tests — host-only, no GPU required for the math.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn lens_default_matches_spec() {
        let c = GptOssConfig::default();
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.rope_theta, 150_000.0);
        assert_eq!(c.rope_factor, 32.0);
        assert_eq!(c.rope_beta_fast, 32.0);
        assert_eq!(c.rope_beta_slow, 1.0);
        assert_eq!(c.rope_original_max_pos, 4096);
        assert!(!c.rope_truncate);
    }

    #[test]
    fn inv_freq_has_half_length() {
        let rope = GptOssRotaryEmbedding::new(&GptOssConfig::default()).unwrap();
        assert_eq!(rope.inv_freq().len(), 32); // head_dim/2 = 64/2
    }

    /// `inv_freq[0]` must equal the unscaled extrapolation value `1.0` — at
    /// k=0, `pos_freqs[0] = theta^0 = 1`, so `inv_freq_extrap[0] = 1` and (by
    /// coincidence at this corner) `inv_freq_interp[0] = 1/factor`. The ramp
    /// at k=0 should land on ramp=0 → extrap wins → `inv_freq[0] = 1.0`.
    #[test]
    fn high_freq_unchanged() {
        let rope = GptOssRotaryEmbedding::new(&GptOssConfig::default()).unwrap();
        assert!(
            approx_eq(rope.inv_freq()[0], 1.0, 1e-12),
            "inv_freq[0] = {} (expected 1.0)",
            rope.inv_freq()[0]
        );
    }

    /// `inv_freq[half-1]` must equal `inv_freq_extrap[half-1] / factor` (fully
    /// interpolated, low-freq end). Anchor value from the Python reference:
    /// 3.023511396804679e-7 for the Lens config.
    #[test]
    fn low_freq_fully_scaled() {
        let cfg = GptOssConfig::default();
        let rope = GptOssRotaryEmbedding::new(&cfg).unwrap();
        let half = cfg.head_dim / 2;
        let last = rope.inv_freq()[half - 1];
        // Expected: 1 / (factor * theta^((dim-2)/dim))
        let expected =
            1.0 / (cfg.rope_factor * cfg.rope_theta.powf((cfg.head_dim - 2) as f64 / cfg.head_dim as f64));
        assert!(
            approx_eq(last, expected, 1e-15),
            "inv_freq[-1] = {} expected {}",
            last,
            expected
        );
        // Cross-check against Python anchor.
        assert!(
            approx_eq(last, 3.023511396804679e-7, 1e-12),
            "inv_freq[-1] = {} expected ≈ 3.0235e-7",
            last
        );
    }

    #[test]
    fn mscale_value() {
        let rope = GptOssRotaryEmbedding::new(&GptOssConfig::default()).unwrap();
        // 0.1 * ln(32) + 1
        let expected = 0.1 * 32.0_f64.ln() + 1.0;
        assert!(approx_eq(rope.attention_scaling(), expected, 1e-15));
        // Anchor from Python.
        assert!(
            approx_eq(rope.attention_scaling(), 1.3465735902799727, 1e-15),
            "mscale = {}",
            rope.attention_scaling()
        );
    }

    #[test]
    fn mscale_returns_one_when_factor_le_one() {
        // get_mscale(1.0) = 1.0 per transformers.
        let cfg = GptOssConfig {
            rope_factor: 1.0,
            ..GptOssConfig::default()
        };
        let rope = GptOssRotaryEmbedding::new(&cfg).unwrap();
        assert_eq!(rope.attention_scaling(), 1.0);
    }

    /// At `pos=0`, `freqs = 0`, so `cos = mscale`, `sin = 0` for every k.
    #[test]
    fn cos_sin_at_pos_zero() {
        let mut rope = GptOssRotaryEmbedding::new(&GptOssConfig::default()).unwrap();
        let mscale = rope.attention_scaling() as f32;
        for k in 0..32 {
            let (c, s) = rope.host_freq_at(0, k);
            assert!((c - mscale).abs() < 1e-6, "cos[0,{}] = {} expected {}", k, c, mscale);
            assert!(s.abs() < 1e-6, "sin[0,{}] = {} expected 0", k, s);
        }
    }

    /// Anchor values from the transformers Python reference. Tolerance is
    /// dominated by host F32 round-off and the (eventual) F32→BF16 cast in
    /// `freqs_for` — but `host_freq_at` returns pre-cast F32, so a tighter
    /// tolerance is justified.
    #[test]
    fn cos_sin_anchor_values() {
        let mut rope = GptOssRotaryEmbedding::new(&GptOssConfig::default()).unwrap();
        let mscale = rope.attention_scaling();

        // (pos=1, k=0): inv_freq[0] = 1.0, angle = 1.0, c = cos(1)*mscale, s = sin(1)*mscale
        let (c10, s10) = rope.host_freq_at(1, 0);
        let exp_c10 = (1.0_f64.cos() * mscale) as f32;
        let exp_s10 = (1.0_f64.sin() * mscale) as f32;
        assert!(
            (c10 - exp_c10).abs() < 1e-5,
            "cos[1,0] = {} expected {}",
            c10,
            exp_c10
        );
        assert!(
            (s10 - exp_s10).abs() < 1e-5,
            "sin[1,0] = {} expected {}",
            s10,
            exp_s10
        );
        // Cross-check vs Python: cos[0,1,0] = 0.7275568842887878, sin = 1.133102536201477
        assert!(
            (c10 as f64 - 0.7275568842887878).abs() < 1e-5,
            "cos[1,0] = {} vs python 0.72755688",
            c10
        );
        assert!(
            (s10 as f64 - 1.133102536201477).abs() < 1e-5,
            "sin[1,0] = {} vs python 1.13310254",
            s10
        );

        // (pos=1, k=31): inv_freq[31] ≈ 3.02e-7, angle ≈ 3.02e-7, sin ≈ angle, cos ≈ 1.
        // From python: cos ≈ 1.3465735912322998, sin ≈ 4.071380601544661e-07
        let (c1_31, s1_31) = rope.host_freq_at(1, 31);
        assert!(
            (c1_31 as f64 - 1.3465735912322998).abs() < 1e-5,
            "cos[1,31] = {} vs python",
            c1_31
        );
        assert!(
            (s1_31 as f64 - 4.071380601544661e-7).abs() < 1e-5,
            "sin[1,31] = {} vs python",
            s1_31
        );

        // (pos=1, k=16): in the ramp interior, mixing branches.
        // From python: cos ≈ 1.3465734720230103, sin ≈ 0.0006146891391836107
        let (c1_16, s1_16) = rope.host_freq_at(1, 16);
        assert!(
            (c1_16 as f64 - 1.3465734720230103).abs() < 1e-5,
            "cos[1,16] = {} vs python",
            c1_16
        );
        assert!(
            (s1_16 as f64 - 0.0006146891391836107).abs() < 1e-5,
            "sin[1,16] = {} vs python",
            s1_16
        );
    }

    /// Verify `find_correction_range` reproduces the python anchor for the
    /// Lens config: low ≈ 8.092779, high ≈ 17.398024 (truncate=false).
    #[test]
    fn correction_range_anchor() {
        let cfg = GptOssConfig::default();
        let (low, high) = find_correction_range(
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_original_max_pos as f64,
            cfg.rope_truncate,
        );
        assert!(
            approx_eq(low, 8.092779115512402, 1e-9),
            "low = {} expected 8.092779",
            low
        );
        assert!(
            approx_eq(high, 17.39802450158856, 1e-9),
            "high = {} expected 17.398024",
            high
        );
    }

    /// Singularity guard: when low == high, transformers adds 0.001 to max.
    /// Our implementation must match — otherwise NaN on divide-by-zero.
    #[test]
    fn ramp_singularity_guarded() {
        let ramp = linear_ramp_factor(5.0, 5.0, 8);
        assert_eq!(ramp.len(), 8);
        for v in &ramp {
            assert!(v.is_finite(), "ramp value NaN/inf: {}", v);
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn odd_head_dim_rejected() {
        let cfg = GptOssConfig {
            head_dim: 63,
            ..GptOssConfig::default()
        };
        assert!(GptOssRotaryEmbedding::new(&cfg).is_err());
    }

    #[test]
    fn zero_factor_rejected() {
        let cfg = GptOssConfig {
            rope_factor: 0.0,
            ..GptOssConfig::default()
        };
        assert!(GptOssRotaryEmbedding::new(&cfg).is_err());
    }

    /// Cache growth: requesting a larger seq_len should rebuild; smaller
    /// should reuse. We can't observe the rebuild directly, but we can check
    /// that `max_seq_len_cached` monotonically grows.
    #[test]
    fn host_cache_grows_monotonically() {
        let mut rope = GptOssRotaryEmbedding::new(&GptOssConfig::default()).unwrap();
        let _ = rope.host_freq_at(10, 0);
        assert!(rope.max_seq_len_cached >= 11);
        let prev = rope.max_seq_len_cached;
        let _ = rope.host_freq_at(5, 0);
        assert_eq!(rope.max_seq_len_cached, prev, "cache must not shrink");
        let _ = rope.host_freq_at(100, 0);
        assert!(rope.max_seq_len_cached >= 101);
    }
}
