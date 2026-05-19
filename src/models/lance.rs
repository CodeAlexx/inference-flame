//! Lance (ByteDance) — pure-Rust T2I inference port. **Chunk 1: foundations only.**
//!
//! This file currently contains ONLY the C1 surfaces required by
//! `inference-flame/ports/lance/BUILD_PLAN.md`:
//!   1. `LanceConfig` — resolved 3B hyperparameters.
//!   2. Flow-matching sampler helpers (`timestep_schedule`, `timestep_shift_transform`,
//!      `denoise_step`).
//!   3. Parameterized 3-axis mRoPE helper (`precompute_mrope_tables`, `apply_mrope`)
//!      generalized over arbitrary `mrope_sections` (vs sensenova_u1's hardcoded
//!      64/32/32 split).
//!   4. `head_rms_norm` per-head RMSNorm helper (forked from sensenova_u1 — see
//!      TODO note on extraction).
//!
//! Subsequent chunks (C2+) add Qwen2 GQA attention, SwiGLU FFN, decoder blocks,
//! MoT routing, weight loader, flow-matching loop, and the CLI bin.

use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ===========================================================================
// Module 1: LanceConfig
// ===========================================================================

/// Resolved Lance configuration. Mirrors Python `LanceConfig`
/// (`/home/alex/Lance/modeling/lance/lance.py:45-74`) merged with
/// `Lance_3B/llm_config.json` Qwen2 backbone hyperparameters.
///
/// No `Default` impl — `device` requires an explicit `Arc<CudaDevice>`.
/// Use `LanceConfig::default_3b(device)` for the canonical 3B configuration.
#[derive(Debug, Clone)]
pub struct LanceConfig {
    // -- Qwen2 backbone (from Lance_3B/llm_config.json) -------------------
    /// Hidden / model dimension. 2048 for the 3B checkpoint.
    pub hidden_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Attention head count (16).
    pub num_attention_heads: usize,
    /// GQA KV-head count (2 — 8:1 KV-share).
    pub num_kv_heads: usize,
    /// Per-head dim (hidden_size / num_attention_heads = 128).
    pub head_dim: usize,
    /// FFN inner dim (SwiGLU gate/up width).
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE theta base (1e6 for Lance, vs 1e4 LLaMA default).
    pub rope_theta: f32,
    /// mRoPE per-axis section sizes [t, h, w]. Sum=64, doubled=128=head_dim.
    pub mrope_sections: [usize; 3],
    /// Tokenizer vocabulary size.
    pub vocab_size: usize,

    // -- Lance-specific (from lance.py + inference_lance.sh) --------------
    /// `(pt, ph, pw)` latent-patch tiling. **(1, 1, 1)** per shell-config truth
    /// in `inference_lance.sh:--latent_patch_size 1 1 1` (the Python code default
    /// `(1,2,2)` is overridden at runtime). Combined with `latent_channels=48`
    /// this yields `patch_latent_dim=48`, matching the published `vae2llm.weight`
    /// shape `[2048, 48]` in `Lance_3B/model.safetensors`.
    pub latent_patch_size: (usize, usize, usize),
    /// Wan 2.2 VAE z_dim — drives `patch_latent_dim = pt*ph*pw*latent_channels`.
    ///
    /// Resolved to **48** from
    /// `/home/alex/Lance/modeling/vae/wan/vae2_2.py::Wan2_2_VAE.__init__`
    /// (line 865: `z_dim=48`). The Wan 2.1 VAE in
    /// `inference-flame/src/vae/wan21_vae.rs` documents `z_dim=16` for the
    /// 2.1 checkpoint; Wan 2.2 doubles the latent count to 48. The build plan
    /// listed this as "TBD" — confirmed during C1.
    pub latent_channels: usize,
    /// Shifted-sigmoid timestep schedule shift.
    /// Python `LanceConfig` dataclass default is 1.0 (lance.py:58 — identity
    /// transform). 4.0 is the `validation_timestep_shift` function-parameter
    /// default in `validation_gen_KVcache` (lance.py:400, :1413).
    /// `inference_lance.sh` uses 3.5. We default to 4.0 (the inference
    /// function's documented default) and expose CLI override.
    pub timestep_shift: f32,
    /// Classifier-free guidance scale for text. 4.0 = shell default.
    pub cfg_text_scale: f32,
    /// Inference timestep count. 30 in shell, 24 in Python default; we follow shell.
    pub num_inference_steps: usize,
    /// Activation / weight dtype for the model. BF16 = canonical.
    pub dtype: DType,

    // -- MoT/MoE toggles (Lance uses `Qwen2MoTDecoderLayer` by default) ---
    /// Visual generation enabled (T2I path). Mirrors `LanceConfig.visual_gen`.
    pub visual_gen: bool,
    /// Visual understanding enabled (VQA path — not used by T2I).
    pub visual_und: bool,
    /// MoT (Mixture-of-Transformers) routed decoder layers.
    /// `true` because shell default `layer_module=Qwen2MoTDecoderLayer`
    /// contains "Mo" → `use_moe = True` (lance.py:91).
    pub use_mot: bool,

    // -- Plumbing ---------------------------------------------------------
    /// CUDA device. Held as `Arc` so config can be cloned freely.
    pub device: Arc<CudaDevice>,
}

impl LanceConfig {
    /// Canonical Lance-3B configuration. All values resolved from
    /// `Lance_3B/llm_config.json` + `inference_lance.sh` defaults.
    pub fn default_3b(device: Arc<CudaDevice>) -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 36,
            num_attention_heads: 16,
            num_kv_heads: 2,
            head_dim: 128,
            intermediate_size: 11008,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            mrope_sections: [16, 24, 24],
            vocab_size: 151_936,
            latent_patch_size: (1, 1, 1),
            latent_channels: 48,
            timestep_shift: 4.0,
            cfg_text_scale: 4.0,
            num_inference_steps: 30,
            dtype: DType::BF16,
            visual_gen: true,
            visual_und: true,
            use_mot: true,
            device,
        }
    }

    /// `pt * ph * pw * latent_channels` — width of the `vae2llm` / `llm2vae`
    /// linear's patch-side dim. For 3B + Wan 2.2 image with shell-config
    /// `latent_patch_size=(1,1,1)` and `latent_channels=48`: `1 * 1 * 1 * 48 = 48`,
    /// which matches the published `vae2llm.weight` shape `[2048, 48]` in
    /// `Lance_3B/model.safetensors` (Lane C state-dict audit, 2026-05-18).
    /// The Python-code default `(1,2,2)` is overridden by the shell at
    /// `inference_lance.sh:--latent_patch_size 1 1 1`.
    pub fn patch_latent_dim(&self) -> usize {
        let (pt, ph, pw) = self.latent_patch_size;
        pt * ph * pw * self.latent_channels
    }
}

// ===========================================================================
// Module 2: Flow-matching sampler helpers
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/lance.py:599-601`, `:726`
//
//   timesteps = torch.linspace(1, 0, num_timesteps + 1, device=device)
//   timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
//   dts       = timesteps[:-1] - timesteps[1:]            # POSITIVE (t is decreasing)
//   ...
//   x_t = x_t - v_t * dts[i]    # velocity points from data to noise (lance.py:726)
//
// Lance's flow-matching loop uses `x_{t+1} = x_t - dt * v_pred` (Euler), where
// `dt = dts[i]` is positive (timesteps decrease 1 → 0 along
// `linspace(1, 0, N+1)`, so `dt = t[i] - t[i+1] > 0`). The velocity `v_pred`
// points from clean data toward noise (Python comment at `lance.py:726`), so
// the integrator subtracts. `denoise_step` takes POSITIVE `dt` and applies the
// subtraction internally — caller passes `dts[i]` verbatim, matching Python.
// Same flow-matching family as Klein, FLUX, Z-Image (shifted-sigmoid timestep
// transform); only the velocity sign convention differs.

/// Returns the **transformed** flow-matching timestep schedule of length
/// `num_steps + 1`. Output is BF16 on `device`.
///
/// Formula: `t' = shift * t / (1 + (shift - 1) * t)` applied to a
/// `linspace(1, 0, num_steps + 1)` raw schedule. With `shift == 1.0` the
/// transform is identity → plain linspace from 1 down to 0.
///
/// The schedule is monotonically decreasing (`t'[0] == 1.0`,
/// `t'[num_steps] == 0.0`).
pub fn timestep_schedule(num_steps: usize, shift: f32, device: &Arc<CudaDevice>) -> Result<Tensor> {
    if num_steps == 0 {
        return Err(Error::InvalidInput(
            "timestep_schedule: num_steps must be > 0".into(),
        ));
    }
    if !(shift > 0.0) {
        return Err(Error::InvalidInput(format!(
            "timestep_schedule: shift must be > 0; got {shift}"
        )));
    }
    let n = num_steps + 1;
    // linspace(1, 0, n): values t_i = 1 - i/(n-1) for i in 0..n
    let denom = (n - 1) as f32;
    let raw: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) / denom).collect();
    let shifted: Vec<f32> = raw
        .into_iter()
        .map(|t| shift * t / (1.0 + (shift - 1.0) * t))
        .collect();
    let t_f32 = Tensor::from_vec(shifted, Shape::from_dims(&[n]), device.clone())?;
    t_f32.to_dtype(DType::BF16)
}

/// Apply the shifted-sigmoid timestep transform `t' = shift * t / (1 + (shift - 1) * t)`
/// elementwise to an arbitrary tensor. With `shift == 1.0` returns the input
/// unchanged (in dtype/value).
///
/// Useful when a single timestep value lives in a tensor (e.g. mid-loop time
/// embedding lookup) and you need the same transform without rebuilding a
/// schedule.
pub fn timestep_shift_transform(t: &Tensor, shift: f32) -> Result<Tensor> {
    if !(shift > 0.0) {
        return Err(Error::InvalidInput(format!(
            "timestep_shift_transform: shift must be > 0; got {shift}"
        )));
    }
    // Implementation: in F32 for numerical stability (the transform is mostly
    // linear in BF16 precision, but division near t=0 benefits from F32), then
    // cast back to original dtype.
    let orig = t.dtype();
    let t_f32 = t.to_dtype(DType::F32)?;
    // numerator = shift * t
    let num = t_f32.mul_scalar(shift)?;
    // denom = 1 + (shift - 1) * t = (shift - 1) * t + 1
    let denom = t_f32.mul_scalar(shift - 1.0)?.add_scalar(1.0)?;
    let out = num.div(&denom)?;
    out.to_dtype(orig)
}

/// One Euler step of Lance's flow-matching ODE: `x_{t+1} = x_t - dt * v_pred`.
///
/// `dt` must be POSITIVE (caller passes `dts[i] = t[i] - t[i+1] > 0` from
/// `timestep_schedule`'s decreasing schedule). The subtraction matches Python
/// `lance.py:726` `x_t = x_t - v_t * dts[i]`, where the velocity points from
/// clean data toward noise. A C5 implementer reading the Python loop should
/// write `denoise_step(x_t, v_pred, dts[i])` and get the right sign without
/// any sign manipulation.
///
/// Shapes of `x_t` and `v_pred` must match. Output shape == input shape.
pub fn denoise_step(x_t: &Tensor, v_pred: &Tensor, dt: f32) -> Result<Tensor> {
    if x_t.shape().dims() != v_pred.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "denoise_step: x_t shape {:?} != v_pred shape {:?}",
            x_t.shape().dims(),
            v_pred.shape().dims()
        )));
    }
    let scaled = v_pred.mul_scalar(dt)?;
    x_t.sub(&scaled)
}

// ===========================================================================
// Module 3: Parameterized 3-axis mRoPE
// ===========================================================================
//
// Qwen2.5-VL canonical mRoPE: ONE shared `inv_freq` of length `head_dim/2`,
// computed against the FULL `head_dim` as denominator (the `default`
// ROPE_INIT_FUNCTIONS in transformers/modeling_rope_utils.py:131, called from
// Qwen2_5_VLRotaryEmbedding.__init__ via the `# HACK: 强制设置为default` at
// modeling_qwen2_5_vl.py:588). The per-axis cos/sin tables are then SLICES of
// the SAME shared freqs at widths `[sections[0], sections[1], sections[2]]`
// (summing to head_dim/2).
//
// Python's `apply_multimodal_rotary_pos_emb` (modeling_qwen2_5_vl.py:691-700)
// builds cos of shape `[3, B, L, head_dim]` (per-axis, each = cat([freqs,
// freqs], -1)), then `cat([m[i%3] for i, m in enumerate(cos.split(
// mrope_section*2, -1))])` produces a single combined cos of width `head_dim`
// whose first `head_dim/2` channels are `[freqs_t_slice, freqs_h_slice,
// freqs_w_slice]` (unique values) and whose second `head_dim/2` channels are
// the same values mirrored (from each axis's internal `cat([freqs, freqs])`).
// Final op is `q * cos + rotate_half(q) * sin` with `rotate_half` pairing
// `(q[d], q[d + head_dim/2])`.
//
// flame's `rope_halfsplit_bf16` kernel pairs `(x[d], x[d + head_dim/2])` and
// consumes a half-width cos/sin of size `head_dim/2` — this encodes Python's
// mirror implicitly. So the matching call is ONE kernel invocation on the
// full `head_dim` input with a half-width cos/sin equal to Python's first-half
// combined cos: `[freqs_t_slice_at_pos_t, freqs_h_slice_at_pos_h,
// freqs_w_slice_at_pos_w]` cat'd along the last dim. `apply_mrope` builds this
// combined half-width cos/sin at call time from the per-axis tables.
//
// Sensenova_u1 (line 1098) uses three INDEPENDENT ropes with different thetas
// per axis and is structurally a different model — do NOT mirror that pattern
// for Lance / Qwen2.5-VL mRoPE.

/// Precomputed (cos, sin) tables for 3-axis mRoPE — one pair per axis.
///
/// Each table has shape `[1, 1, max_pos, sections[i]]` BF16, ready to feed
/// `flame_core::bf16_ops::rope_halfsplit_bf16` after position slicing. The
/// trailing dim is `sections[i]` (NOT `sections[i] * 2`) because the kernel's
/// contract is "operate on `[..., D]` input and consume `cos/sin` of size
/// `D / 2`" — an axis section contributes `sections[i] * 2` channels of
/// `head_dim`, so it reads `sections[i]` (cos, sin) pairs.
#[derive(Debug, Clone)]
pub struct MropeFreqs {
    pub cos_t: Tensor,
    pub sin_t: Tensor,
    pub cos_h: Tensor,
    pub sin_h: Tensor,
    pub cos_w: Tensor,
    pub sin_w: Tensor,
}

/// Build per-axis halfsplit-RoPE tables for the three mRoPE axes.
///
/// Computes ONE shared `inv_freq[j] = theta^(-2j / head_dim)` for
/// `j in 0..head_dim/2`, then slices the resulting `[max_pos, head_dim/2]`
/// cos/sin tables into three axis pieces of widths
/// `[sections[0], sections[1], sections[2]]`. Validates that `sum(sections) ==
/// head_dim/2` (equivalently `2*sum(sections) == head_dim`).
///
/// Output dtype matches `dtype` (BF16 for the production path).
pub fn precompute_mrope_tables(
    sections: [usize; 3],
    theta: f32,
    max_pos: usize,
    head_dim: usize,
    device: &Arc<CudaDevice>,
    dtype: DType,
) -> Result<MropeFreqs> {
    if max_pos == 0 {
        return Err(Error::InvalidInput(
            "precompute_mrope_tables: max_pos must be > 0".into(),
        ));
    }
    for (i, s) in sections.iter().enumerate() {
        if *s == 0 {
            return Err(Error::InvalidInput(format!(
                "precompute_mrope_tables: section {i} must be > 0"
            )));
        }
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "precompute_mrope_tables: head_dim must be positive and even, got {head_dim}"
        )));
    }
    let half = head_dim / 2;
    let section_sum: usize = sections.iter().sum();
    if section_sum != half {
        return Err(Error::InvalidInput(format!(
            "precompute_mrope_tables: sum(sections) {section_sum} != head_dim/2 {half} \
             (equivalently 2*sum(sections) != head_dim {head_dim}) for sections {sections:?}"
        )));
    }

    // ONE shared inv_freq of length head_dim/2 with denominator head_dim.
    // inv_freq[j] = theta^(-2j / head_dim) for j in 0..half.
    let log_theta = theta.ln();
    let scale = -log_theta / (head_dim as f32);
    let inv_freq: Vec<f32> = (0..half).map(|j| ((2.0 * j as f32) * scale).exp()).collect();

    // Build [max_pos, half] angles, cos, sin on CPU in F32, then upload axis
    // slices directly. Keeps per-axis cos/sin tensors guaranteed contiguous
    // and avoids any GPU cat-not-contig risk.
    let mut cos_full: Vec<f32> = Vec::with_capacity(max_pos * half);
    let mut sin_full: Vec<f32> = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        let pf = p as f32;
        for j in 0..half {
            let a = pf * inv_freq[j];
            cos_full.push(a.cos());
            sin_full.push(a.sin());
        }
    }

    // Per-axis slice widths and offsets along the half-dim axis.
    let widths = sections;
    let offs = [0usize, widths[0], widths[0] + widths[1]];

    let extract = |full: &[f32], off: usize, w: usize| -> Result<Tensor> {
        let mut data: Vec<f32> = Vec::with_capacity(max_pos * w);
        for p in 0..max_pos {
            let row_start = p * half + off;
            data.extend_from_slice(&full[row_start..row_start + w]);
        }
        // Shape [1, 1, max_pos, w]
        let t = Tensor::from_vec(data, Shape::from_dims(&[1, 1, max_pos, w]), device.clone())?;
        t.to_dtype(dtype)
    };

    let cos_t = extract(&cos_full, offs[0], widths[0])?;
    let sin_t = extract(&sin_full, offs[0], widths[0])?;
    let cos_h = extract(&cos_full, offs[1], widths[1])?;
    let sin_h = extract(&sin_full, offs[1], widths[1])?;
    let cos_w = extract(&cos_full, offs[2], widths[2])?;
    let sin_w = extract(&sin_full, offs[2], widths[2])?;

    Ok(MropeFreqs { cos_t, sin_t, cos_h, sin_h, cos_w, sin_w })
}

/// Apply parameterized 3-axis mRoPE to a `[B, H, N, head_dim]` tensor with
/// **per-token (t, h, w) positions**.
///
/// Builds a single combined half-width cos/sin table of shape
/// `[1, 1, N, head_dim/2]` by per-token gather + axis cat, then calls
/// `flame_core::bf16_ops::rope_halfsplit_bf16` ONCE on the full `x`.
///
/// **Python parity (`apply_multimodal_rotary_pos_emb` at
/// `modeling_qwen2_5_vl.py:691-700`):** Python builds a per-axis cos tensor of
/// shape `[3, B, L, head_dim]` (each axis is `cat([freqs_axis, freqs_axis])`),
/// then `cos.split(mrope_section * 2, dim=-1)` produces 6 chunks of widths
/// `[s_t, s_h, s_w, s_t, s_h, s_w]`, and `cat([m[i % 3] for i, m in enumerate(...)])`
/// picks chunk-i from axis (i % 3). The cos/sin are indexed by per-token
/// position-ids of shape `[3, B, L]` (one row per axis), producing
/// `[B, L, head_dim]` cos/sin that are then unsqueezed to broadcast across
/// heads. Final operation is `q * cos + rotate_half(q) * sin` where
/// `rotate_half` pairs `(q[d], q[d+head_dim/2])`.
///
/// Because each per-axis cos is `cat([freqs, freqs], -1)` (the second half
/// mirrors the first), Python's combined cos satisfies `cos[d] == cos[d+head_dim/2]`
/// for every `d in 0..head_dim/2`. This is the SAME math as
/// `rope_halfsplit_bf16`, which encodes that mirror implicitly by consuming a
/// half-width cos/sin of size `head_dim/2`. The kernel's `cos[d]` corresponds
/// to Python's combined cos at index `d`, the unique `freqs * pos` value for
/// whichever axis owns that channel (per the `[s_t, s_h, s_w]` split).
///
/// **PREVIOUS BUG (fixed earlier):** the original implementation split x into
/// three sections `[s_t*2, s_h*2, s_w*2]` and ran `rope_halfsplit_bf16`
/// independently on each. That pairs `(q[d], q[d + s_axis])` within each
/// section. Python's `rotate_half` on full head_dim pairs `(q[d], q[d+64])` —
/// different pairs, different math, fails parity. Fix: apply the kernel ONCE
/// on the full head_dim with a single combined half-width cos/sin.
///
/// **PREVIOUS BUG (fixed by this signature change, 2026-05-18):** the prior
/// signature accepted `pos_indices_thw: &[i32; 3]` — a SINGLE fixed (t,h,w)
/// applied to ALL N tokens. Callers at `Lance::prefill_text_context` and
/// `Lance::gen_step` hard-coded `[0, 0, 0]`, producing identity rotation
/// (cos=1, sin=0) on every token. Net effect: mRoPE was a no-op throughout
/// the entire Module 13 forward path — Lance attention degenerated to
/// position-agnostic attention. See `SKEPTIC_FINDINGS_2026-05-18_MODULE13.md`
/// F1. The fix: per-token positions, one position triplet per token along N.
///
/// # Per-token positions contract
///
/// `pos_t`, `pos_h`, `pos_w` are each `[N]` I32 tensors carrying the
/// per-token axis position for `t`, `h`, `w` respectively. Python convention
/// (`qwen2_navit.py:1192-1297` `get_rope_index`) returns a `[3, B, L]` int
/// tensor where row 0 is t, row 1 is h, row 2 is w. We accept the three
/// rows separately as `[N]` tensors because flame-core's `index_select0`
/// only supports 2D tables + 1D indices and we want one gather per axis
/// (which fits the per-axis cos/sin layout exactly).
pub fn apply_mrope(
    x: &Tensor,
    sections: [usize; 3],
    freqs: &MropeFreqs,
    pos_t: &Tensor,
    pos_h: &Tensor,
    pos_w: &Tensor,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "apply_mrope: expected 4D [B, H, N, head_dim], got {dims:?}"
        )));
    }
    let head_dim = dims[3];
    let expected: usize = sections.iter().map(|s| s * 2).sum();
    if head_dim != expected {
        return Err(Error::InvalidInput(format!(
            "apply_mrope: head_dim {head_dim} != 2*sum(sections) {expected} for sections {sections:?}"
        )));
    }
    let (b, h, n) = (dims[0], dims[1], dims[2]);

    // Per-token positions: each must be 1D `[N]`. We validate shape here and
    // delegate the dtype-cast / OOB check to `gather_axis_pos`.
    for (name, t) in [("pos_t", pos_t), ("pos_h", pos_h), ("pos_w", pos_w)] {
        let pd = t.shape().dims();
        if pd.len() != 1 || pd[0] != n {
            return Err(Error::InvalidInput(format!(
                "apply_mrope: {name} shape {pd:?} != [{n}] (per-token positions \
                 must match N along seq dim)"
            )));
        }
    }

    // rope_halfsplit_bf16 requires BF16 + contiguous input. If the caller hands
    // us a strided view (e.g. from a prior narrow), materialize once. Contig
    // tensors are a no-op cheap clone.
    let x_c = x.contiguous()?;

    // Gather each axis's `[max_pos, sections[i]]` cos/sin table at the
    // per-token positions to produce `[N, sections[i]]` slices, then reshape
    // to `[1, 1, N, sections[i]]` for the kernel's broadcast contract.
    // `gather_axis_pos` handles the I32 cast + 2D reshape.
    let cos_t = gather_axis_pos(&freqs.cos_t, pos_t)?;
    let sin_t = gather_axis_pos(&freqs.sin_t, pos_t)?;
    let cos_h = gather_axis_pos(&freqs.cos_h, pos_h)?;
    let sin_h = gather_axis_pos(&freqs.sin_h, pos_h)?;
    let cos_w = gather_axis_pos(&freqs.cos_w, pos_w)?;
    let sin_w = gather_axis_pos(&freqs.sin_w, pos_w)?;

    // Combine axis pieces along last dim into a single `[1, 1, N, head_dim/2]`
    // half-width cos/sin. Matches Python's `cat([m[i%3] for i, m in
    // enumerate(cos.split(mrope_section*2, -1))], -1)[..., :head_dim/2]`.
    //
    // `Tensor::cat` materializes inputs to contig and writes a contig output
    // (Phase 2a safety net), so the kernel's contiguity precondition holds.
    let cos_combined = Tensor::cat(&[&cos_t, &cos_h, &cos_w], 3)?;
    let sin_combined = Tensor::cat(&[&sin_t, &sin_h, &sin_w], 3)?;

    // Single RoPE call on full head_dim — pairs (x[d], x[d + head_dim/2])
    // for d in 0..head_dim/2, applying cos_combined[d] and sin_combined[d].
    // This is Python's `rotate_half`-based math written as a single kernel.
    let r = flame_core::bf16_ops::rope_halfsplit_bf16(&x_c, &cos_combined, &sin_combined)?;

    // rope_halfsplit_bf16 returns shape `[B*H, N, head_dim]` (collapsed) —
    // reshape back to `[B, H, N, head_dim]`. See bf16_ops.rs:1115.
    r.reshape(&[b, h, n, head_dim])
}

/// Per-token gather along axis 0 of a `[1, 1, max_pos, sections[i]]` cos/sin
/// table, producing `[1, 1, N, sections[i]]`.
///
/// Reshapes the table to 2D `[max_pos, sections[i]]`, casts `positions` (1D
/// `[N]`) to I32 if necessary, calls `Tensor::index_select0`, then reshapes
/// the result back to 4D `[1, 1, N, sections[i]]` for the RoPE kernel's
/// broadcast contract.
///
/// `index_select0` is the only per-token-gather API flame-core exposes; it
/// requires a 2D table and a 1D (or any-leading-dim) I32 index tensor. We
/// only use the 1D case here because positions are per-token along N.
fn gather_axis_pos(table: &Tensor, positions: &Tensor) -> Result<Tensor> {
    let dims = table.shape().dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 1 {
        return Err(Error::InvalidInput(format!(
            "gather_axis_pos: expected table [1, 1, max_pos, half], got {dims:?}"
        )));
    }
    let max_pos = dims[2];
    let half = dims[3];

    let pos_dims = positions.shape().dims();
    if pos_dims.len() != 1 {
        return Err(Error::InvalidInput(format!(
            "gather_axis_pos: expected 1D positions [N], got {pos_dims:?}"
        )));
    }
    let n = pos_dims[0];

    // index_select0 requires a 2D table `[V, D]`. Reshape (metadata-only
    // when table is contig, which `precompute_mrope_tables` guarantees).
    let table_2d = table.reshape(&[max_pos, half])?;

    // index_select0 requires I32 indices. Cast at the boundary — same
    // F32→I32 storage-relabel idiom used by `embed_text_tokens`. Per
    // SKEPTIC F8, OOB index values are silently skipped by the gather
    // kernel; we keep that behavior unchanged (callers are responsible
    // for in-range positions, and `precompute_mrope_tables` is sized via
    // `max_pos` at construction time).
    let pos_i32 = if positions.dtype() == DType::I32 {
        positions.clone()
    } else {
        positions.to_dtype(DType::I32)?
    };

    let gathered = table_2d.index_select0(&pos_i32)?; // [N, half]
    gathered.reshape(&[1, 1, n, half])
}

// ===========================================================================
// Module 4: head_rms_norm
// ===========================================================================
//
// TODO: extract to shared module when second consumer (sensenova) adopts.
// Forked here as a local copy to avoid touching sensenova_u1.rs (currently
// broken on it2i — adding move-risk while it's already failing is not worth
// it). Identical body to sensenova_u1.rs:791-803.

/// Per-head RMSNorm on `[B, H, N, D]` with a `[D]` weight.
///
/// Applies `x / sqrt(mean(x^2, dim=-1) + eps) * gain`, where the RMSNorm
/// reduction runs across the last (head) dim. Delegates to
/// `flame_core::cuda_ops_bf16::rms_norm_bf16` after a 2D reshape.
///
/// Used for Q/K-normalization (Lance has `llm_qk_norm = true`) before mRoPE.
pub fn head_rms_norm(x: &Tensor, gain: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "head_rms_norm: expected 4D input [B, H, N, D], got {dims:?}"
        )));
    }
    let last = *dims.last().unwrap();
    let prod: usize = dims[..3].iter().product();
    let flat = x.reshape(&[prod, last])?;
    let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&flat, Some(gain), eps)?;
    out.reshape(&dims)
}

/// F32-output per-head RMSNorm with F32 weighted gain. Same math as
/// `head_rms_norm` but the normalized output and the weight multiply are
/// kept in F32 — `out = (x / sqrt(mean(x^2) + eps)) * gain` is computed
/// as: F32-normalize(BF16 x) → broadcast-mul F32(gain).
///
/// Used for the MoT Q/K-norm path to match Python's explicit F32 upcast
/// (`PackedAttentionMoT.forward_inference` lines 418-441 in
/// `/home/alex/Lance/modeling/lance/qwen2_navit.py`), where Q and K are
/// upcast to F32 BEFORE q_norm/k_norm + mRoPE and cast back to BF16 only
/// right before flash_attn. Documented qwen-style BF16 modulation
/// instability pattern (see `memory/project_qwen_grad_guard_2026-05-09.md`).
///
/// flame-core has no BF16-in/F32-out **weighted** RMS norm; we compose it
/// from `cuda_ops_bf16::rms_norm_bf16_to_f32` (unweighted) + F32 broadcast
/// `mul` against an F32-cast gain. This is strategy B per the C3 bugfix
/// plan: F32 norm output + F32 weight multiply, but mRoPE still runs in
/// BF16 (no F32 RoPE kernel exists in flame-core). The norm-output and
/// gain-multiply are the precision-sensitive operations that drive the
/// modulation-chain instability, so this captures the dominant error
/// source without requiring a new flame-core kernel.
fn head_rms_norm_f32_weighted(
    x: &Tensor,
    gain: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "head_rms_norm_f32_weighted: expected 4D input [B, H, N, D], got {dims:?}"
        )));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "head_rms_norm_f32_weighted: expected BF16 input, got {:?}",
            x.dtype()
        )));
    }
    let last = *dims.last().unwrap();
    let prod: usize = dims[..3].iter().product();
    // Unweighted BF16-in / F32-out RMSNorm: returns [prod, last] F32.
    let flat = x.reshape(&[prod, last])?;
    let norm_f32 = flame_core::cuda_ops_bf16::rms_norm_bf16_to_f32(&flat, eps)?;
    // Cast gain to F32 and broadcast-multiply. Gain is [D]; norm_f32 is
    // [prod, D] — Tensor::mul handles broadcast over the leading dim.
    let gain_f32 = gain.to_dtype(DType::F32)?;
    let weighted = norm_f32.mul(&gain_f32)?;
    weighted.reshape(&dims)
}

/// Build a `[1, 1, N, N]` causal keep-mask for SDPA.
///
/// flame-core's `sdpa(q, k, v, mask)` interprets `mask` as a binary
/// keep-mask: positions with `mask = 1` are kept, `mask = 0` positions
/// have `-inf` added to their attention logit (see
/// `flame-core/src/sdpa.rs` doc on `forward()` — "binary keep-mask
/// multiplying `(1-mask) * -inf`"). For a causal mask we want
/// `mask[i, j] = 1` iff `j <= i` (lower triangle including diagonal).
///
/// `dtype` matches what the caller will pass to SDPA (BF16 for the
/// inference path). We materialize the mask host-side as F32 then cast.
fn build_causal_mask(
    n: usize,
    device: &Arc<CudaDevice>,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            data[i * n + j] = 1.0;
        }
    }
    let t = Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 1, n, n]),
        device.clone(),
        dtype,
    )?;
    Ok(t)
}

// ===========================================================================
// Module 5: Qwen2 GQA Attention (with `llm_qk_norm`)
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:78-226`
// (`PackedAttention.forward_inference`) is the Lance-specific reference.
// The underlying class is `Qwen2Attention` at
// `/home/alex/Lance/modeling/qwen2/modeling_qwen2.py:233-415` (standard HF Qwen2).
//
// **Qwen2 bias convention (verified against
// `modeling_qwen2.py:265-268`):**
//   - `q_proj`, `k_proj`, `v_proj`: `bias=True`
//   - `o_proj`:                       `bias=False`
//
// The task spec drafted from BUILD_PLAN claimed all projections are
// `bias=False`. That's wrong for Qwen2 attention — only o_proj is
// bias-free. We follow the Python truth here. (Verify in `/home/alex/Lance`
// llm_config and weight map at C6 weight-load time.)
//
// **Forward steps (single-batch path — C2 does NOT yet wire KV cache):**
//
//   1. q = q_proj(x)                           # [B, N, num_heads * head_dim]
//      k = k_proj(x)                           # [B, N, num_kv_heads * head_dim]
//      v = v_proj(x)                           # [B, N, num_kv_heads * head_dim]
//   2. Reshape + permute to head-major:
//      q -> [B, num_heads,    N, head_dim]
//      k -> [B, num_kv_heads, N, head_dim]
//      v -> [B, num_kv_heads, N, head_dim]
//   3. Q/K RMSNorm BEFORE mRoPE.  This is the `llm_qk_norm=true` path
//      (`qwen2_navit.py:82-87` for the norm definitions,
//       `qwen2_navit.py:179-180` for the call site: q_norm/k_norm fire
//       *before* `apply_rotary_pos_emb` at line 183). V is NEVER normed.
//   4. Apply mRoPE to q AND k at pre-repeat KV head count.  Applying mRoPE
//      before the GQA expansion is mathematically the same as after (mRoPE
//      is a deterministic per-token rotation; rotating then repeating the
//      KV head 8x is identical to repeating then rotating each copy with
//      the same cos/sin), but it's the standard convention from Python
//      (`qwen2_navit.py:183` rotates K at num_kv_heads, then the
//      `forward_train` GQA path at lines 123-126 expands via repeat).
//      Cheaper too: 8x fewer K rotations.
//   5. GQA repeat-kv: expand K, V from `num_kv_heads` to `num_heads` via
//      `Tensor::stack` + `reshape` (sensenova_u1's exact pattern; see
//      sensenova_u1.rs:832-844).
//   6. SDPA: `flame_core::attention::sdpa(q, k_g, v_g, None)`.  The default
//      scale `1/sqrt(head_dim)` is applied INSIDE the SDPA dispatcher
//      (`flame_core/src/attention/sdpa.rs:412-413`).  Do NOT pre-scale q;
//      that would double-apply.  No causal mask for T2I (non-autoregressive
//      diffusion).  No mask in C2 — packed/causal masks are a C3/C4
//      concern.
//   7. Permute back to `[B, N, num_heads, head_dim]` and reshape to
//      `[B, N, hidden]`.
//   8. o_proj: `[B, N, hidden] -> [B, N, hidden]` (no bias).

/// Qwen2 grouped-query attention with optional Q/K RMSNorm (`llm_qk_norm`).
///
/// Configured for Lance 3B: 16 query heads, 2 KV heads (8:1 GQA), head_dim
/// 128, hidden 2048.  Weights are stored in PyTorch row-major `[out, in]`
/// layout (the layout produced by HF safetensors loaders); the projection
/// path goes through `flame_core::ops::fused_inference::fused_linear3d_native`
/// which consumes that layout directly via cuBLASLt `TRANSA=T`.
///
/// **C2 scope:** single-pass forward only.  KV cache integration is C3+.
/// Mask is hard-coded to `None` (non-causal, full-context).
#[derive(Debug)]
pub struct Qwen2Attention {
    /// `[hidden, hidden]`  (out=num_heads*head_dim=2048, in=hidden=2048).
    pub q_proj: Tensor,
    /// `[hidden]`  bias for q_proj (Qwen2 `bias=True`).
    pub q_bias: Tensor,
    /// `[num_kv_heads*head_dim, hidden] = [256, 2048]`.
    pub k_proj: Tensor,
    /// `[256]` bias for k_proj.
    pub k_bias: Tensor,
    /// `[256, 2048]`.
    pub v_proj: Tensor,
    /// `[256]` bias for v_proj.
    pub v_bias: Tensor,
    /// `[hidden, hidden]`. `o_proj` has `bias=False` (Qwen2 convention).
    pub o_proj: Tensor,
    /// `[head_dim=128]` RMSNorm gain for Q.
    pub q_norm: Tensor,
    /// `[head_dim=128]` RMSNorm gain for K.
    pub k_norm: Tensor,

    /// 16 (Q heads).
    pub num_heads: usize,
    /// 2 (KV heads).
    pub num_kv_heads: usize,
    /// 128.
    pub head_dim: usize,
    /// 1e-6.
    pub rms_norm_eps: f32,
    /// `[16, 24, 24]`. Sum=64=head_dim/2.
    pub mrope_sections: [usize; 3],
}

impl Qwen2Attention {
    /// Build a `Qwen2Attention` with deterministically-seeded random weights
    /// of the right shape. Used by C2 shape tests; NOT a production loader.
    /// C6 will provide the real weight-loading path.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = cfg.device.clone();
        let dt = cfg.dtype;

        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_kv_heads;
        let d = cfg.head_dim;

        // Shapes: PyTorch row-major [out, in].
        let q_proj = Self::randn_dt(&[nq * d, h], 0.02, 1001, &dev, dt)?;
        let k_proj = Self::randn_dt(&[nkv * d, h], 0.02, 1002, &dev, dt)?;
        let v_proj = Self::randn_dt(&[nkv * d, h], 0.02, 1003, &dev, dt)?;
        let o_proj = Self::randn_dt(&[h, nq * d], 0.02, 1004, &dev, dt)?;
        let q_bias = Self::zeros_dt(&[nq * d], &dev, dt)?;
        let k_bias = Self::zeros_dt(&[nkv * d], &dev, dt)?;
        let v_bias = Self::zeros_dt(&[nkv * d], &dev, dt)?;
        // RMSNorm gain initialized to 1.0 (Qwen2 default).
        let q_norm = Self::ones_dt(&[d], &dev, dt)?;
        let k_norm = Self::ones_dt(&[d], &dev, dt)?;

        Ok(Self {
            q_proj,
            q_bias,
            k_proj,
            k_bias,
            v_proj,
            v_bias,
            o_proj,
            q_norm,
            k_norm,
            num_heads: nq,
            num_kv_heads: nkv,
            head_dim: d,
            rms_norm_eps: cfg.rms_norm_eps,
            mrope_sections: cfg.mrope_sections,
        })
    }

    /// Forward pass.
    ///
    /// `x`:        `[B, N, hidden]`
    /// `pos_t`:    `[N]` per-token t-axis position (I32 or castable to I32)
    /// `pos_h`:    `[N]` per-token h-axis position
    /// `pos_w`:    `[N]` per-token w-axis position
    /// `mrope`:    precomputed cos/sin tables from `precompute_mrope_tables`
    ///
    /// **Module 13 cascade (2026-05-18):** `pos_t/h/w` replace the prior
    /// `pos_indices_thw: &[i32; 3]` single-triplet argument so callers can
    /// supply per-token positions matching Python `get_rope_index`
    /// (`qwen2_navit.py:1192-1297`). This base (non-MoT) path is dead code
    /// for Lance 3B production but its signature must cascade in lockstep
    /// with the MoT path so `LanceBlock::Base` and `Qwen2DecoderLayer` keep
    /// compiling.
    pub fn forward(
        &self,
        x: &Tensor,
        pos_t: &Tensor,
        pos_h: &Tensor,
        pos_w: &Tensor,
        mrope: &MropeFreqs,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Qwen2Attention::forward: expected 3D input [B, N, hidden], got {dims:?}"
            )));
        }
        let b = dims[0];
        let n = dims[1];
        let h = dims[2];
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let d = self.head_dim;
        if h != nq * d {
            return Err(Error::InvalidInput(format!(
                "Qwen2Attention::forward: hidden {h} != num_heads {nq} * head_dim {d}"
            )));
        }
        let n_rep = nq / nkv;
        if n_rep * nkv != nq {
            return Err(Error::InvalidInput(format!(
                "Qwen2Attention::forward: num_heads {nq} not divisible by num_kv_heads {nkv}"
            )));
        }

        // --- 1. Q/K/V projections (with bias). ---
        // fused_linear3d_native: input [B, N, Cin], weight [Cout, Cin] PyTorch
        // row-major, optional bias [Cout]. Output [B, N, Cout].
        let q = flame_core::ops::fused_inference::fused_linear3d_native(
            x,
            &self.q_proj,
            Some(&self.q_bias),
        )?; // [B, N, nq*d]
        let k = flame_core::ops::fused_inference::fused_linear3d_native(
            x,
            &self.k_proj,
            Some(&self.k_bias),
        )?; // [B, N, nkv*d]
        let v = flame_core::ops::fused_inference::fused_linear3d_native(
            x,
            &self.v_proj,
            Some(&self.v_bias),
        )?; // [B, N, nkv*d]

        // --- 2. Reshape + permute to head-major [B, H, N, D]. ---
        // fused_linear3d_native produces a fresh contig output, so reshape is
        // a cheap metadata op; permute changes strides — downstream consumers
        // (head_rms_norm reshape, mRoPE narrow+contiguous) handle stride
        // properly. NO `.contiguous()` here.
        let q = q.reshape(&[b, n, nq, d])?.permute(&[0, 2, 1, 3])?; // [B, nq, N, D]
        let k = k.reshape(&[b, n, nkv, d])?.permute(&[0, 2, 1, 3])?; // [B, nkv, N, D]
        let v = v.reshape(&[b, n, nkv, d])?.permute(&[0, 2, 1, 3])?; // [B, nkv, N, D]

        // --- 3. Q/K RMSNorm BEFORE mRoPE (`llm_qk_norm=true`). V untouched. ---
        // head_rms_norm reshapes to [prod, last] which materializes a contig
        // buffer; output is contig with shape [B, H, N, D].
        let q = head_rms_norm(&q, &self.q_norm, self.rms_norm_eps)?;
        let k = head_rms_norm(&k, &self.k_norm, self.rms_norm_eps)?;

        // --- 4. mRoPE on Q and K (pre-repeat-kv). ---
        // apply_mrope splits last dim into three sections and runs
        // rope_halfsplit_bf16 per section, then cats back. Output is BF16,
        // contig. K is rotated at num_kv_heads (cheaper than rotating at
        // num_heads after expand).
        let q = apply_mrope(&q, self.mrope_sections, mrope, pos_t, pos_h, pos_w)?;
        let k = apply_mrope(&k, self.mrope_sections, mrope, pos_t, pos_h, pos_w)?;

        // --- 5. GQA repeat-kv: [B, nkv, N, D] -> [B, nq, N, D]. ---
        // Mirrors sensenova_u1::repeat_kv (stack copies along a new dim then
        // reshape into the H axis). Materializes; SDPA needs contig head-major.
        let k_g = Self::repeat_kv(&k, n_rep)?;
        let v_g = Self::repeat_kv(&v, n_rep)?;

        // --- 6. SDPA. Scale = 1/sqrt(head_dim) applied internally. ---
        // No mask in C2: T2I diffusion is non-causal. (Packed-seq attention
        // mask is C3+ when MoT decoder + multi-image packing lands.)
        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, None)?; // [B, nq, N, D]

        // --- 7. Merge heads back to [B, N, hidden]. ---
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, nq * d])?;

        // --- 8. o_proj (no bias). ---
        // fused_linear3d_native supports bias=None.
        let out = flame_core::ops::fused_inference::fused_linear3d_native(
            &attn,
            &self.o_proj,
            None,
        )?; // [B, N, hidden]

        Ok(out)
    }

    /// Expand `[B, H_kv, N, D]` to `[B, H_kv * n_rep, N, D]` by repeating
    /// each KV head `n_rep` times (the LLaMA / Qwen2 GQA pattern). Mirrors
    /// sensenova_u1's `repeat_kv` (sensenova_u1.rs:832-844). Materializes
    /// via stack+reshape, leaving the output BF16 + contig for SDPA.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "repeat_kv: expected 4D [B, H_kv, N, D], got {dims:?}"
            )));
        }
        let b = dims[0];
        let h_kv = dims[1];
        let n = dims[2];
        let d = dims[3];
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    fn randn_dt(
        shape: &[usize],
        std: f32,
        seed: u64,
        dev: &Arc<CudaDevice>,
        dt: DType,
    ) -> Result<Tensor> {
        let t = Tensor::randn_seeded(Shape::from_dims(shape), 0.0, std, seed, dev.clone())?;
        if t.dtype() != dt {
            t.to_dtype(dt)
        } else {
            Ok(t)
        }
    }

    fn zeros_dt(shape: &[usize], dev: &Arc<CudaDevice>, dt: DType) -> Result<Tensor> {
        Tensor::zeros(Shape::from_dims(shape), dev.clone())?.to_dtype(dt)
    }

    fn ones_dt(shape: &[usize], dev: &Arc<CudaDevice>, dt: DType) -> Result<Tensor> {
        Tensor::ones(Shape::from_dims(shape), dev.clone())?.to_dtype(dt)
    }
}

// ===========================================================================
// Module 6: Qwen2 SwiGLU MLP
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/qwen2/modeling_qwen2.py:206-217`.
//
//   class Qwen2MLP(nn.Module):
//       def __init__(self, config):
//           self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
//           self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
//           self.down_proj = nn.Linear(intermediate, hidden, bias=False)
//           self.act_fn    = silu                          # hidden_act='silu'
//
//       def forward(self, x):
//           return down(silu(gate(x)) * up(x))
//
// All three projections are `bias=False`. Activation is `silu`
// (`hidden_act='silu'` in `llm_config.json` per BUILD_PLAN).

/// Qwen2 SwiGLU FFN.
///
/// All projections are bias-free (Qwen2 convention). Weights are stored in
/// PyTorch row-major `[out, in]` layout — same convention as `Qwen2Attention`.
#[derive(Debug)]
pub struct Qwen2MLP {
    /// `[intermediate, hidden] = [11008, 2048]`.
    pub gate_proj: Tensor,
    /// `[intermediate, hidden]`.
    pub up_proj: Tensor,
    /// `[hidden, intermediate] = [2048, 11008]`.
    pub down_proj: Tensor,
}

impl Qwen2MLP {
    /// Random-init constructor for C2 shape tests. C6 supplies real weights.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = Qwen2Attention::randn_dt(&[i, h], 0.02, 2001, dev, dt)?;
        let up_proj = Qwen2Attention::randn_dt(&[i, h], 0.02, 2002, dev, dt)?;
        let down_proj = Qwen2Attention::randn_dt(&[h, i], 0.02, 2003, dev, dt)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    /// `down(silu(gate(x)) * up(x))`.
    ///
    /// Shape contract: input `[B, N, hidden]` → output `[B, N, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Qwen2MLP::forward: expected 3D input [B, N, hidden], got {dims:?}"
            )));
        }
        // [B, N, hidden] @ [intermediate, hidden]^T -> [B, N, intermediate]
        let gate = flame_core::ops::fused_inference::fused_linear3d_native(
            x,
            &self.gate_proj,
            None,
        )?;
        let up = flame_core::ops::fused_inference::fused_linear3d_native(
            x,
            &self.up_proj,
            None,
        )?;
        let hidden = gate.silu()?.mul(&up)?;
        // [B, N, intermediate] @ [hidden, intermediate]^T -> [B, N, hidden]
        let out = flame_core::ops::fused_inference::fused_linear3d_native(
            &hidden,
            &self.down_proj,
            None,
        )?;
        Ok(out)
    }
}

// ===========================================================================
// Module 7: Qwen2 base decoder block (non-MoT)
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:487-572`
// (`Qwen2DecoderLayer.forward_train` and `.forward_inference`).
//
// Block body (identical in train and inference paths, modulo KV cache):
//
//   residual = x
//   x = input_layernorm(x)                # pre-attn RMSNorm
//   x = self_attn(x, ...)                 # GQA attention
//   x = residual + x
//
//   residual = x
//   x = post_attention_layernorm(x)       # pre-FFN RMSNorm
//   x = mlp(x)                            # SwiGLU
//   x = residual + x
//
// Both RMSNorms have learnable gain of shape `[hidden]` (Qwen2RMSNorm).

/// Base Qwen2 decoder block (non-MoT variant).
///
/// **Scope:** C2 builds the BASE block. The MoT variant (parallel und/gen
/// paths with routing mask) is module 8, scheduled for C3.
#[derive(Debug)]
pub struct Qwen2DecoderLayer {
    /// `[hidden]` RMSNorm gain — pre-attention.
    pub input_layernorm: Tensor,
    /// `[hidden]` RMSNorm gain — pre-FFN.
    pub post_attention_layernorm: Tensor,
    /// GQA attention with Q/K-norm.
    pub self_attn: Qwen2Attention,
    /// SwiGLU FFN.
    pub mlp: Qwen2MLP,

    /// 1e-6 (carried for the two residual-stream RMSNorms).
    pub rms_norm_eps: f32,
}

impl Qwen2DecoderLayer {
    /// Random-init constructor for C2 shape tests.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        // RMSNorm gains init to 1.0 (Qwen2 default).
        let input_layernorm = Qwen2Attention::ones_dt(&[h], dev, dt)?;
        let post_attention_layernorm = Qwen2Attention::ones_dt(&[h], dev, dt)?;
        let self_attn = Qwen2Attention::new_random(cfg)?;
        let mlp = Qwen2MLP::new_random(cfg)?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            self_attn,
            mlp,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    /// Forward pass.
    ///
    /// `x`:        `[B, N, hidden]`
    /// `pos_t/h/w`: `[N]` per-token (t, h, w) mRoPE positions
    /// `mrope`:    precomputed mRoPE tables
    pub fn forward(
        &self,
        x: &Tensor,
        pos_t: &Tensor,
        pos_h: &Tensor,
        pos_w: &Tensor,
        mrope: &MropeFreqs,
    ) -> Result<Tensor> {
        // --- Attention block. ---
        let residual = x.clone();
        let h = Self::rms_norm_apply(x, &self.input_layernorm, self.rms_norm_eps)?;
        let h = self.self_attn.forward(&h, pos_t, pos_h, pos_w, mrope)?;
        let h = residual.add(&h)?;

        // --- FFN block. ---
        let residual = h.clone();
        let h = Self::rms_norm_apply(&h, &self.post_attention_layernorm, self.rms_norm_eps)?;
        let h = self.mlp.forward(&h)?;
        let h = residual.add(&h)?;
        Ok(h)
    }

    /// Residual-stream RMSNorm: input `[B, N, hidden]` → output same shape.
    ///
    /// Mirrors `sensenova_u1::rms_norm_apply` exactly: reshape to
    /// `[batch, hidden]`, call `flame_core::cuda_ops_bf16::rms_norm_bf16`,
    /// reshape back. We deliberately do NOT call `flame_core::norm::rms_norm`
    /// here because that top-level wrapper has an NHWC assertion for rank-4
    /// inputs and the 2D path is what every other Qwen2 port in this codebase
    /// (sensenova_u1) already uses successfully.
    fn rms_norm_apply(x: &Tensor, gain: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen2DecoderLayer::rms_norm_apply: input has no dims".into(),
            ));
        }
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(gain), eps)?;
        out.reshape(&dims)
    }
}

// ===========================================================================
// Module 8: Qwen2 MoT (Mixture-of-Transformers) attention
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:229-484`
// (`PackedAttentionMoT.__init__` and `.forward_inference`).
//
// **MoT routing pattern (state_dict-verified, see
// `inference-flame/ports/lance/STATE_DICT_KEYS.md`):** every projection
// and Q/K-norm gain on the attention has a PAIRED gen variant suffixed
// `_moe_gen`. Tokens carrying `gen_mask = True` flow through the gen path
// (`*_moe_gen` weights); tokens with `gen_mask = False` flow through the
// und path (the base weights). The SDPA itself is **shared** — only the
// projections and Q/K-norms diverge.
//
// **Python uses scatter/gather over packed token indexes** (lines 282-292,
// 405-412, 477-478): write into a pre-zeroed buffer at index sets
// `packed_und_token_indexes` and `packed_text_indexes`/`packed_vae_token_indexes`.
// In our dense `[B, N, hidden]` inference contract with a `[B, N] bool`
// mask, the mathematically equivalent op is `where_mask(gen_mask, gen_val,
// und_val)`: select per-token between the two parallel-computed paths.
//
// **Why Strategy A (parallel-compute + mask) and not Strategy B
// (scatter/gather):**
//   - Strategy B requires `index_select` with dynamically-computed index
//     tensors from `gen_mask.argwhere()`. `flame_core::Tensor::index_select`
//     exists but `argwhere` does not (and producing the index tensor on the
//     host would defeat fused GPU flow).
//   - Strategy A doubles the projection FLOPs per layer (each token does
//     both paths, then is masked). For 36 layers at 2048 hidden with
//     batch=1, N=4096 tokens, this is ~1 GFLOP extra per layer — measurable
//     but tolerable for C3's shape-and-structure milestone.
//   - The result is bit-identical to Strategy B for any input where
//     `where_mask` is exact (it is, since `mask*a + (1-mask)*b = a` for
//     `mask=1` exactly and `= b` for `mask=0` exactly).
//   - C5 (KV cache + packed-seq inference) can revisit if profiling shows
//     this is the bottleneck.
//
// `Tensor::where_mask` (`flame_core::ops_ext::where_mask`) handles
// broadcasting and dtype-casting internally; we only need to reshape the
// mask to `[B, N, 1]` so the trailing hidden-dim broadcasts. For the
// `[B, H, N, D]` head-major Q/K/V, we reshape to `[B, 1, N, 1]` to
// broadcast against head and dim axes.

/// MoT-routed Qwen2 GQA attention. Holds two parallel projection/norm sets
/// (und + gen) keyed by a per-token `gen_mask`. SDPA itself is shared.
///
/// Layout matches `Qwen2Attention` for the base path; every weight on the
/// base path has a `_moe_gen` sibling here.
#[derive(Debug)]
pub struct Qwen2MoTAttention {
    // ---- und path (base weights — same layout as Qwen2Attention) -----
    pub q_proj: Tensor,
    pub q_bias: Tensor,
    pub k_proj: Tensor,
    pub k_bias: Tensor,
    pub v_proj: Tensor,
    pub v_bias: Tensor,
    pub o_proj: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,

    // ---- gen path (`_moe_gen` siblings) -----------------------------
    pub q_proj_moe_gen: Tensor,
    pub q_bias_moe_gen: Tensor,
    pub k_proj_moe_gen: Tensor,
    pub k_bias_moe_gen: Tensor,
    pub v_proj_moe_gen: Tensor,
    pub v_bias_moe_gen: Tensor,
    pub o_proj_moe_gen: Tensor,
    pub q_norm_moe_gen: Tensor,
    pub k_norm_moe_gen: Tensor,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub mrope_sections: [usize; 3],
}

impl Qwen2MoTAttention {
    /// Random-init for C3 shape tests. Both paths get DIFFERENT seeds so
    /// the test that flips `gen_mask` from all-und to all-gen can detect
    /// "did the gen weights actually get used" (output should differ).
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = cfg.device.clone();
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_kv_heads;
        let d = cfg.head_dim;

        // und path — reuse Qwen2Attention seed range (1001..1004 + norms).
        let q_proj = Qwen2Attention::randn_dt(&[nq * d, h], 0.02, 8001, &dev, dt)?;
        let k_proj = Qwen2Attention::randn_dt(&[nkv * d, h], 0.02, 8002, &dev, dt)?;
        let v_proj = Qwen2Attention::randn_dt(&[nkv * d, h], 0.02, 8003, &dev, dt)?;
        let o_proj = Qwen2Attention::randn_dt(&[h, nq * d], 0.02, 8004, &dev, dt)?;
        let q_bias = Qwen2Attention::zeros_dt(&[nq * d], &dev, dt)?;
        let k_bias = Qwen2Attention::zeros_dt(&[nkv * d], &dev, dt)?;
        let v_bias = Qwen2Attention::zeros_dt(&[nkv * d], &dev, dt)?;
        let q_norm = Qwen2Attention::ones_dt(&[d], &dev, dt)?;
        let k_norm = Qwen2Attention::ones_dt(&[d], &dev, dt)?;

        // gen path — DIFFERENT seeds so the routing test can tell them apart.
        let q_proj_moe_gen = Qwen2Attention::randn_dt(&[nq * d, h], 0.02, 8101, &dev, dt)?;
        let k_proj_moe_gen = Qwen2Attention::randn_dt(&[nkv * d, h], 0.02, 8102, &dev, dt)?;
        let v_proj_moe_gen = Qwen2Attention::randn_dt(&[nkv * d, h], 0.02, 8103, &dev, dt)?;
        let o_proj_moe_gen = Qwen2Attention::randn_dt(&[h, nq * d], 0.02, 8104, &dev, dt)?;
        let q_bias_moe_gen = Qwen2Attention::zeros_dt(&[nq * d], &dev, dt)?;
        let k_bias_moe_gen = Qwen2Attention::zeros_dt(&[nkv * d], &dev, dt)?;
        let v_bias_moe_gen = Qwen2Attention::zeros_dt(&[nkv * d], &dev, dt)?;
        let q_norm_moe_gen = Qwen2Attention::ones_dt(&[d], &dev, dt)?;
        let k_norm_moe_gen = Qwen2Attention::ones_dt(&[d], &dev, dt)?;

        Ok(Self {
            q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, o_proj, q_norm, k_norm,
            q_proj_moe_gen, q_bias_moe_gen, k_proj_moe_gen, k_bias_moe_gen,
            v_proj_moe_gen, v_bias_moe_gen, o_proj_moe_gen,
            q_norm_moe_gen, k_norm_moe_gen,
            num_heads: nq,
            num_kv_heads: nkv,
            head_dim: d,
            rms_norm_eps: cfg.rms_norm_eps,
            mrope_sections: cfg.mrope_sections,
        })
    }

    /// Forward pass.
    ///
    /// `x`:                `[B, N, hidden]` BF16.
    /// `gen_mask`:         `[B, N]` selecting per-token between und (false)
    ///                     and gen (true) paths. Any dtype accepted by
    ///                     `where_mask` (Bool, BF16, F32 — internally cast
    ///                     to `x.dtype()`).
    /// `pos_t/h/w`:        `[N]` per-token mRoPE position rows matching
    ///                     Python's `get_rope_index` output
    ///                     (`qwen2_navit.py:1192-1297` returns `[3, B, L]`,
    ///                     which we pass as three `[N]` tensors for B=1).
    /// `mrope`:            precomputed cos/sin tables.
    /// `is_causal`:        when `true`, a causal `[1, 1, N, N]` keep-mask
    ///                     is built and passed to SDPA. Required for the
    ///                     text-context prefill path (Python
    ///                     `lance.py:1672, 1909`,
    ///                     `qwen2_navit.py:384, 471` `causal=is_causal` to
    ///                     `flash_attn_varlen_func`). VAE-denoise
    ///                     (`lance.py:1725, 1743, 1759`) passes `false`.
    ///
    /// **Routing semantics:** per-token weighted blend
    /// `out = where(gen_mask, gen_path, und_path)` after both paths are
    /// computed. SDPA itself sees the combined (routed) Q/K/V — it is NOT
    /// run twice; there is only one attention computation.
    ///
    /// **F1 fix (2026-05-18):** the Q/K-norm path runs in F32 to match
    /// Python's `packed_query_states.to(torch.float32)` upcast
    /// (`qwen2_navit.py:418-424`). See `head_rms_norm_f32_weighted`. mRoPE
    /// remains BF16 (no F32 RoPE kernel in flame-core; documented).
    ///
    /// **F3 fix (2026-05-18):** `layer_idx` + `cache` arguments. When
    /// `cache.is_some()`, K/V at this layer are concatenated with the
    /// per-layer cache slot ALONG THE SEQ DIM (dim=2 in `[B, H_kv, N, D]`)
    /// BEFORE GQA repeat-kv and BEFORE SDPA. After SDPA, the merged K/V
    /// (post-mRoPE, pre-repeat-kv) are written back into the cache. This
    /// mirrors Python `qwen2_navit.py:443-458, 480-482`. Without this the
    /// gen-step attention only sees current latents — text context is
    /// invisible and CFG output is garbage.
    pub fn forward(
        &self,
        x: &Tensor,
        gen_mask: &Tensor,
        pos_t: &Tensor,
        pos_h: &Tensor,
        pos_w: &Tensor,
        mrope: &MropeFreqs,
        is_causal: bool,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
        update_cache: bool,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Qwen2MoTAttention::forward: expected 3D input [B, N, hidden], got {dims:?}"
            )));
        }
        let b = dims[0];
        let n = dims[1];
        let h = dims[2];
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let d = self.head_dim;
        if h != nq * d {
            return Err(Error::InvalidInput(format!(
                "Qwen2MoTAttention::forward: hidden {h} != num_heads {nq} * head_dim {d}"
            )));
        }
        let gen_dims = gen_mask.shape().dims();
        if gen_dims != [b, n] {
            return Err(Error::InvalidInput(format!(
                "Qwen2MoTAttention::forward: gen_mask shape {gen_dims:?} != [{b}, {n}]"
            )));
        }
        let n_rep = nq / nkv;
        if n_rep * nkv != nq {
            return Err(Error::InvalidInput(format!(
                "Qwen2MoTAttention::forward: num_heads {nq} not divisible by num_kv_heads {nkv}"
            )));
        }

        // --- 1. Q/K/V projections — BOTH paths in parallel, then route. ---
        // Each `where_mask` call broadcasts `gen_mask` from [B, N] to
        // [B, N, F] where F is the projection out-dim. We reshape the mask
        // to [B, N, 1] first so the trailing 1 broadcasts cleanly.
        let mask_bn1 = gen_mask.reshape(&[b, n, 1])?;

        let q_und = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.q_proj, Some(&self.q_bias),
        )?; // [B, N, nq*d]
        let q_gen = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.q_proj_moe_gen, Some(&self.q_bias_moe_gen),
        )?;
        let q = Tensor::where_mask(&mask_bn1, &q_gen, &q_und)?; // [B, N, nq*d]

        let k_und = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.k_proj, Some(&self.k_bias),
        )?;
        let k_gen = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.k_proj_moe_gen, Some(&self.k_bias_moe_gen),
        )?;
        let k = Tensor::where_mask(&mask_bn1, &k_gen, &k_und)?; // [B, N, nkv*d]

        let v_und = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.v_proj, Some(&self.v_bias),
        )?;
        let v_gen = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.v_proj_moe_gen, Some(&self.v_bias_moe_gen),
        )?;
        let v = Tensor::where_mask(&mask_bn1, &v_gen, &v_und)?; // [B, N, nkv*d]

        // --- 2. Reshape + permute to head-major [B, H, N, D]. ---
        let q = q.reshape(&[b, n, nq, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, nkv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, nkv, d])?.permute(&[0, 2, 1, 3])?;

        // --- 3. Q/K RMSNorm BEFORE mRoPE, per-path then route. ---
        // F1 fix (2026-05-18): F32 upcast for Q/K norm + per-path weighted
        // gain, matching Python `qwen2_navit.py:418-424`. The where_mask
        // routing runs in F32 (same dtype on both sides) and the F32→BF16
        // cast happens RIGHT BEFORE mRoPE (Python casts back at lines
        // 439-441 right before flash_attn; mRoPE in flame-core is BF16-
        // only so we cast one op earlier).
        //
        // The mask broadcasts from [B, N] to [B, H, N, D] by reshaping to
        // [B, 1, N, 1]; where_mask's internal mul/sub kernels accept F32.
        let mask_b1n1 = gen_mask.reshape(&[b, 1, n, 1])?;
        let q_und_n = head_rms_norm_f32_weighted(&q, &self.q_norm, self.rms_norm_eps)?;
        let q_gen_n = head_rms_norm_f32_weighted(&q, &self.q_norm_moe_gen, self.rms_norm_eps)?;
        let q_f32 = Tensor::where_mask(&mask_b1n1, &q_gen_n, &q_und_n)?;
        let q = q_f32.to_dtype(DType::BF16)?;

        let k_und_n = head_rms_norm_f32_weighted(&k, &self.k_norm, self.rms_norm_eps)?;
        let k_gen_n = head_rms_norm_f32_weighted(&k, &self.k_norm_moe_gen, self.rms_norm_eps)?;
        let k_f32 = Tensor::where_mask(&mask_b1n1, &k_gen_n, &k_und_n)?;
        let k = k_f32.to_dtype(DType::BF16)?;

        // --- 4. mRoPE on Q and K (pre-repeat-kv). Shared (no per-path split). ---
        // BF16 mRoPE — flame-core has no F32 RoPE kernel. This is a
        // documented residual precision gap vs Python (which keeps Q/K
        // in F32 through `apply_multimodal_rotary_pos_emb` until the
        // pre-SDPA cast at qwen2_navit.py:439-441). Adding an F32 RoPE
        // kernel is out of scope for this bugfix (no new flame-core
        // kernels per task constraint).
        let q = apply_mrope(&q, self.mrope_sections, mrope, pos_t, pos_h, pos_w)?;
        let k = apply_mrope(&k, self.mrope_sections, mrope, pos_t, pos_h, pos_w)?;

        // --- 4b. KV cache concat (F3 fix 2026-05-18). When a cache is
        //     provided AND the per-layer slot is populated, prepend the
        //     cached K/V to the current K/V along the seq dim (dim=2 in
        //     `[B, H_kv, N, D]`). This mirrors Python
        //     `qwen2_navit.py:443-458`: past_key_value is concatenated
        //     onto current key_states, producing `merged_key_states`
        //     which is then handed to flash_attn. Past comes FIRST so
        //     the cached prefix sits at the low indices and the current
        //     latents sit at the high indices — matching Python's
        //     scatter pattern (`packed_key_value_indexes` covers the
        //     past range, `packed_query_indexes` covers the current
        //     range, and the latter is always after the former in the
        //     standard T2I prefix-then-gen layout).
        //
        //     CRITICAL: cache stores K/V AT `num_kv_heads`, NOT
        //     `num_heads`, and POST-mRoPE (no re-rotation on read). The
        //     repeat_kv expansion happens AFTER the cache merge so the
        //     cached past_k already has the right shape to concat with
        //     current k. Q is NEVER cached.
        let (k, v) = if let Some(cache_ref) = cache.as_ref() {
            let merged_k = match &cache_ref.k[layer_idx] {
                Some(past_k) => Tensor::cat(&[past_k, &k], 2)?,
                None => k,
            };
            let merged_v = match &cache_ref.v[layer_idx] {
                Some(past_v) => Tensor::cat(&[past_v, &v], 2)?,
                None => v,
            };
            (merged_k, merged_v)
        } else {
            (k, v)
        };

        // Snapshot pre-repeat-kv K/V for cache write-back below. Cheap:
        // these are Arc-backed handles, no GPU memcpy until something
        // takes ownership.
        let k_for_cache = k.clone();
        let v_for_cache = v.clone();

        // --- 5. GQA repeat-kv: [B, nkv, N_kv, D] -> [B, nq, N_kv, D].
        //     After F3 cache merge, N_kv may be larger than the query
        //     seq_len N (text prefix + current latents); that's exactly
        //     what we want — Q is the current-step queries, K/V span
        //     past + current.
        let k_g = Qwen2Attention::repeat_kv(&k, n_rep)?;
        let v_g = Qwen2Attention::repeat_kv(&v, n_rep)?;

        // --- 6. SHARED SDPA. F2 fix (2026-05-18): pass causal mask when
        //        is_causal=true. Python `qwen2_navit.py:384, 471` threads
        //        `causal=is_causal` to flash_attn_varlen_func; the
        //        text-context prefill path (`lance.py:1672`) uses causal
        //        attention to build the gen KV-cache. flame-core SDPA has
        //        no causal flag, so we materialize a [1,1,N,N] BF16 keep-
        //        mask (lower triangle including diagonal). ---
        let causal_mask = if is_causal {
            Some(build_causal_mask(n, x.device(), DType::BF16)?)
        } else {
            None
        };
        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, causal_mask.as_ref())?; // [B, nq, N, D]

        // --- 7. Merge heads back to [B, N, hidden]. ---
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, nq * d])?;

        // --- 8. o_proj — paired (no bias on either path). ---
        let out_und = flame_core::ops::fused_inference::fused_linear3d_native(
            &attn, &self.o_proj, None,
        )?;
        let out_gen = flame_core::ops::fused_inference::fused_linear3d_native(
            &attn, &self.o_proj_moe_gen, None,
        )?;
        let out = Tensor::where_mask(&mask_bn1, &out_gen, &out_und)?;

        // --- 9. KV cache write-back (F3 fix). Stores the merged
        //     post-mRoPE pre-repeat-kv K/V. Mirrors Python
        //     `qwen2_navit.py:480-482` where `past_key_values.key_cache
        //     [self.layer_idx] = merged_key_states`. The write is gated
        //     on `update_cache` to mirror Python's
        //     `update_past_key_values` parameter
        //     (`qwen2_navit.py:171, 222-224, 383, 480-482`):
        //       - PREFILL (`lance.py:1908`, `update_gen_context`) passes
        //         `update_past_key_values=True` so the prefix K/V is
        //         persisted into the cache.
        //       - GEN-STEP (`lance.py:1724, 1742, 1758`) passes
        //         `update_past_key_values=False` — the cache is READ
        //         (the prefix K/V is concatenated for SDPA) but NOT
        //         written, so per-step latent K/V do not pollute the
        //         prefix cache across denoise steps.
        //     The merged K stored is the MERGED post-cat tensor (Python
        //     `qwen2_navit.py:223-224` writes `merged_key_states`, the
        //     post-concat state, not just the increment), which is what
        //     `k_for_cache` already holds.
        if let Some(cache_mut) = cache {
            if update_cache {
                cache_mut.k[layer_idx] = Some(k_for_cache);
                cache_mut.v[layer_idx] = Some(v_for_cache);
            }
        }

        Ok(out)
    }
}

// ===========================================================================
// Module 8b: Qwen2 MoT MLP (paired SwiGLU)
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:588-589` —
// `self.mlp = Qwen2MLP(config); self.mlp_moe_gen = Qwen2MLP(config)`. Both
// MLPs are independent SwiGLU triplets; routing is per-token via
// `gen_mask`. As with attention, we parallel-compute both paths and route
// via `where_mask` on the output.
//
// We do NOT split per-projection inside the MLP (gate/up/down) because the
// SwiGLU non-linearity `silu(gate(x)) * up(x)` is not linear, so masking
// AFTER the SwiGLU is the only correct path. Each path is a full
// independent SwiGLU.

/// Paired SwiGLU MLP routed by per-token gen_mask.
#[derive(Debug)]
pub struct Qwen2MoTMLP {
    // und path
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    // gen path
    pub gate_proj_moe_gen: Tensor,
    pub up_proj_moe_gen: Tensor,
    pub down_proj_moe_gen: Tensor,
}

impl Qwen2MoTMLP {
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: Qwen2Attention::randn_dt(&[i, h], 0.02, 9001, dev, dt)?,
            up_proj:   Qwen2Attention::randn_dt(&[i, h], 0.02, 9002, dev, dt)?,
            down_proj: Qwen2Attention::randn_dt(&[h, i], 0.02, 9003, dev, dt)?,
            gate_proj_moe_gen: Qwen2Attention::randn_dt(&[i, h], 0.02, 9101, dev, dt)?,
            up_proj_moe_gen:   Qwen2Attention::randn_dt(&[i, h], 0.02, 9102, dev, dt)?,
            down_proj_moe_gen: Qwen2Attention::randn_dt(&[h, i], 0.02, 9103, dev, dt)?,
        })
    }

    /// Forward. Both paths are computed end-to-end (gate, up, silu*up,
    /// down) independently, then `where_mask` selects per-token.
    pub fn forward(&self, x: &Tensor, gen_mask: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Qwen2MoTMLP::forward: expected 3D input [B, N, hidden], got {dims:?}"
            )));
        }
        let b = dims[0];
        let n = dims[1];

        let gen_dims = gen_mask.shape().dims();
        if gen_dims != [b, n] {
            return Err(Error::InvalidInput(format!(
                "Qwen2MoTMLP::forward: gen_mask shape {gen_dims:?} != [{b}, {n}]"
            )));
        }

        // und path
        let gate_u = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.gate_proj, None,
        )?;
        let up_u = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.up_proj, None,
        )?;
        let hidden_u = gate_u.silu()?.mul(&up_u)?;
        let out_u = flame_core::ops::fused_inference::fused_linear3d_native(
            &hidden_u, &self.down_proj, None,
        )?;

        // gen path
        let gate_g = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.gate_proj_moe_gen, None,
        )?;
        let up_g = flame_core::ops::fused_inference::fused_linear3d_native(
            x, &self.up_proj_moe_gen, None,
        )?;
        let hidden_g = gate_g.silu()?.mul(&up_g)?;
        let out_g = flame_core::ops::fused_inference::fused_linear3d_native(
            &hidden_g, &self.down_proj_moe_gen, None,
        )?;

        // Route. Mask broadcast from [B, N] -> [B, N, hidden] via reshape.
        let mask_bn1 = gen_mask.reshape(&[b, n, 1])?;
        Tensor::where_mask(&mask_bn1, &out_g, &out_u)
    }
}

// ===========================================================================
// Module 8c: Qwen2 MoT decoder layer
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:575-710`.
//
// Structure matches `Qwen2DecoderLayer` (Module 7) with paired norms:
//
//   residual = x
//   x = where(gen_mask, input_layernorm_moe_gen(x), input_layernorm(x))
//   x = self_attn(x, gen_mask, ...)
//   x = residual + x
//
//   residual = x
//   x = where(gen_mask, post_attn_norm_moe_gen(x), post_attn_norm(x))
//   x = mlp(x, gen_mask)
//   x = residual + x
//
// Both RMSNorm gains are `[hidden]`. Routing only matters at the norm and
// MLP/attention layer boundaries — residuals add the (un-routed) full
// stream.

/// Paired Qwen2 decoder block for the MoT path.
#[derive(Debug)]
pub struct Qwen2MoTDecoderLayer {
    pub input_layernorm: Tensor,
    pub input_layernorm_moe_gen: Tensor,
    pub post_attention_layernorm: Tensor,
    pub post_attention_layernorm_moe_gen: Tensor,
    pub self_attn: Qwen2MoTAttention,
    pub mlp: Qwen2MoTMLP,
    pub rms_norm_eps: f32,
}

impl Qwen2MoTDecoderLayer {
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        Ok(Self {
            input_layernorm: Qwen2Attention::ones_dt(&[h], dev, dt)?,
            input_layernorm_moe_gen: Qwen2Attention::ones_dt(&[h], dev, dt)?,
            post_attention_layernorm: Qwen2Attention::ones_dt(&[h], dev, dt)?,
            post_attention_layernorm_moe_gen: Qwen2Attention::ones_dt(&[h], dev, dt)?,
            self_attn: Qwen2MoTAttention::new_random(cfg)?,
            mlp: Qwen2MoTMLP::new_random(cfg)?,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        gen_mask: &Tensor,
        pos_t: &Tensor,
        pos_h: &Tensor,
        pos_w: &Tensor,
        mrope: &MropeFreqs,
        is_causal: bool,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
        update_cache: bool,
    ) -> Result<Tensor> {
        // --- Attention block. ---
        let residual = x.clone();
        let h_und = Self::rms_norm_apply(x, &self.input_layernorm, self.rms_norm_eps)?;
        let h_gen = Self::rms_norm_apply(x, &self.input_layernorm_moe_gen, self.rms_norm_eps)?;
        let mask_bn1 = {
            let bn = x.shape().dims();
            gen_mask.reshape(&[bn[0], bn[1], 1])?
        };
        let h = Tensor::where_mask(&mask_bn1, &h_gen, &h_und)?;
        let h = self.self_attn.forward(&h, gen_mask, pos_t, pos_h, pos_w, mrope, is_causal, layer_idx, cache, update_cache)?;
        let h = residual.add(&h)?;

        // --- FFN block. ---
        let residual = h.clone();
        let g_und = Self::rms_norm_apply(&h, &self.post_attention_layernorm, self.rms_norm_eps)?;
        let g_gen = Self::rms_norm_apply(&h, &self.post_attention_layernorm_moe_gen, self.rms_norm_eps)?;
        let mask_bn1 = {
            let bn = h.shape().dims();
            gen_mask.reshape(&[bn[0], bn[1], 1])?
        };
        let g = Tensor::where_mask(&mask_bn1, &g_gen, &g_und)?;
        let g = self.mlp.forward(&g, gen_mask)?;
        let out = residual.add(&g)?;
        Ok(out)
    }

    /// Same as `Qwen2DecoderLayer::rms_norm_apply` (Module 7) — kept
    /// private here to avoid coupling C2/C3 module visibility.
    fn rms_norm_apply(x: &Tensor, gain: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen2MoTDecoderLayer::rms_norm_apply: input has no dims".into(),
            ));
        }
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(gain), eps)?;
        out.reshape(&dims)
    }
}

// ===========================================================================
// Module 8d: Per-layer KV cache for the MoT attention path
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:49-66` — the
// Python `NaiveCache` is a per-layer dict of (K, V) entries, optional until
// the first prefill writes them. Lance T2I's gen-step path (`lance.py:1716-
// 1727`) ALWAYS passes a `past_key_values` to `language_model.forward_inference`;
// the cache is built up by `update_gen_context` (`lance.py:1886-1915`) during
// text-context prefill, then read+optionally-updated on every VAE-denoise step.
//
// Without this concat, the gen-step attention sees ONLY the current latent
// tokens — text context is invisible. CFG output is garbage. See
// `SKEPTIC_FINDINGS_2026-05-18_C3.md` F3 (HIGH severity).
//
// Layout choice (matches Python at the Rust call site of `Qwen2MoTAttention`):
//   - K stored at `[B, num_kv_heads, N_cached, head_dim]`, BF16. POST-mRoPE,
//     PRE-GQA-repeat-kv. This matches Python where `past_key_states` (line
//     444) is the post-rotary state stashed by an earlier call's
//     `merged_key_states[packed_query_indexes] = packed_key_states` (line 450)
//     where `packed_key_states` has already been through `apply_rotary_pos_emb`
//     (lines 426-437) AND been cast back to BF16 (line 440). Repeat-kv (the
//     equivalent of GQA head expansion) happens inside flash_attn_varlen_func
//     internally, so the cache holds pre-expansion K. We mirror that.
//   - V stored at `[B, num_kv_heads, N_cached, head_dim]`, BF16. Same layout.
//   - Q is NEVER cached.

/// Per-layer K/V cache for Lance T2I gen-mode attention.
///
/// Mirrors Python `NaiveCache` (`qwen2_navit.py:49-66`). One `Option<Tensor>`
/// K slot and one V slot per layer. None until first populated. Cache state
/// grows along the seq_len dim (dim=2, the `N` of `[B, H_kv, N, D]`) as
/// prefill and gen-step calls populate.
///
/// API contract: the user creates one cache per model invocation with
/// `KvCache::new(num_layers)`, threads it through `LanceBlockStack::forward`
/// as `Some(&mut cache)`, and either reuses it for subsequent gen steps (for
/// the typical "build prefix once, run N denoise steps" pattern) or clears
/// it between batches via `KvCache::clear`.
///
/// **Clone semantics (C5):** the derived `Clone` is cheap — each `Tensor`
/// clone is an `Arc` bump on the underlying CUDA storage (per
/// `flame_core::Tensor`'s `#[derive(Clone)]` at `tensor.rs:136`), so
/// cloning a populated cache is `O(num_layers)` ref-bumps, not a real
/// device copy. C5's `denoise_loop` relies on this to thread two
/// independent `&mut KvCache` handles (one for cond, one for uncond)
/// through `gen_step` without aliasing the borrows held by the caller.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Per-layer K. Shape when present: `[B, num_kv_heads, N_cached, head_dim]` BF16.
    pub k: Vec<Option<Tensor>>,
    /// Per-layer V. Shape when present: `[B, num_kv_heads, N_cached, head_dim]` BF16.
    pub v: Vec<Option<Tensor>>,
}

impl KvCache {
    /// Empty cache for `num_layers` layers — all slots None.
    pub fn new(num_layers: usize) -> Self {
        Self {
            k: (0..num_layers).map(|_| None).collect(),
            v: (0..num_layers).map(|_| None).collect(),
        }
    }

    /// Cached seq_len at `layer_idx` (0 if the slot is empty).
    pub fn seq_len(&self, layer_idx: usize) -> usize {
        self.k[layer_idx]
            .as_ref()
            .map(|t| t.shape().dims()[2]) // [B, H_kv, N, D] → N
            .unwrap_or(0)
    }

    /// Free all cached K/V tensors (resets to fresh state).
    pub fn clear(&mut self) {
        for slot in &mut self.k {
            *slot = None;
        }
        for slot in &mut self.v {
            *slot = None;
        }
    }
}

// ===========================================================================
// Module 9: LanceBlockStack — 36 decoder layers + paired final norm
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/qwen2_navit.py:820-870`
// (`Qwen2NavitModel.__init__`). Each layer is dispatched between
// `Qwen2DecoderLayer` and `Qwen2MoTDecoderLayer` based on config — for
// Lance 3B, `layer_module=Qwen2MoTDecoderLayer` so all 36 are MoT.
// State-dict-derived: every layer key under
// `language_model.model.layers.{0..35}.*` carries the `_moe_gen` siblings
// (see STATE_DICT_KEYS.md), confirming all 36 are MoT for the 3B
// checkpoint.
//
// The top-level `language_model.model.norm` (final RMSNorm before LM head /
// `llm2vae`) is ALSO paired with `language_model.model.norm_moe_gen` per
// Lane C's audit. We apply the same `where_mask` routing here.

/// One block in the Lance decoder stack — either the base (non-MoT)
/// variant or the MoT-paired variant. Enum chosen over `Box<dyn Trait>`
/// to avoid heap-dispatch in the hot loop; the variant set is closed.
#[derive(Debug)]
pub enum LanceBlock {
    Base(Qwen2DecoderLayer),
    MoT(Qwen2MoTDecoderLayer),
}

impl LanceBlock {
    pub fn forward(
        &self,
        x: &Tensor,
        gen_mask: &Tensor,
        pos_t: &Tensor,
        pos_h: &Tensor,
        pos_w: &Tensor,
        mrope: &MropeFreqs,
        is_causal: bool,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
        update_cache: bool,
    ) -> Result<Tensor> {
        match self {
            // Base layer ignores gen_mask, is_causal, AND the KV cache.
            // It is dead code for Lance 3B production (all 36 layers are
            // MoT per STATE_DICT_KEYS.md). The is_causal + cache +
            // update_cache threading lives on the MoT arm only.
            LanceBlock::Base(layer) => {
                let _ = (is_causal, layer_idx, cache, update_cache, gen_mask);
                layer.forward(x, pos_t, pos_h, pos_w, mrope)
            }
            LanceBlock::MoT(layer) => {
                layer.forward(x, gen_mask, pos_t, pos_h, pos_w, mrope, is_causal, layer_idx, cache, update_cache)
            }
        }
    }
}

/// The 36-layer Lance decoder stack with paired final norm.
///
/// `final_norm` is the und-path final RMSNorm gain (`[hidden]`);
/// `final_norm_moe_gen` is the gen-path sibling. If no MoT layers are
/// present (`use_mot = false`), `final_norm_moe_gen` is still allocated
/// but unused — keep it for state-dict compatibility.
#[derive(Debug)]
pub struct LanceBlockStack {
    pub blocks: Vec<LanceBlock>,
    pub final_norm: Tensor,
    pub final_norm_moe_gen: Tensor,
    pub rms_norm_eps: f32,
}

impl LanceBlockStack {
    /// Random-init for C3 shape tests. Defaults to MoT per
    /// `LanceConfig.use_mot = true` for Lance 3B. The `num_layers`
    /// argument allows tests to use a shortened stack (e.g. 2 layers)
    /// without building the full 36-layer 3B in random init.
    ///
    /// Production loader (C6) will build from state-dict directly.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            if cfg.use_mot {
                blocks.push(LanceBlock::MoT(Qwen2MoTDecoderLayer::new_random(cfg)?));
            } else {
                blocks.push(LanceBlock::Base(Qwen2DecoderLayer::new_random(cfg)?));
            }
        }
        Ok(Self {
            blocks,
            final_norm: Qwen2Attention::ones_dt(&[h], dev, dt)?,
            final_norm_moe_gen: Qwen2Attention::ones_dt(&[h], dev, dt)?,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    /// Forward: run all blocks sequentially, then apply paired final
    /// RMSNorm with `gen_mask` routing.
    ///
    /// **F3 fix (2026-05-18):** `cache` threads per-layer K/V state
    /// through the block loop. Passing `Some(&mut KvCache)` enables
    /// gen-mode prefix attention; passing `None` is the
    /// no-cache prefill mode (text-only forward, or random-init test
    /// forward). The `as_deref_mut()` per-iteration re-borrow is the
    /// standard Rust idiom for threading `&mut` through a loop.
    pub fn forward(
        &self,
        x: &Tensor,
        gen_mask: &Tensor,
        pos_t: &Tensor,
        pos_h: &Tensor,
        pos_w: &Tensor,
        mrope: &MropeFreqs,
        is_causal: bool,
        mut cache: Option<&mut KvCache>,
        update_cache: bool,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward(
                &h,
                gen_mask,
                pos_t,
                pos_h,
                pos_w,
                mrope,
                is_causal,
                layer_idx,
                cache.as_deref_mut(),
                update_cache,
            )?;
        }
        // Final paired RMSNorm.
        let h_und = Self::rms_norm_apply(&h, &self.final_norm, self.rms_norm_eps)?;
        let h_gen = Self::rms_norm_apply(&h, &self.final_norm_moe_gen, self.rms_norm_eps)?;
        let dims = h.shape().dims();
        let mask_bn1 = gen_mask.reshape(&[dims[0], dims[1], 1])?;
        Tensor::where_mask(&mask_bn1, &h_gen, &h_und)
    }

    fn rms_norm_apply(x: &Tensor, gain: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.is_empty() {
            return Err(Error::InvalidInput(
                "LanceBlockStack::rms_norm_apply: input has no dims".into(),
            ));
        }
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(gain), eps)?;
        out.reshape(&dims)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device 0")
    }

    // ---- Module 2 ---------------------------------------------------------

    #[test]
    fn test_timestep_schedule_shift_4() {
        let d = dev();
        let sched = timestep_schedule(30, 4.0, &d).unwrap();
        let dims = sched.shape().dims();
        assert_eq!(dims, &[31usize]);

        // Read back as F32 and assert endpoints + monotonicity.
        let f32_sched = sched.to_dtype(DType::F32).unwrap();
        let v = f32_sched.to_vec().unwrap();
        assert_eq!(v.len(), 31);
        // First value = transform(1) = shift * 1 / (1 + (shift-1) * 1) = shift / shift = 1.0
        assert!((v[0] - 1.0).abs() < 1e-3, "v[0] = {} expected ~1.0", v[0]);
        // Last value = transform(0) = 0.
        assert!(v[30].abs() < 1e-3, "v[30] = {} expected ~0.0", v[30]);
        // Strictly decreasing.
        for i in 0..30 {
            assert!(
                v[i] > v[i + 1],
                "not monotonically decreasing at i={i}: {} -> {}",
                v[i],
                v[i + 1]
            );
        }
    }

    #[test]
    fn test_timestep_schedule_shift_1_is_linspace() {
        let d = dev();
        let sched = timestep_schedule(8, 1.0, &d).unwrap();
        let v = sched.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        // shift=1 → identity → linspace(1, 0, 9) = [1, 7/8, ..., 1/8, 0].
        let expected: Vec<f32> = (0..9).map(|i| 1.0 - (i as f32) / 8.0).collect();
        for (i, (a, b)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 5e-3,
                "shift=1 mismatch at i={i}: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_denoise_step_shape_preserved() {
        let d = dev();
        let x = Tensor::zeros(Shape::from_dims(&[2, 3, 4, 5]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::ones(Shape::from_dims(&[2, 3, 4, 5]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = denoise_step(&x, &v, 0.25).unwrap();
        assert_eq!(out.shape().dims(), x.shape().dims());
        // Output dtype is BF16 (matches inputs).
        assert_eq!(out.dtype(), DType::BF16);
    }

    // ---- Module 3 ---------------------------------------------------------

    /// Test helper: build three `[n]` I32 position tensors filled with the
    /// same scalar value. Mirrors the prior "single triplet for all tokens"
    /// shape but in the new per-token API. For `(v_t, v_h, v_w) = (0, 0, 0)`
    /// this reproduces the identity-rotation case used by every existing
    /// shape-preserving test.
    fn pos_uniform(
        n: usize,
        v_t: i32,
        v_h: i32,
        v_w: i32,
        dev: &Arc<CudaDevice>,
    ) -> (Tensor, Tensor, Tensor) {
        let t = Tensor::from_vec(
            vec![v_t as f32; n],
            Shape::from_dims(&[n]),
            dev.clone(),
        )
        .unwrap()
        .to_dtype(DType::I32)
        .unwrap();
        let h = Tensor::from_vec(
            vec![v_h as f32; n],
            Shape::from_dims(&[n]),
            dev.clone(),
        )
        .unwrap()
        .to_dtype(DType::I32)
        .unwrap();
        let w = Tensor::from_vec(
            vec![v_w as f32; n],
            Shape::from_dims(&[n]),
            dev.clone(),
        )
        .unwrap()
        .to_dtype(DType::I32)
        .unwrap();
        (t, h, w)
    }

    #[test]
    fn test_mrope_sections_sum_to_head_dim() {
        let sections = [16usize, 24, 24];
        // Qwen2.5-VL mRoPE: sum(sections) == head_dim/2; 2*sum == head_dim.
        let half_sum: usize = sections.iter().sum();
        assert_eq!(half_sum, 64, "Lance mRoPE sections must sum to head_dim/2 = 64");
        assert_eq!(half_sum * 2, 128, "2*sum(sections) must equal head_dim = 128");
    }

    #[test]
    fn test_apply_mrope_shape_preserved() {
        let d = dev();
        let sections = [16usize, 24, 24];
        let head_dim: usize = sections.iter().map(|s| s * 2).sum();
        let (b, h, n) = (1usize, 16usize, 64usize);

        let x = Tensor::ones(Shape::from_dims(&[b, h, n, head_dim]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let freqs =
            precompute_mrope_tables(sections, 1_000_000.0, 128, head_dim, &d, DType::BF16).unwrap();
        // Per-token uniform-(5,3,7) positions reproduce the old single-triplet
        // shape contract for this shape test.
        let (pt, ph, pw) = pos_uniform(n, 5, 3, 7, &d);
        let out = apply_mrope(&x, sections, &freqs, &pt, &ph, &pw).unwrap();
        assert_eq!(out.shape().dims(), &[b, h, n, head_dim]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn test_apply_mrope_identity_at_pos_zero() {
        let d = dev();
        let sections = [16usize, 24, 24];
        let head_dim: usize = sections.iter().map(|s| s * 2).sum();
        let (b, h, n) = (1usize, 4usize, 8usize);

        // Use a non-trivial input so we can compare values.
        let raw: Vec<f32> = (0..(b * h * n * head_dim))
            .map(|i| ((i % 17) as f32) * 0.01 - 0.05)
            .collect();
        let x = Tensor::from_vec(raw, Shape::from_dims(&[b, h, n, head_dim]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let freqs =
            precompute_mrope_tables(sections, 1_000_000.0, 16, head_dim, &d, DType::BF16).unwrap();
        // At pos=(0,0,0): cos=1, sin=0 everywhere → halfsplit RoPE is identity.
        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out = apply_mrope(&x, sections, &freqs, &pt, &ph, &pw).unwrap();

        let xv = x.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(xv.len(), ov.len());
        // BF16 round-trips through the kernel — tolerance set to a single ULP
        // worth of BF16 (~7.8e-3 relative). Use a generous absolute tolerance
        // since the input has magnitude < 0.2.
        let mut max_diff = 0.0f32;
        for (a, b_) in xv.iter().zip(ov.iter()) {
            let d_ = (a - b_).abs();
            if d_ > max_diff {
                max_diff = d_;
            }
        }
        assert!(
            max_diff < 1e-2,
            "identity-at-pos-0 max_abs diff {max_diff} exceeds 1e-2 BF16 tolerance"
        );
    }

    /// **Per-token mRoPE positions actually produce different rotations.**
    ///
    /// With per-token t-positions `[0, 1, 2]` (and h=w=0 to isolate the t-axis):
    ///   - token 0 sees cos(0)=1, sin(0)=0 → identity rotation → output ≈ input
    ///   - tokens 1, 2 see non-trivial rotations → output != input
    ///
    /// Confirms the per-token gather actually reads different table rows per
    /// token (not a stale "all tokens get the same row" bug that the old
    /// fixed-triplet API silently had at the call sites).
    #[test]
    fn test_mrope_per_token_positions_produce_different_rotations() {
        let d = dev();
        let sections = [16usize, 24, 24];
        let head_dim: usize = sections.iter().map(|s| s * 2).sum();
        let (b, h, n) = (1usize, 4usize, 3usize);

        // Non-trivial deterministic input.
        let raw: Vec<f32> = (0..(b * h * n * head_dim))
            .map(|i| ((i % 19) as f32) * 0.01 - 0.05)
            .collect();
        let x = Tensor::from_vec(raw, Shape::from_dims(&[b, h, n, head_dim]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let freqs =
            precompute_mrope_tables(sections, 1_000_000.0, 16, head_dim, &d, DType::BF16).unwrap();

        // Per-token t = [0, 1, 2], h = w = [0, 0, 0]. theta=1e6 + head_dim=128
        // gives a measurable rotation at t=1, t=2 for the first few channels
        // (low-index freqs ~ 1e-6 ... 1, so pos=1,2 → cos values up to ~0.54).
        let pos_t = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0],
            Shape::from_dims(&[n]),
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::I32)
        .unwrap();
        let pos_h = Tensor::from_vec(vec![0.0f32; n], Shape::from_dims(&[n]), d.clone())
            .unwrap()
            .to_dtype(DType::I32)
            .unwrap();
        let pos_w = pos_h.clone();
        let out = apply_mrope(&x, sections, &freqs, &pos_t, &pos_h, &pos_w).unwrap();

        let xv = x.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();

        // The tensor is laid out [B, H, N, head_dim] row-major: stride along
        // N is head_dim per head, but row-by-row we need to slice per
        // (b, h, n). For B=1 the linear index is `head_idx * (N * head_dim)
        // + n_idx * head_dim + d_idx`. Compute max-abs diff per token by
        // walking heads.
        let row_stride = head_dim;
        let head_stride = n * head_dim;
        let max_diff_at = |n_idx: usize| -> f32 {
            let mut m = 0.0f32;
            for head_idx in 0..h {
                let base = head_idx * head_stride + n_idx * row_stride;
                for d_idx in 0..head_dim {
                    let a = xv[base + d_idx];
                    let b_ = ov[base + d_idx];
                    let dd = (a - b_).abs();
                    if dd > m {
                        m = dd;
                    }
                }
            }
            m
        };

        let diff0 = max_diff_at(0);
        let diff1 = max_diff_at(1);
        let diff2 = max_diff_at(2);

        // Token 0: t=h=w=0 → identity (within BF16 round-trip).
        assert!(
            diff0 < 1e-2,
            "token 0 at pos (0,0,0) should be identity-rotated; got max_abs_diff = {diff0}"
        );
        // Tokens 1, 2: non-zero t → non-trivial rotation.
        assert!(
            diff1 > 1e-3,
            "token 1 at pos (1,0,0) should differ from input; got max_abs_diff = {diff1}"
        );
        assert!(
            diff2 > 1e-3,
            "token 2 at pos (2,0,0) should differ from input; got max_abs_diff = {diff2}"
        );
        // Tokens 1 and 2 should produce DIFFERENT rotated outputs.
        let mut max_12 = 0.0f32;
        for head_idx in 0..h {
            let base1 = head_idx * head_stride + 1 * row_stride;
            let base2 = head_idx * head_stride + 2 * row_stride;
            for d_idx in 0..head_dim {
                let dd = (ov[base1 + d_idx] - ov[base2 + d_idx]).abs();
                if dd > max_12 {
                    max_12 = dd;
                }
            }
        }
        assert!(
            max_12 > 1e-3,
            "token 1 and token 2 outputs should differ (different t-positions); got max_abs_diff = {max_12}"
        );
    }

    // ---- Module 4 ---------------------------------------------------------

    #[test]
    fn test_head_rms_norm_unit_gain_identity_norm() {
        let d = dev();
        let (b, h, n, dim) = (1usize, 2usize, 4usize, 128usize);

        // Build x such that each row's RMS is exactly 1.0 — then `x / rms = x`.
        // Use a constant row value c with c == 1 → mean(x^2) = 1, rms = 1.
        let total = b * h * n * dim;
        let raw = vec![1.0f32; total];
        let x = Tensor::from_vec(raw.clone(), Shape::from_dims(&[b, h, n, dim]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let gain = Tensor::ones(Shape::from_dims(&[dim]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = head_rms_norm(&x, &gain, 1e-6).unwrap();
        assert_eq!(out.shape().dims(), x.shape().dims());

        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for (i, v) in ov.iter().enumerate() {
            assert!(
                (*v - 1.0).abs() < 1e-2,
                "head_rms_norm unit-input/unit-gain: out[{i}] = {v}, expected ~1.0"
            );
        }
    }

    // ---- Regression / shape pinning --------------------------------------

    /// F6: pin LanceConfig::default_3b field values against accidental edits.
    #[test]
    fn test_lance_config_default_3b_values() {
        let cfg = LanceConfig::default_3b(dev());
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 11008);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
        assert_eq!(cfg.mrope_sections, [16, 24, 24]);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.latent_patch_size, (1, 1, 1)); // shell-config truth, matches vae2llm.weight=[2048, 48]
        assert_eq!(cfg.latent_channels, 48); // Wan 2.2 VAE z_dim
        assert!((cfg.timestep_shift - 4.0).abs() < 1e-6);
        assert!((cfg.cfg_text_scale - 4.0).abs() < 1e-6);
        assert_eq!(cfg.num_inference_steps, 30);
        assert_eq!(cfg.dtype, DType::BF16);
        assert!(cfg.visual_gen);
        assert!(cfg.visual_und);
        assert!(cfg.use_mot);
        // Sanity: patch_latent_dim derives from latent_patch_size * latent_channels.
        // Shell-config (1,1,1) * z_dim 48 = 48, matches vae2llm.weight=[2048, 48].
        assert_eq!(cfg.patch_latent_dim(), 1 * 1 * 1 * 48);
    }

    /// F7: pin per-axis cos/sin shapes from `precompute_mrope_tables`.
    #[test]
    fn test_precompute_mrope_table_shapes() {
        let d = dev();
        let freqs = precompute_mrope_tables(
            [16, 24, 24], // sections
            1_000_000.0,  // theta
            128,          // max_pos
            128,          // head_dim
            &d,
            DType::BF16,
        )
        .unwrap();

        // Per-axis cos/sin shapes: [1, 1, max_pos, sections[i]]
        assert_eq!(freqs.cos_t.shape().dims(), &[1, 1, 128, 16]);
        assert_eq!(freqs.sin_t.shape().dims(), &[1, 1, 128, 16]);
        assert_eq!(freqs.cos_h.shape().dims(), &[1, 1, 128, 24]);
        assert_eq!(freqs.sin_h.shape().dims(), &[1, 1, 128, 24]);
        assert_eq!(freqs.cos_w.shape().dims(), &[1, 1, 128, 24]);
        assert_eq!(freqs.sin_w.shape().dims(), &[1, 1, 128, 24]);
    }

    /// F7 (bonus): at position 0 cos = 1, sin = 0 exactly — catches any
    /// off-by-one in the `arange(0, dim, 2)` frequency construction.
    /// Uses F32 precompute to avoid BF16 rounding noise.
    #[test]
    fn test_precompute_mrope_pos_zero_is_identity() {
        let d = dev();
        let freqs = precompute_mrope_tables(
            [16, 24, 24],
            1_000_000.0,
            4, // max_pos
            128,
            &d,
            DType::F32,
        )
        .unwrap();

        // Layout: [1, 1, max_pos=4, section]. Row 0 == first `section` entries
        // of the flattened F32 buffer (contiguous, row-major).
        let cos_t_all = freqs.cos_t.to_vec().unwrap();
        let sin_t_all = freqs.sin_t.to_vec().unwrap();
        let cos_h_all = freqs.cos_h.to_vec().unwrap();
        let sin_h_all = freqs.sin_h.to_vec().unwrap();
        let cos_w_all = freqs.cos_w.to_vec().unwrap();
        let sin_w_all = freqs.sin_w.to_vec().unwrap();

        for &c in &cos_t_all[..16] {
            assert!((c - 1.0).abs() < 1e-5, "cos_t at pos 0 must be 1.0, got {c}");
        }
        for &s in &sin_t_all[..16] {
            assert!(s.abs() < 1e-5, "sin_t at pos 0 must be 0.0, got {s}");
        }
        for &c in &cos_h_all[..24] {
            assert!((c - 1.0).abs() < 1e-5, "cos_h at pos 0 must be 1.0, got {c}");
        }
        for &s in &sin_h_all[..24] {
            assert!(s.abs() < 1e-5, "sin_h at pos 0 must be 0.0, got {s}");
        }
        for &c in &cos_w_all[..24] {
            assert!((c - 1.0).abs() < 1e-5, "cos_w at pos 0 must be 1.0, got {c}");
        }
        for &s in &sin_w_all[..24] {
            assert!(s.abs() < 1e-5, "sin_w at pos 0 must be 0.0, got {s}");
        }
    }

    // ---- C2: Module 5 (Qwen2Attention) ------------------------------------

    #[test]
    fn test_qwen2_attention_shape_preserved() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let attn = Qwen2Attention::new_random(&cfg).unwrap();
        // max_pos=64 is enough for pos_indices = [0,0,0] and N=16.
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections,
            cfg.rope_theta,
            64,
            cfg.head_dim,
            &d,
            DType::BF16,
        )
        .unwrap();

        // Small input: B=1, N=16, hidden=2048. Use a small std so projections
        // don't overflow BF16 dynamic range. Tensor::from_vec gives a known,
        // contig buffer.
        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;
        let raw: Vec<f32> = (0..(b * n * h))
            .map(|i| ((i % 23) as f32) * 0.005 - 0.05)
            .collect();
        let x = Tensor::from_vec(raw, Shape::from_dims(&[b, n, h]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out = attn.forward(&x, &pt, &ph, &pw, &mrope).unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn test_qwen2_mlp_shape_preserved() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let mlp = Qwen2MLP::new_random(&cfg).unwrap();

        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;
        let raw: Vec<f32> = (0..(b * n * h))
            .map(|i| ((i % 31) as f32) * 0.005 - 0.075)
            .collect();
        let x = Tensor::from_vec(raw, Shape::from_dims(&[b, n, h]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn test_qwen2_decoder_layer_shape_preserved() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let layer = Qwen2DecoderLayer::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections,
            cfg.rope_theta,
            64,
            cfg.head_dim,
            &d,
            DType::BF16,
        )
        .unwrap();

        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;
        let raw: Vec<f32> = (0..(b * n * h))
            .map(|i| ((i % 19) as f32) * 0.005 - 0.04)
            .collect();
        let x = Tensor::from_vec(raw, Shape::from_dims(&[b, n, h]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out = layer.forward(&x, &pt, &ph, &pw, &mrope).unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    /// Detect Q/K-norm-before-mRoPE ordering at runtime.
    ///
    /// **Idea.** Build an attention with `q_norm = 0` (zero RMSNorm gain).
    /// With the documented order — projection → reshape/permute → Q-norm
    /// → mRoPE → repeat-kv → SDPA — Q becomes EXACTLY zero after the Q-norm
    /// step (`x * 0 = 0`), then mRoPE rotates a zero vector and gets zero,
    /// then SDPA computes `softmax(0 * K^T) @ V`, which is the identity
    /// uniform-over-N mix of `V`. The output then equals `o_proj(V_mean)`
    /// independent of `x`.
    ///
    /// If a future maintainer reordered to mRoPE-then-Q-norm, the rotated Q
    /// would still get zeroed, so this test would NOT distinguish them.
    /// To actually pin the order, we build TWO inputs `x1`, `x2` that differ
    /// only in their K and V content (V values, which are not normed) and
    /// assert the outputs ALSO differ. Then we build TWO inputs that differ
    /// in their Q-input but with V held fixed via zero-Q routing, asserting
    /// outputs are EQUAL (because Q has been zeroed before any consumer).
    ///
    /// The cleaner runtime test: build the attention with `q_norm = 0` AND
    /// `k_norm = 0` AND scale both V biases / proj to a known value. Then:
    ///   - softmax(0 @ 0) is uniform over N, so attn ≈ mean(V) per token.
    ///   - V depends on input via v_proj; for two inputs x_a, x_b the outputs
    ///     should differ ONLY through V (not through Q or K).
    /// We verify the simpler property: replacing the INPUT with a same-shape
    /// random tensor, while keeping q_norm = k_norm = 0, must still produce
    /// a finite, non-NaN output dependent on V (which depends on x). This
    /// catches a reorder that makes Q non-zero (mRoPE-first), because mRoPE
    /// applied to a non-zero Q before zeroing would leave nondeterministic
    /// post-projection-then-rope rotation in q before Q-norm... but the
    /// halfsplit RoPE is also a linear operator on Q, so `q_norm = 0` still
    /// kills it post-norm. **So the runtime distinguisher is weak.**
    ///
    /// **Conclusion.** A true runtime test for ordering is invasive (needs
    /// stage-split forward). We instead pin the ordering by documentation
    /// (see the `forward` method's step comments at the top of Module 5)
    /// AND by this smoke test that runs the documented path on the
    /// `q_norm = k_norm = 0` config and asserts:
    ///   - output is finite (no NaN / Inf)
    ///   - output shape is preserved
    ///   - output is dependent on `x` through V (so the FFN-less attention
    ///     block isn't producing a constant by accident)
    #[test]
    fn test_qwen2_attention_q_norm_before_mrope_smoke() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let mut attn = Qwen2Attention::new_random(&cfg).unwrap();
        // Zero Q and K norm gains.
        attn.q_norm = Qwen2Attention::zeros_dt(&[cfg.head_dim], &d, cfg.dtype).unwrap();
        attn.k_norm = Qwen2Attention::zeros_dt(&[cfg.head_dim], &d, cfg.dtype).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections,
            cfg.rope_theta,
            64,
            cfg.head_dim,
            &d,
            DType::BF16,
        )
        .unwrap();

        let b = 1usize;
        let n = 8usize;
        let h = cfg.hidden_size;
        let raw_a: Vec<f32> = (0..(b * n * h))
            .map(|i| ((i % 17) as f32) * 0.005 - 0.04)
            .collect();
        let raw_b: Vec<f32> = (0..(b * n * h))
            .map(|i| ((i % 11) as f32) * 0.007 + 0.01)
            .collect();
        let xa = Tensor::from_vec(raw_a, Shape::from_dims(&[b, n, h]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let xb = Tensor::from_vec(raw_b, Shape::from_dims(&[b, n, h]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let oa = attn.forward(&xa, &pt, &ph, &pw, &mrope).unwrap();
        let ob = attn.forward(&xb, &pt, &ph, &pw, &mrope).unwrap();
        // Shape preserved.
        assert_eq!(oa.shape().dims(), &[b, n, h]);
        assert_eq!(ob.shape().dims(), &[b, n, h]);
        // Finite (no NaN/Inf from zeroed norms).
        let va = oa.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let vb = ob.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in va.iter().chain(vb.iter()) {
            assert!(v.is_finite(), "non-finite element {v} in zeroed-norm output");
        }
        // V flows through despite Q/K being zeroed → outputs must differ
        // between the two inputs (because v_proj depends on x). If a
        // misorder accidentally killed V too, this would catch it.
        let mut max_diff = 0.0f32;
        for (a, b_) in va.iter().zip(vb.iter()) {
            let d_ = (a - b_).abs();
            if d_ > max_diff {
                max_diff = d_;
            }
        }
        assert!(
            max_diff > 1e-3,
            "zeroed-Q/K attention produced identical outputs for distinct inputs (max_diff = {max_diff})"
        );

        // NOTE: this is a SMOKE test for V-flow + finiteness. The true
        // ordering pin is documented in `Qwen2Attention::forward`'s step
        // comments (Module 5 doc-block). A future maintainer who reorders
        // norm and mRoPE must read those comments.
    }

    // ---- C3: MoT modules (8a/8b/8c, 9) ------------------------------------

    /// Build a `[B, N]` BF16 mask from a 0/1 f32 pattern.
    fn make_mask(b: usize, n: usize, pattern: &[f32], dev: &Arc<CudaDevice>) -> Tensor {
        assert_eq!(pattern.len(), b * n);
        Tensor::from_vec(pattern.to_vec(), Shape::from_dims(&[b, n]), dev.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    /// Build an `[B, N, H]` BF16 input from a deterministic value pattern.
    fn make_input(b: usize, n: usize, h: usize, seed: i32, dev: &Arc<CudaDevice>) -> Tensor {
        let raw: Vec<f32> = (0..(b * n * h))
            .map(|i| (((i as i32 + seed) % 23) as f32) * 0.005 - 0.05)
            .collect();
        Tensor::from_vec(raw, Shape::from_dims(&[b, n, h]), dev.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    #[test]
    fn test_mot_attention_shape_preserved() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let attn = Qwen2MoTAttention::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections, cfg.rope_theta, 64, cfg.head_dim, &d, DType::BF16,
        )
        .unwrap();
        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;

        // Mixed mask: first half und, second half gen.
        let pattern: Vec<f32> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        let gen_mask = make_mask(b, n, &pattern, &d);
        let x = make_input(b, n, h, 0, &d);
        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out = attn
            .forward(&x, &gen_mask, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
        // Finite check — routing path must not explode.
        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in ov.iter() {
            assert!(v.is_finite(), "non-finite mot attn output element {v}");
        }
    }

    #[test]
    fn test_mot_mlp_shape_preserved() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let mlp = Qwen2MoTMLP::new_random(&cfg).unwrap();
        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;
        let pattern: Vec<f32> = (0..n).map(|i| (i & 1) as f32).collect(); // alternating
        let gen_mask = make_mask(b, n, &pattern, &d);
        let x = make_input(b, n, h, 7, &d);
        let out = mlp.forward(&x, &gen_mask).unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in ov.iter() {
            assert!(v.is_finite(), "non-finite mot mlp output element {v}");
        }
    }

    #[test]
    fn test_mot_decoder_layer_shape_preserved() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let layer = Qwen2MoTDecoderLayer::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections, cfg.rope_theta, 64, cfg.head_dim, &d, DType::BF16,
        )
        .unwrap();
        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;
        let pattern: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let gen_mask = make_mask(b, n, &pattern, &d);
        let x = make_input(b, n, h, 11, &d);
        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out = layer
            .forward(&x, &gen_mask, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in ov.iter() {
            assert!(v.is_finite(), "non-finite mot layer output element {v}");
        }
    }

    #[test]
    fn test_block_stack_shape_preserved() {
        let d = dev();
        // Shorten the stack from 36 → 2 layers for test speed (full 36
        // would allocate 36 × 2 × (q+k+v+o+mlp) ≈ a lot of GPU memory
        // just for shape testing).
        let mut cfg = LanceConfig::default_3b(d.clone());
        cfg.num_hidden_layers = 2;
        let stack = LanceBlockStack::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections, cfg.rope_theta, 64, cfg.head_dim, &d, DType::BF16,
        )
        .unwrap();
        let b = 1usize;
        let n = 16usize;
        let h = cfg.hidden_size;
        let pattern: Vec<f32> = (0..n).map(|i| if i < 4 { 1.0 } else { 0.0 }).collect();
        let gen_mask = make_mask(b, n, &pattern, &d);
        let x = make_input(b, n, h, 13, &d);
        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out = stack
            .forward(&x, &gen_mask, &pt, &ph, &pw, &mrope, false, None, false)
            .unwrap();
        assert_eq!(out.shape().dims(), &[b, n, h]);
        assert_eq!(out.dtype(), DType::BF16);
        // Finite — propagated through 2 MoT layers + final paired norm.
        let ov = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in ov.iter() {
            assert!(v.is_finite(), "non-finite stack output element {v}");
        }
    }

    /// **Routing actually selects per-token.**
    ///
    /// Build a single MoT attention layer. Run it with `gen_mask = all 0`
    /// (all-und). Run again with `gen_mask = all 1` (all-gen). Because the
    /// gen weights are seeded DIFFERENTLY from the und weights in
    /// `new_random` (8001..8004 vs 8101..8104), the two outputs MUST
    /// differ. If they were identical, either (a) the gen weights are not
    /// being used, or (b) routing collapses to a single path.
    ///
    /// We also verify that a MIXED mask (half-and-half) gives a different
    /// output than either all-und or all-gen — this catches a bug where
    /// the mask is silently ignored and one path always wins.
    #[test]
    fn test_mot_attention_routing_picks_different_paths() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let attn = Qwen2MoTAttention::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections, cfg.rope_theta, 64, cfg.head_dim, &d, DType::BF16,
        )
        .unwrap();
        let b = 1usize;
        let n = 8usize;
        let h = cfg.hidden_size;
        let x = make_input(b, n, h, 23, &d);

        let mask_und = make_mask(b, n, &vec![0.0f32; b * n], &d);
        let mask_gen = make_mask(b, n, &vec![1.0f32; b * n], &d);
        let mask_mixed_pattern: Vec<f32> = (0..b * n).map(|i| (i & 1) as f32).collect();
        let mask_mixed = make_mask(b, n, &mask_mixed_pattern, &d);

        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let o_und = attn
            .forward(&x, &mask_und, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();
        let o_gen = attn
            .forward(&x, &mask_gen, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();
        let o_mix = attn
            .forward(&x, &mask_mixed, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();

        let v_und = o_und.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let v_gen = o_gen.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let v_mix = o_mix.to_dtype(DType::F32).unwrap().to_vec().unwrap();

        // und vs gen must differ (different weights, same input).
        let max_ug = v_und
            .iter()
            .zip(v_gen.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_ug > 1e-3,
            "all-und vs all-gen outputs are identical (max_diff = {max_ug}) — gen weights ignored?"
        );

        // mixed must differ from both endpoints (otherwise mask is
        // collapsing to one path).
        let max_um = v_und
            .iter()
            .zip(v_mix.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_gm = v_gen
            .iter()
            .zip(v_mix.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_um > 1e-3,
            "mixed mask output equals all-und output (mask ignored or stuck at und)"
        );
        assert!(
            max_gm > 1e-3,
            "mixed mask output equals all-gen output (mask ignored or stuck at gen)"
        );
    }

    /// **Sanity:** with `gen_mask = all 0` the MoT path must produce a
    /// finite, distinct output from the same `make_input`. (Equivalence to
    /// a hand-built `Qwen2Attention` with the same und weights is invasive
    /// to set up — needs sharing tensor handles across two structs — so
    /// we instead check the looser property that the und-only routed
    /// output is finite and stable across two invocations.)
    #[test]
    fn test_mot_attention_all_und_finite_and_stable() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let attn = Qwen2MoTAttention::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections, cfg.rope_theta, 64, cfg.head_dim, &d, DType::BF16,
        )
        .unwrap();
        let b = 1usize;
        let n = 8usize;
        let h = cfg.hidden_size;
        let x = make_input(b, n, h, 29, &d);
        let mask_und = make_mask(b, n, &vec![0.0f32; b * n], &d);

        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let o1 = attn
            .forward(&x, &mask_und, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();
        let o2 = attn
            .forward(&x, &mask_und, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();
        let v1 = o1.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let v2 = o2.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        // Determinism check: same input + same mask + same weights → same out.
        let max_diff = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "MoT attn forward not deterministic across two calls (max_diff = {max_diff})"
        );
        for v in v1.iter() {
            assert!(v.is_finite(), "non-finite und-only output {v}");
        }
    }

    /// **F2 fix verification (2026-05-18):** causal vs non-causal SDPA
    /// paths must produce DIFFERENT outputs on the same input. The
    /// causal keep-mask zeros out attention to future positions, so
    /// rows past index 0 must see different attention distributions
    /// than the unmasked path. If they were equal, the `is_causal`
    /// flag is silently ignored.
    #[test]
    fn test_mot_attention_causal_and_noncausal_differ() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let attn = Qwen2MoTAttention::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections, cfg.rope_theta, 64, cfg.head_dim, &d, DType::BF16,
        )
        .unwrap();
        let b = 1usize;
        let n = 8usize;
        let h = cfg.hidden_size;
        // Use a non-trivial input so the causal mask actually changes
        // the attention output (a constant input would still differ
        // because rows past 0 have fewer keys to attend to, but a
        // patterned input gives a stronger signal).
        let x = make_input(b, n, h, 9001, &d);
        // All-und mask — exercise the und path; causal/non-causal split
        // is the only difference between the two forward calls.
        let mask_und = make_mask(b, n, &vec![0.0f32; b * n], &d);

        let (pt, ph, pw) = pos_uniform(n, 0, 0, 0, &d);
        let out_causal = attn
            .forward(&x, &mask_und, &pt, &ph, &pw, &mrope, true, 0, None, false)
            .unwrap();
        let out_noncausal = attn
            .forward(&x, &mask_und, &pt, &ph, &pw, &mrope, false, 0, None, false)
            .unwrap();

        // Shapes match.
        assert_eq!(out_causal.shape().dims(), out_noncausal.shape().dims());
        assert_eq!(out_causal.shape().dims(), &[b, n, h]);

        let v_c = out_causal.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let v_nc = out_noncausal.to_dtype(DType::F32).unwrap().to_vec().unwrap();

        // Outputs MUST differ — causal mask zeros out attention to
        // future positions, which materially changes the attention
        // distribution for every token except index 0 (which attends
        // only to itself in both paths).
        let max_diff = v_c
            .iter()
            .zip(v_nc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "causal and non-causal outputs are identical (max_diff = {max_diff}) — is_causal flag silently ignored?"
        );

        // Both outputs must be finite.
        for v in v_c.iter() {
            assert!(v.is_finite(), "non-finite causal output {v}");
        }
        for v in v_nc.iter() {
            assert!(v.is_finite(), "non-finite non-causal output {v}");
        }
    }

    /// **F3 fix verification (2026-05-18):** KV cache grows along the
    /// seq dim across consecutive forward calls. A 2-layer block stack
    /// is fed 8 tokens (cache should populate to N=8 for both layers),
    /// then 4 more tokens with the same cache (cache should grow to
    /// N=12 for both layers). If the cache write-back is silently
    /// skipped, seq_len stays at 0. If the concat is wrong, growth
    /// breaks.
    #[test]
    fn test_mot_attention_kv_cache_grows_seq() {
        let d = dev();
        let mut cfg = LanceConfig::default_3b(d.clone());
        cfg.num_hidden_layers = 2;
        let stack = LanceBlockStack::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections,
            cfg.rope_theta,
            64,
            cfg.head_dim,
            &d,
            DType::BF16,
        )
        .unwrap();

        let b = 1usize;
        let h = cfg.hidden_size;

        let mut cache = KvCache::new(cfg.num_hidden_layers);
        assert_eq!(cache.seq_len(0), 0);
        assert_eq!(cache.seq_len(1), 0);

        // --- First call: 8 tokens. Cache populates to N=8 per layer. ---
        let n1 = 8usize;
        let gen_mask_8 = make_mask(b, n1, &vec![0.0f32; b * n1], &d);
        let x1 = make_input(b, n1, h, 9001, &d);
        let (pt1, ph1, pw1) = pos_uniform(n1, 0, 0, 0, &d);
        let _ = stack
            .forward(
                &x1,
                &gen_mask_8,
                &pt1,
                &ph1,
                &pw1,
                &mrope,
                false,
                Some(&mut cache),
                true,
            )
            .unwrap();
        assert_eq!(
            cache.seq_len(0),
            8,
            "layer 0 cache should be 8 after first call"
        );
        assert_eq!(
            cache.seq_len(1),
            8,
            "layer 1 cache should be 8 after first call"
        );
        // Per-layer K shape check: [B, num_kv_heads, N, head_dim]
        let k0_dims = cache.k[0].as_ref().unwrap().shape().dims().to_vec();
        assert_eq!(
            k0_dims,
            vec![b, cfg.num_kv_heads, 8, cfg.head_dim],
            "cache K layer 0 shape mismatch"
        );
        let v0_dims = cache.v[0].as_ref().unwrap().shape().dims().to_vec();
        assert_eq!(
            v0_dims,
            vec![b, cfg.num_kv_heads, 8, cfg.head_dim],
            "cache V layer 0 shape mismatch"
        );

        // --- Second call: 4 more tokens. Cache should grow to N=12. ---
        let n2 = 4usize;
        let gen_mask_4 = make_mask(b, n2, &vec![0.0f32; b * n2], &d);
        let x2 = make_input(b, n2, h, 9002, &d);
        let (pt2, ph2, pw2) = pos_uniform(n2, 0, 0, 0, &d);
        let _ = stack
            .forward(
                &x2,
                &gen_mask_4,
                &pt2,
                &ph2,
                &pw2,
                &mrope,
                false,
                Some(&mut cache),
                true,
            )
            .unwrap();
        assert_eq!(
            cache.seq_len(0),
            12,
            "layer 0 cache should be 12 after second call (8 + 4)"
        );
        assert_eq!(
            cache.seq_len(1),
            12,
            "layer 1 cache should be 12 after second call (8 + 4)"
        );

        // --- Clear resets all slots. ---
        cache.clear();
        assert_eq!(cache.seq_len(0), 0);
        assert_eq!(cache.seq_len(1), 0);
    }

    /// **Parity-blocking gap fix verification (2026-05-18):**
    /// `update_cache=false` means the cache is READ but NOT written.
    ///
    /// Mirrors Python `qwen2_navit.py:222-224` (write gated on
    /// `update_past_key_values`) and `lance.py:1724, 1742, 1758` where
    /// per-step gen calls pass `update_past_key_values=False` so per-
    /// step latent K/V do not pollute the prefix cache across the N
    /// denoise steps.
    ///
    /// Without this gating, calling forward N times with the same
    /// cache makes it balloon to `[text_prefix | step1_K | step2_K |
    /// ... | stepN_K]` and per-step attention sees stale K/V — T2I
    /// with 30 steps produces garbage.
    #[test]
    fn test_mot_attention_cache_no_update_doesnt_grow() {
        let d = dev();
        let mut cfg = LanceConfig::default_3b(d.clone());
        cfg.num_hidden_layers = 2;
        let stack = LanceBlockStack::new_random(&cfg).unwrap();
        let mrope = precompute_mrope_tables(
            cfg.mrope_sections,
            cfg.rope_theta,
            64,
            cfg.head_dim,
            &d,
            DType::BF16,
        )
        .unwrap();

        let b = 1usize;
        let h = cfg.hidden_size;

        let mut cache = KvCache::new(cfg.num_hidden_layers);

        // --- PREFILL: 8 tokens, update_cache=true → cache grows to 8. ---
        let n1 = 8usize;
        let gen_mask_8 = make_mask(b, n1, &vec![0.0f32; b * n1], &d);
        let x1 = make_input(b, n1, h, 9001, &d);
        let (pt1, ph1, pw1) = pos_uniform(n1, 0, 0, 0, &d);
        let _ = stack
            .forward(
                &x1,
                &gen_mask_8,
                &pt1,
                &ph1,
                &pw1,
                &mrope,
                false,
                Some(&mut cache),
                true,
            )
            .unwrap();
        assert_eq!(
            cache.seq_len(0),
            8,
            "layer 0 cache should be 8 after PREFILL"
        );
        assert_eq!(
            cache.seq_len(1),
            8,
            "layer 1 cache should be 8 after PREFILL"
        );

        // --- GEN-STEP 1: 4 tokens, update_cache=false → cache stays at 8. ---
        let n2 = 4usize;
        let gen_mask_4 = make_mask(b, n2, &vec![0.0f32; b * n2], &d);
        let x2 = make_input(b, n2, h, 9002, &d);
        let (pt2, ph2, pw2) = pos_uniform(n2, 0, 0, 0, &d);
        let _ = stack
            .forward(
                &x2,
                &gen_mask_4,
                &pt2,
                &ph2,
                &pw2,
                &mrope,
                false,
                Some(&mut cache),
                false,
            )
            .unwrap();
        assert_eq!(
            cache.seq_len(0),
            8,
            "layer 0 cache must stay at 8 after GEN-STEP (not 12)"
        );
        assert_eq!(
            cache.seq_len(1),
            8,
            "layer 1 cache must stay at 8 after GEN-STEP (not 12)"
        );

        // --- GEN-STEP 2: another 4 tokens, update_cache=false → still 8. ---
        let x3 = make_input(b, n2, h, 9003, &d);
        let _ = stack
            .forward(
                &x3,
                &gen_mask_4,
                &pt2,
                &ph2,
                &pw2,
                &mrope,
                false,
                Some(&mut cache),
                false,
            )
            .unwrap();
        assert_eq!(
            cache.seq_len(0),
            8,
            "layer 0 cache must STILL be 8 after second GEN-STEP"
        );
        assert_eq!(
            cache.seq_len(1),
            8,
            "layer 1 cache must STILL be 8 after second GEN-STEP"
        );
    }
}

// ===========================================================================
// C4 Module 10: `vae2llm` and `llm2vae` Linear projections
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/lance.py:103-107`
//   self.patch_latent_dim = pt * ph * pw * latent_channel
//   self.vae2llm = nn.Linear(patch_latent_dim, hidden_size)
//   self.llm2vae = nn.Linear(hidden_size, patch_latent_dim)
//
// Both Linear layers carry `bias=True` (state-dict has `vae2llm.bias [2048]`
// and `llm2vae.bias [48]`, per `STATE_DICT_KEYS.md`).
//
// Shape contract (shell-config `latent_patch_size=(1,1,1)`, `latent_channels=48`):
//   patch_latent_dim = 1 * 1 * 1 * 48 = 48      (matches state_dict)
//   vae2llm.weight   = [hidden=2048,  patch_latent_dim=48]
//   vae2llm.bias     = [hidden=2048]
//   llm2vae.weight   = [patch_latent_dim=48, hidden=2048]
//   llm2vae.bias     = [patch_latent_dim=48]
//
// Both projections use `flame_core::ops::fused_inference::fused_linear3d_native`
// — the same fused cuBLASLt path that Module 5 (`Qwen2Attention`) and Module 6
// (`Qwen2MLP`) use for their Q/K/V/O and gate/up/down Linears. Input must be
// 3D `[B, N, Cin]`, weight `[Cout, Cin]` PyTorch row-major, optional bias `[Cout]`.

/// `vae2llm`: project patchified VAE latent tokens into the LLM's hidden space.
///
/// Forward: `[B, L_image, patch_latent_dim] -> [B, L_image, hidden_size]`.
#[derive(Debug)]
pub struct Vae2Llm {
    /// `[hidden_size=2048, patch_latent_dim=48]` PyTorch row-major.
    pub weight: Tensor,
    /// `[hidden_size=2048]`.
    pub bias: Tensor,
}

impl Vae2Llm {
    /// Random-init constructor for C4 shape tests. C6 supplies real weights.
    /// Uses the same `randn_seeded` + dtype-cast pattern as Qwen2Attention.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let p = cfg.patch_latent_dim();
        let weight = Qwen2Attention::randn_dt(&[h, p], 0.02, 4001, dev, dt)?;
        let bias = Qwen2Attention::zeros_dt(&[h], dev, dt)?;
        Ok(Self { weight, bias })
    }

    /// Forward: `[B, L_image, patch_latent_dim] -> [B, L_image, hidden_size]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Vae2Llm::forward: expected 3D input [B, L_image, patch_latent_dim], got {dims:?}"
            )));
        }
        // fused_linear3d_native: input [B, N, Cin], weight [Cout, Cin] PyTorch
        // row-major, optional bias [Cout]. Output [B, N, Cout].
        flame_core::ops::fused_inference::fused_linear3d_native(x, &self.weight, Some(&self.bias))
    }
}

/// `llm2vae`: project LLM hidden states back to patchified VAE latent space.
///
/// Forward: `[B, L_image, hidden_size] -> [B, L_image, patch_latent_dim]`.
#[derive(Debug)]
pub struct Llm2Vae {
    /// `[patch_latent_dim=48, hidden_size=2048]` PyTorch row-major.
    pub weight: Tensor,
    /// `[patch_latent_dim=48]`.
    pub bias: Tensor,
}

impl Llm2Vae {
    /// Random-init constructor for C4 shape tests. C6 supplies real weights.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let p = cfg.patch_latent_dim();
        let weight = Qwen2Attention::randn_dt(&[p, h], 0.02, 4002, dev, dt)?;
        let bias = Qwen2Attention::zeros_dt(&[p], dev, dt)?;
        Ok(Self { weight, bias })
    }

    /// Forward: `[B, L_image, hidden_size] -> [B, L_image, patch_latent_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Llm2Vae::forward: expected 3D input [B, L_image, hidden_size], got {dims:?}"
            )));
        }
        flame_core::ops::fused_inference::fused_linear3d_native(x, &self.weight, Some(&self.bias))
    }
}

// ===========================================================================
// C4 Module 11: `TimeEmbedder` — sinusoidal scalar embedding + 2-layer MLP
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/modeling_utils.py:110-146`
// (`class TimestepEmbedder`). Python class structure:
//
//   self.mlp = nn.Sequential(
//       nn.Linear(frequency_embedding_size=256, hidden_size=2048, bias=True),
//       nn.SiLU(),
//       nn.Linear(hidden_size=2048, hidden_size=2048, bias=True),
//   )
//
//   @staticmethod
//   def timestep_embedding(t, dim=256, max_period=10000):
//       half = dim // 2                                          # 128
//       freqs = torch.exp(
//           -math.log(max_period) * torch.arange(0, half, dtype=float32) / half
//       ).to(device=t.device)
//       args  = t[:, None].float() * freqs[None]                 # [L, 128]
//       embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # [L, 256]
//       return embedding
//
//   def forward(self, t):
//       t_freq = self.timestep_embedding(t, 256)
//       return self.mlp(t_freq)
//
// **Parity notes against the task prompt's recipe:**
//   - `max_period = 10000` (NOT a `theta` config-driven value; hardcoded by Python at
//     `modeling_utils.py:124`).
//   - The cat order is **`cat([cos, sin], -1)`** — cos comes first, sin second.
//     This differs from the generic 1D sin-cos pos embed at
//     `modeling_utils.py:52-70` (`get_1d_sincos_pos_embed_from_grid`) which uses
//     `cat([sin, cos], -1)`. The task-prompt recipe says `cat([sin, cos], -1)`
//     for the time embedder, which is WRONG against Python. We follow Python.
//   - Python upcasts `t` to F32 before the multiply (`t[:, None].float()`). We
//     mirror that by building the freq/arg/cos/sin pipeline in F32 on CPU
//     (the same pattern Module 3 mRoPE uses), then casting to BF16 before the
//     MLP — this is the lowest-noise, no-new-kernel path.
//   - `mlp.0` and `mlp.2` indices imply `nn.Sequential([Linear, Activation, Linear])`
//     with the SiLU activation at index 1 (no params). Our struct stores only
//     the two Linear weights + biases.

/// `TimeEmbedder`: scalar flow-matching timestep → `[L, hidden_size]`
/// per-token time embedding.
///
/// Pipeline:
///   1. Sinusoidal embedding: `[L]` → `[L, sinusoidal_dim=256]`
///      with `freqs = exp(-ln(10000) * arange(0, 128) / 128)`,
///      `args = t * freqs`, `emb = cat([cos(args), sin(args)], -1)`
///      (cos-first per Python `modeling_utils.py:138`).
///   2. MLP layer 0: Linear(256 → 2048) + bias.
///   3. SiLU activation.
///   4. MLP layer 2: Linear(2048 → 2048) + bias.
#[derive(Debug)]
pub struct TimeEmbedder {
    /// `mlp.0`: `[hidden_size=2048, sinusoidal_dim=256]` PyTorch row-major.
    pub mlp_0_weight: Tensor,
    /// `mlp.0` bias: `[hidden_size=2048]`.
    pub mlp_0_bias: Tensor,
    /// `mlp.2`: `[hidden_size=2048, hidden_size=2048]` PyTorch row-major.
    pub mlp_2_weight: Tensor,
    /// `mlp.2` bias: `[hidden_size=2048]`.
    pub mlp_2_bias: Tensor,
    /// 256 (hardcoded in Python `__init__` default arg).
    pub sinusoidal_dim: usize,
    /// 2048 — must match `cfg.hidden_size`.
    pub hidden_size: usize,
}

impl TimeEmbedder {
    /// Random-init constructor for C4 shape tests. C6 supplies real weights.
    pub fn new_random(cfg: &LanceConfig) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;
        let h = cfg.hidden_size;
        let s = 256usize; // sinusoidal_dim, Python `frequency_embedding_size=256`
        let mlp_0_weight = Qwen2Attention::randn_dt(&[h, s], 0.02, 4011, dev, dt)?;
        let mlp_0_bias = Qwen2Attention::zeros_dt(&[h], dev, dt)?;
        let mlp_2_weight = Qwen2Attention::randn_dt(&[h, h], 0.02, 4012, dev, dt)?;
        let mlp_2_bias = Qwen2Attention::zeros_dt(&[h], dev, dt)?;
        Ok(Self {
            mlp_0_weight,
            mlp_0_bias,
            mlp_2_weight,
            mlp_2_bias,
            sinusoidal_dim: s,
            hidden_size: h,
        })
    }

    /// Build the sinusoidal embedding `[L, sinusoidal_dim]` for `timesteps: [L]`.
    ///
    /// Mirrors `TimestepEmbedder.timestep_embedding` exactly. `t` is read on
    /// host (after upcast to F32) and the entire cos/sin pipeline runs in F32
    /// on CPU, then the result is uploaded and cast to `out_dtype` for the
    /// downstream MLP. This is the same CPU-build pattern as Module 3's
    /// `precompute_mrope_tables`.
    ///
    /// The output cat order is `[cos, sin]` per Python `modeling_utils.py:138`.
    fn timestep_embedding(
        timesteps: &Tensor,
        sinusoidal_dim: usize,
        out_dtype: DType,
        dev: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        // Python: half = dim // 2; cat order is [cos, sin], so dim must be even.
        if sinusoidal_dim == 0 || sinusoidal_dim % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "TimeEmbedder::timestep_embedding: sinusoidal_dim must be positive and even, got {sinusoidal_dim}"
            )));
        }
        let dims = timesteps.shape().dims().to_vec();
        if dims.len() != 1 {
            return Err(Error::InvalidInput(format!(
                "TimeEmbedder::timestep_embedding: expected 1D [L] timesteps, got {dims:?}"
            )));
        }
        let l = dims[0];
        let half = sinusoidal_dim / 2;

        // Python: freqs = exp(-log(max_period=10000) * arange(0, half) / half)
        // Identical formula to mRoPE's inv_freq with theta=10000 and head_dim=2*half.
        let max_period: f32 = 10_000.0;
        let log_period = max_period.ln();
        let inv_half = 1.0f32 / (half as f32);
        let freqs: Vec<f32> = (0..half)
            .map(|j| (-log_period * (j as f32) * inv_half).exp())
            .collect();

        // Read `t` to host in F32 (Python casts `t.float()` before the multiply).
        // `to_dtype(F32).to_vec()` is the canonical pattern (used by mRoPE +
        // tests above) and bounded to L values — tiny payload.
        let t_f32: Vec<f32> = timesteps.to_dtype(DType::F32)?.to_vec()?;
        if t_f32.len() != l {
            return Err(Error::InvalidInput(format!(
                "TimeEmbedder::timestep_embedding: host vec len {} != L {}",
                t_f32.len(),
                l
            )));
        }

        // Build [L, sinusoidal_dim] = [L, 2*half] as `cat([cos(args), sin(args)], -1)`.
        // Row-major flat layout: row p holds the L_p embedding,
        //   row[0..half]                = cos(t_p * freqs[..])
        //   row[half..sinusoidal_dim]   = sin(t_p * freqs[..])
        let mut data: Vec<f32> = Vec::with_capacity(l * sinusoidal_dim);
        for p in 0..l {
            let tp = t_f32[p];
            // cos block first.
            for j in 0..half {
                data.push((tp * freqs[j]).cos());
            }
            // sin block second.
            for j in 0..half {
                data.push((tp * freqs[j]).sin());
            }
        }
        let emb_f32 = Tensor::from_vec(
            data,
            Shape::from_dims(&[l, sinusoidal_dim]),
            dev.clone(),
        )?;
        emb_f32.to_dtype(out_dtype)
    }

    /// Forward: `[L]` scalar timesteps → `[L, hidden_size]` per-token time emb.
    ///
    /// Input dtype is consumed as-is for the cos/sin build (host F32 upcast
    /// handles non-F32 inputs). The MLP pipeline runs in `cfg.dtype` (BF16 in
    /// production) end-to-end via `fused_linear3d_native`.
    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let dev = timesteps.device();
        // Step 1: sinusoidal embedding `[L]` -> `[L, sinusoidal_dim]`.
        // `mlp_0_weight.dtype()` is the canonical out-dtype: matches the
        // downstream Linears.
        let dt = self.mlp_0_weight.dtype();
        let emb = Self::timestep_embedding(timesteps, self.sinusoidal_dim, dt, &dev)?;

        // fused_linear3d_native consumes a 3D `[B, N, Cin]` input. Reshape to
        // `[1, L, sinusoidal_dim]` for the call, then squeeze back to `[L, hidden]`.
        // Reshape is a metadata-only op on a fresh contig buffer.
        let l = emb.shape().dims()[0];
        let emb3 = emb.reshape(&[1, l, self.sinusoidal_dim])?;

        // Step 2: Linear(256 -> 2048) + bias.
        let h0 = flame_core::ops::fused_inference::fused_linear3d_native(
            &emb3,
            &self.mlp_0_weight,
            Some(&self.mlp_0_bias),
        )?; // [1, L, 2048]

        // Step 3: SiLU. `Tensor::silu` dispatches to the canonical BF16/F32 op.
        let h0_act = h0.silu()?;

        // Step 4: Linear(2048 -> 2048) + bias.
        let h2 = flame_core::ops::fused_inference::fused_linear3d_native(
            &h0_act,
            &self.mlp_2_weight,
            Some(&self.mlp_2_bias),
        )?; // [1, L, 2048]

        // Squeeze the batch axis back: [1, L, hidden] -> [L, hidden].
        h2.reshape(&[l, self.hidden_size])
    }
}

// ===========================================================================
// C4 Module 12: `LatentPosEmbed` — sinusoidal position embedding (runtime)
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/modeling_utils.py:183-198`
// (`class PositionEmbedding3D`) + `modeling_utils.py:73-102`
// (`get_3d_sincos_pos_embed*`) + `modeling_utils.py:52-70`
// (`get_1d_sincos_pos_embed_from_grid`).
//
// **Python ground truth.** `PositionEmbedding3D` stores a precomputed
// `[max_num_latent_frames * max_latent_size**2, hidden_size]` buffer built
// by `get_3d_sincos_pos_embed(hidden_size, t=max_num_latent_frames,
// h=max_latent_size, w=max_latent_size)`. At runtime the buffer is indexed
// by `position_ids` (which Lance computes externally). The published
// state-dict ships this buffer at `latent_pos_embed.pos_embed [4096, 2048]
// F32`; the Python loader POPS it on load and recomputes from the
// `hidden_size + max_num_latent_frames + max_latent_size` config triple.
//
// **3D split arithmetic** (mirrors `get_3d_sincos_pos_embed_from_grid` lines 79-89):
//   d = embed_dim // 3                             # 2048 // 3 = 682
//   d = d if d % 2 == 0 else d - 1                 # 682 is even, stays 682
//   dim_t = d                                      # 682
//   dim_h = d                                      # 682
//   dim_w = embed_dim - 2 * d                      # 2048 - 1364 = 684
//   assert dim_w % 2 == 0                          # 684 % 2 == 0 ✓
//
//   emb_t = get_1d_sincos_pos_embed_from_grid(682, grid_t)  # [T*H*W, 682]
//   emb_h = get_1d_sincos_pos_embed_from_grid(682, grid_h)  # [T*H*W, 682]
//   emb_w = get_1d_sincos_pos_embed_from_grid(684, grid_w)  # [T*H*W, 684]
//   return concat([emb_t, emb_h, emb_w], 1)                  # [T*H*W, 2048]
//
// **1D piece** (`modeling_utils.py:52-70`):
//   omega = arange(D/2, f64) / (D/2)
//   omega = 1 / 10000**omega                                  # [D/2]
//   out = einsum('m,d->md', pos.reshape(-1), omega)           # [M, D/2]
//   emb = concat([sin(out), cos(out)], 1)                     # [M, D]
//   ↑ NOTE: 1D piece uses cat order [sin, cos] — DIFFERENT from
//   the TimeEmbedder, which uses [cos, sin]. This is correct per Python.
//
// **Position layout in flat row index** (`get_3d_sincos_pos_embed` lines 92-102):
//   tt, hh, ww = meshgrid(arange(T), arange(H), arange(W), indexing="ij")
//   flat row r is `(t, h, w)` with row-major (t outermost, w innermost):
//     r = t * H * W + h * W + w
//   So at flat index r,
//     grid_t[r] = r // (H*W)
//     grid_h[r] = (r // W) % H
//     grid_w[r] = r % W
//
// **Runtime-rebuild signature.** `build(num_t, num_h, num_w, dev, dtype) ->
// [num_t*num_h*num_w, hidden_size]`. Python recomputes the table on every
// `forward` for the active grid (`Lance/modeling/lance/lance.py:131` pops the
// stored buffer at load time; `modeling_utils.py:79-102` rebuilds per call).
// Rust mirrors that. The state-dict precompute `[4096, 2048]` is recoverable
// by calling `build(1, 64, 64, ..)`; production T2I resolutions need much
// larger tables (1024² → `build(1, 128, 128, ..)` → `[16384, 2048]`).
//
// **Open question deferred to Module 13.** True parity at inference also
// requires Lance's `vae_position_ids` index tensor (3D-flat indices for the
// active resolution). That belongs in the top-level forward, not this leaf.

/// `LatentPosEmbed`: sinusoidal 3D position embedding helper, computed at
/// runtime per resolution. No persistent parameters — mirrors Python
/// `PositionEmbedding3D` semantics where `pos_embed` is popped from the
/// state-dict on load and rebuilt from config.
///
/// **Runtime sizing.** Python recomputes the table on every `forward` for the
/// active `(num_t, num_h, num_w)` grid (see `Lance/modeling/lance/lance.py:131`
/// loader-pop + `modeling_utils.py:79-102`). Rust mirrors that: no precomputed
/// `max_*` ceiling, no first-N-rows slicing. Production resolutions need much
/// larger tables than the state-dict's `[4096, 2048]` precompute (e.g. 1024²
/// → 128² latent → 16384 rows; 2048² → 256² → 65536 rows), so a fixed ceiling
/// would silently corrupt or fault.
#[derive(Debug, Clone)]
pub struct LatentPosEmbed {
    /// 2048 — embedding dimension. Must match `cfg.hidden_size`.
    pub hidden_size: usize,
}

impl LatentPosEmbed {
    /// Construct a `LatentPosEmbed`. Only `hidden_size` is captured; the
    /// `(num_t, num_h, num_w)` grid is passed per `build` call.
    pub fn new(cfg: &LanceConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
        }
    }

    /// Compute the 1D sin-cos position embedding `[M, embed_dim]` per Python
    /// `get_1d_sincos_pos_embed_from_grid` (lines 52-70). Cat order is
    /// **[sin, cos]** — DIFFERENT from the time embedder's [cos, sin]. This is
    /// Python ground truth; do NOT "fix" the inconsistency.
    ///
    /// Computed on CPU in F32 (positions are `arange(0..M)`, embed dim ≤ 1024
    /// per axis in our config, so payload is tiny). Caller casts to target dtype.
    fn build_1d_sincos(positions: &[f32], embed_dim: usize, out: &mut Vec<f32>) {
        // omega = arange(D/2) / (D/2);  omega = 1 / 10000**omega
        assert!(embed_dim % 2 == 0);
        let half = embed_dim / 2;
        let inv_half = 1.0f32 / (half as f32);
        let omega: Vec<f32> = (0..half)
            .map(|j| (-(10_000.0f32.ln()) * (j as f32) * inv_half).exp())
            .collect();
        // For each position, write `sin(p*omega)` then `cos(p*omega)`.
        for &p in positions {
            for j in 0..half {
                out.push((p * omega[j]).sin());
            }
            for j in 0..half {
                out.push((p * omega[j]).cos());
            }
        }
    }

    /// Build the 3D sin-cos position embedding for the requested
    /// `(num_t, num_h, num_w)` latent grid. Output shape is
    /// `[num_t * num_h * num_w, hidden_size]` in `dtype`. Mirror of Python
    /// `PositionEmbedding3D.forward` + `get_3d_sincos_pos_embed`
    /// (`modeling_utils.py:79-102`).
    ///
    /// Axis-dim split (Python lines 79-83):
    ///   `d = embed_dim // 3` (made even); `dim_t = dim_h = d; dim_w = embed_dim - 2*d`.
    /// For `embed_dim=2048` this gives `dim_t=dim_h=682, dim_w=684`.
    ///
    /// Row order matches Python `meshgrid(..., indexing='ij')` row-major
    /// flatten: `r = t * (num_h * num_w) + h * num_w + w`.
    pub fn build(
        &self,
        num_t: usize,
        num_h: usize,
        num_w: usize,
        device: &Arc<CudaDevice>,
        dtype: DType,
    ) -> Result<Tensor> {
        if num_t == 0 || num_h == 0 || num_w == 0 {
            return Err(Error::InvalidInput(format!(
                "LatentPosEmbed::build: grid dims must be > 0, got ({num_t}, {num_h}, {num_w})"
            )));
        }
        let embed_dim = self.hidden_size;
        if embed_dim == 0 || embed_dim % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "LatentPosEmbed::build: hidden_size must be positive and even, got {embed_dim}"
            )));
        }
        let d_raw = embed_dim / 3;
        let d = if d_raw % 2 == 0 { d_raw } else { d_raw - 1 };
        let dim_t = d;
        let dim_h = d;
        let dim_w = embed_dim - 2 * d;
        if dim_w % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "LatentPosEmbed::build: dim_w {dim_w} must be even (embed_dim={embed_dim}, d={d})"
            )));
        }

        let m = num_t * num_h * num_w;

        // Python `meshgrid(grid_t, grid_h, grid_w, indexing='ij')` flattens
        // with `t` outermost (`r = t*H*W + h*W + w`). Build per-axis position
        // sequences of length M according to that layout.
        let mut pos_t: Vec<f32> = Vec::with_capacity(m);
        let mut pos_h: Vec<f32> = Vec::with_capacity(m);
        let mut pos_w: Vec<f32> = Vec::with_capacity(m);
        for t in 0..num_t {
            for h in 0..num_h {
                for w in 0..num_w {
                    pos_t.push(t as f32);
                    pos_h.push(h as f32);
                    pos_w.push(w as f32);
                }
            }
        }

        // Per-axis 1D embeddings concatenated row-wise into the final
        // `[M, embed_dim]` table per Python line 88-89:
        //   emb = concat([emb_t, emb_h, emb_w], dim=1)
        let mut emb_t: Vec<f32> = Vec::with_capacity(m * dim_t);
        let mut emb_h: Vec<f32> = Vec::with_capacity(m * dim_h);
        let mut emb_w: Vec<f32> = Vec::with_capacity(m * dim_w);
        Self::build_1d_sincos(&pos_t, dim_t, &mut emb_t);
        Self::build_1d_sincos(&pos_h, dim_h, &mut emb_h);
        Self::build_1d_sincos(&pos_w, dim_w, &mut emb_w);

        // Interleave into row-major [M, embed_dim] = [M, dim_t + dim_h + dim_w].
        let mut data: Vec<f32> = Vec::with_capacity(m * embed_dim);
        for r in 0..m {
            let s_t = r * dim_t;
            let s_h = r * dim_h;
            let s_w = r * dim_w;
            data.extend_from_slice(&emb_t[s_t..s_t + dim_t]);
            data.extend_from_slice(&emb_h[s_h..s_h + dim_h]);
            data.extend_from_slice(&emb_w[s_w..s_w + dim_w]);
        }

        let t_f32 = Tensor::from_vec(
            data,
            Shape::from_dims(&[m, embed_dim]),
            device.clone(),
        )?;
        t_f32.to_dtype(dtype)
    }
}

// ===========================================================================
// C4 Tests (Modules 10/11/12)
// ===========================================================================
//
// New tests for Modules 10/11/12. Lives in a separate `tests_c4` module so we
// don't collide with the C2/C3 test block (which another agent may be editing
// for the mRoPE pair-up bugfix). Reuses the `dev()` helper pattern from the
// existing test block via a local copy.
#[cfg(test)]
mod tests_c4 {
    use super::*;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device 0")
    }

    #[test]
    fn test_vae2llm_shape() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = Vae2Llm::new_random(&cfg).unwrap();
        let x = Tensor::randn(
            Shape::from_dims(&[1, 64, cfg.patch_latent_dim()]),
            0.0,
            0.02,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = m.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 64, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn test_llm2vae_shape() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = Llm2Vae::new_random(&cfg).unwrap();
        let x = Tensor::randn(
            Shape::from_dims(&[1, 64, cfg.hidden_size]),
            0.0,
            0.02,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = m.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 64, cfg.patch_latent_dim()]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn test_vae2llm_llm2vae_roundtrip_shape_preserved() {
        // Compose: x -> vae2llm -> llm2vae should preserve shape (not values).
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let vae2llm = Vae2Llm::new_random(&cfg).unwrap();
        let llm2vae = Llm2Vae::new_random(&cfg).unwrap();
        let x = Tensor::randn(
            Shape::from_dims(&[1, 64, cfg.patch_latent_dim()]),
            0.0,
            0.02,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let mid = vae2llm.forward(&x).unwrap();
        assert_eq!(mid.shape().dims(), &[1, 64, cfg.hidden_size]);
        let out = llm2vae.forward(&mid).unwrap();
        assert_eq!(out.shape().dims(), &[1, 64, cfg.patch_latent_dim()]);
    }

    #[test]
    fn test_patch_latent_dim_matches_state_dict() {
        // Pin the resolved shell-config value: 1 * 1 * 1 * 48 = 48,
        // matching `vae2llm.weight=[2048, 48]` in `Lance_3B/model.safetensors`.
        let d = dev();
        let cfg = LanceConfig::default_3b(d);
        assert_eq!(cfg.patch_latent_dim(), 48);
    }

    #[test]
    fn test_time_embedder_shape() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = TimeEmbedder::new_random(&cfg).unwrap();
        // L=4 timesteps. Use floats in [0, 1] (flow-matching range).
        let timesteps = Tensor::from_vec(
            vec![0.0f32, 0.25, 0.5, 1.0],
            Shape::from_dims(&[4]),
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = m.forward(&timesteps).unwrap();
        assert_eq!(out.shape().dims(), &[4, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    /// At `t == 0`, `cos(0 * freq) == 1` and `sin(0 * freq) == 0` for every
    /// frequency. With Python's `cat([cos, sin], -1)` order, the sinusoidal
    /// row is `[1, 1, ..., 1, 0, 0, ..., 0]` — first half ones, second half
    /// zeros. We verify this BEFORE the MLP by computing it directly.
    #[test]
    fn test_time_embedder_sinusoidal_t_zero_cos_first() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = TimeEmbedder::new_random(&cfg).unwrap();
        // Single timestep t=0.
        let t = Tensor::from_vec(vec![0.0f32], Shape::from_dims(&[1]), d.clone())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let emb = TimeEmbedder::timestep_embedding(&t, m.sinusoidal_dim, DType::F32, &d).unwrap();
        assert_eq!(emb.shape().dims(), &[1, m.sinusoidal_dim]);
        let v = emb.to_vec().unwrap();
        let half = m.sinusoidal_dim / 2;
        // First half: cos(0 * f) = 1.
        for j in 0..half {
            assert!(
                (v[j] - 1.0).abs() < 1e-6,
                "time emb cos block at t=0, j={j}: expected 1.0 got {}",
                v[j]
            );
        }
        // Second half: sin(0 * f) = 0.
        for j in half..m.sinusoidal_dim {
            assert!(
                v[j].abs() < 1e-6,
                "time emb sin block at t=0, j={j}: expected 0.0 got {}",
                v[j]
            );
        }
    }

    #[test]
    fn test_latent_pos_embed_shape() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = LatentPosEmbed::new(&cfg);
        // 1 * 8 * 8 = 64 rows for shape parity with the original assertion.
        let pos = m.build(1, 8, 8, &d, DType::BF16).unwrap();
        assert_eq!(pos.shape().dims(), &[64, cfg.hidden_size]);
        assert_eq!(pos.dtype(), DType::BF16);
    }

    /// At flat position 0 the 3D coords are `(t=0, h=0, w=0)`. For every axis
    /// the 1D sin-cos at pos=0 is `[sin(0)=0 ..., cos(0)=1 ...]` — first half
    /// zeros, second half ones. Concatenating three such rows along the
    /// channel dim yields a row whose first `dim_t/2` entries are zeros, next
    /// `dim_t/2` ones, etc. We assert the overall pattern.
    #[test]
    fn test_latent_pos_embed_pos_zero_is_zeros_and_ones() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = LatentPosEmbed::new(&cfg);
        // 1 * 2 * 2 = 4 rows; row 0 is grid pos (t=0, h=0, w=0).
        let pos = m.build(1, 2, 2, &d, DType::F32).unwrap();
        assert_eq!(pos.shape().dims(), &[4, cfg.hidden_size]);
        let v = pos.to_vec().unwrap();

        // Recompute the 3D dim split to know where each axis block lives.
        let embed_dim = cfg.hidden_size;
        let d_raw = embed_dim / 3;
        let d_even = if d_raw % 2 == 0 { d_raw } else { d_raw - 1 };
        let dim_t = d_even;
        let dim_h = d_even;
        let dim_w = embed_dim - 2 * d_even;
        assert_eq!(dim_t + dim_h + dim_w, embed_dim);

        // Row 0 = position (t=0, h=0, w=0). Each axis block: first half = sin(0) = 0,
        // second half = cos(0) = 1.
        let row0 = &v[0..embed_dim];
        let block_t = &row0[0..dim_t];
        let block_h = &row0[dim_t..dim_t + dim_h];
        let block_w = &row0[dim_t + dim_h..];
        for (name, block) in &[("t", block_t), ("h", block_h), ("w", block_w)] {
            let half = block.len() / 2;
            for j in 0..half {
                assert!(
                    block[j].abs() < 1e-6,
                    "row 0 axis {name} sin half at j={j}: expected 0 got {}",
                    block[j]
                );
            }
            for j in half..block.len() {
                assert!(
                    (block[j] - 1.0).abs() < 1e-6,
                    "row 0 axis {name} cos half at j={j}: expected 1 got {}",
                    block[j]
                );
            }
        }
    }

    /// State-dict reproduction smoke: `build(1, 64, 64, ...)` produces the
    /// `[4096, 2048]` row count + dim count that the published checkpoint
    /// ships at `latent_pos_embed.pos_embed`. We can't compare to the actual
    /// stored values here (no checkpoint at random-init test time), but the
    /// shape match plus the row-0 sin/cos assertion above pin enough of the
    /// math that real-weight parity in C6 will catch any residual divergence.
    #[test]
    fn test_latent_pos_embed_full_state_dict_size() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = LatentPosEmbed::new(&cfg);
        assert_eq!(m.hidden_size, 2048);
        let pos = m.build(1, 64, 64, &d, DType::F32).unwrap();
        assert_eq!(pos.shape().dims(), &[4096, 2048]);
    }

    /// 1024² T2I target: latent 128×128 with T=1. Pre-fix code capped at
    /// `max_seq_len = 4096` and would have errored or returned a wrong-sized
    /// view; runtime-rebuild must produce the full `[16384, 2048]`.
    #[test]
    fn test_latent_pos_embed_runtime_resolution_1024() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let m = LatentPosEmbed::new(&cfg);
        let pos = m.build(1, 128, 128, &d, DType::BF16).unwrap();
        assert_eq!(pos.shape().dims(), &[16384, 2048]);
        assert_eq!(pos.dtype(), DType::BF16);
    }
}

// ===========================================================================
// C4 Module 13: `Lance` — top-level T2I forward (struct + prefill + gen_step)
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/lance.py`
//   * `Lance.__init__`                 (lines  78-160) — module composition
//   * `Lance.init_gen_context`         (lines 1380-1386) — empty KvCache
//   * `Lance.update_gen_context`       (lines 1886-1918) — PREFILL (text → cache)
//   * `Lance.validation_gen_KVcache`   (lines 1660-1769) — GEN step (latent → v_pred)
//
// **Scope (Module 13 of C4).** This wires the prior leaves together into
// one struct + four methods. NOT in scope: real weight loader (C6), the
// 30-step denoise loop with CFG (C5), the CLI bin (C7).
//
// **Design constraints carried from C1/C2/C3 audits.**
//   - `LanceBlockStack::forward` consumes 3D `[B, N, hidden]`. We reshape
//     `[L, hidden]` → `[1, L, hidden]` at the prefill/gen boundary, then
//     reshape back. Reshape on a fresh contig buffer is metadata-only.
//   - `Qwen2MoTAttention::forward` (and `KvCache`) require `gen_mask`
//     `[B, N]` and `pos_indices_thw [t,h,w]`. Per-token mRoPE positions
//     are NOT yet supported by `apply_mrope` (single triplet per call) —
//     Module 13 uses a fixed `[0,0,0]` placeholder and documents the
//     limitation. C5 will replace with per-token mRoPE when the kernel
//     gains that ability.
//   - Cache routing: PREFILL writes (`update_cache=true`, `is_causal=true`,
//     `gen_mask=all_und=0`); GEN step reads-only (`update_cache=false`,
//     `is_causal=false`, `gen_mask=all_gen=1`). This mirrors Python at
//     `lance.py:1676` (`update_past_key_values=True`, `is_causal=is_causal`)
//     and `lance.py:1716-1727` (`update_past_key_values=False`,
//     `is_causal=False`).
//
// **Patchify recipe.** Shell-config `latent_patch_size=(1,1,1)` ⇒
// `pt=ph=pw=1`. Python `lance.py:1539`:
//
//     patches = rearrange(padded_latent,
//         "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)",
//         t=t, pt=pt, h=h, ph=ph, w=w, pw=pw)
//
// Note Python's input layout there is `(T, H, W, C)` (channel-last). Our
// trainer-style input is `[B, C, T, H, W]` (channel-second, batch-first).
// For `pt=ph=pw=1` the rearrange degenerates to a permute+flatten:
//   `[B, 48, T, H, W]` → permute(0, 2, 3, 4, 1) → `[B, T, H, W, 48]`
//                      → reshape `[B*T*H*W, 48]`
// (For non-trivial patch sizes a true space-to-depth would be needed;
// Module 13 documents that gap rather than papering over it.)
//
// **Unpatchify** is the strict reverse: `[B*T*H*W, 48]` → reshape
// `[B, T, H, W, 48]` → permute(0, 4, 1, 2, 3) → `[B, 48, T, H, W]`.
//
// **embed_tokens lookup.** flame-core has `Tensor::index_select0(ids)`
// against a 2D `[V, D]` table with **I32** ids. Token ids enter as a 1D
// integer tensor; we accept any dtype on the caller side and cast to I32
// at the boundary (mirrors Python's `LongTensor` which we don't have).
// Output is `[L, D]` matching the table row dim, exactly the embedding
// table contract.

/// Top-level Lance T2I model. Composes the C1-C4 leaves into a runnable
/// (random-weight) forward.
///
/// **Production loader is C6.** `Lance::new_random` is for shape tests +
/// downstream code that wants a typed handle without a checkpoint on disk.
#[derive(Debug)]
pub struct Lance {
    /// Captured config (held as `Arc` so `Lance` can be cheaply cloned).
    pub config: Arc<LanceConfig>,

    /// Text/vision token embedding table.
    /// Shape `[vocab_size=151936, hidden=2048]` in `cfg.dtype` (BF16).
    /// State-dict key: `language_model.model.embed_tokens.weight`.
    pub embed_tokens: Tensor,

    /// `vae2llm`: `[B, L_image, patch_latent_dim] → [B, L_image, hidden]`.
    pub vae2llm: Vae2Llm,

    /// `llm2vae`: `[B, L_image, hidden] → [B, L_image, patch_latent_dim]`.
    pub llm2vae: Llm2Vae,

    /// Time embedder: `[L] → [L, hidden]` per-token additive time embedding.
    pub time_embedder: TimeEmbedder,

    /// 3D sinusoidal latent position embedder (runtime-rebuilt per grid).
    pub latent_pos_embed: LatentPosEmbed,

    /// 36 decoder layers (mixed Base/MoT per config) + paired final norm.
    pub blocks: LanceBlockStack,
}

impl Lance {
    /// Random-init constructor for shape tests. Wires up the C1-C4 leaves
    /// using each leaf's `new_random` plus a random embedding table.
    ///
    /// **DOES NOT load real weights.** C6 (separate spawn) is the production
    /// loader path against `Lance_3B/model.safetensors`.
    pub fn new_random(cfg: Arc<LanceConfig>) -> Result<Self> {
        let dev = &cfg.device;
        let dt = cfg.dtype;

        // Embedding table `[vocab, hidden]`. Use the same `randn_dt` helper
        // pattern the leaf modules use so seeds are deterministic across
        // test runs. Seed 4101 is a fresh range; std=0.02 matches Qwen2
        // default init.
        let embed_tokens =
            Qwen2Attention::randn_dt(&[cfg.vocab_size, cfg.hidden_size], 0.02, 4101, dev, dt)?;

        let vae2llm = Vae2Llm::new_random(&cfg)?;
        let llm2vae = Llm2Vae::new_random(&cfg)?;
        let time_embedder = TimeEmbedder::new_random(&cfg)?;
        let latent_pos_embed = LatentPosEmbed::new(&cfg);
        let blocks = LanceBlockStack::new_random(&cfg)?;

        Ok(Self {
            config: cfg,
            embed_tokens,
            vae2llm,
            llm2vae,
            time_embedder,
            latent_pos_embed,
            blocks,
        })
    }

    /// Build a fresh (empty) KV cache sized for this model's layer count.
    pub fn new_kv_cache(&self) -> KvCache {
        KvCache::new(self.config.num_hidden_layers)
    }

    /// Precompute mRoPE cos/sin tables for a given `max_pos`. Caller is
    /// expected to reuse this across multiple forward calls at the same
    /// resolution. `dtype` should typically be the active model dtype
    /// (BF16 in production).
    pub fn precompute_mrope(&self, max_pos: usize, dtype: DType) -> Result<MropeFreqs> {
        precompute_mrope_tables(
            self.config.mrope_sections,
            self.config.rope_theta,
            max_pos,
            self.config.head_dim,
            &self.config.device,
            dtype,
        )
    }

    /// Internal: embed a 1D tensor of token ids `[L]` (any integer-ish
    /// dtype) into `[L, hidden]` via `index_select0`. The table requires
    /// I32 ids per `Tensor::index_select0` contract; we cast at the
    /// boundary.
    fn embed_text_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        let dims = token_ids.shape().dims();
        if dims.len() != 1 {
            return Err(Error::InvalidInput(format!(
                "Lance::embed_text_tokens: expected 1D [L] token_ids, got {dims:?}"
            )));
        }
        // `index_select0` requires I32. Cast at the boundary — `to_dtype`
        // is a kernel-level conversion when the input is already integer-
        // ish and a clean truncation otherwise.
        let ids_i32 = if token_ids.dtype() == DType::I32 {
            token_ids.clone()
        } else {
            token_ids.to_dtype(DType::I32)?
        };
        // `[V, hidden]` table + `[L]` ids → `[L, hidden]`.
        self.embed_tokens.index_select0(&ids_i32)
    }

    /// Reshape a 2D `[L, hidden]` into a 3D `[1, L, hidden]` for the block
    /// stack (which requires `[B, N, hidden]`). Reshape on a contig buffer
    /// is metadata-only.
    fn to_3d(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        if dims.len() != 2 {
            return Err(Error::InvalidInput(format!(
                "Lance::to_3d: expected 2D [L, hidden], got {dims:?}"
            )));
        }
        x.reshape(&[1, dims[0], dims[1]])
    }

    /// Build a `[B=1, N]` gen-mask filled with a single value (0.0 = und,
    /// 1.0 = gen). Dtype matches the model (BF16). Used by both prefill
    /// (all-und = 0) and gen_step (all-gen = 1).
    fn build_uniform_gen_mask(&self, n: usize, value: f32) -> Result<Tensor> {
        let dev = &self.config.device;
        let dt = self.config.dtype;
        // `from_vec` + `to_dtype` is the canonical pattern used by the
        // test helper `make_mask` at line ~2362. Keep aligned.
        let raw = vec![value; n];
        let t = Tensor::from_vec(raw, Shape::from_dims(&[1, n]), dev.clone())?;
        if t.dtype() != dt { t.to_dtype(dt) } else { Ok(t) }
    }

    /// **PREFILL.** Embed text token ids, run them through the 36-block
    /// stack with `is_causal=true, gen_mask=all_und=0, update_cache=true`,
    /// and store per-layer K/V into `cache`.
    ///
    /// Mirror of Python `lance.py:1886-1918` (`update_gen_context`) — that
    /// function feeds `current_sequence[current_cond_start:current_cond_end]`
    /// (text or vit embeddings) into `language_model.forward_inference`
    /// with `update_past_key_values=True, is_causal=is_causal, mode="und"`.
    /// Here we cover the pure-text-prefill case (no VIT) since C7 T2I only
    /// needs text context.
    ///
    /// **Returns `()`** — the cache itself is the result. The final hidden
    /// state for the LM-head is NOT needed for T2I (`lm_head` is the
    /// understanding path; image gen reads the gen step's `llm2vae`
    /// output instead).
    ///
    /// **Per-token mRoPE positions (2026-05-18):** for a pure-text prefill
    /// (no image/vision tokens in the prompt), Python `get_rope_index`
    /// (`qwen2_navit.py:1183-1297`) hits the "else" branch at lines
    /// 1278-1297 and produces `position_ids = arange(L).view(1,1,-1).expand(3, -1, -1)`
    /// — i.e. `(t=p, h=p, w=p)` for `p = 0..L_text` along all three axes.
    /// That is the "Examples: pure text" diagonal documented at
    /// `qwen2_navit.py:1136-1141`. We mirror that here: `pos_t = pos_h =
    /// pos_w = arange(L_text)` as I32.
    ///
    /// We DO NOT apply Python's `shift_position_ids(..., pos_shift=1000)`
    /// (`lance.py:1621`) here — that shift is on the image-gen-token range
    /// of the position ids returned by `get_rope_index`, not the text
    /// range. Text positions stay at 0..L_text; image positions get
    /// shifted by 1000 inside `gen_step`. The two ranges are disjoint by
    /// design so attention can distinguish reference-image-feature vs
    /// noise-latent vs text tokens via mRoPE alone.
    pub fn prefill_text_context(
        &self,
        text_token_ids: &Tensor,
        mrope: &MropeFreqs,
        cache: &mut KvCache,
    ) -> Result<()> {
        let _ = self.prefill_text_context_capture(text_token_ids, mrope, cache)?;
        Ok(())
    }

    /// Same as `prefill_text_context` but returns the final post-norm hidden
    /// state `[1, L_text, hidden]`. Useful for parity probes — the standard
    /// `prefill_text_context` discards the output because production callers
    /// only need the side-effect on the KV cache.
    pub fn prefill_text_context_capture(
        &self,
        text_token_ids: &Tensor,
        mrope: &MropeFreqs,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let h2d = self.embed_text_tokens(text_token_ids)?;
        let l_text = h2d.shape().dims()[0];
        let h3d = Self::to_3d(&h2d)?;
        let gen_mask = self.build_uniform_gen_mask(l_text, 0.0)?;
        let (pos_t, pos_h, pos_w) = self.build_text_prefill_positions(l_text)?;
        self.blocks.forward(
            &h3d,
            &gen_mask,
            &pos_t,
            &pos_h,
            &pos_w,
            mrope,
            true,
            Some(cache),
            true,
        )
    }

    /// Build the diagonal per-axis position tensors for pure-text prefill.
    ///
    /// Returns three `[L]` I32 tensors, each holding `0..L` along their
    /// respective axis. Mirrors Python `qwen2_navit.py:1286-1290`:
    /// `arange(L).view(1,1,-1).expand(3, B=1, -1)`.
    fn build_text_prefill_positions(
        &self,
        l_text: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = &self.config.device;
        let vals: Vec<f32> = (0..l_text).map(|i| i as f32).collect();
        // F32 → I32 storage-relabel: same idiom as `embed_text_tokens`. For
        // L_text < 2^24 the integer-valued F32 round-trips bit-exact through
        // the gather's `static_cast<int>`.
        let t = Tensor::from_vec(vals.clone(), Shape::from_dims(&[l_text]), dev.clone())?
            .to_dtype(DType::I32)?;
        let h = Tensor::from_vec(vals.clone(), Shape::from_dims(&[l_text]), dev.clone())?
            .to_dtype(DType::I32)?;
        let w = Tensor::from_vec(vals, Shape::from_dims(&[l_text]), dev.clone())?
            .to_dtype(DType::I32)?;
        Ok((t, h, w))
    }

    /// Build the per-axis position tensors for image-gen tokens enumerating
    /// the `(T, H, W)` latent grid.
    ///
    /// Per Python `qwen2_navit.py:1255-1265`:
    ///   `t_index = arange(T).view(-1,1).expand(-1, H*W).flatten()`
    ///   `h_index = arange(H).view(1,-1,1).expand(T,-1,W).flatten()`
    ///   `w_index = arange(W).view(1,1,-1).expand(T,H,-1).flatten()`
    ///
    /// Each axis is then offset by `offset` (the equivalent of Python's
    /// `text_len + st_idx` PLUS `shift_position_ids(pos_shift=1000)`). All
    /// three axes share the SAME offset (Python adds `text_len + st_idx`
    /// uniformly to the stacked `[t_index, h_index, w_index]` at
    /// `qwen2_navit.py:1265`).
    ///
    /// Returns three `[T*H*W]` I32 tensors in (T-outer, H-middle, W-inner)
    /// row-major order so position `i` corresponds to image-grid coord
    /// `(t = i / (H*W), h = (i / W) % H, w = i % W)`. This matches the
    /// patchify layout produced in step 1 of `gen_step`:
    /// `permute(0,2,3,4,1).reshape([L_image, p])` enumerates row-major
    /// over `[T, H, W]` with W innermost.
    fn build_image_gen_positions(
        &self,
        t: usize,
        h: usize,
        w: usize,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = &self.config.device;
        let l_image = t * h * w;
        let off_f32 = offset as f32;

        // T-axis multiplier per Qwen2.5-VL get_rope_index
        // (`qwen2_5_vl/modeling_qwen2_5_vl.py`):
        //   time_tensor = expanded_range * second_per_grid_t * tokens_per_second
        // Lance video config (`Lance_3B_Video/llm_config.json:vision_config`):
        //   tokens_per_second = 2, temporal_patch_size = 2
        // Lance inference path (`lance.py:245,612`): second_per_grid_ts = [1.0]
        // → T positions are 0, 2, 4, ... for T-axis (vs spatial axes 0, 1, 2).
        // For T2I (T=1) the multiplier is invisible; only T2V cares.
        const T_AXIS_MULTIPLIER: f32 = 2.0; // tokens_per_second * second_per_grid_t

        let mut t_vec: Vec<f32> = Vec::with_capacity(l_image);
        let mut h_vec: Vec<f32> = Vec::with_capacity(l_image);
        let mut w_vec: Vec<f32> = Vec::with_capacity(l_image);
        for it in 0..t {
            for ih in 0..h {
                for iw in 0..w {
                    t_vec.push(off_f32 + (it as f32) * T_AXIS_MULTIPLIER);
                    h_vec.push(off_f32 + ih as f32);
                    w_vec.push(off_f32 + iw as f32);
                }
            }
        }

        let t_tensor = Tensor::from_vec(t_vec, Shape::from_dims(&[l_image]), dev.clone())?
            .to_dtype(DType::I32)?;
        let h_tensor = Tensor::from_vec(h_vec, Shape::from_dims(&[l_image]), dev.clone())?
            .to_dtype(DType::I32)?;
        let w_tensor = Tensor::from_vec(w_vec, Shape::from_dims(&[l_image]), dev.clone())?
            .to_dtype(DType::I32)?;
        Ok((t_tensor, h_tensor, w_tensor))
    }

    /// **GEN STEP.** Given (a) the current noisy latent at timestep `t`
    /// and (b) a prefilled text-context KV cache, produce the next-step
    /// velocity prediction `v_pred` in patch-latent space (same shape as
    /// the input latent).
    ///
    /// Pipeline (mirror of `lance.py:1699-1729`):
    ///   1. Patchify latent  `[B, 48, T, H, W]` → `[L_image, 48]`
    ///      (`L_image = B*T*H*W` since `latent_patch_size=(1,1,1)`).
    ///   2. `vae2llm`            `[1, L_image, 48] → [1, L_image, hidden]`
    ///   3. `time_embedder(t)`   `[L_image] → [L_image, hidden]` additive
    ///   4. `latent_pos_embed`   `[L_image, hidden]` additive (per resolution)
    ///   5. Build all-gen mask `[1, L_image]` (image-gen tokens).
    ///   6. Block stack forward with `is_causal=false, update_cache=false,
    ///      cache=Some(&mut cache)` — the prefilled prefix is READ but
    ///      not WRITTEN (mirror Python `update_past_key_values=False`).
    ///   7. `llm2vae`            `[1, L_image, hidden] → [1, L_image, 48]`
    ///   8. Unpatchify back      `[1, L_image, 48] → [B, 48, T, H, W]`
    ///
    /// **Per-token mRoPE positions (2026-05-18):** image-gen tokens
    /// enumerate the 3D latent grid per Python `qwen2_navit.py:1255-1265`:
    /// `t_index = arange(T).view(-1,1).expand(-1, H*W).flatten()`,
    /// `h_index = arange(H).view(1,-1,1).expand(T,-1,W).flatten()`,
    /// `w_index = arange(W).view(1,1,-1).expand(T,H,-1).flatten()`.
    /// For our shell-config `latent_patch_size=(1,1,1)` the grid `(T, H,
    /// W)` matches the latent dims directly. Each axis tensor is then
    /// shifted by `text_len + st_idx`, and Python further applies
    /// `shift_position_ids(pos_shift=1000)` (`lance.py:1621`) on the
    /// image span to keep image positions disjoint from text.
    ///
    /// We use `t_offset = cache.seq_len(0) + 1000` to mirror BOTH the
    /// `text_len + st_idx` offset (covered by `cache.seq_len(0)` — the
    /// number of text tokens already in the cache for layer 0) AND the
    /// 1000-shift. If the cache is empty (no prefill), `t_offset = 1000`
    /// — that still places image tokens in the shifted range matching
    /// Python's pos_shift discipline. The per-axis position values are
    /// then `(t_offset + t, t_offset + h, t_offset + w)` per token,
    /// **with the same offset added to all three axes** (Python
    /// `qwen2_navit.py:1265` adds `text_len + st_idx` uniformly to the
    /// stacked `[t_index, h_index, w_index]`).
    ///
    /// **Limitations (carried forward to C5):**
    ///   - `timestep` is broadcast to `[L_image]` by repetition — this
    ///     mirrors Python's `torch.zeros(x_t.shape[0]).fill_(timestep_)`
    ///     at `lance.py:1691`.
    pub fn gen_step(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        mrope: &MropeFreqs,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        // ---- Validate input latent shape: `[B, C=patch_latent_dim, T, H, W]`. ----
        let ldims = latent.shape().dims().to_vec();
        if ldims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step: expected 5D latent [B, C, T, H, W], got {ldims:?}"
            )));
        }
        let (b, c, t, h, w) = (ldims[0], ldims[1], ldims[2], ldims[3], ldims[4]);
        let p = self.config.patch_latent_dim();
        if c != p {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step: latent C={c} != patch_latent_dim={p}"
            )));
        }
        // For shell-config `latent_patch_size=(1,1,1)` the patchify
        // collapses to a permute+flatten (see file-level recipe).
        let (pt, ph, pw) = self.config.latent_patch_size;
        if (pt, ph, pw) != (1, 1, 1) {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step: only latent_patch_size=(1,1,1) supported in Module 13, got ({pt},{ph},{pw})"
            )));
        }

        // ---- 1. Patchify: [B, 48, T, H, W] → [B*T*H*W, 48]. ----
        //
        // permute(0, 2, 3, 4, 1) → [B, T, H, W, 48], then reshape to
        // [B*T*H*W, 48]. Permute over a contig input + reshape is the
        // standard space-to-depth-when-patch=1 idiom.
        let l_image = b * t * h * w;
        let latent_p = latent.permute(&[0, 2, 3, 4, 1])?; // [B, T, H, W, 48]
        // permute output isn't guaranteed contig; reshape may require a
        // contig copy. flame-core handles this internally via the
        // reshape impl. (Considered `.contiguous()` here but didn't —
        // see "spots considered" in the report.)
        let x_patch_2d = latent_p.reshape(&[l_image, p])?; // [L_image, 48]
        let x_patch_3d = x_patch_2d.reshape(&[1, l_image, p])?; // [1, L_image, 48]

        // ---- 2. vae2llm: [1, L_image, 48] → [1, L_image, hidden]. ----
        let mut h_seq = self.vae2llm.forward(&x_patch_3d)?;

        // ---- 3. Time embedding: [L_image] → [L_image, hidden]. ----
        //
        // Python `lance.py:1686, 1691`: `timestep = torch.zeros(x_t.shape[0]).fill_(timestep_)`.
        // Our caller passes a scalar `[1]` timestep; broadcast to `[L_image]`
        // by replicating the single value (cheap on host since L_image ≤ ~10K
        // for typical resolutions). For multi-token timesteps (i2i with
        // vae-condition mask), Module 13 documents the gap — C5 will
        // accept per-token timesteps.
        let t_dims = timestep.shape().dims();
        if t_dims.len() != 1 || t_dims[0] != 1 {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step: expected scalar timestep [1] (Module 13 limitation), got {t_dims:?}"
            )));
        }
        // Read the scalar, build a length-L vec, push back as a Tensor. This
        // is the same host-build pattern Module 11 uses for sinusoidal embed.
        let t_scalar = timestep.to_dtype(DType::F32)?.to_vec()?[0];
        let t_vec = vec![t_scalar; l_image];
        let t_tensor_f32 = Tensor::from_vec(
            t_vec,
            Shape::from_dims(&[l_image]),
            self.config.device.clone(),
        )?;
        let t_tensor = if t_tensor_f32.dtype() != self.config.dtype {
            t_tensor_f32.to_dtype(self.config.dtype)?
        } else {
            t_tensor_f32
        };
        let time_emb = self.time_embedder.forward(&t_tensor)?; // [L_image, hidden]

        // Add time embedding (broadcast [L_image, hidden] into [1, L_image, hidden]).
        let time_emb_3d = Self::to_3d(&time_emb)?;
        h_seq = h_seq.add(&time_emb_3d)?;

        // ---- 4. Latent pos embedding: [L_image, hidden] additive. ----
        let pos_emb = self
            .latent_pos_embed
            .build(t, h, w, &self.config.device, self.config.dtype)?; // [T*H*W, hidden]
        // For B=1 (the only supported batch in Module 13), L_image = T*H*W.
        // Multi-batch would need a different pos-embed broadcast; we
        // validate up-front to keep that explicit.
        if b != 1 {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step: only B=1 supported in Module 13, got B={b}"
            )));
        }
        let pos_emb_3d = Self::to_3d(&pos_emb)?;
        h_seq = h_seq.add(&pos_emb_3d)?;

        // ---- 5. Gen-mask: all 1.0 (every token routes to the gen path). ----
        let gen_mask = self.build_uniform_gen_mask(l_image, 1.0)?;

        // ---- 5b. Per-token mRoPE positions for the image-gen tokens. ----
        // Mirror Python `qwen2_navit.py:1255-1265` (3D latent grid enumeration)
        // + the disjoint-shift discipline of `shift_position_ids(pos_shift=1000)`
        // at `lance.py:1621`. `t_offset` includes the cached text-prefix length
        // (option (b) from the task spec: read from `cache.seq_len(0)`) AND the
        // 1000-shift, so text mRoPE positions in 0..L_text and image mRoPE
        // positions in 1000+ ... stay fully disjoint.
        //
        // Iteration order matches the patchify layout (T-outer, H-middle,
        // W-inner) so the i-th row of `pos_*` corresponds to the i-th token
        // of `h_seq` (which is `(t h w) (c)`-ordered per step 1's
        // `permute(0,2,3,4,1).reshape([L_image, p])`).
        let prefix_len = cache.seq_len(0);
        let t_offset: usize = prefix_len + 1000;
        let (pos_t_tensor, pos_h_tensor, pos_w_tensor) =
            self.build_image_gen_positions(t, h, w, t_offset)?;

        // ---- 6. Block stack forward. is_causal=false, update_cache=false. ----
        let h_out_3d = self.blocks.forward(
            &h_seq,
            &gen_mask,
            &pos_t_tensor,
            &pos_h_tensor,
            &pos_w_tensor,
            mrope,
            false,           // is_causal — VAE-denoise is non-causal
            Some(cache),
            false,           // update_cache — READ-ONLY (mirror Python L1724)
        )?;

        // ---- 7. llm2vae: [1, L_image, hidden] → [1, L_image, 48]. ----
        let v_2d_3d = self.llm2vae.forward(&h_out_3d)?; // [1, L_image, 48]
        let v_flat = v_2d_3d.reshape(&[l_image, p])?; // [L_image, 48]

        // ---- 8. Unpatchify: [L_image, 48] → [B, 48, T, H, W]. ----
        // Reverse of step 1: reshape → permute back.
        let v_5d_thwc = v_flat.reshape(&[b, t, h, w, p])?; // [B, T, H, W, 48]
        v_5d_thwc.permute(&[0, 4, 1, 2, 3]) // [B, 48, T, H, W]
    }

    // =======================================================================
    // T2V denoise path (G1+G2+G4 from T2V_DENOISE_PORT.md)
    // =======================================================================
    //
    // Lance Python feeds a 770-token "noise span" `[SOI, vae×768, EOI]` into
    // `forward_inference` per denoise step, with positions in the SAME range
    // as the text prefix (no +1000 shift — that shift only applies for T2I
    // per `shift_position_ids(pro_type=10)`). See parity captures verified
    // 2026-05-19:
    //   - SOI = 151652, EOI = 151653
    //   - SOI at noise-span idx 0 with diagonal position (L_text, L_text, L_text)
    //   - VAE token (t, h, w) at noise-span idx `1 + t*H*W + h*W + w`
    //     with position (L_text+1+2t, L_text+1+h, L_text+1+w)
    //   - EOI at noise-span idx L_vae+1 with diagonal position
    //     (M+1, M+1, M+1) where M = max VAE position across all 3 axes

    /// Lance video bracket token ids (verified against `input_text_tokens`
    /// capture at packed indices 65 and 835):
    ///   `vision_start` = 151652, `vision_end` = 151653.
    /// Tokenizer name in Qwen2.5-VL: `<|vision_start|>`, `<|vision_end|>`.
    const VISION_START_TOKEN_ID: i32 = 151652;
    const VISION_END_TOKEN_ID: i32 = 151653;

    /// Build per-token mRoPE positions for the 770-token noise span used in
    /// T2V denoise. Returns three `[L_noise]` I32 tensors in noise-span
    /// order: `[SOI, vae(t=0,h=0,w=0), ..., vae(T-1,H-1,W-1), EOI]`.
    ///
    /// Verified against `current_pos_ids_post_shift.safetensors` (Python
    /// parity capture, 2026-05-19):
    ///   - SOI at idx 0: position `(L_text, L_text, L_text)`
    ///   - VAE at idx `1+t*H*W+h*W+w`: position
    ///     `(L_text+1+2t, L_text+1+h, L_text+1+w)`
    ///   - EOI at idx L_vae+1: position `(M+1, M+1, M+1)` where M is the
    ///     global max over all three axes of VAE positions, i.e.
    ///     `M = L_text + max(1+2*(T-1), H, W)`.
    fn build_t2v_noise_span_positions(
        &self,
        text_len: usize,
        t_lat: usize,
        h_lat: usize,
        w_lat: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = &self.config.device;
        let l_vae = t_lat * h_lat * w_lat;
        let l_noise = l_vae + 2;
        let mut pos_t: Vec<f32> = Vec::with_capacity(l_noise);
        let mut pos_h: Vec<f32> = Vec::with_capacity(l_noise);
        let mut pos_w: Vec<f32> = Vec::with_capacity(l_noise);

        // SOI: diagonal at L_text (continues from text prefix).
        let l_text_f = text_len as f32;
        pos_t.push(l_text_f);
        pos_h.push(l_text_f);
        pos_w.push(l_text_f);

        // VAE: T-major H-mid W-fast enumeration matching the patchify layout
        // `permute(0,2,3,4,1).reshape([L_vae, p])`. T-axis multiplier is 2.0
        // per Qwen2.5-VL get_rope_index (tokens_per_second=2 *
        // second_per_grid_t=1.0).
        const T_AXIS_MULTIPLIER: f32 = 2.0;
        let base_f = (text_len + 1) as f32;
        let mut max_t = 0f32;
        let mut max_h = 0f32;
        let mut max_w = 0f32;
        for it in 0..t_lat {
            let t_val = base_f + (it as f32) * T_AXIS_MULTIPLIER;
            for ih in 0..h_lat {
                let h_val = base_f + ih as f32;
                for iw in 0..w_lat {
                    let w_val = base_f + iw as f32;
                    pos_t.push(t_val);
                    pos_h.push(h_val);
                    pos_w.push(w_val);
                    if t_val > max_t { max_t = t_val; }
                    if h_val > max_h { max_h = h_val; }
                    if w_val > max_w { max_w = w_val; }
                }
            }
        }

        // EOI: diagonal at (global_max + 1). The capture shows the same
        // value used for all three axes — EOI is a text/bracket token, so
        // it gets a diagonal position past the last image position.
        let eoi_val = max_t.max(max_h).max(max_w) + 1.0;
        pos_t.push(eoi_val);
        pos_h.push(eoi_val);
        pos_w.push(eoi_val);

        let to_i32 = |v: Vec<f32>| -> Result<Tensor> {
            Tensor::from_vec(v, Shape::from_dims(&[l_noise]), dev.clone())?
                .to_dtype(DType::I32)
        };
        Ok((to_i32(pos_t)?, to_i32(pos_h)?, to_i32(pos_w)?))
    }

    /// **GEN STEP — T2V.** Like `gen_step` but mirrors Python's packed-
    /// sequence T2V denoise path: forwards a 770-token noise span
    /// `[SOI, vae×768, EOI]` through the language model and selects the
    /// VAE-position outputs.
    ///
    /// Differences from `gen_step` (= G1, G2 from T2V_DENOISE_PORT.md):
    ///   - **G1**: concatenates SOI and EOI text-token embeddings around
    ///     the patchified VAE embeddings before forwarding.
    ///   - **G2**: positions match Python's `get_rope_index` output (no
    ///     +1000 shift; T-axis multiplied by 2).
    ///   - Output `llm2vae` is applied to the full 770-token output, then
    ///     the SOI/EOI slots are dropped (positions 0 and L_noise-1) and
    ///     only the L_vae VAE slots are unpatchified.
    ///
    /// Time embedding is added ONLY to the VAE positions (mirrors Python
    /// — bracket tokens carry pure text embedding, no time signal).
    ///
    /// Positional embedding `latent_pos_embed` is added ONLY to the VAE
    /// positions (it's a per-resolution image embedding; bracket tokens
    /// get nothing from it).
    pub fn gen_step_t2v(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        mrope: &MropeFreqs,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        // ---- Validate input latent shape: [B=1, C, T, H, W]. ----
        let ldims = latent.shape().dims().to_vec();
        if ldims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step_t2v: expected 5D latent [B, C, T, H, W], got {ldims:?}"
            )));
        }
        let (b, c, t, h, w) = (ldims[0], ldims[1], ldims[2], ldims[3], ldims[4]);
        let p = self.config.patch_latent_dim();
        if c != p {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step_t2v: latent C={c} != patch_latent_dim={p}"
            )));
        }
        if b != 1 {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step_t2v: only B=1 supported, got B={b}"
            )));
        }
        let (pt, ph, pw) = self.config.latent_patch_size;
        if (pt, ph, pw) != (1, 1, 1) {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step_t2v: only latent_patch_size=(1,1,1) supported, got ({pt},{ph},{pw})"
            )));
        }

        // ---- 1. Patchify: [1, C, T, H, W] → [L_vae, C]. ----
        let l_vae = t * h * w;
        let latent_p = latent.permute(&[0, 2, 3, 4, 1])?; // [1, T, H, W, C]
        let x_patch_2d = latent_p.reshape(&[l_vae, p])?; // [L_vae, C]
        let x_patch_3d = x_patch_2d.reshape(&[1, l_vae, p])?; // [1, L_vae, C]

        // ---- 2. vae2llm: [1, L_vae, C] → [1, L_vae, hidden]. ----
        let mut h_vae = self.vae2llm.forward(&x_patch_3d)?;

        // ---- 3. Time embedding (vae-only). ----
        let t_dims = timestep.shape().dims();
        if t_dims.len() != 1 || t_dims[0] != 1 {
            return Err(Error::InvalidInput(format!(
                "Lance::gen_step_t2v: expected scalar timestep [1], got {t_dims:?}"
            )));
        }
        let t_scalar = timestep.to_dtype(DType::F32)?.to_vec()?[0];
        let t_vec = vec![t_scalar; l_vae];
        let t_tensor_f32 =
            Tensor::from_vec(t_vec, Shape::from_dims(&[l_vae]), self.config.device.clone())?;
        let t_tensor = if t_tensor_f32.dtype() != self.config.dtype {
            t_tensor_f32.to_dtype(self.config.dtype)?
        } else {
            t_tensor_f32
        };
        let time_emb = self.time_embedder.forward(&t_tensor)?; // [L_vae, hidden]
        let time_emb_3d = Self::to_3d(&time_emb)?;
        h_vae = h_vae.add(&time_emb_3d)?;

        // ---- 4. Latent pos embedding (vae-only). ----
        let pos_emb =
            self.latent_pos_embed
                .build(t, h, w, &self.config.device, self.config.dtype)?; // [L_vae, hidden]
        let pos_emb_3d = Self::to_3d(&pos_emb)?;
        h_vae = h_vae.add(&pos_emb_3d)?;

        // ---- 5. Embed SOI/EOI bracket tokens and concat. ----
        // [SOI_id, EOI_id] → [2, hidden]. We keep one tensor and slice it
        // into two [1, hidden] strips for cat.
        let bracket_ids = Tensor::from_vec(
            vec![Self::VISION_START_TOKEN_ID as f32, Self::VISION_END_TOKEN_ID as f32],
            Shape::from_dims(&[2]),
            self.config.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let bracket_emb_2d = self.embed_text_tokens(&bracket_ids)?; // [2, hidden]
        let soi_2d = bracket_emb_2d.narrow(0, 0, 1)?; // [1, hidden]
        let eoi_2d = bracket_emb_2d.narrow(0, 1, 1)?; // [1, hidden]
        let soi_3d = soi_2d.reshape(&[1, 1, soi_2d.shape().dims()[1]])?; // [1, 1, hidden]
        let eoi_3d = eoi_2d.reshape(&[1, 1, eoi_2d.shape().dims()[1]])?; // [1, 1, hidden]

        // h_vae is [1, L_vae, hidden]; concat on dim=1 → [1, L_noise=L_vae+2, hidden].
        let h_noise = Tensor::cat(&[&soi_3d, &h_vae, &eoi_3d], 1)?;
        let l_noise = l_vae + 2;
        debug_assert_eq!(h_noise.shape().dims(), &[1, l_noise, self.config.hidden_size]);

        // ---- 6. Gen-mask: all-gen for the noise span. ----
        let gen_mask = self.build_uniform_gen_mask(l_noise, 1.0)?;

        // ---- 7. Positions for the 770-token noise span. ----
        let prefix_len = cache.seq_len(0);
        let (pos_t_tensor, pos_h_tensor, pos_w_tensor) =
            self.build_t2v_noise_span_positions(prefix_len, t, h, w)?;

        // ---- 8. Block stack: is_causal=false, update_cache=false. ----
        let h_out_3d = self.blocks.forward(
            &h_noise,
            &gen_mask,
            &pos_t_tensor,
            &pos_h_tensor,
            &pos_w_tensor,
            mrope,
            false, // is_causal — non-causal within noise span
            Some(cache),
            false, // update_cache — READ-ONLY
        )?;

        // ---- 9. llm2vae over the full noise span, then drop SOI/EOI. ----
        let v_noise_2d = self.llm2vae.forward(&h_out_3d)?; // [1, L_noise, C]
        // Select VAE positions [1..1+L_vae) from the noise span.
        let v_vae_3d = v_noise_2d.narrow(1, 1, l_vae)?; // [1, L_vae, C]
        let v_vae_flat = v_vae_3d.reshape(&[l_vae, p])?; // [L_vae, C]

        // ---- 10. Unpatchify: [L_vae, C] → [B, C, T, H, W]. ----
        let v_5d_thwc = v_vae_flat.reshape(&[b, t, h, w, p])?; // [B, T, H, W, C]
        v_5d_thwc.permute(&[0, 4, 1, 2, 3]) // [B, C, T, H, W]
    }
}

// ===========================================================================
// C4 Module 13 tests
// ===========================================================================
//
// Lives in its own `tests_module13` block to avoid colliding with the C2/C3
// `tests` module and the C4-leaves `tests_c4` module.
#[cfg(test)]
mod tests_module13 {
    use super::*;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device 0")
    }

    /// Build a 2-layer Lance config — full 36 layers would allocate too
    /// much GPU memory just for shape tests (each MoT layer has paired
    /// 11008-wide MLPs + paired q/k/v/o + paired norms).
    fn small_cfg() -> LanceConfig {
        let mut cfg = LanceConfig::default_3b(dev());
        cfg.num_hidden_layers = 2;
        cfg
    }

    #[test]
    fn test_lance_new_random_basic() {
        let cfg = small_cfg();
        let vocab = cfg.vocab_size;
        let hidden = cfg.hidden_size;
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();
        assert_eq!(
            lance.embed_tokens.shape().dims(),
            &[vocab, hidden],
            "embed_tokens table shape"
        );
    }

    #[test]
    fn test_lance_prefill_text_context_grows_cache() {
        let cfg = small_cfg();
        let n_layers = cfg.num_hidden_layers;
        let d = cfg.device.clone();
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();

        let mut cache = lance.new_kv_cache();
        // Module 13 tests need max_pos large enough to cover the image-gen
        // path's `t_offset = cache.seq_len(0) + 1000` per-token positions.
        // 2048 fits 1000-shift + small grid sizes. Production callers will
        // size this larger (typical T2I needs `text_len + 1000 + T*H*W`).
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();

        // 5 text token ids as f32 (will be cast to I32 internally for
        // index_select0). Values are well within vocab_size=151936.
        let text_ids = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            Shape::from_dims(&[5]),
            d.clone(),
        )
        .unwrap();
        lance
            .prefill_text_context(&text_ids, &mrope, &mut cache)
            .unwrap();

        for layer_idx in 0..n_layers {
            assert_eq!(
                cache.seq_len(layer_idx),
                5,
                "layer {layer_idx} cache should have 5 K positions after prefill"
            );
        }
    }

    #[test]
    fn test_lance_gen_step_shape() {
        let cfg = small_cfg();
        let d = cfg.device.clone();
        let c = cfg.latent_channels;
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();

        let mut cache = lance.new_kv_cache();
        // Module 13 tests need max_pos large enough to cover the image-gen
        // path's `t_offset = cache.seq_len(0) + 1000` per-token positions.
        // 2048 fits 1000-shift + small grid sizes. Production callers will
        // size this larger (typical T2I needs `text_len + 1000 + T*H*W`).
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();

        // Empty cache (degenerate): gen_step must still produce a v_pred
        // with the same shape as the input latent.
        let latent = Tensor::randn_seeded(
            Shape::from_dims(&[1, c, 1, 8, 8]), // [B, 48, T=1, H=8, W=8]
            0.0,
            1.0,
            4242,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let timestep = Tensor::from_vec(vec![0.5f32], Shape::from_dims(&[1]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let v_pred = lance.gen_step(&latent, &timestep, &mrope, &mut cache).unwrap();
        assert_eq!(
            v_pred.shape().dims(),
            latent.shape().dims(),
            "v_pred shape must equal latent shape"
        );

        // gen_step with update_cache=false → cache UNCHANGED (still empty).
        for layer_idx in 0..lance.config.num_hidden_layers {
            assert_eq!(
                cache.seq_len(layer_idx),
                0,
                "layer {layer_idx} cache must remain empty after gen_step (update_cache=false)"
            );
        }
    }

    #[test]
    fn test_lance_gen_step_with_prefilled_cache_uses_history() {
        // Pin the structural property "cache history affects v_pred".
        // Two prefills with DIFFERENT text tokens build two DIFFERENT
        // KV caches; running the same latent + timestep through
        // gen_step against each must produce DIFFERENT v_pred.
        //
        // If the gen step ignored the cache (or always saw the same
        // K/V), the two outputs would be bit-identical — that would
        // mean F3 (cache read in gen step) is silently broken at the
        // top-level wiring even though the unit tests pass.
        let cfg = small_cfg();
        let d = cfg.device.clone();
        let c = cfg.latent_channels;
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();

        // Module 13 tests need max_pos large enough to cover the image-gen
        // path's `t_offset = cache.seq_len(0) + 1000` per-token positions.
        // 2048 fits 1000-shift + small grid sizes. Production callers will
        // size this larger (typical T2I needs `text_len + 1000 + T*H*W`).
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();
        let latent = Tensor::randn_seeded(
            Shape::from_dims(&[1, c, 1, 4, 4]),
            0.0,
            1.0,
            4242,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let timestep = Tensor::from_vec(vec![0.5f32], Shape::from_dims(&[1]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // --- Scenario A: prefill text_a, then gen step. ---
        let mut cache_a = lance.new_kv_cache();
        let text_a = Tensor::from_vec(
            (1..=10).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[10]),
            d.clone(),
        )
        .unwrap();
        lance
            .prefill_text_context(&text_a, &mrope, &mut cache_a)
            .unwrap();
        let v_pred_a = lance
            .gen_step(&latent, &timestep, &mrope, &mut cache_a)
            .unwrap();

        // --- Scenario B: prefill text_b (different ids), then gen step. ---
        let mut cache_b = lance.new_kv_cache();
        let text_b = Tensor::from_vec(
            (11..=20).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[10]),
            d.clone(),
        )
        .unwrap();
        lance
            .prefill_text_context(&text_b, &mrope, &mut cache_b)
            .unwrap();
        let v_pred_b = lance
            .gen_step(&latent, &timestep, &mrope, &mut cache_b)
            .unwrap();

        // Compare in F32 to avoid BF16 rounding eating a small diff.
        let va = v_pred_a.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let vb = v_pred_b.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(va.len(), vb.len(), "v_pred elem counts must match");

        let max_diff = va
            .iter()
            .zip(vb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "cache history must affect v_pred — but max_diff between v_pred_a and v_pred_b was {max_diff}; cache reads in gen_step may be ignored"
        );

        // Both must be finite.
        for v in va.iter() {
            assert!(v.is_finite(), "non-finite v_pred_a element {v}");
        }
        for v in vb.iter() {
            assert!(v.is_finite(), "non-finite v_pred_b element {v}");
        }
    }

    /// **T2V T1 gate.** End-to-end `gen_step` with T>1 latent — the entire
    /// existing test surface uses T=1, leaving the T-axis path through
    /// patchify+positions+mRoPE+block-stack+unpatchify untested at the
    /// integration level. mRoPE primitive coverage at T>1 lives in
    /// `test_mrope_per_token_positions_produce_different_rotations`; this
    /// test pins that the wiring around it also handles T>1.
    ///
    /// Two assertions, both structural:
    ///   1. Shape round-trip: `[B, 48, T, H, W]` in → same shape out for T=3.
    ///   2. Position discrimination: t=0 slice and t=2 slice of the v_pred
    ///      must differ. If the entire forward path were silently T-agnostic
    ///      (e.g. position-build only used h,w and dropped t, or block stack
    ///      collapsed T into batch), all T slices would be bit-identical.
    #[test]
    fn test_lance_gen_step_t_greater_than_one() {
        let cfg = small_cfg();
        let d = cfg.device.clone();
        let c = cfg.latent_channels;
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();

        let mut cache = lance.new_kv_cache();
        // T*H*W = 3*4*4 = 48 image tokens + 1000 shift → max_pos 1100 fits.
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();

        // T=3 latent, deterministic random init.
        let latent = Tensor::randn_seeded(
            Shape::from_dims(&[1, c, 3, 4, 4]), // [B=1, 48, T=3, H=4, W=4]
            0.0,
            1.0,
            7777,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let timestep = Tensor::from_vec(vec![0.5f32], Shape::from_dims(&[1]), d.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let v_pred = lance.gen_step(&latent, &timestep, &mrope, &mut cache).unwrap();
        assert_eq!(
            v_pred.shape().dims(),
            latent.shape().dims(),
            "T2V T1 gate: v_pred shape must equal latent shape at T=3"
        );

        // Pull t=0 and t=2 frames out for slice-discrimination check.
        // v_pred layout: [B=1, C=48, T=3, H=4, W=4]. Narrow along dim 2.
        let frame0 = v_pred.narrow(2, 0, 1).unwrap();
        let frame2 = v_pred.narrow(2, 2, 1).unwrap();
        let f0 = frame0.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let f2 = frame2.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(f0.len(), f2.len(), "frame slice lengths must match");

        let max_diff = f0
            .iter()
            .zip(f2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // If the forward path were T-agnostic (e.g. mRoPE only built h,w
        // positions and dropped t), t=0 and t=2 v_pred slices would be
        // bit-identical. The per-token gen position builder offsets by
        // `text_len + 1000 + t`, so different t produces different mRoPE
        // rotation → different attention output → different v_pred.
        assert!(
            max_diff > 1e-3,
            "T2V T1 gate: t=0 and t=2 v_pred slices must differ (T-axis live); got max_abs_diff = {max_diff}. \
             If 0.0, mRoPE T-axis or position builder is silently T-agnostic."
        );

        // All elements finite.
        for v in f0.iter().chain(f2.iter()) {
            assert!(v.is_finite(), "non-finite v_pred element {v}");
        }
    }
}

// ===========================================================================
// C6: Production weight loader — Lance_3B/model.safetensors
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/lance.py:111-140`
// (`init_from_model_path_if_needed`). The Python loader:
//   1. Reads `Lance_3B/model.safetensors` into a single state-dict
//      (1021 tensors per `inference-flame/ports/lance/STATE_DICT_KEYS.md`).
//   2. POPs `latent_pos_embed.pos_embed` (line 131) — the position table is
//      recomputed per-resolution at runtime by `LatentPosEmbed::build`, so
//      the stored `[4096, 2048]` buffer is wasted memory and must be
//      discarded before `load_state_dict(..., strict=True)` accepts the
//      remaining keys.
//   3. Loads everything else by `load_state_dict`.
//
// **F32 → BF16 downcast.** Per Lane C audit (STATE_DICT_KEYS.md "Dtype
// finding"): every tensor on disk is F32 (4 bytes each, ~23 GB body),
// despite `llm_config.json` declaring `torch_dtype: bfloat16`. flame-core's
// `load_file` preserves the on-disk dtype (F32 stays F32). We therefore run
// an explicit pass after load that casts every F32 weight to BF16 (matching
// `cfg.dtype = BF16` in the canonical 3B config) before populating the
// model struct. This keeps the resident set at ~12 GB rather than ~23 GB.
//
// **No per-layer `mlp` vs `mlp_moe_gen` ambiguity.** STATE_DICT_KEYS.md
// lines 71-77 confirm the key naming: `mlp.gate_proj.weight` and
// `mlp_moe_gen.gate_proj.weight` are SIBLINGS under
// `language_model.model.layers.<i>.`, NOT nested as
// `mlp.gate_proj_moe_gen.weight`. The Python parent uses
// `self.mlp_moe_gen = Qwen2MLP(...)` (qwen2_navit.py:589), making it a
// separate `nn.Module` attribute whose own state-dict prefix is
// `mlp_moe_gen.`. Same pattern for `input_layernorm` /
// `input_layernorm_moe_gen` (sibling RMSNorm gain tensors); inside
// `self_attn.` the per-Linear attributes `q_proj` / `q_proj_moe_gen` are
// also siblings at the same nesting level.

impl Vae2Llm {
    /// Load `vae2llm` weights from a state-dict.
    ///
    /// Expected keys (per `STATE_DICT_KEYS.md` line 30):
    ///   `<prefix>weight`  shape `[hidden_size=2048, patch_latent_dim=48]`
    ///   `<prefix>bias`    shape `[hidden_size=2048]`
    ///
    /// Caller passes `prefix = "vae2llm."`.
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        _cfg: &LanceConfig,
    ) -> Result<Self> {
        let w_key = format!("{prefix}weight");
        let b_key = format!("{prefix}bias");
        let weight = weights.get(&w_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("Vae2Llm::load_from_state_dict: missing key {w_key}"))
        })?;
        let bias = weights.get(&b_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("Vae2Llm::load_from_state_dict: missing key {b_key}"))
        })?;
        Ok(Self { weight, bias })
    }
}

impl Llm2Vae {
    /// Load `llm2vae` weights from a state-dict.
    ///
    /// Expected keys (per `STATE_DICT_KEYS.md` line 31):
    ///   `<prefix>weight`  shape `[patch_latent_dim=48, hidden_size=2048]`
    ///   `<prefix>bias`    shape `[patch_latent_dim=48]`
    ///
    /// Caller passes `prefix = "llm2vae."`.
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        _cfg: &LanceConfig,
    ) -> Result<Self> {
        let w_key = format!("{prefix}weight");
        let b_key = format!("{prefix}bias");
        let weight = weights.get(&w_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("Llm2Vae::load_from_state_dict: missing key {w_key}"))
        })?;
        let bias = weights.get(&b_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("Llm2Vae::load_from_state_dict: missing key {b_key}"))
        })?;
        Ok(Self { weight, bias })
    }
}

impl TimeEmbedder {
    /// Load time embedder weights.
    ///
    /// Expected keys (per `STATE_DICT_KEYS.md` line 32):
    ///   `<prefix>mlp.0.weight` `[2048, 256]`
    ///   `<prefix>mlp.0.bias`   `[2048]`
    ///   `<prefix>mlp.2.weight` `[2048, 2048]`
    ///   `<prefix>mlp.2.bias`   `[2048]`
    ///
    /// Index 1 in Python's `nn.Sequential` is the SiLU activation (no
    /// params), so there is no `mlp.1.*` key.
    ///
    /// Caller passes `prefix = "time_embedder."`.
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &LanceConfig,
    ) -> Result<Self> {
        let k0w = format!("{prefix}mlp.0.weight");
        let k0b = format!("{prefix}mlp.0.bias");
        let k2w = format!("{prefix}mlp.2.weight");
        let k2b = format!("{prefix}mlp.2.bias");
        let mlp_0_weight = weights.get(&k0w).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("TimeEmbedder::load_from_state_dict: missing key {k0w}"))
        })?;
        let mlp_0_bias = weights.get(&k0b).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("TimeEmbedder::load_from_state_dict: missing key {k0b}"))
        })?;
        let mlp_2_weight = weights.get(&k2w).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("TimeEmbedder::load_from_state_dict: missing key {k2w}"))
        })?;
        let mlp_2_bias = weights.get(&k2b).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("TimeEmbedder::load_from_state_dict: missing key {k2b}"))
        })?;
        Ok(Self {
            mlp_0_weight,
            mlp_0_bias,
            mlp_2_weight,
            mlp_2_bias,
            sinusoidal_dim: 256,
            hidden_size: cfg.hidden_size,
        })
    }
}

impl LatentPosEmbed {
    /// Loader — there are no learnable parameters for the position embed.
    /// The state-dict's `latent_pos_embed.pos_embed [4096, 2048]` is
    /// popped at the top-level `Lance::load` (mirror of Python
    /// `lance.py:131`); the table is recomputed at runtime per resolution
    /// by `Self::build`.
    ///
    /// We accept the `weights` + `prefix` args for API uniformity with the
    /// other leaves; both are ignored. Caller passes
    /// `prefix = "latent_pos_embed."`.
    pub fn load_from_state_dict(
        _weights: &HashMap<String, Tensor>,
        _prefix: &str,
        cfg: &LanceConfig,
    ) -> Result<Self> {
        Ok(Self::new(cfg))
    }
}

impl Qwen2MoTAttention {
    /// Load MoT attention weights from a state-dict.
    ///
    /// Expected keys (per `STATE_DICT_KEYS.md` lines 45-67), all relative
    /// to `<prefix>` (caller passes
    /// `"language_model.model.layers.<i>.self_attn."`):
    ///
    /// und path:
    ///   `q_proj.weight` `[2048,2048]`, `q_proj.bias` `[2048]`
    ///   `k_proj.weight` `[256, 2048]`, `k_proj.bias` `[256]`
    ///   `v_proj.weight` `[256, 2048]`, `v_proj.bias` `[256]`
    ///   `o_proj.weight` `[2048,2048]`  (NO bias)
    ///   `q_norm.weight` `[128]`        (RMSNorm gain at head_dim)
    ///   `k_norm.weight` `[128]`
    ///
    /// gen path (siblings under the same `self_attn.` prefix; the
    /// `_moe_gen` suffix is on the inner attribute name, NOT the parent):
    ///   `q_proj_moe_gen.weight`, `q_proj_moe_gen.bias`
    ///   `k_proj_moe_gen.weight`, `k_proj_moe_gen.bias`
    ///   `v_proj_moe_gen.weight`, `v_proj_moe_gen.bias`
    ///   `o_proj_moe_gen.weight`  (NO bias)
    ///   `q_norm_moe_gen.weight`, `k_norm_moe_gen.weight`
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &LanceConfig,
    ) -> Result<Self> {
        // Small helper closure to fetch with consistent error reporting.
        let pick = |sub: &str| -> Result<Tensor> {
            let k = format!("{prefix}{sub}");
            weights.get(&k).cloned().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Qwen2MoTAttention::load_from_state_dict: missing key {k}"
                ))
            })
        };

        // und path
        let q_proj = pick("q_proj.weight")?;
        let q_bias = pick("q_proj.bias")?;
        let k_proj = pick("k_proj.weight")?;
        let k_bias = pick("k_proj.bias")?;
        let v_proj = pick("v_proj.weight")?;
        let v_bias = pick("v_proj.bias")?;
        let o_proj = pick("o_proj.weight")?; // NO bias for o_proj
        let q_norm = pick("q_norm.weight")?;
        let k_norm = pick("k_norm.weight")?;

        // gen path (_moe_gen siblings)
        let q_proj_moe_gen = pick("q_proj_moe_gen.weight")?;
        let q_bias_moe_gen = pick("q_proj_moe_gen.bias")?;
        let k_proj_moe_gen = pick("k_proj_moe_gen.weight")?;
        let k_bias_moe_gen = pick("k_proj_moe_gen.bias")?;
        let v_proj_moe_gen = pick("v_proj_moe_gen.weight")?;
        let v_bias_moe_gen = pick("v_proj_moe_gen.bias")?;
        let o_proj_moe_gen = pick("o_proj_moe_gen.weight")?; // NO bias
        let q_norm_moe_gen = pick("q_norm_moe_gen.weight")?;
        let k_norm_moe_gen = pick("k_norm_moe_gen.weight")?;

        Ok(Self {
            q_proj,
            q_bias,
            k_proj,
            k_bias,
            v_proj,
            v_bias,
            o_proj,
            q_norm,
            k_norm,
            q_proj_moe_gen,
            q_bias_moe_gen,
            k_proj_moe_gen,
            k_bias_moe_gen,
            v_proj_moe_gen,
            v_bias_moe_gen,
            o_proj_moe_gen,
            q_norm_moe_gen,
            k_norm_moe_gen,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            mrope_sections: cfg.mrope_sections,
        })
    }
}

impl Qwen2MoTMLP {
    /// Load MoT MLP weights from a state-dict.
    ///
    /// **Key naming.** STATE_DICT_KEYS.md lines 71-77 confirm the MLP
    /// `_moe_gen` parent is a SIBLING `nn.Module`, not a per-attribute
    /// suffix. So the keys are:
    ///   und:    `<layer_prefix>mlp.gate_proj.weight` etc.
    ///   gen:    `<layer_prefix>mlp_moe_gen.gate_proj.weight` etc.
    ///
    /// The caller of this loader passes `layer_prefix =
    /// "language_model.model.layers.<i>."` and we form the two
    /// sub-prefixes (`mlp.`, `mlp_moe_gen.`) internally.
    ///
    /// All six tensors are bias-free (Qwen2 standard SwiGLU MLP).
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        layer_prefix: &str,
        _cfg: &LanceConfig,
    ) -> Result<Self> {
        let pick = |sub: &str| -> Result<Tensor> {
            let k = format!("{layer_prefix}{sub}");
            weights.get(&k).cloned().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Qwen2MoTMLP::load_from_state_dict: missing key {k}"
                ))
            })
        };

        let gate_proj = pick("mlp.gate_proj.weight")?;
        let up_proj = pick("mlp.up_proj.weight")?;
        let down_proj = pick("mlp.down_proj.weight")?;

        let gate_proj_moe_gen = pick("mlp_moe_gen.gate_proj.weight")?;
        let up_proj_moe_gen = pick("mlp_moe_gen.up_proj.weight")?;
        let down_proj_moe_gen = pick("mlp_moe_gen.down_proj.weight")?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_moe_gen,
            up_proj_moe_gen,
            down_proj_moe_gen,
        })
    }
}

impl Qwen2MoTDecoderLayer {
    /// Load a MoT decoder layer (the only layer type Lance 3B uses; the
    /// state-dict has zero non-MoT layers per `STATE_DICT_KEYS.md`).
    ///
    /// Caller passes `layer_prefix = "language_model.model.layers.<i>."`.
    /// Sub-keys under that prefix:
    ///   `input_layernorm.weight`                       `[2048]`
    ///   `input_layernorm_moe_gen.weight`               `[2048]`
    ///   `post_attention_layernorm.weight`              `[2048]`
    ///   `post_attention_layernorm_moe_gen.weight`      `[2048]`
    ///   `self_attn.*`                                  → `Qwen2MoTAttention`
    ///   `mlp.*` + `mlp_moe_gen.*`                      → `Qwen2MoTMLP`
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        layer_prefix: &str,
        cfg: &LanceConfig,
    ) -> Result<Self> {
        let pick = |sub: &str| -> Result<Tensor> {
            let k = format!("{layer_prefix}{sub}");
            weights.get(&k).cloned().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Qwen2MoTDecoderLayer::load_from_state_dict: missing key {k}"
                ))
            })
        };

        let input_layernorm = pick("input_layernorm.weight")?;
        let input_layernorm_moe_gen = pick("input_layernorm_moe_gen.weight")?;
        let post_attention_layernorm = pick("post_attention_layernorm.weight")?;
        let post_attention_layernorm_moe_gen = pick("post_attention_layernorm_moe_gen.weight")?;

        let self_attn = Qwen2MoTAttention::load_from_state_dict(
            weights,
            &format!("{layer_prefix}self_attn."),
            cfg,
        )?;
        let mlp = Qwen2MoTMLP::load_from_state_dict(weights, layer_prefix, cfg)?;

        Ok(Self {
            input_layernorm,
            input_layernorm_moe_gen,
            post_attention_layernorm,
            post_attention_layernorm_moe_gen,
            self_attn,
            mlp,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }
}

impl Qwen2Attention {
    /// **Not loaded.** Lance 3B's state-dict contains zero non-MoT layers
    /// (per `STATE_DICT_KEYS.md` — all 36 layers carry `_moe_gen`
    /// siblings, so they are `Qwen2MoTDecoderLayer`, not
    /// `Qwen2DecoderLayer`). This method exists for API completeness with
    /// the MoT variants; calling it against the published checkpoint will
    /// fail with a clear error.
    pub fn load_from_state_dict(
        _weights: &HashMap<String, Tensor>,
        _prefix: &str,
        _cfg: &LanceConfig,
    ) -> Result<Self> {
        Err(Error::InvalidInput(
            "Qwen2Attention::load_from_state_dict: Lance 3B has zero non-MoT layers in its \
             state-dict; use Qwen2MoTAttention::load_from_state_dict instead"
                .to_string(),
        ))
    }
}

impl Qwen2MLP {
    /// **Not loaded.** See `Qwen2Attention::load_from_state_dict`.
    pub fn load_from_state_dict(
        _weights: &HashMap<String, Tensor>,
        _prefix: &str,
        _cfg: &LanceConfig,
    ) -> Result<Self> {
        Err(Error::InvalidInput(
            "Qwen2MLP::load_from_state_dict: Lance 3B has zero non-MoT layers in its \
             state-dict; use Qwen2MoTMLP::load_from_state_dict instead"
                .to_string(),
        ))
    }
}

impl Qwen2DecoderLayer {
    /// **Not loaded.** See `Qwen2Attention::load_from_state_dict`.
    pub fn load_from_state_dict(
        _weights: &HashMap<String, Tensor>,
        _prefix: &str,
        _cfg: &LanceConfig,
    ) -> Result<Self> {
        Err(Error::InvalidInput(
            "Qwen2DecoderLayer::load_from_state_dict: Lance 3B has zero non-MoT layers in its \
             state-dict; use Qwen2MoTDecoderLayer::load_from_state_dict instead"
                .to_string(),
        ))
    }
}

impl LanceBlockStack {
    /// Load the full 36-layer decoder stack + paired final RMSNorm.
    ///
    /// Caller passes `prefix = "language_model.model."` — the
    /// `layers.<i>.` per-layer prefix and `norm.weight` /
    /// `norm_moe_gen.weight` are formed internally.
    ///
    /// Always produces MoT layers (`LanceBlock::MoT`) per
    /// `STATE_DICT_KEYS.md`: Lance 3B has zero base (non-MoT) layers in
    /// the published checkpoint.
    pub fn load_from_state_dict(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &LanceConfig,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_prefix = format!("{prefix}layers.{i}.");
            let layer = Qwen2MoTDecoderLayer::load_from_state_dict(weights, &layer_prefix, cfg)?;
            blocks.push(LanceBlock::MoT(layer));
        }

        let fn_key = format!("{prefix}norm.weight");
        let final_norm = weights.get(&fn_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!(
                "LanceBlockStack::load_from_state_dict: missing key {fn_key}"
            ))
        })?;
        let fn_moe_key = format!("{prefix}norm_moe_gen.weight");
        let final_norm_moe_gen = weights.get(&fn_moe_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!(
                "LanceBlockStack::load_from_state_dict: missing key {fn_moe_key}"
            ))
        })?;

        Ok(Self {
            blocks,
            final_norm,
            final_norm_moe_gen,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }
}

impl Lance {
    /// Load Lance T2I model from a `Lance_3B/` directory containing
    /// `model.safetensors`.
    ///
    /// Pipeline:
    ///   1. Mmap-load `model_path/model.safetensors` via
    ///      `flame_core::serialization::load_file` (1021 tensors).
    ///   2. Pop `latent_pos_embed.pos_embed` — recomputed at runtime per
    ///      resolution by `LatentPosEmbed::build`. Mirror of Python
    ///      `lance.py:131`.
    ///   3. Downcast every F32 weight to `cfg.dtype` (BF16 in production).
    ///      Per `STATE_DICT_KEYS.md`, the on-disk file is F32 despite
    ///      `llm_config.json: torch_dtype=bfloat16`. The downcast saves
    ///      ~11 GB resident memory.
    ///   4. Wire every leaf via its `load_from_state_dict` method, then
    ///      pull `embed_tokens` from the residual map.
    ///
    /// **Streaming loader (2026-05-18 fix).** The previous implementation
    /// used `flame_core::serialization::load_file` (which uploads every
    /// tensor to GPU in its on-disk dtype — F32 for Lance 3B) and only
    /// then ran a second pass to cast each F32 → BF16. Peak VRAM during
    /// the cast pass was the full F32 working set (~24 GB) plus the
    /// growing BF16 cast results, which OOMs a 24 GB GPU. User is remote
    /// (see `feedback_remote_no_reboot_staged_exec`) so OOM is
    /// unrecoverable.
    ///
    /// The streaming loop below mmaps the safetensors file via
    /// `memmap2::Mmap`, parses the header with `serde_json`, then for
    /// each tensor: reads the F32 bytes from the mmap region into a host
    /// `Vec<f32>`, calls `Tensor::from_vec_dtype(.., target_dtype)` which
    /// uploads to GPU and converts to BF16 in-kernel (the F32 cuda
    /// buffer it allocates internally is dropped when the call returns),
    /// and inserts the BF16 tensor into the output map. Peak VRAM is
    /// thus ~12 GB BF16 cumulative + ~2 × largest_tensor F32 transient
    /// during a single conversion. The largest Lance tensors are the
    /// 2048×2048 attention projections (~16 MB F32 each) and the
    /// `embed_tokens` table (`[vocab, hidden]` ≈ 1.2 GB F32 for the
    /// 151k-vocab Qwen2 tokenizer), so transient overhead is bounded
    /// at ~2.4 GB → total peak ≈ 14.5 GB, comfortably under 24 GB.
    ///
    /// Note: `wan22_vae.rs::Wan22VaeDecoder::load` has the same
    /// `load_file`+cast pattern but its 1.3 GB BF16 weight set is far
    /// below the OOM ceiling. Fix deferred — see follow-up task.
    pub fn load(
        model_path: &Path,
        cfg: Arc<LanceConfig>,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;

        let safetensors_path = model_path.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(Error::InvalidInput(format!(
                "Lance::load: model.safetensors not found at {}",
                safetensors_path.display()
            )));
        }

        // 1. Mmap the safetensors file. `memmap2` exposes the data on
        //    demand via the kernel page cache — no bulk heap allocation.
        let mut file = File::open(&safetensors_path).map_err(|e| {
            Error::Io(format!(
                "Lance::load: open '{}': {e}",
                safetensors_path.display()
            ))
        })?;
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes)
            .map_err(|e| Error::Io(format!("Lance::load: read header size: {e}")))?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;
        if header_size > 100 * 1024 * 1024 {
            return Err(Error::Io(format!(
                "Lance::load: safetensors header too large ({header_size} bytes)"
            )));
        }
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes)
            .map_err(|e| Error::Io(format!("Lance::load: read header: {e}")))?;
        let header: serde_json::Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| Error::Io(format!("Lance::load: parse header: {e}")))?;
        drop(header_bytes);

        // Mmap the *whole file* (header + data). Tensor `data_offsets`
        // in the safetensors header are relative to the start of the
        // data segment, i.e. after the 8-byte length prefix and the
        // header JSON. We compute the absolute byte offset by adding
        // `data_start = 8 + header_size`.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            Error::Io(format!(
                "Lance::load: mmap '{}': {e}",
                safetensors_path.display()
            ))
        })?;
        let data_start: usize = 8 + header_size;

        // 2. Iterate tensor entries in the header. For each: skip the
        //    recomputable `latent_pos_embed.pos_embed`, skip non-data
        //    metadata, then load → cast → insert. Drop the F32 host
        //    buffer at each scope-exit (`Tensor::from_vec_dtype` consumes
        //    it). The F32 cuda intermediate inside `from_vec_dtype` is
        //    likewise dropped before the next iteration.
        let target_dtype = cfg.dtype;
        let header_obj = header
            .as_object()
            .ok_or_else(|| Error::Io("Lance::load: header is not a JSON object".to_string()))?;

        let mut weights: HashMap<String, Tensor> = HashMap::with_capacity(header_obj.len());
        let mut casts_done: usize = 0;
        let mut total_loaded: usize = 0;
        let mut dropped_pos_embed = false;

        for (name, info) in header_obj.iter() {
            // Skip the safetensors metadata block.
            if name == "__metadata__" {
                continue;
            }
            // Skip the recomputable latent pos-embed buffer (Python
            // `lance.py:131`). `LatentPosEmbed::build` recomputes per
            // resolution at runtime.
            if name == "latent_pos_embed.pos_embed" {
                dropped_pos_embed = true;
                continue;
            }

            let dtype_str = info
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| {
                    Error::Io(format!("Lance::load: tensor '{name}' missing dtype"))
                })?;
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let offsets = info
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| {
                    Error::Io(format!(
                        "Lance::load: tensor '{name}' missing data_offsets"
                    ))
                })?;
            let off_start = offsets
                .first()
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    Error::Io(format!(
                        "Lance::load: tensor '{name}' bad data_offsets[0]"
                    ))
                })? as usize;
            let off_end = offsets
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    Error::Io(format!(
                        "Lance::load: tensor '{name}' bad data_offsets[1]"
                    ))
                })? as usize;

            let abs_start = data_start + off_start;
            let abs_end = data_start + off_end;
            if abs_end > mmap.len() {
                return Err(Error::Io(format!(
                    "Lance::load: tensor '{name}' range {abs_start}..{abs_end} exceeds mmap len {}",
                    mmap.len()
                )));
            }
            let bytes = &mmap[abs_start..abs_end];

            // Decode this single tensor, cast to target dtype, insert.
            // The F32 cuda intermediate (when dtype=BF16) lives inside
            // `Tensor::from_vec_dtype` and is dropped at function exit
            // — peak transient is one tensor's F32 worth of VRAM.
            let tensor = match dtype_str {
                "F32" => {
                    let num = bytes.len() / 4;
                    let mut data = vec![0.0f32; num];
                    for (value, chunk) in data.iter_mut().zip(bytes.chunks_exact(4)) {
                        *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    if target_dtype != DType::F32 {
                        casts_done += 1;
                    }
                    Tensor::from_vec_dtype(
                        data,
                        Shape::from_dims(&shape),
                        device.clone(),
                        target_dtype,
                    )?
                }
                "BF16" => {
                    // Already BF16 on disk. Upload via the existing
                    // BF16 raw-slice path; no F32 transient. If the
                    // target dtype is not BF16 (unusual), fall back to
                    // `to_dtype` which still drops the BF16 source on
                    // the next iteration.
                    let num = bytes.len() / 2;
                    let mut bf16_u16 = vec![0u16; num];
                    for (value, chunk) in bf16_u16.iter_mut().zip(bytes.chunks_exact(2)) {
                        *value = u16::from_le_bytes([chunk[0], chunk[1]]);
                    }
                    let mut t = Tensor::zeros_dtype(
                        Shape::from_dims(&shape),
                        DType::BF16,
                        device.clone(),
                    )?;
                    t.copy_from_bf16_slice(&bf16_u16)?;
                    if target_dtype != DType::BF16 {
                        casts_done += 1;
                        t.to_dtype(target_dtype)?
                    } else {
                        t
                    }
                }
                other => {
                    return Err(Error::InvalidInput(format!(
                        "Lance::load: tensor '{name}' has unsupported dtype {other}"
                    )));
                }
            };

            weights.insert(name.clone(), tensor);
            total_loaded += 1;
        }

        log::info!(
            "[Lance] streamed {} tensors from {} ({} cast to {:?}{})",
            total_loaded,
            safetensors_path.display(),
            casts_done,
            target_dtype,
            if dropped_pos_embed {
                ", dropped latent_pos_embed.pos_embed"
            } else {
                ""
            }
        );

        // 4. Wire leaves. Each loader returns its struct with the relevant
        //    keys copied out; remaining keys are accounted for by the
        //    explicit `weights.remove` calls for top-level singletons.
        let vae2llm = Vae2Llm::load_from_state_dict(&weights, "vae2llm.", &cfg)?;
        let llm2vae = Llm2Vae::load_from_state_dict(&weights, "llm2vae.", &cfg)?;
        let time_embedder =
            TimeEmbedder::load_from_state_dict(&weights, "time_embedder.", &cfg)?;
        let latent_pos_embed =
            LatentPosEmbed::load_from_state_dict(&weights, "latent_pos_embed.", &cfg)?;
        let blocks = LanceBlockStack::load_from_state_dict(
            &weights,
            "language_model.model.",
            &cfg,
        )?;

        // embed_tokens is a top-level singleton.
        let embed_tokens_key = "language_model.model.embed_tokens.weight".to_string();
        let embed_tokens = weights.get(&embed_tokens_key).cloned().ok_or_else(|| {
            Error::InvalidInput(format!("Lance::load: missing key {embed_tokens_key}"))
        })?;

        // `language_model.lm_head.weight` exists in the file but is only
        // used by the VQA / text-loss path; T2I inference never consumes
        // it (`STATE_DICT_KEYS.md` "What is NOT paired" + Python
        // `lance.py` gen-step path). We deliberately do NOT thread it
        // into the struct; future understanding-path work can extend the
        // loader.

        Ok(Self {
            config: cfg,
            embed_tokens,
            vae2llm,
            llm2vae,
            time_embedder,
            latent_pos_embed,
            blocks,
        })
    }
}

#[cfg(test)]
mod tests_c6_loader {
    use super::*;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device 0")
    }

    /// Pin that the `Lance::load` method exists with the right signature
    /// and fails cleanly on a missing path. We deliberately do NOT load
    /// the real 24 GB checkpoint here — that's a smoke-test concern, not
    /// a unit-test concern.
    #[test]
    fn test_lance_load_method_compiles_and_errors_on_missing_path() {
        let d = dev();
        let cfg = Arc::new(LanceConfig::default_3b(d.clone()));
        let result = Lance::load(Path::new("/tmp/nonexistent_lance_path_c6"), cfg, &d);
        assert!(
            result.is_err(),
            "Lance::load should fail on nonexistent path"
        );
    }

    /// Smoke-test the per-leaf signatures by feeding empty/sparse
    /// state-dicts and asserting they return the expected "missing key"
    /// errors. This pins the API surface without touching real weights.
    #[test]
    fn test_leaf_loaders_report_missing_keys() {
        let d = dev();
        let cfg = LanceConfig::default_3b(d.clone());
        let empty: HashMap<String, Tensor> = HashMap::new();

        let r1 = Vae2Llm::load_from_state_dict(&empty, "vae2llm.", &cfg);
        assert!(r1.is_err(), "Vae2Llm loader must fail on empty state-dict");

        let r2 = Llm2Vae::load_from_state_dict(&empty, "llm2vae.", &cfg);
        assert!(r2.is_err(), "Llm2Vae loader must fail on empty state-dict");

        let r3 = TimeEmbedder::load_from_state_dict(&empty, "time_embedder.", &cfg);
        assert!(
            r3.is_err(),
            "TimeEmbedder loader must fail on empty state-dict"
        );

        // LatentPosEmbed is a no-op: empty state-dict is FINE because the
        // position table is recomputed at runtime.
        let r4 = LatentPosEmbed::load_from_state_dict(&empty, "latent_pos_embed.", &cfg);
        assert!(
            r4.is_ok(),
            "LatentPosEmbed loader must succeed on empty state-dict (no-op)"
        );

        let r5 = Qwen2MoTAttention::load_from_state_dict(&empty, "x.self_attn.", &cfg);
        assert!(
            r5.is_err(),
            "Qwen2MoTAttention loader must fail on empty state-dict"
        );

        let r6 = Qwen2MoTMLP::load_from_state_dict(&empty, "x.", &cfg);
        assert!(
            r6.is_err(),
            "Qwen2MoTMLP loader must fail on empty state-dict"
        );

        let r7 = Qwen2MoTDecoderLayer::load_from_state_dict(&empty, "x.", &cfg);
        assert!(
            r7.is_err(),
            "Qwen2MoTDecoderLayer loader must fail on empty state-dict"
        );

        let r8 =
            LanceBlockStack::load_from_state_dict(&empty, "language_model.model.", &cfg);
        assert!(
            r8.is_err(),
            "LanceBlockStack loader must fail on empty state-dict"
        );

        // Base (non-MoT) loaders should ALWAYS error — Lance 3B has zero
        // non-MoT layers, so calling these against the canonical
        // checkpoint is itself a misuse.
        let r9 = Qwen2Attention::load_from_state_dict(&empty, "x.", &cfg);
        assert!(r9.is_err(), "Qwen2Attention loader must always error");
        let r10 = Qwen2MLP::load_from_state_dict(&empty, "x.", &cfg);
        assert!(r10.is_err(), "Qwen2MLP loader must always error");
        let r11 = Qwen2DecoderLayer::load_from_state_dict(&empty, "x.", &cfg);
        assert!(r11.is_err(), "Qwen2DecoderLayer loader must always error");
    }
}

// ===========================================================================
// Module 14 + 15 (C5): Flow-matching denoise loop + classifier-free guidance
// ===========================================================================
//
// Reference: `/home/alex/Lance/modeling/lance/lance.py:1660-1769`
// (`validation_gen_KVcache`'s T2I per-step body). The Python loop is the
// source of truth.
//
// Architecture:
//   - `Lance::gen_step` (Module 13) is the single-call denoise step:
//     it consumes `latent, timestep, mrope, &mut KvCache` and returns
//     `v_pred` (velocity at that timestep). It is documented READ-ONLY on
//     the cache (it hard-codes `update_cache=false`).
//   - C5 wraps `gen_step` in an Euler integrator over the shifted
//     flow-matching schedule (`timestep_schedule`, Module 2), and adds
//     a classifier-free guidance (CFG) combine over two parallel
//     `gen_step` calls — one against the cond KV-cache, one against the
//     uncond KV-cache.
//
// CFG combine. Python `lance.py:1768`:
//     v_t_ = cfg_text_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t)
// where `v_t` is the COND output and `cfg_text_v_t` is the UNCOND output.
// In our variable names:
//     v = v_uncond + cfg_scale * (v_cond - v_uncond)
//
// Sign convention (per F3-followup fix documented on `denoise_step`):
//   - `timestep_schedule` returns a DECREASING schedule (`t[0] = 1.0`,
//     `t[N] = 0.0`), so `dts[i] = t[i] - t[i+1]` is POSITIVE.
//   - `denoise_step(x, v, dt)` computes `x - dt * v` and matches
//     `lance.py:1759` (`x_t = x_t - v_t * dts[i]`). C5 passes positive
//     `dts[i]` and lets `denoise_step` handle the subtraction.
//
// Memory plan (CONTEXT.md: remote-no-reboot). C5 does NOT load weights and
// does NOT instantiate a VAE — the staged-execution discipline is the
// caller's (C7 CLI bin) responsibility. C5 is a pure tensor algorithm over
// an already-resident Lance model.

/// CFG combine: `out = uncond + scale * (cond - uncond)`.
///
/// Equivalent to `(1 - scale) * uncond + scale * cond` algebraically, but
/// we write it the first way to mirror Python `lance.py:1768` exactly.
/// `scale = 1.0` collapses to `out = cond` (no guidance); `scale = 0.0`
/// collapses to `out = uncond`. The typical Lance default is `4.0`.
///
/// Shapes of `cond` and `uncond` must match; output shape == input shape.
/// Dtype must also match (no implicit casts).
pub fn combine_cfg(uncond: &Tensor, cond: &Tensor, scale: f32) -> Result<Tensor> {
    if uncond.shape().dims() != cond.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "combine_cfg: uncond shape {:?} != cond shape {:?}",
            uncond.shape().dims(),
            cond.shape().dims()
        )));
    }
    if uncond.dtype() != cond.dtype() {
        return Err(Error::InvalidInput(format!(
            "combine_cfg: uncond dtype {:?} != cond dtype {:?}",
            uncond.dtype(),
            cond.dtype()
        )));
    }
    // (cond - uncond) * scale, then + uncond. Mirrors Python line 1768
    // operator order.
    let diff = cond.sub(uncond)?;
    let scaled = diff.mul_scalar(scale)?;
    uncond.add(&scaled)
}

impl Lance {
    /// Run the full flow-matching denoise loop with classifier-free guidance.
    ///
    /// Inputs:
    ///   - `cond_cache`: prefilled with the cond (positive prompt) text context
    ///     via [`Lance::prefill_text_context`]. NOT modified by this call.
    ///   - `uncond_cache`: prefilled with the uncond (typically "" / null
    ///     prompt) text context. NOT modified by this call.
    ///   - `initial_noise`: `[B=1, C=patch_latent_dim, T=1, H, W]` BF16
    ///     starting latent at `t = 1.0`.
    ///   - `mrope`: precomputed at sufficient `max_pos` to cover both prompts
    ///     plus the image grid (`max(L_text_cond, L_text_uncond) + 1000 +
    ///     T*H*W`).
    ///   - `num_steps`: e.g. 30 (shell) / 24 (code).
    ///   - `shift`: flow-matching shift (3.5 shell / 4.0 code).
    ///   - `cfg_scale`: classifier-free guidance scale (e.g. 4.0).
    ///
    /// Returns the final denoised latent at `t = 0`, shape `[B, C, T, H, W]`.
    ///
    /// **Caches are cloned internally.** `gen_step` requires a `&mut KvCache`
    /// signature (Module 13), but with `update_cache=false` it is read-only
    /// in effect. Cloning the cache is `O(num_layers)` `Arc` bumps on the
    /// resident CUDA storage (see `KvCache`'s Clone-semantics note); no
    /// device copy. This decouples the loop from any caller-held borrows on
    /// `cond_cache` / `uncond_cache`.
    pub fn denoise_loop(
        &self,
        cond_cache: &KvCache,
        uncond_cache: &KvCache,
        initial_noise: &Tensor,
        mrope: &MropeFreqs,
        num_steps: usize,
        shift: f32,
        cfg_scale: f32,
    ) -> Result<Tensor> {
        if num_steps == 0 {
            return Err(Error::InvalidInput(
                "Lance::denoise_loop: num_steps must be > 0".into(),
            ));
        }

        // ---- 1. Build shifted timestep schedule and host-side dt vector. ----
        //
        // `timestep_schedule` returns BF16 `[num_steps+1]`. We read it to host
        // F32 once so each step has cheap scalar access for both the t-value
        // (constructed as a fresh per-step `[1]` tensor) and the dt-scalar
        // passed into `denoise_step`. Mirrors Python `lance.py:1691`'s
        // `dts = timesteps[:-1] - timesteps[1:]`.
        let timesteps = timestep_schedule(num_steps, shift, &self.config.device)?;
        // `to_vec` on a BF16 tensor: flame-core handles the BF16→F32
        // unpack inside `Tensor::to_vec` (bf16_u16 path) and returns F32 host
        // data. We use F32 host for the schedule throughout to keep the
        // subtraction `t[i] - t[i+1]` numerically clean — BF16 deltas near
        // `t=0` can underflow.
        let timesteps_host: Vec<f32> = timesteps.to_dtype(DType::F32)?.to_vec()?;
        if timesteps_host.len() != num_steps + 1 {
            return Err(Error::InvalidInput(format!(
                "Lance::denoise_loop: timestep_schedule returned {} values, expected {}",
                timesteps_host.len(),
                num_steps + 1
            )));
        }
        let dts: Vec<f32> = (0..num_steps)
            .map(|i| timesteps_host[i] - timesteps_host[i + 1])
            .collect();

        // ---- 2. Clone caches into local mutable copies. ----
        //
        // `gen_step` takes `&mut KvCache` (Module 13 signature) even though
        // it is read-only when `update_cache=false`. Cloning is cheap (Arc
        // bumps on resident K/V tensors). Avoids forcing the caller to give
        // up unique borrows on its prefilled caches.
        let mut cond_cache_local = cond_cache.clone();
        let mut uncond_cache_local = uncond_cache.clone();

        // ---- 3. Initial latent. ----
        let mut x_t = initial_noise.clone();

        // ---- 4. Per-step Euler integration loop. ----
        for i in 0..num_steps {
            let t = timesteps_host[i];
            // Build a `[1]` BF16 timestep tensor for `gen_step`. F32 host →
            // F32 device → BF16 cast: identical pattern to the existing
            // tests at `tests_module13::test_gen_step_basic_shape`
            // (around line 4474).
            let t_tensor = Tensor::from_vec(
                vec![t],
                Shape::from_dims(&[1]),
                self.config.device.clone(),
            )?;
            let t_tensor = if t_tensor.dtype() != self.config.dtype {
                t_tensor.to_dtype(self.config.dtype)?
            } else {
                t_tensor
            };

            // CFG: two parallel gen_step calls. The cache values are
            // unchanged across both calls (update_cache=false inside
            // gen_step per Module 13).
            let v_cond = self.gen_step(&x_t, &t_tensor, mrope, &mut cond_cache_local)?;
            let v_uncond = self.gen_step(&x_t, &t_tensor, mrope, &mut uncond_cache_local)?;

            // CFG combine. Mirrors `lance.py:1768` exactly.
            let v = combine_cfg(&v_uncond, &v_cond, cfg_scale)?;

            // Euler step `x_{t+1} = x_t - dt * v`. `denoise_step` enforces
            // shape parity and applies the subtraction.
            x_t = denoise_step(&x_t, &v, dts[i])?;

            log::info!(
                "[Lance denoise] step {}/{}, t={:.4}, dt={:.4}",
                i + 1,
                num_steps,
                t,
                dts[i]
            );
        }
        Ok(x_t)
    }

    /// Convenience: single-prompt entry point.
    ///
    /// Equivalent to:
    ///   1. Allocate a fresh `KvCache`, prefill with `cond_tokens`.
    ///   2. Allocate a fresh `KvCache`, prefill with `uncond_tokens`.
    ///   3. Run [`Lance::denoise_loop`] with both caches and config-supplied
    ///      defaults (`num_inference_steps`, `timestep_shift`,
    ///      `cfg_text_scale`).
    ///
    /// Use this when you want one-call T2I and don't need to share caches
    /// across multiple latents. For multi-sample-per-prompt batching, call
    /// `prefill_text_context` once each then call `denoise_loop` per
    /// sample — the caches are cheap to share across calls (see Clone
    /// semantics note).
    pub fn t2i_with_cfg(
        &self,
        cond_tokens: &Tensor,
        uncond_tokens: &Tensor,
        initial_noise: &Tensor,
        mrope: &MropeFreqs,
    ) -> Result<Tensor> {
        let mut cond_cache = self.new_kv_cache();
        self.prefill_text_context(cond_tokens, mrope, &mut cond_cache)?;
        let mut uncond_cache = self.new_kv_cache();
        self.prefill_text_context(uncond_tokens, mrope, &mut uncond_cache)?;
        self.denoise_loop(
            &cond_cache,
            &uncond_cache,
            initial_noise,
            mrope,
            self.config.num_inference_steps,
            self.config.timestep_shift,
            self.config.cfg_text_scale,
        )
    }
}

#[cfg(test)]
mod tests_module14 {
    use super::*;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device 0")
    }

    /// 2-layer config — full 36 layers would allocate too much memory just
    /// for C5 shape tests. Mirrors `tests_module13::small_cfg`.
    fn small_cfg() -> LanceConfig {
        let mut cfg = LanceConfig::default_3b(dev());
        cfg.num_hidden_layers = 2;
        cfg
    }

    /// Max-abs difference between two BF16 tensors of identical shape.
    /// Reads both to F32 host. Test-only helper.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let av = a.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = b.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(av.len(), bv.len(), "tensor numel mismatch");
        av.iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn test_combine_cfg_basic() {
        let d = dev();
        // uncond = ones, cond = twos, scale = 4.0
        //   out = 1 + 4 * (2 - 1) = 5
        let uncond = Tensor::from_vec(
            vec![1.0f32, 1.0, 1.0, 1.0],
            Shape::from_dims(&[2, 2]),
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let cond = Tensor::from_vec(
            vec![2.0f32, 2.0, 2.0, 2.0],
            Shape::from_dims(&[2, 2]),
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = combine_cfg(&uncond, &cond, 4.0).unwrap();
        let out_host: Vec<f32> = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for v in out_host {
            assert!((v - 5.0).abs() < 1e-2, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn test_combine_cfg_scale_one_returns_cond() {
        // scale = 1.0 → out = cond exactly.
        let d = dev();
        let uncond = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 0.4],
            Shape::from_dims(&[4]),
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let cond = Tensor::from_vec(
            vec![0.7f32, 0.6, 0.5, 0.4],
            Shape::from_dims(&[4]),
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = combine_cfg(&uncond, &cond, 1.0).unwrap();
        let diff = max_abs_diff(&out, &cond);
        // BF16 round-trip + sub/add chain; ~1e-2 is tight enough.
        assert!(diff < 1e-2, "scale=1.0 should return cond, max_abs_diff={diff}");
    }

    #[test]
    fn test_denoise_loop_shape_preserved() {
        let cfg = small_cfg();
        let c = cfg.patch_latent_dim();
        let d = cfg.device.clone();
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();
        // Empty caches — exercises the cache.seq_len(0) == 0 prefix-len
        // branch inside gen_step. Test pins shape preservation only.
        let cond_cache = lance.new_kv_cache();
        let uncond_cache = lance.new_kv_cache();
        let initial = Tensor::randn_seeded(
            Shape::from_dims(&[1, c, 1, 4, 4]),
            0.0,
            1.0,
            42,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = lance
            .denoise_loop(&cond_cache, &uncond_cache, &initial, &mrope, 3, 4.0, 4.0)
            .unwrap();
        assert_eq!(
            out.shape().dims(),
            initial.shape().dims(),
            "denoise_loop must preserve latent shape"
        );
    }

    #[test]
    fn test_t2i_with_cfg_shape_preserved() {
        let mut cfg = small_cfg();
        cfg.num_inference_steps = 3; // speed up
        let c = cfg.patch_latent_dim();
        let d = cfg.device.clone();
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();
        // Use f32 token IDs (`Tensor::from_vec` is F32-only; `gen_step`'s
        // path casts to I32 inside `embed_text_tokens`). Token values
        // are well within vocab_size=151936.
        let cond_tokens = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            Shape::from_dims(&[5]),
            d.clone(),
        )
        .unwrap();
        let uncond_tokens = Tensor::from_vec(
            vec![1.0f32, 2.0],
            Shape::from_dims(&[2]),
            d.clone(),
        )
        .unwrap();
        let initial = Tensor::randn_seeded(
            Shape::from_dims(&[1, c, 1, 4, 4]),
            0.0,
            1.0,
            42,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = lance
            .t2i_with_cfg(&cond_tokens, &uncond_tokens, &initial, &mrope)
            .unwrap();
        assert_eq!(
            out.shape().dims(),
            initial.shape().dims(),
            "t2i_with_cfg must preserve latent shape"
        );
    }

    #[test]
    fn test_denoise_loop_cfg_scale_affects_output() {
        // Same inputs, two different cfg scales → different outputs.
        // Pins that cfg actually mixes cond + uncond (i.e. we don't silently
        // throw away one of the two gen_step calls).
        let cfg = small_cfg();
        let c = cfg.patch_latent_dim();
        let d = cfg.device.clone();
        let lance = Lance::new_random(Arc::new(cfg)).unwrap();
        let mrope = lance.precompute_mrope(2048, DType::BF16).unwrap();

        // Prefill DIFFERENT cond vs uncond contexts so the two gen_step
        // calls produce different velocities — otherwise cfg_scale would
        // have no effect (cond - uncond = 0).
        let mut cond_cache = lance.new_kv_cache();
        let cond_tokens = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            Shape::from_dims(&[5]),
            d.clone(),
        )
        .unwrap();
        lance
            .prefill_text_context(&cond_tokens, &mrope, &mut cond_cache)
            .unwrap();

        let mut uncond_cache = lance.new_kv_cache();
        let uncond_tokens = Tensor::from_vec(
            vec![10.0f32, 11.0],
            Shape::from_dims(&[2]),
            d.clone(),
        )
        .unwrap();
        lance
            .prefill_text_context(&uncond_tokens, &mrope, &mut uncond_cache)
            .unwrap();

        let initial = Tensor::randn_seeded(
            Shape::from_dims(&[1, c, 1, 4, 4]),
            0.0,
            1.0,
            42,
            d.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let out_low = lance
            .denoise_loop(&cond_cache, &uncond_cache, &initial, &mrope, 3, 4.0, 1.0)
            .unwrap();
        let out_high = lance
            .denoise_loop(&cond_cache, &uncond_cache, &initial, &mrope, 3, 4.0, 7.0)
            .unwrap();
        let diff = max_abs_diff(&out_low, &out_high);
        assert!(
            diff > 1e-2,
            "cfg=1.0 vs cfg=7.0 should produce different latents; max_abs_diff={diff}"
        );
    }
}
