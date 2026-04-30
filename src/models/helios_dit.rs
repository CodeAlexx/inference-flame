//! Helios-Distilled 14B video DiT — pure-Rust BF16 inference port.
//!
//! Helios.2 (this commit): single-block forward + PyTorch parity test.
//! Architecture: 40-layer single-stream DiT (Wan 2.1 14B fine-tune) with three
//! Helios-specific extras documented in `HANDOFF_2026-04-29_HELIOS_AUDIT.md`:
//! - 6-channel modulation via per-block learnable `scale_shift_table` + temb
//! - FP32 LayerNorm on hidden states (cast in/out)
//! - GELU-approximate (tanh-based) FFN
//! - `qk_norm = "rms_norm_across_heads"` — RMSNorm over the FULL inner_dim
//!   (not per-head head_dim), with affine weight shape `(inner_dim,)`
//! - Optional `guidance_cross_attn` history split (skipped in this port —
//!   Distilled config can run without; future Helios.6 wires history)
//!
//! References:
//!   diffusers/models/transformers/transformer_helios.py
//!     HeliosTransformerBlock.forward (374-494)
//!     HeliosAttnProcessor (99-156)
//!     apply_rotary_emb_transposed (52-61)
//!
//! RoPE: Helios's `apply_rotary_emb_transposed` reduces to standard
//! interleaved-pair complex multiply because `freqs.repeat_interleave(2,
//! dim=0)` makes cos[2i] == cos[2i+1] (and same for sin). Extracting every
//! other element recovers the (head_dim/2)-length unique cos/sin we feed to
//! flame's `rope_fused_bf16`.

use flame_core::attention::sdpa;
use flame_core::bf16_ops::rope_fused_bf16;
use flame_core::cuda_ops_bf16::rms_norm_bf16;
use flame_core::layer_norm::layer_norm;
use flame_core::ops::fused_inference::fused_linear3d_native;
use flame_core::{DType, Error, Result, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HeliosConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub ffn_dim: usize,
    pub text_dim: usize,
    pub rope_dim: [usize; 3],
    pub rope_theta: f32,
    pub freq_dim: usize,
    pub patch_size: [usize; 3],
    pub in_channels: usize,
    pub out_channels: usize,
    pub eps: f32,
    pub cross_attn_norm: bool,
    pub guidance_cross_attn: bool,
    pub is_amplify_history: bool,
}

impl HeliosConfig {
    /// Hard-coded defaults from `transformer/config.json` of
    /// `BestWishYsh/Helios-Distilled` checkpoint.
    pub fn helios_distilled_default() -> Self {
        Self {
            num_layers: 40,
            num_heads: 40,
            head_dim: 128,
            hidden_size: 5120,
            ffn_dim: 13824,
            text_dim: 4096,
            rope_dim: [44, 42, 42],
            rope_theta: 10_000.0,
            freq_dim: 256,
            patch_size: [1, 2, 2],
            in_channels: 16,
            out_channels: 16,
            eps: 1e-6,
            cross_attn_norm: true,
            guidance_cross_attn: true,
            is_amplify_history: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Block weights
// ---------------------------------------------------------------------------

/// Per-block weight bundle for Helios.
///
/// Cross-attention's `to_k`/`to_v` consume the encoder hidden state at full
/// `hidden_size` (the top-level `text_embedder` projects `text_dim →
/// hidden_size` once). The block's own `to_k`/`to_v` are the cross-attn
/// projections — there is no separate `add_k_proj`/`add_v_proj` because
/// `added_kv_proj_dim = None` in the production config.
pub struct HeliosBlock {
    /// (1, 6, dim) BF16 — added to `temb` BEFORE chunking the 6 modulation
    /// channels. Single broadcast across batch + sequence.
    pub scale_shift_table: Tensor,
    /// `norm1` (pre-self-attn) is FP32LayerNorm with `elementwise_affine=False`.
    /// No weights to store.
    /// `norm2` (pre-cross-attn) is FP32LayerNorm with `elementwise_affine=True`
    /// when `cross_attn_norm=True`.
    pub norm2_w: Tensor, // (dim,)
    pub norm2_b: Tensor,
    /// `norm3` (pre-FFN) — same as norm1 (no affine, no weights).
    /// Self-attention (attn1)
    pub a1_to_q_w: Tensor, // (dim, dim)
    pub a1_to_q_b: Tensor,
    pub a1_to_k_w: Tensor,
    pub a1_to_k_b: Tensor,
    pub a1_to_v_w: Tensor,
    pub a1_to_v_b: Tensor,
    pub a1_to_out_w: Tensor,
    pub a1_to_out_b: Tensor,
    /// `norm_q`, `norm_k`: RMSNorm across the FULL inner_dim with weight
    /// shape `(inner_dim,)`. `qk_norm="rms_norm_across_heads"`.
    pub a1_norm_q_w: Tensor, // (inner_dim,)
    pub a1_norm_k_w: Tensor,
    /// Cross-attention (attn2) — same projection layout as attn1, applied
    /// to `(query=hidden_states, key=value=encoder_hidden_states)`.
    pub a2_to_q_w: Tensor,
    pub a2_to_q_b: Tensor,
    pub a2_to_k_w: Tensor,
    pub a2_to_k_b: Tensor,
    pub a2_to_v_w: Tensor,
    pub a2_to_v_b: Tensor,
    pub a2_to_out_w: Tensor,
    pub a2_to_out_b: Tensor,
    pub a2_norm_q_w: Tensor,
    pub a2_norm_k_w: Tensor,
    /// FFN: `Linear(dim, ffn_dim)` → GELU-approximate → `Linear(ffn_dim, dim)`.
    /// FeedForward(activation_fn="gelu-approximate") layout puts `GELU` at
    /// `net[0]` (with its OWN proj inside) and the down `Linear` at `net[2]`.
    /// Diffusers's `GELU(dim, ffn_dim, approximate="tanh")` IS itself a Linear
    /// followed by activation — so `net[0].proj` is the up projection.
    pub ffn_up_w: Tensor,   // (ffn_dim, dim)
    pub ffn_up_b: Tensor,
    pub ffn_down_w: Tensor, // (dim, ffn_dim)
    pub ffn_down_b: Tensor,
}

impl HeliosBlock {
    /// Single-block forward (mirrors `HeliosTransformerBlock.forward` lines
    /// 424-494). For the parity-test fixture we pass:
    /// - `temb`: (B, 6, dim) — the 3D path. The 4D per-token path (used by
    ///   the multi-term-memory feature) is a future task.
    /// - `freqs_cis`: (B, S, 2*head_dim) — already flattened+transposed by the
    ///   caller from HeliosRotaryPosEmbed's (B, 2*head_dim, T, H, W) output.
    /// - `encoder_hidden_states`: (B, S_txt, hidden_size) — already projected
    ///   to hidden_size by the model's top-level text_embedder.
    pub fn forward(
        &self,
        hidden: &Tensor,        // (B, S, D)
        encoder: &Tensor,       // (B, S_txt, D)
        temb: &Tensor,          // (B, 6, D) global  OR  (B, S, 6, D) per-token
        freqs_cis: &Tensor,     // (B, S, 2*head_dim)
        cfg: &HeliosConfig,
    ) -> Result<Tensor> {
        let dims = hidden.shape().dims();
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let h = cfg.num_heads;
        let hd = cfg.head_dim;

        // ---- Modulation: scale_shift_table (1,6,D) + temb -> 6 channels
        // 3-D temb path (B,6,D): chunk to 6 tensors of (B, 1, D), broadcast over S.
        // 4-D temb path (B,S,6,D): scale_shift_table.unsqueeze(0) → (1,1,6,D)
        //   added → (B,S,6,D), chunk(6, dim=2) → 6 of (B,S,1,D), squeeze(2) → (B,S,D).
        let temb_ndim = temb.shape().dims().len();
        let (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa) = if temb_ndim == 4 {
            let sst_4d = self.scale_shift_table.unsqueeze(0)?; // (1,1,6,D)
            let sum = sst_4d.add(temb)?; // (B,S,6,D)
            let cs = sum.chunk(6, 2)?;
            (
                cs[0].squeeze(Some(2))?,
                cs[1].squeeze(Some(2))?,
                cs[2].squeeze(Some(2))?,
                cs[3].squeeze(Some(2))?,
                cs[4].squeeze(Some(2))?,
                cs[5].squeeze(Some(2))?,
            )
        } else {
            let sum = self.scale_shift_table.add(temb)?; // (B,6,D)
            let cs = sum.chunk(6, 1)?;
            (
                cs[0].clone(),
                cs[1].clone(),
                cs[2].clone(),
                cs[3].clone(),
                cs[4].clone(),
                cs[5].clone(),
            )
        };
        let shift_msa = &shift_msa;
        let scale_msa = &scale_msa;
        let gate_msa = &gate_msa;
        let c_shift_msa = &c_shift_msa;
        let c_scale_msa = &c_scale_msa;
        let c_gate_msa = &c_gate_msa;

        // ---- 1. Self-attention -----------------------------------------
        // norm1: FP32 LayerNorm, no affine, eps=1e-6
        let normed = layer_norm(hidden, &[d], None, None, cfg.eps)?;
        let scale1 = scale_msa.add_scalar(1.0)?;
        let normed = normed.mul(&scale1)?.add(shift_msa)?;

        let attn1_out = self.attention(
            &normed,
            None, // self-attn: no encoder
            Some(freqs_cis),
            &self.a1_to_q_w,
            &self.a1_to_q_b,
            &self.a1_to_k_w,
            &self.a1_to_k_b,
            &self.a1_to_v_w,
            &self.a1_to_v_b,
            &self.a1_to_out_w,
            &self.a1_to_out_b,
            &self.a1_norm_q_w,
            &self.a1_norm_k_w,
            cfg,
        )?;

        // residual: x + gate_msa * attn (NO `tanh` like Nucleus — direct multiply)
        let gated = gate_msa.mul(&attn1_out)?;
        let x = hidden.add(&gated)?;

        // ---- 2. Cross-attention ---------------------------------------
        // norm2: FP32 LayerNorm, AFFINE (cross_attn_norm=True)
        // F32 norm with weight+bias for affine
        let normed2 = layer_norm(&x, &[d], Some(&self.norm2_w), Some(&self.norm2_b), cfg.eps)?;
        let attn2_out = self.attention(
            &normed2,
            Some(encoder),
            None, // no rotary on cross-attn
            &self.a2_to_q_w,
            &self.a2_to_q_b,
            &self.a2_to_k_w,
            &self.a2_to_k_b,
            &self.a2_to_v_w,
            &self.a2_to_v_b,
            &self.a2_to_out_w,
            &self.a2_to_out_b,
            &self.a2_norm_q_w,
            &self.a2_norm_k_w,
            cfg,
        )?;
        // residual: x + attn (NO gate, NO modulation)
        let x = x.add(&attn2_out)?;

        // ---- 3. Feed-forward (GELU-approximate) -----------------------
        // norm3: FP32 LayerNorm, no affine
        let normed3 = layer_norm(&x, &[d], None, None, cfg.eps)?;
        let scale2 = c_scale_msa.add_scalar(1.0)?;
        let normed3 = normed3.mul(&scale2)?.add(c_shift_msa)?;

        // Up projection
        let h1 = fused_linear3d_native(&normed3, &self.ffn_up_w, Some(&self.ffn_up_b))?;
        // GELU-approximate (tanh) — flame's gelu_bf16 IS the tanh approx
        let h1 = h1.gelu()?;
        // Down projection
        let ff_out = fused_linear3d_native(&h1, &self.ffn_down_w, Some(&self.ffn_down_b))?;

        // residual: x + c_gate_msa * ff_out
        let gated = c_gate_msa.mul(&ff_out)?;
        let _ = (b, s, h, hd); // silence unused if all paths used parameters
        x.add(&gated)
    }

    /// Generic multi-head attention that handles both self- and cross-attn.
    /// `qk_norm="rms_norm_across_heads"` normalizes Q/K BEFORE reshape to heads
    /// using a weight of shape `(inner_dim,)`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        x: &Tensor,                  // (B, S_q, D) — query source
        encoder: Option<&Tensor>,    // (B, S_kv, D) — key/value source for cross-attn; None for self
        freqs_cis: Option<&Tensor>,  // (B, S, 2*head_dim) for self-attn; None for cross-attn
        to_q_w: &Tensor,
        to_q_b: &Tensor,
        to_k_w: &Tensor,
        to_k_b: &Tensor,
        to_v_w: &Tensor,
        to_v_b: &Tensor,
        to_out_w: &Tensor,
        to_out_b: &Tensor,
        norm_q_w: &Tensor,
        norm_k_w: &Tensor,
        cfg: &HeliosConfig,
    ) -> Result<Tensor> {
        let kv_src = encoder.unwrap_or(x);
        let q_dims = x.shape().dims();
        let kv_dims = kv_src.shape().dims();
        let (b, s_q, d) = (q_dims[0], q_dims[1], q_dims[2]);
        let s_kv = kv_dims[1];
        let h = cfg.num_heads;
        let hd = cfg.head_dim;

        // Q/K/V projections (with bias)
        let q = fused_linear3d_native(x, to_q_w, Some(to_q_b))?;
        let k = fused_linear3d_native(kv_src, to_k_w, Some(to_k_b))?;
        let v = fused_linear3d_native(kv_src, to_v_w, Some(to_v_b))?;

        // qk_norm = "rms_norm_across_heads": RMSNorm over the FULL inner_dim
        // BEFORE reshape to heads, weight shape (inner_dim,)
        let q_flat = q.reshape(&[b * s_q, d])?;
        let k_flat = k.reshape(&[b * s_kv, d])?;
        let q = rms_norm_bf16(&q_flat, Some(norm_q_w), cfg.eps)?;
        let k = rms_norm_bf16(&k_flat, Some(norm_k_w), cfg.eps)?;
        let q = q.reshape(&[b, s_q, h, hd])?;
        let k = k.reshape(&[b, s_kv, h, hd])?;
        let v = v.reshape(&[b, s_kv, h, hd])?;

        // RoPE on Q and K (self-attn only)
        let (q, k) = if let Some(fc) = freqs_cis {
            let (cos, sin) = freqs_cis_to_cos_sin(fc, hd)?;
            // (B, S, H, D) -> (B, H, S, D) for rope_fused_bf16
            let q4 = q.permute(&[0, 2, 1, 3])?.contiguous()?;
            let k4 = k.permute(&[0, 2, 1, 3])?.contiguous()?;
            let q4 = rope_fused_bf16(&q4, &cos, &sin)?;
            let k4 = rope_fused_bf16(&k4, &cos, &sin)?;
            (q4, k4)
        } else {
            let q4 = q.permute(&[0, 2, 1, 3])?.contiguous()?;
            let k4 = k.permute(&[0, 2, 1, 3])?.contiguous()?;
            (q4, k4)
        };
        let v = v.permute(&[0, 2, 1, 3])?.contiguous()?;

        // SDPA — (B, H, S, head_dim) all heads same count, no GQA
        let attn = sdpa(&q, &k, &v, None)?;
        // (B, H, S_q, head_dim) -> (B, S_q, H*head_dim=D)
        let attn = attn.permute(&[0, 2, 1, 3])?.contiguous()?;
        let attn = attn.reshape(&[b, s_q, d])?;

        // Output projection (with bias)
        fused_linear3d_native(&attn, to_out_w, Some(to_out_b))
    }
}

/// Convert Helios's `freqs_cis` `(B, S, 2*head_dim)` into flame's
/// `(1, 1, S, head_dim/2)` cos/sin tensors.
///
/// Diffusers's `apply_rotary_emb_transposed` works because
/// `freqs.repeat_interleave(2, dim=0)` makes pairs of equal cos/sin values
/// adjacent. The math reduces to standard interleaved-pair complex multiply:
///   out[2i]   = x[2i] * cos[2i] - x[2i+1] * sin[2i+1]
///   out[2i+1] = x[2i] * sin[2i+1] + x[2i+1] * cos[2i]
/// where cos[2i] == cos[2i+1] and sin[2i] == sin[2i+1] (post-repeat-2). So
/// we can extract every-other element to recover the unique (head_dim/2)
/// length cos/sin. The model's `freqs_cis.flatten(2).transpose(1, 2)`
/// produces a layout where the last dim 2*head_dim splits into [cos | sin],
/// each head_dim long, with the repeat-2 pattern.
fn freqs_cis_to_cos_sin(freqs_cis: &Tensor, head_dim: usize) -> Result<(Tensor, Tensor)> {
    let dims = freqs_cis.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "freqs_cis_to_cos_sin: expected 3-D (B, S, 2*head_dim), got {dims:?}"
        )));
    }
    let (b, s, two_d) = (dims[0], dims[1], dims[2]);
    if two_d != 2 * head_dim {
        return Err(Error::InvalidOperation(format!(
            "freqs_cis_to_cos_sin: last dim {two_d} != 2*head_dim={}",
            2 * head_dim
        )));
    }
    if b != 1 {
        return Err(Error::InvalidOperation(format!(
            "freqs_cis_to_cos_sin: only B=1 supported (got {b}); generalise via reshape if needed"
        )));
    }
    // chunk along last dim: cos = freqs[..., :head_dim], sin = freqs[..., head_dim:]
    let chunks = freqs_cis.chunk(2, 2)?;
    let cos_full = &chunks[0]; // (B, S, head_dim) with repeat-2 pattern
    let sin_full = &chunks[1];

    // Extract every-other element along last dim → (B, S, head_dim/2).
    // Use reshape trick: (B, S, head_dim) → (B, S, head_dim/2, 2) → take [..., 0]
    let half = head_dim / 2;
    let cos_pair = cos_full.reshape(&[b, s, half, 2])?;
    let sin_pair = sin_full.reshape(&[b, s, half, 2])?;
    let cos = cos_pair.narrow(3, 0, 1)?.reshape(&[1, 1, s, half])?;
    let sin = sin_pair.narrow(3, 0, 1)?.reshape(&[1, 1, s, half])?;
    Ok((cos.to_dtype(DType::BF16)?, sin.to_dtype(DType::BF16)?))
}

// ---------------------------------------------------------------------------
// Top-level DiT (Helios.3 — full forward, no history, no multi-term-memory)
// ---------------------------------------------------------------------------

/// Top-level Helios DiT (assembly only — Helios.3 scope).
///
/// Real-weight runtime + BlockOffloader streaming arrives in Helios.5.
/// This struct holds all weights resident — used by parity tests with toy
/// configs.
pub struct HeliosDit {
    pub config: HeliosConfig,
    pub blocks: Vec<HeliosBlock>,
    /// `patch_embedding` Conv3d viewed as a linear: weight reshaped from
    /// `(out=dim, in=in_channels, kT, kH, kW)` to `(dim, in_channels*kT*kH*kW)`,
    /// applied per spatio-temporal patch. Bias `(dim,)`.
    pub patch_embed_w: Tensor, // (dim, in_channels * patch_t * patch_h * patch_w)
    pub patch_embed_b: Tensor,
    /// `condition_embedder.time_embedder.linear_{1,2}` (TimestepEmbedding 2-Linear MLP).
    pub time_l1_w: Tensor, // (dim, freq_dim)
    pub time_l1_b: Tensor,
    pub time_l2_w: Tensor, // (dim, dim)
    pub time_l2_b: Tensor,
    /// `condition_embedder.time_proj` Linear (dim → 6*dim).
    pub time_proj_w: Tensor, // (6*dim, dim)
    pub time_proj_b: Tensor,
    /// `condition_embedder.text_embedder` PixArtAlphaTextProjection — two
    /// Linears with `gelu_tanh` between.
    pub text_l1_w: Tensor, // (dim, text_dim)
    pub text_l1_b: Tensor,
    pub text_l2_w: Tensor, // (dim, dim)
    pub text_l2_b: Tensor,
    /// `norm_out` HeliosOutputNorm — own (1, 2, dim) scale_shift_table.
    pub norm_out_sst: Tensor, // (1, 2, dim)
    /// `proj_out` Linear (dim → out_channels * prod(patch_size)).
    pub proj_out_w: Tensor,
    pub proj_out_b: Tensor,
}

impl HeliosDit {
    /// Convert a scalar `timestep` to the sinusoidal embedding the Helios
    /// `Timesteps` module produces. Matches diffusers's `get_timestep_embedding`
    /// with `flip_sin_to_cos=True`, `downscale_freq_shift=0`, default
    /// `max_period=10000`, `scale=1.0`.
    fn time_proj(&self, timestep: &Tensor) -> Result<Tensor> {
        let freq_dim = self.config.freq_dim;
        let half = freq_dim / 2;
        let max_period = 10_000.0_f64;
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(
            freq_data,
            Shape::from_dims(&[1, half]),
            timestep.device().clone(),
        )?;
        let t_f32 = timestep.to_dtype(DType::F32)?;
        let args = t_f32.unsqueeze(1)?.matmul(&freqs)?; // (B, half)
        let cos_part = args.cos()?;
        let sin_part = args.sin()?;
        // flip_sin_to_cos=True → final order is (cos, sin)
        let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;
        emb.to_dtype(DType::BF16)
    }

    fn condition_embedder(
        &self,
        timestep: &Tensor,
        encoder: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // time_proj → time_embedder(MLP) → temb
        let proj = self.time_proj(timestep)?; // (B, freq_dim) BF16
        let proj_3d = proj.unsqueeze(1)?;
        let h1 = fused_linear3d_native(&proj_3d, &self.time_l1_w, Some(&self.time_l1_b))?;
        let h1 = h1.silu()?;
        let temb = fused_linear3d_native(&h1, &self.time_l2_w, Some(&self.time_l2_b))?;
        let temb = temb.squeeze(Some(1))?; // (B, dim)

        // timestep_proj: Linear(dim → 6*dim) of silu(temb)
        let temb_silu_3d = temb.silu()?.unsqueeze(1)?;
        let timestep_proj =
            fused_linear3d_native(&temb_silu_3d, &self.time_proj_w, Some(&self.time_proj_b))?;
        let timestep_proj = timestep_proj.squeeze(Some(1))?; // (B, 6*dim)

        // text_embedder: Linear → gelu_tanh → Linear
        let h1 = fused_linear3d_native(encoder, &self.text_l1_w, Some(&self.text_l1_b))?;
        let h1 = h1.gelu()?; // gelu_tanh approximate path
        let encoder_proj =
            fused_linear3d_native(&h1, &self.text_l2_w, Some(&self.text_l2_b))?;

        Ok((temb, timestep_proj, encoder_proj))
    }

    /// Build per-position 3D RoPE. Output shape `(B, F*H*W, 2*head_dim)` with
    /// the layout the block expects (post `flatten(2).transpose(1, 2)`).
    fn build_rope(
        &self,
        f: usize,
        h: usize,
        w: usize,
        device: std::sync::Arc<flame_core::CudaDevice>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let theta = cfg.rope_theta as f64;

        // Per-axis half counts.
        let halves = [
            cfg.rope_dim[0] / 2,
            cfg.rope_dim[1] / 2,
            cfg.rope_dim[2] / 2,
        ];
        let inv_freqs: Vec<Vec<f32>> = (0..3)
            .map(|axis| {
                let half = halves[axis];
                let dim = cfg.rope_dim[axis];
                (0..half)
                    .map(|k| (1.0 / theta.powf(2.0 * k as f64 / dim as f64)) as f32)
                    .collect()
            })
            .collect();

        // For each token at (t, y, x), build 2*head_dim values:
        //   first half: cos at axis t, axis y, axis x — concat'd, each with
        //     repeat-2 (cos[2i] = cos[2i+1] = cos(freq_k * pos))
        //   second half: sin same way.
        // Total per axis after repeat-2: rope_dim[axis] elements (cos), same for sin.
        // First-half (cos) layout: [cos_t (D_t), cos_y (D_y), cos_x (D_x)] = head_dim
        // Second-half (sin) same.
        let s = f * h * w;
        let mut out = vec![0.0_f32; s * 2 * head_dim];
        for ti in 0..f {
            for yi in 0..h {
                for xi in 0..w {
                    let token = (ti * h + yi) * w + xi;
                    // cos block
                    let mut col = 0;
                    for axis in 0..3 {
                        let pos = match axis {
                            0 => ti as f32,
                            1 => yi as f32,
                            _ => xi as f32,
                        };
                        let invs = &inv_freqs[axis];
                        for &freq in invs {
                            let ang = pos * freq;
                            let cv = ang.cos();
                            // repeat-2: write to col and col+1
                            out[token * 2 * head_dim + col] = cv;
                            out[token * 2 * head_dim + col + 1] = cv;
                            col += 2;
                        }
                    }
                    // sin block
                    let mut col = head_dim;
                    for axis in 0..3 {
                        let pos = match axis {
                            0 => ti as f32,
                            1 => yi as f32,
                            _ => xi as f32,
                        };
                        let invs = &inv_freqs[axis];
                        for &freq in invs {
                            let ang = pos * freq;
                            let sv = ang.sin();
                            out[token * 2 * head_dim + col] = sv;
                            out[token * 2 * head_dim + col + 1] = sv;
                            col += 2;
                        }
                    }
                }
            }
        }
        let t = Tensor::from_vec(out, Shape::from_dims(&[1, s, 2 * head_dim]), device)?;
        t.to_dtype(DType::BF16)
    }

    /// Patchify: `(B, C_in, F, H, W)` → `(B, F, H/p_h, W/p_w, C_in * p_t * p_h * p_w)` → linear → `(B, S, dim)`.
    /// Equivalent to `Conv3d(in_channels, dim, kernel=patch_size, stride=patch_size)` with no padding.
    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = x.shape().dims();
        let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let pt = cfg.patch_size[0];
        let ph = cfg.patch_size[1];
        let pw = cfg.patch_size[2];
        let f_out = f / pt;
        let h_out = h / ph;
        let w_out = w / pw;
        // (B, C, F, H, W) → (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
        let v = x.reshape(&[b, c, f_out, pt, h_out, ph, w_out, pw])?;
        // permute → (B, F/pt, H/ph, W/pw, C, pt, ph, pw)
        let v = v.permute(&[0, 2, 4, 6, 1, 3, 5, 7])?.contiguous()?;
        // flatten last 4 dims → (B, F/pt, H/ph, W/pw, C*pt*ph*pw)
        let in_per_patch = c * pt * ph * pw;
        let s = f_out * h_out * w_out;
        let v = v.reshape(&[b, s, in_per_patch])?;
        // Linear (out=dim, in=in_per_patch)
        let out = fused_linear3d_native(&v, &self.patch_embed_w, Some(&self.patch_embed_b))?;
        Ok(out)
    }

    /// Inverse of `patchify`: `(B, S, out_ch * pt * ph * pw)` → `(B, out_ch, F, H, W)`.
    fn unpatchify(&self, x: &Tensor, f_grid: usize, h_grid: usize, w_grid: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = x.shape().dims();
        let (b, _s, out_per_patch) = (dims[0], dims[1], dims[2]);
        let pt = cfg.patch_size[0];
        let ph = cfg.patch_size[1];
        let pw = cfg.patch_size[2];
        let out_ch = out_per_patch / (pt * ph * pw);
        // (B, F, H, W, out_ch * pt * ph * pw) → (B, F, H, W, pt, ph, pw, out_ch)
        let v = x.reshape(&[b, f_grid, h_grid, w_grid, pt, ph, pw, out_ch])?;
        // permute → (B, out_ch, F, pt, H, ph, W, pw) i.e. (0, 7, 1, 4, 2, 5, 3, 6)
        let v = v.permute(&[0, 7, 1, 4, 2, 5, 3, 6])?.contiguous()?;
        // flatten (F, pt) → F*pt; (H, ph) → H*ph; (W, pw) → W*pw
        let v = v.reshape(&[b, out_ch, f_grid * pt, h_grid * ph, w_grid * pw])?;
        Ok(v)
    }

    /// Apply HeliosOutputNorm: `LN(x.float()) * (1+scale) + shift` where
    /// `(shift, scale) = chunk(scale_shift_table.unsqueeze(0) + temb.unsqueeze(2), 2, dim=2)`.
    /// `temb` shape: `(B, S, dim)`; `scale_shift_table`: `(1, 2, dim)`.
    fn norm_out(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        // (1,2,D).unsqueeze(0) → (1,1,2,D); temb.unsqueeze(2) → (B,S,1,D); sum → (B,S,2,D)
        let sst_4d = self.norm_out_sst.unsqueeze(0)?;
        let temb_4d = temb.unsqueeze(2)?;
        let sum = sst_4d.add(&temb_4d)?;
        let cs = sum.chunk(2, 2)?;
        let shift = cs[0].squeeze(Some(2))?;
        let scale = cs[1].squeeze(Some(2))?;
        let normed = layer_norm(x, &[d], None, None, cfg.eps)?;
        let scale_p1 = scale.add_scalar(1.0)?;
        let scaled = normed.mul(&scale_p1)?;
        scaled.add(&shift)
    }

    /// One forward pass through the full DiT. No history latents, no
    /// multi-term-memory patches (Helios.3 scope). Mirrors
    /// `HeliosTransformer3DModel.forward` lines 657-820 with the
    /// `latents_history_*=None`, `is_amplify_history=False` simplifications.
    pub fn forward(
        &self,
        hidden_states: &Tensor,        // (B, in_channels, F, H, W)
        encoder_hidden_states: &Tensor, // (B, S_txt, text_dim)
        timestep: &Tensor,             // (B,) or scalar
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = hidden_states.shape().dims();
        let (b, _c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let pt = cfg.patch_size[0];
        let ph = cfg.patch_size[1];
        let pw = cfg.patch_size[2];
        let f_grid = f / pt;
        let h_grid = h / ph;
        let w_grid = w / pw;
        let s = f_grid * h_grid * w_grid;

        // 1. Patchify
        let x = self.patchify(hidden_states)?; // (B, S, dim)

        // 2. Build 3D RoPE (B=1 supported; matches build_rope)
        let _ = b; // future: per-batch frame_indices
        let rotary = self.build_rope(f_grid, h_grid, w_grid, x.device().clone())?; // (1, S, 2*head_dim)

        // 3. Time + text embeddings
        let (temb, timestep_proj, encoder_proj) =
            self.condition_embedder(timestep, encoder_hidden_states)?;
        // timestep_proj: (B, 6*dim) → (B, 6, dim)
        let timestep_proj_3d = timestep_proj.reshape(&[b, 6, cfg.hidden_size])?;
        // Without history: main_repeat_size = original_context_length = S.
        // temb expanded: (B, S, dim)
        let temb_per_token = temb
            .unsqueeze(1)?
            .repeat_axis_device(1, s)?;
        // timestep_proj per token: (B, 6, dim) → (B, S, 6, dim)
        // diffusers does: view(B, 6, 1, dim).expand(B, 6, S, dim) → permute(0, 2, 1, 3) → (B, S, 6, dim)
        let tp_4d = timestep_proj_3d
            .reshape(&[b, 6, 1, cfg.hidden_size])?
            .repeat_axis_device(2, s)?;
        let tp_4d = tp_4d.permute(&[0, 2, 1, 3])?.contiguous()?; // (B, S, 6, dim)

        // 4. Stack blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, &encoder_proj, &tp_4d, &rotary, cfg)?;
        }

        // 5. Output norm + projection
        let x = self.norm_out(&x, &temb_per_token)?;
        let x = fused_linear3d_native(&x, &self.proj_out_w, Some(&self.proj_out_b))?;

        // 6. Unpatchify
        self.unpatchify(&x, f_grid, h_grid, w_grid)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;
    use flame_core::serialization::load_file;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn take(map: &mut HashMap<String, Tensor>, key: &str) -> Tensor {
        map.remove(key)
            .unwrap_or_else(|| panic!("fixture missing key: {key}"))
    }

    fn build_block(map: &mut HashMap<String, Tensor>) -> HeliosBlock {
        // Diffusers-named keys (param-flat under HeliosTransformerBlock)
        HeliosBlock {
            scale_shift_table: take(map, "block.scale_shift_table"),
            norm2_w: take(map, "block.norm2.weight"),
            norm2_b: take(map, "block.norm2.bias"),
            a1_to_q_w: take(map, "block.attn1.to_q.weight"),
            a1_to_q_b: take(map, "block.attn1.to_q.bias"),
            a1_to_k_w: take(map, "block.attn1.to_k.weight"),
            a1_to_k_b: take(map, "block.attn1.to_k.bias"),
            a1_to_v_w: take(map, "block.attn1.to_v.weight"),
            a1_to_v_b: take(map, "block.attn1.to_v.bias"),
            a1_to_out_w: take(map, "block.attn1.to_out.0.weight"),
            a1_to_out_b: take(map, "block.attn1.to_out.0.bias"),
            a1_norm_q_w: take(map, "block.attn1.norm_q.weight"),
            a1_norm_k_w: take(map, "block.attn1.norm_k.weight"),
            a2_to_q_w: take(map, "block.attn2.to_q.weight"),
            a2_to_q_b: take(map, "block.attn2.to_q.bias"),
            a2_to_k_w: take(map, "block.attn2.to_k.weight"),
            a2_to_k_b: take(map, "block.attn2.to_k.bias"),
            a2_to_v_w: take(map, "block.attn2.to_v.weight"),
            a2_to_v_b: take(map, "block.attn2.to_v.bias"),
            a2_to_out_w: take(map, "block.attn2.to_out.0.weight"),
            a2_to_out_b: take(map, "block.attn2.to_out.0.bias"),
            a2_norm_q_w: take(map, "block.attn2.norm_q.weight"),
            a2_norm_k_w: take(map, "block.attn2.norm_k.weight"),
            ffn_up_w: take(map, "block.ffn.net.0.proj.weight"),
            ffn_up_b: take(map, "block.ffn.net.0.proj.bias"),
            ffn_down_w: take(map, "block.ffn.net.2.weight"),
            ffn_down_b: take(map, "block.ffn.net.2.bias"),
        }
    }

    fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        assert_eq!(a.len(), b.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
        }
        let mean = (sum_abs / a.len() as f64) as f32;
        (max_abs, mean, sum_abs as f32)
    }

    fn build_block_at(map: &mut HashMap<String, Tensor>, prefix: &str) -> HeliosBlock {
        let r = |m: &mut HashMap<String, Tensor>, k: &str| {
            m.remove(k).unwrap_or_else(|| panic!("fixture missing key: {k}"))
        };
        HeliosBlock {
            scale_shift_table: r(map, &format!("{prefix}scale_shift_table")),
            norm2_w: r(map, &format!("{prefix}norm2.weight")),
            norm2_b: r(map, &format!("{prefix}norm2.bias")),
            a1_to_q_w: r(map, &format!("{prefix}attn1.to_q.weight")),
            a1_to_q_b: r(map, &format!("{prefix}attn1.to_q.bias")),
            a1_to_k_w: r(map, &format!("{prefix}attn1.to_k.weight")),
            a1_to_k_b: r(map, &format!("{prefix}attn1.to_k.bias")),
            a1_to_v_w: r(map, &format!("{prefix}attn1.to_v.weight")),
            a1_to_v_b: r(map, &format!("{prefix}attn1.to_v.bias")),
            a1_to_out_w: r(map, &format!("{prefix}attn1.to_out.0.weight")),
            a1_to_out_b: r(map, &format!("{prefix}attn1.to_out.0.bias")),
            a1_norm_q_w: r(map, &format!("{prefix}attn1.norm_q.weight")),
            a1_norm_k_w: r(map, &format!("{prefix}attn1.norm_k.weight")),
            a2_to_q_w: r(map, &format!("{prefix}attn2.to_q.weight")),
            a2_to_q_b: r(map, &format!("{prefix}attn2.to_q.bias")),
            a2_to_k_w: r(map, &format!("{prefix}attn2.to_k.weight")),
            a2_to_k_b: r(map, &format!("{prefix}attn2.to_k.bias")),
            a2_to_v_w: r(map, &format!("{prefix}attn2.to_v.weight")),
            a2_to_v_b: r(map, &format!("{prefix}attn2.to_v.bias")),
            a2_to_out_w: r(map, &format!("{prefix}attn2.to_out.0.weight")),
            a2_to_out_b: r(map, &format!("{prefix}attn2.to_out.0.bias")),
            a2_norm_q_w: r(map, &format!("{prefix}attn2.norm_q.weight")),
            a2_norm_k_w: r(map, &format!("{prefix}attn2.norm_k.weight")),
            ffn_up_w: r(map, &format!("{prefix}ffn.net.0.proj.weight")),
            ffn_up_b: r(map, &format!("{prefix}ffn.net.0.proj.bias")),
            ffn_down_w: r(map, &format!("{prefix}ffn.net.2.weight")),
            ffn_down_b: r(map, &format!("{prefix}ffn.net.2.bias")),
        }
    }

    fn build_dit_from_fixture(map: &mut HashMap<String, Tensor>, cfg: HeliosConfig) -> HeliosDit {
        let blocks: Vec<_> = (0..cfg.num_layers)
            .map(|i| build_block_at(map, &format!("model.blocks.{i}.")))
            .collect();

        // Reshape Conv3d patch_embedding weight (dim, in_ch, kT, kH, kW) → (dim, in_ch*kT*kH*kW)
        // for use with fused_linear3d_native. Memory layout is preserved.
        let patch_w = take(map, "model.patch_embedding.weight");
        let pw_dims = patch_w.shape().dims().to_vec();
        let dim = pw_dims[0];
        let in_per_patch: usize = pw_dims[1..].iter().product();
        let patch_w = patch_w.reshape(&[dim, in_per_patch]).expect("reshape patch_w");
        let patch_b = take(map, "model.patch_embedding.bias");

        HeliosDit {
            config: cfg,
            blocks,
            patch_embed_w: patch_w,
            patch_embed_b: patch_b,
            time_l1_w: take(map, "model.condition_embedder.time_embedder.linear_1.weight"),
            time_l1_b: take(map, "model.condition_embedder.time_embedder.linear_1.bias"),
            time_l2_w: take(map, "model.condition_embedder.time_embedder.linear_2.weight"),
            time_l2_b: take(map, "model.condition_embedder.time_embedder.linear_2.bias"),
            time_proj_w: take(map, "model.condition_embedder.time_proj.weight"),
            time_proj_b: take(map, "model.condition_embedder.time_proj.bias"),
            text_l1_w: take(map, "model.condition_embedder.text_embedder.linear_1.weight"),
            text_l1_b: take(map, "model.condition_embedder.text_embedder.linear_1.bias"),
            text_l2_w: take(map, "model.condition_embedder.text_embedder.linear_2.weight"),
            text_l2_b: take(map, "model.condition_embedder.text_embedder.linear_2.bias"),
            norm_out_sst: take(map, "model.norm_out.scale_shift_table"),
            proj_out_w: take(map, "model.proj_out.weight"),
            proj_out_b: take(map, "model.proj_out.bias"),
        }
    }

    #[test]
    fn helios_dit_small_parity_vs_pytorch() {
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/helios/dit_small_parity.safetensors");
        if !fixture.exists() {
            eprintln!("fixture missing — generate via scripts/generate_helios_dit_small.py");
            return;
        }
        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture, &device).expect("load fixture");

        // Tiny config matching the fixture script.
        let cfg = HeliosConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            hidden_size: 128,
            ffn_dim: 384,
            text_dim: 64,
            rope_dim: [12, 10, 10],
            rope_theta: 10_000.0,
            freq_dim: 64,
            patch_size: [1, 2, 2],
            in_channels: 16,
            out_channels: 16,
            eps: 1e-6,
            cross_attn_norm: true,
            guidance_cross_attn: false,
            is_amplify_history: false,
        };

        let inp_x = take(&mut map, "inputs.hidden_states");
        let inp_enc = take(&mut map, "inputs.encoder_hidden_states");
        let inp_t = take(&mut map, "inputs.timestep");
        let expected = take(&mut map, "expected.output");

        let dit = build_dit_from_fixture(&mut map, cfg);
        let out = dit
            .forward(&inp_x, &inp_enc, &inp_t)
            .expect("forward");
        let g = out.to_vec_f32().expect("got");
        let e = expected.to_vec_f32().expect("expected");
        let (max_abs, mean_abs, _sum) = diff_stats(&g, &e);
        eprintln!("helios dit_small parity: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
        assert!(mean_abs < 1e-3, "mean_abs {mean_abs} exceeds 1e-3");
        assert!(max_abs < 1e-1, "max_abs {max_abs} exceeds 1e-1");
    }

    #[test]
    fn helios_block_parity_vs_pytorch() {
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/helios/block_parity.safetensors");
        if !fixture.exists() {
            eprintln!("fixture missing — generate via scripts/generate_helios_block.py");
            return;
        }
        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture, &device).expect("load fixture");

        let cfg = HeliosConfig {
            num_layers: 1,
            num_heads: 4,
            head_dim: 32,
            hidden_size: 128,
            ffn_dim: 384,
            text_dim: 128,
            rope_dim: [12, 10, 10],
            rope_theta: 10_000.0,
            freq_dim: 256,
            patch_size: [1, 2, 2],
            in_channels: 16,
            out_channels: 16,
            eps: 1e-6,
            cross_attn_norm: true,
            guidance_cross_attn: false,
            is_amplify_history: false,
        };

        let inp_x = take(&mut map, "inputs.hidden_states");
        let inp_enc = take(&mut map, "inputs.encoder_hidden_states");
        let inp_temb = take(&mut map, "inputs.temb");
        let inp_freqs = take(&mut map, "inputs.freqs_cis");
        let expected = take(&mut map, "expected.output");

        let block = build_block(&mut map);
        let out = block
            .forward(&inp_x, &inp_enc, &inp_temb, &inp_freqs, &cfg)
            .expect("forward");
        let g = out.to_vec_f32().expect("got vec");
        let e = expected.to_vec_f32().expect("expected vec");
        let (max_abs, mean_abs, _sum) = diff_stats(&g, &e);
        eprintln!("helios block parity: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
        assert!(mean_abs < 1e-3, "mean_abs {mean_abs} exceeds 1e-3");
        assert!(max_abs < 1e-1, "max_abs {max_abs} exceeds 1e-1");
    }
}
