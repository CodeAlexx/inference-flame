//! Nucleus-Image MoE DiT — pure-Rust BF16 inference port (Phase 6, in progress).
//!
//! 17 B sparse Mixture-of-Experts diffusion transformer. Single-stream DiT with
//! cross-attention to a Qwen3-VL text encoder (4096-d). 32 layers: 3 dense + 29
//! MoE (64 experts each, expert-choice routing, capacity 4 for layers 3-4 and 2
//! for 5-31).
//!
//! References:
//! - `diffusers/models/transformers/transformer_nucleusmoe_image.py`
//! - `HANDOFF_2026-04-29_MOE_KERNELS_AND_NUCLEUS.md`     — parent (Phase 0-5)
//! - `HANDOFF_2026-04-29_NUCLEUS_WEIGHT_AUDIT.md`        — surface audit
//!
//! Phase 6.2 (this commit): config + dense block forward + PyTorch parity test.
//! Phase 6.3+: MoE block, full DiT, KV cache, end-to-end inference.

use flame_core::attention::sdpa;
use flame_core::bf16_ops::{rope_fused_bf16, swiglu_fused_bf16};
use flame_core::cuda_ops_bf16::rms_norm_bf16;
use flame_core::layer_norm::layer_norm;
use flame_core::ops::fused_inference::fused_linear3d_native;
use flame_core::ops::nucleus_moe::nucleus_moe_expert_forward;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Error, Result, Tensor};
use flame_diffusion::block_offload::BlockFacilitator;
use flame_diffusion::BlockOffloader;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NucleusConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub joint_attention_dim: usize,
    pub mlp_ratio: f32,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
    pub num_experts: usize,
    pub moe_intermediate_dim: usize,
    /// Per-layer capacity factor. Length = `num_layers`. 0 marks a dense layer
    /// (matches diffusers `dense_moe_strategy="leave_first_three_blocks_dense"`
    /// where layers 0..2 are dense → factor=0).
    pub capacity_factors: Vec<f32>,
    pub use_sigmoid: bool,
    pub route_scale: f32,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub eps: f32,
}

impl NucleusConfig {
    /// Hard-coded defaults from `transformer/config.json` of the public
    /// NucleusAI/Nucleus-Image checkpoint.
    pub fn nucleus_image_default() -> Self {
        let mut capacity_factors = vec![0.0_f32; 32];
        capacity_factors[3] = 4.0;
        capacity_factors[4] = 4.0;
        for c in capacity_factors.iter_mut().take(32).skip(5) {
            *c = 2.0;
        }
        Self {
            num_layers: 32,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            hidden_size: 2048,
            joint_attention_dim: 4096,
            mlp_ratio: 4.0,
            axes_dims_rope: [16, 56, 56],
            rope_theta: 10_000.0,
            num_experts: 64,
            moe_intermediate_dim: 1344,
            capacity_factors,
            use_sigmoid: false,
            route_scale: 2.5,
            patch_size: 2,
            in_channels: 64,
            out_channels: 16,
            eps: 1e-6,
        }
    }

    /// Dense (non-MoE) block FFN intermediate dim. Computed as
    /// `int(D * mlp_ratio * 2/3) // 128 * 128`. For default config: 5376.
    pub fn dense_inner_dim(&self) -> usize {
        let raw = (self.hidden_size as f32 * self.mlp_ratio * 2.0 / 3.0) as usize;
        (raw / 128) * 128
    }
}

// ---------------------------------------------------------------------------
// Per-block weight storage
// ---------------------------------------------------------------------------

/// MLP weights — dense for layers 0..2, MoE for 3..31.
pub enum NucleusMlpWeights {
    Dense {
        /// `(2 * dense_inner_dim, hidden_size)` BF16. Diffusers `SwiGLU`
        /// layout: chunk(2) → `[hidden_up, gate]`. flame's
        /// `swiglu_fused_bf16(gate, up)` is `silu(gate) * up`, so the dense
        /// FFN passes `swiglu(chunk[1], chunk[0])`.
        gate_up_w: Tensor,
        /// `(hidden_size, dense_inner_dim)` BF16.
        down_w: Tensor,
    },
    /// Phase 6.3 — fields populated, forward stubbed for now.
    Moe {
        /// `(num_experts, 2 * hidden_size)` BF16. Router takes a 2*D input
        /// (concat of timestep_expanded and unmodulated hidden_states).
        gate_w: Tensor,
        /// `(num_experts, hidden_size, 2 * moe_intermediate_dim)` BF16.
        /// `SwiGLUExperts` layout: chunk(2) → `[gate, up]` (opposite of dense).
        gate_up_w: Tensor,
        /// `(num_experts, moe_intermediate_dim, hidden_size)` BF16.
        down_w: Tensor,
        /// `(2 * moe_intermediate_dim, hidden_size)` BF16. Shared expert is a
        /// `FeedForward(activation_fn="swiglu")` — same `[hidden_up, gate]`
        /// layout as dense.
        shared_gate_up_w: Tensor,
        /// `(hidden_size, moe_intermediate_dim)` BF16.
        shared_down_w: Tensor,
        capacity_factor: f32,
    },
}

pub struct NucleusBlock {
    /// Modulation: `(4 * hidden_size, hidden_size)` BF16.
    pub mod_w: Tensor,
    pub mod_b: Tensor,
    /// `(hidden_size, joint_attention_dim)` BF16 — projects per-block context
    /// from text-encoder dim down to model dim before attn K/V.
    pub enc_proj_w: Tensor,
    pub enc_proj_b: Tensor,
    /// Q/K/V/out projections (no biases; out has only `to_out.0`).
    pub to_q_w: Tensor,
    pub to_k_w: Tensor,
    pub to_v_w: Tensor,
    pub to_out_w: Tensor,
    /// Text K/V projections (no `add_q_proj` — image is the only Q).
    pub add_k_w: Tensor,
    pub add_v_w: Tensor,
    /// Per-head RMSNorm weights, shape `(head_dim,)`.
    pub norm_q_w: Tensor,
    pub norm_k_w: Tensor,
    pub norm_added_k_w: Tensor,
    // `attn.norm_added_q.weight` is in the checkpoint but unused by the
    // diffusers `NucleusMoEAttnProcessor2_0`; intentionally NOT stored.
    pub mlp: NucleusMlpWeights,
}

impl NucleusBlock {
    /// Single-block forward (mirrors
    /// `NucleusMoEImageTransformerBlock.forward` line-for-line).
    ///
    /// Parameters:
    /// - `x`: BF16 `[B, S_img, D]` image hidden states.
    /// - `enc`: BF16 `[B, S_txt, joint_attention_dim]` already passed through
    ///   the global `txt_norm` RMSNorm.
    /// - `temb`: BF16 `[B, D]` timestep embedding (post `time_text_embed`).
    /// - `img_cos`/`img_sin`: BF16 `[1, 1, S_img, head_dim/2]` interleaved-
    ///   pair RoPE tables (built by `build_3d_rope` for img positions).
    /// - `txt_cos`/`txt_sin`: BF16 `[1, 1, S_txt, head_dim/2]` for txt.
    /// - `attn_mask`: optional joint-attention mask
    ///   `[B, 1, 1, S_img+S_txt]` or broadcast-compatible.
    pub fn forward(
        &self,
        x: &Tensor,
        enc: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        attn_mask: Option<&Tensor>,
        cfg: &NucleusConfig,
    ) -> Result<Tensor> {
        self.forward_with_cache(
            x, Some(enc), temb, img_cos, img_sin, Some(txt_cos), Some(txt_sin), None, attn_mask,
            cfg,
        )
    }

    /// Forward with optional pre-computed text K/V cache. Phase 6.5: when
    /// `cached_txt` is `Some`, the per-block txt projection + norm + RoPE is
    /// skipped — the cached values were produced by a prior `compute_txt_kv`
    /// pass and are reused across all 50 denoise steps. `enc`, `txt_cos`,
    /// `txt_sin` are unused when `cached_txt` is `Some`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        enc: Option<&Tensor>,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: Option<&Tensor>,
        txt_sin: Option<&Tensor>,
        cached_txt: Option<(&Tensor, &Tensor)>,
        attn_mask: Option<&Tensor>,
        cfg: &NucleusConfig,
    ) -> Result<Tensor> {
        // ---- Modulation: scale1, gate1, scale2, gate2 = Linear(SiLU(temb)).chunk(4)
        // The diffusers `img_mod = Sequential(SiLU, Linear(D, 4D))` applies
        // SiLU **inside** so we apply it before the linear.
        let temb_silu = temb.silu()?;
        let temb_3d = temb_silu.unsqueeze(1)?; // [B, 1, D]
        let mod_out = fused_linear3d_native(&temb_3d, &self.mod_w, Some(&self.mod_b))?; // [B,1,4D]
        let chunks = mod_out.chunk(4, 2)?;
        let scale1 = &chunks[0];
        let gate1 = chunks[1].clamp(-2.0, 2.0)?;
        let scale2 = &chunks[2];
        let gate2 = chunks[3].clamp(-2.0, 2.0)?;

        // ---- encoder_proj: (B, S_txt, joint_dim) -> (B, S_txt, D)
        // Skipped when txt K/V are cached; diffusers does the same:
        //   context = None if attn_kwargs.get("cached_txt_key") is not None else self.encoder_proj(...)
        let context_opt = match cached_txt {
            Some(_) => None,
            None => {
                let enc = enc.ok_or_else(|| {
                    Error::InvalidOperation(
                        "NucleusBlock::forward_with_cache: enc required when cached_txt is None"
                            .into(),
                    )
                })?;
                Some(fused_linear3d_native(
                    enc,
                    &self.enc_proj_w,
                    Some(&self.enc_proj_b),
                )?)
            }
        };

        // ---- pre_attn LayerNorm (no affine), modulate
        let img_normed = layer_norm(x, &[cfg.hidden_size], None, None, cfg.eps)?;
        let scale1_p1 = scale1.add_scalar(1.0)?;
        let img_mod1 = img_normed.mul(&scale1_p1)?;

        // ---- Joint attention (img Q; img+txt K/V)
        let attn_out = self.attention(
            &img_mod1,
            context_opt.as_ref(),
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            cached_txt,
            attn_mask,
            cfg,
        )?;

        // ---- Residual: x = x + tanh(gate1) * attn_out
        let g1_tanh = gate1.tanh()?;
        let res1 = g1_tanh.mul(&attn_out)?;
        let x = x.add(&res1)?;

        // ---- pre_mlp LayerNorm + modulate
        let img_normed2 = layer_norm(&x, &[cfg.hidden_size], None, None, cfg.eps)?;
        let scale2_p1 = scale2.add_scalar(1.0)?;
        let img_mod2 = img_normed2.mul(&scale2_p1)?;

        // ---- MLP
        let mlp_out = match &self.mlp {
            NucleusMlpWeights::Dense { gate_up_w, down_w } => {
                Self::dense_ffn(&img_mod2, gate_up_w, down_w)?
            }
            NucleusMlpWeights::Moe {
                gate_w,
                gate_up_w,
                down_w,
                shared_gate_up_w,
                shared_down_w,
                capacity_factor,
            } => Self::moe_ffn(
                &img_mod2,
                &img_normed2,
                temb,
                gate_w,
                gate_up_w,
                down_w,
                shared_gate_up_w,
                shared_down_w,
                *capacity_factor,
                cfg,
            )?,
        };

        // ---- Residual: x = x + tanh(gate2) * mlp_out
        let g2_tanh = gate2.tanh()?;
        let res2 = g2_tanh.mul(&mlp_out)?;
        x.add(&res2)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        img: &Tensor,                              // (B, S_img, D)
        ctx: Option<&Tensor>,                      // (B, S_txt, D) — already encoder_proj'd; None if cached
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: Option<&Tensor>,
        txt_sin: Option<&Tensor>,
        cached_txt: Option<(&Tensor, &Tensor)>, // (txt_k, txt_v) post-norm/post-RoPE, [B, kv_h, S_txt, hd]
        attn_mask: Option<&Tensor>,
        cfg: &NucleusConfig,
    ) -> Result<Tensor> {
        let dims = img.shape().dims();
        let (b, s_img, d) = (dims[0], dims[1], dims[2]);
        let h = cfg.num_heads;
        let kvh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let n_rep = h / kvh;

        // Q/K/V from img
        let img_q = fused_linear3d_native(img, &self.to_q_w, None)?; // (B,S_img,D)
        let img_k = fused_linear3d_native(img, &self.to_k_w, None)?; // (B,S_img,kvh*hd)
        let img_v = fused_linear3d_native(img, &self.to_v_w, None)?;

        let img_q = img_q.reshape(&[b, s_img, h, hd])?;
        let img_k = img_k.reshape(&[b, s_img, kvh, hd])?;
        let img_v = img_v.reshape(&[b, s_img, kvh, hd])?;

        // QK norm — RMSNorm along last axis with weight (head_dim,)
        let img_q = qk_norm(&img_q, &self.norm_q_w, b * s_img * h, hd, cfg.eps)?;
        let img_k = qk_norm(&img_k, &self.norm_k_w, b * s_img * kvh, hd, cfg.eps)?;
        // (img_v is unchanged — diffusers does not RMSNorm V)
        let img_q = img_q.reshape(&[b, s_img, h, hd])?;
        let img_k = img_k.reshape(&[b, s_img, kvh, hd])?;

        // Permute to (B, H, S, head_dim) for RoPE + SDPA
        let img_q = img_q.permute(&[0, 2, 1, 3])?.contiguous()?;
        let img_k = img_k.permute(&[0, 2, 1, 3])?.contiguous()?;
        let img_v = img_v.permute(&[0, 2, 1, 3])?.contiguous()?;

        let img_q = rope_fused_bf16(&img_q, img_cos, img_sin)?;
        let img_k = rope_fused_bf16(&img_k, img_cos, img_sin)?;

        // Text K/V — either from cache or freshly projected
        let (txt_k, txt_v) = match cached_txt {
            Some((k, v)) => (k.clone(), v.clone()),
            None => {
                let ctx = ctx.ok_or_else(|| {
                    Error::InvalidOperation(
                        "attention: ctx required when cached_txt is None".into(),
                    )
                })?;
                let txt_cos = txt_cos.ok_or_else(|| {
                    Error::InvalidOperation(
                        "attention: txt_cos required when cached_txt is None".into(),
                    )
                })?;
                let txt_sin = txt_sin.ok_or_else(|| {
                    Error::InvalidOperation(
                        "attention: txt_sin required when cached_txt is None".into(),
                    )
                })?;
                self.compute_txt_kv(ctx, txt_cos, txt_sin, cfg)?
            }
        };

        // Concat along sequence axis (axis=2)
        let joint_k = Tensor::cat(&[&img_k, &txt_k], 2)?;
        let joint_v = Tensor::cat(&[&img_v, &txt_v], 2)?;

        // GQA: replicate K/V to match Q heads
        let joint_k = repeat_kv(&joint_k, n_rep)?;
        let joint_v = repeat_kv(&joint_v, n_rep)?;

        let attn_out = sdpa(&img_q, &joint_k, &joint_v, attn_mask)?;

        // (B, H, S_img, hd) -> (B, S_img, H, hd) -> (B, S_img, D)
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.contiguous()?;
        let attn_out = attn_out.reshape(&[b, s_img, d])?;

        fused_linear3d_native(&attn_out, &self.to_out_w, None)
    }

    /// Compute the per-block txt K/V given an already-encoder_proj'd
    /// context. Public so `NucleusDit::compute_kv_cache` can call it.
    /// Returns `(txt_k_post_rope, txt_v)` shaped `[B, kv_h, S_txt, head_dim]`.
    pub fn compute_txt_kv(
        &self,
        ctx: &Tensor, // (B, S_txt, D) — post encoder_proj
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        cfg: &NucleusConfig,
    ) -> Result<(Tensor, Tensor)> {
        let dims = ctx.shape().dims();
        let (b, s_txt) = (dims[0], dims[1]);
        let kvh = cfg.num_kv_heads;
        let hd = cfg.head_dim;

        let txt_k = fused_linear3d_native(ctx, &self.add_k_w, None)?;
        let txt_v = fused_linear3d_native(ctx, &self.add_v_w, None)?;
        let txt_k = txt_k.reshape(&[b, s_txt, kvh, hd])?;
        let txt_v = txt_v.reshape(&[b, s_txt, kvh, hd])?;
        let txt_k = qk_norm(&txt_k, &self.norm_added_k_w, b * s_txt * kvh, hd, cfg.eps)?;
        let txt_k = txt_k.reshape(&[b, s_txt, kvh, hd])?;
        let txt_k = txt_k.permute(&[0, 2, 1, 3])?.contiguous()?;
        let txt_v = txt_v.permute(&[0, 2, 1, 3])?.contiguous()?;
        let txt_k = rope_fused_bf16(&txt_k, txt_cos, txt_sin)?;
        Ok((txt_k, txt_v))
    }

    /// MoE expert FFN. Caller supplies both modulated and unmodulated
    /// hidden states because diffusers `NucleusMoELayer.forward`:
    /// - feeds **modulated** hidden states to the routed experts AND the
    ///   shared expert,
    /// - feeds **unmodulated** hidden states (concatenated with the tiled
    ///   `temb`) to the **router** only.
    ///
    /// Mirrors `transformer_nucleusmoe_image.py:NucleusMoELayer.forward`
    /// step-for-step.
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn(
        modulated: &Tensor,        // (B, S, D) BF16
        unmodulated: &Tensor,      // (B, S, D) BF16 — `img_normed2`
        temb: &Tensor,             // (B, D)   BF16
        gate_w: &Tensor,           // (E, 2*D) BF16
        gate_up_w: &Tensor,        // (E, D, 2*moe_inter) BF16
        down_w: &Tensor,           // (E, moe_inter, D) BF16
        shared_gate_up_w: &Tensor, // (2*moe_inter, D) BF16
        shared_down_w: &Tensor,    // (D, moe_inter) BF16
        capacity_factor: f32,
        cfg: &NucleusConfig,
    ) -> Result<Tensor> {
        let dims = modulated.shape().dims();
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let e = cfg.num_experts;

        // Expert-choice capacity per (batch, expert).
        let capacity = ((capacity_factor * s as f32) / e as f32).ceil() as usize;
        let capacity = capacity.max(1);

        // ---- Router input: cat([temb_tile, unmodulated], dim=-1) -> (B, S, 2D)
        let temb_tile = temb.unsqueeze(1)?.repeat_axis_device(1, s)?;
        let router_in = Tensor::cat(&[&temb_tile, unmodulated], 2)?;

        // ---- Router logits + softmax + transpose -> affinity
        // diffusers does:
        //     scores = softmax(logits.float(), -1).to(logits.dtype)   # BF16
        //     affinity = scores.transpose(1, 2)
        // The flame wrapper takes affinity as F32 but does its top-K on
        // whatever values it gets. If we keep the F32 softmax output
        // straight, top-K can pick different winners than diffusers's BF16
        // top-K when two affinity values differ by less than 1 BF16 ULP.
        // Round through BF16 here so the F32 affinity has BF16 precision,
        // matching the diffusers recipe bit-for-bit on top-K choices.
        let logits = fused_linear3d_native(&router_in, gate_w, None)?; // (B, S, E) BF16
        let scores_f32 = logits.to_dtype(DType::F32)?.softmax(-1)?;
        let scores_bf16 = scores_f32.to_dtype(DType::BF16)?;
        let affinity = scores_bf16
            .to_dtype(DType::F32)?
            .permute(&[0, 2, 1])?
            .contiguous()?; // (B, E, S) F32 with BF16 precision

        // ---- Routed expert FFN (caller flattens; wrapper does the rest)
        let modulated_flat = modulated.reshape(&[b * s, d])?;
        let routed_flat = nucleus_moe_expert_forward(
            &modulated_flat,
            &affinity,
            gate_up_w,
            down_w,
            capacity,
            cfg.route_scale,
        )?; // (B*S, D) BF16
        let routed = routed_flat.reshape(&[b, s, d])?;

        // ---- Shared expert: SwiGLU FF on modulated, same `[hidden, gate]`
        // layout as a dense FFN's `net.0.proj`.
        let shared = Self::dense_ffn(modulated, shared_gate_up_w, shared_down_w)?;

        // ---- Final: out = shared + routed
        shared.add(&routed)
    }

    fn dense_ffn(x: &Tensor, gate_up_w: &Tensor, down_w: &Tensor) -> Result<Tensor> {
        // Diffusers `SwiGLU.forward`:
        //   hidden_states, gate = proj(x).chunk(2, dim=-1)   # [hidden, gate]
        //   return hidden_states * silu(gate)
        // flame's `swiglu_fused_bf16(gate, up) = silu(gate) * up`, so we must
        // pass gate=second_half, up=first_half. (Opposite of MoE expert FFN
        // which uses [gate, up] layout — see Phase 6.3.)
        let gu = fused_linear3d_native(x, gate_up_w, None)?;
        let last = gu.shape().dims().len() - 1;
        let chunks = gu.chunk(2, last)?;
        let up = &chunks[0];
        let gate = &chunks[1];
        let act = swiglu_fused_bf16(gate, up)?;
        fused_linear3d_native(&act, down_w, None)
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// Per-block text K/V cache for fixed-prompt 50-step denoise loops.
/// Each entry is `(txt_k_post_rope, txt_v)` shaped
/// `[B, num_kv_heads, S_txt, head_dim]`. Phase 6.5.
pub struct NucleusKVCache {
    pub entries: Vec<(Tensor, Tensor)>,
    pub txt_seq_len: usize,
}

/// Top-level Nucleus-Image DiT (assembly only — Phase 6.4 scope).
///
/// Holds shared weights (img_in, time_text_embed, txt_norm, norm_out,
/// proj_out) plus an ordered `Vec` of per-layer blocks. Streaming /
/// BlockOffloader integration arrives in Phase 6.6.
pub struct NucleusDit {
    pub config: NucleusConfig,
    pub img_in_w: Tensor,
    pub img_in_b: Tensor,
    pub time_proj_l1_w: Tensor,
    pub time_proj_l1_b: Tensor,
    pub time_proj_l2_w: Tensor,
    pub time_proj_l2_b: Tensor,
    pub time_norm_w: Tensor,
    pub txt_norm_w: Tensor,
    pub norm_out_w: Tensor,
    pub norm_out_b: Tensor,
    pub proj_out_w: Tensor,
    pub blocks: Vec<NucleusBlock>,
}

impl NucleusDit {
    /// Sinusoidal timestep embedding matching diffusers's
    /// `Timesteps(num_channels=embedding_dim, flip_sin_to_cos=True,
    /// downscale_freq_shift=0, scale=1000)`. Same algebra as
    /// `qwenimage_dit::time_proj`.
    fn time_proj(&self, timestep: &Tensor) -> Result<Tensor> {
        let dim = self.config.hidden_size;
        let half = dim / 2;
        let max_period = 10_000.0_f64;
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(
            freq_data,
            flame_core::Shape::from_dims(&[1, half]),
            timestep.device().clone(),
        )?;
        let t_f32 = timestep.to_dtype(DType::F32)?.mul_scalar(1000.0)?;
        let args = t_f32.unsqueeze(1)?.matmul(&freqs)?;
        let sin_part = args.sin()?;
        let cos_part = args.cos()?;
        // flip_sin_to_cos=True → final order is (cos, sin)
        let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;
        emb.to_dtype(DType::BF16)
    }

    /// Compose `temb = RMSNorm(linear_2(silu(linear_1(time_proj(t)))))`.
    fn time_text_embed(&self, timestep: &Tensor) -> Result<Tensor> {
        let proj = self.time_proj(timestep)?; // (B, D)
        let proj_3d = proj.unsqueeze(1)?; // (B, 1, D)
        let l1 = fused_linear3d_native(&proj_3d, &self.time_proj_l1_w, Some(&self.time_proj_l1_b))?;
        let l1 = l1.silu()?;
        let l2 = fused_linear3d_native(&l1, &self.time_proj_l2_w, Some(&self.time_proj_l2_b))?;
        let l2_2d = l2.squeeze(Some(1))?;
        // RMSNorm
        rms_norm_bf16(&l2_2d, Some(&self.time_norm_w), self.config.eps)
    }

    /// Build the per-block text K/V cache for a fixed prompt. Phase 6.5.
    ///
    /// Runs the txt-side prefix once: top-level `txt_norm` on
    /// `encoder_hidden_states`, then per-block `encoder_proj`, `add_k_proj`,
    /// `add_v_proj`, `norm_added_k`, and RoPE on K.
    ///
    /// The returned `Vec` is indexed by block; pass it to `forward_cached`
    /// to skip recomputing the txt path on every denoise step.
    pub fn compute_kv_cache(
        &self,
        encoder_hidden_states: &Tensor,
        img_shapes: (usize, usize, usize),
    ) -> Result<NucleusKVCache> {
        let cfg = &self.config;
        let enc_dims = encoder_hidden_states.shape().dims();
        let (enc_b, enc_s, enc_d) = (enc_dims[0], enc_dims[1], enc_dims[2]);
        let enc_flat = encoder_hidden_states.reshape(&[enc_b * enc_s, enc_d])?;
        let enc_normed = rms_norm_bf16(&enc_flat, Some(&self.txt_norm_w), cfg.eps)?;
        let enc = enc_normed.reshape(&[enc_b, enc_s, enc_d])?;

        // Build txt RoPE only (img RoPE not needed for the cache itself).
        let (_img_cos, _img_sin, txt_cos, txt_sin) = build_nucleus_3d_rope(
            img_shapes,
            enc_s,
            cfg.axes_dims_rope,
            cfg.rope_theta,
            true,
            encoder_hidden_states.device().clone(),
        )?;

        let mut entries = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            // Per-block encoder_proj: (B, S_txt, joint_dim) -> (B, S_txt, D)
            let context =
                fused_linear3d_native(&enc, &block.enc_proj_w, Some(&block.enc_proj_b))?;
            let (k, v) = block.compute_txt_kv(&context, &txt_cos, &txt_sin, cfg)?;
            entries.push((k, v));
        }
        Ok(NucleusKVCache {
            entries,
            txt_seq_len: enc_s,
        })
    }

    /// Forward pass using a pre-computed text K/V cache. Cheaper per step
    /// than `forward` since the txt prefix work is amortized once across
    /// the whole 50-step denoise loop.
    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        img_shapes: (usize, usize, usize),
        kv_cache: &NucleusKVCache,
    ) -> Result<Tensor> {
        self.forward_inner(hidden_states, None, timestep, img_shapes, Some(kv_cache))
    }

    /// One forward pass through the full DiT (one denoise step).
    ///
    /// - `hidden_states`: BF16 `[B, S_img, in_channels=64]` patchified noise latent.
    /// - `encoder_hidden_states`: BF16 `[B, S_txt, joint_attention_dim]` raw text encoder output.
    /// - `encoder_mask`: optional `[B, S_txt]` bool — currently treated as "all ones" if None.
    /// - `timestep`: `[B]` (any dtype).
    /// - `img_shapes`: `(F, H, W)` patchified video shape (`H = pixel_h / patch_size`, etc).
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_mask: Option<&Tensor>,
        timestep: &Tensor,
        img_shapes: (usize, usize, usize),
    ) -> Result<Tensor> {
        let _ = encoder_mask; // TODO Phase 6.5+: mask plumbing
        self.forward_inner(
            hidden_states,
            Some(encoder_hidden_states),
            timestep,
            img_shapes,
            None,
        )
    }

    fn forward_inner(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        timestep: &Tensor,
        img_shapes: (usize, usize, usize),
        kv_cache: Option<&NucleusKVCache>,
    ) -> Result<Tensor> {
        let cfg = &self.config;

        // img_in: (B, S, in_channels) -> (B, S, D)
        let mut x = fused_linear3d_native(hidden_states, &self.img_in_w, Some(&self.img_in_b))?;

        // Determine text seq len (for RoPE size) — from cache if cached, else
        // from encoder_hidden_states.
        let s_txt = if let Some(c) = kv_cache {
            c.txt_seq_len
        } else {
            encoder_hidden_states
                .ok_or_else(|| {
                    Error::InvalidOperation(
                        "forward_inner: encoder_hidden_states required when kv_cache is None"
                            .into(),
                    )
                })?
                .shape()
                .dims()[1]
        };

        // txt_norm + per-block context only when not cached.
        let enc_post_norm: Option<Tensor> = match kv_cache {
            Some(_) => None,
            None => {
                let enc = encoder_hidden_states.unwrap();
                let enc_dims = enc.shape().dims();
                let (enc_b, enc_s, enc_d) = (enc_dims[0], enc_dims[1], enc_dims[2]);
                let enc_flat = enc.reshape(&[enc_b * enc_s, enc_d])?;
                let enc_normed = rms_norm_bf16(&enc_flat, Some(&self.txt_norm_w), cfg.eps)?;
                Some(enc_normed.reshape(&[enc_b, enc_s, enc_d])?)
            }
        };

        // temb
        let temb = self.time_text_embed(timestep)?;

        // 3D RoPE (img + txt). When cached, txt RoPE values are baked into the
        // cache already, but we still need img cos/sin every step.
        let (img_cos, img_sin, txt_cos, txt_sin) = build_nucleus_3d_rope(
            img_shapes,
            s_txt,
            cfg.axes_dims_rope,
            cfg.rope_theta,
            true,
            x.device().clone(),
        )?;

        for (i, block) in self.blocks.iter().enumerate() {
            let cached = kv_cache.map(|c| {
                let (k, v) = &c.entries[i];
                (k, v)
            });
            x = block.forward_with_cache(
                &x,
                enc_post_norm.as_ref(),
                &temb,
                &img_cos,
                &img_sin,
                Some(&txt_cos),
                Some(&txt_sin),
                cached,
                None,
                cfg,
            )?;
        }

        // norm_out: AdaLayerNormContinuous(layer_norm, no affine, eps=1e-6).
        // Formula: emb = linear(silu(temb)); scale, shift = emb.chunk(2, -1);
        //          out = LN(x) * (1+scale) + shift
        let temb_silu = temb.silu()?;
        let temb_3d = temb_silu.unsqueeze(1)?; // (B, 1, D)
        let mod_emb = fused_linear3d_native(&temb_3d, &self.norm_out_w, Some(&self.norm_out_b))?;
        let last = mod_emb.shape().dims().len() - 1;
        let chunks = mod_emb.chunk(2, last)?;
        let scale = &chunks[0]; // (B, 1, D) — first half is scale (NOT shift)
        let shift = &chunks[1]; // (B, 1, D)
        let x_normed = layer_norm(&x, &[cfg.hidden_size], None, None, cfg.eps)?;
        let scale_p1 = scale.add_scalar(1.0)?;
        let x_scaled = x_normed.mul(&scale_p1)?;
        let x_modulated = x_scaled.add(shift)?;

        // proj_out: (B, S, D) -> (B, S, patch² * out_channels = 64)
        fused_linear3d_native(&x_modulated, &self.proj_out_w, None)
    }
}

/// Build the per-axis Nucleus 3D RoPE tables exactly mirroring
/// `NucleusMoEEmbedRope.forward(scale_rope=True)` for the image stream and
/// the per-layer fixed-offset text stream.
///
/// Returns `(img_cos, img_sin, txt_cos, txt_sin)` BF16 each shaped
/// `[1, 1, S, head_dim/2]` for direct consumption by `rope_fused_bf16`.
fn build_nucleus_3d_rope(
    img_shapes: (usize, usize, usize),
    max_txt_seq_len: usize,
    axes_dims: [usize; 3],
    theta: f32,
    scale_rope: bool,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (f, h, w) = img_shapes;
    let head_dim = axes_dims[0] + axes_dims[1] + axes_dims[2];
    if head_dim % 2 != 0 {
        return Err(Error::InvalidOperation(format!(
            "build_nucleus_3d_rope: head_dim {head_dim} must be even"
        )));
    }
    let half = head_dim / 2;

    // Per-axis position lists. For img:
    //   t (frame):    pos_index 0..f
    //   h (height):   if scale_rope: signed [-h/2, h/2) i.e. neg first then pos
    //                 else: 0..h
    //   w (width):    same as h
    // For txt: positions = max_vid_index + arange(max_txt_seq_len), where
    //   max_vid_index = max(h//2, w//2) if scale_rope else max(h, w).
    let pos_t: Vec<f32> = (0..f).map(|i| i as f32).collect();
    let pos_h: Vec<f32> = if scale_rope {
        let neg_count = h - h / 2;
        let mut v = Vec::with_capacity(h);
        for i in 0..neg_count {
            // neg_index = arange(4096).flip(0) * -1 - 1, sliced from the
            // tail. We just emit (-(neg_count - i)).
            v.push(-(neg_count as f32) + i as f32);
        }
        for i in 0..(h / 2) {
            v.push(i as f32);
        }
        v
    } else {
        (0..h).map(|i| i as f32).collect()
    };
    let pos_w: Vec<f32> = if scale_rope {
        let neg_count = w - w / 2;
        let mut v = Vec::with_capacity(w);
        for i in 0..neg_count {
            v.push(-(neg_count as f32) + i as f32);
        }
        for i in 0..(w / 2) {
            v.push(i as f32);
        }
        v
    } else {
        (0..w).map(|i| i as f32).collect()
    };

    let max_vid_index = if scale_rope {
        (h / 2).max(w / 2)
    } else {
        h.max(w)
    };
    let pos_txt: Vec<f32> = (0..max_txt_seq_len)
        .map(|i| (max_vid_index + i) as f32)
        .collect();

    // Per-axis frequency vectors. axes_dims[axis] is the *real* element
    // count; the half is `axes_dims[axis] / 2` complex pairs.
    let mut img_cos = vec![0.0f32; f * h * w * half];
    let mut img_sin = vec![0.0f32; f * h * w * half];
    let mut txt_cos = vec![0.0f32; max_txt_seq_len * half];
    let mut txt_sin = vec![0.0f32; max_txt_seq_len * half];

    let mut axis_offset = 0;
    for (axis_idx, &axis_dim) in axes_dims.iter().enumerate() {
        let half_axis = axis_dim / 2;
        let inv_freqs: Vec<f32> = (0..half_axis)
            .map(|i| 1.0_f32 / theta.powf(i as f32 / half_axis as f32))
            .collect();

        // For img stream, this axis's positions vary along (f, h, w):
        //   axis 0: pos = pos_t[fi]
        //   axis 1: pos = pos_h[hi]
        //   axis 2: pos = pos_w[wi]
        for fi in 0..f {
            for hi in 0..h {
                for wi in 0..w {
                    let pos = match axis_idx {
                        0 => pos_t[fi],
                        1 => pos_h[hi],
                        _ => pos_w[wi],
                    };
                    let token_idx = (fi * h + hi) * w + wi;
                    for (k, &freq) in inv_freqs.iter().enumerate() {
                        let ang = pos * freq;
                        img_cos[token_idx * half + axis_offset + k] = ang.cos();
                        img_sin[token_idx * half + axis_offset + k] = ang.sin();
                    }
                }
            }
        }

        // Text stream: only axis 0 carries position; axes 1,2 use position 0
        // ... actually no: looking at diffusers code,
        //   txt_freqs = self.pos_freqs[max_vid_index + arange(text_seq_len)]
        // and `self.pos_freqs` is the row-stacked all-axes frequency table.
        // i.e. text uses pos = max_vid_index + i for ALL axes — not just t.
        // The per-axis split happens by column slicing of pos_freqs.
        for (i, &pos) in pos_txt.iter().enumerate() {
            for (k, &freq) in inv_freqs.iter().enumerate() {
                let ang = pos * freq;
                txt_cos[i * half + axis_offset + k] = ang.cos();
                txt_sin[i * half + axis_offset + k] = ang.sin();
            }
        }

        axis_offset += half_axis;
    }

    let img_seq = f * h * w;
    let to_bf16 = |data: Vec<f32>, seq: usize| -> Result<Tensor> {
        Tensor::from_vec_dtype(
            data,
            flame_core::Shape::from_dims(&[1, 1, seq, half]),
            device.clone(),
            DType::BF16,
        )
    };
    let img_cos_t = to_bf16(img_cos, img_seq)?;
    let img_sin_t = to_bf16(img_sin, img_seq)?;
    let txt_cos_t = to_bf16(txt_cos, max_txt_seq_len)?;
    let txt_sin_t = to_bf16(txt_sin, max_txt_seq_len)?;
    Ok((img_cos_t, img_sin_t, txt_cos_t, txt_sin_t))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply `RMSNorm(weight, eps)` over the last axis. Reshape-flatten avoids
/// allocating an `RMSNorm` struct; mirrors `qwen3_encoder::rms_norm_apply`.
fn qk_norm(
    x_4d: &Tensor, // (B, S, H, head_dim)
    weight: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Tensor> {
    let flat = x_4d.reshape(&[rows, cols])?;
    rms_norm_bf16(&flat, Some(weight), eps)
}

/// Replicate KV heads `n_rep` times along axis=1 of `[B, H_kv, S, head_dim]`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    let b = dims[0];
    let h_kv = dims[1];
    let n = dims[2];
    let d = dims[3];
    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    let stacked = Tensor::stack(&copies, 2)?; // (B, H_kv, n_rep, N, D)
    stacked.reshape(&[b, h_kv * n_rep, n, d])
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

    /// Pull a tensor out of the fixture map by key, with a clear error on miss.
    fn take(map: &mut HashMap<String, Tensor>, key: &str) -> Tensor {
        map.remove(key)
            .unwrap_or_else(|| panic!("fixture missing key: {key}"))
    }

    fn build_block_dense(map: &mut HashMap<String, Tensor>) -> NucleusBlock {
        let mlp = NucleusMlpWeights::Dense {
            gate_up_w: take(map, "block.img_mlp.net.0.proj.weight"),
            down_w: take(map, "block.img_mlp.net.2.weight"),
        };
        build_block_with_mlp(map, mlp)
    }

    fn build_block_moe(map: &mut HashMap<String, Tensor>, capacity_factor: f32) -> NucleusBlock {
        let mlp = NucleusMlpWeights::Moe {
            gate_w: take(map, "block.img_mlp.gate.weight"),
            gate_up_w: take(map, "block.img_mlp.experts.gate_up_proj"),
            down_w: take(map, "block.img_mlp.experts.down_proj"),
            shared_gate_up_w: take(map, "block.img_mlp.shared_expert.net.0.proj.weight"),
            shared_down_w: take(map, "block.img_mlp.shared_expert.net.2.weight"),
            capacity_factor,
        };
        build_block_with_mlp(map, mlp)
    }

    fn build_block_with_mlp(
        map: &mut HashMap<String, Tensor>,
        mlp: NucleusMlpWeights,
    ) -> NucleusBlock {
        build_block_prefixed(map, "block.", mlp)
    }

    fn build_block_prefixed(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        mlp: NucleusMlpWeights,
    ) -> NucleusBlock {
        // Drop the orphan norm_added_q if present (loaded but unused).
        let _ = map.remove(&format!("{prefix}attn.norm_added_q.weight"));
        NucleusBlock {
            mod_w: take(map, &format!("{prefix}img_mod.1.weight")),
            mod_b: take(map, &format!("{prefix}img_mod.1.bias")),
            enc_proj_w: take(map, &format!("{prefix}encoder_proj.weight")),
            enc_proj_b: take(map, &format!("{prefix}encoder_proj.bias")),
            to_q_w: take(map, &format!("{prefix}attn.to_q.weight")),
            to_k_w: take(map, &format!("{prefix}attn.to_k.weight")),
            to_v_w: take(map, &format!("{prefix}attn.to_v.weight")),
            to_out_w: take(map, &format!("{prefix}attn.to_out.0.weight")),
            add_k_w: take(map, &format!("{prefix}attn.add_k_proj.weight")),
            add_v_w: take(map, &format!("{prefix}attn.add_v_proj.weight")),
            norm_q_w: take(map, &format!("{prefix}attn.norm_q.weight")),
            norm_k_w: take(map, &format!("{prefix}attn.norm_k.weight")),
            norm_added_k_w: take(map, &format!("{prefix}attn.norm_added_k.weight")),
            mlp,
        }
    }

    fn build_dit_from_fixture(
        map: &mut HashMap<String, Tensor>,
        cfg: NucleusConfig,
    ) -> NucleusDit {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let prefix = format!("model.transformer_blocks.{i}.");
            let cap_factor = cfg.capacity_factors[i];
            let mlp = if cap_factor == 0.0 {
                NucleusMlpWeights::Dense {
                    gate_up_w: take(map, &format!("{prefix}img_mlp.net.0.proj.weight")),
                    down_w: take(map, &format!("{prefix}img_mlp.net.2.weight")),
                }
            } else {
                NucleusMlpWeights::Moe {
                    gate_w: take(map, &format!("{prefix}img_mlp.gate.weight")),
                    gate_up_w: take(map, &format!("{prefix}img_mlp.experts.gate_up_proj")),
                    down_w: take(map, &format!("{prefix}img_mlp.experts.down_proj")),
                    shared_gate_up_w: take(
                        map,
                        &format!("{prefix}img_mlp.shared_expert.net.0.proj.weight"),
                    ),
                    shared_down_w: take(
                        map,
                        &format!("{prefix}img_mlp.shared_expert.net.2.weight"),
                    ),
                    capacity_factor: cap_factor,
                }
            };
            blocks.push(build_block_prefixed(map, &prefix, mlp));
        }
        NucleusDit {
            config: cfg,
            img_in_w: take(map, "model.img_in.weight"),
            img_in_b: take(map, "model.img_in.bias"),
            time_proj_l1_w: take(
                map,
                "model.time_text_embed.timestep_embedder.linear_1.weight",
            ),
            time_proj_l1_b: take(
                map,
                "model.time_text_embed.timestep_embedder.linear_1.bias",
            ),
            time_proj_l2_w: take(
                map,
                "model.time_text_embed.timestep_embedder.linear_2.weight",
            ),
            time_proj_l2_b: take(
                map,
                "model.time_text_embed.timestep_embedder.linear_2.bias",
            ),
            time_norm_w: take(map, "model.time_text_embed.norm.weight"),
            txt_norm_w: take(map, "model.txt_norm.weight"),
            norm_out_w: take(map, "model.norm_out.linear.weight"),
            norm_out_b: take(map, "model.norm_out.linear.bias"),
            proj_out_w: take(map, "model.proj_out.weight"),
            blocks,
        }
    }

    fn meta_usize(map: &HashMap<String, Tensor>, key: &str) -> usize {
        let t = map
            .get(key)
            .unwrap_or_else(|| panic!("fixture missing meta {key}"));
        let v = t.to_vec_f32().expect("meta to_vec_f32");
        v[0] as usize
    }

    fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        assert_eq!(a.len(), b.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut max_rel = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            sum_abs += diff as f64;
            let denom = x.abs().max(y.abs()).max(1e-6);
            let rel = diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        (max_abs, max_rel, (sum_abs / a.len() as f64) as f32)
    }

    /// 99.9th-percentile absolute error. Robust to sparse BF16 scatter-add
    /// outliers (a few positions with up to ~32 ULP error from PyTorch's
    /// round-then-add MoE scatter vs flame's F32-accumulate-then-round).
    fn p999_abs(a: &[f32], b: &[f32]) -> f32 {
        let mut diffs: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).collect();
        diffs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let idx = (diffs.len() as f64 * 0.999) as usize;
        diffs[idx.min(diffs.len() - 1)]
    }

    fn run_block_parity(fixture_name: &str, build: impl FnOnce(&mut HashMap<String, Tensor>) -> NucleusBlock) {
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/nucleus")
            .join(fixture_name);
        if !fixture_path.exists() {
            eprintln!(
                "fixture missing — generate with `python3 scripts/generate_*.py`: {}",
                fixture_path.display()
            );
            return;
        }

        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture_path, &device).expect("load fixture");

        let cfg = NucleusConfig::nucleus_image_default();

        let inp_x = take(&mut map, "inputs.hidden_states");
        let inp_enc = take(&mut map, "inputs.encoder_hidden_states");
        let inp_temb = take(&mut map, "inputs.temb");
        let img_cos = take(&mut map, "rope.img_cos");
        let img_sin = take(&mut map, "rope.img_sin");
        let txt_cos = take(&mut map, "rope.txt_cos");
        let txt_sin = take(&mut map, "rope.txt_sin");
        let expected = take(&mut map, "expected.output");

        let block = build(&mut map);

        let out = block
            .forward(
                &inp_x, &inp_enc, &inp_temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None, &cfg,
            )
            .expect("forward");

        let got = out.to_vec_f32().expect("got to_vec_f32");
        let exp = expected.to_vec_f32().expect("expected to_vec_f32");
        let (max_abs, max_rel, mean_abs) = diff_stats(&got, &exp);
        let p999 = p999_abs(&got, &exp);
        eprintln!(
            "{fixture_name}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} mean_abs={mean_abs:.4e} p999={p999:.4e}"
        );
        // Mean-abs floor is the dominant signal of correctness.
        assert!(mean_abs < 1e-3, "mean_abs {mean_abs} exceeds floor 1e-3");
        // 99.9th percentile catches systematic drift while ignoring sparse
        // BF16 scatter-add outliers (round-then-add vs add-then-round).
        assert!(p999 < 1e-2, "p999 {p999} exceeds floor 1e-2");
        // Soft cap on max — only trips on a true bug, not BF16 ULP noise.
        assert!(max_abs < 1e-1, "max_abs {max_abs} exceeds floor 1e-1");
    }

    #[test]
    fn nucleus_dense_block_parity_vs_pytorch() {
        // Locate fixture relative to this crate's manifest.
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/nucleus/dense_block_parity.safetensors");
        if !fixture_path.exists() {
            eprintln!(
                "fixture missing — generate with `python3 scripts/generate_nucleus_dense_block.py`"
            );
            // Treat as skip rather than fail until the fixture is committed.
            return;
        }

        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture_path, &device).expect("load fixture");

        let cfg = NucleusConfig::nucleus_image_default();

        let inp_x = take(&mut map, "inputs.hidden_states");
        let inp_enc = take(&mut map, "inputs.encoder_hidden_states");
        let inp_temb = take(&mut map, "inputs.temb");
        let img_cos = take(&mut map, "rope.img_cos");
        let img_sin = take(&mut map, "rope.img_sin");
        let txt_cos = take(&mut map, "rope.txt_cos");
        let txt_sin = take(&mut map, "rope.txt_sin");
        let expected = take(&mut map, "expected.output");

        // Verify config matches fixture metadata
        assert_eq!(meta_usize(&map, "meta.D"), cfg.hidden_size);
        assert_eq!(meta_usize(&map, "meta.heads"), cfg.num_heads);
        assert_eq!(meta_usize(&map, "meta.kv_heads"), cfg.num_kv_heads);
        assert_eq!(meta_usize(&map, "meta.head_dim"), cfg.head_dim);
        assert_eq!(meta_usize(&map, "meta.joint_dim"), cfg.joint_attention_dim);
        assert_eq!(meta_usize(&map, "meta.dense_inner_dim"), cfg.dense_inner_dim());

        let block = build_block_dense(&mut map);

        let out = block
            .forward(
                &inp_x, &inp_enc, &inp_temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None, &cfg,
            )
            .expect("forward");

        // Compare in F32 to avoid double BF16-cast noise on top of the math.
        let got = out.to_vec_f32().expect("got to_vec_f32");
        let exp = expected.to_vec_f32().expect("expected to_vec_f32");
        let (max_abs, max_rel, mean_abs) = diff_stats(&got, &exp);
        eprintln!("nucleus_dense_block_parity: max_abs={max_abs:.4e} max_rel={max_rel:.4e} mean_abs={mean_abs:.4e}");

        // Floor mirrors Phase 5 — same BF16 numerics regime.
        assert!(max_abs < 1e-2, "max_abs {max_abs} exceeds floor 1e-2");
        assert!(mean_abs < 1e-3, "mean_abs {mean_abs} exceeds floor 1e-3");
    }

    #[test]
    fn nucleus_dit_small_parity_vs_pytorch() {
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/nucleus/dit_small_parity.safetensors");
        if !fixture_path.exists() {
            eprintln!(
                "fixture missing — generate with `python3 scripts/generate_nucleus_dit_small.py`"
            );
            return;
        }

        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture_path, &device).expect("load fixture");

        // Tiny config (matches scripts/generate_nucleus_dit_small.py)
        let cfg = NucleusConfig {
            num_layers: 5,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 32,
            hidden_size: 128,
            joint_attention_dim: 64,
            mlp_ratio: 4.0,
            axes_dims_rope: [4, 14, 14],
            rope_theta: 10_000.0,
            num_experts: 4,
            moe_intermediate_dim: 16,
            capacity_factors: vec![0.0, 0.0, 0.0, 4.0, 2.0],
            use_sigmoid: false,
            route_scale: 2.5,
            patch_size: 2,
            in_channels: 8,
            out_channels: 2,
            eps: 1e-6,
        };

        let inp_x = take(&mut map, "inputs.hidden_states");
        let inp_enc = take(&mut map, "inputs.encoder_hidden_states");
        let inp_t = take(&mut map, "inputs.timestep");
        let expected = take(&mut map, "expected.output");

        let dit = build_dit_from_fixture(&mut map, cfg);

        let out = dit
            .forward(&inp_x, &inp_enc, None, &inp_t, (1, 4, 4))
            .expect("forward");

        let got = out.to_vec_f32().expect("got to_vec_f32");
        let exp = expected.to_vec_f32().expect("expected to_vec_f32");
        let (max_abs, max_rel, mean_abs) = diff_stats(&got, &exp);
        let p999 = p999_abs(&got, &exp);
        eprintln!(
            "dit_small_parity: max_abs={max_abs:.4e} max_rel={max_rel:.4e} mean_abs={mean_abs:.4e} p999={p999:.4e}"
        );
        assert!(mean_abs < 1e-3, "mean_abs {mean_abs} exceeds floor 1e-3");
        assert!(p999 < 1e-2, "p999 {p999} exceeds floor 1e-2");
        assert!(max_abs < 1e-1, "max_abs {max_abs} exceeds floor 1e-1");
    }

    #[test]
    fn nucleus_dit_kv_cache_roundtrip() {
        // Reuse the small-DiT fixture: run forward twice — uncached then
        // cached — and assert bit-identical output. Verifies the cached
        // path doesn't drift from the uncached recipe.
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/nucleus/dit_small_parity.safetensors");
        if !fixture_path.exists() {
            eprintln!("fixture missing — generate via 6.4 script first");
            return;
        }

        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture_path, &device).expect("load fixture");

        let cfg = NucleusConfig {
            num_layers: 5,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 32,
            hidden_size: 128,
            joint_attention_dim: 64,
            mlp_ratio: 4.0,
            axes_dims_rope: [4, 14, 14],
            rope_theta: 10_000.0,
            num_experts: 4,
            moe_intermediate_dim: 16,
            capacity_factors: vec![0.0, 0.0, 0.0, 4.0, 2.0],
            use_sigmoid: false,
            route_scale: 2.5,
            patch_size: 2,
            in_channels: 8,
            out_channels: 2,
            eps: 1e-6,
        };

        let inp_x = take(&mut map, "inputs.hidden_states");
        let inp_enc = take(&mut map, "inputs.encoder_hidden_states");
        let inp_t = take(&mut map, "inputs.timestep");
        let _expected = take(&mut map, "expected.output");

        let dit = build_dit_from_fixture(&mut map, cfg);

        let out_uncached = dit
            .forward(&inp_x, &inp_enc, None, &inp_t, (1, 4, 4))
            .expect("forward uncached");

        let kv_cache = dit
            .compute_kv_cache(&inp_enc, (1, 4, 4))
            .expect("compute_kv_cache");

        let out_cached = dit
            .forward_cached(&inp_x, &inp_t, (1, 4, 4), &kv_cache)
            .expect("forward cached");

        let a = out_uncached.to_vec_f32().expect("uncached vec");
        let b = out_cached.to_vec_f32().expect("cached vec");
        let (max_abs, _max_rel, mean_abs) = diff_stats(&a, &b);
        eprintln!(
            "kv_cache_roundtrip: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} (should be 0)"
        );
        // Cached path must be bit-identical to uncached.
        assert!(max_abs == 0.0, "cached path diverges: max_abs {max_abs}");
    }

    #[test]
    fn nucleus_moe_block_parity_vs_pytorch() {
        // Layer-3-style MoE block: capacity_factor=4.0 (matches diffusers
        // `capacity_factors` = [0,0,0, 4,4, 2,2,...]).
        run_block_parity("moe_block_parity.safetensors", |map| {
            super::tests::build_block_moe(map, 4.0)
        });
    }
}

// ---------------------------------------------------------------------------
// Runtime loader (Phase 6.6) — real-weight inference with BlockOffloader for
// the 29 MoE expert weight layers
// ---------------------------------------------------------------------------

/// Per-block streaming classifier: the BlockOffloader streams ONLY the MoE
/// expert weights (the bulk of the 17 B parameter count). Everything else
/// — modulation, encoder_proj, attention, dense FFN, shared expert, gate
/// router — stays resident on GPU.
///
/// MoE block index mapping: DiT layer `i` for `i in 3..32` maps to offloader
/// block `i - 3` (so 0..29). Dense layers (`i in 0..3`) never touch the
/// offloader.
pub struct NucleusFacilitator {
    /// Number of MoE blocks (= DiT MoE layer count = 29 for the default model).
    pub num_moe_blocks: usize,
}

impl NucleusFacilitator {
    pub fn new(cfg: &NucleusConfig) -> Self {
        let n = cfg
            .capacity_factors
            .iter()
            .filter(|&&c| c > 0.0)
            .count();
        Self { num_moe_blocks: n }
    }
}

impl BlockFacilitator for NucleusFacilitator {
    fn block_count(&self) -> usize {
        self.num_moe_blocks
    }

    fn classify_key(&self, key: &str) -> Option<usize> {
        // transformer_blocks.{i}.img_mlp.experts.{gate_up_proj|down_proj}
        let rest = key.strip_prefix("transformer_blocks.")?;
        let dot = rest.find('.')?;
        let layer_idx: usize = rest[..dot].parse().ok()?;
        if layer_idx < 3 {
            return None; // dense layers stay resident
        }
        let suffix = &rest[dot + 1..];
        if suffix == "img_mlp.experts.gate_up_proj" || suffix == "img_mlp.experts.down_proj" {
            Some(layer_idx - 3)
        } else {
            None
        }
    }
}

/// Real-weight runtime DiT for Nucleus-Image. Holds resident shared weights
/// (~3.5 GB BF16 — top-level scaffolding + all per-block attention,
/// modulation, encoder_proj, dense FFN, gate router, shared expert) and
/// streams the bulk MoE expert weights (~15.3 GB) on demand from pinned host
/// RAM via `BlockOffloader`.
///
/// Loading two-phase:
///   1. `BlockOffloader::load` reads each shard once, copying tensors that
///      `NucleusFacilitator::classify_key` accepts into pinned host memory.
///   2. `load_file_filtered` reads each shard again, this time keeping only
///      the keys NOT accepted by the facilitator. These end up resident on
///      GPU.
///
/// The dual-pass is cheap thanks to mmap: each shard's bytes are touched
/// twice but only the kept tensors are actually copied/decoded.
pub struct NucleusInferDit {
    pub config: NucleusConfig,
    /// Resident weights, keyed by full diffusers name
    /// (`transformer_blocks.{i}.<...>` or top-level).
    pub resident: HashMap<String, Tensor>,
    /// Streams MoE expert weights for `transformer_blocks.{i}.img_mlp.experts.{gate_up_proj|down_proj}`.
    /// Indexed 0..29 corresponding to DiT layers 3..32.
    pub offloader: BlockOffloader,
    pub device: Arc<CudaDevice>,
}

impl NucleusInferDit {
    /// Load all weights from `<snapshot>/transformer/`. The directory must
    /// contain `diffusion_pytorch_model.safetensors.index.json` and the
    /// `diffusion_pytorch_model-{NNNNN}-of-{TOTAL}.safetensors` shards it
    /// references.
    pub fn load(transformer_dir: &Path, device: Arc<CudaDevice>) -> Result<Self> {
        let config = NucleusConfig::nucleus_image_default();
        let facilitator = NucleusFacilitator::new(&config);

        // 1) Discover shards via the index.
        let index_path = transformer_dir.join("diffusion_pytorch_model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path).map_err(|e| {
            Error::Io(format!(
                "NucleusInferDit::load: cannot read {:?}: {e}",
                index_path
            ))
        })?;
        let index: serde_json::Value = serde_json::from_str(&index_text).map_err(|e| {
            Error::InvalidInput(format!(
                "NucleusInferDit::load: malformed index json {:?}: {e}",
                index_path
            ))
        })?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "NucleusInferDit::load: index missing 'weight_map' at {:?}",
                    index_path
                ))
            })?;
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        shard_names.sort();
        let shard_paths: Vec<PathBuf> = shard_names
            .iter()
            .map(|n| transformer_dir.join(n))
            .collect();
        let shard_strs: Vec<String> = shard_paths
            .iter()
            .map(|p| {
                p.to_str()
                    .map(str::to_string)
                    .ok_or_else(|| Error::Io(format!("non-utf8 shard path: {:?}", p)))
            })
            .collect::<Result<Vec<_>>>()?;
        let shard_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();

        // 2) Stream MoE expert weights into pinned RAM.
        let offloader = BlockOffloader::load(&shard_refs, &facilitator, device.clone())
            .map_err(|e| {
                Error::InvalidInput(format!("NucleusInferDit BlockOffloader::load: {e}"))
            })?;

        // 3) Resident weights: filter every shard for keys the facilitator
        //    rejects. The orphan `attn.norm_added_q.weight` is loaded along
        //    with everyone else and ignored downstream (see audit gotcha #12).
        let mut resident: HashMap<String, Tensor> = HashMap::new();
        for path in &shard_paths {
            let part = load_file_filtered(path, &device, |key| {
                facilitator.classify_key(key).is_none()
            })?;
            resident.extend(part);
        }

        // 4) Validate critical shared keys.
        let must_have: &[&str] = &[
            "img_in.weight",
            "img_in.bias",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.norm.weight",
            "txt_norm.weight",
            "norm_out.linear.weight",
            "norm_out.linear.bias",
            "proj_out.weight",
        ];
        for k in must_have {
            if !resident.contains_key(*k) {
                return Err(Error::InvalidInput(format!(
                    "NucleusInferDit::load: missing required key {k}"
                )));
            }
        }
        for i in 0..config.num_layers {
            for suffix in &[
                "img_mod.1.weight",
                "img_mod.1.bias",
                "encoder_proj.weight",
                "encoder_proj.bias",
                "attn.to_q.weight",
                "attn.to_k.weight",
                "attn.to_v.weight",
                "attn.to_out.0.weight",
                "attn.add_k_proj.weight",
                "attn.add_v_proj.weight",
                "attn.norm_q.weight",
                "attn.norm_k.weight",
                "attn.norm_added_k.weight",
            ] {
                let key = format!("transformer_blocks.{i}.{suffix}");
                if !resident.contains_key(&key) {
                    return Err(Error::InvalidInput(format!(
                        "NucleusInferDit::load: missing per-layer key {key}"
                    )));
                }
            }
            if i < 3 {
                for suffix in &["img_mlp.net.0.proj.weight", "img_mlp.net.2.weight"] {
                    let key = format!("transformer_blocks.{i}.{suffix}");
                    if !resident.contains_key(&key) {
                        return Err(Error::InvalidInput(format!(
                            "NucleusInferDit::load: missing dense FFN key {key}"
                        )));
                    }
                }
            } else {
                for suffix in &[
                    "img_mlp.gate.weight",
                    "img_mlp.shared_expert.net.0.proj.weight",
                    "img_mlp.shared_expert.net.2.weight",
                ] {
                    let key = format!("transformer_blocks.{i}.{suffix}");
                    if !resident.contains_key(&key) {
                        return Err(Error::InvalidInput(format!(
                            "NucleusInferDit::load: missing MoE non-stream key {key}"
                        )));
                    }
                }
            }
        }

        log::info!(
            "[NucleusInferDit] loaded: {} resident keys, {} MoE blocks streaming",
            resident.len(),
            offloader.block_count()
        );
        eprintln!(
            "[NucleusInferDit] loaded: {} resident keys, {} MoE blocks streaming",
            resident.len(),
            offloader.block_count()
        );

        Ok(Self {
            config,
            resident,
            offloader,
            device,
        })
    }

    fn r(&self, key: &str) -> Result<Tensor> {
        self.resident
            .get(key)
            .cloned()
            .ok_or_else(|| Error::InvalidInput(format!("NucleusInferDit: missing resident {key}")))
    }

    /// Build a `NucleusBlock` for layer `layer_idx`. For dense layers, all
    /// weights come from the resident map. For MoE layers, the streaming
    /// `experts.gate_up_proj` and `.down_proj` come from the supplied
    /// `streamed` map (keys still carry the full `transformer_blocks.{i}.`
    /// prefix; the BlockOffloader preserves diffusers-named keys).
    fn build_block_for_layer(
        &self,
        layer_idx: usize,
        streamed: Option<&HashMap<String, Tensor>>,
    ) -> Result<NucleusBlock> {
        let prefix = format!("transformer_blocks.{layer_idx}.");
        let r = |suffix: &str| self.r(&format!("{prefix}{suffix}"));

        let mlp = if layer_idx < 3 {
            NucleusMlpWeights::Dense {
                gate_up_w: r("img_mlp.net.0.proj.weight")?,
                down_w: r("img_mlp.net.2.weight")?,
            }
        } else {
            let s = streamed.ok_or_else(|| {
                Error::InvalidInput(format!(
                    "build_block_for_layer({layer_idx}): MoE requires streamed weights"
                ))
            })?;
            let take_streamed = |suffix: &str| -> Result<Tensor> {
                let key = format!("{prefix}{suffix}");
                s.get(&key).cloned().ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "build_block_for_layer: missing streamed key {key}"
                    ))
                })
            };
            NucleusMlpWeights::Moe {
                gate_w: r("img_mlp.gate.weight")?,
                gate_up_w: take_streamed("img_mlp.experts.gate_up_proj")?,
                down_w: take_streamed("img_mlp.experts.down_proj")?,
                shared_gate_up_w: r("img_mlp.shared_expert.net.0.proj.weight")?,
                shared_down_w: r("img_mlp.shared_expert.net.2.weight")?,
                capacity_factor: self.config.capacity_factors[layer_idx],
            }
        };

        Ok(NucleusBlock {
            mod_w: r("img_mod.1.weight")?,
            mod_b: r("img_mod.1.bias")?,
            enc_proj_w: r("encoder_proj.weight")?,
            enc_proj_b: r("encoder_proj.bias")?,
            to_q_w: r("attn.to_q.weight")?,
            to_k_w: r("attn.to_k.weight")?,
            to_v_w: r("attn.to_v.weight")?,
            to_out_w: r("attn.to_out.0.weight")?,
            add_k_w: r("attn.add_k_proj.weight")?,
            add_v_w: r("attn.add_v_proj.weight")?,
            norm_q_w: r("attn.norm_q.weight")?,
            norm_k_w: r("attn.norm_k.weight")?,
            norm_added_k_w: r("attn.norm_added_k.weight")?,
            mlp,
        })
    }

    /// Build the per-block text K/V cache. Every block's encoder_proj +
    /// add_k/v + norm_added_k + RoPE-on-K runs once over the encoder
    /// hidden states; the resulting K/V are reused unchanged across all
    /// 50 denoise steps. Only resident weights are needed.
    pub fn compute_kv_cache(
        &self,
        encoder_hidden_states: &Tensor,
        img_shapes: (usize, usize, usize),
    ) -> Result<NucleusKVCache> {
        let cfg = &self.config;

        let enc_dims = encoder_hidden_states.shape().dims();
        let (enc_b, enc_s, enc_d) = (enc_dims[0], enc_dims[1], enc_dims[2]);
        let txt_norm_w = self.r("txt_norm.weight")?;
        let enc_flat = encoder_hidden_states.reshape(&[enc_b * enc_s, enc_d])?;
        let enc_normed = rms_norm_bf16(&enc_flat, Some(&txt_norm_w), cfg.eps)?;
        let enc = enc_normed.reshape(&[enc_b, enc_s, enc_d])?;

        let (_img_cos, _img_sin, txt_cos, txt_sin) = build_nucleus_3d_rope(
            img_shapes,
            enc_s,
            cfg.axes_dims_rope,
            cfg.rope_theta,
            true,
            self.device.clone(),
        )?;

        let mut entries = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            // Every block has its own resident encoder_proj + add_k/v.
            let enc_proj_w =
                self.r(&format!("transformer_blocks.{layer_idx}.encoder_proj.weight"))?;
            let enc_proj_b =
                self.r(&format!("transformer_blocks.{layer_idx}.encoder_proj.bias"))?;
            let add_k_w =
                self.r(&format!("transformer_blocks.{layer_idx}.attn.add_k_proj.weight"))?;
            let add_v_w =
                self.r(&format!("transformer_blocks.{layer_idx}.attn.add_v_proj.weight"))?;
            let norm_added_k_w =
                self.r(&format!("transformer_blocks.{layer_idx}.attn.norm_added_k.weight"))?;

            let context = fused_linear3d_native(&enc, &enc_proj_w, Some(&enc_proj_b))?;

            // Build a tiny stub block so we can reuse `compute_txt_kv`.
            let stub = NucleusBlock {
                // The compute_txt_kv path only touches add_k_w, add_v_w,
                // norm_added_k_w. Other weights are placeholders here —
                // we use the real ones to keep the type inhabited.
                mod_w: self.r(&format!("transformer_blocks.{layer_idx}.img_mod.1.weight"))?,
                mod_b: self.r(&format!("transformer_blocks.{layer_idx}.img_mod.1.bias"))?,
                enc_proj_w,
                enc_proj_b,
                to_q_w: self.r(&format!("transformer_blocks.{layer_idx}.attn.to_q.weight"))?,
                to_k_w: self.r(&format!("transformer_blocks.{layer_idx}.attn.to_k.weight"))?,
                to_v_w: self.r(&format!("transformer_blocks.{layer_idx}.attn.to_v.weight"))?,
                to_out_w: self.r(&format!(
                    "transformer_blocks.{layer_idx}.attn.to_out.0.weight"
                ))?,
                add_k_w,
                add_v_w,
                norm_q_w: self.r(&format!(
                    "transformer_blocks.{layer_idx}.attn.norm_q.weight"
                ))?,
                norm_k_w: self.r(&format!(
                    "transformer_blocks.{layer_idx}.attn.norm_k.weight"
                ))?,
                norm_added_k_w,
                // dense placeholder — unused in compute_txt_kv
                mlp: if layer_idx < 3 {
                    NucleusMlpWeights::Dense {
                        gate_up_w: self.r(&format!(
                            "transformer_blocks.{layer_idx}.img_mlp.net.0.proj.weight"
                        ))?,
                        down_w: self.r(&format!(
                            "transformer_blocks.{layer_idx}.img_mlp.net.2.weight"
                        ))?,
                    }
                } else {
                    // Use a placeholder Dense variant so compute_txt_kv
                    // (which doesn't touch MLP) doesn't need the streamed
                    // tensors. Take any same-shape resident tensor.
                    NucleusMlpWeights::Dense {
                        gate_up_w: self.r(&format!(
                            "transformer_blocks.{layer_idx}.img_mlp.shared_expert.net.0.proj.weight"
                        ))?,
                        down_w: self.r(&format!(
                            "transformer_blocks.{layer_idx}.img_mlp.shared_expert.net.2.weight"
                        ))?,
                    }
                },
            };

            let (k, v) = stub.compute_txt_kv(&context, &txt_cos, &txt_sin, cfg)?;
            entries.push((k, v));
        }

        Ok(NucleusKVCache {
            entries,
            txt_seq_len: enc_s,
        })
    }

    /// One forward pass with the pre-computed KV cache. Streams the MoE
    /// expert weights one block at a time via the offloader's
    /// prefetch / await_block API.
    pub fn forward_cached(
        &mut self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        img_shapes: (usize, usize, usize),
        kv_cache: &NucleusKVCache,
    ) -> Result<Tensor> {
        // Borrow-split so the offloader can be `&mut` while config + resident
        // stay `&`. Same shape as SenseNova's forward.
        let cfg_ref = &self.config;
        let resident_ref = &self.resident;
        let device = self.device.clone();
        let offloader = &mut self.offloader;

        let cfg = cfg_ref;

        // ---- img_in ----
        let img_in_w = resident_ref
            .get("img_in.weight")
            .ok_or_else(|| Error::InvalidInput("missing img_in.weight".into()))?;
        let img_in_b = resident_ref
            .get("img_in.bias")
            .ok_or_else(|| Error::InvalidInput("missing img_in.bias".into()))?;
        let mut x = fused_linear3d_native(hidden_states, img_in_w, Some(img_in_b))?;

        // ---- temb = RMSNorm(linear_2(silu(linear_1(time_proj(t))))) ----
        let temb = {
            let dim = cfg.hidden_size;
            let half = dim / 2;
            let max_period = 10_000.0_f64;
            let freq_data: Vec<f32> = (0..half)
                .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
                .collect();
            let freqs = Tensor::from_vec(
                freq_data,
                flame_core::Shape::from_dims(&[1, half]),
                device.clone(),
            )?;
            let t_f32 = timestep.to_dtype(DType::F32)?.mul_scalar(1000.0)?;
            let args = t_f32.unsqueeze(1)?.matmul(&freqs)?;
            let cos_part = args.cos()?;
            let sin_part = args.sin()?;
            let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?.to_dtype(DType::BF16)?;
            let proj_3d = emb.unsqueeze(1)?;
            let l1_w = resident_ref
                .get("time_text_embed.timestep_embedder.linear_1.weight")
                .ok_or_else(|| Error::InvalidInput("missing time linear_1.w".into()))?;
            let l1_b = resident_ref
                .get("time_text_embed.timestep_embedder.linear_1.bias")
                .ok_or_else(|| Error::InvalidInput("missing time linear_1.b".into()))?;
            let l2_w = resident_ref
                .get("time_text_embed.timestep_embedder.linear_2.weight")
                .ok_or_else(|| Error::InvalidInput("missing time linear_2.w".into()))?;
            let l2_b = resident_ref
                .get("time_text_embed.timestep_embedder.linear_2.bias")
                .ok_or_else(|| Error::InvalidInput("missing time linear_2.b".into()))?;
            let norm_w = resident_ref
                .get("time_text_embed.norm.weight")
                .ok_or_else(|| Error::InvalidInput("missing time norm.w".into()))?;
            let h1 = fused_linear3d_native(&proj_3d, l1_w, Some(l1_b))?.silu()?;
            let l2 = fused_linear3d_native(&h1, l2_w, Some(l2_b))?;
            let l2_2d = l2.squeeze(Some(1))?;
            rms_norm_bf16(&l2_2d, Some(norm_w), cfg.eps)?
        };

        // ---- 3D RoPE (img cos/sin) — txt cos/sin are baked into KV cache.
        // We still build txt cos/sin to forward to dense layers via
        // `forward_with_cache`'s shape contract, but the cached path skips
        // their use anyway. Just pass dummy refs.
        let (img_cos, img_sin, txt_cos, txt_sin) = build_nucleus_3d_rope(
            img_shapes,
            kv_cache.txt_seq_len,
            cfg.axes_dims_rope,
            cfg.rope_theta,
            true,
            device.clone(),
        )?;

        let total_layers = cfg.num_layers;
        let num_dense = 3usize;

        // Prefetch the FIRST MoE block ahead of layer 3.
        if cfg.num_layers > num_dense {
            offloader
                .prefetch_block(0)
                .map_err(|e| Error::InvalidInput(format!("prefetch MoE block 0: {e}")))?;
        }

        let view = NucleusInferDitView {
            config: cfg,
            resident: resident_ref,
        };

        for layer_idx in 0..num_dense {
            let block = view.build_block_for_layer(layer_idx, None)?;
            let (k, v) = (&kv_cache.entries[layer_idx].0, &kv_cache.entries[layer_idx].1);
            x = block.forward_with_cache(
                &x,
                None,
                &temb,
                &img_cos,
                &img_sin,
                Some(&txt_cos),
                Some(&txt_sin),
                Some((k, v)),
                None,
                cfg,
            )?;
        }

        for layer_idx in num_dense..total_layers {
            let moe_block_idx = layer_idx - num_dense;
            let raw = offloader.await_block(moe_block_idx).map_err(|e| {
                Error::InvalidInput(format!("await MoE block {moe_block_idx}: {e}"))
            })?;
            // Prefetch next MoE block if any.
            if moe_block_idx + 1 < offloader.block_count() {
                offloader
                    .prefetch_block(moe_block_idx + 1)
                    .map_err(|e| {
                        Error::InvalidInput(format!(
                            "prefetch MoE block {}: {e}",
                            moe_block_idx + 1
                        ))
                    })?;
            }

            let lw = NucleusInferDit::untranspose_streamed_weights(&raw)?;
            let block = view.build_block_for_layer(layer_idx, Some(&lw))?;

            let (k, v) = (&kv_cache.entries[layer_idx].0, &kv_cache.entries[layer_idx].1);
            x = block.forward_with_cache(
                &x,
                None,
                &temb,
                &img_cos,
                &img_sin,
                Some(&txt_cos),
                Some(&txt_sin),
                Some((k, v)),
                None,
                cfg,
            )?;
        }

        // ---- norm_out (AdaLNContinuous, [scale, shift] chunk) + proj_out ----
        let norm_out_w = resident_ref
            .get("norm_out.linear.weight")
            .ok_or_else(|| Error::InvalidInput("missing norm_out.linear.weight".into()))?;
        let norm_out_b = resident_ref
            .get("norm_out.linear.bias")
            .ok_or_else(|| Error::InvalidInput("missing norm_out.linear.bias".into()))?;
        let proj_out_w = resident_ref
            .get("proj_out.weight")
            .ok_or_else(|| Error::InvalidInput("missing proj_out.weight".into()))?;

        let temb_silu = temb.silu()?;
        let temb_3d = temb_silu.unsqueeze(1)?;
        let mod_emb = fused_linear3d_native(&temb_3d, norm_out_w, Some(norm_out_b))?;
        let last = mod_emb.shape().dims().len() - 1;
        let chunks = mod_emb.chunk(2, last)?;
        let scale = &chunks[0];
        let shift = &chunks[1];

        let x_normed = layer_norm(&x, &[cfg.hidden_size], None, None, cfg.eps)?;
        let scale_p1 = scale.add_scalar(1.0)?;
        let x_scaled = x_normed.mul(&scale_p1)?;
        let x_modulated = x_scaled.add(shift)?;

        fused_linear3d_native(&x_modulated, proj_out_w, None)
    }

    /// Pre-transposed 2D weights coming back from BlockOffloader's
    /// prepare_weights need to be un-transposed for `fused_linear3d_native`
    /// (which expects PyTorch `[Cout, Cin]`). 3D MoE expert weights
    /// (`gate_up_proj` (E, D, 2*inter), `down_proj` (E, inter, D)) are not
    /// touched by the offloader's prepare step (it only handles 2D), so
    /// they pass through as-is.
    fn untranspose_streamed_weights(
        raw: &Arc<HashMap<String, Tensor>>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::with_capacity(raw.len());
        for (k, v) in raw.iter() {
            if k.ends_with(".weight") && v.shape().dims().len() == 2 {
                out.insert(k.clone(), v.transpose()?);
            } else {
                out.insert(k.clone(), v.clone());
            }
        }
        Ok(out)
    }
}

/// Tiny non-owning view used to call `build_block_for_layer` from inside
/// `forward_cached` while the offloader is borrowed `&mut`.
struct NucleusInferDitView<'a> {
    config: &'a NucleusConfig,
    resident: &'a HashMap<String, Tensor>,
}

impl<'a> NucleusInferDitView<'a> {
    fn r(&self, key: &str) -> Result<Tensor> {
        self.resident
            .get(key)
            .cloned()
            .ok_or_else(|| Error::InvalidInput(format!("NucleusInferDit view: missing {key}")))
    }

    fn build_block_for_layer(
        &self,
        layer_idx: usize,
        streamed: Option<&HashMap<String, Tensor>>,
    ) -> Result<NucleusBlock> {
        let prefix = format!("transformer_blocks.{layer_idx}.");
        let r = |suffix: &str| self.r(&format!("{prefix}{suffix}"));

        let mlp = if layer_idx < 3 {
            NucleusMlpWeights::Dense {
                gate_up_w: r("img_mlp.net.0.proj.weight")?,
                down_w: r("img_mlp.net.2.weight")?,
            }
        } else {
            let s = streamed.ok_or_else(|| {
                Error::InvalidInput("MoE layer requires streamed weights".into())
            })?;
            let take_streamed = |suffix: &str| -> Result<Tensor> {
                let key = format!("{prefix}{suffix}");
                s.get(&key).cloned().ok_or_else(|| {
                    Error::InvalidInput(format!("missing streamed {key}"))
                })
            };
            NucleusMlpWeights::Moe {
                gate_w: r("img_mlp.gate.weight")?,
                gate_up_w: take_streamed("img_mlp.experts.gate_up_proj")?,
                down_w: take_streamed("img_mlp.experts.down_proj")?,
                shared_gate_up_w: r("img_mlp.shared_expert.net.0.proj.weight")?,
                shared_down_w: r("img_mlp.shared_expert.net.2.weight")?,
                capacity_factor: self.config.capacity_factors[layer_idx],
            }
        };

        Ok(NucleusBlock {
            mod_w: r("img_mod.1.weight")?,
            mod_b: r("img_mod.1.bias")?,
            enc_proj_w: r("encoder_proj.weight")?,
            enc_proj_b: r("encoder_proj.bias")?,
            to_q_w: r("attn.to_q.weight")?,
            to_k_w: r("attn.to_k.weight")?,
            to_v_w: r("attn.to_v.weight")?,
            to_out_w: r("attn.to_out.0.weight")?,
            add_k_w: r("attn.add_k_proj.weight")?,
            add_v_w: r("attn.add_v_proj.weight")?,
            norm_q_w: r("attn.norm_q.weight")?,
            norm_k_w: r("attn.norm_k.weight")?,
            norm_added_k_w: r("attn.norm_added_k.weight")?,
            mlp,
        })
    }
}
