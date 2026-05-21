//! `HiDreamDecoderLayer` — one Qwen3-VL text decoder block.
//!
//! Reference (verbatim port targets):
//! - `qwen3_vl_transformers.py:496-539` — `Qwen3VLTextDecoderLayer`
//! - `qwen3_vl_transformers.py:405-477` — `Qwen3VLTextAttention`
//! - `qwen3_vl_transformers.py:480-493` — `Qwen3VLTextMLP` (SwiGLU)
//! - `qwen3_vl_transformers.py:358-374` — `Qwen3VLTextRMSNorm`
//! - `qwen3_vl_transformers.py:378-402` — `apply_rotary_pos_emb`
//!
//! Layout (per layer):
//! ```text
//!   x ──► RMSNorm (input_layernorm)
//!     ├─► self_attn:
//!     │     q_proj / k_proj / v_proj
//!     │     (per-head) q_norm  /  k_norm           ← `head_dim`-RMSNorm, NOT hidden-RMSNorm
//!     │     RoPE (half-split, MRoPE table)
//!     │     repeat_kv (GQA: 8 → 32)
//!     │     SDPA (structured prefix-causal/full policy)
//!     │     o_proj
//!     ├─► residual ●  +
//!     ├─► RMSNorm (post_attention_layernorm)
//!     │     SwiGLU(gate_proj, up_proj, down_proj)
//!     └─► residual ●  +
//! ```
//!
//! Differences vs `qwen3_encoder.rs`'s text-only Qwen3 layer:
//! - **MRoPE cos/sin** are passed in (already accounting for the
//!   stride-3 T/H/W interleave) rather than 1D-RoPE built from sequence
//!   positions inside the layer. The half-split kernel (`rope_halfsplit_bf16`)
//!   is the same — only the per-position table changes.
//! - Caller supplies a structured prefix-causal/full policy: causal triangle
//!   on text/AR rows, bidirectional attention on image rows
//!   (`qwen3_vl_transformers.py:1497-1504`).
//!
//! Why this re-implements rather than re-uses `Qwen3Encoder::layer_forward`:
//! `Qwen3Encoder` owns `weights: HashMap<String, Tensor>` and its
//! `layer_forward(layer_idx, ..)` indexes that map by string. The HiDream
//! pipeline holds individual `Linear` / `RMSNorm` modules per layer (so we
//! can support LoRA / per-layer overrides cleanly later). The math is
//! identical; the dispatch is by struct field instead of by string key.
//!
//! All weights are stored / computed in BF16 after loader-side conversion.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::attention::{sdpa as flame_sdpa, sdpa_prefix_causal_full};
use flame_core::nn::Linear;
use flame_core::norm::RMSNorm;
use flame_core::{bf16_ops, cuda_ops_bf16, CudaDevice, DType, Error, Result, Tensor};

use super::HiDreamO1Config;

/// One Qwen3-VL text decoder block, configured for HiDream-O1.
///
/// Field naming matches the Python parameter names (`q_proj`, `k_proj`,
/// `v_proj`, `o_proj`, `q_norm`, `k_norm`, `gate_proj`, `up_proj`,
/// `down_proj`, `input_layernorm`, `post_attention_layernorm`) so that a
/// safetensors loader can iterate `model.layers.{i}.self_attn.q_proj.weight`
/// → `layers[i].q_proj.weight` mechanically.
pub struct HiDreamDecoderLayer {
    /// `model.layers.{i}.input_layernorm` (`qwen3_vl_transformers.py:504`).
    pub input_layernorm: RMSNorm,
    /// `model.layers.{i}.self_attn.q_proj`, output features
    /// `num_attention_heads * head_dim` (= 4096 for 8B).
    pub q_proj: Linear,
    /// `model.layers.{i}.self_attn.k_proj`, output features
    /// `num_key_value_heads * head_dim` (= 1024 for 8B).
    pub k_proj: Linear,
    /// `model.layers.{i}.self_attn.v_proj`, same shape as `k_proj`.
    pub v_proj: Linear,
    /// `model.layers.{i}.self_attn.o_proj`, projects merged-head output
    /// back to `hidden_size`.
    pub o_proj: Linear,
    /// Per-head Q-RMSNorm — **applied to `head_dim` only** after view
    /// `[B, S, H, head_dim]` (`qwen3_vl_transformers.py:430-433, 448-449`).
    pub q_norm: RMSNorm,
    /// Per-head K-RMSNorm.
    pub k_norm: RMSNorm,
    /// `model.layers.{i}.post_attention_layernorm`
    /// (`qwen3_vl_transformers.py:505`).
    pub post_attention_layernorm: RMSNorm,
    /// `model.layers.{i}.mlp.gate_proj` (SwiGLU "gate") — F.silu side.
    pub gate_proj: Linear,
    /// `model.layers.{i}.mlp.up_proj` — multiplier side.
    pub up_proj: Linear,
    /// `model.layers.{i}.mlp.down_proj` — `intermediate_size → hidden_size`.
    pub down_proj: Linear,
    /// Saved for use during `forward` (head counts, head_dim, eps, etc.).
    pub config: HiDreamO1Config,
}

impl HiDreamDecoderLayer {
    /// Instantiate one decoder layer with random Xavier-init Linear weights
    /// and ones-init RMSNorm weights. Loader fills these in from the
    /// safetensors via `Linear::copy_weight_from` / `copy_bias_from` and
    /// `RMSNorm::copy_weight_from`.
    pub fn new(
        config: &HiDreamO1Config,
        _layer_idx: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let q_out = config.num_attention_heads * head_dim;
        let kv_out = config.num_kv_heads * head_dim;
        let inter = config.intermediate_size;
        let bias = config.attention_bias;
        let eps = config.rms_norm_eps;

        let input_layernorm = RMSNorm::new(vec![hidden], eps, true, device.clone())?;
        let q_proj = Linear::new(hidden, q_out, bias, device)?;
        let k_proj = Linear::new(hidden, kv_out, bias, device)?;
        let v_proj = Linear::new(hidden, kv_out, bias, device)?;
        let o_proj = Linear::new(q_out, hidden, bias, device)?;

        // Per-head RMSNorm: `normalized_shape = [head_dim]`, applied along
        // the last dim of `[B, S, H, head_dim]`. Python uses
        // `Qwen3VLTextRMSNorm(self.head_dim, eps=...)`
        // (`qwen3_vl_transformers.py:430-433`).
        let q_norm = RMSNorm::new(vec![head_dim], eps, true, device.clone())?;
        let k_norm = RMSNorm::new(vec![head_dim], eps, true, device.clone())?;

        let post_attention_layernorm = RMSNorm::new(vec![hidden], eps, true, device.clone())?;
        let gate_proj = Linear::new(hidden, inter, /*bias=*/ false, device)?;
        let up_proj = Linear::new(hidden, inter, /*bias=*/ false, device)?;
        let down_proj = Linear::new(inter, hidden, /*bias=*/ false, device)?;

        Ok(Self {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
            config: config.clone(),
        })
    }

    /// Run one transformer layer.
    ///
    /// `hidden_states`: `[B, S, hidden]` BF16.
    /// `cos_sin`: a precomputed `(cos, sin)` pair, each `[1, S, head_dim/2]`
    ///            BF16, produced by `mrope::interleaved_mrope_cos_sin`.
    /// `attention_mask`: legacy explicit mask override. Production O1 passes
    ///   `None` and uses `two_pass_ar_len` with structured prefix-causal/full
    ///   attention through Flame's exact mixed-mask primitive.
    ///
    /// Returns `[B, S, hidden]` BF16.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos_sin: &(Tensor, Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_attention_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;

        let dims = hidden_states.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "HiDreamDecoderLayer::forward: hidden_states must be [B,S,H], got {:?}",
                dims
            )));
        }
        let b = dims[0];
        let n = dims[1];

        // ─── 1) input_layernorm + self-attention ─────────────────────────
        let normed = self.input_layernorm.forward(hidden_states)?;

        let q = self.q_proj.forward(&normed)?; // [B, S, H*D]
        let k = self.k_proj.forward(&normed)?; // [B, S, Hkv*D]
        let v = self.v_proj.forward(&normed)?; // [B, S, Hkv*D]

        // Reshape to [B, H, S, D] / [B, Hkv, S, D].
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // After permute the storage may not be contiguous; flame-core's
        // norm/RoPE kernels need contiguous input. Materialize once.
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Per-head Q/K RMSNorm: normalize along `head_dim`. The fused BF16
        // kernel takes `[batch, hidden]` and normalizes the last dim, so
        // we flatten everything but `head_dim`. This matches Qwen3VL's
        // `q_norm(q.view(*input_shape, -1, head_dim))` which is a
        // `head_dim`-RMS over the per-head slice.
        let q_norm_w = self
            .q_norm
            .weight
            .as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("q_norm.weight missing".into()))?;
        let k_norm_w = self
            .k_norm
            .weight
            .as_ref()
            .ok_or_else(|| flame_core::Error::InvalidInput("k_norm.weight missing".into()))?;

        let q_flat = q.reshape(&[b * h * n, d])?;
        let q_normed = cuda_ops_bf16::rms_norm_bf16(&q_flat, Some(q_norm_w), cfg.rms_norm_eps)?;
        let q = q_normed.reshape(&[b, h, n, d])?;

        let k_flat = k.reshape(&[b * h_kv * n, d])?;
        let k_normed = cuda_ops_bf16::rms_norm_bf16(&k_flat, Some(k_norm_w), cfg.rms_norm_eps)?;
        let k = k_normed.reshape(&[b, h_kv, n, d])?;

        // RoPE — half-split, identical kernel to qwen3_encoder.rs. The
        // table itself is the **MRoPE** table (axis-aware for vision
        // positions); the rotate is the standard HF half-split rotate.
        // (`qwen3_vl_transformers.py:378-402` — `apply_rotary_pos_emb`.)
        let (pe_cos, pe_sin) = cos_sin;
        let q = bf16_ops::rope_halfsplit_bf16_pytorch(&q, pe_cos, pe_sin)?;
        let k = bf16_ops::rope_halfsplit_bf16_pytorch(&k, pe_cos, pe_sin)?;

        // GQA: replicate KV heads to match Q head count.
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // SDPA. Mask is multiplicative (1=keep, 0=block) per
        // `flame_core::sdpa::forward_f32` semantics; caller builds it.
        let attn_out = flame_sdpa(&q, &k, &v, attention_mask)?;

        // [B, H, S, D] → [B, S, H*D]
        let attn_out = attn_out
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, n, h * d])?;
        let attn_out = self.o_proj.forward(&attn_out)?;

        let hidden_states = hidden_states.add(&attn_out)?;

        // ─── 2) post_attention_layernorm + SwiGLU MLP ────────────────────
        let normed2 = self.post_attention_layernorm.forward(&hidden_states)?;

        let gate = self.gate_proj.forward(&normed2)?;
        let up = self.up_proj.forward(&normed2)?;
        // SwiGLU = silu(gate) * up. Use Tensor::swiglu (fused BF16 kernel that
        // records Op::FusedSwiGLU for autograd) instead of the inference-only
        // `bf16_ops::silu_bf16` which returns a detached tensor — that path
        // killed the `mlp.gate_proj` LoRA gradient in the trainer (G2 bug).
        let mlp_inner = gate.swiglu(&up)?;
        let mlp_out = self.down_proj.forward(&mlp_inner)?;

        hidden_states.add(&mlp_out)
    }
}

/// Stateless decoder forward driven by a per-layer weight HashMap.
///
/// Used by the BlockOffloader path in `HiDreamO1Model::forward`: each step
/// prefetches the next block's weights into pinned-host→GPU pinned buffers,
/// awaits the current block, and calls this function with the resulting
/// `&HashMap<String, Tensor>` (already un-transposed to PyTorch `[Cout, Cin]`).
///
/// Math is identical to `HiDreamDecoderLayer::forward`. Inputs:
/// - `cfg`: `HiDreamO1Config` (head/dim counts, eps, attention_bias).
/// - `layer_idx`: index into `model.language_model.layers.{i}` (used both
///   to build the lookup keys and for error messages).
/// - `weights`: keyed by the **full** safetensors key, e.g.
///   `"model.language_model.layers.{i}.self_attn.q_proj.weight"`. This is
///   what `BlockOffloader::await_block(i)` returns after the loader's
///   facilitator routed by layer index.
pub fn decoder_forward_with_weights(
    cfg: &HiDreamO1Config,
    layer_idx: usize,
    hidden_states: &Tensor,
    cos_sin: &(Tensor, Tensor),
    attention_mask: Option<&Tensor>,
    weights: &HashMap<String, Tensor>,
) -> Result<Tensor> {
    decoder_forward_with_weights_lora(
        cfg,
        layer_idx,
        hidden_states,
        cos_sin,
        attention_mask,
        weights,
        None,
        None,
    )
}

/// LoRA-aware variant of [`decoder_forward_with_weights`].
///
/// When `lora` is `Some(&registry)`, each of the 7 fused-linear call sites
/// in the layer looks up `(layer_idx, suffix)` in the registry and dispatches
/// to `fused_linear3d_native_lora`. When the lookup misses (or `lora` is
/// `None`), it falls through to the original `fused_linear3d_native` call,
/// preserving byte-identical inference behavior.
///
/// See `super::lora::LoraRegistry` for the registry contract and target
/// suffix naming.
#[allow(clippy::too_many_arguments)]
pub fn decoder_forward_with_weights_lora(
    cfg: &HiDreamO1Config,
    layer_idx: usize,
    hidden_states: &Tensor,
    cos_sin: &(Tensor, Tensor),
    attention_mask: Option<&Tensor>,
    weights: &HashMap<String, Tensor>,
    lora: Option<&super::lora::LoraRegistry>,
    two_pass_ar_len: Option<usize>,
) -> Result<Tensor> {
    // Instrumentation: HIDREAM_MEM_LOG=1 logs free MiB at each pre-attention
    // stage of layer 0 only (the OOM hits inside the first layer at 2048²).
    let mem_log = layer_idx == 0
        && std::env::var("HIDREAM_MEM_LOG").ok().as_deref() == Some("1");
    let log_mem = |label: &str| {
        if mem_log {
            let free = flame_core::cuda::utils::cuda_mem_get_free_mb()
                .map(|m| format!("{} MiB free", m))
                .unwrap_or_else(|| "??? MiB free".to_string());
            eprintln!("[hidream_mem]   layer0.{:24} {}", label, free);
        }
    };
    log_mem("entry");

    let p = format!("model.language_model.layers.{layer_idx}");
    let h = cfg.num_attention_heads;
    let h_kv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let n_rep = h / h_kv;

    let dims = hidden_states.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "decoder_forward_with_weights[{layer_idx}]: hidden_states must be [B,S,H], got {:?}",
            dims
        )));
    }
    let b = dims[0];
    let n = dims[1];

    let wget = |suffix: &str| -> Result<&Tensor> {
        let k = format!("{p}.{suffix}");
        weights.get(&k).ok_or_else(|| {
            Error::InvalidInput(format!(
                "decoder_forward_with_weights[layer={layer_idx}]: missing weight {k}"
            ))
        })
    };

    // ─── 1) input_layernorm ──────────────────────────────────────────
    let in_ln_w = wget("input_layernorm.weight")?;
    let normed = rms_norm_apply(hidden_states, in_ln_w, cfg.rms_norm_eps)?;
    log_mem("after_input_rms");

    // ─── 2) self-attention QKV projections ───────────────────────────
    let q_w = wget("self_attn.q_proj.weight")?;
    let k_w = wget("self_attn.k_proj.weight")?;
    let v_w = wget("self_attn.v_proj.weight")?;
    // Phase 2c does not enable attention_bias; HiDream-O1-Dev has it false.
    // If a future variant flips it, the bias keys are auto-streamed by the
    // BlockOffloader (their key matches the `layers.{i}.` prefix) and pulled
    // from `weights` here.
    let (q_b, k_b, v_b): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>) =
        if cfg.attention_bias {
            (
                Some(wget("self_attn.q_proj.bias")?),
                Some(wget("self_attn.k_proj.bias")?),
                Some(wget("self_attn.v_proj.bias")?),
            )
        } else {
            (None, None, None)
        };

    // Dispatch helper: when a matching LoRA adapter exists for this layer +
    // suffix, route through `fused_linear3d_native_lora`; otherwise fall back
    // to the plain `fused_linear3d_native` (byte-identical no-LoRA path).
    let lora_linear = |x: &Tensor,
                       w: &Tensor,
                       b: Option<&Tensor>,
                       suffix: &str|
     -> Result<Tensor> {
        match lora.and_then(|r| r.get(layer_idx, suffix)) {
            Some(adapter) => {
                let a_t = adapter.a_tensor()?;
                let b_t = adapter.b_tensor()?;
                flame_core::ops::fused_inference::fused_linear3d_native_lora(
                    x,
                    w,
                    b,
                    Some(&a_t),
                    Some(&b_t),
                    adapter.scale,
                )
            }
            None => flame_core::ops::fused_inference::fused_linear3d_native(x, w, b),
        }
    };

    let q = lora_linear(&normed, q_w, q_b, "self_attn.q_proj")?;
    log_mem("after_q_proj");
    let k = lora_linear(&normed, k_w, k_b, "self_attn.k_proj")?;
    log_mem("after_k_proj");
    let v = lora_linear(&normed, v_w, v_b, "self_attn.v_proj")?;
    log_mem("after_v_proj");
    // Soul.md trap (layer 0, V-path): record v_proj's output ID. With
    // gradient checkpointing, the *first* forward runs no-autograd and the
    // ID we capture there isn't on any tape. The ID we want is from the
    // *recompute* forward (autograd enabled). Re-record on every call so
    // last-writer-wins gives us the recompute ID, and on recompute also push
    // it into the additive retain set so the sub-tape backward retains its
    // grad. (Outer-tape retain snapshot has already fired by this point.)
    // Soul.md trap: probe the LAST decoder layer. Layer 35's o_proj LoRA-B
    // grad is cos≈0.999 (clean upstream signal) while q/k/v_proj LoRA-B
    // grads are cos≈0.05 (corrupt downstream output) — so at layer 35 we
    // see the bug fire locally without 35 layers of cascade noise.
    if layer_idx == cfg.num_layers - 1 && super::trap::is_armed() {
        super::trap::record_probe("v_proj_out", v.id());
        if flame_core::autograd::AutogradContext::is_checkpoint_recompute() {
            let mut s = std::collections::HashSet::new();
            s.insert(v.id());
            flame_core::autograd::AutogradContext::retain_intermediate_grads_add(s);
        }
    }

    // Reshape to [B, H, S, D] / [B, Hkv, S, D].
    let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
    let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
    let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    log_mem("after_qkv_contig");

    // Per-head Q/K RMSNorm over `head_dim`.
    let q_norm_w = wget("self_attn.q_norm.weight")?;
    let k_norm_w = wget("self_attn.k_norm.weight")?;
    let q_flat = q.reshape(&[b * h * n, d])?;
    let q_normed = rms_norm_apply(&q_flat, q_norm_w, cfg.rms_norm_eps)?;
    let q = q_normed.reshape(&[b, h, n, d])?;
    let k_flat = k.reshape(&[b * h_kv * n, d])?;
    let k_normed = rms_norm_apply(&k_flat, k_norm_w, cfg.rms_norm_eps)?;
    let k = k_normed.reshape(&[b, h_kv, n, d])?;
    log_mem("after_qk_norm");

    let (pe_cos, pe_sin) = cos_sin;
    let q = bf16_ops::rope_halfsplit_bf16_pytorch(&q, pe_cos, pe_sin)?;
    let k = bf16_ops::rope_halfsplit_bf16_pytorch(&k, pe_cos, pe_sin)?;
    log_mem("after_rope");

    let k = repeat_kv(&k, n_rep)?;
    let v = repeat_kv(&v, n_rep)?;
    log_mem("after_repeat_kv");
    // Soul.md trap (layer 0): probe V after repeat_kv = SDPA's V input.
    // Splits the search between SDPA bwd (above) and repeat_kv+reshape bwd
    // (below) for the V LoRA-B grad corruption.
    // Soul.md trap: probe the LAST decoder layer. Layer 35's o_proj LoRA-B
    // grad is cos≈0.999 (clean upstream signal) while q/k/v_proj LoRA-B
    // grads are cos≈0.05 (corrupt downstream output) — so at layer 35 we
    // see the bug fire locally without 35 layers of cascade noise.
    if layer_idx == cfg.num_layers - 1 && super::trap::is_armed() {
        super::trap::record_probe("v_post_repeat_kv", v.id());
        if flame_core::autograd::AutogradContext::is_checkpoint_recompute() {
            let mut s = std::collections::HashSet::new();
            s.insert(v.id());
            flame_core::autograd::AutogradContext::retain_intermediate_grads_add(s);
        }
    }

    // edv2-reference's HiDream-O1 `use_flash_attn=True` path avoids a 4D mixed
    // mask. Flame owns that structured prefix-causal/full policy in
    // `sdpa_prefix_causal_full`; the training path records it as one custom
    // backward op instead of two independent SDPA nodes over shared K/V.
    let attn_out = match two_pass_ar_len {
        Some(ar_len) if attention_mask.is_none() => {
            hidream_o1_two_pass_attention(&q, &k, &v, ar_len)?
        }
        _ => chunked_sdpa(&q, &k, &v, attention_mask)?,
    };
    log_mem("after_sdpa");
    // Soul.md trap (layer 0): record SDPA-output ID. Same checkpoint dance
    // as v_proj_out above — register into the additive retain set during
    // recompute so the sub-tape backward keeps its grad.
    // Soul.md trap: probe the LAST decoder layer. Layer 35's o_proj LoRA-B
    // grad is cos≈0.999 (clean upstream signal) while q/k/v_proj LoRA-B
    // grads are cos≈0.05 (corrupt downstream output) — so at layer 35 we
    // see the bug fire locally without 35 layers of cascade noise.
    if layer_idx == cfg.num_layers - 1 && super::trap::is_armed() {
        super::trap::record_probe("attn_out", attn_out.id());
        if flame_core::autograd::AutogradContext::is_checkpoint_recompute() {
            let mut s = std::collections::HashSet::new();
            s.insert(attn_out.id());
            flame_core::autograd::AutogradContext::retain_intermediate_grads_add(s);
        }
    }
    let attn_out = attn_out
        .permute(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[b, n, h * d])?;

    let o_w = wget("self_attn.o_proj.weight")?;
    let o_b = if cfg.attention_bias {
        Some(wget("self_attn.o_proj.bias")?)
    } else {
        None
    };
    let attn_out = lora_linear(&attn_out, o_w, o_b, "self_attn.o_proj")?;

    let hidden_states = hidden_states.add(&attn_out)?;

    // ─── 3) post_attention_layernorm + SwiGLU MLP ────────────────────
    let post_ln_w = wget("post_attention_layernorm.weight")?;
    let normed2 = rms_norm_apply(&hidden_states, post_ln_w, cfg.rms_norm_eps)?;

    let gate_w = wget("mlp.gate_proj.weight")?;
    let up_w = wget("mlp.up_proj.weight")?;
    let down_w = wget("mlp.down_proj.weight")?;

    let gate = lora_linear(&normed2, gate_w, None, "mlp.gate_proj")?;
    let up = lora_linear(&normed2, up_w, None, "mlp.up_proj")?;
    // SwiGLU = silu(gate) * up. Use Tensor::swiglu (autograd-registered fused
    // BF16 kernel) so the `mlp.gate_proj` LoRA branch sees gradient. The old
    // `bf16_ops::silu_bf16` is an inference-only primitive (returns
    // requires_grad=false, records no op) — using it here detached gate's
    // autograd path and produced 72 dead LoRA params in G2.
    let mlp_inner = gate.swiglu(&up)?;
    let mlp_out = lora_linear(&mlp_inner, down_w, None, "mlp.down_proj")?;

    hidden_states.add(&mlp_out)
}

/// Diagnostic mirror of `decoder_forward_with_weights_lora` for one decoder
/// layer. Used only by parity dumps; production forward does not call this
/// unless `HIDREAM_DUMP_LAYERS` is set.
pub fn decoder_forward_probe_with_weights_lora(
    cfg: &HiDreamO1Config,
    layer_idx: usize,
    hidden_states: &Tensor,
    cos_sin: &(Tensor, Tensor),
    attention_mask: Option<&Tensor>,
    weights: &HashMap<String, Tensor>,
    lora: Option<&super::lora::LoraRegistry>,
    two_pass_ar_len: Option<usize>,
) -> Result<HashMap<String, Tensor>> {
    let p = format!("model.language_model.layers.{layer_idx}");
    let h = cfg.num_attention_heads;
    let h_kv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let n_rep = h / h_kv;

    let dims = hidden_states.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "decoder_forward_probe_with_weights_lora[layer={layer_idx}]: hidden_states must be [B,S,H], got {:?}",
            dims
        )));
    }
    let b = dims[0];
    let n = dims[1];

    let wget = |suffix: &str| -> Result<&Tensor> {
        let k = format!("{p}.{suffix}");
        weights.get(&k).ok_or_else(|| {
            Error::InvalidInput(format!(
                "decoder_forward_probe_with_weights_lora[layer={layer_idx}]: missing weight {k}"
            ))
        })
    };
    let lora_linear = |x: &Tensor,
                       w: &Tensor,
                       b: Option<&Tensor>,
                       suffix: &str|
     -> Result<Tensor> {
        match lora.and_then(|r| r.get(layer_idx, suffix)) {
            Some(adapter) => {
                let a_t = adapter.a_tensor()?;
                let b_t = adapter.b_tensor()?;
                flame_core::ops::fused_inference::fused_linear3d_native_lora(
                    x,
                    w,
                    b,
                    Some(&a_t),
                    Some(&b_t),
                    adapter.scale,
                )
            }
            None => flame_core::ops::fused_inference::fused_linear3d_native(x, w, b),
        }
    };

    let mut out = HashMap::new();
    let mut put = |key: &str, tensor: &Tensor| -> Result<()> {
        out.insert(format!("layer{layer_idx:02}.{key}"), tensor.to_dtype(DType::F32)?);
        Ok(())
    };

    let normed = rms_norm_apply(hidden_states, wget("input_layernorm.weight")?, cfg.rms_norm_eps)?;
    put("normed", &normed)?;

    let q_w = wget("self_attn.q_proj.weight")?;
    let k_w = wget("self_attn.k_proj.weight")?;
    let v_w = wget("self_attn.v_proj.weight")?;
    let (q_b, k_b, v_b): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>) =
        if cfg.attention_bias {
            (
                Some(wget("self_attn.q_proj.bias")?),
                Some(wget("self_attn.k_proj.bias")?),
                Some(wget("self_attn.v_proj.bias")?),
            )
        } else {
            (None, None, None)
        };

    let q_proj = lora_linear(&normed, q_w, q_b, "self_attn.q_proj")?;
    let k_proj = lora_linear(&normed, k_w, k_b, "self_attn.k_proj")?;
    let v_proj = lora_linear(&normed, v_w, v_b, "self_attn.v_proj")?;
    put("q_proj", &q_proj)?;
    put("k_proj", &k_proj)?;
    put("v_proj", &v_proj)?;

    let q_heads = q_proj.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let k_heads = k_proj.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;
    let v_heads = v_proj.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;
    put("q_heads", &q_heads)?;
    put("k_heads", &k_heads)?;
    put("v_heads", &v_heads)?;

    let q_norm_w = wget("self_attn.q_norm.weight")?;
    let k_norm_w = wget("self_attn.k_norm.weight")?;
    let q_flat = q_heads.reshape(&[b * h * n, d])?;
    let q_mean_sq = cuda_ops_bf16::rms_norm_bf16_mean_sq_head128(&q_flat)?
        .reshape(&[b, h, n])?;
    let q_inv = cuda_ops_bf16::rms_norm_bf16_inv_rms(&q_flat, cfg.rms_norm_eps)?
        .reshape(&[b, h, n])?;
    let q_unit_flat = cuda_ops_bf16::rms_norm_bf16(&q_flat, None, cfg.rms_norm_eps)?;
    let q_unit = q_unit_flat.reshape(&[b, h, n, d])?;
    let q_normed = q_unit_flat.mul(q_norm_w)?.reshape(&[b, h, n, d])?;
    let k_flat = k_heads.reshape(&[b * h_kv * n, d])?;
    let k_mean_sq = cuda_ops_bf16::rms_norm_bf16_mean_sq_head128(&k_flat)?
        .reshape(&[b, h_kv, n])?;
    let k_inv = cuda_ops_bf16::rms_norm_bf16_inv_rms(&k_flat, cfg.rms_norm_eps)?
        .reshape(&[b, h_kv, n])?;
    let k_unit_flat = cuda_ops_bf16::rms_norm_bf16(&k_flat, None, cfg.rms_norm_eps)?;
    let k_unit = k_unit_flat.reshape(&[b, h_kv, n, d])?;
    let k_normed = k_unit_flat.mul(k_norm_w)?.reshape(&[b, h_kv, n, d])?;
    put("q_mean_sq", &q_mean_sq)?;
    put("k_mean_sq", &k_mean_sq)?;
    put("q_inv", &q_inv)?;
    put("k_inv", &k_inv)?;
    put("q_unit", &q_unit)?;
    put("k_unit", &k_unit)?;
    put("q_normed", &q_normed)?;
    put("k_normed", &k_normed)?;

    let (pe_cos, pe_sin) = cos_sin;
    put("cos_half", pe_cos)?;
    put("sin_half", pe_sin)?;
    let q_rope = bf16_ops::rope_halfsplit_bf16_pytorch(&q_normed, pe_cos, pe_sin)?;
    let k_rope = bf16_ops::rope_halfsplit_bf16_pytorch(&k_normed, pe_cos, pe_sin)?;
    put("q_rope", &q_rope)?;
    put("k_rope", &k_rope)?;

    let k_repeat = repeat_kv(&k_rope, n_rep)?;
    let v_repeat = repeat_kv(&v_heads, n_rep)?;
    put("k_repeat", &k_repeat)?;
    put("v_repeat", &v_repeat)?;

    let sdpa_out = match two_pass_ar_len {
        Some(ar_len) if attention_mask.is_none() => {
            hidream_o1_two_pass_attention(&q_rope, &k_repeat, &v_repeat, ar_len)?
        }
        _ => chunked_sdpa(&q_rope, &k_repeat, &v_repeat, attention_mask)?,
    };
    put("sdpa_out", &sdpa_out)?;

    let o_proj_in = sdpa_out
        .permute(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[b, n, h * d])?;
    put("o_proj_in", &o_proj_in)?;

    let o_w = wget("self_attn.o_proj.weight")?;
    let o_b = if cfg.attention_bias {
        Some(wget("self_attn.o_proj.bias")?)
    } else {
        None
    };
    let attn_out = lora_linear(&o_proj_in, o_w, o_b, "self_attn.o_proj")?;
    put("attn_out", &attn_out)?;
    let after_attn = hidden_states.add(&attn_out)?;
    put("after_attn", &after_attn)?;

    let normed2 = rms_norm_apply(
        &after_attn,
        wget("post_attention_layernorm.weight")?,
        cfg.rms_norm_eps,
    )?;
    put("normed2", &normed2)?;

    let gate = lora_linear(&normed2, wget("mlp.gate_proj.weight")?, None, "mlp.gate_proj")?;
    let up = lora_linear(&normed2, wget("mlp.up_proj.weight")?, None, "mlp.up_proj")?;
    put("gate", &gate)?;
    put("up", &up)?;
    let mlp_inner = gate.swiglu(&up)?;
    put("mlp_inner", &mlp_inner)?;
    let mlp_out = lora_linear(&mlp_inner, wget("mlp.down_proj.weight")?, None, "mlp.down_proj")?;
    put("mlp_out", &mlp_out)?;
    let hidden_out = after_attn.add(&mlp_out)?;
    put("hidden_out", &hidden_out)?;

    Ok(out)
}

/// Apply RMSNorm with weight: reshape to [batch, hidden], norm, reshape back.
///
/// `rms_norm_bf16` expects a 2D `[batch, hidden]` input; this helper preserves
/// the rank-N shape of the caller while doing the kernel call in 2D.
fn rms_norm_apply(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = cuda_ops_bf16::rms_norm_bf16(&x_2d, None, eps)?.mul(weight)?;
    out.reshape(&dims)
}

/// Repeat KV heads to match Q head count for GQA.
///
/// Input: `[B, H_kv, S, D]`. Output: `[B, H_kv * n_rep, S, D]`.
/// Mirrors `qwen3_vl_transformers.py:145-156` (`repeat_kv`) and the same
/// helper in `qwen3_encoder.rs`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "repeat_kv expected [B,Hkv,S,D], got {:?}",
            dims
        )));
    }
    let b = dims[0];
    let h_kv = dims[1];
    let s = dims[2];
    let d = dims[3];

    // PyTorch's repeat_kv:
    //   x[:, :, None, :, :].expand(B, Hkv, n_rep, S, D).reshape(B, Hkv*n_rep, S, D)
    // We reproduce via stack along axis=2 (creates the "n_rep" dim).
    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    let stacked = Tensor::stack(&copies, 2)?; // [B, Hkv, n_rep, S, D]
    stacked.reshape(&[b, h_kv * n_rep, s, d])
}

/// SDPA over Q rows.
///
/// Legacy explicit-mask SDPA over Q rows. Production HiDream-O1 uses
/// `sdpa_prefix_causal_full`; this exists only for callers that pass an
/// explicit mask override.
///
/// For masked attention, route directly to flame-core's streaming BF16 SDPA so
/// scores are tiled through a small workspace instead of materialized. Keep the
/// old chunked generic path only as an unsupported-backend fallback.
///
/// `HIDREAM_SDPA_CHUNK` controls the fallback workspace row chunk (default
/// 1024). The optimized streaming launcher uses its own tile env vars
/// (`STREAMING_SDPA_CHUNK_MAX`, `FLAME_SDPA_MAX_Q_TILE`, etc.).
fn chunked_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let q_dims = q.shape().dims();
    if q_dims.len() != 4 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "chunked_sdpa: expected Q to be [B,H,Sq,D], got {:?}",
            q_dims
        )));
    }
    let sq = q_dims[2];

    let chunk_size: usize = std::env::var("HIDREAM_SDPA_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let needs_autograd = flame_core::autograd::AutogradContext::is_recording()
        && (q.requires_grad() || k.requires_grad() || v.requires_grad());

    let disable_direct_stream = std::env::var("HIDREAM_O1_DISABLE_STREAM_SDPA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if let Some(keep_mask) = mask.filter(|_| !needs_autograd && !disable_direct_stream) {
        // Public flame SDPA masks use HiDream's multiplicative keep-mask
        // convention (1 = attend, 0 = block). flame-core handles any backend
        // conversion internally, so inference can use the streaming path while
        // keeping the generic SDPA fallback numerically aligned.
        match cuda_ops_bf16::sdpa_stream_bf16(
            q,
            k,
            v,
            Some(keep_mask),
            chunk_size.max(1),
            false,
            None,
        ) {
            Ok(out) => return Ok(out),
            Err(Error::Unsupported(reason)) => {
                log::warn!(
                    "hidream_o1: streaming masked SDPA unsupported ({}); falling back to generic chunked SDPA",
                    reason
                );
            }
            Err(err) => return Err(err),
        }
    }

    if sq <= chunk_size {
        return flame_sdpa(q, k, v, mask);
    }

    let mut out_chunks: Vec<Tensor> = Vec::with_capacity((sq + chunk_size - 1) / chunk_size);
    let mut start = 0usize;
    while start < sq {
        let len = std::cmp::min(chunk_size, sq - start);
        let q_chunk = q.narrow(2, start, len)?;
        let mask_chunk_owned;
        let mask_chunk = match mask {
            Some(m) => {
                let m_dims = m.shape().dims();
                if m_dims.len() == 4 && m_dims[2] == sq {
                    mask_chunk_owned = m.narrow(2, start, len)?;
                    Some(&mask_chunk_owned)
                } else {
                    // Broadcast-shaped mask (e.g. [B,1,1,Sk]) — pass through.
                    Some(m)
                }
            }
            None => None,
        };
        let chunk_out = flame_sdpa(&q_chunk, k, v, mask_chunk)?;
        out_chunks.push(chunk_out);
        start += len;
    }

    let chunk_refs: Vec<&Tensor> = out_chunks.iter().collect();
    Tensor::cat(&chunk_refs, 2)
}

/// HiDream-O1 two-pass attention dispatcher.
///
/// Ported from Kijai's ComfyUI `hidream_o1` branch
/// (`comfy/ldm/hidream_o1/attention.py::make_two_pass_attention`,
/// commit `974aab796d`, also present at
/// `https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/hidream_o1/attention.py`).
///
/// Tokens `[0, ar_len)` are the AR / text prefix (causal); tokens
/// `[ar_len, T)` are the image-generation tail (full bidirectional over the
/// whole sequence). Splitting Q at the boundary lets us avoid the
/// `(B, 1, T, T)` additive mask the materialized-mask path would build
/// (~500 MiB at `T ≈ 16384`). Flame's public mixed primitive owns the
/// backend choice and, for training, records one exact mixed-attention
/// backward op.
///
/// Four dispatch cases (matches Kijai's Python verbatim):
///
/// 1. **KV-cache hot path** — `q_len < kv_len`: Q is shorter than K/V, so all
///    fresh Q positions live in the gen region; single non-causal SDPA call.
///    (Currently unreachable from `inference-flame`'s HiDream-O1 pipeline,
///    which recomputes per step, but the surface is ported.)
/// 2. **All-causal** — `ar_len >= q_len`: every Q row is in the AR prefix;
///    single `flame_sdpa_causal` call.
/// 3. **All-gen** — `ar_len == 0`: every Q row is in the gen region; single
///    non-causal SDPA call.
/// 4. **Mixed** — `0 < ar_len < q_len`: AR-causal head + full-attention tail,
///    concatenated along sequence. Delegates to
///    `flame_core::attention::sdpa_prefix_causal_full`, which owns the exact
///    mixed-mask policy in Flame. In training, Flame records this as one
///    custom backward op so shared K/V gradients match the single-mask oracle;
///    inference uses the same public API surface.
///
/// `q`: `[B, H, T_q, D]`. `k`/`v`: `[B, H, T_kv, D]` (already GQA-replicated).
/// `ar_len`: number of AR / text prefix rows in the **Q** sequence.
///
/// Returns `[B, H, T_q, D]`.
pub fn hidream_o1_two_pass_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ar_len: usize,
) -> Result<Tensor> {
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "hidream_o1_two_pass_attention: expected [B,H,T,D] tensors, got q={:?} k={:?}",
            q_dims, k_dims
        )));
    }
    let t_q = q_dims[2];
    let t_kv = k_dims[2];

    // Case 1: KV-cache hot path. Q is shorter than K/V — every Q row is in
    // the gen region (the cached AR prefix is in K/V only). Single full call.
    if t_q < t_kv {
        return flame_sdpa(q, k, v, None);
    }

    // From here on this is self-attention (t_q == t_kv). The other branches
    // require this for `sdpa_prefix_causal_full`.
    if ar_len >= t_q {
        // Case 2: all-causal. Every Q row is in the AR prefix.
        return flame_core::attention::sdpa_causal(q, k, v);
    }
    if ar_len == 0 {
        // Case 3: all-gen. Every Q row is in the gen region.
        return flame_sdpa(q, k, v, None);
    }
    // Case 4: mixed. Structured AR-causal + gen-full.
    sdpa_prefix_causal_full(q, k, v, ar_len)
}
