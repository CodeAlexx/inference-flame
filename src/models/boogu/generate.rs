//! C9 — Boogu-Image prompt **rewriter**: Qwen3-VL-8B autoregressive GENERATE.
//!
//! The Boogu pipeline ships a Python dev-tool
//! (`utils/t2i_external_prompt_rewriter.py`) that expands a short / rough user
//! idea into a fuller T2I instruction before it feeds the normal pipeline. It
//! reuses the **same** `mllm` = Qwen3-VL-8B language tower the C7
//! [`BooguTextEncoder`](crate::models::boogu::encoder::BooguTextEncoder) loads —
//! but in `.generate` mode (autoregressive decode), with Boogu's own English
//! rewrite system prompt.
//!
//! This module is the pure-Rust replacement: a KV-cached, incremental Qwen3-VL
//! decode loop + a temperature/greedy sampler, mirroring the verified
//! [`gemma4`](crate::models::gemma4) generate path. No Python at runtime.
//!
//! ## What is reused (do NOT reimplement)
//!
//! - **Qwen3 layer math** — the per-layer forward (RMSNorm → q/k/v proj →
//!   per-head QK-RMSNorm → half-split RoPE → GQA SDPA → o_proj → SwiGLU MLP) is
//!   the *same* math as
//!   [`Qwen3Encoder::layer_forward`](crate::models::qwen3_encoder); the only
//!   change is incremental (q_len=1) attention against a growing per-layer
//!   KV-cache instead of a single full-sequence forward.
//! - **Config** — [`Qwen3Config::qwen3_vl_text`] (4096 / 36 layers / kv 8 /
//!   head_dim 128 / θ 5e6 / eps 1e-6).
//! - **mllm loader remap** — the `model.language_model.*` → `model.*` strip from
//!   `encoder.rs`, *plus* the two keys the encoder dropped: `lm_head.weight`
//!   (generate NEEDS it — `tie_word_embeddings=false` in the mllm config) and
//!   `model.norm.weight` (the final RMSNorm, which the encoder's `encode()` does
//!   not apply because it taps `hidden_states[-1]` pre-final-norm).
//! - **Sampler pattern** — `gemma4::TemperatureSampler` (greedy argmax when
//!   `temperature < eps`, else multinomial over a host-side cumsum).
//! - **KV-cache cat-grow** — `gemma4::kv_cache` grows K/V via `Tensor::cat`
//!   along the seq dim each step (Qwen3 has NO sliding window, so a single
//!   "full" variant suffices).
//!
//! ## Weights stored in NATIVE `[Cout, Cin]` layout
//!
//! Unlike `Qwen3Encoder` (which pre-transposes every 2-D weight to feed its
//! `matmul`-based linear), this engine keeps weights in the safetensors-native
//! PyTorch `[out, in]` layout and calls
//! [`fused_linear3d_native`](flame_core::ops::fused_inference::fused_linear3d_native)
//! (cuBLASLt TRANSA=T inside the GEMM). Same convention `gemma4` uses.
//!
//! Inference port — autograd is OFF (the bin wraps `generate` in
//! `AutogradContext::no_grad()`). BF16 throughout (matches the checkpoint).

use crate::models::qwen3_encoder::{expected_weight_keys, Qwen3Config};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Number of decoder layers in the Boogu mllm language tower.
pub const BOOGU_MLLM_NUM_LAYERS: usize = 36;

/// Qwen pad / bos id (`pad_token == bos == 151643`). Also a terminal token in
/// the mllm `generation_config.json` eos list.
pub const QWEN_PAD_ID: i32 = 151_643;

/// `<|im_end|>` (151645) — the assistant turn terminator and primary eos.
pub const QWEN_IM_END_ID: u32 = 151_645;

/// End-of-sequence ids that stop decode. From the mllm
/// `generation_config.json`: `eos_token_id = [151645, 151643]`. Either halts
/// the loop (matches HF `.generate`).
pub const QWEN_EOS_IDS: [u32; 2] = [QWEN_IM_END_ID, 151_643];

// ---------------------------------------------------------------------------
// Per-layer KV cache (cat-grow, full-attention only — Qwen3 has no SWA)
// ---------------------------------------------------------------------------

/// One decoder layer's K/V cache. Grows along the seq dim via `Tensor::cat` on
/// each `append` — the same strategy `gemma4::kv_cache` and `sensenova_u1` use
/// (flame-core has no scatter-into-slice primitive; `cat` is the public grow
/// surface). Qwen3 is fully causal with NO sliding window, so we never trim.
struct LayerKvCache {
    /// `[1, num_kv_heads, valid_len, head_dim]` BF16 (post-RoPE K).
    k: Tensor,
    /// `[1, num_kv_heads, valid_len, head_dim]` BF16 (V, un-RoPE'd).
    v: Tensor,
    valid_len: usize,
}

impl LayerKvCache {
    fn new(num_kv_heads: usize, head_dim: usize, device: &Arc<CudaDevice>) -> Result<Self> {
        // Zero-length on the seq dim; first `append` materializes storage.
        let init = Shape::from_dims(&[1, num_kv_heads, 0, head_dim]);
        let k = Tensor::zeros_dtype(init.clone(), DType::BF16, device.clone())?;
        let v = Tensor::zeros_dtype(init, DType::BF16, device.clone())?;
        Ok(Self { k, v, valid_len: 0 })
    }

    /// Append this step's K/V (`[1, h_kv, S_new, d]`) and return the full
    /// cached `(K, V)` for attention. On prefill `S_new = prompt_len`; on
    /// decode `S_new = 1`.
    fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let s_new = new_k.shape().dims()[2];
        if self.valid_len == 0 {
            self.k = new_k.clone();
            self.v = new_v.clone();
        } else {
            self.k = Tensor::cat(&[&self.k, new_k], 2)?;
            self.v = Tensor::cat(&[&self.v, new_v], 2)?;
        }
        self.valid_len += s_new;
        Ok((self.k.clone(), self.v.clone()))
    }
}

// ---------------------------------------------------------------------------
// The generate engine
// ---------------------------------------------------------------------------

/// Qwen3-VL-8B autoregressive generate engine (KV-cache + sampler).
///
/// Holds the mllm language-tower weights in native `[Cout, Cin]` BF16, plus
/// `lm_head.weight` and the final `model.norm.weight`. Reuses the Qwen3 layer
/// math (mirrors `Qwen3Encoder`) adapted to incremental KV-cached decode.
pub struct BooguRewriter {
    weights: HashMap<String, Tensor>,
    config: Qwen3Config,
    device: Arc<CudaDevice>,
    /// Deterministic RNG for multinomial sampling (greedy ignores it). Seeded
    /// once at `load`; advances per sampled token.
    rng: StdRng,
}

impl BooguRewriter {
    /// Load the Boogu mllm Qwen3-VL language tower for GENERATE from `mllm_dir`.
    ///
    /// Reads every `model-*.safetensors` shard, keeps the language tower
    /// (`model.language_model.*` → `model.*`), AND keeps `lm_head.weight` +
    /// `model.norm.weight` (the encoder dropped both). Weights stay in native
    /// `[out, in]` layout for `fused_linear3d_native`.
    pub fn load(mllm_dir: impl AsRef<Path>, seed: u64, device: Arc<CudaDevice>) -> Result<Self> {
        let raw = load_sharded_weights(mllm_dir.as_ref(), &device)?;
        let weights = remap_for_generate(raw)?;
        Ok(Self {
            weights,
            config: Qwen3Config::qwen3_vl_text(),
            device,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Sample one token id from `logits` (`[1, vocab]` BF16).
    ///
    /// Mirrors `gemma4::TemperatureSampler::sample` math (greedy argmax when
    /// `temperature < 1e-5`, else temperature-scaled softmax + host cumsum
    /// multinomial) but returns the `u32` index DIRECTLY on the host — it does
    /// NOT round-trip the index through `Tensor::from_vec(..).to_dtype(I32)`,
    /// because flame-core's F32→I32 `to_dtype` is a BIT-reinterpret (not a
    /// numeric cast), which corrupts the packed index (e.g. 64.0f32 →
    /// 1115684864i32). The heavy elementwise (softmax) still runs on GPU.
    fn sample_logits(&mut self, logits: &Tensor, temperature: f32) -> Result<u32> {
        let dims = logits.shape().dims().to_vec();
        let vocab = *dims.last().unwrap();
        if temperature < 1e-5 {
            // Greedy argmax over the F32 logits row.
            let host = logits.to_vec_f32()?;
            let row = &host[..vocab];
            let mut best_i = 0usize;
            let mut best_x = row[0];
            for (i, &x) in row.iter().enumerate().skip(1) {
                if x > best_x {
                    best_x = x;
                    best_i = i;
                }
            }
            Ok(best_i as u32)
        } else {
            // Temperature-scaled softmax on GPU, then host cumsum multinomial.
            let scaled = logits.mul_scalar(1.0 / temperature)?;
            let probs = scaled.softmax(-1)?;
            let host = probs.to_vec_f32()?;
            let row = &host[..vocab];
            let u: f32 = self.rng.gen_range(0.0f32..1.0f32);
            let mut acc = 0.0f32;
            let mut picked = vocab - 1;
            for (i, &p) in row.iter().enumerate() {
                acc += p;
                if u <= acc {
                    picked = i;
                    break;
                }
            }
            Ok(picked as u32)
        }
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("rewriter: missing weight {key}")))
    }

    // -- embedding lookup ---------------------------------------------------

    /// Gather `[1, S, hidden]` BF16 embeddings for `token_ids`.
    fn embed(&self, token_ids: &[i32]) -> Result<Tensor> {
        let embed_w = self.w("model.embed_tokens.weight")?;
        let s = token_ids.len();
        let ids = Tensor::from_vec(
            token_ids.iter().map(|&i| i as f32).collect(),
            Shape::from_dims(&[s]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let rows = embed_w.index_select0(&ids)?; // [S, hidden]
        rows.unsqueeze(0) // [1, S, hidden]
    }

    // -- RoPE table (half-split, position-offset for incremental decode) -----

    /// Build half-split RoPE cos/sin tables of length `seq_len` starting at
    /// absolute position `position_offset`. Returns `[1,1,seq_len,head_dim/2]`
    /// BF16 — the layout `rope_halfsplit_bf16` expects (HF Qwen3 convention).
    ///
    /// During PREFILL: `seq_len = prompt_len`, `position_offset = 0`. During
    /// DECODE: `seq_len = 1`, `position_offset = past_len` (the absolute
    /// position of the new token). RoPE positions are ABSOLUTE, not
    /// cache-relative.
    fn build_rope(&self, seq_len: usize, position_offset: usize) -> Result<(Tensor, Tensor)> {
        let head_dim = self.config.head_dim;
        let theta = self.config.rope_theta;
        let half = head_dim / 2;
        let mut cos_data = vec![0.0f32; seq_len * half];
        let mut sin_data = vec![0.0f32; seq_len * half];
        for p_local in 0..seq_len {
            let pos = (position_offset + p_local) as f64;
            for i in 0..half {
                let inv_freq = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
                let angle = pos * inv_freq;
                cos_data[p_local * half + i] = angle.cos() as f32;
                sin_data[p_local * half + i] = angle.sin() as f32;
            }
        }
        let shape = Shape::from_dims(&[1, 1, seq_len, half]);
        let cos = Tensor::from_vec(cos_data, shape.clone(), self.device.clone())?
            .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(sin_data, shape, self.device.clone())?.to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    // -- causal mask for prefill (S>1); decode (S=1) needs none --------------

    /// Build a `[1,1,S,K]` BF16 keep-mask (1.0 attend, 0.0 masked) for a query
    /// block of length `S` whose first row is at absolute position
    /// `query_offset`, attending to `k_total` cached keys (positions
    /// `0..k_total`). Causal: query row r (absolute `query_offset+r`) attends
    /// to key col c iff `c <= query_offset + r`.
    fn build_causal_mask(&self, s: usize, query_offset: usize, k_total: usize) -> Result<Tensor> {
        let mut data = vec![0.0f32; s * k_total];
        for r in 0..s {
            let q_abs = query_offset + r;
            for c in 0..k_total {
                if c <= q_abs {
                    data[r * k_total + c] = 1.0;
                }
            }
        }
        Tensor::from_vec(data, Shape::from_dims(&[1, 1, s, k_total]), self.device.clone())?
            .to_dtype(DType::BF16)
    }

    // -- one decoder layer, incremental + KV-cached --------------------------

    /// Run one Qwen3 decoder layer with KV-cache. `x`: `[1, S, hidden]`.
    /// `position_offset`: absolute position of `x`'s first row (for RoPE +
    /// mask). Mirrors `Qwen3Encoder::layer_forward`, adapted to grow the cache
    /// and attend over all cached keys.
    fn layer_forward(
        &self,
        layer_idx: usize,
        x: &Tensor,
        position_offset: usize,
        cos: &Tensor,
        sin: &Tensor,
        kv: &mut LayerKvCache,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let prefix = format!("model.layers.{layer_idx}");

        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let s = dims[1];

        // --- self-attention sub-block ---
        let residual = x.clone();

        let in_norm_w = self.w(&format!("{prefix}.input_layernorm.weight"))?;
        let normed = rms_norm(x, in_norm_w, cfg.rms_norm_eps)?;

        let q_w = self.w(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k_w = self.w(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v_w = self.w(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let q = flame_core::ops::fused_inference::fused_linear3d_native(&normed, q_w, None)?;
        let k_new = flame_core::ops::fused_inference::fused_linear3d_native(&normed, k_w, None)?;
        let v_new = flame_core::ops::fused_inference::fused_linear3d_native(&normed, v_w, None)?;

        // [1, S, h*d] -> [1, h, S, d]
        let mut q = q.reshape(&[b, s, h, d])?.permute(&[0, 2, 1, 3])?.contiguous()?;
        let mut k_new = k_new
            .reshape(&[b, s, h_kv, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;
        let v_new = v_new
            .reshape(&[b, s, h_kv, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;

        // per-head QK-RMSNorm (Qwen3) BEFORE RoPE — same as the encoder.
        let q_norm_w = self.w(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm_w = self.w(&format!("{prefix}.self_attn.k_norm.weight"))?;
        q = per_head_rms_norm(&q, q_norm_w, cfg.rms_norm_eps)?;
        k_new = per_head_rms_norm(&k_new, k_norm_w, cfg.rms_norm_eps)?;

        // half-split RoPE on Q and the NEW K only (cached K is already rotated).
        let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, cos, sin)?;
        let k_new = flame_core::bf16_ops::rope_halfsplit_bf16(&k_new, cos, sin)?;

        // grow the cache; fetch the full rotated K + V.
        let (k_full, v_full) = kv.append(&k_new, &v_new)?;
        let k_total = kv.valid_len;

        // GQA: repeat KV heads to match Q heads.
        let k_full = repeat_kv(&k_full, n_rep)?;
        let v_full = repeat_kv(&v_full, n_rep)?;

        // SDPA. Prefill (S>1) uses a causal keep-mask; decode (S=1) attends to
        // all cached keys (single query, position == k_total-1 attends to all),
        // so no mask is needed.
        let attn = if s > 1 {
            let mask = self.build_causal_mask(s, position_offset, k_total)?;
            flame_core::attention::sdpa(&q, &k_full, &v_full, Some(&mask))?
        } else {
            flame_core::attention::sdpa(&q, &k_full, &v_full, None)?
        };

        // [1, h, S, d] -> [1, S, h*d]
        let attn = attn.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[b, s, h * d])?;
        let o_w = self.w(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let attn = flame_core::ops::fused_inference::fused_linear3d_native(&attn, o_w, None)?;
        let x = residual.add(&attn)?;

        // --- MLP sub-block (SwiGLU = down(silu(gate)*up)) ---
        let residual = x.clone();
        let post_w = self.w(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normed2 = rms_norm(&x, post_w, cfg.rms_norm_eps)?;
        let gate_w = self.w(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_w = self.w(&format!("{prefix}.mlp.up_proj.weight"))?;
        let down_w = self.w(&format!("{prefix}.mlp.down_proj.weight"))?;
        let gate = flame_core::ops::fused_inference::fused_linear3d_native(&normed2, gate_w, None)?;
        let up = flame_core::ops::fused_inference::fused_linear3d_native(&normed2, up_w, None)?;
        let mlp = gate.silu()?.mul(&up)?;
        let mlp = flame_core::ops::fused_inference::fused_linear3d_native(&mlp, down_w, None)?;
        residual.add(&mlp)
    }

    /// Run the backbone over `input_ids` (`[1, S]` host ids) starting at
    /// absolute `position_offset`, updating the per-layer KV caches in place.
    /// Returns the FINAL hidden states `[1, S, hidden]` (post final RMSNorm).
    fn backbone_forward(
        &self,
        token_ids: &[i32],
        position_offset: usize,
        kv: &mut [LayerKvCache],
    ) -> Result<Tensor> {
        let s = token_ids.len();
        let mut x = self.embed(token_ids)?;
        let (cos, sin) = self.build_rope(s, position_offset)?;
        for i in 0..self.config.num_layers {
            x = self.layer_forward(i, &x, position_offset, &cos, &sin, &mut kv[i])?;
        }
        // final RMSNorm (model.norm) — the encoder skips this; generate needs it.
        let norm_w = self.w("model.norm.weight")?;
        rms_norm(&x, norm_w, self.config.rms_norm_eps)
    }

    /// LM head over a single position. `hidden_last`: `[1, hidden]`. Returns
    /// logits `[1, vocab]` BF16. `lm_head.weight` is `[vocab, hidden]` native
    /// (NOT tied — `tie_word_embeddings=false`).
    fn lm_head(&self, hidden_last: &Tensor) -> Result<Tensor> {
        let dims = hidden_last.shape().dims().to_vec();
        let (b, hidden) = (dims[0], dims[1]);
        let x = hidden_last.reshape(&[b, 1, hidden])?;
        let lm_w = self.w("lm_head.weight")?;
        let logits = flame_core::ops::fused_inference::fused_linear3d_native(&x, lm_w, None)?;
        let vocab = logits.shape().dims()[2];
        logits.reshape(&[b, vocab])
    }

    /// Full autoregressive generation from already-tokenized prompt ids.
    ///
    /// Prefills the prompt (fills KV-cache, single sample off the last row),
    /// then decodes token-by-token until an eos id ([`QWEN_EOS_IDS`]) or
    /// `max_new_tokens`. Returns the generated token ids (NOT including the
    /// prompt, NOT including the terminating eos).
    ///
    /// `temperature < 1e-5` ⇒ greedy argmax (deterministic — used for parity).
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Err(flame_core::Error::InvalidInput(
                "rewriter: empty prompt".into(),
            ));
        }
        let prompt_len = prompt_ids.len();

        // fresh KV-cache, one slot per layer.
        let mut kv: Vec<LayerKvCache> = (0..self.config.num_layers)
            .map(|_| LayerKvCache::new(self.config.num_kv_heads, self.config.head_dim, &self.device))
            .collect::<Result<_>>()?;

        // --- PREFILL ---
        let t0 = Instant::now();
        let hidden = self.backbone_forward(prompt_ids, 0, &mut kv)?;
        let h_dims = hidden.shape().dims().to_vec();
        let hidden_last = hidden
            .narrow_owning(1, h_dims[1] - 1, 1)?
            .reshape(&[h_dims[0], h_dims[2]])?; // [1, hidden]
        let logits = self.lm_head(&hidden_last)?;
        let mut next_id = self.sample_logits(&logits, temperature)?;
        log::info!(
            "rewriter prefill: {prompt_len} tokens in {:.2}s",
            t0.elapsed().as_secs_f32()
        );

        // --- DECODE ---
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let t_dec = Instant::now();
        while !QWEN_EOS_IDS.contains(&next_id) && generated.len() < max_new_tokens {
            generated.push(next_id);
            // absolute position of the token we are about to feed: it sits right
            // after everything already in the cache. The cache currently holds
            // prompt_len + (generated.len() - 1) columns (prompt + the tokens
            // emitted in prior iterations); the new token's position id equals
            // that count.
            let past_len = prompt_len + generated.len() - 1;
            let hidden = self.backbone_forward(&[next_id as i32], past_len, &mut kv)?;
            let dims = hidden.shape().dims().to_vec();
            let hidden_last = hidden.reshape(&[dims[0], dims[2]])?; // [1, hidden]
            let logits = self.lm_head(&hidden_last)?;
            next_id = self.sample_logits(&logits, temperature)?;
        }
        log::info!(
            "rewriter decode: {} tokens in {:.2}s ({:.1} tok/s)",
            generated.len(),
            t_dec.elapsed().as_secs_f32(),
            generated.len() as f32 / t_dec.elapsed().as_secs_f32().max(1e-6)
        );
        Ok(generated)
    }

    /// Borrow the config (head/layer counts, etc.).
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Shared layer-math helpers (mirror the encoder; kept module-local so the
// generate engine is self-contained)
// ---------------------------------------------------------------------------

/// RMSNorm over the last dim, any rank. Flatten → `rms_norm_bf16` → reshape.
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x2 = x.reshape(&[batch, hidden])?;
    let n = flame_core::cuda_ops_bf16::rms_norm_bf16(&x2, Some(weight), eps)?;
    n.reshape(&dims)
}

/// Per-head RMSNorm for `[B,H,S,d]` (reduces over head_dim only).
fn per_head_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let head_dim = dims[3];
    let flat: usize = dims[..3].iter().product();
    let x2 = x.reshape(&[flat, head_dim])?;
    let n = flame_core::cuda_ops_bf16::rms_norm_bf16(&x2, Some(weight), eps)?;
    n.reshape(&dims)
}

/// GQA KV broadcast: `[1,h_kv,S,d]` → `[1,h_kv*n_rep,S,d]` (stack+reshape).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    let (b, h_kv, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    let stacked = Tensor::stack(&copies, 2)?; // [b, h_kv, n_rep, s, d]
    stacked.reshape(&[b, h_kv * n_rep, s, d])
}

// ---------------------------------------------------------------------------
// Weight loading + remap (generate variant: KEEPS lm_head + final norm)
// ---------------------------------------------------------------------------

/// Load every `model-*.safetensors` shard in `dir` into one flat BF16 map.
/// Mirrors `encoder::load_sharded_weights`.
fn load_sharded_weights(dir: &Path, device: &Arc<CudaDevice>) -> Result<HashMap<String, Tensor>> {
    let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| {
            flame_core::Error::InvalidInput(format!(
                "rewriter: cannot read mllm dir {}: {e}",
                dir.display()
            ))
        })?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with("model-") && n.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err(flame_core::Error::InvalidInput(format!(
            "rewriter: no model-*.safetensors shards in {}",
            dir.display()
        )));
    }
    let mut all = HashMap::new();
    for (i, path) in shard_paths.iter().enumerate() {
        let t0 = Instant::now();
        let shard = flame_core::serialization::load_file(path, device)?;
        log::info!(
            "rewriter mllm shard {}/{}: {} keys ({:.1}s)",
            i + 1,
            shard_paths.len(),
            shard.len(),
            t0.elapsed().as_secs_f32()
        );
        all.extend(shard);
    }
    Ok(all)
}

/// Remap raw mllm keys for GENERATE: strip the `language_model.` segment
/// (`model.language_model.* → model.*`), and ADDITIONALLY keep:
///   - `lm_head.weight` (bare key, NOT under `language_model.`; the rewriter
///     needs it — `tie_word_embeddings=false`).
///   - `model.norm.weight` (the final RMSNorm — comes through the strip as
///     `model.language_model.norm.weight → model.norm.weight`).
/// Drops `model.visual.*` (vision tower). Verifies the 398 Qwen3 keys + lm_head.
fn remap_for_generate(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    const LM_PREFIX: &str = "model.language_model.";
    let mut out = HashMap::with_capacity(raw.len());
    let mut have_lm_head = false;
    for (k, v) in raw.into_iter() {
        if let Some(rest) = k.strip_prefix(LM_PREFIX) {
            out.insert(format!("model.{rest}"), v); // includes model.norm.weight
        } else if k == "lm_head.weight" {
            out.insert("lm_head.weight".to_string(), v);
            have_lm_head = true;
        }
        // else: model.visual.* etc. -> dropped.
    }

    // Completeness: the 398 language-tower keys (embed + 36×11 + final norm)
    // must all resolve, PLUS lm_head.
    let expected = expected_weight_keys(BOOGU_MLLM_NUM_LAYERS);
    let missing: Vec<&String> = expected.iter().filter(|k| !out.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(flame_core::Error::InvalidInput(format!(
            "rewriter: {} language-tower key(s) missing after remap (e.g. {:?})",
            missing.len(),
            missing.iter().take(5).collect::<Vec<_>>()
        )));
    }
    if !have_lm_head {
        return Err(flame_core::Error::InvalidInput(
            "rewriter: lm_head.weight not found in mllm shards (generate needs it; \
             tie_word_embeddings=false)"
                .into(),
        ));
    }
    log::info!(
        "rewriter mllm: remapped {} keys (lang tower + lm_head + final norm; vision dropped)",
        out.len()
    );
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_set_matches_generation_config() {
        // mllm generation_config.json: eos_token_id = [151645, 151643].
        assert!(QWEN_EOS_IDS.contains(&151_645));
        assert!(QWEN_EOS_IDS.contains(&151_643));
        assert_eq!(QWEN_IM_END_ID, 151_645);
    }

    #[test]
    fn config_is_qwen3_vl_text() {
        let c = Qwen3Config::qwen3_vl_text();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_layers, BOOGU_MLLM_NUM_LAYERS);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.rope_theta, 5_000_000.0);
    }

    #[test]
    fn remap_keeps_lm_head_and_norm_via_string_logic() {
        // Mirror the strip/keep decision on representative keys (no tensors).
        const LM_PREFIX: &str = "model.language_model.";
        let cases = [
            ("model.language_model.embed_tokens.weight", Some("model.embed_tokens.weight")),
            ("model.language_model.norm.weight", Some("model.norm.weight")),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                Some("model.layers.0.self_attn.q_proj.weight"),
            ),
        ];
        for (src, want) in cases {
            let got = src.strip_prefix(LM_PREFIX).map(|r| format!("model.{r}"));
            assert_eq!(got.as_deref(), want);
        }
        // lm_head.weight is NOT under the prefix → strip returns None, kept by
        // the explicit `else if` branch.
        assert!("lm_head.weight".strip_prefix(LM_PREFIX).is_none());
        // model.visual.* dropped (no prefix match, not lm_head).
        assert!("model.visual.blocks.0.attn.qkv.weight"
            .strip_prefix(LM_PREFIX)
            .is_none());
    }

    #[test]
    fn rope_half_is_64() {
        // head_dim 128 → half 64 (the cos/sin table width).
        let c = Qwen3Config::qwen3_vl_text();
        assert_eq!(c.head_dim / 2, 64);
    }
}
