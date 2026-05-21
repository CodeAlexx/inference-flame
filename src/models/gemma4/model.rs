//! Top-level Gemma-4 text model + causal-LM wrapper.
//!
//! ```text
//! Gemma4TextModel:
//!   embed_tokens(input_ids) → x  // multiplied by sqrt(hidden_size) per Gemma family
//!   for layer in layers:
//!     x = layer(x, kv_cache[layer])
//!   x = final_norm(x)            // RMSNorm
//!
//! Gemma4ForCausalLM:
//!   hidden = Gemma4TextModel(input_ids)
//!   logits = hidden @ embed_tokens.weight.T  // tied embedding
//!   logits = tanh(logits / softcap) * softcap  // final_logit_softcapping
//! ```
//!
//! Reference: `modeling_gemma4.py` lines 1498-1699 (TextModel) and
//! 1700-2073 (ForCausalLM).

use crate::models::gemma4::decoder::Gemma4DecoderLayer;
use crate::models::gemma4::kv_cache::Gemma4KvCache;
use crate::models::gemma4::sampler::TemperatureSampler;
use crate::models::gemma4::tokenizer::Gemma4Tokenizer;
use crate::models::gemma4::{Gemma4Config, LayerType};
use crate::models::hidream_o1::prompt_agent::RewriteBackend;
use flame_core::{DType, Result, Shape, Tensor};
use std::path::Path;
use std::sync::Arc;

/// Backbone — embedding + 60 decoder layers + final RMSNorm.
pub struct Gemma4TextModel {
    pub cfg: Gemma4Config,
    /// `embed_tokens.weight` — `[vocab_size, hidden_size]` BF16.
    /// Also used as the LM head (tied) via transposed matmul.
    pub embed_w: Tensor,
    /// Final RMSNorm weight `[hidden_size]`.
    pub final_norm_w: Tensor,
    /// 60 decoder layers, in order. Each layer's weights are streamed
    /// from pinned RAM by `BlockOffloader` immediately before its
    /// forward call.
    pub layers: Vec<Gemma4DecoderLayer>,
}

/// Build a cos/sin RoPE table over a contiguous span of positions.
///
/// Mirrors the helper at `inference-flame/src/models/t5gemma2_encoder.rs:742`
/// (Gemma family pattern). Returns BF16 tensors of shape `[1, 1, seq_len, rot_dim/2]`.
///
/// `position_offset`: position id of the first row. During PREFILL = 0;
/// during DECODE = past_seq_len. This lets the decode path build a
/// single-row table for the current token without rebuilding the
/// whole history.
///
/// AGENT-DEFAULT (Builder 3): scaling factor is hard-coded to 1.0.
/// The Gemma-4 config flags `rope_type="proportional"` on full
/// layers; the reference implementation interprets proportional as
/// "no rescale at default max_position_embeddings" — verify against
/// `modeling_gemma4.py:1035-1124` during parity. If a scaling factor
/// is needed, route it via an extra parameter rather than adding a
/// second function.
fn build_rope_table(
    seq_len: usize,
    rot_dim: usize,
    position_offset: usize,
    theta: f64,
    device: &Arc<flame_core::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half = rot_dim / 2;
    let mut cos_data = vec![0.0f32; seq_len * half];
    let mut sin_data = vec![0.0f32; seq_len * half];
    for pos_local in 0..seq_len {
        let pos = (position_offset + pos_local) as f64;
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * i) as f64 / rot_dim as f64);
            let angle = pos * inv_freq;
            cos_data[pos_local * half + i] = angle.cos() as f32;
            sin_data[pos_local * half + i] = angle.sin() as f32;
        }
    }
    let shape = Shape::from_dims(&[1, 1, seq_len, half]);
    let cos = Tensor::from_vec(cos_data, shape.clone(), device.clone())?.to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(sin_data, shape, device.clone())?.to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Build a causal mask for prefill. `seq_len > 1`. During decode (S=1)
/// no mask is needed and the caller passes `None` to the layer.
///
/// Convention: 1.0 = attend, 0.0 = masked. flame-core's SDPA converts
/// to additive (-inf) internally. Returns `[1, 1, S, S]` BF16.
///
/// `sliding_window`: when `Some(w)`, additionally restricts each row
/// to its `w` most recent keys. Matches the pattern at
/// `inference-flame/src/models/gemma3_encoder.rs:333`.
fn build_causal_mask(
    seq_len: usize,
    sliding_window: Option<usize>,
    device: &Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let is_causal = j <= i;
            let in_window = sliding_window.map(|w| i.saturating_sub(w) < j + 1).unwrap_or(true);
            if is_causal && in_window {
                data[i * seq_len + j] = 1.0;
            }
        }
    }
    let mask = Tensor::from_vec(
        data,
        Shape::from_dims(&[1, 1, seq_len, seq_len]),
        device.clone(),
    )?;
    mask.to_dtype(DType::BF16)
}

impl Gemma4TextModel {
    /// Forward over a sequence of input ids, updating the KV cache in place.
    ///
    /// `input_ids`: `[B, S]` I32. During PREFILL, S = prompt length.
    /// During DECODE, S = 1.
    /// `position_ids_start`: position id of the first column of
    /// `input_ids` — for prefill = 0; for decode = past_seq_len.
    /// `kv_cache`: per-layer KV cache. New columns appended in place.
    ///
    /// Returns hidden states `[B, S, hidden_size]` BF16 from the
    /// final RMSNorm.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids_start: usize,
        kv_cache: &mut Gemma4KvCache,
    ) -> Result<Tensor> {
        let dims = input_ids.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(flame_core::Error::InvalidInput(format!(
                "Gemma4TextModel::forward: input_ids must be [B, S], got {dims:?}"
            )));
        }
        let b = dims[0];
        let s = dims[1];
        let hidden = self.cfg.hidden_size;
        let device_arc = self.embed_w.device().clone();

        // ── 1. Embed + Gemma scale (multiply by sqrt(hidden_size)). ──────────
        // The scale lives inside the embedding per Gemma4TextScaledWordEmbedding
        // (modeling_gemma4.py:1414-1497). We apply it explicitly after
        // index_select0 since we don't carry a scaled-embedding wrapper.
        //
        // index_select0 takes a 1-D index tensor and returns [N, hidden];
        // we reshape to [B, S, hidden] then apply the scale.
        let ids_flat = input_ids.reshape(&[b * s])?;
        let embedded_flat = self.embed_w.index_select0(&ids_flat)?; // [B*S, hidden]
        let embedded = embedded_flat.reshape(&[b, s, hidden])?;
        let scale = (hidden as f32).sqrt();
        let mut x = embedded.mul_scalar(scale)?;

        // ── 2. Build per-rope-flavor cos/sin tables for THIS forward. ────────
        // For prefill (S>1) we build tables of length S over positions
        // [position_ids_start .. position_ids_start + S).
        //
        // Sliding layers: rotate full head_dim, theta=10K.
        // Full layers: rotate `rotary_dim_full` (= 2*floor(head_dim*0.25))
        // dims, theta=1M, proportional scaling (factor=1 here — see
        // AGENT-DEFAULT note in build_rope_table).
        let rotary_dim_full = {
            let pairs = ((self.cfg.head_dim as f32) * self.cfg.partial_rotary_factor_full / 2.0)
                .floor() as usize;
            pairs * 2
        };
        let (cos_sliding, sin_sliding) = build_rope_table(
            s,
            self.cfg.head_dim,
            position_ids_start,
            self.cfg.rope_theta_sliding as f64,
            &device_arc,
        )?;
        let (cos_full, sin_full) = build_rope_table(
            s,
            rotary_dim_full,
            position_ids_start,
            self.cfg.rope_theta_full as f64,
            &device_arc,
        )?;

        // ── 3. Build masks. None during decode (S=1, single query attends
        //       to all cached keys; SDPA's no-mask path is correct).
        //       During prefill, separate full vs sliding causal masks. ──────
        let (full_mask, sliding_mask): (Option<Tensor>, Option<Tensor>) = if s > 1 {
            let f = build_causal_mask(s, None, &device_arc)?;
            let sw = build_causal_mask(s, Some(self.cfg.sliding_window), &device_arc)?;
            (Some(f), Some(sw))
        } else {
            (None, None)
        };

        // ── 4. Layer loop ────────────────────────────────────────────────────
        for (i, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = match self.cfg.layer_types[i] {
                LayerType::Sliding => (&cos_sliding, &sin_sliding),
                LayerType::Full => (&cos_full, &sin_full),
            };
            let mask = match self.cfg.layer_types[i] {
                LayerType::Sliding => sliding_mask.as_ref(),
                LayerType::Full => full_mask.as_ref(),
            };
            x = layer.forward(&x, &self.cfg, &mut kv_cache.layers[i], cos, sin, mask)?;
        }

        // ── 5. Final RMSNorm ─────────────────────────────────────────────────
        // Gemma-family pattern: weight is added to 1.0 inside the norm
        // (see gemma3_encoder::gemma3_rms_norm). Builder 2's decoder layer
        // is responsible for using the same convention internally; here
        // we apply the plain BF16 RMSNorm primitive and trust the weight
        // file to store (weight-1) if needed.
        //
        // AGENT-DEFAULT (Builder 3): we apply `rms_norm_bf16(x, &final_norm_w, eps)`
        // directly. Parity check against the Python reference will reveal
        // if the Gemma-4 final norm needs the +1.0 wrapper. If yes, swap
        // for the gemma3_encoder::gemma3_rms_norm pattern.
        let x = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &x,
            Some(&self.final_norm_w),
            self.cfg.rms_norm_eps,
        )?;

        Ok(x)
    }
}

/// Causal-LM wrapper: backbone + tied LM head + final softcap.
pub struct Gemma4ForCausalLM {
    pub backbone: Gemma4TextModel,
}

impl Gemma4ForCausalLM {
    /// Apply LM head to a single position (typically the last column
    /// of a prefill, or the only column of a decode step).
    ///
    /// `hidden_last`: `[B, hidden_size]` (already sliced to the
    /// position-of-interest). Returns logits `[B, vocab_size]` BF16
    /// with final softcap applied: `tanh(logits / softcap) * softcap`.
    pub fn lm_head(&self, hidden_last: &Tensor) -> Result<Tensor> {
        let dims = hidden_last.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(flame_core::Error::InvalidInput(format!(
                "Gemma4ForCausalLM::lm_head: expected [B, H], got {dims:?}"
            )));
        }
        let b = dims[0];
        let hidden = dims[1];

        // fused_linear3d_native expects [B, S, K]. Reshape to [B, 1, H],
        // call the linear, then squeeze the singleton seq dim.
        // The weight is embed_w in PyTorch layout [vocab, hidden] —
        // fused_linear3d_native applies TRANSA=T internally so we pass
        // it as-is.
        let x_3d = hidden_last.reshape(&[b, 1, hidden])?;
        let logits_3d = flame_core::ops::fused_inference::fused_linear3d_native(
            &x_3d,
            &self.backbone.embed_w,
            None,
        )?;
        // [B, 1, vocab] → [B, vocab]
        let vocab = logits_3d.shape().dims()[2];
        let logits = logits_3d.reshape(&[b, vocab])?;

        // ── Final logit softcap: `tanh(x / softcap) * softcap`. ──────────────
        // AGENT-DEFAULT (Builder 3): composed from three primitives
        // (mul_scalar, tanh, mul_scalar). flame-core has no fused logit
        // softcap kernel; this is a single-call site per token at
        // [B, vocab=262K], so the per-token cost is small (one BF16
        // pass × 3 over ~520 KB at ~600 GB/s ≈ 0.5 μs).
        //
        // If the soft-cap shows up as a hot spot in a profile, the
        // right fix is a new flame-core fused `logit_softcap_bf16` —
        // do it once in flame-core, not as a per-model kernel here.
        let softcap = self.backbone.cfg.final_logit_softcapping;
        let scaled = logits.mul_scalar(1.0 / softcap)?;
        let capped = scaled.tanh()?;
        capped.mul_scalar(softcap)
    }
}

/// Public load + decode entry point for the Gemma-4-31B-it text decoder.
///
/// Manages: weight loader, BlockOffloader, KV cache lifecycle,
/// autoregressive sampling loop. This is what the prompt agent
/// trait implementation drives.
pub struct Gemma4Decoder {
    pub model: Gemma4ForCausalLM,
    pub tokenizer: Gemma4Tokenizer,
    pub sampler: TemperatureSampler,
    pub device: flame_core::Device,
    /// Cap on KV cache allocation. Default 8192 — enough for the agent's
    /// system 3K + user 0.2K + gen 4K. Bump via `with_max_seq` if needed.
    pub max_seq: usize,
}

impl Gemma4Decoder {
    /// Load the model from a Hugging-Face-style directory containing
    /// `config.json`, `tokenizer.json`, `model.safetensors.index.json`,
    /// and the `model-*.safetensors` shards. Weights are paged via
    /// `BlockOffloader` so the resident GPU footprint stays under the
    /// 24 GB limit.
    ///
    /// AGENT-DEFAULT (Builder 3): this entry point depends on
    /// `Gemma4WeightLoader::open` (Builder 1) and the layer-builder
    /// path for `Gemma4DecoderLayer` (Builder 2). Until both Builder 1
    /// and Builder 2 land their loaders, `load` cannot run end-to-end.
    /// The skeleton below is the wiring once those land.
    pub fn load(model_dir: &Path, device: flame_core::Device) -> anyhow::Result<Self> {
        // 1. Config — for the 31B-it default we can short-circuit the
        //    config.json parser until Gemma4Config::from_config_json
        //    lands (it's a TODO in mod.rs:198). Other model sizes need
        //    the JSON path.
        let cfg = Gemma4Config::gemma4_31b_it();
        let _ = model_dir; // model_dir is consumed by the loader/tokenizer

        // 2. Tokenizer.
        let tokenizer = Gemma4Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;

        // 3. Weight loader — owns the BlockOffloader and pinned-RAM staging.
        //    Returns the resident embed_w / final_norm_w plus a builder for
        //    each layer's `Gemma4DecoderLayer` struct. Until Builder 1
        //    finalizes that API we keep this path symbolic; once the
        //    loader is in, this becomes:
        //
        //    let loader = Gemma4WeightLoader::open(model_dir, &cfg, &device)?;
        //    let embed_w = loader.embed_tokens()?;
        //    let final_norm_w = loader.final_norm()?;
        //    let layers: Vec<Gemma4DecoderLayer> = (0..cfg.num_hidden_layers)
        //        .map(|i| loader.build_layer(i))
        //        .collect::<Result<_, _>>()?;
        //
        // For the moment we surface a clear error so callers know which
        // pieces are still owed by other builders.
        anyhow::bail!(
            "Gemma4Decoder::load awaits Builder 1 (Gemma4WeightLoader::open + \
             Gemma4Tokenizer::from_file) and Builder 2 (Gemma4DecoderLayer construction \
             from loaded weights). Once those land, replace this bail with the \
             commented wiring above. cfg.num_hidden_layers={}, tokenizer_loaded={}, \
             device={:?}",
            cfg.num_hidden_layers,
            tokenizer.encode("").is_ok(),
            device.ordinal()
        )
    }

    /// Full autoregressive generation. `chat_template_prompt` is the
    /// already-rendered Gemma-4 chat template
    /// (see `hidream_o1::prompt_agent::render_chat_template`).
    /// Returns the decoded text up to (but not including) the first
    /// EOS token, or up to `max_new_tokens`, whichever comes first.
    pub fn generate(
        &mut self,
        chat_template_prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> anyhow::Result<String> {
        let cfg = &self.model.backbone.cfg;
        let device_arc = self.device.cuda_device_arc();

        // ── 1. Encode prompt to u32 token ids. ──────────────────────────────
        let prompt_ids: Vec<u32> = self.tokenizer.encode(chat_template_prompt)?;
        if prompt_ids.is_empty() {
            anyhow::bail!("Gemma4Decoder::generate: tokenizer returned 0 tokens");
        }
        let prompt_len = prompt_ids.len();
        if prompt_len + max_new_tokens > self.max_seq {
            anyhow::bail!(
                "Gemma4Decoder::generate: prompt_len {} + max_new_tokens {} > max_seq {}",
                prompt_len,
                max_new_tokens,
                self.max_seq
            );
        }

        // ── 2. Build the input_ids tensor [1, prompt_len] I32 for PREFILL. ──
        let ids_f32: Vec<f32> = prompt_ids.iter().map(|&id| id as f32).collect();
        let input_ids = Tensor::from_vec(
            ids_f32,
            Shape::from_dims(&[1, prompt_len]),
            device_arc.clone(),
        )?
        .to_dtype(DType::I32)?;

        // ── 3. KV cache (60 layers, one slot each). ─────────────────────────
        let mut kv_cache = Gemma4KvCache::new(cfg, 1, self.max_seq, &self.device)?;

        // ── 4. PREFILL: run the full prompt through the backbone. ───────────
        let hidden = self.model.backbone.forward(&input_ids, 0, &mut kv_cache)?;
        // hidden: [1, prompt_len, hidden_size] — take only the last row.
        let h_dims = hidden.shape().dims().to_vec();
        let hidden_last_3d = hidden.narrow_owning(1, h_dims[1] - 1, 1)?; // [1, 1, H]
        let hidden_last = hidden_last_3d.reshape(&[h_dims[0], h_dims[2]])?; // [1, H]

        // ── 5. First sample. ────────────────────────────────────────────────
        let logits = self.model.lm_head(&hidden_last)?;
        let next_id_tensor = self.sampler.sample(&logits, temperature)?;
        let mut next_id: u32 = next_id_tensor.to_vec_i32()?[0] as u32;

        // ── 6. Generated buffer; collect tokens up to EOS / cap. ────────────
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let eos_set: &[u32] = &cfg.eos_token_ids;

        while !eos_set.contains(&next_id) && generated.len() < max_new_tokens {
            generated.push(next_id);

            // Build the next input tensor [1, 1] I32.
            let next_ids_t = Tensor::from_vec(
                vec![next_id as f32],
                Shape::from_dims(&[1, 1]),
                device_arc.clone(),
            )?
            .to_dtype(DType::I32)?;

            // past_len: total positions already in KV cache.
            // For full layers, this equals prompt_len + generated.len() - 1
            // (we just appended one in the previous iteration's append).
            // Sliding layers cap at sliding_window but the *position id* we
            // feed to RoPE keeps growing — RoPE positions are absolute,
            // not cache-relative.
            let past_len = prompt_len + generated.len() - 1;

            let hidden = self.model.backbone.forward(&next_ids_t, past_len, &mut kv_cache)?;
            // [1, 1, H] → [1, H]
            let dims = hidden.shape().dims().to_vec();
            let hidden_last = hidden.reshape(&[dims[0], dims[2]])?;
            let logits = self.model.lm_head(&hidden_last)?;
            let next_id_tensor = self.sampler.sample(&logits, temperature)?;
            next_id = next_id_tensor.to_vec_i32()?[0] as u32;
        }

        // ── 7. Decode collected ids back to text. ───────────────────────────
        let decoded = self.tokenizer.decode(&generated)?;
        Ok(decoded)
    }
}

/// Bridge: connect this concrete decoder to the prompt-agent
/// abstraction so `hidream_o1::prompt_agent::rewrite_prompt(...)` can
/// drive it.
impl RewriteBackend for Gemma4Decoder {
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> anyhow::Result<String> {
        Gemma4Decoder::generate(self, prompt, max_new_tokens, temperature)
    }
}
