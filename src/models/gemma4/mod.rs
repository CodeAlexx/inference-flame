//! Gemma-4 text decoder — pure-Rust port for autoregressive generation.
//!
//! Scoped to the **text-only decoder** of `google/gemma-4-31B-it`. The
//! upstream `Gemma4ForConditionalGeneration` also has a vision tower
//! (`Gemma4VisionEncoder`) and an audio path (`Gemma4Audio*`); neither
//! is reached when the model is invoked text-only, so we skip them
//! entirely.
//!
//! ## What this module is for
//!
//! HiDream-O1's prompt agent
//! (`inference-flame/src/models/hidream_o1/prompt_agent.rs`) needs
//! to call a Gemma-4-31B-it text decoder to rewrite raw user prompts
//! into the SCALIST creative-director's-brief format HiDream-O1's
//! Qwen3-VL text encoder expects. Per TENET 1, the decoder itself
//! lives here as a reusable primitive — any future text-LM need in
//! inference-flame should call into this module rather than ship its
//! own Gemma decoder.
//!
//! ## Architecture (from `google/gemma-4-31B-it/config.json`)
//!
//! | Field | Value |
//! |---|---|
//! | `num_hidden_layers` | 60 |
//! | `hidden_size` | 5376 |
//! | `num_attention_heads` | 32 |
//! | `num_key_value_heads` | 16 (GQA, repeat factor 2) |
//! | `head_dim` | 256 |
//! | `intermediate_size` | 21504 (SwiGLU FFN with `gelu_pytorch_tanh`) |
//! | `vocab_size` | 262_144 |
//! | `rms_norm_eps` | 1e-6 |
//! | `tie_word_embeddings` | true (no separate LM head matrix) |
//! | `final_logit_softcapping` | 30.0 (`tanh(x/30) * 30`) |
//! | `sliding_window` | 1024 |
//! | `layer_types` | `[S,S,S,S,S,F]` × 10 |
//! | RoPE (sliding) | theta=10_000, default rotation |
//! | RoPE (full) | theta=1_000_000, **partial_rotary_factor=0.25**, **proportional** scaling |
//! | dtype | bfloat16 |
//!
//! Notably ABSENT compared to Gemma-3: q_norm, k_norm. MoE fields are
//! all null/false — this is a dense model despite Gemma-4 family
//! having MoE variants.
//!
//! ## Runtime constraint
//!
//! All execution stays on GPU. Weights stream from pinned RAM
//! through `flame_core::activation_offload::BlockOffloader` (the
//! same pattern used for 22B LTX-2 on a 24 GB 3090 Ti), but the
//! matmul / softmax / etc. ALWAYS run on the GPU. No CPU compute
//! fallback at any path. See `weight_loader.rs` for the layer
//! lifetime contract.
//!
//! ## Reference Python
//!
//! `/home/alex/EriDiffusion/.venv_cache/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py`
//! key lines:
//!
//! | Class | Lines | Maps to |
//! |---|---|---|
//! | Gemma4TextMLP | 1016-1033 | `decoder.rs::mlp_forward` |
//! | Gemma4TextRotaryEmbedding | 1035-1124 | `decoder.rs::rope` |
//! | Gemma4TextAttention | 1126-1249 | `decoder.rs::attention_forward` |
//! | Gemma4TextDecoderLayer | 1325-1413 | `decoder.rs::Gemma4DecoderLayer::forward` |
//! | Gemma4TextScaledWordEmbedding | 1414-1497 | `model.rs::embed_with_scale` |
//! | Gemma4TextModel | 1498-1699 | `model.rs::Gemma4TextModel::forward` |
//! | Gemma4ForCausalLM | 1700-2073 | `model.rs::Gemma4ForCausalLM::forward` (LM head + softcap) |

pub mod decoder;
pub mod kv_cache;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod weight_loader;

pub use decoder::Gemma4DecoderLayer;
pub use kv_cache::Gemma4KvCache;
pub use model::{Gemma4ForCausalLM, Gemma4TextModel};
pub use sampler::TemperatureSampler;
pub use tokenizer::Gemma4Tokenizer;
pub use weight_loader::Gemma4WeightLoader;

/// Per-layer attention pattern. `[S,S,S,S,S,F]` × 10 for the 31B
/// variant — 5 sliding-window layers per full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Sliding-window attention. Window size = `Gemma4Config::sliding_window`.
    /// RoPE: theta=10K, default rotation.
    Sliding,
    /// Full bidirectional attention over the whole context.
    /// RoPE: theta=1M, partial_rotary_factor=0.25, proportional scaling.
    Full,
}

/// Gemma-4 text decoder configuration. Populated from a model's
/// `config.json::text_config` field by
/// [`Gemma4Config::from_config_json`].
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // ── Architecture ─────────────────────────────────────────────────
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,

    // ── Output projection ────────────────────────────────────────────
    /// `tie_word_embeddings` — true means LM head reuses the embedding
    /// matrix transposed; no separate `lm_head.weight` is loaded.
    pub tie_word_embeddings: bool,
    /// `final_logit_softcapping`. Logits are clamped via
    /// `tanh(x / softcap) * softcap` before sampling. The 31B variant
    /// uses 30.0.
    pub final_logit_softcapping: f32,

    // ── Attention pattern ────────────────────────────────────────────
    /// Per-layer attention type. `layer_types[i]` ∈ {Sliding, Full}.
    pub layer_types: Vec<LayerType>,
    /// Sliding-window width. 1024 for 31B.
    pub sliding_window: usize,
    /// `attention_k_eq_v` from `config.json`. When true AND the layer is
    /// a Full-attention layer (`use_alternative_attention = attention_k_eq_v
    /// AND NOT is_sliding`), the upstream Python source ties V's projection
    /// to K's output (`v_proj=None`, `value_states = key_states` before the
    /// per-head norms). The current 31B inference port (per PORT_SPEC §2
    /// and the 2026-05-21 coordinator brief) codes against SEPARATE K and V
    /// projections because the 31B safetensors ship both, so this flag is
    /// stored but not currently branched on in the decoder. If a future
    /// variant with truly tied K=V projections ships, add a branch in
    /// `Gemma4DecoderLayer::forward` to skip `v_proj` for Full layers.
    /// 31B value: true.
    pub attention_k_eq_v: bool,

    // ── RoPE (two thetas; layer chooses by `layer_types[i]`) ─────────
    /// Sliding-attention RoPE theta. 10_000 for 31B.
    pub rope_theta_sliding: f32,
    /// Full-attention RoPE theta. 1_000_000 for 31B.
    pub rope_theta_full: f32,
    /// Full-attention partial rotary factor. 0.25 means only the
    /// first `floor(head_dim * 0.25)` dims of Q/K get rotated; the
    /// remainder is passed through unmodified. Applies ONLY to full
    /// attention layers — sliding layers rotate all dims.
    pub partial_rotary_factor_full: f32,

    // ── Position encoding limits ─────────────────────────────────────
    /// `max_position_embeddings`. 262144 for 31B. We never need that
    /// much for prompt rewriting (system ~3K + user ~200 + gen ~3K ≈ 6K).
    pub max_position_embeddings: usize,

    // ── Special tokens (read from tokenizer_config.json) ─────────────
    /// `<bos>` token id.
    pub bos_token_id: u32,
    /// Primary end-of-sequence token. 31B uses 1 (the `<eos>` id);
    /// the model can also stop at `106` (`<end_of_turn>`) — the
    /// `eos_token_ids` field carries both.
    pub eos_token_ids: Vec<u32>,
    /// Padding token id. 0 (typically `<pad>`).
    pub pad_token_id: u32,

    // ── Dtype ────────────────────────────────────────────────────────
    /// Storage dtype. Always `flame_core::DType::BF16` for inference.
    pub dtype: flame_core::DType,
}

impl Gemma4Config {
    /// Default `google/gemma-4-31B-it` config (verified against the
    /// model's actual `config.json` on 2026-05-21).
    pub fn gemma4_31b_it() -> Self {
        let n_layers = 60;
        // `[S,S,S,S,S,F]` × 10 from `config.json::text_config::layer_types`.
        let layer_types: Vec<LayerType> = (0..n_layers)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            })
            .collect();
        Self {
            num_hidden_layers: n_layers,
            hidden_size: 5376,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 256,
            intermediate_size: 21504,
            vocab_size: 262_144,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            final_logit_softcapping: 30.0,
            layer_types,
            sliding_window: 1024,
            attention_k_eq_v: true,
            rope_theta_sliding: 10_000.0,
            rope_theta_full: 1_000_000.0,
            partial_rotary_factor_full: 0.25,
            max_position_embeddings: 262_144,
            bos_token_id: 2,
            // `config.json::eos_token_id == [1, 106]`. 1 = <eos>, 106 = <end_of_turn>.
            eos_token_ids: vec![1, 106],
            pad_token_id: 0,
            dtype: flame_core::DType::BF16,
        }
    }

    /// Load from a model directory's `config.json`. Parses only the
    /// fields under `text_config` (we skip vision_config and
    /// audio_config since this port is text-only).
    ///
    /// Strategy (AGENT-DEFAULT): start from [`Self::gemma4_31b_it`] as a
    /// hand-verified baseline, then overwrite each field whose JSON value
    /// is present. Missing/null fields fall back to the baseline. This
    /// is more forgiving than a serde-derive struct given the upstream
    /// config carries many fields we don't consume (`global_head_dim`,
    /// `num_experts`, `attention_k_eq_v`, etc.) and a few we do want to
    /// flag explicitly if absent (e.g. `final_logit_softcapping`).
    pub fn from_config_json(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::{anyhow, Context};
        let bytes = std::fs::read(path)
            .with_context(|| format!("Gemma4Config::from_config_json: read {}", path.display()))?;
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("Gemma4Config::from_config_json: parse {}", path.display()))?;

        // Text-only port. The top-level `text_config` block carries every
        // architecture field; we never touch `vision_config` / `audio_config`.
        let tc = root
            .get("text_config")
            .ok_or_else(|| anyhow!("{}: missing text_config block", path.display()))?;

        // Optional outer `tie_word_embeddings` override (config.json carries it
        // both at root and under text_config; root wins for the 31B file).
        let outer_tie = root
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool());

        let mut cfg = Self::gemma4_31b_it();

        let as_usize = |v: &serde_json::Value| -> Option<usize> {
            v.as_u64().map(|u| u as usize)
        };
        let as_f32 = |v: &serde_json::Value| -> Option<f32> {
            v.as_f64().map(|f| f as f32)
        };

        if let Some(v) = tc.get("num_hidden_layers").and_then(as_usize) {
            cfg.num_hidden_layers = v;
        }
        if let Some(v) = tc.get("hidden_size").and_then(as_usize) {
            cfg.hidden_size = v;
        }
        if let Some(v) = tc.get("num_attention_heads").and_then(as_usize) {
            cfg.num_attention_heads = v;
        }
        if let Some(v) = tc.get("num_key_value_heads").and_then(as_usize) {
            cfg.num_key_value_heads = v;
        }
        if let Some(v) = tc.get("head_dim").and_then(as_usize) {
            cfg.head_dim = v;
        }
        if let Some(v) = tc.get("intermediate_size").and_then(as_usize) {
            cfg.intermediate_size = v;
        }
        if let Some(v) = tc.get("vocab_size").and_then(as_usize) {
            cfg.vocab_size = v;
        }
        if let Some(v) = tc.get("rms_norm_eps").and_then(as_f32) {
            cfg.rms_norm_eps = v;
        }
        if let Some(v) = tc.get("final_logit_softcapping").and_then(as_f32) {
            cfg.final_logit_softcapping = v;
        }
        if let Some(v) = tc.get("sliding_window").and_then(as_usize) {
            cfg.sliding_window = v;
        }
        if let Some(v) = tc.get("max_position_embeddings").and_then(as_usize) {
            cfg.max_position_embeddings = v;
        }
        // root tie_word_embeddings takes precedence over text_config's (matches
        // transformers' behavior — the root field is what binds for the
        // CausalLM wrapper).
        if let Some(t) = outer_tie.or_else(|| tc.get("tie_word_embeddings").and_then(|v| v.as_bool())) {
            cfg.tie_word_embeddings = t;
        }

        // layer_types: vector of strings → Vec<LayerType>. Each entry is
        // "sliding_attention" or "full_attention".
        if let Some(arr) = tc.get("layer_types").and_then(|v| v.as_array()) {
            let mut types: Vec<LayerType> = Vec::with_capacity(arr.len());
            for (i, entry) in arr.iter().enumerate() {
                let s = entry
                    .as_str()
                    .ok_or_else(|| anyhow!("{}: layer_types[{}] is not a string", path.display(), i))?;
                let lt = match s {
                    "sliding_attention" => LayerType::Sliding,
                    "full_attention" => LayerType::Full,
                    other => {
                        return Err(anyhow!(
                            "{}: layer_types[{}] = {:?} not recognized (need sliding_attention | full_attention)",
                            path.display(),
                            i,
                            other
                        ))
                    }
                };
                types.push(lt);
            }
            if types.len() != cfg.num_hidden_layers {
                return Err(anyhow!(
                    "{}: layer_types length {} != num_hidden_layers {}",
                    path.display(),
                    types.len(),
                    cfg.num_hidden_layers
                ));
            }
            cfg.layer_types = types;
        }

        // RoPE parameters: dual-theta nested object.
        //
        //   text_config.rope_parameters = {
        //       "full_attention":    { "rope_theta": 1_000_000, "rope_type": "proportional", "partial_rotary_factor": 0.25 },
        //       "sliding_attention": { "rope_theta": 10_000,    "rope_type": "default" },
        //   }
        if let Some(rp) = tc.get("rope_parameters") {
            if let Some(theta) = rp
                .get("sliding_attention")
                .and_then(|s| s.get("rope_theta"))
                .and_then(as_f32)
            {
                cfg.rope_theta_sliding = theta;
            }
            if let Some(full) = rp.get("full_attention") {
                if let Some(theta) = full.get("rope_theta").and_then(as_f32) {
                    cfg.rope_theta_full = theta;
                }
                if let Some(prf) = full.get("partial_rotary_factor").and_then(as_f32) {
                    cfg.partial_rotary_factor_full = prf;
                }
            }
        }

        // Special tokens. bos_token_id and pad_token_id live under text_config.
        // eos_token_id at the ROOT carries the multi-id stop set (`[1, 106]`);
        // text_config.eos_token_id is a single id and is the "primary" eos.
        if let Some(b) = tc.get("bos_token_id").and_then(|v| v.as_u64()) {
            cfg.bos_token_id = b as u32;
        }
        if let Some(p) = tc.get("pad_token_id").and_then(|v| v.as_u64()) {
            cfg.pad_token_id = p as u32;
        }
        match root.get("eos_token_id") {
            Some(serde_json::Value::Array(arr)) => {
                let mut ids = Vec::with_capacity(arr.len());
                for v in arr {
                    if let Some(u) = v.as_u64() {
                        ids.push(u as u32);
                    }
                }
                if !ids.is_empty() {
                    cfg.eos_token_ids = ids;
                }
            }
            Some(serde_json::Value::Number(n)) => {
                if let Some(u) = n.as_u64() {
                    cfg.eos_token_ids = vec![u as u32];
                }
            }
            _ => {
                // fall back to text_config.eos_token_id if root absent.
                if let Some(u) = tc.get("eos_token_id").and_then(|v| v.as_u64()) {
                    cfg.eos_token_ids = vec![u as u32];
                }
            }
        }

        // dtype always BF16 for this inference port — the rule from CLAUDE.md
        // is "no F32 fallbacks in inference code".
        cfg.dtype = flame_core::DType::BF16;

        Ok(cfg)
    }
}
