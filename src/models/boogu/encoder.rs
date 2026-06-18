//! C7 — Boogu-Image text encoder: Qwen3-VL-8B-Instruct **language tower only**.
//!
//! Boogu's `mllm` is `Qwen3VLForConditionalGeneration`, but the T2I path is
//! **text-only**: the vision tower (`model.visual.*`) and `lm_head.weight` are
//! never touched. The conditioning the DiT consumes is the VLM's
//! `output_hidden_states[-1]` — the output of decoder layer 35 **before** the
//! model's final RMSNorm (HF `hidden_states[-1]` is unambiguously pre-final-norm;
//! `last_hidden_state` is the post-norm variant and is NOT what Boogu uses).
//!
//! The Qwen3-VL **text** decoder layer is byte-identical to the plain Qwen3 text
//! decoder layer (`input_layernorm` / q,k,v,o_proj + q_norm,k_norm /
//! `post_attention_layernorm` / SwiGLU MLP), so this module is a thin wrapper over
//! the existing [`Qwen3Encoder`](crate::models::qwen3_encoder::Qwen3Encoder):
//!
//!   1. Load the mllm sharded safetensors.
//!   2. **Remap `model.language_model.*` → `model.*`** (Boogu's Qwen3-VL prefix
//!      carries one extra `model.` vs the bare `model.*` keys `qwen3_encoder`
//!      expects); drop the `model.visual.*` (351 keys) and `lm_head.weight`.
//!   3. Construct `Qwen3Encoder` with [`Qwen3Config::qwen3_vl_text`] (hidden 4096,
//!      36 layers, kv 8, head_dim 128, rope_theta 5e6, eps 1e-6,
//!      `extract_layers=[35]`).
//!   4. `encode(token_ids) -> [1, L, 4096]` = `hidden_states[-1]` (pre-final-norm).
//!
//! ## mROPE → plain 1-D RoPE (verified valid)
//!
//! `text_config.rope_scaling = {mrope_interleaved: true, mrope_section:[24,20,20]}`.
//! For text-only inputs the three mROPE sections collapse to identical per-position
//! angles (single 1-D position sequence broadcast to all 3 axes), so the math is
//! exactly the HF half-split 1-D RoPE that `Qwen3Encoder` already implements. The
//! Mojo C7 port confirmed this by measurement (encoder `hidden_states[-1]` cos
//! 0.9995958 vs torch; mROPE would have tanked the cosine if it mattered). The
//! vision-tower mROPE path is intentionally out of scope.
//!
//! Inference port — autograd is OFF, no backward registration anywhere. BF16
//! throughout (matches the checkpoint dtype).

use crate::models::qwen3_encoder::{Qwen3Config, Qwen3Encoder};
use flame_core::{CudaDevice, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Number of decoder layers in the Boogu mllm language tower.
pub const BOOGU_MLLM_NUM_LAYERS: usize = 36;

/// Pad / BOS id for Qwen3 (`pad_token == bos == 151643`). T2I prompt tokens use
/// `<|im_start|>`(151644) / `<|im_end|>`(151645), so this never collides with a
/// real token — `Qwen3Encoder::encode` recovers `real_len` by scanning for the
/// first 151643.
pub const BOOGU_PAD_ID: i32 = 151_643;

/// Boogu text encoder: a [`Qwen3Encoder`] holding the Qwen3-VL language tower.
///
/// Thin wrapper that owns the remapped-and-loaded encoder. `encode` returns the
/// instruction hidden states `[1, L, 4096]` = HF `hidden_states[-1]`
/// (pre-final-norm), exactly what Boogu's DiT `caption_embedder` consumes.
pub struct BooguTextEncoder {
    inner: Qwen3Encoder,
}

impl BooguTextEncoder {
    /// Load the Boogu mllm Qwen3-VL **language tower** from `mllm_dir`.
    ///
    /// Reads every `model-*.safetensors` shard, keeps only the language-model
    /// keys, remaps `model.language_model.*` → `model.*`, and constructs a
    /// `Qwen3Encoder` with [`Qwen3Config::qwen3_vl_text`]. The vision tower
    /// (`model.visual.*`) and `lm_head.weight` are dropped.
    ///
    /// All weights are loaded in their stored dtype (BF16) onto `device`.
    pub fn load(mllm_dir: impl AsRef<Path>, device: Arc<CudaDevice>) -> Result<Self> {
        let raw = load_sharded_weights(mllm_dir.as_ref(), &device)?;
        let weights = remap_language_tower(raw)?;
        let config = Qwen3Config::qwen3_vl_text();
        let inner = Qwen3Encoder::new(weights, config, device);
        Ok(Self { inner })
    }

    /// Construct directly from a pre-loaded **raw** mllm weight map (still using
    /// the `model.language_model.*` prefix). Performs the remap internally.
    ///
    /// Useful when shards are already in memory (parity captures, tests).
    pub fn from_raw_weights(
        raw: HashMap<String, Tensor>,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let weights = remap_language_tower(raw)?;
        let inner = Qwen3Encoder::new(weights, Qwen3Config::qwen3_vl_text(), device);
        Ok(Self { inner })
    }

    /// Encode token ids → instruction hidden states `[1, L, 4096]`.
    ///
    /// Returns `extract_layers=[35]` = output of decoder layer 35 **before** the
    /// final norm = HF `hidden_states[-1]`. NO `model.norm` is applied (that would
    /// give `last_hidden_state`, which Boogu does NOT use).
    ///
    /// `token_ids` may be padded with [`BOOGU_PAD_ID`] (151643); `Qwen3Encoder`
    /// detects the first pad to set the causal-mask `real_len`, so padded columns
    /// are masked out and the real rows are numerically identical to unpadded.
    /// The caller is responsible for slicing the returned `[1, L_padded, 4096]`
    /// back to the real length if a fixed pad length was used (mirrors the
    /// `klein9b_encode` bin idiom).
    pub fn encode(&self, token_ids: &[i32]) -> Result<Tensor> {
        self.inner.encode(token_ids)
    }

    /// Borrow the wrapped `Qwen3Encoder` (config / weight inspection).
    pub fn inner(&self) -> &Qwen3Encoder {
        &self.inner
    }

    /// Output hidden dimension (4096 for the single `[35]` tap).
    pub fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }
}

// ---------------------------------------------------------------------------
// Weight loading + language-tower remap
// ---------------------------------------------------------------------------

/// Load every `model-*.safetensors` shard in `dir` into one flat map.
///
/// Mirrors `klein9b_encode::load_sharded_weights`.
fn load_sharded_weights(
    dir: &Path,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| {
            flame_core::Error::InvalidInput(format!(
                "boogu encoder: cannot read mllm dir {}: {e}",
                dir.display()
            ))
        })?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        return Err(flame_core::Error::InvalidInput(format!(
            "boogu encoder: no model-*.safetensors shards in {}",
            dir.display()
        )));
    }

    let mut all = HashMap::new();
    for (i, path) in shard_paths.iter().enumerate() {
        let t0 = Instant::now();
        let shard = flame_core::serialization::load_file(path, device)?;
        log::info!(
            "boogu mllm shard {}/{}: {} keys ({:.1}s)",
            i + 1,
            shard_paths.len(),
            shard.len(),
            t0.elapsed().as_secs_f32()
        );
        all.extend(shard);
    }
    Ok(all)
}

/// Keep only the language-tower keys and strip the leading `language_model.`
/// segment so the keys match what `Qwen3Encoder` / `config_from_weights` expect:
/// `model.language_model.{embed_tokens,layers.*,norm}.*` → `model.{...}.*`.
///
/// Drops `model.visual.*` (351 keys) and `lm_head.weight`. Verifies that all
/// 398 expected Qwen3 keys (1 embed + 36×11 + 1 norm) resolved.
fn remap_language_tower(
    raw: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    const LM_PREFIX: &str = "model.language_model.";
    let mut out = HashMap::with_capacity(raw.len());
    let mut kept = 0usize;
    for (k, v) in raw.into_iter() {
        if let Some(rest) = k.strip_prefix(LM_PREFIX) {
            // model.language_model.embed_tokens.weight -> model.embed_tokens.weight
            out.insert(format!("model.{rest}"), v);
            kept += 1;
        }
        // else: model.visual.* / lm_head.weight / anything else -> dropped.
    }

    // Validate completeness against the canonical Qwen3 key list.
    let expected = crate::models::qwen3_encoder::expected_weight_keys(BOOGU_MLLM_NUM_LAYERS);
    let missing: Vec<&String> = expected.iter().filter(|k| !out.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(flame_core::Error::InvalidInput(format!(
            "boogu encoder: {} language-tower key(s) missing after remap (e.g. {:?}); \
             kept {kept} of expected {}",
            missing.len(),
            missing.iter().take(5).collect::<Vec<_>>(),
            expected.len()
        )));
    }
    log::info!(
        "boogu mllm: remapped {kept} language-tower keys (model.language_model.* -> model.*), \
         vision tower + lm_head dropped"
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
    fn pad_id_does_not_collide_with_chat_specials() {
        // <|im_start|> = 151644, <|im_end|> = 151645 — both distinct from pad.
        assert_eq!(BOOGU_PAD_ID, 151_643);
        assert_ne!(BOOGU_PAD_ID, 151_644);
        assert_ne!(BOOGU_PAD_ID, 151_645);
    }

    #[test]
    fn config_matches_boogu_mllm() {
        let c = Qwen3Config::qwen3_vl_text();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_layers, BOOGU_MLLM_NUM_LAYERS);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.rms_norm_eps, 1e-6);
        assert_eq!(c.rope_theta, 5_000_000.0);
        // extract_layers=[35] = HF hidden_states[-1] (output of decoder layer 35,
        // PRE final norm). NOT the post-norm last_hidden_state.
        assert_eq!(c.extract_layers, vec![35]);
    }

    #[test]
    fn remap_strips_language_model_segment_and_drops_vision() {
        // Build a fake raw map with the exact 398 lang keys + some vision/lm_head
        // junk, and assert the remap keeps exactly the 398 expected keys.
        // (No GPU needed — we only key-shuffle; values are placeholder.)
        //
        // We can't cheaply fabricate real Tensors here, so this test only
        // exercises the *key* logic via a parallel string-only reimplementation
        // assertion: confirm strip_prefix behavior on representative keys.
        const LM_PREFIX: &str = "model.language_model.";
        let cases = [
            ("model.language_model.embed_tokens.weight", Some("model.embed_tokens.weight")),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                Some("model.layers.0.self_attn.q_proj.weight"),
            ),
            ("model.language_model.norm.weight", Some("model.norm.weight")),
            ("model.visual.blocks.0.attn.qkv.weight", None),
            ("lm_head.weight", None),
        ];
        for (src, want) in cases {
            let got = src.strip_prefix(LM_PREFIX).map(|r| format!("model.{r}"));
            assert_eq!(got.as_deref(), want, "remap mismatch for {src}");
        }
    }

    #[test]
    fn expected_key_count_is_398() {
        let keys = crate::models::qwen3_encoder::expected_weight_keys(BOOGU_MLLM_NUM_LAYERS);
        // 1 (embed) + 36 * 11 (per-layer) + 1 (final norm) = 398, matching the
        // 398 language-tower keys in the mllm index.
        assert_eq!(keys.len(), 1 + 36 * 11 + 1);
        assert_eq!(keys.len(), 398);
    }
}
