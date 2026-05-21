//! Gemma-4 tokenizer — thin wrapper around HF `tokenizers` crate.
//!
//! Loads `tokenizer.json` directly. Gemma-4 uses a multilingual
//! SentencePiece BPE (vocab 262_144) — the `tokenizers` crate
//! handles the format natively, no custom code needed.
//!
//! Skip-special-tokens is HARD-FALSE during encode (we want the
//! chat-template's `<bos>`, `<|turn>`, etc. tokens) and HARD-TRUE
//! during decode (we want the model's emitted JSON without
//! `<turn|>` markers).
//!
//! ### Encode convention (AGENT-DEFAULT)
//!
//! `tokenizers::Tokenizer::encode(text, add_special_tokens=false)`.
//! The chat template renderer (`prompt_agent::render_chat_template`)
//! emits `<bos>` and the `<start_of_turn>` markers as literal strings —
//! we depend on the tokenizer recognizing them as single special tokens
//! during BPE merge. If we set `add_special_tokens=true`, the tokenizer
//! prepends ANOTHER `<bos>` on top of the one already in our string,
//! which would double-bos and confuse the decoder.

use std::path::Path;

use anyhow::{anyhow, Context};
use tokenizers::Tokenizer;

/// Tokenizer wrapper. The underlying `tokenizers::Tokenizer` does all
/// the heavy lifting; this struct exists so we can swap implementations
/// (e.g. for a future hand-rolled SentencePiece reader) without
/// touching call sites.
pub struct Gemma4Tokenizer {
    /// The HF tokenizers instance loaded from tokenizer.json.
    pub inner: TokenizerInner,
}

/// Opaque wrapper. Concrete impl uses `tokenizers::Tokenizer`. Kept
/// behind a struct so we can drop in a different backend without API
/// churn at call sites.
pub struct TokenizerInner {
    pub tokenizer: Tokenizer,
}

impl Gemma4Tokenizer {
    /// Load from a model dir's `tokenizer.json` (the format HF Hub ships).
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        // `tokenizers::Tokenizer::from_file` returns a `tokenizers::Result`
        // whose error type is `Box<dyn Error + Send + Sync>`. anyhow can't
        // From-convert it without an explicit map (the Send+Sync bounds
        // aren't carried), so we stringify.
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("Gemma4Tokenizer::from_file({}): {}", path.display(), e))
            .with_context(|| format!("loading {}", path.display()))?;
        Ok(Self {
            inner: TokenizerInner { tokenizer },
        })
    }

    /// Encode a chat-template-rendered prompt into token ids.
    /// `add_special_tokens=false` since we render `<bos>` etc. ourselves
    /// in `prompt_agent::render_chat_template` — otherwise we'd
    /// double-emit the BOS.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .tokenizer
            .encode(text, /*add_special_tokens=*/ false)
            .map_err(|e| anyhow!("Gemma4Tokenizer::encode: {}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a slice of token ids to text. `skip_special_tokens=true`
    /// so we drop `<turn|>`, `<eos>`, etc. that the model emits before
    /// (or interleaved with) the JSON output.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .tokenizer
            .decode(ids, /*skip_special_tokens=*/ true)
            .map_err(|e| anyhow!("Gemma4Tokenizer::decode: {}", e))
    }
}
