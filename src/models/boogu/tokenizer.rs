//! C7 — Boogu-Image T2I chat-template builder + tokenizer driver.
//!
//! Turns a T2I instruction string into the exact `input_ids` the Boogu pipeline
//! feeds the Qwen3-VL text encoder.
//!
//! Reuses the HF `tokenizers` crate (already a dependency) loading the Qwen3-VL
//! `tokenizer.json` from the mllm dir — NO BPE reimplementation. The only new
//! code here is the chat-template string assembly and a thin tokenize driver,
//! mirroring the verified Mojo C7b port (`boogu_tokenizer.mojo`, IDs
//! byte-identical to the oracle processor).
//!
//! ## Oracle conditioning (`pipeline_boogu.py`)
//!
//! `_apply_chat_template` builds `messages = [{role:system, content:SYS},
//! {role:user, content:instruction}]` then `processor.apply_chat_template(...,
//! tokenize=True)`. For a text-only T2I request (no images), the Qwen3-VL chat
//! template (`processor/chat_template.jinja`) with no tools and
//! `add_generation_prompt` unset renders exactly:
//!
//! ```text
//! <|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n
//! ```
//!
//! with **NO trailing `<|im_start|>assistant\n`** — this is an encoder, not a
//! generator. The specials `<|im_start|>`(151644) / `<|im_end|>`(151645) ship in
//! the tokenizer's `added_tokens` as `special=true`, so they encode atomically as
//! single ids even though we write them as literal text. The Qwen post-processor
//! adds no extra BOS/EOS, so we call `encode(text, add_special_tokens=false)`.
//!
//! ## System prompts (from `pipeline_boogu.py:231-235`)
//!
//! - **Conditional (positive) T2I** — no images, non-empty instruction →
//!   `SYSTEM_PROMPT_4_T2I` ([`BOOGU_SYSTEM_PROMPT_T2I`]).
//! - **CFG unconditional (negative)** — empty instruction `""` →
//!   `SYSTEM_PROMPT_DROP` ([`BOOGU_SYSTEM_PROMPT_DROP`]) with an empty user body.

use tokenizers::Tokenizer;

/// Conditional / positive T2I system prompt (`SYSTEM_PROMPT_4_T2I_UNIFIED`,
/// `pipeline_boogu.py:232`). Used for the positive instruction branch.
pub const BOOGU_SYSTEM_PROMPT_T2I: &str =
    "You are a helpful assistant that generates high-quality images based on user \
     instructions. The instructions are as follows.";

/// CFG unconditional / negative system prompt (`SYSTEM_PROMPT_DROP` =
/// `SYSTEM_PROMPT_4_TI2I_UNIFIED`, `pipeline_boogu.py:231,235`). Selected when the
/// (negative) instruction is empty — i.e. the default empty-negative CFG branch
/// (`pipeline_boogu.py:2491-2493` sets `negative_instruction = ""`).
pub const BOOGU_SYSTEM_PROMPT_DROP: &str =
    "Describe the key features of the input image (color, shape, size, texture, \
     objects, background), then explain how the user's text instruction should alter \
     or modify the image. Generate a new image that meets the user's requirements \
     while maintaining consistency with the original input where appropriate.";

/// Pad / BOS id for Qwen3 (`pad_token == bos == 151643`).
pub const BOOGU_PAD_ID: i32 = 151_643;

/// Build the exact Boogu T2I conditioning string for `system_prompt` + `instruction`.
///
/// Specials are written as their literal content; the tokenizer matches them
/// atomically because they are registered as special added tokens. No assistant
/// generation prompt is appended.
fn chat_template_with(system_prompt: &str, instruction: &str) -> String {
    format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n"
    )
}

/// Build the positive (conditional) T2I conditioning string for `instruction`.
///
/// Uses [`BOOGU_SYSTEM_PROMPT_T2I`]. Equivalent to the oracle's
/// `apply_chat_template([{system: SYS_T2I}, {user: instruction}])` for a
/// text-only request.
pub fn boogu_chat_template(instruction: &str) -> String {
    chat_template_with(BOOGU_SYSTEM_PROMPT_T2I, instruction)
}

/// Build the CFG **unconditional** conditioning string: the DROP system prompt
/// with an EMPTY user body. Mirrors `apply_chat_template([{system: SYS_DROP},
/// {user: ""}])`, which expands to:
///
/// ```text
/// <|im_start|>system\n{DROP}<|im_end|>\n<|im_start|>user\n<|im_end|>\n
/// ```
pub fn boogu_chat_template_uncond() -> String {
    chat_template_with(BOOGU_SYSTEM_PROMPT_DROP, "")
}

/// Tokenize an already-assembled chat-template `text` to Qwen ids.
///
/// `add_special_tokens=false` (the template carries every special literally and
/// the Qwen post-processor adds no extra BOS/EOS). Returns ids as `i32` for the
/// encoder.
pub fn boogu_tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>, String> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| format!("boogu tokenize failed: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

/// Tokenize a positive (conditional) T2I instruction → Qwen ids.
pub fn boogu_tokenize(tokenizer: &Tokenizer, instruction: &str) -> Result<Vec<i32>, String> {
    boogu_tokenize_text(tokenizer, &boogu_chat_template(instruction))
}

/// Tokenize the CFG unconditional (empty-negative) conditioning → Qwen ids.
pub fn boogu_tokenize_uncond(tokenizer: &Tokenizer) -> Result<Vec<i32>, String> {
    boogu_tokenize_text(tokenizer, &boogu_chat_template_uncond())
}

/// Load the Qwen3-VL tokenizer from `mllm_dir/tokenizer.json`.
///
/// The mllm dir's `tokenizer.json` is the text tokenizer paired with the language
/// tower (the diffusers `processor/` dir holds a parallel copy with image/video
/// special tokens; for the text-only T2I path both produce identical ids on text
/// content). Mirrors the `klein9b_encode` idiom.
pub fn load_boogu_tokenizer(mllm_dir: impl AsRef<std::path::Path>) -> Result<Tokenizer, String> {
    let path = mllm_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(&path)
        .map_err(|e| format!("boogu: failed to load tokenizer {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_exact_string() {
        let got = boogu_chat_template("a red cube on grass");
        let want = "<|im_start|>system\nYou are a helpful assistant that generates \
high-quality images based on user instructions. The instructions are as follows.\
<|im_end|>\n<|im_start|>user\na red cube on grass<|im_end|>\n";
        assert_eq!(got, want);
    }

    #[test]
    fn chat_template_uncond_exact_string() {
        let got = boogu_chat_template_uncond();
        let want = "<|im_start|>system\nDescribe the key features of the input image \
(color, shape, size, texture, objects, background), then explain how the user's text \
instruction should alter or modify the image. Generate a new image that meets the \
user's requirements while maintaining consistency with the original input where \
appropriate.<|im_end|>\n<|im_start|>user\n<|im_end|>\n";
        assert_eq!(got, want);
    }

    #[test]
    fn no_assistant_generation_suffix() {
        // Encoder, not generator: the rendered string must NOT end with an
        // assistant generation prompt.
        let s = boogu_chat_template("anything");
        assert!(!s.contains("<|im_start|>assistant"));
        assert!(s.ends_with("<|im_end|>\n"));
    }

    #[test]
    fn template_structure_markers_present_and_ordered() {
        let s = boogu_chat_template("X");
        let sys_at = s.find("<|im_start|>system\n").expect("system marker");
        let user_at = s.find("<|im_start|>user\n").expect("user marker");
        assert!(sys_at < user_at, "system must precede user");
        // exactly two <|im_end|> closers (system, then user)
        assert_eq!(s.matches("<|im_end|>").count(), 2);
    }

    #[test]
    fn system_prompts_match_pipeline_source() {
        // Pin the exact source strings (pipeline_boogu.py:231-232). These are the
        // single most parity-critical literals in C7.
        assert_eq!(
            BOOGU_SYSTEM_PROMPT_T2I,
            "You are a helpful assistant that generates high-quality images based on \
             user instructions. The instructions are as follows."
        );
        assert!(BOOGU_SYSTEM_PROMPT_DROP.starts_with("Describe the key features"));
        assert!(BOOGU_SYSTEM_PROMPT_DROP.ends_with("where appropriate."));
    }

    /// Round-trip / id-shape test against the real Qwen3-VL tokenizer in the
    /// checkpoint. Marked `#[ignore]` because it depends on the local mllm dir
    /// (38.5 GB checkpoint); run with `cargo test -- --ignored` when present.
    #[test]
    #[ignore = "requires local Boogu mllm tokenizer.json"]
    fn tokenize_shapes_and_specials() {
        let mllm = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm";
        let tok = load_boogu_tokenizer(mllm).expect("load tokenizer");
        let ids = boogu_tokenize(&tok, "a small dog").expect("tokenize");
        assert!(!ids.is_empty(), "ids must be non-empty");
        // <|im_start|>=151644 and <|im_end|>=151645 must each appear (specials
        // encoded atomically). At least 2 of each (system+user open/close).
        let im_start = ids.iter().filter(|&&i| i == 151_644).count();
        let im_end = ids.iter().filter(|&&i| i == 151_645).count();
        assert_eq!(im_start, 2, "expected two <|im_start|> ids, got {im_start}");
        assert_eq!(im_end, 2, "expected two <|im_end|> ids, got {im_end}");
        // No accidental BOS injection (add_special_tokens=false).
        assert_ne!(ids[0], BOOGU_PAD_ID, "no BOS/pad should lead the sequence");
    }
}
