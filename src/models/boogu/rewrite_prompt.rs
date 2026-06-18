//! C9 — Boogu T2I rewrite chat-prompt builder.
//!
//! Byte-faithful Rust port of `utils/t2i_external_prompt_rewriter.py`
//! (`build_messages(prompt, lang="en")` + the Qwen3-VL chat template with
//! `add_generation_prompt=True`).
//!
//! The oracle puts the **entire** EN rewrite system prompt INTO a single *user*
//! message (there is NO system role), then `apply_chat_template(...,
//! add_generation_prompt=True)` wraps it as:
//!
//! ```text
//! <|im_start|>user\n{SYS}{SUFFIX(idea)}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! The trailing `<|im_start|>assistant\n` is what makes the model GENERATE the
//! rewrite (unlike the C7 encoder template, which has NO assistant prompt).
//!
//! The exact `T2I_REWRITE_SYSTEM_PROMPT_EN` and `_SUFFIX["en"]` strings are
//! embedded verbatim from the oracle via `include_str!` (the system prompt is
//! ~16 KB of carefully-worded rules + examples with quotes/`$`/backslashes;
//! transcribing it by hand would risk a single-char parity break).

/// The exact EN rewrite system prompt (`T2I_REWRITE_SYSTEM_PROMPT_EN`),
/// byte-identical to the oracle (verified via `diff`). Begins with a leading
/// `\n` and ends with `...without any extra reply.\n`.
pub const T2I_REWRITE_SYSTEM_PROMPT_EN: &str =
    include_str!("t2i_rewrite_system_prompt_en.txt");

/// The exact EN suffix (`_SUFFIX["en"]`) with a `{prompt}` placeholder, e.g.
/// `"\nOriginal prompt: {prompt}\n(Make sure ...) Output the rewritten prompt
/// directly:"`.
pub const T2I_REWRITE_SUFFIX_EN: &str = include_str!("t2i_rewrite_suffix_en.txt");

/// Build the user-message body: `SYS + SUFFIX.format(prompt=idea)`.
///
/// Mirrors `build_messages`'s `SYSTEM_PROMPTS["en"] + _SUFFIX["en"].format(...)`.
pub fn build_user_content(idea: &str) -> String {
    let suffix = T2I_REWRITE_SUFFIX_EN.replace("{prompt}", idea);
    format!("{T2I_REWRITE_SYSTEM_PROMPT_EN}{suffix}")
}

/// Build the full chat prompt string that the model decodes from — INCLUDING
/// the `<|im_start|>assistant\n` generation prompt. Specials are written as
/// their literal content; the Qwen tokenizer matches them atomically because
/// they are registered special added tokens.
///
/// Equals the oracle's `apply_chat_template(build_messages(idea, "en"),
/// add_generation_prompt=True, tokenize=False)`.
pub fn build_rewrite_prompt(idea: &str) -> String {
    let user = build_user_content(idea);
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_embedded_and_bounded() {
        // Sanity: the include_str! asset is present and is the expected EN text.
        assert!(T2I_REWRITE_SYSTEM_PROMPT_EN.starts_with("\nYou are a prompt optimizer."));
        assert!(T2I_REWRITE_SYSTEM_PROMPT_EN
            .trim_end()
            .ends_with("Rewrite the prompt directly, without any extra reply."));
        // ~16 KB of rules + examples.
        assert!(T2I_REWRITE_SYSTEM_PROMPT_EN.len() > 15_000);
    }

    #[test]
    fn suffix_has_placeholder_and_tail() {
        assert!(T2I_REWRITE_SUFFIX_EN.contains("{prompt}"));
        assert!(T2I_REWRITE_SUFFIX_EN
            .trim_end()
            .ends_with("Output the rewritten prompt directly:"));
        assert!(T2I_REWRITE_SUFFIX_EN.starts_with("\nOriginal prompt: "));
    }

    #[test]
    fn prompt_wraps_with_assistant_generation_prompt() {
        let p = build_rewrite_prompt("a cat in a hat");
        assert!(p.starts_with("<|im_start|>user\n"));
        // The idea is substituted into the suffix.
        assert!(p.contains("Original prompt: a cat in a hat"));
        // CRITICAL: ends with the assistant generation prompt (=> GENERATE).
        assert!(p.ends_with("<|im_end|>\n<|im_start|>assistant\n"));
        // exactly one user open, one im_end before assistant.
        assert_eq!(p.matches("<|im_start|>user").count(), 1);
        assert!(!p.contains("<|im_start|>system"));
    }
}
