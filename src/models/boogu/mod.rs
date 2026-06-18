//! Boogu-Image-0.1-Base — pure-Rust INFERENCE port (foundational chunk).
//!
//! Boogu is a Flux/Qwen-Image-class flow-matching T2I DiT (Lumina2/OmniGen2
//! lineage, 38.5 GB bf16). Two-stage transformer: **8 double-stream**
//! joint-attention blocks then **32 single-stream** blocks on the fused
//! `[instruct ; img]` sequence, plus **2 context-refiner** + **2 noise-refiner**
//! blocks; velocity prediction, Euler flow-matching, the single transformer run
//! twice for CFG. bf16 throughout — NO quantization.
//!
//! Reference (parity oracle): the Python `boogu` package at
//! `/home/alex/Boogu-Image` (installed editable in
//! `/home/alex/serenityflow-v2/.venv`). Architecture cross-check: the verified
//! Mojo port handoff
//! (`/home/alex/mojodiffusion/serenitymojo/docs/BOOGU_PORT_HANDOFF.md`).
//!
//! ## This chunk (build-order steps 1-2 + C1, C2)
//!
//! 1. [`config`] — [`BooguConfig`] architecture constants (all of `config.json`).
//! 2. [`loader`] — sharded-safetensors → BF16 weight map (mirrors
//!    `ideogram4/loader.rs`, minus the FP8 path Boogu doesn't need).
//! 3. **C1** [`embed`] — timestep embedder + caption embedder
//!    (`Lumina2CombinedTimestepCaptionEmbedding`).
//! 4. **C2** [`rope`] — 3-axis interleaved-complex RoPE table builder + apply
//!    (reuses `flame_core::bf16_ops::rope_fused_bf16`, the interleaved kernel).
//! 5. **C3** [`attention`] — GQA self-attention helper (28q/7kv, repeat_kv ×4,
//!    QK-RMSNorm, interleaved RoPE, pad head_dim 120→128 → SDPA scale 1/√120 →
//!    slice back) for the single-stream / refiner block.
//! 6. **C3** [`block`] — [`BooguBlock`]: the single-stream / refiner
//!    `BooguImageTransformerBlock` (modulation True = single_stream/noise_refiner,
//!    False = context_refiner). LuminaRMSNormZero modulation, SwiGLU FFN,
//!    tanh-gated dual residuals.
//! 7. **C4** [`block_double`] — [`BooguDoubleStreamBlock`]: the double-stream
//!    `BooguImageDoubleStreamTransformerBlock` (the 8 `double_stream_layers`).
//!    Separate per-stream q/k/v joint cross-attn over `[instruct ; img]`
//!    (instruct-first) + img self-attn; 3 img + 2 instruct LuminaRMSNormZero;
//!    tanh-gated dual residuals + `shift_mlp`. Joint attn lives in
//!    [`attention::joint_gqa_attention`] (shared SDPA core with C3).
//!
//! 10. **C7** [`encoder`] — [`BooguTextEncoder`]: the Qwen3-VL-8B language tower
//!     (text-only T2I), wrapping `qwen3_encoder::Qwen3Encoder` with the
//!     `model.language_model.*` → `model.*` remap and `extract_layers=[35]`
//!     (= HF `hidden_states[-1]`, pre-final-norm). [`tokenizer`] — the T2I
//!     chat-template builder + HF-tokenizer driver (cond + CFG-uncond).
//!
//! 8. **C5** [`final_layer`] — `norm_out` (`LuminaLayerNormContinuous`,
//!    affine-free LayerNorm eps 1e-6, temb-modulated) + channel-FASTEST
//!    unpatchify `(p1 p2 c)` (reuses `flame_core::bf16_elementwise::unpatchify_bf16`,
//!    which IS the channel-fastest variant — verified bit-exact round-trip).
//! 9. **C6** [`transformer`] — [`BooguDiT`]: the full DiT forward, resident.
//!    Wires C1–C5: patchify → embed → ctx_refiner×2(modF) → x_embed+noise_refiner×2(modT)
//!    → 8 double → instruct-first fuse → 32 single → norm_out → extract[cap_len:] → unpatch.
//!
//! NOT built here: scheduler/VAE/bins (C8), rewriter (C9).
//! Inference port — autograd is OFF, no backward registration anywhere.

pub mod attention;
pub mod block;
pub mod block_double;
pub mod config;
pub mod decode_1024;
pub mod embed;
pub mod encoder;
pub mod final_layer;
pub mod generate;
pub mod loader;
pub mod rewrite_prompt;
pub mod rope;
pub mod tokenizer;
pub mod transformer;

pub use block::BooguBlock;
pub use block_double::BooguDoubleStreamBlock;
pub use config::BooguConfig;
pub use encoder::BooguTextEncoder;
pub use generate::BooguRewriter;
pub use rewrite_prompt::{build_rewrite_prompt, build_user_content, T2I_REWRITE_SYSTEM_PROMPT_EN};
pub use tokenizer::{
    boogu_chat_template, boogu_chat_template_uncond, boogu_tokenize, boogu_tokenize_uncond,
    load_boogu_tokenizer, BOOGU_SYSTEM_PROMPT_DROP, BOOGU_SYSTEM_PROMPT_T2I,
};
pub use transformer::BooguDiT;
