//! Chroma — Stage 1: T5-XXL text encoding only.
//!
//! Loads T5-XXL, encodes the positive + negative prompts, saves both
//! hidden-state tensors to a single safetensors file, and exits. Nothing else
//! is loaded — no DiT, no VAE — so the GPU only ever holds T5 and the two
//! resulting [1, 512, 4096] BF16 tensors.
//!
//! ## Why a separate binary
//! Chroma requires CFG, so each step needs both `t5_cond` and `t5_uncond`.
//! Loading T5 (~10 GB) alongside the Chroma DiT (~17 GB) on a 24 GB card
//! risks OOM. Splitting Stage 1 (encode) from Stage 2 (gen) keeps each stage
//! within budget.
//!
//! ## Usage
//! ```bash
//! ./target/release/chroma_encode "your prompt" "your negative prompt" \
//!     /path/to/embeddings.safetensors
//! ```
//! Then run `chroma_gen` with the same path:
//! ```bash
//! ./target/release/chroma_gen /path/to/embeddings.safetensors output.png
//! ```
//!
//! Output safetensors keys:
//!   `t5_cond`   — [1, 512, 4096] BF16
//!   `t5_uncond` — [1, 512, 4096] BF16

use std::collections::HashMap;
use std::time::Instant;

use flame_core::global_cuda_device;

use inference_flame::models::t5_encoder::T5Encoder;

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";
const DEFAULT_OUTPUT: &str = "/home/alex/serenity/output/chroma_embeddings.safetensors";

const T5_SEQ_LEN: usize = 512;

const DEFAULT_PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";
const DEFAULT_NEGATIVE: &str = "";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let negative = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUTPUT.to_string());

    println!("=== Chroma — Stage 1 (T5-XXL encode) ===");
    println!("Prompt:    {:?}", prompt);
    println!("Negative:  {:?}", negative);
    println!("Output:    {}", out_path);
    println!();

    // ------------------------------------------------------------------
    // Load T5, encode cond + uncond, drop T5
    // ------------------------------------------------------------------
    println!("--- Loading T5-XXL ---");
    let t0 = Instant::now();
    let mut t5 = T5Encoder::load(T5_PATH, &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    println!("--- Encoding cond ---");
    let t0 = Instant::now();
    let cond_tokens = tokenize_t5(&prompt);
    let cond_hidden = t5.encode(&cond_tokens)?;
    println!("  cond hidden: {:?} in {:.1}s",
        cond_hidden.shape().dims(), t0.elapsed().as_secs_f32());

    println!("--- Encoding uncond ---");
    let t0 = Instant::now();
    let uncond_tokens = tokenize_t5(&negative);
    let uncond_hidden = t5.encode(&uncond_tokens)?;
    println!("  uncond hidden: {:?} in {:.1}s",
        uncond_hidden.shape().dims(), t0.elapsed().as_secs_f32());

    // Drop T5 explicitly so the FlameSwap-backed weights free before save.
    drop(t5);
    println!("  T5 weights evicted");
    println!();

    // ------------------------------------------------------------------
    // Save to safetensors
    // ------------------------------------------------------------------
    println!("--- Saving embeddings ---");
    let t0 = Instant::now();

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut tensors: HashMap<String, flame_core::Tensor> = HashMap::new();
    tensors.insert("t5_cond".to_string(), cond_hidden);
    tensors.insert("t5_uncond".to_string(), uncond_hidden);

    flame_core::serialization::save_file(&tensors, &out_path)?;
    println!("  Saved {} tensors in {:.1}s", tensors.len(), t0.elapsed().as_secs_f32());
    println!();

    println!("============================================================");
    println!("EMBEDDINGS SAVED: {}", out_path);
    println!("Total time:       {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    println!();
    println!("Next: run `chroma_gen {} <output.png>`", out_path);

    let _ = device;
    Ok(())
}

/// T5-XXL tokenize, pad to T5_SEQ_LEN with 0 (T5 pad token).
fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            eprintln!("[chroma_encode] T5 tokenizer load failed: {}", e);
            eprintln!("                 falling back to all-zero tokens — output will be GARBAGE");
            vec![0i32; T5_SEQ_LEN]
        }
    }
}
