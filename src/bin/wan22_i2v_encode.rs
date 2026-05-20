//! Wan2.2 I2V — Stage 1: UMT5-XXL text encoding (pure Rust).
//!
//! Loads UMT5-XXL, encodes positive + negative prompts, saves hidden states
//! to safetensors so `wan22_i2v_gen` can be run in pure-Rust mode without
//! the Python `scripts/wan22_i2v_encode.py` preprocess.
//!
//! Mirrors `chroma_encode.rs` (T5-XXL for Chroma); swapped to UMT5 since
//! Wan2.2 was trained on UMT5 (multilingual T5 variant). Same staged design —
//! load big text encoder, save embeds, drop, then `wan22_i2v_gen` loads
//! the embeds + image without holding UMT5 + Wan2.2 DiT simultaneously.
//!
//! ## Output schema (matches what `wan22_i2v_gen.rs:153-155` reads)
//!   `cond`           — [1, 512, 4096] BF16 (positive prompt)
//!   `uncond`         — [1, 512, 4096] BF16 (negative prompt)
//!   `cond_real_len`  — [1] F32 (real token count, pre-pad)
//!   `uncond_real_len`— [1] F32
//!
//! Optional companion: pass `--image PATH.png` to `wan22_i2v_gen` to encode
//! the image latent at gen-time. This binary does TEXT only.
//!
//! ## Usage
//! ```bash
//! ./target/release/wan22_i2v_encode "your prompt" "your negative prompt" \
//!     /path/to/embeddings.safetensors
//! ```
//! Then:
//! ```bash
//! ./target/release/wan22_i2v_gen --embeds /path/to/embeddings.safetensors \
//!     --image /path/to/input.png --output /path/to/out.mp4
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use flame_core::global_cuda_device;

use inference_flame::models::wan::Umt5Encoder;

const UMT5_PATH: &str = "/home/alex/.serenity/models/text_encoders/umt5_xxl_fp16.safetensors";
const UMT5_TOKENIZER: &str = "/home/alex/.serenity/models/upscalers/umt5-xxl/tokenizer.json";
const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/wan22_i2v_embeds.safetensors";

const UMT5_SEQ_LEN: usize = 512;

const DEFAULT_PROMPT: &str =
    "a beautiful young woman in a flowing white gown standing on a coastal cliff at sunset, cinematic";
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

    println!("=== Wan2.2 I2V — Stage 1 (UMT5-XXL encode) ===");
    println!("Prompt:    {:?}", prompt);
    println!("Negative:  {:?}", negative);
    println!("Output:    {}", out_path);
    println!();

    // ---- Load UMT5, encode cond + uncond, drop ----
    println!("--- Loading UMT5-XXL ---");
    let t0 = Instant::now();
    let mut umt5 = Umt5Encoder::load(Path::new(UMT5_PATH), &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    println!("--- Encoding cond ---");
    let t0 = Instant::now();
    let (cond_tokens, cond_real_len) = tokenize_with_len(&prompt);
    let cond_hidden = umt5.encode(&cond_tokens)?;
    println!(
        "  cond hidden: {:?} (real_len={}) in {:.1}s",
        cond_hidden.shape().dims(),
        cond_real_len,
        t0.elapsed().as_secs_f32()
    );

    println!("--- Encoding uncond ---");
    let t0 = Instant::now();
    let (uncond_tokens, uncond_real_len) = tokenize_with_len(&negative);
    let uncond_hidden = umt5.encode(&uncond_tokens)?;
    println!(
        "  uncond hidden: {:?} (real_len={}) in {:.1}s",
        uncond_hidden.shape().dims(),
        uncond_real_len,
        t0.elapsed().as_secs_f32()
    );

    drop(umt5);
    println!("  UMT5 weights evicted");
    println!();

    // ---- Save ----
    println!("--- Saving embeddings ---");
    let t0 = Instant::now();

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut tensors: HashMap<String, flame_core::Tensor> = HashMap::new();
    tensors.insert("cond".to_string(), cond_hidden);
    tensors.insert("uncond".to_string(), uncond_hidden);
    let cond_len_t = flame_core::Tensor::from_vec(
        vec![cond_real_len as f32],
        flame_core::Shape::from_dims(&[1]),
        device.clone(),
    )?;
    let uncond_len_t = flame_core::Tensor::from_vec(
        vec![uncond_real_len as f32],
        flame_core::Shape::from_dims(&[1]),
        device.clone(),
    )?;
    tensors.insert("cond_real_len".to_string(), cond_len_t);
    tensors.insert("uncond_real_len".to_string(), uncond_len_t);

    flame_core::serialization::save_file(&tensors, &out_path)?;
    println!(
        "  Saved {} tensors in {:.1}s",
        tensors.len(),
        t0.elapsed().as_secs_f32()
    );
    println!();

    println!("============================================================");
    println!("EMBEDDINGS SAVED: {}", out_path);
    println!("Total time:       {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    println!();
    println!(
        "Next: wan22_i2v_gen --embeds {} --image <input.png> --output <out.mp4>",
        out_path
    );

    let _ = device;
    Ok(())
}

/// UMT5 tokenize, pad to UMT5_SEQ_LEN with 0. Returns (tokens, real_len).
/// `real_len` = post-truncation, pre-pad count.
fn tokenize_with_len(prompt: &str) -> (Vec<i32>, usize) {
    match tokenizers::Tokenizer::from_file(UMT5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("umt5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(UMT5_SEQ_LEN);
            let real_len = ids.len();
            while ids.len() < UMT5_SEQ_LEN {
                ids.push(0);
            }
            (ids, real_len)
        }
        Err(e) => {
            eprintln!("[wan22_i2v_encode] UMT5 tokenizer load failed: {}", e);
            eprintln!("                   falling back to all-zero tokens — output will be GARBAGE");
            (vec![0i32; UMT5_SEQ_LEN], 0)
        }
    }
}
