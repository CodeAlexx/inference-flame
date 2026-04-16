//! Motif-Video Stage 1: T5Gemma2 text encoding.
//!
//! Tokenizes with GemmaTokenizer, runs T5Gemma2 encoder on prompt + optional
//! negative prompt, saves embeddings to safetensors.

use std::collections::HashMap;
use std::time::Instant;
use flame_core::{global_cuda_device, Tensor};
use inference_flame::models::t5gemma2_encoder::{T5Gemma2Config, T5Gemma2Encoder};

const BASE_DIR: &str = "/home/alex/.serenity/models/checkpoints/motif-video-2b";
const DEFAULT_TOKENIZER: &str =
    "/home/alex/.serenity/models/checkpoints/motif-video-2b/tokenizer/tokenizer.json";
const DEFAULT_OUTPUT: &str = "/home/alex/serenity/output/motif_embeds.safetensors";
const MAX_SEQ_LEN: usize = 512;  // adjust based on T5Gemma2 defaults
const PAD_ID: i32 = 0; // Gemma pad token

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_default();
    let negative = args.get(2).cloned().unwrap_or_default();
    let out_path = args.get(3).cloned().unwrap_or_else(|| DEFAULT_OUTPUT.to_string());

    if prompt.is_empty() {
        eprintln!("Usage: motif_encode <prompt> [negative] [output_path]");
        std::process::exit(1);
    }

    println!("=== Motif-Video Stage 1 (T5Gemma2 encode) ===");
    println!("  Prompt:   {:?}", prompt);
    println!("  Negative: {:?}", negative);
    println!("  Output:   {}", out_path);

    let device = global_cuda_device();

    // Tokenizer
    let tokenizer_path = std::env::var("MOTIF_TOKENIZER")
        .unwrap_or_else(|_| DEFAULT_TOKENIZER.to_string());
    println!("--- Loading tokenizer ---");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {}", e))?;

    // tokenizer.json for this model has `padding.strategy = Fixed 512` baked in,
    // so `tokenizer.encode()` auto-pads every prompt to 512. `enc.get_ids()`
    // returns 512 entries where only the prefix is real tokens and the rest
    // are `pad_id=0`. The true real-token count is `count(id != PAD_ID)`, NOT
    // `ids.len()` — previous code conflated them and handed the encoder a mask
    // of all-ones over all 512 positions. Under Motif's bidirectional attention
    // that let real tokens attend to 494 pad positions, diluted attention
    // scores, and produced embeddings with cos≈0.80 vs Python (vs near-1.0
    // expected). Downstream: CFG amplified the gap, DiT denoised to abstract
    // blur instead of the intended subject. Build a proper mask and call
    // `encode_with_mask`.
    // Match Python pipeline `_get_default_embeds`: `padding='max_length',
    // max_length=512, truncation=True`. Keep the 512-token shape the DiT was
    // trained with; the encoder masks pads out of attention, so the real-vs-
    // pad distinction is made via `attention_mask`, not by shortening the
    // sequence. Truncating the sequence to `real_len` here caused the DiT's
    // cross-attention to see a very short context (`text_seq_length=24`
    // instead of 512) and produce washed-out output — the model has a strong
    // prior on the padded 512-length text vector.
    let tokenize = |text: &str| -> anyhow::Result<(Vec<i32>, Vec<i32>, usize)> {
        let enc = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&id| id as i32).collect();
        ids.truncate(MAX_SEQ_LEN);
        ids.resize(MAX_SEQ_LEN, PAD_ID); // pad to exactly 512
        let mask: Vec<i32> = ids.iter().map(|&id| if id == PAD_ID { 0 } else { 1 }).collect();
        let real_len = mask.iter().filter(|&&v| v == 1).count();
        Ok((ids, mask, real_len))
    };

    let (cond_ids, cond_mask, cond_len) = tokenize(&prompt)?;
    let (uncond_ids, uncond_mask, uncond_len) = tokenize(&negative)?;
    println!("  Cond: {} real / {} padded (mask sum = {})",
             cond_len, cond_ids.len(), cond_mask.iter().sum::<i32>());
    println!("  Uncond: {} real / {} padded (mask sum = {})",
             uncond_len, uncond_ids.len(), uncond_mask.iter().sum::<i32>());

    // Load encoder (all 4 shards... wait, text_encoder is a single file)
    let encoder_path = format!("{}/text_encoder/model.safetensors", BASE_DIR);
    println!("--- Loading T5Gemma2 encoder from {} ---", encoder_path);
    let t0 = Instant::now();
    let weights = flame_core::serialization::load_file(
        std::path::Path::new(&encoder_path),
        &device,
    )?;
    // Keys ship with `encoder.` prefix — T5Gemma2Encoder expects them as-is.
    println!("  Loaded {} keys in {:.1}s", weights.len(), t0.elapsed().as_secs_f32());

    let config = T5Gemma2Config::default();
    println!(
        "  Config: hidden={}, layers={}, heads={}, kv={}, head_dim={}",
        config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads, config.head_dim
    );
    let encoder = T5Gemma2Encoder::new(weights, config, device.clone());

    // Encode with explicit pad mask so the encoder's bidirectional attention
    // masks pad tokens out of the score computation (matching Python).
    println!("--- Encoding cond ---");
    let t0 = Instant::now();
    let cond_hidden = encoder.encode_with_mask(&cond_ids, &cond_mask)?;
    println!("  cond: {:?} in {:.1}s", cond_hidden.shape().dims(), t0.elapsed().as_secs_f32());

    println!("--- Encoding uncond ---");
    let t0 = Instant::now();
    let uncond_hidden = encoder.encode_with_mask(&uncond_ids, &uncond_mask)?;
    println!("  uncond: {:?} in {:.1}s", uncond_hidden.shape().dims(), t0.elapsed().as_secs_f32());

    drop(encoder);

    // Slice to real token prefix before saving. The Rust DiT's cross-attention
    // has no attention-mask support (line 1198 in motif_video_dit.rs uses
    // `sdpa(..., None)`), so any pad positions in the embedding would pollute
    // cross-attention softmax weights across 494 zero-mask positions. Python's
    // DiT applies the mask; slicing here is the shape-level equivalent. Also
    // save `cond_real_len` / `uncond_real_len` metadata in case a consumer
    // wants to reconstruct the original [1, 512] tensor with pads zeroed.
    let cond_sliced = cond_hidden.narrow(1, 0, cond_len)?;
    let uncond_sliced = uncond_hidden.narrow(1, 0, uncond_len)?;
    drop(cond_hidden);
    drop(uncond_hidden);

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("cond".into(), cond_sliced);
    tensors.insert("uncond".into(), uncond_sliced);
    tensors.insert(
        "cond_real_len".into(),
        Tensor::from_vec(vec![cond_len as f32], flame_core::Shape::from_dims(&[1]), device.clone())?,
    );
    tensors.insert(
        "uncond_real_len".into(),
        Tensor::from_vec(vec![uncond_len as f32], flame_core::Shape::from_dims(&[1]), device.clone())?,
    );
    flame_core::serialization::save_file(&tensors, &out_path)?;
    println!("Saved to {} ({:.1}s total)", out_path, t_total.elapsed().as_secs_f32());
    Ok(())
}
