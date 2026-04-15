//! Standalone test for Mistral Small 3.1 24B text encoder.
//!
//! Loads the FLUX 2 text encoder via BlockOffloader, tokenizes a test prompt,
//! runs forward pass, and prints embedding statistics.
//!
//! ⚠️ This does NOT connect to any inference pipeline.
//! ⚠️ It is a standalone validation tool.
//!
//! Usage:
//!   RUST_LOG=info cargo run --release --bin mistral_test

use std::time::Instant;

use inference_flame::models::mistral_encoder::MistralEncoder;

const SNAP: &str = "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/6aab690f8379b70adc89edfa6bb99b3537ba52a3";

const TOKENIZER_PATH: &str = concat!(
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/",
    "snapshots/6aab690f8379b70adc89edfa6bb99b3537ba52a3/tokenizer.json"
);

const MAX_LENGTH: usize = 512;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    println!("=== Mistral Small 3.1 24B Encoder Test ===\n");

    // --- Step 1: Find text encoder shards ---
    let mut shards: Vec<String> = Vec::new();
    for i in 1..=10 {
        let path = format!("{SNAP}/text_encoder/model-{i:05}-of-00010.safetensors");
        if std::path::Path::new(&path).exists() {
            shards.push(path);
        }
    }
    println!("Found {} text encoder shards", shards.len());
    if shards.is_empty() {
        eprintln!("ERROR: No text encoder shards found at {SNAP}/text_encoder/");
        return Ok(());
    }

    // --- Step 2: Tokenize ---
    println!("\nTokenizing...");
    let prompt = "a cat walking in a garden";

    let (token_ids, attention_mask) = tokenize(prompt)?;
    let real_count = attention_mask.iter().filter(|&&m| m != 0).count();
    println!("  prompt: \"{}\"", prompt);
    println!("  tokens: {} total, {} real", token_ids.len(), real_count);
    println!("  first 10 ids: {:?}", &token_ids[..10.min(token_ids.len())]);

    // --- Step 3: Load encoder ---
    println!("\nLoading Mistral via BlockOffloader...");
    let t0 = Instant::now();
    let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();
    let mut encoder = MistralEncoder::load(&shard_refs, &device)?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!("  Config: {:?}", encoder.config());

    // --- Step 4: Forward pass ---
    println!("\nRunning forward pass...");
    let t1 = Instant::now();
    let output = encoder.encode(&token_ids, &attention_mask)?;
    let encode_time = t1.elapsed().as_secs_f32();

    let out_dims = output.shape().dims().to_vec();
    println!("  Output shape: {:?}", out_dims);
    println!("  Expected:     [1, {}, {}]", MAX_LENGTH, encoder.output_dim());
    println!("  Encode time:  {:.2}s", encode_time);

    // --- Step 5: Stats ---
    let out_f32 = output.to_dtype(flame_core::DType::F32)?;
    let flat = out_f32.reshape(&[out_dims.iter().product::<usize>()])?;
    let n = flat.shape().dims()[0];

    // Read values to CPU for stats
    let values = flat.to_vec_f32()?;
    let mean: f32 = values.iter().sum::<f32>() / n as f32;
    let min: f32 = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max: f32 = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let variance: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();

    println!("\n  Embedding stats:");
    println!("    mean: {:.6}", mean);
    println!("    std:  {:.6}", std);
    println!("    min:  {:.6}", min);
    println!("    max:  {:.6}", max);

    // --- Step 6: Save for comparison ---
    let out_path = "/home/alex/ltx2-refs/mistral/mistral_embeddings.safetensors";
    let out_dir = std::path::Path::new(out_path).parent().unwrap();
    std::fs::create_dir_all(out_dir)?;

    let mut save_map = std::collections::HashMap::new();
    save_map.insert("embeddings".to_string(), output);
    flame_core::serialization::save_file(&save_map, out_path)?;
    println!("\n  Saved embeddings to {}", out_path);

    println!("\n=== DONE ===");
    Ok(())
}

/// Tokenize a prompt using the HuggingFace tokenizers crate.
/// Returns (token_ids, attention_mask) both padded/truncated to MAX_LENGTH.
fn tokenize(prompt: &str) -> Result<(Vec<i32>, Vec<i32>), Box<dyn std::error::Error>> {
    use tokenizers::Tokenizer;

    let mut tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    // Configure padding to MAX_LENGTH, left-padded (matching Mistral convention)
    let padding = tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(MAX_LENGTH),
        direction: tokenizers::PaddingDirection::Left,
        pad_id: 0, // Will be set properly below
        pad_token: "[PAD]".to_string(),
        ..Default::default()
    };
    tokenizer.with_padding(Some(padding));

    // Configure truncation
    let truncation = tokenizers::TruncationParams {
        max_length: MAX_LENGTH,
        ..Default::default()
    };
    tokenizer.with_truncation(Some(truncation))
        .map_err(|e| format!("Truncation config error: {e}"))?;

    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| format!("Tokenization failed: {e}"))?;

    let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let mask: Vec<i32> = encoding.get_attention_mask().iter().map(|&m| m as i32).collect();

    Ok((ids, mask))
}
