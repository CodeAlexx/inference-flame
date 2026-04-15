//! Standalone test for Gemma-3 12B text encoder.
//! Compares output against official Python encoder output.
//!
//! ⚠️ This does NOT connect to the inference pipeline.
//! ⚠️ It is a standalone validation tool.
//!
//! Usage:
//!   RUST_LOG=info cargo run --release --bin test_gemma_encoder

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use flame_core::{DType, Tensor};
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;

const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const REF_DIR: &str = "/home/alex/ltx2-refs/gemma3";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    println!("=== Gemma-3 12B Encoder Test ===\n");

    // --- 1. Load reference tokens ---
    let tokens_json = std::fs::read_to_string(format!("{REF_DIR}/tokens.json"))?;
    let tokens: serde_json::Value = serde_json::from_str(&tokens_json)?;
    let input_ids: Vec<i32> = tokens["input_ids"]
        .as_array().unwrap()
        .iter().map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let attention_mask: Vec<i32> = tokens["attention_mask"]
        .as_array().unwrap()
        .iter().map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let prompt = tokens["prompt"].as_str().unwrap_or("?");
    let real_count = attention_mask.iter().filter(|&&m| m != 0).count();
    println!("Prompt: \"{}\"", prompt);
    println!("Tokens: {} total, {} real\n", input_ids.len(), real_count);

    // --- 2. Find Gemma safetensors shards ---
    let mut shards: Vec<String> = Vec::new();
    for i in 1..=5 {
        let path = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
        if std::path::Path::new(&path).exists() {
            shards.push(path);
        }
    }
    println!("Found {} Gemma shards", shards.len());
    if shards.is_empty() {
        eprintln!("ERROR: No Gemma shards found at {GEMMA_ROOT}");
        return Ok(());
    }

    // --- 3. Load encoder via BlockOffloader ---
    println!("\nLoading Gemma-3 via BlockOffloader...");
    let t0 = Instant::now();
    let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();
    let mut encoder = Gemma3Encoder::load(&shard_refs, &device, input_ids.len())?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- 4. Forward pass ---
    println!("\nRunning forward pass...");
    let t1 = Instant::now();
    let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;
    let encode_time = t1.elapsed().as_secs_f32();

    println!("  {} hidden states", all_hidden.len());
    println!("  Each shape: {:?}", all_hidden[0].shape());
    println!("  Mask shape: {:?}", mask_out.shape());
    println!("  Encode time: {:.1}s", encode_time);

    // --- 5. Compare against reference ---
    println!("\n--- Comparing against Python reference ---");

    // Hidden state 0 (embedding output)
    let ref_h0 = load_ref_tensor(&format!("{REF_DIR}/hidden_state_0.safetensors"), "h", &device)?;
    let err_h0 = max_abs_diff(&all_hidden[0], &ref_h0)?;
    let pass_h0 = err_h0 < 0.1; // BF16 rounding on embedding scale (sqrt(3840))
    println!("  hidden_state_0: max_err={:.6} {}", err_h0,
        if pass_h0 { "PASS" } else { "FAIL" });

    // Hidden state 48 (final layer after norm)
    // Note: max error includes PAD tokens which diverge freely.
    // For LTX-2, only real token embeddings matter.
    let ref_h48 = load_ref_tensor(&format!("{REF_DIR}/hidden_state_48.safetensors"), "h", &device)?;
    let err_h48 = max_abs_diff(&all_hidden[48], &ref_h48)?;
    let pass_h48 = err_h48 < 50.0; // BF16 vs FP32 RMSNorm over 48 layers; pad tokens inflate max
    println!("  hidden_state_48: max_err={:.6} (includes pad tokens) {}",
        err_h48, if pass_h48 { "PASS" } else { "FAIL" });

    // --- 6. Stats on final hidden state ---
    let h48_f32 = all_hidden[48].to_dtype(flame_core::DType::F32)?;
    let flat = h48_f32.reshape(&[h48_f32.shape().elem_count()])?;
    let values = flat.to_vec_f32()?;
    let n = values.len();
    let mean: f32 = values.iter().sum::<f32>() / n as f32;
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n  hidden_state_48 stats:");
    println!("    mean: {:.6}", mean);
    println!("    min:  {:.6}", min);
    println!("    max:  {:.6}", max);

    // --- 7. Save our outputs ---
    // Save h1 (layer 0 output) for debugging single-layer accuracy
    let out_h1_path = format!("{REF_DIR}/rust_hidden_state_1.safetensors");
    let mut save_h1 = HashMap::new();
    save_h1.insert("h".to_string(), all_hidden[1].clone());
    flame_core::serialization::save_file(&save_h1, &out_h1_path)?;
    println!("  Saved rust h1 (layer 0 output)");

    // Save h0 for debugging
    let out_h0_path = format!("{REF_DIR}/rust_hidden_state_0.safetensors");
    let mut save_h0 = HashMap::new();
    save_h0.insert("h".to_string(), all_hidden[0].clone());
    flame_core::serialization::save_file(&save_h0, &out_h0_path)?;

    let out_path = format!("{REF_DIR}/rust_hidden_state_48.safetensors");
    let mut save_map = HashMap::new();
    save_map.insert("h".to_string(), all_hidden[48].clone());
    flame_core::serialization::save_file(&save_map, &out_path)?;
    println!("\n  Saved to {}", out_path);

    // ===================================================================
    // Phase 2: Feature Extractor + aggregate_embed
    // ===================================================================
    println!("\n--- Phase 2: Feature Extractor ---");

    // Free Gemma encoder to reclaim VRAM
    drop(encoder);
    println!("  Freed Gemma encoder");

    // Load aggregate_embed weights from LTX-2 checkpoint
    println!("  Loading aggregate_embed from LTX-2 checkpoint...");
    let agg_weights = flame_core::serialization::load_file_filtered(
        Path::new(LTX_CHECKPOINT),
        &device,
        |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
    )?;

    let agg_w = agg_weights.get("text_embedding_projection.video_aggregate_embed.weight")
        .ok_or("Missing video_aggregate_embed.weight")?;
    let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");
    println!("  aggregate_embed weight: {:?}", agg_w.shape());
    if let Some(b) = agg_b {
        println!("  aggregate_embed bias: {:?}", b.shape());
    }

    // Run feature extractor
    let hidden_refs: Vec<&Tensor> = all_hidden.iter().collect();
    let t_feat = Instant::now();
    let video_feats = feature_extractor::feature_extract_and_project(
        &all_hidden,
        &mask_out,
        agg_w,
        agg_b,
        4096, // target_dim
    )?;
    println!("  Feature extraction took {:.2}s", t_feat.elapsed().as_secs_f32());
    println!("  Output shape: {:?}", video_feats.shape());

    // Compare against Python reference
    let ref_feats = load_ref_tensor(
        &format!("{REF_DIR}/feature_extractor_output.safetensors"), "video_feats", &device,
    )?;
    let err_feats = max_abs_diff(&video_feats, &ref_feats)?;
    let pass_feats = err_feats < 50.0; // pad tokens diverge
    println!("  feature_extractor: max_err={:.6} (includes pad tokens) {}",
        err_feats, if pass_feats { "PASS" } else { "FAIL" });

    // Check real tokens only
    let ref_feats_f32 = ref_feats.to_dtype(DType::F32)?;
    let our_feats_f32 = video_feats.to_dtype(DType::F32)?;
    // Real tokens start at position (seq_len - real_count)
    let real_start = input_ids.len() - real_count;
    let ref_real = ref_feats_f32.narrow(1, real_start, real_count)?;
    let our_real = our_feats_f32.narrow(1, real_start, real_count)?;
    let diff_real = ref_real.sub(&our_real)?.abs()?;
    let diff_flat = diff_real.reshape(&[diff_real.shape().elem_count()])?;
    let diff_vals = diff_flat.to_vec_f32()?;
    let real_max = diff_vals.iter().cloned().fold(0.0f32, f32::max);
    let real_mean: f32 = diff_vals.iter().sum::<f32>() / diff_vals.len() as f32;
    let pass_feats_real = real_max < 10.0; // Gemma errors propagate through 188160→4096 projection
    println!("  feature_extractor (real tokens): max={:.4} mean={:.6} {}",
        real_max, real_mean, if pass_feats_real { "PASS" } else { "FAIL" });

    // Save our feature output
    let mut save_feats = HashMap::new();
    save_feats.insert("video_feats".to_string(), video_feats.clone());
    flame_core::serialization::save_file(&save_feats, &format!("{REF_DIR}/rust_feature_extractor_output.safetensors"))?;

    let all_pass = pass_h0 && pass_h48 && pass_feats && pass_feats_real;
    println!("\n=== {} ===", if all_pass { "ALL PASS" } else { "SOME FAILED" });
    Ok(())
}

fn load_ref_tensor(
    path: &str,
    key: &str,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> Result<flame_core::Tensor, Box<dyn std::error::Error>> {
    let weights = flame_core::serialization::load_file(std::path::Path::new(path), device)?;
    let t = weights.get(key).ok_or_else(|| format!("Key '{}' not found in {}", key, path))?;
    Ok(t.clone())
}

fn max_abs_diff(
    a: &flame_core::Tensor,
    b: &flame_core::Tensor,
) -> Result<f32, Box<dyn std::error::Error>> {
    let a_f32 = a.to_dtype(flame_core::DType::F32)?;
    let b_f32 = b.to_dtype(flame_core::DType::F32)?;
    let diff = a_f32.sub(&b_f32)?;
    let abs_diff = diff.abs()?;
    let flat = abs_diff.reshape(&[abs_diff.shape().elem_count()])?;
    let values = flat.to_vec_f32()?;
    Ok(values.iter().cloned().fold(0.0f32, f32::max))
}
