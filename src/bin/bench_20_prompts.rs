//! 20-prompt benchmark: Gemma-3 → FeatureExtractor → compare vs Python reference.
//!
//! Usage:
//!   RUST_LOG=info cargo run --release --bin bench_20_prompts

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use flame_core::{DType, Tensor};
use inference_flame::models::gemma3_encoder::Gemma3Encoder;
use inference_flame::models::feature_extractor;

const GEMMA_ROOT: &str = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone";
const LTX_CHECKPOINT: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const BENCH_DIR: &str = "/home/alex/ltx2-refs/bench20";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    println!("=== 20-Prompt Gemma-3 Benchmark ===\n");

    // --- 1. Find Gemma shards ---
    let mut shards: Vec<String> = Vec::new();
    for i in 1..=5 {
        let path = format!("{GEMMA_ROOT}/model-{i:05}-of-00005.safetensors");
        if Path::new(&path).exists() {
            shards.push(path);
        }
    }
    println!("Found {} Gemma shards", shards.len());

    // --- 2. Load prompts ---
    let prompts_json = std::fs::read_to_string(format!("{BENCH_DIR}/prompts.json"))?;
    let prompts: Vec<String> = serde_json::from_str(&prompts_json)?;
    println!("Loaded {} prompts\n", prompts.len());

    // --- 3. Load encoder ---
    println!("Loading Gemma-3 via FlameSwap...");
    let t_load = Instant::now();
    let shard_refs: Vec<&str> = shards.iter().map(|s| s.as_str()).collect();
    // Use max seq len from token files (all are 256)
    let mut encoder = Gemma3Encoder::load(&shard_refs, &device, 256)?;
    println!("  Loaded in {:.1}s\n", t_load.elapsed().as_secs_f32());

    // --- 4. Load aggregate_embed weights ---
    println!("Loading aggregate_embed from LTX-2 checkpoint...");
    let agg_weights = flame_core::serialization::load_file_filtered(
        Path::new(LTX_CHECKPOINT),
        &device,
        |key| key.starts_with("text_embedding_projection.video_aggregate_embed"),
    )?;
    let agg_w = agg_weights.get("text_embedding_projection.video_aggregate_embed.weight")
        .ok_or("Missing video_aggregate_embed.weight")?;
    let agg_b = agg_weights.get("text_embedding_projection.video_aggregate_embed.bias");
    println!("  weight: {:?}, bias: {:?}\n", agg_w.shape(),
        agg_b.map(|b| format!("{:?}", b.shape())).unwrap_or("None".into()));

    // --- 5. Encode all 20 prompts ---
    println!("--- Encoding 20 prompts (Rust) ---");
    let mut rust_times = Vec::new();
    let mut all_video_feats = Vec::new();

    for i in 0..prompts.len() {
        // Load tokens
        let token_path = format!("{BENCH_DIR}/tokens_{i:02}.json");
        let token_json = std::fs::read_to_string(&token_path)?;
        let tokens: serde_json::Value = serde_json::from_str(&token_json)?;

        let input_ids: Vec<i32> = tokens["input_ids"]
            .as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let attention_mask: Vec<i32> = tokens["attention_mask"]
            .as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let real_count = attention_mask.iter().filter(|&&m| m != 0).count();

        // Encode
        let t0 = Instant::now();
        let (all_hidden, mask_out) = encoder.encode(&input_ids, &attention_mask)?;

        // Feature extract
        let video_feats = feature_extractor::feature_extract_and_project(
            &all_hidden,
            &mask_out,
            agg_w,
            agg_b,
            4096,
        )?;
        let dt = t0.elapsed().as_secs_f32();
        rust_times.push(dt);

        println!("  [{i:2}] {dt:.2}s  feats={:?}  real={real_count}  {}",
            video_feats.shape(), &prompts[i][..prompts[i].len().min(50)]);

        // Save Rust output for comparison
        let mut save_map = HashMap::new();
        save_map.insert("video_context".to_string(), video_feats.clone());
        flame_core::serialization::save_file(
            &save_map,
            &format!("{BENCH_DIR}/rust_output_{i:02}.safetensors"),
        )?;

        all_video_feats.push(video_feats);
    }

    let total_rust: f32 = rust_times.iter().sum();
    println!("\n  Rust total: {total_rust:.1}s  ({:.2}s/prompt avg)",
        total_rust / prompts.len() as f32);

    // --- 6. Compare against Python reference (if available) ---
    println!("\n--- Comparing against Python reference ---");
    let mut pass_count = 0;
    let mut fail_count = 0;

    for i in 0..prompts.len() {
        let ref_path = format!("{BENCH_DIR}/ref_output_{i:02}.safetensors");
        if !Path::new(&ref_path).exists() {
            println!("  [{i:2}] SKIP (no Python reference yet)");
            continue;
        }

        let ref_weights = flame_core::serialization::load_file(Path::new(&ref_path), &device)?;
        let ref_video = ref_weights.get("video_context")
            .ok_or(format!("Missing video_context in {ref_path}"))?;

        let real_count = {
            let token_path = format!("{BENCH_DIR}/tokens_{i:02}.json");
            let token_json = std::fs::read_to_string(&token_path)?;
            let tokens: serde_json::Value = serde_json::from_str(&token_json)?;
            let mask: Vec<i32> = tokens["attention_mask"]
                .as_array().unwrap()
                .iter().map(|v| v.as_i64().unwrap() as i32)
                .collect();
            mask.iter().filter(|&&m| m != 0).count()
        };
        let seq_len = all_video_feats[i].shape().dims()[1];
        let real_start = seq_len - real_count;

        // Compare real tokens only
        let our_f32 = all_video_feats[i].to_dtype(DType::F32)?;
        let ref_f32 = ref_video.to_dtype(DType::F32)?;
        let our_real = our_f32.narrow(1, real_start, real_count)?;
        let ref_real = ref_f32.narrow(1, real_start, real_count)?;

        let diff = our_real.sub(&ref_real)?.abs()?;
        let flat = diff.reshape(&[diff.shape().elem_count()])?;
        let vals = flat.to_vec_f32()?;
        let max_err = vals.iter().cloned().fold(0.0f32, f32::max);
        let mean_err: f32 = vals.iter().sum::<f32>() / vals.len() as f32;

        let pass = max_err < 10.0;
        if pass { pass_count += 1; } else { fail_count += 1; }

        println!("  [{i:2}] max={max_err:.4} mean={mean_err:.6} {}  {}",
            if pass { "PASS" } else { "FAIL" },
            &prompts[i][..prompts[i].len().min(40)]);
    }

    if pass_count + fail_count > 0 {
        println!("\n  {pass_count}/{} passed", pass_count + fail_count);
    } else {
        println!("\n  No Python references available yet — run bench_20_prompts.py first");
    }

    println!("\n=== Done ===");
    Ok(())
}
