//! Compare EVERY tensor in block 0 between sync and FlameSwap paths.
//! Prints a table showing key, dtype, shape, and max absolute diff.

use flame_core::{global_cuda_device, DType, Result, Shape, Tensor};
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use std::collections::{BTreeMap, HashMap};

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();
    let config = LTX2Config::default();

    println!("Loading model globals...");
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;

    // --- Sync path: get raw block weights ---
    println!("Loading block 0 via sync (load_block_from_disk)...");
    let key_prefix = model.key_prefix.clone();
    let sync_weights = {
        let pfx = format!("{key_prefix}transformer_blocks.0.");
        let raw = flame_core::serialization::load_file_filtered(
            MODEL_PATH, &device,
            |key| key.starts_with(&pfx) && !key.contains("audio") ||
                  (key.starts_with(&pfx) && key.contains("audio_to_video")),
        )?;
        // Strip prefix
        raw.into_iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix(&key_prefix).unwrap_or(&k).to_string();
                (stripped, v)
            })
            .collect::<BTreeMap<String, Tensor>>()
    };

    // --- FlameSwap path: get raw block weights ---
    println!("Initializing FlameSwap...");
    model.init_swap()?;

    println!("Loading block 0 via FlameSwap...");
    let swap = model.swap.as_mut().unwrap();
    swap.prefetch(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let raw = swap.await_block(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let swap_weights: BTreeMap<String, Tensor> = raw.into_iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix(&key_prefix).unwrap_or(&k).to_string();
            (stripped, v)
        })
        .collect();

    // Also get f32_cache entries
    let f32_cached: BTreeMap<String, Tensor> = if !model.f32_cache.is_empty() {
        model.f32_cache[0].iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    } else {
        BTreeMap::new()
    };

    // --- Compare ---
    println!("\n{:<65} {:>6} {:>6} {:>12} {:>10}", "KEY", "SYNC", "SWAP", "SHAPE", "MAX_DIFF");
    println!("{}", "-".repeat(105));

    // All unique keys
    let mut all_keys: Vec<String> = sync_weights.keys().cloned().collect();
    for k in swap_weights.keys() {
        if !all_keys.contains(k) { all_keys.push(k.clone()); }
    }
    for k in f32_cached.keys() {
        if !all_keys.contains(k) { all_keys.push(k.clone()); }
    }
    all_keys.sort();

    let mut n_match = 0;
    let mut n_differ = 0;
    let mut n_missing = 0;

    for key in &all_keys {
        // Skip audio-only keys
        if key.contains("audio") && !key.contains("audio_to_video") { continue; }
        if key.contains("video_to_audio") { continue; }

        let sync_t = sync_weights.get(key);
        let swap_t = swap_weights.get(key).or_else(|| f32_cached.get(key));

        let sync_dtype = sync_t.map(|t| format!("{:?}", t.dtype())).unwrap_or("MISS".into());
        let swap_dtype = swap_t.map(|t| format!("{:?}", t.dtype())).unwrap_or("MISS".into());

        let shape_str = sync_t.or(swap_t)
            .map(|t| format!("{:?}", t.shape().dims()))
            .unwrap_or("?".into());

        if sync_t.is_none() || swap_t.is_none() {
            println!("{:<65} {:>6} {:>6} {:>12} {:>10}",
                &key[..key.len().min(65)], sync_dtype, swap_dtype, shape_str, "MISSING");
            n_missing += 1;
            continue;
        }

        let a = sync_t.unwrap();
        let b = swap_t.unwrap();

        // Convert both to f32 for comparison
        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;

        if a_data.len() != b_data.len() {
            println!("{:<65} {:>6} {:>6} {:>12} {:>10}",
                &key[..key.len().min(65)], sync_dtype, swap_dtype, shape_str, "SIZE_DIFF");
            n_differ += 1;
            continue;
        }

        let max_diff: f32 = a_data.iter().zip(b_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let status = if max_diff < 0.0001 { "=" } else if max_diff < 0.01 { "~" } else { "!" };

        println!("{:<65} {:>6} {:>6} {:>12} {:>10.6} {}",
            &key[..key.len().min(65)], sync_dtype, swap_dtype, shape_str, max_diff, status);

        if max_diff < 0.0001 { n_match += 1; } else { n_differ += 1; }
    }

    println!("\n{}", "-".repeat(105));
    println!("MATCH (diff<0.0001): {}", n_match);
    println!("DIFFER (diff>=0.0001): {}", n_differ);
    println!("MISSING: {}", n_missing);

    Ok(())
}
