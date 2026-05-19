//! Klein 9B parity dump (Rust side).
//!
//! Fixed short prompt, tokenizes with Qwen3 tokenizer + Klein template,
//! pads to 512, runs Qwen3 8B encoder, saves token IDs + hidden states
//! to /tmp/klein9b_rust_parity.safetensors for numeric comparison with Python.
//!
//! Mirrors the Python script /tmp/klein9b_py_parity.py.

use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

const ENCODER_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";
const OUT_PATH: &str = "/tmp/klein9b_rust_parity.safetensors";

// Fixed short prompt — same as Python script
const PROMPT: &str = "a cute cat sitting on a bench";

const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const TXT_PAD_LEN: usize = 512;
const PAD_ID: i32 = 151643; // <|endoftext|>

fn load_sharded_weights(
    dir: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut all_weights = HashMap::new();
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();
    println!("  Loading {} shards...", shard_paths.len());
    for (i, path) in shard_paths.iter().enumerate() {
        let t0 = Instant::now();
        let shard = flame_core::serialization::load_file(path, device)?;
        println!("    Shard {}/{}: {} keys ({:.1}s)",
            i + 1, shard_paths.len(), shard.len(), t0.elapsed().as_secs_f32());
        all_weights.extend(shard);
    }
    Ok(all_weights)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Klein 9B Parity Dump (Rust)");
    println!("============================================================");
    println!("  Prompt: {}", PROMPT);

    let device = global_cuda_device();

    // Step 1: Tokenize
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {}", e))?;

    let formatted = format!("{KLEIN_TEMPLATE_PRE}{PROMPT}{KLEIN_TEMPLATE_POST}");
    println!("  Template (first 60): {:?}", &formatted[..formatted.len().min(60)]);

    let enc = tokenizer.encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let raw_ids: Vec<i32> = enc.get_ids().iter().map(|&id| id as i32).collect();
    println!("[rust] raw token count: {}", raw_ids.len());
    println!("[rust] first 10 token IDs: {:?}", &raw_ids[..10.min(raw_ids.len())]);

    let mut padded_ids = raw_ids.clone();
    padded_ids.resize(TXT_PAD_LEN, PAD_ID);

    // Step 2: Load encoder
    println!("\n--- Loading Qwen3 8B encoder ---");
    let t0 = Instant::now();
    let enc_weights = load_sharded_weights(ENCODER_DIR, &device)?;
    let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
    println!("  Config: hidden={}, layers={}, heads={}",
        enc_config.hidden_size, enc_config.num_layers, enc_config.num_heads);
    let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Step 3: Encode
    println!("\n--- Encoding ---");
    let t0 = Instant::now();
    let hidden = encoder.encode(&padded_ids)?;
    let dims = hidden.shape().dims().to_vec();
    println!("[rust] hidden shape: {:?}", dims);
    println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

    // Print stats
    let hidden_f32 = hidden.to_dtype(DType::F32)?;
    let data = hidden_f32.to_vec()?;
    let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("[rust] hidden norm: {:.4}", norm);
    println!("[rust] hidden[0,:5]: {:?}", &data[..5]);

    // Step 4: Save
    println!("\n--- Saving to {} ---", OUT_PATH);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    // Save token IDs (raw and padded)
    let raw_ids_f: Vec<f32> = raw_ids.iter().map(|&x| x as f32).collect();
    let raw_ids_t = Tensor::from_vec(
        raw_ids_f, Shape::from_dims(&[raw_ids.len()]), device.clone()
    )?.to_dtype(DType::I32)?;
    tensors.insert("rust_input_ids_raw".to_string(), raw_ids_t);

    let padded_f: Vec<f32> = padded_ids.iter().map(|&x| x as f32).collect();
    let padded_t = Tensor::from_vec(
        padded_f, Shape::from_dims(&[TXT_PAD_LEN]), device.clone()
    )?.to_dtype(DType::I32)?;
    tensors.insert("rust_input_ids".to_string(), padded_t);

    // Save hidden (already [1, 512, 12288] in BF16, convert to F32 for comparison)
    let hidden_2d = hidden_f32
        .reshape(&[dims[1] as usize, dims[2] as usize])?; // [512, 12288]
    tensors.insert("rust_hidden_combined".to_string(), hidden_2d);

    flame_core::serialization::save_file(&tensors, OUT_PATH)?;
    println!("  Saved {} tensors", tensors.len());

    println!("\nTotal time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
