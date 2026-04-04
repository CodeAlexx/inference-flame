//! Klein 9B text encoding — save embeddings to safetensors for offline generation.
//!
//! Usage: klein9b_encode ["prompt"] ["negative"]
//!
//! Loads Qwen3 8B, encodes positive + negative prompts, saves to
//! output/klein9b_embeddings.safetensors. The klein9b_infer binary
//! loads these cached embeddings so the encoder never shares VRAM with the model.

use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use flame_core::{global_cuda_device, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const ENCODER_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/klein9b_embeddings.safetensors";

const DEFAULT_PROMPT: &str = "A young oracle girl kneeling on ancient mossy temple stones in heavy rain, both hands cupped upward holding a single floating orb of pure white light that contains a swirling miniature cosmos of futures yet to happen, her soaked ceremonial white robes clinging to her frame and glowing faintly from the orb light below her face, her expression one of profound solitary devotion, rain falling in visible silver needles through the orb's light creating prismatic micro-rainbows around her hands, the ancient temple columns behind her half-consumed by darkness, photorealistic fantasy cinematic style, cold rain light versus warm divine orb glow, Terrence Malick visual language, 8K";
const DEFAULT_NEGATIVE: &str = "lowres, bad quality, worst quality, bad anatomy, blurry, watermark, simple background, transparent background, sketch, jpeg artifacts, ugly, poorly drawn, censor";

const TXT_PAD_LEN: usize = 512;
const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Load all sharded safetensors from a directory.
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

    let prompt = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let negative = std::env::args().nth(2).unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());

    println!("============================================================");
    println!("Klein 9B — Text Encoding (Qwen3 8B)");
    println!("============================================================");
    println!("  Prompt: {}", prompt);
    println!("  Negative: {}", negative);

    let device = global_cuda_device();

    // Tokenize
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let pos_formatted = format!("{KLEIN_TEMPLATE_PRE}{prompt}{KLEIN_TEMPLATE_POST}");
    let neg_formatted = format!("{KLEIN_TEMPLATE_PRE}{negative}{KLEIN_TEMPLATE_POST}");

    let pos_enc = tokenizer.encode(pos_formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;
    let neg_enc = tokenizer.encode(neg_formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;

    let pad_id = 151643i32;
    let mut pos_ids: Vec<i32> = pos_enc.get_ids().iter().map(|&id| id as i32).collect();
    let mut neg_ids: Vec<i32> = neg_enc.get_ids().iter().map(|&id| id as i32).collect();
    println!("  Pos tokens: {}, Neg tokens: {}", pos_ids.len(), neg_ids.len());
    pos_ids.resize(TXT_PAD_LEN, pad_id);
    neg_ids.resize(TXT_PAD_LEN, pad_id);

    // Load encoder
    println!("\n--- Loading Qwen3 8B encoder ---");
    let t0 = Instant::now();
    let enc_weights = load_sharded_weights(ENCODER_DIR, &device)?;
    let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
    println!("  Config: hidden={}, layers={}, heads={}",
        enc_config.hidden_size, enc_config.num_layers, enc_config.num_heads);
    let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Encode
    println!("\n--- Encoding ---");
    let t0 = Instant::now();
    let pos_hidden = encoder.encode(&pos_ids)?;
    let neg_hidden = encoder.encode(&neg_ids)?;
    println!("  pos: {:?}, neg: {:?}", pos_hidden.dims(), neg_hidden.dims());
    println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

    // Save
    println!("\n--- Saving embeddings ---");
    let mut tensors = HashMap::new();
    tensors.insert("pos_hidden".to_string(), pos_hidden);
    tensors.insert("neg_hidden".to_string(), neg_hidden);
    flame_core::serialization::save_file(&tensors, OUTPUT_PATH)?;
    println!("  Saved to {}", OUTPUT_PATH);

    println!("\nTotal time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
