//! Qwen-Image-2512 — Stage 1 (Rust text encode).
//!
//! Port of `scripts/qwenimage_encode.py`. Loads the Qwen2.5-VL-7B text
//! encoder (4 safetensors shards) through the in-tree `Qwen25VLEncoder`,
//! applies the diffusers prompt template, drops the first 34 system-
//! prompt tokens, trims to non-padded length, and writes cond + uncond
//! embeddings to a safetensors file in the exact format `qwenimage_gen`
//! expects.
//!
//! Usage:
//!     target/release/qwenimage_encode \
//!         "a photograph of an astronaut riding a horse on mars" \
//!         ""  \
//!         /path/to/embeds.safetensors
//!
//! Knobs:
//!     QWEN_TE_MAX_LEN   — tokenizer max length (default 1024).
//!     QWEN_TE_SHARDS    — colon-separated shard paths (default: Qwen-Image-2512 snapshot).
//!     QWEN_TOKENIZER    — tokenizer.json path (default: Qwen-Image-2512 snapshot).
//!
//! Stage 2 then consumes the written embeddings:
//!     target/release/qwenimage_gen <embeds.safetensors> <out.png>

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::qwen25vl_encoder::{Qwen25VLConfig, Qwen25VLEncoder};

const SNAP: &str = "/home/alex/.serenity/models/checkpoints/qwen-image-2512";

/// System prefix + user wrapper applied before tokenization.
/// Matches `pipeline_qwenimage.py::PROMPT_TEMPLATE_ENCODE`.
const PROMPT_PREFIX: &str =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
     texture, quantity, text, spatial relationships of the objects and background:\
     <|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Number of leading tokens to drop (the system prompt). Match Python
/// constant `PROMPT_TEMPLATE_ENCODE_START_IDX`.
const DROP_IDX: usize = 34;

/// Qwen2.5-VL pad token id.
const PAD_TOKEN: i32 = 151643;

fn text_encoder_shards() -> Vec<PathBuf> {
    if let Ok(s) = std::env::var("QWEN_TE_SHARDS") {
        return s.split(':').map(PathBuf::from).collect();
    }
    (1..=4)
        .map(|i| PathBuf::from(format!("{SNAP}/text_encoder/model-0000{i}-of-00004.safetensors")))
        .collect()
}

fn tokenizer_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_TOKENIZER") {
        return PathBuf::from(p);
    }
    PathBuf::from(format!("{SNAP}/tokenizer/tokenizer.json"))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Tokenize with the Qwen template, pad to `max_len`, truncate if over.
fn tokenize_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_len: usize,
) -> anyhow::Result<Vec<i32>> {
    let wrapped = format!("{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}");
    let enc = tokenizer
        .encode(wrapped, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let ids = enc.get_ids();
    let mut out: Vec<i32> = ids.iter().take(max_len).map(|&i| i as i32).collect();
    while out.len() < max_len {
        out.push(PAD_TOKEN);
    }
    Ok(out)
}

/// Run the encoder, drop the first `DROP_IDX` tokens, trim trailing
/// pad tokens. Returns `[1, L, 3584]` BF16.
fn encode_and_trim(
    encoder: &Qwen25VLEncoder,
    token_ids: &[i32],
) -> anyhow::Result<Tensor> {
    let hidden = encoder.encode(token_ids)?; // [1, seq_len, 3584]
    let dims = hidden.shape().dims().to_vec();
    let seq_len = dims[1];
    let hidden_size = dims[2];

    let real_len = token_ids
        .iter()
        .position(|&id| id == PAD_TOKEN)
        .unwrap_or(seq_len);
    if real_len <= DROP_IDX {
        anyhow::bail!(
            "prompt produced only {real_len} tokens; expected > {DROP_IDX} after Qwen template"
        );
    }
    let kept_len = real_len - DROP_IDX;
    // narrow(1, DROP_IDX, kept_len) gives [1, kept_len, hidden_size]
    let out = hidden.narrow(1, DROP_IDX, kept_len)?;
    let _ = hidden_size;
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    env_logger::try_init().ok();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_else(|| {
        "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed"
            .to_string()
    });
    let negative = args.get(2).cloned().unwrap_or_default();
    let out_path = args.get(3).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/qwenimage_embeds.safetensors".to_string()
    });

    let max_len = env_usize("QWEN_TE_MAX_LEN", 1024) + DROP_IDX;

    println!("=== Qwen-Image-2512 — Stage 1 (Rust text encode) ===");
    println!("Prompt:   {:?}", prompt);
    println!("Negative: {:?}", negative);
    println!("Output:   {out_path}");
    println!();

    // Tokenizer
    let tok_path = tokenizer_path();
    if !tok_path.exists() {
        anyhow::bail!(
            "tokenizer.json not found at {:?}. Regenerate with: \
             python -c \"from transformers import AutoTokenizer as T; \
             T.from_pretrained('{}/tokenizer').save_pretrained('{}/tokenizer')\"",
            tok_path,
            SNAP,
            SNAP
        );
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer from {:?}: {e}", tok_path))?;

    println!("--- Tokenizing ---");
    let pos_ids = tokenize_prompt(&tokenizer, &prompt, max_len)?;
    let neg_ids = tokenize_prompt(&tokenizer, &negative, max_len)?;
    let pos_real = pos_ids.iter().position(|&i| i == PAD_TOKEN).unwrap_or(pos_ids.len());
    let neg_real = neg_ids.iter().position(|&i| i == PAD_TOKEN).unwrap_or(neg_ids.len());
    println!("  pos: {pos_real}/{max_len} real tokens (trimmed output: {})", pos_real - DROP_IDX);
    println!("  neg: {neg_real}/{max_len} real tokens (trimmed output: {})", neg_real - DROP_IDX);
    println!();

    // Load encoder shards (BF16 → BF16). Filter to text-only keys (drop
    // `visual.*` so we don't carry ~7B of vision weights into GPU RAM).
    println!("--- Loading text encoder shards ---");
    let t_load = Instant::now();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for shard in text_encoder_shards() {
        let shard_weights = flame_core::serialization::load_file_filtered(
            &shard,
            &device,
            |k| !k.starts_with("visual."),
        )?;
        for (k, v) in shard_weights {
            weights.insert(k, v);
        }
    }
    // Upcast F32/F16 shards to BF16 if needed.
    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in keys {
        let t = &weights[&k];
        if t.dtype() != DType::BF16 {
            let bf = t.to_dtype(DType::BF16)?;
            weights.insert(k, bf);
        }
    }
    println!(
        "  loaded {} tensors in {:.1}s",
        weights.len(),
        t_load.elapsed().as_secs_f32()
    );

    println!("--- Building encoder ---");
    let mut cfg = Qwen25VLEncoder::config_from_weights(&weights)?;
    cfg.max_seq_len = max_len.max(cfg.max_seq_len);
    println!(
        "  config: layers={} hidden={} heads={}/{} head_dim={} max_seq={}",
        cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len
    );
    let encoder = Qwen25VLEncoder::new(weights, cfg, device.clone());

    println!();
    println!("--- Encoding ---");
    let t_enc = Instant::now();
    let cond = encode_and_trim(&encoder, &pos_ids)?;
    let uncond = encode_and_trim(&encoder, &neg_ids)?;
    println!(
        "  cond: {:?}  uncond: {:?}  in {:.1}s",
        cond.shape().dims(),
        uncond.shape().dims(),
        t_enc.elapsed().as_secs_f32()
    );

    drop(encoder);

    // Save in the exact format qwenimage_gen.rs expects.
    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = HashMap::new();
    out.insert("cond".to_string(), cond);
    out.insert("uncond".to_string(), uncond);
    flame_core::serialization::save_file(&out, Path::new(&out_path))?;

    let dt = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("EMBEDDINGS SAVED: {out_path}");
    println!("Total time:       {dt:.1}s");
    println!("============================================================");
    println!();
    println!("Next: target/release/qwenimage_gen {out_path} <output.png>");

    let _ = device; // keep device alive
    Shape::from_dims(&[1]); // touch an import so unused-warnings are clean
    Ok(())
}
