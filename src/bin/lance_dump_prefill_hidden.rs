//! Lance encoder-forward parity capture (Rust side).
//!
//! Mirrors `/tmp/lance_py_prefill.py`: runs Lance prefill for the same fixed
//! prompt, captures the final hidden state after the LLM stack, saves to
//! safetensors as `rust_hidden_final` for numeric comparison with Python.
//!
//! Differs from `parity_lance.rs` in scope — that captures top-level latent
//! noise + final denoised latent + VAE output; this captures the LLM hidden
//! state mid-pipeline.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use flame_core::{DType, Shape, Tensor};
use inference_flame::models::lance::{Lance, LanceConfig};
use tokenizers::Tokenizer;

const TOK_PATH: &str = "/home/alex/.serenity/models/lance/Lance_3B/tokenizer.json";
const MODEL_PATH: &str = "/home/alex/.serenity/models/lance/Lance_3B";
const OUT_PATH: &str = "/tmp/lance_rust_prefill_hidden.safetensors";

const T2I_SYS: &str = "Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";
const PROMPT: &str = "a small red apple on a white plate";

fn main() -> Result<()> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    let tok = Tokenizer::from_file(PathBuf::from(TOK_PATH))
        .map_err(|e| anyhow::anyhow!("tok load: {e}"))?;
    let template = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        T2I_SYS, PROMPT
    );
    let ids: Vec<i32> = tok
        .encode(template, false)
        .map_err(|e| anyhow::anyhow!("tok encode: {e}"))?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    println!("[rust] template tokens: {}", ids.len());
    println!("[rust] first 10: {:?}", &ids[..10.min(ids.len())]);

    let l = ids.len();

    let mut cfg = LanceConfig::default_3b(device.clone());
    cfg.num_inference_steps = 1; // unused for prefill
    let cfg = Arc::new(cfg);

    println!("[rust] loading Lance...");
    let lance = Lance::load(&PathBuf::from(MODEL_PATH), cfg.clone(), &device)?;
    println!("[rust] Lance loaded");

    let max_pos = l + 1000 + 64;
    let mrope = lance.precompute_mrope(max_pos, cfg.dtype)?;

    // Build token tensor + cache
    let token_tensor = Tensor::from_vec(
        ids.iter().map(|&i| i as f32).collect(),
        Shape::from_dims(&[l]),
        device.clone(),
    )?
    .to_dtype(DType::I32)?;
    let mut cache = lance.new_kv_cache();

    println!("[rust] running prefill ...");
    let hidden = lance.prefill_text_context_capture(&token_tensor, &mrope, &mut cache)?;
    let dims = hidden.shape().dims().to_vec();
    println!("[rust] final hidden shape={dims:?} dtype={:?}", hidden.dtype());

    // Print first/last for sanity
    let host = hidden.to_dtype(DType::F32)?.to_vec_f32()?;
    // hidden shape is [1, L, H]
    let h = dims[dims.len() - 1];
    let l_out = dims[dims.len() - 2];
    let first5: Vec<f32> = host[..5].to_vec();
    let last5: Vec<f32> = host[(l_out - 1) * h..(l_out - 1) * h + 5].to_vec();
    println!("[rust] hidden[0,:5]={:?}", first5);
    println!("[rust] hidden[L-1,:5]={:?}", last5);
    let norm: f32 = host.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("[rust] hidden.norm()={:.4}", norm);

    // Save — strip the leading B=1 dim to match Python's [L, H]
    let hidden_2d = hidden.reshape(&[l_out, h])?.to_dtype(DType::F32)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("rust_hidden_final".to_string(), hidden_2d);
    let ids_tensor = Tensor::from_vec(
        ids.iter().map(|&i| i as f32).collect(),
        Shape::from_dims(&[l]),
        device.clone(),
    )?
    .to_dtype(DType::I32)?;
    tensors.insert("rust_input_ids".to_string(), ids_tensor);

    flame_core::serialization::save_file(&tensors, OUT_PATH)?;
    println!("[rust] saved {OUT_PATH}");
    Ok(())
}
