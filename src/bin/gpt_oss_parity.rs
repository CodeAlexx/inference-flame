//! Per-layer parity test: pure-Rust `GptOssEncoder` vs HF `GptOssForCausalLM`.
//!
//! M2 D5b validation gate. Loads the real `microsoft/Lens/text_encoder/` checkpoint
//! into our Rust encoder, runs a fixed input through both implementations, and
//! reports cosine similarity per selected layer.
//!
//! **Reference captures** must be generated first by running
//! `inference-flame/lens/parity/capture_gpt_oss_ref.py`. The bin reads:
//!   - `input_ids.safetensors`         (I32 token IDs, shape [1, S])
//!   - `attention_mask.safetensors`    (I32, shape [1, S])
//!   - `hidden_layer_{05,11,17,23}.safetensors`  (BF16, shape [1, S, 2880])
//!
//! **Bar**: cos ≥ 0.999 per layer. Lower is a bug.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::{GptOssConfig, GptOssEncoder};

const MODEL_DIR: &str = "/home/alex/.serenity/models/microsoft_lens/text_encoder";
const DEFAULT_CAPTURES_DIR: &str = "/home/alex/EriDiffusion/inference-flame/lens/parity/captures_text";

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    let model_dir = std::env::var("LENS_TE_DIR").unwrap_or_else(|_| MODEL_DIR.to_string());
    let captures_dir = std::env::var("LENS_TE_CAPTURES")
        .unwrap_or_else(|_| DEFAULT_CAPTURES_DIR.to_string());

    println!("--- GPT-OSS encoder parity ---");
    println!("  model_dir:    {model_dir}");
    println!("  captures_dir: {captures_dir}");

    // 1. Load reference inputs + per-layer captures from Python.
    let captures_dir_path = PathBuf::from(&captures_dir);
    if !captures_dir_path.exists() {
        return Err(anyhow!(
            "captures dir not found: {}\nRun: python {}/capture_gpt_oss_ref.py first.",
            captures_dir,
            "/home/alex/EriDiffusion/inference-flame/lens/parity"
        ));
    }

    let input_ids =
        load_single_tensor(&captures_dir_path.join("input_ids.safetensors"), &device)?;
    let attention_mask =
        load_single_tensor(&captures_dir_path.join("attention_mask.safetensors"), &device)?;
    let ids_dims = input_ids.shape().dims().to_vec();
    println!(
        "  input_ids: shape={:?} dtype={:?}",
        ids_dims,
        input_ids.dtype()
    );
    let (b, s) = (ids_dims[0], ids_dims[1]);
    let input_ids = ensure_i32(input_ids)?;
    let attention_mask = ensure_i32(attention_mask)?;

    // Match the Python selected_layers default. Override via env if needed.
    let selected_layers: Vec<usize> = std::env::var("LENS_TE_LAYERS")
        .map(|s| {
            s.split(',')
                .filter_map(|tok| tok.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_else(|_| vec![5, 11, 17, 23]);
    println!("  selected_layers: {selected_layers:?}");

    let ref_tensors: Vec<(usize, Tensor)> = selected_layers
        .iter()
        .map(|&idx| {
            let path = captures_dir_path.join(format!("hidden_layer_{idx:02}.safetensors"));
            let t = load_single_tensor(&path, &device)?;
            Ok::<_, anyhow::Error>((idx, t))
        })
        .collect::<Result<Vec<_>>>()?;
    for (idx, t) in &ref_tensors {
        println!(
            "  ref layer {idx}: shape={:?} dtype={:?}",
            t.shape().dims(),
            t.dtype()
        );
    }

    // 2. Build the Rust encoder and load weights.
    println!("\n--- Loading Rust encoder ---");
    let mut config = GptOssConfig::lens_default();
    config.selected_layer_index = selected_layers.clone();
    let mut encoder = GptOssEncoder::new(config, &device)
        .map_err(|e| anyhow!("GptOssEncoder::new: {e}"))?;
    let stats = encoder
        .load_from_directory(Path::new(&model_dir), &device)
        .map_err(|e| anyhow!("load_from_directory: {e}"))?;
    println!(
        "  loaded: consumed={} mxfp4_dequants={} skipped={}",
        stats.consumed, stats.mxfp4_dequants, stats.skipped
    );

    // 3. Run our forward.
    println!("\n--- Rust forward ---");
    let t0 = std::time::Instant::now();
    let our_captures = encoder
        .encode(&input_ids, &attention_mask)
        .map_err(|e| anyhow!("encode: {e}"))?;
    println!(
        "  encode wall: {:.2}s  (B={b}, S={s}, layers={})",
        t0.elapsed().as_secs_f64(),
        our_captures.len()
    );

    // 4. Compare.
    println!("\n--- Per-layer cosine similarity ---");
    println!(
        "{:>10} | {:>10} | {:>10} | {:>10} | {:>7}",
        "layer", "cos", "max_abs", "mean_abs", "status"
    );
    let mut all_pass = true;
    for ((idx, ref_t), our_t) in ref_tensors.iter().zip(our_captures.iter()) {
        let ref_f32 = ref_t.to_dtype(DType::F32)?.to_vec_f32()?;
        let our_f32 = our_t.to_dtype(DType::F32)?.to_vec_f32()?;
        if ref_f32.len() != our_f32.len() {
            return Err(anyhow!(
                "layer {idx}: ref len {} != our len {}",
                ref_f32.len(),
                our_f32.len()
            ));
        }
        let (cos, max_abs, mean_abs) = cosine_stats(&our_f32, &ref_f32);
        let our_max = our_f32.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let ref_max = ref_f32.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let our_mean = our_f32.iter().map(|v| v.abs()).sum::<f32>() / our_f32.len() as f32;
        let ref_mean = ref_f32.iter().map(|v| v.abs()).sum::<f32>() / ref_f32.len() as f32;
        let status = if cos >= 0.999 { "PASS" } else { "FAIL" };
        if cos < 0.999 {
            all_pass = false;
        }
        println!(
            "{:>10} | {:>10.6} | {:>10.4e} | {:>10.4e} | {:>7}   our[max={:.3e},mean={:.3e}] ref[max={:.3e},mean={:.3e}]",
            idx, cos, max_abs, mean_abs, status,
            our_max, our_mean, ref_max, ref_mean
        );
    }

    println!(
        "\nVerdict: {}",
        if all_pass { "GREEN — all layers cos>=0.999" } else { "FAIL — see table above" }
    );
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}

/// Load a single-tensor safetensors file. The Python writer uses
/// `save_file({"tensor": ...})`, so we extract the "tensor" key.
fn load_single_tensor(
    path: &Path,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let map = flame_core::serialization::load_file(path, device)
        .map_err(|e| anyhow!("load {}: {e}", path.display()))?;
    map.get("tensor")
        .cloned()
        .with_context(|| format!("missing 'tensor' key in {}", path.display()))
}

/// Cast a tensor to I32. The Python parity script (`capture_gpt_oss_ref.py`)
/// stores `input_ids` and `attention_mask` as F32 because the flame-core
/// safetensors loader skips I32/I64 dtypes. We cast back here.
fn ensure_i32(t: Tensor) -> Result<Tensor> {
    if t.dtype() == DType::I32 {
        Ok(t)
    } else {
        t.to_dtype(DType::I32)
            .map_err(|e| anyhow!("ensure_i32: {e}"))
    }
}

/// Cosine similarity + max-abs + mean-abs of (our - ref).
///
/// Returns `(cos, max_abs_diff, mean_abs_diff)` computed in F64 for stability.
fn cosine_stats(ours: &[f32], reference: &[f32]) -> (f64, f32, f32) {
    debug_assert_eq!(ours.len(), reference.len());
    let n = ours.len();
    let mut dot = 0.0f64;
    let mut nn1 = 0.0f64;
    let mut nn2 = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for i in 0..n {
        let a = ours[i] as f64;
        let b = reference[i] as f64;
        dot += a * b;
        nn1 += a * a;
        nn2 += b * b;
        let d = (a - b).abs() as f32;
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
    }
    let cos = if nn1 > 0.0 && nn2 > 0.0 {
        dot / (nn1.sqrt() * nn2.sqrt())
    } else {
        0.0
    };
    let mean_abs = (sum_abs / n as f64) as f32;
    (cos, max_abs, mean_abs)
}
