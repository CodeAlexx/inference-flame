//! Ideogram-4 — text encode stage (separate process, per the memory plan).
//!
//! Loads the Qwen3-VL-8B-Instruct **language tower** (FP8 weight-only e4m3 with
//! per-output-row scale — same scheme as the DiT), encodes one verbatim prompt
//! (the `run_inference.py --no-magic-prompt` path), taps the 13 oracle hidden
//! states, concatenates them along hidden → `llm_features [1, L, 53248]`, and
//! writes the cache that `ideogram4_infer` loads. The encoder is then freed
//! (process exit) so it never shares VRAM with the two resident DiTs. Mirrors the
//! `klein9b_encode` / `boogu_encode` split idiom.
//!
//! Usage:
//!   ideogram4_encode ["a photorealistic red fox sitting in autumn leaves"]
//!
//! Output cache (`output/ideogram4_embeddings.safetensors`):
//!   - `llm_features` : `[1, L, 53248]`  (the 13-tap concat, H-major/tap-minor)
//!   - `num_text`     : `[1]` i32-as-f32 (real text-token count = L)
//!
//! ## The 13 taps (`constants.py::QWEN3_VL_ACTIVATION_LAYERS`)
//!
//! `(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)` — each is the output of
//! decoder layer `idx` BEFORE the final `model.norm` (HF per-decoder-layer
//! `hidden_states`). `Qwen3Encoder::encode_multi_taps` returns them in that order
//! (its extract index i = output of `layer_forward(i)`, which is exactly the
//! reference's "after decoder_layer[idx]").
//!
//! ## Concat order (KEY parity item — `pipeline_ideogram4.py:472-474`)
//!
//! The reference does `stack(taps, dim=0) → permute(1,2,3,0) → reshape(B,L,-1)`,
//! i.e. the last dim is **H-major, tap-minor**: `[h0·t0,h0·t1,…,h0·t12, h1·t0,…]`.
//! This is the OPPOSITE of `Qwen3Encoder::encode`'s tap-major/H-minor concat, so
//! we build the H-major order explicitly here (`stack(taps, 3)` →
//! `[B,L,H,taps]` → reshape `[B,L,H*taps]`). The DiT's `llm_cond_proj` was
//! trained on this order; getting it wrong scrambles the conditioning.
//!
//! ## Chat template (`pipeline_ideogram4.py::_tokenize`)
//!
//! `apply_chat_template([{user: [{text: prompt}]}], add_generation_prompt=True)`
//! with the shipped `chat_template.jinja` (no system, no tools) renders exactly:
//!   `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
//! tokenized with `add_special_tokens=False`.
//!
//! Pure-Rust runtime — no Python.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};

use flame_core::{DType, Shape, Tensor};
use inference_flame::models::qwen3_encoder::{Qwen3Config, Qwen3Encoder};

/// Checkpoint root (the FP8 repo).
const REPO: &str = "/home/alex/.serenity/models/ideogram-4-fp8";
/// Qwen3-VL text encoder component dir (FP8 weight-only).
const TEXT_ENCODER_SUBDIR: &str = "text_encoder";
/// Tokenizer dir.
const TOKENIZER_SUBDIR: &str = "tokenizer";
/// Output cache consumed by `ideogram4_infer`.
const OUTPUT_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ideogram4_embeddings.safetensors";

const DEFAULT_PROMPT: &str = "a photorealistic red fox sitting in autumn leaves";

/// The 13 Qwen3-VL activation layers tapped by Ideogram-4
/// (`constants.py::QWEN3_VL_ACTIVATION_LAYERS`).
const QWEN3_VL_ACTIVATION_LAYERS: [usize; 13] =
    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];

/// safetensors dtype string for weight-only e4m3 FP8.
const FP8_DTYPE: &str = "F8_E4M3";
/// Per-output-row scale key suffix.
const FP8_SCALE_SUFFIX: &str = "_scale"; // keys are `<name>.weight` + `<name>.weight_scale`

/// Resolve the text-encoder shard paths (sharded index, else single file).
fn resolve_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    let single = dir.join("model.safetensors");
    if index.is_file() {
        let bytes = std::fs::read(&index).with_context(|| format!("read {}", index.display()))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse {}", index.display()))?;
        let wm = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("{} missing weight_map", index.display()))?;
        let mut names: Vec<String> = wm
            .values()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        names.sort();
        names.dedup();
        Ok(names.into_iter().map(|s| dir.join(s)).collect())
    } else if single.is_file() {
        Ok(vec![single])
    } else {
        Err(anyhow!("no text_encoder checkpoint in {}", dir.display()))
    }
}

/// Load the FP8-per-row Qwen3-VL language tower → BF16 `model.*` weight map.
///
/// FP8 linear weights (`*.weight` with dtype F8_E4M3) are dequantized per-row:
/// `weight_bf16 = fp8.to(bf16) * weight_scale.to(bf16)[:, None]` (the same scheme
/// as `Ideogram4RawWeight::to_bf16_tensor`). `serialization::load_file` is NOT
/// usable here — it dequantizes FP8 with a per-TENSOR scalar (reads only the
/// first f32 of the `[out]` scale row), which is wrong for per-row scale.
///
/// Keys are remapped `language_model.* → model.*` (Qwen3-VL strips the top
/// `model.`, leaving `language_model.*`; `Qwen3Encoder` wants bare `model.*`).
/// `visual.*` and `lm_head.*` are dropped.
fn load_fp8_language_tower(
    dir: &Path,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let shards = resolve_shards(dir)?;
    let mut out: HashMap<String, Tensor> = HashMap::new();

    for shard in &shards {
        let mmap = eri_safetensors::MmapFile::open_path(shard)
            .map_err(|e| anyhow!("mmap {}: {e}", shard.display()))?;

        // Index the scale tensors present in this shard.
        let scale_keys: Vec<String> = mmap
            .tensors
            .keys()
            .filter(|k| k.ends_with(FP8_SCALE_SUFFIX) && k.contains(".weight"))
            .cloned()
            .collect();

        for (name, tref) in &mmap.tensors {
            // Skip scale companions (folded into their FP8 weight).
            if name.ends_with(FP8_SCALE_SUFFIX) && name.contains(".weight") {
                continue;
            }
            // Keep language tower only; remap language_model.* -> model.*.
            let dst = if let Some(rest) = name.strip_prefix("language_model.") {
                format!("model.{rest}")
            } else {
                // visual.* / lm_head.* / anything else -> dropped.
                continue;
            };

            let bytes = mmap
                .tensor_bytes(name)
                .ok_or_else(|| anyhow!("missing bytes for {name}"))?;
            let shape = tref.shape.clone();

            let tensor = match tref.dtype.as_str() {
                FP8_DTYPE => {
                    // FP8 weight: per-row dequant.
                    let scale_key = format!("{name}{FP8_SCALE_SUFFIX}");
                    if !scale_keys.iter().any(|k| k == &scale_key) {
                        return Err(anyhow!("FP8 weight {name} has no {scale_key}"));
                    }
                    let scale_bytes = mmap
                        .tensor_bytes(&scale_key)
                        .ok_or_else(|| anyhow!("missing scale bytes {scale_key}"))?;
                    let scale_f32: Vec<f32> = scale_bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    if shape.len() != 2 || scale_f32.len() != shape[0] {
                        return Err(anyhow!(
                            "FP8 {name}: shape {shape:?} / scale len {} mismatch",
                            scale_f32.len()
                        ));
                    }
                    let gpu: CudaSlice<u8> = device
                        .htod_copy(bytes.to_vec())
                        .map_err(|e| anyhow!("htod fp8 {name}: {e:?}"))?;
                    // raw dequant (scale=1.0) then per-row broadcast multiply.
                    let raw = flame_core::ops::fused_inference::dequant_fp8_to_bf16(
                        &gpu,
                        1.0,
                        Shape::from_dims(&shape),
                        device,
                    )
                    .map_err(|e| anyhow!("dequant {name}: {e}"))?;
                    let scale_col = Tensor::from_vec_dtype(
                        scale_f32,
                        Shape::from_dims(&[shape[0], 1]),
                        device.clone(),
                        DType::BF16,
                    )
                    .map_err(|e| anyhow!("scale tensor {name}: {e}"))?;
                    raw.mul(&scale_col).map_err(|e| anyhow!("rowscale {name}: {e}"))?
                }
                "BF16" => {
                    let u16s: Vec<u16> = bytes
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let mut t =
                        Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())
                            .map_err(|e| anyhow!("alloc bf16 {name}: {e}"))?;
                    t.copy_from_bf16_slice(&u16s)
                        .map_err(|e| anyhow!("copy bf16 {name}: {e}"))?;
                    t
                }
                "F32" => {
                    let u16s: Vec<u16> = bytes
                        .chunks_exact(4)
                        .map(|c| {
                            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            half::bf16::from_f32(f).to_bits()
                        })
                        .collect();
                    let mut t =
                        Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())
                            .map_err(|e| anyhow!("alloc f32->bf16 {name}: {e}"))?;
                    t.copy_from_bf16_slice(&u16s)
                        .map_err(|e| anyhow!("copy f32->bf16 {name}: {e}"))?;
                    t
                }
                // Integer / bool (e.g. computed rotary caches) — skipped; the
                // encoder builds its own RoPE.
                _ => continue,
            };
            out.insert(dst, tensor);
        }
    }

    // Validate the 398 canonical Qwen3 keys resolved (1 embed + 36×11 + 1 norm).
    let expected = inference_flame::models::qwen3_encoder::expected_weight_keys(36);
    let missing: Vec<&String> = expected.iter().filter(|k| !out.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "text encoder: {} language-tower key(s) missing after load (e.g. {:?})",
            missing.len(),
            missing.iter().take(5).collect::<Vec<_>>()
        ));
    }
    Ok(out)
}

fn main() -> Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("============================================================");
    println!("Ideogram-4 — Text Encode (Qwen3-VL-8B language tower, FP8)");
    println!("============================================================");
    println!("  Prompt: {prompt}");

    let device: Arc<CudaDevice> = CudaDevice::new(0).context("cuda device 0")?;

    // --- Tokenize with the chat template (no system, add_generation_prompt) ---
    let tok_path = Path::new(REPO).join(TOKENIZER_SUBDIR).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("load tokenizer {}: {e}", tok_path.display()))?;
    // Mirror apply_chat_template([{user:[{text:prompt}]}], add_generation_prompt=True)
    // over the shipped jinja (no system / no tools):
    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let enc = tokenizer
        .encode(formatted.as_str(), false) // add_special_tokens=False
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let token_ids: Vec<i32> = enc.get_ids().iter().map(|&id| id as i32).collect();
    let num_text = token_ids.len();
    println!("  text tokens (L): {num_text}");

    // --- Load FP8 language tower → BF16, construct encoder ---
    println!("\n--- Loading Qwen3-VL-8B language tower (FP8 → BF16 dequant) ---");
    let t0 = Instant::now();
    let te_dir = Path::new(REPO).join(TEXT_ENCODER_SUBDIR);
    let weights = load_fp8_language_tower(&te_dir, &device)?;
    let mut config = Qwen3Config::qwen3_vl_text();
    config.extract_layers = QWEN3_VL_ACTIVATION_LAYERS.to_vec();
    let encoder = Qwen3Encoder::new(weights, config, device.clone());
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Encode: 13 taps, H-major/tap-minor concat ---
    println!("\n--- Encoding (13 taps) ---");
    let t0 = Instant::now();
    let taps = encoder
        .encode_multi_taps(&token_ids)
        .map_err(|e| anyhow!("encode_multi_taps: {e}"))?; // 13 × [1, L, 4096]
    if taps.len() != QWEN3_VL_ACTIVATION_LAYERS.len() {
        return Err(anyhow!(
            "expected {} taps, got {}",
            QWEN3_VL_ACTIVATION_LAYERS.len(),
            taps.len()
        ));
    }
    let d0 = taps[0].shape().dims().to_vec();
    let (b, l, h) = (d0[0], d0[1], d0[2]);
    let num_taps = taps.len();

    // Reference: stack(taps, dim=0) -> permute(1,2,3,0) -> reshape(B,L,H*taps).
    // Equivalent: stack along a NEW last axis -> [B,L,H,taps] -> reshape
    // [B,L,H*taps]. stack(taps, 3) gives [B,L,H,taps] directly (H-major,
    // tap-minor) — exactly the reference's last-dim element order.
    let stacked = Tensor::stack(&taps, 3).map_err(|e| anyhow!("stack taps: {e}"))?;
    let llm_features = stacked
        .reshape(&[b, l, h * num_taps])
        .map_err(|e| anyhow!("reshape concat: {e}"))?
        .contiguous()
        .map_err(|e| anyhow!("contiguous concat: {e}"))?;
    let feat_dims = llm_features.shape().dims().to_vec();
    println!(
        "  llm_features: {:?}  (expected [1,{l},{}])  {:.1}s",
        feat_dims,
        h * num_taps,
        t0.elapsed().as_secs_f32()
    );
    if feat_dims != vec![1, l, h * num_taps] || h * num_taps != 53248 {
        return Err(anyhow!(
            "llm_features shape {feat_dims:?} != [1,{l},53248] (h={h} taps={num_taps})"
        ));
    }

    // --- Save ---
    println!("\n--- Saving embeddings ---");
    if let Some(parent) = Path::new(OUTPUT_PATH).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let num_text_t = Tensor::from_vec(
        vec![num_text as f32],
        Shape::from_dims(&[1]),
        device.clone(),
    )
    .map_err(|e| anyhow!("num_text tensor: {e}"))?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("llm_features".to_string(), llm_features);
    tensors.insert("num_text".to_string(), num_text_t);
    flame_core::serialization::save_file(&tensors, OUTPUT_PATH)
        .map_err(|e| anyhow!("save embeddings: {e}"))?;
    println!("  saved to {OUTPUT_PATH}");

    // Encoder freed on scope exit (then process exits) — DiTs never co-reside.
    println!("\nTotal time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
