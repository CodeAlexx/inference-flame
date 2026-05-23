//! Microsoft Lens 1024² T2I inference.
//!
//! M2 D7 path: full encoder → DiT → VAE end-to-end with a real prompt.
//! Memory strategy is **sequential**: load the GPT-OSS encoder, tokenize +
//! encode the prompt to per-layer hidden states, **drop the encoder**, then
//! load the DiT and run the 20-step denoise. Both models cannot coexist on
//! a 24 GB card; D6 (BlockOffloader for concurrent residency) is the next
//! chunk.
//!
//! Modes:
//!   * Default (smoke): real encoder → DiT → VAE → PNG.
//!   * `--use-cached-features`: skip the encoder, feed zeroed text features
//!     (the M1 stepping-stone path). Used for parity testing against the
//!     `lens/parity/captures/*` zeroed-feature Python references.
//!   * `--parity`: load `hidden_states_pre_step_NN.safetensors` and compare
//!     our DiT forward output to the captured `noise_pred` (DiT-only parity).
//!
//! Per CONTEXT tenet, this binary disables autograd at `main()` to prevent
//! the 48-block × 20-step autograd tape from retaining (the HiDream-O1 OOM
//! pattern from 2026-05-09).

use anyhow::{anyhow, bail, Context, Result};
use flame_core::device::Device as FlameDevice;
use flame_core::serialization::load_file;
use flame_core::{global_cuda_device, trim_cuda_mempool, DType, Shape, Tensor};
use inference_flame::models::lens_dit::{LensDiTConfig, LensTransformer2DModel};
use inference_flame::models::{GptOssConfig, GptOssEncoder};
use inference_flame::sampling::lens_flowmatch::{
    apply_exponential_shift, build_sigmas, compute_empirical_mu, euler_step,
};
use inference_flame::vae::lens_vae_wrapper::LensVaeWrapper;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    transformer_dir: PathBuf,
    text_encoder_dir: PathBuf,
    tokenizer_dir: PathBuf,
    vae: PathBuf,
    captures_dir: PathBuf,
    output: PathBuf,
    prompt: String,
    negative_prompt: String,
    steps: usize,
    cfg: f32,
    seed: u64,
    width: usize,
    height: usize,
    max_text_len: usize,
    parity: bool,
    /// M1 stepping-stone: bypass the encoder and feed zeroed text features.
    /// Kept available for parity testing against `lens/parity/captures/`.
    use_cached_features: bool,
}

impl Args {
    fn default_output() -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        PathBuf::from(format!(
            "/home/alex/EriDiffusion/inference-flame/output/lens_{stamp}.png"
        ))
    }
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        transformer_dir: PathBuf::from(
            "/home/alex/.serenity/models/microsoft_lens/transformer",
        ),
        text_encoder_dir: PathBuf::from(
            "/home/alex/.serenity/models/microsoft_lens/text_encoder",
        ),
        tokenizer_dir: PathBuf::from(
            "/home/alex/.serenity/models/microsoft_lens/tokenizer",
        ),
        vae: PathBuf::from(
            "/home/alex/.serenity/models/microsoft_lens/vae/diffusion_pytorch_model.safetensors",
        ),
        captures_dir: PathBuf::from(
            "/home/alex/EriDiffusion/inference-flame/lens/parity/captures",
        ),
        output: Args::default_output(),
        prompt: "A scenic landscape with a serene lake".to_string(),
        negative_prompt: String::new(),
        steps: 20,
        cfg: 5.0,
        seed: 42,
        width: 1024,
        height: 1024,
        max_text_len: 512,
        parity: false,
        use_cached_features: false,
    };
    let mut i = 1;
    while i < argv.len() {
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| {
                anyhow!(
                    "missing value for {}",
                    argv.get(*i - 1).cloned().unwrap_or_default()
                )
            })
        };
        match argv[i].as_str() {
            "--transformer-dir" => a.transformer_dir = PathBuf::from(take(&mut i)?),
            "--text-encoder-dir" => a.text_encoder_dir = PathBuf::from(take(&mut i)?),
            "--tokenizer-dir" => a.tokenizer_dir = PathBuf::from(take(&mut i)?),
            "--vae" => a.vae = PathBuf::from(take(&mut i)?),
            "--captures-dir" => a.captures_dir = PathBuf::from(take(&mut i)?),
            "--output" => a.output = PathBuf::from(take(&mut i)?),
            "--prompt" => a.prompt = take(&mut i)?,
            "--negative-prompt" => a.negative_prompt = take(&mut i)?,
            "--steps" => a.steps = take(&mut i)?.parse()?,
            "--cfg" | "--guidance" => a.cfg = take(&mut i)?.parse()?,
            "--seed" => a.seed = take(&mut i)?.parse()?,
            "--width" => a.width = take(&mut i)?.parse()?,
            "--height" => a.height = take(&mut i)?.parse()?,
            "--max-text-len" => a.max_text_len = take(&mut i)?.parse()?,
            "--parity" => a.parity = true,
            "--use-cached-features" => a.use_cached_features = true,
            "-h" | "--help" => {
                eprintln!(
                    "lens_infer [--transformer-dir DIR] [--text-encoder-dir DIR] \
                     [--tokenizer-dir DIR] [--vae PATH] [--captures-dir DIR] \
                     [--output PNG] [--prompt STR] [--negative-prompt STR] \
                     [--steps N] [--cfg G] [--seed S] [--width W] [--height H] \
                     [--max-text-len N] [--parity] [--use-cached-features]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
        i += 1;
    }
    Ok(a)
}

// ---------------------------------------------------------------------------
// Cosine-similarity report against a single captured tensor.
//
// Mirrors `flame_core::parity::ParityHarness::compare` but accepts a tensor
// loaded from any safetensors file with any key — the harness's public API
// keys reports by name from a single dump file, which doesn't match our
// per-step captures (we have 60+ separate files). The math is the same.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ParityMetric {
    cos: f32,
    max_abs: f32,
    mean_abs: f32,
}

fn to_f32_vec(t: &Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(if t.dtype() == DType::F32 {
        t.to_vec()?
    } else {
        t.to_dtype(DType::F32)?.to_vec()?
    })
}

fn compare_tensors(ours: &Tensor, reference: &Tensor) -> Result<ParityMetric> {
    if ours.shape().dims() != reference.shape().dims() {
        bail!(
            "shape mismatch: ours {:?} vs ref {:?}",
            ours.shape().dims(),
            reference.shape().dims()
        );
    }
    let a = to_f32_vec(ours)?;
    let b = to_f32_vec(reference)?;
    let n = a.len();
    let (mut dot, mut sa2, mut sb2) = (0f64, 0f64, 0f64);
    let (mut max_abs, mut sum_abs) = (0f32, 0f64);
    for i in 0..n {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        sa2 += av * av;
        sb2 += bv * bv;
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
    }
    let denom = (sa2 * sb2).sqrt();
    let cos = if denom > 0.0 {
        (dot / denom) as f32
    } else {
        f32::NAN
    };
    let mean_abs = if n > 0 { (sum_abs / n as f64) as f32 } else { 0.0 };
    Ok(ParityMetric {
        cos,
        max_abs,
        mean_abs,
    })
}

fn load_single_tensor(
    path: &Path,
    expected_key: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let map = load_file(path, device)
        .with_context(|| format!("loading {}", path.display()))?;
    map.get(expected_key)
        .cloned()
        .ok_or_else(|| anyhow!("key {expected_key:?} not in {}", path.display()))
}

// ---------------------------------------------------------------------------
// Chat-template rendering + tokenization for the Lens text encoder.
//
// Pure-Rust render of the GPT-OSS Harmony chat template for our specific
// single-prompt T2I use case. The full upstream Jinja template at
// `/home/alex/.serenity/models/microsoft_lens/tokenizer/chat_template.jinja`
// handles tool calls / browsing / etc., but Lens's `LensPipeline._build_chat_inputs`
// passes a fixed 3-message conversation (system / user / assistant-with-thinking),
// no tools, no developer message, `add_generation_prompt=false`. Walking the
// template by hand for those inputs produces the literal string below.
//
// Verified against `transformers.PreTrainedTokenizerFast.apply_chat_template`
// for the same inputs on 2026-05-23: 126 tokens for a 7-word prompt, first
// IDs `[200006 (<|start|>), 17360 ("system"), 200008 (<|message|>), ...]`.
//
// The Jinja template extracts the system message into the `developer_message`
// slot, so the rendered text contains BOTH a `<|start|>system<|message|>`
// block (with default ChatGPT preamble + current date) AND a
// `<|start|>developer<|message|>` block (with `# Instructions\n\n<system content>\n\n`).
// This matches what `LensPipeline` actually sends to the encoder.
// ---------------------------------------------------------------------------

const CHAT_SYSTEM: &str = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background.";
const CHAT_ASSISTANT_THINKING: &str = "Need to generate one image according to the description.";
/// `DEFAULT_TXT_OFFSET` from `LensPipeline` (pipeline.py:60). Trims the
/// chat-template overhead so the DiT only sees the prompt-influenced tail of
/// the encoder hidden states.
const DEFAULT_TXT_OFFSET: usize = 97;

/// Compute the UTC date as `YYYY-MM-DD` from epoch seconds (no chrono dep).
/// Mirrors `strftime_now("%Y-%m-%d")` in the chat template (the Python side
/// uses local time but `transformers` jinja env wires UTC; the day boundary
/// difference is at most ±1 day which can shift token IDs by ~1. For real
/// parity work, regenerate the Python reference on the same day.)
fn today_utc_yyyy_mm_dd() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let mut days = secs / 86400;
    // Civil-from-days algorithm (Howard Hinnant). Days from 1970-01-01.
    days += 719468;
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = (days - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Render the Lens chat template for a single prompt and return the string
/// already post-`.split("<|return|>")[0]` (i.e. the encoder input text).
///
/// Matches `LensPipeline._build_chat_inputs` (pipeline.py:163-187) for the
/// pipeline's exact 3-message conversation.
fn render_lens_chat_template(prompt: &str) -> String {
    let date = today_utc_yyyy_mm_dd();
    // System block (with the default ChatGPT preamble + current date + reasoning).
    let mut s = String::new();
    s.push_str("<|start|>system<|message|>");
    s.push_str("You are ChatGPT, a large language model trained by OpenAI.\n");
    s.push_str("Knowledge cutoff: 2024-06\n");
    s.push_str("Current date: ");
    s.push_str(&date);
    s.push_str("\n\n");
    s.push_str("Reasoning: medium\n\n");
    s.push_str("# Valid channels: analysis, commentary, final. Channel must be included for every message.");
    s.push_str("<|end|>");
    // Developer block (jinja extracts messages[0] into developer_message slot
    // when role=='system'; "# Instructions\n\n" + content + "\n\n").
    s.push_str("<|start|>developer<|message|>");
    s.push_str("# Instructions\n\n");
    s.push_str(CHAT_SYSTEM);
    s.push_str("\n\n");
    s.push_str("<|end|>");
    // User block.
    s.push_str("<|start|>user<|message|>");
    s.push_str(prompt);
    s.push_str("<|end|>");
    // Assistant with thinking — last message, add_generation_prompt=false, content="".
    // Renders: analysis (with thinking) <|end|> + final (with empty content) <|return|>.
    // Then .split("<|return|>")[0] keeps the leading slice up to (not incl) <|return|>.
    s.push_str("<|start|>assistant<|channel|>analysis<|message|>");
    s.push_str(CHAT_ASSISTANT_THINKING);
    s.push_str("<|end|>");
    s.push_str("<|start|>assistant<|channel|>final<|message|>");
    // (we stop here, mirroring the .split("<|return|>")[0] trim)
    s
}

/// Tokenize `text` with the GPT-OSS tokenizer. Pads/truncates to `max_len`.
/// Returns (input_ids[1, S] I32, attention_mask[1, S] I32, real_len).
///
/// `add_special_tokens=false`: the tokenizer's post-processor is `ByteLevel`
/// which does NOT add BOS; the rendered chat-template string already contains
/// the literal `<|start|>` / `<|end|>` etc. specials, which the tokenizer
/// resolves via its `added_tokens` table.
fn tokenize_chat_text(
    tokenizer: &Tokenizer,
    text: &str,
    max_len: usize,
    pad_id: u32,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor, usize)> {
    let enc = tokenizer
        .encode(text, /*add_special_tokens=*/ false)
        .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
    let ids_u32 = enc.get_ids();
    let truncated = ids_u32.len().min(max_len);
    let mut ids: Vec<f32> = Vec::with_capacity(max_len);
    let mut mask: Vec<f32> = Vec::with_capacity(max_len);
    for &id in &ids_u32[..truncated] {
        ids.push(id as f32);
        mask.push(1.0);
    }
    while ids.len() < max_len {
        ids.push(pad_id as f32);
        mask.push(0.0);
    }
    let real_len = truncated;
    let shape = Shape::from_dims(&[1, max_len]);
    let ids_t = Tensor::from_vec(ids, shape.clone(), device.clone())?
        .to_dtype(DType::I32)?;
    let mask_t = Tensor::from_vec(mask, shape, device.clone())?
        .to_dtype(DType::I32)?;
    Ok((ids_t, mask_t, real_len))
}

/// Encode a single prompt to per-layer hidden states via the GPT-OSS encoder.
///
/// Returns `(features, mask, real_len, raw_seq_len)`:
///   * `features`: Vec of N selected layers, each `[1, S_post, 2880]` BF16
///     where `S_post = raw_seq_len - DEFAULT_TXT_OFFSET` (or 0 if too short).
///   * `mask`: `[1, S_post]` BF16 keep-mask (1.0 = real token, 0.0 = padding).
///   * `real_len`: number of non-padding tokens in the post-offset window
///     (clamped to >= 0).
///   * `raw_seq_len`: tokenizer output length pre-offset (for diagnostics).
fn encode_prompt_with_encoder(
    encoder: &mut GptOssEncoder,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_len: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<(Vec<Tensor>, Tensor, usize, usize)> {
    let pad_id: u32 = 199999; // <|endoftext|>, per tokenizer_config.json
    let text = render_lens_chat_template(prompt);
    let (input_ids, attn_mask, real_len_raw) =
        tokenize_chat_text(tokenizer, &text, max_len, pad_id, device)?;

    let raw_seq_len = max_len;
    eprintln!(
        "  tokenize: text_chars={} tokens={} (real={}, padded_to={})",
        text.chars().count(),
        max_len,
        real_len_raw,
        max_len
    );

    let captures = encoder
        .encode(&input_ids, &attn_mask)
        .map_err(|e| anyhow!("encoder.encode failed: {e}"))?;
    // captures: Vec<[1, S, 2880]> BF16.

    // Trim chat-template overhead. Match pipeline.py:196-205: if S > offset,
    // slice [:, offset:, :]; else return zero-shape features.
    if raw_seq_len > DEFAULT_TXT_OFFSET {
        let post_len = raw_seq_len - DEFAULT_TXT_OFFSET;
        let trimmed: Vec<Tensor> = captures
            .iter()
            .map(|f| f.narrow(1, DEFAULT_TXT_OFFSET, post_len))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("feature narrow failed: {e}"))?;
        // Mask post-offset: I32 [1, S_post] → BF16 keep-mask.
        let mask_i32 = attn_mask
            .narrow(1, DEFAULT_TXT_OFFSET, post_len)
            .map_err(|e| anyhow!("mask narrow failed: {e}"))?;
        let mask_bf16 = mask_i32
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("mask cast failed: {e}"))?;
        // real_len in the post-offset window.
        let post_real = real_len_raw.saturating_sub(DEFAULT_TXT_OFFSET);
        // Cast features to BF16 (encoder returns BF16 already, but be defensive).
        let trimmed_bf16: Vec<Tensor> = trimmed
            .into_iter()
            .map(|t| {
                if t.dtype() == DType::BF16 {
                    Ok(t)
                } else {
                    t.to_dtype(DType::BF16).map_err(|e| anyhow!("feature cast: {e}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((trimmed_bf16, mask_bf16, post_real, raw_seq_len))
    } else {
        // Degenerate short-prompt branch. Match Python's zero-shape behavior.
        let zero_feat_shape = Shape::from_dims(&[1, 0, captures[0].shape().dims()[2]]);
        let zero_features: Vec<Tensor> = (0..captures.len())
            .map(|_| Tensor::zeros_dtype(zero_feat_shape.clone(), DType::BF16, device.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("zero feature alloc failed: {e}"))?;
        let zero_mask = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 0]),
            DType::BF16,
            device.clone(),
        )
        .map_err(|e| anyhow!("zero mask alloc failed: {e}"))?;
        Ok((zero_features, zero_mask, 0, raw_seq_len))
    }
}

// ---------------------------------------------------------------------------
// Streaming per-tensor BF16 loader for the Lens DiT.
//
// Background: microsoft/Lens ships its DiT as F32 weights (~16.4 GB across
// 2 safetensors shards). The model destination is BF16 (~8.2 GB). Pre-streaming
// the loader used `flame_core::serialization::load_file` which materializes
// an entire shard's tensors as F32 on GPU (~9.5 GB) before per-key BF16
// conversion. Peak resident = ~9.5 GB BF16 dest + ~9.5 GB F32 source = ~19.4
// GB, which is fine on its own but leaves zero headroom for the 1024² SDPA
// scratch (~2.7 GB) and OOMs on a 24 GB card.
//
// Streaming path (this function): open each shard with `eri_safetensors::
// MmapFile`, iterate its tensor index, and for each entry upload ONE tensor
// to GPU as F32 (or BF16 if on-disk), then immediately hand it to
// `partial_load_lens` (which drains it via `HashMap::remove`, casts to BF16
// inside the model, drops the F32 source). Peak transient F32-on-GPU after
// the model is built ≈ max(per-tensor) ≈ 55 MB (largest single weight is
// `txt_mlp.w*` at 2880×4096 F32 = 47 MB). Total load peak drops to roughly
// 8.2 GB BF16 destinations + ~55 MB working buffer ≈ 8.3 GB.
//
// On-disk dtypes accepted: F32 (the actual Lens distribution) and BF16
// (defensive — covers a future re-packed checkpoint without code changes).
// F16/FP8 raise an error: this loader is dedicated to the Lens DiT, where
// those dtypes are not in scope.
// ---------------------------------------------------------------------------

fn load_lens_dit_sharded(
    model: &mut LensTransformer2DModel,
    transformer_dir: &Path,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    use eri_safetensors::MmapFile;

    let mut shards: Vec<PathBuf> = std::fs::read_dir(transformer_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!(
            "no .safetensors files in transformer dir: {}",
            transformer_dir.display()
        );
    }

    let mut consumed_total: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_uploaded: usize = 0;
    let mut total_skipped_dtype: usize = 0;

    for shard in &shards {
        eprintln!("  shard: {}", shard.display());
        flame_core::trim_cuda_mempool(0);

        let mmap = MmapFile::open_path(shard)
            .map_err(|e| anyhow!("mmap open {}: {:?}", shard.display(), e))?;
        eprintln!("    mmap'd {} tensor entries", mmap.tensors.len());

        // Pre-sort key order so the log lines are reproducible across runs.
        let mut keys: Vec<&String> = mmap.tensors.keys().collect();
        keys.sort();

        let mut uploaded_this_shard: usize = 0;
        for name in keys {
            if consumed_total.contains(name) {
                // Already taken from a prior shard. Skip without uploading.
                continue;
            }
            let tref = &mmap.tensors[name];
            let bytes = mmap.tensor_bytes(name).ok_or_else(|| {
                anyhow!("mmap missing bytes for '{}' in {}", name, shard.display())
            })?;

            // Parse this tensor's bytes into a Vec<f32> on host (one tensor's
            // worth — ~50 MB max). The Vec<f32> is consumed by Tensor::from_vec
            // and dropped immediately after the GPU upload completes.
            let shape = flame_core::Shape::from_dims(&tref.shape);
            let host_f32: Vec<f32> = match tref.dtype.as_str() {
                "F32" => {
                    let mut v = vec![0.0f32; shape.elem_count()];
                    if bytes.len() != v.len() * 4 {
                        bail!(
                            "tensor '{}' F32 byte length mismatch: {} bytes vs {} elems * 4",
                            name,
                            bytes.len(),
                            v.len()
                        );
                    }
                    for (dst, chunk) in v.iter_mut().zip(bytes.chunks_exact(4)) {
                        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    v
                }
                "BF16" => {
                    let mut v = vec![0.0f32; shape.elem_count()];
                    if bytes.len() != v.len() * 2 {
                        bail!(
                            "tensor '{}' BF16 byte length mismatch: {} bytes vs {} elems * 2",
                            name,
                            bytes.len(),
                            v.len()
                        );
                    }
                    for (dst, chunk) in v.iter_mut().zip(bytes.chunks_exact(2)) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        *dst = half::bf16::from_bits(bits).to_f32();
                    }
                    v
                }
                other => {
                    // Lens ships F32; future re-packs may ship BF16. Anything
                    // else (F16, FP8) isn't in scope for this binary and is
                    // almost certainly a wrong-file caller error.
                    total_skipped_dtype += 1;
                    eprintln!(
                        "    WARN: skipping '{}' with unsupported on-disk dtype {} (Lens loader accepts F32/BF16 only)",
                        name, other
                    );
                    continue;
                }
            };

            // Upload to GPU as F32 (the model's `partial_load_lens` does the
            // BF16 cast inside `copy_weight_from` / `to_dtype`).
            let src = Tensor::from_vec(host_f32, shape, device.clone())
                .with_context(|| format!("upload '{}' as F32 to GPU", name))?;

            // Single-entry HashMap so partial_load_lens drains and drops it
            // before we move to the next tensor. After this block exits,
            // `one` is empty and `src` has been moved into and consumed by
            // partial_load_lens (which casts to BF16 and stores into model).
            let mut one: HashMap<String, Tensor> = HashMap::with_capacity(1);
            one.insert(name.clone(), src);
            partial_load_lens(model, &mut one, &mut consumed_total)?;
            // If the key wasn't consumed by partial_load_lens, the model
            // doesn't want it. Drop the leftover entry so the F32 GPU buffer
            // is released here, not at end-of-shard.
            drop(one);
            uploaded_this_shard += 1;
        }
        total_uploaded += uploaded_this_shard;
        eprintln!(
            "    uploaded {} tensors from this shard (total consumed: {})",
            uploaded_this_shard,
            consumed_total.len()
        );

        drop(mmap);
        flame_core::trim_cuda_mempool(0);
    }
    eprintln!(
        "  streaming load done: {} tensors uploaded, {} skipped (dtype), {} consumed total (expected 1264)",
        total_uploaded,
        total_skipped_dtype,
        consumed_total.len()
    );

    // Verify completeness: re-build the expected key set and diff.
    let expected = expected_lens_keys(model);
    let missing: Vec<&String> = expected
        .iter()
        .filter(|k| !consumed_total.contains(*k))
        .collect();
    if !missing.is_empty() {
        bail!(
            "Lens load: {} expected keys never consumed across shards. First few: {:?}",
            missing.len(),
            missing.iter().take(8).collect::<Vec<_>>()
        );
    }
    Ok(())
}

/// Copy whatever keys in `weights` match the model's expected schema; record
/// each match in `consumed`. Unmatched entries in `weights` are ignored
/// (assumed to live in another shard). Missing entries in the expected
/// schema are also ignored at this stage — completeness is verified after
/// all shards have been processed.
///
/// IMPORTANT: this REMOVES entries from `weights` as they are consumed so
/// that the source F32 storage is dropped immediately after the BF16
/// conversion. Without this, holding ~10 GB of F32 source tensors across
/// the whole copy phase puts us at ~19 GB resident (8.2 BF16 model + 10 GB
/// F32 leftovers) and the subsequent SDPA forward OOMs on a 24 GB GPU.
fn partial_load_lens(
    model: &mut LensTransformer2DModel,
    weights: &mut HashMap<String, Tensor>,
    consumed: &mut std::collections::HashSet<String>,
) -> Result<()> {
    use flame_core::linear::Linear;

    fn try_copy_linear_weight(
        linear: &mut Linear,
        key: &str,
        weights: &mut HashMap<String, Tensor>,
        consumed: &mut std::collections::HashSet<String>,
    ) -> Result<()> {
        if let Some(src) = weights.remove(key) {
            linear.copy_weight_from(&src)?;
            linear.weight = linear.weight.clone().requires_grad_(false);
            drop(src);
            consumed.insert(key.to_string());
        }
        Ok(())
    }
    fn try_copy_linear_bias(
        linear: &mut Linear,
        key: &str,
        weights: &mut HashMap<String, Tensor>,
        consumed: &mut std::collections::HashSet<String>,
    ) -> Result<()> {
        if let Some(src) = weights.remove(key) {
            linear.copy_bias_from(&src)?;
            if let Some(b) = linear.bias.take() {
                linear.bias = Some(b.requires_grad_(false));
            }
            drop(src);
            consumed.insert(key.to_string());
        }
        Ok(())
    }
    fn try_copy_norm(
        dst: &mut Tensor,
        key: &str,
        weights: &mut HashMap<String, Tensor>,
        consumed: &mut std::collections::HashSet<String>,
    ) -> Result<()> {
        if let Some(src) = weights.remove(key) {
            if src.shape().dims() != dst.shape().dims() {
                bail!(
                    "shape mismatch for '{}': expected {:?}, got {:?}",
                    key,
                    dst.shape().dims(),
                    src.shape().dims()
                );
            }
            let cast = if src.dtype() != DType::BF16 {
                src.to_dtype(DType::BF16)?
            } else {
                src.clone()
            };
            drop(src);
            *dst = cast.requires_grad_(false);
            consumed.insert(key.to_string());
        }
        Ok(())
    }

    // Top-level
    try_copy_linear_weight(&mut model.img_in, "img_in.weight", weights, consumed)?;
    try_copy_linear_bias(&mut model.img_in, "img_in.bias", weights, consumed)?;
    try_copy_linear_weight(&mut model.txt_in, "txt_in.weight", weights, consumed)?;
    try_copy_linear_bias(&mut model.txt_in, "txt_in.bias", weights, consumed)?;
    for i in 0..model.config.selected_layer_index.len() {
        let key = format!("txt_norm.{}.weight", i);
        try_copy_norm(&mut model.txt_norm[i], &key, weights, consumed)?;
    }
    try_copy_linear_weight(&mut model.time_embed_linear_1,
        "time_text_embed.timestep_embedder.linear_1.weight", weights, consumed)?;
    try_copy_linear_bias(&mut model.time_embed_linear_1,
        "time_text_embed.timestep_embedder.linear_1.bias", weights, consumed)?;
    try_copy_linear_weight(&mut model.time_embed_linear_2,
        "time_text_embed.timestep_embedder.linear_2.weight", weights, consumed)?;
    try_copy_linear_bias(&mut model.time_embed_linear_2,
        "time_text_embed.timestep_embedder.linear_2.bias", weights, consumed)?;
    try_copy_linear_weight(&mut model.norm_out_linear, "norm_out.linear.weight", weights, consumed)?;
    try_copy_linear_bias(&mut model.norm_out_linear, "norm_out.linear.bias", weights, consumed)?;
    try_copy_linear_weight(&mut model.proj_out, "proj_out.weight", weights, consumed)?;
    try_copy_linear_bias(&mut model.proj_out, "proj_out.bias", weights, consumed)?;

    // Per-block
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("transformer_blocks.{}.", i);
        try_copy_linear_weight(&mut block.img_qkv, &format!("{p}attn.img_qkv.weight"), weights, consumed)?;
        try_copy_linear_bias  (&mut block.img_qkv, &format!("{p}attn.img_qkv.bias"),   weights, consumed)?;
        try_copy_linear_weight(&mut block.txt_qkv, &format!("{p}attn.txt_qkv.weight"), weights, consumed)?;
        try_copy_linear_bias  (&mut block.txt_qkv, &format!("{p}attn.txt_qkv.bias"),   weights, consumed)?;
        try_copy_norm(&mut block.norm_q,       &format!("{p}attn.norm_q.weight"),       weights, consumed)?;
        try_copy_norm(&mut block.norm_k,       &format!("{p}attn.norm_k.weight"),       weights, consumed)?;
        try_copy_norm(&mut block.norm_added_q, &format!("{p}attn.norm_added_q.weight"), weights, consumed)?;
        try_copy_norm(&mut block.norm_added_k, &format!("{p}attn.norm_added_k.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.img_out, &format!("{p}attn.to_out.0.weight"), weights, consumed)?;
        try_copy_linear_bias  (&mut block.img_out, &format!("{p}attn.to_out.0.bias"),   weights, consumed)?;
        try_copy_linear_weight(&mut block.txt_out, &format!("{p}attn.to_add_out.weight"), weights, consumed)?;
        try_copy_linear_bias  (&mut block.txt_out, &format!("{p}attn.to_add_out.bias"),   weights, consumed)?;
        try_copy_linear_weight(&mut block.img_mod, &format!("{p}img_mod.1.weight"), weights, consumed)?;
        try_copy_linear_bias  (&mut block.img_mod, &format!("{p}img_mod.1.bias"),   weights, consumed)?;
        try_copy_linear_weight(&mut block.txt_mod, &format!("{p}txt_mod.1.weight"), weights, consumed)?;
        try_copy_linear_bias  (&mut block.txt_mod, &format!("{p}txt_mod.1.bias"),   weights, consumed)?;
        try_copy_norm(&mut block.img_norm1, &format!("{p}img_norm1.weight"), weights, consumed)?;
        try_copy_norm(&mut block.img_norm2, &format!("{p}img_norm2.weight"), weights, consumed)?;
        try_copy_norm(&mut block.txt_norm1, &format!("{p}txt_norm1.weight"), weights, consumed)?;
        try_copy_norm(&mut block.txt_norm2, &format!("{p}txt_norm2.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.img_mlp_w1, &format!("{p}img_mlp.w1.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.img_mlp_w2, &format!("{p}img_mlp.w2.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.img_mlp_w3, &format!("{p}img_mlp.w3.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.txt_mlp_w1, &format!("{p}txt_mlp.w1.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.txt_mlp_w2, &format!("{p}txt_mlp.w2.weight"), weights, consumed)?;
        try_copy_linear_weight(&mut block.txt_mlp_w3, &format!("{p}txt_mlp.w3.weight"), weights, consumed)?;
    }
    Ok(())
}

/// Compute the full expected key set for the Lens DiT, mirroring BUILD_PLAN.
fn expected_lens_keys(model: &LensTransformer2DModel) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    out.extend([
        "img_in.weight", "img_in.bias",
        "txt_in.weight", "txt_in.bias",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_1.bias",
        "time_text_embed.timestep_embedder.linear_2.weight",
        "time_text_embed.timestep_embedder.linear_2.bias",
        "norm_out.linear.weight", "norm_out.linear.bias",
        "proj_out.weight", "proj_out.bias",
    ].iter().map(|s| s.to_string()));
    for i in 0..model.config.selected_layer_index.len() {
        out.push(format!("txt_norm.{}.weight", i));
    }
    for i in 0..model.config.num_layers {
        let p = format!("transformer_blocks.{}.", i);
        out.push(format!("{p}attn.img_qkv.weight"));
        out.push(format!("{p}attn.img_qkv.bias"));
        out.push(format!("{p}attn.txt_qkv.weight"));
        out.push(format!("{p}attn.txt_qkv.bias"));
        out.push(format!("{p}attn.norm_q.weight"));
        out.push(format!("{p}attn.norm_k.weight"));
        out.push(format!("{p}attn.norm_added_q.weight"));
        out.push(format!("{p}attn.norm_added_k.weight"));
        out.push(format!("{p}attn.to_out.0.weight"));
        out.push(format!("{p}attn.to_out.0.bias"));
        out.push(format!("{p}attn.to_add_out.weight"));
        out.push(format!("{p}attn.to_add_out.bias"));
        out.push(format!("{p}img_mod.1.weight"));
        out.push(format!("{p}img_mod.1.bias"));
        out.push(format!("{p}txt_mod.1.weight"));
        out.push(format!("{p}txt_mod.1.bias"));
        out.push(format!("{p}img_norm1.weight"));
        out.push(format!("{p}img_norm2.weight"));
        out.push(format!("{p}txt_norm1.weight"));
        out.push(format!("{p}txt_norm2.weight"));
        out.push(format!("{p}img_mlp.w1.weight"));
        out.push(format!("{p}img_mlp.w2.weight"));
        out.push(format!("{p}img_mlp.w3.weight"));
        out.push(format!("{p}txt_mlp.w1.weight"));
        out.push(format!("{p}txt_mlp.w2.weight"));
        out.push(format!("{p}txt_mlp.w3.weight"));
    }
    out
}

// ---------------------------------------------------------------------------
// Build zeroed pre-cached text features (matches the Python capture script's
// `prompt_embeds` + `prompt_mask`: 4 layers of zeros + zero bool mask, both
// duplicated for the CFG batch B=2 inside the pipeline).
//
// Returns `(features, mask)` where:
//   * features: Vec of 4 tensors each [B, S_txt, ENC_HIDDEN_DIM=2880] BF16
//   * mask:     [B, S_txt] BF16 keep-mask (all zeros = all-padded)
// ---------------------------------------------------------------------------

fn build_zero_text_features(
    batch: usize,
    s_txt: usize,
    enc_hidden_dim: usize,
    n_layers: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<(Vec<Tensor>, Tensor)> {
    let feat_shape = Shape::from_dims(&[batch, s_txt, enc_hidden_dim]);
    let mut features = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        features.push(Tensor::zeros_dtype(
            feat_shape.clone(),
            DType::BF16,
            device.clone(),
        )?);
    }
    let mask_shape = Shape::from_dims(&[batch, s_txt]);
    let mask = Tensor::zeros_dtype(mask_shape, DType::BF16, device.clone())?;
    Ok((features, mask))
}

// ---------------------------------------------------------------------------
// CFG-Norm rescale (Stage A6) — bit-faithful to pipeline.py:502-511.
//
//   cond, uncond = noise.chunk(2, dim=0)            # split CFG batch
//   comb = uncond + guidance_scale * (cond - uncond)
//   cond_norm = ||cond||_2 over last dim, keepdim
//   comb_norm = ||comb||_2 over last dim, keepdim
//   scale = where(comb_norm > 0, cond_norm / comb_norm.clamp_min(1e-12), 1.0)
//   noise_pred = comb * scale
//
// With zeroed text features the model output for cond == uncond, so
// `comb == cond == uncond`, `cond_norm == comb_norm`, and `scale == 1.0`.
// We still compute the full rescale to exercise the math path; this matches
// what Stage A6 of BUILD_PLAN.md asks for and what Python does at all steps.
// ---------------------------------------------------------------------------

/// CFG norm-rescale for separate cond / uncond tensors.
///
/// Bit-equivalent to `cfg_norm_rescale` but takes the two B=1 forwards
/// directly (no `chunk(2, 0)` step). Mirrors `pipeline.py:502-511`:
///   comb = uncond + cfg * (cond - uncond)
///   cond_norm = ||cond||_2 over last dim (keepdim)
///   comb_norm = ||comb||_2 over last dim (keepdim, clamp_min(1e-12))
///   noise_pred = comb * (cond_norm / comb_norm)
fn cfg_norm_rescale_pair(cond: &Tensor, uncond: &Tensor, guidance_scale: f32) -> Result<Tensor> {
    let diff = cond.sub(uncond)?;
    let comb = uncond.add(&diff.mul_scalar(guidance_scale)?)?;
    let last = cond.shape().dims().len() - 1;
    let cond_norm = cond.pow(2.0)?.sum_dim_keepdim(last)?.sqrt()?;
    let comb_norm = comb
        .pow(2.0)?
        .sum_dim_keepdim(last)?
        .sqrt()?
        .maximum_scalar(1e-12)?;
    let scale = cond_norm.div(&comb_norm)?;
    Ok(comb.mul(&scale)?)
}

#[allow(dead_code)]
fn cfg_norm_rescale(noise_cfg_batch: &Tensor, guidance_scale: f32) -> Result<Tensor> {
    // noise_cfg_batch: [2, S_img, 128] BF16
    let chunks = noise_cfg_batch.chunk(2, 0)?;
    if chunks.len() != 2 {
        bail!("cfg_norm_rescale: chunk(2, 0) returned {} parts", chunks.len());
    }
    let cond = &chunks[0];
    let uncond = &chunks[1];

    // comb = uncond + cfg * (cond - uncond)
    let diff = cond.sub(uncond)?;
    let comb = uncond.add(&diff.mul_scalar(guidance_scale)?)?;

    // L2 norm over the last (channel) dim with keepdim:
    //   x.pow(2).sum_dim_keepdim(-1).sqrt()
    let last = cond.shape().dims().len() - 1;
    let cond_norm = cond.pow(2.0)?.sum_dim_keepdim(last)?.sqrt()?;
    let comb_norm_raw = comb.pow(2.0)?.sum_dim_keepdim(last)?.sqrt()?;
    // clamp_min(1e-12) — implement via maximum_scalar
    let comb_norm = comb_norm_raw.maximum_scalar(1e-12)?;

    // scale = cond_norm / comb_norm; with zeroed embeds these are equal so
    // scale ≈ 1.0 numerically. (The pipeline's `where(comb_norm > 0, ...)`
    // branch is degenerate-safe via the clamp_min, so we don't need an
    // explicit where: 1e-12 floor + cond_norm = 0 in those positions gives
    // scale = 0, and multiplied by comb = 0 in those positions still yields
    // 0 — same as the Python `where` branch.)
    let scale = cond_norm.div(&comb_norm)?;
    let noise_pred = comb.mul(&scale)?;
    Ok(noise_pred)
}

// ---------------------------------------------------------------------------
// Save a [1, 3, H, W] BF16 tensor in [-1, 1] to a PNG.
// ---------------------------------------------------------------------------

fn save_png(t: &Tensor, path: &Path) -> Result<()> {
    let f32t = t.to_dtype(DType::F32)?;
    let data = f32t.to_vec()?;
    let dims = f32t.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        bail!("save_png expects [1, 3, H, W], got {dims:?}");
    }
    let (h, w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                let v = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * w + x) * 3 + c] = v;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("failed to build image buffer"))?
        .save(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let args = parse_args()?;

    // Disable autograd for the duration of inference. Without this the
    // 48-block × 20-step compute graph would be retained — the HiDream-O1
    // multi-step OOM trap (memory: project_hidream_o1_2026-05-09.md).
    flame_core::AutogradContext::set_enabled(false);

    let device = global_cuda_device();
    let flame_device = FlameDevice::from_arc(device.clone());

    println!("============================================================");
    println!("Lens Inference (pure-Rust) — M2 D7 (encoder → DiT → VAE)");
    println!("============================================================");
    println!("transformer-dir   : {}", args.transformer_dir.display());
    println!("text-encoder-dir  : {}", args.text_encoder_dir.display());
    println!("tokenizer-dir     : {}", args.tokenizer_dir.display());
    println!("vae               : {}", args.vae.display());
    println!("captures-dir      : {}", args.captures_dir.display());
    println!(
        "steps={} cfg={} seed={} {}x{} parity={} use_cached_features={}",
        args.steps, args.cfg, args.seed, args.width, args.height,
        args.parity, args.use_cached_features
    );
    println!("prompt            : {:?}", args.prompt);
    println!("negative_prompt   : {:?}", args.negative_prompt);

    // ----- Resolution + sigmas + mu -----
    if args.width % 16 != 0 || args.height % 16 != 0 {
        bail!("width / height must be multiples of 16, got {}x{}", args.width, args.height);
    }
    let latent_h = args.height / 16;
    let latent_w = args.width / 16;
    let seq_len = latent_h * latent_w;

    let mu = compute_empirical_mu(seq_len, args.steps);
    let raw_sigmas = build_sigmas(args.steps);
    let shifted_sigmas = apply_exponential_shift(&raw_sigmas, mu);
    println!("  mu = {:.10}", mu);
    println!("  sigmas[0]={:.6} sigmas[N-1]={:.6}",
        shifted_sigmas[0], shifted_sigmas[args.steps - 1]);

    let cfg = LensDiTConfig::default();
    let report_mem = |label: &str| {
        use cudarc::driver::result::mem_get_info;
        if let Ok((free, total)) = mem_get_info() {
            println!(
                "  GPU [{}]: {:.2}/{:.2} GB free",
                label,
                free as f64 / 1e9,
                total as f64 / 1e9
            );
        }
    };

    // ===========================================================
    // Encoder phase: tokenize + encode positive (and optionally
    // negative) prompt to per-layer hidden states. Then drop the
    // encoder before loading the DiT — sequential memory strategy
    // since both don't fit on a 24 GB card.
    //
    // M1 stepping-stone path: --use-cached-features bypasses the
    // encoder entirely and feeds zeroed features (matches the
    // Python parity ref at lens/parity/captures/).
    // ===========================================================

    // Used to size the zeroed-features fallback (M1 path) and as a sanity
    // log for the encoded path.
    let s_txt_fallback: usize = {
        let p = args.captures_dir.join("capture_metadata.json");
        if p.exists() {
            let raw = std::fs::read_to_string(&p).unwrap_or_default();
            serde_json::from_str::<serde_json::Value>(&raw)
                .ok()
                .and_then(|v| v.get("fake_text_seq_len").and_then(|x| x.as_u64()))
                .map(|x| x as usize)
                .unwrap_or(256)
        } else {
            256
        }
    };

    let (cond_features, cond_mask, uncond_features, uncond_mask): (
        Vec<Tensor>,
        Tensor,
        Vec<Tensor>,
        Tensor,
    );

    let t_encode = Instant::now();
    if args.use_cached_features {
        println!("\n--- M1 path: --use-cached-features (zeroed text features) ---");
        let (f, m) = build_zero_text_features(
            1,
            s_txt_fallback,
            cfg.enc_hidden_dim,
            cfg.selected_layer_index.len(),
            &device,
        )?;
        // cond == uncond == zeros at all positions.
        let f2: Vec<Tensor> = f.iter().map(|t| t.clone()).collect();
        cond_features = f;
        cond_mask = m.clone();
        uncond_features = f2;
        uncond_mask = m;
    } else {
        println!("\n--- M2 D7: Load text encoder + tokenizer ---");
        report_mem("pre-encoder");
        // Tokenizer.
        let tok_path = args.tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", tok_path.display()))?;

        // Encoder.
        let mut enc_config = GptOssConfig::lens_default();
        enc_config.selected_layer_index = cfg.selected_layer_index.iter().copied().collect();
        let t0 = Instant::now();
        let mut encoder = GptOssEncoder::new(enc_config, &device)
            .map_err(|e| anyhow!("GptOssEncoder::new: {e}"))?;
        report_mem("post-encoder-new");
        let stats = encoder
            .load_from_directory(&args.text_encoder_dir, &device)
            .map_err(|e| anyhow!("encoder.load_from_directory: {e}"))?;
        println!(
            "  encoder loaded in {:.1}s: consumed={} mxfp4_dequants={} skipped={}",
            t0.elapsed().as_secs_f32(),
            stats.consumed, stats.mxfp4_dequants, stats.skipped
        );
        report_mem("post-encoder-load");

        // Encode positive.
        println!("  encoding positive prompt ...");
        let t0 = Instant::now();
        let (pos_feats, pos_mask, pos_real, pos_raw) = encode_prompt_with_encoder(
            &mut encoder,
            &tokenizer,
            &args.prompt,
            args.max_text_len,
            &device,
        )?;
        println!(
            "  positive encode: {:.1}s  raw_seq={} post_offset_real={}",
            t0.elapsed().as_secs_f32(),
            pos_raw, pos_real
        );

        // Encode negative. Match Python pipeline.py:256-261: empty negative
        // ⇒ zeros_like(pos). Else run a second encode.
        let (neg_feats, neg_mask) = if args.negative_prompt.trim().is_empty() {
            println!("  negative empty → zeros_like(positive)");
            let nf: Vec<Tensor> = pos_feats
                .iter()
                .map(|t| {
                    Tensor::zeros_dtype(t.shape().clone(), DType::BF16, device.clone())
                        .map_err(|e| anyhow!("neg zeros alloc: {e}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let nm = Tensor::zeros_dtype(pos_mask.shape().clone(), DType::BF16, device.clone())
                .map_err(|e| anyhow!("neg mask alloc: {e}"))?;
            (nf, nm)
        } else {
            println!("  encoding negative prompt ...");
            let t0 = Instant::now();
            let (nf, nm, nreal, nraw) = encode_prompt_with_encoder(
                &mut encoder,
                &tokenizer,
                &args.negative_prompt,
                args.max_text_len,
                &device,
            )?;
            println!(
                "  negative encode: {:.1}s  raw_seq={} post_offset_real={}",
                t0.elapsed().as_secs_f32(),
                nraw, nreal
            );
            (nf, nm)
        };

        // Drop the encoder — DiT phase needs the memory.
        drop(encoder);
        drop(tokenizer);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::memory_pool::MEMORY_POOL.clear_all_caches();
        let _ = device.synchronize();
        trim_cuda_mempool(0);
        let _ = device.synchronize();
        report_mem("post-encoder-drop");

        cond_features = pos_feats;
        cond_mask = pos_mask;
        uncond_features = neg_feats;
        uncond_mask = neg_mask;
    }
    println!(
        "  encoder phase wall: {:.1}s",
        t_encode.elapsed().as_secs_f32()
    );
    println!(
        "  text feature shape (cond): {:?}, mask: {:?}",
        cond_features[0].shape().dims(),
        cond_mask.shape().dims()
    );

    // ===========================================================
    // DiT phase: load weights, run denoise loop, drop, VAE decode.
    // ===========================================================

    println!("\n--- Stage A5: Load DiT weights ({} shards) ---",
        std::fs::read_dir(&args.transformer_dir)
            .map(|d| d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
                .count())
            .unwrap_or(0));
    let t0 = Instant::now();
    report_mem("pre-new");
    let mut model = LensTransformer2DModel::new(cfg.clone(), &device)?;
    report_mem("post-new");
    load_lens_dit_sharded(&mut model, &args.transformer_dir, &device)?;
    report_mem("post-load");
    let _ = device.synchronize();
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::memory_pool::MEMORY_POOL.clear_all_caches();
    let _ = device.synchronize();
    trim_cuda_mempool(0);
    let _ = device.synchronize();
    report_mem("post-trim");
    println!("  Loaded {:.1}s", t0.elapsed().as_secs_f32());

    // ===========================================================
    // Mode 1: PARITY — load captured hidden_states, run DiT only.
    // Uses zeroed features (the M1 captures were built with zeros).
    // ===========================================================
    if args.parity {
        return run_parity(&model, &args, &device, &cfg, s_txt_fallback, &shifted_sigmas);
    }

    // ===========================================================
    // Mode 2: SMOKE — full denoise loop + VAE decode + PNG.
    // ===========================================================

    println!("\n--- Stage A6: Build noise + sigma schedule ---");
    // Bit-exact torch.randn port — Philox4x32-10. Bug 1 fix per
    // BUGFIX_TRIAGE_2026-05-22: Tensor::randn_seeded uses StdRng + CPU Box-Muller,
    // which is statistically N(0,1) but produces a completely different sample
    // than torch.randn at the same seed. Capture was generated on this 3090 Ti so
    // SM-count caveat in flame_core::rng::randn_torch docs is satisfied here.
    let single_noise = flame_core::rng::randn_torch(
        args.seed,
        Shape::from_dims(&[1, seq_len, 128]),
        device.clone(),
    )?;
    let mut latents = if single_noise.dtype() == DType::BF16 {
        single_noise
    } else {
        single_noise.to_dtype(DType::BF16)?
    };
    // DEBUG: optional initial-noise dump (triage 2026-05-23).
    if let Ok(path) = std::env::var("LENS_DUMP_INIT_NOISE") {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("noise".to_string(), latents.clone());
        flame_core::serialization::save_file(&m, &path)?;
        println!("  [debug] wrote initial noise to {path}");
    }
    // DEBUG: optional initial-noise LOAD from external safetensors (lets us
    // smoke-test downstream parity with Python's exact initial noise).
    if let Ok(path) = std::env::var("LENS_LOAD_INIT_NOISE") {
        let map = load_file(std::path::Path::new(&path), &device)?;
        let key = if map.contains_key("noise") { "noise" } else if map.contains_key("hs") { "hs" } else { "latent" };
        let loaded = map.get(key).ok_or_else(|| anyhow!("no usable key in {path}"))?.clone();
        // Capture file has shape [2, S, 128]; we run B=1, take first half.
        let dims = loaded.shape().dims().to_vec();
        let loaded = if dims.len() == 3 && dims[0] == 2 {
            loaded.narrow(0, 0, 1)?
        } else {
            loaded
        };
        latents = if loaded.dtype() == DType::BF16 { loaded } else { loaded.to_dtype(DType::BF16)? };
        println!("  [debug] loaded initial noise from {path}");
    }

    // ---- Denoise loop ----
    //
    // CFG: two separate B=1 forwards (cond + uncond), then combine via the
    // norm-rescale formula in `cfg_norm_rescale_pair`. Two forwards keeps
    // each SDPA at B=1, BH=24 (no B=2 score blow-up).
    //
    // When the encoder produced features (default path), cond ≠ uncond and
    // CFG actually shapes the output. When `--use-cached-features` was set,
    // both equal zeros so CFG degenerates to scale=1.0.
    println!(
        "\n--- Stage C1: Denoise ({} steps, cfg={}, two-forward CFG) ---",
        args.steps, args.cfg
    );
    let t_denoise = Instant::now();
    for step in 0..args.steps {
        let sigma_curr = shifted_sigmas[step];
        let sigma_next = if step + 1 < args.steps {
            shifted_sigmas[step + 1]
        } else {
            0.0
        };

        // Build timestep tensor [1] = sigma_curr (Python passes `t/1000`
        // where t = sigma*1000, so we pass sigma directly).
        let t_vec = Tensor::from_vec(
            vec![sigma_curr],
            Shape::from_dims(&[1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // Conditional forward.
        let noise_cond = model.forward(
            &latents,
            &cond_features,
            &cond_mask,
            &t_vec,
            (1, latent_h, latent_w),
        )?;

        // Unconditional forward (unless features are identical, in which case
        // we can reuse — degenerate M1 path).
        let noise_uncond = if args.use_cached_features {
            noise_cond.clone()
        } else {
            model.forward(
                &latents,
                &uncond_features,
                &uncond_mask,
                &t_vec,
                (1, latent_h, latent_w),
            )?
        };

        // CFG norm rescale (cond_norm / comb_norm) → noise_pred.
        let noise_pred = cfg_norm_rescale_pair(&noise_cond, &noise_uncond, args.cfg)?;

        // DEBUG: optional per-step noise_pred dump (triage 2026-05-23).
        if let Ok(dir) = std::env::var("LENS_DUMP_STEPS_DIR") {
            let mut m: HashMap<String, Tensor> = HashMap::new();
            m.insert("noise".to_string(), noise_pred.clone());
            flame_core::serialization::save_file(
                &m,
                &format!("{dir}/rust_noise_step_{:02}.safetensors", step),
            )?;
        }

        // Scheduler step (single-batch latents).
        latents = euler_step(&latents, &noise_pred, sigma_curr, sigma_next)?;

        // DEBUG: optional per-step post-latent dump (triage 2026-05-23).
        if let Ok(dir) = std::env::var("LENS_DUMP_STEPS_DIR") {
            let mut m: HashMap<String, Tensor> = HashMap::new();
            m.insert("latents".to_string(), latents.clone());
            flame_core::serialization::save_file(
                &m,
                &format!("{dir}/rust_latents_step_{:02}.safetensors", step),
            )?;
        }

        if step == 0 || step == args.steps - 1 || (step + 1) % 5 == 0 {
            println!(
                "  step {:>2}/{} sigma {:.4} -> {:.4} ({:.1}s)",
                step + 1,
                args.steps,
                sigma_curr,
                sigma_next,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    println!(
        "  denoise done in {:.1}s ({:.2}s/step)",
        t_denoise.elapsed().as_secs_f32(),
        t_denoise.elapsed().as_secs_f32() / args.steps as f32
    );

    // ---- VAE decode ----
    println!("\n--- Stage B2: VAE decode ---");
    // DEBUG: optional pre-VAE latent dump (triage 2026-05-23).
    if let Ok(path) = std::env::var("LENS_DUMP_PRE_VAE") {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("latent".to_string(), latents.clone());
        flame_core::serialization::save_file(&m, &path)?;
        println!("  [debug] wrote pre-VAE latent to {path}");
    }
    let t0 = Instant::now();
    drop(model);
    trim_cuda_mempool(0);
    let vae_weights = load_file(&args.vae, &device)?;
    let vae = LensVaeWrapper::load(&vae_weights, &flame_device)?;
    drop(vae_weights);
    let rgb = vae.decode(&latents, latent_h, latent_w)?;
    // DEBUG: optional post-VAE pixel dump (triage 2026-05-23).
    if let Ok(path) = std::env::var("LENS_DUMP_VAE_OUT") {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("pixel".to_string(), rgb.clone());
        flame_core::serialization::save_file(&m, &path)?;
        println!("  [debug] wrote VAE output pixel to {path}");
    }
    println!(
        "  decoded {:?} in {:.1}s",
        rgb.dims(),
        t0.elapsed().as_secs_f32()
    );

    // ---- Save PNG ----
    save_png(&rgb, &args.output)?;
    println!(
        "\n============================================================\n\
         IMAGE SAVED: {}\nTotal time: {:.1}s\n\
         ============================================================",
        args.output.display(),
        t_total.elapsed().as_secs_f32()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Parity mode — DiT-only, step-by-step vs captured tensors.
// ---------------------------------------------------------------------------

fn run_parity(
    model: &LensTransformer2DModel,
    args: &Args,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    cfg: &LensDiTConfig,
    s_txt: usize,
    shifted_sigmas: &[f32],
) -> Result<()> {
    println!("\n--- PARITY MODE — DiT forward vs captured `noise_pred` ---");
    println!("  note: captured tensors are CFG batch B=2 but with zero text features ");
    println!("        both halves are bit-identical; we run B=1 (no OOM) and compare ");
    println!("        against captured[0:1].");

    // Zeroed text features for B=1 (CFG-batch collapse safe at zero embeds).
    let (encoder_features_single, encoder_mask_single) = build_zero_text_features(
        1,
        s_txt,
        cfg.enc_hidden_dim,
        cfg.selected_layer_index.len(),
        device,
    )?;

    let latent_h = args.height / 16;
    let latent_w = args.width / 16;

    // ----- Per-step DiT parity -----
    let mut pass_count = 0usize;
    let mut step_metrics: Vec<(usize, ParityMetric)> = Vec::new();
    let bar = 0.999_f32;
    for step in 0..args.steps {
        let pre_path = args
            .captures_dir
            .join(format!("hidden_states_pre_step_{step:02}.safetensors"));
        let exp_path = args
            .captures_dir
            .join(format!("noise_pred_step_{step:02}.safetensors"));
        if !pre_path.exists() || !exp_path.exists() {
            eprintln!(
                "  step {step:02}: missing captures (pre={}, exp={}); stopping",
                pre_path.exists(),
                exp_path.exists()
            );
            break;
        }

        let hs_cfg = load_single_tensor(&pre_path, "hs", device)?;
        let expected_full = load_single_tensor(&exp_path, "noise", device)?;
        // CFG-batch input → slice to first half (both halves identical for zero embeds).
        let hs_single = hs_cfg.narrow(0, 0, 1)?;
        let expected = expected_full.narrow(0, 0, 1)?;

        let sigma_curr = shifted_sigmas[step];
        let t_vec = Tensor::from_vec(
            vec![sigma_curr],
            Shape::from_dims(&[1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let ours = model.forward(
            &hs_single,
            &encoder_features_single,
            &encoder_mask_single,
            &t_vec,
            (1, latent_h, latent_w),
        )?;

        let m = compare_tensors(&ours, &expected)?;
        let ok = m.cos.is_finite() && m.cos >= bar;
        if ok {
            pass_count += 1;
        }
        println!(
            "  step {:>2}: cos={:.6} max_abs={:.3e} mean_abs={:.3e} {}",
            step,
            m.cos,
            m.max_abs,
            m.mean_abs,
            if ok { "PASS" } else { "FAIL" }
        );
        step_metrics.push((step, m));
    }

    let total = step_metrics.len();
    println!(
        "\n  per-step parity summary: {}/{} steps passed cos >= {:.4}",
        pass_count, total, bar
    );
    let mut worst = (0usize, f32::INFINITY);
    for (s, m) in &step_metrics {
        if m.cos < worst.1 {
            worst = (*s, m.cos);
        }
    }
    if total > 0 {
        println!("  worst step: {} (cos={:.6})", worst.0, worst.1);
    }

    // ----- Per-block parity at step 0 -----
    println!("\n--- Per-block parity at step 0 ---");
    let pre_path_0 = args.captures_dir.join("hidden_states_pre_step_00.safetensors");
    if !pre_path_0.exists() {
        eprintln!("  no step-0 capture; skipping per-block parity");
    } else {
        let hs0 = load_single_tensor(&pre_path_0, "hs", device)?;
        // Reuse the public forward at step 0 to get end-state numbers (block
        // boundaries are not currently exposed publicly; getting per-block
        // outputs cleanly would require either making LensTransformer2DModel
        // helpers `pub` or duplicating the forward chain here. Per the task
        // brief: "if too tricky, skip this fine-grained check." We instead
        // run the full forward and additionally compare it to step-0 expected
        // noise — already done above — and report the captured block-by-block
        // values' counts as a presence check.
        let _ = hs0;
        let mut found_blocks = 0usize;
        for i in 0..cfg.num_layers {
            let p = args
                .captures_dir
                .join(format!("block_{i:02}_step0.safetensors"));
            if p.exists() {
                found_blocks += 1;
            }
        }
        println!(
            "  captures present for {} / {} blocks at step 0 (per-block fine-grained \
             parity not exercised — public forward does NOT expose intermediate \
             block outputs; step-0 end-to-end DiT comparison is the primary gate)",
            found_blocks, cfg.num_layers
        );
    }

    // ----- temb parity -----
    println!("\n--- Timestep embedding (temb) parity at step 0 ---");
    let temb_path = args.captures_dir.join("temb_step0.safetensors");
    if temb_path.exists() {
        let expected_temb = load_single_tensor(&temb_path, "temb", device)?;
        // Reproduce the timestep embedding by running model.forward at step 0
        // and recovering the implicit value is not viable without exposing
        // internals; document this here.
        let _ = expected_temb;
        println!("  temb capture present (shape match left to per-step cos check)");
    }

    // ----- VAE parity (not exercised in --parity mode; smoke covers VAE) -----

    // ----- Final verdict -----
    let pass = pass_count == total && total == args.steps;
    println!(
        "\n============================================================\n\
         M1 PARITY VERDICT: {}\n\
         steps_run={} expected={} passed={} cos_bar={:.4}\n\
         ============================================================",
        if pass { "PASS" } else { "FAIL" },
        total,
        args.steps,
        pass_count,
        bar
    );

    Ok(())
}
