//! Boogu-Image C8 — text encode stage (separate process, per the memory plan).
//!
//! Loads the Qwen3-VL-8B **language tower** ([`BooguTextEncoder`]), encodes the
//! conditional (positive) instruction AND the CFG unconditional (DROP-template,
//! empty user), and writes both hidden-state tensors to a safetensors cache that
//! `boogu_infer` loads. The encoder is then freed (process exit) so it never
//! shares VRAM with the resident DiT. Mirrors the `klein9b_encode` split idiom.
//!
//! Usage:
//!   boogu_encode ["instruction"]
//!
//! Output cache (`output/boogu_embeddings.safetensors`):
//!   - `cond_hidden` : `[1, Lc, 4096]`  (positive instruction `hidden_states[-1]`)
//!   - `uncond_hidden`: `[1, Lu, 4096]` (DROP-template/empty `hidden_states[-1]`)
//!
//! Both are sliced back to their **real** (non-pad) length: the encoder pads to a
//! supported SDPA length (64/128/...) with pad id 151643 and masks the pad rows,
//! so the real rows are numerically identical to unpadded; we slice before save
//! so the DiT consumes exactly the caption length (`cap_len`).
//!
//! Pure-Rust runtime — no Python.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaDevice;

use flame_core::Tensor;
use inference_flame::models::boogu::encoder::BooguTextEncoder;
use inference_flame::models::boogu::tokenizer::{
    boogu_tokenize, boogu_tokenize_uncond, load_boogu_tokenizer, BOOGU_PAD_ID,
};

/// Checkpoint root.
const REPO: &str = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base";
/// The mllm (Qwen3-VL-8B) component dir.
const MLLM_SUBDIR: &str = "mllm";
/// Output cache consumed by `boogu_infer`.
const OUTPUT_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/boogu_embeddings.safetensors";

const DEFAULT_INSTRUCTION: &str = "a photo of an astronaut riding a horse on mars";

/// Supported SDPA sequence lengths (smallest >= real_len is chosen). The encoder
/// path is happiest at these granularities; pad rows are masked out.
const SUPPORTED_PAD_LENS: [usize; 5] = [64, 128, 256, 512, 1024];

/// Pad `ids` up to the smallest supported length >= `ids.len()`, filling with the
/// Qwen pad id (151643). Returns `(padded_ids, real_len)`.
fn pad_to_supported(mut ids: Vec<i32>) -> Result<(Vec<i32>, usize)> {
    let real_len = ids.len();
    let pad_len = SUPPORTED_PAD_LENS
        .iter()
        .copied()
        .find(|&l| l >= real_len)
        .ok_or_else(|| {
            anyhow!(
                "boogu_encode: instruction tokenizes to {real_len} tokens, exceeds max \
                 supported pad len {}",
                SUPPORTED_PAD_LENS.last().copied().unwrap_or(0)
            )
        })?;
    ids.resize(pad_len, BOOGU_PAD_ID);
    Ok((ids, real_len))
}

fn main() -> Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let instruction = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_INSTRUCTION.to_string());

    println!("============================================================");
    println!("Boogu-Image — Text Encode (Qwen3-VL-8B language tower)");
    println!("============================================================");
    println!("  Instruction: {instruction}");

    // `CudaDevice::new` returns Arc<CudaDevice>.
    let device = CudaDevice::new(0).context("cuda device 0")?;

    // --- Tokenize cond + uncond ---
    let mllm_dir = std::path::Path::new(REPO).join(MLLM_SUBDIR);
    let tokenizer = load_boogu_tokenizer(&mllm_dir)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;

    let cond_ids = boogu_tokenize(&tokenizer, &instruction)
        .map_err(|e| anyhow!("tokenize cond: {e}"))?;
    let uncond_ids = boogu_tokenize_uncond(&tokenizer)
        .map_err(|e| anyhow!("tokenize uncond: {e}"))?;

    let (cond_padded, cond_real) = pad_to_supported(cond_ids)?;
    let (uncond_padded, uncond_real) = pad_to_supported(uncond_ids)?;
    println!(
        "  cond tokens: {cond_real} (pad->{}), uncond tokens: {uncond_real} (pad->{})",
        cond_padded.len(),
        uncond_padded.len()
    );

    // --- Load the Qwen3-VL language tower ---
    println!("\n--- Loading Qwen3-VL-8B language tower ---");
    let t0 = Instant::now();
    let encoder = BooguTextEncoder::load(&mllm_dir, device.clone())
        .map_err(|e| anyhow!("BooguTextEncoder::load: {e}"))?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Encode both, slice back to real length ---
    println!("\n--- Encoding ---");
    let t0 = Instant::now();
    let cond_full = encoder
        .encode(&cond_padded)
        .map_err(|e| anyhow!("encode cond: {e}"))?; // [1, pad, 4096]
    let uncond_full = encoder
        .encode(&uncond_padded)
        .map_err(|e| anyhow!("encode uncond: {e}"))?;

    // Slice [1, pad, 4096] -> [1, real, 4096]; .contiguous() so the saved tensor
    // owns its bytes (narrow is a view).
    let cond_hidden = cond_full
        .narrow(1, 0, cond_real)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("slice cond: {e}"))?;
    let uncond_hidden = uncond_full
        .narrow(1, 0, uncond_real)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("slice uncond: {e}"))?;
    println!(
        "  cond_hidden: {:?}, uncond_hidden: {:?}  ({:.1}s)",
        cond_hidden.shape().dims(),
        uncond_hidden.shape().dims(),
        t0.elapsed().as_secs_f32()
    );

    // --- Save ---
    println!("\n--- Saving embeddings ---");
    if let Some(parent) = std::path::Path::new(OUTPUT_PATH).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("cond_hidden".to_string(), cond_hidden);
    tensors.insert("uncond_hidden".to_string(), uncond_hidden);
    flame_core::serialization::save_file(&tensors, OUTPUT_PATH)
        .map_err(|e| anyhow!("save embeddings: {e}"))?;
    println!("  Saved to {OUTPUT_PATH}");

    // Encoder freed on scope exit (then process exits) — DiT never co-resides.
    println!("\nTotal time: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
