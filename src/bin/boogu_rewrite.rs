//! Boogu-Image C9 — prompt **rewriter** (pure-Rust, no Python).
//!
//! Expands a short / rough user idea into a fuller T2I instruction using the
//! same Qwen3-VL-8B `mllm` language tower the encoder loads, run in GENERATE
//! mode with Boogu's own EN rewrite system prompt. The output text is what you
//! feed `boogu_encode` / the pipeline as the instruction.
//!
//! Usage:
//!   boogu_rewrite "a cat in a hat"                 # greedy (deterministic)
//!   boogu_rewrite "a cat in a hat" --temp 0.7      # sampling (oracle default)
//!   boogu_rewrite "a cat in a hat" --max-new 512   # cap generated tokens
//!   boogu_rewrite "a cat in a hat" --dump-ids out.json   # dump prompt+gen ids (parity)
//!
//! Defaults: greedy (temperature 0) for deterministic parity, max_new_tokens
//! 1024 (the oracle default). The dev-tool oracle samples at temp 0.7; we
//! default to greedy so the output is reproducible and the parity gate is
//! token-comparable.

use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaDevice;

use flame_core::AutogradContext;
use inference_flame::models::boogu::generate::{BooguRewriter, QWEN_EOS_IDS};
use inference_flame::models::boogu::rewrite_prompt::build_rewrite_prompt;
use inference_flame::models::boogu::tokenizer::{boogu_tokenize_text, load_boogu_tokenizer};

/// Checkpoint root + mllm component dir.
const REPO: &str = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base";
const MLLM_SUBDIR: &str = "mllm";

const DEFAULT_IDEA: &str = "a cat in a hat";
const DEFAULT_MAX_NEW: usize = 1024;
/// Fixed seed for sampling-mode reproducibility (greedy ignores it).
const SEED: u64 = 42;

struct Args {
    idea: String,
    temperature: f32,
    max_new: usize,
    dump_ids: Option<String>,
}

fn parse_args() -> Args {
    let mut idea: Option<String> = None;
    let mut temperature = 0.0f32; // greedy default
    let mut max_new = DEFAULT_MAX_NEW;
    let mut dump_ids: Option<String> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--temp" | "--temperature" => {
                temperature = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(temperature);
            }
            "--greedy" => temperature = 0.0,
            "--max-new" | "--max-new-tokens" => {
                max_new = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_new);
            }
            "--dump-ids" => dump_ids = it.next(),
            other => {
                if idea.is_none() && !other.starts_with("--") {
                    idea = Some(other.to_string());
                }
            }
        }
    }
    Args {
        idea: idea.unwrap_or_else(|| DEFAULT_IDEA.to_string()),
        temperature,
        max_new,
        dump_ids,
    }
}

fn main() -> Result<()> {
    env_logger::init();
    // INFERENCE: autograd OFF (no graph retention — the generate loop would
    // otherwise pin every layer's activations for the whole sequence).
    let _no_grad = AutogradContext::no_grad();

    let args = parse_args();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Boogu-Image — Prompt Rewriter (Qwen3-VL-8B generate)");
    println!("============================================================");
    println!("  Idea       : {}", args.idea);
    println!(
        "  Mode       : {}",
        if args.temperature < 1e-5 {
            "greedy (deterministic)".to_string()
        } else {
            format!("sampling (temp {:.2}, seed {SEED})", args.temperature)
        }
    );
    println!("  max_new    : {}", args.max_new);

    let device = CudaDevice::new(0).context("cuda device 0")?;
    let mllm_dir = std::path::Path::new(REPO).join(MLLM_SUBDIR);

    // --- tokenize the rewrite chat prompt (incl. assistant generation prompt) ---
    let tokenizer =
        load_boogu_tokenizer(&mllm_dir).map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let prompt_str = build_rewrite_prompt(&args.idea);
    let prompt_ids = boogu_tokenize_text(&tokenizer, &prompt_str)
        .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
    println!("  prompt ids : {} tokens", prompt_ids.len());

    // --- load the generate engine ---
    println!("\n--- Loading Qwen3-VL-8B language tower (+ lm_head) ---");
    let t0 = Instant::now();
    let mut rewriter = BooguRewriter::load(&mllm_dir, SEED, device.clone())
        .map_err(|e| anyhow!("BooguRewriter::load: {e}"))?;
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- generate ---
    println!("\n--- Generating ---");
    let gen_ids = rewriter
        .generate_ids(&prompt_ids, args.max_new, args.temperature)
        .map_err(|e| anyhow!("generate: {e}"))?;
    println!("  generated  : {} tokens", gen_ids.len());

    // --- detokenize (skip specials, like the oracle's skip_special_tokens=True) ---
    let text = tokenizer
        .decode(&gen_ids, true)
        .map_err(|e| anyhow!("decode: {e}"))?;
    // Oracle does .strip().replace("\\n", " "); we keep newlines collapsed to a
    // single space for a clean one-line instruction.
    let text_clean = text.trim().replace('\n', " ");

    // --- optional id dump for parity comparison ---
    if let Some(path) = &args.dump_ids {
        let payload = serde_json::json!({
            "idea": args.idea,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new,
            "prompt_len": prompt_ids.len(),
            "prompt_ids": prompt_ids,
            "generated_ids": gen_ids,
            "eos_ids": QWEN_EOS_IDS,
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload)?)
            .with_context(|| format!("write dump-ids to {path}"))?;
        println!("  dumped ids -> {path}");
    }

    println!("\n============================================================");
    println!("REWRITTEN INSTRUCTION:");
    println!("============================================================");
    println!("{text_clean}");
    println!("\n  total {:.1}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
