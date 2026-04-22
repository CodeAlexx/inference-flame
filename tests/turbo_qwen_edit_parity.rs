#![cfg(feature = "turbo")]

//! BF16-bit-identical parity between `qwenimage_edit_gen` and
//! `qwenimage_edit_gen_turbo`. Skipped (with a clean log line) when the
//! shipped Stage-1 embeddings file or the Qwen-Image-Edit DiT checkpoint is
//! unavailable.
//!
//! `#[ignore]`'d because Qwen-Image-Edit-2511 1024² × 50 steps × 2
//! forwards/step is ~23 minutes per binary. Skeptic / orchestrator runs it
//! explicitly via
//! `cargo test --features turbo --test turbo_qwen_edit_parity -- --ignored`.

use std::path::Path;
use std::process::Command;

const EMBEDS_PROBE: &str = "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_embeds.safetensors";
const DIT_PROBE: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00001-of-00005.safetensors";

#[ignore = "23-min Qwen-Image-Edit denoise; run with --ignored"]
#[test]
fn qwen_edit_turbo_matches_baseline() {
    if !Path::new(EMBEDS_PROBE).exists() {
        eprintln!("skipped, Stage-1 embeddings unavailable: {EMBEDS_PROBE}");
        return;
    }
    if !Path::new(DIT_PROBE).exists() {
        eprintln!("skipped, Qwen-Image-Edit weights unavailable: {DIT_PROBE}");
        return;
    }

    let baseline_out = "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_latents.safetensors";
    let turbo_out = "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_latents_turbo.safetensors";

    let baseline = Command::new(env!("CARGO_BIN_EXE_qwenimage_edit_gen"))
        .arg(EMBEDS_PROBE)
        .arg(baseline_out)
        .status()
        .expect("qwenimage_edit_gen launch");
    assert!(baseline.success(), "qwenimage_edit_gen failed: {baseline:?}");

    let turbo = Command::new(env!("CARGO_BIN_EXE_qwenimage_edit_gen_turbo"))
        .arg(EMBEDS_PROBE)
        .arg(turbo_out)
        .status()
        .expect("qwenimage_edit_gen_turbo launch");
    assert!(turbo.success(), "qwenimage_edit_gen_turbo failed: {turbo:?}");

    // Compare the saved safetensors files byte-for-byte. Both binaries write
    // BF16 packed_latent + BF16 height/width with identical key order, so
    // serialized bytes match iff the latents match bit-for-bit.
    let baseline_bytes = std::fs::read(baseline_out).expect("read baseline latents");
    let turbo_bytes = std::fs::read(turbo_out).expect("read turbo latents");

    assert_eq!(
        baseline_bytes.len(), turbo_bytes.len(),
        "baseline/turbo latents byte length differs ({} vs {})",
        baseline_bytes.len(), turbo_bytes.len(),
    );
    assert!(
        baseline_bytes == turbo_bytes,
        "baseline/turbo latents bytes differ — Turbo Flame Phase 2 (Qwen-Edit) failed BF16 parity",
    );
}
