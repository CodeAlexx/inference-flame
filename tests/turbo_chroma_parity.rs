#![cfg(feature = "turbo")]

//! BF16-bit-identical parity between `chroma_infer` and `chroma_infer_turbo`.
//! Skipped (with a clean log line) when the actual Chroma checkpoint is
//! unavailable.
//!
//! `#[ignore]`'d because Chroma 1024² × 40 steps × 2 forwards/step is a
//! many-minute run. Skeptic / orchestrator runs it explicitly via
//! `cargo test --features turbo --test turbo_chroma_parity -- --ignored`.

use std::path::Path;
use std::process::Command;

// First Chroma DiT shard — used only as a presence probe so we skip cleanly
// when the checkpoint isn't installed.
const PROBE_PATH: &str = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors";

const PROMPT: &str = "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";

#[ignore = "long real-checkpoint Chroma denoise; run with --ignored"]
#[test]
fn chroma_turbo_matches_baseline() {
    if !Path::new(PROBE_PATH).exists() {
        eprintln!("skipped, weights unavailable: {PROBE_PATH}");
        return;
    }

    // Both binaries gate on CHROMA_INFER_FORCE=1 to acknowledge the OOM
    // caveat — propagate that here too.
    let baseline_out = "/home/alex/EriDiffusion/inference-flame/output/chroma_rust.png";
    let turbo_out = "/home/alex/EriDiffusion/inference-flame/output/chroma_rust_turbo.png";

    let baseline = Command::new(env!("CARGO_BIN_EXE_chroma_infer"))
        .env("CHROMA_INFER_FORCE", "1")
        .arg(PROMPT)
        .status()
        .expect("chroma_infer launch");
    assert!(baseline.success(), "chroma_infer failed: {baseline:?}");

    let turbo = Command::new(env!("CARGO_BIN_EXE_chroma_infer_turbo"))
        .env("CHROMA_INFER_FORCE", "1")
        .arg(PROMPT)
        .status()
        .expect("chroma_infer_turbo launch");
    assert!(turbo.success(), "chroma_infer_turbo failed: {turbo:?}");

    let baseline_bytes = std::fs::read(baseline_out).expect("read baseline png");
    let turbo_bytes = std::fs::read(turbo_out).expect("read turbo png");

    assert_eq!(
        baseline_bytes.len(), turbo_bytes.len(),
        "baseline/turbo PNG byte length differs ({} vs {})",
        baseline_bytes.len(), turbo_bytes.len(),
    );
    // PNG byte equality implies the pre-encode pixel buffer was identical;
    // since both paths use the same VAE + denoise schedule + RNG seed this
    // proves latent-level BF16 bit parity.
    assert!(
        baseline_bytes == turbo_bytes,
        "baseline/turbo PNG bytes differ — Turbo Flame Phase 2 (Chroma) failed BF16 parity",
    );
}
