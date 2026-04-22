#![cfg(feature = "turbo")]

//! BF16-bit-identical parity between `klein9b_infer` and
//! `klein9b_infer_turbo`. Skipped (with a clean log line) when the actual
//! 17 GB Klein 9B checkpoint is unavailable.
//!
//! We assume both binaries have been built — `cargo test --features turbo`
//! pulls in their build deps automatically.

use std::path::Path;
use std::process::Command;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const PROMPT: &str = "a cat in a hat";

#[ignore = "30-min full Klein 9B denoise; run with --ignored"]
#[test]
fn klein9b_turbo_matches_baseline() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("skipped, weights unavailable: {MODEL_PATH}");
        return;
    }

    // The two binaries write to fixed paths; capture both outputs as raw
    // bytes and compare. Each binary takes the prompt as argv[1].
    let baseline_out = "/home/alex/EriDiffusion/inference-flame/output/klein9b_rust.png";
    let turbo_out = "/home/alex/EriDiffusion/inference-flame/output/klein9b_rust_turbo.png";

    let baseline = Command::new(env!("CARGO_BIN_EXE_klein9b_infer"))
        .arg(PROMPT)
        .status()
        .expect("klein9b_infer launch");
    assert!(baseline.success(), "klein9b_infer failed: {baseline:?}");

    let turbo = Command::new(env!("CARGO_BIN_EXE_klein9b_infer_turbo"))
        .arg(PROMPT)
        .status()
        .expect("klein9b_infer_turbo launch");
    assert!(turbo.success(), "klein9b_infer_turbo failed: {turbo:?}");

    let baseline_bytes = std::fs::read(baseline_out).expect("read baseline png");
    let turbo_bytes = std::fs::read(turbo_out).expect("read turbo png");

    assert_eq!(
        baseline_bytes.len(), turbo_bytes.len(),
        "baseline/turbo PNG byte length differs ({} vs {})",
        baseline_bytes.len(), turbo_bytes.len(),
    );
    // PNG byte equality implies the pre-encode pixel buffer was identical,
    // which since both paths use the same VAE + the same denoise schedule +
    // the same RNG seed implies the latents matched bit-for-bit.
    assert!(
        baseline_bytes == turbo_bytes,
        "baseline/turbo PNG bytes differ — Turbo Flame Phase 1 failed BF16 parity",
    );
}
