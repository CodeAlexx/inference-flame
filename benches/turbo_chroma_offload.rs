//! Chroma off-the-bench: measure total wall-clock, per-step time, per-swap
//! overlap. Compares Turbo OFF (chroma_infer) vs Turbo ON
//! (chroma_infer_turbo). Requires the actual Chroma checkpoint.
//!
//! Usage:
//!   cargo bench --features turbo --bench turbo_chroma_offload -- --nocapture
//!
//! No criterion or external bench harness — keeps the dep budget at zero.
//! Scaffolding only — orchestrator runs the actual GPU benches and pastes
//! numbers into PHASE2_TIMING_REPORT.md.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

const PROBE_PATH: &str = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors";
const PROMPT: &str = "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";
const NUM_STEPS: usize = 40;

fn time_binary(env_bin: &str, label: &str) -> f64 {
    let t0 = Instant::now();
    let status = Command::new(env_bin)
        .env("CHROMA_INFER_FORCE", "1")
        .arg(PROMPT)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| panic!("{label} launch: {e}"));
    if !status.success() {
        panic!("{label} exited {:?}", status.code());
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("[{label}] total wall-clock: {dt:.2} s");
    dt
}

fn main() {
    if !Path::new(PROBE_PATH).exists() {
        eprintln!(
            "skipped: Chroma checkpoint not found at {PROBE_PATH}. Provide \
             the file or rerun on a machine that has it."
        );
        return;
    }

    let baseline_bin = env!("CARGO_BIN_EXE_chroma_infer");
    let turbo_bin = env!("CARGO_BIN_EXE_chroma_infer_turbo");

    println!("=== Chroma 1024² × {} steps × CFG 4.0 ===", NUM_STEPS);
    println!("Run 1 / 3 (warmup, ignored):");
    let _ = time_binary(baseline_bin, "warmup-baseline");
    let _ = time_binary(turbo_bin, "warmup-turbo");

    println!("\nRun 2 / 3 (measured):");
    let off_a = time_binary(baseline_bin, "off");
    let on_a = time_binary(turbo_bin, "on");

    println!("\nRun 3 / 3 (measured):");
    let off_b = time_binary(baseline_bin, "off");
    let on_b = time_binary(turbo_bin, "on");

    let off_mean = (off_a + off_b) / 2.0;
    let on_mean = (on_a + on_b) / 2.0;
    let speedup = off_mean / on_mean;

    println!("\n+----------+----------------+----------------+----------+");
    println!("|  Mode    |  Wall-clock s  |  Per step ms   |  vs OFF  |");
    println!("+----------+----------------+----------------+----------+");
    println!("|  OFF     |  {:>11.2}   |  {:>11.1}   |   1.00x  |", off_mean, off_mean * 1000.0 / NUM_STEPS as f64);
    println!("|  ON      |  {:>11.2}   |  {:>11.1}   |   {:.2}x  |", on_mean, on_mean * 1000.0 / NUM_STEPS as f64, speedup);
    println!("+----------+----------------+----------------+----------+");
}
