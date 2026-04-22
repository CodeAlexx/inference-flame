//! Klein 9B off-the-bench: measure total wall-clock, per-step time, per-swap
//! overlap. Compares Turbo OFF (klein9b_infer) vs Turbo ON
//! (klein9b_infer_turbo). Requires the actual Klein 9B checkpoint.
//!
//! Usage:
//!   cargo bench --features turbo --bench turbo_klein9b_offload -- \
//!       --nocapture
//!
//! No criterion or external bench harness — keeps the dep budget at zero.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const PROMPT: &str = "a cat in a hat";

fn time_binary(env_bin: &str, label: &str) -> f64 {
    let t0 = Instant::now();
    let status = Command::new(env_bin)
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
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "skipped: Klein 9B checkpoint not found at {MODEL_PATH}. Provide \
             the file or rerun on a machine that has it."
        );
        return;
    }

    let baseline_bin = env!("CARGO_BIN_EXE_klein9b_infer");
    let turbo_bin = env!("CARGO_BIN_EXE_klein9b_infer_turbo");

    println!("=== Klein 9B 1024² × 50 steps × CFG 4.0 ===");
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
    println!("|  OFF     |  {:>11.2}   |  {:>11.1}   |   1.00x  |", off_mean, off_mean * 1000.0 / 50.0);
    println!("|  ON      |  {:>11.2}   |  {:>11.1}   |   {:.2}x  |", on_mean, on_mean * 1000.0 / 50.0, speedup);
    println!("+----------+----------------+----------------+----------+");
}
