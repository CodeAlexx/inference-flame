//! Klein 9B off-the-bench: measure total wall-clock, per-step time, per-swap
//! overlap. Compares Turbo OFF (klein9b_infer) vs Turbo ON
//! (klein9b_infer_turbo). Requires the actual Klein 9B checkpoint.
//!
//! Usage:
//!   cargo bench --features turbo --bench turbo_klein9b_offload -- \
//!       --nocapture
//!
//! The bench sets `KLEIN_STEPS` for BOTH child binaries so they run the
//! same step count — otherwise klein9b_infer defaults to 50 and
//! klein9b_infer_turbo defaults to 30, and dividing both by 50 mixes
//! apples and oranges. Override the step count with
//! `KLEIN_STEPS=<N> cargo bench ...`; default is 50 (baseline's default)
//! so the "OFF" column reports its ordinary production cost.
//!
//! No criterion or external bench harness — keeps the dep budget at zero.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const PROMPT: &str = "a cat in a hat";
const DEFAULT_STEPS: usize = 50;

fn resolve_steps() -> usize {
    std::env::var("KLEIN_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_STEPS)
}

fn time_binary(env_bin: &str, label: &str, steps: usize) -> f64 {
    let t0 = Instant::now();
    let status = Command::new(env_bin)
        .arg(PROMPT)
        .env("KLEIN_STEPS", steps.to_string())
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
    let steps = resolve_steps();
    let steps_f = steps as f64;

    println!("=== Klein 9B 1024² × {steps} steps × CFG 4.0 ===");
    println!("Run 1 / 3 (warmup, ignored):");
    let _ = time_binary(baseline_bin, "warmup-baseline", steps);
    let _ = time_binary(turbo_bin, "warmup-turbo", steps);

    println!("\nRun 2 / 3 (measured):");
    let off_a = time_binary(baseline_bin, "off", steps);
    let on_a = time_binary(turbo_bin, "on", steps);

    println!("\nRun 3 / 3 (measured):");
    let off_b = time_binary(baseline_bin, "off", steps);
    let on_b = time_binary(turbo_bin, "on", steps);

    let off_mean = (off_a + off_b) / 2.0;
    let on_mean = (on_a + on_b) / 2.0;
    let speedup = off_mean / on_mean;

    println!("\n+----------+----------------+----------------+----------+");
    println!("|  Mode    |  Wall-clock s  |  Per step ms   |  vs OFF  |");
    println!("+----------+----------------+----------------+----------+");
    println!("|  OFF     |  {:>11.2}   |  {:>11.1}   |   1.00x  |", off_mean, off_mean * 1000.0 / steps_f);
    println!("|  ON      |  {:>11.2}   |  {:>11.1}   |   {:.2}x  |", on_mean, on_mean * 1000.0 / steps_f, speedup);
    println!("+----------+----------------+----------------+----------+");
}
