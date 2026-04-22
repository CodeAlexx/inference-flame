//! Qwen-Image-Edit off-the-bench: measure total wall-clock, per-step time.
//! Compares Turbo OFF (qwenimage_edit_gen) vs Turbo ON
//! (qwenimage_edit_gen_turbo). Requires Stage-1 cached embeddings + the
//! Qwen-Image-Edit DiT checkpoint.
//!
//! Usage:
//!   cargo bench --features turbo --bench turbo_qwen_edit_offload -- --nocapture
//!
//! Scaffolding only — orchestrator runs the actual GPU benches and pastes
//! numbers into PHASE2_TIMING_REPORT.md.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

const EMBEDS_PROBE: &str = "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_embeds.safetensors";
const DIT_PROBE: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00001-of-00005.safetensors";
const NUM_STEPS: usize = 50;

fn time_binary(env_bin: &str, label: &str, out_path: &str) -> f64 {
    let t0 = Instant::now();
    let status = Command::new(env_bin)
        .arg(EMBEDS_PROBE)
        .arg(out_path)
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
    if !Path::new(EMBEDS_PROBE).exists() {
        eprintln!("skipped: Stage-1 embeddings not found at {EMBEDS_PROBE}");
        return;
    }
    if !Path::new(DIT_PROBE).exists() {
        eprintln!("skipped: Qwen-Image-Edit checkpoint not found at {DIT_PROBE}");
        return;
    }

    let baseline_bin = env!("CARGO_BIN_EXE_qwenimage_edit_gen");
    let turbo_bin = env!("CARGO_BIN_EXE_qwenimage_edit_gen_turbo");

    let baseline_out = "/home/alex/EriDiffusion/inference-flame/output/_bench_qwen_edit_baseline.safetensors";
    let turbo_out = "/home/alex/EriDiffusion/inference-flame/output/_bench_qwen_edit_turbo.safetensors";

    println!("=== Qwen-Image-Edit-2511 1024² × {} steps × true-CFG 4.0 ===", NUM_STEPS);
    println!("Run 1 / 3 (warmup, ignored):");
    let _ = time_binary(baseline_bin, "warmup-baseline", baseline_out);
    let _ = time_binary(turbo_bin, "warmup-turbo", turbo_out);

    println!("\nRun 2 / 3 (measured):");
    let off_a = time_binary(baseline_bin, "off", baseline_out);
    let on_a = time_binary(turbo_bin, "on", turbo_out);

    println!("\nRun 3 / 3 (measured):");
    let off_b = time_binary(baseline_bin, "off", baseline_out);
    let on_b = time_binary(turbo_bin, "on", turbo_out);

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
