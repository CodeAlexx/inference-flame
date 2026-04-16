//! Motif-Video parity test: Rust DiT forward vs Python reference.
//!
//! Loads `motif_block_dump.safetensors` (produced by
//! `scripts/motif_block_dump.py`), runs the Rust `MotifDit::forward` on the
//! same inputs, and compares the final output + reports per-block cosines.

use std::time::Instant;
use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::motif_video_dit::MotifDit;

const BASE_DIR: &str = "/home/alex/.serenity/models/checkpoints/motif-video-2b";

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn abs_max(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

fn compare(label: &str, rust: &Tensor, python: &Tensor) -> anyhow::Result<()> {
    let r = rust.to_dtype(DType::F32)?.to_vec_f32()?;
    let p = python.to_dtype(DType::F32)?.to_vec_f32()?;
    if r.len() != p.len() {
        println!("  {:<20} SHAPE MISMATCH rust={} py={}", label, r.len(), p.len());
        return Ok(());
    }
    let cos = cosine(&r, &p);
    let mad = max_abs(&r, &p);
    let r_max = abs_max(&r);
    let p_max = abs_max(&p);
    let marker = if cos < 0.99 { " ⚠️" } else if cos < 0.999 { " ~" } else { "" };
    println!("  {:<20} cos={:.6}  max_abs={:.2}  |r|max={:.2} |p|max={:.2}{}",
             label, cos, mad, r_max, p_max, marker);
    Ok(())
}

fn dump_elem_sample(label: &str, rust: &Tensor, python: &Tensor) -> anyhow::Result<()> {
    let r = rust.to_dtype(DType::F32)?.to_vec_f32()?;
    let p = python.to_dtype(DType::F32)?.to_vec_f32()?;
    let n = r.len().min(p.len());
    // Q5 from handoff: does Rust correlate more with +Python or -Python?
    // If corr(r, -p) > 0, ~half the signs are being systematically flipped
    // (structured inversion, e.g., RoPE mis-application). If both near 0,
    // byte-level corruption.
    let corr_pos = cosine(&r, &p);
    let neg_p: Vec<f32> = p.iter().map(|x| -x).collect();
    let corr_neg = cosine(&r, &neg_p);
    println!("  [{}] corr(r, p)={:.6}  corr(r, -p)={:.6}", label, corr_pos, corr_neg);
    // Histograms of r and p
    let (rmin, rmax) = r.iter().fold((f32::MAX, f32::MIN), |(lo,hi), &v| (lo.min(v), hi.max(v)));
    let (pmin, pmax) = p.iter().fold((f32::MAX, f32::MIN), |(lo,hi), &v| (lo.min(v), hi.max(v)));
    // Ratio analysis: for each idx, if p != 0, compute r/p
    let mut ratios: Vec<f32> = r.iter().zip(&p).filter(|(_, pv)| pv.abs() > 1e-3).map(|(rv, pv)| rv / pv).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = ratios.len() / 2;
    let ratio_p10 = ratios[ratios.len()/10];
    let ratio_p50 = ratios[mid];
    let ratio_p90 = ratios[ratios.len()*9/10];
    println!("  [{}] Rust range [{:.3}, {:.3}]  Python range [{:.3}, {:.3}]  n={}",
             label, rmin, rmax, pmin, pmax, n);
    println!("  [{}] r/p ratio percentiles: p10={:.3} p50={:.3} p90={:.3}",
             label, ratio_p10, ratio_p50, ratio_p90);
    // Top-5 positions by absolute diff
    let mut idx_diffs: Vec<(usize, f32)> = (0..n).map(|i| (i, (r[i] - p[i]).abs())).collect();
    idx_diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  [{}] top-5 worst idx:", label);
    for (i, d) in idx_diffs.iter().take(5) {
        println!("    idx={} r={:+.4} p={:+.4} r/p={:+.3} diff={:.4}",
                 i, r[*i], p[*i], if p[*i].abs() > 1e-3 { r[*i]/p[*i] } else { 0.0 }, d);
    }
    // How many elements are "bad" (|r-p| > 0.2) and what fraction?
    let bad: usize = (0..n).filter(|&i| (r[i] - p[i]).abs() > 0.2).count();
    println!("  [{}] elements with |diff|>0.2: {}/{} ({:.1}%)",
             label, bad, n, 100.0 * bad as f32 / n as f32);
    // Sign-flipped: positions where r and p have opposite sign with |p|>0.1
    let flipped: usize = (0..n).filter(|&i| p[i].abs() > 0.1 && r[i].signum() != p[i].signum()).count();
    println!("  [{}] sign-flipped (|p|>0.1): {}/{}", label, flipped, n);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    let dump_path = std::env::args().nth(1).unwrap_or_else(||
        "/home/alex/serenity/output/motif_block_dump.safetensors".into());

    println!("=== Motif-Video Parity Test ===");
    println!("Dump: {}", dump_path);

    let ref_data = flame_core::serialization::load_file(std::path::Path::new(&dump_path), &device)?;
    println!("Loaded {} reference tensors", ref_data.len());

    let hidden = ref_data.get("input_hidden").ok_or_else(|| anyhow::anyhow!("missing input_hidden"))?
        .clone().to_dtype(DType::BF16)?;
    let text = ref_data.get("text_embeds").ok_or_else(|| anyhow::anyhow!("missing text_embeds"))?
        .clone().to_dtype(DType::BF16)?;
    let timestep = ref_data.get("timestep").ok_or_else(|| anyhow::anyhow!("missing timestep"))?
        .clone().to_dtype(DType::BF16)?;

    println!("hidden: {:?}, text: {:?}", hidden.shape().dims(), text.shape().dims());

    // Load DiT
    println!("Loading DiT...");
    let shard = format!("{}/transformer/diffusion_pytorch_model.safetensors", BASE_DIR);
    let t0 = Instant::now();
    let mut dit = MotifDit::load(&[&shard], &device)?;
    // Capture imposes state (img.clone() → Arc alias per block). Disable when
    // running MOTIF_DOUBLE_FWD to isolate forward() from capture effects.
    if std::env::var("MOTIF_NO_CAPTURE").is_err() {
        dit.capture = Some(std::collections::HashMap::new());
    }
    println!("DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Forward — optionally run twice to detect in-process non-determinism
    println!("Running Rust forward...");
    let t0 = Instant::now();
    let out = dit.forward(&hidden, &text, &timestep, None)?;
    println!("Forward in {:.1}s. Output: {:?}\n", t0.elapsed().as_secs_f32(), out.shape().dims());

    if std::env::var("MOTIF_DOUBLE_FWD").is_ok() {
        let n_extra: usize = std::env::var("MOTIF_DOUBLE_FWD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let a = out.to_dtype(DType::F32)?.to_vec_f32()?;
        let mut prev = a.clone();
        let mut times_ms = Vec::with_capacity(n_extra);
        // Reset copy stats so we see only per-loop contributions.
        let _ = flame_core::staging::bf16_copy_stats_snapshot(true);
        for i in 1..=n_extra {
            let t_fwd = Instant::now();
            let out_k = dit.forward(&hidden, &text, &timestep, None)?;
            let _ = out_k.to_dtype(DType::F32)?.to_vec_f32()?; // force sync via D2H
            let ms = t_fwd.elapsed().as_secs_f64() * 1000.0;
            times_ms.push(ms);
            let b = out_k.to_dtype(DType::F32)?.to_vec_f32()?;
            let cos = {
                let dot: f64 = prev.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
                let na: f64 = prev.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                dot / (na * nb + 1e-12)
            };
            let cos_vs0 = {
                let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
                let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                dot / (na * nb + 1e-12)
            };
            let mad = prev.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("FWD#{}: cos_vs_prev={:.9} cos_vs_0={:.9} max_abs_vs_prev={:.4}  time={:.1}ms",
                     i, cos, cos_vs0, mad, ms);
            prev = b;
        }
        if !times_ms.is_empty() {
            let n = times_ms.len() as f64;
            let sum: f64 = times_ms.iter().sum();
            let mean = sum / n;
            let mut sorted = times_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            let min = sorted[0];
            let max = sorted[sorted.len() - 1];
            println!("TIMING: n={} mean={:.1}ms median={:.1}ms min={:.1}ms max={:.1}ms",
                     times_ms.len(), mean, median, min, max);
            // Dump bf16_copy call-site breakdown (populated when FLAME_COPY_TRACE=1).
            let mut stats = flame_core::staging::bf16_copy_stats_snapshot(true);
            if !stats.is_empty() {
                stats.sort_by(|a, b| b.1.cmp(&a.1));
                let total_calls: u64 = stats.iter().map(|(_, c, _)| c).sum();
                let total_elems: u64 = stats.iter().map(|(_, _, e)| e).sum();
                println!("COPY_STATS: total_calls={} total_MB={:.1} across {} forwards",
                         total_calls, (total_elems as f64 * 2.0) / (1024.0 * 1024.0), times_ms.len());
                for (tag, calls, elems) in &stats {
                    println!("  {:>30}: calls={:>6} ({:>5.1}%)  elems={:>11}  MB={:>7.1}",
                             tag, calls,
                             100.0 * *calls as f64 / total_calls.max(1) as f64,
                             elems,
                             (*elems as f64 * 2.0) / (1024.0 * 1024.0));
                }
            }
        }
    }

    // With MOTIF_NO_CAPTURE=1, skip per-block comparisons and jump to final.
    let empty_cap: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    let cap = dit.capture.as_ref().unwrap_or(&empty_cap);

    // Compare pre-block
    println!("--- Pre-block ---");
    for name in &["x_embedder", "context_embedder", "time_text_embed"] {
        if let (Some(r), Some(p)) = (cap.get(*name), ref_data.get(*name)) {
            compare(name, r, p)?;
        }
    }

    // Compare per-block
    println!("\n--- Dual-stream blocks (0..11) ---");
    for i in 0..12 {
        let key = format!("dual_{i}");
        if let (Some(r), Some(p)) = (cap.get(&key), ref_data.get(&key)) {
            compare(&key, r, p)?;
        }
    }

    println!("\n--- Single-stream blocks (0..15 encoder, 16..23 decoder) ---");
    for i in 0..24 {
        let key = format!("single_{i}");
        if let (Some(r), Some(p)) = (cap.get(&key), ref_data.get(&key)) {
            compare(&key, r, p)?;
        }
    }


    // Compare final
    println!("\n--- Final output ---");
    if let Some(p) = ref_data.get("final_output") {
        compare("final_output", &out, p)?;
        if std::env::var("MOTIF_DUMP_ELEM").is_ok() {
            dump_elem_sample("final_output", &out, p)?;
        }
    }

    // Post-block
    println!("\n--- Post-block ---");
    for name in &["norm_out", "proj_out"] {
        if let (Some(r), Some(p)) = (cap.get(*name), ref_data.get(*name)) {
            compare(name, r, p)?;
        }
    }

    Ok(())
}
