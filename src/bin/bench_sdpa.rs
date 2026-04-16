//! SDPA bench: sweep sequence length at fixed head_dim/heads.
//!
//! Measures `flame_core::attention::sdpa` per-call latency over a range of
//! sequence lengths. Uses whatever SDPA backend the env selects — the caller
//! decides via `FLAME_USE_TORCH_SDPA=1` (opt into libtorch flash), or default
//! (in-tree WMMA kernel at `flash_attention_fwd.cu`).
//!
//! Output: one TIMING line per N, median over `ITERS` iterations after
//! `WARMUP` warmup calls.

use std::time::Instant;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

const HEADS: usize = 12;
const HEAD_DIM: usize = 128;
const WARMUP: usize = 3;
const ITERS: usize = 20;

fn bench_one(b: usize, h: usize, n: usize, d: usize) -> anyhow::Result<()> {
    let device = global_cuda_device();
    let shape = Shape::from_dims(&[b, h, n, d]);
    // Use random-ish inputs — identical across calls so output is comparable.
    let q_data: Vec<f32> = (0..b * h * n * d).map(|i| ((i % 137) as f32 / 137.0 - 0.5) * 0.1).collect();
    let k_data: Vec<f32> = (0..b * h * n * d).map(|i| ((i % 211) as f32 / 211.0 - 0.5) * 0.1).collect();
    let v_data: Vec<f32> = (0..b * h * n * d).map(|i| ((i % 89)  as f32 / 89.0  - 0.5) * 0.1).collect();
    let q = Tensor::from_vec(q_data, shape.clone(), device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(k_data, shape.clone(), device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(v_data, shape.clone(), device.clone())?.to_dtype(DType::BF16)?;

    // Warmup
    for _ in 0..WARMUP {
        let _o = flame_core::attention::sdpa(&q, &k, &v, None)?;
        // Force D2H sync so the timing below measures per-call wall clock.
        let _ = _o.to_dtype(DType::F32)?.to_vec_f32()?;
    }

    let mut times_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let o = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let _ = o.to_dtype(DType::F32)?.to_vec_f32()?; // host sync
        times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_ms[ITERS / 2];
    let min = times_ms[0];
    let max = times_ms[ITERS - 1];
    let mean: f64 = times_ms.iter().sum::<f64>() / ITERS as f64;

    let backend = if std::env::var("FLAME_USE_TORCH_SDPA").ok().as_deref() == Some("1") {
        "torch"
    } else if std::env::var("FLAME_NO_TORCH_SDPA").ok().as_deref() == Some("0") {
        "torch(legacy)"
    } else {
        "wmma"
    };
    println!(
        "TIMING backend={} B={} H={} N={:>5} D={} mean={:.2}ms median={:.2}ms min={:.2}ms max={:.2}ms",
        backend, b, h, n, d, mean, median, min, max
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Include motif 496 (full dump), motif video-res 5120, zimage/flux 4096.
    let n_list: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n_list = if n_list.is_empty() {
        vec![256, 512, 1024, 2048, 4096]
    } else {
        n_list
    };
    for n in n_list {
        bench_one(1, HEADS, n, HEAD_DIM)?;
    }
    Ok(())
}
