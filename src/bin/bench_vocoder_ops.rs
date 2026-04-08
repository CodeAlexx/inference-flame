//! Microbenchmark the individual tensor ops used by the vocoder hot path,
//! to find where the per-call overhead really lives.

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use flame_core::conv1d::{conv1d, conv1d_grouped};
use std::time::Instant;

const N_ITER: usize = 200;

fn bench<F: Fn() -> anyhow::Result<()>>(name: &str, f: F) -> anyhow::Result<()> {
    let device = global_cuda_device();
    // warmup
    for _ in 0..10 {
        f()?;
    }
    device.synchronize().ok();
    let t0 = Instant::now();
    for _ in 0..N_ITER {
        f()?;
    }
    device.synchronize().ok();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_call_us = elapsed_ms / N_ITER as f64 * 1000.0;
    println!("  {name:<50} {per_call_us:8.1} µs/call");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = global_cuda_device();
    println!("=== Vocoder op microbenchmark ===");
    println!("  (warmup 10, measure {N_ITER})\n");

    // Realistic AMPBlock1 shapes at stage 0 (L=80, C=768) and stage 5 (L=2560, C=24).
    for &(c, l, label) in &[(768usize, 80usize, "stage0"), (24usize, 2560usize, "stage5")] {
        println!("--- {label}: [1, {c}, {l}] ---");
        let x = Tensor::zeros_dtype(
            Shape::from_dims(&[1, c, l]),
            DType::BF16,
            device.clone(),
        )?;
        let w_small = Tensor::zeros_dtype(
            Shape::from_dims(&[c, c, 3]),
            DType::BF16,
            device.clone(),
        )?;
        let b_small = Tensor::zeros_dtype(
            Shape::from_dims(&[c]),
            DType::BF16,
            device.clone(),
        )?;
        let w_group = Tensor::zeros_dtype(
            Shape::from_dims(&[c, 1, 12]),
            DType::BF16,
            device.clone(),
        )?;

        // conv1d k=3 — the inner AMPBlock conv
        bench("conv1d k=3", || {
            let _ = conv1d(&x, &w_small, Some(&b_small), 1, 1, 1, 1)?;
            Ok(())
        })?;

        // grouped conv1d k=12 — the activation1d downsample
        bench("conv1d_grouped k=12 s=2", || {
            let padded = x.pad1d(5, 6)?;
            let _ = conv1d_grouped(&padded, &w_group, 2, 0, c)?;
            Ok(())
        })?;

        // tiny scalar ops
        bench("mul_scalar", || {
            let _ = x.mul_scalar(2.0)?;
            Ok(())
        })?;
        bench("sin", || {
            let _ = x.sin()?;
            Ok(())
        })?;
        bench("mul (elementwise same-shape)", || {
            let _ = x.mul(&x)?;
            Ok(())
        })?;
        bench("add (elementwise same-shape)", || {
            let _ = x.add(&x)?;
            Ok(())
        })?;

        // View ops
        bench("narrow", || {
            let _ = x.narrow(2, 1, l - 2)?;
            Ok(())
        })?;
        bench("reshape flat", || {
            let _ = x.reshape(&[1, c * l])?;
            Ok(())
        })?;
        bench("expand", || {
            let _ = x.narrow(2, 0, 1)?.expand(&[1, c, 5])?;
            Ok(())
        })?;

        // Cat 3 tensors (the replicate_pad core)
        bench("cat 3 tensors", || {
            let a = x.narrow(2, 0, 5)?;
            let b = x.clone();
            let c_ = x.narrow(2, l - 5, 5)?;
            let _ = Tensor::cat(&[&a, &b, &c_], 2)?;
            Ok(())
        })?;

        // Zero allocate
        bench("zeros alloc [B,C,L]", || {
            let _ = Tensor::zeros_dtype(
                Shape::from_dims(&[1, c, l]),
                DType::BF16,
                device.clone(),
            )?;
            Ok(())
        })?;

        println!();
    }
    Ok(())
}
