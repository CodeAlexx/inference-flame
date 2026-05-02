//! Smoke test for `flame_core::ops::deinterleave::deinterleave_pair_f32`.
//! Verifies the kernel output matches the legacy reshape→narrow→contiguous
//! reference path for a few small inputs.

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, Shape, Tensor};

fn dtoh(t: &Tensor) -> Result<Vec<f32>> {
    t.to_vec_f32().map_err(|e| anyhow!("to_vec_f32: {e:?}"))
}

fn from_host(host: &[f32], shape: &[usize]) -> Result<Tensor> {
    let dev = global_cuda_device();
    Tensor::from_slice(host, Shape::from_dims(shape), dev)
        .map_err(|e| anyhow!("from_slice: {e:?}"))
}

fn legacy_split(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = x.shape().dims().to_vec();
    let last = dims.len() - 1;
    let half = dims[last] / 2;
    let mut split_shape = dims.clone();
    split_shape[last] = half;
    split_shape.push(2);
    let x_split = x.reshape(&split_shape)?;
    let even = x_split
        .narrow(last + 1, 0, 1)?
        .squeeze(Some(last + 1))?
        .contiguous()?;
    let odd = x_split
        .narrow(last + 1, 1, 1)?
        .squeeze(Some(last + 1))?
        .contiguous()?;
    Ok((even, odd))
}

fn check(label: &str, host: &[f32], shape: &[usize]) -> Result<()> {
    let x = from_host(host, shape)?;
    let (e_new, o_new) = flame_core::ops::deinterleave::deinterleave_pair_f32(&x)?;
    let (e_legacy, o_legacy) = legacy_split(&x)?;
    let e_new_h = dtoh(&e_new)?;
    let o_new_h = dtoh(&o_new)?;
    let e_legacy_h = dtoh(&e_legacy)?;
    let o_legacy_h = dtoh(&o_legacy)?;
    assert_eq!(e_new.shape().dims(), e_legacy.shape().dims());
    assert_eq!(o_new.shape().dims(), o_legacy.shape().dims());
    let e_match = e_new_h == e_legacy_h;
    let o_match = o_new_h == o_legacy_h;
    println!(
        "[{label}] shape={shape:?}  even={}  odd={}  e_kernel={:?} ... (len {})  e_legacy={:?} ... (len {})",
        if e_match { "OK" } else { "MISMATCH" },
        if o_match { "OK" } else { "MISMATCH" },
        &e_new_h[..e_new_h.len().min(8)],
        e_new_h.len(),
        &e_legacy_h[..e_legacy_h.len().min(8)],
        e_legacy_h.len(),
    );
    if !e_match || !o_match {
        // dump first few mismatches
        for k in 0..e_new_h.len().min(16) {
            if e_new_h[k] != e_legacy_h[k] {
                println!("    even mismatch [{k}]: kernel={} legacy={}", e_new_h[k], e_legacy_h[k]);
            }
        }
        anyhow::bail!("[{label}] mismatch");
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    // tiny
    let tiny: Vec<f32> = (0..8).map(|i| i as f32).collect();
    check("tiny [2,4]", &tiny, &[2, 4])?;

    // 1D shape (rank 1)
    check("rank1 [8]", &tiny, &[8])?;

    // larger, irregular
    let mid: Vec<f32> = (0..(13 * 32)).map(|i| (i as f32) * 0.01 - 0.5).collect();
    check("[13, 32]", &mid, &[13, 32])?;

    // big, swiglu-like sizes (834 * 2 * 5504)
    let big_n = 834 * 2 * 1024; // smaller than real but realistic
    let big: Vec<f32> = (0..big_n).map(|i| ((i as i64 * 7919) % 1000) as f32 / 1000.0).collect();
    check("[834, 2048]", &big, &[834, 2048])?;

    println!("All deinterleave smoke checks passed.");
    Ok(())
}
