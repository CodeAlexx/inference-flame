//! Phase 4 end-to-end validation: run a small Klein 4B forward via the
//! existing in-process CPU staging path (`KleinOffloaded::forward`) and
//! again via FlameSwap (`KleinOffloaded::forward_with_swap`), then compare
//! the resulting latent tensors numerically.
//!
//! This is a forward-only check: the goal is to prove that swap-driven
//! block weights produce bit-identical (within BF16 noise) outputs to the
//! sync staging path.  No text encoder, no VAE, no image — those add
//! megabytes of dependent code that don't help us validate FlameSwap.
//!
//! Inputs are deterministic synthetic tensors so the comparison is
//! reproducible run-to-run.

use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use flame_swap::FlameSwap;
use inference_flame::models::klein::KleinOffloaded;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors";

fn make_inputs(
    inner_dim: usize,
    in_channels: usize,
    joint_dim: usize,
    n_img: usize,
    n_txt: usize,
    latent_h: usize,
    latent_w: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    // img: [1, n_img, in_channels] BF16
    let img: Vec<f32> = (0..n_img * in_channels)
        .map(|i| ((i % 257) as f32 - 128.0) / 256.0)
        .collect();
    let img = Tensor::from_f32_to_bf16(
        img,
        Shape::from_dims(&[1, n_img, in_channels]),
        device.clone(),
    )?;

    // txt: [1, n_txt, joint_dim] BF16
    let txt: Vec<f32> = (0..n_txt * joint_dim)
        .map(|i| ((i % 311) as f32 - 155.0) / 256.0)
        .collect();
    let txt = Tensor::from_f32_to_bf16(
        txt,
        Shape::from_dims(&[1, n_txt, joint_dim]),
        device.clone(),
    )?;

    // timesteps: [1] sigma value
    let t = Tensor::from_f32_to_bf16(
        vec![0.5f32],
        Shape::from_dims(&[1]),
        device.clone(),
    )?;

    // img_ids: [n_img, 4]
    let mut img_ids = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_ids[idx * 4 + 1] = r as f32;
            img_ids[idx * 4 + 2] = c as f32;
        }
    }
    let img_ids = Tensor::from_f32_to_bf16(
        img_ids,
        Shape::from_dims(&[n_img, 4]),
        device.clone(),
    )?;

    // txt_ids: [n_txt, 4] zeros
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[n_txt, 4]),
        DType::BF16,
        device.clone(),
    )?;

    // hint: silence the warning about unused inner_dim
    let _ = inner_dim;

    Ok((img, txt, t, img_ids, txt_ids))
}

fn tensor_stats(label: &str, t: &Tensor) -> anyhow::Result<(f32, f32)> {
    let v = t.to_vec()?;
    let n = v.len() as f32;
    let mean: f32 = v.iter().sum::<f32>() / n;
    let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "  {label}: shape={:?} mean={:.6} std={:.6} min={:.6} max={:.6}",
        t.shape().dims(), mean, std, min, max,
    );
    Ok((mean, std))
}

fn diff_latents(a: &Tensor, b: &Tensor) -> anyhow::Result<(f32, f32, usize, usize)> {
    let av = a.to_vec()?;
    let bv = b.to_vec()?;
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut n_diff_1e3 = 0usize;
    let mut n_diff_1e2 = 0usize;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > max_abs { max_abs = d; }
        sum_abs += d;
        if d > 1e-3 { n_diff_1e3 += 1; }
        if d > 1e-2 { n_diff_1e2 += 1; }
    }
    let mean_abs = sum_abs / av.len() as f32;
    Ok((max_abs, mean_abs, n_diff_1e3, n_diff_1e2))
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("--- Klein 4B FlameSwap end-to-end validation ---");
    println!("Model: {MODEL_PATH}");

    println!("\nLoading KleinOffloaded (shared on GPU + cpu_weights for sync path)...");
    let t0 = Instant::now();
    let model = KleinOffloaded::from_safetensors(MODEL_PATH)?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let cfg = model.config().clone();
    let total_blocks = cfg.num_double + cfg.num_single;
    println!(
        "  config: inner_dim={} in_channels={} joint_dim={} double={} single={} ({} swap blocks total)",
        cfg.inner_dim, cfg.in_channels, cfg.joint_attention_dim,
        cfg.num_double, cfg.num_single, total_blocks,
    );

    // Small synthetic forward inputs.  256x256 latent grid → 16x16 packed.
    let latent_h = 16usize;
    let latent_w = 16usize;
    let n_img = latent_h * latent_w;
    let n_txt = 64usize;
    let (img_in, txt_in, t_in, img_ids, txt_ids) = make_inputs(
        cfg.inner_dim,
        cfg.in_channels,
        cfg.joint_attention_dim,
        n_img,
        n_txt,
        latent_h,
        latent_w,
        &device,
    )?;
    println!(
        "  inputs: img={:?} txt={:?} timesteps={:?}",
        img_in.shape().dims(), txt_in.shape().dims(), t_in.shape().dims(),
    );

    // ---- Sync path ----
    println!("\n--- Sync forward (CPU staging dict) ---");
    let t0 = Instant::now();
    let out_sync = model.forward(&img_in, &txt_in, &t_in, &img_ids, &txt_ids)?;
    let sync_ms = t0.elapsed().as_millis();
    println!("  forward in {sync_ms}ms");
    tensor_stats("out_sync", &out_sync)?;

    // ---- Init FlameSwap with the same checkpoint + Klein block_fn ----
    println!("\n--- Initialising FlameSwap on the same checkpoint ---");
    let num_double = cfg.num_double;
    let mut swap = FlameSwap::load(
        &[MODEL_PATH],
        &device,
        |name| {
            if let Some(rest) = name.strip_prefix("double_blocks.") {
                rest.split('.').next()?.parse().ok()
            } else if let Some(rest) = name.strip_prefix("single_blocks.") {
                let idx: usize = rest.split('.').next()?.parse().ok()?;
                Some(num_double + idx)
            } else {
                None
            }
        },
    ).map_err(|e| anyhow::anyhow!("FlameSwap::load: {e}"))?;

    // ---- Swap path ----
    println!("\n--- Swap forward (FlameSwap v2) ---");
    let t0 = Instant::now();
    let out_swap = model.forward_with_swap(
        &img_in, &txt_in, &t_in, &img_ids, &txt_ids, &mut swap,
    )?;
    let swap_ms = t0.elapsed().as_millis();
    println!("  forward in {swap_ms}ms");
    tensor_stats("out_swap", &out_swap)?;

    // ---- Compare ----
    println!("\n--- Numerical diff ---");
    let (max_abs, mean_abs, n_1e3, n_1e2) = diff_latents(&out_sync, &out_swap)?;
    let total = out_sync.shape().elem_count();
    println!(
        "  max_abs={:.6e}  mean_abs={:.6e}  total_elems={}",
        max_abs, mean_abs, total,
    );
    println!(
        "  > 1e-3: {} ({:.2}%)   > 1e-2: {} ({:.4}%)",
        n_1e3, n_1e3 as f64 * 100.0 / total as f64,
        n_1e2, n_1e2 as f64 * 100.0 / total as f64,
    );

    // Acceptance: BF16-on-BF16 path should be near-exact.  Allow 1e-2 max
    // (the spec target) so transient floating point reorderings between
    // dispatch orders don't trip the check.
    if max_abs <= 1e-2 {
        println!("\n--- PASS (max_abs {max_abs:.4e} ≤ 1e-2) ---");
        Ok(())
    } else {
        Err(anyhow::anyhow!("FAIL: max_abs {max_abs} exceeds 1e-2 tolerance"))
    }
}
