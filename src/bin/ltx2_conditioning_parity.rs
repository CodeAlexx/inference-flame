//! LTX-2.3 multi-keyframe conditioning parity test.
//!
//! Compares `src/models/ltx2_conditioning.rs` against
//! `scripts/ltx2_conditioning_mask_ref.py`. Requires running the Python
//! script first:
//!
//!     python3 scripts/ltx2_conditioning_mask_ref.py
//!     cargo run --release --bin ltx2_conditioning_parity
//!
//! Passes if all tensors match at F32 within a tight tolerance (exact
//! for pure mask math, BF16-noise tolerance for noise injection).

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::ltx2_conditioning::{
    add_image_cond_noise, pack_conditioning_mask_for_transformer, prepare_conditioning,
    ConditioningItem,
};
use std::collections::HashMap;

const REF_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/ltx2_conditioning_mask_ref.safetensors";

fn max_abs_diff(a: &Tensor, b: &Tensor, label: &str) -> anyhow::Result<f64> {
    let av = a.to_dtype(DType::F32)?.to_vec()?;
    let bv = b.to_dtype(DType::F32)?.to_vec()?;
    if av.len() != bv.len() {
        anyhow::bail!(
            "{label}: length mismatch  rust={} python={}",
            av.len(),
            bv.len()
        );
    }
    let mut max = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        if d > max {
            max = d;
        }
    }
    Ok(max)
}

fn pass_or_fail(label: &str, diff: f64, tol: f64) -> bool {
    let ok = diff <= tol;
    println!(
        "  {label:<38}  max_abs={:.2e}  tol={:.0e}  {}",
        diff,
        tol,
        if ok { "OK" } else { "FAIL" }
    );
    ok
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("=== LTX-2.3 Conditioning Parity ===");
    println!("  ref: {}", REF_PATH);
    if !std::path::Path::new(REF_PATH).exists() {
        anyhow::bail!(
            "reference file missing; run `python3 scripts/ltx2_conditioning_mask_ref.py` first"
        );
    }

    let refs: HashMap<String, Tensor> =
        flame_core::serialization::load_file(std::path::Path::new(REF_PATH), &device)?;
    let get = |k: &str| {
        refs.get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing key: {k}"))
    };

    // Match the Python script's item layout (see its docstring).
    let init_latents = get("init_latents_input")?;
    let item0 = get("item0_latent")?;
    let item1 = get("item1_latent")?;
    let item2 = get("item2_latent")?;

    let strengths = get("_param_strengths")?.to_dtype(DType::F32)?.to_vec()?;
    let frame_numbers = get("_param_frame_numbers")?.to_dtype(DType::F32)?.to_vec()?;
    let param_t = get("_param_t")?.to_dtype(DType::F32)?.to_vec()?[0];
    let param_noise = get("_param_noise_scale")?.to_dtype(DType::F32)?.to_vec()?[0];

    let items = vec![
        ConditioningItem {
            latent: item0,
            frame_number: frame_numbers[0] as usize,
            strength: strengths[0],
        },
        ConditioningItem {
            latent: item1,
            frame_number: frame_numbers[1] as usize,
            strength: strengths[1],
        },
        ConditioningItem {
            latent: item2,
            frame_number: frame_numbers[2] as usize,
            strength: strengths[2],
        },
    ];

    // --- prepare_conditioning ---
    let (merged, mask5d) = prepare_conditioning(&init_latents, &items)?;

    let exp_merged = get("merged_latents")?;
    let exp_mask5d = get("conditioning_mask_5d")?;

    println!("\n-- prepare_conditioning --");
    let mut all_ok = true;
    all_ok &= pass_or_fail(
        "merged_latents",
        max_abs_diff(&merged, &exp_merged, "merged_latents")?,
        // Pure F32 lerp math against F32 reference → should be bit-exact.
        // Allow one ULP-worth of tolerance for FMA reorderings.
        1e-6,
    );
    all_ok &= pass_or_fail(
        "conditioning_mask_5d",
        max_abs_diff(&mask5d, &exp_mask5d, "conditioning_mask_5d")?,
        0.0,
    );

    // --- pack_conditioning_mask_for_transformer ---
    let packed = pack_conditioning_mask_for_transformer(&mask5d)?;
    let exp_packed = get("conditioning_mask_packed")?;
    all_ok &= pass_or_fail(
        "conditioning_mask_packed",
        max_abs_diff(&packed, &exp_packed, "conditioning_mask_packed")?,
        0.0,
    );

    // --- add_image_cond_noise ---
    let noise = get("noise_for_add_noise")?;
    let latents_after_rust = add_image_cond_noise(
        &init_latents, // init snapshot
        &merged,       // current latents (= merged pre-denoise in this test)
        &mask5d,
        &noise,
        param_t,
        param_noise,
    )?;
    let exp_after = get("latents_after_addnoise")?;

    println!("\n-- add_image_cond_noise --");
    all_ok &= pass_or_fail(
        "latents_after_addnoise",
        max_abs_diff(&latents_after_rust, &exp_after, "latents_after_addnoise")?,
        // where_mask does a*m + b*(1-m) rather than torch.where. With our
        // quantized binary mask this collapses to torch.where for mask==1
        // OR mask==0, which is exactly our case. Full F32 → bit-exact.
        1e-6,
    );

    println!(
        "\n{}",
        if all_ok {
            "ALL CHECKS OK"
        } else {
            "SOME CHECKS FAILED"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
