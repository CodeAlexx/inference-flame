//! Parity test: compare my Rust DiT one-step output against the diffusers
//! reference at the SAME prompt + seed + intermediate inputs.
//!
//! Workflow:
//! 1. `dump_nucleus_reference.py` runs the diffusers pipeline on the prompt
//!    "an orange tabby cat sitting on a wooden table, photorealistic" with
//!    seed=42 at 512² and dumps prompt_embeds (cond + uncond), the packed
//!    initial latent, and one denoise step's velocities (cond / uncond /
//!    CFG-combined) to `diffusers_reference.safetensors`.
//! 2. This binary loads those tensors, builds my `NucleusInferDit`, runs
//!    one DiT forward each for cond and uncond using diffusers's
//!    prompt_embeds and initial latent (i.e. all my upstream stages
//!    bypassed), then computes the CFG combine + Zero* rescale, and
//!    prints diff stats vs each diffusers reference.
//!
//! If `velocity.step_0_cond` matches within BF16 noise, my DiT forward is
//! correct. Any divergence localizes the bug to the DiT or its inputs.
//!
//! Run:
//!   FLAME_ALLOC_POOL=0 cargo run --release --bin nucleus_parity

use anyhow::{anyhow, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use inference_flame::models::nucleus_dit::NucleusInferDit;
use inference_flame::vae::qwenimage_decoder::QwenImageVaeDecoder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const SNAPSHOT: &str = "/home/alex/.cache/huggingface/hub/models--NucleusAI--Nucleus-Image/snapshots/4c5d6738f71bb5ea8dea55f4b0c0fd911e0deefd";
const FIXTURE: &str = "tests/pytorch_fixtures/nucleus/diffusers_reference.safetensors";

/// `(B, S_img, Z*P²)` → `(B, Z, 1, H, W)`.
fn unpack_latents(latents: &Tensor, h_grid: usize, w_grid: usize, patch: usize) -> Result<Tensor> {
    let dims = latents.shape().dims();
    let (b, _s, channels) = (dims[0], dims[1], dims[2]);
    let z = channels / (patch * patch);
    let v = latents.reshape(&[b, h_grid, w_grid, z, patch, patch])?;
    let v = v.permute(&[0, 3, 1, 4, 2, 5])?.contiguous()?;
    let h = h_grid * patch;
    let w = w_grid * patch;
    Ok(v.reshape(&[b, z, 1, h, w])?)
}

fn norm_along_last(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.mul(&x_f32)?;
    let last = sq.shape().dims().len() - 1;
    let sum_sq = sq.sum_dim_keepdim(last)?;
    let norm = sum_sq.sqrt()?;
    Ok(norm.to_dtype(DType::BF16)?)
}

fn diff_stats(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    if av.len() != bv.len() {
        return Err(anyhow!(
            "{label}: shape mismatch: rust={:?} py={:?}",
            a.shape().dims(),
            b.shape().dims()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;
    let mut sum_ab = 0.0f64;
    for (&x, &y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
        sum_a2 += (x as f64) * (x as f64);
        sum_b2 += (y as f64) * (y as f64);
        sum_ab += (x as f64) * (y as f64);
        let denom = x.abs().max(y.abs()).max(1e-6);
        let r = d / denom;
        if r > max_rel {
            max_rel = r;
        }
    }
    let mean_abs = (sum_abs / av.len() as f64) as f32;
    let cosine = (sum_ab / (sum_a2.sqrt() * sum_b2.sqrt())) as f32;
    let mean_a = av.iter().map(|&x| x.abs() as f64).sum::<f64>() / av.len() as f64;
    let mean_b = bv.iter().map(|&x| x.abs() as f64).sum::<f64>() / bv.len() as f64;
    eprintln!(
        "{label}: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} max_rel={max_rel:.2e} cos={cosine:.6} |rust|≈{:.4} |py|≈{:.4}",
        mean_a, mean_b
    );
    Ok(())
}

fn main() -> Result<()> {
    env_logger::try_init().ok();
    let device: Arc<CudaDevice> = CudaDevice::new(0)?;

    eprintln!("=== Nucleus parity (DiT one-step vs diffusers) ===");

    // --- 1. Load reference fixture ---
    let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    if !fixture_path.exists() {
        return Err(anyhow!(
            "fixture missing — generate with `python3 scripts/dump_nucleus_reference.py`"
        ));
    }
    let mut map = flame_core::serialization::load_file(&fixture_path, &device)?;

    let cond_embeds = map
        .remove("prompt_embeds.cond")
        .ok_or_else(|| anyhow!("missing prompt_embeds.cond"))?;
    let uncond_embeds = map
        .remove("prompt_embeds.uncond")
        .ok_or_else(|| anyhow!("missing prompt_embeds.uncond"))?;
    let initial_latent = map
        .remove("latents.initial")
        .ok_or_else(|| anyhow!("missing latents.initial"))?;
    let timestep = map
        .remove("inputs.timestep")
        .ok_or_else(|| anyhow!("missing inputs.timestep"))?;
    let ref_v_cond = map
        .remove("velocity.step_0_cond")
        .ok_or_else(|| anyhow!("missing velocity.step_0_cond"))?;
    let ref_v_uncond = map
        .remove("velocity.step_0_uncond")
        .ok_or_else(|| anyhow!("missing velocity.step_0_uncond"))?;
    let ref_v_cfg = map
        .remove("velocity.step_0_cfg")
        .ok_or_else(|| anyhow!("missing velocity.step_0_cfg"))?;
    let ref_latent_after = map
        .remove("latents.after_step_0")
        .ok_or_else(|| anyhow!("missing latents.after_step_0"))?;

    let h_grid = map
        .get("meta.h_grid")
        .ok_or_else(|| anyhow!("missing meta.h_grid"))?
        .to_vec_f32()?[0] as usize;
    let w_grid = map
        .get("meta.w_grid")
        .ok_or_else(|| anyhow!("missing meta.w_grid"))?
        .to_vec_f32()?[0] as usize;
    let guidance = map
        .get("meta.guidance_scale")
        .ok_or_else(|| anyhow!("missing meta.guidance_scale"))?
        .to_vec_f32()?[0];

    eprintln!(
        "fixture: cond_embeds={:?}  uncond_embeds={:?}  latent={:?}  h_grid={h_grid} w_grid={w_grid} guidance={guidance}",
        cond_embeds.shape().dims(),
        uncond_embeds.shape().dims(),
        initial_latent.shape().dims()
    );

    // --- 2. Load real Nucleus DiT ---
    let t0 = Instant::now();
    let transformer_dir = Path::new(SNAPSHOT).join("transformer");
    let mut dit = NucleusInferDit::load(&transformer_dir, device.clone())?;
    eprintln!("DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // --- 3. Build cond + uncond KV caches ---
    let img_shapes = (1, h_grid, w_grid);
    let cond_kv = dit
        .compute_kv_cache(&cond_embeds, img_shapes)
        .map_err(|e| anyhow!("compute_kv_cache(cond): {e}"))?;
    let uncond_kv = dit
        .compute_kv_cache(&uncond_embeds, img_shapes)
        .map_err(|e| anyhow!("compute_kv_cache(uncond): {e}"))?;
    eprintln!("KV caches built");

    // --- 4. Forward step 0 on cond + uncond ---
    let timestep_bf16 = timestep.to_dtype(DType::BF16)?;

    let t_cond = Instant::now();
    let v_cond = dit
        .forward_cached(&initial_latent, &timestep_bf16, img_shapes, &cond_kv)
        .map_err(|e| anyhow!("forward(cond): {e}"))?;
    eprintln!("cond forward: {:.1}s", t_cond.elapsed().as_secs_f32());
    diff_stats("v_cond  vs diffusers", &v_cond, &ref_v_cond)?;

    let t_uncond = Instant::now();
    let v_uncond = dit
        .forward_cached(&initial_latent, &timestep_bf16, img_shapes, &uncond_kv)
        .map_err(|e| anyhow!("forward(uncond): {e}"))?;
    eprintln!("uncond forward: {:.1}s", t_uncond.elapsed().as_secs_f32());
    diff_stats("v_uncond vs diffusers", &v_uncond, &ref_v_uncond)?;

    // --- 5. CFG combine + Zero* rescale ---
    let diff = v_cond.sub(&v_uncond)?;
    let scaled_diff = diff.mul_scalar(guidance)?;
    let comb = v_uncond.add(&scaled_diff)?;
    let cond_norm = norm_along_last(&v_cond)?;
    let comb_norm = norm_along_last(&comb)?;
    let ratio = cond_norm.div(&comb_norm)?;
    let v_cfg = comb.mul(&ratio)?;
    diff_stats("v_cfg   vs diffusers", &v_cfg, &ref_v_cfg)?;

    // --- 6. Apply scheduler step (Euler), check post-step latent ---
    // Pipeline: noise_pred = -v_cfg; latent_new = latent + (sigma_next - sigma) * noise_pred
    // For our setup sigma_0 = 1.0 (timestep[0]) and sigma_1 (next) = 1.0 - 1.0/30 = 0.9667.
    // dt = sigma - sigma_next = 1/30 ≈ 0.0333.
    // latent_new = latent + (1/30) * v_cfg.
    let sigma = timestep.to_vec_f32()?[0];
    let sigma_next = sigma - 1.0 / 30.0; // matches our sigma schedule
    let dt = sigma - sigma_next;
    let dv = v_cfg.mul_scalar(dt)?;
    let latent_after = initial_latent.add(&dv)?;
    diff_stats(
        "latent_after_step_0 vs diffusers",
        &latent_after,
        &ref_latent_after,
    )?;

    // --- 7. VAE parity: decode diffusers's final latent through MY VAE and
    //        compare to diffusers's decoded RGB image.
    //
    // Wan21VaeDecoder denormalizes internally (`z * std + mean`). So we should
    // pass the just-unpacked, NOT-yet-denormalized latent. To verify both
    // paths and prove the bug, we test:
    //   A) feed `latents.final` (packed, normalized) — unpack myself, decode.
    //   B) feed `latents.unpacked_denormalized` — should produce the OLD bug
    //      (double-denorm).
    let final_latent = map.remove("latents.final");
    let diff_image = map.remove("image.decoded");
    let unpacked_denorm = map.remove("latents.unpacked_denormalized");
    if let (Some(final_lat), Some(diff_image)) = (final_latent, diff_image) {
        eprintln!("\n--- VAE-only parity (diffusers final latent → my VAE) ---");
        // Drop DiT to free GPU before VAE.
        drop(cond_kv);
        drop(uncond_kv);
        drop(dit);
        flame_core::cuda_alloc_pool::clear_pool_cache();

        let vae_path = Path::new(SNAPSHOT)
            .join("vae")
            .join("diffusion_pytorch_model.safetensors");
        let vae = QwenImageVaeDecoder::from_safetensors(
            vae_path.to_str().unwrap(),
            &device,
        )
        .map_err(|e| anyhow!("VAE load: {e}"))?;
        eprintln!("VAE loaded");

        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("output");
        std::fs::create_dir_all(&out_dir)?;

        // Save reference for visual comparison.
        let diff_img_5d = diff_image.unsqueeze(2)?;
        save_png(&diff_img_5d, &out_dir, "parity_ref_diffusers_seed42")?;

        // PATH A — feed the un-pre-denorm latent through my VAE (correct path).
        // Unpack `latents.final` (packed model output, normalized) with my pack
        // helper, then decode. My VAE denormalizes internally.
        let dims = final_lat.shape().dims();
        eprintln!("latents.final shape: {dims:?}");
        let h_grid_v = (dims[1] as f64).sqrt() as usize;
        let w_grid_v = h_grid_v; // assume square
        let unpacked = unpack_latents(&final_lat, h_grid_v, w_grid_v, 2)?;
        eprintln!("unpacked (normalized) shape: {:?}", unpacked.shape().dims());

        let my_decoded_a = vae.decode(&unpacked).map_err(|e| anyhow!("VAE decode A: {e}"))?;
        let my_img_4d_a = my_decoded_a.squeeze(Some(2))?;
        diff_stats("VAE A (no pre-denorm) vs diffusers", &my_img_4d_a, &diff_image)?;
        save_png(&my_decoded_a, &out_dir, "parity_myvae_path_A_correct")?;

        // PATH B — feed pre-denormalized latent (the OLD bug path, expect bad).
        if let Some(unpacked_denorm) = unpacked_denorm {
            let my_decoded_b = vae.decode(&unpacked_denorm).map_err(|e| anyhow!("VAE decode B: {e}"))?;
            let my_img_4d_b = my_decoded_b.squeeze(Some(2))?;
            diff_stats("VAE B (double denorm — old bug) vs diffusers", &my_img_4d_b, &diff_image)?;
            save_png(&my_decoded_b, &out_dir, "parity_myvae_path_B_double_denorm")?;
        }
    } else {
        eprintln!("(latents.final / image.decoded missing — skipping VAE parity)");
    }

    eprintln!("\n--- summary ---");
    eprintln!("If v_cond mean_abs is sub-1e-1 we're in the BF16 noise regime; >1e-1 means a real bug.");
    Ok(())
}

/// Save a `[1, 3, 1, H, W]` BF16/F32 tensor in `[-1, 1]` as PNG.
fn save_png(image_tensor: &Tensor, out_dir: &Path, stem: &str) -> Result<PathBuf> {
    let dims = image_tensor.shape().dims();
    let (_b, c, _f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let data_f32 = image_tensor.to_dtype(DType::F32)?.to_vec_f32()?;
    let plane_size = h * w;
    let mut rgb: Vec<u8> = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let idx = ch * plane_size + y * w + x;
                let v = data_f32[idx];
                rgb.push(((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    let path = out_dir.join(format!("{stem}.png"));
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("RgbImage::from_raw"))?;
    img.save(&path)?;
    eprintln!("saved {}", path.display());
    Ok(path)
}
