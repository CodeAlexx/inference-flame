//! Qwen-Image-Edit Stage-2 denoise — Turbo Flame Phase 2 (VMM-backed
//! double-buffered block loader).
//!
//! Same pipeline as `qwenimage_edit_gen` but the per-block weight staging
//! goes through `inference_flame::turbo::TurboBlockLoader` instead of the
//! BlockOffloader.
//!
//! Behind feature `turbo`. On a non-VMM-capable device this binary logs a
//! hint and exits with code 2 — no silent fallback.
//!
//! ## Sharded checkpoint caveat
//!
//! `TurboBlockLoader::new` reads a single safetensors file. Qwen-Image-Edit-2511
//! ships as 5 diffusers shards. Set `QWEN_TURBO_SAFETENSORS=/path/to/single.safetensors`
//! to point this binary at a merged single-file checkpoint. (Sharded support
//! would require extending the Phase 1 loader, which is out of scope for
//! Phase 2.)

#![cfg(feature = "turbo")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::qwenimage_dit::QwenImageDit;
use inference_flame::turbo::{TurboBlockLoader, VmmArena, VmmError};

// Mirror constants from qwenimage_edit_gen.rs.
const DEFAULT_DIT_SHARDS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
];

const VAE_SCALE_FACTOR: usize = 8;
const PATCH_SIZE: usize = 2;
const IN_CHANNELS: usize = 16;
const PACKED_CHANNELS: usize = 64;

// Qwen-Image config (must match qwenimage_dit::QwenImageConfig::default()).
const QWEN_NUM_LAYERS: usize = 60;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_embeds.safetensors".to_string()
    });
    let out_latents = args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_latents_turbo.safetensors".to_string()
    });

    let num_steps: usize = env_usize("QWEN_STEPS", 50);
    let true_cfg_scale: f32 = env_f32("QWEN_CFG", 4.0);
    let seed: u64 = env_u64("QWEN_SEED", 42);
    let noise_path: Option<String> = std::env::var("QWEN_NOISE_PATH").ok();
    let dump_dir: Option<String> = std::env::var("QWEN_DUMP_DIR").ok();
    if let Some(ref d) = dump_dir {
        std::fs::create_dir_all(d).ok();
        println!("  [dump] per-step tensors → {d}");
    }
    if let Some(ref p) = noise_path {
        println!("  [noise] loading from {p} (RNG seed ignored)");
    }

    let dit_shards_owned: Vec<String> = match std::env::var("QWEN_DIT_SHARDS") {
        Ok(s) => s.split(':').map(|p| p.to_string()).collect(),
        Err(_) => DEFAULT_DIT_SHARDS.iter().map(|s| s.to_string()).collect(),
    };
    let dit_shards: Vec<&str> = dit_shards_owned.iter().map(|s| s.as_str()).collect();

    // TurboBlockLoader needs a single safetensors file. Default to the first
    // shard but error out clearly if no override is set — sharded support is
    // out of scope for Phase 2.
    let turbo_model_path = std::env::var("QWEN_TURBO_SAFETENSORS")
        .unwrap_or_else(|_| dit_shards_owned[0].clone());

    println!("=== Qwen-Image-Edit — Turbo Flame Phase 2 (VMM) ===");
    println!("Embeddings: {}", embeds_path);
    println!("Output lat: {}", out_latents);
    println!("Steps:      {}, true_cfg: {}, seed: {}", num_steps, true_cfg_scale, seed);
    println!("Turbo file: {}", turbo_model_path);
    println!();

    // ------------------------------------------------------------------
    // Stage A: load cached cond + uncond + image_latents + dimensions
    // ------------------------------------------------------------------
    println!("--- Loading cached embeddings ---");
    let t0 = Instant::now();
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&embeds_path),
        &device,
    )?;
    let cond = ensure_bf16(tensors.get("cond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'cond'"))?
        .clone())?;
    let uncond = ensure_bf16(tensors.get("uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'uncond'"))?
        .clone())?;
    let image_latents = ensure_bf16(tensors.get("image_latents")
        .ok_or_else(|| anyhow::anyhow!("Missing 'image_latents'"))?
        .clone())?;
    let image_h_t = tensors.get("image_h")
        .ok_or_else(|| anyhow::anyhow!("Missing 'image_h'"))?
        .clone()
        .to_dtype(DType::F32)?;
    let image_w_t = tensors.get("image_w")
        .ok_or_else(|| anyhow::anyhow!("Missing 'image_w'"))?
        .clone()
        .to_dtype(DType::F32)?;
    drop(tensors);

    // Pre-encoded input tensors (cond/uncond/image_latents + their duplicated
    // map entries) are in the pool. Release before loading DiT — same pattern
    // as other turbo bins.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);

    let image_h_vec = image_h_t.to_vec_f32()?;
    let image_w_vec = image_w_t.to_vec_f32()?;
    if image_h_vec.is_empty() {
        return Err(anyhow::anyhow!("'image_h' tensor is empty"));
    }
    if image_w_vec.is_empty() {
        return Err(anyhow::anyhow!("'image_w' tensor is empty"));
    }
    let raw_h = image_h_vec[0];
    let raw_w = image_w_vec[0];
    if !(raw_h.is_finite() && (1.0..=8192.0).contains(&raw_h)) {
        return Err(anyhow::anyhow!("'image_h' out of valid range [1, 8192]: {}", raw_h));
    }
    if !(raw_w.is_finite() && (1.0..=8192.0).contains(&raw_w)) {
        return Err(anyhow::anyhow!("'image_w' out of valid range [1, 8192]: {}", raw_w));
    }
    let height = raw_h as usize;
    let width = raw_w as usize;
    if height % (VAE_SCALE_FACTOR * PATCH_SIZE) != 0
        || width % (VAE_SCALE_FACTOR * PATCH_SIZE) != 0
    {
        return Err(anyhow::anyhow!(
            "image dimensions ({}x{}) must be multiples of {} (VAE_SCALE_FACTOR*PATCH_SIZE)",
            width, height, VAE_SCALE_FACTOR * PATCH_SIZE
        ));
    }

    println!("  cond:          {:?}", cond.shape().dims());
    println!("  uncond:        {:?}", uncond.shape().dims());
    println!("  image_latents: {:?}", image_latents.shape().dims());
    println!("  target dims:   {}x{}", width, height);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage B: Load Qwen-Image-Edit DiT (BlockOffloader for shared weights)
    //          + TurboBlockLoader (VMM) for per-block staging
    // ------------------------------------------------------------------
    println!("--- Loading Qwen-Image-Edit DiT (Turbo / VMM) ---");
    let t0 = Instant::now();
    let dit = QwenImageDit::load(&dit_shards, &device)?;
    println!("  Shared weights loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let copy_stream = Arc::new(
        device
            .fork_default_stream()
            .map_err(|e| anyhow::anyhow!("create copy stream: {e:?}"))?,
    );

    let arena = match VmmArena::new_for_qwen_image_edit(device.clone(), copy_stream.clone()) {
        Ok(a) => Arc::new(a),
        Err(VmmError::Unsupported) => {
            log::error!(
                "VMM unsupported on device {}, re-run with qwenimage_edit_gen (non-turbo)",
                device.ordinal()
            );
            std::process::exit(2);
        }
        Err(e) => {
            log::error!("VMM arena init failed: {e}");
            std::process::exit(2);
        }
    };

    // 60 transformer blocks: prefix `transformer_blocks.{i}.`.
    let block_prefixes: Vec<String> = (0..QWEN_NUM_LAYERS)
        .map(|i| format!("transformer_blocks.{i}."))
        .collect();

    let mut loader = TurboBlockLoader::new(
        turbo_model_path.clone(),
        device.clone(),
        arena.clone(),
        block_prefixes,
    )
    .map_err(|e| anyhow::anyhow!(
        "TurboBlockLoader: {e}\n\
         hint: Phase 2 turbo expects a single-file Qwen-Image-Edit safetensors. Set \
         QWEN_TURBO_SAFETENSORS=/path/to/qwen_image_edit_2511.safetensors and re-run."
    ))?;

    println!(
        "  loader: {} blocks, {:.1} MiB pinned host  ({:.1}s total)",
        loader.block_count(),
        loader.pinned_bytes() as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f32(),
    );

    // ------------------------------------------------------------------
    // Stage C: Build noise for the TARGET region
    // ------------------------------------------------------------------
    println!("\n--- Denoise ({} steps, true_cfg={}) ---", num_steps, true_cfg_scale);

    let target_h = height;
    let target_w = width;
    let h_latent_full = target_h / VAE_SCALE_FACTOR;
    let w_latent_full = target_w / VAE_SCALE_FACTOR;
    let h_patched = h_latent_full / PATCH_SIZE;
    let w_patched = w_latent_full / PATCH_SIZE;
    let target_seq_len = h_patched * w_patched;

    let ref_dims = image_latents.shape().dims().to_vec();
    if ref_dims.len() != 3 || ref_dims[2] != PACKED_CHANNELS {
        return Err(anyhow::anyhow!(
            "Bad image_latents shape {:?} — expected [1, seq, {}]",
            ref_dims, PACKED_CHANNELS
        ));
    }
    let ref_seq_len = ref_dims[1];
    if ref_seq_len != target_seq_len {
        return Err(anyhow::anyhow!(
            "ref_seq_len {} != target_seq_len {}",
            ref_seq_len, target_seq_len
        ));
    }

    let mut latents = if let Some(np) = &noise_path {
        let loaded = flame_core::serialization::load_file(
            std::path::Path::new(np),
            &device,
        )?;
        let t = loaded.get("noise_packed")
            .ok_or_else(|| anyhow::anyhow!("noise file missing 'noise_packed' key"))?
            .clone();
        let d = t.shape().dims();
        if d.len() != 3 || d[0] != 1 || d[1] != target_seq_len || d[2] != PACKED_CHANNELS {
            return Err(anyhow::anyhow!(
                "Bad 'noise_packed' shape {:?}, expected [1, {}, {}]",
                d, target_seq_len, PACKED_CHANNELS
            ));
        }
        ensure_bf16(t)?
    } else {
        let numel = IN_CHANNELS * h_latent_full * w_latent_full;
        let noise_data: Vec<f32> = {
            use rand::prelude::*;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut v = Vec::with_capacity(numel);
            for _ in 0..numel / 2 {
                let u1: f32 = rng.gen::<f32>().max(1e-10);
                let u2: f32 = rng.gen::<f32>();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                v.push(r * theta.cos());
                v.push(r * theta.sin());
            }
            if numel % 2 == 1 {
                let u1: f32 = rng.gen::<f32>().max(1e-10);
                let u2: f32 = rng.gen::<f32>();
                v.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
            }
            v
        };

        let mut packed = vec![0.0f32; target_seq_len * PACKED_CHANNELS];
        for c in 0..IN_CHANNELS {
            for hp in 0..h_patched {
                for wp in 0..w_patched {
                    for dh in 0..PATCH_SIZE {
                        for dw in 0..PATCH_SIZE {
                            let src_h = hp * PATCH_SIZE + dh;
                            let src_w = wp * PATCH_SIZE + dw;
                            let src_idx = c * h_latent_full * w_latent_full + src_h * w_latent_full + src_w;
                            let dst_seq = hp * w_patched + wp;
                            let dst_chan = c * PATCH_SIZE * PATCH_SIZE + dh * PATCH_SIZE + dw;
                            packed[dst_seq * PACKED_CHANNELS + dst_chan] = noise_data[src_idx];
                        }
                    }
                }
            }
        }
        Tensor::from_f32_to_bf16(
            packed,
            Shape::from_dims(&[1, target_seq_len, PACKED_CHANNELS]),
            device.clone(),
        )?
    };

    if let Some(ref d) = dump_dir {
        let mut tmap: HashMap<String, Tensor> = HashMap::new();
        tmap.insert("noise_packed".to_string(), latents.clone());
        flame_core::serialization::save_file(&tmap, &format!("{d}/noise.safetensors"))?;
    }

    // Sigma schedule (FlowMatchEulerDiscreteScheduler) — same constants as
    // qwenimage_edit_gen.rs.
    let base_shift: f32 = env_f32("QWEN_BASE_SHIFT", 0.5);
    let max_shift: f32 = env_f32("QWEN_MAX_SHIFT", 0.9);
    let base_seq_len: f32 = env_f32("QWEN_BASE_SEQ_LEN", 256.0);
    let max_seq_len_shift: f32 = env_f32("QWEN_MAX_SEQ_LEN", 8192.0);
    let shift_terminal: f32 = env_f32("QWEN_SHIFT_TERMINAL", 0.02);

    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len);
    let bb = base_shift - m * base_seq_len;
    let mu = (target_seq_len as f32) * m + bb;
    let exp_mu = mu.exp();

    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / (num_steps - 1) as f32;
            1.0 - t * (1.0 - 1.0 / num_steps as f32)
        })
        .collect();
    for s in sigmas.iter_mut() {
        let denom = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / denom;
    }
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    if one_minus_last.abs() > 1e-12 {
        let scale = one_minus_last / (1.0 - shift_terminal);
        for s in sigmas.iter_mut() {
            let o = 1.0 - *s;
            *s = 1.0 - o / scale;
        }
    }
    sigmas.push(0.0);
    println!("  sigmas[0]={:.4}  sigmas[-2]={:.4}  sigmas[-1]={:.4}",
        sigmas[0], sigmas[num_steps - 1], sigmas[num_steps]);

    // Two regions: target + reference.
    let regions = vec![
        (1usize, h_patched, w_patched),
        (1usize, h_patched, w_patched),
    ];

    // CFG Euler loop — turbo version uses forward_edit_with_ref_timestep_turbo.
    let t_denoise = Instant::now();
    for step in 0..num_steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;

        let latents_in_snap = dump_dir.as_ref().map(|_| latents.clone());

        let (next_x, dump_snap) = {
            let t_vec = Tensor::from_vec(
                vec![sigma_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            let concat_input = Tensor::cat(&[&latents, &image_latents], 1)?;

            // zero_cond_t: reference uses a near-zero timestep so the model
            // treats it as clean conditioning. Same QWEN_T_REF env knob.
            let t_ref_val: f32 = std::env::var("QWEN_T_REF")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let t_ref_vec = Tensor::from_vec(
                vec![t_ref_val],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            // Turbo dispatch — analogous to lines 393/396 in qwenimage_edit_gen.rs.
            let cond_pred_full = dit.forward_edit_with_ref_timestep_turbo(
                &concat_input, &cond, &t_vec, Some(&t_ref_vec), &regions, &mut loader,
            )?;
            let uncond_pred_full = dit.forward_edit_with_ref_timestep_turbo(
                &concat_input, &uncond, &t_vec, Some(&t_ref_vec), &regions, &mut loader,
            )?;

            let expected_seq = target_seq_len + ref_seq_len;
            let cd = cond_pred_full.shape().dims();
            let ud = uncond_pred_full.shape().dims();
            if cd.len() != 3 || cd[1] != expected_seq || cd[2] != PACKED_CHANNELS {
                return Err(anyhow::anyhow!(
                    "forward_edit_turbo (cond) returned shape {:?}, expected [_, {}, {}]",
                    cd, expected_seq, PACKED_CHANNELS
                ));
            }
            if ud.len() != 3 || ud[1] != expected_seq || ud[2] != PACKED_CHANNELS {
                return Err(anyhow::anyhow!(
                    "forward_edit_turbo (uncond) returned shape {:?}, expected [_, {}, {}]",
                    ud, expected_seq, PACKED_CHANNELS
                ));
            }

            let cond_pred = cond_pred_full.narrow(1, 0, target_seq_len)?;
            let uncond_pred = uncond_pred_full.narrow(1, 0, target_seq_len)?;

            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(true_cfg_scale)?;
            let comb = uncond_pred.add(&scaled)?;

            let noise_pred = norm_rescale_cfg(&cond_pred, &comb).unwrap_or(comb);

            let step_tensor = noise_pred.mul_scalar(dt)?;
            let out = latents.add(&step_tensor)?;

            let snap = if dump_dir.is_some() {
                Some((cond_pred_full.clone(), uncond_pred_full.clone(), noise_pred.clone()))
            } else {
                None
            };
            (out, snap)
        };
        latents = next_x;

        if let (Some(d), Some((cond_full, uncond_full, pred_cfg))) =
            (dump_dir.as_ref(), dump_snap)
        {
            let mut tmap: HashMap<String, Tensor> = HashMap::new();
            if let Some(li) = latents_in_snap {
                tmap.insert("latents_in".to_string(), li);
            }
            tmap.insert("cond_pred_full".to_string(), cond_full);
            tmap.insert("uncond_pred_full".to_string(), uncond_full);
            tmap.insert("noise_pred_cfg".to_string(), pred_cfg);
            tmap.insert("latents_out".to_string(), latents.clone());
            tmap.insert(
                "sigma_curr".to_string(),
                Tensor::from_vec(vec![sigma_curr], Shape::from_dims(&[1]), device.clone())?,
            );
            tmap.insert(
                "sigma_next".to_string(),
                Tensor::from_vec(vec![sigma_next], Shape::from_dims(&[1]), device.clone())?,
            );
            flame_core::serialization::save_file(
                &tmap,
                &format!("{d}/step_{:03}.safetensors", step),
            )?;
        }

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps {
            println!(
                "  step {}/{}  sigma={:.4}  ({:.1}s elapsed)",
                step + 1,
                num_steps,
                sigma_curr,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step)",
        dt_denoise,
        dt_denoise / num_steps as f32,
    );

    drop(loader);
    drop(dit);
    drop(cond);
    drop(uncond);
    drop(image_latents);

    // DiT shared weights + VMM slot physical back in the pool. Release to
    // driver — keeps this bin's post-run footprint clean for any follow-up
    // tool reading the saved latents.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);

    // ------------------------------------------------------------------
    // Stage D: save latents
    // ------------------------------------------------------------------
    println!("\n--- Saving target latents ---");
    if let Some(parent) = std::path::Path::new(&out_latents).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut output = HashMap::new();
    output.insert("packed_latent".to_string(), latents);
    output.insert(
        "height".to_string(),
        Tensor::from_vec(vec![height as f32], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?,
    );
    output.insert(
        "width".to_string(),
        Tensor::from_vec(vec![width as f32], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?,
    );
    flame_core::serialization::save_file(&output, &out_latents)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("LATENTS SAVED: {}", out_latents);
    println!("Total time:    {:.1}s", dt_total);
    println!("============================================================");

    let _ = device;
    Ok(())
}

fn norm_rescale_cfg(cond: &Tensor, comb: &Tensor) -> anyhow::Result<Tensor> {
    let cond_sq = cond.mul(cond)?;
    let comb_sq = comb.mul(comb)?;
    let cond_sum = cond_sq.sum_dim_keepdim(2)?;
    let comb_sum = comb_sq.sum_dim_keepdim(2)?;
    let cond_norm = cond_sum.sqrt()?;
    let comb_norm = comb_sum.sqrt()?;
    let ratio = cond_norm.div(&comb_norm)?;
    let out = comb.mul(&ratio)?;
    Ok(out)
}

fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        Ok(t.to_dtype(DType::BF16)?)
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
