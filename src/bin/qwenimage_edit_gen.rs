//! Qwen-Image-Edit — Stage 2 (Rust DiT denoise → save latents).
//!
//! Loads cached embeddings + image_latents from `qwenimage_edit_encode.py`,
//! loads the QwenImage-Edit DiT (BlockOffloader), runs the true-CFG Euler loop with
//! the diffusers Edit-specific concatenation trick, and saves the final latents
//! for Stage 3 to decode.
//!
//! ## What's different from `qwenimage_gen.rs` (T2I)
//!
//! 1. **Concatenated input**: each forward sees `[noisy_target, image_latents]`
//!    along the seq dim. The reference image latents are constant across all
//!    denoise steps.
//! 2. **Multi-region RoPE**: the DiT computes RoPE positions for two regions
//!    (target + reference) via `forward_edit`. The reference's frame index is
//!    offset by 1 so the position embeddings of the two regions don't collide.
//! 3. **Noise prediction slicing**: after the forward, we keep only
//!    `[:, :target_seq_len, :]` of the prediction — the reference portion of
//!    the prediction is discarded.
//! 4. **Different default checkpoint**: points at `Qwen/Qwen-Image-Edit` shards
//!    instead of `Qwen-Image-2512`.
//! 5. **Variable target dimensions**: the Edit pipeline picks `(h, w)` to match
//!    the reference image's aspect ratio at ~1024² total area, rounded to
//!    multiples of 16. Stage 1 already chose this and saved `image_h`/`image_w`
//!    in the embeddings file.
//!
//! ## Pipeline split (OOM-safe: one model resident at a time)
//!   Stage 1 (Python): Qwen2.5-VL text encoder + AutoencoderKLQwenImage VAE encode
//!   Stage 2 (Rust):   QwenImage-Edit DiT only (BlockOffloader)
//!   Stage 3 (Python): AutoencoderKLQwenImage decode (reuse `qwenimage_decode.py`)
//!
//! ## True CFG with norm rescale (same as T2I)
//!   comb      = neg + scale * (cond - neg)
//!   cond_norm = ‖cond‖_2 along last dim
//!   comb_norm = ‖comb‖_2 along last dim
//!   out       = comb * (cond_norm / comb_norm)
//!
//! ⚠️ BUILD-COMPLETE but UNTESTED. DO NOT RUN until user validates.

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::qwenimage_dit::QwenImageDit;

// Qwen-Image-Edit-2511 (Dec 2025). Same 60-layer / head_dim=128 transformer
// architecture as the Aug 2025 base, repackaged into 5 bf16 shards. The only
// config diff is `zero_cond_t: true` at the pipeline level; the ComfyUI
// shard repack normalizes weight keys to the same 1933 identifiers this bin
// already loads. Override via QWEN_DIT_SHARDS=path1:path2:...
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

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_embeds.safetensors".to_string()
    });
    let out_latents = args.get(2).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_latents.safetensors".to_string()
    });

    // Knobs
    let num_steps: usize = env_usize("QWEN_STEPS", 50);
    let true_cfg_scale: f32 = env_f32("QWEN_CFG", 4.0);
    let seed: u64 = env_u64("QWEN_SEED", 42);

    let dit_shards_owned: Vec<String> = match std::env::var("QWEN_DIT_SHARDS") {
        Ok(s) => s.split(':').map(|p| p.to_string()).collect(),
        Err(_) => DEFAULT_DIT_SHARDS.iter().map(|s| s.to_string()).collect(),
    };
    let dit_shards: Vec<&str> = dit_shards_owned.iter().map(|s| s.as_str()).collect();

    println!("=== Qwen-Image-Edit — Stage 2 (Rust DiT denoise) ===");
    println!("Embeddings: {}", embeds_path);
    println!("Output lat: {}", out_latents);
    println!("Steps:      {}, true_cfg: {}, seed: {}", num_steps, true_cfg_scale, seed);
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
    println!("  image_latents: {:?}  (this is the packed reference)", image_latents.shape().dims());
    println!("  target dims:   {}x{} (chosen by Stage 1 from the reference aspect)", width, height);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage B: Load Qwen-Image-Edit DiT (BlockOffloader)
    // ------------------------------------------------------------------
    println!("--- Loading Qwen-Image-Edit DiT (BlockOffloader, 9 shards) ---");
    let t0 = Instant::now();
    let mut dit = QwenImageDit::load(&dit_shards, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage C: Build noise for the TARGET region (same shape as the reference)
    // ------------------------------------------------------------------
    println!("--- Denoise ({} steps, true_cfg={}) ---", num_steps, true_cfg_scale);

    // The Edit pipeline uses the target output at the same dims as the resized
    // reference. So target_h/w match image_h/w from Stage 1.
    let target_h = height;
    let target_w = width;
    let h_latent_full = target_h / VAE_SCALE_FACTOR;
    let w_latent_full = target_w / VAE_SCALE_FACTOR;
    let h_patched = h_latent_full / PATCH_SIZE;
    let w_patched = w_latent_full / PATCH_SIZE;
    let target_seq_len = h_patched * w_patched;
    println!(
        "  Target latent  [B,C,H,W] = [1, {}, {}, {}]   →   patched grid = ({}, {})",
        IN_CHANNELS, h_latent_full, w_latent_full, h_patched, w_patched
    );

    // The reference latents are already packed by Stage 1. Verify the seq len
    // matches what we expect (should be the same h_patched * w_patched since
    // Stage 1 used the same dims).
    let ref_dims = image_latents.shape().dims().to_vec();
    if ref_dims.len() != 3 || ref_dims[2] != PACKED_CHANNELS {
        return Err(anyhow::anyhow!(
            "Bad image_latents shape {:?} — expected [1, seq, {}]",
            ref_dims, PACKED_CHANNELS
        ));
    }
    let ref_seq_len = ref_dims[1];
    println!("  Reference seq:  {} (must equal target seq for this single-region edit case)", ref_seq_len);
    if ref_seq_len != target_seq_len {
        return Err(anyhow::anyhow!(
            "ref_seq_len {} != target_seq_len {} — Stage 1 used different dims than Stage 2 expects",
            ref_seq_len, target_seq_len
        ));
    }

    // Seeded Gaussian noise for the target latent.
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

    // CPU-pack the noise into [1, target_seq_len, 64].
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
    let mut latents = Tensor::from_f32_to_bf16(
        packed,
        Shape::from_dims(&[1, target_seq_len, PACKED_CHANNELS]),
        device.clone(),
    )?;
    println!("  Packed noise (target): {:?}", latents.shape().dims());

    // ── Sigma schedule (same exponential time shift as T2I) ──
    let base_shift: f32 = 0.5;
    let max_shift: f32 = 1.15;
    let base_seq_len: f32 = 256.0;
    let max_seq_len_shift: f32 = 4096.0;
    let shift_terminal: f32 = 0.02;

    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len);
    let bb = base_shift - m * base_seq_len;
    // Note: image_seq_len here is just `target_seq_len` (the noise tokens),
    // NOT `target_seq_len + ref_seq_len`. The diffusers pipeline sets
    // `image_seq_len = latents.shape[1]` BEFORE the latent_model_input concat.
    let mu = (target_seq_len as f32) * m + bb;
    let exp_mu = mu.exp();
    println!("  mu = {:.4}, exp(mu) = {:.4}", mu, exp_mu);

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
    println!();

    // ── Multi-region RoPE shapes for forward_edit ──
    // Two regions: target + reference. Both have the same patched grid since
    // Stage 1 picked the target dims to match the reference.
    let regions = vec![
        (1usize, h_patched, w_patched),  // region 0: target (noise)
        (1usize, h_patched, w_patched),  // region 1: reference
    ];

    // ── CFG Euler loop (same scoped-block pattern from the audit fix) ──
    let t_denoise = Instant::now();
    for step in 0..num_steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;

        // Scoped block: every per-step temporary drops at `}` before the
        // next iteration's allocations. Same OOM-safety pattern as
        // qwenimage_gen.rs after the audit-fix pass.
        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![sigma_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            // Reference region timestep. 2511's QwenImageEditPlusPipeline uses
            // `index_timestep_zero` — the reference latents are clean
            // conditioning, so they get t=0 while target uses sigma_curr.
            // Without this, every token gets the same AdaLN modulation and
            // the model "denoises" the reference back to itself, leaving
            // prompt influence dominated by the reference image.
            let t_ref_vec = Tensor::from_vec(
                vec![0.0f32],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            // Concat the noisy target with the (constant) reference latents
            // along the seq dim. This is the diffusers
            // `latent_model_input = torch.cat([latents, image_latents], dim=1)`
            // step.
            let concat_input = Tensor::cat(&[&latents, &image_latents], 1)?;

            let cond_pred_full = dit.forward_edit_with_ref_timestep(
                &concat_input, &cond, &t_vec, Some(&t_ref_vec), &regions,
            )?;
            let uncond_pred_full = dit.forward_edit_with_ref_timestep(
                &concat_input, &uncond, &t_vec, Some(&t_ref_vec), &regions,
            )?;

            // Defensive: forward_edit must return [B, target_seq_len + ref_seq_len, 64].
            // Catch any future shape regression before the narrow silently
            // returns garbage from the wrong dimension.
            let expected_seq = target_seq_len + ref_seq_len;
            let cd = cond_pred_full.shape().dims();
            let ud = uncond_pred_full.shape().dims();
            if cd.len() != 3 || cd[1] != expected_seq || cd[2] != PACKED_CHANNELS {
                return Err(anyhow::anyhow!(
                    "forward_edit (cond) returned shape {:?}, expected [_, {}, {}]",
                    cd, expected_seq, PACKED_CHANNELS
                ));
            }
            if ud.len() != 3 || ud[1] != expected_seq || ud[2] != PACKED_CHANNELS {
                return Err(anyhow::anyhow!(
                    "forward_edit (uncond) returned shape {:?}, expected [_, {}, {}]",
                    ud, expected_seq, PACKED_CHANNELS
                ));
            }

            // Slice back to the target portion: keep only the first
            // `target_seq_len` tokens, drop the reference prediction.
            //   `noise_pred = noise_pred[:, :latents.size(1)]`
            let cond_pred = cond_pred_full.narrow(1, 0, target_seq_len)?;
            let uncond_pred = uncond_pred_full.narrow(1, 0, target_seq_len)?;

            // True CFG: comb = neg + scale * (cond - neg)
            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(true_cfg_scale)?;
            let comb = uncond_pred.add(&scaled)?;

            // Norm rescale to preserve cond magnitude (same as T2I).
            let noise_pred = norm_rescale_cfg(&cond_pred, &comb).unwrap_or(comb);

            // Euler step on the target latents only.
            let step_tensor = noise_pred.mul_scalar(dt)?;
            latents.add(&step_tensor)?
        };
        latents = next_x;

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
    println!();

    drop(dit);
    drop(cond);
    drop(uncond);
    drop(image_latents);
    println!("  DiT + cached embeddings evicted");

    // ------------------------------------------------------------------
    // Stage D: save latents for Python Stage 3 VAE decode
    // ------------------------------------------------------------------
    println!("--- Saving target latents ---");
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
    println!();
    println!("Next: python scripts/qwenimage_decode.py {} <output.png>", out_latents);

    let _ = device;
    Ok(())
}

/// Same per-token L2 norm rescale as `qwenimage_gen.rs::norm_rescale_cfg`.
/// Returns the rescaled `comb` matching the cond's per-token norm; falls back
/// to the raw `comb` if any reduction op isn't supported.
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
