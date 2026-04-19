//! Qwen-Image-2512 — Stage 2 (DiT denoise) + Stage 3 (Rust VAE decode).
//!
//! Loads the cached cond+uncond embeddings saved by `scripts/qwenimage_encode.py`,
//! loads the Qwen DiT (BlockOffloader), runs the true-CFG Euler loop,
//! unpacks + VAE-decodes the result through the in-tree
//! `QwenImageVaeDecoder`, and writes a PNG directly. No Python decode
//! bridge.
//!
//! ## Pipeline split (OOM-safe: one model resident at a time)
//!   Stage 1 (Python, still): Qwen2.5-VL text encoder only
//!   Stage 2 (Rust):          Qwen-Image DiT only (BlockOffloader)
//!   Stage 3 (Rust):          AutoencoderKLQwenImage decoder — parity-
//!                            verified against diffusers in
//!                            `qwenimage_vae_decode_parity`.
//!
//! Stage 1 is the only remaining Python dependency. The Rust
//! `Qwen25VLEncoder` in `src/models/qwen25vl_encoder.rs` would cover it
//! but no bin has been wired up yet. Use QWEN_SAVE_LATENT=1 to skip
//! Stage 3 and dump the packed latent for debugging.
//!
//! ## Scheduler — FlowMatchEulerDiscreteScheduler with dynamic exponential shift
//!
//! Config from `scheduler/scheduler_config.json`:
//!   base_shift = 0.5, max_shift = 0.9  (note: different from FLUX 1.15)
//!   base_image_seq_len = 256, max_image_seq_len = 8192
//!   shift = 1.0, shift_terminal = 0.02
//!   time_shift_type = "exponential"
//!   use_dynamic_shifting = True
//!
//! mu is computed as in `pipeline_qwenimage.calculate_shift`:
//!   m = (max_shift - base_shift) / (max_seq - base_seq)
//!   b = base_shift - m * base_seq
//!   mu = image_seq_len * m + b
//!
//! Exponential time shift (scheduling_flow_match_euler_discrete.py:598):
//!   sigma_shifted = exp(mu) / (exp(mu) + (1/t - 1)^shift)   with shift=1.0
//!                 = exp(mu) / (exp(mu) + 1/t - 1)
//!
//! Stretch-to-terminal (scheduling_flow_match_euler_discrete.py:230):
//!   one_minus_z  = 1 - t
//!   scale        = one_minus_z[-1] / (1 - shift_terminal)
//!   stretched_t  = 1 - (one_minus_z / scale)
//!
//! ## True CFG with norm rescaling (pipeline_qwenimage.py:704-708)
//!   comb      = neg + scale * (cond - neg)
//!   cond_norm = ‖cond‖_2 along last dim
//!   comb_norm = ‖comb‖_2 along last dim
//!   out       = comb * (cond_norm / comb_norm)
//!
//! This Qwen-specific norm rescale keeps the magnitude of the combined
//! prediction equal to the cond prediction — unlike FLUX's raw CFG.
//!
//! ⚠️ BUILD-COMPLETE but UNTESTED. DO NOT RUN until user validates.

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::qwenimage_dit::QwenImageDit;
use inference_flame::sampling::exponential_multistep::{
    deis_3m_step, dpmpp_2m_step, lambda_from_sigma, res_2m_step, res_3m_step, MultistepHistory,
};

/// Sampler selection via `QWEN_SAMPLER` env var. Default `euler`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Sampler {
    Euler,
    DpmPp2m,
    Res2m,
    Res3m,
    Deis3m,
}

impl Sampler {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "euler" | "" => Some(Self::Euler),
            "dpmpp_2m" | "dpmpp2m" | "dpm++2m" => Some(Self::DpmPp2m),
            "res_2m" | "res2m" => Some(Self::Res2m),
            "res_3m" | "res3m" => Some(Self::Res3m),
            "deis_3m" | "deis3m" => Some(Self::Deis3m),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::DpmPp2m => "dpmpp_2m",
            Self::Res2m => "res_2m",
            Self::Res3m => "res_3m",
            Self::Deis3m => "deis_3m",
        }
    }
}

const DEFAULT_DIT_SHARDS: &[&str] = &[
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
];

// VAE scale factor: 2 ** len(temperal_downsample) = 2^3 = 8
const VAE_SCALE_FACTOR: usize = 8;
const PATCH_SIZE: usize = 2;
const IN_CHANNELS: usize = 16;      // raw VAE z_dim
const PACKED_CHANNELS: usize = 64;  // 16 * patch^2 = 64

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    // Args: [embeds_path] [out_png_or_latents]
    //   default output is now a PNG via the Rust VAE decoder. Set
    //   QWEN_SAVE_LATENT=1 to fall back to the old "save packed latent,
    //   decode via Python" flow — the second positional arg then
    //   interprets as the latent path instead of the PNG path.
    let args: Vec<String> = std::env::args().collect();
    let embeds_path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/qwenimage_embeds.safetensors".to_string()
    });
    let save_latent_only = std::env::var("QWEN_SAVE_LATENT").ok().as_deref() == Some("1");
    let out_path = args.get(2).cloned().unwrap_or_else(|| {
        if save_latent_only {
            "/home/alex/EriDiffusion/inference-flame/output/qwenimage_latents.safetensors".to_string()
        } else {
            "/home/alex/EriDiffusion/inference-flame/output/qwenimage_rust.png".to_string()
        }
    });

    // Knobs
    let height: usize = env_usize("QWEN_HEIGHT", 1024);
    let width: usize = env_usize("QWEN_WIDTH", 1024);
    let num_steps: usize = env_usize("QWEN_STEPS", 50);
    let true_cfg_scale: f32 = env_f32("QWEN_CFG", 4.0);
    let seed: u64 = env_u64("QWEN_SEED", 42);
    let sampler = match std::env::var("QWEN_SAMPLER") {
        Ok(s) => match Sampler::parse(&s) {
            Some(v) => v,
            None => {
                eprintln!(
                    "QWEN_SAMPLER='{}' not recognized; falling back to euler. \
                     Supported: euler | dpmpp_2m | res_2m | res_3m | deis_3m",
                    s
                );
                Sampler::Euler
            }
        },
        Err(_) => Sampler::Euler,
    };

    // Shards (colon-separated override)
    let dit_shards_owned: Vec<String> = match std::env::var("QWEN_DIT_SHARDS") {
        Ok(s) => s.split(':').map(|p| p.to_string()).collect(),
        Err(_) => DEFAULT_DIT_SHARDS.iter().map(|s| s.to_string()).collect(),
    };
    let dit_shards: Vec<&str> = dit_shards_owned.iter().map(|s| s.as_str()).collect();

    println!("=== Qwen-Image-2512 — Stage 2 (Rust DiT denoise) ===");
    println!("Embeddings: {}", embeds_path);
    println!(
        "Output:     {} ({})",
        out_path,
        if save_latent_only { "packed latent" } else { "PNG via Rust VAE" }
    );
    println!(
        "Size:       {}x{}, steps={}, true_cfg={}, sampler={}",
        width, height, num_steps, true_cfg_scale, sampler.label()
    );
    println!("Seed:       {}", seed);
    println!();

    // ------------------------------------------------------------------
    // Stage A: load cached cond + uncond embeddings
    // ------------------------------------------------------------------
    println!("--- Loading cached embeddings ---");
    let t0 = Instant::now();
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&embeds_path),
        &device,
    )?;
    let cond = ensure_bf16(tensors.get("cond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'cond' in embeddings file"))?
        .clone())?;
    let uncond = ensure_bf16(tensors.get("uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'uncond' in embeddings file"))?
        .clone())?;
    drop(tensors);
    println!("  cond:   {:?}", cond.shape().dims());
    println!("  uncond: {:?}", uncond.shape().dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage B: load Qwen-Image DiT (BlockOffloader)
    // ------------------------------------------------------------------
    println!("--- Loading Qwen-Image DiT (BlockOffloader, 9 shards) ---");
    let t0 = Instant::now();
    let mut dit = QwenImageDit::load(&dit_shards, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage C: build noise, pack, run CFG denoise
    // ------------------------------------------------------------------
    println!("--- Denoise ({} steps, true_cfg={}) ---", num_steps, true_cfg_scale);

    // QwenImage latent geometry:
    //   raw latent:       [B, 1, 16, H/8, W/8]   (single-frame 3D)
    //   after packing:    [B, (H/8/2)*(W/8/2), 16*2*2 = 64]
    //
    // For 1024x1024: H/8 = 128, /2 = 64 → packed seq len = 64*64 = 4096, dim 64.
    let h_latent_full = height / VAE_SCALE_FACTOR;  // 128
    let w_latent_full = width / VAE_SCALE_FACTOR;
    let h_patched = h_latent_full / PATCH_SIZE;     // 64
    let w_patched = w_latent_full / PATCH_SIZE;
    let seq_len = h_patched * w_patched;
    println!(
        "  Latent raw [B,F,C,H,W] = [1, 1, {}, {}, {}]",
        IN_CHANNELS, h_latent_full, w_latent_full
    );
    println!("  Packed [B, seq, C*p*p] = [1, {}, {}]", seq_len, PACKED_CHANNELS);

    // Seeded Gaussian noise, Box-Muller.
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

    // Pack latents: diffusers-style `_pack_latents` (pipeline_qwenimage.py:317-323):
    //   view as [B, C, H/2, 2, W/2, 2], permute to [B, H/2, W/2, C, 2, 2],
    //   reshape to [B, (H/2)(W/2), C*4].
    //
    // We do it on CPU as vec math since we have the raw noise vector already.
    let mut packed = vec![0.0f32; 1 * seq_len * PACKED_CHANNELS];
    for c in 0..IN_CHANNELS {
        for hp in 0..h_patched {
            for wp in 0..w_patched {
                for dh in 0..PATCH_SIZE {
                    for dw in 0..PATCH_SIZE {
                        let src_h = hp * PATCH_SIZE + dh;
                        let src_w = wp * PATCH_SIZE + dw;
                        let src_idx = c * h_latent_full * w_latent_full + src_h * w_latent_full + src_w;

                        // Destination in [seq=hp*W_p+wp, c*4 + dh*2 + dw]
                        let dst_seq = hp * w_patched + wp;
                        let dst_chan = c * PATCH_SIZE * PATCH_SIZE + dh * PATCH_SIZE + dw;
                        packed[dst_seq * PACKED_CHANNELS + dst_chan] = noise_data[src_idx];
                    }
                }
            }
        }
    }
    let mut x = Tensor::from_f32_to_bf16(
        packed,
        Shape::from_dims(&[1, seq_len, PACKED_CHANNELS]),
        device.clone(),
    )?;
    println!("  Packed noise: {:?}", x.shape().dims());

    // ── Sigma schedule: dynamic exponential shift ──
    // Reference: pipeline_qwenimage.py:634-649 + scheduling_flow_match_euler_discrete.py
    //
    //   sigmas = linspace(1.0, 1/num_steps, num_steps)
    //   mu = calculate_shift(seq_len, base_seq=256, max_seq=4096, base=0.5, max=1.15)
    //   sigmas = exp(mu) / (exp(mu) + (1/sigmas - 1))            # shift_exp, shift=1
    //   sigmas = stretch_to_terminal(sigmas, shift_terminal=0.02)
    //   sigmas = cat(sigmas, [0.0])                              # append terminal
    //
    // Note: the pipeline uses max_seq=4096 (hardcoded at call site) even though
    // scheduler_config has max_image_seq_len=8192. We follow the pipeline.
    // From scheduler_config.json — pipeline uses .get() so these config values win
    // over the function defaults (base_shift=0.5, max_shift=1.15).
    let base_shift: f32 = 0.5;
    let max_shift: f32 = 0.9;
    let base_seq_len: f32 = 256.0;
    let max_seq_len_shift: f32 = 8192.0;
    let shift_terminal: f32 = 0.02;

    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len);
    let bb = base_shift - m * base_seq_len;
    let mu = (seq_len as f32) * m + bb;
    let exp_mu = mu.exp();
    println!("  mu = {:.4}, exp(mu) = {:.4}", mu, exp_mu);

    // 1. Linear sigmas in descending order
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / (num_steps - 1) as f32; // [0, 1]
            1.0 - t * (1.0 - 1.0 / num_steps as f32)
        })
        .collect();
    // 2. Exponential time shift (shift=1.0 so (1/t - 1)^1 = 1/t - 1)
    for s in sigmas.iter_mut() {
        let denom = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / denom;
    }
    // 3. Stretch to terminal: one_minus_z = 1 - t; scale = one_minus_z[-1] / (1 - shift_terminal)
    //    stretched = 1 - one_minus_z / scale
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    if one_minus_last.abs() > 1e-12 {
        let scale = one_minus_last / (1.0 - shift_terminal);
        for s in sigmas.iter_mut() {
            let o = 1.0 - *s;
            *s = 1.0 - o / scale;
        }
    }
    // 4. Append terminal sigma = 0
    sigmas.push(0.0);
    println!("  sigmas[0]={:.4}  sigmas[1]={:.4}  sigmas[-2]={:.4}  sigmas[-1]={:.4}",
        sigmas[0], sigmas[1], sigmas[num_steps - 1], sigmas[num_steps]);
    println!();

    // ── CFG sampler loop ──
    // Per-step: two forward passes (cond + uncond), norm-rescaled CFG combine,
    // then sampler step. Euler is `x + (σ_next - σ)·v`; the RES4LYF multistep
    // samplers (dpmpp_2m / res_2m / res_3m / deis_3m) take the data-prediction
    //   denoised = x - σ·v
    // and a short ring buffer of past `denoised` samples.
    let frame = 1;
    let t_denoise = Instant::now();
    // Capacity 3 is enough for every multistep we ship (deis_3m peeks back 2).
    let mut history = MultistepHistory::new(3);
    for step in 0..num_steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;

        // Scoped block: every per-step temporary drops at the closing `}`
        // BEFORE the next iteration allocates. Prevents VRAM accumulation.
        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![sigma_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            let cond_pred = dit.forward(&x, &cond, &t_vec, (frame, h_patched, w_patched))?;
            let uncond_pred = dit.forward(&x, &uncond, &t_vec, (frame, h_patched, w_patched))?;

            // True CFG: comb = neg + scale * (cond - neg)
            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(true_cfg_scale)?;
            let comb = uncond_pred.add(&scaled)?;
            let noise_pred = norm_rescale_cfg(&cond_pred, &comb).unwrap_or(comb);

            match sampler {
                Sampler::Euler => {
                    // x_next = x + dt·v
                    let step_tensor = noise_pred.mul_scalar(dt)?;
                    x.add(&step_tensor)?
                }
                _ => {
                    // denoised = x - σ·v  (data prediction)
                    let v_sig = noise_pred.mul_scalar(sigma_curr)?;
                    let denoised = x.sub(&v_sig)?;
                    let next = match sampler {
                        Sampler::Euler => unreachable!(),
                        Sampler::DpmPp2m => {
                            dpmpp_2m_step(&x, &denoised, sigma_curr, sigma_next, &history)?
                        }
                        Sampler::Res2m => {
                            res_2m_step(&x, &denoised, sigma_curr, sigma_next, &history)?
                        }
                        Sampler::Res3m => {
                            res_3m_step(&x, &denoised, sigma_curr, sigma_next, &history)?
                        }
                        Sampler::Deis3m => {
                            deis_3m_step(&x, &denoised, sigma_curr, sigma_next, &history)?
                        }
                    };
                    history.push(denoised, lambda_from_sigma(sigma_curr));
                    next
                }
            }
        };
        x = next_x;

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
    println!("  DiT + cached embeddings evicted");

    // ------------------------------------------------------------------
    // Stage D: VAE decode + save PNG (pure Rust default).
    //          With QWEN_SAVE_LATENT=1 we instead dump the packed latent
    //          for an external decoder — kept for regression/debug only.
    // ------------------------------------------------------------------
    if save_latent_only {
        println!("--- Saving packed latent (debug path) ---");
        if let Some(parent) = std::path::Path::new(&out_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut output = HashMap::new();
        output.insert("packed_latent".to_string(), x);
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
        flame_core::serialization::save_file(&output, &out_path)?;
        let dt_total = t_total.elapsed().as_secs_f32();
        println!();
        println!("============================================================");
        println!("LATENTS SAVED: {out_path}");
        println!("Total time:    {dt_total:.1}s");
        println!("============================================================");
        let _ = device;
        return Ok(());
    }

    println!("--- Stage D: VAE decode (Rust) ---");
    let vae_path = std::env::var("QWEN_VAE_PATH").unwrap_or_else(|_|
        "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors".to_string()
    );
    // Unpack packed latent [1, (H/16)*(W/16), 64] -> [1, 16, 1, H/8, W/8].
    // Matches `scripts/qwenimage_decode.py::unpack_latents`.
    let h_lat = 2 * (height / (VAE_SCALE_FACTOR * 2));
    let w_lat = 2 * (width / (VAE_SCALE_FACTOR * 2));
    let unpacked = x
        .reshape(&[1, h_lat / 2, w_lat / 2, PACKED_CHANNELS / 4, 2, 2])?
        .permute(&[0, 3, 1, 4, 2, 5])?
        .reshape(&[1, PACKED_CHANNELS / (2 * 2), 1, h_lat, w_lat])?;

    let t_vae = Instant::now();
    let decoder = inference_flame::vae::QwenImageVaeDecoder::from_safetensors(&vae_path, &device)?;
    let rgb = decoder.decode(&unpacked)?; // [1, 3, 1, H, W] in [-1, 1]
    println!(
        "  VAE decoded {:?} in {:.1}s",
        rgb.shape().dims(),
        t_vae.elapsed().as_secs_f32()
    );

    // [1, 3, 1, H, W] -> [1, 3, H, W], to [0, 255] u8, save PNG.
    let rgb = rgb.narrow(2, 0, 1)?.squeeze(Some(2))?;
    let rgb_f32 = rgb.to_dtype(DType::F32)?.to_vec_f32()?;
    let dims = rgb.shape().dims().to_vec();
    let (h_img, w_img) = (dims[2], dims[3]);
    let mut buf = vec![0u8; h_img * w_img * 3];
    for y in 0..h_img {
        for xp in 0..w_img {
            for ch in 0..3 {
                let v = rgb_f32[ch * h_img * w_img + y * w_img + xp];
                let v = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).clamp(0.0, 255.0);
                buf[(y * w_img + xp) * 3 + ch] = v as u8;
            }
        }
    }
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::save_buffer(&out_path, &buf, w_img as u32, h_img as u32, image::ColorType::Rgb8)?;
    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {out_path}");
    println!("Total time:  {dt_total:.1}s");
    println!("============================================================");

    let _ = device;
    Ok(())
}

/// Rescale the combined CFG prediction to match the cond prediction's
/// per-token L2 norm (the Qwen-specific CFG post-process — see
/// `pipeline_qwenimage.py:706-708`).
///
/// If any intermediate op isn't supported, returns None and the caller
/// falls back to the raw `comb`.
fn norm_rescale_cfg(cond: &Tensor, comb: &Tensor) -> anyhow::Result<Tensor> {
    // Both tensors are [B, seq, dim]. Compute sqrt(sum(x^2, dim=-1, keepdim=True)).
    // We use mul_then_sum via flame-core primitives.
    let cond_sq = cond.mul(cond)?;
    let comb_sq = comb.mul(comb)?;
    // Sum over last dim, keepdim. If sum_dim_keepdim isn't available,
    // fall through with error.
    let cond_sum = cond_sq.sum_dim_keepdim(2)?;
    let comb_sum = comb_sq.sum_dim_keepdim(2)?;
    let cond_norm = cond_sum.sqrt()?;
    let comb_norm = comb_sum.sqrt()?;
    // ratio = cond_norm / comb_norm — broadcast over last dim
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
