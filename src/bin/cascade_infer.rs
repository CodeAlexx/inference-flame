//! Stable Cascade end-to-end inference — pure Rust, flame-core.
//!
//! Pipeline:
//!   1. CLIP-ViT-bigG-14 → (hidden [1,77,1280], pooled [1,1280])
//!   2. Stage C (prior UNet) → denoise [1,16,Hc,Wc] for `--steps-c` steps (with CFG)
//!   3. Stage B (decoder UNet) → denoise [1,4,Hb,Wb] conditioned on Stage C latent
//!   4. Stage A (Paella VQGAN decoder) → RGB [1,3,H,W]
//!   5. PNG to `--out`
//!
//! Memory: load one stage at a time. Stage C (~7 GB BF16) is the largest; a
//! single 24 GB card fits it with headroom.
//!
//! CFG (classifier-free guidance):
//!   v = v_uncond + cfg * (v_cond - v_uncond)
//! Defaults: Stage C cfg=4.0, Stage B cfg=1.1 (near-deterministic).

use std::path::PathBuf;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::paella_vq::PaellaVQDecoder;
use inference_flame::models::wuerstchen_unet::{WuerstchenUNet, WuerstchenUNetConfig};
use inference_flame::sampling::ddpm_wuerstchen::{randn_bf16, DDPMWuerstchenScheduler};

// ---------------------------------------------------------------------------
// Hardcoded checkpoint paths
// ---------------------------------------------------------------------------

const CKPT_ROOT: &str =
    "/home/alex/.cache/huggingface/hub/models--stabilityai--stable-cascade/snapshots/a89f66d459ae653e3b4d4f992a7c3789d0dc4d16";

fn clip_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("text_encoder/model.bf16.safetensors")
}
fn clip_tokenizer_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("tokenizer/tokenizer.json")
}
fn stage_c_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_c_bf16.safetensors")
}
fn stage_b_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_b_bf16.safetensors")
}
fn stage_a_path() -> PathBuf {
    PathBuf::from(CKPT_ROOT).join("stage_a.safetensors")
}

const CLIP_SEQ_LEN: usize = 77;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    prompt_path: String,
    out_path: String,
    steps_c: usize,
    steps_b: usize,
    cfg_c: f32,
    cfg_b: f32,
    width: usize,
    height: usize,
    seed: u64,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = std::env::args().skip(1).collect::<Vec<_>>();
        let mut prompt_path = "output/cascade_gen/prompt.txt".to_string();
        let mut out_path = "output/cascade_gen/cascade_out.png".to_string();
        let mut steps_c = 20usize;
        let mut steps_b = 10usize;
        let mut cfg_c = 4.0f32;
        let mut cfg_b = 1.1f32;
        let mut width = 1024usize;
        let mut height = 1024usize;
        let mut seed = 42u64;

        let mut i = 0;
        while i < args.len() {
            let k = args[i].clone();
            let v = args.get(i + 1).cloned();
            let take = |x: Option<String>, key: &str| {
                x.ok_or_else(|| anyhow::anyhow!("missing value for {}", key))
            };
            match k.as_str() {
                "--prompt" => {
                    prompt_path = take(v, &k)?;
                    i += 2;
                }
                "--out" => {
                    out_path = take(v, &k)?;
                    i += 2;
                }
                "--steps-c" => {
                    steps_c = take(v, &k)?.parse()?;
                    i += 2;
                }
                "--steps-b" => {
                    steps_b = take(v, &k)?.parse()?;
                    i += 2;
                }
                "--cfg-c" => {
                    cfg_c = take(v, &k)?.parse()?;
                    i += 2;
                }
                "--cfg-b" => {
                    cfg_b = take(v, &k)?.parse()?;
                    i += 2;
                }
                "--width" => {
                    width = take(v, &k)?.parse()?;
                    i += 2;
                }
                "--height" => {
                    height = take(v, &k)?.parse()?;
                    i += 2;
                }
                "--seed" => {
                    seed = take(v, &k)?.parse()?;
                    i += 2;
                }
                _ => {
                    eprintln!("Unknown arg: {}", k);
                    i += 1;
                }
            }
        }
        let _ = args;
        Ok(Self {
            prompt_path,
            out_path,
            steps_c,
            steps_b,
            cfg_c,
            cfg_b,
            width,
            height,
            seed,
        })
    }
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str, tokenizer_path: &std::path::Path) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(tokenizer_path) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("clip tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(CLIP_SEQ_LEN);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407); // pad with EOS
            }
            ids
        }
        Err(e) => {
            eprintln!("[cascade] tokenizer failed: {e}; using BOS/EOS fallback");
            let mut ids = vec![49406i32];
            ids.push(49407);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// CFG helper
// ---------------------------------------------------------------------------

/// Apply CFG: `v_cfg = v_uncond + cfg * (v_cond - v_uncond)`.
fn cfg_combine(v_cond: &Tensor, v_uncond: &Tensor, cfg: f32) -> anyhow::Result<Tensor> {
    // v_uncond + cfg * (v_cond - v_uncond)
    let diff = v_cond.sub(v_uncond)?;
    let scaled = diff.mul_scalar(cfg)?;
    let out = v_uncond.add(&scaled)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::try_init().ok();
    let t_total = Instant::now();
    let args = Args::parse()?;

    let device = global_cuda_device();

    let prompt = std::fs::read_to_string(&args.prompt_path)
        .map_err(|e| anyhow::anyhow!("failed to read prompt file {}: {}", args.prompt_path, e))?;
    let prompt = prompt.trim().to_string();

    println!("=== Stable Cascade — pure Rust inference ===");
    println!("Prompt: {}", &prompt.chars().take(120).collect::<String>());
    println!(
        "Size:   {}x{}, steps_c={}, steps_b={}, cfg_c={}, cfg_b={}, seed={}",
        args.width, args.height, args.steps_c, args.steps_b, args.cfg_c, args.cfg_b, args.seed
    );
    println!();

    // ---------------------------------------------------------------
    // Spatial shapes.
    //   Stage C compresses by 32x from pixel (plus a further /2 from patch-in/out).
    //   For 1024x1024: Stage C latent is [1, 16, 24, 24] (rounded).
    //   Actually Wuerstchen v3: Stage C latent spatial = ceil(width / 32) / ?
    //   Let's use the config from the pipeline: latent_dim_scale = 10.67, i.e.
    //   height / 42.67, but in practice the diffusers pipeline rounds via:
    //     latent_height = ceil(height / compression)
    //   where compression = 32 * 4 * 2 = ... messy.
    //
    // Empirically, Stable Cascade at 1024x1024 uses Stage C [1,16,24,24] and
    // Stage B [1,4,256,256]. We reproduce those numbers.
    // ---------------------------------------------------------------
    let stage_c_h = ((args.height + 31) / 32 * 3) / 4;  // ≈ 24 for 1024
    let stage_c_w = ((args.width + 31) / 32 * 3) / 4;
    // Fallback to exact 24 for the 1024 case until we match the formula.
    let stage_c_h = if args.height == 1024 { 24 } else { stage_c_h.max(1) };
    let stage_c_w = if args.width == 1024 { 24 } else { stage_c_w.max(1) };
    let stage_b_h = args.height / 4; // Stage A/B latent = pixel / 4
    let stage_b_w = args.width / 4;

    println!(
        "Stage C latent: [1, 16, {}, {}]; Stage B latent: [1, 4, {}, {}]",
        stage_c_h, stage_c_w, stage_b_h, stage_b_w
    );
    println!();

    // ---------------------------------------------------------------
    // Stage 1: CLIP-G encode prompt + uncond.
    // ---------------------------------------------------------------
    println!("--- Stage 1: CLIP-G encode ---");
    let t0 = Instant::now();
    let clip_weights_raw = flame_core::serialization::load_file(clip_path(), &device)?;
    let clip_weights: std::collections::HashMap<String, Tensor> = clip_weights_raw
        .into_iter()
        .map(|(k, v)| {
            let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
            Ok::<_, flame_core::Error>((k, t))
        })
        .collect::<Result<_, _>>()?;
    println!("  Loaded {} CLIP-G weights in {:.1}s", clip_weights.len(), t0.elapsed().as_secs_f32());

    let clip_cfg = ClipConfig::clip_g();
    let clip = ClipEncoder::new(clip_weights, clip_cfg, device.clone());

    let pos_tokens = tokenize_clip(&prompt, &clip_tokenizer_path());
    let (pos_hidden, pos_pooled) = clip.encode_cascade(&pos_tokens)?;
    let neg_tokens = tokenize_clip("", &clip_tokenizer_path());
    let (neg_hidden, neg_pooled) = clip.encode_cascade(&neg_tokens)?;
    println!(
        "  pos_hidden: {:?}  pos_pooled: {:?}",
        pos_hidden.shape().dims(),
        pos_pooled.shape().dims()
    );
    drop(clip);
    println!("  CLIP done in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ---------------------------------------------------------------
    // Stage 2: Stage C denoise (prior).
    // ---------------------------------------------------------------
    println!("--- Stage 2: Stage C denoise ({} steps) ---", args.steps_c);
    let t0 = Instant::now();

    // Seed the RNG for reproducibility.
    // flame-core's Tensor::randn uses the global RNG; we set it via std (no reseed API exposed here).
    // For now, the seed only affects logical dispatch ordering. Production would want a deterministic RNG.
    let _ = args.seed;

    let stage_c_latent = {
        let mut unet_c = WuerstchenUNet::load(
            stage_c_path().to_str().unwrap(),
            WuerstchenUNetConfig::stage_c(),
            &device,
        )?;
        println!("  Stage C loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let scheduler = DDPMWuerstchenScheduler::new(args.steps_c);
        let mut x = randn_bf16(&[1, 16, stage_c_h, stage_c_w], &device)?;

        let timesteps = scheduler.timesteps.clone();
        for step in 0..args.steps_c {
            let r = timesteps[step];
            let r_next = timesteps[step + 1];

            let v_cond = unet_c.forward(
                &x,
                r,
                Some(&pos_pooled),
                Some(&pos_hidden),
                None,
            )?;
            if step == 0 {
                let vf = v_cond.to_dtype(DType::F32)?.to_vec_f32()?;
                let xf = x.to_dtype(DType::F32)?.to_vec_f32()?;
                let vmax = vf.iter().cloned().fold(f32::MIN, f32::max);
                let vmin = vf.iter().cloned().fold(f32::MAX, f32::min);
                let xmax = xf.iter().cloned().fold(f32::MIN, f32::max);
                let xmin = xf.iter().cloned().fold(f32::MAX, f32::min);
                // Correlation between v and x (if eps-prediction at t=1, corr ~1).
                let (mx, mv) = (xf.iter().sum::<f32>() / xf.len() as f32, vf.iter().sum::<f32>() / vf.len() as f32);
                let mut num = 0f32;
                let mut dx = 0f32;
                let mut dv = 0f32;
                for (xi, vi) in xf.iter().zip(vf.iter()) {
                    num += (xi - mx) * (vi - mv);
                    dx += (xi - mx).powi(2);
                    dv += (vi - mv).powi(2);
                }
                let corr = num / (dx.sqrt() * dv.sqrt() + 1e-6);
                println!("  [dbg step 0] v: min={:.2} max={:.2}  x: min={:.2} max={:.2}  corr(x,v)={:.3}", vmin, vmax, xmin, xmax, corr);
            }
            let v = if args.cfg_c > 1.0 {
                let v_uncond = unet_c.forward(
                    &x,
                    r,
                    Some(&neg_pooled),
                    Some(&neg_hidden),
                    None,
                )?;
                cfg_combine(&v_cond, &v_uncond, args.cfg_c)?
            } else {
                v_cond
            };

            // Use DDIM-style eps-prediction step (deterministic, no added noise).
            // The stochastic DDPM step empirically diverges in our implementation (root
            // cause unclear; possibly related to the small residual correlation gap
            // between the UNet eps prediction and x_t at t=1.0 being amplified by
            // 1/sqrt(alpha) ≈ 50 per step). DDIM avoids the noise-injection blow-up.
            x = scheduler.step_eps_ddim(&v, r, r_next, &x)?;
            if step == 0 || step + 1 == args.steps_c || (step + 1) % 5 == 0 {
                println!("    step {}/{}  t={:.4}", step + 1, args.steps_c, r);
            }
        }

        drop(unet_c);
        x
    };
    println!(
        "  Stage C done in {:.1}s; latent: {:?}",
        t0.elapsed().as_secs_f32(),
        stage_c_latent.shape().dims()
    );
    // Debug: print latent statistics.
    {
        let f32 = stage_c_latent.to_dtype(DType::F32)?.to_vec_f32()?;
        let n = f32.len();
        let mean: f32 = f32.iter().sum::<f32>() / n as f32;
        let max = f32.iter().cloned().fold(f32::MIN, f32::max);
        let min = f32.iter().cloned().fold(f32::MAX, f32::min);
        let var = f32.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        println!(
            "  Stage C latent stats: mean={:.4}  std={:.4}  min={:.4}  max={:.4}",
            mean,
            var.sqrt(),
            min,
            max
        );
    }
    println!();

    // ---------------------------------------------------------------
    // Stage 3: Stage B denoise (decoder).
    // ---------------------------------------------------------------
    println!("--- Stage 3: Stage B denoise ({} steps) ---", args.steps_b);
    let t0 = Instant::now();
    let stage_b_latent = {
        let mut unet_b = WuerstchenUNet::load(
            stage_b_path().to_str().unwrap(),
            WuerstchenUNetConfig::stage_b(),
            &device,
        )?;
        println!("  Stage B loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let scheduler = DDPMWuerstchenScheduler::new(args.steps_b);
        let mut x = randn_bf16(&[1, 4, stage_b_h, stage_b_w], &device)?;

        let timesteps = scheduler.timesteps.clone();
        for step in 0..args.steps_b {
            let r = timesteps[step];
            let r_next = timesteps[step + 1];

            let v_cond = unet_b.forward(
                &x,
                r,
                Some(&pos_pooled),
                Some(&pos_hidden),
                Some(&stage_c_latent),
            )?;
            let v = if args.cfg_b > 1.0 {
                let v_uncond = unet_b.forward(
                    &x,
                    r,
                    Some(&neg_pooled),
                    Some(&neg_hidden),
                    Some(&stage_c_latent),
                )?;
                cfg_combine(&v_cond, &v_uncond, args.cfg_b)?
            } else {
                v_cond
            };

            x = scheduler.step_eps_ddim(&v, r, r_next, &x)?;
            if step == 0 || step + 1 == args.steps_b || (step + 1) % 5 == 0 {
                println!("    step {}/{}  t={:.4}", step + 1, args.steps_b, r);
            }
        }
        drop(unet_b);
        x
    };
    println!(
        "  Stage B done in {:.1}s; latent: {:?}",
        t0.elapsed().as_secs_f32(),
        stage_b_latent.shape().dims()
    );
    {
        let f32 = stage_b_latent.to_dtype(DType::F32)?.to_vec_f32()?;
        let n = f32.len();
        let mean: f32 = f32.iter().sum::<f32>() / n as f32;
        let max = f32.iter().cloned().fold(f32::MIN, f32::max);
        let min = f32.iter().cloned().fold(f32::MAX, f32::min);
        let var = f32.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        println!(
            "  Stage B latent stats: mean={:.4}  std={:.4}  min={:.4}  max={:.4}",
            mean,
            var.sqrt(),
            min,
            max
        );
    }
    println!();

    // ---------------------------------------------------------------
    // Stage 4: Stage A VQGAN decode.
    // ---------------------------------------------------------------
    println!("--- Stage 4: Stage A (Paella VQGAN) decode ---");
    let t0 = Instant::now();
    let rgb = {
        let vae = PaellaVQDecoder::load(stage_a_path().to_str().unwrap(), &device)?;
        println!("  Stage A loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let img = vae.decode(&stage_b_latent)?;
        drop(vae);
        img
    };
    println!("  Decoded RGB: {:?} in {:.1}s", rgb.shape().dims(), t0.elapsed().as_secs_f32());
    println!();

    // ---------------------------------------------------------------
    // Stage 5: Save PNG.
    // ---------------------------------------------------------------
    println!("--- Stage 5: Save PNG ---");
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec_f32()?;
    let dims = rgb_f32.shape().dims().to_vec();
    let (out_c, out_h, out_w) = (dims[1], dims[2], dims[3]);
    if out_c != 3 {
        anyhow::bail!("expected 3-channel RGB, got {}", out_c);
    }
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                // Paella VQ output is in ~[0, 1] (clamped). Rescale to [0, 255].
                let v = data[idx].clamp(0.0, 1.0);
                let u = (v * 255.0).round().clamp(0.0, 255.0) as u8;
                pixels[(y * out_w + x) * 3 + c] = u;
            }
        }
    }

    if let Some(parent) = std::path::Path::new(&args.out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to build RgbImage"))?;
    img.save(&args.out_path)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("========================================================");
    println!("IMAGE SAVED: {}", args.out_path);
    println!("Total time:  {:.1}s", dt_total);
    println!("========================================================");

    let _ = device; // keep alive
    Ok(())
}
