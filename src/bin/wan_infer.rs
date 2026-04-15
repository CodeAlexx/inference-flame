//! `wan_infer` — text-to-video inference for Wan2.2 TI2V-5B.
//!
//! Pipeline (sequential, 24GB-safe):
//!   1. Load UMT5-XXL encoder → encode prompt + negative → drop encoder.
//!   2. Load DiT shards via BlockOffloader → run 50-step Euler flow-matching
//!      (shift=5.0, guidance=5.0, with classifier-free guidance) → drop DiT.
//!   3. Load Wan2.2 VAE decoder → decode latent → write `.mp4` via
//!      `inference_flame::mux::write_mp4`.
//!
//! ## Checkpoint conversion (one-off)
//! The official TI2V-5B release ships `models_t5_umt5-xxl-enc-bf16.pth` and
//! `Wan2.2_VAE.pth` as PyTorch pickles. Convert once with:
//!
//! ```python
//! import torch
//! from safetensors.torch import save_file
//! save_file(torch.load("Wan2.2_VAE.pth", map_location="cpu"), "wan22_vae.safetensors")
//! save_file(torch.load("models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu"),
//!           "umt5_xxl.safetensors")
//! ```
//!
//! ## CLI
//! ```text
//! wan_infer \
//!     --ckpt-dir   /path/to/Wan2.2-TI2V-5B         \
//!     --vae        /path/to/wan22_vae.safetensors   \
//!     --t5         /path/to/umt5_xxl.safetensors    \
//!     --tokenizer  /path/to/tokenizer.json          \
//!     --prompt     "Two cats boxing on a stage"     \
//!     --size       1280x704                         \
//!     --frames     121                              \
//!     --steps      50                               \
//!     --guidance   5.0                              \
//!     --shift      5.0                              \
//!     --seed       42                               \
//!     --output     output/wan22_out.mp4             \
//!     [--t5-cpu]
//! ```
//!
//! Constraints:
//!   - `frames` must satisfy `(N - 1) % 4 == 0`.
//!   - `width % 16 == 0` and `height % 16 == 0` (Wan2.2 VAE stride=16).
//!   - After the DiT's patch size (1,2,2) the post-patch grid must be whole:
//!     `(H/16) % 2 == 0` and `(W/16) % 2 == 0`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::wan::{
    sampler, EulerSigmas, Umt5Encoder, Wan22VaeDecoder, WanConfig, WanTransformer,
};

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    ckpt_dir: PathBuf,
    vae: PathBuf,
    t5: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    negative: String,
    width: usize,
    height: usize,
    frames: usize,
    steps: usize,
    guidance: f32,
    shift: f32,
    seed: u64,
    output: PathBuf,
    t5_cpu: bool,
    fps: u32,
}

fn usage() -> ! {
    eprintln!(
        "\
usage: wan_infer --ckpt-dir DIR --vae PATH --t5 PATH --tokenizer PATH \\
                 --prompt STRING --size WxH --frames N --steps N \\
                 --guidance F --shift F --seed U64 --output PATH [--t5-cpu]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut ckpt_dir: Option<PathBuf> = None;
    let mut vae: Option<PathBuf> = None;
    let mut t5: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut negative: String = String::new();
    let mut size: Option<(usize, usize)> = None;
    let mut frames: usize = 121;
    let mut steps: usize = 50;
    let mut guidance: f32 = 5.0;
    let mut shift: f32 = 5.0;
    let mut seed: u64 = 42;
    let mut output: Option<PathBuf> = None;
    let mut t5_cpu = false;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let a = raw[i].as_str();
        let next = || -> String {
            raw.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {a}");
                std::process::exit(2)
            })
        };
        match a {
            "--ckpt-dir" => { ckpt_dir = Some(PathBuf::from(next())); i += 2; }
            "--vae"      => { vae = Some(PathBuf::from(next())); i += 2; }
            "--t5"       => { t5 = Some(PathBuf::from(next())); i += 2; }
            "--tokenizer"=> { tokenizer = Some(PathBuf::from(next())); i += 2; }
            "--prompt"   => { prompt = Some(next()); i += 2; }
            "--negative" => { negative = next(); i += 2; }
            "--size"     => {
                let s = next();
                let parts: Vec<&str> = s.split('x').collect();
                if parts.len() != 2 { usage(); }
                let w: usize = parts[0].parse().unwrap_or_else(|_| usage());
                let h: usize = parts[1].parse().unwrap_or_else(|_| usage());
                size = Some((w, h));
                i += 2;
            }
            "--frames"   => { frames   = next().parse().unwrap_or_else(|_| usage()); i += 2; }
            "--steps"    => { steps    = next().parse().unwrap_or_else(|_| usage()); i += 2; }
            "--guidance" => { guidance = next().parse().unwrap_or_else(|_| usage()); i += 2; }
            "--shift"    => { shift    = next().parse().unwrap_or_else(|_| usage()); i += 2; }
            "--seed"     => { seed     = next().parse().unwrap_or_else(|_| usage()); i += 2; }
            "--output"   => { output = Some(PathBuf::from(next())); i += 2; }
            "--t5-cpu"   => { t5_cpu = true; i += 1; }
            _ => { eprintln!("unknown arg: {a}"); usage(); }
        }
    }

    let (width, height) = size.unwrap_or_else(|| { eprintln!("--size required"); usage() });

    // Validate shape constraints.
    assert!((frames - 1) % 4 == 0, "--frames must satisfy (N-1)%4 == 0, got {frames}");
    assert!(width  % 16 == 0, "--size width must be multiple of 16, got {width}");
    assert!(height % 16 == 0, "--size height must be multiple of 16, got {height}");
    let lh = height / 16;
    let lw = width / 16;
    assert!(lh % 2 == 0, "H/16 must be even (DiT patch_size=2), got {lh}");
    assert!(lw % 2 == 0, "W/16 must be even (DiT patch_size=2), got {lw}");

    Args {
        ckpt_dir:  ckpt_dir.unwrap_or_else(|| { eprintln!("--ckpt-dir required"); usage() }),
        vae:       vae.unwrap_or_else(|| { eprintln!("--vae required"); usage() }),
        t5:        t5.unwrap_or_else(|| { eprintln!("--t5 required"); usage() }),
        tokenizer: tokenizer.unwrap_or_else(|| { eprintln!("--tokenizer required"); usage() }),
        prompt:    prompt.unwrap_or_else(|| { eprintln!("--prompt required"); usage() }),
        negative,
        width, height, frames, steps, guidance, shift, seed,
        output:    output.unwrap_or_else(|| PathBuf::from("output/wan22_out.mp4")),
        t5_cpu,
        fps: 24,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tokenize(tokenizer_path: &Path, text: &str) -> anyhow::Result<Vec<i32>> {
    use tokenizers::Tokenizer;
    let tok = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let enc = tok.encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    Ok(enc.get_ids().iter().map(|&u| u as i32).collect())
}

fn gaussian_noise_bf16(
    numel: usize,
    seed: u64,
    shape: &[usize],
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<Tensor> {
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
    let t = Tensor::from_vec(v, Shape::from_dims(shape), device.clone())?
        .to_dtype(DType::BF16)?;
    Ok(t)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let cfg = WanConfig::ti2v_5b();

    // Latent geometry.
    let lat_t = (args.frames - 1) / 4 + 1;
    let lat_h = args.height / 16;
    let lat_w = args.width / 16;
    let patch_t = lat_t / cfg.patch_size[0];
    let patch_h = lat_h / cfg.patch_size[1];
    let patch_w = lat_w / cfg.patch_size[2];
    let seq_len = patch_t * patch_h * patch_w;
    println!(
        "Latent:   [{}, {}, {}, {}]  (C, F, H, W)",
        cfg.in_channels, lat_t, lat_h, lat_w
    );
    println!(
        "Patched:  ({}, {}, {})  seq_len={}",
        patch_t, patch_h, patch_w, seq_len
    );

    // -------------------------------------------------------------------
    // Stage 1 — UMT5-XXL encode prompt + negative
    // -------------------------------------------------------------------
    println!("--- Stage 1: UMT5-XXL text encoding ---");
    let t0 = Instant::now();
    // NOTE: --t5-cpu is honored as a user hint; flame-core's globall device
    // is CUDA. CPU-mode encoding would require a separate host-side path;
    // for now we always run on GPU and drop before loading DiT.
    let _ = args.t5_cpu;
    println!("  [trace] about to Umt5Encoder::load({:?})", args.t5);
    let mut t5 = Umt5Encoder::load(&args.t5, &device)
        .map_err(|e| anyhow::anyhow!("Umt5Encoder::load failed: {e:?}"))?;
    println!("  T5 loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let pos_ids = tokenize(&args.tokenizer, &args.prompt)?;
    let neg_ids = tokenize(&args.tokenizer, &args.negative)?;
    println!("  tokens  pos={}  neg={}", pos_ids.len(), neg_ids.len());

    let cond   = t5.encode(&pos_ids)?;     // [1, 512, 4096]
    let uncond = t5.encode(&neg_ids)?;     // [1, 512, 4096]
    println!("  cond   shape: {:?}", cond.shape().dims());
    println!("  uncond shape: {:?}", uncond.shape().dims());
    drop(t5);
    println!("  T5 dropped, stage 1: {:.1}s", t0.elapsed().as_secs_f32());

    // -------------------------------------------------------------------
    // Stage 2 — DiT Euler denoise
    // -------------------------------------------------------------------
    println!("--- Stage 2: Wan2.2 DiT denoise ---");
    let t0 = Instant::now();
    let mut dit = WanTransformer::load_ti2v_5b(&args.ckpt_dir, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Initial Gaussian noise, [C, F, H, W].
    let numel = cfg.in_channels * lat_t * lat_h * lat_w;
    let mut latent = gaussian_noise_bf16(
        numel,
        args.seed,
        &[cfg.in_channels, lat_t, lat_h, lat_w],
        &device,
    )?;

    // Schedule.
    let EulerSigmas { sigmas, timesteps } = sampler::shifted_sigma_schedule(args.steps, args.shift);
    println!("  sigmas[0..3]={:.4} {:.4} {:.4}  last={:.4}",
        sigmas[0], sigmas[1.min(sigmas.len() - 1)], sigmas[2.min(sigmas.len() - 1)],
        sigmas[sigmas.len() - 1]);

    for step in 0..args.steps {
        let ts       = timesteps[step];
        let sigma    = sigmas[step];
        let sigma_nx = sigmas[step + 1];
        let dt       = sigma_nx - sigma;

        let cond_pred   = dit.forward(&latent, ts, &cond, seq_len)?;
        let uncond_pred = dit.forward(&latent, ts, &uncond, seq_len)?;
        let noise_pred  = sampler::cfg_combine(&cond_pred, &uncond_pred, args.guidance)?;
        latent = sampler::euler_step(&latent, &noise_pred, dt)?;

        if step == 0 || (step + 1) % 5 == 0 || step + 1 == args.steps {
            println!(
                "  step {:>3}/{}  ts={:.1}  sigma={:.4}  ({:.1}s)",
                step + 1, args.steps, ts, sigma, t0.elapsed().as_secs_f32()
            );
        }
    }
    drop(dit);
    drop(cond);
    drop(uncond);
    println!("  DiT dropped, stage 2: {:.1}s", t0.elapsed().as_secs_f32());

    // Print denoised latent stats for diagnostics.
    {
        let v = latent.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let abs_mean = v.iter().map(|x| x.abs()).sum::<f32>() / n;
        let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "  [denoised] mean={mean:.4} |mean|={abs_mean:.4} range=[{lo:.4}, {hi:.4}]"
        );
        // Save for external inspection.
        let mut m = std::collections::HashMap::new();
        m.insert("latent".to_string(), latent.clone());
        let _ = flame_core::serialization::save_file(
            &m, "/tmp/wan22_denoised.safetensors",
        );
    }

    // -------------------------------------------------------------------
    // Stage 3 — VAE decode + mp4 mux
    // -------------------------------------------------------------------
    println!("--- Stage 3: Wan2.2 VAE decode + MP4 mux ---");
    let t0 = Instant::now();
    let vae = Wan22VaeDecoder::load(&args.vae, &device)?;
    println!("  VAE loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // VAE expects [B, C, F, H, W] — add batch dim.
    let latent_b = latent.unsqueeze(0)?;
    let rgb = vae.decode(&latent_b)?; // [1, 3, F_out, H_out, W_out]
    drop(vae);

    let out_dims = rgb.shape().dims().to_vec();
    assert_eq!(out_dims.len(), 5, "VAE output must be 5D, got {:?}", out_dims);
    let (b, c, f_out, h_out, w_out) = (
        out_dims[0], out_dims[1], out_dims[2], out_dims[3], out_dims[4],
    );
    assert_eq!(b, 1);
    assert_eq!(c, 3, "expected RGB output, got {c} channels");
    println!("  decoded: {}x{}, {} frames", w_out, h_out, f_out);

    // Move to host f32 and convert to RGB u8.
    let rgb_f32 = rgb.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    // Strip batch dim — tensor is already contiguous [1, 3, F, H, W].
    let frame_bytes = inference_flame::mux::video_tensor_to_rgb_u8(&rgb_f32, f_out, h_out, w_out);

    // Silent stereo audio (write_mp4 requires audio; we emit zeros matching
    // clip duration so ffmpeg still produces a valid container).
    let duration_secs = f_out as f32 / args.fps as f32;
    let audio_rate: u32 = 48000;
    let n_samples = (duration_secs * audio_rate as f32) as usize;
    let silent: Vec<i16> = vec![0i16; n_samples * 2];

    inference_flame::mux::write_mp4(
        &args.output,
        &frame_bytes,
        f_out,
        w_out,
        h_out,
        args.fps as f32,
        &silent,
        audio_rate,
    )?;

    println!("  MP4 written: {}", args.output.display());
    println!("============================================================");
    println!("Total:   {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
