//! Boogu-Image C8 — denoise + VAE decode → PNG (the runnable T2I tail).
//!
//! Loads the cond + uncond instruction hidden-state cache produced by
//! `boogu_encode`, loads the resident [`BooguDiT`], inits a seeded Gaussian noise
//! latent, runs the flow-match Euler denoise loop (DiT run TWICE per step for
//! CFG), then VAE-decodes (stock FLUX 16ch [`LdmVAEDecoder`]) and writes a PNG.
//!
//! Usage:
//!   boogu_infer [--size N] [--steps N] [--seed N]
//!
//! Defaults: `--size 1024 --steps 20 --seed 42`. Use `--size 256` or `--size 512`
//! for a fast smoke (the resident DiT dominates VRAM; small sizes fit easily).
//!
//! ## Denoise (matches `pipeline_boogu.py` T2I path + the v1 scheduler)
//!
//! - Schedule: [`build_boogu_timesteps`] (`linspace(0,1,N+1)[:-1]` → v1 static
//!   shift mu=1.15 → trailing 1.0), ASCENDING t.
//! - Per step `i` (t=ts[i], t_next=ts[i+1]):
//!     `pred_cond  = DiT(latent, [t], cond_hidden)`
//!     `pred_uncond = DiT(latent, [t], uncond_hidden)`
//!     `v = pred_cond + (text_guidance_scale - 1)*(pred_cond - pred_uncond)`   (cfg=4.0)
//!     `latent = latent + (t_next - t) * v`                                    (dt>0, NO sign flip)
//! - The DiT is fed the **RAW** timestep `[t]` (it ×1000 internally — PORT_STATE
//!   open issue #5 / skeptic F4).
//! - The latent is held in **F32** across the whole loop; the DiT casts its
//!   inputs to BF16 with round-to-nearest-even internally
//!   (`transformer.rs` `to_dtype(BF16)`), and the BF16 velocity is upcast back to
//!   F32 for the Euler update (the NAVA F32-latent/RNE-cast discipline; matches
//!   the oracle's `sample.to(float32)` before the step).
//! - CFG uncond is the DROP-template encoding (from `boogu_encode`); cond/uncond
//!   may differ in seq length — the DiT handles each independently.
//!
//! ## VAE
//!
//! Reuses the standalone FLUX `ae.safetensors` (identical 16ch weights to
//! FLUX.1/Chroma/Z-Image). `LdmVAEDecoder::decode` folds the rescale
//! `z/scaling_factor + shift_factor` (0.3611 / 0.1159) **inside** decode — so we
//! pass the RAW final latent, NOT a pre-rescaled one (mirrors `flux1_infer`).
//!
//! Pure-Rust runtime — no Python.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaDevice;

use flame_core::serialization::load_file;
use flame_core::{DType, Shape, Tensor};

use inference_flame::models::boogu::config::BooguConfig;
use inference_flame::models::boogu::loader::load_component;
use inference_flame::models::boogu::transformer::BooguDiT;
use inference_flame::sampling::boogu_sampling::build_boogu_timesteps;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

const REPO: &str = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/ae.safetensors";
const EMBED_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/boogu_embeddings.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/boogu_rust.png";

// FLUX 16ch VAE constants (folded inside decode).
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;
/// FLUX VAE spatial downsample factor (pixel -> latent).
const VAE_DOWNSAMPLE: usize = 8;

/// CFG `text_guidance_scale` (`pipeline_boogu.py` default 4.0).
const TEXT_GUIDANCE_SCALE: f32 = 4.0;

// Defaults (overridable via flags).
const DEFAULT_SIZE: usize = 1024;
const DEFAULT_STEPS: usize = 20;
const DEFAULT_SEED: u64 = 42;

struct Args {
    size: usize,
    steps: usize,
    seed: u64,
    /// PARITY: load `[1,16,H,W]` F32 initial noise (key `tensor`) instead of
    /// seeding Box-Muller — byte-identical to the torch oracle's noise.
    noise_file: Option<String>,
    /// PARITY: dump the final pre-decode F32 latent (key `tensor`) for cos compare.
    latent_out: Option<String>,
    /// PARITY (bisection): load cond/uncond hidden states (keys `cond_hidden` /
    /// `uncond_hidden`) instead of the encoder cache — injects the torch encoder's
    /// hidden states to isolate encoder (C7) from the denoise/CFG/VAE trajectory.
    hidden_file: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        size: DEFAULT_SIZE,
        steps: DEFAULT_STEPS,
        seed: DEFAULT_SEED,
        noise_file: None,
        latent_out: None,
        hidden_file: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--size" => {
                a.size = it
                    .next()
                    .ok_or_else(|| anyhow!("--size needs a value"))?
                    .parse()
                    .context("--size")?;
            }
            "--steps" => {
                a.steps = it
                    .next()
                    .ok_or_else(|| anyhow!("--steps needs a value"))?
                    .parse()
                    .context("--steps")?;
            }
            "--seed" => {
                a.seed = it
                    .next()
                    .ok_or_else(|| anyhow!("--seed needs a value"))?
                    .parse()
                    .context("--seed")?;
            }
            "--noise-file" => {
                a.noise_file =
                    Some(it.next().ok_or_else(|| anyhow!("--noise-file needs a path"))?);
            }
            "--latent-out" => {
                a.latent_out =
                    Some(it.next().ok_or_else(|| anyhow!("--latent-out needs a path"))?);
            }
            "--hidden-file" => {
                a.hidden_file =
                    Some(it.next().ok_or_else(|| anyhow!("--hidden-file needs a path"))?);
            }
            other => {
                return Err(anyhow!(
                    "unknown arg `{other}` (use --size/--steps/--seed/\
                     --noise-file/--latent-out/--hidden-file)"
                ))
            }
        }
    }
    if a.size % (VAE_DOWNSAMPLE * 2) != 0 {
        return Err(anyhow!(
            "--size {} must be divisible by {} (8x VAE downsample x 2 patch)",
            a.size,
            VAE_DOWNSAMPLE * 2
        ));
    }
    Ok(a)
}

/// Seeded Box-Muller Gaussian noise (F32), same idiom as `flux1_infer`.
fn seeded_noise_f32(numel: usize, seed: u64) -> Vec<f32> {
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
}

/// Load `[1, L, 4096]` BF16 tensor by key from the embedding cache.
fn load_hidden(map: &std::collections::HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    let t = map
        .get(key)
        .ok_or_else(|| anyhow!("embedding cache missing `{key}` (run boogu_encode first)"))?;
    let d = t.shape().dims();
    if d.len() != 3 || d[0] != 1 || d[2] != 4096 {
        return Err(anyhow!("`{key}` must be [1,L,4096], got {d:?}"));
    }
    // Ensure BF16 (the DiT instruction path expects BF16).
    if t.dtype() == DType::BF16 {
        Ok(t.clone())
    } else {
        t.to_dtype(DType::BF16).map_err(|e| anyhow!("cast {key}: {e}"))
    }
}

fn main() -> Result<()> {
    env_logger::init();
    // INFERENCE: autograd OFF for the whole run. Without this guard the global
    // autograd context (which defaults to ENABLED) records every op of the
    // 20-step × 2-CFG denoise loop, saving Arc clones of the DiT weight + intermediate
    // tensors into the thread-local graph. Those saved tensors keep the DiT's
    // ~23 GB of CUDA slabs alive even after `drop(dit)`, so the subsequent VAE
    // decode OOMs (measured: pool free-list empty at decode, DiT still resident).
    // Held for the whole of main() — dropping at scope exit restores prior state.
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let t_total = Instant::now();
    let args = parse_args()?;

    let cfg = BooguConfig::default();
    let latent_h = args.size / VAE_DOWNSAMPLE;
    let latent_w = args.size / VAE_DOWNSAMPLE;
    let img_tokens = (latent_h / cfg.patch_size) * (latent_w / cfg.patch_size);

    println!("============================================================");
    println!("Boogu-Image — Inference (denoise + VAE decode)");
    println!("============================================================");
    println!(
        "  size={}x{} latent=[1,{},{},{}] img_tokens={} steps={} seed={} cfg={}",
        args.size, args.size, AE_IN_CHANNELS, latent_h, latent_w, img_tokens, args.steps, args.seed,
        TEXT_GUIDANCE_SCALE
    );

    let device: Arc<CudaDevice> = CudaDevice::new(0).context("cuda device 0")?;

    // --- Load instruction hidden-state cache (cond + uncond) ---
    // PARITY: --hidden-file overrides the encoder cache (inject torch encoder's
    // hidden states to bisect encoder vs trajectory); default = boogu_encode cache.
    let hidden_src = args.hidden_file.as_deref().unwrap_or(EMBED_PATH);
    println!("\n--- Loading instruction cache ---  src={hidden_src}");
    let embed_map = load_file(Path::new(hidden_src), &device)
        .map_err(|e| anyhow!("load {hidden_src}: {e}"))?;
    let cond_hidden = load_hidden(&embed_map, "cond_hidden")?;
    let uncond_hidden = load_hidden(&embed_map, "uncond_hidden")?;
    drop(embed_map);
    println!(
        "  cond {:?}, uncond {:?}",
        cond_hidden.shape().dims(),
        uncond_hidden.shape().dims()
    );

    // --- Load resident DiT ---
    println!("\n--- Loading DiT (resident) ---");
    let t0 = Instant::now();
    let weights = load_component(Path::new(REPO), "transformer", &device)
        .map_err(|e| anyhow!("load transformer: {e}"))?;
    let dit = BooguDiT::load(weights, cfg, device.clone())
        .map_err(|e| anyhow!("BooguDiT::load: {e}"))?;
    println!("  DiT resident in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Init noise latent (F32 master across the loop) ---
    // PARITY: --noise-file loads byte-identical F32 noise (key `tensor`) so both
    // sides start from the SAME randn (we do NOT rely on Rust RNG matching torch).
    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise = if let Some(npath) = args.noise_file.as_deref() {
        let nmap =
            load_file(Path::new(npath), &device).map_err(|e| anyhow!("load noise {npath}: {e}"))?;
        let nt = nmap
            .get("tensor")
            .ok_or_else(|| anyhow!("noise file {npath} missing 'tensor' key"))?
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("noise->f32: {e}"))?;
        let nv = nt.to_vec_f32().map_err(|e| anyhow!("noise to vec: {e}"))?;
        if nv.len() != numel {
            return Err(anyhow!(
                "noise file numel {} != expected {} (size {}x{} -> latent {}x{}x{})",
                nv.len(),
                numel,
                args.size,
                args.size,
                AE_IN_CHANNELS,
                latent_h,
                latent_w
            ));
        }
        println!("  initial latent: LOADED from {npath} ({numel} F32, parity noise)");
        nv
    } else {
        println!("  initial latent: SEEDED Box-Muller (seed {})", args.seed);
        seeded_noise_f32(numel, args.seed)
    };
    let mut latent = Tensor::from_vec(
        noise,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )
    .map_err(|e| anyhow!("init latent: {e}"))?; // F32

    // --- Denoise loop ---
    let timesteps = build_boogu_timesteps(args.steps); // N+1 ascending, trailing 1.0
    println!(
        "\n--- Denoise ({} steps) ---  schedule t[0]={:.5} t[1]={:.5} t[-1]={:.5}",
        args.steps,
        timesteps[0],
        timesteps[1],
        timesteps[timesteps.len() - 1]
    );
    let t_loop = Instant::now();
    let mut step_times = Vec::with_capacity(args.steps);
    for i in 0..args.steps {
        let t = timesteps[i];
        let t_next = timesteps[i + 1];
        let dt = t_next - t; // > 0
        let t_step = Instant::now();

        // RAW timestep fed to the DiT (it scales ×1000 internally). Latent is F32;
        // the DiT casts inputs to BF16 (RNE) inside forward.
        let pred_cond = dit
            .forward(&latent, &[t], &cond_hidden)
            .map_err(|e| anyhow!("DiT cond fwd (step {i}): {e}"))?;
        let pred_uncond = dit
            .forward(&latent, &[t], &uncond_hidden)
            .map_err(|e| anyhow!("DiT uncond fwd (step {i}): {e}"))?;

        // v = pred_cond + (cfg-1)*(pred_cond - pred_uncond)  -- combine in F32.
        let pred_cond_f32 = pred_cond
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("cond->f32: {e}"))?;
        let pred_uncond_f32 = pred_uncond
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("uncond->f32: {e}"))?;
        let delta = pred_cond_f32
            .sub(&pred_uncond_f32)
            .map_err(|e| anyhow!("cond-uncond: {e}"))?;
        let velocity = pred_cond_f32
            .add(&delta.mul_scalar(TEXT_GUIDANCE_SCALE - 1.0)?)
            .map_err(|e| anyhow!("cfg combine: {e}"))?;

        // Euler: latent = latent + dt * velocity (F32, dt>0, no sign flip).
        latent = latent
            .add(&velocity.mul_scalar(dt)?)
            .map_err(|e| anyhow!("euler step {i}: {e}"))?;

        let st = t_step.elapsed().as_secs_f32();
        step_times.push(st);
        println!(
            "  step {:2}/{}  t={:.5} -> {:.5} (dt={:.5})  {:.2}s",
            i + 1,
            args.steps,
            t,
            t_next,
            dt,
            st
        );
    }
    let loop_s = t_loop.elapsed().as_secs_f32();
    let avg = step_times.iter().sum::<f32>() / step_times.len().max(1) as f32;
    println!("  denoise total {:.1}s, avg {:.2}s/step", loop_s, avg);

    // PARITY: dump the final pre-decode F32 latent (key `tensor`) for cos compare.
    if let Some(lpath) = args.latent_out.as_deref() {
        let latent_f32 = latent
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("latent_out->f32: {e}"))?;
        let mut m: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        m.insert("tensor".to_string(), latent_f32);
        if let Some(parent) = Path::new(lpath).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        flame_core::serialization::save_file(&m, lpath)
            .map_err(|e| anyhow!("save latent_out {lpath}: {e}"))?;
        println!("  PARITY: dumped final pre-decode latent -> {lpath}");
    }

    // Free DiT before VAE (the resident DiT dominates VRAM). Dropping the DiT
    // tensors only returns their slabs to flame_core's CUDA alloc pool (cached,
    // NOT cudaFree'd) — so ~20 GB stays held and the VAE conv's large transient
    // im2col alloc (~1.15 GB single block at the 512² up-blocks) OOMs against the
    // still-full device. `clear_pool_cache()` does the actual cudaFree, releasing
    // the cached slabs back to the OS so the VAE (and the tiled decode tiles) can
    // allocate. Without this, even the per-tile decode OOMs.
    drop(dit);
    drop(cond_hidden);
    drop(uncond_hidden);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    println!("  DiT evicted (pool cache cleared)");

    // --- VAE decode (rescale folded INSIDE decode → pass RAW latent) ---
    println!("\n--- VAE decode ---");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        VAE_PATH,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &device,
    )
    .map_err(|e| anyhow!("VAE load: {e}"))?;
    // decode() does z = z/scale + shift internally; feed the raw (un-rescaled)
    // final latent.
    //
    // The MONOLITHIC decode fits ≤768² but OOMs at 1024² on a 24 GB GPU (conv
    // im2col @ 1024² + mid-attn over the full spatial grid needs ~+2.25 GB past
    // the 21.7 GB DiT peak). For size ≥ 1024 we switch to TILED decode (a fixed
    // 3×3 grid of overlapping LAT/2 latent crops → 512² tiles → feathered 3×3
    // blend → seamless 1024²), mirroring the Mojo pipeline's `_decode_and_save`
    // which folds tiled decode in-process. Threshold = latent ≥ 128 (= 1024 px).
    const TILED_LATENT_THRESHOLD: usize = 128; // 1024 px / 8
    let rgb = if latent_h >= TILED_LATENT_THRESHOLD || latent_w >= TILED_LATENT_THRESHOLD {
        println!(
            "  TILED decode (latent {latent_h}x{latent_w} ≥ {TILED_LATENT_THRESHOLD}): \
             9× overlapping {0}x{0} crops → feathered 3×3 blend",
            latent_h / 2
        );
        // decode_tiled_1024 casts each crop to BF16 for decode and returns F32;
        // it feeds the RAW latent crops (rescale folded inside decode).
        let img = inference_flame::models::boogu::decode_1024::decode_tiled_1024(&vae, &latent)
            .map_err(|e| anyhow!("tiled VAE decode: {e}"))?;
        drop(latent);
        drop(vae);
        img
    } else {
        // The conv path is BF16-only, so cast the F32 master latent to BF16 here
        // (the terminal cast; the F32-master discipline held through the whole
        // denoise loop). Same as `flux1_infer`, whose latent is already BF16.
        let latent_bf16 = latent
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("latent->bf16: {e}"))?;
        let rgb = vae
            .decode(&latent_bf16)
            .map_err(|e| anyhow!("VAE decode: {e}"))?;
        drop(latent_bf16);
        drop(latent);
        drop(vae);
        rgb
    };
    println!(
        "  decoded {:?} in {:.1}s",
        rgb.shape().dims(),
        t0.elapsed().as_secs_f32()
    );

    // --- Denormalize [-1,1] -> u8, CHW->HWC, save PNG ---
    let rgb_f32 = rgb.to_dtype(DType::F32).map_err(|e| anyhow!("rgb->f32: {e}"))?;
    let data = rgb_f32.to_vec_f32().map_err(|e| anyhow!("rgb to vec: {e}"))?;
    let dims = rgb_f32.shape().dims().to_vec();
    let (out_c, out_h, out_w) = (dims[1], dims[2], dims[3]);
    if out_c != 3 {
        return Err(anyhow!("VAE decoder must return 3 channels, got {out_c}"));
    }
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                let v = data[idx].clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * out_w + x) * 3 + c] = u;
            }
        }
    }
    if let Some(parent) = Path::new(OUTPUT_PATH).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow!("failed to build RgbImage"))?;
    img.save(OUTPUT_PATH)?;

    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {OUTPUT_PATH}");
    println!("Total time:  {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    let _ = device; // keep alive
    Ok(())
}
