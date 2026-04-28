//! Qwen-Image-2512 with a runtime-applied LoRA — pure Rust.
//!
//! Loads the cached cond+uncond embeddings (saved by
//! `scripts/qwenimage_encode.py`), loads the Qwen DiT (BlockOffloader),
//! attaches a LoRA stack via `set_lora` (base weights NEVER mutated),
//! runs the true-CFG denoise loop, and decodes via the in-tree
//! `QwenImageVaeDecoder`. No Python in the denoise path.
//!
//! ## LoRA application
//!
//! Standard ai-toolkit / OneTrainer / musubi-tuner LoRAs trained
//! against the diffusers Qwen-Image checkpoint use
//! `diffusion_model.transformer_blocks.{i}.<sub>.lora_{A,B}.weight`
//! naming. The generic ai-toolkit fallback in `inference-flame::lora`
//! strips `diffusion_model.` and matches against the base weight key —
//! no qwenimage-specific prefix mapper is needed for the standard
//! split-Q/K/V 2512 base. (Fused-base "turbo loader" checkpoints with
//! `attn.to_qkv.weight` / `attn.add_qkv_proj.weight` would need a
//! `Slot::RowRange` mapping; not yet wired — see
//! `HANDOFF_2026-04-28_INFERENCE_FLAME_LORA_ROLLOUT.md` Step 4.)
//!
//! ## Usage
//!
//!     qwenimage_lora_gen \
//!         --embeds /path/to/qwenimage_embeds.safetensors \
//!         --lora   /path/to/lora.safetensors \
//!         --output /path/to/out.png \
//!         [--multiplier 0.7] \
//!         [--width 1024] [--height 1024] [--steps 50] [--cfg 4.0] \
//!         [--seed 42] [--sampler euler|dpmpp_2m|res_2m|res_3m|deis_3m]
//!
//! The DiT base shards default to the standard Qwen-Image-2512 layout
//! under `~/.serenity/models/checkpoints/qwen-image-2512/transformer/`.
//! Override with `--base-shards path1:path2:...` (colon-separated) or
//! the legacy `QWEN_DIT_SHARDS` env var.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::lora::LoraStack;
use inference_flame::models::qwenimage_dit::QwenImageDit;
use inference_flame::sampling::exponential_multistep::{
    deis_3m_step, dpmpp_2m_step, lambda_from_sigma, res_2m_step, res_3m_step, MultistepHistory,
};

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
const DEFAULT_VAE: &str =
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors";
const DEFAULT_EMBEDS: &str =
    "/home/alex/EriDiffusion/inference-flame/output/qwenimage_embeds.safetensors";
const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/qwenimage_lora.png";

const VAE_SCALE_FACTOR: usize = 8;
const PATCH_SIZE: usize = 2;
const IN_CHANNELS: usize = 16;
const PACKED_CHANNELS: usize = 64;

struct Args {
    embeds: PathBuf,
    output: PathBuf,
    lora: PathBuf,
    vae: PathBuf,
    base_shards: Vec<String>,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    multiplier: f32,
    sampler: Sampler,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let env_shards: Option<Vec<String>> = std::env::var("QWEN_DIT_SHARDS")
        .ok()
        .map(|s| s.split(':').map(|p| p.to_string()).collect());
    let mut a = Args {
        embeds: PathBuf::from(DEFAULT_EMBEDS),
        output: PathBuf::from(DEFAULT_OUTPUT),
        lora: PathBuf::new(),
        vae: PathBuf::from(
            std::env::var("QWEN_VAE_PATH").unwrap_or_else(|_| DEFAULT_VAE.to_string()),
        ),
        base_shards: env_shards
            .unwrap_or_else(|| DEFAULT_DIT_SHARDS.iter().map(|s| s.to_string()).collect()),
        width: 1024,
        height: 1024,
        steps: 50,
        cfg: 4.0,
        seed: 42,
        multiplier: 1.0,
        sampler: Sampler::Euler,
    };
    let mut i = 1;
    while i < argv.len() {
        let take = |i: &mut usize| -> anyhow::Result<String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "missing value for {}",
                    argv.get(*i - 1).cloned().unwrap_or_default()
                )
            })
        };
        match argv[i].as_str() {
            "--embeds" => a.embeds = PathBuf::from(take(&mut i)?),
            "--lora" => a.lora = PathBuf::from(take(&mut i)?),
            "--vae" => a.vae = PathBuf::from(take(&mut i)?),
            "--output" | "-o" => a.output = PathBuf::from(take(&mut i)?),
            "--base-shards" => {
                a.base_shards = take(&mut i)?.split(':').map(|s| s.to_string()).collect();
            }
            "--width" => a.width = take(&mut i)?.parse()?,
            "--height" => a.height = take(&mut i)?.parse()?,
            "--steps" => a.steps = take(&mut i)?.parse()?,
            "--cfg" | "--guidance" => a.cfg = take(&mut i)?.parse()?,
            "--seed" => a.seed = take(&mut i)?.parse()?,
            "--multiplier" | "--strength" => a.multiplier = take(&mut i)?.parse()?,
            "--sampler" => {
                let s = take(&mut i)?;
                a.sampler = Sampler::parse(&s)
                    .ok_or_else(|| anyhow::anyhow!("unknown sampler '{}'", s))?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
        i += 1;
    }
    if a.lora.as_os_str().is_empty() {
        anyhow::bail!("--lora is required");
    }
    Ok(a)
}

fn print_usage() {
    eprintln!(
        "qwenimage_lora_gen --lora LORA [--embeds PATH] [--output PNG] \
         [--vae VAE] [--base-shards p1:p2:...] [--width W] [--height H] \
         [--steps N] [--cfg G] [--seed S] [--multiplier M] \
         [--sampler euler|dpmpp_2m|res_2m|res_3m|deis_3m]"
    );
}

/// Read every weight key from a list of safetensors shards by parsing
/// the JSON header only — no tensor data is loaded. Cheap (each header
/// is tens of KB even for multi-GB shards).
fn read_safetensors_keys(paths: &[String]) -> anyhow::Result<HashSet<String>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut keys: HashSet<String> = HashSet::new();
    for p in paths {
        let mut f = std::fs::File::open(p)
            .map_err(|e| anyhow::anyhow!("open {p}: {e}"))?;
        let mut size_buf = [0u8; 8];
        f.read_exact(&mut size_buf)?;
        let header_size = u64::from_le_bytes(size_buf) as usize;
        let mut header_buf = vec![0u8; header_size];
        f.read_exact(&mut header_buf)?;
        let _ = f.seek(SeekFrom::Start(0));
        let v: serde_json::Value = serde_json::from_slice(&header_buf)?;
        if let Some(obj) = v.as_object() {
            for k in obj.keys() {
                if k != "__metadata__" {
                    keys.insert(k.clone());
                }
            }
        }
    }
    Ok(keys)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();
    let args = parse_args()?;

    println!("============================================================");
    println!("Qwen-Image LoRA Inference (pure Rust)");
    println!("============================================================");
    println!("embeds  : {}", args.embeds.display());
    println!("lora    : {}", args.lora.display());
    println!("vae     : {}", args.vae.display());
    println!("output  : {}", args.output.display());
    println!(
        "size    : {}x{}, steps={}, cfg={}, sampler={}, seed={}, mult={:.3}",
        args.width,
        args.height,
        args.steps,
        args.cfg,
        args.sampler.label(),
        args.seed,
        args.multiplier,
    );

    // ------------------------------------------------------------------
    // Stage A: cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage A: Loading cached embeddings ---");
    let t0 = Instant::now();
    let tensors = flame_core::serialization::load_file(&args.embeds, &device)?;
    let cond = ensure_bf16(
        tensors
            .get("cond")
            .ok_or_else(|| anyhow::anyhow!("Missing 'cond' in {}", args.embeds.display()))?
            .clone(),
    )?;
    let uncond = ensure_bf16(
        tensors
            .get("uncond")
            .ok_or_else(|| anyhow::anyhow!("Missing 'uncond' in {}", args.embeds.display()))?
            .clone(),
    )?;
    drop(tensors);
    println!("  cond:   {:?}", cond.shape().dims());
    println!("  uncond: {:?}", uncond.shape().dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage B: base shard headers → base_keys for kohya naming, then DiT
    // ------------------------------------------------------------------
    println!("\n--- Stage B: Loading Qwen-Image DiT (BlockOffloader) ---");
    let base_keys = read_safetensors_keys(&args.base_shards)?;
    println!("  base shards: {} ({} unique keys)", args.base_shards.len(), base_keys.len());
    let t0 = Instant::now();
    let shards: Vec<&str> = args.base_shards.iter().map(|s| s.as_str()).collect();
    let mut dit = QwenImageDit::load(&shards, &device)?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage C: LoRA stack → set_lora
    // ------------------------------------------------------------------
    println!("\n--- Stage C: Loading LoRA ---");
    let t0 = Instant::now();
    let lora_path_str = args
        .lora
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("lora path is not utf8"))?;
    let stack = LoraStack::load(lora_path_str, &base_keys, args.multiplier, &device)?;
    println!(
        "  loaded {} target weight(s) in {:.1}s",
        stack.target_count(),
        t0.elapsed().as_secs_f32()
    );
    dit.set_lora(Arc::new(stack));

    // ------------------------------------------------------------------
    // Stage D: build noise, pack, run CFG denoise
    // ------------------------------------------------------------------
    println!("\n--- Stage D: Denoise ({} steps, cfg={}) ---", args.steps, args.cfg);
    let h_latent_full = args.height / VAE_SCALE_FACTOR;
    let w_latent_full = args.width / VAE_SCALE_FACTOR;
    let h_patched = h_latent_full / PATCH_SIZE;
    let w_patched = w_latent_full / PATCH_SIZE;
    let seq_len = h_patched * w_patched;
    println!(
        "  Latent raw [B,F,C,H,W] = [1, 1, {}, {}, {}]",
        IN_CHANNELS, h_latent_full, w_latent_full
    );
    println!("  Packed [B, seq, C*p*p] = [1, {}, {}]", seq_len, PACKED_CHANNELS);

    let numel = IN_CHANNELS * h_latent_full * w_latent_full;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
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

    // Pack: diffusers `_pack_latents`.
    let mut packed = vec![0.0f32; seq_len * PACKED_CHANNELS];
    for c in 0..IN_CHANNELS {
        for hp in 0..h_patched {
            for wp in 0..w_patched {
                for dh in 0..PATCH_SIZE {
                    for dw in 0..PATCH_SIZE {
                        let src_h = hp * PATCH_SIZE + dh;
                        let src_w = wp * PATCH_SIZE + dw;
                        let src_idx =
                            c * h_latent_full * w_latent_full + src_h * w_latent_full + src_w;
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

    // ── Sigma schedule: dynamic exponential shift ──
    let base_shift: f32 = 0.5;
    let max_shift: f32 = 0.9;
    let base_seq_len_f: f32 = 256.0;
    let max_seq_len_shift: f32 = 8192.0;
    let shift_terminal: f32 = 0.02;
    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len_f);
    let bb = base_shift - m * base_seq_len_f;
    let mu = (seq_len as f32) * m + bb;
    let exp_mu = mu.exp();

    let mut sigmas: Vec<f32> = (0..args.steps)
        .map(|i| {
            let t = i as f32 / (args.steps - 1) as f32;
            1.0 - t * (1.0 - 1.0 / args.steps as f32)
        })
        .collect();
    for s in sigmas.iter_mut() {
        *s = exp_mu / (exp_mu + (1.0 / *s - 1.0));
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
    println!(
        "  sigmas[0]={:.4}  sigmas[-2]={:.4}  sigmas[-1]={:.4}",
        sigmas[0],
        sigmas[args.steps - 1],
        sigmas[args.steps]
    );

    // ── CFG sampler loop ──
    let frame = 1;
    let t_denoise = Instant::now();
    let mut history = MultistepHistory::new(3);
    for step in 0..args.steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;

        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![sigma_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            let cond_pred = dit.forward(&x, &cond, &t_vec, (frame, h_patched, w_patched))?;
            let uncond_pred = dit.forward(&x, &uncond, &t_vec, (frame, h_patched, w_patched))?;

            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(args.cfg)?;
            let comb = uncond_pred.add(&scaled)?;
            let noise_pred = norm_rescale_cfg(&cond_pred, &comb).unwrap_or(comb);

            match args.sampler {
                Sampler::Euler => {
                    let step_tensor = noise_pred.mul_scalar(dt)?;
                    x.add(&step_tensor)?
                }
                _ => {
                    let v_sig = noise_pred.mul_scalar(sigma_curr)?;
                    let denoised = x.sub(&v_sig)?;
                    let next = match args.sampler {
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

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == args.steps {
            println!(
                "  step {}/{}  sigma={:.4}  ({:.1}s elapsed)",
                step + 1,
                args.steps,
                sigma_curr,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step)",
        dt_denoise,
        dt_denoise / args.steps as f32,
    );

    drop(dit);
    drop(cond);
    drop(uncond);
    println!("  DiT + embeddings evicted");

    // ------------------------------------------------------------------
    // Stage E: VAE decode + save PNG
    // ------------------------------------------------------------------
    println!("\n--- Stage E: VAE decode (Rust) ---");
    let h_lat = 2 * (args.height / (VAE_SCALE_FACTOR * 2));
    let w_lat = 2 * (args.width / (VAE_SCALE_FACTOR * 2));
    let unpacked = x
        .reshape(&[1, h_lat / 2, w_lat / 2, PACKED_CHANNELS / 4, 2, 2])?
        .permute(&[0, 3, 1, 4, 2, 5])?
        .reshape(&[1, PACKED_CHANNELS / (2 * 2), 1, h_lat, w_lat])?;

    let t_vae = Instant::now();
    let decoder = inference_flame::vae::QwenImageVaeDecoder::from_safetensors(
        args.vae.to_str().expect("vae path utf8"),
        &device,
    )?;
    let rgb = decoder.decode(&unpacked)?;
    println!(
        "  VAE decoded {:?} in {:.1}s",
        rgb.shape().dims(),
        t_vae.elapsed().as_secs_f32()
    );

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
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::save_buffer(&args.output, &buf, w_img as u32, h_img as u32, image::ColorType::Rgb8)?;
    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", args.output.display());
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    let _ = (device, HashMap::<String, Tensor>::new());
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
