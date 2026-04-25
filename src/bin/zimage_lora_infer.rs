//! Z-Image inference with a LoRA merged in — pure Rust.
//!
//! Loads the Z-Image base checkpoint, applies a LoRA via
//! `inference_flame::lora_merge::merge_klein_lora` (auto-detects
//! ZImageTrainer / AiToolkit format), builds `NextDiT::new_resident` from
//! the merged weights, and runs the standard Z-Image sampling pipeline.
//!
//! Mirrors `zimage_infer` plus the merge step. Architecture rule honored:
//! LoRA-merge logic lives in inference-flame, not in flame-diffusion.
//!
//! Usage:
//!     zimage_lora_infer \
//!         --base /path/to/z_image_base_bf16.safetensors \
//!         --lora /path/to/lora_step_NNNN.safetensors \
//!         --vae /path/to/vae.safetensors \
//!         --embeddings /path/to/embeds.safetensors \
//!         --output /path/to/out.png
//!
//! Optional: --alpha, --rank, --multiplier, --steps, --cfg, --shift,
//! --width, --height, --seed.

use cudarc::driver::CudaDevice;
use flame_core::{
    global_cuda_device, serialization::load_file_filtered, trim_cuda_mempool, DType, Error,
    Result, Shape, Tensor,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use inference_flame::lora_merge::merge_klein_lora;
use inference_flame::models::zimage_nextdit::NextDiT;
use inference_flame::sampling::euler::euler_step;
use inference_flame::sampling::schedules::build_sigma_schedule;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

struct Args {
    base: PathBuf,
    lora: PathBuf,
    vae: PathBuf,
    embeddings: PathBuf,
    output: PathBuf,
    height: usize,
    width: usize,
    steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
    alpha: Option<f32>,
    rank: Option<usize>,
    multiplier: f32,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        base: PathBuf::from("/home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors"),
        lora: PathBuf::new(),
        vae: PathBuf::from(
            "/home/alex/.serenity/models/zimage_base/vae/diffusion_pytorch_model.safetensors",
        ),
        embeddings: PathBuf::new(),
        output: PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/zimage_lora.png"),
        height: 1024,
        width: 1024,
        steps: 30,
        cfg_scale: 4.0,
        shift: 3.0,
        seed: 42,
        alpha: None,
        rank: None,
        multiplier: 1.0,
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
            "--base" => a.base = PathBuf::from(take(&mut i)?),
            "--lora" => a.lora = PathBuf::from(take(&mut i)?),
            "--vae" => a.vae = PathBuf::from(take(&mut i)?),
            "--embeddings" | "--embeds" => a.embeddings = PathBuf::from(take(&mut i)?),
            "--output" => a.output = PathBuf::from(take(&mut i)?),
            "--width" => a.width = take(&mut i)?.parse()?,
            "--height" => a.height = take(&mut i)?.parse()?,
            "--steps" => a.steps = take(&mut i)?.parse()?,
            "--cfg" | "--cfg_scale" => a.cfg_scale = take(&mut i)?.parse()?,
            "--shift" => a.shift = take(&mut i)?.parse()?,
            "--seed" => a.seed = take(&mut i)?.parse()?,
            "--alpha" => a.alpha = Some(take(&mut i)?.parse()?),
            "--rank" => a.rank = Some(take(&mut i)?.parse()?),
            "--multiplier" | "--strength" => a.multiplier = take(&mut i)?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "zimage_lora_infer --base BASE --lora LORA --vae VAE \
--embeddings EMBEDS [--output PNG] [--width W] [--height H] \
[--steps N] [--cfg G] [--shift S] [--seed N] [--alpha A] [--rank R] [--multiplier M]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
        i += 1;
    }
    if a.lora.as_os_str().is_empty() {
        anyhow::bail!("--lora is required");
    }
    if a.embeddings.as_os_str().is_empty() {
        anyhow::bail!("--embeddings is required");
    }
    Ok(a)
}

fn save_png(rgb: &Tensor, path: &std::path::Path) -> Result<()> {
    let dims = rgb.shape().dims().to_vec();
    let (_b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                pixels[(y * w + x) * 3 + c] = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| Error::InvalidInput("PNG buffer build failed".into()))?
        .save(path)
        .map_err(|e| Error::Io(format!("PNG save: {e}")))
}

fn run(args: Args, device: Arc<CudaDevice>) -> anyhow::Result<()> {
    let t_total = Instant::now();
    let latent_h = args.height / 8;
    let latent_w = args.width / 8;

    println!("=== Z-Image LoRA Inference (pure Rust) ===");
    println!("base: {}", args.base.display());
    println!("lora: {}", args.lora.display());
    println!("vae:  {}", args.vae.display());

    // Stage 1: Load embeddings (cheap; keep on GPU through all stages).
    println!("\n--- Stage 1: Load embeddings ---");
    let emb = flame_core::serialization::load_file(&args.embeddings, &device)?;
    let cap_feats = emb
        .get("cap_feats")
        .ok_or_else(|| anyhow::anyhow!("embeds missing 'cap_feats'"))?
        .to_dtype(DType::BF16)?;
    let cap_feats_uncond = match emb.get("cap_feats_uncond") {
        Some(t) => Some(t.to_dtype(DType::BF16)?),
        None => None,
    };
    println!("  cap: {:?}", cap_feats.shape().dims());
    if let Some(ref u) = cap_feats_uncond {
        println!("  uncond: {:?}", u.shape().dims());
    }
    drop(emb);

    // Stage 2: Load base + LoRA, merge, build NextDiT.
    println!("\n--- Stage 2: Load base + apply LoRA, build NextDiT ---");
    let t_load = Instant::now();
    let base_p = args.base.as_path();
    let mut all_weights = if base_p.is_dir() {
        let mut weights: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        let mut entries: Vec<_> = std::fs::read_dir(base_p)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in &entries {
            let partial = flame_core::serialization::load_file(entry.path(), &device)?;
            weights.extend(partial);
        }
        weights
    } else {
        load_file_filtered(&args.base, &device, |_| true)?
    };
    println!("  base: {} tensors", all_weights.len());

    let lora = flame_core::serialization::load_file(&args.lora, &device)?;
    println!("  lora: {} tensors", lora.len());

    let inferred_rank = lora
        .iter()
        .filter_map(|(k, v)| {
            if k.ends_with(".lora_A") || k.ends_with(".lora_A.weight") {
                Some(v.shape().dims()[0])
            } else {
                None
            }
        })
        .next();
    let rank = args.rank.or(inferred_rank).unwrap_or(16);
    let alpha = args.alpha.unwrap_or(rank as f32);
    println!(
        "  alpha={:.1} rank={} multiplier={:.2} → scale={:.4}",
        alpha,
        rank,
        args.multiplier,
        (alpha / rank as f32) * args.multiplier
    );

    let n_merged = merge_klein_lora(&mut all_weights, &lora, alpha, rank, args.multiplier)?;
    drop(lora);
    trim_cuda_mempool(0);
    println!(
        "  merged {n_merged} LoRA modules in {:.1}s",
        t_load.elapsed().as_secs_f32()
    );

    let mut model = NextDiT::new_resident(all_weights, device.clone());

    // Stage 3: Sample.
    println!("\n--- Stage 3: Denoise ({} steps, cfg={}) ---", args.steps, args.cfg_scale);
    let x = Tensor::randn(
        Shape::from_dims(&[1, 16, latent_h, latent_w]),
        0.0,
        1.0,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let sigmas = build_sigma_schedule(args.steps, args.shift);
    let mut x = x;
    let t_denoise = Instant::now();
    for step in 0..args.steps {
        x = euler_step(
            &mut model,
            &x,
            sigmas[step],
            sigmas[step + 1],
            &cap_feats,
            cap_feats_uncond.as_ref(),
            args.cfg_scale,
        )?;
    }
    println!(
        "  denoised in {:.1}s ({:.2} s/step)",
        t_denoise.elapsed().as_secs_f32(),
        t_denoise.elapsed().as_secs_f32() / args.steps as f32,
    );

    // Stage 4: VAE decode (drop DiT first per the staged-loading pattern).
    println!("\n--- Stage 4: VAE decode ---");
    drop(model);
    trim_cuda_mempool(0);

    let vae = LdmVAEDecoder::from_safetensors(
        &args.vae.to_string_lossy(),
        16,
        0.3611,
        0.1159,
        &device,
    )?;
    let t_vae = Instant::now();
    let rgb = vae.decode(&x)?;
    println!(
        "  decoded {:?} in {:.1}s",
        rgb.shape().dims(),
        t_vae.elapsed().as_secs_f32()
    );
    drop(vae);
    drop(x);

    save_png(&rgb, &args.output)?;
    println!("\nIMAGE SAVED: {}", args.output.display());
    println!("Total time: {:.1}s", t_total.elapsed().as_secs_f32());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args()?;
    let device = global_cuda_device();
    run(args, device)
}
