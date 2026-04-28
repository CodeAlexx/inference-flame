//! Klein 4B/9B image generation with a runtime-applied klein-trainer LoRA —
//! pure Rust, no Python.
//!
//! Loads the base checkpoint, builds a `KleinTransformer`, then attaches a
//! `LoraStack` via `set_lora`. The base weights are NEVER mutated — at
//! every double/single-block linear chokepoint the model adds
//! `scale * up(down(x))` from any matching LoRA entries to the base
//! matmul output. This matches how ai-toolkit, OneTrainer, and
//! musubi-tuner all apply LoRAs at sampling time.
//!
//! Klein-4B single-block linear1 is `[3*inner+2*mlp_hidden, inner_dim]`,
//! but external LoRAs target only the QKV slice (first 3*inner_dim rows).
//! `Slot::Rows(KLEIN_4B_SINGLE_QKV_ROWS)` in `lora.rs` handles that.
//! Constants in `lora.rs` are Klein-4B specific; Klein-9B (inner_dim=4096)
//! would need different `KLEIN_9B_*` constants and a slot variant per
//! model size — not yet wired.
//!
//! Usage:
//!     klein_lora_infer \
//!         --base   /path/to/flux-2-klein-base-4b.safetensors \
//!         --lora   /path/to/lora_step_001500.safetensors \
//!         --vae    /path/to/flux2-vae.safetensors \
//!         --qwen3  /path/to/qwen_3_4b.safetensors \
//!         --tokenizer /path/to/tokenizer.json \
//!         --prompt "a queen in royal robes, photorealistic" \
//!         --output /path/to/out.png
//!
//! Optional knobs: --alpha, --rank, --multiplier, --steps, --guidance,
//! --width, --height, --seed, --negative.

use flame_core::{global_cuda_device, trim_cuda_mempool, DType, Shape, Tensor};
use inference_flame::lora::LoraStack;
use inference_flame::models::klein::KleinTransformer;
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::sampling::klein_sampling::{box_muller_noise, euler_denoise, get_schedule};
use inference_flame::vae::klein_vae::KleinVaeDecoder;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const TXT_PAD_LEN: usize = 512;
const PAD_ID: i32 = 151643;

const DEFAULT_NEGATIVE: &str =
    "lowres, bad quality, worst quality, bad anatomy, blurry, watermark, simple background, transparent background, sketch, jpeg artifacts, ugly, poorly drawn, censor";

struct Args {
    base: PathBuf,
    lora: PathBuf,
    vae: PathBuf,
    qwen3: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    negative: String,
    output: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f32,
    seed: u64,
    alpha: Option<f32>,
    rank: Option<usize>,
    multiplier: f32,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        base: PathBuf::from(
            "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-4b.safetensors",
        ),
        lora: PathBuf::new(),
        vae: PathBuf::from("/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors"),
        qwen3: PathBuf::from("/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"),
        tokenizer: PathBuf::from(
            "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json",
        ),
        prompt: String::new(),
        negative: DEFAULT_NEGATIVE.to_string(),
        output: PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/klein_lora.png"),
        width: 1024,
        height: 1024,
        steps: 50,
        guidance: 4.0,
        seed: 42,
        alpha: None,
        rank: None,
        multiplier: 1.0,
    };
    let mut i = 1;
    while i < argv.len() {
        let take = |i: &mut usize| -> anyhow::Result<String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| anyhow::anyhow!(
                "missing value for --{}",
                argv.get(*i - 1).cloned().unwrap_or_default()
            ))
        };
        match argv[i].as_str() {
            "--base" => a.base = PathBuf::from(take(&mut i)?),
            "--lora" => a.lora = PathBuf::from(take(&mut i)?),
            "--vae" => a.vae = PathBuf::from(take(&mut i)?),
            "--qwen3" => a.qwen3 = PathBuf::from(take(&mut i)?),
            "--tokenizer" => a.tokenizer = PathBuf::from(take(&mut i)?),
            "--prompt" => a.prompt = take(&mut i)?,
            "--negative" => a.negative = take(&mut i)?,
            "--output" => a.output = PathBuf::from(take(&mut i)?),
            "--width" => a.width = take(&mut i)?.parse()?,
            "--height" => a.height = take(&mut i)?.parse()?,
            "--steps" => a.steps = take(&mut i)?.parse()?,
            "--guidance" | "--cfg" => a.guidance = take(&mut i)?.parse()?,
            "--seed" => a.seed = take(&mut i)?.parse()?,
            "--alpha" => a.alpha = Some(take(&mut i)?.parse()?),
            "--rank" => a.rank = Some(take(&mut i)?.parse()?),
            "--multiplier" | "--strength" => a.multiplier = take(&mut i)?.parse()?,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("unknown arg: {other}");
            }
        }
        i += 1;
    }
    if a.lora.as_os_str().is_empty() {
        anyhow::bail!("--lora is required");
    }
    if a.prompt.is_empty() {
        anyhow::bail!("--prompt is required");
    }
    Ok(a)
}

fn print_usage() {
    eprintln!(
        "klein_lora_infer --base BASE --lora LORA --vae VAE --qwen3 QWEN3 \
         --tokenizer TOK --prompt PROMPT [--negative NEG] [--output PNG] \
         [--width W] [--height H] [--steps N] [--guidance G] [--seed S] \
         [--alpha A] [--rank R] [--multiplier M]"
    );
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let args = parse_args()?;
    let device = global_cuda_device();

    println!("============================================================");
    println!("Klein LoRA Inference (pure Rust)");
    println!("============================================================");
    println!("base : {}", args.base.display());
    println!("lora : {}", args.lora.display());
    println!("vae  : {}", args.vae.display());

    // ------------------------------------------------------------------
    // Stage 1: Text encoding FIRST (drops encoder before loading 18 GB DiT
    // base — staged loading is the only way Klein 9B fits 24 GB).
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Text Encoding ---");
    println!("  Prompt: {}", args.prompt);
    let t0 = Instant::now();
    let (pos_hidden, neg_hidden) = {
        let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let pos_formatted =
            format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", args.prompt);
        let neg_formatted =
            format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", args.negative);
        let pos_enc = tokenizer
            .encode(pos_formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;
        let neg_enc = tokenizer
            .encode(neg_formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;
        let mut pos_ids: Vec<i32> = pos_enc.get_ids().iter().map(|&id| id as i32).collect();
        let mut neg_ids: Vec<i32> = neg_enc.get_ids().iter().map(|&id| id as i32).collect();
        println!("  Pos tokens: {} (template-wrapped)", pos_ids.len());
        println!("  Neg tokens: {} (template-wrapped)", neg_ids.len());
        pos_ids.resize(TXT_PAD_LEN, PAD_ID);
        neg_ids.resize(TXT_PAD_LEN, PAD_ID);

        // Accept either a single safetensors file (Qwen3-4B for Klein 4B)
        // or a directory of shards (Qwen3-8B for Klein 9B).
        let enc_weights = if args.qwen3.is_dir() {
            let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&args.qwen3)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().and_then(|s| s.to_str()) == Some("safetensors")
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.starts_with("model-"))
                            .unwrap_or(false)
                })
                .collect();
            shards.sort();
            let mut all = std::collections::HashMap::new();
            for p in &shards {
                let s = flame_core::serialization::load_file(p, &device)?;
                all.extend(s);
            }
            println!("    encoder: {} shards from dir", shards.len());
            all
        } else {
            flame_core::serialization::load_file(&args.qwen3, &device)?
        };
        let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
        let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());
        let pos_h = encoder.encode(&pos_ids)?;
        let neg_h = encoder.encode(&neg_ids)?;
        println!("  pos: {:?}, neg: {:?}", pos_h.dims(), neg_h.dims());
        println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());
        drop(encoder);
        trim_cuda_mempool(0); // Release encoder weights back to driver before DiT load.
        (pos_h, neg_h)
    };
    let txt_seq_len = TXT_PAD_LEN;

    // ------------------------------------------------------------------
    // Stage 2: img_ids / txt_ids
    // ------------------------------------------------------------------
    let latent_h = args.height / 16;
    let latent_w = args.width / 16;
    let n_img = latent_h * latent_w;
    let mut img_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_data[idx * 4 + 1] = r as f32;
            img_data[idx * 4 + 2] = c as f32;
        }
    }
    let img_ids =
        Tensor::from_f32_to_bf16(img_data, Shape::from_dims(&[n_img, 4]), device.clone())?;
    let txt_ids =
        Tensor::zeros_dtype(Shape::from_dims(&[txt_seq_len, 4]), DType::BF16, device.clone())?;

    // ------------------------------------------------------------------
    // Stage 3: Load base + apply LoRA, then build the model.
    // Order matters: encoder is already dropped above, so we have the full
    // GPU available for base (~8 GB / 18 GB) + LoRA + merge intermediates.
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Load base + build LoRA stack, build model ---");
    let t0 = Instant::now();
    let base = flame_core::serialization::load_file(&args.base, &device)?;
    println!("  base   : {} tensors", base.len());

    // Build the runtime LoRA stack. `LoraStack::load` reads per-module
    // .alpha for every format, so --alpha / --rank CLI args are no longer
    // load-bearing. Klein-trainer split-QKV → fused-QKV mapping is in
    // map_prefix_klein_trainer.
    if args.alpha.is_some() || args.rank.is_some() {
        eprintln!(
            "  note: --alpha/--rank are ignored when LoRA file ships per-module .alpha tensors"
        );
    }
    let base_keys: std::collections::HashSet<String> = base.keys().cloned().collect();
    let lora_stack = LoraStack::load(
        args.lora.to_str().expect("lora path utf8"),
        &base_keys,
        args.multiplier,
        &device,
    )?;
    println!(
        "  lora   : {} target weight(s), multiplier={:.2}",
        lora_stack.target_count(),
        args.multiplier
    );
    trim_cuda_mempool(0);

    let mut model = KleinTransformer::from_weights(base)?;
    model.set_lora(Arc::new(lora_stack));
    println!("  Config: {:?}", model.config());
    println!("  Built in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 4: Noise + sigma schedule
    // ------------------------------------------------------------------
    let numel = 128 * latent_h * latent_w;
    let noise_data = box_muller_noise(numel, args.seed);
    let noise_spatial = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        device.clone(),
    )?;
    let noise = noise_spatial
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, latent_h * latent_w, 128])?;
    let timesteps = get_schedule(args.steps, n_img);
    println!(
        "  Schedule: {} values, t[0]={:.4}, t[-2]={:.4}",
        timesteps.len(),
        timesteps[0],
        timesteps[args.steps - 1]
    );

    // ------------------------------------------------------------------
    // Stage 5: Denoise (Euler + true CFG)
    // ------------------------------------------------------------------
    println!(
        "\n--- Stage 5: Denoise ({} steps, guidance={}) ---",
        args.steps, args.guidance
    );
    let t0 = Instant::now();
    let denoised = euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;
            let pred_cond = model.forward(x, &pos_hidden, &t_vec, &img_ids, &txt_ids)?;
            let pred_uncond = model.forward(x, &neg_hidden, &t_vec, &img_ids, &txt_ids)?;
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(args.guidance)?)
        },
        noise,
        &timesteps,
    )?;
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step)",
        dt,
        dt / args.steps as f32
    );

    // ------------------------------------------------------------------
    // Stage 6: VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 6: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);
    trim_cuda_mempool(0); // CRITICAL at 1024²: release ~8 GB of DiT weights from
                          // the cuda cache before VAE load, otherwise VAE alloc OOMs.
    let latents = denoised
        .reshape(&[1, latent_h, latent_w, 128])?
        .permute(&[0, 3, 1, 2])?;
    let vae_weights = flame_core::serialization::load_file(&args.vae, &device)?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    let rgb = vae.decode(&latents)?;
    println!("  Decoded: {:?} in {:.1}s", rgb.dims(), t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 7: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let dims = rgb_f32.dims();
    let (out_h, out_w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * out_w + x) * 3 + c] = val;
            }
        }
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to build image buffer"))?
        .save(&args.output)?;

    println!(
        "\n============================================================\nIMAGE SAVED: {}\nTotal time: {:.1}s\n============================================================",
        args.output.display(),
        t_total.elapsed().as_secs_f32()
    );
    Ok(())
}
