//! ERNIE-Image LoRA inference — pure Rust.
//!
//! Same pipeline as `ernie_image_infer` but merges an ernie-trainer LoRA
//! into the resident DiT weights before constructing ErnieImageModel.
//!
//! Usage:
//!   ernie_lora_infer --lora /path/to/lora_step_NNNNNN.safetensors \
//!                    --prompt "..." \
//!                    [--lora-scale 1.0] [--out /path.png]
//!                    [--width 256] [--height 256]
//!                    [--steps 50] [--cfg 4.0] [--seed 42]
//!
//! ernie-trainer LoRA save format (ernie-trainer/src/model.rs:save_lora_weights):
//!   layers.<i>.self_attention.to_q.lora_{A,B}
//!   layers.<i>.self_attention.to_k.lora_{A,B}
//!   layers.<i>.self_attention.to_v.lora_{A,B}
//!   layers.<i>.self_attention.to_out.0.lora_{A,B}
//!
//! Each pair maps to base `<prefix>.weight` by appending `.weight`. The
//! base weight is stored as [out, in] on disk — delta = (B @ A) is also
//! [out, in], added in BF16. ErnieImageModel::load transposes after merge.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::ernie_image::{ErnieImageConfig, ErnieImageModel};
use inference_flame::models::mistral3b_encoder::Mistral3bEncoder;
use inference_flame::sampling::ernie_sampling::{ernie_euler_step, ernie_schedule, sigma_to_timestep};
use inference_flame::vae::klein_vae::KleinVaeDecoder;

const TRANSFORMER_DIR: &str = "/home/alex/models/ERNIE-Image/transformer";
const TEXT_ENCODER: &str = "/home/alex/models/ERNIE-Image/text_encoder/model.safetensors";
const TOKENIZER: &str = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json";
const VAE_PATH: &str = "/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors";

const DEFAULT_OUTPUT: &str = "/home/alex/EriDiffusion/inference-flame/output/ernie_lora.png";
const DEFAULT_PROMPT: &str = "a photograph of a person, soft natural light";

struct Args {
    lora: PathBuf,
    lora_scale: f32,
    prompt: String,
    out: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    max_text_len: usize,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        lora: PathBuf::new(),
        lora_scale: 1.0,
        prompt: DEFAULT_PROMPT.to_string(),
        out: PathBuf::from(DEFAULT_OUTPUT),
        width: 256,
        height: 256,
        steps: 50,
        cfg: 4.0,
        seed: 42,
        max_text_len: 128,
    };
    let mut i = 0;
    while i < argv.len() {
        let take = || -> anyhow::Result<&str> {
            argv.get(i + 1).map(String::as_str).ok_or_else(|| {
                anyhow::anyhow!("flag {} expects a value", argv[i])
            })
        };
        match argv[i].as_str() {
            "--lora" => { a.lora = PathBuf::from(take()?); i += 2; }
            "--lora-scale" => { a.lora_scale = take()?.parse()?; i += 2; }
            "--prompt" => { a.prompt = take()?.to_string(); i += 2; }
            "--out" => { a.out = PathBuf::from(take()?); i += 2; }
            "--width" => { a.width = take()?.parse()?; i += 2; }
            "--height" => { a.height = take()?.parse()?; i += 2; }
            "--steps" => { a.steps = take()?.parse()?; i += 2; }
            "--cfg" => { a.cfg = take()?.parse()?; i += 2; }
            "--seed" => { a.seed = take()?.parse()?; i += 2; }
            "--max-text-len" => { a.max_text_len = take()?.parse()?; i += 2; }
            "-h" | "--help" => {
                println!("Usage: ernie_lora_infer --lora <path> [--prompt ...] [--lora-scale 1.0]");
                println!("                        [--width 256] [--height 256]");
                println!("                        [--steps 50] [--cfg 4.0] [--seed 42] [--out <png>]");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    if a.lora.as_os_str().is_empty() {
        anyhow::bail!("--lora is required");
    }
    Ok(a)
}

/// Merge ernie-trainer LoRAs into a base-weight HashMap in place.
fn merge_ernie_lora(
    base_weights: &mut HashMap<String, Tensor>,
    lora: HashMap<String, Tensor>,
    scale: f32,
) -> anyhow::Result<(usize, usize)> {
    use std::collections::BTreeMap;

    // Group by base prefix.
    let mut groups: BTreeMap<String, (Option<Tensor>, Option<Tensor>)> = BTreeMap::new();
    for (key, val) in lora {
        if let Some(prefix) = key.strip_suffix(".lora_A") {
            groups.entry(prefix.to_string()).or_default().0 = Some(val);
        } else if let Some(prefix) = key.strip_suffix(".lora_B") {
            groups.entry(prefix.to_string()).or_default().1 = Some(val);
        }
    }

    let mut merged = 0usize;
    let mut skipped = 0usize;
    for (prefix, (lora_a, lora_b)) in groups {
        let (lora_a, lora_b) = match (lora_a, lora_b) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                eprintln!("[lora] {prefix}: missing A or B half — skipping");
                skipped += 1;
                continue;
            }
        };
        let base_key = format!("{prefix}.weight");
        let Some(base) = base_weights.get(&base_key) else {
            eprintln!("[lora] {prefix}: base key {base_key} not in model — skipping");
            skipped += 1;
            continue;
        };

        let a_bf16 = if lora_a.dtype() == DType::BF16 { lora_a } else { lora_a.to_dtype(DType::BF16)? };
        let b_bf16 = if lora_b.dtype() == DType::BF16 { lora_b } else { lora_b.to_dtype(DType::BF16)? };
        let delta = b_bf16.matmul(&a_bf16)?;
        let delta = if (scale - 1.0).abs() > f32::EPSILON {
            delta.mul_scalar(scale)?
        } else {
            delta
        };
        let merged_w = base.add(&delta)?;
        base_weights.insert(base_key, merged_w);
        merged += 1;
    }

    Ok((merged, skipped))
}

fn main() {
    env_logger::init();
    if let Err(e) = run() {
        eprintln!("ernie_lora_infer failed: {e:?}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_args()?;
    let device = global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let t_total = Instant::now();

    let latent_h = args.height / 16;
    let latent_w = args.width / 16;

    println!("=== ERNIE-Image LoRA infer ===");
    println!("  prompt: {:?}", args.prompt);
    println!("  lora: {} (scale={})", args.lora.display(), args.lora_scale);
    println!("  size: {}x{}, latent: {}x{}, steps: {}, cfg: {}, seed: {}",
        args.width, args.height, latent_w, latent_h, args.steps, args.cfg, args.seed);

    // Stage 1: Mistral-3 text encode (cond + uncond), then drop encoder.
    println!("\n[1/4] Text encoding (Mistral-3 3B)...");
    let t0 = Instant::now();
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let cond_ids: Vec<i32> = tokenizer.encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids().iter().map(|&id| id as i32).collect();
    let cond_len = cond_ids.len();
    let uncond_ids: Vec<i32> = tokenizer.encode("", true)
        .map_err(|e| anyhow::anyhow!("tokenize empty: {e}"))?
        .get_ids().iter().map(|&id| id as i32).collect();
    let uncond_len = uncond_ids.len();
    println!("  tokenized: cond={cond_len} uncond={uncond_len}");

    let (cond_embeds, uncond_embeds) = {
        let encoder = Mistral3bEncoder::load(TEXT_ENCODER, &device)?;
        let cond = encoder.encode(&cond_ids, args.max_text_len)?;
        let uncond = encoder.encode(&uncond_ids, args.max_text_len)?;
        (cond, uncond)
    };
    println!("  text encoding done in {:.1}s", t0.elapsed().as_secs_f32());

    // Stage 2: Load base + merge LoRA.
    println!("\n[2/4] Loading ERNIE-Image DiT + merging LoRA...");
    let t1 = Instant::now();
    let shard_paths: Vec<String> = {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(TRANSFORMER_DIR)? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                paths.push(p.to_string_lossy().into_owned());
            }
        }
        paths.sort();
        paths
    };
    let mut all_weights = HashMap::new();
    for path in &shard_paths {
        let partial = flame_core::serialization::load_file(path, &device)?;
        for (k, v) in partial { all_weights.insert(k, v); }
    }
    println!("  base: {} keys", all_weights.len());

    let lora = flame_core::serialization::load_file(&args.lora, &device)?;
    println!("  lora: {} keys", lora.len());
    let (merged, skipped) = merge_ernie_lora(&mut all_weights, lora, args.lora_scale)?;
    println!("  merge: {} modules merged, {} skipped", merged, skipped);
    if merged == 0 {
        anyhow::bail!("0 LoRA modules merged — check that the LoRA matches ernie-trainer naming");
    }

    let config = ErnieImageConfig::default();
    let model = ErnieImageModel::load(all_weights, config.clone())?;
    println!("  DiT loaded in {:.1}s ({} blocks)", t1.elapsed().as_secs_f32(), config.num_layers);

    // Stage 3: Denoise with sequential CFG.
    println!("\n[3/4] Denoising ({} steps, sequential CFG)...", args.steps);
    let t2 = Instant::now();
    let sigmas = ernie_schedule(args.steps);

    let noise = {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        let n = 128 * latent_h * latent_w;
        let mut data = vec![0.0f32; n];
        for v in &mut data {
            let u1: f32 = rand::Rng::gen_range(&mut rng, 1e-7f32..1.0);
            let u2: f32 = rand::Rng::gen_range(&mut rng, 0.0f32..std::f32::consts::TAU);
            *v = (-2.0 * u1.ln()).sqrt() * u2.cos();
        }
        Tensor::from_vec(data, Shape::from_dims(&[1, 128, latent_h, latent_w]), device.clone())?
            .to_dtype(DType::BF16)?
    };
    let mut latent = noise;

    let cond_3d = if cond_embeds.rank() == 2 { cond_embeds.unsqueeze(0)? } else { cond_embeds.clone() };
    let uncond_3d = if uncond_embeds.rank() == 2 { uncond_embeds.unsqueeze(0)? } else { uncond_embeds.clone() };
    let cond_trim = cond_3d.narrow(1, 0, cond_len.min(args.max_text_len))?;
    let uncond_trim = uncond_3d.narrow(1, 0, uncond_len.min(args.max_text_len))?;
    let cond_lens = vec![cond_len.min(args.max_text_len)];
    let uncond_lens = vec![uncond_len.min(args.max_text_len)];

    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

        let pred = if args.cfg > 1.0 {
            let pred_cond = model.forward(&latent, &t_tensor, &cond_trim, &cond_lens)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            let pred_uncond = model.forward(&latent, &t_tensor, &uncond_trim, &uncond_lens)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(args.cfg)?)?
        } else {
            let p = model.forward(&latent, &t_tensor, &cond_trim, &cond_lens)?;
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::device::trim_cuda_mempool(0);
            p
        };
        latent = ernie_euler_step(&latent, &pred, sigma, sigma_next)?;

        if step % 5 == 0 || step == args.steps - 1 {
            println!("  step {}/{} sigma={sigma:.4} t={t:.1} ({:.1}s)",
                step + 1, args.steps, t2.elapsed().as_secs_f32());
        }
    }
    println!("  denoising done in {:.1}s ({:.1}s/step)",
        t2.elapsed().as_secs_f32(), t2.elapsed().as_secs_f32() / args.steps as f32);

    // Stage 4: Drop DiT, VAE decode.
    println!("\n[4/4] VAE decode...");
    let t3 = Instant::now();
    drop(model);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);

    let vae_weights = flame_core::serialization::load_file(VAE_PATH, &device)?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    drop(vae_weights);

    let decoded = vae.decode(&latent)?;
    println!("  VAE done in {:.1}s", t3.elapsed().as_secs_f32());

    // Save PNG.
    let img_data = decoded.to_vec_f32()?;
    let dims = decoded.shape().dims();
    let (c, h, w) = (dims[1], dims[2], dims[3]);
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3.min(c) {
                let v = img_data[ch * h * w + y * w + x];
                let v = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                rgb[(y * w + x) * 3 + ch] = v;
            }
        }
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow::anyhow!("failed to create image"))?
        .save(&args.out)?;

    println!("\n=== Done ===  output: {} ({:.1}s total)",
        args.out.display(), t_total.elapsed().as_secs_f32());
    Ok(())
}
