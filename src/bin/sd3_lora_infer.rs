//! SD3.5 Medium LoRA inference — pure Rust.
//!
//! Same end-to-end pipeline as `sd3_medium_infer` but merges a sd3-trainer
//! LoRA into the resident weights before constructing SD3MMDiT.
//!
//! Usage:
//!   sd3_lora_infer --lora /path/to/sd3_lora.safetensors \
//!                  --prompt "..." \
//!                  [--lora-scale 1.0] [--out /path.png]
//!
//! sd3-trainer LoRA format (from `flame-diffusion/sd3-trainer/src/model.rs`):
//!   joint_blocks.<i>.x_block.attn.qkv.lora_A    — [rank, in_features]
//!   joint_blocks.<i>.x_block.attn.qkv.lora_B    — [out_features, rank]
//!   joint_blocks.<i>.x_block.attn.proj.lora_*
//!   joint_blocks.<i>.x_block.mlp.fc{1,2}.lora_*
//!   joint_blocks.<i>.context_block.attn.qkv.lora_*       (depth-1 blocks only)
//!   joint_blocks.<i>.context_block.attn.proj.lora_*
//!   joint_blocks.<i>.context_block.mlp.fc{1,2}.lora_*
//!   joint_blocks.<i>.x_block.attn2.{qkv,proj}.lora_*     (dual-attention blocks only)
//!
//! Each pair maps to a base weight by appending `.weight`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::clip_encoder::{ClipConfig, ClipEncoder};
use inference_flame::models::sd3_mmdit::{load_sd3_all, SD3MMDiT};
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

const CLIP_L_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const CLIP_L_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json";
const CLIP_G_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors";
const CLIP_G_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json";
const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const DEFAULT_MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors";
const DEFAULT_OUTPUT_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/sd3_lora_rust.png";
const DEFAULT_PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";

const NUM_STEPS: usize = 28;
const CFG_SCALE: f32 = 4.5;
const SHIFT: f32 = 3.0;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;
const CLIP_SEQ_LEN: usize = 77;
const T5_SEQ_LEN: usize = 256;

const VAE_IN_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;
const VAE_SHIFT: f32 = 0.0609;

/// Minimal --flag value parser; no clap dependency.
struct Args {
    lora: PathBuf,
    prompt: Option<String>,
    neg_prompt: String,
    lora_scale: f32,
    model: PathBuf,
    out: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
}

fn parse_args() -> anyhow::Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        lora: PathBuf::new(),
        prompt: None,
        neg_prompt: String::new(),
        lora_scale: 1.0,
        model: PathBuf::from(DEFAULT_MODEL_PATH),
        out: PathBuf::from(DEFAULT_OUTPUT_PATH),
        width: WIDTH,
        height: HEIGHT,
        steps: NUM_STEPS,
        cfg: CFG_SCALE,
        seed: SEED,
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
            "--prompt" => { a.prompt = Some(take()?.to_string()); i += 2; }
            "--neg-prompt" => { a.neg_prompt = take()?.to_string(); i += 2; }
            "--lora-scale" => { a.lora_scale = take()?.parse()?; i += 2; }
            "--model" => { a.model = PathBuf::from(take()?); i += 2; }
            "--out" => { a.out = PathBuf::from(take()?); i += 2; }
            "--width" => { a.width = take()?.parse()?; i += 2; }
            "--height" => { a.height = take()?.parse()?; i += 2; }
            "--steps" => { a.steps = take()?.parse()?; i += 2; }
            "--cfg" => { a.cfg = take()?.parse()?; i += 2; }
            "--seed" => { a.seed = take()?.parse()?; i += 2; }
            "-h" | "--help" => {
                println!("Usage: sd3_lora_infer --lora <path> [--prompt ...] [--lora-scale 1.0]");
                println!("                       [--model <sd3_5_medium.safetensors>] [--out <png>]");
                println!("                       [--width 1024] [--height 1024] [--steps 28] [--cfg 4.5] [--seed 42]");
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

// ---------------------------------------------------------------------------
// Schedule (flow-matching shifted Euler — same as sd3_medium_infer)
// ---------------------------------------------------------------------------

fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Tokenizers
// ---------------------------------------------------------------------------

fn tokenize_clip(prompt: &str, tokenizer_path: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(tokenizer_path) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("clip tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(CLIP_SEQ_LEN);
            while ids.len() < CLIP_SEQ_LEN {
                ids.push(49407);
            }
            ids
        }
        Err(e) => {
            eprintln!("[sd3] CLIP tokenizer failed: {e}; using BOS+EOS fallback");
            let mut ids = vec![49406i32, 49407];
            ids.resize(CLIP_SEQ_LEN, 49407);
            ids
        }
    }
}

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.push(1);
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN {
                ids.push(0);
            }
            ids
        }
        Err(e) => {
            eprintln!("[sd3] T5 tokenizer failed: {e}; using EOS fallback");
            let mut ids = vec![1i32];
            ids.resize(T5_SEQ_LEN, 0);
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Text encoding (cond + uncond, staged loading — same as sd3_medium_infer)
// ---------------------------------------------------------------------------

fn encode_text_pair(
    prompt: &str,
    neg_prompt: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
    fn load_clip_weights(
        path: &str,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> anyhow::Result<HashMap<String, Tensor>> {
        let raw = flame_core::serialization::load_file(std::path::Path::new(path), device)?;
        let weights = raw
            .into_iter()
            .map(|(k, v)| {
                let t = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
                Ok::<_, flame_core::Error>((k, t))
            })
            .collect::<Result<_, _>>()?;
        Ok(weights)
    }

    println!("  CLIP-L...");
    let t0 = Instant::now();
    let (clip_l_hidden, clip_l_pooled, clip_l_hidden_u, clip_l_pooled_u) = {
        let weights = load_clip_weights(CLIP_L_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::default(), device.clone());
        let (hc, pc) = clip.encode_sd3(&tokenize_clip(prompt, CLIP_L_TOKENIZER))?;
        let (hu, pu) = clip.encode_sd3(&tokenize_clip(neg_prompt, CLIP_L_TOKENIZER))?;
        (hc, pc, hu, pu)
    };
    println!("    {:.1}s", t0.elapsed().as_secs_f32());

    println!("  CLIP-G...");
    let t0 = Instant::now();
    let (clip_g_hidden, clip_g_pooled, clip_g_hidden_u, clip_g_pooled_u) = {
        let weights = load_clip_weights(CLIP_G_PATH, device)?;
        let clip = ClipEncoder::new(weights, ClipConfig::clip_g(), device.clone());
        let (hc, pc) = clip.encode_sd3(&tokenize_clip(prompt, CLIP_G_TOKENIZER))?;
        let (hu, pu) = clip.encode_sd3(&tokenize_clip(neg_prompt, CLIP_G_TOKENIZER))?;
        (hc, pc, hu, pu)
    };
    println!("    {:.1}s", t0.elapsed().as_secs_f32());

    println!("  T5-XXL...");
    let t0 = Instant::now();
    let (t5_hidden, t5_hidden_u) = {
        let mut t5 = T5Encoder::load(T5_PATH, device)?;
        let hc = t5.encode(&tokenize_t5(prompt))?;
        let hu = t5.encode(&tokenize_t5(neg_prompt))?;
        (hc, hu)
    };
    let t5_hidden = t5_hidden.narrow(1, 0, t5_hidden.dims()[1].min(T5_SEQ_LEN))?;
    let t5_hidden_u = t5_hidden_u.narrow(1, 0, t5_hidden_u.dims()[1].min(T5_SEQ_LEN))?;
    println!("    {:.1}s", t0.elapsed().as_secs_f32());

    let cl_pad = zero_pad_last_dim(&clip_l_hidden, 4096)?;
    let cg_pad = zero_pad_last_dim(&clip_g_hidden, 4096)?;
    let context = Tensor::cat(&[&cl_pad, &cg_pad, &t5_hidden], 1)?;
    let pooled = Tensor::cat(&[&clip_l_pooled, &clip_g_pooled], 1)?;

    let cl_pad_u = zero_pad_last_dim(&clip_l_hidden_u, 4096)?;
    let cg_pad_u = zero_pad_last_dim(&clip_g_hidden_u, 4096)?;
    let context_u = Tensor::cat(&[&cl_pad_u, &cg_pad_u, &t5_hidden_u], 1)?;
    let pooled_u = Tensor::cat(&[&clip_l_pooled_u, &clip_g_pooled_u], 1)?;

    Ok((context, pooled, context_u, pooled_u))
}

fn zero_pad_last_dim(x: &Tensor, target_dim: usize) -> anyhow::Result<Tensor> {
    let dims = x.dims();
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    if c >= target_dim {
        return Ok(x.clone());
    }
    let pad = Tensor::zeros_dtype(
        Shape::from_dims(&[b, n, target_dim - c]),
        DType::BF16,
        x.device().clone(),
    )?;
    Ok(Tensor::cat(&[x, &pad], 2)?)
}

// ---------------------------------------------------------------------------
// SD3 LoRA merge
// ---------------------------------------------------------------------------

/// Merge sd3-trainer LoRAs into a base-weight HashMap in place. Each pair
/// `<base_path>.lora_A` + `<base_path>.lora_B` produces a delta
///   delta = scale * (B @ A)         shape: [out, in]
/// added to the base weight at `<base_path>.weight`.
///
/// Returns (modules_merged, modules_skipped, errors).
fn merge_sd3_lora(
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
        } else {
            // Silently skip unknown keys — older trainers may have aux state.
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

        // Compute delta in BF16 then add to base. lora_b @ lora_a with
        // shapes [out, rank] @ [rank, in] = [out, in] matches base.
        let a_bf16 = if lora_a.dtype() == DType::BF16 {
            lora_a
        } else {
            lora_a.to_dtype(DType::BF16)?
        };
        let b_bf16 = if lora_b.dtype() == DType::BF16 {
            lora_b
        } else {
            lora_b.to_dtype(DType::BF16)?
        };
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args()?;
    let t_total = Instant::now();
    let device = global_cuda_device();
    let prompt = args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("============================================================");
    println!("SD3.5 Medium LoRA — Pure Rust");
    println!("  Base:   {}", args.model.display());
    println!("  LoRA:   {}", args.lora.display());
    println!("  Scale:  {}", args.lora_scale);
    println!("  Prompt: {:?}", prompt);
    println!("  {}x{}, {} steps, CFG {}, shift {}", args.width, args.height, args.steps, args.cfg, SHIFT);
    println!("============================================================");

    // Stage 1: Text encoding
    println!("\n--- Stage 1: Text Encoding ---");
    let t0 = Instant::now();
    let (context, pooled, context_uncond, pooled_uncond) =
        encode_text_pair(&prompt, &args.neg_prompt, &device)?;
    println!("  Total encode: {:.1}s", t0.elapsed().as_secs_f32());

    // Stage 2: Load base + merge LoRA
    println!("\n--- Stage 2: Load SD3.5 Medium + Merge LoRA ---");
    let t0 = Instant::now();
    let mut resident = load_sd3_all(args.model.to_str().unwrap(), &device)?;
    println!("  base: {} keys", resident.len());

    let lora = flame_core::serialization::load_file(&args.lora, &device)?;
    println!("  lora: {} keys", lora.len());
    let (merged, skipped) = merge_sd3_lora(&mut resident, lora, args.lora_scale)?;
    println!("  merge: {} modules merged, {} skipped", merged, skipped);
    if merged == 0 {
        anyhow::bail!("0 LoRA modules merged — check that the LoRA matches sd3-trainer naming \
                       (joint_blocks.<i>.x_block.attn.qkv.lora_{{A,B}})");
    }

    let mut model = SD3MMDiT::new(args.model.to_str().unwrap().to_string(), resident, device.clone());
    println!(
        "  depth={}, hidden={}, heads={}, dual_attn={} ({:.1}s)",
        model.config.depth,
        model.config.hidden_size,
        model.config.num_heads,
        model.config.use_dual_attention,
        t0.elapsed().as_secs_f32(),
    );

    // Stage 3: Denoise
    let latent_h = args.height / 8;
    let latent_w = args.width / 8;
    let numel = VAE_IN_CHANNELS * latent_h * latent_w;
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
    let noise = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, VAE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    let timesteps = build_schedule(args.steps, SHIFT);

    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", args.steps, args.cfg);
    let t0 = Instant::now();
    let cfg = args.cfg;
    let denoised = inference_flame::sampling::klein_sampling::euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr * 1000.0],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;
            let pred_cond = model.forward(x, &t_vec, &context, &pooled)?;
            let pred_uncond = model.forward(x, &t_vec, &context_uncond, &pooled_uncond)?;
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(cfg)?)
        },
        noise,
        &timesteps,
    )?;
    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / args.steps as f32);

    drop(model);
    drop(context);
    drop(pooled);
    drop(context_uncond);
    drop(pooled_uncond);

    // Stage 4: VAE decode
    println!("\n--- Stage 4: VAE Decode ---");
    let t0 = Instant::now();
    let vae = LdmVAEDecoder::from_safetensors(
        args.model.to_str().unwrap(), VAE_IN_CHANNELS, VAE_SCALE, VAE_SHIFT, &device,
    )?;
    let rgb = vae.decode(&denoised)?;
    drop(denoised);
    drop(vae);
    println!("  {:.1}s", t0.elapsed().as_secs_f32());

    // Stage 5: Save PNG
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec_f32()?;
    let dims = rgb_f32.dims();
    let (out_h, out_w) = (dims[2], dims[3]);

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
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image creation failed"))?
        .save(&args.out)?;

    println!("\n============================================================");
    println!("Saved: {} ({:.1}s total)", args.out.display(), t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
