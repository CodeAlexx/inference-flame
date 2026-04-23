//! Chroma end-to-end inference — Turbo Flame Phase 2 (VMM-backed
//! double-buffered block loader).
//!
//! Same pipeline as `chroma_infer` but the per-block weight staging goes
//! through `inference_flame::turbo::TurboBlockLoader` instead of the
//! BlockOffloader.
//!
//! Behind feature `turbo`. On a non-VMM-capable device this binary logs a
//! hint and exits with code 2 — no silent fallback.

#![cfg(feature = "turbo")]

use std::sync::Arc;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::chroma_dit::ChromaDit;
use inference_flame::models::t5_encoder::T5Encoder;
use inference_flame::sampling::flux1_sampling::{
    get_schedule, pack_latent, unpack_latent,
};
use inference_flame::turbo::{TurboBlockLoader, VmmArena, VmmError};
use inference_flame::vae::ldm_decoder::LdmVAEDecoder;

// ---------------------------------------------------------------------------
// Paths / knobs (mirror of `chroma_infer.rs`)
// ---------------------------------------------------------------------------

const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const CHROMA_DIT_SHARDS: &[&str] = &[
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
];

const CHROMA_VAE: &str = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors";

const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/chroma_rust_turbo.png";

const DEFAULT_PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, cinematic lighting, highly detailed";
const DEFAULT_NEGATIVE: &str = "";

const SEED: u64 = 42;
const HEIGHT: usize = 1024;
const WIDTH: usize = 1024;
const NUM_STEPS: usize = 40;
const GUIDANCE_SCALE: f32 = 4.0;

const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

const T5_SEQ_LEN: usize = 512;

// Chroma config (keep in sync with chroma_dit.rs::ChromaConfig::default()).
const CHROMA_NUM_DOUBLE: usize = 19;
const CHROMA_NUM_SINGLE: usize = 38;

fn main() -> anyhow::Result<()> {
    eprintln!("WARNING: chroma_infer_turbo loads T5 + DiT in the same process.");
    eprintln!("         T5 (~10 GB) is dropped before the turbo loader spins up.");
    eprintln!();
    if std::env::var("CHROMA_INFER_FORCE").as_deref() != Ok("1") {
        eprintln!("Set CHROMA_INFER_FORCE=1 to acknowledge the OOM caveat.");
        std::process::exit(2);
    }
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let prompt = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let negative = std::env::args().nth(2).unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());

    println!("=== Chroma — Turbo Flame Phase 2 (VMM) ===");
    println!("Prompt:    {:?}", prompt);
    println!("Negative:  {:?}", negative);
    println!("Size:      {}x{}, steps={}, guidance={}", WIDTH, HEIGHT, NUM_STEPS, GUIDANCE_SCALE);
    println!("Seed:      {}", SEED);
    println!();

    // ------------------------------------------------------------------
    // Stage 1: T5-XXL encode (cond + uncond, both DROPPED after)
    // ------------------------------------------------------------------
    println!("--- Stage 1: T5-XXL encode (cond + uncond) ---");
    let t0 = Instant::now();
    let (t5_cond_hidden, t5_uncond_hidden) = {
        let mut t5 = T5Encoder::load(T5_PATH, &device)?;
        println!("  Loaded T5 in {:.1}s", t0.elapsed().as_secs_f32());

        let cond_tokens = tokenize_t5(&prompt);
        let cond_hidden = t5.encode(&cond_tokens)?;
        println!("  cond hidden: {:?}", cond_hidden.shape().dims());

        let uncond_tokens = tokenize_t5(&negative);
        let uncond_hidden = t5.encode(&uncond_tokens)?;
        println!("  uncond hidden: {:?}", uncond_hidden.shape().dims());

        println!("  T5 done in {:.1}s", t0.elapsed().as_secs_f32());
        (cond_hidden, uncond_hidden)
    };
    println!("  T5 weights evicted");

    // T5-XXL weights are in flame-core's pool free lists (~10 GB). Release
    // to driver before loading the DiT — same pattern as inference_ui's
    // klein.rs worker.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!();

    // ------------------------------------------------------------------
    // Stage 2: Build VMM arena + TurboBlockLoader for Chroma
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load Chroma DiT (Turbo / VMM) ---");
    let t0 = Instant::now();

    // NonBlocking copy stream: H2D prefetch runs in parallel with compute on
    // device.cu_stream(). Same construction as klein9b_infer_turbo.
    let copy_stream = Arc::new(
        device
            .fork_default_stream()
            .map_err(|e| anyhow::anyhow!("create copy stream: {e:?}"))?,
    );

    // Build the model's shared weights on GPU via the existing offloaded
    // path; we only swap the *block* staging.
    let dit = ChromaDit::load(CHROMA_DIT_SHARDS, &device)?;

    let arena = match VmmArena::new_for_chroma(device.clone(), copy_stream.clone()) {
        Ok(a) => Arc::new(a),
        Err(VmmError::Unsupported) => {
            log::error!(
                "VMM unsupported on device {}, re-run with chroma_infer (non-turbo)",
                device.ordinal()
            );
            std::process::exit(2);
        }
        Err(e) => {
            log::error!("VMM arena init failed: {e}");
            std::process::exit(2);
        }
    };

    // Block prefixes in canonical order: doubles 0..19 then singles 0..38.
    // Chroma's BlockFacilitator classifies the same way (transformer_blocks.{i}
    // → block i, single_transformer_blocks.{i} → block num_double + i), so the
    // turbo loader's per-block layout matches.
    let mut block_prefixes: Vec<String> = Vec::with_capacity(CHROMA_NUM_DOUBLE + CHROMA_NUM_SINGLE);
    for i in 0..CHROMA_NUM_DOUBLE {
        block_prefixes.push(format!("transformer_blocks.{i}."));
    }
    for i in 0..CHROMA_NUM_SINGLE {
        block_prefixes.push(format!("single_transformer_blocks.{i}."));
    }

    // TurboBlockLoader works on a single safetensors file (it mmaps + parses
    // header). Chroma's diffusers checkpoint is sharded into two files, so
    // pointing at `CHROMA_DIT_SHARDS[0]` silently loads only half of each
    // block's tensors — the `Missing: single_transformer_blocks.3.attn.to_q.weight`
    // symptom. Default to the merged single-file BF16 checkpoint at
    // `.serenity/models/checkpoints/chroma1_hd_bf16.safetensors`; the user
    // can override via `CHROMA_TURBO_SAFETENSORS`.
    const CHROMA_TURBO_DEFAULT_MERGED: &str =
        "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors";
    let model_path = std::env::var("CHROMA_TURBO_SAFETENSORS")
        .unwrap_or_else(|_| CHROMA_TURBO_DEFAULT_MERGED.to_string());

    let mut loader = TurboBlockLoader::new(
        model_path.clone(),
        device.clone(),
        arena.clone(),
        block_prefixes,
    )
    .map_err(|e| anyhow::anyhow!(
        "TurboBlockLoader: {e}\n\
         hint: Phase 2 turbo expects a single-file Chroma safetensors. Set \
         CHROMA_TURBO_SAFETENSORS=/path/to/Chroma1-HD.safetensors (the \
         single-file release) and re-run."
    ))?;

    println!(
        "  loader: {} blocks, {:.1} MiB pinned host  ({:.1}s)",
        loader.block_count(),
        loader.pinned_bytes() as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f32(),
    );

    // ------------------------------------------------------------------
    // Stage 3: Build noise, pack, run CFG denoise loop
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Denoise ({} steps, CFG={}) ---", NUM_STEPS, GUIDANCE_SCALE);

    let latent_h = 2 * ((HEIGHT + 15) / 16);
    let latent_w = 2 * ((WIDTH + 15) / 16);

    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
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
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    let (img_packed, img_ids) = pack_latent(&noise_nchw, &device)?;
    drop(noise_nchw);

    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[T5_SEQ_LEN, 3]),
        DType::BF16,
        device.clone(),
    )?;

    let n_img = img_packed.shape().dims()[1];
    let timesteps = get_schedule(NUM_STEPS, n_img, 0.5, 1.15, true);
    println!("  Schedule: {} steps, t[0]={:.4}, t[-1]={:.4}",
        timesteps.len() - 1, timesteps[0], timesteps[NUM_STEPS]);

    let t0 = Instant::now();
    let mut x = img_packed;
    for step in 0..NUM_STEPS {
        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        let next_x = {
            let t_vec = Tensor::from_vec(
                vec![t_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;

            // Turbo path: dispatch through forward_with_turbo with the shared
            // TurboBlockLoader. Both cond and uncond reuse the same loader
            // instance — the loader's two-slot rotation handles the back-to-back
            // forwards (cond ends on slot N, uncond starts by re-fetching block 0).
            let cond_pred = dit.forward_with_turbo(
                &x, &t5_cond_hidden, &t_vec, &img_ids, &txt_ids, &mut loader,
            )?;
            let uncond_pred = dit.forward_with_turbo(
                &x, &t5_uncond_hidden, &t_vec, &img_ids, &txt_ids, &mut loader,
            )?;

            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(GUIDANCE_SCALE)?;
            let pred = uncond_pred.add(&scaled)?;

            let step_tensor = pred.mul_scalar(dt)?;
            x.add(&step_tensor)?
        };
        x = next_x;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == NUM_STEPS {
            log::info!("[Chroma turbo] step {}/{} t={:.4}", step + 1, NUM_STEPS, t_curr);
        }
    }
    let dt_denoise = t0.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step)",
        dt_denoise,
        dt_denoise / NUM_STEPS as f32,
    );

    drop(loader);
    drop(dit);
    drop(t5_cond_hidden);
    drop(t5_uncond_hidden);

    // DiT shared weights + VMM slot physical back in the pool. Release to
    // driver before VAE conv workspace allocations.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);

    // ------------------------------------------------------------------
    // Stage 4: Unpack + VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: VAE decode ---");
    let t0 = Instant::now();

    let latent = unpack_latent(&x, HEIGHT, WIDTH)?;
    drop(x);

    let vae = LdmVAEDecoder::from_safetensors(
        CHROMA_VAE,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &device,
    )?;

    let rgb = vae.decode(&latent)?;
    drop(latent);
    drop(vae);
    println!("  VAE stage in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 5: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec_f32()?;
    let dims = rgb_f32.shape().dims().to_vec();
    let (out_c, out_h, out_w) = (dims[1], dims[2], dims[3]);
    assert_eq!(out_c, 3, "VAE decoder must return 3 channels");

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

    if let Some(parent) = std::path::Path::new(OUTPUT_PATH).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to build RgbImage"))?;
    img.save(OUTPUT_PATH)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    let _ = device;
    Ok(())
}

fn tokenize_t5(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tok) => {
            let enc = tok.encode(prompt, true).expect("t5 tokenize");
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            ids.truncate(T5_SEQ_LEN);
            while ids.len() < T5_SEQ_LEN { ids.push(0); }
            ids
        }
        Err(e) => {
            eprintln!("[chroma_infer_turbo] T5 tokenizer load failed: {}", e);
            vec![0i32; T5_SEQ_LEN]
        }
    }
}
