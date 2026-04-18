//! Klein 4B image editing — pure Rust, no Python.
//!
//! Usage: cargo run --release --bin klein_edit_infer [-- source_image "prompt text"]
//!
//! Pipeline:
//! 1. Load + resize source image
//! 2. VAE encode source image → reference latent tokens
//! 3. Tokenize + encode text (Qwen3 4B)
//! 4. Load Klein 4B model
//! 5. Create noise + sigma schedule + reference IDs
//! 6. Denoise with reference concatenation (noise + ref tokens)
//! 7. VAE decode → RGB → PNG

use inference_flame::models::klein::KleinTransformer;
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::sampling::klein_sampling::{build_img2img_sigmas, euler_denoise, prepare_reference_ids};
use inference_flame::vae::klein_vae::{KleinVaeDecoder, KleinVaeEncoder};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const SOURCE_IMAGE: &str = "/home/alex/Downloads/59605771.png";
const MODEL_PATH: &str = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-4b.safetensors";
const ENCODER_PATH: &str = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer.json";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/klein4b_edit_rust.png";

const DEFAULT_PROMPT: &str = "change her dress to blue";
const DEFAULT_NEGATIVE: &str = "lowres, bad quality, worst quality, bad anatomy, blurry, watermark, simple background, transparent background, sketch, jpeg artifacts, ugly, poorly drawn, censor";
const REF_T_OFFSET: f32 = 10.0;
const NUM_STEPS: usize = 35;
const GUIDANCE: f32 = 3.5;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;
const SHIFT: f32 = 2.02;
const DENOISE_STRENGTH: f32 = 1.0;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Klein 4B Edit — Pure Rust Inference (inference-flame)");
    println!("============================================================");

    let device = global_cuda_device();

    let args: Vec<String> = std::env::args().collect();
    let source_path = args.get(1).map(|s| s.as_str()).unwrap_or(SOURCE_IMAGE);
    let prompt = args.get(2).cloned().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("  Source: {}", source_path);
    println!("  Prompt: {}", prompt);
    println!("  Size: {}x{}, Steps: {}, Guidance: {}, Shift: {}", WIDTH, HEIGHT, NUM_STEPS, GUIDANCE, SHIFT);
    println!("  Denoise: {}, Ref T-offset: {}", DENOISE_STRENGTH, REF_T_OFFSET);

    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;
    let n_img = latent_h * latent_w;

    // ------------------------------------------------------------------
    // Stage 1: Load + resize source image → [1, 3, H, W] BF16 in [-1, 1]
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load Source Image ---");
    let t0 = Instant::now();

    let src_img = image::open(source_path)?.to_rgb8();
    let resized = image::imageops::resize(
        &src_img,
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let mut img_data = vec![0.0f32; 3 * HEIGHT * WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                img_data[c * HEIGHT * WIDTH + y * WIDTH + x] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    let img_tensor = Tensor::from_f32_to_bf16(
        img_data,
        Shape::from_dims(&[1, 3, HEIGHT, WIDTH]),
        device.clone(),
    )?;
    println!("  Image: {:?} loaded in {:.1}s", img_tensor.dims(), t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: VAE encode source image → reference tokens
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: VAE Encode Source Image ---");
    let t0 = Instant::now();

    let vae_weights = flame_core::serialization::load_file(
        std::path::Path::new(VAE_PATH),
        &device,
    )?;
    println!("  VAE weights loaded: {} keys", vae_weights.len());

    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae_enc = KleinVaeEncoder::load(&vae_weights, &vae_device)?;
    println!("  VAE encoder built");

    // encode_raw: patchified but NO batchnorm (edit ref tokens must be un-normalized)
    let encoded = vae_enc.encode_raw(&img_tensor)?;
    println!("  Encoded: {:?}", encoded.dims());

    // Pack to sequence: [1, 128, h, w] → permute → [1, h*w, 128]
    let ref_packed = encoded
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, latent_h * latent_w, 128])?;
    println!("  Reference tokens: {:?}", ref_packed.dims());

    // Free VAE encoder + source image tensors
    drop(vae_enc);
    drop(img_tensor);
    drop(encoded);
    // Keep vae_weights around — decoder will reuse them
    println!("  VAE encode in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Text encoding (Qwen3 4B)
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Text Encoding ---");
    println!("  Prompt: {}", prompt);
    let t0 = Instant::now();

    const TXT_PAD_LEN: usize = 512;
    const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
    const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    let (pos_hidden, neg_hidden) = {
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let pos_formatted = format!("{KLEIN_TEMPLATE_PRE}{prompt}{KLEIN_TEMPLATE_POST}");
        let negative = DEFAULT_NEGATIVE;
        let neg_formatted = format!("{KLEIN_TEMPLATE_PRE}{negative}{KLEIN_TEMPLATE_POST}");

        let pos_enc = tokenizer
            .encode(pos_formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;
        let neg_enc = tokenizer
            .encode(neg_formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;

        let pad_id = 151643i32; // Qwen3 PAD/EOS
        let mut pos_ids: Vec<i32> = pos_enc.get_ids().iter().map(|&id| id as i32).collect();
        let mut neg_ids: Vec<i32> = neg_enc.get_ids().iter().map(|&id| id as i32).collect();
        println!("  Pos tokens: {} (template-wrapped)", pos_ids.len());
        println!("  Neg tokens: {} (template-wrapped)", neg_ids.len());
        pos_ids.resize(TXT_PAD_LEN, pad_id);
        neg_ids.resize(TXT_PAD_LEN, pad_id);

        let enc_weights = flame_core::serialization::load_file(
            std::path::Path::new(ENCODER_PATH),
            &device,
        )?;
        let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
        let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());

        let pos_h = encoder.encode(&pos_ids)?;
        let neg_h = encoder.encode(&neg_ids)?;
        println!("  pos: {:?}, neg: {:?}", pos_h.dims(), neg_h.dims());
        println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

        drop(encoder);
        (pos_h, neg_h)
    };
    let txt_seq_len = TXT_PAD_LEN;

    // ------------------------------------------------------------------
    // Stage 4: Load Klein 4B model
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Load Klein 4B model ---");
    let t0 = Instant::now();
    let model = KleinTransformer::from_safetensors(MODEL_PATH)?;
    println!("  Config: {:?}", model.config());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 5: Create noise + schedule + build IDs
    // ------------------------------------------------------------------
    println!("\n--- Stage 5: Prepare noise + sigmas + IDs ---");

    // Noise img_ids: [N_img, 4] where each row = [0, row, col, 0]
    let mut img_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            img_data[idx * 4 + 1] = r as f32;
            img_data[idx * 4 + 2] = c as f32;
        }
    }
    let noise_img_ids = Tensor::from_f32_to_bf16(
        img_data,
        Shape::from_dims(&[n_img, 4]),
        device.clone(),
    )?;

    // Reference IDs: same spatial grid but T=REF_T_OFFSET instead of T=0
    let ref_ids = prepare_reference_ids(latent_h, latent_w, REF_T_OFFSET, &device)?;
    println!("  Reference IDs: {:?} (t={})", ref_ids.dims(), REF_T_OFFSET);

    // Combined img_ids: [noise_ids ; ref_ids] along dim 0
    let combined_img_ids = Tensor::cat(&[&noise_img_ids, &ref_ids], 0)?;
    println!("  Combined img_ids: {:?}", combined_img_ids.dims());

    // txt_ids: [N_txt, 4] — all zeros
    let txt_ids = Tensor::zeros_dtype(
        Shape::from_dims(&[txt_seq_len, 4]),
        DType::BF16,
        device.clone(),
    )?;
    println!("  txt_ids: {:?}", txt_ids.dims());

    // Seeded noise using Box-Muller
    let numel = 128 * latent_h * latent_w;
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

    let noise_spatial = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        device.clone(),
    )?;

    // Pack: [1, 128, H, W] -> [1, H, W, 128] -> [1, H*W, 128]
    let noise = noise_spatial
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, latent_h * latent_w, 128])?;
    println!("  Noise: {:?} (packed)", noise.dims());

    // Fixed-shift schedule (edit pipeline uses fixed shift, not dynamic mu)
    let timesteps = build_img2img_sigmas(NUM_STEPS, SHIFT, DENOISE_STRENGTH);
    println!(
        "  Schedule: {} values, t[0]={:.4}, t[-2]={:.4}",
        timesteps.len(),
        timesteps[0],
        timesteps[NUM_STEPS - 1]
    );

    // For denoise_strength < 1.0: blend noise with encoded latents
    // noise_input = sigma * noise + (1 - sigma) * ref_packed
    // At DENOISE_STRENGTH = 1.0, just use pure noise.
    let noise_input = if DENOISE_STRENGTH < 0.9999 {
        let sigma_start = timesteps[0];
        println!("  Blending noise with reference at sigma={:.4}", sigma_start);
        let scaled_noise = noise.mul_scalar(sigma_start)?;
        let scaled_ref = ref_packed.mul_scalar(1.0 - sigma_start)?;
        scaled_noise.add(&scaled_ref)?
    } else {
        println!("  Pure noise (full generation with reference conditioning)");
        noise
    };

    let noise_seq_len = n_img;
    println!("  Noise seq len: {}, Ref seq len: {}, Total img tokens: {}",
        noise_seq_len, n_img, noise_seq_len + n_img);

    // ------------------------------------------------------------------
    // Stage 6: Denoise with reference concatenation
    // ------------------------------------------------------------------
    println!(
        "\n--- Stage 6: Denoise ({} steps, guidance={}, ref-conditioned) ---",
        NUM_STEPS, GUIDANCE
    );
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, t_curr| {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;

            // Concatenate [noise, ref_packed] along sequence dim (dim 1)
            let combined_img = Tensor::cat(&[x, &ref_packed], 1)?;

            // Forward with combined img_ids (noise + ref positions)
            let pred_cond = model.forward(
                &combined_img,
                &pos_hidden,
                &t_vec,
                &combined_img_ids,
                &txt_ids,
            )?;

            // Slice only the noise tokens (first noise_seq_len)
            let pred_cond = pred_cond.narrow(1, 0, noise_seq_len)?;

            // Unconditional pass
            let pred_uncond = model.forward(
                &combined_img,
                &neg_hidden,
                &t_vec,
                &combined_img_ids,
                &txt_ids,
            )?;
            let pred_uncond = pred_uncond.narrow(1, 0, noise_seq_len)?;

            // CFG: uncond + guidance * (cond - uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(GUIDANCE)?)
        },
        noise_input,
        &timesteps,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);
    println!("  Output: {:?}", denoised.dims());

    // ------------------------------------------------------------------
    // Stage 7: Unpack + VAE decode + save PNG
    // ------------------------------------------------------------------
    println!("\n--- Stage 7: VAE Decode ---");
    let t0 = Instant::now();

    // Free DiT weights to make room for VAE. Pool cache is saturated by
    // the denoise loop; clear + trim so the VAE decode's big NHWC conv
    // buffers (e.g. [1,1024,1024,256] at the head) can allocate cleanly.
    drop(model);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  DiT weights freed + pool trimmed");

    // Unpack: [1, H*W, 128] -> [1, H, W, 128] -> [1, 128, H, W]
    let latents = denoised
        .reshape(&[1, latent_h, latent_w, 128])?
        .permute(&[0, 3, 1, 2])?;
    println!("  Latents: {:?}", latents.dims());

    // Load VAE decoder (reuse weights already in memory)
    println!("  Building VAE decoder...");
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    println!("  VAE decoder built");

    // Decode
    println!("  Decoding...");
    let rgb = vae.decode(&latents)?;
    println!("  Decoded: {:?}", rgb.dims());
    println!("  VAE in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Save PNG
    // ------------------------------------------------------------------
    println!("\n--- Save PNG ---");

    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let (_, _, out_h, out_w) = {
        let d = rgb_f32.dims();
        (d[0], d[1], d[2], d[3])
    };

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

    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
    img.save(OUTPUT_PATH)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!("\n============================================================");
    println!("EDITED IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total time: {:.1}s", dt_total);
    println!("============================================================");

    Ok(())
}
