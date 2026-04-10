//! Klein 9B image editing — pure Rust, reference-conditioned pipeline.
//!
//! Usage: klein9b_edit_infer [source_image] [prompt]
//!
//! Pipeline:
//! 1. Load + resize source image → VAE encode → pack to sequence tokens
//! 2. Qwen3 8B encode → drop encoder (frees ~15GB, warms mempool)
//! 3. Drop VAE encoder (frees VRAM before DiT)
//! 4. Klein 9B all-on-GPU (try, fallback to offloaded)
//! 5. Create noise + schedule + reference IDs
//! 6. Denoise with reference tokens concatenated in sequence dimension
//! 7. Drop model, VAE decode → PNG
//!
//! The source image is VAE-encoded, packed to sequence tokens with T-offset=10
//! in the RoPE position IDs. The model attends to both noise and reference
//! tokens via joint attention, distinguishing them by their T-coordinate.

use inference_flame::models::klein::{KleinTransformer, KleinOffloaded};
use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use inference_flame::sampling::klein_sampling::{
    build_img2img_sigmas, euler_denoise, prepare_reference_ids,
};
use inference_flame::vae::klein_vae::{KleinVaeDecoder, KleinVaeEncoder};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

const SOURCE_IMAGE: &str = "/home/alex/Downloads/59605771.png";
const MODEL_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const ENCODER_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";
const VAE_PATH: &str = "/home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/klein9b_edit_rust.png";

const DEFAULT_PROMPT: &str = "change her dress to blue";
const DEFAULT_NEGATIVE: &str = "";

const REF_T_OFFSET: f32 = 10.0;
const NUM_STEPS: usize = 35;
const GUIDANCE: f32 = 3.5;
const SEED: u64 = 42;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;
const SHIFT: f32 = 2.02;
const DENOISE_STRENGTH: f32 = 1.0;

fn load_sharded_weights(
    dir: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut all_weights = HashMap::new();
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();
    for (i, path) in shard_paths.iter().enumerate() {
        let t0 = Instant::now();
        let shard = flame_core::serialization::load_file(path, device)?;
        println!(
            "    Shard {}/{}: {} keys ({:.1}s)",
            i + 1,
            shard_paths.len(),
            shard.len(),
            t0.elapsed().as_secs_f32()
        );
        all_weights.extend(shard);
    }
    Ok(all_weights)
}

/// Try loading all-on-GPU. If OOM, fall back to offloaded.
enum Model {
    OnGpu(KleinTransformer),
    Offloaded(KleinOffloaded),
}

impl Model {
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        t: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> flame_core::Result<Tensor> {
        match self {
            Model::OnGpu(m) => m.forward(img, txt, t, img_ids, txt_ids),
            Model::Offloaded(m) => m.forward(img, txt, t, img_ids, txt_ids),
        }
    }
    fn config_str(&self) -> String {
        match self {
            Model::OnGpu(m) => format!("{:?} (all on GPU)", m.config()),
            Model::Offloaded(m) => format!("{:?} (offloaded)", m.config()),
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let source_path = args.get(1).map(|s| s.as_str()).unwrap_or(SOURCE_IMAGE);
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("============================================================");
    println!("Klein 9B Edit — Pure Rust (Reference Conditioning)");
    println!(
        "  {}x{}, {} steps, guidance {}, seed {}",
        WIDTH, HEIGHT, NUM_STEPS, GUIDANCE, SEED
    );
    println!("  denoise={}, shift={}, ref_t={}", DENOISE_STRENGTH, SHIFT, REF_T_OFFSET);
    println!("  Source: {}", source_path);
    println!("  Prompt: {}", &prompt[..prompt.len().min(80)]);
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load + VAE Encode source image → packed reference tokens
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Encode Source Image (VAE) ---");
    let t0 = Instant::now();

    // Load and resize source image
    let src_img = image::open(source_path)?.to_rgb8();
    let resized = image::imageops::resize(
        &src_img,
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Lanczos3,
    );
    println!(
        "  Source: {}x{} → resized {}x{}",
        src_img.width(),
        src_img.height(),
        WIDTH,
        HEIGHT
    );

    // Convert to [1, 3, H, W] BF16 in [-1, 1]
    let mut img_data = vec![0.0f32; 3 * HEIGHT * WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                img_data[c * HEIGHT * WIDTH + y * WIDTH + x] =
                    pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    let img_tensor = Tensor::from_f32_to_bf16(
        img_data,
        Shape::from_dims(&[1, 3, HEIGHT, WIDTH]),
        device.clone(),
    )?;

    // Load VAE encoder and encode
    let vae_weights =
        flame_core::serialization::load_file(std::path::Path::new(VAE_PATH), &device)?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae_encoder = KleinVaeEncoder::load(&vae_weights, &vae_device)?;
    let encoded = vae_encoder.encode_raw(&img_tensor)?; // [1, 128, H/16, W/16] — NO BN
    drop(img_tensor);

    let enc_dims = encoded.shape().dims();
    let (lat_h, lat_w) = (enc_dims[2], enc_dims[3]);
    println!("  Encoded: {:?}", enc_dims);

    // Pack to sequence: [1, 128, h, w] → permute → [1, h*w, 128]
    let ref_packed = encoded
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, lat_h * lat_w, 128])?;
    println!("  Reference tokens: [1, {}, 128]", lat_h * lat_w);

    // Drop VAE encoder weights (keep vae_weights for decoder later)
    drop(vae_encoder);
    println!("  VAE encode: {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Encode text (Qwen3 8B) — then drop (warms mempool)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Text Encoding (Qwen3 8B) ---");
    let t0 = Instant::now();

    let (pos_hidden, neg_hidden) = {
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Tokenizer: {}", e))?;

        let pos_fmt = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let neg_text = if DEFAULT_NEGATIVE.is_empty() {
            "".to_string()
        } else {
            DEFAULT_NEGATIVE.to_string()
        };
        let neg_fmt = format!(
            "<|im_start|>user\n{neg_text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );

        let pad_id = 151643i32;
        let mut pos_ids: Vec<i32> = tokenizer
            .encode(pos_fmt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();
        let mut neg_ids: Vec<i32> = tokenizer
            .encode(neg_fmt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();
        println!("  Tokens: {} pos, {} neg", pos_ids.len(), neg_ids.len());
        pos_ids.resize(512, pad_id);
        neg_ids.resize(512, pad_id);

        let enc_weights = load_sharded_weights(ENCODER_DIR, &device)?;
        let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
        let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());

        let pos_h = encoder.encode(&pos_ids)?;
        let neg_h = encoder.encode(&neg_ids)?;
        println!("  Output: {:?}", pos_h.dims());
        drop(encoder); // frees ~15GB — mempool keeps it cached
        (pos_h, neg_h)
    };
    println!("  Encoded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Load Klein 9B — try all-on-GPU first, fallback to offloaded
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Load Klein 9B ---");
    let t0 = Instant::now();

    let model = match KleinTransformer::from_safetensors(MODEL_PATH) {
        Ok(m) => {
            println!("  All weights on GPU");
            Model::OnGpu(m)
        }
        Err(e) => {
            println!(
                "  GPU load failed ({:?}), falling back to offloaded...",
                e
            );
            Model::Offloaded(KleinOffloaded::from_safetensors(MODEL_PATH)?)
        }
    };
    println!("  {}", model.config_str());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 4: Noise + schedule + reference IDs
    // ------------------------------------------------------------------
    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;
    let n_img = latent_h * latent_w;
    let txt_seq_len = 512usize;

    // Noise image IDs: [N_img, 4] where row = [0, row, col, 0]
    let mut noise_id_data = vec![0.0f32; n_img * 4];
    for r in 0..latent_h {
        for c in 0..latent_w {
            let idx = r * latent_w + c;
            noise_id_data[idx * 4 + 1] = r as f32;
            noise_id_data[idx * 4 + 2] = c as f32;
        }
    }
    let noise_img_ids =
        Tensor::from_f32_to_bf16(noise_id_data, Shape::from_dims(&[n_img, 4]), device.clone())?;

    // Reference IDs: [N_img, 4] where row = [10.0, row, col, 0]
    let ref_ids = prepare_reference_ids(latent_h, latent_w, REF_T_OFFSET, &device)?;

    // Combined IDs for forward pass: [2*N_img, 4]
    let combined_img_ids = Tensor::cat(&[&noise_img_ids, &ref_ids], 0)?;

    // Text IDs: zeros [512, 4]
    let txt_ids =
        Tensor::zeros_dtype(Shape::from_dims(&[txt_seq_len, 4]), DType::BF16, device.clone())?;

    // Generate noise
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
    let noise = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        device.clone(),
    )?
    .permute(&[0, 2, 3, 1])?
    .reshape(&[1, n_img, 128])?;

    // Build sigma schedule (fixed shift for edit pipeline)
    let timesteps = build_img2img_sigmas(NUM_STEPS, SHIFT, DENOISE_STRENGTH);

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

    let noise_seq_len = n_img; // number of noise tokens

    println!(
        "\n--- Stage 5: Denoise ({} steps, guidance={}, ref_t={}) ---",
        NUM_STEPS, GUIDANCE, REF_T_OFFSET
    );
    println!(
        "  Noise tokens: {}, Reference tokens: {}, Total img: {}",
        noise_seq_len,
        lat_h * lat_w,
        noise_seq_len + lat_h * lat_w
    );
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, t_curr| {
            let t_vec =
                Tensor::from_f32_to_bf16(vec![t_curr], Shape::from_dims(&[1]), device.clone())?;

            // Concatenate noise + reference tokens: [1, N_noise + N_ref, 128]
            let packed_img = Tensor::cat(&[x, &ref_packed], 1)?;

            // Conditional forward with combined IDs
            let pred_cond =
                model.forward(&packed_img, &pos_hidden, &t_vec, &combined_img_ids, &txt_ids)?;
            // Slice: keep only noise tokens
            let pred_cond = pred_cond.narrow(1, 0, noise_seq_len)?;

            // Unconditional forward
            let pred_uncond =
                model.forward(&packed_img, &neg_hidden, &t_vec, &combined_img_ids, &txt_ids)?;
            let pred_uncond = pred_uncond.narrow(1, 0, noise_seq_len)?;

            // CFG
            let diff = pred_cond.sub(&pred_uncond)?;
            pred_uncond.add(&diff.mul_scalar(GUIDANCE)?)
        },
        noise_input,
        &timesteps,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);

    // ------------------------------------------------------------------
    // Stage 6: VAE Decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 6: VAE Decode ---");
    let t0 = Instant::now();
    drop(model);

    let latents = denoised
        .reshape(&[1, latent_h, latent_w, 128])?
        .permute(&[0, 3, 1, 2])?;

    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    let rgb = vae.decode(&latents)?;
    println!("  {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 7: Save PNG
    // ------------------------------------------------------------------
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let d = rgb_f32.dims();
    let (out_h, out_w) = (d[2], d[3]);

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

    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image creation failed"))?
        .save(OUTPUT_PATH)?;

    println!("\n============================================================");
    println!("EDITED IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
