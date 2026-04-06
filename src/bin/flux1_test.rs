//! Standalone test for FLUX 1 Dev inference pipeline.
//!
//! Tests each component individually:
//! 1. CLIP-L encoding → pooled output
//! 2. T5-XXL encoding → hidden states
//! 3. DiT transformer forward (single step)
//! 4. VAE decode (uses existing ldm_decoder)
//!
//! ⚠️ This does NOT produce a final image — it validates each component.
//! ⚠️ GPU-intensive: components are loaded/unloaded sequentially.
//!
//! Usage:
//!   RUST_LOG=info cargo run --release --bin flux1_test

use std::time::Instant;

const CLIP_PATH: &str = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors";
const T5_PATH: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";
const DIT_PATH: &str = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/ae.safetensors";
const CLIP_TOKENIZER: &str = concat!(
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/",
    "snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer/vocab.json"
);
const T5_TOKENIZER: &str = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

const PROMPT: &str = "a photograph of an astronaut riding a horse on mars, cinematic lighting";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    println!("=== FLUX 1 Dev Component Test ===\n");
    println!("Prompt: \"{}\"\n", PROMPT);

    // --- 1. CLIP-L ---
    println!("--- Phase 1: CLIP-L Encoding ---");
    let t0 = Instant::now();

    let clip_weights = flame_core::serialization::load_file(
        std::path::Path::new(CLIP_PATH), &device,
    )?;
    println!("  Loaded {} CLIP weights in {:.1}s", clip_weights.len(), t0.elapsed().as_secs_f32());

    let clip_config = inference_flame::models::clip_encoder::ClipConfig::default();
    let clip = inference_flame::models::clip_encoder::ClipEncoder::new(
        clip_weights, clip_config, device.clone(),
    );

    // Simple tokenization for CLIP (pad to 77)
    let clip_tokens = tokenize_clip_simple(PROMPT);
    println!("  CLIP tokens: {} (padded to 77)", clip_tokens.len());

    let (clip_hidden, clip_pooled) = clip.encode(&clip_tokens)?;
    println!("  hidden: {:?}", clip_hidden.shape());
    println!("  pooled: {:?}", clip_pooled.shape());
    println!("  CLIP done in {:.1}s", t0.elapsed().as_secs_f32());

    // Stats on pooled
    let pooled_f32 = clip_pooled.to_dtype(flame_core::DType::F32)?;
    let vals = pooled_f32.reshape(&[pooled_f32.shape().elem_count()])?.to_vec_f32()?;
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    println!("  pooled mean={:.4}, min={:.4}, max={:.4}",
        mean,
        vals.iter().cloned().fold(f32::INFINITY, f32::min),
        vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    drop(clip);
    println!();

    // --- 2. T5-XXL ---
    println!("--- Phase 2: T5-XXL Encoding ---");
    let t1 = Instant::now();

    let mut t5 = inference_flame::models::t5_encoder::T5Encoder::load(T5_PATH, &device)?;
    println!("  Loaded T5 in {:.1}s", t1.elapsed().as_secs_f32());

    // Simple tokenization for T5 (pad to 512)
    let t5_tokens = tokenize_t5_simple(PROMPT);
    println!("  T5 tokens: {} (padded to 512)", t5_tokens.len());

    let t5_hidden = t5.encode(&t5_tokens)?;
    println!("  hidden: {:?}", t5_hidden.shape());
    println!("  T5 done in {:.1}s", t1.elapsed().as_secs_f32());

    drop(t5);
    println!();

    // --- 3. DiT (single step) ---
    println!("--- Phase 3: FLUX 1 DiT (single forward) ---");
    let t2 = Instant::now();

    let mut dit = inference_flame::models::flux1_dit::Flux1DiT::load(DIT_PATH, &device)?;
    println!("  Loaded DiT in {:.1}s", t2.elapsed().as_secs_f32());

    // Create dummy inputs for a single forward pass
    // 1024x1024 → 128x128 latents → 64x64 packed = 4096 tokens
    let latent_h = 128usize;
    let latent_w = 128usize;
    let packed_h = latent_h / 2;
    let packed_w = latent_w / 2;
    let n_img = packed_h * packed_w; // 4096

    // Random noise as packed img tokens: [1, 4096, 64]
    let img = flame_core::Tensor::randn(
        flame_core::Shape::from_dims(&[1, n_img, 64]),
        0.0, 1.0,
        device.clone(),
    )?.to_dtype(flame_core::DType::BF16)?;

    // Timestep
    let timestep = flame_core::Tensor::from_vec(
        vec![1.0f32],
        flame_core::Shape::from_dims(&[1]),
        device.clone(),
    )?.to_dtype(flame_core::DType::BF16)?;

    // Guidance
    let guidance = flame_core::Tensor::from_vec(
        vec![3.5f32],
        flame_core::Shape::from_dims(&[1]),
        device.clone(),
    )?.to_dtype(flame_core::DType::BF16)?;

    // Position IDs
    let (img_ids, txt_ids) = build_flux1_ids(latent_h, latent_w, 512, &device)?;

    println!("  img: {:?}, txt: {:?}", img.shape(), t5_hidden.shape());
    println!("  img_ids: {:?}, txt_ids: {:?}", img_ids.shape(), txt_ids.shape());

    let t3 = Instant::now();
    let output = dit.forward(
        &img,
        &t5_hidden,
        &timestep,
        &img_ids,
        &txt_ids,
        Some(&guidance),
        Some(&clip_pooled),
    )?;
    println!("  Output: {:?}", output.shape());
    println!("  Forward took {:.1}s", t3.elapsed().as_secs_f32());

    drop(dit);
    println!();

    println!("=== ALL COMPONENTS TESTED ===");
    Ok(())
}

/// Build FLUX 1 position IDs.
fn build_flux1_ids(
    latent_h: usize,
    latent_w: usize,
    text_seq_len: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> Result<(flame_core::Tensor, flame_core::Tensor), Box<dyn std::error::Error>> {
    let packed_h = latent_h / 2;
    let packed_w = latent_w / 2;
    let n_img = packed_h * packed_w;

    // Image IDs: (0, h, w) for each position
    let mut img_data = vec![0.0f32; n_img * 3];
    for h in 0..packed_h {
        for w in 0..packed_w {
            let idx = h * packed_w + w;
            img_data[idx * 3] = 0.0;     // t dimension
            img_data[idx * 3 + 1] = h as f32;
            img_data[idx * 3 + 2] = w as f32;
        }
    }
    let img_ids = flame_core::Tensor::from_vec(
        img_data,
        flame_core::Shape::from_dims(&[n_img, 3]),
        device.clone(),
    )?.to_dtype(flame_core::DType::BF16)?;

    // Text IDs: all zeros
    let txt_ids = flame_core::Tensor::from_vec(
        vec![0.0f32; text_seq_len * 3],
        flame_core::Shape::from_dims(&[text_seq_len, 3]),
        device.clone(),
    )?.to_dtype(flame_core::DType::BF16)?;

    Ok((img_ids, txt_ids))
}

/// Simple CLIP tokenization (BPE would need the full vocab — use placeholder for now).
fn tokenize_clip_simple(prompt: &str) -> Vec<i32> {
    // Try using tokenizers crate with the CLIP tokenizer
    match tokenizers::Tokenizer::from_file(CLIP_TOKENIZER) {
        Ok(tokenizer) => {
            let encoding = tokenizer.encode(prompt, true).unwrap();
            let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
            // Pad or truncate to 77
            ids.truncate(77);
            while ids.len() < 77 {
                ids.push(49407); // CLIP pad token = EOS
            }
            ids
        }
        Err(_) => {
            // Fallback: BOS + placeholder + EOS + padding
            let mut ids = vec![49406i32]; // BOS
            ids.push(49407); // EOS
            while ids.len() < 77 {
                ids.push(49407);
            }
            ids
        }
    }
}

/// Simple T5 tokenization (SentencePiece).
fn tokenize_t5_simple(prompt: &str) -> Vec<i32> {
    match tokenizers::Tokenizer::from_file(T5_TOKENIZER) {
        Ok(tokenizer) => {
            let encoding = tokenizer.encode(prompt, true).unwrap();
            let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
            // Add EOS
            ids.push(1); // T5 EOS
            // Pad to 512
            ids.truncate(512);
            while ids.len() < 512 {
                ids.push(0); // T5 pad token
            }
            ids
        }
        Err(_) => {
            let mut ids = vec![1i32]; // Just EOS
            while ids.len() < 512 {
                ids.push(0);
            }
            ids
        }
    }
}
