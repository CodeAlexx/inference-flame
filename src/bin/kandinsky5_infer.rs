//! Kandinsky-5 text-to-image inference — pure Rust using flame-core.
//!
//! Usage: cargo run --release --bin kandinsky5_infer
//!
//! Requires:
//!   - Qwen2.5-VL-7B text encoder weights
//!   - CLIP-L text encoder weights
//!   - Kandinsky-5 DiT checkpoint (e.g. kandinsky5lite_t2i.safetensors)
//!   - HunyuanVideo VAE weights
//!   - Tokenizers for Qwen2.5-VL and CLIP

use inference_flame::models::kandinsky5_dit::Kandinsky5DiT;
use inference_flame::models::qwen25vl_encoder::Qwen25VLEncoder;
use inference_flame::models::clip_encoder::ClipEncoder;
use inference_flame::vae::hunyuan_vae::HunyuanVaeDecoder;
use inference_flame::sampling::kandinsky5_sampling;

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

// Placeholder paths — user must set these
const QWEN_ENCODER_PATH: &str = "/home/alex/models/Qwen2.5-VL-7B-Instruct";
const CLIP_ENCODER_PATH: &str = "/home/alex/models/clip-vit-large-patch14";
const DIT_PATH: &str = "/home/alex/models/kandinsky5/kandinsky5lite_t2i.safetensors";
const VAE_PATH: &str = "/home/alex/models/kandinsky5/vae";
const QWEN_TOKENIZER_PATH: &str =
    "/home/alex/models/Qwen2.5-VL-7B-Instruct/tokenizer.json";
const CLIP_TOKENIZER_PATH: &str =
    "/home/alex/models/clip-vit-large-patch14/tokenizer.json";

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;
const NUM_STEPS: usize = 50;
const GUIDANCE: f32 = 5.0;
const SCHEDULER_SCALE: f32 = 3.0; // 3.0 for images, 10.0 for video
const OUTPUT_PATH: &str = "/home/alex/serenity/output/kandinsky5_output.png";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    println!("============================================================");
    println!("Kandinsky-5 Text-to-Image — Pure Rust Inference");
    println!("============================================================");

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            "A beautiful landscape with mountains and a lake at sunset, highly detailed, 8k"
                .to_string()
        });
    let negative = "";

    // ------------------------------------------------------------------
    // Stage 1: Text Encoding (Qwen2.5-VL)
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Qwen2.5-VL text encoding ---");
    let t0 = Instant::now();

    let qwen_tokenizer = tokenizers::Tokenizer::from_file(QWEN_TOKENIZER_PATH)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen tokenizer: {e}"))?;

    let pos_encoding = qwen_tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
    let pos_ids: Vec<i32> = pos_encoding.get_ids().iter().map(|&id| id as i32).collect();

    let neg_encoding = qwen_tokenizer
        .encode(negative, false)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
    let neg_ids: Vec<i32> = neg_encoding.get_ids().iter().map(|&id| id as i32).collect();

    println!("  Prompt tokens: {}", pos_ids.len());
    println!("  Negative tokens: {}", neg_ids.len());

    // NOTE: In production, load Qwen2.5-VL safetensors and run encoder forward pass.
    // The encoder produces hidden states that feed into the DiT cross-attention.
    // For now we show the pipeline structure — actual weights needed to run.
    println!(
        "  Qwen tokenization done in {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // ------------------------------------------------------------------
    // Stage 2: CLIP Encoding
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: CLIP text encoding ---");
    let t1 = Instant::now();

    let clip_tokenizer = tokenizers::Tokenizer::from_file(CLIP_TOKENIZER_PATH)
        .map_err(|e| anyhow::anyhow!("Failed to load CLIP tokenizer: {e}"))?;

    let clip_encoding = clip_tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("CLIP tokenize error: {e}"))?;
    let clip_ids: Vec<i32> = clip_encoding.get_ids().iter().map(|&id| id as i32).collect();
    println!("  CLIP tokens: {}", clip_ids.len());

    // CLIP produces pooled [1, 768] embedding used as conditioning vector.
    // Requires loading clip_encoder weights and running forward pass.
    println!(
        "  CLIP tokenization done in {:.1}ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // ------------------------------------------------------------------
    // Stage 3: Load DiT
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Loading Kandinsky-5 DiT ---");
    let t2 = Instant::now();
    let dit = Kandinsky5DiT::load(&[DIT_PATH], &device)?;
    println!(
        "  DiT loaded in {:.2}s",
        t2.elapsed().as_secs_f64()
    );

    // ------------------------------------------------------------------
    // Stage 4: Prepare noise and schedule
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Preparing noise ---");
    let latent_h = HEIGHT / 8; // VAE 8x spatial compression
    let latent_w = WIDTH / 8;
    let duration = 1; // T2I = single frame
    let _patch_h = latent_h / 2; // patch_size [1,2,2]
    let _patch_w = latent_w / 2;

    // Random noise: (bs*duration, latent_h, latent_w, channels)
    let noise = Tensor::randn(
        Shape::from_dims(&[duration, latent_h, latent_w, 16]),
        0.0,
        1.0,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let schedule =
        kandinsky5_sampling::build_velocity_schedule(NUM_STEPS, SCHEDULER_SCALE);
    println!(
        "  Noise shape: [{}, {}, {}, 16]",
        duration, latent_h, latent_w
    );
    println!(
        "  Schedule: {} steps, scale={}",
        NUM_STEPS, SCHEDULER_SCALE
    );

    // ------------------------------------------------------------------
    // Stage 5: Denoise loop
    // ------------------------------------------------------------------
    println!("\n--- Stage 5: Denoising ---");
    // TODO: Full denoise loop requires text embeddings from stages 1+2.
    // Structure:
    //   for i in 0..NUM_STEPS {
    //       let t = schedule[i];
    //       let dt = schedule[i+1] - schedule[i];
    //       let velocity = dit.forward(&noise, &text_embed, &pooled, &time, ...)?;
    //       noise = kandinsky5_sampling::euler_velocity_step(&velocity, &noise, dt)?;
    //   }
    println!("  (Denoise loop requires encoder weights — skipping)");

    // ------------------------------------------------------------------
    // Stage 6: VAE Decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 6: VAE decode ---");
    // let vae = HunyuanVaeDecoder::load(VAE_PATH, &device)?;
    // let rgb = vae.decode(&latents)?;
    // image::save_buffer(OUTPUT_PATH, &rgb_bytes, WIDTH as u32, HEIGHT as u32, image::ColorType::Rgb8)?;
    println!("  (VAE decode requires completed latents — skipping)");

    // ------------------------------------------------------------------
    // Done
    // ------------------------------------------------------------------
    println!(
        "\n============================================================"
    );
    println!(
        "Pipeline structure complete — needs weights to run end-to-end"
    );
    println!("Output would be saved to: {}", OUTPUT_PATH);
    println!(
        "Total elapsed: {:.2}s",
        t_total.elapsed().as_secs_f64()
    );
    println!("============================================================");

    Ok(())
}
