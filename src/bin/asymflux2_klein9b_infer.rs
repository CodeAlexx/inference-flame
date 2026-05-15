//! AsymFLUX.2 Klein 9B — pixel-space pure-Rust inference.
//!
//! Replaces the VAE with Oklab color encoding and wraps Klein 9B with
//! AsymFlow velocity reconstruction. Algorithm matches LakonLab's
//! `PixelFlux2KleinPipeline` (Apache 2.0).
//!
//! Math foundations (all tested green, see `models/asymflux2.rs` tests):
//! - `vae::oklab` — Oklab color encoder/decoder (Step 1)
//! - `models::asymflux2::{compute_calibration, asymflow_velocity}` (Step 2)
//! - `models::asymflux2::{patchify, pack, unpack, unpatchify}` (Step 3)
//! - `models::asymflux2::guidance_bias` — orthogonal CFG (Step 4)
//! - `models::asymflux2::{klein_dynamic_shift, compute_sigma_schedule,
//!    euler_step, clamp_denoised_oklab}` — scheduler (Step 5)
//! - `models::asymflux2::extract_asymflow_buffers` — buffer loader (Step 6)
//!
//! Usage:
//!   asymflux2_klein9b_infer [PROMPT]
//!
//! Environment overrides:
//!   ASYMFLUX2_ADAPTER  - path to adapter safetensors
//!                       (default: ~/EriDiffusion/Models/checkpoints/asymflux2-klein-9b.safetensors)
//!   ASYMFLUX2_STEPS    - number of inference steps (default 38)
//!   ASYMFLUX2_W        - image width (default 512)
//!   ASYMFLUX2_H        - image height (default 512)
//!
//! Conditioned on the adapter loading cleanly — the diffusers-style key
//! naming in the HuggingFace adapter (`x_embedder.*`,
//! `transformer_blocks.*`) does **not** match the BFL key naming
//! (`img_in.*`, `double_blocks.*`) that `KleinTransformer::from_weights`
//! expects. If the load fails, the binary prints the first 30 keys it
//! found and bails — see the `// KEY MAPPING NOTE` block below for the
//! translator that needs to be written before this can actually generate
//! images.

use std::collections::HashMap;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::asymflux2;
use inference_flame::models::klein::KleinTransformer;
use inference_flame::models::qwen3_encoder::Qwen3Encoder;

// --------------------------------------------------------------------- //
// Defaults / config
// --------------------------------------------------------------------- //

const DEFAULT_ADAPTER_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/asymflux2-klein-9b.safetensors";
const DEFAULT_BASE_PATH: &str =
    "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors";
const NUM_DOUBLE: usize = 8;
const NUM_SINGLE: usize = 24;
const ENCODER_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";
const TOKENIZER_PATH: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json";
const OUTPUT_PATH_TEMPLATE: &str =
    "/home/alex/EriDiffusion/inference-flame/output/asymflux2_klein9b_rust_{}.png";

const DEFAULT_NEG_PROMPT: &str =
    "Low quality, worst quality, blurry, deformed, bad anatomy, unclear text";

const DEFAULT_PROMPTS: &[&str] = &[
    "High-angle close-up of a 1910s young woman, shown in 3/4 view, with a melancholic expression and light hair in a braid, roller skating in a crowded indoor rink. She is dressed in a long-sleeved dark dress adorned with a delicate white ruffled lace collar and matching white ruffles at her wrists. The scene is captured on a worn wooden floor composed of long planks, with numerous other skaters blurred in the hazy background under distant hanging banners. The image has a vintage sepia-toned aesthetic with a fine grain, creating a nostalgic and quiet mood. Soft, diffused overhead lighting casts gentle shadows, emphasizing the texture of the lace and wood.",
    "Close-up of a woman with dark hair and a hummingbird. The woman is shown in side profile, with her eyes closed in a serene expression, and her lips are softly parted. A small hummingbird hovers in the air, its long beak pointing upward, touching the woman's lower lip, as its wings are spread wide in flight. The composition is set against a plain, neutral background, emphasizing the intimate interaction between the two subjects. The color palette is monochromatic with desaturated teal and cream tones, applied with a soft, painterly canvas texture. Diffused lighting creates gentle shadows on the woman's neck and jawline, enhancing the tranquil and ethereal mood of this surrealist art piece.",
    "Extreme close-up of the left half face of a young woman with fair skin and wet, dark hair partially submerged in water. The waterline rests just below a striking, deep green eye that gazes directly at the viewer with an intense, mysterious expression. Warm, golden sunlight illuminates the right side of the face and forehead, casting intricate shadows from damp hair strands across the textured skin and dark eyebrows. The green iris is highly detailed with flecks of gold. The surrounding water surfaces are a mix of teal and amber, capturing sparkling light reflections in the foreground. The composition is tight and intimate, emphasizing the subject's gaze. The color palette features a contrast between warm golden-orange tones and the cool, clear water, highlighting the vivid emerald eye as the central focal point.",
];

const DEFAULT_NUM_STEPS: usize = 38;
// Both overridable via env. Lower guidance gives less-amplified output;
// orthogonal=0 falls back to plain CFG to isolate orthogonal-projection bugs.
fn cfg_scale() -> f32 {
    std::env::var("ASYMFLUX2_CFG").ok().and_then(|s| s.parse().ok()).unwrap_or(4.0)
}
fn cfg_ortho() -> f32 {
    std::env::var("ASYMFLUX2_ORTHO").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0)
}
const SIGMA_MIN: f32 = 1e-4;
const PATCH_SIZE: usize = 16;
const SEED: u64 = 42;

// --------------------------------------------------------------------- //
// CLI / env parsing
// --------------------------------------------------------------------- //

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

// --------------------------------------------------------------------- //
// Adapter loading (Step 6 integration)
// --------------------------------------------------------------------- //

/// Result of loading the AsymFLUX.2 Klein 9B adapter: the transformer
/// (BFL-key naming after translation), the AsymFlow `proj_buffer`
/// `(768, 128) F32`, and the host scalar `scale_buffer`.
struct AsymFlux2Klein {
    transformer: KleinTransformer,
    proj_buffer: Tensor,
    scale_buffer: f32,
}

// KEY MAPPING NOTE
// ----------------
// The HuggingFace adapter `Lakonik/AsymFLUX.2-klein-9B` is **NOT** a full
// state dict — it's a LoRA on top of `FLUX.2-klein-base-9B` plus a small
// set of "modified" non-LoRA weights that absorb the AsymFlow proj_mat
// pre-multiplication.
//
// Adapter contents (verified by inspecting the safetensors header):
//   Non-LoRA replacement weights (5 keys, diffusers naming):
//     x_embedder.weight              [4096, 768]   BF16
//     proj_out.weight                [768, 4096]   BF16
//     norm_out.linear.weight         [8192, 4096]  BF16
//     proj_buffer                    [768, 128]    BF16 (AsymFlow buffer)
//     scale_buffer                   []            BF16 (AsymFlow buffer)
//   LoRA pairs (lora_A / lora_B, diffusers naming), each rank-256:
//     time_guidance_embed.timestep_embedder.linear_{1,2}
//     transformer_blocks.{i}.ff.linear_{in,out}        (i in 0..8)
//     transformer_blocks.{i}.ff_context.linear_{in,out}  (i in 0..8)
//     single_transformer_blocks.{i}.attn.to_out          (i in 0..24)
//
// Total: 58 LoRA fusions + 3 raw weight replacements + 2 buffers.
//
// The fusion math per target is `W_new = W + B @ A` (LoRA scaling
// assumed 1.0 — the file has no metadata; LakonLab calls
// `self.fuse_lora()` from PEFT with default scaling = alpha / rank, and
// for AsymFLUX.2's `lora_alpha=lora_rank` config (`asymflux2.py:355-356`)
// the ratio is 1).

/// Mapping from (diffusers LoRA module prefix) → (BFL base key for the
/// weight to be patched). The LoRA fuses into the base BFL weight at
/// `bfl_key`.
fn lora_target_mappings() -> Vec<(String, String)> {
    let mut out = Vec::new();
    // Timestep embedder
    out.push((
        "time_guidance_embed.timestep_embedder.linear_1".into(),
        "time_in.in_layer.weight".into(),
    ));
    out.push((
        "time_guidance_embed.timestep_embedder.linear_2".into(),
        "time_in.out_layer.weight".into(),
    ));
    // Double-stream blocks (i in 0..NUM_DOUBLE): ff = img_mlp, ff_context = txt_mlp
    for i in 0..NUM_DOUBLE {
        out.push((
            format!("transformer_blocks.{i}.ff.linear_in"),
            format!("double_blocks.{i}.img_mlp.0.weight"),
        ));
        out.push((
            format!("transformer_blocks.{i}.ff.linear_out"),
            format!("double_blocks.{i}.img_mlp.2.weight"),
        ));
        out.push((
            format!("transformer_blocks.{i}.ff_context.linear_in"),
            format!("double_blocks.{i}.txt_mlp.0.weight"),
        ));
        out.push((
            format!("transformer_blocks.{i}.ff_context.linear_out"),
            format!("double_blocks.{i}.txt_mlp.2.weight"),
        ));
    }
    // Single-stream blocks: attn.to_out → linear2 (which fuses attn_proj + ffn_down)
    for i in 0..NUM_SINGLE {
        out.push((
            format!("single_transformer_blocks.{i}.attn.to_out"),
            format!("single_blocks.{i}.linear2.weight"),
        ));
    }
    out
}

/// Fuse a single LoRA pair into the base weight in place.
/// `W_new = W + B @ A`. All BF16.
fn fuse_one(
    base: &mut HashMap<String, Tensor>,
    base_key: &str,
    lora_a: &Tensor,
    lora_b: &Tensor,
) -> anyhow::Result<()> {
    let w = base.get(base_key).ok_or_else(|| {
        anyhow::anyhow!("fuse_one: base key not found: {}", base_key)
    })?;
    // B [out, rank] @ A [rank, in] → delta [out, in]
    let delta = lora_b.matmul(lora_a)?;
    // Optional scaling — PEFT applies alpha/rank during fusion. The adapter
    // has no metadata; LakonLab's config uses alpha=rank so scaling=1.0.
    // Overridable via ASYMFLUX2_LORA_SCALE for diagnostic sweeps.
    let scale = std::env::var("ASYMFLUX2_LORA_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let delta = if scale != 1.0 {
        delta.mul_scalar(scale)?
    } else {
        delta
    };
    // Dtype must match for add. Both adapter LoRAs and base are BF16.
    let delta_t = if delta.dtype() == w.dtype() {
        delta
    } else {
        delta.to_dtype(w.dtype())?
    };
    let fused = w.add(&delta_t)?;
    // Force BF16 regardless of upstream dtype quirks — KleinTransformer's
    // pre-transpose path requires BF16 specifically.
    let fused_bf16 = if fused.dtype() == DType::BF16 {
        fused
    } else {
        fused.to_dtype(DType::BF16)?
    };
    base.insert(base_key.to_string(), fused_bf16);
    Ok(())
}

/// Pre-flight check: every rank-2 `.weight` in `base` must be BF16
/// (KleinTransformer pre-transpose requires it). Returns the first
/// offender, or `Ok(())` if all clean.
fn assert_all_bf16(base: &HashMap<String, Tensor>) -> anyhow::Result<()> {
    for (k, t) in base {
        if k.ends_with(".weight")
            && !k.ends_with(".scale")
            && t.shape().dims().len() == 2
            && t.dtype() != DType::BF16
        {
            anyhow::bail!(
                "weight {} is {:?}, expected BF16 (shape {:?})",
                k,
                t.dtype(),
                t.shape().dims()
            );
        }
    }
    Ok(())
}

fn load_adapter(adapter_path: &str, base_path: &str) -> anyhow::Result<AsymFlux2Klein> {
    let device = global_cuda_device();

    // 1. Load base.
    println!("  Loading BASE weights from {}", base_path);
    let t0 = Instant::now();
    let mut base = flame_core::serialization::load_file(base_path, &device)?;
    println!(
        "    {} base tensors in {:.1}s",
        base.len(),
        t0.elapsed().as_secs_f32()
    );

    // 2. Load adapter.
    println!("  Loading ADAPTER from {}", adapter_path);
    let t0 = Instant::now();
    let adapter = flame_core::serialization::load_file(adapter_path, &device)?;
    println!(
        "    {} adapter tensors in {:.1}s",
        adapter.len(),
        t0.elapsed().as_secs_f32()
    );

    // 3. Extract AsymFlow buffers from adapter.
    let (proj_buffer, scale_buffer) = asymflux2::extract_asymflow_buffers(&adapter)
        .map_err(|e| anyhow::anyhow!("extract_asymflow_buffers: {:?}", e))?;
    println!(
        "    proj_buffer: {:?} F32, scale_buffer = {}",
        proj_buffer.shape().dims(),
        scale_buffer
    );

    // 4. Apply 3 raw weight replacements.
    let raw_replacements: &[(&str, &str)] = &[
        ("x_embedder.weight", "img_in.weight"),
        ("proj_out.weight", "final_layer.linear.weight"),
        ("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight"),
    ];
    for (from, to) in raw_replacements {
        let new_w = adapter.get(*from).ok_or_else(|| {
            anyhow::anyhow!("raw replacement missing adapter key {}", from)
        })?;
        let old_shape = base
            .get(*to)
            .map(|t| format!("{:?}", t.shape().dims()))
            .unwrap_or_else(|| "<missing>".into());
        // The adapter stores these 3 weights as F16; `serialization::load_file`
        // surfaces them as F32. KleinTransformer requires BF16 — cast.
        let new_w_bf16 = if new_w.dtype() == DType::BF16 {
            new_w.clone()
        } else {
            new_w.to_dtype(DType::BF16)?
        };
        // SHIFT/SCALE ORDER FIX
        // ---------------------
        // Diffusers' `AdaLayerNormContinuous` (the source of
        // `norm_out.linear.weight`) outputs `[scale, shift]` along the row
        // axis. BFL's klein9b `final_layer` (the target slot
        // `final_layer.adaLN_modulation.1.weight`) decodes `[shift, scale]`
        // (first/second halves via `narrow`, see klein.rs:815-820).
        //
        // Copying the adapter weight straight in would silently swap the
        // two halves at every inference step → the model multiplies by the
        // shift and adds the scale. That mis-modulation manifests as the
        // fine-period cross-hatch we observed in earlier outputs. Fix:
        // swap row blocks `[0..H/2]` and `[H/2..H]` of the replacement
        // before insertion.
        let to_insert = if *to == "final_layer.adaLN_modulation.1.weight" {
            let dims = new_w_bf16.shape().dims();
            let h = dims[0];
            assert!(
                h % 2 == 0,
                "norm_out.linear.weight row count {} must be even",
                h
            );
            let half = h / 2;
            let scale_rows = new_w_bf16.narrow(0, 0, half)?;
            let shift_rows = new_w_bf16.narrow(0, half, half)?;
            let swapped = Tensor::cat(&[&shift_rows, &scale_rows], 0)?;
            println!(
                "    REPLACE {} {} → {} {:?} (cast {:?}→BF16, swapped [scale,shift]→[shift,scale])",
                to,
                old_shape,
                from,
                new_w.shape().dims(),
                new_w.dtype()
            );
            swapped
        } else {
            println!(
                "    REPLACE {} {} → {} {:?} (cast {:?}→BF16)",
                to,
                old_shape,
                from,
                new_w.shape().dims(),
                new_w.dtype()
            );
            new_w_bf16
        };
        base.insert((*to).to_string(), to_insert);
    }

    // 5. Apply LoRA fusions.
    let t0 = Instant::now();
    let mappings = lora_target_mappings();
    let mut fused_count = 0;
    let mut missing = Vec::new();
    for (prefix, base_key) in &mappings {
        let a_key = format!("{prefix}.lora_A.weight");
        let b_key = format!("{prefix}.lora_B.weight");
        match (adapter.get(&a_key), adapter.get(&b_key)) {
            (Some(a), Some(b)) => {
                fuse_one(&mut base, base_key, a, b)?;
                fused_count += 1;
            }
            _ => {
                missing.push(prefix.clone());
            }
        }
    }
    println!(
        "    fused {} LoRA pairs in {:.1}s (missing: {})",
        fused_count,
        t0.elapsed().as_secs_f32(),
        missing.len()
    );
    if !missing.is_empty() {
        for m in missing.iter().take(5) {
            eprintln!("      missing LoRA: {}", m);
        }
    }

    // 5b. Sanity-check dtypes before handing to KleinTransformer.
    assert_all_bf16(&base)?;

    // 6. Build the transformer from the fused base.
    let transformer = KleinTransformer::from_weights(base).map_err(|e| {
        anyhow::anyhow!("KleinTransformer::from_weights failed: {:?}", e)
    })?;

    Ok(AsymFlux2Klein {
        transformer,
        proj_buffer,
        scale_buffer,
    })
}

// --------------------------------------------------------------------- //
// Text encoding (mirror klein9b_infer)
// --------------------------------------------------------------------- //

fn encode_text(
    prompt: &str,
    negative_prompt: &str,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor)> {
    println!("  Encoding text (Qwen3 8B)...");
    let t0 = Instant::now();

    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {}", e))?;

    let fmt = |p: &str| {
        format!(
            "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    };

    let pad_id = 151643_i32;
    let mut pos_ids: Vec<i32> = tokenizer
        .encode(fmt(prompt).as_str(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    let mut neg_ids: Vec<i32> = tokenizer
        .encode(fmt(negative_prompt).as_str(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    pos_ids.resize(512, pad_id);
    neg_ids.resize(512, pad_id);

    let enc_weights = load_sharded_weights(ENCODER_DIR, device)?;
    let enc_config = Qwen3Encoder::config_from_weights(&enc_weights)?;
    let encoder = Qwen3Encoder::new(enc_weights, enc_config, device.clone());
    let pos_h = encoder.encode(&pos_ids)?;
    let neg_h = encoder.encode(&neg_ids)?;
    drop(encoder);

    println!("    encoded in {:.1}s", t0.elapsed().as_secs_f32());
    Ok((pos_h, neg_h))
}

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
    for path in shard_paths {
        let shard = flame_core::serialization::load_file(&path, device)?;
        all_weights.extend(shard);
    }
    Ok(all_weights)
}

// --------------------------------------------------------------------- //
// Initial noise — pixel space
// --------------------------------------------------------------------- //

fn make_pixel_noise(
    b: usize,
    h: usize,
    w: usize,
    seed: u64,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<Tensor> {
    use rand::prelude::*;
    let numel = b * 3 * h * w;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    let n_pairs = numel / 2;
    for _ in 0..n_pairs {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data.push(r * theta.cos());
        data.push(r * theta.sin());
    }
    if numel % 2 == 1 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    Ok(Tensor::from_vec(
        data,
        Shape::from_dims(&[b, 3, h, w]),
        device,
    )?)
}

// --------------------------------------------------------------------- //
// img_ids / txt_ids for FLUX rotary embeddings
// --------------------------------------------------------------------- //

fn make_img_ids(
    h_p: usize,
    w_p: usize,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<Tensor> {
    let n = h_p * w_p;
    let mut data = vec![0.0_f32; n * 4];
    for r in 0..h_p {
        for c in 0..w_p {
            let idx = r * w_p + c;
            data[idx * 4 + 1] = r as f32;
            data[idx * 4 + 2] = c as f32;
        }
    }
    Ok(Tensor::from_f32_to_bf16(
        data,
        Shape::from_dims(&[n, 4]),
        device,
    )?)
}

// --------------------------------------------------------------------- //
// Per-step forward through the wrapped transformer
// --------------------------------------------------------------------- //

#[allow(clippy::too_many_arguments)]
fn wrapped_forward(
    transformer: &KleinTransformer,
    x_t_pixel: &Tensor,           // (1, 3, H, W) F32
    proj_buffer: &Tensor,         // (768, 128) F32
    text_emb: &Tensor,            // (1, 512, joint_dim) BF16
    timestep: f32,                // [0, 1] flow-matching sigma
    scale_buffer: f32,
    img_ids: &Tensor,             // (n_img, 4) BF16
    txt_ids: &Tensor,             // (512, 4) BF16
) -> anyhow::Result<Tensor> {
    // 1. Patchify pixel input to (1, 768, h_p, w_p) then pack to (1, n_img, 768).
    let patched = asymflux2::patchify(x_t_pixel, PATCH_SIZE)?;
    let x_t_packed = asymflux2::pack(&patched)?;

    // 2. Calibration: k, cal_timestep.
    let cal = asymflux2::compute_calibration(timestep, scale_buffer, 1.0);

    // 3. Scale input by k before x_embedder. Keep in F32 (the transformer
    //    will cast to BF16 internally where it needs to).
    let hidden_states = x_t_packed.mul_scalar(cal.k)?;
    let hidden_states_bf16 = hidden_states.to_dtype(DType::BF16)?;

    // 4. Transformer forward — expects cal_timestep, NOT raw timestep.
    let t_vec =
        Tensor::from_f32_to_bf16(vec![cal.cal_timestep], Shape::from_dims(&[1]), x_t_pixel.device().clone())?;
    let u_a_packed = transformer.forward(&hidden_states_bf16, text_emb, &t_vec, img_ids, txt_ids)?;

    // 5. AsymFlow velocity reconstruction. Inputs are mixed-dtype; the
    //    function casts to F32 internally. Output is F32 (1, n_img, 768).
    let velocity_packed = asymflux2::asymflow_velocity(
        &u_a_packed,
        &x_t_packed,
        &cal,
        proj_buffer,
        SIGMA_MIN,
    )?;

    // 6. Unpack + unpatchify back to (1, 3, H, W).
    let dims = patched.shape().dims();
    let (h_p, w_p) = (dims[2], dims[3]);
    let unpacked = asymflux2::unpack(&velocity_packed, h_p, w_p)?;
    let velocity_pixel = asymflux2::unpatchify(&unpacked, PATCH_SIZE)?;
    Ok(velocity_pixel)
}

// --------------------------------------------------------------------- //
// Save PNG from (1, 3, H, W) F32 sRGB pixels in [-1, 1]
// --------------------------------------------------------------------- //

fn save_png_planar(pixels: &[f32], h: usize, w: usize, path: &str) -> anyhow::Result<()> {
    assert_eq!(pixels.len(), 3 * h * w);
    let plane = h * w;
    let mut rgb = vec![0_u8; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = pixels[c * plane + y * w + x];
                let v = (v.clamp(-1.0, 1.0) + 1.0) * 127.5;
                rgb[(y * w + x) * 3 + c] = v as u8;
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow::anyhow!("image::RgbImage::from_raw failed"))?
        .save(path)?;
    Ok(())
}

// --------------------------------------------------------------------- //
// Sampling loop
// --------------------------------------------------------------------- //

#[allow(clippy::too_many_arguments)]
fn sample(
    model: &AsymFlux2Klein,
    pos_emb: &Tensor,
    neg_emb: &Tensor,
    img_ids: &Tensor,
    txt_ids: &Tensor,
    h: usize,
    w: usize,
    num_steps: usize,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
    seed: u64,
) -> anyhow::Result<Tensor> {
    // 1. Pixel-space noise.
    let mut x_t = make_pixel_noise(1, h, w, seed, device)?;

    // 2. Sigma schedule with dynamic shift.
    let shift = asymflux2::klein_dynamic_shift(h, w);
    let sigmas = asymflux2::compute_sigma_schedule(num_steps, shift);
    println!(
        "  Sigmas: shift={:.3}, first={:.4} last={:.4} (+ trailing 0)",
        shift,
        sigmas[0],
        sigmas[num_steps - 1]
    );

    let t_total = Instant::now();
    for step in 0..num_steps {
        let sigma_cur = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = Instant::now();

        // Cond forward.
        let u_cond = wrapped_forward(
            &model.transformer,
            &x_t,
            &model.proj_buffer,
            pos_emb,
            sigma_cur,
            model.scale_buffer,
            img_ids,
            txt_ids,
        )?;

        // CFG: uncond forward + orthogonal projection.
        let u_final = if cfg_scale() != 1.0 {
            let u_uncond = wrapped_forward(
                &model.transformer,
                &x_t,
                &model.proj_buffer,
                neg_emb,
                sigma_cur,
                model.scale_buffer,
                img_ids,
                txt_ids,
            )?;
            // denoised = x_t - u_cond * sigma_cur  (in F32; u_cond is F32)
            let scaled = u_cond.mul_scalar(sigma_cur)?;
            let denoised = x_t.sub(&scaled)?;
            let bias = asymflux2::guidance_bias(
                &u_cond,
                &u_uncond,
                cfg_scale(),
                cfg_ortho(),
                &denoised,
            )?;
            u_cond.add(&bias)?
        } else {
            u_cond
        };

        // Optional: clamp through Oklab gamut. Disable with
        // ASYMFLUX2_NO_CLAMP=1 to isolate banding-source bugs.
        let u_for_step = if std::env::var("ASYMFLUX2_NO_CLAMP").ok().as_deref() == Some("1") {
            u_final
        } else {
            asymflux2::clamp_denoised_oklab(&x_t, &u_final, sigma_cur, SIGMA_MIN)?
        };

        // Euler step.
        x_t = asymflux2::euler_step(&x_t, &u_for_step, sigma_cur, sigma_next)?;

        if step == 0 || step == num_steps - 1 || step % 5 == 0 {
            println!(
                "    step {:3}/{} σ {:.4}→{:.4} ({:.2}s)",
                step + 1,
                num_steps,
                sigma_cur,
                sigma_next,
                t_step.elapsed().as_secs_f32()
            );
        }
    }
    println!(
        "  Sampling: {:.1}s ({:.2}s/step)",
        t_total.elapsed().as_secs_f32(),
        t_total.elapsed().as_secs_f32() / num_steps as f32
    );

    Ok(x_t)
}

// --------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------- //

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // Disable autograd globally for inference. Matches HiDream-O1 fix
    // (memory: project_hidream_o1_2026-05-09.md) — without this, the
    // multi-step compute graph is retained across steps and we OOM.
    flame_core::autograd::AutogradContext::set_enabled(false);

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PROMPTS[0].to_string());
    let num_steps = env_usize("ASYMFLUX2_STEPS", DEFAULT_NUM_STEPS);
    let w = env_usize("ASYMFLUX2_W", 512);
    let h = env_usize("ASYMFLUX2_H", 512);
    if h % PATCH_SIZE != 0 || w % PATCH_SIZE != 0 {
        anyhow::bail!(
            "H={} W={} must be multiples of patch_size={}",
            h,
            w,
            PATCH_SIZE
        );
    }
    let adapter_path = env_string("ASYMFLUX2_ADAPTER", DEFAULT_ADAPTER_PATH);
    let base_path = env_string("ASYMFLUX2_BASE", DEFAULT_BASE_PATH);

    println!("============================================================");
    println!("AsymFLUX.2 Klein 9B — pure Rust (pixel space + Oklab + AsymFlow)");
    println!(
        "  {}x{}, {} steps, g={}, ortho={}, seed={}",
        w, h, num_steps, cfg_scale(), cfg_ortho(), SEED
    );
    println!("  Prompt: {}", &prompt[..prompt.len().min(120)]);
    println!("============================================================");

    let device = global_cuda_device();

    // Stage 1: text encode.
    println!("\n--- Stage 1: Text encoding ---");
    let (pos_emb, neg_emb) = encode_text(&prompt, DEFAULT_NEG_PROMPT, &device)?;

    // Stage 2: adapter load.
    println!("\n--- Stage 2: Adapter load ---");
    let model = load_adapter(&adapter_path, &base_path)?;

    // Stage 3: img_ids / txt_ids.
    let h_p = h / PATCH_SIZE;
    let w_p = w / PATCH_SIZE;
    let n_img = h_p * w_p;
    let img_ids = make_img_ids(h_p, w_p, device.clone())?;
    let txt_ids = Tensor::zeros_dtype(Shape::from_dims(&[512, 4]), DType::BF16, device.clone())?;
    println!(
        "  patches: {h_p}x{w_p}={} ({}px image / {}px patch)",
        n_img, h, PATCH_SIZE
    );

    // Stage 4: sample.
    println!("\n--- Stage 3: Sampling ---");
    let x_t_final = sample(
        &model,
        &pos_emb,
        &neg_emb,
        &img_ids,
        &txt_ids,
        h,
        w,
        num_steps,
        device,
        SEED,
    )?;

    // Stage 5: Oklab decode and save.
    println!("\n--- Stage 4: Oklab decode + save ---");
    let t0 = Instant::now();
    let oklab_host = x_t_final.to_vec()?;
    let mut srgb = vec![0.0_f32; oklab_host.len()];
    inference_flame::vae::oklab::decode_planar(&oklab_host, &mut srgb);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let out_path = OUTPUT_PATH_TEMPLATE.replace("{}", &timestamp.to_string());
    save_png_planar(&srgb, h, w, &out_path)?;
    println!("  decoded + saved in {:.1}s", t0.elapsed().as_secs_f32());

    println!("\n============================================================");
    println!("IMAGE SAVED: {}", out_path);
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}
