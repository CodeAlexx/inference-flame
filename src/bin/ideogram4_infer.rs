//! Ideogram-4 — denoise (dual-transformer asymmetric CFG) + VAE decode → PNG.
//!
//! Loads the `llm_features` cache from `ideogram4_encode`, the TWO resident FP8
//! transformers (conditional `transformer/` + unconditional
//! `unconditional_transformer/`), and the Flux2 32-ch VAE, then runs the
//! logit-normal Euler flow-matching denoise loop with the oracle's **asymmetric**
//! CFG and decodes. Mirrors `boogu_infer` / `klein9b_infer`.
//!
//! Usage:
//!   ideogram4_infer [--size N] [--steps N] [--seed N] [--preset NAME]
//!                   [--noise-file P] [--latent-out P]
//!
//! Defaults: `--size 512 --steps 20 --seed 0 --preset V4_DEFAULT_20`. Start at
//! 512 for the first correctness run (two 9.3B FP8 DiTs resident dominate VRAM).
//!
//! ## Denoise (verbatim `pipeline_ideogram4.py.__call__:587-615`)
//!
//! - schedule = `get_schedule_for_resolution((H,W), known_mean=mu, std=std)`
//! - `step_intervals = make_step_intervals(N)`  (N+1 points)
//! - `z = randn(1, num_image_tokens, 128)` (f32); `text_z_padding = zeros(1, max_text, 128)`
//! - for i in (N-1 ..= 0):
//!     `t_val = schedule(step_intervals[i+1])`,  `s_val = schedule(step_intervals[i])`
//!     `pos_z = cat([text_z_padding, z])`                                  # [1, L, 128]
//!     `pos_out = cond(llm_features, pos_z, t, pos_ids, seg, ind)`         # full seq
//!     `pos_v = pos_out[:, max_text:]`                                     # image span
//!     `neg_v = uncond(0·llm, z, t, pos_ids[max_text:], seg[max_text:], ind[max_text:])`
//!     `gw = gw_per_step[i]`            (loop-INDEX order, idx0 = LAST step / polish)
//!     `v  = gw·pos_v + (1-gw)·neg_v`   (asymmetric CFG)
//!     `z  = z + v·(s_val - t_val)`     (Euler flow step)
//!
//! The uncond branch is image-ONLY with **zeroed** llm_features (per spec). The
//! latent `z` is held in F32 across the loop; the DiT casts to BF16 (RNE)
//! internally and the F32 velocity is upcast for the Euler update.
//!
//! ## VAE (verbatim `_decode:619-637`)
//!
//! `z = z·LATENT_SCALE + LATENT_SHIFT` (128-vec f32 constants from
//! `latent_norm.py`, NOT the checkpoint's BF16-rounded bn stats — they DIFFER),
//! then unpatchify `view(B,gh,gw,2,2,32) → permute(0,5,1,3,2,4) → view(B,32,gh·2,gw·2)`,
//! then `autoencoder.decoder(z)` (= `KleinVaeDecoder::decode_normalized`, body
//! only), then `clamp(-1,1) → (x+1)·127.5 → uint8`.
//!
//! Pure-Rust runtime — no Python. autograd OFF. No Flash Attention (head_dim 256
//! falls to the SDPA math/streaming fallback inside the DiT — correct, slow).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaDevice;

use flame_core::serialization::load_file;
use flame_core::{DType, Shape, Tensor};

use inference_flame::models::ideogram4::inputs::{build_inputs, is_mask_all_true};
use inference_flame::models::ideogram4::loader::load_transformer;
use inference_flame::models::ideogram4::scheduler::{
    get_schedule_for_resolution, make_step_intervals, preset,
};
use inference_flame::models::ideogram4::transformer::transformer_forward;
use inference_flame::models::ideogram4::Ideogram4Config;
use inference_flame::vae::klein_vae::KleinVaeDecoder;

const REPO: &str = "/home/alex/.serenity/models/ideogram-4-fp8";
const COND_SUBDIR: &str = "transformer";
const UNCOND_SUBDIR: &str = "unconditional_transformer";
/// VAE: the checkpoint's own `vae/` (diffusers-keyed Flux2 32-ch).
const VAE_PATH: &str = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors";
const EMBED_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/ideogram4_embeddings.safetensors";
const OUTPUT_PATH: &str = "/home/alex/EriDiffusion/inference-flame/output/ideogram4_infer.png";

/// in_channels after patchify (ae 32 × 2²).
const IN_CHANNELS: usize = 128;
/// patch_size × ae_scale_factor = 2 × 8 = 16 (px per image token side).
const PATCH: usize = 16;
/// VAE spatial channels (ae z_channels).
const AE_CH: usize = 32;
/// patch_size (the 2×2 unpatchify factor).
const PATCH2: usize = 2;

const DEFAULT_SIZE: usize = 512;
const DEFAULT_STEPS: usize = 20;
const DEFAULT_SEED: u64 = 0;
const DEFAULT_PRESET: &str = "V4_DEFAULT_20";

/// Latent normalization shift, 128-vec f32 (`latent_norm.py::LATENT_SHIFT`).
const LATENT_SHIFT: [f32; 128] = [
    0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549, 0.05021099,
    -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.0622626, 0.10140969, -0.06739428,
    0.3758261, -0.233712, 0.35164491, -0.02590912, -0.0271935, -0.10833897, -0.1476848,
    -0.01130957, -0.2298372, 0.23526423, -0.10893522, 0.11957631, 0.04047799, 0.3134589,
    -0.17225064, -0.18646109, -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294,
    0.26952152, -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.1548502, -0.18971131,
    0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937, 0.35705471,
    -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137, -0.21863696,
    0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594, -0.17420591, -0.19006285,
    -0.34905377, -0.04025005, 0.01924137, 0.07652984, 0.2995608, 0.2628057, -0.22011674,
    -0.12715361, 0.04879879, -0.14075719, -0.15935895, -0.2123584, 0.01974813, 0.05523547,
    0.10011992, -0.06428964, 0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.0310082,
    -0.10761415, -0.14730405, -0.02475182, -0.2285588, 0.2515081, -0.10445128, 0.12446,
    0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177, 0.02578168,
    0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155, 0.04059549, -0.14043529,
    -0.16111187, -0.2020305, 0.02602069, 0.04852717, 0.10432153, -0.06309942, 0.38402443,
    -0.22397003, 0.34814481, -0.03774432, -0.03381438, -0.11245691, -0.14128767,
    -0.02853208, -0.21752016, 0.24872463, -0.11399775, 0.1222687, 0.05620835, 0.309178,
    -0.18065738, -0.19401479, -0.34495114, -0.01760592,
];

/// Latent normalization scale, 128-vec f32 (`latent_norm.py::LATENT_SCALE`).
const LATENT_SCALE: [f32; 128] = [
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584, 1.56853198,
    1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993, 1.63322129, 1.56074359,
    1.73419528, 1.7919265, 1.64040632, 1.66802808, 1.60390303, 1.75480492, 1.63187587,
    1.64334594, 1.61722884, 1.60146046, 1.63459219, 1.55291476, 1.68771497, 1.68415657,
    1.78966054, 1.66631641, 1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756,
    1.91310663, 1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
    1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606, 1.63732541,
    1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667, 1.61659342, 1.59722044,
    1.64103121, 1.5408531, 1.68610394, 1.67772755, 1.78998563, 1.66621713, 1.65458955,
    1.66041308, 1.64710857, 1.68163503, 1.74000294, 1.92784786, 1.67411194, 1.67395548,
    1.57406532, 1.62199356, 1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305,
    1.55644029, 1.73124302, 1.80666627, 1.6463621, 1.65932006, 1.60816188, 1.75682671,
    1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068, 1.65534289,
    1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259, 1.65339716, 1.67540638,
    1.73298504, 1.94067348, 1.67893609, 1.70635117, 1.5730906, 1.61928553, 1.87148809,
    1.56244866, 1.56697152, 1.61584394, 1.62759496, 1.55480378, 1.73484107, 1.79055143,
    1.64688773, 1.66121492, 1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592,
    1.60792883, 1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
    1.65655173, 1.68533454,
];

struct Args {
    size: usize,
    steps: usize,
    seed: u64,
    preset: String,
    noise_file: Option<String>,
    latent_out: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        size: DEFAULT_SIZE,
        steps: DEFAULT_STEPS,
        seed: DEFAULT_SEED,
        preset: DEFAULT_PRESET.to_string(),
        noise_file: None,
        latent_out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--size" => a.size = it.next().ok_or_else(|| anyhow!("--size needs a value"))?.parse()?,
            "--steps" => {
                a.steps = it.next().ok_or_else(|| anyhow!("--steps needs a value"))?.parse()?
            }
            "--seed" => a.seed = it.next().ok_or_else(|| anyhow!("--seed needs a value"))?.parse()?,
            "--preset" => {
                a.preset = it.next().ok_or_else(|| anyhow!("--preset needs a value"))?
            }
            "--noise-file" => {
                a.noise_file = Some(it.next().ok_or_else(|| anyhow!("--noise-file needs a path"))?)
            }
            "--latent-out" => {
                a.latent_out = Some(it.next().ok_or_else(|| anyhow!("--latent-out needs a path"))?)
            }
            other => return Err(anyhow!("unknown arg `{other}`")),
        }
    }
    if a.size % PATCH != 0 {
        return Err(anyhow!("--size {} must be divisible by {PATCH}", a.size));
    }
    Ok(a)
}

/// Seeded Box-Muller Gaussian noise (F32) — same idiom as `boogu_infer`.
fn seeded_noise_f32(numel: usize, seed: u64) -> Vec<f32> {
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
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
}

fn main() -> Result<()> {
    env_logger::init();
    // INFERENCE: autograd OFF for the whole run. Without this the global autograd
    // context records every op of the N-step × 2-DiT denoise loop into the
    // thread-local graph, pinning the two ~9.3 GB FP8 DiTs alive past their drop
    // and OOMing the VAE decode (see boogu_infer for the measured failure mode).
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let t_total = Instant::now();
    let args = parse_args()?;

    let cfg = Ideogram4Config::default();
    let height = args.size;
    let width = args.size;
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    let num_image_tokens = grid_h * grid_w;

    let preset = preset(&args.preset)
        .ok_or_else(|| anyhow!("unknown preset `{}` (V4_QUALITY_48|V4_DEFAULT_20|V4_TURBO_12)", args.preset))?;
    preset.validate().map_err(|e| anyhow!("preset invalid: {e}"))?;
    let num_steps = if args.steps != 0 { args.steps } else { preset.num_steps };
    if num_steps != preset.guidance_schedule.len() {
        return Err(anyhow!(
            "--steps {num_steps} != preset `{}` guidance_schedule len {} (use the preset's num_steps)",
            args.preset,
            preset.guidance_schedule.len()
        ));
    }

    println!("============================================================");
    println!("Ideogram-4 — Inference (dual-DiT asymmetric CFG + VAE decode)");
    println!("============================================================");
    println!(
        "  size={height}x{width} grid={grid_h}x{grid_w} img_tokens={num_image_tokens} \
         steps={num_steps} seed={} preset={} (mu={} std={})",
        args.seed, args.preset, preset.mu, preset.std
    );

    let device: Arc<CudaDevice> = CudaDevice::new(0).context("cuda device 0")?;

    // --- Load llm_features cache ---
    println!("\n--- Loading llm_features cache ---  src={EMBED_PATH}");
    let embed_map = load_file(Path::new(EMBED_PATH), &device)
        .map_err(|e| anyhow!("load {EMBED_PATH}: {e}"))?;
    let llm_features_text = embed_map
        .get("llm_features")
        .ok_or_else(|| anyhow!("cache missing `llm_features` (run ideogram4_encode first)"))?
        .to_dtype(DType::BF16)
        .map_err(|e| anyhow!("llm_features->bf16: {e}"))?;
    let lf_dims = llm_features_text.shape().dims().to_vec();
    if lf_dims.len() != 3 || lf_dims[0] != 1 || lf_dims[2] != cfg.llm_features_dim {
        return Err(anyhow!(
            "llm_features must be [1,L,{}], got {lf_dims:?}",
            cfg.llm_features_dim
        ));
    }
    let max_text_tokens = lf_dims[1];
    println!("  llm_features (text) {lf_dims:?}  (max_text_tokens={max_text_tokens})");

    // --- Build the packed sequence (positions / segment / indicator) ---
    // The token ids themselves are unused by the DiT (it consumes llm_features);
    // pass a dummy prompt of length max_text_tokens so the layout is correct.
    let dummy_tokens = vec![0i64; max_text_tokens];
    let inputs = build_inputs(&dummy_tokens, height, width, &cfg)
        .map_err(|e| anyhow!("build_inputs: {e}"))?;
    if inputs.num_image_tokens != num_image_tokens || inputs.seq_len != max_text_tokens + num_image_tokens {
        return Err(anyhow!(
            "input layout mismatch: img_tokens {} seq_len {} vs expected {} / {}",
            inputs.num_image_tokens,
            inputs.seq_len,
            num_image_tokens,
            max_text_tokens + num_image_tokens
        ));
    }
    // Build the FULL-sequence llm_features [1, L=max_text+num_image, 53248]:
    // text features in the text span [0..max_text], zeros over the image span.
    // This is numerically the oracle's `_encode_text` output — the encoder's
    // attention_mask masks image positions out, so the text rows are identical
    // to a text-only encode, and `text_mask` zeros every non-LLM position. The
    // DiT's `llm_token_mask` then zeros the image span again (redundant but
    // harmless), so what matters is the text rows land at [0..max_text].
    let llm_zeros_img = Tensor::zeros_dtype(
        Shape::from_dims(&[1, num_image_tokens, cfg.llm_features_dim]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| anyhow!("llm img zeros: {e}"))?;
    let llm_features = Tensor::cat(&[&llm_features_text, &llm_zeros_img], 1)
        .map_err(|e| anyhow!("cat full llm_features: {e}"))?
        .contiguous()
        .map_err(|e| anyhow!("contig full llm_features: {e}"))?;
    drop(llm_zeros_img);
    println!(
        "  llm_features (full) {:?}",
        llm_features.shape().dims()
    );

    // Full-sequence indicator as i32 (transformer_forward wants &[i32]).
    let indicator_full: Vec<i32> = inputs.indicator.iter().map(|&x| x as i32).collect();
    // Image-only slice (uncond branch is image tokens only).
    let img_pos_t: Vec<u32> = inputs.pos_t[max_text_tokens..].to_vec();
    let img_pos_h: Vec<u32> = inputs.pos_h[max_text_tokens..].to_vec();
    let img_pos_w: Vec<u32> = inputs.pos_w[max_text_tokens..].to_vec();
    let img_indicator: Vec<i32> = indicator_full[max_text_tokens..].to_vec();

    // B=1 unpadded → segment mask all-True → SDPA mask = None (both branches).
    let mask_full = if is_mask_all_true(&inputs.segment_ids) {
        None
    } else {
        return Err(anyhow!(
            "B=1 unpadded expected an all-True segment mask; got a padded layout (unsupported)"
        ));
    };
    let mask_img = mask_full; // image-only span is also single-segment.

    // --- Load BOTH FP8 transformers resident ---
    println!("\n--- Loading conditional transformer (FP8 resident) ---");
    let t0 = Instant::now();
    let cond_w = load_transformer(Path::new(REPO), COND_SUBDIR, &device, &cfg)
        .map_err(|e| anyhow!("load conditional: {e}"))?;
    println!("  conditional resident in {:.1}s ({} weights)", t0.elapsed().as_secs_f32(), cond_w.len());

    println!("\n--- Loading unconditional transformer (FP8 resident) ---");
    let t0 = Instant::now();
    let uncond_w = load_transformer(Path::new(REPO), UNCOND_SUBDIR, &device, &cfg)
        .map_err(|e| anyhow!("load unconditional: {e}"))?;
    println!("  unconditional resident in {:.1}s ({} weights)", t0.elapsed().as_secs_f32(), uncond_w.len());

    // --- Zeroed uncond llm_features ([1, num_image_tokens, 53248]) ---
    let neg_llm = Tensor::zeros_dtype(
        Shape::from_dims(&[1, num_image_tokens, cfg.llm_features_dim]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| anyhow!("neg_llm zeros: {e}"))?;

    // --- text_z_padding ([1, max_text_tokens, 128]) F32 ---
    let text_z_padding = Tensor::zeros_dtype(
        Shape::from_dims(&[1, max_text_tokens, IN_CHANNELS]),
        DType::F32,
        device.clone(),
    )
    .map_err(|e| anyhow!("text_z_padding: {e}"))?;

    // --- Init noise z ([1, num_image_tokens, 128] F32) ---
    let numel = num_image_tokens * IN_CHANNELS;
    let noise = if let Some(npath) = args.noise_file.as_deref() {
        let nmap = load_file(Path::new(npath), &device).map_err(|e| anyhow!("load noise {npath}: {e}"))?;
        let nt = nmap
            .get("tensor")
            .ok_or_else(|| anyhow!("noise file {npath} missing 'tensor'"))?
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("noise->f32: {e}"))?;
        let nv = nt.to_vec_f32().map_err(|e| anyhow!("noise to vec: {e}"))?;
        if nv.len() != numel {
            return Err(anyhow!("noise numel {} != expected {numel}", nv.len()));
        }
        println!("  initial latent: LOADED from {npath} ({numel} F32, parity)");
        nv
    } else {
        println!("  initial latent: SEEDED Box-Muller (seed {})", args.seed);
        seeded_noise_f32(numel, args.seed)
    };
    let mut z = Tensor::from_vec(
        noise,
        Shape::from_dims(&[1, num_image_tokens, IN_CHANNELS]),
        device.clone(),
    )
    .map_err(|e| anyhow!("init z: {e}"))?; // F32

    // --- Schedule ---
    let schedule = get_schedule_for_resolution(height, width, preset.mu, preset.std);
    let step_intervals = make_step_intervals(num_steps); // N+1 f32 points
    let gw = &preset.guidance_schedule; // loop-INDEX order (idx0 = LAST step)

    println!(
        "\n--- Denoise ({num_steps} steps, dual-DiT) ---  head_dim={} (SDPA math fallback)",
        cfg.head_dim()
    );
    let t_loop = Instant::now();
    let mut step_times = Vec::with_capacity(num_steps);

    // Loop i = N-1 .. 0 (reverse), reading gw[i] in loop-INDEX order.
    for i in (0..num_steps).rev() {
        let t_step = Instant::now();
        let t_val = schedule.eval(step_intervals[i + 1] as f64);
        let s_val = schedule.eval(step_intervals[i] as f64);
        let delta = s_val - t_val;

        // pos_z = cat([text_z_padding, z]) along seq dim → [1, L, 128].
        let pos_z = Tensor::cat(&[&text_z_padding, &z], 1)
            .map_err(|e| anyhow!("cat pos_z (step {i}): {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("contig pos_z (step {i}): {e}"))?;

        // Conditional (full sequence) → velocity, take the image span.
        let pos_out = transformer_forward(
            &cond_w,
            &cfg,
            &pos_z,
            &[t_val],
            &llm_features,
            &inputs.pos_t,
            &inputs.pos_h,
            &inputs.pos_w,
            mask_full,
            &indicator_full,
            &device,
        )
        .map_err(|e| anyhow!("cond forward (step {i}): {e}"))?;
        let pos_v = pos_out
            .narrow(1, max_text_tokens, num_image_tokens)
            .map_err(|e| anyhow!("slice pos_v (step {i}): {e}"))?;

        // Unconditional (image-only seq, zeroed llm_features) → velocity.
        let neg_v = transformer_forward(
            &uncond_w,
            &cfg,
            &z,
            &[t_val],
            &neg_llm,
            &img_pos_t,
            &img_pos_h,
            &img_pos_w,
            mask_img,
            &img_indicator,
            &device,
        )
        .map_err(|e| anyhow!("uncond forward (step {i}): {e}"))?;

        // Asymmetric CFG: v = gw·pos_v + (1-gw)·neg_v   (both F32 from forward).
        let gw_i = gw[i];
        let pos_v_f32 = pos_v.to_dtype(DType::F32).map_err(|e| anyhow!("pos_v->f32: {e}"))?;
        let neg_v_f32 = neg_v.to_dtype(DType::F32).map_err(|e| anyhow!("neg_v->f32: {e}"))?;
        let v = pos_v_f32
            .mul_scalar(gw_i)
            .map_err(|e| anyhow!("gw·pos (step {i}): {e}"))?
            .add(&neg_v_f32.mul_scalar(1.0 - gw_i).map_err(|e| anyhow!("(1-gw)·neg: {e}"))?)
            .map_err(|e| anyhow!("cfg combine (step {i}): {e}"))?;

        // Euler flow step: z = z + v·(s_val - t_val).
        z = z
            .add(&v.mul_scalar(delta).map_err(|e| anyhow!("v·delta (step {i}): {e}"))?)
            .map_err(|e| anyhow!("euler (step {i}): {e}"))?;

        let st = t_step.elapsed().as_secs_f32();
        step_times.push(st);
        println!(
            "  step {:2}  i={:2} gw={:.1}  t={:.5} -> s={:.5} (Δ={:.5})  {:.2}s",
            num_steps - i,
            i,
            gw_i,
            t_val,
            s_val,
            delta,
            st
        );
    }
    let loop_s = t_loop.elapsed().as_secs_f32();
    let avg = step_times.iter().sum::<f32>() / step_times.len().max(1) as f32;
    println!("  denoise total {:.1}s, avg {:.2}s/step (2× DiT/step)", loop_s, avg);

    // PARITY: dump final pre-decode F32 latent (key `tensor`) for cos compare.
    if let Some(lpath) = args.latent_out.as_deref() {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("tensor".to_string(), z.to_dtype(DType::F32).map_err(|e| anyhow!("latent_out->f32: {e}"))?);
        if let Some(parent) = Path::new(lpath).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        flame_core::serialization::save_file(&m, lpath).map_err(|e| anyhow!("save latent_out: {e}"))?;
        println!("  PARITY: dumped final latent -> {lpath}");
    }

    // --- Evict DiTs before VAE (they dominate VRAM) ---
    drop(cond_w);
    drop(uncond_w);
    drop(llm_features);
    drop(neg_llm);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    println!("  DiTs evicted (pool cache cleared)");

    // --- Latent rescale + unpatchify (verbatim _decode) ---
    // z: [1, L=gh·gw, 128] F32.  z = z·LATENT_SCALE + LATENT_SHIFT (128-vec).
    let scale_t = Tensor::from_vec(
        LATENT_SCALE.to_vec(),
        Shape::from_dims(&[1, 1, IN_CHANNELS]),
        device.clone(),
    )
    .map_err(|e| anyhow!("scale tensor: {e}"))?;
    let shift_t = Tensor::from_vec(
        LATENT_SHIFT.to_vec(),
        Shape::from_dims(&[1, 1, IN_CHANNELS]),
        device.clone(),
    )
    .map_err(|e| anyhow!("shift tensor: {e}"))?;
    let z_norm = z
        .mul(&scale_t)
        .map_err(|e| anyhow!("z·scale: {e}"))?
        .add(&shift_t)
        .map_err(|e| anyhow!("+shift: {e}"))?;
    drop(z);

    // unpatchify: view(1,gh,gw,2,2,32) -> permute(0,5,1,3,2,4) -> view(1,32,gh·2,gw·2)
    let z6 = z_norm
        .reshape(&[1, grid_h, grid_w, PATCH2, PATCH2, AE_CH])
        .map_err(|e| anyhow!("unpatch reshape: {e}"))?;
    let z_perm = z6
        .permute(&[0, 5, 1, 3, 2, 4])
        .map_err(|e| anyhow!("unpatch permute: {e}"))?
        .contiguous()
        .map_err(|e| anyhow!("unpatch contiguous: {e}"))?;
    let z_nchw = z_perm
        .reshape(&[1, AE_CH, grid_h * PATCH2, grid_w * PATCH2])
        .map_err(|e| anyhow!("unpatch nchw reshape: {e}"))?;

    // --- VAE decode (body only — rescale + unpatchify already applied) ---
    println!("\n--- VAE decode ---");
    let t0 = Instant::now();
    let vae_weights = load_file(Path::new(VAE_PATH), &device).map_err(|e| anyhow!("VAE load: {e}"))?;
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device).map_err(|e| anyhow!("VAE build: {e}"))?;
    drop(vae_weights);
    let z_bf16 = z_nchw.to_dtype(DType::BF16).map_err(|e| anyhow!("z->bf16: {e}"))?;
    let rgb = vae
        .decode_normalized(&z_bf16)
        .map_err(|e| anyhow!("VAE decode_normalized: {e}"))?;
    println!("  decoded {:?} in {:.1}s", rgb.shape().dims(), t0.elapsed().as_secs_f32());

    // --- Denormalize [-1,1] -> u8, CHW->HWC, save PNG ---
    let rgb_f32 = rgb.to_dtype(DType::F32).map_err(|e| anyhow!("rgb->f32: {e}"))?;
    let data = rgb_f32.to_vec_f32().map_err(|e| anyhow!("rgb to vec: {e}"))?;
    let d = rgb_f32.shape().dims().to_vec();
    let (out_c, out_h, out_w) = (d[1], d[2], d[3]);
    if out_c != 3 {
        return Err(anyhow!("VAE must return 3 channels, got {out_c}"));
    }
    // Pixel stats (correctness sanity).
    let mut ch_sum = [0f64; 3];
    let mut ch_sq = [0f64; 3];
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                let raw = data[idx];
                ch_sum[c] += raw as f64;
                ch_sq[c] += (raw as f64) * (raw as f64);
                let v = raw.clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * out_w + x) * 3 + c] = u;
            }
        }
    }
    let npix = (out_h * out_w) as f64;
    for c in 0..3 {
        let mean = ch_sum[c] / npix;
        let var = (ch_sq[c] / npix - mean * mean).max(0.0);
        println!("  channel {c}: mean={:.4} std={:.4}", mean, var.sqrt());
    }

    if let Some(parent) = Path::new(OUTPUT_PATH).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow!("failed to build RgbImage"))?
        .save(OUTPUT_PATH)?;

    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {OUTPUT_PATH}  ({out_w}x{out_h})");
    println!("Total time:  {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    let _ = device;
    Ok(())
}
