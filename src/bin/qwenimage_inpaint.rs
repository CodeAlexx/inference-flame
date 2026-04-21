//! Qwen-Image inpainting — pure Rust, flame-core + LanPaint.
//!
//! Mirrors `flux_inpaint.rs` with Qwen-Image-specific bits:
//!   - Text encoder: Qwen2.5-VL-7B (run offline via the existing
//!     `qwenimage_encode` script). This bin loads pre-computed embeddings
//!     from a safetensors file with keys `cond` + `uncond` — same format
//!     `qwenimage_gen.rs` consumes.
//!   - Real CFG with Qwen-specific norm rescale (matches `qwenimage_gen`):
//!     `comb = uncond + scale * (cond - uncond);  out = comb * (‖cond‖/‖comb‖)`.
//!   - VAE: Qwen-Image 3D VAE with a *single* time frame. Input is
//!     `[1, 3, 1, H, W]`; output latent is `[1, 16, 1, H/8, W/8]`. Inside
//!     this bin we keep the LanPaint state in 4D NCHW `[1, 16, H/8, W/8]`
//!     by squeezing/unsqueezing the time axis at VAE boundaries.
//!   - DiT: 60-layer QwenImageDit, packed-latent format
//!     `[1, (H/16)*(W/16), 64]` (16 ch × 2×2 patch). Pack/unpack lives
//!     inside the velocity closure.
//!   - Schedule: dynamic exponential shift with terminal stretch (matches
//!     `qwenimage_gen.rs`).
//!
//! CLI:
//!   qwenimage_inpaint --prompt "..." \
//!                     --input-image input.png --mask mask.png \
//!                     --embeddings cached.safetensors \
//!                     [--output-path inpaint_out.png]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use cudarc::driver::CudaDevice;
use flame_core::serialization::load_file;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

use lanpaint_flame::{LanPaint, LanPaintConfig};

use inference_flame::inpaint::{blend_output, lanpaint_step, prepare_inpaint, InpaintConfig};
use inference_flame::models::qwenimage_dit::QwenImageDit;
use inference_flame::vae::{QwenImageVaeDecoder, QwenImageVaeEncoder};

// ---------------------------------------------------------------------------
// Defaults — match qwenimage_gen.rs
// ---------------------------------------------------------------------------

const DEFAULT_DIT_SHARDS: &[&str] = &[
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
];
const DEFAULT_VAE: &str = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors";

const DEFAULT_OUTPUT: &str =
    "/home/alex/EriDiffusion/inference-flame/output/qwenimage_inpaint.png";
const DEFAULT_PROMPT: &str = "a photograph of a sleeping cat";

const DEFAULT_SEED: u64 = 42;
const DEFAULT_STEPS: usize = 50;
const DEFAULT_CFG: f32 = 4.0;

// VAE: 8× downscale (2^3); patch_size=2 lives inside the DiT pack.
const VAE_DOWNSCALE: usize = 8;
const PATCH_SIZE: usize = 2;
const IN_CHANNELS: usize = 16;
const PACKED_CHANNELS: usize = 64; // 16 * 2 * 2

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    prompt: String,
    input_image: PathBuf,
    mask: PathBuf,
    output_path: PathBuf,
    embeddings_path: PathBuf,
    vae_path: PathBuf,
    dit_shards: Vec<PathBuf>,
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
}

fn print_help() {
    println!(
        "qwenimage_inpaint — Qwen-Image inpainting via LanPaint\n\
         \n\
         USAGE:\n  \
           qwenimage_inpaint --prompt <TEXT> --input-image <PATH> --mask <PATH> \\\n  \
                             --embeddings <PATH> [OPTIONS]\n\
         \n\
         REQUIRED:\n  \
           --prompt <TEXT>          Text prompt (informational; embeddings are pre-computed)\n  \
           --input-image <PATH>     Input image (PNG/JPG/WEBP)\n  \
           --mask <PATH>            Mask image (white=inpaint, black=preserve)\n  \
           --embeddings <PATH>      Pre-computed safetensors with 'cond' + 'uncond' keys\n  \
                                    (run scripts/qwenimage_encode.py to produce)\n\
         \n\
         OPTIONS:\n  \
           --vae <PATH>             VAE safetensors [default: qwen-image-2512]\n  \
           --dit-shards <a:b:c>     Colon-separated list of DiT shard paths\n  \
                                    [default: 9 shards from qwen-image-2512]\n  \
           --output-path <PATH>     Output PNG path [default: {default_output}]\n  \
           --width <N>              Output width [default: 1024]\n  \
           --height <N>             Output height [default: 1024]\n  \
           --steps <N>              Diffusion steps [default: {steps}]\n  \
           --cfg <F>                Classifier-free guidance scale [default: {cfg}]\n  \
           --seed <N>               RNG seed [default: {seed}]\n  \
           -h, --help               Print this help",
        default_output = DEFAULT_OUTPUT,
        steps = DEFAULT_STEPS,
        cfg = DEFAULT_CFG,
        seed = DEFAULT_SEED,
    );
}

fn parse_cli() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().collect();

    let mut prompt: Option<String> = None;
    let mut input_image: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut output_path: PathBuf = PathBuf::from(DEFAULT_OUTPUT);
    let mut embeddings_path: Option<PathBuf> = None;
    let mut vae_path: PathBuf = PathBuf::from(DEFAULT_VAE);
    let mut dit_shards: Option<Vec<PathBuf>> = None;
    let mut width: usize = 1024;
    let mut height: usize = 1024;
    let mut steps: usize = DEFAULT_STEPS;
    let mut cfg: f32 = DEFAULT_CFG;
    let mut seed: u64 = DEFAULT_SEED;

    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--prompt" => prompt = Some(next_arg(&raw, &mut i, "--prompt")?),
            "--input-image" => {
                input_image = Some(PathBuf::from(next_arg(&raw, &mut i, "--input-image")?));
            }
            "--mask" => mask = Some(PathBuf::from(next_arg(&raw, &mut i, "--mask")?)),
            "--output-path" | "--output" => {
                output_path = PathBuf::from(next_arg(&raw, &mut i, "--output-path")?);
            }
            "--embeddings" => {
                embeddings_path =
                    Some(PathBuf::from(next_arg(&raw, &mut i, "--embeddings")?));
            }
            "--vae" => vae_path = PathBuf::from(next_arg(&raw, &mut i, "--vae")?),
            "--dit-shards" => {
                let s = next_arg(&raw, &mut i, "--dit-shards")?;
                dit_shards = Some(s.split(':').map(PathBuf::from).collect());
            }
            "--width" => {
                width = next_arg(&raw, &mut i, "--width")?
                    .parse()
                    .map_err(|e| anyhow!("--width: {e}"))?;
            }
            "--height" => {
                height = next_arg(&raw, &mut i, "--height")?
                    .parse()
                    .map_err(|e| anyhow!("--height: {e}"))?;
            }
            "--steps" => {
                steps = next_arg(&raw, &mut i, "--steps")?
                    .parse()
                    .map_err(|e| anyhow!("--steps: {e}"))?;
            }
            "--cfg" | "--guidance" => {
                cfg = next_arg(&raw, &mut i, "--cfg")?
                    .parse()
                    .map_err(|e| anyhow!("--cfg: {e}"))?;
            }
            "--seed" => {
                seed = next_arg(&raw, &mut i, "--seed")?
                    .parse()
                    .map_err(|e| anyhow!("--seed: {e}"))?;
            }
            other => {
                return Err(anyhow!("unknown argument: {other}. Use --help for usage."));
            }
        }
        i += 1;
    }

    let prompt = prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let input_image =
        input_image.ok_or_else(|| anyhow!("missing required --input-image. Use --help."))?;
    let mask = mask.ok_or_else(|| anyhow!("missing required --mask. Use --help."))?;
    let embeddings_path =
        embeddings_path.ok_or_else(|| anyhow!("missing required --embeddings. Use --help."))?;
    let dit_shards =
        dit_shards.unwrap_or_else(|| DEFAULT_DIT_SHARDS.iter().map(PathBuf::from).collect());

    Ok(CliArgs {
        prompt,
        input_image,
        mask,
        output_path,
        embeddings_path,
        vae_path,
        dit_shards,
        width,
        height,
        steps,
        cfg,
        seed,
    })
}

fn next_arg(raw: &[String], i: &mut usize, flag: &str) -> Result<String> {
    *i += 1;
    raw.get(*i)
        .cloned()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();

    let args = parse_cli()?;
    let t_total = Instant::now();
    let device = global_cuda_device();

    println!("=== Qwen-Image — pure Rust INPAINT (LanPaint) ===");
    println!("Prompt:       {:?}", args.prompt);
    println!("Input image:  {}", args.input_image.display());
    println!("Mask:         {}", args.mask.display());
    println!("Output:       {}", args.output_path.display());
    println!("Embeddings:   {}", args.embeddings_path.display());
    println!("VAE:          {}", args.vae_path.display());
    println!(
        "Size: {}x{}, steps={}, cfg={}",
        args.width, args.height, args.steps, args.cfg
    );
    println!("Seed: {}", args.seed);
    println!();

    if args.width % VAE_DOWNSCALE != 0 || args.height % VAE_DOWNSCALE != 0 {
        return Err(anyhow!(
            "Qwen-Image requires width/height divisible by {VAE_DOWNSCALE}; got {}x{}",
            args.width,
            args.height
        ));
    }
    if (args.width / VAE_DOWNSCALE) % PATCH_SIZE != 0
        || (args.height / VAE_DOWNSCALE) % PATCH_SIZE != 0
    {
        return Err(anyhow!(
            "Qwen-Image requires (W/{VAE_DOWNSCALE}) and (H/{VAE_DOWNSCALE}) divisible by {PATCH_SIZE} (DiT patchify)"
        ));
    }

    // ------------------------------------------------------------------
    // Stage 1: Load cached cond + uncond embeddings
    // ------------------------------------------------------------------
    println!("--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let tensors = load_file(&args.embeddings_path, &device)
        .map_err(|e| anyhow!("load embeddings: {e:?}"))?;
    let cond = tensors
        .get("cond")
        .ok_or_else(|| anyhow!("embeddings file must contain 'cond' key"))?
        .to_dtype(DType::BF16)?;
    let uncond = tensors
        .get("uncond")
        .ok_or_else(|| anyhow!("embeddings file must contain 'uncond' key"))?
        .to_dtype(DType::BF16)?;
    drop(tensors);
    println!("  cond:   {:?}", cond.shape().dims());
    println!("  uncond: {:?}", uncond.shape().dims());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 2: VAE encode (BEFORE DiT load) — Qwen 3D VAE wants
    // [B, 3, T, H, W]; we feed T=1 then squeeze the time axis on output
    // so prepare_inpaint sees a 4D NCHW latent.
    // ------------------------------------------------------------------
    println!("--- Stage 2: Load VAE encoder + prepare inpaint ---");
    let t0 = Instant::now();
    let inputs = {
        let vae_enc =
            QwenImageVaeEncoder::from_safetensors(args.vae_path.to_str().unwrap(), &device)
                .map_err(|e| anyhow!("VAE encoder load: {e:?}"))?;
        println!("  VAE encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let cfg = InpaintConfig {
            image_path: args.input_image.clone(),
            mask_path: args.mask.clone(),
            // Mask aligns with the latent NCHW grid (H/8 × W/8).
            // Pack/unpack to the DiT's [seq, 64] format happens inside the
            // velocity closure.
            vae_scale: VAE_DOWNSCALE,
            width: args.width,
            height: args.height,
        };
        prepare_inpaint(&cfg, device.clone(), |img_bchw| {
            // img_bchw: [1, 3, H, W] BF16; insert T=1 at dim 2.
            let dims = img_bchw.shape().dims();
            if dims.len() != 4 {
                return Err(anyhow!(
                    "vae_encode closure expected 4D input, got {:?}",
                    dims
                ));
            }
            let img_bcthw = img_bchw.reshape(&[dims[0], dims[1], 1, dims[2], dims[3]])?;
            let lat_bcthw = vae_enc
                .encode(&img_bcthw)
                .map_err(|e| anyhow!("qwen vae encode: {e:?}"))?;
            // lat_bcthw: [B, 16, 1, H/8, W/8] -> squeeze T -> [B, 16, H/8, W/8]
            let ld = lat_bcthw.shape().dims();
            if ld.len() != 5 || ld[2] != 1 {
                return Err(anyhow!(
                    "qwen vae returned unexpected shape {:?}, expected [B, C, 1, H, W]",
                    ld
                ));
            }
            let lat_bchw = lat_bcthw.reshape(&[ld[0], ld[1], ld[3], ld[4]])?;
            Ok(lat_bchw)
        })?
    };
    println!("  latent_image: {:?}", inputs.latent_image.shape().dims());
    println!("  latent_mask:  {:?}", inputs.latent_mask.shape().dims());
    println!("  pixel_mask:   {:?}", inputs.pixel_mask.shape().dims());
    println!("  Inpaint inputs prepared in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 3: Load Qwen-Image DiT (BlockOffloader, 9 shards)
    // ------------------------------------------------------------------
    println!("--- Stage 3: Load Qwen-Image DiT ---");
    let t_load = Instant::now();
    let shard_strs: Vec<String> =
        args.dit_shards.iter().map(|p| p.display().to_string()).collect();
    let shard_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();
    let dit = QwenImageDit::load(&shard_refs, &device)?;
    println!("  DiT loaded in {:.1}s", t_load.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Stage 4: Build noise + denoise (LanPaint inner + Qwen Euler outer)
    // ------------------------------------------------------------------
    println!(
        "--- Stage 4: Denoise ({} steps, CFG={}, LanPaint inner) ---",
        args.steps, args.cfg
    );

    let h_lat = args.height / VAE_DOWNSCALE;
    let w_lat = args.width / VAE_DOWNSCALE;
    let h_patched = h_lat / PATCH_SIZE;
    let w_patched = w_lat / PATCH_SIZE;
    let seq_len = h_patched * w_patched;
    println!(
        "  Latent NCHW = [1, {}, {}, {}], packed seq = {} x {}",
        IN_CHANNELS, h_lat, w_lat, seq_len, PACKED_CHANNELS
    );

    let li_dims = inputs.latent_image.shape().dims();
    if li_dims[1] != IN_CHANNELS || li_dims[2] != h_lat || li_dims[3] != w_lat {
        return Err(anyhow!(
            "latent shape mismatch: VAE gave {:?}, expected [_, {}, {}, {}]",
            li_dims,
            IN_CHANNELS,
            h_lat,
            w_lat
        ));
    }

    // Seeded noise [1, 16, H/8, W/8] BF16.
    let numel = IN_CHANNELS * h_lat * w_lat;
    let noise_data = box_muller_noise(args.seed, numel);
    let noise_nchw = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, IN_CHANNELS, h_lat, w_lat]),
        device.clone(),
    )?;
    let mut x_nchw = noise_nchw.clone_result()?;

    // Schedule: dynamic exponential shift with terminal stretch
    // (matches qwenimage_gen.rs).
    let sigmas = build_qwen_schedule(args.steps, seq_len);
    println!(
        "  sigmas[0]={:.4}  sigmas[1]={:.4}  sigmas[-2]={:.4}  sigmas[-1]={:.4}",
        sigmas[0],
        sigmas[1],
        sigmas[args.steps - 1],
        sigmas[args.steps]
    );

    let lanpaint_cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
        is_flow: true,
    };

    let dit_cell = RefCell::new(dit);
    let cond_ref = &cond;
    let uncond_ref = &uncond;
    let cfg_scale = args.cfg;
    let frame = 1usize;
    let h_p = h_patched;
    let w_p = w_patched;
    let h_l = h_lat;
    let w_l = w_lat;

    // One Qwen forward — packs NCHW input via diffusers-style _pack_latents,
    // runs cond + uncond, applies norm-rescaled CFG, unpacks back to NCHW.
    let qwen_velocity = |x_nchw_in: &Tensor, t_vec: &Tensor| -> Result<Tensor> {
        let x_packed = pack_qwen_nchw(x_nchw_in, h_p, w_p)?;
        let mut d = dit_cell.borrow_mut();
        let pred_cond = d.forward(&x_packed, cond_ref, t_vec, (frame, h_p, w_p))?;
        let pred_uncond = d.forward(&x_packed, uncond_ref, t_vec, (frame, h_p, w_p))?;
        drop(d);
        // True CFG: comb = uncond + scale * (cond - uncond)
        let diff = pred_cond.sub(&pred_uncond)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        let comb = pred_uncond.add(&scaled)?;
        // Norm-rescale CFG (Qwen-specific): out = comb * (‖cond‖_2 / ‖comb‖_2)
        let pred_packed = norm_rescale_cfg(&pred_cond, &comb).unwrap_or(comb);
        unpack_qwen_to_nchw(&pred_packed, h_l, w_l)
    };

    let t_denoise = Instant::now();
    for step in 0..args.steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];

        // Flow-matching scalars (sigma == t).
        let flow_t = sigma_curr;
        let abt_val = {
            let one_minus = 1.0 - flow_t;
            let denom = one_minus * one_minus + flow_t * flow_t;
            if denom > 0.0 {
                (one_minus * one_minus) / denom
            } else {
                1.0
            }
        };
        let ve_sigma_val = if (1.0 - flow_t).abs() > 1e-6 {
            flow_t / (1.0 - flow_t)
        } else {
            1.0e6
        };
        let sigma_scalar = make_b1_bf16(ve_sigma_val, &device)?;
        let abt_scalar = make_b1_bf16(abt_val, &device)?;
        let tflow_scalar = make_b1_bf16(flow_t, &device)?;
        let t_vec_step = make_b1_bf16(flow_t, &device)?;

        // LanPaint inner Langevin loop.
        // Velocity convention: model output IS velocity; x_0 = x - t * v.
        let advanced_x = {
            let inner_model_fn = |x: &Tensor, t: &Tensor| -> flame_core::Result<Tensor> {
                let v = qwen_velocity(x, t).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!("qwen inner: {e:?}"))
                })?;
                let x_f32 = x.to_dtype(DType::F32)?;
                let v_f32 = v.to_dtype(DType::F32)?;
                let t_f32 = t.to_dtype(DType::F32)?;
                let img_dim = x.shape().dims().len();
                let b = t.shape().dims()[0];
                let mut tdims = vec![b];
                tdims.extend(std::iter::repeat(1).take(img_dim - 1));
                let t_b = t_f32.reshape(&tdims)?;
                let x0 = x_f32.sub(&t_b.mul(&v_f32)?)?;
                x0.to_dtype(x.dtype())
            };

            let lanpaint = LanPaint::new(
                LanPaintConfig {
                    n_steps: lanpaint_cfg.n_steps,
                    lambda_: lanpaint_cfg.lambda_,
                    friction: lanpaint_cfg.friction,
                    beta: lanpaint_cfg.beta,
                    step_size: lanpaint_cfg.step_size,
                    is_flow: lanpaint_cfg.is_flow,
                },
                Box::new(inner_model_fn),
            );

            let (_lp_x0, advanced_x) = lanpaint_step(
                &lanpaint,
                &x_nchw,
                &inputs,
                &noise_nchw,
                &sigma_scalar,
                &abt_scalar,
                &tflow_scalar,
            )?;
            drop(lanpaint);
            advanced_x
        };

        // Qwen Euler step: x_next = x + (sigma_next - sigma_curr) * v
        let v_nchw = qwen_velocity(&advanced_x, &t_vec_step)?;
        let dt = sigma_next - sigma_curr;
        let next_x = advanced_x.add(&v_nchw.mul_scalar(dt)?)?;

        // Mask blend on known region.
        let nf = sigma_next;
        let one_minus_nf = 1.0 - nf;
        let scaled_image = inputs.latent_image.mul_scalar(one_minus_nf)?;
        let scaled_noise = noise_nchw.mul_scalar(nf)?;
        let noisy_image = scaled_image.add(&scaled_noise)?;
        let blended = Tensor::where_mask(&inputs.latent_mask, &noisy_image, &next_x)?;
        x_nchw = blended;

        if step == 0 || step + 1 == args.steps || (step + 1) % 5 == 0 {
            println!(
                "  step {}/{}  sigma={:.4} -> {:.4}  ({:.1}s elapsed)",
                step + 1,
                args.steps,
                sigma_curr,
                sigma_next,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!(
        "  Denoised in {:.1}s ({:.2}s/step, 2 forwards/step + LanPaint inner)",
        dt_denoise,
        dt_denoise / args.steps as f32
    );
    println!();

    drop(dit_cell);
    drop(cond);
    drop(uncond);
    println!("  DiT + cached embeddings evicted");

    // ------------------------------------------------------------------
    // Stage 5: VAE decode + pixel-space blend
    // ------------------------------------------------------------------
    println!("--- Stage 5: VAE decode ---");
    let t0 = Instant::now();
    let decoder =
        QwenImageVaeDecoder::from_safetensors(args.vae_path.to_str().unwrap(), &device)?;
    println!("  VAE decoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Decoder expects [B, 16, T, H, W] — re-insert T=1.
    let x_bcthw = x_nchw.reshape(&[1, IN_CHANNELS, 1, h_lat, w_lat])?;
    let rgb_bcthw = decoder.decode(&x_bcthw)?; // [1, 3, 1, H, W]
    drop(x_nchw);
    drop(decoder);
    let rd = rgb_bcthw.shape().dims();
    println!("  Decoded: {:?}", rd);

    // [1, 3, 1, H, W] -> [3, H, W]
    let rgb_3chw = rgb_bcthw
        .narrow(2, 0, 1)?
        .reshape(&[1, 3, args.height, args.width])?
        .narrow(0, 0, 1)?
        .reshape(&[3, args.height, args.width])?;
    let rgb_blended = blend_output(&rgb_3chw, &inputs.input_image, &inputs.pixel_mask)?;
    println!("  Blended pixel output: {:?}", rgb_blended.shape().dims());
    println!();

    // ------------------------------------------------------------------
    // Stage 6: Save PNG
    // ------------------------------------------------------------------
    println!("--- Stage 6: Save PNG ---");
    save_chw_f32_to_png(&rgb_blended, args.height, args.width, &args.output_path)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!();
    println!("============================================================");
    println!("IMAGE SAVED: {}", args.output_path.display());
    println!("Total time:  {:.1}s", dt_total);
    println!("============================================================");

    let _ = device;
    Ok(())
}

// ---------------------------------------------------------------------------
// Pack / unpack — diffusers-style _pack_latents
// ---------------------------------------------------------------------------

/// Pack `[1, 16, H_lat, W_lat]` BF16 → `[1, h_p*w_p, 64]` BF16.
///
/// Mirrors `pipeline_qwenimage._pack_latents`:
///   view as [B, C, H/2, 2, W/2, 2], permute to [B, H/2, W/2, C, 2, 2],
///   reshape to [B, (H/2)(W/2), C*4].
fn pack_qwen_nchw(x_nchw: &Tensor, h_p: usize, w_p: usize) -> Result<Tensor> {
    let dims = x_nchw.shape().dims();
    let b = dims[0];
    let c = dims[1];
    let h = dims[2];
    let w = dims[3];
    if h != h_p * PATCH_SIZE || w != w_p * PATCH_SIZE {
        return Err(anyhow!(
            "pack_qwen_nchw: shape {:?} not divisible into {}x{} patches of size {}",
            dims,
            h_p,
            w_p,
            PATCH_SIZE
        ));
    }
    let packed = x_nchw
        .reshape(&[b, c, h_p, PATCH_SIZE, w_p, PATCH_SIZE])?
        .permute(&[0, 2, 4, 1, 3, 5])?
        .reshape(&[b, h_p * w_p, c * PATCH_SIZE * PATCH_SIZE])?;
    Ok(packed)
}

/// Inverse of `pack_qwen_nchw`: `[1, h_p*w_p, 64]` → `[1, 16, H_lat, W_lat]`.
///
/// Matches `qwenimage_gen.rs`'s decode-time unpack:
///   reshape to [B, h_p, w_p, C, 2, 2], permute to [B, C, h_p, 2, w_p, 2],
///   reshape to [B, C, H_lat, W_lat].
fn unpack_qwen_to_nchw(packed: &Tensor, h_lat: usize, w_lat: usize) -> Result<Tensor> {
    let dims = packed.shape().dims();
    let b = dims[0];
    let seq = dims[1];
    let pc = dims[2];
    let h_p = h_lat / PATCH_SIZE;
    let w_p = w_lat / PATCH_SIZE;
    if seq != h_p * w_p {
        return Err(anyhow!(
            "unpack_qwen_to_nchw: seq {} != h_p*w_p {}",
            seq,
            h_p * w_p
        ));
    }
    let c = pc / (PATCH_SIZE * PATCH_SIZE);
    let nchw = packed
        .reshape(&[b, h_p, w_p, c, PATCH_SIZE, PATCH_SIZE])?
        .permute(&[0, 3, 1, 4, 2, 5])?
        .reshape(&[b, c, h_lat, w_lat])?;
    Ok(nchw)
}

// ---------------------------------------------------------------------------
// CFG norm-rescale (matches qwenimage_gen.rs)
// ---------------------------------------------------------------------------

fn norm_rescale_cfg(cond: &Tensor, comb: &Tensor) -> Result<Tensor> {
    let cond_sq = cond.mul(cond)?;
    let comb_sq = comb.mul(comb)?;
    let cond_sum = cond_sq.sum_dim_keepdim(2)?;
    let comb_sum = comb_sq.sum_dim_keepdim(2)?;
    let cond_norm = cond_sum.sqrt()?;
    let comb_norm = comb_sum.sqrt()?;
    let ratio = cond_norm.div(&comb_norm)?;
    Ok(comb.mul(&ratio)?)
}

// ---------------------------------------------------------------------------
// Schedule (qwenimage_gen.rs verbatim)
// ---------------------------------------------------------------------------

fn build_qwen_schedule(num_steps: usize, seq_len: usize) -> Vec<f32> {
    let base_shift: f32 = 0.5;
    let max_shift: f32 = 0.9;
    let base_seq_len: f32 = 256.0;
    let max_seq_len_shift: f32 = 8192.0;
    let shift_terminal: f32 = 0.02;

    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len);
    let bb = base_shift - m * base_seq_len;
    let mu = (seq_len as f32) * m + bb;
    let exp_mu = mu.exp();

    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / (num_steps - 1) as f32;
            1.0 - t * (1.0 - 1.0 / num_steps as f32)
        })
        .collect();
    for s in sigmas.iter_mut() {
        let denom = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / denom;
    }
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    if one_minus_last.abs() > 1e-12 {
        let scale = one_minus_last / (1.0 - shift_terminal);
        for s in sigmas.iter_mut() {
            let o = 1.0 - *s;
            *s = 1.0 - o / scale;
        }
    }
    sigmas.push(0.0);
    sigmas
}

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

fn make_b1_bf16(v: f32, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::from_vec(vec![v], Shape::from_dims(&[1]), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

fn box_muller_noise(seed: u64, numel: usize) -> Vec<f32> {
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

fn save_chw_f32_to_png(rgb: &Tensor, h: usize, w: usize, path: &PathBuf) -> Result<()> {
    let rgb_f32 = if rgb.dtype() == DType::F32 {
        rgb.clone_result()?
    } else {
        rgb.to_dtype(DType::F32)?
    };
    let data = rgb_f32.to_vec_f32()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                let v = data[idx].clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("Failed to build RgbImage"))?;
    img.save(path)?;
    Ok(())
}
