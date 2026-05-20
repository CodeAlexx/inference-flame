//! Wan2.2-I2V-A14B — pure-Rust image-to-video.
//!
//! Same dual-expert DiT architecture as T2V but:
//! - patch_embedding accepts 36 channels (16 noise + 4 mask + 16 VAE image = 36)
//! - `y` (image conditioning) is concatenated with noise before patchify
//! - Different config: boundary=0.900, shift=5.0, cfg=(3.5, 3.5)
//!
//! Two modes for the image-conditioning `y`:
//!
//! 1. **Image mode (new, pure Rust):** pass `--image foo.png`. The binary loads
//!    the PNG, resizes to (height, width), normalizes to [-1, 1], builds a
//!    `[1, 3, F, H, W]` video tensor with the image as frame 0 and zeros for
//!    the rest, encodes via `Wan21VaeEncoder`, then concatenates the 4-channel
//!    I2V mask + 16-channel encoded latent to form `y [20, F_lat, H_lat, W_lat]`.
//!
//! 2. **Pre-encoded mode (legacy):** pass `--embeds embeds.safetensors`
//!    containing keys `cond`, `uncond`, `y`, `target_h`, `target_w`,
//!    `frame_num` (matches `scripts/wan22_i2v_encode.py`).
//!
//! Text embeddings (`cond`, `uncond`) are always loaded from `--embeds`
//! (porting the Wan2.2 text encoder is out of scope).
//!
//! After denoising the latent is decoded in-process via `Wan21VaeDecoder` and
//! written to an MP4 via the ffmpeg-backed `mux` module — no Python needed.

use std::path::PathBuf;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::wan22_dit::Wan22Dit;
use inference_flame::vae::{Wan21VaeDecoder, Wan21VaeEncoder};

const HIGH_NOISE_PATH: &str = "/home/alex/.serenity/models/checkpoints/wan2.2_i2v_high_noise_14b_fp16.safetensors";
const LOW_NOISE_PATH: &str = "/home/alex/.serenity/models/checkpoints/wan2.2_i2v_low_noise_14b_fp16.safetensors";
const VAE_DEFAULT_PATH: &str = "/home/alex/.serenity/models/vaes/wan_2.1_vae.safetensors";

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BOUNDARY: f32 = 0.900;        // I2V boundary (higher than T2V's 0.875)
const SHIFT: f32 = 5.0;             // I2V shift (lower than T2V's 12.0)
const VAE_STRIDE: [usize; 3] = [4, 8, 8];
const Z_DIM: usize = 16;
const PATCH_SIZE: [usize; 3] = [1, 2, 2];
const SAMPLE_FPS: f32 = 16.0;

struct CliArgs {
    embeds_path: String,
    image_path: Option<String>,
    output_path: String,
    vae_path: String,
    width: Option<usize>,
    height: Option<usize>,
    frame_num: Option<usize>,
    num_steps: usize,
    seed: u64,
    cfg_low: f32,
    cfg_high: f32,
    shift: f32,
}

fn parse_args() -> CliArgs {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut take = |flag: &str| -> Option<String> {
        let mut i = 0;
        while i < args.len() {
            if args[i] == flag {
                if i + 1 >= args.len() { return None; }
                let v = args.remove(i + 1);
                args.remove(i);
                return Some(v);
            }
            // also support `--flag=value`
            let eq = format!("{flag}=");
            if args[i].starts_with(&eq) {
                let v = args[i][eq.len()..].to_string();
                args.remove(i);
                return Some(v);
            }
            i += 1;
        }
        None
    };

    let embeds_path = take("--embeds").unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_i2v_embeds.safetensors".to_string()
    });
    let image_path = take("--image");
    let output_path = take("--output").unwrap_or_else(|| {
        "/home/alex/EriDiffusion/inference-flame/output/wan22_i2v_out.mp4".to_string()
    });
    let vae_path = take("--vae").unwrap_or_else(|| VAE_DEFAULT_PATH.to_string());
    let width = take("--width").and_then(|s| s.parse().ok());
    let height = take("--height").and_then(|s| s.parse().ok());
    let frame_num = take("--frame-num").and_then(|s| s.parse().ok());
    let num_steps = take("--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_usize("WAN_STEPS", 20));
    let seed = take("--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_u64("WAN_SEED", 42));
    let cfg_low = take("--cfg-low")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_f32("WAN_CFG_LOW", 3.5));
    let cfg_high = take("--cfg-high")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_f32("WAN_CFG_HIGH", 3.5));
    let shift = take("--shift")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_f32("WAN_SHIFT", SHIFT));

    // Backward-compat positional args (legacy: embeds out_latents).
    // If the first two leftover positionals look like paths, treat them as
    // embeds and (ignored) latents — keep them harmless.
    if !args.is_empty() && image_path.is_none() {
        // honor legacy positional embeds path
        let p = args.remove(0);
        let embeds_path = if !p.starts_with("--") { p } else { embeds_path };
        return CliArgs { embeds_path, image_path, output_path, vae_path, width, height, frame_num,
            num_steps, seed, cfg_low, cfg_high, shift };
    }

    CliArgs { embeds_path, image_path, output_path, vae_path, width, height, frame_num,
        num_steps, seed, cfg_low, cfg_high, shift }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();
    let device = global_cuda_device();

    let cli = parse_args();

    println!("=== Wan2.2-I2V-A14B — pure-Rust image → mp4 ===");
    println!("Embeddings: {}", cli.embeds_path);
    if let Some(img) = &cli.image_path {
        println!("Image:      {}", img);
        println!("VAE:        {}", cli.vae_path);
    } else {
        println!("(no --image: using pre-encoded `y` from embeddings file)");
    }
    println!("Output mp4: {}", cli.output_path);
    println!();

    // ------------------------------------------------------------------
    // Load cached text embeddings + (optionally) pre-encoded `y`
    // ------------------------------------------------------------------
    println!("--- Loading cached text embeddings ---");
    let t0 = Instant::now();
    let tensors = flame_core::serialization::load_file(
        std::path::Path::new(&cli.embeds_path),
        &device,
    )?;
    let cond = ensure_bf16(tensors.get("cond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'cond' in {}", cli.embeds_path))?.clone())?;
    let uncond = ensure_bf16(tensors.get("uncond")
        .ok_or_else(|| anyhow::anyhow!("Missing 'uncond' in {}", cli.embeds_path))?.clone())?;

    // Geometry: CLI overrides embeds metadata; embeds metadata fills the rest.
    let target_h = cli.height
        .or_else(|| tensors.get("target_h")
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().map(|f| *f as usize)))
        .ok_or_else(|| anyhow::anyhow!("Need --height or 'target_h' in embeds"))?;
    let target_w = cli.width
        .or_else(|| tensors.get("target_w")
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().map(|f| *f as usize)))
        .ok_or_else(|| anyhow::anyhow!("Need --width or 'target_w' in embeds"))?;
    let frame_num = cli.frame_num
        .or_else(|| tensors.get("frame_num")
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().map(|f| *f as usize)))
        .ok_or_else(|| anyhow::anyhow!("Need --frame-num or 'frame_num' in embeds"))?;

    // Validate geometry against VAE_STRIDE and PATCH_SIZE.
    if (frame_num - 1) % VAE_STRIDE[0] != 0 {
        anyhow::bail!("frame_num-1 must be divisible by VAE_STRIDE[0]=4; got {frame_num}");
    }
    if target_h % (VAE_STRIDE[1] * PATCH_SIZE[1]) != 0 {
        anyhow::bail!("height must be divisible by {}; got {target_h}",
            VAE_STRIDE[1] * PATCH_SIZE[1]);
    }
    if target_w % (VAE_STRIDE[2] * PATCH_SIZE[2]) != 0 {
        anyhow::bail!("width must be divisible by {}; got {target_w}",
            VAE_STRIDE[2] * PATCH_SIZE[2]);
    }

    let lat_f = (frame_num - 1) / VAE_STRIDE[0] + 1;
    let lat_h = target_h / VAE_STRIDE[1];
    let lat_w = target_w / VAE_STRIDE[2];

    // Resolve `y`: either encode from an image, or pull pre-encoded from embeds.
    let y = if let Some(image_path) = &cli.image_path {
        let pre_encoded = tensors.get("y");
        if pre_encoded.is_some() {
            println!("  (note: --image overrides pre-encoded 'y' in embeds)");
        }
        drop(tensors);
        encode_image_to_y(image_path, &cli.vae_path, target_h, target_w, frame_num,
            lat_h, lat_w, lat_f, &device)?
    } else {
        let y = ensure_bf16(tensors.get("y")
            .ok_or_else(|| anyhow::anyhow!(
                "No --image given and embeds has no 'y' tensor"))?.clone())?;
        drop(tensors);
        y
    };

    println!("  cond:   {:?}", cond.shape().dims());
    println!("  uncond: {:?}", uncond.shape().dims());
    println!("  y:      {:?}", y.shape().dims());
    println!("  target: {}x{}, frames={}", target_w, target_h, frame_num);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Compute patched seq_len
    // ------------------------------------------------------------------
    let patch_f = lat_f;
    let patch_h = lat_h / PATCH_SIZE[1];
    let patch_w = lat_w / PATCH_SIZE[2];
    let seq_len = patch_f * patch_h * patch_w;

    println!("Size:       {}x{}, frames={}, steps={}", target_w, target_h, frame_num, cli.num_steps);
    println!("CFG:        low={}, high={}, shift={}", cli.cfg_low, cli.cfg_high, cli.shift);
    println!("Seed:       {}", cli.seed);
    println!("  Latent: [{}, {}, {}, {}]", Z_DIM, lat_f, lat_h, lat_w);
    println!("  Grid:   ({}, {}, {}), seq_len={}", patch_f, patch_h, patch_w, seq_len);
    println!();

    // ------------------------------------------------------------------
    // Generate noise
    // ------------------------------------------------------------------
    let numel = Z_DIM * lat_f * lat_h * lat_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(cli.seed);
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
    let noise = Tensor::from_vec(
        noise_data,
        Shape::from_dims(&[Z_DIM, lat_f, lat_h, lat_w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    // ------------------------------------------------------------------
    // Build sigma schedule
    // ------------------------------------------------------------------
    let sigma_max: f32 = 1.0 - 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let sigma_min: f32 = 1.0 / NUM_TRAIN_TIMESTEPS as f32;
    let mut sigmas: Vec<f32> = (0..cli.num_steps)
        .map(|i| {
            let t = i as f32 / cli.num_steps as f32;
            sigma_max + t * (sigma_min - sigma_max)
        })
        .collect();
    for s in sigmas.iter_mut() {
        *s = cli.shift * *s / (1.0 + (cli.shift - 1.0) * *s);
    }
    sigmas.push(0.0);
    let timesteps: Vec<f32> = sigmas.iter().map(|s| s * NUM_TRAIN_TIMESTEPS as f32).collect();
    let boundary_ts = BOUNDARY * NUM_TRAIN_TIMESTEPS as f32;

    println!("  sigmas[0]={:.4}  sigmas[-2]={:.4}", sigmas[0], sigmas[cli.num_steps - 1]);
    println!("  timesteps[0]={:.1}  boundary={:.1}", timesteps[0], boundary_ts);
    println!();

    // ------------------------------------------------------------------
    // Load DiT
    // ------------------------------------------------------------------
    let first_is_high = timesteps[0] >= boundary_ts;
    println!("--- Loading {} model first ---",
        if first_is_high { "high_noise" } else { "low_noise" });
    let t0 = Instant::now();
    let mut current_is_high = first_is_high;
    let mut dit = if first_is_high {
        Wan22Dit::load(HIGH_NOISE_PATH, &device)?
    } else {
        Wan22Dit::load(LOW_NOISE_PATH, &device)?
    };
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // ------------------------------------------------------------------
    // Euler denoise loop (I2V: concat y with noise before each forward)
    // ------------------------------------------------------------------
    println!("--- Denoise ({} steps, I2V Euler flow-matching) ---", cli.num_steps);
    let t_denoise = Instant::now();
    let mut latent = noise;

    for step in 0..cli.num_steps {
        let ts = timesteps[step];
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma;

        let need_high = ts >= boundary_ts;
        if need_high != current_is_high {
            println!("  [SWITCH] {} → {} at ts={:.1}",
                if current_is_high { "high" } else { "low" },
                if need_high { "high" } else { "low" }, ts);
            drop(dit);
            let t_switch = Instant::now();
            dit = if need_high {
                Wan22Dit::load(HIGH_NOISE_PATH, &device)?
            } else {
                Wan22Dit::load(LOW_NOISE_PATH, &device)?
            };
            current_is_high = need_high;
            println!("  [SWITCH] Loaded in {:.1}s", t_switch.elapsed().as_secs_f32());
        }

        let guide_scale = if need_high { cli.cfg_high } else { cli.cfg_low };

        // I2V: concatenate [noise, y] along channel dim before forward
        // noise: [16, F_lat, H_lat, W_lat], y: [20, F_lat, H_lat, W_lat]
        // → x_input: [36, F_lat, H_lat, W_lat]
        let next_latent = {
            let x_input = Tensor::cat(&[&latent, &y], 0)?;

            let cond_pred = dit.forward_i2v(&x_input, ts, &cond, seq_len)?;
            let x_input_uncond = Tensor::cat(&[&latent, &y], 0)?;
            let uncond_pred = dit.forward_i2v(&x_input_uncond, ts, &uncond, seq_len)?;

            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(guide_scale)?;
            let noise_pred = uncond_pred.add(&scaled)?;

            let step_delta = noise_pred.mul_scalar(dt)?;
            latent.add(&step_delta)?
        };
        latent = next_latent;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == cli.num_steps {
            println!(
                "  step {}/{}  ts={:.1}  sigma={:.4}  cfg={:.1}  ({:.1}s elapsed)",
                step + 1, cli.num_steps, ts, sigma, guide_scale,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }

    let dt_denoise = t_denoise.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt_denoise, dt_denoise / cli.num_steps as f32);
    println!();

    drop(dit);
    drop(cond);
    drop(uncond);
    drop(y);
    flame_core::cuda_alloc_pool::clear_pool_cache();

    // ------------------------------------------------------------------
    // VAE decode → MP4
    // ------------------------------------------------------------------
    println!("--- VAE decode + MP4 mux ---");
    let t_dec = Instant::now();
    let vae = Wan21VaeDecoder::load(&cli.vae_path, &device)
        .map_err(|e| anyhow::anyhow!("Wan21VaeDecoder::load: {e}"))?;
    println!("  VAE loaded in {:.1}s", t_dec.elapsed().as_secs_f32());

    // Decoder expects [B, 16, T, H, W]; saved latent is [16, T, H, W].
    let latent_5d = latent.reshape(&[1, Z_DIM, lat_f, lat_h, lat_w])?;
    let t_decode = Instant::now();
    let video = vae.decode(&latent_5d)
        .map_err(|e| anyhow::anyhow!("vae.decode: {e}"))?;
    let dims = video.shape().dims().to_vec();
    // Decoder output: [1, 3, F_pix, H_pix, W_pix] BF16 in [-1, 1].
    let (_, _, t_pix, h_pix, w_pix) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    println!(
        "  Decoded {} px frames @ {}x{} in {:.1}s",
        t_pix, w_pix, h_pix, t_decode.elapsed().as_secs_f32()
    );

    // Move to CPU as F32 in [3, F, H, W] order (drop batch dim).
    let video_chw = video.reshape(&[3, t_pix, h_pix, w_pix])?;
    let video_f32 = video_chw.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    drop(video_chw);
    drop(video);
    drop(vae);
    flame_core::cuda_alloc_pool::clear_pool_cache();

    let frames_u8 = inference_flame::mux::video_tensor_to_rgb_u8(&video_f32, t_pix, h_pix, w_pix);

    let out_path = PathBuf::from(&cli.output_path);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    inference_flame::mux::write_mp4_video_only(
        &out_path,
        &frames_u8,
        t_pix,
        w_pix,
        h_pix,
        SAMPLE_FPS,
    ).map_err(|e| anyhow::anyhow!("write_mp4_video_only: {e}"))?;

    println!();
    println!("============================================================");
    println!("VIDEO SAVED: {}", cli.output_path);
    println!("  {} frames, {}x{}, {:.0} fps", t_pix, w_pix, h_pix, SAMPLE_FPS);
    println!("Total time:    {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");

    Ok(())
}

/// Load PNG → resize → [-1,1] → build [1,3,F,H,W] video (image @ frame 0,
/// zeros elsewhere) → VAE-encode → build I2V mask → concat → y.
///
/// Returns y in shape `[20, F_lat, H_lat, W_lat]` BF16 (4 mask + 16 latent
/// channels), matching the Python reference.
fn encode_image_to_y(
    image_path: &str,
    vae_path: &str,
    target_h: usize,
    target_w: usize,
    frame_num: usize,
    lat_h: usize,
    lat_w: usize,
    lat_f: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<Tensor> {
    println!("--- Encoding image via Wan2.1 VAE ---");
    let t0 = Instant::now();

    // 1. Load PNG.
    let img = image::open(image_path)
        .map_err(|e| anyhow::anyhow!("failed to open image {image_path}: {e}"))?
        .to_rgb8();
    println!("  Loaded PNG: {}x{}", img.width(), img.height());

    // 2. Resize to (target_h, target_w). Lanczos3 — Python uses bicubic, but
    //    Lanczos3 is closer to it than bilinear, and the conditioning is
    //    a single frame of context, not a parity reference.
    let resized = image::imageops::resize(
        &img,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // 3. Build the [1, 3, F, H, W] video buffer in F32, normalized to [-1, 1].
    //    Layout follows the encoder convention: channel-major over time.
    let frame_pixels = 3 * target_h * target_w;
    let mut data = vec![0.0f32; 1 * 3 * frame_num * target_h * target_w];
    // Frame 0: image; frames 1..F: zeros (already initialized to 0).
    // For a 5D layout [B=1, C=3, F, H, W], the channel-stride is F*H*W,
    // the frame-stride is H*W, etc.
    let cfhw = 3 * frame_num * target_h * target_w; // batch stride (B=1, so unused)
    let _ = cfhw;
    let fhw = frame_num * target_h * target_w;      // channel stride
    let hw = target_h * target_w;                   // frame stride
    let _ = frame_pixels;
    for c in 0..3 {
        for y in 0..target_h {
            for x in 0..target_w {
                let p = resized.get_pixel(x as u32, y as u32);
                let v = (p[c] as f32) / 127.5 - 1.0;
                // Frame 0 only.
                let idx = c * fhw + 0 * hw + y * target_w + x;
                data[idx] = v;
            }
        }
    }

    let video = Tensor::from_vec(
        data,
        Shape::from_dims(&[1, 3, frame_num, target_h, target_w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    // 4. VAE encode.
    let encoder = Wan21VaeEncoder::load(vae_path, device)
        .map_err(|e| anyhow::anyhow!("Wan21VaeEncoder::load: {e}"))?;
    let z = encoder.encode(&video)
        .map_err(|e| anyhow::anyhow!("Wan21VaeEncoder::encode: {e}"))?;
    drop(encoder);
    drop(video);
    flame_core::cuda_alloc_pool::clear_pool_cache();

    // Encoder returns [B=1, 16, F_lat, H_lat, W_lat]; drop the batch dim to
    // match the rest of the I2V pipeline's [C, F, H, W] convention.
    let z_dims = z.shape().dims().to_vec();
    if z_dims.len() != 5 || z_dims[0] != 1 || z_dims[1] != Z_DIM as usize {
        anyhow::bail!("encoder produced unexpected shape {:?}", z_dims);
    }
    if z_dims[2] != lat_f || z_dims[3] != lat_h || z_dims[4] != lat_w {
        anyhow::bail!(
            "encoder output [{}, {}, {}, {}] but expected [{}, {}, {}, {}]",
            z_dims[1], z_dims[2], z_dims[3], z_dims[4],
            Z_DIM, lat_f, lat_h, lat_w);
    }
    let z_4d = z.reshape(&[Z_DIM, lat_f, lat_h, lat_w])?;

    // 5. Build I2V mask [4, F_lat, H_lat, W_lat]. Mirrors build_i2v_mask in
    //    scripts/wan22_i2v_encode.py:
    //
    //      msk = ones(1, F_pix, lat_h, lat_w);  msk[:, 1:] = 0
    //      # repeat frame 0 four times so total time = F_pix + 3
    //      msk = cat([repeat(msk[:, 0:1], 4, dim=1), msk[:, 1:]], dim=1)
    //      # group into chunks of 4 along time, then transpose
    //      msk = msk.view(1, (F_pix+3)//4, 4, lat_h, lat_w).transpose(1, 2)[0]
    //
    //    Result: msk[t, f_lat, h, w]. Channel 0 of every f_lat is 1 (the
    //    first frame in each 4-frame VAE chunk); the other channels are 0
    //    except in the very first f_lat where they're also 1 (since the
    //    first chunk is the repeated frame-0).
    //
    //    Easier path: build the [4, F_lat, lat_h, lat_w] mask directly.
    //    Per the python:
    //      For f_lat = 0: the 4 channels come from the 4 repeated-frame-0
    //        copies → all 1.0.
    //      For f_lat = 1..F_lat: the 4 channels come from input frames
    //        [(f_lat-1)*4 + 1 .. (f_lat-1)*4 + 5] → all 0.
    //    (Because input frame 0 was the only "1"; the 3 extra copies of it
    //    fill chunk 0, and chunks 1+ are all real-input frames 1..F-1
    //    which are all 0.)
    let mut msk_data = vec![0.0f32; 4 * lat_f * lat_h * lat_w];
    // Set the 4 channels of f_lat=0 to 1.0.
    let chw = 4 * lat_h * lat_w;
    let _ = chw;
    // mask layout: [4, F_lat, H, W]
    //   stride_c = F_lat*H*W, stride_f = H*W, stride_h = W
    let stride_c = lat_f * lat_h * lat_w;
    let stride_f = lat_h * lat_w;
    for c in 0..4 {
        let f = 0;
        for h in 0..lat_h {
            for w in 0..lat_w {
                msk_data[c * stride_c + f * stride_f + h * lat_w + w] = 1.0;
            }
        }
    }
    let msk = Tensor::from_vec(
        msk_data,
        Shape::from_dims(&[4, lat_f, lat_h, lat_w]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    // 6. Concat [4 mask, 16 latent] along channel dim → y [20, F_lat, H, W].
    let y = Tensor::cat(&[&msk, &z_4d], 0)?;
    drop(msk);
    drop(z_4d);
    drop(z);

    println!("  Encoded in {:.1}s, y shape {:?}", t0.elapsed().as_secs_f32(), y.shape().dims());
    Ok(y)
}

fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    if t.dtype() == DType::BF16 { Ok(t) } else { Ok(t.to_dtype(DType::BF16)?) }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
