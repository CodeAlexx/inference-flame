//! Lance I2V — pure-Rust CLI for the Lance 3B image-to-video pipeline.
//!
//! Mirrors `lance_t2v.rs` but adds a reference-image-conditioning path:
//!   1. Load + resize + normalize the PNG to `[1, 3, 1, H, W]` BF16 in [-1, 1].
//!   2. VAE-encode with `Wan22VaeEncoder` → `[1, 48, 1, H/16, W/16]`.
//!   3. Drop the encoder.
//!   4. Tokenize cond + uncond prompts.
//!   5. Sample initial noise (`flame_core::rng::randn_torch`) at the full
//!      `[1, 48, T_lat, H/16, W/16]` shape.
//!   6. Load Lance, precompute mRoPE, call `Lance::i2v_with_cfg`. The denoise
//!      loop splices `ref_latent` into the first T_lat=1 frame and masks the
//!      timestep at those positions to zero (no noise to remove).
//!   7. Decode + mux to mp4 (identical to `lance_t2v.rs`).
//!
//! The first output frame will be ≈ the (re-encoded then re-decoded) input
//! image; subsequent frames are denoised normally.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use inference_flame::models::lance::{Lance, LanceConfig};
use inference_flame::mux::{video_tensor_to_rgb_u8, write_mp4_video_only};
use inference_flame::vae::wan22_vae::Wan22VaeDecoder;
use inference_flame::vae::Wan22VaeEncoder;
use tokenizers::Tokenizer;

const DEFAULT_PROMPT: &str = "A cinematic motion shot of the scene in the image, smooth camera push-in, gentle parallax, photorealistic, dramatic lighting.";

fn t_lat_from_pixel_frames(num_frames: usize) -> usize {
    debug_assert!(num_frames >= 1);
    1 + (num_frames - 1) / 4
}

fn t_pixel_from_t_lat(t_lat: usize) -> usize {
    (t_lat - 1) * 4 + 1
}

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    vae_path: PathBuf,
    image: PathBuf,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    num_frames: usize,
    fps: f32,
    steps: usize,
    shift: f32,
    cfg: f32,
    seed: u64,
    output: Option<PathBuf>,
    text_template: bool,
}

impl Args {
    fn defaults() -> Self {
        Self {
            model_path: PathBuf::from(
                "/home/alex/.serenity/models/lance/Lance_3B_Video",
            ),
            vae_path: PathBuf::from(
                "/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors",
            ),
            image: PathBuf::new(),
            prompt: DEFAULT_PROMPT.to_string(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            num_frames: 49,
            fps: 16.0,
            steps: 30,
            shift: 3.5,
            cfg: 4.0,
            seed: 42,
            output: None,
            text_template: true,
        }
    }
}

fn parse_args() -> std::result::Result<Args, String> {
    let mut a = Args::defaults();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        let mut next = || -> std::result::Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--model-path" | "--model_path" => a.model_path = PathBuf::from(next()?),
            "--vae-path" | "--vae_path" => a.vae_path = PathBuf::from(next()?),
            "--image" | "--input-image" | "--input_image" => a.image = PathBuf::from(next()?),
            "--prompt" => a.prompt = next()?,
            "--negative-prompt" | "--negative_prompt" => a.negative_prompt = next()?,
            "--width" | "--video-width" | "--video_width" => {
                a.width = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--height" | "--video-height" | "--video_height" => {
                a.height = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--num-frames" | "--num_frames" | "--frames" => {
                a.num_frames = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--fps" => {
                a.fps = next()?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--steps" => {
                a.steps = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--shift" => {
                a.shift = next()?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--cfg" => {
                a.cfg = next()?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--seed" => {
                a.seed = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--output" | "--output-mp4" | "-o" => a.output = Some(PathBuf::from(next()?)),
            "--text-template" | "--text_template" => {
                let v = next()?;
                a.text_template = matches!(
                    v.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                );
            }
            "--no-text-template" => a.text_template = false,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if a.image.as_os_str().is_empty() {
        return Err("--image PATH is required (input PNG/JPG for I2V conditioning)".to_string());
    }
    Ok(a)
}

fn print_usage() {
    eprintln!(
        r#"lance_i2v_gen — Lance 3B image-to-video generation

Usage:
  lance_i2v_gen --image PATH [options]

Options:
  --image       <PATH>  input image (REQUIRED)
  --model-path  <DIR>   Lance_3B_Video weights dir  [default: /home/alex/.serenity/models/lance/Lance_3B_Video]
  --vae-path    <PATH>  Wan22 VAE safetensors      [default: /home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors]
  --prompt      <STR>   prompt text
  --negative-prompt <STR> uncond prompt            [default: ""]
  --width       <N>     output width  (mod 16)     [default: 512]
  --height      <N>     output height (mod 16)     [default: 512]
  --num-frames  <N>     pixel frames; T_lat=1+(N-1)/4 [default: 49]
  --fps         <F>     mp4 frame rate             [default: 16.0]
  --steps       <N>     denoise steps              [default: 30]
  --shift       <F>     flow-matching shift        [default: 3.5]
  --cfg         <F>     CFG scale                  [default: 4.0]
  --seed        <N>     RNG seed                   [default: 42]
  --output      <PATH>  mp4 output path
"#
    );
}

fn default_output_path(width: usize, height: usize, num_frames: usize) -> PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(format!(
        "inference-flame/output/lance_i2v_{}x{}_{}f_{}.mp4",
        width, height, num_frames, secs
    ))
}

// ---------------------------------------------------------------------------
// Tokenizer + prompt template — mirrors lance_t2v.rs.
// ---------------------------------------------------------------------------

const IM_START_ID: i32 = 151644;
const IM_END_ID: i32 = 151645;

const T2V_SYSTEM_PROMPT: &str = "Describe the video by detailing the color, quantity, visible text, shape, size, texture, spatial relationships and motion/camera movements of the objects and background:";

fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
    let p = model_path.join("tokenizer.json");
    Tokenizer::from_file(&p).map_err(|e| anyhow!("Tokenizer::from_file({}): {e}", p.display()))
}

fn tokenize_to_ids(tok: &Tokenizer, text: &str, template: bool) -> Result<Vec<i32>> {
    if template {
        let rendered = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            T2V_SYSTEM_PROMPT, text
        );
        let enc = tok
            .encode(rendered, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    } else {
        let enc = tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<i32> = Vec::with_capacity(enc.get_ids().len() + 2);
        ids.push(IM_START_ID);
        ids.extend(enc.get_ids().iter().map(|&id| id as i32));
        ids.push(IM_END_ID);
        Ok(ids)
    }
}

fn ids_to_tensor(ids: &[i32], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let n = ids.len();
    let f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let t = Tensor::from_vec(f, Shape::from_dims(&[n]), device.clone())?;
    t.to_dtype(DType::I32)
        .map_err(|e| anyhow!("cast tokens to I32: {e}"))
}

// ---------------------------------------------------------------------------
// PNG load → [1, 3, 1, H, W] BF16 in [-1, 1].
// ---------------------------------------------------------------------------

fn load_image_to_video_tensor(
    path: &Path,
    target_h: usize,
    target_w: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| anyhow!("failed to open image {}: {e}", path.display()))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
    // Layout: [C=3, H, W] row-major, then unsqueeze to [1, 3, 1, H, W].
    let mut data = vec![0.0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * target_h * target_w + y * target_w + x] =
                    (p[c] as f32) / 127.5 - 1.0;
            }
        }
    }
    let t = Tensor::from_vec(
        data,
        Shape::from_dims(&[1, 3, 1, target_h, target_w]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    Ok(t)
}

// ---------------------------------------------------------------------------
// Watchdog (advisory) — same as lance_t2v.rs.
// ---------------------------------------------------------------------------

fn spawn_watchdog(start: Instant) {
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let elapsed = start.elapsed().as_secs();
        if elapsed > 20 * 60 {
            log::warn!(
                "[lance_i2v watchdog] wall time {} s exceeds 20 min cap (advisory)",
                elapsed
            );
        }
        if let Ok(out) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Some(line) = s.lines().next() {
                    if let Ok(t) = line.trim().parse::<u32>() {
                        if t >= 78 {
                            log::warn!(
                                "[lance_i2v watchdog] GPU0 temperature {}°C exceeds 78°C cap (advisory)",
                                t
                            );
                        }
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Stage 2a: VAE encode the reference image, then drop the encoder.
// ---------------------------------------------------------------------------

fn stage2a_encode_image(
    args: &Args,
    device: &Arc<CudaDevice>,
) -> Result<(Vec<f32>, Vec<usize>)> {
    log::info!("[lance_i2v] Stage 2a: load Wan22 VAE encoder, encode reference image");
    let t0 = Instant::now();
    let enc = Wan22VaeEncoder::load(&args.vae_path, device)
        .with_context(|| format!("Wan22VaeEncoder::load({})", args.vae_path.display()))?;
    log::info!(
        "[lance_i2v]   encoder loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let img_tensor = load_image_to_video_tensor(&args.image, args.height, args.width, device)?;
    log::info!(
        "[lance_i2v]   loaded+resized image to shape={:?}",
        img_tensor.shape().dims()
    );

    let t1 = Instant::now();
    let ref_latent = enc.encode(&img_tensor)?;
    log::info!(
        "[lance_i2v]   encode complete in {:.1}s, ref_latent shape={:?}",
        t1.elapsed().as_secs_f32(),
        ref_latent.shape().dims()
    );

    // ref_latent should be [1, 48, 1, H/16, W/16].
    let dims = ref_latent.shape().dims().to_vec();
    if dims.len() != 5 || dims[0] != 1 || dims[2] != 1 {
        return Err(anyhow!(
            "ref_latent shape unexpected: got {:?}, expected [1, 48, 1, H_lat, W_lat]",
            dims
        ));
    }

    let host = ref_latent.to_dtype(DType::F32)?.to_vec_f32()?;

    drop(enc);
    drop(ref_latent);
    drop(img_tensor);

    Ok((host, dims))
}

// ---------------------------------------------------------------------------
// Stage 2b: Lance denoise — same shape contract as lance_t2v but with I2V.
// ---------------------------------------------------------------------------

fn stage2b_denoise(
    args: &Args,
    cond_ids: &[i32],
    uncond_ids: &[i32],
    t_lat: usize,
    ref_latent_host: Vec<f32>,
    ref_latent_shape: Vec<usize>,
    device: &Arc<CudaDevice>,
) -> Result<(Vec<f32>, Vec<usize>)> {
    log::info!("[lance_i2v] Stage 2b: load Lance, precompute mRoPE, I2V denoise");

    let mut cfg = LanceConfig::default_3b(device.clone());
    cfg.num_inference_steps = args.steps;
    cfg.timestep_shift = args.shift;
    cfg.cfg_text_scale = args.cfg;
    let cfg = Arc::new(cfg);

    if args.width % 16 != 0 || args.height % 16 != 0 {
        return Err(anyhow!(
            "width/height must be divisible by 16 (Wan22 VAE spatial factor); got {}x{}",
            args.width,
            args.height
        ));
    }
    let h_latent = args.height / 16;
    let w_latent = args.width / 16;
    let l_image = t_lat * h_latent * w_latent;

    // Sanity check ref_latent shape matches expectations.
    if ref_latent_shape != vec![1, cfg.patch_latent_dim(), 1, h_latent, w_latent] {
        return Err(anyhow!(
            "ref_latent shape mismatch: got {:?}, expected [1, {}, 1, {}, {}]",
            ref_latent_shape,
            cfg.patch_latent_dim(),
            h_latent,
            w_latent
        ));
    }

    let max_pos = cond_ids.len().max(uncond_ids.len()) + 1000 + l_image + 64;
    log::info!(
        "[lance_i2v]   latent grid T={} H={} W={} (L_image={}), max_pos={}",
        t_lat,
        h_latent,
        w_latent,
        l_image,
        max_pos
    );

    let t0 = Instant::now();
    let lance = Lance::load(&args.model_path, cfg.clone(), device)
        .with_context(|| format!("Lance::load({})", args.model_path.display()))?;
    log::info!(
        "[lance_i2v]   Lance loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let mrope = lance.precompute_mrope(max_pos, cfg.dtype)?;

    let cond_tokens = ids_to_tensor(cond_ids, device)?;
    let uncond_tokens = ids_to_tensor(uncond_ids, device)?;

    // Rehydrate ref_latent on device.
    let ref_latent = Tensor::from_vec(
        ref_latent_host,
        Shape::from_dims(&ref_latent_shape),
        device.clone(),
    )?
    .to_dtype(cfg.dtype)?;

    // Initial noise: same packed-sequence-aligned construction as lance_t2v.rs.
    let c = cfg.patch_latent_dim();
    let l = t_lat * h_latent * w_latent;
    let noise_lc = flame_core::rng::randn_torch(
        args.seed,
        Shape::from_dims(&[l, c]),
        device.clone(),
    )?;
    let initial_noise = noise_lc
        .reshape(&[t_lat, h_latent, w_latent, c])?
        .permute(&[3, 0, 1, 2])?
        .reshape(&[1, c, t_lat, h_latent, w_latent])?
        .to_dtype(cfg.dtype)?;
    log::info!(
        "[lance_i2v]   initial noise shape={:?}, ref_latent shape={:?}, seed={}",
        initial_noise.shape().dims(),
        ref_latent.shape().dims(),
        args.seed
    );

    let t1 = Instant::now();
    let latent = lance.i2v_with_cfg(
        &cond_tokens,
        &uncond_tokens,
        &ref_latent,
        &initial_noise,
        &mrope,
    )?;
    log::info!(
        "[lance_i2v]   denoise complete in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    let host_data = latent.to_dtype(DType::F32)?.to_vec_f32()?;
    let shape = latent.shape().dims().to_vec();

    drop(lance);
    drop(mrope);

    Ok((host_data, shape))
}

// ---------------------------------------------------------------------------
// Stage 3: Wan22 VAE video decode + mp4 mux.
// (Identical to lance_t2v.rs Stage 3.)
// ---------------------------------------------------------------------------

fn stage3_decode_and_mux(
    args: &Args,
    latent_host: Vec<f32>,
    latent_shape: Vec<usize>,
    device: &Arc<CudaDevice>,
    output: &Path,
) -> Result<()> {
    log::info!("[lance_i2v] Stage 3: load Wan22 VAE, decode video, mux mp4");
    let t0 = Instant::now();
    let vae = Wan22VaeDecoder::load(&args.vae_path, device)
        .with_context(|| format!("Wan22VaeDecoder::load({})", args.vae_path.display()))?;
    log::info!(
        "[lance_i2v]   VAE loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let latent_gpu = Tensor::from_vec(latent_host, Shape::from_dims(&latent_shape), device.clone())?
        .to_dtype(DType::BF16)?;
    log::info!(
        "[lance_i2v]   decoding latent shape={:?}",
        latent_gpu.shape().dims()
    );

    let t1 = Instant::now();
    let video = vae.decode(&latent_gpu)?;
    let video_dims = video.shape().dims().to_vec();
    log::info!(
        "[lance_i2v]   decode complete in {:.1}s, video shape={:?}",
        t1.elapsed().as_secs_f32(),
        video_dims
    );

    if video_dims.len() != 5 || video_dims[0] != 1 || video_dims[1] != 3 {
        return Err(anyhow!(
            "video tensor must be [1, 3, T, H, W], got {:?}",
            video_dims
        ));
    }
    let t_pixel = video_dims[2];
    let h_pixel = video_dims[3];
    let w_pixel = video_dims[4];

    drop(vae);

    let video_host = video.to_dtype(DType::F32)?.to_vec_f32()?;
    drop(video);
    if video_host.len() != 3 * t_pixel * h_pixel * w_pixel {
        return Err(anyhow!(
            "video buffer length {} != 3*T*H*W ({})",
            video_host.len(),
            3 * t_pixel * h_pixel * w_pixel
        ));
    }

    let rgb = video_tensor_to_rgb_u8(&video_host, t_pixel, h_pixel, w_pixel);

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    write_mp4_video_only(output, &rgb, t_pixel, w_pixel, h_pixel, args.fps)
        .with_context(|| format!("write_mp4_video_only → {}", output.display()))?;
    log::info!("[lance_i2v]   wrote mp4 → {}", output.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Run + main
// ---------------------------------------------------------------------------

fn run(args: &Args) -> Result<()> {
    let start = Instant::now();
    spawn_watchdog(start);

    let device = flame_core::global_cuda_device();

    log::info!("[lance_i2v] Stage 1: tokenize");
    let tok = load_tokenizer(&args.model_path)?;
    let cond_ids = tokenize_to_ids(&tok, &args.prompt, args.text_template)?;
    let uncond_ids = tokenize_to_ids(&tok, &args.negative_prompt, args.text_template)?;
    log::info!(
        "[lance_i2v]   cond_len={}  uncond_len={}",
        cond_ids.len(),
        uncond_ids.len()
    );
    if uncond_ids.is_empty() {
        return Err(anyhow!(
            "uncond_len=0; pass --negative-prompt with at least one non-empty token"
        ));
    }
    drop(tok);

    if args.num_frames < 1 {
        return Err(anyhow!("--num-frames must be >= 1"));
    }
    if (args.num_frames - 1) % 4 != 0 {
        log::warn!(
            "[lance_i2v] --num-frames={} doesn't satisfy (N-1) %% 4 == 0; \
             video decoder produces T_pixel=(T_lat-1)*4+1 = {}",
            args.num_frames,
            t_pixel_from_t_lat(t_lat_from_pixel_frames(args.num_frames))
        );
    }
    let t_lat = t_lat_from_pixel_frames(args.num_frames);
    log::info!(
        "[lance_i2v]   T_pixel_request={} → T_lat={} → T_pixel_actual={}",
        args.num_frames,
        t_lat,
        t_pixel_from_t_lat(t_lat)
    );

    // Stage 2a: VAE-encode the reference image, then drop the encoder.
    let (ref_latent_host, ref_latent_shape) = stage2a_encode_image(args, &device)?;

    // Stage 2b: Lance I2V denoise.
    let (latent_host, latent_shape) = stage2b_denoise(
        args,
        &cond_ids,
        &uncond_ids,
        t_lat,
        ref_latent_host,
        ref_latent_shape,
        &device,
    )?;

    // Stage 3: VAE decode + mp4.
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(args.width, args.height, args.num_frames));
    stage3_decode_and_mux(args, latent_host, latent_shape, &device, &output)?;

    log::info!(
        "[lance_i2v] done in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

fn main() -> ExitCode {
    env_logger::init();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    if let Err(e) = run(&args) {
        log::error!("[lance_i2v] FAILED: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
