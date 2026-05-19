//! Lance T2V — pure-Rust CLI for the Lance 3B text-to-video pipeline.
//!
//! Reference: `/home/alex/Lance/inference_lance.sh` with `TASK_NAME=t2v`
//! (Python: `Lance.validation_gen_KVcache` → `Wan2_2_VAE.decode`).
//!
//! Pipeline (staged execution — MANDATORY for remote-no-reboot per
//! `.claude/port-docs/CONTEXT.md`):
//!   Stage 1 — tokenize cond + uncond prompts (CPU-only).
//!   Stage 2 — load Lance video DiT, precompute mRoPE, run prefill +
//!             flow-matching denoise, copy final latent to host, drop Lance.
//!   Stage 3 — load `Wan22VaeDecoder`, call `decode()` (video path, not
//!             `decode_image`), convert tensor → u8 frames, write mp4, drop VAE.
//!
//! Latent T-axis relationship to pixel T (Wan22 VAE):
//!   T_pixel = (T_lat - 1) * 4 + 1   (PORT_SPEC_T2V.md, vae2_2.py:787-813)
//!   so `--num-frames 49` → T_lat = 13.
//! The CLI flag is `--num-frames` (pixel count); we derive T_lat for the
//! initial-noise tensor.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use inference_flame::models::lance::{Lance, LanceConfig};
use inference_flame::mux::{video_tensor_to_rgb_u8, write_mp4_video_only};
use inference_flame::vae::wan22_vae::Wan22VaeDecoder;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;

// Default prompt (deliberately motion-forward — T2I prompt is too static).
const DEFAULT_PROMPT: &str = "A majestic orchestral conductor stands on a cliff at sunset, golden light streaming, conducting an unseen symphony as glowing notes flow from her baton into the sky, hair and dress flowing in the wind, cinematic, dramatic camera push-in, painterly realism, ocean waves crashing far below.";

// Wan22 VAE T_lat → T_pixel formula.
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
            // Default points at the T2V checkpoint dir (see PORT_SPEC_T2V.md).
            model_path: PathBuf::from(
                "/home/alex/.serenity/models/lance/Lance_3B_Video",
            ),
            vae_path: PathBuf::from(
                "/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors",
            ),
            prompt: DEFAULT_PROMPT.to_string(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            // 49 = 12*4 + 1 → T_lat = 13. Matches Python Lance T2V defaults.
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
    Ok(a)
}

fn print_usage() {
    eprintln!(
        r#"lance_t2v — Lance 3B text-to-video generation

Usage:
  lance_t2v [options]

Options:
  --model-path  <DIR>   Lance_3B_Video weights dir  [default: /home/alex/.serenity/models/lance/Lance_3B_Video]
  --vae-path    <PATH>  Wan22 VAE safetensors      [default: /home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors]
  --prompt      <STR>   prompt text                [default: motion-forward conductor prompt]
  --negative-prompt <STR> uncond prompt            [default: ""]
  --width       <N>     output width  (mod 16)     [default: 512]
  --height      <N>     output height (mod 16)     [default: 512]
  --num-frames  <N>     pixel frames; T_lat=1+(N-1)/4 [default: 49]
  --fps         <F>     mp4 frame rate             [default: 16.0]
  --steps       <N>     denoise steps              [default: 30]
  --shift       <F>     flow-matching shift        [default: 3.5]
  --cfg         <F>     CFG scale                  [default: 4.0]
  --seed        <N>     RNG seed                   [default: 42]
  --output      <PATH>  mp4 output path            [default: inference-flame/output/lance_t2v_<W>x<H>_<F>f_<DATE>.mp4]
"#
    );
}

fn default_output_path(width: usize, height: usize, num_frames: usize) -> PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(format!(
        "inference-flame/output/lance_t2v_{}x{}_{}f_{}.mp4",
        width, height, num_frames, secs
    ))
}

// ---------------------------------------------------------------------------
// Tokenizer + prompt template — mirrors lance_t2i.rs.
// The Lance Python `system_prompt_render.py` uses a different system_prompt
// for T2V vs T2I: T2V's `system_prompt_type="t2v"`, `vision_type="video"`.
// Python `data/common.py::generate_system_prompt` constructs the T2V variant.
// ---------------------------------------------------------------------------

const IM_START_ID: i32 = 151644;
const IM_END_ID: i32 = 151645;

// Verbatim from Lance Python `data/common.py:30-31`
// (`generate_system_prompt(system_prompt_type="t2v", vision_type="video")`).
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

// Same deterministic Box-Muller noise generator as lance_t2i.rs / sensenova_u1.
fn deterministic_normal_noise(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    data
}

// ---------------------------------------------------------------------------
// Watchdog (advisory) — same as lance_t2i.rs.
// ---------------------------------------------------------------------------

fn spawn_watchdog(start: Instant) {
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let elapsed = start.elapsed().as_secs();
        if elapsed > 20 * 60 {
            log::warn!(
                "[lance_t2v watchdog] wall time {} s exceeds 20 min cap (advisory)",
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
                                "[lance_t2v watchdog] GPU0 temperature {}°C exceeds 78°C cap (advisory)",
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
// Stage 2: Lance denoise — same shape contract as lance_t2i but with T_lat>1.
// ---------------------------------------------------------------------------

fn stage2_denoise(
    args: &Args,
    cond_ids: &[i32],
    uncond_ids: &[i32],
    t_lat: usize,
    device: &Arc<CudaDevice>,
) -> Result<(Vec<f32>, Vec<usize>)> {
    log::info!("[lance_t2v] Stage 2: load Lance, precompute mRoPE, denoise");

    let mut cfg = LanceConfig::default_3b(device.clone());
    cfg.num_inference_steps = args.steps;
    cfg.timestep_shift = args.shift;
    cfg.cfg_text_scale = args.cfg;
    let cfg = Arc::new(cfg);

    // Wan22 VAE spatial is 16×.
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

    let max_pos = cond_ids.len().max(uncond_ids.len()) + 1000 + l_image + 64;
    log::info!(
        "[lance_t2v]   latent grid T={} H={} W={} (L_image={}), max_pos={}",
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
        "[lance_t2v]   Lance loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let mrope = lance.precompute_mrope(max_pos, cfg.dtype)?;

    let cond_tokens = ids_to_tensor(cond_ids, device)?;
    let uncond_tokens = ids_to_tensor(uncond_ids, device)?;

    // Initial noise: torch.randn-compatible Philox stream (= bit-identical
    // to Python's `torch.randn((L, C), generator=Generator(device='cuda').
    // manual_seed(seed))` inside Lance). Reshape (L, C) → (T, H, W, C) →
    // permute (C, T, H, W) → unsqueeze batch, mirroring how Lance Python's
    // `randn(L, C)` packed-sequence maps onto the [B, C, T, H, W] gen_step
    // input. Verified by `parity_lance_t2v --noise-from` cross-check.
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
        "[lance_t2v]   initial noise shape={:?}, seed={}",
        initial_noise.shape().dims(),
        args.seed
    );

    // `t2v_with_cfg` mirrors Python's packed-sequence T2V denoise path:
    // [SOI, vae*768, EOI] with Python-matching mRoPE positions (no +1000
    // shift) and post-CFG renorm. Parity-verified 2026-05-19 against
    // `lance.py:1660-1769` (G1+G2+G4 in T2V_DENOISE_PORT.md). At step 0
    // with shared noise, rs_v_cond cos vs Python = 0.983.
    let t1 = Instant::now();
    let latent = lance.t2v_with_cfg(&cond_tokens, &uncond_tokens, &initial_noise, &mrope)?;
    log::info!(
        "[lance_t2v]   denoise complete in {:.1}s",
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
// ---------------------------------------------------------------------------

fn stage3_decode_and_mux(
    args: &Args,
    latent_host: Vec<f32>,
    latent_shape: Vec<usize>,
    device: &Arc<CudaDevice>,
    output: &Path,
) -> Result<()> {
    log::info!("[lance_t2v] Stage 3: load Wan22 VAE, decode video, mux mp4");
    let t0 = Instant::now();
    let vae = Wan22VaeDecoder::load(&args.vae_path, device)
        .with_context(|| format!("Wan22VaeDecoder::load({})", args.vae_path.display()))?;
    log::info!(
        "[lance_t2v]   VAE loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let latent_gpu = Tensor::from_vec(latent_host, Shape::from_dims(&latent_shape), device.clone())?
        .to_dtype(DType::BF16)?;
    log::info!(
        "[lance_t2v]   decoding latent shape={:?}",
        latent_gpu.shape().dims()
    );

    // Video decode path (NOT decode_image). `decode` runs the per-frame
    // feat_cache loop and unpatchify_5d head shipped in BUILD_PLAN_T2V T2.
    let t1 = Instant::now();
    let video = vae.decode(&latent_gpu)?;
    let video_dims = video.shape().dims().to_vec();
    log::info!(
        "[lance_t2v]   decode complete in {:.1}s, video shape={:?}",
        t1.elapsed().as_secs_f32(),
        video_dims
    );

    // Expected: [1, 3, T_pixel, H_pixel, W_pixel]
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

    // Read decoded frames to host as a flat F32 buffer in `[3, T, H, W]`
    // memory order (post-squeeze of B=1).
    let video_host = video.to_dtype(DType::F32)?.to_vec_f32()?;
    drop(video);
    if video_host.len() != 3 * t_pixel * h_pixel * w_pixel {
        return Err(anyhow!(
            "video buffer length {} != 3*T*H*W ({})",
            video_host.len(),
            3 * t_pixel * h_pixel * w_pixel
        ));
    }

    // [3, T, H, W] f32 in [-1, 1] → [T, H, W, 3] u8.
    let rgb = video_tensor_to_rgb_u8(&video_host, t_pixel, h_pixel, w_pixel);

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    write_mp4_video_only(output, &rgb, t_pixel, w_pixel, h_pixel, args.fps)
        .with_context(|| format!("write_mp4_video_only → {}", output.display()))?;
    log::info!("[lance_t2v]   wrote mp4 → {}", output.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Run + main
// ---------------------------------------------------------------------------

fn run(args: &Args) -> Result<()> {
    let start = Instant::now();
    spawn_watchdog(start);

    let device = flame_core::global_cuda_device();

    // ---- STAGE 1: tokenize. ----
    log::info!("[lance_t2v] Stage 1: tokenize");
    let tok = load_tokenizer(&args.model_path)?;
    let cond_ids = tokenize_to_ids(&tok, &args.prompt, args.text_template)?;
    let uncond_ids = tokenize_to_ids(&tok, &args.negative_prompt, args.text_template)?;
    log::info!(
        "[lance_t2v]   cond_len={}  uncond_len={}",
        cond_ids.len(),
        uncond_ids.len()
    );
    if uncond_ids.is_empty() {
        return Err(anyhow!(
            "uncond_len=0; pass --negative-prompt with at least one non-empty token \
             (open item: empty-uncond CUDA INVALID_VALUE crash)"
        ));
    }
    drop(tok);

    // ---- Derive T_lat from --num-frames. ----
    if args.num_frames < 1 {
        return Err(anyhow!("--num-frames must be >= 1"));
    }
    if (args.num_frames - 1) % 4 != 0 {
        log::warn!(
            "[lance_t2v] --num-frames={} doesn't satisfy (N-1) %% 4 == 0; \
             video decoder produces T_pixel=(T_lat-1)*4+1 = {}",
            args.num_frames,
            t_pixel_from_t_lat(t_lat_from_pixel_frames(args.num_frames))
        );
    }
    let t_lat = t_lat_from_pixel_frames(args.num_frames);
    log::info!(
        "[lance_t2v]   T_pixel_request={} → T_lat={} → T_pixel_actual={}",
        args.num_frames,
        t_lat,
        t_pixel_from_t_lat(t_lat)
    );

    // ---- STAGE 2: Lance denoise. ----
    let (latent_host, latent_shape) =
        stage2_denoise(args, &cond_ids, &uncond_ids, t_lat, &device)?;

    // ---- STAGE 3: VAE decode + mp4. ----
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(args.width, args.height, args.num_frames));
    stage3_decode_and_mux(args, latent_host, latent_shape, &device, &output)?;

    log::info!(
        "[lance_t2v] done in {:.1}s",
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
        log::error!("[lance_t2v] FAILED: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
