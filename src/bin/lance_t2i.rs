//! Lance T2I — pure-Rust CLI for the Lance 3B image generation pipeline.
//!
//! Reference: `/home/alex/Lance/modeling/lance/lance.py::Lance.validation_gen_KVcache`
//! + `inference_lance.sh` (shell defaults: steps=30, shift=3.5, cfg=4.0).
//!
//! Pipeline (staged execution — MANDATORY for remote-no-reboot per
//! `.claude/port-docs/CONTEXT.md`):
//!   Stage 1 — tokenize cond + uncond prompts (CPU-only, no model).
//!   Stage 2 — load Lance, precompute mRoPE, run prefill + denoise loop,
//!             copy final latent to host, DROP Lance to free GPU memory.
//!   Stage 3 — load `Wan22VaeDecoder`, decode latent → image, save PNG, drop VAE.
//!
//! Never have all three stages resident at once. Lance is ~12 GB BF16 resident
//! and Wan22 VAE decode peak is several GB — together they would OOM 24 GB.
//!
//! GPU budgets (per CONTEXT.md): hard caps are 78°C and 20 min wall. This bin
//! does NOT auto-kill; the user is expected to ctrl-C if numbers go wrong. A
//! lightweight background watchdog logs a warning if either threshold is
//! crossed.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use inference_flame::models::lance::{Lance, LanceConfig};
use inference_flame::vae::wan22_vae::Wan22VaeDecoder;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Default prompt (PROMPTS.md primary — DO NOT substitute a shorter test
// prompt; this is the user-specified evaluation prompt).
// ---------------------------------------------------------------------------

const DEFAULT_PROMPT: &str = "A majestic symphonic fantasy concept art image of a woman standing on the edge of a black ocean beneath a cathedral of storm clouds. The song should appear as visible orchestral power: enormous translucent sound waves rise from the sea like glass tidal walls, each wave etched with glowing musical notes, choir symbols, violin lines, and drumbeat patterns. Behind the woman, a ruined marble stage emerges from the water, covered with broken harps, burning candles, and silver roses trembling from bass resonance. The sky splits open into radiant gold and icy blue light, shaped like a heavenly choir frozen in a crescendo. Her long dark dress moves upward from the sonic pressure, while her hands hold a glowing heart shaped music box leaking streams of sheet music into the wind. The mood should feel tragic, grand, romantic, and overwhelming, balancing beauty and devastation. Use epic scale, cinematic lighting, painterly realism, detailed ocean spray, sacred architecture, visible resonance rings, emotional intensity, and the feeling of an entire symphony becoming a storm around one lonely soul.";

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    vae_path: PathBuf,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    steps: usize,
    shift: f32,
    cfg: f32,
    seed: u64,
    output: Option<PathBuf>,
    /// `true` = render full Jinja prompt template (matches Python shell
    /// `TEXT_TEMPLATE=true`). `false` = minimal wrap (im_start + caption +
    /// im_end). Default true to match creators.
    text_template: bool,
}

impl Args {
    fn defaults() -> Self {
        Self {
            model_path: PathBuf::from("/home/alex/.serenity/models/lance/Lance_3B"),
            vae_path: PathBuf::from("/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors"),
            prompt: DEFAULT_PROMPT.to_string(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
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
            "--width" => {
                a.width = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--height" => {
                a.height = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
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
            "--output" | "-o" => a.output = Some(PathBuf::from(next()?)),
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
        r#"lance_t2i — Lance 3B text-to-image generation

Usage:
  lance_t2i [options]

Options:
  --model-path  <DIR>   Lance_3B weights dir   [default: /home/alex/.serenity/models/lance/Lance_3B]
  --vae-path    <PATH>  Wan22 VAE safetensors  [default: /home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors]
  --prompt      <STR>   prompt text            [default: symphonic-fantasy primary]
  --negative-prompt <STR> uncond prompt        [default: ""]
  --width       <N>     output width           [default: 512]
  --height      <N>     output height          [default: 512]
  --steps       <N>     denoise steps          [default: 30]
  --shift       <F>     flow-matching shift    [default: 3.5]
  --cfg         <F>     CFG scale              [default: 4.0]
  --seed        <N>     RNG seed               [default: 42]
  --output      <PATH>  PNG output path        [default: inference-flame/output/lance_<W>_<STEPS>steps_<DATE>.png]
"#
    );
}

fn default_output_path(width: usize, steps: usize) -> PathBuf {
    // Use seconds-since-epoch as a coarse "date" tag — avoids pulling in chrono.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(format!(
        "inference-flame/output/lance_{}_{}steps_{}.png",
        width, steps, secs
    ))
}

// ---------------------------------------------------------------------------
// Helpers — tokenizer, deterministic noise, PNG save
// ---------------------------------------------------------------------------

/// Load the Lance Qwen2 fast tokenizer from `<model_path>/tokenizer.json`.
///
/// Lance ships a unified `tokenizer.json` (verified via `ls Lance_3B/`), so
/// the simpler `Tokenizer::from_file` path applies. Sensenova-U1's
/// `vocab.json`+`merges.txt`+`added_tokens.json` reconstruction is NOT needed.
fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
    let p = model_path.join("tokenizer.json");
    Tokenizer::from_file(&p).map_err(|e| anyhow!("Tokenizer::from_file({}): {e}", p.display()))
}

/// Encode `text` for Lance prefill.
///
/// Two wrap modes:
///   - `template=false` ("simple wrap"): `[<|im_start|>] + tokens + [<|im_end|>]`.
///     Mirrors `validation_dataset.py:process_text` lines 343-358 (the
///     `text_template=False` branch). Good enough for simple prompts.
///   - `template=true` ("full Jinja"): renders the Python `JINJA_PROMPT_TMPL`
///     from `system_prompt_render.py` for the T2I task — system_prompt +
///     user caption + assistant role marker (prefill stops here; the model
///     generates the vision tokens that would follow). Matches the Python
///     `inference_lance.sh` default `TEXT_TEMPLATE=true`. The T2I
///     `system_prompt` comes from `data/common.py::generate_system_prompt`
///     (system_prompt_type="t2i", vision_type="image").
const IM_START_ID: i32 = 151644; // <|im_start|>
const IM_END_ID: i32 = 151645;   // <|im_end|>

const T2I_SYSTEM_PROMPT: &str = "Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

fn tokenize_to_ids(tok: &Tokenizer, text: &str, template: bool) -> Result<Vec<i32>> {
    if template {
        let rendered = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            T2I_SYSTEM_PROMPT, text
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

/// Build a 1D I32 token-id tensor `[L]` on `device`. This is what
/// `Lance::prefill_text_context` expects (it casts to I32 internally if not
/// already I32; we pre-cast for clarity).
fn ids_to_tensor(ids: &[i32], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let n = ids.len();
    // F32 → I32 storage cast is the same idiom as in lance.rs:4139.
    let f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let t = Tensor::from_vec(f, Shape::from_dims(&[n]), device.clone())?;
    t.to_dtype(DType::I32)
        .map_err(|e| anyhow!("cast tokens to I32: {e}"))
}

/// Box-Muller seeded standard-normal noise. Same RNG path as
/// `sensenova_u1_gen::make_noise_image` for cross-bin determinism.
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

/// Save a `[1, 3, H, W]` BF16/F32 tensor in `[-1, 1]` (Wan22 decoder clamp
/// range, per `Wan22VaeDecoder::decode_image`) as an 8-bit PNG.
fn save_png(image: &Tensor, path: &Path) -> Result<()> {
    let f32_img = image.to_dtype(DType::F32)?;
    let dims = f32_img.shape().dims().to_vec();
    // Accept either [1, 3, H, W] or [1, 3, T=1, H, W] (Wan22 image-mode keeps
    // T=1; some callers may squeeze beforehand).
    let (h, w, data) = match dims.len() {
        4 => {
            if dims[0] != 1 || dims[1] != 3 {
                return Err(anyhow!("save_png expects [1, 3, H, W], got {dims:?}"));
            }
            (dims[2], dims[3], f32_img.to_vec_f32()?)
        }
        5 => {
            if dims[0] != 1 || dims[1] != 3 || dims[2] != 1 {
                return Err(anyhow!(
                    "save_png expects [1, 3, 1, H, W] for 5D, got {dims:?}"
                ));
            }
            (dims[3], dims[4], f32_img.to_vec_f32()?)
        }
        _ => return Err(anyhow!("save_png expects 4D or 5D, got {dims:?}")),
    };
    let plane = h * w;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                // Wan22 decoder clamps to [-1, 1]; map to [0, 1] then [0, 255].
                let v = data[c * plane + y * w + x];
                let v01 = (v + 1.0) * 0.5;
                let u = (v01.clamp(0.0, 1.0) * 255.0).round() as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("RgbImage::from_raw failed"))?
        .save(path)
        .with_context(|| format!("save PNG → {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Hard-cap watchdog (advisory, per CONTEXT.md GPU budgets)
// ---------------------------------------------------------------------------

/// Spawn a background thread that polls `nvidia-smi` every 5s and logs a
/// warning when temperature exceeds 78°C or wall time exceeds 20 min. Does
/// NOT kill the process — the user is expected to ctrl-C if numbers diverge.
fn spawn_watchdog(start: Instant) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let elapsed = start.elapsed().as_secs();
            if elapsed > 20 * 60 {
                log::warn!(
                    "[lance_t2i watchdog] wall time {} s exceeds 20 min cap (advisory)",
                    elapsed
                );
            }
            let out = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ])
                .output();
            if let Ok(out) = out {
                if let Ok(s) = std::str::from_utf8(&out.stdout) {
                    if let Some(line) = s.lines().next() {
                        if let Ok(t) = line.trim().parse::<u32>() {
                            if t >= 78 {
                                log::warn!(
                                    "[lance_t2i watchdog] GPU0 temperature {}°C exceeds 78°C cap (advisory)",
                                    t
                                );
                            }
                        }
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Stage 2: Lance load + denoise — scoped so Lance drops before VAE load.
// ---------------------------------------------------------------------------

/// Run Stage 2: load Lance, precompute mRoPE, prefill cond+uncond, denoise
/// loop. Returns (host_f32_latent_flat, shape) so Lance can be dropped before
/// the caller loads the VAE.
fn stage2_denoise(
    args: &Args,
    cond_ids: &[i32],
    uncond_ids: &[i32],
    device: &Arc<CudaDevice>,
) -> Result<(Vec<f32>, Vec<usize>)> {
    log::info!("[lance_t2i] Stage 2: load Lance, precompute mRoPE, denoise");

    // Lance config — overridable from CLI args.
    let mut cfg = LanceConfig::default_3b(device.clone());
    cfg.num_inference_steps = args.steps;
    cfg.timestep_shift = args.shift;
    cfg.cfg_text_scale = args.cfg;
    let cfg = Arc::new(cfg);

    // Wan22 VAE is 16× spatial: 3 up_stages (2^3=8) × unpatchify_5d(_, 2)
    // at the head (×2) = 16. T=1 for image mode.
    if args.width % 16 != 0 || args.height % 16 != 0 {
        return Err(anyhow!(
            "width/height must be divisible by 16 (Wan22 VAE spatial factor); got {}x{}",
            args.width,
            args.height
        ));
    }
    let h_latent = args.height / 16;
    let w_latent = args.width / 16;
    let t_latent: usize = 1;
    let l_image = t_latent * h_latent * w_latent;

    // mRoPE max position: cover text prefill + image gen tokens + the 1000
    // pos_shift used by `gen_step` to keep image-token positions disjoint
    // from text. Adding a small safety margin.
    let max_pos = cond_ids.len().max(uncond_ids.len()) + 1000 + l_image + 64;
    log::info!(
        "[lance_t2i]   latent grid T={} H={} W={} (L_image={}), max_pos={}",
        t_latent,
        h_latent,
        w_latent,
        l_image,
        max_pos
    );

    let t0 = Instant::now();
    let lance = Lance::load(&args.model_path, cfg.clone(), device)
        .with_context(|| format!("Lance::load({})", args.model_path.display()))?;
    log::info!(
        "[lance_t2i]   Lance loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let mrope = lance.precompute_mrope(max_pos, cfg.dtype)?;

    let cond_tokens = ids_to_tensor(cond_ids, device)?;
    let uncond_tokens = ids_to_tensor(uncond_ids, device)?;

    // Initial noise: [B=1, C=patch_latent_dim, T=1, H_latent, W_latent].
    let c = cfg.patch_latent_dim();
    let noise_n = 1 * c * t_latent * h_latent * w_latent;
    let noise = deterministic_normal_noise(args.seed, noise_n);
    let initial_noise = Tensor::from_vec(
        noise,
        Shape::from_dims(&[1, c, t_latent, h_latent, w_latent]),
        device.clone(),
    )?
    .to_dtype(cfg.dtype)?;
    log::info!(
        "[lance_t2i]   initial noise shape={:?}, seed={}",
        initial_noise.shape().dims(),
        args.seed
    );

    let t1 = Instant::now();
    let latent = lance.t2i_with_cfg(&cond_tokens, &uncond_tokens, &initial_noise, &mrope)?;
    log::info!(
        "[lance_t2i]   denoise complete in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    // Copy to host BEFORE dropping Lance. `to_vec_f32` reads the device
    // buffer and returns a host `Vec<f32>` regardless of source dtype.
    let host_data = latent.to_dtype(DType::F32)?.to_vec_f32()?;
    let shape = latent.shape().dims().to_vec();

    // `lance` drops here at scope-exit → ~12 GB BF16 weights + KV caches +
    // activations are freed before Stage 3 loads the VAE.
    drop(lance);
    drop(mrope);

    Ok((host_data, shape))
}

// ---------------------------------------------------------------------------
// Stage 3: VAE decode — scoped so the VAE drops after PNG save.
// ---------------------------------------------------------------------------

fn stage3_decode_and_save(
    args: &Args,
    latent_host: Vec<f32>,
    latent_shape: Vec<usize>,
    device: &Arc<CudaDevice>,
    output: &Path,
) -> Result<()> {
    log::info!("[lance_t2i] Stage 3: load Wan22 VAE, decode, save");
    let t0 = Instant::now();
    let vae = Wan22VaeDecoder::load(&args.vae_path, device)
        .with_context(|| format!("Wan22VaeDecoder::load({})", args.vae_path.display()))?;
    log::info!(
        "[lance_t2i]   VAE loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let latent_gpu = Tensor::from_vec(latent_host, Shape::from_dims(&latent_shape), device.clone())?
        .to_dtype(DType::BF16)?;
    log::info!(
        "[lance_t2i]   decoding latent shape={:?}",
        latent_gpu.shape().dims()
    );

    let t1 = Instant::now();
    let image = vae.decode_image(&latent_gpu)?;
    log::info!(
        "[lance_t2i]   decode complete in {:.1}s, image shape={:?}",
        t1.elapsed().as_secs_f32(),
        image.shape().dims()
    );

    save_png(&image, output)?;
    log::info!("[lance_t2i]   saved → {}", output.display());

    drop(vae);
    Ok(())
}

// ---------------------------------------------------------------------------
// Run + main
// ---------------------------------------------------------------------------

fn run(args: &Args) -> Result<()> {
    let start = Instant::now();
    spawn_watchdog(start);

    let device = flame_core::global_cuda_device();

    // ---- STAGE 1: tokenize (CPU). ----
    log::info!("[lance_t2i] Stage 1: tokenize");
    let tok = load_tokenizer(&args.model_path)?;
    let cond_ids = tokenize_to_ids(&tok, &args.prompt, args.text_template)?;
    let uncond_ids = tokenize_to_ids(&tok, &args.negative_prompt, args.text_template)?;
    log::info!(
        "[lance_t2i]   cond_len={}  uncond_len={}",
        cond_ids.len(),
        uncond_ids.len()
    );
    // Drop the tokenizer — its vocab tables are no longer needed.
    drop(tok);

    // ---- STAGE 2: Lance denoise. Result: host-side latent + shape. ----
    let (latent_host, latent_shape) = stage2_denoise(args, &cond_ids, &uncond_ids, &device)?;

    // ---- STAGE 3: VAE decode + PNG save. ----
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(args.width, args.steps));
    stage3_decode_and_save(args, latent_host, latent_shape, &device, &output)?;

    log::info!(
        "[lance_t2i] done in {:.1}s",
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
        log::error!("[lance_t2i] FAILED: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
