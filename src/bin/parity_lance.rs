//! parity_lance — capture top-level tensors from the Lance T2I pipeline at the
//! same logical points as the Python reference (`gen_refs.py`).
//!
//! **Capture strategy: Option C (top-level only) for the first parity round.**
//! The full per-block / per-MoT-branch capture surface (Option B — instrumented
//! gen_step / prefill) requires inlining Module 13's body, which is
//! out-of-scope for the C7 deliverable. Top-level captures (input tokens,
//! input noise, final latent, decoded image) are enough to bisect catastrophic
//! divergence; if any of these four diverge, a v2 of this binary will be added
//! with mid-pipeline captures.
//!
//! See `inference-flame/ports/lance/BUILD_PLAN.md` ("Parity plan" section) for
//! the full capture list. Items captured here:
//!   - `input.text_tokens`      → cond token ids (F32 storage)
//!   - `input.uncond_text_tokens` → uncond token ids (F32 storage)
//!   - `input.latent_noise`     → seeded initial noise (F32)
//!   - `lance.final_latent`     → output of `t2i_with_cfg` (F32, CPU)
//!   - `vae_decode.out`         → final image tensor in [-1, 1] (F32)
//!
//! Output: `<refs-out>/<capture_name>.safetensors`, one tensor each, keyed by
//! `<capture_name>` with dots replaced by underscores (safetensors disallows
//! certain characters in some viewers; underscore form is the canonical port
//! convention shared with Python `gen_refs.py`).
//!
//! Staged execution matches `lance_t2i.rs` — Stage 1 tokenize → Stage 2 Lance
//! denoise (drop Lance) → Stage 3 VAE decode (drop VAE). No PNG is written.

use std::collections::HashMap;
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

const DEFAULT_PROMPT: &str = "A majestic symphonic fantasy concept art image of a woman standing on the edge of a black ocean beneath a cathedral of storm clouds. The song should appear as visible orchestral power: enormous translucent sound waves rise from the sea like glass tidal walls, each wave etched with glowing musical notes, choir symbols, violin lines, and drumbeat patterns. Behind the woman, a ruined marble stage emerges from the water, covered with broken harps, burning candles, and silver roses trembling from bass resonance. The sky splits open into radiant gold and icy blue light, shaped like a heavenly choir frozen in a crescendo. Her long dark dress moves upward from the sonic pressure, while her hands hold a glowing heart shaped music box leaking streams of sheet music into the wind. The mood should feel tragic, grand, romantic, and overwhelming, balancing beauty and devastation. Use epic scale, cinematic lighting, painterly realism, detailed ocean spray, sacred architecture, visible resonance rings, emotional intensity, and the feeling of an entire symphony becoming a storm around one lonely soul.";

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
    refs_out: PathBuf,
    #[allow(dead_code)]
    num_captures: usize,
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
            refs_out: PathBuf::from("inference-flame/ports/lance/parity/refs_rust"),
            num_captures: 5,
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
            "--refs-out" | "--refs_out" => a.refs_out = PathBuf::from(next()?),
            "--num-captures" | "--num_captures" => {
                a.num_captures = next()?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
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
        r#"parity_lance — Lance T2I top-level parity captures

Usage:
  parity_lance [options]

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
  --refs-out    <DIR>   capture output dir     [default: inference-flame/ports/lance/parity/refs_rust]
  --num-captures <N>    expected count (advisory) [default: 5]
"#
    );
}

// ---------------------------------------------------------------------------
// Helpers shared in spirit with lance_t2i.rs — duplicated rather than
// promoted to a shared module (bin-only utilities, the duplication is small
// and keeps each bin self-contained).
// ---------------------------------------------------------------------------

fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
    let p = model_path.join("tokenizer.json");
    Tokenizer::from_file(&p).map_err(|e| anyhow!("Tokenizer::from_file({}): {e}", p.display()))
}

fn tokenize_to_ids(tok: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

fn ids_to_tensor(ids: &[i32], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let n = ids.len();
    let f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let t = Tensor::from_vec(f, Shape::from_dims(&[n]), device.clone())?;
    t.to_dtype(DType::I32)
        .map_err(|e| anyhow!("cast tokens to I32: {e}"))
}

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
// Capture writer
// ---------------------------------------------------------------------------

/// Save a single-tensor safetensors file at `<refs_dir>/<name>.safetensors`,
/// keyed by `name.replace('.', '_')`. Tensor is read to F32 host (parity
/// tooling is dtype-agnostic; comparing F32 vs F32 avoids BF16 round-trip
/// noise during compare).
///
/// `name` examples: `"input.text_tokens"` → file `input.text_tokens.safetensors`,
/// key `input_text_tokens`.
fn save_capture(name: &str, tensor: &Tensor, refs_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(refs_dir)
        .with_context(|| format!("mkdir -p {}", refs_dir.display()))?;
    let path = refs_dir.join(format!("{}.safetensors", name));
    let key = name.replace('.', "_");
    let f32_t = tensor.to_dtype(DType::F32)?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert(key.clone(), f32_t);
    flame_core::serialization::save_file(&map, &path)
        .with_context(|| format!("save_file({})", path.display()))?;
    log::info!(
        "[parity_lance]   saved capture '{}' (key='{}') → {}",
        name,
        key,
        path.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 2: Lance denoise. Returns (final_latent_host_f32, latent_shape).
// Captures: input.text_tokens, input.uncond_text_tokens, input.latent_noise,
// lance.final_latent.
// ---------------------------------------------------------------------------

fn stage2_denoise_capture(
    args: &Args,
    cond_ids: &[i32],
    uncond_ids: &[i32],
    device: &Arc<CudaDevice>,
    refs_dir: &Path,
) -> Result<(Vec<f32>, Vec<usize>)> {
    log::info!("[parity_lance] Stage 2: load Lance, precompute mRoPE, denoise");

    let mut cfg = LanceConfig::default_3b(device.clone());
    cfg.num_inference_steps = args.steps;
    cfg.timestep_shift = args.shift;
    cfg.cfg_text_scale = args.cfg;
    let cfg = Arc::new(cfg);

    // Wan22 VAE is 16× spatial: 3 up_stages (2^3=8) × unpatchify_5d(_, 2) at head.
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
    let max_pos = cond_ids.len().max(uncond_ids.len()) + 1000 + l_image + 64;

    // ---- Capture: input.text_tokens (cond) + uncond_text_tokens. ----
    // Store as F32 (parity tooling is dtype-agnostic; ids fit losslessly in F32).
    let cond_tokens = ids_to_tensor(cond_ids, device)?;
    let uncond_tokens = ids_to_tensor(uncond_ids, device)?;
    save_capture("input.text_tokens", &cond_tokens, refs_dir)?;
    save_capture("input.uncond_text_tokens", &uncond_tokens, refs_dir)?;

    // ---- Capture: input.latent_noise. ----
    let c = cfg.patch_latent_dim();
    let noise_n = 1 * c * t_latent * h_latent * w_latent;
    let noise = deterministic_normal_noise(args.seed, noise_n);
    // Store F32 BEFORE casting to BF16, so the captured tensor matches the
    // raw RNG output bit-for-bit. The Lance forward consumes the BF16-cast
    // version below.
    let initial_noise_f32 = Tensor::from_vec(
        noise.clone(),
        Shape::from_dims(&[1, c, t_latent, h_latent, w_latent]),
        device.clone(),
    )?;
    save_capture("input.latent_noise", &initial_noise_f32, refs_dir)?;
    let initial_noise = initial_noise_f32.to_dtype(cfg.dtype)?;

    // ---- Lance load + denoise. ----
    let t0 = Instant::now();
    let lance = Lance::load(&args.model_path, cfg.clone(), device)
        .with_context(|| format!("Lance::load({})", args.model_path.display()))?;
    log::info!(
        "[parity_lance]   Lance loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let mrope = lance.precompute_mrope(max_pos, cfg.dtype)?;
    let t1 = Instant::now();
    let latent = lance.t2i_with_cfg(&cond_tokens, &uncond_tokens, &initial_noise, &mrope)?;
    log::info!(
        "[parity_lance]   denoise complete in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    // ---- Capture: lance.final_latent. ----
    save_capture("lance.final_latent", &latent, refs_dir)?;

    let host = latent.to_dtype(DType::F32)?.to_vec_f32()?;
    let shape = latent.shape().dims().to_vec();
    drop(lance);
    drop(mrope);
    Ok((host, shape))
}

// ---------------------------------------------------------------------------
// Stage 3: VAE decode + capture.
// ---------------------------------------------------------------------------

fn stage3_decode_capture(
    args: &Args,
    latent_host: Vec<f32>,
    latent_shape: Vec<usize>,
    device: &Arc<CudaDevice>,
    refs_dir: &Path,
) -> Result<()> {
    log::info!("[parity_lance] Stage 3: load Wan22 VAE, decode, capture");
    let vae = Wan22VaeDecoder::load(&args.vae_path, device)?;
    let latent_gpu =
        Tensor::from_vec(latent_host, Shape::from_dims(&latent_shape), device.clone())?
            .to_dtype(DType::BF16)?;
    let image = vae.decode_image(&latent_gpu)?;
    save_capture("vae_decode.out", &image, refs_dir)?;
    drop(vae);
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let device = flame_core::global_cuda_device();
    let refs_dir = &args.refs_out;
    std::fs::create_dir_all(refs_dir)
        .with_context(|| format!("mkdir -p {}", refs_dir.display()))?;

    // ---- Stage 1: tokenize. ----
    log::info!("[parity_lance] Stage 1: tokenize");
    let tok = load_tokenizer(&args.model_path)?;
    let cond_ids = tokenize_to_ids(&tok, &args.prompt)?;
    let uncond_ids = tokenize_to_ids(&tok, &args.negative_prompt)?;
    log::info!(
        "[parity_lance]   cond_len={}  uncond_len={}",
        cond_ids.len(),
        uncond_ids.len()
    );
    drop(tok);

    // ---- Stage 2 + Stage 3. ----
    let (latent_host, latent_shape) =
        stage2_denoise_capture(args, &cond_ids, &uncond_ids, &device, refs_dir)?;
    stage3_decode_capture(args, latent_host, latent_shape, &device, refs_dir)?;

    log::info!("[parity_lance] all captures written to {}", refs_dir.display());
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
        log::error!("[parity_lance] FAILED: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
