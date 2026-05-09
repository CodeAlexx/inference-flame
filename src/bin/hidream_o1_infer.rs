//! HiDream-O1-Image — pure-Rust T2I inference CLI.
//!
//! Mirrors `/home/alex/HiDream-O1-Image/inference.py:23-91` (Python CLI).
//!
//! Usage:
//!   hidream_o1_infer [--model-path /path/to/HiDream-O1-Image-Dev-weights]
//!                    [--prompt "..."] [--negative-prompt " "]
//!                    [--output-image output.png]
//!                    [--height 2048] [--width 2048]
//!                    [--model-type dev|full]
//!                    [--seed 32] [--guidance-scale 5.0]
//!
//! Build only — does NOT run by default. Caller is expected to invoke the
//! built binary themselves.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use flame_core::{global_cuda_device, DType};

use inference_flame::models::hidream_o1::{
    FlashFlowMatchEulerDiscreteScheduler, HiDreamO1Config, HiDreamO1Pipeline,
    HiDreamO1WeightLoader,
};

const DEFAULT_MODEL_PATH: &str = "/home/alex/HiDream-O1-Image-Dev-weights";
const DEFAULT_PROMPT: &str =
    "medium shot, eye-level, front view. A woman is seated in an ornate bedroom, \
     illuminated by candlelight, with a calm and composed expression. The subject \
     is a young woman with fair skin, light brown hair styled in an updo with loose \
     tendrils framing her face, and blue eyes. She wears a cream-colored satin robe \
     with delicate floral embroidery and lace trim along the neckline.";
const DEFAULT_OUTPUT: &str = "output.png";

struct Args {
    model_path: PathBuf,
    prompt: String,
    negative_prompt: String,
    output_image: PathBuf,
    height: usize,
    width: usize,
    model_type: String,
    seed: u64,
    guidance_scale: f32,
    allow_any_resolution: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(DEFAULT_MODEL_PATH),
            prompt: DEFAULT_PROMPT.to_string(),
            negative_prompt: String::new(),
            output_image: PathBuf::from(DEFAULT_OUTPUT),
            height: 2048,
            width: 2048,
            model_type: "dev".to_string(),
            seed: 32,
            // Per inference.py: dev variant uses guidance_scale=0.0 (no CFG);
            // full variant uses 5.0. We default to dev here, but keep the
            // CLI flag to override.
            guidance_scale: 0.0,
            allow_any_resolution: false,
        }
    }
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args::default();
    let mut i = 0;
    while i < argv.len() {
        let take = |i: usize| -> Result<&str> {
            argv.get(i + 1).map(String::as_str).ok_or_else(|| {
                anyhow::anyhow!("flag {} expects a value", argv[i])
            })
        };
        match argv[i].as_str() {
            "--model-path" | "--model_path" => {
                a.model_path = PathBuf::from(take(i)?);
                i += 2;
            }
            "--prompt" => {
                a.prompt = take(i)?.to_string();
                i += 2;
            }
            "--negative-prompt" | "--negative_prompt" | "--neg-prompt" => {
                a.negative_prompt = take(i)?.to_string();
                i += 2;
            }
            "--output-image" | "--output_image" | "--out" => {
                a.output_image = PathBuf::from(take(i)?);
                i += 2;
            }
            "--height" => {
                a.height = take(i)?.parse().context("--height parse")?;
                i += 2;
            }
            "--width" => {
                a.width = take(i)?.parse().context("--width parse")?;
                i += 2;
            }
            "--model-type" | "--model_type" => {
                a.model_type = take(i)?.to_string();
                i += 2;
            }
            "--seed" => {
                a.seed = take(i)?.parse().context("--seed parse")?;
                i += 2;
            }
            "--guidance-scale" | "--guidance_scale" | "--cfg" => {
                a.guidance_scale = take(i)?.parse().context("--guidance-scale parse")?;
                i += 2;
            }
            "--allow-any-resolution" | "--allow_any_resolution" => {
                a.allow_any_resolution = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!("Usage: hidream_o1_infer [options]");
                println!("  --model-path <dir>            (default: {DEFAULT_MODEL_PATH})");
                println!("  --prompt <str>");
                println!("  --negative-prompt <str>       (default: empty -> CFG disabled)");
                println!("  --output-image <path>         (default: {DEFAULT_OUTPUT})");
                println!("  --height <int>                (default: 2048)");
                println!("  --width <int>                 (default: 2048)");
                println!("  --model-type dev|full         (default: dev)");
                println!("  --seed <int>                  (default: 32)");
                println!("  --guidance-scale <float>      (default: 0.0 for dev, suggest 5.0 for full)");
                println!("  --allow-any-resolution        (smoke-test: bypass predefined-resolution snap)");
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    env_logger::init();

    // Pure inference — disable autograd so the per-step compute graph
    // doesn't accumulate across the 28-step denoise loop (otherwise
    // every step's intermediates are retained by the tape and OOM
    // hits around step 7-8 even at 1024²).
    flame_core::AutogradContext::set_enabled(false);

    let args = parse_args()?;

    log::info!("[hidream_o1_infer] model={}", args.model_path.display());
    log::info!("[hidream_o1_infer] prompt: {}", &args.prompt);
    log::info!(
        "[hidream_o1_infer] {}x{} model_type={} seed={} guidance_scale={}",
        args.width, args.height, args.model_type, args.seed, args.guidance_scale,
    );

    // 1) Tokenizer.
    let tokenizer_path = args.model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer::from_file({}): {e}", tokenizer_path.display()))?;

    // 2) Config — bake-in defaults match config.json's text_config (verified
    //    at scope time). Token IDs are validated by HiDreamO1Pipeline::new
    //    against the loaded tokenizer.
    let config = HiDreamO1Config::dev_8b();

    // 3) Device + weight loader.
    let device = global_cuda_device();
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .with_context(|| format!("loader for {}", args.model_path.display()))?;
    let model = loader
        .load_model(&config, &device)
        .context("HiDreamO1WeightLoader::load_model")?;

    // 4) Scheduler — Dev uses 28-step Flash, Full uses 50-step Default.
    let scheduler = match args.model_type.as_str() {
        "dev" => FlashFlowMatchEulerDiscreteScheduler::dev_28step(),
        "full" => FlashFlowMatchEulerDiscreteScheduler::full_50step(),
        other => bail!("--model-type must be 'dev' or 'full', got '{other}'"),
    };

    // 5) Pipeline (validates token IDs).
    let mut pipeline = HiDreamO1Pipeline::new(
        model,
        scheduler,
        tokenizer,
        config,
        device,
        DType::BF16,
    )
    .context("HiDreamO1Pipeline::new")?;
    pipeline.set_allow_any_resolution(args.allow_any_resolution);

    // 6) Generate.
    log::info!("[hidream_o1_infer] starting denoise loop...");
    let t0 = std::time::Instant::now();
    let image = pipeline
        .generate(
            &args.prompt,
            &args.negative_prompt,
            args.height,
            args.width,
            args.seed,
            args.guidance_scale,
        )
        .context("HiDreamO1Pipeline::generate")?;
    let elapsed = t0.elapsed();
    log::info!(
        "[hidream_o1_infer] denoise done in {:.2}s",
        elapsed.as_secs_f64()
    );

    // 7) Save PNG.
    HiDreamO1Pipeline::save_png(&image, &args.output_image)
        .with_context(|| format!("save_png to {}", args.output_image.display()))?;

    log::info!(
        "[hidream_o1_infer] done — wrote {}",
        args.output_image.display()
    );
    Ok(())
}
