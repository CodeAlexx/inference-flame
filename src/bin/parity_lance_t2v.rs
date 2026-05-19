//! parity_lance_t2v — capture top-level + step-0 tensors from the Rust Lance
//! T2V denoise path at the same logical points as the Python reference
//! (`parity/gen_refs_lance_t2v.py`).
//!
//! Pair this with `gen_refs_lance_t2v.py`: run both at the same
//! prompt/seed/resolution/frame-count, then diff the resulting
//! `.safetensors` files by name. The first capture where Rust diverges from
//! Python localizes the bug among G1-G5 in `T2V_DENOISE_PORT.md`.
//!
//! Captures written to `<refs-out>/<name>.safetensors`:
//!
//! - `input.text_tokens`               — cond prompt token ids (F32 storage of i32 values)
//! - `input.uncond_text_tokens`        — uncond prompt token ids
//! - `input.latent_noise`              — seeded initial noise, shape [1, C, T_lat, H_lat, W_lat]
//! - `cache.cond_prefix_len`           — cond cache.seq_len(0) after prefill, scalar i64 → f32
//! - `cache.uncond_prefix_len`         — uncond cache.seq_len(0) after prefill, scalar
//! - `geom.t_lat`, `geom.h_lat`, `geom.w_lat`, `geom.l_image` — derived ints as scalars
//! - `step0.v_cond`                    — gen_step output for cond at step 0, [1, C, T, H, W]
//! - `step0.v_uncond`                  — gen_step output for uncond at step 0
//! - `step0.v_cfg`                     — combine_cfg(uncond, cond, cfg) at step 0 (no renorm — gap G4)
//! - `step0.x_t_post_step`             — x_t - v_cfg * dt[0]
//! - `lance.final_latent`              — final tensor after all steps
//! - `vae_decode.out`                  — pixel tensor `[1, 3, T_pixel, H, W]` in [-1, 1]
//!
//! See `T2V_DENOISE_PORT.md` for the structural gap analysis these captures
//! were designed to surface.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use inference_flame::models::lance::{
    combine_cfg, denoise_step, timestep_schedule, Lance, LanceConfig,
};
use inference_flame::vae::wan22_vae::Wan22VaeDecoder;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;

// Default prompt mirrors the lance_t2v.rs binary so identical seeds reproduce
// the same prompt token stream.
const DEFAULT_PROMPT: &str = "A detailed cinematic portrait of a beautiful young woman playing a grand piano in a luminous marble music hall.";

const IM_START_ID: i32 = 151644;
const IM_END_ID: i32 = 151645;

// Verbatim from Lance Python `data/common.py:30-31` for T2V.
const T2V_SYSTEM_PROMPT: &str = "Describe the video by detailing the color, quantity, visible text, shape, size, texture, spatial relationships and motion/camera movements of the objects and background:";

fn t_lat_from_pixel_frames(num_frames: usize) -> usize {
    debug_assert!(num_frames >= 1);
    1 + (num_frames - 1) / 4
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
    steps: usize,
    shift: f32,
    cfg: f32,
    seed: u64,
    refs_out: PathBuf,
    text_template: bool,
    skip_vae: bool,
    noise_from: Option<PathBuf>,
}

impl Args {
    fn defaults() -> Self {
        Self {
            model_path: PathBuf::from("/home/alex/.serenity/models/lance/Lance_3B_Video"),
            vae_path: PathBuf::from("/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors"),
            prompt: DEFAULT_PROMPT.to_string(),
            negative_prompt: String::new(),
            width: 256,
            height: 256,
            num_frames: 9,
            steps: 30,
            shift: 3.5,
            cfg: 4.0,
            seed: 42,
            refs_out: PathBuf::from("inference-flame/ports/lance/parity/refs_t2v_rust"),
            text_template: true,
            skip_vae: false,
            noise_from: None,
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
            "--num-frames" | "--num_frames" => {
                a.num_frames = next()?
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
            "--no-text-template" => a.text_template = false,
            "--skip-vae" => a.skip_vae = true,
            "--noise-from" | "--noise_from" => a.noise_from = Some(PathBuf::from(next()?)),
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
        r#"parity_lance_t2v — Lance T2V parity captures (Rust side)

Usage:
  parity_lance_t2v [options]

Options:
  --model-path  <DIR>   Lance_3B_Video weights dir [default: /home/alex/.serenity/models/lance/Lance_3B_Video]
  --vae-path    <PATH>  Wan22 VAE safetensors      [default: /home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors]
  --prompt      <STR>   prompt text
  --negative-prompt <STR> uncond prompt            [default: ""]
  --width       <N>     output width               [default: 256]
  --height      <N>     output height              [default: 256]
  --num-frames  <N>     pixel frames (N-1 % 4 == 0) [default: 9]
  --steps       <N>     denoise steps              [default: 30]
  --shift       <F>     flow-matching shift        [default: 3.5]
  --cfg         <F>     CFG scale                  [default: 4.0]
  --seed        <N>     RNG seed                   [default: 42]
  --refs-out    <DIR>   capture output dir         [default: inference-flame/ports/lance/parity/refs_t2v_rust]
  --no-text-template    disable chat-template wrap (raw tokens)
  --skip-vae            skip VAE decode (capture lance.final_latent only)
  --noise-from  <PATH>  load Python's input.latent_noise.safetensors instead
                        of generating; eliminates RNG mismatch for parity
"#
    );
}

// ---------------------------------------------------------------------------
// Helpers — mirror lance_t2v.rs idiom so token streams + noise are identical.
// ---------------------------------------------------------------------------

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

/// Load Python's `input.latent_noise.safetensors` (key `input.latent_noise`,
/// shape `(L, C)` where `L = T_lat * H_lat * W_lat`, dtype BF16) and reshape
/// to the Rust gen_step input layout `[1, C, T_lat, H_lat, W_lat]`.
///
/// The Python noise is captured straight from `torch.randn(L, C)` inside
/// Lance, where row L=i corresponds to packed-sequence position i. Lance
/// Python enumerates positions T-major then H then W (row-major over
/// `(T_lat, H_lat, W_lat)`), so reshape `(L, C) → (T, H, W, C)` is direct.
/// Then permute `(3, 0, 1, 2) → (C, T, H, W)` and add a batch dim.
///
/// This matches Rust's `gen_step` patchify path:
///   `latent.permute(0, 2, 3, 4, 1).reshape(L, C)` → `(L=THW, C)` Python-form.
///
/// Returns (capture_lc_f32, model_input_5d_f32).
fn load_python_noise(
    path: &Path,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    c: usize,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let map = flame_core::serialization::load_file(path, device)
        .with_context(|| format!("load_file({})", path.display()))?;
    let key = "input.latent_noise";
    let t = map
        .get(key)
        .ok_or_else(|| anyhow!("{} missing key '{}'; found {:?}", path.display(), key, map.keys().collect::<Vec<_>>()))?
        .clone();
    let l = t_lat * h_lat * w_lat;
    let dims = t.shape().dims().to_vec();
    if dims != [l, c] {
        return Err(anyhow!(
            "noise file shape {:?} != expected [L={}, C={}]; check geometry matches gen_refs_lance_t2v.py run",
            dims,
            l,
            c
        ));
    }
    let lc_f32 = t.to_dtype(DType::F32)?;
    // Reshape (L, C) → (T, H, W, C) → permute (C, T, H, W) → unsqueeze batch.
    let thw_c = lc_f32.reshape(&[t_lat, h_lat, w_lat, c])?;
    let c_thw = thw_c.permute(&[3, 0, 1, 2])?;
    let model_5d = c_thw.reshape(&[1, c, t_lat, h_lat, w_lat])?;
    log::info!(
        "[parity_lance_t2v] loaded Python noise from {} (shape (L,C)=({},{}) → [1,{},{},{},{}])",
        path.display(),
        l,
        c,
        c,
        t_lat,
        h_lat,
        w_lat
    );
    Ok((lc_f32, model_5d))
}

// Same deterministic Box-Muller noise generator as parity_lance.rs / lance_t2v.rs.
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
        "[parity_lance_t2v]   saved capture '{}' (key='{}') → {} shape={:?}",
        name,
        key,
        path.display(),
        tensor.shape().dims()
    );
    Ok(())
}

/// Save a scalar (1-element tensor) for derived geometry/length values.
fn save_scalar(name: &str, value: f32, device: &Arc<CudaDevice>, refs_dir: &Path) -> Result<()> {
    let t = Tensor::from_vec(vec![value], Shape::from_dims(&[1]), device.clone())?;
    save_capture(name, &t, refs_dir)
}

// ---------------------------------------------------------------------------
// Main denoise-with-capture path. Mirrors `Lance::denoise_loop` body but
// captures `v_cond`, `v_uncond`, `v_cfg`, and `x_t` at step 0.
// ---------------------------------------------------------------------------

fn denoise_loop_capture(
    lance: &Lance,
    cfg: &LanceConfig,
    cond_cache: &inference_flame::models::lance::KvCache,
    uncond_cache: &inference_flame::models::lance::KvCache,
    initial_noise: &Tensor,
    mrope: &inference_flame::models::lance::MropeFreqs,
    num_steps: usize,
    shift: f32,
    cfg_scale: f32,
    refs_dir: &Path,
) -> Result<Tensor> {
    if num_steps == 0 {
        return Err(anyhow!("denoise_loop_capture: num_steps must be > 0"));
    }

    // Replicates Lance::denoise_loop schedule build.
    let timesteps = timestep_schedule(num_steps, shift, &cfg.device)?;
    let timesteps_host: Vec<f32> = timesteps.to_dtype(DType::F32)?.to_vec_f32()?;
    if timesteps_host.len() != num_steps + 1 {
        return Err(anyhow!(
            "timestep_schedule returned {} values, expected {}",
            timesteps_host.len(),
            num_steps + 1
        ));
    }
    let dts: Vec<f32> = (0..num_steps)
        .map(|i| timesteps_host[i] - timesteps_host[i + 1])
        .collect();

    let mut cond_cache_local = cond_cache.clone();
    let mut uncond_cache_local = uncond_cache.clone();
    let mut x_t = initial_noise.clone();

    for i in 0..num_steps {
        let t = timesteps_host[i];
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), cfg.device.clone())?;
        let t_tensor = if t_tensor.dtype() != cfg.dtype {
            t_tensor.to_dtype(cfg.dtype)?
        } else {
            t_tensor
        };

        let v_cond = lance.gen_step(&x_t, &t_tensor, mrope, &mut cond_cache_local)?;
        let v_uncond = lance.gen_step(&x_t, &t_tensor, mrope, &mut uncond_cache_local)?;
        let v = combine_cfg(&v_uncond, &v_cond, cfg_scale)?;

        if i == 0 {
            save_capture("step0.v_cond", &v_cond, refs_dir)?;
            save_capture("step0.v_uncond", &v_uncond, refs_dir)?;
            save_capture("step0.v_cfg", &v, refs_dir)?;
            // dt scalar for parity (Python computes the same on its side from
            // the timestep_shift schedule — useful to verify the schedule
            // itself matches).
            save_scalar("step0.dt", dts[0], &cfg.device, refs_dir)?;
            save_scalar("step0.timestep", t, &cfg.device, refs_dir)?;
        }

        x_t = denoise_step(&x_t, &v, dts[i])?;

        if i == 0 {
            save_capture("step0.x_t_post_step", &x_t, refs_dir)?;
        }

        log::info!(
            "[parity_lance_t2v] step {}/{}, t={:.4}, dt={:.4}",
            i + 1,
            num_steps,
            t,
            dts[i]
        );
    }
    Ok(x_t)
}

// ---------------------------------------------------------------------------
// Stage 2: Lance denoise + capture.
// ---------------------------------------------------------------------------

fn stage2_denoise_capture(
    args: &Args,
    cond_ids: &[i32],
    uncond_ids: &[i32],
    t_lat: usize,
    device: &Arc<CudaDevice>,
    refs_dir: &Path,
) -> Result<(Vec<f32>, Vec<usize>)> {
    log::info!("[parity_lance_t2v] Stage 2: load Lance, precompute mRoPE, denoise w/ capture");

    let mut cfg = LanceConfig::default_3b(device.clone());
    cfg.num_inference_steps = args.steps;
    cfg.timestep_shift = args.shift;
    cfg.cfg_text_scale = args.cfg;
    let cfg = Arc::new(cfg);

    if args.width % 16 != 0 || args.height % 16 != 0 {
        return Err(anyhow!(
            "width/height must be divisible by 16 (Wan22 VAE); got {}x{}",
            args.width,
            args.height
        ));
    }
    let h_latent = args.height / 16;
    let w_latent = args.width / 16;
    let l_image = t_lat * h_latent * w_latent;
    let max_pos = cond_ids.len().max(uncond_ids.len()) + 1000 + l_image + 64;

    save_scalar("geom.t_lat", t_lat as f32, device, refs_dir)?;
    save_scalar("geom.h_lat", h_latent as f32, device, refs_dir)?;
    save_scalar("geom.w_lat", w_latent as f32, device, refs_dir)?;
    save_scalar("geom.l_image", l_image as f32, device, refs_dir)?;

    // ---- Capture: input.text_tokens, input.uncond_text_tokens ----
    let cond_tokens = ids_to_tensor(cond_ids, device)?;
    let uncond_tokens = ids_to_tensor(uncond_ids, device)?;
    save_capture("input.text_tokens", &cond_tokens, refs_dir)?;
    save_capture("input.uncond_text_tokens", &uncond_tokens, refs_dir)?;

    // ---- Capture: input.latent_noise ----
    let c = cfg.patch_latent_dim();
    let initial_noise = if let Some(path) = args.noise_from.as_ref() {
        // Same-seed preflight: load Python's noise tensor (L, C) so the
        // RNG-mismatch confound is eliminated. Save the capture in the
        // (L, C) layout so diff_t2v.py's flatten ordering matches the
        // Python-side capture.
        let (lc_f32, model_5d_f32) =
            load_python_noise(path, t_lat, h_latent, w_latent, c, device)?;
        save_capture("input.latent_noise", &lc_f32, refs_dir)?;
        model_5d_f32.to_dtype(cfg.dtype)?
    } else {
        let noise_n = c * t_lat * h_latent * w_latent;
        let noise = deterministic_normal_noise(args.seed, noise_n);
        let initial_noise_f32 = Tensor::from_vec(
            noise.clone(),
            Shape::from_dims(&[1, c, t_lat, h_latent, w_latent]),
            device.clone(),
        )?;
        save_capture("input.latent_noise", &initial_noise_f32, refs_dir)?;
        initial_noise_f32.to_dtype(cfg.dtype)?
    };

    // ---- Load Lance + mRoPE ----
    let t0 = Instant::now();
    let lance = Lance::load(&args.model_path, cfg.clone(), device)
        .with_context(|| format!("Lance::load({})", args.model_path.display()))?;
    log::info!(
        "[parity_lance_t2v]   Lance loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let mrope = lance.precompute_mrope(max_pos, cfg.dtype)?;

    // ---- Prefill cond + uncond ----
    let mut cond_cache = lance.new_kv_cache();
    lance.prefill_text_context(&cond_tokens, &mrope, &mut cond_cache)?;
    let cond_prefix_len = cond_cache.seq_len(0);
    save_scalar(
        "cache.cond_prefix_len",
        cond_prefix_len as f32,
        device,
        refs_dir,
    )?;

    let mut uncond_cache = lance.new_kv_cache();
    lance.prefill_text_context(&uncond_tokens, &mrope, &mut uncond_cache)?;
    let uncond_prefix_len = uncond_cache.seq_len(0);
    save_scalar(
        "cache.uncond_prefix_len",
        uncond_prefix_len as f32,
        device,
        refs_dir,
    )?;

    log::info!(
        "[parity_lance_t2v]   cond_prefix_len={}, uncond_prefix_len={}",
        cond_prefix_len,
        uncond_prefix_len
    );

    // ---- Denoise with step-0 capture ----
    let t1 = Instant::now();
    let latent = denoise_loop_capture(
        &lance,
        cfg.as_ref(),
        &cond_cache,
        &uncond_cache,
        &initial_noise,
        &mrope,
        args.steps,
        args.shift,
        args.cfg,
        refs_dir,
    )?;
    log::info!(
        "[parity_lance_t2v]   denoise complete in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    save_capture("lance.final_latent", &latent, refs_dir)?;

    let host = latent.to_dtype(DType::F32)?.to_vec_f32()?;
    let shape = latent.shape().dims().to_vec();
    drop(lance);
    drop(mrope);
    Ok((host, shape))
}

// ---------------------------------------------------------------------------
// Stage 3: VAE video decode + capture.
// ---------------------------------------------------------------------------

fn stage3_decode_capture(
    args: &Args,
    latent_host: Vec<f32>,
    latent_shape: Vec<usize>,
    device: &Arc<CudaDevice>,
    refs_dir: &Path,
) -> Result<()> {
    log::info!("[parity_lance_t2v] Stage 3: load Wan22 VAE, decode, capture");
    let vae = Wan22VaeDecoder::load(&args.vae_path, device)?;
    let latent_gpu =
        Tensor::from_vec(latent_host, Shape::from_dims(&latent_shape), device.clone())?
            .to_dtype(DType::BF16)?;
    // Video decode (NOT decode_image) for T2V.
    let video = vae.decode(&latent_gpu)?;
    save_capture("vae_decode.out", &video, refs_dir)?;
    drop(vae);
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let device = flame_core::global_cuda_device();
    let refs_dir = &args.refs_out;
    std::fs::create_dir_all(refs_dir)
        .with_context(|| format!("mkdir -p {}", refs_dir.display()))?;

    // Frame-count guardrail: Wan22 VAE requires (N-1) % 4 == 0.
    if (args.num_frames as i64 - 1) % 4 != 0 {
        return Err(anyhow!(
            "--num-frames must satisfy (N-1) % 4 == 0; got {}",
            args.num_frames
        ));
    }
    let t_lat = t_lat_from_pixel_frames(args.num_frames);

    // ---- Stage 1: tokenize. ----
    log::info!("[parity_lance_t2v] Stage 1: tokenize");
    let tok = load_tokenizer(&args.model_path)?;
    let cond_ids = tokenize_to_ids(&tok, &args.prompt, args.text_template)?;
    let uncond_ids = tokenize_to_ids(&tok, &args.negative_prompt, args.text_template)?;
    log::info!(
        "[parity_lance_t2v]   cond_len={}  uncond_len={}",
        cond_ids.len(),
        uncond_ids.len()
    );
    drop(tok);

    // ---- Stage 2. ----
    let (latent_host, latent_shape) =
        stage2_denoise_capture(args, &cond_ids, &uncond_ids, t_lat, &device, refs_dir)?;

    // ---- Stage 3 (optional). ----
    if !args.skip_vae {
        stage3_decode_capture(args, latent_host, latent_shape, &device, refs_dir)?;
    } else {
        log::info!("[parity_lance_t2v]   --skip-vae: omitting VAE decode capture");
    }

    log::info!(
        "[parity_lance_t2v] all captures written to {}",
        refs_dir.display()
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
        log::error!("[parity_lance_t2v] FAILED: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
