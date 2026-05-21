//! Shared helpers for the three Cosmos-Predict2.5-2B inference binaries
//! (`cosmos_predict25_t2v_infer`, `_i2v_infer`, `_v2v_infer`).
//!
//! Included into each binary via `#[path = "cosmos_predict25_common.rs"] mod common;`.
//! This is the chunk-5 entry point of the cosmos-predict25-2b port
//! (BUILD_PLAN.md steps 11a/b/c). See `inference-flame/ports/cosmos-predict25-2b/`
//! for full port docs.
//!
//! ## Stage discipline
//!
//! Each binary follows strict load-encode-drop sequencing to fit on 24 GB:
//!
//! 1. **Stage 1 — text encode** (`encode_text`): load Cosmos-Reason1-7B (~15 GB
//!    BF16), tokenize + chat-template, run forward with `EmbeddingConcatStrategy::FullConcat`,
//!    drop encoder. Output: `[1, 512, 100352]` (FULL_CONCAT) plus optional negative.
//! 2. **Stage 2 — conditioning encode** (`encode_conditioning_frames`, i2v / v2v
//!    only): load Wan21 VAE encoder (~1 GB), encode pixel frames to
//!    `[1, 16, T_lat, H_lat, W_lat]`, drop encoder.
//! 3. **Stage 3 — DiT denoise** (`denoise`): load DiT (~4 GB BF16), run UniPC
//!    multistep loop with CFG, mix conditioning frames into x_t each step, drop DiT.
//! 4. **Stage 4 — VAE decode** (`decode_pixels`): load Wan21 VAE decoder (~1 GB),
//!    decode latent to pixels, drop decoder.
//! 5. **Stage 5 — mp4** (`write_output`): write mp4 via ffmpeg.
//!
//! ## CFG convention
//!
//! Python (`video2world_model_rectified_flow.py:208`) uses
//! `velocity = cond + guidance * (cond - uncond)`, with default `guidance=7`.
//! This is equivalent to the canonical `out = uncond + cfg * (cond - uncond)`
//! with `cfg = 1 + guidance`. We expose `--cfg` matching Python's `guidance`
//! semantics (default 7.0) and apply `cosmos_cfg_combine` below verbatim.
//!
//! ## Conditioning mask
//!
//! Cosmos's i2v / v2v conditioning is implemented in *latent space*, not via
//! cross-attention. Steps:
//! - Encode the (full-T) pixel video with the VAE → `gt_lat [B, 16, T_lat, H, W]`.
//! - Build `mask [B, 1, T_lat, H, W]` with `1` at first
//!   `num_latent_conditional_frames` T positions, `0` elsewhere.
//! - During denoise, each step: `xt = gt_lat * mask + xt * (1 - mask)`
//!   (Python `:104-106`).
//! - The DiT's `forward(x_b_c_t_h_w, ...)` is then called on the mixed `xt`;
//!   the DiT itself does not see the mask.
//! - After velocity prediction, in Python `denoise_replace_gt_frames=True`
//!   replaces velocity on conditional positions with the analytical velocity
//!   `noise - gt_lat`. We do the same.
//!
//! `num_latent_conditional_frames` is `1` for i2v (image is the first latent
//! frame, since `read_and_process_image` pads with zeros so pixel-frame 0 is
//! the image; 8×8×4 VAE compresses to latent-frame 0 holding the image
//! signal). For v2v, Python supports `1` or `2` (`video2world.py:211-212`).

#![allow(dead_code)] // each binary uses a subset

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;
use flame_core::{global_cuda_device, DType, Error as FlameError, Shape, Tensor};
use tokenizers::Tokenizer;

use inference_flame::models::cosmos_predict25_dit::{
    CosmosPredict25Config, CosmosPredict25Dit,
};
use inference_flame::models::cosmos_reason1::{
    CosmosReason1Encoder, EmbeddingConcatStrategy,
};
use inference_flame::models::qwen25vl_encoder::Qwen25VLEncoder;
use inference_flame::sampling::cosmos_unipc::CosmosUniPcMultistepScheduler;
use inference_flame::vae::wan21_encoder::Wan21VaeEncoder;
use inference_flame::vae::wan21_vae::Wan21VaeDecoder;

// ---------------------------------------------------------------------------
// CLI Args
// ---------------------------------------------------------------------------

/// Resolution presets matching upstream `--resolution`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    Res480p,
    Res720p,
}

impl Resolution {
    /// Returns `(width, height)` in pixels.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            // Cosmos `--resolution 480p` → 832×480 (upstream README).
            Resolution::Res480p => (832, 480),
            // 720p target shape — DiT supports up to 1280×704 (PORT_SPEC.md).
            Resolution::Res720p => (1280, 704),
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "480p" | "480" => Some(Resolution::Res480p),
            "720p" | "720" => Some(Resolution::Res720p),
            _ => None,
        }
    }
}

/// Variant subdirectory under the Cosmos-Predict2.5-2B repo root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    PreTrained,
    PostTrained,
    Distilled,
}

impl Variant {
    pub fn subdir(&self) -> &'static str {
        match self {
            Variant::PreTrained => "pre-trained",
            Variant::PostTrained => "post-trained",
            Variant::Distilled => "distilled",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pre-trained" | "pretrained" | "pre" => Some(Variant::PreTrained),
            "post-trained" | "posttrained" | "post" => Some(Variant::PostTrained),
            "distilled" | "distill" => Some(Variant::Distilled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// Default — full FlowUniPCMultistepScheduler with bh2 corrector.
    /// Matches Python's actual inference path (`text2world_model_rectified_flow.py:142`).
    UniPc,
    /// Plain Euler integrator over the same sigma schedule. Useful as a
    /// debugging baseline / parity reference for the UniPC corrector.
    Euler,
}

impl Sampler {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "unipc" | "unipcm" => Some(Sampler::UniPc),
            "euler" => Some(Sampler::Euler),
            _ => None,
        }
    }
}

/// Default `.cosmos-predict25/` directory under the EriDiffusion repo root.
pub const DEFAULT_COSMOS_DIR: &str = "/home/alex/.cosmos-predict25";

/// CLI args common to t2v / i2v / v2v.
#[derive(Debug, Clone)]
pub struct CosmosPredict25Args {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// Pixel-frame count. Must satisfy `(N - 1) % 4 == 0` because the Wan VAE
    /// has 4× temporal compression: `T_latent = (N - 1)/4 + 1`. Default 81.
    pub num_frames: usize,
    pub num_steps: usize,
    /// Python's `guidance` scalar (default 7.0). See module docstring re. CFG convention.
    pub cfg: f32,
    pub seed: u64,
    pub resolution: Resolution,
    pub variant: Variant,
    pub sampler: Sampler,
    pub output_dir: PathBuf,
    /// Toggles BlockOffloader. Default true at 720p, false at 480p.
    pub use_block_offload: bool,
    /// Path to the converted DiT safetensors. Falls back to
    /// `DEFAULT_COSMOS_DIR/base/<variant>/cosmos_predict25_2b_dit.safetensors`.
    pub dit_path: Option<PathBuf>,
    /// Path to the Wan2.1 VAE safetensors (converted from `tokenizer.pth`).
    pub vae_path: Option<PathBuf>,
    /// Path to the Cosmos-Reason1-7B encoder directory (HF safetensors shards).
    pub reason1_dir: Option<PathBuf>,
    /// Path to the Qwen2.5-VL tokenizer JSON.
    pub tokenizer_path: Option<PathBuf>,
    /// Optional image conditioning input (i2v).
    pub input_image: Option<PathBuf>,
    /// Optional video conditioning input (v2v).
    pub input_video: Option<PathBuf>,
    /// For v2v: number of *latent* conditional frames (1 or 2 per Python `:211-212`).
    pub num_latent_conditional_frames: usize,
    /// Output frame rate (mp4 mux).
    pub fps: f32,
}

impl CosmosPredict25Args {
    /// (width, height) in pixels.
    pub fn dims_wh(&self) -> (usize, usize) {
        self.resolution.dims()
    }
    /// Latent frame count: `(N - 1)/4 + 1` (Wan VAE temporal compression).
    pub fn num_latent_frames(&self) -> usize {
        (self.num_frames - 1) / 4 + 1
    }
    pub fn dit_path_resolved(&self) -> PathBuf {
        self.dit_path.clone().unwrap_or_else(|| {
            PathBuf::from(DEFAULT_COSMOS_DIR)
                .join("base")
                .join(self.variant.subdir())
                .join("cosmos_predict25_2b_dit.safetensors")
        })
    }
    pub fn vae_path_resolved(&self) -> PathBuf {
        if let Some(p) = &self.vae_path { return p.clone(); }
        if let Ok(env) = std::env::var("WAN21_VAE_COSMOS_SAFETENSORS") {
            return PathBuf::from(env);
        }
        PathBuf::from(DEFAULT_COSMOS_DIR).join("wan21_vae.safetensors")
    }
    pub fn reason1_dir_resolved(&self) -> PathBuf {
        if let Some(p) = &self.reason1_dir { return p.clone(); }
        if let Ok(env) = std::env::var("COSMOS_REASON1_PATH") {
            return PathBuf::from(env);
        }
        PathBuf::from(DEFAULT_COSMOS_DIR).join("Cosmos-Reason1-7B")
    }
    pub fn tokenizer_path_resolved(&self) -> PathBuf {
        if let Some(p) = &self.tokenizer_path { return p.clone(); }
        if let Ok(env) = std::env::var("COSMOS_REASON1_TOKENIZER") {
            return PathBuf::from(env);
        }
        self.reason1_dir_resolved().join("tokenizer.json")
    }

    /// Validate that num_frames satisfies the Wan VAE 4×temporal constraint
    /// and resolution divisibility constraints (Wan VAE stride=8 → spatial dims
    /// multiple of 8; DiT patch_s=2 → latent dim must be even).
    pub fn validate(&self) -> Result<(), String> {
        if self.num_frames == 0 {
            return Err("--num-frames must be > 0".to_string());
        }
        if (self.num_frames - 1) % 4 != 0 {
            return Err(format!(
                "--num-frames must satisfy (N-1) % 4 == 0 (Wan VAE 4x temporal), got {}",
                self.num_frames
            ));
        }
        let (w, h) = self.dims_wh();
        if w % 16 != 0 || h % 16 != 0 {
            return Err(format!("Resolution ({w}x{h}) must be multiples of 16"));
        }
        // After 8× VAE spatial downsample and 2× DiT patch: latent dim/8 must be even.
        if (h / 8) % 2 != 0 || (w / 8) % 2 != 0 {
            return Err(format!(
                "Latent spatial dims ({}, {}) must be even (DiT patch_size=2)",
                h / 8, w / 8
            ));
        }
        if let Some(s) = self.input_image.as_ref().or(self.input_video.as_ref()) {
            if !s.exists() {
                return Err(format!("Input path does not exist: {}", s.display()));
            }
        }
        if self.num_latent_conditional_frames > 2 {
            return Err(format!(
                "--num-latent-conditional-frames must be 0, 1, or 2; got {}",
                self.num_latent_conditional_frames
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CLI parsing (shared across t2v/i2v/v2v with mode-specific tweaks)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    T2V,
    I2V,
    V2V,
}

/// Print the help banner for a given mode, then exit.
pub fn print_usage_and_exit(mode: Mode, prog: &str) -> ! {
    let mode_str = match mode {
        Mode::T2V => "cosmos_predict25_t2v_infer",
        Mode::I2V => "cosmos_predict25_i2v_infer",
        Mode::V2V => "cosmos_predict25_v2v_infer",
    };
    eprintln!("\nusage: {prog} (also known as {mode_str})");
    eprintln!("  --prompt STRING              (required)");
    eprintln!("  --negative-prompt STRING     (optional)");
    eprintln!("  --num-frames N               default 81 (must satisfy (N-1)%4==0)");
    eprintln!("  --num-steps N                default 35");
    eprintln!("  --cfg FLOAT                  default 7.0 (Python `guidance`)");
    eprintln!("  --seed U64                   default 42");
    eprintln!("  --resolution {{480p|720p}}     default 480p");
    eprintln!("  --variant {{pre-trained|post-trained|distilled}}  default post-trained");
    eprintln!("  --sampler {{unipc|euler}}      default unipc");
    eprintln!("  --output-dir PATH            default ./output");
    eprintln!("  --dit-path PATH              (overrides default + COSMOS_DIT_PATH env)");
    eprintln!("  --vae-path PATH              (overrides default + WAN21_VAE_COSMOS_SAFETENSORS env)");
    eprintln!("  --reason1-path PATH          (overrides default + COSMOS_REASON1_PATH env)");
    eprintln!("  --tokenizer-path PATH        (overrides default + COSMOS_REASON1_TOKENIZER env)");
    eprintln!("  --fps FLOAT                  default 16.0");
    if matches!(mode, Mode::I2V) {
        eprintln!("  --input-image PATH           (required)");
    }
    if matches!(mode, Mode::V2V) {
        eprintln!("  --input-video PATH               (required)");
        eprintln!("  --num-latent-conditional-frames N   default 2 (1 or 2)");
    }
    std::process::exit(2);
}

/// Parse argv into `CosmosPredict25Args` for the given mode.
///
/// Caller passes the program name (`std::env::args().next()`) for usage messages.
pub fn parse_args(mode: Mode) -> CosmosPredict25Args {
    let argv: Vec<String> = std::env::args().collect();
    let prog = argv.first().cloned().unwrap_or_else(|| "<prog>".into());

    let mut prompt: Option<String> = None;
    let mut negative_prompt: Option<String> = None;
    let mut num_frames: usize = 81;
    let mut num_steps: usize = 35;
    let mut cfg: f32 = 7.0;
    let mut seed: u64 = 42;
    let mut resolution: Resolution = Resolution::Res480p;
    let mut variant: Variant = Variant::PostTrained;
    let mut sampler: Sampler = Sampler::UniPc;
    let mut output_dir: PathBuf = PathBuf::from("./output");
    let mut dit_path: Option<PathBuf> = None;
    let mut vae_path: Option<PathBuf> = None;
    let mut reason1_dir: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut input_image: Option<PathBuf> = None;
    let mut input_video: Option<PathBuf> = None;
    // v2v default is 2 (Python `:163`). i2v hardwires 1.
    let mut num_latent_conditional_frames: usize = match mode {
        Mode::T2V => 0,
        Mode::I2V => 1,
        Mode::V2V => 2,
    };
    let mut fps: f32 = 16.0;

    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let need_value = |i: usize| -> String {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {}", argv[i]);
                print_usage_and_exit(mode, &prog);
            })
        };
        match arg {
            "--prompt" => { prompt = Some(need_value(i)); i += 2; }
            "--negative-prompt" => { negative_prompt = Some(need_value(i)); i += 2; }
            "--num-frames" => { num_frames = need_value(i).parse().expect("num-frames usize"); i += 2; }
            "--num-steps" => { num_steps = need_value(i).parse().expect("num-steps usize"); i += 2; }
            "--cfg" => { cfg = need_value(i).parse().expect("cfg f32"); i += 2; }
            "--seed" => { seed = need_value(i).parse().expect("seed u64"); i += 2; }
            "--resolution" => {
                let v = need_value(i);
                resolution = Resolution::parse(&v).unwrap_or_else(|| {
                    eprintln!("unknown resolution: {v} (want 480p or 720p)");
                    print_usage_and_exit(mode, &prog);
                });
                i += 2;
            }
            "--variant" => {
                let v = need_value(i);
                variant = Variant::parse(&v).unwrap_or_else(|| {
                    eprintln!("unknown variant: {v}");
                    print_usage_and_exit(mode, &prog);
                });
                i += 2;
            }
            "--sampler" => {
                let v = need_value(i);
                sampler = Sampler::parse(&v).unwrap_or_else(|| {
                    eprintln!("unknown sampler: {v} (want unipc or euler)");
                    print_usage_and_exit(mode, &prog);
                });
                i += 2;
            }
            "--output-dir" => { output_dir = PathBuf::from(need_value(i)); i += 2; }
            "--dit-path" => { dit_path = Some(PathBuf::from(need_value(i))); i += 2; }
            "--vae-path" => { vae_path = Some(PathBuf::from(need_value(i))); i += 2; }
            "--reason1-path" => { reason1_dir = Some(PathBuf::from(need_value(i))); i += 2; }
            "--tokenizer-path" => { tokenizer_path = Some(PathBuf::from(need_value(i))); i += 2; }
            "--input-image" => { input_image = Some(PathBuf::from(need_value(i))); i += 2; }
            "--input-video" => { input_video = Some(PathBuf::from(need_value(i))); i += 2; }
            "--num-latent-conditional-frames" => {
                num_latent_conditional_frames = need_value(i).parse().expect("usize");
                i += 2;
            }
            "--fps" => { fps = need_value(i).parse().expect("fps f32"); i += 2; }
            "-h" | "--help" => print_usage_and_exit(mode, &prog),
            other => {
                eprintln!("unknown arg: {other}");
                print_usage_and_exit(mode, &prog);
            }
        }
    }

    // Mode-specific required flags.
    if matches!(mode, Mode::I2V) && input_image.is_none() {
        eprintln!("--input-image required for i2v");
        print_usage_and_exit(mode, &prog);
    }
    if matches!(mode, Mode::V2V) && input_video.is_none() {
        eprintln!("--input-video required for v2v");
        print_usage_and_exit(mode, &prog);
    }
    let prompt = prompt.unwrap_or_else(|| {
        eprintln!("--prompt required");
        print_usage_and_exit(mode, &prog);
    });

    // Auto-enable BlockOffloader at 720p.
    let use_block_offload = matches!(resolution, Resolution::Res720p);

    CosmosPredict25Args {
        prompt,
        negative_prompt,
        num_frames,
        num_steps,
        cfg,
        seed,
        resolution,
        variant,
        sampler,
        output_dir,
        use_block_offload,
        dit_path,
        vae_path,
        reason1_dir,
        tokenizer_path,
        input_image,
        input_video,
        num_latent_conditional_frames,
        fps,
    }
}

// ---------------------------------------------------------------------------
// Stage 1 — text encode
// ---------------------------------------------------------------------------

/// Load Cosmos-Reason1-7B encoder, encode (prompt, optional negative), drop encoder.
///
/// Returns `(prompt_emb [1, 512, 100352], negative_emb [1, 512, 100352] or None)` in BF16.
///
/// The encoder applies Cosmos's chat template + per-layer mean-normalize +
/// FULL_CONCAT aggregation. Output goes through the DiT's `crossattn_proj`
/// (100352 → 1024) inside `forward_inner`, so we don't project here.
///
/// Shard loader filters out any `visual.*` keys (Cosmos-Reason1-7B is a fine-tune
/// of Qwen2.5-VL; the visual encoder shards are present but unused for text-only).
pub fn encode_text(
    args: &CosmosPredict25Args,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<(Tensor, Option<Tensor>)> {
    println!("--- Stage 1: Cosmos-Reason1-7B text encode ---");
    let t0 = Instant::now();

    let reason1_dir = args.reason1_dir_resolved();
    let tokenizer_path = args.tokenizer_path_resolved();
    println!("  encoder dir : {}", reason1_dir.display());
    println!("  tokenizer   : {}", tokenizer_path.display());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

    // Load all sharded safetensors under reason1_dir.
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut shard_count = 0usize;
    // Try common shard layouts: top-level files matching *.safetensors and
    // a "text_encoder" subdir.
    let candidates: Vec<PathBuf> = collect_safetensor_shards(&reason1_dir);
    if candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "No safetensors shards found under {}",
            reason1_dir.display()
        ));
    }
    for shard in &candidates {
        let w = flame_core::serialization::load_file_filtered(
            shard, device, |k| !k.starts_with("visual."),
        )?;
        for (k, v) in w {
            weights.insert(k, v);
        }
        shard_count += 1;
    }
    println!(
        "  loaded {} tensors from {} shards in {:.1}s",
        weights.len(), shard_count, t0.elapsed().as_secs_f32()
    );

    // Coerce non-BF16 to BF16 in-place.
    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in keys {
        let t = &weights[&k];
        if t.dtype() != DType::BF16 {
            let bf = t.to_dtype(DType::BF16)?;
            weights.insert(k, bf);
        }
    }

    let cfg = Qwen25VLEncoder::config_from_weights(&weights)?;
    println!(
        "  qwen cfg: layers={} hidden={} heads={}/{} head_dim={}",
        cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    );

    let inner = Qwen25VLEncoder::new(weights, cfg, device.clone());
    let encoder = CosmosReason1Encoder::new(inner, EmbeddingConcatStrategy::FullConcat);
    println!("  cosmos-reason1 strategy=FullConcat output_dim={}", encoder.output_dim());

    let t_enc = Instant::now();
    let prompt_emb = encoder.encode_prompt(&args.prompt, &tokenizer)?;
    {
        let ef = prompt_emb.to_dtype(DType::F32).unwrap();
        let emax = ef.abs().unwrap().max_all().unwrap();
        let emean = ef.mean_dim(&[0,1,2], false).unwrap().to_vec1::<f32>().unwrap()[0];
        let emse = ef.mul(&ef).unwrap().mean_dim(&[0,1,2], false).unwrap().to_vec1::<f32>().unwrap()[0];
        println!(
            "  prompt emb {:?} |e|_inf={:.3} mean={:.4} rms={:.4} ({:.1}s)",
            prompt_emb.shape().dims(), emax, emean, emse.sqrt(), t_enc.elapsed().as_secs_f32()
        );
    }

    // Negative: always produce one (empty string default, mirroring Python's
    // `_DEFAULT_NEGATIVE_PROMPT` semantics — explicit user override allowed).
    // CFG with the velocity_fn does require uncond, but if user passes
    // `--cfg 1.0` (Python guidance=0 = no cfg) callers can skip the second
    // forward; we still produce uncond so the run looks uniform.
    let neg_text = args.negative_prompt.clone().unwrap_or_default();
    let neg_emb = if !neg_text.is_empty() || args.cfg.abs() > 1e-6 {
        let t_neg = Instant::now();
        let neg = encoder.encode_prompt(&neg_text, &tokenizer)?;
        println!(
            "  negative emb {:?} in {:.1}s",
            neg.shape().dims(),
            t_neg.elapsed().as_secs_f32()
        );
        Some(neg)
    } else {
        None
    };

    drop(encoder);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  text encoder dropped, stage 1 total: {:.1}s", t0.elapsed().as_secs_f32());

    Ok((prompt_emb, neg_emb))
}

fn collect_safetensor_shards(dir: &Path) -> Vec<PathBuf> {
    // Try (a) text_encoder subdir (mirrors the qwen25vl_parity convention),
    // (b) the directory itself.
    let mut out = Vec::new();
    let probe = dir.join("text_encoder");
    let scan = if probe.is_dir() { probe } else { dir.to_path_buf() };
    if let Ok(rd) = std::fs::read_dir(&scan) {
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Stage 2 — conditioning encode (i2v / v2v)
// ---------------------------------------------------------------------------

/// Load Wan21 VAE encoder, encode the conditioning input (image or video),
/// drop encoder. Returns `(gt_lat [1, 16, T_lat, H_lat, W_lat], num_latent_conditional_frames)`.
///
/// For i2v: image is loaded, placed at pixel-frame 0, frames 1..num_frames padded
/// with zeros. VAE encodes the full pixel video; we keep all T_latent frames as
/// `gt_lat` but only the first `1` is marked as conditioning via the mask.
/// (Python `read_and_process_image:148`.)
///
/// For v2v: last `4*(num_cond_lat-1) + 1` pixel frames are extracted from the
/// input video, padded with the last frame to `num_frames`, encoded via VAE
/// to produce `gt_lat`. The first `num_latent_conditional_frames` latent frames
/// are conditioning. (Python `read_and_process_video:204-238`.)
pub fn encode_conditioning_frames(
    input_path: &Path,
    is_video: bool,
    args: &CosmosPredict25Args,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<(Tensor, usize)> {
    println!("--- Stage 2: Wan21 VAE encode conditioning ---");
    let t0 = Instant::now();
    let (w, h) = args.dims_wh();
    let f = args.num_frames;
    let video = if is_video {
        load_video_to_tensor(input_path, w, h, f, args.num_latent_conditional_frames, device)?
    } else {
        load_image_to_video_tensor(input_path, w, h, f, device)?
    };
    println!("  pixel video shape: {:?}", video.shape().dims());

    let vae_path = args.vae_path_resolved();
    println!("  vae           : {}", vae_path.display());
    let encoder = Wan21VaeEncoder::load(
        vae_path.to_str().ok_or_else(|| anyhow::anyhow!("non-utf8 vae path"))?,
        device,
    )?;
    println!("  vae encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let gt_lat = encoder.encode(&video)?;
    println!("  gt_lat shape: {:?}", gt_lat.shape().dims());

    drop(encoder);
    drop(video);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  vae encoder dropped, stage 2 total: {:.1}s", t0.elapsed().as_secs_f32());

    Ok((gt_lat, args.num_latent_conditional_frames))
}

// ---------------------------------------------------------------------------
// Pixel loaders
// ---------------------------------------------------------------------------

/// Decode an image, build `[1, 3, num_frames, H, W]` in BF16 with the image
/// at frame 0 and zeros elsewhere. Mirrors Python `read_and_process_image:138-156`.
///
/// Pixel range: `[-1, 1]` (Wan VAE expects `2*x - 1`; image `[0,1]` → `[-1, 1]`).
fn load_image_to_video_tensor(
    image_path: &Path,
    target_w: usize,
    target_h: usize,
    num_frames: usize,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    let img = image::open(image_path)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", image_path.display()))?
        .to_rgb8();
    println!("  image loaded: {}x{}", img.width(), img.height());
    let resized = image::imageops::resize(
        &img, target_w as u32, target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
    // [1, 3, F, H, W], frame 0 = image, rest = 0
    let mut data = vec![0.0f32; 3 * num_frames * target_h * target_w];
    let fhw = num_frames * target_h * target_w;
    let hw = target_h * target_w;
    for c in 0..3 {
        for y in 0..target_h {
            for x in 0..target_w {
                let p = resized.get_pixel(x as u32, y as u32);
                let v = (p[c] as f32) / 127.5 - 1.0;
                data[c * fhw + 0 * hw + y * target_w + x] = v;
            }
        }
    }
    let t = Tensor::from_vec(
        data, Shape::from_dims(&[1, 3, num_frames, target_h, target_w]), device.clone(),
    )?.to_dtype(DType::BF16)?;
    Ok(t)
}

/// Decode a video via ffmpeg subprocess into raw RGB frames, then build the
/// `[1, 3, num_frames, H, W]` tensor.
///
/// We invoke `ffmpeg -i <video> -vf scale=W:H -vframes <wanted> -f rawvideo -pix_fmt rgb24 -`
/// to extract frames. Then pad/truncate per Python `read_and_process_video:204-238`.
///
/// Wanted pixel frames = `4*(num_latent_cond - 1) + 1` (last frames from input
/// video), then padded to `num_frames` with the last frame repeated.
fn load_video_to_tensor(
    video_path: &Path,
    target_w: usize,
    target_h: usize,
    num_frames: usize,
    num_latent_cond: usize,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    if num_latent_cond == 0 {
        return Err(anyhow::anyhow!(
            "load_video_to_tensor called with num_latent_cond=0 (t2v should not load video)"
        ));
    }
    let frames_to_extract = 4 * (num_latent_cond.saturating_sub(1)) + 1;

    // We use ffmpeg to extract the *last* `frames_to_extract` frames at the
    // target resolution. ffmpeg's `select='gte(n,N-K)'` is awkward without
    // knowing total frame count first; we probe with a separate ffprobe call.
    let nb_frames = probe_video_frame_count(video_path)?;
    if nb_frames < frames_to_extract {
        return Err(anyhow::anyhow!(
            "video has {nb_frames} frames but needs at least {frames_to_extract} for \
             num_latent_conditional_frames={num_latent_cond}"
        ));
    }
    let start_idx = nb_frames.saturating_sub(frames_to_extract);

    let raw = run_ffmpeg_extract_rgb(
        video_path, target_w, target_h, start_idx, frames_to_extract,
    )?;
    let expected = 3 * target_h * target_w * frames_to_extract;
    if raw.len() != expected {
        return Err(anyhow::anyhow!(
            "ffmpeg produced {} bytes, expected {expected} (3 x W x H x F)",
            raw.len()
        ));
    }

    // Build [1, 3, num_frames, H, W] in F32 normalized [-1, 1].
    let frame_pix = 3 * target_h * target_w;
    let fhw = num_frames * target_h * target_w;
    let hw = target_h * target_w;
    let mut data = vec![0.0f32; 3 * num_frames * target_h * target_w];
    // The frames come out from ffmpeg as F sequential frames of HxWx3 packed RGB.
    // We need to scatter into channel-major [3, F, H, W] layout.
    for f_in in 0..frames_to_extract {
        let f_out = f_in; // first N frames hold extracted content
        let base = f_in * frame_pix;
        for y in 0..target_h {
            for x in 0..target_w {
                let p = base + (y * target_w + x) * 3;
                let r = (raw[p + 0] as f32) / 127.5 - 1.0;
                let g = (raw[p + 1] as f32) / 127.5 - 1.0;
                let b = (raw[p + 2] as f32) / 127.5 - 1.0;
                data[0 * fhw + f_out * hw + y * target_w + x] = r;
                data[1 * fhw + f_out * hw + y * target_w + x] = g;
                data[2 * fhw + f_out * hw + y * target_w + x] = b;
            }
        }
    }
    // Pad remaining frames with the last extracted frame (Python `:231-235`).
    if frames_to_extract < num_frames {
        let last = frames_to_extract - 1;
        for f_out in frames_to_extract..num_frames {
            for c in 0..3 {
                for y in 0..target_h {
                    for x in 0..target_w {
                        let src_idx = c * fhw + last * hw + y * target_w + x;
                        let dst_idx = c * fhw + f_out * hw + y * target_w + x;
                        data[dst_idx] = data[src_idx];
                    }
                }
            }
        }
    }
    let t = Tensor::from_vec(
        data, Shape::from_dims(&[1, 3, num_frames, target_h, target_w]), device.clone(),
    )?.to_dtype(DType::BF16)?;
    Ok(t)
}

fn probe_video_frame_count(video_path: &Path) -> anyhow::Result<usize> {
    use std::process::Command;
    let out = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
        ])
        .arg(video_path)
        .output()
        .map_err(|e| anyhow::anyhow!("ffprobe failed (is it installed?): {e}"))?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.parse::<usize>()
        .map_err(|e| anyhow::anyhow!("ffprobe returned non-numeric `{s}`: {e}"))
}

fn run_ffmpeg_extract_rgb(
    video_path: &Path, w: usize, h: usize, start: usize, count: usize,
) -> anyhow::Result<Vec<u8>> {
    use std::process::{Command, Stdio};
    // `-vf` chain: trim to [start, start+count) then scale.
    let select = format!("select='between(n\\,{start}\\,{end})',scale={w}:{h}",
                         end = start + count - 1);
    let child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "warning",
               "-i"])
        .arg(video_path)
        .args(["-vf", &select,
               "-vsync", "0",
               "-f", "rawvideo",
               "-pix_fmt", "rgb24",
               "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("ffmpeg spawn: {e}"))?;
    let out = child.wait_with_output()
        .map_err(|e| anyhow::anyhow!("ffmpeg wait: {e}"))?;
    if !out.status.success() {
        return Err(anyhow::anyhow!(
            "ffmpeg exited {}: {}", out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

// ---------------------------------------------------------------------------
// Stage 3 — DiT denoise
// ---------------------------------------------------------------------------

/// Build the conditioning mask `[1, 1, T_lat, H_lat, W_lat]` with `1` at first
/// `num_cond` T-positions and `0` elsewhere. Returns BF16 on the latent device.
fn build_condition_mask(
    b: usize, t: usize, h: usize, w: usize, num_cond: usize, device: &Arc<CudaDevice>,
) -> Result<Tensor, FlameError> {
    if num_cond > t {
        return Err(FlameError::InvalidInput(format!(
            "build_condition_mask: num_cond {num_cond} > T_latent {t}"
        )));
    }
    let mut data = vec![0.0f32; b * 1 * t * h * w];
    let hw = h * w;
    let thw = t * h * w;
    for bi in 0..b {
        for ti in 0..num_cond {
            for hi in 0..h {
                for wi in 0..w {
                    data[bi * thw + ti * hw + hi * w + wi] = 1.0;
                }
            }
        }
    }
    Tensor::from_vec(
        data, Shape::from_dims(&[b, 1, t, h, w]), device.clone(),
    )?.to_dtype(DType::BF16)
}

/// Build a `timesteps_B_T` tensor (shape `[1, T_lat]`) by broadcasting a scalar
/// timestep value. Used per UniPC step.
fn build_timesteps_b_t(ts: f32, t_lat: usize, device: &Arc<CudaDevice>) -> Result<Tensor, FlameError> {
    let data = vec![ts; t_lat];
    Tensor::from_vec(data, Shape::from_dims(&[1, t_lat]), device.clone())?.to_dtype(DType::BF16)
}

/// Cosmos CFG combine: `cond + guidance * (cond - uncond)` (Python `:208`).
/// `guidance=0` is no-CFG (`uncond_v == cond_v` cancels). Default 7.0.
fn cosmos_cfg_combine(
    cond_v: &Tensor, uncond_v: &Tensor, guidance: f32,
) -> Result<Tensor, FlameError> {
    let diff = cond_v.sub(uncond_v)?;
    let scaled = diff.mul_scalar(guidance)?;
    cond_v.add(&scaled)
}

/// Apply `xt = gt_lat * mask_C + xt * (1 - mask_C)` for conditional video2world.
/// `mask_b_1_thw` is broadcast across C=16. Stays in BF16 throughout.
fn apply_video_mask(
    xt: &Tensor, gt_lat: &Tensor, mask_b_1_thw: &Tensor,
) -> Result<Tensor, FlameError> {
    let xd = xt.shape().dims().to_vec();
    if xd.len() != 5 {
        return Err(FlameError::InvalidInput(format!("apply_video_mask: xt rank != 5: {:?}", xd)));
    }
    let (b, c, t, h, w) = (xd[0], xd[1], xd[2], xd[3], xd[4]);
    // Broadcast mask [B,1,T,H,W] → [B,C,T,H,W].
    let mask_c = mask_b_1_thw.broadcast_to(&Shape::from_dims(&[b, c, t, h, w]))?;
    // (1 - mask)
    let one = Tensor::from_vec(
        vec![1.0f32],
        Shape::from_dims(&[1, 1, 1, 1, 1]),
        xt.device().clone(),
    )?.to_dtype(xt.dtype())?
     .broadcast_to(&Shape::from_dims(&[b, c, t, h, w]))?;
    let one_minus_mask = one.sub(&mask_c)?;
    let cond_part = gt_lat.mul(&mask_c)?;
    let gen_part = xt.mul(&one_minus_mask)?;
    cond_part.add(&gen_part)
}

/// Stage 3: load DiT, run UniPC (default) or Euler denoise, drop DiT.
///
/// Returns the final latent `[1, 16, T_lat, H_lat, W_lat]`.
///
/// `text_emb`: `[1, 512, 100352]` from Cosmos-Reason1-7B (FULL_CONCAT, BF16).
/// `negative_emb`: same shape; required when `args.cfg > 0`.
/// `conditioning_latents`: `[1, 16, T_lat, H_lat, W_lat]` (i2v / v2v only).
/// `num_latent_conditional_frames`: 0 for t2v.
pub fn denoise(
    text_emb: &Tensor,
    negative_emb: Option<&Tensor>,
    conditioning_latents: Option<&Tensor>,
    num_latent_conditional_frames: usize,
    args: &CosmosPredict25Args,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    println!("--- Stage 3: DiT denoise ({} steps, sampler={:?}, cfg={}) ---",
             args.num_steps, args.sampler, args.cfg);
    let t0 = Instant::now();

    let dit_path = args.dit_path_resolved();
    println!("  dit path: {}", dit_path.display());

    // Use the production preset — required for crossattn_proj (100352 → 1024)
    // to be present.
    let cfg = CosmosPredict25Config::cosmos_v2_2b_production();
    let dit = CosmosPredict25Dit::from_safetensors(&dit_path, cfg, device.clone())
        .map_err(|e| anyhow::anyhow!("DiT load: {e}"))?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());

    if args.use_block_offload {
        log::warn!(
            "  --use-block-offload requested at 720p; BlockOffloader integration \
             not yet wired into cosmos_predict25_dit. Proceeding with all-on-GPU; \
             expect potential OOM on 24 GB at 720p."
        );
    }

    // Latent geometry.
    let (w_pix, h_pix) = args.dims_wh();
    let t_lat = args.num_latent_frames();
    let h_lat = h_pix / 8;
    let w_lat = w_pix / 8;
    let in_c = 16;

    println!("  latent geometry: [1, {in_c}, {t_lat}, {h_lat}, {w_lat}]");

    // Initial noise (Gaussian, BF16) — Box-Muller.
    let numel = in_c * t_lat * h_lat * w_lat;
    let noise = gaussian_noise_bf16(
        numel, args.seed,
        &[1, in_c, t_lat, h_lat, w_lat],
        device,
    )?;

    // Condition mask (i2v / v2v only).
    let cond_mask = if num_latent_conditional_frames > 0 {
        Some(build_condition_mask(1, t_lat, h_lat, w_lat, num_latent_conditional_frames, device)?)
    } else {
        None
    };

    // FPS for RoPE modulation (Cosmos default 16 fps base for video).
    let fps_val: Option<f32> = Some(args.fps);

    // Scheduler.
    enum Sched {
        UniPc(CosmosUniPcMultistepScheduler),
        // For Euler we just keep the sigma list.
        Euler { sigmas: Vec<f32>, timesteps: Vec<f32>, idx: usize },
    }
    let mut sched: Sched = match args.sampler {
        Sampler::UniPc => Sched::UniPc(CosmosUniPcMultistepScheduler::new(
            1000, args.num_steps, 5.0, 2,
        )),
        Sampler::Euler => {
            let s = inference_flame::sampling::cosmos_rf::RectifiedFlowSampler::new(
                args.num_steps, args.cfg, 5.0,
            );
            let sigmas = s.sigmas();
            let n_train = 1000.0_f32;
            let timesteps: Vec<f32> = sigmas[..args.num_steps].iter().map(|s| s * n_train).collect();
            Sched::Euler { sigmas, timesteps, idx: 0 }
        }
    };

    // Clone the schedule out of `sched` so we can borrow `sched` mutably in the
    // loop below without the immutable reference here holding it hostage.
    let (sigmas_vec, timesteps_vec): (Vec<f32>, Vec<f32>) = match &sched {
        Sched::UniPc(u) => (u.sigmas().to_vec(), u.timesteps().to_vec()),
        Sched::Euler { sigmas, timesteps, .. } => (sigmas.clone(), timesteps.clone()),
    };
    println!(
        "  sigma[0..3]={:.4} {:.4} {:.4} ... last={:.4}",
        sigmas_vec[0],
        sigmas_vec[1.min(sigmas_vec.len() - 1)],
        sigmas_vec[2.min(sigmas_vec.len() - 1)],
        sigmas_vec[sigmas_vec.len() - 1],
    );

    let mut x = noise.clone();
    // Probe initial noise per-channel stats.
    {
        let xf = noise.to_dtype(DType::F32).unwrap();
        let per_ch_sq = xf.mul(&xf).unwrap().mean_dim(&[2,3,4], false).unwrap();
        let psqv = per_ch_sq.to_vec1::<f32>().unwrap();
        print!("  INITIAL noise per-ch rms: [");
        for v in psqv.iter().take(16) { print!("{:.3} ", v.sqrt()); }
        println!("]");
    }

    for step in 0..args.num_steps {
        // For video2world: mix in conditioning latent before forward.
        if let (Some(mask), Some(gt)) = (cond_mask.as_ref(), conditioning_latents) {
            x = apply_video_mask(&x, gt, mask)?;
        }

        let ts = timesteps_vec[step];
        let timesteps_b_t = build_timesteps_b_t(ts, t_lat, device)?;

        // Cond forward. The production preset wraps the DiT as
        // `MinimalV1LVGDiT`, which expects `condition_video_input_mask`
        // (shape [B, 1, T_lat, H_lat, W_lat]) — pass our `cond_mask` if
        // present, else None (the model synthesizes zeros for image-mode).
        let cond_pred = dit.forward(
            &x, &timesteps_b_t, text_emb,
            cond_mask.as_ref(),
            None,
            fps_val,
            None,
        ).map_err(|e| anyhow::anyhow!("DiT cond forward: {e}"))?;

        // Uncond forward (only when CFG enabled and we have negative_emb).
        let velocity = if args.cfg.abs() > 1e-6 {
            let neg = negative_emb.ok_or_else(|| anyhow::anyhow!(
                "cfg > 0 but negative_emb is None — must run text encode with non-empty negative"
            ))?;
            let uncond_pred = dit.forward(
                &x, &timesteps_b_t, neg,
                cond_mask.as_ref(),
                None,
                fps_val,
                None,
            ).map_err(|e| anyhow::anyhow!("DiT uncond forward: {e}"))?;
            if step == 0 || (step + 1) % 5 == 0 || step + 1 == args.num_steps {
                let cond_max = cond_pred.to_dtype(DType::F32)?.abs()?.max_all()?;
                let uncond_max = uncond_pred.to_dtype(DType::F32)?.abs()?.max_all()?;
                let sigma_curr = sigmas_vec[step];
                let v_scaled = cond_pred.mul_scalar(sigma_curr)?;
                let x0 = x.sub(&v_scaled)?;
                let x0_rms = x0.to_dtype(DType::F32)?.mul(&x0.to_dtype(DType::F32)?)?
                    .mean_dim(&[0,1,2,3,4], false)?.to_vec1::<f32>()?[0].sqrt();
                println!("    [s{step:>2}] |cond_v|={:.3} |uncond_v|={:.3} x0_pred_rms={:.3}", cond_max, uncond_max, x0_rms);
            }
            cosmos_cfg_combine(&cond_pred, &uncond_pred, args.cfg)?
        } else {
            cond_pred
        };

        // For v2v / i2v: replace velocity on conditional positions with the
        // analytical velocity `noise - gt_lat` (Python `:130-135`,
        // `denoise_replace_gt_frames=True`).
        let velocity = if let (Some(mask), Some(gt)) = (cond_mask.as_ref(), conditioning_latents) {
            let gt_velocity = noise.sub(gt)?;
            apply_video_mask(&velocity, &gt_velocity, mask)?
        } else {
            velocity
        };

        // Apply scheduler step.
        x = match &mut sched {
            Sched::UniPc(u) => u.step(&velocity, &x)
                .map_err(|e| anyhow::anyhow!("UniPC step: {e}"))?,
            Sched::Euler { sigmas, idx, .. } => {
                let dt = sigmas[*idx + 1] - sigmas[*idx];
                let delta = velocity.mul_scalar(dt)?;
                *idx += 1;
                x.add(&delta)?
            }
        };
        // Post-step magnitude (after sched.step has been applied).
        if step + 1 == args.num_steps {
            let xf = x.to_dtype(flame_core::DType::F32).unwrap();
            let xmax = xf.abs().unwrap().max_all().unwrap();
            let xmean = xf.mean_dim(&[0,1,2,3,4], false).ok()
                .and_then(|m| m.to_vec1::<f32>().ok())
                .map(|v| v.get(0).copied().unwrap_or(f32::NAN))
                .unwrap_or(f32::NAN);
            let xsq = xf.mul(&xf).unwrap().mean_dim(&[0,1,2,3,4], false).unwrap();
            let xrms = xsq.to_vec1::<f32>().unwrap()[0].sqrt();
            println!("  POST-FINAL  |x|_inf={:.3e}  mean={:.3e}  rms={:.3e}", xmax, xmean, xrms);
            // Per-channel stats: mean over (T,H,W) for each of 16 channels.
            // Compare to Wan21 VAE normalize constants — a *clean* latent in
            // normalized space should have per-ch mean ~0 and rms ~1.
            let per_ch_mean = xf.mean_dim(&[2,3,4], false).unwrap();
            let pmv = per_ch_mean.to_vec1::<f32>().unwrap();
            let per_ch_sq = xf.mul(&xf).unwrap().mean_dim(&[2,3,4], false).unwrap();
            let psqv = per_ch_sq.to_vec1::<f32>().unwrap();
            println!("  per-ch mean: {:?}", &pmv[..16.min(pmv.len())]);
            print!("  per-ch rms : [");
            for v in psqv.iter().take(16) { print!("{:.3} ", v.sqrt()); }
            println!("]");
        }

        if step == 0 || (step + 1) % 5 == 0 || step + 1 == args.num_steps {
            // Magnitude probe (debug-only; remove later).
            let v_max = velocity.to_dtype(flame_core::DType::F32).ok()
                .and_then(|t| t.abs().ok())
                .and_then(|t| t.max_all().ok())
                .unwrap_or(f32::NAN);
            let x_max = x.to_dtype(flame_core::DType::F32).ok()
                .and_then(|t| t.abs().ok())
                .and_then(|t| t.max_all().ok())
                .unwrap_or(f32::NAN);
            // Spatial-correlation indicator: mean over (H,W) per (B, C, T) and the
            // std of that mean. If x is pure IID noise, mean-over-(H,W) ≈ 0; if
            // there's denoised spatial structure, the per-channel means vary.
            let xf = x.to_dtype(flame_core::DType::F32).unwrap();
            // x is [B, C=16, T, H, W]; mean over H (axis 3) and W (axis 4) gives [B, C, T]
            let xmean_chw = xf.mean_dim(&[3, 4], false).ok();
            let mean_std = xmean_chw.as_ref().and_then(|m| {
                let mean_v = m.to_vec1::<f32>().ok()?;
                let mu: f32 = mean_v.iter().sum::<f32>() / mean_v.len() as f32;
                let var: f32 = mean_v.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / mean_v.len() as f32;
                Some(var.sqrt())
            }).unwrap_or(f32::NAN);
            println!(
                "  step {:>3}/{}  ts={:.1}  sigma={:.4}  |v|_inf={:.3e}  |x|_inf={:.3e}  spatial_std={:.3e}  ({:.1}s)",
                step + 1, args.num_steps, ts,
                sigmas_vec[step], v_max, x_max, mean_std, t0.elapsed().as_secs_f32(),
            );
        }
    }

    drop(dit);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  DiT dropped, stage 3 total: {:.1}s", t0.elapsed().as_secs_f32());

    Ok(x)
}

// ---------------------------------------------------------------------------
// Stage 4 — VAE decode
// ---------------------------------------------------------------------------

/// Load Wan21 VAE decoder, decode latent `[1, 16, T_lat, H, W]` to pixel
/// `[1, 3, T_pix, H_pix, W_pix]`, drop decoder.
pub fn decode_pixels(
    latent: &Tensor, args: &CosmosPredict25Args, device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    println!("--- Stage 4: Wan21 VAE decode ---");
    let t0 = Instant::now();
    let vae_path = args.vae_path_resolved();
    println!("  vae path: {}", vae_path.display());
    let vae = Wan21VaeDecoder::load(
        vae_path.to_str().ok_or_else(|| anyhow::anyhow!("non-utf8 vae path"))?,
        device,
    )?;
    println!("  vae loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let rgb = vae.decode(latent)?;
    let dims = rgb.shape().dims().to_vec();
    println!("  decoded pixels: {:?}", dims);
    {
        let rgbf = rgb.to_dtype(DType::F32).unwrap();
        let pmax = rgbf.abs().unwrap().max_all().unwrap();
        let pmean = rgbf.mean_dim(&[0,1,2,3,4], false).unwrap().to_vec1::<f32>().unwrap()[0];
        let pmse = rgbf.mul(&rgbf).unwrap().mean_dim(&[0,1,2,3,4], false).unwrap().to_vec1::<f32>().unwrap()[0];
        println!("  pixel range: |p|_inf={:.3} mean={:.3} rms={:.3}", pmax, pmean, pmse.sqrt());
    }
    drop(vae);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    println!("  vae dropped, stage 4 total: {:.1}s", t0.elapsed().as_secs_f32());
    Ok(rgb)
}

// ---------------------------------------------------------------------------
// Stage 5 — mp4 mux
// ---------------------------------------------------------------------------

/// Write decoded pixels to an mp4 (video-only) under `args.output_dir`.
///
/// `pixels` is `[1, 3, T_pix, H, W]` BF16, range expected to be `[-1, 1]`
/// after VAE decode (Wan VAE normalizes to that range).
pub fn write_output(
    pixels: &Tensor, args: &CosmosPredict25Args, filename: &str,
) -> anyhow::Result<PathBuf> {
    println!("--- Stage 5: mp4 mux ---");
    std::fs::create_dir_all(&args.output_dir)?;
    let out_path = args.output_dir.join(filename);

    let dims = pixels.shape().dims().to_vec();
    if dims.len() != 5 || dims[0] != 1 || dims[1] != 3 {
        return Err(anyhow::anyhow!(
            "write_output: expected pixels [1,3,T,H,W], got {:?}", dims
        ));
    }
    let (_b, _c, t_pix, h_pix, w_pix) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

    let rgb_f32 = pixels.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    // Convert [1, 3, F, H, W] F32 ∈ [-1, 1] → [F, H, W, 3] u8 ∈ [0, 255].
    let frame_bytes = inference_flame::mux::video_tensor_to_rgb_u8(
        &rgb_f32, t_pix, h_pix, w_pix,
    );

    inference_flame::mux::write_mp4_video_only(
        &out_path, &frame_bytes, t_pix, w_pix, h_pix, args.fps,
    )?;

    println!("  wrote {}", out_path.display());
    Ok(out_path)
}

// ---------------------------------------------------------------------------
// Gaussian noise (Box-Muller, BF16) — same shape as wan_infer's helper.
// ---------------------------------------------------------------------------

fn gaussian_noise_bf16(
    numel: usize, seed: u64, shape: &[usize], device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
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
    let t = Tensor::from_vec(v, Shape::from_dims(shape), device.clone())?
        .to_dtype(DType::BF16)?;
    Ok(t)
}

// ---------------------------------------------------------------------------
// Top-level orchestrator for each mode
// ---------------------------------------------------------------------------

pub fn run(mode: Mode) -> anyhow::Result<()> {
    env_logger::init();
    let args = parse_args(mode);
    if let Err(e) = args.validate() {
        eprintln!("argument error: {e}");
        std::process::exit(2);
    }
    let device = global_cuda_device();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Cosmos-Predict2.5-2B  mode={:?}  resolution={:?}  variant={:?}",
             mode, args.resolution, args.variant);
    let (w, h) = args.dims_wh();
    println!("  pixel dims  : {w}x{h}");
    println!("  num_frames  : {} (latent T = {})",
             args.num_frames, args.num_latent_frames());
    println!("  num_steps   : {}  cfg={}  seed={}  sampler={:?}",
             args.num_steps, args.cfg, args.seed, args.sampler);
    if matches!(mode, Mode::I2V | Mode::V2V) {
        println!("  num_latent_cond_frames: {}", args.num_latent_conditional_frames);
    }
    println!("============================================================");

    // Stage 1: text encode.
    let (text_emb, neg_emb) = encode_text(&args, &device)?;

    // Stage 2: conditioning encode (i2v / v2v only).
    let (cond_lat, num_cond) = match mode {
        Mode::T2V => (None, 0),
        Mode::I2V => {
            let img = args.input_image.clone()
                .expect("--input-image required (validated earlier)");
            let (lat, n) = encode_conditioning_frames(&img, false, &args, &device)?;
            (Some(lat), n)
        }
        Mode::V2V => {
            let vid = args.input_video.clone()
                .expect("--input-video required (validated earlier)");
            let (lat, n) = encode_conditioning_frames(&vid, true, &args, &device)?;
            (Some(lat), n)
        }
    };

    // Stage 3: DiT denoise.
    let latent = denoise(
        &text_emb, neg_emb.as_ref(), cond_lat.as_ref(), num_cond, &args, &device,
    )?;
    drop(text_emb);
    drop(neg_emb);
    drop(cond_lat);

    // Stage 4: VAE decode.
    let pixels = decode_pixels(&latent, &args, &device)?;
    drop(latent);

    // Stage 5: mp4 mux.
    let filename = match mode {
        Mode::T2V => format!(
            "cosmos_predict25_t2v_{}_{}step.mp4", args.seed, args.num_steps
        ),
        Mode::I2V => format!(
            "cosmos_predict25_i2v_{}_{}step.mp4", args.seed, args.num_steps
        ),
        Mode::V2V => format!(
            "cosmos_predict25_v2v_{}_{}step.mp4", args.seed, args.num_steps
        ),
    };
    write_output(&pixels, &args, &filename)?;

    println!("============================================================");
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}
