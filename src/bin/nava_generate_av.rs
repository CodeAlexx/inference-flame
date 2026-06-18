//! NAVA (Ovi joint AV MM-DiT) text-to-audio-video generation — pure Rust.
//!
//! Scope: Base T2AV, B=1, BF16, single-GPU. NO timbre/speaker control (the
//! 3-pass align CFG path, not 4-pass), NO sequence-parallel.
//!
//! Pipeline (mirrors the sibling `ltx2_generate_av.rs`):
//!   Stage 1 (text encode): load umt5-xxl, encode positive + negative prompt,
//!     cache the contexts to `output/nava_embed_cache/`, then DROP the encoder
//!     and trim the CUDA pool so the 11 GB umt5 is NOT co-resident with the
//!     6.3 B DiT (the whole point of the memory plan).
//!   Stage 2 (denoise): `AutogradContext::set_enabled(false)`, load `WanAVModel`
//!     via `nava_loader::load_nava` (BlockOffloader streaming lives in the
//!     loader — out of this bin's scope), init video+audio noise, run the
//!     UniPC dual-scheduler with the 3-pass CFG (`NavaCfgConfig`).
//!   Stage 3 (decode + mux): video latents → Wan2.2 VAE → frames; audio latents
//!     → `nava_decode_audio` (LTX-2 audio VAE + vocoder) → 16 kHz waveform; mux
//!     to `output/nava_<seed>.mp4`.
//!
//! ## Reference loop
//! `ports/nava/nava_src/pipeline_nava.py:457-552` (3-pass align CFG, dual UniPC)
//! and `inference_nava.py` (shapes: video latent `[L_vid, 48]` ⇄ `[48, T, H, W]`,
//! audio latent `[audio_len, 128]`; `audio_len = ceil(((frames-1)*4+1)/fps * 25)`).
//!
//! The model's `WanAVModel::forward` takes `vid: [48, T, H, W]`, `audio:
//! [L_aud, 128]`, scalar timestep `t`, the shared text context (BOTH streams
//! cross-attend the video text context in base T2AV — see
//! `nava_blocks.rs:833`), `skip_layers`, `masking_modality`; it returns
//! `(eps_vid [48, T, H, W], eps_audio [L_aud, 128])`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use flame_core::{global_cuda_device, AutogradContext, CudaDevice, DType, Shape, Tensor};

use inference_flame::models::nava_av::NavaAVConfig;
use inference_flame::models::nava_loader;
use inference_flame::models::wan::t5::Umt5Encoder;
use inference_flame::sampling::nava_sampling::{
    nava_cfg_combine, NavaCfgConfig, NavaCfgInputs, NavaCfgPass, NavaDualScheduler,
};
use inference_flame::vae::{nava_decode_audio, LTX2AudioVaeDecoder, LTX2VocoderWithBWE, Wan22VaeDecoder};
use inference_flame::mux;

// ---------------------------------------------------------------------------
// Default weight paths. Grep the `const` (CONTEXT.md rule) before assuming —
// these mirror the PORT_SPEC `Weights` section. Overridable via CLI where it
// matters (`--ckpt`, `--out`).
// ---------------------------------------------------------------------------

/// NAVA DiT checkpoint (converted .ckpt(pickle) → safetensors, BF16). The
/// pickle→safetensors conversion is a one-time dev step (`ports/nava/parity/`).
const NAVA_CKPT: &str = "/home/alex/.serenity/models/checkpoints/nava_6b_bf16.safetensors";
/// umt5-xxl text encoder (same checkpoint Wan2.2 TI2V-5B uses), HF
/// `T5EncoderModel` layout safetensors.
const UMT5_PATH: &str = "/home/alex/.serenity/models/text_encoders/umt5_xxl.safetensors";
/// umt5 tokenizer (`google/umt5-xxl`).
const TOKENIZER_PATH: &str = "/home/alex/.serenity/models/text_encoders/umt5_xxl_tokenizer.json";
/// Wan2.2 video VAE (decode), safetensors.
const WAN22_VAE_PATH: &str = "/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors";
/// LTX-2 audio VAE + vocoder (NAVA vendors `ltx_core`; same family as our
/// `ltx2_audio_vae.rs` / `ltx2_vocoder.rs`). Both decoder + vocoder load from
/// this one file via their `from_file` constructors.
const LTX_AUDIO_VAE_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev_audio_vae.safetensors";

const OUTPUT_DIR: &str = "/home/alex/EriDiffusion/inference-flame/output";

// ---------------------------------------------------------------------------
// Generation defaults — SMOKE-FIRST LOW SETTINGS.
//
// Full-res NAVA is 1280×704; these defaults are intentionally tiny so the first
// smoke run is cheap. Bump via CLI once it runs. `frames`/`height`/`width` are
// LATENT-grid drivers: video latent T == `frames`, H_lat = height/16,
// W_lat = width/16 (Wan2.2 VAE spatial stride 16). See `inference_nava.py`
// (`h,w = height//patch_size, width//patch_size`, `video_latents [frames,h,w,48]`).
// ---------------------------------------------------------------------------
const DEFAULT_PROMPT: &str =
    "A person speaking to the camera in a bright room, clear voice, natural lighting.";
const DEFAULT_NEG: &str = "blurry, distorted, low quality, noisy, jittery";
const DEFAULT_WIDTH: usize = 512; // smoke-first; full-res is 1280
const DEFAULT_HEIGHT: usize = 320; // smoke-first; full-res is 704
const DEFAULT_FRAMES: usize = 5; // LATENT frames (smoke); ref default is 5
const DEFAULT_FPS: usize = 24;
const DEFAULT_STEPS: usize = 20;
const DEFAULT_SEED: u64 = 42;

/// Wan2.2 VAE spatial downsample factor (`spatial_downsample`/`patch_size=16`).
const VAE_SPATIAL_STRIDE: usize = 16;
/// Audio tokens per second of video. The class default is 31.25 (`t2v.py:324`),
/// but `configs/nava.yaml:55` sets `audio_tokens_per_sec: 25` and the runner reads
/// the config value (`inference_nava.py:500`), so NAVA's resolved value is 25.
const AUDIO_TOKENS_PER_SEC: f32 = 25.0;
/// Wan2.2 VAE temporal expansion: pixel frames = (latent_T - 1) * 4 + 1.
const VAE_TEMPORAL_STRIDE: usize = 4;

// ---------------------------------------------------------------------------
// CLI parsing (mirror ltx2_generate_av's get_arg / collect_* helpers).
// ---------------------------------------------------------------------------

fn get_arg<T: std::str::FromStr>(flag: &str) -> Option<T> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            return args.get(i + 1).and_then(|s| s.parse().ok());
        } else if let Some(val) = args[i].strip_prefix(&format!("{flag}=")) {
            return val.parse().ok();
        }
        i += 1;
    }
    None
}

/// Read a string flag (`--flag "with spaces"` or `--flag=...`), last-wins.
fn get_str_arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut out: Option<String> = None;
    let eq = format!("{flag}=");
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            out = args.get(i + 1).cloned();
            i += 2;
        } else if let Some(val) = args[i].strip_prefix(&eq) {
            out = Some(val.to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    // ---- CLI ----
    let prompt = get_str_arg("--prompt").unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let neg = get_str_arg("--neg").unwrap_or_else(|| DEFAULT_NEG.to_string());
    let width: usize = get_arg("--width").unwrap_or(DEFAULT_WIDTH);
    let height: usize = get_arg("--height").unwrap_or(DEFAULT_HEIGHT);
    let frames: usize = get_arg("--frames").unwrap_or(DEFAULT_FRAMES);
    let fps: usize = get_arg("--fps").unwrap_or(DEFAULT_FPS);
    let steps: usize = get_arg("--steps").unwrap_or(DEFAULT_STEPS);
    let seed: u64 = get_arg("--seed").unwrap_or(DEFAULT_SEED);
    let ckpt = get_str_arg("--ckpt").unwrap_or_else(|| NAVA_CKPT.to_string());
    let out_path =
        get_str_arg("--out").unwrap_or_else(|| format!("{OUTPUT_DIR}/nava_{seed}.mp4"));

    if width % VAE_SPATIAL_STRIDE != 0 || height % VAE_SPATIAL_STRIDE != 0 {
        anyhow::bail!("width/height must be multiples of {VAE_SPATIAL_STRIDE}");
    }

    let device = global_cuda_device();

    // ---- Latent geometry (matches inference_nava.py) ----
    // Video latent grid: T == `frames` (latent frames), H_lat/W_lat = size/16.
    let lat_t = frames;
    let lat_h = height / VAE_SPATIAL_STRIDE;
    let lat_w = width / VAE_SPATIAL_STRIDE;
    let cfg = NavaAVConfig::default();
    // Video patch (1,2,2) tokens — informational only; the model patchifies
    // internally. lat_h/lat_w must be even (patch 2) for the video patch-embed.
    if lat_h % cfg.patch_size[1] != 0 || lat_w % cfg.patch_size[2] != 0 {
        anyhow::bail!(
            "latent H/W ({lat_h}x{lat_w}) must be divisible by video patch {:?}; \
             pick width/height multiples of {}",
            cfg.patch_size,
            VAE_SPATIAL_STRIDE * cfg.patch_size[1]
        );
    }

    // Audio latent length: video_duration in *pixel* seconds, then 25 tok/s (nava.yaml).
    // pixel_frames = (lat_t - 1) * 4 + 1  (Wan2.2 temporal expansion).
    let pixel_frames = (lat_t - 1) * VAE_TEMPORAL_STRIDE + 1;
    let video_duration = pixel_frames as f32 / fps as f32;
    let audio_len = (video_duration * AUDIO_TOKENS_PER_SEC).ceil() as usize;
    let audio_ch = cfg.audio_in_dim; // 128

    println!("============================================================");
    println!("NAVA Audio+Video Generation — Pure Rust (Base T2AV)");
    println!("============================================================");
    println!("  {width}x{height}, latent grid [48,{lat_t},{lat_h},{lat_w}], {steps} steps");
    println!("  audio latent [{audio_len}, {audio_ch}], video duration {video_duration:.2}s @ {fps}fps");
    println!("  (SMOKE-FIRST low resolution; full-res NAVA is 1280x704)");
    let preview = |s: &str| if s.len() > 100 { format!("{}...", &s[..100]) } else { s.to_string() };
    println!("  prompt = {:?}", preview(&prompt));
    println!("  neg    = {:?}", preview(&neg));

    // =====================================================================
    // STAGE 1: Text encode (umt5-xxl) — SEPARATE, then drop before DiT load.
    // =====================================================================
    let cache_dir = format!("{OUTPUT_DIR}/nava_embed_cache");
    let ctx_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        prompt.hash(&mut h);
        neg.hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let pos_cache = format!("{cache_dir}/pos_{ctx_hash}.safetensors");
    let neg_cache = format!("{cache_dir}/neg_{ctx_hash}.safetensors");

    let (pos_ctx, neg_ctx) = if Path::new(&pos_cache).exists()
        && Path::new(&neg_cache).exists()
        && std::env::var("NAVA_REENCODE").is_err()
    {
        println!("\n--- Stage 1: Text encode (cached) ---");
        let pc = flame_core::serialization::load_file(Path::new(&pos_cache), &device)?;
        let nc = flame_core::serialization::load_file(Path::new(&neg_cache), &device)?;
        let p = pc
            .get("context")
            .ok_or_else(|| anyhow::anyhow!("cached pos ctx missing `context`: {pos_cache}"))?
            .to_dtype(DType::BF16)?;
        let n = nc
            .get("context")
            .ok_or_else(|| anyhow::anyhow!("cached neg ctx missing `context`: {neg_cache}"))?
            .to_dtype(DType::BF16)?;
        println!("  loaded pos_ctx={:?} neg_ctx={:?}", p.dims(), n.dims());
        (p, n)
    } else {
        println!("\n--- Stage 1: Text encode (umt5-xxl) ---");
        let t0 = Instant::now();
        let pos_ids = tokenize(TOKENIZER_PATH, &prompt)?;
        let neg_ids = tokenize(TOKENIZER_PATH, &neg)?;

        let mut t5 = Umt5Encoder::load(Path::new(UMT5_PATH), &device)
            .map_err(|e| anyhow::anyhow!("Umt5Encoder::load({UMT5_PATH}): {e:?}"))?;
        // umt5 encode → [1, 512, 4096] BF16 (zero-padded to text_len).
        let pos_ctx = t5.encode(&pos_ids)?;
        let neg_ctx = t5.encode(&neg_ids)?;
        println!(
            "  encoded pos={:?} neg={:?} in {:.1}s",
            pos_ctx.dims(),
            neg_ctx.dims(),
            t0.elapsed().as_secs_f32()
        );

        // Cache for re-runs.
        std::fs::create_dir_all(&cache_dir)?;
        let mut pm = HashMap::new();
        pm.insert("context".to_string(), pos_ctx.clone());
        flame_core::serialization::save_tensors(
            &pm,
            Path::new(&pos_cache),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        let mut nm = HashMap::new();
        nm.insert("context".to_string(), neg_ctx.clone());
        flame_core::serialization::save_tensors(
            &nm,
            Path::new(&neg_cache),
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;
        println!("  cached: {pos_cache} + {neg_cache}");

        // CRITICAL (memory plan): umt5 (~11 GB) must NOT co-reside with the
        // 6.3 B DiT. Drop the encoder and trim the CUDA pool before Stage 2.
        // Mirrors ltx2_generate_av.rs:420-437.
        drop(t5);
        let _ = device.synchronize();
        trim_cuda_pool();
        println!("  umt5 dropped + CUDA pool trimmed (VRAM reclaimed for DiT)");
        (pos_ctx, neg_ctx)
    };
    log_vram("after Stage 1");

    // =====================================================================
    // STAGE 2: Denoise (joint AV, 3-pass CFG, dual UniPC).
    // =====================================================================
    // Inference: graph retention would OOM (HiDream-O1 lesson). Disable early.
    AutogradContext::set_enabled(false);

    println!("\n--- Stage 2: Load DiT ({ckpt}) ---");
    let t0 = Instant::now();
    // The BlockOffloader streaming (6.3 B on 24 GB) is the loader's
    // responsibility — this bin only drives the loop. Assumed signature:
    //   nava_loader::load_nava(path: &str, cfg: &NavaAVConfig, device) -> Result<WanAVModel>
    // If it differs, the integration compile will flag it here (single call site).
    let model = nava_loader::load_nava(Path::new(&ckpt), &cfg, device.clone())
        .map_err(|e| anyhow::anyhow!("nava_loader::load_nava({ckpt}): {e:?}"))?;
    println!("  DiT loaded in {:.1}s", t0.elapsed().as_secs_f32());
    log_vram("after DiT load");

    // ---- Init noise ----
    // Video latent stored model-native as [48, T, H_lat, W_lat] (the forward
    // patchifies internally). Audio latent [audio_len, 128] (channel-last, as
    // the ChannelLastConv1d patch-embed expects).
    println!("\n--- Stage 2: Init noise + scheduler ---");
    let video_numel = cfg.vid_in_dim * lat_t * lat_h * lat_w;
    let mut video_x = make_noise(
        video_numel,
        seed,
        &[cfg.vid_in_dim, lat_t, lat_h, lat_w],
        &device,
    )?;
    let audio_numel = audio_len * audio_ch;
    let mut audio_x = make_noise(audio_numel, seed + 1, &[audio_len, audio_ch], &device)?;
    println!("  video noise {:?}, audio noise {:?}", video_x.dims(), audio_x.dims());

    // Dual UniPC scheduler — shift=5 for both modalities (configs/nava.yaml).
    let mut sched = NavaDualScheduler::new(steps, 5.0, 5.0);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();
    let cfg_cfg = NavaCfgConfig::default(); // video g=3, audio g=2, align_3d=true
    let passes = cfg_cfg.passes(); // [Cond, Uncond, Align]
    println!(
        "  {} steps, {} CFG passes/step ({:?})",
        timesteps.len(),
        passes.len(),
        passes
    );

    // ---- Denoise loop ----
    println!("\n--- Stage 2: Denoise ---");
    let t0 = Instant::now();
    for (step, &t) in timesteps.iter().enumerate() {
        let t_step = Instant::now();

        // Run each CFG pass; route context + skip/mask via pass_args. The model
        // forward returns per-modality velocities (eps_vid, eps_audio).
        let mut cond: Option<(Tensor, Tensor)> = None;
        let mut uncond: Option<(Tensor, Tensor)> = None;
        let mut align: Option<(Tensor, Tensor)> = None;

        for &pass in &passes {
            let pa = cfg_cfg.pass_args(pass);
            let text_ctx = if pa.use_neg_context { &neg_ctx } else { &pos_ctx };
            // Base T2AV: BOTH streams share the (video) text context. The model
            // forward currently uses `vid_ctx` and ignores `audio_ctx`
            // (nava_blocks.rs:833) — we pass the same tensor for both to stay
            // faithful to the "single shared context" reference behavior.
            let (eps_vid, eps_audio) = model
                .forward(
                    &video_x,
                    &audio_x,
                    t,
                    text_ctx,
                    text_ctx,
                    &pa.skip_layers,
                    pa.masking_modality,
                )
                .map_err(|e| anyhow::anyhow!("WanAVModel::forward (pass {pass:?}): {e:?}"))?;
            match pass {
                NavaCfgPass::Cond => cond = Some((eps_vid, eps_audio)),
                NavaCfgPass::Uncond => uncond = Some((eps_vid, eps_audio)),
                NavaCfgPass::Align => align = Some((eps_vid, eps_audio)),
            }
        }

        let (cond_v, cond_a) = cond.expect("cond pass always present");
        let (uncond_v, uncond_a) = uncond.expect("uncond pass always present");
        let (align_v, align_a) = match align {
            Some((v, a)) => (Some(v), Some(a)),
            None => (None, None),
        };

        // Cross-modal CFG combine (handles align_3d formula internally).
        let inputs = NavaCfgInputs {
            cond_vid: &cond_v,
            cond_audio: &cond_a,
            uncond_vid: &uncond_v,
            uncond_audio: &uncond_a,
            align_vid: align_v.as_ref(),
            align_audio: align_a.as_ref(),
        };
        let (eps_vid, eps_audio) = nava_cfg_combine(&cfg_cfg, &inputs)?;

        // Dual UniPC step (video + audio in lockstep).
        let (nv, na) = sched.step(&eps_vid, &video_x, &eps_audio, &audio_x)?;
        video_x = nv;
        audio_x = na;

        println!(
            "  step {}/{} t={:.2} dt={:.1}s",
            step + 1,
            timesteps.len(),
            t,
            t_step.elapsed().as_secs_f32()
        );
    }
    println!(
        "  denoised in {:.1}s ({:.2}s/step)",
        t0.elapsed().as_secs_f32(),
        t0.elapsed().as_secs_f32() / timesteps.len().max(1) as f32
    );

    // Free DiT before VAE decode to keep peak VRAM down.
    drop(model);
    let _ = device.synchronize();
    trim_cuda_pool();
    log_vram("after DiT drop");

    // =====================================================================
    // STAGE 3: Decode + mux.
    // =====================================================================
    println!("\n--- Stage 3: Decode video ---");
    let t0 = Instant::now();
    let vae = Wan22VaeDecoder::load(Path::new(WAN22_VAE_PATH), &device)
        .map_err(|e| anyhow::anyhow!("Wan22VaeDecoder::load({WAN22_VAE_PATH}): {e:?}"))?;
    // VAE decode wants [B, 48, T_lat, H_lat, W_lat]; add the batch dim.
    let video_5d = video_x.reshape(&[1, cfg.vid_in_dim, lat_t, lat_h, lat_w])?;
    // decode → [1, 3, F_pix, H_pix, W_pix], pixels in [-1, 1].
    let frames_t = vae
        .decode(&video_5d)
        .map_err(|e| anyhow::anyhow!("Wan22 VAE decode: {e:?}"))?;
    let fdims = frames_t.dims().to_vec();
    println!("  video decoded {:?} in {:.1}s", fdims, t0.elapsed().as_secs_f32());
    if fdims.len() != 5 {
        anyhow::bail!("expected video VAE output [1,3,F,H,W], got {fdims:?}");
    }
    let (out_f, out_h, out_w) = (fdims[2], fdims[3], fdims[4]);
    // mux::video_tensor_to_rgb_u8 expects [3, F, H, W] (channel-major) flat f32.
    let frames_f32 = frames_t
        .reshape(&[3, out_f, out_h, out_w])?
        .to_dtype(DType::F32)?
        .to_vec_f32()?;
    let rgb = mux::video_tensor_to_rgb_u8(&frames_f32, out_f, out_h, out_w);

    println!("\n--- Stage 3: Decode audio ---");
    let t0 = Instant::now();
    let audio_dec = LTX2AudioVaeDecoder::from_file(LTX_AUDIO_VAE_PATH, &device)
        .map_err(|e| anyhow::anyhow!("LTX2AudioVaeDecoder::from_file: {e:?}"))?;
    let vocoder = LTX2VocoderWithBWE::from_file(LTX_AUDIO_VAE_PATH, &device)
        .map_err(|e| anyhow::anyhow!("LTX2VocoderWithBWE::from_file: {e:?}"))?;
    // nava_decode_audio wants [B, 128, L]; we hold [L, 128] → [1, 128, L].
    let audio_b128l = audio_x
        .reshape(&[1, audio_len, audio_ch])?
        .transpose_dims(1, 2)? // [1, 128, L]
        .contiguous()?;
    let waveform = nava_decode_audio(&audio_b128l, &audio_dec, &vocoder)
        .map_err(|e| anyhow::anyhow!("nava_decode_audio: {e:?}"))?;
    let wdims = waveform.dims().to_vec();
    println!("  audio decoded {:?} (16 kHz) in {:.1}s", wdims, t0.elapsed().as_secs_f32());
    // waveform is [B, C, T] @ 16 kHz BF16. Flatten to interleaved s16 PCM.
    let (a_b, a_c, a_t) = (wdims[0], wdims[1], wdims[2]);
    let _ = a_b;
    let wav_f32 = waveform.to_dtype(DType::F32)?.to_vec_f32()?;
    let pcm = mux::audio_tensor_to_pcm_i16(&wav_f32, a_c, a_t, /*channels_first=*/ true);

    println!("\n--- Stage 3: Mux ---");
    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    mux::write_mp4(
        Path::new(&out_path),
        &rgb,
        out_f,
        out_w,
        out_h,
        fps as f32,
        &pcm,
        16_000,
    )
    .map_err(|e| anyhow::anyhow!("write_mp4({out_path}): {e}"))?;

    println!("\n============================================================");
    println!("  wrote {out_path}");
    println!("  total {:.1}s", t_total.elapsed().as_secs_f32());
    println!("============================================================");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tokenize with the umt5 `tokenizers`-crate tokenizer (adds special tokens),
/// mirroring `wan_infer::tokenize`.
fn tokenize(tokenizer_path: &str, text: &str) -> anyhow::Result<Vec<i32>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load ({tokenizer_path}): {e}"))?;
    let enc = tok
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

/// Box-Muller Gaussian noise (matches the sibling bin's `make_noise`).
fn make_noise(
    numel: usize,
    seed: u64,
    shape: &[usize],
    device: &Arc<CudaDevice>,
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
    Ok(Tensor::from_f32_to_bf16(v, Shape::from_dims(shape), device.clone())?)
}

/// Trim the CUDA default mem pool to free cached blocks (ltx2_generate_av:427-437).
fn trim_cuda_pool() {
    unsafe {
        extern "C" {
            fn cudaMemPoolTrimTo(pool: *mut std::ffi::c_void, min_bytes: usize) -> i32;
            fn cudaDeviceGetDefaultMemPool(pool: *mut *mut std::ffi::c_void, device: i32) -> i32;
        }
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let _ = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        if !pool.is_null() {
            let _ = cudaMemPoolTrimTo(pool, 0);
        }
    }
}

fn log_vram(when: &str) {
    if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
        println!(
            "  VRAM ({when}): {:.1}GB used / {:.1}GB total ({:.1}GB free)",
            (total - free) as f64 / 1e9,
            total as f64 / 1e9,
            free as f64 / 1e9
        );
    }
}
