//! daVinci-MagiHuman T2V inference (pure-Rust end-to-end).
//!
//! Usage:
//!     magihuman_infer --prompt-embeds-path X.safetensors \
//!                     --width 320 --height 192 --seconds 2 --steps 8 \
//!                     --out out.mp4
//!
//! Defaults are intentionally tiny (~3500 tokens) so the loop can be smoke-tested
//! on a 24 GB GPU before scaling up. For full quality, scale to
//! --width 832 --height 480 --seconds 4.

use std::path::{Path, PathBuf};

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use flame_core::{cuda_alloc_pool::clear_pool_cache, global_cuda_device, serialization, DType, Shape, Tensor};
use inference_flame::models::magihuman_dit::{
    MagiHumanDiTSwapped, AUDIO_IN_CHANNELS, TEXT_IN_CHANNELS,
};
use inference_flame::models::magihuman_sr_dit::MagiHumanSrDiTSwapped;
use inference_flame::models::sa_audio_vae::OobleckDecoder;
use inference_flame::models::turbo_vaed::{TurboVaedConfig, TurboVAED};
use inference_flame::models::wan::Wan22VaeEncoder;
use inference_flame::mux::{audio_tensor_to_pcm_i16, video_tensor_to_rgb_u8, write_mp4};

// ===========================================================================
// Minimal args (no clap dep — keep skeleton self-contained)
// ===========================================================================

struct Cli {
    prompt_embeds_path: PathBuf,
    image_path: Option<PathBuf>,
    width: usize,
    height: usize,
    seconds: usize,
    fps: usize,
    steps: usize,
    shift: f64,
    seed: u64,
    out: PathBuf,
    dit_weights: PathBuf,
    turbo_vaed_weights: PathBuf,
    sa_vae_weights: PathBuf,
    wan_vae_weights: PathBuf,
    // SR phase 2 (transformer2) — refines base 256p output to 1080p.
    // Disabled unless `sr_weights` is provided.
    sr_weights: Option<PathBuf>,
    sr_width: usize,
    sr_height: usize,
    sr_steps: usize,
    sr_noise_value: usize,         // 0..1000 index into ZeroSNR sigma table
    sr_audio_noise_scale: f64,
    // Classifier-free guidance. Mirrors Wan2GP's _run_diffusion_phase:
    //   cfg_number = 2 when cfg_scale > 1.0 OR audio_cfg_scale > 1.0
    //   pred = pred_uncond + scale * (pred_cond - pred_uncond)
    // Time-conditional video schedule: scale = cfg_scale when sigma_curr > 0.5
    // (Python's `t > 500`), else 2.0. Audio uses constant audio_cfg_scale.
    // SR phase uses sr_cfg_scale uniformly (no time switch).
    cfg_scale: f64,
    audio_cfg_scale: f64,
    sr_cfg_scale: f64,
    // Optional WAV file to drive talking-head motion. Without this audio_lat
    // is `randn_seeded` random noise → animation has no audio conditioning,
    // which is why the talking-head identity drifts and mouth motion is
    // disconnected from anything meaningful. With audio, the model's
    // audio cross-attention can drive lip-sync + expression.
    audio_path: Option<PathBuf>,
    // SR `cfg_trick`: creator's pipeline (see video_generate.py:415..418)
    // applies a lower CFG for the first `sr_cfg_trick_frame` LATENT frames,
    // then jumps to `sr_cfg_scale` for the rest. Creator defaults are
    // sr_cfg_scale=3.5 and cfg_trick_value=2.0 with start_frame=13.
    // For 1-sec / 7-frame outputs, all latent frames are < 13 so cfg_trick
    // value (2.0) applies throughout. For 5-sec / 32-frame outputs, frames
    // 0..12 use 2.0 and 13..31 use 3.5.
    // Pass `--sr-cfg-trick-frame 0` to disable.
    sr_cfg_trick_frame: usize,
    sr_cfg_trick_value: f64,
}

impl Cli {
    fn parse() -> Result<Self> {
        let mut prompt_embeds_path: Option<PathBuf> = None;
        let mut image_path: Option<PathBuf> = None;
        let mut width = 320;
        let mut height = 192;
        let mut seconds = 2;
        let mut fps = 25;
        let mut steps = 8;
        let mut shift = 5.0f64;
        let mut seed = 42u64;
        let mut out = PathBuf::from("magihuman_out.mp4");
        let mut dit_weights = PathBuf::from("/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors");
        let mut turbo_vaed_weights = PathBuf::from("/home/alex/.serenity/models/vaes/magihuman_turbo_vaed_decoder.safetensors");
        let mut sa_vae_weights = PathBuf::from("/home/alex/.serenity/models/vaes/stable_audio_oobleck_vae.safetensors");
        let mut wan_vae_weights = PathBuf::from("/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors");
        let mut sr_weights: Option<PathBuf> = None;
        let mut sr_width = 1920usize;
        let mut sr_height = 1088usize;
        let mut sr_steps = 5usize;
        let mut sr_noise_value = 220usize;
        let mut sr_audio_noise_scale = 0.7f64;
        let mut cfg_scale = 1.0f64;
        let mut audio_cfg_scale = 1.0f64;
        // SR CFG defaults match creator's official `example/sr_1080p/config.json`:
        // `sr_cfg_number=1` (no SR CFG, just cond pass). With sr_cfg_scale=1.0 the
        // sr_cfg_active branch is skipped and only the cond forward runs.
        // To enable CFG manually (creator's variant with `sr_cfg_number=2`), set
        // `--sr-cfg-scale 3.5` (creator's `sr_video_txt_guidance_scale`); the
        // cfg_trick logic below will then use --sr-cfg-trick-value (default 2.0)
        // for the first --sr-cfg-trick-frame latent frames (default 13).
        let mut sr_cfg_scale = 1.0f64;
        let mut sr_cfg_trick_frame = 13usize;
        let mut sr_cfg_trick_value = 2.0f64;
        let mut audio_path: Option<PathBuf> = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--prompt-embeds-path" => prompt_embeds_path = it.next().map(PathBuf::from),
                "--image-path" => image_path = it.next().map(PathBuf::from),
                "--width" => width = it.next().unwrap().parse()?,
                "--height" => height = it.next().unwrap().parse()?,
                "--seconds" => seconds = it.next().unwrap().parse()?,
                "--fps" => fps = it.next().unwrap().parse()?,
                "--steps" => steps = it.next().unwrap().parse()?,
                "--shift" => shift = it.next().unwrap().parse()?,
                "--seed" => seed = it.next().unwrap().parse()?,
                "--out" => out = it.next().unwrap().into(),
                "--dit-weights" => dit_weights = it.next().unwrap().into(),
                "--turbo-vaed-weights" => turbo_vaed_weights = it.next().unwrap().into(),
                "--sa-vae-weights" => sa_vae_weights = it.next().unwrap().into(),
                "--wan-vae-weights" => wan_vae_weights = it.next().unwrap().into(),
                "--sr-weights" => sr_weights = it.next().map(PathBuf::from),
                "--sr-width" => sr_width = it.next().unwrap().parse()?,
                "--sr-height" => sr_height = it.next().unwrap().parse()?,
                "--sr-steps" => sr_steps = it.next().unwrap().parse()?,
                "--sr-noise-value" => sr_noise_value = it.next().unwrap().parse()?,
                "--sr-audio-noise-scale" => sr_audio_noise_scale = it.next().unwrap().parse()?,
                "--cfg-scale" => cfg_scale = it.next().unwrap().parse()?,
                "--audio-cfg-scale" => audio_cfg_scale = it.next().unwrap().parse()?,
                "--sr-cfg-scale" => sr_cfg_scale = it.next().unwrap().parse()?,
                "--sr-cfg-trick-frame" => sr_cfg_trick_frame = it.next().unwrap().parse()?,
                "--sr-cfg-trick-value" => sr_cfg_trick_value = it.next().unwrap().parse()?,
                "--audio-path" => audio_path = it.next().map(PathBuf::from),
                other => anyhow::bail!("unknown arg: {other}"),
            }
        }
        Ok(Self {
            prompt_embeds_path: prompt_embeds_path
                .ok_or_else(|| anyhow!("--prompt-embeds-path required"))?,
            image_path,
            width, height, seconds, fps, steps, shift, seed, out,
            dit_weights, turbo_vaed_weights, sa_vae_weights, wan_vae_weights,
            sr_weights, sr_width, sr_height, sr_steps,
            sr_noise_value, sr_audio_noise_scale,
            cfg_scale, audio_cfg_scale, sr_cfg_scale,
            audio_path,
            sr_cfg_trick_frame, sr_cfg_trick_value,
        })
    }
}

// ===========================================================================
// Reference image → latent (i2v conditioning)
// ===========================================================================
//
// daVinci-MagiHuman is fundamentally i2v: the model takes a "reference image
// latent" as one of its inputs (per the model card and architecture diagram).
// In the Wan2GP / upstream pipeline, this reference latent is fed via the
// first temporal frame of the noisy video latent — at every step, BEFORE the
// forward pass:
//
//     latent_video[:, :, :1] = image_latent[:, :, :1]
//
// (See `_run_diffusion_phase` in Wan2GP's magi_human_model.py.)
//
// Without this injection, the model has no clean reference to attend to and
// produces garbage velocity. The "noise output despite plausible velocity
// magnitudes" symptom from the prior debugging session matches this exact
// failure mode.

/// Load `path` as RGB image, center-crop to target aspect ratio, Lanczos-
/// resize to `target_h × target_w`, return `[1, 3, 1, H, W]` BF16 tensor in
/// `[-1, 1]` for the Wan2.2 VAE encoder.
///
/// Mirrors creator's `resizecrop` (video_process.py:111..125): crop to target
/// aspect first to preserve subject proportions, THEN resize. Without the
/// crop, a square reference squishes vertically when the SR target is 16:9.
fn load_reference_image(
    path: &Path,
    target_h: usize,
    target_w: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| anyhow!("failed to open image {}: {e}", path.display()))?
        .to_rgb8();
    // Center-crop to target aspect ratio at native res.
    let (w, h) = (img.width() as usize, img.height() as usize);
    let aspect_target = target_h as f64 / target_w as f64;
    let aspect_image = h as f64 / w as f64;
    let cropped = if (w == target_w && h == target_h) || (aspect_image - aspect_target).abs() < 1e-6 {
        img
    } else {
        let (new_w, new_h) = if aspect_image > aspect_target {
            // Image is taller than target → crop top/bottom.
            let new_w = w;
            let new_h = ((new_w as f64) * (target_h as f64) / (target_w as f64)).round() as usize;
            (new_w, new_h)
        } else {
            // Image is wider than target → crop left/right.
            let new_h = h;
            let new_w = ((new_h as f64) * (target_w as f64) / (target_h as f64)).round() as usize;
            (new_w, new_h)
        };
        let left = (w.saturating_sub(new_w)) / 2;
        let top = (h.saturating_sub(new_h)) / 2;
        image::imageops::crop_imm(&img, left as u32, top as u32, new_w as u32, new_h as u32)
            .to_image()
    };
    let resized = image::imageops::resize(
        &cropped,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
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

/// Encode the reference image to the model's latent space using the Wan2.2
/// VAE encoder. Output: `[1, 48, 1, latent_h, latent_w]` F32 (post-normalized
/// — same scale as the noisy video latent).
///
/// Loads the encoder, encodes, drops the encoder before returning so we don't
/// hold ~1.4 GB of VAE weights through the DiT loop.
fn encode_reference_image(
    image_path: &Path,
    wan_vae_weights: &Path,
    pixel_h: usize,
    pixel_w: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    println!("[i2v  ] loading Wan2.2 VAE encoder from {}", wan_vae_weights.display());
    let encoder = Wan22VaeEncoder::load(wan_vae_weights, device)
        .map_err(|e| anyhow!("Wan22VaeEncoder load: {e}"))?;
    let image = load_reference_image(image_path, pixel_h, pixel_w, device)?;
    println!(
        "[i2v  ] image {} → tensor {:?} (BF16, [-1, 1])",
        image_path.display(),
        image.shape().dims()
    );
    let t0 = std::time::Instant::now();
    let latent = encoder
        .encode(&image)
        .map_err(|e| anyhow!("Wan22VaeEncoder encode: {e}"))?;
    eprintln!(
        "[i2v  ] encoded image latent shape={:?} in {} ms",
        latent.shape().dims(),
        t0.elapsed().as_millis()
    );
    drop(encoder);
    clear_pool_cache();
    latent.to_dtype(DType::F32).map_err(|e| anyhow!("{e:?}"))
}

/// Replace the first temporal frame of `video_lat` with `image_latent`.
/// Mirrors `latent_video[:, :, :1] = image_latent[:, :, :1]` from the Python
/// pipeline (called at the start of every step, before the forward pass).
fn inject_image_latent(video_lat: &Tensor, image_latent: &Tensor) -> Result<Tensor> {
    let dims = video_lat.shape().dims();
    if dims.len() != 5 || dims[2] < 1 {
        anyhow::bail!("inject_image_latent: bad video_lat shape {:?}", dims);
    }
    let img_dims = image_latent.shape().dims();
    if img_dims.len() != 5
        || img_dims[0] != dims[0]
        || img_dims[1] != dims[1]
        || img_dims[2] < 1
        || img_dims[3] != dims[3]
        || img_dims[4] != dims[4]
    {
        anyhow::bail!(
            "inject_image_latent: image_latent {:?} does not match video_lat {:?} on (B,C,H,W)",
            img_dims,
            dims
        );
    }
    let head = image_latent.narrow(2, 0, 1)?;
    if dims[2] == 1 {
        return Ok(head.contiguous()?);
    }
    let tail = video_lat.narrow(2, 1, dims[2] - 1)?;
    Tensor::cat(&[&head, &tail], 2).map_err(|e| anyhow!("{e:?}"))
}

// ===========================================================================
// Constants matching MagiHuman config
// ===========================================================================

const VAE_STRIDE_T: usize = 4;  // matches inference/common/config.py vae_stride[0]
const VAE_STRIDE_HW: usize = 16; // TurboVAED spatial_compression_ratio (config.py vae_stride[1,2])
const T_PATCH: usize = 1;
const HW_PATCH: usize = 2;
const Z_DIM: usize = 48; // Wan2.2 latent channels
const MAX_IN_CH: usize = TEXT_IN_CHANNELS; // 3584 — pad video/audio rows up to this width

// ===========================================================================
// Token assembly — turn raw latents + prompt embeds into [L, MAX_IN_CH] x and
// [L, 9] coords with V-A-T sort order.
// ===========================================================================

/// Build the (t, h, w, T, H, W, ref_T, ref_H, ref_W) coord rows for one modality.
/// Mirrors `data_proxy.get_coords` with offset_thw=[0,0,0].
fn make_coords(
    shape: [usize; 3],
    ref_shape: [usize; 3],
    offset: [i64; 3],
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let (t, h, w) = (shape[0], shape[1], shape[2]);
    let n = t * h * w;
    let mut data = Vec::with_capacity(n * 9);
    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                data.push((ti as i64 + offset[0]) as f32);
                data.push((hi as i64 + offset[1]) as f32);
                data.push((wi as i64 + offset[2]) as f32);
                data.push(t as f32);
                data.push(h as f32);
                data.push(w as f32);
                data.push(ref_shape[0] as f32);
                data.push(ref_shape[1] as f32);
                data.push(ref_shape[2] as f32);
            }
        }
    }
    Tensor::from_vec(data, Shape::from_dims(&[n, 9]), device.clone()).map_err(|e| anyhow!("{e:?}"))
}

/// Patchify a [1, 48, T_lat, H_lat, W_lat] latent into [V_count, 192] tokens.
///
/// **C-MAJOR per-token channel order** — matches Python's `UnfoldNd` (the
/// `(C, pT, pH, pW)` flattening used by `data_proxy.img2tokens`). Verified
/// experimentally: `unfoldNd.UnfoldNd(kernel=(1,2,2), stride=(1,2,2))` on
/// `arange(8).reshape(1,2,1,2,2)` outputs `[0,1,2,3,4,5,6,7]` (C outer).
///
/// Pre-fix this function used `(pT, pH, pW, C)` order — the model saw
/// scrambled video tokens and output stayed at noise scale across all sigma
/// steps regardless of step count or coords convention.
fn patchify_video(latent: &Tensor) -> Result<Tensor> {
    let dims = latent.shape().dims();
    if dims.len() != 5 || dims[0] != 1 || dims[1] != Z_DIM {
        anyhow::bail!("patchify_video: expected [1, {Z_DIM}, T, H, W], got {:?}", dims);
    }
    let (t, h, w) = (dims[2], dims[3], dims[4]);
    if h % HW_PATCH != 0 || w % HW_PATCH != 0 || t % T_PATCH != 0 {
        anyhow::bail!("patchify_video: dims not divisible by patch (t={t} h={h} w={w})");
    }
    let lt = t / T_PATCH;
    let lh = h / HW_PATCH;
    let lw = w / HW_PATCH;
    // [1, C, T, H, W] → [1, C, lt, pT, lh, pH, lw, pW]
    //                 → [1, lt, lh, lw, C, pT, pH, pW]   (C outer of patch dims)
    //                 → [V_count, C*pT*pH*pW]
    let x = latent
        .reshape(&[1, Z_DIM, lt, T_PATCH, lh, HW_PATCH, lw, HW_PATCH])?
        .permute(&[0, 2, 4, 6, 1, 3, 5, 7])?
        .contiguous()?
        .reshape(&[lt * lh * lw, Z_DIM * T_PATCH * HW_PATCH * HW_PATCH])?;
    Ok(x)
}

/// Inverse of patchify_video — but **C-INNER**, NOT the same order as the
/// patchify (which is C-outer). The model's input and output use different
/// per-token channel orderings:
///
/// - **Input** (patchify_video / data_proxy.img2tokens): `(C, pT, pH, pW)`
///   per-token, C-outer. UnfoldNd's permute lays out col_dim as
///   `[C, kt, kh, kw]`.
/// - **Output** (unpatchify_video / data_proxy.depack_token_sequence): einops
///   pattern `"(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)"` — C is
///   INNERMOST in the model output 192-dim.
///
/// This asymmetry comes from the `final_linear_video` layout: the linear's
/// output channels were trained against targets with C as the inner axis of
/// the patch.
///
/// The pre-fix code used the same C-outer order as patchify; it scrambled
/// every velocity prediction. Frame 0 still looked correct in i2v mode only
/// because we re-inject the encoded reference latent at frame 0 after every
/// step (bypassing this unpatchify); frames 1+ stayed at noise because the
/// model's velocity for them was being unpacked into the wrong channel slots.
fn unpatchify_video(tokens: &Tensor, lt: usize, lh: usize, lw: usize) -> Result<Tensor> {
    // tokens [V_count, 192] → [1, lt, lh, lw, pT, pH, pW, C]   (C inner)
    //                       → [1, C, lt, pT, lh, pH, lw, pW]
    //                       → [1, C, T, H, W]
    Ok(tokens
        .reshape(&[1, lt, lh, lw, T_PATCH, HW_PATCH, HW_PATCH, Z_DIM])?
        .permute(&[0, 7, 1, 4, 2, 5, 3, 6])?
        .contiguous()?
        .reshape(&[1, Z_DIM, lt * T_PATCH, lh * HW_PATCH, lw * HW_PATCH])?)
}

/// Pad token rows up to MAX_IN_CH (3584) along dim 1 with zeros.
/// Required so V/A/T can be concat'd into [L, MAX_IN_CH].
fn pad_channels(t: &Tensor, target: usize) -> Result<Tensor> {
    let dims = t.shape().dims();
    let cur = dims[dims.len() - 1];
    if cur == target {
        return Ok(t.clone());
    }
    if cur > target {
        anyhow::bail!("pad_channels: cur {cur} > target {target}");
    }
    let mut zero_dims = dims.to_vec();
    *zero_dims.last_mut().unwrap() = target - cur;
    let zeros = Tensor::zeros_dtype(Shape::from_dims(&zero_dims), t.dtype(), t.device().clone())?;
    Tensor::cat(&[t, &zeros], dims.len() - 1).map_err(|e| anyhow!("{e:?}"))
}

struct PackedInput {
    x: Tensor,            // [L, MAX_IN_CH] F32
    coords: Tensor,       // [L, 9] F32
    group_sizes: [usize; 3], // [V, A, T]
    video_grid: (usize, usize, usize), // (lt, lh, lw) for unpatchify
}

fn pack_inputs(
    video_lat: &Tensor,
    audio_lat: &Tensor,
    text_embeds: &Tensor,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<PackedInput> {
    // ----- Video -----
    let dims = video_lat.shape().dims();
    let (lt, lh, lw) = (dims[2] / T_PATCH, dims[3] / HW_PATCH, dims[4] / HW_PATCH);
    let v_tokens = patchify_video(video_lat)?; // [V, 192]
    let v_count = v_tokens.shape().dims()[0];

    // ----- Audio -----
    let a_dims = audio_lat.shape().dims();
    if a_dims[0] != 1 || a_dims[2] != AUDIO_IN_CHANNELS {
        anyhow::bail!("pack_inputs: audio_lat must be [1, N, {AUDIO_IN_CHANNELS}], got {:?}", a_dims);
    }
    let a_count = a_dims[1];
    let a_tokens = audio_lat.reshape(&[a_count, AUDIO_IN_CHANNELS])?; // [A, 64]

    // ----- Text -----
    let t_dims = text_embeds.shape().dims();
    if t_dims[1] != TEXT_IN_CHANNELS {
        anyhow::bail!("pack_inputs: text_embeds last dim must be {TEXT_IN_CHANNELS}, got {:?}", t_dims);
    }
    let t_count = t_dims[0];

    // ----- Pad each to MAX_IN_CH and concat in V→A→T order -----
    let v_pad = pad_channels(&v_tokens, MAX_IN_CH)?;
    let a_pad = pad_channels(&a_tokens, MAX_IN_CH)?;
    let t_pad = pad_channels(text_embeds, MAX_IN_CH)?;
    let x = Tensor::cat(&[&v_pad, &a_pad, &t_pad], 0)?.to_dtype(DType::F32)?;

    // ----- Coords (production default: v2 per inference/common/config.py:109).
    let video_coords = make_coords([lt, lh, lw], [lt, lh, lw], [0, 0, 0], device)?;
    let magic_audio_ref_t = (a_count.saturating_sub(1)) / 4 + 1;
    let audio_coords = make_coords(
        [a_count, 1, 1],
        [magic_audio_ref_t / T_PATCH, 1, 1],
        [0, 0, 0],
        device,
    )?;
    let text_coords = make_coords(
        [t_count, 1, 1],
        [1, 1, 1],
        [-(t_count as i64), 0, 0],
        device,
    )?;
    let coords = Tensor::cat(&[&video_coords, &audio_coords, &text_coords], 0)?;

    Ok(PackedInput {
        x,
        coords,
        group_sizes: [v_count, a_count, t_count],
        video_grid: (lt, lh, lw),
    })
}

// ===========================================================================
// SR phase 2 (transformer2) helpers — refines base 256p output to 1080p.
// ===========================================================================
//
// Pipeline (matches Wan2GP magi_human_model.py:732..765):
//   1. Encode reference image at FULL SR resolution (Wan2.2 VAE).
//   2. Trilinear-upsample base latent_video (lt unchanged) to SR latent dims.
//      `align_corners=True` per Python; we mirror via a 2D bilinear with
//      align_corners on each frame (T_in == T_out so trilinear collapses).
//   3. Add noise: lat = lat * sigma + randn * sqrt(1 - sigma²) where sigma
//      comes from `_sr_sigmas[sr_noise_value]` (ZeroSNR-derived, decreasing).
//   4. Audio noise blend: a_lat = randn * sr_audio_noise_scale +
//                                 a_lat * (1 - sr_audio_noise_scale).
//   5. Re-load DiT with transformer2 weights (same architecture).
//   6. Run sampling for `sr_num_inference_steps` (default 5) with v1 coords,
//      shift=5, FlowUniPC sigmas. update_audio=False (audio not denoised).
//   7. Final image_latent injection.
//
// Defaults from magi_human_distill_sr1080.json: sr_num_inference_steps=5,
// sr_noise_value=220, sr_audio_noise_scale=0.7. CFG is disabled
// (sr_cfg_number=1 is the default for distill).

/// 3D bilinear upsample, spatial axes only (T preserved). Mirrors
/// `F.interpolate(mode="trilinear", align_corners=True)` for the case where
/// input_T == output_T (which is what the SR phase uses — only HW changes).
fn bilinear_upsample_3d_spatial(x: &Tensor, new_h: usize, new_w: usize) -> Result<Tensor> {
    use flame_core::cuda_ops::GpuOps;
    let dims = x.shape().dims();
    if dims.len() != 5 {
        anyhow::bail!("bilinear_upsample_3d_spatial: expected 5D, got {:?}", dims);
    }
    let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    // (B, C, T, H, W) → (B, T, C, H, W) → (B*T, C, H, W)
    let r = x.permute(&[0, 2, 1, 3, 4])?.contiguous()?;
    let r = r.reshape(&[b * t, c, h, w])?;
    // Wan2GP uses align_corners=True for the trilinear interp (line 737).
    let up = GpuOps::upsample2d_bilinear(&r, (new_h, new_w), true)?;
    let r = up.reshape(&[b, t, c, new_h, new_w])?;
    r.permute(&[0, 2, 1, 3, 4])?
        .contiguous()
        .map_err(|e| anyhow!("{e:?}"))
}

/// Compute the full ZeroSNR sigma table (1000 entries, decreasing 0.999..0).
///
/// Mirrors `_ZeroSNRDDPMDiscretization()(num_timesteps=1000, do_append_zero=False, flip=True)`
/// from magi_human_model.py:88..132 with default args (linear_start=0.00085,
/// linear_end=0.0120, shift_scale=1.0, post_shift=False).
fn zero_snr_sigmas(num_timesteps: usize) -> Vec<f64> {
    let linear_start: f64 = 0.00085;
    let linear_end: f64 = 0.0120;
    let n = num_timesteps;
    // betas = linspace(sqrt(start), sqrt(end), N)^2
    let sqrt_start = linear_start.sqrt();
    let sqrt_end = linear_end.sqrt();
    let mut betas = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let b = sqrt_start + t * (sqrt_end - sqrt_start);
        betas.push(b * b);
    }
    // alphas_cumprod = cumprod(1 - betas)
    let mut acp = Vec::with_capacity(n);
    let mut acc = 1.0f64;
    for &b in &betas {
        acc *= 1.0 - b;
        acp.push(acc);
    }
    // sqrt → rescale: sqrt -= sqrt[-1]; sqrt *= sqrt[0] / (sqrt[0] - sqrt[-1]);
    let mut sqrt_acp: Vec<f64> = acp.iter().map(|x| x.sqrt()).collect();
    let s0 = sqrt_acp[0];
    let st = sqrt_acp[n - 1];
    let scale = s0 / (s0 - st);
    for v in sqrt_acp.iter_mut() {
        *v = (*v - st) * scale;
    }
    // get_sigmas does flip(); __call__ with flip=True flips again. Net: original order.
    // sqrt_acp[0] ≈ 0.999 (max), sqrt_acp[N-1] = 0 (min). Decreasing.
    sqrt_acp
}

/// Build production-`v1` coords (used by sr_data_proxy in Wan2GP). Differs
/// from v2:
///   - Audio ref_feat_shape = (lt, 1, 1) instead of (magic_audio_ref_t, 1, 1)
///   - Text ref_feat_shape = (2, 1, 1) instead of (1, 1, 1)
///   - Text offset = 0 instead of -txt_feat_len
fn make_coords_v1(
    lt: usize,
    lh: usize,
    lw: usize,
    a_count: usize,
    t_count: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let video_coords = make_coords([lt, lh, lw], [lt, lh, lw], [0, 0, 0], device)?;
    let audio_coords = make_coords(
        [a_count, 1, 1],
        [lt / T_PATCH, 1, 1],
        [0, 0, 0],
        device,
    )?;
    let text_coords = make_coords(
        [t_count, 1, 1],
        [2, 1, 1],
        [0, 0, 0],
        device,
    )?;
    Tensor::cat(&[&video_coords, &audio_coords, &text_coords], 0).map_err(|e| anyhow!("{e:?}"))
}

/// Build the v1-coords PackedInput (same packing as `pack_inputs` but with v1
/// coords). The modality concat order V→A→T is identical so the model sees
/// the same row layout.
fn pack_inputs_v1(
    video_lat: &Tensor,
    audio_lat: &Tensor,
    text_embeds: &Tensor,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<PackedInput> {
    let dims = video_lat.shape().dims();
    let (lt, lh, lw) = (dims[2] / T_PATCH, dims[3] / HW_PATCH, dims[4] / HW_PATCH);
    let v_tokens = patchify_video(video_lat)?;
    let v_count = v_tokens.shape().dims()[0];
    let a_dims = audio_lat.shape().dims();
    let a_count = a_dims[1];
    let a_tokens = audio_lat.reshape(&[a_count, AUDIO_IN_CHANNELS])?;
    let t_count = text_embeds.shape().dims()[0];

    let v_pad = pad_channels(&v_tokens, MAX_IN_CH)?;
    let a_pad = pad_channels(&a_tokens, MAX_IN_CH)?;
    let t_pad = pad_channels(text_embeds, MAX_IN_CH)?;
    let x = Tensor::cat(&[&v_pad, &a_pad, &t_pad], 0)?.to_dtype(DType::F32)?;

    let coords = make_coords_v1(lt, lh, lw, a_count, t_count, device)?;
    Ok(PackedInput {
        x,
        coords,
        group_sizes: [v_count, a_count, t_count],
        video_grid: (lt, lh, lw),
    })
}

// ===========================================================================
// UniPC multistep predictor-corrector (bh2 solver, predict_x0, flow_prediction).
// Used for the SR phase. Direct port of FlowUniPCMultistepScheduler.step()
// with solver_order=2, lower_order_final=True. Accuracy is order_eff=order+1
// because of the corrector — that's why Python uses 5 SR steps where DDIM
// would need ~16 to match.
//
// State maintained across steps:
//   - sigmas: full precomputed sigma table (length N+1, decreasing high→0).
//   - step_index: which step we're on (0..N-1).
//   - lower_order_nums: incremented after each step; warms up to solver_order.
//   - this_order: order used for this step's predictor (and stored for the
//                 NEXT step's corrector).
//   - last_sample: pre-predictor sample from previous step.
//   - model_outputs: ring of last `solver_order` converted model outputs
//                    (= x0_pred for predict_x0 + flow_prediction).
// ===========================================================================

struct UniPCSampler {
    sigmas: Vec<f64>,
    solver_order: usize,
    step_index: usize,
    lower_order_nums: usize,
    this_order: usize,
    last_sample: Option<Tensor>,
    model_outputs: Vec<Option<Tensor>>,
}

impl UniPCSampler {
    fn new(sigmas: Vec<f64>, solver_order: usize) -> Self {
        Self {
            sigmas,
            solver_order,
            step_index: 0,
            lower_order_nums: 0,
            this_order: 0,
            last_sample: None,
            model_outputs: vec![None; solver_order],
        }
    }

    /// Run one step. Takes the raw model velocity output and current sample,
    /// returns the next sample.
    fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma_curr = self.sigmas[self.step_index] as f32;
        // convert_model_output (predict_x0 + flow_prediction):
        //     m_convert = sample - sigma_curr * model_output   (= x0_pred)
        let m_convert = sample.sub(&model_output.mul_scalar(sigma_curr)?)?;

        // Corrector: applies if step_index > 0 AND we have last_sample.
        let use_corrector = self.step_index > 0 && self.last_sample.is_some();
        let mut current_sample = sample.clone();
        if use_corrector {
            let last = self.last_sample.clone().unwrap();
            current_sample = self.uni_c_update(&m_convert, &last, &current_sample, self.this_order)?;
        }

        // Shift model_outputs left, append new converted output at end.
        for i in 0..(self.solver_order - 1) {
            self.model_outputs[i] = self.model_outputs[i + 1].clone();
        }
        self.model_outputs[self.solver_order - 1] = Some(m_convert);

        // Determine this_order: lower_order_final + warmup.
        let n_total = self.sigmas.len() - 1;
        let this_order_lof = std::cmp::min(self.solver_order, n_total - self.step_index);
        self.this_order = std::cmp::min(this_order_lof, self.lower_order_nums + 1);

        // Save current_sample as last_sample for next step's corrector input.
        self.last_sample = Some(current_sample.clone());

        // Run predictor.
        let prev_sample = self.uni_p_update(&current_sample, self.this_order)?;

        // Update state.
        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;

        Ok(prev_sample)
    }

    /// UniP (predictor) — bh2, predict_x0, flow_prediction.
    /// Order 1: x_t = (sigma_t/sigma_s0)*x - alpha_t*h_phi_1*m0.
    /// Order 2: above MINUS alpha_t * B_h * 0.5 * (m_prev - m0) / rk.
    fn uni_p_update(&self, sample: &Tensor, order: usize) -> Result<Tensor> {
        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .ok_or_else(|| anyhow!("uni_p: m0 None"))?;
        let sigma_t = self.sigmas[self.step_index + 1];
        let sigma_s0 = self.sigmas[self.step_index];
        let (alpha_t, alpha_s0) = (1.0 - sigma_t, 1.0 - sigma_s0);
        let lambda_t = (alpha_t / sigma_t).ln();
        let lambda_s0 = (alpha_s0 / sigma_s0).ln();
        let h = lambda_t - lambda_s0;
        let hh = -h; // predict_x0
        let h_phi_1 = hh.exp_m1();
        let bh = h_phi_1; // bh2

        // x_t_ = (sigma_t / sigma_s0) * x - alpha_t * h_phi_1 * m0
        let x_t_ = sample
            .mul_scalar((sigma_t / sigma_s0) as f32)?
            .sub(&m0.mul_scalar((alpha_t * h_phi_1) as f32)?)?;
        if order == 1 {
            return Ok(x_t_);
        }

        // Order 2: D1 = (m_prev - m0) / rk, where rk = (lambda_prev - lambda_s0) / h.
        let m_prev = self.model_outputs[self.solver_order - 2]
            .as_ref()
            .ok_or_else(|| anyhow!("uni_p order=2: m_prev None"))?;
        let sigma_prev = self.sigmas[self.step_index - 1];
        let alpha_prev = 1.0 - sigma_prev;
        let lambda_prev = (alpha_prev / sigma_prev).ln();
        let rk = (lambda_prev - lambda_s0) / h;
        let d1 = m_prev.sub(m0)?.mul_scalar((1.0 / rk) as f32)?;
        // pred_res = 0.5 * D1; x_t = x_t_ - alpha_t * B_h * pred_res.
        let x_t = x_t_.sub(&d1.mul_scalar((alpha_t * bh * 0.5) as f32)?)?;
        Ok(x_t)
    }

    /// UniC (corrector) — bh2, predict_x0, flow_prediction.
    /// Order 1: simplified rhos_c = [0.5]. No D1s.
    ///     x_t = (sigma_t/sigma_s0)*last_sample - alpha_t*h_phi_1*m0
    ///         - alpha_t * B_h * 0.5 * (m_this - m0)
    /// Order 2: solve 2x2 R*rho = b system. Uses m_prev_prev (2 steps back).
    ///     x_t = above_x_t_ - alpha_t * B_h * (rho_0 * D1_old + rho_1 * (m_this - m0))
    fn uni_c_update(
        &self,
        m_this: &Tensor,
        last_sample: &Tensor,
        this_sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        // m0 = previously stored last model output (m_{i-1}).
        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .ok_or_else(|| anyhow!("uni_c: m0 None"))?;
        let sigma_t = self.sigmas[self.step_index];
        let sigma_s0 = self.sigmas[self.step_index - 1];
        let (alpha_t, alpha_s0) = (1.0 - sigma_t, 1.0 - sigma_s0);
        let lambda_t = (alpha_t / sigma_t).ln();
        let lambda_s0 = (alpha_s0 / sigma_s0).ln();
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let bh = h_phi_1;

        let x_t_ = last_sample
            .mul_scalar((sigma_t / sigma_s0) as f32)?
            .sub(&m0.mul_scalar((alpha_t * h_phi_1) as f32)?)?;
        let _ = this_sample; // pre-corrector sample is what's already in `last_sample` of caller

        let d1_t = m_this.sub(m0)?;

        if order == 1 {
            // No D1s. rhos_c = [0.5]. corr_res = 0.
            // x_t = x_t_ - alpha_t * B_h * 0.5 * D1_t
            let correction = d1_t.mul_scalar((alpha_t * bh * 0.5) as f32)?;
            return Ok(x_t_.sub(&correction)?);
        }

        // Order 2:
        //   m_prev_prev = model_outputs[-2] (= m_{i-2})
        //   rk = (lambda_{i-2} - lambda_s0) / h
        //   D1_old = (m_prev_prev - m0) / rk
        //   Solve [[1, 1], [rk, 1]] * [rho_0, rho_1]^T = [b0, b1]^T
        //     b0 = h_phi_k_1 / bh,                where h_phi_k_1 = h_phi_1/hh - 1
        //     b1 = 2 * h_phi_k_2 / bh,            where h_phi_k_2 = h_phi_k_1/hh - 1/2
        //   x_t = x_t_ - alpha_t * B_h * (rho_0 * D1_old + rho_1 * D1_t)
        let m_prev_prev = self.model_outputs[self.solver_order - 2]
            .as_ref()
            .ok_or_else(|| anyhow!("uni_c order=2: m_prev_prev None"))?;
        let sigma_pp = self.sigmas[self.step_index - 2];
        let alpha_pp = 1.0 - sigma_pp;
        let lambda_pp = (alpha_pp / sigma_pp).ln();
        let rk = (lambda_pp - lambda_s0) / h;
        let d1_old = m_prev_prev.sub(m0)?.mul_scalar((1.0 / rk) as f32)?;

        let h_phi_k_1 = h_phi_1 / hh - 1.0;
        let b0 = h_phi_k_1 / bh;
        let h_phi_k_2 = h_phi_k_1 / hh - 0.5;
        let b1 = 2.0 * h_phi_k_2 / bh;
        let det = 1.0 - rk;
        let rho_0 = (b0 - b1) / det;
        let rho_1 = (b1 - rk * b0) / det;

        let combined = d1_old
            .mul_scalar(rho_0 as f32)?
            .add(&d1_t.mul_scalar(rho_1 as f32)?)?;
        let x_t = x_t_.sub(&combined.mul_scalar((alpha_t * bh) as f32)?)?;
        Ok(x_t)
    }
}

// ===========================================================================
// Sampling step (DDIM with stochastic resample) — mirrors step_ddim in
// FlowUniPCMultistepScheduler.
// ===========================================================================

fn step_ddim(
    velocity: &Tensor,
    state: &Tensor,
    sigma_curr: f32,
    sigma_next: f32,
    noise: &Tensor,
) -> Result<Tensor> {
    // x0_estimate = state - sigma_curr * velocity
    let x0 = state.sub(&velocity.mul_scalar(sigma_curr)?)?;
    // prev = sigma_next * noise + (1 - sigma_next) * x0
    let term_noise = noise.mul_scalar(sigma_next)?;
    let term_x0 = x0.mul_scalar(1.0 - sigma_next)?;
    term_noise.add(&term_x0).map_err(|e| anyhow!("{e:?}"))
}

// ===========================================================================
// Velocity extraction helper. Splits a DiT forward output into the
// video and audio velocity tensors. Used in cond and uncond CFG passes.
// ===========================================================================
//
// Returns (v_velocity: [1, Z_DIM, lt*T_PATCH, lh*HW_PATCH, lw*HW_PATCH] F32,
//          a_velocity: [1, a_count, AUDIO_IN_CHANNELS] F32).
fn extract_velocities(
    dit_out: &Tensor,
    v_count: usize,
    a_count: usize,
    lt: usize, lh: usize, lw: usize,
) -> Result<(Tensor, Tensor)> {
    let v_vel_tokens = dit_out.narrow(0, 0, v_count)?;
    let v_velocity = unpatchify_video(&v_vel_tokens, lt, lh, lw)?;
    let a_vel_tokens = dit_out
        .narrow(0, v_count, a_count)?
        .narrow(1, 0, AUDIO_IN_CHANNELS)?;
    let a_velocity = a_vel_tokens.reshape(&[1, a_count, AUDIO_IN_CHANNELS])?;
    Ok((v_velocity, a_velocity))
}

// ===========================================================================
// Sigma table (mirrors set_timesteps in scheduler)
// ===========================================================================

fn build_sigmas(num_inference_steps: usize, shift: f64) -> Vec<f32> {
    // Linear sigmas = 1 - alpha_cumprod where alphas decreasing from 1 → 1/1000
    // Then shift and subsample.
    let n_train = 1000usize;
    let alphas: Vec<f64> = (0..n_train)
        .map(|i| {
            let t = i as f64 / (n_train - 1) as f64; // 0..=1
            1.0 - t * (1.0 - 1.0 / 1000.0) // alpha = 1 → 1/1000
        })
        .collect();
    let sigmas: Vec<f64> = alphas.iter().map(|a| 1.0 - a).collect();
    let shifted: Vec<f64> = sigmas
        .iter()
        .map(|s| shift * s / (1.0 + (shift - 1.0) * s))
        .collect();
    // Subsample: linspace(0, n_train, n+1) → take indices, drop last (per scheduler conv)
    let mut out = Vec::with_capacity(num_inference_steps + 1);
    for i in 0..=num_inference_steps {
        let idx = ((i as f64 / num_inference_steps as f64) * (n_train as f64))
            .round() as usize;
        let idx = idx.min(n_train - 1);
        out.push(shifted[idx] as f32);
    }
    // Reverse so we go from high noise to low noise (sigma_T ≈ 1 down to sigma_0 ≈ 0)
    out.reverse();
    // After reverse: sigmas[0] = highest, sigmas[num_inference_steps] = 0
    out
}

// ===========================================================================
// Seed helpers — keep noise reproducible without one big global RNG.
// ===========================================================================

fn seed_for(base: u64, role: u64, step: u64) -> u64 {
    base.wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(role.wrapping_mul(0xBF58476D1CE4E5B9))
        .wrapping_add(step.wrapping_mul(0x94D049BB133111EB))
}

// ===========================================================================
// Main
// ===========================================================================

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse()?;
    let device = global_cuda_device();

    // --- Resolve shapes ---
    let num_frames = cli.seconds * cli.fps + 1; // matches video_generate.py
    let latent_t = (num_frames - 1) / VAE_STRIDE_T + 1;
    let latent_h = (cli.height / VAE_STRIDE_HW / HW_PATCH) * HW_PATCH;
    let latent_w = (cli.width / VAE_STRIDE_HW / HW_PATCH) * HW_PATCH;
    println!(
        "[shapes] frames={num_frames} latent=[1,{Z_DIM},{latent_t},{latent_h},{latent_w}]"
    );
    println!(
        "[shapes] tokens: V={} A={} T=(from prompt embeds)",
        latent_t * latent_h * latent_w / (T_PATCH * HW_PATCH * HW_PATCH),
        num_frames,
    );

    // --- Load prompt embeds (the only thing we don't synthesize) ---
    // Accept either `pos_embed` (from dump_magihuman_t5gemma_embeds.py — has
    // shape [1, target_length, 3584]) or `prompt_embeds` ([target_length, 3584]).
    //
    // CRITICAL: the dump script pads to t5_gemma_target_length=640 with zeros,
    // but the production pipeline TRIMS to the real text length before feeding
    // the model (`data_proxy.py:143`: `self.txt_feat = self.txt_feat[:self.txt_feat_len]`).
    // We must do the same — passing 627 zero-padded text tokens corrupts
    // attention with noise (output is pure noise even at 32 steps without this).
    let prompt_fix = flame_core::serialization::load_file(&cli.prompt_embeds_path, &device)?;
    let raw = prompt_fix
        .get("pos_embed")
        .or_else(|| prompt_fix.get("prompt_embeds"))
        .ok_or_else(|| anyhow!(
            "prompt_embeds_path missing key `pos_embed` or `prompt_embeds`"
        ))?
        .to_dtype(DType::F32)?;
    let mut prompt_embeds = match raw.shape().dims() {
        // [1, S, 3584] → squeeze batch
        [1, s, c] => raw.reshape(&[*s, *c])?,
        // [S, 3584] → already flat
        [_, _] => raw.clone(),
        other => anyhow::bail!("prompt embed shape must be [S,3584] or [1,S,3584], got {:?}", other),
    };
    // Trim to the original (pre-padding) text length if `pos_len` is present.
    if let Some(pos_len_t) = prompt_fix.get("pos_len") {
        let pos_len_vec = pos_len_t.to_dtype(DType::F32)?.to_vec_f32()?;
        let pos_len = pos_len_vec[0] as usize;
        if pos_len > 0 && pos_len < prompt_embeds.shape().dims()[0] {
            println!(
                "[load ] trimming padded text [{}] → real length [{}]",
                prompt_embeds.shape().dims()[0], pos_len
            );
            prompt_embeds = prompt_embeds.narrow(0, 0, pos_len)?.contiguous()?;
        }
    }
    println!("[load ] prompt_embeds shape={:?}", prompt_embeds.shape().dims());

    // --- Load uncond text embeds for CFG. Required when cfg_scale > 1.0 OR
    //     audio_cfg_scale > 1.0 OR sr_cfg_scale > 1.0. The embed file must
    //     contain `neg_embed` and `neg_len` (regenerate with
    //     dump_magihuman_t5gemma_embeds.py without --skip-negative).
    let cfg_active = cli.cfg_scale > 1.0 || cli.audio_cfg_scale > 1.0;
    let sr_cfg_active = cli.sr_cfg_scale > 1.0;
    let neg_embeds: Option<Tensor> = if cfg_active || sr_cfg_active {
        let raw_neg = prompt_fix.get("neg_embed").ok_or_else(|| anyhow!(
            "CFG enabled but `neg_embed` missing in {}; regenerate without --skip-negative",
            cli.prompt_embeds_path.display()
        ))?.to_dtype(DType::F32)?;
        let mut neg = match raw_neg.shape().dims() {
            [1, s, c] => raw_neg.reshape(&[*s, *c])?,
            [_, _] => raw_neg.clone(),
            other => anyhow::bail!("neg_embed shape must be [S,3584] or [1,S,3584], got {:?}", other),
        };
        if let Some(neg_len_t) = prompt_fix.get("neg_len") {
            let n = neg_len_t.to_dtype(DType::F32)?.to_vec_f32()?[0] as usize;
            if n > 0 && n < neg.shape().dims()[0] {
                println!("[load ] trimming padded neg [{}] → real [{}]", neg.shape().dims()[0], n);
                neg = neg.narrow(0, 0, n)?.contiguous()?;
            }
        }
        println!("[load ] neg_embeds shape={:?}", neg.shape().dims());
        Some(neg)
    } else {
        None
    };

    // --- Init noise (seeded for reproducibility) ---
    let mut video_lat = Tensor::randn_seeded(
        Shape::from_dims(&[1, Z_DIM, latent_t, latent_h, latent_w]),
        0.0,
        1.0,
        seed_for(cli.seed, 1, 0),
        device.clone(),
    )?
    .to_dtype(DType::F32)?;
    let mut audio_lat = if let Some(audio_path) = cli.audio_path.as_ref() {
        // Pure-Rust path: WAV → 51,200 Hz resample → SA Open VAE encode →
        // bottleneck-mean → [1, num_frames, 64].
        let lat = inference_flame::audio::load_and_encode_audio(
            audio_path,
            &cli.sa_vae_weights,
            num_frames,
            &device,
        )?;
        // Verify shape — load_and_encode_audio already does this but
        // belt-and-suspenders for the pack_inputs assertion.
        let d = lat.shape().dims();
        if d != [1, num_frames, AUDIO_IN_CHANNELS] {
            anyhow::bail!(
                "audio_lat shape {:?}, expected [1, {}, {}]",
                d,
                num_frames,
                AUDIO_IN_CHANNELS
            );
        }
        lat
    } else {
        Tensor::randn_seeded(
            Shape::from_dims(&[1, num_frames, AUDIO_IN_CHANNELS]),
            0.0,
            1.0,
            seed_for(cli.seed, 2, 0),
            device.clone(),
        )?
        .to_dtype(DType::F32)?
    };

    // --- i2v reference: encode image → first-frame conditioning latent.
    //     Done BEFORE the DiT load so the VAE encoder's weights are freed
    //     before we pin 28 GB for the transformer.
    let image_latent: Option<Tensor> = if let Some(path) = cli.image_path.as_ref() {
        let pixel_h = latent_h * VAE_STRIDE_HW;
        let pixel_w = latent_w * VAE_STRIDE_HW;
        let lat = encode_reference_image(path, &cli.wan_vae_weights, pixel_h, pixel_w, &device)?;
        // Wan2.2 VAE on a single-frame input produces T_lat=1; shape check.
        let d = lat.shape().dims().to_vec();
        if d != vec![1, Z_DIM, 1, latent_h, latent_w] {
            anyhow::bail!(
                "image latent shape mismatch: got {:?}, expected [1, {Z_DIM}, 1, {latent_h}, {latent_w}]",
                d
            );
        }
        Some(lat)
    } else {
        eprintln!(
            "[i2v  ] WARNING: no --image-path given. magi_human_distill is i2v-conditioned; \
             pure-T2V (all-noise latent) produces noise output regardless of prompt. \
             Pass --image-path <PNG/JPG> to enable image conditioning."
        );
        None
    };

    // MAGI_LOAD_BASE_LATENT=path overrides initial video_lat/audio_lat from a
    // saved fixture and sets effective_steps = 0 to skip the base sampling loop.
    // Used to iterate on the SR phase without re-running the 22-min base loop.
    // (Base DiT still loads, so peak RAM is unchanged. ~30s wasted per run.)
    let skip_base = std::env::var("MAGI_LOAD_BASE_LATENT").ok();
    if let Some(load_path) = skip_base.as_ref() {
        eprintln!("[base ] MAGI_LOAD_BASE_LATENT={load_path} — overriding init latents, skipping sampling loop");
        let tensors = flame_core::serialization::load_file(load_path, &device)?;
        video_lat = tensors
            .get("video_lat")
            .ok_or_else(|| anyhow!("MAGI_LOAD_BASE_LATENT: missing video_lat key"))?
            .to_dtype(DType::F32)?;
        audio_lat = tensors
            .get("audio_lat")
            .ok_or_else(|| anyhow!("MAGI_LOAD_BASE_LATENT: missing audio_lat key"))?
            .to_dtype(DType::F32)?;
        eprintln!(
            "[base ] loaded video_lat={:?} audio_lat={:?}",
            video_lat.shape().dims(),
            audio_lat.shape().dims()
        );
    }

    // --- Load full MagiHuman DiT (adapter + final heads on GPU, 40 transformer
    //     layers streamed via BlockOffloader). Internally opens the safetensors
    //     twice: once filtered for adapter/final_*, once for per-layer streaming.
    let mut model = MagiHumanDiTSwapped::load(
        cli.dit_weights.to_str().unwrap(),
        &device,
    )
    .map_err(|e| anyhow!("MagiHumanDiTSwapped load: {e}"))?;

    // --- Sigma schedule ---
    let sigmas = build_sigmas(cli.steps, cli.shift);
    println!("[sched] sigmas (high→0): {:?}", sigmas);

    // --- CFG: when active, base sampler switches from step_ddim to UniPC
    //     (matches Python video_scheduler.step path used when cfg_number=2).
    //     Audio gets its own UniPC instance only if audio_cfg_scale > 1.0.
    // MAGI_FORCE_DDIM=1 disables UniPC and uses step_ddim even with CFG. Used
    // to bisect whether visible artifacts at 32-step CFG come from UniPC drift.
    let force_ddim = std::env::var("MAGI_FORCE_DDIM").is_ok();
    if force_ddim {
        eprintln!("[sched] MAGI_FORCE_DDIM=1 — using step_ddim instead of UniPC");
    }
    let mut base_video_unipc: Option<UniPCSampler> = if cfg_active && !force_ddim {
        let sigmas_f64: Vec<f64> = sigmas.iter().map(|&s| s as f64).collect();
        Some(UniPCSampler::new(sigmas_f64, 2))
    } else {
        None
    };
    let mut base_audio_unipc: Option<UniPCSampler> = if cli.audio_cfg_scale > 1.0 && !force_ddim {
        let sigmas_f64: Vec<f64> = sigmas.iter().map(|&s| s as f64).collect();
        Some(UniPCSampler::new(sigmas_f64, 2))
    } else {
        None
    };

    // --- Sampling loop ---
    let effective_steps = if skip_base.is_some() { 0 } else { cli.steps };
    for step in 0..effective_steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t_step = std::time::Instant::now();
        println!("[step {step}] sigma {sigma_curr:.4} → {sigma_next:.4}");

        // i2v conditioning: replace the first temporal frame with the encoded
        // reference image latent BEFORE every forward pass. Mirrors
        // `latent_video[:, :, :1] = image_latent[:, :, :1]` in Wan2GP's
        // _run_diffusion_phase.
        if let Some(img_lat) = image_latent.as_ref() {
            video_lat = inject_image_latent(&video_lat, img_lat)?;
        }

        // Build packed input from current latents + prompt embeds.
        let packed = pack_inputs(&video_lat, &audio_lat, &prompt_embeds, &device)?;
        let v_count = packed.group_sizes[0];
        let a_count = packed.group_sizes[1];
        let (lt, lh, lw) = packed.video_grid;

        // === Conditional DiT forward ===
        // Returns [L, max(VIDEO_IN_CHANNELS, AUDIO_IN_CHANNELS)] F32.
        // - rows 0..v_count           → video velocity (192 channels valid)
        // - rows v_count..v_count+a_count → audio velocity (first 64 channels valid)
        // - rows v_count+a_count..L   → text rows (zero, discard)
        let dit_out_cond = model
            .forward(&packed.x, &packed.coords, &packed.group_sizes)
            .map_err(|e| anyhow!("DiT forward (cond): {e}"))?;

        // Optional parity dump: at step 0, collect input + cond + uncond + mixed
        // velocities into a single safetensors file so a Python script can
        // recompute the mix and verify Rust's CFG implementation bit-for-bit.
        // Built incrementally across cond and uncond/mix branches; saved at end
        // of the loop iteration.
        let mut step0_dump: Option<HashMap<String, Tensor>> =
            if step == 0 && std::env::var("MAGI_DUMP_STEP0").is_ok() {
                Some(HashMap::new())
            } else {
                None
            };
        if let Some(d) = step0_dump.as_mut() {
            d.insert("x".to_string(), packed.x.to_dtype(DType::F32)?);
            d.insert("coords".to_string(), packed.coords.to_dtype(DType::F32)?);
            d.insert(
                "group_sizes".to_string(),
                Tensor::from_vec(
                    packed.group_sizes.iter().map(|&s| s as f32).collect(),
                    Shape::from_dims(&[3]),
                    device.clone(),
                )?,
            );
            d.insert(
                "video_grid".to_string(),
                Tensor::from_vec(
                    vec![lt as f32, lh as f32, lw as f32],
                    Shape::from_dims(&[3]),
                    device.clone(),
                )?,
            );
            d.insert(
                "video_lat_pre_patchify".to_string(),
                video_lat.to_dtype(DType::F32)?,
            );
            d.insert("audio_lat".to_string(), audio_lat.to_dtype(DType::F32)?);
            d.insert("prompt_embeds".to_string(), prompt_embeds.to_dtype(DType::F32)?);
            d.insert("dit_out".to_string(), dit_out_cond.to_dtype(DType::F32)?);
            let v_vel_dump = unpatchify_video(&dit_out_cond.narrow(0, 0, v_count)?, lt, lh, lw)?;
            d.insert("v_velocity_unpatch".to_string(), v_vel_dump);
            let a_vel_dump = dit_out_cond
                .narrow(0, v_count, a_count)?
                .narrow(1, 0, AUDIO_IN_CHANNELS)?
                .reshape(&[1, a_count, AUDIO_IN_CHANNELS])?;
            d.insert("a_velocity_unpatch".to_string(), a_vel_dump);
            d.insert(
                "sigma_curr".to_string(),
                Tensor::from_vec(vec![sigma_curr], Shape::from_dims(&[1]), device.clone())?,
            );
        }

        let (v_vel_cond, a_vel_cond) =
            extract_velocities(&dit_out_cond, v_count, a_count, lt, lh, lw)?;
        drop(dit_out_cond);

        // === Unconditional DiT forward (CFG only) ===
        // Re-pack with neg embeds (different text token count after trim) and
        // run the model again. v_count and a_count are unchanged — only the
        // text rows differ.
        let (v_velocity, a_velocity) = if cfg_active {
            let neg = neg_embeds.as_ref().expect("cfg_active implies neg_embeds loaded");
            let packed_uncond = pack_inputs(&video_lat, &audio_lat, neg, &device)?;
            let dit_out_uncond = model
                .forward(&packed_uncond.x, &packed_uncond.coords, &packed_uncond.group_sizes)
                .map_err(|e| anyhow!("DiT forward (uncond): {e}"))?;

            // Dump uncond raw output BEFORE drop. Also dump the uncond pack so
            // a Python script can re-run the SAME inputs through the reference
            // model and bit-compare to dit_out_uncond.
            if let Some(d) = step0_dump.as_mut() {
                d.insert("dit_out_uncond".to_string(), dit_out_uncond.to_dtype(DType::F32)?);
                d.insert("x_uncond".to_string(), packed_uncond.x.to_dtype(DType::F32)?);
                d.insert("coords_uncond".to_string(), packed_uncond.coords.to_dtype(DType::F32)?);
                d.insert(
                    "group_sizes_uncond".to_string(),
                    Tensor::from_vec(
                        packed_uncond.group_sizes.iter().map(|&s| s as f32).collect(),
                        Shape::from_dims(&[3]),
                        device.clone(),
                    )?,
                );
                d.insert("prompt_embeds_uncond".to_string(), neg.to_dtype(DType::F32)?);
            }

            let (v_vel_uncond, a_vel_uncond) =
                extract_velocities(&dit_out_uncond, v_count, a_count, lt, lh, lw)?;
            drop(dit_out_uncond);

            // Mix per modality. Python's video schedule: scale = cfg_scale when
            // t > 500 (timestep, 0..1000), else 2.0. Per scheduler_unipc.py:203,
            // `timesteps = sigmas * 1000` where `sigmas` is the post-shift schedule,
            // so `t > 500` is EXACTLY `sigma_curr > 0.5` for our (already-shifted)
            // sigma_curr. The 2.0 floor is critical: late-step CFG above this
            // over-saturates fine-tuned detail.
            let video_scale = if sigma_curr > 0.5 { cli.cfg_scale as f32 } else { 2.0 };
            let audio_scale = cli.audio_cfg_scale as f32;
            eprintln!("[step {step}] cfg video={video_scale:.2}  audio={audio_scale:.2}");

            // pred = uncond + scale * (cond - uncond)
            let v_diff = v_vel_cond.sub(&v_vel_uncond)?;
            let v_mixed = v_vel_uncond.add(&v_diff.mul_scalar(video_scale)?)?;
            let a_diff = a_vel_cond.sub(&a_vel_uncond)?;
            let a_mixed = a_vel_uncond.add(&a_diff.mul_scalar(audio_scale)?)?;

            // Dump v_uncond/a_uncond + mixed + scales for parity verification.
            if let Some(d) = step0_dump.as_mut() {
                d.insert("v_velocity_uncond".to_string(), v_vel_uncond.to_dtype(DType::F32)?);
                d.insert("a_velocity_uncond".to_string(), a_vel_uncond.to_dtype(DType::F32)?);
                d.insert("v_velocity_mixed".to_string(), v_mixed.to_dtype(DType::F32)?);
                d.insert("a_velocity_mixed".to_string(), a_mixed.to_dtype(DType::F32)?);
                d.insert(
                    "cfg_video_scale".to_string(),
                    Tensor::from_vec(vec![video_scale], Shape::from_dims(&[1]), device.clone())?,
                );
                d.insert(
                    "cfg_audio_scale".to_string(),
                    Tensor::from_vec(vec![audio_scale], Shape::from_dims(&[1]), device.clone())?,
                );
            }

            (v_mixed, a_mixed)
        } else {
            (v_vel_cond, a_vel_cond)
        };

        // Save the step-0 dump (if collected) once both passes have populated it.
        if let Some(d) = step0_dump.take() {
            let dump_path = std::env::var("MAGI_DUMP_STEP0").unwrap();
            let key_list: Vec<&str> = d.keys().map(|s| s.as_str()).collect();
            serialization::save_file(&d, &dump_path)?;
            eprintln!("[parity] dumped step-0 fixture → {dump_path}");
            eprintln!("[parity] keys ({}): {:?}", key_list.len(), key_list);
        }

        // === Sampler step ===
        // UniPC when CFG active (matches Python video_scheduler.step path);
        // otherwise step_ddim with seeded noise (legacy distill path).
        if let Some(unipc) = base_video_unipc.as_mut() {
            video_lat = unipc.step(&v_velocity, &video_lat)?;
        } else {
            let noise_v = Tensor::randn_seeded(
                video_lat.shape().clone(),
                0.0,
                1.0,
                seed_for(cli.seed, 3, step as u64),
                device.clone(),
            )?
            .to_dtype(DType::F32)?;
            video_lat = step_ddim(&v_velocity, &video_lat, sigma_curr, sigma_next, &noise_v)?;
        }
        if let Some(unipc) = base_audio_unipc.as_mut() {
            audio_lat = unipc.step(&a_velocity, &audio_lat)?;
        } else {
            let noise_a = Tensor::randn_seeded(
                audio_lat.shape().clone(),
                0.0,
                1.0,
                seed_for(cli.seed, 4, step as u64),
                device.clone(),
            )?
            .to_dtype(DType::F32)?;
            audio_lat = step_ddim(&a_velocity, &audio_lat, sigma_curr, sigma_next, &noise_a)?;
        }

        // Sanity stats. v_max_abs of state should shrink as denoising. vel_max
        // and vel_mean tell us if the model is actually producing meaningful
        // velocity, or just outputting zero (in which case state stays at noise).
        let v_max = video_lat.to_vec_f32()?.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let vel_vec = v_velocity.to_vec_f32()?;
        let vel_max = vel_vec.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let vel_mean: f32 = vel_vec.iter().map(|x| x.abs()).sum::<f32>() / vel_vec.len() as f32;
        eprintln!(
            "[step {step}] {} ms  v_max={v_max:.3}  vel_max={vel_max:.3}  vel_mean_abs={vel_mean:.4}",
            t_step.elapsed().as_millis()
        );

        clear_pool_cache();
    }

    // Final i2v frame injection (matches the post-loop line in Python).
    if let Some(img_lat) = image_latent.as_ref() {
        video_lat = inject_image_latent(&video_lat, img_lat)?;
    }

    // Optional final-latent dump for off-line decoder parity tests.
    if let Ok(dump_path) = std::env::var("MAGI_DUMP_FINAL_LATENT") {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("video_lat".to_string(), video_lat.to_dtype(DType::F32)?);
        tensors.insert("audio_lat".to_string(), audio_lat.to_dtype(DType::F32)?);
        serialization::save_file(&tensors, &dump_path)?;
        eprintln!("[parity] dumped final video_lat + audio_lat → {dump_path}");
    }

    println!(
        "[done ] final video_lat={:?} audio_lat={:?}",
        video_lat.shape().dims(),
        audio_lat.shape().dims()
    );

    // --- Drop the BASE DiT before loading anything else.
    drop(model);
    clear_pool_cache();

    // --- SR phase 2 (transformer2) ---
    // Refine 256p base latent → 1080p via the SR DiT. Only runs if the user
    // passed --sr-weights. Mirrors Wan2GP's _run_diffusion_phase(use_sr_model=True).
    let (final_video_lat, final_audio_lat) = if let Some(sr_path) = cli.sr_weights.as_ref() {
        eprintln!("\n[sr   ] running SR phase 2: {}x{} → {}x{}",
            cli.width, cli.height, cli.sr_width, cli.sr_height);

        let sr_latent_h = (cli.sr_height / VAE_STRIDE_HW / HW_PATCH) * HW_PATCH;
        let sr_latent_w = (cli.sr_width / VAE_STRIDE_HW / HW_PATCH) * HW_PATCH;
        let sr_pixel_h = sr_latent_h * VAE_STRIDE_HW;
        let sr_pixel_w = sr_latent_w * VAE_STRIDE_HW;
        eprintln!("[sr   ] SR latent dims: lt={latent_t} lh={sr_latent_h} lw={sr_latent_w}");

        // Re-encode reference image at full SR resolution. Required by Wan2GP
        // line 734: sr_image_latent = self._encode_image_latent(image_batch, height, width, ...).
        let sr_image_latent = if let Some(img_path) = cli.image_path.as_ref() {
            let lat = encode_reference_image(
                img_path,
                &cli.wan_vae_weights,
                sr_pixel_h,
                sr_pixel_w,
                &device,
            )?;
            let d = lat.shape().dims().to_vec();
            if d != vec![1, Z_DIM, 1, sr_latent_h, sr_latent_w] {
                anyhow::bail!(
                    "SR image_latent shape mismatch: got {:?}, expected [1, {Z_DIM}, 1, {sr_latent_h}, {sr_latent_w}]",
                    d
                );
            }
            lat
        } else {
            anyhow::bail!("--sr-weights given but --image-path missing; SR requires reference image");
        };

        // Trilinear-upsample base latent to SR latent dims (HW only — T preserved).
        let sr_video_lat = bilinear_upsample_3d_spatial(&video_lat, sr_latent_h, sr_latent_w)?;
        eprintln!("[sr   ] upsampled base latent: {:?}", sr_video_lat.shape().dims());

        // Add ZeroSNR-derived noise: lat = lat * sigma + randn * sqrt(1 - sigma²).
        // Matches Wan2GP line 738..740.
        let noise_sigmas = zero_snr_sigmas(1000);
        let sr_sigma = noise_sigmas[cli.sr_noise_value] as f32;
        eprintln!("[sr   ] sr_noise_value={} → sigma={sr_sigma:.4}", cli.sr_noise_value);
        let mut sr_video_lat = if cli.sr_noise_value > 0 {
            let n = Tensor::randn_seeded(
                sr_video_lat.shape().clone(),
                0.0,
                1.0,
                seed_for(cli.seed, 5, 0),
                device.clone(),
            )?
            .to_dtype(DType::F32)?;
            let scale_n = (1.0_f32 - sr_sigma * sr_sigma).sqrt();
            sr_video_lat
                .mul_scalar(sr_sigma)?
                .add(&n.mul_scalar(scale_n)?)?
        } else {
            sr_video_lat
        };

        // Audio noise blend (Wan2GP line 741):
        //   sr_audio = randn * sr_audio_noise_scale + audio * (1 - sr_audio_noise_scale)
        let sr_audio_noise_scale_f = cli.sr_audio_noise_scale as f32;
        let n_a = Tensor::randn_seeded(
            audio_lat.shape().clone(),
            0.0,
            1.0,
            seed_for(cli.seed, 6, 0),
            device.clone(),
        )?
        .to_dtype(DType::F32)?;
        let mut sr_audio_lat = n_a
            .mul_scalar(sr_audio_noise_scale_f)?
            .add(&audio_lat.mul_scalar(1.0 - sr_audio_noise_scale_f)?)?;

        // Load SR DiT (transformer2 — different weight layout: split linears
        // and per-modality MM linears, see `models/magihuman_sr_dit.rs`).
        eprintln!("[sr   ] loading SR DiT from {}", sr_path.display());
        let mut sr_model = MagiHumanSrDiTSwapped::load(
            sr_path.to_str().unwrap(),
            &device,
        )
        .map_err(|e| anyhow!("SR DiT load: {e}"))?;

        // SR sampling loop. Same shift / build_sigmas as base, fewer steps,
        // v1 coords, no audio update (update_audio=False per Python line 763).
        // Uses UniPC bh2 (predict_x0, flow_prediction) — Python uses scheduler.step()
        // which is the full multistep predictor-corrector. solver_order=2.
        let sr_sigmas = build_sigmas(cli.sr_steps, cli.shift);
        eprintln!("[sr   ] sigmas: {:?}", sr_sigmas);
        let sigmas_f64: Vec<f64> = sr_sigmas.iter().map(|&s| s as f64).collect();
        let mut unipc = UniPCSampler::new(sigmas_f64, 2);

        for step in 0..cli.sr_steps {
            let sigma_curr = sr_sigmas[step];
            let sigma_next = sr_sigmas[step + 1];
            let t_step = std::time::Instant::now();
            println!("[sr step {step}] sigma {sigma_curr:.4} → {sigma_next:.4}");

            // Inject SR image_latent at frame 0 BEFORE forward.
            sr_video_lat = inject_image_latent(&sr_video_lat, &sr_image_latent)?;

            // Pack with v1 coords.
            let packed = pack_inputs_v1(&sr_video_lat, &sr_audio_lat, &prompt_embeds, &device)?;
            let v_count = packed.group_sizes[0];
            let a_count = packed.group_sizes[1];
            let (lt2, lh2, lw2) = packed.video_grid;

            // === Conditional SR DiT forward ===
            let dit_out_cond = sr_model
                .forward(&packed.x, &packed.coords, &packed.group_sizes)
                .map_err(|e| anyhow!("SR DiT forward (cond): {e}"))?;

            // Optional parity dump: at SR step 0, save the exact input + output of
            // the conditional SR DiT forward so a Python re-run can localize any
            // divergence between the per-modality split-linear forward and the
            // Python reference. Always dumps the cond pass.
            // Companion script: scripts/magihuman_sr_dit_parity_from_rust.py.
            if step == 0 {
                if let Ok(dump_path) = std::env::var("MAGI_DUMP_SR_STEP0") {
                    let mut tensors: HashMap<String, Tensor> = HashMap::new();
                    tensors.insert("x".to_string(), packed.x.to_dtype(DType::F32)?);
                    tensors.insert("coords".to_string(), packed.coords.to_dtype(DType::F32)?);
                    tensors.insert(
                        "group_sizes".to_string(),
                        Tensor::from_vec(
                            packed.group_sizes.iter().map(|&s| s as f32).collect(),
                            Shape::from_dims(&[3]),
                            device.clone(),
                        )?,
                    );
                    tensors.insert(
                        "video_grid".to_string(),
                        Tensor::from_vec(
                            vec![lt2 as f32, lh2 as f32, lw2 as f32],
                            Shape::from_dims(&[3]),
                            device.clone(),
                        )?,
                    );
                    tensors.insert("dit_out".to_string(), dit_out_cond.to_dtype(DType::F32)?);
                    serialization::save_file(&tensors, &dump_path)?;
                    eprintln!("[sr parity] dumped SR step-0 fixture → {dump_path}");
                    eprintln!("[sr parity] keys: x coords group_sizes video_grid dit_out");
                }
            }

            let (v_vel_cond, _a_vel_cond) =
                extract_velocities(&dit_out_cond, v_count, a_count, lt2, lh2, lw2)?;
            drop(dit_out_cond);

            // === Unconditional SR pass (CFG only) ===
            // SR uses per-latent-frame CFG with `cfg_trick`: first
            // `sr_cfg_trick_frame` frames use `sr_cfg_trick_value`, rest use
            // `sr_cfg_scale`. Mirrors creator video_generate.py:411..418.
            // Audio NOT mixed — SR phase doesn't denoise audio
            // (update_audio=False per video_generate.py:763).
            let v_velocity = if sr_cfg_active {
                let neg = neg_embeds.as_ref().expect("sr_cfg_active implies neg_embeds loaded");
                let packed_uncond = pack_inputs_v1(&sr_video_lat, &sr_audio_lat, neg, &device)?;
                let dit_out_uncond = sr_model
                    .forward(&packed_uncond.x, &packed_uncond.coords, &packed_uncond.group_sizes)
                    .map_err(|e| anyhow!("SR DiT forward (uncond): {e}"))?;
                let (v_vel_uncond, _) =
                    extract_velocities(&dit_out_uncond, v_count, a_count, lt2, lh2, lw2)?;
                drop(dit_out_uncond);
                let v_diff = v_vel_cond.sub(&v_vel_uncond)?;
                let trick_t = std::cmp::min(cli.sr_cfg_trick_frame, lt2);
                let scale_main = cli.sr_cfg_scale as f32;
                let scale_trick = cli.sr_cfg_trick_value as f32;
                let scaled_diff = if trick_t == 0 || trick_t >= lt2 {
                    // No split: all frames use the same scale.
                    let s = if trick_t >= lt2 { scale_trick } else { scale_main };
                    eprintln!("[sr step {step}] cfg video={s:.2} (uniform, frames=0..{lt2})");
                    v_diff.mul_scalar(s)?
                } else {
                    // Split: frames [0..trick_t] use scale_trick, [trick_t..lt2] use scale_main.
                    eprintln!(
                        "[sr step {step}] cfg video={scale_trick:.2} for frames 0..{trick_t}, {scale_main:.2} for {trick_t}..{lt2}",
                    );
                    let early = v_diff
                        .narrow(2, 0, trick_t)?
                        .contiguous()?
                        .mul_scalar(scale_trick)?;
                    let late = v_diff
                        .narrow(2, trick_t, lt2 - trick_t)?
                        .contiguous()?
                        .mul_scalar(scale_main)?;
                    Tensor::cat(&[&early, &late], 2)?
                };
                v_vel_uncond.add(&scaled_diff)?
            } else {
                v_vel_cond
            };

            // Optional per-step UniPC dump for parity bisection.
            // MAGI_DUMP_SR_UNIPC=path → save sample_in, velocity, sample_out per step.
            let unipc_dump_path = std::env::var("MAGI_DUMP_SR_UNIPC").ok();
            let sr_lat_in = sr_video_lat.clone();

            // UniPC step: corrector + predictor.
            // MAGI_FORCE_DDIM=1 also disables SR UniPC and uses step_ddim, for
            // bisecting whether SR UniPC trajectory is what introduces the
            // chromatic-streak artifact. step_ddim needs noise; seed mirrors
            // the base-step seeding scheme using a unique role (7).
            sr_video_lat = if force_ddim {
                let noise = Tensor::randn_seeded(
                    sr_video_lat.shape().clone(),
                    0.0,
                    1.0,
                    seed_for(cli.seed, 7, step as u64),
                    device.clone(),
                )?
                .to_dtype(DType::F32)?;
                step_ddim(&v_velocity, &sr_video_lat, sigma_curr, sigma_next, &noise)?
            } else {
                unipc.step(&v_velocity, &sr_video_lat)?
            };

            if let Some(ref path) = unipc_dump_path {
                let mut tensors: HashMap<String, Tensor> = HashMap::new();
                tensors.insert(format!("step_{step}_sample_in"),  sr_lat_in.to_dtype(DType::F32)?);
                tensors.insert(format!("step_{step}_velocity"),   v_velocity.to_dtype(DType::F32)?);
                tensors.insert(format!("step_{step}_sample_out"), sr_video_lat.to_dtype(DType::F32)?);
                tensors.insert(
                    format!("step_{step}_sigmas"),
                    Tensor::from_vec(
                        vec![sigma_curr, sigma_next],
                        Shape::from_dims(&[2]),
                        device.clone(),
                    )?,
                );
                if step == 0 {
                    // Dump full sigma schedule once
                    let all_sigmas: Vec<f32> = unipc.sigmas.iter().map(|x| *x as f32).collect();
                    let n = all_sigmas.len();
                    tensors.insert(
                        "all_sigmas".to_string(),
                        Tensor::from_vec(all_sigmas, Shape::from_dims(&[n]), device.clone())?,
                    );
                }
                // Append-or-create: load existing dump if present, merge, save.
                if std::path::Path::new(path).exists() {
                    if let Ok(existing) = flame_core::serialization::load_file(
                        std::path::Path::new(path),
                        &device,
                    ) {
                        for (k, v) in existing {
                            tensors.entry(k).or_insert(v);
                        }
                    }
                }
                flame_core::serialization::save_file(&tensors, path)?;
                eprintln!("[sr unipc dump] step {step} → {}", path);
            }
            // audio_lat NOT updated in SR phase (matches update_audio=False)

            let v_max = sr_video_lat.to_vec_f32()?.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let vel_vec = v_velocity.to_vec_f32()?;
            let vel_max = vel_vec.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let vel_mean: f32 = vel_vec.iter().map(|x| x.abs()).sum::<f32>() / vel_vec.len() as f32;
            eprintln!(
                "[sr step {step}] {} ms  v_max={v_max:.3}  vel_max={vel_max:.3}  vel_mean_abs={vel_mean:.4}",
                t_step.elapsed().as_millis()
            );

            clear_pool_cache();
        }

        // Final injection.
        sr_video_lat = inject_image_latent(&sr_video_lat, &sr_image_latent)?;

        eprintln!(
            "[sr   ] final SR video_lat={:?}  audio_lat={:?}",
            sr_video_lat.shape().dims(),
            sr_audio_lat.shape().dims()
        );

        drop(sr_model);
        clear_pool_cache();

        (sr_video_lat, sr_audio_lat)
    } else {
        (video_lat, audio_lat)
    };

    let video_lat = final_video_lat;
    let audio_lat = final_audio_lat;

    // Save post-SR (or post-base if SR skipped) latent before VAE decode.
    // Allows recovery if TurboVAED decode crashes after a long base run —
    // the latent represents the costly part; decode can be re-run cheaply
    // from the saved fixture.
    if let Ok(dump_path) = std::env::var("MAGI_DUMP_PRE_DECODE_LATENT") {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("video_lat".to_string(), video_lat.to_dtype(DType::F32)?);
        tensors.insert("audio_lat".to_string(), audio_lat.to_dtype(DType::F32)?);
        serialization::save_file(&tensors, &dump_path)?;
        eprintln!("[parity] dumped pre-decode latent → {dump_path}");
    }

    // --- TurboVAED decode: [1, 48, T_lat, H_lat, W_lat] → [1, 3, T_pix, H_pix, W_pix] BF16 in [-1, 1]
    println!("[decode] loading TurboVAED from {}", cli.turbo_vaed_weights.display());
    let vae_cfg = TurboVaedConfig::shipped_default();
    let turbo_vae = TurboVAED::load(
        cli.turbo_vaed_weights.to_str().unwrap(),
        &vae_cfg,
        &device,
    )
    .map_err(|e| anyhow!("TurboVAED load: {e}"))?;
    let t_dec = std::time::Instant::now();
    let video = turbo_vae
        .decode(&video_lat.to_dtype(DType::BF16)?)
        .map_err(|e| anyhow!("TurboVAED decode: {e}"))?;
    eprintln!("[decode] video {} ms shape={:?}", t_dec.elapsed().as_millis(), video.shape().dims());
    drop(turbo_vae);
    clear_pool_cache();

    let v_dims = video.shape().dims().to_vec();
    if v_dims.len() != 5 || v_dims[0] != 1 || v_dims[1] != 3 {
        anyhow::bail!("TurboVAED decode returned unexpected shape: {:?}", v_dims);
    }
    let frame_count = v_dims[2];
    let pix_h = v_dims[3];
    let pix_w = v_dims[4];
    let video_f32 = video.to_dtype(DType::F32)?.to_vec_f32()?;
    drop(video);
    let frames_u8 = video_tensor_to_rgb_u8(&video_f32, frame_count, pix_h, pix_w);
    drop(video_f32);

    // --- SA Oobleck decode: pipeline stores audio_lat as [1, T_lat, 64] but
    //     OobleckDecoder takes [B, 64, T_lat]. Permute, decode, then output is
    //     [1, 2, T_lat * 2048] F32 raw stereo at (fps * 2048) Hz native.
    println!("[decode] loading Oobleck from {}", cli.sa_vae_weights.display());
    let oobleck = OobleckDecoder::load_default(
        cli.sa_vae_weights.to_str().unwrap(),
        &device,
    )
    .map_err(|e| anyhow!("Oobleck load: {e}"))?;
    let audio_z = audio_lat
        .permute(&[0, 2, 1])?
        .contiguous()?; // [1, 64, num_frames]
    let t_aud = std::time::Instant::now();
    let audio = oobleck
        .decode(&audio_z)
        .map_err(|e| anyhow!("Oobleck decode: {e}"))?;
    eprintln!("[decode] audio {} ms shape={:?}", t_aud.elapsed().as_millis(), audio.shape().dims());
    drop(oobleck);
    clear_pool_cache();

    let a_dims = audio.shape().dims().to_vec();
    if a_dims.len() != 3 || a_dims[0] != 1 || a_dims[1] != 2 {
        anyhow::bail!("Oobleck decode returned unexpected shape: {:?}", a_dims);
    }
    let n_samples = a_dims[2];
    let audio_f32 = audio.to_vec_f32()?;
    drop(audio);
    let pcm = audio_tensor_to_pcm_i16(&audio_f32, 2, n_samples, true);
    drop(audio_f32);

    // Native sample rate from Oobleck: strides [2,4,4,8,8] → 2048× upsample,
    // and the pipeline keeps 1 latent frame per video frame (latent rate = fps).
    let audio_sr: u32 = (cli.fps * 2048) as u32;
    eprintln!(
        "[mux  ] frames={frame_count} {pix_w}x{pix_h} @ {} fps  audio={n_samples} samples @ {audio_sr} Hz → {}",
        cli.fps, cli.out.display()
    );
    write_mp4(
        &cli.out,
        &frames_u8,
        frame_count,
        pix_w,
        pix_h,
        cli.fps as f32,
        &pcm,
        audio_sr,
    )
    .map_err(|e| anyhow!("write_mp4: {e}"))?;
    println!("[done ] wrote {}", cli.out.display());

    Ok(())
}
