//! Decode-only recovery bin for daVinci-MagiHuman.
//!
//! Loads a saved (video_lat, audio_lat) safetensors fixture (produced by
//! magihuman_infer with MAGI_DUMP_PRE_DECODE_LATENT) and runs only the
//! TurboVAED video decode + Oobleck audio decode + MP4 mux. Useful when a
//! long base+SR run completes successfully but the inline decode crashes
//! due to accumulated CUDA allocator state — re-running the full pipeline
//! costs 90 minutes; this bin recovers in ~1 minute from a fresh process.
//!
//! Usage:
//!     magihuman_decode --latent <pre_decode.safetensors> --out <out.mp4>
//!     [--turbo-vaed-weights <path>] [--sa-vae-weights <path>] [--fps 25]

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use flame_core::{cuda_alloc_pool::clear_pool_cache, cuda_ops::GpuOps, global_cuda_device, serialization, DType, Tensor};
use inference_flame::models::sa_audio_vae::OobleckDecoder;
use inference_flame::models::turbo_vaed::{TurboVaedConfig, TurboVAED};
use inference_flame::mux::{audio_tensor_to_pcm_i16, video_tensor_to_rgb_u8, write_mp4};

struct Cli {
    latent: PathBuf,
    out: PathBuf,
    turbo_vaed_weights: PathBuf,
    sa_vae_weights: PathBuf,
    fps: usize,
    upsample_h: Option<usize>,
    upsample_w: Option<usize>,
    chunk_latent_frames: Option<usize>,
}

impl Cli {
    fn parse() -> Result<Self> {
        let mut latent: Option<PathBuf> = None;
        let mut out = PathBuf::from("magihuman_decode_out.mp4");
        let mut turbo_vaed_weights = PathBuf::from(
            "/home/alex/.serenity/models/vaes/magihuman_turbo_vaed_decoder.safetensors",
        );
        let mut sa_vae_weights = PathBuf::from(
            "/home/alex/.serenity/models/vaes/stable_audio_oobleck_vae.safetensors",
        );
        let mut fps = 25;
        let mut upsample_h: Option<usize> = None;
        let mut upsample_w: Option<usize> = None;
        let mut chunk_latent_frames: Option<usize> = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--latent" => latent = it.next().map(PathBuf::from),
                "--out" => out = it.next().unwrap().into(),
                "--turbo-vaed-weights" => turbo_vaed_weights = it.next().unwrap().into(),
                "--sa-vae-weights" => sa_vae_weights = it.next().unwrap().into(),
                "--fps" => fps = it.next().unwrap().parse()?,
                "--upsample-h" => upsample_h = Some(it.next().unwrap().parse()?),
                "--upsample-w" => upsample_w = Some(it.next().unwrap().parse()?),
                "--chunk-latent-frames" => {
                    chunk_latent_frames = Some(it.next().unwrap().parse()?)
                }
                other => anyhow::bail!("unknown arg: {other}"),
            }
        }
        Ok(Self {
            latent: latent.ok_or_else(|| anyhow!("--latent required"))?,
            out,
            turbo_vaed_weights,
            sa_vae_weights,
            fps,
            upsample_h,
            upsample_w,
            chunk_latent_frames,
        })
    }
}

/// Bilinear-upsample a 5D (B, C, T, H, W) tensor in HW only, T preserved.
fn bilinear_upsample_3d_spatial(x: &Tensor, new_h: usize, new_w: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 5 {
        anyhow::bail!("bilinear_upsample_3d_spatial: expected 5D, got {:?}", dims);
    }
    let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let r = x.permute(&[0, 2, 1, 3, 4])?.contiguous()?;
    let r = r.reshape(&[b * t, c, h, w])?;
    let up = GpuOps::upsample2d_bilinear(&r, (new_h, new_w), true)?;
    let r = up.reshape(&[b, t, c, new_h, new_w])?;
    r.permute(&[0, 2, 1, 3, 4])?
        .contiguous()
        .map_err(|e| anyhow!("{e:?}"))
}

fn main() -> Result<()> {
    let cli = Cli::parse()?;
    let device = global_cuda_device();
    println!("[load ] latent fixture: {}", cli.latent.display());
    let tensors = serialization::load_file(&cli.latent, &device)?;
    let mut video_lat = tensors
        .get("video_lat")
        .ok_or_else(|| anyhow!("latent missing key `video_lat`"))?
        .to_dtype(DType::F32)?;
    let audio_lat = tensors
        .get("audio_lat")
        .ok_or_else(|| anyhow!("latent missing key `audio_lat`"))?
        .to_dtype(DType::F32)?;
    println!(
        "[load ] video_lat={:?}  audio_lat={:?}",
        video_lat.shape().dims(),
        audio_lat.shape().dims()
    );

    if let (Some(uh), Some(uw)) = (cli.upsample_h, cli.upsample_w) {
        eprintln!("[upsample] bilinear-upsampling video_lat HW → ({uh}, {uw})");
        video_lat = bilinear_upsample_3d_spatial(&video_lat, uh, uw)?;
        let v = video_lat.to_vec_f32()?;
        let abs_max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        eprintln!(
            "[upsample] new shape={:?}  abs_max={abs_max:.4}  mean={mean:.4}  std={:.4}",
            video_lat.shape().dims(),
            var.sqrt()
        );
    }

    // --- TurboVAED video decode ---
    println!(
        "[decode] loading TurboVAED from {}",
        cli.turbo_vaed_weights.display()
    );
    let vae_cfg = TurboVaedConfig::shipped_default();
    let turbo_vae = TurboVAED::load(
        cli.turbo_vaed_weights.to_str().unwrap(),
        &vae_cfg,
        &device,
    )
    .map_err(|e| anyhow!("TurboVAED load: {e}"))?;
    let t_dec = std::time::Instant::now();
    let video_bf16 = video_lat.to_dtype(DType::BF16)?;
    let total_lat_t = video_bf16.shape().dims()[2];

    // Chunked decode: TurboVAED's sliding-window path crashes on long latents
    // (≥26 latent frames; see turbo_vaed.rs:878 KNOWN ISSUE comment). Workaround
    // is to slice the latent into ≤chunk-latent-frames-sized pieces and decode
    // each as an independent first-chunk decode, then concatenate the pixel
    // outputs. Loses the cross-chunk temporal overlap blending but produces
    // a usable video at long durations.
    let video = if let Some(chunk_lt) = cli.chunk_latent_frames {
        if chunk_lt == 0 {
            anyhow::bail!("--chunk-latent-frames must be > 0");
        }
        eprintln!(
            "[decode] chunked path: {total_lat_t} latent frames → chunks of {chunk_lt}"
        );
        let mut pixel_chunks: Vec<flame_core::Tensor> = Vec::new();
        let mut start = 0usize;
        let mut chunk_idx = 0usize;
        while start < total_lat_t {
            let take = (total_lat_t - start).min(chunk_lt);
            let z_chunk = video_bf16.narrow(2, start, take)?;
            let t_chunk = std::time::Instant::now();
            let pix_chunk = turbo_vae
                .decode(&z_chunk)
                .map_err(|e| anyhow!("TurboVAED decode chunk {chunk_idx} (lt {start}..{}): {e}", start + take))?;
            eprintln!(
                "[decode] chunk {chunk_idx} lt {start}..{} → pix {:?} in {} ms",
                start + take,
                pix_chunk.shape().dims(),
                t_chunk.elapsed().as_millis()
            );
            pixel_chunks.push(pix_chunk);
            start += take;
            chunk_idx += 1;
            clear_pool_cache();
        }
        let chunk_refs: Vec<&flame_core::Tensor> = pixel_chunks.iter().collect();
        flame_core::Tensor::cat(&chunk_refs, 2)?
    } else {
        turbo_vae
            .decode(&video_bf16)
            .map_err(|e| anyhow!("TurboVAED decode: {e}"))?
    };
    eprintln!(
        "[decode] video {} ms shape={:?}",
        t_dec.elapsed().as_millis(),
        video.shape().dims()
    );
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

    // --- Oobleck audio decode ---
    println!(
        "[decode] loading Oobleck from {}",
        cli.sa_vae_weights.display()
    );
    let oobleck = OobleckDecoder::load_default(cli.sa_vae_weights.to_str().unwrap(), &device)
        .map_err(|e| anyhow!("Oobleck load: {e}"))?;
    let audio_z = audio_lat.permute(&[0, 2, 1])?.contiguous()?;
    let t_aud = std::time::Instant::now();
    let audio = oobleck
        .decode(&audio_z)
        .map_err(|e| anyhow!("Oobleck decode: {e}"))?;
    eprintln!(
        "[decode] audio {} ms shape={:?}",
        t_aud.elapsed().as_millis(),
        audio.shape().dims()
    );
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

    let audio_sr: u32 = (cli.fps * 2048) as u32;
    eprintln!(
        "[mux  ] frames={frame_count} {pix_w}x{pix_h} @ {} fps  audio={n_samples} samples @ {audio_sr} Hz → {}",
        cli.fps,
        cli.out.display()
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
    )?;
    println!("[done ] wrote {}", cli.out.display());
    Ok(())
}
