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
use flame_core::{cuda_alloc_pool::clear_pool_cache, global_cuda_device, serialization, DType};
use inference_flame::models::sa_audio_vae::OobleckDecoder;
use inference_flame::models::turbo_vaed::{TurboVaedConfig, TurboVAED};
use inference_flame::mux::{audio_tensor_to_pcm_i16, video_tensor_to_rgb_u8, write_mp4};

struct Cli {
    latent: PathBuf,
    out: PathBuf,
    turbo_vaed_weights: PathBuf,
    sa_vae_weights: PathBuf,
    fps: usize,
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
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--latent" => latent = it.next().map(PathBuf::from),
                "--out" => out = it.next().unwrap().into(),
                "--turbo-vaed-weights" => turbo_vaed_weights = it.next().unwrap().into(),
                "--sa-vae-weights" => sa_vae_weights = it.next().unwrap().into(),
                "--fps" => fps = it.next().unwrap().parse()?,
                other => anyhow::bail!("unknown arg: {other}"),
            }
        }
        Ok(Self {
            latent: latent.ok_or_else(|| anyhow!("--latent required"))?,
            out,
            turbo_vaed_weights,
            sa_vae_weights,
            fps,
        })
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse()?;
    let device = global_cuda_device();
    println!("[load ] latent fixture: {}", cli.latent.display());
    let tensors = serialization::load_file(&cli.latent, &device)?;
    let video_lat = tensors
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
    let video = turbo_vae
        .decode(&video_lat.to_dtype(DType::BF16)?)
        .map_err(|e| anyhow!("TurboVAED decode: {e}"))?;
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
