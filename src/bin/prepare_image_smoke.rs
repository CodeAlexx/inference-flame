//! Smoke test for `preprocess::qwen25vl::prepare_image_for_vit`.
//!
//! Usage:
//!   prepare_image_smoke [--image PATH] [--resolution N]
//!
//! Defaults: resolution=476. If `--image` is not supplied (or the path does
//! not exist) a synthetic 512x384 RGB PNG is written to `/tmp` and used.
//!
//! Prints input dims, processed dims, grid_thw, pixel_values stats, and
//! exits nonzero if any NaN or Inf is observed.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use flame_core::DType;
use image::{ImageBuffer, Rgb};

use inference_flame::preprocess::bucket_resize::BucketResize;
use inference_flame::preprocess::qwen25vl::{
    prepare_image_for_vit, PreparedVitImage, DEFAULT_ASPECT_RATIOS, VIT_BUCKET_STRIDE,
};

fn parse_args() -> (Option<PathBuf>, u32) {
    let mut args = std::env::args().skip(1);
    let mut image: Option<PathBuf> = None;
    let mut resolution: u32 = 476;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image = args.next().map(PathBuf::from),
            "--resolution" => {
                if let Some(v) = args.next() {
                    resolution = v.parse().expect("--resolution N");
                }
            }
            "-h" | "--help" => {
                eprintln!("usage: prepare_image_smoke [--image PATH] [--resolution N]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    (image, resolution)
}

fn ensure_input_image(supplied: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = supplied {
        if p.exists() {
            return Ok(p);
        }
        eprintln!("supplied --image {} does not exist; falling back to synthetic", p.display());
    }
    let path = PathBuf::from("/tmp/preprocess_smoke_input.png");
    let img: image::RgbImage = ImageBuffer::from_fn(512, 384, |x, y| {
        Rgb([
            ((x * 7) % 256) as u8,
            ((y * 11) % 256) as u8,
            (((x + y) * 13) % 256) as u8,
        ])
    });
    img.save(&path).context("write synthetic input png")?;
    Ok(path)
}

fn run() -> Result<i32> {
    let (image_arg, resolution) = parse_args();
    let path = ensure_input_image(image_arg)?;

    let (in_w, in_h) = image::image_dimensions(&path).context("probe input dims")?;
    println!("input image:   {} ({}x{})", path.display(), in_w, in_h);
    println!("resolution:    {resolution}");

    // Diagnostic: print the bucket list this resolution produces.
    let b = BucketResize::new(
        DEFAULT_ASPECT_RATIOS,
        resolution * resolution,
        VIT_BUCKET_STRIDE,
        image::imageops::FilterType::CatmullRom,
    )?;
    println!("buckets ({} aspect ratios, max_area={}, stride={}):", DEFAULT_ASPECT_RATIOS.len(), resolution * resolution, VIT_BUCKET_STRIDE);
    for (name, bk) in DEFAULT_ASPECT_RATIOS.iter().zip(b.buckets.iter()) {
        println!(
            "  {:>5}  h={:4}  w={:4}  ratio={:.4}  area={}",
            name,
            bk.h,
            bk.w,
            bk.ratio,
            (bk.h as u64) * (bk.w as u64)
        );
    }

    let device = flame_core::global_cuda_device();

    let PreparedVitImage {
        pixel_values,
        grid_thw,
        processed_h,
        processed_w,
    } = prepare_image_for_vit(Path::new(&path), resolution, &device)?;

    println!("processed dims: {processed_w}x{processed_h}");
    println!("grid_thw:       [{}, {}, {}]", grid_thw[0], grid_thw[1], grid_thw[2]);

    let shape = pixel_values.shape().dims().to_vec();
    let dtype = pixel_values.dtype();
    println!("pixel_values:   shape={:?} dtype={:?}", shape, dtype);
    assert_eq!(dtype, DType::BF16, "pixel_values dtype must be BF16");
    assert_eq!(shape.len(), 2, "pixel_values must be 2-D");
    assert_eq!(shape[1], 1176, "pixel_values last dim must be 1176");

    // Pull to host as F32 for stats (Tensor::to_vec auto-casts BF16→F32).
    let v = pixel_values.to_vec().context("pixel_values to_vec")?;
    let n = v.len();
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut sum_abs = 0f64;
    let mut max_abs = 0f32;
    for &x in &v {
        if x.is_nan() {
            nan += 1;
        } else if x.is_infinite() {
            inf += 1;
        } else {
            let a = x.abs();
            sum_abs += a as f64;
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    let mean_abs = if n > 0 { sum_abs / n as f64 } else { 0.0 };
    println!("stats:          n={n} mean_abs={mean_abs:.6} max_abs={max_abs:.6} nan={nan} inf={inf}");

    if nan != 0 || inf != 0 {
        eprintln!("FAIL: pixel_values contains NaN/Inf");
        return Ok(1);
    }
    println!("OK");
    Ok(0)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("error: {e:?}");
            ExitCode::from(1)
        }
    }
}
