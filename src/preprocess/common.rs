//! Common image preprocessing scaffold shared across multimodal ports.

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, RgbImage};

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const OPENAI_CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const OPENAI_CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
pub const HALF_HALF_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const HALF_HALF_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Load an image as RGB, alpha-compositing transparent pixels onto pure white.
///
/// The 255-alpha-blend matches the PIL convention used by SenseNova/Lance
/// Python preprocessing for transparent inputs.
pub fn load_image_rgb_white_bg(path: &std::path::Path) -> Result<DynamicImage> {
    let img = image::open(path).with_context(|| format!("open {}", path.display()))?;
    match img.color() {
        image::ColorType::Rgba8 | image::ColorType::Rgba16 => {
            let (w, h) = img.dimensions();
            let mut bg = RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
            let rgba = img.to_rgba8();
            for (x, y, p) in rgba.enumerate_pixels() {
                let alpha = p.0[3] as f32 / 255.0;
                let r = (p.0[0] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                let g = (p.0[1] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                let b = (p.0[2] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
                bg.put_pixel(x, y, image::Rgb([r, g, b]));
            }
            Ok(DynamicImage::ImageRgb8(bg))
        }
        _ => Ok(DynamicImage::ImageRgb8(img.to_rgb8())),
    }
}

/// Qwen2-VL `smart_resize`: snap (h, w) to multiples of `factor` while keeping
/// pixel count within `[min_pixels, max_pixels]`.
pub fn smart_resize_qwen(
    h: u32,
    w: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> (u32, u32) {
    fn round_by(v: f32, f: u32) -> u32 {
        ((v / f as f32).round() as u32).max(1) * f
    }
    fn ceil_by(v: f32, f: u32) -> u32 {
        ((v / f as f32).ceil() as u32).max(1) * f
    }
    fn floor_by(v: f32, f: u32) -> u32 {
        ((v / f as f32).floor() as u32).max(1) * f
    }
    let h_bar = factor.max(round_by(h as f32, factor));
    let w_bar = factor.max(round_by(w as f32, factor));
    if h_bar * w_bar > max_pixels {
        let beta = ((h as f32 * w as f32) / max_pixels as f32).sqrt();
        (
            factor.max(floor_by(h as f32 / beta, factor)),
            factor.max(floor_by(w as f32 / beta, factor)),
        )
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (h as f32 * w as f32)).sqrt();
        (
            ceil_by(h as f32 * beta, factor),
            ceil_by(w as f32 * beta, factor),
        )
    } else {
        (h_bar, w_bar)
    }
}

/// Resize a `DynamicImage` to exact (new_w, new_h) using the given filter and
/// return it as an `RgbImage`.
pub fn resize_dynamic(
    img: &DynamicImage,
    new_w: u32,
    new_h: u32,
    filter: image::imageops::FilterType,
) -> RgbImage {
    img.resize_exact(new_w, new_h, filter).to_rgb8()
}

/// Normalize an RGB image into a CHW (channel-first, row-major) `Vec<f32>`
/// of length `3 * H * W`. Each value is `(channel/255 - mean[c]) / std[c]`.
pub fn chw_normalize_from_rgb(
    rgb: &RgbImage,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let (w, h) = rgb.dimensions();
    let n = (h * w) as usize;
    let mut chw = vec![0f32; 3 * n];
    for (i, p) in rgb.pixels().enumerate() {
        let r = p.0[0] as f32 / 255.0;
        let g = p.0[1] as f32 / 255.0;
        let b = p.0[2] as f32 / 255.0;
        chw[i] = (r - mean[0]) / std[0];
        chw[n + i] = (g - mean[1]) / std[1];
        chw[2 * n + i] = (b - mean[2]) / std[2];
    }
    chw
}
