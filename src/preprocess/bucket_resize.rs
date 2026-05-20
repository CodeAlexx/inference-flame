//! Pure-Rust port of Lance `BucketResize` + `DivisibleCrop`.
//!
//! Mirrors `/home/alex/Lance/data/video/transforms/bucket_resize.py` and
//! `/home/alex/Lance/data/video/transforms/divisible_crop.py`.
//!
//! Filter choice: Python Lance uses PIL/torchvision `InterpolationMode.BICUBIC`
//! (set at `NaResize` level, see `na_resize.py:24`). The `image` crate does not
//! ship an exact PIL-BICUBIC + antialias filter, so we use
//! [`image::imageops::FilterType::CatmullRom`], which is the closest cubic
//! kernel available. Numerical equality with the Python pipeline is therefore
//! approximate; Phase D (skeptic) will quantify the gap.

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};

/// One pre-computed bucket: target `(h, w)` and its aspect ratio `w/h`.
#[derive(Debug, Clone, Copy)]
pub struct Bucket {
    pub h: u32,
    pub w: u32,
    pub ratio: f32,
}

/// `BucketResize` from `bucket_resize.py`. Selects the nearest pre-computed
/// `(h, w)` bucket by aspect ratio, then deterministically center-crops the
/// source to that aspect ratio and bicubic-resizes it to the bucket size.
pub struct BucketResize {
    pub buckets: Vec<Bucket>,
    pub interpolation: FilterType,
}

impl BucketResize {
    /// `aspect_ratios`: list of `"W:H"` strings, e.g. `"21:9"`.
    /// `max_area`: image will fit within this many pixels (`resolution^2`).
    /// `stride`: bucket dimensions are multiples of this.
    pub fn new(
        aspect_ratios: &[&str],
        max_area: u32,
        stride: u32,
        interpolation: FilterType,
    ) -> Result<Self> {
        let buckets = init_buckets(aspect_ratios, max_area as f64, stride)?;
        Ok(Self {
            buckets,
            interpolation,
        })
    }

    /// Pick the bucket whose ratio is closest to `width / height`.
    pub fn find_nearest_bucket(&self, width: u32, height: u32) -> Bucket {
        let image_ratio = width as f32 / height as f32;
        let mut best = self.buckets[0];
        let mut best_diff = (image_ratio - best.ratio).abs();
        for &b in &self.buckets[1..] {
            let d = (image_ratio - b.ratio).abs();
            if d < best_diff {
                best_diff = d;
                best = b;
            }
        }
        best
    }

    /// Find nearest bucket, then deterministic center-crop + bicubic resize.
    ///
    /// At `scale=(1,1) ratio=(r,r)` torchvision's `RandomResizedCrop` reduces
    /// to: take the largest centered rectangle of aspect ratio `r` that fits
    /// inside the source, then resize to `(bucket_h, bucket_w)`.
    pub fn apply(&self, img: &DynamicImage) -> RgbImage {
        let (w_src, h_src) = img.dimensions();
        let bucket = self.find_nearest_bucket(w_src, h_src);
        let r = bucket.ratio;

        // Largest centered (w_crop, h_crop) with w_crop/h_crop == r, fitting inside src.
        let src_ratio = w_src as f32 / h_src as f32;
        let (w_crop, h_crop) = if src_ratio > r {
            // Source is wider → constrain height, crop width.
            let h_c = h_src as f32;
            let w_c = h_c * r;
            (w_c.round() as u32, h_src)
        } else {
            // Source is taller (or equal) → constrain width, crop height.
            let w_c = w_src as f32;
            let h_c = w_c / r;
            (w_src, h_c.round() as u32)
        };
        let w_crop = w_crop.min(w_src).max(1);
        let h_crop = h_crop.min(h_src).max(1);
        let x0 = (w_src - w_crop) / 2;
        let y0 = (h_src - h_crop) / 2;

        let cropped = img.crop_imm(x0, y0, w_crop, h_crop);
        let resized = cropped.resize_exact(bucket.w, bucket.h, self.interpolation);
        resized.to_rgb8()
    }
}

/// Center-crop image so both dims are multiples of `stride`.
/// Mirrors `divisible_crop.py`.
pub fn divisible_crop(img: &RgbImage, stride: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let new_w = w - (w % stride);
    let new_h = h - (h % stride);
    if new_w == w && new_h == h {
        return img.clone();
    }
    let x0 = (w - new_w) / 2;
    let y0 = (h - new_h) / 2;
    image::imageops::crop_imm(img, x0, y0, new_w, new_h).to_image()
}

/// Port of `BucketResize.init_buckets`. For each aspect ratio name, tries two
/// candidate `(bucket_h, bucket_w)` derivations (width-first, height-first)
/// and picks whichever lands closer to the requested ratio; ties broken by
/// area proximity to `max_area`. All buckets are stride-aligned.
fn init_buckets(aspect_ratio_names: &[&str], max_area: f64, stride: u32) -> Result<Vec<Bucket>> {
    let mut out = Vec::with_capacity(aspect_ratio_names.len());
    let s = stride as f64;
    for name in aspect_ratio_names {
        let (wn, hn) = parse_ratio(name)?;
        let aspect = wn / hn;

        // Candidate 1: pick width from sqrt(max_area * aspect), then height.
        let resize_w1 = (max_area * aspect).sqrt();
        let bw1 = ((resize_w1 / s).round() * s) as u32;
        let resize_h1 = bw1 as f64 / aspect;
        let bh1 = ((resize_h1 / s).round() * s) as u32;
        let ratio1 = bw1 as f64 / bh1 as f64;
        let area1 = bw1 as f64 * bh1 as f64;

        // Candidate 2: pick height from sqrt(max_area / aspect), then width.
        let resize_h2 = (max_area / aspect).sqrt();
        let bh2 = ((resize_h2 / s).round() * s) as u32;
        let resize_w2 = bh2 as f64 * aspect;
        let bw2 = ((resize_w2 / s).round() * s) as u32;
        let ratio2 = bw2 as f64 / bh2 as f64;
        let area2 = bw2 as f64 * bh2 as f64;

        let d1 = (ratio1 - aspect).abs();
        let d2 = (ratio2 - aspect).abs();
        let (bw, bh) = if d1 < d2 {
            (bw1, bh1)
        } else if d1 > d2 {
            (bw2, bh2)
        } else if (area1 - max_area).abs() <= (area2 - max_area).abs() {
            (bw1, bh1)
        } else {
            (bw2, bh2)
        };

        if bw == 0 || bh == 0 {
            return Err(anyhow!(
                "bucket init produced zero dim for '{name}' (max_area={max_area}, stride={stride})"
            ));
        }

        let ratio = bw as f32 / bh as f32;
        out.push(Bucket { h: bh, w: bw, ratio });
    }
    Ok(out)
}

fn parse_ratio(s: &str) -> Result<(f64, f64)> {
    let (w, h) = s
        .split_once(':')
        .ok_or_else(|| anyhow!("invalid aspect ratio '{s}', expected 'W:H'"))?;
    let w: f64 = w.trim().parse().map_err(|e| anyhow!("bad ratio W in '{s}': {e}"))?;
    let h: f64 = h.trim().parse().map_err(|e| anyhow!("bad ratio H in '{s}': {e}"))?;
    if h == 0.0 {
        return Err(anyhow!("aspect ratio H must be nonzero: '{s}'"));
    }
    Ok((w, h))
}
