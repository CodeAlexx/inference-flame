//! `lanpaint_gpu_smoke` — standalone smoke test for the LanPaint inpainting
//! sampler running on real GPU tensors at Klein-latent-like shape
//! `[1, 4, 64, 64]` BF16.
//!
//! Purpose: validate the `lanpaint-flame` crate produces finite output at a
//! realistic latent size with a deterministic non-identity inner model.
//!
//! The inputs are deterministic (fixed-seed PRNG on the CPU, uploaded to GPU)
//! so the test is reproducible run-to-run. The LanPaint inner loop still uses
//! `Tensor::randn` for its Langevin noise, which is NOT seeded — so per-pixel
//! outputs will differ across runs. The NaN/Inf checks and statistical bounds
//! are run-stable.
//!
//! Usage:
//!   cargo build --release --bin lanpaint_gpu_smoke
//!   ./target/release/lanpaint_gpu_smoke
//!
//! The cuDNN rpath is baked in `.cargo/config.toml`, so no LD_LIBRARY_PATH.

use std::process::ExitCode;

use flame_core::{global_cuda_device, DType, Shape, Tensor};
use lanpaint_flame::{LanPaint, LanPaintConfig};

/// Tiny deterministic PRNG (SplitMix64) used to populate fixed-seed latents
/// on CPU. We don't care about statistical quality — we care about
/// reproducibility across runs. `Tensor::randn` doesn't accept a seed in
/// flame-core, so we build the input vectors ourselves.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// U(0,1) in F32.
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits of the u64 -> F32 in [0, 1).
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / (1u32 << 24) as f32
    }

    /// Box-Muller pair, returning one sample per call. Cached pair simplifies
    /// the call path.
    fn next_normal(&mut self) -> f32 {
        // Polar / Box-Muller — straightforward, no cache for simplicity.
        loop {
            let u1 = self.next_f32();
            if u1 > 1e-7 {
                let u2 = self.next_f32();
                let r = (-2.0 * u1.ln()).sqrt();
                return r * (2.0 * std::f32::consts::PI * u2).cos();
            }
        }
    }
}

fn fixed_seed_latent(seed: u64, numel: usize) -> Vec<f32> {
    let mut rng = SplitMix64(seed);
    (0..numel).map(|_| rng.next_normal()).collect()
}

/// Checkerboard mask: 1.0 where `(y + x) % 2 == 0`, else 0.0.
fn checkerboard_mask(c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(c * h * w);
    for _cc in 0..c {
        for y in 0..h {
            for x in 0..w {
                let bit = ((y + x) % 2 == 0) as u32 as f32;
                out.push(bit);
            }
        }
    }
    out
}

struct Stats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    has_nonfinite: bool,
    n: usize,
}

fn stats_of(t: &Tensor) -> anyhow::Result<Stats> {
    let data = if t.dtype() == DType::F32 {
        t.to_vec_f32()?
    } else {
        t.to_dtype(DType::F32)?.to_vec_f32()?
    };
    let n = data.len();
    let nonfinite = data.iter().any(|v| !v.is_finite());
    if nonfinite {
        return Ok(Stats {
            mean: f32::NAN,
            std: f32::NAN,
            min: f32::NAN,
            max: f32::NAN,
            has_nonfinite: true,
            n,
        });
    }
    let mean = data.iter().copied().sum::<f32>() / n as f32;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &data {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    Ok(Stats {
        mean,
        std,
        min: lo,
        max: hi,
        has_nonfinite: false,
        n,
    })
}

fn print_stats(label: &str, s: &Stats) {
    if s.has_nonfinite {
        println!("  {label}: NON-FINITE DETECTED  n={}", s.n);
    } else {
        println!(
            "  {label}: mean={:+.5}  std={:.5}  min={:+.5}  max={:+.5}  n={}",
            s.mean, s.std, s.min, s.max, s.n
        );
    }
}

fn main() -> ExitCode {
    let device = global_cuda_device();

    // Shape [B=1, C=4, H=64, W=64] — Klein-latent-like size.
    let b: usize = 1;
    let c: usize = 4;
    let h: usize = 64;
    let w: usize = 64;
    let numel = b * c * h * w;
    let shape = Shape::from_dims(&[b, c, h, w]);

    println!("lanpaint_gpu_smoke: shape={:?}  dtype=BF16  n={}", shape.dims(), numel);

    // --- Fixed-seed inputs on CPU, then upload + cast to BF16 on GPU.
    let x0_cpu = fixed_seed_latent(0xD1CE_5EED_u64, numel);
    let latent_image_cpu = fixed_seed_latent(0xFEE1_600D_u64, numel);
    let noise_cpu = fixed_seed_latent(0xFA1A_FE11_u64, numel);
    let mask_cpu = checkerboard_mask(c, h, w);

    macro_rules! upload_bf16 {
        ($data:expr, $label:literal) => {{
            let t = Tensor::from_vec($data, shape.clone(), device.clone())
                .unwrap_or_else(|e| {
                    eprintln!("FAIL: upload {} -> GPU: {e:?}", $label);
                    std::process::exit(1);
                });
            t.to_dtype(DType::BF16).unwrap_or_else(|e| {
                eprintln!("FAIL: cast {} to BF16: {e:?}", $label);
                std::process::exit(1);
            })
        }};
    }

    let mut x = upload_bf16!(x0_cpu, "x");
    let latent_image = upload_bf16!(latent_image_cpu, "latent_image");
    let noise = upload_bf16!(noise_cpu, "noise");
    let latent_mask = upload_bf16!(mask_cpu, "latent_mask");

    // sigma, abt, tflow: [B]. Pick mid-diffusion values.
    let b_shape = Shape::from_dims(&[b]);
    let sigma = Tensor::from_vec(vec![0.5f32; b], b_shape.clone(), device.clone())
        .expect("sigma")
        .to_dtype(DType::BF16)
        .expect("sigma bf16");
    let abt = Tensor::from_vec(vec![0.5f32; b], b_shape.clone(), device.clone())
        .expect("abt")
        .to_dtype(DType::BF16)
        .expect("abt bf16");
    let tflow = Tensor::from_vec(vec![0.5f32; b], b_shape, device.clone())
        .expect("tflow")
        .to_dtype(DType::BF16)
        .expect("tflow bf16");

    // Non-identity inner model: f(x) = 0.9 * x + 0.01
    let cfg = LanPaintConfig {
        n_steps: 5,
        lambda_: 4.0,
        friction: 20.0,
        beta: 1.0,
        step_size: 0.15,
    };
    let lanpaint = LanPaint::new(
        cfg,
        Box::new(|x: &Tensor, _t: &Tensor| Ok(x.mul_scalar(0.9)?.add_scalar(0.01)?)),
    );

    // 3 outer denoise steps, each with n_steps=5 inner iterations (from cfg).
    let outer_steps = 3;
    let mut any_failure = false;

    for step in 0..outer_steps {
        let (out, advanced_x) = match lanpaint.run(
            &x,
            &latent_image,
            &noise,
            &sigma,
            &abt,
            &tflow,
            &latent_mask,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("FAIL: outer step {step} lanpaint.run error: {e:?}");
                return ExitCode::from(1);
            }
        };

        let s_out = match stats_of(&out) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("FAIL: outer step {step} out.stats: {e:?}");
                return ExitCode::from(1);
            }
        };
        let s_adv = match stats_of(&advanced_x) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("FAIL: outer step {step} advanced_x.stats: {e:?}");
                return ExitCode::from(1);
            }
        };

        println!("outer step {step}:");
        print_stats("out        ", &s_out);
        print_stats("advanced_x ", &s_adv);

        if s_out.has_nonfinite {
            eprintln!("FAIL: outer step {step} out has NaN/Inf");
            any_failure = true;
        }
        if s_adv.has_nonfinite {
            eprintln!("FAIL: outer step {step} advanced_x has NaN/Inf");
            any_failure = true;
        }

        // Thread advanced_x back in as x for next outer step — this mirrors
        // upstream `input_x.copy_(x)` semantics.
        x = advanced_x;
    }

    if any_failure {
        eprintln!("lanpaint_gpu_smoke FAIL");
        return ExitCode::from(1);
    }
    println!("LanPaint GPU smoke PASS");
    ExitCode::from(0)
}
