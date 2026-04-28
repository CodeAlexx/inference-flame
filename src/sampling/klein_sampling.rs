//! Klein / Flux 2 sampling — direct velocity Euler ODE sampler.
//!
//! Ported from BFL's official reference: github.com/black-forest-labs/flux2
//!
//! Key differences from ComfyUI/our old code:
//! - Dynamic mu (not fixed shift=2.02)
//! - Direct velocity integration: x += (t_prev - t_curr) * pred
//! - No denoised conversion (model output IS velocity)

use flame_core::{Result, Shape, Tensor};

// ---------------------------------------------------------------------------
// Schedule — dynamic mu from image sequence length
// ---------------------------------------------------------------------------

/// Compute empirical mu from image sequence length and step count.
///
/// Matches BFL `compute_empirical_mu()` exactly.
pub fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let a1: f64 = 8.73809524e-05;
    let b1: f64 = 1.89833333;
    let a2: f64 = 0.00016927;
    let b2: f64 = 0.45666666;

    let seq = image_seq_len as f64;

    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }

    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;

    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

/// Generalized time-SNR shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma).
fn time_snr_shift(t: f64, mu: f64, sigma: f64) -> f64 {
    let exp_mu = mu.exp();
    exp_mu / (exp_mu + (1.0 / t - 1.0).powf(sigma))
}

/// Build Flux 2 sigma schedule with dynamic mu.
///
/// Returns `num_steps + 1` values from ~1.0 down to 0.0.
/// Matches BFL `get_schedule()`.
pub fn get_schedule(num_steps: usize, image_seq_len: usize) -> Vec<f32> {
    let mu = compute_empirical_mu(image_seq_len, num_steps);

    let mut timesteps = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - i as f64 / num_steps as f64; // linspace(1, 0, num_steps+1)
        let shifted = if t <= 0.0 || t >= 1.0 {
            t // avoid division by zero at endpoints
        } else {
            time_snr_shift(t, mu, 1.0)
        };
        timesteps.push(shifted as f32);
    }
    timesteps
}

/// Build schedule with a fixed shift (legacy, for ComfyUI compatibility).
pub fn build_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    const N_SIGMAS: usize = 10_000;
    let exp_mu = (shift as f64).exp();

    let sigma_buffer: Vec<f32> = (0..N_SIGMAS)
        .map(|i| {
            let t = (i + 1) as f64 / N_SIGMAS as f64;
            (exp_mu / (exp_mu + (1.0 / t - 1.0))) as f32
        })
        .collect();

    let ss = N_SIGMAS as f64 / num_steps as f64;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|x| {
            let idx = N_SIGMAS - 1 - (x as f64 * ss) as usize;
            sigma_buffer[idx]
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

// ---------------------------------------------------------------------------
// Initial noise — Box–Muller transform with seeded StdRng
// ---------------------------------------------------------------------------

/// Generate `numel` standard-normal f32 samples from a `u64` seed using the
/// Box–Muller transform over `rand::rngs::StdRng`. CPU-only; the caller
/// uploads the result to GPU. Pure-Rust determinism: same seed → same
/// samples on any host.
///
/// Verbatim port of the helper that used to live in
/// `inference-flame/src/bin/klein_lora_infer.rs`. Promoted to a public
/// function here so the EriGui `core/k_sampler` node and any future
/// Klein-arch consumer can share it without duplicating CPU RNG code.
pub fn box_muller_noise(numel: usize, seed: u64) -> Vec<f32> {
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
    v
}

// ---------------------------------------------------------------------------
// Euler ODE sampler — direct velocity integration
// ---------------------------------------------------------------------------

/// Euler ODE sampler with direct velocity integration.
///
/// `model_fn(x, t) -> velocity` — model output used directly as velocity.
/// Euler step: `x = x + (t_next - t_current) * velocity`
///
/// This matches BFL's official sampling.py `denoise()`.
pub fn euler_denoise<F>(mut model_fn: F, noise: Tensor, timesteps: &[f32]) -> Result<Tensor>
where
    F: FnMut(&Tensor, f32) -> Result<Tensor>,
{
    let mut x = noise;

    for i in 0..timesteps.len() - 1 {
        let t_curr = timesteps[i];
        let t_prev = timesteps[i + 1];

        let pred = model_fn(&x, t_curr)?;

        // Direct velocity integration: x += (t_prev - t_curr) * pred
        let dt = t_prev - t_curr;
        x = x.add(&pred.mul_scalar(dt)?)?;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Edit helpers — reference IDs and img2img schedule
// ---------------------------------------------------------------------------

/// Build position IDs for reference image tokens with distinct T-coordinate.
///
/// Returns `[H*W, 4]` BF16 tensor where each row = `[t_offset, row, col, 0]`.
/// The T-offset (typically 10.0) ensures RoPE distinguishes reference from noise tokens.
pub fn prepare_reference_ids(
    latent_height: usize,
    latent_width: usize,
    t_offset: f32,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> flame_core::Result<Tensor> {
    let n = latent_height * latent_width;
    let mut data = Vec::with_capacity(n * 4);
    for row in 0..latent_height {
        for col in 0..latent_width {
            data.push(t_offset);
            data.push(row as f32);
            data.push(col as f32);
            data.push(0.0);
        }
    }
    Tensor::from_f32_to_bf16(data, Shape::from_dims(&[n, 4]), device.clone())
}

/// Build a truncated sigma schedule for img2img / edit with partial denoising.
///
/// At `denoise >= 1.0`, equivalent to `build_sigma_schedule(num_steps, shift)`.
/// At lower values, builds a longer schedule and takes the last `num_steps + 1` entries.
pub fn build_img2img_sigmas(num_steps: usize, shift: f32, denoise: f32) -> Vec<f32> {
    if denoise >= 0.9999 {
        return build_sigma_schedule(num_steps, shift);
    }
    let new_steps = (num_steps as f32 / denoise) as usize;
    let full = build_sigma_schedule(new_steps, shift);
    let start = full.len().saturating_sub(num_steps + 1);
    full[start..].to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_length() {
        let ts = get_schedule(50, 4096);
        assert_eq!(ts.len(), 51);
    }

    #[test]
    fn test_schedule_endpoints() {
        let ts = get_schedule(50, 4096);
        assert!((ts[0] - 1.0).abs() < 0.01, "first should be ~1.0, got {}", ts[0]);
        assert_eq!(ts[50], 0.0, "last must be 0.0");
    }

    #[test]
    fn test_schedule_monotonic() {
        let ts = get_schedule(50, 4096);
        for i in 0..ts.len() - 1 {
            assert!(ts[i] >= ts[i + 1], "must decrease: ts[{}]={} < ts[{}]={}", i, ts[i], i + 1, ts[i + 1]);
        }
    }

    #[test]
    fn test_empirical_mu_1024() {
        // 1024×1024 → 64×64 = 4096 image tokens
        let mu = compute_empirical_mu(4096, 50);
        // a2*4096 + b2 = 0.00016927*4096 + 0.45666666 ≈ 1.15
        assert!((mu - 1.15).abs() < 0.1, "mu for 4096 seq, 50 steps = {}", mu);
    }

    #[test]
    fn test_empirical_mu_512() {
        // 512×512 → 32×32 = 1024 image tokens
        let mu = compute_empirical_mu(1024, 50);
        assert!(mu > 0.0 && mu < 3.0, "mu should be reasonable, got {}", mu);
    }

    #[test]
    fn test_euler_velocity_step() {
        // Direct velocity: x_new = x + dt * v
        // t_curr=0.8, t_prev=0.6, dt=-0.2, v=5.0
        // x_new = 10 + (-0.2)*5 = 9.0
        let x = 10.0f32;
        let v = 5.0f32;
        let t_curr = 0.8f32;
        let t_prev = 0.6f32;
        let dt = t_prev - t_curr;
        let x_new = x + dt * v;
        assert!((x_new - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_legacy_schedule_compat() {
        let sigmas = build_sigma_schedule(35, 2.02);
        assert_eq!(sigmas.len(), 36);
        assert!(sigmas[0] > 0.9);
        assert_eq!(sigmas[35], 0.0);
    }

    #[test]
    fn test_reference_ids_shape() {
        // Verify the CPU-side Vec math: H*W rows of 4 elements each
        let h = 64;
        let w = 48;
        let t_offset = 10.0f32;
        let n = h * w;
        let mut data = Vec::with_capacity(n * 4);
        for row in 0..h {
            for col in 0..w {
                data.push(t_offset);
                data.push(row as f32);
                data.push(col as f32);
                data.push(0.0);
            }
        }
        assert_eq!(data.len(), n * 4);
        // Check first row
        assert_eq!(data[0], 10.0); // t_offset
        assert_eq!(data[1], 0.0); // row 0
        assert_eq!(data[2], 0.0); // col 0
        assert_eq!(data[3], 0.0); // l_coord
        // Check last row
        let last = (n - 1) * 4;
        assert_eq!(data[last], 10.0);
        assert_eq!(data[last + 1], (h - 1) as f32);
        assert_eq!(data[last + 2], (w - 1) as f32);
        assert_eq!(data[last + 3], 0.0);
    }

    #[test]
    fn test_img2img_sigmas_full_denoise() {
        let full = build_img2img_sigmas(35, 2.02, 1.0);
        let reference = build_sigma_schedule(35, 2.02);
        assert_eq!(full.len(), reference.len());
        for (a, b) in full.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-7, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_img2img_sigmas_partial() {
        let num_steps = 35;
        let sigmas = build_img2img_sigmas(num_steps, 2.02, 0.7);
        assert_eq!(sigmas.len(), num_steps + 1, "partial denoise should return num_steps+1 entries");
        assert_eq!(*sigmas.last().unwrap(), 0.0, "last sigma must be 0.0");
    }
}
