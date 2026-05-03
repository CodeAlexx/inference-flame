//! Sample-rate conversion via rubato's high-quality sinc resampler.
//!
//! Used to bring arbitrary input WAVs (44.1 kHz, 48 kHz, 22.05 kHz, etc.)
//! to the SA Open VAE's training rate of 51,200 Hz before encoding.

use anyhow::{anyhow, Result};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Resample mono `f32` samples from `in_rate` Hz to `out_rate` Hz.
/// High-quality sinc with `Blackman-Harris²` window and 256 taps.
pub fn resample_to(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>> {
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    // Process in one shot — pass the entire buffer as the chunk size.
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, input.len(), 1)
        .map_err(|e| anyhow!("rubato init: {e:?}"))?;

    let out = resampler
        .process(&[input.to_vec()], None)
        .map_err(|e| anyhow!("rubato process: {e:?}"))?;
    Ok(out.into_iter().next().unwrap_or_default())
}
