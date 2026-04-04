//! Sigma schedule builders for flow-matching and diffusion models.

/// Build a linearly-spaced sigma schedule with optional shift.
///
/// Returns `num_steps + 1` values from 1.0 down to 0.0 (before shift).
/// If `shift != 1.0`, applies the flow-matching shift:
///   sigma' = shift * sigma / (1 + (shift - 1) * sigma)
pub fn build_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            *v = shift * *v / (1.0 + (shift - 1.0) * *v);
        }
    }
    t
}
