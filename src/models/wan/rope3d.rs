//! 3-axis RoPE for Wan2.2 (documentation only).
//!
//! The actual RoPE application kernel lives on [`super::super::wan22_dit::Wan22Dit`]
//! because Wan TI2V-5B and Wan A14B share the **same head_dim=128** and the
//! same axis split, so a second implementation would be dead weight.
//!
//! ## Axis split for head_dim=128
//! From `wan/modules/model.py::rope_params`:
//! ```text
//! d = head_dim = 128
//! d6 = d // 6 = 21
//! axes = [d - 4*d6, 2*d6, 2*d6] = [44, 42, 42]
//! ```
//! That is, the first 44 values of each head encode *frame* position, the next
//! 42 encode *height*, and the last 42 encode *width*. Each axis is treated as
//! `(axis_size/2)` complex pairs; positions index into a sinusoidal freq table
//! with θ = 10000.
//!
//! ## Why the Klein `rope_halfsplit_perhead_bf16` kernel is not reused
//! Klein applies a *single-axis* RoPE with a simple half-split (`(even, odd)`
//! → rotate as one complex-per-pair block). Wan splits each head into **three**
//! independent axes of different sizes, and positions come from the 3D latent
//! grid `(t, h, w)`. The two kernels are not substitutable.
//!
//! See `Wan22Dit::apply_rope` for the per-token inner loop.

/// Compute the (ax_t, ax_h, ax_w) split for a given head dimension.
///
/// For Wan TI2V-5B (head_dim=128) this returns `(44, 42, 42)`.
pub fn axis_split(head_dim: usize) -> (usize, usize, usize) {
    let d6 = head_dim / 6;
    let t = head_dim - 4 * d6;
    let h = 2 * d6;
    let w = 2 * d6;
    debug_assert_eq!(t + h + w, head_dim);
    (t, h, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ti2v_5b_split() {
        assert_eq!(axis_split(128), (44, 42, 42));
    }
}
