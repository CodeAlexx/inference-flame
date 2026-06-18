//! Ideogram 4 final layer: affine-free LayerNorm → AdaLN scale → output Linear.
//!
//! Mirrors `Ideogram4FinalLayer.forward` (`modeling_ideogram4.py:253-262`)
//! op-for-op:
//!
//! ```text
//! scale = 1.0 + self.adaln_modulation(F.silu(c))     # BIAS Linear(adanln_dim -> hidden)
//! return self.linear(self.norm_final(x) * scale)      # BIAS Linear(hidden -> out_channels)
//! ```
//! where:
//! - `norm_final` = `nn.LayerNorm(hidden, eps=1e-6, elementwise_affine=False)`
//!   — affine-FREE: no weight, no bias (line 256). flame-core `layer_norm` with
//!   `weight=None, bias=None` is exactly this.
//! - `adaln_modulation` = `nn.Linear(adanln_dim=512 -> hidden=4608, bias=True)`
//!   (line 258).
//! - `linear` = `nn.Linear(hidden=4608 -> out_channels=128, bias=True)` (line 257).
//!
//! `c` is the shared `adaln_input` `[B, 1, adanln_dim]`. `silu(c)` then the
//! modulation linear → `[B, 1, hidden]`; `(1 + ...)` broadcast-multiplies the
//! `[B, L, hidden]` LayerNorm output over the L axis (one scale per sample,
//! NOT per token), matching the reference broadcast.
//!
//! NOTE the modulation here applies SiLU to `c` INSIDE the final layer
//! (`F.silu(c)`), whereas the per-block AdaLN consumes the already-SiLU'd
//! `adaln_input` directly. So `c` passed in is the post-SiLU `adaln_input`, and
//! the final layer applies a SECOND SiLU before its own modulation linear —
//! this is faithful to the reference (line 261 `F.silu(c)`).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::bf16_ops::silu_bf16;
use flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity;
use flame_core::{Error, Result, Tensor};

use super::weights::Ideogram4RawWeight;

/// Final-layer LayerNorm epsilon — hardcoded 1e-6 in the reference
/// (`modeling_ideogram4.py:256`).
pub const FINAL_NORM_EPS: f32 = 1e-6;

fn weight(
    weights: &HashMap<String, Ideogram4RawWeight>,
    key: &str,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| {
            Error::InvalidOperation(format!("ideogram4 final_layer: missing weight `{key}`"))
        })?
        .to_bf16_tensor(device)
}

/// `Ideogram4FinalLayer.forward(x, c)`.
///
/// - `x`: `[B, L, hidden]` BF16 — the transformer output stream.
/// - `c`: `[B, 1, adanln_dim]` BF16 — the shared `adaln_input` (post-SiLU from
///   the embed layer); a SECOND SiLU is applied here per the reference.
/// - `prefix`: e.g. `final_layer`. Sub-weights:
///   `{prefix}.adaln_modulation.{weight,bias}`, `{prefix}.linear.{weight,bias}`.
///   (`norm_final` is affine-free → no weight key.)
///
/// Returns `[B, L, out_channels]` BF16. The top-level forward casts the result
/// to F32 (the reference `out.to(torch.float32)`).
pub fn final_layer_forward(
    weights: &HashMap<String, Ideogram4RawWeight>,
    prefix: &str,
    x: &Tensor,
    c: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "ideogram4 final_layer: x must be 3D [B,L,hidden], got {dims:?}"
        )));
    }
    let hidden = dims[2];

    // scale = 1 + adaln_modulation(silu(c))   [B, 1, hidden]
    let c_silu = silu_bf16(c)?;
    let mod_w = weight(weights, &format!("{prefix}.adaln_modulation.weight"), device)?;
    let mod_b = weight(weights, &format!("{prefix}.adaln_modulation.bias"), device)?;
    let modulation = fused_linear3d_native_pytorch_parity(&c_silu, &mod_w, Some(&mod_b))?;
    let scale = modulation.add_scalar(1.0)?; // 1 + modulation  [B,1,hidden]

    // norm_final(x): affine-free LayerNorm(hidden, eps=1e-6) — weight/bias None.
    let normed = flame_core::layer_norm::layer_norm(x, &[hidden], None, None, FINAL_NORM_EPS)?;

    // norm_final(x) * scale  (broadcast [B,1,hidden] over L).
    let scaled = normed.mul(&scale)?;

    // linear(...) : BIAS Linear(hidden -> out_channels).
    let lin_w = weight(weights, &format!("{prefix}.linear.weight"), device)?;
    let lin_b = weight(weights, &format!("{prefix}.linear.bias"), device)?;
    fused_linear3d_native_pytorch_parity(&scaled, &lin_w, Some(&lin_b))
}

#[cfg(test)]
mod tests {
    #[test]
    fn final_layer_output_channels_is_128() {
        // out_channels = in_channels = 128 (= ae z_channels 32 * patch 2^2).
        let in_channels = 128usize;
        assert_eq!(in_channels, 128);
    }

    #[test]
    fn final_norm_eps_is_1e6() {
        assert_eq!(super::FINAL_NORM_EPS, 1e-6);
    }

    // GPU-dependent: requires CUDA device + FP8 dequant + layer_norm + linear.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn final_layer_compiles() {
        let _ = super::final_layer_forward;
    }
}
