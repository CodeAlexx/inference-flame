//! OutEmbed — final adaLN + Linear projection back to patch-space pixels.
//!
//! Mirrors `embeddings.py::OutEmbed` (lines 93-114):
//!
//! ```python
//! self.norm_final      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
//! self.linear          = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
//! self.adaLN_modulation = Sequential(SiLU, Linear(hidden_size, 2 * hidden_size, bias=True))
//!
//! def forward(self, x, adaln_input):
//!     shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=1)
//!     x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
//!     x = self.linear(x)
//!     return x
//! ```
//!
//! `x`: `[B, N_img, dim]`, `adaln_input`: `[B, dim]`. Returns `[B, N_img,
//! patch_size * patch_size * out_channels]`.
//!
//! LORA-TARGET: `adaLN_modulation.1.weight`, `linear.weight`.

use std::collections::HashMap;

use flame_core::{Result, Tensor};

fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
    out.reshape(&dims)
}

fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 3 && dims[0] > 1 {
        let (b, n, c) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[1, b * n, c])?;
        let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, Some(bias))?;
        let out_c = weight.shape().dims()[0];
        out.reshape(&[b, n, out_c])
    } else {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
    }
}

/// Run the final OutEmbed.
///
/// Weight keys expected in `shared`:
///   - `final_layer.adaLN_modulation.1.weight` `[2*dim, dim]`
///   - `final_layer.adaLN_modulation.1.bias`   `[2*dim]`
///   - `final_layer.linear.weight`             `[patch*patch*out_channels, dim]`
///   - `final_layer.linear.bias`               `[patch*patch*out_channels]`
pub fn forward(
    x: &Tensor,
    adaln_input: &Tensor,
    shared: &HashMap<String, Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let g = |k: &str| -> Result<&Tensor> {
        shared
            .get(k)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing final-layer weight: {k}")))
    };

    // adaLN_modulation: SiLU + Linear(dim → 2*dim)
    // adaln is [B, dim]; use 2D-compat helper per BUG #1 fix.
    let adaln = adaln_input.silu()?;
    let mod_out = super::model::linear_compat(
        &adaln,
        g("final_layer.adaLN_modulation.1.weight")?,
        Some(g("final_layer.adaLN_modulation.1.bias")?),
    )?;
    // chunk(2, dim=1) on [B, 2*dim]: shift = [B, dim], scale = [B, dim].
    let two_dim = mod_out.shape().dims()[1];
    let dim = two_dim / 2;
    let shift = mod_out.narrow(1, 0, dim)?;
    let scale = mod_out.narrow(1, dim, dim)?;

    // x: [B, N, dim]. norm_final → mul (1+scale.unsqueeze(1)) → add shift.unsqueeze(1)
    let normed = layer_norm_no_affine(x, eps)?;
    let one_plus = scale.add_scalar(1.0)?.unsqueeze(1)?;
    let scaled = normed.mul(&one_plus)?;
    let modulated = scaled.add(&shift.unsqueeze(1)?)?;

    // linear
    linear_bias(&modulated, g("final_layer.linear.weight")?, g("final_layer.linear.bias")?)
}
