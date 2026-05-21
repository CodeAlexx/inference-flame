//! `TimestepEmbedder` — sinusoidal timestep embedding + 2-MLP.
//!
//! Reference: `qwen3_vl_transformers.py:980-1019`.
//!
//! ```python
//! class TimestepEmbedder(nn.Module):
//!     def __init__(self, config, hidden_size, frequency_embedding_size=256):
//!         self.mlp = nn.Sequential(
//!             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
//!             nn.SiLU(),
//!             nn.Linear(hidden_size, hidden_size, bias=True),
//!         )
//!         self.frequency_embedding_size = frequency_embedding_size
//!
//!     @staticmethod
//!     def timestep_embedding(t, dim, max_period=10000):
//!         half = dim // 2
//!         freqs = torch.exp(-math.log(max_period)
//!                           * torch.arange(0, half).float() / half)
//!         args = t[:, None].float() * freqs[None]
//!         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
//!         return embedding
//!
//!     def forward(self, t):
//!         t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
//!         t_emb  = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
//!         return t_emb
//! ```
//!
//! **Important**: Python multiplies `t` by **1000 inside forward**. The pipeline
//! pre-divides by 1000 (`pipeline.py:344`: `t_pixeldit = 1.0 - step_t / 1000.0`),
//! so the model receives `t ∈ [0, 1]`, then internally rescales to [0, 1000]
//! before feeding the sinusoid. This Rust port preserves that exact behavior.
//!
//! **Layout note**: Python concatenates `[cos(args), sin(args)]` (cos first,
//! sin second) along the last dim — see `qwen3_vl_transformers.py:1011`.

use std::sync::Arc;

use flame_core::nn::Linear;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};

use super::HiDreamO1Config;
use super::lora::LoraRegistry;

/// 2-layer MLP that turns a scalar timestep into a `[B, hidden_size]` vector,
/// which is then scattered into the sequence at every `<|tms_token|>` slot
/// (`qwen3_vl_transformers.py:1443-1452`).
pub struct TimestepEmbedder {
    /// `mlp.0` — `Linear(frequency_embedding_size → hidden_size)`. Has bias.
    pub mlp_in: Linear,
    /// `mlp.2` — `Linear(hidden_size → hidden_size)`. Has bias.
    /// (Index 1 in the original `nn.Sequential` is `nn.SiLU()`.)
    pub mlp_out: Linear,
    /// `frequency_embedding_size` — 256 for HiDream-O1.
    pub frequency_embedding_size: usize,
    /// `hidden_size` — 4096 for the 8B variant.
    pub hidden_size: usize,
    device: Arc<CudaDevice>,
}

impl TimestepEmbedder {
    pub fn new(config: &HiDreamO1Config, device: &Arc<CudaDevice>) -> Result<Self> {
        let mlp_in = Linear::new_zeroed(
            config.timestep_freq_dim,
            config.hidden_size,
            /*bias=*/ true,
            device,
        )?;
        let mlp_out = Linear::new_zeroed(
            config.hidden_size,
            config.hidden_size,
            /*bias=*/ true,
            device,
        )?;
        Ok(Self {
            mlp_in,
            mlp_out,
            frequency_embedding_size: config.timestep_freq_dim,
            hidden_size: config.hidden_size,
            device: device.clone(),
        })
    }

    /// Forward.
    ///
    /// Input: `t : [B]` (or `[1]`), values in `[0, 1]` (the pipeline-side
    /// schedule fraction). Internally rescales `t * 1000` before the
    /// sinusoid, mirroring Python.
    ///
    /// Output: `[B, hidden_size]`, dtype matches `mlp_in.weight` (BF16).
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        self.forward_lora(t, None)
    }

    pub fn forward_lora(&self, t: &Tensor, lora: Option<&LoraRegistry>) -> Result<Tensor> {
        // Build sinusoidal embedding on CPU (small, batch-1 typical) at FP32,
        // then upload as BF16 — matches the `t_freq.to(self.mlp[0].weight.dtype)`
        // cast in Python.
        let t_freq = Self::timestep_embedding(
            t,
            self.frequency_embedding_size,
            10_000.0,
            1000.0,
            &self.device,
        )?;
        let batch = t_freq.shape().dims()[0];
        let t_freq_3d = t_freq.reshape(&[batch, 1, self.frequency_embedding_size])?;
        let h_3d = match lora.and_then(|r| r.get_global("t_embedder1.mlp.0")) {
            Some(adapter) => {
                let a = adapter.a_tensor()?;
                let b = adapter.b_tensor()?;
                flame_core::ops::fused_inference::fused_linear3d_native_lora(
                    &t_freq_3d,
                    &self.mlp_in.weight,
                    self.mlp_in.bias.as_ref(),
                    Some(&a),
                    Some(&b),
                    adapter.scale,
                )?
            }
            None => flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
                &t_freq_3d,
                &self.mlp_in.weight,
                self.mlp_in.bias.as_ref(),
            )?,
        };
        let h = h_3d.reshape(&[batch, self.hidden_size])?;
        let h = h.silu()?;
        let h_3d = h.reshape(&[batch, 1, self.hidden_size])?;
        let out_3d = match lora.and_then(|r| r.get_global("t_embedder1.mlp.2")) {
            Some(adapter) => {
                let a = adapter.a_tensor()?;
                let b = adapter.b_tensor()?;
                flame_core::ops::fused_inference::fused_linear3d_native_lora(
                    &h_3d,
                    &self.mlp_out.weight,
                    self.mlp_out.bias.as_ref(),
                    Some(&a),
                    Some(&b),
                    adapter.scale,
                )?
            }
            None => flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
                &h_3d,
                &self.mlp_out.weight,
                self.mlp_out.bias.as_ref(),
            )?,
        };
        out_3d.reshape(&[batch, self.hidden_size])
    }

    /// Sinusoidal timestep embedding.
    ///
    /// Mirrors `qwen3_vl_transformers.py:996-1014`. `t_scale` is the
    /// pre-multiplier applied to `t` *inside this function* — Python passes
    /// `t * 1000` into `timestep_embedding`, which we collapse here so the
    /// caller can pass `t ∈ [0, 1]` directly.
    ///
    /// Returns BF16 tensor of shape `[B, dim]`.
    pub fn timestep_embedding(
        t: &Tensor,
        dim: usize,
        max_period: f32,
        t_scale: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        if dim < 2 || dim % 2 != 0 {
            return Err(Error::InvalidOperation(format!(
                "TimestepEmbedder::timestep_embedding: dim={} must be even and ≥2",
                dim
            )));
        }
        let half = dim / 2;
        let log_max = max_period.ln();
        // freqs[i] = exp(-log(max_period) * i / half)  for i in 0..half
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-log_max * (i as f32) / (half as f32)).exp())
            .collect();

        // Pull `t` to host so we can build the sinusoid without an extra
        // FP32 elementwise kernel. Batch size is typically 1 (or 2 for CFG).
        let t_host_bf16 = t.to_dtype(DType::F32)?.to_vec()?;
        let batch = t_host_bf16.len();

        let mut emb = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let t_val = t_host_bf16[b] * t_scale;
            for i in 0..half {
                let arg = t_val * freqs[i];
                // Python: cat([cos, sin], dim=-1) — cos first half, sin second.
                emb[b * dim + i] = arg.cos();
                emb[b * dim + half + i] = arg.sin();
            }
        }

        Tensor::from_vec_dtype(emb, Shape::from_dims(&[batch, dim]), device.clone(), DType::BF16)
    }
}
