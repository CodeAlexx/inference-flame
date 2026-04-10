//! Kandinsky-5 DiffusionTransformer3D — pure Rust, flame-core + FlameSwap.
//!
//! Port of `kandinsky/models/dit.py` + `kandinsky/models/nn.py` from the
//! Kandinsky-5 repo to Rust for the inference-flame project.
//!
//! ## Architecture (Lite variant)
//! - **in_visual_dim**: 16, **out_visual_dim**: 16
//! - **time_dim**: 512, **model_dim**: 1792, **ff_dim**: 7168
//! - **patch_size**: [1, 2, 2]
//! - **num_text_blocks**: 2 (encoder, self-attention + FFN, with AdaLN)
//! - **num_visual_blocks**: 32 (decoder, self-attn + cross-attn + FFN, with AdaLN)
//! - **axes_dims**: [16, 24, 24] → head_dim = 64, num_heads = 1792/64 = 28
//! - **in_text_dim**: 3584 (Qwen hidden), **in_text_dim2**: 768 (CLIP pooled)
//! - All projections use **bias** (unlike FLUX/Klein)
//! - FeedForward uses **no bias** + **GELU** (not SiLU/SwiGLU)
//! - RoPE uses **rotation matrices**, decomposed to cos/sin for `rope_fused_bf16`
//!
//! ## Forward pass
//! 1. Embed text (Linear+LayerNorm), time (sinusoidal+MLP), pooled_text, visual (patchify+Linear)
//! 2. Build text RoPE (1D), run 2 encoder blocks
//! 3. Build visual RoPE (3D with scale_factor), flatten visual to (1, D*H/2*W/2, dim)
//! 4. Run 32 decoder blocks (self-attn + cross-attn to text + FFN)
//! 5. Unflatten visual, OutLayer (modulate + Linear + unpatch)
//!
//! ## FlameSwap layout
//! Only the 32 visual decoder blocks are swapped. The 2 text encoder blocks (small)
//! and all embedding/output weights stay GPU-resident in `shared`.
//!
//! ## Weight key format
//! ```text
//! time_embeddings.in_layer.{weight,bias}
//! time_embeddings.out_layer.{weight,bias}
//! text_embeddings.in_layer.{weight,bias}
//! text_embeddings.norm.{weight,bias}
//! pooled_text_embeddings.in_layer.{weight,bias}
//! pooled_text_embeddings.norm.{weight,bias}
//! visual_embeddings.in_layer.{weight,bias}
//! text_rope_embeddings.args
//! visual_rope_embeddings.args_{0,1,2}
//! text_transformer_blocks.{i}.text_modulation.out_layer.{weight,bias}
//! text_transformer_blocks.{i}.self_attention.to_{query,key,value}.{weight,bias}
//! text_transformer_blocks.{i}.self_attention.{query,key}_norm.weight
//! text_transformer_blocks.{i}.self_attention.out_layer.{weight,bias}
//! text_transformer_blocks.{i}.feed_forward.in_layer.weight       (NO bias)
//! text_transformer_blocks.{i}.feed_forward.out_layer.weight      (NO bias)
//! visual_transformer_blocks.{i}.visual_modulation.out_layer.{weight,bias}
//! visual_transformer_blocks.{i}.self_attention.to_{query,key,value}.{weight,bias}
//! visual_transformer_blocks.{i}.self_attention.{query,key}_norm.weight
//! visual_transformer_blocks.{i}.self_attention.out_layer.{weight,bias}
//! visual_transformer_blocks.{i}.cross_attention.to_{query,key,value}.{weight,bias}
//! visual_transformer_blocks.{i}.cross_attention.{query,key}_norm.weight
//! visual_transformer_blocks.{i}.cross_attention.out_layer.{weight,bias}
//! visual_transformer_blocks.{i}.feed_forward.in_layer.weight     (NO bias)
//! visual_transformer_blocks.{i}.feed_forward.out_layer.weight    (NO bias)
//! out_layer.modulation.out_layer.{weight,bias}
//! out_layer.out_layer.{weight,bias}
//! ```

use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_swap::FlameSwap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Kandinsky5Config {
    pub in_visual_dim: usize,      // 16
    pub out_visual_dim: usize,     // 16
    pub in_text_dim: usize,        // 3584 (Qwen)
    pub in_text_dim2: usize,       // 768 (CLIP pooled)
    pub time_dim: usize,           // 512
    pub model_dim: usize,          // 1792
    pub ff_dim: usize,             // 7168
    pub patch_size: [usize; 3],    // [1, 2, 2]
    pub num_text_blocks: usize,    // 2
    pub num_visual_blocks: usize,  // 32
    pub axes_dims: [usize; 3],     // [16, 24, 24]
    pub max_period: f64,           // 10000.0
    pub eps: f32,                  // 1e-6
}

impl Kandinsky5Config {
    pub fn head_dim(&self) -> usize {
        self.axes_dims[0] + self.axes_dims[1] + self.axes_dims[2]
    }

    pub fn num_heads(&self) -> usize {
        self.model_dim / self.head_dim()
    }

    pub fn patch_dim(&self) -> usize {
        self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.in_visual_dim
    }

    pub fn out_patch_dim(&self) -> usize {
        self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.out_visual_dim
    }
}

impl Default for Kandinsky5Config {
    fn default() -> Self {
        Self {
            in_visual_dim: 16,
            out_visual_dim: 16,
            in_text_dim: 3584,
            in_text_dim2: 768,
            time_dim: 512,
            model_dim: 1792,
            ff_dim: 7168,
            patch_size: [1, 2, 2],
            num_text_blocks: 2,
            num_visual_blocks: 32,
            axes_dims: [16, 24, 24],
            max_period: 10000.0,
            eps: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Kandinsky5DiT
// ---------------------------------------------------------------------------

/// Kandinsky-5 DiffusionTransformer3D.
///
/// Visual decoder blocks (`visual_transformer_blocks.{i}`) are managed by
/// FlameSwap for GPU memory efficiency. Text encoder blocks (only 2) and all
/// embedding/output weights stay GPU-resident in `shared`.
pub struct Kandinsky5DiT {
    shared: HashMap<String, Tensor>,
    swap: FlameSwap,
    config: Kandinsky5Config,
    device: Arc<CudaDevice>,
}

impl Kandinsky5DiT {
    /// Load Kandinsky-5 DiT from one or more safetensors checkpoint files.
    pub fn load(
        checkpoint_paths: &[&str],
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = Kandinsky5Config::default();

        // FlameSwap manages the 32 visual decoder blocks.
        let swap = FlameSwap::load(
            checkpoint_paths,
            device,
            |name| {
                if let Some(rest) = name.strip_prefix("visual_transformer_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(idx);
                }
                None
            },
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap Kandinsky5: {e}")))?;

        // Shared weights: everything NOT inside visual_transformer_blocks.
        let shared_prefixes = [
            "time_embeddings.",
            "text_embeddings.",
            "pooled_text_embeddings.",
            "visual_embeddings.",
            "text_rope_embeddings.",
            "visual_rope_embeddings.",
            "text_transformer_blocks.",
            "out_layer.",
        ];
        let mut shared = HashMap::new();
        for path in checkpoint_paths {
            let part = load_file_filtered(Path::new(path), device, |key| {
                shared_prefixes.iter().any(|p| key.starts_with(p))
            })?;
            for (k, v) in part {
                let v_bf16 = if v.dtype() != DType::BF16 {
                    v.to_dtype(DType::BF16).unwrap_or(v)
                } else {
                    v
                };
                shared.insert(k, v_bf16);
            }
        }

        log::info!(
            "[Kandinsky5] Loaded: {} visual blocks via FlameSwap, {} shared weights",
            swap.num_blocks(),
            shared.len()
        );
        log::info!(
            "[Kandinsky5] Config: model_dim={}, ff_dim={}, heads={}, head_dim={}, text_blocks={}, visual_blocks={}",
            config.model_dim, config.ff_dim, config.num_heads(), config.head_dim(),
            config.num_text_blocks, config.num_visual_blocks,
        );

        Ok(Self {
            shared,
            swap,
            config,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &Kandinsky5Config {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Linear with bias. Flattens batch dims for cuBLASLt compatibility.
    fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() == 3 && dims[0] > 1 {
            let (b, n, c) = (dims[0], dims[1], dims[2]);
            let flat = x.reshape(&[1, b * n, c])?;
            let out =
                flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, Some(bias))?;
            let out_c = weight.shape().dims()[0];
            out.reshape(&[b, n, out_c])
        } else {
            flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
        }
    }

    /// Linear without bias.
    fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() == 3 && dims[0] > 1 {
            let (b, n, c) = (dims[0], dims[1], dims[2]);
            let flat = x.reshape(&[1, b * n, c])?;
            let out =
                flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, None)?;
            let out_c = weight.shape().dims()[0];
            out.reshape(&[b, n, out_c])
        } else {
            flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
        }
    }

    /// RMSNorm (weight-only, no bias). Used for Q/K normalization.
    fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
        out.reshape(&dims)
    }

    /// LayerNorm with affine (weight + bias). Used for text/pooled embeddings.
    fn layer_norm_affine(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out =
            flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, Some(weight), Some(bias), eps)?;
        out.reshape(&dims)
    }

    /// LayerNorm without affine. Used inside AdaLN modulation.
    fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
        out.reshape(&dims)
    }

    /// Fused modulate: LayerNorm(x) * (1 + scale) + shift
    fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        flame_core::bf16_ops::modulate_pre_fused_bf16(x, shift, scale, eps)
    }

    // -----------------------------------------------------------------------
    // Embeddings
    // -----------------------------------------------------------------------

    /// Sinusoidal time embedding.
    ///
    /// Python reference (`get_freqs`):
    ///   `freqs = exp(-ln(max_period) * arange(0, dim) / dim)`
    ///
    /// Then: `args = outer(time, freqs)`, embed = `cat(cos(args), sin(args))`
    /// Followed by MLP: Linear(model_dim, time_dim) → SiLU → Linear(time_dim, time_dim).
    ///
    /// NOTE: The `time` input is expected to already be scaled by 1000 (done by the
    /// caller / sampling loop, matching `get_velocity` in the Python reference which
    /// passes `t * 1000` to the DiT).
    fn time_embedding(&self, time: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let model_dim = cfg.model_dim;
        let half = model_dim / 2;
        let max_period = cfg.max_period;

        // Time is already scaled by 1000 by the caller (matching Python get_velocity).
        // TimeEmbeddings.forward in Python does NOT scale — it uses time directly.
        let time_f32 = time.to_dtype(DType::F32)?;

        // Build freq vector: exp(-ln(max_period) * i / dim) for i in 0..half
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-(max_period.ln()) * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs =
            Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), self.device.clone())?;

        // args = outer(time, freqs) → [B, half]
        let t_col = time_f32.unsqueeze(1)?; // [B, 1]
        let args = t_col.matmul(&freqs)?; // [B, half]

        // embed = cat(cos, sin) → [B, model_dim]
        let cos = args.cos()?;
        let sin = args.sin()?;
        let embed = Tensor::cat(&[&cos, &sin], 1)?.to_dtype(DType::BF16)?;

        // MLP: in_layer → SiLU → out_layer
        let in_w = self.get_shared("time_embeddings.in_layer.weight")?;
        let in_b = self.get_shared("time_embeddings.in_layer.bias")?;
        let out_w = self.get_shared("time_embeddings.out_layer.weight")?;
        let out_b = self.get_shared("time_embeddings.out_layer.bias")?;

        let embed_3d = embed.unsqueeze(0)?; // [1, B, model_dim]
        let h = Self::linear_bias(&embed_3d, in_w, in_b)?;
        let h = h.silu()?;
        let h = Self::linear_bias(&h, out_w, out_b)?;
        let h = h.squeeze(Some(0))?; // [B, time_dim]
        Ok(h)
    }

    /// Text embedding: Linear(in_text_dim, model_dim) + LayerNorm(affine=True).
    fn text_embedding(&self, text: &Tensor) -> Result<Tensor> {
        let in_w = self.get_shared("text_embeddings.in_layer.weight")?;
        let in_b = self.get_shared("text_embeddings.in_layer.bias")?;
        let norm_w = self.get_shared("text_embeddings.norm.weight")?;
        let norm_b = self.get_shared("text_embeddings.norm.bias")?;

        let h = Self::linear_bias(text, in_w, in_b)?;
        Self::layer_norm_affine(&h, norm_w, norm_b, self.config.eps)
    }

    /// Pooled text embedding: Linear(in_text_dim2=768, time_dim=512) + LayerNorm(affine=True).
    /// Output is added to the time embedding.
    fn pooled_text_embedding(&self, pooled: &Tensor) -> Result<Tensor> {
        let in_w = self.get_shared("pooled_text_embeddings.in_layer.weight")?;
        let in_b = self.get_shared("pooled_text_embeddings.in_layer.bias")?;
        let norm_w = self.get_shared("pooled_text_embeddings.norm.weight")?;
        let norm_b = self.get_shared("pooled_text_embeddings.norm.bias")?;

        let h = Self::linear_bias(pooled, in_w, in_b)?;
        Self::layer_norm_affine(&h, norm_w, norm_b, self.config.eps)
    }

    /// Visual embedding: patchify then Linear(patch_dim, model_dim).
    ///
    /// Input: `[D, H, W, 16]` (duration, height, width, channels).
    /// Patchify by `[1, 2, 2]` → `[D, H/2, W/2, 64]`.
    /// Linear → `[D, H/2, W/2, model_dim]`.
    fn visual_embedding(&self, x: &Tensor) -> Result<(Tensor, usize, usize, usize)> {
        let cfg = &self.config;
        let dims = x.shape().dims().to_vec();
        let (dur, h, w, _c) = (dims[0], dims[1], dims[2], dims[3]);
        let (pt, ph, pw) = (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]);
        let d_out = dur / pt;
        let h_out = h / ph;
        let w_out = w / pw;

        // Patchify: [D, H, W, C] → [D/pt, H/ph, W/pw, pt*ph*pw*C]
        let patched = self.patchify(x, dur, h, w)?;
        // patched: [d_out * h_out * w_out, patch_dim]

        // Linear projection
        let in_w = self.get_shared("visual_embeddings.in_layer.weight")?;
        let in_b = self.get_shared("visual_embeddings.in_layer.bias")?;

        // Reshape to 3D for linear: [1, n_patches, patch_dim]
        let n_patches = d_out * h_out * w_out;
        let patched_3d = patched.reshape(&[1, n_patches, cfg.patch_dim()])?;
        let projected = Self::linear_bias(&patched_3d, in_w, in_b)?;
        // Reshape to spatial: [d_out, h_out, w_out, model_dim]
        let spatial = projected.reshape(&[d_out, h_out, w_out, cfg.model_dim])?;

        Ok((spatial, d_out, h_out, w_out))
    }

    // -----------------------------------------------------------------------
    // RoPE
    // -----------------------------------------------------------------------

    /// Build 1D RoPE cos/sin from the precomputed `text_rope_embeddings.args` buffer.
    ///
    /// The `args` buffer is `[max_pos, head_dim/2]` precomputed angles.
    /// `text_rope_pos` is an integer position tensor `[seq_len]`.
    ///
    /// Returns `(cos, sin)` each `[1, 1, seq_len, head_dim/2]` BF16, ready for
    /// `rope_fused_bf16`.
    fn build_text_rope(&self, text_rope_pos: &Tensor) -> Result<(Tensor, Tensor)> {
        // args buffer: [max_pos, head_dim/2] stored as BF16 (cast from F32 at load)
        let args_buf = self.get_shared("text_rope_embeddings.args")?;
        // Index into args using positions
        let args_buf_f32 = args_buf.to_dtype(DType::F32)?;
        let pos_data = text_rope_pos.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        let args_dims = args_buf_f32.shape().dims().to_vec();
        let half_dim = args_dims[1]; // head_dim/2
        let args_flat = args_buf_f32.to_vec1::<f32>()?;

        let seq_len = pos_data.len();
        let mut cos_data = vec![0.0f32; seq_len * half_dim];
        let mut sin_data = vec![0.0f32; seq_len * half_dim];

        for (si, &pos) in pos_data.iter().enumerate() {
            let pos_idx = pos as usize;
            for d in 0..half_dim {
                let angle = args_flat[pos_idx * half_dim + d];
                cos_data[si * half_dim + d] = angle.cos();
                sin_data[si * half_dim + d] = angle.sin();
            }
        }

        // Shape: [1, 1, seq_len, half_dim] for broadcasting over [B, H, S, D/2]
        let cos = Tensor::from_vec(
            cos_data,
            Shape::from_dims(&[seq_len, half_dim]),
            self.device.clone(),
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(
            sin_data,
            Shape::from_dims(&[seq_len, half_dim]),
            self.device.clone(),
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::BF16)?;

        Ok((cos, sin))
    }

    /// Build 3D RoPE cos/sin from the precomputed `visual_rope_embeddings.args_{0,1,2}` buffers.
    ///
    /// `visual_rope_pos`: 3 tensors `[D_out]`, `[H_out]`, `[W_out]` position indices.
    /// `scale_factor`: `[f32; 3]` per-axis scale factors.
    ///
    /// Returns `(cos, sin)` each `[1, 1, D*H*W, head_dim/2]` BF16.
    fn build_visual_rope(
        &self,
        d_out: usize,
        h_out: usize,
        w_out: usize,
        visual_rope_pos: &[Tensor; 3],
        scale_factor: [f32; 3],
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let axes_dims = &cfg.axes_dims;
        let total_half: usize = axes_dims.iter().map(|d| d / 2).sum();
        let seq_len = d_out * h_out * w_out;

        // Load the 3 args buffers and position arrays
        let mut axis_cos: Vec<Vec<f32>> = Vec::with_capacity(3);
        let mut axis_sin: Vec<Vec<f32>> = Vec::with_capacity(3);

        for axis in 0..3usize {
            let args_key = format!("visual_rope_embeddings.args_{axis}");
            let args_buf = self.get_shared(&args_key)?;
            let args_f32 = args_buf.to_dtype(DType::F32)?;
            let args_flat = args_f32.to_vec1::<f32>()?;
            let half_dim = axes_dims[axis] / 2;
            let args_stride = half_dim; // args is [max_pos, half_dim]

            let pos_data = visual_rope_pos[axis].to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let axis_len = pos_data.len(); // D_out, H_out, or W_out

            let sf = scale_factor[axis] as f64;
            let mut cos_vec = vec![0.0f32; axis_len * half_dim];
            let mut sin_vec = vec![0.0f32; axis_len * half_dim];

            for (pi, &pos) in pos_data.iter().enumerate() {
                let pos_idx = pos as usize;
                for d in 0..half_dim {
                    let angle = args_flat[pos_idx * args_stride + d] as f64 / sf;
                    cos_vec[pi * half_dim + d] = angle.cos() as f32;
                    sin_vec[pi * half_dim + d] = angle.sin() as f32;
                }
            }
            axis_cos.push(cos_vec);
            axis_sin.push(sin_vec);
        }

        // Combine 3 axes into full cos/sin: expand each axis over the spatial grid
        // axis 0 (temporal): [D, 1, 1, half0] → broadcast over H, W
        // axis 1 (height):   [1, H, 1, half1] → broadcast over D, W
        // axis 2 (width):    [1, 1, W, half2] → broadcast over D, H
        // Concatenated: [D, H, W, total_half] → flatten → [D*H*W, total_half]
        let half_dims: Vec<usize> = axes_dims.iter().map(|d| d / 2).collect();
        let mut full_cos = vec![0.0f32; seq_len * total_half];
        let mut full_sin = vec![0.0f32; seq_len * total_half];

        for di in 0..d_out {
            for hi in 0..h_out {
                for wi in 0..w_out {
                    let flat_idx = di * h_out * w_out + hi * w_out + wi;
                    let mut offset = 0usize;

                    // Axis 0 (temporal)
                    for d in 0..half_dims[0] {
                        full_cos[flat_idx * total_half + offset + d] =
                            axis_cos[0][di * half_dims[0] + d];
                        full_sin[flat_idx * total_half + offset + d] =
                            axis_sin[0][di * half_dims[0] + d];
                    }
                    offset += half_dims[0];

                    // Axis 1 (height)
                    for d in 0..half_dims[1] {
                        full_cos[flat_idx * total_half + offset + d] =
                            axis_cos[1][hi * half_dims[1] + d];
                        full_sin[flat_idx * total_half + offset + d] =
                            axis_sin[1][hi * half_dims[1] + d];
                    }
                    offset += half_dims[1];

                    // Axis 2 (width)
                    for d in 0..half_dims[2] {
                        full_cos[flat_idx * total_half + offset + d] =
                            axis_cos[2][wi * half_dims[2] + d];
                        full_sin[flat_idx * total_half + offset + d] =
                            axis_sin[2][wi * half_dims[2] + d];
                    }
                }
            }
        }

        // Shape: [1, 1, seq_len, total_half]
        let cos = Tensor::from_vec(
            full_cos,
            Shape::from_dims(&[seq_len, total_half]),
            self.device.clone(),
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(
            full_sin,
            Shape::from_dims(&[seq_len, total_half]),
            self.device.clone(),
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::BF16)?;

        Ok((cos, sin))
    }

    /// Apply RoPE to Q and K tensors using the fused kernel.
    fn apply_rope(
        q: &Tensor,
        k: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q_out = flame_core::bf16_ops::rope_fused_bf16(q, pe_cos, pe_sin)?;
        let k_out = flame_core::bf16_ops::rope_fused_bf16(k, pe_cos, pe_sin)?;
        Ok((q_out, k_out))
    }

    // -----------------------------------------------------------------------
    // Modulation
    // -----------------------------------------------------------------------

    /// Run modulation layer: SiLU(time_embed) → Linear → chunk.
    ///
    /// `time_embed` is `[B, time_dim]`, output is `[B, num_params * model_dim]`.
    /// We chunk it into `num_params` tensors of `[B, model_dim]`.
    fn modulation(
        &self,
        time_embed: &Tensor,
        mod_weight: &Tensor,
        mod_bias: &Tensor,
        num_params: usize,
    ) -> Result<Vec<Tensor>> {
        let cfg = &self.config;
        let dim = cfg.model_dim;

        // SiLU → Linear
        let activated = time_embed.silu()?;
        let activated_3d = activated.unsqueeze(1)?; // [B, 1, time_dim]
        let params = Self::linear_bias(&activated_3d, mod_weight, mod_bias)?;
        // params: [B, 1, num_params * model_dim]
        let params = params.squeeze(Some(1))?; // [B, num_params * model_dim]

        // Chunk into num_params pieces along the last dim
        let b = params.shape().dims()[0];
        let mut chunks = Vec::with_capacity(num_params);
        for i in 0..num_params {
            let chunk = params.narrow(1, i * dim, dim)?; // [B, model_dim]
            chunks.push(chunk);
        }
        let _ = b;
        Ok(chunks)
    }

    // -----------------------------------------------------------------------
    // Patchify / Unpatchify
    // -----------------------------------------------------------------------

    /// Patchify: `[D, H, W, C]` → `[D/pt * H/ph * W/pw, pt*ph*pw*C]`
    ///
    /// Python reference (`VisualEmbeddings.forward`):
    /// ```python
    /// x.view(D//p0, p0, H//p1, p1, W//p2, p2, C).permute(0,2,4,1,3,5,6).flatten(3,6)
    /// ```
    fn patchify(&self, x: &Tensor, dur: usize, h: usize, w: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let c = cfg.in_visual_dim;
        let (pt, ph, pw) = (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]);
        let d_out = dur / pt;
        let h_out = h / ph;
        let w_out = w / pw;
        let patch_dim = pt * ph * pw * c;
        let n_patches = d_out * h_out * w_out;

        let x_f32 = x.to_dtype(DType::F32)?;
        let x_data = x_f32.to_vec1::<f32>()?;
        let mut out = vec![0.0f32; n_patches * patch_dim];

        // Input layout: [D, H, W, C] — row-major
        for di in 0..d_out {
            for hi in 0..h_out {
                for wi in 0..w_out {
                    let patch_idx = di * h_out * w_out + hi * w_out + wi;
                    for pti in 0..pt {
                        for phi in 0..ph {
                            for pwi in 0..pw {
                                for ci in 0..c {
                                    let src_d = di * pt + pti;
                                    let src_h = hi * ph + phi;
                                    let src_w = wi * pw + pwi;
                                    // [D, H, W, C] index
                                    let src_idx =
                                        src_d * h * w * c + src_h * w * c + src_w * c + ci;
                                    // After permute(0,2,4,1,3,5,6).flatten(3,6):
                                    // patch order is [pti, phi, pwi, ci]
                                    let dst_ch = pti * ph * pw * c + phi * pw * c + pwi * c + ci;
                                    out[patch_idx * patch_dim + dst_ch] = x_data[src_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_f32_to_bf16(out, Shape::from_dims(&[n_patches, patch_dim]), self.device.clone())
    }

    /// Unpatchify: `[D_out, H_out, W_out, out_patch_dim]` → `[D, H, W, out_visual_dim]`
    ///
    /// Python (`OutLayer.forward`):
    /// ```python
    /// x.view(D, H, W, C_out, p0, p1, p2).permute(0,4,1,5,2,6,3)
    ///  .flatten(0,1).flatten(1,2).flatten(2,3)
    /// ```
    fn unpatchify(
        &self,
        x: &Tensor,
        d_out: usize,
        h_out: usize,
        w_out: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let c = cfg.out_visual_dim;
        let (pt, ph, pw) = (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]);
        let out_d = d_out * pt;
        let out_h = h_out * ph;
        let out_w = w_out * pw;

        let x_f32 = x.to_dtype(DType::F32)?;
        let x_data = x_f32.to_vec1::<f32>()?;
        let mut out = vec![0.0f32; out_d * out_h * out_w * c];

        // x layout: [D_out, H_out, W_out, C_out * p0 * p1 * p2]
        let out_patch_dim = c * pt * ph * pw;
        for di in 0..d_out {
            for hi in 0..h_out {
                for wi in 0..w_out {
                    let patch_idx = di * h_out * w_out + hi * w_out + wi;
                    // Python: view(D,H,W, C_out, p0, p1, p2)
                    // then permute(0,4,1,5,2,6,3)
                    // meaning output[d*pt+pti, h*ph+phi, w*pw+pwi, ci]
                    // = input[d, h, w, ci*pt*ph*pw + pti*ph*pw + phi*pw + pwi]
                    for ci in 0..c {
                        for pti in 0..pt {
                            for phi in 0..ph {
                                for pwi in 0..pw {
                                    let src_ch =
                                        ci * pt * ph * pw + pti * ph * pw + phi * pw + pwi;
                                    let val = x_data[patch_idx * out_patch_dim + src_ch];
                                    let dst_d = di * pt + pti;
                                    let dst_h = hi * ph + phi;
                                    let dst_w = wi * pw + pwi;
                                    // [D, H, W, C] output
                                    let dst_idx =
                                        dst_d * out_h * out_w * c + dst_h * out_w * c + dst_w * c + ci;
                                    out[dst_idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_f32_to_bf16(
            out,
            Shape::from_dims(&[out_d, out_h, out_w, c]),
            self.device.clone(),
        )
    }

    // -----------------------------------------------------------------------
    // Self-attention (shared between encoder and decoder blocks)
    // -----------------------------------------------------------------------

    /// Self-attention: Q/K/V projections → RMSNorm on Q,K → RoPE → SDPA → out proj.
    ///
    /// Input `x`: `[B, N, model_dim]`
    /// `pe_cos/pe_sin`: `[1, 1, N, head_dim/2]` for RoPE.
    /// Returns `[B, N, model_dim]`.
    fn self_attention(
        x: &Tensor,
        qw: &Tensor,
        qb: &Tensor,
        kw: &Tensor,
        kb: &Tensor,
        vw: &Tensor,
        vb: &Tensor,
        qnorm: &Tensor,
        knorm: &Tensor,
        ow: &Tensor,
        ob: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // Q, K, V projections
        let q = Self::linear_bias(x, qw, qb)?;
        let k = Self::linear_bias(x, kw, kb)?;
        let v = Self::linear_bias(x, vw, vb)?;

        // Reshape to [B, N, H, D]
        let q = q.reshape(&[b, n, num_heads, head_dim])?;
        let k = k.reshape(&[b, n, num_heads, head_dim])?;
        let v = v.reshape(&[b, n, num_heads, head_dim])?;

        // RMSNorm on Q, K (per-head, scale is [head_dim])
        let q = Self::rms_norm(&q, qnorm, eps)?;
        let k = Self::rms_norm(&k, knorm, eps)?;

        // Transpose to [B, H, N, D] for SDPA
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Apply RoPE
        let (q, k) = Self::apply_rope(&q, &k, pe_cos, pe_sin)?;

        // SDPA
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Transpose back and project out: [B, H, N, D] → [B, N, H*D]
        let attn_out = attn_out
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, n, num_heads * head_dim])?;
        Self::linear_bias(&attn_out, ow, ob)
    }

    /// Cross-attention: Q from visual, K/V from text. No RoPE on cross-attention.
    ///
    /// `x`: visual `[B, N_vis, model_dim]`
    /// `cond`: text `[B, N_txt, model_dim]`
    fn cross_attention(
        x: &Tensor,
        cond: &Tensor,
        qw: &Tensor,
        qb: &Tensor,
        kw: &Tensor,
        kb: &Tensor,
        vw: &Tensor,
        vb: &Tensor,
        qnorm: &Tensor,
        knorm: &Tensor,
        ow: &Tensor,
        ob: &Tensor,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let dims_x = x.shape().dims().to_vec();
        let dims_c = cond.shape().dims().to_vec();
        let (b, n_vis) = (dims_x[0], dims_x[1]);
        let n_txt = dims_c[1];

        let q = Self::linear_bias(x, qw, qb)?;
        let k = Self::linear_bias(cond, kw, kb)?;
        let v = Self::linear_bias(cond, vw, vb)?;

        let q = q.reshape(&[b, n_vis, num_heads, head_dim])?;
        let k = k.reshape(&[b, n_txt, num_heads, head_dim])?;
        let v = v.reshape(&[b, n_txt, num_heads, head_dim])?;

        // RMSNorm on Q, K
        let q = Self::rms_norm(&q, qnorm, eps)?;
        let k = Self::rms_norm(&k, knorm, eps)?;

        // Transpose to [B, H, N, D]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // No RoPE for cross-attention
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        let attn_out = attn_out
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, n_vis, num_heads * head_dim])?;
        Self::linear_bias(&attn_out, ow, ob)
    }

    /// FeedForward: Linear(dim, ff_dim, no bias) → GELU → Linear(ff_dim, dim, no bias)
    fn feed_forward(x: &Tensor, in_w: &Tensor, out_w: &Tensor) -> Result<Tensor> {
        let h = Self::linear_nobias(x, in_w)?;
        let h = h.gelu()?;
        Self::linear_nobias(&h, out_w)
    }

    // -----------------------------------------------------------------------
    // Text encoder block
    // -----------------------------------------------------------------------

    /// TransformerEncoderBlock forward.
    ///
    /// 6-param modulation: (shift, scale, gate) for self_attn + (shift, scale, gate) for FF.
    fn encoder_block_forward(
        &self,
        x: &Tensor,
        time_embed: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let nh = cfg.num_heads();
        let hd = cfg.head_dim();
        let eps = cfg.eps;
        let prefix = format!("text_transformer_blocks.{block_idx}");

        let s = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            self.shared.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        // Modulation: 6 params
        let mod_w = s("text_modulation.out_layer.weight")?;
        let mod_b = s("text_modulation.out_layer.bias")?;
        let mod_params = self.modulation(time_embed, mod_w, mod_b, 6)?;

        // Self-attention: shift=mod[0], scale=mod[1], gate=mod[2]
        let sa_modulated = Self::modulate_pre(x, &mod_params[0], &mod_params[1], eps)?;

        let sa_out = Self::self_attention(
            &sa_modulated,
            s("self_attention.to_query.weight")?,
            s("self_attention.to_query.bias")?,
            s("self_attention.to_key.weight")?,
            s("self_attention.to_key.bias")?,
            s("self_attention.to_value.weight")?,
            s("self_attention.to_value.bias")?,
            s("self_attention.query_norm.weight")?,
            s("self_attention.key_norm.weight")?,
            s("self_attention.out_layer.weight")?,
            s("self_attention.out_layer.bias")?,
            pe_cos,
            pe_sin,
            nh,
            hd,
            eps,
        )?;

        // Gated residual: x = x + gate * sa_out
        let gate_sa = mod_params[2].unsqueeze(1)?; // [B, 1, dim]
        let gated_sa = sa_out.mul(&gate_sa)?;
        let x = x.add(&gated_sa)?;

        // FeedForward: shift=mod[3], scale=mod[4], gate=mod[5]
        let ff_modulated = Self::modulate_pre(&x, &mod_params[3], &mod_params[4], eps)?;
        let ff_out = Self::feed_forward(
            &ff_modulated,
            s("feed_forward.in_layer.weight")?,
            s("feed_forward.out_layer.weight")?,
        )?;

        let gate_ff = mod_params[5].unsqueeze(1)?;
        let gated_ff = ff_out.mul(&gate_ff)?;
        x.add(&gated_ff)
    }

    // -----------------------------------------------------------------------
    // Visual decoder block
    // -----------------------------------------------------------------------

    /// TransformerDecoderBlock forward.
    ///
    /// 9-param modulation: self_attn(3) + cross_attn(3) + FF(3).
    fn decoder_block_forward(
        &self,
        visual_embed: &Tensor,
        text_embed: &Tensor,
        time_embed: &Tensor,
        vis_pe_cos: &Tensor,
        vis_pe_sin: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let nh = cfg.num_heads();
        let hd = cfg.head_dim();
        let eps = cfg.eps;
        let prefix = format!("visual_transformer_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        // Modulation: 9 params (3 for self_attn, 3 for cross_attn, 3 for FF)
        let mod_w = w("visual_modulation.out_layer.weight")?;
        let mod_b = w("visual_modulation.out_layer.bias")?;
        let mod_params = self.modulation(time_embed, mod_w, mod_b, 9)?;

        // ── Self-attention ──
        let sa_modulated =
            Self::modulate_pre(visual_embed, &mod_params[0], &mod_params[1], eps)?;

        let sa_out = Self::self_attention(
            &sa_modulated,
            w("self_attention.to_query.weight")?,
            w("self_attention.to_query.bias")?,
            w("self_attention.to_key.weight")?,
            w("self_attention.to_key.bias")?,
            w("self_attention.to_value.weight")?,
            w("self_attention.to_value.bias")?,
            w("self_attention.query_norm.weight")?,
            w("self_attention.key_norm.weight")?,
            w("self_attention.out_layer.weight")?,
            w("self_attention.out_layer.bias")?,
            vis_pe_cos,
            vis_pe_sin,
            nh,
            hd,
            eps,
        )?;

        let gate_sa = mod_params[2].unsqueeze(1)?;
        let gated_sa = sa_out.mul(&gate_sa)?;
        let visual_embed = visual_embed.add(&gated_sa)?;

        // ── Cross-attention ──
        let ca_modulated =
            Self::modulate_pre(&visual_embed, &mod_params[3], &mod_params[4], eps)?;

        let ca_out = Self::cross_attention(
            &ca_modulated,
            text_embed,
            w("cross_attention.to_query.weight")?,
            w("cross_attention.to_query.bias")?,
            w("cross_attention.to_key.weight")?,
            w("cross_attention.to_key.bias")?,
            w("cross_attention.to_value.weight")?,
            w("cross_attention.to_value.bias")?,
            w("cross_attention.query_norm.weight")?,
            w("cross_attention.key_norm.weight")?,
            w("cross_attention.out_layer.weight")?,
            w("cross_attention.out_layer.bias")?,
            nh,
            hd,
            eps,
        )?;

        let gate_ca = mod_params[5].unsqueeze(1)?;
        let gated_ca = ca_out.mul(&gate_ca)?;
        let visual_embed = visual_embed.add(&gated_ca)?;

        // ── FeedForward ──
        let ff_modulated =
            Self::modulate_pre(&visual_embed, &mod_params[6], &mod_params[7], eps)?;
        let ff_out = Self::feed_forward(
            &ff_modulated,
            w("feed_forward.in_layer.weight")?,
            w("feed_forward.out_layer.weight")?,
        )?;

        let gate_ff = mod_params[8].unsqueeze(1)?;
        let gated_ff = ff_out.mul(&gate_ff)?;
        visual_embed.add(&gated_ff)
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Full forward pass.
    ///
    /// ## Arguments
    /// - `x`: visual latents `[D, H, W, 16]` (duration, height, width, channels)
    /// - `text_embed`: Qwen text hidden states `[B, N_txt, 3584]`
    /// - `pooled_text_embed`: CLIP pooled text embedding `[B, 768]`
    /// - `time`: diffusion timestep `[B]`
    /// - `visual_rope_pos`: 3 position index tensors `[D_out]`, `[H_out]`, `[W_out]`
    /// - `text_rope_pos`: text position indices `[N_txt]`
    /// - `scale_factor`: per-axis RoPE scale factors `[f32; 3]`
    ///
    /// Returns: `[D, H, W, 16]` denoised prediction.
    pub fn forward(
        &mut self,
        x: &Tensor,
        text_embed: &Tensor,
        pooled_text_embed: &Tensor,
        time: &Tensor,
        visual_rope_pos: &[Tensor; 3],
        text_rope_pos: &Tensor,
        scale_factor: [f32; 3],
    ) -> Result<Tensor> {
        let cfg = self.config.clone();

        // ── 1. Embeddings ──
        let mut text_embed = self.text_embedding(text_embed)?;
        let time_embed = self.time_embedding(time)?;
        let pooled_embed = self.pooled_text_embedding(pooled_text_embed)?;
        // time_embed = time_embed + pooled_text_embed
        let time_embed = time_embed.add(&pooled_embed)?;
        let (visual_embed, d_out, h_out, w_out) = self.visual_embedding(x)?;

        // ── 2. Text RoPE + encoder blocks ──
        let (text_cos, text_sin) = self.build_text_rope(text_rope_pos)?;

        for i in 0..cfg.num_text_blocks {
            text_embed =
                self.encoder_block_forward(&text_embed, &time_embed, &text_cos, &text_sin, i)?;
            log::info!(
                "[Kandinsky5] Text encoder block {}/{}",
                i + 1,
                cfg.num_text_blocks
            );
        }

        // ── 3. Visual RoPE + flatten ──
        let (vis_cos, vis_sin) =
            self.build_visual_rope(d_out, h_out, w_out, visual_rope_pos, scale_factor)?;

        // Flatten visual: [D_out, H_out, W_out, model_dim] → [1, D*H*W, model_dim]
        let n_patches = d_out * h_out * w_out;
        let mut visual_flat = visual_embed.reshape(&[1, n_patches, cfg.model_dim])?;

        // ── 4. Visual decoder blocks (FlameSwap) ──
        let total_blocks = cfg.num_visual_blocks;
        self.swap
            .prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..total_blocks {
            let raw = self
                .swap
                .await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.swap
                    .prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            visual_flat = self.decoder_block_forward(
                &visual_flat,
                &text_embed,
                &time_embed,
                &vis_cos,
                &vis_sin,
                &raw,
                i,
            )?;

            if i % 8 == 0 || i == total_blocks - 1 {
                log::info!("[Kandinsky5] Visual decoder block {}/{}", i + 1, total_blocks);
            }
        }

        // ── 5. OutLayer ──
        // Unflatten: [1, D*H*W, model_dim] → [D_out, H_out, W_out, model_dim]
        let visual_spatial = visual_flat.reshape(&[d_out, h_out, w_out, cfg.model_dim])?;
        let output = self.out_layer(&visual_spatial, &time_embed, d_out, h_out, w_out)?;

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // OutLayer
    // -----------------------------------------------------------------------

    /// OutLayer: modulation(2) → LayerNorm(no affine) → modulate → Linear → unpatchify.
    ///
    /// Input: `[D_out, H_out, W_out, model_dim]`
    /// Output: `[D, H, W, out_visual_dim]`
    fn out_layer(
        &self,
        visual_embed: &Tensor,
        time_embed: &Tensor,
        d_out: usize,
        h_out: usize,
        w_out: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;

        // Modulation: 2 params (shift, scale)
        let mod_w = self.get_shared("out_layer.modulation.out_layer.weight")?;
        let mod_b = self.get_shared("out_layer.modulation.out_layer.bias")?;
        let mod_params = self.modulation(time_embed, mod_w, mod_b, 2)?;

        // shift, scale are [B, model_dim]. For the spatial tensor [D, H, W, dim],
        // we broadcast shift/scale over all spatial positions.
        // Python: scale[:, None, None], shift[:, None, None] — broadcasting over D, H, W dims.
        // Since our visual_embed is [D, H, W, dim], reshape to [D*H*W, dim] to use layer_norm,
        // then apply modulation, then reshape back.
        let n = d_out * h_out * w_out;
        let flat = visual_embed.reshape(&[n, cfg.model_dim])?;

        // LayerNorm (no affine)
        let normed = Self::layer_norm_no_affine(&flat, cfg.eps)?;

        // Modulate: normed * (1 + scale) + shift
        // scale/shift are [B, dim] — for batch=1, just [1, dim]
        let one_plus_scale = mod_params[1].add_scalar(1.0)?; // [B, dim]
        let normed_3d = normed.unsqueeze(0)?; // [1, n, dim]
        let scaled = normed_3d.mul(&one_plus_scale.unsqueeze(1)?)?;
        let shifted = scaled.add(&mod_params[0].unsqueeze(1)?)?;
        let modulated = shifted.squeeze(Some(0))?; // [n, dim]

        // Linear → [n, out_patch_dim]
        let out_w = self.get_shared("out_layer.out_layer.weight")?;
        let out_b = self.get_shared("out_layer.out_layer.bias")?;
        let modulated_3d = modulated.unsqueeze(0)?; // [1, n, dim]
        let projected = Self::linear_bias(&modulated_3d, out_w, out_b)?;
        let projected = projected.squeeze(Some(0))?; // [n, out_patch_dim]

        // Reshape to [D_out, H_out, W_out, out_patch_dim] then unpatchify
        let spatial = projected.reshape(&[d_out, h_out, w_out, cfg.out_patch_dim()])?;
        self.unpatchify(&spatial, d_out, h_out, w_out)
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    fn get_shared(&self, key: &str) -> Result<&Tensor> {
        self.shared
            .get(key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing shared weight: {key}")))
    }
}

// ---------------------------------------------------------------------------
// Module-level helper: build mod_proj_buffer is not needed for Kandinsky5
// (unlike Chroma's approximator). Silence unused imports.
// ---------------------------------------------------------------------------

/// Precompute `get_freqs` for a given dim and max_period.
///
/// `freqs[i] = exp(-ln(max_period) * i / dim)` for `i in 0..dim`.
///
/// This matches the Python `get_freqs` from `kandinsky/models/utils.py`.
#[allow(dead_code)]
fn get_freqs(dim: usize, max_period: f64) -> Vec<f32> {
    (0..dim)
        .map(|i| (-(max_period.ln()) * i as f64 / dim as f64).exp() as f32)
        .collect()
}
