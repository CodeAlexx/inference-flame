//! FLUX 1 Dev/Schnell DiT transformer — pure flame-core with FlameSwap.
//!
//! 12B parameter model: 19 double blocks + 38 single blocks.
//!
//! Key differences from Klein (flux2):
//!   - HAS biases everywhere
//!   - PER-BLOCK modulation (not shared)
//!   - GELU MLP (4x ratio) instead of SwiGLU (6x ratio)
//!   - Has guidance_in (Dev only) + vector_in (CLIP pooled, 768→3072)
//!   - in_channels=64, joint_attention_dim=4096 (T5)
//!   - 3-axis RoPE (16,56,56) with complex multiply (not halfsplit)
//!
//! ## Weight key format (BFL flux1-dev.safetensors):
//!   img_in.weight/bias                         [3072, 64] / [3072]
//!   txt_in.weight/bias                         [3072, 4096] / [3072]
//!   time_in.in_layer.weight/bias               [3072, 256] / [3072]
//!   time_in.out_layer.weight/bias              [3072, 3072] / [3072]
//!   guidance_in.in_layer.weight/bias           (same shape, Dev only)
//!   guidance_in.out_layer.weight/bias
//!   vector_in.in_layer.weight/bias             [3072, 768] / [3072]
//!   vector_in.out_layer.weight/bias            [3072, 3072] / [3072]
//!   double_blocks.{i}.img_mod.lin.weight/bias  [18432, 3072] = 6*3072
//!   double_blocks.{i}.img_attn.qkv.weight/bias [9216, 3072] = 3*3072
//!   double_blocks.{i}.img_attn.proj.weight/bias [3072, 3072]
//!   double_blocks.{i}.img_attn.norm.{query,key}_norm.scale [128]
//!   double_blocks.{i}.img_mlp.{0,2}.weight/bias  GELU MLP
//!   double_blocks.{i}.txt_*  (mirrors img_*)
//!   single_blocks.{i}.modulation.lin.weight/bias [9216, 3072] = 3*3072
//!   single_blocks.{i}.linear1.weight/bias        [21504, 3072] = 7*3072
//!   single_blocks.{i}.linear2.weight/bias        [3072, 15360] = 5*3072
//!   single_blocks.{i}.norm.{query,key}_norm.scale [128]
//!   final_layer.adaLN_modulation.1.weight/bias   [6144, 3072]
//!   final_layer.linear.weight/bias                [64, 3072]
//!
//! ⚠️ STANDALONE — does NOT connect to any inference pipeline.

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
pub struct Flux1Config {
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    pub inner_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub mlp_ratio: f32,
    pub timestep_dim: usize,
    pub has_guidance: bool,
    pub vector_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f64,
}

impl Default for Flux1Config {
    fn default() -> Self {
        Self {
            num_double_blocks: 19,
            num_single_blocks: 38,
            inner_dim: 3072,
            num_heads: 24,
            head_dim: 128,
            in_channels: 64,
            joint_attention_dim: 4096,
            mlp_ratio: 4.0,
            timestep_dim: 256,
            has_guidance: true,
            vector_dim: 768,
            axes_dims_rope: [16, 56, 56],
            rope_theta: 10000.0,
        }
    }
}

impl Flux1Config {
    pub fn dev() -> Self {
        Self { has_guidance: true, ..Self::default() }
    }
    pub fn schnell() -> Self {
        Self { has_guidance: false, ..Self::default() }
    }
}

// ---------------------------------------------------------------------------
// FLUX 1 DiT with FlameSwap
// ---------------------------------------------------------------------------

/// FLUX 1 Dev/Schnell DiT transformer.
///
/// Double blocks (0..18) and single blocks (0..37) are managed by two
/// separate FlameSwap instances (or combined into one with offset indexing).
/// Shared weights (projections, embeddings, final_layer) stay on GPU.
pub struct Flux1DiT {
    shared: HashMap<String, Tensor>,
    swap: FlameSwap,
    config: Flux1Config,
    device: Arc<CudaDevice>,
}

impl Flux1DiT {
    /// Load FLUX 1 from BFL single-file safetensors.
    pub fn load(
        checkpoint_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = Flux1Config::dev(); // auto-detect later from keys

        // FlameSwap: classify blocks.
        // double_blocks.N → block N
        // single_blocks.N → block (num_double + N)
        let num_double = config.num_double_blocks;
        let swap = FlameSwap::load(
            &[checkpoint_path],
            device,
            |name| {
                if let Some(rest) = name.strip_prefix("double_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(idx);
                }
                if let Some(rest) = name.strip_prefix("single_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    return Some(num_double + idx);
                }
                None
            },
        ).map_err(|e| flame_core::Error::InvalidInput(format!("FlameSwap Flux1: {e}")))?;

        // Load shared (non-block) weights
        let shared_prefixes = [
            "img_in.", "txt_in.", "time_in.", "guidance_in.", "vector_in.", "final_layer.",
        ];
        let shared_weights = load_file_filtered(
            Path::new(checkpoint_path),
            device,
            |key| shared_prefixes.iter().any(|p| key.starts_with(p)),
        )?;

        log::info!("[Flux1] Loaded: {} blocks via FlameSwap, {} shared weights",
            swap.num_blocks(), shared_weights.len());

        // Log shared weight keys
        for key in shared_weights.keys() {
            log::info!("[Flux1] Shared: {} {:?}", key, shared_weights[key].shape());
        }

        Ok(Self {
            shared: shared_weights,
            swap,
            config,
            device: device.clone(),
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        let shape = x.shape().dims().to_vec();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x_2d = x.reshape(&[b * n, c])?;
        let out = x_2d.matmul(&wt)?;
        let out_dim = out.shape().dims()[1];
        let out = out.add(&bias.unsqueeze(0)?.expand(&[b * n, out_dim])?)?;
        out.reshape(&[b, n, out_dim])
    }

    fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        let shape = x.shape().dims().to_vec();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x_2d = x.reshape(&[b * n, c])?;
        let out = x_2d.matmul(&wt)?;
        let out_dim = out.shape().dims()[1];
        out.reshape(&[b, n, out_dim])
    }

    /// RMSNorm: standard (not Gemma formulation).
    fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
        out.reshape(&dims)
    }

    /// LayerNorm (no affine) — used in modulate_pre.
    fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, eps)?;
        out.reshape(&dims)
    }

    /// Modulate: (1 + scale) * LayerNorm(x) + shift
    ///
    /// ## PyTorch reference:
    /// ```python
    /// x_normed = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6).to(dtype)
    /// return x_normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    /// ```
    fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let normed = Self::layer_norm_no_affine(x, 1e-6)?;
        let one_plus_scale = scale.add_scalar(1.0)?;
        let scaled = normed.mul(&one_plus_scale.unsqueeze(1)?)?;
        scaled.add(&shift.unsqueeze(1)?)
    }

    /// Sinusoidal timestep embedding.
    ///
    /// ## PyTorch reference:
    /// ```python
    /// t = time_factor * t  # time_factor=1000
    /// freqs = exp(-ln(10000) * arange(half) / half)
    /// args = t[:, None] * freqs[None]
    /// return cat([cos(args), sin(args)], dim=-1)
    /// ```
    fn timestep_embedding(t: &Tensor, dim: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let t_scaled = t.mul_scalar(1000.0)?; // time_factor
        let half = dim / 2;
        let max_period = 10000.0f64;
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), device.clone())?
            .to_dtype(DType::BF16)?;

        let t_col = t_scaled.unsqueeze(1)?; // [B, 1]
        let args = t_col.matmul(&freqs)?; // [B, half]
        let cos = args.cos()?;
        let sin = args.sin()?;
        Tensor::cat(&[&cos, &sin], 1) // [B, dim]
    }

    /// MLP: silu(in_layer(x)) → out_layer
    fn timestep_mlp(x: &Tensor, in_w: &Tensor, in_b: &Tensor, out_w: &Tensor, out_b: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let x_2d = if shape.len() == 2 { x.clone() } else { x.reshape(&[shape[0], shape[1]])? };

        let in_wt = flame_core::bf16_elementwise::transpose2d_bf16(in_w)?;
        let h = x_2d.matmul(&in_wt)?;
        let h = h.add(&in_b.unsqueeze(0)?.expand(&h.shape().dims().to_vec())?)?;
        let h = h.silu()?;

        let out_wt = flame_core::bf16_elementwise::transpose2d_bf16(out_w)?;
        let out = h.matmul(&out_wt)?;
        out.add(&out_b.unsqueeze(0)?.expand(&out.shape().dims().to_vec())?)
    }

    // -----------------------------------------------------------------------
    // RoPE (complex multiply, 3-axis)
    // -----------------------------------------------------------------------

    /// Build 3-axis RoPE for FLUX 1.
    ///
    /// img_ids: [N_img, 3] — (t, h, w) positions for image patches
    /// txt_ids: [N_txt, 3] — all zeros for text
    /// Returns: [1, 1, N_txt+N_img, head_dim/2] complex freqs
    pub fn build_rope_2d(
        img_ids: &Tensor,
        txt_ids: &Tensor,
        config: &Flux1Config,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        // Concatenate: [txt_ids; img_ids] along dim 0
        let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?; // [N, 3]
        let all_ids_f32 = all_ids.to_dtype(DType::F32)?;

        let n_total = all_ids.shape().dims()[0];
        let mut all_cos = Vec::new();
        let mut all_sin = Vec::new();

        for (axis, &dim) in config.axes_dims_rope.iter().enumerate() {
            let half = dim / 2;
            // Extract positions for this axis: all_ids[:, axis]
            let pos = all_ids_f32.narrow(1, axis, 1)?.squeeze(Some(1))?; // [N]

            // Frequencies
            let freq_data: Vec<f32> = (0..half)
                .map(|i| 1.0 / (config.rope_theta as f32).powf(2.0 * i as f32 / dim as f32))
                .collect();
            let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), device.clone())?;

            // Outer product: [N, 1] * [1, half] = [N, half]
            let pos_col = pos.unsqueeze(1)?;
            let angles = pos_col.matmul(&freqs)?; // [N, half]

            all_cos.push(angles.cos()?);
            all_sin.push(angles.sin()?);
        }

        // Concatenate across axes: [N, sum(half_dims)]
        let cos_refs: Vec<&Tensor> = all_cos.iter().collect();
        let sin_refs: Vec<&Tensor> = all_sin.iter().collect();
        let cos_cat = Tensor::cat(&cos_refs, 1)?; // [N, 64] (16/2 + 56/2 + 56/2)
        let sin_cat = Tensor::cat(&sin_refs, 1)?;

        // Stack as [N, 64, 2] for complex representation, then reshape
        // But flame_core doesn't have complex ops. We'll return cos and sin separately.
        // Actually, the Python code uses torch.polar(ones, angles) → complex multiply.
        // For our SDPA path, we apply RoPE differently — we'll store cos/sin
        // as [1, 1, N, half] and use a rotation approach.
        let cos_out = cos_cat.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let sin_out = sin_cat.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;

        // Return concatenated [cos; sin] as [1, 1, N, 2*half] or two separate tensors.
        // We'll pack them into one tensor: [1, 1, N, half, 2] for the complex representation.
        // For now, return cos for the caller to handle both.
        // Actually, let's just stack them: [2, 1, 1, N, 64]
        Tensor::stack(&[cos_out, sin_out], 0) // [2, 1, 1, N, 64]
    }

    /// Apply complex RoPE to q and k.
    ///
    /// pe: [2, 1, 1, N, half] where pe[0]=cos, pe[1]=sin.
    /// q, k: [B, H, N, D] where D=128.
    ///
    /// Complex multiply: (q_re + i*q_im) * (cos + i*sin)
    ///   = (q_re*cos - q_im*sin) + i*(q_re*sin + q_im*cos)
    fn apply_rope_complex(q: &Tensor, k: &Tensor, pe: &Tensor) -> Result<(Tensor, Tensor)> {
        let pe_cos = pe.narrow(0, 0, 1)?.squeeze(Some(0))?; // [1, 1, N, 64]
        let pe_sin = pe.narrow(0, 1, 1)?.squeeze(Some(0))?; // [1, 1, N, 64]

        let q_rot = Self::rotate_complex(q, &pe_cos, &pe_sin)?;
        let k_rot = Self::rotate_complex(k, &pe_cos, &pe_sin)?;
        Ok((q_rot, k_rot))
    }

    /// Rotate a tensor using complex multiplication.
    /// x: [B, H, N, D] where D is split into (D/2, 2) as (real, imag).
    fn rotate_complex(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let d = dims[3]; // 128
        let half = d / 2; // 64

        // Split x into even and odd elements (real and imaginary parts)
        // x_re = x[..., 0::2], x_im = x[..., 1::2]
        let x_re = x.narrow(3, 0, half)?; // Take first half
        let x_im = x.narrow(3, half, half)?; // Take second half

        // Wait — FLUX uses view_as_complex which interleaves: [re0, im0, re1, im1, ...]
        // So we need stride-2 access. Let's reshape instead:
        // x: [B, H, N, D] → [B, H, N, D/2, 2]
        let (b, h, n, _d) = (dims[0], dims[1], dims[2], dims[3]);
        let x_pairs = x.reshape(&[b, h, n, half, 2])?;
        let x_re = x_pairs.narrow(4, 0, 1)?.squeeze(Some(4))?; // [B, H, N, half]
        let x_im = x_pairs.narrow(4, 1, 1)?.squeeze(Some(4))?; // [B, H, N, half]

        // Complex multiply: out_re = x_re*cos - x_im*sin
        //                   out_im = x_re*sin + x_im*cos
        let out_re = x_re.mul(cos)?.sub(&x_im.mul(sin)?)?;
        let out_im = x_re.mul(sin)?.add(&x_im.mul(cos)?)?;

        // Interleave back: [B, H, N, half, 2]
        let out_re_exp = out_re.unsqueeze(4)?;
        let out_im_exp = out_im.unsqueeze(4)?;
        let interleaved = Tensor::cat(&[&out_re_exp, &out_im_exp], 4)?; // [B, H, N, half, 2]
        interleaved.reshape(&[b, h, n, d])
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Forward pass.
    ///
    /// ## Arguments
    /// - `img`: packed image latents [B, N_img, 64]
    /// - `txt`: T5 hidden states [B, N_txt, 4096]
    /// - `timesteps`: sigma values [B]
    /// - `img_ids`: image position IDs [N_img, 3]
    /// - `txt_ids`: text position IDs [N_txt, 3]
    /// - `guidance`: guidance scale [B] (Dev only, None for Schnell)
    /// - `vector`: CLIP pooled [B, 768]
    pub fn forward(
        &mut self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        guidance: Option<&Tensor>,
        vector: Option<&Tensor>,
    ) -> Result<Tensor> {
        let cfg = self.config.clone();
        let dim = cfg.inner_dim;

        let img_len = img.shape().dims()[1];
        let txt_len = txt.shape().dims()[1];

        // --- Input projections ---
        let img = Self::linear_bias(
            img,
            self.shared.get("img_in.weight").ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.weight".into()))?,
            self.shared.get("img_in.bias").ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.bias".into()))?,
        )?;
        let txt = Self::linear_bias(
            txt,
            self.shared.get("txt_in.weight").ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.weight".into()))?,
            self.shared.get("txt_in.bias").ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.bias".into()))?,
        )?;

        // --- Timestep + guidance + vector embeddings ---
        let t_emb = Self::timestep_embedding(timesteps, cfg.timestep_dim, &self.device)?;
        let mut vec = Self::timestep_mlp(
            &t_emb,
            self.shared.get("time_in.in_layer.weight").unwrap(),
            self.shared.get("time_in.in_layer.bias").unwrap(),
            self.shared.get("time_in.out_layer.weight").unwrap(),
            self.shared.get("time_in.out_layer.bias").unwrap(),
        )?;

        if let Some(g) = guidance {
            if let (Some(gw1), Some(gb1), Some(gw2), Some(gb2)) = (
                self.shared.get("guidance_in.in_layer.weight"),
                self.shared.get("guidance_in.in_layer.bias"),
                self.shared.get("guidance_in.out_layer.weight"),
                self.shared.get("guidance_in.out_layer.bias"),
            ) {
                let g_emb = Self::timestep_embedding(g, cfg.timestep_dim, &self.device)?;
                let g_vec = Self::timestep_mlp(&g_emb, gw1, gb1, gw2, gb2)?;
                vec = vec.add(&g_vec)?;
            }
        }

        if let Some(v) = vector {
            let v_vec = Self::timestep_mlp(
                v,
                self.shared.get("vector_in.in_layer.weight").unwrap(),
                self.shared.get("vector_in.in_layer.bias").unwrap(),
                self.shared.get("vector_in.out_layer.weight").unwrap(),
                self.shared.get("vector_in.out_layer.bias").unwrap(),
            )?;
            vec = vec.add(&v_vec)?;
        }

        // --- Build RoPE ---
        let pe = Self::build_rope_2d(img_ids, txt_ids, &cfg, &self.device)?;

        // --- Double blocks ---
        let mut img = img;
        let mut txt = txt;
        let total_blocks = cfg.num_double_blocks + cfg.num_single_blocks;

        self.swap.prefetch(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;

        for i in 0..cfg.num_double_blocks {
            let raw = self.swap.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < total_blocks {
                self.swap.prefetch(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            let (new_img, new_txt) = self.double_block_forward(
                &img, &txt, &vec, &pe, &raw, i,
            )?;
            img = new_img;
            txt = new_txt;

            if i % 5 == 0 || i == cfg.num_double_blocks - 1 {
                log::info!("[Flux1] Double block {}/{}", i + 1, cfg.num_double_blocks);
            }
        }

        // --- Merge txt + img for single blocks ---
        let mut x = Tensor::cat(&[&txt, &img], 1)?; // [B, N_txt+N_img, dim]

        // --- Single blocks ---
        for i in 0..cfg.num_single_blocks {
            let block_idx = cfg.num_double_blocks + i;
            let raw = self.swap.await_block(block_idx)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if block_idx + 1 < total_blocks {
                self.swap.prefetch(block_idx + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            x = self.single_block_forward(&x, &vec, &pe, txt_len, &raw, i)?;

            if i % 10 == 0 || i == cfg.num_single_blocks - 1 {
                log::info!("[Flux1] Single block {}/{}", i + 1, cfg.num_single_blocks);
            }
        }

        // --- Extract image portion and apply final layer ---
        let img_out = x.narrow(1, txt_len, img_len)?;
        self.final_layer_forward(&img_out, &vec)
    }

    // -----------------------------------------------------------------------
    // Block forwards
    // -----------------------------------------------------------------------

    fn double_block_forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
        pe: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let prefix = format!("double_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let dims_img = img.shape().dims().to_vec();
        let dims_txt = txt.shape().dims().to_vec();
        let (b, n_img) = (dims_img[0], dims_img[1]);
        let n_txt = dims_txt[1];

        // Per-block modulation: 6 mods each for img and txt
        let img_mod_w = w("img_mod.lin.weight")?;
        let img_mod_b = w("img_mod.lin.bias")?;
        let txt_mod_w = w("txt_mod.lin.weight")?;
        let txt_mod_b = w("txt_mod.lin.bias")?;

        let vec_act = vec.silu()?;
        let img_mods = Self::linear_bias(&vec_act.unsqueeze(1)?, img_mod_w, img_mod_b)?
            .squeeze(Some(1))?; // [B, 6*dim]
        let txt_mods = Self::linear_bias(&vec_act.unsqueeze(1)?, txt_mod_w, txt_mod_b)?
            .squeeze(Some(1))?;

        // Split into 6 mod vectors
        let chunk_size = cfg.inner_dim;
        let img_shift1 = img_mods.narrow(1, 0, chunk_size)?;
        let img_scale1 = img_mods.narrow(1, chunk_size, chunk_size)?;
        let img_gate1 = img_mods.narrow(1, 2 * chunk_size, chunk_size)?;
        let img_shift2 = img_mods.narrow(1, 3 * chunk_size, chunk_size)?;
        let img_scale2 = img_mods.narrow(1, 4 * chunk_size, chunk_size)?;
        let img_gate2 = img_mods.narrow(1, 5 * chunk_size, chunk_size)?;

        let txt_shift1 = txt_mods.narrow(1, 0, chunk_size)?;
        let txt_scale1 = txt_mods.narrow(1, chunk_size, chunk_size)?;
        let txt_gate1 = txt_mods.narrow(1, 2 * chunk_size, chunk_size)?;
        let txt_shift2 = txt_mods.narrow(1, 3 * chunk_size, chunk_size)?;
        let txt_scale2 = txt_mods.narrow(1, 4 * chunk_size, chunk_size)?;
        let txt_gate2 = txt_mods.narrow(1, 5 * chunk_size, chunk_size)?;

        // Modulate + QKV
        let img_normed = Self::modulate_pre(img, &img_shift1, &img_scale1)?;
        let txt_normed = Self::modulate_pre(txt, &txt_shift1, &txt_scale1)?;

        let img_qkv = Self::linear_bias(&img_normed, w("img_attn.qkv.weight")?, w("img_attn.qkv.bias")?)?;
        let txt_qkv = Self::linear_bias(&txt_normed, w("txt_attn.qkv.weight")?, w("txt_attn.qkv.bias")?)?;

        // Reshape QKV: [B, N, 3*dim] → 3x [B, H, N, D]
        let (img_q, img_k, img_v) = Self::split_qkv(&img_qkv, b, n_img, h, d)?;
        let (txt_q, txt_k, txt_v) = Self::split_qkv(&txt_qkv, b, n_txt, h, d)?;

        // QK norm
        let img_q = Self::rms_norm(&img_q, w("img_attn.norm.query_norm.scale")?, 1e-6)?;
        let img_k = Self::rms_norm(&img_k, w("img_attn.norm.key_norm.scale")?, 1e-6)?;
        let txt_q = Self::rms_norm(&txt_q, w("txt_attn.norm.query_norm.scale")?, 1e-6)?;
        let txt_k = Self::rms_norm(&txt_k, w("txt_attn.norm.key_norm.scale")?, 1e-6)?;

        // Joint attention: concat then apply RoPE
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?; // [B, H, N_txt+N_img, D]
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

        let (q, k) = Self::apply_rope_complex(&q, &k, pe)?;

        // SDPA
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Split back
        let txt_attn = attn_out.narrow(2, 0, n_txt)?;
        let img_attn = attn_out.narrow(2, n_txt, n_img)?;

        let img_attn = img_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img, h * d])?;
        let txt_attn = txt_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_txt, h * d])?;

        let img_attn = Self::linear_bias(&img_attn, w("img_attn.proj.weight")?, w("img_attn.proj.bias")?)?;
        let txt_attn = Self::linear_bias(&txt_attn, w("txt_attn.proj.weight")?, w("txt_attn.proj.bias")?)?;

        // Gated residual
        let img = img.add(&img_gate1.unsqueeze(1)?.mul(&img_attn)?)?;
        let txt = txt.add(&txt_gate1.unsqueeze(1)?.mul(&txt_attn)?)?;

        // MLP with modulation
        let img_mlp_in = Self::modulate_pre(&img, &img_shift2, &img_scale2)?;
        let txt_mlp_in = Self::modulate_pre(&txt, &txt_shift2, &txt_scale2)?;

        // GELU MLP: Linear → GELU → Linear
        let img_mlp = Self::linear_bias(&img_mlp_in, w("img_mlp.0.weight")?, w("img_mlp.0.bias")?)?;
        let img_mlp = img_mlp.gelu()?;
        let img_mlp = Self::linear_bias(&img_mlp, w("img_mlp.2.weight")?, w("img_mlp.2.bias")?)?;

        let txt_mlp = Self::linear_bias(&txt_mlp_in, w("txt_mlp.0.weight")?, w("txt_mlp.0.bias")?)?;
        let txt_mlp = txt_mlp.gelu()?;
        let txt_mlp = Self::linear_bias(&txt_mlp, w("txt_mlp.2.weight")?, w("txt_mlp.2.bias")?)?;

        let img = img.add(&img_gate2.unsqueeze(1)?.mul(&img_mlp)?)?;
        let txt = txt.add(&txt_gate2.unsqueeze(1)?.mul(&txt_mlp)?)?;

        Ok((img, txt))
    }

    fn single_block_forward(
        &self,
        x: &Tensor,
        vec: &Tensor,
        pe: &Tensor,
        txt_len: usize,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let dim = cfg.inner_dim;
        let mlp_hidden = (dim as f32 * cfg.mlp_ratio) as usize;
        let prefix = format!("single_blocks.{block_idx}");

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let dims = x.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // Modulation (3 mods)
        let vec_act = vec.silu()?;
        let mods = Self::linear_bias(&vec_act.unsqueeze(1)?, w("modulation.lin.weight")?, w("modulation.lin.bias")?)?
            .squeeze(Some(1))?;
        let shift = mods.narrow(1, 0, dim)?;
        let scale = mods.narrow(1, dim, dim)?;
        let gate = mods.narrow(1, 2 * dim, dim)?;

        let x_normed = Self::modulate_pre(x, &shift, &scale)?;

        // Fused linear1: QKV + MLP_up
        let qkv_mlp = Self::linear_bias(&x_normed, w("linear1.weight")?, w("linear1.bias")?)?;
        let qkv = qkv_mlp.narrow(2, 0, 3 * dim)?;
        let mlp_in = qkv_mlp.narrow(2, 3 * dim, mlp_hidden)?;

        let (q, k, v) = Self::split_qkv(&qkv, b, n, h, d)?;
        let q = Self::rms_norm(&q, w("norm.query_norm.scale")?, 1e-6)?;
        let k = Self::rms_norm(&k, w("norm.key_norm.scale")?, 1e-6)?;

        let (q, k) = Self::apply_rope_complex(&q, &k, pe)?;

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        // GELU on MLP path
        let mlp_out = mlp_in.gelu()?;

        // Fused linear2: concat attn_proj + mlp_down
        let fused_in = Tensor::cat(&[&attn_out, &mlp_out], 2)?; // [B, N, dim+mlp_hidden]
        let out = Self::linear_bias(&fused_in, w("linear2.weight")?, w("linear2.bias")?)?;

        x.add(&gate.unsqueeze(1)?.mul(&out)?)
    }

    fn final_layer_forward(&self, x: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let adaLN_w = self.shared.get("final_layer.adaLN_modulation.1.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer.adaLN_modulation.1.weight".into()))?;
        let adaLN_b = self.shared.get("final_layer.adaLN_modulation.1.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer.adaLN_modulation.1.bias".into()))?;

        let vec_act = vec.silu()?;
        let mods = Self::linear_bias(&vec_act.unsqueeze(1)?, adaLN_w, adaLN_b)?.squeeze(Some(1))?;
        let dim = self.config.inner_dim;
        let shift = mods.narrow(1, 0, dim)?;
        let scale = mods.narrow(1, dim, dim)?;

        let x = Self::modulate_pre(x, &shift, &scale)?;

        let out_w = self.shared.get("final_layer.linear.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer.linear.weight".into()))?;
        let out_b = self.shared.get("final_layer.linear.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer.linear.bias".into()))?;

        Self::linear_bias(&x, out_w, out_b)
    }

    // -----------------------------------------------------------------------
    // QKV split helper
    // -----------------------------------------------------------------------

    fn split_qkv(qkv: &Tensor, b: usize, n: usize, h: usize, d: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let reshaped = qkv.reshape(&[b, n, 3, h, d])?;
        let permuted = reshaped.permute(&[2, 0, 3, 1, 4])?; // [3, B, H, N, D]
        let q = permuted.narrow(0, 0, 1)?.squeeze(Some(0))?;
        let k = permuted.narrow(0, 1, 1)?.squeeze(Some(0))?;
        let v = permuted.narrow(0, 2, 1)?.squeeze(Some(0))?;
        Ok((q, k, v))
    }

    pub fn config(&self) -> &Flux1Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux1_config() {
        let cfg = Flux1Config::dev();
        assert_eq!(cfg.num_double_blocks, 19);
        assert_eq!(cfg.num_single_blocks, 38);
        assert_eq!(cfg.inner_dim, 3072);
        assert_eq!(cfg.num_heads * cfg.head_dim, 3072);
        assert_eq!(cfg.in_channels, 64);
        assert!(cfg.has_guidance);
    }

    #[test]
    fn test_schnell_config() {
        let cfg = Flux1Config::schnell();
        assert!(!cfg.has_guidance);
    }

    #[test]
    fn test_rope_dims() {
        let cfg = Flux1Config::default();
        let total_rope: usize = cfg.axes_dims_rope.iter().map(|d| d / 2).sum();
        assert_eq!(total_rope, 64); // 8 + 28 + 28 = 64 = head_dim/2
    }
}
