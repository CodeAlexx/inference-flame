//! FLUX 1 Dev/Schnell DiT transformer — pure flame-core with BlockOffloader.
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
use flame_diffusion::block_offload::BlockFacilitator;
use flame_diffusion::BlockOffloader;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BlockFacilitator for FLUX 1:
//   double_blocks.{i} → block i
//   single_blocks.{i} → block (num_double + i)
// ---------------------------------------------------------------------------

struct Flux1Facilitator {
    num_double: usize,
    total_blocks: usize,
}

impl BlockFacilitator for Flux1Facilitator {
    fn block_count(&self) -> usize { self.total_blocks }
    fn classify_key(&self, key: &str) -> Option<usize> {
        if let Some(rest) = key.strip_prefix("double_blocks.") {
            let idx: usize = rest.split('.').next()?.parse().ok()?;
            return Some(idx);
        }
        if let Some(rest) = key.strip_prefix("single_blocks.") {
            let idx: usize = rest.split('.').next()?.parse().ok()?;
            return Some(self.num_double + idx);
        }
        None
    }
}

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
// FLUX 1 DiT with BlockOffloader
// ---------------------------------------------------------------------------

/// FLUX 1 Dev/Schnell DiT transformer.
///
/// Double blocks (0..18) and single blocks (0..37) are managed by
/// BlockOffloader with offset indexing.
/// Shared weights (projections, embeddings, final_layer) stay on GPU.
pub struct Flux1DiT {
    shared: HashMap<String, Tensor>,
    offloader: BlockOffloader,
    config: Flux1Config,
    device: Arc<CudaDevice>,
}

impl Flux1DiT {
    /// Load FLUX 1 from BFL single-file safetensors.
    pub fn load(
        checkpoint_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // H3 fix: default config will be patched after we see shared weight keys.
        let mut config = Flux1Config::dev();

        // BlockOffloader: classify blocks.
        // double_blocks.N → block N
        // single_blocks.N → block (num_double + N)
        let num_double = config.num_double_blocks;
        let total_blocks = num_double + config.num_single_blocks;
        let facilitator = Flux1Facilitator { num_double, total_blocks };
        let offloader = BlockOffloader::load(
            &[checkpoint_path],
            &facilitator,
            device.clone(),
        ).map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader Flux1: {e}")))?;

        // Load shared (non-block) weights
        let shared_prefixes = [
            "img_in.", "txt_in.", "time_in.", "guidance_in.", "vector_in.", "final_layer.",
        ];
        let shared_weights = load_file_filtered(
            Path::new(checkpoint_path),
            device,
            |key| shared_prefixes.iter().any(|p| key.starts_with(p)),
        )?;

        // H3: auto-detect Dev vs Schnell by presence of guidance_in.* weights.
        //
        // BFL reference — src/flux/model.py:58-60
        //   self.guidance_in = (
        //       MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        //   )
        config.has_guidance = shared_weights.contains_key("guidance_in.in_layer.weight");
        log::info!("[Flux1] has_guidance (autodetected) = {}", config.has_guidance);

        log::info!("[Flux1] Loaded: {} blocks via BlockOffloader, {} shared weights",
            offloader.block_count(), shared_weights.len());

        // Log shared weight keys
        for key in shared_weights.keys() {
            log::info!("[Flux1] Shared: {} {:?}", key, shared_weights[key].shape());
        }

        Ok(Self {
            shared: shared_weights,
            offloader,
            config,
            device: device.clone(),
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Per-stage telemetry, env-gated.
    ///   FLUX1_STAGE_DEBUG=1 — print abs_mean / min / max / NaN per capture.
    ///   FLUX1_SAVE_STAGES=1 — dump each capture to safetensors for parity
    ///                        diff vs diffusers FLUX.1-dev.
    /// Early-return if both unset; zero cost on normal runs.
    fn log_stage(name: &str, t: &Tensor) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let stats_on = std::env::var("FLUX1_STAGE_DEBUG").is_ok();
        let save_on  = std::env::var("FLUX1_SAVE_STAGES").is_ok();
        if !stats_on && !save_on {
            return;
        }
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        const MAX_CALLS: usize = 200;
        let idx = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        if idx >= MAX_CALLS {
            return;
        }
        if save_on {
            use std::path::Path;
            let safe_name = name.replace(':', "_").replace('.', "_");
            let p_str = format!(
                "/home/alex/EriDiffusion/inference-flame/output/flux1_stage_{idx:03}_{safe_name}.safetensors"
            );
            let p = Path::new(&p_str);
            let mut m: HashMap<String, Tensor> = HashMap::new();
            if let Ok(c) = t.clone_result() {
                m.insert("value".to_string(), c);
                let _ = flame_core::serialization::save_file(&m, p);
            }
        }
        if !stats_on {
            return;
        }
        let f32t = match t.to_dtype(DType::F32) { Ok(x) => x, Err(_) => return };
        let data = match f32t.to_vec() { Ok(v) => v, Err(_) => return };
        let n = data.len();
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut abs_sum = 0f64;
        let mut nan_cnt = 0usize;
        let mut inf_cnt = 0usize;
        for v in &data {
            if v.is_nan() { nan_cnt += 1; continue; }
            if v.is_infinite() { inf_cnt += 1; continue; }
            if *v < mn { mn = *v; }
            if *v > mx { mx = *v; }
            abs_sum += v.abs() as f64;
        }
        let abs_mean = if n > 0 { abs_sum / n as f64 } else { 0.0 };
        eprintln!(
            "[flux1] {idx:03} {name:<24} shape={:?} abs_mean={:.4e} min={:+.4e} max={:+.4e} nan={} inf={}",
            t.shape().dims(), abs_mean, mn, mx, nan_cnt, inf_cnt
        );
    }

    /// 3D linear with fused bias epilogue via cuBLASLt.
    /// Weight is in standard PyTorch `[Cout, Cin]` row-major layout (no
    /// pre-transpose). The transpose happens inside the GEMM via TRANSA=T,
    /// eliminating the per-call `transpose2d_bf16` pass and the separate
    /// bias-add kernel that the previous version was paying.
    fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, Some(bias))
    }

    fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
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
    /// ## BFL reference — src/flux/modules/layers.py:28-49
    /// ```python
    /// def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    ///     t = time_factor * t
    ///     half = dim // 2
    ///     freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
    ///         t.device
    ///     )
    ///
    ///     args = t[:, None].float() * freqs[None]
    ///     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    ///     if dim % 2:
    ///         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    ///     if torch.is_floating_point(t):
    ///         embedding = embedding.to(t)
    ///     return embedding
    /// ```
    ///
    /// H1 fix: keep freqs and args in FP32, cast to input dtype once at the end.
    fn timestep_embedding(t: &Tensor, dim: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let in_dtype = t.dtype();
        // t = time_factor * t   (done in F32)
        let t_f32 = t.to_dtype(DType::F32)?;
        let t_scaled = t_f32.mul_scalar(1000.0)?;
        let half = dim / 2;
        let max_period = 10000.0f64;
        // freqs = exp(-ln(max_period) * arange(half) / half)   (F32)
        let freq_data: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), device.clone())?;
        // args = t[:, None].float() * freqs[None]              (F32)
        let t_col = t_scaled.unsqueeze(1)?; // [B, 1]
        let args = t_col.matmul(&freqs)?; // [B, half] F32
        // embedding = cat([cos(args), sin(args)], dim=-1)
        let cos = args.cos()?;
        let sin = args.sin()?;
        let emb = Tensor::cat(&[&cos, &sin], 1)?; // [B, dim] F32
        // cast back to input dtype
        emb.to_dtype(in_dtype)
    }

    /// MLPEmbedder: out_layer(silu(in_layer(x)))
    ///
    /// ## BFL reference — src/flux/modules/layers.py:52-60
    /// ```python
    /// class MLPEmbedder(nn.Module):
    ///     def __init__(self, in_dim: int, hidden_dim: int):
    ///         super().__init__()
    ///         self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
    ///         self.silu = nn.SiLU()
    ///         self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
    ///
    ///     def forward(self, x: Tensor) -> Tensor:
    ///         return self.out_layer(self.silu(self.in_layer(x)))
    /// ```
    ///
    /// H4: require a 2-D [B, D] input — all callers (time, guidance, vector)
    /// pass 2-D tensors in BFL.
    fn timestep_mlp(x: &Tensor, in_w: &Tensor, in_b: &Tensor, out_w: &Tensor, out_b: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        assert_eq!(
            shape.len(),
            2,
            "MLPEmbedder expects a 2-D [B, D] input, got shape {:?}",
            shape
        );
        let x_2d = x.clone();

        let in_wt = flame_core::bf16_elementwise::transpose2d_bf16(in_w)?;
        let h = x_2d.matmul(&in_wt)?;
        let h = h.add(&in_b.unsqueeze(0)?.expand(&h.shape().dims().to_vec())?)?;
        let h = h.silu()?;

        let out_wt = flame_core::bf16_elementwise::transpose2d_bf16(out_w)?;
        let out = h.matmul(&out_wt)?;
        out.add(&out_b.unsqueeze(0)?.expand(&out.shape().dims().to_vec())?)
    }

    // -----------------------------------------------------------------------
    // RoPE — verbatim port of BFL src/flux/math.py and EmbedND from layers.py
    // -----------------------------------------------------------------------

    /// Build RoPE cos/sin tables (FLUX 1 / Klein interleaved format).
    ///
    /// FLUX 1 BFL `rope()` builds a per-pair 2x2 rotation matrix that is
    /// algebraically `[[cos, -sin], [sin, cos]]`. `flame_core::bf16_ops::rope_fused_bf16`
    /// expects exactly this rotation expressed as separate `cos` / `sin` tables
    /// (interleaved-pair convention, comment in the kernel says
    /// "Used by Klein/Flux, LTX, HunyuanVideo"), so we precompute the same omega·pos
    /// table BFL does and emit `(cos, sin)` directly. This drops the per-block
    /// 5s F32-narrow path.
    ///
    /// img_ids: [N_img, 3]   txt_ids: [N_txt, 3]
    /// Returns `(pe_cos, pe_sin)` each `[1, 1, N, sum(axes)/2]` BF16.
    pub fn build_rope_2d(
        img_ids: &Tensor,
        txt_ids: &Tensor,
        config: &Flux1Config,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?; // [N, 3]
        let all_ids_f32 = all_ids.to_dtype(DType::F32)?;
        let n_total = all_ids_f32.shape().dims()[0];

        let mut cos_parts: Vec<Tensor> = Vec::with_capacity(config.axes_dims_rope.len());
        let mut sin_parts: Vec<Tensor> = Vec::with_capacity(config.axes_dims_rope.len());

        for (axis, &axis_dim) in config.axes_dims_rope.iter().enumerate() {
            assert!(axis_dim % 2 == 0, "rope dim must be even");
            let half = axis_dim / 2;

            // omega = 1 / (theta ** (arange(0, dim, 2) / dim))
            let omega_data: Vec<f32> = (0..half)
                .map(|i| {
                    let scale = (2 * i) as f64 / axis_dim as f64;
                    (1.0 / config.rope_theta.powf(scale)) as f32
                })
                .collect();
            let omega = Tensor::from_vec(
                omega_data,
                Shape::from_dims(&[1, half]),
                device.clone(),
            )?; // [1, half] F32

            // pos = ids[:, axis]   shape [N]
            let pos = all_ids_f32.narrow(1, axis, 1)?.squeeze(Some(1))?;
            // outer product: [N, 1] @ [1, half] = [N, half]
            let pos_col = pos.unsqueeze(1)?;
            let angles = pos_col.matmul(&omega)?;

            cos_parts.push(angles.cos()?);
            sin_parts.push(angles.sin()?);
        }

        // BFL concatenates the per-axis rotation matrices on the half axis.
        // For (cos, sin) tables that's the same: cat on the last (half) axis.
        let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
        let cos_full = Tensor::cat(&cos_refs, 1)?; // [N, sum(half)]
        let sin_full = Tensor::cat(&sin_refs, 1)?;

        // [1, 1, N, sum(half)] **F32** — kept in FP32 to match BFL's apply_rope
        // precision. The ~1 MiB table cost is trivial; the ~4e-3 BF16 floor on
        // cos/sin otherwise accumulates across 57×20×2=2280 RoPE applications
        // per inference (blocks × steps × Q+K) and shows up as muddy detail.
        let pe_cos = cos_full.unsqueeze(0)?.unsqueeze(0)?;
        let pe_sin = sin_full.unsqueeze(0)?.unsqueeze(0)?;
        Ok((pe_cos, pe_sin))
    }

    /// Apply RoPE to q and k via the fused kernel (F32 PE variant).
    ///
    /// `pe_cos`, `pe_sin`: `[1, 1, N, D/2]` **F32**. `q`, `k`: `[B, H, N, D]` BF16.
    /// See `build_rope_2d` for rationale on keeping PE in F32.
    fn apply_rope_complex(
        q: &Tensor,
        k: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q_out = flame_core::bf16_ops::rope_fused_bf16_f32pe(q, pe_cos, pe_sin)?;
        let k_out = flame_core::bf16_ops::rope_fused_bf16_f32pe(k, pe_cos, pe_sin)?;
        Ok((q_out, k_out))
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
    /// BlockOffloader's `prepare_weights` auto-transposes 2D `.weight` tensors
    /// to `[Cin, Cout]` for its own matmul fast path. `fused_linear3d_native`
    /// (which our block-forward helpers call) expects the original PyTorch
    /// layout `[Cout, Cin]`. Un-transpose here so every 2D weight is back in
    /// the layout the kernel checks. Same pattern Chroma uses in
    /// `chroma_dit::untranspose_block_weights`.
    fn untranspose_block_weights(
        raw: &std::sync::Arc<HashMap<String, Tensor>>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::with_capacity(raw.len());
        for (k, v) in raw.iter() {
            if k.ends_with(".weight") && v.shape().dims().len() == 2 {
                out.insert(k.clone(), v.transpose()?);
            } else {
                out.insert(k.clone(), v.clone());
            }
        }
        Ok(out)
    }

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

        // H3: enforce BFL model.py:100-103
        //   if self.params.guidance_embed:
        //       if guidance is None:
        //           raise ValueError("Didn't get guidance strength for guidance distilled model.")
        if cfg.has_guidance && guidance.is_none() {
            return Err(flame_core::Error::InvalidInput(
                "Didn't get guidance strength for guidance distilled model.".into(),
            ));
        }

        let img_len = img.shape().dims()[1];
        let txt_len = txt.shape().dims()[1];

        // --- Input projections ---
        let img = Self::linear_bias(
            img,
            self.shared.get("img_in.weight").ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.weight".into()))?,
            self.shared.get("img_in.bias").ok_or_else(|| flame_core::Error::InvalidInput("Missing img_in.bias".into()))?,
        )?;
        // H2: txt_in uses nn.Linear default bias=True, but allow LoRA-wrapped
        // checkpoints that may not ship a bias by falling back to linear_nobias.
        let txt_in_w = self.shared.get("txt_in.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing txt_in.weight".into()))?;
        let txt = if let Some(txt_in_b) = self.shared.get("txt_in.bias") {
            Self::linear_bias(txt, txt_in_w, txt_in_b)?
        } else {
            Self::linear_nobias(txt, txt_in_w)?
        };

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

        // --- Build RoPE (fused-kernel cos/sin tables) ---
        let (pe_cos, pe_sin) = Self::build_rope_2d(img_ids, txt_ids, &cfg, &self.device)?;

        Self::log_stage("0:img_in", &img);
        Self::log_stage("0:txt_in", &txt);
        Self::log_stage("0:vec", &vec);

        // --- Double blocks ---
        let mut img = img;
        let mut txt = txt;
        let total_blocks = cfg.num_double_blocks + cfg.num_single_blocks;

        let profile = std::env::var("FLUX1_PROFILE").ok().as_deref() == Some("1");
        let mut total_await_ms: u128 = 0;
        let mut total_compute_ms: u128 = 0;
        let mut total_prefetch_ms: u128 = 0;
        let t_pf0 = std::time::Instant::now();
        self.offloader.prefetch_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
        if profile {
            total_prefetch_ms += t_pf0.elapsed().as_millis();
        }

        for i in 0..cfg.num_double_blocks {
            flame_core::device::trim_cuda_mempool(0);
            flame_core::cuda_alloc_pool::clear_pool_cache();
            let t_aw = std::time::Instant::now();
            let raw_arc = self.offloader.await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            let aw_ms = t_aw.elapsed().as_millis();
            total_await_ms += aw_ms;

            let t_pf = std::time::Instant::now();
            if i + 1 < total_blocks {
                self.offloader.prefetch_block(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }
            let pf_ms = t_pf.elapsed().as_millis();
            total_prefetch_ms += pf_ms;

            let raw = Self::untranspose_block_weights(&raw_arc)?;

            // Dump block-0 weights after untranspose — for comparing to
            // the raw checkpoint keys in Python.
            if i == 0 && std::env::var("FLUX1_DUMP_BLOCK0").is_ok() {
                use std::path::Path;
                let p = Path::new("/home/alex/EriDiffusion/inference-flame/output/flux1_block0_weights.safetensors");
                let _ = flame_core::serialization::save_file(&raw, p);
            }

            let t_fwd = std::time::Instant::now();
            let (new_img, new_txt) = self.double_block_forward(
                &img, &txt, &vec, &pe_cos, &pe_sin, &raw, i,
            )?;
            // Sync ONLY when profiling so the per-block timer is accurate.
            // Outside profiling we want consecutive blocks to dispatch and
            // overlap with BlockOffloader H2D streaming, not serialize on a sync.
            if profile {
                let _ = self.device.synchronize();
            }
            let fwd_ms = t_fwd.elapsed().as_millis();
            total_compute_ms += fwd_ms;
            img = new_img;
            txt = new_txt;

            if i == 0 || i == 4 || i == 9 || i == 14 || i == cfg.num_double_blocks - 1 {
                Self::log_stage(&format!("1:dbl_img.{i:02}"), &img);
                Self::log_stage(&format!("1:dbl_txt.{i:02}"), &txt);
            }

            if profile {
                eprintln!("[PROF] double {:>2}/{}: await={:>4}ms prefetch={:>3}ms compute={:>5}ms",
                    i + 1, cfg.num_double_blocks, aw_ms, pf_ms, fwd_ms);
            } else if i % 5 == 0 || i == cfg.num_double_blocks - 1 {
                log::info!("[Flux1] Double block {}/{}", i + 1, cfg.num_double_blocks);
            }
        }

        // --- Merge txt + img for single blocks ---
        let mut x = Tensor::cat(&[&txt, &img], 1)?; // [B, N_txt+N_img, dim]

        // --- Single blocks ---
        for i in 0..cfg.num_single_blocks {
            let block_idx = cfg.num_double_blocks + i;

            flame_core::device::trim_cuda_mempool(0);
            flame_core::cuda_alloc_pool::clear_pool_cache();
            let t_aw = std::time::Instant::now();
            let raw_arc = self.offloader.await_block(block_idx)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            let aw_ms = t_aw.elapsed().as_millis();
            total_await_ms += aw_ms;

            let t_pf = std::time::Instant::now();
            if block_idx + 1 < total_blocks {
                self.offloader.prefetch_block(block_idx + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }
            let pf_ms = t_pf.elapsed().as_millis();
            total_prefetch_ms += pf_ms;

            let raw = Self::untranspose_block_weights(&raw_arc)?;

            let t_fwd = std::time::Instant::now();
            x = self.single_block_forward(&x, &vec, &pe_cos, &pe_sin, txt_len, &raw, i)?;
            if profile {
                let _ = self.device.synchronize();
            }
            let fwd_ms = t_fwd.elapsed().as_millis();
            total_compute_ms += fwd_ms;

            if i == 0 || i == 9 || i == 18 || i == 27 || i == cfg.num_single_blocks - 1 {
                Self::log_stage(&format!("2:sgl.{i:02}"), &x);
            }

            if profile {
                eprintln!("[PROF] single {:>2}/{}: await={:>4}ms prefetch={:>3}ms compute={:>5}ms",
                    i + 1, cfg.num_single_blocks, aw_ms, pf_ms, fwd_ms);
            } else if i % 10 == 0 || i == cfg.num_single_blocks - 1 {
                log::info!("[Flux1] Single block {}/{}", i + 1, cfg.num_single_blocks);
            }
        }

        if profile {
            eprintln!("[PROF] TOTAL await={}ms compute={}ms prefetch={}ms",
                total_await_ms, total_compute_ms, total_prefetch_ms);
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
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let prefix = format!("double_blocks.{block_idx}");
        // Section profiling: only fires when FLUX1_BLOCK_PROF=1 and only on block 0.
        let prof = block_idx == 0 && std::env::var("FLUX1_BLOCK_PROF").ok().as_deref() == Some("1");
        let dev = self.device.clone();
        let mark = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = dev.synchronize();
                eprintln!("[BPROF dbl] {:<22} {:>6}ms", label, t.elapsed().as_millis());
            }
        };
        let t_block = std::time::Instant::now();

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

        // ── 1. modulation projection (silu + 2 linears + 12 narrows) ──
        let t = std::time::Instant::now();
        let img_mod_w = w("img_mod.lin.weight")?;
        let img_mod_b = w("img_mod.lin.bias")?;
        let txt_mod_w = w("txt_mod.lin.weight")?;
        let txt_mod_b = w("txt_mod.lin.bias")?;
        let vec_act = vec.silu()?;
        let img_mods = Self::linear_bias(&vec_act.unsqueeze(1)?, img_mod_w, img_mod_b)?
            .squeeze(Some(1))?;
        let txt_mods = Self::linear_bias(&vec_act.unsqueeze(1)?, txt_mod_w, txt_mod_b)?
            .squeeze(Some(1))?;
        let chunk_size = cfg.inner_dim;
        let img_shift1 = img_mods.narrow(1, 0, chunk_size)?;
        let img_scale1 = img_mods.narrow(1, chunk_size, chunk_size)?;
        let img_gate1  = img_mods.narrow(1, 2 * chunk_size, chunk_size)?;
        let img_shift2 = img_mods.narrow(1, 3 * chunk_size, chunk_size)?;
        let img_scale2 = img_mods.narrow(1, 4 * chunk_size, chunk_size)?;
        let img_gate2  = img_mods.narrow(1, 5 * chunk_size, chunk_size)?;
        let txt_shift1 = txt_mods.narrow(1, 0, chunk_size)?;
        let txt_scale1 = txt_mods.narrow(1, chunk_size, chunk_size)?;
        let txt_gate1  = txt_mods.narrow(1, 2 * chunk_size, chunk_size)?;
        let txt_shift2 = txt_mods.narrow(1, 3 * chunk_size, chunk_size)?;
        let txt_scale2 = txt_mods.narrow(1, 4 * chunk_size, chunk_size)?;
        let txt_gate2  = txt_mods.narrow(1, 5 * chunk_size, chunk_size)?;
        mark("1.mod_proj", t);

        // ── 2a. modulate_pre × 2 ──
        let t = std::time::Instant::now();
        let img_normed = Self::modulate_pre(img, &img_shift1, &img_scale1)?;
        let txt_normed = Self::modulate_pre(txt, &txt_shift1, &txt_scale1)?;
        mark("2a.modulate_pre", t);

        // ── 2b. qkv linears × 2 ──
        let t = std::time::Instant::now();
        let img_qkv = Self::linear_bias(&img_normed, w("img_attn.qkv.weight")?, w("img_attn.qkv.bias")?)?;
        let txt_qkv = Self::linear_bias(&txt_normed, w("txt_attn.qkv.weight")?, w("txt_attn.qkv.bias")?)?;
        mark("2b.qkv_linears", t);

        // ── 2c. split_qkv × 2 ──
        let t = std::time::Instant::now();
        let (img_q, img_k, img_v) = Self::split_qkv(&img_qkv, b, n_img, h, d)?;
        let (txt_q, txt_k, txt_v) = Self::split_qkv(&txt_qkv, b, n_txt, h, d)?;
        mark("2c.split_qkv", t);

        // ── 3. q/k RMSNorm × 4 ──
        let t = std::time::Instant::now();
        let img_q = Self::rms_norm(&img_q, w("img_attn.norm.query_norm.scale")?, 1e-6)?;
        let img_k = Self::rms_norm(&img_k, w("img_attn.norm.key_norm.scale")?, 1e-6)?;
        let txt_q = Self::rms_norm(&txt_q, w("txt_attn.norm.query_norm.scale")?, 1e-6)?;
        let txt_k = Self::rms_norm(&txt_k, w("txt_attn.norm.key_norm.scale")?, 1e-6)?;
        mark("3.qk_rmsnorm", t);

        // ── 4. cat × 3 + RoPE ──
        let t = std::time::Instant::now();
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;
        mark("4a.cat_qkv", t);

        let t = std::time::Instant::now();
        let (q, k) = Self::apply_rope_complex(&q, &k, pe_cos, pe_sin)?;
        mark("4b.rope", t);

        // ── 5. SDPA ──
        let t = std::time::Instant::now();
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        mark("5.sdpa", t);

        // ── 6. split back + permute + reshape + o-proj × 2 ──
        let t = std::time::Instant::now();
        let txt_attn = attn_out.narrow(2, 0, n_txt)?;
        let img_attn = attn_out.narrow(2, n_txt, n_img)?;
        let img_attn = img_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img, h * d])?;
        let txt_attn = txt_attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n_txt, h * d])?;
        let img_attn = Self::linear_bias(&img_attn, w("img_attn.proj.weight")?, w("img_attn.proj.bias")?)?;
        let txt_attn = Self::linear_bias(&txt_attn, w("txt_attn.proj.weight")?, w("txt_attn.proj.bias")?)?;
        mark("6.attn_out+oproj", t);

        // ── 7. gated residual 1 ──
        let t = std::time::Instant::now();
        let img = img.add(&img_gate1.unsqueeze(1)?.mul(&img_attn)?)?;
        let txt = txt.add(&txt_gate1.unsqueeze(1)?.mul(&txt_attn)?)?;
        mark("7.gated_resid1", t);

        // ── 8. modulate2 + MLP (linear→gelu→linear) × 2 ──
        let t = std::time::Instant::now();
        let img_mlp_in = Self::modulate_pre(&img, &img_shift2, &img_scale2)?;
        let txt_mlp_in = Self::modulate_pre(&txt, &txt_shift2, &txt_scale2)?;
        let img_mlp = Self::linear_bias(&img_mlp_in, w("img_mlp.0.weight")?, w("img_mlp.0.bias")?)?;
        let img_mlp = img_mlp.gelu()?;
        let img_mlp = Self::linear_bias(&img_mlp, w("img_mlp.2.weight")?, w("img_mlp.2.bias")?)?;
        let txt_mlp = Self::linear_bias(&txt_mlp_in, w("txt_mlp.0.weight")?, w("txt_mlp.0.bias")?)?;
        let txt_mlp = txt_mlp.gelu()?;
        let txt_mlp = Self::linear_bias(&txt_mlp, w("txt_mlp.2.weight")?, w("txt_mlp.2.bias")?)?;
        mark("8.modulate2+mlp", t);

        // ── 9. gated residual 2 ──
        let t = std::time::Instant::now();
        let img = img.add(&img_gate2.unsqueeze(1)?.mul(&img_mlp)?)?;
        let txt = txt.add(&txt_gate2.unsqueeze(1)?.mul(&txt_mlp)?)?;
        mark("9.gated_resid2", t);

        if prof {
            let _ = dev.synchronize();
            eprintln!("[BPROF dbl] TOTAL                 {:>6}ms", t_block.elapsed().as_millis());
        }

        Ok((img, txt))
    }

    fn single_block_forward(
        &self,
        x: &Tensor,
        vec: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        _txt_len: usize,
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let dim = cfg.inner_dim;
        let mlp_hidden = (dim as f32 * cfg.mlp_ratio) as usize;
        let prefix = format!("single_blocks.{block_idx}");
        // Section profiling: only fires when FLUX1_BLOCK_PROF=1 and only on block 0.
        let prof = block_idx == 0
            && std::env::var("FLUX1_BLOCK_PROF").ok().as_deref() == Some("1");
        let dev = self.device.clone();
        let mark = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = dev.synchronize();
                eprintln!("[BPROF sgl] {:<22} {:>6}ms", label, t.elapsed().as_millis());
            }
        };
        let t_block = std::time::Instant::now();

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing: {key}"))
            })
        };

        let dims = x.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // ── 1. modulation ──
        let t = std::time::Instant::now();
        let vec_act = vec.silu()?;
        let mods = Self::linear_bias(&vec_act.unsqueeze(1)?, w("modulation.lin.weight")?, w("modulation.lin.bias")?)?
            .squeeze(Some(1))?;
        let shift = mods.narrow(1, 0, dim)?;
        let scale = mods.narrow(1, dim, dim)?;
        let gate = mods.narrow(1, 2 * dim, dim)?;
        mark("1.mod_proj", t);

        // ── 2. modulate_pre ──
        let t = std::time::Instant::now();
        let x_normed = Self::modulate_pre(x, &shift, &scale)?;
        mark("2.modulate_pre", t);

        // ── 3. fused linear1 (QKV + MLP_up) ──
        let t = std::time::Instant::now();
        let qkv_mlp = Self::linear_bias(&x_normed, w("linear1.weight")?, w("linear1.bias")?)?;
        let qkv = qkv_mlp.narrow(2, 0, 3 * dim)?;
        let mlp_in = qkv_mlp.narrow(2, 3 * dim, mlp_hidden)?;
        mark("3.linear1", t);

        // ── 4. split_qkv + qk_rmsnorm ──
        let t = std::time::Instant::now();
        let (q, k, v) = Self::split_qkv(&qkv, b, n, h, d)?;
        let q = Self::rms_norm(&q, w("norm.query_norm.scale")?, 1e-6)?;
        let k = Self::rms_norm(&k, w("norm.key_norm.scale")?, 1e-6)?;
        mark("4.split+qk_norm", t);

        // ── 5. RoPE ──
        let t = std::time::Instant::now();
        let (q, k) = Self::apply_rope_complex(&q, &k, pe_cos, pe_sin)?;
        mark("5.rope", t);

        // ── 6. SDPA ──
        let t = std::time::Instant::now();
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        mark("6.sdpa", t);

        // ── 7. permute + reshape ──
        let t = std::time::Instant::now();
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        mark("7.attn_out_permute", t);

        // ── 8. GELU on MLP path ──
        let t = std::time::Instant::now();
        let mlp_out = mlp_in.gelu()?;
        mark("8.gelu", t);

        // ── 9. fused linear2 (attn_proj + mlp_down via cat) ──
        let t = std::time::Instant::now();
        let fused_in = Tensor::cat(&[&attn_out, &mlp_out], 2)?;
        let out = Self::linear_bias(&fused_in, w("linear2.weight")?, w("linear2.bias")?)?;
        mark("9.linear2", t);

        // ── 10. gated residual ──
        let t = std::time::Instant::now();
        let result = x.add(&gate.unsqueeze(1)?.mul(&out)?)?;
        mark("10.gated_resid", t);

        if prof {
            let _ = dev.synchronize();
            eprintln!("[BPROF sgl] TOTAL                 {:>6}ms", t_block.elapsed().as_millis());
        }

        Ok(result)
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
        Self::log_stage("3:pre_final_linear", &x);

        let out_w = self.shared.get("final_layer.linear.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer.linear.weight".into()))?;
        let out_b = self.shared.get("final_layer.linear.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer.linear.bias".into()))?;

        let out = Self::linear_bias(&x, out_w, out_b)?;
        Self::log_stage("4:final_out", &out);
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // QKV split helper
    // -----------------------------------------------------------------------

    fn split_qkv(qkv: &Tensor, b: usize, n: usize, h: usize, d: usize) -> Result<(Tensor, Tensor, Tensor)> {
        // The previous implementation reshaped to 5D `[B,N,3,H,D]` then permuted
        // `[2,0,3,1,4]`. flame-core has no GPU kernel for 5D permutes — it falls
        // through to the general "CPU copy fallback" path in tensor.rs which
        // costs ~2.3s on a 28MB BF16 QKV tensor (per profile, 2026-04-06).
        //
        // Replace with: 3D narrow on the contiguous output dim → 4D reshape →
        // the specialized `permute_0213` hot path (`GpuOps::permute_0213`,
        // already optimized for "Flux attention" reshapes — see tensor.rs:2587).
        // Drops split_qkv from 2333ms to <5ms per block.
        let dim = h * d;
        let split_one = |start: usize| -> Result<Tensor> {
            let s = qkv.narrow(2, start, dim)?;          // [B, N, dim]
            let r = s.reshape(&[b, n, h, d])?;           // [B, N, H, D]  (view)
            r.permute(&[0, 2, 1, 3])                     // [B, H, N, D]  (hot path)
        };
        let q = split_one(0)?;
        let k = split_one(dim)?;
        let v = split_one(2 * dim)?;
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
