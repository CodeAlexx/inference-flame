//! Cosmos-Predict2.5-2B DiT (`MiniTrainDIT`) — pure-Rust port.
//!
//! Inference-only port of NVIDIA's `MiniTrainDIT` (variant `COSMOS_V2_2B_NET`)
//! from `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`. This file
//! covers BUILD_PLAN.md **steps 1-3** only:
//!
//!   1. Config struct + skeleton + safetensors loader.
//!   2. Embedding modules: sinusoidal `Timesteps`, `TimestepEmbedding`
//!      (with optional adaLN-LoRA branch), `LearnablePosEmbAxis`.
//!   3. 3-axis RoPE frequency builder (`build_cosmos_rope_freqs`).
//!
//! Attention, blocks, and the sampler are out of scope for this chunk and
//! will be added in subsequent builder passes.
//!
//! ## Architecture summary (variant `COSMOS_V2_2B_NET`)
//!
//! - 28 blocks, hidden=2048, heads=16, head_dim=128.
//! - Patch: spatial=2, temporal=1; in_c=16, out_c=16; `concat_padding_mask=true`
//!   so the effective patch-embed input is 17 channels.
//! - 3D RoPE (GPT-NeoX half-split): axis split is (dim_t=44, dim_h=42, dim_w=42).
//! - Learnable per-axis additive abs pos emb (`extra_per_block_abs_pos_emb=true`)
//!   added at every block.
//! - `use_adaln_lora=true`, `adaln_lora_dim=256`.
//! - Max grid: 240×240 spatial × 128 temporal latent frames.
//!
//! ## RoPE layout (read this before touching `build_cosmos_rope_freqs`)
//!
//! Python source (`minimal_v4_dit.py:785-793`) returns a freq tensor
//! `[T*H*W, 1, 1, head_dim]` built as `cat([t, h, w] * 2, dim=-1)`. With
//! `transformer_engine.apply_rotary_pos_emb(..., fused=True)` that's exactly
//! the GPT-NeoX half-split layout: index `d` pairs with index `d + head_dim/2`,
//! and both share the *same* per-axis angle (the cat is `[A,B,C,A,B,C]`,
//! so the first half and the second half are identical angle vectors).
//!
//! We therefore feed `flame_core::bf16_ops::rope_halfsplit_bf16` cos/sin
//! tensors of shape `[T*H*W, head_dim/2]`. Each row is
//!     `[cos(t_freqs), cos(h_freqs), cos(w_freqs)]`  (and likewise for sin),
//! and the half-split kernel pairs `(d, d + half)` so the *full* head_dim is
//! rotated, with the same angle applied to both elements of every pair.
//!
//! We keep cos/sin in F32 here; cast to BF16 happens at the apply site
//! (see [[project_bf16_rope_pattern_audit_2026-05-19]] — many ports lose
//! precision by casting at construction).

use flame_core::serialization::load_file;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for `MiniTrainDIT`. Default = `COSMOS_V2_2B_NET`
/// (see `cosmos_predict2/_src/predict2/configs/text2world/defaults/net.py:79`).
#[derive(Debug, Clone)]
pub struct CosmosPredict25Config {
    // shape limits (pre-patch latent grid; divide by patch_spatial / patch_temporal
    // for post-patch axis lengths — see `axis_lens()`)
    pub max_img_h: usize,           // 240
    pub max_img_w: usize,           // 240
    pub max_frames: usize,          // 128

    // io
    pub in_channels: usize,         // 16 (16 latent + 1 padding mask added if concat_padding_mask)
    pub out_channels: usize,        // 16
    pub patch_spatial: usize,       // 2
    pub patch_temporal: usize,      // 1
    pub concat_padding_mask: bool,  // true

    /// When `true`, this DiT is wrapped by `MinimalV1LVGDiT` (the shipped
    /// production checkpoint). The wrapper concatenates a
    /// `condition_video_input_mask` (shape `[B, 1, T, H, W]`) along the
    /// channel axis BEFORE the parent `MiniTrainDIT.forward` runs (which
    /// then concats the parent `padding_mask`). The net effect on the
    /// patch-embed input channels is:
    ///     `patch_in_c = in_channels + lvg(+1) + concat_padding_mask(+1) = 18`
    /// for V2_2B (16 + 1 + 1 = 18). When `false` only the parent
    /// `concat_padding_mask` add applies, giving the historical 17. See
    /// `cosmos_predict2/_src/predict2/networks/minimal_v1_lvg_dit.py:27`.
    pub lvg_wrapper: bool,

    // attention
    pub model_channels: usize,      // 2048 (hidden)
    pub num_blocks: usize,          // 28
    pub num_heads: usize,           // 16
    pub mlp_ratio: f32,             // 4.0

    // cross-attn
    pub crossattn_emb_channels: usize, // 1024 (default)
    pub extra_image_context_dim: Option<usize>, // None for stock V2_2B; set in i2v path

    /// When `true`, project `crossattn_emb` from `crossattn_proj_in_channels` to
    /// `crossattn_emb_channels` via `Linear(bias=True) + GELU` **before** the
    /// block loop. Mirrors Python `minimal_v4_dit.py:1565-1569` and the
    /// production override at `model_2B_reason_1p1_rectified_flow.py:146-149`
    /// (and the duplicate at `:322-324`).
    ///
    /// Base `COSMOS_V2_2B_NET` does **not** enable this (default `false`); the
    /// shipped production checkpoint expects the projection because it pairs
    /// the DiT with Cosmos-Reason1-7B FULL_CONCAT text embeddings
    /// (100352-dim = 28 layers × 3584 hidden).
    pub use_crossattn_projection: bool,

    /// Input dim of `crossattn_proj` Linear. Set to 100352 in production
    /// (FULL_CONCAT of 28 mean-normalized Qwen2.5-VL-7B per-layer outputs).
    /// Ignored when `use_crossattn_projection=false`.
    pub crossattn_proj_in_channels: usize,

    // positional emb
    pub pos_emb_cls: PosEmbCls,     // Rope3d
    pub pos_emb_interpolation: PosEmbInterp, // Crop
    pub pos_emb_learnable: bool,    // true
    pub min_fps: u32,               // 1
    pub max_fps: u32,               // 30
    pub base_fps: u32,              // 24 (VideoRopePosition3DEmb default)

    // adaLN-LoRA
    pub use_adaln_lora: bool,       // true
    pub adaln_lora_dim: usize,      // 256

    // RoPE NTK
    pub rope_h_extrapolation_ratio: f32,    // 1.0
    pub rope_w_extrapolation_ratio: f32,    // 1.0
    pub rope_t_extrapolation_ratio: f32,    // 1.0 (V2_2B inherits 2.0 from V1_7B but V1_2B overrides to 1.0; V2_2B does NOT override -> 2.0 from V1_7B? See note.)
    pub rope_enable_fps_modulation: bool,   // true

    // extra per-block abs pos emb
    pub extra_per_block_abs_pos_emb: bool,  // true (V2_2B)
    pub extra_h_extrapolation_ratio: f32,   // 1.0
    pub extra_w_extrapolation_ratio: f32,   // 1.0
    pub extra_t_extrapolation_ratio: f32,   // 1.0
}

/// Positional-embedding kind. Cosmos only ships `Rope3d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosEmbCls {
    Rope3d,
}

/// Positional-embedding interpolation. Cosmos only ships `Crop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosEmbInterp {
    Crop,
}

impl CosmosPredict25Config {
    /// Match `COSMOS_V2_2B_NET` exactly. This is the post-trained 2B variant
    /// shipped at `nvidia/Cosmos-Predict2.5-2B/base/post-trained/`.
    ///
    /// NOTE on `rope_t_extrapolation_ratio`: `COSMOS_V2_2B_NET` is built via
    /// `L(MiniTrainDIT)(...)` and does *not* set `rope_t_extrapolation_ratio`,
    /// so it falls back to `MiniTrainDIT.__init__`'s default of `1.0`.
    /// (The `_MININET` cousins inherit 2.0 from `COSMOS_V1_7B_NET_MININET`
    /// and then override back to 1.0 for the 2B/14B sizes — see
    /// `configs/text2world/defaults/net.py:54,61`.)
    pub fn cosmos_v2_2b() -> Self {
        Self {
            max_img_h: 240,
            max_img_w: 240,
            max_frames: 128,
            in_channels: 16,
            out_channels: 16,
            patch_spatial: 2,
            patch_temporal: 1,
            concat_padding_mask: true,
            lvg_wrapper: false,
            model_channels: 2048,
            num_blocks: 28,
            num_heads: 16,
            mlp_ratio: 4.0,
            crossattn_emb_channels: 1024,
            extra_image_context_dim: None,
            // Base `COSMOS_V2_2B_NET` does not set `use_crossattn_projection`.
            // Production checkpoints DO (see `cosmos_v2_2b_production` below).
            use_crossattn_projection: false,
            crossattn_proj_in_channels: 1024, // unused when `use_crossattn_projection=false`; mirrors Python default at `minimal_v4_dit.py:1465`
            pos_emb_cls: PosEmbCls::Rope3d,
            pos_emb_interpolation: PosEmbInterp::Crop,
            pos_emb_learnable: true,
            min_fps: 1,
            max_fps: 30,
            base_fps: 24,
            use_adaln_lora: true,
            adaln_lora_dim: 256,
            rope_h_extrapolation_ratio: 1.0,
            rope_w_extrapolation_ratio: 1.0,
            rope_t_extrapolation_ratio: 1.0,
            rope_enable_fps_modulation: true,
            extra_per_block_abs_pos_emb: true,
            extra_h_extrapolation_ratio: 1.0,
            extra_w_extrapolation_ratio: 1.0,
            extra_t_extrapolation_ratio: 1.0,
        }
    }

    /// Production preset for the shipped 2B checkpoint with Cosmos-Reason1-7B
    /// FULL_CONCAT text embeddings (100352-dim → 1024-dim crossattn_proj).
    ///
    /// Source: `configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:146-149`
    /// and the duplicate `:322-324`. The base `cosmos_v2_2b()` config matches
    /// `COSMOS_V2_2B_NET`, which does not enable the projection — but the
    /// shipped post-trained checkpoint includes `crossattn_proj.0.{weight,bias}`
    /// and the trainer expects them to be applied.
    ///
    /// If you load a checkpoint that emits 100352-dim text embeddings (i.e.
    /// from `CosmosReason1Encoder::encode_prompt` with `FullConcat`), use this
    /// preset rather than `cosmos_v2_2b()`.
    ///
    /// CHECKPOINT-DERIVED OVERRIDES (verified 2026-05-21 against the
    /// 4.1 GB shipped `cosmos_predict25_2b_dit.safetensors`):
    ///   - `lvg_wrapper=true`: the shipped checkpoint is a `MinimalV1LVGDiT`,
    ///     so `x_embedder.proj.1.weight` is `[2048, 72]` where 72 = 18 ch * 2² * 1
    ///     and the 18 comes from `16 + 1 (LVG) + 1 (parent padding mask)`.
    ///   - `extra_per_block_abs_pos_emb=false`: the production checkpoint
    ///     ships NO `extra_pos_embedder.*` keys (only `pos_embedder.*` rope3d
    ///     computed buffers, which we skip on load and rebuild at runtime).
    pub fn cosmos_v2_2b_production() -> Self {
        Self {
            use_crossattn_projection: true,
            crossattn_proj_in_channels: 100352,
            crossattn_emb_channels: 1024,
            lvg_wrapper: true,
            extra_per_block_abs_pos_emb: false,
            // Production RoPE config from experiment override
            // (`configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:139-142`):
            //   rope_enable_fps_modulation=False
            //   rope_h_extrapolation_ratio=3.0
            //   rope_w_extrapolation_ratio=3.0
            //   rope_t_extrapolation_ratio=24.0/24=1.0
            rope_h_extrapolation_ratio: 3.0,
            rope_w_extrapolation_ratio: 3.0,
            rope_t_extrapolation_ratio: 1.0,
            rope_enable_fps_modulation: false,
            ..Self::cosmos_v2_2b()
        }
    }

    /// `head_dim = model_channels / num_heads`.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.model_channels / self.num_heads
    }

    /// Per-axis RoPE split for the half-rotated layout. Mirrors
    /// `VideoRopePosition3DEmb.__init__` at `minimal_v4_dit.py:694-698`:
    ///
    /// ```text
    ///     dim_h = head_dim / 6 * 2     // integer div
    ///     dim_w = dim_h
    ///     dim_t = head_dim - 2 * dim_h
    /// ```
    ///
    /// For `head_dim = 128`: `dim_t = 44, dim_h = 42, dim_w = 42`.
    #[inline]
    pub fn rope_axis_split(&self) -> (usize, usize, usize) {
        let d = self.head_dim();
        let dim_h = d / 6 * 2;
        let dim_w = dim_h;
        let dim_t = d - 2 * dim_h;
        (dim_t, dim_h, dim_w)
    }

    /// Patch-embedded input channel count after the LVG concat (when
    /// `lvg_wrapper=true`) and the parent `concat_padding_mask` concat (when
    /// `concat_padding_mask=true`). The LVG concat happens FIRST in the
    /// shipped production module, then the parent `MiniTrainDIT.forward`
    /// concats its padding mask. For V2_2B production:
    ///   `16 + 1 (LVG) + 1 (padding mask) = 18`
    /// For the base config (no LVG wrapper, padding mask only): 17.
    /// For a hypothetical config with neither: 16.
    #[inline]
    pub fn patch_embed_in_channels(&self) -> usize {
        let lvg = if self.lvg_wrapper { 1 } else { 0 };
        let pad = if self.concat_padding_mask { 1 } else { 0 };
        self.in_channels + lvg + pad
    }

    /// Per-axis lengths used to size the learnable abs pos emb buffers
    /// (and the cos/sin precompute upper bounds). Matches
    /// `build_pos_embed` at `minimal_v4_dit.py:1627-1629`:
    ///
    /// ```python
    ///   len_h = self.max_img_h // self.patch_spatial,
    ///   len_w = self.max_img_w // self.patch_spatial,
    ///   len_t = self.max_frames // self.patch_temporal,
    /// ```
    ///
    /// For V2_2B (max_img_h=240, max_img_w=240, max_frames=128,
    /// patch_spatial=2, patch_temporal=1) this is `(128, 120, 120)`.
    #[inline]
    pub fn axis_lens(&self) -> (usize, usize, usize) {
        let len_t = self.max_frames / self.patch_temporal;
        let len_h = self.max_img_h / self.patch_spatial;
        let len_w = self.max_img_w / self.patch_spatial;
        (len_t, len_h, len_w)
    }
}

impl Default for CosmosPredict25Config {
    fn default() -> Self { Self::cosmos_v2_2b() }
}

// ---------------------------------------------------------------------------
// Model skeleton
// ---------------------------------------------------------------------------

/// `MiniTrainDIT` skeleton — for now just owns the loaded weight HashMap and
/// the config. Forward pass is implemented in follow-up chunks (attention,
/// block, sampler). The struct shape is intentionally close to other
/// inference-flame DiTs (`Wan22Dit`, `KleinTransformer`) so it slots into
/// the same patterns.
pub struct CosmosPredict25Dit {
    pub config: CosmosPredict25Config,
    pub device: Arc<CudaDevice>,
    /// All loaded weights, keyed by PyTorch attribute path (post-prefix-strip).
    /// Subsequent build chunks pull tensors from here by name.
    pub weights: HashMap<String, Tensor>,
}

impl CosmosPredict25Dit {
    /// Predicate: should this checkpoint key be loaded into the live model?
    ///
    /// The shipped production checkpoint (4.1 GB, 689 tensors) contains
    /// several keys that are NOT model parameters and must NOT be inserted
    /// into the weight map:
    ///
    /// 1. **Training-time scalars** (4 keys, shape `[]`): training-progress
    ///    accumulators, irrelevant at inference.
    /// 2. **rope3d computed buffers** (3 keys under `pos_embedder.*`): are
    ///    just `arange`-style tables that we rebuild at runtime in
    ///    `build_cosmos_rope_freqs`. Loading them would shadow nothing
    ///    useful and confuse code that expects `pos_embedder.*` to either
    ///    not exist or be the learnable axis buffers.
    /// 3. **TransformerEngine extra-state** (`*._extra_state`, shape `[5]`):
    ///    FP8/BF16 calibration metadata for TE-fused kernels. Our Rust port
    ///    uses flame-core kernels directly and has no use for these.
    ///
    /// Everything else is a live parameter and we load it. Returning `false`
    /// here is purely a load-time skip — it does not affect runtime.
    fn is_loadable_key(key: &str) -> bool {
        // 1. training scalars
        if key.starts_with("accum_") {
            return false;
        }
        // 2. rope3d buffers
        if matches!(
            key,
            "pos_embedder.seq" | "pos_embedder.dim_spatial_range" | "pos_embedder.dim_temporal_range"
        ) {
            return false;
        }
        // 3. TE extra-state metadata
        if key.ends_with("._extra_state") {
            return false;
        }
        true
    }

    /// Load weights from a converted safetensors file.
    ///
    /// Source: `parity/convert_dit_pt_to_safetensors.py` produces a flat
    /// safetensors file from the upstream `*_ema_bf16.pt` checkpoint. Keys
    /// follow the `MiniTrainDIT.__init__` attribute paths (e.g.
    /// `x_embedder.proj.1.weight`, `blocks.0.self_attn.q_proj.weight`, …).
    ///
    /// All tensors are coerced to BF16 (the upstream dtype) — anything that
    /// came in as F16/F32 is cast on the device.
    ///
    /// Keys that match `is_loadable_key=false` are SILENTLY SKIPPED at
    /// `info` log level with a single summary line at the end.
    pub fn from_safetensors(
        path: &Path,
        config: CosmosPredict25Config,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let raw = load_file(path, &device)?;
        let mut weights: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        let mut skipped: usize = 0;
        let mut skipped_examples: Vec<String> = Vec::new();
        let total = raw.len();
        for (k, v) in raw.into_iter() {
            if !Self::is_loadable_key(&k) {
                skipped += 1;
                if skipped_examples.len() < 5 {
                    skipped_examples.push(k);
                }
                continue;
            }
            let v_bf16 = if v.dtype() != DType::BF16 {
                v.to_dtype(DType::BF16).unwrap_or_else(|e| {
                    panic!(
                        "[CosmosPredict25] BF16 cast failed for weight `{}` \
                         (dtype={:?}): {}. Downstream BF16-only kernels would \
                         silently reject this tensor; failing loudly instead.",
                        k, v.dtype(), e
                    )
                })
            } else {
                v
            };
            weights.insert(k, v_bf16);
        }
        log::info!(
            "[CosmosPredict25] Loaded {} tensors ({} skipped: training-scalars + rope3d-buffers + _extra_state) from {}",
            weights.len(), skipped, path.display()
        );
        if skipped > 0 && !skipped_examples.is_empty() {
            log::debug!(
                "[CosmosPredict25] sample skipped keys: {:?}{}",
                skipped_examples,
                if skipped > skipped_examples.len() { " ..." } else { "" }
            );
        }
        let _ = total; // suppress unused warning if logging is off
        Ok(Self { config, device, weights })
    }

    /// Read a weight by attribute path, returning a clear error if missing.
    pub fn weight(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            Error::InvalidInput(format!("[CosmosPredict25] missing weight `{}`", key))
        })
    }

    // -----------------------------------------------------------------------
    // Step 2 — Embedding helpers
    // -----------------------------------------------------------------------

    /// Sinusoidal timestep embedding matching Cosmos's `Timesteps`
    /// (`minimal_v4_dit.py:859`):
    ///
    /// ```python
    ///     half = num_channels // 2
    ///     exponent = -ln(10000) * arange(half) / (half - 0)
    ///     emb = exp(exponent)
    ///     emb = timesteps[:, None] * emb[None, :]
    ///     emb = cat([cos(emb), sin(emb)], dim=-1)
    /// ```
    ///
    /// Note the `cos` comes **first** then `sin`, which is the inverse of the
    /// canonical "sin first" convention used by some other DiTs (e.g. Wan).
    /// Cosmos uses cos-first; we match.
    ///
    /// Input: timestep tensor of shape `[B, T]` (matching Python's `Timesteps.forward`
    /// which takes `timesteps_B_T`). The rearrange `(b t) d -> b t d` is folded in.
    /// Output: `[B, T, num_channels]` F32 tensor on `self.device`.
    pub fn sinusoidal_timesteps(&self, timesteps: &Tensor) -> Result<Tensor> {
        let dim = self.config.model_channels;
        if dim < 2 || dim % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "sinusoidal_timesteps: model_channels must be even and >= 2, got {dim}"
            )));
        }
        let dims = timesteps.shape().dims();
        if dims.len() != 2 {
            return Err(Error::InvalidInput(format!(
                "sinusoidal_timesteps: expected [B, T] timesteps, got shape {:?}", dims
            )));
        }
        let b = dims[0];
        let t_len = dims[1];
        let n = b * t_len;
        // Read scalar timestep values; Python uses float32 arange but float64 ln(10000).
        let t_data_f32 = timesteps.to_dtype(DType::F32)?.to_vec()?;
        let half = dim / 2;
        let mut data = vec![0.0f32; n * dim];
        // exponent[i] = -ln(10000) * i / half      (note: PyTorch source divides by `half - 0.0`
        //                                          which is just `half`; we follow exactly)
        let log_10000 = 10000.0_f64.ln();
        for (t_idx, &t) in t_data_f32.iter().enumerate() {
            let t = t as f64;
            for i in 0..half {
                let exponent = -log_10000 * (i as f64) / (half as f64);
                let freq = exponent.exp();
                let angle = t * freq;
                // cos in first half, sin in second half (Cosmos order).
                data[t_idx * dim + i]        = angle.cos() as f32;
                data[t_idx * dim + half + i] = angle.sin() as f32;
            }
        }
        Tensor::from_vec(data, Shape::from_dims(&[b, t_len, dim]), self.device.clone())
    }

    /// Apply the `TimestepEmbedding` MLP (matching
    /// `TimestepEmbedding.forward` at `minimal_v4_dit.py:908-920`):
    ///
    /// ```text
    ///   x  = linear_1(sample)          // [..., model_channels]
    ///   x  = silu(x)
    ///   y  = linear_2(x)               // either [..., model_channels] (use_adaln_lora=false)
    ///                                  // or     [..., 3*model_channels] (use_adaln_lora=true)
    ///
    ///   if use_adaln_lora:
    ///       emb_B_T_D       = sample        // the *input* sinusoidal vector
    ///       adaln_lora_B_T_3D = y           // the 3*hidden vector
    ///   else:
    ///       emb_B_T_D       = y
    ///       adaln_lora_B_T_3D = None
    /// ```
    ///
    /// IMPORTANT: when `use_adaln_lora=true`, `linear_1` has **no bias**
    /// (`bias=not use_adaln_lora` in the Python source). For V2_2B this is
    /// the case, so `linear_1.bias` will NOT be present in the checkpoint.
    ///
    /// `sample`: `[..., model_channels]` BF16 (output of `sinusoidal_timesteps`
    /// cast to BF16).
    ///
    /// Returns `(emb_b_t_d, adaln_lora_b_t_3d)` per the Python contract.
    /// When `use_adaln_lora=false`, the second element is `None`.
    ///
    /// Weight keys consumed (matches `MiniTrainDIT.__init__` line 1521-1524
    /// — `t_embedder` is a `Sequential(Timesteps, TimestepEmbedding)`, so the
    /// inner MLP lives under `t_embedder.1.linear_{1,2}`):
    /// - `t_embedder.1.linear_1.weight`  (+ `linear_1.bias` if NOT adaln_lora)
    /// - `t_embedder.1.linear_2.weight`
    pub fn timestep_embedding(
        &self,
        sample: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let w1 = self.weight("t_embedder.1.linear_1.weight")?;
        let bias1 = if !self.config.use_adaln_lora {
            // Python sets bias=not use_adaln_lora; only loaded when adaln_lora is off.
            Some(self.weight("t_embedder.1.linear_1.bias")?)
        } else {
            None
        };
        let w2 = self.weight("t_embedder.1.linear_2.weight")?;

        let x = flame_core::ops::fused_inference::fused_linear3d_native(sample, w1, bias1)?;
        let x = x.silu()?;
        let y = flame_core::ops::fused_inference::fused_linear3d_native(&x, w2, None)?;

        if self.config.use_adaln_lora {
            Ok((sample.clone(), Some(y)))
        } else {
            Ok((y, None))
        }
    }

    /// Top-level timestep preparation matching `MiniTrainDIT.forward`
    /// (`minimal_v4_dit.py:1753-1754`):
    ///
    /// ```python
    ///   t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
    ///   t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
    /// ```
    ///
    /// Composes `sinusoidal_timesteps` + `timestep_embedding`, then applies
    /// the post-`t_embedder` RMSNorm (`t_embedding_norm`, eps=1e-6,
    /// declared at `minimal_v4_dit.py:1556`) to the t-embedding branch only
    /// — the adaln_lora branch is left untouched.
    ///
    /// Input: `[B, T]` timesteps.
    /// Output: `(t_embedding_normed [B, T, D], opt_adaln_lora [B, T, 3*D])`.
    ///
    /// Weight key consumed (in addition to those used by `timestep_embedding`):
    /// - `t_embedding_norm.weight`
    pub fn prepare_timestep(
        &self,
        timesteps: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Sinusoidal F32 [B, T, D] → cast to BF16 for the MLP (weights are BF16).
        let sample_f32 = self.sinusoidal_timesteps(timesteps)?;
        let sample = sample_f32.to_dtype(DType::BF16)?;
        let (t_emb, adaln_lora) = self.timestep_embedding(&sample)?;

        // Apply `t_embedding_norm` (RMSNorm, eps=1e-6) to the t-embedding only.
        let weight = self.weight("t_embedding_norm.weight")?;
        let d = self.config.model_channels;
        let t_emb_normed = flame_core::norm::rms_norm(&t_emb, &[d], Some(weight), 1e-6)?;

        Ok((t_emb_normed, adaln_lora))
    }

    /// `LearnablePosEmbAxis.generate_embeddings` (`minimal_v4_dit.py:835-848`).
    ///
    /// Returns a `[B, T, H, W, model_channels]` BF16 tensor formed by
    /// broadcast-summing the three per-axis learnable parameter slices
    /// (with `interpolation="crop"` we just take the first `T`/`H`/`W` rows).
    ///
    /// Weight keys consumed:
    /// - `extra_pos_embedder.pos_emb_t`  (when `extra_per_block_abs_pos_emb=true`,
    ///   which is the V2_2B case) or `pos_embedder.pos_emb_t` otherwise
    /// - `extra_pos_embedder.pos_emb_h`
    /// - `extra_pos_embedder.pos_emb_w`
    ///
    /// NOTE on naming: `MiniTrainDIT.__init__` builds *two* `LearnablePosEmbAxis`
    /// instances when `extra_per_block_abs_pos_emb=true`:
    ///   - `self.pos_embedder` (the rope3d one — for V2_2B this is
    ///     `VideoRopePosition3DEmb`, *not* learnable per-axis buffers)
    ///   - `self.extra_pos_embedder` (the actual `LearnablePosEmbAxis` whose
    ///     output is added every block)
    /// So for V2_2B the learnable axis buffers are under `extra_pos_embedder.*`,
    /// confirmed at `minimal_v4_dit.py:1644-1652`. The BUILD_PLAN.md mapping
    /// table lists them under `pos_embedder.*` — that mapping is incorrect for
    /// the V2_2B variant. Flagged in the build report.
    pub fn learnable_pos_emb(&self, b: usize, t: usize, h: usize, w: usize) -> Result<Tensor> {
        let prefix = if self.config.extra_per_block_abs_pos_emb {
            "extra_pos_embedder"
        } else {
            "pos_embedder"
        };
        let pe_t = self.weight(&format!("{prefix}.pos_emb_t"))?;
        let pe_h = self.weight(&format!("{prefix}.pos_emb_h"))?;
        let pe_w = self.weight(&format!("{prefix}.pos_emb_w"))?;

        let d = self.config.model_channels;

        // Validate shapes against the configured maxima (interpolation="crop":
        // we need T <= len_t, H <= len_h, W <= len_w).
        let pe_t_dims = pe_t.shape().dims();
        let pe_h_dims = pe_h.shape().dims();
        let pe_w_dims = pe_w.shape().dims();
        if pe_t_dims.len() != 2 || pe_t_dims[1] != d {
            return Err(Error::InvalidInput(format!(
                "{prefix}.pos_emb_t: expected [len_t, {d}], got {:?}", pe_t_dims
            )));
        }
        if pe_h_dims.len() != 2 || pe_h_dims[1] != d {
            return Err(Error::InvalidInput(format!(
                "{prefix}.pos_emb_h: expected [len_h, {d}], got {:?}", pe_h_dims
            )));
        }
        if pe_w_dims.len() != 2 || pe_w_dims[1] != d {
            return Err(Error::InvalidInput(format!(
                "{prefix}.pos_emb_w: expected [len_w, {d}], got {:?}", pe_w_dims
            )));
        }
        if t > pe_t_dims[0] || h > pe_h_dims[0] || w > pe_w_dims[0] {
            return Err(Error::InvalidInput(format!(
                "learnable_pos_emb: requested T={t} H={h} W={w} exceeds buffer \
                 dims (len_t={}, len_h={}, len_w={})",
                pe_t_dims[0], pe_h_dims[0], pe_w_dims[0]
            )));
        }

        // Crop to [T, D], [H, D], [W, D] and broadcast all three to a common
        // [1, T, H, W, D] shape before summing. `Tensor::add` does support
        // broadcast on its slow path, but materialising the operands at
        // matching shapes first keeps the path off the BF16 contig fast-path
        // assert in `add`.
        let target = Shape::from_dims(&[1, t, h, w, d]);
        let pe_t_b = pe_t.narrow(0, 0, t)?.reshape(&[1, t, 1, 1, d])?.broadcast_to(&target)?;
        let pe_h_b = pe_h.narrow(0, 0, h)?.reshape(&[1, 1, h, 1, d])?.broadcast_to(&target)?;
        let pe_w_b = pe_w.narrow(0, 0, w)?.reshape(&[1, 1, 1, w, d])?.broadcast_to(&target)?;

        let sum_thw = pe_t_b.add(&pe_h_b)?.add(&pe_w_b)?;

        // Output normalization (Python `LearnablePosEmbAxis.generate_embeddings`
        // minimal_v4_dit.py:850-852):
        //   norm = vector_norm(emb, dim=-1, keepdim=True, dtype=float32)
        //   norm = 1e-6 + sqrt(norm.numel() / emb.numel()) * norm
        //         = 1e-6 + (1/sqrt(D)) * ||emb||_2_lastdim
        //   return emb / norm.to(emb.dtype)
        //
        // Compute in F32 per Python's dtype=torch.float32, then cast back.
        let in_dtype = sum_thw.dtype();
        let sum_f32 = sum_thw.to_dtype(DType::F32)?;
        // L2 norm over last dim, keepdim (i.e. shape [1, T, H, W, 1]).
        let sq = sum_f32.square()?;
        let sumsq = sq.sum_dim(4)?;                          // [1, T, H, W]
        let sumsq_keep = sumsq.reshape(&[1, t, h, w, 1])?;   // restore keepdim
        let l2 = sumsq_keep.sqrt()?;
        // Python: norm = torch.add(1e-6, norm, alpha=sqrt(N/M))
        //         = 1e-6 + alpha * norm, where alpha = sqrt(1/D).
        let scale: f32 = 1.0_f32 / (d as f32).sqrt();
        let denom = l2.mul_scalar(scale)?.add_scalar(1e-6_f32)?;
        // Broadcast-divide the f32 sum by the f32 denom, then cast back to in_dtype.
        let denom_b = denom.broadcast_to(&Shape::from_dims(&[1, t, h, w, d]))?;
        let normed_f32 = sum_f32.div(&denom_b)?;
        let normed = normed_f32.to_dtype(in_dtype)?;

        // [1, T, H, W, D] → [B, T, H, W, D]
        if b == 1 {
            Ok(normed)
        } else {
            normed.broadcast_to(&Shape::from_dims(&[b, t, h, w, d]))
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3 — 3-axis RoPE freq builder
// ---------------------------------------------------------------------------

/// Build Cosmos-Predict2.5 3D RoPE `(cos, sin)` tables for one (T, H, W) grid.
///
/// This is the Rust port of `VideoRopePosition3DEmb.generate_embeddings`
/// (`minimal_v4_dit.py:730-795`). The output is laid out for direct
/// consumption by `flame_core::bf16_ops::rope_halfsplit_bf16`:
///
///   - shape: `[T*H*W, head_dim / 2]` (F32)
///   - row `t*H*W + h*W + w` is the concatenation
///         `[cos(t_angles), cos(h_angles), cos(w_angles)]`
///     (and likewise for sin), with `t_angles` of length `dim_t/2`,
///     `h_angles` of length `dim_h/2`, `w_angles` of length `dim_w/2`.
///   - `(dim_t/2 + dim_h/2 + dim_w/2) == head_dim/2` is checked.
///
/// The half-split kernel pairs index `d` with `d + head_dim/2` and uses
/// `cos[..., d]` / `sin[..., d]` for both members of the pair. Cosmos's
/// `cat([t,h,w] * 2, dim=-1)` (`minimal_v4_dit.py:785-793`) means the *full*
/// `[T,H,W,head_dim]` angle tensor satisfies `angles[..., d] == angles[..., d + half]`,
/// so the half-split rotation applies the matching axis-angle to each pair.
/// We materialise only the first half (= `cos/sin` table); the kernel
/// implicitly re-uses it for the second half via its `(d, d+half)` pairing.
///
/// `dim_h` and `dim_w` are both `head_dim/6*2`; `dim_t = head_dim - 2*dim_h`.
/// For `head_dim=128`: `dim_t=44, dim_h=42, dim_w=42`. With half-rotation that's
/// `dim_t/2=22, dim_h/2=21, dim_w/2=21`, summing to `64 = 128/2`.
///
/// Math is done in F32 throughout (matches Python `.float()` calls). Per
/// `[[project_bf16_rope_pattern_audit_2026-05-19]]`, do not cast cos/sin to
/// BF16 here — the caller casts just before the kernel.
///
/// FPS modulation: when `fps.is_some() && enable_fps_modulation`, temporal
/// positions are scaled `arange(T) * base_fps / fps`. When `fps` is `None`
/// the function assumes image-mode (`T == 1`); a `T > 1` request without `fps`
/// returns an error.
pub fn build_cosmos_rope_freqs(
    head_dim: usize,
    t: usize,
    h: usize,
    w: usize,
    fps: Option<f32>,
    base_fps: f32,
    h_extrapolation_ratio: f32,
    w_extrapolation_ratio: f32,
    t_extrapolation_ratio: f32,
    enable_fps_modulation: bool,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    // head_dim must be even and >= 6. Python axis split (`dim_h = head_dim // 6 * 2`)
    // doesn't require head_dim itself to be divisible by 6 — head_dim=128 (Cosmos V2_2B)
    // gives dim_h=42, dim_w=42, dim_t=44 (all even, sum=128).
    if head_dim == 0 || head_dim < 6 || head_dim % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "build_cosmos_rope_freqs: head_dim must be even and >= 6 (got {head_dim})"
        )));
    }
    // Axis split exactly matches Python: dim_h = head_dim // 6 * 2; dim_w = dim_h;
    // dim_t = head_dim - 2*dim_h. All three must be even (each is rotated as half-split).
    let dim_h: usize = head_dim / 6 * 2;
    let dim_w: usize = dim_h;
    let dim_t: usize = head_dim - 2 * dim_h;
    if dim_h % 2 != 0 || dim_w % 2 != 0 || dim_t % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "build_cosmos_rope_freqs: axis dims must all be even, got \
             (dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w}) for head_dim={head_dim}"
        )));
    }
    if dim_h + dim_w + dim_t != head_dim {
        return Err(Error::InvalidInput(format!(
            "build_cosmos_rope_freqs: axis dims don't sum to head_dim ({} != {head_dim})",
            dim_h + dim_w + dim_t
        )));
    }

    // Python (minimal_v4_dit.py:770-783) distinguishes three cases:
    //   enable_fps_modulation && fps is None  → asserts T == 1 (image mode)
    //   enable_fps_modulation && fps is Some  → modulates positions by base_fps/fps
    //   !enable_fps_modulation                → integer positions, no assertion on T
    // The previous gate rejected the third (valid) case; only the first
    // case should error.
    if enable_fps_modulation && fps.is_none() && t != 1 {
        return Err(Error::InvalidInput(format!(
            "build_cosmos_rope_freqs: fps=None + enable_fps_modulation=true \
             requires T=1 (image mode), got T={t}"
        )));
    }

    let half_t: usize = dim_t / 2;
    let half_h: usize = dim_h / 2;
    let half_w: usize = dim_w / 2;
    let half_d: usize = head_dim / 2;
    debug_assert_eq!(half_t + half_h + half_w, half_d);

    // NTK scaling — Python uses `dim_*` (the full per-axis dim) as the
    // exponent base/divisor; the formula is:
    //   ntk = ratio ** (dim / (dim - 2))
    //   theta = 10000 * ntk
    //   freqs = 1 / theta ** dim_range
    // where `dim_range` = arange(0, dim, 2)[: dim/2] / dim   (a Python *float*
    // range in [0, (dim-2)/dim)). NB the Python source converts dim_h, dim_w,
    // dim_t to float64 implicitly via `dim_h / (dim_h - 2)` — we mirror with f64.
    let h_ntk = (h_extrapolation_ratio as f64).powf(dim_h as f64 / (dim_h as f64 - 2.0));
    let w_ntk = (w_extrapolation_ratio as f64).powf(dim_w as f64 / (dim_w as f64 - 2.0));
    let t_ntk = (t_extrapolation_ratio as f64).powf(dim_t as f64 / (dim_t as f64 - 2.0));

    let h_theta = 10000.0_f64 * h_ntk;
    let w_theta = 10000.0_f64 * w_ntk;
    let t_theta = 10000.0_f64 * t_ntk;

    // dim_*_range[k] = (2k) / dim_*   for k in 0..dim_*/2
    // freqs_*[k] = 1 / theta_* ** range[k]   (Python's `1 / theta ** range`)
    let build_freqs = |theta: f64, dim_axis: usize, half: usize| -> Vec<f64> {
        let mut v = Vec::with_capacity(half);
        for k in 0..half {
            let range_k = (2 * k) as f64 / dim_axis as f64;
            v.push(1.0 / theta.powf(range_k));
        }
        v
    };
    let h_freqs = build_freqs(h_theta, dim_h, half_h);
    let w_freqs = build_freqs(w_theta, dim_w, half_w);
    let t_freqs = build_freqs(t_theta, dim_t, half_t);

    // Position vectors:
    //   - h_pos[i] = i           (i in 0..h)
    //   - w_pos[i] = i           (i in 0..w)
    //   - t_pos[i] = i           (image mode or fps modulation disabled)
    //              = i / fps * base_fps   (video mode with FPS modulation)
    let mut t_pos = vec![0.0_f64; t];
    for i in 0..t { t_pos[i] = i as f64; }
    if enable_fps_modulation {
        if let Some(f) = fps {
            // Python: torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
            for i in 0..t {
                t_pos[i] = (i as f64) / (f as f64) * (base_fps as f64);
            }
        }
        // fps=None → image mode, no modulation. t==1 already enforced above.
    }
    // else: just integer positions, regardless of fps.

    let mut h_pos = vec![0.0_f64; h]; for i in 0..h { h_pos[i] = i as f64; }
    let mut w_pos = vec![0.0_f64; w]; for i in 0..w { w_pos[i] = i as f64; }

    // Outer products: half_emb_*[pos][k] = pos * freq_*[k]
    // We materialise per-axis [N, axis_half] then broadcast-sum into the
    // single [T*H*W, half_d] cos/sin tables in one pass.
    let mut cos_data = vec![0.0_f32; t * h * w * half_d];
    let mut sin_data = vec![0.0_f32; t * h * w * half_d];

    // Precompute per-axis angle rows for cache friendliness.
    let mut t_angles = vec![0.0_f64; t * half_t];
    for ti in 0..t {
        for k in 0..half_t {
            t_angles[ti * half_t + k] = t_pos[ti] * t_freqs[k];
        }
    }
    let mut h_angles = vec![0.0_f64; h * half_h];
    for hi in 0..h {
        for k in 0..half_h {
            h_angles[hi * half_h + k] = h_pos[hi] * h_freqs[k];
        }
    }
    let mut w_angles = vec![0.0_f64; w * half_w];
    for wi in 0..w {
        for k in 0..half_w {
            w_angles[wi * half_w + k] = w_pos[wi] * w_freqs[k];
        }
    }

    // Concatenate [t_angles, h_angles, w_angles] along the half_d axis for
    // every (ti, hi, wi) — i.e. for each output row, write the three axis
    // segments back-to-back. The kernel later treats `[d, d+half_d]` as a
    // pair sharing the same angle; the half_d layout produced here is what
    // it expects.
    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                let row = ti * h * w + hi * w + wi;
                let row_off = row * half_d;
                // segment 0: t angles  [0 .. half_t)
                for k in 0..half_t {
                    let a = t_angles[ti * half_t + k];
                    cos_data[row_off + k]                  = a.cos() as f32;
                    sin_data[row_off + k]                  = a.sin() as f32;
                }
                // segment 1: h angles  [half_t .. half_t+half_h)
                for k in 0..half_h {
                    let a = h_angles[hi * half_h + k];
                    cos_data[row_off + half_t + k]         = a.cos() as f32;
                    sin_data[row_off + half_t + k]         = a.sin() as f32;
                }
                // segment 2: w angles  [half_t+half_h .. half_d)
                for k in 0..half_w {
                    let a = w_angles[wi * half_w + k];
                    cos_data[row_off + half_t + half_h + k] = a.cos() as f32;
                    sin_data[row_off + half_t + half_h + k] = a.sin() as f32;
                }
            }
        }
    }

    let shape = Shape::from_dims(&[t * h * w, half_d]);
    let cos = Tensor::from_vec(cos_data, shape.clone(), device.clone())?;
    let sin = Tensor::from_vec(sin_data, shape,         device.clone())?;
    Ok((cos, sin))
}

// ---------------------------------------------------------------------------
// Step 4-6 — Attention modules, MLP, transformer block
// ---------------------------------------------------------------------------
//
// Python source: `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`
//   - `Attention`              (line 388) — used for both self-attn and
//     plain text-only cross-attn.
//   - `I2VCrossAttention`      (line 582) — extends `Attention` with a second
//     K/V branch (`k_img`, `v_img`) for image-context conditioning. This
//     branch only exists when the wrapping `MiniTrainDIT.extra_image_context_dim`
//     is `Some`. For the stock V2_2B config (`net.py:79-97`) it is `None`,
//     so the 2B base ships **without** the dual-K/V branch — image
//     conditioning for i2v/v2v is delivered via conditional latent frames
//     plus the padding-mask channel, NOT via this dual-K/V path. We
//     implement the dual-K/V branch anyway and gate it on
//     `config.extra_image_context_dim.is_some()`, but exercise only the
//     text-only path for V2_2B.
//   - `GPT2FeedForward`        (line 237) — `Linear → GELU → Linear`, no bias.
//     Python uses bare `nn.GELU()` which defaults to the exact-erf form.
//     flame-core only ships the tanh-approx variant (`Tensor::gelu`). The
//     two differ by ~0.02% magnitude; that gap is flagged in the build
//     report as a flame-core gap.
//   - `Block.forward`          (line 1257) — three sub-blocks (self-attn,
//     cross-attn, FFN) each with `LayerNorm(elementwise_affine=False) +
//     adaLN-LoRA modulation + gate residual`. The same `adaln_lora_B_T_3D`
//     output of `prepare_timestep` is added into each sub-block's
//     modulation. Chunk order per sub-block is `(shift, scale, gate)`
//     confirmed against Python `:1272-1280`. Per-block additive learnable
//     pos emb is added INSIDE the block forward, per Python `:1267-1268`.

impl CosmosPredict25Dit {
    /// Helper: row-major `[..., Cin] @ weight.T` where `weight` is
    /// `[Cout, Cin]` (PyTorch row-major convention, what
    /// `fused_linear3d_native` expects). Mirrors `anima::linear_no_bias`.
    ///
    /// The input may be any rank ≥ 2. We collapse the leading dims to a
    /// 3D `[B, N, Cin]` shape so we can route the call to
    /// `fused_linear3d_native` (the live cuBLASLt path); the output is
    /// reshaped back to `[..., Cout]`.
    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.weight(weight_key)?;
        let in_dims = x.shape().dims().to_vec();
        if in_dims.len() < 2 {
            return Err(Error::InvalidInput(format!(
                "linear_no_bias({weight_key}): input rank must be >= 2, got {:?}",
                in_dims
            )));
        }
        let cin = *in_dims.last().unwrap();
        // Flatten everything except the last dim into a single N axis.
        let n: usize = in_dims[..in_dims.len() - 1].iter().product();
        let x_3d = x.reshape(&[1, n, cin])?;
        let y_3d = flame_core::ops::fused_inference::fused_linear3d_native(&x_3d, weight, None)?;
        let cout = weight.shape().dims()[0];
        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(cout);
        y_3d.reshape(&out_dims)
    }

    /// Apply the optional `crossattn_proj = Sequential(Linear(bias=True), GELU)`
    /// projection to the text embedding. Mirrors Python `:1565-1569` (module
    /// declaration) and `:1738-1739` (the conditional call site in `forward`).
    ///
    /// Behavior:
    ///   1. Linear with bias: `[..., crossattn_proj_in_channels] @ W^T + b`
    ///      where `W` is `[crossattn_emb_channels, crossattn_proj_in_channels]`
    ///      (PyTorch row-major), `b` is `[crossattn_emb_channels]`. Routes
    ///      through `fused_linear3d_native` with the bias epilogue.
    ///   2. **Exact-erf GELU** (Python source `:1568` uses bare `nn.GELU()`
    ///      which defaults to `approximate='none'`, i.e. exact-erf). flame-core
    ///      ships `Tensor::gelu_exact()`; the tanh-approx variant would drift
    ///      by ~9e-4 per element at x=1.
    ///
    /// Weight keys: `crossattn_proj.0.weight`, `crossattn_proj.0.bias`. The
    /// `.0.` index matches Python `nn.Sequential(Linear, GELU)` numbering:
    /// index 0 = Linear, index 1 = GELU (no learnable params).
    ///
    /// Input: `[..., crossattn_proj_in_channels]` BF16.
    /// Output: `[..., crossattn_emb_channels]` BF16.
    fn apply_crossattn_proj(&self, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.shape().dims().to_vec();
        if in_dims.len() < 2 {
            return Err(Error::InvalidInput(format!(
                "apply_crossattn_proj: input rank must be >= 2, got {:?}",
                in_dims
            )));
        }
        let cin = *in_dims.last().unwrap();
        if cin != self.config.crossattn_proj_in_channels {
            return Err(Error::InvalidInput(format!(
                "apply_crossattn_proj: last-dim {cin} != config.crossattn_proj_in_channels {}",
                self.config.crossattn_proj_in_channels
            )));
        }
        let weight = self.weight("crossattn_proj.0.weight")?;
        let bias = self.weight("crossattn_proj.0.bias")?;
        let n: usize = in_dims[..in_dims.len() - 1].iter().product();
        let x_3d = x.reshape(&[1, n, cin])?;
        let y_3d = flame_core::ops::fused_inference::fused_linear3d_native(
            &x_3d, weight, Some(bias),
        )?;
        let cout = weight.shape().dims()[0];
        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(cout);
        let linear_out = y_3d.reshape(&out_dims)?;
        // Python uses bare `nn.GELU()` (exact-erf), NOT `approximate='tanh'`.
        // Confirmed via `minimal_v4_dit.py:1568`.
        linear_out.gelu_exact()
    }

    /// Per-head RMSNorm over the last dim, for tensors shaped `[B, H, N, D]`.
    /// Weight shape is `[D]` (`head_dim`). Matches anima's
    /// `rms_norm_per_head_bhsd`.
    fn rms_norm_per_head_bhnd(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.weight(weight_key)?;
        let dims = x.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "rms_norm_per_head_bhnd({weight_key}): expected [B,H,N,D], got {:?}",
                dims
            )));
        }
        let (b, h, n, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * h * n, d])?;
        let normed = flame_core::norm::rms_norm(&flat, &[d], Some(weight), 1e-6)?;
        normed.reshape(&[b, h, n, d])
    }

    /// Per-head RMSNorm over the last dim, for tensors shaped `[B, N, H, D]`.
    fn rms_norm_per_head_bnhd(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.weight(weight_key)?;
        let dims = x.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "rms_norm_per_head_bnhd({weight_key}): expected [B,N,H,D], got {:?}",
                dims
            )));
        }
        let (b, n, h, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * n * h, d])?;
        let normed = flame_core::norm::rms_norm(&flat, &[d], Some(weight), 1e-6)?;
        normed.reshape(&[b, n, h, d])
    }

    /// Flatten `[B, T, H, W, D]` to `[B, T*H*W, D]` (sequence form expected by
    /// `Attention.forward`). The Python source rearranges the same way at
    /// `Block.forward:1328`.
    #[inline]
    fn flatten_thw(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "flatten_thw: expected [B,T,H,W,D], got {:?}", dims
            )));
        }
        let (b, t, h, w, d) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        x.reshape(&[b, t * h * w, d])
    }

    /// Restore `[B, T*H*W, D]` → `[B, T, H, W, D]`.
    #[inline]
    fn unflatten_thw(x: &Tensor, t: usize, h: usize, w: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "unflatten_thw: expected [B,N,D], got {:?}", dims
            )));
        }
        let (b, n, d) = (dims[0], dims[1], dims[2]);
        if n != t * h * w {
            return Err(Error::InvalidInput(format!(
                "unflatten_thw: N={n} != T*H*W={}", t * h * w
            )));
        }
        x.reshape(&[b, t, h, w, d])
    }

    // -----------------------------------------------------------------------
    // Step 4 — Self-attention
    // -----------------------------------------------------------------------

    /// Cosmos `Attention` (self-attn variant). Mirrors Python `Attention.forward`
    /// (`minimal_v4_dit.py:559`) with `context=None`, `rope_emb` provided.
    ///
    /// Flow:
    ///   1. Project Q/K/V from x via `{prefix}.{q,k,v}_proj.weight`.
    ///   2. Reshape to `[B, N, H, D]` then permute to `[B, H, N, D]` for SDPA.
    ///   3. RMSNorm per-head on Q and K (`q_norm.weight`, `k_norm.weight`).
    ///   4. Apply half-split RoPE to Q and K (cos/sin laid out as
    ///      `[1, 1, N, head_dim/2]` BF16). V skips RoPE (Python `:527`).
    ///   5. `sdpa(q, k, v, None)` → `[B, H, N, D]`.
    ///   6. Permute back to `[B, N, H*D]`, project via `output_proj`.
    ///
    /// `x`: `[B, T, H, W, model_channels]` BF16 (post-modulation).
    /// `rope_cos`, `rope_sin`: F32, shape `[T*H*W, head_dim/2]`
    /// (produced by `build_cosmos_rope_freqs`). Cast to BF16 + reshaped to
    /// `[1, 1, T*H*W, head_dim/2]` here, matching `rope_halfsplit_bf16`'s
    /// expected layout.
    ///
    /// Returns `[B, T, H, W, model_channels]` BF16.
    pub fn self_attention(
        &self,
        x_b_t_h_w_d: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{block_idx}.self_attn");
        let dims = x_b_t_h_w_d.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "self_attention: expected [B,T,H,W,D], got {:?}", dims
            )));
        }
        let (b, t, h, w, _d) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let seq = t * h * w;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();

        // Flatten to [B, N, D]
        let x_seq = Self::flatten_thw(x_b_t_h_w_d)?;

        // Q/K/V projections.
        let q = self.linear_no_bias(&x_seq, &format!("{prefix}.q_proj.weight"))?;
        let k = self.linear_no_bias(&x_seq, &format!("{prefix}.k_proj.weight"))?;
        let v = self.linear_no_bias(&x_seq, &format!("{prefix}.v_proj.weight"))?;

        // [B, N, H*D] → [B, N, H, D] → [B, H, N, D]
        let q = q.reshape(&[b, seq, num_heads, head_dim])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq, num_heads, head_dim])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq, num_heads, head_dim])?.permute(&[0, 2, 1, 3])?;

        // Per-head RMSNorm on Q and K (NOT V — Python uses Identity for v_norm).
        let q = self.rms_norm_per_head_bhnd(&q, &format!("{prefix}.q_norm.weight"))?;
        let k = self.rms_norm_per_head_bhnd(&k, &format!("{prefix}.k_norm.weight"))?;

        // RoPE — cast cos/sin to BF16 and reshape to [1, 1, N, half] which
        // is what `rope_halfsplit_bf16` accepts (per docs at
        // `bf16_ops.rs:1101-1106`).
        let half = head_dim / 2;
        let cos_shape = rope_cos.shape().dims().to_vec();
        if cos_shape.len() != 2 || cos_shape[0] != seq || cos_shape[1] != half {
            return Err(Error::InvalidInput(format!(
                "self_attention: rope_cos shape must be [{seq}, {half}], got {:?}", cos_shape
            )));
        }
        let cos_b = rope_cos.to_dtype(DType::BF16)?.reshape(&[1, 1, seq, half])?;
        let sin_b = rope_sin.to_dtype(DType::BF16)?.reshape(&[1, 1, seq, half])?;

        let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, &cos_b, &sin_b)?;
        let k = flame_core::bf16_ops::rope_halfsplit_bf16(&k, &cos_b, &sin_b)?;

        // SDPA → [B, H, N, D]
        let out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // [B, H, N, D] → [B, N, H, D] → [B, N, H*D]
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq, num_heads * head_dim])?;

        // Output projection.
        let out = self.linear_no_bias(&out, &format!("{prefix}.output_proj.weight"))?;

        Self::unflatten_thw(&out, t, h, w)
    }

    // -----------------------------------------------------------------------
    // Step 4 — Cross-attention (text only) + optional I2V dual-K/V branch
    // -----------------------------------------------------------------------

    /// Cosmos cross-attn. For the stock V2_2B config (`extra_image_context_dim
    /// is None`) Python uses plain `Attention` with text K/V. When the wrapping
    /// model is built with `extra_image_context_dim` set, an `I2VCrossAttention`
    /// is used instead, which has a second K/V branch (`k_img`, `v_img`) fed
    /// from a projected image-context tensor; the two SDPA outputs are
    /// **summed** before the output projection (Python `:611-614`).
    ///
    /// Cross-attention does NOT apply RoPE (Python `:527`: `is_selfattn and
    /// rope_emb is not None`).
    ///
    /// `x_b_t_h_w_d`: `[B, T, H, W, model_channels]` BF16, queries.
    /// `text_context`: `[B, S_txt, crossattn_emb_channels]` BF16.
    ///   For V2_2B this is 1024-dim (`crossattn_emb_channels=1024`); the
    ///   k_proj/v_proj weight shapes are `[2048, 1024]`.
    /// `image_context`: optional `[B, S_img, model_channels]` BF16. Only
    ///   read when `config.extra_image_context_dim.is_some()`; ignored
    ///   otherwise (with a debug log if a non-None is passed for a
    ///   text-only model).
    ///
    /// Returns `[B, T, H, W, model_channels]` BF16.
    pub fn cross_attention(
        &self,
        x_b_t_h_w_d: &Tensor,
        text_context: &Tensor,
        image_context: Option<&Tensor>,
        block_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{block_idx}.cross_attn");
        let dims = x_b_t_h_w_d.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "cross_attention: expected [B,T,H,W,D], got {:?}", dims
            )));
        }
        let (b, t, h, w, _d) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let seq_q = t * h * w;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();

        let ctx_dims = text_context.shape().dims().to_vec();
        if ctx_dims.len() != 3 || ctx_dims[0] != b {
            return Err(Error::InvalidInput(format!(
                "cross_attention: text_context must be [B={b}, S, C], got {:?}", ctx_dims
            )));
        }
        let seq_txt = ctx_dims[1];

        // Q from x.
        let x_seq = Self::flatten_thw(x_b_t_h_w_d)?;
        let q = self.linear_no_bias(&x_seq, &format!("{prefix}.q_proj.weight"))?;
        // K, V from text_context.
        let k_text = self.linear_no_bias(text_context, &format!("{prefix}.k_proj.weight"))?;
        let v_text = self.linear_no_bias(text_context, &format!("{prefix}.v_proj.weight"))?;

        // [B, N, H, D]
        let q = q.reshape(&[b, seq_q, num_heads, head_dim])?;
        let k_text = k_text.reshape(&[b, seq_txt, num_heads, head_dim])?;
        let v_text = v_text.reshape(&[b, seq_txt, num_heads, head_dim])?;

        // Per-head RMSNorm on Q and K_text.
        let q = self.rms_norm_per_head_bnhd(&q, &format!("{prefix}.q_norm.weight"))?;
        let k_text = self.rms_norm_per_head_bnhd(&k_text, &format!("{prefix}.k_norm.weight"))?;

        // [B, N, H, D] → [B, H, N, D] for SDPA.
        let q = q.permute(&[0, 2, 1, 3])?;
        let k_text = k_text.permute(&[0, 2, 1, 3])?;
        let v_text = v_text.permute(&[0, 2, 1, 3])?;

        // First SDPA (text branch).
        let out_text = flame_core::attention::sdpa(&q, &k_text, &v_text, None)?;

        // Optional image-context branch (I2VCrossAttention.compute_attention,
        // Python `:611-614`). Only active when the model was built with
        // `extra_image_context_dim` set. For V2_2B base this is None, so we
        // skip — but anyone wiring the i2v variant gets the full path.
        let out_combined = if let (true, Some(img_ctx)) =
            (self.config.extra_image_context_dim.is_some(), image_context)
        {
            let img_dims = img_ctx.shape().dims().to_vec();
            if img_dims.len() != 3 || img_dims[0] != b {
                return Err(Error::InvalidInput(format!(
                    "cross_attention: image_context must be [B={b}, S, model_channels], got {:?}",
                    img_dims
                )));
            }
            let seq_img = img_dims[1];

            let k_img = self.linear_no_bias(img_ctx, &format!("{prefix}.k_img.weight"))?;
            let v_img = self.linear_no_bias(img_ctx, &format!("{prefix}.v_img.weight"))?;
            let k_img = k_img.reshape(&[b, seq_img, num_heads, head_dim])?;
            let v_img = v_img.reshape(&[b, seq_img, num_heads, head_dim])?;

            // `k_img_norm` is applied (Python `:609`); v_img is left
            // un-normed (matches the no-v_norm pattern from base Attention).
            let k_img = self.rms_norm_per_head_bnhd(&k_img, &format!("{prefix}.k_img_norm.weight"))?;

            let k_img = k_img.permute(&[0, 2, 1, 3])?;
            let v_img = v_img.permute(&[0, 2, 1, 3])?;

            let out_img = flame_core::attention::sdpa(&q, &k_img, &v_img, None)?;
            // SUM (not concat) per Python `:614`: `result + result_img`.
            out_text.add(&out_img)?
        } else {
            if image_context.is_some() && self.config.extra_image_context_dim.is_none() {
                log::debug!(
                    "[CosmosPredict25] block {block_idx} cross_attention got image_context \
                     but model was built without extra_image_context_dim — branch skipped."
                );
            }
            out_text
        };

        // [B, H, N, D] → [B, N, H*D]
        let out = out_combined.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq_q, num_heads * head_dim])?;
        let out = self.linear_no_bias(&out, &format!("{prefix}.output_proj.weight"))?;

        Self::unflatten_thw(&out, t, h, w)
    }

    // -----------------------------------------------------------------------
    // Step 5 — GPT2FeedForward (MLP)
    // -----------------------------------------------------------------------

    /// `GPT2FeedForward` (Python `:237`):
    ///   `x → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)`
    /// All Linears are bias-free. `d_ff = d_model * mlp_ratio` (2048 * 4 = 8192
    /// for V2_2B).
    ///
    /// Important: Python uses bare `nn.GELU()` which is the **exact-erf**
    /// variant (`approximate='none'`). flame-core's `Tensor::gelu` is the
    /// **tanh-approx** form (see `bf16_ops.rs:32-57` and `cuda/unary/gelu.cu`).
    /// Use `Tensor::gelu_exact()` here (added 2026-05-21 specifically for
    /// Cosmos parity; see `flame-core/src/cuda/unary/gelu_exact.cu` and
    /// `tests/tensor_iterator_gelu_exact_parity.rs`). The previous ~0.02%
    /// per-block FFN divergence vs PyTorch reference is now resolved up to
    /// residual BF16 ULP-level noise.
    ///
    /// Weight keys: `blocks.{i}.mlp.layer1.weight`, `blocks.{i}.mlp.layer2.weight`.
    /// Note: Python attribute names are `layer1`/`layer2`, NOT `fc1`/`fc2`
    /// as listed in early BUILD_PLAN.md drafts. Confirmed against
    /// `minimal_v4_dit.py:241-242`.
    pub fn mlp(&self, x_b_t_h_w_d: &Tensor, block_idx: usize) -> Result<Tensor> {
        let prefix = format!("blocks.{block_idx}.mlp");
        let h = self.linear_no_bias(x_b_t_h_w_d, &format!("{prefix}.layer1.weight"))?;
        let h = h.gelu_exact()?;
        self.linear_no_bias(&h, &format!("{prefix}.layer2.weight"))
    }

    // -----------------------------------------------------------------------
    // Step 6 — adaLN-LoRA modulation generator
    // -----------------------------------------------------------------------

    /// Compute the 9 modulation tensors for one Cosmos block.
    ///
    /// Python source (`Block.forward`, `minimal_v4_dit.py:1271-1280`): each of
    /// the three sub-blocks (self-attn, cross-attn, FFN) has its own
    /// `adaln_modulation_*` sequential — `SiLU + Linear(D, 256) + Linear(256, 3D)` —
    /// applied to the conditioning embedding `emb_B_T_D`. The result is
    /// then **added to `adaln_lora_B_T_3D`** (the rank-256 LoRA path output
    /// of `prepare_timestep`) and split into `(shift, scale, gate)` via
    /// `.chunk(3, dim=-1)`. Same `adaln_lora_B_T_3D` is shared across all
    /// three sub-blocks.
    ///
    /// Weight keys (per block, per sub-block):
    ///   - `blocks.{i}.adaln_modulation_self_attn.1.weight`  [256, D]
    ///   - `blocks.{i}.adaln_modulation_self_attn.2.weight`  [3D, 256]
    ///   - `blocks.{i}.adaln_modulation_cross_attn.{1,2}.weight`
    ///   - `blocks.{i}.adaln_modulation_mlp.{1,2}.weight`
    /// The `.0` index in Python is the bare `SiLU` (no params).
    ///
    /// `emb_b_t_d`: `[B, T, model_channels]` BF16 — the post-`t_embedding_norm`
    /// conditioning vector (`prepare_timestep` output #0).
    /// `adaln_lora_b_t_3d`: `[B, T, 3*model_channels]` BF16 — `prepare_timestep`
    /// output #1. Required for V2_2B (`use_adaln_lora=true`).
    /// `sub_name`: one of "self_attn", "cross_attn", "mlp".
    ///
    /// Returns `(shift, scale, gate)` each `[B, T, model_channels]` BF16.
    fn adaln_modulation_chunk(
        &self,
        emb_b_t_d: &Tensor,
        adaln_lora_b_t_3d: &Tensor,
        block_idx: usize,
        sub_name: &str,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let prefix = format!("blocks.{block_idx}.adaln_modulation_{sub_name}");
        // SiLU comes first in the Sequential (Python `:1202`).
        let h0 = emb_b_t_d.silu()?;
        // Linear 1: [D] → [256], no bias.
        let h1 = self.linear_no_bias(&h0, &format!("{prefix}.1.weight"))?;
        // Linear 2: [256] → [3D], no bias.
        let h2 = self.linear_no_bias(&h1, &format!("{prefix}.2.weight"))?;

        // Add the LoRA path output (Python `:1273`):
        // `(adaln_modulation(emb) + adaln_lora_B_T_3D).chunk(3, dim=-1)`.
        let summed = h2.add(adaln_lora_b_t_3d)?;

        // chunk(3, dim=-1): split along last axis into three [B, T, D] slices.
        let d = self.config.model_channels;
        let shift = summed.narrow(2, 0, d)?;
        let scale = summed.narrow(2, d, d)?;
        let gate  = summed.narrow(2, 2 * d, d)?;
        Ok((shift, scale, gate))
    }

    /// Apply the (shift, scale) modulation pair to a `[B, T, H, W, D]` tensor:
    ///   `LayerNorm(x) * (1 + scale) + shift`
    /// where the LayerNorm is `elementwise_affine=False, eps=1e-6` (Python
    /// `:1168, 1179, 1196`).
    ///
    /// `shift`, `scale` are `[B, T, D]`; broadcast across (H, W) is done by
    /// reshaping the LN'd x to `[B*T, H*W, D]` and the modulators to
    /// `[B*T, D]`, then calling `flame_core::bf16_ops::modulate_pre_fused_bf16`
    /// which fuses the (no-affine) LayerNorm with the modulate step.
    fn apply_layer_norm_modulate(
        &self,
        x_b_t_h_w_d: &Tensor,
        shift_b_t_d: &Tensor,
        scale_b_t_d: &Tensor,
    ) -> Result<Tensor> {
        let dims = x_b_t_h_w_d.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "apply_layer_norm_modulate: x must be [B,T,H,W,D], got {:?}", dims
            )));
        }
        let (b, t, h, w, d) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        // [B, T, H, W, D] → [B*T, H*W, D]; shift/scale [B, T, D] → [B*T, D].
        let x_bt_hw_d = x_b_t_h_w_d.reshape(&[b * t, h * w, d])?;
        let shift_bt_d = shift_b_t_d.reshape(&[b * t, d])?;
        let scale_bt_d = scale_b_t_d.reshape(&[b * t, d])?;
        let y = flame_core::bf16_ops::modulate_pre_fused_bf16(
            &x_bt_hw_d, &shift_bt_d, &scale_bt_d, 1e-6,
        )?;
        y.reshape(&[b, t, h, w, d])
    }

    /// Multiply a `[B, T, H, W, D]` tensor by a `[B, T, D]` gate, broadcasting
    /// across (H, W). Returns a freshly materialized tensor.
    fn apply_gate(
        &self,
        x_b_t_h_w_d: &Tensor,
        gate_b_t_d: &Tensor,
    ) -> Result<Tensor> {
        let dims = x_b_t_h_w_d.shape().dims().to_vec();
        let (b, t, h, w, d) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        // Make the gate broadcastable: [B, T, D] → [B, T, 1, 1, D] → broadcast to full.
        let gate_b_t_1_1_d = gate_b_t_d.reshape(&[b, t, 1, 1, d])?;
        let target = Shape::from_dims(&[b, t, h, w, d]);
        let gate_full = gate_b_t_1_1_d.broadcast_to(&target)?;
        x_b_t_h_w_d.mul(&gate_full)
    }

    // -----------------------------------------------------------------------
    // Step 6 — transformer_block
    // -----------------------------------------------------------------------

    /// Single Cosmos block forward. Mirrors `Block.forward`
    /// (`minimal_v4_dit.py:1257-1382`) with the modulate-then-attn-then-gate
    /// pattern for three sub-blocks (self-attn, cross-attn, FFN).
    ///
    /// Inputs:
    /// - `x_b_t_h_w_d`: residual stream `[B, T, H, W, model_channels]` BF16.
    /// - `emb_b_t_d`: `[B, T, model_channels]` post-`t_embedding_norm`
    ///   conditioning (from `prepare_timestep`).
    /// - `adaln_lora_b_t_3d`: `[B, T, 3*model_channels]` — required when
    ///   `use_adaln_lora=true` (true for V2_2B).
    /// - `text_context`: `[B, S_txt, crossattn_emb_channels]` text embedding.
    /// - `image_context`: optional `[B, S_img, model_channels]` (only meaningful
    ///   when `config.extra_image_context_dim.is_some()`).
    /// - `extra_per_block_pos_emb`: optional `[B, T, H, W, D]` learnable pos
    ///   emb tensor. Python adds this once at the top of the block when
    ///   provided (Python `:1267-1268`). The caller (chunk 7) is expected to
    ///   pre-compute it via `learnable_pos_emb` and pass it to every block.
    /// - `rope_cos`, `rope_sin`: F32 RoPE tables `[T*H*W, head_dim/2]`,
    ///   shared across all blocks.
    pub fn transformer_block(
        &self,
        x_b_t_h_w_d: &Tensor,
        emb_b_t_d: &Tensor,
        adaln_lora_b_t_3d: &Tensor,
        text_context: &Tensor,
        image_context: Option<&Tensor>,
        extra_per_block_pos_emb: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        // F32 residual stream — Cosmos hidden values reach >30k by block 27 in
        // BF16, well past the trained distribution. Anima oracle does the same
        // (`anima.rs:471, 483, 496, 508`): residual lives in F32, only the
        // sub-block forwards (attn/cross/mlp) run in BF16. Gate multiplication
        // and residual add happen in F32 to preserve magnitude.
        let mut x_f32 = if let Some(pe) = extra_per_block_pos_emb {
            x_b_t_h_w_d.add(pe)?.to_dtype(DType::F32)?
        } else {
            x_b_t_h_w_d.to_dtype(DType::F32)?
        };

        // --- Self-attention ---
        let (sh_sa, sc_sa, ga_sa) = self.adaln_modulation_chunk(
            emb_b_t_d, adaln_lora_b_t_3d, block_idx, "self_attn",
        )?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;
        let x_mod = self.apply_layer_norm_modulate(&x_bf16, &sh_sa, &sc_sa)?;
        let attn_out = self.self_attention(&x_mod, rope_cos, rope_sin, block_idx)?;
        let attn_out_gated_f32 = self
            .apply_gate(&attn_out, &ga_sa)?
            .to_dtype(DType::F32)?;
        x_f32 = x_f32.add(&attn_out_gated_f32)?;

        // --- Cross-attention ---
        let (sh_ca, sc_ca, ga_ca) = self.adaln_modulation_chunk(
            emb_b_t_d, adaln_lora_b_t_3d, block_idx, "cross_attn",
        )?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;
        let x_mod = self.apply_layer_norm_modulate(&x_bf16, &sh_ca, &sc_ca)?;
        let cross_out = self.cross_attention(&x_mod, text_context, image_context, block_idx)?;
        let cross_out_gated_f32 = self
            .apply_gate(&cross_out, &ga_ca)?
            .to_dtype(DType::F32)?;
        x_f32 = x_f32.add(&cross_out_gated_f32)?;

        // --- MLP ---
        let (sh_mlp, sc_mlp, ga_mlp) = self.adaln_modulation_chunk(
            emb_b_t_d, adaln_lora_b_t_3d, block_idx, "mlp",
        )?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;
        let x_mod = self.apply_layer_norm_modulate(&x_bf16, &sh_mlp, &sc_mlp)?;
        let mlp_out = self.mlp(&x_mod, block_idx)?;
        let mlp_out_gated_f32 = self
            .apply_gate(&mlp_out, &ga_mlp)?
            .to_dtype(DType::F32)?;
        x_f32 = x_f32.add(&mlp_out_gated_f32)?;

        x_f32.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Step 7 — Patchify / Unpatchify / Patch embed / FinalLayer / forward
    // -----------------------------------------------------------------------
    //
    // Python source:
    //   - `PatchEmbed.forward` (`minimal_v4_dit.py:1023`): inner Rearrange
    //         "b c (t r) (h m) (w n) -> b t h w (c r m n)"
    //     followed by `Linear(in_c*p_s²*p_t -> hidden, bias=False)` at
    //     `self.proj.1`. Channels vary slowest, then temporal patch index,
    //     then spatial-h patch, then spatial-w patch. So the inner-dim layout
    //     is `[c0_r0_m0_n0, c0_r0_m0_n1, ...]`, i.e. column "w-patch" varies
    //     fastest, "c" varies slowest.
    //   - `MiniTrainDIT.unpatchify` (`:1702`): inverse rearrange
    //         "b t h w (p1 p2 t' c) -> b c (t t') (h p1) (w p2)"
    //     Inverse axis order: the trailing patch dim is decomposed as
    //     (p1=spatial_h, p2=spatial_w, t'=temporal, c=out_channels), with `c`
    //     varying slowest. p1, p2, t' are spatial_patch_size, spatial_patch_size,
    //     temporal_patch_size respectively.
    //   - `MiniTrainDIT.forward` (`:1712`): full pass — concat padding mask,
    //     patchify, x_embedder, prepare_timestep, learnable_pos_emb (once),
    //     loop 28 blocks (each adds extra_per_block_pos_emb), FinalLayer,
    //     unpatchify.

    /// Patchify `[B, C, T, H, W]` → `[B, T_p, H_p, W_p, C * p_s² * p_t]`.
    ///
    /// Python rearrange (`PatchEmbed.forward`):
    ///   `"b c (t r) (h m) (w n) -> b t h w (c r m n)"`
    /// where `r = patch_temporal`, `m = patch_spatial`, `n = patch_spatial`.
    ///
    /// Concretely:
    /// - axis 0 of input (B)            stays as axis 0 of output
    /// - axis 1 (C)                    becomes the slowest part of the inner patch dim
    /// - axis 2 (T)                    decomposes into outer T_p and inner `r` (varying medium-slow in patch dim)
    /// - axis 3 (H)                    decomposes into outer H_p and inner `m` (varying medium-fast)
    /// - axis 4 (W)                    decomposes into outer W_p and inner `n` (varying fastest)
    ///
    /// The inner patch dim is therefore `(c, r, m, n)` with `c` slowest and
    /// `n` fastest — so a stride-aware reshape after the (c, T_p, r, H_p, m, W_p, n)
    /// permute produces the right byte layout. Caller must guarantee
    /// `T % patch_temporal == 0` and `H % patch_spatial == 0`, `W % patch_spatial == 0`.
    ///
    /// Input `[B, C, T, H, W]` BF16 contiguous. Output `[B, T_p, H_p, W_p,
    /// C * p_s² * p_t]` BF16 contiguous.
    pub fn patchify(&self, x_b_c_t_h_w: &Tensor) -> Result<Tensor> {
        let dims = x_b_c_t_h_w.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "patchify: expected [B,C,T,H,W], got {:?}", dims
            )));
        }
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let p_s = self.config.patch_spatial;
        let p_t = self.config.patch_temporal;
        if t % p_t != 0 || h % p_s != 0 || w % p_s != 0 {
            return Err(Error::InvalidInput(format!(
                "patchify: T={t} must be divisible by patch_temporal={p_t} \
                 and H={h}, W={w} by patch_spatial={p_s}"
            )));
        }
        let t_p = t / p_t;
        let h_p = h / p_s;
        let w_p = w / p_s;
        let patch_dim = c * p_s * p_s * p_t;

        // [B, C, T_p, r, H_p, m, W_p, n]
        let x_r = x_b_c_t_h_w.reshape(&[b, c, t_p, p_t, h_p, p_s, w_p, p_s])?;
        // Permute to put outer dims first then (c, r, m, n) trailing as the
        // patch dim — Python einops puts c slowest, then r, m, n.
        //  axes:           0=B, 1=C, 2=T_p, 3=r, 4=H_p, 5=m, 6=W_p, 7=n
        // target order:    0,    2,    4,    6,   1,   3,   5,   7
        // i.e.            [B, T_p, H_p, W_p, C, r, m, n]
        let x_p = x_r.permute(&[0, 2, 4, 6, 1, 3, 5, 7])?;
        x_p.reshape(&[b, t_p, h_p, w_p, patch_dim])
    }

    /// Unpatchify `[B, T_p, H_p, W_p, p_s² * p_t * out_c]` → `[B, out_c, T, H, W]`.
    ///
    /// Inverse of `patchify`. Python rearrange
    /// (`MiniTrainDIT.unpatchify`):
    ///   `"B T H W (p1 p2 t' c) -> B c (T t') (H p1) (W p2)"`
    /// — patch dim decomposes as `(p1=spatial_h, p2=spatial_w, t'=temporal, c=out_channels)`,
    /// with `c` SLOWEST. WARNING: the patchify side puts axes as `(c, r, m, n)`,
    /// the unpatchify side ASSUMES `(p1, p2, t', c)`. These are different
    /// orderings of the same patch-dim symbols! That asymmetry is in the
    /// Python source — the FinalLayer's output Linear is initialized
    /// consistently with the unpatchify's expected layout, NOT with the
    /// patchify's. We mirror the Python verbatim.
    pub fn unpatchify(&self, x_b_t_h_w_o: &Tensor) -> Result<Tensor> {
        let dims = x_b_t_h_w_o.shape().dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "unpatchify: expected [B,T_p,H_p,W_p,O], got {:?}", dims
            )));
        }
        let (b, t_p, h_p, w_p, o) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let p_s = self.config.patch_spatial;
        let p_t = self.config.patch_temporal;
        let out_c = self.config.out_channels;
        if o != p_s * p_s * p_t * out_c {
            return Err(Error::InvalidInput(format!(
                "unpatchify: trailing patch dim {o} != p_s²*p_t*out_c = {}*{}*{}*{} = {}",
                p_s, p_s, p_t, out_c, p_s*p_s*p_t*out_c
            )));
        }

        // Decompose trailing dim into (p1, p2, t', c):
        // [B, T_p, H_p, W_p, p1, p2, t', c]
        let x_r = x_b_t_h_w_o.reshape(&[b, t_p, h_p, w_p, p_s, p_s, p_t, out_c])?;
        // Python target: "b c (t t') (h p1) (w p2)" — output layout
        // [B, c, T_p, t', H_p, p1, W_p, p2].
        //  axes:           0=B, 1=T_p, 2=H_p, 3=W_p, 4=p1, 5=p2, 6=t', 7=c
        // target order:    0,    7,   1,   6,   2,   4,   3,   5
        let x_p = x_r.permute(&[0, 7, 1, 6, 2, 4, 3, 5])?;
        // Flatten: [B, c, T_p*t', H_p*p1, W_p*p2]
        x_p.reshape(&[b, out_c, t_p * p_t, h_p * p_s, w_p * p_s])
    }

    /// Apply the `x_embedder` patch-embedding Linear (`x_embedder.proj.1.weight`
    /// per `PatchEmbed.proj = Sequential(Rearrange, Linear)`).
    ///
    /// Input: `[B, T_p, H_p, W_p, patch_dim]` BF16 where
    /// `patch_dim = in_c * p_s² * p_t` (= 68 for V2_2B with concat_padding_mask).
    /// Output: `[B, T_p, H_p, W_p, model_channels]` BF16.
    pub fn x_embedder(&self, x: &Tensor) -> Result<Tensor> {
        self.linear_no_bias(x, "x_embedder.proj.1.weight")
    }

    /// `FinalLayer.forward` (`minimal_v4_dit.py:1097-1127`):
    ///
    ///   1. `adaln_modulation(emb)` = SiLU → Linear(D→256) → Linear(256→2D)
    ///   2. `(adaln_modulation_out + adaln_lora_B_T_3D[:, :, :2*D]).chunk(2, dim=-1)`
    ///      → `(shift, scale)`, each `[B, T, D]`. NOTE the FinalLayer is
    ///      **2-chunk** (no gate), and reads only the FIRST `2*D` of the
    ///      `3*D`-wide `adaln_lora_B_T_3D` produced by `prepare_timestep`.
    ///   3. `LayerNorm(x, elementwise_affine=False, eps=1e-6) * (1+scale) + shift`
    ///   4. Final `Linear(D → p_s² * p_t * out_c, bias=False)` at
    ///      `final_layer.linear.weight`.
    ///
    /// Returns `[B, T_p, H_p, W_p, p_s² * p_t * out_c]` BF16 (caller then
    /// runs `unpatchify`).
    ///
    /// Weight keys consumed:
    ///   - `final_layer.adaln_modulation.1.weight`  [256, model_channels]
    ///   - `final_layer.adaln_modulation.2.weight`  [2*model_channels, 256]
    ///   - `final_layer.linear.weight`              [p_s²*p_t*out_c, model_channels]
    pub fn final_layer(
        &self,
        x_b_t_h_w_d: &Tensor,
        emb_b_t_d: &Tensor,
        adaln_lora_b_t_3d: &Tensor,
    ) -> Result<Tensor> {
        let d = self.config.model_channels;

        // SiLU + Linear(D, 256) + Linear(256, 2D)
        let h0 = emb_b_t_d.silu()?;
        let h1 = self.linear_no_bias(&h0, "final_layer.adaln_modulation.1.weight")?;
        let h2 = self.linear_no_bias(&h1, "final_layer.adaln_modulation.2.weight")?;

        // Add the first 2*D of adaln_lora (Python `:1110`).
        let adaln_2d = adaln_lora_b_t_3d.narrow(2, 0, 2 * d)?;
        let summed = h2.add(&adaln_2d)?;

        // Chunk(2, dim=-1) → (shift, scale)
        let shift = summed.narrow(2, 0, d)?;
        let scale = summed.narrow(2, d, d)?;

        // LayerNorm(elementwise_affine=False) * (1+scale) + shift — reuse the
        // existing 5D modulate helper (no-affine LN + fused modulate).
        let x_mod = self.apply_layer_norm_modulate(x_b_t_h_w_d, &shift, &scale)?;

        // Final Linear (D → p_s²*p_t*out_c, no bias).
        self.linear_no_bias(&x_mod, "final_layer.linear.weight")
    }

    /// Default padding mask when caller passes `None` — Python inference uses
    /// `torch.zeros(B, 1, H, W)` (no padding anywhere). Caller passes pixel
    /// `H, W` here (NOT post-patch); they must already match the spatial dims
    /// of the latent the model is denoising. Returns BF16 `[B, 1, H, W]`.
    fn default_padding_mask(&self, b: usize, h: usize, w: usize) -> Result<Tensor> {
        Tensor::zeros_dtype(
            Shape::from_dims(&[b, 1, h, w]),
            DType::BF16,
            self.device.clone(),
        )
    }

    /// `MiniTrainDIT.forward` (with optional `MinimalV1LVGDiT` wrapper) —
    /// full forward pass.
    ///
    /// Inputs (matching Python `MinimalV1LVGDiT.forward` + parent
    /// `MiniTrainDIT.forward:1712`):
    /// - `x_b_c_t_h_w`: latent video `[B, in_channels=16, T, H, W]` BF16.
    /// - `timesteps_b_t`: `[B, T]` timesteps (Python `:1751` unsqueezes 1D).
    /// - `crossattn_emb`: text context `[B, S_txt, crossattn_emb_channels]`
    ///   (or `crossattn_proj_in_channels` when `use_crossattn_projection=true`).
    /// - `condition_video_input_mask`: REQUIRED when `lvg_wrapper=true` —
    ///   `[B, 1, T, H, W]` BF16 mask marking conditioning frames. The shipped
    ///   production checkpoint expects this. When `lvg_wrapper=false`,
    ///   the argument is ignored.
    /// - `padding_mask`: optional `[B, 1, H, W]` mask (parent
    ///   `MiniTrainDIT` adds this). When `None`, defaults to zeros.
    /// - `fps`: optional FPS scalar for 3D RoPE temporal modulation.
    /// - `image_context`: optional `[B, S_img, model_channels]` BF16 image
    ///   context for the I2V dual-K/V branch (only when
    ///   `config.extra_image_context_dim.is_some()`).
    ///
    /// Concat order (Python-mirrored):
    ///   1. `cat([x, condition_video_input_mask], dim=1)` — LVG wrapper, Python
    ///      `minimal_v1_lvg_dit.py:47`. Only when `lvg_wrapper=true`.
    ///   2. `cat([x, padding_mask.broadcast(T)], dim=1)` — parent
    ///      `MiniTrainDIT`, Python `minimal_v4_dit.py:1686`. Only when
    ///      `concat_padding_mask=true`.
    /// A single `.contiguous()` is applied after the LAST cat (the one
    /// allowed `.contiguous()` in forward).
    ///
    /// Returns `[B, out_channels=16, T, H, W]` BF16 — the velocity prediction.
    ///
    /// Magnitude probe: each block's output L∞ norm is logged at `debug`
    /// level (see `RUST_LOG=debug`). Tracks the anima oracle's BF16/F32
    /// divergence concern.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x_b_c_t_h_w: &Tensor,
        timesteps_b_t: &Tensor,
        crossattn_emb: &Tensor,
        condition_video_input_mask: Option<&Tensor>,
        padding_mask: Option<&Tensor>,
        fps: Option<f32>,
        image_context: Option<&Tensor>,
    ) -> Result<Tensor> {
        let in_dims = x_b_c_t_h_w.shape().dims().to_vec();
        if in_dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "forward: expected x as [B,C,T,H,W], got {:?}", in_dims
            )));
        }
        let (b, c_in, t, h, w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3], in_dims[4]);
        if c_in != self.config.in_channels {
            return Err(Error::InvalidInput(format!(
                "forward: input channels {c_in} != config.in_channels {}",
                self.config.in_channels
            )));
        }

        // ----- 1a. LVG wrapper concat (FIRST, when enabled) -----
        // Python `minimal_v1_lvg_dit.py:46-47`:
        //     x = torch.cat([x, condition_video_input_mask.type_as(x)], dim=1)
        // The mask is `[B, 1, T, H, W]` already (no broadcast needed).
        // When `lvg_wrapper=true` this MUST be provided; for inference we
        // soften the requirement and synthesize a zeros mask if omitted
        // (image-mode default — Python falls into the `else` branch that
        // concats zeros).
        let after_lvg: Tensor = if self.config.lvg_wrapper {
            let owned_lvg: Tensor;
            let lvg_5d: &Tensor = match condition_video_input_mask {
                Some(m) => {
                    let md = m.shape().dims();
                    if md.len() != 5 || md[0] != b || md[1] != 1 || md[2] != t
                        || md[3] != h || md[4] != w {
                        return Err(Error::InvalidInput(format!(
                            "forward: condition_video_input_mask must be \
                             [B={b}, 1, T={t}, H={h}, W={w}], got {:?}", md
                        )));
                    }
                    m
                }
                None => {
                    // image-mode fallback: zeros mask (Python lvg_dit.py:49-52).
                    owned_lvg = Tensor::zeros_dtype(
                        Shape::from_dims(&[b, 1, t, h, w]),
                        x_b_c_t_h_w.dtype(),
                        self.device.clone(),
                    )?;
                    &owned_lvg
                }
            };
            // Contiguize each cat to be defensive: flame-core's `Tensor::cat`
            // doesn't guarantee contig output (CONTEXT.md known trap), and the
            // LVG concat feeds into the parent padding-mask cat below. If both
            // remain non-contig, downstream conv/Linear reads garbage at edges.
            let x_contig = if x_b_c_t_h_w.is_contiguous() {
                x_b_c_t_h_w.clone()
            } else {
                x_b_c_t_h_w.contiguous()?
            };
            let lvg_contig = if lvg_5d.is_contiguous() {
                lvg_5d.clone()
            } else {
                lvg_5d.contiguous()?
            };
            Tensor::cat(&[&x_contig, &lvg_contig], 1)?.contiguous()?
        } else {
            // No-op clone — caller's tensor is the starting point.
            x_b_c_t_h_w.clone()
        };

        // ----- 1b. Parent MiniTrainDIT padding-mask concat -----
        // Python `:1682-1688`. `padding_mask` is `[B, 1, H, W]` (no T dim);
        // expanded to `[B, 1, T, H, W]` and cat'd along channel axis 1.
        let x_with_mask = if self.config.concat_padding_mask {
            let default_owned: Tensor;
            let mask_4d: &Tensor = match padding_mask {
                Some(m) => {
                    let md = m.shape().dims();
                    if md.len() != 4 || md[0] != b || md[1] != 1 || md[2] != h || md[3] != w {
                        return Err(Error::InvalidInput(format!(
                            "forward: padding_mask must be [B={b}, 1, H={h}, W={w}], got {:?}", md
                        )));
                    }
                    m
                }
                None => {
                    default_owned = self.default_padding_mask(b, h, w)?;
                    &default_owned
                }
            };
            let mask_5d = mask_4d.reshape(&[b, 1, 1, h, w])?
                .broadcast_to(&Shape::from_dims(&[b, 1, t, h, w]))?
                .contiguous()?;
            let after_lvg_contig = if after_lvg.is_contiguous() {
                after_lvg.clone()
            } else {
                after_lvg.contiguous()?
            };
            Tensor::cat(&[&after_lvg_contig, &mask_5d], 1)?.contiguous()?
        } else if self.config.lvg_wrapper {
            // LVG cat happened but no padding mask — contiguize once here.
            after_lvg.contiguous()?
        } else {
            after_lvg
        };

        self.forward_inner(&x_with_mask, timesteps_b_t, crossattn_emb,
                           fps, image_context)
    }

    /// Inner forward dispatcher — assumes any padding-mask concat already
    /// happened (caller passes the channel-expanded latent if
    /// `concat_padding_mask=true`). The channel count of
    /// `x_with_optional_mask` is what determines patchify.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        x_with_optional_mask: &Tensor,
        timesteps_b_t: &Tensor,
        crossattn_emb: &Tensor,
        fps: Option<f32>,
        image_context: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dims = x_with_optional_mask.shape().dims().to_vec();
        let (b, _c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // ----- 1b. Optional crossattn projection -----
        // Python `:1738-1739`:
        //     if self.use_crossattn_projection:
        //         crossattn_emb = self.crossattn_proj(crossattn_emb)
        // Applied BEFORE the block loop so all 28 blocks see the projected
        // 1024-dim tensor. The base `cosmos_v2_2b()` skips this; production
        // checkpoints enable it (FULL_CONCAT 100352 → 1024).
        let projected_text;
        let crossattn_emb: &Tensor = if self.config.use_crossattn_projection {
            projected_text = self.apply_crossattn_proj(crossattn_emb)?;
            &projected_text
        } else {
            crossattn_emb
        };

        // ----- 2. Patchify -----
        // [B, C, T, H, W] → [B, T_p, H_p, W_p, patch_dim]
        let x_patches = self.patchify(x_with_optional_mask)?;
        let dims = x_patches.shape().dims();
        let (t_p, h_p, w_p) = (dims[1], dims[2], dims[3]);

        // ----- 3. Patch embed (Linear[patch_dim → model_channels]) -----
        let mut x_b_t_h_w_d = self.x_embedder(&x_patches)?;

        // ----- 4. Timestep conditioning -----
        let (t_emb, adaln_lora_opt) = self.prepare_timestep(timesteps_b_t)?;
        let adaln_lora = adaln_lora_opt.ok_or_else(|| Error::InvalidInput(
            "forward: V2_2B requires use_adaln_lora=true but prepare_timestep \
             returned None for adaln_lora_b_t_3d".to_string()
        ))?;

        // ----- 5. Build 3D RoPE cos/sin tables ONCE (shared across blocks) -----
        let head_dim = self.config.head_dim();
        let (rope_cos, rope_sin) = build_cosmos_rope_freqs(
            head_dim, t_p, h_p, w_p,
            fps, self.config.base_fps as f32,
            self.config.rope_h_extrapolation_ratio,
            self.config.rope_w_extrapolation_ratio,
            self.config.rope_t_extrapolation_ratio,
            self.config.rope_enable_fps_modulation,
            &self.device,
        )?;

        // ----- 6. Build extra_per_block_pos_emb ONCE (added inside each block) -----
        let extra_pos_emb_opt: Option<Tensor> = if self.config.extra_per_block_abs_pos_emb {
            Some(self.learnable_pos_emb(b, t_p, h_p, w_p)?)
        } else {
            None
        };

        // ----- 7. Loop over `num_blocks` transformer blocks -----
        // Magnitude probe: log L∞ norm after each block to catch the anima
        // BF16/F32 divergence concern early (chunk-2 skeptic callout).
        let probe_enabled = log::log_enabled!(log::Level::Debug);

        for block_idx in 0..self.config.num_blocks {
            x_b_t_h_w_d = self.transformer_block(
                &x_b_t_h_w_d,
                &t_emb,
                &adaln_lora,
                crossattn_emb,
                image_context,
                extra_pos_emb_opt.as_ref(),
                &rope_cos,
                &rope_sin,
                block_idx,
            )?;

            if probe_enabled {
                // Compute |x|.max() lazily — only when debug logging is on.
                // Cast BF16 → F32 first because `max_all` routes through
                // `GpuOps::reduce_max` (the F32-heavy reduction path).
                let probe = x_b_t_h_w_d
                    .to_dtype(DType::F32)
                    .and_then(|f| f.abs())
                    .and_then(|a| a.max_all());
                match probe {
                    Ok(m) => log::debug!(
                        "[cosmos_predict25] block {block_idx:02} L∞={m:.4}"
                    ),
                    Err(e) => log::debug!(
                        "[cosmos_predict25] block {block_idx:02} L∞=<err: {e}>"
                    ),
                }
            }
        }

        // ----- 8. FinalLayer (LayerNorm-no-affine + 2-chunk adaLN-LoRA + Linear) -----
        let x_final = self.final_layer(&x_b_t_h_w_d, &t_emb, &adaln_lora)?;

        // ----- 9. Unpatchify back to [B, out_c, T, H, W] -----
        let _ = (t, h, w); // input spatial — not directly needed (patchify carried these forward)
        self.unpatchify(&x_final)
    }
}

// ---------------------------------------------------------------------------
// Tests — layout / sanity checks only. No GPU parity here (handled later in
// the per-layer parity harness).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_split_head_dim_128() {
        let cfg = CosmosPredict25Config::cosmos_v2_2b();
        assert_eq!(cfg.head_dim(), 128);
        let (dim_t, dim_h, dim_w) = cfg.rope_axis_split();
        assert_eq!(dim_t, 44);
        assert_eq!(dim_h, 42);
        assert_eq!(dim_w, 42);
        assert_eq!(dim_t + dim_h + dim_w, 128);
    }

    #[test]
    fn patch_embed_in_channels_base_is_17() {
        // Base config: no LVG wrapper, padding mask only → 16 + 0 + 1 = 17.
        let cfg = CosmosPredict25Config::cosmos_v2_2b();
        assert!(!cfg.lvg_wrapper);
        assert!(cfg.concat_padding_mask);
        assert_eq!(cfg.patch_embed_in_channels(), 17);
    }

    #[test]
    fn patch_embed_in_channels_production_is_18() {
        // Production config: LVG wrapper + padding mask → 16 + 1 + 1 = 18.
        // This matches the shipped checkpoint's
        // `x_embedder.proj.1.weight` shape `[2048, 72]` where 72 = 18 * 2² * 1.
        let cfg = CosmosPredict25Config::cosmos_v2_2b_production();
        assert!(cfg.lvg_wrapper);
        assert!(cfg.concat_padding_mask);
        assert_eq!(cfg.patch_embed_in_channels(), 18);
    }

    #[test]
    fn cosmos_v2_2b_config_matches_python() {
        let cfg = CosmosPredict25Config::cosmos_v2_2b();
        assert_eq!(cfg.model_channels, 2048);
        assert_eq!(cfg.num_blocks, 28);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.patch_spatial, 2);
        assert_eq!(cfg.patch_temporal, 1);
        assert!(cfg.concat_padding_mask);
        assert!(cfg.use_adaln_lora);
        assert_eq!(cfg.adaln_lora_dim, 256);
        assert!(cfg.extra_per_block_abs_pos_emb);
        assert!(cfg.rope_enable_fps_modulation);
        assert_eq!(cfg.base_fps, 24);
    }

    /// Compute the cos/sin layout in pure CPU math (no GPU) and check the
    /// half-split / per-axis structure of the output.
    ///
    /// This mirrors what `build_cosmos_rope_freqs` writes into its `Vec<f32>`
    /// buffers — we don't push to GPU here because the unit-test environment
    /// may have no CUDA device available.
    fn build_cpu_layout(
        head_dim: usize,
        t: usize, h: usize, w: usize,
        fps: Option<f32>, base_fps: f32,
        enable_fps_modulation: bool,
    ) -> (Vec<f32>, Vec<f32>, usize, usize, usize) {
        let dim_h: usize = head_dim / 6 * 2;
        let dim_w: usize = dim_h;
        let dim_t: usize = head_dim - 2 * dim_h;
        let half_t = dim_t / 2;
        let half_h = dim_h / 2;
        let half_w = dim_w / 2;
        let half_d = head_dim / 2;

        let theta = 10000.0_f64;
        let mk_freqs = |dim_axis: usize, half: usize| -> Vec<f64> {
            (0..half).map(|k| 1.0 / theta.powf((2 * k) as f64 / dim_axis as f64)).collect()
        };
        let t_freqs = mk_freqs(dim_t, half_t);
        let h_freqs = mk_freqs(dim_h, half_h);
        let w_freqs = mk_freqs(dim_w, half_w);

        let t_pos: Vec<f64> = (0..t).map(|i| {
            if enable_fps_modulation {
                if let Some(f) = fps { (i as f64) / (f as f64) * (base_fps as f64) }
                else { i as f64 }
            } else { i as f64 }
        }).collect();
        let h_pos: Vec<f64> = (0..h).map(|i| i as f64).collect();
        let w_pos: Vec<f64> = (0..w).map(|i| i as f64).collect();

        let mut cos = vec![0.0_f32; t*h*w*half_d];
        let mut sin = vec![0.0_f32; t*h*w*half_d];
        for ti in 0..t { for hi in 0..h { for wi in 0..w {
            let row_off = (ti*h*w + hi*w + wi) * half_d;
            for k in 0..half_t {
                let a = t_pos[ti] * t_freqs[k];
                cos[row_off + k] = a.cos() as f32;
                sin[row_off + k] = a.sin() as f32;
            }
            for k in 0..half_h {
                let a = h_pos[hi] * h_freqs[k];
                cos[row_off + half_t + k] = a.cos() as f32;
                sin[row_off + half_t + k] = a.sin() as f32;
            }
            for k in 0..half_w {
                let a = w_pos[wi] * w_freqs[k];
                cos[row_off + half_t + half_h + k] = a.cos() as f32;
                sin[row_off + half_t + half_h + k] = a.sin() as f32;
            }
        }}}
        (cos, sin, half_t, half_h, half_w)
    }

    #[test]
    fn rope_layout_position_zero_is_identity() {
        // At (t=0, h=0, w=0) every angle is 0 → cos=1, sin=0.
        let (cos, sin, _, _, _) =
            build_cpu_layout(128, 2, 3, 4, Some(24.0), 24.0, true);
        let half_d = 128 / 2;
        for k in 0..half_d {
            assert!((cos[k] - 1.0).abs() < 1e-6, "cos[0,{k}] = {} != 1.0", cos[k]);
            assert!(sin[k].abs() < 1e-6,         "sin[0,{k}] = {} != 0.0", sin[k]);
        }
    }

    #[test]
    fn rope_layout_segment_sizes_sum_to_half() {
        let (cos, sin, half_t, half_h, half_w) =
            build_cpu_layout(128, 2, 3, 4, Some(24.0), 24.0, true);
        assert_eq!(half_t, 22);
        assert_eq!(half_h, 21);
        assert_eq!(half_w, 21);
        assert_eq!(half_t + half_h + half_w, 64);

        // Sanity: every entry is in [-1, 1].
        for &c in &cos { assert!(c.is_finite() && c.abs() <= 1.0 + 1e-5); }
        for &s in &sin { assert!(s.is_finite() && s.abs() <= 1.0 + 1e-5); }
    }

    #[test]
    fn rope_layout_t_segment_constant_across_hw() {
        // For a fixed t, the first `half_t` cos values should not depend on h or w.
        let (cos, _, half_t, _, _) =
            build_cpu_layout(128, 2, 3, 4, Some(24.0), 24.0, true);
        let (t, h, w) = (2_usize, 3_usize, 4_usize);
        let half_d = 128 / 2;
        for ti in 0..t {
            // pick the (ti, 0, 0) row as reference
            let ref_row = (ti * h * w) * half_d;
            for hi in 0..h {
                for wi in 0..w {
                    let row = (ti * h * w + hi * w + wi) * half_d;
                    for k in 0..half_t {
                        let a = cos[ref_row + k];
                        let b = cos[row + k];
                        assert!(
                            (a - b).abs() < 1e-6,
                            "t-segment varies with (h,w): ti={ti} hi={hi} wi={wi} k={k} \
                             {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rope_layout_w_segment_only_depends_on_w() {
        // For a fixed w, the last `half_w` cos values (after the t and h
        // segments) should not depend on t or h.
        let (cos, _, half_t, half_h, half_w) =
            build_cpu_layout(128, 2, 3, 4, Some(24.0), 24.0, true);
        let (t, h, w) = (2_usize, 3_usize, 4_usize);
        let half_d = 128 / 2;
        let w_off = half_t + half_h;
        for wi in 0..w {
            // pick (0, 0, wi) as reference
            let ref_row = wi * half_d;
            for ti in 0..t {
                for hi in 0..h {
                    let row = (ti * h * w + hi * w + wi) * half_d;
                    for k in 0..half_w {
                        let a = cos[ref_row + w_off + k];
                        let b = cos[row     + w_off + k];
                        assert!(
                            (a - b).abs() < 1e-6,
                            "w-segment varies with (t,h): wi={wi} ti={ti} hi={hi} k={k} \
                             {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // F1 unit test (learnable_pos_emb normalization). GPU-gated.
    // -----------------------------------------------------------------------

    fn maybe_cuda_device() -> Option<Arc<cudarc::driver::CudaDevice>> {
        cudarc::driver::CudaDevice::new(0).ok()
    }

    #[test]
    fn learnable_pos_emb_applies_output_norm() {
        // Verifies F1: `learnable_pos_emb` applies the per-row L2-norm
        // scaling that Python's `LearnablePosEmbAxis.generate_embeddings`
        // (minimal_v4_dit.py:850-852) does before returning. Without the
        // scaling the output equals `pe_t + pe_h + pe_w` (raw sum); with
        // the scaling the magnitudes are bounded near ~sqrt(D).
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return, // CPU-only CI box → silent skip (anima pattern)
        };

        let d: usize = 16;
        let t_max: usize = 4;
        let h_max: usize = 4;
        let w_max: usize = 4;

        // Build a stub model with synthetic per-axis pos emb buffers full of 1.0
        // (so unnormalized sum = 3.0 in every cell, easy to compare).
        let cfg = CosmosPredict25Config {
            model_channels: d,
            extra_per_block_abs_pos_emb: true,
            ..CosmosPredict25Config::cosmos_v2_2b()
        };

        let mk = |rows: usize| {
            Tensor::from_vec(
                vec![1.0_f32; rows * d],
                Shape::from_dims(&[rows, d]),
                dev.clone(),
            ).expect("from_vec")
        };

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("extra_pos_embedder.pos_emb_t".to_string(), mk(t_max));
        weights.insert("extra_pos_embedder.pos_emb_h".to_string(), mk(h_max));
        weights.insert("extra_pos_embedder.pos_emb_w".to_string(), mk(w_max));

        let model = CosmosPredict25Dit { config: cfg, device: dev.clone(), weights };

        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let out = model.learnable_pos_emb(b, t, h, w).expect("learnable_pos_emb");

        // Shape sanity.
        assert_eq!(out.shape().dims(), &[1, t, h, w, d]);

        // Output values: with pe_*=1 everywhere, unnormalized sum is 3.0 per
        // cell. After normalization Python computes:
        //   ||emb||_2 over last dim = sqrt(D * 3^2) = 3 * sqrt(D)
        //   denom = 1e-6 + (1/sqrt(D)) * (3 * sqrt(D)) = 1e-6 + 3 ≈ 3
        //   out = 3 / 3 ≈ 1.0
        // So every entry should be ≈ 1.0 (not 3.0).
        let data = out.to_vec().expect("to_vec");
        let expected: f32 = 3.0 / (1e-6 + 3.0);
        let mut max_abs_err: f32 = 0.0;
        for &v in &data {
            let err = (v - expected).abs();
            if err > max_abs_err { max_abs_err = err; }
        }
        // BF16 round-trip — tolerate 1e-2 absolute. Crucially, the un-normalized
        // value would be ~3.0, far outside this tolerance, so this catches a
        // regression that drops the normalization.
        assert!(
            max_abs_err < 1e-2,
            "learnable_pos_emb output not normalized: max_abs_err={} \
             (expected entries ≈ {}, but unnormalized would be 3.0)",
            max_abs_err, expected
        );
        assert!(
            (data[0] - 3.0).abs() > 0.5,
            "learnable_pos_emb appears to return unnormalized sum: data[0]={}",
            data[0]
        );
    }

    #[test]
    fn rope_layout_fps_modulation_changes_t_segment_only() {
        let (cos_a, _, half_t, _, _) =
            build_cpu_layout(128, 4, 1, 1, Some(24.0), 24.0, true);
        let (cos_b, _, _, _, _) =
            build_cpu_layout(128, 4, 1, 1, Some(12.0), 24.0, true); // half the fps -> doubled t-pos
        let half_d = 128 / 2;
        // At t=0 still everything matches.
        for k in 0..half_d {
            assert!((cos_a[k] - cos_b[k]).abs() < 1e-6);
        }
        // At t=2, the t-segment differs (because angle doubles); the rest is
        // (H=1, W=1) → all zero angles → identical.
        let row = 2 * half_d;
        // h+w segments still 1.0:
        for k in half_t..half_d {
            assert!((cos_a[row + k] - 1.0).abs() < 1e-6);
            assert!((cos_b[row + k] - 1.0).abs() < 1e-6);
        }
        // Some t-segment entry must differ:
        let any_diff = (0..half_t).any(|k| (cos_a[row + k] - cos_b[row + k]).abs() > 1e-3);
        assert!(any_diff, "fps modulation should change the t-segment");
    }

    // -----------------------------------------------------------------------
    // Chunk 2 — attention / mlp / transformer_block tests. All GPU-gated.
    //
    // Strategy per skeptic finding F4: exercise the *real production
    // functions* (`self_attention`, `cross_attention`, `mlp`,
    // `transformer_block`), not a CPU duplicate. We construct stub models
    // with synthetic weights at small shapes, run on GPU when available,
    // and silent-skip when no CUDA device is present (anima pattern).
    // -----------------------------------------------------------------------

    /// Tiny test config: `model_channels=12, num_heads=2, head_dim=6` (smallest
    /// head_dim that satisfies the divisible-by-6 rope-axis constraint).
    /// Crossattn dim is reduced to 8 to keep weights tiny.
    fn tiny_test_config() -> CosmosPredict25Config {
        CosmosPredict25Config {
            max_img_h: 16,
            max_img_w: 16,
            max_frames: 16,
            in_channels: 16,
            out_channels: 16,
            patch_spatial: 2,
            patch_temporal: 1,
            concat_padding_mask: true,
            lvg_wrapper: false,
            model_channels: 12,
            num_blocks: 1,
            num_heads: 2,
            mlp_ratio: 4.0,
            crossattn_emb_channels: 8,
            extra_image_context_dim: None,
            use_crossattn_projection: false,
            crossattn_proj_in_channels: 8,
            pos_emb_cls: PosEmbCls::Rope3d,
            pos_emb_interpolation: PosEmbInterp::Crop,
            pos_emb_learnable: true,
            min_fps: 1,
            max_fps: 30,
            base_fps: 24,
            use_adaln_lora: true,
            adaln_lora_dim: 4,
            rope_h_extrapolation_ratio: 1.0,
            rope_w_extrapolation_ratio: 1.0,
            rope_t_extrapolation_ratio: 1.0,
            rope_enable_fps_modulation: true,
            extra_per_block_abs_pos_emb: true,
            extra_h_extrapolation_ratio: 1.0,
            extra_w_extrapolation_ratio: 1.0,
            extra_t_extrapolation_ratio: 1.0,
        }
    }

    /// Build a deterministic BF16 weight tensor of the given shape. Values are
    /// a small sine wave seeded by the key string, so two calls with the same
    /// key produce the same data but different keys differ.
    fn mk_weight(
        dev: &Arc<cudarc::driver::CudaDevice>,
        key: &str,
        shape: &[usize],
    ) -> Tensor {
        let n: usize = shape.iter().product();
        // Tiny scaled values so we stay within BF16 range and softmax stays sane.
        let seed: u64 = key
            .bytes()
            .fold(0xC05_905u64, |acc, b| acc.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(b as u64));
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64).mul_add(0.013, (seed & 0xFFFF) as f64 * 1e-5);
            let v = (t.sin() * 0.05) as f32; // small magnitude
            data.push(v);
        }
        let t = Tensor::from_vec(data, Shape::from_dims(shape), dev.clone()).expect("from_vec");
        t.to_dtype(DType::BF16).expect("bf16")
    }

    /// Build a deterministic BF16 ones-ish tensor.
    fn mk_ones_weight(
        dev: &Arc<cudarc::driver::CudaDevice>,
        shape: &[usize],
    ) -> Tensor {
        let n: usize = shape.iter().product();
        let t = Tensor::from_vec(vec![1.0_f32; n], Shape::from_dims(shape), dev.clone())
            .expect("from_vec");
        t.to_dtype(DType::BF16).expect("bf16")
    }

    /// Populate the weights map for one block (block_idx=0). Self-attn,
    /// cross-attn, mlp, adaln_modulation_* for self/cross/mlp.
    fn populate_block_weights(
        weights: &mut HashMap<String, Tensor>,
        dev: &Arc<cudarc::driver::CudaDevice>,
        cfg: &CosmosPredict25Config,
        block_idx: usize,
    ) {
        let d = cfg.model_channels;
        let head_dim = cfg.head_dim();
        let inner = cfg.num_heads * head_dim;
        let ctx = cfg.crossattn_emb_channels;
        let mlp_hidden = (d as f32 * cfg.mlp_ratio) as usize;
        let alora = cfg.adaln_lora_dim;

        // Self-attn
        let pfx = format!("blocks.{block_idx}.self_attn");
        weights.insert(format!("{pfx}.q_proj.weight"), mk_weight(dev, &format!("{pfx}.q"), &[inner, d]));
        weights.insert(format!("{pfx}.k_proj.weight"), mk_weight(dev, &format!("{pfx}.k"), &[inner, d]));
        weights.insert(format!("{pfx}.v_proj.weight"), mk_weight(dev, &format!("{pfx}.v"), &[inner, d]));
        weights.insert(format!("{pfx}.q_norm.weight"), mk_ones_weight(dev, &[head_dim]));
        weights.insert(format!("{pfx}.k_norm.weight"), mk_ones_weight(dev, &[head_dim]));
        weights.insert(format!("{pfx}.output_proj.weight"), mk_weight(dev, &format!("{pfx}.o"), &[d, inner]));

        // Cross-attn
        let pfx = format!("blocks.{block_idx}.cross_attn");
        weights.insert(format!("{pfx}.q_proj.weight"), mk_weight(dev, &format!("{pfx}.q"), &[inner, d]));
        weights.insert(format!("{pfx}.k_proj.weight"), mk_weight(dev, &format!("{pfx}.k"), &[inner, ctx]));
        weights.insert(format!("{pfx}.v_proj.weight"), mk_weight(dev, &format!("{pfx}.v"), &[inner, ctx]));
        weights.insert(format!("{pfx}.q_norm.weight"), mk_ones_weight(dev, &[head_dim]));
        weights.insert(format!("{pfx}.k_norm.weight"), mk_ones_weight(dev, &[head_dim]));
        weights.insert(format!("{pfx}.output_proj.weight"), mk_weight(dev, &format!("{pfx}.o"), &[d, inner]));
        // I2V branch (only inserted when test enables extra_image_context_dim).
        if cfg.extra_image_context_dim.is_some() {
            weights.insert(format!("{pfx}.k_img.weight"), mk_weight(dev, &format!("{pfx}.ki"), &[inner, d]));
            weights.insert(format!("{pfx}.v_img.weight"), mk_weight(dev, &format!("{pfx}.vi"), &[inner, d]));
            weights.insert(format!("{pfx}.k_img_norm.weight"), mk_ones_weight(dev, &[head_dim]));
        }

        // MLP
        let pfx = format!("blocks.{block_idx}.mlp");
        weights.insert(format!("{pfx}.layer1.weight"), mk_weight(dev, &format!("{pfx}.l1"), &[mlp_hidden, d]));
        weights.insert(format!("{pfx}.layer2.weight"), mk_weight(dev, &format!("{pfx}.l2"), &[d, mlp_hidden]));

        // adaLN modulation (per sub-block)
        for sub in ["self_attn", "cross_attn", "mlp"].iter() {
            let pfx = format!("blocks.{block_idx}.adaln_modulation_{sub}");
            weights.insert(format!("{pfx}.1.weight"), mk_weight(dev, &format!("{pfx}.1"), &[alora, d]));
            weights.insert(format!("{pfx}.2.weight"), mk_weight(dev, &format!("{pfx}.2"), &[3 * d, alora]));
        }
    }

    /// Build a fresh `CosmosPredict25Dit` with the tiny test config and a
    /// single block's worth of weights.
    fn make_stub_model(
        dev: Arc<cudarc::driver::CudaDevice>,
        cfg: CosmosPredict25Config,
    ) -> CosmosPredict25Dit {
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        populate_block_weights(&mut weights, &dev, &cfg, 0);
        CosmosPredict25Dit { config: cfg, device: dev, weights }
    }

    /// Test that `self_attention` runs end-to-end and produces a finite
    /// `[B, T, H, W, D]` output with the right shape. We use a tiny grid and
    /// the real `build_cosmos_rope_freqs` for cos/sin.
    #[test]
    fn self_attention_forward_runs() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let head_dim = cfg.head_dim();
        let d = cfg.model_channels;
        let model = make_stub_model(dev.clone(), cfg);

        // Build real RoPE cos/sin.
        let (cos, sin) = build_cosmos_rope_freqs(
            head_dim, t, h, w,
            Some(24.0), 24.0, 1.0, 1.0, 1.0, true, &dev,
        ).expect("rope freqs");

        // Stub input.
        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let out = model.self_attention(&x, &cos, &sin, 0).expect("self_attention");
        assert_eq!(out.shape().dims(), &[b, t, h, w, d]);

        // Finite check.
        let data = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "self_attention out[{i}] = {v} non-finite");
        }
    }

    /// Test that `cross_attention` runs for the T2V path (image_context=None,
    /// extra_image_context_dim=None). Also test that passing an image_context
    /// when the model is configured without `extra_image_context_dim` is a
    /// silent no-op (the branch is skipped).
    #[test]
    fn cross_attention_t2v_runs_and_ignores_image_context() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();   // extra_image_context_dim = None
        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let ctx = cfg.crossattn_emb_channels;
        let s_txt = 5_usize;
        let model = make_stub_model(dev.clone(), cfg);

        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let ctx_data: Vec<f32> = (0..b * s_txt * ctx).map(|i| (i as f32) * 0.002).collect();
        let text_ctx = Tensor::from_vec(ctx_data, Shape::from_dims(&[b, s_txt, ctx]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        // T2V path.
        let out_t2v = model.cross_attention(&x, &text_ctx, None, 0).expect("cross_attn t2v");
        assert_eq!(out_t2v.shape().dims(), &[b, t, h, w, d]);

        // Pass image_context anyway — with extra_image_context_dim=None the
        // branch must be skipped, output should equal the t2v output.
        let img_data: Vec<f32> = vec![1.0; b * 3 * d];
        let img_ctx = Tensor::from_vec(img_data, Shape::from_dims(&[b, 3, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");
        let out_with_img = model.cross_attention(&x, &text_ctx, Some(&img_ctx), 0)
            .expect("cross_attn with ignored img");
        // Outputs should match (the image branch was skipped).
        let a = out_t2v.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_with_img.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let d = (x - y).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff == 0.0,
            "cross_attention with image_context but no extra_image_context_dim \
             must equal t2v output, max_diff={max_diff}");
    }

    /// Test that `cross_attention` exercises the I2V dual-K/V branch when the
    /// model is configured with `extra_image_context_dim`. The branch must
    /// change the output (otherwise the second SDPA + add is dead code).
    #[test]
    fn cross_attention_i2v_changes_output() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = CosmosPredict25Config {
            // Same as tiny_test_config but enable the i2v branch.
            extra_image_context_dim: Some(12),
            ..tiny_test_config()
        };
        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let ctx = cfg.crossattn_emb_channels;
        let s_txt = 5_usize;
        let s_img = 3_usize;
        let model = make_stub_model(dev.clone(), cfg);

        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let ctx_data: Vec<f32> = (0..b * s_txt * ctx).map(|i| (i as f32) * 0.002).collect();
        let text_ctx = Tensor::from_vec(ctx_data, Shape::from_dims(&[b, s_txt, ctx]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        // Image context. NOTE: the block expects [B, S_img, model_channels],
        // i.e. already projected from `extra_image_context_dim` through
        // `img_context_proj` by the caller. So shape is `[B, S_img, D]`.
        let img_data: Vec<f32> = (0..b * s_img * d).map(|i| 0.5 + (i as f32) * 0.003).collect();
        let img_ctx = Tensor::from_vec(img_data, Shape::from_dims(&[b, s_img, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        // Without image branch.
        let out_text_only = model.cross_attention(&x, &text_ctx, None, 0)
            .expect("cross_attn text only");
        // With image branch.
        let out_with_img = model.cross_attention(&x, &text_ctx, Some(&img_ctx), 0)
            .expect("cross_attn with img");

        // Outputs must differ — otherwise the image branch is dead.
        let a = out_text_only.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_with_img.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let d = (x - y).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff > 1e-4,
            "cross_attention i2v branch is a no-op: max_diff={max_diff} (expected > 1e-4)");
        // And finite.
        for &v in &bv {
            assert!(v.is_finite(), "i2v cross_attention output non-finite");
        }
    }

    /// Test that `mlp` runs end-to-end and preserves shape. GELU should not
    /// produce NaN for finite small inputs.
    #[test]
    fn mlp_forward_runs_and_preserves_shape() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let model = make_stub_model(dev.clone(), cfg);

        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.01).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let out = model.mlp(&x, 0).expect("mlp");
        assert_eq!(out.shape().dims(), &[b, t, h, w, d]);

        let data = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "mlp out[{i}] = {v} non-finite");
        }
    }

    /// Test that `transformer_block` runs end-to-end and that the modulation
    /// path is *load-bearing*: feeding `t_cond = zero` vs `t_cond = nonzero`
    /// must produce different outputs. Otherwise the adaLN-LoRA wiring is
    /// dead (this is the classical "modulation forgotten" silent failure).
    #[test]
    fn transformer_block_modulation_is_load_bearing() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let head_dim = cfg.head_dim();
        let ctx = cfg.crossattn_emb_channels;
        let s_txt = 5_usize;
        let model = make_stub_model(dev.clone(), cfg);

        // Stub inputs.
        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let ctx_data: Vec<f32> = (0..b * s_txt * ctx).map(|i| (i as f32) * 0.002).collect();
        let text_ctx = Tensor::from_vec(ctx_data, Shape::from_dims(&[b, s_txt, ctx]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let (cos, sin) = build_cosmos_rope_freqs(
            head_dim, t, h, w,
            Some(24.0), 24.0, 1.0, 1.0, 1.0, true, &dev,
        ).expect("rope freqs");

        // adaln_lora_b_t_3d (zeros) — only the per-block adaln_modulation
        // varies between the two runs.
        let lora_zero = Tensor::from_vec(
            vec![0.0_f32; b * t * 3 * d],
            Shape::from_dims(&[b, t, 3 * d]),
            dev.clone(),
        ).expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        // Run 1: emb = zero (every modulation collapses to LN(x)*1 + 0 = LN(x))
        let emb_zero = Tensor::from_vec(
            vec![0.0_f32; b * t * d],
            Shape::from_dims(&[b, t, d]),
            dev.clone(),
        ).expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let out_zero = model.transformer_block(
            &x, &emb_zero, &lora_zero, &text_ctx, None, None, &cos, &sin, 0,
        ).expect("block zero emb");
        assert_eq!(out_zero.shape().dims(), &[b, t, h, w, d]);

        // Run 2: emb = nonzero — modulation should now produce different x_mod
        // and therefore different attn/cross/mlp outputs.
        let emb_nonzero_data: Vec<f32> = (0..b * t * d).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let emb_nonzero = Tensor::from_vec(emb_nonzero_data, Shape::from_dims(&[b, t, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");
        let out_nonzero = model.transformer_block(
            &x, &emb_nonzero, &lora_zero, &text_ctx, None, None, &cos, &sin, 0,
        ).expect("block nonzero emb");

        let a = out_zero.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_nonzero.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let dd = (x - y).abs();
            if dd > max_diff { max_diff = dd; }
        }
        assert!(max_diff > 1e-4,
            "transformer_block: t_cond appears not to affect output \
             (max_diff={max_diff}). adaLN-LoRA wiring is dead.");

        // Also verify finite.
        for &v in &bv {
            assert!(v.is_finite(), "transformer_block output non-finite");
        }
    }

    /// Test that `transformer_block` consumes `extra_per_block_pos_emb` —
    /// the additive pos emb should change the output.
    #[test]
    fn transformer_block_uses_extra_per_block_pos_emb() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let head_dim = cfg.head_dim();
        let ctx = cfg.crossattn_emb_channels;
        let s_txt = 4_usize;
        let model = make_stub_model(dev.clone(), cfg);

        let x_data: Vec<f32> = vec![0.1; b * t * h * w * d];
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");
        let ctx_data: Vec<f32> = vec![0.05; b * s_txt * ctx];
        let text_ctx = Tensor::from_vec(ctx_data, Shape::from_dims(&[b, s_txt, ctx]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");
        let (cos, sin) = build_cosmos_rope_freqs(
            head_dim, t, h, w,
            Some(24.0), 24.0, 1.0, 1.0, 1.0, true, &dev,
        ).expect("rope");
        let emb = Tensor::from_vec(
            vec![0.02_f32; b * t * d],
            Shape::from_dims(&[b, t, d]), dev.clone(),
        ).expect("from_vec").to_dtype(DType::BF16).expect("bf16");
        let lora = Tensor::from_vec(
            vec![0.0_f32; b * t * 3 * d],
            Shape::from_dims(&[b, t, 3 * d]), dev.clone(),
        ).expect("from_vec").to_dtype(DType::BF16).expect("bf16");

        let out_no_pe = model.transformer_block(
            &x, &emb, &lora, &text_ctx, None, None, &cos, &sin, 0,
        ).expect("block no pe");

        // Build a non-zero pos emb tensor.
        let pe_data: Vec<f32> = (0..b * t * h * w * d).map(|i| 0.05 + (i as f32) * 0.001).collect();
        let pe = Tensor::from_vec(pe_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .expect("from_vec").to_dtype(DType::BF16).expect("bf16");
        let out_with_pe = model.transformer_block(
            &x, &emb, &lora, &text_ctx, None, Some(&pe), &cos, &sin, 0,
        ).expect("block with pe");

        let a = out_no_pe.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_with_pe.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let dd = (x - y).abs();
            if dd > max_diff { max_diff = dd; }
        }
        assert!(max_diff > 1e-4,
            "extra_per_block_pos_emb not consumed: max_diff={max_diff}");
    }

    // -----------------------------------------------------------------------
    // Chunk 3 tests — patchify, unpatchify, final_layer, full forward.
    // GPU-gated. Exercise the real production functions, not CPU duplicates.
    // -----------------------------------------------------------------------

    /// Populate the global (non-block) weights for chunk 3's full forward
    /// path: x_embedder, t_embedder MLP, t_embedding_norm, learnable pos emb,
    /// FinalLayer.
    fn populate_global_weights(
        weights: &mut HashMap<String, Tensor>,
        dev: &Arc<cudarc::driver::CudaDevice>,
        cfg: &CosmosPredict25Config,
        t_max: usize,
        h_max: usize,
        w_max: usize,
    ) {
        let d = cfg.model_channels;
        let alora = cfg.adaln_lora_dim;
        let p_s = cfg.patch_spatial;
        let p_t = cfg.patch_temporal;
        let patch_dim_in = cfg.patch_embed_in_channels() * p_s * p_s * p_t;
        let patch_dim_out = p_s * p_s * p_t * cfg.out_channels;

        // x_embedder Linear: [patch_dim_in -> D]
        weights.insert(
            "x_embedder.proj.1.weight".to_string(),
            mk_weight(dev, "xemb", &[d, patch_dim_in]),
        );
        // t_embedder MLP (linear_1, linear_2 — V2_2B has no biases because use_adaln_lora=true)
        weights.insert(
            "t_embedder.1.linear_1.weight".to_string(),
            mk_weight(dev, "t_l1", &[d, d]),
        );
        weights.insert(
            "t_embedder.1.linear_2.weight".to_string(),
            mk_weight(dev, "t_l2", &[3 * d, d]),
        );
        // t_embedding_norm (RMSNorm gain)
        weights.insert(
            "t_embedding_norm.weight".to_string(),
            mk_ones_weight(dev, &[d]),
        );
        // LearnablePosEmbAxis (extra_per_block_abs_pos_emb=true → extra_pos_embedder.*)
        weights.insert(
            "extra_pos_embedder.pos_emb_t".to_string(),
            mk_weight(dev, "pe_t", &[t_max, d]),
        );
        weights.insert(
            "extra_pos_embedder.pos_emb_h".to_string(),
            mk_weight(dev, "pe_h", &[h_max, d]),
        );
        weights.insert(
            "extra_pos_embedder.pos_emb_w".to_string(),
            mk_weight(dev, "pe_w", &[w_max, d]),
        );
        // FinalLayer
        weights.insert(
            "final_layer.adaln_modulation.1.weight".to_string(),
            mk_weight(dev, "fl_a1", &[alora, d]),
        );
        weights.insert(
            "final_layer.adaln_modulation.2.weight".to_string(),
            mk_weight(dev, "fl_a2", &[2 * d, alora]),
        );
        weights.insert(
            "final_layer.linear.weight".to_string(),
            mk_weight(dev, "fl_lin", &[patch_dim_out, d]),
        );
    }

    /// Build a tiny full-stack model with `num_blocks` populated.
    fn make_full_model(
        dev: Arc<cudarc::driver::CudaDevice>,
        cfg: CosmosPredict25Config,
        t_max: usize,
        h_max: usize,
        w_max: usize,
    ) -> CosmosPredict25Dit {
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        for i in 0..cfg.num_blocks {
            populate_block_weights(&mut weights, &dev, &cfg, i);
        }
        populate_global_weights(&mut weights, &dev, &cfg, t_max, h_max, w_max);
        CosmosPredict25Dit { config: cfg, device: dev, weights }
    }

    /// patchify must be the inverse of unpatchify when patch dim is identity.
    /// We don't have a "real" identity weight to test e2e parity, but we can
    /// at least verify shapes and that running patchify on a synthetic [B,C,T,H,W]
    /// of the right divisible shape produces [B, T/pt, H/ps, W/ps, C*ps*ps*pt].
    #[test]
    fn patchify_shape_contract() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // tiny_test_config has in_channels=16, patch_spatial=2, patch_temporal=1
        let cfg = tiny_test_config();
        let model = make_stub_model(dev.clone(), cfg.clone());

        // Build a [B, 17, T, H, W] tensor (caller has already concat'd mask)
        let (b, t, h, w) = (1_usize, 2_usize, 4_usize, 4_usize);
        let c = cfg.patch_embed_in_channels(); // 17
        let x_data: Vec<f32> = (0..b * c * t * h * w).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, c, t, h, w]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let patches = model.patchify(&x).expect("patchify");
        let p_s = cfg.patch_spatial;
        let p_t = cfg.patch_temporal;
        assert_eq!(patches.shape().dims(), &[b, t/p_t, h/p_s, w/p_s, c*p_s*p_s*p_t]);
    }

    /// patchify(x) round-tripped through unpatchify(reshape-to-output-shape)
    /// must recover x bit-identically — verifies the dim-order is the
    /// expected inverse pair. The Python source uses asymmetric einops
    /// rearranges (patchify: "c r m n" trailing; unpatchify: "p1 p2 t c"
    /// trailing) — so a direct patchify→unpatchify is NOT a perfect inverse
    /// for arbitrary patch dims. However, when `out_channels == in_channels`
    /// and `patch_temporal == patch_spatial == 1`, both rearranges become
    /// trivial and the round-trip is identity.
    ///
    /// For our tiny test with `patch_spatial=2, patch_temporal=1`, we
    /// build a config-specific "identity remap" by NOT using the Linear in
    /// between and feeding a tensor with `c=out_c` (=16) into patchify-as-if-no-mask.
    /// We just verify shape goes round-trip; bit-identical recovery is
    /// blocked by the Python asymmetric einops layout.
    #[test]
    fn patchify_unpatchify_shape_roundtrip() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = CosmosPredict25Config {
            concat_padding_mask: false, // skip mask so patchify input C == out C
            ..tiny_test_config()
        };
        let model = make_stub_model(dev.clone(), cfg.clone());

        let (b, t, h, w) = (1_usize, 2_usize, 4_usize, 4_usize);
        let c = cfg.in_channels;
        let x_data: Vec<f32> = (0..b * c * t * h * w).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, c, t, h, w]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let p_s = cfg.patch_spatial;
        let p_t = cfg.patch_temporal;
        let patches = model.patchify(&x).expect("patchify");
        // patches: [B, T_p, H_p, W_p, in_c*ps²*pt]
        // For unpatchify we need trailing dim = out_c*ps²*pt; since out_c == in_c
        // here that's already what we have. Feed straight in.
        let recovered = model.unpatchify(&patches).expect("unpatchify");
        assert_eq!(recovered.shape().dims(), &[b, c, t/p_t, h/p_s*p_s, w/p_s*p_s]);
        // (= [b, c, t, h, w] since divisible)
    }

    /// final_layer modulation must be load-bearing — emb=0 vs emb=nonzero
    /// should change the output (because the LayerNorm-scaled shift/scale
    /// path is nontrivial).
    #[test]
    fn final_layer_modulation_is_load_bearing() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        let model = make_stub_model(dev.clone(), cfg.clone());
        // populate FinalLayer weights and adaln_lora dummies that don't matter
        // because the test uses zero lora.
        let mut full_weights = model.weights;
        populate_global_weights(&mut full_weights, &dev, &cfg, 8, 8, 8);
        let model = CosmosPredict25Dit { config: cfg.clone(), device: dev.clone(), weights: full_weights };

        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let lora_zero = Tensor::from_vec(
            vec![0.0_f32; b * t * 3 * d],
            Shape::from_dims(&[b, t, 3 * d]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let emb_zero = Tensor::from_vec(
            vec![0.0_f32; b * t * d],
            Shape::from_dims(&[b, t, d]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let out_zero = model.final_layer(&x, &emb_zero, &lora_zero).expect("fl zero");

        let emb_data: Vec<f32> = (0..b * t * d).map(|i| 0.1 + (i as f32) * 0.005).collect();
        let emb = Tensor::from_vec(emb_data, Shape::from_dims(&[b, t, d]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let out_nz = model.final_layer(&x, &emb, &lora_zero).expect("fl nonzero");

        let a = out_zero.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_nz.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let dd = (x - y).abs();
            if dd > max_diff { max_diff = dd; }
        }
        assert!(max_diff > 1e-4,
            "final_layer: t_cond appears not to affect output (max_diff={max_diff}); \
             adaLN modulation is dead.");
    }

    /// F9 companion (skeptic chunk 3): pin `emb` to a fixed nonzero value
    /// and vary `adaln_lora` between zero and nonzero. The output MUST
    /// differ — proves the `+ adaln_lora` add at `final_layer` is wired
    /// (independent of the existing test which kept lora=0). If someone
    /// removes the lora add, the original test still passes; this one
    /// catches it.
    #[test]
    fn final_layer_adaln_lora_is_load_bearing() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        let model = make_stub_model(dev.clone(), cfg.clone());
        let mut full_weights = model.weights;
        populate_global_weights(&mut full_weights, &dev, &cfg, 8, 8, 8);
        let model = CosmosPredict25Dit { config: cfg.clone(), device: dev.clone(), weights: full_weights };

        let (b, t, h, w) = (1_usize, 2_usize, 2_usize, 2_usize);
        let d = cfg.model_channels;
        let x_data: Vec<f32> = (0..b * t * h * w * d).map(|i| (i as f32) * 0.001).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, t, h, w, d]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        // emb pinned to a fixed nonzero value (so both runs see the same
        // adaln_modulation(emb) contribution and the difference is purely
        // from the lora path).
        let emb_data: Vec<f32> = (0..b * t * d).map(|i| 0.07 + (i as f32) * 0.003).collect();
        let emb = Tensor::from_vec(emb_data, Shape::from_dims(&[b, t, d]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        // Lora: zero vs nonzero. Note: the final layer only consumes the
        // first `2 * d` columns of the `3 * d`-wide adaln_lora tensor (it's
        // shared with the per-block adaLN which is 3-chunk; final_layer is
        // 2-chunk, see `:1457-1462`). So we MUST place non-zero values in
        // the first 2*d slice to exercise the path.
        let lora_zero = Tensor::from_vec(
            vec![0.0_f32; b * t * 3 * d],
            Shape::from_dims(&[b, t, 3 * d]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();
        let lora_nz_data: Vec<f32> = (0..b * t * 3 * d)
            // First 2*d slice gets nonzero values; rest gets zeros (so
            // anything in the unused tail can't accidentally leak).
            .map(|i| {
                let col = i % (3 * d);
                if col < 2 * d { 0.05 + (i as f32) * 0.002 } else { 0.0 }
            })
            .collect();
        let lora_nz = Tensor::from_vec(
            lora_nz_data,
            Shape::from_dims(&[b, t, 3 * d]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let out_zero = model.final_layer(&x, &emb, &lora_zero).expect("fl lora=zero");
        let out_nz = model.final_layer(&x, &emb, &lora_nz).expect("fl lora=nonzero");

        let a = out_zero.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_nz.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let dd = (x - y).abs();
            if dd > max_diff { max_diff = dd; }
        }
        assert!(max_diff > 1e-4,
            "final_layer: adaln_lora appears not to affect output (max_diff={max_diff}); \
             the `+ adaln_lora` add at the final layer is dead.");
    }

    /// Full forward end-to-end with synthetic weights. Verify output shape,
    /// finiteness, sanity of magnitude (BF16 catches infs as huge values).
    #[test]
    fn forward_runs_end_to_end_with_synthetic_weights() {
        // Try to enable env_logger so the magnitude probe is visible under
        // `RUST_LOG=debug`. `try_init` returns Err if already initialized;
        // we don't care either way.
        let _ = env_logger::builder().is_test(true).try_init();

        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Tiny config but enough to exercise patchify→blocks→final.
        let cfg = CosmosPredict25Config {
            num_blocks: 2,
            ..tiny_test_config()
        };
        let model = make_full_model(dev.clone(), cfg.clone(), 8, 8, 8);

        let (b, t, h, w) = (1_usize, 2_usize, 4_usize, 4_usize);
        let in_c = cfg.in_channels; // 16 (NOT including padding mask channel)
        // Input must be [B, C=in_channels, T, H, W] — model adds the padding-mask channel.
        let x_data: Vec<f32> = (0..b * in_c * t * h * w).map(|i| 0.05 + (i as f32) * 0.0003).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, in_c, t, h, w]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        // Timesteps [B, T] (float scalars in [0,1]-ish range)
        let ts_data: Vec<f32> = (0..b * t).map(|i| 0.4 + (i as f32) * 0.05).collect();
        let timesteps = Tensor::from_vec(ts_data, Shape::from_dims(&[b, t]), dev.clone())
            .unwrap();
        // BF16 sinusoidal handled internally; pass F32, prepare_timestep casts.

        // crossattn_emb: [B, S_txt, crossattn_emb_channels]
        let s_txt = 5_usize;
        let ctx_data: Vec<f32> = (0..b * s_txt * cfg.crossattn_emb_channels)
            .map(|i| 0.01 + (i as f32) * 0.001).collect();
        let crossattn_emb = Tensor::from_vec(
            ctx_data, Shape::from_dims(&[b, s_txt, cfg.crossattn_emb_channels]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let out = model.forward(
            &x, &timesteps, &crossattn_emb,
            None,         // condition_video_input_mask (LVG wrapper disabled in tiny config)
            None,         // padding_mask=None → uses zero default
            Some(24.0),   // fps for the 3D RoPE temporal modulation
            None,         // image_context
        ).expect("forward");
        // Output shape: [B, out_c, T, H, W]
        assert_eq!(out.shape().dims(), &[b, cfg.out_channels, t, h, w]);

        // Finite + bounded sanity (catches inf/nan from BF16 saturation).
        let data = out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_abs: f32 = 0.0;
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "forward out[{i}] = {v} non-finite");
            if v.abs() > max_abs { max_abs = v.abs(); }
        }
        // Tiny stub weights → magnitudes should remain small. The chunk-2
        // skeptic flagged anima reaching 200+ at full scale; at our tiny
        // scale (1e-3 inputs, 0.05 weights) we'd be astonished to see >10.
        // Loose bound 10000 catches obvious inf-equivalents.
        assert!(max_abs < 10000.0, "forward output unreasonably large: {max_abs}");

        // Probe the per-block magnitudes via the public API by re-running
        // with debug logging enabled. (Note: log already initialized; if
        // not, the probe simply skips. We don't assert log content here —
        // the probe is informational.)
    }

    /// `concat_padding_mask=true` (default) cats a 17th channel; `false`
    /// skips. Verify the input channel-count contract.
    #[test]
    fn padding_mask_concat_changes_patchify_input_channels() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // With concat_padding_mask=true (default for V2_2B): patch_embed_in_channels = 17
        let cfg_with_mask = tiny_test_config();
        assert!(cfg_with_mask.concat_padding_mask);
        assert_eq!(cfg_with_mask.patch_embed_in_channels(), 17);

        // With concat_padding_mask=false: patch_embed_in_channels = 16
        let cfg_no_mask = CosmosPredict25Config {
            concat_padding_mask: false,
            ..tiny_test_config()
        };
        assert!(!cfg_no_mask.concat_padding_mask);
        assert_eq!(cfg_no_mask.patch_embed_in_channels(), 16);

        // Build a minimal model with no_mask config and verify patchify takes
        // the bare in_channels=16 directly without expecting 17.
        let model = make_full_model(dev.clone(), cfg_no_mask.clone(), 8, 8, 8);
        let (b, t, h, w) = (1_usize, 2_usize, 4_usize, 4_usize);
        let in_c = cfg_no_mask.in_channels;
        let x_data: Vec<f32> = (0..b * in_c * t * h * w).map(|i| 0.05 + (i as f32) * 0.0003).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, in_c, t, h, w]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let ts = Tensor::from_vec(
            vec![0.5_f32; b * t], Shape::from_dims(&[b, t]), dev.clone()
        ).unwrap();
        let ctx = Tensor::from_vec(
            vec![0.01_f32; b * 4 * cfg_no_mask.crossattn_emb_channels],
            Shape::from_dims(&[b, 4, cfg_no_mask.crossattn_emb_channels]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        // Note: the x_embedder weight in `populate_global_weights` was sized
        // by `patch_embed_in_channels()` — so for `concat_padding_mask=false`
        // it matches `in_channels` directly. forward should run.
        let out = model.forward(&x, &ts, &ctx, None, None, Some(24.0), None).expect("forward no mask");
        assert_eq!(out.shape().dims(), &[b, cfg_no_mask.out_channels, t, h, w]);
    }

    /// F4 companion (skeptic chunk 3): exercise the non-zero padding mask
    /// path explicitly. Build two `[B, 1, H, W]` masks — one zeros, one
    /// with a non-zero value at a specific (h, w) — and confirm that the
    /// forward outputs DIFFER. This tests the stride-0 broadcast across T →
    /// cat → patchify chain that the existing test never exercised.
    #[test]
    fn padding_mask_nonzero_propagates_through_forward() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        let cfg = tiny_test_config();
        assert!(cfg.concat_padding_mask);
        let model = make_full_model(dev.clone(), cfg.clone(), 8, 8, 8);

        let (b, t, h, w) = (1_usize, 2_usize, 4_usize, 4_usize);
        let in_c = cfg.in_channels;
        // Identical content latents for both runs.
        let x_data: Vec<f32> = (0..b * in_c * t * h * w).map(|i| 0.05 + (i as f32) * 0.0003).collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, in_c, t, h, w]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let ts = Tensor::from_vec(
            vec![0.5_f32; b * t], Shape::from_dims(&[b, t]), dev.clone()
        ).unwrap();
        let ctx_data: Vec<f32> = (0..b * 5 * cfg.crossattn_emb_channels)
            .map(|i| 0.01 + (i as f32) * 0.001).collect();
        let ctx = Tensor::from_vec(
            ctx_data,
            Shape::from_dims(&[b, 5, cfg.crossattn_emb_channels]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        // Mask A: all zeros (equivalent to no padding).
        let mask_zero_data: Vec<f32> = vec![0.0_f32; b * 1 * h * w];
        let mask_zero = Tensor::from_vec(
            mask_zero_data,
            Shape::from_dims(&[b, 1, h, w]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        // Mask B: non-zero values at a specific (h, w). We want to verify
        // that a non-zero mask CHANGES the output (proves the data flows
        // through). The mask should also span multiple non-zero positions
        // so the broadcast-across-T path is exercised non-trivially.
        let mut mask_nz_data: Vec<f32> = vec![0.0_f32; b * 1 * h * w];
        for hh in 0..h {
            for ww in 0..w {
                if (hh + ww) % 2 == 0 {
                    mask_nz_data[hh * w + ww] = 0.7;
                }
            }
        }
        let mask_nz = Tensor::from_vec(
            mask_nz_data,
            Shape::from_dims(&[b, 1, h, w]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let out_zero = model.forward(&x, &ts, &ctx, None, Some(&mask_zero), Some(24.0), None)
            .expect("forward mask=zero");
        let out_nz = model.forward(&x, &ts, &ctx, None, Some(&mask_nz), Some(24.0), None)
            .expect("forward mask=nonzero");

        // Outputs must differ — the non-zero mask should propagate to all
        // T positions and affect every block. If the stride-0 broadcast or
        // the cat silently zeroed the mask, the outputs would be identical.
        let a = out_zero.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let bv = out_nz.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(a.len(), bv.len(), "shape mismatch");
        let mut max_diff: f32 = 0.0;
        for (x, y) in a.iter().zip(bv.iter()) {
            let dd = (x - y).abs();
            if dd > max_diff { max_diff = dd; }
        }
        assert!(
            max_diff > 1e-4,
            "Non-zero padding mask did not affect forward output (max_diff={max_diff}). \
             Stride-0 broadcast or cat may have silently zeroed the mask channel."
        );
    }

    /// `apply_crossattn_proj` projects `[..., crossattn_proj_in_channels]` to
    /// `[..., crossattn_emb_channels]` via Linear(bias=True) + exact-erf GELU.
    /// Builds a tiny config with synthesised weights, then verifies:
    ///   1. Output shape is `[..., crossattn_emb_channels]`.
    ///   2. Output equals `gelu_exact(linear(x))` computed via an independent
    ///      reference path (matmul-by-hand + bias add + gelu_exact).
    #[test]
    fn crossattn_proj_changes_text_embedding_dim() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };

        // Tiny dims so the reference computation is cheap.
        let in_d = 32_usize;
        let out_d = 8_usize;
        let b = 1_usize;
        let n = 4_usize;

        let cfg = CosmosPredict25Config {
            use_crossattn_projection: true,
            crossattn_proj_in_channels: in_d,
            crossattn_emb_channels: out_d,
            ..tiny_test_config()
        };

        // Identity-ish weight: small deterministic values via mk_weight.
        // Bias: zeros so output exactly equals gelu_exact(x @ W^T).
        let weight = mk_weight(&dev, "cap_w", &[out_d, in_d]);
        let bias_data: Vec<f32> = vec![0.0_f32; out_d];
        let bias = Tensor::from_vec(bias_data, Shape::from_dims(&[out_d]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("crossattn_proj.0.weight".to_string(), weight.clone());
        weights.insert("crossattn_proj.0.bias".to_string(), bias.clone());
        let model = CosmosPredict25Dit { config: cfg.clone(), device: dev.clone(), weights };

        // Input: [B, N, in_d], BF16, small magnitudes.
        let x_data: Vec<f32> = (0..b * n * in_d)
            .map(|i| 0.02 + (i as f32) * 0.003)
            .collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, n, in_d]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();

        let y = model.apply_crossattn_proj(&x).expect("apply_crossattn_proj");

        // 1. Shape check.
        assert_eq!(y.shape().dims(), &[b, n, out_d],
            "crossattn_proj output shape mismatch");

        // 2. Independent reference: compute Linear(no-bias-since-zero) + GELU_exact
        //    via fused_linear3d_native (no bias) then gelu_exact, compare to
        //    the function's output (which uses fused_linear3d_native WITH bias).
        //    With bias=zero, the two should match within BF16 rounding.
        let ref_linear = flame_core::ops::fused_inference::fused_linear3d_native(
            &x, &weight, None,
        ).expect("ref linear");
        let ref_out = ref_linear.gelu_exact().expect("ref gelu_exact");

        let got = y.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let want = ref_out.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(got.len(), want.len());
        let mut max_diff: f32 = 0.0;
        for (a, b) in got.iter().zip(want.iter()) {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
        }
        // BF16 epsilon ~ 1e-2; identical compute paths should be near-zero.
        assert!(max_diff < 1e-3,
            "crossattn_proj output diverges from reference (max_diff={max_diff})");

        // 3. Sanity: the activation IS applied — compare to pre-GELU linear.
        //    For nonzero inputs in [-x, x], gelu_exact != linear pointwise.
        let pre_gelu = ref_linear.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut max_diff_vs_pre_gelu: f32 = 0.0;
        for (a, b) in got.iter().zip(pre_gelu.iter()) {
            let d = (a - b).abs();
            if d > max_diff_vs_pre_gelu { max_diff_vs_pre_gelu = d; }
        }
        assert!(max_diff_vs_pre_gelu > 1e-4,
            "crossattn_proj output equals pre-GELU linear (max_diff={max_diff_vs_pre_gelu}); \
             GELU activation may not be applied.");
    }

    /// When `use_crossattn_projection=false`, the forward path must NOT touch
    /// the `crossattn_emb` tensor. Verify by running the full model twice with
    /// `use_crossattn_projection=false` and confirming the output matches a
    /// run where the cross-attn input is identical — i.e. the no-op branch
    /// passes the tensor through without applying any projection.
    ///
    /// We construct a model WITHOUT registering `crossattn_proj.0.{weight,bias}`
    /// and confirm forward succeeds (a missing-key error would prove the path
    /// is taken when it shouldn't be).
    #[test]
    fn crossattn_proj_skipped_when_disabled() {
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => return,
        };
        // tiny_test_config defaults `use_crossattn_projection=false`.
        let cfg = tiny_test_config();
        assert!(!cfg.use_crossattn_projection);

        // Build a full model — but deliberately omit crossattn_proj.0.weight
        // and crossattn_proj.0.bias. `make_full_model` already does this; we
        // just need to verify the populate_global_weights map doesn't insert
        // them (it doesn't — see `populate_global_weights`).
        let model = make_full_model(dev.clone(), cfg.clone(), 8, 8, 8);
        assert!(model.weights.get("crossattn_proj.0.weight").is_none(),
            "test setup: crossattn_proj weights should not be present");
        assert!(model.weights.get("crossattn_proj.0.bias").is_none(),
            "test setup: crossattn_proj bias should not be present");

        // Run forward — it should succeed because the projection branch is
        // skipped. A failure here (missing-weight error) would mean the
        // `if self.config.use_crossattn_projection` guard is broken.
        let (b, t, h, w) = (1_usize, 2_usize, 4_usize, 4_usize);
        let in_c = cfg.in_channels;
        let x_data: Vec<f32> = (0..b * in_c * t * h * w)
            .map(|i| 0.05 + (i as f32) * 0.0003)
            .collect();
        let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, in_c, t, h, w]), dev.clone())
            .unwrap().to_dtype(DType::BF16).unwrap();
        let ts = Tensor::from_vec(
            vec![0.5_f32; b * t], Shape::from_dims(&[b, t]), dev.clone()
        ).unwrap();
        let ctx_data: Vec<f32> = (0..b * 5 * cfg.crossattn_emb_channels)
            .map(|i| 0.01 + (i as f32) * 0.001).collect();
        let ctx = Tensor::from_vec(
            ctx_data,
            Shape::from_dims(&[b, 5, cfg.crossattn_emb_channels]), dev.clone(),
        ).unwrap().to_dtype(DType::BF16).unwrap();

        let out = model.forward(&x, &ts, &ctx, None, None, Some(24.0), None)
            .expect("forward with use_crossattn_projection=false");
        assert_eq!(out.shape().dims(), &[b, cfg.out_channels, t, h, w]);
    }

    // -----------------------------------------------------------------------
    // Production-checkpoint load test. Gated on `COSMOS_DIT_PATH` env var so
    // CI boxes without weights still pass. When the env var is set:
    //   - Open the safetensors file (4.1 GB, 689 tensors).
    //   - Use `cosmos_v2_2b_production()` config.
    //   - Verify clean load with no panics.
    //   - Verify a few canonical keys exist with the expected shapes
    //     (the ones our DiT actually reads at forward time).
    //   - Report counts: loaded vs skipped.
    // -----------------------------------------------------------------------
    #[test]
    fn production_checkpoint_loads_cleanly() {
        // Skip silently when env var unset (default CI path).
        let path_str = match std::env::var("COSMOS_DIT_PATH") {
            Ok(v) => v,
            Err(_) => return,
        };
        let path = std::path::Path::new(&path_str);
        if !path.exists() {
            eprintln!("COSMOS_DIT_PATH={} does not exist — skipping", path_str);
            return;
        }
        let dev = match maybe_cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device — skipping production checkpoint load test");
                return;
            }
        };

        let _ = env_logger::builder().is_test(true).try_init();

        let cfg = CosmosPredict25Config::cosmos_v2_2b_production();
        // Sanity on the config itself.
        assert!(cfg.lvg_wrapper, "production preset must set lvg_wrapper=true");
        assert!(!cfg.extra_per_block_abs_pos_emb,
            "production preset must set extra_per_block_abs_pos_emb=false");
        assert_eq!(cfg.patch_embed_in_channels(), 18,
            "production preset must yield patch_embed_in_channels=18");
        assert!(cfg.use_crossattn_projection);
        assert_eq!(cfg.crossattn_proj_in_channels, 100352);
        assert_eq!(cfg.crossattn_emb_channels, 1024);

        let model = CosmosPredict25Dit::from_safetensors(path, cfg, dev.clone())
            .expect("from_safetensors clean load");

        // Canonical keys with expected shapes.
        // The check is shape-only — dtype is normalized to BF16 by the loader.
        let canonical: &[(&str, &[usize])] = &[
            // x_embedder: in 18 ch * 2² * 1 = 72 → hidden=2048
            ("x_embedder.proj.1.weight", &[2048, 72]),
            // t_embedder.1: linear_1 [2048, 2048], linear_2 [6144, 2048]
            ("t_embedder.1.linear_1.weight", &[2048, 2048]),
            ("t_embedder.1.linear_2.weight", &[6144, 2048]),
            // t_embedding_norm gain
            ("t_embedding_norm.weight", &[2048]),
            // crossattn_proj: [1024, 100352] + bias [1024]
            ("crossattn_proj.0.weight", &[1024, 100352]),
            ("crossattn_proj.0.bias", &[1024]),
            // Block 0 sample
            ("blocks.0.self_attn.q_proj.weight", &[2048, 2048]),
            ("blocks.0.self_attn.k_proj.weight", &[2048, 2048]),
            ("blocks.0.self_attn.v_proj.weight", &[2048, 2048]),
            ("blocks.0.self_attn.output_proj.weight", &[2048, 2048]),
            ("blocks.0.self_attn.q_norm.weight", &[128]),
            ("blocks.0.self_attn.k_norm.weight", &[128]),
            // Block 0 adaLN: snake_case, 3 sub-blocks
            ("blocks.0.adaln_modulation_self_attn.1.weight", &[256, 2048]),
            ("blocks.0.adaln_modulation_self_attn.2.weight", &[6144, 256]),
            ("blocks.0.adaln_modulation_cross_attn.1.weight", &[256, 2048]),
            ("blocks.0.adaln_modulation_cross_attn.2.weight", &[6144, 256]),
            ("blocks.0.adaln_modulation_mlp.1.weight", &[256, 2048]),
            ("blocks.0.adaln_modulation_mlp.2.weight", &[6144, 256]),
            // Block 0 cross-attn — text K/V is from 1024-dim (crossattn_emb_channels=1024)
            ("blocks.0.cross_attn.q_proj.weight", &[2048, 2048]),
            ("blocks.0.cross_attn.k_proj.weight", &[2048, 1024]),
            ("blocks.0.cross_attn.v_proj.weight", &[2048, 1024]),
            // Block 0 MLP
            ("blocks.0.mlp.layer1.weight", &[8192, 2048]),
            ("blocks.0.mlp.layer2.weight", &[2048, 8192]),
            // Last block exists too (28 blocks total, index 27)
            ("blocks.27.self_attn.q_proj.weight", &[2048, 2048]),
            // FinalLayer: adaln 2-chunk, linear → 16*2²*1 = 64
            ("final_layer.adaln_modulation.1.weight", &[256, 2048]),
            ("final_layer.adaln_modulation.2.weight", &[4096, 256]),
            ("final_layer.linear.weight", &[64, 2048]),
        ];

        let mut missing: Vec<(&str, Vec<usize>)> = Vec::new();
        for (key, want) in canonical {
            match model.weights.get(*key) {
                Some(t) => {
                    let got = t.shape().dims();
                    if got != *want {
                        missing.push((*key, got.to_vec()));
                    }
                }
                None => {
                    missing.push((*key, vec![]));
                }
            }
        }
        if !missing.is_empty() {
            for (k, got) in &missing {
                eprintln!("  MISSING/MISMATCH: {} (got shape: {:?})", k, got);
            }
            panic!("{} canonical keys missing or shape-mismatched", missing.len());
        }

        // Skipped-key sanity: production checkpoint has 689 keys total, of which
        // 120 are non-loadable (4 training scalars + 3 rope3d buffers + 113
        // _extra_state). The loader should hold the remaining 569 in `weights`.
        let n_loaded = model.weights.len();
        eprintln!(
            "[production_checkpoint_loads_cleanly] loaded={} tensors (expected 569; \
             total in ckpt=689; skipped=120 = 4 accum + 3 rope3d_buf + 113 _extra_state)",
            n_loaded
        );
        // Be loose on the exact load count in case the checkpoint format
        // shifts slightly (e.g. one more or less `_extra_state` key) — but
        // the canonical-keys check above is the load-bearing assertion.
        assert!(
            n_loaded >= 560 && n_loaded <= 580,
            "loaded tensor count {n_loaded} far outside expected range [560, 580]"
        );

        // Verify NO non-loadable keys leaked into the weights map.
        for key in model.weights.keys() {
            assert!(
                CosmosPredict25Dit::is_loadable_key(key),
                "non-loadable key `{}` leaked into weights map", key
            );
        }
        // Specifically check no _extra_state ended up in the live map.
        let leaked_extras: Vec<_> = model.weights.keys()
            .filter(|k| k.ends_with("._extra_state"))
            .collect();
        assert!(
            leaked_extras.is_empty(),
            "extra_state keys leaked: {:?}", leaked_extras
        );
    }
}
