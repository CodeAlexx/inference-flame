//! Qwen2.5-VL Vision Tower (ViT) for Lance image_edit / video_edit / I2V.
//!
//! Pure-Rust port of `/home/alex/Lance/modeling/vit/qwen2_5_vl_vit.py`,
//! restricted to the SDPA attention variant (matches our flame-core SDPA
//! path; FlashAttention2 + eager variants from Python are ignored).
//!
//! ## Architecture (Lance 3B Video `vit_config` from llm_config.json)
//!
//! ```text
//! depth                  = 32      blocks
//! hidden_size            = 1280    per-patch embed (input to blocks)
//! intermediate_size      = 3420    MLP hidden
//! num_heads              = 16      => head_dim = 80
//! out_hidden_size        = 2048    post-merger (LM hidden)
//! patch_size             = 14
//! spatial_merge_size     = 2       (4 tokens → 1 after merger)
//! temporal_patch_size    = 2
//! window_size            = 112     (8 merge-units per side)
//! fullatt_block_indexes  = [7, 15, 23, 31]
//! hidden_act             = silu
//! rms_norm_eps           = 1e-6
//! rotary theta           = 10000.0
//! ```
//!
//! ## Weight key map (`vit_model.*` prefix in the Lance checkpoint)
//!
//! - `vit_model.patch_embed.proj.weight`               → [1280, 3, 2, 14, 14] Conv3d
//!   (kernel == stride == [2,14,14]; equivalent to a linear projection over
//!   flat patch vectors of length `3*2*14*14 = 1176`).
//! - `vit_model.blocks.{i}.norm1.weight`               → [1280]  RMSNorm
//! - `vit_model.blocks.{i}.norm2.weight`               → [1280]  RMSNorm
//! - `vit_model.blocks.{i}.attn.qkv.weight`            → [3840, 1280]
//! - `vit_model.blocks.{i}.attn.qkv.bias`              → [3840]
//! - `vit_model.blocks.{i}.attn.proj.weight`           → [1280, 1280]
//! - `vit_model.blocks.{i}.attn.proj.bias`             → [1280]
//! - `vit_model.blocks.{i}.mlp.gate_proj.weight`       → [3420, 1280]
//! - `vit_model.blocks.{i}.mlp.gate_proj.bias`         → [3420]
//! - `vit_model.blocks.{i}.mlp.up_proj.weight`         → [3420, 1280]
//! - `vit_model.blocks.{i}.mlp.up_proj.bias`           → [3420]
//! - `vit_model.blocks.{i}.mlp.down_proj.weight`       → [1280, 3420]
//! - `vit_model.blocks.{i}.mlp.down_proj.bias`         → [1280]
//! - `vit_model.merger.ln_q.weight`                    → [1280]   RMSNorm (pre-merge)
//! - `vit_model.merger.mlp.0.weight`                   → [5120, 5120]  Linear (4*1280→5120)
//! - `vit_model.merger.mlp.0.bias`                     → [5120]
//! - `vit_model.merger.mlp.2.weight`                   → [2048, 5120]  Linear (→out_hidden_size)
//! - `vit_model.merger.mlp.2.bias`                     → [2048]
//!
//! ## Conventions
//!
//! - All weights stored in BF16 (cast at load); F32 only for RMSNorm internals
//!   (delegated to `flame_core::cuda_ops_bf16::rms_norm_bf16`) and for the
//!   rotary cos/sin tables.
//! - 2D weight matrices pre-transposed at load time → matmul takes `[M, K] @ [K, N]`.
//! - `grid_thw` is passed in as a host-side `&[[u32; 3]]` — it drives small
//!   integer metadata (cu_seqlens, window_index, rotary positions), all
//!   computed CPU-side and pushed to GPU as small tensors.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Config for the Qwen2.5-VL Vision Tower.
///
/// Defaults match the Lance 3B Video checkpoint (verified against
/// `/home/alex/.serenity/models/lance/Lance_3B_Video/llm_config.json` and
/// against the `vit_model.*` weight shapes in `model.safetensors`).
#[derive(Debug, Clone)]
pub struct Qwen25VLVitConfig {
    pub patch_size: usize,           // 14
    pub temporal_patch_size: usize,  // 2
    pub in_channels: usize,          // 3
    pub embed_dim: usize,            // 1280 (== hidden_size for blocks)
    pub num_layers: usize,           // 32
    pub num_heads: usize,            // 16
    pub intermediate_size: usize,    // 3420 (MLP hidden)
    pub out_hidden_size: usize,      // 2048 (post-merger)
    pub spatial_merge_size: usize,   // 2
    pub window_size: usize,          // 112
    pub fullatt_block_indexes: Vec<usize>, // [7, 15, 23, 31]
    pub eps: f32,                    // 1e-6
    pub rope_theta: f32,             // 10000.0
}

impl Default for Qwen25VLVitConfig {
    fn default() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            embed_dim: 1280,
            num_layers: 32,
            num_heads: 16,
            intermediate_size: 3420,
            out_hidden_size: 2048,
            spatial_merge_size: 2,
            window_size: 112,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

impl Qwen25VLVitConfig {
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    #[inline]
    pub fn spatial_merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// Flat patch vector length: `in_channels * temporal_patch_size * patch_size^2`.
    /// For the default config this is `3 * 2 * 14 * 14 = 1176`.
    #[inline]
    pub fn flat_patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// The Vision Tower itself.
///
/// Stores all weights as a flat `HashMap<String, Tensor>` (no `.weight`
/// suffix stripping; keys match the safetensors file 1:1). All 2D weight
/// tensors are pre-transposed at load time so the forward path can call
/// `x.matmul(weight_t)` directly.
pub struct Qwen25VLVisionTower {
    cfg: Qwen25VLVitConfig,
    weights: HashMap<String, Tensor>,
    /// Flat-patch projection weight `[flat_patch_dim, embed_dim]` (already
    /// transposed). Derived from `vit_model.patch_embed.proj.weight`'s
    /// 5D Conv3d layout `[embed_dim, 3, T, H, W]` — since `kernel == stride`,
    /// the Conv3d is bit-equivalent to a linear projection.
    patch_proj_t: Tensor,
    device: Arc<CudaDevice>,
}

impl Qwen25VLVisionTower {
    /// Load all `vit_model.*` weights from a safetensors file and build the tower.
    ///
    /// The loader uses memory-mapped filtered loading, so a 28 GB checkpoint
    /// only pages in the ~390 ViT tensors.
    pub fn load(path: &Path, device: &Arc<CudaDevice>) -> Result<Self> {
        let raw = flame_core::serialization::load_file_filtered(path, device, |k| {
            k.starts_with("vit_model.")
        })?;
        Self::from_weights(raw, Qwen25VLVitConfig::default(), device.clone())
    }

    /// Construct from a pre-loaded weight map. Performs:
    /// 1. dtype-promotion to BF16 for everything except integer/embed tables
    /// 2. pre-transpose of all 2D `.weight` matrices
    /// 3. reshape + transpose of the Conv3d patch_embed weight into a 2D matmul
    pub fn from_weights(
        raw: HashMap<String, Tensor>,
        cfg: Qwen25VLVitConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // Step 1: cast everything to BF16 (RMSNorm internals will re-promote
        // to F32 inside the kernel as needed).
        let mut weights: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, v) in raw.into_iter() {
            let t = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)?
            };
            weights.insert(k, t);
        }

        // Step 2: pre-transpose all 2D weight matrices for fast matmul.
        // Skip 1D bias/norm tensors and the 5D Conv3d weight.
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if !key.ends_with(".weight") {
                continue;
            }
            let w = &weights[key];
            if w.shape().dims().len() != 2 {
                continue;
            }
            // Skip norm weights (1D, won't hit anyway). Linear weights only.
            let wt = flame_core::bf16_elementwise::transpose2d_bf16(w)?;
            weights.insert(key.clone(), wt);
        }

        // Step 3: Conv3d patch_embed → linear projection. Reshape weight
        // [embed_dim, 3, T, H, W] → [embed_dim, flat_patch_dim], then
        // transpose → [flat_patch_dim, embed_dim] for `flat_patches @ W`.
        let pe_key = "vit_model.patch_embed.proj.weight";
        let pe_w = weights
            .get(pe_key)
            .ok_or_else(|| Error::InvalidInput(format!("missing {pe_key}")))?;
        let flat_dim = cfg.flat_patch_dim();
        let pe_w_2d = pe_w.reshape(&[cfg.embed_dim, flat_dim])?;
        let patch_proj_t = flame_core::bf16_elementwise::transpose2d_bf16(&pe_w_2d)?;
        // Drop the 5D version — it served its purpose.
        weights.remove(pe_key);

        Ok(Self {
            cfg,
            weights,
            patch_proj_t,
            device,
        })
    }

    /// Returns a read-only view of the loaded weight map (post-transpose).
    /// Useful for tests / debugging.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    pub fn config(&self) -> &Qwen25VLVitConfig {
        &self.cfg
    }

    /// Lookup a weight, returning a friendly error if missing.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| Error::InvalidInput(format!("Missing vit weight: {key}")))
    }

    // ------------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------------

    /// `[B, N, K] @ [K, M] → [B, N, M]` (or 2D `[N,K] @ [K,M] → [N,M]`).
    fn linear(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() == 2 {
            return x.matmul(weight_t);
        }
        let last = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let out = x.reshape(&[batch, last])?.matmul(weight_t)?;
        let out_dim = out.shape().dims()[1];
        let mut new_shape = dims;
        *new_shape.last_mut().unwrap() = out_dim;
        out.reshape(&new_shape)
    }

    fn linear_bias(x: &Tensor, weight_t: &Tensor, bias: &Tensor) -> Result<Tensor> {
        Self::linear(x, weight_t)?.add(bias)
    }

    /// RMSNorm: reshape to 2D, normalize, reshape back.
    fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
        out_2d.reshape(&dims)
    }

    // ------------------------------------------------------------------
    // Patch embedding (Conv3d collapsed to a matmul)
    // ------------------------------------------------------------------

    /// `[N_patches, flat_patch_dim] → [N_patches, embed_dim]`.
    ///
    /// Equivalent to Python's
    /// `Conv3d(3, embed_dim, kernel=stride=[T, H, W])(x.view(-1, 3, T, H, W))`
    /// because `kernel == stride`: each patch is consumed exactly once with
    /// no overlap, so the whole op is a per-patch linear projection.
    fn patch_embed(&self, flat_patches: &Tensor) -> Result<Tensor> {
        flat_patches.matmul(&self.patch_proj_t)
    }

    // ------------------------------------------------------------------
    // 2D rotary cos/sin built from grid_thw
    // ------------------------------------------------------------------

    /// Compute the (cos, sin) rotary tables for the full sequence.
    ///
    /// Mirrors `Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb` then
    /// `emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)`. Returns
    /// (cos, sin), each shaped `[1, 1, seq_len, head_dim/2]` in BF16
    /// (half-split RoPE; flame's `rope_halfsplit_bf16` broadcasts over the
    /// other half internally).
    ///
    /// `pos_ids_reordered` is the [seq_len, 2] index tensor *after* window
    /// reordering — caller computes it once via [`Self::window_layout`].
    fn build_rope_cossin(
        &self,
        pos_ids: &[(u32, u32)], // (h_idx, w_idx) per token, AFTER window reorder
        max_grid_size: u32,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        // The vision rotary uses `head_dim / 2` total positions (half for h,
        // half for w). Python: `Qwen2_5_VisionRotaryEmbedding(head_dim // 2)`.
        // The `inv_freq` length is `head_dim / 4`.
        let half_dim = cfg.head_dim() / 2; // total embedding width (per token, per axis)
        let quarter = half_dim / 2; // inv_freq length

        // inv_freq = 1 / theta^(2k / half_dim) for k in 0..quarter
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|k| {
                let exp = (2 * k) as f32 / half_dim as f32;
                cfg.rope_theta.powf(-exp)
            })
            .collect();

        // freqs_full[s, k] = s * inv_freq[k] for s in 0..max_grid_size
        // shape: [max_grid_size, quarter]
        let mg = max_grid_size as usize;
        let mut freqs_full = vec![0f32; mg * quarter];
        for s in 0..mg {
            for k in 0..quarter {
                freqs_full[s * quarter + k] = (s as f32) * inv_freq[k];
            }
        }

        // For each token, concat freqs_full[h_idx] || freqs_full[w_idx] →
        // shape [seq_len, half_dim] = [seq_len, quarter*2].
        // Then `emb = cat(rotary_pos_emb, rotary_pos_emb, -1)` → [seq_len, head_dim].
        // For flame's rope_halfsplit_bf16 we just need the half-size cos/sin
        // (`[1, 1, seq_len, head_dim/2]`).
        let seq_len = pos_ids.len();
        let mut cos_buf = vec![0f32; seq_len * half_dim];
        let mut sin_buf = vec![0f32; seq_len * half_dim];
        for (i, &(h, w)) in pos_ids.iter().enumerate() {
            let h = h as usize;
            let w = w as usize;
            let row_off = i * half_dim;
            // first quarter*2 entries: h-axis freqs, then w-axis freqs
            for k in 0..quarter {
                let theta_h = freqs_full[h * quarter + k];
                cos_buf[row_off + k] = theta_h.cos();
                sin_buf[row_off + k] = theta_h.sin();
            }
            for k in 0..quarter {
                let theta_w = freqs_full[w * quarter + k];
                cos_buf[row_off + quarter + k] = theta_w.cos();
                sin_buf[row_off + quarter + k] = theta_w.sin();
            }
        }

        let cos = Tensor::from_vec(
            cos_buf,
            Shape::from_dims(&[1, 1, seq_len, half_dim]),
            self.device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(
            sin_buf,
            Shape::from_dims(&[1, 1, seq_len, half_dim]),
            self.device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    // ------------------------------------------------------------------
    // Window reordering — `rot_pos_emb` + `get_window_index`
    // ------------------------------------------------------------------

    /// Per-image / per-frame layout. Returns:
    /// - `pos_ids_orig`:    `[seq_len, 2]` = (h_idx, w_idx) in the ORIGINAL token order
    /// - `window_index`:    permutation of `0..seq_len_post_merge` (units of `spatial_merge_unit`)
    /// - `cu_window_seqlens`: cumulative seq lens for the windowed-attention blocks (token units)
    /// - `cu_seqlens_full`:   cumulative seq lens per image/frame for fullatt blocks (token units)
    /// - `max_grid_size`: max(h, w) across all grids
    /// - `seq_len`: total number of input tokens (pre-merge)
    fn window_layout(
        &self,
        grid_thw: &[[u32; 3]],
    ) -> (
        Vec<(u32, u32)>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        u32,
        usize,
    ) {
        let cfg = &self.cfg;
        let smerge = cfg.spatial_merge_size as u32;
        let smerge_unit = cfg.spatial_merge_unit();
        // vit_merger_window_size = window_size // spatial_merge_size // patch_size
        let vit_merger_window_size =
            (cfg.window_size / cfg.spatial_merge_size / cfg.patch_size) as u32;

        // Pass 1: build per-token pos_ids in ORIGINAL order
        // and per-image (t,h,w) for the window pass.
        let mut pos_ids_orig: Vec<(u32, u32)> = Vec::new();
        let mut max_grid_size: u32 = 0;
        for &[t, h, w] in grid_thw {
            // Python: hpos_ids = arange(h).unsqueeze(1).expand(-1, w) → [h, w]
            //        reshape(h/m, m, w/m, m).permute(0,2,1,3).flatten()
            // wpos_ids analogously.
            let hsm = h / smerge;
            let wsm = w / smerge;
            // build hpos_ids[h, w] = h_idx
            let mut hpos = vec![0u32; (h * w) as usize];
            let mut wpos = vec![0u32; (h * w) as usize];
            for i in 0..h {
                for j in 0..w {
                    hpos[(i * w + j) as usize] = i;
                    wpos[(i * w + j) as usize] = j;
                }
            }
            // Reorder via reshape(hsm, m, wsm, m).permute(0, 2, 1, 3).flatten()
            // Original layout indexed as [h_outer, h_inner, w_outer, w_inner]
            // → output index [h_outer, w_outer, h_inner, w_inner].
            let reorder = |src: &[u32]| -> Vec<u32> {
                let mut out = vec![0u32; src.len()];
                let mut idx = 0usize;
                let m = smerge as usize;
                for ho in 0..hsm as usize {
                    for wo in 0..wsm as usize {
                        for hi in 0..m {
                            for wi in 0..m {
                                let h_total = ho * m + hi;
                                let w_total = wo * m + wi;
                                out[idx] = src[h_total * (w as usize) + w_total];
                                idx += 1;
                            }
                        }
                    }
                }
                out
            };
            let hpos_r = reorder(&hpos);
            let wpos_r = reorder(&wpos);

            // Repeat for each temporal frame t
            for _ in 0..t {
                for k in 0..(h * w) as usize {
                    pos_ids_orig.push((hpos_r[k], wpos_r[k]));
                }
            }
            if h > max_grid_size {
                max_grid_size = h;
            }
            if w > max_grid_size {
                max_grid_size = w;
            }
        }

        // Pass 2: window index + cu_window_seqlens.
        // Python computes index over the POST-MERGE token grid
        // (grid_t * llm_grid_h * llm_grid_w units), pads to windows, and
        // then flattens with -100 entries dropped.
        let mut window_index: Vec<u32> = Vec::new();
        let mut cu_window_seqlens: Vec<u32> = vec![0u32];
        let mut window_index_id: u32 = 0;
        let smerge_unit_u32 = smerge_unit as u32;

        for &[t, h, w] in grid_thw {
            let llm_h = (h / smerge) as i64;
            let llm_w = (w / smerge) as i64;
            let pad_h = (vit_merger_window_size as i64 - llm_h % vit_merger_window_size as i64)
                .rem_euclid(vit_merger_window_size as i64);
            let pad_w = (vit_merger_window_size as i64 - llm_w % vit_merger_window_size as i64)
                .rem_euclid(vit_merger_window_size as i64);
            let padded_h = llm_h + pad_h;
            let padded_w = llm_w + pad_w;
            let num_windows_h = padded_h / vit_merger_window_size as i64;
            let num_windows_w = padded_w / vit_merger_window_size as i64;

            // index_padded[t, h, w] = t*llm_h*llm_w + h*llm_w + w  (or -100 if padded)
            // Layout after pad: [grid_t, padded_h, padded_w]
            // After reshape + permute (per Python):
            //   [grid_t, num_windows_h, num_windows_w, wm, wm]
            // wm = vit_merger_window_size
            let wm = vit_merger_window_size as i64;
            let merge_unit_count = (llm_h * llm_w) as i64;
            for ti in 0..t as i64 {
                for nh in 0..num_windows_h {
                    for nw in 0..num_windows_w {
                        let mut seqlen_this: u32 = 0;
                        for hi in 0..wm {
                            for wi in 0..wm {
                                let hh = nh * wm + hi;
                                let ww = nw * wm + wi;
                                if hh < llm_h && ww < llm_w {
                                    let idx_post_merge =
                                        ti * merge_unit_count + hh * llm_w + ww;
                                    window_index.push(window_index_id + idx_post_merge as u32);
                                    seqlen_this += 1;
                                }
                            }
                        }
                        // cu_window_seqlens cumulative in TOKEN units
                        let prev = *cu_window_seqlens.last().unwrap();
                        cu_window_seqlens.push(prev + seqlen_this * smerge_unit_u32);
                    }
                }
            }
            window_index_id += (t as i64 * llm_h * llm_w) as u32;
        }

        // unique_consecutive (skip duplicate boundaries from zero-pad windows)
        let mut dedup: Vec<u32> = Vec::with_capacity(cu_window_seqlens.len());
        for v in cu_window_seqlens {
            if dedup.last().copied() != Some(v) {
                dedup.push(v);
            }
        }

        // cu_seqlens_full per image/frame (full-attention blocks):
        //   repeat_interleave(h * w, t).cumsum() then prepend 0
        let mut cu_seqlens_full: Vec<u32> = vec![0u32];
        for &[t, h, w] in grid_thw {
            let per_frame = (h * w) as u32;
            for _ in 0..t {
                let prev = *cu_seqlens_full.last().unwrap();
                cu_seqlens_full.push(prev + per_frame);
            }
        }

        let seq_len = pos_ids_orig.len();
        (
            pos_ids_orig,
            window_index,
            dedup,
            cu_seqlens_full,
            max_grid_size,
            seq_len,
        )
    }

    /// Build a `[1, 1, seq_len, seq_len]` BF16 mask whose ones-positions are
    /// allowed and zeros-positions are blocked, matching flame's SDPA
    /// "boolean-style" mask convention.
    ///
    /// `cu_seqlens` is in TOKEN units (post-merge accounting already baked in).
    /// Within each block defined by `[cu_seqlens[i-1], cu_seqlens[i])` we
    /// allow full attention; cross-block is blocked.
    fn build_block_mask(
        seq_len: usize,
        cu_seqlens: &[u32],
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0f32; seq_len * seq_len];
        for i in 1..cu_seqlens.len() {
            let lo = cu_seqlens[i - 1] as usize;
            let hi = cu_seqlens[i] as usize;
            for r in lo..hi {
                for c in lo..hi {
                    data[r * seq_len + c] = 1.0;
                }
            }
        }
        Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)
    }

    /// Reorder a `[seq_len, ...]` tensor by window units. The reordering is
    /// expressed at the unit-of-`spatial_merge_unit` granularity:
    /// `out[i*U : (i+1)*U] = inp[window_index[i]*U : (window_index[i]+1)*U]`.
    fn reorder_by_window(
        &self,
        x: &Tensor,           // [seq_len, last_dim]
        window_index: &[u32],
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        assert_eq!(dims.len(), 2, "reorder_by_window: expected 2D, got {dims:?}");
        let seq_len = dims[0];
        let last = dims[1];
        let unit = self.cfg.spatial_merge_unit();
        let n_units = seq_len / unit;
        debug_assert_eq!(seq_len, n_units * unit);
        debug_assert_eq!(n_units, window_index.len());

        // Build a flat index list over seq_len positions, then gather.
        let mut full_idx: Vec<i32> = Vec::with_capacity(seq_len);
        for &wi in window_index {
            let base = (wi as usize) * unit;
            for j in 0..unit {
                full_idx.push((base + j) as i32);
            }
        }
        let idx_t = Tensor::from_vec(
            full_idx.iter().map(|&v| v as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let out = x.index_select0(&idx_t)?;
        out.reshape(&[seq_len, last])
    }

    /// Inverse permutation of `window_index`.
    fn argsort_window_index(window_index: &[u32]) -> Vec<u32> {
        let n = window_index.len();
        let mut rev = vec![0u32; n];
        for (out_pos, &orig_unit) in window_index.iter().enumerate() {
            rev[orig_unit as usize] = out_pos as u32;
        }
        rev
    }

    // ------------------------------------------------------------------
    // Vision block forward
    // ------------------------------------------------------------------

    fn block_forward(
        &self,
        block_idx: usize,
        hidden: &Tensor,           // [seq_len, embed_dim] BF16
        cos: &Tensor,              // [1, 1, seq_len, head_dim/2] BF16
        sin: &Tensor,
        attn_mask: &Tensor,        // [1, 1, seq_len, seq_len] BF16
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let h = cfg.num_heads;
        let d = cfg.head_dim();
        let prefix = format!("vit_model.blocks.{block_idx}");
        let seq_len = hidden.shape().dims()[0];

        // ---- Self-attention pre-norm ----
        let norm1_w = self.w(&format!("{prefix}.norm1.weight"))?;
        let normed = Self::rms_norm(hidden, norm1_w, cfg.eps)?;

        // ---- Fused QKV → split ----
        let qkv_w = self.w(&format!("{prefix}.attn.qkv.weight"))?;
        let qkv_b = self.w(&format!("{prefix}.attn.qkv.bias"))?;
        // [seq_len, 3 * embed_dim]
        let qkv = Self::linear_bias(&normed, qkv_w, qkv_b)?;
        // Reshape → [seq_len, 3, num_heads, head_dim]
        let qkv = qkv.reshape(&[seq_len, 3, h, d])?;
        // Permute → [3, seq_len, num_heads, head_dim], unbind dim 0.
        let qkv = qkv.permute(&[1, 0, 2, 3])?;
        // Slice into Q, K, V along the leading dim (length-3).
        let q = qkv.narrow(0, 0, 1)?.reshape(&[seq_len, h, d])?;
        let k = qkv.narrow(0, 1, 1)?.reshape(&[seq_len, h, d])?;
        let v = qkv.narrow(0, 2, 1)?.reshape(&[seq_len, h, d])?;

        // ---- 2D RoPE on Q/K (half-split) ----
        // flame's rope_halfsplit_bf16 expects [B, H, N, D] with cos/sin
        // [1,1,N,D/2]. Permute q,k to [1, H, N, D] for that.
        let q4 = q
            .reshape(&[1, seq_len, h, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;
        let k4 = k
            .reshape(&[1, seq_len, h, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;
        let q4 = flame_core::bf16_ops::rope_halfsplit_bf16(&q4, cos, sin)?;
        let k4 = flame_core::bf16_ops::rope_halfsplit_bf16(&k4, cos, sin)?;

        // V → [1, H, N, D]
        let v4 = v
            .reshape(&[1, seq_len, h, d])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;

        // ---- SDPA ----
        let attn_out = flame_sdpa(&q4, &k4, &v4, Some(attn_mask))?;
        // [1, H, N, D] → [N, H, D] → [N, H*D]
        let attn_out = attn_out
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[seq_len, h * d])?;

        // ---- Output proj ----
        let proj_w = self.w(&format!("{prefix}.attn.proj.weight"))?;
        let proj_b = self.w(&format!("{prefix}.attn.proj.bias"))?;
        let attn_out = Self::linear_bias(&attn_out, proj_w, proj_b)?;

        // Residual #1
        let hidden = hidden.add(&attn_out)?;

        // ---- MLP pre-norm + SwiGLU ----
        let norm2_w = self.w(&format!("{prefix}.norm2.weight"))?;
        let mlp_in = Self::rms_norm(&hidden, norm2_w, cfg.eps)?;
        let gate_w = self.w(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let gate_b = self.w(&format!("{prefix}.mlp.gate_proj.bias"))?;
        let up_w = self.w(&format!("{prefix}.mlp.up_proj.weight"))?;
        let up_b = self.w(&format!("{prefix}.mlp.up_proj.bias"))?;
        let down_w = self.w(&format!("{prefix}.mlp.down_proj.weight"))?;
        let down_b = self.w(&format!("{prefix}.mlp.down_proj.bias"))?;

        let gate = Self::linear_bias(&mlp_in, gate_w, gate_b)?;
        let up = Self::linear_bias(&mlp_in, up_w, up_b)?;
        let mlp_mid = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_bias(&mlp_mid, down_w, down_b)?;

        // Residual #2
        hidden.add(&mlp_out)
    }

    // ------------------------------------------------------------------
    // Patch merger
    // ------------------------------------------------------------------

    /// `[seq_len, embed_dim] → [seq_len / spatial_merge_unit, out_hidden_size]`.
    ///
    /// Python (paraphrased):
    /// ```text
    /// x = RMSNorm(x)
    /// x = x.view(-1, embed_dim * spatial_merge_unit)
    /// x = Linear1(x); x = GELU(x); x = Linear2(x)
    /// ```
    fn patch_merger(&self, x: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let ln_w = self.w("vit_model.merger.ln_q.weight")?;
        let l0_w = self.w("vit_model.merger.mlp.0.weight")?;
        let l0_b = self.w("vit_model.merger.mlp.0.bias")?;
        let l2_w = self.w("vit_model.merger.mlp.2.weight")?;
        let l2_b = self.w("vit_model.merger.mlp.2.bias")?;

        let normed = Self::rms_norm(x, ln_w, cfg.eps)?;
        let merged_dim = cfg.embed_dim * cfg.spatial_merge_unit();
        let dims = normed.shape().dims().to_vec();
        let n_units = dims[0] / cfg.spatial_merge_unit();
        let flat = normed.reshape(&[n_units, merged_dim])?;
        let h0 = Self::linear_bias(&flat, l0_w, l0_b)?;
        let h0 = h0.gelu()?;
        Self::linear_bias(&h0, l2_w, l2_b)
    }

    // ------------------------------------------------------------------
    // Top-level forward
    // ------------------------------------------------------------------

    /// Run the vision tower.
    ///
    /// `pixel_values`: `[N_patches, in_channels * temporal_patch_size *
    ///                  patch_size * patch_size]` BF16, pre-patchified —
    ///                  matches Python's flat-patch input format.
    /// `grid_thw`: host-side `(T, H, W)` per image/video sample. `H` and `W`
    ///             count *patch* cells (not pixels): they must be divisible
    ///             by `spatial_merge_size`.
    ///
    /// Returns `[N_patches / spatial_merge_unit, out_hidden_size]` BF16.
    /// Diagnostic forward: returns `(post_merger, captures)` where
    /// `captures` always contains `"pre_block_0"` and an entry per
    /// `capture_layers` keyed `"block.{idx}"`.
    ///
    /// The plain [`forward`] entry point just calls this and discards the
    /// captures.
    pub fn forward_capture(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
        capture_layers: &[usize],
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        self.forward_capture_with_opts(pixel_values, grid_thw, capture_layers, false)
    }

    /// Same as [`forward_capture`], but with a `force_fullatt` override that
    /// makes EVERY block use the full-attention mask (i.e., treats
    /// `fullatt_block_indexes` as `0..num_layers`). Used by the Phase G drift
    /// investigation to isolate the windowed-attention path.
    pub fn forward_capture_with_opts(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
        capture_layers: &[usize],
        force_fullatt: bool,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        let cfg = &self.cfg;
        let mut captures = std::collections::HashMap::new();

        let expected: usize = grid_thw.iter().map(|&[t, h, w]| (t * h * w) as usize).sum();
        let got = pixel_values.shape().dims()[0];
        if expected != got {
            return Err(Error::InvalidInput(format!(
                "qwen25vl_vit forward: pixel_values has {got} patches but grid_thw implies {expected}"
            )));
        }
        let flat_dim = cfg.flat_patch_dim();
        if pixel_values.shape().dims()[1] != flat_dim {
            return Err(Error::InvalidInput(format!(
                "qwen25vl_vit forward: pixel_values last dim {} != expected {}",
                pixel_values.shape().dims()[1],
                flat_dim
            )));
        }

        let mut h = self.patch_embed(pixel_values)?;
        let (pos_ids_orig, window_index, cu_window_seqlens, cu_seqlens_full, max_grid, seq_len) =
            self.window_layout(grid_thw);
        debug_assert_eq!(seq_len, h.shape().dims()[0]);
        h = self.reorder_by_window(&h, &window_index)?;

        let unit = cfg.spatial_merge_unit();
        let mut pos_ids_reord: Vec<(u32, u32)> = Vec::with_capacity(seq_len);
        for &wi in &window_index {
            let base = (wi as usize) * unit;
            for j in 0..unit {
                pos_ids_reord.push(pos_ids_orig[base + j]);
            }
        }
        let (cos, sin) = self.build_rope_cossin(&pos_ids_reord, max_grid)?;

        let mask_full =
            Self::build_block_mask(seq_len, &cu_seqlens_full, &self.device)?;
        let mask_window =
            Self::build_block_mask(seq_len, &cu_window_seqlens, &self.device)?;

        captures.insert("pre_block_0".to_string(), h.clone());

        // Dump cu_window_seqlens + window_index when the diagnostic env flag
        // is set. Phase G investigation only; gated so production paths
        // never pay for this.
        if std::env::var("VIT_DUMP_WINDOW_META").as_deref() == Ok("1") {
            eprintln!(
                "[VIT_DUMP_WINDOW_META] seq_len={} cu_window_seqlens.len={} cu_seqlens_full={:?}",
                seq_len,
                cu_window_seqlens.len(),
                cu_seqlens_full
            );
            eprintln!(
                "[VIT_DUMP_WINDOW_META] cu_window_seqlens (first 32): {:?}",
                &cu_window_seqlens[..cu_window_seqlens.len().min(32)]
            );
            eprintln!(
                "[VIT_DUMP_WINDOW_META] cu_window_seqlens (last 8): {:?}",
                &cu_window_seqlens[cu_window_seqlens.len().saturating_sub(8)..]
            );
            eprintln!(
                "[VIT_DUMP_WINDOW_META] window_index.len={} first16={:?} last16={:?}",
                window_index.len(),
                &window_index[..window_index.len().min(16)],
                &window_index[window_index.len().saturating_sub(16)..]
            );
            if force_fullatt {
                eprintln!("[VIT_DUMP_WINDOW_META] force_fullatt=TRUE — every block uses mask_full");
            }
        }

        for layer_idx in 0..cfg.num_layers {
            let mask = if force_fullatt || cfg.fullatt_block_indexes.contains(&layer_idx) {
                &mask_full
            } else {
                &mask_window
            };
            h = self.block_forward(layer_idx, &h, &cos, &sin, mask)?;
            if capture_layers.contains(&layer_idx) {
                captures.insert(format!("block.{layer_idx}"), h.clone());
            }
        }

        let merged = self.patch_merger(&h)?;
        let rev = Self::argsort_window_index(&window_index);
        let n_units = window_index.len();
        let last = cfg.out_hidden_size;
        let mut rev_idx: Vec<i32> = Vec::with_capacity(n_units);
        for &v in &rev {
            rev_idx.push(v as i32);
        }
        let idx_t = Tensor::from_vec(
            rev_idx.iter().map(|&v| v as f32).collect(),
            Shape::from_dims(&[n_units]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let out = merged.index_select0(&idx_t)?.reshape(&[n_units, last])?;
        captures.insert("post_merger".to_string(), out.clone());
        Ok((out, captures))
    }

    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[[u32; 3]]) -> Result<Tensor> {
        let cfg = &self.cfg;

        // Sanity: total patches must equal sum(t*h*w).
        let expected: usize = grid_thw.iter().map(|&[t, h, w]| (t * h * w) as usize).sum();
        let got = pixel_values.shape().dims()[0];
        if expected != got {
            return Err(Error::InvalidInput(format!(
                "qwen25vl_vit forward: pixel_values has {got} patches but grid_thw implies {expected}"
            )));
        }
        let flat_dim = cfg.flat_patch_dim();
        if pixel_values.shape().dims()[1] != flat_dim {
            return Err(Error::InvalidInput(format!(
                "qwen25vl_vit forward: pixel_values last dim {} != expected {}",
                pixel_values.shape().dims()[1],
                flat_dim
            )));
        }

        // ---- Patch embed ----
        let mut h = self.patch_embed(pixel_values)?; // [seq_len, embed_dim]

        // ---- Window layout / rotary metadata ----
        let (pos_ids_orig, window_index, cu_window_seqlens, cu_seqlens_full, max_grid, seq_len) =
            self.window_layout(grid_thw);
        debug_assert_eq!(seq_len, h.shape().dims()[0]);

        // Reorder hidden_states and pos_ids by window units.
        h = self.reorder_by_window(&h, &window_index)?;

        // Build reordered pos_ids (apply same unit-permutation to the token list).
        let unit = cfg.spatial_merge_unit();
        let mut pos_ids_reord: Vec<(u32, u32)> = Vec::with_capacity(seq_len);
        for &wi in &window_index {
            let base = (wi as usize) * unit;
            for j in 0..unit {
                pos_ids_reord.push(pos_ids_orig[base + j]);
            }
        }

        let (cos, sin) = self.build_rope_cossin(&pos_ids_reord, max_grid)?;

        // Pre-build the two attention masks (full + windowed). Reused across blocks.
        let mask_full =
            Self::build_block_mask(seq_len, &cu_seqlens_full, &self.device)?;
        let mask_window =
            Self::build_block_mask(seq_len, &cu_window_seqlens, &self.device)?;

        // ---- Blocks ----
        for layer_idx in 0..cfg.num_layers {
            let mask = if cfg.fullatt_block_indexes.contains(&layer_idx) {
                &mask_full
            } else {
                &mask_window
            };
            h = self.block_forward(layer_idx, &h, &cos, &sin, mask)?;
        }

        // ---- Merger ----
        let merged = self.patch_merger(&h)?; // [seq_len/unit, out_hidden_size]

        // ---- Reverse window reorder at unit-granularity (single tokens now) ----
        let rev = Self::argsort_window_index(&window_index);
        let n_units = window_index.len();
        let last = cfg.out_hidden_size;
        let mut rev_idx: Vec<i32> = Vec::with_capacity(n_units);
        for &v in &rev {
            rev_idx.push(v as i32);
        }
        let idx_t = Tensor::from_vec(
            rev_idx.iter().map(|&v| v as f32).collect(),
            Shape::from_dims(&[n_units]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let out = merged.index_select0(&idx_t)?.reshape(&[n_units, last])?;
        Ok(out)
    }
}
