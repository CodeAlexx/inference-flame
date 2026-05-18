//! HiDreamI1Dit — top-level DiT mirror of `HiDreamImageTransformer2DModel`.
//!
//! Mirror: `transformer_hidream_image.py:230-505`.
//!
//! ## Forward pipeline
//! 1. `timesteps` → `t_embedder` → `[B, dim]` BF16.
//! 2. `pooled_embeds` (CLIP-L+CLIP-G concat → `[B, 2048]`) → `p_embedder` → `[B, dim]`.
//! 3. `adaln_input = t_emb + p_emb`. This is the modulation input every block gets.
//! 4. **Patchify** the latents: `[B, in_channels, H, W]` →
//!    `[B, N_img, in_channels*patch*patch]`. Square training path takes the
//!    rearrange branch (no per-sample image_tokens_masks).
//! 5. Build `img_ids` `[N_img, 3]` with `(0, py, px)` triples and
//!    `txt_ids = zeros(N_text_initial + N_llama, 3)`. EmbedND RoPE table built once.
//! 6. **Project** image patches through `x_embedder` (a single `Linear`).
//! 7. **Project** text streams: each of `caption_projection[0..num_double+num_single]`
//!    handles a Llama layer; `caption_projection[-1]` handles T5.
//! 8. Build `initial_encoder = cat([t5_projected, last_llama_projected], dim=1)`.
//!    Per-double-block: `text_tokens = cat([initial_encoder, cur_llama], dim=1)`,
//!    fed to double_block.forward. The block returns `(image_tokens, text_tokens)`;
//!    we slice text_tokens back to `initial_encoder_seq_len` (drops the per-block
//!    Llama slot).
//! 9. After double blocks: concat `[hidden_states, initial_encoder]` to feed the
//!    single blocks. Per-single-block: `hidden = cat([hidden, cur_llama], dim=1)`,
//!    run single_block, slice back.
//! 10. **Final layer**: adaLN + Linear → unpatchify back to `[B, out_channels, H, W]`.
//!
//! ## Caption projection routing
//! `caption_channels = [T5_dim, Llama_dim]` (a 2-element list passed to
//! `__init__`). Internally the model builds a list of length
//! `num_double + num_single + 1`:
//!   - indices `0..num_double+num_single` each project Llama hidden states
//!     (in_dim = Llama_dim = 4096)
//!   - index `num_double+num_single` projects T5 (in_dim = T5_dim = 4096)
//!
//! ## llama_layers
//! `self.llama_layers` is a list of length `num_double + num_single + 1`
//! that selects which Llama hidden-states-layer to feed into each
//! `caption_projection[i]` (per `transformer_hidream_image.py:410`).
//! For HiDream-I1, this is set by the model config from the HF repo's
//! `config.json::llama_layers`. We thread it through here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_diffusion::BlockOffloader;

use super::double_block::{self, DoubleBlockCfg};
use super::final_layer;
use super::single_block::{self, SingleBlockCfg};
use super::text_encoder_cat::HiDreamEncoderInputs;
use super::weight_loader::{is_shared_key, HiDreamI1Facilitator};

/// Inner / canonical HiDream-I1 config.
///
/// Default values via [`HiDreamI1Config::i1_full`] correspond to the
/// `HiDream-ai/HiDream-I1-Full` checkpoint. Source-of-truth fields documented
/// at `transformer_hidream_image.py:237-253` (`__init__` signature).
#[derive(Clone, Debug)]
pub struct HiDreamI1Config {
    /// Patch size. Default 2.
    pub patch_size: usize,
    /// Raw VAE latent channel count (= 16 for FLUX VAE). The packed
    /// x_embedder input dim is derived as `latent_channels * patch_size^2`
    /// (= 64 for I1-Full). BUG #3 fix.
    pub latent_channels: usize,
    /// Output channel count for unpatchify (= `latent_channels` for
    /// flow-match velocity, i.e. 16 for I1-Full). BUG #3 fix.
    pub out_channels: usize,
    /// Double-stream block count. Default 16.
    pub num_layers: usize,
    /// Single-stream block count. Default 32.
    pub num_single_layers: usize,
    /// Per-head dim. Default 128.
    pub attention_head_dim: usize,
    /// Number of attention heads. Default 20.
    pub num_attention_heads: usize,
    /// `caption_channels = [T5_dim, Llama_dim]`. Default `[4096, 4096]`.
    pub caption_channels: [usize; 2],
    /// Pooled text embedding dim (CLIP-L 768 + CLIP-G 1280). Default 2048.
    pub text_emb_dim: usize,
    /// MoE routed expert count. Default 4.
    pub num_routed_experts: usize,
    /// MoE activated (top-k) count. Default 2.
    pub num_activated_experts: usize,
    /// EmbedND per-axis dims. Default `[64, 32, 32]` for HF I1-Full
    /// (`axis 0` is the always-zero positional channel; axes 1/2 are y/x).
    /// Sum must equal `head_dim` (128 for I1-Full). BUG #2 fix.
    pub axes_dims_rope: [usize; 3],
    /// RoPE θ. Default 10000.
    pub rope_theta: f32,
    /// Max RoPE resolution (max_h, max_w) = (128, 128) → max 4096 image tokens.
    pub max_resolution: (usize, usize),
    /// Layer indices used by `caption_projection` to pick Llama hidden states.
    /// Length = `num_layers + num_single_layers` = `total_blocks` (BUG #5
    /// fix — the previous `+1` was wrong; T5 is projected separately by the
    /// last caption_projection head). HF I1-Full canonical:
    /// `[0..31] ++ [31]*16` (length 48).
    pub llama_layers: Vec<usize>,
    /// LayerNorm epsilon. 1e-6 (matches `nn.LayerNorm(elementwise_affine=False, eps=1e-6)`).
    /// BUG #6 fix: split from `eps_rms` so RMSNorm uses its own 1e-5 default.
    /// `eps` is retained as an alias for `eps_ln` for backward compat.
    pub eps: f32,
    /// RMSNorm epsilon. 1e-5 (matches `nn.RMSNorm(...)` default in Python).
    /// Affects Q/K RMSNorm sites in both double and single blocks. BUG #6 fix.
    pub eps_rms: f32,
    /// Timestep frequency embedding dim (fed into TimestepEmbedding). 256.
    pub timestep_freq_dim: usize,
}

impl HiDreamI1Config {
    /// HiDream-I1-Full defaults (per `transformer_hidream_image.py:237-253`
    /// signature defaults + the HF I1-Full `config.json`).
    pub fn i1_full() -> Self {
        let num_layers = 16;
        let num_single_layers = 32;
        // BUG #5 fix: canonical HF I1-Full `llama_layers` is length 48
        // (= total_blocks), distributed as `[0..31] ++ [31]*16`. Blocks
        // 0..32 each pick a unique Llama hidden-states layer; blocks 32..48
        // all reuse the last Llama layer (31). The `+1` from the prior
        // length is dropped — T5 has its own caption_projection head.
        let total_blocks = num_layers + num_single_layers; // 48
        let mut llama_layers: Vec<usize> = Vec::with_capacity(total_blocks);
        for i in 0..total_blocks {
            llama_layers.push(if i < 32 { i } else { 31 });
        }
        Self {
            patch_size: 2,
            // BUG #3 fix: latent_channels = 16 (raw VAE), x_embedder input
            // dim = 16 * 2 * 2 = 64; out_channels = 16 for unpatchify.
            latent_channels: 16,
            out_channels: 16,
            num_layers,
            num_single_layers,
            attention_head_dim: 128,
            num_attention_heads: 20,
            caption_channels: [4096, 4096],
            text_emb_dim: 2048,
            num_routed_experts: 4,
            num_activated_experts: 2,
            axes_dims_rope: [64, 32, 32],
            rope_theta: 10000.0,
            max_resolution: (128, 128),
            llama_layers,
            eps: 1e-6,
            eps_rms: 1e-5,
            timestep_freq_dim: 256,
        }
    }

    /// `inner_dim = num_attention_heads * attention_head_dim`. For I1: 20×128=2560.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    /// Packed `x_embedder` input dim = `latent_channels * patch_size^2`
    /// (= 64 for I1-Full). BUG #3.
    pub fn x_embedder_in_dim(&self) -> usize {
        self.latent_channels * self.patch_size * self.patch_size
    }
    pub fn total_blocks(&self) -> usize {
        self.num_layers + self.num_single_layers
    }
    pub fn caption_projection_count(&self) -> usize {
        self.num_layers + self.num_single_layers + 1
    }
}

/// HiDream-I1 DiT.
///
/// Shared (always-resident) weights: `t_embedder`, `p_embedder`, `x_embedder`,
/// `caption_projection[0..total_blocks+1]`, `final_layer`.
/// Per-block weights (`double_stream_blocks.{i}.block.*` and
/// `single_stream_blocks.{i}.block.*`) flow through [`BlockOffloader`].
pub struct HiDreamI1Dit {
    config: HiDreamI1Config,
    device: Arc<CudaDevice>,
    shared: HashMap<String, Tensor>,
    offloader: BlockOffloader,
}

impl HiDreamI1Dit {
    /// Load from a sharded HF safetensors checkpoint.
    ///
    /// `checkpoint_paths` is the list of `diffusion_pytorch_model-*.safetensors`
    /// shards from the `transformer/` subfolder of `HiDream-ai/HiDream-I1-Full`.
    pub fn load(
        checkpoint_paths: &[&str],
        device: &Arc<CudaDevice>,
        config: HiDreamI1Config,
    ) -> Result<Self> {
        let num_double = config.num_layers;
        let total_blocks = config.total_blocks();
        let facilitator = HiDreamI1Facilitator { num_double, total_blocks };
        let offloader = BlockOffloader::load(
            checkpoint_paths,
            &facilitator,
            device.clone(),
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader HiDreamI1: {e}")))?;

        let mut shared = HashMap::new();
        for path in checkpoint_paths {
            let part = load_file_filtered(Path::new(path), device, |key| is_shared_key(key))?;
            for (k, v) in part {
                // BUG #4 fix: HF I1-Full ships shared weights as F16 on disk;
                // `fused_linear3d_native` requires BF16. Promote in-place.
                let v_bf16 = if v.dtype() != DType::BF16 {
                    v.to_dtype(DType::BF16)?
                } else {
                    v
                };
                shared.insert(k, v_bf16);
            }
        }

        log::info!(
            "[HiDreamI1] Loaded: {} blocks via BlockOffloader, {} shared weights",
            offloader.block_count(),
            shared.len()
        );

        Ok(Self { config, device: device.clone(), shared, offloader })
    }

    pub fn config(&self) -> &HiDreamI1Config {
        &self.config
    }

    /// Convenience wrapper: pre-compute `pe` (RoPE table) and `adaln_input`
    /// for the CFG loop, then run cond+uncond forwards with the cache.
    ///
    /// `latents`: `[B, in_channels, H, W]` BF16.
    /// `timesteps`: `[B]` (BF16 or F32 — coerced internally).
    /// `enc`: text-encoder bundle (T5 + Llama stack + CLIP-L + CLIP-G).
    pub fn forward(
        &mut self,
        latents: &Tensor,
        timesteps: &Tensor,
        enc: &HiDreamEncoderInputs<'_>,
    ) -> Result<Tensor> {
        let (adaln_input, pe) = self.precompute_step_cache(timesteps, enc)?;
        self.forward_cached(latents, enc, &adaln_input, &pe)
    }

    /// Pre-compute per-step cache shared between cond+uncond forwards.
    ///
    /// Returns `(adaln_input, rope_table)`. RoPE is built for the max-seq
    /// length implied by `image latent size + initial_encoder + 1 llama layer`.
    pub fn precompute_step_cache(
        &self,
        timesteps: &Tensor,
        enc: &HiDreamEncoderInputs<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let dim = self.config.inner_dim();
        let device = &self.device;

        // 1. t_embedder: Timesteps(256, flip_sin_to_cos=True, shift=0) →
        //    TimestepEmbedding(256 → dim).
        // Python multiplies timesteps NOT by 1000 here — `hidream_model.py:330`
        // passes `t` already in [0, 1000].
        let t_proj = timesteps_embedding_flip(
            timesteps,
            self.config.timestep_freq_dim,
            device,
        )?;
        let t_emb = linear_silu_linear(
            &t_proj,
            self.shared.get("t_embedder.timestep_embedder.linear_1.weight")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing t_embedder.timestep_embedder.linear_1.weight".into()))?,
            self.shared.get("t_embedder.timestep_embedder.linear_1.bias")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing t_embedder.timestep_embedder.linear_1.bias".into()))?,
            self.shared.get("t_embedder.timestep_embedder.linear_2.weight")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing t_embedder.timestep_embedder.linear_2.weight".into()))?,
            self.shared.get("t_embedder.timestep_embedder.linear_2.bias")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing t_embedder.timestep_embedder.linear_2.bias".into()))?,
        )?;

        // 2. p_embedder: TimestepEmbedding(text_emb_dim → dim) applied to pooled.
        let pooled = enc.pool_concat()?;
        let p_emb = linear_silu_linear(
            &pooled,
            self.shared.get("p_embedder.pooled_embedder.linear_1.weight")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing p_embedder.pooled_embedder.linear_1.weight".into()))?,
            self.shared.get("p_embedder.pooled_embedder.linear_1.bias")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing p_embedder.pooled_embedder.linear_1.bias".into()))?,
            self.shared.get("p_embedder.pooled_embedder.linear_2.weight")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing p_embedder.pooled_embedder.linear_2.weight".into()))?,
            self.shared.get("p_embedder.pooled_embedder.linear_2.bias")
                .ok_or_else(|| flame_core::Error::InvalidInput("Missing p_embedder.pooled_embedder.linear_2.bias".into()))?,
        )?;

        // 3. adaln_input = t_emb + p_emb (`transformer_hidream_image.py:397`)
        let adaln_input = t_emb.add(&p_emb)?;

        // 4. RoPE table — built for the worst-case seq length used by the
        //    single blocks: N_img + N_initial_enc + N_llama_per_block.
        //    N_initial_enc = T5 seq + last-llama-layer seq (both 128 default).
        //    The double blocks see only N_img + N_initial_enc + N_llama_per_block
        //    too because the per-block Llama slot is concatenated each block.
        let _ = dim;
        // Defer RoPE table construction until forward_cached (when we know
        // image latent dims). Just emit a placeholder zero-rank tensor so the
        // signature matches. Callers should treat this as "not yet built".
        //
        // For real CFG-cached usage, the trainer/sampler should call
        // `build_rope_for_resolution` directly and pass the table to
        // `forward_cached_with_rope`.
        let placeholder = Tensor::zeros_dtype(
            Shape::from_dims(&[1]),
            DType::BF16,
            device.clone(),
        )?;
        Ok((adaln_input, placeholder))
    }

    /// Forward with externally-supplied `adaln_input`. The `_rope_placeholder`
    /// arg is currently unused — RoPE is rebuilt inside per-call against the
    /// actual `n_img + initial_seq + cur_llama_seq`. TODO(M2.2 / BUG #10):
    /// either build rope once per resolution and thread it through, or drop
    /// this arg entirely from the signature.
    pub fn forward_cached(
        &mut self,
        latents: &Tensor,
        enc: &HiDreamEncoderInputs<'_>,
        adaln_input: &Tensor,
        _rope_placeholder: &Tensor,
    ) -> Result<Tensor> {
        // 1. Patchify: [B, C, H, W] → [B, N_img, C * patch * patch].
        let cfg = &self.config;
        let lat_dims = latents.shape().dims().to_vec();
        if lat_dims.len() != 4 {
            return Err(flame_core::Error::InvalidInput(format!(
                "HiDreamI1Dit::forward_cached: expected [B,C,H,W], got {lat_dims:?}"
            )));
        }
        let (b, _c, h_full, w_full) = (lat_dims[0], lat_dims[1], lat_dims[2], lat_dims[3]);
        let p = cfg.patch_size;
        if h_full % p != 0 || w_full % p != 0 {
            return Err(flame_core::Error::InvalidInput(format!(
                "HiDreamI1Dit::forward_cached: H, W must be divisible by patch_size={p}"
            )));
        }
        let p_h = h_full / p;
        let p_w = w_full / p;
        let n_img = p_h * p_w;
        // Recommendation #9 / missing edge case: enforce max_seq based on
        // `max_resolution` so we don't silently train at a length the model
        // was never conditioned on.
        let max_n_img =
            cfg.max_resolution.0 * cfg.max_resolution.1 / (cfg.patch_size * cfg.patch_size);
        if n_img > max_n_img {
            return Err(flame_core::Error::InvalidInput(format!(
                "HiDreamI1Dit::forward_cached: n_img={n_img} exceeds max_resolution-derived cap {max_n_img} (max_resolution={:?}, patch_size={p})",
                cfg.max_resolution
            )));
        }
        let img_patches = patchify_square(latents, p)?;

        // 2. x_embedder: Linear(in_channels * patch * patch, inner_dim)
        let xemb_w = self.shared.get("x_embedder.proj.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing x_embedder.proj.weight".into()))?;
        let xemb_b = self.shared.get("x_embedder.proj.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing x_embedder.proj.bias".into()))?;
        let mut img_tokens = linear_bias_3d(&img_patches, xemb_w, xemb_b)?; // [B, N_img, dim]

        // 3. Project all caption inputs through caption_projection[0..N+1].
        //    Encoder list per `transformer_hidream_image.py:412-421`:
        //      enc[k] = caption_projection[k](llama_layers[k])  for k in 0..total_blocks
        //      enc[total_blocks] = caption_projection[-1](T5)
        //    All projections produce [B, S, dim] BF16.
        let total_blocks = cfg.total_blocks();
        let n_cap = cfg.caption_projection_count(); // = total_blocks + 1
        let mut projected: Vec<Tensor> = Vec::with_capacity(n_cap);
        for k in 0..total_blocks {
            // Pick llama layer (`enc.llama_stack[self.llama_layers[k]]`).
            let layer_idx = cfg.llama_layers.get(k).copied().ok_or_else(|| {
                flame_core::Error::InvalidInput(format!(
                    "cfg.llama_layers too short ({}) for index {k}",
                    cfg.llama_layers.len()
                ))
            })?;
            let llama_k = enc.pick_llama_layer(layer_idx)?;
            let cap_w = self.shared.get(&format!("caption_projection.{k}.linear.weight"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing caption_projection.{k}.linear.weight")))?;
            // bias is False per `TextProjection` definition (line 24 of transformer_hidream_image.py).
            let proj = linear_nobias_3d(&llama_k, cap_w)?; // [B, S, dim]
            // Python reshapes to [B, -1, hidden]. With S already on dim 1 and
            // hidden on dim 2 this is a no-op for our layout; keep as-is.
            projected.push(proj);
        }
        // T5 projection via the last head.
        let t5_proj = {
            let cap_w = self.shared.get(&format!("caption_projection.{total_blocks}.linear.weight"))
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing caption_projection.{total_blocks}.linear.weight")))?;
            linear_nobias_3d(enc.t5, cap_w)?
        };
        projected.push(t5_proj);
        // Now projected.len() == n_cap.

        // 4. Initial text+T5 concat used as carry-state through double blocks.
        // Per Python line 434: `initial_encoder_hidden_states = cat([proj[-1], proj[-2]], dim=1)`
        // where proj[-1] is T5 and proj[-2] is the LAST Llama-layer projection.
        let initial = Tensor::cat(&[&projected[n_cap - 1], &projected[n_cap - 2]], 1)?;
        let initial_seq_len = initial.shape().dims()[1];

        // 5. RoPE table sized for the max seq length we'll see inside a block:
        //    `n_img + initial_seq_len + cur_llama_seq_len` (the cur_llama is added
        //    each block then sliced back off).
        let cur_llama_seq_len = projected[0].shape().dims()[1];
        let total_seq = n_img + initial_seq_len + cur_llama_seq_len;
        let pe = build_rope_table(
            b,
            n_img,
            initial_seq_len + cur_llama_seq_len,
            cfg,
            &self.device,
        )?;

        // 6. Double blocks ----------------------------------------------------
        let HiDreamI1Dit { shared: _shared, offloader, config: dcfg, device, .. } = self;
        let dcfg = dcfg.clone(); // avoid double-borrow on &self when offloader is &mut
        offloader
            .prefetch_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
        let mut initial_running = initial;
        for i in 0..dcfg.num_layers {
            let block = offloader
                .await_block(i)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if i + 1 < dcfg.total_blocks() {
                offloader
                    .prefetch_block(i + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }
            // BlockOffloader auto-transposes 2D .weight to [Cin, Cout]; restore
            // to [Cout, Cin] for fused_linear3d_native.
            let raw = untranspose_block_weights(&*block)?;
            // Strip the "double_stream_blocks.{i}.block." prefix.
            let prefix = format!("double_stream_blocks.{i}.block.");
            let weights = strip_prefix(&raw, &prefix);

            // The per-block Llama projection cur_llama is appended to the text
            // stream for this block, then sliced off after.
            let cur_llama = projected[i].clone_result()?;
            let block_text = Tensor::cat(&[&initial_running, &cur_llama], 1)?;
            let (new_img, new_text) = double_block::forward(
                &DoubleBlockCfg {
                    dim: dcfg.inner_dim(),
                    num_heads: dcfg.num_attention_heads,
                    head_dim: dcfg.attention_head_dim,
                    num_routed_experts: dcfg.num_routed_experts,
                    num_activated_experts: dcfg.num_activated_experts,
                    eps: dcfg.eps,
                    eps_rms: dcfg.eps_rms,
                },
                &img_tokens,
                &block_text,
                adaln_input,
                None, // square training path → no image_tokens_masks
                &pe,
                &weights,
                device,
            )?;
            img_tokens = new_img;
            // Slice off the per-block Llama slot from the carry-state.
            initial_running = new_text.narrow(1, 0, initial_seq_len)?;
            if i % 4 == 0 || i + 1 == dcfg.num_layers {
                log::info!("[HiDreamI1] Double block {}/{}", i + 1, dcfg.num_layers);
            }
        }

        // 7. Concat [img_tokens, initial_encoder] for the single-stream loop.
        let mut hidden = Tensor::cat(&[&img_tokens, &initial_running], 1)?;
        let hidden_no_llama_seq_len = hidden.shape().dims()[1];

        for i in 0..dcfg.num_single_layers {
            let block_idx = dcfg.num_layers + i;
            let block = offloader
                .await_block(block_idx)
                .map_err(|e| flame_core::Error::InvalidInput(format!("await: {e}")))?;
            if block_idx + 1 < dcfg.total_blocks() {
                offloader
                    .prefetch_block(block_idx + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }
            let raw = untranspose_block_weights(&*block)?;
            let prefix = format!("single_stream_blocks.{i}.block.");
            let weights = strip_prefix(&raw, &prefix);

            // Append cur_llama for this block index.
            let cur_llama = projected[block_idx].clone_result()?;
            hidden = Tensor::cat(&[&hidden, &cur_llama], 1)?;
            hidden = single_block::forward(
                &SingleBlockCfg {
                    dim: dcfg.inner_dim(),
                    num_heads: dcfg.num_attention_heads,
                    head_dim: dcfg.attention_head_dim,
                    num_routed_experts: dcfg.num_routed_experts,
                    num_activated_experts: dcfg.num_activated_experts,
                    eps: dcfg.eps,
                    eps_rms: dcfg.eps_rms,
                },
                &hidden,
                adaln_input,
                None,
                &pe,
                &weights,
                device,
            )?;
            // Slice off cur_llama again to keep the carry size fixed.
            hidden = hidden.narrow(1, 0, hidden_no_llama_seq_len)?;
            if i % 8 == 0 || i + 1 == dcfg.num_single_layers {
                log::info!("[HiDreamI1] Single block {}/{}", i + 1, dcfg.num_single_layers);
            }
        }

        // 8. Drop the text-carry portion: keep only image tokens.
        let img_out = hidden.narrow(1, 0, n_img)?;

        // 9. Final layer.
        let out = final_layer::forward(&img_out, adaln_input, &self.shared, dcfg.eps)?;

        // 10. Unpatchify back to [B, out_channels, H, W].
        let unpatched = unpatchify_square(&out, b, dcfg.out_channels, p_h, p_w, dcfg.patch_size)?;
        // silence unused
        let _ = total_seq;
        Ok(unpatched)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn strip_prefix(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    for (k, v) in weights.iter() {
        if let Some(rest) = k.strip_prefix(prefix) {
            out.insert(rest.to_string(), v.clone());
        }
    }
    out
}

fn untranspose_block_weights(raw: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
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

/// Sinusoidal timesteps (flip_sin_to_cos=True, downscale_freq_shift=0).
fn timesteps_embedding_flip(
    t: &Tensor,
    dim: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let in_dtype = t.dtype();
    let t_f32 = t.to_dtype(DType::F32)?;
    let half = dim / 2;
    let max_period = 10000.0f64;
    let freq_data: Vec<f32> = (0..half)
        .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]), device.clone())?;
    let t_col = t_f32.unsqueeze(1)?;
    let args = t_col.matmul(&freqs)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let emb = Tensor::cat(&[&cos, &sin], 1)?;
    if in_dtype == DType::F32 {
        Ok(emb)
    } else {
        emb.to_dtype(in_dtype)
    }
}

/// Run a linear projection that accepts either 2D `[B, C]` or 3D `[B, N, C]`
/// inputs. `fused_linear3d_native` requires 3D, so 2D inputs are temporarily
/// promoted via unsqueeze(0) → squeeze(0). BUG #1 fix.
pub(crate) fn linear_compat(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 2 {
        let x3 = x.unsqueeze(0)?;
        let o3 = flame_core::ops::fused_inference::fused_linear3d_native(&x3, weight, bias)?;
        o3.squeeze(Some(0))
    } else {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, bias)
    }
}

/// Apply `TimestepEmbedding` (= `Linear → SiLU → Linear`). Shapes: x `[B,in]`,
/// w1 `[hidden, in]`, b1 `[hidden]`, w2 `[hidden_out, hidden]`, b2 `[hidden_out]`.
fn linear_silu_linear(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let h = linear_compat(x, w1, Some(b1))?;
    let h = h.silu()?;
    linear_compat(&h, w2, Some(b2))
}

/// 3D Linear with bias for the square training path (B>1 reshape trick).
fn linear_bias_3d(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 3 && dims[0] > 1 {
        let (b, n, c) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[1, b * n, c])?;
        let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, Some(bias))?;
        let out_c = weight.shape().dims()[0];
        out.reshape(&[b, n, out_c])
    } else {
        linear_compat(x, weight, Some(bias))
    }
}

fn linear_nobias_3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 3 && dims[0] > 1 {
        let (b, n, c) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[1, b * n, c])?;
        let out = flame_core::ops::fused_inference::fused_linear3d_native(&flat, weight, None)?;
        let out_c = weight.shape().dims()[0];
        out.reshape(&[b, n, out_c])
    } else {
        linear_compat(x, weight, None)
    }
}

/// Square-only patchify (no per-sample masks).
///
/// `latents`: `[B, C, H, W]` → `[B, (H/p)*(W/p), p*p*C]`.
/// Python (`transformer_hidream_image.py:356`):
///   `x = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=p, p2=p)`
fn patchify_square(x: &Tensor, p: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let (ph, pw) = (h / p, w / p);
    // [B, C, H, W] → [B, C, ph, p, pw, p]
    let r = x.reshape(&[b, c, ph, p, pw, p])?;
    // → [B, ph, pw, p, p, C]
    let r = r.permute(&[0, 2, 4, 3, 5, 1])?;
    // → [B, ph*pw, p*p*C]
    r.reshape(&[b, ph * pw, p * p * c])
}

/// Square-only unpatchify.
///
/// `x`: `[B, N_img, p*p*out_channels]` → `[B, out_channels, ph*p, pw*p]`.
/// Python (`transformer_hidream_image.py:325-333`):
///   `einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)', p1=p, p2=p)`
fn unpatchify_square(
    x: &Tensor,
    b: usize,
    out_channels: usize,
    ph: usize,
    pw: usize,
    p: usize,
) -> Result<Tensor> {
    // [B, ph*pw, p*p*out_channels] → [B, ph, pw, p, p, out_channels]
    let r = x.reshape(&[b, ph, pw, p, p, out_channels])?;
    // → [B, out_channels, ph, p, pw, p]
    let r = r.permute(&[0, 5, 1, 3, 2, 4])?;
    // → [B, out_channels, ph*p, pw*p]
    r.reshape(&[b, out_channels, ph * p, pw * p])
}

/// Build the FLUX-style EmbedND RoPE table for the given image and text seq counts.
///
/// Mirrors `transformer_hidream_image.py:399-430` + `embeddings.py::EmbedND`:
///   - img_ids = [pH, pW, 3]: `(0, py, px)` (axis 0 = 0 by convention).
///   - txt_ids = zeros(N_text_total, 3).
///   - ids = cat([img_ids, txt_ids], dim=1).
///   - pe  = EmbedND(theta=10000, axes_dim=(32, 32))(ids).
///     This produces shape `[B?, 1, S, axes_dim_sum / 2, 2, 2]`.
///
/// Our axes_dims_rope is (32, 32) so total = 64 = head_dim/2 entries; per
/// pair we emit a 2x2 matrix block `[[cos, -sin], [sin, cos]]`.
///
/// Returns `[1, 1, S, 32, 2, 2]` BF16.
fn build_rope_table(
    _batch_size: usize,
    n_img: usize,
    n_text: usize,
    cfg: &HiDreamI1Config,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    // img_ids: build (0, py, px) for each token; flatten.
    // ph = (n_img as f64).sqrt() — but we don't know p_h/p_w here. Pass them
    // through? For now we assume square N_img = K*K. The caller passes the
    // actual n_img after patchify so reconstruct K.
    let k = (n_img as f64).sqrt() as usize;
    if k * k != n_img {
        return Err(flame_core::Error::InvalidInput(format!(
            "build_rope_table: non-square n_img={n_img}; M2 supports only square images"
        )));
    }
    let n_total = n_img + n_text;

    // Per-axis IDs as f32. HF I1-Full uses `axes_dims_rope = [64, 32, 32]` so
    // EmbedND consumes THREE axes: axis 0 is the always-zero positional
    // channel (`img_ids[..., 0] = 0` in Python), axes 1 and 2 hold y,x.
    // BUG #7 fix: include axis 0 zeros so axes 1/2 land on the correct
    // head_dim sub-slices.
    let zeros: Vec<f32> = vec![0.0; n_total];
    let mut y_ids: Vec<f32> = Vec::with_capacity(n_total);
    let mut x_ids: Vec<f32> = Vec::with_capacity(n_total);
    for py in 0..k {
        for px in 0..k {
            y_ids.push(py as f32);
            x_ids.push(px as f32);
        }
    }
    // Text token ids are all zeros.
    for _ in 0..n_text {
        y_ids.push(0.0);
        x_ids.push(0.0);
    }

    // For each axis with dim d_axis, compute `rope(pos, d_axis, theta)`:
    //   freqs = theta**(-arange(0, d_axis, 2) / d_axis)         shape [d_axis/2]
    //   out   = pos[..., None] * freqs[None, :]                 shape [S, d_axis/2]
    //   pe    = cat([cos, -sin, sin, cos]).reshape(S, d_axis/2, 2, 2)
    //
    // Then EmbedND concatenates per-axis pes along dim -3 (the pair dim).
    let axes = cfg.axes_dims_rope;
    let mut per_axis: Vec<Tensor> = Vec::with_capacity(axes.len());
    for (axis_i, &d_axis) in axes.iter().enumerate() {
        assert!(d_axis % 2 == 0, "axes_dim must be even");
        let half = d_axis / 2;
        let pos = match axis_i {
            0 => &zeros,
            1 => &y_ids,
            _ => &x_ids,
        };
        let mut data = Vec::with_capacity(n_total * half * 4);
        for &p in pos {
            for i in 0..half {
                let scale = (2 * i) as f64 / d_axis as f64;
                let freq = (1.0 / (cfg.rope_theta as f64).powf(scale)) as f32;
                let arg = (p as f64 * freq as f64) as f32;
                let c = arg.cos();
                let s = arg.sin();
                // matrix [[cos, -sin], [sin, cos]] in row-major:
                // (0,0)=cos, (0,1)=-sin, (1,0)=sin, (1,1)=cos
                data.push(c);
                data.push(-s);
                data.push(s);
                data.push(c);
            }
        }
        let t = Tensor::from_vec(
            data,
            Shape::from_dims(&[n_total, half, 2, 2]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        per_axis.push(t);
    }
    // Cat along dim -3 (the pair-count axis = dim 1 in our [S, half, 2, 2] layout).
    let refs: Vec<&Tensor> = per_axis.iter().collect();
    let cat = Tensor::cat(&refs, 1)?; // [S, total_half, 2, 2]
    // Add leading batch + head dims to match Python's `.unsqueeze(2)` semantics.
    // Python output rank is [B?, 1, S, D/2, 2, 2]; ours is [1, 1, S, D/2, 2, 2].
    cat.unsqueeze(0)?.unsqueeze(0)
}
