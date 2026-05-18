//! `HiDreamO1Model` — full Qwen3-VL text spine + 3 HiDream heads.
//!
//! Reference (`qwen3_vl_transformers.py`):
//! - `Qwen3VLModel.__init__` lines 1030-1050 — heads + decoder wiring
//! - `Qwen3VLModel._forward_generation` lines 1379-1532 — denoise step
//! - `Qwen3VLTextModel.forward` lines 806-933 — decoder loop + final norm
//!
//! Phase 2b scope: a single forward pass for one denoising step. The
//! scheduler, CFG batching, and the chat-template tokenizer all live in
//! Phase 2c (`pipeline.rs`).
//!
//! ## Forward summary (T2I, no ref-image, single batch)
//!
//! ```text
//! input_ids                  text tokens incl. <|tms_token|>
//!         │                  + N "image" placeholder tokens
//!         ▼ embed_tokens
//! text_emb [B, S_text, H]
//!         │
//!         │  t_emb = TimestepEmbedder(timestep)               [B, H]
//!         ▼  scatter t_emb into every <|tms_token|> slot
//!         │  (Python: torch.where(tms_mask_3d, t_emb_expanded, x))
//!         │                          ─ qwen3_vl_transformers.py:1449-1452
//!         │
//!         │  patch_emb = BottleneckPatchEmbed(noise_patches)  [B, L, H]
//!         ▼  cat([text_emb, patch_emb], dim=1)                [B, S_total, H]
//!         │                          ─ qwen3_vl_transformers.py:1458-1459
//!         │
//!         │  build mixed causal/full mask from `vinput_mask`  [B, 1, S, S]
//!         │  build MRoPE cos/sin from position_ids            [1, S, head_dim/2]
//!         ▼
//!     for layer in self.layers:
//!         x = HiDreamDecoderLayer.forward(x, cos_sin, mask)
//!         │                          ─ qwen3_vl_transformers.py:893-902
//!         ▼
//!     x = self.norm(x)                                        [B, S_total, H]
//!         │                          ─ qwen3_vl_transformers.py:926
//!         ▼
//!     x_pred = FinalLayer(x)                                  [B, S_total, P*P*C]
//!         │                          ─ qwen3_vl_transformers.py:1525-1526
//!         ▼
//!     return x_pred  -- caller (`pipeline.rs`) will gather
//!                        x_pred[:, vinput_mask, :] for the L gen-image rows.
//! ```
//!
//! ## What's deferred to Phase 2c
//!
//! - Tokenizer / chat-template builder (`pipeline.py:36-78`).
//! - `position_ids` builder (`utils.py:get_rope_index_fix_point`).
//!   Phase 2a's `mrope::build_mrope_positions` covers the simple
//!   single-gen case; multi-image / ref-image builders go in pipeline.rs.
//! - The flow-matching / flash schedulers (`flash_scheduler.py`).
//! - Image-conditioned generation (vision tower for ref-image;
//!   `pixel_values` is intentionally `None` in Phase 2b).
//! - LoRA / FP8 / KV-cache. We run a clean BF16 forward with no caches.
//!
//! ## Edge case parity (`/tmp/hidream_scope_bugs.md`)
//!
//! - **B1**: MRoPE interleave is in `mrope.rs`, table built per forward.
//! - **B5**: `Qwen3VLTextRMSNorm` upcasts to FP32 internally — flame-core's
//!   `rms_norm_bf16` does the same (`fp32 reduce, bf16 store`). Match.
//! - **C7**: noise scaling, scheduler, and CFG live in pipeline.rs;
//!   Phase 2b takes `noise_patches` already scaled (caller's job).
//! - **A6**: token_types {1,2} both map to `vinput_mask = True`
//!   (`pipeline.py:279`). We accept the binarized mask directly.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::nn::Embedding;
use flame_core::norm::RMSNorm;
use flame_core::{DType, Error, Result, Shape, Tensor};
use flame_diffusion::BlockOffloader;

use super::bottleneck_patch_embed::BottleneckPatchEmbed;
use super::decoder::decoder_forward_with_weights_lora;
use super::final_layer::FinalLayer;
use super::lora::LoraRegistry;
use super::mrope::{interleaved_mrope_cos_sin, MRopePositions};
use super::timestep_embedder::TimestepEmbedder;
use super::HiDreamO1Config;

/// Full HiDream-O1 model.
///
/// Per-layer transformer weights stream from pinned host RAM via
/// `BlockOffloader`; resident-shared weights (embed_tokens, final norm,
/// HiDream additions) live on the GPU. This mirrors the
/// `inference-flame/src/models/sensenova_u1.rs` pattern (`SenseNovaU1`).
///
/// Memory budget (BF16 8B variant, see weight_loader.rs):
/// - `embed_tokens`: 152K × 4096 × 2 B  ≈  1.2 GB resident
/// - `norm` + bottleneck_patch_embed + timestep_embedder + final_layer
///   ≈ a few hundred MB resident
/// - 2 active block slots from BlockOffloader: ~2 × 220 MB = 440 MB
/// - All 36 layers' worth of pinned host RAM: ~7.9 GB host (pinned)
pub struct HiDreamO1Model {
    pub config: HiDreamO1Config,
    /// `model.embed_tokens` — text token lookup
    /// (`qwen3_vl_transformers.py:793`).
    pub embed_tokens: Embedding,
    /// `model.norm` — final RMSNorm before any head
    /// (`qwen3_vl_transformers.py:797, 926`).
    pub norm: RMSNorm,
    /// HiDream-only — noisy-patch embedder
    /// (`qwen3_vl_transformers.py:1042`, named `x_embedder`).
    pub bottleneck_patch_embed: BottleneckPatchEmbed,
    /// HiDream-only — timestep MLP (`qwen3_vl_transformers.py:1041`,
    /// named `t_embedder1`).
    pub timestep_embedder: TimestepEmbedder,
    /// HiDream-only — pixel head (`qwen3_vl_transformers.py:1046`,
    /// named `final_layer2`).
    pub final_layer: FinalLayer,
    /// Per-layer transformer weights, streamed from pinned host RAM. Each
    /// `await_block(i)` returns an `Arc<HashMap<short_key, Tensor>>` where
    /// short_key is the trailing portion of the safetensors key (the
    /// `model.language_model.layers.{i}.` prefix is stripped by the
    /// facilitator). Pre-transposed by `prepare_weights` to `[Cin, Cout]`;
    /// the loader un-transposes back to PyTorch `[Cout, Cin]` so
    /// `fused_linear3d_native` works directly.
    pub offloader: BlockOffloader,
    /// Cached for kernel calls.
    device: Arc<flame_core::CudaDevice>,
}

impl HiDreamO1Model {
    /// Allocate the resident-shared modules (random-init `Linear` and
    /// ones-init `RMSNorm`) and stash an externally-built `BlockOffloader`
    /// for the per-layer transformer weights. Phase 2c's loader then calls
    /// `copy_weight_from` / `copy_bias_from` on each resident module to
    /// populate them from the `model.safetensors.index.json`; the offloader
    /// is already pre-loaded with all 36 blocks in pinned host RAM.
    ///
    /// The `_dtype` argument is accepted for API symmetry but ignored: the
    /// checkpoint is BF16 and every flame-core kernel we touch is BF16-only.
    pub fn new(
        config: HiDreamO1Config,
        device: &Arc<flame_core::CudaDevice>,
        offloader: BlockOffloader,
        _dtype: DType,
    ) -> Result<Self> {
        if offloader.block_count() != config.num_layers {
            return Err(Error::InvalidInput(format!(
                "HiDreamO1Model::new: offloader.block_count={} but config.num_layers={}",
                offloader.block_count(),
                config.num_layers
            )));
        }

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, device.clone())?;

        let norm = RMSNorm::new(vec![config.hidden_size], config.rms_norm_eps, true, device.clone())?;

        let bottleneck_patch_embed = BottleneckPatchEmbed::new(&config, device)?;
        let timestep_embedder = TimestepEmbedder::new(&config, device)?;
        let final_layer = FinalLayer::new(&config, device)?;

        Ok(Self {
            config,
            embed_tokens,
            norm,
            bottleneck_patch_embed,
            timestep_embedder,
            final_layer,
            offloader,
            device: device.clone(),
        })
    }

    /// Single denoise-step forward.
    ///
    /// # Arguments
    /// - `input_ids`: `[B, S_text]` I32 — text tokens including the
    ///   `<|tms_token|>` slot AND the L `<|image_pad|>` placeholders that
    ///   reserve space for the noise patches AT THE END of the text portion.
    ///   (Per `pipeline.py:51-54` the gen-image patches sit *inside* the text
    ///   stream as `image_token_id`s — the very first one is replaced by
    ///   `vision_start_token_id` per A2 in `bugs.md`.)
    ///
    ///   Layout reminder (after pipeline.py:54-55, single gen image):
    ///   ```
    ///     [ ... user prompt ... <|tms_token|>
    ///       <|vision_start|>  <|image_pad|>  <|image_pad|>  ...  <|image_pad|> ]
    ///                               ^^^^^^^^^^^^^^^^^^^ L slots
    ///   ```
    ///   But pipeline.py:73 then **strips** the gen-image trailing slots
    ///   from `input_ids` before passing to the model — `input_ids` here
    ///   covers only the prefix, and `noise_patches` are appended after.
    ///   Re-read `pipeline.py:71-75` carefully when wiring Phase 2c.
    ///
    /// - `timestep`: `[B]` BF16/F32, the flow-matching timestep in
    ///   `[0, 1]` (the model rescales by 1000 internally).
    /// - `noise_patches`: `[B, L, P*P*C]` BF16 — current noisy image
    ///   patches (P=32, C=3 → last dim 3072).
    /// - `position_ids_thw`: 3 row-vectors `[u32; S_total]` for
    ///   `(t, h, w)` positions, produced by
    ///   [`super::mrope::build_mrope_positions`].
    /// - `vinput_mask`: `[B, S_total]` BF16 1.0/0.0 — 1.0 at the L
    ///   image-patch positions, 0.0 at text positions
    ///   (`pipeline.py:69`: `vinput_mask = (token_types == 1)`,
    ///   `pipeline.py:279`: with refs, `(token_types==1 | ==2)`).
    ///   The mask drives both the per-row attention pattern (full vs.
    ///   causal) and the caller-side gather of `x_pred`.
    /// - `attention_mask`: an optional override. When `None`, we build
    ///   one from `vinput_mask` (the standard path).
    ///
    /// # Returns
    /// `x_pred : [B, S_total, P*P*C]` BF16. The caller is expected to
    /// `index_select(dim=1, indices=where(vinput_mask))` to take only
    /// the L gen-image rows (matches `pipeline.py:329`:
    /// `x_pred[0, sample['vinput_mask'][0]]`).
    /// LoRA-aware sibling of [`Self::forward`]. When `lora == None` this is
    /// byte-identical to `forward`; when `Some`, every decoder layer's 7 fused
    /// linears route through the registry per
    /// [`super::decoder::decoder_forward_with_weights_lora`]. The same memory /
    /// offloader semantics apply.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_lora(
        &mut self,
        input_ids: &Tensor,
        timestep: &Tensor,
        noise_patches: &Tensor,
        position_ids_thw: &MRopePositions<'_>,
        vinput_mask: &Tensor,
        token_types_bin: &Tensor,
        attention_mask: Option<&Tensor>,
        lora: Option<&LoraRegistry>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            timestep,
            noise_patches,
            position_ids_thw,
            vinput_mask,
            token_types_bin,
            attention_mask,
            lora,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        timestep: &Tensor,
        noise_patches: &Tensor,
        position_ids_thw: &MRopePositions<'_>,
        vinput_mask: &Tensor,
        token_types_bin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            timestep,
            noise_patches,
            position_ids_thw,
            vinput_mask,
            token_types_bin,
            attention_mask,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        timestep: &Tensor,
        noise_patches: &Tensor,
        position_ids_thw: &MRopePositions<'_>,
        vinput_mask: &Tensor,
        token_types_bin: &Tensor,
        attention_mask: Option<&Tensor>,
        lora: Option<&LoraRegistry>,
    ) -> Result<Tensor> {
        // `vinput_mask` is `(token_types == 1)` — image rows only. Currently
        // only the caller uses it (for `gather_image_rows`); kept on the
        // forward signature so signature shape stays parallel to the cache
        // record and so future image-region routing inside the model has it.
        // `token_types_bin` is `(token_types > 0)` — image rows + the TMS token
        // (type=3). Drives `build_mixed_attention_mask` so the TMS row gets
        // FULL attention, matching `qwen3_vl_transformers.py:1501-1502`.
        // The two CANNOT be merged — Python uses them at different sites
        // (`pipeline.py:69` builds `vinput_mask`; `qwen3_vl:1501` reads
        // `token_types_bin`).
        let _ = vinput_mask;
        // Instrumentation gate: HIDREAM_MEM_LOG=1 prints free MiB at each
        // major forward stage and per-layer (last log line before OOM
        // identifies the failing allocation).
        let mem_log = std::env::var("HIDREAM_MEM_LOG").ok().as_deref() == Some("1");
        let log_mem = |label: &str| {
            if mem_log {
                let free = flame_core::cuda::utils::cuda_mem_get_free_mb()
                    .map(|m| format!("{} MiB free", m))
                    .unwrap_or_else(|| "??? MiB free".to_string());
                eprintln!("[hidream_mem] {:32} {}", label, free);
            }
        };
        log_mem("forward.entry");

        // 1) text embeddings: [B, S_text, H]
        let text_emb = self.embed_tokens.forward(input_ids)?;
        log_mem("after.embed_tokens");

        // 2) timestep embedding: [B, H], then scatter into <|tms_token|> rows.
        //    Python (`qwen3_vl_transformers.py:1449-1452`):
        //        tms_mask = (input_ids == tms_token_id)        # [B, S_text]
        //        tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(text_emb)
        //        t_emb_exp   = t_emb.unsqueeze(1).expand_as(text_emb)
        //        text_emb    = where(tms_mask_3d, t_emb_exp, text_emb)
        //
        //    flame-core has `Tensor::where_mask(mask, a, b)` which selects `a`
        //    where mask != 0. We build the [B, S_text, H] expanded mask /
        //    expanded t_emb host-side from input_ids (a tiny CPU op — text
        //    sequence is at most a few thousand tokens).
        let t_emb = self.timestep_embedder.forward_lora(timestep, lora)?; // [B, H]
        log_mem("after.timestep_embed");

        let text_emb_with_t = self.scatter_tms_token(&text_emb, input_ids, &t_emb)?;
        log_mem("after.scatter_tms");

        // 3) patch embedding: [B, L, P*P*C] -> [B, L, H]
        let patch_emb = self.bottleneck_patch_embed.forward_lora(noise_patches, lora)?;
        log_mem("after.patch_embed");

        // 4) concat along seq dim: [B, S_total, H]  with S_total = S_text + L.
        let inputs_embeds = Tensor::cat(&[&text_emb_with_t, &patch_emb], 1)?;
        log_mem("after.cat_inputs");

        let dims = inputs_embeds.shape().dims().to_vec();
        let b = dims[0];
        let s_total = dims[1];

        // 5) MRoPE table: cos/sin shaped [1, S_total, head_dim/2] BF16.
        let head_dim = self.config.head_dim;
        let theta = self.config.rope_theta;
        let mrope_section = self.config.mrope_section;
        let cos_sin = interleaved_mrope_cos_sin(
            position_ids_thw.t,
            position_ids_thw.h,
            position_ids_thw.w,
            head_dim,
            theta,
            mrope_section,
            &self.device,
        )?;
        log_mem("after.cos_sin");

        // 6) attention mask: caller may pre-build (e.g. for CFG batching);
        //    otherwise we synthesize from token_types_bin (NOT vinput_mask —
        //    deep-investigation 2026-05-17 isolated a structural bug here:
        //    Python reads `token_types_bin = (token_types > 0)` which is
        //    True at type=3 (TMS) too; `vinput_mask = (token_types == 1)` is
        //    False at TMS, so using it left the TMS row CAUSAL in Rust where
        //    Python makes it FULL. See
        //    `EriDiffusion-v2/docs/hidream_o1_g0_deep_investigation.md`).
        //
        //    Python (`qwen3_vl_transformers.py:1495-1504`):
        //      causal = full(min, [S, S]); causal = triu(causal, diag=1)  # 0 below+diag, -inf above
        //      gen_positions = token_types[b].bool()   # token_types_bin: type>0
        //      causal[gen_positions, :] = 0            # gen rows attend to ALL
        //
        //    We emit a multiplicative {0, 1} mask the SDPA path expects: 1
        //    where attention is allowed, 0 where blocked. The SDPA path
        //    (`flame_core/sdpa.rs:466-469`) converts to additive -inf
        //    internally.
        let disable_two_pass = std::env::var("HIDREAM_O1_DISABLE_TWO_PASS_ATTN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let use_two_pass = attention_mask.is_none()
            && b == 1
            && !disable_two_pass
            && !flame_core::autograd::AutogradContext::is_recording();
        let two_pass_ar_len = if use_two_pass {
            Some(Self::ar_prefix_len(b, s_total, token_types_bin)?)
        } else {
            None
        };

        let owned_mask;
        let mask_ref = if two_pass_ar_len.is_some() {
            None
        } else {
            Some(match attention_mask {
                Some(m) => m,
                None => {
                    owned_mask =
                        self.build_mixed_attention_mask(b, s_total, token_types_bin)?;
                    &owned_mask
                }
            })
        };
        log_mem("after.attn_policy");

        // 7) decoder loop: 36 layers for 8B, streamed via BlockOffloader.
        //
        // Pattern mirrors `inference-flame/src/models/sensenova_u1.rs`:586-601:
        //  - prefetch block 0 outside the loop;
        //  - per-step: await(i), prefetch(i+1) overlap, un-transpose, forward;
        //  - drop the per-step weights to release the slot before the next
        //    await rolls the active slot.
        //
        // The split-borrow pattern keeps `offloader` borrowed `&mut` while
        // `config`, `embed_tokens`, `norm`, and the head modules stay `&`.
        let total = self.config.num_layers;
        let cfg = self.config.clone();
        let mut hidden = inputs_embeds;
        let offloader = &mut self.offloader;
        offloader
            .prefetch_block(0)
            .map_err(|e| Error::InvalidInput(format!("prefetch block 0: {e}")))?;

        // Instrumentation gate: HIDREAM_DUMP_LAYERS=<path.safetensors> writes
        // the post-residual hidden state at the end of every decoder layer to
        // a safetensors file. Keys: `hidden_layer_{i:02d}` (F32).
        // Used by tests/parity/hidream_o1_g0_per_layer_ref.py + the parity
        // binary's --per-layer-dump mode to validate that per-layer cosine
        // drift is monotonic numerical noise vs a structural jump.
        // Off when env var is absent — zero overhead in production.
        let dump_layers_path: Option<String> =
            std::env::var("HIDREAM_DUMP_LAYERS").ok().filter(|s| !s.is_empty());
        let mut layer_dump: Option<HashMap<String, Tensor>> =
            dump_layers_path.as_ref().map(|_| HashMap::with_capacity(total + 2));
        if let Some(ref mut d) = layer_dump {
            // Also save the input to layer 0 (the "embedding+patch+cat+mask" stack).
            // We save in F32 for safe round-trip vs the Python ref dump (also F32).
            d.insert(
                "hidden_input_layer_00".to_string(),
                hidden.to_dtype(DType::F32)?,
            );
        }

        for i in 0..total {
            // Per-layer log only for the first few layers to keep output small;
            // OOM in this port hits within the first layer at 2048².
            if mem_log && (i < 3 || i == total - 1) {
                log_mem(&format!("layer{:02}.before_await", i));
            }
            let raw = offloader
                .await_block_handle(i)
                .map_err(|e| Error::InvalidInput(format!("await block {i}: {e}")))?;
            if mem_log && (i < 3 || i == total - 1) {
                log_mem(&format!("layer{:02}.after_await", i));
            }
            if i + 1 < total {
                offloader
                    .prefetch_block(i + 1)
                    .map_err(|e| Error::InvalidInput(format!("prefetch block {}: {e}", i + 1)))?;
            }
            let lw = Self::untranspose_block_weights(raw.weights())?;
            if mem_log && (i < 3 || i == total - 1) {
                log_mem(&format!("layer{:02}.after_untranspose", i));
            }
            hidden = decoder_forward_with_weights_lora(
                &cfg,
                i,
                &hidden,
                &cos_sin,
                mask_ref,
                &lw,
                lora,
                two_pass_ar_len,
            )?;
            drop(raw);
            if mem_log && (i < 3 || i == total - 1) {
                log_mem(&format!("layer{:02}.after_forward", i));
            }
            if let Some(ref mut d) = layer_dump {
                d.insert(
                    format!("hidden_layer_{i:02}"),
                    hidden.to_dtype(DType::F32)?,
                );
            }
        }
        log_mem("after.layer_loop");

        // 8) final RMSNorm.
        let hidden = self.norm.forward(&hidden)?;
        if let Some(ref mut d) = layer_dump {
            d.insert(
                "hidden_final_norm".to_string(),
                hidden.to_dtype(DType::F32)?,
            );
        }

        // Flush the dump (best-effort: an instrumentation error must not
        // corrupt the production forward result).
        if let (Some(path), Some(d)) = (dump_layers_path.as_ref(), layer_dump.as_ref()) {
            if let Err(e) =
                flame_core::serialization::save_file(d, std::path::Path::new(path))
            {
                eprintln!("[hidream_dump] WARN failed to write {}: {e}", path);
            } else {
                eprintln!(
                    "[hidream_dump] wrote {} keys to {}",
                    d.len(),
                    path
                );
            }
        }

        // 9) final pixel head — projects EVERY position. Caller filters by
        //    vinput_mask. (Per `pipeline.py:329` the gather is
        //    `x_pred[0, sample['vinput_mask'][0]]`.)
        self.final_layer.forward_lora(&hidden, None, lora)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// `BlockOffloader::prepare_weights` pre-transposes 2D `.weight` tensors
    /// to `[Cin, Cout]` for its internal matmul fast path.
    /// `fused_linear3d_native` expects PyTorch `[Cout, Cin]`, so un-transpose
    /// 2D weights here. Same pattern as
    /// `inference-flame/src/models/sensenova_u1.rs::untranspose_block_weights`.
    fn untranspose_block_weights(
        raw: &HashMap<String, Tensor>,
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

    /// Replace every `<|tms_token|>` row in `text_emb` with `t_emb`.
    ///
    /// Implementation: we scan `input_ids` host-side to build a
    /// `[B, S_text, H]` indicator mask in BF16 (1.0 at tms slots, 0.0
    /// elsewhere). Since `input_ids` is on the device but small, we
    /// pull it back via `to_vec_f32` (it's I32; `to_vec_f32` returns
    /// the int values as f32 per FLAME convention — same trick used by
    /// `qwen3_encoder.rs:663-666`).
    fn scatter_tms_token(
        &self,
        text_emb: &Tensor,
        input_ids: &Tensor,
        t_emb: &Tensor,
    ) -> Result<Tensor> {
        let dims = text_emb.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidOperation(format!(
                "scatter_tms_token: text_emb must be [B,S,H], got {:?}",
                dims
            )));
        }
        let (b, s_text, h) = (dims[0], dims[1], dims[2]);

        let id_dims = input_ids.shape().dims();
        if id_dims.len() != 2 || id_dims[0] != b || id_dims[1] != s_text {
            return Err(Error::InvalidOperation(format!(
                "scatter_tms_token: input_ids shape {:?} doesn't match text_emb [{},{}]",
                id_dims, b, s_text
            )));
        }

        let id_dtype = input_ids.dtype();
        let id_host_f32 = match id_dtype {
            DType::I32 | DType::F32 => input_ids.to_vec_f32()?,
            _ => input_ids.to_dtype(DType::F32)?.to_vec_f32()?,
        };

        let tms = self.config.tms_token_id as f32;
        let mut keep_text_data = vec![1.0f32; b * s_text * h];
        let mut selector_data = vec![0.0f32; b * s_text * b];
        let mut any_hit = false;
        for bi in 0..b {
            for si in 0..s_text {
                if id_host_f32[bi * s_text + si] == tms {
                    let row_off = (bi * s_text + si) * h;
                    for d in 0..h {
                        keep_text_data[row_off + d] = 0.0;
                    }
                    selector_data[(bi * s_text + si) * b + bi] = 1.0;
                    any_hit = true;
                }
            }
        }

        // Fast-path: no tms token in this batch. The Python code still
        // runs the where (it's a no-op), so we mirror that — return
        // text_emb unchanged.
        if !any_hit {
            return Ok(text_emb.clone());
        }

        let keep_text = Tensor::from_vec_dtype(
            keep_text_data,
            Shape::from_dims(&[b * s_text, h]),
            self.device.clone(),
            DType::BF16,
        )?;
        let selector = Tensor::from_vec_dtype(
            selector_data,
            Shape::from_dims(&[b * s_text, b]),
            self.device.clone(),
            DType::BF16,
        )?;

        let t_dims = t_emb.shape().dims();
        if t_dims.len() != 2 || t_dims[0] != b || t_dims[1] != h {
            return Err(Error::InvalidOperation(format!(
                "scatter_tms_token: t_emb shape {:?} should be [{},{}]",
                t_dims, b, h
            )));
        }

        // Use matmul, not where/broadcast, so autograd reaches t_emb and the
        // timestep MLP LoRA adapters. `selector` places each batch row's
        // t_emb only at its <|tms_token|> row; `keep_text` zeros the original
        // embedding at that row.
        let text_flat = text_emb.reshape(&[b * s_text, h])?;
        let text_kept = text_flat.mul(&keep_text)?;
        let t_rows = selector.matmul(t_emb)?;
        text_kept.add(&t_rows)?.reshape(&[b, s_text, h])
    }

    /// Build the **mixed causal/full** attention mask.
    ///
    /// `token_types_bin`: `[B, S_total]` BF16, 1.0 at gen rows (image + TMS),
    /// 0.0 at pure-text rows. This is `(token_types > 0)` per
    /// `qwen3_vl_transformers.py:1501`, NOT `vinput_mask` (which is
    /// `token_types == 1` and excludes the TMS row).
    ///
    /// Output: `[B, 1, S_total, S_total]` BF16 with 1.0 = "attend" and
    /// 0.0 = "block". The SDPA path converts the binary mask to additive
    /// `-inf` internally (`flame_core/sdpa.rs:466-469`).
    ///
    /// Algorithm (mirrors `qwen3_vl_transformers.py:1497-1504`):
    /// ```
    /// for each batch:
    ///   start with strict-lower-tri ones (causal: row i ≥ col j)
    ///   for every row i where token_types_bin[i] == 1:  set whole row i to 1
    /// ```
    /// Effect: text rows attend causally; gen rows (image AND TMS) attend
    /// to everything. (Cross-attention text→image is zero because the image
    /// columns sit AFTER the text rows in S_total layout, so the lower-tri
    /// zero blocks them. Image→text, image→image, and TMS→all are full
    /// bidirectional. ✓ matches Python.)
    fn build_mixed_attention_mask(
        &self,
        b: usize,
        s_total: usize,
        token_types_bin: &Tensor,
    ) -> Result<Tensor> {
        let vmask_dims = token_types_bin.shape().dims();
        if vmask_dims.len() != 2 || vmask_dims[0] != b || vmask_dims[1] != s_total {
            return Err(Error::InvalidOperation(format!(
                "build_mixed_attention_mask: token_types_bin shape {:?} must be [{},{}]",
                vmask_dims, b, s_total
            )));
        }
        let vmask_host = token_types_bin.to_dtype(DType::F32)?.to_vec_f32()?;

        let mut data = vec![0.0f32; b * s_total * s_total];
        for bi in 0..b {
            // Per-row logic.
            for i in 0..s_total {
                let is_image_row = vmask_host[bi * s_total + i] != 0.0;
                let row_off = (bi * s_total + i) * s_total;
                if is_image_row {
                    // Bidirectional attention — open everything.
                    for j in 0..s_total {
                        data[row_off + j] = 1.0;
                    }
                } else {
                    // Causal — j <= i.
                    for j in 0..=i {
                        data[row_off + j] = 1.0;
                    }
                }
            }
        }

        let mask_4d = Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[b, 1, s_total, s_total]),
            self.device.clone(),
            DType::BF16,
        )?;
        Ok(mask_4d)
    }

    fn ar_prefix_len(b: usize, s_total: usize, token_types_bin: &Tensor) -> Result<usize> {
        let vmask_dims = token_types_bin.shape().dims();
        if vmask_dims.len() != 2 || vmask_dims[0] != b || vmask_dims[1] != s_total {
            return Err(Error::InvalidOperation(format!(
                "ar_prefix_len: token_types_bin shape {:?} must be [{},{}]",
                vmask_dims, b, s_total
            )));
        }
        if b != 1 {
            return Err(Error::InvalidOperation(
                "ar_prefix_len currently expects batch size 1".into(),
            ));
        }

        let host = token_types_bin.to_dtype(DType::F32)?.to_vec_f32()?;
        let mut ar_len = 0usize;
        while ar_len < s_total && host[ar_len] == 0.0 {
            ar_len += 1;
        }
        for (idx, value) in host.iter().enumerate().skip(ar_len) {
            if *value == 0.0 {
                return Err(Error::InvalidOperation(format!(
                    "ar_prefix_len: token_types_bin is not prefix-AR at position {}",
                    idx
                )));
            }
        }
        Ok(ar_len)
    }
}
