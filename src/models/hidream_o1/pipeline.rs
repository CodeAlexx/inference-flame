//! `HiDreamO1Pipeline` — token-stream construction + denoise loop + CFG.
//!
//! Reference: `/home/alex/HiDream-O1-Image/models/pipeline.py`
//!
//! ## Phase 2c scope (T2I only — no ref images)
//!
//! Mirrors `pipeline.py:103-393` for the `ref_image_paths is None` branch.
//! The ref-image branch (`pipeline.py:174-289`) is deferred to Phase 2d/e —
//! it requires a Qwen3-VL vision tower and the multi-grid position-id
//! builder, neither of which is in scope for Phase 2c.
//!
//! ## Critical edge cases (`/tmp/hidream_scope_bugs.md`)
//!
//! - **A2**: `vision_tokens[0,0] = vision_start_token_id`; remaining slots
//!   are `image_token_id`. `skip_vision_start_token = [1]` (Python `pipeline.py:51-58`).
//! - **C2/C3**: 28-step Dev uses hardcoded `DEFAULT_TIMESTEPS`; sigmas are
//!   `[t/1000 for t in timesteps] + [0.0]` (handled in `scheduler.rs`).
//! - **C7**: initial latent noise is `noise_scale_start * randn(...)`,
//!   std ≈ 7.5/8.0 (NOT 1.0). CPU generator seeded by `seed + 1`
//!   (`pipeline.py:291-294`).
//! - **C9**: per-step injected noise uses CUDA RNG seeded by `seed + 1`
//!   (`pipeline.py:308-309`).
//! - **C13**: CFG happens on **velocity**, not on `x_pred`
//!   (`pipeline.py:354`).
//! - **D1**: `vinput_mask` gather extracts only image rows from `x_pred`
//!   `[B, S_total, 3072]` (`pipeline.py:329`).
//! - **F3**: `model_output = -v_guided` flip BEFORE the scheduler step
//!   (`pipeline.py:374`).
//! - **G1/G2/G3**: unpatchify uses `(C p1 p2)` order, `(H W)` row-major
//!   flatten, output `[H, W, 3]` u8 (`pipeline.py:391-393`).

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result as AnyResult};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

use super::bottleneck_patch_embed::BottleneckPatchEmbed;
use super::lora::LoraRegistry;
use super::model::HiDreamO1Model;
use super::mrope::{build_mrope_positions, MRopePositions};
use super::scheduler::{HiDreamScheduler, HiDreamSchedulerKind};
use super::HiDreamO1Config;

/// HiDream-O1 generation pipeline (T2I, no ref-images, Phase 2c).
pub struct HiDreamO1Pipeline {
    pub model: HiDreamO1Model,
    pub scheduler: HiDreamScheduler,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: HiDreamO1Config,
    pub device: Arc<CudaDevice>,
    pub dtype: DType,
    /// Backward-compatible debug flag. Current edv2-reference O1 only rounds to
    /// the patch multiple, so this flag no longer changes normal generation.
    pub allow_any_resolution: bool,
    /// Optional LoRA adapters routed through `model.forward_lora`. When
    /// `None`, generation calls `model.forward` (byte-identical to the
    /// pre-M4 inference path). Set via [`Self::set_lora`] after
    /// constructing the pipeline.
    pub lora: Option<LoraRegistry>,
}

/// Legacy predefined resolutions list.
///
/// Older HiDream-O1 scripts snapped requests to these 2048-area presets.
/// Current edv2-reference's O1 pipeline only rounds to the patch multiple; keep
/// the list for tests/docs that still reference the old behavior.
pub const PREDEFINED_RESOLUTIONS: &[(usize, usize)] = &[
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
];

/// Find the closest predefined resolution by aspect ratio
/// (`utils.py:20-30`).
pub fn find_closest_resolution(width: usize, height: usize) -> (usize, usize) {
    let img_ratio = width as f64 / height as f64;
    let mut best = PREDEFINED_RESOLUTIONS[0];
    let mut min_diff = f64::INFINITY;
    for &(w, h) in PREDEFINED_RESOLUTIONS {
        let ratio = w as f64 / h as f64;
        let diff = (ratio - img_ratio).abs();
        if diff < min_diff {
            min_diff = diff;
            best = (w, h);
        }
    }
    best
}

impl HiDreamO1Pipeline {
    /// Construct.
    ///
    /// **Token-id validation** (edge case I1): re-reads the special tokens
    /// from the loaded `tokenizer` and checks them against the bake-in
    /// constants in `config`. Returns `Err` if any mismatch — the model
    /// would silently miss the timestep injection otherwise.
    pub fn new(
        model: HiDreamO1Model,
        scheduler: HiDreamScheduler,
        tokenizer: tokenizers::Tokenizer,
        config: HiDreamO1Config,
        device: Arc<CudaDevice>,
        dtype: DType,
    ) -> AnyResult<Self> {
        // Edge case I1 — verify token IDs match the tokenizer.
        Self::validate_token_id(&tokenizer, "<|tms_token|>", config.tms_token_id)?;
        Self::validate_token_id(&tokenizer, "<|image_pad|>", config.image_token_id)?;
        Self::validate_token_id(
            &tokenizer,
            "<|vision_start|>",
            config.vision_start_token_id,
        )?;
        // boi_token is added externally; just check it can be found.
        Self::validate_token_present(&tokenizer, "<|boi_token|>")?;

        Ok(Self {
            model,
            scheduler,
            tokenizer,
            config,
            device,
            dtype,
            allow_any_resolution: false,
            lora: None,
        })
    }

    /// Backward-compatible no-op for callers that used the old snap bypass.
    pub fn set_allow_any_resolution(&mut self, allow: bool) {
        self.allow_any_resolution = allow;
    }

    /// Attach a LoRA registry (M4). When set, both the cond and uncond
    /// forwards in [`Self::generate`] route through `model.forward_lora`
    /// with `Some(&registry)`; clearing it (via `set_lora(None)`) restores
    /// the no-LoRA forward path. Loaded checkpoints must be PEFT-format
    /// (see [`LoraRegistry::from_safetensors`]).
    pub fn set_lora(&mut self, lora: Option<LoraRegistry>) {
        self.lora = lora;
    }

    fn validate_token_id(
        tokenizer: &tokenizers::Tokenizer,
        token_str: &str,
        expected_id: u32,
    ) -> AnyResult<()> {
        let id = tokenizer
            .token_to_id(token_str)
            .ok_or_else(|| anyhow!("tokenizer missing special token {}", token_str))?;
        if id != expected_id {
            return Err(anyhow!(
                "tokenizer token-id mismatch for {}: bake-in {} vs tokenizer {}",
                token_str,
                expected_id,
                id
            ));
        }
        Ok(())
    }

    fn validate_token_present(tokenizer: &tokenizers::Tokenizer, token_str: &str) -> AnyResult<()> {
        tokenizer
            .token_to_id(token_str)
            .ok_or_else(|| anyhow!("tokenizer missing special token {}", token_str))?;
        Ok(())
    }

    // ─── Chat template (minimal, no Jinja) ─────────────────────────────

    /// Apply the HiDream chat template for a text-only T2I prompt.
    ///
    /// Produces (verbatim, mirroring the Jinja template in
    /// `/home/alex/HiDream-O1-Image-Full-weights/chat_template.json`):
    ///
    /// ```text
    /// <|im_start|>user
    /// {prompt}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    ///
    /// Then `pipeline.py:42-43` appends `<|boi_token|>` + `TIMESTEP_TOKEN_NUM`
    /// `<|tms_token|>` tokens. We do that here as a single template-string
    /// build for token-encoder-friendly output.
    fn apply_chat_template_t2i(prompt: &str) -> String {
        // The Python `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
        // for `messages = [{"role": "user", "content": prompt_string}]` with
        // text-only content produces this exact string (per the Jinja
        // template, lines after `{%- elif messages[0].role == 'system' %}`
        // — for us no system, so we go straight to the user-message branch).
        //
        // Jinja produces: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
        // (note the trailing \n on `assistant\n`).
        let mut s = String::new();
        s.push_str("<|im_start|>user\n");
        s.push_str(prompt);
        s.push_str("<|im_end|>\n");
        s.push_str("<|im_start|>assistant\n");
        // pipeline.py:42-43 — append boi + (TIMESTEP_TOKEN_NUM × tms).
        s.push_str("<|boi_token|>");
        s.push_str("<|tms_token|>"); // TIMESTEP_TOKEN_NUM == 1
        s
    }

    // ─── Token stream + position IDs + masks ───────────────────────────

    /// Build the T2I sample for `prompt` at `(height, width)`.
    /// Returns (input_ids `[B,S_text]` I32, position_ids_thw,
    /// vinput_mask `[B,S_total]` BF16, token_types_bin `[B,S_total]` BF16).
    ///
    /// `token_types_bin = (token_types > 0)` includes the TMS row at
    /// `txt_seq_len - 1` (type=3) in addition to the image rows (type=1).
    /// Required by `model.forward` for the attention mask construction
    /// (matches `qwen3_vl_transformers.py:1501`).
    ///
    /// **Layout**:
    /// - `input_ids` is the text portion ONLY (chat template + boi + tms).
    ///   The L = (h/32)(w/32) image-pad tokens are appended in
    ///   `position_ids_thw` (which is built over the FULL stream) and in
    ///   `vinput_mask` (also full-stream), but the pixel-DiT noise patches
    ///   replace those slots inside `model.forward` (Phase 2b's path).
    /// - `position_ids_thw` covers the full `S_total = S_text + L` stream.
    /// - `vinput_mask` covers the full `S_total` stream, 1.0 at the L
    ///   image-patch slots (`token_types == 1` in Python), 0.0 elsewhere.
    ///
    /// Mirrors `pipeline.py:30-77` (`build_t2i_text_sample`).
    pub fn build_t2i_input(
        &self,
        prompt: &str,
        height: usize,
        width: usize,
    ) -> AnyResult<(Tensor, MRopePositionsOwned, Tensor, Tensor)> {
        if height % self.config.patch_size != 0 || width % self.config.patch_size != 0 {
            return Err(anyhow!(
                "build_t2i_input: H={} W={} must be divisible by patch_size={}",
                height,
                width,
                self.config.patch_size
            ));
        }
        let p = self.config.patch_size;
        let h_patches = height / p;
        let w_patches = width / p;
        let image_len = h_patches * w_patches;

        // 1) Apply chat template + boi + tms.
        let template = Self::apply_chat_template_t2i(prompt);

        // 2) Encode (no special tokens added — they're already in the template).
        // Python uses `tokenizer.encode(template_caption, ..., add_special_tokens=False)`
        // (`pipeline.py:45`).
        let enc = self
            .tokenizer
            .encode(template.as_str(), false)
            .map_err(|e| anyhow!("Tokenize failed: {}", e))?;
        let mut text_ids: Vec<u32> = enc.get_ids().to_vec();
        let txt_seq_len = text_ids.len();

        // 3) Build the FULL stream (text + image_pad slots) for position-id
        //    construction. Slot 0 of the image grid is replaced by
        //    `vision_start_token_id` (edge case A2 / `pipeline.py:51-52`).
        let mut full_ids: Vec<u32> = Vec::with_capacity(txt_seq_len + image_len);
        full_ids.extend_from_slice(&text_ids);
        // First image slot gets vision_start_token_id; the rest are image_token_id.
        full_ids.push(self.config.vision_start_token_id);
        for _ in 1..image_len {
            full_ids.push(self.config.image_token_id);
        }

        let all_seq_len = full_ids.len();
        debug_assert_eq!(all_seq_len, txt_seq_len + image_len);

        // 4) Build MRoPE position ids over the FULL stream (T2I single gen image).
        // skip_vision_start_token = [1] for T2I (`pipeline.py:58`).
        let (t_pos, h_pos, w_pos) = build_mrope_positions(
            &full_ids,
            self.config.image_token_id,
            self.config.video_token_id,
            self.config.vision_start_token_id,
            &[(1, h_patches, w_patches)],
            &[1],
            Some(self.config.fix_point),
        );

        // 5) Build vinput_mask: 1.0 at the L image-patch slots, 0.0 elsewhere.
        //    Python (`pipeline.py:64-69`):
        //      bgn = txt_seq_len - TIMESTEP_TOKEN_NUM  (= txt_seq_len - 1)
        //      token_types[0, bgn : bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
        //      token_types[0, txt_seq_len - 1 : txt_seq_len] = 3
        //      vinput_mask = (token_types == 1)   # bool, 3 doesn't pass
        //
        //    Net: vinput_mask is 1 over the range `[bgn, bgn + image_len + 1)`
        //    EXCEPT slot `txt_seq_len - 1` (the tms token) which is overridden
        //    to type 3 → False in the mask. So the 1's cover:
        //      [bgn .. txt_seq_len - 1)  → empty (length 0)
        //      [txt_seq_len ..  txt_seq_len + image_len)  → the L image slots.
        //    That's exactly the L image positions.
        let mut vinput_mask_data = vec![0.0_f32; all_seq_len];
        for i in txt_seq_len..(txt_seq_len + image_len) {
            vinput_mask_data[i] = 1.0;
        }
        let vinput_mask = Tensor::from_vec_dtype(
            vinput_mask_data,
            Shape::from_dims(&[1, all_seq_len]),
            self.device.clone(),
            DType::BF16,
        )?;

        // 5b) token_types_bin = (token_types > 0). For T2I this is True over
        //     the SAME range as `vinput_mask` PLUS the TMS slot at
        //     `txt_seq_len - 1` (type=3). Used by `model.forward` for the
        //     attention mask (Python's `gen_positions = token_types[b].bool()`
        //     at `qwen3_vl_transformers.py:1501`). The deep-investigation at
        //     `EriDiffusion-v2/docs/hidream_o1_g0_deep_investigation.md`
        //     showed that using `vinput_mask` instead breaks parity at token
        //     22 (the TMS row).
        let mut token_types_bin_data = vec![0.0_f32; all_seq_len];
        // Image rows.
        for i in txt_seq_len..(txt_seq_len + image_len) {
            token_types_bin_data[i] = 1.0;
        }
        // TMS row (last text slot, type=3 in Python, > 0 ⇒ bin = True).
        if txt_seq_len > 0 {
            token_types_bin_data[txt_seq_len - 1] = 1.0;
        }
        let token_types_bin = Tensor::from_vec_dtype(
            token_types_bin_data,
            Shape::from_dims(&[1, all_seq_len]),
            self.device.clone(),
            DType::BF16,
        )?;

        // 6) input_ids is the TEXT portion ONLY for `model.forward`
        //    (Phase 2b's contract: text + tms; the L image slots get
        //    replaced by `BottleneckPatchEmbed(noise_patches)` inside the model).
        let input_ids_i32: Vec<f32> = text_ids.iter().map(|&id| id as f32).collect();
        let input_ids = Tensor::from_vec(
            input_ids_i32,
            Shape::from_dims(&[1, txt_seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;

        // Silence unused variable warnings (text_ids only used through input_ids_i32).
        let _ = &mut text_ids;

        Ok((
            input_ids,
            MRopePositionsOwned {
                t: t_pos,
                h: h_pos,
                w: w_pos,
            },
            vinput_mask,
            token_types_bin,
        ))
    }

    // ─── Generation entrypoint ────────────────────────────────────────

    /// Full T2I generation.
    ///
    /// # Arguments
    /// - `prompt`: positive prompt.
    /// - `negative_prompt`: typically `" "` (single space) for CFG;
    ///   pass `""` to disable CFG entirely (uses single forward).
    ///   Python uses `" "` literally (`pipeline.py:160`).
    /// - `height`, `width`: must be in `PREDEFINED_RESOLUTIONS`. If not,
    ///   they're snapped to the closest aspect-ratio match (edge case D3).
    /// - `seed`: noise seed. Initial latent uses `seed + 1`; per-step
    ///   noise uses `seed + 1` as well (`pipeline.py:293, 308-309`).
    /// - `guidance_scale`: CFG strength. Dev default 0.0 → CFG disabled.
    ///   Full default 5.0.
    ///
    /// # Returns
    /// Final image `[1, 3, H, W]` BF16 in `[-1, 1]` range. Caller can
    /// `save_png` to write to disk.
    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        height: usize,
        width: usize,
        seed: u64,
        guidance_scale: f32,
    ) -> AnyResult<Tensor> {
        let p = self.config.patch_size;
        // 1) Current edv2-reference O1 rounds to the patch multiple only.
        let rounded_width = (width / p) * p;
        let rounded_height = (height / p) * p;
        if rounded_width == 0 || rounded_height == 0 {
            return Err(anyhow!(
                "HiDream-O1 resolution must be at least {}x{}, got {}x{}",
                p, p, width, height
            ));
        }
        if rounded_width != width || rounded_height != height {
            eprintln!(
                "[hidream_o1] Resolution rounded from {}x{} to {}x{}",
                width, height, rounded_width, rounded_height
            );
        }
        let (height, width) = (rounded_height, rounded_width);
        let h_patches = height / p;
        let w_patches = width / p;

        // 2) Build cond + (optional) uncond samples.
        let do_cfg = guidance_scale > 1.0;
        let (cond_input_ids, cond_pos, cond_vmask, cond_token_types_bin) =
            self.build_t2i_input(prompt, height, width)
                .map_err(|e| anyhow!("build_t2i_input(cond): {}", e))?;

        let uncond = if do_cfg {
            // Python uses literal `" "` (single space) for uncond
            // (`pipeline.py:160`); empty string and " " produce different
            // token streams. Match the Python contract.
            let prompt_uncond = if negative_prompt.is_empty() { " " } else { negative_prompt };
            let (i, p, v, ttb) = self
                .build_t2i_input(prompt_uncond, height, width)
                .map_err(|e| anyhow!("build_t2i_input(uncond): {}", e))?;
            Some((i, p, v, ttb))
        } else {
            None
        };

        // 3) Initial noise — `noise_scale_start * randn(B,3,H,W)` (edge case C7).
        //    For Dev defaults (`inference.py:33-34`) noise_scale_start = 7.5.
        //    For Full (`pipeline.py:117-118`) NOISE_SCALE = 8.0.
        let noise_scale_start = match self.scheduler.kind() {
            HiDreamSchedulerKind::FlashStochastic => 7.5_f32, // Dev default (inference.py:33)
            HiDreamSchedulerKind::FlowMatchEuler => 8.0_f32,  // Full default (pipeline.py:14, 117)
            // UniPC: deterministic; the initial-noise std doesn't feed back into
            // a per-step injection. Use the 50-step Full default for the initial
            // latent magnitude. This is an AGENT-DEFAULT choice.
            HiDreamSchedulerKind::UniPc => 8.0_f32,
        };
        let noise_scale_end = noise_scale_start; // both endpoints equal in default cfg

        let z_init = self.draw_initial_noise(height, width, seed, noise_scale_start)?;
        // patchify: [1, 3, H, W] -> [1, L, 3*32*32]
        let mut z = BottleneckPatchEmbed::patchify(&z_init, p)?
            .to_dtype(self.dtype)?;

        // 4) Build per-step noise-scale schedule (linear interpolation).
        //    Mirror `pipeline.py:300-306`.
        let num_steps = self.scheduler.num_inference_steps();
        let noise_scale_schedule: Vec<f32> = if num_steps > 1 {
            (0..num_steps)
                .map(|i| {
                    noise_scale_start
                        + (noise_scale_end - noise_scale_start) * (i as f32)
                            / ((num_steps - 1) as f32)
                })
                .collect()
        } else {
            vec![noise_scale_start]
        };

        // 5) Per-step noise RNG. Python uses CUDA-side RNG seeded by `seed+1`
        //    (`pipeline.py:308-309`). We emulate with `StdRng::seed_from_u64(seed+1)`
        //    on the host — bit-identical CUDA RNG would require a flame-core
        //    randn that we don't have; this matches existing inference-flame
        //    binaries (e.g. `sd3_lora_infer.rs:350`).
        use rand::SeedableRng;
        let mut step_rng = rand::rngs::StdRng::seed_from_u64(seed + 1);

        // For Dev: noise_clip_std default 2.5 (inference.py:35).
        // FlowMatch / UniPC: no per-step noise injection → noise_clip_std unused.
        let noise_clip_std = match self.scheduler.kind() {
            HiDreamSchedulerKind::FlashStochastic => 2.5_f32,
            HiDreamSchedulerKind::FlowMatchEuler => 0.0_f32,
            HiDreamSchedulerKind::UniPc => 0.0_f32,
        };

        // 6) Denoise loop. Mirror `pipeline.py:343-388`.
        for step_idx in 0..num_steps {
            let step_start = std::time::Instant::now();
            let step_t = self.scheduler.timesteps()[step_idx];
            let t_pixeldit = 1.0_f32 - step_t / 1000.0_f32;
            let sigma_clamped = (step_t / 1000.0_f32).max(0.001_f32);

            // Run cond forward.
            let pos_thw = MRopePositions {
                t: &cond_pos.t,
                h: &cond_pos.h,
                w: &cond_pos.w,
            };
            let t_tensor = Tensor::from_vec_dtype(
                vec![t_pixeldit],
                Shape::from_dims(&[1]),
                self.device.clone(),
                DType::BF16,
            )?;
            let x_pred_full = self.model.forward_lora(
                &cond_input_ids,
                &t_tensor,
                &z,
                &pos_thw,
                &cond_vmask,
                &cond_token_types_bin,
                None,
                self.lora.as_ref(),
            )?;
            // Gather rows where vinput_mask == 1 (edge case D1 / pipeline.py:329).
            let x_pred_cond = self.gather_image_rows(&x_pred_full, &cond_vmask)?;

            // v_cond = (x_pred - z) / sigma  (FP32 per pipeline.py:349)
            let v_cond = self.compute_velocity(&x_pred_cond, &z, sigma_clamped)?;

            let v_guided = if let Some((u_input_ids, u_pos, u_vmask, u_token_types_bin)) = &uncond {
                let u_pos_thw = MRopePositions {
                    t: &u_pos.t,
                    h: &u_pos.h,
                    w: &u_pos.w,
                };
                let x_pred_full_u = self.model.forward_lora(
                    u_input_ids,
                    &t_tensor,
                    &z,
                    &u_pos_thw,
                    u_vmask,
                    u_token_types_bin,
                    None,
                    self.lora.as_ref(),
                )?;
                let x_pred_uncond = self.gather_image_rows(&x_pred_full_u, u_vmask)?;
                let v_uncond = self.compute_velocity(&x_pred_uncond, &z, sigma_clamped)?;

                // v_guided = v_uncond + s * (v_cond - v_uncond)  (pipeline.py:354)
                let diff = v_cond.sub(&v_uncond)?;
                let scaled = diff.mul_scalar(guidance_scale)?;
                v_uncond.add(&scaled)?
            } else {
                v_cond
            };

            // Edge case F3 / pipeline.py:374 — flip sign before scheduler.
            let model_output = v_guided.mul_scalar(-1.0)?;

            // Draw per-step noise (matches model_output shape).
            // Only stochastic schedulers need per-step noise. FlowMatchEuler
            // and UniPC are deterministic — draw nothing and pass None to the
            // unified step().
            let noise_for_step = if self.scheduler.kind().needs_step_noise() {
                Some(self.draw_step_noise(model_output.shape(), &mut step_rng)?)
            } else {
                None
            };

            // Scheduler step.
            let s_noise = noise_scale_schedule[step_idx];
            z = self.scheduler.step(
                &model_output,
                step_idx,
                &z,
                noise_for_step.as_ref(),
                s_noise,
                noise_clip_std,
                &self.device,
            )?;
            log::info!(
                "[hidream_o1] denoise step {}/{} t={:.3} sigma={:.6} done in {:.2}s",
                step_idx + 1,
                num_steps,
                step_t,
                sigma_clamped,
                step_start.elapsed().as_secs_f64()
            );
        }

        // 7) Unpatchify → [1, 3, H, W] in [-1, 1] range.
        // Python (`pipeline.py:390-391`) computes `(z + 1) / 2` BEFORE
        // unpatchify and then a final clip+round to u8. We instead return
        // `z` in [-1, 1] (still patchified through unpatchify) so the caller
        // can do `save_png` (which handles the [-1,1] → u8 conversion).
        BottleneckPatchEmbed::unpatchify(&z, h_patches, w_patches, p).map_err(|e| anyhow!(e))
    }

    // ─── Helpers ──────────────────────────────────────────────────────

    /// Draw initial noise `[1, 3, H, W]` BF16 with std `noise_scale_start`.
    /// Seeded by `seed + 1` per `pipeline.py:293`.
    fn draw_initial_noise(
        &self,
        height: usize,
        width: usize,
        seed: u64,
        noise_scale_start: f32,
    ) -> Result<Tensor> {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1);
        let numel = 1 * 3 * height * width;
        let mut data = Vec::with_capacity(numel);
        // Box-Muller, matching sd3_lora_infer.rs:352-364 pattern.
        let pairs = numel / 2;
        for _ in 0..pairs {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt() * noise_scale_start;
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(r * theta.cos());
            data.push(r * theta.sin());
        }
        if numel % 2 == 1 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * noise_scale_start);
        }
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[1, 3, height, width]),
            self.device.clone(),
            self.dtype,
        )
    }

    /// Draw per-step noise of `shape` using the shared StdRng state.
    /// Standard normal (no scaling — `s_noise` is applied inside scheduler step).
    fn draw_step_noise(&self, shape: &Shape, rng: &mut rand::rngs::StdRng) -> Result<Tensor> {
        use rand::Rng;
        let numel = shape.elem_count();
        let mut data = Vec::with_capacity(numel);
        let pairs = numel / 2;
        for _ in 0..pairs {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(r * theta.cos());
            data.push(r * theta.sin());
        }
        if numel % 2 == 1 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }
        Tensor::from_vec_dtype(data, shape.clone(), self.device.clone(), DType::F32)
    }

    /// Gather the `L` image rows from `x_pred [B, S_total, 3072]` using
    /// `vinput_mask [B, S_total]` (1.0 at image rows). Output `[B, L, 3072]`.
    /// Edge case D1 / `pipeline.py:329`.
    ///
    /// Implementation: pull the mask to host, build an I32 indices tensor,
    /// then `index_select` along dim 1 via permute+slice. Since the patches
    /// in the T2I builder are CONTIGUOUS at the tail (last `image_len` rows),
    /// we can just `narrow(dim=1, start=S_text, length=L)`.
    fn gather_image_rows(&self, x_pred: &Tensor, vinput_mask: &Tensor) -> Result<Tensor> {
        let dims = x_pred.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "gather_image_rows: x_pred must be [B,S,D], got {:?}",
                dims
            )));
        }

        // Find the contiguous run of 1.0s in vinput_mask. For T2I-build they
        // are always at the tail.
        let mask_host = vinput_mask.to_dtype(DType::F32)?.to_vec_f32()?;
        let s_total = dims[1];
        // Locate first non-zero and last non-zero on row 0 (B=1 typical).
        let mut first = None;
        let mut last = None;
        for i in 0..s_total {
            if mask_host[i] != 0.0 {
                if first.is_none() {
                    first = Some(i);
                }
                last = Some(i);
            }
        }
        let (first, last) = match (first, last) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "gather_image_rows: vinput_mask is all zeros".into(),
                ))
            }
        };
        // Sanity — confirm contiguous.
        for i in first..=last {
            if mask_host[i] == 0.0 {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "gather_image_rows: vinput_mask is non-contiguous (zero at slot {} inside run {}-{})",
                    i, first, last
                )));
            }
        }
        let len = last - first + 1;
        x_pred.narrow(1, first, len)
    }

    /// Compute velocity `v = (x_pred - z) / sigma` in FP32.
    fn compute_velocity(&self, x_pred: &Tensor, z: &Tensor, sigma: f32) -> Result<Tensor> {
        let x_f = x_pred.to_dtype(DType::F32)?;
        let z_f = z.to_dtype(DType::F32)?;
        let diff = x_f.sub(&z_f)?;
        diff.mul_scalar(1.0 / sigma)
    }

    // ─── PNG writer ───────────────────────────────────────────────────

    /// Save a `[1, 3, H, W]` BF16 (or F32) tensor in `[-1, 1]` range as PNG.
    ///
    /// Mirrors `pipeline.py:390-393`:
    /// ```python
    /// img = (z + 1) / 2
    /// arr = round(clip(img * 255, 0, 255)).astype(uint8)  # [H, W, 3]
    /// Image.fromarray(arr).convert("RGB").save(path)
    /// ```
    pub fn save_png(image: &Tensor, path: &Path) -> AnyResult<()> {
        let dims = image.shape().dims().to_vec();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(anyhow!(
                "save_png: expected [1,3,H,W] image, got {:?}",
                dims
            ));
        }
        let (_, _c, h_out, w_out) = (dims[0], dims[1], dims[2], dims[3]);

        let img_f32 = image.to_dtype(DType::F32)?;
        let data = img_f32.to_vec_f32()?;

        // Layout: [1, 3, H, W] row-major → for each c, then row, then col.
        // Output: [H, W, 3] u8 with `round(clip((x+1)/2 * 255, 0, 255))`.
        let mut pixels = vec![0u8; h_out * w_out * 3];
        for y in 0..h_out {
            for x in 0..w_out {
                for c in 0..3 {
                    let idx = c * h_out * w_out + y * w_out + x;
                    let v = data[idx];
                    let img01 = (v + 1.0) * 0.5;
                    let u = (img01 * 255.0).clamp(0.0, 255.0).round() as u8;
                    pixels[(y * w_out + x) * 3 + c] = u;
                }
            }
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        image::RgbImage::from_raw(w_out as u32, h_out as u32, pixels)
            .ok_or_else(|| anyhow!("save_png: RgbImage::from_raw failed"))?
            .save(path)?;
        Ok(())
    }
}

/// Owned variant of [`MRopePositions`] used to ferry positions across
/// pipeline boundaries. Builds a borrow-friendly `MRopePositions<'_>`
/// view via `.as_view()`.
pub struct MRopePositionsOwned {
    pub t: Vec<u32>,
    pub h: Vec<u32>,
    pub w: Vec<u32>,
}

impl MRopePositionsOwned {
    pub fn as_view(&self) -> MRopePositions<'_> {
        MRopePositions {
            t: &self.t,
            h: &self.h,
            w: &self.w,
        }
    }
}
