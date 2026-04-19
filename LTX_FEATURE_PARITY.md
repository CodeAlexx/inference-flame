# LTX-Video Feature Parity Reference

Sources: https://github.com/Lightricks/LTX-Video (main), https://github.com/Lightricks/LTX-Video-Trainer (main), https://github.com/Lightricks/ComfyUI-LTXVideo (master), HF weights `Lightricks/LTX-Video` + IC-LoRA repos.

Line numbers reference the files fetched from `main` as of 2026-04-19. Paths are repo-relative; prepend the repo URL for raw access.

---

## 1. Generation modes

LTX-Video has ONE top-level class — `LTXVideoPipeline` in `ltx_video/pipelines/pipeline_ltx_video.py`. The different "modes" are all expressed as combinations of two inputs to `__call__`:

- `media_items: Optional[torch.Tensor]` — the vid2vid source (whole sequence, to be noised back to some partial timestep)
- `conditioning_items: Optional[List[ConditioningItem]]` — frame/keyframe anchors injected as hard-conditioned latents

`ConditioningItem` (dataclass at pipeline_ltx_video.py:193-210):
```
media_item:           torch.Tensor   # (b, 3, f, h, w), float in [-1, 1]
media_frame_number:   int            # start frame in output video (must be % 8 == 0 unless 0)
conditioning_strength: float         # 1.0 = hard conditioning
media_x, media_y:     Optional[int]  # spatial placement (only allowed if frame_number == 0)
```

The `__call__` dispatcher is at pipeline_ltx_video.py:753-795 — `height, width, num_frames, frame_rate` are mandatory; `prompt`, `negative_prompt`, `num_inference_steps=20`, `guidance_scale=4.5`, `stg_scale=1.0`, `rescaling_scale=0.7`, `skip_layer_strategy`, `skip_block_list`, `stochastic_sampling=False`, `conditioning_items`, `media_items`, `image_cond_noise_scale`, `decode_timestep`, `decode_noise_scale`, `tone_map_compression_ratio` are the key optional knobs.

| Mode | How it's invoked | media_items | conditioning_items | skip_initial_inference_steps | Notes |
|---|---|---|---|---|---|
| **T2V** | only `prompt`; `is_video=True` | None | None | 0 | Pure noise start. `media_item = None` in `ltx_video/inference.py:522-529`. |
| **I2V** | `--conditioning_media_paths img.png --conditioning_start_frames 0` | None | `[ConditioningItem(img_tensor, 0, strength)]` | 0 | Single-frame cond at frame 0. See `prepare_conditioning` pipeline_ltx_video.py:1383-1587. |
| **Multi-keyframe** | N media paths + N `conditioning_start_frames` | None | `[ConditioningItem(img_i, frame_i, strength_i)]` | 0 | Each item gets patchified and prepended as extra tokens; see `_get_latent_spatial_position` (:1605-1650) for frame==0 and `_handle_non_first_conditioning_sequence` (:1652-1726) for middle/end frames. |
| **Video extension forward** | `--conditioning_media_paths prev_clip.mp4 --conditioning_start_frames 0` | None | single item with full-length source clip at frame 0 | 0 | Reuses multi-keyframe machinery; source encoded then lerped with init noise at strength. |
| **Video extension backward** | `--conditioning_media_paths next_clip.mp4 --conditioning_start_frames N` where N > 0 lands the clip at the END of the output | None | single item at nonzero frame (8-aligned) | 0 | Goes through `_handle_non_first_conditioning_sequence` branch with `prefix_latents_mode="concat"` (default). |
| **V2V (noise a full video)** | `--input_media_path video.mp4` with enough noise timesteps skipped | full video tensor | optional | `skip_initial_inference_steps > 0` (required when `media_items` given; assert at pipeline_ltx_video.py:933-940) | Encodes the full source, noises to `timesteps[0]`, then denoises only the latter part of the schedule. `prepare_latents` :623-701 handles the lerp `t*noise + (1-t)*latents`. |
| **V2V via IC-LoRA** | load IC-LoRA (canny/depth/pose/detailer), run T2V as normal | None (or media_items) | None | 0 | The reference video goes through preprocessing (Canny/Depth/Pose estimator) EXTERNALLY, then is used as the IC-LoRA "reference_video" (see §3). The main LTX-Video repo's `inference.py` does NOT expose this path; it's implemented in the LTX-Video-Trainer pipeline (`src/ltxv_trainer/ltxv_pipeline.py:825` `reference_video=` arg) and in ComfyUI-LTXVideo's `LTXAddVideoICLoRAGuide` node (`iclora.py:14-254`). |
| **Multi-scale (distilled/dev default)** | `pipeline_type: multi-scale` in yaml + `spatial_upscaler_model_path` | as above | as above | — | Wraps the base pipeline in `LTXMultiScalePipeline` (pipeline_ltx_video.py:1821-1895). First pass at `downscale_factor`; upsample latents via `LatentUpsampler`; AdaIN-re-normalize; second pass at 2× (see §6). |

The top-level orchestrator is `infer(config: InferenceConfig)` in `ltx_video/inference.py:389-634`. The `InferenceConfig` dataclass is at `ltx_video/inference.py:316-386`.

Latent shape invariants (enforced in `ltx_video/inference.py:461-463`):
- `height_padded = ((height - 1) // 32 + 1) * 32`
- `width_padded = ((width - 1) // 32 + 1) * 32`
- `num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1`

Patch size 1 (SymmetricPatchifier with `patch_size=1`, inference.py:242) → tokens per latent frame = `H_lat * W_lat`. VAE scale: `vae_scale_factor=32`, `video_scale_factor=8`.

---

## 2. LoRA key formats and types

LoRA loading is NOT implemented in the base LTX-Video pipeline class. It is applied at the ComfyUI layer (`LTXICLoRALoaderModelOnly` in ComfyUI-LTXVideo/iclora.py:466-520) via `comfy.sd.load_lora_for_models`, and in diffusers-format pipelines via `LTXVideoLoraLoaderMixin` (imported in `src/ltxv_trainer/ltxv_pipeline.py:25`).

Target module list is stabilized by the trainer (see `configs/ltxv_13b_lora_template.yaml`): `to_k, to_q, to_v, to_out.0, ff.net.0.proj, ff.net.2`. IC-LoRA default adds the same set. Trainer uses peft's `LoraConfig`.

### Key prefixes (measured from actual .safetensors headers)

Two coexisting naming schemes for the same weights:

- **ComfyUI / "native" prefix** — `diffusion_model.` + transformer submodule path
- **Diffusers prefix** — `transformer.` + the same path

Each linear gets `lora_A.weight` (down-projection, shape `[rank, in_features]`) and `lora_B.weight` (up-projection, shape `[out_features, rank]`). Bias is not produced by default.

Example (verified from canny IC-LoRA safetensors header):
```
# ComfyUI variant (ltxv-097-ic-lora-canny-control-comfyui.safetensors)
diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight   [8, 4096]  BF16
diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight   [4096, 8]  BF16

# Diffusers variant (ltxv-097-ic-lora-canny-control-diffusers.safetensors)
transformer.transformer_blocks.0.attn1.to_q.lora_A.weight       [8, 4096]  BF16
transformer.transformer_blocks.0.attn1.to_q.lora_B.weight       [4096, 8]  BF16
```

Standard strength application: `W_eff = W + (alpha/rank) * lora_B @ lora_A * strength_model`. The peft convention used by the trainer stores `alpha == rank` (see `ltxv_13b_lora_template.yaml:14-15`: `rank: [RANK], alpha: [RANK]`), giving a default effective scale of `1.0 * strength_model`.

Multiple LoRAs stack additively: ComfyUI wires multiple `LTXICLoRALoaderModelOnly` nodes in series, each calling `load_lora_for_models(model, None, lora, strength_model, 0)` on the mutating model. Nothing clamps total strength.

### Per-LoRA-type inventory

All measured from actual safetensors headers at HuggingFace on 2026-04-19.

| Type | File pattern | HF repo | Rank | Target modules | Total tensors | `reference_downscale_factor` metadata? |
|---|---|---|---|---|---|---|
| **Distilled LoRA128** | `ltxv-13b-0.9.7-distilled-lora128.safetensors` | `Lightricks/LTX-Video` | 128 | EVERY linear in the transformer (`attn1.*`, `attn2.*`, `ff.net.0.proj`, `ff.net.2`, plus `adaln_single.*`, `caption_projection.*`, `patchify_proj`, `proj_out`) | 974 | `__metadata__: {}` |
| **IC-LoRA Canny** | `ltxv-097-ic-lora-canny-control-{comfyui,diffusers}.safetensors` | `Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7` | 8 | attn1+attn2 (q/k/v/out), ff.net.0.proj, ff.net.2 | 960 (= 48 blocks × 10 modules × 2) | `__metadata__: {}` (empty — defaults to 1.0, see ComfyUI-LTXVideo/iclora.py:504-512) |
| **IC-LoRA Depth** | `ltxv-097-ic-lora-depth-control-{comfyui,diffusers}.safetensors` | `Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7` | 8 | same as Canny | 960 | `__metadata__: {}` |
| **IC-LoRA Pose** | `ltxv-097-ic-lora-pose-control-{comfyui,diffusers}.safetensors` | `Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7` | 24 | attn1+attn2 only (NO ff) | 768 (= 48 × 8 × 2) | `__metadata__: {}` |
| **IC-LoRA Detailer** | `ltxv-098-ic-lora-detailer-{comfyui,diffusers}.safetensors` | `Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8` | 128 | attn1+attn2 (q/k/v/out), ff.net.0.proj, ff.net.2 | 960 | `__metadata__: {}` |
| **Style / Effect LoRAs** (user-trained, e.g. cakeify/squish) | trainer output | user-created | 32 / 64 / 128 typical | `to_k, to_q, to_v, to_out.0` (template default) | 4 modules × 48 blocks × 2 | Written by trainer if IC-LoRA |

**IMPORTANT about `reference_downscale_factor`:** The loader reads it from `__metadata__["reference_downscale_factor"]` (ComfyUI-LTXVideo/iclora.py:506). If absent or unparseable, it defaults to `1.0`. Of the four public Lightricks IC-LoRAs I inspected, **none currently embed this key** — they all have empty `__metadata__`. The mechanism exists so user-trained IC-LoRAs at reduced grids (e.g. the detailer trained at 1/3 resolution) can declare themselves. Treat "absent" as "1.0 = same resolution as target".

Strength application details (ComfyUI-LTXVideo/iclora.py:514-520):
- `strength_model == 0` → return the base model unchanged (used to extract metadata without applying the LoRA — there's a different loader `LTXICLoRAMetadataOnly` pattern).
- else: `comfy.sd.load_lora_for_models(model, clip=None, lora, strength_model=X, strength_clip=0)`. Clip is always 0 because the IC-LoRAs don't patch the T5 text encoder.

---

## 3. IC-LoRA reference conditioning

IC-LoRA is **not** a separate forward path. The LoRA weights themselves are what makes the concatenated-reference trick work. At inference:

1. **Reference video preprocessing** happens OUTSIDE the pipeline. Lightricks provides no built-in estimators; the user is expected to run Canny/Depth/Pose on their source video and feed the resulting video as the IC-LoRA reference.
   - For Canny, see `LTX-Video-Trainer/scripts/compute_condition.py:43-81`: `cv2.Canny(image_np, threshold1=100, threshold2=200)`, applied per-frame after `TF.rgb_to_grayscale`, then broadcast to 3 channels. Input tensor normalized to [0,1] before Canny.
   - For Depth/Pose, no first-party script is provided. Users use MiDaS/DPT, OpenPose/DWPose externally.
   - The detailer LoRA's "reference" is the *up-sampled low-resolution output* of a prior generation; the LoRA learns to add high-frequency details. No external estimator needed.
2. **Encode reference to latents** via the same `CausalVideoAutoencoder` used for targets (`src/ltxv_trainer/ltxv_pipeline.py:1074-1125`).
   - Resize with center crop to target `(H, W)`, possibly divided by `latent_downscale_factor` (ComfyUI-LTXVideo/iclora.py:121-124).
   - Scale pixel values from `[0,1]` to `[-1,1]`.
   - Trim to `(K * temporal_compression_ratio) + 1` frames (i.e. latent-frame-aligned).
   - Encode → normalize per-channel (`_normalize_latents` using `latents_mean, latents_std`).
   - Pack into `(B, seq, C)` patchified form.
3. **Concatenate at the front of the sequence** (src/ltxv_trainer/ltxv_pipeline.py:1158):
   ```python
   latents = torch.cat([reference_latents, latents], dim=1)  # (B, ref_seq + tgt_seq, C)
   ```
   Video coordinates for the reference start at `frame_index=0`, identical to the target coordinate grid for frames `[0..ref_latent_frames-1]`. Both occupy the *same* positional embedding range; the positional-embedding collision is what teaches the trained LoRA to treat the reference tokens as "context" rather than target prefix.
4. **Conditioning mask = 1.0** for the entire reference segment (src/ltxv_trainer/ltxv_pipeline.py:1166-1175) and 0 for the target. The timestep is then forced to `min(t, (1 - conditioning_mask) * 1000)` (src/ltxv_trainer/ltxv_pipeline.py:1240) — so reference tokens always have timestep 0 (clean / fully denoised) and are never stepped by the scheduler.
5. **Transformer forward** sees a single `hidden_states` tensor of length `ref_seq + tgt_seq`. Attention is unmasked between the two halves — this is how the reference "guides" the target. The LoRA adapters on `attn1` (self-attention) are what actually inject the conditioning behavior.
6. **Output extraction**: after denoising, only the target half is decoded (`latents[:, ref_seq:]`). No explicit crop code is needed in the upstream pipeline because the VAE is only run on the target latents; in ComfyUI this happens implicitly because the latent tensor handed back has the reference tokens already stripped.

Training is the mirror image: `ReferenceVideoTrainingStrategy.prepare_batch` at `src/ltxv_trainer/training_strategies.py:289-423`. Loss is computed only on the target half (`:425-439`): `target_pred = model_pred[:, -target_seq_len:]`.

`latent_downscale_factor` (ComfyUI-LTXVideo/iclora.py:58-65 and 196-212) is a *spatial* downscale applied only to the IC-LoRA guide before encoding, then "dilated" back (nearest-neighbor expansion in latent space) to match the target grid. Used by small-grid detailer-style LoRAs. The `LTXVDilateLatent` node is in `latents.py` — a latent-space nearest-neighbor upsampler.

**No cross-attention is involved** in IC-LoRA. No separate path. No MoE. Just: concat tokens at the start, mask them as clean conditioning, let self-attention flow.

---

## 4. Guidance / sampling modes

All guidance lives in the denoising loop at `ltx_video/pipelines/pipeline_ltx_video.py:1114-1290`.

### CFG (classifier-free guidance)

Formula at pipeline_ltx_video.py:1242-1244:
```python
noise_pred = noise_pred_uncond + guidance_scale[i] * (noise_pred_text - noise_pred_uncond)
```
- Default `guidance_scale = 4.5` (pipeline_ltx_video.py:766).
- README-recommended 3.0–3.5 (README.md "Guidance Scale: 3-3.5").
- `guidance_scale` may be a list (time-varying); mapped to timesteps via `guidance_timesteps` (pipeline_ltx_video.py:958-977).
- **Distilled models set `guidance_scale=1`** (configs/ltxv-13b-0.9.8-distilled.yaml) → CFG disabled, only one forward pass per step.

Optional **CFG-star rescale** (`cfg_star_rescale=True`, pipeline_ltx_video.py:1227-1240): projects the uncond noise onto the cond noise before the add-sub step:
```
alpha = dot(eps_text, eps_uncond) / ||eps_uncond||^2
eps_uncond ← alpha * eps_uncond
```
The dev configs enable this (`cfg_star_rescale: true`).

### STG (Spatiotemporal Guidance, "skip-layer" perturbation)

Third forward pass with selected transformer blocks skipped on the positive prompt:
```python
noise_pred = noise_pred + stg_scale[i] * (noise_pred_text - noise_pred_text_perturb)
```
(pipeline_ltx_video.py:1247-1250).

Skip strategies (ltx_video/utils/skip_layer_strategy.py:4-8): `AttentionSkip`, `AttentionValues`, `Residual`, `TransformerBlock`. The `stg_mode` yaml field accepts: `"stg_av" | "attention_values" | "stg_as" | "attention_skip" | "stg_r" | "residual" | "stg_t" | "transformer_block"` (ltx_video/inference.py:548-557).

Skip mechanics:
- `AttentionValues` (default): after attention, blend `hidden_states_a` (computed attention output) with `value_for_stg` (the pre-attention V projection) using the per-layer skip mask (attention.py:1078-1084). This replaces the attention output with the value tensor for selected blocks on the perturb pass.
- `AttentionSkip`: blend `hidden_states_a` with the pre-attention `hidden_states` input (attention.py:1071-1077).
- `Residual`: keep attention output but set `hidden_states += residual * skip_mask` (i.e. drop the residual on non-selected blocks — but conditional on `attn.residual_connection`) (attention.py:1097-1110).
- `TransformerBlock`: blend the post-block state back to the pre-block state (attention.py:312-319).

Skip mask is built by `Transformer3DModel.create_skip_layer_mask` (transformer3d.py:173-188):
```python
mask = torch.ones((num_layers, batch_size * num_conds))
for block_idx in skip_block_list:
    mask[block_idx, ptb_index::num_conds] = 0
```
i.e. only the perturb slot within the batch (last chunk) has its mask zeroed at the skipped blocks.

`skip_block_list` is per-timestep. For ltxv-13b-0.9.8-dev the list walks from `[]` (no skip at t=1.0) → `[11,25,35,39]` → `[22,35,39]` → `[28]` → `[28]` → `[28]` → `[28]` across 7 guidance windows (configs/ltxv-13b-0.9.8-dev.yaml lines 27-29 `first_pass`).

Optional STG **rescaling** (`rescaling_scale`, default 0.7, pipeline_ltx_video.py:1251-1262): matches the std of the guided prediction back to the unperturbed conditional:
```
factor = std(text) / std(guided)
factor = rescale * factor + (1 - rescale)
noise_pred *= factor
```

### PAG (Perturbed Attention Guidance)

Not a separate code path in LTX-Video. PAG is effectively subsumed by STG-AttentionValues (same math: replace attention output with V on the perturb pass). There is no `pag_scale` argument. The README mentions PAG historically (0.9.1 release notes), but the parameter is `stg_scale` with `stg_mode=attention_values`.

### Stochastic sampling

`stochastic_sampling: bool` (pipeline_ltx_video.py:791, default `False`). When True, the scheduler's step does SDE reinjection (schedulers/rf.py:364-367):
```
x0   = sample - t * model_output
x_{t-1} = add_noise(x0, randn_like(sample), t - dt)
```
vs the deterministic ODE `x_{t-1} = sample - dt * model_output` (schedulers/rf.py:369). All stock 0.9.8 configs disable it (`stochastic_sampling: false`).

### Default step counts

- **13B dev (ltxv-13b-0.9.8-dev.yaml)**: first pass 30 steps with `skip_final_inference_steps: 3` (i.e. 27 effective), second pass 30 with `skip_initial_inference_steps: 17` (i.e. 13 effective). Total ≈ 40 forwards, each with 3 batched conds (CFG+STG) → ~120 transformer calls.
- **13B distilled (ltxv-13b-0.9.8-distilled.yaml)**: first pass 7 explicit timesteps `[1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725]`, second pass 3 timesteps `[0.9094, 0.725, 0.4219]`. `guidance_scale=1, stg_scale=0` → **no CFG, no STG, 1 forward per step**. Total 10 forwards.
- **2B distilled**: same 7+3 schedule as 13B distilled.
- `allowed_inference_steps` embedded in safetensors metadata for distilled models gates valid t values: `[1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219]` (verified from `ltxv-13b-0.9.8-distilled.safetensors` header).

### Timestep scheduling

`RectifiedFlowScheduler` in `ltx_video/schedulers/rf.py:176-386`. Three `sampler` modes (:201-214):
- `Uniform` — `linspace(1, 1/N, N)`
- `LinearQuadratic` — linear first half (slope `threshold_noise/linear_steps=0.025/half`), quadratic second half, tail reversed to descend from 1.0 (ltx_video/schedulers/rf.py:25-46). This is what all 0.9.8 dev/distilled checkpoints use.
- `Constant` — shifted once by `time_shift(shift, 1, linspace)`.

Optional resolution-dependent shift (`shifting = "SD3" | "SimpleDiffusion"`, :216-225) not used in 0.9.8 configs.

---

## 5. Negative prompts

Encoded exactly like the positive prompt, via the same T5 (`PixArt-alpha/PixArt-XL-2-1024-MS`, subfolder `text_encoder`). See pipeline_ltx_video.py:421-475.

Default negative: `"worst quality, inconsistent motion, blurry, jittery, distorted"` (ltx_video/inference.py:351-354, and the same string appears in trainer validation configs).

**Always encoded**, but only USED when `do_classifier_free_guidance = guidance_scale[i] > 1.0` (pipeline_ltx_video.py:1116, 1224-1244). For distilled models (`guidance_scale=1`), the negative is still encoded (:1024-1040) but never enters the CFG combination.

Batch layout (pipeline_ltx_video.py:1060-1070): `torch.cat([negative, positive, positive], dim=0)` — three slots for uncond/cond/perturb. The slicing at :1126-1133 picks which are actually forwarded based on `do_classifier_free_guidance` and `do_spatio_temporal_guidance`.

If the user passes `negative_prompt_embeds=None`, it's synthesized from the negative_prompt string (:1020-1040); if both are None, a `torch.zeros_like(prompt_embeds)` fills in (:1049-1058).

**Not mandatory** — users can pass `negative_prompt=""`.

---

## 6. Upscalers

### Spatial upscaler (`LatentUpsampler`, ltx_video/models/autoencoders/latent_upsampler.py)

- Input shape: `(B, 128, F, H, W)` (VAE latent space)
- Output shape: `(B, 128, F, 2H, 2W)` — spatial 2× only when `spatial_upsample=True, temporal_upsample=False`
- Architecture (latent_upsampler.py:55-107):
  1. `initial_conv` 3×3 Conv3d: 128 → `mid_channels=512`, with GroupNorm(32), SiLU
  2. 4× `ResBlock` (512→512→512) (:79-81)
  3. `upsampler` for spatial-only: `Conv2d(512, 2048, 3)` + `PixelShuffleND(2)` → (B, 512, F, 2H, 2W). Notice 2D pixel shuffle applied per-frame via einops rearrange (:112-127). For spatial+temporal it would be `Conv3d(512, 4096, 3) + PixelShuffleND(3)` but the shipped checkpoint is spatial-only.
  4. 4× post-upsample ResBlocks (:103-105)
  5. `final_conv` 3×3 Conv3d: 512 → 128
- Weights: `ltxv-spatial-upscaler-0.9.8.safetensors` (481 MB), `ltxv-spatial-upscaler-0.9.7.safetensors` (481 MB) on `Lightricks/LTX-Video`.
- Config is embedded in `__metadata__["config"]` of the safetensors (latent_upsampler.py:185-193).

### Temporal upscaler

- File: `ltxv-temporal-upscaler-0.9.8.safetensors` (499 MB), `ltxv-temporal-upscaler-0.9.7.safetensors` (499 MB).
- Same `LatentUpsampler` class, config'd with `spatial_upsample=False, temporal_upsample=True, dims=3`. Conv is `Conv3d(512, 1024, 3) + PixelShuffleND(1)` → doubles the F axis (:93-97). Cropped with `x[:, :, 1:]` post-upsample (:138) to drop the duplicated first frame.
- **Not wired into `LTXMultiScalePipeline`.** The multiscale orchestrator only uses the spatial upscaler; the temporal upscaler is currently loaded only through ComfyUI workflows.

### Multi-scale pipeline (`LTXMultiScalePipeline`, pipeline_ltx_video.py:1821-1895)

```
kwargs['output_type']='latent'
kwargs['width', 'height'] *= downscale_factor     # default 0.6666666 → ~2/3
first_pass_result = video_pipeline(**first_pass_kwargs)
upsampled = latent_upsampler(first_pass_result.latents)    # 2× spatial
upsampled = adain_filter_latent(upsampled, reference=first_pass_result.latents)
second_pass_result = video_pipeline(
    latents=upsampled,
    width=downscaled_w * 2, height=downscaled_h * 2,
    skip_initial_inference_steps=17,   # from second_pass yaml
    **second_pass_kwargs,
)
# Final bilinear resize to the user's requested (orig_w, orig_h).
```

`adain_filter_latent` (pipeline_ltx_video.py:1790-1818) matches upsampled-latent per-channel mean/std to the first-pass reference, then lerps with `factor=1.0` (pure adapt).

`un_normalize_latents` / `normalize_latents` (imported from vae_encode.py) bracket the upsampler so it operates in un-normalized latent space, then returns to per-channel-normalized for the second pass.

---

## 7. Prompt enhancement

`ltx_video/utils/prompt_enhance_utils.py:64-110`, wired in at `pipeline_ltx_video.py:1002-1018`.

**Two models, both optional:**
- Image captioner: `MiaoshouAI/Florence-2-large-PromptGen-v2.0` → 2-stage Florence-2 for I2V mode.
- LLM enhancer: `unsloth/Llama-3.2-3B-Instruct` (bf16).

**When applied:** `enhance_prompt=True` AND `prompt_word_count < prompt_enhancement_words_threshold` (ltx_video/inference.py:479-488). All 0.9.8 configs have `prompt_enhancement_words_threshold: 120`.

**T2V branch** (prompt_enhance_utils.py:76-82): only the LLM is called. System prompt is `T2V_CINEMATIC_PROMPT` (prompt_enhance_utils.py:9-25) — a 24-line cinematic-director instruction template with 150-word cap.

**I2V branch** (prompt_enhance_utils.py:83-108): caption first frame of `conditioning_items[0]` with Florence-2 (`<DETAILED_CAPTION>` task, `num_beams=3`, `do_sample=False`, `max_new_tokens=1024`; prompt_enhance_utils.py:194-207), then LLM with `I2V_CINEMATIC_PROMPT` (:27-44) which adds "Align to the image caption if it contradicts the user text input".

Only the *first* conditioning item at frame 0 is supported for enhancement; multi-keyframe falls through with a warning (:84-88).

Output is `List[str]` one-per-batch; replaces the user prompt before T5 encoding.

---

## 8. Model variants

Measured from safetensors headers on `Lightricks/LTX-Video`:

| Variant | Size | Transformer | Guidance | Steps | VAE | Notes |
|---|---|---|---|---|---|---|
| `ltxv-13b-0.9.8-dev` | 27.3 GB bf16 | 48 layers × 32 heads × 128 dim (inner=4096), `qk_norm=rms_norm`, `standardization_norm=rms_norm`, `causal_temporal_positioning=True`, `timestep_scale_multiplier=1000`, `positional_embedding_max_pos=[20,2048,2048]`, `theta=10000.0` | CFG + STG (time-varying `guidance_scale=[1,1,6,8,6,1,1]`, `stg_scale=[0,0,4,4,4,2,1]`) | 30 first pass (27 effective) + 30 second (13 effective) | timestep-conditioned causal VAE, 128 latent channels, patch_size=4 | `cfg_star_rescale=true` |
| `ltxv-13b-0.9.8-dev-fp8` | 14.97 GB fp8_e4m3fn | same | same | same | same | Requires [q8_kernels](https://github.com/Lightricks/LTXVideo-Q8-Kernels) patched transformer (`ltx_video/inference.py:186-200`) |
| `ltxv-13b-0.9.8-distilled` | 27.3 GB bf16 | same as dev | `guidance_scale=1, stg_scale=0` — **no CFG, no STG, deterministic** | first pass 7 explicit timesteps, second pass 3 (see §4). `allowed_inference_steps: [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219]` — asserted at pipeline_ltx_video.py:952-956 | same | `tone_map_compression_ratio: 0.6` on second pass (pipeline_ltx_video.py:1748-1787) |
| `ltxv-13b-0.9.8-distilled-fp8` | 14.97 GB fp8 | same | same | same | same | |
| `ltxv-2b-0.9.8-distilled` | 6.0 GB bf16 | smaller (see distilled config) | same 7+3 distilled schedule | same | same 128-ch VAE | Uses the **same** spatial upscaler as 13B (`ltxv-spatial-upscaler-0.9.8.safetensors`) |
| `ltxv-2b-0.9.8-distilled-fp8` | 4.26 GB fp8 | same | | | | |
| Older: 2b-0.9.0/0.9.1/0.9.5/0.9.6 | 5–9 GB | different (no `causal_temporal_positioning`, different norms) | various | | Older VAE variants | Pre-multiscale |

Key **stochastic vs deterministic**: stochasticity is `stochastic_sampling=True` (off in all shipped configs). "Dev vs distilled" is not about stochasticity — it's about CFG/STG enablement and timestep count.

**2b vs 13b** — different num_layers / inner_dim (13B: 48/4096, 2B: smaller — config embedded in each safetensors). Both consume the same VAE and text encoder. Both use patch_size=1 patchifier.

---

## 9. TeaCache

Not in the LTX-Video main repo. README.md acknowledges it as a community contribution:

> TeaCache for LTX-Video 🍵
> Repository: https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video

TeaCache is a training-free residual cache: at timestep `t`, compute a cheap "modulation signal" from `embedded_timestep`; if `|δsignal|` vs the last full-compute step is below a threshold, reuse the prior residual instead of running the transformer. Implementation lives in the TeaCache repo as a transformer forward wrapper. The ComfyUI integration is via a separate node repo (ComfyUI-LTXTricks). No stub in the main pipeline.

---

## 10. Other features

### Camera control

**Not a first-party feature.** LTX-Video does not ship explicit camera embeddings. Motion is controlled via prompt text ("camera pans left", "dolly in"), multi-keyframe conditioning, or community IC-LoRAs. The `sparse_tracks.py` node in ComfyUI-LTXVideo is for sparse-motion conditioning (a different community feature, not camera poses).

### Version differences

- **0.9.0 → 0.9.1**: STG/PAG support added; Diffusers config conversion on-the-fly; CPU offloading; timestep-conditioned VAE decoder introduced.
- **0.9.1 → 0.9.5**: OpenRail-M license; keyframes & video extension; better VAE; higher resolutions.
- **0.9.5 → 0.9.6**: Default resolution 1216×704 @30fps; distilled 2B (15× faster, no CFG/STG); stochastic sampling switch.
- **0.9.6 → 0.9.7**: 13B model; distilled 13B variant (+ distilled-lora128 file); fp8 variants; multi-scale pipeline debuts; spatial+temporal upscalers released; IC-LoRA control models released 2025-07-08 (depth/pose/canny).
- **0.9.7 → 0.9.8**: Long-shot generation (up to 60 seconds); detailer IC-LoRA; 2B distilled added to 0.9.8; new `tone_map_compression_ratio` in distilled second pass (0.6); explicit-timesteps-with-`allowed_inference_steps` gating.
- **0.9.8 → LTX-2**: Separate repo (`Lightricks/LTX-2`), audio+video joint model, 4K/50fps. Not covered here.

### Conditioning-item latent noise injection

`image_cond_noise_scale` (default 0.15 from inference.py:365-368, active in denoising loop at pipeline_ltx_video.py:1151-1159 via `add_noise_to_image_conditioning_latents` :596-620): adds `t² * noise_scale * randn` to hard-conditioned (mask==1.0) latents. Helps motion continuity when conditioning on a single frame. Formula: `latents = init_latents + noise_scale * noise * t^2`, only applied where `conditioning_mask > 1.0 - eps`.

### Decode-time noise injection

Timestep-conditioned VAE decoder requires a `decode_timestep` (0.05 default) and `decode_noise_scale` (0.025 default) blend at decode start (pipeline_ltx_video.py:1307-1326):
```
latents = latents * (1 - decode_noise_scale) + noise * decode_noise_scale
image = vae_decode(latents, vae, is_video, timestep=decode_timestep)
```
Only if `self.vae.decoder.timestep_conditioning == True`.

### Tone mapping

`tone_map_latents` (pipeline_ltx_video.py:1748-1787): sigmoid-based compression of high-amplitude latent values applied right before VAE decode. `compression=0.0` is identity; distilled configs use 0.6 on second pass only.

### CRF pre-compression

`ltx_video/pipelines/crf_compressor.py` applies JPEG-like compression to conditioning inputs before VAE encoding (inference.py:99-103: gaussian blur sigma 1.0, kernel 3; then CRF compress; then normalize `(x/127.5) - 1.0`). Reduces VAE artefacts on clean JPEG inputs.

### Resolution binning

`ASPECT_RATIO_1024_BIN` and `ASPECT_RATIO_512_BIN` dicts (pipeline_ltx_video.py:51-121) and `classify_height_width_bin` (:703-711) — exposed but **not auto-invoked** in the default code path (`use_resolution_binning` parameter is mentioned in docstring but not actually used in `__call__`). The user-provided H/W is simply padded to the nearest multiple of 32 in `inference.py:461-462`.

### TPU / flash attention path

`use_tpu_flash_attention` flag on `Transformer3DModel` (transformer3d.py:76, set via `set_use_tpu_flash_attention` :162-171). Changes attention-mask shape and disables the STG "conditioning token at end" strip-logic (pipeline_ltx_video.py:1571-1580). Not relevant for CUDA/GPU inference.

### 8-bit / quantization

- fp8_e4m3fn: native transformer precision via `q8_kernels` patch (inference.py:186-200). Requires external kernel install.
- Community: LTX-VideoQ8 repo (`KONAKONA666/LTX-Video`), Diffusers 8-bit integration (`sayakpaul/q8-ltx-video`). Both are independent forks, not wired into this pipeline.

### ROPE specifics

Transformer3DModel.precompute_freqs_cis at transformer3d.py:204-257. Three-axis 3D RoPE over `(frame, height, width)` fractional positions. `theta=10000`, `spacing="exp"` by default (log-spaced indices). `positional_embedding_max_pos=[20, 2048, 2048]` for 13B — fractional positions normalized by these. Note: `fractional_coords[:, 0] *= 1/frame_rate` at pipeline_ltx_video.py:1149 — the *temporal* coordinate is divided by frame_rate before RoPE, so "30fps motion" produces the same RoPE values as "60fps motion at 2× frames".
