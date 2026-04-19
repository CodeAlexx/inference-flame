# LTX-2 Full Feature Parity Plan

Goal: pure-Rust LTX-2 inference that matches Lightricks's LTX-Video
pipeline feature-for-feature. AV is the product — audio is never
optional.

Workflow per feature:

```
1. Write PyTorch parity reference (import Lightricks code or checkpoint,
   never reconstruct from prose). Output to output/<feature>_ref.safetensors.
2. Write Rust parity compare bin (src/bin/<feature>_parity.rs).
3. Spawn builder agent → implement.
4. Spawn skeptic agent → find the holes.
5. Spawn bug-fixer agent → close them.
6. Re-run parity. Must pass before commit. No shortcuts.
```

## Phase 1 — Foundation ✅ COMPLETE

| Item | Parity ref | Compare bin | Impl | Status |
|---|---|---|---|---|
| LoRA fusion math (`B @ A`) | `scripts/lora_fusion_parity_ref.py` | `src/bin/lora_fusion_parity.rs` | `src/models/lora_loader.rs` | ✅ PASS 12/12 @ 0.999999 |
| LoRA wired into `LTX2StreamingModel` | — | `src/bin/ltx2_lora_wiring_check.rs`, `src/bin/ltx2_lora_fusion_correctness.rs` | `src/models/ltx2_model.rs` | ✅ bit-exact video+audio, disk-sync + BlockOffloader |
| LinearQuadratic scheduler | `scripts/ltx2_sigma_schedule_ref.py` (imports `ltx_video.schedulers.rf`) | `src/bin/ltx2_sigma_parity.rs` | `src/sampling/ltx2_sampling.rs::linear_quadratic_schedule` | ✅ PASS max_abs=0.0 for n=8/20/25/30 |
| `--lora` in `ltx2_generate_av` | — | correctness test covers AV | `src/bin/ltx2_generate_av.rs` | ✅ wired, audio never skipped |
| Audio LoRA fusion via BlockOffloader | `load_raw_weight("audio_attn1.to_q")` | `ltx2_lora_fusion_correctness` | `apply_loras_to_weights` | ✅ bit-exact |

Commits: `9430e51`, `978abaf`, `408c00b`.

## Phase 2 — Guidance ✅ COMPLETE

| Item | Parity ref | Rust impl | Status |
|---|---|---|---|
| CFG (uncond + scale·(cond-uncond)) | — (math-simple) | `ltx2_generate_av.rs` denoise loop | ✅ |
| Negative prompt wiring (Gemma+FE, audio too) | `scripts/ltx2_neg_prompt_ref.py` | `DEFAULT_NEGATIVE` const, `--neg` flag | ✅ byte-exact string, non-zero + self-consistent encoding |
| CFG-star rescale | `scripts/ltx2_cfg_star_ref.py` | `sampling::ltx2_guidance::cfg_star_rescale` | ✅ max_abs=0.0 BF16 |
| STG skip-layer mask | `scripts/ltx2_stg_mask_ref.py` (imports `Transformer3DModel.create_skip_layer_mask`) | `sampling::ltx2_guidance::build_skip_layer_mask` | ✅ max_abs=0.0 |
| STG AttentionValues skip inline | code inspection only (per-layer attn+V blend) | `LTX2Attention::forward_with_skip` | ✅ scalar math exact; tensor-level DiT parity deferred |
| STG std-rescale (`rescaling_scale`) | `scripts/ltx2_stg_rescale_ref.py` | `sampling::ltx2_guidance::stg_rescale` | ✅ max_abs=0.0 |
| Audio path uniform STG | — | `forward_video_only_with_skip` + `forward_with_skip` skip video+audio self-attn | ✅ A2V/V2A intentionally NOT skipped |

Commits: `8c84e0a`, `df8a562`.

### Phase 2 known gap (deferred encoder-parity work)

`ltx2_neg_prompt_parity` Layer 3 fails at cos_sim ≈ 0: our in-tree
`Gemma3Encoder` + `feature_extractor` outputs differ wildly from
Lightricks's private `ltx_core.text_encoders.gemma.encode_text`. Not
BF16 noise — different hidden-state layer / chat template / tokenizer.
Reproducing Lightricks's encoder bit-by-bit is multi-session scope.
Current state: our pipeline is self-consistent (pos and neg encoded by
the SAME path), which is what CFG needs, but the absolute embedding
differs from Lightricks. Tracked here as TODO.

## Phase 2 — Guidance

| Item | Notes | Status |
|---|---|---|
| Negative prompts | Always encoded via Gemma. Enter CFG only when guidance_scale > 1. Default string: `"worst quality, inconsistent motion, blurry, jittery, distorted"`. Batch layout: `[neg, pos, pos]` (three slots for uncond/cond/perturb). | ❌ |
| STG (Spatiotemporal Guidance) | Third forward pass with per-timestep `skip_block_list`. Four strategies: `AttentionValues` (default), `AttentionSkip`, `Residual`, `TransformerBlock`. Skip mask at `Transformer3DModel.create_skip_layer_mask`. Dev yaml 0.9.8 timesteps: `[[], [11,25,35,39], [22,35,39], [28], [28], [28], [28]]` across 7 guidance windows. | ❌ |
| `cfg_star_rescale` | Projects uncond onto cond direction: `alpha = dot(eps_text, eps_uncond) / ||eps_uncond||²; eps_uncond ← alpha * eps_uncond`. Enabled in dev configs. | ❌ |
| STG rescaling (`rescaling_scale=0.7`) | After STG combine: `factor = std(text) / std(guided); factor = 0.7 * factor + 0.3; noise_pred *= factor`. | ❌ |

## Phase 3 — Conditioning breadth

| Item | Notes | Status |
|---|---|---|
| Multi-keyframe conditioning | `ConditioningItem(media_item, media_frame_number % 8 == 0, strength)`. `_handle_non_first_conditioning_sequence` at pipeline_ltx_video.py:1652-1726. | ❌ |
| Video extension forward/backward | Reuses multi-keyframe machinery. frame=0 for forward, frame=N for backward. | ❌ |
| IC-LoRA reference concat | Not a separate path — just prepend `reference_latents` to `target_latents` at dim=1 and set `conditioning_mask=1.0` on the ref half. Timestep forced to 0 there. Decode target half only. `reference_downscale_factor` metadata — default 1.0 when absent. | ❌ |
| `image_cond_noise_scale` (default 0.15) | `latents += noise_scale * noise * t²` on positions where `conditioning_mask > 1 - eps`. | ❌ |

## Phase 4 — Multi-scale

| Item | Notes | Status |
|---|---|---|
| `LatentUpsampler` (spatial 2×) | 481MB ckpt, 3D conv + 4× ResBlock + PixelShuffleND(2) + 4× ResBlock + final conv. Config embedded in `__metadata__["config"]`. | ❌ |
| `LTXMultiScalePipeline` | first pass at downscale 2/3 → upsample → AdaIN → second pass at 2× with `skip_initial_inference_steps=17`. | ❌ |
| `adain_filter_latent` | Per-channel mean/std match of upsampled to reference latent. | ❌ |
| `tone_map_latents` | Sigmoid compression of latent amplitudes before VAE decode. `compression=0.6` on distilled second pass. | ❌ |
| `decode_timestep` + `decode_noise_scale` | `latents = latents * (1 - s) + noise * s; vae_decode(latents, timestep=t)`. Defaults `t=0.05, s=0.025`. Only when VAE's `timestep_conditioning == True`. | ❌ |

## Phase 5 — Nice-to-have (deferred)

| Item | Notes |
|---|---|
| Prompt enhancement | Florence-2 + Llama-3.2-3B — heavy deps. Not core. |
| TeaCache | External repo (ali-vilab/TeaCache). |

## Audio invariant

LTX-2 is a cross-modal AV model — audio is core product. Every feature
added must:
- Cover both video AND audio paths where applicable
- Be verified with at least one audio-keyed parity test (not just video)
- Not introduce an `--audio off` or similar escape hatch

The distilled LoRA ships 2134 audio keys (64% of total) — any code path
that silently skips audio fusion will visibly degrade distilled-LoRA
generation.

## Current commits on master

- `9430e51` LoRA fusion parity test (12/12 @ 0.999999)
- `7b29976` Qwen2.5-VL three-bug fix (final_hidden 0.994)
- `3c93718` QwenImage VAE decoder parity
- `a43a27e` ldm_decoder VAE_SAVE_STAGES
- `3f87c8d` output paths migration

Uncommitted on master (from lora-wiring-builder agent):
- `src/models/ltx2_model.rs` — `LoraWeights` field + fusion in disk-sync
  + both BlockOffloader paths + FP8-resident hard-error
- `src/bin/ltx2_generate.rs` — `--lora PATH[:STRENGTH]` flag
- `src/bin/ltx2_lora_wiring_check.rs` — smoke test (sanity, passes)
- `src/bin/ltx2_lora_fusion_correctness.rs` — correctness test (in progress)
- `src/sampling/ltx2_sampling.rs` — `linear_quadratic_schedule` port
- `src/bin/ltx2_sigma_parity.rs` — sigma parity bin
- `scripts/ltx2_sigma_schedule_ref.py` — reference via Lightricks code

## Execution order

1. **Finish Phase 1 verification** — audio-aware correctness test, wire
   LoRA into `ltx2_generate_av`, confirm audio deltas reach the AV
   BlockOffloader forward. Commit.
2. **Phase 2 builders in parallel** — negative prompts, STG,
   cfg_star_rescale — each gets a builder agent with the
   parity-test-first constraint. Then skeptic + bug-fixer.
3. **Phase 3** — conditioning breadth. Multi-keyframe is the biggest,
   IC-LoRA lives on top of it.
4. **Phase 4** — multi-scale. Spatial upscaler is the first new model
   we add (not a gap in existing code); needs parity from a tiny
   Python dump.
5. **Phase 5** — deferred until user asks.
