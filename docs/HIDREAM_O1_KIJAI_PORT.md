# HiDream-O1 Kijai port

Survey + port of upstream ComfyUI / Kijai HiDream-O1 work into the pure-Rust
`inference-flame` HiDream-O1 module. Inference-only — no trainer changes.

Sources (read 2026-05-21):

- ComfyUI master `comfy/ldm/hidream_o1/` (4 files):
  - `attention.py` — raw URL:
    `https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/hidream_o1/attention.py`
  - `model.py`, `conditioning.py`, `utils.py` — same path, swap the filename.
- Kijai's `hidream_o1` branch, initial commit `974aab796d` "Initial
  HiDream01-image support" (patch URL:
  `https://github.com/kijai/ComfyUI/compare/master...hidream_o1.patch`).
- Kijai's `hidream_conds` branch (already merged upstream as PR #13944):
  single commit `468203916b47f1fb7242f72b7554ea212173ae9d`, 7 lines in
  `comfy/model_base.py` for area-conditioning noise cropping. **Not ported**
  (touches the model-base "extra_conds" plumbing that doesn't exist on our
  pipeline; safe to revisit when ref-image conditioning lands).

## What was ported

### Two-pass attention

Upstream: `comfy/ldm/hidream_o1/attention.py::make_two_pass_attention`
(41 lines). Port: `inference-flame/src/models/hidream_o1/decoder.rs::
hidream_o1_two_pass_attention` (new public helper), with four-case dispatch
matching Kijai verbatim. The dispatcher then delegates to existing
flame-core primitives — no new CUDA kernel.

The mixed-case (case 4) was already wired in via the prior
`flame_core::attention::sdpa_prefix_causal_full` call at
`decoder.rs:480-483`. The Kijai port adds the three early-out cases on top
and consolidates the four dispatch arms under one named entry point.

### Files changed

- `inference-flame/src/models/hidream_o1/decoder.rs`
  - +71 lines: new `pub fn hidream_o1_two_pass_attention` at file bottom.
  - 3 lines changed: dispatch site at the formerly-direct
    `sdpa_prefix_causal_full` call (line 480) now routes through the new
    helper.
- `inference-flame/src/models/hidream_o1/mod.rs`
  - 1 line changed: re-export `hidream_o1_two_pass_attention`.

## Two-pass attention design

Tokens `[0, ar_len)` are the AR / text prefix and must attend causally.
Tokens `[ar_len, T)` are the image-generation tail and attend
bidirectionally over the full sequence.

The naive way to express this is an additive mask of shape `(B, 1, T, T)`
with `-inf` in the masked entries. At `T ≈ 16384` (a typical edit at
1024², 32-patch + text prompt) that is `1 * 1 * 16384 * 16384 * 4 = ~1 GiB`
in FP32 or `~512 MiB` in BF16 just for the mask buffer. Materializing it
also pulls the SDPA call onto the generic chunked/streamed path and off
cuDNN's fused SDPA.

The two-pass split avoids both costs:

| Case | Condition | Implementation | Why |
|------|-----------|----------------|-----|
| 1 — KV-cache hot | `T_q < T_kv` | `flame_sdpa(q, k, v, None)` | All fresh Q rows are in the gen region; cached AR rows live in K/V only. Single full call. |
| 2 — all-causal | `ar_len >= T_q` | `flame_core::attention::sdpa_causal(q, k, v)` | Every Q row is AR; the gen tail is empty. Single causal call. |
| 3 — all-gen | `ar_len == 0` | `flame_sdpa(q, k, v, None)` | Every Q row is gen; the AR head is empty. Single full call. |
| 4 — mixed | `0 < ar_len < T_q` | `flame_core::attention::sdpa_prefix_causal_full(q, k, v, ar_len)` | AR-causal head + gen-full tail, concatenated along sequence. Internal calls: `sdpa_causal` on `Q[:, :, :ar_len]` + `sdpa(q, k, v, None)` on the full sequence + narrow the gen rows out. Both halves stay on cuDNN. |

`ar_len` is `Self::ar_prefix_len(b, s_total, token_types_bin)` in
`model.rs:636`. It counts the leading run of `token_types == 0` (AR) rows
in `token_types_bin`. It already errors if the sequence isn't strictly
prefix-AR (i.e. an AR row mixed in after a gen row would be a bug; not
something HiDream-O1's tokenizer produces).

Case 1 (KV-cache hot) is unreachable from the current `inference-flame`
HiDream-O1 pipeline — the model recomputes per step rather than caching
the AR-prefix K/V. The dispatch arm is ported anyway so the helper's
surface matches Kijai's and a future caller that does maintain a KV cache
can use this without changes.

## Public API surface

```rust
// inference-flame::models::hidream_o1
pub fn hidream_o1_two_pass_attention(
    q: &Tensor,        // [B, H, T_q,  D] BF16
    k: &Tensor,        // [B, H, T_kv, D] BF16, already GQA-replicated
    v: &Tensor,        // [B, H, T_kv, D] BF16, already GQA-replicated
    ar_len: usize,
) -> Result<Tensor>;   // [B, H, T_q, D] BF16
```

Reachable from `inference_flame::models::hidream_o1::hidream_o1_two_pass_attention`.

## Memory / perf impact

At HiDream-O1-Dev's typical T2I sequence length (`T ≈ 1k–4k` text + `1024 *
32-patch / patch² = 1024` image patches at 1024²; ≈ 2k total) the
materialized mask is `~32 MiB` BF16 — not a major problem. At edit
resolutions (`T ≈ 16k`+) the mask balloons to `~512 MiB` BF16, which is
the headline savings Kijai's branch was after.

Speed: we already routed the mixed case to `sdpa_prefix_causal_full` (which
ports the same idea) before this commit. The new helper's net delta vs.
the prior dispatch is:

- All-causal short-circuit avoids one `narrow + cat` of full-length tensors.
- All-gen short-circuit collapses to a single fused SDPA call instead of
  one degenerate causal call + one full call + a narrow + a cat.
- KV-cache hot short-circuit (case 1) — not reachable today.

I have not run a microbenchmark; expected impact for the dominant `T ≈
2-16k` T2I path is sub-percent because both arms already used cuDNN
SDPA. The structural win was already captured by the prior
`sdpa_prefix_causal_full` wire-up; this commit is the cleanup + Kijai-parity
edge cases.

## Surveys (not ported)

### `conditioning.py` — ref-image dual-path conditioning

230 lines (after the Kijai patch's update). Builds two parallel
representations of each reference image:

1. **32-patch path**: each ref image is resized so its longest side fits
   `ref_max_size(target_max_dim, k)`, patchified at `PATCH_SIZE=32`, and
   concatenated as flat patches to the noised target on the input sequence.
2. **Qwen3-VL ViT path**: each ref is resized to the K-dependent
   `cond_image_size(k)`, normalized with `[0.5, 0.5, 0.5]` mean/std,
   patchified at `VIT_PATCH=16` with `VIT_MERGE=2` and `VIT_TEMPORAL_PATCH=2`,
   then run through the Qwen3-VL ViT (`Qwen35VisionModel`). The output
   tokens are scattered into `input_ids` at `<|image_pad|>` positions.

`build_extra_conds` assembles the unified-sequence tensors and routes
them through `get_rope_index_fix_point` to build MRoPE positions.

What porting this would touch in `inference-flame`:

- A new `prepare_ref_images(refs, target_h, target_w) -> RefBundle` helper
  in a new `hidream_o1/conditioning.rs`.
- A pure-Rust Qwen3-VL ViT (`Qwen35VisionModel`). The text decoder side
  already exists at `qwen3_encoder.rs`; the vision tower would be a new
  module.
- Plumbing in `pipeline.rs` to accept `Vec<RefImage>` and to feed
  `ref_pixel_values`, `ref_image_grid_thw`, `ref_patches`, `input_ids`,
  `position_ids`, `token_types`, `vinput_mask` into `HiDreamO1Model::forward`.
- Extending `build_mrope_positions` to iterate multiple image grids with
  per-image `skip_vision_start_token` flags (currently single-image only).
- Tokenizer constants `IMAGE_TOKEN_ID = 151655`, `VIDEO_TOKEN_ID = 151656`,
  `VISION_START_ID = 151652`, `VISION_END_ID = 151653` (already present in
  `HiDreamO1Config`).

This is non-trivial — order of a week — and gated on whether we want
ref-image / IP-Adapter style conditioning at all. Surveyed but not ported.

### `utils.py` — `get_rope_index_fix_point`

Audited against `mrope.rs::build_mrope_positions`. Structurally aligned
for the single-image T2I path: same `fix_point=4096`, same `st_idx`
threading, same `text_len = ed - st - skip` math. The HiDream-O1 sample
target — a single noised gen image — produces the same `T/H/W` arrays in
both implementations.

Gap (only relevant for ref-image support): the upstream function loops
over `image_nums` images with a per-image `skip_vision_start_token[i]`
flag, and the cross-image `st_idx` cascade carries over (`st_idx = last
position + 1`). Our `build_mrope_positions` short-circuits on the first
image grid. When ref-image conditioning lands, extend the loop to iterate
`image_grid_thw` and track `st_idx` per iteration.

No port action required for the current T2I-only inference path.

## Known issues / out of scope

- **Trainer `t_embedder1.mlp.{0,2}` LoRA gradient does not propagate.**
  Likely the `where_mask` call in `scatter_tms_token` (model.rs around line
  633) detaching through `clone_result`. Out of scope for this inference
  port — this doc is inference-only and the trainer is a separate code path.
- **Ref-image conditioning** — not ported. Surface enumerated above.
- **`hidream_conds` branch noise-cropping** — single 7-line ComfyUI-internal
  patch; touches `comfy/model_base.py`'s `extra_conds` plumbing that has no
  analogue in `inference-flame`. Skip.
- **Kijai `sample_euler_flash_flowmatch`** — present in Kijai's patch
  (`comfy/k_diffusion/sampling.py`) but our scheduler
  (`FlashFlowMatchEulerDiscreteScheduler` in `scheduler.rs`) already
  implements equivalent math (linear `s_noise` interpolation + noise
  std-clamp in F32). No port required; if a future variant changes the
  semantics, port at that time.

## Future work

- Port the Qwen3-VL `Qwen35VisionModel` ViT and ref-image conditioning
  (gated on demand).
- Microbenchmark the four cases on real edit sequences (`T ≈ 16k`) to
  quantify the cuDNN-fast-path savings vs the materialized-mask path.
- Extend `build_mrope_positions` to multi-image with per-image
  `skip_vision_start_token` — required for ref images.
- Investigate `t_embedder1.mlp.{0,2}` LoRA dead-grad on the trainer side
  (separate doc, separate session).
