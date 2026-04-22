# Turbo Flame Phase 2 v5.1 — Three-Model Timing Report

**Status:** Scaffold. The orchestrator runs the GPU benches after Bug Fixer +
Skeptic sign-off and fills in the `?` placeholders with measured numbers from
RTX 3090 Ti.

## Bench prerequisites

Phase 1's `TurboBlockLoader::new` reads a single safetensors file. The Chroma
and Qwen-Image-Edit checkpoints ship as diffusers-format **sharded**
safetensors. Before running the new benches you need a single-file merged
checkpoint per model on disk and the corresponding env vars set:

- `CHROMA_TURBO_SAFETENSORS=/path/to/Chroma1-HD.safetensors`
- `QWEN_TURBO_SAFETENSORS=/path/to/qwen_image_edit_2511_merged.safetensors`

Without these the new turbo binaries (and the parity tests / benches that
shell out to them) will error out cleanly with a hint pointing at the env
var. Multi-shard support inside `TurboBlockLoader` is Phase 3+ work; spec
§10 forbade changing the loader API in Phase 2 by design.

## Bench protocol notes (Skeptic)

- Warmup pass before the first measured run on each model (cold-path
  `cuMemMap` cost is captured but discarded as noise).
- Cooldown between Qwen-Image-Edit runs (~46 min combined for off+on);
  thermal throttling on a 3090 Ti is real on back-to-back 23-min runs.
- Capture ambient GPU temp at start of each measured run; flag deltas >5 °C
  between off-run and on-run as a confounder.

Builder ships:
- The bench scaffolding (`benches/turbo_chroma_offload.rs`,
  `benches/turbo_qwen_edit_offload.rs`, plus the existing
  `benches/turbo_klein9b_offload.rs`).
- The two new turbo binaries (`chroma_infer_turbo`, `qwenimage_edit_gen_turbo`).
- Two `#[ignore]`'d parity tests (`turbo_chroma_parity`, `turbo_qwen_edit_parity`).

Builder does NOT run the long benches (Qwen-Image-Edit alone is ~46 min for
both off + on runs; spec §7).

---

## Per-model timing (RTX 3090 Ti)

```
                   Turbo OFF       Turbo ON        Delta     Overlap %

Klein 9B (1024²/50 CFG 4.0)
  Model Load       19.0 s          ?               ?
  Text Encode      20.0 s          ?               ?
  Denoise total    247.0 s         ?               ?
    per-step       4.94 s          ?               ?
    per-swap       ? μs            ? μs            ?         ?
  VAE Decode       7.7 s           ?               ?
  Total            295.0 s         ?               ?

Chroma (1024²/40 CFG 4.0)
  Model Load       ?               ?               ?
  Text Encode      ?               ?               ?
  Denoise total    ?               ?               ?
    per-step       ?               ?               ?
    per-swap       ? μs            ? μs            ?         ?
  VAE Decode       ?               ?               ?
  Total            ?               ?               ?

Qwen-Image-Edit-2511 (1024²/50 true-CFG 4.0)
  Model Load       ?               ?               ?
  Text Encode      ?               ?               ?
  Denoise total    ~1380 s         ?               ?
    per-step       ~27.6 s         ?               ?
    per-swap       ? μs            ? μs            ?         ?
  VAE Decode       ?               ?               ?
  Total            ~23 min         ?               ?
```

The `~` rows for Qwen-Image-Edit-2511 are pre-Phase-2 estimates from prior
handoffs; orchestrator should overwrite them with the freshly-measured baseline.

---

## Replication-cost rows

```
                     Trait+Impl LoC   Per-model Extraction LoC   Calendar Time to Green Parity
One-time             ~95              —                          —
Klein (Phase 1)      —                — (had param-style)        — (Phase 1 baseline)
Chroma               —                169                        ?
Qwen-Image-Edit      —                196                        ?
```

### Notes on the cost numbers

**Trait+Impl (one-time, ~95 LoC)** — `src/offload_api.rs` (~62 LoC including
docs) + `src/turbo/api.rs` (~33 LoC). Both numbers include doc comments; the
substantive code is ~25 + ~15 LoC respectively, plus the `OffloaderBlock` /
`TurboAwaited` newtypes.

**Chroma per-model extraction (169 LoC by `git diff --stat src/models/chroma_dit.rs`)**
— BLOWN OUT vs the spec's "single digits to low tens" target. The diff
includes:
- `forward_inner` body (a pure move from `forward_cached` ~115 LoC of loop
  body + post-loop norm_out/proj_out).
- `forward_with_turbo` wrapper (~35 LoC, duplicates the input-projection
  setup that lives in `forward_cached` because Chroma's projections need
  `&self.shared` and `forward_inner` is an associated function — wrapping in
  a shared helper would have added a third function).
- Two new helpers (`untranspose_block_weights_map`, `passthrough_block_weights`)
  for the bool-gate prep step.
- `double_block_forward` / `single_block_forward` converted to associated
  functions taking `cfg: &ChromaConfig` explicitly (so `forward_inner` can be
  called from a destructure-borrowed context where `self` is split into
  disjoint field borrows).
- Doc comments explaining the bool gate and the destructure pattern.

**Qwen-Image-Edit per-model extraction (196 LoC)** — same shape:
`forward_edit_with_ref_timestep_inner` body (the loop + post-loop work),
`forward_edit_with_ref_timestep_turbo` wrapper, two helpers, two block_forward
methods converted to associated functions, doc comments.

**The honest take.** The spec's "single digits to low tens" claim is for the
per-model-specific delta after the trait abstraction is in place. With the
trait approach + the destructure-borrow pattern, each model still pays ~150-200
LoC because:

1. Input projections must run through `&self.shared`, which can't easily be
   done from an associated `forward_inner`. Hence the duplicated input
   projection code in `forward_with_turbo`.
2. The bool-gate weight prep needs both `untranspose_block_weights_map` (the
   real prep) and `passthrough_block_weights` (the no-op that still builds a
   uniform `HashMap`).
3. The block_forward helpers had to lose their `&self` receiver.

A future cleanup that deduplicates the input-projection setup into a private
helper would shave ~40-60 LoC per model, bringing each closer to the target.
That cleanup is out of scope for Phase 2 v5.1 — it's a refactor pass on the
DESTINATION shape, not a Phase 2 deliverable.

---

## Methodology notes for the orchestrator

1. **Warmup**: each bench runs each binary 3 times. Discard run 1 (cold-cache).
2. **Cooldown between runs**: between baseline and turbo for the same model,
   wait until GPU edge temp < 60 °C (a 23-min Qwen-Edit run will heat-soak).
3. **Per-swap timing**: the bench scaffolds emit total wall-clock; per-swap
   numbers come from nvtx range markers (see TODO in each bench file). Wire
   nvtx in if/when needed by the report.
4. **Model load timing**: parse from each binary's stdout — both print
   `Loaded ... in {dt:.1}s` lines.
5. **Denoise/VAE/Text-Encode breakdown**: same — parse stdout. Each binary
   prints stage-level timings.
6. **For Qwen-Image-Edit**: Stage 1 (Python encode) is out of scope. Both
   binaries consume a pre-computed `qwenimage_edit_embeds.safetensors`.
7. **Sharded checkpoint caveat**: `chroma_infer_turbo` and
   `qwenimage_edit_gen_turbo` need single-file safetensors (Phase 1's
   `TurboBlockLoader` is single-file). Set `CHROMA_TURBO_SAFETENSORS` /
   `QWEN_TURBO_SAFETENSORS` to single-file checkpoints before running.
