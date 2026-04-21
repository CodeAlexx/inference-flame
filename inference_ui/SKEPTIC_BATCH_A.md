# Skeptic Review — Batch A (worker/flux.rs + worker/chroma.rs)

Reviewed against reference bins `inference-flame/src/bin/flux1_infer.rs` and
`chroma_infer.rs`, the proven `worker/zimage.rs`, and library
`sampling/flux1_sampling.rs` (the source of truth for `get_schedule`,
`pack_latent`, `unpack_latent`, and the BFL Euler convention
`img = img + (t_prev - t_curr) * pred`).

## P0 — correctness bugs

**None found.** The schedule call args, pack/unpack symmetry, Euler velocity
convention, FLUX distilled-guidance wiring, Chroma two-pass CFG composition,
and the T5/DiT drop dance all match the reference bins.

## P1 — likely bugs / VRAM concerns

### 1. Chroma 24 GB OOM risk is real and undermitigated
`chroma_infer.rs:69-77` ships with a hard runtime check that `chroma_infer`
**will OOM** on a 24 GB card unless `CHROMA_INFER_FORCE=1` is set, and
explicitly recommends the two-stage `chroma_encode + chroma_gen` pattern.

`worker/chroma.rs` reproduces the single-process flow with a drop-T5 +
`clear_pool_cache` + `trim_cuda_mempool` between encode and DiT load
(chroma.rs:216-218). That same dance works in `worker/flux.rs` because the
FLUX DiT is ~12 GB; for Chroma it is ~14-17 GB so the headroom is
significantly tighter even after the flush.

The mitigation is identical to FLUX's, but the bin explicitly says it isn't
enough for Chroma. There is no fallback path here — first cold run on a real
24 GB card may OOM. AGENT-DEFAULT in the file header acknowledges this is
acceptable, but the reference bin's behavior says otherwise. Real test on
hardware is needed before declaring this safe.

**Recommendation**: keep as-is for Phase 5b but plan a real two-stage path
(parallel to chroma_encode/chroma_gen) for Phase 5c if first run OOMs. Until
hardware-tested, treat this as "may not work first try."

### 2. Cancel mid-job leaks DiT (and possibly T5 hidden) until next job
On `RunError::Cancelled` the function returns from `run_inner` with `dit`
(and on FLUX, the loaded `t5_hidden`/`pos_hidden`) still alive in the stack
frame. Drop happens automatically as the frame unwinds, BUT the post-drop
`clear_pool_cache` + `trim_cuda_mempool` calls are skipped — they only run
on the success path (flux.rs:430-431, chroma.rs:363-364).

If the user cancels mid-denoise then immediately starts a new FLUX job, the
new job's `T5Encoder::load` call will be racing against a 12 GB DiT
allocation that hasn't been pool-flushed yet. Could OOM the T5 load on a
24 GB card.

**Mitigation**: wrap the body in a struct with `Drop` impl that always
flushes, or move the flush into a `finally`-style closure. Acceptable to
defer to Phase 5c, but flag this for the next batch.

### 3. FLUX `clip_pooled` may be invalidated when `state.qwen3 = None` analog isn't present — actually OK, just verifying
Unlike `zimage.rs` which drops the encoder mid-flight (line 349), `flux.rs`
keeps CLIP-L resident for the whole pipeline (CLIP only ~150 MB). `clip_pooled`
is captured into a local Tensor and the borrow on `clip` is released after
`encode` returns. No use-after-drop risk.

## P2 — minor

### 4. Unused import in flux.rs
`flux.rs:54` imports `std::collections::HashMap` solely for the CLIP weights
cast in `ensure_clip` (line 474). Used. **Not actually a bug.** Withdrawn.

### 5. Chroma drops `cond_hidden`/`uncond_hidden` after DiT drop, not before
chroma.rs:360-362 drops DiT first, then T5 hiddens. T5 hiddens are
`[1, 512, 4096]` BF16 ≈ 4 MB each — negligible. Order doesn't matter for VRAM.

### 6. Empty negative prompt edge case in Chroma
`tokenize_t5("")` will hit the tokenizer path and produce `[0]` BOS-only or
similar, then pad to 512 with 0. Actual encoded uncond hidden will be the T5
embedding of an all-pad sequence. This matches chroma_infer.rs:110-111
behavior — same risk as the bin, no regression.

### 7. Schedule indexing
`flux.rs:348` and `chroma.rs:287` use `timesteps[steps as usize]`. `get_schedule`
returns `steps + 1` items (verified flux1_sampling.rs:67), so the last index
is `steps` — valid. Good.

### 8. ETA when steps=1 division
`elapsed / step1 as f32` with `step1=1` → fine. `(steps - step1) = 0` → 0 ETA.
No divide-by-zero. Good.

## Schedule + CFG + velocity per model

### FLUX 1 Dev (worker/flux.rs)
- **Schedule**: `get_schedule(steps, n_img, 0.5, 1.15, true)` — exact match to
  flux1_infer.rs:209.
- **CFG**: distilled guidance via `guidance_vec` (single forward per step).
  `cfg_scale ≤ 0` → 3.5 default (matches BFL reference). `negative` ignored.
  Reinterpretation is documented inline (flux.rs:14-19, 102-104). Correct.
- **Velocity**: `x = x + dt * pred` where `dt = t_prev - t_curr` (negative
  for descending schedule). Matches `flux1_sampling.rs::flux1_denoise`
  line 272: `img = img.add(&pred.mul_scalar(t_prev - t_curr)?)?`. Inline
  loop is a faithful reimplementation of the helper, just with cancel-check
  + Progress events woven in.
- **Pack/unpack**: `pack_latent(noise_nchw, &device)` → `(packed, img_ids)`
  and `unpack_latent(&x, height, width)`. Symmetric. Matches flux1_infer.rs.
- **`txt_ids`**: `zeros [T5_SEQ_LEN=512, 3]` BF16. Matches flux1_infer.rs:184.
- **`guidance_vec`**: BF16 `[batch=1]`. Matches flux1_denoise's vec build.

### Chroma (worker/chroma.rs)
- **Schedule**: same `get_schedule(steps, n_img, 0.5, 1.15, true)` — match
  to chroma_infer.rs:181.
- **CFG**: real two-pass.
  - Both `cap_feats` and `cap_feats_uncond` encoded by T5 inside the same
    scope (chroma.rs:198-211).
  - Per-step inner runs `dit.forward(cond)` then `dit.forward(uncond)`
    sequentially with `[1, ...]` shapes — NOT batched into `[2, ...]`
    (matches chroma_infer.rs:206-219 — Chroma's `forward` takes `[1, ...]`,
    confirmed by `models/chroma_dit.rs:620`).
  - Composition: `pred = uncond + cfg * (cond - uncond)` (chroma.rs:326-335)
    — matches chroma_infer.rs:222-224 byte-for-byte. Correct.
- **Velocity**: same Euler `x = x + dt * pred`, with `dt = t_next - t_curr`.
  Matches chroma_infer.rs:228.
- **Pack/unpack**: same FLUX helpers. Symmetric.
- **`txt_ids`**: same zeros `[512, 3]` BF16. Matches.
- **No `guidance_vec`, no `pooled`**: correct for Chroma — modulation comes
  from the model's internal `distilled_guidance_layer`.

## VRAM strategy verification per model

### FLUX 1 Dev
| Stage | Resident peak | Order in flux.rs | Matches bin? |
|---|---|---|---|
| 1. Load CLIP | 0.15 GB | `ensure_clip` (lazy) | bin loads fresh per run; UI caches — safe |
| 2. Encode CLIP → keep pooled | 0.15 GB | line 232 | yes |
| 3. Load T5 + encode + drop in scope | +9.4 GB → 0.15 GB | line 243-264 | yes (`flux1_infer.rs:115-127`) |
| 4. `clear_pool_cache` + `trim_cuda_mempool` | 0.15 GB | line 265-266 | extra (bin relies on scope drop) — defensive plus |
| 5. Load DiT (BlockOffloader) | +12-14 GB | line 279 (`Flux1DiT::load`) | yes — uses BlockOffloader path (verified flux1_dit.rs:139) |
| 6. Denoise | +2-3 GB act | inline loop | yes |
| 7. Drop DiT + flush | 0.15 GB + VAE | line 429-431 | yes (`flux1_infer.rs:244`, also matches `zimage.rs:445`) |
| 8. Load VAE (lazy) + decode | +0.3 GB + workspace | line 440-447 | yes |

**FLUX VRAM strategy: correct.** The drop-and-flush dance matches both the
bin and the validated zimage.rs pattern.

### Chroma
| Stage | Resident peak | Order in chroma.rs | Matches bin? |
|---|---|---|---|
| 1. Load T5 + encode cond + uncond + drop in scope | +9.4 GB → 0 | line 192-215 | yes (matches chroma_infer.rs:102-117 with extra uncond pass) |
| 2. `clear_pool_cache` + `trim_cuda_mempool` | 0 | line 216-217 | extra — defensive plus |
| 3. Load Chroma DiT shards (BlockOffloader) | +14-17 GB | line 232 (`ChromaDit::load(CHROMA_DIT_SHARDS, &device)`) | yes — verified chroma_dit.rs:226 takes `&[&str]` slice and uses BlockOffloader |
| 4. Two-pass denoise | +2-3 GB act | inline loop | yes |
| 5. Drop DiT + flush | 0 + VAE | line 360-364 | yes |
| 6. Load VAE (lazy) + decode | +0.3 GB + workspace | line 373-380 | yes |

**Chroma VRAM strategy: correct in pattern, risky in headroom.** The
reference bin warns this single-process arrangement OOMs on a 24 GB card; the
worker version uses the same dance plus an explicit pool flush, but real
hardware test required.

## Dispatch ordering audit

`mod.rs::from_model_string` (mod.rs:65-78):
1. `z-image-turbo` / `zimage-turbo` → ZImageTurbo
2. `z-image-base` / `zimage-base` → ZImageBase
3. `chroma` → Chroma  ← BEFORE flux
4. `flux` → FluxDev
5. else → Mock

`IMAGE_MODELS` (sections/model.rs:21-32) entries:
- `z-image-base.safetensors` → ZImageBase ✓
- `z-image-turbo.safetensors` → ZImageTurbo ✓
- `flux1-dev.safetensors` → FluxDev ✓ (matches "flux")
- `flux1-schnell.safetensors` → FluxDev ✓ (matches "flux", same DiT shape — comment in mod.rs:62 acknowledges schedule may differ)
- `chroma.safetensors` → Chroma ✓
- everything else → Mock

**Dispatch is correct.** Chroma-before-FLUX ordering is defensive (matches a
hypothetical "flux-chroma-mix" filename to Chroma, intentional per builder's
note). No filename in the current `IMAGE_MODELS` list collides.

`mock.rs::run` outer-loop arms (mock.rs:90-147) wire all four kinds correctly,
each with lazy `*State::new()` and CUDA-failure fallback emitting `Failed`
without killing the worker thread. Same pattern as ZImageBase/Turbo arm.
Started/Progress/Done/Failed event protocol matches across all three real
backends.

## AGENT-DEFAULT assessment (9 items)

The builder's report enumerates 9 AGENT-DEFAULTs. Reading them from the file
headers:

1. **Hardcoded weight paths (flux.rs:79-83, chroma.rs:53-60)** — ACCEPTABLE.
   Same pattern as zimage.rs; mirrors what reference bins do. Migration
   target documented.

2. **Reinterpret `cfg` slider as distilled guidance for FLUX (flux.rs:155-161)**
   — ACCEPTABLE. Documented in the file header; correct semantics. Future
   UI work can swap label per ModelKind.

3. **Default guidance 3.5 for FLUX, 4.0 for Chroma, 20/40 default steps**
   — ACCEPTABLE. Match BFL/Chroma reference values exactly.

4. **Drop T5 every job, pay reload tax (~10s) (flux.rs:39-42)** — ACCEPTABLE
   with caveat. Safe choice over keeping T5 resident (which would risk OOM
   at decode). Documented trade-off.

5. **Drop DiT every job, pay reload tax (8-15s) (flux.rs:43)** — ACCEPTABLE.
   Same dance as zimage.rs:454 (which also drops DiT before VAE).

6. **`Tensor::randn_seeded` instead of Box-Muller (flux.rs:310-315)** —
   ACCEPTABLE for a UI worker. Statistically equivalent, byte-different.
   Not asserting parity. Both flux.rs and chroma.rs comments call this out
   honestly.

7. **Both `cond` and `uncond` always encoded for Chroma even at cfg=1
   (chroma.rs:206-211)** — ACCEPTABLE. Cost is a single T5 forward (~few
   hundred ms), negligible vs. doubled denoise. Code is simpler.

8. **`forward_cached` API not used for Chroma (chroma.rs:28-31)** —
   ACCEPTABLE deferral. Documented as a future ~15% speedup. Premature
   optimization to wire up before the basic path is hardware-tested.

9. **Path constants for Chroma point at HF cache snapshots
   (chroma.rs:65-70)** — ACCEPTABLE but FRAGILE. The HF snapshot revision
   string `0e0c60ece1e82b17cb7f77342d765ba5024c40c0` is hardcoded; a model
   re-download or HF cache eviction would break this. Worth flagging:
   prefer `~/models/...` or a stable symlink in a future phase. Not a P1
   because the path is verified to exist on the dev box (per builder note).

## Unverified

- **Hardware behavior**: cannot run; cannot confirm Chroma single-process
  flow actually fits in 24 GB despite the bin's own OOM warning.
- **Cold-load times**: T5 ~10s, DiT ~8-15s claims are not measured here;
  trust the reference bins' anecdotal numbers.
- **`ChromaDit::load` BlockOffloader behavior**: confirmed signature and
  that it accepts a `&[&str]` slice, but did not trace the loader to verify
  it actually uses BlockOffloader pinned-RAM (vs. naive resident). Builder
  claims it does and the bin uses it — taking that on trust.
- **`Flux1DiT::load` similarly confirmed by signature, not by tracing the
  body.**
- **Compilation hygiene**: builder reports clean compile w/ 2 pre-existing
  warnings; not re-verified by re-building per task instruction.
- **Empty-prompt T5 encode behavior**: behavior of `tokenize_t5("")` is the
  same as the reference bin (which is itself untested in that case per the
  bin's `WARNING` header). No regression vs. the bin baseline.
- **Schedule monotonicity for very small step counts (e.g. steps=1)**: the
  schedule helper has a unit test for steps=20 only. flux.rs/chroma.rs do
  not clamp steps to a minimum; if `job.steps == 1` the loop runs once.
  Assumed safe based on `flux1_sampling.rs` math but not exhaustively
  verified.
