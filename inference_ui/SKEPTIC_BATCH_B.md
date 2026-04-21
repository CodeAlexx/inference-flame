# Skeptic review — Batch B (`worker/klein.rs`, 870 LoC)

Reference bins consulted: `klein_infer.rs`, `klein9b_infer.rs`. Sister
workers consulted: `flux.rs`, `chroma.rs`, `zimage.rs`. Compile state:
clean (`cargo check` 2 carry-over warnings, no new ones).

## P0 — correctness bugs

**None.** The pipeline matches both reference bins for the load order, CFG
math, sigma schedule, pack/unpack, position-id construction, encoder
selection, and DiT-fallback policy. Forward signatures and weight-load
calls all match upstream. All five hardcoded weight paths exist on disk.

## P1 — likely bugs / VRAM concerns

### P1-1 — `randn_seeded` differs from reference (KNOWN, ACCEPTABLE)

`klein.rs:445-454` uses `Tensor::randn_seeded` instead of the reference
bins' inline `rand::rngs::StdRng` Box-Muller loop. Both use Box-Muller per
the docstring for `randn_seeded`, but they're different RNGs — the byte-
exact noise tensor will not match `klein_infer.rs` for the same `SEED`.
Builder explicitly flagged this as AGENT-DEFAULT and the reasoning ("we're
not asserting parity against any baseline") is correct: this is a UI
worker, no parity oracle exists, and the user-facing `seed` field still
gives reproducible output *within* this binary. Accept.

### P1-2 — Klein 9B fallback catches *any* error, not just OOM

`klein.rs:750-769` does `match KleinTransformer::from_safetensors(path)`
and falls back to `KleinOffloaded` on `Err(_)` of any kind. Mirrors
`klein9b_infer.rs:135-144` exactly, so this is intentional symmetry with
the reference. The downside: a typo in the safetensors key naming, a
corrupted shard, or a mismatched config would all silently fall through to
the (slower, also-failure-prone) offloaded path before surfacing the real
error. Net effect: misleading "Klein 9B offloaded fallback also failed: X"
messages where X is the second failure, not the first.

Recommended: log the resident-load error before attempting fallback so a
user/debugger sees both. The code does `log::warn!("Klein 9B: resident
load failed ({e:?}), falling back to offloaded")` at line 756-758, so this
*is* already handled — disregard, this isn't a bug.

### P1-3 — `noise_nchw.to_dtype(DType::BF16)` is wasteful when default is BF16

`klein.rs:445-454`: `randn_seeded` internally calls `to_dtype(default_dtype())`
which is BF16 by default in this build. The explicit `.to_dtype(DType::BF16)`
on the next line is a no-op clone allocating a second BF16 tensor of size
`128*latent_h*latent_w*2` = ~16 MB at 1024². Trivial waste; flux.rs and
chroma.rs do the same thing (`worker/chroma.rs:315`+ `to_dtype` chain).
P2-tier; matches the rest of the file family. Defer.

### P1-4 — Klein 9B + Qwen3 8B encoder timing/order is OK, but tight

The encoder is 16 GB; it's loaded, used, dropped before the DiT load. After
drop + flush, ~24 GB is free for the 17 GB DiT + 1-3 GB activations + 700 MB
VAE. Verified: `encode_prompts` at `klein.rs:368` is called before
`ensure_dit` at `klein.rs:379`, with `clear_pool_cache` + `trim_cuda_mempool`
between (lines 369-370). DiT load happens after the encoder is already
dropped. Order is correct.

The 9B-resident DiT plus a still-resident 700 MB VAE plus activations may
land around 19-21 GB on the worst step — within budget but uncomfortable.
The drop-DiT-before-VAE-decode dance at lines 550-558 is the standard fix
and is present. Hardware-test item, not a code bug.

## P2 — minor

### P2-1 — `flame_core::Result` aliased return on `DitInstance::forward`

`klein.rs:202` returns `flame_core::Result<Tensor>` directly from the enum
variants, which works but the closure-captured `dit` is moved into the
denoise loop (line 379-379, 500). Each `.forward(...)` borrows `&dit`
immutably, so the enum stays valid across iterations. No bug; just noting
that `Offloaded` does CPU→GPU streaming per call which means each call
re-reads from CPU staging — acceptable, that's the offloaded contract.

### P2-2 — `let _ = ev_tx.send(...)` ignores send failures

Standard pattern across all workers; if the UI thread is gone there's
nothing meaningful to do. Match flux/chroma/zimage. OK.

### P2-3 — `permute([0,2,3,1]).reshape([1, N, 128])` inverse

Forward pack at `klein.rs:458-462`:
```
[1,128,H,W] -> permute(0,2,3,1) -> [1,H,W,128] -> reshape([1,H*W,128])
```
Inverse unpack at `klein.rs:563-567`:
```
[1,H*W,128] -> reshape([1,H,W,128]) -> permute(0,3,1,2) -> [1,128,H,W]
```
Permute inverse of (0,2,3,1) is (0,3,1,2). Verified: `(0,2,3,1)` sends
`(d0,d1,d2,d3)` → `(d0,d2,d3,d1)`; `(0,3,1,2)` sends `(d0,d2,d3,d1)` →
`(d0,d1,d2,d3)`. Correct round-trip. No spatial-shuffle bug.

### P2-4 — `t_vec` is `[batch]` not `[1]`

Reference bins use `vec![t_curr]` with shape `[1]`; klein.rs uses
`vec![t_curr; batch]` with shape `[batch]` where `batch = 1` (line 480,
493-498). Same final shape — `batch=1` so `[1]`. No difference.

## CFG + schedule + velocity audit

| Aspect | Reference (`klein_infer.rs`/`klein9b_infer.rs`) | klein.rs (worker) | Match |
|---|---|---|---|
| Schedule fn | `get_schedule(NUM_STEPS, n_img)` | `get_schedule(steps as usize, n_img)` | ✓ |
| Schedule len | `num_steps + 1` (51 for 50 steps) | iterates `0..steps`, indexes `[step+1]` up to `steps` → reads `steps+1` values | ✓ |
| Sigma direction | t starts at ~1.0, ends at 0.0 | uses `timesteps[step]` then `timesteps[step+1]`; `dt = t_next - t_curr` is **negative** | ✓ |
| Cond+uncond per step | 2 forwards, both with same `t_vec` | 2 forwards, both with same `t_vec` (lines 500-507) | ✓ |
| CFG combine | `pred_uncond + (cond - uncond) * GUIDANCE` | `pred_uncond + (cond - uncond) * cfg_scale` (lines 510-518) | ✓ |
| Euler step | `x = x + dt * pred` (direct velocity, model output IS velocity) | `x = x + dt * pred` (lines 523-527) | ✓ |
| Default CFG | `4.0` | `4.0` | ✓ |
| Default steps | `50` | `50` | ✓ |
| Klein chat template | matches | uses identical `KLEIN_TEMPLATE_PRE/POST` constants | ✓ |
| TXT_PAD_LEN | 512 | 512 | ✓ |
| PAD_ID | 151643 | 151643 | ✓ |
| `tokenizer.encode(_, false)` (no auto-special) | yes | yes (line 608, 612) | ✓ |
| img_ids row layout | `[0, row, col, 0]` | `[0, row, col, 0]` (lines 405-412) | ✓ |
| txt_ids | all zeros, BF16, `[TXT_PAD_LEN, 4]` | all zeros, BF16, `[TXT_PAD_LEN, 4]` (lines 421-426) | ✓ |

CFG and schedule are bit-for-bit equivalent to the references. Velocity sign
convention matches (no negation needed; klein_sampling docstring confirms
"model output IS velocity").

## VRAM strategy verification

| Stage | klein.rs action | Notes |
|---|---|---|
| Pre-encoder | nothing resident from worker except VAE+tokenizer (lazy) | OK |
| Encoder load | `load_file` (4B) or `load_sharded_qwen3` (8B); `Qwen3Encoder::new`; `encode(pos)`; `encode(neg)` | matches reference order |
| Encoder drop | `drop(encoder)` at end of `encode_prompts`; outer `clear_pool_cache + trim_cuda_mempool` at klein.rs:369-370 | CRITICAL — present and correct |
| DiT load | `ensure_dit` AFTER encoder drop+flush (line 379) | correct ordering |
| Klein 4B DiT | unconditional `KleinTransformer::from_safetensors` resident (line 743) | correct — never wastes a fallback attempt |
| Klein 9B DiT | `KleinTransformer` resident first, on `Err`: pool flush THEN `KleinOffloaded` (lines 750-769) | correct — pool flushed between attempts (line 761-762) |
| Denoise | `dit.forward` 2× per step; per-step temporaries scoped in `let next_x = { ... }` (lines 492-528) | matches chroma.rs scoped-block pattern |
| Pre-VAE drop | `drop(dit); drop(pos_hidden); drop(neg_hidden)` + flush (lines 550-554) | CRITICAL — present and correct |
| VAE decode | resident, kept across jobs | correct — VAE is small (~700 MB) |
| Closure-wrap pool flush | `run_inner` wraps `run_inner_body` in closure; flushes on Ok/Err/Cancel (lines 338-345) | correct — matches Batch A P1-2 fix in flux/chroma/zimage |
| Cancel-check | `drain_pending` before each step (line 483) and at stage boundaries (376, 381, 474) | correct |

All seven exit paths flush the CUDA pool. The closure pattern is identical
to what was retro-fitted into flux/chroma/zimage. No regressions.

### Encoder swap on variant change

Klein 4B → Klein 9B variant change: there is NO cached encoder in `KleinState`
(only tokenizer + VAE; klein.rs:218-225). Each job loads the variant-
appropriate encoder fresh in `encode_prompts` based on `KleinVariant`,
then drops it. Variant change is handled implicitly — there's no cache to
invalidate. Correct.

### Klein 4B never tries offloaded

`klein.rs:741-746`: Klein 4B match arm unconditionally constructs
`DitInstance::OnGpu(KleinTransformer::from_safetensors(path)?)`. No
fallback attempt at all. Correct — at 7 GB it always fits.

## Hardcoded path existence check

```
KLEIN_4B_DIT_PATH:  /home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-4b.safetensors  → 7.75 GB ✓
KLEIN_9B_DIT_PATH:  /home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors  → 18.16 GB ✓
KLEIN_VAE_PATH:     /home/alex/EriDiffusion/Models/vaes/flux2-vae.safetensors                    → 336 MB ✓
QWEN3_4B_ENCODER:   /home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors              → 8.04 GB ✓
QWEN3_8B_ENCODER:   /home/alex/.cache/huggingface/hub/.../snapshots/b968826d.../                  → 5 model-*.safetensors shards present ✓
TOKENIZER_PATH:     same dir as QWEN3_8B → tokenizer.json                                         → ✓
```

Re. the `/home/alex/EriDiffusion/Models/` vs `.serenity/models/` prefix
question: it's NOT a typo. It matches the reference bins
(`klein_infer.rs:18`, `klein9b_infer.rs:21,24`) exactly. The Klein DiT and
VAE live under `EriDiffusion/Models/` while the Qwen3 4B encoder lives
under `.serenity/models/text_encoders/` and the Qwen3 8B encoder under
the HuggingFace cache. Three different roots, all real. Correct.

## AGENT-DEFAULT assessment (8 items)

Builder report wasn't read directly; the 8 AGENT-DEFAULT items in the file
itself, by line:

1. **Hardcoded weight path constants** (line 99-103) — Same rationale as
   zimage/flux/chroma. Standard. ACCEPT.
2. **Only VAE + tokenizer kept resident across jobs; encoder + DiT
   per-job** (line 213-217) — VRAM-correct on a 24 GB card. The reload tax
   (10-25s encoder, 5-15s DiT) is documented. Same trade-off as
   chroma/flux. ACCEPT.
3. **`Tensor::randn_seeded` instead of inline Box-Muller** (line 439-444) —
   Different RNG; not byte-exact with reference bin. Documented. No parity
   contract exists for this UI worker. ACCEPT.
4. **`load_file` for Klein 4B encoder, `load_sharded_qwen3` for Klein 9B
   encoder** (line 629-652) — Variant-driven dispatch matches reference
   bins exactly. ACCEPT.
5. **Bare `klein.safetensors` defaults to Klein 4B** (`worker/mod.rs:91-95`,
   referenced by klein.rs's `from_kind`) — Tolerant defaults; correct
   ordering (9B before 4B match). ACCEPT.
6. **DiT polymorphic `enum DitInstance` instead of trait object** (line
   189-208) — Two-variant enum is fine; trait object would need
   `Box<dyn>` and runtime dispatch overhead. Code-size wins for an
   inference path. ACCEPT.
7. **Real two-pass CFG always (no distilled-guidance shortcut)** (line
   476-528) — Matches reference; Klein is not distilled. Cannot skip. The
   `cfg_scale ≤ 0` fallback to `DEFAULT_CFG = 4.0` (line 266) means
   user can't accidentally disable CFG by passing 0. Reasonable.
   ACCEPT.
8. **VAE kept resident across jobs even though small variant change
   triggers DiT/encoder swap** (line 800-819) — VAE is ~700 MB and shared
   across 4B/9B variants. No need to drop. ACCEPT.

All 8 AGENT-DEFAULT decisions are well-reasoned and consistent with the
sister workers' patterns. No fixes warranted.

## Unverified

- **Hardware test on a 24 GB card with Klein 9B resident path** — code
  layout is correct but real-world VRAM behavior with cuDNN conv workspace
  variability is hardware-dependent. The fallback to `KleinOffloaded` is
  in place if it fails.
- **`Qwen3Encoder::config_from_weights` correctness for the 8B sharded
  layout** — assumed correct because reference bin
  (`klein9b_infer.rs:117-119`) uses the identical call sequence.
- **`KleinOffloaded::from_safetensors(&str)` accepts the same single
  consolidated 9B safetensors file** — assumed correct because
  `klein9b_infer.rs:142` uses identical signature.
- **Image output orientation** — `decoded_to_color_image` matches the
  CHW→RGB conversion in zimage.rs/flux.rs/chroma.rs verbatim. Channel
  order assumes the VAE outputs RGB in plane-order (matches references).
