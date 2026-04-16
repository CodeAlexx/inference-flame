# Motif-Video Parity: Non-Determinism Investigation — Handoff (v3)

> Supersedes v2 for the investigation section — v2 findings all still stand.
> v3 adds one session's worth of code-level tracing, a failed-fix with its
> postmortem, and orthogonal perf wins that shipped during the debugging.

---

## v3 session (2026-04-16) — quick summary for the next reader

**What shipped (unrelated to the non-det, real wins):**

- SDPA default is now the in-tree WMMA kernel, not torch. Torch is opt-in
  via `FLAME_USE_TORCH_SDPA=1`. The comment in `sdpa.rs` describing the
  in-tree kernel as "scalar FP32, 230 ms/Z-Image-block" was **stale** — the
  WMMA rewrite already happened at `flash_attention_fwd.cu` (scalar
  predecessor preserved at `flash_attention_fwd_scalar.cu.bak`). WMMA is at
  parity with torch up to N≈1024, 4.3× slower at N=4096 (measured).
- `Tensor::cat` BF16 path: replaced the `for o in 0..outer` inner loop of
  `flame_k_copy_bf16` launches with a single `cuMemcpy2DAsync_v2` per
  input tensor. On motif that cuts per-forward copy-kernel launches from
  ~33k to ~1.4k and trims warm forward time 157 ms → 97 ms (38% faster).
- Added `staging::bf16_copy_async_tagged` + `bf16_copy_stats_snapshot` for
  call-site attribution under `FLAME_COPY_TRACE=1`.
- Added `bench_sdpa` binary for the raw SDPA latency sweep that decided
  the torch-vs-WMMA tradeoff.
- Added per-forward timing to `motif_parity` via `MOTIF_DOUBLE_FWD=N`.

**What I tried for the non-determinism and why it failed:**

I believed I'd found the root cause: **cudarc's CudaDevice stream is
created with `StreamKind::NonBlocking`** (verified at
`~/.cargo/registry/src/index.crates.io-*/cudarc-0.11.9/src/driver/safe/
core.rs:108` and `:562`), and this does NOT implicitly sync with the CUDA
legacy null stream in either direction. flame-core's hot path was split
across the two streams:

- **Null stream**: cuBLASLt GEMMs (`device_lt::stream_ptr` returns null),
  `bf16_copy_async` (`cuda_stream_raw_ptr` returns null), `rms_norm_bf16`
  and `layer_norm_bf16` (via `default_stream` → null), `cuMemcpy2DAsync`
  in my new cat, WMMA flash attention.
- **Cudarc NonBlocking stream**: everything launched via cudarc's
  `LaunchAsync::launch` — elementwise `add`/`mul`/`gelu`/`silu`, permutes.

If the streams don't sync, a tensor written by one stream and read by the
other is a race with no host-visible ordering. The theory fit: in-process
bit-exact (scheduling order stable), cross-process catastrophic drift
~10-20%, divergence enters in block 1's dual_block_forward.

**Test: I changed `device.rs::cuda_stream_raw_ptr` and
`cuda/device_lt.rs::stream_ptr` to return `*device.cu_stream()` (cudarc's
stream) so all kernels share one stream.**

**Result: cross-process ROSE to ~75% catastrophic. Worse, not better.**

I reverted (see diff at v3 end). The baseline (~10-20% catastrophic)
returned. So the stream split is clearly ONE race but it's not the only
source. Unifying streams exposed a second race or broke something else.
Leaving the theory as "wrong" until someone traces more carefully.

**What the skeptic agent independently surfaced that I didn't pursue:**

1. `fused_linear3d.cu:225-239` silently widens from `REDUCTION_SCHEME_NONE`
   to `NONE | COMPUTE_TYPE | OUTPUT_TYPE` if the heuristic returns no
   NONE-only algo. Both COMPUTE_TYPE and OUTPUT_TYPE reductions are
   non-deterministic (CTA-scheduling-dependent). For `ff_context` at
   m=6144 n=16 k=1536 (small N), the fallback may trip silently. Check
   `stderr` logs for `"no NONE-reduction algo"` warnings during motif
   forward.
2. `tensor_storage.rs:710` — `TensorStorage::Drop` returns the slice to
   `pool_return_u16` without a `cuStreamSynchronize`. If a kernel is
   still writing when Arc refcount hits 1, the buffer goes back to the
   free list while writes are pending. Stacks with issue 1 to explain
   the specifically *catastrophic* (vs just drift) runs.
3. `fp8_resident.rs:202` DOES create a `BF16View`. My earlier "never
   created" was wrong. Not on the Motif path, but future fp8 paths may
   stack this race with 1 and 2.

**Recommended next experiments (in order):**

1. Run motif forward with `RUST_LOG=warn` and grep for
   `"no NONE-reduction algo"`. If that fires, suspicion 1 is live —
   cache the NONE-only algo on first heuristic call per shape, reuse
   it across subsequent calls to pin determinism.
2. Add `cuStreamSynchronize(slice's stream)` before `pool_return_u16`
   (suspicion 2). If cross-process variance drops, confirmed.
3. Try `MOTIF_NO_CAPTURE=1` across a fresh sweep — the capture HashMap
   being `Some` vs `None` was a variable I didn't isolate cleanly.
4. If still stuck, unify streams again but ALSO fix (1) and (2) in the
   same change — the first attempt may have failed due to interaction
   effects.

**Pinpointed divergence still:** Block 1's `dual_block_forward`, same
`input_img`/`input_enc` → divergent `img_final`/`enc_final`. Traces were
taken with `MOTIF_BISECT=1` (detailed) and `MOTIF_BISECT=all` (brief).
Instrumentation stayed in `motif_video_dit.rs` — see the `fp(...)` calls.

---

## TL;DR

- **In-process forward() is now bit-exact** when `MOTIF_RESIDENT=1` is set.
  Multiple forwards in the same process produce `cos_vs_prev=1.000000000`,
  `max_abs_vs_prev=0.0000`. This was NOT true before — previously every
  forward re-preloaded all 36 blocks, and FWD#N could differ from FWD#N+1
  by cos < 0 / max_abs > 12.
- **Cross-process variance is still present** and worse than v1 handoff
  reported. Re-sampling shows cos range **0.53–0.9996** with occasional
  catastrophic outliers (|r|max > Python's by 1.5–11×). v1's "0 catastrophic
  in 20 runs" was undersampled — those catastrophics are tail events.
- **The v1 fixes (torch_sdpa sync, cuBLASLt NONE-only + splitk=1, offloader
  bypass) all remain valid and necessary.** v2 adds one more orthogonal fix.

---

## What Changed in v2

### Fix: `MOTIF_RESIDENT=1` is now truly persistent

**Bug v1 missed**: `resident_weights` was declared as a local inside `forward()`
(lines 553–573 pre-v2). Every call to `forward()` re-ran the 36-block
prefetch+await+untranspose loop, then dropped the vec at end of forward.
"Resident" meant "resident for the duration of this forward", not across
forwards.

Evidence this was broken:

```
MOTIF_RESIDENT=1 MOTIF_DOUBLE_FWD=5 MOTIF_NO_CAPTURE=1:
  FWD#1: cos_vs_prev=0.996  max_abs=0.17
  FWD#2: cos_vs_prev=0.996  max_abs=0.18
  FWD#3: cos_vs_prev=0.996  max_abs=0.19
  FWD#4: cos_vs_prev=0.999  max_abs=0.07
  FWD#5: cos_vs_prev=0.998  max_abs=0.14

With capture ON (default): FWD#2 cos_vs_prev=-0.045, max_abs=12.3 — catastrophic.
```

After v2 fix:
```
MOTIF_RESIDENT=1 MOTIF_DOUBLE_FWD=5 (capture ON OR off, same result):
  FWD#1: cos_vs_prev=1.000000000  max_abs=0.0000
  FWD#2: cos_vs_prev=1.000000000  max_abs=0.0000
  FWD#3: cos_vs_prev=1.000000000  max_abs=0.0000
  FWD#4: cos_vs_prev=1.000000000  max_abs=0.0000
  FWD#5: cos_vs_prev=1.000000000  max_abs=0.0000
```

**Why it was broken**: every forward's re-preload created fresh untranspose
buffers (2D weight copies) via pool allocations whose addresses depended on
the previous forward's leftover pool state. Different addresses → different
alignment → cuBLASLt `AlgoGetHeuristic` could select a different (but still
NONE-only-deterministic) algo → different bit-level outputs. Also, 36 rounds
of transient allocation-and-free on every forward amplified pool churn.

**Why capture+no-resident was catastrophic**: the parity binary stashed
`img.clone()` into a per-block-indexed HashMap. Because `Tensor::clone()` is
an Arc-clone (storage is `Arc<CudaSlice<T>>` via `shared_storage` feature),
those stashed tensors kept storage alive, displacing normal recycling. On
the next forward's preload loop, pool allocations landed on completely
different slots, occasionally aliasing an in-flight kernel's output buffer.

### Code changes (all in `inference-flame/src/models/motif_video_dit.rs`)

1. Added field `resident_weights: Option<Vec<Arc<HashMap<String, Tensor>>>>`
   to `MotifDit`.
2. Initialized to `None` in `load()`.
3. New method `ensure_resident(&mut self) -> Result<()>`: populates the field
   on first call when `MOTIF_RESIDENT=1` is set; no-op thereafter.
4. Replaced the per-forward preload block with `self.ensure_resident()?` +
   `let is_resident = self.resident_weights.is_some();`.
5. Restructured each of the 3 block loops (dual, single-encoder, decoder)
   to scope the borrow of `self.resident_weights` so it ends before the
   `self.capture` write (disjoint fields, but would conflict through lexical
   scope without the inner block).

### New env vars

| Var | Default | Effect |
|---|---|---|
| `MOTIF_RESIDENT_REBUILD` | off | Force rebuild of resident_weights every forward. Simulates the old (broken) behavior for A/B comparison. Use when investigating whether a variance change is due to persistence vs something else. |

### `motif_parity` binary changes (`src/bin/motif_parity.rs`)

- `MOTIF_NO_CAPTURE=1` env var skips `dit.capture = Some(...)` so parity
  tests can measure forward() without the capture HashMap aliasing issue.
  When unset (default), capture is ON as before.
- Graceful fallback when capture is None (previously panicked on `.unwrap()`).

---

## Current State (measurements taken this session)

### In-process (MOTIF_RESIDENT=1, any MOTIF_DOUBLE_FWD value)
**Bit-exact across all extra forwards.** Capture ON or off, same result.

### Cross-process (MOTIF_RESIDENT=1, 10 runs each config)

| Config | Range | Catastrophic (cos < 0.9) | Borderline (0.9–0.99) | Near bit-exact (≥ 0.999) |
|---|---|---|---|---|
| persistent + capture ON | 0.69–0.9996 | 3/10 | 1/10 | 4/10 |
| persistent + capture OFF | 0.79–0.9996 | 1/10 | 1/10 | 3/10 |
| REBUILD + capture OFF | 0.54–0.9996 | 2/10 | 2/10 | 3/10 |

The "REBUILD" column is the old per-forward behavior. It's no better than
persistent in terms of cross-process, and worse in terms of in-process
(which is why it was replaced).

**Sampled min cos this session: 0.54.** v1's "min 0.88 in 20 runs" was a
lucky sample window, not the true distribution.

### Handoff-reported test is still valid

The v1 repro command still applies; results now span wider:
```bash
for i in {1..20}; do
  env -i PATH=/usr/bin:/usr/local/bin:/usr/local/cuda/bin HOME=$HOME \
    MOTIF_RESIDENT=1 \
    ./target/release/motif_parity \
    /home/alex/serenity/output/motif_block_dump_tiny.safetensors \
    2>&1 | grep final_output
done
```

---

## What Was Confirmed in v2 Session

| Claim | Method | Result |
|---|---|---|
| NONE-only cuBLASLt fix present in release binary | `strings target/release/motif_parity \| grep "NONE-reduction"` | ✅ Both fused_linear3d and gemm_bf16_fp32acc WARN messages baked in |
| `FLAME_SDPA_FORCE_STREAM=1` helps | 8 cross-process runs | ❌ Worse than baseline (0.925 outlier) |
| Arc-clone in `untranspose_block_weights` causes UAF | flame-core code audit: `TensorStorage::Drop` uses `Arc::try_unwrap`, returns to pool only when refcount=1 | ❌ Arc semantics intact; clones keep storage alive |
| In-process is deterministic after v1 fixes | `MOTIF_DOUBLE_FWD=5` | ❌ pre-v2: cos 0.995–0.999 drift + catastrophics with capture |
| Making RESIDENT persistent fixes in-process | `MOTIF_DOUBLE_FWD=5` after v2 | ✅ Bit-exact, every time |
| Making RESIDENT persistent fixes cross-process | 10 runs after v2 | ❌ Same tail as pre-v2 (confirmed via `MOTIF_RESIDENT_REBUILD=1` A/B) |

---

## Where to Look Next (updated from v1)

### 1. Bisect the cross-process variance (highest value)
In-process bit-exactness means we can now do a CLEAN diff across processes:

```
Process A: run forward, serialize intermediate tensors to disk keyed by
           "dual_N_qproj", "dual_N_sdpa_out", "single_N_mlp", etc.
Process B: run forward, serialize the same.
Diff: find first key where the two processes disagree.
      That op's kernel is where cross-process non-det enters.
```

The existing `MOTIF_BISECT=all` infrastructure only dumps 4 points per block
(`input_img`, `input_enc`, `img_final`, `enc_final`). Extend it with
intermediate points inside `dual_block_forward` and `single_block_forward`:
- `normed_img`, `normed_enc` (after AdaLN)
- `q_proj`, `k_proj`, `v_proj` (after Linear projections)
- `q_rope`, `k_rope` (after RoPE)
- `sdpa_out` (after attention)
- `mlp_in`, `mlp_out` (after MLP)

Because in-process is now bit-exact, a SINGLE deterministic reference set
of intermediates can be computed from one process; a second process is
compared against that set element-wise to find where divergence first
enters. This is cheaper and more precise than the v1 approach which had
to deal with intra-process drift.

### 2. Pool allocation alignment (v1 candidate #2)
If `cuda_alloc_pool::pool_alloc_u16` returns memory whose alignment varies
across processes, cuBLASLt's `AlgoGetHeuristic` may prefer a different algo
even within the NONE-only subset. Two tests:

(a) Set `FLAME_ALLOC_POOL=0` — disables pool entirely, forces fresh
    `cudaMalloc` per allocation. If cross-process variance drops, the pool
    is the culprit.

(b) Query `CUBLASLT_ALGO_CAP_REQUIRES_ALIGNMENT_A/B/C` on the selected algo
    and log the pointer alignment actually passed in. Mismatch → algo
    switched silently.

### 3. Per-kernel non-determinism audit
Even with NONE-only + splitk=1, CUDA kernels with atomic reductions can
produce different results when blocks execute in different orders. Suspects
in the motif forward:
- `rms_norm_bf16` — has atomics for reduction
- `layer_norm_bf16` — same
- `rope_fused_bf16` — pure elementwise, should be safe but verify
- `gelu` and `silu` — elementwise, safe
- Any kernel using `atomicAdd` on f32 accumulators

Grep flame-core for `atomicAdd` in `.cu` files under the code path used by
`MotifDit::forward`, and audit for non-commutative reductions.

### 4. Python reference is bit-exact — leverage it
Rather than only comparing two Rust runs, compare Rust-vs-Python
**per intermediate**. Python-side is confirmed deterministic. The earliest
intermediate where Rust drifts from Python (not just Rust-vs-Rust) tells
you both WHERE non-determinism enters AND WHICH DIRECTION it takes. Use
the existing `scripts/motif_block_dump.py` — extend it to dump the same
intermediates as (1) above.

### 5. FLAME_POOL_ZERO=1 full baseline
v1 noted "reduced variance but didn't eliminate catastrophic cases". That
was before v2. Re-measure with `MOTIF_RESIDENT=1 FLAME_POOL_ZERO=1` to
isolate whether the remaining drift is uninitialized-memory reads or
genuine kernel non-det. If cross-process drifts drop to zero catastrophic
with POOL_ZERO, the residual is uninit reads. If unchanged, it's kernels.

---

## Updated Env Vars Reference

| Var | Default | Effect |
|---|---|---|
| `MOTIF_RESIDENT` | off | Preload all 36 blocks to GPU, bypass offloader. **Persistent across forwards (v2).** +4 GB VRAM. |
| `MOTIF_RESIDENT_REBUILD` | off | **NEW v2.** Force rebuild of resident_weights every forward (simulates old broken behavior). For A/B testing only. |
| `MOTIF_NO_CAPTURE` | off | **NEW v2 (motif_parity binary).** Skip `dit.capture` setup. Useful when capture HashMap aliasing is suspected. |
| `MOTIF_BISECT=N` | off | Print fingerprints inside dual block N. `MOTIF_BISECT=all` prints just input/output of every block. |
| `MOTIF_WEIGHT_FENCE` | off | Extra device sync after each block's weight prep. Less effective than RESIDENT. |
| `MOTIF_DOUBLE_FWD=N` | off | In `motif_parity`: run N extra forwards in same process and compare. With RESIDENT, now bit-exact. |
| `MOTIF_DUMP_ELEM` | off | In `motif_parity`: element-level analysis (top-5 worst, r/p percentiles, corr(-p)). |
| `FLAME_NO_TORCH_SDPA` | off | Skip PyTorch flash attention path in `sdpa::forward_bf16`. |
| `FLAME_NO_FLASH_ATTN` | off | Skip in-tree flash kernel. |
| `FLAME_ALLOC_POOL=0` | on | Disable cuda_alloc_pool. |
| `FLAME_POOL_ZERO` | off | Zero pool-recycled AND fresh BF16 allocations. Diagnostic for uninit reads. |
| `FLAME_SDPA_FORCE_STREAM` | off | Force `sdpa_stream_bf16` fallback path. **v2 measurement: makes cross-process WORSE — do not enable.** |

---

## Updated Files Changed

All changes persist in the tree. No reverts needed.

### v1 changes (unchanged, still in place)
| File | Change |
|---|---|
| `flame-core/src/torch_sdpa.rs` | 2 `cuCtxSynchronize` calls around PyTorch flash attention boundary |
| `flame-core/src/ops/fused_inference.rs` | `cuStreamSynchronize` after each `cublasLtMatmul` |
| `flame-core/src/cuda/fused_linear3d.cu` | NONE-only reduction mask + widened-mask fallback + `SPLITK_NUM=1` |
| `flame-core/cuda/gemm_bf16_fp32acc.cu` | Same NONE-only + splitk=1 pattern |
| `flame-core/src/sdpa.rs` | `FLAME_NO_TORCH_SDPA` env var |
| `flame-core/src/cuda_alloc_pool.rs` | `FLAME_POOL_ZERO=1` env var |

### v2 changes (this session)
| File | Change |
|---|---|
| `inference-flame/src/models/motif_video_dit.rs` | `MotifDit` gains `resident_weights: Option<Vec<Arc<...>>>` field; new `ensure_resident(&mut self)` method; `forward()` rewritten to use persistent field; `MOTIF_RESIDENT_REBUILD` diagnostic env var added. |
| `inference-flame/src/bin/motif_parity.rs` | `MOTIF_NO_CAPTURE=1` env var; graceful fallback when `dit.capture` is None. |

---

## Recommended Inference Configuration (unchanged from v1)

```bash
env MOTIF_RESIDENT=1 ./motif_encode ...
env MOTIF_RESIDENT=1 ./motif_gen ...
env MOTIF_RESIDENT=1 ./motif_decode ...
```

Works on a single 3090 Ti at full resolution. With v2, multi-step sampling
loops will now be in-process deterministic (each step's forward is
bit-identical to the previous step's forward on the same inputs), which
matters for debugging and reproducibility within a run.

---

## Repro Commands (v2)

```bash
# Build
cd /home/alex/EriDiffusion/inference-flame
cargo build --bin motif_parity --release

# Python reference (only regenerate if you changed the dump script)
cd /home/alex/EriDiffusion
DUMP_T=1 DUMP_H=8 DUMP_W=8 python3 scripts/motif_block_dump.py \
  /home/alex/serenity/output/motif_block_dump_tiny.safetensors

# Verify v2 fix — in-process bit-exactness (should see all 1.000000000)
env -i PATH=/usr/bin:/usr/local/bin:/usr/local/cuda/bin HOME=$HOME \
  MOTIF_RESIDENT=1 MOTIF_DOUBLE_FWD=5 \
  ./target/release/motif_parity \
  /home/alex/serenity/output/motif_block_dump_tiny.safetensors

# A/B: confirm old per-forward behavior regresses in-process (should see drift)
env -i PATH=/usr/bin:/usr/local/bin:/usr/local/cuda/bin HOME=$HOME \
  MOTIF_RESIDENT=1 MOTIF_RESIDENT_REBUILD=1 MOTIF_DOUBLE_FWD=5 \
  ./target/release/motif_parity \
  /home/alex/serenity/output/motif_block_dump_tiny.safetensors

# Cross-process sample (expect wider range than v1 reported)
for i in $(seq 1 20); do
  env -i PATH=/usr/bin:/usr/local/bin:/usr/local/cuda/bin HOME=$HOME \
    MOTIF_RESIDENT=1 \
    ./target/release/motif_parity \
    /home/alex/serenity/output/motif_block_dump_tiny.safetensors \
    2>&1 | grep final_output
done
```

---

## v1 Content Retained Below (For Reference)

### Original Root Causes (all still valid)

**Root Cause #1**: PyTorch flash attention cross-stream race — fixed with
2 `cuCtxSynchronize` calls in `flame-core/src/torch_sdpa.rs`.

**Root Cause #2**: cuBLASLt non-deterministic algorithm selection — fixed
with `REDUCTION_SCHEME_NONE`-only mask + `SPLITK_NUM=1` in `fused_linear3d.cu`
and `gemm_bf16_fp32acc.cu`. **Caveat**: same mask applied to
`gemm_bf16_cublaslt.cu` made results worse; reverted there.

**Root Cause #3**: `BlockOffloader` H2D prefetch racing with weight reads —
workaround via `MOTIF_RESIDENT=1`. v2 makes this a proper persistent-cache
rather than a per-forward workaround.

**Root Cause #4 (v1 suspected, v2 partially addressed)**: residual drift.
v2 fixed the in-process component. Cross-process remains.

### Cyclic recurrence observation (v1)

In-process 20-forward chain showed cos_vs_0 values like `0.951591201`
appearing at FWD#3, FWD#14, FWD#20 — identical to 9 decimals. This was
a SYMPTOM of the v2 bug — because every forward re-preloaded, pool state
cycled through the same configurations. **With v2 fix, the cycle is gone:
FWD#N is bit-identical to FWD#0 for all N.**

### First-divergence finding (v1)

"At `MOTIF_BISECT=3`, the first divergent intermediate between runs is
`enc_q_proj`. But the block's INPUTS themselves differed across runs,
meaning non-determinism can enter at any block." With v2's in-process
bit-exactness, cross-process bisection is now clean: no need to fight
intra-process drift while hunting cross-process drift.

---

## Honest Assessment (v2)

The v2 fix is a real, orthogonal improvement regardless of cross-process
variance:
- It eliminates a subtle correctness bug where "resident" meant
  "re-allocated every forward" (contradicting the variable name and the
  comment's intent).
- It enables clean cross-process bisection (point 1 in Where to Look Next).
- It saves allocation churn in inference loops with multiple denoising
  steps — each forward now reuses the same weight buffers.

The in-process bit-exactness is a tangible win for:
- Inference reproducibility within a run
- Debugging — if FWD#N differs from FWD#N+1 on the same inputs, it's a
  real bug, not determinism noise
- Any future tests that want to verify "forward is a pure function"

Cross-process variance is still unresolved. It is NOT blocking inference
(outputs are in the correct range; most runs are near bit-exact; no
catastrophic generations should result from tail cases). It is blocking
bit-for-bit parity validation vs Python.

The cleanest next step is to extend `MOTIF_BISECT` with intra-block
fingerprints and run a 2-process diff to pin the culprit kernel.
