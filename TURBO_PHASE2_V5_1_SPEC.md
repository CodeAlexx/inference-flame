# Turbo Flame — Phase 2 v5.1 CC Prompt

**Supersedes v5.** v5 assumed Chroma and Qwen-Image-Edit had `forward_with_offloader(&mut offloader)` parameter-style methods like Klein. They don't — both store `offloader: BlockOffloader` as a struct field and use `self.offloader` inside their `forward` methods. v5.1 honors the original Phase 1 claim ("turbo replication is mechanical for BlockOffloader-wired models") via Option A: extract per-model block loops into `forward_inner<L: OffloaderApi>(...)`, which both the offloader path and the turbo path call. Trait + wrappers are a one-time cost; per-model extraction is the marginal cost being measured.

**The reframed claim:** future BlockOffloader-wired models get turbo for ~10 lines of binary edits + a one-time extraction of their forward loop body into `forward_inner(&mut impl OffloaderApi)`. v5.1 measures both the one-time and per-model costs.

Pollution contract: **zero changes to `flame-core` and `flame-diffusion`.** `git diff flame-core/ flame-diffusion/` returns empty at phase end. Trait + foreign-type impl live entirely in inference-flame (orphan rule allows local-trait + foreign-type).

---

## 0. Context already verified

The orchestrator already verified all of this. Builder does NOT re-investigate.

- Phase 1 v4 shipped at `797c8bc`. `TurboBlockLoader` and `TurboBlock` final API in `src/turbo/loader.rs` and `src/turbo/block.rs`. Klein's `forward_with_turbo` is at `src/models/klein.rs:1221`. Klein binary is `src/bin/klein9b_infer_turbo.rs`.
- **Chroma DiT** (`src/models/chroma_dit.rs`): `pub struct ChromaDit { offloader: BlockOffloader, ... }` at line 212. Single forward method at line 620 (`pub fn forward`). Uses `self.offloader` at lines 676/680/683/713/716. 19 double + 38 single blocks. **Extract this single `forward`.**
- **Qwen-Image-Edit DiT** (`src/models/qwenimage_dit.rs`): `pub struct QwenImageDit { offloader: BlockOffloader, ... }` at line 137. Three forward methods: `forward` (565), `forward_edit` (685), `forward_edit_with_ref_timestep` (708). Lines 609 and 770 use `self.offloader` — line 609 is in `forward`, line 770 is in `forward_edit_with_ref_timestep`. **Phase 2 only extracts `forward_edit_with_ref_timestep`.** It's the method `qwenimage_edit_gen.rs:393,396` calls — the 2511 zero_cond_t edit path, the actual Phase 2 hard case. Other two stay untouched.
- **BlockOffloader API** (flame-diffusion): `prefetch_block(&mut self, idx) -> anyhow::Result<()>` (block_offload.rs:408), `await_block(&mut self, idx) -> anyhow::Result<Arc<HashMap<String, Tensor>>>` (block_offload.rs:531), `block_count(&self) -> usize` (622), `pinned_bytes(&self) -> usize` (627). `unsafe impl Send + Sync for BlockOffloader` already exists at line 142.
- **TurboBlockLoader API** (Phase 1, our code): `prefetch_block(&mut self, idx) -> Result<(), VmmError>`, `await_block(&mut self, idx) -> Result<Arc<TurboBlock>, VmmError>`, `block_count`, `pinned_bytes`. `TurboBlock { handle: Arc<ResidentHandle>, weights: HashMap<String, Tensor> }`.
- **Binaries to clone**: `src/bin/chroma_infer.rs` → `chroma_infer_turbo.rs` (uses `ChromaDit` only). `src/bin/qwenimage_edit_gen.rs` → `qwenimage_edit_gen_turbo.rs` (uses `QwenImageDit::forward_edit_with_ref_timestep`).

---

## 1. Goal

Extract Chroma's `forward` and Qwen's `forward_edit_with_ref_timestep` into generic `forward_inner<L: OffloaderApi>` methods. Both today's `self.offloader.X` call paths AND new `forward_with_turbo(loader: &mut TurboBlockLoader)` paths route through the same `forward_inner`. Zero behavior change in the non-turbo path. New turbo binaries `chroma_infer_turbo` and `qwenimage_edit_gen_turbo` produce BF16-bit-identical output to their non-turbo counterparts on fixed seed + prompt. Three-model timing report (Klein / Chroma / Qwen-Edit × off/on) on RTX 3090 Ti.

---

## 2. Out of Scope

- Any `flame-core`, `flame-diffusion`, `lanpaint-flame`, `eri-lycoris/lycoris-rs` file.
- Any model outside Chroma + Qwen-Image-Edit.
- `chroma_gen.rs`, `qwenimage_gen.rs` (non-edit), all other binaries.
- Chroma's untranspose helper (Phase 3+).
- CUDA graph capture, async PrefetchWorker wiring (Phase 3+).
- Pre-VAE latent compare (Phase 3+).
- FLUX, ERNIE, LTX-2, SD3, SDXL, SD15, Cascade, Anima, Z-Image — Phase 3+.
- `TurboBlockLoader`, `TurboBlock`, `VmmArena`, anything under `src/turbo/vmm/` — already final from Phase 1.
- Klein's `forward_with_offloader` and `forward_with_turbo` — already shipped, do not refactor through the trait. Phase 1 stays as-is. The trait + impls coexist with Klein's direct call paths; no conflict.
- Refactoring `forward` (line 565) or `forward_edit` (line 685) on `QwenImageDit`. Only `forward_edit_with_ref_timestep` (708) is in scope.
- Any new sibling crate.

If any of the above is tempting, stop and flag.

---

## 3. Deliverables

### 3.1 The trait + wrappers (one-time cost)

1. **New file `inference-flame/src/offload_api.rs`** (NOT inside `src/turbo/`; always compiled):

   ```rust
   use std::collections::HashMap;
   use std::ops::Deref;
   use std::sync::Arc;
   use flame_core::Tensor;
   use flame_diffusion::BlockOffloader;

   /// Common surface that BlockOffloader and TurboBlockLoader both expose,
   /// so a model's block-loop can be generic in which loader it consumes.
   pub trait OffloaderApi {
       /// Smart-pointer-like handle returned by await_block. Derefs to a
       /// HashMap so loop bodies can write `block.get(name)` uniformly.
       type Block: Deref<Target = HashMap<String, Tensor>>;

       fn prefetch_block(&mut self, idx: usize) -> anyhow::Result<()>;
       fn await_block(&mut self, idx: usize) -> anyhow::Result<Self::Block>;
       fn block_count(&self) -> usize;
       fn pinned_bytes(&self) -> usize;
   }

   /// Wrapper: BlockOffloader returns Arc<HashMap<...>>; Arc<HashMap> already
   /// Derefs to HashMap, but we wrap to give the trait Block type a single
   /// concrete shape across both impls.
   pub struct OffloaderBlock(pub Arc<HashMap<String, Tensor>>);

   impl Deref for OffloaderBlock {
       type Target = HashMap<String, Tensor>;
       fn deref(&self) -> &HashMap<String, Tensor> { &self.0 }
   }

   impl OffloaderApi for BlockOffloader {
       type Block = OffloaderBlock;
       fn prefetch_block(&mut self, idx: usize) -> anyhow::Result<()> {
           BlockOffloader::prefetch_block(self, idx)
       }
       fn await_block(&mut self, idx: usize) -> anyhow::Result<Self::Block> {
           BlockOffloader::await_block(self, idx).map(OffloaderBlock)
       }
       fn block_count(&self) -> usize { BlockOffloader::block_count(self) }
       fn pinned_bytes(&self) -> usize { BlockOffloader::pinned_bytes(self) }
   }
   ```

2. **New file `inference-flame/src/turbo/api.rs`** (turbo-gated; module file):

   ```rust
   use std::collections::HashMap;
   use std::ops::Deref;
   use std::sync::Arc;
   use flame_core::Tensor;
   use crate::offload_api::OffloaderApi;
   use crate::turbo::{TurboBlock, TurboBlockLoader};

   /// Wrapper preserving Arc<TurboBlock> refcount semantics (Phase 1 safety:
   /// Arc keeps the slot's ResidentHandle alive until reader drops) while
   /// providing the same Deref<Target=HashMap> shape as OffloaderBlock.
   pub struct TurboAwaited(pub Arc<TurboBlock>);

   impl Deref for TurboAwaited {
       type Target = HashMap<String, Tensor>;
       fn deref(&self) -> &HashMap<String, Tensor> { &self.0.weights }
   }

   impl OffloaderApi for TurboBlockLoader {
       type Block = TurboAwaited;
       fn prefetch_block(&mut self, idx: usize) -> anyhow::Result<()> {
           TurboBlockLoader::prefetch_block(self, idx)
               .map_err(|e| anyhow::anyhow!("turbo prefetch_block: {e}"))
       }
       fn await_block(&mut self, idx: usize) -> anyhow::Result<Self::Block> {
           TurboBlockLoader::await_block(self, idx)
               .map(TurboAwaited)
               .map_err(|e| anyhow::anyhow!("turbo await_block: {e}"))
       }
       fn block_count(&self) -> usize { TurboBlockLoader::block_count(self) }
       fn pinned_bytes(&self) -> usize { TurboBlockLoader::pinned_bytes(self) }
   }
   ```

3. **`inference-flame/src/lib.rs`**: add `pub mod offload_api;` (always compiled, NO cfg gate).

4. **`inference-flame/src/turbo/mod.rs`**: add `pub mod api;` and re-export `pub use api::TurboAwaited;`.

### 3.2 Chroma extraction

1. **In `src/models/chroma_dit.rs`:**
   - Extract the entire block loop body from `pub fn forward(...)` (line 620 onwards, the part starting from `self.offloader.prefetch_block(0)` at line 676 through the end of the single-stream loop) into a new method:
     ```rust
     fn forward_inner<L: crate::offload_api::OffloaderApi>(
         &self,
         /* same args as forward, plus precomputed embeddings */,
         loader: &mut L,
     ) -> Result<Tensor>
     ```
   - Replace the original loop in `forward` with a call to `self.forward_inner(..., &mut self.offloader)`. The existing struct field stays as-is. `forward` stays public, behavior identical.
   - Add new method, gated:
     ```rust
     #[cfg(feature = "turbo")]
     pub fn forward_with_turbo(
         &self,
         /* same args as forward */,
         loader: &mut crate::turbo::TurboBlockLoader,
     ) -> Result<Tensor> {
         self.forward_inner(..., loader)
     }
     ```

   The extraction must be mechanical — only changes are: parameterize the loader type via the trait, replace `self.offloader.X` with `loader.X`, and replace `let raw_arc = self.offloader.await_block(i)?` (which yielded `Arc<HashMap<String, Tensor>>`) with `let block = loader.await_block(i)?` (which yields `L::Block`, deref'ing to `&HashMap`). All downstream `block.get(name)` calls work identically because both wrappers Deref to HashMap.

   **If the extraction requires changing the `prepare_block` helper's signature** (currently takes `Arc<HashMap<String, Tensor>>` per Klein; Chroma's structure may differ), adjust to take `&HashMap<String, Tensor>` — the Deref auto-coerces from either wrapper. **If the extraction requires anything else, stop and flag.**

### 3.3 Qwen-Image-Edit extraction

1. **In `src/models/qwenimage_dit.rs`:**
   - Extract the block loop body from `forward_edit_with_ref_timestep` (line 708 onwards, around line 770 prefetch through end of block loop) into:
     ```rust
     fn forward_edit_with_ref_timestep_inner<L: crate::offload_api::OffloaderApi>(
         &self,
         /* same args, plus precomputed embeddings */,
         loader: &mut L,
     ) -> Result<Tensor>
     ```
   - Replace the loop in `forward_edit_with_ref_timestep` with a call to `self.forward_edit_with_ref_timestep_inner(..., &mut self.offloader)`.
   - Add gated method `pub fn forward_edit_with_ref_timestep_turbo(&self, ..., loader: &mut TurboBlockLoader) -> Result<Tensor>` that calls the inner with the turbo loader.

   `forward` (line 565) and `forward_edit` (line 685) are NOT touched. The other `self.offloader.prefetch_block(0)` at line 609 stays as-is — it's inside `forward`, out of scope.

   **Multi-region RoPE math, zero_cond_t conditioning math, per-token modulation: untouched.** The extraction touches only the block loop, which iterates over weights regardless of conditioning shape.

### 3.4 Turbo binaries

1. **`inference-flame/src/bin/chroma_infer_turbo.rs`** — clone of `src/bin/chroma_infer.rs`. Edits:
   - Add `#[cfg(feature = "turbo")]` at the file top.
   - Build a `VmmArena::new_for_klein9b`-equivalent: add `pub fn new_for_chroma(device, copy_stream)` to `src/turbo/arena.rs` with appropriate sizing (Chroma 12B-class, 19+38 blocks).
   - Build a `TurboBlockLoader::new(...)` with Chroma's block prefixes.
   - Call `dit.forward_with_turbo(...)` instead of `dit.forward(...)`.
   - Same VMM-unsupported error handling as `klein9b_infer_turbo`.
   - Same CLI, env vars, output convention as the non-turbo bin.

2. **`inference-flame/src/bin/qwenimage_edit_gen_turbo.rs`** — clone of `src/bin/qwenimage_edit_gen.rs`. Same edits:
   - Feature-gated.
   - Add `pub fn new_for_qwen_image_edit(device, copy_stream)` to `src/turbo/arena.rs` with sizing for Qwen-Image's 60-layer DiT (largest block ~ check at runtime; assume comparable to Klein 9B since both fit on 24GB).
   - Call `dit.forward_edit_with_ref_timestep_turbo(...)` instead of `dit.forward_edit_with_ref_timestep(...)` at the cond + uncond call sites (lines 393, 396 in the source).
   - Same CLI, env vars, output convention.

3. **`inference-flame/Cargo.toml`** — add two `[[bin]]` entries:
   ```toml
   [[bin]]
   name = "chroma_infer_turbo"
   path = "src/bin/chroma_infer_turbo.rs"
   required-features = ["turbo"]

   [[bin]]
   name = "qwenimage_edit_gen_turbo"
   path = "src/bin/qwenimage_edit_gen_turbo.rs"
   required-features = ["turbo"]
   ```

### 3.5 Parity tests

1. **`inference-flame/tests/turbo_chroma_parity.rs`** — `#[cfg(feature = "turbo")]`. Run `chroma_infer_turbo` and `chroma_infer` on fixed seed + prompt, BF16-bit-identical (u16 bit-cast compare on the latents OR byte-equal PNG, mirroring `turbo_klein9b_parity.rs`). Skip cleanly if Chroma weights unavailable. Add `#[ignore = "long real-checkpoint denoise; run with --ignored"]` — Chroma is faster than Qwen-Edit but still many-minute scale.

2. **`inference-flame/tests/turbo_qwen_edit_parity.rs`** — `#[cfg(feature = "turbo")]`. Same pattern for `qwenimage_edit_gen_turbo` vs `qwenimage_edit_gen`. Fixed seed, fixed reference image, fixed prompt. `#[ignore = "23-min Qwen-Image-Edit denoise; run with --ignored"]`.

3. **`Cargo.toml`** — add two `[[test]]` entries with `required-features = ["turbo"]`.

### 3.6 Benches

1. **`inference-flame/benches/turbo_chroma_offload.rs`** — Chroma 1024², 40 steps, CFG 4.0, seed 42, RTX 3090 Ti. Measurements: total wall-clock, per-step denoise, per-swap cost (nvtx markers around `loader.prefetch_block` / `loader.await_block` / `offloader.prefetch_block` / `offloader.await_block`), H2D overlap ratio. Compare turbo off vs on. Same shape as `turbo_klein9b_offload.rs`.

2. **`inference-flame/benches/turbo_qwen_edit_offload.rs`** — Qwen-Image-Edit 1024², 50 steps, true-CFG 4.0, seed 42, fixed reference image, RTX 3090 Ti. Same measurements.

3. **`Cargo.toml`** — add two `[[bench]]` entries.

### 3.7 Three-model timing report

**`PHASE2_TIMING_REPORT.md`** at inference-flame repo root. Filled in from measured numbers, no `?` placeholders left, no estimates. Structure:

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

Plus replication-cost rows:

```
                     Trait+Impl LoC   Per-model Extraction LoC   Calendar Time to Green Parity
One-time             ~80              —                          —
Klein (Phase 1)      —                — (had param-style)        — (Phase 1 baseline)
Chroma               —                ?                          ?
Qwen-Image-Edit      —                ?                          ?
```

Where "Per-model Extraction LoC" is the diff size of the `chroma_dit.rs` and `qwenimage_dit.rs` changes — measured by `git diff --stat` of those two files. Calendar time is wall-clock from "start coding model X" to "turbo_<model>_parity passes" — approximate hours.

If the extraction LoC per model is small (single digits to low tens), the mechanical-replication claim holds. If it's large or wildly different between Chroma and Qwen, that's the interesting finding.

### 3.8 Doc updates

- **`inference-flame/README.md`**: change the existing "Turbo (experimental — Klein 9B)" section to "Turbo (experimental — Klein 9B / Chroma / Qwen-Image-Edit)". List all three binaries. Add the three-model timing table once measured. Note the `OffloaderApi` trait abstraction in one sentence.
- **No other doc edits.** Specifically: no `flame-core/docs/*` or `flame-diffusion/docs/*` edits.

### 3.9 Revert prompt

**`turbo_flame_phase2_v5_1_revert.md`** at inference-flame repo root. Phase 1 stays shipped; this rolls back only Phase 2.

---

## 4. Technical Spec

Phase 2 v5.1 reuses Phase 1's `TurboBlockLoader`, `TurboBlock`, `VmmArena`, `ResidentHandle`. **Not redesigning. Not rewriting. Not extending the loader API.** New code:

1. `src/offload_api.rs` — trait + BlockOffloader impl + OffloaderBlock wrapper (~50 LoC).
2. `src/turbo/api.rs` — TurboBlockLoader impl + TurboAwaited wrapper (~30 LoC).
3. `src/turbo/arena.rs` — add `new_for_chroma` and `new_for_qwen_image_edit` constructors (small, mirror `new_for_klein9b`).
4. `src/models/chroma_dit.rs` — extract `forward_inner`, retarget `forward`, add `forward_with_turbo`. Net per-model LoC delta: target single digits to low tens.
5. `src/models/qwenimage_dit.rs` — same for `forward_edit_with_ref_timestep` only. Net per-model LoC delta: same target.
6. Two binaries, two parity tests, two benches.

**Chroma untranspose.** Chroma's `forward` calls `prepare_block` to undo BlockOffloader's auto-transpose (chroma_dit.rs has logic similar to Klein at klein.rs:1135-1151). For turbo: TurboBlockLoader's await_block returns weights in their on-disk safetensors layout (no auto-transpose), so the prepare_block step needs to behave differently per-loader. Two solutions:
- **(a)** `prepare_block` keeps doing the transpose; `OffloaderApi` exposes a `needs_transpose() -> bool` method (default-false; BlockOffloader returns true). `forward_inner` checks and conditionally transposes. **NO** — this adds a method to the trait, which is a loader API change.
- **(b)** Mirror Klein's Phase 1 fix: `forward_with_turbo` skips the transpose call (consistent with `klein.rs:1265-1269` comment about turbo using on-disk `[out, in]` layout directly). `forward_inner` accepts a `transpose_2d_weights: bool` parameter, callers pass `true` for the offloader path and `false` for turbo.

   **Use solution (b).** It's a function parameter, not an API change. The `forward` method passes `true` (preserving current behavior); `forward_with_turbo` passes `false`.

If Chroma's `prepare_block` does ANYTHING beyond transposing 2D `.weight` tensors (e.g., converting dtype, reshaping), the `transpose_2d_weights` parameter widens to `prepare_mode: PrepareMode { Offloader, Turbo }` — but flag this BEFORE making the change. The expectation is a one-line conditional gated on the bool parameter.

**Qwen `forward_edit_with_ref_timestep` zero_cond_t handling.** This logic lives OUTSIDE the block loop (it builds the timestep tensors before the loop, then the loop just iterates blocks). Extraction does not touch it. If the inner block_forward calls take per-region t-sigma tensors as args, those args are passed into `forward_edit_with_ref_timestep_inner` along with everything else.

---

## 5. Hot-Path and Safety Requirements

Same as Phase 1 v4. `ensure_resident` fast path ≤ 1 μs p99 (already verified Phase 1, won't regress here). Arc<ResidentHandle> + event-gated Drop preserved. Pollution contract holds: `git diff flame-core/ flame-diffusion/` empty.

**New Phase 2 checks:**
1. **Trait location**: `git grep 'trait.*Block\|trait OffloaderApi' flame-core/ flame-diffusion/` returns empty. Trait must live entirely in inference-flame.
2. **Phase 1 Klein tests still green**: `cargo test -p inference-flame --features turbo --test turbo_ensure_resident_hot --test turbo_ensure_resident_cold --test turbo_vmm_arena_basic --test turbo_vmm_unsupported --test turbo_tensor_over_vmm --test turbo_reader_outlives_prefetch` must all pass. (The 30-min `turbo_klein9b_parity` is `#[ignore]`'d; Skeptic decides whether to run it.)
3. **Per-model extraction LoC ceiling**: `git diff --stat src/models/chroma_dit.rs src/models/qwenimage_dit.rs` should show small numbers per file. If either file's diff exceeds ~50 lines, the claim is straining; flag for Skeptic. The `forward_inner` body itself is a pure move (~zero net diff); the new `forward_with_turbo` wrapper is a few lines.
4. **`forward_inner` body parity**: the extracted body in chroma_dit's `forward_inner` and the analogous Klein code (klein.rs `forward_with_offloader` body) should differ ONLY in model-specific math. The structural shape (`prefetch(0); for i { let block = loader.await(i); if next < total { loader.prefetch(next); } ... }`) must be identical. If structurally different, flag.
5. **`forward` non-turbo behavior unchanged**: `chroma_infer` and `qwenimage_edit_gen` (the existing non-turbo binaries) must produce identical output to their pre-Phase-2 behavior. A diff against pre-Phase-2 output is the regression check. (If non-turbo binaries go through `forward_inner` now, they exercise the trait abstraction — must produce zero behavior difference.)

---

## 6. Test Plan

Three must-pass tests + Phase 1 regression:
- `turbo_chroma_parity` — bit-identical Chroma output (with `#[ignore]`; run via `--ignored`).
- `turbo_qwen_edit_parity` — bit-identical Qwen-Edit output (with `#[ignore]`; run via `--ignored`).
- All Phase 1 turbo tests still pass (`turbo_tensor_over_vmm`, `turbo_reader_outlives_prefetch`, `turbo_ensure_resident_*`, `turbo_vmm_arena_basic`, `turbo_vmm_unsupported`).
- Both default and `--features turbo` builds compile clean.

---

## 7. Bench Plan

§3.6 benches + §3.7 timing report. The orchestrator runs the actual GPU benches after Bug Fixer + Skeptic sign off; Builder ships the bench scaffolding only. Reasoning: Builder runtime is bounded; ~1 hour of GPU bench runs (Klein turbo on/off + Chroma turbo on/off + Qwen-Edit turbo on/off — Qwen-Edit alone is ~46 min for both runs) shouldn't burn agent budget.

---

## 8. Doc Updates

§3.8.

---

## 9. Agent Roles

### Builder
- Implement §3.1–§3.6 exactly. No loader API changes.
- `#[cfg(feature = "turbo")]` correctness: trait + BlockOffloader impl in `src/offload_api.rs` are NOT gated. TurboBlockLoader impl in `src/turbo/api.rs` IS gated (already inside `src/turbo/`).
- Run `cargo build -p inference-flame` (default) — must succeed, must NOT pull in turbo code.
- Run `cargo build -p inference-flame --features turbo` — must succeed, must include all four turbo binaries (`klein9b_infer_turbo`, `chroma_infer_turbo`, `qwenimage_edit_gen_turbo`).
- Run `cargo test -p inference-flame --features turbo` — Phase 1 regression tests must pass. Parity tests are `#[ignore]`'d; do NOT run them.
- Report `git diff flame-core/ flame-diffusion/` (must be empty) and `git diff --stat src/models/chroma_dit.rs src/models/qwenimage_dit.rs` (per-model LoC measurement) as part of the deliverable.
- Do NOT run benches. Do NOT spawn sub-agents.
- If Chroma's `prepare_block` does anything beyond 2D transpose, STOP and report — needs design call before proceeding.

### Bug Fixer
- Audit the trait location: `git grep 'trait.*Block\|trait OffloaderApi' flame-core/ flame-diffusion/` empty.
- Verify `forward` (non-turbo) on Chroma and Qwen still produces structurally-identical kernel sequences as before extraction. The simplest evidence is: `forward_inner` is called from `forward` with `&mut self.offloader`, which goes through the same `BlockOffloader` API as before — just routed via the trait. No conditioning math change, no kernel call change. Audit by reading the `forward` body diff.
- Phase 1 regression: re-run all Phase 1 tests, confirm green.
- Trait error mapping: `TurboBlockLoader`'s `VmmError` → `anyhow::Error` via `anyhow::anyhow!("turbo prefetch_block: {e}")`. Confirm `VmmError` implements `Display` (it does — Phase 1 ships it).
- Audit Arc<TurboAwaited>'s lifetime vs prior Phase 1 contract: `TurboAwaited(Arc<TurboBlock>)` must preserve the Arc refcount → ResidentHandle → event-gated Drop chain. The newtype wrapper does nothing other than expose Deref; refcount semantics are the same as `Arc<TurboBlock>` directly.
- Per-model extraction LoC measurement (per §5).
- Bit-identical parity tests: BF16 u16 bit-compare or PNG byte-compare, NOT FP32 near-equal.
- `Send/Sync` for the trait: `BlockOffloader` is already `Send + Sync` (`block_offload.rs:142`); `TurboBlockLoader` needs auditing — check Phase 1 ships it. If not, this phase doesn't add the bound to the trait (that would constrain the pattern unnecessarily).
- Grep the new `forward_with_turbo` methods for any code outside the three approved categories: parameter type swap, weight access (now via Deref), constructor differences. Anything else: flag.

### Skeptic
- Is the replication claim now validated by N=2 (Chroma + Qwen-Edit, two different DiT shapes — single-stream Chroma vs 60-layer Qwen with multi-region RoPE)? Or does it need N=3 before publishing? Argue.
- Trait extraction overhead: `forward_inner` is generic on `<L: OffloaderApi>`. Does monomorphization cost (two instantiations per model: one for `BlockOffloader`, one for `TurboBlockLoader`) materially bloat the binary or impact the inner loop's branch prediction? Likely no, but worth reasoning about.
- Chroma's `prepare_block` transpose: with the bool parameter pattern from §4 (b), does that scale? Each new model adding turbo will need its own bool gate (or none, if its weights don't need transpose). Is that cleaner than a `prepare_mode: enum`? Argue.
- Qwen-Image-Edit's `zero_cond_t` math: confirm via reading that it's truly outside the extracted block loop. If it leaks into the loop body via captured timestep tensors, the extraction either (a) takes those tensors as args or (b) the extraction failed to be mechanical.
- For the timing report: the orchestrator runs benches after sign-off. What measurement hazards apply across three models — thermals on a 23-min Qwen-Edit run, warm-vs-cold model load on the second run of the same model, GPU clock throttling? Recommend warmup/cooldown protocol.
- Revert prompt dry-run: forgot files?
- Verdict: SHIP / SHIP-WITH-DOC / BLOCKER. Blocker only if a real correctness or pollution issue is found.

All three sign off before commit.

---

## 10. Success Criteria

Done when all true:

1. `git diff flame-core/ flame-diffusion/` empty.
2. `cargo build -p inference-flame` (default, no turbo) succeeds.
3. `cargo build -p inference-flame --features turbo` succeeds (includes all four turbo binaries).
4. `cargo test -p inference-flame --features turbo` — all Phase 1 turbo tests pass (regression check).
5. `chroma_infer_turbo --seed 42 "<prompt>"` produces BF16-bit-identical output to `chroma_infer --seed 42 "<prompt>"` (verified via `turbo_chroma_parity` with `--ignored`).
6. `qwenimage_edit_gen_turbo --seed 42 ...` produces BF16-bit-identical output to `qwenimage_edit_gen --seed 42 ...` (verified via `turbo_qwen_edit_parity` with `--ignored`).
7. `PHASE2_TIMING_REPORT.md` has all numbers filled in, no `?` placeholders, no estimates.
8. Replication-cost table in the report has LoC and (best-effort) calendar-time filled in.
9. `inference-flame/README.md` Turbo section updated.
10. `turbo_flame_phase2_v5_1_revert.md` exists, Skeptic dry-ran it.
11. **Trait location check**: `git grep 'trait.*OffloaderApi\|trait.*Block' flame-core/ flame-diffusion/` empty.
12. Per-model extraction LoC reasonable (single digits to low tens net delta in `forward_*` methods, plus the one-time `forward_inner` extraction body which is mostly a pure move).

---

## 11. What NOT to do

- Do not touch `TurboBlockLoader`, `TurboBlock`, `VmmArena`, or anything under `src/turbo/vmm/`. Phase 1 final.
- Do not touch Klein. Klein's Phase 1 paths stay direct (no trait routing). Refactoring Klein through the trait is out of scope and would re-litigate Phase 1.
- Do not touch `flame-core` or `flame-diffusion`. Pollution contract is the hard line.
- Do not extract `forward` or `forward_edit` on `QwenImageDit`. Only `forward_edit_with_ref_timestep`. The other two stay as-is.
- Do not introduce a `prepare_mode` enum without flagging — if Chroma needs anything beyond a bool transpose toggle, that's a design call.
- Do not generalize the trait further than the four methods. No `evict()`, no `set_priority()`, no `stats()`. Keep the surface minimal — the smaller the trait, the easier future models adopt it.
- Do not turn `TurboBlock` into anything other than what Phase 1 shipped. The wrapper `TurboAwaited(Arc<TurboBlock>)` is the only new shape.
- Do not run the long parity tests (`#[ignore]`'d for a reason). The orchestrator does that as part of §11 task.
- Do not wire FLUX, ERNIE, LTX-2, or any other model.
- Do not generalize `forward_with_turbo` and `forward_with_offloader` into a single method via a runtime-dispatched offloader. Static dispatch via the trait + monomorphization is the design.
- Do not worktree. Serial on master.
- Do not give Albert manual git instructions. Revert prompt is the recovery.

---

**End of Phase 2 v5.1 prompt. Builder follows this verbatim. Bug Fixer audits against it. Skeptic verdicts. Orchestrator runs benches and commits.**
