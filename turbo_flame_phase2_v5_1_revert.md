# Turbo Flame Phase 2 v5.1 — Revert Prompt

Albert: paste this verbatim into a fresh Claude Code session in
`/home/alex/EriDiffusion/inference-flame/` to roll back Phase 2 v5.1 cleanly.
**Phase 1 stays shipped** — this only undoes Phase 2 v5.1. (Phase 1's HEAD is
`797c8bc`; we're rolling forward edits made on top of that.)

---

## What to remove

### Files to delete
- `src/offload_api.rs`
- `src/turbo/api.rs`
- `src/bin/chroma_infer_turbo.rs`
- `src/bin/qwenimage_edit_gen_turbo.rs`
- `tests/turbo_chroma_parity.rs`
- `tests/turbo_qwen_edit_parity.rs`
- `benches/turbo_chroma_offload.rs`
- `benches/turbo_qwen_edit_offload.rs`
- `PHASE2_TIMING_REPORT.md`
- `turbo_flame_phase2_v5_1_revert.md` (this file)

### Files to revert (only the Phase 2 hunks; preserve any unrelated edits)

**`Cargo.toml`** — remove:
- `[[bin]] chroma_infer_turbo`
- `[[bin]] qwenimage_edit_gen_turbo`
- `[[bench]] turbo_chroma_offload`
- `[[bench]] turbo_qwen_edit_offload`
- `[[test]] turbo_chroma_parity`
- `[[test]] turbo_qwen_edit_parity`

**`src/lib.rs`** — remove the `pub mod offload_api;` line.

**`src/turbo/mod.rs`** — remove `pub mod api;` and `pub use api::TurboAwaited;`.

**`src/turbo/arena.rs`** — remove `new_for_chroma` and `new_for_qwen_image_edit`
constructors (keep `new_for_klein9b`).

**`src/models/chroma_dit.rs`** — three reverts:
1. Restore `double_block_forward` and `single_block_forward` to `&self`-method
   form (revert their `cfg: &ChromaConfig` parameterisation).
2. Inline `forward_inner`'s body back into `forward_cached` and remove
   `forward_inner`.
3. Remove `forward_with_turbo`.
4. Remove the `untranspose_block_weights_map` and `passthrough_block_weights`
   helpers (revert `untranspose_block_weights` to its old direct
   implementation).

**`src/models/qwenimage_dit.rs`** — same three reverts:
1. Restore `block_forward` and `block_forward_per_region` to `&self`-method
   form.
2. Inline `forward_edit_with_ref_timestep_inner` back into
   `forward_edit_with_ref_timestep` and remove the inner method.
3. Remove `forward_edit_with_ref_timestep_turbo`.
4. Remove the new helpers; revert `untranspose_block_weights` to direct form.

**`README.md`** — revert the "Turbo (experimental — Klein 9B / Chroma /
Qwen-Image-Edit)" section back to "Turbo (experimental — Klein 9B)" with the
Klein-only text.

---

## Validation steps after revert

```bash
# 1. Default build still green
cargo build -p inference-flame

# 2. Turbo build still green (Phase 1 only — klein9b_infer_turbo)
cargo build -p inference-flame --features turbo

# 3. Phase 1 turbo tests still pass
cargo test -p inference-flame --features turbo \
  --test turbo_vmm_arena_basic \
  --test turbo_ensure_resident_hot \
  --test turbo_ensure_resident_cold \
  --test turbo_vmm_unsupported \
  --test turbo_tensor_over_vmm \
  --test turbo_reader_outlives_prefetch

# 4. Pollution still empty
git -C /home/alex/EriDiffusion/flame-core diff --stat        # → empty
git -C /home/alex/EriDiffusion/flame-diffusion diff --stat   # → empty

# 5. No residual Phase 2 references in inference-flame
git grep -l "OffloaderApi\|TurboAwaited\|forward_with_turbo\|forward_edit_with_ref_timestep_turbo\|forward_edit_with_ref_timestep_inner\|forward_inner" inference-flame/src
# Expected: only klein.rs:forward_with_turbo (Phase 1, NOT a Phase 2 artifact)
```

If step 5 turns up anything other than Klein's existing `forward_with_turbo`,
something in the revert was incomplete.

---

## What NOT to revert

- Klein's `forward_with_turbo` and `forward_with_offloader` — these are Phase
  1, predate Phase 2, stay shipped.
- Anything under `src/turbo/vmm/`, `src/turbo/loader.rs`, `src/turbo/block.rs`,
  or the existing `arena.rs::new_for_klein9b` — Phase 1, untouched by Phase 2
  v5.1.
- The existing `klein9b_infer_turbo` binary, `turbo_klein9b_offload` bench, or
  any Phase 1 test.
- `flame-core`, `flame-diffusion`, `lanpaint-flame`, `eri-lycoris/lycoris-rs`
  — Phase 2 never touched them; the pollution contract is intact.

---

## After revert: the codebase is at Phase 1 v4 (HEAD = `797c8bc`)

`forward` and `forward_cached` on `ChromaDit` and `forward_edit_with_ref_timestep`
on `QwenImageDit` are back to their pre-Phase-2 shapes. The non-turbo Chroma
and Qwen-Image-Edit binaries (`chroma_infer`, `qwenimage_edit_gen`) produce
the same output as they did before Phase 2 — that's the regression check that
proves the revert was clean.
