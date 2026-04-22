# Turbo Flame Phase 1 v4 — Revert Prompt

Paste this into CC if Phase 1 needs to be backed out cleanly. Reverts only Phase 1 v4 work; leaves all prior commits intact.

## Pollution contract — verify before reverting

The phase shipped with `git diff flame-core/ flame-diffusion/` empty. If those crates have changes attributed to this phase, they are bugs and must be reverted separately. Run:

```bash
git -C /home/alex/EriDiffusion/flame-core diff --stat
git -C /home/alex/EriDiffusion/flame-diffusion diff --stat
```

Both must be empty. If not, stop and investigate before continuing.

## Files to remove (created by Phase 1 v4)

```
inference-flame/src/turbo/                              # entire directory
  mod.rs
  arena.rs
  block.rs
  loader.rs
  vmm/
    mod.rs
    cuda_ffi.rs
    error.rs
    slab.rs
    allocator.rs
    eviction.rs
    prefetch.rs
    handle.rs
    PORT_NOTES.md
inference-flame/src/bin/klein9b_infer_turbo.rs
inference-flame/tests/turbo_vmm_arena_basic.rs
inference-flame/tests/turbo_ensure_resident_hot.rs
inference-flame/tests/turbo_ensure_resident_cold.rs
inference-flame/tests/turbo_vmm_unsupported.rs
inference-flame/tests/turbo_tensor_over_vmm.rs
inference-flame/tests/turbo_reader_outlives_prefetch.rs
inference-flame/tests/turbo_klein9b_parity.rs
inference-flame/benches/turbo_klein9b_offload.rs
inference-flame/turbo_flame_phase1_v4_revert.md         # this file
```

## Files to edit back (modified by Phase 1 v4)

1. **`inference-flame/Cargo.toml`** — remove:
   - The `turbo = []` line in `[features]`.
   - The `[[bin]] name = "klein9b_infer_turbo" ... required-features = ["turbo"]` block.
   - The `[[bench]] name = "turbo_klein9b_offload" ...` block.
   - All seven `[[test]]` blocks with `required-features = ["turbo"]`:
     `turbo_vmm_arena_basic`, `turbo_ensure_resident_hot`,
     `turbo_ensure_resident_cold`, `turbo_vmm_unsupported`,
     `turbo_tensor_over_vmm`, `turbo_reader_outlives_prefetch`,
     `turbo_klein9b_parity`.
     Without these, `cargo build --features turbo` fails after the test
     files are `rm`'d with "couldn't read tests/turbo_*.rs: No such file".

2. **`inference-flame/src/lib.rs`** — remove:
   ```rust
   #[cfg(feature = "turbo")]
   pub mod turbo;
   ```

3. **`inference-flame/src/models/klein.rs`** — remove the `forward_with_turbo` method (currently around line 1218–1335 — search for `pub fn forward_with_turbo` and delete the whole `#[cfg(feature = "turbo")] pub fn forward_with_turbo(...)` block including its doc comment and closing brace).

4. **`inference-flame/README.md`** — remove the "Turbo (experimental — Klein 9B)" section (between "## Turbo" heading and the next "## Adapters & samplers" heading).

## Revert procedure

```bash
cd /home/alex/EriDiffusion/inference-flame

# 1. Remove created files
rm -rf src/turbo/
rm -f src/bin/klein9b_infer_turbo.rs
rm -f tests/turbo_vmm_arena_basic.rs \
      tests/turbo_ensure_resident_hot.rs \
      tests/turbo_ensure_resident_cold.rs \
      tests/turbo_vmm_unsupported.rs \
      tests/turbo_tensor_over_vmm.rs \
      tests/turbo_reader_outlives_prefetch.rs \
      tests/turbo_klein9b_parity.rs
rm -f benches/turbo_klein9b_offload.rs
rm -f turbo_flame_phase1_v4_revert.md

# 2. Edit the 4 files above by hand (or via Edit tool calls). Cannot script
#    safely because line numbers depend on other in-flight edits.

# 3. Verify both build modes still pass
cargo build -p inference-flame
cargo build -p inference-flame --features turbo  # MUST FAIL after revert (turbo feature removed)

# 4. If the phase was committed, instead use:
#    git revert <phase-commit-sha>
#    and skip the manual edits above.
```

## Sanity check after revert

```bash
git status                                            # only show files you touched yourself
grep -rn "turbo" inference-flame/src/                 # should match nothing turbo-related
grep -rn "TurboBlockLoader\|VmmArena\|TurboBlock" .   # should be empty across the repo
cargo build -p inference-flame --features turbo       # MUST error cleanly (unknown feature
                                                       # `turbo` after Cargo.toml revert)
```

If any hits remain, they are stragglers from this phase that the revert procedure missed — file as a bug against the revert prompt.

## Why a phase might be reverted

- `turbo_tensor_over_vmm` test reveals BF16View reuse incompatible with a kernel — blocks v4 premise; need to fall back to v2 (new flame-core storage arm).
- VMM event-gated Drop has a race the Bug Fixer caught — needs redesign.
- Bench shows Total wall-clock regressed materially with turbo on, and the cause is structural (not just a missed prefetch overlap).
- Pollution contract violated — flame-core or flame-diffusion got edits this phase shouldn't have made.

Anything narrower than the above (one test failing, one warning, one edge case) should be fixed in place, not reverted.
