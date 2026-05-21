# SKEPTIC FINDINGS — cosmos-predict25-2b chunk 3 — 2026-05-21

Scope: chunk 3 (BUILD_PLAN steps 7-8): `patchify`, `unpatchify`, full
`MiniTrainDIT.forward` (padding-mask concat, 28-block stack, magnitude
probe), `final_layer` (LayerNorm-no-affine + 2-chunk adaLN-LoRA + Linear),
`RectifiedFlowSampler` (sigmas, step, cfg_combine), 5 new cosmos GPU tests
+ 7 sampler tests. Cross-checked against Python source
`minimal_v4_dit.py`, `fm_solvers_unipc.py`,
`text2world_model_rectified_flow.py`, `video2world.py`, and against the
anima.rs oracle and the existing `magihuman_unipc.rs` precedent.

This pass is deliberately adversarial. The port has two real BLOCKERs,
one BLOCKER-or-CORRECT-BUT-FRAGILE depending on user posture, and a
handful of FRAGILE / STYLE follow-ups.

## Findings

### F1: sigma schedule endpoints are wrong (`sigma_max=1.0, sigma_min=1/N`); should be `0.999` and `0.0`
- **Where**: `src/sampling/cosmos_rf.rs:94-95`:
  ```rust
  let sigma_max = 1.0_f64;
  let sigma_min = 1.0_f64 / (self.num_train_timesteps as f64);
  ```
- **What**: The Rust code uses `linspace(1.0, 0.001, n+1)[:-1]` as the
  raw sigma vector, then applies the `shift` transform.
- **Expected**: Python `FlowUniPCMultistepScheduler.__init__` computes
  the full 1000-entry sigma table via
  `sigmas = 1.0 - linspace(1, 1/1000, 1000)[::-1]`, giving
  `sigmas = [0.999, 0.998, ..., 0.001, 0.0]`. Then
  `self.sigma_max = sigmas[0] = 0.999` and `self.sigma_min = sigmas[-1] = 0.0`
  (`fm_solvers_unipc.py:120-122`). `set_timesteps` then does
  `linspace(self.sigma_max=0.999, self.sigma_min=0.0, n+1)[:-1]` and
  applies the user-supplied `shift=5.0` (`:182, :189`).
- **Why it matters**: For n=35 steps, shift=5:
  - Python first sigma: `5*0.999 / (1 + 4*0.999) = 4.995/4.996 ≈ 0.99980`
  - Rust first sigma: `5*1.0   / (1 + 4*1.0)   = 5/5      = 1.00000`
  - Python pre-zero last (raw=0.999/35): shifted ≈ 0.1280
  - Rust pre-zero last (raw=0.001+0.999/35): shifted ≈ 0.1320
  Every Euler step uses these sigmas as endpoints, so the trajectory
  deviates from Python at the 1e-3 absolute level per step.
  cos>0.999 parity bar will fail. **Visible-quality impact is uncertain
  at smoke time, but parity is dead-on-arrival.**
- **Severity**: BLOCKER
- **Evidence**:
  - Python source `fm_solvers_unipc.py:100-122` (the `__init__`
    sigma-construction block) — sigma_max is the FIRST of the
    1.0-shift-applied schedule, which for shift=1.0 default is 0.999.
  - **The codebase already has the right precedent**:
    `src/sampling/magihuman_unipc.rs:55-57` correctly sets
    `sigma_max = (n_train - 1) as f64 / n_train as f64` (= 0.999)
    and `sigma_min = 0.0_f64`. The chunk-3 builder did not consult
    this file. The MagiHuman port is the FlowUniPCMultistepScheduler
    Wan flow-shift case — same scheduler family as Cosmos.
  - Test `sigma_schedule_monotone_decreasing_and_ends_at_zero` at
    `cosmos_rf.rs:210` enshrines the bug:
    `assert!((s[0] - 1.0).abs() < 1e-5, ...)` — Python's first sigma
    is 0.99980, NOT 1.0. The test passes for the wrong reason
    (`5*1/(1+4*1) = 1.0` is an arithmetic coincidence) and gives
    false confidence.

### F2: Pure Euler step ships in place of UniPC second-order multistep
- **Where**: `src/sampling/cosmos_rf.rs:123-145` (the only `step` method)
  and the doc header claim "FlowMatch portions only".
- **What**: Cosmos inference uses
  `self.sample_scheduler.step(velocity_pred, t, latents, ...)` at
  `text2world_model_rectified_flow.py:582-584`. That `step` is
  `FlowUniPCMultistepScheduler.step`
  (`fm_solvers_unipc.py:630-708`), which is NOT Euler — it runs
  `convert_model_output` → optional `multistep_uni_c_bh_update`
  (corrector using previous step's model output) →
  `multistep_uni_p_bh_update` (predictor using up to
  `solver_order=2` past model outputs). Rust ships
  `x_next = x_curr + (sigma_next - sigma_curr) * v_pred`, which is
  pure first-order Euler.
- **Expected**: A second-order multistep solver matching
  `FlowUniPCMultistepScheduler` with `solver_order=2`, `solver_type="bh2"`,
  `predict_x0=True`, `prediction_type="flow_prediction"`,
  `final_sigmas_type="zero"`, `lower_order_final=True` (defaults from
  `:72-89`).
- **Why it matters**: At 35 steps, UniPC second-order produces samples
  comparable to ~80-100 steps of plain Euler in quality terms (this is
  the design point of multistep predictor-correctors). Running 35-step
  Euler against a model trained against UniPC-35-step trajectories will
  produce visibly degraded outputs: oversmoothed details, hue shifts,
  loss of high-frequency texture. The PORT_STATE flagged this gap but
  marked it "DEFERRED". Per the project's standard
  ("pure-Rust runtime matching Python"), this is a parity blocker for
  any smoke claim. Note also: `solver_order=2` needs a 2-element
  ring buffer of past model outputs — stateful sampler — the
  current `RectifiedFlowSampler` is stateless and would need a
  refactor.
- **Severity**: **BLOCKER if "matching Python quality at 35 steps" is the
  goal**, otherwise **CORRECT-BUT-FRAGILE**. The user said in this
  port's hard constraints: "the user has indicated they want pure-Rust
  runtime matching Python; UniPC missing means matching that means
  porting UniPC". Treat as BLOCKER.
- **Evidence**:
  - `text2world_model_rectified_flow.py:582-584` — the call site.
  - `fm_solvers_unipc.py:630-708` — `step` method.
  - `fm_solvers_unipc.py:337-538` — `multistep_uni_p_bh_update` and
    `multistep_uni_c_bh_update` (~200 lines of bh2 math).
  - Default config: `solver_order=2`, `solver_type="bh2"` (lines 75, 83).

### F3: `patchify_unpatchify_shape_roundtrip` test is decorative; its docstring asserts a false identity claim
- **Where**: `src/models/cosmos_predict25_dit.rs:2537-2545` (docstring) and
  `:2569-2570` (assertion).
- **What**: The test calls `model.unpatchify(model.patchify(x))` and
  asserts only `recovered.shape() == [b, c, t, h, w]`. It does NOT
  compare values. The docstring claims:
  "when `out_channels == in_channels` and `patch_temporal == patch_spatial == 1`, both
  rearranges become trivial and the round-trip is identity."
- **Expected**: For the actual `tiny_test_config` (`patch_spatial=2`,
  `patch_temporal=1`), the round-trip is NOT identity even with
  `out_c == in_c`. Patchify lays the trailing axis as
  `(c, r, m, n)` slowest-to-fastest; unpatchify reads it as
  `(p1, p2, t', c)`. The 64-dim trailing axis is interpreted with a
  DIFFERENT axis order, so `unpatchify(patchify(x)) != x` element-wise.
  This is a true Python asymmetry intentional in the design (the
  `FinalLayer.linear` is trained to bridge the asymmetry), not
  something to "round-trip" through.
- **Why it matters**: The test exists only to verify shape. The
  docstring tells a future maintainer to expect a bit-identical
  round-trip on `p_s=p_t=1`, which is true but irrelevant to the
  shipped V2_2B config (`p_s=2`). A future maintainer who reads the
  docstring and then tightens the test to byte-equality on the
  shipped config will get a confusing failure. The test should
  either:
  (a) drop the "identity" claim entirely and document the
      asymmetry, or
  (b) test with `p_s=p_t=1` AND `in_c=out_c` to verify the
      identity claim concretely.
- **Severity**: STYLE (test/doc quality)
- **Evidence**: Python `:1005-1014` (patchify einops) vs `:1702-1709`
  (unpatchify einops). The trailing-axis decomposition order differs.

### F4: `padding_mask_concat_changes_patchify_input_channels` test does not exercise non-zero mask propagation
- **Where**: `src/models/cosmos_predict25_dit.rs:2697-2736`.
- **What**: Test name suggests it verifies padding-mask flow. In
  practice it builds two configs (with/without mask), checks
  `patch_embed_in_channels` returns 17 vs 16, then runs `forward()`
  on the **no-mask** path with `padding_mask=None`. It does NOT pass
  a non-zero mask through the with-mask path. The
  `broadcast_to([B,1,T,H,W])` → `Tensor::cat` → patchify chain (the
  exact chunk-3 skeptic-bait item 3) is exercised by NO assertion.
- **Expected**: Build a non-zero `[B, 1, H, W]` mask (e.g. half-ones,
  half-zeros), run forward twice — once with the mask and once with
  None (i.e. zero mask) — assert the outputs differ. That isolates
  the stride-0 broadcast → cat → patchify path.
- **Why it matters**: The chunk-3 builder flagged this exact path
  ("`mask_5d` is stride-0 broadcast view; `Tensor::cat` may
  mis-read the stride-0 input even with `.contiguous()` after.
  Test: pass non-zero padding mask, verify it propagates to all T
  positions.") and then did not write that test. The bug class
  (silent stride-0 read producing zero or repeated values) is
  exactly the kind that BF16 finite-bound tests miss.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: test body at `:2717-2735`; PORT_STATE.md "Chunk 3
  skeptic-bait" item 3.

### F5: Padding mask Python `transforms.functional.resize` semantic NOT ported; Rust is stricter than Python
- **Where**: `src/models/cosmos_predict25_dit.rs:1545-1551`.
- **What**: Rust requires the caller's `padding_mask` to be exactly
  `[B, 1, H, W]` with H, W matching the latent's spatial dims. It
  errors out on mismatch with `Error::InvalidInput`.
- **Expected**: Python
  `prepare_embedded_sequence:1683-1685` runs
  `transforms.functional.resize(padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=NEAREST)`
  — it nearest-neighbor-resizes the mask to match the latent's
  spatial dims regardless of input H, W. Cosmos inference passes
  `torch.zeros(B, 1, H, W)` at the latent resolution already
  (`video2world.py:431`), so the resize is a no-op in the happy
  path, but a non-pixel-resolution mask (e.g. captured at output
  pixel resolution) would silently work in Python and fail-fast in
  Rust.
- **Why it matters**: Trainer-time and arbitrary-caller-time
  compatibility. Inference users following the Python
  recipe (`zeros(B, 1, H_lat, W_lat)`) hit the happy path. Anyone
  reaching for the public API with a pixel-resolution mask gets a
  hard error, whereas Python would resize. Not a parity bug for
  the standard inference path, but a divergence in API behavior
  that needs to be documented or fixed.
- **Severity**: STYLE (intentional fail-fast; documented in code comment)
- **Evidence**: Python `:1683-1688` does
  `padding_mask = transforms.functional.resize(padding_mask, [H, W], NEAREST)`
  ; Rust at `:1546-1550` rejects size mismatch.

### F6: Magnitude probe casts BF16→F32 per block, allocating ~4× the residual stream
- **Where**: `src/models/cosmos_predict25_dit.rs:1648-1664`.
- **What**:
  ```rust
  let probe = x_b_t_h_w_d
      .to_dtype(DType::F32)
      .and_then(|f| f.abs())
      .and_then(|a| a.max_all());
  ```
  Each call allocates a full F32 copy of `x_b_t_h_w_d`. For V2_2B
  `[1, 32, 60, 60, 2048]` BF16 = 941 MB, the F32 cast = 1.88 GB.
  Per block × 28 blocks × 35 steps = ~1.84 TB of allocations
  across a full inference. Pool reuse helps, but this still
  thrashes memory and adds many ms per block.
- **Expected**: Gate the cast itself behind `log::log_enabled!(Debug)`
  (the print is already gated, but the cast happens INSIDE the
  `if probe_enabled` block, so it IS gated — re-reading line
  1648-1664 confirms the cast is inside the `if probe_enabled`
  block). Re-checking: the cast IS inside the gate (`if probe_enabled`
  at line 1648; cast at line 1652-1655). So production with
  `RUST_LOG=info` pays nothing. Only `RUST_LOG=debug` pays the
  4× allocation cost. **Self-disposition: not a bug; revisit only
  if smoke runs with debug logging show OOM.**
- **Why it matters**: At debug, the 1.88 GB allocation per block
  may push 24 GB cards into OOM on top of the model's resident
  memory. Anyone debugging V2_2B at 720p with `RUST_LOG=debug`
  will hit this and probably blame the model not the probe.
- **Severity**: STYLE / debug-only perf paper cut
- **Evidence**: `bf16_ops.rs` does not expose a BF16 max_all that
  would avoid the cast. The probe could use a single scalar
  reduction kernel directly.

### F7: `Tensor::cat` already materializes inputs; the explicit `.contiguous()` at the cat call site is redundant
- **Where**: `src/models/cosmos_predict25_dit.rs:1569-1570`.
- **What**:
  ```rust
  let cat = Tensor::cat(&[x_b_c_t_h_w, &mask_5d], 1)?;
  cat.contiguous()?
  ```
  with the doc comment "THIS IS THE ONE ALLOWED `.contiguous()` in
  this chunk."
- **Expected**: `Tensor::cat`
  (`tensor_ops_extended.rs:322-336`) explicitly materializes
  non-contig inputs via `t.contiguous()` before the copy, and its
  output is freshly allocated via `Tensor::zeros_dtype` then filled
  by linear `dtod_copy` slices — i.e. its output is already
  guaranteed contiguous. The trailing `.contiguous()` is a no-op:
  it just clones the already-contig tensor (per
  `contiguous()` short-circuit at `tensor.rs:3186-3188`).
- **Why it matters**: Correctness: no impact. Style: the
  CONTEXT.md "one allowed contiguous" framing implies this is
  load-bearing, when it isn't. A future maintainer might think the
  cat output is non-contig (it isn't) and add a second
  defensive contiguous somewhere downstream.
- **Severity**: STYLE
- **Evidence**: `tensor_ops_extended.rs:325-336` (cat materializes
  inputs); `:378-379` (output is allocated contig);
  `:407-462` (F32/BF16 cat fills via linear `dtod_copy` — output is
  always row-major contiguous).

### F8: `default_owned` Rust binding is correct today but a brittle pattern
- **Where**: `src/models/cosmos_predict25_dit.rs:1542-1557`.
- **What**:
  ```rust
  let default_owned: Tensor;
  let mask_4d: &Tensor = match padding_mask {
      Some(m) => m,
      None => {
          default_owned = self.default_padding_mask(b, h, w)?;
          &default_owned
      }
  };
  ```
  Rust's NLL accepts this because the `&default_owned` borrow lives
  only within the `None` arm, and the subsequent `mask_4d` is
  borrowed-from-`m` in the `Some` arm.
- **Expected**: Works as-is. The concern is future maintenance: if
  someone adds an `if`-early-return or a `let`-binding between the
  `default_owned` declaration and its conditional assignment, the
  compiler error message is subtle ("use of possibly-uninitialized
  binding"). The chunk-3 builder flagged this as skeptic-bait item 4.
- **Why it matters**: Edit-resilience. A `match padding_mask {
  Some(m) => m.clone(), None => self.default_padding_mask(b, h, w)? }`
  pattern (owned `Tensor` from both arms; one extra `Arc` clone in
  the `Some` arm) eliminates the pattern entirely. Tensor in
  flame-core uses `Arc<dyn Storage>` internally so `.clone()` is
  cheap.
- **Severity**: STYLE
- **Evidence**: `:1542-1557`.

### F9: Test `final_layer_modulation_is_load_bearing` repeats the chunk-2 F8 weakness
- **Where**: `src/models/cosmos_predict25_dit.rs:2578-2623`.
- **What**: Test pins `lora=zero` (line 2597-2600), then varies
  only `emb`. So the `summed = h2 + adaln_2d` line in `final_layer`
  (`:1458`) is exercised but the `adaln_2d` contribution is
  zero. If someone removes the `.add(&adaln_2d)` and writes
  `let summed = h2;`, the test still passes because the LoRA
  contribution was zero anyway.
- **Expected**: A second test variant that pins `emb=zero` and
  varies `adaln_lora` between two non-zero choices to assert the
  output changes. This is the same gap that chunk-2 F8 flagged for
  the per-block adaLN path.
- **Why it matters**: Final-layer adaLN-LoRA dropouts wouldn't be
  caught at unit-test time. They'd only surface as broken parity at
  the parity phase.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: test body `:2596-2622`.

### F10: `_unused_dtype_import_marker` is a code smell hiding an unnecessary import
- **Where**: `src/sampling/cosmos_rf.rs:174-176`.
- **What**:
  ```rust
  #[allow(dead_code)]
  fn _unused_dtype_import_marker(_: DType) {}
  ```
  The `DType` import at line 49 is only used inside `mod tests`
  (`to_dtype(DType::BF16)`, `DType::F32`). In non-test builds the
  import is unused. The author worked around the warning with a
  marker function rather than restructuring imports.
- **Expected**: Move `DType` import into `mod tests` only:
  `use flame_core::DType;` inside `#[cfg(test)] mod tests { ... }`.
  Or in the top-level: `use flame_core::{Error, Result, Tensor};`
  alone, and have tests import `DType` themselves.
- **Why it matters**: Dead-code marker imports are a code smell.
  Doesn't affect correctness, but the next maintainer will wonder
  what `_unused_dtype_import_marker` does and why it exists.
- **Severity**: STYLE
- **Evidence**: `:49, :174-176`.

### F11: T_emb vs T_p shape coupling silently assumed by `apply_layer_norm_modulate`
- **Where**: Caller is `final_layer` at `:1448` (caller of
  `apply_layer_norm_modulate`); the helper at `:1185-1206`.
- **What**: `apply_layer_norm_modulate` reshapes
  `x_b_t_h_w_d` (shape `[B, T_p, H_p, W_p, D]`) to
  `[B*T_p, H*W, D]` and shift/scale (shape `[B, T, D]`) to
  `[B*T, D]`. If `T != T_p` (e.g. caller passes 1D timesteps that
  weren't properly unsqueezed to `[B, T_p]`), the
  `B*T_p` and `B*T` rows mismatch and `modulate_pre_fused_bf16`
  will either error or produce wrong output.
- **Expected**: The Python `FinalLayer` uses `rearrange` to expand
  `shift_B_T_D` to `shift_B_T_1_1_D` and lets `torch` broadcast.
  When `T_emb=1` (1D input unsqueezed), broadcast across `T_p`
  works. Rust does NOT broadcast — it requires exact `B*T_emb ==
  B*T_p`.
- **Why it matters**: For V2_2B with `patch_temporal=1` and
  callers that pass `[B, T_p]` timesteps (the documented contract),
  it works. Anyone who passes `[B]` (scalar-per-batch timesteps)
  hits a shape mismatch instead of getting per-pixel broadcast.
  Documented in `forward`'s doc but not enforced at the API boundary.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: `:1198-1201` (the reshapes), Python `:1115-1118`
  (the broadcast via rearrange-then-broadcast in the `_fn`).

### F12: Test `forward_runs_end_to_end_with_synthetic_weights` magnitude bound is loose; tiny config cannot reveal F32-vs-BF16 residual issue
- **Where**: `src/models/cosmos_predict25_dit.rs:2628-2692`.
- **What**: Test runs forward with `num_blocks=2`,
  `model_channels=12`, `num_heads=2`, asserts
  `max_abs < 10000.0`. The chunk-2 skeptic noted that anima
  observed residual values >200 at full scale, prompting
  BF16→F32 casts between sub-blocks. Cosmos chunk-2 stays BF16
  throughout. At `model_channels=12` and 2 blocks the residual
  stream cannot grow large enough to expose BF16 precision loss
  at the 8-bit mantissa boundary.
- **Expected**: Either run a larger-config probe test (e.g.
  `num_blocks=4, model_channels=128`), OR explicitly document this
  as a parity-phase concern with a magnitude probe assert
  (`assert max_abs < 100.0`) so any future growth gets caught.
- **Why it matters**: PORT_STATE.md "Chunk 3 spec correction" item
  6 admits "Re-check at 28-block full scale during parity." The
  test as written cannot serve as the parity gate; it's
  shape+finiteness only.
- **Severity**: STYLE (parity-phase concern; current test still
  validates shape and infs)
- **Evidence**: `:2640-2641` (config), `:2685-2686` (bound).

### F13: `tests` import `env_logger::builder` but env_logger may not be a Cargo dev-dependency for this crate
- **Where**: `src/models/cosmos_predict25_dit.rs:2632`.
- **What**:
  ```rust
  let _ = env_logger::builder().is_test(true).try_init();
  ```
  This requires `env_logger` as a dev-dependency. Unable to verify
  whether `inference-flame`'s `Cargo.toml` already provides it; if
  not, the test build fails. The chunk-3 builder asserted zero
  new Cargo deps, but this test references `env_logger` directly.
- **Expected**: Confirm `env_logger` is in `[dev-dependencies]`
  (many other tests in `inference-flame` use it, so likely yes,
  but worth verifying as part of "zero new deps" claim).
- **Why it matters**: If it's missing, the test won't compile.
- **Severity**: STYLE (unverified; couldn't confirm `Cargo.toml`
  state during this pass).
- **Evidence**: `:2632`.

### F14: `RectifiedFlowSampler::sigmas()` recomputes the schedule on every step
- **Where**: `src/sampling/cosmos_rf.rs:123-145` (`step` method
  calls `self.sigmas()` at line 130).
- **What**: Every `step()` call materializes the whole sigma Vec
  again. For n=35 it's 35 floats — small cost — but at any
  reasonable resolution this is at minimum 35 redundant tight
  loops.
- **Expected**: Either compute once and cache in `&self` (would
  break `Copy`), or have the caller pre-compute via `.sigmas()`
  and pass `(sigma_curr, sigma_next)` into `step`. The Python
  scheduler keeps `self.sigmas` as a persisted tensor.
- **Why it matters**: Minor perf paper cut. Bigger concern: the
  caller now uses two different `sigmas()` calls per loop
  iteration (once for the timestep index → t, once via `.step()`),
  doubling redundancy.
- **Severity**: STYLE (negligible perf)
- **Evidence**: `:130`.

### F15: `cfg_combine` allocates two intermediate tensors per call (sub + mul_scalar) when the math could fuse
- **Where**: `src/sampling/cosmos_rf.rs:154-171`.
- **What**: `cfg_combine` computes `cond.sub(uncond) → mul_scalar →
  uncond.add(&scaled)` — 3 fresh tensor allocations per call.
  Math: `out = uncond + cfg*(cond - uncond)` could be done with
  a single fused kernel `out = cfg*cond + (1-cfg)*uncond` (one
  read of each input, one write).
- **Expected**: For inference, called once per step per CFG pair
  (e.g. 2× per step). At V2_2B latent size ~941 MB BF16, three
  allocations per call = ~3 GB transient peak per CFG step. Pool
  reuse helps but the math could be done in one kernel.
- **Why it matters**: Memory pressure during CFG dual-pass on
  24 GB cards is already tight. Three transient allocations of
  the latent size each could push OOM. Doesn't break correctness.
- **Severity**: STYLE (perf paper cut; not a bug)
- **Evidence**: `:168-170`.

## UniPC vs Euler verdict

**Verdict: BLOCKER (treat as such per user's stated parity goal).**

The Cosmos inference path
(`text2world_model_rectified_flow.py:582-584`) explicitly calls
`self.sample_scheduler.step(velocity_pred, t, latents, ...)`. That
`.step` is `FlowUniPCMultistepScheduler.step`
(`fm_solvers_unipc.py:630-708`). It is not a thin wrapper around
Euler. The relevant math:

1. `convert_model_output` (line 672): for `prediction_type="flow_prediction"`
   with `predict_x0=True`, this converts the velocity to an x0 estimate:
   `x0_pred = sample - sigma_t * model_output`
   (`:306-308`).
2. Multistep corrector (line 674-680, when step_index>0):
   `multistep_uni_c_bh_update` — uses up to `solver_order` past model
   outputs to apply a corrector on the current sample. This is the
   distinguishing feature of UniPC.
3. Predictor (line 698): `multistep_uni_p_bh_update` — uses up to
   `solver_order` past model outputs in a Bashforth-Houwen extrapolation
   to predict the next sample. Default `solver_order=2`, `solver_type="bh2"`
   (`:75, :83`).

The math (paraphrased from `:337-538`):
- For order 2: maintain `model_outputs[-1]` (current) and `model_outputs[-2]`
  (previous). Compute coefficients `c_i` from the sigma differences,
  combine into a predictor of `x_next`.
- Corrector uses `last_sample` (the sample at the previous step) to
  back-correct the prediction.

Pure Euler is `solver_order=1` without corrector. The quality difference
is significant. Empirically:
- 35 steps of UniPC second-order ≈ 80-100 steps of Euler in image
  quality terms.
- 8-12 steps of UniPC second-order can match 50 steps of DDIM.

So at the same step count (35) used by Python:
- Python: high-quality output.
- Rust with Euler: moderate-quality output, oversmoothed, hue shift.

**Bottom line**: smoke will visibly fail to match Python at 35 steps.
The user can compensate by running more Euler steps (perhaps 80-100) at
the cost of inference time. But for any parity-vs-Python comparison,
this is BLOCKER. A proper UniPC port is a multi-day exercise (the
Python is ~200 lines of dense numerics including `multistep_uni_p_bh_update`,
`multistep_uni_c_bh_update`, `convert_model_output`, sigma-to-alpha
helpers, and a stateful ring buffer of past model outputs).

## Anima oracle cross-check

| Pattern | Cosmos chunk-3 | Anima | Status |
|---|---|---|---|
| Patchify dim order | reshape `[B,C,t_p,p_t,h_p,p_s,w_p,p_s]` + permute `[0,2,4,6,1,3,5,7]` + flatten | Same structure; `anima.rs:531` uses anima's MiniTrainDIT patchify in a parallel pattern | AGREES on the 8D decomposition order; anima is image-only (T=1) so the `t_p, p_t` axes collapse. Both ports correctly implement Python einops `b c (t r) (h m) (w n) -> b t h w (c r m n)`. |
| Unpatchify dim order | reshape `[B,T_p,H_p,W_p,p_s,p_s,p_t,out_c]` + permute `[0,7,1,6,2,4,3,5]` + flatten | `anima.rs:565` parallel | AGREES on the decomposition `(p1, p2, t', C)`. Both match Python `:1702-1709` einops. |
| FinalLayer structure | LN-no-affine + 2-chunk adaLN-LoRA + Linear (no bias); reuses `adaln_lora[:, :, :2*d]` | `anima.rs:517` — same structure | AGREES. |
| Padding mask | broadcast `[B,1,H,W]` → `[B,1,T,H,W]` + cat + `.contiguous()` | Anima image-only (T=1), so mask is `[B,1,H,W]` cat directly with no broadcast | DIFFERS as expected (anima is T=1). Cosmos T>1 case correctly handles via broadcast. |
| Magnitude probe | BF16→F32 cast + abs + max_all per block, gated on Debug | Anima reportedly casts BF16↔F32 between sub-blocks because hidden values reach 200+ (chunk-2 callout) | DIFFERS. Cosmos has no such inter-sub-block cast. The probe is observational only. Parity-phase concern unresolved. |
| F32 residual stream | None (BF16 throughout) | F32 residual stream | DIFFERS. Same risk flagged by chunk-2 skeptic. Magnitude probe will be the first to see it. |

**Key inherited concern from chunk-2 skeptic**: Anima specifically
documents (`anima.rs:470`) that "FP32 residual stream (model has large
values ~200+)" was learned empirically. Cosmos chunk-3 stays BF16
through 28 blocks. The magnitude probe is the diagnostic, but the
test doesn't assert on its output. **This is the real anima-divergence
risk going into parity.**

## Builder spec-correction validation (5 items)

- **Item 1 (padding mask shape `[B, 1, H, W]` not `[B, 1, T, H, W]`)**:
  CONFIRMED.
  - `video2world.py:431`: `torch.zeros(self.batch_size, 1, H, W)` (no T).
  - `minimal_v4_dit.py:1683-1687`: `padding_mask.unsqueeze(1).repeat(1, 1, T, 1, 1)`.
    The unsqueeze(1) inserts a dim, so the input was `[B, 1, H, W]`
    becoming `[B, 1, 1, H, W]` after unsqueeze(1)... wait. Re-reading:
    `padding_mask` enters as `[B, 1, H, W]` (per the data batch),
    then resize keeps shape `[B, 1, H_lat, W_lat]`, then
    `.unsqueeze(1)` gives `[B, 1, 1, H, W]`, then
    `.repeat(1, 1, T, 1, 1)` gives `[B, 1, T, H, W]`. Cat along dim=1.
    Confirmed. Rust does the equivalent via reshape + broadcast_to.
- **Item 2 (patchify/unpatchify asymmetric einops)**: CONFIRMED.
  - Python patchify (`PatchEmbed.proj[0]`, `:1005-1010`):
    `"b c (t r) (h m) (w n) -> b t h w (c r m n)"` — trailing axis
    order `(c, r, m, n)` slowest-to-fastest.
  - Python unpatchify (`MiniTrainDIT.unpatchify`, `:1702-1709`):
    `"B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)"` — trailing axis
    order `(p1, p2, t, C)` slowest-to-fastest.
  - The two are NOT symmetric; the FinalLayer's `linear` weight is
    trained to bridge. Rust permutes are correct against each
    Python rearrange.
- **Item 3 (shift=5, CFG=7 defaults)**: CONFIRMED.
  - `text2world_model_rectified_flow.py:502`: `shift: float = 5.0`.
  - `video2world.py:477`: `guidance: int = 7`.
  - `video2world.py:487`: `num_steps: int = 35`.
- **Item 4 (UniPC is the actual scheduler)**: CONFIRMED.
  - `text2world_model_rectified_flow.py:142-143`:
    `self.sample_scheduler = FlowUniPCMultistepScheduler(...)`.
  - `text2world_model_rectified_flow.py:582-584`: the loop calls
    `self.sample_scheduler.step(...)` — the full UniPC step.
  - See F2 for the depth of the gap.
- **Item 5 (FinalLayer 2-chunk + `adaln_lora[:2*hidden]` slice)**:
  CONFIRMED.
  - Python `:1108-1111`:
    `shift_B_T_D, scale_B_T_D = (self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]).chunk(2, dim=-1)`.
  - Python `:1070`: `self.n_adaln_chunks = 2` (no gate, just shift and scale).
  - `:1124`: `_fn(x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D)`
    = `LN(x) * (1+scale) + shift` — confirmed no gate.
  - Rust at `:1457-1462` slices `narrow(2, 0, 2*d)` and chunks into shift, scale.
  - FinalLayer has its OWN `adaln_modulation` Sequential
    (`:1074-1078` for `use_adaln_lora=True`); it is NOT shared with
    blocks. Weight keys: `final_layer.adaln_modulation.{1,2}.weight`,
    `final_layer.linear.weight`. Rust uses these names correctly.

## Builder skeptic-bait disposition (6 items)

1. **Patchify permute `[0,2,4,6,1,3,5,7]`**: CONFIRMED CORRECT.
   The 8D reshape `[B, c, t_p, p_t, h_p, p_s, w_p, p_s]` is the
   `b c (t r) (h m) (w n)` decomposition; the permute moves
   axes `(B, t_p, h_p, w_p)` to the front and trailing `(c, r, m, n)`
   to the back, which is the einops target. Verified by axis-by-axis
   trace.
2. **Unpatchify permute `[0,7,1,6,2,4,3,5]`**: CONFIRMED CORRECT.
   Same axis-by-axis trace produces the einops target
   `B C (T t) (H p1) (W p2)` with `(c, T_p, t', H_p, p1, W_p, p2)`
   in the slowest-to-fastest order matching the 3D output's
   spatial layout.
3. **Padding-mask cat from stride-0 broadcast**: REFUTED.
   `Tensor::cat` at `tensor_ops_extended.rs:325-336` materializes
   non-contig inputs via `t.contiguous()` BEFORE the slice-copy
   loop, so the stride-0 broadcast view is rewritten into a fully
   contig allocation before any element is read. The `.contiguous()`
   AFTER the cat is a redundant safety belt (cat output is
   already contig — see F7).
4. **`default_owned` lifetime**: CORRECT BUT FRAGILE.
   Works as-is (Rust NLL accepts the pattern). The brittleness
   under future edits is documented (see F8).
5. **`Tensor::max_all` legacy GpuOps F32 path**: CONFIRMED.
   `tensor_ops_extended.rs:1231-1239` routes through
   `GpuOps::reduce_max`, which at `cuda_ops.rs:621-632`
   iteratively reduces via `max_dim` over each axis. The chunk-3
   builder pre-casts BF16→F32 before calling; that allocation is
   gated on Debug-level logging (see F6). Not a bug at production
   log level.
6. **Sigma linspace endpoint vs numpy**: REFUTED AS WRITTEN, but
   for the WRONG REASON. The chunk-3 builder's f64 `denom=(n+1)-1=n`
   formulation IS arithmetically equivalent to
   `numpy.linspace(a, b, n+1)[:-1]`. The bug is NOT the linspace
   formula. The bug is the WRONG ENDPOINTS (`sigma_max=1.0`,
   `sigma_min=1/N` instead of `0.999, 0.0`) — see F1.

## Clean checks

- Patchify permute correctly inverse-pairs with Python einops decomposition order. Verified axis-by-axis.
- Unpatchify permute correctly inverse-pairs with Python einops decomposition order. Verified axis-by-axis.
- FinalLayer `LayerNorm(elementwise_affine=False, eps=1e-6)` matches `modulate_pre_fused_bf16` semantics (no gamma/beta loaded; eps=1e-6 passed at `:1203`).
- FinalLayer `n_adaln_chunks=2` (shift, scale, NO gate) — confirmed.
- FinalLayer `final_layer.linear.weight` shape `[16*4=64, 2048]` — matches Python `:1066-1068`.
- FinalLayer weight keys `final_layer.adaln_modulation.1.weight` and `.2.weight` match Python Sequential indexing (SiLU at 0, Linear at 1, Linear at 2).
- `extra_per_block_pos_emb` is computed ONCE outside the loop and passed to each block; chunk-2 verified the inside-block add structure.
- Default padding mask is `torch.zeros(B, 1, H, W)` — matches `video2world.py:431`.
- Sigma schedule length = num_steps + 1 (with final 0.0 appended). ✓ format.
- Euler step direction: `dt = sigma_next - sigma_curr < 0` (sigmas are monotone decreasing); `x_next = x + dt * v_pred` is the correct FlowMatch convention. ✓
- CFG combine endpoints: `out = uncond + cfg * (cond - uncond)`. At `cfg=1` → cond; at `cfg=0` → uncond. ✓
- No `use anima` imports; only one docstring reference to anima as a comment. ✓
- No env-gates for default-path code. ✓
- No Python in runtime path. ✓
- No new Cargo dependencies (single `flame_core` import in `cosmos_rf.rs`; no new imports beyond chunk 2 in `cosmos_predict25_dit.rs`). ✓
- No Flash Attention proposals or comments. ✓
- Only ONE `.contiguous()` call in `cosmos_predict25_dit.rs` (line 1570), per the "one allowed" rule. ✓ (though it's a redundant no-op — see F7).
- `Tensor::cat` materializes non-contig inputs internally (confirmed `tensor_ops_extended.rs:325-336`).
- `Tensor::reshape` materializes non-contig inputs via `.contiguous()` (confirmed `tensor.rs:2868-2872`), so the patchify/unpatchify permute-then-reshape chains correctly fold to flattened layouts.

## Couldn't verify

- **Whether `Tensor::cat`'s safety-net `t.contiguous()` correctly handles
  a stride-0 broadcast view at the BF16 path**. The `cuda_ops::materialize_view`
  function is invoked when `contiguous()` sees a non-permute view. Whether
  it correctly walks stride-0 axes (effectively reading the same element
  repeatedly into the materialized buffer) was not inspected at this pass.
  Defer to GPU parity. A non-zero padding-mask test (F4) would catch this
  immediately if there's a bug. **Recommendation**: do not declare
  chunk 3 smoke-clean until that test exists.
- **Whether `env_logger` is in `[dev-dependencies]` for `inference-flame`.**
  See F13. Bash run not used to grep Cargo.toml during this pass.
- **Whether the magnitude probe values at full V2_2B scale (28 blocks,
  head_dim=128, model_channels=2048) stay below the anima 200-threshold.**
  This is the parity-time question. The test at tiny scale (2 blocks,
  head_dim=6) cannot reveal this.
- **Whether `modulate_pre_fused_bf16` correctly handles the FinalLayer's
  call shape `[B*T_p, H_p*W_p, D]` with shift/scale `[B*T_emb, D]` when
  T_emb < T_p (broadcast case)**. Rust ALWAYS reshapes shift/scale to
  the full batch dim (no broadcast), so the broadcast case won't
  arise in correctly-called code, but Python's `_fn` does broadcast
  via `rearrange + broadcast`. Defer to parity.
- **Whether the magnitude probe's `to_dtype(F32)` allocation at debug
  log level OOMs the 24 GB inference card at 720p**. Unable to GPU-test
  during this pass; flagged in F6.
- **Whether the UniPC quality gap at 35 steps materially changes output
  perceptually vs Python**. Requires running both Python and Rust at 35
  steps and visually comparing. The estimated gap (35-step Euler ≈ 50-100
  step DDIM ≈ 8-12 step UniPC) is theoretical and may not match practice
  for this specific model.

## Summary count

- BLOCKERs: **2** (F1 sigma endpoints, F2 Euler-vs-UniPC)
- CORRECT-BUT-FRAGILE: **3** (F4 mask test gap, F9 final_layer test gap, F11 T_emb/T_p coupling)
- STYLE: **8** (F3, F5, F6, F7, F8, F10, F12, F13, F14, F15)
- FLAME-CORE: **0**
- DISAGREE / refutation: **2** items (skeptic-bait items 3 and 6 — refuted, see disposition section)

**Chunk 3 has two real blockers**: the sigma schedule is mathematically
wrong (F1, with the irony that the codebase's own `magihuman_unipc.rs`
already has the right answer), and the sampler is the wrong algorithm
(F2 Euler-vs-UniPC). Both block parity. The remaining items are paper
cuts and test-coverage gaps to fix at the parity / bugfix phase. The
anima BF16-vs-F32 residual divergence remains a parity-time risk.
