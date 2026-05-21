# SKEPTIC FINDINGS — cosmos-predict25-2b — 2026-05-21

Scope: chunk 1 (BUILD_PLAN steps 1-3): skeleton, `sinusoidal_timesteps`,
`timestep_embedding`, `learnable_pos_emb`, `build_cosmos_rope_freqs`, Python
parity scripts. Cross-checked against `anima.rs` oracle.

## Findings

### F1: `learnable_pos_emb` omits the mandatory output normalization
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:445-451`
- **What**: The function returns `pe_t + pe_h + pe_w` directly (after broadcast).
- **Expected**: Python `LearnablePosEmbAxis.generate_embeddings`
  (`minimal_v4_dit.py:850-852`) applies a per-row F32 vector-norm scaling **before
  returning**:
  ```python
  norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
  norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
  return emb / norm.to(emb.dtype)
  ```
  This is a load-bearing normalization (last-dim L2 norm scaled by
  `sqrt(N/M)` where N=norm.numel(), M=emb.numel(), plus 1e-6 epsilon).
- **Why it matters**: Without it, the additive pos emb fed into every
  block has roughly `sqrt(model_channels)`-times the magnitude the model
  was trained against. Silent-failure trap: forward pass runs, output is
  visually wrong, cos≈low.
- **Severity**: BLOCKER
- **Evidence**: `grep "vector_norm\|norm\." cosmos_predict25_dit.rs` returns
  empty. The Rust docstring (`:373-379`) says "broadcast-summing the three
  per-axis learnable parameter slices … (with `interpolation=\"crop\"` we
  just take the first `T`/`H`/`W` rows)" — does not mention normalization.

### F2: `timestep_embedding` omits the post-`t_embedder` RMSNorm
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:349-371`
- **What**: Returns `(sample.clone(), Some(y))` when `use_adaln_lora=true`.
  The sample (sinusoidal embedding) is passed through untouched.
- **Expected**: `MiniTrainDIT.forward` (`minimal_v4_dit.py:1753-1754`)
  applies `self.t_embedding_norm` (TE RMSNorm, eps=1e-6, weight=`t_embedding_norm.weight`)
  to `t_embedding_B_T_D` **after** `t_embedder(...)` returns:
  ```python
  t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
  t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
  ```
  Module declared at `minimal_v4_dit.py:1556`: `self.t_embedding_norm =
  te.pytorch.RMSNorm(model_channels, eps=1e-6)`.
- **Why it matters**: The block-path conditioning is the RMSNorm'd sinusoidal,
  not the raw sinusoidal. Anima oracle does this explicitly at
  `anima.rs:280` (`self.rms_norm(&emb, "net.t_embedding_norm.weight", 1e-6)`).
  Without it, modulation magnitudes are wrong from step 0.
- **Severity**: BLOCKER
- **Evidence**: Anima oracle: `anima.rs:241-283` (`prepare_timestep`)
  applies rms_norm and returns `(t_cond, base_adaln)`. The Rust chunk-1
  function name (`timestep_embedding`) and its docstring frame this as
  "the `TimestepEmbedding.forward`" only — which is technically a
  literal port of `TimestepEmbedding`. But the function is the entry
  point for chunk-1 and its caller will have no obvious place to insert
  the RMSNorm later unless the wiring is explicit. Either the function
  must also apply the norm, OR the public API must explicitly document
  that the caller must apply it. Currently neither.

### F3: `axis_lens` returns pre-patch dimensions; Python uses post-patch
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:198-203`
  (and inline comments at `:60-61`).
- **What**: Returns `(max_frames / patch_temporal, max_img_h, max_img_w)`
  = `(128, 240, 240)` for V2_2B.
- **Expected**: Python `build_pos_embed` (`minimal_v4_dit.py:1627-1629`):
  ```python
  len_h=self.max_img_h // self.patch_spatial,
  len_w=self.max_img_w // self.patch_spatial,
  len_t=self.max_frames // self.patch_temporal,
  ```
  Both Python `pos_embedder` and `extra_pos_embedder` are constructed with
  these *post-patch* lengths. For V2_2B: `(len_t=128, len_h=120, len_w=120)`.
- **Why it matters**: The learnable buffers in the actual checkpoint will be
  `pos_emb_h: [120, 2048]`, `pos_emb_w: [120, 2048]`, `pos_emb_t: [128, 2048]`.
  Today nothing consumes `axis_lens`, but a future builder will reach
  for it to validate buffer shapes, and the comment at `:60-61` ("240
  (latent-patch grid)") will mislead them. The shape-validate check in
  `learnable_pos_emb` (`:412-432`) only asserts `[?, D]` rank and that
  `T/H/W ≤ pe.shape[0]` — it would pass *either* 240 or 120 buffers
  silently.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**:
  - Rust:
    ```rust
    let len_h = self.max_img_h;            // <-- 240, not 120
    let len_w = self.max_img_w;
    ```
  - Python: `len_h=self.max_img_h // self.patch_spatial`

### F4: Unit tests for `build_cosmos_rope_freqs` exercise a duplicated CPU re-implementation, not the production function
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:703-755`
  (`build_cpu_layout`) used by tests `:757-861`.
- **What**: All five layout tests call `build_cpu_layout`, a hand-rolled CPU
  copy of the algorithm. The production function
  `build_cosmos_rope_freqs` is **not invoked anywhere in the test module**.
- **Expected**: At minimum one test that calls `build_cosmos_rope_freqs` and
  compares against `build_cpu_layout` (or against an inline expected value).
  Since the production function takes `&Arc<CudaDevice>`, it can't run on
  a CUDA-less CI box, but a feature-gated test or a `#[cfg(test)]` device
  shim would catch divergence. As-is the 8/8 green is a tautology:
  `build_cpu_layout` passes its own shape contract.
- **Why it matters**: Several real-code paths in `build_cosmos_rope_freqs`
  are **not covered** by these tests, including:
    1. NTK ratio scaling (the helper hard-codes `theta = 10000.0` and
       skips `ratio.powf(dim/(dim-2))`).
    2. `fps.is_none() && t != 1` error path.
    3. The `enable_fps_modulation=false` branch (the helper's branch logic
       differs from production: production at `:577-586` only modulates if
       `enable_fps_modulation && fps.is_some()`; helper at `:725-730`
       modulates whenever `enable_fps_modulation && fps.is_some()`. Same in
       intent, but a regression in either path goes undetected by tests).
    4. The output `Tensor` materialisation and device upload (the helper
       returns `Vec<f32>`, not a `Tensor`).
- **Severity**: CORRECT-BUT-FRAGILE (BLOCKER for test-coverage claims)
- **Evidence**: `grep "build_cosmos_rope_freqs" cosmos_predict25_dit.rs` →
  only line 492 (definition), no test-side call.

### F5: `convert_dit_pt_to_safetensors.py` cannot strip the `net.` prefix
- **Where**: `parity/convert_dit_pt_to_safetensors.py:36`.
- **What**: `_PREFIX_CANDIDATES = ("module.", "ema.", "model.", "_orig_mod.")`.
  No `"net."`.
- **Expected**: Anima oracle's checkpoint convention (`anima.rs:1210-1214`)
  exposes keys as `net.x_embedder.*`, `net.t_embedder.*`, `net.blocks.{i}.*`,
  `net.t_embedding_norm.*`, `net.final_layer.*`, `net.llm_adapter.*`.
  If Cosmos's `.pt` follows the same wrapping convention (likely — it's
  the same training-codebase family: `model.net = MiniTrainDIT(...)`),
  every key in the output safetensors will start with `net.`, but the
  Rust loader at `cosmos_predict25_dit.rs:243-253` will look up
  `t_embedder.1.linear_1.weight` and fail.
- **Why it matters**: Until a real checkpoint is dumped with
  `dump_state_dict_keys.py`, we don't know whether the prefix is present.
  If it IS, every weight lookup fails at load time. Even if it isn't,
  the script lacks a contingency.
- **Severity**: BLOCKER (gates the entire load path) — pending real-checkpoint
  inspection that the builder explicitly deferred (PORT_STATE Open issues #1
  and #3).
- **Evidence**: Anima loader: `anima.rs:1210-1214`. Script: line 36 omits
  `"net."`. Stripping logic at `:68-76` only iterates the candidate tuple.

### F6: `head_dim` validation uses confused boolean logic, error message lies
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:505-509`.
- **What**:
  ```rust
  if head_dim == 0 || head_dim % 6 != 0 && head_dim % 2 != 0 {
      return Err(... "head_dim must be even, got {head_dim}" ...);
  }
  ```
  Rust precedence: `&&` binds tighter than `||`, so this is
  `head_dim == 0 || (head_dim % 6 != 0 && head_dim % 2 != 0)`. That
  accepts `head_dim=128` only because `128 % 2 == 0` makes the
  RHS-conjunction false; it does NOT enforce evenness in any general
  sense. e.g. `head_dim = 10` passes (`10%6 != 0` true, `10%2 != 0` false
  → false → accepted) but `head_dim = 9` (odd, divisible by neither 6
  nor 2 → wait `9%2 != 0` true → accepted? no `9%6 != 0` true and `9%2
  != 0` true → true → rejected). And `head_dim = 3` (`3%6 != 0` true,
  `3%2 != 0` true → rejected). So odd numbers below 6 are rejected, but
  even values pass for the wrong reason.
- **Expected**: Either `head_dim == 0 || head_dim % 6 != 0 || head_dim % 2 != 0`
  (head_dim must be a multiple of 6, given the `/6*2` split) or simply
  `head_dim < 6 || head_dim % 2 != 0` with a different error message.
- **Why it matters**: V2_2B head_dim=128 passes by accident. Anyone reading
  the function thinks evenness is enforced. A future model with
  head_dim=10 (silly example) would be silently accepted; `dim_h = 10/6*2 = 2`,
  `dim_t = 10 - 4 = 6` — the later `dim_h % 2 != 0` check
  (`:515`) would catch nothing because both are even. So invalid splits
  could pass.
- **Severity**: STYLE
- **Evidence**: Line 505 in the source.

### F7: `learnable_pos_emb`'s "no `.contiguous()`" justification is unverified
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:436-443`.
- **What**: The code chains `narrow → reshape → broadcast_to` and then sums:
  ```rust
  let pe_t_b = pe_t.narrow(0, 0, t)?.reshape(&[1, t, 1, 1, d])?.broadcast_to(&target)?;
  // ...
  let sum_thw = pe_t_b.add(&pe_h_b)?.add(&pe_w_b)?;
  ```
  Comment claims "materialising the operands at matching shapes first
  keeps the path off the BF16 contig fast-path assert in `add`."
- **Expected**: `broadcast_to` does NOT materialize — it sets broadcast
  strides on a view. The fast-path assert in `bf16_ops::add` (if any)
  inspects strides, and broadcast strides are not contiguous. The claim
  in the comment may be wrong. No test exercises this code path on GPU.
  Closest analog: `feedback_flame_core_bf16_fused_autograd` memory (BF16
  fused ops + non-contig inputs are a documented hazard).
- **Why it matters**: At first call against a real checkpoint, `add` may
  refuse the broadcast operand, OR worse, take a slow generic path that
  silently writes wrong output. Aliasing risk also: anima docs RoPE
  fails silently with non-contig inputs (see `feedback_flame_conv1d_k1_fast_path`).
- **Severity**: CORRECT-BUT-FRAGILE (unverified — code-only chunk, GPU
  smoke deferred).
- **Evidence**: No GPU smoke run; no test of `learnable_pos_emb` against
  even a synthetic device. `cargo test` runs the layout tests only.

### F8: `from_safetensors` BF16-coercion swallows cast failures
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:245-252`.
- **What**:
  ```rust
  if v.dtype() != DType::BF16 {
      let v_bf16 = v.to_dtype(DType::BF16).unwrap_or(v);
      (k, v_bf16)
  } else { (k, v) }
  ```
  `unwrap_or(v)` silently falls back to the original (non-BF16) tensor
  if `to_dtype` errors.
- **Expected**: Propagate the error. If a weight tensor is the wrong
  dtype AND `to_dtype` cannot cast it, the model will load with mixed
  dtypes (e.g. F32 entries) and downstream BF16-only kernels will
  reject it at random points.
- **Why it matters**: Defensive error swallowing in a load path. Should
  fail loudly. Particularly so because the convert script already does
  the cast — but the loader can be called with hand-prepared safetensors.
- **Severity**: STYLE / FLAME-CORE-ADJACENT
- **Evidence**: line 247 `.unwrap_or(v)`.

### F9: `build_cosmos_rope_freqs` accepts `enable_fps_modulation=false` + `t > 1` without `fps`, while Python errors
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:528-532`.
- **What**: The error gate is:
  ```rust
  if fps.is_none() && t != 1 {
      return Err(...);
  }
  ```
  This rejects only the `enable_fps_modulation=true` + `fps=None` +
  `t>1` shape. But Python (`minimal_v4_dit.py:770-783`) handles three
  cases distinctly:
    - `enable_fps_modulation=True && fps is None` → asserts `T == 1`.
    - `enable_fps_modulation=True && fps is not None` → modulates.
    - `enable_fps_modulation=False` → uses integer positions
      `self.seq[:T]`. No assertion on T.
- **Expected**: Two separate branches; the Rust gate currently rejects
  the *valid* `enable_fps_modulation=false && fps=None && t>1` case,
  which is the path Python takes for non-FPS-modulated video.
- **Why it matters**: V2_2B has `rope_enable_fps_modulation=true`, so
  this code path doesn't fire today. But the function is *public* and
  parameterised on `enable_fps_modulation` (`:502`). Anyone passing
  `enable_fps_modulation=false, fps=None, t=8` will hit the error,
  whereas Python would compute integer-position embeddings.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: Lines 528-532 of source vs Python `:770-783`.

### F10: `Timesteps` return shape differs from Python
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:290-315`.
- **What**: Returns `[n, num_channels]` flat. The function is described
  as a port of `Timesteps.forward`.
- **Expected**: Python `Timesteps.forward` (`minimal_v4_dit.py:864-880`)
  takes a 2D input `[B, T]` and returns `[B, T, D]` via
  `rearrange("(b t) d -> b t d")`. Rust takes a 1D `&[f32]` and returns
  rank-2.
- **Why it matters**: API divergence rather than numerical bug. The
  caller in chunk-2+ will have to rearrange. Worth flagging because
  Cosmos uses `[B, T, D]` conditioning through the whole block stack
  (the leading `B*T` flatten is undone immediately). If chunk-2 forgets
  to reshape, modulation broadcasts will be wrong.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: Function signature `fn sinusoidal_timesteps(&self, timesteps: &[f32]) -> Result<Tensor>`
  vs Python `def forward(self, timesteps_B_T)`.

### F11: `axis_lens` is dead code (never called)
- **Where**: `inference-flame/src/models/cosmos_predict25_dit.rs:198-203`.
- **What**: Defined but no caller in the file. Compounds F3 — the dead
  code carries an incorrect formula and will be wired in later.
- **Severity**: STYLE
- **Evidence**: `grep "axis_lens" cosmos_predict25_dit.rs` → 1 result
  (definition only).

## Anima oracle cross-check results

- **Timesteps math (cos/sin order, exponent, magnitude)**: **AGREES.**
  Rust `sinusoidal_timesteps` (`:303-313`) does
  `cos in first half, sin in second half`; anima `prepare_timestep`
  (`anima.rs:255-256`) does
  `emb_data[b*dim + i] = angle.cos(); emb_data[b*dim + half + i] = angle.sin()`.
  Same exponent (`-ln(10000) * i / half`). Rust uses f64 intermediates;
  anima uses f32. F64 is the more parity-faithful choice (Python uses
  float32 arange but float64 ln(10000)).

- **3D RoPE cos/sin layout (axis split, axis ordering, repeat pattern)**:
  **AGREES.** Same `dim_h = head_dim/6*2`, `dim_w = dim_h`,
  `dim_t = head_dim - 2*dim_h` (Rust `:512-514`, anima `:1055-1057`). Same
  `[t-angles, h-angles, w-angles]` row-write order (Rust `:622-647`,
  anima `:1089-1116`). Cosmos outputs F32 (correct per
  `project_bf16_rope_pattern_audit_2026-05-19`); anima casts to BF16 at
  construction (a known precision floor, listed in that audit).

- **Weight key prefix convention**: **DIFFERS.** Anima uses `net.` on
  every key (`anima.rs:1210-1214` plus all inline usages). Cosmos's
  Rust loader (`:243-253`) expects no prefix. If the upstream `.pt`
  follows anima's `net.`-wrap convention, the convert script will not
  strip it (only strips `module./ema./model./_orig_mod.`), and the Rust
  weight lookups will all fail. See F5.

## Builder skeptic-bait disposition (5 items from PORT_STATE.md)

- **Item 1 (`extra_pos_embedder` vs `pos_embedder` prefix choice)**:
  **CONFIRMED CORRECT** as far as Python source — `MiniTrainDIT.build_pos_embed`
  (`minimal_v4_dit.py:1640-1650`) constructs `pos_embedder =
  VideoRopePosition3DEmb(...)` and *only when* `extra_per_block_abs_pos_emb`
  *also* constructs `extra_pos_embedder = LearnablePosEmbAxis(...)`. So
  the learnable buffers live under `extra_pos_embedder.pos_emb_{t,h,w}`
  for V2_2B, exactly as the Rust loader expects. Builder was right to
  correct BUILD_PLAN.md.
  ⚠ Real-checkpoint verification deferred (Open issue #3 / F5).

- **Item 2 (sinusoidal exponent calculation)**: **REFUTED — agrees with
  Python.** Per-element `-ln(10000) * i / half` is the row-by-row
  equivalent of the Python vectorized form `-ln(10000) * arange(half) /
  (half - 0.0)`. Both treat `half-0.0` as float-`half`. Cross-check with
  anima `prepare_timestep` (anima.rs:253) confirms.

- **Item 3 (RoPE row-write order `[t, h, w]`)**: **REFUTED — agrees with
  Python.** Python `:785-793` builds
  `cat([repeat_t, repeat_h, repeat_w] * 2, dim=-1)` — t first, then h,
  then w; both halves identical. Rust writes
  `[t_angles, h_angles, w_angles]` into each row's `half_d` columns
  (`:622-647`). Half-split kernel then pairs `(d, d+half_d)` with the
  same angle. Cross-check with anima `:1089-1116` confirms.

- **Item 4 (NTK ratio `dim_axis as f64 / (dim_axis as f64 - 2.0)`)**:
  **REFUTED — agrees with Python.** Float division throughout
  (`:548-550`); ratios are also `f64`. Cross-check with anima
  `:1064-1066` which uses the same formula in `f64`.

- **Item 5 (convert script heuristic)**: **CONFIRMED FRAGILE** — see F5.
  The four-prefix loop is insufficient for the `net.`-wrap convention.
  Builder's own flag was honest.

## Clean checks

- BF16 RoPE pattern: cosmos outputs F32 cos/sin, defers cast to apply
  site. Matches `project_bf16_rope_pattern_audit_2026-05-19` guidance.
- No new Cargo deps added (verified via `grep` of mod.rs and the file —
  only `flame_core::*` imports).
- No env-gates introduced for default paths.
- No Flash Attention.
- No Python in the runtime path; both scripts are dev-tools only.
- `cosmos_v2_2b_config_matches_python` test reflects the values present
  in `cosmos_predict2/_src/predict2/configs/text2world/defaults/net.py:79-97`
  — verified line-by-line.
- Axis split `(44, 42, 42)` for head_dim=128 is correct.
- `patch_embed_in_channels()` returns 17 (16 + 1 padding mask) for V2_2B,
  matching Python `:1610`.
- The `extra_per_block_abs_pos_emb=true` => `extra_pos_embedder.pos_emb_*`
  weight-key prefix choice in `learnable_pos_emb` (`:396-400`) matches
  Python `build_pos_embed` (`:1640-1650`).
- The `use_adaln_lora=true` => no `linear_1.bias` key load
  (`:354-359`) matches Python `:891` `bias=not use_adaln_lora`.

## Couldn't verify

- The actual `.pt` checkpoint layout (whether keys are `net.`-wrapped) —
  requires a real checkpoint dump that the builder deferred (Open issues
  #1 and #3). F5 turns on this.
- The runtime behavior of `learnable_pos_emb`'s `broadcast_to + add`
  chain on a real device — F7. Code-only chunk; GPU smoke is step 12,
  deferred.
- Whether `flame_core::ops::fused_inference::fused_linear3d_native`
  tolerates the `[B, T, D]`-shaped sample tensor Cosmos passes in
  (instead of `[B, D]` 2D). Did not trace into flame-core for chunk-1.
- The exact dtype boundary of the convert script's `n_skipped_nontensor`
  branch: are scheduler-state, optimizer-step counters, etc. always
  scalars? Not verified against a real checkpoint.
- Whether `dump_state_dict_keys.py` correctly walks all PyTorch
  checkpoint shapes (e.g. `OrderedDict`, `LazyStateDict`). Tested only
  by reading the script; no real checkpoint available in scope.
