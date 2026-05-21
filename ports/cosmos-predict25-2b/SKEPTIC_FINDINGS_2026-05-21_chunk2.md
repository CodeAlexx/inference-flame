# SKEPTIC FINDINGS — cosmos-predict25-2b chunk 2 — 2026-05-21

Scope: chunk 2 (BUILD_PLAN steps 4-6): `self_attention`, `cross_attention`
(T2V + gated I2V branch), `mlp`, `transformer_block`, modulation helpers,
6 new GPU tests. Cross-checked against `anima.rs` oracle and Python
source `minimal_v4_dit.py`.

Note: this skeptic pass was line-by-line against Python `:388-624`
(Attention / I2VCrossAttention), `:237-264` (GPT2FeedForward), and
`:1130-1382` (Block), and against `anima.rs:158-511` for the working
oracle. flame-core kernel internals confirmed via `bf16_ops.rs:1380-1426`
(`modulate_pre_bf16_kernel`), `bf16_ops.rs:784-817` (`rope_halfsplit_bf16`),
`bf16_ops.rs:32-57` (`gelu_bf16_kernel` is tanh-approx),
`bf16_ops.rs:1508-1551` (`modulate_pre_fused_bf16` dispatcher),
`tensor.rs:2832-2873` (reshape materializes via `contiguous()`).

## Findings

### F1: GELU is tanh-approx; Python uses exact-erf — confirmed parity ceiling
- **Where**: `cosmos_predict25_dit.rs:1116` (`mlp` calls `h.gelu()`); flame-core
  `bf16_ops.rs:32-57` (`CUDA_GELU` constant: `0.7978845608f * (v.x + 0.044715f * v.x * v.x * v.x)` then `tanh`).
- **What**: Rust calls `Tensor::gelu()` which dispatches to `gelu_bf16` —
  the tanh-approximation form.
- **Expected**: Python `GPT2FeedForward.__init__` line 240 instantiates
  `nn.GELU()` bare — no `approximate=` parameter. PyTorch default is
  `approximate='none'` → exact-erf.
- **Why it matters**: Per-element divergence ≈ 1e-4 BF16 magnitude in
  the activation, compounding through layer2's projection. Per-block
  parity ceiling ~0.02% magnitude. Doesn't kill the model but blocks
  any parity check tighter than ~1e-3 cosine.
- **Severity**: FLAME-CORE (missing exact-erf BF16 kernel; not a
  cosmos-port bug per se. Builder correctly flagged this as a known gap
  in the docstring at lines 1100-1106.)
- **Evidence**: tanh approximation in NVRTC source `bf16_ops.rs:44-46`:
  ```
  float c0 = 0.7978845608f * (v.x + 0.044715f * v.x * v.x * v.x);
  float g0 = 0.5f * v.x * (1.0f + tanhf(c0));
  ```
  Python source `minimal_v4_dit.py:240`: `self.activation = nn.GELU()`
  (no `approximate=` arg).

### F2: `apply_gate` accepts non-5D input silently — destructure can panic
- **Where**: `cosmos_predict25_dit.rs:1210-1222` (`apply_gate`).
- **What**: The function destructures `dims[0..5]` with no rank check:
  ```rust
  let dims = x_b_t_h_w_d.shape().dims().to_vec();
  let (b, t, h, w, d) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
  ```
  If `x` is rank-3 or rank-4, the function panics with an array-out-of-bounds
  rather than returning a clean `Error::InvalidInput`. Compare with the
  guard in `apply_layer_norm_modulate` (`:1190-1196`) which DOES check
  `dims.len() != 5`.
- **Expected**: Consistent rank-check pattern across both helpers; both
  should return `Err(Error::InvalidInput)` on shape mismatch.
- **Why it matters**: Today the only caller is `transformer_block`,
  which always passes rank-5. Refactor-resistant only. But this is the
  same class of foot-gun as chunk-1 F8 (`unwrap_or` swallowing dtype
  errors): one missing guard surface area.
- **Severity**: STYLE
- **Evidence**: `cosmos_predict25_dit.rs:1215` direct `dims[0..5]` access
  vs `:1191-1196` explicit check in the sibling helper.

### F3: `apply_gate` `broadcast_to + mul` exercises the strided fallback path — F7 chunk-1 hazard still live
- **Where**: `cosmos_predict25_dit.rs:1218-1221`.
- **What**:
  ```rust
  let gate_b_t_1_1_d = gate_b_t_d.reshape(&[b, t, 1, 1, d])?;
  let gate_full = gate_b_t_1_1_d.broadcast_to(&target)?;  // stride-0 view on (H,W)
  x_b_t_h_w_d.mul(&gate_full)
  ```
  `broadcast_to` creates a view with stride 0 on the H and W axes. The
  resulting `gate_full.is_contiguous()` returns false, so `Tensor::mul`
  (`tensor.rs:2222-2224`) bypasses the BF16 contig fast path
  (`mul_bf16_contig_direct`) and falls into `dispatch_binary_bf16` /
  `GpuOps::mul`. The math is correct but slower; more importantly, the
  chunk-1 F7 finding "broadcast+mul stride-0 chain unverified" still
  applies and was not addressed.
- **Expected**: Either (a) materialize the broadcast (`broadcast_to` →
  `contiguous`) to hit the fast path, or (b) confirm the strided
  TensorIterator dispatch is correctness-clean on stride-0 inputs (the
  builder's note ATIVA defers this to GPU smoke). The 6 GPU tests do
  exercise this path but only check finiteness and load-bearingness, not
  correctness against a reference.
- **Why it matters**: If the strided BF16 mul path has any edge case
  bug with stride-0 inputs (e.g. wrong batch index for repeated reads),
  the gate would silently zero or duplicate. Tests would still pass
  finiteness. This is the kind of bug only per-layer parity catches.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: `tensor.rs:2219-2241` fast-path is gated on `self.is_contiguous() && other.is_contiguous()`; broadcasted gate fails the second condition.

### F4: `cross_attention` allows BUT IGNORES `image_context` when `extra_image_context_dim` is `None` — silent semantics mismatch with Python
- **Where**: `cosmos_predict25_dit.rs:1046-1081`.
- **What**: When `config.extra_image_context_dim.is_none()` and the
  caller passes `Some(img_ctx)`, the function logs at debug level and
  returns the T2V-only result.
- **Expected**: Python doesn't have this softening. In Python, an
  `Attention` (not `I2VCrossAttention`) is built when
  `extra_image_context_dim is None` (`minimal_v4_dit.py:1181-1184`);
  there's no `image_context` parameter on `Attention.forward`. If a
  caller tried to pass one, it would TypeError. Cosmos Rust accepts it
  silently, which is the opposite of "fail-fast" hygiene the project
  prefers (per CLAUDE.md / EMPOWERMENT). The test
  `cross_attention_t2v_runs_and_ignores_image_context` actively asserts
  this lenient behavior, so the design choice is intentional but
  divergent.
- **Why it matters**: A future i2v variant call-site that erroneously
  builds a T2V config but passes an image_context will silently produce
  a wrong-but-runs output, with only a `log::debug!` trail.
- **Severity**: DISAGREE (intentional design; flagging because it
  diverges from Python and from the project's general fail-fast posture.
  Not a parity bug — purely API hygiene.)
- **Evidence**: `cosmos_predict25_dit.rs:1074-1080` skips silently;
  Python `:1181-1184` would TypeError on extra positional arg.

### F5: I2V dual-K/V branch reuses the cross-attn `q` AFTER it has been `rms_norm`-ed AND `permute`-d — matches Python; BUT documentation is wrong about norm order
- **Where**: `cosmos_predict25_dit.rs:1031-1037, 1070`.
- **What**: Cosmos cross-attn does `rms_norm_per_head_bnhd(q, ...)` on
  the BNHD-format `q` (line 1031), THEN permutes to BHND (line 1035).
  For the I2V branch the same already-normed-and-permuted `q` is reused
  for the second SDPA (line 1070). This is mathematically correct (`q`
  is shared across both K/V branches, normalized once).
- **Expected**: Python `I2VCrossAttention.compute_qkv` returns
  `q, k, v, k_img_norm(k_img), v_img` (line 609) — `q` is computed once
  via `super().compute_qkv` and never touched again. ✓
- **Why it matters**: Not a bug; flagged because the chunk-2 skeptic-
  bait listed this as "RMSNorm-then-permute order vs self-attn's
  permute-then-RMSNorm". Both orders are mathematically equivalent
  (norm is along last dim, permute doesn't change last dim), and the
  shared `q` reuse path is correct. Confirmed.
- **Severity**: DISAGREE (this was the builder's own skeptic-bait; I'm
  refuting it as not-a-bug.)
- **Evidence**: math + `bf16_ops.rs:784-817` (RMSNorm kernel reads last
  dim; layout-agnostic over outer dims).

### F6: `head_dim=6` test config drives RoPE axis split `(2, 2, 2)` — passes mod-6 check but exercises an UNTESTED edge of `build_cosmos_rope_freqs`
- **Where**: `cosmos_predict25_dit.rs:1603` (`tiny_test_config()` sets
  `model_channels=12, num_heads=2 → head_dim=6`).
- **What**: With `head_dim=6`: `dim_h = 6/6*2 = 2`, `dim_w = 2`, `dim_t = 2`.
  Half-rotated: `half_t=1, half_h=1, half_w=1`. Per-axis the RoPE only
  produces 1 frequency bin each. This corner case has never been
  exercised against the V2_2B real `head_dim=128` (where each axis has
  21 or 22 bins). The half-split kernel SHOULD handle `half=3` (= total
  half_d = head_dim/2 = 3), and dim_t/dim_h/dim_w=2 still satisfy "must
  be even".
- **Expected**: Same kernel semantics regardless of head_dim, but small
  head dims are a different numerical regime (no warp reductions; few
  channels per token).
- **Why it matters**: The GPU tests verify "runs end-to-end and is
  finite", not "produces the correct numerical answer". A bug specific
  to `half_d=3` (such as `LaunchConfig::for_num_elems(total)` choosing
  block_dim that's strange for tiny N) wouldn't be caught by finite-
  ness alone. Parity testing at the real head_dim=128 is needed.
- **Severity**: STYLE (mitigated at parity time)
- **Evidence**: `cosmos_predict25_dit.rs:1614-1615` (heads=2, channels=12)
  vs V2_2B `:142-143` (heads=16, channels=2048).

### F7: I2V `cross_attention_i2v_changes_output` test only asserts `max_diff > 1e-4` — passes even if image-branch output gets near-zeroed
- **Where**: `cosmos_predict25_dit.rs:1871`.
- **What**:
  ```rust
  assert!(max_diff > 1e-4,
      "cross_attention i2v branch is a no-op: max_diff={max_diff}");
  ```
  A `1e-4` threshold against BF16 outputs is barely above noise. If the
  image branch were partially broken (wrong head-permute, wrong
  k_img_norm weight, broadcast wrong axis) and contributed e.g. only
  `1e-3` magnitude, this test passes. The complementary check would be
  "output of i2v with all-zero img_ctx data EXACTLY equals text-only
  output", which would be a stronger isolation but is not asserted.
- **Expected**: At minimum, also assert that text-only output bytes
  exactly equal i2v(img_ctx=zeros), to prove the image branch's
  inactive case is bit-identical to the no-branch path.
- **Why it matters**: The 6 chunk-2 tests are explicitly load-bearing
  per the chunk-1 F4 lesson. This particular threshold is generous; a
  half-broken image branch would slip through.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: `cosmos_predict25_dit.rs:1844-1872`.

### F8: `transformer_block_modulation_is_load_bearing` test sets `lora=zero` and uses ONLY emb-driven diff — does NOT prove `adaln_lora_b_t_3d` is wired
- **Where**: `cosmos_predict25_dit.rs:1909-1981`.
- **What**: The test passes `lora_zero` in both runs and only flips
  `emb_zero` vs `emb_nonzero`. This proves the SiLU+Linear+Linear chain
  is connected. It does NOT prove the `+ adaln_lora_b_t_3d` add (line
  1166) is wired correctly — if that line were `let summed = h2.clone();`
  (forgetting the add), the test still passes because lora=0 anyway.
- **Expected**: Companion test: pin emb to a fixed value, vary
  `adaln_lora_b_t_3d` between two non-zero choices, assert outputs
  differ. That isolates the lora-add path which is the V2_2B-specific
  feature (`use_adaln_lora=true`).
- **Why it matters**: Skipping the `+ lora` is exactly the kind of bug
  load-bearing tests are meant to catch. The current test set has a
  blind spot for it.
- **Severity**: CORRECT-BUT-FRAGILE
- **Evidence**: tests at `:1939-1965`.

### F9: `cross_attention_t2v_runs_and_ignores_image_context` test contains a redundant assertion AFTER `assert!(max_diff == 0.0)`
- **Where**: `cosmos_predict25_dit.rs:1815-1817`.
- **What**: `assert!(max_diff == 0.0)` is an exact equality check between
  the two `f32` Vec contents. This is fine because both runs use the
  same input and skip the image branch — they should be byte-identical.
  But it's brittle: any non-determinism (e.g. cuBLASLt heuristic search
  picking different algos for the two calls) would fail it. The
  alternative `(max_diff - 0.0).abs() < ulp` is more robust.
- **Expected**: `assert_eq!(a, bv)` or a tight epsilon tolerance.
- **Why it matters**: cuBLASLt occasionally picks different algorithms
  on the second invocation of an identical GEMM (workspace re-use). This
  would surface as flaky test failure, not a real bug.
- **Severity**: STYLE
- **Evidence**: `:1815`.

### F10: `linear_no_bias` ALWAYS collapses to `[1, n, cin]` — fine numerically but eliminates the `B>1` cuBLASLt heuristic
- **Where**: `cosmos_predict25_dit.rs:792-810`.
- **What**: Cosmos `linear_no_bias` does
  ```rust
  let n: usize = in_dims[..in_dims.len() - 1].iter().product();
  let x_3d = x.reshape(&[1, n, cin])?;  // <-- always B=1
  fused_linear3d_native(&x_3d, weight, None)
  ```
  vs anima `linear_no_bias` (`anima.rs:165-168`):
  ```rust
  let x_2d = x.reshape(&[batch, in_features])?;
  let wt = weight.permute(&[1, 0])?;     // explicit transpose
  let out_2d = x_2d.matmul(&wt)?;
  ```
- **Expected**: Both paths produce mathematically identical results.
  `fused_linear3d_native` (cuBLASLt with TRANSA=T) is faster than
  anima's `permute+matmul`. The B=1 collapse is a clever way to use
  the 3D fused API; the cuBLASLt heuristic sees `M = batch_size * seq_len`
  internally regardless of how the caller carves `B` vs `N`, so this is
  not a perf bug.
- **Why it matters**: Confirmed safe. Builder skeptic-bait item 1 is
  refuted.
- **Severity**: DISAGREE (the bait flagged this as potential bug; I
  refute — it's correctness-equivalent and performance-equivalent.)
- **Evidence**: `fused_inference.rs:368-376` — kernel does
  `M = batch_size * seq_len`, so B-vs-N split is irrelevant.

### F11: `extra_per_block_pos_emb` add uses `x.add(pe)` with no broadcast-shape check — passes 5D shape match silently
- **Where**: `cosmos_predict25_dit.rs:1260-1264`.
- **What**: `x_b_t_h_w_d.add(pe)` requires same-shape (or broadcast-able)
  inputs. The function does not check that `pe.shape() == x.shape()`.
  If the caller (chunk 7) computes pe at a wrong (T, H, W) and passes
  it in, `Tensor::add` may either error (good), broadcast wrongly (bad),
  or panic.
- **Expected**: Defensive shape-check on `pe.shape() == x.shape()`
  before the add, with a clear error message naming the block_idx.
- **Why it matters**: Builder reports this is implemented in
  `transformer_block`. The chunk-7 caller is the one that prepares
  `pe` once at the top and reuses across all 28 blocks. A single
  miscomputed `pe` would silently corrupt all 28 blocks.
- **Severity**: STYLE
- **Evidence**: `:1260-1264` — no rank/shape guard.

### F12: `adaln_modulation_chunk` allocates THREE `narrow` views into `summed` and `summed` itself goes out of scope at function exit
- **Where**: `cosmos_predict25_dit.rs:1166-1173`.
- **What**:
  ```rust
  let summed = h2.add(adaln_lora_b_t_3d)?;
  let shift = summed.narrow(2, 0, d)?;
  let scale = summed.narrow(2, d, d)?;
  let gate  = summed.narrow(2, 2 * d, d)?;
  Ok((shift, scale, gate))
  ```
  Per project memory `narrow_owning` was specifically introduced to
  avoid view-pinning the parent storage for long-lived narrows. Here,
  `summed` will go out of scope at the end of the function, but the
  three views returned share its storage via Arc. The 3 views keep
  `summed`'s storage alive — that's fine — but the `.contiguous()`
  materialization that subsequently runs inside `reshape(&[b*t, d])`
  for shift/scale (called in `apply_layer_norm_modulate:1199-1201`)
  and inside `gate.reshape(&[b, t, 1, 1, d])` (called in `apply_gate`)
  forces a full BF16 copy per narrow. So 3 narrow + 3 materialize per
  block per sub-block. For 28 blocks × 3 sub-blocks = 84
  modulation chunks per forward pass = 252 unnecessary BF16 copies.
- **Expected**: Two options: (a) explicit `narrow_owning` to materialize
  once and reuse, or (b) `chunk(3, dim=-1)` style that returns 3 owned
  tensors. Per the chunk-1 CONTEXT.md "narrow view-pinning" trap, the
  current path is the SLOWER trap variant, not the leak variant.
- **Why it matters**: Performance, not correctness. ~252 small BF16
  copies per step is not catastrophic but it's the kind of paper cut
  flame-core's perf audit explicitly calls out.
- **Severity**: STYLE (defer to perf phase; no parity impact)
- **Evidence**: `:1166-1173`; memory note
  `[[project_active_handoff]]` re narrow_owning.

### F13: `cross_attention` test stub uses `k_img.weight` shape `[inner, d]` not `[inner, img_latent_dim]` — test config makes them equal but glosses over the distinction
- **Where**: `cosmos_predict25_dit.rs:1705-1709`.
- **What**: Test stub:
  ```rust
  if cfg.extra_image_context_dim.is_some() {
      weights.insert(format!("{pfx}.k_img.weight"), mk_weight(dev, ..., &[inner, d]));
  ```
  Here `d = model_channels = 12`. The test sets
  `extra_image_context_dim: Some(12)` which causes Python's
  `MiniTrainDIT` line 1539 to pass `image_context_dim=model_channels=12`
  to `Block`, which then passes `img_latent_dim=12` to
  `I2VCrossAttention`. So `k_img.weight` in Python has shape
  `[inner_dim=12, img_latent_dim=12]`. With `model_channels=12`
  coincidentally equal to `extra_image_context_dim=12`, the test
  cannot distinguish the case where the port hard-codes `d` instead of
  reading `img_latent_dim` from config.
- **Expected**: A more discriminating test would use
  `extra_image_context_dim: Some(20)` (≠ model_channels=12), then
  expect the caller to project from 20→12 via `img_context_proj`
  externally and pass the projected `[B, S, 12]` tensor — but then
  `k_img.weight` is `[12, 12]` again (since the block expects already-
  projected input per the chunk-2 spec correction item 5). So in fact
  the chosen test is correct given that the input to `cross_attention`
  is already-projected. Net: no bug, but the test stub's `&[inner, d]`
  shape is correct because of the OUTSIDE projection — confirm the
  builder is aware of this and chunk-7 will perform the
  `img_context_proj` step.
- **Severity**: DISAGREE (refuting my own initial concern; the test is
  consistent with the chunk-2 spec correction item 5.)
- **Evidence**: `minimal_v4_dit.py:1539`, `:1557-1563`, `:1745`.

## Anima oracle cross-check results

| Pattern | Cosmos chunk-2 | Anima | Status |
|---|---|---|---|
| `linear_no_bias` reshape | `reshape([1, n, cin])` + `fused_linear3d_native` | `reshape([batch, cin])` + `permute(weight, [1,0])` + `matmul` | DIFFERS but functionally identical; cosmos path is faster (`fused_linear3d_native` uses cuBLASLt TRANSA=T). |
| `mlp` activation | `Tensor::gelu()` (tanh-approx) | `Tensor::gelu()` (tanh-approx) | AGREES (both use same kernel). Both diverge from Python `nn.GELU()` (exact-erf). |
| `transformer_block` residual structure | (1) pos_emb add, (2) `x = x + gate*attn`, (3) `x = gate*cross + x`, (4) `x = x + gate*mlp` | Same residual order, no pos_emb (Anima is image-only) | AGREES on the shared portions. Cosmos adds pos_emb inside block per Python `:1267-1268`. |
| `transformer_block` modulation precision | BF16 throughout (matches `use_wan_fp32_strategy=False` Python path) | F32 residual stream (Anima specifically casts BF16↔F32 to handle large values ~200+) | DIFFERS. Anima learned the hard way that BF16 residual loses precision on Cosmos-class models. Cosmos chunk-2 stays in BF16. This may surface as a parity gap during the parity phase; flagging for awareness. |
| `rms_norm_per_head_bnhd` / `_bhnd` | `flame_core::norm::rms_norm` over last dim | `cuda_ops_bf16::rms_norm_bf16` over last dim | Same math; different code path. Both eps=1e-6. |
| Cross-attn K/V branch | T2V single K/V + optional I2V dual-K/V (gated) | T2V single K/V only | DIFFERS as expected (Anima is Cosmos Predict2 image-only, no I2V variant). Cosmos chunk-2 extends correctly. |

**Notable divergence from Anima oracle**: Cosmos stays in BF16 for the
transformer_block residual stream; Anima explicitly casts to F32 between
sub-blocks because empirically Cosmos hidden values reach ~200+
magnitude and BF16's 8-bit mantissa loses precision rapidly. The
Cosmos chunk-2 builder accepts this risk silently. This may show up as
a parity gap on the real V2_2B once chunk 7 wires up the full forward
pass with real weight magnitudes. Worth a deliberate parity check on
"hidden value magnitude after 5 blocks" before committing to BF16
through-stream.

## Builder spec-correction validation (6 items)

- **Item 1 (GELU exact-erf in Python)**: CONFIRMED. `minimal_v4_dit.py:240`
  has bare `nn.GELU()` — no `approximate=` argument. PyTorch default is
  `approximate='none'` → exact-erf.
- **Item 2 (V2_2B has NO I2V dual-K/V)**: CONFIRMED.
  `configs/text2world/defaults/net.py:79-97` (`COSMOS_V2_2B_NET`) does
  NOT set `extra_image_context_dim`, so it defaults to `None` in
  `MiniTrainDIT.__init__` (line 1466). Block-build logic at `:1539`
  passes `image_context_dim=None`, so `Block.__init__` line 1181-1184
  builds plain `Attention`, not `I2VCrossAttention`.
- **Item 3 (`extra_per_block_pos_emb` added inside Block.forward)**:
  CONFIRMED. `minimal_v4_dit.py:1267-1268`:
  ```python
  if extra_per_block_pos_emb is not None:
      x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
  ```
  Add happens before the with-autocast block (line 1270), so it lives
  in BF16 for V2_2B's `use_wan_fp32_strategy=False` path.
- **Item 4 (weight names `q_proj`, `k_proj`, `v_proj`, `output_proj`,
  `layer1`, `layer2`)**: CONFIRMED. Lines `:453, 456, 459, 462` for
  `q_proj/k_proj/v_proj/output_proj`; lines `:241-242` for
  `layer1/layer2`.
- **Item 5 (`k_img` input dim is `model_channels`=2048, not 1024)**:
  CONFIRMED. `:1539` passes `image_context_dim=model_channels` to
  Block, which forwards as `img_latent_dim=image_context_dim` to
  `I2VCrossAttention.__init__` (line 1191). Then `:586` builds
  `nn.Linear(img_latent_dim, inner_dim, bias=False)` — so the input dim
  to `k_img` is `model_channels=2048`. Cosmos chunk-2 correctly
  validates this in the test stub at `:1706` with shape
  `[inner, d=model_channels]`.
- **Item 6 (adaLN chunk order `(shift, scale, gate)` × 3 sub-blocks
  with `+ adaln_lora` shared across all three)**: CONFIRMED. Python
  `:1272-1280`:
  ```python
  shift_self_attn, scale_self_attn, gate_self_attn = (
      self.adaln_modulation_self_attn(emb) + adaln_lora_B_T_3D
  ).chunk(3, dim=-1)
  shift_cross_attn, scale_cross_attn, gate_cross_attn = (
      self.adaln_modulation_cross_attn(emb) + adaln_lora_B_T_3D
  ).chunk(3, dim=-1)
  shift_mlp, scale_mlp, gate_mlp = (
      self.adaln_modulation_mlp(emb) + adaln_lora_B_T_3D
  ).chunk(3, dim=-1)
  ```
  Cosmos `adaln_modulation_chunk` (`:1149-1174`) is called per
  sub-block with the same `adaln_lora_b_t_3d` argument; narrows
  produce `(shift, scale, gate)` in that order. ✓

## Builder skeptic-bait disposition (6 items)

- **Item 1 (B=1 reshape in `linear_no_bias`)**: REFUTED. See F10. The
  `fused_linear3d_native` kernel internally computes
  `M = batch_size * seq_len`; whether the caller splits work as
  `(1, n)` or `(B, N)` is irrelevant to the cuBLASLt M-dim, so
  performance and correctness are equivalent to anima's
  `permute+matmul` path. Confirmed via `fused_inference.rs:368-376`.
- **Item 2 (RMSNorm-then-permute vs permute-then-RMSNorm)**: REFUTED.
  See F5. Both orders are mathematically identical (RMSNorm operates
  on the last dim; permute doesn't reorder the last dim). Both code
  paths reach the same kernel.
- **Item 3 (`apply_gate` broadcast_to + mul stride-0 view)**:
  CONFIRMED-AS-HAZARD-NOT-BUG. See F3. The path is correct under the
  current `Tensor::mul` strided fallback, but exercises the slow
  TensorIterator dispatch. No GPU test currently asserts correctness
  against a reference; only finiteness. The chunk-1 F7 hazard is
  inherited here intact.
- **Item 4 (`modulate_pre_fused_bf16` batch indexing)**: REFUTED.
  Kernel source at `bf16_ops.rs:1380-1426`:
  ```cuda
  int batch_idx = row / seq_len;
  scale_row = SCALE + batch_idx * dim;
  shift_row = SHIFT + batch_idx * dim;
  ```
  with `seq_len = n` from `x.shape().dims()[1]`. Cosmos passes
  `x=[B*T, H*W, D]`, so `n = H*W` and `batch_idx = row / (H*W)` which
  maps each row to its (B*T) slot. The shift/scale come in as
  `[B*T, D]` (Cosmos `:1200-1201`), so `batch_idx * dim` correctly
  indexes the right modulation tuple. Math is sound.
- **Item 5 (`unflatten_thw` row order assumption)**: CONFIRMED-SAFE.
  RoPE cos/sin in `build_cosmos_rope_freqs` are written in row-major
  `t*H*W + h*W + w` (cosmos_predict25_dit.rs:723). The SDPA path
  takes `x_seq` = `flatten_thw(x)` which reshapes `[B, T, H, W, D]`
  → `[B, T*H*W, D]` collapsing in the same row-major order. After
  SDPA, `unflatten_thw` reshapes back with the same row layout. So
  cos/sin row k matches token k. ✓
- **Item 6 (`output_proj` weight key name in cross-attn)**: CONFIRMED.
  Both self-attn and cross-attn (via `I2VCrossAttention.__init__`
  inheriting from `Attention`) have `self.output_proj = nn.Linear(...)`
  at `:462`. HF checkpoint will name it `blocks.{i}.cross_attn.output_proj.weight`.

## Clean checks

- **RoPE applied to Q and K only, not V**: confirmed at
  `cosmos_predict25_dit.rs:951-952`; V is never touched after permute.
- **RoPE NOT applied in cross-attention**: confirmed by absence of
  `rope_halfsplit_bf16` calls in `cross_attention` body.
- **RoPE applied AFTER Q/K rms_norm**: confirmed at `:935-952` — norm
  first, then rope.
- **RoPE uses `rope_halfsplit_bf16` (NOT `rope_fused_bf16`)**:
  confirmed at `:951-952`.
- **Q/K RMSNorm weight shape `[head_dim]`, eps=1e-6**: confirmed at
  `:826, 842` (`flame_core::norm::rms_norm(&flat, &[d], Some(weight),
  1e-6)`).
- **Pre-sub-block norm is LayerNorm (not RMSNorm)**: confirmed —
  `modulate_pre_fused_bf16` kernel at `bf16_ops.rs:1390-1416` computes
  `mean`, `variance`, `(x - mean) * inv_std` — that's LayerNorm
  (matches Python `nn.LayerNorm(elementwise_affine=False, eps=1e-6)`).
- **Modulation formula `(1+scale)*x + shift`**: confirmed at
  `bf16_ops.rs:1423`: `(1.0f + sc) * normed + sh`.
- **No `Tensor::cat` in chunk 2**: grep clean — chunk 2 uses two SDPA
  calls + `add`, no cat (per the chunk-2 builder).
- **No new Cargo dependencies**: confirmed — imports are
  `flame_core::*`, `std::collections::HashMap`, `std::path::Path`,
  `std::sync::Arc` (lines 45-49). No new deps.
- **No `.contiguous()` workarounds**: grep clean — chunk 2 has zero
  `.contiguous()` calls. Materialization happens via `reshape` (which
  internally calls `contiguous()` for non-contig inputs, but this is
  the documented fallback, not a workaround).
- **No env-gates for default-path code**: confirmed.
- **No `unwrap_or(default)` swallowing errors in chunk-2 code**: chunk-2
  code uses `?` and explicit `ok_or_else(...)` (e.g. `:155`).
- **Tests exercise real production functions (F4 lesson)**: confirmed —
  all 6 chunk-2 tests call `model.self_attention`, `model.cross_attention`,
  `model.mlp`, `model.transformer_block` directly.
- **I2V test load-bearing assertion exists**:
  `cross_attention_i2v_changes_output:1871` asserts `max_diff > 1e-4`.
  Threshold weak but assertion present (F7).
- **Modulation test load-bearing assertion exists**:
  `transformer_block_modulation_is_load_bearing:1973-1975` asserts
  `max_diff > 1e-4`. Weakness noted in F8 (doesn't isolate `+ lora`).
- **pos_emb test load-bearing assertion exists**:
  `transformer_block_uses_extra_per_block_pos_emb:2037` asserts
  `max_diff > 1e-4`.
- **Chunk order `(shift, scale, gate)`**: confirmed at `:1170-1172`.
- **adaLN modulation Sequential is `SiLU → Linear → Linear`**: SiLU
  applied at `:1158` BEFORE Linear; Linear[0] = SiLU, Linear[1] = Linear,
  Linear[2] = Linear — keys `.1.weight` and `.2.weight` are read at
  `:1160, 1162` matching `nn.Sequential(SiLU, Linear, Linear)` index
  layout.
- **adaLN lora added BEFORE chunk(3)**: confirmed at `:1166-1170`.
- **`v_norm` is Identity (no-op, not called)**: confirmed; cosmos
  doesn't read a `v_norm.weight` key anywhere.
- **`k_img_norm` IS applied (Python `:609`)**: confirmed at
  `cosmos_predict25_dit.rs:1065`.
- **Cross-attn `result + result_img` happens BEFORE `output_proj`**:
  confirmed at `:1072` (sum) then `:1086` (`output_proj`).
- **No Flash Attention proposals**: grep clean.
- **No Python in runtime path**: grep clean (Python only in
  `parity/*.py` scripts, which are dev tools).

## Couldn't verify

- **Whether the BF16-throughout residual stream causes magnitude blow-up
  on real V2_2B weights**: Anima documents this concern at
  `anima.rs:470` ("FP32 residual stream (model has large values ~200+)").
  Without running the real forward on the real checkpoint and
  inspecting hidden magnitudes after 5+ blocks, I can't quantify the
  parity gap this will create. Defer to parity phase. The skeptic
  recommends a magnitude-tracking probe (`assert |x| < 100` after each
  block) before declaring chunk 2 "smoke-clean" in step 12.
- **Whether `modulate_pre_fused_bf16` correctly handles
  `n = H*W = 1` corner case**: the chunk-2 tests use H=W=2 (n=4). The
  kernel uses `block_dim.x` reduction over `dim` (not `n`); per-row
  computation should be N-agnostic. Not exercised.
- **Whether the strided `Tensor::mul` BF16 dispatch is bit-correct on
  stride-0 broadcast inputs of arbitrary rank-5 shape**: chunk-2 tests
  only assert finiteness and load-bearingness, not numerical
  correctness against a reference. Defer to parity phase.
- **Whether `Tensor::reshape` over a non-contig narrow view's
  `contiguous()` materialization introduces dtype-coercion artifacts**:
  no evidence of issue, but the chain narrow→reshape→reshape→add inside
  `adaln_modulation_chunk` + `apply_layer_norm_modulate` is
  longer than usual. Spot-check on parity.

## Summary count

- BLOCKERs: **0**
- CORRECT-BUT-FRAGILE: **3** (F3, F7, F8)
- STYLE: **5** (F2, F6, F9, F11, F12)
- FLAME-CORE: **1** (F1)
- DISAGREE / refutation: **3** (F4, F5, F10, F13 — all of the builder's
  own skeptic-bait items 1, 2, 6, plus F4 which is intentional API
  divergence)

**Chunk 2 is structurally sound.** The Python reference walk found no
correctness blockers. The 6 builder spec corrections all check out
against Python source. The 6 builder skeptic-bait items are 4 refuted
(safe), 1 confirmed as a hazard-not-bug (F3 = chunk-1 F7 inherited),
and 1 confirmed safe by inspection.

The five real concerns going into parity:
1. F1 — GELU tanh-approx parity ceiling ~0.02% per block.
2. F3 — stride-0 broadcast on the gate multiply is correct but
   exercises the slow TensorIterator path.
3. F7 — I2V test threshold `1e-4` is generous; tighten before
   declaring i2v done.
4. F8 — `+ adaln_lora_b_t_3d` add path is NOT independently load-
   bearing tested.
5. Anima-divergence — Cosmos stays BF16 throughout block residual;
   Anima learned to escalate to F32. Magnitude check at parity phase.
