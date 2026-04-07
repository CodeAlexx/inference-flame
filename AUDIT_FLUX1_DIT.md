# FLUX 1 DiT Rust Port — Delta Audit vs BFL Source of Truth

**Audit target:** `/home/alex/EriDiffusion/inference-flame/src/models/flux1_dit.rs`
**Reference:** `/home/alex/black-forest-labs-flux/src/flux/{model.py,modules/layers.py,math.py}`
**Date:** 2026-04-06

Scope: pure transformer (DiT). Patchify/unpack live outside the DiT in BFL
(`sampling.py::prepare` / `unpack`) and are not part of this audit — both
BFL and the Rust port accept pre-packed `[B, N, 64]` img and emit raw
`[B, N, patch_size^2 * out_channels]` from `final_layer`.

---

## Summary

**3 CRITICAL, 4 HIGH, 4 LOW deltas found**

Blocker for numerical correctness: the RoPE implementation. The Rust port
implements **standard** complex-number rotation `(cos+i·sin)·(re+i·im)`,
but BFL's `apply_rope` implements a hand-rolled 2×2 matrix multiply whose
off-diagonal sign convention is the **conjugate** of that rotation. The
frequency packing layout on the `pe` tensor is also laid out differently
and interacts with how each pair is indexed inside Q/K.

---

## CRITICAL deltas

### C1. RoPE rotation formula has flipped off-diagonal signs

**What:** Rust uses the textbook rotation
`out_re = re·cos − im·sin`, `out_im = re·sin + im·cos`.
BFL's `apply_rope` expands to
`out[0] = cos·re + sin·im`, `out[1] = −sin·re + cos·im`
(i.e. multiplication by the conjugate phasor `cos − i·sin`). Both signs on
the off-diagonal terms are flipped between the two implementations, so
every token is rotated in the **opposite direction**. Queries and keys
end up with phases that no longer match the `pos ⇒ angle` schedule the
trained weights expect. Output will be garbage.

**Where in BFL:** `src/flux/math.py:15-30`
- `rope()` builds
  `stack([cos, -sin, sin, cos], dim=-1) ⇒ rearrange "(i j) -> i j"`
  producing `freqs_cis[...,0,:] = [cos, -sin]`,
  `freqs_cis[...,1,:] = [sin, cos]`.
- `apply_rope()` lines 26–30:
  ```
  xq_ = xq.reshape(..., -1, 1, 2)
  xq_out = freqs_cis[...,0]*xq_[...,0] + freqs_cis[...,1]*xq_[...,1]
  ```
  With `xq_[...,0]=re` and `xq_[...,1]=im`, this expands to
  `component0 = cos·re + sin·im`, `component1 = −sin·re + cos·im`.

**Where in Rust:** `src/models/flux1_dit.rs:345-372` (`rotate_complex`)
- Lines 365-366:
  ```
  let out_re = x_re.mul(cos)?.sub(&x_im.mul(sin)?)?;
  let out_im = x_re.mul(sin)?.add(&x_im.mul(cos)?)?;
  ```

**Severity:** CRITICAL — wrong sign ⇒ attention sees wrong relative
positions ⇒ incoherent outputs.

**Suggested fix:** match BFL verbatim. Keep the pair reshape
`[B,H,N,half,2]`, keep even-index as `re` and odd-index as `im`, but flip
both signs on the off-diagonal terms:

```rust
// BFL math.py apply_rope:
//   out0 =  cos*re + sin*im
//   out1 = -sin*re + cos*im
let out_re = x_re.mul(cos)?.add(&x_im.mul(sin)?)?;
let out_im = x_im.mul(cos)?.sub(&x_re.mul(sin)?)?;
```

(Be careful: `out_im` in the variable naming is really "second element
of the pair", not imaginary part of a standard phasor.)

---

### C2. `freqs_cis` is stored as `{cos, sin}` only — BFL stores the full 2×2

**What:** BFL's `rope()` returns a per-position tensor of shape
`[B, N, D/2, 2, 2]`, concatenated across RoPE axes along `dim=-3` inside
`EmbedND`, giving the final `pe` shape `[B, 1, N, sum(axes)/2, 2, 2]`
(the `1` is the head broadcast from `emb.unsqueeze(1)`).

Rust stores only `cos` and `sin` as two `[1,1,N,64]` tensors stacked into
`[2,1,1,N,64]`. This is legal if — and only if — `rotate_complex` matches
the BFL formula (see C1). As written it does not, so the storage format
is also "wrong" in the sense that it can't be directly swapped in.
Additionally, the current layout forfeits the ability to cover different
future RoPE conventions BFL may change.

**Where in BFL:**
- `src/flux/math.py:20-22` — stacked `[cos, -sin, sin, cos]` then reshaped
  to a `(2, 2)` rotation matrix.
- `src/flux/modules/layers.py:18-25` — `EmbedND.forward`: concatenate
  per-axis rope tensors along `dim=-3`, then `emb.unsqueeze(1)`.

**Where in Rust:** `src/models/flux1_dit.rs:273-325` (`build_rope_2d`)
- Lines 296-310: separately builds `cos`/`sin`, concatenates across
  axes along dim 1.
- Lines 317-324: returns `stack([cos_out, sin_out], 0)` →
  `[2, 1, 1, N, 64]`.

**Severity:** CRITICAL (together with C1). Even once the formula is
fixed, failing to match the BFL packing exactly risks subtle broadcast
or transpose bugs on any downstream change.

**Suggested fix:** materialize the same 2×2 layout BFL uses. Minimum
change that is numerically equivalent and keeps Rust-friendly storage:
build four tensors `r00=cos`, `r01=-sin`, `r10=sin`, `r11=cos`, each
`[1,1,N,half]`, stack into `[2,2,1,1,N,half]` and index
`pe[i,j]` in apply_rope, matching BFL's `freqs_cis[...,i,j]`. Or keep
cos/sin but change `rotate_complex` per C1.

---

### C3. Rust `rotate_complex` treats `x_re`/`x_im` as the first and second halves of `D` BEFORE the pair-reshape — dead code/ambiguous intent

**What:** Lines 352-353 read `x.narrow(3, 0, half)` and
`x.narrow(3, half, half)` — i.e. "first half" vs "second half" of the
last dim — but those bindings are then immediately shadowed on lines
360-361 by the correct pair-reshape (`reshape(..., half, 2)`; pick `[...,0]`
and `[...,1]`). The dead code is confusing but **not** executed.

BFL uses consecutive pairs (`.reshape(..., -1, 1, 2)`), which matches the
second (live) Rust branch. So the executed path is the correct pair
layout; the first three lines are vestigial.

**Where in BFL:** `src/flux/math.py:26-27`.

**Where in Rust:** `src/models/flux1_dit.rs:352-361`.

**Severity:** LOW on correctness (dead code), CRITICAL on maintainability
— any future edit that removes the pair-reshape and keeps the halfsplit
lines would silently introduce an incorrect RoPE.

**Suggested fix:** delete lines 352-353 (the `x.narrow(3, 0, half)` and
`x.narrow(3, half, half)` pair that gets shadowed) and the adjacent
`// Wait — FLUX uses view_as_complex ...` comment. Keep only the
`x.reshape([..,half,2])` → `narrow(4,..)` path.

> Reclassified **LOW** in the final count below because it is currently
> shadowed and doesn't affect runtime.

---

## HIGH deltas

### H1. Timestep embedding freqs table stored in BF16 instead of FP32

**What:** BFL computes `freqs` in FP32
(`torch.arange(..., dtype=torch.float32)`) and multiplies against
`t[:, None].float()`, then casts the final embedding back to `t`'s dtype
only at the end. Rust casts `freqs` to BF16 **before** the matmul, so
the entire angular argument is computed in BF16, losing the low-order
bits of the `exp(-ln 10000 · i / half)` spectrum and of the
`time_factor * t = 1000 * t` scaling.

For guidance Distillation this matters: guidance embeddings live on a
different scale than timesteps and their embeddings feed directly into
`vec` modulations.

**Where in BFL:** `src/flux/modules/layers.py:28-49` (`timestep_embedding`)
- Line 39: `dtype=torch.float32` explicitly.
- Line 43: `t[:, None].float() * freqs[None]`.
- Line 47-48: cast back to `t`'s floating dtype only at return.

**Where in Rust:** `src/models/flux1_dit.rs:232-247` (`timestep_embedding`)
- Line 240: `.to_dtype(DType::BF16)` before matmul.
- Line 243: `t_col.matmul(&freqs)?` ⇒ BF16 ⊗ BF16.

**Severity:** HIGH — visible drift, especially on guidance which passes
through the same function with values typically 1.0–10.0.

**Suggested fix:** keep `freqs` and `t_scaled` in FP32 until the
`cat([cos,sin], 1)` is formed, then cast once at the end. Something like:

```rust
let t_scaled = t.to_dtype(DType::F32)?.mul_scalar(1000.0)?;
let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half]),
                             device.clone())?; // already F32
let args = t_scaled.unsqueeze(1)?.matmul(&freqs)?; // F32
let emb = Tensor::cat(&[&args.cos()?, &args.sin()?], 1)?;
emb.to_dtype(t.dtype()) // cast once
```

---

### H2. `txt_in` bias present in BFL but uses default `nn.Linear` — verify Rust loader is not silently missing it

**What:** BFL `model.py:61` does
`self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)` —
no explicit `bias=` flag, so it defaults to `bias=True`. Rust assumes
this (loads `txt_in.bias` unconditionally in `linear_bias` at
`flux1_dit.rs:411-415`). This is correct. However, the doc comment at
line 15 asserts `txt_in.weight/bias` are present — keep an eye on
checkpoints from Schnell/FluxLoraWrapper where some state_dict filters
may drop biases.

**Where in BFL:** `src/flux/model.py:61`.
**Where in Rust:** `src/models/flux1_dit.rs:411-415`.

**Severity:** HIGH (only if a checkpoint ships without `txt_in.bias`;
the Rust loader will error out at the `Missing: txt_in.bias` branch).

**Suggested fix:** add an `Option` fallback path — if `txt_in.bias`
is absent, call `linear_nobias`. Same for all other modules known to
default to `bias=True` but that could be LoRA-wrapped.

---

### H3. `Flux1Config::load` hardcodes `Flux1Config::dev()`

**What:** Line 116: `let config = Flux1Config::dev();` then
`// auto-detect later from keys`. Schnell checkpoints do **not** have
`guidance_in.*`, so the `if let Some(..)` on lines 428-437 silently
skips guidance — good. But `has_guidance: true` still implies the caller
will pass a non-None `guidance` tensor. If they do for a Schnell model
the weights won't be there and guidance is simply ignored; if they
don't for a Dev model, the `if let Some` drops in and no error is
raised — this means Dev without passing guidance runs as Schnell,
which is a major silent mode switch.

**Where in BFL:** `src/flux/model.py:100-103` — raises `ValueError` if
`guidance_embed` is true and `guidance is None`.

**Where in Rust:** `src/models/flux1_dit.rs:116`, `:427-438`.

**Severity:** HIGH (silent wrong-mode execution on Dev).

**Suggested fix:**
1. Detect Schnell vs Dev at load time by probing
   `shared.contains_key("guidance_in.in_layer.weight")`, set
   `config.has_guidance` accordingly.
2. In `forward`, if `self.config.has_guidance && guidance.is_none()`,
   return `Err(InvalidInput("Dev requires guidance"))`.

---

### H4. `vec` is computed with `timestep_mlp` helper that skips BFL's explicit `SiLU → Linear` layout for `vector_in` on non-pooled paths

**What:** BFL's `vector_in` is an `MLPEmbedder(vec_in_dim, hidden_size)`
applied directly to CLIP pooled `y` (`model.py:104: vec = vec +
self.vector_in(y)`). `MLPEmbedder` is `Linear → SiLU → Linear`
(`layers.py:52-60`). Rust uses its `timestep_mlp` helper on line
441-448 which does `Linear → SiLU → Linear`. That part is correct.

However, the helper was written for the **timestep** path where the
input is a 2-D `[B, 256]` already. When called on `v: [B, 768]`
(CLIP pooled) it enters the `shape.len() == 2` branch — OK. But if
any future caller passes a `[B, 1, D]` input (as time embeddings are
technically `[B, dim]`), the `shape[1]` path will misinterpret the
middle dim as the feature dim. Narrow to 2-D contract via an assert
or dedicated `mlp_embedder_2d(x, ...)` function.

**Where in BFL:** `src/flux/modules/layers.py:52-60`,
`src/flux/model.py:99-104`.

**Where in Rust:** `src/models/flux1_dit.rs:250-262, 418-449`.

**Severity:** HIGH (latent footgun, not wrong today).

**Suggested fix:** rename `timestep_mlp` → `mlp_embedder`, require a
2-D `[B, D]` input, and `assert_eq!(x.shape().dims().len(), 2)`.

---

## LOW deltas

### L1. `Modulation` order of `(shift, scale, gate)` matches BFL but depends on undocumented chunk offsets

**What:** BFL `Modulation.forward` chunks `self.multiplier` ways and
wraps the first 3 as `ModulationOut(shift, scale, gate)`. Rust reads
the 6 chunks at offsets 0..6*dim with names
`(shift1, scale1, gate1, shift2, scale2, gate2)` — matches BFL.

**Where in BFL:** `src/flux/modules/layers.py:106-126`.
**Where in Rust:** `src/models/flux1_dit.rs:548-562`.
**Severity:** LOW. Correct but fragile — add a comment referencing
`ModulationOut(shift, scale, gate)` field order.

### L2. SiLU-then-linear is applied via `vec.silu()` then `linear_bias`, matching BFL

BFL: `self.lin(nn.functional.silu(vec))`. Rust: `vec.silu()?` then
`linear_bias(img_mod_w)`. Match. No action.
(Ref BFL `layers.py:121`; Rust `flux1_dit.rs:542-546`.)

### L3. `final_layer` chunk order is `(shift, scale)` and matches BFL

BFL: `shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)` —
i.e. first chunk is shift.
Rust: `shift = mods.narrow(1, 0, dim)`, `scale = mods.narrow(1, dim, dim)`
— match.
(Ref BFL `layers.py:249-253`; Rust `flux1_dit.rs:684-703`.)

### L4. Dead halfsplit branch inside `rotate_complex`

See C3. Classified as a LOW correctness concern because it is shadowed
by the subsequent correct pair reshape, but left on the books because
the comment "Wait — FLUX uses view_as_complex which interleaves..."
is misleading: BFL uses `reshape(..., -1, 1, 2)` which produces
CONSECUTIVE pairs, not interleaved `[re0, im0, re1, im1, ...]` from
complex view. The pair layout and the complex view layout happen to
coincide when `x` is already contiguous, so the comment is reinforcing
a misconception. Clean up when touching the function.

---

## Things that are correct (spot-checked)

- **Bias presence on linears:** `img_in`, `txt_in`, `time_in.{in,out}_layer`,
  `vector_in.{in,out}_layer`, `guidance_in.{in,out}_layer`, all
  `img_mod.lin` / `txt_mod.lin`, `img_attn.qkv` / `txt_attn.qkv`,
  `img_attn.proj` / `txt_attn.proj`, `img_mlp.{0,2}` / `txt_mlp.{0,2}`,
  `single_blocks.*.linear1` / `linear2` / `modulation.lin`,
  `final_layer.adaLN_modulation.1`, `final_layer.linear` — all
  loaded with `linear_bias`. Matches BFL defaults (`nn.Linear` with
  `bias=True` unless explicitly `False`, and BFL only passes
  `bias=qkv_bias` on `SelfAttention.qkv`, which is `True` per
  `FluxParams.qkv_bias=True` in the dev/schnell configs).
- **QKNorm:** RMSNorm applied to q and k separately before attention,
  inside both `double_block_forward` and `single_block_forward`.
  Matches BFL `SelfAttention.forward` lines 97-103.
- **Joint attention in DoubleStream:** concat along seq axis as
  `[txt, img]`, run SDPA once, split back — matches BFL
  `layers.py:177-182`.
- **SingleStream fused linear1 (qkv+mlp) and fused linear2
  (attn+mlp_act(mlp)):** split at offset `3*hidden` for qkv / mlp_in,
  concat at output — matches BFL `layers.py:230, 238`.
- **`time_factor=1000.0`** — matches BFL `layers.py:37`.
- **LastLayer pre-norm and modulate order:**
  `(1+scale)·LayerNorm(x) + shift → linear` — matches BFL
  `layers.py:249-253`.
- **Weight key naming:** `double_blocks.{i}.img_mod.lin.weight`,
  `double_blocks.{i}.img_attn.qkv.weight`,
  `double_blocks.{i}.img_attn.norm.{query,key}_norm.scale`,
  `single_blocks.{i}.linear1.weight`,
  `final_layer.adaLN_modulation.1.weight` — all match BFL's
  `state_dict` naming from `nn.Sequential(SiLU, Linear)` (`.1.` for the
  Linear) and `QKNorm.{query_norm,key_norm}.scale` (RMSNorm's scale
  parameter).
- **Concat order for ids:** `Tensor::cat([txt_ids, img_ids], 0)` —
  matches `model.py:107: torch.cat((txt_ids, img_ids), dim=1)` modulo
  the batch convention (BFL puts ids as `[B, N, 3]`; Rust build_rope_2d
  accepts `[N, 3]` and concats along dim 0, matching the "token axis").
- **Merge for single blocks:** `Tensor::cat([txt, img], 1)` then extract
  `img` via `narrow(1, txt_len, img_len)` at the end — matches
  `model.py:113-116`.

---

## Final count

- **CRITICAL: 3** (C1 RoPE formula, C2 RoPE layout, C3 dead halfsplit
  branch reclassified as LOW-correctness)
- **HIGH: 4** (H1 freqs precision, H2 txt_in.bias assumption, H3 Dev
  vs Schnell auto-detect, H4 timestep_mlp footgun)
- **LOW: 4** (L1 chunk order comment, L2 SiLU-then-linear confirmed,
  L3 final_layer chunk order confirmed, L4 dead rotate_complex branch)

Actionable effective count for a first fix pass: **C1 + C2 + H1 + H3**.
C1 alone is the difference between garbage and (likely) correct output.

**3 CRITICAL, 4 HIGH, 4 LOW deltas found**
