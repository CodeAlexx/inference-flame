# FLUX 1 Text Encoder Audit — CLIP-L + T5-XXL

Delta report comparing our Rust ports against BFL FLUX source-of-truth and HuggingFace
transformers reference implementations.

**Sources of truth**
- `/home/alex/black-forest-labs-flux/src/flux/modules/conditioner.py` (BFL HFEmbedder)
- `transformers/models/t5/modeling_t5.py` (T5LayerNorm, T5Attention, T5DenseGatedActDense, T5Block)
- `transformers/models/t5/configuration_t5.py` (T5 v1.1 = `gated-gelu` → `dense_act_fn="gelu_new"`)
- `transformers/models/clip/modeling_clip.py` (CLIPTextTransformer, CLIPAttention, CLIPMLP)
- `transformers/models/clip/configuration_clip.py` (default `hidden_act="quick_gelu"`, `eos_token_id=49407`)
- `transformers/modeling_attn_mask_utils.py::_make_causal_mask` (additive `{0,-inf}` causal)

**Rust ports audited**
- `inference-flame/src/models/clip_encoder.rs`
- `inference-flame/src/models/t5_encoder.rs`
- Supporting: `flame-core/src/sdpa.rs` (`forward_f32`, `forward_bf16_fallback`)

---

## Summary

**5 CRITICAL, 4 HIGH, 3 LOW deltas found across CLIP and T5**

Of those, two CRITICALs (T5.C1 position-bias-as-mask, T5.C2 missing attention scaling) are almost
certainly the reason T5 will produce garbage embeddings and FLUX 1 output will be unusable. Both
must be fixed before any integration test.

---

# T5-XXL deltas

## T5.C1 — CRITICAL — Relative position bias passed as binary mask to `flame_sdpa`

**Source of truth** (`modeling_t5.py` lines 529, 558, 561):
```python
scores = torch.matmul(query_states, key_states.transpose(3, 2))   # [B,H,Q,K] float
...
scores += position_bias_masked                                    # additive float bias
attn_weights = nn.functional.softmax(scores.float(), dim=-1)
```
`position_bias` is a dense float tensor `[1, n_heads, seq, seq]` **added** to raw logits.
Values are learned embeddings (can be positive or negative, magnitude O(1)).

**Rust port** (`t5_encoder.rs` line 250):
```rust
// Attention with position bias added to scores
// flame_sdpa adds mask to scores; position_bias acts as the "mask" here
let attn_out = flame_sdpa(&q, &k, &v, Some(position_bias))?;
```
This is catastrophically wrong because `flame_sdpa` treats the `mask` argument as a
**binary keep-mask**. From `flame-core/src/sdpa.rs::forward_f32` lines 184–195:
```rust
let mask_f32 = expanded.to_dtype(DType::F32)?;
let ones = full_like(&mask_f32, 1.0)?;
let complement = ones.sub(&mask_f32)?;
let penalty = complement.mul_scalar(NEG_INF)?;
scores = scores.add(&penalty)?;
```
i.e. every value in `position_bias` is interpreted as `keep=value`, and flame applies
`(1 - value) * -inf`. A learned bias of e.g. `0.37` becomes `0.63 * -∞ = -∞`.
Identical logic in `forward_bf16_fallback` (lines 397–412). Result: **the entire T5
encoder produces softmax over an all `-inf` row per position → NaN or a single hot
value**. T5 output is meaningless. FLUX 1 prompts will not follow.

**Severity**: CRITICAL — blocks entire FLUX 1 text conditioning.

**Verbatim fix** (pick one path):

*Option A — introduce `flame_sdpa_with_bias` that skips the binary conversion and adds
the bias directly to `scores` before softmax:*
```rust
// flame-core/src/sdpa.rs — new public fn
pub fn forward_with_bias(
    q: &Tensor, k: &Tensor, v: &Tensor,
    additive_bias: Option<&Tensor>,  // [*, H|1, Q, K], added to scores before softmax
) -> SdpaResult<Tensor> { ... }
```
Inside `forward_f32`:
```rust
if let Some(bias) = additive_bias {
    let b32 = bias.to_dtype(DType::F32)?;
    scores = scores.add(&b32.broadcast_to(&Shape::from_dims(&target_dims))?)?;
}
```
Then in `t5_encoder.rs::layer_forward` call
`flame_sdpa_with_bias(&q, &k, &v, Some(position_bias))`.

*Option B — manual attention path inside `t5_encoder.rs` (no flame_sdpa):*
```rust
// scores = Q @ K^T  (NO 1/sqrt(d_kv) scaling — T5 uses unscaled attention)
let q32 = q.to_dtype(DType::F32)?;
let k32 = k.to_dtype(DType::F32)?;
let v32 = v.to_dtype(DType::F32)?;
let k_t = k32.transpose(-2, -1)?;                          // [B,H,d,K]
let mut scores = q32.matmul(&k_t)?;                        // [B,H,Q,K]
scores = scores.add(&position_bias.to_dtype(DType::F32)?)?;  // additive bias
let attn = scores.softmax(-1)?;
let attn_out_f32 = attn.matmul(&v32)?;
let attn_out = attn_out_f32.to_dtype(DType::BF16)?;
```

Option B is strictly required because of T5.C2 below (no scaling).

---

## T5.C2 — CRITICAL — T5 attention must NOT apply `1/sqrt(d_kv)` scaling

**Source of truth** (`modeling_t5.py` lines 373, 529, 558):
```python
# Mesh TensorFlow initialization to avoid scaling before softmax
self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
...
scores = torch.matmul(query_states, key_states.transpose(3, 2))  # raw Q·K^T, no scale
...
scores += position_bias_masked
```
T5's `q_proj` is initialized such that the scaling is absorbed into the weights, and
**HF never divides scores by `sqrt(d_kv)`**. There is no `scale=` anywhere in
`T5Attention.forward`.

**Rust port**: `flame_sdpa` always applies `scale = 1.0 / sqrt(d_q)` unconditionally
— see `flame-core/src/sdpa.rs:181`:
```rust
let scale = 1.0 / (d_q as f32).sqrt();
scores = scores.mul_scalar(scale)?;
```
and `forward_bf16_fallback` (same file, line 220). There is no opt-out. With
`d_kv=64` this silently attenuates every T5 logit by `1/8` → softmax output is a
near-uniform distribution → all T5 tokens collapse toward the mean. Even if T5.C1
were fixed, this would still break FLUX prompt adherence.

**Severity**: CRITICAL.

**Verbatim fix**: use the manual path in Option B of T5.C1 (no scale), or extend
`flame_sdpa_with_bias` with a `scale: Option<f32>` parameter and pass `Some(1.0)` for T5.
```rust
pub fn forward_with_bias_scale(
    q: &Tensor, k: &Tensor, v: &Tensor,
    additive_bias: Option<&Tensor>,
    scale: Option<f32>,   // None → 1/sqrt(d), Some(1.0) for T5
) -> SdpaResult<Tensor>
```

---

## T5.C3 — CRITICAL — Softmax in FP32 with upcast before, downcast after

**Source of truth** (`modeling_t5.py` line 561):
```python
attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
```
Explicitly upcasts to FP32 for softmax then casts back. This matters for numerical
stability because of the unscaled-logits path (T5.C2) — raw Q·K^T with d_model=4096,
n_heads=64, d_kv=64 produces logit magnitudes >>1, BF16 softmax loses catastrophic
precision.

**Rust port**: `flame_sdpa` would do this internally, but if you implement the manual
path per T5.C1 Option B you must upcast explicitly. Current Rust code does not exist
yet; when writing the fix, the manual path shown in T5.C1 already upcasts to FP32 —
**do not skip the `to_dtype(DType::F32)` step**.

**Severity**: CRITICAL (without FP32 softmax, BF16 overflow guaranteed on T5-XXL).

**Verbatim fix**: Follow Option B in T5.C1 exactly — both scores computation and softmax
stay in FP32, downcast only after `attn @ V`.

---

## T5.H1 — HIGH — Encoder pad-token attention mask is dropped

**Source of truth** (BFL conditioner.py line 34): `attention_mask=None` explicitly.
BFL does not pass an attention mask, so in HF's path, pad tokens become active keys
in attention. Mathematically this *works* because the T5 embed weight at pad_token_id
is tiny and the model learned to ignore it — BFL relies on this.

**Rust port**: Rust also passes no attention mask. Functionally matches BFL. BUT the
Rust `encode(&self, token_ids: &[i32])` takes an arbitrary slice length and never
pads to 512, while BFL always pads to 512 (`padding="max_length"`). If the caller of
`T5Encoder::encode` passes only the meaningful tokens (not padded to 512), T5 will
produce a shorter hidden state `[1, seq, 4096]` that does not match what FLUX DiT
expects (which is trained on 512-length T5 output).

**Severity**: HIGH — silent shape / distribution mismatch at the DiT txt input.

**Verbatim fix**: either pad inside `encode`
```rust
pub fn encode(&mut self, token_ids: &[i32]) -> Result<Tensor> {
    let max_len = self.config.max_seq_len; // 512
    let mut padded = token_ids.to_vec();
    padded.resize(max_len, 0); // T5 pad_token_id = 0
    padded.truncate(max_len);
    // ... use padded
}
```
or document the contract loudly in the doc comment and enforce
`assert_eq!(token_ids.len(), 512)`.

---

## T5.H2 — HIGH — `t5_relative_position_bucket` handles `bidirectional=true` incorrectly

**Source of truth** (`modeling_t5.py` lines 422–446):
```python
relative_buckets = 0
if bidirectional:
    num_buckets //= 2
    relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
else:
    relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
# relative_position is now in [0, inf)

max_exact = num_buckets // 2  # note: num_buckets was already halved above
is_small = relative_position < max_exact

relative_position_if_large = max_exact + (
    torch.log(relative_position.float() / max_exact)
    / math.log(max_distance / max_exact)
    * (num_buckets - max_exact)
).to(torch.long)
relative_position_if_large = torch.min(relative_position_if_large, num_buckets - 1)

relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
return relative_buckets
```
For `num_buckets=32, bidirectional=True, max_distance=128`: after halving,
`num_buckets=16`, `max_exact=8`, and the log bucket span is `16-8 = 8` buckets.

**Rust port** (`t5_encoder.rs::compute_relative_bias` and
`t5_relative_position_bucket` lines 346–379):
```rust
fn compute_relative_bias(..., num_buckets: usize, ...) {
    let half_buckets = num_buckets / 2;         // 16
    ...
    let bucket = t5_relative_position_bucket(
        relative_position, true, half_buckets, max_distance,  // <-- passes 16
    );
}

fn t5_relative_position_bucket(..., num_buckets: usize, ...) {
    let mut ret = 0usize;
    let mut n = -(relative_position);
    if bidirectional {
        let num_buckets = num_buckets;          // shadow no-op — already pre-halved
        if relative_position > 0 {
            ret += num_buckets;                  // adds 16 ✓
            n = -n;
        }
    } ...
    let half_buckets = num_buckets / 2;         // 8 ✓ (this is "max_exact")
    ...
}
```
Outcome: bucket values are in the range `[0, 32)` which matches the embedding table's
`relative_attention_num_buckets=32` rows. **This path is numerically correct by
accident**, because the caller pre-halves and the callee re-halves. But it is fragile
and confusing — the callee does *not* perform the HF line
`num_buckets //= 2` for bidirectional, and instead expects an already-halved count.

A separate **actual** bug: `n = -(relative_position)` is wrong sign for the
unidirectional branch.
```
let mut n = -(relative_position);     // n = - (j-i) = i-j  (Rust)
// HF unidirectional: relative_position = -torch.min(relative_position, 0)
//   for relative_position = j-i, this zeroes out positive j-i and negates negatives
//   i.e. returns max(0, i-j) = max(0, -relative_position)
if bidirectional { ... }
else if n < 0 { n = 0; }
```
For unidirectional: HF wants `max(0, -(j-i)) = max(0, i-j)`. Rust computes
`n = -(j-i) = i-j`, then if `n<0` sets `n=0`. `max(0, i-j) == max(0, -(j-i))`. ✓
Equivalent.

For bidirectional with `relative_position > 0` (i.e. `j > i`, future token):
`ret += num_buckets; n = -n;` → `n` becomes `-(-(j-i)) = j-i`, positive. ✓

For bidirectional with `relative_position <= 0`:
`n = -(j-i) = i-j`, remains positive. ✓

So **the Rust bucket math happens to be correct** for bidirectional, but the
"shadow" parameter pre-halving contract is undocumented and extremely easy to break.

**Severity**: HIGH — correctness-by-accident. Add test vectors covering
`(i,j) = (0,0), (0,1), (1,0), (0,127), (0,128), (0,511), (511,0)` against HF reference
before shipping.

**Verbatim fix**: inline the HF algorithm verbatim, passing the original
`num_buckets=32` through, and halve inside exactly where HF does:
```rust
fn t5_relative_position_bucket(
    relative_position: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut relative_buckets: usize = 0;
    let mut nb = num_buckets;
    let rp = if bidirectional {
        nb /= 2;
        if relative_position > 0 { relative_buckets += nb; }
        relative_position.unsigned_abs() as usize
    } else {
        (-relative_position).max(0) as usize
    };

    let max_exact = nb / 2;
    let is_small = rp < max_exact;

    let bucket = if is_small {
        rp
    } else {
        let val = max_exact as f64
            + (((rp as f64) / (max_exact as f64)).ln()
                / ((max_distance as f64) / (max_exact as f64)).ln())
            * ((nb - max_exact) as f64);
        (val as usize).min(nb - 1)
    };

    relative_buckets + bucket
}

// caller:
let bucket = t5_relative_position_bucket(
    j as i64 - i as i64,
    true,
    num_buckets,            // pass ORIGINAL 32, not half
    max_distance,
);
```
And drop the `half_buckets` pre-halving in `compute_relative_bias`.

Also note HF uses `memory_position - query_position = j - i`. Rust uses
`relative_position = j as i64 - i as i64` ✓.

---

## T5.L1 — LOW — `t5_layer_norm` FP32 accumulation contract

**Source of truth** (`modeling_t5.py` lines 255–260):
```python
variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
if self.weight.dtype in [torch.float16, torch.bfloat16]:
    hidden_states = hidden_states.to(self.weight.dtype)
return self.weight * hidden_states
```
The variance must be computed in FP32 even when input is BF16.

**Rust port**: delegates to `cuda_ops_bf16::rms_norm_bf16`. Must verify that kernel
upcasts the variance accumulation to FP32. If not, T5 will drift slightly in BF16,
but (unlike T5.C1/C2) is unlikely to be catastrophic.

**Severity**: LOW (verify, don't block).

**Verbatim fix**: grep `rms_norm_bf16` kernel and confirm FP32 accumulator. If absent,
open a separate flame-core issue.

---

## T5.L2 — LOW — Final layer norm weight key fallback

**Source of truth**: T5 encoder final layer norm weight is under
`encoder.final_layer_norm.weight`.

**Rust port** (`t5_encoder.rs` line 100):
```rust
let shared_keys = ["encoder.embed_tokens.weight", "encoder.final_layer_norm.weight", "shared.weight"];
```
Matches. But some T5 safetensor files (e.g. Google T5x exports repackaged by
diffusers) use `final_layer_norm.weight` without the `encoder.` prefix. Add a
fallback or fail loudly.

**Severity**: LOW.

**Verbatim fix**:
```rust
if !shared.contains_key("encoder.final_layer_norm.weight") {
    if let Some(t) = file_weights.get("final_layer_norm.weight") {
        shared.insert("encoder.final_layer_norm.weight".to_string(), t.clone());
    }
}
```

---

# CLIP-L deltas

## CLIP.C1 — CRITICAL — Pooled output uses `max_by_key(id)` instead of first EOS occurrence

**Source of truth** (`modeling_clip.py` lines 638–647, CLIP default
`eos_token_id=49407`):
```python
# The config gets updated eos_token_id from PR #24773
pooled_output = last_hidden_state[
    torch.arange(last_hidden_state.shape[0], device=...),
    (input_ids.to(dtype=torch.int, ...) == self.eos_token_id)
        .int()
        .argmax(dim=-1),     # first index where id == 49407
]
```
`argmax` on an int tensor of `{0,1}` returns the **first** index of the max. CLIP
tokenizer pads with `eos_token_id=49407` (same as EOS), so the first `49407` is the
real EOS.

**Rust port** (`clip_encoder.rs` lines 250–255):
```rust
let eos_pos = token_ids[..seq_len]
    .iter()
    .enumerate()
    .max_by_key(|(_, &id)| id)
    .map(|(i, _)| i)
    .unwrap_or(seq_len - 1);
```
Rust's `Iterator::max_by_key` returns the **last** element on ties (std doc: "If
several elements are equally maximum, the last element is returned"). When the CLIP
tokenizer pads with 49407 (which is what HF's CLIPTokenizer does for FLUX), every
padding slot is also 49407 — `max_by_key` returns `seq_len - 1` (the last padding
token) instead of the first EOS.

**Impact**: pooled output is taken at the last pad slot instead of the real EOS.
Since CLIP is causal, the pad slot at position 76 *does* see all earlier tokens,
so the output is not garbage, but it is **not what CLIPTextModel returns** — the
pooled vector FLUX was trained on is the hidden state at the FIRST EOS position.
FLUX 1 `vector` input drift → global conditioning mismatch → subtle style bias.

Additionally, the Rust code also handles a different case when eos_token_id==2,
but CLIP-L is 49407. We should match the 49407 branch.

**Severity**: CRITICAL — silently wrong pooled vector, matches BFL behavior 0%.

**Verbatim fix**:
```rust
// Find FIRST position where token == eos_token_id (49407 for CLIP-L).
// CLIPTokenizer pads with eos_token_id, so searching for first match is correct.
const CLIP_EOS_TOKEN_ID: i32 = 49407;
let eos_pos = token_ids[..seq_len]
    .iter()
    .position(|&id| id == CLIP_EOS_TOKEN_ID)
    .unwrap_or(seq_len - 1);
```
Add `eos_token_id: i32` to `ClipConfig` (default 49407) for flexibility.

---

## CLIP.C2 — CRITICAL — `build_causal_mask` produces `{1,0}` binary consistent with flame_sdpa, but dtype contract fragile

**Source of truth** (`modeling_attn_mask_utils.py::_make_causal_mask` line 164):
```python
mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
# Result: lower triangle (incl. diagonal) = 0, upper triangle = -inf
# Then added to scores directly.
```
HF semantics: **additive** `{0, -inf}` mask.

**Rust port** (`clip_encoder.rs` lines 114–128):
```rust
let mut data = vec![0.0f32; seq_len * seq_len];
for i in 0..seq_len {
    for j in 0..seq_len {
        if j <= i {
            data[i * seq_len + j] = 1.0;    // allowed
        }
    }
}
```
Produces `{1=allowed, 0=masked}` binary. `flame_sdpa::forward_f32` then converts:
```rust
let ones = full_like(&mask_f32, 1.0)?;
let complement = ones.sub(&mask_f32)?;           // 0 for allowed, 1 for masked
let penalty = complement.mul_scalar(NEG_INF)?;   // 0 for allowed, -inf for masked
scores = scores.add(&penalty)?;
```
Functionally equivalent to the HF additive mask. **However**: `NEG_INF` applied to
float 0 yields `0*-inf = NaN` in IEEE754 strict mode, and to 1 yields `-inf`. Flame
must be using a finite negative (e.g. `-1e30`) to avoid NaN — if it uses true `-inf`
or `f32::NEG_INFINITY`, the `0 * -inf = NaN` produces NaN logits on allowed
positions. Verify.

Checked: `forward_bf16_fallback` (same file lines 397–412) uses the same pattern.
Flame must use a finite sentinel (e.g. `-3.4e38`) to avoid `0 * NEG_INFINITY = NaN`.

**Severity**: CRITICAL if `NEG_INF` is `f32::NEG_INFINITY`, otherwise LOW.

**Verbatim fix**:
1. `grep -n "NEG_INF" /home/alex/EriDiffusion/flame-core/src/sdpa.rs` — if it's
   `f32::NEG_INFINITY`, switch to `-3.0e38_f32`.
2. Alternative in the CLIP port: bypass the binary conversion by constructing the
   additive mask directly and using a future `flame_sdpa_with_bias` (same fn as
   T5.C1 fix):
```rust
fn build_causal_mask_additive(seq_len: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = -3.0e38;   // masked
        }
    }
    Tensor::from_vec(data, Shape::from_dims(&[1, 1, seq_len, seq_len]), device.clone())?
        .to_dtype(DType::BF16)
}
```

---

## CLIP.H1 — HIGH — Position IDs always `0..seq_len`; no truncation if caller passes <77 tokens

**Source of truth** (BFL conditioner.py): tokenizer pads to `max_length=77`, so
CLIPTextTransformer always receives exactly 77 tokens. Position embeddings are
indexed `[0..77)`.

**Rust port** (`clip_encoder.rs` line 211):
```rust
let seq_len = token_ids.len().min(cfg.max_position_embeddings);
```
Takes the *shorter* of caller length and 77. If caller passes only 10 tokens
(unpadded), the encoder runs at seq_len=10, uses position embeddings `[0..10)`, and
then the EOS-pool picks up at position 9. This does not match BFL which always runs
at 77. Same class of bug as T5.H1.

**Severity**: HIGH — silent contract mismatch.

**Verbatim fix**: pad to 77 inside `encode`, or assert the caller passed exactly 77.
```rust
pub fn encode(&self, token_ids: &[i32]) -> Result<(Tensor, Tensor)> {
    let max_len = self.config.max_position_embeddings; // 77
    let mut padded = token_ids.to_vec();
    padded.truncate(max_len);
    while padded.len() < max_len {
        padded.push(49407);  // CLIP pad = eos_token_id
    }
    let token_ids = &padded[..];
    let seq_len = max_len;
    // ...
}
```

---

## CLIP.H2 — HIGH — `final_layer_norm` is applied, but BFL uses `pooler_output`, not `[CLS]`

**Source of truth** (`modeling_clip.py` line 625):
```python
last_hidden_state = self.final_layer_norm(last_hidden_state)
...
pooled_output = last_hidden_state[..., argmax_eos, :]
```
`pooled_output` is the final-LN'd hidden state at EOS — no extra projection, no extra
`post_layernorm`. BFL's `HFEmbedder` reads `outputs["pooler_output"]` which is
exactly this tensor.

**Rust port** (`clip_encoder.rs` lines 244–257): applies `final_layer_norm` then
indexes at `eos_pos`. Matches HF. ✓ — assuming CLIP.C1 is fixed.

*Note*: If one later uses `CLIPTextModelWithProjection` (SDXL style) there would be
an extra `text_projection` linear, but FLUX 1 does **not** use that — BFL loads plain
`CLIPTextModel` (conditioner.py line 14). So no text_projection in the Rust port is
correct.

**Severity**: HIGH documentation — confirm in a comment at the top of `encode` that
this is BFL's `pooler_output`, not `CLIPTextModelWithProjection.text_embeds`.

**Verbatim fix**: add doc comment referencing BFL line 37 (`outputs[self.output_key]`
with `output_key = "pooler_output"`).

---

## CLIP.L1 — LOW — Token IDs built through f32 → i32 round-trip

**Rust port** (`clip_encoder.rs` lines 217–221):
```rust
let ids = Tensor::from_vec(
    token_ids[..seq_len].iter().map(|&id| id as f32).collect(),
    Shape::from_dims(&[seq_len]),
    self.device.clone(),
)?.to_dtype(DType::I32)?;
```
Token IDs up to 49407 fit in f32 exactly (f32 mantissa = 24 bits, 49407 < 2^16), so
no precision loss. But this is cargo-culted and brittle if anyone raises the vocab
size. Same pattern in `t5_encoder.rs` with vocab 32128 (also fits).

**Severity**: LOW.

**Verbatim fix**: `Tensor::from_vec_dtype(..., DType::I32)` or add
`Tensor::from_i32_vec` helper. Not blocking.

---

## CLIP.L2 — LOW — Unused `projection_dim` field in `ClipConfig`

**Rust port** (`clip_encoder.rs` line 48): `projection_dim: 768` is in the config but
nothing reads it. FLUX 1 does not use `CLIPTextModelWithProjection`. Remove to avoid
future confusion.

**Severity**: LOW.

**Verbatim fix**: delete the field or rename to `_unused_projection_dim` with a
comment pointing to this audit.

---

# Additional observations (not scored)

1. **T5 no position embeddings** — confirmed. Rust `encode()` only does
   `token_embeds.unsqueeze(0)`, no additive learned positions. ✓
2. **T5 attention projections have no bias** — confirmed. Rust `layer_forward`
   passes no bias to `linear()`. ✓
3. **T5 FFN activation is `gelu_new` (tanh approximation)** — Rust uses
   `Tensor::gelu()` which dispatches to `fc_gelu_bf16` → kernel uses
   `0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`
   (see `flame-core/src/kernels/geglu_kernels.cu` line 6). ✓ Matches HF `gelu_new`.
4. **T5 pre-norm residual order** — Rust matches HF: `normed = ln(hidden); attn_out
   = attn(normed); hidden = hidden + attn_out`. ✓
5. **T5LayerNorm = RMSNorm no bias, no mean subtraction** — Rust calls
   `rms_norm_bf16(..., Some(weight), eps)` with no bias arg. ✓ (Subject to T5.L1
   verification of FP32 variance accumulation.)
6. **T5 relative bias computed once in layer 0, reused across all layers** — Rust
   computes `position_bias` once from `block[0]` weights before the layer loop and
   passes the same tensor to every `layer_forward`. ✓
7. **CLIP MLP uses `quick_gelu = x * sigmoid(1.702 * x)`** — Rust `quick_gelu` matches
   exactly (line 108, constant `1.702`). ✓
8. **CLIP uses LayerNorm with bias (not RMSNorm)** — Rust `layer_norm` calls
   `layer_norm_bf16(..., Some(weight), Some(bias), eps)`. ✓
9. **CLIP separate Q/K/V projections (not fused)** — Rust loads `q_proj.weight`,
   `k_proj.weight`, `v_proj.weight` separately. ✓ Matches HF state_dict keys.
10. **BFL loads plain `CLIPTextModel` (not `CLIPTextModelWithProjection`)** —
    conditioner.py line 14: `CLIPTextModel.from_pretrained(...)`. So no
    `text_projection` layer is needed in Rust. ✓

---

# Fix priority order

1. **T5.C1 + T5.C2 + T5.C3** — implement `sdpa::forward_with_bias` (or inline
   manual attention in `t5_encoder.rs`) with (a) additive float bias, (b) no scale,
   (c) FP32 softmax. Without this, T5 output is NaN or uniform mush.
2. **CLIP.C1** — switch `max_by_key` to `position(|id| id==49407)`. Without this,
   CLIP pooled vector is wrong in every prompt.
3. **CLIP.C2** — verify `NEG_INF` constant in `flame-core/src/sdpa.rs` is finite
   (e.g. `-3e38`), not `f32::NEG_INFINITY`. If infinity, all CLIP layers produce NaN.
4. **T5.H1 + CLIP.H1** — enforce 512 / 77 padding inside `encode()`.
5. **T5.H2** — inline HF bucket algorithm verbatim with tests against reference.
6. **CLIP.H2** — add doc comment clarifying this is `pooler_output`.
7. Low-severity cleanups.

---

End of audit. Total: **5 CRITICAL, 4 HIGH, 3 LOW** across CLIP and T5.
