# FLUX 1 Dev — Rust Port Audit vs BFL Source of Truth

**Date:** 2026-04-06
**Reference:** `/home/alex/black-forest-labs-flux/` (cloned from `github.com/black-forest-labs/flux`, commit at clone time on 2026-04-06)
**Audited:** `inference-flame/src/models/flux1_dit.rs`, `clip_encoder.rs`, `t5_encoder.rs`, `vae/ldm_decoder.rs`; `flame-core/src/sdpa.rs`

## Methodology

Three parallel audits cross-referenced every line of the Rust port against the
BFL Python source. For each delta the report records:

- **What** — the exact difference
- **Where in BFL** — `file:line` of the canonical Python
- **Where in Rust** — `file:line` of the port
- **Severity** — CRITICAL (output garbage), HIGH (numerical drift), LOW (cosmetic)
- **Verbatim fix** — the BFL code translated into the Rust API surface

The audits were intentionally exhaustive in the regions of risk flagged by
the FLUX 1 handoff: RoPE format, per-block modulation, T5 attention scaling
and bias, CLIP pooled extraction, VAE latent normalization order, and the
end-to-end sampling loop. They were intentionally light on areas already
verified during port (weight key naming, bias presence on linears, qkv
concat order).

## Tally

| Component | CRITICAL | HIGH | LOW |
|---|---|---|---|
| DiT (`flux1_dit.rs`) | 3 | 4 | 4 |
| Encoders (CLIP + T5) | 5 | 4 | 3 |
| VAE + sampling | 5 | 4 | 3 |
| **Total** | **13** | **12** | **10** |

---

## DiT — `flux1_dit.rs`

### CRITICAL

**C1. RoPE rotation signs flipped.** Rust `rotate_complex` (lines 365-366)
implements the textbook complex rotation
`out_re = re·cos − im·sin; out_im = re·sin + im·cos`. BFL's `apply_rope`
(`math.py:25-30`) expanded against the `[cos,-sin,sin,cos]` 2×2 matrix from
`rope()` (`math.py:20`) produces
`component0 = cos·re + sin·im; component1 = −sin·re + cos·im`. Both
off-diagonal signs are inverted — every token rotates in the opposite
direction. **Fix:** verbatim port of BFL formula or, equivalently, call
`flame_core::bf16_ops::rope_fused_bf16` whose kernel comment states
"Used by Klein/Flux, LTX, HunyuanVideo".

**C2. RoPE storage layout differs.** BFL stores `pe` as
`[B, 1, N, D/2, 2, 2]` (the full 2×2 rotation matrix per position,
unsqueezed once for the head broadcast). Rust stores only `{cos, sin}`
stacked as `[2, 1, 1, N, 64]`. As written the two formats can't be
swapped — they only happen to mask each other if C1 is also "wrong" in
a compensating direction.

**C3. Dead halfsplit branch in `rotate_complex`.** Lines 352-353 read
`x.narrow(3, 0, half)` and `x.narrow(3, half, half)` (halfsplit) but
those bindings are immediately shadowed by the correct pair-reshape
on lines 360-361. Currently shadowed (correctness LOW), but the comment
"FLUX uses view_as_complex which interleaves..." is misleading and any
future edit could silently revive the wrong path.

### HIGH

**H1. Timestep embedding loses precision.** BFL keeps `freqs` and
`time_factor * t` in FP32 throughout (`layers.py:39-43`), casting to
the target dtype only at return. Rust casts `freqs` to BF16 before the
matmul, computing the entire angular argument in BF16. Visible drift on
guidance (which runs the same path with values 1.0–10.0).

**H2. `txt_in.bias` assumed unconditionally present.** BFL `model.py:61`
defaults `bias=True`. Rust loads `txt_in.bias` unconditionally; LoRA
wrappers or stripped checkpoints will hit `Missing: txt_in.bias`.

**H3. `Flux1Config::load` hardcodes `dev()`.** Schnell vs Dev is not
auto-detected. A Dev model called without a guidance tensor silently
falls through and runs as Schnell (no error). BFL raises
`ValueError("Didn't get guidance strength for guidance distilled model")`.

**H4. `timestep_mlp` helper has 2-D vs 3-D ambiguity.** Used for both
the timestep path (already 2-D) and CLIP pooled `vector_in`. Future
caller passing `[B, 1, D]` would misinterpret the middle dim. Rename
and assert.

### LOW

**L1.** Modulation chunk order `(shift, scale, gate)` matches BFL but
relies on undocumented offsets — add a comment.

**L2.** SiLU-then-linear order confirmed correct (`vec.silu()?` then
`linear_bias`).

**L3.** `final_layer` `(shift, scale)` chunk order confirmed correct.

**L4.** Dead halfsplit branch (see C3, reclassified to LOW correctness).

---

## Text Encoders — `clip_encoder.rs`, `t5_encoder.rs`

### CRITICAL

**T5.C1. Position bias passed as binary mask to `flame_sdpa`.**
HF T5 (`modeling_t5.py:529, 558, 561`) does
`scores += position_bias_masked` (additive float bias). The Rust port
calls `flame_sdpa(..., Some(position_bias))`. `flame_sdpa` treats the
`mask` argument as a binary keep-mask and applies
`(1 - mask) * NEG_INF`. A learned bias of `0.37` becomes
`0.63 * -∞ = -∞`. Every softmax row collapses to NaN or a single hot
position. **T5 output is meaningless.**

**T5.C2. T5 attention must NOT divide by `√d_kv`.** HF T5 absorbs the
scaling into the q-projection initialization (Mesh TF style). The
T5 forward at `modeling_t5.py:558` is unscaled `Q·K^T`. `flame_sdpa`
unconditionally applies `1/sqrt(d_q)` and has no opt-out. With
`d_kv=64` this attenuates every T5 logit by `1/8` → softmax collapses
to near-uniform → all T5 tokens drift toward the mean.

**T5.C3. Softmax must be FP32.** HF T5 (`modeling_t5.py:561`) explicitly
upcasts to FP32 before softmax then casts back. With T5's unscaled
logits at d_model=4096, BF16 softmax overflows. Either use the manual
attention path that upcasts, or extend the SDPA primitive.

**CLIP.C1. Pooled output picks last EOS-tied pad slot.** HF CLIP
(`modeling_clip.py:638-647`) takes the hidden state at
`(input_ids == eos_token_id).int().argmax(-1)` — `argmax` on `{0,1}`
returns the **first** index. CLIPTokenizer pads with `eos_token_id=49407`,
so the first match is the real EOS. Rust uses `Iterator::max_by_key`
which returns the **last** tied element, landing on the final pad slot
instead of the real EOS. The pooled vector FLUX was trained on is the
hidden state at the FIRST EOS — silent global-conditioning drift on
every prompt.

**CLIP.C2. `NEG_INF` must be a finite sentinel.** The masked-attention
path uses `complement.mul_scalar(NEG_INF)`. `0 * f32::NEG_INFINITY =
NaN` would poison every allowed position. Verified in flame-core that
`NEG_INF = -1.0e9` (finite) — already correct, no fix needed, but flag
to keep it that way.

### HIGH

**T5.H1.** Encoder takes arbitrary token-id slice length and never pads
to 512. BFL always pads (`padding="max_length"`). Caller-side assumption
mismatch.

**T5.H2.** `t5_relative_position_bucket` is correct **by accident** — the
caller pre-halves `num_buckets` and the callee re-halves, producing the
correct value but in a way that's confusing and one wrong edit away
from breaking. Inline the HF algorithm verbatim.

**CLIP.H1.** Same length-contract problem as T5.H1: caller can pass
fewer than 77 tokens and the encoder will silently run at the shorter
length, mismatching the FLUX DiT input distribution.

**CLIP.H2.** Document that the pooled output is HF's `pooler_output`
(final-LN'd hidden at first EOS), not `CLIPTextModelWithProjection.text_embeds`.

### LOW

**T5.L1.** `t5_layer_norm` delegates to `rms_norm_bf16` — verify the
kernel uses an FP32 variance accumulator.

**T5.L2.** Add fallback for `final_layer_norm.weight` keys missing the
`encoder.` prefix.

**CLIP.L1.** Token IDs round-tripped through f32 → i32. Currently safe
(vocab 49407 < 2^16) but cargo-culted.

---

## VAE + Sampling — `vae/ldm_decoder.rs`, no flux1 sampling existed

### CRITICAL

**VAE.C1. `decode()` latent normalization order is reversed.** BFL
(`autoencoder.py:313-315`) does `z = z / scale_factor + shift_factor`.
Rust does `(z - shift_factor) * (1/scale_factor)` which equals
`z/scale - shift/scale` — differs by a constant offset of
`shift + shift/scale ≈ 0.437` per channel. Z-Image happened to work with
the inverted form because its encode side was inverted symmetrically;
**FLUX 1 will produce a colour-shifted, broken decode** until this is
fixed.

**VAE.C2. No FLUX 1 `flux_time_shift` / `get_schedule` exists.**
`klein_sampling.rs::compute_mu` uses Flux **2**'s empirical mu
(`a1=8.738e-05, b1=1.898, a2=1.693e-04, b2=0.457`). FLUX 1 uses a linear
estimator through `(256, 0.5)` and `(4096, 1.15)`. Wrong schedule →
visibly worse images even if everything else is correct.

**VAE.C3. No FLUX 1 denoise loop.** `flux1_test.rs` runs ONE forward and
exits. BFL `sampling.py:308-353` is a single-pass Euler with
`guidance_vec = ones*guidance` injected as a model input — **no CFG**
because Dev is guidance-distilled. No call site for FLUX 1 anywhere.

**VAE.C4. No `pack` / `unpack` rearrange helpers.** BFL packs latent
`(b, 16, h, w) → (b, h*w/4, 64)` via
`rearrange "b c (h ph) (w pw) -> b (h w) (c ph pw)"`. Rust has no
implementation; `flux1_test.rs` fakes a `[1, 4096, 64]` shape directly,
never proving the layout.

**VAE.C5. No `[-1,1] → uint8` PNG denorm.** VAE returns raw `[-1, 1]`
pixels. No writer in inference-flame consumes them for FLUX 1.

### HIGH

**VAE.H1.** AttnBlock uses single-head SDPA with `1/sqrt(C)` scale.
Correct provided `flame_core::sdpa::forward` infers head_dim from the
expected axis. Verification only, no known bug.

**VAE.H2.** `num_resnets=3` is correct (BFL `num_res_blocks=2 + 1`)
but undocumented. Maintenance trap — anyone "fixing" it to 2 will
silently break decoding.

**VAE.H3.** `upsample2d_nearest` kernel must be integer replication,
not bilinear. Confirm in `flame-core/src/cuda_kernels.rs`.

**VAE.H4.** No call site constructs `LdmVAEDecoder` with the FLUX 1
constants `(in_channels=16, scale=0.3611, shift=0.1159)`.

### LOW

**VAE.L1.** Header comment claims the inverted op order — update after
C1 lands.

**VAE.L2.** `num_resnets` comment is ambiguous (uses "layers_per_block",
diffusers terminology).

**VAE.L3.** `flux1_test.rs::img_ids` ordering matches BFL by accident;
add a test once `pack_latent` lands.

---

## Resolution Status

Of the 13 CRITICALs and 12 HIGHs, the following were addressed in the
work that followed the audits:

### Fixed

- **DiT C1/C2 (RoPE)** — first via verbatim 6D port, then replaced with
  `flame_core::bf16_ops::rope_fused_bf16` (single fused kernel,
  150× faster than the verbatim path; see profile section below).
- **DiT H1** — `timestep_embedding` keeps freqs/args/cos/sin in FP32,
  casts once at return.
- **DiT H2** — `txt_in.bias` is now optional, falls back to
  `linear_nobias`.
- **DiT H3** — `has_guidance` auto-detected from
  `guidance_in.in_layer.weight` presence at load time. Forward errors
  if `has_guidance && guidance.is_none()`.
- **DiT H4** — `timestep_mlp` asserts 2-D input.
- **T5 C1+C2+C3** — added `flame_core::attention::sdpa_with_bias` (and
  `flame_core::sdpa::forward_with_bias`) with optional additive float
  bias and optional `scale=Some(1.0)` to skip the `1/sqrt(d)` divide.
  Manual FP32 path with FP32 softmax. Wired into `t5_encoder.rs`.
- **T5 H1** — `encode` pads/truncates to `max_seq_len=512`.
- **T5 H2** — `t5_relative_position_bucket` rewritten verbatim from HF,
  caller passes the original `num_buckets=32`.
- **CLIP C1** — `max_by_key` replaced with `position(|id| id == 49407)`.
- **CLIP C2** — verified `NEG_INF = -1.0e9` (already finite).
- **CLIP H1** — `encode` pads to 77.
- **VAE C1** — `decode` op order corrected to `z / scale + shift`.
- **VAE C2** — `src/sampling/flux1_sampling.rs` created with verbatim
  ports of `time_shift`, `flux1_mu`, `get_schedule`,
  `pack_latent`, `unpack_latent`, `flux1_denoise`.
- **VAE C3** — `flux1_denoise` Euler loop with `guidance_vec` model
  input, no CFG.
- **VAE C4** — `pack_latent` / `unpack_latent` implemented.
- **VAE C5** — denorm + PNG write integrated into `flux1_infer.rs`.

### Pending (or surfaced after first run)

- **DiT C3 / L4** — dead halfsplit branch was removed when RoPE was
  rewritten to use `rope_fused_bf16`. Effectively resolved.
- **T5 / CLIP / DiT linear helpers** still call
  `transpose2d_bf16(weight)` per call instead of pre-transposing at
  load. The per-call transpose is small (~290us each on 4096²)
  and is dwarfed by other costs — left for a later cleanup.
- **VAE GroupNorm dtype mismatch** — `ae.safetensors` is F32 on disk;
  `group_norm_bf16` requires BF16 storage. Surfaces after the DiT runs
  successfully. Same class of bug as the F16/F32 cast trap on CLIP and
  T5 (loader upcasts non-BF16 to F32 internally).
- **F16/BF16 storage trap in flame-core loader** — `serialization::load_file`
  upcasts any F16 source tensor to F32. Models loaded from `*_fp16.safetensors`
  must be cast back to BF16 by the caller before BF16 kernels can consume
  them. Worked around in CLIP (cast at the call site) and T5 (resident
  load + cast in `T5Encoder::load`).
- **flame-swap parser silently skips F16** — `flame-swap/src/swap.rs:630`
  matches only `BF16`, `F32`, `F8_E4M3`. F16 tensors hit `_ => continue`
  and the swap reports zero blocks. T5 was rerouted to a resident load
  (~9.7 GB) as a workaround.

## Profile-Discovered Bottlenecks (post-audit)

After the audit fixes landed, an end-to-end profile of one FLUX 1 DiT
double-block found two non-audit bottlenecks not visible from a static
read of the code:

| Section | Original | After fix | Speedup |
|---|---|---|---|
| RoPE (5D narrow + F32 broadcast path) | 4939 ms | 33 ms (`rope_fused_bf16`) | 150× |
| `split_qkv` (5D permute → CPU fallback) | 2333 ms | 1 ms (3D narrow + 4D `[0,2,1,3]` hot path) | 2333× |
| **Block total** | **7625 ms** | **398 ms** | **19×** |

The audit identified the *correctness* problem with the verbatim 6D RoPE
port (it implemented the wrong rotation, see C1) but did not catch that
the Rust path's 5D narrows would fall through to the F32 CPU staging
fallback and take 5 seconds per call. Both findings only surfaced under
a forced-sync per-section profile of one block.

The `split_qkv` bug was not in the audit scope at all — it was a
copy of the diffusers idiom `reshape(B,N,3,H,D).permute(2,0,3,1,4)`
that hits a 5D permute, which falls through to the "general
permutation fallback (CPU copy)" path in `flame-core/src/tensor.rs`.

## Files

- `inference-flame/src/models/flux1_dit.rs` — DiT (audited and patched)
- `inference-flame/src/models/clip_encoder.rs` — CLIP-L (audited and patched)
- `inference-flame/src/models/t5_encoder.rs` — T5-XXL (audited and patched, then
  resident-loaded as a workaround for the flame-swap F16 skip)
- `inference-flame/src/vae/ldm_decoder.rs` — LDM VAE decoder (audited and patched)
- `inference-flame/src/sampling/flux1_sampling.rs` — created post-audit
- `inference-flame/src/bin/flux1_infer.rs` — created post-audit
- `flame-core/src/sdpa.rs` — added `forward_with_bias`
- `flame-core/src/attention/sdpa.rs` — added `sdpa_with_bias` public wrapper
- `black-forest-labs-flux/src/flux/{model.py, modules/layers.py, math.py,
  modules/conditioner.py, modules/autoencoder.py, sampling.py, util.py}`
  — source of truth, cloned at `/home/alex/black-forest-labs-flux/`
