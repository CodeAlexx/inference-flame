# Batch C Skeptic Review

Reviewed: `worker/{sd3,qwenimage,ernie,anima}.rs` against reference bins
`sd3_medium_infer.rs`, `qwenimage_gen.rs`, `ernie_image_infer.rs`, `anima_infer.rs`.

## P0 — correctness bugs

**None.** All 4 modules faithfully mirror the reference bins for the
load-order, schedule, CFG, pack/unpack, latent geometry, and VAE-decode
boundaries.

## P1 — likely bugs / UX traps

### 1. Qwen-Image embeddings file does not exist on disk (UX trap, hard fail on first click)

`worker/qwenimage.rs:104-105` hardcodes:
```
const EMBEDS_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/qwenimage_embeds.safetensors";
```

`ls /home/alex/EriDiffusion/inference-flame/output/` shows ONLY:
- `qwenimage_embeds.py.safetensors`
- `qwenimage_embeds.rust.safetensors`

There is **no** `qwenimage_embeds.safetensors`. First click of Generate
on Qwen-Image will Fail with the "embeddings not found" error pointing
the user at `scripts/qwenimage_encode.py`. The error message is clear
(line 271-274), but it's a guaranteed first-run failure and the user has
to either (a) symlink/copy one of the existing files, or (b) actually
re-run the Python encode script.

The reference bin `qwenimage_gen.rs:127-129` has the same default path,
so the worker is faithful — but the disk reality differs from the bin's
assumption.

**Recommendation**: either symlink `qwenimage_embeds.py.safetensors →
qwenimage_embeds.safetensors`, or change the default to one of the
existing files, or surface this in a UI tooltip / status strip so the
user knows what's required before clicking Generate. Flag for user.

### 2. Anima embeddings file: same UX trap, but the file DOES exist

`worker/anima.rs:106-107` hardcodes:
```
const EMBEDS_PATH: &str =
    "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors";
```

That file exists on disk (`ls` confirms) — so first-click works. But
`job.prompt` and `job.negative` are still IGNORED with a warning
(line 149-154). Whatever prompt was used to generate that file is what
gets rendered, regardless of what the user typed in the UI prompt box.

The warning log is fine for devs but invisible to UI users — they'll see
the result of an old prompt, not their typed prompt, with no UI hint.
Flag for user; consider a status-strip indicator like
"⚠ using cached embeddings (prompt ignored)".

### 3. Anima VAE parity: bin doesn't decode, worker does — colors unverified

`anima_infer.rs:145-167` literally saves the latent and instructs the
user to run `decode_anima_latent.py`. The Rust VAE decode path is
**not validated** for Anima.

`worker/anima.rs:415-419` calls `QwenImageVaeDecoder::from_safetensors`
directly on the same latent. This is the same decoder used by the
qwenimage worker. The Anima docstring (line 51-57) acknowledges this:

> "If this produces wrong colors it's a parity issue with the anima
> shipping VAE that Batch C isn't resolving."

The shape pipeline looks correct: latent `[1, T=1, H/8, W/8, 16]`
permuted via `[0, 4, 1, 2, 3]` → `[1, 16, T, H, W]` matches what
`anima_infer.rs:152` produces. Decode output `[1, 3, T=1, H, W]` then
narrow+squeeze to `[1, 3, H, W]` — same pattern as `qwenimage.rs:549-553`.

**Real risk**: even if shape & permutation are right, the QwenImage VAE
may not be the *exact* same VAE checkpoint as what Anima trained against.
The latent stats (mean/std) might be calibrated differently. **Flag for
user**: first generation will reveal whether colors match the Python
decode reference.

### 4. SD3 worker uses `job.negative` for uncond, bin uses empty string

`worker/sd3.rs:252` calls `encode_text_pair(&job.prompt, &job.negative, ...)`.
`sd3_medium_infer.rs:135` hardcodes `let empty = ""` and uses it for the
uncond branch.

This is a **deliberate behavior delta**: the worker properly supports
negative prompts, the bin doesn't. Not a bug — just note that any
SD3-specific tuning that assumed empty-string uncond won't translate
1:1 to UI runs where the user enters text in Negative.

## P2 — minor

### 5. Qwen-Image schedule division-by-zero at steps=1

`worker/qwenimage.rs:407`:
```rust
let t = i as f32 / (steps as usize - 1) as f32;
```
For `steps=1`, `(steps-1)=0`. Same code in the bin — accepted limit.
UI default is 50 so unlikely in practice. Defer.

### 6. ERNIE worker — minor `cond_lens` redundancy

`worker/ernie.rs:315-316` builds `cond_lens` and `uncond_lens` Vecs,
but `cond_real_len` was already computed at line 250. Cosmetic only.
Defer.

### 7. SD3 mock fallback for tokenizer failure uses BOS+EOS pad

`worker/sd3.rs:625-630` matches bin behavior: if the CLIP tokenizer
fails to load, falls back to `[49406, 49407, 49407, ...]`. This produces
garbled output but doesn't crash. Same in bin. Acceptable defensive
fallback; defer.

### 8. SD3 schedule log message indexes `timesteps[steps]`

`worker/sd3.rs:330` logs `timesteps[steps as usize]`. `build_schedule`
returns `steps + 1` values, so this is the terminal value (≈0.0).
Correct, no off-by-one. Confirmed against bin (`build_schedule` is
identical).

## SD 3.5 triple-encoder audit

| Check | Status |
|-------|--------|
| Sequential load order CLIP-L → CLIP-G → T5 | ✓ (lines 462-554) |
| Each encoder dropped in its scope before next loads | ✓ (`{ ... }` blocks) |
| `clear_pool_cache + trim_cuda_mempool` between encoders | ✓ (lines 482, 510, 547) |
| Both cond + uncond encoded BEFORE each encoder drops | ✓ |
| CLIP-L pad to 4096 | ✓ (line 557) |
| CLIP-G pad to 4096 | ✓ (line 558) |
| T5 narrowed to 256 (T5 pads to 512 internally) | ✓ (lines 540-546) |
| Context cat order `[CLIP-L, CLIP-G, T5]` along seq dim | ✓ (line 559) |
| Pooled cat order `[CLIP-L_pool, CLIP-G_pool]` (no T5 pool) | ✓ (line 561) |
| Timestep `t * 1000.0` passed to MMDiT | ✓ (line 354) |
| Schedule: linear sigmas + SD3 shift=3.0 | ✓ (lines 596-608, identical to bin) |
| Real two-pass CFG (cond + uncond) | ✓ (lines 360-377) |
| Default cfg=4.5 | ✓ (line 121) |
| DiT + context dropped before VAE | ✓ (lines 404-410) |
| 16-ch VAE with scale=1.5305, shift=0.0609 | ✓ (lines 117-118) |

VRAM order is right. T5 (~9.4 GB) is loaded last so peak doesn't compound.

## Qwen-Image embeddings-file UX assessment

- **File-missing error message**: clear, points at the script
  (line 271-274). ✓
- **Path is hardcoded**: not configurable per-prompt. Same as the bin.
  The user's typed prompt is silently ignored — the embeddings file
  decides what gets rendered. **This is the UX trap.**
- **Failed event reaches UI**: yes, `Err(RunError::Other(msg))` →
  `WorkerEvent::Failed { id, error: msg }` (line 211-216). UI shows
  the error string directly (matches the existing pattern in
  flux/klein/zimage workers).
- **Pack matches `_pack_latents`**: bin lines 254-277 vs worker
  lines 370-390 — IDENTICAL nested-loop math, same dst/src indexing.
  Both produce `[1, seq, 64]` from `[1, 16, H/8, W/8]`. ✓
- **Unpack matches**: worker lines 528-535 vs bin lines 467-472 —
  same reshape/permute/reshape pipeline. ✓
- **T=1 squeeze at VAE boundary**: worker lines 549-553, narrow(2,0,1)
  + squeeze(Some(2)). Identical to bin lines 484. ✓
- **Norm-rescale CFG**: worker lines 563-578 vs bin lines 519-534 —
  algorithmically identical (cond_norm/comb_norm ratio along last dim).
  Worker falls back to raw `comb` on failure (line 477-485) matching
  bin's `.unwrap_or(comb)`. ✓

## ERNIE sequential-CFG flush-point audit

Reference bin `ernie_image_infer.rs:181-186`:
```rust
let pred_cond = model.forward(...)?;
flame_core::cuda_alloc_pool::clear_pool_cache();
flame_core::device::trim_cuda_mempool(0);
let pred_uncond = model.forward(...)?;
flame_core::cuda_alloc_pool::clear_pool_cache();
flame_core::device::trim_cuda_mempool(0);
```

Worker `ernie.rs:432-442`:
```rust
let pred_cond = model.forward(...)?;
flame_core::cuda_alloc_pool::clear_pool_cache();
flame_core::device::trim_cuda_mempool(0);
let pred_uncond = model.forward(...)?;
flame_core::cuda_alloc_pool::clear_pool_cache();
flame_core::device::trim_cuda_mempool(0);
```

✓ **Identical**. Three flush points present:
1. After cond fwd (line 436-437)
2. After uncond fwd (line 441-442)
3. After CFG ≤ 1 single-pass branch (line 457-458)
4. Terminal: after DiT drop, before VAE (line 485-486)
5. Terminal: closure-wrap exit (line 214-215)

| Check | Status |
|-------|--------|
| `ErnieImageModel::load(weights, config)` signature | ✓ matches `ernie_image.rs:383` |
| Schedule = `ernie_schedule(steps)`, shift=3.0, exp time-shift | ✓ (sampling/ernie_sampling.rs:18, SHIFT=3.0) |
| `sigma_to_timestep(sigma) = sigma * 1000` | ✓ (sampling/ernie_sampling.rs:38) |
| `ernie_euler_step` does `x + dt * v` | ✓ (sampling/ernie_sampling.rs:50) |
| Empty-string uncond is encoded (NOT zeros) | ✓ (worker line 255-258, mirrors bin line 73-76) |
| Trim text embeddings to real length (no padding) | ✓ (lines 309-314, mirrors bin 165-168) |
| Mistral-3 dropped before DiT loads | ✓ (line 280 scope close + line 281 flush) |
| DiT dropped before VAE loads | ✓ (line 480-486) |
| Klein VAE used (16-ch, 8× × 2× = 16× downscale) | ✓ (line 374-375 / VAE_DOWNSCALE=16) |

## Anima embeddings-file + VAE parity assessment

**Embeddings file**:
- Path `/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors`
  exists on disk — first run won't hit "file missing".
- Reads `context_cond` + `context_uncond` keys (line 251-258), matching
  `anima_infer.rs:44-45`. ✓
- `job.prompt` ignored with `log::warn!` (line 149-154). Same UX trap as
  Qwen-Image.

**VAE parity** (the unresolved question from the brief):
- Bin saves the latent and runs Python decode. Worker decodes in Rust
  via `QwenImageVaeDecoder` — same struct used by the qwenimage worker.
- 5D channel-LAST → channels-at-1 permute: `[1, 4, 1, 2, 3]`
  (worker line 403). Matches bin line 152 exactly.
- The Anima latent has channels at index 4 (last); bin's permute `[0, 4,
  1, 2, 3]` brings it to `[B, 16, T, H, W]`. ✓
- Decode produces `[B, 3, T=1, H, W]` per `qwen_image_vae.rs` docstring.
  Worker narrow+squeeze to `[B, 3, H, W]` (line 425-429). ✓
- **Whether the QwenImage VAE checkpoint is the right calibration for
  Anima latents is unverifiable from code review** — needs an actual run.
  The fact that the bin punts to Python is the warning sign.

**Schedule + CFG**:
| Check | Status |
|-------|--------|
| Linear sigma `1.0 - i/steps`, NO shift | ✓ (line 328-330, mirrors bin 90-92) |
| Timestep RAW sigma (NOT * 1000) | ✓ (line 346-351, mirrors bin 110) |
| Real two-pass CFG | ✓ (lines 353-368) |
| Default cfg=4.5 | ✓ (line 110) |
| Default steps=30 | ✓ (line 113) |

## Hardcoded path existence

Verified with `ls`:

| Path | Exists |
|------|--------|
| `/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors` | ✓ |
| `/home/alex/.serenity/models/text_encoders/clip_l.safetensors` | ✓ |
| `/home/alex/.serenity/models/text_encoders/clip_g.safetensors` | ✓ |
| `/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors` | ✓ |
| `/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json` | ✓ |
| `/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json` | ✓ |
| `/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json` | ✓ |
| `/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer/diffusion_pytorch_model-00001..00009-of-00009.safetensors` | ✓ all 9 |
| `/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors` | ✓ |
| `/home/alex/EriDiffusion/inference-flame/output/qwenimage_embeds.safetensors` | **✗ MISSING** (only `.py` and `.rust` infix variants exist) |
| `/home/alex/models/ERNIE-Image/transformer/` (2 shards) | ✓ |
| `/home/alex/models/ERNIE-Image/text_encoder/model.safetensors` | ✓ |
| `/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json` | ✓ |
| `/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors` | ✓ |
| `/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/anima-preview2.safetensors` | ✓ |
| `/home/alex/EriDiffusion/Models/anima/split_files/vae/qwen_image_vae.safetensors` | ✓ |
| `/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors` | ✓ |

**Only Qwen-Image's hardcoded `EMBEDS_PATH` is missing.** See P1.1.

## Closure-wrap + cancel-check audit

All 4 modules adopt the closure-wrap pattern from flux/chroma/klein/zimage:

| Module | Closure-wrap location | Pool flush on every exit |
|--------|----------------------|-------------------------|
| `sd3.rs` | lines 227-233 | ✓ |
| `qwenimage.rs` | lines 248-254 | ✓ |
| `ernie.rs` | lines 211-217 | ✓ |
| `anima.rs` | lines 219-225 | ✓ |

Cancel-check via `drain_pending(ui_rx, pending)?` between steps:

| Module | Drain points (pre-step) |
|--------|------------------------|
| `sd3.rs` | lines 262, 283, 333, 342 (in-loop) |
| `qwenimage.rs` | lines 298, 314, 435, 441 (in-loop) |
| `ernie.rs` | lines 288, 318, 366, 415, 420 (in-loop) |
| `anima.rs` | lines 269, 284, 332, 337 (in-loop) |

✓ All 4 perform `drain_pending` at the top of every denoise step. Same
`UiMsg::Cancel | UiMsg::Shutdown` handling as the existing workers.
Cancel-then-immediately-Generate should work.

## ModelKind / dispatch / IMAGE_MODELS audit

**`ModelKind` variants** (`worker/mod.rs:42-78`): Sd35, QwenImage,
ErnieImage, Anima — all 4 added. ✓

**`from_model_string` ordering** (`worker/mod.rs:92-132`):
Order: zimage → chroma → klein9b → klein → flux → sd3/sd-3.5 → qwen →
ernie → anima → Mock. Checked for collisions:

| Test string | Expected | Result |
|------------|----------|--------|
| `"sd3.5-medium.safetensors"` | Sd35 | ✓ matches `contains("sd3")` |
| `"qwen-image.safetensors"` | QwenImage | ✓ no earlier match |
| `"ernie-image-8b.safetensors"` | ErnieImage | ✓ no earlier match |
| `"anima-2b.safetensors"` | Anima | ✓ no earlier match |
| `"sd3.5-large.safetensors"` | Sd35 (intentional) | ✓ same arm |

No collisions. The "Sd35" arm catches both Medium and Large filenames —
but the worker only loads the Medium-pathed checkpoint, so a Large
filename would silently load Medium weights. The `from_model_string`
docstring (lines 116-120) acknowledges this.

**`mock.rs` dispatch arms** (lines 189-256): includes all 4 new variants
with lazy-init pattern matching the existing Z-Image / Flux / Chroma /
Klein arms. ✓

**`IMAGE_MODELS`** (`sections/model.rs:21-44`): includes the 4 new
filename strings. ✓

## AGENT-DEFAULT assessment (4)

| # | Decision | Assessment |
|---|----------|------------|
| 1 | Qwen-Image hardcoded `EMBEDS_PATH`, prompt IGNORED | **fix-or-flag**: file doesn't exist on disk. At minimum surface the requirement in the UI. Otherwise: every first run fails. |
| 2 | Anima hardcoded `EMBEDS_PATH`, prompt IGNORED | **flag-for-user**: file exists, but the silent prompt-ignored behavior is invisible in the UI. Add a visual indicator when these workers run. |
| 3 | Anima VAE uses `QwenImageVaeDecoder`, bin punts to Python | **defer + flag-for-user**: code review can't verify color parity. First real run will reveal it. The shape pipeline is correct. |
| 4 | ERNIE empty-string uncond encoded (not zero) | **acceptable**: matches diffusers `ErnieImagePipeline` reference. Documented in worker docstring (lines 16-22) and bin (lines 68-72). Correct decision. |

## Unverified

- **Tokenizer files at runtime**: workers gracefully fall back to BOS+EOS
  if tokenizer load fails. Not a correctness verification but acceptable
  defensive code.
- **Anima VAE color/contrast parity**: not verifiable from code review.
- **ERNIE per-step pool clear actually prevents OOM at step 5**: worker
  faithfully mirrors bin pattern, but I didn't run the binary to confirm
  the OOM threshold the docstring claims.
- **SD3 with non-empty `job.negative`**: deliberate behavior delta from
  bin. Worker correctness is fine; output quality is unverified.
- **`Tensor::randn_seeded` vs hand-rolled Box-Muller statistical
  equivalence**: documented as AGENT-DEFAULT. Both produce N(0,1) with
  the same seed contract; bit-exact reproducibility is not preserved
  but seed reproducibility within the worker is.
- **Norm-rescale CFG fallback path on `sum_dim_keepdim` errors**: the
  fallback is wired (worker line 477-485), but I didn't audit which
  flame-core dtype combinations might trigger it.
- **Compilation hygiene**: builder reported clean compile + 2 pre-existing
  warnings; not re-verified per task instructions.
