# SKEPTIC_BATCH_D.md

Skeptic review of Batch D workers — SDXL, SD 1.5, Stable Cascade.

Files audited:
- `/home/alex/EriDiffusion/inference-flame/inference_ui/src/worker/sdxl.rs`
- `/home/alex/EriDiffusion/inference-flame/inference_ui/src/worker/sd15.rs`
- `/home/alex/EriDiffusion/inference-flame/inference_ui/src/worker/cascade.rs`
- `/home/alex/EriDiffusion/inference-flame/inference_ui/src/worker/mod.rs` (dispatch)

Reference bins:
- `/home/alex/EriDiffusion/inference-flame/src/bin/sdxl_infer.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/sd15_infer.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/cascade_infer.rs`

Bottom-line counts: **0 P0, 1 P1, 5 P2.** All three workers mirror their bins
faithfully. The schedule, CFG, Euler/DDIM step formulas, and VAE constants
are byte-for-byte equivalent to the bins. The one P1 is a genuine UX trap
(cascade ETA accounting), the rest are minor.

---

## P0 — correctness bugs

**None found.**

I expected to find at least one sign-flip or scale-factor mix-up across three
new workers. I didn't. The three Euler/DDIM formulas, the two VAE scale
constants, the CFG composition, the dual-encoder embeddings load order, and
the VAE attention-key rename all match their bins exactly.

---

## P1 — likely bugs / UX traps

### P1.1 — Cascade ETA computation includes Stage C load time, biasing early progress estimates

`cascade.rs:391-394`:
```rust
let t_stage_c_load = Instant::now();
let t_denoise_start = Instant::now();   // ← same instant, BEFORE load
```

Then per-step ETA at line 443-446:
```rust
let elapsed = t_denoise_start.elapsed().as_secs_f32();
let per_step = elapsed / step_global as f32;
```

`t_denoise_start` is captured **before** the Stage C UNet load, which takes
~3-7 s on warm SSD. So at step 1, `elapsed = load_time + step_1_time`, and
the per-step estimate is ~2-3× too high. ETA bar will start way over and
shrink. By step 5 it's close, by step 10 it's accurate.

Same compounding happens at the Stage C → Stage B boundary: Stage B's load
time is folded into the running average across both stages, so the ETA at
step `steps_c + 1` jumps when it shouldn't.

This is not a correctness bug but it's a real UX trap — users will see "ETA
3 minutes" for a 60s job and either cancel or distrust the bar.

**Fix**: capture `t_denoise_start = Instant::now()` AFTER Stage C load
returns (and similarly track stage-local elapsed for Stage B's contribution).

---

## P2 — minor

### P2.1 — SD 1.5 temp-file path overwritten every job, never cleaned up

`sd15.rs:120`:
```rust
const VAE_TMP_PATH: &str = "/tmp/inference_ui_sd15_vae_remapped.safetensors";
```

The remap workaround writes ~167 MB of safetensors to `/tmp` on every job and
overwrites the previous run's file. Across many jobs it never grows past one
copy, but:

- `/tmp` is often `tmpfs` (RAM-backed) → ~167 MB of RAM consumed for
  the lifetime of the process even after the worker drops the in-memory
  HashMap.
- If two `inference_ui` instances ever run on the same host they will race
  on this single path (one writes mid-load of the other).

**Suggested fix**: use `tempfile::NamedTempFile` (already a likely transitive
dep), drop after `from_safetensors` returns. Alternative: extract the diffusers→LDM
remap into a public function in `ldm_decoder.rs` so the workaround can call
it directly without round-tripping through disk. The bin has the same problem
with `/tmp/sd15_vae_remapped.safetensors` so they could share the fix.

### P2.2 — Cascade 2:1 step split inflates very low step counts

`cascade.rs:185-192`:
```rust
let total = job.steps.max(2);
let c = ((2 * total) + 2) / 3;     // ceil(2*total/3)
let b = total.saturating_sub(c).max(1);
```

For `job.steps == 1`: `total = 2`, `c = 2`, `b = 0.max(1) = 1`, sum = 3.
For `job.steps == 2`: `total = 2`, `c = 2`, `b = 0.max(1) = 1`, sum = 3.

A user requesting 1 or 2 total steps actually gets 3. Not a real failure
since 1-step Cascade is a debug-only request, but the Started event reports
the inflated total so accounting stays consistent.

Note: for typical step counts (≥3) the math is exact. `total = 30` → `c=20,
b=10` (matches bin defaults). `total = 45` → `c=30, b=15`. Clean.

### P2.3 — Cascade per-step ETA averages across stages with very different per-step costs

Stage C denoises a 24×24 latent through a ~7 GB UNet; Stage B denoises a
256×256 latent through a ~3 GB UNet. Per-step wall time differs significantly
(Stage B steps are typically slower in absolute terms despite the smaller
UNet, due to the larger spatial footprint and the effnet_cond adapter).

The naive `elapsed / step_global` averaging produces inaccurate ETA at the
Stage C → Stage B transition: the bar advances at one rate, then suddenly
the per-step rate changes. Tied to P1.1; same fix locale applies (track
per-stage timings separately).

### P2.4 — `pony-diffusion-v6.safetensors` falls through to Mock instead of dispatching to Sdxl

Not in any of the three new files but observed during dispatch verification:
`mod.rs::from_model_string` doesn't match "pony" → falls through to Mock.
Pony Diffusion v6 is SDXL-architecture and would Just Work via the Sdxl arm
if `pony` were added to the SDXL match. Out of Batch D scope; flagging
because it's adjacent to the new SDXL dispatch and would surprise a user
selecting `pony-diffusion-v6.safetensors` from the IMAGE_MODELS list.

### P2.5 — SDXL prompt-ignored warning skips when prompt is empty but negative is non-empty

`sdxl.rs:184`:
```rust
if !job.prompt.is_empty() {
    log::warn!("SDXL: TYPED PROMPT IS IGNORED. ...");
}
```

If the user clears the prompt field but types a negative prompt, no warning
fires — the cached embeddings still get used and the negative prompt is
silently dropped. Mirror behavior of qwenimage / anima, so it's consistent
across the cached-embedding workers, but slightly surprising. Low impact —
empty-prompt + non-empty-negative is an unusual combination.

---

## VE Euler formula audit (SDXL / SD 1.5 / Cascade — three formulas)

**SDXL** (`sdxl.rs:411-468` vs `sdxl_infer.rs:159-222`): identical.
- `c_in = 1 / sqrt(sigma^2 + 1)` ✓
- `x_in = x * c_in`, then `to_dtype(BF16)` for UNet input ✓
- `eps_cfg = uncond + cfg * (cond - uncond)` (FP32) ✓
- `dt = sigma_next - sigma` (negative) ✓
- `x_next = x + dt * eps_cfg` (FP32) ✓
- Initial: `x = noise * sqrt(sigma_max^2 + 1)` ✓

**SD 1.5** (`sd15.rs:369-419` vs `sd15_infer.rs:218-247`): identical.
Same five lines as SDXL minus the `y` (label) embedding. Verified
line-for-line.

**Cascade** (`cascade.rs:412-454` Stage C and `:496-546` Stage B vs
`cascade_infer.rs:313-366` and `:411-439`): identical.
- `cfg_combine(cond, uncond, cfg) = uncond + cfg * (cond - uncond)`
  (`cascade.rs:606-616` vs `cascade_infer.rs:195-201`) ✓
- `scheduler.step_eps_ddim(&v, r, r_next, &x)` argument order matches
  scheduler signature `step_eps_ddim(eps, t, t_next, sample)`
  (`ddpm_wuerstchen.rs:102`) ✓
- The DDIM formula recovers `x0 = (x_t − sqrt(1−a)·eps) / sqrt(a)` then
  forward-noises to `t_next`. Matches the bin's choice ("DDIM avoids the
  noise-injection blow-up" — bin comment line 357-361).
- CFG threshold check `if cfg_c > 1.0` skips the uncond pass when CFG is
  effectively off. Bin does the same.

**FP32 vs BF16 mixing** (SDXL/SD 1.5): the workers keep the denoising state
in FP32 and only convert to BF16 at UNet input. Confirmed against bins
(`sdxl_infer.rs:155-156` and `sd15_infer.rs:215-216`). Cascade keeps state
in BF16 throughout — same as bin (no FP32 conversion in either). The
ε-pred VE workers genuinely benefit from FP32 state because the small
cond/uncond diff would round to zero in BF16 at sigma_min; the Wuerstchen
DDIM step doesn't have the same round-off vulnerability because its arithmetic
is dominated by O(1) coefficients (sqrt(a), sqrt(1−a)).

---

## Schedule + VAE scale audit

**SDXL `build_sdxl_schedule`** (`sdxl.rs:536-574` vs `sdxl_infer.rs:29-66`):
byte-for-byte identical.
- `num_train_steps = 1000`, `beta_start = 0.00085`, `beta_end = 0.012` ✓
- Scaled-linear: `beta[i] = (sqrt(start) + (sqrt(end)-sqrt(start)) * i/(N-1))^2` ✓
- α_cumprod via running product of `(1 - beta)` ✓
- σ = sqrt((1-α)/α) ✓
- Leading spacing with `steps_offset=1`: `ts = (0..N).map(|i| i*step_ratio + 1).reverse()` ✓
- Final sigma = 0.0 appended ✓

**SD 1.5 `build_sd15_schedule`** (`sd15.rs:568-604` vs `sd15_infer.rs:67-102`):
byte-for-byte identical to SDXL's. (Same scaled-linear β, same training-step
count — SD 1.5 and SDXL share their VE schedule.)

**Cascade scheduler** (`DDPMWuerstchenScheduler::new(steps)` from
`ddpm_wuerstchen.rs:49`): cosine schedule with `s = 0.008`, `scaler = 1.0`,
`init_alpha_cumprod = cos(s/(1+s) * π/2)^2`. timesteps = linspace(1.0, 0.0,
steps+1). Worker calls it correctly for both stages.

**VAE scale factors**:
- SDXL `VAE_SCALE = 0.13025`, `VAE_SHIFT = 0.0` (`sdxl.rs:141-142`) ✓
- SD 1.5 `VAE_SCALE = 0.18215`, `VAE_SHIFT = 0.0` (`sd15.rs:126-127`) ✓
- Both workers pass `VAE_IN_CHANNELS = 4` to `LdmVAEDecoder::from_safetensors`.
- Cascade uses Paella VQ-GAN (`PaellaVQDecoder::load`); no scale factor —
  VQ-GAN has its own internal codebook scaling. Output is in [0, 1] range,
  not [-1, 1] like the LDM family. The dedicated `paella_to_color_image`
  helper at `cascade.rs:671-698` clamps `[0, 1]` and multiplies by 255
  (the LDM-family `decoded_to_color_image` clamps `[-1, 1]` and uses
  `(v+1)*127.5`). ✓

---

## SDXL embeddings fallback + SD 1.5 rename workaround

### SDXL embeddings fallback path resolution

`sdxl.rs:281-290`:
```rust
let embeds_path = EMBEDS_PATHS
    .iter()
    .find(|p| Path::new(p).exists())
    .ok_or_else(|| RunError::Other(format!(
        "SDXL embeddings file not found. Checked:\n  - {}\n\
         Generate one of these via `python3 scripts/cache_sdxl_embeddings.py` first.",
        EMBEDS_PATHS.join("\n  - ")
    )))?;
```

- Check order: bare → `.py` → `.rust` (`sdxl.rs:132-136`). Matches qwenimage's
  pattern (`qwenimage.rs:119` + `:297`). ✓
- `Path::new(p).exists()` is the right check (would not be tricked by a
  symlink-to-nowhere; would correctly find a regular file).
- Error message lists all three candidates and points at the script. ✓
- Verified on disk: bare path EXISTS; `.py` and `.rust` variants are absent
  (which is correct — they're optional fallbacks).
- Empty-prompt warning: only fires if `!job.prompt.is_empty()` (see P2.5).

**Verdict**: implementation is correct and matches the qwenimage convention.

### SD 1.5 attention key rename

`sd15.rs:478-519`:
- Loads raw safetensors (full file, encoder + decoder + post_quant).
- Filters to keys starting with `decoder.` / `first_stage_model.decoder.`,
  plus the four `post_quant_conv.{weight,bias}` variants (with and without
  the `first_stage_model.` prefix). ✓
- Strips `first_stage_model.` prefix if present.
- Applies four legacy → modern attention renames:
  - `attentions.0.query.` → `attentions.0.to_q.`
  - `attentions.0.key.`   → `attentions.0.to_k.`
  - `attentions.0.value.` → `attentions.0.to_v.`
  - `attentions.0.proj_attn.` → `attentions.0.to_out.0.`
- Rename map matches `sd15_infer.rs:280-284` exactly.
- BF16 cast applied to each tensor.
- Writes to `/tmp/inference_ui_sd15_vae_remapped.safetensors`, then re-loads
  via `LdmVAEDecoder::from_safetensors`.

The HF SD 1.5 VAE only has one attention block per self-attention site
(`attentions.0.*`), so no other index will collide. There is no
`attentions.1.query` — verified by inspecting the HF VAE schema.

Temp-file hygiene: see P2.1.

---

## Cascade two-stage audit

- **Stage drop-and-reload sequence**:
  - CLIP-G loaded → cond+uncond encoded → CLIP dropped → pool flushed (`cascade.rs:347-373`) ✓
  - Stage C loaded inside `let stage_c_latent = { ... }` scope → unet_c dropped at scope end → pool flushed (`cascade.rs:395-466`) ✓
  - Stage B loaded inside `let stage_b_latent = { ... }` scope → unet_b dropped at scope end (`cascade.rs:479-550`) ✓
  - All four CLIP-G context tensors (pos_hidden, pos_pooled, neg_hidden, neg_pooled) AND stage_c_latent dropped before Stage A loads (`cascade.rs:554-561`) ✓
  - Stage A (Paella) loaded → decoded → dropped (`cascade.rs:579-588`) ✓
- **Single Started event**: emitted ONCE with `total_steps = steps_c + steps_b` (`cascade.rs:202-207`). ✓
- **Monotonic Progress**: Stage C emits `step_global = step+1` (range `1..=steps_c`); Stage B emits `step_global = steps_c + step + 1` (range `steps_c+1..=total`). No regression at boundary. ✓
- **CFG threshold guard**: both stages do `if cfg > 1.0 { do uncond pass } else { use v_cond directly }` matching the bin's behavior. The cascade defaults `cfg_b = 1.1` is just barely above 1, so both passes fire by default; if a user sets cfg=1.0 the worker correctly skips the uncond pass.
- **`step_eps_ddim` argument order**: `(&v, r, r_next, &x)` matches scheduler signature `step_eps_ddim(eps, t, t_next, sample)`. ✓
- **Cancel-check in BOTH stages**: `drain_pending(...)?` at the top of each step in both loops (lines 413 and 497). ✓
- **Drop tracking**: I noticed the worker captures `let stage_c_latent_dims = stage_c_latent.shape().dims().to_vec();` BEFORE dropping (line 558) — this is to keep the log message printable after drop. Defensive and correct.

---

## ModelKind dispatch ordering

`mod.rs:110-175`. Walked the match against representative IMAGE_MODELS
strings:

| Input string | Expected | Resolved | Notes |
|---|---|---|---|
| `z-image-base.safetensors` | ZImageBase | ZImageBase | ✓ |
| `z-image-turbo.safetensors` | ZImageTurbo | ZImageTurbo | ✓ (turbo arm checked before base) |
| `flux1-dev.safetensors` | FluxDev | FluxDev | ✓ |
| `chroma.safetensors` | Chroma | Chroma | ✓ |
| `klein-4b.safetensors` | Klein4B | Klein4B | ✓ |
| `klein-9b.safetensors` | Klein9B | Klein9B | ✓ (9b arm checked before bare klein) |
| `sd3.5-large.safetensors` | Sd35 | Sd35 | ✓ |
| `sd3.5-medium.safetensors` | Sd35 | Sd35 | ✓ |
| `sdxl-base-1.0.safetensors` | Sdxl | Sdxl | ✓ |
| `sdxl-turbo.safetensors` | Sdxl | Sdxl | ✓ (Turbo schedule not wired — known limitation) |
| `qwen-image.safetensors` | QwenImage | QwenImage | ✓ |
| `ernie-image-8b.safetensors` | ErnieImage | ErnieImage | ✓ |
| `anima-2b.safetensors` | Anima | Anima | ✓ |
| `stable-cascade.safetensors` | Cascade | Cascade | ✓ |
| `sd15.safetensors` | Sd15 | Sd15 | ✓ |
| `pony-diffusion-v6.safetensors` | (Sdxl?) | Mock | See P2.4 |

**SD3 BEFORE SDXL** ordering is correct — `sd3.5-medium.safetensors` contains
`sd3` and `medium` (no `xl`), so it correctly hits the SD3 arm before
falling through to SDXL.

**Cascade BEFORE SD15** ordering: cascade is matched before sd15. The
inputs are disjoint (`stable-cascade.safetensors` vs `sd15.safetensors`),
so order doesn't matter for these specific strings, but the comment block
in `mod.rs:159-165` accurately describes the rationale.

**Wuerstchen variant matching** (`mod.rs:160`): both `wurstchen` (no diacritic)
and `würstchen` (with diacritic) match. Defensive — fine.

---

## Hardcoded path existence

Verified on disk (16 paths checked):

| Path | Status |
|---|---|
| `/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors` | ✓ |
| `/home/alex/EriDiffusion/Models/checkpoints/sd_xl_base_1.0.safetensors` | ✓ |
| `…/inference-flame/output/sdxl_embeddings.safetensors` | ✓ |
| `…/inference-flame/output/sdxl_embeddings.py.safetensors` | absent (optional fallback) |
| `…/inference-flame/output/sdxl_embeddings.rust.safetensors` | absent (optional fallback) |
| `…/snapshots/451f4fe…/unet/diffusion_pytorch_model.safetensors` | ✓ |
| `…/snapshots/451f4fe…/vae/diffusion_pytorch_model.safetensors` | ✓ |
| `…/snapshots/451f4fe…/text_encoder/model.safetensors` | ✓ |
| `/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json` | ✓ |
| `…/snapshots/a89f66d…/text_encoder/model.bf16.safetensors` | ✓ |
| `…/snapshots/a89f66d…/tokenizer/tokenizer.json` | ✓ |
| `…/snapshots/a89f66d…/stage_c_bf16.safetensors` | ✓ |
| `…/snapshots/a89f66d…/stage_b_bf16.safetensors` | ✓ |
| `…/snapshots/a89f66d…/stage_a.safetensors` | ✓ |

The two SDXL embeddings fallback paths are intentionally absent — they're
secondary candidates the user-pre-computation script may write to. The
worker's first-existing-wins logic correctly picks the bare canonical name.

---

## AGENT-DEFAULT assessment (10 substantive items, brief said 8)

I found 10 distinct AGENT-DEFAULT decisions across the three files (some
are repeated docstring/code mentions of the same decision; below I deduplicate).

1. **SDXL/SD 1.5 use `Tensor::randn_seeded` instead of bin's `StdRng + Box-Muller`**
   (`sdxl.rs:366`, `sd15.rs:327`). Statistical distribution matches
   (Box-Muller per docstring); bit-exact bytes do not. Same seed will
   produce different images vs the bin. **Acceptable.** Cascade is
   bit-exact because it uses `flame_core::rng::set_seed` + `randn_bf16`
   identically to its bin.

2. **SDXL: cached-embeddings convention with three-path fallback and
   ignored typed prompt** (`sdxl.rs:23`, `:118`). Matches qwenimage and
   anima. Workaround for missing CLIP-G `text_projection` wiring in Rust.
   **Acceptable; flag for future Rust dual-encoder port.**

3. **SDXL/SD 1.5/Cascade hardcoded weight paths** (`sdxl.rs:105`,
   `sd15.rs` paths, `cascade.rs:115`). Mirrors flux/klein convention.
   **Acceptable** for a personal-machine UI; a multi-user deployment
   would want config/env-var override.

4. **SD 1.5 typed prompt is the actual prompt (no cache)** (`sd15.rs:23`).
   Inverse of SDXL convention because CLIP-L is small and Rust has the
   encoder. **Correct decision.**

5. **Cascade typed prompt is the actual prompt** (`cascade.rs:23`). CLIP-G
   already has a working Rust encoder (used by other workers). **Correct.**

6. **Cascade single CFG slider drives both `cfg_c` and `cfg_b`**
   (`cascade.rs:33`). Bin allows independent `--cfg-c` / `--cfg-b`. Default
   bin values (4.0 / 1.1) are very different — driving both at the same
   value is a real behavior change. With UI cfg=4.0, both stages run at
   4.0. With UI cfg=1.1, both run at 1.1 (Stage C nearly disabled).
   **Acceptable but worth tracking** — if the UI ever adds a "Stage B
   CFG" slider, this is the place to wire it. The fallback to per-stage
   defaults when `job.cfg <= 0.0` is a sensible safety net.

7. **Cascade single Started event with combined `total_steps`** (`cascade.rs:66`).
   UI has one progress bar. **Correct decision** — alternative (two
   Started events) would look like a regression mid-job.

8. **Cascade 2:1 step split when `job.steps != 0`** (`cascade.rs:89`).
   Reasonable proportional mapping. Edge case for very low step counts
   (1-2) inflates to 3 steps total — see P2.2.

9. **Cascade BF16 denoising state (no FP32 conversion)**. Implicit in
   the worker; bin does the same. **Correct** — DDIM step's coefficients
   are O(1), no rounding vulnerability.

10. **VAE temp file at fixed `/tmp/inference_ui_sd15_vae_remapped.safetensors`**
    (`sd15.rs:120`). Implicit AGENT-DEFAULT — bin uses same pattern with
    a different filename. See P2.1 for hygiene concerns.

---

## Unverified

- **End-to-end image quality vs the bins.** Builder reports clean compile;
  I did not run any of the three workers. Bit-exact noise differences
  (item 1 above) mean SDXL/SD 1.5 outputs will differ from the bin's
  reference images even at the same seed. Statistical equivalence is
  expected.
- **VRAM peaks under real workload.** Workers claim 5 GB SDXL UNet and
  7 GB Stage C UNet from the docstrings; not measured this session.
- **Cascade Stage A peak during VQ-GAN decode.** The `paella_to_color_image`
  pulls the entire 1024×1024×3 BF16 tensor to host then expands to F32 for
  clamping. ~6 MB host allocation — trivial — but I didn't trace whether
  the GPU-side intermediate is freed before the host download completes.
- **SD 1.5 attention rename completeness.** I confirmed the four standard
  legacy → modern renames are present and match the bin. I did not enumerate
  every attention key in the actual HF VAE safetensors file to verify there
  are no extra naming variants the workaround might miss.
- **The `step_eps_ddim` is correct for the Wuerstchen UNet.** Soul.md
  notes session 10's debug found the eps treatment works post-fix
  (`cos_sim 0.9999+ on both stages`), but I did not re-run the bin's
  parity test against this worker's call sequence.
