# Phase 2 — Skeptic Review

Scope: `/home/alex/EriDiffusion/inference_ui/src/{sections,widgets,panels.rs,state.rs}` reviewed against `/tmp/flame_ui_design/design_handoff_flame_ui/README.md` lines 52–104, 161–208, 277–290, plus `components/params-panel.jsx` for layout details.

Builder reports `cargo build` passes; not re-run. Code reviewed line-by-line.

---

## P0 — correctness/spec deviations that must fix

### P0 #1 — Sampler list ordering deviates from spec
**File:** `src/sections/sampling.rs:14-25`
**Issue:** Spec line 76 explicitly orders the sampler list as: `Euler, Euler a, DPM++ 2M, DPM++ 2M SDE, DPM++ 3M SDE, UniPC, LCM, DDIM, Heun, LMS`. Code matches verbatim. ✅ **No defect — moved to "Spec Compliance Table" below.** This entry retracted.

(Initial scan flagged this; re-reading code at sampling.rs:14-25 confirms `["Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ 3M SDE","UniPC","LCM","DDIM","Heun","LMS"]` — exact match. No P0 found here.)

**Result:** No actual P0 issues. Moving everything down a tier.

---

## P1 — likely bugs / spec deviations

### P1 #1 — Random-seed range is 100× narrower than JSX intent
**File:** `src/sections/seed.rs:21`
**Issue:** `rng.gen_range(0..1_000_000_000)` — generates seeds in `[0, 1e9)`. JSX prototype (`params-panel.jsx:145`) uses `Math.floor(Math.random()*9e9)`, i.e. `[0, 9e9)`. The field range itself is `-1..=9_999_999_999` (10 digits). The 🎲 button thus only ever produces values up to 10 digits truncated to ~9 digits — visually inconsistent with what the user can manually type.
**Severity:** Minor functional but visible — a seed-roll user expects to occasionally see ~10-digit seeds.
**Fix shape:** widen to `0..10_000_000_000_i64` (or match JSX `0..9_000_000_000`).

### P1 #2 — `/` separator between Steps and CFG is not centered/aligned per spec
**File:** `src/sections/sampling.rs:42-44`
**Issue:** Spec line 75 says "two `DragValue`s on one row with a `/` separator". Code uses `ui.label(RichText::new("/").size(11.0).color(t.text_mute))` between two 56px DragValues. Visually fine, but the surrounding `labeled_row` allocates 80px for the label + remaining width for body — Steps DragValue + `/` + CFG DragValue may not center the slash relative to the row baseline since `ui.label` height differs from `DragValue` height (22px). Worth visual verification.
**Severity:** Cosmetic; only flag-worthy if visual misalign is obvious.

### P1 #3 — Manual W/H edits never sync `resolution_preset` field back to "Custom"
**File:** `src/sections/resolution.rs:62-74`
**Issue:** Builder added `"Custom"` as the last entry in `IMAGE_PRESETS` / `VIDEO_PRESETS`, but the Size DragValues at lines 63 + 65 mutate `cn.width` / `cn.height` without touching `cn.resolution_preset`. After a manual edit, the preset dropdown still displays the previously-selected preset string (e.g. `"1024×1024  ·  1:1"`) even when W/H no longer match. JSX behavior at lines 105-110 also doesn't sync the preset back, so this is a JSX-faithful behavior, but the addition of a `"Custom"` entry implies the builder intended to switch to it on manual edit. As-is, `"Custom"` is selectable but never auto-engages.
**Severity:** Inconsistent UX; either remove `"Custom"` or wire the auto-switch.

### P1 #4 — Advanced footer counter says "+4 sections" but spec text says "three"
**File:** `src/panels.rs:120-126`
**Issue:** Footer label reads `"+4 sections"`. README line 99 prose says "When checked, **three** extra collapsibles appear above the footer", but README lines 100-104 then enumerate **four** (ControlNet/img2img, Advanced sampling, Performance, Output). The JSX footer at `params-panel.jsx:242` says `"+3 sections"` — matching the prose count, but the JSX itself renders 4 advanced sections (`params-panel.jsx:200,208,216` + the `ControlNet/img2img` at line 180 which isn't gated by `advanced` in JSX). This is a spec-internal inconsistency. Code chose to enumerate 4, which matches the README's structural list. **Acceptable** — flag to spec-author as confusion, not a code defect.
**Severity:** Spec inconsistency; pick one.

### P1 #5 — `ControlNet/img2img` section content is mode-agnostic but spec splits behavior
**File:** `src/sections/advanced.rs:35-94`
**Issue:** README line 101 says "For Video + I2V, this is where the first-frame image goes." JSX line 180 changes the section title to `"Input (img2vid)"` when `mode === 'video'` and conditionally hides the ControlNet picker (`mode !== 'video'` at line 192). Rust code uses static title `"ControlNet / img2img"` regardless of mode and always shows the ControlNet picker. So in Video mode the user sees a confusingly-named section with an irrelevant `Model` picker.
**Severity:** Cosmetic but real — wrong label in Video tab.

### P1 #6 — `flat_collapsing` chevron-background and indent NOT removed
**File:** `src/widgets/section_header.rs:1-39`
**Issue:** Spec line 56 demands "**flat** (no chevron background, no indent)". The wrapper acknowledges (lines 8-11) that egui 0.29 can't suppress those via public API and ships the stock `CollapsingHeader`. Builder flagged this as AGENT-DEFAULT #7. Visually each section will have a hover-highlighted chevron triangle and a left indent on the body — both spec deviations. Acceptable phase trade-off, but the resulting visuals will diverge noticeably from the JSX mock.
**Severity:** Visible spec deviation, called out by builder. Defer if pixel-fidelity not blocking.

### P1 #7 — LoRA section bypasses `flat_collapsing` and re-implements styling
**File:** `src/sections/lora.rs:33-41`
**Issue:** All other sections use `flat_collapsing(...)`, which applies `.to_uppercase()` and the spec'd font/color. LoRA section calls `CollapsingHeader::new(title)` directly (line 38) with a hand-built `RichText`. The hand-built title format is `"LORA STACK   {counter}"` — three spaces between label and counter. This works but bypasses the wrapper indirection, so any future styling fix to `flat_collapsing` won't propagate to LoRA. Also, the counter is rendered as part of the title text (left-aligned with the title), not right-aligned per spec. Builder flagged this as AGENT-DEFAULT #3.
**Severity:** Acknowledged AGENT-DEFAULT, acceptable.

### P1 #8 — Ghost button used as icon-only browse/open-folder, but `ghost_button` has no icon variant
**File:** `src/sections/advanced.rs:159,163`
**Issue:** `ghost_button(ui, t, "📂")` and `ghost_button(ui, t, "↗")`. The `ghost_button` wrapper at `widgets/ghost_button.rs:10-14` has `min_size(0.0, 22.0)` — but no max-width clamp, and emojis are wide. The two buttons sit on the same row as a TextEdit that consumes `available_width() - 80.0` (advanced.rs:153). The 80px reserved buffer may not fit two emoji buttons + their padding on narrower window sizes. Worth a visual check at `min_width(240.0)` (the params panel's lower bound).
**Severity:** Possible layout overflow at min width.

### P1 #9 — Output Folder TextEdit has no `available_width().max(80.0)` floor when only `80.0` is reserved
**File:** `src/sections/advanced.rs:153-155`
**Issue:** `let avail = ui.available_width() - 80.0;` followed by `avail.max(80.0)`. At narrow widths, `available_width()` could be e.g. 90px, giving `avail = 10` then `max(80) = 80` — pushing the buttons off the row. This is mostly defensive and works in practice, but the `80.0` magic number is unexplained (presumably "two ghost buttons + spacing").
**Severity:** Minor robustness, no immediate bug.

### P1 #10 — `seed_locked` toggle changes glyph (🔒/🔓) but spec says always 🔒
**File:** `src/sections/seed.rs:26`
**Issue:** Spec line 81 says the icon is `🔒 lock` — implying the lock icon is always shown, with state communicated via active/pressed styling, not glyph swap. Code swaps to 🔓 when unlocked. The glyph swap is a clearer affordance but deviates from spec. Cosmetic.
**Severity:** Cosmetic; arguably better UX than spec.

---

## P2 — minor / acceptable

### P2 #1 — `combo_str` width clamps at `min(220.0)`
**File:** `src/widgets/combo.rs:18,45`
**Issue:** ComboBox width = `ui.available_width().min(220.0)`. At wide column widths, dropdowns cap at 220px — which is fine since the params column itself caps at 480px. No spec violation.

### P2 #2 — `labeled_row` body width allocation is implicit
**File:** `src/sections/model.rs:106-125`
**Issue:** Label gets fixed 80px allocation; body inherits whatever `ui.horizontal` allots. No spec issue but worth noting if layout misbehaves.

### P2 #3 — `Frame::none()` deprecated in newer egui
**File:** Multiple sections (model.rs:88, lora.rs:74, resolution.rs implicit, advanced.rs:46)
**Issue:** `Frame::none()` works in egui 0.29 (Cargo.toml) but is deprecated in 0.30+. Future upgrade churn.

### P2 #4 — Stub size on Model info strip is hardcoded by mode, not by selected model
**File:** `src/sections/model.rs:83-87`
**Issue:** Always shows `23.8 GB` for Image and `28.2 GB` for Video regardless of which `Base` model is selected. Spec line 65 says "Size comes from file metadata" — explicitly out of scope per Phase 2 brief. Acceptable.

### P2 #5 — Image preset list contains 9 entries (8 presets + Custom), Video has 7 (6 + Custom)
**File:** `src/sections/resolution.rs:14-34`
**Issue:** JSX has 8 image presets and 6 video. Code adds `"Custom"` as 9th/7th. Acceptable per builder note in P1 #3.

### P2 #6 — `lock` button doesn't actually prevent advancing in Phase 2
**File:** `src/sections/seed.rs:23-37`
**Issue:** Builder noted "actual 'prevents advancing' behavior wires up when the worker pulls a seed". Phase 2 just persists the bit. ✅ Acceptable per scope.

### P2 #7 — `model.rs:124` `out.unwrap()` is safe but reads dangerously
**File:** `src/sections/model.rs:124`
**Issue:** `out` is initialized to `None`, then assigned inside `ui.horizontal(|ui| { out = Some(body(ui)); })`. The closure runs synchronously in egui (immediate-mode), so `out` is always `Some` by the unwrap. Could be expressed without `Option` via `let mut out = std::mem::MaybeUninit` or just by structuring differently, but the current form is idiomatic egui.

### P2 #8 — `ScrollArea` uses `auto_shrink([false; 2])` — fills column even when empty
**File:** `src/panels.rs:67`
**Issue:** Correct for filling vertical space below the footer. ✅

### P2 #9 — `seed_mode` doesn't auto-switch to `Fixed` when user clicks 🎲 or types a seed
**File:** `src/sections/seed.rs`
**Issue:** Spec doesn't pin behavior. JSX doesn't either. Acceptable.

### P2 #10 — Image preset uses Unicode `×` (U+00D7), Video preset uses Unicode `×`. Default state uses `×`. Parser handles both `×` and lowercase `x`. ✅

### P2 #11 — `image_default()` includes `loras: vec![...]` with 3 entries; `video_default()` has empty Vec
**File:** `src/state.rs:279-283, 306`
**Issue:** Spec doesn't pin LoRA defaults. Image gets 3 mock entries (matching JSX MOCK_LORAS); Video gets none. Acceptable mock behavior. The LoRA stack header counter in Video tab will show `0 / 0`.

### P2 #12 — `LoraSlot::default()` produces `name: "new-lora.safetensors"` placeholder
**File:** `src/state.rs:151-163`
**Issue:** Spec line 96 says "+ Add LoRA" opens a file picker; placeholder name is fine until pickers wire up.

### P2 #13 — `Frames` and `FPS` DragValue widths default to `60.0` (not in spec)
**File:** `src/sections/resolution.rs:85,95`
**Issue:** Spec doesn't pin Frames/FPS widths. JSX uses 60. ✅

### P2 #14 — `Batch Count` and `Batch Size` DragValue widths default to `60.0` (not in spec)
**File:** `src/sections/batch.rs:16,19`
**Issue:** Spec doesn't pin. Acceptable.

### P2 #15 — VRAM budget spinner range `1..=192` GB (not in spec)
**File:** `src/sections/advanced.rs:139`
**Issue:** Spec line 103 says "VRAM budget DragValue" without range. 192 GB upper bound is generous; reasonable.

---

## Per-mode state separation audit

**Verdict: PASSES.** No leaks found. All per-mode access goes through `state.current_mut()` / `state.current()`.

| Section | Per-mode? | Access pattern | Verdict |
|---------|-----------|----------------|---------|
| Model (Task/Base/VAE/Precision) | yes | `state.current_mut()` (model.rs:42) | ✅ |
| Resolution (preset/W/H) | yes | `state.current_mut()` (resolution.rs:42) | ✅ |
| Frames/FPS | yes (Video only) | `cn.frames` / `cn.fps` (resolution.rs:81-99) | ✅ |
| Sampling (sampler/scheduler/steps/cfg) | yes | `state.current_mut()` (sampling.rs:38) | ✅ |
| Seed (value + locked + mode) | **shared** (on AppState) | `state.seed` etc. (seed.rs:17,21) | ✅ matches spec lines 178-180 |
| Batch (count/size) | **shared** (on AppState) | `state.batch_count` (batch.rs:16,19) | ✅ matches spec lines 177-178 |
| LoRA stack | yes | `state.current_mut().loras` (lora.rs:19,56,63,85) | ✅ |
| ControlNet | yes | `state.current_mut().controlnet` (advanced.rs:42-92) | ✅ matches spec line 201 |
| Advanced sampling | yes | `state.current_mut().advanced_sampling` (advanced.rs:100) | ✅ |
| Performance | yes | `state.current_mut().perf` (advanced.rs:124) | ✅ |
| Output | yes | `state.current_mut().output` (advanced.rs:148) | ✅ |

**Mental tab-switch test:** Set Image=2048×1152 → switch to Video. Image's `width=2048, height=1152` lives in `state.image.width/height`; switching tab reads `state.video.width/height`. Independent. ✅

**Frames/FPS rendering:** `if matches!(mode, Mode::Video) { ... }` at resolution.rs:78. Image tab never renders these rows. ✅

**Image vs Video preset distinct:** `IMAGE_PRESETS` (8 presets + Custom) ≠ `VIDEO_PRESETS` (6 + Custom). Defaults: Image `1024×1024  ·  1:1`, Video `1280×720  ·  16:9`. ✅

---

## ComboBox / DragValue spec compliance table

### ComboBoxes

| Field | Spec items | Code items | Order match | File:line |
|---|---|---|---|---|
| Sampler | 10 | 10 | ✅ exact | sampling.rs:14-25 |
| Scheduler | 7 | 7 | ✅ exact | sampling.rs:27-35 |
| Precision | 6 | 6 | ✅ exact | state.rs:104-113 |
| ControlNet model | 6 (none, canny, depth, pose, tile, lineart) | 6 | ✅ exact | state.rs:193-202 |
| Task (Image) | 3 (T2I, I2I, IC-LoRA) | 3 | ✅ exact | state.rs:67-69 |
| Task (Video) | 4 (T2V, I2V, A2V, IC-LoRA) | 4 | ✅ exact | state.rs:71-73 |
| Attention backend | 4 (flash-attn-2, sdpa, xformers, math) | 4 | ✅ exact | advanced.rs:21 |
| CPU offload | 4 (none, cpu, sequential, model) | 4 | ✅ exact | advanced.rs:22 |
| Seed mode | 3 (random, fixed, increment) | 3 | ✅ exact | state.rs:138-140 |

### DragValues

| Field | Spec range/step/width | Code range/step/width | File:line | Verdict |
|---|---|---|---|---|
| Width | 64..=4096 step 8 width 60 | 64..=4096 step 8 width 60 | resolution.rs:63 | ✅ |
| Height | 64..=4096 step 8 width 60 | 64..=4096 step 8 width 60 | resolution.rs:65 | ✅ |
| Steps | 1..=150 int width 56 | 1..=150 step 1.0 width 56 | sampling.rs:42 | ✅ |
| CFG | 1..=30 step 0.1 width 56 | 1.0..=30.0 step 0.1 width 56 | sampling.rs:44 | ✅ |
| Frames | 8..=241 step 1 (no width) | 8..=241 step 1.0 width 60 | resolution.rs:85 | ✅ |
| FPS | 6..=60 step 1 (no width) | 6..=60 step 1.0 width 60 | resolution.rs:95 | ✅ |
| Seed | -1..=9_999_999_999 width 90 | -1..=9_999_999_999 width 90 | seed.rs:17 | ✅ |
| Batch count | 1..=64 (no width) | 1..=64 width 60 | batch.rs:16 | ✅ |
| Batch size | 1..=8 + tooltip (no width) | 1..=8 width 60 + tooltip | batch.rs:19 | ✅ |
| LoRA strength | 0.00..=2.00 step 0.05 width 56 | 0.0..=2.0 step 0.05 width 56 dec=2 | lora.rs:118 | ✅ |
| ControlNet strength | 0..=1.0 (no width/step) | 0.0..=1.0 step 0.01 width 60 dec=2 | advanced.rs:79 | ✅ |
| ControlNet denoise | 0..=1.0 (no width/step) | 0.0..=1.0 step 0.01 width 60 dec=2 | advanced.rs:82 | ✅ |

**Spec compliance: 100% on items, ranges, and explicit widths.** No deviations.

---

## AGENT-DEFAULT decisions — assessment of each

(Per the brief, builder flagged 11 AGENT-DEFAULT decisions. The Rust code itself only annotates a subset by name. Numbering is best-effort matched against the brief.)

| # | Decision | Assessment | Rationale |
|---|----------|------------|-----------|
| 1 | `seed_locked` placed on `AppState` (shared, not per-mode) | **Acceptable** | Spec puts `seed` on AppState; lock should follow seed's home. Per-mode would require lock-twice. |
| 2 | `LoraSlot::default().active = true` (override `derive(Default)`) | **Acceptable** | Matches JSX `on: true` for fresh-added entries. Required correctness. |
| 3 | LoRA header counter rendered as title suffix, not right-aligned badge | **Acceptable / Defer** | egui CollapsingHeader doesn't support right-side header chrome. Would need a fully custom widget. Cosmetic divergence. |
| 4 | LoRA × button always visible vs hover-only | **Acceptable** | Same widget-API constraint. Mild discoverability loss. |
| 5 | Drag handle (≡) is inert (no `egui-dnd`) | **Acceptable** | Brief explicitly defers reorder to later phase. Cargo.toml comment confirms. |
| 6 | Video preset list rebuilt rather than copied JSX | **Acceptable** | List matches JSX `VIDEO_RES`; spec doesn't pin. |
| 7 | `flat_collapsing` ships stock CollapsingHeader (chevron + indent visible) | **Defer / Real concern** | Visible spec deviation (line 56). If pixel-fidelity matters, schedule a custom widget; otherwise accept. |
| 8 | `+ Add LoRA` stub appends `LoraSlot::default()` with placeholder name | **Acceptable** | File picker out of scope for Phase 2. |
| 9 | ControlNet section title doesn't change for Video mode | **P1 fix** (see P1 #5) | JSX changes to "Input (img2vid)". Trivial code change; should fix in Phase 2. |
| 10 | Browse/open-folder buttons stubbed with no `rfd` integration | **Acceptable** | `rfd` explicitly out of scope per brief. |
| 11 | Random seed range narrower than JSX (`0..1e9` vs `0..9e9`) | **P1 fix** (see P1 #1) | Trivial constant change, observable to user. |

**Summary:** 8 acceptable, 1 defer (flat collapsing), 2 trivial Phase-2 fixes (#9 ControlNet title in Video mode, #11 random seed range).

---

## Scope creep audit

Out-of-scope per brief, checked against code:

| Out-of-scope item | Snuck in? | Evidence |
|---|---|---|
| Real disk scan for Base/VAE | ❌ no | Hardcoded const arrays at model.rs:16-38 |
| Real file pickers (`rfd`) | ❌ no | All `Browse…` actions are stubs (advanced.rs:64-66, 161, 163) |
| LoRA drag-reorder | ❌ no | Drag handle painted but inert (lora.rs:83 comment) |
| Image preview rendering for ControlNet dropzone | ❌ no | Just an icon + text label (advanced.rs:51-60) |
| Filename template variable expansion | ❌ no | Hint label only (advanced.rs:182-188) |
| ✨ Enhance / Template buttons | ❌ no | Lives in canvas panel header (Phase 3) |
| Generate button | ❌ no | Phase 3 |
| Persistence | ❌ no | `state.rs` has `Serialize/Deserialize` derives but no save/load wiring |
| `egui-dnd` dependency | ❌ no | Cargo.toml comment confirms it's deferred |
| `rfd` dependency | ❌ no | Not in Cargo.toml |
| `image` crate for previews | ❌ no | Not in Cargo.toml |

**No scope creep detected.** Builder respected the brief boundaries.

---

## Unverified

- **Compilation hygiene:** `cargo build` not re-run by reviewer. Builder claimed pass. Phase 1 had 3 acknowledged warnings; not re-counted.
- **Visual fidelity:** No screenshot comparison performed. Flat-collapsing visual divergence (P1 #6) and `/` separator alignment (P1 #2) need eyeballs.
- **Layout overflow at min width 240px:** P1 #8 / P1 #9 are conjecture; never tested with the panel narrowed.
- **`labeled_row` borrow correctness across all sections:** Spot-checked sampling.rs and seed.rs; `state.current_mut()` taken once at top of each `show()` and reused. No double-borrows observed in scan, but full audit would require typechecking against the egui 0.29 closure API.
- **`flat_collapsing` `body_returned` value discarded:** `flat_collapsing` returns `Option<R>` (closing-state-aware) but every caller ignores the return. Functionally fine (no caller cares whether the section is open), but means the `Option` wrapper is currently useless.
- **`labeled_row` height of 22px:** Hardcoded matches DragValue height; `ComboBox` rows may be slightly taller, causing inter-row gap inconsistencies. Visual check needed.

---

## Summary

**0 P0, 10 P1, 15 P2.**

**Key issues:**
1. **P1 #5 — ControlNet section title doesn't change to "Input (img2vid)" in Video mode** (`advanced.rs:36`); JSX-spec deviation; trivial fix.
2. **P1 #1 — Random seed range is `0..1e9` not `0..9e9`** (`seed.rs:21`); 100× narrower than JSX intent; trivial fix.
3. **P1 #3 — Manual W/H edits don't auto-switch preset to "Custom"** (`resolution.rs:62-74`); inconsistent with the added `"Custom"` entry; either remove the entry or wire the auto-switch.
4. **P1 #6 — `flat_collapsing` ships stock CollapsingHeader** (`widgets/section_header.rs`); visible spec deviation per README line 56 ("flat — no chevron background, no indent"). Acknowledged AGENT-DEFAULT; defer or schedule custom widget work.
5. **P1 #4 — Spec internal contradiction** ("three" vs four advanced sections); footer says "+4 sections", JSX says "+3". Code chose "+4" matching the structural list. Spec-side fix recommended.

Per-mode state separation: **clean.** ComboBox/DragValue spec compliance: **100% on items + ranges + explicit widths.** Scope creep: **none detected.** AGENT-DEFAULT decisions: 8 acceptable, 1 defer, 2 trivially fixable.
