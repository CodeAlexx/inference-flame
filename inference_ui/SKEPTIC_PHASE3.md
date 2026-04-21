# Phase 3 Skeptic Review — Canvas Panel

Code reviewed:
- `inference_ui/src/sections/prompt.rs` (243 LoC)
- `inference_ui/src/sections/action_bar.rs` (95 LoC)
- `inference_ui/src/sections/canvas.rs` (280 LoC)
- `inference_ui/src/state.rs` (additions: `generating`, `current_step`, `total_steps`)
- `inference_ui/src/tokens.rs` (additions: CHECKER_*)
- `inference_ui/src/panels.rs` (canvas_panel wiring)

Spec reference: `/tmp/flame_ui_design/design_handoff_flame_ui/README.md` lines 106-130.
Visual ref: `/tmp/flame_ui_design/design_handoff_flame_ui/components/canvas-panel.jsx`.

---

## P0 — must fix

### P0-1. `CHECKER_DARK` and `CHECKER_LIGHT` names are inverted (dark theme)
File: `tokens.rs:85-86`
```
pub const CHECKER_DARK:  Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2e); // 42,42,46
pub const CHECKER_LIGHT: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x22); // 30,30,34
```
`0x2a > 0x1e`, so `CHECKER_DARK` is *brighter* than `CHECKER_LIGHT`. The names lie. Spec line 124 lists the two hexes in `#2a2a2e / #1e1e22` order without semantic naming, so the values themselves are correct, but `canvas.rs:67-68` then does:
```
let (dark, light) = checker_pair(state.theme);
painter.rect_filled(rect, 0.0, dark);          // base = #2a2a2e (the LIGHTER hue)
...
shapes.push(Shape::rect_filled(tile_rect, 0.0, light));  // overlay = #1e1e22 (the DARKER hue)
```
Reading the code at face value — "fill dark, overlay lighter tiles" — produces the opposite of what runs. This is a maintenance trap, not a visual bug. Either swap the constant *values* or swap the *names*.

(Light-theme constants `CHECKER_DARK_LIGHT_THEME = #c8c8cc` < `CHECKER_LIGHT_LIGHT_THEME = #d8d8dc` are named correctly.)

### P0-2. Header row height is 30px, spec says 38px
File: `prompt.rs:38`
```
vec2(ui.available_width(), 30.0),
```
Comment two lines above (`prompt.rs:34-36`) explicitly references the spec's 38px and claims the allocation pins it — but the value used is 30. The 38px figure in spec line 108 is load-bearing for vertical rhythm; everything below the header row shifts up by 8px.

### P0-3. Toolbar shows `step 0/0` while generating
File: `canvas.rs:239` + `state.rs:357-362` + `action_bar.rs:42-51`
Generate toggles `state.generating = true` but `current_step`/`total_steps` stay 0 (the action bar only resets them to 0 when stopping; nothing ever sets them while running). The toolbar then renders `⏵ step 0/0` in green. This is the user-facing default state of a "running" job in Phase 3 and looks broken.

Either:
- Hide the step counter and show `⏵ sampling…` while `total_steps == 0` (stub-honest), or
- Bump `total_steps = state.current().steps` on click so it shows `step 0/28` (still wrong but at least bounded).

The brief said "Phase 3 stub: clicking toggles `state.generating`" but the toolbar already references the step counts, so they need a non-bogus value.

---

## P1 — likely bugs

### P1-1. Toolbar draws a full border instead of `border-top` only
File: `canvas.rs:223`
```
.stroke(Stroke::new(1.0, t.border))
```
JSX (`canvas-panel.jsx:91`) is `borderTop: \`1px solid ${T.borderSoft}\``. egui `Frame::stroke` strokes all four sides. The toolbar will get a visible left/right/bottom border that doesn't exist in the design.

### P1-2. Token counter rect is computed from outer `Frame` rect, but TextEdit's content area extends to that bottom edge → counter overlaps last line of text
File: `prompt.rs:138-148`
```
let counter_pos = pos2(
    frame_resp.response.rect.right() - 8.0,
    frame_resp.response.rect.bottom() - 12.0,
);
```
The TextEdit was sized with `add_sized([avail.x, box_height - 20.0], edit)` (line 125), which only reserves 20px of the *frame's inner space* for the counter — but the outer `frame_resp.response.rect.bottom()` is the same as the TextEdit bottom plus the frame's rounding, *not* a separate counter gutter. With egui's default `Frame::none().show()` no inner padding is added, so `box_height - 20.0` *inside* a `box_height`-sized frame leaves 20px of unallocated space at the bottom of the frame **only because** `set_min_height(box_height)` runs before the TextEdit. This works empirically but is fragile and undocumented; if anyone bumps `box_height` or adds inner padding, the counter floats inside the text editor area.

### P1-3. Negative prompt id reuse on widget swap may steal/lose focus on the swap frame
File: `prompt.rs:157-213`
The same `Id::new("negative_textedit")` is shared between a `TextEdit::singleline` and a `TextEdit::multiline`. egui's focus tracking is by Id, so in steady state focus persists. But on the frame focus is *first acquired*, the previous singleline reports `has_focus == true` only after the click → the *next* frame swaps to multiline. Because `add_sized` allocates a *new widget instance* with a different layout shape, the multiline can lose the cursor position and the IME state. This is a known egui caveat for "morphing" widgets. Worth verifying interactively (single click → focus → expand → does the cursor land where I clicked?).

### P1-4. Action bar readout always includes frame count when `cn.frames.is_some()`, but Image mode also relies on that
File: `action_bar.rs:76-79`
```
let res = match (cn.frames, cn.fps) {
    (Some(f), _) => format!("{}×{} · {}f", cn.width, cn.height, f),
    _ => format!("{}×{}", cn.width, cn.height),
};
```
This works *today* because `image_default()` (state.rs:287) sets `frames: None`. But the spec wording (line 121) and JSX (line 60) gate the frame suffix on **mode**, not on the *presence* of a frame value. If anyone later sets `image.frames = Some(N)` (e.g. for animated PNG output), the Image readout suddenly grows a `· Nf` segment. Should be `match state.tab` not `match cn.frames`.

### P1-5. Canvas tile count is ~2.5× higher than the builder reported
File: `canvas.rs:74-92`
At a typical canvas of ~1200×800px: cols = ⌈1200/8⌉+1 = 151, rows = ⌈800/8⌉+1 = 101. That's 15,251 cells iterated, ~7,600 light tiles pushed. The builder claimed "~3000 shapes". The actual cost is fine for egui's batching, but the number in the comment is wrong, and at 1920×1200 we're at ~36k iterations / ~18k shapes per frame *every frame the canvas paints* — still cheap, but worth knowing. If the canvas ever needs a tinted overlay or live preview repaint at 60Hz, switch to a tiled texture or two-triangle mesh now rather than later.

### P1-6. `painter.galley(...)` re-paints with `t.text` after `layout_no_wrap` already baked `t.text` into the galley
File: `canvas.rs:200-216` (`paint_badge`)
```
let galley = painter.layout_no_wrap(text.to_string(), font, t.text);
...
painter.galley(pos, galley, t.text);
```
The 3-arg `painter.galley(pos, galley, fallback_color)` only uses the 3rd arg if the galley has no explicit color set. Since `layout_no_wrap` already sets it, the second `t.text` is dead code. Not a bug, just confusing. (Also fine on `prompt.rs:235` for the badge — same pattern.)

### P1-7. Action bar "Stop" doesn't show progress %
File: `action_bar.rs:21-23`
JSX line 50: `Stop · {Math.round(progress*100)}%`. Rust just shows `⏹ Stop`. Phase 3 has no progress source so an honest stub is fine, but the canvas-panel.jsx is the design source of truth and shows it; flag it as a known deviation rather than miss it.

### P1-8. Missing 3px progress bar under action bar
File: `action_bar.rs` (entire file) vs `canvas-panel.jsx:70-74`
The JSX renders a 3px-tall progress bar across the bottom of the action-bar panel while running. Rust has nothing. Same Phase-3-no-progress excuse, but the design shows it as part of the action bar's chrome, not as live data — even an empty 3px strip at `t.panel_sunk` would match the silhouette.

---

## P2 — minor / acceptable

### P2-1. `prompt.rs` PROMPT/NEGATIVE labels use `.monospace()` but spec says sans + uppercase
Spec line 13 (JSX): no monospace; just uppercase, weight 600, letter-spacing. The other labels in Phase 2 sections (e.g. `MODEL`, `RESOLUTION`) presumably do the same — consistency wins. Acceptable, but if Phase 4 cleans up labels, sweep these too.

### P2-2. Prompt box hint text size 13 forced via RichText
File: `prompt.rs:117-121`. egui's `TextEdit::hint_text` already inherits the font; passing a sized `RichText` is redundant. Harmless.

### P2-3. `prompt.rs:175` shadows variable name `neg_stroke`
```
let neg_stroke = if neg_focused { t.accent } else { t.border };
let neg_frame = Frame::none().fill(...).stroke(Stroke::new(width, neg_stroke))
```
Stylistic. Fine.

### P2-4. Toolbar separator uses `t.border` (1px wide × 14px tall solid line)
File: `canvas.rs:275-280`. JSX uses `T.border` width 1, height 14, margin `0 4px` — Rust matches. ✅

### P2-5. Bottom badges use `t.text` (light) on `Color32::from_black_alpha(140)`
Works on dark theme. On *light* theme `t.text = #1b1b1e` (near-black) over a translucent black backdrop → near-invisible. Light-theme not in spec for badges, but expect a visual bug.

### P2-6. `state.generating` lives outside `ModeSettings`
Sensible (a single inference job is global, not per-tab). Not flagged in the brief but worth noting: switching tabs while generating won't change the indicator.

### P2-7. `action_bar.rs:86` joins with `"  ·  "` (two spaces around) vs spec `" · "` (single)
Tiny visual difference; the JSX uses `gap: 10` between flex children plus separator spans. Acceptable.

### P2-8. `tokens.rs` token counter behavior in red over budget — see AGENT-DEFAULT #3 below

### P2-9. Compilation hygiene: `COL_GAP` unused warning
`tokens.rs:101` — builder flagged this. Still present. If unused after Phase 3 it should either be deleted or referenced (e.g. in `panels.rs` for the inter-column gap).

### P2-10. No `unwrap()` panics in any of the three section files. ✅

### P2-11. `aspect = cn.width / cn.height.max(1)` guards div-by-zero ✅ (`canvas.rs:99`)

---

## Section-by-section

### Header row (`prompt.rs:33-86`)
- PROMPT label: `FONT_SECTION_LABEL` (10.5), `text_dim`, `.strong()`, `.monospace()` — spec says sans (P2-1).
- Task badge (`task_badge`, lines 220-243): renders `task.short_label()`, accent_soft fill, accent border — matches spec.
- Right side: `✦ Enhance` and `✨ Template` ghost buttons present; click logs only.
- **Total height: 30px not 38px.** P0-2.

### Prompt box (`prompt.rs:90-148`)
- 100px tall ✅
- Manual focus ring via `ui.memory(|m| m.has_focus(prompt_id))` — stroke swaps color and width (1.0→1.5). Fires correctly.
- Token counter: `state.prompt.split_whitespace().count()` placeholder, `<n> / 256` mono. Color swaps to `t.danger` over budget (AGENT-DEFAULT #3).
- Counter positioned via `frame_resp.response.rect.bottom() - 12.0` — fragile, see P1-2.

### Negative prompt (`prompt.rs:152-213`)
- Singleline by default, multiline on focus. Id stable across the swap (`Id::new("negative_textedit")`).
- See P1-3 about cursor on the swap frame.

### Action bar (`action_bar.rs`)
- Generate button: 140×32, amber fill, `t.bg` text. ✅
- Stop variant: red fill, white text. ✅ (no `· N%`)
- `+ Queue` ghost: 32px tall, transparent, border stroke. ✅
- Right-side readout: built and joined into one label inside `right_to_left` layout. Matches spec content (P1-4 caveat).
- Generate click toggles `state.generating`; resets `current_step`/`total_steps` to 0 only on stop, never sets non-zero values on start (P0-3).
- Missing 3px progress strip (P1-8).

### Canvas surface (`canvas.rs:54-189`)
- Reserves `total.y - 32` for canvas, `120px` floor.
- Checker bg via base fill + alternating tile shapes. P0-1 naming, P1-5 count.
- `request_repaint_after(16ms)` only inside `if state.generating` ✅ — no idle CPU drain.
- Aspect-locked preview rect with 16px padding ✅.
- 6 hardcoded warm bands when generating (AGENT-DEFAULT #6).
- Scan line at `y = top + height * (time % 2.0 / 2.0)` — sweeps 2s top→bottom, plus a wider faint glow line. ✅
- Two badges: bottom-left (mode-aware), bottom-right (`seed N`). ✅

### Toolbar (`canvas.rs:220-280`)
- 32px Frame (`set_min_height(32.0)`). ✅
- Five ghost buttons: ⊞ Fit, 100%, 📋 Copy, 📂 Open folder, 📤 Send to input. ✅
- Separator after `100%` ✅
- Spacer + status label (right-aligned).
- Status text: `⏵ idle` or `⏵ step <n>/<total>` (P0-3 will display `0/0`).
- Status color: `t.success` when generating, `t.text_mute` when idle (AGENT-DEFAULT #4).
- Frame stroke draws all 4 sides instead of border-top only (P1-1).

---

## AGENT-DEFAULT decisions — assessment of the 7

| # | Decision | Verdict | Rationale |
|---|----------|---------|-----------|
| 1 | Light-theme checker pair `#c8c8cc`/`#d8d8dc` | **Acceptable** | Matches DCC convention (Photoshop/Krita use grey-on-grey checkers regardless of theme). Naming is internally consistent here, unlike the dark pair (P0-1). |
| 2 | Generate stub: bool toggle, no fake timer | **Acceptable** | Honest stub matches brief "Phase 3 stub: clicking toggles state.generating". But pair with P0-3 fix so the toolbar doesn't show `0/0`. |
| 3 | Token counter goes red when over budget | **Defer** (revert for now) | Spec only specifies "muted mono `<n> / 256`". Adding a danger color implies real-time validation that doesn't exist (the 256 number is a placeholder, the splitter is whitespace not BPE). Showing red over a fake threshold misleads the user. Either remove the conditional color or wait until Phase 5 wires real tokenization. |
| 4 | Toolbar status: `success` when generating, `text_mute` when idle | **Acceptable but mild deviation** | Spec doesn't pin a color. JSX uses `T.textMute` for both. Using `success` (green) telegraphs "running OK" which is a reasonable UX nudge. Document it. |
| 5 | Negative prompt focus expansion via stable `Id::new("negative_textedit")` | **Acceptable, monitor** | Approach is correct; only concern is the swap-frame cursor behavior (P1-3). Test interactively. |
| 6 | Preview gradient: 6 hand-picked warm bands | **Acceptable** | The brief explicitly allows a placeholder gradient before real preview pixels arrive. OKLCH in Rust egui would need a color-space conversion crate; not worth it for a stub. |
| 7 | Enhance/Template buttons rendered as ghost in header right | **Acceptable** | Spec line 108 names them explicitly. Builder rendered them with `Color32::TRANSPARENT` fill + `Stroke::NONE` + muted text. Matches "ghost" idiom. |

---

## Scope creep audit

Nothing snuck in. All Phase 5+ items (real generation, real BPE tokenization, drag-drop, real preview pixels, button wiring, keyboard shortcuts, save/open behaviors) are absent or stubbed with `log::debug!`. ✅

One soft scope note: the AGENT-DEFAULT #3 (red token counter) is the closest thing to "real validation" that bled in from the future-self. Reverting it keeps Phase 3 truly stub.

---

## Unverified

- **Visual** check that `add_sized([avail.x, box_height - 20.0], edit)` actually leaves a 20px gutter inside the prompt frame across egui versions. Worked for the builder; should be eyeballed.
- **Negative prompt focus** behavior on the swap frame (P1-3) — needs interactive click testing.
- **Light theme appearance** of the bottom badges (P2-5) — needs theme toggle test.
- **Toolbar full-border** vs border-top only (P1-1) — visual confirmation.
- Whether `Theme::Light` was tested at all for any of these; tokens compile, but no light-theme screenshot mentioned in the builder report.
- The builder report claimed "1 warning (COL_GAP unused)". Not re-verified by running cargo, but no new warning sources spotted in the three section files.
