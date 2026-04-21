# Phase 1 — Skeptic Review

Scope under review: `/home/alex/EriDiffusion/inference_ui/` against
`/tmp/flame_ui_design/design_handoff_flame_ui/README.md`. Builder did not run
`cargo build` for me; rebuild here yields 3 acknowledged warnings only.

---

## P0 — visual/behavior bugs that would land wrong

### 1. Mode tabs render the underline INSIDE the menu-bar bottom border

`chrome.rs:225` allocates `vec2(64.0, 26.0)` for each tab, then
`chrome.rs:239-243` paints a 2px underline at `rect.bottom() - 2.0`. The menu
bar is exactly 34px tall (`tokens.rs:78`) with `inner_margin(symmetric(4.0,
0.0))` (`chrome.rs:160`) and a 1px bottom border via `Stroke::new(1.0,
t.border)` on its frame. Centered vertically (`horizontal_centered`), a 26px
allocation sits at y ≈ 4..30 inside the panel; the 2px underline lands at
y=28..30 — i.e. **2px above** the menu-bar's own bottom border, not flush with
it. The JSX reference (`chrome.jsx:131`) explicitly uses
`borderBottom: '2px solid …'; marginBottom: -1; height: 26; boxSizing:
border-box` so the underline visually replaces the panel border. The Rust
version paints a free-floating bar inside the panel.

Result: the active tab's amber underline will appear as a short stripe
floating above the menu/canvas seam, with the gray border still visible
underneath it. Looks broken at 100% zoom.

### 2. Tab label is painted twice

`chrome.rs:220-223` builds a `RichText` with size + color + strong weight,
but `chrome.rs:230-236` then calls `ui.painter().text(..., label.text(),
FontId::proportional(FONT_BODY), label_color)` — passing only the raw
`&str`, dropping the `RichText`'s strong/weight styling. The earlier
`RichText` build is dead code (the value is shadowed by the painter call).
The tab label will render at default weight even though spec says
`fontWeight: 500` (chrome.jsx:128) and the README shows it bold.

Minor follow-on: because it's painted via `painter().text()` rather than
`ui.label()`, the tab is **not focusable** and won't participate in egui's
keyboard nav.

### 3. Title-bar drag interact swallows the win-controls

`chrome.rs:40-49` calls
`ui.interact(bar_rect, ui.id().with("titlebar_drag"), Sense::click_and_drag())`
on **the entire bar** *before* allocating the row that contains the win
buttons. egui's interaction is z-ordered by allocation, but with `Sense::
click_and_drag` on the full rect and the win-buttons being later
`allocate_exact_size(... Sense::click())` allocations within a child layout,
the drag region's `is_pointer_button_down_on()` will fire when you mouse-down
on a win-button area. Practical effect: clicking the X / max / min may also
emit `StartDrag`, and the buttons' click responses race with viewport-drag.

The JSX reference (chrome.jsx:5-21) does NOT make the win-button div part of
the drag region — they're separate flex children. The Rust code needs to
either (a) compute a sub-rect that excludes the win-controls (`bar_rect`
minus 3 × WIN_BTN_W on the right) or (b) not interact the whole bar and
rely on a left-cluster sub-allocation.

Builder needs to confirm whether double-click-to-maximize on the win-buttons
also fires the maximize toggle on bar — currently it would.

### 4. `MENUBAR_H = 34` but actual rendered height is 35+

`chrome.rs:159` sets `Frame::none().stroke(Stroke::new(1.0, t.border))` on
the menu bar with `exact_height(MENUBAR_H)` = 34px. egui paints the stroke
**inside** the frame's outer rect by default but adds it to the layout space
needed; with a 1px stroke on all sides the available content height drops to
~32px. Same issue on the title bar (`chrome.rs:33`) and status bar
(`chrome.rs:329`). The README is explicit about 32 / 34 / 22 — these
specifications are the **outer** heights per the bottom-bordered Win11 mock.
The current code likely renders the title bar at 34px (32 + 1 border each
side) etc. Not catastrophic visually, but the perfectionist reading of
"32px tall" is off by 1–2px for every chrome row, and the cumulative drift
is 4–6px shaved off the canvas.

---

## P1 — spec deviations (sizes, colors, menu items)

### 5. Menu-bar background — wrong token

`chrome.rs:158` sets `Frame::fill(t.panel)`. The README "Window & chrome"
section doesn't pin a color for the menu bar, but the JSX prototype
(`chrome.jsx:62`) also uses `T.panel`. Acceptable, but note that panel is
`#25252a` which is the same fill as the params/queue side panels — there is
no visual seam between the menu bar and the side panels below it, only the
1px border. In the JSX mock the seam is more visible because the side
panels sit on `T.bg` (`#1a1a1c`) outside their own frames; here the side
panels are also `t.panel` (`panels.rs:22, 50`), so menu-bar→side-panel
is a continuous fill. Probably wrong vs. the design intent. Cite as P1.

### 6. Mode-tab hit area is wrong width

`chrome.rs:225` hard-codes `vec2(64.0, 26.0)` per tab. JSX
(`chrome.jsx:127-132`) uses `padding: '5px 12px'` plus an icon plus the
label — natural width, ~70-80px depending on label. Hard-coded 64px makes
both tabs the same width regardless of label length. "Image" fits, "Video"
fits, but if anyone changes the labels they'll truncate. Minor.

### 7. Mode tabs lack icons

JSX `chrome.jsx:100-101` passes `icon={<Ico.image size={12}/>}` and
`<Ico.video size={12}/>` to each tab. The Rust mode_tab function takes only
the label. README only weakly implies icons (says "phosphor / lucide"
generally). Acceptable for Phase 1 if scope was "no icons yet", but the
mode tabs are part of chrome — flag.

### 8. Theme-toggle glyph mapping is inverted from the JSX

`chrome.rs:185-188`: when current theme is **Dark**, it shows ☀ ("click for
light"). JSX `chrome.jsx:105`: `dark ? <Ico.moon/> : <Ico.sun/>` — when
dark, it shows the **moon** (current state), not the sun. Different mental
model. The README doesn't pin which one, so this is a defensible
AGENT-DEFAULT, but it diverges from the prototype and the comment
"currently dark → click for light" telegraphs that the builder picked the
opposite-of-prototype convention. Worth a user call.

### 9. Status-bar font is **section-label sized, not mono**

`chrome.rs:344-348` uses `RichText::new(line).size(FONT_SECTION_LABEL)
.monospace()`. The README "Window & chrome" says: "Status bar: 22px tall at
the bottom, **mono font**, muted." Typography section: status bar should be
"JetBrains Mono 11px" (the `FONT_MONO` constant exists at `tokens.rs:90`
but isn't used here). Builder defaulted to 10.5 (FONT_SECTION_LABEL)
instead of 11. Single point but the constant is right there.

Also `monospace()` on egui `RichText` shifts to the bundled monospace
family — but Phase 1 hasn't loaded JetBrains Mono yet (`theme.rs:7-9`
explicitly defers font loading). That's acceptable per the deferral, but
worth flagging that the spec'd typography is not yet in place.

### 10. Edit menu — Preferences removed correctly, but `—` separator left orphaned in the JSX is gone here too

JSX `chrome.jsx:38` shows: `Edit: [['Copy prompt'], ['Paste prompt'],
['Clear prompt'], '—', ['Preferences…','Ctrl+,']]`. README
`Edit` definition (line 155): "Copy prompt Ctrl+C, Paste prompt Ctrl+V,
Clear prompt" (no separator, no Preferences). Builder correctly stripped
both — `chrome.rs:265-271` matches the README exactly. ✓

### 11. Menu items use `ui.button("text   shortcut")`, not real shortcut display

`chrome.rs:254-262` etc. encode shortcuts as padding inside the label string
(`"New generation        Ctrl+N"`). egui has `ui.add(egui::Button::new(...)
.shortcut_text(...))` for properly right-aligned shortcut hints (matches the
JSX visual at chrome.jsx:90-91 with `flex:1` + right-aligned mono span).
Current implementation will render a single left-aligned label with
whitespace padding that depends on the label length and proportional font
metrics — shortcuts won't line up across menu items.

### 12. Menu-bar separator between menus and tabs is misaligned

`chrome.rs:174-176` allocates `(1.0, 18.0)` and paints with `t.border`.
JSX `chrome.jsx:98` uses `width: 1, height: 18, margin: '0 8px'`. Rust
uses `add_space(6.0)` before and `add_space(4.0)` after — total 10px gap,
not 16px. Off by 6px.

### 13. Title-bar font weight wrong

`chrome.rs:60-64` makes "Flame" bold (`.strong()`). JSX `chrome.jsx:11`:
`fontWeight: 500` (medium, not bold). egui's `.strong()` is heavier than
500. Cosmetic but specific.

### 14. Win-button glyphs use Unicode text, not real icons

`chrome.rs:78-84` uses `"✕"`, `"▢"`, `"—"`. The JSX uses dedicated SVG
glyphs (`Ico.winMin`, `Ico.winMax`, `Ico.winClose`). The bundled egui font
likely renders ✕ but `▢` (U+25A2) is **rare** and may render as tofu
(`□`) or fallback ugly. The spec says "Use `egui-phosphor` or `lucide`
variants" (README line 289). Phase 1 deferred icon loading entirely
(reasonable), but "▢" is a particularly poor placeholder. "🗖" or even
just an outlined square painted via primitives would render more
predictably.

### 15. Maximize toggles via `Maximized(bool)` reading `viewport().maximized`

`chrome.rs:46-48` and 79-81 do `Maximized(!current)` with `unwrap_or(false)`.
egui 0.29 has `ViewportCommand::Maximized(bool)`; toggling requires reading
state. The implementation is correct *if* the viewport reports the maximized
state synchronously. If a wayland compositor or X11 WM doesn't update
`viewport().maximized` synchronously, two rapid double-clicks could end up
desynced. Low risk, but no `Toggle` variant is used because egui 0.29
doesn't expose one — acceptable.

### 16. Status bar lacks the `· generating · 42%` segments

README example: `● backend connected · D:\flame\weights · 16 models · ready
/ generating · 42%`. The Rust string at `chrome.rs:337-343` only shows
state ("ready"), no progress percentage segment. Builder's comment says
"placeholder" — fine for Phase 1, but the README explicitly lists 5
segments. Note the structure for later wiring.

### 17. Outer 6px window padding is missing

README "Layout" line 40: "Three resizable columns with 6px gap and 6px
**outer padding**". The Rust app paints `TopBottomPanel`s flush against the
window edge (no 6px inset on the title bar or status bar) and the side
panels sit flush against the side edges. Only the canvas's CentralPanel has
`Margin::same(OUTER_PAD)` (`panels.rs:72`), and even that goes on the
inside of the central area, not as window-edge padding around the whole UI.
The 6px outer pad isn't realized.

### 18. 6px inter-column gap is missing

README same line: "6px gap" between columns. egui `SidePanel`s have no
explicit gap — the params and queue panels are flush against the
CentralPanel's outer edge. The CentralPanel's `inner_margin(Margin::same(
OUTER_PAD))` (panels.rs:72) provides padding **inside** the central column
on all sides; that gives 6px padding around the inner card's left/right,
which approximates the gap, but it's not a real 6px gap (it's
panel-border + 6px inner-margin + inner-card-border per side, ~8-10px
total). The Frame stroke on the side panels (1px) and the inner card stroke
(1px) compound visually.

### 19. Inner-card pattern in the canvas is wrong shape

`panels.rs:76-89` puts a `Frame` *inside* the CentralPanel that has its own
`fill(t.panel)`, `stroke`, and `inner_margin(8.0)` — basically a card
within a panel. The README says the canvas IS the central panel
(transparent checker bg, `t.bg`), with a 38px header for prompt/badge and
the canvas drawing area below (lines 107-130). This nested card is not in
the spec; it makes the canvas look like a third side-panel with rounded
corners. Builder added it for "visual consistency" but it's not in the
brief.

### 20. Border radius on side panels not applied

The README "Spacing" section: "3px on controls, 4–6px on panels and
popovers." `panels.rs:24` sets `inner_margin(Margin::same(8.0))` but no
`.rounding(...)` on the side-panel Frame. SidePanels filled flush with
the window edge can't have rounded corners on the outer edges anyway, but
the inner panel rectangles in the JSX have rounded corners on the
canvas-facing side. Note for later, not blocking Phase 1.

---

## P2 — minor / cosmetic / acceptable trade-offs

### 21. Theme `Default` returns `Dark` (state.rs:147 → theme.rs:23)

Reasonable, matches "high-fidelity dark first". OK.

### 22. `Mode::Image` is the default tab (state.rs:21). README doesn't pin a default; matches JSX. OK.

### 23. `Task::T2I` is default for both image AND video `ModeSettings`

`state.rs:48-51` makes T2I the universal default, but the video mode's
default should be T2V (per README line 61). Since `ModeSettings: Default`
is derived, both `image` and `video` get T2I. When the user first switches
to Video tab in Phase 2, the Task picker will show "T2I" highlighted
which isn't valid for video. Phase 2 will set valid defaults; flagging
because it's a structural data issue, not a UI issue.

### 24. `LoraSlot::active` defaults to `false`

State default makes new LoRAs inactive. Reasonable for "user adds, then
opts in", but the JSX reference (params-panel) defaults to active when
added. Phase 2 concern.

### 25. `batch_count` and `batch_size` default to 0

`AppState::default()` derives all numeric fields to 0. README says
batch_count is `1..=64`, batch_size is `1..=8`. Default of 0 is out of
range. Phase 1 has no widgets so this never displays, but the persistence
layer in Phase 6 will need to clamp / use `Default` impls that respect
ranges. Note for later.

### 26. `seed` defaults to 0, not -1

README line 80: "DragValue -1..=9_999_999_999, **-1 = random**". `seed:
i64` defaults to 0 from derive(Default), but seed_mode defaults to Random
which means seed value doesn't matter. Mismatch between the field defaults
is not critical but inconsistent.

### 27. `current_mut()` / `current()` (state.rs:166, 173) are dead code in Phase 1

Builder added these accessors prophylactically. Useful in Phase 2; warning
is harmless.

### 28. SVG asset shipped but unused

`assets/flame.svg` exists (542 bytes) but `chrome.rs:94-113` paints the
icon with rect primitives. Builder noted this as AGENT-DEFAULT. Asset is
bytes-only-overhead until Phase 2+. Fine.

### 29. `ron` dependency declared in Cargo.toml but unused in Phase 1

`Cargo.toml:19` pulls `ron = "0.8"` but no module uses it. Persist layer
is Phase 6. Marginal, no warning emitted.

### 30. `tokens::COL_GAP` declared but never used

`tokens.rs:84`. Warning is one of the 3 acknowledged.

---

## Builder's AGENT-DEFAULT decisions — assess each

| Decision | Where | Assessment |
|---|---|---|
| Paint flame icon with rect primitives instead of loading SVG | `chrome.rs:94-113` | **Acceptable for Phase 1.** Avoids egui_extras dep. But the result (two stacked rounded rects, one with white-tinted alpha overlay) does **not** look like a flame. It looks like a stadium-shape pill with a translucent blob inside. Pure rectangles cannot convey a flame silhouette without paths. The shipped `flame.svg` has actual teardrop curves. Builder should either (a) load the SVG via `egui::Image::from_bytes` + `egui_extras` with the `svg` feature, or (b) paint with `painter.add(Shape::Path { ... })` using bezier-approximation points. Current "primitive" version is a placeholder masquerading as the design icon. P1-leaning. |
| Defer font loading (Inter / JetBrains Mono) | `theme.rs:7-9` | **Acceptable.** Crate has no asset deps. `RichText::monospace()` falls back to egui's bundled monospace. Note that all "mono" copy in chrome (`v0.4.2`, status bar) renders in egui's default mono, not JetBrains Mono. README typography is therefore not satisfied; deferral is documented. |
| Drop "Preferences…" from Edit menu | `chrome.rs:265-271` | **Correct.** README line 35: "No Preferences / Settings modal, no Tweaks — explicitly removed." Builder went by README, not the JSX prototype which still has it. |
| Use `t.panel` for menu-bar fill | `chrome.rs:158` | Defensible; matches JSX. README doesn't specify. |
| Inner card inside CentralPanel | `panels.rs:76-89` | **Wrong direction.** README says canvas central panel has the checker bg directly with a 38px header inline; this adds a containing card. Should be removed in Phase 3. |
| `min_width(240)` and `max_width(480)` on params SidePanel | `panels.rs:18-19` | Reasonable safety rails. README only specifies default 316. OK. |
| `min_width(200)` and `max_width(420)` on queue SidePanel | `panels.rs:46-47` | Same. OK. |
| Theme toggle shows "click target" glyph (sun when dark) | `chrome.rs:186-187` | Diverges from JSX prototype which shows "current state" glyph. Either is a defensible UX. Worth user call. |
| `Maximized(bool)` instead of any toggle helper | `chrome.rs:46-48, 79-81` | Forced by egui 0.29 API. OK. |
| `Mode::label()` returns `&'static str` | `state.rs:26-32` | Clean. OK. |

---

## Scope creep audit

Checked Phase 1 brief vs. delivered code. Disallowed scope: params widgets,
canvas rendering, queue/history list, prompt box behavior, generate button,
NVML perf, persistence, drag-drop.

- ✅ No params widgets — `panels.rs:30-37` is placeholder text only.
- ✅ No canvas rendering — `panels.rs:81-88` is placeholder text.
- ✅ No queue list — `panels.rs:54-62` is placeholder text.
- ✅ No prompt box — placeholder.
- ✅ No generate button — not present.
- ✅ No NVML / perf — not present.
- ✅ No persistence — `ron` is in Cargo.toml but unused; no `state.ron`
  loader/saver.
- ✅ No drag-drop — not wired.

Marginal: `state.rs` defines the **complete** ModeSettings + sub-types
(LoraSlot, ControlNet, AdvSamplingOpts, PerfOpts, OutputOpts) — but Phase 1
brief explicitly says "state struct declarations" so this is in scope, not
creep. The defaults are auto-derived; no behavior is wired.

**No scope creep.**

---

## Unverified — couldn't check

- **Visual rendering at runtime.** I read the code but did not launch the
  binary; everything above is from spec compliance reading. Any rendering
  glitches that depend on the actual painted output (icon legibility, tab
  underline alignment vs. menu bar bottom-border, hover states) need a
  human eye on a real window.
- **`ViewportCommand::StartDrag` actually drags on Wayland vs. X11.**
  eframe 0.29 supports both per `Cargo.toml:14` features, but the win-
  controls overlap concern (P0 #3) might present differently per backend.
- **Whether double-click on bar maximizes** in practice given the
  interact-on-full-bar shadowing the win-controls.
- **Font fallback chain** for "▢" U+25A2 in egui's bundled font set.
  Likely renders as a fallback or tofu but I haven't confirmed.
- **Whether `cc.egui_ctx.set_visuals` in `FlameInferenceApp::new`** is
  retained across `ctx.set_visuals` calls in the View menu's "Toggle dark
  theme" handler — should work, but order of `set_visuals` vs.
  `apply_density` (theme rebuild always wipes density first) means the
  density tweaks are reapplied on every toggle. Acceptable.

---

## Summary

**3 P0**, **16 P1**, **10 P2**. Key issues:

1. Mode tab underline (P0 #1) — paints *inside* the panel border, not flush
   with the menu/canvas seam → free-floating amber stripe under the active
   tab.
2. Tab label is double-rendered with the styled `RichText` discarded
   (P0 #2) — wrong weight, no keyboard nav.
3. Title-bar drag region overlaps win-controls (P0 #3) — clicks on
   close/max/min may also fire `StartDrag`.
4. Six-pixel outer window padding and column gap (README "Layout") not
   realized (P1 #17, #18).
5. Inner card inside CentralPanel (P1 #19) — not in spec, will need to be
   removed before Phase 3.
6. Flame icon painted as two stacked rounded rectangles, doesn't read as a
   flame (AGENT-DEFAULT assessment).
7. Status bar uses 10.5px size + monospace fallback rather than spec'd
   11px JetBrains Mono (P1 #9) — partly font-deferral, but the size is
   wrong in the constant choice.
8. Menu-item shortcuts encoded as padded label strings (P1 #11), won't
   right-align across items.

Tokens themselves (all 13 dark + 12 light hex values) are byte-accurate.
State struct shapes match the README. Build is clean (3 expected dead-code
warnings). Scope discipline is good — no creep into Phase 2+.
