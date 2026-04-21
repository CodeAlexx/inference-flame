# Phase 4 Skeptic Review — Queue / History panel + Perf footer

Code reviewed:
- `inference_ui/src/sections/queue.rs` (577 LoC)
- `inference_ui/src/sections/perf_footer.rs` (171 LoC)
- `inference_ui/src/widgets/progress_bar.rs` (93 LoC)
- `inference_ui/src/widgets/thumbnail.rs` (163 LoC)
- `inference_ui/src/state.rs` Phase 4 additions (lines 332-506, 547-580)
- `inference_ui/src/panels.rs:136-174` (queue_panel wiring)

Spec reference: `/tmp/flame_ui_design/design_handoff_flame_ui/README.md` lines 131-148.
Visual ref: `/tmp/flame_ui_design/design_handoff_flame_ui/components/queue-panel.jsx`.

---

## P0 — must fix

### P0-1. Active-tab underline is over-painted by the row border line
File: `queue.rs:55-98` + `queue.rs:157-163`

Order of operations in `tab_bar`:
1. `paint_tab(queue)` — paints underline at `rect.bottom()-2..rect.bottom()` if active
2. `paint_tab(history)` — same
3. `painter().line_segment([row_rect.left_bottom(), row_rect.right_bottom()], 1px border_soft)` — paints 1px line over the entire row bottom

The row's `rect.bottom() == row_rect.bottom()`, so the bottom 1px of the 2px-tall accent underline is *over-painted* by the border line — leaving only a 1px accent stripe with a 1px border_soft stripe under it. The chrome.rs `mode_tab` *explicitly* fixed the same z-order trap by painting the underline on `Order::Foreground` via `ctx().layer_painter(...)` (chrome.rs:269-273) — the comment in `paint_tab` ("same trick as the menu-bar mode tabs") is **factually wrong**: queue.rs uses ordinary `ui.painter()`, which is in-order with the border line.

Either: (a) draw the border line BEFORE the tabs, then the underline naturally sits on top; or (b) mirror chrome.rs and use `ctx().layer_painter(LayerId::new(Order::Foreground, ...))`.

The brief explicitly called out: "Verify the underline lands at the right y-coordinate (spec says it should sit at the bottom of the tab bar replacing the border)" — this is exactly the case it asked us to verify, and it currently doesn't replace the border, it co-mingles with it.

### P0-2. Perf footer renders only 2 of the 3 spec'd progress rows — but stat_row's `warn` parameter is *never* set true for VRAM at high values

File: `perf_footer.rs:82-105`

Spec line 144-147:
- Title row: GPU name + temperature
- VRAM row: text + thin bar
- GPU row: text + thin bar

Rust correctly omits the JSX-only Temp progress row (spec-aligned). However: `stat_row(... false)` is hardcoded for both rows. The internal `progress_bar::thin_bar` *does* auto-warn at `progress >= 0.80` (progress_bar.rs:56), so the bar fill turns red ≥80% — but the value text on the right (`19.1 / 24.0 GB`) keeps its `text_dim` color even when VRAM is dangerously full, because `val_color` is gated on the `warn` parameter only. JSX (queue-panel.jsx:169) ties the value-text color to the same `warn` predicate as the bar.

Not a P0 by severity (visual only), but flagging because it inverts the visual contract: **when VRAM hits 95%, the bar will be red, but the GB readout stays muted grey, and the user has no large red text to anchor on**. Demote to P1 if you disagree.

→ Reclassifying to **P1-3** below; keeping P0-2 slot empty.

---

## P1 — likely bugs

### P1-1. Queue panel border is 4-sided but spec says only borderTop on the perf footer

File: `perf_footer.rs:152-158`

```rust
.stroke(Stroke {
    width: 1.0,
    color: t.border,
})
```

Same trap as Phase 3's `canvas.rs` toolbar (SKEPTIC_PHASE3 P1-1). egui `Frame::stroke` strokes all 4 sides. JSX `queue-panel.jsx:145` is `borderTop: 1px solid ${T.border}` only. The footer will get a visible left/right/bottom border; combined with the SidePanel's outer stroke (panels.rs:144), the bottom and sides will be *double-bordered*.

Fix idiom (already used elsewhere): drop the Frame stroke and paint a `line_segment` along the top edge inside `show()`.

### P1-2. Queue and History bodies have zero horizontal padding; JSX specifies `padding: 6`

File: `queue.rs:187-235` (queue_body), `queue.rs:396-464` (history_body), `panels.rs:161` (`inner_margin: Margin::same(0.0)`)

Panel inner_margin is 0; queue/history body code does `ui.add_space(4.0)` for top space only — never adds left/right padding. The frames inside (`Frame::none().fill(t.panel_sunk)...`) draw flush against the panel walls. JSX `queue-panel.jsx:33` and `:43` both wrap in `<div style={{ padding: 6 }}>`. Visually the queue rows touch the panel's left/right edges including its 1px stroke; on the design comp, there's a 6px inset on both sides.

The `tab_bar` *should* be flush (matches JSX `borderBottom` on the container which spans full width), but the *bodies* should be inset.

### P1-3. `stat_row` never marks VRAM/GPU rows as warn even at high values

(Promoted from P0-2 above.) `perf_footer.rs:92` and `:104` both pass `false` literally. JSX matches the bar fill color to the value text color; Rust does only the bar. At 95% VRAM the bar is red but the text reads in `text_dim` grey.

Cheap fix:
```rust
let warn = ratio >= progress_bar::WARN_THRESHOLD;
stat_row(ui, t, "VRAM", &format!(...), ratio, warn);
```

### P1-4. Featured-tile aspect ratio not enforced; hardcoded 120px height ignores panel width

File: `queue.rs:419-425`

Spec / JSX: `aspectRatio: '2 / 1'` for the featured tile. Rust uses `featured_h = 120.0` regardless of `avail_w`. With the default panel width (260px) and 0 inner_margin, panel content width is ~260, so a 2:1 tile would be 130px tall — close enough. But the SidePanel is **resizable** (`min_width: 200`, `max_width: 420` per panels.rs:140-142). At 200px wide the tile becomes 120/200 = 0.6 ratio (5:3, too tall); at 420px it becomes 120/420 = 0.28 ratio (way too short).

Same critique applies to `row_h = 80.0` for the 1:1 cells — at 200px panel each cell is ~98px wide × 80 tall (close to 1:1 only by coincidence at default width).

If the grid is supposed to read as "square thumbnails", they should be sized from `cell_w_now` not constants:
```rust
let row_h = cell_w_now;     // 1:1
let featured_h = avail_w / 2.0;  // 2:1
```

### P1-5. `cell_w` is dead code; `let _ = cell_w` only silences the warning

File: `queue.rs:413` + `:463`

```rust
let cell_w = (avail_w - gap) * 0.5;
...
let _ = cell_w;
```

The local is computed and never read. The `let _ = cell_w` line is a warning-silencer that confirms the dead-code rather than addresses it. The `cell_w_now` *inside* the loop (line 433) supersedes it. Just delete the outer `cell_w` computation.

(Builder report claimed 2 warnings — `COL_GAP` + `thumbnail_path`. This makes the actual count higher: `cell_w` is dead too, and `_ = cell_w;` is a code-smell mask.)

### P1-6. Tab-bar label width hack uses 6.5 px/char but lays out at FONT_MONO + 0.5 = 11.5 px

File: `queue.rs:177-183`

`label_galley_width_hack("Queue")` = 5 × 6.5 = 32.5 px. But `Queue` is rendered at `FontId::proportional(11.5)` via egui's default proportional font. At 11.5 px proportional, "Queue" is closer to 38-40 px wide. So the count `(N)` will collide visually with the right edge of the label, especially for "History" (7 × 6.5 = 45.5 vs actual ~55 px). The drift is "within ~2px for ASCII" claim in the comment is optimistic.

The proper fix is what the comment dismisses: layout the galley first to a `Vec<Galley>` you keep, or use `ui.fonts(|f| f.layout_no_wrap(...))` to measure without consuming. Cost is identical — the optimization argument is a red herring.

### P1-7. Cancel-running × button doesn't close-menu / no real action

File: `queue.rs:299-314`

Click sets `cancel_running = true`, then queue_body logs at debug level only ("Cancel running clicked (Phase 5+ wiring)") and explicitly comments "Don't actually remove — keeps the visual interesting for Phase 4 demos." This is documented but the user clicking the × in Phase 4 will see *nothing happen* — no visual feedback, no removal, no log they'd see in release. Compare to `Remove` on a queued row, which *does* remove (line 230: `state.queue.queued.remove(i)`). The asymmetry is jarring.

Either remove the cancel handler entirely (so the × is honestly inert pending Phase 5), or have it match the queued-row semantics and clear `state.queue.running = None`.

### P1-8. `paint_into` painter is `painter_at(rect)` — clips strokes to the rect, but the 1px border is half-outside

File: `thumbnail.rs:48` + `:117`

`painter_at(rect)` clips to `rect`. Then `painter.rect_stroke(rect, radius, Stroke::new(1.0, t.border_soft))` paints a stroke centered on `rect`'s edge — half inside, half outside. The outside half is clipped, so the visible stroke is ~0.5 px not 1 px. (Same egui idiom in chrome.rs and other widget files presumably has the same artifact; this isn't unique to Phase 4 but it's introduced here.)

### P1-9. `placeholder` returns a Response that's silently dropped at queued_row callsite

File: `thumbnail.rs:24-28` + `queue.rs:258, 338`

`placeholder` allocates the rect with `Sense::click()` and returns the Response. The history grid uses `paint_into` (no allocation) and attaches its own ui.interact for click. The queue rows use `placeholder` and *drop* the response entirely — meaning the thumbnail's rect IS clickable (consumes input) but no handler runs. If the user clicks the running-job thumbnail expecting to "open the active result", nothing happens — and the click also doesn't fall through to any parent.

Either: (a) change `placeholder` to use `Sense::hover()` if no behavior is wired, or (b) attach a debug-log click handler so it's not a dead input zone.

---

## P2 — minor / acceptable

### P2-1. `temperature_c >= TEMP_DANGER_C` (≥ 80) vs spec "above 80" (> 80)
File: `perf_footer.rs:32, 68`. Spec wording is "red tint above 80"; strictly that's `> 80`. Off-by-one at the boundary, never visible in practice.

### P2-2. `now` captured once per frame for relative_time
File: `queue.rs:416`. Fine — relative-time text only shifts every second anyway and the perf footer triggers a 1Hz repaint, so all tiles re-format from a coherent `now`.

### P2-3. JSX has 3 PerfStat rows (VRAM/GPU/Temp); Rust only has 2 + temp in title
File: `perf_footer.rs:82-105`. Rust matches the **spec** (lines 145-147) — the brief says title row + 2 stat rows. JSX is over-spec; Rust correctly limits. ✅ Acceptable per spec, but worth flagging because the builder pulled "verbatim from JSX" for queue mock data — the same fidelity standard isn't applied to perf footer rows.

### P2-4. JSX has only one `<Ico.x/>` per queued row; Rust adds `⏵` Run now in addition
File: `queue.rs:369-376`. Rust matches the **spec** ("Actions on hover: ⏵ run now, × remove") — the brief here is canonical, JSX is incomplete. ✅

### P2-5. `relative_time` returns "—" on clock-skew error
File: `queue.rs:559-563`. Defensive but unlikely. Acceptable.

### P2-6. History `vid` badge missing
JSX `queue-panel.jsx:113-117` paints a `vid` corner badge for video outputs. Spec README does not mention it. `HistoryItem` struct has no video field. Rust correctly omits per spec. Forward-looking reminder for Phase 5.

### P2-7. Mock queue prompts are pulled from JSX prototype verbatim ✅ (matches AGENT-DEFAULT #6)

### P2-8. `fill(t.accent_soft) + stroke(accent)` on running row matches JSX ✅

### P2-9. Two-color gradient via Mesh in thumbnail uses warm hue range (10-90°) — design choice consistent with "Flame"
Slightly different from JSX which uses per-item `hue` field. Rust derives hue from `id * 53 % 80`; deterministic but loses the design's hand-picked `[30, 25, 40, 60, 20]` palette. Acceptable.

### P2-10. `truncate` is duplicated from `lora.rs` (same author note: "duplicated rather than promoted")
Acceptable per the comment, but now there are 2 copies — when the third lands, promote.

### P2-11. `temperature_c` is `f32` rendered as `{:.0}°C`
File: `perf_footer.rs:71`. JSX rounds via `Math.round`; same effect. ✅

### P2-12. `RADIUS_CONTROL = 3.0` used on queued row Frames; JSX uses `borderRadius: 3` ✅

### P2-13. No `.unwrap()` panics in queue.rs / perf_footer.rs / thumbnail.rs / progress_bar.rs ✅

### P2-14. `CronCreate`-style 2 Hz / 1 Hz cadence
The `request_repaint_after` is a *minimum* — actual frame rate depends on what else triggers repaints. During generation, the canvas section's 16ms repaint will dominate, so the perf footer will paint at ~60 FPS. Spec says "updates 2 Hz when generating" — that's about polling cadence, not paint cadence. When NVML wiring lands in Phase 6, the polling logic itself needs to honor the 500/1000ms cadence; just bumping repaint isn't enough.

---

## Tab bar / queue / history / perf — section-by-section

### Tab bar (`queue.rs:55-183`)
- Two tabs: `Queue (N)` | `History (N)` ✅ counts wired (queued.len + running.is_some)
- 28px row height ✅ (matches JSX `height: 28`)
- Active fill: `panel_sunk` ✅
- Inactive hover: `t.row` ✅ (light tint)
- Label color: `text` active, `text_dim` inactive ✅
- Count format: `(N)` mono 10px ✅
- Underline: 2px, accent, inset 2px from sides ✅ — but **z-order broken** (P0-1)
- Width-hack measurement: 6.5 px/char approximation drifts (P1-6)

### Queue tab body (`queue.rs:187-389`)
- Empty state: "Queue is empty" ✅ (matches JSX text)
- Running row layout: 80×60 thumb + prompt + bar + ETA + status badge + cancel × ✅
- Queued row layout: grip + 50×40 thumb + prompt + param summary + (⏵, ×) ✅
- Param summary format: `1024² · 28s · euler` (or `WxH · steps · sampler`) ✅
- Hover actions: always-visible (AGENT-DEFAULT #3, acceptable trade-off)
- Drag handle: `≡` glyph, tooltip "Drag to reorder (Phase 5+)" ✅ visual only
- Action wiring: Remove DOES remove ✅, Run-now logs only ✅, Cancel logs only (P1-7 inconsistency)
- Missing: 6px horizontal padding inside body (P1-2)

### History tab body (`queue.rs:396-464`)
- Empty state: "No history yet" ✅
- Featured first row, full-width ✅ — but height fixed not 2:1 (P1-4)
- Subsequent rows: 2 cells ✅
- Tile painting: gradient + bottom strip overlay + prompt + relative time ✅
- Right-click context menu via `attach_history_context` ✅
- Context items present (in order): `Open`, `Reveal in folder`, `Copy prompt`, `Send to input`, `Upscale`, `Use seed`, `Delete` ✅ all 7, no separators ✅
- Item handlers all log + close_menu ✅
- Missing: 6px horizontal padding inside body (P1-2)
- Dead `cell_w` masked by `_ = cell_w` (P1-5)

### Perf footer (`perf_footer.rs`)
- Title row: `⬛ <GPU name>` + temperature right-aligned ✅
- Temp red threshold: `>= 80` instead of `> 80` (P2-1)
- VRAM row: text + thin bar ✅
- GPU row: text + thin bar ✅
- No Temp progress row (spec-aligned, JSX-divergent — P2-3)
- Refresh cadence: 500ms generating, 1000ms idle ✅ (paint cadence not polling cadence — P2-14)
- 4-sided border instead of borderTop (P1-1)
- No warn-coloring on value text (P1-3)
- Footer auto-sized ~64-66px instead of forced 80 (AGENT-DEFAULT #9, acceptable per builder note)

### State extensions (`state.rs:332-506`, `:547-580`)
- `QueueTab` enum ✅ (Queue/History, default = Queue)
- `QueueJob` struct ✅
- `HistoryItem` struct ✅ (with `thumbnail_path: Option<PathBuf>` reserved for Phase 5+)
- `QueueState` struct ✅ with queued/running/history/active_tab
- `PerfTelemetry` struct ✅
- `AppState.queue` and `.perf` both `#[serde(skip)]` ✅
- `QueueState::mock()` populates 1 running + 3 queued + 5 history ✅
- `PerfTelemetry::mock()` populates RTX 4090 / 19.1 / 24.0 / 62% / 45°C ✅

### Compilation hygiene
Builder reported 2 warnings; actual count is higher:
- `COL_GAP` carry-over (still present, tokens.rs:105) ✅ as reported
- `thumbnail_path` field unused on `HistoryItem` (state.rs:374) ✅ as reported
- `cell_w` (queue.rs:413) **NEW dead local** masked by `let _ = cell_w` (P1-5)
- `cell_w_now` is shadowed/recomputed inside the loop, fine
- `thin_bar_pct` and `thin_bar_ratio` are `#[allow(dead_code)]` annotated — explicitly forward-looking (acceptable)

---

## AGENT-DEFAULT assessment — the 10

| # | Decision | Verdict | Rationale |
|---|----------|---------|-----------|
| 1 | PerfTelemetry/QueueState `#[serde(skip)]` | **Acceptable** | Brief and module comments explicitly defer persistence to Phase 6. History is borderline (most generators DO persist it), but persisting mock data into the user's RON file would be worse than not persisting. |
| 2 | `thin_bar_pct` / `thin_bar_ratio` dead-code helpers with `#[allow(dead_code)]` | **Acceptable** | API ramp for Phase 6 NVML wiring. The annotation makes the intent explicit; not silently dead. |
| 3 | Hover actions always visible | **Acceptable** | Same trade-off used in lora.rs for the × button. egui doesn't have cheap "show on hover only". The accent_soft/transparent buttons read as quiet enough. |
| 4 | Drag handle visual-only | **Acceptable** | Matches Phase 2 LoRA precedent; brief explicitly says "(inert OK)". |
| 5 | Queue panel `inner_margin = 0` | **Partially defensible** | Tab bar correctly spans full width (matches JSX `borderBottom` on container). But the *body* lacks the 6px JSX padding (P1-2). The fix isn't "set inner_margin=6 on the panel" (that would offset the tab bar too) — it's "add `Margin::symmetric(6,4)` on the body Frames or a 6px outer margin on the body container". Half-acceptable as written. |
| 6 | Mock data verbatim from queue-panel.jsx | **Acceptable** ✅ | Matches Phase 3 precedent (canvas mock badges) and gives the design comp pixel parity for free. |
| 7 | `⬛` glyph for GPU icon | **Acceptable** | Spec literally says `⬛ <GPU name>`. Honest. The JSX uses `<Ico.cpu/>` (a real glyph) but the spec wording trumps. If/when an icon font ships, swap. |
| 8 | Tab bar width hack `6.5 px/char` | **Reject** (P1-6) | Drift is too large at the typical label widths. The "performance" justification doesn't hold up — `ui.fonts(|f| f.layout_no_wrap(...))` is identical cost. Fix when convenient. |
| 9 | Footer min-height not forced 80 | **Acceptable** | Spec says "~80px" with the tilde. Builder's note honest about why. If the user wants exact 80, swap to `.exact_height(80.0)` per the source comment. |
| 10 | Cancel-running click logs but doesn't remove | **Reject** (P1-7) | Asymmetric with queued-row Remove which DOES remove. Either both stub (consistent) or both wire (consistent). The current "kept interesting for demos" rationale is just stale-state dressing. |

---

## Scope creep audit

Searched for forbidden Phase 5+ items:
- ❌ NVML — none. No `nvml-wrapper` in Cargo, no live polling in perf_footer. ✅
- ❌ Real image loading — none. `egui_extras::install_image_loaders` not called. Thumbnails are 100% procedural meshes. ✅
- ❌ Right-click items wired to behaviors — all log only, then `ui.close_menu()`. ✅
- ❌ Drag-reorder — no `egui-dnd` import; grip is visual + tooltip. ✅
- ❌ Disk scan for History — `HistoryItem::thumbnail_path` declared but never read; mock data only. ✅
- ❌ Real progress from worker — running job's `progress: 0.60` is hardcoded mock; no IPC. ✅

One soft scope note: the `request_repaint_after(refresh)` in perf_footer.rs:42-47 already implements the polling cadence, even though there's no polling target. That's forward-looking, not creep — when the NVML wiring lands the cadence is already correct. ✅

No forbidden creep detected.

---

## Unverified

- **Visual** confirmation that the over-painted underline (P0-1) is actually visible, not masked by anti-aliasing artifacts on the 1px border line. Strongly suspected to be visible but un-screenshot-tested.
- **Visual** check that the 4-sided footer border (P1-1) creates a visible double-stroke against the panel's outer stroke. Same Phase 3 issue — was the canvas-toolbar P1-1 ever actually fixed? (Should sweep both files together.)
- **Resize behavior** of the history featured tile (P1-4) — needs panel-resize interaction.
- **High-VRAM** appearance (P1-3) — needs mock VRAM at >19.2/24 to see the bar go red while text stays grey.
- **Cancel-running click** (P1-7) — confirm there's no other side effect; `cancel_running` flag is set, only branch is `log::debug!` then comment.
- **Tab-bar width drift** (P1-6) — needs a font screenshot to confirm visible collision between label and count.
- **`placeholder` Response drop** (P1-9) — does egui actually consume the click, or does it bubble? Test by clicking the running-job thumbnail and seeing if Remove on the row beneath fires.
- **`paint_into` clip behavior** (P1-8) — does the 0.5px stroke actually look thinner, or does egui use a different stroke-positioning convention? Visual eyeball.
- **Light theme** appearance of the bottom-strip overlay (`Color32::from_black_alpha(150)` over a possibly bright procedural gradient) — readable in both themes? Mock items have warm hues, OK for now.
- Builder's "0 warnings" claim — `cell_w` should generate one (P1-5). Re-verify with `cargo build` if running.
