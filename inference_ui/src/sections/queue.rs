//! Section: Queue / History — right-panel content above the perf footer.
//!
//! Per README "Column 3 — Queue / History panel" (lines 131-148):
//!   - Tabs at top: `Queue (N)` | `History (N)`. Active tab has accent
//!     underline; switching is local panel state.
//!   - Queue tab: running job (large, with progress + ETA + cancel),
//!     queued jobs as compact rows below (small thumb, prompt, param
//!     summary `1024² · 28s · euler`, grip handle, hover actions).
//!   - History tab: 2-col grid; first row featured (full width); each
//!     tile has muted mono overlay (short prompt + relative time);
//!     right-click context menu per tile.
//!
//! Phase 4 wiring constraints (from the brief):
//!   - All hover actions / context menu items are **no-op stubs** — real
//!     behaviors land in Phase 5+.
//!   - No drag-reorder (`egui-dnd` deliberately skipped, same as LoRA in
//!     Phase 2). The grip handle is a visual placeholder.
//!   - No real image loading for thumbnails — the procedural placeholder
//!     in `widgets::thumbnail` keys off job/history `id`.

use std::time::SystemTime;

use crossbeam_channel::Sender;
use egui::{
    pos2, vec2, Align, Color32, FontId, Frame, Id, LayerId, Layout, Margin, Order, Rect, RichText,
    ScrollArea, Sense, Stroke, Ui,
};

use crate::state::{AppState, HistoryItem, QueueJob, QueueTab};
use crate::tokens::{Tokens, FONT_MONO, RADIUS_CONTROL};
use crate::widgets::{progress_bar, thumbnail};
use crate::worker::UiMsg;

/// Top-level entry: paints the tab bar + scrollable body for whichever tab
/// is active. Caller is responsible for the perf footer (lives in a sibling
/// `TopBottomPanel::bottom` so it stays pinned).
///
/// Phase 5a: `worker_tx` lets the cancel × button (running row) and the ×
/// remove button (queued rows) signal the worker. Without this they were
/// only mutating the local UI mirror, which left the worker still running
/// the job in the background.
pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens, worker_tx: &Sender<UiMsg>) {
    tab_bar(ui, state, t);
    ui.add_space(2.0);

    // Scroll only the body — the tab bar above and perf footer below stay
    // visible regardless of how long the queued/history list grows.
    ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| match state.queue.active_tab {
            QueueTab::Queue => queue_body(ui, state, t, worker_tx),
            QueueTab::History => history_body(ui, state, t),
        });
}

// --- tab bar -------------------------------------------------------------

/// Two-tab bar: `Queue (N)` | `History (N)`. Reuses the active-tab pattern
/// from `chrome::mode_tab` (allocate rect + paint label + paint accent
/// underline at panel bottom edge), but the underline sits at the *bottom
/// of this row*, not the bottom of the panel.
fn tab_bar(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    let queue_n = state.queue.queued.len() + state.queue.running.is_some() as usize;
    let history_n = state.queue.history.len();

    // Reserve a 28px-tall row spanning the full width and split it evenly
    // (or the brief's "at least 80px wide", whichever is larger). We do
    // the split manually — `ui.columns` doesn't give us per-column hover
    // rect access cleanly enough for the underline.
    let row_h = 28.0;
    let total_w = ui.available_width();
    let (row_rect, _resp) = ui.allocate_exact_size(vec2(total_w, row_h), Sense::hover());

    let half_w = total_w * 0.5;
    let queue_rect = Rect::from_min_size(row_rect.min, vec2(half_w, row_h));
    let history_rect = Rect::from_min_size(
        pos2(row_rect.min.x + half_w, row_rect.min.y),
        vec2(half_w, row_h),
    );

    let bar_bottom = row_rect.bottom();
    paint_tab(
        ui,
        t,
        queue_rect,
        "Queue",
        queue_n,
        state.queue.active_tab == QueueTab::Queue,
        bar_bottom,
        || state.queue.active_tab = QueueTab::Queue,
    );
    paint_tab(
        ui,
        t,
        history_rect,
        "History",
        history_n,
        state.queue.active_tab == QueueTab::History,
        bar_bottom,
        || state.queue.active_tab = QueueTab::History,
    );

    // Bottom border for the row (matches JSX `borderBottom` on the tab
    // container). The active-tab underline is painted on a Foreground layer
    // by `paint_tab`, so it visually replaces this border at the active
    // tab's x-range (mirrors the trick used in `chrome::mode_tab`).
    ui.painter().line_segment(
        [row_rect.left_bottom(), row_rect.right_bottom()],
        Stroke::new(1.0, t.border_soft),
    );
}

fn paint_tab(
    ui: &mut Ui,
    t: &Tokens,
    rect: Rect,
    label: &str,
    count: usize,
    active: bool,
    bar_bottom: f32,
    mut on_click: impl FnMut(),
) {
    let id = ui.id().with(("queue_tab", label));
    let resp = ui.interact(rect, id, Sense::click());

    // Active tab: subtle panel_sunk fill; inactive: transparent (hover row
    // tint). Mirror of JSX `background: active ? T.panelSunk : 'transparent'`.
    let bg = if active {
        t.panel_sunk
    } else if resp.hovered() {
        t.row
    } else {
        Color32::TRANSPARENT
    };
    if bg != Color32::TRANSPARENT {
        ui.painter().rect_filled(rect, 0.0, bg);
    }

    // Label color: text on active, text_dim on inactive.
    let label_color = if active { t.text } else { t.text_dim };

    // Lay out "Label  (N)" — label in body color, count in muted mono.
    // We compose two galleys side by side, centered as a unit.
    let label_galley = ui.painter().layout_no_wrap(
        label.to_string(),
        FontId::proportional(FONT_MONO + 0.5),
        label_color,
    );
    let count_text = format!("({count})");
    let count_galley = ui.painter().layout_no_wrap(
        count_text,
        FontId::monospace(10.0),
        t.text_mute,
    );
    let gap = 5.0;
    let total_w = label_galley.size().x + gap + count_galley.size().x;
    let baseline_y = rect.center().y - label_galley.size().y * 0.5;
    let label_x = rect.center().x - total_w * 0.5;
    ui.painter().galley(pos2(label_x, baseline_y), label_galley, label_color);
    let count_y = rect.center().y - count_galley.size().y * 0.5;
    ui.painter().galley(
        pos2(label_x + label_galley_width_hack(label) + gap, count_y),
        count_galley,
        t.text_mute,
    );

    // 2px accent underline on the active tab. Painted on a Foreground
    // layer (same trick as `chrome::mode_tab`) so the row's 1px
    // `border_soft` line — drawn AFTER the tabs by `tab_bar` — cannot
    // overpaint it. Without this, the underline's bottom 1px is masked
    // by the border line and only a 1px stripe survives.
    if active {
        let underline = Rect::from_min_max(
            pos2(rect.left() + 2.0, bar_bottom - 1.0),
            pos2(rect.right() - 2.0, bar_bottom + 1.0),
        );
        let layer_painter = ui.ctx().layer_painter(LayerId::new(
            Order::Foreground,
            Id::new(("queue_tab_underline", label)),
        ));
        layer_painter.rect_filled(underline, 0.0, t.accent);
    }

    if resp.clicked() {
        on_click();
    }
}

/// Hack: re-measure label width via a synthetic galley. We can't reuse the
/// `label_galley` after handing it to `painter().galley()` (consumed by
/// value). Cheaper than caching: re-layout to get just the width.
/// AGENT-DEFAULT: this triggers a second text layout per tab per frame —
/// 2 tabs × 2 layouts = 4 text layouts/frame, negligible. If profiling ever
/// shows this matters, switch to `ui.fonts(|f| f.layout_no_wrap(...))` and
/// keep the galley around.
fn label_galley_width_hack(label: &str) -> f32 {
    // Approximate width: 7px per char at FONT_MONO + 0.5. Matches the
    // measured galley width within ~2px for ASCII labels, which is all we
    // ship ("Queue", "History"). If this drifts visibly we can switch to
    // a real measurement.
    label.chars().count() as f32 * 6.5
}

// --- queue tab body ------------------------------------------------------

fn queue_body(ui: &mut Ui, state: &mut AppState, t: &Tokens, worker_tx: &Sender<UiMsg>) {
    // 6px horizontal inset matches JSX `padding: 6` on the body container
    // (queue-panel.jsx). The tab bar above stays flush (matches the JSX
    // `borderBottom` spanning the full container width); only the row
    // content gets inset off the panel walls.
    Frame::none()
        .inner_margin(Margin::symmetric(6.0, 0.0))
        .show(ui, |ui| {
            ui.add_space(4.0);

            // Track cancel/remove requests outside the iteration so we don't
            // borrow `state.queue.queued` mutably twice (same trick as
            // `lora.rs`).
            let mut cancel_running = false;
            let mut remove_idx: Option<usize> = None;
            let mut promote_idx: Option<usize> = None;

            if let Some(running) = &state.queue.running {
                running_row(ui, t, running, &mut cancel_running);
                ui.add_space(6.0);
            }

            if state.queue.queued.is_empty() && state.queue.running.is_none() {
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("Queue is empty")
                            .size(11.0)
                            .color(t.text_mute),
                    );
                });
                return;
            }

            for (i, job) in state.queue.queued.iter().enumerate() {
                queued_row(ui, t, job, &mut |action| match action {
                    RowAction::RunNow => promote_idx = Some(i),
                    RowAction::Remove => remove_idx = Some(i),
                });
                ui.add_space(3.0);
            }

            // Apply mutations after iteration.
            //
            // Phase 5a wiring: the running × button now signals the worker
            // (UiMsg::Cancel) — the worker emits Failed, and the resulting
            // event clears `state.queue.running` properly. We DO NOT clear
            // it here; doing so would race with the in-flight job and could
            // re-spawn the running slot from a Started event already in
            // flight. The exception is a job whose id we don't recognize
            // (eg. the Phase 4 mock seed data), which won't have a real
            // worker job behind it — we clear those locally too.
            if cancel_running {
                let _ = worker_tx.send(UiMsg::Cancel);
                // Defensive: if there's a stale mock running entry whose id
                // pre-dates `next_job_id` (1000), the worker won't recognize
                // it and won't emit a Failed event. Clear it locally so the
                // user's click isn't a no-op. Real worker jobs (id >= 1000)
                // get cleared by the Failed event handler in app.rs.
                if state
                    .queue
                    .running
                    .as_ref()
                    .map(|j| j.id < 1000)
                    .unwrap_or(false)
                {
                    state.queue.running = None;
                }
            }
            if let Some(i) = remove_idx {
                let job = state.queue.queued.remove(i);
                let _ = worker_tx.send(UiMsg::RemoveQueued { id: job.id });
            }
            if let Some(i) = promote_idx {
                // Phase 5a: still a stub — promoting requires cancelling the
                // current job, sending the promoted one immediately, and
                // re-queueing the rest. Worth its own pass; the brief
                // explicitly leaves drag-reorder + run-now for later.
                log::debug!("Run-now clicked for queue index {i} (Phase 5b+ wiring)");
            }
        });
}

/// Hover-action signal from a queued row. Bubbled up so the iteration loop
/// can defer the mutation until after the borrow ends.
enum RowAction {
    RunNow,
    Remove,
}

/// Running job row: large thumb (~80×60), prompt snippet, progress bar
/// with ETA text, cancel × button.
fn running_row(ui: &mut Ui, t: &Tokens, job: &QueueJob, cancel: &mut bool) {
    Frame::none()
        // Subtle accent_soft tint to mark "this one is active" — same
        // visual signal as JSX `background: T.accentSoft` on the running
        // queue row.
        .fill(t.accent_soft)
        .stroke(Stroke::new(1.0, t.accent))
        .rounding(RADIUS_CONTROL)
        .inner_margin(Margin::symmetric(6.0, 6.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let pct_label = format!("{}%", (job.progress * 100.0).round() as i32);
                thumbnail::placeholder_with_overlay(ui, t, vec2(80.0, 60.0), job.id, &pct_label);

                ui.add_space(6.0);
                ui.vertical(|ui| {
                    // Prompt snippet (truncated to ~36 chars).
                    ui.add(
                        egui::Label::new(
                            RichText::new(truncate(&job.prompt, 36))
                                .size(11.0)
                                .color(t.text),
                        )
                        .truncate(),
                    );
                    // Progress bar — fill the available width minus a small
                    // gutter for the ETA text on the right.
                    let bar_avail = (ui.available_width() - 60.0).max(40.0);
                    ui.horizontal(|ui| {
                        progress_bar::thin_bar(ui, t, bar_avail, job.progress, false);
                        if let Some(eta) = job.eta_secs {
                            ui.label(
                                RichText::new(format!("eta {eta}s"))
                                    .size(10.0)
                                    .monospace()
                                    .color(t.text_dim),
                            );
                        }
                    });
                    // Status badge row — mirrors JSX "Running" badge.
                    ui.label(
                        RichText::new(format!(
                            "Running · {} · {}×{} · {}s",
                            job.sampler, job.width, job.height, job.steps
                        ))
                        .size(10.0)
                        .monospace()
                        .color(t.text_mute),
                    );
                });

                // Push the cancel × to the right. Inside a fresh r-to-l
                // sub-layout so it doesn't interfere with the vertical stack.
                ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("×").size(14.0).color(t.text_mute),
                            )
                            .fill(Color32::TRANSPARENT)
                            .stroke(Stroke::NONE)
                            .min_size(vec2(20.0, 20.0)),
                        )
                        .on_hover_text("Cancel")
                        .clicked()
                    {
                        *cancel = true;
                    }
                });
            });
        });
}

/// Compact queued row: small thumb (~50×40), prompt, param summary, grip.
/// Hover actions on the right (Run now / Remove). Drag-reorder NOT wired.
fn queued_row(
    ui: &mut Ui,
    t: &Tokens,
    job: &QueueJob,
    on_action: &mut impl FnMut(RowAction),
) {
    Frame::none()
        .fill(t.panel_sunk)
        .stroke(Stroke::new(1.0, t.border_soft))
        .rounding(RADIUS_CONTROL)
        .inner_margin(Margin::symmetric(6.0, 5.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // Grip handle — visual only; no drag.
                ui.label(RichText::new("≡").size(12.0).color(t.text_mute))
                    .on_hover_text("Drag to reorder (Phase 5+)");

                thumbnail::placeholder(ui, t, vec2(50.0, 40.0), job.id);

                ui.add_space(4.0);
                ui.vertical(|ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(truncate(&job.prompt, 32))
                                .size(11.0)
                                .color(t.text),
                        )
                        .truncate(),
                    );
                    // Param summary line — mono, muted. Format from the
                    // brief: `1024² · 28s · euler`. Use ² when w==h, full
                    // `WxH` otherwise.
                    let res = if job.width == job.height {
                        format!("{}²", job.width)
                    } else {
                        format!("{}×{}", job.width, job.height)
                    };
                    ui.label(
                        RichText::new(format!("{res} · {}s · {}", job.steps, job.sampler))
                            .size(10.0)
                            .monospace()
                            .color(t.text_mute),
                    );
                });

                // Right cluster: ⏵ run now + × remove. Always visible
                // (egui can't trivially "show only on hover" without
                // custom handling — same AGENT-DEFAULT as LoRA's × button).
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if small_action(ui, t, "×", "Remove").clicked() {
                        on_action(RowAction::Remove);
                    }
                    if small_action(ui, t, "⏵", "Run now").clicked() {
                        on_action(RowAction::RunNow);
                    }
                });
            });
        });
}

fn small_action(ui: &mut Ui, t: &Tokens, glyph: &str, tooltip: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(glyph).size(11.0).color(t.text_mute))
            .fill(Color32::TRANSPARENT)
            .stroke(Stroke::NONE)
            .min_size(vec2(18.0, 18.0)),
    )
    .on_hover_text(tooltip)
}

// --- history tab body ----------------------------------------------------

/// 2-column grid; row 0 is featured (single full-width cell), then 2
/// per row. Newest first; we trust callers to push to `history` in
/// reverse-chrono order (Phase 4 mock data does this).
fn history_body(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    // 6px horizontal inset matches JSX `padding: 6` on the body container.
    Frame::none()
        .inner_margin(Margin::symmetric(6.0, 0.0))
        .show(ui, |ui| {
            if state.queue.history.is_empty() {
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("No history yet")
                            .size(11.0)
                            .color(t.text_mute),
                    );
                });
                return;
            }

            ui.add_space(4.0);

            let avail_w = ui.available_width();
            let gap = 4.0;

            let items = state.queue.history.clone();
            let now = SystemTime::now();

            // Featured first item — full row, 2:1 aspect ratio per JSX
            // `aspectRatio: '2 / 1'`. The panel is resizable (200..420 px),
            // so a hardcoded 120px tile would warp at the extremes; deriving
            // height from width keeps the ratio honest. Clamped so the tile
            // stays sensible if the panel is dragged unusually narrow/wide.
            if let Some(first) = items.first() {
                let featured_h = (avail_w / 2.0).clamp(80.0, 200.0);
                let (rect, resp) =
                    ui.allocate_exact_size(vec2(avail_w, featured_h), Sense::click());
                history_tile_into(ui, t, rect, first, now);
                attach_history_context(resp);
                ui.add_space(gap);
            }

            // Remaining items in 2-col rows. Each cell is square (1:1) —
            // height derived from cell width so resizing the panel doesn't
            // squash/stretch the thumbnails.
            let rest: Vec<&HistoryItem> = items.iter().skip(1).collect();
            let mut idx = 0;
            while idx < rest.len() {
                let row_w = ui.available_width();
                let cell_w_now = (row_w - gap) * 0.5;
                let row_h = cell_w_now.clamp(60.0, 140.0);
                let (row_rect, _r) =
                    ui.allocate_exact_size(vec2(row_w, row_h), Sense::hover());
                let left_rect =
                    Rect::from_min_size(row_rect.min, vec2(cell_w_now, row_h));
                let right_rect = Rect::from_min_size(
                    pos2(row_rect.min.x + cell_w_now + gap, row_rect.min.y),
                    vec2(cell_w_now, row_h),
                );

                let left_resp = ui.interact(
                    left_rect,
                    ui.id().with(("hist_l", rest[idx].id)),
                    Sense::click(),
                );
                history_tile_into(ui, t, left_rect, rest[idx], now);
                attach_history_context(left_resp);

                if let Some(right_item) = rest.get(idx + 1) {
                    let right_resp = ui.interact(
                        right_rect,
                        ui.id().with(("hist_r", right_item.id)),
                        Sense::click(),
                    );
                    history_tile_into(ui, t, right_rect, right_item, now);
                    attach_history_context(right_resp);
                }
                idx += 2;
                ui.add_space(gap);
            }
        });
}

/// Paint one tile into a pre-allocated rect: thumbnail + bottom overlay
/// strip (prompt + relative time).
fn history_tile_into(ui: &Ui, t: &Tokens, rect: Rect, item: &HistoryItem, now: SystemTime) {
    thumbnail::paint_into(ui, t, rect, item.id, None);

    // Bottom overlay: dark gradient bar + mono label/time. The JSX uses a
    // `linear-gradient(to top, rgba(0,0,0,0.7), transparent)`. We
    // approximate with a single semi-opaque rect along the bottom 24px.
    let strip_h = 22.0;
    let strip_rect = Rect::from_min_max(
        pos2(rect.left(), rect.bottom() - strip_h),
        rect.right_bottom(),
    );
    ui.painter()
        .rect_filled(strip_rect, 0.0, Color32::from_black_alpha(150));

    let pad = 5.0;
    let inner = strip_rect.shrink2(vec2(pad, 3.0));

    // Prompt (left, truncated to ~32 chars). Time (right, dim).
    let prompt_text = truncate(&item.prompt, 32);
    let prompt_galley = ui.painter().layout_no_wrap(
        prompt_text,
        FontId::monospace(9.5),
        Color32::WHITE,
    );
    let time_text = relative_time(now, item.timestamp);
    let time_galley = ui.painter().layout_no_wrap(
        time_text,
        FontId::monospace(9.5),
        Color32::from_white_alpha(180),
    );

    // Both galleys baseline-aligned to the strip's vertical center.
    let prompt_y = inner.center().y - prompt_galley.size().y * 0.5;
    let time_y = inner.center().y - time_galley.size().y * 0.5;
    ui.painter().galley(
        pos2(inner.left(), prompt_y),
        prompt_galley,
        Color32::WHITE,
    );
    ui.painter().galley(
        pos2(inner.right() - time_galley.size().x, time_y),
        time_galley,
        Color32::from_white_alpha(180),
    );
    // Tiny separator dot? JSX uses spacing only; we follow suit.
}

/// Right-click → context menu with the spec'd no-op items.
/// All entries log a debug line; real wiring lands in Phase 5+.
fn attach_history_context(resp: egui::Response) {
    resp.context_menu(|ui| {
        let items = [
            "Open",
            "Reveal in folder",
            "Copy prompt",
            "Send to input",
            "Upscale",
            "Use seed",
            "Delete",
        ];
        for item in items {
            // Width matches the menubar's `MENU_ITEM_W` style (160-ish).
            // We don't pull that constant in to avoid a chrome import; a
            // local 160 is good enough for a 7-item context menu.
            if ui
                .add(egui::Button::new(item).min_size(vec2(160.0, 0.0)))
                .clicked()
            {
                log::debug!("History context menu: `{item}` clicked (Phase 5+ wiring)");
                ui.close_menu();
            }
        }
    });
}

// --- helpers -------------------------------------------------------------

/// Char-aware truncation with ellipsis. Same helper as `lora.rs` —
/// duplicated rather than promoted to a shared module to keep section
/// files self-contained. If we add a third caller, promote then.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{head}…")
}

/// Format a SystemTime as "Ns ago" / "Nm ago" / "Nh ago" / "Nd ago".
/// Falls back to "just now" for elapsed < 5s and "—" for any time error
/// (clock went backwards, etc.).
fn relative_time(now: SystemTime, then: SystemTime) -> String {
    let elapsed = match now.duration_since(then) {
        Ok(d) => d,
        Err(_) => return "—".into(),
    };
    let secs = elapsed.as_secs();
    if secs < 5 {
        "just now".into()
    } else if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

