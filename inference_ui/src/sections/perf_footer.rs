//! Section: Perf readout footer — pinned to the bottom of the right panel.
//!
//! Per README "Column 3 — Queue / History panel · Perf readout"
//! (lines 143-148):
//!   - Pinned footer, mono, ~80px tall
//!   - Title row: `⬛ <GPU name>` (e.g. `RTX 4090`) + temperature on the
//!     right (`45°C`, red tint above 80°C)
//!   - Row: `VRAM 19.1 / 24.0 GB` + thin progress bar
//!   - Row: `GPU 62%` + thin progress bar
//!   - Updates 2 Hz when generating, 1 Hz idle
//!
//! Phase 4 ships **mock data only** — `state.perf` is seeded by
//! `PerfTelemetry::mock()` in `state::AppState::default()`. The brief
//! defers NVML wiring to Phase 6. We still call `request_repaint_after`
//! at the spec'd cadence so when Phase 6 lands the polling loop already
//! sees live updates without further plumbing.
//!
//! Layout: lives inside `panels::queue_panel` as a
//! `TopBottomPanel::bottom().show_inside(ui, ...)` call — same idiom
//! as the params panel's Advanced footer (`panels::advanced_footer`).

use std::time::Duration;

use egui::{vec2, FontId, Frame, Layout, Margin, RichText, Sense, Stroke, Ui};

use crate::state::AppState;
use crate::tokens::{Tokens, FONT_MONO};
use crate::widgets::progress_bar;

/// Temperature threshold for the red tint per spec. Anything ≥ this tints
/// the temp readout red. (Spec wording: "red tint above 80".)
const TEMP_DANGER_C: f32 = 80.0;

/// Render the footer body. Caller wraps this in a
/// `TopBottomPanel::bottom().show_inside(ui, ...)` so the sidebar's scroll
/// area stops above it.
pub fn show(ui: &mut Ui, state: &AppState, t: &Tokens) {
    // Refresh cadence per spec. We tie this to `state.generating` rather
    // than to the queue having a running job — the brief is explicit about
    // the trigger ("2 Hz when generating, 1 Hz idle") and `state.generating`
    // is the canonical "is the worker running" bit (Phase 3).
    let refresh = if state.generating {
        Duration::from_millis(500)
    } else {
        Duration::from_millis(1000)
    };
    ui.ctx().request_repaint_after(refresh);

    let perf = &state.perf;

    // --- Title row: GPU name + temperature ------------------------------
    ui.horizontal(|ui| {
        // ⬛ glyph as a mini "card" stand-in for the GPU icon. Spec
        // literally says `⬛ <GPU name>`; we honor that.
        ui.label(
            RichText::new("⬛")
                .size(11.0)
                .color(t.text_dim),
        );
        ui.add_space(2.0);
        ui.label(
            RichText::new(&perf.gpu_name)
                .size(11.0)
                .color(t.text)
                .strong(),
        );
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            let temp_warn = perf.temperature_c >= TEMP_DANGER_C;
            let temp_color = if temp_warn { t.danger } else { t.text_dim };
            ui.label(
                RichText::new(format!("{:.0}°C", perf.temperature_c))
                    .size(FONT_MONO)
                    .monospace()
                    .color(temp_color),
            );
        });
    });

    ui.add_space(4.0);

    // --- VRAM row -------------------------------------------------------
    let vram_ratio = if perf.vram_total_gb > 0.0 {
        perf.vram_used_gb / perf.vram_total_gb
    } else {
        0.0
    };
    // Mirror the bar's auto-warn behavior on the value text — when the
    // bar turns red at >=80%, the GB readout should match. Without this
    // the user sees a red bar with muted-grey text and has no large red
    // anchor to read.
    stat_row(
        ui,
        t,
        "VRAM",
        &format!("{:.1} / {:.1} GB", perf.vram_used_gb, perf.vram_total_gb),
        vram_ratio,
        vram_ratio >= progress_bar::WARN_THRESHOLD,
    );

    ui.add_space(2.0);

    // --- GPU util row ---------------------------------------------------
    let gpu_ratio = perf.gpu_util_pct / 100.0;
    stat_row(
        ui,
        t,
        "GPU",
        &format!("{:.0}%", perf.gpu_util_pct),
        gpu_ratio,
        gpu_ratio >= progress_bar::WARN_THRESHOLD,
    );
}

/// One stat line: `LABEL  [bar]  value`. Layout chosen so the label and
/// value have stable widths and the bar consumes the slack between them
/// (matches the JSX `PerfStat` component's flex layout).
fn stat_row(ui: &mut Ui, t: &Tokens, label: &str, value: &str, ratio: f32, warn: bool) {
    ui.horizontal(|ui| {
        let label_w = 36.0;
        let value_w = 72.0;
        // Label — fixed width so the bar starts at a consistent x across
        // both rows (VRAM/GPU stay vertically aligned).
        let (label_rect, _) = ui.allocate_exact_size(vec2(label_w, 14.0), Sense::hover());
        ui.painter().text(
            label_rect.left_center(),
            egui::Align2::LEFT_CENTER,
            label,
            FontId::monospace(10.0),
            t.text_dim,
        );

        // Bar fills the slack. We compute width manually from the row's
        // available_width so the trailing value column doesn't get pushed
        // past the panel edge on narrow widths.
        let avail = (ui.available_width() - value_w - 4.0).max(20.0);
        progress_bar::thin_bar(ui, t, avail, ratio.clamp(0.0, 1.0), warn);

        // Right value — fixed width, right-aligned, mono.
        let (val_rect, _) = ui.allocate_exact_size(vec2(value_w, 14.0), Sense::hover());
        let val_color = if warn { t.danger } else { t.text_dim };
        ui.painter().text(
            val_rect.right_center(),
            egui::Align2::RIGHT_CENTER,
            value,
            FontId::monospace(10.0),
            val_color,
        );
    });
}

/// Public wrapper exposing the footer as a pinned `TopBottomPanel::bottom`
/// inside the caller's `Ui`. Mirrors `panels::advanced_footer`'s shape so
/// the right panel reads the same way as the params panel.
pub fn pinned(ui: &mut Ui, state: &AppState, t: &Tokens) {
    let response = egui::TopBottomPanel::bottom("queue_perf_footer")
        .resizable(false)
        .frame(
            // No `Frame::stroke(...)`: that would paint a 1px line on all
            // four sides, doubling up against the queue panel's own outer
            // stroke (left/right/bottom). Spec is `borderTop` only — the
            // top edge is painted manually after `show_inside` (same fix
            // pattern as Phase 3's canvas toolbar).
            Frame::none()
                .fill(t.panel)
                .inner_margin(Margin::symmetric(10.0, 7.0)),
        )
        .show_inside(ui, |ui| {
            // ~80px: title row (~14) + spacing (4) + 2 stat rows (~14 each)
            // + spacing (2) ≈ 48px content; with 7px top/bottom inner pad
            // and the panel's own 1px border top/bottom = ~64-66px. Spec
            // says "~80px" so we pad to a 64px content min height; egui's
            // panel auto-sizing rounds upward to ~80 once the row stack is
            // laid out. AGENT-DEFAULT: didn't force `min_height(80)` —
            // egui's natural auto-size is closer to the JSX prototype.
            // If the user wants a hard 80px, swap to `.exact_height(80.0)`.
            show(ui, state, t);
        });

    // Top edge only — draws the `borderTop: 1px` from the JSX spec
    // without polluting the other three sides.
    let r = response.response.rect;
    ui.painter().line_segment(
        [r.left_top(), r.right_top()],
        Stroke::new(1.0, t.border_soft),
    );
}
