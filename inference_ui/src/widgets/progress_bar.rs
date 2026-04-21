//! Thin progress bar — used by the queue's running-job row and the
//! perf footer's VRAM/GPU rows.
//!
//! Per README "Column 3 — Queue / History panel" + the JSX `PerfStat`
//! component (`queue-panel.jsx:162-171`):
//!   - 4-6px tall (we ship 4px to match the JSX exactly)
//!   - background: `panel_sunk` with a 1px `border_soft` outline
//!   - fill: `accent` for normal, `danger` once `value >= warn_threshold`
//!     (the brief specifies "danger if value > 80% of max" — callers pass
//!     a 0.0..=1.0 ratio, so the warn threshold is just `0.80`)
//!   - rounded 2px (matches JSX `borderRadius: 2`)
//!
//! AGENT-DEFAULT: the egui built-in `egui::ProgressBar` paints a chunky
//! 18px bar with embedded text and animation — totally wrong for a
//! 4px-tall ribbon. Hand-rolled rect_filled is the right call here.

use egui::{vec2, Rect, Sense, Stroke, Ui};

use crate::tokens::Tokens;

/// Default warn threshold: 80% of full per the brief.
pub const WARN_THRESHOLD: f32 = 0.80;

/// Paint a thin progress bar of the given width. Allocates the full width
/// (caller picks how much horizontal room) and a fixed 4px height.
///
/// `progress` is clamped to [0.0, 1.0] before painting.
/// Pass `warn = true` to force the danger color regardless of value (the
/// perf footer uses this for the temperature row even though temperature
/// isn't a "fraction of max" semantically).
pub fn thin_bar(ui: &mut Ui, t: &Tokens, width: f32, progress: f32, warn: bool) {
    let progress = progress.clamp(0.0, 1.0);
    let height = 4.0;
    let (rect, _resp) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    paint(ui, t, rect, progress, warn);
}

/// Paint into a pre-allocated `Rect`. Useful when the caller has already
/// reserved space (e.g. inside a layout that mixes the bar with sibling
/// labels). Same color rules as `thin_bar`.
pub fn paint(ui: &Ui, t: &Tokens, rect: Rect, progress: f32, warn: bool) {
    let progress = progress.clamp(0.0, 1.0);
    let radius = 2.0;
    let painter = ui.painter_at(rect);

    // Background trough.
    painter.rect(rect, radius, t.panel_sunk, Stroke::new(1.0, t.border_soft));

    if progress <= 0.0 {
        return;
    }

    // Filled portion. Use `danger` when warn is set OR when value crosses
    // the threshold; the brief covers both — perf temp row passes warn
    // explicitly, while VRAM/GPU rely on the auto-threshold.
    let fill_color = if warn || progress >= WARN_THRESHOLD {
        t.danger
    } else {
        t.accent
    };

    // Inset by 1px so the fill sits inside the border stroke (otherwise
    // a 100% bar visually overshoots the trough by half a pixel).
    let inner = rect.shrink(1.0);
    if inner.width() <= 0.0 || inner.height() <= 0.0 {
        return;
    }
    let mut fill_rect = inner;
    fill_rect.set_width(inner.width() * progress);
    painter.rect_filled(fill_rect, radius, fill_color);
}

/// Convenience: percentage form (0..=100) for callers that already speak in
/// human percent. Currently unused — `perf_footer::stat_row` does the
/// `pct / 100.0` conversion inline since it also needs the formatted string.
/// Kept around as forward-looking API for Phase 6 NVML wiring (which will
/// likely route raw 0..=100 utilization through here).
#[allow(dead_code)]
pub fn thin_bar_pct(ui: &mut Ui, t: &Tokens, width: f32, pct: f32, warn: bool) {
    thin_bar(ui, t, width, pct / 100.0, warn);
}

/// Convenience: ratio form for a `used / total` pair (perf-footer VRAM row).
/// Returns 0 when `total <= 0` so a missing-NVML state degrades gracefully
/// instead of NaN-painting. Currently unused for the same reason as
/// `thin_bar_pct` — `stat_row` inlines the divide so it can also format
/// the textual value. Forward-looking API for Phase 6.
#[allow(dead_code)]
pub fn thin_bar_ratio(ui: &mut Ui, t: &Tokens, width: f32, used: f32, total: f32, warn: bool) {
    let p = if total > 0.0 { used / total } else { 0.0 };
    thin_bar(ui, t, width, p, warn);
}
