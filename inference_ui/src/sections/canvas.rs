//! Section: Canvas — checker background + preview area + scan-line + badges +
//! toolbar.
//!
//! Per README "Column 2 — Canvas panel · Canvas" (lines 123-129):
//!   - Transparent checker background, 8px tiles, `#2a2a2e` / `#1e1e22`.
//!   - Preview fills the area with `object-fit: contain`. Aspect-ratio
//!     locked to current W×H from `state.current()`.
//!   - While running: scan-line overlay sweeping top→bottom (~2s loop) +
//!     placeholder gradient for the latent preview.
//!   - Bottom-left badge: monospace `image · 1024×1024` (or
//!     `video · 1280×720 · 81f @ 16fps`).
//!   - Bottom-right badge: monospace `seed <seed>`.
//!   - Canvas toolbar (32px tall) — Fit / 100% / Copy / Open folder /
//!     Send to input · spacer · idle / step counter.
//!
//! Phase 3 stubs: toolbar buttons no-op; preview is a simple gradient when
//! generating. Real preview pixels arrive via the worker channel in Phase 5.

use std::time::Duration;

use egui::{
    pos2, vec2, Align, Align2, Color32, FontId, Frame, Layout, Margin, Pos2, Rect, RichText,
    Sense, Shape, Stroke, Ui,
};

use crate::state::{AppState, Mode};
use crate::theme::Theme;
use crate::tokens::{
    Tokens, CHECKER_DARK, CHECKER_DARK_LIGHT_THEME, CHECKER_LIGHT, CHECKER_LIGHT_LIGHT_THEME,
    CHECKER_TILE,
};

/// Pick the checker color pair based on the active theme. Spec gives literal
/// hexes for the dark theme; light-theme analogues are AGENT-DEFAULT
/// (documented in `tokens.rs`).
fn checker_pair(theme: Theme) -> (Color32, Color32) {
    match theme {
        Theme::Dark => (CHECKER_DARK, CHECKER_LIGHT),
        Theme::Light => (CHECKER_DARK_LIGHT_THEME, CHECKER_LIGHT_LIGHT_THEME),
    }
}

/// Top-level entry: canvas surface + toolbar stacked vertically.
///
/// Phase 5a: `last_image` is the GPU-uploaded most-recent worker result. When
/// present and we're not currently generating, it replaces the warm gradient
/// placeholder in the preview rect. While generating, the gradient (+ scan
/// line) takes over again — the running result isn't shown until Done. Phase
/// 5b will populate `WorkerEvent::Preview` and we'll show the in-flight
/// latent decode here too.
pub fn show(
    ui: &mut Ui,
    state: &mut AppState,
    t: &Tokens,
    last_image: Option<&egui::TextureHandle>,
) {
    // Reserve 32px at the bottom for the toolbar; the rest is the canvas.
    let total_avail = ui.available_size();
    let toolbar_h = 32.0;
    let canvas_h = (total_avail.y - toolbar_h).max(120.0);

    canvas_surface(ui, state, t, canvas_h, last_image);
    toolbar(ui, state, t);
}

fn canvas_surface(
    ui: &mut Ui,
    state: &mut AppState,
    t: &Tokens,
    height: f32,
    last_image: Option<&egui::TextureHandle>,
) {
    let (rect, surface_resp) =
        ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click());
    if surface_resp.double_clicked() && last_image.is_some() {
        state.image_zoomed = true;
    }

    let painter = ui.painter_at(rect);

    // --- Checker background ----------------------------------------------
    // Approach: paint the dark fill across the whole rect, then overlay
    // alternating light tiles. At 8px tiles a 1200×800 canvas yields ~7500
    // tiles total (~3750 light overlays). egui batches these into a single
    // mesh per draw call so it stays O(N) but cheap. If this turns out to
    // dominate frame time we can switch to a `Mesh` with two triangles per
    // tile or tile a pre-rendered texture — but premature optimization.
    let (dark, light) = checker_pair(state.theme);
    painter.rect_filled(rect, 0.0, dark);

    // Build a Shape::Mesh? Simpler: extend with rect_filled per tile. Egui's
    // `Painter::extend` takes `Vec<Shape>` and appends in one batch.
    let mut shapes: Vec<Shape> = Vec::new();
    let tile = CHECKER_TILE;
    let cols = (rect.width() / tile).ceil() as i32 + 1;
    let rows = (rect.height() / tile).ceil() as i32 + 1;
    for row in 0..rows {
        for col in 0..cols {
            // Checker pattern: light tile when (row + col) is odd.
            if (row + col) & 1 != 1 {
                continue;
            }
            let x0 = rect.left() + col as f32 * tile;
            let y0 = rect.top() + row as f32 * tile;
            // Clip the tile to the rect bounds (cheap intersect).
            let tile_rect = Rect::from_min_size(pos2(x0, y0), vec2(tile, tile))
                .intersect(rect);
            if tile_rect.width() <= 0.0 || tile_rect.height() <= 0.0 {
                continue;
            }
            shapes.push(Shape::rect_filled(tile_rect, 0.0, light));
        }
    }
    painter.extend(shapes);

    // --- Aspect-locked preview area --------------------------------------
    // `object-fit: contain` semantics: shrink the (W, H) box uniformly
    // until it fits the canvas with padding. Center it.
    let cn = state.current();
    let aspect = cn.width as f32 / cn.height.max(1) as f32;
    let pad = 16.0;
    let avail_w = (rect.width() - pad * 2.0).max(40.0);
    let avail_h = (rect.height() - pad * 2.0).max(40.0);
    let (pw, ph) = if avail_w / aspect <= avail_h {
        (avail_w, avail_w / aspect)
    } else {
        (avail_h * aspect, avail_h)
    };
    let preview_rect = Rect::from_center_size(rect.center(), vec2(pw, ph));

    if state.generating {
        // Placeholder gradient — stand-in for a decoded latent. Real preview
        // pixels arrive from the worker in Phase 5b (currently None). Phase
        // 5a still uses the warm-gradient bands here so something is visibly
        // happening while the mock worker counts down.
        //
        // AGENT-DEFAULT: when `last_image` is also Some (a previous job
        // completed), we still favor the gradient + scan line during a new
        // generation — the in-progress affordance dominates the "look at
        // your last result" affordance. The just-completed result reappears
        // when this job ends.
        let bands = [
            Color32::from_rgb(0x3a, 0x28, 0x14),
            Color32::from_rgb(0x6b, 0x40, 0x18),
            Color32::from_rgb(0xa0, 0x5e, 0x1f),
            Color32::from_rgb(0xd0, 0x82, 0x2b),
            Color32::from_rgb(0xe8, 0xa3, 0x3d),
            Color32::from_rgb(0xf2, 0xc0, 0x6b),
        ];
        let band_h = preview_rect.height() / bands.len() as f32;
        for (i, c) in bands.iter().enumerate() {
            let band = Rect::from_min_size(
                pos2(
                    preview_rect.left(),
                    preview_rect.top() + i as f32 * band_h,
                ),
                vec2(preview_rect.width(), band_h + 0.5),
            );
            painter.rect_filled(band, 0.0, *c);
        }
        // Subtle vignette stroke around the preview to lift it off the
        // checker bg.
        painter.rect_stroke(
            preview_rect,
            2.0,
            Stroke::new(1.0, Color32::from_black_alpha(140)),
        );

        // --- Scan-line overlay -----------------------------------------
        // Use ctx.input(|i| i.time) for continuous animation, then request a
        // repaint at ~60Hz so the line keeps moving.
        let time = ui.ctx().input(|i| i.time);
        let cycle_secs: f64 = 2.0;
        let y_frac = ((time % cycle_secs) / cycle_secs) as f32;
        let y = preview_rect.top() + preview_rect.height() * y_frac;
        let scan_color = t.accent.gamma_multiply(0.6);
        painter.line_segment(
            [pos2(preview_rect.left(), y), pos2(preview_rect.right(), y)],
            Stroke::new(1.5, scan_color),
        );
        // Soft glow band — a wider, fainter line behind the main scan line.
        let glow_color = t.accent.gamma_multiply(0.25);
        painter.line_segment(
            [pos2(preview_rect.left(), y), pos2(preview_rect.right(), y)],
            Stroke::new(6.0, glow_color),
        );
        ui.ctx().request_repaint_after(Duration::from_millis(16));
    } else if let Some(tex) = last_image {
        // Idle + we have a completed result → paint it into the preview rect.
        // egui's `Image::paint_at` does the texture sampling; aspect-fitting
        // is already handled by `preview_rect` above (which is sized from
        // the configured W:H, matching the texture's source aspect ratio
        // because we generated it at exactly that ratio in mock.rs).
        //
        // We use `Image::from_texture` rather than the lower-level
        // `painter.image(tex.id(), ..., Color32::WHITE)` so we get egui's
        // built-in handling of texture options + min-mag filter.
        egui::Image::from_texture(tex).paint_at(ui, preview_rect);
        // Same vignette stroke as the in-flight path so the result has a
        // consistent silhouette against the checker bg.
        painter.rect_stroke(
            preview_rect,
            2.0,
            Stroke::new(1.0, Color32::from_black_alpha(140)),
        );
    }

    // --- Badges (always rendered, on top of preview/checker) -------------
    let mode = state.tab;
    let bottom_left_text = match mode {
        Mode::Image => format!("image · {}×{}", cn.width, cn.height),
        Mode::Video => {
            let frames = cn.frames.unwrap_or(0);
            let fps = cn.fps.unwrap_or(0);
            format!("video · {}×{} · {}f @ {}fps", cn.width, cn.height, frames, fps)
        }
    };
    let bottom_right_text = format!("seed {}", state.seed);

    paint_badge(
        &painter,
        pos2(rect.left() + 10.0, rect.bottom() - 10.0),
        Align2::LEFT_BOTTOM,
        &bottom_left_text,
        t,
    );
    paint_badge(
        &painter,
        pos2(rect.right() - 10.0, rect.bottom() - 10.0),
        Align2::RIGHT_BOTTOM,
        &bottom_right_text,
        t,
    );
}

/// Paint a translucent black pill containing monospace text. Used for the
/// bottom-left/right badges on the canvas.
fn paint_badge(
    painter: &egui::Painter,
    anchor: Pos2,
    align: Align2,
    text: &str,
    t: &Tokens,
) {
    let font = FontId::monospace(10.5);
    let galley = painter.layout_no_wrap(text.to_string(), font, t.text);
    let pad_x = 7.0;
    let pad_y = 3.0;
    let size = vec2(galley.size().x + pad_x * 2.0, galley.size().y + pad_y * 2.0);
    let rect = align.anchor_rect(Rect::from_min_size(anchor, size));
    painter.rect(
        rect,
        2.0,
        Color32::from_black_alpha(140),
        Stroke::NONE,
    );
    painter.galley(
        pos2(rect.left() + pad_x, rect.center().y - galley.size().y * 0.5),
        galley,
        t.text,
    );
}

/// 32px-tall toolbar pinned below the canvas. All buttons are Phase 3 stubs.
///
/// JSX (`canvas-panel.jsx:91`) sets `borderTop: 1px solid borderSoft` — only
/// the *top* edge separates the toolbar from the canvas above. egui's
/// `Frame::stroke` always paints all four sides, so we drop the frame stroke
/// and paint the top edge manually after the frame body.
fn toolbar(ui: &mut Ui, state: &AppState, t: &Tokens) {
    let resp = Frame::none()
        .fill(t.panel)
        .inner_margin(Margin::symmetric(6.0, 0.0))
        .show(ui, |ui| {
            ui.set_min_height(32.0);
            ui.horizontal_centered(|ui| {
                tool_btn(ui, t, "⊞ Fit");
                tool_btn(ui, t, "100%");
                separator(ui, t);
                tool_btn(ui, t, "📋 Copy");
                tool_btn(ui, t, "📂 Open folder");
                tool_btn(ui, t, "📤 Send to input");

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let status = if state.generating {
                        // Phase 3 stub: `total_steps` is seeded from the
                        // configured step count in `action_bar.rs` on click.
                        // Phase 5 will replace this readout with real per-step
                        // updates from the inference worker channel.
                        format!("⏵ step {}/{}", state.current_step, state.total_steps)
                    } else {
                        "⏵ idle".to_string()
                    };
                    let color = if state.generating {
                        // Use the success token while a job is running — the
                        // toolbar status is "things are progressing", which is
                        // the canonical use of the green hue.
                        t.success
                    } else {
                        t.text_mute
                    };
                    ui.label(
                        RichText::new(status)
                            .size(11.0)
                            .monospace()
                            .color(color),
                    );
                });
            });
        });

    // Border-top only: paint a 1px line along the top edge of the toolbar
    // frame. The frame's full rect (incl. inner margin) is in `resp.response.rect`.
    let bar_rect = resp.response.rect;
    ui.painter().line_segment(
        [bar_rect.left_top(), bar_rect.right_top()],
        Stroke::new(1.0, t.border_soft),
    );
}

fn tool_btn(ui: &mut Ui, t: &Tokens, label: &str) {
    let resp = ui.add(
        egui::Button::new(RichText::new(label).size(11.0).color(t.text_dim))
            .fill(Color32::TRANSPARENT)
            .stroke(Stroke::NONE)
            .min_size(vec2(0.0, 22.0))
            .rounding(crate::tokens::RADIUS_CONTROL),
    );
    if resp.clicked() {
        log::debug!("Canvas toolbar `{}` clicked (Phase 5+ wiring)", label);
    }
}

fn separator(ui: &mut Ui, t: &Tokens) {
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(vec2(1.0, 14.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, t.border);
    ui.add_space(4.0);
}
