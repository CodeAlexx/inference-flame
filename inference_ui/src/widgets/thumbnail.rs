//! Procedurally-painted thumbnail placeholder.
//!
//! Phase 4 has no real image loader — `egui_extras::install_image_loaders`
//! + the `image` crate would pull in JPEG/PNG decoders, file I/O, and an
//! async glue layer. The brief explicitly defers that to "Phase 5+".
//!
//! In the meantime we paint a deterministic colored gradient keyed off an
//! `id: u64` so:
//!   - Repeated frames render the *same* color for the same job (no flicker).
//!   - Different jobs are visually distinguishable at a glance.
//!   - The result reads as a "fake thumbnail" not as missing-content.
//!
//! Visual style mirrors the JSX prototype's
//! `linear-gradient(135deg, oklch(...), oklch(...))` (`queue-panel.jsx:78`).
//! egui has no OKLCH; we use HSV via `Color32::from`/manual RGB.

use egui::{pos2, vec2, Color32, FontId, Rect, Sense, Shape, Stroke, Ui, Vec2};

use crate::tokens::Tokens;

/// Reserve a `size`-sized rect and paint a placeholder thumbnail keyed on
/// `id`. Returns the response (so callers can attach `.context_menu(...)`,
/// hover, click, etc.).
pub fn placeholder(ui: &mut Ui, t: &Tokens, size: Vec2, id: u64) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    paint_into(ui, t, rect, id, None);
    resp
}

/// Same as `placeholder` but overlays a centered label string (used for
/// the running job's progress percentage in the queue row).
pub fn placeholder_with_overlay(
    ui: &mut Ui,
    t: &Tokens,
    size: Vec2,
    id: u64,
    overlay: &str,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    paint_into(ui, t, rect, id, Some(overlay));
    resp
}

/// Paint a thumbnail into a *pre-allocated* rect. Use when laying out
/// inside a custom container where the caller has already reserved space
/// (e.g. the history grid).
pub fn paint_into(ui: &Ui, t: &Tokens, rect: Rect, id: u64, overlay: Option<&str>) {
    let painter = ui.painter_at(rect);
    let radius = 2.0;

    // Deterministic two-color gradient. Hue rotates with id so each job
    // looks distinct; we keep the palette warm-ish (yellow/orange/red) so
    // it reads as "Flame app" rather than rainbow noise. Range is the
    // 0..360° hue circle but biased toward 0..80° (warm half).
    let hue = ((id.wrapping_mul(53)) % 80) as f32 + 10.0;
    let dark = hsv_to_color32(hue, 0.55, 0.32);
    let light = hsv_to_color32(hue + 18.0, 0.50, 0.72);

    // Two-triangle gradient via a quad mesh. egui's Mesh API takes 4
    // colored vertices and a 2-tri index buffer; the rasterizer interpolates
    // colors per-pixel. This is the cheapest way to approximate the JSX
    // `linear-gradient(135deg, ...)` without a shader.
    let mut mesh = egui::Mesh::default();
    let tl = mesh.vertices.len() as u32;
    mesh.vertices.push(egui::epaint::Vertex {
        pos: rect.left_top(),
        uv: egui::epaint::WHITE_UV,
        color: dark,
    });
    mesh.vertices.push(egui::epaint::Vertex {
        pos: rect.right_top(),
        uv: egui::epaint::WHITE_UV,
        color: blend(dark, light, 0.5),
    });
    mesh.vertices.push(egui::epaint::Vertex {
        pos: rect.right_bottom(),
        uv: egui::epaint::WHITE_UV,
        color: light,
    });
    mesh.vertices.push(egui::epaint::Vertex {
        pos: rect.left_bottom(),
        uv: egui::epaint::WHITE_UV,
        color: blend(dark, light, 0.5),
    });
    mesh.indices.extend([tl, tl + 1, tl + 2, tl, tl + 2, tl + 3]);
    painter.add(Shape::mesh(mesh));

    // Soft inner highlight (the JSX overlays radial gradients with
    // `mixBlendMode: overlay`). Approximate with a small bright spot near
    // the upper-left and a darker spot near the lower-right.
    let bright_spot = Rect::from_center_size(
        pos2(
            rect.left() + rect.width() * 0.30,
            rect.top() + rect.height() * 0.40,
        ),
        vec2(rect.width() * 0.35, rect.height() * 0.35),
    );
    painter.rect_filled(
        bright_spot,
        bright_spot.width() * 0.5,
        Color32::from_white_alpha(28),
    );
    let dark_spot = Rect::from_center_size(
        pos2(
            rect.left() + rect.width() * 0.75,
            rect.top() + rect.height() * 0.75,
        ),
        vec2(rect.width() * 0.40, rect.height() * 0.40),
    );
    painter.rect_filled(
        dark_spot,
        dark_spot.width() * 0.5,
        Color32::from_black_alpha(40),
    );

    // Optional overlay text — running-job progress %. Painted on a
    // semi-transparent black scrim that covers the whole tile so the text
    // reads regardless of the gradient color underneath. Drawn before the
    // border so the border still shows cleanly on top of the scrim.
    if let Some(text) = overlay {
        painter.rect_filled(rect, radius, Color32::from_black_alpha(120));
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            FontId::monospace(10.0),
            Color32::WHITE,
        );
    }

    // 1px border so tiles separate cleanly when packed in a grid. Painted
    // on the *parent* painter (not the clipped `painter_at(rect)` used
    // for fills) — `rect_stroke` centers the stroke on the rect edge, so
    // the outer half would be clipped to ~0.5px if drawn on the clipped
    // painter. Using the unclipped parent painter keeps the full 1px
    // visible. Fills above are still clipped, so this doesn't risk
    // overdraw outside `rect`.
    ui.painter()
        .rect_stroke(rect, radius, Stroke::new(1.0, t.border_soft));
}

/// HSV → Color32 (RGB). H in degrees [0, 360), S/V in [0, 1].
/// Standard piecewise formula — copy of the formula in any computer-graphics
/// textbook. We don't want a chroma crate dependency for one helper.
fn hsv_to_color32(h: f32, s: f32, v: f32) -> Color32 {
    let h = h.rem_euclid(360.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as i32 / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let to_byte = |f: f32| ((f + m) * 255.0).clamp(0.0, 255.0) as u8;
    Color32::from_rgb(to_byte(r), to_byte(g), to_byte(b))
}

/// Linear blend between two colors. `t` is 0..=1 (0 = a, 1 = b).
fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| ((x as f32) * (1.0 - t) + (y as f32) * t) as u8;
    Color32::from_rgb(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
    )
}
