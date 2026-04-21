//! Custom client-side decorations: title bar, menu bar, status bar.
//!
//! All three are `TopBottomPanel`s with exact heights from the README
//! "Window & chrome" section (32 / 34 / 22 px). The title bar is the OS
//! drag region and hosts the win-controls; the menu bar holds File/Edit/...,
//! the Image|Video mode tabs, and the theme toggle; the status bar shows
//! placeholder telemetry text.

use eframe::egui;
use egui::{
    pos2, vec2, Align, Align2, Button, Color32, FontId, Frame, Layout, Margin, Order, Rect,
    RichText, Sense, Stroke, TopBottomPanel, ViewportCommand,
};

use crate::app::FlameInferenceApp;
use crate::state::Mode;
use crate::theme::Theme;
use crate::tokens::{
    self, CLOSE_HOVER, FONT_BODY, FONT_MONO, MENUBAR_H, STATUSBAR_H, TITLEBAR_H, WIN_BTN_W,
};

// --- title bar -----------------------------------------------------------

pub fn title_bar(ctx: &egui::Context, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    TopBottomPanel::top("titlebar")
        .exact_height(TITLEBAR_H)
        .resizable(false)
        .frame(
            Frame::none()
                .fill(t.titlebar)
                .stroke(Stroke::new(1.0, t.border))
                .inner_margin(Margin::symmetric(0.0, 0.0)),
        )
        .show(ctx, |ui| {
            // Drag region must EXCLUDE the rightmost ~3 × WIN_BTN_W reserved
            // for the close/max/min controls — otherwise mousedown on those
            // buttons also fires StartDrag (P0 #3 in SKEPTIC_PHASE1.md).
            let bar_rect = ui.max_rect();
            let drag_rect = Rect::from_min_max(
                bar_rect.min,
                pos2(bar_rect.max.x - 3.0 * WIN_BTN_W, bar_rect.max.y),
            );
            let drag_resp = ui.interact(
                drag_rect,
                ui.id().with("titlebar_drag"),
                Sense::click_and_drag(),
            );
            if drag_resp.is_pointer_button_down_on() {
                ctx.send_viewport_cmd(ViewportCommand::StartDrag);
            }
            // Double-click toggles maximize, matching native behavior.
            if drag_resp.double_clicked() {
                ctx.send_viewport_cmd(ViewportCommand::Maximized(
                    !ctx.input(|i| i.viewport().maximized.unwrap_or(false)),
                ));
            }

            // Left cluster: flame icon · "Flame" · v0.4.2 (mono, muted).
            ui.allocate_ui_with_layout(
                ui.available_size(),
                Layout::left_to_right(Align::Center),
                |ui| {
                    ui.add_space(12.0);
                    flame_icon(ui, t.accent, 13.0);
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("Flame")
                            .size(FONT_BODY)
                            .color(t.text)
                            .strong(),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("v0.4.2")
                            .size(FONT_MONO)
                            .monospace()
                            .color(t.text_mute),
                    );
                    // Push win-controls to the right edge.
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        // Painted right-to-left: close (rightmost), max, min.
                        win_btn(ui, app.state.theme, "✕", true, || {
                            ctx.send_viewport_cmd(ViewportCommand::Close);
                        });
                        win_btn(ui, app.state.theme, "▢", false, || {
                            let cur = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                            ctx.send_viewport_cmd(ViewportCommand::Maximized(!cur));
                        });
                        win_btn(ui, app.state.theme, "—", false, || {
                            ctx.send_viewport_cmd(ViewportCommand::Minimized(true));
                        });
                    });
                },
            );
        });
}

/// Minimal flame glyph painted with primitives (no SVG dep). The real
/// `assets/flame.svg` is shipped as a future-phase asset; for Phase 1 we
/// avoid pulling in egui_extras image loaders.
fn flame_icon(ui: &mut egui::Ui, color: Color32, size: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    let painter = ui.painter();
    // Two stacked teardrop ellipses approximating a flame silhouette.
    let outer = Rect::from_center_size(
        pos2(rect.center().x, rect.center().y + size * 0.05),
        vec2(size * 0.78, size * 0.95),
    );
    painter.rect(outer, size * 0.45, color, Stroke::NONE);
    let inner = Rect::from_center_size(
        pos2(rect.center().x, rect.center().y + size * 0.18),
        vec2(size * 0.45, size * 0.55),
    );
    painter.rect(
        inner,
        size * 0.30,
        Color32::from_rgba_unmultiplied(255, 255, 255, 60),
        Stroke::NONE,
    );
}

fn win_btn(
    ui: &mut egui::Ui,
    theme: Theme,
    glyph: &str,
    is_close: bool,
    mut on_click: impl FnMut(),
) {
    let t = theme.tokens();
    let (rect, resp) =
        ui.allocate_exact_size(vec2(WIN_BTN_W, TITLEBAR_H), Sense::click());
    let hovered = resp.hovered();
    let fill = if hovered {
        if is_close { CLOSE_HOVER } else { t.row }
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 0.0, fill);
    let fg = if hovered && is_close {
        Color32::WHITE
    } else {
        t.text_dim
    };
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        glyph,
        FontId::proportional(12.0),
        fg,
    );
    if resp.clicked() {
        on_click();
    }
}

// --- menu bar ------------------------------------------------------------

pub fn menu_bar(ctx: &egui::Context, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    TopBottomPanel::top("menubar")
        .exact_height(MENUBAR_H)
        .resizable(false)
        .frame(
            Frame::none()
                .fill(t.panel)
                .stroke(Stroke::new(1.0, t.border))
                .inner_margin(Margin::symmetric(4.0, 0.0)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                egui::menu::bar(ui, |ui| {
                    file_menu(ui);
                    edit_menu(ui);
                    view_menu(ui, app);
                    models_menu(ui);
                    queue_menu(ui);
                    help_menu(ui);

                    // Vertical separator between menus and mode tabs.
                    // JSX uses `margin: '0 8px'` → 8px on each side.
                    ui.add_space(8.0);
                    let sep_h = 18.0;
                    let (rect, _) = ui.allocate_exact_size(vec2(1.0, sep_h), Sense::hover());
                    ui.painter().rect_filled(rect, 0.0, t.border);
                    ui.add_space(8.0);

                    // Mode tabs: Image | Video.
                    mode_tab(ui, app, Mode::Image);
                    mode_tab(ui, app, Mode::Video);

                    // Theme toggle pinned far right.
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let glyph = match app.state.theme {
                            Theme::Dark => "☀",   // currently dark → click for light
                            Theme::Light => "☾",
                        };
                        let (rect, resp) =
                            ui.allocate_exact_size(vec2(26.0, 22.0), Sense::click());
                        let fill = if resp.hovered() {
                            t.row
                        } else {
                            Color32::TRANSPARENT
                        };
                        ui.painter()
                            .rect(rect, tokens::RADIUS_CONTROL, fill, Stroke::NONE);
                        ui.painter().text(
                            rect.center(),
                            Align2::CENTER_CENTER,
                            glyph,
                            FontId::proportional(14.0),
                            t.text_dim,
                        );
                        if resp.clicked() {
                            app.state.theme.toggle();
                            ctx.set_visuals(app.state.theme.visuals());
                            crate::theme::apply_density(ctx);
                        }
                    });
                });
            });
        });
}

fn mode_tab(ui: &mut egui::Ui, app: &mut FlameInferenceApp, mode: Mode) {
    let t = app.state.theme.tokens();
    let active = app.state.tab == mode;
    let label_color = if active { t.accent } else { t.text_dim };

    // JSX (chrome.jsx:128) is `fontWeight: 500` (medium). egui has no medium
    // weight; .strong() is the closest approximation we have without loading
    // a real font (Phase 2 deferral).
    let rich = RichText::new(mode.label())
        .size(FONT_BODY)
        .color(label_color)
        .strong();

    // Reserve the tab footprint and get a real interactable response (this
    // is focusable for keyboard nav). Then paint hover bg into that rect,
    // then layout the styled RichText into a galley and paint it centered.
    // Fixes P0 #2 — the previous `painter().text(..., label.text(), ...)`
    // discarded the RichText weight/color.
    let (rect, resp) = ui.allocate_exact_size(vec2(64.0, 26.0), Sense::click());
    if resp.hovered() && !active {
        ui.painter()
            .rect_filled(rect, tokens::RADIUS_CONTROL, t.row);
    }
    let galley = egui::WidgetText::from(rich).into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        rect.width(),
        egui::TextStyle::Button,
    );
    let text_pos = rect.center() - galley.size() * 0.5;
    ui.painter().galley(text_pos, galley, label_color);

    if active {
        // Active-tab underline must sit AT the menu bar's bottom edge so it
        // visually replaces the panel's 1px border at this x-range — matches
        // the JSX `borderBottom: 2px ...; marginBottom: -1` pattern.
        // Painted on a Foreground layer so the panel-border shape (which
        // gets drawn during the Frame end of `TopBottomPanel`) cannot occlude
        // it (fixes P0 #1).
        let bar_bottom = ui.max_rect().bottom();
        let underline = Rect::from_min_max(
            pos2(rect.left() + 6.0, bar_bottom - 1.0),
            pos2(rect.right() - 6.0, bar_bottom + 1.0),
        );
        let layer_painter = ui.ctx().layer_painter(egui::LayerId::new(
            Order::Foreground,
            ui.id().with(("mode_tab_underline", mode)),
        ));
        layer_painter.rect_filled(underline, 0.0, t.accent);
    }

    if resp.clicked() {
        app.state.tab = mode;
    }
}

// All menu items are no-op closures in Phase 1. The structure must be
// present so later phases can wire shortcuts and handlers.
//
// Width constant matches the JSX prototype's `minWidth: 220` so menu items
// inside a single dropdown share a baseline width and the right-aligned
// shortcut text from `Button::shortcut_text` lines up across rows.
const MENU_ITEM_W: f32 = 220.0;

fn item(label: &str, shortcut: &str) -> Button<'static> {
    let mut b = Button::new(label).min_size(vec2(MENU_ITEM_W, 0.0));
    if !shortcut.is_empty() {
        b = b.shortcut_text(shortcut);
    }
    b
}

fn file_menu(ui: &mut egui::Ui) {
    ui.menu_button("File", |ui| {
        let _ = ui.add(item("New generation", "Ctrl+N"));
        let _ = ui.add(item("Open output folder", "Ctrl+O"));
        let _ = ui.add(item("Save preset", "Ctrl+S"));
        ui.separator();
        let _ = ui.add(item("Import params…", ""));
        let _ = ui.add(item("Export params…", ""));
        ui.separator();
        let _ = ui.add(item("Exit", "Alt+F4"));
    });
}

fn edit_menu(ui: &mut egui::Ui) {
    ui.menu_button("Edit", |ui| {
        let _ = ui.add(item("Copy prompt", "Ctrl+C"));
        let _ = ui.add(item("Paste prompt", "Ctrl+V"));
        let _ = ui.add(item("Clear prompt", ""));
    });
}

fn view_menu(ui: &mut egui::Ui, app: &mut FlameInferenceApp) {
    ui.menu_button("View", |ui| {
        if ui.add(item("Toggle dark theme", "")).clicked() {
            app.state.theme.toggle();
            ui.ctx().set_visuals(app.state.theme.visuals());
            crate::theme::apply_density(ui.ctx());
            ui.close_menu();
        }
        if ui.add(item("Toggle advanced", "")).clicked() {
            app.state.advanced = !app.state.advanced;
            ui.close_menu();
        }
        ui.separator();
        let _ = ui.add(item("Zoom canvas to fit", "F"));
        let _ = ui.add(item("Actual size", "1"));
    });
}

fn models_menu(ui: &mut egui::Ui) {
    ui.menu_button("Models", |ui| {
        let _ = ui.add(item("Rescan weights folder", "F5"));
        let _ = ui.add(item("Open weights folder", ""));
        ui.separator();
        let _ = ui.add(item("Download manager…", ""));
        let _ = ui.add(item("Model info…", ""));
    });
}

fn queue_menu(ui: &mut egui::Ui) {
    ui.menu_button("Queue", |ui| {
        let _ = ui.add(item("Run next", "F9"));
        let _ = ui.add(item("Pause queue", ""));
        let _ = ui.add(item("Clear queue", ""));
        ui.separator();
        let _ = ui.add(item("Retry last", ""));
    });
}

fn help_menu(ui: &mut egui::Ui) {
    ui.menu_button("Help", |ui| {
        let _ = ui.add(item("Documentation", ""));
        let _ = ui.add(item("Keyboard shortcuts", "?"));
        let _ = ui.add(item("About Flame…", ""));
    });
}

// --- status bar ----------------------------------------------------------

pub fn status_bar(ctx: &egui::Context, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    TopBottomPanel::bottom("status")
        .exact_height(STATUSBAR_H)
        .resizable(false)
        .frame(
            Frame::none()
                .fill(t.titlebar)
                .stroke(Stroke::new(1.0, t.border))
                .inner_margin(Margin::symmetric(10.0, 2.0)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                // Phase 1 placeholder text — same shape as the README example
                // ("● backend connected · D:\flame\weights · 16 models · ready")
                // so later phases can swap each segment without changing layout.
                let line = format!(
                    "● {connection} · {weights} · {n} models · {state}",
                    connection = "backend connected",
                    weights = r"D:\flame\weights",
                    n = 16,
                    state = "ready",
                );
                ui.label(
                    RichText::new(line)
                        // README "Window & chrome": status bar mono @ 11px.
                        .size(FONT_MONO)
                        .monospace()
                        .color(t.text_dim),
                );
            });
        });
}
