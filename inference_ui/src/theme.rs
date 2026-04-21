//! Theme = Dark | Light + the egui `Visuals` factory.
//!
//! Phase 1: dark/light palettes from the README design tokens. Spacing
//! overrides keep the UI dense (instrument-panel feel, not roomy form).
//!
//! NOTE on fonts: the README asks for Inter (UI) + JetBrains Mono (numerics).
//! Phase 1 falls back to egui's bundled `Proportional` and `Monospace` so the
//! crate has zero external file deps. Real font loading goes in a later phase
//! once we ship the .ttf files in `assets/`.

use egui::{Rounding, Stroke, Visuals};
use serde::{Deserialize, Serialize};

use crate::tokens::{self, Tokens};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    Dark,
    Light,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Dark
    }
}

impl Theme {
    pub fn toggle(&mut self) {
        *self = match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        };
    }

    pub fn tokens(self) -> Tokens {
        match self {
            Theme::Dark => tokens::dark(),
            Theme::Light => tokens::light(),
        }
    }

    pub fn visuals(self) -> Visuals {
        let t = self.tokens();
        let mut v = match self {
            Theme::Dark => Visuals::dark(),
            Theme::Light => Visuals::light(),
        };
        v.override_text_color = Some(t.text);
        v.window_fill = t.bg;
        v.panel_fill = t.bg;
        v.faint_bg_color = t.panel_sunk;
        v.extreme_bg_color = t.panel_sunk;

        v.widgets.noninteractive.bg_fill = t.panel;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, t.border);
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, t.text_dim);

        v.widgets.inactive.bg_fill = t.panel_sunk;
        v.widgets.inactive.bg_stroke = Stroke::new(1.0, t.border);
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, t.text);

        v.widgets.hovered.bg_fill = t.row;
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, t.border);
        v.widgets.hovered.fg_stroke = Stroke::new(1.0, t.text);

        v.widgets.active.bg_fill = t.accent;
        v.widgets.active.bg_stroke = Stroke::new(1.0, t.accent);
        v.widgets.active.fg_stroke = Stroke::new(1.0, t.bg);

        v.selection.bg_fill = t.accent_soft;
        v.selection.stroke = Stroke::new(1.0, t.accent);

        v.hyperlink_color = t.accent;
        v.warn_fg_color = t.accent_hi;
        v.error_fg_color = t.danger;
        v
    }
}

/// Tighten egui's default spacing — README "Spacing" section.
pub fn apply_density(ctx: &egui::Context) {
    ctx.style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(8.0, 4.0);
        s.spacing.interact_size.y = 22.0;

        let r_ctrl = Rounding::same(tokens::RADIUS_CONTROL);
        let r_panel = Rounding::same(tokens::RADIUS_PANEL);
        s.visuals.widgets.noninteractive.rounding = r_panel;
        s.visuals.widgets.inactive.rounding = r_ctrl;
        s.visuals.widgets.hovered.rounding = r_ctrl;
        s.visuals.widgets.active.rounding = r_ctrl;
    });
}
