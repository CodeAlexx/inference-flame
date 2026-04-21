//! Color + sizing tokens for Flame Inference.
//!
//! Hex values come straight from the README "Design tokens" section
//! of `/tmp/flame_ui_design/design_handoff_flame_ui/README.md`. Don't
//! hand-tune these without updating the design doc first.

use egui::Color32;

#[derive(Debug, Clone, Copy)]
pub struct Tokens {
    pub bg: Color32,
    pub panel: Color32,
    pub panel_sunk: Color32,
    pub titlebar: Color32,
    pub row: Color32,
    pub border: Color32,
    pub border_soft: Color32,
    pub text: Color32,
    pub text_dim: Color32,
    pub text_mute: Color32,
    pub accent: Color32,
    pub accent_hi: Color32,
    pub accent_soft: Color32,
    pub success: Color32,
    pub danger: Color32,
}

/// Dark theme — README "Design tokens / Dark theme".
pub fn dark() -> Tokens {
    Tokens {
        bg:          Color32::from_rgb(0x1a, 0x1a, 0x1c),
        panel:       Color32::from_rgb(0x25, 0x25, 0x2a),
        panel_sunk:  Color32::from_rgb(0x1e, 0x1e, 0x21),
        titlebar:    Color32::from_rgb(0x19, 0x19, 0x1b),
        row:         Color32::from_rgb(0x2e, 0x2e, 0x34),
        border:      Color32::from_rgb(0x3a, 0x3a, 0x40),
        border_soft: Color32::from_rgb(0x2a, 0x2a, 0x2f),
        text:        Color32::from_rgb(0xe6, 0xe6, 0xe8),
        text_dim:    Color32::from_rgb(0xa8, 0xa8, 0xad),
        text_mute:   Color32::from_rgb(0x6e, 0x6e, 0x74),
        accent:      Color32::from_rgb(0xe8, 0xa3, 0x3d),
        accent_hi:   Color32::from_rgb(0xff, 0xb8, 0x5a),
        // rgba(232,163,61,0.15) — alpha = round(0.15*255) = 38 = 0x26
        accent_soft: Color32::from_rgba_unmultiplied(0xe8, 0xa3, 0x3d, 0x26),
        success:     Color32::from_rgb(0x5d, 0xd2, 0x7e),
        danger:      Color32::from_rgb(0xe5, 0x48, 0x4d),
    }
}

/// Light theme — README "Design tokens / Light theme".
pub fn light() -> Tokens {
    Tokens {
        bg:          Color32::from_rgb(0xf4, 0xf4, 0xf6),
        panel:       Color32::from_rgb(0xff, 0xff, 0xff),
        panel_sunk:  Color32::from_rgb(0xec, 0xec, 0xef),
        titlebar:    Color32::from_rgb(0xe8, 0xe8, 0xeb),
        row:         Color32::from_rgb(0xdf, 0xdf, 0xe3),
        border:      Color32::from_rgb(0xcd, 0xcd, 0xd2),
        border_soft: Color32::from_rgb(0xe2, 0xe2, 0xe6),
        text:        Color32::from_rgb(0x1b, 0x1b, 0x1e),
        text_dim:    Color32::from_rgb(0x55, 0x55, 0x5b),
        text_mute:   Color32::from_rgb(0x8a, 0x8a, 0x90),
        accent:      Color32::from_rgb(0xc1, 0x7a, 0x1e),
        accent_hi:   Color32::from_rgb(0xd4, 0x8b, 0x2a),
        // rgba(193,122,30,0.15) — alpha 0x26
        accent_soft: Color32::from_rgba_unmultiplied(0xc1, 0x7a, 0x1e, 0x26),
        // README's light table doesn't list success/danger; reuse the dark hues.
        success:     Color32::from_rgb(0x5d, 0xd2, 0x7e),
        danger:      Color32::from_rgb(0xe5, 0x48, 0x4d),
    }
}

/// Close button hover-fill — README chrome spec ("Close is red on hover #c42b1c").
pub const CLOSE_HOVER: Color32 = Color32::from_rgb(0xc4, 0x2b, 0x1c);

/// Canvas checker tiles — README "Column 2 — Canvas panel" line 124:
/// "Transparent checker background (8px tiles, `#2a2a2e` / `#1e1e22`)".
/// The dark/light pair is theme-fixed (always dark hues) per the spec wording —
/// the checker pattern represents the *transparency* layer behind any future
/// preview, not the panel chrome. AGENT-DEFAULT: also use these in light theme;
/// the brief mentions "light theme analogues" parenthetically but does not
/// prescribe values, and a checker pattern is conventionally dark even on
/// light themes (matches Photoshop / Krita / Blender). If the user wants
/// theme-tinted checkers we'll roll those into a `Tokens` field later.
///
/// Naming reflects relative brightness: `CHECKER_DARK` is the darker of the
/// two hexes (used as the base fill), `CHECKER_LIGHT` is the brighter overlay.
pub const CHECKER_DARK: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x22);
pub const CHECKER_LIGHT: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2e);
/// Light-theme checker pair — slightly brighter so the pattern reads on a
/// pale background while still being a "transparency indicator" idiom.
/// Same dark-base / light-overlay convention as the dark-theme pair.
pub const CHECKER_DARK_LIGHT_THEME: Color32 = Color32::from_rgb(0xc8, 0xc8, 0xcc);
pub const CHECKER_LIGHT_LIGHT_THEME: Color32 = Color32::from_rgb(0xd8, 0xd8, 0xdc);
pub const CHECKER_TILE: f32 = 8.0;

// Layout sizes from README "Window & chrome" + "Layout" sections.
pub const TITLEBAR_H: f32 = 32.0;
pub const MENUBAR_H: f32 = 34.0;
pub const STATUSBAR_H: f32 = 22.0;
pub const WIN_BTN_W: f32 = 44.0;

pub const PARAMS_W: f32 = 316.0;
pub const QUEUE_W: f32 = 260.0;
pub const COL_GAP: f32 = 6.0;
pub const OUTER_PAD: f32 = 6.0;

// Typography sizes from README "Typography" section.
pub const FONT_BODY: f32 = 12.0;
pub const FONT_SECTION_LABEL: f32 = 10.5;
pub const FONT_MONO: f32 = 11.0;

pub const RADIUS_CONTROL: f32 = 3.0;
pub const RADIUS_PANEL: f32 = 5.0;
