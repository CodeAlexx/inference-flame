//! Section: Resolution (Image) / Output (Video).
//!
//! Per README "Section: Resolution / Output" (lines 67-72). Width/height
//! live in the per-mode `ModeSettings` so switching tabs preserves both.
//! Frames + FPS are Video-only (`Option<u32>`).

use egui::{RichText, Ui};

use crate::sections::model::labeled_row;
use crate::state::{AppState, Mode};
use crate::tokens::{Tokens, FONT_MONO};
use crate::widgets::{combo_str, drag_u32, flat_collapsing};

const IMAGE_PRESETS: &[&str] = &[
    "512×512  ·  1:1",
    "768×768  ·  1:1",
    "1024×1024  ·  1:1",
    "1216×832  ·  3:2",
    "832×1216  ·  2:3",
    "1344×768  ·  16:9",
    "768×1344  ·  9:16",
    "1536×640  ·  21:9",
    "Custom",
];

const VIDEO_PRESETS: &[&str] = &[
    "512×512  ·  1:1",
    "720×480  ·  3:2",
    "848×480  ·  16:9",
    "1024×576  ·  16:9",
    "1280×720  ·  16:9",
    "720×1280  ·  9:16",
    "Custom",
];

pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    let mode = state.tab;
    let title = match mode {
        Mode::Image => "Resolution",
        Mode::Video => "Output",
    };
    let cn = state.current_mut();

    flat_collapsing(ui, t, "sec_resolution", title, true, |ui| {
        // Preset picker — parsing the `<W>×<H>  ·  <ratio>` string updates
        // width/height. Manual edits to W/H below leave the preset string
        // alone (keeps the visible label as-is even if it now mismatches).
        labeled_row(ui, t, "Preset", |ui| {
            let opts = match mode {
                Mode::Image => IMAGE_PRESETS,
                Mode::Video => VIDEO_PRESETS,
            };
            if combo_str(ui, "preset", &mut cn.resolution_preset, opts) {
                if let Some((w, h)) = parse_preset(&cn.resolution_preset) {
                    cn.width = w;
                    cn.height = h;
                }
            }
        });

        // Size: W × H, swap button on the right. Manual edits to W or H
        // re-evaluate the preset string: if (W, H) matches a known preset
        // we snap the dropdown label to it; otherwise the label flips to
        // "Custom" so the visible preset never lies about the current size.
        let presets = match mode {
            Mode::Image => IMAGE_PRESETS,
            Mode::Video => VIDEO_PRESETS,
        };
        labeled_row(ui, t, "Size", |ui| {
            let mut size_changed = false;
            size_changed |= drag_u32(ui, &mut cn.width, 64..=4096, 8.0, 60.0).changed();
            ui.label(RichText::new("×").size(11.0).color(t.text_mute));
            size_changed |= drag_u32(ui, &mut cn.height, 64..=4096, 8.0, 60.0).changed();
            // Swap button — round-trip W↔H. Tooltip fires on hover.
            if ui
                .small_button("⇄")
                .on_hover_text("Swap W/H")
                .clicked()
            {
                std::mem::swap(&mut cn.width, &mut cn.height);
                size_changed = true;
            }
            if size_changed {
                cn.resolution_preset = match_preset(cn.width, cn.height, presets)
                    .map(str::to_owned)
                    .unwrap_or_else(|| "Custom".to_string());
            }
        });

        // Video-only fields. README spec: 8..=241 frames, 6..=60 fps,
        // mono readout `= 5.06s` next to Frames computed as frames / fps.
        if matches!(mode, Mode::Video) {
            // Take or initialize frames/fps so the DragValues have stable
            // mut refs. We re-pack into `Option` after the row.
            let mut frames = cn.frames.unwrap_or(81);
            let mut fps = cn.fps.unwrap_or(16);

            labeled_row(ui, t, "Frames", |ui| {
                drag_u32(ui, &mut frames, 8..=241, 1.0, 60.0);
                let secs = frames as f32 / fps.max(1) as f32;
                ui.label(
                    RichText::new(format!("= {secs:.2}s"))
                        .size(FONT_MONO)
                        .monospace()
                        .color(t.text_mute),
                );
            });
            labeled_row(ui, t, "FPS", |ui| {
                drag_u32(ui, &mut fps, 6..=60, 1.0, 60.0);
            });

            cn.frames = Some(frames);
            cn.fps = Some(fps);
        }
    });
}

/// Parse a preset like `"1024×1024  ·  1:1"` into `(width, height)`. Both
/// the unicode `×` (U+00D7) and lowercase `x` are accepted.
fn parse_preset(s: &str) -> Option<(u32, u32)> {
    let normalized = s.replace('×', "x");
    let head = normalized.split('·').next().unwrap_or("");
    let mut parts = head.split('x').map(|p| p.trim());
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    Some((w, h))
}

/// Find the preset string whose `<W>×<H>` head matches the given size.
/// "Custom" entries (which don't parse) are skipped naturally. Returns
/// `None` if no preset matches — caller should fall back to "Custom".
fn match_preset(w: u32, h: u32, presets: &[&'static str]) -> Option<&'static str> {
    presets
        .iter()
        .copied()
        .find(|p| parse_preset(p) == Some((w, h)))
}
