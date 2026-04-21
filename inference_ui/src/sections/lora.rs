//! Section: LoRA stack — drag handle + checkbox + name + strength + ×.
//!
//! Per README "Section: LoRA stack" (lines 88-96). Header right side shows
//! `<active> / <total>` muted mono counter. Drag-reorder is **not** wired
//! in Phase 2 (per brief: visible drag handle is enough; `egui-dnd` skipped).
//!
//! AGENT-DEFAULT: the truncated name + ellipsis is approximated with
//! `egui::TextEdit` set to display-only or a `Label::truncate(true)`.
//! Phase 2 uses a non-interactive label with manual char-count truncation.

use egui::{CollapsingHeader, Frame, Margin, RichText, Stroke, Ui};

use crate::state::{AppState, LoraSlot};
use crate::tokens::{Tokens, FONT_SECTION_LABEL, RADIUS_CONTROL};
use crate::widgets::{drag_f32, ghost_button};

pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    // Counter for the right side of the header.
    let total = state.current().loras.len();
    let active = state.current().loras.iter().filter(|l| l.active).count();
    let counter = format!("{active} / {total}");

    // We need a header with a right-aligned counter; the stock
    // CollapsingHeader title is left-aligned only. Use a horizontal layout
    // around the header + counter sharing one row. The trick: render the
    // CollapsingHeader inside a layout that also paints the counter on the
    // right.
    //
    // For Phase 2, ship the counter inside a parenthetical suffix on the
    // header label rather than a free-floating right-aligned span. That's a
    // pragmatic divergence from the JSX (which uses an absolute right-side
    // badge) — flagged as AGENT-DEFAULT.
    let title = RichText::new(format!("LORA STACK   {counter}"))
        .size(FONT_SECTION_LABEL)
        .color(t.text_dim)
        .strong();

    CollapsingHeader::new(title)
        .id_salt("sec_lora")
        .default_open(true)
        .show(ui, |ui| {
            // Track removals + adds out-of-loop so we don't mutate the
            // Vec while iterating. (egui can't borrow the Vec twice.)
            let mut remove_idx: Option<usize> = None;

            // Iterate by index so each row's IDs are stable.
            let n = state.current().loras.len();
            for i in 0..n {
                if render_row(ui, state, t, i) {
                    remove_idx = Some(i);
                }
                ui.add_space(2.0);
            }

            if let Some(i) = remove_idx {
                state.current_mut().loras.remove(i);
            }

            // [+ Add LoRA] ghost button. Out-of-scope: real file picker.
            // Stub: append a default LoraSlot.
            ui.add_space(6.0);
            if ghost_button(ui, t, "+ Add LoRA").clicked() {
                state.current_mut().loras.push(LoraSlot::default());
            }
        });
}

/// Render one LoRA row. Returns `true` if the user clicked × (caller
/// handles removal after the loop to avoid mid-iteration mutation).
fn render_row(ui: &mut Ui, state: &mut AppState, t: &Tokens, i: usize) -> bool {
    let mut requested_remove = false;

    // Card-style background (`panel_sunk` + `border_soft`).
    Frame::none()
        .fill(t.panel_sunk)
        .stroke(Stroke::new(1.0, t.border_soft))
        .rounding(RADIUS_CONTROL)
        .inner_margin(Margin::symmetric(6.0, 5.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // Drag handle (≡). Phase 2 placeholder — not wired to a
                // reorder gesture. Painted as a muted mono glyph.
                ui.label(RichText::new("≡").size(13.0).color(t.text_mute));

                let lora = &mut state.current_mut().loras[i];

                // Active checkbox.
                ui.checkbox(&mut lora.active, "");

                // Name label, truncated. Active rows are full-text color,
                // inactive rows muted (mirrors the JSX rule).
                let name_color = if lora.active { t.text } else { t.text_mute };
                let truncated = truncate(&lora.name, 24);
                ui.add(
                    egui::Label::new(
                        RichText::new(truncated)
                            .size(11.0)
                            .monospace()
                            .color(name_color),
                    )
                    .truncate(),
                );

                // Right-align: strength + ×. Use a right-to-left sub-layout.
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        // Remove (×) — visible always (spec says "on hover"
                        // but egui can't show-only-on-hover without a custom
                        // widget; ship always-visible per AGENT-DEFAULT).
                        if ui
                            .small_button(RichText::new("×").color(t.text_mute))
                            .on_hover_text("Remove")
                            .clicked()
                        {
                            requested_remove = true;
                        }
                        drag_f32(ui, &mut lora.strength, 0.0..=2.0, 0.05, 56.0, 2);
                    },
                );

            });
        });

    requested_remove
}

/// Truncate a string to `max` chars + "…" suffix when oversize. Operates
/// on chars (not bytes) so multi-byte LoRA filenames don't slice mid-codepoint.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{head}…")
}
