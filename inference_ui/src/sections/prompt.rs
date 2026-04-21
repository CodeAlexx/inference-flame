//! Section: Prompt + Negative — top of the canvas panel.
//!
//! Per README "Column 2 — Canvas panel" (lines 106-116):
//!   - Header row (38px tall): `PROMPT` label · task badge · spacer ·
//!     Enhance/Template ghost buttons.
//!   - Multiline prompt box ~100px tall, 13px sans, accent focus ring,
//!     token counter at bottom-right of the box (`<n> / 256`).
//!   - `NEGATIVE` label + single-line input that expands to multiline on focus.
//!
//! Phase 3 stubs: Enhance / Template buttons are no-ops (Phase 5+ wiring).
//! Token counter uses whitespace count — real BPE tokenization defers to
//! Phase 5+.

use egui::{
    pos2, vec2, Align, Color32, FontId, Frame, Layout, Margin, RichText, Sense, Stroke, TextEdit,
    Ui,
};

use crate::state::AppState;
use crate::tokens::{Tokens, FONT_MONO, FONT_SECTION_LABEL, RADIUS_CONTROL};

/// Approximate prompt token budget — matches the spec literal `<count> / 256`.
/// Real model context lengths (CLIP 77, T5 256/512, Llama 4096…) are
/// model-specific; until the worker wires up tokenization in Phase 5+ we
/// hard-code 256 to mirror the design mock.
const TOKEN_BUDGET: usize = 256;

/// Header + prompt box + negative prompt. Bound to the shared `prompt` and
/// `negative` fields on `AppState`.
pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    let task = state.current().task;

    // --- Header row (38px tall) ----------------------------------------
    // The 38px figure includes vertical padding; using `allocate_ui_with_layout`
    // pinned to that height keeps the header consistent regardless of the
    // contents (badge present or missing).
    ui.allocate_ui_with_layout(
        vec2(ui.available_width(), 38.0),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.label(
                RichText::new("PROMPT")
                    .size(FONT_SECTION_LABEL)
                    .monospace()
                    .color(t.text_dim)
                    .strong(),
            );
            ui.add_space(6.0);
            task_badge(ui, t, task.short_label());

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                // Phase 3 stubs — no behavior wired. Spec line 108 names them
                // explicitly; clicks log a debug line and nothing else.
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("✨ Template")
                                .size(11.0)
                                .color(t.text_mute),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::NONE)
                        .min_size(vec2(0.0, 22.0)),
                    )
                    .clicked()
                {
                    log::debug!("Template button clicked (Phase 5+ wiring)");
                }
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("✦ Enhance")
                                .size(11.0)
                                .color(t.text_mute),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::NONE)
                        .min_size(vec2(0.0, 22.0)),
                    )
                    .clicked()
                {
                    log::debug!("Enhance button clicked (Phase 5+ wiring)");
                }
            });
        },
    );

    ui.add_space(4.0);

    // --- Prompt box (multiline, ~100px tall) ---------------------------
    // We render the TextEdit inside a manually-painted Frame so the focus
    // ring (accent stroke) is explicit and consistent with the spec's "soft
    // focus ring in accent color". egui's TextEdit::frame(true) draws its
    // own border that we can't easily theme per state, so we draw the box
    // ourselves and use TextEdit::frame(false) inside.
    let box_height = 100.0;
    let prompt_id = egui::Id::new("prompt_textedit");
    let focused = ui.memory(|m| m.has_focus(prompt_id));
    let stroke_color = if focused { t.accent } else { t.border };

    let frame = Frame::none()
        .fill(t.panel_sunk)
        .stroke(Stroke::new(if focused { 1.5 } else { 1.0 }, stroke_color))
        .rounding(RADIUS_CONTROL);

    // Reserve the bottom ~18px of the box for the token counter. Sizing the
    // TextEdit explicitly to (box_height - COUNTER_GUTTER) leaves a band of
    // unallocated frame space at the bottom that the counter is painted into.
    // We capture the TextEdit's actual response rect (not the outer Frame's)
    // so the counter position is independent of any future frame padding /
    // inner_margin tweaks — this was the fragility the skeptic flagged.
    const COUNTER_GUTTER: f32 = 18.0;
    let mut edit_rect: Option<egui::Rect> = None;
    let _frame_resp = frame.show(ui, |ui| {
        ui.set_min_height(box_height);
        let avail = ui.available_size();
        let edit = TextEdit::multiline(&mut state.prompt)
            .id(prompt_id)
            .frame(false)
            .desired_width(avail.x)
            .desired_rows(4)
            .hint_text(
                RichText::new("Describe the image you want…")
                    .size(13.0)
                    .color(t.text_mute),
            )
            .font(FontId::proportional(13.0))
            .text_color(t.text)
            .margin(Margin::symmetric(8.0, 6.0));
        let r = ui.add_sized([avail.x, box_height - COUNTER_GUTTER], edit);
        edit_rect = Some(r.rect);
    });

    // Token counter — anchored to the TextEdit's own bottom-right corner with
    // a hardcoded 4px inset on each axis. Whitespace tokenization is a Phase 3
    // placeholder; real BPE counting wires up in Phase 5. Color stays muted
    // mono regardless of count (no fake budget validation).
    let token_count = state.prompt.split_whitespace().count();
    let counter_text = format!("{} / {}", token_count, TOKEN_BUDGET);
    if let Some(er) = edit_rect {
        let counter_pos = pos2(er.right() - 4.0, er.bottom() + (COUNTER_GUTTER * 0.5));
        ui.painter().text(
            counter_pos,
            egui::Align2::RIGHT_CENTER,
            counter_text,
            FontId::monospace(FONT_MONO),
            t.text_mute,
        );
    }

    ui.add_space(6.0);

    // --- Negative prompt -----------------------------------------------
    // Spec: "single-line by default; expand to multiline when focused."
    // egui has no built-in "grow on focus" widget; we approximate by tracking
    // focus and swapping between singleline (22px) and multiline (~60px)
    // based on the previous frame's focus state.
    let neg_id = egui::Id::new("negative_textedit");
    let neg_focused = ui.memory(|m| m.has_focus(neg_id));

    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            vec2(70.0, 22.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.label(
                    RichText::new("NEGATIVE")
                        .size(FONT_SECTION_LABEL)
                        .monospace()
                        .color(t.text_dim)
                        .strong(),
                );
            },
        );

        let neg_stroke = if neg_focused { t.accent } else { t.border };
        let neg_frame = Frame::none()
            .fill(t.panel_sunk)
            .stroke(Stroke::new(if neg_focused { 1.5 } else { 1.0 }, neg_stroke))
            .rounding(RADIUS_CONTROL);

        neg_frame.show(ui, |ui| {
            let avail = ui.available_width();
            if neg_focused {
                let edit = TextEdit::multiline(&mut state.negative)
                    .id(neg_id)
                    .frame(false)
                    .desired_width(avail)
                    .desired_rows(3)
                    .hint_text(
                        RichText::new("blurry, low quality, worst quality, extra fingers…")
                            .size(12.0)
                            .color(t.text_mute),
                    )
                    .font(FontId::proportional(12.0))
                    .text_color(t.text)
                    .margin(Margin::symmetric(8.0, 5.0));
                ui.add_sized([avail, 60.0], edit);
            } else {
                let edit = TextEdit::singleline(&mut state.negative)
                    .id(neg_id)
                    .frame(false)
                    .desired_width(avail)
                    .hint_text(
                        RichText::new("blurry, low quality, worst quality, extra fingers…")
                            .size(12.0)
                            .color(t.text_mute),
                    )
                    .font(FontId::proportional(12.0))
                    .text_color(t.text)
                    .margin(Margin::symmetric(8.0, 5.0));
                ui.add_sized([avail, 22.0], edit);
            }
        });
    });

}

/// Small accent-tinted pill used for the task short-code (T2I, IC-LoRA, …).
/// Sits in the header row next to the PROMPT label.
fn task_badge(ui: &mut Ui, t: &Tokens, label: &str) {
    let padding = vec2(8.0, 3.0);
    let font = FontId::monospace(10.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font.clone(), t.accent_hi);
    let size = galley.size() + padding * 2.0;
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter();
    painter.rect(
        rect,
        2.0,
        t.accent_soft,
        Stroke::new(1.0, t.accent.gamma_multiply(0.5)),
    );
    painter.galley(
        pos2(
            rect.left() + padding.x,
            rect.center().y - galley.size().y * 0.5,
        ),
        galley,
        t.accent_hi,
    );
}
