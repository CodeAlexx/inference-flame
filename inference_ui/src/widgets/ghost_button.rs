//! "Ghost" outline-style button used for `+ Add LoRA` and similar
//! secondary actions. README spec: 22px tall, 3px radius, transparent
//! fill, 1px border in `border`, hover fill = `row`.

use egui::{Button, Color32, Response, Stroke, Ui};

use crate::tokens::Tokens;

pub fn ghost_button(ui: &mut Ui, t: &Tokens, label: &str) -> Response {
    let btn = Button::new(label)
        .fill(Color32::TRANSPARENT)
        .stroke(Stroke::new(1.0, t.border))
        .min_size(egui::vec2(0.0, 22.0));
    ui.add(btn)
}
