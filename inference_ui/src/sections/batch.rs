//! Section: Batch — Count + Size, Size has a VRAM hover hint.
//!
//! Per README "Section: Batch" (lines 85-86). Both fields are shared
//! across modes (live on AppState).

use egui::Ui;

use crate::sections::model::labeled_row;
use crate::state::AppState;
use crate::tokens::Tokens;
use crate::widgets::{drag_u32, flat_collapsing};

pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    flat_collapsing(ui, t, "sec_batch", "Batch", true, |ui| {
        labeled_row(ui, t, "Count", |ui| {
            drag_u32(ui, &mut state.batch_count, 1..=64, 1.0, 60.0);
        });
        labeled_row(ui, t, "Size", |ui| {
            drag_u32(ui, &mut state.batch_size, 1..=8, 1.0, 60.0)
                .on_hover_text("VRAM cost scales linearly with batch size");
        });
    });
}
