//! Section: Seed — value DragValue + 🎲 randomize + 🔒 lock + mode picker.
//!
//! Per README "Section: Seed" (lines 80-83). The seed value is shared
//! across modes (lives on `AppState`, not `ModeSettings`).

use egui::Ui;
use rand::Rng;

use crate::sections::model::labeled_row;
use crate::state::{AppState, SeedMode};
use crate::tokens::Tokens;
use crate::widgets::{combo_enum, drag_i64, flat_collapsing};

pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    flat_collapsing(ui, t, "sec_seed", "Seed", true, |ui| {
        labeled_row(ui, t, "Seed", |ui| {
            drag_i64(ui, &mut state.seed, -1..=9_999_999_999, 1.0, 90.0);
            // 🎲 randomize. Writes a fresh seed in [0, 9_999_999_999] to
            // cover the full 10-digit seed range the DragValue accepts.
            if ui.small_button("🎲").on_hover_text("Randomize seed").clicked() {
                let mut rng = rand::thread_rng();
                state.seed = rng.gen_range(0u64..=9_999_999_999u64) as i64;
            }
            // 🔒 lock. Toggles `seed_locked` — actual "prevents advancing"
            // behavior wires up when the worker pulls a seed for each gen
            // (Phase 3+). Phase 2 just persists the bit.
            let lock_label = if state.seed_locked { "🔒" } else { "🔓" };
            if ui
                .small_button(lock_label)
                .on_hover_text(if state.seed_locked {
                    "Locked (will not advance)"
                } else {
                    "Unlocked"
                })
                .clicked()
            {
                state.seed_locked = !state.seed_locked;
            }
        });

        labeled_row(ui, t, "Mode", |ui| {
            combo_enum(
                ui,
                "seed_mode",
                &mut state.seed_mode,
                SeedMode::all(),
                SeedMode::label,
            );
        });
    });
}
