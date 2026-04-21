//! Section: Sampling — Steps/CFG (one row), Sampler, Scheduler.
//!
//! Per README "Section: Sampling" (lines 74-78). Steps + CFG share a row
//! with a `/` separator. README explicitly bans sliders here — only
//! DragValues.

use egui::{RichText, Ui};

use crate::sections::model::labeled_row;
use crate::state::AppState;
use crate::tokens::Tokens;
use crate::widgets::{combo_str, drag_f32, drag_u32, flat_collapsing};

const SAMPLERS: &[&str] = &[
    "Euler",
    "Euler a",
    "DPM++ 2M",
    "DPM++ 2M SDE",
    "DPM++ 3M SDE",
    "UniPC",
    "LCM",
    "DDIM",
    "Heun",
    "LMS",
];

const SCHEDULERS: &[&str] = &[
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "normal",
    "beta",
    "ddim_uniform",
];

pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    let cn = state.current_mut();

    flat_collapsing(ui, t, "sec_sampling", "Sampling", true, |ui| {
        labeled_row(ui, t, "Steps / CFG", |ui| {
            drag_u32(ui, &mut cn.steps, 1..=150, 1.0, 56.0);
            ui.label(RichText::new("/").size(11.0).color(t.text_mute));
            drag_f32(ui, &mut cn.cfg, 1.0..=30.0, 0.1, 56.0, 1);
        });

        labeled_row(ui, t, "Sampler", |ui| {
            combo_str(ui, "sampler", &mut cn.sampler, SAMPLERS);
        });

        labeled_row(ui, t, "Scheduler", |ui| {
            combo_str(ui, "scheduler", &mut cn.scheduler, SCHEDULERS);
        });
    });
}
