//! Section: Advanced — four sub-collapsibles gated by the bottom toggle.
//!
//! Per README "Section: Advanced (footer toggle)" (lines 98-104):
//! - ControlNet / img2img
//! - Advanced sampling
//! - Performance
//! - Output
//!
//! These render ABOVE the footer toggle (which lives in `panels.rs` so it
//! can sit pinned outside the scroll region). This module exposes one
//! `show_extras` function the params panel calls in the scroll region
//! when `state.advanced` is true.

use egui::{Frame, Margin, RichText, Stroke, Ui};

use crate::sections::model::labeled_row;
use crate::state::{AppState, ControlNetModel, Mode};
use crate::tokens::{Tokens, FONT_MONO, RADIUS_CONTROL};
use crate::widgets::{combo_enum, combo_str, drag_f32, drag_u32, flat_collapsing, ghost_button};

const ATTENTION_BACKENDS: &[&str] = &["flash-attn-2", "sdpa", "xformers", "math"];
const CPU_OFFLOAD: &[&str] = &["none", "cpu", "sequential", "model"];

/// Render all four advanced collapsibles. Caller must check
/// `state.advanced` first and skip this when false.
pub fn show_extras(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    controlnet_section(ui, state, t);
    sampling_section(ui, state, t);
    performance_section(ui, state, t);
    output_section(ui, state, t);
}

// --- ControlNet / img2img -------------------------------------------------

fn controlnet_section(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    // README line 101: in Video mode this section becomes the first-frame
    // image input for I2V. Title flips and the ControlNet model picker
    // hides — only the source dropzone, strength and denoise stay.
    let mode = state.tab;
    let title = match mode {
        Mode::Image => "ControlNet / img2img",
        Mode::Video => "Input (img2vid)",
    };
    flat_collapsing(ui, t, "sec_adv_cnet", title, false, |ui| {
        // Source image dropzone — placeholder rectangle. Real drag-drop
        // wires up later (Phase 3+).
        let source_label = state
            .current()
            .controlnet
            .source
            .clone()
            .unwrap_or_else(|| "drop image or click to browse".into());

        Frame::none()
            .fill(t.panel_sunk)
            .stroke(Stroke::new(1.0, t.border))
            .rounding(RADIUS_CONTROL)
            .inner_margin(Margin::symmetric(8.0, 14.0))
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("🖼").size(16.0).color(t.text_mute));
                    ui.label(
                        RichText::new(source_label)
                            .size(11.0)
                            .color(t.text_mute),
                    );
                });
            });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ghost_button(ui, t, "Browse…").clicked() {
                // Stub. Real `rfd::FileDialog` integration deferred.
                state.current_mut().controlnet.source =
                    Some("(picked image stub)".into());
            }
            if state.current().controlnet.source.is_some()
                && ghost_button(ui, t, "Clear").clicked()
            {
                state.current_mut().controlnet.source = None;
            }
        });

        ui.add_space(6.0);
        let cn = state.current_mut();

        labeled_row(ui, t, "Strength", |ui| {
            drag_f32(ui, &mut cn.controlnet.strength, 0.0..=1.0, 0.01, 60.0, 2);
        });
        labeled_row(ui, t, "Denoise", |ui| {
            drag_f32(ui, &mut cn.controlnet.denoise, 0.0..=1.0, 0.01, 60.0, 2);
        });
        // ControlNet model picker is image-only. In Video mode the
        // dropzone above is the first-frame image input, not a CN map.
        if matches!(mode, Mode::Image) {
            labeled_row(ui, t, "Model", |ui| {
                combo_enum(
                    ui,
                    "cnet_model",
                    &mut cn.controlnet.model,
                    ControlNetModel::all(),
                    ControlNetModel::label,
                );
            });
        }
    });
}

// --- Advanced sampling ----------------------------------------------------

fn sampling_section(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    flat_collapsing(ui, t, "sec_adv_sampling", "Advanced sampling", false, |ui| {
        let s = &mut state.current_mut().advanced_sampling;

        labeled_row(ui, t, "Clip-skip", |ui| {
            drag_u32(ui, &mut s.clip_skip, 0..=12, 1.0, 60.0);
        });
        labeled_row(ui, t, "Eta", |ui| {
            drag_f32(ui, &mut s.eta, 0.0..=1.0, 0.01, 60.0, 2);
        });
        labeled_row(ui, t, "Sigma min", |ui| {
            drag_f32(ui, &mut s.sigma_min, 0.0..=5.0, 0.01, 60.0, 3);
        });
        labeled_row(ui, t, "Sigma max", |ui| {
            drag_f32(ui, &mut s.sigma_max, 0.0..=99.0, 0.1, 60.0, 2);
        });
        labeled_row(ui, t, "Restart sampling", |ui| {
            ui.checkbox(&mut s.restart_sampling, "");
        });
    });
}

// --- Performance ----------------------------------------------------------

fn performance_section(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    flat_collapsing(ui, t, "sec_adv_perf", "Performance", false, |ui| {
        let p = &mut state.current_mut().perf;

        labeled_row(ui, t, "Attention", |ui| {
            combo_str(ui, "perf_attn", &mut p.attention, ATTENTION_BACKENDS);
        });
        labeled_row(ui, t, "torch.compile", |ui| {
            ui.checkbox(&mut p.torch_compile, "");
        });
        labeled_row(ui, t, "Tiled VAE", |ui| {
            ui.checkbox(&mut p.tiled_vae, "");
        });
        labeled_row(ui, t, "CPU offload", |ui| {
            combo_str(ui, "perf_offload", &mut p.cpu_offload, CPU_OFFLOAD);
        });
        labeled_row(ui, t, "VRAM (GB)", |ui| {
            drag_u32(ui, &mut p.vram_budget_gb, 1..=192, 1.0, 60.0);
        });
    });
}

// --- Output ---------------------------------------------------------------

fn output_section(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    flat_collapsing(ui, t, "sec_adv_output", "Output", false, |ui| {
        let o = &mut state.current_mut().output;

        labeled_row(ui, t, "Folder", |ui| {
            // Read-only path display + browse stub. Spec wants a clickable
            // path; egui's `TextEdit` is the closest single-line widget.
            let avail = ui.available_width() - 80.0;
            ui.add_sized(
                [avail.max(80.0), 22.0],
                egui::TextEdit::singleline(&mut o.folder)
                    .font(egui::TextStyle::Monospace),
            );
            if ghost_button(ui, t, "📂").on_hover_text("Browse…").clicked() {
                // Stub.
                o.folder = r"D:\flame\out\(picked stub)".into();
            }
            if ghost_button(ui, t, "↗").on_hover_text("Open in explorer").clicked() {
                // Stub.
            }
        });

        labeled_row(ui, t, "Filename", |ui| {
            ui.add_sized(
                [ui.available_width(), 22.0],
                egui::TextEdit::singleline(&mut o.filename_template)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("{seed}-{model}-{steps}"),
            );
        });

        labeled_row(ui, t, "Save metadata", |ui| {
            ui.checkbox(&mut o.save_metadata, "");
        });

        // Tiny hint about template variables (mono, muted).
        ui.add_space(4.0);
        ui.label(
            RichText::new("variables: {seed} {model} {steps} {cfg} {w} {h}")
                .size(FONT_MONO - 0.5)
                .monospace()
                .color(t.text_mute),
        );
    });
}
