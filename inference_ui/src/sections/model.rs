//! Section: Model — Task / Base / VAE / Precision + info strip.
//!
//! Per README "Section: Model" (lines 58-65). Task options depend on the
//! current Mode (Image: T2I/I2I/IC-LoRA; Video: T2V/I2V/A2V/IC-LoRA).
//! Base + VAE pickers use placeholder lists; the real disk scan lands in
//! a later phase (out-of-scope per Phase 2 brief).

use egui::{Frame, Margin, RichText, Stroke, Ui};

use crate::state::{AppState, Mode, Precision, Task};
use crate::tokens::{Tokens, FONT_MONO, RADIUS_CONTROL};
use crate::widgets::{combo_enum, combo_str, flat_collapsing};

// Placeholder model lists (mirrors `MOCK_MODELS` in params-panel.jsx). Real
// scans of `D:\flame\weights\{image|video}` are deferred per Phase 2 brief.
//
// Phase 5b: `z-image-base.safetensors` and `z-image-turbo.safetensors`
// strings are matched by `ModelKind::from_model_string` to dispatch to the
// real Z-Image inference worker. Other entries here still route to Mock
// until those models get wired up.
// Placeholder model list. Matches `ModelKind::from_model_string` dispatch
// arms; the `.gguf` entries coexist with `.safetensors` because each worker
// dispatches on the resolved path's suffix (see e.g. `worker/zimage.rs`
// `ensure_dit`). Keeps two GGUF quant variants per family (Q4_K_M + Q8_0)
// as a reasonable default — user renames their actual file to match or we
// surface a "not found at <path>" error. AGENT-DEFAULT: chose Q4_K_M + Q8_0
// over enumerating every possible type (Q2..Q6) to avoid ComboBox bloat.
// GGUF DiT files that use BlockOffloader (flux, chroma, qwen-image) are
// still listed so the user sees them, but the workers return an explicit
// "GGUF + BlockOffloader not yet supported" error at load time.
const IMAGE_MODELS: &[&str] = &[
    "z-image-base.safetensors",
    "z-image-base-Q4_K_M.gguf",
    "z-image-base-Q8_0.gguf",
    "z-image-turbo.safetensors",
    "z-image-turbo-Q4_K_M.gguf",
    "z-image-turbo-Q8_0.gguf",
    "flux1-dev.safetensors",
    "flux1-dev-Q4_K_M.gguf",
    "flux1-dev-Q8_0.gguf",
    "flux1-schnell.safetensors",
    "sd3.5-large.safetensors",
    "sdxl-base-1.0.safetensors",
    "sdxl-base-1.0-Q4_K_M.gguf",
    "sdxl-base-1.0-Q8_0.gguf",
    "sdxl-turbo.safetensors",
    "pony-diffusion-v6.safetensors",
    // Klein (Flux 2 base) variants. Filenames intentionally include the
    // size token (4b / 9b) so `ModelKind::from_model_string` routes each
    // to its correct dispatch arm.
    "klein-4b.safetensors",
    "klein-4b-Q4_K_M.gguf",
    "klein-4b-Q8_0.gguf",
    "klein-9b.safetensors",
    "klein-9b-Q4_K_M.gguf",
    "klein-9b-Q8_0.gguf",
    "chroma.safetensors",
    "chroma-Q4_K_M.gguf",
    "chroma-Q8_0.gguf",
    // Batch C additions. Filename strings carry the discriminator that
    // `ModelKind::from_model_string` matches on (`sd3` / `qwen` / `ernie` /
    // `anima`). The actual on-disk paths are hardcoded inside each worker —
    // see `worker/{sd3,qwenimage,ernie,anima}.rs`.
    "sd3.5-medium.safetensors",
    "sd3.5-medium-Q4_K_M.gguf",
    "sd3.5-medium-Q8_0.gguf",
    "qwen-image.safetensors",
    "qwen-image-Q4_K_M.gguf",
    "qwen-image-Q8_0.gguf",
    "ernie-image-8b.safetensors",
    "ernie-image-8b-Q4_K_M.gguf",
    "ernie-image-8b-Q8_0.gguf",
    "anima-2b.safetensors",
    "anima-2b-Q4_K_M.gguf",
    "anima-2b-Q8_0.gguf",
    // Batch D additions — legacy VE image models (ε-prediction, scaled-linear
    // β schedule, NOT flow). `sdxl-*.safetensors` entries above already
    // dispatch to the Sdxl arm via `ModelKind::from_model_string`.
    // `sd15.safetensors` and `stable-cascade.safetensors` match their
    // respective ModelKind arms — see `worker/{sdxl,sd15,cascade}.rs`.
    // Actual on-disk weight paths are hardcoded inside each worker.
    "sd15.safetensors",
    "sd15-Q4_K_M.gguf",
    "sd15-Q8_0.gguf",
    "stable-cascade.safetensors",
    "stable-cascade-Q4_K_M.gguf",
    "stable-cascade-Q8_0.gguf",
];

const VIDEO_MODELS: &[&str] = &[
    "klein9b.safetensors",
    "wan-2.1-i2v-14B.safetensors",
    "hunyuan-video-t2v.safetensors",
    "ltx-video-0.9.safetensors",
    "mochi-1-preview.safetensors",
];

const VAE_MODELS: &[&str] = &[
    "ae.safetensors (auto)",
    "sdxl-vae-fp16-fix",
    "flux-vae",
    "taesd",
];

pub fn show(ui: &mut Ui, state: &mut AppState, t: &Tokens) {
    let mode = state.tab;
    let cn = state.current_mut();

    flat_collapsing(ui, t, "sec_model", "Model", true, |ui| {
        // Task picker — options depend on current Mode. If the stored task
        // doesn't belong to this mode (e.g. user defaulted via persisted
        // state), the ComboBox will display its label but selecting a new
        // option clamps it back into the valid set.
        labeled_row(ui, t, "Task", |ui| {
            let valid = match mode {
                Mode::Image => Task::for_image(),
                Mode::Video => Task::for_video(),
            };
            combo_enum(ui, "task", &mut cn.task, valid, Task::label);
        });

        labeled_row(ui, t, "Base", |ui| {
            let opts = match mode {
                Mode::Image => IMAGE_MODELS,
                Mode::Video => VIDEO_MODELS,
            };
            combo_str(ui, "base", &mut cn.model, opts);
        });

        labeled_row(ui, t, "VAE", |ui| {
            combo_str(ui, "vae", &mut cn.vae, VAE_MODELS);
        });

        labeled_row(ui, t, "Precision", |ui| {
            combo_enum(
                ui,
                "precision",
                &mut cn.precision,
                Precision::all(),
                Precision::label,
            );
        });

        // Info strip — placeholder size + status. The model size lookup
        // is a stub (flat 23.8 GB if image, 28.2 GB if video) — Phase 2
        // brief explicitly defers real disk metadata.
        ui.add_space(6.0);
        let stub_size = match mode {
            Mode::Image => "23.8 GB",
            Mode::Video => "28.2 GB",
        };
        let info = format!("{stub_size} · loaded · gpu:0");
        Frame::none()
            .fill(t.panel_sunk)
            .stroke(Stroke::new(1.0, t.border_soft))
            .rounding(RADIUS_CONTROL)
            .inner_margin(Margin::symmetric(8.0, 6.0))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(info)
                        .size(FONT_MONO)
                        .monospace()
                        .color(t.text_mute),
                );
            });
    });
}

/// Two-column row: 80px label on the left, body fills the rest. Mirrors
/// the JSX `Row` primitive (params-panel.jsx) but compact for egui.
pub(super) fn labeled_row<R>(
    ui: &mut Ui,
    t: &Tokens,
    label: &str,
    body: impl FnOnce(&mut Ui) -> R,
) -> R {
    let mut out = None;
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(80.0, 22.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(RichText::new(label).size(11.5).color(t.text_dim));
            },
        );
        out = Some(body(ui));
    });
    ui.add_space(2.0);
    out.unwrap()
}
