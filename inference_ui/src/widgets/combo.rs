//! ComboBox wrappers — one for `&str` lists, one for `Copy` enums.
//!
//! egui's `ComboBox` API is fine but verbose. These two helpers cover every
//! select in the params panel: string lists for VAE/Base/Sampler/etc., and
//! enum lists for Task/Precision/SeedMode/ControlNetModel.

use egui::{ComboBox, Ui};

/// Returns `true` if a different option was picked this frame.
pub fn combo_str(
    ui: &mut Ui,
    id: &str,
    value: &mut String,
    options: &[&str],
) -> bool {
    let mut changed = false;
    ComboBox::from_id_salt(id)
        .width(ui.available_width().min(220.0))
        .selected_text(value.clone())
        .show_ui(ui, |ui| {
            for opt in options {
                if ui
                    .selectable_label(value.as_str() == *opt, *opt)
                    .clicked()
                {
                    *value = (*opt).to_string();
                    changed = true;
                }
            }
        });
    changed
}

/// Generic enum picker. Caller supplies the list and the label fn.
/// Returns `true` if a different variant was picked this frame.
pub fn combo_enum<T: Copy + PartialEq>(
    ui: &mut Ui,
    id: &str,
    value: &mut T,
    options: &[T],
    label: impl Fn(T) -> &'static str,
) -> bool {
    let mut changed = false;
    ComboBox::from_id_salt(id)
        .width(ui.available_width().min(220.0))
        .selected_text(label(*value))
        .show_ui(ui, |ui| {
            for opt in options {
                if ui.selectable_label(*value == *opt, label(*opt)).clicked() {
                    *value = *opt;
                    changed = true;
                }
            }
        });
    changed
}
