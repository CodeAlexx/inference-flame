//! Flat `CollapsingHeader` styled per README "Column 1 — Parameters panel":
//! "Sections are `egui::CollapsingHeader` styled to be **flat** (no chevron
//! background, no indent), with a caret triangle and uppercase 11px label.
//! Open by default."
//!
//! egui doesn't expose much in the way of CollapsingHeader styling. We do
//! what we can: small uppercase RichText label, default-open. The chevron
//! background cannot be removed via the public API in egui 0.29 — that
//! would need either a fork or a manual collapsible reimplementation.
//! AGENT-DEFAULT: ship the stock collapsing header for now; if the user
//! wants pixel-perfect flat headers we'll roll a custom one in a follow-up.

use egui::{CollapsingHeader, RichText, Ui};

use crate::tokens::{Tokens, FONT_SECTION_LABEL};

/// Open a collapsible section with the spec'd label styling. Body runs
/// inside the open region; the caller adds inner widgets.
pub fn flat_collapsing<R>(
    ui: &mut Ui,
    t: &Tokens,
    id_salt: &str,
    title: &str,
    default_open: bool,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    // Build the styled label first, then hand it to CollapsingHeader so the
    // RichText weight/color sticks (avoids the SKEPTIC P0 #2 mistake).
    let label = RichText::new(title.to_uppercase())
        .size(FONT_SECTION_LABEL)
        .color(t.text_dim)
        .strong();

    CollapsingHeader::new(label)
        .id_salt(id_salt)
        .default_open(default_open)
        .show(ui, add_contents)
        .body_returned
}
