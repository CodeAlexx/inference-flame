//! Reusable widgets for Flame Inference.
//!
//! Phase 2: thin wrappers around egui primitives that pin the spec'd
//! widths/heights/colors so individual section modules don't repeat the
//! styling. These are intentionally low-power — they just enforce the
//! README "Spacing" + "Typography" rules so we can swap egui versions
//! without re-tuning every section.

pub mod combo;
pub mod drag_value;
pub mod ghost_button;
pub mod progress_bar;
pub mod section_header;
pub mod thumbnail;

pub use combo::{combo_enum, combo_str};
pub use drag_value::{drag_f32, drag_i64, drag_u32};
pub use ghost_button::ghost_button;
pub use section_header::flat_collapsing;
