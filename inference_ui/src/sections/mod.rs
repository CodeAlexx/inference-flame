//! Phase 2: parameter panel sections. One module per `CollapsingHeader`.
//!
//! Each module exposes `pub fn show(ui, state, t)` and paints itself.
//! `panels::params_panel` calls them in spec order.
//!
//! Phase 3 adds canvas-panel sections (`prompt`, `action_bar`, `canvas`) — they
//! follow the same `pub fn show(ui, ...)` shape and are wired into
//! `panels::canvas_panel` instead of `params_panel`.

pub mod advanced;
pub mod batch;
pub mod lora;
pub mod model;
pub mod resolution;
pub mod sampling;
pub mod seed;

// Phase 3 — canvas panel sections.
pub mod action_bar;
pub mod canvas;
pub mod prompt;

// Phase 4 — right-panel queue/history + perf footer.
pub mod perf_footer;
pub mod queue;
