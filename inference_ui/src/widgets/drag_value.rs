//! `DragValue` wrappers that pin the README "Spacing" widths.
//!
//! Three flavors: `u32` (frames, fps, batch, steps), `i64` (seed), `f32`
//! (cfg, strength, eta, sigma, denoise). The width is the only spec
//! variable — height comes from `interact_size.y = 22` set in
//! `theme::apply_density`.

use egui::{DragValue, Response, Ui};
use std::ops::RangeInclusive;

pub fn drag_u32(
    ui: &mut Ui,
    value: &mut u32,
    range: RangeInclusive<u32>,
    speed: f64,
    width: f32,
) -> Response {
    ui.add_sized(
        [width, 22.0],
        DragValue::new(value).range(range).speed(speed),
    )
}

pub fn drag_i64(
    ui: &mut Ui,
    value: &mut i64,
    range: RangeInclusive<i64>,
    speed: f64,
    width: f32,
) -> Response {
    ui.add_sized(
        [width, 22.0],
        DragValue::new(value).range(range).speed(speed),
    )
}

pub fn drag_f32(
    ui: &mut Ui,
    value: &mut f32,
    range: RangeInclusive<f32>,
    speed: f64,
    width: f32,
    fixed_decimals: usize,
) -> Response {
    ui.add_sized(
        [width, 22.0],
        DragValue::new(value)
            .range(range)
            .speed(speed)
            .fixed_decimals(fixed_decimals),
    )
}
