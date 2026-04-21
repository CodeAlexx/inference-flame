//! Flame Inference — eframe entry point.

mod app;
mod chrome;
mod panels;
mod perf;
mod persist;
mod sections;
mod state;
mod theme;
mod tokens;
mod widgets;
mod worker;

use eframe::egui;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_transparent(false)
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([1024.0, 640.0])
            .with_title("Flame"),
        ..Default::default()
    };

    eframe::run_native(
        "Flame",
        native_options,
        Box::new(|cc| Ok(Box::new(app::FlameInferenceApp::new(cc)))),
    )
}
