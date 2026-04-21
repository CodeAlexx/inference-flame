//! Three-column layout shell: params (left, 316px) · canvas (center) · queue (right, 260px).
//!
//! Phase 2 fills the params panel with all section widgets. The queue and
//! canvas panels remain placeholder until Phase 3/4.

use eframe::egui;
use egui::{CentralPanel, Frame, Margin, RichText, ScrollArea, SidePanel, Stroke, TopBottomPanel};

use crate::app::FlameInferenceApp;
use crate::sections;
use crate::tokens::{FONT_SECTION_LABEL, OUTER_PAD, PARAMS_W, QUEUE_W};

// `section_header` was a Phase 1 placeholder for the right-panel "QUEUE /
// HISTORY" label. Phase 4 replaces that label with the real tab bar
// (`Queue (N)` / `History (N)`) inside `sections::queue`, so the helper is
// no longer used. Removed rather than left around with `#[allow(dead_code)]`.

pub fn params_panel(ctx: &egui::Context, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    SidePanel::left("params")
        .default_width(PARAMS_W)
        .resizable(true)
        .min_width(240.0)
        .max_width(480.0)
        .frame(
            Frame::none()
                .fill(t.panel)
                .stroke(Stroke::new(1.0, t.border))
                // 6px outer pad on the LEFT (window edge) + 6px top/bottom.
                // No right pad — the 6px gap between columns is provided by
                // the central panel's inner_margin.left = OUTER_PAD.
                .outer_margin(Margin {
                    left: OUTER_PAD,
                    right: 0.0,
                    top: OUTER_PAD,
                    bottom: OUTER_PAD,
                })
                .inner_margin(Margin::same(8.0)),
        )
        .show(ctx, |ui| {
            // Header row: PARAMETERS label on the left, search icon on the
            // right (placeholder, not wired).
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("PARAMETERS")
                        .size(FONT_SECTION_LABEL)
                        .monospace()
                        .color(t.text_dim)
                        .strong(),
                );
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        // Search icon — Phase 2 placeholder. No interaction
                        // wired (search itself is out of scope for now).
                        ui.label(RichText::new("🔍").size(11.0).color(t.text_mute));
                    },
                );
            });
            ui.add_space(6.0);

            // Pinned Advanced footer must live OUTSIDE the scroll area,
            // anchored to the bottom of the params panel. egui's
            // `TopBottomPanel::bottom` works inside any `Ui`, so we use it
            // here scoped to the params panel — that gives us a stable
            // bottom strip with the rest of the params content scrolling
            // above it.
            advanced_footer(ui, app);

            // Scrollable body — sections in spec order.
            ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    let theme_tokens = t;
                    sections::model::show(ui, &mut app.state, &theme_tokens);
                    ui.add_space(10.0);
                    sections::resolution::show(ui, &mut app.state, &theme_tokens);
                    ui.add_space(10.0);
                    sections::sampling::show(ui, &mut app.state, &theme_tokens);
                    ui.add_space(10.0);
                    sections::seed::show(ui, &mut app.state, &theme_tokens);
                    ui.add_space(10.0);
                    sections::batch::show(ui, &mut app.state, &theme_tokens);
                    ui.add_space(10.0);
                    sections::lora::show(ui, &mut app.state, &theme_tokens);
                    if app.state.advanced {
                        ui.add_space(10.0);
                        sections::advanced::show_extras(ui, &mut app.state, &theme_tokens);
                    }
                    ui.add_space(8.0);
                });
        });
}

/// The pinned-bottom Advanced toggle. Lives outside the scroll region per
/// README spec ("Pinned to the bottom of the params panel, not inside the
/// scroll region").
fn advanced_footer(ui: &mut egui::Ui, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    TopBottomPanel::bottom("params_advanced_footer")
        .resizable(false)
        .frame(
            Frame::none()
                .fill(t.panel)
                .stroke(Stroke {
                    width: 1.0,
                    color: t.border,
                })
                .inner_margin(Margin::symmetric(10.0, 6.0)),
        )
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                let mut adv = app.state.advanced;
                if ui
                    .checkbox(
                        &mut adv,
                        RichText::new("Advanced").size(11.0).color(t.text_dim),
                    )
                    .changed()
                {
                    app.state.advanced = adv;
                }
                if app.state.advanced {
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("+4 sections")
                            .size(10.0)
                            .monospace()
                            .color(t.accent),
                    );
                }
            });
        });
}

pub fn queue_panel(ctx: &egui::Context, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    // Same borrow-splitting trick as canvas_panel: clone the Sender so the
    // queue section can fire UiMsg::Cancel / RemoveQueued without needing a
    // reference to `app` itself.
    let worker_tx = app.worker.tx.clone();
    SidePanel::right("queue")
        .default_width(QUEUE_W)
        .resizable(true)
        .min_width(200.0)
        .max_width(420.0)
        .frame(
            Frame::none()
                .fill(t.panel)
                .stroke(Stroke::new(1.0, t.border))
                // Mirror of params_panel: 6px on RIGHT (window edge) + top/bottom.
                .outer_margin(Margin {
                    left: 0.0,
                    right: OUTER_PAD,
                    top: OUTER_PAD,
                    bottom: OUTER_PAD,
                })
                // No inner_margin — the tab bar sits flush with the top
                // edge (matches JSX `borderBottom` on the tab container),
                // and the perf footer provides its own padding. The queue
                // body's frames have their own inner_margin so content
                // still has breathing room. This diverges from params
                // panel (which does use `Margin::same(8.0)`) but matches
                // the JSX `noPad` prop on the queue-panel `Panel`.
                .inner_margin(Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            // Perf footer pinned to bottom — must be declared BEFORE the
            // body so the central area calculates its height correctly
            // (same ordering as the params panel's Advanced footer).
            sections::perf_footer::pinned(ui, &app.state, &t);

            // Scrollable body (tab bar + queue/history content). The
            // `queue::show` function owns its own ScrollArea internally
            // so the tab bar stays pinned as the user scrolls job lists.
            sections::queue::show(ui, &mut app.state, &t, &worker_tx);
        });
}

pub fn canvas_panel(ctx: &egui::Context, app: &mut FlameInferenceApp) {
    let t = app.state.theme.tokens();
    // Borrow-splitting: the action_bar needs `&mut state` AND `&worker.tx +
    // &mut next_job_id` simultaneously, and the canvas needs `&state` plus
    // `&Option<TextureHandle>`. Pulling the field references out before the
    // closure means each child sees disjoint borrows of `app`.
    //
    // We also need to allocate job ids (which mutates `app.next_job_id`) and
    // build jobs from the live state from inside the action_bar callback.
    // To keep the action_bar API straightforward we use a small closure
    // (`mint_job`) that captures `&mut next_job_id` + `&state`. The Sender
    // is `Clone`, so the action_bar gets its own clone to send through.
    let worker_tx = app.worker.tx.clone();
    CentralPanel::default()
        .frame(
            Frame::none()
                .fill(t.bg)
                // Inner margin doubles as both the outer pad on top/bottom and
                // the inter-column gap on left/right. README spec: 6px pad +
                // 6px column gap.
                .inner_margin(Margin::same(OUTER_PAD)),
        )
        .show(ctx, |ui| {
            // Phase 3 layout (top→bottom):
            //   1) Prompt section (header row + prompt box + negative)
            //   2) Action bar (Generate/Stop · +Queue · spacer · readout)
            //   3) Canvas surface + toolbar (fills remaining vertical space)
            //
            // The canvas surface itself paints its own checker bg directly —
            // no inner card wrapper (per SKEPTIC_PHASE1 P1 #19).
            sections::prompt::show(ui, &mut app.state, &t);
            ui.add_space(6.0);
            // Phase 5a: action_bar wires Generate/Stop/+Queue to the worker
            // channel. It needs both `&mut state` and the worker tx + a way
            // to allocate fresh job ids — passed in directly here so the
            // section module stays free of any reference to `FlameInferenceApp`.
            sections::action_bar::show(
                ui,
                &mut app.state,
                &t,
                &worker_tx,
                &mut app.next_job_id,
            );
            ui.add_space(6.0);
            // Phase 5a: canvas reads `app.last_image` to paint completed
            // results into the preview rect. The texture handle outlives the
            // closure, so a borrow is fine.
            sections::canvas::show(ui, &mut app.state, &t, app.last_image.as_ref());
        });
}

