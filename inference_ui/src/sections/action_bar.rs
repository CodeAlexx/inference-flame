//! Section: Action bar — Generate / Stop · + Queue · spacer · param readout.
//!
//! Per README "Column 2 — Canvas panel · Action bar" (lines 117-121):
//!   - Generate: primary button, amber fill, ~140px wide × 32px tall.
//!     Becomes Stop (red) while running.
//!   - + Queue: ghost button, same height. Phase 5a wires it to the worker
//!     channel (UiMsg::Queue).
//!   - Spacer.
//!   - Right side: muted mono `<W>×<H> · <steps> steps · cfg <cfg> · seed <seed>`.
//!
//! Phase 5a wires Generate/Stop/+Queue to the inference worker via a
//! crossbeam Sender passed in by `panels::canvas_panel`. Generate also
//! pre-populates `state.queue.running` so the right panel reflects the new
//! job immediately, before the worker's `Started` event arrives. The
//! `WorkerEvent::Started` handler in `app.rs` won't overwrite the slot when
//! the ids match.

use crossbeam_channel::Sender;
use egui::{vec2, Align, Button, Color32, Layout, RichText, Stroke, Ui};

use crate::state::{AppState, Mode, QueueJob};
use crate::tokens::{Tokens, FONT_MONO};
use crate::worker::{paths::resolve_image_model_path, GenerateJob, ModelKind, UiMsg};

pub fn show(
    ui: &mut Ui,
    state: &mut AppState,
    t: &Tokens,
    worker_tx: &Sender<UiMsg>,
    next_job_id: &mut u64,
) {
    ui.horizontal(|ui| {
        // --- Primary Generate / Stop button ----------------------------
        let (label, fill, fg) = if state.generating {
            ("⏹ Stop", t.danger, Color32::WHITE)
        } else {
            ("▶ Generate", t.accent, t.bg)
        };

        // Hover brightens the fill — egui doesn't auto-blend on `.fill()`,
        // so we look at a hover hint manually via interact-style swap.
        // Simpler and good enough: render with the base fill; egui's
        // selection/hovered visuals add a subtle outline on hover.
        let btn = Button::new(
            RichText::new(label)
                .size(13.0)
                .strong()
                .color(fg),
        )
        .fill(fill)
        .stroke(Stroke::NONE)
        .min_size(vec2(140.0, 32.0))
        .rounding(crate::tokens::RADIUS_CONTROL);

        if ui.add(btn).clicked() {
            if state.generating {
                // Stop = cancel the worker. We DON'T flip `state.generating`
                // locally here — let the worker confirm via `Failed` event,
                // which is what actually clears the spinner. This mirrors
                // how a real cancel races with an in-flight step (it might
                // already be on the last step when Cancel arrives, in which
                // case `Done` wins and the UI shows the result anyway).
                let _ = worker_tx.send(UiMsg::Cancel);
            } else {
                // Generate = mint a job id, build the snapshot, send to
                // worker, and pre-populate the running slot so the queue
                // panel updates instantly. Total steps will be reaffirmed
                // by the Started event but we set them here so the canvas
                // toolbar reads `step 0/<configured>` immediately rather
                // than `0/0` for the first frame.
                let id = *next_job_id;
                *next_job_id += 1;
                let job = build_job_from_state(id, state);
                // Snapshot the fields we need from `state.current()` *before*
                // touching `state.queue.running` — borrowck rejects holding
                // an immutable `&ModeSettings` (via `current()`) while
                // simultaneously assigning `state.queue.running`.
                let (w, h, steps, sampler) = {
                    let cn = state.current();
                    (cn.width, cn.height, cn.steps, cn.sampler.clone())
                };
                let queue_job = QueueJob {
                    id,
                    prompt: state.prompt.clone(),
                    width: w,
                    height: h,
                    steps,
                    sampler,
                    progress: 0.0,
                    eta_secs: None,
                };
                state.queue.running = Some(queue_job);
                state.generating = true;
                state.current_step = 0;
                state.total_steps = steps;
                let _ = worker_tx.send(UiMsg::Generate { job });
            }
        }

        ui.add_space(4.0);

        // --- + Queue (ghost) -------------------------------------------
        // Spec: ghost button, same 32px height. Phase 5a wires it.
        let queue_btn = Button::new(
            RichText::new("+ Queue")
                .size(12.0)
                .color(t.text_dim),
        )
        .fill(Color32::TRANSPARENT)
        .stroke(Stroke::new(1.0, t.border))
        .min_size(vec2(0.0, 32.0))
        .rounding(crate::tokens::RADIUS_CONTROL);

        if ui.add(queue_btn).clicked() {
            // Mint id, snapshot params, and push to BOTH the worker queue
            // AND the visible state.queue.queued list. The worker has its
            // own internal queue (we can't read it from here) so the
            // user-facing list is the UI's own mirror — kept in sync by
            // sending the same ids both places. Phase 5b real wiring will
            // need a sync event if jobs ever get rejected by the worker
            // (eg. validation failure); the current mock accepts everything.
            let id = *next_job_id;
            *next_job_id += 1;
            let job = build_job_from_state(id, state);
            // Same borrow split as the Generate branch above.
            let (w, h, steps, sampler) = {
                let cn = state.current();
                (cn.width, cn.height, cn.steps, cn.sampler.clone())
            };
            state.queue.queued.push(QueueJob {
                id,
                prompt: state.prompt.clone(),
                width: w,
                height: h,
                steps,
                sampler,
                progress: 0.0,
                eta_secs: None,
            });
            let _ = worker_tx.send(UiMsg::Queue { job });
        }

        // --- Spacer + readout ------------------------------------------
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let cn = state.current();
            let mut parts: Vec<String> = Vec::with_capacity(5);
            // Resolution segment — gate the frame-count suffix on the active
            // tab (Mode), not on `cn.frames.is_some()`. The latter would also
            // grow a `· Nf` segment if anyone ever set `image.frames = Some(N)`
            // (animated PNG, etc.) on the image side. Mode is the source of
            // truth per spec line 121 + JSX line 60.
            let res = if state.tab == Mode::Video {
                let f = cn.frames.unwrap_or(0);
                format!("{}×{} · {}f", cn.width, cn.height, f)
            } else {
                format!("{}×{}", cn.width, cn.height)
            };
            parts.push(res);
            parts.push(format!("{} steps", cn.steps));
            parts.push(format!("cfg {:.1}", cn.cfg));
            parts.push(format!("seed {}", state.seed));
            // R-to-L layout reverses item order; build the string already
            // joined with " · " separators so we can ship a single label.
            let text = parts.join("  ·  ");
            ui.label(
                RichText::new(text)
                    .size(FONT_MONO)
                    .monospace()
                    .color(t.text_dim),
            );
        });
    });
}

/// Snapshot the current params into a `GenerateJob`. Pulled out so both
/// Generate and +Queue use identical logic. `app.rs::build_job` is the
/// equivalent on the App side; we duplicate here so this section doesn't
/// need a back-reference to `FlameInferenceApp`.
fn build_job_from_state(id: u64, state: &AppState) -> GenerateJob {
    let cn = state.current();
    // Phase 5b: derive `model_kind` from the Base model string. Anything
    // we don't recognize falls through to Mock (synthetic gradient) so
    // the UI still produces a placeholder image.
    let model_kind = ModelKind::from_model_string(&cn.model);
    // Resolve the ComboBox filename → absolute disk path so the worker's
    // hardcoded .safetensors default doesn't silently override a GGUF
    // selection. `None` means "worker uses its hardcoded default path" —
    // see `worker/paths.rs` for the per-ModelKind resolution strategy.
    let path = resolve_image_model_path(&cn.model, model_kind);
    GenerateJob {
        id,
        model_kind,
        prompt: state.prompt.clone(),
        negative: state.negative.clone(),
        width: cn.width,
        height: cn.height,
        steps: cn.steps,
        cfg: cn.cfg,
        seed: state.seed,
        sampler: cn.sampler.clone(),
        scheduler: cn.scheduler.clone(),
        path,
    }
}
