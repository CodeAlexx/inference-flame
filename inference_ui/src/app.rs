//! Top-level application state + `eframe::App::update` loop.
//!
//! Phase 1 owned `AppState` and orchestrated the chrome + three-column
//! layout. Phase 5a adds the inference worker: a background thread that
//! emits Started/Progress/Done/Failed events over a crossbeam channel. The
//! UI drains those events at the top of every `update()` call and applies
//! them to `AppState`.
//!
//! The worker handle and the last completed image deliberately live on
//! `FlameInferenceApp`, not `AppState` — they're transient process state and
//! don't need to round-trip through the Phase 6 RON persist layer. Keeping
//! them here means `AppState` stays a pure data container with serde derives,
//! and the worker handle never needs to be `Clone`/`Serialize`.

use std::time::{Duration, Instant};

use eframe::egui;

use crate::perf::NvmlBackend;
use crate::persist::{self, DebouncedSave};
use crate::state::{AppState, HistoryItem, Mode, QueueJob};
use crate::worker::{spawn_worker, WorkerEvent, WorkerHandle};

pub struct FlameInferenceApp {
    pub state: AppState,
    /// Channel handle for talking to the inference worker thread. Dropped
    /// on app exit, which closes the channel and triggers the worker's
    /// `recv_timeout` to return `Disconnected` so the thread terminates.
    pub worker: WorkerHandle,
    /// GPU texture for the most recently completed result image. Lazily
    /// uploaded inside `apply_event` when a `Done` event arrives. The
    /// canvas section reads this and paints it into the preview rect; if
    /// `None`, the canvas falls back to the warm gradient placeholder.
    pub last_image: Option<egui::TextureHandle>,
    /// Monotonic job-id counter. Each Generate/Queue click bumps this and
    /// stamps the next job. We start at 1000 so the Phase 4 mock queue ids
    /// (1..=4 / 100..=104) don't collide with newly minted real ones.
    pub next_job_id: u64,
    /// NVML telemetry backend. `None` if `Nvml::init()` failed (no NVIDIA
    /// driver on this box, AMD/Intel GPU, headless container, etc.) — the
    /// perf footer then shows whatever's currently in `state.perf` (the
    /// Phase 4 mock values from `PerfTelemetry::mock()`). Never panics.
    pub perf_backend: Option<NvmlBackend>,
    /// Last NVML poll timestamp, gated by the spec'd cadence (500ms while
    /// generating, 1000ms idle). Set once on `new()` and updated on every
    /// successful poll.
    pub last_perf_poll: Instant,
    /// AppState persistence debouncer. Marks dirty on detected change,
    /// fires `persist::save` once the user has paused for ~500ms.
    pub save_debouncer: DebouncedSave,
    /// Previous-frame state serialized to RON. Compared against the
    /// current frame's serialization at end-of-update to detect changes.
    /// AGENT-DEFAULT: RON-string compare instead of `Hash` derive — many
    /// nested structs hold `f32` fields (cfg, strength, eta, sigma_*),
    /// and `f32: !Hash`. Wrapping every float to bits would be invasive.
    /// String compare is O(state-size) per frame; AppState is a few KB so
    /// it's negligible (<<1ms per frame on any modern CPU).
    pub state_prev_ron: String,
}

impl FlameInferenceApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Phase 6: try to load persisted state first, fall back to defaults.
        // Loaded state has `serde(skip)` fields like `queue` / `perf` /
        // `generating` reset to their `Default` impls — which means the
        // Phase 4 mock queue + perf are NOT restored on launch. That's
        // intentional: queue items + perf telemetry are transient.
        //
        // Re-seed the visible mock content after load so the right panel
        // isn't empty on a fresh launch with a saved state.ron present.
        let state = match persist::load() {
            Some(mut s) => {
                if s.queue.queued.is_empty()
                    && s.queue.running.is_none()
                    && s.queue.history.is_empty()
                {
                    s.queue = crate::state::QueueState::mock();
                }
                if s.perf.gpu_name.is_empty() {
                    s.perf = crate::state::PerfTelemetry::mock();
                }
                log::info!("Loaded AppState from {}", persist::state_path().display());
                s
            }
            None => AppState::default(),
        };
        cc.egui_ctx.set_visuals(state.theme.visuals());
        crate::theme::apply_density(&cc.egui_ctx);
        // Spawn the worker BEFORE returning so the channel exists for the
        // first `update()` call. Cloning the egui Context is cheap (Arc-y
        // internals) and gives the worker a way to request repaints.
        let worker = spawn_worker(cc.egui_ctx.clone());

        // Try to bring up NVML. Failure is non-fatal — perf footer falls
        // back to the mock snapshot already in state.perf. Logged at info
        // level so the absence of a driver is visible in dev runs but
        // doesn't shout in production.
        let perf_backend = match NvmlBackend::new() {
            Ok(b) => {
                log::info!("NVML initialized — live GPU telemetry active");
                Some(b)
            }
            Err(e) => {
                log::info!("NVML unavailable ({e}); using mock perf telemetry");
                None
            }
        };

        // Seed the prev-state snapshot from the loaded state so the first
        // `update()` doesn't see a phantom diff and trigger a no-op save.
        let state_prev_ron = ron::ser::to_string(&state).unwrap_or_default();

        Self {
            state,
            worker,
            last_image: None,
            next_job_id: 1000,
            perf_backend,
            last_perf_poll: Instant::now(),
            save_debouncer: DebouncedSave::new(500),
            state_prev_ron,
        }
    }

    /// Apply a single `WorkerEvent` to local state. Pulled out of `update()`
    /// so the event loop stays compact and so the match is easy to extend in
    /// Phase 5b (real preview frames, save-to-disk hooks, etc.).
    fn apply_event(&mut self, ev: WorkerEvent, ctx: &egui::Context) {
        match ev {
            WorkerEvent::Started { id, job, total_steps } => {
                self.state.total_steps = total_steps;
                self.state.current_step = 0;
                self.state.generating = true;
                // If the running slot didn't get pre-populated by the
                // action-bar (eg. a queued job promoting after a previous
                // job finished), seed it from the **job snapshot** the
                // worker is actually running — NOT from `state.current()`,
                // which reflects whatever the user has been twiddling
                // since the job was submitted (possibly minutes ago for a
                // queued job). Drop the queued mirror entry too so the
                // queue panel doesn't show the same job twice.
                if self.state.queue.running.as_ref().map(|j| j.id) != Some(id) {
                    self.state.queue.queued.retain(|j| j.id != id);
                    self.state.queue.running = Some(QueueJob {
                        id,
                        prompt: job.prompt.clone(),
                        width: job.width,
                        height: job.height,
                        steps: total_steps,
                        sampler: job.sampler.clone(),
                        progress: 0.0,
                        eta_secs: None,
                    });
                } else if let Some(running) = &mut self.state.queue.running {
                    running.steps = total_steps;
                    running.progress = 0.0;
                }
            }
            WorkerEvent::Progress { id, step, total, eta_secs } => {
                self.state.current_step = step;
                self.state.total_steps = total;
                if let Some(running) = &mut self.state.queue.running {
                    if running.id == id {
                        running.progress = step as f32 / total.max(1) as f32;
                        running.eta_secs = Some(eta_secs);
                    }
                }
            }
            WorkerEvent::Preview { .. } => {
                // Phase 5a: mock worker never sends a populated preview. If
                // a future stage emits one we'd upload it as a texture and
                // store it the same way `Done` does. Intentionally a no-op
                // here so we don't have a half-implemented code path.
            }
            WorkerEvent::Done { id, image, prompt } => {
                // Upload the result to the GPU as a TextureHandle. egui's
                // `load_texture` takes a `ColorImage` by value and returns
                // a handle whose Drop frees the GPU texture — storing it in
                // `self.last_image` (replacing any previous handle) is the
                // textbook pattern and means we never leak.
                let tex = ctx.load_texture(
                    format!("flame_result_{id}"),
                    image,
                    egui::TextureOptions::LINEAR,
                );
                self.last_image = Some(tex);

                // Push to history (front = newest first, matching how the
                // history grid renders). thumbnail_path stays None — Phase
                // 5b/6 will own disk-save + thumbnail wiring.
                self.state.queue.history.insert(
                    0,
                    HistoryItem {
                        id,
                        prompt,
                        thumbnail_path: None,
                        timestamp: std::time::SystemTime::now(),
                    },
                );

                // Clear running state. Do NOT auto-promote a queued job
                // here — the worker thread does that itself by popping
                // from its own internal queue and emitting another Started
                // event, which we'll see on the next update tick.
                self.state.queue.running = None;
                self.state.generating = false;
                self.state.current_step = 0;
                self.state.total_steps = 0;
            }
            WorkerEvent::Failed { id, error } => {
                log::warn!("Worker job {id} failed: {error}");
                if self.state.queue.running.as_ref().map(|j| j.id) == Some(id) {
                    self.state.queue.running = None;
                }
                self.state.generating = false;
                self.state.current_step = 0;
                self.state.total_steps = 0;
            }
            WorkerEvent::Idle => {
                // Worker reports its queue is empty. No state change needed
                // — Done/Failed already cleared `generating`. Logged at
                // trace level only so it doesn't spam normal logs.
                log::trace!("Worker idle");
            }
        }
    }
}

impl eframe::App for FlameInferenceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ------ 1. Keyboard shortcuts -----------------------------------
        // Consumed BEFORE painting so widget-level handlers (notably the
        // prompt TextEdit, which would swallow Enter) don't double-fire.
        // `consume_key` on the event filter only takes the event from the
        // queue if the modifier+key match, so non-matching frames pay
        // ~zero cost.
        self.handle_shortcuts(ctx);

        // ------ 2. Drain worker events ----------------------------------
        // Drain ALL pending worker events before painting anything. This
        // lets a burst of Progress events collapse into a single redraw
        // (only the latest progress value matters for the UI), and ensures
        // the rest of the frame sees a coherent state snapshot.
        // `try_iter()` returns immediately when the channel is empty.
        let events: Vec<WorkerEvent> = self.worker.rx.try_iter().collect();
        for ev in events {
            self.apply_event(ev, ctx);
        }

        // ------ 3. NVML poll at the spec'd cadence ----------------------
        // 2 Hz while generating, 1 Hz idle. Matches the perf footer's own
        // `request_repaint_after` cadence so polling fires naturally as
        // the footer requests redraws — no separate timer thread.
        if let Some(backend) = &self.perf_backend {
            let cadence = if self.state.generating {
                Duration::from_millis(500)
            } else {
                Duration::from_millis(1000)
            };
            if self.last_perf_poll.elapsed() >= cadence {
                match backend.poll(0) {
                    Ok(snap) => self.state.perf = snap,
                    Err(e) => log::trace!("NVML poll failed: {e}"),
                }
                self.last_perf_poll = Instant::now();
            }
        }

        // ------ 4. Paint chrome + panels --------------------------------
        // Order matters: TopBottomPanels (title, menu) first; bottom status
        // bar next; then SidePanels and finally the CentralPanel for the canvas.
        crate::chrome::title_bar(ctx, self);
        crate::chrome::menu_bar(ctx, self);
        crate::chrome::status_bar(ctx, self);

        crate::panels::params_panel(ctx, self);
        crate::panels::queue_panel(ctx, self);
        crate::panels::canvas_panel(ctx, self);

        // ------ 4b. Lightbox overlay (if image_zoomed) ------------------
        // Painted LAST so it's on top of everything. Click anywhere or
        // press Escape (handled in handle_shortcuts) to close.
        if self.state.image_zoomed {
            if let Some(tex) = &self.last_image {
                let screen = ctx.screen_rect();
                egui::Area::new(egui::Id::new("lightbox"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(screen.min)
                    .show(ctx, |ui| {
                        let painter = ui.painter();
                        // Backdrop: semi-opaque dark over the whole window
                        painter.rect_filled(screen, 0.0, egui::Color32::from_black_alpha(220));
                        // Aspect-fit the image into a 90% box
                        let pad = 32.0;
                        let avail_w = (screen.width() - 2.0 * pad).max(40.0);
                        let avail_h = (screen.height() - 2.0 * pad).max(40.0);
                        let [tw, th] = tex.size();
                        let aspect = tw as f32 / th.max(1) as f32;
                        let (iw, ih) = if avail_w / aspect <= avail_h {
                            (avail_w, avail_w / aspect)
                        } else {
                            (avail_h * aspect, avail_h)
                        };
                        let img_rect = egui::Rect::from_center_size(
                            screen.center(),
                            egui::vec2(iw, ih),
                        );
                        egui::Image::from_texture(tex).paint_at(ui, img_rect);
                        // Click anywhere closes (alt to Esc)
                        let resp = ui.allocate_rect(screen, egui::Sense::click());
                        if resp.clicked() {
                            self.state.image_zoomed = false;
                        }
                    });
            } else {
                // Defensive: no image to show, drop the flag
                self.state.image_zoomed = false;
            }
        }

        // ------ 5. Detect state change → debounced save -----------------
        // After the UI pass, snapshot the (potentially mutated) state and
        // compare against the previous frame's snapshot. Any difference
        // marks the debouncer dirty; a periodic check then saves once the
        // user has paused interacting for ~500ms.
        //
        // We re-serialize on every frame regardless of whether anything
        // changed. AppState is a few KB; RON serialization is in the tens
        // of microseconds. Cheaper than maintaining a parallel hash that
        // has to chase every f32 field.
        let cur_ron = ron::ser::to_string(&self.state).unwrap_or_default();
        if cur_ron != self.state_prev_ron {
            self.save_debouncer.mark();
            self.state_prev_ron = cur_ron;
        }
        if self.save_debouncer.should_save() {
            match persist::save(&self.state) {
                Ok(()) => self.save_debouncer.flush(),
                Err(e) => log::warn!("Failed to persist state: {e}"),
            }
        }
    }

    /// Called by eframe just before the app exits. Force-flush any
    /// pending state changes so a quick edit + close doesn't lose work.
    /// Errors are logged, not propagated — the app is already shutting
    /// down and there's nothing more we could do about it.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.save_debouncer.has_pending() {
            if let Err(e) = persist::save(&self.state) {
                log::warn!("Failed to persist state on exit: {e}");
            } else {
                log::info!("Flushed pending state on exit");
            }
        }
    }
}

impl FlameInferenceApp {
    /// Process the keyboard shortcuts spec'd in the README "Interactions"
    /// section. Each shortcut consumes the input event so the underlying
    /// widget (eg. the prompt TextEdit) doesn't see it. Order matters:
    /// the more-specific modifier combos (Ctrl+Shift+Enter) are consumed
    /// before the less-specific (Ctrl+Enter), otherwise the bare Ctrl+Enter
    /// branch would fire on a Ctrl+Shift+Enter press.
    ///
    /// AGENT-DEFAULT: text-input keys (R, A, T, 1, 2) only fire when no
    /// TextEdit currently holds focus — otherwise typing a literal "r" in
    /// the prompt would re-roll the seed. egui's `wants_keyboard_input`
    /// at the context level tells us if any widget has focus.
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, Modifiers};

        let typing = ctx.wants_keyboard_input();

        // Ctrl+Shift+Enter — + Queue. Checked before Ctrl+Enter so the
        // shift-bearing variant wins on a combined press.
        if ctx.input_mut(|i| {
            i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Enter)
        }) {
            self.action_queue();
        }

        // Ctrl+Enter — Generate (idempotent if already generating).
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Enter)) {
            if !self.state.generating {
                self.action_generate();
            }
        }

        // Esc — priority: close lightbox zoom > Stop > (fall through to
        // egui's popup dismissal). egui itself uses Esc to dismiss popups,
        // so we only consume when there's something specific to undo.
        if self.state.image_zoomed
            && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
        {
            self.state.image_zoomed = false;
        } else if self.state.generating
            && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
        {
            self.action_stop();
        }

        // Single-letter / digit shortcuts — gate on "no widget has focus"
        // so the user can still type these characters in TextEdits.
        if !typing {
            // R — reroll seed.
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::R)) {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                self.state.seed = rng.gen_range(0u64..=9_999_999_999u64) as i64;
            }
            // A — toggle Advanced.
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::A)) {
                self.state.advanced = !self.state.advanced;
            }
            // 1 — Image tab.
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Num1)) {
                self.state.tab = Mode::Image;
            }
            // 2 — Video tab.
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Num2)) {
                self.state.tab = Mode::Video;
            }
            // T — focus the prompt TextEdit. The id matches the one set
            // in `sections::prompt::show` (`Id::new("prompt_textedit")`).
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::T)) {
                let prompt_id = egui::Id::new("prompt_textedit");
                ctx.memory_mut(|m| m.request_focus(prompt_id));
            }
        }
    }

    /// Shared Generate logic — used by both the action-bar button and the
    /// Ctrl+Enter shortcut. Mirrors `sections::action_bar`'s click path:
    /// mint id → snapshot → pre-populate running slot → send to worker.
    fn action_generate(&mut self) {
        use crate::worker::{GenerateJob, ModelKind, UiMsg};
        let id = self.next_job_id;
        self.next_job_id += 1;
        let cn = self.state.current();
        let job = GenerateJob {
            id,
            model_kind: ModelKind::from_model_string(&cn.model),
            prompt: self.state.prompt.clone(),
            negative: self.state.negative.clone(),
            width: cn.width,
            height: cn.height,
            steps: cn.steps,
            cfg: cn.cfg,
            seed: self.state.seed,
            sampler: cn.sampler.clone(),
            scheduler: cn.scheduler.clone(),
        };
        let (w, h, steps, sampler) = (cn.width, cn.height, cn.steps, cn.sampler.clone());
        self.state.queue.running = Some(QueueJob {
            id,
            prompt: self.state.prompt.clone(),
            width: w,
            height: h,
            steps,
            sampler,
            progress: 0.0,
            eta_secs: None,
        });
        self.state.generating = true;
        self.state.current_step = 0;
        self.state.total_steps = steps;
        let _ = self.worker.tx.send(UiMsg::Generate { job });
    }

    /// Shared +Queue logic for the Ctrl+Shift+Enter shortcut.
    fn action_queue(&mut self) {
        use crate::worker::{GenerateJob, ModelKind, UiMsg};
        let id = self.next_job_id;
        self.next_job_id += 1;
        let cn = self.state.current();
        let job = GenerateJob {
            id,
            model_kind: ModelKind::from_model_string(&cn.model),
            prompt: self.state.prompt.clone(),
            negative: self.state.negative.clone(),
            width: cn.width,
            height: cn.height,
            steps: cn.steps,
            cfg: cn.cfg,
            seed: self.state.seed,
            sampler: cn.sampler.clone(),
            scheduler: cn.scheduler.clone(),
        };
        let (w, h, steps, sampler) = (cn.width, cn.height, cn.steps, cn.sampler.clone());
        self.state.queue.queued.push(QueueJob {
            id,
            prompt: self.state.prompt.clone(),
            width: w,
            height: h,
            steps,
            sampler,
            progress: 0.0,
            eta_secs: None,
        });
        let _ = self.worker.tx.send(UiMsg::Queue { job });
    }

    /// Esc shortcut — symmetric with the action-bar Stop button. We don't
    /// flip `state.generating` locally; the worker confirms via Failed.
    fn action_stop(&mut self) {
        use crate::worker::UiMsg;
        let _ = self.worker.tx.send(UiMsg::Cancel);
    }
}
