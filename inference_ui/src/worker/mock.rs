//! Mock worker implementation.
//!
//! Emits `Started` → `Progress` (one per step at ~80ms each) → `Done` with a
//! synthetic gradient image. Listens for `Cancel` between steps and aborts
//! cleanly with a `Failed { error: "cancelled" }` event. After every event
//! sent, calls `ctx.request_repaint()` so the UI redraws promptly even when
//! the mouse is idle.
//!
//! Phase 5b will replace `run_one`'s body with a real flame-inference call;
//! the surrounding queue-drain loop is intended to stay roughly as-is.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use egui::{Color32, ColorImage};

use super::{
    anima, cascade, chroma, ernie, flux, klein, qwenimage, sd15, sd3, sdxl, sensenova, zimage,
    GenerateJob, ModelKind, UiMsg, WorkerEvent,
};

/// Per-step sleep for the mock generator. 80ms × 28 steps ≈ 2.2s — long
/// enough to see the progress bar advance and click Stop, short enough not
/// to bore us during dev.
const STEP_DURATION: Duration = Duration::from_millis(80);

/// Idle wait between checking the channel when the queue is empty. Long
/// enough that we don't burn CPU spinning, short enough that submitted jobs
/// start "instantly" from the user's POV.
const IDLE_POLL: Duration = Duration::from_secs(1);

pub fn run(ui_rx: Receiver<UiMsg>, ev_tx: Sender<WorkerEvent>, ctx: egui::Context) {
    let mut queue: VecDeque<GenerateJob> = VecDeque::new();
    // Holding buffer for non-Cancel/Shutdown messages received mid-`run_one`.
    // The inner loop can't apply Queue/RemoveQueued/Reorder to `queue` (it
    // doesn't have a reference) so it pushes them here instead, and we
    // process them at the top of the outer loop alongside fresh ui_rx msgs.
    let mut pending: VecDeque<UiMsg> = VecDeque::new();

    // Z-Image state — lazily constructed on first Z-Image job. Holds the
    // CUDA device handle plus three Option<...> caches for the encoder, DiT,
    // and VAE. Resident across iterations so subsequent generations skip
    // the multi-minute model load. See `worker/zimage.rs` for the dance.
    //
    // AGENT-DEFAULT: build this lazily via `Option<ZImageState>` instead of
    // forcing CUDA init at worker spawn. CUDA init can fail (no driver,
    // headless container) and we want the mock path to still work in that
    // environment.
    let mut zimage_state: Option<zimage::ZImageState> = None;

    // FLUX 1 Dev state — same lazy-init pattern. Caches CLIP-L + VAE; T5 +
    // DiT are loaded per-job and dropped to keep peak VRAM under 24 GB.
    let mut flux_state: Option<flux::FluxState> = None;

    // Chroma state — caches VAE only; T5 + DiT loaded per-job.
    let mut chroma_state: Option<chroma::ChromaState> = None;

    // Klein 4B + 9B state — caches VAE + tokenizer only; Qwen3 encoder +
    // DiT loaded per-job and dropped to keep VRAM safe (especially for the
    // 9B variant on a 24 GB card). One state struct serves both variants;
    // `klein::run` dispatches on `job.model_kind` internally.
    let mut klein_state: Option<klein::KleinState> = None;

    // SD 3.5 Medium state — same lazy-init pattern. The struct holds only
    // the CudaDevice handle; encoders/DiT/VAE are all loaded per-job.
    let mut sd3_state: Option<sd3::Sd3State> = None;

    // Qwen-Image state — same lazy-init. Reads cached embeddings per-job;
    // DiT + VAE loaded per-job too.
    let mut qwenimage_state: Option<qwenimage::QwenImageState> = None;

    // ERNIE-Image state — same lazy-init. Mistral encoder loaded per-job
    // and dropped before DiT load (sequential CFG with explicit pool flush).
    let mut ernie_state: Option<ernie::ErnieState> = None;

    // Anima state — same lazy-init. Reads cached context per-job; DiT
    // (3.9 GB) loaded per-job.
    let mut anima_state: Option<anima::AnimaState> = None;

    // SDXL state — same lazy-init. Reads cached dual-encoder embeddings
    // per-job; UNet + VAE loaded per-job.
    let mut sdxl_state: Option<sdxl::SdxlState> = None;

    // SD 1.5 state — same lazy-init. CLIP-L encoded inline per-job; UNet
    // + VAE loaded per-job (small model, fast reload).
    let mut sd15_state: Option<sd15::Sd15State> = None;

    // Cascade state — same lazy-init. CLIP-G encoded inline per-job; three
    // stages (Stage C prior, Stage B decoder, Paella VQ-GAN) all loaded
    // sequentially per-job. Most VRAM-fragmented of all workers.
    let mut cascade_state: Option<cascade::CascadeState> = None;

    // SenseNova-U1 state — same lazy-init pattern. Unlike most workers above,
    // BOTH the tokenizer and the model are kept resident across jobs because
    // the model load (BlockOffloader populating ~32 GB into pinned host RAM)
    // is ~80 s on first call and re-doing it per job would be unusable in a
    // UI. Steady-state GPU footprint is ~11 GB regardless of resident model
    // size — see `worker/sensenova.rs` for the rationale.
    let mut sensenova_state: Option<sensenova::SenseNovaState> = None;

    loop {
        // Drain any messages that the inner loop captured during the last
        // run_one call, then any pending UiMsgs from the channel. Order
        // matters: pending-from-inner first preserves user intent ordering
        // (a +Queue clicked before a fresh Generate during the SAME job
        // should be handled before later messages).
        let drain: Vec<UiMsg> = pending.drain(..).chain(ui_rx.try_iter()).collect();
        for msg in drain {
            match msg {
                UiMsg::Generate { job } => queue.push_front(job), // immediate
                UiMsg::Queue { job } => queue.push_back(job),
                UiMsg::Cancel => queue.clear(),
                UiMsg::RemoveQueued { id } => queue.retain(|j| j.id != id),
                UiMsg::Reorder { from, to } => {
                    if from < queue.len() && to < queue.len() && from != to {
                        if let Some(job) = queue.remove(from) {
                            // After remove, indices >= `from` shift left by 1.
                            // Clamp `to` so we don't overshoot when caller
                            // expressed the destination in pre-remove indices.
                            let dest = to.min(queue.len());
                            queue.insert(dest, job);
                        }
                    }
                }
                UiMsg::Shutdown => return,
            }
        }

        if let Some(job) = queue.pop_front() {
            // Phase 5b: dispatch on `model_kind`. Mock stays as fallback for
            // anything we haven't wired up yet (FLUX, Klein, Chroma, ...).
            // Z-Image variants share the same backend module + state cache
            // (variant change inside zimage::run drops the DiT but keeps
            // encoder + VAE resident).
            match job.model_kind {
                ModelKind::Mock => run_one(&job, &ui_rx, &ev_tx, &ctx, &mut pending),
                ModelKind::ZImageBase | ModelKind::ZImageTurbo => {
                    // Lazily init Z-Image state on first call. If CUDA init
                    // fails (no driver, etc.), emit Failed and continue —
                    // the worker thread itself stays alive so subsequent
                    // Mock jobs still work.
                    if zimage_state.is_none() {
                        match zimage::ZImageState::new() {
                            Ok(s) => zimage_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("Z-Image CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = zimage_state.as_mut().unwrap();
                    zimage::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::FluxDev => {
                    if flux_state.is_none() {
                        match flux::FluxState::new() {
                            Ok(s) => flux_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("FLUX CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = flux_state.as_mut().unwrap();
                    flux::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Chroma => {
                    if chroma_state.is_none() {
                        match chroma::ChromaState::new() {
                            Ok(s) => chroma_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("Chroma CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = chroma_state.as_mut().unwrap();
                    chroma::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Klein4B | ModelKind::Klein9B => {
                    if klein_state.is_none() {
                        match klein::KleinState::new() {
                            Ok(s) => klein_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("Klein CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = klein_state.as_mut().unwrap();
                    klein::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Sd35 => {
                    if sd3_state.is_none() {
                        match sd3::Sd3State::new() {
                            Ok(s) => sd3_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("SD3.5 CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = sd3_state.as_mut().unwrap();
                    sd3::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::QwenImage => {
                    if qwenimage_state.is_none() {
                        match qwenimage::QwenImageState::new() {
                            Ok(s) => qwenimage_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("Qwen-Image CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = qwenimage_state.as_mut().unwrap();
                    qwenimage::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::ErnieImage => {
                    if ernie_state.is_none() {
                        match ernie::ErnieState::new() {
                            Ok(s) => ernie_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("ERNIE-Image CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = ernie_state.as_mut().unwrap();
                    ernie::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Anima => {
                    if anima_state.is_none() {
                        match anima::AnimaState::new() {
                            Ok(s) => anima_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("Anima CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = anima_state.as_mut().unwrap();
                    anima::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Sdxl => {
                    if sdxl_state.is_none() {
                        match sdxl::SdxlState::new() {
                            Ok(s) => sdxl_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("SDXL CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = sdxl_state.as_mut().unwrap();
                    sdxl::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Sd15 => {
                    if sd15_state.is_none() {
                        match sd15::Sd15State::new() {
                            Ok(s) => sd15_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("SD 1.5 CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = sd15_state.as_mut().unwrap();
                    sd15::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::Cascade => {
                    if cascade_state.is_none() {
                        match cascade::CascadeState::new() {
                            Ok(s) => cascade_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("Cascade CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = cascade_state.as_mut().unwrap();
                    cascade::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
                ModelKind::SenseNovaU1 => {
                    if sensenova_state.is_none() {
                        match sensenova::SenseNovaState::new() {
                            Ok(s) => sensenova_state = Some(s),
                            Err(e) => {
                                let _ = ev_tx.send(WorkerEvent::Failed {
                                    id: job.id,
                                    error: format!("SenseNova-U1 CUDA init failed: {e}"),
                                });
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }
                    let state = sensenova_state.as_mut().unwrap();
                    sensenova::run(&job, state, &ui_rx, &ev_tx, &ctx, &mut pending);
                }
            }
            // After a job ends (done OR cancelled), continue the outer loop
            // so any queued jobs run next. The Idle event is emitted only
            // when the queue actually drains in the else-branch below.
        } else {
            // Idle — emit Idle once and block-wait for the next message.
            // We don't spam Idle every frame; the UI doesn't currently need
            // it but the protocol guarantees one per drain.
            let _ = ev_tx.send(WorkerEvent::Idle);
            ctx.request_repaint();
            match ui_rx.recv_timeout(IDLE_POLL) {
                Ok(msg) => {
                    // Re-handle the message. The `while try_recv` at the top
                    // of the next loop iteration will catch any backlog.
                    match msg {
                        UiMsg::Generate { job } => queue.push_front(job),
                        UiMsg::Queue { job } => queue.push_back(job),
                        UiMsg::Cancel => {} // nothing to cancel when idle
                        UiMsg::RemoveQueued { id } => queue.retain(|j| j.id != id),
                        UiMsg::Reorder { .. } => {} // nothing to reorder
                        UiMsg::Shutdown => return,
                    }
                }
                Err(RecvTimeoutError::Timeout) => {} // loop again, recheck
                Err(RecvTimeoutError::Disconnected) => return, // UI dropped
            }
        }
    }
}

/// Run one job to completion (or cancellation). Sends `Started`, N×`Progress`,
/// then either `Done` or `Failed`.
///
/// `pending` is an out-param: any non-Cancel/non-Shutdown messages received
/// during the cancel-check loop get pushed here so the outer `run` loop can
/// apply them to its job queue after we return. This is what makes +Queue
/// clicks during an in-flight job actually take effect (instead of being
/// silently dropped on the floor).
fn run_one(
    job: &GenerateJob,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: job.steps,
    });
    ctx.request_repaint();

    // Guard against zero-step jobs — the README mock data and per-mode
    // defaults all use steps>=1, but defensive: skip the loop, jump to Done.
    let steps = job.steps.max(1);
    let start = Instant::now();
    for step in 1..=steps {
        // Cancellation check between steps. Drain *all* pending messages so
        // we don't pretend to ignore a Cancel just because there's also a
        // queued job behind it. Non-cancel messages (Queue/RemoveQueued/
        // Reorder) get forwarded to `pending` so the outer loop sees them
        // after run_one returns — this is what fixes the silent-drop bug
        // when the user clicks +Queue while a job is running.
        while let Ok(msg) = ui_rx.try_recv() {
            match msg {
                UiMsg::Cancel | UiMsg::Shutdown => {
                    let _ = ev_tx.send(WorkerEvent::Failed {
                        id: job.id,
                        error: "cancelled".into(),
                    });
                    ctx.request_repaint();
                    // Re-route the Shutdown signal so the outer loop also
                    // exits after we return. Cancel doesn't need this — its
                    // job-clearing semantics already happened above (queue
                    // was empty for THIS job; further queued items will
                    // also be cleared by the outer loop's Cancel handling
                    // if any other Cancel arrives).
                    if matches!(msg, UiMsg::Shutdown) {
                        pending.push_back(UiMsg::Shutdown);
                    }
                    return;
                }
                other => pending.push_back(other),
            }
        }

        std::thread::sleep(STEP_DURATION);

        let elapsed = start.elapsed().as_secs_f32();
        let per_step = elapsed / step as f32;
        let eta_secs = ((steps - step) as f32 * per_step).round().max(0.0) as u32;
        let _ = ev_tx.send(WorkerEvent::Progress {
            id: job.id,
            step,
            total: steps,
            eta_secs,
        });
        ctx.request_repaint();
    }

    // Synthesize a "result" image. Cheap deterministic gradient keyed off
    // seed so re-running with the same seed produces the same picture (the
    // way real diffusion behaves).
    let image = generate_mock_image(job.width as usize, job.height as usize, job.seed);
    let _ = ev_tx.send(WorkerEvent::Done {
        id: job.id,
        image,
        prompt: job.prompt.clone(),
    });
    ctx.request_repaint();
}

/// Synthetic seed-keyed gradient. Phase 5b replaces with a real VAE decode
/// of the final latent.
///
/// AGENT-DEFAULT: cap the synthesized image at 512×512 regardless of the
/// requested W×H. A 1024×1024 ColorImage is 4 MiB and we'd be re-creating it
/// every Done event during dev — wasteful for a placeholder. Egui's image()
/// painter will scale it up to fit the preview rect anyway. The job's real
/// w/h are already echoed in the canvas badge separately.
fn generate_mock_image(w: usize, h: usize, seed: i64) -> ColorImage {
    const MAX_DIM: usize = 512;
    let scale = (w.max(h)).max(1);
    let (out_w, out_h) = if scale > MAX_DIM {
        let f = MAX_DIM as f32 / scale as f32;
        (((w as f32 * f) as usize).max(8), ((h as f32 * f) as usize).max(8))
    } else {
        (w.max(8), h.max(8))
    };

    let mut pixels = Vec::with_capacity(out_w * out_h);
    let s = (seed.unsigned_abs() % 360) as f32 / 360.0;
    for y in 0..out_h {
        for x in 0..out_w {
            let u = x as f32 / out_w as f32;
            let v = y as f32 / out_h as f32;
            let r = (u * 0.7 + s).fract();
            let g = (v * 0.5 + s * 0.3).fract();
            let b = ((u + v) * 0.4 + s * 0.7).fract();
            pixels.push(Color32::from_rgb(
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
            ));
        }
    }
    ColorImage {
        size: [out_w, out_h],
        pixels,
    }
}
