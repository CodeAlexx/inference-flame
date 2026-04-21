# SKEPTIC review — Phase 5a (worker + mock + Generate wiring)

Scope: `src/worker/{mod,mock}.rs`, `src/app.rs`,
`src/sections/{action_bar,canvas,queue}.rs`, `src/panels.rs`.
Focus: concurrency + protocol soundness.

---

## P0

None. The protocol works for the common single-job click-Generate-watch-Done
path, which is what Phase 5a's brief asks to ship. Bugs below are real but
either user-uncommon, mock-only, or future-protocol concerns.

---

## P1

### P1-1. `+ Queue` mid-generation is silently dropped by the worker

`mock.rs:114-127` — inside `run_one`, the cancel-check `while try_recv` loop
discards any non-`Cancel`/`Shutdown` message ("drop on the floor for now").
But `action_bar.rs:104-141` ships `+ Queue` as a fully wired button: it
pushes to `state.queue.queued` (UI mirror) AND sends `UiMsg::Queue { job }`
to the worker.

User flow that breaks:
1. Click Generate (jobA, ~2.2s mock duration)
2. While jobA is running, click `+ Queue` → jobB
3. UI shows jobB in `state.queue.queued` (visible in right panel)
4. Worker finishes jobA → Done → fires Idle → never sees jobB
5. UI shows jobB queued forever; worker is asleep in `recv_timeout`

The builder's mock.rs comment says "Phase 5a brief explicitly says don't
bother", but the brief lists `+ Queue` as in-scope (it's not in the "out of
scope" list). The drop-on-the-floor design is a real Phase 5a UX bug:
clicking +Queue while a job is running silently fails.

**Minimum fix:** in `run_one`'s cancel-check, route `Queue { job }` and
`RemoveQueued { id }` to a shared mutable queue (or send them back through a
side channel) instead of discarding. The simplest patch is to make `queue`
in `run` a `Arc<Mutex<VecDeque>>` shared with `run_one`, OR to pass the
inner-loop's drained non-cancel msgs back to the outer loop.

### P1-2. Worker runs Started for a queued job; UI seeds running slot from *live state*, not the job's snapshot

`app.rs:68-83` — when a Started event arrives for an `id` that doesn't match
the current `running.id`, the handler synthesizes a `QueueJob` from
`state.current()` + `state.prompt`. That's the live UI state at *Started
time*, not the params the worker actually has baked into the job it's
running.

User flow that drifts:
1. Submit jobA (steps=20) via Generate
2. Change steps slider to 50; click +Queue → jobB (steps=50)
3. Change steps slider to 100
4. jobA finishes; Started{jobB.id} arrives
5. UI seeds `running` with steps=100 (live state) — but worker is running
   jobB with steps=50 (the snapshot taken at +Queue click)
6. Progress events use `total = job.steps = 50` so the bar says `step
   N / 50`. The QueueJob slot says `steps: 100`. The two disagree until
   Progress overwrites `running.steps` (which it doesn't — see Progress
   handler in app.rs:85-94, which only updates `progress` and `eta_secs`,
   never `steps`).

**Minimum fix:** include the job's `(prompt, w, h, steps, sampler)` in the
`Started` payload and seed `running` from the event, not from live state.
Or: maintain a UI-side `HashMap<u64, QueueJob>` for in-flight/queued ids and
look up on Started. Either way, the protocol design "Started = id +
total_steps" loses critical context.

(Today this only fires after P1-1 is fixed and a queued job actually
promotes — Phase 5a never gets there because nothing queues. But the
protocol is locked in for Phase 5b.)

---

## P2

### P2-1. Idle event triggers a UI repaint every 1s while truly idle

`mock.rs:64-86` — when the worker queue is empty, it sends `WorkerEvent::Idle`,
calls `ctx.request_repaint()`, then `recv_timeout(1s)`. On every `Timeout`
return, the outer loop iterates: pops nothing → else branch → another Idle
+ another repaint. So the UI redraws once per second indefinitely while
idle. Egui will repaint the whole frame for nothing.

Fix: only send Idle on the *transition* into idle (track a `was_idle: bool`
in `run`), not every poll cycle. Or drop the `request_repaint()` call after
the Idle send — Idle has no UI consequence so a repaint is wasteful.

### P2-2. `next_job_id = 1000` to avoid mock-seed-id collision

`app.rs:34, 50`; `queue.rs:265-273`. `QueueState::mock()` populates running
with id=1 and queued with ids 2..=4. The cancel-on-running-row path needs
to clear the local slot when worker won't (because worker doesn't know
about the mock entry); it does this by checking `id < 1000`.

This works because new ids are u64-monotonic from 1000. But it's fragile:
- The threshold is implicit (no const, no doc on the mock side)
- If anyone adds a 5th mock entry with id=1001 it silently breaks
- A test that exercises `QueueState::mock()` then submits 1 job will
  produce real-id 1000 (still > all mock ids — fine for now)

Cleaner: tag mock entries with an explicit `is_mock: bool` field, OR start
real ids at e.g. `u64::MAX / 2` and assert mocks below that, OR just don't
seed the queue panel with mock data once Phase 5a ships real submissions.
**Flag for Phase 6 cleanup.** Builder already noted this.

### P2-3. Worker thread is never joined on app exit

No `Drop` impl on `FlameInferenceApp` and no `eframe` `on_close_requested`
hook. `WorkerHandle` drop closes the `Sender<UiMsg>` and `Receiver<WorkerEvent>`,
which causes the worker's `recv_timeout` to return `Disconnected` and
return. But: if the worker is mid-`run_one` (sleeping 80ms or generating
the gradient image), the process may exit before the worker thread
finishes. No state corruption (worker holds nothing the OS can leak), but
the spawn `expect("failed to spawn inference worker thread")` and the
detached thread mean an unclean shutdown.

Severity: low. Mock generates images in <2s and the OS reaps the thread
anyway. **Phase 5b** (real flame-inference, possibly holding GPU tensors)
needs an explicit graceful-shutdown path: send `UiMsg::Shutdown`, then
join. The `Shutdown` variant is already in the protocol — just no UI
sender.

### P2-4. `Image::from_texture(tex).paint_at(ui, preview_rect)` paints into the parent Ui, not the painter at the preview rect

`canvas.rs:195` — uses `egui::Image::from_texture(tex).paint_at(ui, rect)`.
This works (it queues a textured rect into the parent Ui's painter) but the
painter being used to draw the checker bg is `ui.painter_at(rect)` (line
76) which is a child painter clipped to the canvas surface. The image
paint goes through the parent `ui` — different clip rect. In practice
preview_rect is fully inside the canvas rect so no visible difference, but
if the canvas were ever clipped tighter (e.g. inside a scroll area), the
image would draw outside. Cosmetic / future-proofing — not a Phase 5a bug.

### P2-5. `state.current_step / state.total_steps` AND `state.queue.running.progress` carry overlapping data

The toolbar's `step N/M` readout (canvas.rs:287) reads
`state.current_step/total_steps`. The running row's progress bar
(queue.rs:327) reads `running.progress`. Both come from the same Progress
event but are stored separately. If a future event handler updates one and
forgets the other, the two readouts disagree. Currently both are updated
in lock-step in `apply_event::Progress`. Minor — flag as a normalization
opportunity.

---

## Concurrency / protocol audit

| Concern | Status | Notes |
|---|---|---|
| Unbounded channel growth | **Acceptable for Phase 5a** | Mock worker emits ≤30 events per ~2s job; UI drains every frame (60Hz). Phase 5b real preview frames could change this — flag for Phase 5b: bound the preview channel or coalesce. |
| Worker shutdown on app exit | **Works but not graceful** | See P2-3. Channel-disconnect-as-shutdown is the textbook crossbeam pattern; just no `join()`. |
| `try_iter().collect()` drains all events per frame | **Verified correct** | No N-cap. Burst of Progress events collapse into single frame's apply loop. |
| Done-vs-Cancel race | **Verified safe** | Done arrives → app.rs::Done clears `running` and `generating=false`. Subsequent Cancel sent by Stop click is a no-op (worker idle). Failed-after-Done can't happen (run_one returns after Done; outer loop doesn't re-emit). |
| Cancel-during-final-step | **Late by ≤80ms** | Mock checks Cancel only between steps. If clicked during the last step's sleep, Done wins. UI shows result. Acceptable per builder. |
| Cancel-then-Generate-fast | **Guarded by button swap** | Generate button is "Stop" while `state.generating == true`. action_bar pre-sets `state.generating = true` synchronously on Generate, so a second Generate click within the same frame sends Cancel, not Generate. No way for the user to send two back-to-back `Generate` ids without first seeing Done/Failed. |
| Texture upload on UI thread | **Verified correct** | `ctx.load_texture` runs in `apply_event` inside `update()`. Worker only constructs `ColorImage` (CPU pixels) and ships them. |
| Channel `send().unwrap()` panics | **None** | All sends use `let _ = ...send(...)`. All receives use `try_recv` / `recv_timeout`. Safe on disconnect. |
| Match exhaustiveness | **Verified** | Both UiMsg matches in mock.rs are exhaustive (no `_` arm). Adding a new variant forces a compile error. |
| Borrow splits in action_bar / panels | **Verified clean** | `worker_tx = app.worker.tx.clone()` before `CentralPanel::show`; `next_job_id` taken as `&mut u64`; param snapshots taken in a fresh scope before mutating `state.queue.running`. No re-borrow surprises. |
| State sync UI mirror ↔ worker queue | **Drifts under +Queue** | See P1-1. Single-Generate path stays in sync; +Queue mid-job desyncs. |
| Worker holds egui::Context | **Verified safe** | Cloned into worker; egui::Context is Arc-based and `request_repaint()` is thread-safe. |

---

## AGENT-DEFAULT assessment

Builder claimed 8 AGENT-DEFAULT decisions. Three are explicitly tagged in
new Phase 5a code; the remaining five are implicit design choices. Listing
all I could identify:

1. **Generate/Queue carry full GenerateJob, not opaque ids** (mod.rs:26).
   Sound. UI already has the params; sending them avoids a worker→UI
   round-trip. Cost: P1-2 above (the protocol then assumes the UI keeps a
   matching mirror, which it doesn't).

2. **Mock image capped at 512×512** (mock.rs:158). Sound. Saves memory for
   a placeholder; canvas scales to fit anyway.

3. **Scan-line + gradient overrides last_image during generation**
   (canvas.rs:134). Sound UX choice. In-flight affordance > stale-result
   affordance.

4. **crossbeam-channel over std::sync::mpsc** (mod.rs:11). Sound. Clone-
   able senders + native `recv_timeout`. Minor extra dep.

5. **Unbounded channels** (mod.rs:125-126). Acceptable for Phase 5a (small
   event volume). **Should revisit for Phase 5b** if preview frames are
   per-step. A bounded channel with `try_send` (drop oldest preview) is
   the textbook fix.

6. **`next_job_id` starts at 1000** (app.rs:50). Hack. See P2-2. Flag for
   Phase 6.

7. **Worker thread is detached, not joined** (mod.rs:128-130). Acceptable
   for Phase 5a mock; needs revisiting for Phase 5b GPU work. See P2-3.

8. **`state.queue.running` pre-populated synchronously by action_bar**
   (action_bar.rs:82-93). The Started handler then conditionally re-seeds
   only when ids differ. Sound for Generate, but feeds into P1-2 for
   queued-job promotion. Documented; would be cleaner if action_bar
   emitted the same `QueueJob` snapshot the worker eventually echoes via
   Started.

Bonus (not flagged by builder but worth noting):
- **Idle event semantics**: builder ships "Idle on every recv_timeout cycle"
  rather than "Idle on transition". See P2-1.

---

## Scope creep

None observed. Worker/mock/wiring is all Phase 5a-bracketed. The
`Reorder` / `Shutdown` / `Preview` variants are protocol slots without UI
senders — `#[allow(dead_code)]` is honest. No History thumbnails, no
right-click wiring, no save-to-disk, no keyboard shortcuts, no persist.

The `+ Queue` button is wired (in scope) but with the silent-drop bug
above — not creep, just incomplete.

---

## Unverified

- `egui::Image::from_texture(tex).paint_at()` aspect-ratio behavior across
  egui versions. Builder's claim is "preview_rect is sized from configured
  W:H, matching the texture's source aspect ratio because we generated it
  at exactly that ratio in mock.rs". `mock::generate_mock_image` does
  preserve aspect when capping to 512 max-dim. Cross-check: if
  `width=2048, height=1024`, `scale=2048`, `f=0.25`, out=`512×256`.
  preview_rect uses `aspect = 2048/1024 = 2`, correct. ✅ Logically sound;
  not visually verified.

- The `time = ctx.input(|i| i.time)` continuous-animation scan line:
  haven't run the binary, can't confirm 60Hz repaint actually happens (the
  `ctx.request_repaint_after(Duration::from_millis(16))` says yes, but
  egui has been known to coalesce).

- Whether the `current_step / total_steps` in the toolbar (canvas.rs:287)
  visibly twitches between `0/0` (post-Done) and `step N / M`
  (post-Started) when a queued job promotes. With P1-1 unfixed this
  doesn't fire in Phase 5a, so untestable today.

- `crossbeam-channel` is in `Cargo.toml` (assumed; the builder report says
  cargo build passed). Not re-verified.
