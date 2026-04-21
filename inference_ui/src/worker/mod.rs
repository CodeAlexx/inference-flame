//! Inference worker thread + UI message channels.
//!
//! Phase 5a wires the **mock** generator: clicking Generate routes a
//! `GenerateJob` over `crossbeam-channel` to a background thread, which then
//! emits `Started` → `Progress` × N → `Done` events back to the UI. The UI
//! side drains events at the top of every `App::update` frame and updates
//! `AppState` accordingly. Phase 5b replaces `mock::run` with a real
//! flame-inference call but the channel protocol is meant to be stable from
//! here on out — additive changes only.
//!
//! Why crossbeam-channel and not `std::sync::mpsc`? Two reasons:
//!   1) Both ends need `Sender`/`Receiver` to be `Clone`-able so future
//!      multi-producer scenarios (eg. dnd file drop also enqueueing jobs)
//!      Just Work.
//!   2) crossbeam supports `recv_timeout` natively without the platform
//!      quirks `std::sync::mpsc` has on some Linux distros — we use it in
//!      the worker idle-loop to cleanly wake on incoming messages.

use crossbeam_channel::{unbounded, Receiver, Sender};
use egui::ColorImage;

pub mod anima;
pub mod cascade;
pub mod chroma;
pub mod ernie;
pub mod flux;
pub mod klein;
pub mod mock;
pub mod paths;
pub mod qwenimage;
pub mod sd15;
pub mod sd3;
pub mod sdxl;
pub mod zimage;

/// Which model the worker should run for a given job. The mock generator is
/// the catch-all fallback so unmapped model strings still produce a visible
/// result instead of a hard error — Phase 5b only wires Z-Image, the rest
/// stay on Mock until later phases attach them.
//
// `Clone, Copy, Eq` are needed because `app::action_generate` snapshots the
// job into `state.queue.running` (which is `Clone`) and the worker's outer
// loop matches on it.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelKind {
    /// Synthetic gradient generator. No GPU. Used as a placeholder for any
    /// model the real backend hasn't been wired to yet.
    Mock,
    /// Z-Image NextDiT base — 28 steps default, real CFG via Qwen3 cond+uncond.
    ZImageBase,
    /// Z-Image NextDiT turbo — 8 steps default, no CFG (single-pass cond only).
    ZImageTurbo,
    /// FLUX 1 Dev — 20 steps default, distilled guidance (single-pass with
    /// guidance scalar; the user-facing CFG slider is reinterpreted as that
    /// scalar, no real classifier-free guidance and `negative` is ignored).
    FluxDev,
    /// Chroma — 40 steps default, real two-pass CFG (T5-only, no CLIP).
    Chroma,
    /// Klein 4B (Flux 2 base) — 50 steps default, real two-pass CFG, Qwen3
    /// 4B encoder. Smaller of the two Klein variants — fits resident on
    /// 24 GB.
    Klein4B,
    /// Klein 9B (Flux 2 base) — 50 steps default, real two-pass CFG, Qwen3
    /// 8B encoder. Tries resident DiT first, falls back to BlockOffloader
    /// on OOM.
    Klein9B,
    /// SD 3.5 Medium — 28 steps default, real two-pass CFG, triple text
    /// encoder (CLIP-L + CLIP-G + T5-XXL), 16-ch SD3 VAE.
    Sd35,
    /// Qwen-Image-2512 — 50 steps default, real two-pass CFG with Qwen norm
    /// rescale, 60-layer DiT, 3D VAE. Reads pre-computed Qwen2.5-VL
    /// embeddings from a hardcoded safetensors file.
    QwenImage,
    /// ERNIE-Image — 50 steps default, sequential CFG (per-pass pool flush),
    /// Mistral-3 3B encoder, Klein VAE.
    ErnieImage,
    /// Anima (Cosmos Predict2) — 30 steps default, real two-pass CFG.
    /// Reads pre-computed Anima context tensors from a hardcoded
    /// safetensors file (post-adapter Qwen3 0.6B).
    Anima,
    /// SDXL Base 1.0 — 30 steps default, real two-pass CFG, VE eps-prediction
    /// (scaled-linear β schedule, NOT flow). Reads pre-computed dual-encoder
    /// (CLIP-L + CLIP-G) embeddings from a hardcoded safetensors file.
    /// 4-channel SDXL VAE (scale=0.13025).
    Sdxl,
    /// SD 1.5 — 30 steps default 512×512, real two-pass CFG, VE eps-prediction
    /// (same scaled-linear β schedule as SDXL). CLIP-L encoded inline (no
    /// cache file). 4-channel SD VAE (scale=0.18215). Handles the legacy
    /// pre-0.14-diffusers attention key naming in the VAE safetensors.
    Sd15,
    /// Stable Cascade (Würstchen v3) — 20+10 step default, real two-pass
    /// CFG on both stages. CLIP-ViT-bigG-14 text encoder (inline). Three
    /// sequentially-loaded stages: Stage C prior → Stage B decoder →
    /// Paella VQ-GAN. DDIM-style eps step per ddpm_wuerstchen.
    Cascade,
}

impl ModelKind {
    /// Map the user-facing model filename string (from the Model section's
    /// Base ComboBox) to the internal kind. Anything we don't recognize
    /// falls back to Mock so the UI still produces a placeholder image.
    ///
    /// Order matters: more-specific prefixes first (z-image-turbo before
    /// z-image-base). FLUX matches "flux" anywhere in the name — the only
    /// FLUX entry in IMAGE_MODELS today is `flux1-dev.safetensors`, but if
    /// later phases add Schnell or other distilled variants they'd want the
    /// same dispatch (Schnell uses the same DiT shape; only the timestep
    /// schedule differs and that lives in our worker, not the brief). If a
    /// future variant needs to differ, split the match here.
    pub fn from_model_string(s: &str) -> Self {
        // Strip trailing `.gguf` / `.safetensors` so the match arms below
        // don't need to enumerate both variants of every filename. The
        // downstream workers dispatch on the suffix themselves when deciding
        // loader path (GGUF dequant vs safetensors mmap).
        let raw_lower = s.to_ascii_lowercase();
        let trimmed = raw_lower
            .strip_suffix(".gguf")
            .or_else(|| raw_lower.strip_suffix(".safetensors"))
            .unwrap_or(&raw_lower);
        let lower = trimmed;
        if lower.contains("z-image-turbo") || lower.contains("zimage-turbo") {
            Self::ZImageTurbo
        } else if lower.contains("z-image-base") || lower.contains("zimage-base") {
            Self::ZImageBase
        } else if lower.contains("chroma") {
            Self::Chroma
        // Klein matching: 9b before 4b (substring "klein9b" contains "klein"
        // but NOT "4b", so a naive single-arm "klein" → 4B would route 9B
        // wrong). Match the explicit size token first; bare "klein" with no
        // size hint defaults to 4B (smaller, fits comfortably).
        } else if lower.contains("klein") && (lower.contains("9b") || lower.contains("9-b"))
        {
            Self::Klein9B
        } else if lower.contains("klein") {
            // Bare "klein.safetensors" or "klein-4b.safetensors" → 4B.
            // AGENT-DEFAULT: defaulting bare "klein" to 4B rather than
            // erroring keeps the Mode→Base ComboBox tolerant of placeholder
            // names. If a future user adds a non-base klein variant (edit,
            // inpaint) they'll need an explicit token to route correctly.
            Self::Klein4B
        } else if lower.contains("flux") {
            Self::FluxDev
        // SD 3.5 — match "sd3" or "sd-3.5". The IMAGE_MODELS list uses
        // "sd3.5-medium.safetensors"; older "sd3.5-large.safetensors" entries
        // also dispatch here even though we only wire the Medium pipeline
        // (paths point at stablediffusion35_medium). AGENT-DEFAULT: keep one
        // arm and fail at load-time if the wrong file is present.
        //
        // Order note: this MUST come before the SDXL arm, because
        // `sd3.5-medium.safetensors` contains the substring "sd3" but also
        // doesn't contain "xl". SDXL match needs "xl"-anchored tokens.
        } else if lower.contains("sd3") || lower.contains("sd-3.5") {
            Self::Sd35
        } else if lower.contains("qwen") {
            Self::QwenImage
        } else if lower.contains("ernie") {
            Self::ErnieImage
        } else if lower.contains("anima") {
            Self::Anima
        // SDXL — match "sdxl" or "sd-xl" or "sd_xl". The IMAGE_MODELS list
        // uses e.g. "sdxl-base-1.0.safetensors" and "sdxl-turbo.safetensors";
        // both route here. AGENT-DEFAULT: Turbo's 1-step schedule isn't
        // wired; we still use the 30-step default and CFG=7.5, which will
        // over-steer a Turbo checkpoint. If a user picks Turbo, the result
        // won't match upstream Turbo quality. Tracked as future work.
        } else if lower.contains("sdxl") || lower.contains("sd-xl") || lower.contains("sd_xl") {
            Self::Sdxl
        // Stable Cascade / Würstchen — match either.
        } else if lower.contains("cascade") || lower.contains("wurstchen") || lower.contains("würstchen") {
            Self::Cascade
        // SD 1.5 — match "sd15", "sd-1.5", or "sdv1". Placed LAST among the
        // SD-family arms because "sd15" doesn't collide with anything else
        // we match, but "sd-1.5" could theoretically get confused with a
        // hypothetical "sd3-1.5" variant — SD3 is matched above.
        } else if lower.contains("sd15")
            || lower.contains("sd-1.5")
            || lower.contains("sd_1.5")
            || lower.contains("sdv1")
        {
            Self::Sd15
        } else {
            Self::Mock
        }
    }
}

/// Messages the UI thread sends to the worker.
///
/// AGENT-DEFAULT: `Generate` and `Queue` carry the entire `GenerateJob`
/// rather than an opaque id. The job is built on the UI side from the live
/// `AppState` snapshot at click time, so the UI never has to ask the worker
/// "what params did you use?" — it already knows. This also keeps the worker
/// thread free of any reference to UI state.
//
// `Reorder` and `Shutdown` are reserved for Phase 5b/6 wiring (drag-reorder
// in the queue panel + explicit graceful shutdown). The handler match in
// the worker already dispatches them; only the UI senders are missing.
#[allow(dead_code)]
#[derive(Clone)]
pub enum UiMsg {
    /// Submit a job for immediate generation. If the worker is currently
    /// running another job, this jumps to the front of the queue and runs
    /// next once the current job finishes (Phase 5a behavior matches the
    /// brief: `push_front` rather than preempt).
    Generate { job: GenerateJob },
    /// Append a job to the queue (don't run yet).
    Queue { job: GenerateJob },
    /// Cancel the running job and any queued ones.
    Cancel,
    /// Remove a specific queued job.
    RemoveQueued { id: u64 },
    /// Reorder queued jobs. Phase 5a: handler is wired but no UI sender
    /// emits this yet (drag-reorder is Phase 6).
    Reorder { from: usize, to: usize },
    /// Shut down the worker (called on app exit). Phase 5a doesn't currently
    /// fire this — eframe drops the worker handle which closes the channel,
    /// causing the worker's `recv_timeout` to return `Disconnected` and
    /// terminate. Reserved for explicit graceful shutdown later.
    Shutdown,
}

/// Events the worker emits back to the UI.
//
// `Preview` is reserved for Phase 5b — the mock generator never emits one,
// but the variant + UI handler are wired so adding real preview frames is
// purely additive.
#[allow(dead_code)]
#[derive(Clone)]
pub enum WorkerEvent {
    /// Job started; total_steps known. UI uses this to seed the progress
    /// bar baseline before the first `Progress` event arrives.
    ///
    /// Carries the full `GenerateJob` snapshot so the UI can seed the
    /// `running` slot from the params the worker is actually using, not
    /// from live `AppState` (which may have drifted since the job was
    /// submitted — e.g. a queued job whose Resolution slider was changed
    /// after +Queue was clicked).
    Started { id: u64, job: GenerateJob, total_steps: u32 },
    /// Per-step progress.
    Progress { id: u64, step: u32, total: u32, eta_secs: u32 },
    /// Optional latent preview frame. Phase 5a always sends `None` (the mock
    /// generator has nothing meaningful to preview mid-flight). Phase 5b
    /// will populate this with periodic VAE decodes of the current latent.
    Preview { id: u64, image: Option<ColorImage> },
    /// Job finished successfully.
    Done { id: u64, image: ColorImage, prompt: String },
    /// Job failed (or was cancelled — Phase 5a uses error="cancelled").
    Failed { id: u64, error: String },
    /// Worker idle, ready for next job. Used by the UI to know when it can
    /// lower the spinner; not currently consumed in Phase 5a (we lower the
    /// spinner on `Done`/`Failed` directly), but kept in the protocol so
    /// future "queue-drained" UI signals can latch on without protocol
    /// changes.
    Idle,
}

/// Snapshot of the params needed to run a single generation. Built on the
/// UI side at click time from `AppState::current()` + shared fields. The
/// worker side never reaches back into `AppState`.
//
// `negative`, `cfg`, `sampler`, `scheduler` are part of the protocol the
// UI captures + sends, but the Phase 5a mock generator only consumes
// `width`/`height`/`steps`/`seed` (gradient + step count). Phase 5b's real
// flame-inference call will consume all fields. Marked allow(dead_code) so
// the cargo build stays warning-clean.
#[allow(dead_code)]
#[derive(Clone)]
pub struct GenerateJob {
    pub id: u64,
    /// Which backend should run this job. Set on the UI side from the live
    /// Base model string at click-time. Worker dispatches on this before
    /// touching any params — see `mock::run`'s outer loop.
    pub model_kind: ModelKind,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f32,
    pub seed: i64,
    pub sampler: String,
    pub scheduler: String,
    /// Absolute path to the main DiT/UNet weights file (or directory) chosen
    /// via the Base ComboBox. `None` means the worker should fall back to its
    /// hardcoded default path (backward-compat — preserves the existing
    /// variant-based dispatch for Z-Image base/turbo, Klein 4B/9B, etc.).
    /// When `Some`, this OVERRIDES the worker's `const *_PATH` for the main
    /// DiT/UNet load ONLY. Text encoder / VAE paths stay hardcoded — the
    /// ComboBox drives the main checkpoint selection, not the satellite
    /// encoders. Used primarily so a user's GGUF selection in the UI actually
    /// reaches the load site (without this, every `.ends_with(".gguf")` check
    /// in the workers is dead code; see SKEPTIC_GGUF_WIRING.md P0-1).
    pub path: Option<String>,
}

/// Owned by `FlameInferenceApp`. Drop = channel close = worker shutdown.
pub struct WorkerHandle {
    pub tx: Sender<UiMsg>,
    pub rx: Receiver<WorkerEvent>,
}

/// Spawn the worker thread. Returns a handle the UI uses to talk to it.
///
/// The `egui::Context` is moved into the worker so it can call
/// `ctx.request_repaint()` after every event — without that, egui won't
/// know to redraw until the user moves the mouse, and progress events would
/// queue up invisibly.
pub fn spawn_worker(ctx: egui::Context) -> WorkerHandle {
    let (ui_tx, ui_rx) = unbounded::<UiMsg>();
    let (ev_tx, ev_rx) = unbounded::<WorkerEvent>();
    std::thread::Builder::new()
        .name("flame-inference-worker".into())
        .spawn(move || mock::run(ui_rx, ev_tx, ctx))
        .expect("failed to spawn inference worker thread");
    WorkerHandle { tx: ui_tx, rx: ev_rx }
}
