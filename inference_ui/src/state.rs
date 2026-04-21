//! Persistent application state — declared in Phase 1, populated in later phases.
//!
//! These types mirror the README "State shape" section verbatim. Phase 1 only
//! gives us the type names + Default impls so later phases can extend them
//! without churning every callsite. Don't add UI behavior here — these are
//! pure data containers with `serde` derives ready for the Phase 6 persist
//! layer (`%APPDATA%\Flame\state.ron`).

use std::path::PathBuf;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mode {
    Image,
    Video,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Image
    }
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Image => "Image",
            Mode::Video => "Video",
        }
    }
}

/// Image-mode tasks: T2I, I2I, IC-LoRA. Video-mode tasks: T2V, I2V, A2V, IC-LoRA.
/// They share the enum so per-mode `ModeSettings` can both reference it; the
/// Task picker filters which variants are valid per current `Mode` (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    T2I,
    I2I,
    T2V,
    I2V,
    A2V,
    IcLora,
}

impl Default for Task {
    fn default() -> Self {
        Task::T2I
    }
}

impl Task {
    /// Long-form label as shown in the Task ComboBox (matches JSX
    /// `TASK_LABELS` table).
    pub fn label(self) -> &'static str {
        match self {
            Task::T2I => "T2I — Text to Image",
            Task::I2I => "I2I — Image to Image",
            Task::T2V => "T2V — Text to Video",
            Task::I2V => "I2V — Image to Video",
            Task::A2V => "A2V — Audio to Video",
            Task::IcLora => "IC-LoRA — In-Context LoRA",
        }
    }

    pub fn for_image() -> &'static [Task] {
        &[Task::T2I, Task::I2I, Task::IcLora]
    }

    pub fn for_video() -> &'static [Task] {
        &[Task::T2V, Task::I2V, Task::A2V, Task::IcLora]
    }

    /// Short badge form (T2I, I2I, IC-LoRA, T2V, I2V, A2V) — used by the
    /// canvas panel header per spec line 108. The long `label()` form is
    /// the prefix with " — " stripped off.
    pub fn short_label(self) -> &'static str {
        match self {
            Task::T2I => "T2I",
            Task::I2I => "I2I",
            Task::T2V => "T2V",
            Task::I2V => "I2V",
            Task::A2V => "A2V",
            Task::IcLora => "IC-LoRA",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    Fp16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    Q8_0,
    Q4KM,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::Bf16
    }
}

impl Precision {
    pub fn label(self) -> &'static str {
        match self {
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
            Precision::Fp8E4m3 => "fp8_e4m3",
            Precision::Fp8E5m2 => "fp8_e5m2",
            Precision::Q8_0 => "q8_0",
            Precision::Q4KM => "q4_k_m",
        }
    }

    pub fn all() -> &'static [Precision] {
        &[
            Precision::Fp16,
            Precision::Bf16,
            Precision::Fp8E4m3,
            Precision::Fp8E5m2,
            Precision::Q8_0,
            Precision::Q4KM,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedMode {
    Random,
    Fixed,
    Increment,
}

impl Default for SeedMode {
    fn default() -> Self {
        SeedMode::Random
    }
}

impl SeedMode {
    pub fn label(self) -> &'static str {
        match self {
            SeedMode::Random => "random",
            SeedMode::Fixed => "fixed",
            SeedMode::Increment => "increment",
        }
    }

    pub fn all() -> &'static [SeedMode] {
        &[SeedMode::Random, SeedMode::Fixed, SeedMode::Increment]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraSlot {
    pub name: String,
    pub path: String,
    pub strength: f32,
    pub active: bool,
}

impl Default for LoraSlot {
    fn default() -> Self {
        // Newly-added LoRAs are active by default — matches the JSX prototype
        // (`params-panel.jsx:175`) where a freshly-added entry has `on: true`.
        // SKEPTIC P2 #24 flagged the auto-derived `false`.
        Self {
            name: "new-lora.safetensors".into(),
            path: String::new(),
            strength: 1.0,
            active: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlNetModel {
    None,
    Canny,
    Depth,
    Pose,
    Tile,
    Lineart,
}

impl Default for ControlNetModel {
    fn default() -> Self {
        ControlNetModel::None
    }
}

impl ControlNetModel {
    pub fn label(self) -> &'static str {
        match self {
            ControlNetModel::None => "none",
            ControlNetModel::Canny => "canny",
            ControlNetModel::Depth => "depth",
            ControlNetModel::Pose => "pose",
            ControlNetModel::Tile => "tile",
            ControlNetModel::Lineart => "lineart",
        }
    }

    pub fn all() -> &'static [ControlNetModel] {
        &[
            ControlNetModel::None,
            ControlNetModel::Canny,
            ControlNetModel::Depth,
            ControlNetModel::Pose,
            ControlNetModel::Tile,
            ControlNetModel::Lineart,
        ]
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ControlNet {
    pub source: Option<String>,
    pub model: ControlNetModel,
    pub strength: f32,
    pub denoise: f32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AdvSamplingOpts {
    pub clip_skip: u32,
    pub eta: f32,
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub restart_sampling: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerfOpts {
    pub attention: String,        // "flash-attn-2" | "sdpa" | "xformers" | "math"
    pub torch_compile: bool,
    pub tiled_vae: bool,
    pub cpu_offload: String,      // "none" | "cpu" | "sequential" | "model"
    pub vram_budget_gb: u32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OutputOpts {
    pub folder: String,
    pub filename_template: String,
    pub save_metadata: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModeSettings {
    pub task: Task,
    pub model: String,                // filename relative to weights dir
    pub vae: String,
    pub precision: Precision,
    pub resolution_preset: String,
    pub width: u32,
    pub height: u32,
    pub frames: Option<u32>,          // video only
    pub fps: Option<u32>,             // video only
    pub sampler: String,
    pub scheduler: String,
    pub steps: u32,
    pub cfg: f32,
    pub loras: Vec<LoraSlot>,
    pub controlnet: ControlNet,
    pub advanced_sampling: AdvSamplingOpts,
    pub perf: PerfOpts,
    pub output: OutputOpts,
}

impl ModeSettings {
    /// Sensible first-launch defaults for the Image tab. These doubled as
    /// SKEPTIC P2 #23 / #25 / #26 fixes: `Task::T2I`, `width/height = 1024`,
    /// non-zero steps, etc.
    pub fn image_default() -> Self {
        Self {
            task: Task::T2I,
            model: "flux1-dev.safetensors".into(),
            vae: "ae.safetensors (auto)".into(),
            precision: Precision::Bf16,
            resolution_preset: "1024×1024  ·  1:1".into(),
            width: 1024,
            height: 1024,
            frames: None,
            fps: None,
            sampler: "Euler".into(),
            scheduler: "karras".into(),
            steps: 28,
            cfg: 4.5,
            loras: vec![
                LoraSlot { name: "detail-tweaker-xl".into(), path: String::new(), strength: 0.8, active: true },
                LoraSlot { name: "film-photography-v2".into(), path: String::new(), strength: 1.1, active: true },
                LoraSlot { name: "anime-style-pony".into(), path: String::new(), strength: 0.6, active: false },
            ],
            controlnet: ControlNet { model: ControlNetModel::None, strength: 1.0, denoise: 0.6, source: None },
            advanced_sampling: AdvSamplingOpts { clip_skip: 2, eta: 0.0, sigma_min: 0.03, sigma_max: 14.6, restart_sampling: false },
            perf: PerfOpts { attention: "flash-attn-2".into(), torch_compile: false, tiled_vae: false, cpu_offload: "none".into(), vram_budget_gb: 24 },
            output: OutputOpts { folder: r"D:\flame\out\2026-04-19".into(), filename_template: "{seed}-{model}-{steps}".into(), save_metadata: true },
        }
    }

    pub fn video_default() -> Self {
        Self {
            task: Task::T2V,
            model: "klein9b.safetensors".into(),
            vae: "ae.safetensors (auto)".into(),
            precision: Precision::Bf16,
            resolution_preset: "1280×720  ·  16:9".into(),
            width: 1280,
            height: 720,
            frames: Some(81),
            fps: Some(16),
            sampler: "UniPC".into(),
            scheduler: "karras".into(),
            steps: 28,
            cfg: 5.0,
            loras: Vec::new(),
            controlnet: ControlNet { model: ControlNetModel::None, strength: 1.0, denoise: 0.6, source: None },
            advanced_sampling: AdvSamplingOpts { clip_skip: 0, eta: 0.0, sigma_min: 0.03, sigma_max: 14.6, restart_sampling: false },
            perf: PerfOpts { attention: "flash-attn-2".into(), torch_compile: false, tiled_vae: true, cpu_offload: "none".into(), vram_budget_gb: 24 },
            output: OutputOpts { folder: r"D:\flame\out\2026-04-19".into(), filename_template: "{seed}-{model}-{steps}".into(), save_metadata: true },
        }
    }
}

// --- Phase 4 — Queue / History / Perf ------------------------------------
//
// These types are transient (`#[serde(skip)]` on the AppState fields) — none
// of this should survive across launches. The brief explicitly states that
// running/queued jobs and live telemetry are "telemetry-class state" not
// suited for the on-disk RON persist layer (Phase 6). History *could*
// eventually be persisted (it survives across launches in most generators),
// but Phase 4 keeps it transient since real disk-scan / thumbnail loading
// arrives in a later phase — until then a serialized list of mock items
// would just spam the persist file with stub data.

/// Active right-panel tab. Local to that panel — not persisted.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum QueueTab {
    #[default]
    Queue,
    History,
}

/// One job slot — either currently running (in `QueueState::running`) or
/// pending (`QueueState::queued[..]`). `progress` and `eta_secs` are only
/// meaningful for the running job; queued entries leave them at defaults.
#[derive(Debug, Clone)]
pub struct QueueJob {
    pub id: u64,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub sampler: String,
    /// 0.0..=1.0 — running job only. Queued jobs leave at 0.0.
    pub progress: f32,
    /// Seconds remaining for the running job. None for queued.
    pub eta_secs: Option<u32>,
}

/// One completed output. `thumbnail_path` is reserved for Phase 5+ disk
/// loading — Phase 4 paints a procedural colored rectangle keyed off `id`.
#[derive(Debug, Clone)]
pub struct HistoryItem {
    pub id: u64,
    pub prompt: String,
    pub thumbnail_path: Option<PathBuf>,
    pub timestamp: SystemTime,
}

/// Right-panel queue/history container. `active_tab` lives here (not on
/// AppState) because it's strictly local to the panel — switching modes
/// or tabs in the menubar must not affect which sub-tab is open.
#[derive(Debug, Clone, Default)]
pub struct QueueState {
    pub queued: Vec<QueueJob>,
    pub running: Option<QueueJob>,
    pub history: Vec<HistoryItem>,
    pub active_tab: QueueTab,
}

/// GPU telemetry snapshot. Phase 4 ships **mock data only** — the
/// `nvml-wrapper` integration lands in Phase 6. Fields use SI-ish units
/// (`gb`, `pct`, `c` for Celsius) so the UI stays unit-agnostic.
#[derive(Debug, Clone, Default)]
pub struct PerfTelemetry {
    pub gpu_name: String,
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
    pub gpu_util_pct: f32,
    pub temperature_c: f32,
}

impl QueueState {
    /// Seed with visible mock content so Phase 4 has something to render.
    /// AGENT-DEFAULT: prompts/sizes/timing pulled from the `queue-panel.jsx`
    /// `MOCK_QUEUE` / `MOCK_HISTORY` arrays so the visual feel matches the
    /// design prototype byte-for-byte where it's already been pixel-tuned.
    pub fn mock() -> Self {
        // SystemTime::now() - n minutes; saturating_sub handles the
        // theoretical case where clock is at UNIX_EPOCH (won't happen on
        // a real desktop but defensive).
        let now = SystemTime::now();
        let ago = |mins: u64| {
            now.checked_sub(std::time::Duration::from_secs(mins * 60))
                .unwrap_or(now)
        };
        Self {
            running: Some(QueueJob {
                id: 1,
                prompt: "cinematic portrait, 85mm, warm afternoon light, film grain".into(),
                width: 1024,
                height: 1024,
                steps: 28,
                sampler: "euler".into(),
                progress: 0.60,
                eta_secs: Some(12),
            }),
            queued: vec![
                QueueJob {
                    id: 2,
                    prompt: "top-down tactical map of a cyberpunk city, neon accents".into(),
                    width: 1024,
                    height: 1024,
                    steps: 28,
                    sampler: "euler".into(),
                    progress: 0.0,
                    eta_secs: None,
                },
                QueueJob {
                    id: 3,
                    prompt: "wes anderson still frame, symmetrical, pastel kitchen".into(),
                    width: 1280,
                    height: 720,
                    steps: 32,
                    sampler: "dpm++".into(),
                    progress: 0.0,
                    eta_secs: None,
                },
                QueueJob {
                    id: 4,
                    prompt: "macro shot of molten copper on black stone, shallow dof".into(),
                    width: 1024,
                    height: 1024,
                    steps: 24,
                    sampler: "unipc".into(),
                    progress: 0.0,
                    eta_secs: None,
                },
            ],
            history: vec![
                HistoryItem {
                    id: 100,
                    prompt: "volcanic landscape at dusk".into(),
                    thumbnail_path: None,
                    timestamp: ago(2),
                },
                HistoryItem {
                    id: 101,
                    prompt: "forge sparks, slow motion".into(),
                    thumbnail_path: None,
                    timestamp: ago(4),
                },
                HistoryItem {
                    id: 102,
                    prompt: "ember macro, shallow dof".into(),
                    thumbnail_path: None,
                    timestamp: ago(9),
                },
                HistoryItem {
                    id: 103,
                    prompt: "desert dune at golden hour".into(),
                    thumbnail_path: None,
                    timestamp: ago(12),
                },
                HistoryItem {
                    id: 104,
                    prompt: "lava cracks abstract".into(),
                    thumbnail_path: None,
                    timestamp: ago(18),
                },
            ],
            active_tab: QueueTab::Queue,
        }
    }
}

impl PerfTelemetry {
    /// Phase 4 placeholder values matching the brief.
    pub fn mock() -> Self {
        Self {
            gpu_name: "RTX 4090".into(),
            vram_used_gb: 19.1,
            vram_total_gb: 24.0,
            gpu_util_pct: 62.0,
            temperature_c: 45.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppState {
    pub theme: Theme,
    pub tab: Mode,
    pub advanced: bool,

    // Independent per-mode settings — switching tabs must NOT stomp the other.
    pub image: ModeSettings,
    pub video: ModeSettings,

    // Shared
    pub prompt: String,
    pub negative: String,
    pub batch_count: u32,
    pub batch_size: u32,
    pub seed: i64,                    // -1 = random
    pub seed_mode: SeedMode,
    // The 🔒 button next to the seed input. AGENT-DEFAULT: shared (not
    // per-mode), since the seed value itself lives on AppState. The brief
    // doesn't pin where this field goes — only "add a `seed_locked: bool`
    // field". A per-mode placement would require the user to lock twice.
    pub seed_locked: bool,

    // Phase 3 — canvas runtime state. `generating` drives the Generate↔Stop
    // button swap and the scan-line overlay. `current_step`/`total_steps` are
    // populated by the inference worker channel in Phase 5; Phase 3 leaves
    // them at 0/0 so the toolbar shows `step 0/0` while the stub is active.
    // None of these are persisted (telemetry-class state).
    #[serde(skip)]
    pub generating: bool,
    #[serde(skip)]
    pub current_step: u32,
    #[serde(skip)]
    pub total_steps: u32,

    // Phase 4 — right-panel queue/history + GPU telemetry. Both transient:
    // the queue is rebuilt on launch (jobs are submitted via the action bar,
    // not persisted), and telemetry is intrinsically per-process. Marked
    // `serde(skip)` so the Phase 6 RON persist round-trip ignores them.
    #[serde(skip)]
    pub queue: QueueState,
    #[serde(skip)]
    pub perf: PerfTelemetry,
}

impl Default for AppState {
    fn default() -> Self {
        // SKEPTIC P2 #23, #25, #26: derive(Default) gave T2I to both modes,
        // batch_count/size = 0, seed = 0. Build legitimate defaults here so
        // the UI never shows out-of-range values on first run.
        Self {
            theme: Theme::default(),
            tab: Mode::default(),
            advanced: false,
            image: ModeSettings::image_default(),
            video: ModeSettings::video_default(),
            prompt: String::new(),
            negative: String::new(),
            batch_count: 1,
            batch_size: 1,
            seed: -1,
            seed_mode: SeedMode::Random,
            seed_locked: false,
            generating: false,
            current_step: 0,
            total_steps: 0,
            // Seed queue + perf with mock data so the right panel has visible
            // content on first launch. Phase 5 will replace mock with real
            // worker submissions; Phase 6 with NVML-driven perf updates.
            queue: QueueState::mock(),
            perf: PerfTelemetry::mock(),
        }
    }
}

impl AppState {
    pub fn current_mut(&mut self) -> &mut ModeSettings {
        match self.tab {
            Mode::Image => &mut self.image,
            Mode::Video => &mut self.video,
        }
    }

    pub fn current(&self) -> &ModeSettings {
        match self.tab {
            Mode::Image => &self.image,
            Mode::Video => &self.video,
        }
    }
}
