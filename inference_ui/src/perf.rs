//! GPU telemetry via NVML.
//!
//! The brief (Phase 6) calls for live GPU readouts — name, temperature,
//! VRAM used/total, GPU utilization — feeding the perf footer in the
//! right panel at 2 Hz while generating, 1 Hz idle.
//!
//! NVML is gated behind a feature flag (`nvml`, default-on). When the
//! feature is off, or the runtime `Nvml::init()` fails (no driver, AMD
//! machine, headless container, etc.), we fall back to a no-op backend
//! that surfaces a placeholder snapshot. The perf footer in that case
//! shows whatever is already in `state.perf` — Phase 4's mock values
//! still display until a real backend boots up. Acceptable per spec.
//!
//! AGENT-DEFAULT: single-GPU only (device 0). Multi-GPU enumeration is
//! explicitly out of scope per the brief.

use crate::state::PerfTelemetry;

#[cfg(feature = "nvml")]
mod backend {
    use super::*;
    use nvml_wrapper::Nvml;

    /// Live NVML handle. Holding `Nvml` keeps the underlying NVML library
    /// initialized for the lifetime of the app — re-initializing on every
    /// poll would be measurably slower and is unnecessary.
    pub struct NvmlBackend {
        nvml: Nvml,
    }

    impl NvmlBackend {
        pub fn new() -> anyhow::Result<Self> {
            let nvml = Nvml::init()?;
            Ok(Self { nvml })
        }

        /// Snapshot the current GPU state. Errors propagate up so the
        /// caller can log + leave `state.perf` untouched (rather than
        /// blanking the readout on a transient driver hiccup).
        pub fn poll(&self, device_idx: u32) -> anyhow::Result<PerfTelemetry> {
            let dev = self.nvml.device_by_index(device_idx)?;
            // Many NVML calls return `Result` because the driver may
            // refuse some queries on certain devices (eg. consumer cards
            // without ECC). We swallow individual sub-errors with `.ok()`
            // so a missing temperature sensor doesn't blank the entire
            // readout — only `device_by_index` and `memory_info` are
            // treated as hard failures.
            let name = dev.name().unwrap_or_else(|_| "Unknown GPU".into());
            let memory = dev.memory_info()?;
            let util = dev.utilization_rates().ok();
            let temp = dev
                .temperature(
                    nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                )
                .ok();
            // NVML returns bytes — convert to GiB (binary) to match how
            // every other VRAM readout in the ML world reports it. The
            // perf footer formats as "X.X / Y.Y GB" so the precision is
            // already truncated to a single decimal.
            const BYTES_PER_GIB: f32 = 1024.0 * 1024.0 * 1024.0;
            Ok(PerfTelemetry {
                gpu_name: name,
                vram_used_gb: memory.used as f32 / BYTES_PER_GIB,
                vram_total_gb: memory.total as f32 / BYTES_PER_GIB,
                gpu_util_pct: util.map(|u| u.gpu as f32).unwrap_or(0.0),
                temperature_c: temp.map(|t| t as f32).unwrap_or(0.0),
            })
        }
    }
}

#[cfg(not(feature = "nvml"))]
mod backend {
    use super::*;

    /// No-op backend used when the `nvml` feature is disabled at compile
    /// time. `new()` succeeds so the app's polling code path stays
    /// uniform; `poll()` returns a marker snapshot rather than mock
    /// numbers — the perf footer should make it visible to the user that
    /// telemetry is genuinely unavailable rather than just stale.
    pub struct NvmlBackend;

    impl NvmlBackend {
        pub fn new() -> anyhow::Result<Self> {
            Ok(Self)
        }
        pub fn poll(&self, _device_idx: u32) -> anyhow::Result<PerfTelemetry> {
            Ok(PerfTelemetry {
                gpu_name: "GPU (NVML disabled)".into(),
                vram_used_gb: 0.0,
                vram_total_gb: 0.0,
                gpu_util_pct: 0.0,
                temperature_c: 0.0,
            })
        }
    }
}

pub use backend::NvmlBackend;
