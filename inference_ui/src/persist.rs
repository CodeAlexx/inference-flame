//! AppState load/save via RON, with a debounced writer.
//!
//! Per spec (README "State shape"): the app persists to
//!   - Linux:   ~/.config/flame/state.ron
//!   - macOS:   ~/Library/Application Support/flame/state.ron
//!   - Windows: %APPDATA%\Flame\state.ron
//! on every change, debounced ~500ms so a long DragValue drag doesn't
//! pound the disk on every frame.
//!
//! The transient `serde(skip)` fields on `AppState` (queue, perf,
//! generating, …) are intentionally not round-tripped: their `Default`
//! / `mock()` impls run on load. So a launch reads back saved settings
//! but boots a fresh queue + perf snapshot.
//!
//! AGENT-DEFAULT: write strategy is "atomic via tmp + rename". On Unix
//! `rename` is atomic across the same filesystem; on Windows it is
//! atomic since at least NTFS. This avoids a half-written state.ron if
//! the process is killed mid-write.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;

use crate::state::AppState;

/// Cross-platform application config dir. Falls back to the current
/// working directory if `dirs::config_dir()` is `None` (a profile-less
/// environment — extremely rare, but worth not panicking on).
pub fn config_dir() -> PathBuf {
    if let Some(d) = dirs::config_dir() {
        d.join("flame")
    } else {
        PathBuf::from(".")
    }
}

/// Full path to the on-disk state file.
pub fn state_path() -> PathBuf {
    config_dir().join("state.ron")
}

/// Try to load a previously-persisted AppState. Returns `None` on any
/// error (file missing, parse failure, schema drift, etc.) — the caller
/// then falls back to `AppState::default()`. We intentionally don't
/// surface load errors to the user: a corrupt or mismatched state file
/// shouldn't block app launch.
pub fn load() -> Option<AppState> {
    let path = state_path();
    if !path.exists() {
        return None;
    }
    let s = std::fs::read_to_string(&path).ok()?;
    match ron::from_str::<AppState>(&s) {
        Ok(state) => Some(state),
        Err(e) => {
            log::warn!(
                "Failed to deserialize {} — falling back to defaults: {}",
                path.display(),
                e
            );
            None
        }
    }
}

/// Atomically write the AppState to disk. Errors propagate so the app
/// can log them; we never panic on a save failure (transient permission
/// issue, filesystem full, etc. — losing one save is recoverable).
pub fn save(state: &AppState) -> anyhow::Result<()> {
    let path = state_path();
    if let Some(parent) = path.parent() {
        // create_dir_all is idempotent — first launch makes the
        // ~/.config/flame/ dir, subsequent launches no-op.
        std::fs::create_dir_all(parent).ok();
    }
    let s = ron::ser::to_string_pretty(state, ron::ser::PrettyConfig::default())
        .context("serialize state to RON")?;
    // Atomic write: write to <path>.tmp, then rename over <path>. Rename
    // is atomic on POSIX + NTFS, so a kill mid-write at worst leaves an
    // orphan .tmp file behind (cleaned up on the next save).
    let tmp = path.with_extension("ron.tmp");
    std::fs::write(&tmp, s).context("write tmp state file")?;
    std::fs::rename(&tmp, &path).context("rename tmp state file into place")?;
    Ok(())
}

/// Debouncer for state-changed → save flow.
///
/// Usage pattern, called from `App::update`:
/// ```ignore
/// if state_changed { debouncer.mark(); }
/// if debouncer.should_save() {
///     persist::save(&state)?;
///     debouncer.flush();
/// }
/// ```
///
/// `mark()` records "user just changed something". `should_save()`
/// returns true once `delay` has elapsed since the *last* mark — so a
/// continuous slider drag doesn't trigger a save until the user pauses.
pub struct DebouncedSave {
    /// Most recent change timestamp. `None` when no changes are
    /// pending (either we've never seen a change, or the latest changes
    /// have already been flushed).
    last_change: Option<Instant>,
    /// Most recent successful save timestamp. Used to gate `should_save()`
    /// so we don't repeatedly save the same state.
    last_saved: Option<Instant>,
    delay: Duration,
}

impl DebouncedSave {
    pub fn new(delay_ms: u64) -> Self {
        Self {
            last_change: None,
            last_saved: None,
            delay: Duration::from_millis(delay_ms),
        }
    }

    /// Record that the state changed. Resets the debounce timer.
    pub fn mark(&mut self) {
        self.last_change = Some(Instant::now());
    }

    /// True when:
    ///   - There IS a pending change (last_change is Some), AND
    ///   - That change is older than `delay`, AND
    ///   - We haven't already saved a snapshot newer than that change.
    pub fn should_save(&self) -> bool {
        match self.last_change {
            Some(t) if t.elapsed() >= self.delay => match self.last_saved {
                Some(s) => s < t,
                None => true,
            },
            _ => false,
        }
    }

    /// Mark a save as complete. Called after a successful `persist::save`.
    pub fn flush(&mut self) {
        self.last_saved = Some(Instant::now());
    }

    /// True if there are pending changes that haven't been written yet.
    /// Used by the on-exit hook to force one final flush regardless of
    /// the debounce window.
    pub fn has_pending(&self) -> bool {
        match (self.last_change, self.last_saved) {
            (Some(c), Some(s)) => s < c,
            (Some(_), None) => true,
            (None, _) => false,
        }
    }
}
