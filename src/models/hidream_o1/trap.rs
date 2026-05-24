//! Soul.md-style trap for HiDream-O1's attention backward chain.
//!
//! Purpose: capture tensor IDs of layer-0 attention intermediates during
//! forward, then look up their post-backward `.grad` to walk the gradient
//! chain and find where Q/K/V LoRA-B grad cos collapses from ~1.0 to ~0.05.
//!
//! Pattern from `soul.md` 2026-04-25: build the trap *before* you think you
//! need it; the test that walks intermediate `cos_sim` cheaply names the bug
//! site (the first probe that flips from 1.0 to <0.5 is where the corruption
//! starts).
//!
//! Lifecycle:
//!   1) test setup: [`arm_probes`]
//!   2) forward: layer 0 hot path calls [`record_probe`] for each intermediate
//!   3) post-forward, pre-backward: [`take_probes`] returns the captured IDs;
//!      caller passes them to `AutogradContext::retain_intermediate_grads(...)`
//!   4) post-backward: caller calls `take_retained_intermediate_grads()` and
//!      matches the grads back to probe names by ID.
//!
//! Probes recorded for the selected decoder layer:
//! - `"q_proj_out"`  — Q tensor immediately after `q_proj` forward (pre-reshape).
//! - `"k_proj_out"`  — K tensor immediately after `k_proj` forward (pre-reshape).
//! - `"v_proj_out"`  — V tensor immediately after `v_proj` forward (pre-reshape).
//!   Gradient here is the last point upstream of `v_proj.lora_B.grad`.
//! - `"attn_out"`    — SDPA output before reshape into `o_proj` input.
//!   Gradient here is what enters SDPA's backward from the o_proj side.
//! - `"mlp_gate_out"` / `"mlp_up_out"` — SwiGLU inputs after gate/up projection.
//! - `"mlp_inner"`   — SwiGLU output before down projection.
//! - `"mlp_out"`     — MLP output before residual add.
//!
//! ## 2026-05-23 extension — env-driven block_trap parallel
//!
//! The 2026-05-20 `arm_probes()` / `take_probes()` flow is the older API used
//! by `parity_hidream_o1_train_step`. For the new
//! `hidream_o1_grad_chain_parity` binary we mirror the L2P `block_trap`
//! pattern: env-driven arming via `HIDREAM_BLOCK_PROBE_LAYER=N`, a separate
//! probe store that the new binary owns, and a richer `is_target_layer`
//! check so probe code can stay in the hot path with zero overhead when not
//! armed.
//!
//! Both APIs share `record_probe` (the existing entrypoint) — every call site
//! also calls [`record_block`] which routes into the new env-driven store.
//! This keeps the existing `parity_hidream_o1_train_step` consumer working
//! while letting the new grad-chain-parity binary use the L2P-style API
//! verbatim.

use std::collections::HashMap;
use std::sync::Mutex;

use flame_core::TensorId;

static TRAP_PROBES: Mutex<Option<HashMap<String, TensorId>>> = Mutex::new(None);

/// Enable probe capture. Call before the forward pass you want to probe.
/// Clears any previously-captured probes.
pub fn arm_probes() {
    if let Ok(mut g) = TRAP_PROBES.lock() {
        *g = Some(HashMap::new());
    }
}

/// Disable probe capture (does NOT clear already-armed slot). Use to skip
/// future record_probe calls without triggering [`take_probes`] flow.
pub fn disarm_probes() {
    if let Ok(mut g) = TRAP_PROBES.lock() {
        *g = None;
    }
}

/// Record a tensor ID under a probe name. No-op when not armed. Called by
/// decoder hot-path code on the layer 0 iteration.
pub fn record_probe(name: &str, id: TensorId) {
    if let Ok(mut g) = TRAP_PROBES.lock() {
        if let Some(map) = g.as_mut() {
            map.insert(name.to_string(), id);
        }
    }
}

/// Return whether the trap is currently armed (any record_probe call will be
/// recorded). Cheap; used by decoder hot path to skip the probe construction
/// when not active.
pub fn is_armed() -> bool {
    matches!(TRAP_PROBES.lock(), Ok(g) if g.is_some())
}

/// Take the currently-armed probe map. Returns None if not armed. After this
/// call the trap is disarmed.
pub fn take_probes() -> Option<HashMap<String, TensorId>> {
    if let Ok(mut g) = TRAP_PROBES.lock() {
        g.take()
    } else {
        None
    }
}

// ─── 2026-05-23 — L2P-style env-driven block_trap parallel ────────────────────

static BLOCK_PROBES: Mutex<Option<HashMap<String, TensorId>>> = Mutex::new(None);

/// Arm probes for the layer specified by `HIDREAM_BLOCK_PROBE_LAYER` env var.
/// No-op if env var isn't set. Clears any previous capture. Mirrors
/// `inference-flame/src/models/l2p/block_trap.rs::arm_for_env`.
pub fn arm_for_env() {
    if std::env::var("HIDREAM_BLOCK_PROBE_LAYER").is_ok() {
        if let Ok(mut g) = BLOCK_PROBES.lock() {
            *g = Some(HashMap::new());
        }
    }
}

/// Returns true if the given decoder-layer index matches the active probe
/// layer specified by `HIDREAM_BLOCK_PROBE_LAYER`. Cheap — single env-var
/// read per call. Pair with [`record_block`] inside the layer hot path.
pub fn is_target_layer(layer_idx: usize) -> bool {
    if let Ok(target) = std::env::var("HIDREAM_BLOCK_PROBE_LAYER") {
        if let Ok(n) = target.parse::<usize>() {
            return n == layer_idx;
        }
    }
    false
}

/// Record a tensor's ID under a block probe name. No-op when not armed via
/// [`arm_for_env`]. Safe to call alongside [`record_probe`] — they target
/// independent stores.
pub fn record_block(name: &str, id: TensorId) {
    if let Ok(mut g) = BLOCK_PROBES.lock() {
        if let Some(map) = g.as_mut() {
            map.insert(name.to_string(), id);
        }
    }
}

/// Returns true if the L2P-style block-probe store is armed.
pub fn block_is_armed() -> bool {
    matches!(BLOCK_PROBES.lock(), Ok(g) if g.is_some())
}

/// Take and reset the captured block probes. Returns an empty map if not
/// armed. Mirrors `inference-flame/src/models/l2p/block_trap.rs::take`.
pub fn take_block_probes() -> HashMap<String, TensorId> {
    if let Ok(mut g) = BLOCK_PROBES.lock() {
        g.take().unwrap_or_default()
    } else {
        HashMap::new()
    }
}
