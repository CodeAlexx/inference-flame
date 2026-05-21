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
//! Probes recorded for layer 0 (V-path only as of 2026-05-20):
//! - `"v_proj_out"`  — V tensor immediately after `v_proj` forward (pre-reshape).
//!   Gradient here is the last point upstream of `v_proj.lora_B.grad`.
//! - `"attn_out"`    — SDPA output before reshape into `o_proj` input.
//!   Gradient here is what enters SDPA's backward from the o_proj side.

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
