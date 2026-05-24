//! L2P transformer-block intra-block probe registry — bisecting the body
//! gradient cascade.
//!
//! Set env `L2P_BLOCK_PROBE_LAYER=29` (or any layer number) to capture probe
//! TensorIds inside that single layer's `transformer_block`. After backward,
//! caller can read out grads for each probe via flame_core's
//! `take_retained_intermediate_grads`.

use std::collections::HashMap;
use std::sync::Mutex;

use flame_core::tensor::TensorId;

static BLOCK_PROBES: Mutex<Option<HashMap<String, TensorId>>> = Mutex::new(None);

/// Arm probes for the layer specified by `L2P_BLOCK_PROBE_LAYER` env var.
/// No-op if env var isn't set. Clears any previous capture.
pub fn arm_for_env() {
    if std::env::var("L2P_BLOCK_PROBE_LAYER").is_ok() {
        if let Ok(mut g) = BLOCK_PROBES.lock() {
            *g = Some(HashMap::new());
        }
    }
}

/// Returns true if the given `prefix` (e.g., "layers.29") matches the
/// active probe layer. Cheap — single env-var read per call.
pub fn is_target_layer(prefix: &str) -> bool {
    if let Ok(target) = std::env::var("L2P_BLOCK_PROBE_LAYER") {
        return prefix == format!("layers.{target}");
    }
    false
}

/// Record a tensor's ID under a probe name. No-op when not armed.
pub fn record(name: &str, id: TensorId) {
    if let Ok(mut g) = BLOCK_PROBES.lock() {
        if let Some(map) = g.as_mut() {
            map.insert(name.to_string(), id);
        }
    }
}

/// Take and reset the captured probes.
pub fn take() -> HashMap<String, TensorId> {
    if let Ok(mut g) = BLOCK_PROBES.lock() {
        g.take().unwrap_or_default()
    } else {
        HashMap::new()
    }
}
