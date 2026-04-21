//! Key-name remapping for GGUF → flame-core conventions.
//!
//! GGUF files in the wild carry keys using whatever convention the exporter
//! used. The two dominant ones for image models are:
//!
//!   - `model.diffusion_model.<rest>` — stable-diffusion.cpp / comfyanonymous
//!     convention for UNets and DiTs.
//!   - `transformer.<rest>`          — HuggingFace diffusers convention for
//!     DiT-style models (FLUX, Chroma, SD3).
//!   - `first_stage_model.<rest>`    — SD.cpp convention for the VAE (rare
//!     in DiT GGUFs; included for completeness).
//!
//! Our existing model loaders consume keys already stripped of those
//! prefixes — they match the pattern the safetensors loader produces.
//! [`default_rename`] strips whichever of the three matches, in priority
//! order.
//!
//! Per-model custom mappers (e.g. Kohya→diffusers axis reshaping) live in
//! [`crate::lycoris`] / the model loaders themselves. This module is only
//! about the top-level namespace.

use std::collections::HashMap;

use flame_core::Tensor;

/// Default GGUF key rename used by [`super::load_file_gguf`].
///
/// Strips, in priority order:
///   1. `model.diffusion_model.`
///   2. `transformer.`
///   3. `first_stage_model.`
///
/// If none match, returns the key unchanged. **Only one prefix is stripped
/// per call** — the conventions are mutually exclusive in every
/// currently-shipping diffusion GGUF (FLUX / SD3 / Chroma / SDXL / SDXL
/// VAE). We've surveyed city96's FLUX repo, HuggingFace diffusers DiT
/// exports, and SD.cpp VAE GGUFs; no file nests two prefixes.
///
/// TODO: if SD.cpp FLUX ever ships `model.diffusion_model.transformer.foo`
/// (combined diffusers-export-plus-sd-cpp style), revisit this — we'd
/// need to loop the three `strip_prefix` calls until none match.
/// Not worth the code complexity today.
pub fn default_rename(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.diffusion_model.") {
        rest.to_string()
    } else if let Some(rest) = key.strip_prefix("transformer.") {
        rest.to_string()
    } else if let Some(rest) = key.strip_prefix("first_stage_model.") {
        rest.to_string()
    } else {
        key.to_string()
    }
}

/// Apply a rename function to every key in `map`, returning a new HashMap.
///
/// Key collisions (two input keys mapping to the same output key) surface
/// as the last-write-wins. For our three-prefix default this is not
/// possible — prefixes are disjoint. Callers supplying custom renamers
/// that might collide should filter beforehand.
pub fn rename_keys<F>(map: HashMap<String, Tensor>, f: F) -> HashMap<String, Tensor>
where
    F: Fn(&str) -> String,
{
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map.into_iter() {
        out.insert(f(&k), v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_diffusion_model_prefix() {
        assert_eq!(
            default_rename("model.diffusion_model.double_blocks.0.img_attn.qkv.weight"),
            "double_blocks.0.img_attn.qkv.weight"
        );
    }

    #[test]
    fn strips_transformer_prefix() {
        assert_eq!(
            default_rename("transformer.single_blocks.0.linear1.weight"),
            "single_blocks.0.linear1.weight"
        );
    }

    #[test]
    fn strips_first_stage_model_prefix() {
        assert_eq!(
            default_rename("first_stage_model.encoder.down.0.block.0.conv1.weight"),
            "encoder.down.0.block.0.conv1.weight"
        );
    }

    #[test]
    fn leaves_other_keys_untouched() {
        assert_eq!(
            default_rename("time_in.in_layer.weight"),
            "time_in.in_layer.weight"
        );
        assert_eq!(default_rename(""), "");
    }
}
