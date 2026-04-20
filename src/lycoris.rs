//! Bindings between LyCORIS adapter loaders (`lycoris-rs` crate) and
//! inference-flame model weight conventions.
//!
//! LyCORIS safetensors use Kohya-trainer naming, e.g.
//!   `lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight`
//! while inference-flame model weight dicts use dot-separated names with
//! `.weight` suffixes:
//!   `double_blocks.0.img_attn.qkv.weight`
//!
//! This module provides per-model mappers (`flux_kohya_to_flame`,
//! `sdxl_kohya_to_flame`, etc.) that callers pass into
//! `LycorisCollection::apply_to(weights, strength, name_mapper)`.
//!
//! v1 ships with a FLUX mapper. Other models are TODO — add a
//! `<model>_kohya_to_flame` function following the same pattern.
//!
//! ## Known limitation: split QKV adapters
//!
//! Kohya LoRA trainers sometimes store one adapter per Q/K/V projection
//! (`..._to_q`, `..._to_k`, `..._to_v`) while flame-core's FLUX uses fused
//! `qkv` weights. Mapping a split adapter to fused requires concatenating
//! three separate ΔWs into one tensor. v1 returns `None` for these (skip
//! with a warning); v2 should add a fuse-on-merge path.

pub use lycoris_rs::{LycorisAdapter, LycorisCollection};

/// FLUX Kohya → flame-core key mapper.
///
/// Strips the `lora_unet_` prefix and re-inserts dots between structural
/// boundaries. Returns `None` for split-QKV adapters or unrecognized
/// patterns; the caller's apply_to silently skips those.
///
/// Example mappings:
/// - `lora_unet_double_blocks_0_img_attn_proj` → `double_blocks.0.img_attn.proj.weight`
/// - `lora_unet_single_blocks_5_linear1` → `single_blocks.5.linear1.weight`
/// - `lora_unet_double_blocks_3_img_attn_to_q` → `None` (split QKV, see limitations)
pub fn flux_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    // Split QKV adapters not supported in v1 (would need fuse-on-merge).
    if rest.ends_with("_to_q") || rest.ends_with("_to_k") || rest.ends_with("_to_v") {
        return None;
    }

    // Atoms that contain underscores and must NOT be split. Order matters —
    // longest first so e.g. "double_blocks" matches before "blocks".
    const COMPOUND_ATOMS: &[&str] = &[
        "double_blocks",
        "single_blocks",
        "img_attn",
        "txt_attn",
        "img_mlp",
        "txt_mlp",
        "img_mod",
        "txt_mod",
        "final_layer",
        "time_in",
        "vector_in",
        "guidance_in",
        "txt_in",
        "img_in",
        "key_norm",
        "query_norm",
        "adaLN_modulation",
    ];

    // Tokenize: walk the string, recognize compound atoms or single-segment
    // identifiers (which are everything between underscores). Numeric tokens
    // are kept as-is.
    let mut out = String::with_capacity(rest.len() + 8);
    let bytes = rest.as_bytes();
    let mut i = 0;
    let mut first = true;

    while i < bytes.len() {
        // Try to match a compound atom at position i.
        let mut matched = None;
        for atom in COMPOUND_ATOMS {
            let a = atom.as_bytes();
            if i + a.len() <= bytes.len() && &bytes[i..i + a.len()] == a {
                // Must be followed by `_` or end of string for atom-boundary.
                if i + a.len() == bytes.len() || bytes[i + a.len()] == b'_' {
                    matched = Some(*atom);
                    break;
                }
            }
        }

        if !first {
            out.push('.');
        }
        first = false;

        if let Some(atom) = matched {
            out.push_str(atom);
            i += atom.len();
            if i < bytes.len() && bytes[i] == b'_' {
                i += 1;
            }
        } else {
            // Read until next underscore or end.
            let start = i;
            while i < bytes.len() && bytes[i] != b'_' {
                i += 1;
            }
            out.push_str(&rest[start..i]);
            if i < bytes.len() && bytes[i] == b'_' {
                i += 1;
            }
        }
    }

    out.push_str(".weight");
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_basic_mapping() {
        assert_eq!(
            flux_kohya_to_flame("lora_unet_double_blocks_0_img_attn_proj"),
            Some("double_blocks.0.img_attn.proj.weight".to_string()),
        );
        assert_eq!(
            flux_kohya_to_flame("lora_unet_single_blocks_5_linear1"),
            Some("single_blocks.5.linear1.weight".to_string()),
        );
        assert_eq!(
            flux_kohya_to_flame("lora_unet_double_blocks_3_img_mlp_0"),
            Some("double_blocks.3.img_mlp.0.weight".to_string()),
        );
        assert_eq!(
            flux_kohya_to_flame("lora_unet_double_blocks_7_txt_mod_lin"),
            Some("double_blocks.7.txt_mod.lin.weight".to_string()),
        );
    }

    #[test]
    fn flux_fused_qkv_passes_through() {
        // Kohya names this "qkv" (no split) → mapper accepts it.
        assert_eq!(
            flux_kohya_to_flame("lora_unet_double_blocks_0_img_attn_qkv"),
            Some("double_blocks.0.img_attn.qkv.weight".to_string()),
        );
    }

    #[test]
    fn flux_split_qkv_returns_none() {
        // Until v2 adds fuse-on-merge, split adapters skip with None.
        assert!(flux_kohya_to_flame("lora_unet_double_blocks_0_img_attn_to_q").is_none());
        assert!(flux_kohya_to_flame("lora_unet_double_blocks_0_img_attn_to_k").is_none());
        assert!(flux_kohya_to_flame("lora_unet_double_blocks_0_img_attn_to_v").is_none());
    }

    #[test]
    fn flux_strips_unknown_prefix() {
        assert!(flux_kohya_to_flame("lora_te1_text_model_encoder_layers_0_attn_q").is_none());
        assert!(flux_kohya_to_flame("not_a_lora_key").is_none());
    }

    #[test]
    fn flux_final_layer() {
        assert_eq!(
            flux_kohya_to_flame("lora_unet_final_layer_linear"),
            Some("final_layer.linear.weight".to_string()),
        );
    }
}
