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
//! `zimage_kohya_to_flame`, `chroma_kohya_to_flame`, `klein_kohya_to_flame`,
//! `qwenimage_kohya_to_flame`, `sdxl_kohya_to_flame`, `sd15_kohya_to_flame`)
//! that callers pass into
//! `LycorisCollection::apply_to(weights, strength, name_mapper)`.
//!
//! ## Split-QKV → fused-QKV: `fuse_split_qkv`
//!
//! Several flame-core models (FLUX, Z-Image, Klein) store *fused* `qkv`
//! linears, while Kohya/LyCORIS LoRA trainers commonly emit *split*
//! `..._to_q`, `..._to_k`, `..._to_v` adapters. Without help, the per-model
//! mapper has no way to reassemble those.
//!
//! Callers targeting a fused-QKV model should call
//! [`fuse_split_qkv`] **before** `apply_to`. It scans the collection for
//! `<base>_to_q` / `<base>_to_k` / `<base>_to_v` triples, materialises each
//! adapter's ΔW, concatenates them along the OUT axis, and replaces the
//! triple with a single `<base>_qkv` `Full` adapter that the per-model
//! mapper can route normally.
//!
//! Models that keep split QKV in flame-core (Qwen-Image, Chroma, SDXL,
//! SD1.5) should *not* call `fuse_split_qkv`; their mappers map split keys
//! through unchanged.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::Tensor;
pub use lycoris_rs::{LycorisAdapter, LycorisCollection};
use lycoris_rs::{algorithms::full::FullAdapter, LycorisModule};

// ---------------------------------------------------------------------------
// FLUX
// ---------------------------------------------------------------------------

/// FLUX Kohya → flame-core key mapper.
///
/// Strips the `lora_unet_` prefix and re-inserts dots between structural
/// boundaries. Returns `None` for split-QKV adapters or unrecognized
/// patterns; the caller's apply_to silently skips those.
///
/// Example mappings:
/// - `lora_unet_double_blocks_0_img_attn_proj` → `double_blocks.0.img_attn.proj.weight`
/// - `lora_unet_single_blocks_5_linear1` → `single_blocks.5.linear1.weight`
/// - `lora_unet_double_blocks_3_img_attn_to_q` → `None` (split QKV; fuse first)
pub fn flux_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    // Split QKV adapters: caller should run `fuse_split_qkv` first to merge
    // these into a single `_qkv` entry. Anything still split here cannot be
    // mapped onto a fused `qkv` weight.
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

    Some(retokenize_with_atoms(rest, COMPOUND_ATOMS))
}

// ---------------------------------------------------------------------------
// Z-Image (NextDiT)
// ---------------------------------------------------------------------------

/// Z-Image NextDiT Kohya → flame-core key mapper.
///
/// Z-Image LoRAs use the `lycoris_layers_` prefix (NOT `lora_unet_`) and
/// flame stores per-block weights at `layers.{i}.attention.qkv.weight`,
/// `layers.{i}.feed_forward.w1.weight`, `layers.{i}.adaLN_modulation.0.weight`
/// etc. The `_to_out_0` Kohya suffix maps to flame's `attention.out`.
///
/// Split QKV (`..._to_q/_to_k/_to_v`) must be fused via [`fuse_split_qkv`]
/// first; this mapper rejects any leftover split entries with `None`.
///
/// Example mappings:
/// - `lycoris_layers_0_attention_qkv` → `layers.0.attention.qkv.weight`
/// - `lycoris_layers_0_attention_to_out_0` → `layers.0.attention.out.weight`
/// - `lycoris_layers_0_attention_q_norm` → `layers.0.attention.q_norm.weight`
/// - `lycoris_layers_0_feed_forward_w1` → `layers.0.feed_forward.w1.weight`
/// - `lycoris_layers_0_adaLN_modulation_0` → `layers.0.adaLN_modulation.0.weight`
pub fn zimage_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lycoris_layers_")?;

    // Reject anything still split — caller should have fused first.
    if rest.ends_with("_to_q") || rest.ends_with("_to_k") || rest.ends_with("_to_v") {
        return None;
    }

    // Z-Image Kohya names `attention.to_out.0` as `attention_to_out_0`, but
    // flame stores it as `attention.out.weight`. Rewrite the suffix BEFORE
    // re-tokenising so the remaining parts land naturally.
    let rewritten;
    let to_tokenise: &str = if let Some(stem) = rest.strip_suffix("_to_out_0") {
        rewritten = format!("{stem}_out");
        &rewritten
    } else {
        rest
    };

    // Atoms inside Z-Image NextDiT (verified against
    // `inference-flame/src/models/zimage_nextdit.rs`).
    const COMPOUND_ATOMS: &[&str] = &[
        "feed_forward",
        "attention_norm1",
        "attention_norm2",
        "adaLN_modulation",
        "attention",
        "q_norm",
        "k_norm",
    ];

    // Prepend `layers.` so we end up with `layers.0.attention.qkv.weight`.
    let body = retokenize_with_atoms(to_tokenise, COMPOUND_ATOMS);
    Some(format!("layers.{body}"))
}

// ---------------------------------------------------------------------------
// Chroma
// ---------------------------------------------------------------------------

/// Chroma DiT Kohya → flame-core key mapper.
///
/// Chroma is a FLUX derivative: same `lora_unet_` prefix and most of FLUX's
/// atom set, but Chroma removed `img_mod` / `txt_mod` and added the
/// `distilled_guidance_layer` and the diffusers-style block prefixes
/// `transformer_blocks` / `single_transformer_blocks`. flame-core's Chroma
/// uses split-QKV (`attn.to_q/to_k/to_v` per block) so split adapters pass
/// through unchanged here — do NOT call `fuse_split_qkv` for Chroma.
pub fn chroma_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    // Chroma flame stores split q/k/v, so accept split keys here.
    // (No early `_to_q/_to_k/_to_v` rejection.)

    // Atoms drawn from `inference-flame/src/models/chroma_dit.rs`. Note: no
    // `img_mod` / `txt_mod` (Chroma replaced per-block modulation with the
    // distilled_guidance_layer pathway).
    const COMPOUND_ATOMS: &[&str] = &[
        "distilled_guidance_layer",
        "single_transformer_blocks",
        "transformer_blocks",
        "double_blocks",
        "single_blocks",
        "img_attn",
        "txt_attn",
        "img_mlp",
        "txt_mlp",
        "final_layer",
        "time_in",
        "vector_in",
        "txt_in",
        "img_in",
        "in_proj",
        "out_proj",
        "linear_1",
        "linear_2",
        "key_norm",
        "query_norm",
        "norm_q",
        "norm_k",
        "norm_added_q",
        "norm_added_k",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "to_add_out",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "proj_mlp",
        "proj_out",
        "adaLN_modulation",
    ];

    Some(retokenize_with_atoms(rest, COMPOUND_ATOMS))
}

// ---------------------------------------------------------------------------
// Klein
// ---------------------------------------------------------------------------

/// Klein DiT Kohya → flame-core key mapper.
///
/// Klein is a FLUX-style fused-QKV transformer with `double_blocks` and
/// `single_blocks`. Atom set is FLUX-like minus `guidance_in` and
/// `vector_in` (Klein has neither). Split-QKV adapters must be fused via
/// [`fuse_split_qkv`] first.
pub fn klein_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    if rest.ends_with("_to_q") || rest.ends_with("_to_k") || rest.ends_with("_to_v") {
        return None;
    }

    const COMPOUND_ATOMS: &[&str] = &[
        "double_blocks",
        "single_blocks",
        "img_attn",
        "txt_attn",
        "img_mlp",
        "txt_mlp",
        "single_stream_modulation",
        "final_layer",
        "time_in",
        "txt_in",
        "img_in",
        "in_layer",
        "out_layer",
        "key_norm",
        "query_norm",
        "adaLN_modulation",
    ];

    Some(retokenize_with_atoms(rest, COMPOUND_ATOMS))
}

// ---------------------------------------------------------------------------
// Qwen-Image
// ---------------------------------------------------------------------------

/// Qwen-Image DiT Kohya → flame-core key mapper.
///
/// Qwen-Image stores **split** Q/K/V at
/// `transformer_blocks.{i}.attn.to_q/to_k/to_v.weight` (verified against
/// `inference-flame/src/models/qwenimage_dit.rs`), so split adapters pass
/// through unchanged. Do NOT call `fuse_split_qkv` for Qwen-Image.
pub fn qwenimage_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    const COMPOUND_ATOMS: &[&str] = &[
        "transformer_blocks",
        "img_mlp",
        "txt_mlp",
        "img_mod",
        "txt_mod",
        "img_in",
        "txt_in",
        "txt_norm",
        "norm_out",
        "norm_q",
        "norm_k",
        "norm_added_q",
        "norm_added_k",
        "time_text_embed",
        "timestep_embedder",
        "linear_1",
        "linear_2",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "to_add_out",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "proj_out",
    ];

    Some(retokenize_with_atoms(rest, COMPOUND_ATOMS))
}

// ---------------------------------------------------------------------------
// SDXL
// ---------------------------------------------------------------------------

/// SDXL UNet Kohya → flame-core key mapper.
///
/// SDXL Kohya LoRAs use diffusers-style block naming as confirmed by the
/// real LoRAs on disk:
///   `lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q`
/// → `down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight`
///
/// flame-core's SDXL UNet stores split Q/K/V (verified against
/// `inference-flame/src/models/sdxl_unet.rs`), so split adapters pass
/// through unchanged. Do NOT call `fuse_split_qkv` for SDXL.
pub fn sdxl_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    const COMPOUND_ATOMS: &[&str] = &[
        "down_blocks",
        "mid_block",
        "up_blocks",
        "transformer_blocks",
        "input_blocks",
        "middle_block",
        "output_blocks",
        "time_embedding",
        "add_embedding",
        "conv_in",
        "conv_out",
        "conv_norm_out",
        "conv_shortcut",
        "time_emb_proj",
        "linear_1",
        "linear_2",
        "norm_out",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "proj_in",
        "proj_out",
        "ff_net",
        "in_layers",
        "out_layers",
        "emb_layers",
        "skip_connection",
        "attn1",
        "attn2",
        "norm1",
        "norm2",
        "norm3",
    ];

    Some(retokenize_with_atoms(rest, COMPOUND_ATOMS))
}

// ---------------------------------------------------------------------------
// SD 1.5
// ---------------------------------------------------------------------------

/// SD 1.5 UNet Kohya → flame-core key mapper.
///
/// SD 1.5 Kohya LoRAs use diffusers-style block naming
/// (`down_blocks/mid_block/up_blocks/...`). flame-core's SD 1.5 internally
/// remaps these to LDM-style (`input_blocks/middle_block/output_blocks`)
/// during weight loading via `remap_one_sd15_key`, so the mapper here just
/// produces dotted diffusers keys; the BlockOffloader handles the LDM
/// translation. Split Q/K/V passes through unchanged.
pub fn sd15_kohya_to_flame(kohya_key: &str) -> Option<String> {
    let rest = kohya_key.strip_prefix("lora_unet_")?;

    const COMPOUND_ATOMS: &[&str] = &[
        "down_blocks",
        "mid_block",
        "up_blocks",
        "transformer_blocks",
        "input_blocks",
        "middle_block",
        "output_blocks",
        "time_embedding",
        "conv_in",
        "conv_out",
        "conv_norm_out",
        "conv_shortcut",
        "time_emb_proj",
        "linear_1",
        "linear_2",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "proj_in",
        "proj_out",
        "ff_net",
        "in_layers",
        "out_layers",
        "emb_layers",
        "skip_connection",
        "attn1",
        "attn2",
        "norm1",
        "norm2",
        "norm3",
    ];

    Some(retokenize_with_atoms(rest, COMPOUND_ATOMS))
}

// ---------------------------------------------------------------------------
// Tokeniser shared by every per-model mapper.
// ---------------------------------------------------------------------------

/// Walk the input, recognising compound atoms (longest-first) and
/// single-segment identifiers between underscores. Numeric tokens pass
/// through. Always appends `.weight` at the end (Kohya keys are weights).
fn retokenize_with_atoms(rest: &str, compound_atoms: &[&str]) -> String {
    let mut out = String::with_capacity(rest.len() + 8);
    let bytes = rest.as_bytes();
    let mut i = 0;
    let mut first = true;

    while i < bytes.len() {
        // Try to match a compound atom at position i.
        let mut matched = None;
        for atom in compound_atoms {
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
    out
}

// ---------------------------------------------------------------------------
// Split-QKV → fused-QKV fuse-on-load.
// ---------------------------------------------------------------------------

/// Scan a [`LycorisCollection`] for split Q/K/V triples and replace each
/// triple with a single fused `<base>_qkv` `Full` adapter whose `diff` is
/// the three component ΔWs concatenated along the OUT axis.
///
/// Convention: flame-core stores Linear weights as `[IN, OUT]` and Conv
/// weights as `[KH, KW, IC, OC]`, so the OUT axis is dim 1 for Linear and
/// dim 3 for Conv. All three component adapters must report the same input
/// dimensions; only their OUT (per-projection) dim may differ.
///
/// Returns the number of triples successfully fused. Triples where one or
/// two of `_to_q/_to_k/_to_v` are missing, or where shapes are
/// incompatible, are left alone with a warning logged via `eprintln!`.
///
/// Call this BEFORE [`LycorisCollection::apply_to`] when targeting a
/// fused-QKV flame model (FLUX, Z-Image, Klein). Models that keep split
/// QKV in flame (Qwen-Image, Chroma, SDXL, SD 1.5) should not call it.
pub fn fuse_split_qkv(
    coll: &mut LycorisCollection,
    device: Arc<CudaDevice>,
) -> anyhow::Result<usize> {
    // Collect base prefixes that have a `_to_q` entry. We materialise the
    // list first because we'll mutate `coll.adapters` mid-iteration.
    let bases: Vec<String> = coll
        .adapters
        .keys()
        .filter_map(|k| k.strip_suffix("_to_q").map(|b| b.to_string()))
        .collect();

    let _ = device; // currently unused — flame ops infer device from inputs.

    let mut fused_count = 0usize;
    for base in bases {
        let q_key = format!("{base}_to_q");
        let k_key = format!("{base}_to_k");
        let v_key = format!("{base}_to_v");

        // Only fuse when the full triple is present.
        if !coll.adapters.contains_key(&k_key) || !coll.adapters.contains_key(&v_key) {
            eprintln!(
                "lycoris (fuse_split_qkv): skipping incomplete QKV triple at '{base}' \
                 (missing `_to_k` or `_to_v`); leaving `_to_q` in place."
            );
            continue;
        }

        // Materialise the three ΔW tensors. Borrow checker: get them in
        // sequence then drop the &refs before mutating.
        let (q_delta, k_delta, v_delta) = {
            let q = coll.adapters.get(&q_key).expect("just checked");
            let k = coll.adapters.get(&k_key).expect("just checked");
            let v = coll.adapters.get(&v_key).expect("just checked");
            let q_d = adapter_diff_weight(q)?;
            let k_d = adapter_diff_weight(k)?;
            let v_d = adapter_diff_weight(v)?;
            (q_d, k_d, v_d)
        };

        // Validate shapes: must be all 2D Linear or all 4D Conv, and
        // input-axis dims must agree.
        let q_dims = q_delta.dims().to_vec();
        let k_dims = k_delta.dims().to_vec();
        let v_dims = v_delta.dims().to_vec();
        let rank = q_dims.len();

        let out_axis = match rank {
            2 => 1, // Linear [IN, OUT] → OUT is dim 1
            4 => 3, // Conv [KH, KW, IC, OC] → OC is dim 3
            other => {
                eprintln!(
                    "lycoris (fuse_split_qkv): cannot fuse '{base}_to_q' — unsupported rank {other}D \
                     (only 2D Linear and 4D Conv are handled)."
                );
                continue;
            }
        };
        if k_dims.len() != rank || v_dims.len() != rank {
            eprintln!(
                "lycoris (fuse_split_qkv): rank mismatch at '{base}' \
                 (q={q_dims:?}, k={k_dims:?}, v={v_dims:?}); leaving split."
            );
            continue;
        }
        // All non-OUT dims must agree.
        let mut shape_ok = true;
        for d in 0..rank {
            if d == out_axis {
                continue;
            }
            if q_dims[d] != k_dims[d] || q_dims[d] != v_dims[d] {
                shape_ok = false;
                break;
            }
        }
        if !shape_ok {
            eprintln!(
                "lycoris (fuse_split_qkv): non-OUT shape mismatch at '{base}' \
                 (q={q_dims:?}, k={k_dims:?}, v={v_dims:?}); leaving split."
            );
            continue;
        }

        let fused = Tensor::cat(&[&q_delta, &k_delta, &v_delta], out_axis)
            .map_err(|e| anyhow::anyhow!("cat({base}_to_qkv) failed: {e:?}"))?;

        // Swap the triple for a single Full adapter under `<base>_qkv`.
        coll.adapters.remove(&q_key);
        coll.adapters.remove(&k_key);
        coll.adapters.remove(&v_key);
        let fused_key = format!("{base}_qkv");
        coll.adapters.insert(
            fused_key,
            LycorisAdapter::Full(FullAdapter {
                diff: fused,
                diff_b: None,
            }),
        );

        fused_count += 1;
    }

    Ok(fused_count)
}

/// Compute ΔW for any adapter variant. Mirrors `LycorisAdapter::delta_weight`
/// but uses the `LycorisModule` trait so the alpha/rank scale is folded in
/// (Full adapters return their stored `diff` directly, no scale).
fn adapter_diff_weight(adapter: &LycorisAdapter) -> anyhow::Result<Tensor> {
    match adapter {
        LycorisAdapter::LoCon(m) => m
            .get_diff_weight()
            .map_err(|e| anyhow::anyhow!("LoCon get_diff_weight failed: {e:?}")),
        LycorisAdapter::LoHa(m) => m
            .get_diff_weight()
            .map_err(|e| anyhow::anyhow!("LoHa get_diff_weight failed: {e:?}")),
        LycorisAdapter::LoKr(m) => m
            .get_diff_weight()
            .map_err(|e| anyhow::anyhow!("LoKr get_diff_weight failed: {e:?}")),
        LycorisAdapter::Full(m) => m
            .delta_weight(1.0)
            .map_err(|e| anyhow::anyhow!("Full delta_weight failed: {e:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // FLUX
    // -----------------------------------------------------------------------

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
        // Caller should fuse_split_qkv first; raw split returns None.
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

    // -----------------------------------------------------------------------
    // Z-Image
    // -----------------------------------------------------------------------

    #[test]
    fn zimage_basic_mapping() {
        // Verified by inspecting zimageLokrEri_000002250.safetensors on disk
        // and `inference-flame/src/models/zimage_nextdit.rs` weight reads.
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_0_attention_qkv"),
            Some("layers.0.attention.qkv.weight".to_string()),
        );
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_12_feed_forward_w1"),
            Some("layers.12.feed_forward.w1.weight".to_string()),
        );
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_3_adaLN_modulation_0"),
            Some("layers.3.adaLN_modulation.0.weight".to_string()),
        );
    }

    #[test]
    fn zimage_to_out_0_rewrites_to_out() {
        // Kohya `attention_to_out_0` → flame `attention.out.weight`.
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_0_attention_to_out_0"),
            Some("layers.0.attention.out.weight".to_string()),
        );
    }

    #[test]
    fn zimage_split_qkv_returns_none() {
        // Fuse_split_qkv is expected to have run first.
        assert!(zimage_kohya_to_flame("lycoris_layers_0_attention_to_q").is_none());
        assert!(zimage_kohya_to_flame("lycoris_layers_0_attention_to_k").is_none());
        assert!(zimage_kohya_to_flame("lycoris_layers_0_attention_to_v").is_none());
    }

    #[test]
    fn zimage_unknown_prefix_returns_none() {
        // Z-Image mapper requires the `lycoris_layers_` prefix.
        assert!(zimage_kohya_to_flame("lora_unet_double_blocks_0_img_attn_qkv").is_none());
        assert!(zimage_kohya_to_flame("text_encoder_q_proj").is_none());
    }

    #[test]
    fn zimage_q_norm_and_attention_norm() {
        // Model-specific: q_norm/k_norm and attention_norm1/2.
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_5_attention_q_norm"),
            Some("layers.5.attention.q_norm.weight".to_string()),
        );
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_5_attention_k_norm"),
            Some("layers.5.attention.k_norm.weight".to_string()),
        );
        assert_eq!(
            zimage_kohya_to_flame("lycoris_layers_2_attention_norm1"),
            Some("layers.2.attention_norm1.weight".to_string()),
        );
    }

    // -----------------------------------------------------------------------
    // Chroma
    // -----------------------------------------------------------------------

    #[test]
    fn chroma_basic_mapping() {
        assert_eq!(
            chroma_kohya_to_flame("lora_unet_transformer_blocks_0_attn_to_q"),
            Some("transformer_blocks.0.attn.to_q.weight".to_string()),
        );
        assert_eq!(
            chroma_kohya_to_flame("lora_unet_single_transformer_blocks_5_proj_mlp"),
            Some("single_transformer_blocks.5.proj_mlp.weight".to_string()),
        );
    }

    #[test]
    fn chroma_split_qkv_passes_through() {
        // Chroma flame stores split QKV — split adapters route normally.
        assert_eq!(
            chroma_kohya_to_flame("lora_unet_transformer_blocks_3_attn_to_q"),
            Some("transformer_blocks.3.attn.to_q.weight".to_string()),
        );
        assert_eq!(
            chroma_kohya_to_flame("lora_unet_transformer_blocks_3_attn_to_v"),
            Some("transformer_blocks.3.attn.to_v.weight".to_string()),
        );
    }

    #[test]
    fn chroma_unknown_prefix_returns_none() {
        assert!(chroma_kohya_to_flame("lycoris_layers_0_attention_qkv").is_none());
        assert!(chroma_kohya_to_flame("not_a_lora_key").is_none());
    }

    #[test]
    fn chroma_distilled_guidance_layer() {
        // Chroma-specific atom.
        assert_eq!(
            chroma_kohya_to_flame("lora_unet_distilled_guidance_layer_in_proj"),
            Some("distilled_guidance_layer.in_proj.weight".to_string()),
        );
        assert_eq!(
            chroma_kohya_to_flame("lora_unet_distilled_guidance_layer_layers_2_linear_1"),
            Some("distilled_guidance_layer.layers.2.linear_1.weight".to_string()),
        );
    }

    // -----------------------------------------------------------------------
    // Klein
    // -----------------------------------------------------------------------

    #[test]
    fn klein_basic_mapping() {
        assert_eq!(
            klein_kohya_to_flame("lora_unet_double_blocks_0_img_attn_qkv"),
            Some("double_blocks.0.img_attn.qkv.weight".to_string()),
        );
        assert_eq!(
            klein_kohya_to_flame("lora_unet_single_blocks_4_linear1"),
            Some("single_blocks.4.linear1.weight".to_string()),
        );
    }

    #[test]
    fn klein_split_qkv_returns_none() {
        // Klein flame uses fused QKV — caller must `fuse_split_qkv` first.
        assert!(klein_kohya_to_flame("lora_unet_double_blocks_0_img_attn_to_q").is_none());
        assert!(klein_kohya_to_flame("lora_unet_double_blocks_0_img_attn_to_v").is_none());
    }

    #[test]
    fn klein_unknown_prefix_returns_none() {
        assert!(klein_kohya_to_flame("lycoris_layers_0_attention_qkv").is_none());
    }

    #[test]
    fn klein_single_stream_modulation() {
        // Klein-specific atom.
        assert_eq!(
            klein_kohya_to_flame("lora_unet_single_stream_modulation_lin"),
            Some("single_stream_modulation.lin.weight".to_string()),
        );
    }

    // -----------------------------------------------------------------------
    // Qwen-Image
    // -----------------------------------------------------------------------

    #[test]
    fn qwenimage_basic_mapping() {
        assert_eq!(
            qwenimage_kohya_to_flame("lora_unet_transformer_blocks_0_attn_to_q"),
            Some("transformer_blocks.0.attn.to_q.weight".to_string()),
        );
        assert_eq!(
            qwenimage_kohya_to_flame("lora_unet_transformer_blocks_5_img_mod_1"),
            Some("transformer_blocks.5.img_mod.1.weight".to_string()),
        );
    }

    #[test]
    fn qwenimage_split_qkv_passes_through() {
        // Qwen-Image flame stores split QKV — pass through.
        assert_eq!(
            qwenimage_kohya_to_flame("lora_unet_transformer_blocks_2_attn_to_k"),
            Some("transformer_blocks.2.attn.to_k.weight".to_string()),
        );
    }

    #[test]
    fn qwenimage_unknown_prefix_returns_none() {
        assert!(qwenimage_kohya_to_flame("lycoris_layers_0_attention_qkv").is_none());
    }

    #[test]
    fn qwenimage_time_text_embed() {
        // Qwen-specific compound atom.
        assert_eq!(
            qwenimage_kohya_to_flame("lora_unet_time_text_embed_timestep_embedder_linear_1"),
            Some("time_text_embed.timestep_embedder.linear_1.weight".to_string()),
        );
    }

    // -----------------------------------------------------------------------
    // SDXL
    // -----------------------------------------------------------------------

    #[test]
    fn sdxl_basic_mapping() {
        // From inspecting EriSdxl.safetensors on disk.
        assert_eq!(
            sdxl_kohya_to_flame(
                "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"
            ),
            Some(
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight".to_string()
            ),
        );
        assert_eq!(
            sdxl_kohya_to_flame("lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_v"),
            Some(
                "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight".to_string()
            ),
        );
    }

    #[test]
    fn sdxl_split_qkv_passes_through() {
        // SDXL flame uses split QKV — pass through.
        assert_eq!(
            sdxl_kohya_to_flame(
                "lora_unet_up_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
            ),
            Some(
                "up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".to_string()
            ),
        );
    }

    #[test]
    fn sdxl_unknown_prefix_returns_none() {
        assert!(sdxl_kohya_to_flame("lycoris_layers_0_attention_qkv").is_none());
        assert!(sdxl_kohya_to_flame("lora_te1_text_model_encoder_layers_0_attn_q").is_none());
    }

    #[test]
    fn sdxl_input_blocks_atom() {
        // LDM-style names (input_blocks/middle_block/output_blocks) also
        // recognised for trainers that emit them directly.
        assert_eq!(
            sdxl_kohya_to_flame("lora_unet_input_blocks_1_0_emb_layers_1"),
            Some("input_blocks.1.0.emb_layers.1.weight".to_string()),
        );
    }

    // -----------------------------------------------------------------------
    // SD 1.5
    // -----------------------------------------------------------------------

    #[test]
    fn sd15_basic_mapping() {
        assert_eq!(
            sd15_kohya_to_flame(
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
            ),
            Some(
                "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".to_string()
            ),
        );
    }

    #[test]
    fn sd15_split_qkv_passes_through() {
        assert_eq!(
            sd15_kohya_to_flame(
                "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v"
            ),
            Some(
                "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.weight".to_string()
            ),
        );
    }

    #[test]
    fn sd15_unknown_prefix_returns_none() {
        assert!(sd15_kohya_to_flame("lycoris_layers_0_attention_qkv").is_none());
    }

    #[test]
    fn sd15_input_blocks_atom() {
        // LDM-native naming for trainers that emit it directly; flame's
        // SD 1.5 BlockOffloader recognises both styles.
        assert_eq!(
            sd15_kohya_to_flame("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k"),
            Some(
                "input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight".to_string()
            ),
        );
    }

    // -----------------------------------------------------------------------
    // fuse_split_qkv
    // -----------------------------------------------------------------------

    /// Build a synthetic LoCon Linear adapter with given [IN, OUT] dims and
    /// a deterministic ΔW = `marker * I_in * sum_OUT(...)` style filling so
    /// that each contributor is identifiable in the fused output.
    #[cfg(test)]
    fn make_synthetic_locon(
        in_dim: usize,
        out_dim: usize,
        device: Arc<CudaDevice>,
        fill: f32,
    ) -> lycoris_rs::algorithms::full::FullAdapter {
        // Easiest: skip the Decomposition, just construct a Full adapter
        // with a hand-built diff tensor [IN, OUT] — `adapter_diff_weight`
        // returns Full's diff directly so behaviour is identical for fuse.
        use flame_core::Shape;
        let n = in_dim * out_dim;
        let data = vec![fill; n];
        let diff = flame_core::Tensor::from_vec(data, Shape::from_dims(&[in_dim, out_dim]), device)
            .expect("from_vec")
            .to_dtype(flame_core::DType::BF16)
            .expect("to bf16");
        lycoris_rs::algorithms::full::FullAdapter { diff, diff_b: None }
    }

    #[test]
    fn fuse_split_qkv_concats_three_locon_adapters() {
        // Build a 3-adapter collection sharing base prefix `block.attn`.
        let dev = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA unavailable, skipping fuse_split_qkv_concats test");
                return;
            }
        };

        const IN: usize = 8;
        const OUT_PER: usize = 4;

        let mut coll = LycorisCollection {
            adapters: std::collections::HashMap::new(),
        };
        coll.adapters.insert(
            "block.attn_to_q".to_string(),
            LycorisAdapter::Full(make_synthetic_locon(IN, OUT_PER, dev.clone(), 1.0)),
        );
        coll.adapters.insert(
            "block.attn_to_k".to_string(),
            LycorisAdapter::Full(make_synthetic_locon(IN, OUT_PER, dev.clone(), 2.0)),
        );
        coll.adapters.insert(
            "block.attn_to_v".to_string(),
            LycorisAdapter::Full(make_synthetic_locon(IN, OUT_PER, dev.clone(), 3.0)),
        );

        let fused = fuse_split_qkv(&mut coll, dev).expect("fuse");
        assert_eq!(fused, 1, "should fuse exactly one triple");
        assert_eq!(coll.adapters.len(), 1, "the 3 split entries should be replaced by 1");

        let fused_entry = coll.adapters.get("block.attn_qkv").expect("qkv key");
        let diff = adapter_diff_weight(fused_entry).expect("diff");
        assert_eq!(
            diff.dims(),
            &[IN, 3 * OUT_PER],
            "fused diff must concat along OUT axis"
        );
    }

    #[test]
    fn fuse_split_qkv_skips_partial_triples() {
        let dev = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA unavailable, skipping fuse_split_qkv_skips_partial test");
                return;
            }
        };

        const IN: usize = 4;
        const OUT_PER: usize = 2;

        let mut coll = LycorisCollection {
            adapters: std::collections::HashMap::new(),
        };
        coll.adapters.insert(
            "blockA_to_q".to_string(),
            LycorisAdapter::Full(make_synthetic_locon(IN, OUT_PER, dev.clone(), 1.0)),
        );
        coll.adapters.insert(
            "blockA_to_k".to_string(),
            LycorisAdapter::Full(make_synthetic_locon(IN, OUT_PER, dev.clone(), 1.0)),
        );
        // No `_to_v` for blockA → partial triple, must be skipped.

        let fused = fuse_split_qkv(&mut coll, dev).expect("fuse");
        assert_eq!(fused, 0, "no triples should fuse");
        assert_eq!(
            coll.adapters.len(),
            2,
            "partial entries left untouched"
        );
        assert!(coll.adapters.contains_key("blockA_to_q"));
        assert!(coll.adapters.contains_key("blockA_to_k"));
    }

    // -----------------------------------------------------------------------
    // End-to-end Z-Image LoRA — gated behind --ignored because it requires
    // the user's safetensors at the hard-coded path.
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn zimage_real_lokr_loads_and_fuses() {
        use std::path::Path;

        let dev = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("CUDA unavailable: {e:?}");
                return;
            }
        };
        let path =
            Path::new("/home/alex/.serenity/models/loras/zimageLokrEri_000002250.safetensors");
        if !path.exists() {
            eprintln!("real Z-Image LoRA not present at {path:?}, skipping");
            return;
        }

        let mut coll = LycorisCollection::load(path, dev.clone()).expect("load");
        let pre_count = coll.adapters.len();
        let fused = fuse_split_qkv(&mut coll, dev).expect("fuse");
        let post_count = coll.adapters.len();

        println!(
            "loaded {} adapters; fused {} qkv triples; {} adapters after",
            pre_count, fused, post_count
        );
        assert!(fused >= 20, "expected at least 20 QKV fuses, got {}", fused);
        assert!(
            post_count == pre_count - 2 * fused,
            "fuse should net out: {} = {} - 2*{}",
            post_count,
            pre_count,
            fused
        );

        // Verify a fused key maps via zimage_kohya_to_flame.
        let any_fused_key = coll
            .adapters
            .keys()
            .find(|k| k.ends_with("_qkv"))
            .expect("at least one _qkv after fuse")
            .clone();
        let mapped = zimage_kohya_to_flame(&any_fused_key);
        assert!(mapped.is_some(), "fused key {} should map", any_fused_key);
        println!("e.g. {} -> {:?}", any_fused_key, mapped);
    }
}
