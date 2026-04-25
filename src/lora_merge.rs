//! Pure-Rust LoRA merge for Klein 4B/9B. Handles two formats:
//!
//! ## Klein-trainer format (4B-tuned)
//!
//! Bare `<prefix>.lora_A` / `.lora_B` keys with internal projection names:
//!
//! | LoRA prefix                                          | Base key                                  | Slot         |
//! |------------------------------------------------------|-------------------------------------------|--------------|
//! | `double_blocks.<i>.img_attn.qkv_proj`                | `double_blocks.<i>.img_attn.qkv.weight`   | full overlay |
//! | `double_blocks.<i>.img_attn.out_proj`                | `double_blocks.<i>.img_attn.proj.weight`  | full overlay |
//! | `double_blocks.<i>.txt_attn.qkv_proj`                | `double_blocks.<i>.txt_attn.qkv.weight`   | full overlay |
//! | `double_blocks.<i>.txt_attn.out_proj`                | `double_blocks.<i>.txt_attn.proj.weight`  | full overlay |
//! | `single_blocks.<j>.qkv_proj`                         | `single_blocks.<j>.linear1.weight`        | rows[:9216]  |
//! | `single_blocks.<j>.out_proj`                         | `single_blocks.<j>.linear2.weight`        | cols[:3072]  |
//! | `input_bridges.{latent,text}_bridge.{weight,bias}`   | (not in base — skipped)                   | —            |
//!
//! Slot constants (9216, 3072) are 4B-specific. 9B-tuned klein-trainer
//! LoRAs would need the equivalent (12288, 4096) values; not yet handled.
//!
//! ## ai-toolkit format (Klein 4B/9B)
//!
//! `diffusion_model.<base_key>.lora_A.weight` / `.lora_B.weight`. The
//! base key is just the LoRA prefix with `diffusion_model.` stripped and
//! `.weight` appended; full overlay everywhere. Covers:
//!
//! - `double_blocks.<i>.{img,txt}_attn.{qkv,proj}` → direct base name
//! - `double_blocks.<i>.{img,txt}_mlp.{0,2}` → direct (klein-trainer
//!   doesn't target mlp; ai-toolkit does)
//! - `single_blocks.<j>.{linear1,linear2}` → direct (no slicing — the
//!   LoRA `lora_B` is sized for the full matrix)
//!
//! ## LoRA math (both formats)
//!
//! `delta_W = scale * (lora_B @ lora_A)`, `scale = (alpha/rank) * multiplier`.
//!   `lora_A`: `[rank, in_features]`
//!   `lora_B`: `[out_features, rank]`
//!
//! Reference for klein-trainer: the now-deleted
//! `klein-trainer/scripts/lora_merge.py`. A copy lives next to verified
//! samples at
//! `flame-diffusion/output/klein4b_2k_postbug4/verified_samples/lora_merge.py`.

use flame_core::{trim_cuda_mempool, DType, Error, Result, Tensor};
use std::collections::{BTreeSet, HashMap};

const SINGLE_QKV_ROWS: usize = 9216;
const SINGLE_OUT_COLS: usize = 3072;

/// Z-Image fused QKV is `[3*dim, dim]` with Q/K/V stacked along dim 0.
/// For dim=3840 the per-head ranges are 0..3840, 3840..7680, 7680..11520.
const ZIMAGE_DIM: usize = 3840;

#[derive(Clone, Copy, Debug)]
enum Slot {
    /// Full overlay: base shape == delta shape.
    Full,
    /// Top `n` rows of base get `delta`. base[..n, :] += delta.
    Rows(usize),
    /// Left `n` cols of base get `delta`. base[:, ..n] += delta.
    Cols(usize),
    /// Row-range `[start..start+len]` of base gets `delta`.
    /// Used for Z-Image's split-Q/K/V LoRA delta into fused QKV base.
    RowRange { start: usize, len: usize },
}

/// Detected LoRA file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraFormat {
    /// Klein-trainer: bare `<prefix>.lora_A` keys, custom projection names
    /// (`*_proj`), needs row/col slicing for single_blocks.
    KleinTrainer,
    /// Z-Image trainer (this repo): bare `<prefix>.lora_A` keys with split
    /// Q/K/V (`attention.to_q/to_k/to_v`). Q/K/V deltas merge into the
    /// fused `attention.qkv.weight` row-range. Targets only `layers.<i>.*`.
    ZImageTrainer,
    /// ai-toolkit (FLUX/Klein convention): `diffusion_model.<key>.lora_A.weight`
    /// keys, direct base mapping with `.weight` appended, full overlay.
    AiToolkit,
    /// kohya / sd-scripts SDXL: `lora_(unet|te1|te2)_<path_with_underscores>.lora_(down|up).weight`
    /// + per-module `.alpha` scalar. UNet keys map to LDM-format base by
    /// reversing the underscore-encoding back to dotted path; only the
    /// `lora_unet_*` subset is merged (text-encoder LoRAs are applied at
    /// encode time, not into the UNet weights).
    KohyaSdxl,
}

/// Detect format from the LoRA key shape. ai-toolkit always uses
/// `.lora_A.weight`. Klein-trainer uses `.qkv_proj`/`.out_proj` projection
/// names. Z-Image trainer uses `attention.to_q/to_k/to_v` and `feed_forward.w*`.
/// kohya SDXL uses `lora_(unet|te1|te2)_…lora_(down|up).weight`.
pub fn detect_format(lora: &HashMap<String, Tensor>) -> LoraFormat {
    if lora
        .keys()
        .any(|k| k.starts_with("lora_unet_") || k.starts_with("lora_te1_") || k.starts_with("lora_te2_"))
        && lora
            .keys()
            .any(|k| k.ends_with(".lora_down.weight") || k.ends_with(".lora_up.weight"))
    {
        return LoraFormat::KohyaSdxl;
    }
    if lora.keys().any(|k| k.ends_with(".lora_A.weight") || k.ends_with(".lora_B.weight")) {
        return LoraFormat::AiToolkit;
    }
    if lora
        .keys()
        .any(|k| k.contains(".attention.to_q.lora_") || k.contains(".feed_forward.w1.lora_"))
    {
        return LoraFormat::ZImageTrainer;
    }
    LoraFormat::KleinTrainer
}

fn map_prefix_klein_trainer(prefix: &str) -> Option<(String, Slot)> {
    if prefix.starts_with("input_bridges.") {
        // Bridges aren't in the base; trainer-only. Caller skips silently.
        return None;
    }
    if let Some(rest) = prefix.strip_suffix(".img_attn.qkv_proj") {
        return Some((format!("{rest}.img_attn.qkv.weight"), Slot::Full));
    }
    if let Some(rest) = prefix.strip_suffix(".img_attn.out_proj") {
        return Some((format!("{rest}.img_attn.proj.weight"), Slot::Full));
    }
    if let Some(rest) = prefix.strip_suffix(".txt_attn.qkv_proj") {
        return Some((format!("{rest}.txt_attn.qkv.weight"), Slot::Full));
    }
    if let Some(rest) = prefix.strip_suffix(".txt_attn.out_proj") {
        return Some((format!("{rest}.txt_attn.proj.weight"), Slot::Full));
    }
    if prefix.starts_with("single_blocks.") {
        if let Some(rest) = prefix.strip_suffix(".qkv_proj") {
            return Some((format!("{rest}.linear1.weight"), Slot::Rows(SINGLE_QKV_ROWS)));
        }
        if let Some(rest) = prefix.strip_suffix(".out_proj") {
            return Some((format!("{rest}.linear2.weight"), Slot::Cols(SINGLE_OUT_COLS)));
        }
    }
    None
}

/// ai-toolkit prefix → base key. Strip leading `diffusion_model.` if
/// present and append `.weight`. ai-toolkit always uses full overlay.
fn map_prefix_aitoolkit(prefix: &str) -> Option<(String, Slot)> {
    let stripped = prefix.strip_prefix("diffusion_model.").unwrap_or(prefix);
    // Some ai-toolkit LoRAs (PEFT-style) include `.default` as the adapter
    // name suffix on the prefix. Strip it.
    let stripped = stripped.strip_suffix(".default").unwrap_or(stripped);
    Some((format!("{stripped}.weight"), Slot::Full))
}

/// Build a map from kohya-encoded LoRA prefix → base safetensors key.
///
/// kohya encodes module paths by replacing dots with underscores:
///   `input_blocks.4.1.transformer_blocks.0.attn1.to_q` (LDM dotted)
///   →  `lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q` (kohya).
/// Reversing this mapping is ambiguous in general (`to_out_0` could be
/// `to_out.0` or `to.out.0`), so we do a forward scan of base weight keys
/// and build the reverse table once. Keys not present in base are simply
/// skipped at merge time.
fn build_kohya_unet_table(base: &HashMap<String, Tensor>) -> HashMap<String, String> {
    let mut t = HashMap::new();
    for k in base.keys() {
        let Some(module) = k.strip_suffix(".weight") else { continue };
        let kohya = module.replace('.', "_");
        t.insert(format!("lora_unet_{kohya}"), k.clone());
    }
    t
}

/// Rewrite a kohya UNet prefix from diffusers naming to LDM naming for SDXL.
///
/// Most SDXL LoRAs in the wild (sd-scripts, OneTrainer, ai-toolkit) ship
/// with diffusers-style block names (`down_blocks/up_blocks/mid_block`,
/// `attentions/resnets/upsamplers`, ResBlock submodules `norm1/conv1/...`).
/// Our SDXL UNet checkpoint uses LDM naming (`input_blocks/output_blocks/
/// middle_block`, ResBlock submodules `in_layers/out_layers/...`). This
/// function converts the prefix; the kohya reverse-lookup table then
/// matches the LDM base key.
///
/// SDXL block-index mapping (canonical, see e.g. diffusers'
/// `convert_unet_state_dict_to_sd`):
///
/// | diffusers                            | LDM                       |
/// |--------------------------------------|---------------------------|
/// | down_blocks.0.resnets.{0,1}          | input_blocks.{1,2}.0      |
/// | down_blocks.0.downsamplers.0         | input_blocks.3.0          |
/// | down_blocks.1.{resnets,attentions}.{0,1} | input_blocks.{4,5}.{0,1} |
/// | down_blocks.1.downsamplers.0         | input_blocks.6.0          |
/// | down_blocks.2.{resnets,attentions}.{0,1} | input_blocks.{7,8}.{0,1} |
/// | mid_block.resnets.0                  | middle_block.0            |
/// | mid_block.attentions.0               | middle_block.1            |
/// | mid_block.resnets.1                  | middle_block.2            |
/// | up_blocks.0.{resnets,attentions}.{0,1,2} | output_blocks.{0,1,2}.{0,1} |
/// | up_blocks.0.upsamplers.0             | output_blocks.2.2         |
/// | up_blocks.1.{resnets,attentions}.{0,1,2} | output_blocks.{3,4,5}.{0,1} |
/// | up_blocks.1.upsamplers.0             | output_blocks.5.2         |
/// | up_blocks.2.resnets.{0,1,2}          | output_blocks.{6,7,8}.0   |
///
/// Within a ResBlock, diffusers→LDM submodule rename:
///   norm1→in_layers_0, conv1→in_layers_2, time_emb_proj→emb_layers_1,
///   norm2→out_layers_0, conv2→out_layers_3, conv_shortcut→skip_connection.
///
/// Other root paths:
///   conv_in→input_blocks_0_0, time_embedding.linear_(1,2)→time_embed.(0,2),
///   add_embedding.linear_(1,2)→label_emb.0.(0,2),
///   conv_norm_out→out.0, conv_out→out.2.
fn rewrite_kohya_diffusers_to_ldm(prefix: &str) -> Option<String> {
    let suffix = prefix.strip_prefix("lora_unet_")?;

    // ---- top-level standalone modules ----
    if suffix == "conv_in" {
        return Some("lora_unet_input_blocks_0_0".into());
    }
    if suffix == "conv_norm_out" {
        return Some("lora_unet_out_0".into());
    }
    if suffix == "conv_out" {
        return Some("lora_unet_out_2".into());
    }
    if let Some(n) = suffix.strip_prefix("time_embedding_linear_") {
        let m = match n {
            "1" => "0",
            "2" => "2",
            _ => return None,
        };
        return Some(format!("lora_unet_time_embed_{m}"));
    }
    if let Some(n) = suffix.strip_prefix("add_embedding_linear_") {
        let m = match n {
            "1" => "0",
            "2" => "2",
            _ => return None,
        };
        return Some(format!("lora_unet_label_emb_0_{m}"));
    }

    // ---- helpers ----
    fn rewrite_resblock_submodule(rest: &str) -> String {
        // rest is the part AFTER the resblock identifier, with leading underscore.
        // Map diffusers ResBlock submodule names to LDM equivalents.
        rest.replacen("_norm1_", "_in_layers_0_", 1)
            .replacen("_conv1_", "_in_layers_2_", 1)
            .replacen("_time_emb_proj_", "_emb_layers_1_", 1)
            .replacen("_norm2_", "_out_layers_0_", 1)
            .replacen("_conv2_", "_out_layers_3_", 1)
            .replacen("_conv_shortcut_", "_skip_connection_", 1)
            // Trailing forms (when the submodule terminates the prefix without a
            // following underscore — e.g. for ResBlock's `.lora_down.weight`
            // suffix on `time_emb_proj` itself, the rest is just the leaf name).
            .replacen("_norm1", "_in_layers_0", 1)
            .replacen("_conv1", "_in_layers_2", 1)
            .replacen("_time_emb_proj", "_emb_layers_1", 1)
            .replacen("_norm2", "_out_layers_0", 1)
            .replacen("_conv2", "_out_layers_3", 1)
            .replacen("_conv_shortcut", "_skip_connection", 1)
    }

    // ---- down_blocks ----
    if let Some(rest) = suffix.strip_prefix("down_blocks_") {
        // rest like "0_resnets_0_..." or "1_attentions_0_..." or "0_downsamplers_0_..."
        let mut parts = rest.splitn(3, '_'); // ["0", "resnets", "0_..."] or "0_downsamplers_0..."
        let blk = parts.next()?; // "0", "1", or "2"
        let kind = parts.next()?; // "resnets" | "attentions" | "downsamplers"
        let tail = parts.next().unwrap_or("");
        // tail may start with the index, e.g. "0_norm1_..." or "0_transformer_blocks_..."
        let mut tail_iter = tail.splitn(2, '_');
        let idx = tail_iter.next()?; // "0", "1", "2"
        let rest_after_idx = tail_iter.next().map(|s| format!("_{s}")).unwrap_or_default();
        let (input_idx, sub_idx) = match (blk, kind, idx) {
            ("0", "resnets", "0") => ("1", "0"),
            ("0", "resnets", "1") => ("2", "0"),
            ("0", "downsamplers", "0") => ("3", "0"),
            ("1", "resnets", "0") => ("4", "0"),
            ("1", "attentions", "0") => ("4", "1"),
            ("1", "resnets", "1") => ("5", "0"),
            ("1", "attentions", "1") => ("5", "1"),
            ("1", "downsamplers", "0") => ("6", "0"),
            ("2", "resnets", "0") => ("7", "0"),
            ("2", "attentions", "0") => ("7", "1"),
            ("2", "resnets", "1") => ("8", "0"),
            ("2", "attentions", "1") => ("8", "1"),
            _ => return None,
        };
        let rest_after_idx = if kind == "resnets" {
            rewrite_resblock_submodule(&rest_after_idx)
        } else if kind == "downsamplers" {
            // downsamplers.0 (with .conv inside in diffusers) → input_blocks.N.0.op
            rest_after_idx.replacen("_conv", "_op", 1)
        } else {
            // attentions: keep submodule name as-is (proj_in, proj_out,
            // transformer_blocks.X.attn{1,2}.to_{q,k,v,out_0}, etc.)
            rest_after_idx
        };
        return Some(format!("lora_unet_input_blocks_{input_idx}_{sub_idx}{rest_after_idx}"));
    }

    // ---- mid_block ----
    if let Some(rest) = suffix.strip_prefix("mid_block_") {
        let mut parts = rest.splitn(3, '_');
        let kind = parts.next()?; // "resnets" | "attentions"
        let idx = parts.next()?; // "0" | "1"
        let tail = parts.next().map(|s| format!("_{s}")).unwrap_or_default();
        let mid_idx = match (kind, idx) {
            ("resnets", "0") => "0",
            ("attentions", "0") => "1",
            ("resnets", "1") => "2",
            _ => return None,
        };
        let tail = if kind == "resnets" {
            rewrite_resblock_submodule(&tail)
        } else {
            tail
        };
        return Some(format!("lora_unet_middle_block_{mid_idx}{tail}"));
    }

    // ---- up_blocks ----
    if let Some(rest) = suffix.strip_prefix("up_blocks_") {
        let mut parts = rest.splitn(3, '_');
        let blk = parts.next()?; // "0", "1", "2"
        let kind = parts.next()?; // "resnets" | "attentions" | "upsamplers"
        let tail = parts.next().unwrap_or("");
        let mut tail_iter = tail.splitn(2, '_');
        let idx = tail_iter.next()?;
        let rest_after_idx = tail_iter.next().map(|s| format!("_{s}")).unwrap_or_default();
        let (out_idx, sub_idx) = match (blk, kind, idx) {
            ("0", "resnets", "0") => ("0", "0"),
            ("0", "attentions", "0") => ("0", "1"),
            ("0", "resnets", "1") => ("1", "0"),
            ("0", "attentions", "1") => ("1", "1"),
            ("0", "resnets", "2") => ("2", "0"),
            ("0", "attentions", "2") => ("2", "1"),
            ("0", "upsamplers", "0") => ("2", "2"),
            ("1", "resnets", "0") => ("3", "0"),
            ("1", "attentions", "0") => ("3", "1"),
            ("1", "resnets", "1") => ("4", "0"),
            ("1", "attentions", "1") => ("4", "1"),
            ("1", "resnets", "2") => ("5", "0"),
            ("1", "attentions", "2") => ("5", "1"),
            ("1", "upsamplers", "0") => ("5", "2"),
            ("2", "resnets", "0") => ("6", "0"),
            ("2", "resnets", "1") => ("7", "0"),
            ("2", "resnets", "2") => ("8", "0"),
            _ => return None,
        };
        let rest_after_idx = if kind == "resnets" {
            rewrite_resblock_submodule(&rest_after_idx)
        } else if kind == "upsamplers" {
            // upsamplers.0.conv → output_blocks.<n>.<sub>.conv (LDM
            // wraps the conv inside an Upsample module that forwards
            // to .conv).
            rest_after_idx
        } else {
            rest_after_idx
        };
        return Some(format!("lora_unet_output_blocks_{out_idx}_{sub_idx}{rest_after_idx}"));
    }

    None
}

/// Z-Image trainer prefix → base key + slot.
///
/// Z-Image trainer LoRA targets the main `layers.<i>.*` blocks only:
///
/// | LoRA prefix                                  | Base key                                  | Slot                       |
/// |----------------------------------------------|-------------------------------------------|----------------------------|
/// | `layers.<i>.attention.to_q`                  | `layers.<i>.attention.qkv.weight`         | rows[0..3840]              |
/// | `layers.<i>.attention.to_k`                  | `layers.<i>.attention.qkv.weight`         | rows[3840..7680]           |
/// | `layers.<i>.attention.to_v`                  | `layers.<i>.attention.qkv.weight`         | rows[7680..11520]          |
/// | `layers.<i>.attention.out`                   | `layers.<i>.attention.out.weight`         | full overlay               |
/// | `layers.<i>.feed_forward.w{1,2,3}`           | `layers.<i>.feed_forward.w{1,2,3}.weight` | full overlay               |
fn map_prefix_zimage_trainer(prefix: &str) -> Option<(String, Slot)> {
    if let Some(rest) = prefix.strip_suffix(".attention.to_q") {
        return Some((
            format!("{rest}.attention.qkv.weight"),
            Slot::RowRange { start: 0, len: ZIMAGE_DIM },
        ));
    }
    if let Some(rest) = prefix.strip_suffix(".attention.to_k") {
        return Some((
            format!("{rest}.attention.qkv.weight"),
            Slot::RowRange { start: ZIMAGE_DIM, len: ZIMAGE_DIM },
        ));
    }
    if let Some(rest) = prefix.strip_suffix(".attention.to_v") {
        return Some((
            format!("{rest}.attention.qkv.weight"),
            Slot::RowRange { start: 2 * ZIMAGE_DIM, len: ZIMAGE_DIM },
        ));
    }
    if let Some(rest) = prefix.strip_suffix(".attention.out") {
        return Some((format!("{rest}.attention.out.weight"), Slot::Full));
    }
    if prefix.ends_with(".feed_forward.w1")
        || prefix.ends_with(".feed_forward.w2")
        || prefix.ends_with(".feed_forward.w3")
    {
        return Some((format!("{prefix}.weight"), Slot::Full));
    }
    None
}

/// Merge a Klein LoRA into a base weight dict in-place.
///
/// `alpha / rank` is the scale factor (default training: alpha=16, rank=16
/// → scale=1.0). Pass `multiplier` as a runtime knob to dial the LoRA
/// strength up/down without retraining (multiplier=1.0 = trained strength,
/// 0.0 = base only, 2.0 = double effect).
///
/// Returns the number of modules merged.
pub fn merge_klein_lora(
    base: &mut HashMap<String, Tensor>,
    lora: &HashMap<String, Tensor>,
    alpha: f32,
    rank: usize,
    multiplier: f32,
) -> Result<usize> {
    if rank == 0 {
        return Err(Error::InvalidInput("rank must be > 0".into()));
    }
    let scale = (alpha / rank as f32) * multiplier;
    let format = detect_format(lora);
    eprintln!("[lora] detected format: {format:?}");

    // Per-format suffixes for keying lora_A/lora_B pairs.
    let (suffix_a, suffix_b) = match format {
        LoraFormat::KleinTrainer | LoraFormat::ZImageTrainer => (".lora_A", ".lora_B"),
        LoraFormat::AiToolkit => (".lora_A.weight", ".lora_B.weight"),
        LoraFormat::KohyaSdxl => (".lora_down.weight", ".lora_up.weight"),
    };

    // For kohya we resolve LoRA prefixes against the actual base key set.
    let kohya_table = if matches!(format, LoraFormat::KohyaSdxl) {
        Some(build_kohya_unet_table(base))
    } else {
        None
    };

    // Index LoRA pairs by prefix.
    let mut prefixes: BTreeSet<String> = BTreeSet::new();
    for k in lora.keys() {
        if let Some(p) = k.strip_suffix(suffix_a) {
            prefixes.insert(p.to_string());
        }
    }
    let mut n_skipped_te = 0usize;

    let mut n_merged = 0usize;
    let mut n_skipped_unknown = 0usize;
    let mut n_skipped_bridge = 0usize;

    for prefix in &prefixes {
        let a = lora
            .get(&format!("{prefix}{suffix_a}"))
            .ok_or_else(|| Error::InvalidInput(format!("missing {prefix}{suffix_a}")))?;
        let Some(b) = lora.get(&format!("{prefix}{suffix_b}")) else {
            // Pair half-missing — log and continue.
            eprintln!("[lora] {prefix}{suffix_b} missing, skipping");
            continue;
        };

        // Map to base key + slot.
        let mapped = match format {
            LoraFormat::KleinTrainer => map_prefix_klein_trainer(prefix),
            LoraFormat::ZImageTrainer => map_prefix_zimage_trainer(prefix),
            LoraFormat::AiToolkit => map_prefix_aitoolkit(prefix),
            LoraFormat::KohyaSdxl => {
                if prefix.starts_with("lora_te1_") || prefix.starts_with("lora_te2_") {
                    // Text-encoder LoRAs aren't merged into the UNet base;
                    // the encoder pass would need separate adaptation.
                    n_skipped_te += 1;
                    continue;
                }
                let table = kohya_table.as_ref();
                // Try direct (LDM-named) lookup first.
                let direct = table.and_then(|t| t.get(prefix));
                if let Some(bk) = direct {
                    Some((bk.clone(), Slot::Full))
                } else {
                    // Fall back: rewrite diffusers naming → LDM, retry.
                    rewrite_kohya_diffusers_to_ldm(prefix)
                        .and_then(|ldm_prefix| {
                            table.and_then(|t| t.get(&ldm_prefix)).cloned()
                        })
                        .map(|bk| (bk, Slot::Full))
                }
            }
        };
        let Some((bkey, slot)) = mapped else {
            if prefix.starts_with("input_bridges.") {
                n_skipped_bridge += 1;
            } else {
                eprintln!("[lora] unknown prefix '{prefix}'");
                n_skipped_unknown += 1;
            }
            continue;
        };

        // Per-module alpha override (kohya only). `<prefix>.alpha` is a
        // SCALAR tensor that overrides the global alpha for that module:
        //   delta = (alpha_module / rank) * multiplier * (B @ A)
        // Kohya's lora_down has shape [rank, in], so rank is `a.shape()[0]`.
        let module_scale = if matches!(format, LoraFormat::KohyaSdxl) {
            let alpha_key = format!("{prefix}.alpha");
            match lora.get(&alpha_key) {
                Some(t) => {
                    // Scalar tensor (shape [] in safetensors) — read its single value.
                    let v = t.to_dtype(DType::F32)?.to_vec()?;
                    let alpha_module = v.first().copied().unwrap_or(rank as f32);
                    let module_rank = a.shape().dims()[0];
                    (alpha_module / module_rank as f32) * multiplier
                }
                None => scale,
            }
        } else {
            scale
        };

        // Compute delta = module_scale * (B @ A). Two cases:
        //
        // Linear LoRA: A is [rank, in], B is [out, rank], delta is [out, in].
        // Conv2D LoRA (kohya): A is [rank, ic, kh, kw], B is [oc, rank, 1, 1].
        //   Reshape: A → [rank, ic*kh*kw], B → [oc, rank]; matmul →
        //   [oc, ic*kh*kw]; reshape back to [oc, ic, kh, kw] for the conv merge.
        //   This matches kohya's `LoRAModule.merge_to` for `LoRAModuleConv2d`.
        let delta_native = if a.shape().dims().len() == 4 || b.shape().dims().len() == 4 {
            let a_dims = a.shape().dims().to_vec();
            let b_dims = b.shape().dims().to_vec();
            if a_dims.len() != 4 || b_dims.len() != 4 {
                eprintln!(
                    "[lora] conv LoRA needs both 4D on '{prefix}': A={a_dims:?} B={b_dims:?}"
                );
                continue;
            }
            let (r_a, ic, kh, kw) = (a_dims[0], a_dims[1], a_dims[2], a_dims[3]);
            let (oc, r_b) = (b_dims[0], b_dims[1]);
            if r_a != r_b {
                eprintln!(
                    "[lora] conv LoRA rank mismatch on '{prefix}': A rank={r_a} B rank={r_b}"
                );
                continue;
            }
            let a_2d = a.reshape(&[r_a, ic * kh * kw])?;
            let b_2d = b.reshape(&[oc, r_b])?;
            let delta_2d = b_2d.matmul(&a_2d)?.mul_scalar(module_scale)?;
            delta_2d.reshape(&[oc, ic, kh, kw])?
        } else {
            b.matmul(a)?.mul_scalar(module_scale)?
        };
        let delta_dims = delta_native.shape().dims().to_vec();

        let Some(base_w) = base.get(&bkey) else {
            eprintln!("[lora] base key missing for '{prefix}': '{bkey}'");
            continue;
        };

        let base_dtype = base_w.dtype();
        let base_dims = base_w.shape().dims().to_vec();
        let delta = if delta_native.dtype() == base_dtype {
            delta_native
        } else {
            delta_native.to_dtype(base_dtype)?
        };

        let merged = match slot {
            Slot::Full => {
                if base_dims != delta_dims {
                    eprintln!(
                        "[lora] shape mismatch on '{bkey}': base {base_dims:?} vs delta {delta_dims:?}"
                    );
                    continue;
                }
                base_w.add(&delta)?
            }
            Slot::Rows(n) => {
                if base_dims.len() != 2
                    || delta_dims != vec![n, base_dims[1]]
                    || base_dims[0] < n
                {
                    eprintln!(
                        "[lora] row-merge shape mismatch on '{bkey}': base {base_dims:?} vs delta {delta_dims:?} (rows={n})"
                    );
                    continue;
                }
                let top = base_w.narrow(0, 0, n)?.contiguous()?;
                let bottom = base_w.narrow(0, n, base_dims[0] - n)?.contiguous()?;
                let top_merged = top.add(&delta)?;
                Tensor::cat(&[&top_merged, &bottom], 0)?
            }
            Slot::Cols(n) => {
                if base_dims.len() != 2
                    || delta_dims != vec![base_dims[0], n]
                    || base_dims[1] < n
                {
                    eprintln!(
                        "[lora] col-merge shape mismatch on '{bkey}': base {base_dims:?} vs delta {delta_dims:?} (cols={n})"
                    );
                    continue;
                }
                let left = base_w.narrow(1, 0, n)?.contiguous()?;
                let right = base_w.narrow(1, n, base_dims[1] - n)?.contiguous()?;
                let left_merged = left.add(&delta)?;
                Tensor::cat(&[&left_merged, &right], 1)?
            }
            Slot::RowRange { start, len } => {
                if base_dims.len() != 2
                    || delta_dims != vec![len, base_dims[1]]
                    || start + len > base_dims[0]
                {
                    eprintln!(
                        "[lora] row-range shape mismatch on '{bkey}': base {base_dims:?} vs delta {delta_dims:?} (start={start} len={len})"
                    );
                    continue;
                }
                // Split base into [head | mid (the merge target) | tail]
                // and rejoin after adding delta into mid.
                let head_len = start;
                let tail_len = base_dims[0] - start - len;
                let mut parts: Vec<Tensor> = Vec::with_capacity(3);
                if head_len > 0 {
                    parts.push(base_w.narrow(0, 0, head_len)?.contiguous()?);
                }
                let mid = base_w.narrow(0, start, len)?.contiguous()?;
                let mid_merged = mid.add(&delta)?;
                parts.push(mid_merged);
                if tail_len > 0 {
                    parts.push(base_w.narrow(0, start + len, tail_len)?.contiguous()?);
                }
                let part_refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&part_refs, 0)?
            }
        };

        let merged_native = if merged.dtype() == base_dtype {
            merged.contiguous()?
        } else {
            merged.to_dtype(base_dtype)?.contiguous()?
        };
        base.insert(bkey, merged_native);
        n_merged += 1;

        // Trim every 16 modules to keep the cuda mempool from holding
        // 9B-scale BF16 scratch (delta + merged) across the whole loop.
        if n_merged % 16 == 0 {
            trim_cuda_mempool(0);
        }
    }

    if n_skipped_bridge > 0 {
        eprintln!("[lora] skipped {n_skipped_bridge} input_bridges entries (trainer-only, not in base)");
    }
    if n_skipped_unknown > 0 {
        eprintln!("[lora] skipped {n_skipped_unknown} unknown prefixes");
    }
    if n_skipped_te > 0 {
        eprintln!(
            "[lora] skipped {n_skipped_te} text-encoder (lora_te1_/lora_te2_) modules — not merged into UNet"
        );
    }

    Ok(n_merged)
}
