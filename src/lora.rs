//! Runtime LoRA application for inference — the canonical "wrap forward, never
//! mutate base" pattern used by ai-toolkit, OneTrainer, and musubi-tuner.
//!
//! # Why this exists
//!
//! Every reference trainer applies a LoRA at sampling time as
//! `output = base_forward(x) + scale * lora_up(lora_down(x))` —
//! the base linear's weights are never touched. The LoRA branch is computed
//! fresh per forward pass. This preserves base precision (no BF16
//! accumulation through repeated merges), allows `multiplier` to toggle the
//! LoRA strength at zero cost, and keeps the dataflow obviously correct
//! when split-QKV LoRAs target a fused base matrix (each split's branch
//! contributes to its own output rows; no `Slot::RowRange` weight slicing).
//!
//! Our previous `lora_merge.rs` did the opposite — mutated base in place —
//! which (a) compounded BF16 precision loss, (b) made `Slot::RowRange`
//! boundary bugs invisibly corrupting, (c) silently dropped per-module
//! `.alpha` for 3 of 4 formats. That module is being deleted in favor of
//! this one.
//!
//! # Recipe for adding a new model
//!
//! Each model has a "linear chokepoint" — a single function that runs
//! `out = matmul(weight, x)` for every weight key. Examples:
//!
//! - `models/zimage_nextdit.rs::linear_no_bias`
//! - `models/klein.rs` — single/double block linear forwards
//! - `models/sd3_mmdit.rs` — context_block / x_block linears
//! - `models/sdxl_unet.rs::Linear::forward`
//! - `models/ernie_image.rs` — DiT block linears
//!
//! To wire LoRA support into a model:
//!
//! 1. Add an `Option<Arc<LoraStack>>` field to the model struct.
//! 2. After the base matmul in the chokepoint, call
//!    `lora.apply(weight_key, x, base_out)` and use the returned tensor.
//! 3. In the inference binary, build the stack via
//!    `LoraStack::load(path, base_key_set, multiplier, device)` and pass
//!    it into the model constructor (e.g. `Model::new_with_lora`).
//!
//! That's the entire integration. No model-internal LoRA logic, no merge,
//! no slot juggling at the model layer — `LoraStack` handles all of it.
//!
//! # Format support
//!
//! - **KleinTrainer**: bare `<prefix>.lora_A` / `.lora_B`, custom
//!   projection names (`*_proj`). Single-block linears need row/col slot
//!   handling for Klein-4B (9216, 3072 constants).
//! - **ZImageTrainer**: bare `<prefix>.lora_A` / `.lora_B` with split
//!   `attention.to_q/to_k/to_v` → fused `attention.qkv.weight` via
//!   `Slot::RowRange`.
//! - **AiToolkit**: `diffusion_model.<key>.lora_A.weight`, full overlay.
//!   May ship per-module `.alpha` scalar tensors.
//! - **KohyaSdxl**: `lora_unet_<path>.lora_down/up.weight`. Always ships
//!   per-module `.alpha`. SDXL diffusers names are rewritten to LDM via
//!   the helpers in `kohya_naming` below.
//!
//! # Per-module alpha
//!
//! All four formats can ship a per-module `<prefix>.alpha` scalar tensor.
//! `LoraStack::load` reads it for *every* format (the merge module read it
//! only for kohya, which was a real bug — a musubi-trained Z-Image LoRA
//! with `alpha=3, rank=96` would otherwise be applied at scale=1.0
//! instead of 0.0312, i.e. 32× too strong).
//!
//! # Outstanding work
//!
//! Models that don't yet have a `*_lora_infer` binary (when those are
//! added, just follow the recipe above):
//!
//! - chroma (chroma_dit.rs)
//! - qwenimage (qwenimage_dit.rs)
//! - anima (anima.rs)
//! - hunyuan15 (hunyuan15_dit.rs)
//! - kandinsky5 (kandinsky5_dit.rs)
//! - wan22 / wan_vace (wan22_dit.rs, wan_vace_dit.rs)
//! - flux1 (flux1_dit.rs) — has a trainer but no inference binary
//! - acestep (acestep_dit.rs)
//! - sd15 (sd15_unet.rs)
//! - ltx2 (ltx2_model.rs) — currently uses its own `lora_loader.rs` with
//!   the same merge-in-place architectural problem; migrating it will
//!   require porting the existing fuse semantics.

use flame_core::{trim_cuda_mempool, DType, Error, Result, Tensor};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

/// Z-Image fused QKV is `[3*dim, dim]` with Q/K/V stacked along dim 0.
/// For dim=3840 the per-head ranges are 0..3840, 3840..7680, 7680..11520.
const ZIMAGE_DIM: usize = 3840;
/// Klein-4B single-block `linear1` rows targeted by `qkv_proj` LoRAs.
const KLEIN_4B_SINGLE_QKV_ROWS: usize = 9216;
/// Klein-4B single-block `linear2` columns targeted by `out_proj` LoRAs.
const KLEIN_4B_SINGLE_OUT_COLS: usize = 3072;

/// Where in the base linear's output the LoRA branch's contribution lands.
#[derive(Clone, Copy, Debug)]
pub enum Slot {
    /// LoRA branch output has the same shape as base output. Add directly.
    Full,
    /// Top `n` rows of base output get the delta (delta shape is
    /// `[..., n]`). Used for Klein single-block `linear1` (`qkv_proj`).
    Rows(usize),
    /// Left `n` cols of base input feed the LoRA's down matrix. Branch
    /// output has the full base output shape. Used for Klein single-block
    /// `linear2` (`out_proj`).
    Cols(usize),
    /// Output rows `[start..start+len]` of base output get the delta.
    /// Used for Z-Image trainer's split-Q/K/V → fused QKV.
    RowRange { start: usize, len: usize },
}

/// LoRA file naming convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraFormat {
    /// Klein-trainer: `<prefix>.lora_A` / `.lora_B`, projection names like
    /// `qkv_proj` / `out_proj`. Klein-4B single blocks need slot slicing.
    KleinTrainer,
    /// Z-Image trainer: `<prefix>.lora_A` / `.lora_B` with split QKV
    /// (`attention.to_q/to_k/to_v`).
    ZImageTrainer,
    /// ai-toolkit: `diffusion_model.<key>.lora_A.weight`. Full overlay.
    AiToolkit,
    /// kohya / sd-scripts: `lora_unet_<path>.lora_down/up.weight` plus
    /// per-module `<prefix>.alpha` scalar.
    KohyaSdxl,
}

/// One concrete LoRA branch ready to be applied.
struct LoraEntry {
    slot: Slot,
    /// `lora_A` transposed: `[in_dim_or_slice, rank]`.
    down_t: Tensor,
    /// `lora_B` transposed: `[rank, out_dim_or_slice]`.
    up_t: Tensor,
    /// `(alpha / rank) * multiplier` — applied as a scalar after up.
    scale: f32,
}

/// All LoRA branches for one inference run, indexed by base weight key.
pub struct LoraStack {
    entries: HashMap<String, Vec<LoraEntry>>,
    format: LoraFormat,
}

impl LoraStack {
    /// Load a LoRA file and prepare runtime entries.
    ///
    /// `base_keys` is the set of weight keys present in the base model
    /// (used for kohya naming resolution). `multiplier` is the runtime
    /// strength dial (1.0 = trained strength).
    ///
    /// Returns `(stack, n_skipped)` so callers can log how many entries
    /// went unmapped.
    pub fn load(
        path: &str,
        base_keys: &HashSet<String>,
        multiplier: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let lora = flame_core::serialization::load_file(path, device)?;
        Self::from_tensors(&lora, base_keys, multiplier)
    }

    /// Build a stack from already-loaded LoRA tensors. (Useful when the
    /// caller has already loaded the file for inspection.)
    pub fn from_tensors(
        lora: &HashMap<String, Tensor>,
        base_keys: &HashSet<String>,
        multiplier: f32,
    ) -> Result<Self> {
        let format = detect_format(lora);
        eprintln!("[lora] detected format: {format:?}");

        let (suffix_a, suffix_b) = match format {
            LoraFormat::KleinTrainer | LoraFormat::ZImageTrainer => (".lora_A", ".lora_B"),
            LoraFormat::AiToolkit => (".lora_A.weight", ".lora_B.weight"),
            LoraFormat::KohyaSdxl => (".lora_down.weight", ".lora_up.weight"),
        };

        let kohya_table = if matches!(format, LoraFormat::KohyaSdxl) {
            Some(build_kohya_unet_table(base_keys))
        } else {
            None
        };

        // Index LoRA prefixes.
        let mut prefixes: BTreeSet<String> = BTreeSet::new();
        for k in lora.keys() {
            if let Some(p) = k.strip_suffix(suffix_a) {
                prefixes.insert(p.to_string());
            }
        }

        let mut entries: HashMap<String, Vec<LoraEntry>> = HashMap::new();
        let mut n_loaded = 0usize;
        let mut n_skipped_te = 0usize;
        let mut n_skipped_unknown = 0usize;
        let mut n_skipped_bridge = 0usize;

        for prefix in &prefixes {
            let a = lora
                .get(&format!("{prefix}{suffix_a}"))
                .ok_or_else(|| Error::InvalidInput(format!("missing {prefix}{suffix_a}")))?;
            let Some(b) = lora.get(&format!("{prefix}{suffix_b}")) else {
                eprintln!("[lora] {prefix}{suffix_b} missing, skipping");
                continue;
            };

            // Format-specific prefix → (base_key, slot).
            let mapped = match format {
                LoraFormat::KleinTrainer => map_prefix_klein_trainer(prefix),
                LoraFormat::ZImageTrainer => map_prefix_zimage_trainer(prefix),
                LoraFormat::AiToolkit => map_prefix_aitoolkit(prefix),
                LoraFormat::KohyaSdxl => {
                    if prefix.starts_with("lora_te1_") || prefix.starts_with("lora_te2_") {
                        n_skipped_te += 1;
                        continue;
                    }
                    let table = kohya_table.as_ref();
                    let direct = table.and_then(|t| t.get(prefix));
                    if let Some(bk) = direct {
                        Some((bk.clone(), Slot::Full))
                    } else {
                        kohya_naming::rewrite_diffusers_to_ldm(prefix)
                            .and_then(|ldm| table.and_then(|t| t.get(&ldm)).cloned())
                            .map(|bk| (bk, Slot::Full))
                    }
                }
            };

            let Some((base_key, slot)) = mapped else {
                if prefix.starts_with("input_bridges.") {
                    n_skipped_bridge += 1;
                } else {
                    eprintln!("[lora] unknown prefix '{prefix}'");
                    n_skipped_unknown += 1;
                }
                continue;
            };

            // Per-module .alpha (read for ALL formats — the merge code only
            // did this for kohya, which silently mis-scaled musubi-trained
            // and ai-toolkit-trained LoRAs by `rank/saved_alpha` (often
            // 32× or so for rank=96 alpha=3 LoRAs).
            let alpha_key = format!("{prefix}.alpha");
            let module_rank = a.shape().dims()[0];
            let alpha_value: f32 = match lora.get(&alpha_key) {
                Some(t) => {
                    let v = t.to_dtype(DType::F32)?.to_vec()?;
                    v.first().copied().unwrap_or(module_rank as f32)
                }
                None => module_rank as f32, // alpha=rank → scale=1.0 (matches our trainers' default).
            };
            let scale = (alpha_value / module_rank as f32) * multiplier;

            // Conv2D LoRAs (kohya only): a is [rank, ic, kh, kw],
            // b is [oc, rank, 1, 1]. We don't yet have an inference path
            // that hits conv weights through this stack — the SDXL conv-LoRA
            // case wasn't exercised. Skip with a log so it's visible if a
            // user hits this and needs us to add support.
            if a.shape().dims().len() == 4 || b.shape().dims().len() == 4 {
                eprintln!(
                    "[lora] skipping conv LoRA on '{prefix}' (4D weights not yet supported in runtime path)"
                );
                continue;
            }

            // Pre-transpose down/up so the runtime path is just
            // `(x @ down_t) @ up_t * scale`. Match the trainer's
            // forward_delta layout (a_t = A^T = [in, rank], b_t = B^T = [rank, out]).
            //
            // Upcast to F32. The reference trainers (ai-toolkit, OneTrainer,
            // musubi) compute the LoRA branch in F32 and cast only the final
            // delta to base dtype before add. Doing the chained matmuls in
            // BF16 across 150 modules × N steps blurs the contribution into
            // featureless mush — the soft, identity-less output we saw with
            // BF16 throughout. F32 here costs ~2× the LoRA-branch memory
            // and matches reference numerics.
            let down_t = a.to_dtype(DType::F32)?.transpose()?.contiguous()?;
            let up_t = b.to_dtype(DType::F32)?.transpose()?.contiguous()?;

            entries.entry(base_key).or_default().push(LoraEntry {
                slot,
                down_t,
                up_t,
                scale,
            });
            n_loaded += 1;

            // Trim mempool periodically to avoid scratch buildup during load.
            if n_loaded % 32 == 0 {
                trim_cuda_mempool(0);
            }
        }

        let n_targets = entries.len();
        eprintln!(
            "[lora] loaded {n_loaded} entries → {n_targets} target weight(s){}",
            multiplier_note(multiplier),
        );
        if n_skipped_bridge > 0 {
            eprintln!("[lora] skipped {n_skipped_bridge} input_bridges entries (trainer-only)");
        }
        if n_skipped_unknown > 0 {
            eprintln!("[lora] skipped {n_skipped_unknown} unknown prefixes");
        }
        if n_skipped_te > 0 {
            eprintln!(
                "[lora] skipped {n_skipped_te} text-encoder (lora_te1_/lora_te2_) modules — not applied at UNet"
            );
        }

        Ok(LoraStack { entries, format })
    }

    /// Number of distinct base weight keys that have at least one LoRA branch.
    pub fn target_count(&self) -> usize {
        self.entries.len()
    }

    pub fn format(&self) -> LoraFormat {
        self.format
    }

    /// Apply LoRA contributions for `weight_key` to `base_out`.
    ///
    /// `x` is the input tensor that fed the base linear (so the LoRA branch
    /// can recompute `up(down(x))`). `base_out` is the linear's output. If
    /// no LoRA targets `weight_key`, returns `base_out` unchanged.
    ///
    /// `x` and `base_out` are assumed to be the trailing-feature-dim
    /// arrangement: `x` shape `[..., in_features]`, `base_out` shape
    /// `[..., out_features]`. Higher-rank tensors are flattened to 2D
    /// internally and reshaped back.
    pub fn apply(&self, weight_key: &str, x: &Tensor, base_out: Tensor) -> Result<Tensor> {
        let Some(entries) = self.entries.get(weight_key) else {
            return Ok(base_out);
        };

        let base_dtype = base_out.dtype();

        // Flatten x to [B*..., in].
        let x_dims = x.shape().dims().to_vec();
        let in_dim = *x_dims.last().expect("x has rank ≥ 1");
        let flat_rows: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_2d = if x_dims.len() == 2 {
            x.contiguous()?
        } else {
            x.reshape(&[flat_rows, in_dim])?
        };

        // Flatten base_out to [B*..., out].
        let out_dims = base_out.shape().dims().to_vec();
        let out_features = *out_dims.last().expect("base_out has rank ≥ 1");
        let mut acc = if out_dims.len() == 2 {
            base_out
        } else {
            base_out.reshape(&[flat_rows, out_features])?
        };

        for entry in entries {
            // Slice x for Cols slot — LoRA was trained against only the
            // first `n` input features.
            let x_for_lora_owned;
            let x_for_lora: &Tensor = match entry.slot {
                Slot::Cols(n) => {
                    x_for_lora_owned = x_2d.narrow(1, 0, n)?.contiguous()?;
                    &x_for_lora_owned
                }
                _ => &x_2d,
            };

            // Cast input to LoRA dtype (matches trainer's forward_delta which
            // computes in F32). flame_core's matmul refuses mixed dtypes.
            let lora_dtype = entry.down_t.dtype();
            let x_cast_owned;
            let x_cast: &Tensor = if x_for_lora.dtype() == lora_dtype {
                x_for_lora
            } else {
                x_cast_owned = x_for_lora.to_dtype(lora_dtype)?;
                &x_cast_owned
            };

            // (x @ A^T) @ B^T * scale. Force `.contiguous()` after the
            // matmul chain — cuBLAS matmul can produce a non-row-major
            // layout, and downstream BF16 tensor_iterator kernels read
            // saved tensors as if contiguous (the same hazard documented
            // in flame-diffusion/src/lora.rs:223). Without this, mul_scalar
            // and add silently see garbage strides and produce noisy output.
            let mat_out = x_cast.matmul(&entry.down_t)?.matmul(&entry.up_t)?.contiguous()?;
            let delta = mat_out.mul_scalar(entry.scale)?;

            // Cast delta back to base dtype before add (lossy but
            // matches reference implementations).
            let delta = if delta.dtype() == base_dtype {
                delta
            } else {
                delta.to_dtype(base_dtype)?
            };

            acc = match entry.slot {
                Slot::Full | Slot::Cols(_) => acc.add(&delta)?,
                Slot::Rows(n) => {
                    // Add delta to first n cols of acc (output features).
                    add_at_col_range(&acc, &delta, 0, n)?
                }
                Slot::RowRange { start, len } => {
                    add_at_col_range(&acc, &delta, start, len)?
                }
            };
        }

        // Restore original output shape.
        if out_dims.len() == 2 {
            Ok(acc)
        } else {
            acc.reshape(&out_dims)
        }
    }
}

/// Add `delta [rows, len]` into `base [rows, total]` at columns
/// `start..start+len`. Returns the patched tensor.
fn add_at_col_range(base: &Tensor, delta: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let dims = base.shape().dims();
    if dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "add_at_col_range needs 2D base, got {:?}",
            dims
        )));
    }
    let total = dims[1];
    if start + len > total {
        return Err(Error::InvalidInput(format!(
            "add_at_col_range out of range: start={start} len={len} total={total}"
        )));
    }
    let head_len = start;
    let tail_len = total - start - len;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if head_len > 0 {
        parts.push(base.narrow(1, 0, head_len)?.contiguous()?);
    }
    let mid = base.narrow(1, start, len)?.contiguous()?;
    parts.push(mid.add(delta)?);
    if tail_len > 0 {
        parts.push(base.narrow(1, start + len, tail_len)?.contiguous()?);
    }
    let part_refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&part_refs, 1)
}

fn multiplier_note(m: f32) -> String {
    if (m - 1.0).abs() < f32::EPSILON {
        String::new()
    } else {
        format!(" (multiplier={m:.3})")
    }
}

// ─── format detection ────────────────────────────────────────────────────────

fn detect_format(lora: &HashMap<String, Tensor>) -> LoraFormat {
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

// ─── prefix → (base_key, slot) mappers ──────────────────────────────────────
//
// These mirror lora_merge.rs's mappers but stay local to this module so the
// merge file can be deleted.

fn map_prefix_klein_trainer(prefix: &str) -> Option<(String, Slot)> {
    if prefix.starts_with("input_bridges.") {
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
            return Some((
                format!("{rest}.linear1.weight"),
                Slot::Rows(KLEIN_4B_SINGLE_QKV_ROWS),
            ));
        }
        if let Some(rest) = prefix.strip_suffix(".out_proj") {
            return Some((
                format!("{rest}.linear2.weight"),
                Slot::Cols(KLEIN_4B_SINGLE_OUT_COLS),
            ));
        }
    }
    None
}

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

fn map_prefix_aitoolkit(prefix: &str) -> Option<(String, Slot)> {
    let stripped = prefix.strip_prefix("diffusion_model.").unwrap_or(prefix);
    let stripped = stripped.strip_suffix(".default").unwrap_or(stripped);

    // Z-Image-specific: ai-toolkit (and our trainer's new save) writes split
    // attention as `<...>.attention.to_q/.to_k/.to_v/.to_out.0`, but the
    // flame Z-Image base model has fused `<...>.attention.qkv.weight` and
    // `<...>.attention.out.weight`. Mirror the slot mapping in
    // `map_prefix_zimage_trainer` so ai-toolkit-format Z-Image LoRAs route
    // through the same Q/K/V row ranges + `attention.out` as the trainer's
    // legacy split format.
    if let Some(rest) = stripped.strip_suffix(".attention.to_q") {
        return Some((
            format!("{rest}.attention.qkv.weight"),
            Slot::RowRange { start: 0, len: ZIMAGE_DIM },
        ));
    }
    if let Some(rest) = stripped.strip_suffix(".attention.to_k") {
        return Some((
            format!("{rest}.attention.qkv.weight"),
            Slot::RowRange { start: ZIMAGE_DIM, len: ZIMAGE_DIM },
        ));
    }
    if let Some(rest) = stripped.strip_suffix(".attention.to_v") {
        return Some((
            format!("{rest}.attention.qkv.weight"),
            Slot::RowRange { start: 2 * ZIMAGE_DIM, len: ZIMAGE_DIM },
        ));
    }
    if let Some(rest) = stripped.strip_suffix(".attention.to_out.0") {
        return Some((format!("{rest}.attention.out.weight"), Slot::Full));
    }

    // Generic fallback for everything else (FFN, adaLN, non-Z-Image
    // architectures whose attention path doesn't match the patterns above).
    Some((format!("{stripped}.weight"), Slot::Full))
}

fn build_kohya_unet_table(base_keys: &HashSet<String>) -> HashMap<String, String> {
    let mut t = HashMap::new();
    for k in base_keys {
        let Some(module) = k.strip_suffix(".weight") else { continue };
        let kohya = module.replace('.', "_");
        t.insert(format!("lora_unet_{kohya}"), k.clone());
    }
    t
}

// ─── kohya naming → LDM (SDXL) ──────────────────────────────────────────────
//
// Most kohya/sd-scripts/OneTrainer/ai-toolkit SDXL LoRAs ship with diffusers
// block names; our SDXL UNet uses LDM names. The rewriter converts the
// prefix; the build_kohya_unet_table reverse-lookup then matches the LDM key.
// (This block is identical in spirit to lora_merge.rs's rewriter — it lives
// here now so lora_merge.rs can be deleted.)

mod kohya_naming {
    /// Returns the LDM-format kohya prefix, or None if the input prefix
    /// doesn't match any known diffusers pattern.
    pub fn rewrite_diffusers_to_ldm(prefix: &str) -> Option<String> {
        let suffix = prefix.strip_prefix("lora_unet_")?;

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

        if let Some(rest) = suffix.strip_prefix("down_blocks_") {
            return rewrite_down(rest);
        }
        if let Some(rest) = suffix.strip_prefix("mid_block_") {
            return rewrite_mid(rest);
        }
        if let Some(rest) = suffix.strip_prefix("up_blocks_") {
            return rewrite_up(rest);
        }
        None
    }

    fn rewrite_resblock_submodule(rest: &str) -> String {
        rest.replacen("_norm1_", "_in_layers_0_", 1)
            .replacen("_conv1_", "_in_layers_2_", 1)
            .replacen("_time_emb_proj_", "_emb_layers_1_", 1)
            .replacen("_norm2_", "_out_layers_0_", 1)
            .replacen("_conv2_", "_out_layers_3_", 1)
            .replacen("_conv_shortcut_", "_skip_connection_", 1)
            .replacen("_norm1", "_in_layers_0", 1)
            .replacen("_conv1", "_in_layers_2", 1)
            .replacen("_time_emb_proj", "_emb_layers_1", 1)
            .replacen("_norm2", "_out_layers_0", 1)
            .replacen("_conv2", "_out_layers_3", 1)
            .replacen("_conv_shortcut", "_skip_connection", 1)
    }

    fn rewrite_down(rest: &str) -> Option<String> {
        let mut parts = rest.splitn(3, '_');
        let blk = parts.next()?;
        let kind = parts.next()?;
        let tail = parts.next().unwrap_or("");
        let mut tail_iter = tail.splitn(2, '_');
        let idx = tail_iter.next()?;
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
            rest_after_idx.replacen("_conv", "_op", 1)
        } else {
            rest_after_idx
        };
        Some(format!("lora_unet_input_blocks_{input_idx}_{sub_idx}{rest_after_idx}"))
    }

    fn rewrite_mid(rest: &str) -> Option<String> {
        let mut parts = rest.splitn(3, '_');
        let kind = parts.next()?;
        let idx = parts.next()?;
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
        Some(format!("lora_unet_middle_block_{mid_idx}{tail}"))
    }

    fn rewrite_up(rest: &str) -> Option<String> {
        let mut parts = rest.splitn(3, '_');
        let blk = parts.next()?;
        let kind = parts.next()?;
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
            rest_after_idx.replacen("_conv", "_conv", 1)
        } else {
            rest_after_idx
        };
        Some(format!("lora_unet_output_blocks_{out_idx}_{sub_idx}{rest_after_idx}"))
    }
}
