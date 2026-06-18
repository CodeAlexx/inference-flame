//! NAVA (= Ovi joint audio-video MM-DiT) — weight loader for [`WanAVModel`].
//!
//! Reads the BF16 safetensors produced offline by
//! `ports/nava/parity/convert_nava_ckpt.py` (a dev tool — runtime stays pure
//! Rust) and populates the flat `HashMap<String, Tensor>` that
//! [`super::nava_blocks::WanAVModel`] forwards over.
//!
//! ## Key convention (CONTEXT.md: no silent renames)
//! The converter writes keys **verbatim** from the NAVA `state_dict`, i.e. with
//! the `backbone.` prefix (`backbone.double_blocks.0.self_attn.q.weight`, …).
//! The model holds keys with that prefix **stripped** (`double_blocks.0.…`,
//! `head.head.weight`, `patch_embedding.weight`) — see the `w(&self.weights, …)`
//! call sites in `nava_blocks.rs` / `nava_av.rs`. Stripping `backbone.` is the
//! ONE structural transform this loader performs; it is mechanical (drop a fixed
//! prefix every checkpoint key carries), not a rename. Keys that do not carry
//! the prefix are kept as-is.
//!
//! ## Missing / unexpected key handling (integration safety net)
//! Every weight the model will read is enumerated by [`expected_keys`]. After
//! load we check that set against what the file produced:
//!   - a model key with NO source → **hard error** naming the missing key;
//!   - a file key the model never reads → **warn** (kept in the map, not dropped
//!     — e.g. `speaker_embedding.*` / `double_final_blocks.*` exist in the
//!     checkpoint for the unused `add_spk_emb` / dead-block paths).
//!
//! ## Residency
//! `WanAVModel` holds ALL weights resident in a single `HashMap` — it has no
//! `BlockOffloader` field. This loader therefore builds the full resident map
//! (6.3 B params in BF16 ≈ 12.6 GB). BUILD_PLAN item 12 anticipates the gen bin
//! streaming the 30 blocks via `BlockOffloader` to fit 24 GB alongside the VAE;
//! that wiring belongs to the model struct + gen bin, not here. See the
//! `// TODO: BlockOffloader wiring` note in [`load_nava`].
//!
//! ## Dtype
//! The converter already casts every float tensor to BF16. As a belt-and-braces
//! guard (a re-export or a differently-produced file could carry F32), any
//! non-BF16 float tensor is cast to BF16 at load — matching the wan22 loader.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use flame_core::{CudaDevice, DType, Error, Result, Tensor};

use super::nava_av::NavaAVConfig;
use super::nava_blocks::WanAVModel;

/// Prefix every NAVA `state_dict` key carries (mmdit `WanAVModel` path). The
/// model holds keys with this stripped.
const BACKBONE_PREFIX: &str = "backbone.";

/// Per-block QKV/cross-attn projection names. `o` is the output projection.
const PROJ: [&str; 4] = ["q", "k", "v", "o"];

/// Build the set of weight keys the model will read, in the **stripped**
/// (`backbone.`-removed) namespace the `HashMap` is keyed by.
///
/// Mirrors EXACTLY the `w(weights, …)` call sites in `nava_blocks.rs` and
/// `nava_av.rs`. Kept as a standalone fn so the shape/key contract is testable
/// without any weights present (see the unit test).
pub fn expected_keys(cfg: &NavaAVConfig) -> HashSet<String> {
    let mut keys = HashSet::new();

    // --- top-level embeddings / projections / heads ---
    // video Conv3d patch embed
    keys.insert("patch_embedding.weight".into());
    keys.insert("patch_embedding.bias".into());
    // audio Conv1d patch embed (nn.Sequential index 0 conv + index 2 ConvMLP)
    keys.insert("patch_embedding_audio.0.weight".into());
    keys.insert("patch_embedding_audio.0.bias".into());
    keys.insert("patch_embedding_audio.2.w1.weight".into());
    keys.insert("patch_embedding_audio.2.w2.weight".into());
    keys.insert("patch_embedding_audio.2.w3.weight".into());
    // text embedding (Linear, GELU, Linear)
    for idx in ["0", "2"] {
        keys.insert(format!("text_embedding.{idx}.weight"));
        keys.insert(format!("text_embedding.{idx}.bias"));
    }
    // time embedding (Linear, SiLU, Linear)
    for idx in ["0", "2"] {
        keys.insert(format!("time_embedding.{idx}.weight"));
        keys.insert(format!("time_embedding.{idx}.bias"));
    }
    // time projection (SiLU, Linear) — index 1 is the Linear
    keys.insert("time_projection.1.weight".into());
    keys.insert("time_projection.1.bias".into());
    // heads (video + audio): 2-param modulation + Linear
    keys.insert("head.modulation".into());
    keys.insert("head.head.weight".into());
    keys.insert("head.head.bias".into());
    keys.insert("head_audio.modulation".into());
    keys.insert("head_audio.head.weight".into());
    keys.insert("head_audio.head.bias".into());

    // --- double blocks (alignment ×N): separate vid/audio self+cross attn,
    //     separate modulation+modulation_audio, shared norm3/ffn ---
    for i in 0..cfg.num_double_layers {
        let p = format!("double_blocks.{i}");
        // two modulation params
        keys.insert(format!("{p}.modulation.modulation"));
        keys.insert(format!("{p}.modulation_audio.modulation"));
        // self-attn (vid "" + audio "_audio")
        for suffix in ["", "_audio"] {
            attn_keys(&mut keys, &format!("{p}.self_attn"), suffix);
            attn_keys(&mut keys, &format!("{p}.cross_attn"), suffix);
        }
        // shared norm3 (affine) + shared ffn
        norm3_ffn_keys(&mut keys, &p);
    }

    // --- single blocks (fusion ×N): shared everything, no `_audio` ---
    for i in 0..cfg.num_single_layers {
        let p = format!("single_blocks.{i}");
        keys.insert(format!("{p}.modulation.modulation"));
        attn_keys(&mut keys, &format!("{p}.self_attn"), "");
        attn_keys(&mut keys, &format!("{p}.cross_attn"), "");
        norm3_ffn_keys(&mut keys, &p);
    }

    keys
}

/// One attention sub-module's keys: `{q,k,v,o}{suffix}.{weight,bias}` +
/// `norm_q{suffix}.weight` + `norm_k{suffix}.weight`.
fn attn_keys(keys: &mut HashSet<String>, prefix: &str, suffix: &str) {
    for proj in PROJ {
        keys.insert(format!("{prefix}.{proj}{suffix}.weight"));
        keys.insert(format!("{prefix}.{proj}{suffix}.bias"));
    }
    keys.insert(format!("{prefix}.norm_q{suffix}.weight"));
    keys.insert(format!("{prefix}.norm_k{suffix}.weight"));
}

/// Shared `norm3` (affine LayerNorm) + `ffn.{0,2}` keys for a block.
fn norm3_ffn_keys(keys: &mut HashSet<String>, prefix: &str) {
    keys.insert(format!("{prefix}.norm3.weight"));
    keys.insert(format!("{prefix}.norm3.bias"));
    keys.insert(format!("{prefix}.ffn.0.weight"));
    keys.insert(format!("{prefix}.ffn.0.bias"));
    keys.insert(format!("{prefix}.ffn.2.weight"));
    keys.insert(format!("{prefix}.ffn.2.bias"));
}

/// Load NAVA weights from a BF16 safetensors file into a [`WanAVModel`].
///
/// `safetensors_path` is the file produced by `convert_nava_ckpt.py` (verbatim
/// `backbone.*` keys, BF16). `cfg` is the model config (drives the expected-key
/// set and the `text_dim` cross-check). All weights are loaded RESIDENT.
///
/// Steps:
///  1. `flame_core::serialization::load_file` — full file into a key→Tensor map
///     (single-file checkpoint; sharded layout is handled by the converter's
///     index but the resident model takes one file — sharded support is a
///     `// TODO` if a sharded ckpt ever ships).
///  2. Strip `backbone.` from every key it carries (the model's namespace).
///  3. Cast any stray non-BF16 float tensor to BF16.
///  4. Cross-check `text_embedding.0.weight` shape → confirm `text_dim`.
///  5. Validate against [`expected_keys`]: hard-error on any missing model key,
///     warn on any unexpected file key (kept, not dropped).
///  6. Hand the map to `WanAVModel::new`.
pub fn load_nava(
    safetensors_path: &Path,
    cfg: &NavaAVConfig,
    device: Arc<CudaDevice>,
) -> Result<WanAVModel> {
    if !safetensors_path.is_file() {
        return Err(Error::InvalidInput(format!(
            "nava loader: weights file not found: {}",
            safetensors_path.display()
        )));
    }

    log::info!(
        "[nava] loading weights from {} (resident, BF16)",
        safetensors_path.display()
    );

    // 1. Load the whole file. (load_file mmaps + decodes every tensor.)
    let raw = flame_core::serialization::load_file(safetensors_path, &device)?;

    // 2+3. Strip `backbone.` and BF16-coerce float tensors. Verbatim otherwise.
    let mut weights: std::collections::HashMap<String, Tensor> =
        std::collections::HashMap::with_capacity(raw.len());
    for (key, value) in raw {
        let stripped = key
            .strip_prefix(BACKBONE_PREFIX)
            .map(|s| s.to_string())
            .unwrap_or(key);
        let value = match value.dtype() {
            DType::BF16 => value,
            // Coerce stray float dtypes (a differently-produced file). Non-float
            // (int/bool) dtypes are left untouched — they should not appear in
            // this DiT but we don't silently corrupt them if they do.
            DType::F16 | DType::F32 | DType::F64 => value.to_dtype(DType::BF16)?,
            _ => value,
        };
        weights.insert(stripped, value);
    }

    // 4. text_dim cross-check. `text_embedding.0` is `nn.Linear(text_dim, dim)`;
    //    PyTorch weight layout is `[out=dim, in=text_dim]`. Ground the config's
    //    `text_dim` (flagged as a 4096 default, not read from JSON) against the
    //    actual tensor so a mismatch fails loud here rather than as a GEMM shape
    //    error mid-forward.
    if let Some(te0) = weights.get("text_embedding.0.weight") {
        let dims = te0.shape().dims();
        if dims.len() != 2 {
            return Err(Error::InvalidInput(format!(
                "nava loader: text_embedding.0.weight expected 2D [dim, text_dim], got {dims:?}"
            )));
        }
        let (out_dim, in_dim) = (dims[0], dims[1]);
        if out_dim != cfg.dim {
            return Err(Error::InvalidInput(format!(
                "nava loader: text_embedding.0.weight out-dim {out_dim} != cfg.dim {} \
                 (checkpoint disagrees with config)",
                cfg.dim
            )));
        }
        if in_dim != cfg.text_dim {
            // text_dim was flagged as an ungrounded default — report the truth.
            log::warn!(
                "[nava] text_dim mismatch: cfg.text_dim={} but text_embedding.0.weight \
                 in-dim={in_dim}. Using checkpoint value is not possible (config is \
                 immutable here) — verify NavaAVConfig.text_dim matches the umt5 hidden \
                 size that produced these weights.",
                cfg.text_dim
            );
            return Err(Error::InvalidInput(format!(
                "nava loader: text_embedding.0.weight in-dim {in_dim} != cfg.text_dim {} \
                 — update NavaAVConfig.text_dim to {in_dim} (it was a default, not JSON-grounded)",
                cfg.text_dim
            )));
        }
        log::info!(
            "[nava] text_dim confirmed: text_embedding.0.weight = [{out_dim}, {in_dim}] \
             (dim={}, text_dim={in_dim})",
            cfg.dim
        );
    }
    // (If the key is entirely missing, the expected-key check below catches it.)

    // 5. Missing / unexpected key reconciliation.
    let expected = expected_keys(cfg);
    let present: HashSet<&String> = weights.keys().collect();

    // Missing: every model weight must have a source — fail loud, naming them.
    let mut missing: Vec<&String> = expected
        .iter()
        .filter(|k| !weights.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        missing.sort();
        let shown: Vec<&str> = missing.iter().take(20).map(|s| s.as_str()).collect();
        return Err(Error::InvalidInput(format!(
            "nava loader: {} expected weight(s) missing from {}. First {}:\n  {}",
            missing.len(),
            safetensors_path.display(),
            shown.len(),
            shown.join("\n  ")
        )));
    }

    // Unexpected: file keys the model never reads. Warn (don't drop) — e.g.
    // `speaker_embedding.*` (add_spk_emb base path unused) and any
    // `double_final_blocks.*` (empty/dead). Surfacing them is the safety net.
    let mut unexpected: Vec<&&String> = present
        .iter()
        .filter(|k| !expected.contains(**k))
        .collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        let shown: Vec<&str> = unexpected.iter().take(20).map(|s| s.as_str()).collect();
        log::warn!(
            "[nava] {} weight(s) in checkpoint are not read by the model (kept, not \
             dropped). First {}:\n  {}",
            unexpected.len(),
            shown.len(),
            shown.join("\n  ")
        );
    }

    log::info!(
        "[nava] {} weights resident ({} expected, {} extra) — constructing WanAVModel",
        weights.len(),
        expected.len(),
        weights.len().saturating_sub(expected.len())
    );

    // TODO: BlockOffloader wiring. `WanAVModel` is all-resident today (flat
    // HashMap, no offloader field). For the 24 GB single-card target the gen bin
    // (BUILD_PLAN item 12) will need the 30 double+single blocks streamed via
    // BlockOffloader (mirror `wan22_dit.rs::load_with_config` /
    // `ltx2_generate_av.rs` init_offloader_streaming). That requires the model
    // struct to grow an offloader field + a per-block facilitator keyed on
    // `double_blocks.{i}` / `single_blocks.{i}`. Out of scope for this loader:
    // it lives with the model struct + gen bin, not the resident-map builder.

    // 6. Construct.
    WanAVModel::new(cfg.clone(), weights, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expected-key set is a pure key/shape contract — testable with no
    /// weights, no GPU. Verifies counts and a few representative names so a
    /// future edit to a block's weight reads (in nava_blocks.rs) that drifts
    /// from this generator gets caught.
    #[test]
    fn expected_key_set_matches_block_layout() {
        let cfg = NavaAVConfig::default();
        let keys = expected_keys(&cfg);

        // Per-attn-submodule key count: 4 proj × 2 (w+b) + 2 norms = 10.
        const ATTN: usize = 4 * 2 + 2;
        // norm3 (w+b) + ffn.{0,2}.{w,b} = 2 + 4 = 6.
        const NORM3_FFN: usize = 2 + 4;

        // Single block: 1 modulation + self_attn + cross_attn + norm3/ffn.
        let single_per = 1 + ATTN + ATTN + NORM3_FFN;
        // Double block: 2 modulation + (self+cross)×(vid+audio) + norm3/ffn.
        let double_per = 2 + (ATTN + ATTN) * 2 + NORM3_FFN;

        // Top-level: patch_embedding(2) + patch_embedding_audio(2+3)
        //   + text_embedding(4) + time_embedding(4) + time_projection(2)
        //   + head(3) + head_audio(3) = 23.
        let top_level = 2 + (2 + 3) + 4 + 4 + 2 + 3 + 3;

        let expected_total = top_level
            + cfg.num_single_layers * single_per
            + cfg.num_double_layers * double_per;

        assert_eq!(
            keys.len(),
            expected_total,
            "expected-key count drifted from the per-block layout"
        );

        // Representative keys the forward reads (stripped namespace).
        for k in [
            "patch_embedding.weight",
            "patch_embedding_audio.2.w1.weight",
            "text_embedding.0.weight",
            "time_projection.1.bias",
            "head.head.weight",
            "head_audio.modulation",
            "double_blocks.0.modulation_audio.modulation",
            "double_blocks.9.self_attn.q_audio.weight",
            "double_blocks.0.self_attn.norm_k_audio.weight",
            "double_blocks.0.cross_attn.o_audio.bias",
            "single_blocks.0.self_attn.q.weight",
            "single_blocks.19.ffn.2.bias",
            "single_blocks.5.cross_attn.norm_q.weight",
            "single_blocks.0.norm3.weight",
        ] {
            assert!(keys.contains(k), "expected key set missing {k}");
        }

        // Single blocks must NOT carry `_audio` keys (shared everything).
        assert!(!keys.contains("single_blocks.0.self_attn.q_audio.weight"));
        assert!(!keys.contains("single_blocks.0.modulation_audio.modulation"));

        // No leftover `backbone.` prefix in the model namespace.
        assert!(keys.iter().all(|k| !k.starts_with("backbone.")));
    }
}
