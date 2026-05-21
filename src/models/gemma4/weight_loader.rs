//! Gemma-4 safetensors → flame_core Tensor loader, BlockOffloader-aware.
//!
//! Hugging Face ships Gemma-4-31B-it as multiple safetensors shards
//! plus a `model.safetensors.index.json` that maps every weight key
//! to its shard. This loader:
//!
//! 1. Parses the index.
//! 2. Resolves each Gemma-4 weight key to a flame_core Tensor (BF16,
//!    on device or pinned RAM depending on offload policy).
//! 3. Routes per-layer weights through `BlockOffloader` so we never
//!    have all 60 layers resident at once. The embed/final-norm are
//!    small enough to stay resident.
//!
//! ## Key naming (Gemma-4-31B-it safetensors, verified 2026-05-21)
//!
//! Resident (loaded eagerly to GPU BF16):
//!
//! - `model.language_model.embed_tokens.weight`            `[vocab, hidden]`
//! - `model.language_model.norm.weight`                    `[hidden]`
//!
//! Per-layer (paged via BlockOffloader, 13 keys × 60 layers = 780 tensors):
//!
//! - `model.language_model.layers.{i}.input_layernorm.weight`
//! - `model.language_model.layers.{i}.post_attention_layernorm.weight`
//! - `model.language_model.layers.{i}.pre_feedforward_layernorm.weight`
//! - `model.language_model.layers.{i}.post_feedforward_layernorm.weight`
//! - `model.language_model.layers.{i}.self_attn.q_proj.weight`
//! - `model.language_model.layers.{i}.self_attn.k_proj.weight`
//! - `model.language_model.layers.{i}.self_attn.v_proj.weight`
//! - `model.language_model.layers.{i}.self_attn.o_proj.weight`
//! - `model.language_model.layers.{i}.self_attn.q_norm.weight`
//! - `model.language_model.layers.{i}.self_attn.k_norm.weight`
//! - `model.language_model.layers.{i}.mlp.gate_proj.weight`
//! - `model.language_model.layers.{i}.mlp.up_proj.weight`
//! - `model.language_model.layers.{i}.mlp.down_proj.weight`
//! - `model.language_model.layers.{i}.layer_scalar`         (no `.weight` suffix)
//!
//! NOTE: PORT_SPEC §2 claims q_norm/k_norm are absent on Gemma-4 ("unlike
//! Gemma-3"). The on-disk safetensors index disagrees — every text layer
//! ships both. The decoder layer will need to consume them; flagged in
//! the builder's session report so a follow-up audit can decide whether
//! to apply them or skip them at runtime. We load them either way so the
//! decoder slice has the data when it lands.
//!
//! Skipped (not in the text-only path):
//!
//! - `model.vision_tower.*`         (27-layer ViT, large)
//! - `model.embed_vision.*`         (vision projection)
//! - `model.embed_audio.*` / `audio_*` (audio path)
//! - `lm_head.weight`               (absent; tied to embed_tokens — see config)
//!
//! ## BlockOffloader integration
//!
//! Mirrors `inference-flame/src/models/hidream_o1/weight_loader.rs`. The
//! `BlockFacilitator` returns `Some(i)` for any
//! `model.language_model.layers.{i}.<rest>` key and `None` for everything
//! else; the offloader uses that to bucket pinned tensors per layer.
//! Vision / audio / resident keys come back `None`, the offloader drops
//! them, and we pick the resident subset up in a separate CPU-side
//! F32→BF16 pass.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result as AnyResult};
use flame_core::offload::{BlockFacilitator, BlockOffloader};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

use crate::models::gemma4::Gemma4Config;

/// Per-layer key router for the BlockOffloader.
///
/// Anything matching `model.language_model.layers.{i}.` belongs to block
/// `i`. Everything else (resident shared + vision-tower + audio) returns
/// `None` and is dropped by the offloader.
struct Gemma4BlockFacilitator {
    num_blocks: usize,
}

impl BlockFacilitator for Gemma4BlockFacilitator {
    fn block_count(&self) -> usize {
        self.num_blocks
    }
    fn classify_key(&self, key: &str) -> Option<usize> {
        classify_layer_key(key)
    }
}

/// Classify a safetensors key as belonging to a text-decoder layer.
///
/// Returns `Some(layer_idx)` for
/// `model.language_model.layers.{i}.<anything>`, `None` otherwise.
/// Mirrors `hidream_o1::weight_loader::classify_layer_key`.
pub fn classify_layer_key(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("model.language_model.layers.")?;
    rest.split('.').next()?.parse().ok()
}

/// Gemma-4 weight loader. Holds the resident embed + final-norm tensors
/// and the per-layer streaming offloader.
///
/// AGENT-DEFAULT decision (flag in report): I unify "load index" and
/// "load model" into one constructor (`open`) because Gemma-4's text
/// decoder is a single closed system — the caller never wants to peek
/// at the index without also loading. If a parity tool grows, factor a
/// separate `from_index_only` constructor.
pub struct Gemma4WeightLoader {
    pub cfg: Gemma4Config,
    pub model_dir: PathBuf,
    /// `tensor_name` → `shard_filename`. Useful for follow-up parity
    /// tools that want to know which shard a specific key landed in.
    pub shard_map: HashMap<String, String>,
    /// Streaming per-layer weight cache. `block_count()` == `cfg.num_hidden_layers`.
    /// `prefetch_block(i)` + `await_block_handle(i)` exposes the 13 tensors
    /// for layer `i` as a `HashMap<String, Tensor>` over GPU BF16 memory.
    pub offloader: BlockOffloader,
    /// `model.language_model.embed_tokens.weight` — `[vocab, hidden]` BF16 GPU tensor.
    /// Kept resident so embedding lookup at every decode step is a cheap
    /// gather into a GPU-resident matrix.
    pub embed_tokens: Tensor,
    /// `model.language_model.norm.weight` — `[hidden]` BF16 GPU tensor.
    pub final_norm: Tensor,
}

impl Gemma4WeightLoader {
    /// Open a model directory. Reads index.json, opens each shard via
    /// `safetensors` mmap, builds the layer-handle table via
    /// [`BlockOffloader::load`], and eagerly loads the resident embed
    /// table and final-norm weight on the GPU as BF16.
    ///
    /// `device`: GPU device to bind the resident tensors to. The
    /// offloader also pins its CPU staging buffers to the same device.
    pub fn open(
        model_dir: &Path,
        cfg: &Gemma4Config,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // We use anyhow internally for the rich Context support; convert
        // to flame_core::Error at the boundary to match the signature
        // (`Result` here is `flame_core::Result`, see `use` above).
        Self::open_impl(model_dir, cfg, device)
            .map_err(|e| flame_core::Error::InvalidInput(format!("Gemma4WeightLoader::open: {e:#}")))
    }

    fn open_impl(
        model_dir: &Path,
        cfg: &Gemma4Config,
        device: &Arc<CudaDevice>,
    ) -> AnyResult<Self> {
        // 1) Parse model.safetensors.index.json → key → shard_filename.
        let index_path = model_dir.join("model.safetensors.index.json");
        let bytes = std::fs::read(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse JSON from {}", index_path.display()))?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("{}: missing weight_map", index_path.display()))?;

        let mut shard_map: HashMap<String, String> = HashMap::with_capacity(weight_map.len());
        for (k, v) in weight_map {
            let shard = v
                .as_str()
                .ok_or_else(|| {
                    anyhow!("{}: non-string shard for key {k}", index_path.display())
                })?
                .to_string();
            shard_map.insert(k.clone(), shard);
        }

        // 2) Distinct, sorted absolute shard paths. We visit every shard
        //    even though some only carry vision keys (they're cheap to
        //    mmap-scan) because resident keys may live in any of them.
        let mut unique: HashSet<&str> = shard_map.values().map(|s| s.as_str()).collect();
        let mut shard_paths: Vec<PathBuf> = unique
            .drain()
            .map(|s| model_dir.join(s))
            .collect();
        shard_paths.sort();

        // 3) Build the BlockOffloader. The facilitator drops every key
        //    that isn't `model.language_model.layers.{i}.` — vision,
        //    audio, embed, and final-norm all classify_key() → None and
        //    never enter the offloader's pinned pool.
        let facilitator = Gemma4BlockFacilitator {
            num_blocks: cfg.num_hidden_layers,
        };
        let shard_strs: Vec<String> = shard_paths
            .iter()
            .map(|p| {
                p.to_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("non-utf8 shard path: {:?}", p))
            })
            .collect::<AnyResult<Vec<_>>>()?;
        let shard_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();

        log::info!(
            "[gemma4] BlockOffloader::load: {} shards, {} blocks",
            shard_refs.len(),
            cfg.num_hidden_layers
        );
        // AGENT-DEFAULT: native_layout(true) so the decoder's calls into
        // `fused_linear3d_native` see PyTorch-native `[Cout, Cin]`
        // weights and let cuBLASLt do the transpose inside the GEMM
        // (TRANSA=T). Matches the HiDream-O1 loader's choice.
        let offloader = BlockOffloader::load(&shard_refs, &facilitator, device.clone())
            .map(|o| o.with_native_layout(true))
            .map_err(|e| anyhow!("Gemma4 BlockOffloader::load: {e:#}"))?;

        // 4) Resident-shared keys: text embed + final RMSNorm.
        //
        // Per `inference-flame/src/models/hidream_o1/weight_loader.rs`,
        // we read these CPU-side and do the F32→BF16 conversion before
        // the upload so we never have an F32 transient on the GPU
        // (CLAUDE.md hard rule: "no F32 fallbacks in inference code").
        let resident_keys: HashSet<String> = [
            "model.language_model.embed_tokens.weight".to_string(),
            "model.language_model.norm.weight".to_string(),
        ]
        .iter()
        .cloned()
        .collect();

        let mut resident = load_resident_cpu_bf16(
            &shard_paths,
            &resident_keys,
            device,
        )?;

        let embed_tokens = resident
            .remove("model.language_model.embed_tokens.weight")
            .ok_or_else(|| anyhow!("missing embed_tokens.weight after resident load"))?;

        // Shape sanity: `[vocab, hidden]`.
        {
            let dims = embed_tokens.shape().dims();
            if dims != [cfg.vocab_size, cfg.hidden_size] {
                bail!(
                    "embed_tokens shape mismatch: got {:?}, want [{}, {}]",
                    dims,
                    cfg.vocab_size,
                    cfg.hidden_size
                );
            }
        }

        let final_norm = resident
            .remove("model.language_model.norm.weight")
            .ok_or_else(|| anyhow!("missing language_model.norm.weight after resident load"))?;
        {
            let dims = final_norm.shape().dims();
            if dims != [cfg.hidden_size] {
                bail!(
                    "language_model.norm.weight shape mismatch: got {:?}, want [{}]",
                    dims,
                    cfg.hidden_size
                );
            }
        }

        if !resident.is_empty() {
            // We only asked for 2 keys; leftover is a loader bug.
            bail!(
                "Gemma4WeightLoader: leftover resident keys: {:?}",
                resident.keys().collect::<Vec<_>>()
            );
        }

        // Inference path: freeze the resident weights so any code that
        // later attaches LoRAs or runs forward doesn't accidentally
        // track gradients through these (they're frozen base weights).
        let embed_tokens = embed_tokens.requires_grad_(false);
        let final_norm = final_norm.requires_grad_(false);

        log::info!(
            "[gemma4] resident loaded: embed_tokens {:?}, norm {:?}",
            embed_tokens.shape().dims(),
            final_norm.shape().dims()
        );

        Ok(Self {
            cfg: cfg.clone(),
            model_dir: model_dir.to_path_buf(),
            shard_map,
            offloader,
            embed_tokens,
            final_norm,
        })
    }
}

/// Load a small set of resident keys from sharded safetensors with
/// CPU-side F32→BF16 conversion. The resulting tensors are GPU-resident
/// BF16; F32 never appears on the GPU.
///
/// Mirrors `HiDreamO1WeightLoader::load_resident_cpu_bf16` — same shape,
/// same dtype dispatch (F32 / BF16 / F16 supported; everything else
/// errors). Kept as a free function rather than a method so it can be
/// reused by future parity tools without dragging in the full loader.
fn load_resident_cpu_bf16(
    shard_paths: &[PathBuf],
    wanted: &HashSet<String>,
    device: &Arc<CudaDevice>,
) -> AnyResult<HashMap<String, Tensor>> {
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(wanted.len());

    for shard_path in shard_paths {
        let file = std::fs::File::open(shard_path)
            .with_context(|| format!("open {}", shard_path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap {}", shard_path.display()))?;
        if mmap.len() < 8 {
            bail!("{}: shard too small for safetensors", shard_path.display());
        }
        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_size;
        if header_end > mmap.len() {
            bail!("{}: header extends past EOF", shard_path.display());
        }
        let data_start = header_end;
        let metadata: serde_json::Value = serde_json::from_slice(&mmap[8..header_end])
            .with_context(|| {
                format!("parse safetensors header in {}", shard_path.display())
            })?;
        let metadata_obj = metadata
            .as_object()
            .ok_or_else(|| anyhow!("{}: invalid metadata format", shard_path.display()))?;

        for (name, info) in metadata_obj {
            if name == "__metadata__" {
                continue;
            }
            if !wanted.contains(name) {
                continue;
            }

            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("missing shape for {name}"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let num_elems: usize = shape.iter().product();
            if num_elems == 0 {
                continue;
            }

            let offsets = info["data_offsets"]
                .as_array()
                .ok_or_else(|| anyhow!("missing data_offsets for {name}"))?;
            let start = data_start
                + offsets
                    .first()
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad start offset for {name}"))? as usize;
            let end = data_start
                + offsets
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad end offset for {name}"))? as usize;

            let dtype_str = info["dtype"].as_str().unwrap_or("F32");
            let raw = &mmap[start..end];

            // CPU-side conversion to BF16 bits.
            let bf16_u16: Vec<u16> = match dtype_str {
                "BF16" => {
                    if raw.len() != num_elems * 2 {
                        bail!("{name}: BF16 byte length mismatch ({} != {})", raw.len(), num_elems * 2);
                    }
                    let mut buf = vec![0u16; num_elems];
                    for (v, chunk) in buf.iter_mut().zip(raw.chunks_exact(2)) {
                        *v = u16::from_le_bytes([chunk[0], chunk[1]]);
                    }
                    buf
                }
                "F16" => {
                    if raw.len() != num_elems * 2 {
                        bail!("{name}: F16 byte length mismatch");
                    }
                    let mut buf = vec![0u16; num_elems];
                    for (v, chunk) in buf.iter_mut().zip(raw.chunks_exact(2)) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        *v = half::bf16::from_f32(half::f16::from_bits(bits).to_f32())
                            .to_bits();
                    }
                    buf
                }
                "F32" => {
                    if raw.len() != num_elems * 4 {
                        bail!("{name}: F32 byte length mismatch");
                    }
                    let mut buf = vec![0u16; num_elems];
                    for (v, chunk) in buf.iter_mut().zip(raw.chunks_exact(4)) {
                        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        *v = half::bf16::from_f32(f).to_bits();
                    }
                    buf
                }
                other => bail!("{name}: unsupported dtype {other}"),
            };

            let mut tensor = Tensor::zeros_dtype(
                Shape::from_dims(&shape),
                DType::BF16,
                device.clone(),
            )
            .map_err(|e| anyhow!("{name}: alloc BF16 GPU tensor: {e}"))?;
            tensor
                .copy_from_bf16_slice(&bf16_u16)
                .map_err(|e| anyhow!("{name}: copy_from_bf16_slice: {e}"))?;
            out.insert(name.clone(), tensor);
        }
    }

    let missing: Vec<&String> = wanted.iter().filter(|k| !out.contains_key(k.as_str())).collect();
    if !missing.is_empty() {
        bail!(
            "Gemma4 resident loader: {} missing keys; first few: {:?}",
            missing.len(),
            &missing[..missing.len().min(8)]
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_layer_key_handles_text_decoder_variants() {
        for (key, expected) in [
            ("model.language_model.layers.0.input_layernorm.weight", Some(0)),
            ("model.language_model.layers.13.self_attn.q_proj.weight", Some(13)),
            ("model.language_model.layers.59.mlp.down_proj.weight", Some(59)),
            ("model.language_model.layers.7.self_attn.q_norm.weight", Some(7)),
            ("model.language_model.layers.7.layer_scalar", Some(7)),
            ("model.language_model.embed_tokens.weight", None),
            ("model.language_model.norm.weight", None),
            ("model.vision_tower.encoder.layers.0.input_layernorm.weight", None),
            ("model.embed_vision.embedding_projection.weight", None),
            ("lm_head.weight", None),
        ] {
            assert_eq!(classify_layer_key(key), expected, "key={key}");
        }
    }
}
