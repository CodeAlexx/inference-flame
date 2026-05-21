//! HiDream-O1 sharded safetensors weight loader.
//!
//! Reads `<model_dir>/model.safetensors.index.json` to discover which shard
//! file holds each tensor, then per-shard memory-maps the relevant tensors:
//! - **per-layer** transformer weights stream from pinned host RAM via
//!   [`flame_core::offload::BlockOffloader`] (one block = one decoder layer);
//! - **resident-shared** weights (embed_tokens, final norm, HiDream heads)
//!   are copied directly to BF16 GPU tensors via a CPU-side F32→BF16
//!   conversion to avoid a transient F32-on-GPU peak.
//!
//! ## Why the offloader is required
//!
//! HiDream-O1-Image-Dev ships **FP32** weights (per `config.json`'s implicit
//! `torch_dtype=torch.float32`). The previous loader path went through
//! `Tensor::from_vec` (F32 on GPU) → `to_dtype(BF16)` which doubles GPU
//! memory mid-load. With 8 B parameters and a 24 GB GPU that's a guaranteed
//! OOM in the very first key (`model.language_model.embed_tokens.weight` at
//! 152 K × 4096 × 4 B = 2.5 GB FP32 alone).
//!
//! The offloader handles per-layer weights entirely on the CPU side
//! (`block_offload.rs` lines 465-471: F32 → BF16 in a host buffer, then
//! `cudaMallocHost`-pinned), and the only GPU-resident copies are the two
//! ping-pong slots (≈ 220 MB each for the 8 B variant). The resident-shared
//! path mirrors the same CPU conversion so embed_tokens never appears as
//! FP32 on the GPU.
//!
//! ## Key naming map (HF safetensors → Rust struct fields)
//!
//! Verified against `model.safetensors.index.json` for
//! `HiDream-O1-Image-Dev-weights`:
//!
//! ```text
//! HF Key                                                            → Rust slot
//! ────────────────────────────────────────────────────────────────────────────────
//! model.language_model.embed_tokens.weight                          → resident.embed_tokens.weight
//! model.language_model.norm.weight                                  → resident.norm.weight
//! model.language_model.layers.{i}.input_layernorm.weight            → offloader block i, key as-is
//! model.language_model.layers.{i}.post_attention_layernorm.weight   → offloader block i, key as-is
//! model.language_model.layers.{i}.self_attn.q_proj.weight           → offloader block i, key as-is
//! model.language_model.layers.{i}.self_attn.k_proj.weight           → offloader block i, key as-is
//! model.language_model.layers.{i}.self_attn.v_proj.weight           → offloader block i, key as-is
//! model.language_model.layers.{i}.self_attn.o_proj.weight           → offloader block i, key as-is
//! model.language_model.layers.{i}.self_attn.q_norm.weight           → offloader block i, key as-is
//! model.language_model.layers.{i}.self_attn.k_norm.weight           → offloader block i, key as-is
//! model.language_model.layers.{i}.mlp.gate_proj.weight              → offloader block i, key as-is
//! model.language_model.layers.{i}.mlp.up_proj.weight                → offloader block i, key as-is
//! model.language_model.layers.{i}.mlp.down_proj.weight              → offloader block i, key as-is
//!
//! model.x_embedder.proj1.weight                                     → resident.bottleneck_patch_embed.proj1.weight
//! model.x_embedder.proj2.weight                                     → resident.bottleneck_patch_embed.proj2.weight
//! model.x_embedder.proj2.bias                                       → resident.bottleneck_patch_embed.proj2.bias
//! model.t_embedder1.mlp.0.weight                                    → resident.timestep_embedder.mlp_in.weight
//! model.t_embedder1.mlp.0.bias                                      → resident.timestep_embedder.mlp_in.bias
//! model.t_embedder1.mlp.2.weight                                    → resident.timestep_embedder.mlp_out.weight
//! model.t_embedder1.mlp.2.bias                                      → resident.timestep_embedder.mlp_out.bias
//! model.final_layer2.linear.weight                                  → resident.final_layer.linear.weight
//! model.final_layer2.linear.bias                                    → resident.final_layer.linear.bias
//! ```
//!
//! Keys ignored (vision tower + LM head — not used for T2I generation):
//! - `model.visual.*` (Qwen3-VL vision encoder, ~316 keys)
//! - `lm_head.weight` (autoregressive LM head, not used during diffusion)

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result as AnyResult};
use flame_core::offload::{BlockFacilitator, BlockOffloader};
use flame_core::{CudaDevice, DType, Shape, Tensor};

use super::HiDreamO1Config;
use super::model::HiDreamO1Model;

/// Per-layer key router for the BlockOffloader. Anything matching
/// `model.language_model.layers.{i}.` belongs to block `i`. All other keys
/// (resident-shared and vision-tower) return `None`.
struct HiDreamBlockFacilitator {
    num_blocks: usize,
}

impl BlockFacilitator for HiDreamBlockFacilitator {
    fn block_count(&self) -> usize {
        self.num_blocks
    }
    fn classify_key(&self, key: &str) -> Option<usize> {
        classify_layer_key(key)
    }
}

/// Classify a safetensors key as belonging to a transformer layer.
///
/// Returns `Some(layer_idx)` for `model.language_model.layers.{i}.<...>`,
/// `None` otherwise. Mirrors `sensenova_u1.rs::classify_layer_key`.
pub fn classify_layer_key(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("model.language_model.layers.")?;
    rest.split('.').next()?.parse().ok()
}

/// Discovers shards from `model.safetensors.index.json` and loads the
/// HiDream-O1 model.
pub struct HiDreamO1WeightLoader {
    pub model_dir: PathBuf,
    /// `tensor_name` → `shard_filename`.
    pub shard_map: HashMap<String, String>,
}

impl HiDreamO1WeightLoader {
    /// Read `model.safetensors.index.json` and build the shard map.
    pub fn from_dir(model_dir: &Path) -> AnyResult<Self> {
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
                .ok_or_else(|| anyhow!("{}: non-string shard for key {k}", index_path.display()))?
                .to_string();
            shard_map.insert(k.clone(), shard);
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            shard_map,
        })
    }

    /// Distinct, sorted absolute shard paths used by **any** key in the
    /// index. (Vision-tower keys and lm_head are routed by the loader's
    /// filter logic; we still need to visit every shard because resident
    /// keys could land in any of them.)
    fn all_shards_sorted(&self) -> Vec<PathBuf> {
        let unique: HashSet<&str> = self.shard_map.values().map(|s| s.as_str()).collect();
        let mut out: Vec<PathBuf> = unique
            .into_iter()
            .map(|s| self.model_dir.join(s))
            .collect();
        out.sort();
        out
    }

    /// Build the full set of HF keys we treat as **resident-shared** (loaded
    /// to GPU once at start, never streamed).
    fn resident_keys(&self, _config: &HiDreamO1Config) -> Vec<String> {
        vec![
            // Text spine — embed + final norm.
            "model.language_model.embed_tokens.weight".to_string(),
            "model.language_model.norm.weight".to_string(),
            // HiDream-only heads.
            "model.x_embedder.proj1.weight".to_string(),
            "model.x_embedder.proj2.weight".to_string(),
            "model.x_embedder.proj2.bias".to_string(),
            "model.t_embedder1.mlp.0.weight".to_string(),
            "model.t_embedder1.mlp.0.bias".to_string(),
            "model.t_embedder1.mlp.2.weight".to_string(),
            "model.t_embedder1.mlp.2.bias".to_string(),
            "model.final_layer2.linear.weight".to_string(),
            "model.final_layer2.linear.bias".to_string(),
        ]
    }

    /// Load resident keys with **CPU-side F32→BF16 conversion** so the
    /// transient F32 buffer never lives on the GPU.
    ///
    /// Mirrors the BF16-host path inside `BlockOffloader::load_inner`
    /// (block_offload.rs:449-498). We open each shard via mmap, walk the
    /// safetensors metadata, and for every `wanted` key we materialize a
    /// `Vec<u16>` of BF16 bits on the host, then upload via
    /// `Tensor::zeros_dtype(BF16)` + `copy_from_bf16_slice`.
    fn load_resident_cpu_bf16(
        &self,
        wanted: &HashSet<String>,
        device: &Arc<CudaDevice>,
    ) -> AnyResult<HashMap<String, Tensor>> {
        let mut out: HashMap<String, Tensor> = HashMap::with_capacity(wanted.len());

        for shard_path in self.all_shards_sorted() {
            let file = std::fs::File::open(&shard_path)
                .with_context(|| format!("open {}", shard_path.display()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .with_context(|| format!("mmap {}", shard_path.display()))?;
            if mmap.len() < 8 {
                bail!("{}: shard too small for safetensors", shard_path.display());
            }
            let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header_end = 8 + header_size;
            let data_start = header_end;
            let metadata: serde_json::Value = serde_json::from_slice(&mmap[8..header_end])
                .with_context(|| format!("parse safetensors header in {}", shard_path.display()))?;
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
                    + offsets.first().and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("bad start offset for {name}"))? as usize;
                let end = data_start
                    + offsets.get(1).and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("bad end offset for {name}"))? as usize;

                let dtype_str = info["dtype"].as_str().unwrap_or("F32");
                let raw = &mmap[start..end];

                // Convert to BF16 on the host. Match the four supported
                // safetensors dtypes that show up in HiDream-O1 (F32 in
                // practice, but BF16/F16 are tolerated for forward-compat).
                let bf16_u16: Vec<u16> = match dtype_str {
                    "BF16" => {
                        let mut out = vec![0u16; num_elems];
                        for (v, chunk) in out.iter_mut().zip(raw.chunks_exact(2)) {
                            *v = u16::from_le_bytes([chunk[0], chunk[1]]);
                        }
                        out
                    }
                    "F16" => {
                        let mut out = vec![0u16; num_elems];
                        for (v, chunk) in out.iter_mut().zip(raw.chunks_exact(2)) {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            *v = half::bf16::from_f32(half::f16::from_bits(bits).to_f32())
                                .to_bits();
                        }
                        out
                    }
                    "F32" => {
                        let mut out = vec![0u16; num_elems];
                        for (v, chunk) in out.iter_mut().zip(raw.chunks_exact(4)) {
                            let f =
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            *v = half::bf16::from_f32(f).to_bits();
                        }
                        out
                    }
                    other => bail!("{name}: unsupported dtype {other}"),
                };

                let mut tensor = Tensor::zeros_dtype(
                    Shape::from_dims(&shape),
                    DType::BF16,
                    device.clone(),
                )?;
                tensor.copy_from_bf16_slice(&bf16_u16)?;
                out.insert(name.clone(), tensor);
            }
        }

        // Validate every requested key was found.
        let missing: Vec<&String> = wanted.iter().filter(|k| !out.contains_key(k.as_str())).collect();
        if !missing.is_empty() {
            bail!(
                "HiDream-O1 resident loader: {} missing keys; first few: {:?}",
                missing.len(),
                &missing[..missing.len().min(8)]
            );
        }
        Ok(out)
    }

    /// Load only resident HiDream-O1 tensors. Used by parity tools that need
    /// to inspect the layer-0 input assembly without paying the full decoder
    /// offloader startup cost.
    pub fn load_resident_weights_bf16(
        &self,
        config: &HiDreamO1Config,
        device: &Arc<CudaDevice>,
    ) -> AnyResult<HashMap<String, Tensor>> {
        let resident_list = self.resident_keys(config);
        let resident_set: HashSet<String> = resident_list.iter().cloned().collect();
        self.load_resident_cpu_bf16(&resident_set, device)
    }

    /// Load a caller-selected resident subset with the same CPU-side BF16
    /// conversion as the full resident loader.
    pub fn load_selected_resident_weights_bf16(
        &self,
        keys: &[&str],
        device: &Arc<CudaDevice>,
    ) -> AnyResult<HashMap<String, Tensor>> {
        let wanted: HashSet<String> = keys.iter().map(|k| (*k).to_string()).collect();
        let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
        for key in &wanted {
            let shard = self
                .shard_map
                .get(key)
                .ok_or_else(|| anyhow!("model index missing selected resident key: {key}"))?;
            by_shard.entry(shard.clone()).or_default().push(key.clone());
        }

        let mut out: HashMap<String, Tensor> = HashMap::with_capacity(wanted.len());
        for (shard_name, shard_keys) in by_shard {
            let shard_path = self.model_dir.join(&shard_name);
            let file = std::fs::File::open(&shard_path)
                .with_context(|| format!("open {}", shard_path.display()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .with_context(|| format!("mmap {}", shard_path.display()))?;
            if mmap.len() < 8 {
                bail!("{}: shard too small for safetensors", shard_path.display());
            }
            let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header_end = 8 + header_size;
            let data_start = header_end;
            let metadata: serde_json::Value = serde_json::from_slice(&mmap[8..header_end])
                .with_context(|| format!("parse safetensors header in {}", shard_path.display()))?;
            let metadata_obj = metadata
                .as_object()
                .ok_or_else(|| anyhow!("{}: invalid metadata format", shard_path.display()))?;

            for name in shard_keys {
                let info = metadata_obj
                    .get(&name)
                    .ok_or_else(|| anyhow!("{}: missing tensor {name}", shard_path.display()))?;
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow!("missing shape for {name}"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let num_elems: usize = shape.iter().product();
                let offsets = info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow!("missing data_offsets for {name}"))?;
                let start = data_start
                    + offsets.first().and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("bad start offset for {name}"))? as usize;
                let end = data_start
                    + offsets.get(1).and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("bad end offset for {name}"))? as usize;
                let dtype_str = info["dtype"].as_str().unwrap_or("F32");
                let raw = &mmap[start..end];

                let tensor = match dtype_str {
                    "F32" => {
                        if raw.len() != num_elems * 4 {
                            bail!("{name}: F32 byte length mismatch");
                        }
                        let mut bf16_u16 = vec![0u16; num_elems];
                        for (v, chunk) in bf16_u16.iter_mut().zip(raw.chunks_exact(4)) {
                            let f =
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            *v = half::bf16::from_f32(f).to_bits();
                        }
                        let mut tensor = Tensor::zeros_dtype(
                            Shape::from_dims(&shape),
                            DType::BF16,
                            device.clone(),
                        )?;
                        tensor.copy_from_bf16_slice(&bf16_u16)?;
                        tensor
                    }
                    "BF16" => {
                        if raw.len() != num_elems * 2 {
                            bail!("{name}: BF16 byte length mismatch");
                        }
                        let mut bits = vec![0u16; num_elems];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                raw.as_ptr(),
                                bits.as_mut_ptr() as *mut u8,
                                raw.len(),
                            );
                        }
                        let mut tensor = Tensor::zeros_dtype(
                            Shape::from_dims(&shape),
                            DType::BF16,
                            device.clone(),
                        )?;
                        tensor.copy_from_bf16_slice(&bits)?;
                        tensor
                    }
                    "F16" => {
                        let mut bf16_u16 = vec![0u16; num_elems];
                        for (v, chunk) in bf16_u16.iter_mut().zip(raw.chunks_exact(2)) {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            *v = half::bf16::from_f32(half::f16::from_bits(bits).to_f32())
                                .to_bits();
                        }
                        let mut tensor = Tensor::zeros_dtype(
                            Shape::from_dims(&shape),
                            DType::BF16,
                            device.clone(),
                        )?;
                        tensor.copy_from_bf16_slice(&bf16_u16)?;
                        tensor
                    }
                    other => bail!("{name}: unsupported dtype {other}"),
                };
                out.insert(name, tensor);
            }
        }

        let missing: Vec<&String> = wanted.iter().filter(|k| !out.contains_key(k.as_str())).collect();
        if !missing.is_empty() {
            bail!("selected resident loader missing keys: {:?}", missing);
        }
        Ok(out)
    }

    /// Load the entire HiDream-O1 model from sharded safetensors.
    ///
    /// Vision-tower and `lm_head` keys are intentionally skipped — they're
    /// not needed for the T2I forward pass (Phase 2c only supports the
    /// no-ref-image path). If a later phase wires the vision tower, add
    /// its keys here and route them to the corresponding new modules.
    pub fn load_model(
        &self,
        config: &HiDreamO1Config,
        device: &Arc<CudaDevice>,
    ) -> AnyResult<HiDreamO1Model> {
        // 1) Build the per-layer streaming offloader. The facilitator only
        //    classifies `model.language_model.layers.{i}.` keys; everything
        //    else (resident shared + vision-tower) is dropped on the floor
        //    by the offloader and picked up below.
        let facilitator = HiDreamBlockFacilitator {
            num_blocks: config.num_layers,
        };
        let shard_paths = self.all_shards_sorted();
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
            "[hidream_o1] BlockOffloader::load: {} shards, {} blocks",
            shard_refs.len(),
            config.num_layers
        );
        if std::env::var_os("FLAME_LAYER_OFFLOAD_FRACTION").is_none() {
            unsafe {
                std::env::set_var("FLAME_LAYER_OFFLOAD_FRACTION", "0.77");
            }
        }
        let offloader = BlockOffloader::load(&shard_refs, &facilitator, device.clone())
            .map(|o| o.with_native_layout(true))
            .map_err(|e| anyhow!("HiDream-O1 BlockOffloader::load: {e}"))?;

        // 2) Load resident-shared keys with CPU-side F32→BF16 (no F32-on-GPU
        //    transient).
        let resident_list = self.resident_keys(config);
        let resident_set: HashSet<String> = resident_list.iter().cloned().collect();
        log::info!(
            "[hidream_o1] resident-shared loader: {} keys",
            resident_set.len()
        );
        let mut weights = self.load_resident_cpu_bf16(&resident_set, device)?;

        // 3) Construct the model with the offloader installed; this only
        //    allocates the resident modules + zero-init norms.
        let mut model = HiDreamO1Model::new(config.clone(), device, offloader, DType::BF16)
            .map_err(|e| anyhow!("HiDreamO1Model::new: {e}"))?;

        // Helper: pop a tensor by key with a uniform error message.
        let take = |m: &mut HashMap<String, Tensor>, k: &str| -> AnyResult<Tensor> {
            m.remove(k).ok_or_else(|| anyhow!("missing weight key: {k}"))
        };

        // 4) embed_tokens — direct field assignment (no copy_weight_from API).
        {
            let t = take(&mut weights, "model.language_model.embed_tokens.weight")?;
            // Shape sanity: [vocab, hidden_size].
            let dims = t.shape().dims();
            if dims != [config.vocab_size, config.hidden_size] {
                bail!(
                    "embed_tokens shape mismatch: got {:?}, want [{}, {}]",
                    dims,
                    config.vocab_size,
                    config.hidden_size
                );
            }
            // Embedding stores `weight` as a public field; assign directly.
            // Drop requires_grad — pure inference.
            model.embed_tokens.weight = t;
        }

        // 5) Final RMSNorm.
        {
            let t = take(&mut weights, "model.language_model.norm.weight")?;
            model
                .norm
                .copy_weight_from(&t)
                .map_err(|e| anyhow!("norm.copy_weight_from: {e}"))?;
        }

        // 6) HiDream heads — bottleneck patch embed.
        {
            let proj1_w = take(&mut weights, "model.x_embedder.proj1.weight")?;
            model
                .bottleneck_patch_embed
                .proj1
                .copy_weight_from(&proj1_w)
                .map_err(|e| anyhow!("x_embedder.proj1: {e}"))?;

            let proj2_w = take(&mut weights, "model.x_embedder.proj2.weight")?;
            let proj2_b = take(&mut weights, "model.x_embedder.proj2.bias")?;
            model
                .bottleneck_patch_embed
                .proj2
                .copy_weight_from(&proj2_w)
                .map_err(|e| anyhow!("x_embedder.proj2.weight: {e}"))?;
            model
                .bottleneck_patch_embed
                .proj2
                .copy_bias_from(&proj2_b)
                .map_err(|e| anyhow!("x_embedder.proj2.bias: {e}"))?;
        }

        // 7) HiDream heads — timestep embedder (sinusoidal MLP).
        {
            let mlp0_w = take(&mut weights, "model.t_embedder1.mlp.0.weight")?;
            let mlp0_b = take(&mut weights, "model.t_embedder1.mlp.0.bias")?;
            let mlp2_w = take(&mut weights, "model.t_embedder1.mlp.2.weight")?;
            let mlp2_b = take(&mut weights, "model.t_embedder1.mlp.2.bias")?;
            model
                .timestep_embedder
                .mlp_in
                .copy_weight_from(&mlp0_w)
                .map_err(|e| anyhow!("t_embedder1.mlp.0.weight: {e}"))?;
            model
                .timestep_embedder
                .mlp_in
                .copy_bias_from(&mlp0_b)
                .map_err(|e| anyhow!("t_embedder1.mlp.0.bias: {e}"))?;
            model
                .timestep_embedder
                .mlp_out
                .copy_weight_from(&mlp2_w)
                .map_err(|e| anyhow!("t_embedder1.mlp.2.weight: {e}"))?;
            model
                .timestep_embedder
                .mlp_out
                .copy_bias_from(&mlp2_b)
                .map_err(|e| anyhow!("t_embedder1.mlp.2.bias: {e}"))?;
        }

        // 8) HiDream heads — final pixel head.
        {
            let w = take(&mut weights, "model.final_layer2.linear.weight")?;
            let b = take(&mut weights, "model.final_layer2.linear.bias")?;
            model
                .final_layer
                .linear
                .copy_weight_from(&w)
                .map_err(|e| anyhow!("final_layer2.linear.weight: {e}"))?;
            model
                .final_layer
                .linear
                .copy_bias_from(&b)
                .map_err(|e| anyhow!("final_layer2.linear.bias: {e}"))?;
        }

        // These are frozen base-model tensors. Keeping Linear::new's default
        // requires_grad=true wastes backward memory once resident LoRAs are
        // attached to the same path.
        freeze_resident_weights(&mut model);

        // 9) All resident keys must have been consumed.
        if !weights.is_empty() {
            let leftover: Vec<&String> = weights.keys().take(8).collect();
            bail!(
                "HiDream-O1 weight loader: {} unconsumed resident weight key(s) (loader bug); first few: {:?}",
                weights.len(),
                leftover
            );
        }

        log::info!(
            "[hidream_o1] Model weights loaded successfully ({} blocks streaming, resident shared loaded)",
            config.num_layers
        );
        Ok(model)
    }
}

fn freeze_resident_weights(model: &mut HiDreamO1Model) {
    model.embed_tokens.weight = model.embed_tokens.weight.clone().requires_grad_(false);
    if let Some(w) = model.norm.weight.take() {
        model.norm.weight = Some(w.requires_grad_(false));
    }
    freeze_linear(&mut model.bottleneck_patch_embed.proj1);
    freeze_linear(&mut model.bottleneck_patch_embed.proj2);
    freeze_linear(&mut model.timestep_embedder.mlp_in);
    freeze_linear(&mut model.timestep_embedder.mlp_out);
    freeze_linear(&mut model.final_layer.linear);
}

fn freeze_linear(linear: &mut flame_core::nn::Linear) {
    linear.weight = linear.weight.clone().requires_grad_(false);
    if let Some(bias) = linear.bias.take() {
        linear.bias = Some(bias.requires_grad_(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_layer_key_handles_all_per_layer_variants() {
        for (key, expected) in [
            ("model.language_model.layers.0.input_layernorm.weight", Some(0)),
            ("model.language_model.layers.13.self_attn.q_proj.weight", Some(13)),
            ("model.language_model.layers.35.mlp.down_proj.weight", Some(35)),
            ("model.language_model.embed_tokens.weight", None),
            ("model.language_model.norm.weight", None),
            ("model.x_embedder.proj1.weight", None),
            ("model.t_embedder1.mlp.0.weight", None),
            ("model.final_layer2.linear.weight", None),
            ("model.visual.blocks.0.attn.qkv.weight", None),
            ("lm_head.weight", None),
        ] {
            assert_eq!(classify_layer_key(key), expected, "key={key}");
        }
    }
}
