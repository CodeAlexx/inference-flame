//! LoRA registry for HiDream-O1 decoder layers.
//!
//! ## Why a registry, not module-owned tensors
//!
//! The HiDream-O1 decoder doesn't own its weights — `decoder_forward_with_weights`
//! pulls them from a `HashMap<String, Tensor>` that the `BlockOffloader` builds
//! per-layer on-demand (see `decoder.rs:275-422`). A LoRA adapter that lives as
//! a `LoRALinear` module replacing a `Linear` field wouldn't fit; we'd have to
//! refactor 7 call sites × 36 layers and rewrite the offloader contract.
//!
//! Instead we keep LoRA tensors in this registry, keyed by model logical
//! module paths. Decoder-layer adapters use `"layers.{i}.{suffix}"`; resident
//! O1 heads use keys such as `"x_embedder.proj1"` and `"final_layer2.linear"`.
//!
//! ## Initialization (PEFT convention)
//!
//! - `A` (lora_a): `[rank, Cin]`, small Gaussian (std 1e-4).
//! - `B` (lora_b): `[Cout, rank]`, zeros.
//! - `scale = alpha / rank` (set per-adapter).
//!
//! With `B = 0`, the initial residual `(x A^T) B^T = 0`, so a freshly
//! initialized registry leaves the model output bit-identical to the no-LoRA
//! forward (self-consistency gate C1 per `hidream_o1_trainer_analysis.md` §10).
//!
//! ## Target modules
//!
//! The transformer-only parity target is the 7 decoder linears in each
//! language layer. EDV2 enables the O1 pixel/timestep heads by default because
//! known-good public O1 LoRAs include them:
//! `x_embedder.{proj1,proj2}`, `t_embedder1.mlp.{0,2}`, `final_layer2.linear`.
//!
//! Defaults: `rank = 32`, `alpha = 32` — matches
//! `train_lora_hidream_48.yaml:26-27` (`network.linear`, `network.linear_alpha`).
//!
//! | Suffix              | Cin    | Cout  | Notes                           |
//! |---------------------|--------|-------|----------------------------------|
//! | `self_attn.q_proj`  | 4096   | 4096  | `num_attention_heads * head_dim` |
//! | `self_attn.k_proj`  | 4096   | 1024  | `num_kv_heads * head_dim` (GQA)  |
//! | `self_attn.v_proj`  | 4096   | 1024  | same as k_proj                   |
//! | `self_attn.o_proj`  | 4096   | 4096  | merged heads → hidden            |
//! | `mlp.gate_proj`     | 4096   | 12288 | SwiGLU gate                      |
//! | `mlp.up_proj`       | 4096   | 12288 | SwiGLU up                        |
//! | `mlp.down_proj`     | 12288  | 4096  | SwiGLU down                      |
//!
//! Concrete shapes above are for the 8B variant (`hidden=4096`,
//! `intermediate=12288`, `num_attention_heads=32`, `num_kv_heads=8`,
//! `head_dim=128`). Use `default_target_suffixes()` to enumerate them.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use flame_core::parameter::Parameter;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};

use super::HiDreamO1Config;

/// One LoRA adapter (A, B, scale).
///
/// `a`: `[rank, Cin]`, `requires_grad = true`.
/// `b`: `[Cout, rank]`, `requires_grad = true`.
/// `scale`: typically `alpha / rank` (PEFT convention).
///
/// `a` and `b` are held as `Parameter` so that the optimizer's `set_data`
/// writes during `step()` are observable to subsequent forward passes via
/// [`LoraAdapter::a_tensor`] / [`LoraAdapter::b_tensor`]. Earlier (M1) this
/// stored raw `Tensor`s; that version pinned the initial state and silently
/// no-op'd training because the decoder kept reading the pre-step values.
#[derive(Clone)]
pub struct LoraAdapter {
    pub a: Parameter,
    pub b: Parameter,
    pub scale: f32,
}

impl LoraAdapter {
    /// Clone the current `A` tensor for use in a forward pass. The clone is
    /// cheap (an Arc bump on the device buffer plus an id copy) and carries
    /// `requires_grad=true` so backward sees the parameter.
    pub fn a_tensor(&self) -> Result<Tensor> {
        self.a.tensor()
    }
    pub fn b_tensor(&self) -> Result<Tensor> {
        self.b.tensor()
    }
}

/// Per-(layer_idx, module_suffix) keyed registry of LoRA adapters.
///
/// Key format: `"layers.{i}.{suffix}"` where `suffix` is e.g.
/// `"self_attn.q_proj"`. This is the convention `decoder_forward_with_weights`
/// uses when calling `get`.
#[derive(Clone)]
pub struct LoraRegistry {
    pub adapters: HashMap<String, LoraAdapter>,
    pub rank: usize,
    pub alpha: f32,
}

/// The 7 standard LoRA target suffixes for HiDream-O1 decoder layers.
///
/// Source: every `nn.Linear` inside edv2-reference's `Qwen3VLTextDecoderLayer`
/// (`qwen3_vl_transformers.py:486-598` — `self_attn.{q,k,v,o}_proj` and
/// `mlp.{gate,up,down}_proj`), restricted to `transformer_block_names =
/// ["layers"]` as declared by `HidreamO1Model.get_transformer_block_names`
/// (`hidream_o1_model.py:527`).
pub fn default_target_suffixes() -> &'static [&'static str] {
    &[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
}

/// Non-decoder HiDream-O1 linears. These are compatibility targets; the
/// reference training code only includes them when `transformer_only` is false.
pub fn default_resident_target_keys() -> &'static [&'static str] {
    &[
        "x_embedder.proj1",
        "x_embedder.proj2",
        "t_embedder1.mlp.0",
        "t_embedder1.mlp.2",
        "final_layer2.linear",
    ]
}

/// Substrings edv2-reference refuses to LoRA-adapt for O1
/// (`train_lora_hidream_48.yaml`'s `network_kwargs.ignore_if_contains`).
///
/// The static target list above already excludes all three, but keep the
/// blacklist here so any future dynamic target enumerator preserves
/// edv2-reference parity.
pub const IGNORE_IF_CONTAINS: &[&str] = &["lm_head", "patch_embed", "visual"];

/// Compute `(Cin, Cout)` for a given target suffix on this config.
///
/// Returns `None` for unrecognized suffixes.
pub fn shape_for_suffix(cfg: &HiDreamO1Config, suffix: &str) -> Option<(usize, usize)> {
    let hidden = cfg.hidden_size;
    let q_out = cfg.num_attention_heads * cfg.head_dim;
    let kv_out = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    match suffix {
        "self_attn.q_proj" => Some((hidden, q_out)),
        "self_attn.k_proj" => Some((hidden, kv_out)),
        "self_attn.v_proj" => Some((hidden, kv_out)),
        "self_attn.o_proj" => Some((q_out, hidden)),
        "mlp.gate_proj" => Some((hidden, inter)),
        "mlp.up_proj" => Some((hidden, inter)),
        "mlp.down_proj" => Some((inter, hidden)),
        _ => None,
    }
}

/// Compute `(Cin, Cout)` for a registry key.
pub fn shape_for_key(cfg: &HiDreamO1Config, key: &str) -> Option<(usize, usize)> {
    if let Some(rest) = key.strip_prefix("layers.") {
        let dot = rest.find('.')?;
        return shape_for_suffix(cfg, &rest[dot + 1..]);
    }
    let patch_dim = cfg.patch_size * cfg.patch_size * cfg.patch_in_channels;
    match key {
        "x_embedder.proj1" => Some((patch_dim, cfg.bottleneck_dim)),
        "x_embedder.proj2" => Some((cfg.bottleneck_dim, cfg.hidden_size)),
        "t_embedder1.mlp.0" => Some((cfg.timestep_freq_dim, cfg.hidden_size)),
        "t_embedder1.mlp.2" => Some((cfg.hidden_size, cfg.hidden_size)),
        "final_layer2.linear" => Some((cfg.hidden_size, patch_dim)),
        _ => None,
    }
}

/// Add `scale * ((input @ A.T) @ B.T)` to an already-computed base linear.
pub fn add_lora_residual(base: Tensor, input: &Tensor, adapter: &LoraAdapter) -> Result<Tensor> {
    let a = adapter.a_tensor()?;
    let b = adapter.b_tensor()?;
    validate_lora_compute_dtype("add_lora_residual", &a, &b)?;
    let residual = if a.dtype() == DType::F32 {
        // The reference LoRA path keeps adapter weights in F32, casts the
        // activation into the LoRA branch dtype, then casts the residual back.
        let input_f32 = input.to_dtype(DType::F32)?;
        let a_t = a.transpose()?.contiguous()?;
        let b_t = b.transpose()?.contiguous()?;
        let xa = input_f32.matmul(&a_t)?;
        let xab = xa.matmul(&b_t)?;
        xab.mul_scalar(adapter.scale)?
    } else {
        let a_t = a.transpose()?;
        let b_t = b.transpose()?;
        let xa = input.matmul(&a_t)?;
        let xab = xa.matmul(&b_t)?;
        xab.mul_scalar(adapter.scale)?
    };
    let residual = if residual.dtype() != base.dtype() {
        residual.to_dtype(base.dtype())?
    } else {
        residual
    };
    base.add(&residual)
}

fn validate_lora_compute_dtype(context: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    if a.dtype() != b.dtype() {
        return Err(Error::InvalidInput(format!(
            "{context}: lora_a and lora_b must have the same dtype, got {:?} and {:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    match a.dtype() {
        DType::BF16 | DType::F32 => Ok(()),
        other => Err(Error::InvalidInput(format!(
            "{context}: unsupported LoRA dtype {other:?}; expected BF16 or F32"
        ))),
    }
}

impl LoraRegistry {
    /// Build a fresh registry with PEFT-style init (A ~ N(0, 1e-4), B = 0).
    ///
    /// `num_layers` controls the per-layer fan-out: every entry in
    /// `target_suffixes` gets an adapter for `i in 0..num_layers`.
    ///
    /// Tensors are placed on `device` in BF16 with `requires_grad = true`.
    /// Training code that needs reference parity should call
    /// [`Self::new_with_dtype_and_resident`] with `DType::F32`.
    pub fn new(
        cfg: &HiDreamO1Config,
        rank: usize,
        alpha: f32,
        target_suffixes: &[&str],
        seed: u64,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_dtype_and_resident(
            cfg,
            rank,
            alpha,
            target_suffixes,
            seed,
            device,
            DType::BF16,
            true,
        )
    }

    pub fn new_with_dtype(
        cfg: &HiDreamO1Config,
        rank: usize,
        alpha: f32,
        target_suffixes: &[&str],
        seed: u64,
        device: &Arc<CudaDevice>,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_dtype_and_resident(
            cfg,
            rank,
            alpha,
            target_suffixes,
            seed,
            device,
            dtype,
            true,
        )
    }

    pub fn new_with_dtype_and_resident(
        cfg: &HiDreamO1Config,
        rank: usize,
        alpha: f32,
        target_suffixes: &[&str],
        seed: u64,
        device: &Arc<CudaDevice>,
        dtype: DType,
        include_resident_targets: bool,
    ) -> Result<Self> {
        match dtype {
            DType::BF16 | DType::F32 => {}
            other => {
                return Err(Error::InvalidInput(format!(
                    "LoraRegistry::new: unsupported LoRA dtype {other:?}; expected BF16 or F32"
                )));
            }
        }
        let scale = alpha / (rank as f32);
        let mut adapters = HashMap::new();
        if include_resident_targets {
            for (k, key) in default_resident_target_keys().iter().enumerate() {
                let (cin, cout) = shape_for_key(cfg, key).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "LoraRegistry::new: unknown resident target key {key:?}"
                    ))
                })?;
                let adapter_seed = seed.wrapping_add(10_000).wrapping_add(k as u64 * 17);
                use rand::{rngs::StdRng, Rng, SeedableRng};
                let bound = 1.0 / (cin as f32).sqrt();
                let mut rng = StdRng::seed_from_u64(adapter_seed);
                let a_data: Vec<f32> = (0..rank * cin)
                    .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * bound)
                    .collect();
                let a = Tensor::from_vec(a_data, Shape::from_dims(&[rank, cin]), device.clone())?
                    .to_dtype(dtype)?
                    .requires_grad_(true);
                let b = Tensor::zeros_dtype(
                    Shape::from_dims(&[cout, rank]),
                    dtype,
                    device.clone(),
                )?
                .requires_grad_(true);
                adapters.insert(
                    (*key).to_string(),
                    LoraAdapter {
                        a: Parameter::new(a),
                        b: Parameter::new(b),
                        scale,
                    },
                );
            }
        }
        for layer_idx in 0..cfg.num_layers {
            for (k, suffix) in target_suffixes.iter().enumerate() {
                let (cin, cout) = shape_for_suffix(cfg, suffix).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "LoraRegistry::new: unknown target suffix {suffix:?}"
                    ))
                })?;
                // Distinct seed per adapter so A-init isn't replicated.
                let adapter_seed = seed
                    .wrapping_add((layer_idx as u64) * 131)
                    .wrapping_add(k as u64 * 17);
                // PEFT/Kaiming uniform init on A — matches torch nn.Linear
                // default and edv2-reference's PEFT LoRA init. B is zero, so the
                // initial residual is identically zero regardless of A's
                // magnitude. Prior init (std=1e-4 Gaussian) was ~84× under
                // Kaiming magnitude and caused a step-1 loss spike (see
                // docs/o1_strict_parity_report.md). Matches
                // `eridiffusion-core::LoRALinear::new` pattern used by chroma
                // and other trainers.
                use rand::{rngs::StdRng, Rng, SeedableRng};
                let bound = 1.0 / (cin as f32).sqrt();
                let mut rng = StdRng::seed_from_u64(adapter_seed);
                let a_data: Vec<f32> = (0..rank * cin)
                    .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * bound)
                    .collect();
                let a_raw = Tensor::from_vec(
                    a_data,
                    Shape::from_dims(&[rank, cin]),
                    device.clone(),
                )?
                .to_dtype(dtype)?;
                let a = a_raw;
                let b = Tensor::zeros_dtype(
                    Shape::from_dims(&[cout, rank]),
                    dtype,
                    device.clone(),
                )?;
                let a = a.requires_grad_(true);
                let b = b.requires_grad_(true);
                let a_param = Parameter::new(a);
                let b_param = Parameter::new(b);
                let key = format!("layers.{layer_idx}.{suffix}");
                adapters.insert(
                    key,
                    LoraAdapter {
                        a: a_param,
                        b: b_param,
                        scale,
                    },
                );
            }
        }
        Ok(Self {
            adapters,
            rank,
            alpha,
        })
    }

    /// Look up an adapter by `(layer_idx, module_suffix)`.
    pub fn get(&self, layer_idx: usize, suffix: &str) -> Option<&LoraAdapter> {
        let key = format!("layers.{layer_idx}.{suffix}");
        self.adapters.get(&key)
    }

    /// Look up a non-decoder adapter by registry key.
    pub fn get_global(&self, key: &str) -> Option<&LoraAdapter> {
        self.adapters.get(key)
    }

    /// Iterate `(key, &A_param, &B_param)` triples — feed to an optimizer's
    /// `step(&[Parameter])` after flattening with [`Self::parameters`].
    pub fn iter_trainable(&self) -> impl Iterator<Item = (&String, &Parameter, &Parameter)> {
        self.adapters.iter().map(|(k, v)| {
            // Defensive: catch any future regression that drops requires_grad
            // on A/B (e.g., an accidental no-grad cast or clone via a
            // *_no_grad helper). LoRA tensors must remain trainable.
            debug_assert!(
                v.a.requires_grad() && v.b.requires_grad(),
                "LoraRegistry adapter '{k}' lost requires_grad on A or B"
            );
            (k, &v.a, &v.b)
        })
    }

    /// Flatten to a single `Vec<Parameter>` suitable for AdamW etc. Deterministic
    /// order: sorted by adapter key so resume + grad-flow diagnostics are stable.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut keys: Vec<&String> = self.adapters.keys().collect();
        keys.sort();
        let mut out = Vec::with_capacity(keys.len() * 2);
        for k in keys {
            let ad = self.adapters.get(k).unwrap();
            out.push(ad.a.clone());
            out.push(ad.b.clone());
        }
        out
    }

    /// `(label, &Parameter)` pairs, for `flame_core::diagnostics::assert_grad_flow`.
    /// Labels are `"layers.{i}.{suffix}.lora_A"` / `".lora_B"`.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut keys: Vec<&String> = self.adapters.keys().collect();
        keys.sort();
        let mut out = Vec::with_capacity(keys.len() * 2);
        for k in keys {
            let ad = self.adapters.get(k).unwrap();
            out.push((format!("{k}.lora_A"), ad.a.clone()));
            out.push((format!("{k}.lora_B"), ad.b.clone()));
        }
        out
    }

    /// Number of adapters in this registry.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Load a registry from an edv2-reference-style or older Rust safetensors file.
    ///
    /// Accepted key layouts (per adapter):
    ///   - PEFT canonical (with `default` adapter infix — BUG-2 fix):
    ///     `base_model.model.model.language_model.layers.{i}.{suffix}.lora_A.default.weight`
    ///   - Legacy canonical (no infix):
    ///     `base_model.model.model.language_model.layers.{i}.{suffix}.lora_A.weight`
    ///   - Reference O1:
    ///     `diffusion_model.language_model.layers.{i}.{suffix}.lora_A.weight`
    ///     `diffusion_model.x_embedder.proj1.lora_A.weight`
    ///   - Short:
    ///     `layers.{i}.{suffix}.lora_A.weight` (or `.lora_A.default.weight`)
    ///
    /// `rank`/`alpha` are inferred from the first matching shard's `A` tensor
    /// (`A: [rank, Cin]`) and used to set the per-adapter `scale = alpha/rank`.
    /// `alpha` defaults to `rank as f32` if the saver did not stamp it (we don't
    /// have a header convention yet — this matches PEFT's default).
    pub fn from_safetensors(
        path: &Path,
        cfg: &HiDreamO1Config,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::from_safetensors_with_dtype(path, cfg, device, DType::BF16)
    }

    pub fn from_safetensors_with_dtype(
        path: &Path,
        cfg: &HiDreamO1Config,
        device: &Arc<CudaDevice>,
        dtype: DType,
    ) -> Result<Self> {
        match dtype {
            DType::BF16 | DType::F32 => {}
            other => {
                return Err(Error::InvalidInput(format!(
                    "from_safetensors: unsupported LoRA dtype {other:?}; expected BF16 or F32"
                )));
            }
        }
        let raw = flame_core::serialization::load_file(path, device)?;
        // Accept edv2-reference O1 (`diffusion_model.*.lora_A.weight`), our older
        // PEFT-canonical wrapper (`base_model.model.model.language_model.*`),
        // and short internal keys.
        let old_prefix = "base_model.model.model.language_model.";
        let suffix_a_default = ".lora_A.default.weight";
        let suffix_b_default = ".lora_B.default.weight";
        let suffix_a_legacy = ".lora_A.weight";
        let suffix_b_legacy = ".lora_B.weight";

        // Probe which A-suffix is present (prefer PEFT canonical).
        let has_default_infix = raw.keys().any(|k| k.ends_with(suffix_a_default));
        let (suffix_a, suffix_b) = if has_default_infix {
            (suffix_a_default, suffix_b_default)
        } else {
            (suffix_a_legacy, suffix_b_legacy)
        };

        let mut adapters: HashMap<String, LoraAdapter> = HashMap::new();
        let mut detected_rank: Option<usize> = None;
        for (k, _) in raw.iter() {
            if !k.ends_with(suffix_a) {
                continue;
            }
            let prefix_key = k.trim_end_matches(suffix_a);
            let key = normalize_loaded_key(prefix_key, old_prefix).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "from_safetensors: unsupported HiDream-O1 LoRA key prefix {prefix_key}"
                ))
            })?;
            let a_full = k.clone();
            let b_full = format!("{}{}", k.trim_end_matches(suffix_a), suffix_b);
            let a_t = raw
                .get(&a_full)
                .ok_or_else(|| {
                    Error::InvalidInput(format!("from_safetensors: missing A tensor {a_full}"))
                })?
                .to_dtype(dtype)?
                .requires_grad_(true);
            let b_t = raw
                .get(&b_full)
                .ok_or_else(|| {
                    Error::InvalidInput(format!("from_safetensors: missing B tensor {b_full}"))
                })?
                .to_dtype(dtype)?
                .requires_grad_(true);
            let a_dims = a_t.shape().dims();
            if a_dims.len() != 2 {
                return Err(Error::InvalidInput(format!(
                    "from_safetensors: A tensor {a_full} must be 2-D, got {:?}",
                    a_dims
                )));
            }
            let rank = a_dims[0];
            detected_rank.get_or_insert(rank);
            adapters.insert(
                key,
                LoraAdapter {
                    a: Parameter::new(a_t),
                    b: Parameter::new(b_t),
                    scale: 1.0, // overwritten below
                },
            );
        }
        let rank = detected_rank.ok_or_else(|| {
            Error::InvalidInput(format!(
                "from_safetensors: no `*{suffix_a}` keys in {}",
                path.display()
            ))
        })?;
        let alpha = rank as f32; // header-less default; same as PEFT
        let scale = alpha / (rank as f32);
        for v in adapters.values_mut() {
            v.scale = scale;
        }
        // Sanity-check: every adapter shape matches what `shape_for_suffix`
        // would have allocated for this config. Catches loading a wrong-model
        // checkpoint or a stale key naming convention.
        for (k, v) in adapters.iter() {
            if let Some((cin, cout)) = shape_for_key(cfg, k) {
                let a_dims = v.a.shape();
                let b_dims = v.b.shape();
                if a_dims.dims() != [rank, cin] || b_dims.dims() != [cout, rank] {
                    return Err(Error::InvalidInput(format!(
                        "from_safetensors: shape mismatch for {k}: \
                         A={:?} B={:?}, expected A=[{},{}] B=[{},{}]",
                        a_dims.dims(),
                        b_dims.dims(),
                        rank,
                        cin,
                        cout,
                        rank
                    )));
                }
            }
        }
        Ok(Self {
            adapters,
            rank,
            alpha,
        })
    }

    /// Save to the generic HiDream-O1 LoRA layout:
    ///   `diffusion_model.language_model.layers.{i}...lora_A.weight`
    ///   `diffusion_model.x_embedder.proj1.lora_A.weight`
    ///
    /// `from_safetensors` still accepts the older PEFT wrapper format so
    /// previously-trained Rust checkpoints remain loadable.
    pub fn save_safetensors(&self, path: &Path) -> Result<()> {
        self.save_safetensors_with_export_scale(path, 1.0)
    }

    /// Save a weights-only LoRA, optionally scaling the exported B matrices.
    ///
    /// HiDream-O1 is unusually sensitive to full-strength style LoRAs in the
    /// 512/1024 dev sampler. Scaling `B` at export leaves the training update
    /// path unchanged while producing an ordinary LoRA file that external
    /// loaders can use without an inference-only strength knob.
    pub fn save_safetensors_with_export_scale(&self, path: &Path, export_scale: f32) -> Result<()> {
        if !export_scale.is_finite() || export_scale <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "save_safetensors: export_scale must be finite and > 0, got {export_scale}"
            )));
        }
        let _guard = flame_core::autograd::AutogradContext::no_grad();
        let mut out: HashMap<String, Tensor> = HashMap::with_capacity(self.adapters.len() * 2);
        let mut keys: Vec<&String> = self.adapters.keys().collect();
        keys.sort();
        for k in keys {
            let ad = self.adapters.get(k).unwrap();
            let save_key = save_key_for_registry_key(k);
            let a_key = format!("{save_key}.lora_A.weight");
            let b_key = format!("{save_key}.lora_B.weight");
            out.insert(a_key, ad.a.tensor()?);
            let b = ad.b.tensor()?;
            let b = if (export_scale - 1.0).abs() > f32::EPSILON {
                b.mul_scalar(export_scale)?
            } else {
                b
            };
            out.insert(b_key, b);
        }
        let mut meta = HashMap::new();
        meta.insert("ss_training_comment".to_string(), "edv2 trainer".to_string());
        meta.insert("modelspec.architecture".to_string(), "hidream-o1/lora".to_string());
        meta.insert("modelspec.base_model".to_string(), "hidream_o1".to_string());
        meta.insert("ss_network_dim".to_string(), self.rank.to_string());
        meta.insert("ss_network_alpha".to_string(), self.alpha.to_string());
        meta.insert("edv2.export_scale".to_string(), export_scale.to_string());
        flame_core::serialization::save_tensors_with_metadata(&out, &meta, path)?;
        Ok(())
    }
}

fn normalize_loaded_key(prefix_key: &str, old_prefix: &str) -> Option<String> {
    let mut key = prefix_key;
    for p in [
        old_prefix,
        "base_model.model.model.",
        "diffusion_model.",
        "transformer.model.",
        "transformer.",
        "model.",
    ] {
        if let Some(rest) = key.strip_prefix(p) {
            key = rest;
            break;
        }
    }
    if let Some(rest) = key.strip_prefix("language_model.") {
        key = rest;
    }
    let supported = key.starts_with("layers.")
        || key == "x_embedder.proj1"
        || key == "x_embedder.proj2"
        || key == "t_embedder1.mlp.0"
        || key == "t_embedder1.mlp.2"
        || key == "final_layer2.linear";
    supported.then(|| key.to_string())
}

fn save_key_for_registry_key(key: &str) -> String {
    if key.starts_with("layers.") {
        format!("diffusion_model.language_model.{key}")
    } else {
        format!("diffusion_model.{key}")
    }
}
