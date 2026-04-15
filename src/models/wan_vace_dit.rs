//! Wan2.1-VACE-14B DiT — pure Rust, flame-core + BlockOffloader.
//!
//! Port of `VaceWanModel` from VACE repo (`vace/models/wan/modules/model.py`).
//!
//! ## Architecture
//! ControlNet-style conditioning on top of Wan2.1 base DiT:
//! - **40 base blocks** (`blocks.{0-39}`) — same as Wan2.1 T2V
//! - **8 VACE blocks** (`vace_blocks.{0-7}`) — conditioning network at layers [0,2,4,6,8,10,12,14]
//! - `vace_patch_embedding`: Conv3d(96, 5120, k=(1,2,2), s=(1,2,2))
//! - VACE block 0 has `before_proj` (zero-init), all have `after_proj` (zero-init)
//! - Single model (no dual expert), `patch_embedding` takes 16 channels
//!
//! ## Execution flow
//! 1. Patchify noise → base embeddings [1, seq, 5120]
//! 2. Patchify VACE context → VACE embeddings [1, seq, 5120]
//! 3. Time/text embeddings (shared)
//! 4. Run 8 VACE blocks → produce skip connection hints
//! 5. Run 40 base blocks, adding hints at layers [0,2,4,...,14]
//! 6. Head → unpatchify → [16, F, H, W]
//!
//! ## Weight keys (VACE-specific, beyond standard Wan keys)
//! ```
//! vace_patch_embedding.{weight,bias}            [5120, 96, 1, 2, 2] / [5120]
//! vace_blocks.{i}.before_proj.{weight,bias}     [5120, 5120] (only block 0)
//! vace_blocks.{i}.after_proj.{weight,bias}      [5120, 5120]
//! vace_blocks.{i}.{self_attn,cross_attn,ffn,modulation,norm3}.*  (same as base blocks)
//! ```

use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use flame_diffusion::block_offload::BlockFacilitator;
use flame_diffusion::BlockOffloader;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BlockFacilitator for VACE blocks: `vace_blocks.{i}.*` → block i
// ---------------------------------------------------------------------------

struct VaceFacilitator {
    num_blocks: usize,
}

impl BlockFacilitator for VaceFacilitator {
    fn block_count(&self) -> usize { self.num_blocks }
    fn classify_key(&self, key: &str) -> Option<usize> {
        let rest = key.strip_prefix("vace_blocks.")?;
        rest.split('.').next()?.parse().ok()
    }
}

use crate::models::wan22_dit::{Wan22Config, Wan22Dit};

/// VACE layer mapping: VACE block i corresponds to base block at index 2*i.
const VACE_LAYERS: [usize; 8] = [0, 2, 4, 6, 8, 10, 12, 14];

pub struct WanVaceDit {
    /// The base Wan DiT (handles blocks, shared weights, RoPE, patchify/unpatchify)
    base: Wan22Dit,
    /// BlockOffloader for the 8 VACE conditioning blocks
    vace_offloader: BlockOffloader,
    /// VACE-specific shared weights
    vace_shared: HashMap<String, Tensor>,
    device: Arc<CudaDevice>,
}

impl WanVaceDit {
    pub fn load(
        checkpoint_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        // Load base DiT (handles blocks.{0-39} + shared weights)
        let base = Wan22Dit::load(checkpoint_path, device)?;

        // Load VACE blocks via separate BlockOffloader
        let vace_facilitator = VaceFacilitator { num_blocks: 8 };
        let vace_offloader = BlockOffloader::load(
            &[checkpoint_path],
            &vace_facilitator,
            device.clone(),
        )
        .map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader VACE: {e}")))?;

        // VACE shared weights
        let vace_prefixes = ["vace_patch_embedding."];
        let part = load_file_filtered(Path::new(checkpoint_path), device, |key| {
            vace_prefixes.iter().any(|p| key.starts_with(p))
        })?;
        let vace_shared: HashMap<String, Tensor> = part.into_iter().map(|(k, v)| {
            let v_bf16 = if v.dtype() != DType::BF16 {
                v.to_dtype(DType::BF16).unwrap_or(v)
            } else { v };
            (k, v_bf16)
        }).collect();

        log::info!(
            "[VACE] Loaded: {} VACE blocks, {} VACE shared weights",
            vace_offloader.block_count(),
            vace_shared.len()
        );

        Ok(Self {
            base,
            vace_offloader,
            vace_shared,
            device: device.clone(),
        })
    }

    /// Full VACE forward pass.
    ///
    /// Arguments:
    /// - `x`: noise latent [16, F, H, W]
    /// - `vace_context`: VACE conditioning [96, F, H, W] (z + mask concatenated)
    /// - `timestep`: scalar timestep (0-1000)
    /// - `context`: text embeddings [1, L, 4096] BF16
    /// - `seq_len`: padded sequence length
    /// - `context_scale`: scale for hint injection (default 1.0)
    pub fn forward(
        &mut self,
        x: &Tensor,              // [16, F, H, W]
        vace_context: &Tensor,   // [96, F, H, W]
        timestep: f32,
        context: &Tensor,        // [1, L, 4096]
        seq_len: usize,
        context_scale: f32,
    ) -> Result<Tensor> {
        let cfg = self.base.config().clone();
        let dim = cfg.dim;

        // ── Patchify noise (base path) ──
        let x_dims = x.shape().dims().to_vec();
        let (c_in, f_in, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let h_out = h_in / cfg.patch_size[1];
        let w_out = w_in / cfg.patch_size[2];
        let f_out = f_in / cfg.patch_size[0];
        let n_patches = f_out * h_out * w_out;
        let grid_sizes = (f_out, h_out, w_out);

        let patch_dim = c_in * cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2];
        let patched = self.base.patchify_public(x, f_in, h_in, w_in)?;
        let pe_w = self.base.shared_weight("patch_embedding.weight")?;
        let pe_b = self.base.shared_weight("patch_embedding.bias")?;
        let pe_w_flat = pe_w.reshape(&[dim, patch_dim])?;
        let patched_3d = patched.unsqueeze(0)?;
        let mut img = Wan22Dit::linear_bias_pub(&patched_3d, &pe_w_flat, pe_b)?;

        // Pad to seq_len
        if n_patches < seq_len {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[1, seq_len - n_patches, dim]),
                DType::BF16, self.device.clone(),
            )?;
            img = Tensor::cat(&[&img, &pad], 1)?;
        }

        // ── Patchify VACE context ──
        let vc_dims = vace_context.shape().dims().to_vec();
        let vc_c = vc_dims[0]; // 96
        let vc_patch_dim = vc_c * cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]; // 96*4=384
        let vc_patched = self.base.patchify_public(vace_context, f_in, h_in, w_in)?;
        let vpe_w = self.vace_shared.get("vace_patch_embedding.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing vace_patch_embedding.weight".into()))?;
        let vpe_b = self.vace_shared.get("vace_patch_embedding.bias")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing vace_patch_embedding.bias".into()))?;
        let vpe_w_flat = vpe_w.reshape(&[dim, vc_patch_dim])?;
        let vc_patched_3d = vc_patched.unsqueeze(0)?;
        let mut vace_emb = Wan22Dit::linear_bias_pub(&vc_patched_3d, &vpe_w_flat, vpe_b)?;

        if n_patches < seq_len {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[1, seq_len - n_patches, dim]),
                DType::BF16, self.device.clone(),
            )?;
            vace_emb = Tensor::cat(&[&vace_emb, &pad], 1)?;
        }

        // ── Time + text embeddings (reuse base) ──
        let (e, e0, txt) = self.base.compute_embeddings(timestep, context, seq_len)?;

        // ── Run VACE blocks → produce hints ──
        // VACE block 0: c = before_proj(vace_emb) + img, then run block, save skip
        // VACE block i>0: c = last from stack, run block, save skip
        let num_vace = self.vace_offloader.block_count(); // 8
        let mut hints: Vec<Tensor> = Vec::with_capacity(num_vace);

        self.vace_offloader.prefetch_block(0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("vace prefetch: {e}")))?;

        let mut c = vace_emb; // will be overwritten in block 0

        for vi in 0..num_vace {
            let vace_weights = self.vace_offloader.await_block(vi)
                .map_err(|e| flame_core::Error::InvalidInput(format!("vace await: {e}")))?;
            if vi + 1 < num_vace {
                self.vace_offloader.prefetch_block(vi + 1)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("vace prefetch: {e}")))?;
            }

            if vi == 0 {
                // before_proj(vace_emb) + img
                let bp_w = vace_weights.get("vace_blocks.0.before_proj.weight")
                    .ok_or_else(|| flame_core::Error::InvalidInput("Missing vace_blocks.0.before_proj.weight".into()))?;
                let bp_b = vace_weights.get("vace_blocks.0.before_proj.bias")
                    .ok_or_else(|| flame_core::Error::InvalidInput("Missing vace_blocks.0.before_proj.bias".into()))?;
                let projected = Wan22Dit::linear_bias_pub(&c, bp_w, bp_b)?;
                c = projected.add(&img)?;
            }

            // Run the VACE block (same architecture as base WanAttentionBlock)
            c = self.base.block_forward_pub(
                &c, &e0, &e, &txt, n_patches, grid_sizes, &vace_weights,
                vi, "vace_blocks",
            )?;

            // after_proj → skip connection hint
            let ap_w_key = format!("vace_blocks.{vi}.after_proj.weight");
            let ap_b_key = format!("vace_blocks.{vi}.after_proj.bias");
            let ap_w = vace_weights.get(&ap_w_key)
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing {ap_w_key}")))?;
            let ap_b = vace_weights.get(&ap_b_key)
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing {ap_b_key}")))?;
            let hint = Wan22Dit::linear_bias_pub(&c, ap_w, ap_b)?;
            hints.push(hint);

            log::info!("[VACE] VACE block {}/{}", vi + 1, num_vace);
        }

        // ── Run base blocks with hints ──
        let total_blocks = cfg.num_layers;
        self.base.swap_prefetch(0)?;

        for i in 0..total_blocks {
            let raw = self.base.swap_await(i)?;
            if i + 1 < total_blocks {
                self.base.swap_prefetch(i + 1)?;
            }

            img = self.base.block_forward_pub(
                &img, &e0, &e, &txt, n_patches, grid_sizes, &raw,
                i, "blocks",
            )?;

            // Add hint if this layer has one
            if let Some(vace_idx) = VACE_LAYERS.iter().position(|&l| l == i) {
                if vace_idx < hints.len() {
                    let scaled_hint = hints[vace_idx].mul_scalar(context_scale)?;
                    img = img.add(&scaled_hint)?;
                }
            }

            if i % 10 == 0 || i == total_blocks - 1 {
                log::info!("[VACE] Base block {}/{}", i + 1, total_blocks);
            }
        }

        // ── Head + unpatchify ──
        let out = self.base.head_forward(&img, &e, seq_len)?;
        let out_trimmed = out.narrow(1, 0, n_patches)?;
        self.base.unpatchify_public(&out_trimmed, grid_sizes)
    }
}
