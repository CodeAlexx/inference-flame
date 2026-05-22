//! L2P DiT — config + struct scaffold (Wave 1 chunk 1).
//!
//! The DiT body is structurally identical to Z-Image NextDiT (dim=3840,
//! 30 layers, 30 heads, head_dim=128, RMSNorm, SwiGLU, 3-axis RoPE,
//! adaLN modulation with tanh gates). The differences live in the
//! patch embedder and the absence of a `FinalLayer`:
//!
//! - `patch_size: 16` (Z-Image base = 2)
//! - `in_channels: 3`  (pixel-space; Z-Image base = 16 VAE latent channels)
//! - The output head is the `MicroDiffusionModel` U-Net, not
//!   `FinalLayer + unpatchify`. That sits outside `L2pDiT` (it ships in
//!   a later chunk).
//! - `t_scale_inv: bool` — L2P's pipeline applies
//!   `t = (1000 - t_in) / 1000` and `out = -out` (flow-matching `v` ↔ `-v`).
//!   The flag is set on the config here for visibility; it is consumed
//!   by the pipeline wrapper in a later chunk, not by `L2pDiT::forward`.
//!
//! This file currently exposes only the config + struct scaffold +
//! constructors. The forward pass, transformer block, patchify, and
//! local decoder land in chunks 2-4.

use flame_core::attention::sdpa;
use flame_core::bf16_ops::{gate_residual_fused_bf16, rope_fused_bf16, swiglu_fused_bf16};
use flame_core::norm::rms_norm as norm_rms_norm;
use flame_core::ops::fused_inference::fused_rms_norm;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::lora::LoraStack;
use crate::models::l2p::local_decoder::MicroDiffusionModel;
use crate::models::l2p::rope::build_3d_rope;
use crate::offload::BlockLoader;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// L2P DiT configuration.
///
/// All numeric fields mirror `NextDiTConfig` (Z-Image base) EXCEPT
/// `patch_size`, `in_channels`, plus two L2P-only fields
/// (`out_channels`, `t_scale_inv`). Defaults target the 1024² inference
/// reference shape from the L2P repo's `inference.py`.
pub struct L2pDiTConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub num_noise_refiner: usize,
    pub num_context_refiner: usize,
    pub cap_feat_dim: usize,
    pub mlp_hidden: usize,
    pub min_mod: usize,
    pub t_embedder_hidden: usize,
    /// 16 for L2P (vs 2 for Z-Image base).
    pub patch_size: usize,
    /// 3 for L2P pixel-space (vs 16 for Z-Image base VAE latents).
    pub in_channels: usize,
    /// 3 — output channel count of the local decoder. Not consumed by
    /// `L2pDiT` itself yet; lets the weight loader sanity-check the
    /// `local_decoder.out_conv` shape in a later chunk.
    pub out_channels: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
    pub time_scale: f32,
    pub pad_tokens_multiple: usize,
    /// RMSNorm epsilon. **L2P uses 1e-5** (`ZImageDiT.__init__` default
    /// `norm_eps=1e-5`, applied to all `attention_norm*`, `ffn_norm*`,
    /// `cap_embedder.0`, `norm_q`, `norm_k`). NextDiT hardcoded 1e-6,
    /// which is INCORRECT for L2P parity — use this config value
    /// instead of a hardcoded literal everywhere you call `fused_rms_norm`.
    pub norm_eps: f32,
    /// Apply the L2P pipeline sign-flip + timestep inversion:
    ///   `t = (time_scale - t_in) / time_scale`
    ///   `out = -out`
    /// Default `true` for L2P. Consumed by the pipeline wrapper in a
    /// later chunk; `L2pDiT::forward` itself is sign-agnostic.
    pub t_scale_inv: bool,
}

impl Default for L2pDiTConfig {
    fn default() -> Self {
        Self {
            dim: 3840,
            num_heads: 30,
            head_dim: 128,
            num_layers: 30,
            num_noise_refiner: 2,
            num_context_refiner: 2,
            cap_feat_dim: 2560,
            mlp_hidden: 10240,
            min_mod: 256,
            t_embedder_hidden: 1024,
            patch_size: 16,
            in_channels: 3,
            out_channels: 3,
            axes_dims_rope: [32, 48, 48],
            rope_theta: 256.0,
            time_scale: 1000.0,
            pad_tokens_multiple: 32,
            norm_eps: 1e-5,
            t_scale_inv: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Model scaffold
// ---------------------------------------------------------------------------

/// L2P DiT model — Wave 1 chunk 1 scaffold only.
///
/// Field layout matches `zimage_nextdit::NextDiT`. The forward pass,
/// transformer blocks, patchify, weight loaders, and the local
/// `MicroDiffusionModel` U-Net land in later chunks.
pub struct L2pDiT {
    pub config: L2pDiTConfig,
    resident: HashMap<String, Tensor>,
    loader: Option<BlockLoader>,
    device: Arc<cudarc::driver::CudaDevice>,
    /// Optional runtime LoRA stack — applied at each `linear_no_bias` call.
    /// Base weights are never mutated.
    lora: Option<Arc<LoraStack>>,
    /// L2P U-Net pixel head — replaces `FinalLayer + unpatchify`.
    /// Constructed from the resident weight map; the chunk-4 weight
    /// loader is responsible for placing `local_decoder.*` keys in the
    /// map. If they're missing the constructor errors loudly.
    pub local_decoder: MicroDiffusionModel,
}

impl L2pDiT {
    /// Block-swap mode: blocks streamed from disk via mmap. Mirrors
    /// `NextDiT::new`.
    ///
    /// The U-Net pixel head (`local_decoder`) is built from `resident`,
    /// so all `local_decoder.*.weight/bias` keys must be present even
    /// when the rest of the DiT lives on disk. The chunk-4 weight
    /// loader is responsible for staging them. If any are missing this
    /// panics with the missing key name.
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        let loader = Some(BlockLoader::new(model_path, device.clone()));
        let local_decoder = MicroDiffusionModel::new(&resident, &device)
            .expect("MicroDiffusionModel: local_decoder.* weights missing or malformed");
        Self {
            config: L2pDiTConfig::default(),
            resident,
            loader,
            device,
            lora: None,
            local_decoder,
        }
    }

    /// All-resident mode: every weight on GPU, no disk I/O.
    /// Pre-transposes all 2D weight matrices `[out, in] -> [in, out]`
    /// IN PLACE so matmul never has to transpose during forward.
    /// Mirrors `NextDiT::new_resident`.
    ///
    /// The U-Net pixel head is built BEFORE the pre-transpose pass,
    /// because Conv2d weights are 4D (not 2D) and are not affected by
    /// the transpose — but more importantly, `MicroDiffusionModel::new`
    /// uses `Conv2d::copy_weight_from` which clones a separate copy, so
    /// the source tensors can later be transposed or modified without
    /// affecting the conv layer's owned copy. Errors loudly if any
    /// `local_decoder.*` weight is missing.
    pub fn new_resident(
        mut weights: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        // Build U-Net first (uses 4D conv weights — untouched by the
        // 2D pre-transpose pass that follows).
        let local_decoder = MicroDiffusionModel::new(&weights, &device)
            .expect("MicroDiffusionModel: local_decoder.* weights missing or malformed");

        let keys: Vec<String> = weights
            .keys()
            .filter(|k| k.ends_with(".weight"))
            .cloned()
            .collect();
        let mut transposed = 0;
        for key in &keys {
            let is_2d = weights
                .get(key)
                .map(|t| t.shape().dims().len() == 2)
                .unwrap_or(false);
            if is_2d {
                if let Some(t) = weights.remove(key) {
                    // permute() returns a strided view; gemm reads weights as
                    // contiguous row-major and IGNORES custom_strides, so a
                    // raw permute view becomes garbage on the kernel side.
                    // Materialize via .contiguous() so the in/out layout the
                    // kernel reads matches the post-transpose logical shape.
                    match t.permute(&[1, 0]).and_then(|v| v.contiguous()) {
                        Ok(tt) => {
                            weights.insert(key.clone(), tt);
                            transposed += 1;
                        }
                        Err(_) => {
                            weights.insert(key.clone(), t);
                        }
                    }
                }
            }
        }
        println!("    Pre-transposed {transposed} weight matrices");
        Self {
            config: L2pDiTConfig::default(),
            resident: weights,
            loader: None,
            device,
            lora: None,
            local_decoder,
        }
    }

    /// Attach a runtime LoRA stack. Mirrors `NextDiT::set_lora`.
    pub fn set_lora(&mut self, lora: Arc<LoraStack>) {
        self.lora = Some(lora);
    }
}

// All private fns in the impl block below are exercised by the forward
// pass (chunk 3) and weight-loader smoke (chunk 4). Until then they would
// trip `dead_code`. Mirroring NextDiT exactly — no `pub` until a caller
// exists.
#[allow(dead_code)]
impl L2pDiT {
    // -- BlockLoader helpers --------------------------------------------------

    fn load_block(&mut self, prefix: &str) -> Result<()> {
        if let Some(ref mut loader) = self.loader {
            loader.load_block(prefix)
        } else {
            Ok(())
        }
    }

    fn unload_block(&mut self) {
        if let Some(ref mut loader) = self.loader {
            loader.unload_block();
        }
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        if let Some(ref loader) = self.loader {
            loader.get(key, &self.resident)
        } else {
            self.resident.get(key).ok_or_else(|| {
                Error::InvalidInput(format!("Missing weight key: {key}"))
            })
        }
    }

    fn has_key(&self, key: &str) -> bool {
        if let Some(ref loader) = self.loader {
            loader.cache_contains(key)
        } else {
            self.resident.contains_key(key)
        }
    }

    // -- RMSNorm dispatch: autograd-aware during training ----------------------
    //
    // `fused_rms_norm` (fused_inference.rs) is inference-only: it launches a
    // single CUDA kernel but does NOT record an autograd op. This severs the
    // gradient chain wherever it sits between a LoRA-patched tensor and the
    // loss — specifically the post-attention `attention_norm2` and post-FFN
    // `ffn_norm2` calls strip `requires_grad` off the LoRA delta output,
    // preventing `gate_residual_fused_bf16` from recording its op and making
    // the entire transformer-block output appear disconnected from the loss.
    //
    // When autograd is recording (training), dispatch to
    // `flame_core::norm::rms_norm` which records `Op::RMSNorm` so backward
    // can traverse through the norm layer. At inference (no recording), use
    // the faster fused kernel.
    //
    // This mirrors the Z-Image trainer's `primitive_rms_norm` → `rms_norm`
    // pattern in `block_forward_standalone` vs `block_forward_iflame`.
    fn rms_norm_dispatch(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if flame_core::autograd::AutogradContext::is_recording() {
            // Autograd-aware path. `norm_rms_norm` records Op::RMSNorm so
            // backward can propagate through this norm layer.
            // normalized_shape = last dim of input = weight.elem_count().
            let norm_dim = weight.shape().elem_count();
            norm_rms_norm(input, &[norm_dim], Some(weight), eps)
        } else {
            // Inference-only fast path — single fused CUDA kernel, no tape.
            fused_rms_norm(input, weight, eps)
        }
    }

    // -- Linear helpers -------------------------------------------------------

    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let w_dims = weight.shape().dims();
        let x_dims = x.shape().dims().to_vec();
        let in_features = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();

        // In resident mode weights are pre-transposed [in, out], just matmul.
        // In block-swap mode weights are original [out, in], need transpose.
        let (wt, out_features) = if self.loader.is_none() {
            // Pre-transposed: [in, out]
            (weight, w_dims[1])
        } else {
            // Original: [out, in] -> transpose
            let out_features = w_dims[0];
            let x_2d = x.reshape(&[batch, in_features])?;
            let wt = weight.permute(&[1, 0])?;
            let out_2d = x_2d.matmul(&wt)?;
            let out_2d = match self.lora {
                Some(ref lora) => lora.apply(weight_key, &x_2d, out_2d)?,
                None => out_2d,
            };
            let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
            out_shape.push(out_features);
            return out_2d.reshape(&out_shape);
        };

        let x_2d = x.reshape(&[batch, in_features])?;
        let out_2d = x_2d.matmul(wt)?;
        let out_2d = match self.lora {
            Some(ref lora) => lora.apply(weight_key, &x_2d, out_2d)?,
            None => out_2d,
        };

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        out_2d.reshape(&out_shape)
    }

    fn linear_with_bias(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let out = self.linear_no_bias(x, weight_key)?;
        let bias = self.w(bias_key)?;
        let out_dims = out.shape().dims().to_vec();
        let batch: usize = out_dims[..out_dims.len() - 1].iter().product();
        let out_feat = *out_dims.last().unwrap();
        let bias_1d = bias.reshape(&[1, out_feat])?;
        let out_2d = out.reshape(&[batch, out_feat])?;
        let result_2d = out_2d.add(&bias_1d)?;
        result_2d.reshape(&out_dims)
    }

    // -- SwiGLU FFN (fused kernel) -------------------------------------------

    fn swiglu(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w1_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w1.weight"))?;
        let w3_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w3.weight"))?;
        // Fused silu(w1) * w3 — single kernel
        let hidden = swiglu_fused_bf16(&w1_out, &w3_out)?;
        self.linear_no_bias(&hidden, &format!("{prefix}.feed_forward.w2.weight"))
    }

    // -- Attention (fused RoPE + SDPA) ---------------------------------------

    fn joint_attention(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        prefix: &str,
    ) -> Result<Tensor> {
        // Sub-profiling: fires only on layer 0 when L2P_BLOCK_PROF=1.
        let prof = (prefix.ends_with(".0") || prefix.ends_with("layers.0"))
            && std::env::var("L2P_BLOCK_PROF").ok().as_deref() == Some("1");
        let dev = x.device().clone();
        let mark = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = dev.synchronize();
                eprintln!("[L2PATTN] {:<18} {:>6}ms", label, t.elapsed().as_millis());
            }
        };

        let dims = x.shape().dims().to_vec();
        let (b, seq, num_heads, head_dim) = (dims[0], dims[1], self.config.num_heads, self.config.head_dim);

        let t = std::time::Instant::now();
        // Fused QKV weight key. The chunk-4 weight loader pre-fuses
        // the L2P safetensors' separate `attention.to_q/to_k/to_v`
        // into this single `attention.qkv.weight`.
        let qkv = self.linear_no_bias(x, &format!("{prefix}.attention.qkv.weight"))?;
        let chunks = qkv.chunk(3, 2)?;
        let q = chunks[0].reshape(&[b, seq, num_heads, head_dim])?;
        let k = chunks[1].reshape(&[b, seq, num_heads, head_dim])?;
        let v = chunks[2].reshape(&[b, seq, num_heads, head_dim])?;
        mark("a.qkv_proj+chunk", t);

        let t = std::time::Instant::now();
        let q_w = self.w(&format!("{prefix}.attention.q_norm.weight"))?;
        let k_w = self.w(&format!("{prefix}.attention.k_norm.weight"))?;
        let q_flat = q.reshape(&[b * seq * num_heads, head_dim])?;
        let k_flat = k.reshape(&[b * seq * num_heads, head_dim])?;
        let q = self.rms_norm_dispatch(&q_flat, q_w, self.config.norm_eps)?.reshape(&[b, seq, num_heads, head_dim])?;
        let k = self.rms_norm_dispatch(&k_flat, k_w, self.config.norm_eps)?.reshape(&[b, seq, num_heads, head_dim])?;
        mark("b.qk_rmsnorm", t);

        let t = std::time::Instant::now();
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;
        mark("c.permute_qkv", t);

        let t = std::time::Instant::now();
        let half_d = head_dim / 2;
        let cos = rope_cos.reshape(&[1, 1, seq, half_d])?;
        let sin = rope_sin.reshape(&[1, 1, seq, half_d])?;
        let q = rope_fused_bf16(&q, &cos, &sin)?;
        let k = rope_fused_bf16(&k, &cos, &sin)?;
        mark("d.rope", t);

        let t = std::time::Instant::now();
        let out = sdpa(&q, &k, &v, None)?;
        mark("e.sdpa", t);

        let t = std::time::Instant::now();
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq, num_heads * head_dim])?;
        mark("f.permute_out", t);

        let t = std::time::Instant::now();
        let r = self.linear_no_bias(&out, &format!("{prefix}.attention.out.weight"));
        mark("g.out_proj", t);
        r
    }

    // -- Transformer block (fused kernels) -----------------------------------

    fn transformer_block(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        t_cond: Option<&Tensor>,
        prefix: &str,
    ) -> Result<Tensor> {
        // Section profiler: fires when L2P_BLOCK_PROF=1 and only on block 0.
        let prof = prefix.ends_with(".0") || prefix.ends_with("layers.0");
        let prof = prof && std::env::var("L2P_BLOCK_PROF").ok().as_deref() == Some("1");
        let dev = x.device().clone();
        let mark = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = dev.synchronize();
                eprintln!("[L2PBPROF] {:<22} {:>6}ms", label, t.elapsed().as_millis());
            }
        };
        let t_block = std::time::Instant::now();

        let t = std::time::Instant::now();
        let adaln_key = format!("{prefix}.adaLN_modulation.0.weight");
        let has_adaln = t_cond.is_some() && self.has_key(&adaln_key);

        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if has_adaln {
            let t_cond = t_cond.unwrap();
            let mod_out = self.linear_with_bias(
                t_cond,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &format!("{prefix}.adaLN_modulation.0.bias"),
            )?;
            let chunks = mod_out.chunk(4, mod_out.shape().dims().len() - 1)?;
            (Some(chunks[0].clone()), Some(chunks[1].clone()),
             Some(chunks[2].clone()), Some(chunks[3].clone()))
        } else {
            (None, None, None, None)
        };
        mark("1.adaln_mod", t);

        // --- Attention branch ---
        let t = std::time::Instant::now();
        let norm1_w = self.w(&format!("{prefix}.attention_norm1.weight"))?;
        let x_norm = self.rms_norm_dispatch(x, norm1_w, self.config.norm_eps)?;
        let x_norm = if let Some(ref scale) = scale_msa {
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = scale_unsq.add_scalar(1.0)?;
            x_norm.mul(&factor)?
        } else {
            x_norm
        };
        mark("2.rms1+scale_msa", t);

        let t = std::time::Instant::now();
        let attn_out = self.joint_attention(&x_norm, rope_cos, rope_sin, prefix)?;
        mark("3.joint_attention", t);

        let t = std::time::Instant::now();
        let norm2_w = self.w(&format!("{prefix}.attention_norm2.weight"))?;
        let attn_out = self.rms_norm_dispatch(&attn_out, norm2_w, self.config.norm_eps)?;
        let x_out = if let Some(ref gate) = gate_msa {
            let g = gate.tanh()?;
            gate_residual_fused_bf16(x, &g, &attn_out)?
        } else {
            x.add(&attn_out)?
        };
        mark("4.rms2+gate1", t);

        // --- FFN branch ---
        let t = std::time::Instant::now();
        let ffn_norm1_w = self.w(&format!("{prefix}.ffn_norm1.weight"))?;
        let ff_norm = self.rms_norm_dispatch(&x_out, ffn_norm1_w, self.config.norm_eps)?;
        let ff_norm = if let Some(ref scale) = scale_mlp {
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = scale_unsq.add_scalar(1.0)?;
            ff_norm.mul(&factor)?
        } else {
            ff_norm
        };
        mark("5.rms3+scale_mlp", t);

        let t = std::time::Instant::now();
        let ff_out = self.swiglu(&ff_norm, prefix)?;
        mark("6.swiglu", t);

        let t = std::time::Instant::now();
        let ffn_norm2_w = self.w(&format!("{prefix}.ffn_norm2.weight"))?;
        let ff_out = self.rms_norm_dispatch(&ff_out, ffn_norm2_w, self.config.norm_eps)?;
        let x_out = if let Some(ref gate) = gate_mlp {
            let g = gate.tanh()?;
            gate_residual_fused_bf16(&x_out, &g, &ff_out)?
        } else {
            x_out.add(&ff_out)?
        };
        mark("7.rms4+gate2", t);

        if prof {
            let _ = dev.synchronize();
            eprintln!("[L2PBPROF] TOTAL                 {:>6}ms", t_block.elapsed().as_millis());
        }

        Ok(x_out)
    }

    // -- Timestep embedder ---------------------------------------------------

    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        let freq_dim = self.config.min_mod;
        let half = freq_dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_vec()?;
        let batch = t_data.len();

        let mut emb_data = vec![0.0f32; batch * freq_dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                emb_data[b * freq_dim + i] = angle.cos();
                emb_data[b * freq_dim + half + i] = angle.sin();
            }
        }

        let emb = Tensor::from_vec_dtype(
            emb_data,
            Shape::from_dims(&[batch, freq_dim]),
            self.device.clone(),
            DType::BF16,
        )?;

        let h = self.linear_with_bias(&emb, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    // -- Caption embedder ----------------------------------------------------

    fn caption_embed(&self, cap_feats: &Tensor) -> Result<Tensor> {
        let norm_w = self.w("cap_embedder.0.weight")?;
        let normed = self.rms_norm_dispatch(cap_feats, norm_w, self.config.norm_eps)?;
        if self.has_key("cap_embedder.1.bias") {
            self.linear_with_bias(&normed, "cap_embedder.1.weight", "cap_embedder.1.bias")
        } else {
            self.linear_no_bias(&normed, "cap_embedder.1.weight")
        }
    }

    // -- Pad tokens ----------------------------------------------------------

    fn pad_to_multiple(
        &self,
        tokens: &Tensor,
        pad_token_key: &str,
        multiple: usize,
    ) -> Result<(Tensor, usize)> {
        let seq_len = tokens.shape().dims()[1];
        let pad_len = (multiple - (seq_len % multiple)) % multiple;
        if pad_len == 0 {
            return Ok((tokens.clone(), 0));
        }

        let b = tokens.shape().dims()[0];
        let dim = tokens.shape().dims()[2];
        let pad_token = self.w(pad_token_key)?;

        // Build padding: [1, pad_len, dim] then expand to [B, pad_len, dim]
        let pad_single = pad_token.reshape(&[1, 1, dim])?;
        let pad_expanded = pad_single.expand(&[1, pad_len, dim])?;
        let pad_batch = if b > 1 {
            pad_expanded.expand(&[b, pad_len, dim])?
        } else {
            pad_expanded
        };

        let result = Tensor::cat(&[tokens, &pad_batch], 1)?;
        Ok((result, pad_len))
    }

    // -- Patchify ------------------------------------------------------------

    /// L2P 16×16 pixel-space patchify.
    ///
    /// Input:  `[B, 3, H, W]`      (pixel-space, BF16)
    /// Output: `[B, ph*pw, 768]`   where `ph=H/16, pw=W/16`
    ///                             and `768 = 16*16*3 = p*p*C`.
    ///
    /// Identical reshape/permute pattern to `NextDiT::patchify`; only the
    /// `patch_size` and `in_channels` values differ. The output is fed into
    /// the `x_embedder` Linear (768 → dim=3840) — note the L2P
    /// `x_embedder` has `bias=True` (Python: `nn.Linear(..., bias=True)`),
    /// so the call site in `forward_inner` uses `linear_with_bias`.
    fn patchify16_pixel(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let p = self.config.patch_size; // 16
        let (ph, pw) = (h / p, w / p);

        let x = x.reshape(&[b, c, ph, p, pw, p])?;
        let x = x.permute(&[0, 2, 4, 3, 5, 1])?;
        x.reshape(&[b, ph * pw, p * p * c]).map(|t| (t, ph, pw))
    }

    // -- Full forward pass ---------------------------------------------------

    /// Full L2P DiT forward.
    ///
    /// - `x`: `[B, 3, H, W]` BF16 noisy pixel input. **Input must be
    ///   BF16.** The chunk-4 pipeline wrapper is responsible for casting
    ///   F32 noise to BF16 via `.to_dtype(DType::BF16)?` before invoking
    ///   this method (per PORT_SPEC §"Special / things to watch" #4 the
    ///   reference supplies F32; this port pushes the cast to the caller).
    /// - `timestep`: `[B]` — flow-matching normalized timestep `v ∈ [0, 1]`
    ///   per the FlowMatchScheduler convention. `forward_inner` applies
    ///   the L2P sign convention internally:
    ///   `t = (1.0 - v) * time_scale` (default 1000.0), then negates the
    ///   model output at the end. The chunk-4 pipeline must pass the
    ///   scheduler's normalized timestep value, NOT the raw integer step
    ///   or a `[0, 1000]` scale. **Do not re-negate the returned tensor —
    ///   `forward_inner` already does that.**
    /// - `cap_feats`: `[B, cap_seq, cap_feat_dim]` BF16 caption features
    ///
    /// Returns `[B, 3, H, W]` BF16 pixel-space prediction, with the
    /// sign-flip baked in (matches both Python L2P's `-model_output`
    /// and NextDiT's `mul_scalar(-1.0)`).
    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
    ) -> Result<Tensor> {
        self.forward_inner(x, timestep, cap_feats, None)
    }

    /// Forward with optional intermediate capture for parity testing.
    /// When `capture` is `Some`, named intermediates are inserted into
    /// the map at the same boundaries used by Z-Image's parity tooling,
    /// plus two L2P-specific captures (`feat_map`, `local_decoder_out`).
    pub fn forward_with_capture(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        self.forward_inner(x, timestep, cap_feats, Some(capture))
    }

    fn forward_inner(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
        mut capture: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // Helper: clone-and-insert into capture map. Materializes via
        // contiguous() so saved tensors are independent of any later
        // view rewrites.
        macro_rules! cap {
            ($name:expr, $tensor:expr) => {
                if let Some(c) = capture.as_deref_mut() {
                    let t = $tensor;
                    let owned = t.contiguous().unwrap_or_else(|_| t.clone());
                    c.insert($name.to_string(), owned);
                }
            };
        }

        // Dtype contract: chunk-4 pipeline must pre-cast F32 noise to BF16.
        // patchify16_pixel does only reshape/permute (no cast), and the
        // downstream linear/conv kernels require BF16. Surface the contract.
        debug_assert_eq!(
            x.dtype(),
            DType::BF16,
            "L2pDiT::forward_inner expects BF16 noisy pixel input — chunk-4 pipeline must cast F32→BF16 before calling"
        );

        let pad_mult = self.config.pad_tokens_multiple;

        // Keep the original noisy pixel input around for the U-Net head.
        // Tensor::clone is an Arc-bump on the storage, not a data copy.
        let noisy_pixel = x.clone();

        // L2P timestep convention: caller passes `v ∈ [0, 1]` from the
        // flow-matching scheduler. We map to the t_embedder's expected
        // scale here: `t = (1.0 - v) * time_scale`. Matches NextDiT's
        // pre-invert and L2P's `(time_scale - t_in) / time_scale * time_scale`
        // chain. The final output negation below completes the L2P
        // sign convention.
        let t_data = timestep.to_vec()?;
        let inv_data: Vec<f32> = t_data
            .iter()
            .map(|v| (1.0 - v) * self.config.time_scale)
            .collect();
        let t_scaled = Tensor::from_vec_dtype(
            inv_data,
            timestep.shape().clone(),
            self.device.clone(),
            DType::BF16,
        )?;
        let t_cond = self.timestep_embed(&t_scaled)?;
        cap!("t_emb", &t_cond);

        // L2P-specific patchify: 16×16 pixel patches → 768-dim flat
        // tokens.
        let (x_patches, ph, pw) = self.patchify16_pixel(x)?;

        // L2P `x_embedder` has bias (Python `nn.Linear(..., bias=True)`).
        // Keep the NextDiT-style has_key probe for safety; the L2P
        // safetensors always provides the bias so the `if` branch is
        // always taken.
        let x_emb = if self.has_key("x_embedder.bias") {
            self.linear_with_bias(&x_patches, "x_embedder.weight", "x_embedder.bias")?
        } else {
            self.linear_no_bias(&x_patches, "x_embedder.weight")?
        };
        cap!("x_after_embedder", &x_emb);
        let img_len = x_emb.shape().dims()[1];

        // Embed captions
        let c = self.caption_embed(cap_feats)?;
        cap!("cap_after_embedder", &c);

        // Pad caption and image to multiple of `pad_tokens_multiple`
        let (c, _) = self.pad_to_multiple(&c, "cap_pad_token", pad_mult)?;
        let cap_len = c.shape().dims()[1];
        let (x_emb, img_pad_len) = self.pad_to_multiple(&x_emb, "x_pad_token", pad_mult)?;

        // Build 3D RoPE
        let (rope_cos_full, rope_sin_full) =
            build_3d_rope(&self.device, &self.config, cap_len, ph, pw, img_pad_len)?;

        // Split RoPE for caption and image portions
        let rope_cos_cap = rope_cos_full.narrow(0, 0, cap_len)?;
        let rope_sin_cap = rope_sin_full.narrow(0, 0, cap_len)?;
        let img_seq = x_emb.shape().dims()[1];
        let rope_cos_img = rope_cos_full.narrow(0, cap_len, img_seq)?;
        let rope_sin_img = rope_sin_full.narrow(0, cap_len, img_seq)?;

        // Context refiner: text self-attention (unconditioned)
        let mut c = c;
        for i in 0..self.config.num_context_refiner {
            let prefix = format!("context_refiner.{i}");
            self.load_block(&prefix)?;
            c = self.transformer_block(&c, &rope_cos_cap, &rope_sin_cap, None, &prefix)?;
            self.unload_block();
            cap!(format!("context_refiner_{i}_out", i = i), &c);
        }

        // Noise refiner: image self-attention (conditioned)
        let mut x_emb = x_emb;
        for i in 0..self.config.num_noise_refiner {
            let prefix = format!("noise_refiner.{i}");
            self.load_block(&prefix)?;
            x_emb =
                self.transformer_block(&x_emb, &rope_cos_img, &rope_sin_img, Some(&t_cond), &prefix)?;
            self.unload_block();
            cap!(format!("noise_refiner_{i}_out", i = i), &x_emb);
        }

        // Concatenate text + image for main layers. Order [cap, image]
        // matches NextDiT.
        let mut xc = Tensor::cat(&[&c, &x_emb], 1)?;
        cap!("unified_initial", &xc);

        // Main transformer layers
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{i}");
            self.load_block(&prefix)?;
            xc = self.transformer_block(&xc, &rope_cos_full, &rope_sin_full, Some(&t_cond), &prefix)?;
            self.unload_block();
            cap!(format!("unified_after_layer_{:02}", i), &xc);
        }

        // -- L2P divergence from NextDiT starts here --------------------
        //
        // NextDiT:  final_layer(x_out) → unpatchify → mul_scalar(-1.0)
        // L2P:      reshape x_out to [B, dim, ph, pw] feat_map →
        //           local_decoder(noisy_pixel, feat_map) → mul_scalar(-1.0)
        //
        // No final_layer, no unpatchify. The U-Net produces direct
        // pixels.

        // Extract image tokens (skip text, remove padding)
        let x_out = xc.narrow(1, cap_len, img_len)?;

        // Reshape image tokens to feat_map: [B, img_len, dim]
        //   → [B, ph, pw, dim] → [B, dim, ph, pw]
        // Conv2d in the U-Net reads NCHW contig; the permute returns a
        // strided view so `.contiguous()` is required here.
        let b = x_out.shape().dims()[0];
        let dim = self.config.dim;
        let feat_map = x_out
            .reshape(&[b, ph, pw, dim])?
            .permute(&[0, 3, 1, 2])?
            .contiguous()?;
        cap!("feat_map", &feat_map);

        // U-Net head: pixel-space prediction
        let local_out = self.local_decoder.forward(&noisy_pixel, &feat_map)?;
        cap!("local_decoder_out", &local_out);

        // Sign-flip (matches both NextDiT's `mul_scalar(-1.0)` and
        // Python L2P's `model_output = -model_output`).
        local_out.mul_scalar(-1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Note: a real `timestep_embed` shape/dtype test needs a non-trivial
    // weight fixture (the t_embedder MLP weights at known [256,1024] and
    // [1024,256] plus biases). That wiring is more than ~30 lines and the
    // resulting test largely duplicates what the chunk-4 weight loader
    // smoke + full forward parity will exercise anyway. Skipping here per
    // chunk-2 brief; will land in chunk 3 alongside the full forward test.
}

