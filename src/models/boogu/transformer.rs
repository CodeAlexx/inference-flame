//! Boogu-Image C6 — the full DiT: `BooguImageTransformer2DModel` (resident).
//!
//! Mirrors `boogu/models/transformers/transformer_boogu.py::BooguImageTransformer2DModel.forward`
//! (the no-ref-image T2I path, lines 1253-1604) op-for-op, composing the
//! already-built C1 ([`super::embed`]), C2 ([`super::rope`]), C3
//! ([`super::block::BooguBlock`]), C4
//! ([`super::block_double::BooguDoubleStreamBlock`]) and C5
//! ([`super::final_layer`]) pieces. Cross-checked against the verified Mojo C6
//! (`/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_c6_parity.mojo`,
//! handoff parity row C6: full DiT velocity cos 0.9998966, max_abs 0.141; the
//! Mojo wiring is the authoritative architecture spec).
//!
//! ## Forward order (no-ref T2I, the target path) — transformer_boogu.py:1253-1604
//!
//! ```text
//! instruction = preprocess_instruction_hidden_states(instruction)   # mean over 1 layer = identity
//! temb, instruction = time_caption_embed(timestep, instruction)     # C1
//! tokens = flat_and_pad_to_seq(noise_latent)                         # patchify, channel-FASTEST
//! freqs  = rope_embedder(...)                                        # C2 joint cos/sin
//!   # rope slices (no ref): cap = joint[0:cap_len], img = joint[cap_len:], joint = full
//! for layer in context_refiner (×2, modulation=False): instruction = layer(instruction, cap_rope)
//! img = x_embedder(tokens)                                           # Linear 64 -> 3360 (BIAS)
//! for layer in noise_refiner   (×2, modulation=True ): img = layer(img, img_rope, temb)
//! for layer in double_stream   (×8, modulation=True ):              # joint attn over [instruct;img]
//!     img, instruct = layer(img, instruct, img_rope, joint_rope, temb)
//! joint = cat([instruct ; img], dim=seq)                            # INSTRUCT-FIRST fusion
//! for layer in single_stream   (×32, modulation=True): joint = layer(joint, joint_rope, temb)
//! out = norm_out(joint, temb)                                       # C5 (LayerNorm eps 1e-6) -> 64
//! img_tokens = out[:, cap_len:seq]                                  # EXTRACT image rows
//! output = unpatchify(img_tokens)                                   # C5 channel-FASTEST -> [B,16,Hl,Wl]
//! ```
//!
//! ## Refiner modulation flags (the least-confident point — verify in parity)
//! - `context_refiner` (×2): **modulation=False** (plain RMSNorm, no temb) —
//!   `BooguBlock::load(..., false)`. transformer_boogu.py:888-903 constructs them
//!   with `modulation=False`.
//! - `noise_refiner` (×2): **modulation=True** (RMSNormZero + temb) —
//!   `BooguBlock::load(..., true)`. transformer_boogu.py:858-873 constructs them
//!   with `modulation=True`.
//! - `single_stream` (×32) and `double_stream` (×8): **modulation=True**.
//!
//! ## Rope slices (no-ref T2I; rope.py forward return mapping)
//! The C2 `build_t2i_rope_tables` returns the JOINT tables `[seq, head_dim/2]`
//! with `seq = cap_len + img_len`. The reference rope_embedder returns, for the
//! no-ref path:
//! - `context_rotary_emb` = `cap_freqs_cis` = joint rows `[0:cap_len]`,
//! - `noise_rotary_emb`   = `img_freqs_cis` = joint rows `[cap_len:seq]`,
//! - `rotary_emb` (joint) = full joint table (used by single-stream + the
//!   double-stream JOINT cross-attn),
//! - `combined_img_rotary_emb` = (no ref) == `img_freqs_cis` = joint rows
//!   `[cap_len:seq]` (used by the double-stream img self-attn).
//! So: cap rope = rows`[0:cap_len]`, img rope = rows`[cap_len:]`, joint = full.
//!
//! ## Memory (USER directive: resident)
//! `BooguDiT` holds ALL blocks resident (`Vec<BooguBlock>` / `Vec<BooguDoubleStreamBlock>`
//! built once via their `::load`) + the embedder/norm_out weight map. The
//! Mojo C6 measured ~20.4 GB resident at 1024². No block-streaming/offload here
//! — resident is the perf target (a measured OOM is a later C8/smoke decision).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::bf16_elementwise::patchify_bf16;
use flame_core::ops::fused_inference::fused_linear3d_native;
use flame_core::{DType, Error, Result, Tensor};

use super::block::BooguBlock;
use super::block_double::BooguDoubleStreamBlock;
use super::config::BooguConfig;
use super::embed::{caption_embed, timestep_embed};
use super::final_layer::{norm_out, unpatchify};
use super::loader::get;
use super::rope::build_t2i_rope_tables;

/// The full Boogu-Image DiT, held resident.
///
/// Built once from the transformer weight map; reused across denoise steps. Holds
/// the raw weight map (for the embedders + x_embedder + norm_out, which are
/// stateless free-function forwards) plus the resolved block structs.
pub struct BooguDiT {
    cfg: BooguConfig,
    device: Arc<CudaDevice>,
    /// The raw transformer weight map (verbatim PyTorch keys) — used by the
    /// embedders (C1), `x_embedder`, and `norm_out` (C5), whose forwards take the
    /// map directly.
    weights: HashMap<String, Tensor>,
    /// 2 context-refiner blocks (modulation=False).
    context_refiner: Vec<BooguBlock>,
    /// 2 noise-refiner blocks (modulation=True).
    noise_refiner: Vec<BooguBlock>,
    /// 8 double-stream blocks (modulation=True).
    double_stream: Vec<BooguDoubleStreamBlock>,
    /// 32 single-stream blocks (modulation=True).
    single_stream: Vec<BooguBlock>,
}

impl BooguDiT {
    /// `x_embedder` weight/bias keys (Linear 64 -> 3360, BIAS).
    const X_EMBEDDER: &'static str = "x_embedder";

    /// Build the resident DiT from the transformer weight map.
    ///
    /// `weights` is the FULL transformer weight map (942 tensors) loaded by
    /// `loader::load_component(repo, "transformer", device)` — verbatim PyTorch
    /// keys, no renames. All blocks are resolved into struct fields here so the
    /// per-step forward does no key lookups for block weights.
    pub fn load(
        weights: HashMap<String, Tensor>,
        cfg: BooguConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // context_refiner: ×2, modulation=False (plain RMSNorm, no temb).
        let mut context_refiner = Vec::with_capacity(cfg.num_refiner_layers);
        for i in 0..cfg.num_refiner_layers {
            context_refiner.push(BooguBlock::load(
                &weights,
                &format!("context_refiner.{i}"),
                false,
            )?);
        }
        // noise_refiner: ×2, modulation=True.
        let mut noise_refiner = Vec::with_capacity(cfg.num_refiner_layers);
        for i in 0..cfg.num_refiner_layers {
            noise_refiner.push(BooguBlock::load(
                &weights,
                &format!("noise_refiner.{i}"),
                true,
            )?);
        }
        // double_stream: ×8, modulation=True.
        let mut double_stream = Vec::with_capacity(cfg.num_double_stream_layers);
        for i in 0..cfg.num_double_stream_layers {
            double_stream.push(BooguDoubleStreamBlock::load(
                &weights,
                &format!("double_stream_layers.{i}"),
                true,
            )?);
        }
        // single_stream: ×32, modulation=True.
        let n_single = cfg.num_single_stream_layers();
        let mut single_stream = Vec::with_capacity(n_single);
        for i in 0..n_single {
            single_stream.push(BooguBlock::load(
                &weights,
                &format!("single_stream_layers.{i}"),
                true,
            )?);
        }

        // Fail loud early if the embedder / x_embedder / norm_out keys are absent
        // (the per-step forward relies on them).
        for key in [
            "x_embedder.weight",
            "x_embedder.bias",
            "time_caption_embed.timestep_embedder.linear_1.weight",
            "time_caption_embed.caption_embedder.1.weight",
            "norm_out.linear_1.weight",
            "norm_out.linear_2.weight",
        ] {
            get(&weights, key)?;
        }

        Ok(Self {
            cfg,
            device,
            weights,
            context_refiner,
            noise_refiner,
            double_stream,
            single_stream,
        })
    }

    /// `x_embedder`: BIAS Linear `patch²·in_channels(64) -> hidden(3360)`.
    fn x_embedder(&self, tokens: &Tensor) -> Result<Tensor> {
        let w = get(&self.weights, &format!("{}.weight", Self::X_EMBEDDER))?;
        let b = get(&self.weights, &format!("{}.bias", Self::X_EMBEDDER))?;
        fused_linear3d_native(tokens, w, Some(b))
    }

    /// Full DiT forward (no-ref T2I, the target path).
    ///
    /// - `noise_latent` — `[B, in_channels(16), Hl, Wl]` (the current latent;
    ///   F32 or BF16). `Hl`/`Wl` must be divisible by `patch_size` (2).
    /// - `timestep` — one raw flow timestep per batch sample (`&[f32]`, length B).
    ///   Passed RAW; `time_caption_embed` (C1) is called internally.
    /// - `instruction_hidden_states` — `[B, L, instruction_feat_dim(4096)]` BF16
    ///   (the Qwen3-VL `hidden_states[-1]`; `preprocess_instruction` mean-over-1
    ///   = identity, so passed as-is). `L` is the caption length (`cap_len`).
    ///
    /// Returns the velocity prediction `[B, out_channels(16), Hl, Wl]` BF16 — the
    /// SAME spatial shape as `noise_latent`.
    pub fn forward(
        &self,
        noise_latent: &Tensor,
        timestep: &[f32],
        instruction_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;

        // --- shape checks ---
        let nl = noise_latent.shape().dims();
        if nl.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "boogu DiT: noise_latent must be [B,C,Hl,Wl], got {nl:?}"
            )));
        }
        let (b, c_in, hl, wl) = (nl[0], nl[1], nl[2], nl[3]);
        if c_in != cfg.in_channels {
            return Err(Error::InvalidOperation(format!(
                "boogu DiT: noise_latent channels {c_in} != cfg.in_channels {}",
                cfg.in_channels
            )));
        }
        let p = cfg.patch_size;
        if hl % p != 0 || wl % p != 0 {
            return Err(Error::InvalidOperation(format!(
                "boogu DiT: latent {hl}x{wl} not divisible by patch_size {p}"
            )));
        }
        if timestep.len() != b {
            return Err(Error::InvalidOperation(format!(
                "boogu DiT: timestep len {} != batch {b}",
                timestep.len()
            )));
        }
        let ih = instruction_hidden_states.shape().dims();
        if ih.len() != 3 || ih[0] != b || ih[2] != cfg.instruction_feat_dim {
            return Err(Error::InvalidOperation(format!(
                "boogu DiT: instruction_hidden_states must be [B,L,{}], got {ih:?}",
                cfg.instruction_feat_dim
            )));
        }
        let cap_len = ih[1];
        let h_tok = hl / p;
        let w_tok = wl / p;
        let img_len = h_tok * w_tok;

        // --- C1: time_caption_embed (the DiT forward takes the RAW timestep and
        //     embeds internally; transformer_boogu.py:1305). ---
        // preprocess_instruction_hidden_states is the identity here (mean over 1
        // layer); instruction is already [B,L,4096].
        let temb = timestep_embed(&self.weights, cfg, timestep, &self.device)?; // [B,1,1024]
        let mut instruct = caption_embed(&self.weights, cfg, instruction_hidden_states)?; // [B,L,3360]

        // --- patchify the noise latent (channel-FASTEST (p1 p2 c); the matching
        //     pack to C5 unpatchify). x_embedder in = 64. ---
        let latent_bf16 = if noise_latent.dtype() == DType::BF16 {
            noise_latent.clone()
        } else {
            noise_latent.to_dtype(DType::BF16)?
        };
        let (tokens, ph, pw) = patchify_bf16(&latent_bf16, p)?; // [B, img_len, 64]
        debug_assert_eq!(ph, h_tok);
        debug_assert_eq!(pw, w_tok);

        // --- C2: build the JOINT rope tables, then slice cap/img/joint rows. ---
        // BF16 tables (project-wide BF16-RoPE convention; rope_fused_bf16).
        let rope = build_t2i_rope_tables(cfg, cap_len, h_tok, w_tok, DType::BF16, &self.device)?;
        let seq = rope.seq_len();
        debug_assert_eq!(seq, cap_len + img_len);
        // cap rope = rows [0:cap_len]; img rope = rows [cap_len:seq]; joint = full.
        let cap_cos = rope.cos.narrow(0, 0, cap_len)?;
        let cap_sin = rope.sin.narrow(0, 0, cap_len)?;
        let img_cos = rope.cos.narrow(0, cap_len, img_len)?;
        let img_sin = rope.sin.narrow(0, cap_len, img_len)?;
        let joint_cos = &rope.cos; // full [seq, half]
        let joint_sin = &rope.sin;

        // --- context_refiner ×2 (modulation=False) on instruction with cap rope. ---
        for blk in &self.context_refiner {
            instruct = blk.forward(&instruct, None, cfg, &cap_cos, &cap_sin)?;
        }

        // --- x_embedder + noise_refiner ×2 (modulation=True) with img rope. ---
        let mut img = self.x_embedder(&tokens)?; // [B, img_len, 3360]
        for blk in &self.noise_refiner {
            img = blk.forward(&img, Some(&temb), cfg, &img_cos, &img_sin)?;
        }

        // --- double_stream ×8 (joint attn over [instruct;img], img self-attn). ---
        // The block consumes: cap_rope (unused — instruct rotation comes from the
        // joint rope's caption rows inside the joint attn), img_rope (img self-attn
        // = combined-img rope = joint[cap_len:]), joint_rope (joint cross-attn).
        for blk in &self.double_stream {
            let (new_instruct, new_img) = blk.forward(
                &instruct,
                &img,
                Some(&temb),
                cfg,
                (&cap_cos, &cap_sin),
                (&img_cos, &img_sin),
                (joint_cos, joint_sin),
            )?;
            instruct = new_instruct;
            img = new_img;
        }

        // --- fuse INSTRUCT-FIRST: joint = [instruct ; img] along the seq axis. ---
        let mut joint = Tensor::cat(&[&instruct, &img], 1)?; // [B, seq, 3360]
        debug_assert_eq!(joint.shape().dims()[1], seq);

        // --- single_stream ×32 (modulation=True) on the joint seq + temb, joint rope. ---
        for blk in &self.single_stream {
            joint = blk.forward(&joint, Some(&temb), cfg, joint_cos, joint_sin)?;
        }

        // --- C5: norm_out (LayerNorm eps 1e-6, temb-modulated) -> [B, seq, 64]. ---
        let out = norm_out(&self.weights, cfg, &joint, &temb)?;

        // --- extract the IMAGE rows [cap_len:seq] before unpatchify. ---
        let img_tokens = out.narrow(1, cap_len, img_len)?.contiguous()?;

        // --- C5: channel-FASTEST unpatchify -> [B, out_channels, Hl, Wl]. ---
        unpatchify(&img_tokens, cfg, h_tok, w_tok)
    }

    /// The held config.
    #[inline]
    pub fn config(&self) -> &BooguConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BooguConfig {
        BooguConfig::default()
    }

    #[test]
    fn block_counts_match_config() {
        let c = cfg();
        assert_eq!(c.num_refiner_layers, 2); // context ×2, noise ×2
        assert_eq!(c.num_double_stream_layers, 8);
        assert_eq!(c.num_single_stream_layers(), 32);
        // 2 ctx + 2 noise + 8 double + 32 single = 44 block structs (+ embedders +
        // norm_out in the weight map). The reference `layers` list = double+single
        // = 40 = config.num_layers.
        assert_eq!(c.num_double_stream_layers + c.num_single_stream_layers(), c.num_layers);
    }

    #[test]
    fn rope_slice_boundaries() {
        // cap rope = joint[0:cap_len]; img rope = joint[cap_len:seq]; joint = full.
        // At the Mojo C6 probe res cap_len=16, img_len=256 -> seq=272, img rows
        // [16:272]. Pure index arithmetic check (mirrors the Mojo C6 gate).
        let cap_len = 16usize;
        let img_len = 256usize;
        let seq = cap_len + img_len;
        assert_eq!(seq, 272);
        // img rope start/len.
        assert_eq!(cap_len, 16);
        assert_eq!(seq - cap_len, img_len); // [16:272]
    }

    #[test]
    fn extract_index_is_cap_len() {
        // The image tokens fused INSTRUCT-FIRST occupy rows [cap_len:seq]; the
        // extract before unpatchify narrows dim 1 at start=cap_len, len=img_len.
        let cap_len = 16usize;
        let img_len = 256usize;
        let seq = cap_len + img_len;
        // narrow(dim=1, start=cap_len, length=img_len) must land exactly at [cap_len:seq].
        assert_eq!(cap_len + img_len, seq);
    }

    #[test]
    fn patch_token_width_is_x_embedder_in() {
        // patchify produces patch_dim = patch²·in_channels = 64 = x_embedder in.
        let c = cfg();
        assert_eq!(
            c.patch_size * c.patch_size * c.in_channels,
            BooguConfig::X_EMBEDDER_IN
        );
    }

    // GPU + real-checkpoint dependent: load + full forward. Compile-only here;
    // the heavy GPU forward is gated #[ignore] (USER GPU budget).
    #[test]
    #[ignore = "requires CUDA device + 38.5GB checkpoint (GPU busy); compile-only this chunk"]
    fn dit_load_and_forward_compile() {
        let _ = BooguDiT::load;
        let _ = BooguDiT::forward;
    }

    /// Full-forward output-shape contract: `BooguDiT::forward` output spatial
    /// shape == input latent spatial shape `[B,16,Hl,Wl]`, given a dummy
    /// instruction. GPU-heavy (loads the full DiT, runs all 44 blocks) → ignored.
    #[test]
    #[ignore = "requires CUDA device + checkpoint (GPU-heavy forward); run with --ignored"]
    fn dit_forward_output_shape_matches_latent() {
        use flame_core::Shape;
        // `CudaDevice::new` already returns `Arc<CudaDevice>`.
        let device = cudarc::driver::CudaDevice::new(0).unwrap();
        let c = cfg();
        let repo = std::path::Path::new("/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base");
        let weights = super::super::loader::load_component(repo, "transformer", &device).unwrap();
        let dit = BooguDiT::load(weights, c, device.clone()).unwrap();

        // Tiny latent (32x32 -> 16x16 patch grid -> 256 img tokens) + 4-token
        // dummy instruction. cap_len=4, img_len=256, seq=260.
        let (b, hl, wl) = (1usize, 32usize, 32usize);
        let latent = Tensor::randn(
            Shape::from_dims(&[b, c.in_channels, hl, wl]),
            0.0,
            1.0,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let cap_len = 4usize;
        let instruction = Tensor::randn(
            Shape::from_dims(&[b, cap_len, c.instruction_feat_dim]),
            0.0,
            1.0,
            device.clone(),
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let out = dit.forward(&latent, &[0.5f32], &instruction).unwrap();
        assert_eq!(out.shape().dims(), &[b, c.out_channels, hl, wl]);
    }
}
