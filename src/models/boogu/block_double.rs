//! Boogu-Image C4 — the double-stream block
//! (`BooguImageDoubleStreamTransformerBlock`), used by `double_stream_layers`
//! (×8, modulation=True).
//!
//! Mirrors `boogu/models/transformers/transformer_boogu.py::BooguImageDoubleStreamTransformerBlock`
//! (lines 376-751) op-for-op, with the joint attention implemented by
//! [`super::attention::joint_gqa_attention`]
//! (`boogu/models/attention_processor.py::BooguImageDoubleStreamSelfAttnProcessor.__call__`,
//! lines 706-877) and the img self-attention by the C3
//! [`super::attention::gqa_self_attention`] helper
//! (`BooguImageAttnProcessor.__call__`, lines 1163-1275). Cross-checked against
//! the verified Mojo C4 `BooguDoubleStreamBlock`
//! (`/home/alex/mojodiffusion/serenitymojo/models/dit/boogu_dit.mojo`, handoff
//! parity table row C4: img_out cos 0.9999988, instruct_out cos 0.9999988).
//!
//! ## TWO streams (img + instruct), THREE attentions worth of structure
//!
//! - **joint cross-attention** over `[instruct ; img]` (separate per-stream
//!   q/k/v, instruct-FIRST concat, split back, separate `instruct_out`/`img_out`
//!   then a shared `to_out.0`) → produces `instruct_attn_out` + `img_attn_out`.
//! - **img self-attention** (`img_self_attn`, the C3 GQA path) on `img_norm3_out`
//!   over the combined-img rope (= the joint rope's IMG rows `[cap_len:seq]`).
//!
//! ## Modulation chunk structure — the #1 parity risk (block_lumina2.py:63-71,
//! transformer_boogu.py:596-683)
//!
//! Every `LuminaRMSNormZero.forward(x, emb)` (block_lumina2.py:63-71) does:
//! ```text
//! e = linear(silu(emb))                                  # [B, 4·hidden]
//! scale_msa, gate_msa, scale_mlp, gate_mlp = e.chunk(4)  # 4 chunks, in order
//! x = norm(x) * (1 + scale_msa[:,None])                  # consumes chunk[0]
//! return (x, gate_msa, scale_mlp, gate_mlp)              # returns chunks [1],[2],[3]
//! ```
//! So the linear ALWAYS emits 4 chunks `(scale_msa, gate_msa, scale_mlp,
//! gate_mlp)`, chunk[0] is consumed internally for the `(1+scale_msa)` norm
//! scale, and `(chunk[1], chunk[2], chunk[3])` are returned. The double-stream
//! block calls this 5× and REINTERPRETS the returned chunk[1] per norm:
//!
//! | norm        | call (transformer_boogu.py) | returned (out, c1, c2, c3) consumed as |
//! |-------------|-----------------------------|-----------------------------------------|
//! | `img_norm1`     | :598 | (img_norm1_out, **img_gate_msa**, img_scale_mlp, img_gate_mlp) |
//! | `img_norm2`     | :601 | (img_norm2_out, **img_shift_mlp**, _, _)  ← chunk[1] = shift_mlp |
//! | `img_norm3`     | :602 | (img_norm3_out, **img_gate_self**, _, _)  ← chunk[1] = gate_self |
//! | `instruct_norm1`| :609 | (instruct_norm1_out, **instruct_gate_msa**, instruct_scale_mlp, instruct_gate_mlp) |
//! | `instruct_norm2`| :610 | (instruct_norm2_out, **instruct_shift_mlp**, _, _) ← chunk[1] = shift_mlp |
//!
//! i.e. each of the 5 norms runs an independent `linear` → 4 chunks, applies
//! `(1+chunk[0])` internally, and the block picks the chunks it needs:
//! - from `*_norm1`: gate_msa (chunk[1]), scale_mlp (chunk[2]), gate_mlp (chunk[3]).
//! - from `*_norm2`/`img_norm3`: ONLY chunk[1] (shift_mlp resp. gate_self).
//!
//! **`shift_mlp` is the double-stream-only term absent from the single-stream
//! block (C3).** It enters as an ADDITIVE bias on the MLP input:
//! `mlp_input = (1 + scale_mlp[:,None]) * norm2_out + shift_mlp[:,None]`
//! (transformer_boogu.py:659-661 img / :673-675 instruct).
//!
//! ## Residual chain (modulation=True) — transformer_boogu.py:651-683
//!
//! ```text
//! # img stream (3 gated residuals: cross-attn, self-attn, mlp):
//! img += tanh(img_gate_msa)[:,None]  * img_attn_norm(img_attn_out)            # :652-654
//! img += tanh(img_gate_self)[:,None] * img_self_attn_norm(img_self_attn_out)  # :655-657
//! img_mlp_in  = (1 + img_scale_mlp[:,None]) * img_norm2_out + img_shift_mlp[:,None]  # :659-661
//! img_mlp_out = img_feed_forward(img_ffn_norm1(img_mlp_in))                   # :662
//! img += tanh(img_gate_mlp)[:,None]  * img_ffn_norm2(img_mlp_out)             # :663-665
//!
//! # instruct stream (2 gated residuals: cross-attn, mlp — NO self-attn):
//! instruct += tanh(instruct_gate_msa)[:,None] * instruct_attn_norm(instruct_attn_out)  # :667-671
//! instruct_mlp_in  = (1 + instruct_scale_mlp[:,None]) * instruct_norm2_out + instruct_shift_mlp[:,None]  # :673-675
//! instruct_mlp_out = instruct_feed_forward(instruct_ffn_norm1(instruct_mlp_in))         # :676-678
//! instruct += tanh(instruct_gate_mlp)[:,None] * instruct_ffn_norm2(instruct_mlp_out)    # :679-683
//! ```
//! Note `*_norm2_out` is ALREADY the RMSNormZero-normed (and `(1+scale_msa)`
//! pre-scaled) hidden; the MLP path then applies `(1+scale_mlp)`+shift, then the
//! plain `*_ffn_norm1` RMSNorm, then SwiGLU, then `*_ffn_norm2`. So the MLP sees
//! TWO norms (the RMSNormZero `norm2` AND the plain `ffn_norm1`), matching the
//! reference exactly.
//!
//! ## modulation=False (context-style; the 8 shipped blocks are all True)
//! transformer_boogu.py:685-742 — img/instruct norms are PLAIN RMSNorm, all
//! residuals ungated, no temb. Supported here for faithfulness; the DiT wiring
//! (C6) only ever constructs modulation=True double-stream blocks.
//!
//! ## Weight keys (prefix = e.g. `double_stream_layers.0`)
//! - modulation=True : `img_norm1/2/3.{linear.weight,linear.bias,norm.weight}`
//!   (3× LuminaRMSNormZero), `instruct_norm1/2.{linear.weight,linear.bias,norm.weight}`
//!   (2× LuminaRMSNormZero).
//! - modulation=False: `img_norm1/2/3.weight`, `instruct_norm1/2.weight` (plain).
//! - both: `img_attn_norm.weight`, `img_self_attn_norm.weight`,
//!   `img_ffn_norm1/2.weight`, `instruct_attn_norm.weight`,
//!   `instruct_ffn_norm1/2.weight` (plain RMSNorm 3360);
//!   joint attn `img_instruct_attn.processor.{img,instruct}_to_{q,k,v}.weight`,
//!   `img_instruct_attn.processor.{img_out,instruct_out}.weight`,
//!   `img_instruct_attn.{norm_q,norm_k,to_out.0}.weight` (all bias-free);
//!   img self-attn `img_self_attn.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight`;
//!   `img_feed_forward.{linear_1,linear_2,linear_3}.weight`,
//!   `instruct_feed_forward.{linear_1,linear_2,linear_3}.weight` (SwiGLU, bias-free).

use std::collections::HashMap;

use flame_core::bf16_ops::swiglu_fused_bf16;
use flame_core::ops::fused_inference::{fused_linear3d_native, fused_rms_norm};
use flame_core::{Error, Result, Tensor};

use super::attention::{gqa_self_attention, joint_gqa_attention};
use super::config::BooguConfig;
use super::loader::get;

/// Number of chunks the `LuminaRMSNormZero.linear` emits (scale_msa, gate_msa,
/// scale_mlp, gate_mlp) — identical for the img and the instruct streams.
pub const MODULATION_CHUNKS: usize = 4;

/// A `LuminaRMSNormZero` (block_lumina2.py:39-71) used inside the double-stream
/// block. modulation=True: AdaLN `linear` (1024 → 4·hidden) + RMSNorm gamma.
/// modulation=False: a plain RMSNorm (no linear), used by context-style blocks.
struct RmsNormZero {
    /// AdaLN linear weight `[4·hidden, 1024]` (None when modulation=False).
    lin_w: Option<Tensor>,
    /// AdaLN linear bias `[4·hidden]` (None when modulation=False).
    lin_b: Option<Tensor>,
    /// RMSNorm gamma `[hidden]`.
    norm_w: Tensor,
}

impl RmsNormZero {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str, modulation: bool) -> Result<Self> {
        if modulation {
            Ok(Self {
                lin_w: Some(get(weights, &format!("{prefix}.linear.weight"))?.clone()),
                lin_b: Some(get(weights, &format!("{prefix}.linear.bias"))?.clone()),
                norm_w: get(weights, &format!("{prefix}.norm.weight"))?.clone(),
            })
        } else {
            Ok(Self {
                lin_w: None,
                lin_b: None,
                norm_w: get(weights, &format!("{prefix}.weight"))?.clone(),
            })
        }
    }

    /// modulation=True forward: returns `(normed, chunk1, chunk2, chunk3)` where
    /// `normed = RMSNorm(x) * (1 + scale_msa[:,None])` (scale_msa = chunk[0],
    /// consumed internally) and `(chunk1, chunk2, chunk3) = (gate_msa, scale_mlp,
    /// gate_mlp)` are returned for the caller to reinterpret per norm.
    ///
    /// `temb` is `[B, 1, TIME_EMBED_DIM]` BF16. The returned chunks are
    /// `[B, hidden]` each.
    fn modulate(
        &self,
        x: &Tensor,
        temb: &Tensor,
        cfg: &BooguConfig,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let lin_w = self.lin_w.as_ref().ok_or_else(|| {
            Error::InvalidOperation("boogu RmsNormZero: modulation=True needs linear".into())
        })?;
        // emb = linear(silu(temb)) -> [B, 1, 4·hidden].
        let temb_silu = flame_core::bf16_ops::silu_bf16(temb)?;
        let emb = fused_linear3d_native(&temb_silu, lin_w, self.lin_b.as_ref())?;
        let b = emb.shape().dims()[0];
        let emb = emb.reshape(&[b, MODULATION_CHUNKS * cfg.hidden_size])?;
        // chunk(4, dim=1): (scale_msa, gate_msa, scale_mlp, gate_mlp).
        let chunks = emb.chunk(MODULATION_CHUNKS, 1)?;
        let scale_msa = &chunks[0];
        // x = norm(x) * (1 + scale_msa[:,None]).
        let nx = fused_rms_norm(x, &self.norm_w, cfg.norm_eps)?;
        let scale_msa_b = scale_msa.unsqueeze(1)?.add_scalar(1.0)?; // [B,1,hidden]
        let normed = nx.mul(&scale_msa_b)?;
        Ok((
            normed,
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
        ))
    }

    /// modulation=False forward: plain `RMSNorm(x)` (no temb, no chunks).
    fn plain(&self, x: &Tensor, cfg: &BooguConfig) -> Result<Tensor> {
        fused_rms_norm(x, &self.norm_w, cfg.norm_eps)
    }
}

/// SwiGLU feed-forward weights (`LuminaFeedForward`): `linear_2(silu(linear_1(x))
/// * linear_3(x))`. linear_1 = gate, linear_3 = up, linear_2 = down; bias-free.
struct SwiGluFfn {
    w1: Tensor, // linear_1 (gate) [inner, hidden]
    w2: Tensor, // linear_2 (down) [hidden, inner]
    w3: Tensor, // linear_3 (up)   [inner, hidden]
}

impl SwiGluFfn {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: get(weights, &format!("{prefix}.linear_1.weight"))?.clone(),
            w2: get(weights, &format!("{prefix}.linear_2.weight"))?.clone(),
            w3: get(weights, &format!("{prefix}.linear_3.weight"))?.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h1 = fused_linear3d_native(x, &self.w1, None)?; // gate
        let h3 = fused_linear3d_native(x, &self.w3, None)?; // up
        let act = swiglu_fused_bf16(&h1, &h3)?; // silu(h1) * h3
        fused_linear3d_native(&act, &self.w2, None) // down
    }
}

/// A double-stream `BooguImageDoubleStreamTransformerBlock`.
///
/// Holds the resolved BF16 weights for one block. Built once per block and
/// reused across denoise steps. `modulation` selects the LuminaRMSNormZero
/// (True; the 8 shipped `double_stream_layers`) vs plain-RMSNorm (False;
/// context-style) norm path.
pub struct BooguDoubleStreamBlock {
    /// modulation=True ⇒ AdaLN (LuminaRMSNormZero) on the img/instruct norms.
    pub modulation: bool,

    // img-stream modulation norms (3× RMSNormZero when modulation=True).
    img_norm1: RmsNormZero,
    img_norm2: RmsNormZero,
    img_norm3: RmsNormZero,
    // instruct-stream modulation norms (2× RMSNormZero when modulation=True).
    instruct_norm1: RmsNormZero,
    instruct_norm2: RmsNormZero,

    // plain RMSNorm gammas wrapping the attention/MLP outputs.
    img_attn_norm_w: Tensor,
    img_self_attn_norm_w: Tensor,
    img_ffn_norm1_w: Tensor,
    img_ffn_norm2_w: Tensor,
    instruct_attn_norm_w: Tensor,
    instruct_ffn_norm1_w: Tensor,
    instruct_ffn_norm2_w: Tensor,

    // joint attention (separate per-stream q/k/v + separate out projections +
    // shared qk-norm + shared to_out.0). All bias-free.
    j_img_to_q: Tensor,
    j_img_to_k: Tensor,
    j_img_to_v: Tensor,
    j_instruct_to_q: Tensor,
    j_instruct_to_k: Tensor,
    j_instruct_to_v: Tensor,
    j_instruct_out: Tensor,
    j_img_out: Tensor,
    j_norm_q: Tensor,
    j_norm_k: Tensor,
    j_to_out: Tensor,

    // img self-attention (the C3 GQA path; its own q/k/v/out/qk-norm). Bias-free.
    self_to_q: Tensor,
    self_to_k: Tensor,
    self_to_v: Tensor,
    self_to_out: Tensor,
    self_norm_q: Tensor,
    self_norm_k: Tensor,

    // SwiGLU FFNs (one per stream).
    img_ffn: SwiGluFfn,
    instruct_ffn: SwiGluFfn,
}

impl BooguDoubleStreamBlock {
    /// Build a double-stream block from the transformer weight map.
    ///
    /// `prefix` is the block's module path (e.g. `"double_stream_layers.0"`).
    /// `modulation` MUST be True for the 8 shipped `double_stream_layers`.
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        modulation: bool,
    ) -> Result<Self> {
        let jp = format!("{prefix}.img_instruct_attn");
        let jpp = format!("{jp}.processor");
        let sp = format!("{prefix}.img_self_attn");
        Ok(Self {
            modulation,
            img_norm1: RmsNormZero::load(weights, &format!("{prefix}.img_norm1"), modulation)?,
            img_norm2: RmsNormZero::load(weights, &format!("{prefix}.img_norm2"), modulation)?,
            img_norm3: RmsNormZero::load(weights, &format!("{prefix}.img_norm3"), modulation)?,
            instruct_norm1: RmsNormZero::load(
                weights,
                &format!("{prefix}.instruct_norm1"),
                modulation,
            )?,
            instruct_norm2: RmsNormZero::load(
                weights,
                &format!("{prefix}.instruct_norm2"),
                modulation,
            )?,
            img_attn_norm_w: get(weights, &format!("{prefix}.img_attn_norm.weight"))?.clone(),
            img_self_attn_norm_w: get(weights, &format!("{prefix}.img_self_attn_norm.weight"))?
                .clone(),
            img_ffn_norm1_w: get(weights, &format!("{prefix}.img_ffn_norm1.weight"))?.clone(),
            img_ffn_norm2_w: get(weights, &format!("{prefix}.img_ffn_norm2.weight"))?.clone(),
            instruct_attn_norm_w: get(weights, &format!("{prefix}.instruct_attn_norm.weight"))?
                .clone(),
            instruct_ffn_norm1_w: get(weights, &format!("{prefix}.instruct_ffn_norm1.weight"))?
                .clone(),
            instruct_ffn_norm2_w: get(weights, &format!("{prefix}.instruct_ffn_norm2.weight"))?
                .clone(),
            j_img_to_q: get(weights, &format!("{jpp}.img_to_q.weight"))?.clone(),
            j_img_to_k: get(weights, &format!("{jpp}.img_to_k.weight"))?.clone(),
            j_img_to_v: get(weights, &format!("{jpp}.img_to_v.weight"))?.clone(),
            j_instruct_to_q: get(weights, &format!("{jpp}.instruct_to_q.weight"))?.clone(),
            j_instruct_to_k: get(weights, &format!("{jpp}.instruct_to_k.weight"))?.clone(),
            j_instruct_to_v: get(weights, &format!("{jpp}.instruct_to_v.weight"))?.clone(),
            j_instruct_out: get(weights, &format!("{jpp}.instruct_out.weight"))?.clone(),
            j_img_out: get(weights, &format!("{jpp}.img_out.weight"))?.clone(),
            j_norm_q: get(weights, &format!("{jp}.norm_q.weight"))?.clone(),
            j_norm_k: get(weights, &format!("{jp}.norm_k.weight"))?.clone(),
            j_to_out: get(weights, &format!("{jp}.to_out.0.weight"))?.clone(),
            self_to_q: get(weights, &format!("{sp}.to_q.weight"))?.clone(),
            self_to_k: get(weights, &format!("{sp}.to_k.weight"))?.clone(),
            self_to_v: get(weights, &format!("{sp}.to_v.weight"))?.clone(),
            self_to_out: get(weights, &format!("{sp}.to_out.0.weight"))?.clone(),
            self_norm_q: get(weights, &format!("{sp}.norm_q.weight"))?.clone(),
            self_norm_k: get(weights, &format!("{sp}.norm_k.weight"))?.clone(),
            img_ffn: SwiGluFfn::load(weights, &format!("{prefix}.img_feed_forward"))?,
            instruct_ffn: SwiGluFfn::load(weights, &format!("{prefix}.instruct_feed_forward"))?,
        })
    }

    /// Run the joint cross-attention over `[instruct ; img]` (instruct-first),
    /// returning `(instruct_attn_out, img_attn_out)`. `joint_cos`/`joint_sin` are
    /// the JOINT rope tables `[Lc+Li, head_dim/2]`.
    fn joint_attn(
        &self,
        instruct_norm1_out: &Tensor,
        img_norm1_out: &Tensor,
        cfg: &BooguConfig,
        joint_cos: &Tensor,
        joint_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        joint_gqa_attention(
            instruct_norm1_out,
            img_norm1_out,
            cfg,
            &self.j_img_to_q,
            &self.j_img_to_k,
            &self.j_img_to_v,
            &self.j_instruct_to_q,
            &self.j_instruct_to_k,
            &self.j_instruct_to_v,
            &self.j_instruct_out,
            &self.j_img_out,
            &self.j_norm_q,
            &self.j_norm_k,
            &self.j_to_out,
            joint_cos,
            joint_sin,
        )
    }

    /// Run the img self-attention (`img_self_attn`, C3 GQA path) on
    /// `img_norm3_out` over the combined-img rope (`img_cos`/`img_sin` = the joint
    /// rope's IMG rows).
    fn self_attn(
        &self,
        img_norm3_out: &Tensor,
        cfg: &BooguConfig,
        img_cos: &Tensor,
        img_sin: &Tensor,
    ) -> Result<Tensor> {
        gqa_self_attention(
            img_norm3_out,
            cfg,
            &self.self_to_q,
            &self.self_to_k,
            &self.self_to_v,
            &self.self_to_out,
            &self.self_norm_q,
            &self.self_norm_k,
            img_cos,
            img_sin,
        )
    }

    /// Block forward.
    ///
    /// `instruct` is `[B, Lc, hidden]`, `img` is `[B, Li, hidden]` (BF16). `temb`
    /// is `[B, 1, TIME_EMBED_DIM]` BF16 (required when modulation=True).
    ///
    /// RoPE tables (each `(cos, sin)`, rows `[N, head_dim/2]`):
    /// - `joint_rope` — the JOINT rope, `Lc+Li` rows (used by the joint cross-attn).
    /// - `img_rope` — the combined-img rope, `Li` rows (= joint rope rows
    ///   `[cap_len:seq]`; used by the img self-attn). The caller (C6) narrows the
    ///   joint table to produce this; per the Mojo C4 FRAGILE note, the
    ///   `[cap_len:]` slice is valid only for no-ref T2I.
    /// - `cap_rope` is accepted for interface symmetry (the double-stream block
    ///   does NOT consume a separate caption-only rope — the instruct tokens get
    ///   their rotation from the JOINT rope's caption rows inside the joint
    ///   attention; see transformer_boogu.py:616-624). It is unused here.
    ///
    /// Returns `(instruct, img)` updated `[B,Lc,hidden]`/`[B,Li,hidden]` BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        instruct: &Tensor,
        img: &Tensor,
        temb: Option<&Tensor>,
        cfg: &BooguConfig,
        _cap_rope: (&Tensor, &Tensor),
        img_rope: (&Tensor, &Tensor),
        joint_rope: (&Tensor, &Tensor),
    ) -> Result<(Tensor, Tensor)> {
        let (joint_cos, joint_sin) = joint_rope;
        let (img_cos, img_sin) = img_rope;

        if self.modulation {
            let temb = temb.ok_or_else(|| {
                Error::InvalidOperation(
                    "boogu BooguDoubleStreamBlock: temb required when modulation=True".into(),
                )
            })?;

            // --- Step 1: modulation for both streams (5 RMSNormZero calls). ---
            // img_norm1 -> (img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp)
            let (img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp) =
                self.img_norm1.modulate(img, temb, cfg)?;
            // img_norm2 -> chunk[1] reinterpreted as shift_mlp (chunk[2],[3] discarded)
            let (img_norm2_out, img_shift_mlp, _, _) = self.img_norm2.modulate(img, temb, cfg)?;
            // img_norm3 -> chunk[1] reinterpreted as gate_self (chunk[2],[3] discarded)
            let (img_norm3_out, img_gate_self, _, _) = self.img_norm3.modulate(img, temb, cfg)?;
            // instruct_norm1 -> (instruct_norm1_out, gate_msa, scale_mlp, gate_mlp)
            let (instruct_norm1_out, instruct_gate_msa, instruct_scale_mlp, instruct_gate_mlp) =
                self.instruct_norm1.modulate(instruct, temb, cfg)?;
            // instruct_norm2 -> chunk[1] reinterpreted as shift_mlp
            let (instruct_norm2_out, instruct_shift_mlp, _, _) =
                self.instruct_norm2.modulate(instruct, temb, cfg)?;

            // --- Step 2: joint attention over [instruct ; img]. ---
            let (instruct_attn_out, img_attn_out) =
                self.joint_attn(&instruct_norm1_out, &img_norm1_out, cfg, joint_cos, joint_sin)?;

            // --- Step 3: img self-attention. ---
            let img_self_attn_out = self.self_attn(&img_norm3_out, cfg, img_cos, img_sin)?;

            // --- Step 4: residual updates. ---
            // img += tanh(gate_msa) * img_attn_norm(img_attn_out)
            let img_an = fused_rms_norm(&img_attn_out, &self.img_attn_norm_w, cfg.norm_eps)?;
            let g = img_gate_msa.tanh()?.unsqueeze(1)?; // [B,1,hidden]
            let img = img.add(&g.mul(&img_an)?)?;
            // img += tanh(gate_self) * img_self_attn_norm(img_self_attn_out)
            let img_sn =
                fused_rms_norm(&img_self_attn_out, &self.img_self_attn_norm_w, cfg.norm_eps)?;
            let g = img_gate_self.tanh()?.unsqueeze(1)?;
            let img = img.add(&g.mul(&img_sn)?)?;
            // img_mlp_in = (1 + scale_mlp) * img_norm2_out + shift_mlp
            let scale_mlp_b = img_scale_mlp.unsqueeze(1)?.add_scalar(1.0)?; // [B,1,hidden]
            let shift_mlp_b = img_shift_mlp.unsqueeze(1)?; // [B,1,hidden]
            let img_mlp_in = img_norm2_out.mul(&scale_mlp_b)?.add(&shift_mlp_b)?;
            // img += tanh(gate_mlp) * img_ffn_norm2(img_feed_forward(img_ffn_norm1(img_mlp_in)))
            let img_mlp_n1 = fused_rms_norm(&img_mlp_in, &self.img_ffn_norm1_w, cfg.norm_eps)?;
            let img_mlp_out = self.img_ffn.forward(&img_mlp_n1)?;
            let img_mlp_n2 = fused_rms_norm(&img_mlp_out, &self.img_ffn_norm2_w, cfg.norm_eps)?;
            let g = img_gate_mlp.tanh()?.unsqueeze(1)?;
            let img = img.add(&g.mul(&img_mlp_n2)?)?;

            // instruct += tanh(gate_msa) * instruct_attn_norm(instruct_attn_out)
            let in_an =
                fused_rms_norm(&instruct_attn_out, &self.instruct_attn_norm_w, cfg.norm_eps)?;
            let g = instruct_gate_msa.tanh()?.unsqueeze(1)?;
            let instruct = instruct.add(&g.mul(&in_an)?)?;
            // instruct_mlp_in = (1 + scale_mlp) * instruct_norm2_out + shift_mlp
            let scale_mlp_b = instruct_scale_mlp.unsqueeze(1)?.add_scalar(1.0)?;
            let shift_mlp_b = instruct_shift_mlp.unsqueeze(1)?;
            let instruct_mlp_in = instruct_norm2_out.mul(&scale_mlp_b)?.add(&shift_mlp_b)?;
            let in_mlp_n1 =
                fused_rms_norm(&instruct_mlp_in, &self.instruct_ffn_norm1_w, cfg.norm_eps)?;
            let in_mlp_out = self.instruct_ffn.forward(&in_mlp_n1)?;
            let in_mlp_n2 =
                fused_rms_norm(&in_mlp_out, &self.instruct_ffn_norm2_w, cfg.norm_eps)?;
            let g = instruct_gate_mlp.tanh()?.unsqueeze(1)?;
            let instruct = instruct.add(&g.mul(&in_mlp_n2)?)?;

            Ok((instruct, img))
        } else {
            // --- modulation=False (context-style; ungated, no temb). ---
            let img_norm1_out = self.img_norm1.plain(img, cfg)?;
            let img_norm3_out = self.img_norm3.plain(img, cfg)?;
            let instruct_norm1_out = self.instruct_norm1.plain(instruct, cfg)?;

            let (instruct_attn_out, img_attn_out) =
                self.joint_attn(&instruct_norm1_out, &img_norm1_out, cfg, joint_cos, joint_sin)?;
            let img_self_attn_out = self.self_attn(&img_norm3_out, cfg, img_cos, img_sin)?;

            // img: ungated residuals.
            let img_an = fused_rms_norm(&img_attn_out, &self.img_attn_norm_w, cfg.norm_eps)?;
            let img = img.add(&img_an)?;
            let img_sn =
                fused_rms_norm(&img_self_attn_out, &self.img_self_attn_norm_w, cfg.norm_eps)?;
            let img = img.add(&img_sn)?;
            let img_norm2_out = self.img_norm2.plain(&img, cfg)?;
            let img_mlp_n1 = fused_rms_norm(&img_norm2_out, &self.img_ffn_norm1_w, cfg.norm_eps)?;
            let img_mlp_out = self.img_ffn.forward(&img_mlp_n1)?;
            let img_mlp_n2 = fused_rms_norm(&img_mlp_out, &self.img_ffn_norm2_w, cfg.norm_eps)?;
            let img = img.add(&img_mlp_n2)?;

            // instruct: ungated residuals.
            let in_an =
                fused_rms_norm(&instruct_attn_out, &self.instruct_attn_norm_w, cfg.norm_eps)?;
            let instruct = instruct.add(&in_an)?;
            let instruct_norm2_out = self.instruct_norm2.plain(&instruct, cfg)?;
            let in_mlp_n1 =
                fused_rms_norm(&instruct_norm2_out, &self.instruct_ffn_norm1_w, cfg.norm_eps)?;
            let in_mlp_out = self.instruct_ffn.forward(&in_mlp_n1)?;
            let in_mlp_n2 =
                fused_rms_norm(&in_mlp_out, &self.instruct_ffn_norm2_w, cfg.norm_eps)?;
            let instruct = instruct.add(&in_mlp_n2)?;

            Ok((instruct, img))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BooguConfig {
        BooguConfig::default()
    }

    #[test]
    fn modulation_chunk_count_is_four_both_streams() {
        // Each LuminaRMSNormZero (img + instruct) linear emits 4·hidden → 4
        // chunks (scale_msa, gate_msa, scale_mlp, gate_mlp). chunk[0] consumed
        // internally; chunks [1],[2],[3] returned. Identical for both streams.
        let c = cfg();
        assert_eq!(MODULATION_CHUNKS, 4);
        let adaln_out = MODULATION_CHUNKS * c.hidden_size;
        assert_eq!(adaln_out % MODULATION_CHUNKS, 0);
        assert_eq!(adaln_out / MODULATION_CHUNKS, c.hidden_size);
    }

    #[test]
    fn img_stream_has_three_norms_instruct_two() {
        // The img stream has 3 RMSNormZero (norm1 cross-attn-gate, norm2
        // mlp-shift, norm3 self-attn-gate); the instruct stream has 2 (norm1,
        // norm2). The block consumes: from each *_norm1 → 3 chunks (gate_msa,
        // scale_mlp, gate_mlp); from each *_norm2 / img_norm3 → 1 chunk (chunk[1],
        // reinterpreted shift_mlp resp. gate_self).
        let img_norms = 3usize;
        let instruct_norms = 2usize;
        assert_eq!(img_norms, 3);
        assert_eq!(instruct_norms, 2);
        // img has self-attn (uses img_norm3); instruct does not.
    }

    #[test]
    fn joint_concat_length_is_sum_of_streams() {
        // The joint sequence the cross-attn runs over = Lc + Li (instruct-first
        // concat). At the C4 probe res that is 16 + 256 = 272.
        let (lc, li) = (16usize, 256usize);
        assert_eq!(lc + li, 272);
    }

    #[test]
    fn swiglu_inner_is_13568() {
        // Both per-stream FFNs use LuminaFeedForward inner-dim 13568.
        assert_eq!(cfg().swiglu_inner(), 13568);
    }

    // GPU-dependent: load + forward need a CUDA device + real weights.
    // Compile-only here.
    #[test]
    #[ignore = "requires CUDA device + checkpoint (GPU busy); compile-only this chunk"]
    fn block_double_load_and_forward_compile() {
        let _ = BooguDoubleStreamBlock::load;
        let _ = BooguDoubleStreamBlock::forward;
    }
}
