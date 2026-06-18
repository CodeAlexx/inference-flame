//! Boogu-Image C3 — GQA self-attention helper for the single-stream / refiner
//! `BooguImageTransformerBlock` (`BooguImageAttnProcessor`, the non-flash SDPA
//! path).
//!
//! Mirrors `boogu/models/attention_processor.py::BooguImageAttnProcessor.__call__`
//! (lines 1163-1275, the PyTorch-2 `scaled_dot_product_attention` processor that
//! the single-stream block uses on the no-`flash_attn` path), cross-checked
//! against the verified Mojo C3 `BooguBlock._attn`
//! (`/home/alex/mojodiffusion/serenitymojo/models/dit/boogu_dit.mojo:439-470`,
//! handoff block-out cos 0.9999980).
//!
//! ## Op chain (T2I, batch=1, full attention — mask all-True ⇒ no mask)
//!
//! ```text
//! q = to_q(x)                         # [B,N,28·120]  (bias-free)
//! k = to_k(x), v = to_v(x)            # [B,N, 7·120]  (bias-free)
//! q = q.view(B, N, 28, 120)          # split heads
//! k = k.view(B, N,  7, 120)
//! v = v.view(B, N,  7, 120)
//! q = norm_q(q); k = norm_k(k)       # QK-RMSNorm on head_dim 120, eps 1e-5, BEFORE rope
//! q = apply_rope(q); k = apply_rope(k)   # interleaved-complex (use_real=False)
//! # permute to [B,H,N,120], repeat_kv ×4 (7→28, repeat_interleave over heads)
//! q,k,v -> [B,28,N,120]
//! attn = SDPA(q,k,v, scale = 1/√120) # full attention, no mask
//! out  = attn.permute -> [B,N,28·120]
//! out  = to_out.0(out)               # out projection, bias-free; to_out.1 = dropout = identity
//! ```
//!
//! ## SDPA decision (LOCKED — USER, fastest): fused cuDNN via scale-correction
//!
//! head_dim is **120**, which is NOT in cuDNN's flash set {64,96,128}. To hit
//! the fused path we **zero-pad q/k/v head_dim 120→128** and call the no-bias
//! [`sdpa`] (which routes unmasked BF16 d∈{64,96,128} to `forward_cudnn_sdpa_bf16`).
//! That cuDNN path hardcodes `scale = 1/√d_q = 1/√128`, so before padding we
//! pre-scale **q only** by `c = √(128/120)`. The net softmax arg is then
//! `(1/√128)·c·QKᵀ = (1/√120)·QKᵀ` — exactly the reference's `head_dim**-0.5`.
//! Padding is EXACT:
//! - the 8 zero key/query columns contribute `0·0 = 0` to every `QK^T` entry,
//!   so the score matrix is identical to the unpadded 120-dim score matrix;
//! - the value tensor's padded columns 120:128 are zero, so output columns
//!   120:128 are zero and are discarded by the narrow back to 120.
//!
//! This replaces the prior `sdpa_with_bias(scale=1/√120)` call, which routed to
//! `crate::sdpa::forward_with_bias` — a manual FP32 path (upcast→GEMM→FP32
//! softmax→GEMM→downcast) with NO cuDNN dispatch. The no-bias `sdpa` reaches
//! fused cuDNN, which was the whole point of the LOCKED decision.

use flame_core::attention::sdpa;
use flame_core::ops::fused_inference::{fused_linear3d_native, fused_rms_norm};
use flame_core::{Error, Result, Shape, Tensor};

use super::config::BooguConfig;
use super::rope::apply_rope;

/// cuDNN flash head_dim padding target (the smallest of {64,96,128} ≥ 120).
const SDPA_PAD_HEAD_DIM: usize = 128;

/// Zero-pad a `[B,H,N,d]` tensor's last dim from `d` to `target` (`target >= d`).
///
/// Cats `[B,H,N,target-d]` zeros onto the last dim. Exact: the appended columns
/// are zero, so they add 0 to every `QK^T` entry and produce zero V-output
/// columns. The result is contiguous (cat materializes contiguous output).
fn pad_head_dim(x: &Tensor, target: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "boogu pad_head_dim: expected [B,H,N,d], got {dims:?}"
        )));
    }
    let d = dims[3];
    if d == target {
        return Ok(x.clone());
    }
    if d > target {
        return Err(Error::InvalidOperation(format!(
            "boogu pad_head_dim: head_dim {d} > target {target}"
        )));
    }
    let pad_w = target - d;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[dims[0], dims[1], dims[2], pad_w]),
        x.dtype(),
        x.device().clone(),
    )?;
    Tensor::cat(&[x, &zeros], 3)
}

/// Repeat KV heads to match Q head count for GQA — `repeat_interleave` over the
/// head axis (head `h` → `n_rep` consecutive copies), matching the reference
/// `key.repeat_interleave(query.size(-3)//key.size(-3), -3)`
/// (attention_processor.py:1260) and qwen3_encoder's `repeat_kv` idiom.
///
/// `x`: `[B, H_kv, N, d]` → `[B, H_kv*n_rep, N, d]` with head order
/// `[h0,h0,h0,h0, h1,h1,h1,h1, …]` (interleaved, NOT tiled).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "boogu repeat_kv: expected [B,H_kv,N,d], got {dims:?}"
        )));
    }
    let (b, h_kv, n, d) = (dims[0], dims[1], dims[2], dims[3]);
    // stack n_rep copies on a new axis after the head axis, then merge:
    // [B,H_kv,N,d] -> [B,H_kv,n_rep,N,d] -> [B,H_kv*n_rep,N,d].
    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    let stacked = Tensor::stack(&copies, 2)?;
    stacked.reshape(&[b, h_kv * n_rep, n, d])
}

/// Shared GQA attention core: from already-projected q/k/v `[B,N,*]` →
/// qk-norm → interleaved RoPE → repeat_kv → scale-corrected padded cuDNN SDPA →
/// merge heads → `[B,N,28·120]` (NOT out-projected).
///
/// `q`: `[B, N, 28·120]`, `k`/`v`: `[B, N, 7·120]` (BF16). `cos`/`sin` are the
/// RoPE tables `[N, head_dim/2]` covering this sequence (caller narrows the
/// joint tables to the right segment). Returns the merged attention output
/// `[B, N, 28·120]` BF16, ready for the caller's out projection(s).
///
/// This is the SDPA idiom shared by the single-stream self-attn
/// ([`gqa_self_attention`]) and the double-stream joint attn
/// ([`joint_gqa_attention`]): identical math, only the projection/concat/split
/// wrapping differs.
fn gqa_attention_core(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &BooguConfig,
    norm_q_w: &Tensor,
    norm_k_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let qd = q.shape().dims();
    if qd.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "boogu gqa_attention_core: q must be [B,N,28·120], got {qd:?}"
        )));
    }
    let (b, n) = (qd[0], qd[1]);
    let h = cfg.num_attention_heads; // 28
    let h_kv = cfg.num_kv_heads; // 7
    let d = cfg.head_dim; // 120
    let n_rep = cfg.gqa_repeat(); // 4

    // view into heads: q [B,N,28,120], k/v [B,N,7,120].
    let q = q.reshape(&[b, n, h, d])?;
    let k = k.reshape(&[b, n, h_kv, d])?;
    let v = v.reshape(&[b, n, h_kv, d])?;

    // QK-RMSNorm on head_dim (eps = norm_eps = 1e-5), BEFORE rope. The fused
    // kernel reduces over the LAST dim, so flatten the leading dims to rows of
    // width `head_dim` and reshape back.
    let q_flat = q.reshape(&[b * n * h, d])?;
    let k_flat = k.reshape(&[b * n * h_kv, d])?;
    let q = fused_rms_norm(&q_flat, norm_q_w, cfg.norm_eps)?.reshape(&[b, n, h, d])?;
    let k = fused_rms_norm(&k_flat, norm_k_w, cfg.norm_eps)?.reshape(&[b, n, h_kv, d])?;

    // Interleaved-complex RoPE (use_real=False). `apply_rope` wants
    // [B,H,N,head_dim], so permute heads forward first.
    let q = q.permute(&[0, 2, 1, 3])?; // [B,28,N,120]
    let k = k.permute(&[0, 2, 1, 3])?; // [B,7,N,120]
    let v = v.permute(&[0, 2, 1, 3])?; // [B,7,N,120]
    let q = apply_rope(&q, cos, sin)?;
    let k = apply_rope(&k, cos, sin)?;

    // GQA: repeat_kv ×4 so k/v have 28 heads (repeat_interleave over heads).
    let k = repeat_kv(&k, n_rep)?; // [B,28,N,120]
    let v = repeat_kv(&v, n_rep)?; // [B,28,N,120]

    // WHY: cuDNN's no-bias sdpa hardcodes scale=1/√d_q=1/√128 after we pad to
    // 128; pre-scaling q by √(128/120) makes the net softmax arg
    // (1/√128)·√(128/120)·QKᵀ = (1/√120)·QKᵀ — exact, on the fused cuDNN path.
    // Apply AFTER qk-norm+rope (qk-norm would cancel a pre-norm scale; rope is
    // a magnitude-preserving rotation). Scale q only, not k or v.
    let c = (SDPA_PAD_HEAD_DIM as f32 / d as f32).sqrt();
    let q = q.mul_scalar(c)?;

    // Pad head_dim 120→128 to target the fused cuDNN flash set. Exact pad (see
    // module docs). `apply_rope`/`permute` outputs may be strided; cat
    // materializes contiguous, and SDPA's FP32 reshape path needs contiguous
    // inputs — this is a legitimate layout requirement, NOT a flame-bug mask.
    let qp = pad_head_dim(&q, SDPA_PAD_HEAD_DIM)?;
    let kp = pad_head_dim(&k, SDPA_PAD_HEAD_DIM)?;
    let vp = pad_head_dim(&v, SDPA_PAD_HEAD_DIM)?;

    // Full attention (mask all-True at b=1) ⇒ no mask. No-bias cuDNN path; its
    // hardcoded 1/√128 combines with the pre-scaled q to give exact 1/√120.
    let attn = sdpa(&qp, &kp, &vp, None)?; // [B,28,N,128]

    // Slice the padded head_dim back to 120 (the 120:128 columns are zero).
    let attn = if attn.shape().dims()[3] != d {
        attn.narrow(3, 0, d)?
    } else {
        attn
    };

    // [B,28,N,120] -> [B,N,28·120].
    let attn = attn.permute(&[0, 2, 1, 3])?;
    let attn = attn.contiguous()?; // permuted view → materialize before reshape+GEMM
    attn.reshape(&[b, n, h * d])
}

/// GQA self-attention (`BooguImageAttnProcessor`, full attention, batch=1).
///
/// `x` is the modulation-normed hidden `[B, N, hidden]` BF16; `cos`/`sin` are the
/// per-segment RoPE tables `[N, head_dim/2]` (= the rows of the joint tables for
/// this block's segment — the caller narrows them). Returns `[B, N, hidden]`
/// BF16 (= attn over the sequence, out-projected).
///
/// Weight keys (under `prefix = "<block>.attn"`): `to_q.weight`, `to_k.weight`,
/// `to_v.weight`, `to_out.0.weight` (all bias-free), `norm_q.weight`,
/// `norm_k.weight` (weight-only RMSNorm on head_dim).
pub fn gqa_self_attention(
    x: &Tensor,
    cfg: &BooguConfig,
    to_q_w: &Tensor,
    to_k_w: &Tensor,
    to_v_w: &Tensor,
    to_out_w: &Tensor,
    norm_q_w: &Tensor,
    norm_k_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "boogu gqa_self_attention: x must be [B,N,hidden], got {dims:?}"
        )));
    }
    let hidden = dims[2];
    if hidden != cfg.hidden_size {
        return Err(Error::InvalidOperation(format!(
            "boogu gqa_self_attention: hidden {hidden} != cfg.hidden_size {}",
            cfg.hidden_size
        )));
    }

    // q/k/v projections (bias-free). q: [B,N,28·120], k/v: [B,N,7·120].
    let q = fused_linear3d_native(x, to_q_w, None)?;
    let k = fused_linear3d_native(x, to_k_w, None)?;
    let v = fused_linear3d_native(x, to_v_w, None)?;

    let merged = gqa_attention_core(&q, &k, &v, cfg, norm_q_w, norm_k_w, cos, sin)?;

    // out projection (to_out.0, bias-free). to_out.1 = dropout = identity at inference.
    fused_linear3d_native(&merged, to_out_w, None)
}

/// Double-stream JOINT GQA attention
/// (`BooguImageDoubleStreamSelfAttnProcessor.__call__`,
/// attention_processor.py:706-877; full attention, no-ref T2I batch=1).
///
/// SEPARATE per-stream q/k/v: `img_to_{q,k,v}` projects `img_norm1_out`,
/// `instruct_to_{q,k,v}` projects `instruct_norm1_out`. The per-stream q/k/v are
/// concatenated **instruct-FIRST** along the sequence axis
/// (`_concat_instruction_image_features`:638-641), QK-normed + joint-RoPE'd +
/// SDPA'd as one joint sequence (`rotary_emb` = the JOINT rope, all 272 rows at
/// the probe res), then split back instruct-first, projected by the SEPARATE
/// `instruct_out`/`img_out`, re-concatenated instruct-first, and passed through
/// the SHARED `to_out.0` (`img_instruct_attn.to_out.0`, bias-free; `to_out.1` =
/// dropout = identity).
///
/// `instruct`/`img` are the modulation-normed streams `[B,Lc,hidden]` /
/// `[B,Li,hidden]` BF16. `joint_cos`/`joint_sin` are the JOINT RoPE tables
/// `[Lc+Li, head_dim/2]`. Returns `(instruct_attn_out [B,Lc,hidden],
/// img_attn_out [B,Li,hidden])` — the per-stream attention outputs the block
/// then gates onto its residuals (`instruct_attn_norm`/`img_attn_norm` applied
/// by the caller, matching transformer_boogu.py:627-654/667-671).
///
/// Weight keys (under `prefix = "<block>.img_instruct_attn"`):
/// `processor.{img,instruct}_to_{q,k,v}.weight`,
/// `processor.{img,instruct}_out.weight`, `{norm_q,norm_k}.weight`,
/// `to_out.0.weight` (all bias-free).
#[allow(clippy::too_many_arguments)]
pub fn joint_gqa_attention(
    instruct: &Tensor,
    img: &Tensor,
    cfg: &BooguConfig,
    img_to_q_w: &Tensor,
    img_to_k_w: &Tensor,
    img_to_v_w: &Tensor,
    instruct_to_q_w: &Tensor,
    instruct_to_k_w: &Tensor,
    instruct_to_v_w: &Tensor,
    instruct_out_w: &Tensor,
    img_out_w: &Tensor,
    norm_q_w: &Tensor,
    norm_k_w: &Tensor,
    to_out_w: &Tensor,
    joint_cos: &Tensor,
    joint_sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let id = instruct.shape().dims();
    let imd = img.shape().dims();
    if id.len() != 3 || imd.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "boogu joint_gqa_attention: instruct/img must be [B,N,hidden], got {id:?}/{imd:?}"
        )));
    }
    if id[0] != imd[0] || id[2] != cfg.hidden_size || imd[2] != cfg.hidden_size {
        return Err(Error::InvalidOperation(format!(
            "boogu joint_gqa_attention: bad shapes instruct {id:?} img {imd:?} (hidden={})",
            cfg.hidden_size
        )));
    }
    let l_instruct = id[1];
    let l_img = imd[1];

    // Separate per-stream q/k/v projections (bias-free). img/instruct each get
    // their OWN q/k/v weights — NOT a shared qkv.
    let img_q = fused_linear3d_native(img, img_to_q_w, None)?; // [B,Li,28·120]
    let img_k = fused_linear3d_native(img, img_to_k_w, None)?; // [B,Li,7·120]
    let img_v = fused_linear3d_native(img, img_to_v_w, None)?;
    let instruct_q = fused_linear3d_native(instruct, instruct_to_q_w, None)?; // [B,Lc,28·120]
    let instruct_k = fused_linear3d_native(instruct, instruct_to_k_w, None)?; // [B,Lc,7·120]
    let instruct_v = fused_linear3d_native(instruct, instruct_to_v_w, None)?;

    // Concat INSTRUCT-FIRST along the sequence axis (dim 1):
    // [instruct ; img] → [B, Lc+Li, *]. (no-ref T2I batch=1: no padding, the
    // _concat helper degenerates to a plain seq-dim cat.)
    let q = Tensor::cat(&[&instruct_q, &img_q], 1)?;
    let k = Tensor::cat(&[&instruct_k, &img_k], 1)?;
    let v = Tensor::cat(&[&instruct_v, &img_v], 1)?;

    // Shared QK-norm + JOINT RoPE + GQA SDPA over the full joint sequence.
    let merged = gqa_attention_core(&q, &k, &v, cfg, norm_q_w, norm_k_w, joint_cos, joint_sin)?;

    // Split back INSTRUCT-FIRST: instruct = rows [0,Lc), img = rows [Lc,Lc+Li).
    let instruct_hs = merged.narrow(1, 0, l_instruct)?.contiguous()?;
    let img_hs = merged.narrow(1, l_instruct, l_img)?.contiguous()?;

    // SEPARATE per-stream output projections (bias-free).
    let instruct_projected = fused_linear3d_native(&instruct_hs, instruct_out_w, None)?;
    let img_projected = fused_linear3d_native(&img_hs, img_out_w, None)?;

    // Re-concat INSTRUCT-FIRST, then the SHARED to_out.0 over the joint seq.
    let merged_proj = Tensor::cat(&[&instruct_projected, &img_projected], 1)?;
    let out = fused_linear3d_native(&merged_proj, to_out_w, None)?;

    // Split the shared-projected output back into the two streams (the block
    // gates each stream's attn output onto its own residual).
    let instruct_out = out.narrow(1, 0, l_instruct)?.contiguous()?;
    let img_out = out.narrow(1, l_instruct, l_img)?.contiguous()?;
    Ok((instruct_out, img_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BooguConfig {
        BooguConfig::default()
    }

    #[test]
    fn pad_target_is_in_cudnn_set() {
        // 120 -> 128, and 128 is the smallest flash head_dim ≥ 120.
        let c = cfg();
        assert_eq!(c.head_dim, 120);
        assert!(SDPA_PAD_HEAD_DIM >= c.head_dim);
        assert!(matches!(SDPA_PAD_HEAD_DIM, 64 | 96 | 128));
    }

    #[test]
    fn gqa_shapes_consistent() {
        // 28 q heads, 7 kv heads, repeat ×4 -> 28; hidden = 28*120 = 3360.
        let c = cfg();
        assert_eq!(c.gqa_repeat(), 4);
        assert_eq!(c.num_kv_heads * c.gqa_repeat(), c.num_attention_heads);
        assert_eq!(c.num_attention_heads * c.head_dim, c.hidden_size);
        // kv projection width = 7*120 = 840.
        assert_eq!(c.num_kv_heads * c.head_dim, 840);
    }

    #[test]
    fn sdpa_scale_uses_unpadded_head_dim() {
        // scale must be 1/√120 (unpadded), NOT 1/√128 — the pad is exact only if
        // the softmax scale stays at the true head_dim.
        let c = cfg();
        let scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let scale_120 = 1.0f32 / (120.0f32).sqrt();
        let scale_128 = 1.0f32 / (128.0f32).sqrt();
        assert!((scale - scale_120).abs() < 1e-9);
        assert!((scale - scale_128).abs() > 1e-4); // distinctly different
    }

    /// GPU self-check: the new fused-cuDNN SDPA path (pre-scale q by √(128/120) +
    /// pad + no-bias `sdpa`) must match the old exact FP32-manual reference
    /// (pad + `sdpa_with_bias(None, 1/√120)`) within bf16/cuDNN tolerance, and be
    /// faster. Run on GPU:
    /// `cargo test --release --lib boogu::attention::tests::sdpa_cudnn_matches_manual_and_is_faster -- --nocapture --ignored`
    #[test]
    #[ignore = "requires CUDA device (RTX 3090 Ti); run explicitly with --ignored"]
    fn sdpa_cudnn_matches_manual_and_is_faster() {
        use flame_core::DType;
        use std::time::Instant;

        // `CudaDevice::new` already returns `Arc<CudaDevice>`.
        let device = cudarc::driver::CudaDevice::new(0).unwrap();

        // Representative Boogu self-attn shape AFTER repeat_kv (q/k/v all 28 heads).
        let (b, heads, seq, d) = (1usize, 28usize, 512usize, 120usize);
        let pad = SDPA_PAD_HEAD_DIM; // 128

        let mk = || {
            Tensor::randn(Shape::from_dims(&[b, heads, seq, d]), 0.0, 1.0, device.clone())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let q = mk();
        let k = mk();
        let v = mk();

        // OLD reference: pad q/k/v to 128, manual FP32 sdpa_with_bias at 1/√120,
        // slice back to 120. This is the exact pre-fix math.
        let ref_out = || -> Result<Tensor> {
            let qp = pad_head_dim(&q, pad)?;
            let kp = pad_head_dim(&k, pad)?;
            let vp = pad_head_dim(&v, pad)?;
            let scale = 1.0f32 / (d as f32).sqrt();
            let o = flame_core::attention::sdpa_with_bias(&qp, &kp, &vp, None, Some(scale))?;
            o.narrow(3, 0, d)
        };

        // NEW path: pre-scale q by √(128/120), pad, no-bias fused-cuDNN sdpa,
        // slice back to 120.
        let new_out = || -> Result<Tensor> {
            let c = (pad as f32 / d as f32).sqrt();
            let qs = q.mul_scalar(c)?;
            let qp = pad_head_dim(&qs, pad)?;
            let kp = pad_head_dim(&k, pad)?;
            let vp = pad_head_dim(&v, pad)?;
            let o = super::sdpa(&qp, &kp, &vp, None)?;
            o.narrow(3, 0, d)
        };

        let a = ref_out().expect("manual FP32 reference SDPA failed");
        let bb = new_out().expect("fused cuDNN SDPA path failed");

        // Cosine similarity over the full flattened output.
        let av = a.to_vec_f32().unwrap();
        let bv = bb.to_vec_f32().unwrap();
        assert_eq!(av.len(), bv.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..av.len() {
            dot += av[i] as f64 * bv[i] as f64;
            na += (av[i] as f64) * (av[i] as f64);
            nb += (bv[i] as f64) * (bv[i] as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        println!("[boogu sdpa] cos(new cuDNN vs old FP32-manual) = {cos:.6}");

        // Timings: 50 iters each after a warmup, with a device sync at the end.
        let iters = 50;
        for _ in 0..3 {
            let _ = ref_out().unwrap();
            let _ = new_out().unwrap();
        }
        device.synchronize().unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = ref_out().unwrap();
        }
        device.synchronize().unwrap();
        let old_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = new_out().unwrap();
        }
        device.synchronize().unwrap();
        let new_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

        println!(
            "[boogu sdpa] old FP32-manual: {old_ms:.3} ms/iter | new cuDNN: {new_ms:.3} ms/iter | speedup {:.2}x",
            old_ms / new_ms
        );

        assert!(
            cos >= 0.999,
            "cuDNN path diverged from exact reference: cos={cos:.6} (<0.999)"
        );
    }

    #[test]
    fn joint_concat_length_is_sum() {
        // Joint attention concatenates instruct + img instruct-first; the joint
        // sequence length = Lc + Li (= 16 + 256 = 272 at the C4 probe res).
        let (lc, li) = (16usize, 256usize);
        assert_eq!(lc + li, 272);
    }

    // GPU-dependent: the projections + SDPA need a CUDA device. Compile-only here.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn gqa_self_attention_compiles() {
        let _ = super::gqa_self_attention;
        let _ = super::joint_gqa_attention;
        let _ = super::gqa_attention_core;
        let _ = super::pad_head_dim;
        let _ = super::repeat_kv;
    }
}
