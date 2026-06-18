//! Boogu-Image C2 — 3-axis interleaved-complex RoPE table builder + apply.
//!
//! Mirrors `boogu/models/transformers/rope.py`
//! (`BooguImageDoubleStreamRotaryPosEmbed`, the no-ref T2I batch=1 branch) and
//! `boogu/models/embeddings.py::apply_rotary_emb(use_real=False)` (the Lumina
//! complex-mul path), cross-checked against the verified Mojo C2 (cos↔real,
//! sin↔imag, cos 1.0 / max_abs ~1.2e-6 — handoff parity table row C2).
//!
//! ## Per-axis frequency table (diffusers `get_1d_rotary_pos_embed`)
//!
//! For axis `a` with rotary dim `d = axes_dim_rope[a]` (= 40), `half_a = d/2`
//! (= 20), base `theta` (= 10000):
//! ```text
//! freqs[j] = 1 / theta^(2j / d)         for j in [0, half_a)        # [half_a]
//! angle    = pos * freqs[j]             # outer product over positions
//! freqs_cis = polar(1, angle) = cos(angle) + i*sin(angle)           # complex64
//! ```
//! The 3 axes are gathered per position-id then concatenated along the last
//! dim (`_get_freqs_cis`: `torch.cat(result, dim=-1)`) →
//! `[seq, sum(half_a) = head_dim/2 = 60]` complex. cos table = real part, sin
//! table = imag part.
//!
//! ## Position-ID rule (no-ref T2I batch=1, rope.py forward)
//!
//! - caption token `t` in `[0, cap_len)`: `position_ids = (t, t, t)`
//!   (`repeat(arange(cap_len), "l -> l 3")`).
//! - image token `k` in `[0, img_len)`: `h = k // w_tok`, `w = k % w_tok`
//!   (row-major H-major flatten); `position_ids = (cap_len, h, w)` where axis-0
//!   is the constant caption-length shift `pe_shift`, axis-1 = row id, axis-2 =
//!   col id.
//! - joint `seq = cap_len + img_len` (no ref ⇒ `sum(ref_img_len) = 0`).
//!
//! ## Apply (`apply_rotary_emb(use_real=False)`, Lumina complex-mul)
//!
//! ```text
//! x_c   = view_as_complex(x.reshape(..., head_dim/2, 2))   # adjacent (2i,2i+1) pairs
//! x_out = view_as_real(x_c * freqs_cis).flatten            # complex multiply
//! ```
//! Adjacent even/odd channels form a complex; multiplying by `cos + i*sin`:
//! `out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]`,
//! `out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]`.
//! That is EXACTLY `flame_core::bf16_ops::rope_fused_bf16_kernel` (bf16_ops.rs:766,
//! interleaved adjacent-pair), which expects cos/sin as `[1, 1, N, head_dim/2]`.
//! We DO NOT write a new RoPE kernel — we build the tables and call the existing
//! interleaved kernel.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::bf16_ops::{rope_fused_bf16, rope_fused_bf16_f32pe};
use flame_core::{DType, Error, Result, Shape, Tensor};

use super::config::BooguConfig;

/// Per-axis position-id triple for a token: `(axis0, axis1, axis2)`.
pub type PosId = [u32; 3];

/// Build the token-major `[seq]` position-id list for the no-ref T2I batch=1
/// case (rope.py forward, no-ref branch).
///
/// `cap_len` caption tokens → `(t,t,t)`; then `h_tok*w_tok` image tokens →
/// `(cap_len, k/w_tok, k%w_tok)`. Returns one `PosId` per token in joint order
/// (caption rows `[0, cap_len)`, then image rows `[cap_len, seq)`), exactly the
/// order the joint sequence is laid out in.
pub fn build_t2i_position_ids(cap_len: usize, h_tok: usize, w_tok: usize) -> Vec<PosId> {
    let img_len = h_tok * w_tok;
    let mut ids = Vec::with_capacity(cap_len + img_len);
    // caption: (t, t, t).
    for t in 0..cap_len {
        let tt = t as u32;
        ids.push([tt, tt, tt]);
    }
    // image: (pe_shift=cap_len, h, w), h = k//w_tok, w = k%w_tok (row-major).
    let pe_shift = cap_len as u32;
    for k in 0..img_len {
        let h = (k / w_tok) as u32;
        let w = (k % w_tok) as u32;
        ids.push([pe_shift, h, w]);
    }
    ids
}

/// Host-side cos/sin half-tables for an arbitrary position-id list.
///
/// `positions` is one `PosId` per token (length `seq`). `axes_dim` is the
/// per-axis rotary dim `[d0,d1,d2]` (each even); `theta` the RoPE base. Returns
/// `(cos, sin)` flat `Vec<f32>` each length `seq * half` where
/// `half = sum(d_a/2)`. cos/sin built in F64 then stored F32 (matches the
/// reference, which uses `freqs_dtype=float64` for the freq matmul; the F64→F32
/// store is the only divergence and is what produced the Mojo C2 ~1.2e-6
/// max_abs).
///
/// Column layout: axis0 block `[0, d0/2)`, then axis1 `[d0/2, d0/2+d1/2)`, then
/// axis2 — the diffusers `torch.cat(result, dim=-1)` order.
fn build_cos_sin_f32(
    positions: &[PosId],
    axes_dim: [usize; 3],
    theta: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let halves = [axes_dim[0] / 2, axes_dim[1] / 2, axes_dim[2] / 2];
    let half: usize = halves.iter().sum();
    for (a, &d) in axes_dim.iter().enumerate() {
        if d == 0 || d % 2 != 0 {
            return Err(Error::InvalidOperation(format!(
                "boogu rope: axes_dim[{a}]={d} must be even and > 0"
            )));
        }
    }

    // Per-axis inv_freq[j] = 1 / theta^(2j / d), j in [0, d/2). F64.
    let theta_f64 = theta as f64;
    let inv_freqs: Vec<Vec<f64>> = axes_dim
        .iter()
        .map(|&d| {
            (0..d / 2)
                .map(|j| 1.0 / theta_f64.powf((2 * j) as f64 / d as f64))
                .collect::<Vec<f64>>()
        })
        .collect();

    let seq = positions.len();
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for (si, pos) in positions.iter().enumerate() {
        let mut col = 0usize;
        for axis in 0..3 {
            let p = pos[axis] as f64;
            for &f in &inv_freqs[axis] {
                let angle = p * f;
                cos[si * half + col] = angle.cos() as f32;
                sin[si * half + col] = angle.sin() as f32;
                col += 1;
            }
        }
        debug_assert_eq!(col, half);
    }
    Ok((cos, sin))
}

/// 3-axis RoPE cos/sin tables for the no-ref T2I batch=1 path.
///
/// Returns `(cos, sin)` each `[seq, head_dim/2]` with `seq = cap_len +
/// h_tok*w_tok` and `head_dim/2 = 60` (= `cfg.rope_half()`).
///
/// `dtype` selects the table storage: pass `DType::BF16` for the project-wide
/// BF16-RoPE convention (`rope_fused_bf16`), or `DType::F32` for the F32-PE
/// variant (`rope_fused_bf16_f32pe`) that removes the BF16 quantization floor.
/// The Mojo C2 built F32 tables and matched the F64 oracle at ~1.2e-6.
///
/// These are the JOINT tables. For the segment-sliced returns the reference
/// produces (cap = rows `[0, cap_len)`, img/combined-img = rows `[cap_len,
/// seq)`), slice the rows of the returned tables (the per-segment narrowing is
/// the consumer's job; the table math is identical). See
/// [`BooguRopeTables`].
pub fn build_t2i_rope_tables(
    cfg: &BooguConfig,
    cap_len: usize,
    h_tok: usize,
    w_tok: usize,
    dtype: DType,
    device: &Arc<CudaDevice>,
) -> Result<BooguRopeTables> {
    if dtype != DType::BF16 && dtype != DType::F32 {
        return Err(Error::InvalidOperation(format!(
            "boogu rope tables: dtype must be BF16 or F32, got {dtype:?}"
        )));
    }
    let positions = build_t2i_position_ids(cap_len, h_tok, w_tok);
    let seq = positions.len();
    let half = cfg.rope_half(); // 60
    let (cos_f32, sin_f32) = build_cos_sin_f32(&positions, cfg.axes_dim_rope, cfg.theta)?;
    debug_assert_eq!(cos_f32.len(), seq * half);

    let cos = Tensor::from_vec_dtype(cos_f32, Shape::from_dims(&[seq, half]), device.clone(), dtype)?;
    let sin = Tensor::from_vec_dtype(sin_f32, Shape::from_dims(&[seq, half]), device.clone(), dtype)?;
    Ok(BooguRopeTables {
        cos,
        sin,
        cap_len,
        img_len: h_tok * w_tok,
        half,
    })
}

/// Joint RoPE cos/sin tables for the T2I path, plus the segment boundaries.
///
/// Tables are `[seq, half]` (`seq = cap_len + img_len`, `half = head_dim/2`).
/// `cap` rope = rows `[0, cap_len)`; `img`/`combined_img` rope = rows
/// `[cap_len, seq)` (no ref ⇒ combined == img). The full `joint` rope = all
/// rows. The per-stream slicing for double-stream attention (C4) and the
/// reshape-to-4D for [`apply_rope`] are the consumer's job.
pub struct BooguRopeTables {
    /// cos table `[seq, half]` (dtype as requested).
    pub cos: Tensor,
    /// sin table `[seq, half]` (dtype as requested).
    pub sin: Tensor,
    /// caption row count.
    pub cap_len: usize,
    /// image row count (= h_tok*w_tok).
    pub img_len: usize,
    /// table column count = head_dim/2.
    pub half: usize,
}

impl BooguRopeTables {
    /// Joint sequence length (`cap_len + img_len`).
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.cap_len + self.img_len
    }
}

/// Apply interleaved-complex RoPE to `q`/`k` via the existing flame-core
/// interleaved kernel.
///
/// `x` is `[B, H, N, head_dim]` BF16 (head_dim even). `cos`/`sin` are
/// `[N, head_dim/2]` (the table rows for this segment, in `x`'s dtype-of-table)
/// — they are reshaped to the kernel's `[1, 1, N, half]` contract here. Returns
/// `[B, H, N, head_dim]` BF16 with the rotation applied.
///
/// Dispatches to `rope_fused_bf16` (BF16 tables) or `rope_fused_bf16_f32pe`
/// (F32 tables) by the cos table dtype — BOTH are the SAME interleaved math
/// (`out[2i]=x[2i]cos-x[2i+1]sin`). No new kernel.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let xd = x.shape().dims();
    if xd.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "boogu apply_rope: x must be [B,H,N,head_dim], got {xd:?}"
        )));
    }
    let (n, head_dim) = (xd[2], xd[3]);
    let half = head_dim / 2;
    let cs = cos.shape().dims();
    let ss = sin.shape().dims();
    // Accept either [N, half] (segment table) or [1,1,N,half] (pre-shaped).
    let cos4 = match cs.len() {
        2 if cs == [n, half] => cos.reshape(&[1, 1, n, half])?,
        4 if cs == [1, 1, n, half] => cos.clone(),
        _ => {
            return Err(Error::InvalidOperation(format!(
                "boogu apply_rope: cos must be [N={n},half={half}] or [1,1,N,half], got {cs:?}"
            )))
        }
    };
    let sin4 = match ss.len() {
        2 if ss == [n, half] => sin.reshape(&[1, 1, n, half])?,
        4 if ss == [1, 1, n, half] => sin.clone(),
        _ => {
            return Err(Error::InvalidOperation(format!(
                "boogu apply_rope: sin must be [N={n},half={half}] or [1,1,N,half], got {ss:?}"
            )))
        }
    };
    match cos4.dtype() {
        DType::BF16 => rope_fused_bf16(x, &cos4, &sin4),
        DType::F32 => rope_fused_bf16_f32pe(x, &cos4, &sin4),
        other => Err(Error::InvalidOperation(format!(
            "boogu apply_rope: cos/sin dtype must be BF16 or F32, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BooguConfig {
        BooguConfig::default()
    }

    #[test]
    fn position_ids_layout_no_ref() {
        // cap_len=16, 16x16 image tokens => seq=272 (matches the Mojo C2 probe).
        let cap_len = 16;
        let (h_tok, w_tok) = (16, 16);
        let ids = build_t2i_position_ids(cap_len, h_tok, w_tok);
        assert_eq!(ids.len(), cap_len + h_tok * w_tok); // 272
        // caption token t -> (t,t,t).
        assert_eq!(ids[0], [0, 0, 0]);
        assert_eq!(ids[1], [1, 1, 1]);
        assert_eq!(ids[15], [15, 15, 15]);
        // first image token k=0 -> (cap_len, 0, 0).
        assert_eq!(ids[cap_len], [16, 0, 0]);
        // image token k=1 -> (cap_len, 0, 1) (row-major: h=0,w=1).
        assert_eq!(ids[cap_len + 1], [16, 0, 1]);
        // image token k=w_tok -> (cap_len, 1, 0) (next row).
        assert_eq!(ids[cap_len + w_tok], [16, 1, 0]);
        // last image token k=img_len-1 -> (cap_len, h_tok-1, w_tok-1).
        assert_eq!(ids[cap_len + h_tok * w_tok - 1], [16, 15, 15]);
    }

    #[test]
    fn cos_sin_table_shape_and_concat_order() {
        // Joint table is [seq, 60]; sum of per-axis halves = 20*3 = 60 = head_dim/2.
        let c = cfg();
        let cap_len = 16;
        let (h_tok, w_tok) = (16, 16);
        let positions = build_t2i_position_ids(cap_len, h_tok, w_tok);
        let seq = positions.len();
        let (cos, sin) = build_cos_sin_f32(&positions, c.axes_dim_rope, c.theta).unwrap();
        let half = c.rope_half();
        assert_eq!(half, 60);
        assert_eq!(cos.len(), seq * half);
        assert_eq!(sin.len(), seq * half);

        // Row 0 = caption pos (0,0,0): all angles 0 -> cos=1, sin=0 every column.
        for col in 0..half {
            assert!((cos[col] - 1.0).abs() < 1e-6, "row0 col{col} cos should be 1");
            assert!(sin[col].abs() < 1e-6, "row0 col{col} sin should be 0");
        }

        // First image token k=0 pos (16,0,0): axis1 (cols [20,40)) and axis2
        // (cols [40,60)) have h=w=0 => angle 0 => cos=1, sin=0 there. axis0
        // (cols [0,20)) has pos=16 => nonzero angles (col 0 = 16*1 = 16 rad).
        let img0 = cap_len * half;
        // axis1 block starts at col 20 (= axes_dim[0]/2): cos=1, sin=0.
        assert!((cos[img0 + 20] - 1.0).abs() < 1e-6);
        assert!(sin[img0 + 20].abs() < 1e-6);
        // axis2 block starts at col 40.
        assert!((cos[img0 + 40] - 1.0).abs() < 1e-6);
        assert!(sin[img0 + 40].abs() < 1e-6);
        // axis0 col0: angle = 16 * inv_freq[0] = 16 * 1 = 16 rad => cos(16), sin(16).
        assert!((cos[img0 + 0] - (16.0f32).cos()).abs() < 1e-4);
        assert!((sin[img0 + 0] - (16.0f32).sin()).abs() < 1e-4);
    }

    #[test]
    fn cos_sin_magnitude_is_unit() {
        // |freqs_cis| == 1 (unit complex): cos^2 + sin^2 == 1 every entry.
        let c = cfg();
        let positions = build_t2i_position_ids(4, 4, 4);
        let (cos, sin) = build_cos_sin_f32(&positions, c.axes_dim_rope, c.theta).unwrap();
        for i in 0..cos.len() {
            let mag = cos[i] * cos[i] + sin[i] * sin[i];
            assert!((mag - 1.0).abs() < 1e-4, "entry {i} mag {mag} != 1");
        }
    }

    #[test]
    fn rejects_odd_axis_dim() {
        let positions = build_t2i_position_ids(2, 2, 2);
        let err = build_cos_sin_f32(&positions, [40, 41, 40], 10000.0);
        assert!(err.is_err());
    }

    // GPU-dependent: table upload + apply need a CUDA device + RoPE kernel.
    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn rope_tables_output_shape() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let c = cfg();
        let tables =
            build_t2i_rope_tables(&c, 16, 16, 16, DType::BF16, &device).unwrap();
        assert_eq!(tables.seq_len(), 272);
        assert_eq!(tables.half, 60);
        assert_eq!(tables.cos.shape().dims(), &[272, 60]);
        assert_eq!(tables.sin.shape().dims(), &[272, 60]);
    }

    #[test]
    #[ignore = "requires CUDA device (GPU busy); compile-only this chunk"]
    fn apply_rope_compiles() {
        let _ = super::apply_rope;
    }
}
