//! Multi-axis RoPE (MRoPE) — interleaved layout used by Qwen3-VL / HiDream-O1.
//!
//! References:
//! - `qwen3_vl_transformers.py:293-354` — `Qwen3VLTextRotaryEmbedding`
//! - `qwen3_vl_transformers.py:315-330` — `apply_interleaved_mrope`
//! - `qwen3_vl_transformers.py:378-402` — `apply_rotary_pos_emb` (half-split)
//! - `utils.py:76-198` — `get_rope_index_fix_point` (the position-id builder)
//!
//! ## Why this is the highest-risk port item
//!
//! The frequency layout is **NOT** "concat T||H||W" — it is **stride-3
//! interleaved**. From `qwen3_vl_transformers.py:315-330` (verbatim, my comments):
//!
//! ```python
//! # freqs has shape (3, B, S, head_dim/2). axis 0 = (T, H, W).
//! freqs_t = freqs[0]                                  # start with T everywhere
//! for dim, offset in enumerate((1, 2), start=1):      # dim=1 → H, dim=2 → W
//!     length = mrope_section[dim] * 3                  # 20*3 = 60 for H and W
//!     idx = slice(offset, length, 3)                   # H: 1,4,7,...,58; W: 2,5,8,...,59
//!     freqs_t[..., idx] = freqs[dim, ..., idx]
//! # Net: slot k in [0, head_dim/2) gets axis (k%3) for k<60, else T
//! ```
//!
//! With `mrope_section = [24, 20, 20]` and `head_dim/2 = 64`:
//!
//! ```
//! slot 0:  T   slot 1:  H   slot 2:  W
//! slot 3:  T   slot 4:  H   slot 5:  W
//! ...
//! slot 57: T   slot 58: H   slot 59: W
//! slot 60: T   slot 61: T   slot 62: T   slot 63: T   ← past the section sum (60); T-only
//! ```
//!
//! Number of T slots = 24 (0,3,6,...,57 is 20 slots; plus 60..64 is 4 more → 24 ✓)
//! Number of H slots = 20 (1,4,...,58)
//! Number of W slots = 20 (2,5,...,59)
//!
//! So `mrope_section[i] * 3` is precisely the **boundary** at which axis `i`
//! stops contributing — `T` always contributes (the loop starts with
//! `freqs[0]` everywhere), and only `H`/`W` slots in range `[offset,
//! length)` are overwritten. For text-only tokens where T=H=W, the
//! overwrite is a no-op. For the gen image where the (T,H,W) coordinates
//! diverge (T=fix_point=4096, H/W = patch row/col), the layout matters.
//!
//! ## Apply step
//!
//! After building `freqs_t : [B, S, head_dim/2]`, Python does:
//!
//! ```python
//! emb = torch.cat((freqs_t, freqs_t), dim=-1)         # [B, S, head_dim]
//! cos = emb.cos() * attention_scaling
//! sin = emb.sin() * attention_scaling
//! ```
//!
//! Then `apply_rotary_pos_emb` (`qwen3_vl_transformers.py:378-402`) uses the
//! HuggingFace **half-split** convention:
//! `q_embed = q*cos + rotate_half(q)*sin` where `rotate_half(x) =
//! cat(-x[..., D/2:], x[..., :D/2], dim=-1)`. flame-core ships this exact
//! kernel as `flame_core::bf16_ops::rope_halfsplit_bf16_pytorch`. We use the
//! PyTorch-expression variant so BF16 multiply/add round points match
//! training references.

use std::sync::Arc;

use flame_core::{bf16_ops, CudaDevice, DType, Error, Result, Shape, Tensor};

/// Bundle of T/H/W position arrays for one decoder forward.
///
/// Mirrors the 3-row `position_ids` tensor shape `[3, B, S]` Python passes
/// through `Qwen3VLTextRotaryEmbedding`, but kept on host (small, per-step
/// table). The Phase 2b model takes a single batch's worth at a time
/// (B=1 the typical case; CFG runs two passes).
///
/// The `'a` lifetime makes this a thin view over caller-owned `Vec<u32>`s
/// produced by [`build_mrope_positions`] — no extra allocation.
pub struct MRopePositions<'a> {
    pub t: &'a [u32],
    pub h: &'a [u32],
    pub w: &'a [u32],
}

/// Build per-axis MRoPE position IDs for a HiDream-O1 token stream.
///
/// Returns `(t_positions, h_positions, w_positions)` each `[1, total_seq_len]`.
///
/// Mirrors `get_rope_index_fix_point` in `utils.py:76-198` for the **single
/// gen-image, no ref-image** T2I case (`skip_vision_start_token = [1]`,
/// `image_grid_thw = [(1, h_patches, w_patches)]`).
///
/// ## Algorithm (T2I, single gen image)
///
/// Given `input_ids` containing a `vision_start_token_id` followed by
/// `(h_patches * w_patches)` image-token slots:
///
/// 1. Find the index `vs_idx` of the first `vision_start_token_id`.
/// 2. The **first** `vs_idx + 1` tokens *plus the vision-start token itself*
///    are "text" — `text_len = vs_idx + 1 - skip_vision_start_token[0]`
///    (per `utils.py:151`, `skip_vision_start_token[0] = 1` for the gen
///    image, so `text_len = vs_idx`). Stamp T=H=W=`0..text_len`.
/// 3. The vision tokens (length `H*W` because spatial_merge_size=1) get:
///    - `T = fix_point + 0`  (fix_point=4096 by default; the Python
///      decrements by `st_idx` only after the second skip-image — for a
///      single gen image `st_idx=text_len`, so `fix_point` is left as 4096
///      then re-zeroed; see `utils.py:161-167`).
///    - `H = patch_row` in `0..h_patches`
///    - `W = patch_col` in `0..w_patches`
///
/// **Caveat**: this is the **single-gen-image, no-refs** specialization
/// to keep Phase 2a small. The full multi-image path (refs + gen) will
/// land in Phase 2b's tokens.rs alongside the chat-template builder.
/// The signature here matches the Python verbatim so swapping in the
/// full builder later is mechanical.
pub fn build_mrope_positions(
    input_ids: &[u32],
    image_token_id: u32,
    _video_token_id: u32,
    vision_start_token_id: u32,
    image_grid_thw: &[(usize, usize, usize)],
    skip_vision_start_token: &[usize],
    fix_point: Option<usize>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let total = input_ids.len();
    let mut t_pos = vec![0u32; total];
    let mut h_pos = vec![0u32; total];
    let mut w_pos = vec![0u32; total];

    if image_grid_thw.is_empty() {
        // Pure text fallback: T = H = W = 0..total
        for i in 0..total {
            t_pos[i] = i as u32;
            h_pos[i] = i as u32;
            w_pos[i] = i as u32;
        }
        return (t_pos, h_pos, w_pos);
    }

    // We currently support the single gen-image case used by T2I.
    // For multi-image / ref-image flows, route through Phase 2b's full builder.
    let (_, ph, pw) = image_grid_thw[0];
    let img_len = ph * pw;
    let skip = skip_vision_start_token.first().copied().unwrap_or(0);
    let fp = fix_point.unwrap_or(4096);

    // 1) Locate vision_start_token_id.
    let vs_idx = match input_ids.iter().position(|&id| id == vision_start_token_id) {
        Some(i) => i,
        None => {
            // No vision tokens — fall back to text positions.
            for i in 0..total {
                t_pos[i] = i as u32;
                h_pos[i] = i as u32;
                w_pos[i] = i as u32;
            }
            return (t_pos, h_pos, w_pos);
        }
    };

    // 2) The first `image_token_id` slot defines `ed` in Python, but the
    //    generated-image position block starts immediately after the text
    //    span, not at `ed`. For T2I, `<|vision_start|>` is the first generated
    //    image slot and the first real `image_token_id` follows it:
    //
    //      full stream: [ text..., vision_start, image_token, image_token, ... ]
    //      text_len   = ed - st - skip = (vs_idx + 1) - 0 - 1 = vs_idx
    //      image pos  = [text_len .. text_len + image_len)
    //
    //    The cache/inference vmask uses exactly that same range. Starting the
    //    patch positions at `ed` shifts every image patch by one slot and
    //    leaves the first generated patch at the default `[0,0,0]`, which does
    //    not match the reference `_get_rope_index_t2i`.
    let ed = match input_ids.iter().position(|&id| id == image_token_id) {
        Some(i) => i,
        None => vs_idx + 1, // Fallback: assume the slot right after vs.
    };
    let st = 0usize;
    let text_len = ed.saturating_sub(st).saturating_sub(skip);

    // Stamp text positions: T = H = W = 0..text_len + st_idx.
    // (st_idx starts at 0 for the first iteration.)
    for i in 0..text_len {
        let p = i as u32;
        t_pos[i] = p;
        h_pos[i] = p;
        w_pos[i] = p;
    }
    // The skipped <|vision_start|> token gets the same T=H=W as if it were
    // the next text token at position text_len (matches Python's behavior:
    // it isn't decremented in the text_len computation, but it sits in the
    // gap between text_len and the patches; in get_rope_index_fix_point the
    // vision_start_token gets stamped via the patch loop below as the very
    // first vision position, but only when skip=0. With skip=1 the
    // vision-start row is left at its default zero stamp from the buffer
    // init; we replicate that zero here implicitly).

    // 3) Stamp image patches.
    //    st_idx for the *first* iteration is 0 (`utils.py:154`).
    //    skip[0] = 1 => the fix-point branch runs (`utils.py:161-167`):
    //      if fix_point > 0:
    //          fix_point = fix_point - st_idx
    //      llm_pos_ids_list.append(stack([t_idx, h_idx, w_idx]) + fix_point + st_idx)
    //      fix_point = 0
    //    With st_idx=0, that gives the per-patch coords + 4096 + 0 = +4096.
    let patch_start = text_len;
    if skip > 0 {
        // gen-image branch: fix_point shift
        let mut fp_eff = fp;
        let st_idx = 0usize;
        if fp_eff > 0 {
            fp_eff = fp_eff.saturating_sub(st_idx);
        }
        // Note: t-dim runs 0..1 (frames=1); h-dim 0..ph; w-dim 0..pw.
        // PyTorch flatten order: t fastest-outer, h middle, w fastest-inner.
        // (See utils.py:157-159: t_index outer, then h, then w innermost.)
        let mut k = 0usize;
        for h in 0..ph {
            for w in 0..pw {
                let idx = patch_start + k;
                if idx >= total {
                    break;
                }
                t_pos[idx] = (fp_eff + st_idx) as u32; // t_index = 0 (frames=1)
                h_pos[idx] = (h + fp_eff + st_idx) as u32;
                w_pos[idx] = (w + fp_eff + st_idx) as u32;
                k += 1;
            }
        }
    } else {
        // ref-image branch: stamp at text_len + st_idx with t/h/w from the grid.
        let st_idx = 0usize;
        let mut k = 0usize;
        for h in 0..ph {
            for w in 0..pw {
                let idx = patch_start + k;
                if idx >= total {
                    break;
                }
                t_pos[idx] = (text_len + st_idx) as u32;
                h_pos[idx] = (h + text_len + st_idx) as u32;
                w_pos[idx] = (w + text_len + st_idx) as u32;
                k += 1;
            }
        }
    }

    // 4) Trailing text after the patches (none in the standard HiDream T2I
    //    flow — patches are the last tokens — but kept for safety).
    let after = patch_start + img_len;
    if after < total {
        let st_idx = if skip > 0 {
            // After the gen image, st_idx in Python is `last_max + 1`.
            // In our single-image case the last_max is `fp_eff + max(ph,pw) - 1`,
            // so the next text token starts at `fp_eff + ph` (or wherever the
            // max landed). Simpler — and matching most flows — just stamp 0
            // here and let the position builder be revisited if/when a
            // post-image-text use case shows up.
            // Conservative replication: use last patch's max index + 1.
            let last_t = *t_pos[..after].iter().max().unwrap_or(&0);
            let last_h = *h_pos[..after].iter().max().unwrap_or(&0);
            let last_w = *w_pos[..after].iter().max().unwrap_or(&0);
            (last_t.max(last_h).max(last_w) as usize) + 1
        } else {
            text_len + img_len
        };
        for i in after..total {
            let p = (st_idx + (i - after)) as u32;
            t_pos[i] = p;
            h_pos[i] = p;
            w_pos[i] = p;
        }
    }

    (t_pos, h_pos, w_pos)
}

/// Build the (cos, sin) tables for interleaved MRoPE.
///
/// Output shape: `[1, S, head_dim]` for both, doubled-up form
/// `cat([freqs_t, freqs_t], -1)` per `qwen3_vl_transformers.py:350`.
/// Stored at BF16 (matches what `rope_halfsplit_bf16` consumes).
///
/// ## Frequency table
///
/// `inv_freq[d] = 1 / theta^(2d / head_dim)` for `d in 0..head_dim/2`.
/// (Standard RoPE formula; Qwen3-VL uses θ=5e6.) Then for each axis a∈{T,H,W}:
///
/// ```text
/// freqs[a, b, s, d] = position_a[b, s] * inv_freq[d]
/// ```
///
/// The interleave step then writes `freqs_t[b, s, d]` from the axis indicated
/// by `slot_axis(d)` (see module-level docs).
///
/// ## Why we build this on host
///
/// MRoPE coefficients are **per-position scalars**; computing them on
/// the GPU would need a custom NVRTC kernel. At HiDream's worst case
/// (S=4350, head_dim/2=64, BF16) the cos+sin table is 1.06 MB — trivial
/// to host-build once per forward and upload. The cost amortizes over
/// 36 layers × 28 steps × 2 (CFG).
///
/// ## TODO for Phase 2b/perf
///
/// - Cache `inv_freq` across forwards (same for every step, only positions change).
/// - Move axis selection to a precomputed `Vec<usize>` (T/H/W → 0/1/2) of length
///   `head_dim/2` — currently inlined per-element.
pub fn interleaved_mrope_cos_sin(
    t_pos: &[u32],
    h_pos: &[u32],
    w_pos: &[u32],
    head_dim: usize,
    rope_theta: f32,
    mrope_section: [usize; 3],
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    if head_dim < 2 || head_dim % 2 != 0 {
        return Err(Error::InvalidOperation(format!(
            "interleaved_mrope_cos_sin: head_dim={} must be even and ≥2",
            head_dim
        )));
    }
    let s = t_pos.len();
    if h_pos.len() != s || w_pos.len() != s {
        return Err(Error::InvalidOperation(format!(
            "interleaved_mrope_cos_sin: position arrays must match length, got T={} H={} W={}",
            t_pos.len(),
            h_pos.len(),
            w_pos.len()
        )));
    }
    let half = head_dim / 2;
    let sec_sum_3: [usize; 3] = [
        mrope_section[0] * 3,
        mrope_section[1] * 3,
        mrope_section[2] * 3,
    ];
    if mrope_section.iter().sum::<usize>() != half {
        return Err(Error::InvalidOperation(format!(
            "interleaved_mrope_cos_sin: mrope_section {:?} must sum to head_dim/2={}",
            mrope_section, half
        )));
    }

    // 1) inv_freq[d] = 1 / theta^(2d/head_dim) for d in 0..half.
    //    Match Qwen3-VL exactly: PyTorch builds this as float32 and later
    //    forces the MRoPE matmul/trig block to float32 under autocast off.
    let inv_freq: Vec<f32> = (0..half)
        .map(|d| {
            let exponent = (2.0f32 * d as f32) / head_dim as f32;
            1.0f32 / rope_theta.powf(exponent)
        })
        .collect();

    // 2) Per-slot axis assignment (T=0, H=1, W=2). Slot d is overwritten by
    //    H if d % 3 == 1 and d < sec_sum_3[1]; by W if d % 3 == 2 and d < sec_sum_3[2];
    //    otherwise stays T (the initial fill).
    let slot_axis: Vec<u8> = (0..half)
        .map(|d| {
            let m = d % 3;
            if m == 1 && d < sec_sum_3[1] {
                1 // H
            } else if m == 2 && d < sec_sum_3[2] {
                2 // W
            } else {
                0 // T (covers m==0 always, and m==1/2 past their length)
            }
        })
        .collect();

    // 3) Build freqs_t[s, d] = position_axis(d)[s] * inv_freq[d], then
    //    emb[s, d] = freqs_t[s, d] for d in 0..half AND for d in half..head_dim
    //    (the cat([freqs_t, freqs_t], -1) duplicates).
    let mut cos_data = vec![0.0f32; s * head_dim];
    let mut sin_data = vec![0.0f32; s * head_dim];
    for si in 0..s {
        for d in 0..half {
            let pos = match slot_axis[d] {
                0 => t_pos[si],
                1 => h_pos[si],
                2 => w_pos[si],
                _ => unreachable!(),
            } as f32;
            let arg = pos * inv_freq[d];
            let c = arg.cos();
            let s_ = arg.sin();
            // First half slot d, second half slot d+half (the duplicate).
            cos_data[si * head_dim + d] = c;
            sin_data[si * head_dim + d] = s_;
            cos_data[si * head_dim + half + d] = c;
            sin_data[si * head_dim + half + d] = s_;
        }
    }

    // Note: Qwen3-VL's `attention_scaling` from
    // `qwen3_vl_transformers.py:308` is the standard ROPE_INIT_FUNCTIONS
    // scaling (1.0 for `rope_type="default"`). For the dynamic/Yarn rope
    // types it can differ. Phase 2b should plumb through whatever the
    // checkpoint's `rope_scaling` field requests; for now the canonical
    // path is "default" → 1.0, so we omit the multiplier.

    // The half-split RoPE kernel expects [B, H, N, head_dim] (or the
    // broadcast-friendly [1, 1, N, head_dim/2] variant), but its
    // reshape-based dispatch accepts any layout where `cos.numel() ==
    // cos_bh * N * half`. We give it `[1, 1, S, head_dim]` directly, and
    // let `reshape(&[cos_bh, N, half])` inside kick in (cos_bh = 2 because
    // total_elems = head_dim, half=head_dim/2).
    //
    // Actually simpler: provide cos/sin as [1, S, half] and let the kernel
    // broadcast the head/batch axes. flame_core/bf16_ops.rs:735-740:
    //     let cos_elem = cos.shape().elem_count();
    //     let cos_bh = cos_elem / (n * half);
    //     let cos_flat = cos.reshape(&[cos_bh, n, half])?;
    // For [1, S, half], cos_elem = S*half, cos_bh = 1 → broadcasts.
    //
    // But the half-split kernel iterates [..., d] over `half` entries
    // *and* the duplicate at `half + d` is **inferred internally** (the
    // kernel only stores `half` table entries). So we shouldn't duplicate
    // the cos/sin to head_dim; we should pass the half-sized table.
    //
    // Truncate the duplicate before upload.
    let mut cos_half = vec![0.0f32; s * half];
    let mut sin_half = vec![0.0f32; s * half];
    for si in 0..s {
        for d in 0..half {
            cos_half[si * half + d] = cos_data[si * head_dim + d];
            sin_half[si * half + d] = sin_data[si * head_dim + d];
        }
    }

    let cos = Tensor::from_vec_dtype(
        cos_half,
        Shape::from_dims(&[1, s, half]),
        device.clone(),
        DType::BF16,
    )?;
    let sin = Tensor::from_vec_dtype(
        sin_half,
        Shape::from_dims(&[1, s, half]),
        device.clone(),
        DType::BF16,
    )?;
    Ok((cos, sin))
}

/// Apply MRoPE to Q or K via half-split rotation.
///
/// Input: `x : [B, num_heads, S, head_dim]` BF16.
/// `cos`, `sin`: shape `[1, S, head_dim/2]` BF16
/// (as produced by [`interleaved_mrope_cos_sin`]).
///
/// Returns rotated `x` of the same shape.
///
/// Internally calls `flame_core::bf16_ops::rope_halfsplit_bf16_pytorch` which
/// performs `q*cos + rotate_half(q)*sin` using the HuggingFace half-split
/// convention (`qwen3_vl_transformers.py:378-402`).
pub fn apply_mrope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    bf16_ops::rope_halfsplit_bf16_pytorch(x, cos, sin)
}

/// CPU-side helper: apply the interleave fold to a `[3, S, head_dim/2]`
/// frequency tensor in-place and return `[S, head_dim/2]`.
///
/// This mirrors `apply_interleaved_mrope` 1:1 but operates on a flat
/// `&mut [f32]`. Useful for offline parity tests and not used by the live
/// pipeline (which inlines the interleave inside [`interleaved_mrope_cos_sin`]
/// for efficiency).
///
/// Layout convention: `freqs[axis * S * half + s * half + d]`.
pub fn apply_interleaved_mrope(
    freqs: &mut [f32],
    s: usize,
    half: usize,
    mrope_section: [usize; 3],
) -> Vec<f32> {
    debug_assert_eq!(freqs.len(), 3 * s * half);
    let mut out = vec![0.0f32; s * half];
    // Start with axis 0 (T) everywhere.
    for si in 0..s {
        for d in 0..half {
            out[si * half + d] = freqs[0 * s * half + si * half + d];
        }
    }
    // Overwrite stride-3 slices for H (axis=1, offset=1) and W (axis=2, offset=2).
    let lengths = [
        mrope_section[0] * 3, // unused (axis 0 doesn't overwrite)
        mrope_section[1] * 3,
        mrope_section[2] * 3,
    ];
    for axis in 1..3 {
        let offset = axis;
        let length = lengths[axis];
        let mut d = offset;
        while d < length && d < half {
            for si in 0..s {
                out[si * half + d] = freqs[axis * s * half + si * half + d];
            }
            d += 3;
        }
    }
    out
}
