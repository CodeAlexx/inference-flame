//! Ideogram 4 — packed-sequence input/position/segment/indicator builder.
//!
//! Mirrors `Ideogram4Pipeline._build_inputs`
//! (`/home/alex/ideogram4-ref/src/ideogram4/pipeline_ideogram4.py:344-412`)
//! for the **B=1 single-prompt** inference target. Produces host arrays the
//! (chunk-4b) infer loop uploads, plus the data `mrope::build_cos_sin`
//! consumes (the three position columns).
//!
//! ## Layout (verbatim from `_build_inputs`)
//!
//! For one prompt with `num_text` tokens, `max_text_tokens = num_text` (B=1, no
//! cross-prompt padding), and an image grid `grid_h × grid_w`:
//!
//! ```text
//! sequence index:  [ pad_len zeros ][ text tokens ][ image tokens ]
//!                    └─ pad_len=0 ─┘└── num_text ──┘└ num_image_tokens ┘
//! total L = max_text_tokens + num_image_tokens
//! ```
//!
//! For B=1 `pad_len = max_text_tokens - num_text = 0`, so the layout is simply
//! `[text][image]`. The padding machinery is kept (offset = pad_len) so the
//! code reads 1:1 against the reference and a future B>1 path is a small change.
//!
//! ## Positions (`pipeline_ideogram4.py:366-394`)
//!
//! - text positions: `arange(num_text)` broadcast to all 3 axes (t=h=w=k) —
//!   MRoPE degenerates to 1D RoPE on the text span.
//! - image positions: `stack(t=0, h, w) + IMAGE_POSITION_OFFSET` where `h` is
//!   the row index (`arange(grid_h)` expanded over columns) and `w` is the
//!   column index (`arange(grid_w)` expanded over rows), both flattened
//!   row-major. The `+65536` offset keeps image coords disjoint from text
//!   positions (text positions never exceed `max_text_tokens` ≪ 65536).
//!
//! ## Indicator (`pipeline_ideogram4.py:396-397`)
//!
//! `LLM_TOKEN_INDICATOR=3` over the text span, `OUTPUT_IMAGE_INDICATOR=2` over
//! the image span, `0` over the pad span.
//!
//! ## Segment ids (`pipeline_ideogram4.py:376-400`)
//!
//! Initialized to `SEQUENCE_PADDING_INDICATOR=-1` everywhere, then the real
//! (text+image) span `[offset, offset+total_unpadded)` is set to `1`. So pad
//! slots = -1, real slots = 1. The block-diagonal attention mask is
//! `segment_ids[:,None] == segment_ids[None,:]` (`modeling_ideogram4.py:154`).
//! For B=1 unpadded every real slot has segment 1 → the mask is all-True →
//! droppable (pass `None` to SDPA). See [`build_segment_mask_dense`].

use crate::models::ideogram4::Ideogram4Config;

/// Image grid coords start here so they never collide with text positions
/// (`constants.py:8`). Text positions start at 0 and stay below `max_text_tokens`.
pub const IMAGE_POSITION_OFFSET: i64 = 65536;
/// Indicator value for text (LLM) tokens (`constants.py:4`).
pub const LLM_TOKEN_INDICATOR: i64 = 3;
/// Indicator value for image (output) tokens (`constants.py:3`).
pub const OUTPUT_IMAGE_INDICATOR: i64 = 2;
/// Segment id for padding slots (`constants.py:1`).
pub const SEQUENCE_PADDING_INDICATOR: i64 = -1;

/// Built packed-sequence inputs for ONE prompt (B=1).
///
/// All arrays are host-side, length `seq_len = max_text_tokens + num_image_tokens`
/// unless noted. The infer loop uploads these; `position_ids` (the three
/// columns) feed `mrope::build_cos_sin`.
#[derive(Debug, Clone)]
pub struct Ideogram4Inputs {
    /// Token ids, length `seq_len`. Pad + image slots are 0; text slots carry
    /// the prompt token ids. (Used by the text-encode path; the DiT uses
    /// `llm_features`, not token_ids directly.)
    pub token_ids: Vec<i64>,
    /// Per-token position id, axis T. Length `seq_len`. `u32` for the MRoPE
    /// builder (positions are non-negative; pad slots are 0).
    pub pos_t: Vec<u32>,
    /// Per-token position id, axis H. Length `seq_len`.
    pub pos_h: Vec<u32>,
    /// Per-token position id, axis W. Length `seq_len`.
    pub pos_w: Vec<u32>,
    /// Text-only position id, axis T (the `text_position_ids` the Qwen3-VL
    /// encode path consumes; image slots stay 0). Length `seq_len`.
    pub text_pos_t: Vec<u32>,
    /// Text-only position id, axis H. Length `seq_len`.
    pub text_pos_h: Vec<u32>,
    /// Text-only position id, axis W. Length `seq_len`.
    pub text_pos_w: Vec<u32>,
    /// Segment ids, length `seq_len`. Pad = -1, real = 1.
    pub segment_ids: Vec<i64>,
    /// Role indicator, length `seq_len`. Pad = 0, text = 3, image = 2.
    pub indicator: Vec<i64>,
    /// Image grid rows = `height / patch`.
    pub grid_h: usize,
    /// Image grid cols = `width / patch`.
    pub grid_w: usize,
    /// `grid_h * grid_w`.
    pub num_image_tokens: usize,
    /// Text token count (= `max_text_tokens` for B=1).
    pub max_text_tokens: usize,
    /// Total sequence length = `max_text_tokens + num_image_tokens`.
    pub seq_len: usize,
}

/// Build the packed-sequence inputs for ONE prompt (B=1).
///
/// `prompt_token_ids` is the tokenized prompt (chat-template applied upstream).
/// `height`/`width` are the target image dims; both must be divisible by
/// `patch = patch_size(2) * ae_scale(8) = 16` (`pipeline_ideogram4.py:354-358`).
///
/// Returns `Err` on non-divisible dims (mirrors the reference's `ValueError`).
pub fn build_inputs(
    prompt_token_ids: &[i64],
    height: usize,
    width: usize,
    _config: &Ideogram4Config,
) -> Result<Ideogram4Inputs, String> {
    // patch = patch_size * ae_scale_factor = 2 * 8 = 16
    // (Ideogram4PipelineConfig.patch_size=2, ae_scale_factor=8;
    //  pipeline_ideogram4.py:244-245,354).
    const PATCH: usize = 2 * 8;
    if height % PATCH != 0 || width % PATCH != 0 {
        return Err(format!(
            "height/width must be divisible by patch_size*ae_scale_factor={PATCH}; got {height}x{width}"
        ));
    }
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    let num_image_tokens = grid_h * grid_w;

    let num_text = prompt_token_ids.len();
    // B=1: max_text_tokens == this prompt's token count (no cross-prompt pad).
    let max_text_tokens = num_text;
    let seq_len = max_text_tokens + num_image_tokens;

    // Image position ids (t=0, h=row, w=col) + offset. Row-major flatten:
    // h_idx[k] = k / grid_w (row), w_idx[k] = k % grid_w (col), t_idx[k] = 0.
    // (pipeline_ideogram4.py:367-370 — h expand over cols, w expand over rows.)
    let mut image_pos_t = vec![0u32; num_image_tokens];
    let mut image_pos_h = vec![0u32; num_image_tokens];
    let mut image_pos_w = vec![0u32; num_image_tokens];
    for k in 0..num_image_tokens {
        let row = (k / grid_w) as i64;
        let col = (k % grid_w) as i64;
        image_pos_t[k] = IMAGE_POSITION_OFFSET as u32; // 0 + offset
        image_pos_h[k] = (row + IMAGE_POSITION_OFFSET) as u32;
        image_pos_w[k] = (col + IMAGE_POSITION_OFFSET) as u32;
    }

    // Allocate the full sequence (pad slots = 0 / -1 per the reference).
    let mut token_ids = vec![0i64; seq_len];
    let mut pos_t = vec![0u32; seq_len];
    let mut pos_h = vec![0u32; seq_len];
    let mut pos_w = vec![0u32; seq_len];
    let mut text_pos_t = vec![0u32; seq_len];
    let mut text_pos_h = vec![0u32; seq_len];
    let mut text_pos_w = vec![0u32; seq_len];
    let mut segment_ids = vec![SEQUENCE_PADDING_INDICATOR; seq_len];
    let mut indicator = vec![0i64; seq_len];

    // pad_len = max_text_tokens - num_text = 0 for B=1.
    let pad_len = max_text_tokens - num_text;
    let total_unpadded = num_text + num_image_tokens;
    let offset = pad_len;

    // [pad zeros][text tokens][image tokens]; image token slots stay 0.
    token_ids[offset..offset + num_text].copy_from_slice(prompt_token_ids);

    // Text positions: arange(num_text) on all 3 axes (degenerate MRoPE).
    for j in 0..num_text {
        let k = (j as i64) as u32;
        let si = offset + j;
        pos_t[si] = k;
        pos_h[si] = k;
        pos_w[si] = k;
        text_pos_t[si] = k;
        text_pos_h[si] = k;
        text_pos_w[si] = k;
    }
    // Image positions: stamp the offset image coords into the image span.
    for j in 0..num_image_tokens {
        let si = offset + num_text + j;
        pos_t[si] = image_pos_t[j];
        pos_h[si] = image_pos_h[j];
        pos_w[si] = image_pos_w[j];
        // text_position_ids leaves image slots at 0 (reference only writes the
        // text span into text_position_ids; pipeline_ideogram4.py:392).
    }

    // Indicator: text=3, image=2, pad=0.
    for si in offset..offset + num_text {
        indicator[si] = LLM_TOKEN_INDICATOR;
    }
    for si in offset + num_text..seq_len {
        indicator[si] = OUTPUT_IMAGE_INDICATOR;
    }

    // Segment id 1 for the real (text+image) span; padding stays -1.
    for si in offset..offset + total_unpadded {
        segment_ids[si] = 1;
    }

    Ok(Ideogram4Inputs {
        token_ids,
        pos_t,
        pos_h,
        pos_w,
        text_pos_t,
        text_pos_h,
        text_pos_w,
        segment_ids,
        indicator,
        grid_h,
        grid_w,
        num_image_tokens,
        max_text_tokens,
        seq_len,
    })
}

/// Build the dense block-diagonal attention mask `segment_ids[i] == segment_ids[j]`.
///
/// Returns a flat `seq_len * seq_len` bool array (row-major, `mask[i*L+j]`),
/// `true` = attend. This mirrors `(seg[:,None]==seg[None,:])`
/// (`modeling_ideogram4.py:154`). For SDPA the caller reshapes to `[B,1,L,L]`.
///
/// **For B=1 unpadded, every real slot has segment id 1, so the mask is all-True
/// → pass `None` to SDPA instead (the cheaper path).** Use
/// [`is_mask_all_true`] to decide. This helper exists for the padded / packed
/// path (pad slots with segment -1 never attend to the real -1≠1 span, and the
/// pad span attends only within itself).
pub fn build_segment_mask_dense(segment_ids: &[i64]) -> Vec<bool> {
    let l = segment_ids.len();
    let mut mask = vec![false; l * l];
    for i in 0..l {
        for j in 0..l {
            mask[i * l + j] = segment_ids[i] == segment_ids[j];
        }
    }
    mask
}

/// True iff the segment mask is entirely all-True (every token shares one
/// segment id), in which case SDPA can take `mask = None`.
///
/// For the B=1 unpadded path all real tokens have segment id 1 and there are no
/// pad slots, so this is `true` and the dense mask is unnecessary.
pub fn is_mask_all_true(segment_ids: &[i64]) -> bool {
    if segment_ids.is_empty() {
        return true;
    }
    let first = segment_ids[0];
    segment_ids.iter().all(|&s| s == first)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Ideogram4Config {
        Ideogram4Config::default()
    }

    #[test]
    fn grid_math_512() {
        // 512×512 / 16 = 32×32 → 1024 image tokens.
        let inp = build_inputs(&[10, 11, 12], 512, 512, &cfg()).unwrap();
        assert_eq!(inp.grid_h, 32);
        assert_eq!(inp.grid_w, 32);
        assert_eq!(inp.num_image_tokens, 1024);
        assert_eq!(inp.max_text_tokens, 3);
        assert_eq!(inp.seq_len, 3 + 1024);
    }

    #[test]
    fn grid_math_nonsquare() {
        // 256×512 / 16 = 16×32 → 512 image tokens.
        let inp = build_inputs(&[1], 256, 512, &cfg()).unwrap();
        assert_eq!(inp.grid_h, 16);
        assert_eq!(inp.grid_w, 32);
        assert_eq!(inp.num_image_tokens, 16 * 32);
    }

    #[test]
    fn rejects_non_divisible_dims() {
        assert!(build_inputs(&[1], 500, 512, &cfg()).is_err());
        assert!(build_inputs(&[1], 512, 100, &cfg()).is_err());
        // 16-divisible passes.
        assert!(build_inputs(&[1], 512, 512, &cfg()).is_ok());
    }

    #[test]
    fn layout_text_then_image_b1() {
        // Toy prompt of 3 tokens, tiny image (32×32 = grid 2x2 → 4 image tokens).
        let prompt = vec![100i64, 200, 300];
        let inp = build_inputs(&prompt, 32, 32, &cfg()).unwrap();
        assert_eq!(inp.grid_h, 2);
        assert_eq!(inp.grid_w, 2);
        assert_eq!(inp.num_image_tokens, 4);
        assert_eq!(inp.seq_len, 7); // 3 text + 4 image

        // token_ids: [100,200,300, 0,0,0,0] (image slots zero).
        assert_eq!(inp.token_ids, vec![100, 200, 300, 0, 0, 0, 0]);

        // indicator: [3,3,3, 2,2,2,2].
        assert_eq!(inp.indicator, vec![3, 3, 3, 2, 2, 2, 2]);

        // segment_ids: all real → all 1 (no pad for B=1).
        assert_eq!(inp.segment_ids, vec![1, 1, 1, 1, 1, 1, 1]);
        assert!(is_mask_all_true(&inp.segment_ids));
    }

    #[test]
    fn text_positions_arange_all_axes() {
        let prompt = vec![5i64, 6, 7];
        let inp = build_inputs(&prompt, 32, 32, &cfg()).unwrap();
        // Text span positions 0,1,2 on every axis.
        assert_eq!(&inp.pos_t[..3], &[0, 1, 2]);
        assert_eq!(&inp.pos_h[..3], &[0, 1, 2]);
        assert_eq!(&inp.pos_w[..3], &[0, 1, 2]);
        // text_position_ids match in the text span, 0 in image span.
        assert_eq!(&inp.text_pos_t[..3], &[0, 1, 2]);
        assert_eq!(inp.text_pos_t[3], 0);
        assert_eq!(inp.text_pos_t[6], 0);
    }

    #[test]
    fn image_positions_offset_and_grid() {
        // grid 2x2: image tokens at (h,w) = (0,0),(0,1),(1,0),(1,1) row-major.
        let prompt = vec![5i64, 6, 7];
        let inp = build_inputs(&prompt, 32, 32, &cfg()).unwrap();
        let off = IMAGE_POSITION_OFFSET as u32;
        // image span starts at index 3.
        // t = 0 + offset for all image tokens.
        assert_eq!(&inp.pos_t[3..7], &[off, off, off, off]);
        // h = row + offset: rows [0,0,1,1].
        assert_eq!(&inp.pos_h[3..7], &[off, off, off + 1, off + 1]);
        // w = col + offset: cols [0,1,0,1].
        assert_eq!(&inp.pos_w[3..7], &[off, off + 1, off, off + 1]);
    }

    #[test]
    fn position_columns_match_mrope_input_length() {
        // The three position columns must all be seq_len (build_cos_sin requires
        // equal-length T/H/W). C2-F8 / C3-F12 resolution: S == L.
        let inp = build_inputs(&[1, 2, 3, 4], 64, 64, &cfg()).unwrap();
        assert_eq!(inp.pos_t.len(), inp.seq_len);
        assert_eq!(inp.pos_h.len(), inp.seq_len);
        assert_eq!(inp.pos_w.len(), inp.seq_len);
        assert_eq!(inp.indicator.len(), inp.seq_len);
        assert_eq!(inp.segment_ids.len(), inp.seq_len);
        assert_eq!(inp.token_ids.len(), inp.seq_len);
    }

    #[test]
    fn dense_mask_all_true_single_segment() {
        let seg = vec![1i64, 1, 1, 1];
        let m = build_segment_mask_dense(&seg);
        assert_eq!(m.len(), 16);
        assert!(m.iter().all(|&b| b));
    }

    #[test]
    fn dense_mask_block_diagonal_with_padding() {
        // Pad slots (-1) and real slots (1): real attends real, pad attends pad,
        // cross is False.
        let seg = vec![-1i64, 1, 1];
        let m = build_segment_mask_dense(&seg);
        // mask[0][0] = (-1==-1) = true; mask[0][1]=(-1==1)=false.
        assert!(m[0]); // (0,0)
        assert!(!m[1]); // (0,1)
        assert!(!m[2]); // (0,2)
        assert!(!m[3]); // (1,0)
        assert!(m[4]); // (1,1)
        assert!(m[5]); // (1,2)
        assert!(!is_mask_all_true(&seg));
    }
}
