//! 4-text-encoder concat helpers for HiDream-I1.
//!
//! HiDream-I1's `HiDreamImageTransformer2DModel.forward` takes:
//!   - `encoder_hidden_states = [t5_seq, llama_all_layers_stack]`
//!     where `t5_seq`: `[B, S_t5, 4096]` (T5-XXL hidden states), and
//!     `llama_all_layers_stack`: `[num_llama_layers, B, S_llama, 4096]`
//!     (Llama-3.1-8B hidden_states[1:] stacked).
//!   - `pooled_embeds = cat([clip_l_pool, clip_g_pool], dim=-1)` → `[B, 2048]`.
//!
//! `transformer_hidream_image.py:408-421` then:
//!   1. Picks out `[llama_all_layers_stack[k] for k in self.llama_layers]`
//!      (the config-selected subset).
//!   2. Projects each via `caption_projection[i]` to `inner_dim` (2560).
//!   3. Projects T5 via `caption_projection[-1]`.
//!   4. The resulting list has length `num_double + num_single + 1`.
//!
//! ## ai-toolkit pipeline reference
//! `pipeline_hidream_image.py:_encode_prompt` (lines 375-438) shows the
//! exact ordering: `[t5, llama_stack]` (T5 first, Llama stack second)
//! and `pooled = cat([clip_l, clip_g], -1)`.
//!
//! # Tensor shapes
//! - `t5`:           `[B, S_t5, 4096]` BF16
//! - `llama_stack`:  `[L_llama_total, B, S_llama, 4096]` BF16 — stacked
//!                   `outputs.hidden_states[1:]` (drops the embedding layer)
//! - `clip_l_pool`:  `[B, 768]` BF16
//! - `clip_g_pool`:  `[B, 1280]` BF16
//!
//! L_llama_total is the encoder's full layer count (Llama-3.1-8B = 32).
//! `self.llama_layers` is a smaller list of layer indices the DiT actually
//! consumes (one per double + single + an extra final selector). The
//! length matches `num_double + num_single + 1`.

use flame_core::{Result, Tensor};

/// Borrowed handle to the four encoder outputs the DiT consumes.
///
/// The caller (trainer or sampler) is responsible for:
///   * Running CLIP-L / CLIP-G / T5 / Llama once per prompt.
///   * Stacking Llama hidden states along dim 0 (`torch.stack(hidden_states[1:])`).
///   * Concatenating CLIP-L + CLIP-G pooled along dim -1 OUTSIDE this struct,
///     OR using [`HiDreamEncoderInputs::pool_concat`] to do it on GPU.
pub struct HiDreamEncoderInputs<'a> {
    /// `[B, S_t5, 4096]` BF16.
    pub t5: &'a Tensor,
    /// `[L_llama_total, B, S_llama, 4096]` BF16 (stacked Llama hidden states).
    pub llama_stack: &'a Tensor,
    /// `[B, 768]` BF16 (CLIP-L `pooler_output` or last-hidden first-token).
    pub clip_l_pool: &'a Tensor,
    /// `[B, 1280]` BF16 (CLIP-G `text_projection` output).
    pub clip_g_pool: &'a Tensor,
}

/// Owned variant used when the trainer wants to pre-compute a cache.
/// Same fields as [`HiDreamEncoderInputs`] but the tensors live here.
#[derive(Clone)]
pub struct OwnedEncoderInputs {
    pub t5: Tensor,
    pub llama_stack: Tensor,
    pub clip_l_pool: Tensor,
    pub clip_g_pool: Tensor,
}

impl OwnedEncoderInputs {
    /// Borrow as `HiDreamEncoderInputs<'_>`.
    pub fn as_ref(&self) -> HiDreamEncoderInputs<'_> {
        HiDreamEncoderInputs {
            t5: &self.t5,
            llama_stack: &self.llama_stack,
            clip_l_pool: &self.clip_l_pool,
            clip_g_pool: &self.clip_g_pool,
        }
    }
}

impl<'a> HiDreamEncoderInputs<'a> {
    /// Compute `pooled_embeds = cat([clip_l_pool, clip_g_pool], dim=-1)`
    /// → `[B, 2048]` BF16.
    ///
    /// Mirrors `pipeline_hidream_image.py:420`:
    /// `pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)`
    pub fn pool_concat(&self) -> Result<Tensor> {
        Tensor::cat(&[self.clip_l_pool, self.clip_g_pool], 1)
    }

    /// Pick out the `[B, S_llama, 4096]` slice for one layer index.
    ///
    /// Mirrors `transformer_hidream_image.py:410`:
    /// `encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]`.
    ///
    /// Returns `[B, S_llama, 4096]` BF16 (the leading layer dim is squeezed).
    pub fn pick_llama_layer(&self, layer_idx: usize) -> Result<Tensor> {
        // llama_stack: [L, B, S, D]. narrow at dim 0, squeeze.
        let picked = self.llama_stack.narrow(0, layer_idx, 1)?;
        // squeeze leading dim of length 1 → [B, S, D]
        picked.squeeze(Some(0))
    }
}
