# LTX-2.3 AV Pipeline — Bug Analysis

Analysis of why the AV forward path produces dark/unusable output while the video-only path works.

## Status
- **Stage 1 (video-only works)**: `forward_video_only()` produces recognizable content.
- **AV forward (broken)**: `forward_audio_video()` produces std=0.33 dark latents. Stage 2 refinement can't recover.
- **Speed**: 300s/step (should be ~30s). Audio head dim d=64 falls back to slow GEMM.

## Files Involved
- `src/models/ltx2_model.rs` — main model (3601 lines)
- `src/bin/ltx2_two_stage.rs` — two-stage AV generation binary
- Reference: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- Reference: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`

---

## Bug 1 — `forward_audio_video()` never computes `prompt_timestep`

### Evidence
`LTX2StreamingModel::forward_video_only()` at `src/models/ltx2_model.rs:2882` computes `prompt_timestep` correctly:

```rust
// 6b. Compute prompt_timestep from prompt_adaln_single (ComfyUI 9-param path)
let prompt_timestep = if let Some(prompt_adaln) = &self.prompt_adaln_single {
    let text_seq_len = enc_hs.shape().dims()[1];
    let prompt_ts = timestep.unsqueeze(1)?.expand(&[batch_size, text_seq_len])?;
    let prompt_ts_scaled = prompt_ts.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
    let prompt_ts_flat = prompt_ts_scaled.reshape(&[batch_size * text_seq_len])?;
    let (prompt_mod, _) = prompt_adaln.forward(&prompt_ts_flat)?;
    // reshape to [B, seq, 2*dim]
    ...
};
```

`forward_audio_video()` at `src/models/ltx2_model.rs:3035` has no equivalent. `prompt_adaln_single` is loaded into the struct but never invoked in the AV path. `audio_prompt_adaln_single` isn't even a field.

### Impact
Text cross-attention uses timestep-independent context. The model was trained with timestep-conditioned context and can't properly modulate text influence per denoising step.

---

## Bug 2 — AV block forward skips KV context modulation

### Evidence — Python reference

`packages/ltx-core/src/ltx_core/model/transformer/transformer.py:380-398`:

```python
def apply_cross_attention_adaln(x, context, attn, q_shift, q_scale, q_gate,
                                prompt_scale_shift_table, prompt_timestep,
                                context_mask, norm_eps):
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv        # ← KV modulated
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
```

Called for both video and audio in the AV block (`transformer.py:236` and `:274`).

### Evidence — Rust AV block

`src/models/ltx2_model.rs:810-824` (inside `LTX2TransformerBlock::forward()`):

```rust
// 2. Video/Audio Cross-Attention with text (AdaLN modulated)
let (v_shift_ca, v_scale_ca, v_gate_ca) =
    self.compute_ada_params_ca(&self.scale_shift_table, temb, b, video_dim)?;
let norm_h2 = rms_norm(&hidden_states, self.norm2_weight.as_ref(), self.eps)?;
let mod_h2 = norm_h2.mul(&v_scale_ca.add_scalar(1.0)?)?.add(&v_shift_ca)?;
// KV (encoder_hidden_states) passed UNMODULATED:
let ca_out = self.attn2.forward(&mod_h2, Some(encoder_hidden_states),
                                 encoder_attention_mask, None, None)?;
hidden_states = hidden_states.add(&ca_out.mul(&v_gate_ca)?)?;

// Same bug for audio branch at lines 819-824
let (a_shift_ca, a_scale_ca, a_gate_ca) =
    self.compute_ada_params_ca(&self.audio_scale_shift_table, temb_audio, b, audio_dim)?;
let norm_a2 = rms_norm(&audio_hidden_states, self.audio_norm2_weight.as_ref(), self.eps)?;
let mod_a2 = norm_a2.mul(&a_scale_ca.add_scalar(1.0)?)?.add(&a_shift_ca)?;
let ca_a_out = self.audio_attn2.forward(&mod_a2, Some(audio_encoder_hidden_states), ...)?;
```

Only the query is modulated. Context (KV) is passed raw. Video-only path at `forward_video_only()` line 714-731 DOES modulate KV correctly — it's purely missing from the AV forward.

### Impact
Q modulation without matching KV modulation produces an attention distribution that doesn't match what the model was trained on. Combined with Bug 1 the effect compounds: wrong context, wrong strength.

---

## Bug 3 — `audio_prompt_adaln_single` + per-block `audio_prompt_scale_shift_table` not loaded

### Evidence
`grep "audio_prompt_adaln"` in `src/models/ltx2_model.rs` → zero matches.

Struct fields present:
- `LTX2StreamingModel::prompt_adaln_single: Option<AdaLayerNormSingle>` ✅
- `LTX2TransformerBlock::prompt_scale_shift_table: Option<Tensor>` ✅

Missing:
- `LTX2StreamingModel::audio_prompt_adaln_single` ❌
- `LTX2TransformerBlock::audio_prompt_scale_shift_table` ❌

### Python reference
`transformer.py:120-122`:
```python
if self.cross_attention_adaln and video is not None:
    self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
if self.cross_attention_adaln and audio is not None:
    self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))
```

`model.py:141-171`:
```python
self.prompt_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
self.audio_prompt_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
```

Both exist in the Lightricks model and both are used in AV mode.

### Impact
Even after fixing Bugs 1-2, audio cross-attention would still be half-broken. The audio text modulation table exists in the checkpoint but is never loaded or used.

---

## Bug 4 — Speed: audio attention falls back to materialized GEMM

### Evidence
- `flame-core/src/cuda/flash_attention_fwd.cu` — flash kernel only supports head_dim=128
- Video self-attn: 32 heads × 128 dim ✅ fast path
- Audio self-attn: 32 heads × 64 dim ❌ GEMM path
- A2V/V2A cross-attn: 32 heads × 64 dim ❌ GEMM path
- d=64 flash kernel was attempted → hangs GPU (0% util)

### Scale of slowdown
- 48 blocks × 6 attention ops per block × 2 modalities ≈ 576 attention calls per step
- Of these, 48 are video self-attn (fast), 528 are d=64 (slow GEMM)
- Per-step: ~300s measured vs ~30s target

### Impact
Not a correctness bug — purely a perf bug. Blocks fix validation by turning a ~5 min feedback loop into a ~1 hour one.

---

## Fix Plan (in order, don't skip)

### Step 1 — Load the missing globals
In `LTX2StreamingModel::load_globals()` at `src/models/ltx2_model.rs:2009`:
- Add `audio_prompt_adaln_single: Option<AdaLayerNormSingle>` field to the struct
- Load it alongside `prompt_adaln_single`:
```rust
let audio_prompt_adaln = load_ada_layer_norm_single(&globals, "audio_prompt_adaln_single", 2).ok();
```

### Step 2 — Load per-block audio prompt scale/shift table
In `LTX2TransformerBlock`:
- Add `audio_prompt_scale_shift_table: Option<Tensor>` field
- In the block loaders (`load_block_from_weights_pretransposed` and `load_block_from_weights_static`):
```rust
audio_prompt_scale_shift_table: get_opt(&format!("{pfx}.audio_prompt_scale_shift_table")),
```
- Also ensure the FP8 resident loader filter at line 2258 includes `audio_prompt_scale_shift` (it already does — verify with the actual checkpoint key name).

### Step 3 — Compute `prompt_timestep` + `audio_prompt_timestep` in AV forward
In `forward_audio_video()` at `src/models/ltx2_model.rs:3035`, after text embeddings are computed (~line 3204), add:

```rust
// Compute prompt_timestep for video cross-attn KV modulation
let video_prompt_ts = if let Some(ref padaln) = self.prompt_adaln_single {
    let text_seq_len = enc_hs.shape().dims()[1];
    let prompt_ts = timestep.unsqueeze(1)?.expand(&[batch_size, text_seq_len])?;
    let prompt_ts_scaled = prompt_ts.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
    let prompt_ts_flat = prompt_ts_scaled.reshape(&[batch_size * text_seq_len])?;
    let (prompt_mod, _) = padaln.forward(&prompt_ts_flat)?;
    Some(prompt_mod.reshape(&[batch_size, text_seq_len, 2 * inner_dim])?)
} else { None };

// Same for audio (using audio_prompt_adaln_single + audio_enc_hs)
let audio_prompt_ts = if let Some(ref apadaln) = self.audio_prompt_adaln_single {
    let text_seq_len = audio_enc_hs.shape().dims()[1];
    let prompt_ts = timestep.unsqueeze(1)?.expand(&[batch_size, text_seq_len])?;
    let prompt_ts_scaled = prompt_ts.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
    let prompt_ts_flat = prompt_ts_scaled.reshape(&[batch_size * text_seq_len])?;
    let (prompt_mod, _) = apadaln.forward(&prompt_ts_flat)?;
    Some(prompt_mod.reshape(&[batch_size, text_seq_len, 2 * audio_inner_dim])?)
} else { None };
```

### Step 4 — Thread `prompt_timestep` through block `forward()`
Add two parameters to `LTX2TransformerBlock::forward()`:
```rust
video_prompt_timestep: Option<&Tensor>,  // [B, seq, 2*video_dim]
audio_prompt_timestep: Option<&Tensor>,  // [B, seq, 2*audio_dim]
```

Inside the cross-attention sections, modulate the context before calling `attn2.forward()` and `audio_attn2.forward()`. Copy the pattern from `forward_video_only()` at lines 715-732:

```rust
// Video CA context modulation
let modulated_v_context = if let (Some(psst), Some(pt)) =
    (&self.prompt_scale_shift_table, video_prompt_timestep) {
    let psst_bc = psst.unsqueeze(0)?.unsqueeze(0)?;
    let seq_len = pt.shape().dims()[1];
    let pt_4d = pt.reshape(&[b, seq_len, 2, video_dim])?;
    let combined = psst_bc.add(&pt_4d)?.to_dtype(DType::BF16)?;
    let shift_kv = combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
    let scale_kv = combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
    fused_modulate(encoder_hidden_states, &scale_kv, &shift_kv)?
} else {
    encoder_hidden_states.clone()
};
let ca_out = self.attn2.forward(&mod_h2, Some(&modulated_v_context), ...)?;

// Same pattern for audio CA with audio_prompt_scale_shift_table + audio_prompt_timestep
```

### Step 5 — Update call sites
In both FP8 resident path (line 3256) and FlameSwap path (line 3305), pass the new params:
```rust
block.forward(
    &hs, &ahs,
    &enc_hs, &audio_enc_hs,
    &v_timestep, &a_timestep,
    &v_ca_ss, &a_ca_ss,
    &v_ca_gate, &a_ca_gate,
    Some((&v_cos, &v_sin)), Some((&a_cos, &a_sin)),
    Some((&ca_v_cos, &ca_v_sin)), Some((&ca_a_cos, &ca_a_sin)),
    None, None,
    video_prompt_ts.as_ref(), audio_prompt_ts.as_ref(),  // NEW
)?;
```

### Step 6 — Test
- Run `ltx2_two_stage` on the same prompt.
- Stage 1 latent stats should still look like current output (similar signal, now better conditioned).
- Stage 2 latent stats should show std ~0.8–1.0 instead of 0.33.
- Decode and visually inspect output frames.

### Step 7 (separate) — Speed
Don't touch until quality is fixed. Options:
1. Debug the d=64 flash kernel (shared mem sizing, grid/block dims).
2. Implement a faster GEMM fallback (fused QK^T+softmax+V via cuBLAS batched strided).
3. Accept ~30 min per gen until a proper kernel lands.

---

## What NOT to do
- Don't touch `forward_video_only()` — it works. Don't "simplify" it.
- Don't try to fix speed and quality in the same change.
- Don't skip step 2 — audio side must be loaded or audio cross-attn remains half-broken.
- Don't pretend the fix is done without decoding latents and looking at frames.

## Verified facts from this session
- FP8 dequant is correct (handoff: matches PyTorch to 6 decimal places).
- Euler step is correct (`sample += velocity * dt` matches official `X0Model` + `EulerDiffusionStep`).
- Noise injection formula for stage 2 is correct: `x = (1-σ)·initial + σ·noise` matches `GaussianNoiser`.
- Key name mismatches are fixed (commit 665adb9).
- A2V/V2A scale/shift/gate modulation is present and uses correct slice indices (lines 952-988).
- The only remaining quality bug is the text cross-attention context modulation described above.
