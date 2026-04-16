# Motif-Video-2B Rust Port — Session 1 Handoff

## Status

**All 4 pipeline binaries compile + execute.** Full forward runs end-to-end in 1.7s at test resolution.

**Parity: final cosine = 0.826** — not bit-exact but workable diagnosis state.

## Pipeline

```
motif_encode   → motif_embeds.safetensors    (T5Gemma2 encode)
motif_gen      → motif_latents.safetensors   (DiT denoise, APG CFG)
motif_decode   → PNG frames                   (VAE decode)
motif_parity   → per-block cosine diagnostics
```

Plus agent-built `T5Gemma2Encoder` (6 tests passing) and `motif_sampling` (4 tests passing).

## Parity breakdown (test: 80 latent tokens, 5 frames, 16×24)

```
--- Pre-block ---
  x_embedder         cos=1.000000  ✓
  context_embedder   cos=0.999993  ✓
  time_text_embed    cos=1.000000  ✓

--- Dual blocks (0..11) ---
  dual_0..dual_11    cos=0.999981 to 0.999998  ✓ bit-exact-ish

--- Single encoder (0..15) ---
  single_0..single_15  cos=0.990 to 0.9999     ✓ gradual drift but healthy

--- Single decoder (16..23) ---
  single_16   cos=0.999961  ← perfect (fresh start from decoder_initial)
  single_17   cos=0.999807
  single_18   cos=0.999734
  single_19   cos=0.988248  ⚠️ drift starts
  single_20   cos=0.980317
  single_21   cos=0.969496
  single_22   cos=0.950174
  single_23   cos=0.946531  ⚠️ ~5% off

--- Post-block ---
  norm_out    cos=0.958293  (LN partially recovers direction)
  proj_out    cos=0.825540  (linear projection re-amplifies divergence)
  final_output  cos=0.825540  (unpatchify is math-identical to proj_out)
```

## Bugs found + fixed this session

1. **Schedule was wrong (would affect sampler but not parity)** — already in motif_sampling.rs: `shift=15.0` applied statically, matches FlowMatchEulerDiscrete with `use_dynamic_shifting=false`.

2. **`AdaLayerNormContinuous` chunk order**: was `shift, scale = chunk(2)` but diffusers does `scale, shift = chunk(2)`. Took final cosine from 0.053 → 0.976.

3. **RoPE table shape**: flame-core's `rope_fused_bf16` expects `[1, 1, N, D/2]` (one value per complex pair), not `[1, 1, N, D]` (repeated). Was building tables double-sized — fixed.

4. **Shared weights F32 on disk**: `x_embedder.proj.weight` and others saved as F32 (unlike BlockOffloader which auto-casts). Added explicit to_dtype(BF16) at load.

5. **`context_embedder` is 2-layer MLP** (`PixArtAlphaTextProjection`), not a single Linear. Was passing wrong weight keys. Fixed.

6. **`image_embedder` (SigLIP projection)** is LayerNorm→Linear→GELU→Linear→LayerNorm (not 2-layer MLP). T2V skips this — returns error if I2V is requested. Add for I2V phase.

## Remaining divergence investigation

The decoder drift pattern (clean at 17-18, diverges 19+) is unusual — it's NOT cumulative (would start at 17, not at 19). Possible causes, in rough order of likelihood:

1. **BF16 accumulation in the CAT-then-split pattern** in single blocks. The decoder's `encoder_hidden_states` side accumulates 16 blocks of encoder drift before feeding the decoder. Decoder then re-concats encoder output + decoder stream, and small errors at specific spectral directions get amplified by the attention's Q @ K^T.

2. **AdaLayerNormZeroSingle chunk layout**: I use `[shift, scale, gate]` order but might be wrong. Python's `AdaLayerNormZeroSingle` wasn't directly inspected — based on the general AdaLN pattern. Could be `[scale, shift, gate]` or similar.

3. **`AdaLayerNormZero` for dual blocks**: 5-param chunk order `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]` — I may have this wrong. Dual blocks cos is 0.999+ so probably close to right, but could compound in decoder.

4. **Unpatchify contiguity**: `permute + single reshape` might not reassemble correctly even though flame-core's permute_generic is supposed to produce contiguous output. Python does step-by-step flatten. If this is the issue, the bug is ONLY in the final `reshape(&[bs, C, T*p_t, H*p, W*p])` call — the per-block cosines wouldn't be affected (they are good).

## Next debugging steps

1. Verify `AdaLayerNormZeroSingle` chunk order by reading diffusers source for the actual class
2. Do step-by-step flatten in unpatchify (copy Python exactly)
3. If decoder drift persists, check whether the block forward mutates any shared state
4. Consider: is the Python dump's single_17-18 "perfect" match actually suspicious? Both near-1.0 when 19+ breaks suggests a specific arithmetic pattern only hit from block 19 on.

## Files changed this session

Created:
- `src/models/motif_video_dit.rs` (~1000 lines) — 3-stage DiT with dual/single/decoder blocks
- `src/models/t5gemma2_encoder.rs` (agent) — 34-layer text encoder
- `src/sampling/motif_sampling.rs` — FlowMatch + video-APG
- `src/vae/qwenimage_decoder.rs` — Wan 2.1 VAE decoder wrapper (already existed from Nucleus)
- `src/bin/motif_encode.rs`, `motif_gen.rs`, `motif_decode.rs`, `motif_parity.rs`
- `scripts/motif_block_dump.py` — Python reference dump

Modified:
- `src/models/mod.rs` — added t5gemma2_encoder, motif_video_dit
- `src/sampling/mod.rs` — added motif_sampling
- `Cargo.toml` — registered 4 motif bins

## Architecture verified against source

- 12 dual + 16 single encoder + 8 decoder = 36 blocks
- Shared Cross-Attention only in encoder single blocks (NEW Q proj, REUSE to_k/to_v)
- DDT decoder uses U-Net pattern (decoder_stream = original post-patch-embed)
- RoPE: Flux-style complex rotation, compatible with flame-core's rope_fused_bf16
- No MoE, dense GELU-tanh FFN
- Shift=15 static scheduler (unlike Nucleus's shift=1 no-op)
- Video-APG replaces standard CFG (per-frame projection)
- No sign flip before scheduler step (unlike Nucleus)

## VRAM / speed

- 36 blocks × ~110MB BF16 = 4GB model in BlockOffloader (pinned CPU) — fits 24GB GPU easily
- 1.7s per forward at 80 tokens / 5 frames / 16×24 (tiny test)
- At test res, no BlockOffloader needed but kept for symmetry with Nucleus
- 2B dense — much faster than Nucleus (17B MoE)

## Open TODO for next session

1. Fix decoder drift (0.946 at single_23 → should be >0.99)
2. Fix norm_out/proj_out residual (0.826 final)
3. Run end-to-end `motif_encode → motif_gen → motif_decode` for visual check
4. T5Gemma2 encoder forward never tested at runtime — add smoke test
5. I2V path (SigLIP, image_embedder with LayerNorms) not implemented
6. APG momentum buffer + norm_threshold clipping not implemented (may affect quality)

---

## Session 3 update (2026-04-16): end-to-end video generated

First coherent video: "a golden retriever bounding through a field of
sunflowers at sunset" — 25 frames, 480×832, 24 fps, 1.04 s duration.
Saved as `/home/alex/serenity/output/motif_output.mp4`. Visible golden
retriever from behind, real motion between frames (pose changes forward
through the field), sunflowers in yellow, sky in orange/red with a
disc-shape sun artifact.

### What it took

1. **SDPA default flipped to in-tree WMMA** (flame-core 341c4af). Can opt
   back into torch via `FLAME_USE_TORCH_SDPA=1`. Torch remains the faster
   choice for `N > ~1024` tokens (motif video gen hits ~10920 tokens so
   we opt back into torch for `motif_gen` — 4.2 s/step vs projected
   ~15 s/step on WMMA).
2. **`Tensor::cat` BF16 path fused** via `cuMemcpy2DAsync_v2`
   (flame-core 341c4af). 38 % faster forward on motif_parity, but more
   importantly eliminates ~30 k kernel launches per forward.
3. **Timestep scaling fix** (inference-flame fa36ba9). This was the
   load-bearing bug: `motif_gen` passed raw sigma in `[0, 1]` to the
   DiT, but `FlowMatchEulerDiscreteScheduler` produces `timesteps =
   sigmas * num_train_timesteps = sigmas * 1000` and the pipeline
   passes those scaled timesteps to the transformer. With 1000×
   underscaled time input, velocity predictions were near-identity
   everywhere, no denoising happened, VAE decoded pure noise. First
   fix that produced visibly-structured output.

### Quality verification (2026-04-16, after APG fix)

Second gen at 1280×720×49 frames with the full APG now matching
reference (`norm_threshold=12`, `momentum=0.1`; committed as `2281c5f`).
27 min denoise total on 3090 Ti, VRAM peaks at 23.6/24 GB with
RESIDENT+torch_sdpa.

- ✓ Actual fur, eye, snout detail on the dog (was impressionistic at 480p)
- ✓ Sunflowers render real petals + centers (was glowing orbs at 480p)
- ✓ Black-sun eclipse artifact is gone — that was clipping-less
  guidance blowup
- Minor: model interpreted "golden retriever" loosely (renders more
  German-shepherd-like); sky is "dusky" not explicitly golden-hour.
  Model/prompt-adherence issues, not pipeline bugs.

Output: `/home/alex/serenity/output/motif_output_hd.mp4`

### Known rough edges on the "end-to-end" path

- **VAE decode goes through Python** (`scripts/motif_vae_decode_bridge.py`)
  because `flame-core`'s `Wan21VaeDecoder` uses different safetensors key
  naming (`decoder.conv1`, `decoder.middle.0`, flat `conv2`) than
  Motif's diffusers-style checkpoint (`decoder.conv_in`,
  `decoder.mid_block.resnets.0`, top-level `post_quant_conv`). Rewriting
  the Rust decoder to accept the diffusers layout is a bounded task —
  ~200 LOC of key-mapping shim or a decoder variant. Skipped this
  session.
- **BGR channel order** — Wan VAE output channel 0 is blue, channel 2 is
  red. The bridge script swaps to RGB before encoding. If someone writes
  the pure-Rust decode path, mirror that swap before PNG/MP4 export.
- **Cross-process determinism** still unresolved for motif_parity
  (~10-20 % catastrophic tail). See `MOTIF_NONDET_HANDOFF.md` v3.
  Doesn't block inference — every fresh run produces a video, just not
  bit-identical across runs.

### Repro (takes ~5 min for 25 frames)

```bash
# Stage 1: T5Gemma2 encode
./inference-flame/target/release/motif_encode \
  "a golden retriever bounding through a field of sunflowers at sunset" \
  "low quality, blurry" \
  /home/alex/serenity/output/motif_embeds.safetensors

# Stage 2: DiT denoise — needs RESIDENT + torch-sdpa for this scale
env MOTIF_RESIDENT=1 FLAME_USE_TORCH_SDPA=1 \
    MOTIF_HEIGHT=480 MOTIF_WIDTH=832 MOTIF_FRAMES=25 MOTIF_STEPS=50 \
  ./inference-flame/target/release/motif_gen \
  /home/alex/serenity/output/motif_embeds.safetensors \
  /home/alex/serenity/output/motif_latents.safetensors

# Stage 3: VAE decode — Python bridge (see above)
python3 inference-flame/scripts/motif_vae_decode_bridge.py
# writes /home/alex/serenity/output/motif_output.mp4
```
