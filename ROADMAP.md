# FLAME LTX-2 Pipeline Roadmap

## Vision
Pure Rust video+audio generation pipeline. Best quality. Competitive speed. 
Uses everything LTX-2 offers: dual-stream audio+video, spatial/temporal upscalers, 
two-stage distilled+refined pipeline.

## Current State (2026-04-05)
- ✅ Video generation works (Enterprise at warp speed, 480×288)
- ✅ Split RoPE correct (validated against Python)
- ✅ CFG working (guidance_scale=4.0)
- ✅ FlameSwap block swap (72s/step with F32 cache fix)
- ✅ FP8 resident loading (15.3GB on GPU, 74s/step — bottlenecked by per-block transpose)
- ✅ Distilled + Dev model support
- ✅ Fused CUDA kernels (rms_norm, modulate, residual_gate, flash_attention)
- ✅ FP8→BF16 GPU dequant kernel
- ✅ Gemma-3 encoder in Rust (763 lines, compiles, not wired)
- ❌ Audio stream (weights exist, skipped)
- ❌ Spatial/temporal upscalers
- ❌ Two-stage pipeline
- ❌ VAE decode in Rust

## Phase 1: FP8 Speed (Target: 20s/step)

**Problem**: 74s/step with FP8 resident. Bottleneck is `pre_transpose_weight` — 
10 transposes per block, each allocates + copies.

**Fix**: Pre-transpose ALL weights at load time (once, during init). 
Store pre-transposed weights in the ResidentBlock. During forward, 
just hand them to the GEMM — zero allocation, zero transpose.

This is what ComfyUI does: weights are loaded, transposed once, kept resident.

Steps:
1. In `load_fp8_resident`, after loading each weight, transpose it immediately
2. Store the transposed tensor in ResidentBlock
3. In `to_bf16_block`, FP8 weights: dequant + transpose (one kernel or two)
4. BF16 weights: already transposed at load time, just Arc clone
5. `load_block_from_weights_static` skips `pre_transpose_weight` for pre-transposed weights
6. Target: 17-25s/step

## Phase 2: Quality Parity

**Problem**: We haven't compared Rust output against Python at identical settings.

Steps:
1. Use distilled FP8 checkpoint (just downloaded)
2. Generate reference frames from Python (manual block-swap script)
3. Compare latents numerically (max diff, correlation)
4. Fix any remaining math discrepancies
5. Test with detailed cinematic prompts

## Phase 3: Full Pipeline

### 3a: Two-Stage Pipeline
1. Stage 1: Distilled model at half resolution (fast, 8 steps)
2. Spatial upscaler 2x (small model, fast)
3. Stage 2: Dev model refines at full resolution (few steps)

### 3b: Audio Stream
1. Load audio weights alongside video (dual-stream)
2. Wire audio forward path (self-attn, cross-attn, FFN)
3. Audio-to-video and video-to-audio cross-attention
4. Audio latent output
5. Audio VAE decode (102MB model)
6. Vocoder (BigVGAN, 111MB model)

### 3c: Spatial Upscaler
1. Port the spatial upscaler model (~small transformer)
2. Wire into pipeline: latent → upscale 1.5x or 2x → refine

### 3d: Temporal Upscaler
1. Port the temporal upscaler model
2. Wire: 9 frames → 17 frames (2x interpolation in latent space)

## Phase 4: Rust-Native Components
1. Wire Gemma-3 encoder (already written, needs testing)
2. Port FeatureExtractorV2 (RMS norm + rescale + aggregate_embed)
3. Port Video VAE decoder (3D conv decoder)
4. Port Audio VAE decoder
5. Port BigVGAN vocoder
6. Eliminate ALL Python dependencies

## Phase 5: Performance
1. CUDA graphs for the block loop
2. FP8 native GEMM (cublasLt with FP8 input)
3. Batch CFG (uncond+cond in single forward)
4. torch.compile-equivalent kernel fusion
5. Multi-GPU support (DDP/FSDP with per-GPU FlameSwap)

## Priority Order
1. Phase 1 (FP8 speed) — immediate, biggest user impact
2. Phase 2 (quality parity) — validate before adding features
3. Phase 3a (two-stage) — highest quality improvement
4. Phase 3b (audio) — unique feature, differentiator
5. Phase 3c/3d (upscalers) — quality at scale
6. Phase 4 (Rust-native) — eliminate Python
7. Phase 5 (performance) — competitive with/beat ComfyUI
