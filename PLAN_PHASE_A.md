# Phase A: Quality & Correctness — Execution Plan

## Status: IN PROGRESS
## Date: 2026-04-05

### What Works
- Pure Rust LTX-2.3 22B denoising with CFG=4.0
- Split RoPE (fixed from interleaved)
- Official 4096-dim Gemma embeddings (cached via Python)
- FlameSwap triple-pipelined block swapper
- Fused CUDA kernels: rms_norm, modulate, residual_gate, flash_attention
- VAE decode via Python
- Output: 288×480, 9 frames, recognizable video ("cat in garden", "starship in space")

### Known Issues
1. **Negative embedding is zeros** — should be official empty-string encoding
2. **FP8 checkpoint not supported** — `load_file_filtered` in flame-core skips F8_E4M3
3. **FlameSwap wiring reverted** — `init_swap()` + async forward path lost, model uses slow sync fallback
4. **Key prefix handling** — model.diffusion_model. prefix needs consistent stripping
5. **Gemma encoder exists in Rust but unused** — 763 lines at gemma3_encoder.rs, not wired

### Task List

#### Task 1: Cache proper negative embedding ✅ READY TO RUN
- Script: `/home/alex/cache_official_embeds.py` (modify for empty string)
- Output: `/home/alex/EriDiffusion/inference-flame/cached_ltx2_negative.safetensors`
- Then update `ltx2_generate.rs` to load it instead of zeros

#### Task 2: Add FP8 dequant to flame-core serialization
- File: `flame-core/src/serialization.rs`
- Lines 450, 574: add "F8_E4M3" to the dtype match
- Need to read weight_scale from header, dequant: `bf16_value = fp8_byte_to_f32 * scale`
- Scale keys: `foo.weight_scale` (scalar F32) applies to `foo.weight` (F8_E4M3)

#### Task 3: Re-wire FlameSwap permanently
- Add `swap: Option<flame_swap::FlameSwap>` back to LTX2StreamingModel
- Add `init_swap(&mut self)` method
- Add async forward path in `forward_video_only`
- Add `key_prefix: String` field
- Wire in ltx2_generate.rs: call `model.init_swap()?` after load_globals

#### Task 4: Test at 544×960
- After tasks 1-3, test at official resolution
- 544×960, 9 frames, 10 steps, CFG=4.0
- Compare quality against 288×480

#### Task 5: Audit forward path (skeptic review)
- Compare Rust `forward_video_only` against official Python line-by-line
- Check: timestep scaling, adaln param extraction, cross-attention modulation
- Check: Euler step formula vs official `EulerDiffusionStep`
- Check: output processing (scale_shift_table + norm_out + proj_out)
- Verify sigma schedule matches official for dev model

### Files Modified This Session
- `flame-core/src/cuda/device_lt.rs` — stream set to null (default)
- `flame-core/src/cuda/ffi.rs` — FFI for fused kernels + flash attention
- `flame-core/src/ops/fused_inference.rs` — Rust wrappers for fused kernels
- `flame-core/src/cuda/fused_*.cu` — CUDA kernels
- `flame-core/src/cuda/flash_attention_fwd.cu` — Flash attention
- `flame-core/src/sdpa.rs` — Flash attention dispatch
- `flame-core/src/tensor.rs` — `from_bf16_slice_gpu` constructor
- `flame-core/build.rs` — Added kernel .cu files
- `flame-swap/src/swap.rs` — Triple pipeline, F32/FP8 support, split dtype enum
- `inference-flame/src/models/ltx2_model.rs` — Split RoPE, fused ops, skip-4096 path, rescale
- `inference-flame/src/bin/ltx2_generate.rs` — CFG, dev schedule, resolution
- `inference-flame/src/sampling/ltx2_sampling.rs` — FnMut for CFG closure
