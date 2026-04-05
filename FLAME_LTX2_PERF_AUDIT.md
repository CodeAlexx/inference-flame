# FLAME LTX-2 Performance Audit

**Date**: 2026-04-05
**Model**: LTX-2.3 22B (48 blocks, 768 tokens at 512x512x17)
**Hardware**: RTX 3090 Ti 24GB
**Observed**: ~5-7 min/step, 8 steps = ~40-50 min total
**Expected**: PyTorch does same in ~2-3 min total (~15-20s/step)

## Root Cause: Death by Kernel Launches

~75 GPU kernel launches per block. 48 blocks x 8 steps = ~28,800 launches per generation. With small tensors (768 tokens x 4096 dim), most kernels finish in microseconds but launch overhead (~5-10us each) dominates.

## Matmul: cuBLAS — NOT the bottleneck

BF16 path uses `cuda_ops_bf16::gemm_bf16` via cublasLt with FP32 accumulation. TF32 tensor cores enabled for F32 fallback. This is fast.

**File**: `flame-core/src/ops/gemm.rs`

## Attention: Materialized GEMM, no Flash Attention

`sdpa.rs:232` — For LTX-2 (32 heads x 768 tokens), attn_elems = 18M, well under the 768M threshold. Uses `forward_bf16_fallback`: two batched cuBLASLt GEMMs + FP32 softmax.

Correct and uses tensor cores, but ~3x slower than FlashAttention-2. For 768 tokens the attention matrix is only 36MB so memory isn't the issue — it's the kernel count (bmm + cast + softmax + cast + bmm = 5+ kernels vs FlashAttention's 1).

**File**: `flame-core/src/sdpa.rs:209-286`

## Bottleneck 1: `linear3d` reshapes per call

```rust
fn linear3d(x, weight, bias):
    x_2d = x.reshape([B*N, C])       // GPU kernel
    out = matmul_weight_t(x_2d, w)   // cuBLAS GEMM
    result = out.reshape([B, N, D])   // GPU kernel
    result = result + bias            // GPU kernel
```

4 kernel launches per linear. One block has ~10 linear calls (Q, K, V, out x2 attentions + 2 FFN linears + gate_logits). That's ~40 kernel launches just for linears.

**Fix**: Use cublasLt strided batched GEMM directly on 3D tensors. Or fuse bias into the GEMM (cublasLt supports epilogue bias add).

**File**: `inference-flame/src/models/ltx2_model.rs:158-180`

## Bottleneck 2: RMS Norm — 6 kernels instead of 1

```rust
fn rms_norm(x, weight, eps):
    x_f32 = x.to_dtype(F32)          // cast kernel
    x_sq = x_f32 * x_f32             // elementwise kernel
    mean_sq = mean(x_sq)             // reduction kernel
    rsqrt = (mean_sq + eps).rsqrt()  // elementwise kernel
    normed = x_f32 * rsqrt -> BF16   // elementwise + cast kernel
    normed * weight                   // elementwise kernel
```

6 kernel launches per RMS norm. 3 norms per block = 18 launches.

PyTorch's `torch.nn.functional.rms_norm` is a single fused CUDA kernel.

**Fix**: Write a fused RMS norm CUDA kernel. Input BF16, output BF16, weight multiply included. One kernel launch instead of 6. flame-core already has custom CUDA kernels (`src/cuda/`).

**File**: `inference-flame/src/models/ltx2_model.rs:209-232`

## Bottleneck 3: AdaLN modulation — 4 kernels instead of 1

```rust
// (1 + scale) * norm_x + shift
scale.add_scalar(1.0)               // elementwise kernel
    .to_dtype(BF16)                  // cast kernel (may be no-op)
norm.mul(&scaled)                    // elementwise kernel
    .add(&shift)                     // elementwise kernel
```

4 launches per modulation, 3 modulations per block = 12 launches.

**Fix**: Fused `modulate(x, shift, scale)` kernel: `x * (1 + scale) + shift` in one pass.

**File**: `inference-flame/src/models/ltx2_model.rs:686, 700, 740`

## Bottleneck 4: No kernel fusion at all

PyTorch with `torch.compile` fuses chains of elementwise ops into single kernels. FLAME launches every op as a separate kernel. Residual adds, gating multiplies, sigmoid — all individual launches.

## Per-Block Kernel Count Breakdown

| Operation | Launches | Count/Block | Total |
|-----------|----------|-------------|-------|
| linear3d (reshape+gemm+reshape+bias) | 4 | 10 | 40 |
| rms_norm (cast+sq+mean+rsqrt+mul+cast) | 6 | 3 | 18 |
| AdaLN modulate (add+cast+mul+add) | 4 | 3 | 12 |
| SDPA (bmm+softmax+bmm+reshape) | ~5 | 1 | 5 |
| Gating, residuals, misc | ~1 | ~10 | 10 |
| **Total per block** | | | **~75** |
| **48 blocks x 8 steps** | | | **~28,800** |

## Fix Priority

| Fix | Launches Saved/Block | Difficulty | Impact |
|-----|---------------------|------------|--------|
| 1. Fused RMS norm kernel | 15 (18→3) | Medium (CUDA kernel) | High |
| 2. Fused linear3d (no reshape) | 20 (40→20) | Medium (cublasLt API) | High |
| 3. Fused modulation kernel | 9 (12→3) | Easy (CUDA kernel) | Medium |
| 4. FlashAttention | 4 (5→1) | Hard (complex kernel) | Medium |
| 5. Fused bias in GEMM | 10 (separate bias adds) | Easy (cublasLt epilogue) | Medium |

Fixes 1-3 alone would cut kernel count from ~75 to ~30 per block — roughly 2.5x speedup from launch overhead reduction alone.

## What's NOT slow

- cuBLAS GEMM — already optimal
- FlameSwap block loading — overlapped with compute, not on critical path
- Memory allocation — CUDA mempool with infinite caching, allocations are near-free
- No spurious device syncs in the hot path
