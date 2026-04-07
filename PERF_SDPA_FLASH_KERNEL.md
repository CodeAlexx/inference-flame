# flame-core Flash Attention Kernel Has No Tensor Cores

**Date:** 2026-04-07
**Symptom:** Z-Image NextDiT at 1024² runs at 9.1 s/step vs PyTorch reference
~1 s/step on the same hardware. **9× gap to PyTorch.** T5-XXL encode takes
37 s (24 layers × ~1.5 s each) for the same class of reason.
**Root cause:** `flame-core/src/cuda/flash_attention_fwd.cu` is a scalar FP32
dot-product kernel with tile size 32×32. No `wmma`, no `mma.sync`, no tensor
core usage whatsoever. Effective throughput is 1 TFLOPS FP32 FFMA instead of
300+ TFLOPS BF16 tensor core on an Ampere card — ~50× off peak.
**Scope:** every attention call in every model routed through
`flame_core::attention::sdpa` when head_dim ∈ {64, 96, 128} and mask is
`None`. That's Z-Image (30 layers), FLUX 1 DiT (57 blocks), Klein (38+ single
blocks), T5-XXL text encoder (24 layers), CLIP, LTX, Hunyuan.

## Measurement — Z-Image block forward profile

Z-Image 1024² latent (seq=4096, num_heads=30, head_dim=128), added sync
timers per section of `transformer_block` and `joint_attention` behind
`ZIMAGE_BLOCK_PROF=1`. Steady-state (post-warmup) numbers for block 0 over
multiple denoise steps:

```
Block total                        280 ms
   1.adaln_mod                       0 ms
   2.rms1+scale_msa                  4 ms
   3.joint_attention               250 ms   ← 90% of block
       a.qkv_proj+chunk                5 ms
       b.qk_rmsnorm                    0 ms
       c.permute_qkv (hot path)        0 ms
       d.rope (fused kernel)           0 ms
       e.sdpa                        230 ms   ← 82% of block
       f.permute_out                   0 ms
       g.out_proj                      1 ms
   4.rms2+gate1                      1 ms
   5.rms3+scale_mlp                  4 ms
   6.swiglu                         17 ms
   7.rms4+gate2                      1 ms
```

30 layers × 280 ms = 8.4 s/step — matches the observed 9.1 s/step within
warmup noise.

**SDPA is 230 ms per call.** Theoretical lower bound for this shape on an
RTX 3090 Ti:

- `BH=30, Q=K=4096, d=128`
- Two BMMs: `2 × 30 × 4096 × 4096 × 128 = 128 GFLOP`
- Peak BF16 tensor core: ~160 TFLOPS on 3090 Ti
- Theoretical: **~0.8 ms** (memory-bound) to **~2 ms** (50% tensor core
  utilisation)
- Observed: **230 ms**
- **Gap: ~100×**

## Root cause — the kernel

`flame-core/src/cuda/flash_attention_fwd.cu` defines three specialisations
for `head_dim ∈ {64, 96, 128}` via a macro, all with identical structure:

```cuda
#define BQ 32
#define BKV 32
#define THREADS 128

#define DEFINE_FLASH_ATTN_KERNEL(HD)                                          \
__global__ void flash_attn_fwd_hd##HD(...) {                                  \
    // ... shared-mem tile load in FP32 ...                                   \
    for (int idx = tid; idx < q_rows * kv_rows; idx += THREADS) {             \
        const int qi = idx / kv_rows;                                         \
        const int kj = idx % kv_rows;                                         \
        float dot = 0.0f;                                                     \
        const float* q_row = s_Q + qi * HD;                                   \
        const float* k_row = s_K + kj * HD;                                   \
        _Pragma("unroll 16")                                                  \
        for (int d = 0; d < HD; d++) {                                        \
            dot += q_row[d] * k_row[d];   // <-- SCALAR FP32 DOT PRODUCT       \
        }                                                                     \
        s_S[qi * BKV + kj] = dot * scale;                                     \
    }                                                                         \
    // ... online softmax and PV matmul in the same scalar style ...          \
}
```

Three problems compound:

1. **No tensor cores.** The inner loop is a plain FP32 FFMA sequence. On
   SM_80+ you get ~19 TFLOPS FP32 per SM vs ~312 TFLOPS BF16 tensor core.
   The kernel runs at roughly 1 TFLOPS effective throughput on big shapes
   because thread occupancy is limited.
2. **Tile sizes are too small.** `BQ=32, BKV=32` with `THREADS=128`. A
   real flash attention tile is 64×64 or 128×64 with each warp responsible
   for one output tile row via MMA. The 32×32 tile is what a first draft
   looks like before the MMA rewrite.
3. **All FP32 in shared memory.** Each Q/K/V byte is upcast to 4 bytes
   when loaded into shared. Halves the tiles that fit simultaneously,
   kills occupancy.

The comment at the top of the file says *"Supports head_dim = 64, 96, 128
via compile-time specialization. SD3 uses 64, Mistral 96, FLUX/LTX/Klein
128."* — the specialisation is just loop unrolling, nothing algorithmic.

## Confirming the diagnosis

### Check 1: disable flash, see what happens

Set `FLAME_NO_FLASH_ATTN=1`. Z-Image's `sdpa_forward` now skips flash and
hits `forward_bf16_fallback`, which is a cuBLASLt-based path: two batched
BF16 GEMMs + FP32 softmax staging. That path **does** use tensor cores
(via cuBLASLt). Profile block 0 again:

```
e.sdpa   87 ms    (was 230 ms)
```

2.6× faster on the same shape. The cuBLASLt BMMs are running on tensor
cores and finishing in a few milliseconds; the overhead is the FP32
softmax staging and the BF16↔F32 round trips.

### Check 2: the fallback OOMs on Z-Image

The fallback allocates `logits_bf16 [BH*Q*K]` and its F32 upcast for
softmax. For Z-Image `[BH=30, Q=4096, K=4096]`:

- `logits_bf16`: 30 × 4096 × 4096 × 2 bytes = **1.00 GB**
- `logits_f32`: 30 × 4096 × 4096 × 4 bytes = **2.00 GB**

Combined with ~10 GB of resident Z-Image weights + activations, peak VRAM
goes over 14 GB and eventually OOMs on a 24 GB card by step 2:

```
CUDA alloc_aligned: HUGE allocation requested = 503316480 elements (1.875 GB)
CUDA allocation failed with size 503316480: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
Error: CudaDriver
```

So the "fast" path is unusable as-is. Either the kernel needs tensor cores
or the fallback needs Q-tiling.

## Why Klein and flux1 aren't as obviously broken

They hit the same kernel but absolute times look smaller:

- Klein 4B has ~38 blocks vs Z-Image 30 but smaller inner dim and a shorter
  effective sequence length per swap load — observed 158 s / 20 steps at
  1024² ≈ **7.9 s/step**, close to Z-Image's 9.1 s/step. Same problem,
  roughly the same total magnitude.
- FLUX 1 DiT at 1024² has 19+38=57 blocks with joint txt+img seq ≈ 4608.
  DiT profile earlier this session showed SDPA at 257 ms per double-block
  call — same ~230-250 ms regime. But FlameSwap load overhead (~120 ms per
  block) dominated my previous profile's absolute times, so the SDPA
  wasn't the visible hot spot.

Fixing the flash kernel affects **every** model in the tree.

## T5-XXL 37 s is the same bug

T5 encoder runs 24 layers × 1 attention call each. T5 uses
`sdpa_with_bias` → `forward_with_bias`, which is a manual FP32 path
(no flash eligibility because T5 passes an additive position bias). The
underlying BMMs should still be cuBLASLt-fast, but the path has its own
F32 staging + softmax + cast overhead. Measured ~1.5 s per layer × 24
layers = ~36 s. Not directly the flash bug — but the same class of
"unoptimised attention path dominates total runtime".

If we give the flash kernel real tensor cores, T5 won't benefit
automatically. Separate fix for T5 after this one lands.

## Permute grep — bonus check

Per the session directive, I grepped every `.permute(&[...])` call across
inference-flame for dim orders outside the four hot paths (`[1,0]`,
`[0,2,1]`, `[0,2,1,3]`, and the just-wired generic fallback). Two hits:

- `zimage_nextdit.rs:394` — `permute([0, 2, 4, 3, 5, 1])` in `patchify`
- `zimage_nextdit.rs:404` — `permute([0, 5, 1, 3, 2, 4])` in `unpatchify`

Both are 6D permutes in the outer patchify/unpatchify, called **once per
forward pass** (not per block). With the flame-core permute fix they
now route through `GpuOps::permute_generic` (the real GPU scatter
kernel) and are sub-millisecond. Not a bottleneck.

Every attention permute in every model is `[0,2,1,3]` — the specialised
hot path `GpuOps::permute_0213`. Clean.

## Fix options

### Option 1 — Rewrite flash kernel with `wmma` tensor cores

Replace the scalar FP32 inner loop with `nvcuda::wmma` fragments on
SM_80+. Canonical flash attention implementation:

- `BQ = BKV = 64` (or 128 for head_dim=64)
- Each warp computes one 16×16 output block via `wmma::mma_sync`
- BF16 fragments for Q, K, V; FP32 accumulator for softmax correction
- Online softmax unchanged (it's already correct)

Work estimate: a few hundred lines of CUDA. `wmma` has sharp edges
(fragment layouts, warp-level sync) but flame-core has no other tensor
core kernel to borrow from so this has to be written from scratch.

Expected result: ~3-5 ms per call on Z-Image shape → **~50× speedup**
on SDPA, block forward drops from 280 ms to ~45 ms, Z-Image denoise
drops from ~9 s/step to **~1.3 s/step**. Within noise of the PyTorch
reference.

### Option 2 — Q-tile the cuBLASLt fallback

Apply the same tiling pattern I used for the VAE mid-block attention:
process Q in chunks, each chunk runs its own BMM → softmax → BMM
pipeline, concatenate outputs. All three sub-steps already use tensor
cores via cuBLASLt; the only reason the fallback was slow on Z-Image is
that it was materializing a 2 GB logits tensor and OOMing.

Work estimate: ~50 lines in `flame-core/src/sdpa.rs`
`forward_bf16_fallback`. Same math, lower peak memory.

Expected result: ~15-25 ms per call on Z-Image shape → **~10× speedup**
on SDPA, block forward drops from 280 ms to ~60 ms, Z-Image denoise
drops from ~9 s/step to **~2 s/step**. Still above PyTorch but a huge
win and very low risk.

### Option 3 — cuDNN flash attention via graph API

cuDNN 8.9+ ships `cudnnGraphAPI` with a multi-head attention node that
internally dispatches a production flash attention kernel. flame-core
already links cuDNN for Conv2d so the runtime is present.

Work estimate: depends on whether flame-core has any cuDNN graph
scaffolding (likely not). Writing it from scratch is ~300 lines of
boilerplate.

Expected result: matches or beats Option 1 since cuDNN's kernel is
professionally tuned. But highest up-front boilerplate cost and
cuDNN-version dependent.

## Files touched by the profile work

- `inference-flame/src/models/zimage_nextdit.rs` — added section-level
  sync profiling behind `ZIMAGE_BLOCK_PROF=1` in `transformer_block`
  and sub-section profiling in `joint_attention`. Only fires on layer
  0 so noise is small. Should be stripped or kept as a permanent
  diagnostic switch in the same spirit as the `FLAME_AUTOGRAD_OFF`
  switch in `zimage_infer.rs`.

## Files NOT touched (but should be, in the fix)

- `flame-core/src/cuda/flash_attention_fwd.cu` — the naive scalar
  kernel. Replace per Option 1.
- `flame-core/src/sdpa.rs` `forward_bf16_fallback` — the materialized
  cuBLASLt path. Q-tile per Option 2.
- `flame-core/src/sdpa.rs` `forward_with_bias` — same class of fix
  for T5 (separately, after the main flash kernel lands).

## Recommendation

**Land Option 2 first** (Q-tile the fallback). It's an afternoon of work,
eliminates the OOM, and gives us a ~10× speedup across every model.
Gets Z-Image to ~2 s/step, FLUX 1 DiT block to ~60 ms, T5 encode to
~5 s.

**Then do Option 1** (wmma flash kernel) to close the remaining 3× gap.
That's the proper long-term answer and the last 3× gap to PyTorch
peak. A day of careful CUDA work.

Option 3 is the fallback for Option 1 if `wmma` turns out to be too
sharp-edged.

## Numbers summary

| Stage | Current | Option 2 (est) | Option 1 (est) |
|---|---|---|---|
| Z-Image SDPA per call | 230 ms | 20 ms | 3 ms |
| Z-Image block total | 280 ms | 70 ms | 45 ms |
| Z-Image per step (30 layers) | 9.1 s | ~2.1 s | ~1.4 s |
| Z-Image 1024² 8 steps total | 84 s | ~19 s | ~13 s |
| FLUX 1 DiT block (57 blocks) | ~440 ms | ~200 ms* | ~130 ms* |
| FLUX 1 1024² 4 steps total | 144 s | ~75 s* | ~50 s* |
| T5-XXL encode | 37 s | ~12 s | ~5 s |

`*` FLUX 1 numbers include FlameSwap block-load overhead which is
fixed-cost per block regardless of attention speed. Real FLUX 1 gains
will be slightly smaller than the shape ratios suggest.

## Reproduction

```bash
# Build with profiling scaffolding (always present, gated on env var).
cargo build --release --bin zimage_infer

# Run with block profile. Fires on layer 0 only.
ZIMAGE_BLOCK_PROF=1 ./target/release/zimage_infer \
    --model /home/alex/.serenity/models/checkpoints/z_image_turbo_bf16.safetensors \
    --vae /home/alex/.serenity/models/zimage_base/vae/diffusion_pytorch_model.safetensors \
    --embeddings /home/alex/serenity/output/zimage_embeddings.safetensors \
    --output output/z_prof.png \
    --steps 2 --seed 42 2>&1 | grep -E "ZBPROF|ZATTN"

# Confirm the fallback path is faster (and OOMs on step 2).
FLAME_NO_FLASH_ATTN=1 ./target/release/zimage_infer [same args]
```

## Commits to land for this fix session

This doc, plus the Z-Image profiling scaffolding in
`zimage_nextdit.rs::transformer_block` and `::joint_attention`. The
actual kernel/fallback fix is a separate commit per Option 1 or 2.
