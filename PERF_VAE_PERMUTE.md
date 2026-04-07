# VAE Decode Slowness — Permute CPU Fallback Bug

**Date:** 2026-04-06
**Symptom:** Z-Image 1024² end-to-end took 6 min; the VAE decode alone was
198.6 s while the transformer denoise (8 steps) was 73.8 s.
**Affected:** `inference-flame/src/vae/ldm_decoder.rs`, and by proxy every
caller of `LdmVAEDecoder` (Z-Image, SD3 Medium, FLUX 1, SDXL int8, SD3.5,
Kandinsky, and anything else routing through this decoder).
**Root cause:** a systemic flame-core issue, not a VAE bug.

## Summary

Three flame-core `Tensor::permute` cases that look like hot paths are
actually **scalar Rust CPU loops that also silently downcast BF16 to F32**:

- `[0, 2, 3, 1]` (NCHW → NHWC) — falls through to the general 4D
  `to_vec()` + 4-deep Rust loop + `from_vec` path in
  `flame-core/src/tensor.rs:2594-2649`.
- `[0, 3, 1, 2]` (NHWC → NCHW) — same general fallback.
- `[0, 1, 3, 2]` — has its *own* function `permute_0132` at
  `flame-core/src/tensor.rs:2726`, but that function is also a
  `to_vec()` + scalar 4-deep loop (line 2735-2748) and returns via
  `Tensor::from_vec` which produces an F32 tensor.

`LdmVAEDecoder::decode` is the biggest victim: its `group_norm_nchw`
helper calls `permute([0,2,3,1])` before every group norm and
`permute([0,3,1,2])` after. With ~25 group norms in the VAE, many of
them at `[1, 128, 1024, 1024]`, that's 50+ calls to the CPU fallback
on 134M-element BF16 tensors.

## Reproduction

Baseline run, Z-Image turbo 1024² 8 steps, no fixes:

```
cd /home/alex/EriDiffusion/inference-flame
./target/release/zimage_infer \
  --model   /home/alex/.serenity/models/checkpoints/z_image_turbo_bf16.safetensors \
  --vae     /home/alex/.serenity/models/zimage_base/vae/diffusion_pytorch_model.safetensors \
  --embeddings /home/alex/serenity/output/zimage_embeddings.safetensors \
  --output  output/zimage_baseline.png \
  --steps 8 --height 1024 --width 1024 --seed 42
```

Result: **362.2 s total** (73.8 s denoise + 198.6 s VAE + overhead).

## Investigation — per-stage VAE profile

Added sync timing around each section of `LdmVAEDecoder::decode` behind
`VAE_PROF=1` and re-ran with `--steps 2`:

```
[VAE] 0.norm              0 ms
[VAE] 1.conv_in           75 ms   [1,512,128,128]
[VAE] 2.mid_block       5362 ms   [1,512,128,128]   (HUGE alloc warnings fire here)
[VAE] up_block[0]       4993 ms   [1,512,256,256]
[VAE] up_block[1]      20701 ms   [1,512,512,512]
[VAE] up_block[2]      47684 ms   [1,256,1024,1024]
[VAE] up_block[3]      95658 ms   [1,128,1024,1024]
[VAE] 4.norm_out       13509 ms   single GroupNorm on [1,128,1024,1024]
[VAE] 5.silu               7 ms
[VAE] 6.conv_out          43 ms   [1,3,1024,1024]
[VAE] TOTAL           188037 ms   ≈ 188 s
```

**Two things stood out immediately:**

1. `norm_out`, which is a single GroupNorm on 134M elements, took
   **13.5 seconds**. Bandwidth limit says a well-implemented group
   norm on that tensor is ~0.5 ms. We were ~27,000× off.
2. The up blocks scale much worse than resolution. up_block[3] at
   `[1,128,1024,1024]` is 96 s; up_block[1] at `[1,512,512,512]` is
   21 s. The element count ratio is 1.7×; the time ratio is 4.6×.
   That rules out "the actual math got bigger" and points at
   per-element overhead with a kernel-launch or host-roundtrip term.

Both clues pointed at a helper that's not running on the GPU.

## Root cause

`LdmVAEDecoder` uses this helper to run `group_norm_bf16` (which
expects NHWC) on NCHW tensors:

```rust
fn to_nhwc(x: &Tensor) -> Result<Tensor> { x.permute(&[0, 2, 3, 1]) }
fn to_nchw(x: &Tensor) -> Result<Tensor> { x.permute(&[0, 3, 1, 2]) }

fn group_norm_nchw(x, ...) -> Result<Tensor> {
    let nhwc = to_nhwc(x)?;
    let out_nhwc = group_norm(&nhwc, ...)?;
    to_nchw(&out_nhwc)
}
```

`Tensor::permute` in `flame-core/src/tensor.rs:2580-2650` dispatches:

```rust
let mut output = if shape.len() == 2 && dims == [1, 0] {
    self.transpose()?
} else if shape.len() == 3 && dims == [0, 2, 1] {
    GpuOps::permute_021(self)?                    // real GPU kernel
} else if shape.len() == 4 && dims == [0, 2, 1, 3] {
    GpuOps::permute_0213(self)?                   // real GPU kernel
} else if shape.len() == 4 && dims == [0, 1, 3, 2] {
    self.permute_0132()?                          // FALSE HOT PATH — CPU loop
} else {
    // General permutation fallback (CPU copy for now)
    let src_data = self.to_vec()?;                // GPU → CPU of the whole tensor
    let mut dst_data = vec![0.0f32; total];
    for (dst_idx, dst_val) in dst_data.iter_mut().enumerate() {
        // 4-deep index decomposition, scalar, Rust
        ...
    }
    let tmp = Tensor::from_vec_dtype(..., DType::F32)?;  // F32 output
    if self.dtype() == DType::F32 { tmp } else { tmp.to_dtype(self.dtype())? }
};
```

`NCHW → NHWC` needs `[0,2,3,1]` which is neither in the switch nor a
composition of the others, so it hits the general fallback. For a
`[1, 128, 1024, 1024]` tensor that means:

1. Allocate 268 MB staging on device.
2. `to_vec()` — upcast BF16 → F32 (134M u16 → 134M f32, 512 MB write)
   and copy the F32 down to host memory over PCIe (~268 MB via
   PCIe 4 x16 @ ~30 GB/s → ~9 ms minimum, more with kernel launch
   overhead).
3. Scalar Rust loop, 134M iterations, each one running
   `for (axis, stride) in dst_strides.iter().enumerate()` to
   decompose the destination index and reconstruct the source
   index. A Rust scalar loop with per-element Vec allocations and
   division ops clocks around 30M–50M it/s. 134M iters / 40 Mit/s
   = ~3 s.
4. `from_vec` creates an F32 tensor on host and copies back to the
   device. Another ~500 MB write on the device.
5. `to_dtype(BF16)` on 134M elements. Another full pass over device
   memory, plus the aligned-F32 staging buffer alloc (this is where
   the `CUDA alloc_aligned: HUGE allocation requested = 268435456`
   warnings come from — one per `to_dtype`).
6. Same thing again on the way back (`to_nchw`).

Measured cost: **~8 seconds per call** at this size.

`LdmVAEDecoder::decode` triggers this helper inside every `ResBlock`
(2 GroupNorms/block), every `AttnBlock` (1), and the final `norm_out`.
Across the four up blocks plus mid plus final norm, the highest-res
stages accumulate the 96-second up_block[3] and 48-second up_block[2]
figures from the profile.

**None of that is about the VAE architecture.** The decoder structure
is correct. The flame-core kernel `group_norm_bf16` is a legitimate
fused CUDA kernel. The only actual GPU work in the 188 seconds is the
conv2d calls (75 ms + 40 ms + the inner ResBlock convs) and the
`group_norm_bf16` calls themselves (~100 ms total estimated). The
other ~187 seconds is layout conversion thrash.

## Fix

The only 4D permute primitive in flame-core that's actually on the GPU
is `permute_0213`. The only general one is `permute_021` on 3D tensors
(`GpuOps::permute_021` — a real CUDA kernel that preserves BF16
storage). Both NCHW↔NHWC cases can be expressed by reshaping to 3D
first:

```
NCHW [N,C,H,W] → reshape [N, C, H*W] (view)
               → permute [0,2,1]       (GPU kernel, BF16-preserving)
               → reshape [N, H, W, C]  (view)      = NHWC

NHWC [N,H,W,C] → reshape [N, H*W, C] (view)
               → permute [0,2,1]       (same kernel)
               → reshape [N, C, H, W] (view)       = NCHW
```

Both reshapes are free — inputs are contiguous in the expected layout
going in, and `permute_021` writes a fresh contiguous output going
out. The whole thing is one CUDA kernel launch per permute.

Applied as a local patch to `inference-flame/src/vae/ldm_decoder.rs`:

```rust
fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let x3 = x.reshape(&[n, c, h * w])?;
    let x3 = x3.permute(&[0, 2, 1])?;
    x3.reshape(&[n, h, w, c])
}

fn to_nchw(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (n, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
    let x3 = x.reshape(&[n, h * w, c])?;
    let x3 = x3.permute(&[0, 2, 1])?;
    x3.reshape(&[n, c, h, w])
}
```

Two false starts before this landed, both instructive:

- **First attempt:** chain `[0,2,1,3]` and `[0,1,3,2]` to reach
  `[0,2,3,1]`. Compiled fine, runtime failed with
  `group_norm::in: expected logical/storage BF16, got logical=F32
  storage=F32`. Root cause: `permute_0132` (the `[0,1,3,2]`
  "hot path") is actually a CPU loop that returns F32.
- **Hypothesis-before-profile:** I guessed the attention block was
  the culprit (16384² scores matrix at head_dim=512 can't hit flash).
  The profile showed mid_block was only 5.3 s of the 188 s — the
  attention is real work but not the bottleneck. It's in the
  permute path.

## Results

Same binary, same seed, same 1024² 2-step run:

```
                  BEFORE      AFTER     SPEEDUP
0.norm                0 ms     1 ms
1.conv_in            75 ms    74 ms
2.mid_block        5362 ms  1252 ms      4.3×
up_block[0]        4993 ms    37 ms    135×
up_block[1]       20701 ms   182 ms    114×
up_block[2]       47684 ms   406 ms    117×
up_block[3]       95658 ms   689 ms    139×
4.norm_out        13509 ms    68 ms    199×
5.silu                7 ms     7 ms
6.conv_out           43 ms    40 ms
─────────────────────────────────────────────
VAE TOTAL        188037 ms  2760 ms     68×
Total pipeline    362.2 s    35.3 s    10.3×  (includes 73.8 s denoise)
```

PNG: `output/zimage_vaefix.png`, 1024×1024 RGB, 1.19 MB, visually
checked.

The `mid_block` speedup is smaller (4.3×) because it's dominated by
the genuine 16384² attention scores materialization, which is
real GPU work regardless of the permute fix. The `HUGE alloc`
warnings still fire (11 of them, 1 GB each) from inside that
attention — those are legitimate scores-tensor allocs, not leaks.

## Systemic impact

Every caller of `LdmVAEDecoder` inherits the fix automatically
through the local patch: Z-Image, SD3 Medium, FLUX 1, SDXL,
Kandinsky, and any future model routed through the same decoder.

But this is the **same family of bug** we found earlier in FLUX 1's
`split_qkv` (a 5D `[2,0,3,1,4]` permute → general CPU fallback →
~2.3 s per block). Both bugs hit any code that does a "natural"
layout-swap permute that isn't in the four-case switch in
`Tensor::permute`. The real fix is in flame-core, not in the
model files.

**Recommended flame-core work (separate from this patch):**

1. Fix the lying `permute_0132`. Either replace the scalar CPU loop
   with a real GPU kernel, or remove it and let `[0,1,3,2]` fall
   through to the same fast general permute — once that general
   path exists.
2. Add a generic 4D BF16/F32 permute GPU kernel (`launch_permute_4d`
   parameterized by source strides) and route the general fallback
   through it instead of `to_vec()`. One kernel covers every dim
   order and every rank up to 4 (which is all anyone actually
   uses in diffusion models). Output dtype matches input dtype.
3. Delete the per-element Rust loop path entirely. Anything that
   makes it into the fallback today is "model is running 1000× too
   slow and nobody knows why".

These three changes together would auto-fix:
- The VAE (this doc), without the local patch.
- FLUX 1 `split_qkv` (currently worked around via a 3D-narrow +
  `[0,2,1,3]` hot path manual decomposition in `flux1_dit.rs`).
- Any future model that does a natural layout permute and
  silently tanks.

Until that lands, the rule is: **never call `Tensor::permute` on a
dim order that isn't one of `[1,0]`, `[0,2,1]`, `[0,2,1,3]`, or
`[0,1,3,2]` — and `[0,1,3,2]` also downcasts BF16 to F32, so
treat it as broken too.** Everything else will silently run a
multi-second CPU roundtrip.

## Files touched

- `inference-flame/src/vae/ldm_decoder.rs` — rewrote `to_nhwc` and
  `to_nchw` to use 3D reshape + `permute([0,2,1])`. Added a header
  comment explaining the flame-core fallback issue so the next
  person doesn't "clean it up" back to the naive permute.
- (No flame-core changes. The profiling scaffolding behind
  `VAE_PROF=1` is still in `decode()` and should probably be
  removed or gated once we're done measuring.)

## Files NOT touched (but should be, in a follow-up)

- `flame-core/src/tensor.rs:2580-2650` — the general permutation
  fallback. The `eprintln`-free, scalar, F32-downcasting CPU loop.
- `flame-core/src/tensor.rs:2726-2762` — `permute_0132`. Same
  pattern, smaller scope.
- Any other model that calls a 4D layout-swap permute directly.
  `flux1_dit.rs::split_qkv` was already rewritten; a grep for
  `permute(&[` across `inference-flame/src/models/` would find any
  others that are silently slow.

## How to reproduce the investigation

```bash
# 1. Build the binary with VAE_PROF sync scaffolding already in place.
cargo build --release --bin zimage_infer

# 2. Baseline (only if you want to see the 188 s VAE one more time).
VAE_PROF=1 ./target/release/zimage_infer --model ... --vae ... \
    --embeddings ... --steps 2 --height 1024 --width 1024

# 3. Apply the to_nhwc/to_nchw rewrite (this patch) and rerun.
# Expected: VAE total under 3 seconds, up_block[3] under 1 second.
```
