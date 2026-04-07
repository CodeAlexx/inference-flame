# flame-core `Tensor::permute` Fallback Bug — Fixed

**Date:** 2026-04-06
**Symptom:** Any `Tensor::permute` call with a dim order outside the four
hand-written hot paths silently ran a GPU → CPU scalar Rust loop → CPU → GPU
roundtrip and downcast BF16 to F32 along the way. Multi-second per call on
large tensors.
**Impact:** All of the 50–2000× slowdowns we found this session — FLUX 1
`split_qkv`, FLUX 1 6D RoPE narrow, Z-Image / SDXL / SD3 / LTX-2 / Hunyuan
VAE decoders — traced back to this one wire-up.
**Fix:** three lines in `flame-core/src/tensor.rs`. No new CUDA code.

## The investigation (abridged)

Earlier in the session we found and worked around three instances of what
looked like separate bugs:

1. **FLUX 1 RoPE** — the verbatim BFL port used a 5-D `narrow` that fell
   through to a CPU path. Fixed by replacing with
   `flame_core::bf16_ops::rope_fused_bf16` (150× speedup). Documented in
   `AUDIT_FLUX1.md`.
2. **FLUX 1 `split_qkv`** — `reshape(b,n,3,h,d).permute([2,0,3,1,4])`. The
   5-D permute hit the CPU fallback. Fixed by rewriting as a 3-D narrow
   + the existing `[0,2,1,3]` hot path (2333× speedup). Also in
   `AUDIT_FLUX1.md`.
3. **Z-Image / LDM VAE** — the NHWC↔NCHW layout shuffles needed by
   `group_norm_nchw` were `permute([0,2,3,1])` and `permute([0,3,1,2])`,
   neither of which is a hot path. Fixed by reshaping to 3-D and using
   `permute([0,2,1])` (68× VAE speedup, 10.3× end-to-end). Documented in
   `PERF_VAE_PERMUTE.md`.

By (3) I had noticed these were the *same* bug surfacing three different
places: `Tensor::permute` in `flame-core/src/tensor.rs` had a special-case
switch with exactly four "hot path" branches and a `// General permutation
fallback (CPU copy for now)` arm that handled everything else. The comment
is accurate — it really was a CPU copy with a scalar Rust loop:

```rust
// tensor.rs:2594-2649 (before)
} else {
    // General permutation fallback (CPU copy for now)
    let src_data = self.to_vec()?;                 // GPU → CPU staging
    let mut dst_data = vec![0.0f32; total];
    for (dst_idx, dst_val) in dst_data.iter_mut().enumerate() {
        // 4-deep index decomposition, scalar, per element
        ...
    }
    let tmp = Tensor::from_vec_dtype(..., DType::F32)?;  // CPU → GPU, F32
    if self.dtype() == DType::F32 { tmp } else { tmp.to_dtype(self.dtype())? }
}
```

The `to_vec()` → `from_vec_dtype(..., DType::F32)` round-trip explains the
silent BF16→F32 downcast. The `for` loop with per-iteration index
decomposition explains the ~3–5 seconds of CPU work for ~134 M elements.

And there was also a fake-hot-path `[0,1,3,2]` branch that called a private
`permute_0132` function, which was *itself* a scalar CPU loop with a
`from_vec` F32 return — same pattern, smaller scope. A comment at the call
site even said "Hot path for Flux attention" which was wrong.

## I almost wrote a redundant kernel

My first instinct was to write a new generic 4-D CUDA kernel, add FFI
declarations, add it to `build.rs`, and wire it through `cuda_ops.rs`. I
had about 100 lines of CUDA (`cuda/permute_nd.cu`) when I decided to grep
for existing permute code one more time and found this:

```
flame-core/src/cuda_kernels.rs:2362: pub fn permute_generic(&self, tensor: &Tensor, perm: &[usize]) -> Result<Tensor>
flame-core/src/cuda_kernel_sources.rs:658: extern "C" __global__ void permute_generic_f32_kernel(...)
flame-core/src/cuda_kernel_sources.rs:690: extern "C" __global__ void permute_generic_bf16_kernel(...)
```

**The generic GPU permute kernel already existed.** It was a JIT-loaded
runtime kernel (through cudarc), F32 and BF16 variants, rank up to 8,
BF16-preserving, correctly handling arbitrary dim orders via precomputed
src/dst strides. Sitting in the source tree, reachable via
`GpuOps::permute_generic`. Just never connected to `Tensor::permute`.

The bug wasn't "missing kernel". The bug was "kernel never wired up".

I deleted the new `permute_nd.cu`, undid the FFI and build.rs edits, and
the actual fix was a three-branch `else` in `Tensor::permute`:

```rust
// tensor.rs (after)
let mut output = if shape.len() == 2 && dims == [1, 0] {
    self.transpose()?
} else if shape.len() == 3 && dims == [0, 2, 1] {
    GpuOps::permute_021(self)?
} else if shape.len() == 4 && dims == [0, 2, 1, 3] {
    GpuOps::permute_0213(self)?
} else {
    // General N-D GPU permute (rank <= 8, BF16-preserving).
    GpuOps::permute_generic(self, dims)?
};
```

Plus deleting the private `permute_0132` (nothing called it once the
`[0,1,3,2]` branch was removed from the switch).

## Verification

### Z-Image 1024² 2-step re-run, naive 4-D permute in VAE

I reverted the VAE's 3-D reshape workaround (the one from
`PERF_VAE_PERMUTE.md`) so `to_nhwc`/`to_nchw` call the natural
`permute([0,2,3,1])` / `permute([0,3,1,2])` directly, and re-ran with the
flame-core fix in place:

```
                    CPU fallback   3-D workaround   4-D + flame-core fix
0.norm                  0 ms          1 ms             0 ms
1.conv_in              75 ms         74 ms            81 ms
2.mid_block          5362 ms       1252 ms           415 ms    ← bonus
up_block[0]          4993 ms         37 ms            35 ms
up_block[1]         20701 ms        182 ms           192 ms
up_block[2]         47684 ms        406 ms           405 ms
up_block[3]         95658 ms        689 ms           650 ms
4.norm_out          13509 ms         68 ms            59 ms
5.silu                  7 ms          7 ms             7 ms
6.conv_out             43 ms         40 ms            48 ms
─────────────────────────────────────────────────────────────────
VAE TOTAL          188037 ms       2760 ms          1898 ms
Total pipeline      362.2 s         35.3 s           31.1 s
```

The naive 4-D path is **faster than the 3-D reshape workaround** (1898 ms
vs 2760 ms) because the single GPU permute avoids the reshape-and-copy
sequence the workaround needed.

The mid_block dropped from 1252 ms to **415 ms** — that was not expected.
I hadn't touched the mid block's attention code; the fact that it sped up
means there was *another* natural-order permute inside the AttnBlock that
I had never noticed, quietly hitting the same CPU fallback, and the
flame-core fix swept it up for free.

### Numerical correctness

Byte-identical PNG output compared to the 3-D workaround run:

```
f637b34a9226b027a80912c808beaf09  output/zimage_vaefix.png    (3-D workaround)
f637b34a9226b027a80912c808beaf09  output/zimage_naive.png     (4-D + flame-core fix)
```

Same MD5. The flame-core fix produces bit-exact results — no BF16→F32
precision loss, no numerical drift.

### Training path not touched

The `requires_grad` / `AutogradContext::record_op` block at
`tensor.rs:2652-2666` is still in `Tensor::permute` and is unchanged. The
fix only replaces the *output computation* in the else arm, not the
gradient tracking. Autograd v3 recording still happens identically.

## What this unlocks automatically

Any model that does a natural-order permute now gets the GPU path for
free, with no per-model change required:

- **All LDM-style VAE decoders** (Z-Image, SDXL, SD3 Medium, FLUX 1,
  LTX-2, Hunyuan, Kandinsky) — they all use the same NCHW↔NHWC
  `group_norm_nchw` helper pattern.
- **FLUX 1 `split_qkv`** — the 5-D `[2,0,3,1,4]` permute that we
  worked around in `flux1_dit.rs::split_qkv` with a 3-D narrow + 4-D
  hot path decomposition. That workaround is no longer needed — the
  naive 5-D permute now hits the GPU generic kernel.
- **Anything else doing higher-rank permutes** — any 5-D or 6-D
  reshape dance in LTX-2 video or Hunyuan Video, which I haven't
  audited but which likely had the same footgun.

I'm leaving the existing workarounds in place for now (they're correct
and tested); they can be simplified in a follow-up sweep.

## Files changed

- `flame-core/src/tensor.rs` — `Tensor::permute` fallback now calls
  `GpuOps::permute_generic` instead of the scalar CPU loop. Removed
  the `[0,1,3,2]` false-hot-path branch. Deleted the private
  `permute_0132` function (no callers).
- `inference-flame/src/vae/ldm_decoder.rs` — reverted `to_nhwc`/
  `to_nchw` to use natural 4-D permutes, since flame-core now handles
  them correctly. Kept the diagnostic comments pointing at
  `PERF_VAE_PERMUTE.md` for context.

**Zero new CUDA code.** The kernel was already there.

## Lesson

Before writing a new kernel: `grep -rn "kernel_name\|generic_kernel"`
across the entire crate. Something might already exist but not be
connected. The bug I spent all session chasing through three different
models was one dead else-branch in a single switch statement in
`tensor.rs`.
