# SKEPTIC review — Phase 1 GGUF loader

Scope: `src/gguf/{mod,reader,dequant,remap}.rs` + `tests/gguf.rs`.
Reference: `/home/alex/llama.cpp/ggml/src/ggml-quants.c` (canonical C) and
`/home/alex/llama.cpp/ggml/src/ggml-common.h` (struct layouts).

TL;DR: 0 P0, 3 P1, 4 P2. The dequant math for Q8_0 / Q4_K / Q5_K / Q6_K is a
faithful byte-for-byte port of llama.cpp's `dequantize_row_*`. Header parsing
is solid. The main concerns are (a) the Q*_K tests are self-consistency
tautologies rather than llama.cpp ground-truth, (b) dim reversal is
model-family-dependent and one of the only ways silently-wrong weights reach
the GPU, and (c) a handful of ergonomic edge cases.

---

## P0 — correctness bugs (wrong math = silently-wrong weights)

None found. Every quant kernel matches the reference implementation
line-for-line.

---

## P1 — likely bugs / format edge cases

### P1-1. Q*_K tests are self-consistency, not ground truth

`tests/gguf.rs::dequant_q4_k_known_values`, `dequant_q5_k_known_values`,
`dequant_q6_k_known_values` construct a superblock, then assert the output
equals what the formula-on-paper predicts. That's the same formula the code
implements. If the code inverted the nibble order, or flipped the scale/min
packing bit-trick, many of these tests would still pass because the test
expectations were derived from the same mental model.

Concrete weakness of each test:

- **Q4_K**: exercises one nonzero scale (`sc[0]=1`) and one qs byte value
  (`0x77`). Never exercises `j>=4` branch of `get_scale_min_k4`. Never
  exercises a nonzero `m` (mins). Never exercises dmin≠0.
- **Q5_K**: exercises one scale, the `u1` mask only (u2 also set but on a
  sub-block with sc=0, so the result is zero and doesn't validate u2).
  Never advances to the 2nd through 4th pair iterations where `u1,u2 <<= 2`
  and thus never validates the mask shift arithmetic for bits 2..7 of qh.
- **Q6_K**: ql=qh=0 throughout. Never exercises the 2-bit high-byte packing
  `((qh[l] >> shift) & 3) << 4`. Never exercises the second half of the
  superblock (the second `for _ in 0..2` iteration). Only the `scales[0]`
  slot is nonzero; never validates `sc_ofs+is+2`, `+4`, `+6`.

If the implementation had a bug isolated to those unexercised paths — e.g.
wrong shift in Q6_K's qh unpacking, wrong u1/u2 shift step in Q5_K, wrong
scale/min offset for `j>=4` in Q4_K — the tests would still pass.

**Fix path** (future work, not blocking): take any real GGUF file
(`city96/FLUX.1-dev-gguf`, a Q6_K one would exercise Q6_K; a Q4_K_M would
exercise Q4_K and Q5_K and Q6_K mixed), extract one block of each quant
type, dequant with a Python reference (`gguf-py/gguf/quants.py` or
`llama.cpp`'s Python binding), pickle the first 256 output floats, and
hard-code them as the expected values.

### P1-2. `default_rename` strips at most one prefix

`remap.rs::default_rename` strips exactly one of
`model.diffusion_model.` | `transformer.` | `first_stage_model.` per call,
in that order. The module comment acknowledges this is "only one prefix is
stripped per call — the conventions are mutually exclusive in practice."

"In practice" is doing load-bearing work there. If a GGUF ever ships with
`model.diffusion_model.transformer.foo` (not known today in FLUX/SD3/Chroma
GGUFs, but not impossible for some future diffusers-exported GGUF), the
current code would leave `transformer.foo` in the output, and every
downstream model loader that does `weights.get("foo")` would silently find
nothing.

**Mitigation**: after `strip_prefix("model.diffusion_model.")` succeeds,
also try stripping `transformer.` / `first_stage_model.` from the result.

Severity: low today, becomes P0 the moment a GGUF with nested prefixes
appears.

### P1-3. Dim reversal is trusted but not tested against a real file

`reader.rs` line 329 reverses GGUF dims to match safetensors's
`[out, in, ...]` convention:

```rust
let dims: Vec<usize> = dims_ggml.into_iter().rev().collect();
```

This is correct for linear weights (`ggml` `ne[0]=in, ne[1]=out` → reversed
→ `[out, in]`, matching PyTorch). It is also correct for 2D conv weights
(ggml `ne[0]=kW, ne[1]=kH, ne[2]=IC, ne[3]=OC` → reversed → `[OC, IC, kH, kW]`,
matching PyTorch NCHW).

But: **there is no test that validates the reversal against any real GGUF**.
The synthetic test uses zero tensors. The `load_real_file` test asserts
`tensors.len() > 0` but doesn't assert any shape.

A single transposed linear weight would load with the "right" shape (both
dims look valid), multiply without error, and produce garbage outputs that
are impossible to diagnose from a stack trace. This is the highest-risk
silent failure mode in the entire loader.

**Fix path** (future work): when `GGUF_TEST_FILE` is set to a real FLUX
checkpoint, pull one known-shape tensor (e.g.
`double_blocks.0.img_attn.qkv.weight`, expected `[9216, 3072]` for FLUX.1-dev)
and assert the loaded shape matches. The Python `gguf` library can produce
the expected shape for any given file, so one-time setup is cheap.

---

## P2 — minor

### P2-1. Q6_K scales slice uses `unsafe` reinterpret

`dequant.rs` line 402-405:
```rust
let scales: &[i8] = unsafe {
    std::slice::from_raw_parts(block[192..192 + 16].as_ptr() as *const i8, 16)
};
```

Functionally sound (u8 and i8 have identical layout), but gratuitous —
`block[192..208].iter().map(|&b| b as i8).collect::<Vec<_>>()` or
indexing and casting inline (`block[192 + sc_ofs + is] as i8`) avoids the
unsafe block at zero perf cost (this runs once per 256-element block,
not per element).

### P2-2. No handling for 0-element tensors

`GgufTensorInfo::n_elements()` returns `1` when `dims` is empty (scalar
convention), and `dims.iter().product(1u64, ...)` would also return 0 if
any dim is 0 (true empty tensor). The `dequantize_to_f32` Q4_K/Q5_K/Q6_K
paths all gate on `n % QK_K == 0`, which holds for n=0, and then
`chunks_exact(BLOCK)` yields zero iterations — so an empty Q-tensor returns
an empty Vec. The `Tensor::from_f32_to_bf16` call on an empty Vec may or
may not be well-defined depending on flame-core; not verified.

Unlikely to occur in real diffusion GGUFs (every weight has >0 elements),
but worth a unit test.

### P2-3. `byte_size` silently rounds down on invalid `n_elements`

`reader.rs::GgufQuantType::byte_size`:
```rust
Self::Q4_K => Some((n_elements / 256) * 144),
```

If a (malformed) file has a Q4_K tensor with `n_elements = 300`, this
returns `144` bytes (for one block, 256 elements), not an error. The
downstream dequant does `bail!("n_elements {n} not divisible by 256")`,
so the error surfaces eventually — but the bounds check in `mod.rs`
(`end > mmap.len()`) wouldn't catch the mismatch. Defense in depth would
be to return `None` / bail in `byte_size` when `n_elements % block_size
!= 0`.

### P2-4. `load_real_file` integration test doesn't validate shapes

As noted in P1-3: the only end-to-end test asserts `tensors.len() > 0`.
A single-tensor shape assertion against a known file would close the
biggest silent-failure hole in the loader.

---

## Per-quant-type math audit

Cross-referenced against
`/home/alex/llama.cpp/ggml/src/ggml-quants.c` at the following line numbers:

### Q8_0 — `dequantize_row_q8_0` (line 401)

Reference:
```c
const float d = GGML_FP16_TO_FP32(x[i].d);
for (int j = 0; j < qk; ++j) y[i*qk + j] = x[i].qs[j]*d;
```

Port (`dequant.rs::dequant_q8_0_block`):
```rust
let d = half::f16::from_bits(...).to_f32();
for i in 0..32 { out[i] = d * block[2+i] as i8 as f32; }
```

Match: ✓ byte-exact. `block[2+i] as i8 as f32` correctly treats the byte as
signed before widening. The f16 → f32 conversion uses the `half` crate.

Block layout check: `block_q8_0` = `{ ggml_half d; int8_t qs[32]; }` = 34 B.
Port reads `block[0..2]` as d, `block[2..34]` as qs. ✓

### Q4_K — `dequantize_row_q4_K` (line 1352)

Reference:
```c
int is = 0;
for (int j = 0; j < QK_K; j += 64) {
    get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
    const float d1 = d * sc; const float m1 = min * m;
    get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
    const float d2 = d * sc; const float m2 = min * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
    q += 32; is += 2;
}
```

Port (`dequant_q4_k_block`): matches — same `is` step, same `y` / `q` strides,
same low-nibble-first for y[0..32], high-nibble for y[32..64]. ✓

Block layout check: `block_q4_K` = `{ ggml_half d; ggml_half dmin;
uint8_t scales[12]; uint8_t qs[128]; }` = 144 B. Port reads
d=block[0..2], dmin=block[2..4], scales=block[4..16], qs=block[16..144]. ✓

`get_scale_min_k4` (line 703):
```c
if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
       *m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4); }
```

Port (`get_scale_min_k4`): matches identically, including the subtle
`q[j-0]` vs `q[j]` (same thing in C) used for the `m` in the `j>=4`
branch. ✓

### Q5_K — `dequantize_row_q5_K` (line 1554)

Reference:
```c
uint8_t u1 = 1, u2 = 2;
for (int j = 0; j < QK_K; j += 64) {
    // ... d1, m1, d2, m2 same as Q4_K
    for (l 0..32) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    for (l 0..32) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    ql += 32; is += 2;
    u1 <<= 2; u2 <<= 2;
}
```

Port (`dequant_q5_k_block`): matches, including that `qh` is NOT advanced
across iterations — it's indexed by `l` (0..32) with a bit-plane shift
(`u1,u2`) that progresses each iter. The Rust port correctly does
`qh[l]` (not `qh[qh_ofs + l]`) and shifts `u1 <<= 2; u2 <<= 2;` at loop
tail. ✓

Block layout check: `block_q5_K` = `{ d, dmin, scales[12], qh[32], qs[128] }`
= 176 B. Port: d=[0..2], dmin=[2..4], scales=[4..16], qh=[16..48],
qs=[48..176]. ✓ (Note: qh comes before qs in the on-disk layout, and the
port reads it in that order.)

### Q6_K — `dequantize_row_q6_K` (line 1762)

Reference:
```c
for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; ++l) {
        int is = l/16;
        const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l +  0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128; ql += 64; qh += 32; sc += 8;
}
```

Port (`dequant_q6_k_block`): matches. The `- 32` re-centering is expressed
as `... as i32 - 32` which produces the same value (0..63 - 32 = -32..31).
The C `(int8_t)...` cast truncates a u8-ish to signed, which for a value
in 0..63 is identical to widening (no high bit ever set). The Rust port's
`as i32 - 32` is equivalent. ✓

Block layout check: `block_q6_K` = `{ ql[128], qh[64], scales[16] (int8_t),
ggml_half d }` = 210 B. **d is at the tail**, not the head — easy to get
wrong. Port: ql=[0..128], qh=[128..192], scales=[192..208], d=[208..210]. ✓
(The builder explicitly flagged this in the module docs.)

### F32 / F16 / BF16 passthroughs

Straightforward chunked-read-and-cast. Uses little-endian byte reads via
`from_le_bytes`, and `half::{f16, bf16}::from_bits(...)` for the 2-byte
types. ✓

---

## Header parsing audit

### Magic + version

- Magic: compares against `b"GGUF"` literal. ✓
- Version: accepts 2 or 3, rejects others. ✓ (v1 is ancient and not worth
  supporting.)

### Little-endian integer reads

Every multi-byte integer is read via `u32::from_le_bytes` / `u64::from_le_bytes`.
No use of `ByteOrder::NativeEndian`. ✓

### Strings

`read_string` reads `u64 len` then `len` UTF-8 bytes, **not** null-terminated.
Uses `String::from_utf8(bytes.to_vec())` which validates UTF-8. Matches spec. ✓

### Metadata value types

Handles all 13 canonical types (U8/I8/U16/I16/U32/I32/F32/BOOL/STRING/ARRAY/
U64/I64/F64). Arrays recurse through `skip_value`. ✓ Type codes match
`ggml_type` definitions in `ggml-common.h` / gguf.md spec.

### Alignment

- Default 32 (matches spec).
- `general.alignment` parsing handles U32, U64, I32, I64 — any non-integer
  type is silently skipped, leaving the default 32. Reasonable.
- `if alignment == 0 { alignment = DEFAULT_ALIGNMENT; }` guards against
  malformed files declaring `alignment=0`.
- `align_up(x, a) = (x + a - 1) / a * a` — textbook formula, correct for
  any `a > 0`. (Not required to be a power of two, though in practice
  alignment is always a power of 2 and 32 is typical.) ✓

### Tensor offset resolution

Per spec, tensor offsets are relative to the start of the post-alignment
data blob. Port does:
```rust
let header_end = cur.pos as u64;
let data_base = align_up(header_end, alignment);
for info in infos.iter_mut() {
    info.data_offset = data_base + info.data_offset;
}
```

`header_end` is the cursor position after reading the last tensor info entry
(byte position right before the aligned data blob), which is correct. ✓

### n_dims bound

`if n_dims > 8 { bail!(...) }` — prevents absurd allocations from corrupted
files. ✓

### Streaming memory

Loop in `mod.rs::load_file_gguf_raw` allocates a fresh `Vec<f32>` per tensor,
moves it into `Tensor::from_f32_to_bf16`, and inserts the tensor into the
output HashMap. The `Vec<f32>` is dropped at the end of the `let tensor = ...`
expression or at the next loop iteration. No `HashMap<String, Vec<f32>>`
accumulation. ✓ Peak host RAM is bounded by `max_tensor_size_elements * 4`
(plus the final `HashMap<String, Tensor>`, which holds device pointers, not
host data).

---

## Dim reversal audit (HIGH RISK)

See P1-3. The reversal is correct *in theory* for both 2D linear weights
(GGUF `[in, out]` → `[out, in]` matches PyTorch) and 2D conv weights
(GGUF `[kW, kH, IC, OC]` → `[OC, IC, kH, kW]` matches PyTorch NCHW). The
theory is also correct for 1D biases (length-N stays length-N under
reversal) and 4D Conv weights.

What's missing is empirical validation. The `load_real_file` test reads a
GGUF header and asserts the tensor count is nonzero, but never checks a
single shape. If any model family's GGUF ever uses a different convention
(e.g. an ggml-Python encoder that didn't reverse the dims, or a transposed
weight format), we'd load silently-wrong tensors. Wrong-shape tensors
would hit an error at matmul time; wrong-order-same-product tensors
(e.g. a square matrix) would silently produce garbage.

Risk is mitigated by the fact that (a) all mainstream diffusion GGUF
exporters go through `gguf-py` which produces consistent dim ordering, and
(b) the first few forward passes on a real model would immediately produce
visibly-broken output if the weights were transposed. But "visibly broken"
only holds if you actually run inference — which the current test suite
doesn't.

---

## Test-tautology check

Did the tests use llama.cpp reference output, or self-consistency?

**Self-consistency only.** Every "known-values" test constructs a block
from first principles, then derives the expected output from the same
formula the code uses, then asserts they match. This is a refactor-safety
net, not a correctness proof.

Specifically:
- `dequant_q8_0_known_values`: expected = `2.0 * q`. The code computes
  `d * q` with `d = 2.0`. Trivially tautological.
- `dequant_q4_k_known_values`: expected = `1.0 * 1 * 7 = 7.0` for y[0..32].
  The code computes `d * sc0 * (q & 0xF) - m0f` with d=1, sc0=1, q=0x77, m0f=0.
  Matches by construction.
- `dequant_q5_k_known_values`: expected = 16 because "qh bit is set → add 16".
  The code computes "qh bit is set → add 16". Matches by construction.
- `dequant_q6_k_known_values`: expected = -32 because "q=0-32". The code
  computes `(q_raw) - 32` with q_raw=0. Matches by construction.

A genuine ground-truth test would, for each quant type:
1. Take a random (seeded) 256-element f32 weight vector.
2. Quantize it via `llama.cpp`'s `quantize_row_*` (C reference).
3. Dequantize the quantized bytes via THIS Rust implementation.
4. Compare to the f32 vector recovered by `llama.cpp`'s `dequantize_row_*`
   (same C, for round-trip consistency), OR to the expected quantization
   error bound (~1-2% for Q4_K, ~0.5% for Q5_K, ~0.25% for Q6_K).

The builder's claim that they "cross-checked by constructing a superblock
where only sub-block 0 has a non-zero scale" is accurate, but the
cross-check is against their own derivation of the formula, not against
an independent implementation. Not a bug today — the code IS correct —
but the tests are weak regression nets, not correctness oracles.

---

## BF16 upload path audit

`mod.rs::load_file_gguf_raw`:
```rust
let data_f32 = dequant::dequantize_to_f32(info.quant, bytes, n_elems)?;
let tensor = Tensor::from_f32_to_bf16(
    data_f32,
    Shape::from_dims(&info.dims),
    device.clone(),
)?;
out.insert(info.name.clone(), tensor);
```

- `data_f32` is consumed by move into the BF16 constructor. No clone. ✓
- `device.clone()` clones `Arc<CudaDevice>`, which is a refcount bump. ✓
- `Shape::from_dims(&info.dims)` — `info.dims` is the REVERSED GGUF dims
  (see dim reversal audit). For linear weights this becomes `[out, in]`
  matching safetensors. ✓
- `Tensor::from_f32_to_bf16` delegates to `BF16Ops::from_f32` in flame-core.
  Not audited here; assumed correct.
- No `Vec<f32>` accumulation across tensors — each iteration's `data_f32`
  drops when the `let tensor = ...` expression ends (the moved-from Vec
  is consumed by `from_f32_to_bf16`; it may or may not retain internal
  staging, but that's flame-core's problem). ✓

---

## Alignment off-by-one audit

Potential bugs looked for:
- Using `header_end` instead of `align_up(header_end, alignment)` as the
  data base → **not present, code uses `align_up`.**
- Using absolute tensor_info offset (pre-alignment) as the data offset →
  **not present, code re-absolutes via `data_base + data_offset_rel`.**
- Off-by-one in `align_up` (e.g. `(x + a) / a * a` which would over-align
  when x is already aligned) → **not present, uses `(x + a - 1) / a * a`.**
- Alignment applied to string lengths or metadata values → **not present,
  alignment only applies to the transition from tensor-info table to
  tensor-data blob.**

---

## Key remap audit

See P1-2. `default_rename` strips one of three prefixes, in priority:
`model.diffusion_model.` > `transformer.` > `first_stage_model.`. This is
correct for every diffusion GGUF I know of today (FLUX `city96` uses
`model.diffusion_model.`, HF diffusers uses `transformer.`, SD.cpp VAE
uses `first_stage_model.`). It would break silently for a future file that
nested two of the prefixes.

Tests cover all four cases (three prefixes + passthrough). ✓

---

## Unverified

1. **Actual FLUX GGUF round-trip**: no real GGUF was loaded during this
   review. The `load_real_file` test requires `GGUF_TEST_FILE=...` and
   CUDA, and none was run. Dim reversal and shape correctness on a real
   file are assumed from the code analysis, not proven.
2. **flame-core `from_f32_to_bf16` semantics** for zero-element tensors:
   not inspected. If a GGUF contains an empty tensor, behavior is unknown.
3. **flame-core `Shape::from_dims(&[])`** for 0-dim scalar tensors: not
   inspected. Not known to occur in diffusion GGUFs.
4. **6-bit scale packing edge cases**: the tests only exercise the `j<4`
   branch of `get_scale_min_k4`. The `j>=4` branch is ported correctly
   by visual comparison but not unit-tested.
5. **Q6_K `((qh[l] >> shift) & 3) << 4` for shifts 2, 4, 6**: the test
   exercises shift 0 only (qh=0 everywhere). Other shifts are ported by
   visual comparison only.
6. **Q5_K u1/u2 progression past first iteration**: the test exits before
   `u1 <<= 2` and `u2 <<= 2` would take effect. Subsequent iterations'
   mask arithmetic is unit-untested.
7. **Pre-existing bin compile failure**: builder noted `cargo test --test
   gguf` fails due to unrelated binary compile errors. Not a GGUF-loader
   bug, but it means CI signal for this module is currently noisy.
