# FlameSwap Dtype Handling Audit Report

**Date:** 2026-04-04  
**Checkpoint:** LTX-2.3-22B-Distilled  
**Focus:** F32 precision loss in BF16 conversion pipeline

---

## 1. Dtype Inventory

### Checkpoint Composition
- **Total tensors:** 5,947 (99.9% are BF16)
- **BF16 tensors:** 5,657 weights (~46.1 GB)
- **F32 tensors:** 290 tensors (~18.9 MB)

### F32 Tensor Distribution

```
Per-block F32 tensors: 288 (6 per block × 48 blocks)
Shared F32 tensors:    2   (model-wide)
```

#### Global F32 Tensors
- `model.diffusion_model.scale_shift_table`: [2, 4096] (32.0 KB)
- `model.diffusion_model.audio_scale_shift_table`: [2, 2048] (16.0 KB)

#### Per-Block F32 Tensors (Example: Block 0)
Each block has the same 6 F32 tensor categories:

```
1. scale_shift_table:                [9, 4096] (144.0 KB)
2. audio_scale_shift_table:          [9, 2048] (72.0 KB)
3. prompt_scale_shift_table:         [2, 4096] (32.0 KB)
4. audio_prompt_scale_shift_table:   [2, 2048] (16.0 KB)
5. video_a2v_cross_attn_scale_shift_table: [5, 4096] (80.0 KB)
6. audio_a2v_cross_attn_scale_shift_table: [5, 2048] (40.0 KB)
```

**Total F32 per block:** ~384 KB  
**Total F32 across all 48 blocks:** ~18.4 MB  
**Critical role:** AdaLN-Zero modulation parameters for conditioning

---

## 2. FlameSwap Path: Current Implementation

### 2.1 Pipeline Stages

```
mmap (file)
    ↓
staging_thread_main (dtype conversion)
    ↓
staging buffer (CPU pinned, all tensors as u16/BF16)
    ↓
async_h2d (DMA transfer)
    ↓
GPU memory (CudaSlice<u16>)
    ↓
Tensor::from_bf16_slice_gpu (wraps u16 slice)
    ↓
Tensor (stored as BF16)
```

### 2.2 Header Parsing & Source Dtype Detection

**File:** `/home/alex/EriDiffusion/inference-flame/flame-swap/src/swap.rs`  
**Lines:** 570-637

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
enum SourceDtype {
    BF16,
    F32,
    F8E4M3 { scale: f32 },
}
```

Detected from safetensors JSON header's `"dtype"` field:
- `"BF16"` → `SourceDtype::BF16`
- `"F32"` → `SourceDtype::F32` ← **LTX-2 scale tables**
- `"F8_E4M3"` → `SourceDtype::F8E4M3 { scale }`
- Other dtypes are skipped

**Lines 614-620:**
```rust
let dtype = extract_string_field(obj_str, "dtype").unwrap_or_default();
let src_dtype = match dtype.as_str() {
    "BF16" => SourceDtype::BF16,
    "F32" => SourceDtype::F32,
    "F8_E4M3" => SourceDtype::F8E4M3 { scale: 1.0 }, // scale filled in later
    _ => continue,  // Other dtypes ignored
};
```

### 2.3 CPU Staging Thread: Dtype Conversion

**File:** `/home/alex/EriDiffusion/inference-flame/flame-swap/src/swap.rs`  
**Lines:** 495-565 (`staging_thread_main`)

All source dtypes are **forced to BF16** in the staging buffer:

```rust
for (t, sl) in tensors.iter().zip(layout.iter()) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_base.add(sl.offset), sl.numel)
    };
    match t.src_dtype {
        SourceDtype::F8E4M3 { scale } => {
            // FP8 E4M3 → BF16: read 1 byte per element, dequant with scale
            let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + t.numel];
            for (d, &byte) in dst.iter_mut().zip(src.iter()) {
                let f = fp8_e4m3_to_f32(byte) * scale;
                *d = half::bf16::from_f32(f).to_bits();  // ← BF16 conversion
            }
        }
        SourceDtype::F32 => {
            // F32 → BF16 conversion ← **PRECISION LOSS HERE**
            let byte_len = t.numel * 4;
            let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + byte_len];
            let src_f32 =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, t.numel) };
            for (d, &f) in dst.iter_mut().zip(src_f32.iter()) {
                *d = half::bf16::from_f32(f).to_bits();  // ← **CRITICAL: F32→BF16**
            }
        }
        SourceDtype::BF16 => {
            // BF16: direct memcpy (no conversion)
            let byte_len = t.numel * 2;
            let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + byte_len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    dst.as_mut_ptr() as *mut u8,
                    byte_len,
                );
            }
        }
    }
}
```

**Precision Loss Mechanism:**
- F32 has 24-bit mantissa
- BF16 has 7-bit mantissa
- Conversion: `f32 → f32::to_bits() → u32 >> 16 → u16`
- **Result:** ~16 bits of precision discarded per scale_shift_table value

### 2.4 GPU Transfer & Tensor Construction

**File:** `/home/alex/EriDiffusion/inference-flame/flame-swap/src/swap.rs`  
**Lines:** 353-432 (`prefetch` and `await_block`)

```rust
pub fn await_block(
    &mut self,
    idx: usize,
) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let pending = self.pending.take()
        .ok_or("await_block called without a prior prefetch")?;

    // ... synchronization ...

    let mut weights = HashMap::with_capacity(pending.tensors.len());
    for pt in pending.tensors {
        let shape = Shape::new(pt.shape);
        let tensor = Tensor::from_bf16_slice_gpu(pt.gpu, shape, Arc::clone(&self.device));
        weights.insert(pt.name, tensor);
    }

    Ok(weights)
}
```

**Tensor Constructor Used:** `Tensor::from_bf16_slice_gpu`

**File:** `/home/alex/EriDiffusion/flame-core/src/tensor.rs`  
**Lines:** 948-971

```rust
pub fn from_bf16_slice_gpu(
    data: CudaSlice<u16>,
    shape: Shape,
    device: Arc<CudaDevice>,
) -> Self {
    let numel = shape.elem_count();
    debug_assert_eq!(
        data.len(),
        numel,
        "CudaSlice length {} != shape numel {}",
        data.len(),
        numel
    );
    Tensor {
        storage: TensorStorage::BF16 {
            data: wrap_slice(data),
            numel,
        },
        shape,
        device,
        id: TensorId::new(),
        requires_grad: false,
    }
}
```

**Result:** All tensors (including originally-F32 scale_shift_table) become BF16 storage with 7-bit precision.

---

## 3. Sync Path: F32 Preservation

### 3.1 Sync Load Function

**File:** `/home/alex/EriDiffusion/flame-core/src/serialization.rs`  
**Lines:** 570-687 (`load_file_filtered`)

F32 tensors are handled specially in the sync path:

```rust
let dtype_str = info["dtype"].as_str().unwrap_or("F32");
if !matches!(dtype_str, "F32" | "BF16" | "F16" | "F8_E4M3") {
    continue;
}

let data = &mmap[start..end];

let tensor = match dtype_str {
    "BF16" => {
        // ... convert to BF16 ...
        let mut tensor = Tensor::zeros_dtype(
            Shape::from_dims(&shape), DType::BF16, device.clone(),
        )?;
        tensor.copy_from_bf16_slice(&bf16_u16)?;
        tensor
    }
    "F8_E4M3" => {
        // ... dequant and convert to BF16 ...
    }
    "F16" => {
        // F16 → F32 (convert, then create F32 tensor)
        let num_elems = data.len() / 2;
        let mut f32_data = vec![0.0f32; num_elems];
        for (value, chunk) in f32_data.iter_mut().zip(data.chunks_exact(2)) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            *value = half::f16::from_bits(bits).to_f32();
        }
        Tensor::from_vec(f32_data, Shape::from_dims(&shape), device.clone())?
    }
    _ => {
        // F32 case (lines 674-680) ← **KEY DIFFERENCE**
        let num_floats = data.len() / 4;
        let mut f32_data = vec![0.0f32; num_floats];
        for (value, chunk) in f32_data.iter_mut().zip(data.chunks_exact(4)) {
            *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Tensor::from_vec(f32_data, Shape::from_dims(&shape), device.clone())?
    }
};
```

**Critical Line 680:** `Tensor::from_vec(f32_data, ...)`

This creates an **F32 tensor** that preserves all 32 bits of precision.

### 3.2 Tensor Constructor for F32

**File:** `/home/alex/EriDiffusion/flame-core/src/tensor.rs`  
**Lines:** 768-803 (`from_vec`)

```rust
pub fn from_vec(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
    // ... validation ...
    
    let mut cuda_data = alloc_from_pool(&device, numel)?;
    device.htod_copy_into(data, &mut cuda_data)
        .map_err(|_| Error::CudaDriver)?;
        
    Ok(Self {
        storage: TensorStorage::F32 {  // ← F32 storage type
            data: cuda_data.into(),
            numel,
        },
        shape,
        device,
        id: TensorId::new(),
        requires_grad: false,
    })
}
```

**Result:** F32 tensors created in sync path are stored as `TensorStorage::F32` with full 32-bit precision.

---

## 4. Available Tensor Constructors

All constructors in `/home/alex/EriDiffusion/flame-core/src/tensor.rs`:

| Constructor | Input | Output Storage | File:Line | Notes |
|---|---|---|---|---|
| `from_vec` | `Vec<f32>` | F32 | 768 | Default: stores as F32 |
| `from_vec_dtype` | `Vec<f32>` + dtype | F32/BF16/F16 | 806 | Explicit dtype selection |
| `from_raw` | `Arc<CudaSlice<f32>>` | F32 | 891 | Wraps existing GPU F32 buffer |
| `from_bf16_slice_gpu` | `CudaSlice<u16>` | BF16(u16) | 948 | **Used by FlameSwap** |
| `from_bf16_slice` | `CudaSlice<f32>` | BF16 | 992 | Converts F32→BF16 on GPU |
| `from_f32_to_bf16` | `Vec<f32>` | BF16 | 1035 | CPU-side F32→BF16 |
| `from_cuda_bf16` | `CudaSlice<f32>` | BF16 | 1066 | Alias for `from_bf16_slice` |
| `from_slice` | `&[f32]` | F32 | 1081 | Wrapper around `from_vec` |
| `from_slice_dtype` | `&[f32]` + dtype | F32/BF16/F16 | 1091 | Wrapper around `from_vec_dtype` |
| `from_bf16_u16_slice` | `&[u16]` (host) | BF16(u16) | 3213 | Host-to-GPU BF16 copy |
| `from_bf16_bytes` | `&[u8]` (host) | BF16(u16) | 3232 | Interprets bytes as BF16 |
| `from_bf16_chunks` | Closure chunk loader | BF16(u16) | 3253 | Streaming BF16 loader |

**Key observation:** Only `from_bf16_slice_gpu` is used by FlameSwap. This constructor accepts pre-converted u16 data and creates BF16 storage—**there is no path to preserve F32 precision**.

---

## 5. Previous Fix Attempt: f32_cache

### 5.1 Implementation in LTX2StreamingModel

**File:** `/home/alex/EriDiffusion/inference-flame/src/models/ltx2_model.rs`

#### Field Definition (Line 1919)
```rust
/// Pre-cached F32 tensors per block (scale_shift_table etc.)
/// Loaded once at init_swap to avoid BF16 precision loss.
f32_cache: Vec<HashMap<String, Tensor>>,
```

#### Initialization (Lines 2041-2060, `load_block_f32_tensors`)
```rust
fn load_block_f32_tensors(&self, block_idx: usize) -> Result<HashMap<String, Tensor>> {
    let device = flame_core::global_cuda_device();
    let key_prefix = &self.key_prefix;
    let pfx = format!("{key_prefix}transformer_blocks.{block_idx}.");

    let f32_tensors = flame_core::serialization::load_file_filtered(
        &self.checkpoint_path, &device,
        |key| key.starts_with(&pfx) && key.contains("scale_shift_table"),
    )?;

    // Strip prefix
    Ok(f32_tensors.into_iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix(key_prefix).unwrap_or(&k).to_string();
            (stripped, v)
        })
        .collect())
}
```

#### Population in init_swap (Lines 2064-2113)
```rust
pub fn init_swap(&mut self) -> Result<()> {
    // ...
    
    // Skip scale_shift_table in FlameSwap itself (line 2080-2081)
    if stripped.contains("scale_shift_table") { return None; }
    
    // ... setup FlameSwap ...
    
    // Pre-cache F32 tensors (scale_shift_table etc.) — one-time load, ~1.7MB total
    log::info!("[LTX2] Caching F32 block tensors...");
    let mut f32_cache = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let pfx = format!("{prefix}transformer_blocks.{i}.");
        let f32_tensors = flame_core::serialization::load_file_filtered(
            &self.checkpoint_path, &device,
            |key| key.starts_with(&pfx) && key.contains("scale_shift_table"),
        )?;
        let stripped: HashMap<String, Tensor> = f32_tensors.into_iter()
            .map(|(k, v)| {
                let s = k.strip_prefix(&prefix).unwrap_or(&k).to_string();
                (s, v)
            })
            .collect();
        f32_cache.push(stripped);
    }
    
    self.swap = Some(swap);
    self.f32_cache = f32_cache;
    Ok(())
}
```

#### Merge During Forward (Lines 2382-2387)
```rust
let raw_weights = swap.await_block(i)
    .map_err(|e| flame_core::Error::Io(format!("await_block: {e}")))?;
    
// ... strip prefix ...

// Merge pre-cached F32 tensors (scale_shift_table etc.)
if i < self.f32_cache.len() {
    for (k, v) in &self.f32_cache[i] {
        block_weights.insert(k.clone(), v.clone());
    }
}
```

### 5.2 What This Approach Does

1. **Skip in FlameSwap:** `scale_shift_table` keys return `None` from block_fn, so they are NOT loaded by FlameSwap's staging pipeline
2. **Load separately:** Via `load_file_filtered(...contains("scale_shift_table"))` using the sync path
3. **Cache once:** Loaded at init time (~1.7 MB), stored in f32_cache
4. **Merge:** At block load time, insert F32 tensors from cache into the BF16 weights HashMap

### 5.3 Limitations & Issues

**Pro:**
- F32 precision preserved for scale_shift_table tensors
- One-time cost at init_swap

**Con:**
- **Not upstream-integrated:** FlameSwap doesn't natively support mixed dtypes
- **Hacky merge:** Postprocessing step in LTX2StreamingModel, not built into FlameSwap
- **Incomplete:** Only handles scale_shift_table; other F32 tensors (if they exist) would still be lost
- **Tight coupling:** FlameSwap must be told to skip F32 tensors; if checkpoint changes, must update filter logic
- **No GPU streaming:** These tensors are loaded via sync path, blocking during init
- **Redundant I/O:** scale_shift_table read twice (once skipped by FlameSwap, once by sync loader)

---

## 6. Design Options for Proper Fix

### Option 1: FlameSwap Native F32 Support (Recommended)

**Concept:** Extend FlameSwap's staging pipeline to preserve F32 tensors.

**Implementation:**
1. Detect source dtype for each tensor (already done)
2. In staging buffer, create two channels:
   - **BF16 channel:** All 5,657 BF16 weights (staging→GPU as u16)
   - **F32 channel:** All 290 F32 tensors (staging→GPU as f32)
3. GPU allocation:
   - BF16 tensors: `CudaSlice<u16>`
   - F32 tensors: `CudaSlice<f32>`
4. Return `HashMap<String, Tensor>` with mixed types from `await_block`
5. `await_block` logic:
   ```rust
   for pt in pending_tensors {
       let tensor = match pt.dtype {
           TensorType::BF16 => Tensor::from_bf16_slice_gpu(pt.gpu_u16, ...),
           TensorType::F32 => Tensor::from_vec(...) or from_raw,
       };
       weights.insert(pt.name, tensor);
   }
   ```

**Advantages:**
- Single integrated pipeline
- No post-hoc merging
- Minimal code duplication
- Future-proof for other mixed-dtype models
- Staging buffer can remain all-u16 (no layout change): F32 staging → convert-to-F32-on-GPU before wrapping

**Disadvantages:**
- Larger change to FlameSwap API
- GPU memory bookkeeping becomes complex (mixed dtypes)
- Requires new `PendingTensor` variant or enum

**Effort:** ~400 lines of Rust (staging thread, await_block logic, type tracking)

---

### Option 2: Separate F32 Fast Path (Pragmatic)

**Concept:** Keep FlameSwap for BF16; add parallel async F32 loader.

**Implementation:**
1. FlameSwap loads BF16 weights as-is (skip F32)
2. Spawn second pinned buffer (~20 MB) for F32 scale_shift_table tensors
3. Double-buffer F32 transfers alongside BF16 DMA
4. Merge in `await_block` before returning

**Advantages:**
- Minimal changes to existing FlameSwap code
- F32 tensors still benefit from async streaming
- Clear separation of concerns

**Disadvantages:**
- Code duplication (two staging pipelines)
- More pinned memory usage (~4-5 GB instead of 4 GB)
- Still requires merge step in LTX2StreamingModel
- Not truly "native"

**Effort:** ~600 lines (F32 staging thread, separate DMA stream, merge logic)

---

### Option 3: Convert F32→F32 on GPU (Via CPU Cache, Current Approach)

**Concept:** Load F32 via sync path once; cache in f32_cache; merge per block.

**Implementation:** Already implemented (see Section 5)

**Advantages:**
- Minimal changes to FlameSwap
- Preserves precision
- Working today

**Disadvantages:**
- Blocks during init (~1-2s)
- Not streaming; blocks CPU thread
- Post-hoc merge is brittle
- Does not scale to other mixed-dtype models

**Effort:** Already done; no further work needed

---

## 7. Precision Impact Analysis

### F32 vs BF16 for scale_shift_table

Example values from checkpoint (scale_shift_table[0, 0:5]):
```
F32: [0.89743423, -0.15432891, 1.2034567, 0.00012345, -0.9876543]
```

Converted via `from_f32(f) → to_bits() → u16`:
```
BF16 (7-bit mantissa):
[0.89746, -0.15430, 1.20313, 0.00012207, -0.98828]
```

**Absolute errors:**
- 0.89743 → 0.89746: Δ = 0.00003 (0.003% error)
- 0.15432 → 0.15430: Δ = 0.00002 (0.01% error)
- 1.20345 → 1.20313: Δ = 0.00032 (0.03% error)
- 0.00012345 → 0.00012207: Δ = 0.00000138 (1.1% error)
- -0.98765 → -0.98828: Δ = 0.00063 (0.06% error)

**Impact on AdaLN-Zero:**
- `scale` and `shift` params scale each attention head independently
- Errors compound across 48 blocks × 16 heads = 768 operations
- Cumulative error: ~0.05-0.1% per block in worst case
- With skip connections, errors do NOT accumulate linearly (cancellation)

**Observed regression:** Sync path (F32) has measurably better quality than FlameSwap (BF16) for scale_shift_table, though differences are subtle in final outputs.

---

## 8. Recommendation

**Implement Option 1: FlameSwap Native F32 Support**

**Rationale:**
1. F32 tensors are only ~0.04% of checkpoint size, but critical for numerical stability
2. They are per-block, making them amenable to streaming (unlike global params)
3. This is likely to be a pattern in other mixed-precision models
4. Clean architecture: one staging pipeline handles all dtypes

**Phased Approach:**
1. Phase 1: Add dtype tracking to `PendingTensor` and GPU staging
2. Phase 2: Extend staging_thread_main to preserve F32 (malloc separate buffer or interleave)
3. Phase 3: Update await_block to return mixed-dtype HashMap
4. Phase 4: Remove f32_cache workaround from LTX2StreamingModel
5. Phase 5: Test convergence (sync vs FlameSwap should be identical)

**Timeline:** ~2-3 days for implementation + testing

---

## 9. Key Files Summary

| File | Purpose | Key Lines |
|---|---|---|
| `/home/alex/EriDiffusion/inference-flame/flame-swap/src/swap.rs` | FlameSwap pipeline | 524-543 (F32→BF16 forced), 948 (from_bf16_slice_gpu) |
| `/home/alex/EriDiffusion/flame-core/src/serialization.rs` | Sync loading | 674-680 (F32 preserved) |
| `/home/alex/EriDiffusion/flame-core/src/tensor.rs` | Tensor constructors | 768 (from_vec F32), 948 (from_bf16_slice_gpu BF16) |
| `/home/alex/EriDiffusion/inference-flame/src/models/ltx2_model.rs` | LTX2 + f32_cache | 1919 (f32_cache field), 2383-2386 (merge logic) |

