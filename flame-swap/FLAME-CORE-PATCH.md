# flame-core additions for flame-swap

Three small changes needed in flame-core before flame-swap compiles.

---

## 1. `Tensor::from_bf16_slice_gpu` constructor

**File**: `flame-core/src/tensor.rs`

Add after the existing `from_bf16_arena` constructor (~line 916):

```rust
/// Construct a BF16 tensor from a pre-populated GPU buffer.
///
/// Takes ownership of the `CudaSlice<u16>`.  No copy — the slice IS
/// the tensor's storage.  Used by flame-swap to wrap async-transferred
/// block weights as tensors.
#[cfg(feature = "bf16_u16")]
pub fn from_bf16_slice_gpu(
    data: CudaSlice<u16>,
    shape: Shape,
    device: Arc<CudaDevice>,
) -> Self {
    let numel = shape.numel();
    debug_assert_eq!(
        data.len(), numel,
        "CudaSlice length {} != shape numel {}",
        data.len(), numel
    );
    Tensor {
        #[cfg(feature = "shared_storage")]
        storage: TensorStorage::BF16 { data: Arc::new(data), numel },
        #[cfg(not(feature = "shared_storage"))]
        storage: TensorStorage::BF16 { data, numel },
        shape,
        device,
        id: TensorId::new(),
        requires_grad: false,
    }
}
```

---

## 2. Make `PinnedHostBuffer::as_ptr()` public

**File**: `flame-core/src/pinned.rs`

The `as_ptr()` method (if it exists) needs to be `pub`. If it doesn't exist, add:

```rust
/// Raw pointer to the underlying pinned allocation.
pub fn as_ptr(&self) -> *const T {
    self.ptr.as_ptr() as *const T
}

/// Number of initialized elements.
pub fn len(&self) -> usize {
    self.len
}
```

---

## 3. Export `flame_cuda_memcpy_async` from flame-core (optional)

Currently declared in `flame-core/src/cuda/ffi.rs` but may not be `pub`.

**Option A**: Make it `pub` so flame-swap can use it directly.

**Option B**: flame-swap re-declares the same extern (what the current code does). Works fine — the linker resolves to the same symbol.

Option B is simpler and avoids touching flame-core's public API. The crate already does this.

---

## Summary

| Change | File | Lines |
|--------|------|-------|
| `Tensor::from_bf16_slice_gpu` | `tensor.rs` | ~20 |
| `PinnedHostBuffer::as_ptr/len` | `pinned.rs` | ~8 |
| (optional) export memcpy_async | `cuda/ffi.rs` | 1 |

Total: ~30 lines added to flame-core.
