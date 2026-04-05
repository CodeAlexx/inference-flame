# Wire flame-swap into FLAME inference

## Prerequisites
- `git stash` before doing anything
- Read `flame-swap/FLAME-CORE-PATCH.md` first — apply those 3 changes to flame-core

## Phase 1: Apply flame-core patches

1. Add `Tensor::from_bf16_slice_gpu` to `flame-core/src/tensor.rs` (see patch file)
2. Ensure `PinnedHostBuffer::as_ptr()` and `PinnedHostBuffer::len()` are `pub` in `flame-core/src/pinned.rs`
3. Run `cargo check -p flame-core` — must compile clean

## Phase 2: Add flame-swap to workspace

1. Add `flame-swap` as a workspace member in the root `Cargo.toml`
2. Run `cargo check -p flame-swap` — fix any compile errors
3. Show me the terminal output

## Phase 3: Wire into KleinOffloaded

**File**: `inference-flame/src/models/klein.rs`

Replace the synchronous `load_block_to_gpu` / `upload_weight` pattern in `KleinOffloaded::forward` with `FlameSwap`.

The current pattern (from the audit):
```
for each block:
    upload all weights synchronously (alloc + htod_sync_copy)
    forward pass
    drop weights
```

New pattern:
```
let mut swap = FlameSwap::load(&[model_path], &device, |name| {
    // "double_blocks.5.img_attn.qkv.weight" -> Some(5)
    let rest = name.strip_prefix("double_blocks.")?;
    rest.split('.').next()?.parse().ok()
})?;

swap.prefetch(0)?;
for i in 0..num_blocks {
    if i + 1 < num_blocks {
        swap.prefetch(i + 1)?;
    }
    let weights = swap.await_block(i)?;
    // forward pass using weights HashMap (same key lookup as before)
    // weights dropped automatically
}
```

- `CpuWeight` struct and its `Vec<u16>` storage can be removed — FlameSwap handles CPU storage in pinned memory
- Shared/non-block weights (embeddings, final norm, etc.) still load via the existing `load_file_filtered` path
- Do NOT change the forward math — only the weight loading path

## Phase 4: Wire into LTX2StreamingModel

**File**: `inference-flame/src/models/ltx2_model.rs`

Same pattern. Replace `load_block` / `preload_blocks` with FlameSwap.

The block prefix for LTX-2 will be something like `transformer_blocks.` — check the actual safetensors key names by running:
```bash
python3 -c "import json; f=open('model.safetensors','rb'); n=int.from_bytes(f.read(8),'little'); h=json.loads(f.read(n)); [print(k) for k in sorted(h.keys()) if k!='__metadata__']" 
```

Or use flame-core's existing header parser.

## Rules
- One phase at a time — show terminal output after each `cargo check`
- Do NOT modify flame-swap source unless there's a compile error
- Do NOT touch forward pass math — only weight loading paths
- If you break something, show me the error before attempting a fix
