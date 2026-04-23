//! TurboBlockLoader — VMM-backed double-buffered block loader.
//!
//! ## Architecture
//!
//! At construction time we:
//!   * mmap the safetensors file once,
//!   * walk the metadata for every weight matching one of the configured
//!     block prefixes, parse each tensor's BF16 bytes into a pinned host
//!     `Vec<u16>` (converting from F16/F32/BF16 as needed), and
//!   * compute, per block, a *layout* (key → `(byte_offset, num_bytes,
//!     shape)`) and the block's total byte size after granularity rounding.
//!   * reserve one virtual slab in the VMM allocator that fits two slot
//!     regions sized to the largest block.
//!
//! Each `prefetch_block(idx)`:
//!   1. Picks the non-current slot (errors if it is still `Prepared` and
//!      hasn't been awaited).
//!   2. Calls `arena.allocator.ensure_resident(slab, region, copy_stream)`.
//!      Cold path on first use → maps physical memory; hot path thereafter is
//!      a single atomic load + refcount bump.
//!   3. Issues `cudaMemcpyAsync` (via flame-core's pinned helper, on
//!      `copy_stream`) for every weight in the block, writing into
//!      `slot_handle.as_ptr() + key.byte_offset`.
//!   4. Records `ready_event` on `copy_stream` and stores the handle +
//!      event in the slot.
//!
//! Each `await_block(idx)`:
//!   1. Finds the slot whose pending block matches.
//!   2. If `Staging`, makes the caller's compute stream wait for the
//!      `ready_event` via `cuStreamWaitEvent` (GPU-side, no host sync).
//!   3. Builds a `HashMap<String, Tensor>` whose `Tensor`s are constructed
//!      via `Tensor::from_bf16_device_ptr_non_owning` over `slot_ptr +
//!      offset`.
//!   4. Wraps in `Arc<TurboBlock>` carrying a clone of `Arc<ResidentHandle>`,
//!      transitions slot to `Prepared`, returns.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{PinnedAllocFlags, PinnedHostBuffer, Shape, Tensor};

use crate::turbo::arena::VmmArena;
use crate::turbo::block::TurboBlock;
use crate::turbo::vmm::cuda_ffi::{self, CUevent, CU_EVENT_DISABLE_TIMING};
use crate::turbo::vmm::{ResidentHandle, SlabId, RegionId, VmmError};

// ---------------------------------------------------------------------------
// Per-block CPU-side weight layout
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WeightLayout {
    /// Byte offset into the slot region for this weight.
    byte_offset: usize,
    /// Number of BF16 elements (u16).
    num_elems: usize,
    shape: Vec<usize>,
}

struct BlockLayout {
    /// Total bytes (granularity-rounded by the VMM allocator) for this block.
    total_bytes: usize,
    /// key → (byte_offset, num_elems, shape).  Iteration order does not
    /// matter for correctness — `await_block` returns a HashMap.
    weights: HashMap<String, WeightLayout>,
    /// CUDA-pinned host bytes for the whole block, packed at the same
    /// offsets as `weights`. cudaMemcpyAsync over pinned host memory stays
    /// truly asynchronous w.r.t. copy_stream — required for compute/H2D
    /// overlap.
    host_buffer: PinnedHostBuffer<u16>,
}

// ---------------------------------------------------------------------------
// Slot state
// ---------------------------------------------------------------------------

enum SlotState {
    Empty,
    Staging {
        block_idx: usize,
        handle: Arc<ResidentHandle>,
        ready_event: CUevent,
    },
    Prepared {
        block_idx: usize,
        block: Arc<TurboBlock>,
    },
}

impl SlotState {
    fn block_idx(&self) -> Option<usize> {
        match self {
            SlotState::Empty => None,
            SlotState::Staging { block_idx, .. } => Some(*block_idx),
            SlotState::Prepared { block_idx, .. } => Some(*block_idx),
        }
    }

    fn take(&mut self) -> SlotState {
        std::mem::replace(self, SlotState::Empty)
    }
}

// ---------------------------------------------------------------------------
// TurboBlockLoader
// ---------------------------------------------------------------------------

pub struct TurboBlockLoader {
    arena: Arc<VmmArena>,
    device: Arc<CudaDevice>,

    /// Per-block CPU layout + packed host bytes.
    blocks: Vec<BlockLayout>,

    /// VMM slab holding our two slots. Two regions defined inside it.
    slab_id: SlabId,
    slot_regions: [RegionId; 2],

    slots: [SlotState; 2],
    /// Index of the slot most recently produced via `await_block`. The other
    /// slot is the prefetch target.
    active: usize,

    pinned_bytes_total: usize,
}

unsafe impl Send for TurboBlockLoader {}
unsafe impl Sync for TurboBlockLoader {}

impl TurboBlockLoader {
    pub fn new(
        model_path: String,
        device: Arc<CudaDevice>,
        arena: Arc<VmmArena>,
        block_prefixes: Vec<String>,
    ) -> Result<Self, VmmError> {
        // Parse safetensors header → for each weight matching a prefix, build
        // BF16-packed CPU layout.
        let blocks = build_block_layouts(&model_path, &block_prefixes)
            .map_err(|e| {
                log::error!("[turbo loader] safetensors parse failed for {model_path}: {e}");
                VmmError::InvalidRegion
            })?;

        let pinned_bytes_total: usize = blocks.iter().map(|b| b.host_buffer.len_bytes()).sum();

        // Slot size = max block size.  ensure_resident maps the whole region.
        let max_block_bytes = blocks.iter().map(|b| b.total_bytes).max().unwrap_or(0);
        if max_block_bytes == 0 {
            log::error!("[turbo loader] no block weights found under any of {:?}", block_prefixes);
            return Err(VmmError::InvalidRegion);
        }

        let slab_id = arena.allocator.create_slab(VmmArena::virtual_reserve_bytes())?;
        let slot_a = arena.allocator.define_region(slab_id, 0, max_block_bytes)?;
        // The allocator rounds region size up to granularity; place slot_b
        // beyond slot_a's effective extent. Query allocator stats to learn
        // the rounded size of slot_a.
        let stats = arena.allocator.stats();
        let rounded = stats
            .slabs
            .get(slab_id)
            .and_then(|opt| opt.as_ref())
            .and_then(|ss| ss.regions.first())
            .map(|rs| rs.size)
            .unwrap_or(max_block_bytes);
        let slot_b = arena.allocator.define_region(slab_id, rounded, max_block_bytes)?;
        // The slab watermark is auto-set to 1 (regions.len() at first
        // define) and only extends on `set_priority`. Bump priority so the
        // watermark covers both slot regions.
        arena.allocator.set_priority(slab_id, 1)?;

        log::info!(
            "[turbo loader] slab {slab_id}: 2 slot regions × {:.1} MiB ({} blocks, {:.1} MiB pinned host)",
            rounded as f64 / (1024.0 * 1024.0),
            blocks.len(),
            pinned_bytes_total as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            arena,
            device,
            blocks,
            slab_id,
            slot_regions: [slot_a, slot_b],
            slots: [SlotState::Empty, SlotState::Empty],
            active: 0,
            pinned_bytes_total,
        })
    }

    pub fn block_count(&self) -> usize { self.blocks.len() }

    pub fn pinned_bytes(&self) -> usize { self.pinned_bytes_total }

    /// Stage `block_idx` into the non-active slot (no host sync). Errors if
    /// the target slot already holds an unawaited Prepared block.
    pub fn prefetch_block(&mut self, block_idx: usize) -> Result<(), VmmError> {
        if block_idx >= self.blocks.len() {
            return Err(VmmError::InvalidRegion);
        }
        // Already on either slot? No-op (matches BlockOffloader semantics).
        // Invariant: only reached after await_block has already promoted the
        // slot Staging → Prepared and consumed the ready_event, so callers do
        // not observe a missing event on the subsequent await.
        if self.slots[0].block_idx() == Some(block_idx)
            || self.slots[1].block_idx() == Some(block_idx)
        {
            return Ok(());
        }

        let target = 1 - self.active;
        // Snapshot whether the target slot holds a Prepared block we need to
        // displace — Klein's forward loop awaits block N (slot becomes
        // active+Prepared) and then immediately prefetches block N+1, at
        // which point the OTHER slot still holds the *previous* iteration's
        // Prepared block.  That previous block's user-visible Arc has been
        // dropped (the local `block` binding fell out of scope) so only the
        // slot itself owns the Arc; we can safely retire the slot.
        //
        // If the user is still holding the Arc, `Arc::strong_count > 1` and
        // we MUST refuse — silently overwriting the slot would race the
        // user's compute reads against the new H2D.
        let needs_event_wait = match &self.slots[target] {
            SlotState::Prepared { block, .. } => {
                if Arc::strong_count(block) > 1 {
                    log::error!(
                        "[turbo loader] prefetch_block({block_idx}): target slot {target} \
                         still Prepared with Arc strong_count={} (caller still holds it)",
                        Arc::strong_count(block),
                    );
                    return Err(VmmError::SlabNotEmpty);
                }
                // Slot owns the only Arc.  When we replace the slot below,
                // the TurboBlock and its ResidentHandle drop synchronously,
                // and ResidentHandle::Drop records a compute-stream event in
                // region.mutable.last_use_event.  We must then have copy_stream
                // wait on that event before issuing the new H2D so the writer
                // doesn't race any compute kernels still reading the slot.
                true
            }
            SlotState::Staging { .. } | SlotState::Empty => false,
        };

        // Replace the slot with Empty.  This drops the prior contents
        // synchronously:
        //   * Staging:  destroy the orphaned ready_event explicitly (the H2D
        //               it gated will be overwritten, no consumer awaits it)
        //               and let the inner handle Arc drop run.
        //   * Prepared: drop the TurboBlock Arc.  When strong_count was 1
        //               (verified above), the TurboBlock and its
        //               ResidentHandle drop here on the calling thread, and
        //               ResidentHandle::Drop records a compute-stream event
        //               into region.mutable.last_use_event for the
        //               wait_for_last_use_event gate below to consume.
        if let SlotState::Staging { ready_event, .. } = self.slots[target].take() {
            unsafe { let _ = cuda_ffi::cuEventDestroy_v2(ready_event); }
        }

        let copy_stream_ptr = self.arena.copy_stream.stream as cuda_ffi::CUstream;
        // Reader's compute stream is the device's default stream — the same
        // one flame-core's BF16 ops launch their kernels on. We pass it as
        // the handle's "stream" so the event-gated Drop chain records on
        // compute, not on the copy stream. Eviction will then
        // cuStreamWaitEvent + cuEventSynchronize on a compute-stream event,
        // which guarantees no kernel still reads the slot when we unmap.
        let compute_stream_ptr = (*self.device.cu_stream()) as cuda_ffi::CUstream;

        // Gate copy_stream behind the prior consumer's compute-stream event
        // BEFORE ensure_resident bumps refcount — the event was just stored
        // by the synchronous handle Drop above.  Skipped when the slot was
        // Empty or Staging (no prior consumer).
        if needs_event_wait {
            self.arena
                .allocator
                .wait_for_last_use_event(
                    self.slab_id,
                    self.slot_regions[target],
                    copy_stream_ptr,
                )?;
        }

        let handle = self
            .arena
            .allocator
            .ensure_resident(self.slab_id, self.slot_regions[target], compute_stream_ptr)?;

        let slot_base = unsafe { handle.as_ptr() };

        let layout = &self.blocks[block_idx];

        // Bulk H2D: the pinned host_buffer is packed with all weights at the
        // same 16-byte-aligned offsets that the slot uses, with padding bytes
        // in between. One cuMemcpyHtoDAsync_v2 covering [0, total_bytes)
        // writes every weight at its correct device offset. Padding bytes are
        // copied too but never read — slot consumers index via
        // w.byte_offset + num_elems which excludes padding. Replaces a loop
        // of N per-weight memcpy calls with 1, eliminating ~15x host-side
        // driver overhead per block.
        //
        // SAFETY: host_base is pinned; slot is VMM-mapped with write access;
        // total_bytes <= slot.region.size by construction (slot sized to
        // max_block_bytes, and layout.total_bytes <= max_block_bytes).
        let host_base = layout.host_buffer.as_ptr() as *const c_void;
        let r = unsafe {
            cuda_ffi::cuMemcpyHtoDAsync_v2(
                slot_base,
                host_base,
                layout.total_bytes,
                copy_stream_ptr,
            )
        };
        if r != cuda_ffi::CUDA_SUCCESS {
            return Err(VmmError::CudaError(r));
        }

        // Record completion event on copy_stream for the consumer's
        // cuStreamWaitEvent gate in await_block.
        let mut event: CUevent = std::ptr::null_mut();
        // SAFETY: cuEventCreate writes to a valid local pointer.
        let er = unsafe { cuda_ffi::cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING) };
        if er != cuda_ffi::CUDA_SUCCESS {
            return Err(VmmError::CudaError(er));
        }
        // SAFETY: event is fresh, copy_stream_ptr is alive.
        let er = unsafe { cuda_ffi::cuEventRecord(event, copy_stream_ptr) };
        if er != cuda_ffi::CUDA_SUCCESS {
            unsafe { let _ = cuda_ffi::cuEventDestroy_v2(event); }
            return Err(VmmError::CudaError(er));
        }

        self.slots[target] = SlotState::Staging {
            block_idx,
            handle: Arc::new(handle),
            ready_event: event,
        };
        Ok(())
    }

    /// Await the prefetched block, build BF16View Tensors, return Arc<TurboBlock>.
    pub fn await_block(&mut self, block_idx: usize) -> Result<Arc<TurboBlock>, VmmError> {
        if block_idx >= self.blocks.len() {
            return Err(VmmError::InvalidRegion);
        }

        // First check both slots for a match. If Prepared, swap active and
        // return the existing Arc — a downstream forward loop calling
        // `await_block(N)` after `prefetch_block(N)` already-resolved hits
        // this path.
        for slot_idx in 0..2 {
            if self.slots[slot_idx].block_idx() != Some(block_idx) {
                continue;
            }
            if let SlotState::Prepared { ref block, .. } = self.slots[slot_idx] {
                self.active = slot_idx;
                return Ok(Arc::clone(block));
            }
            // Promote Staging → Prepared.
            let SlotState::Staging { handle, ready_event, .. } = self.slots[slot_idx].take() else {
                unreachable!("Staging branch already matched");
            };

            // Get default (compute) stream — this is what subsequent flame
            // kernels run on. cuStreamWaitEvent is GPU-side; no host sync.
            let compute_stream_ptr = self.device.cu_stream();
            // SAFETY: ready_event was recorded on copy_stream; compute stream
            // is valid for the lifetime of self.device.
            let er = unsafe {
                cuda_ffi::cuStreamWaitEvent(
                    *compute_stream_ptr as cuda_ffi::CUstream,
                    ready_event,
                    0,
                )
            };
            if er != cuda_ffi::CUDA_SUCCESS {
                unsafe { let _ = cuda_ffi::cuEventDestroy_v2(ready_event); }
                return Err(VmmError::CudaError(er));
            }
            // The event is now safely awaited by both copy_stream sequencing
            // and any compute kernels enqueued after this point.  Release.
            unsafe { let _ = cuda_ffi::cuEventDestroy_v2(ready_event); }

            let slot_base = unsafe { handle.as_ptr() };
            let layout = &self.blocks[block_idx];

            let mut weights: HashMap<String, Tensor> =
                HashMap::with_capacity(layout.weights.len());
            for (key, w) in layout.weights.iter() {
                let ptr = (slot_base + w.byte_offset as u64) as u64;
                // SAFETY: ptr is within the slot region currently mapped and
                // refcounted by `handle`. Tensor stays valid as long as the
                // handle (held in TurboBlock) is alive.
                let tensor = unsafe {
                    Tensor::from_bf16_device_ptr_non_owning(
                        ptr,
                        w.num_elems,
                        Shape::from_dims(&w.shape),
                        self.device.clone(),
                    )
                };
                weights.insert(key.clone(), tensor);
            }

            let block = Arc::new(TurboBlock { handle, weights });
            self.slots[slot_idx] = SlotState::Prepared {
                block_idx,
                block: Arc::clone(&block),
            };
            self.active = slot_idx;
            return Ok(block);
        }

        // Miss path: caller asked for a block that wasn't prefetched. Stage
        // it now (this also performs the H2D synchronously w.r.t. the
        // copy_stream) and re-call ourselves.
        self.prefetch_block(block_idx)?;
        self.await_block(block_idx)
    }
}

impl Drop for TurboBlockLoader {
    fn drop(&mut self) {
        // Release any in-flight ready_events; the residency handles in
        // Staging/Prepared get dropped by the slot replacement, which records
        // the eviction-gate event correctly.
        for i in 0..2 {
            if let SlotState::Staging { ready_event, .. } = self.slots[i].take() {
                unsafe { let _ = cuda_ffi::cuEventDestroy_v2(ready_event); }
            }
        }
        // Destroy the slab.  destroy_slab will refuse if any handle is still
        // alive — that's fine, the AllocatorInner Drop chain will clean up
        // after the last Arc<TurboBlock> dies.
        let _ = self.arena.allocator.destroy_slab(self.slab_id);
    }
}

// ---------------------------------------------------------------------------
// safetensors parsing → per-block CPU layout (BF16 packed)
// ---------------------------------------------------------------------------

fn build_block_layouts(
    path: &str,
    block_prefixes: &[String],
) -> anyhow::Result<Vec<BlockLayout>> {
    use serde_json::Value;

    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open {path}: {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| anyhow::anyhow!("mmap {path}: {e}"))?;
    if mmap.len() < 8 {
        anyhow::bail!("safetensors too small: {path}");
    }

    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_size;
    let data_start = header_end;
    if header_end > mmap.len() {
        anyhow::bail!("header runs past EOF in {path}");
    }

    let metadata: Value = serde_json::from_slice(&mmap[8..header_end])
        .map_err(|e| anyhow::anyhow!("safetensors header parse: {e}"))?;
    let metadata_obj = metadata
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors metadata not an object"))?;

    // Per-block accumulation: a Vec of (name, dtype, shape, num_elems, src_start, src_end).
    // dtype held as owned String so fused post-pass can move entries around
    // without borrow headaches on the JSON metadata's lifetime.
    let mut per_block: Vec<Vec<(String, String, Vec<usize>, usize, usize, usize)>> =
        (0..block_prefixes.len()).map(|_| Vec::new()).collect();

    for (name, info) in metadata_obj {
        if name == "__metadata__" { continue; }

        // classify by prefix index
        let block_idx = match block_prefixes.iter().position(|p| name.starts_with(p)) {
            Some(idx) => idx,
            None => continue,
        };

        let shape: Vec<usize> = match info["shape"].as_array() {
            Some(arr) => arr.iter().filter_map(|v| v.as_u64().map(|u| u as usize)).collect(),
            None => continue,
        };
        let num_elems: usize = shape.iter().product();
        if num_elems == 0 { continue; }

        let dtype_str: &str = info["dtype"].as_str().unwrap_or("F32");
        if !matches!(dtype_str, "BF16" | "F16" | "F32") { continue; }

        let offsets = match info["data_offsets"].as_array() {
            Some(arr) => arr,
            None => continue,
        };
        let start = data_start + offsets[0].as_u64().unwrap_or(0) as usize;
        let end = data_start + offsets[1].as_u64().unwrap_or(0) as usize;

        per_block[block_idx].push((name.clone(), dtype_str.to_string(), shape, num_elems, start, end));
    }

    let mut blocks = Vec::with_capacity(block_prefixes.len());
    for (block_idx, mut entries) in per_block.into_iter().enumerate() {
        // Sort by key for determinism (BF16-bit parity assumes the same
        // packed offsets across runs).
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // -- QKV fusion pass -----------------------------------------------
        //
        // The DiT hot path calls:
        //   q = linear(x, W_q); k = linear(x, W_k); v = linear(x, W_v)
        // 3 separate cuBLASLt GEMMs per attention per block. Concat the
        // weights along output-dim at load time to one [3*Cout, Cin] tensor
        // — then forward calls a single bigger GEMM and splits via the fused
        // qkv_split_permute_bf16 kernel. Applies to every attention with
        // separate Q/K/V weights: FLUX-lineage (attn.to_{q,k,v}),
        // cross-attention (attn.add_{q,k,v}_proj), single-stream
        // (attn.to_{q,k,v} after the `attn.` prefix already applied).
        //
        // Detect triples by key suffix, synthesize a fused entry with
        // `parts: [q_range, k_range, v_range]` that the packer concatenates
        // along dim 0. Originals are dropped — forward code is responsible
        // for calling the fused name.
        let entries_with_parts = fuse_qkv_entries(entries);
        let entries = entries_with_parts;

        let mut layout_map: HashMap<String, WeightLayout> = HashMap::with_capacity(entries.len());
        let mut byte_offset = 0usize;
        // First pass: assign offsets (16-byte aligned per weight to keep
        // BF16 access well-aligned for cuBLASLt downstream).
        let mut packed_entries: Vec<(String, String, Vec<usize>, usize, Vec<(usize, usize, usize)>, usize)> =
            Vec::with_capacity(entries.len());
        for (name, dtype, shape, num_elems, parts) in entries {
            byte_offset = (byte_offset + 15) & !15;
            let this_offset = byte_offset;
            byte_offset += num_elems * 2;
            packed_entries.push((name, dtype, shape, num_elems, parts, this_offset));
        }

        let total_bytes = byte_offset;
        // Allocate the pinned host slab once per block. We size it to total_bytes / 2
        // u16 elements (rounded up by 1 to avoid zero-sized alloc on empty
        // blocks).
        let cap = (total_bytes / 2).max(1);
        let mut host_buffer =
            PinnedHostBuffer::<u16>::with_capacity_elems(cap, PinnedAllocFlags::default())
                .map_err(|e| anyhow::anyhow!("pinned alloc for block {block_idx}: {e}"))?;
        // Initialize the visible region as filled — slot offsets only touch
        // the bytes we explicitly write below, so the gap bytes are
        // arbitrary, but a clean SetLen keeps `as_slice()` semantics sane.
        unsafe { host_buffer.set_len(cap); }

        {
            let dst = host_buffer.as_mut_slice();
            for (name, dtype, shape, num_elems, parts, this_offset) in packed_entries {
                // Pack parts sequentially into the destination window. For a
                // regular weight, `parts` has one entry; for a fused QKV,
                // three (Q then K then V, along output-dim axis).
                let mut cursor = this_offset / 2;
                for (start, end, part_num_elems) in &parts {
                    let raw = &mmap[*start..*end];
                    match dtype.as_str() {
                        "BF16" => {
                            for (i, chunk) in raw.chunks_exact(2).enumerate().take(*part_num_elems) {
                                dst[cursor + i] = u16::from_le_bytes([chunk[0], chunk[1]]);
                            }
                        }
                        "F16" => {
                            for (i, chunk) in raw.chunks_exact(2).enumerate().take(*part_num_elems) {
                                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                let f = f16_to_f32(bits);
                                dst[cursor + i] = f32_to_bf16(f);
                            }
                        }
                        "F32" => {
                            for (i, chunk) in raw.chunks_exact(4).enumerate().take(*part_num_elems) {
                                let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                                dst[cursor + i] = f32_to_bf16(f);
                            }
                        }
                        _ => unreachable!(),
                    }
                    cursor += part_num_elems;
                }
                layout_map.insert(
                    name,
                    WeightLayout {
                        byte_offset: this_offset,
                        num_elems,
                        shape,
                    },
                );
            }
        }

        if layout_map.is_empty() {
            log::warn!("[turbo loader] block {block_idx} had no matching weights");
        }

        blocks.push(BlockLayout {
            total_bytes,
            weights: layout_map,
            host_buffer,
        });
    }

    Ok(blocks)
}

/// QKV-triple detection + fusion.
///
/// Accepts the raw per-block entry list from the safetensors parse and
/// returns a new list where Q/K/V weight triples (and their biases) sharing
/// the same attention prefix are replaced by a single fused entry of shape
/// `[3*Cout, Cin]` (weights) or `[3*Cout]` (bias). The three original
/// entries are dropped — forward-side code must call the fused name.
///
/// Patterns handled (prefix up to and including the last `.`):
///   `<p>.to_q.weight`, `<p>.to_k.weight`, `<p>.to_v.weight`         → `<p>.to_qkv.weight`
///   `<p>.to_q.bias`,   `<p>.to_k.bias`,   `<p>.to_v.bias`           → `<p>.to_qkv.bias`
///   `<p>.add_q_proj.weight` + k + v                                 → `<p>.add_qkv_proj.weight`
///   (same for bias)
///   `<p>.q.weight`, `<p>.k.weight`, `<p>.v.weight`                  → `<p>.qkv.weight`
///   (legacy FLUX-style)
fn fuse_qkv_entries(
    entries: Vec<(String, String, Vec<usize>, usize, usize, usize)>,
) -> Vec<(String, String, Vec<usize>, usize, Vec<(usize, usize, usize)>)> {
    // First, lift each raw entry to the new shape (name, dtype, shape,
    // num_elems, parts=[single (start, end, num_elems)]).
    let lifted: Vec<(String, String, Vec<usize>, usize, Vec<(usize, usize, usize)>)> =
        entries
            .into_iter()
            .map(|(n, dt, sh, ne, s, e)| (n, dt, sh, ne, vec![(s, e, ne)]))
            .collect();

    // Triple patterns: (q_suffix, k_suffix, v_suffix, fused_suffix)
    // Applied to both `.weight` and `.bias` variants.
    let triples = [
        ("to_q", "to_k", "to_v", "to_qkv"),
        ("add_q_proj", "add_k_proj", "add_v_proj", "add_qkv_proj"),
        ("q", "k", "v", "qkv"),
    ];

    let mut fused_out: Vec<(String, String, Vec<usize>, usize, Vec<(usize, usize, usize)>)> =
        Vec::with_capacity(lifted.len());
    let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // For each (q-suffix, weight-or-bias) index-find the triple.
    for (i, (name_i, _, _, _, _)) in lifted.iter().enumerate() {
        if consumed.contains(&i) { continue; }
        let name = name_i.as_str();

        // Try each triple pattern × {weight, bias}
        let mut fused_ok = false;
        for (qs, ks, vs, fused_suffix) in &triples {
            for tail in &[".weight", ".bias"] {
                let q_full = format!("{qs}{tail}");
                if !name.ends_with(&q_full) { continue; }
                // Extract prefix: everything before q_full
                let prefix_end = name.len() - q_full.len();
                let prefix = &name[..prefix_end];
                let k_name = format!("{prefix}{ks}{tail}");
                let v_name = format!("{prefix}{vs}{tail}");
                let fused_name = format!("{prefix}{fused_suffix}{tail}");

                // Locate k and v in lifted by name.
                let k_idx = lifted.iter().enumerate()
                    .find(|(j, (nm, _, _, _, _))| !consumed.contains(j) && nm == &k_name)
                    .map(|(j, _)| j);
                let v_idx = lifted.iter().enumerate()
                    .find(|(j, (nm, _, _, _, _))| !consumed.contains(j) && nm == &v_name)
                    .map(|(j, _)| j);

                if let (Some(ki), Some(vi)) = (k_idx, v_idx) {
                    // Validate: same dtype, compatible shapes (bias: [C],
                    // weight: [Cout, Cin] with identical Cin).
                    let (_, q_dt, q_sh, q_ne, q_parts) = &lifted[i];
                    let (_, k_dt, k_sh, k_ne, k_parts) = &lifted[ki];
                    let (_, v_dt, v_sh, v_ne, v_parts) = &lifted[vi];
                    if q_dt != k_dt || q_dt != v_dt {
                        break;
                    }
                    // Fused shape: append along dim 0.
                    let mut fused_shape = q_sh.clone();
                    if fused_shape.is_empty() {
                        break;
                    }
                    if q_sh.len() != k_sh.len() || q_sh.len() != v_sh.len() {
                        break;
                    }
                    // Tail dims must match
                    if q_sh[1..] != k_sh[1..] || q_sh[1..] != v_sh[1..] {
                        break;
                    }
                    fused_shape[0] = q_sh[0] + k_sh[0] + v_sh[0];
                    let fused_ne = q_ne + k_ne + v_ne;
                    let mut fused_parts: Vec<(usize, usize, usize)> =
                        Vec::with_capacity(q_parts.len() + k_parts.len() + v_parts.len());
                    fused_parts.extend(q_parts.iter().copied());
                    fused_parts.extend(k_parts.iter().copied());
                    fused_parts.extend(v_parts.iter().copied());

                    let dt = q_dt.clone();
                    fused_out.push((fused_name, dt, fused_shape, fused_ne, fused_parts));
                    // Intentionally do NOT mark originals as consumed — they
                    // are retained in the final layout so forward code that
                    // still references `to_q`/`to_k`/`to_v` individually keeps
                    // working. Forward code that wants the fast path looks up
                    // the synthesized `to_qkv` entry instead. A future pass
                    // can drop the originals once every forward migrates.
                    fused_ok = true;
                    break;
                }
            }
            if fused_ok { break; }
        }
    }

    // Keep any entry not consumed by a triple.
    for (i, ent) in lifted.into_iter().enumerate() {
        if !consumed.contains(&i) {
            fused_out.push(ent);
        }
    }
    fused_out.sort_by(|a, b| a.0.cmp(&b.0));
    fused_out
}

// ---------------------------------------------------------------------------
// f16/bf16 helpers (copied from flame-diffusion::block_offload to avoid a
// public-API dependency)
// ---------------------------------------------------------------------------

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13));
    }
    if exp == 0x1F {
        if frac == 0 { return f32::from_bits((sign << 31) | (0xFF << 23)); }
        return f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
}

#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}
