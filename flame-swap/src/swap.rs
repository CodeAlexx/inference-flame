//! On-demand block swapper for FLAME inference.
//!
//! Reads block weights from mmap, dequants FP8→BF16 using multiple CPU threads,
//! copies to pinned buffer, then async DMA to GPU. Double-buffered so GPU compute
//! overlaps with the next block's CPU dequant + DMA.
//!
//! NO pre-caching — dequant happens per-block on demand. Uses 8 threads for
//! parallel dequant so FP8→BF16 conversion is fast (~50-100ms per block).
//!
//! Peak CPU RAM: ~4GB (two pinned staging buffers only).
//!
//! # Usage
//! ```ignore
//! let mut swap = FlameSwap::load(&["model.safetensors"], &device, block_fn)?;
//! swap.prefetch(0)?;
//! for i in 0..swap.num_blocks() {
//!     let weights = swap.await_block(i)?;
//!     if i + 1 < swap.num_blocks() { swap.prefetch(i + 1)?; }
//!     x = block_forward(&x, &weights);
//! }
//! ```

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use memmap2::Mmap;

use flame_core::pinned::{PinnedAllocFlags, PinnedHostBuffer};
use flame_core::Shape;
use flame_core::tensor::Tensor;

use crate::ffi::{self, Event, Stream};

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum SourceDtype {
    BF16,
    F32,
    F8E4M3 { scale: f32 },
}

#[derive(Debug, Clone)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    numel: usize,
    file_offset: usize,
    file_idx: usize,
    src_dtype: SourceDtype,
}

#[derive(Debug)]
struct BlockMeta {
    tensors: Vec<TensorMeta>,
    /// Total bytes written to staging for this block (sum of raw_bytes per tensor).
    staging_bytes: usize,
    /// Total bytes the gpu_buf must hold for this block (final BF16 region
    /// for every tensor + a transient raw region for FP8 tensors).
    gpu_bytes: usize,
}

#[derive(Debug, Clone)]
struct StagingLayout {
    /// Byte offset into the pinned staging buffer for this tensor's raw data.
    offset: usize,
    /// Bytes copied from the safetensors mmap into the pinned buffer.
    /// BF16/F32 tensors use their full byte length; FP8 tensors copy `numel`
    /// bytes (one per element).
    raw_bytes: usize,
}

/// Pre-computed layout for a tensor inside a pre-allocated GPU buffer.
///
/// `final_offset` always names the BF16 destination region the kernel/caller
/// will read from.  `raw_offset` names the staging region the H2D copies into:
/// for BF16 tensors it equals `final_offset` (DMA writes directly to the
/// final region); for FP8 tensors it points into a separate transient region
/// past the end of the BF16 area, which the dequant kernel reads from.
#[derive(Debug, Clone)]
struct GpuTensorLayout {
    final_offset: usize,    // byte offset of BF16 output in gpu_buf[slot]
    final_numel: usize,     // BF16 element count
    raw_offset: usize,      // byte offset of raw H2D landing zone
    raw_bytes: usize,       // bytes copied from staging to raw_offset
    src_dtype: SourceDtype, // dtype of the raw bytes
}

// ---------------------------------------------------------------------------
// Slot state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Idle,
    StagingRequested,
    Staged,
    Transferring,
    Ready,
    InCompute,
}

// ---------------------------------------------------------------------------
// Pending DMA
// ---------------------------------------------------------------------------

/// In Phase 3 prefetch only registers a staging request and remembers which
/// (block, slot) pair await_block should pick up.  No DMA happens until
/// await_block.
#[derive(Clone, Copy)]
struct PendingRequest {
    block_idx: usize,
    slot: Slot,
}

// ---------------------------------------------------------------------------
// Double-buffer slot
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot { A = 0, B = 1 }

impl Slot {
    fn flip(self) -> Self { match self { Slot::A => Slot::B, Slot::B => Slot::A } }
    fn idx(self) -> usize { self as usize }
}

// ---------------------------------------------------------------------------
// Staging thread communication
// ---------------------------------------------------------------------------

struct StagingState {
    request: Option<(usize, Slot)>,
    complete: Option<(usize, Slot)>,
    shutdown: bool,
}

struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}

/// Bytes one tensor occupies in the safetensors mmap (and therefore in the
/// pinned staging buffer once raw-copied).
fn raw_byte_size(t: &TensorMeta) -> usize {
    match t.src_dtype {
        SourceDtype::BF16 => t.numel * 2,
        SourceDtype::F32 => t.numel * 4,
        SourceDtype::F8E4M3 { .. } => t.numel, // 1 byte per element
    }
}

// ---------------------------------------------------------------------------
// FlameSwap
// ---------------------------------------------------------------------------

pub struct FlameSwap {
    mmaps: Arc<Vec<Mmap>>,
    blocks: Vec<BlockMeta>,
    layouts: Vec<Vec<StagingLayout>>,
    gpu_layouts: Vec<Vec<GpuTensorLayout>>,
    staging: [PinnedHostBuffer<u8>; 2],
    gpu_buf: [CudaSlice<u8>; 2],
    slot_state: [SlotState; 2],
    max_staging_bytes: usize,
    max_gpu_bytes: usize,
    transfer: Stream,
    /// Per-slot "data is ready on GPU" event.  Recorded on the transfer
    /// stream after DMA + dequant; the default stream waits on it before
    /// caller-issued compute kernels run.
    ready_event: [Event; 2],
    /// Per-slot "compute on this slot's data has been submitted to default
    /// stream" event.  Recorded on the default stream by the next prefetch
    /// that wants to reuse the slot, then the transfer stream waits on it
    /// before overwriting the slot.
    done_event: [Event; 2],
    pending: Option<PendingRequest>,
    next_slot: Slot,
    state: Arc<(Mutex<StagingState>, Condvar)>,
    stage_thread: Option<thread::JoinHandle<()>>,
    device: Arc<CudaDevice>,
}

impl FlameSwap {
    pub fn load<P, F>(
        paths: &[P],
        device: &Arc<CudaDevice>,
        block_fn: F,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        P: AsRef<Path>,
        F: Fn(&str) -> Option<usize>,
    {
        let mut all_tensors: HashMap<usize, Vec<TensorMeta>> = HashMap::new();

        let mmaps: Vec<Mmap> = paths
            .iter()
            .map(|p| {
                let f = std::fs::File::open(p)?;
                unsafe { Mmap::map(&f) }.map_err(Into::into)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

        for (file_idx, mmap) in mmaps.iter().enumerate() {
            let header_entries = parse_safetensors_header(mmap)?;

            let mut scale_map: HashMap<String, f32> = HashMap::new();
            for entry in &header_entries {
                if entry.name.ends_with("_scale") && entry.shape.is_empty() {
                    let bytes = &mmap[entry.data_offset..entry.data_offset + 4];
                    let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    let target = entry.name[..entry.name.len() - 6].to_string();
                    scale_map.insert(target, scale);
                }
            }

            for entry in &header_entries {
                if let Some(block_idx) = block_fn(&entry.name) {
                    let numel: usize = entry.shape.iter().product();
                    let src_dtype = match entry.src_dtype {
                        SourceDtype::F8E4M3 { .. } => {
                            let scale = scale_map.get(&entry.name).copied().unwrap_or(1.0);
                            SourceDtype::F8E4M3 { scale }
                        }
                        other => other,
                    };
                    all_tensors.entry(block_idx).or_default().push(TensorMeta {
                        name: entry.name.clone(),
                        shape: entry.shape.clone(),
                        numel,
                        file_offset: entry.data_offset,
                        file_idx,
                        src_dtype,
                    });
                }
            }
        }

        let mut block_indices: Vec<usize> = all_tensors.keys().copied().collect();
        block_indices.sort();
        let num_blocks = block_indices.last().map(|m| m + 1).unwrap_or(0);

        let mut blocks: Vec<BlockMeta> = (0..num_blocks)
            .map(|_| BlockMeta { tensors: Vec::new(), staging_bytes: 0, gpu_bytes: 0 })
            .collect();

        for idx in &block_indices {
            let tensors = all_tensors.remove(idx).unwrap();
            let mut staging_bytes = 0usize;
            let mut final_bytes = 0usize;
            let mut fp8_bytes = 0usize;
            for t in &tensors {
                let raw = raw_byte_size(t);
                staging_bytes += raw;
                final_bytes += t.numel * 2;
                if matches!(t.src_dtype, SourceDtype::F8E4M3 { .. }) {
                    fp8_bytes += t.numel; // transient raw region for dequant
                }
            }
            blocks[*idx] = BlockMeta {
                tensors,
                staging_bytes,
                gpu_bytes: final_bytes + fp8_bytes,
            };
        }

        // Per-tensor staging layout: tensors packed sequentially at byte
        // offsets in the pinned buffer.
        let layouts: Vec<Vec<StagingLayout>> = blocks
            .iter()
            .map(|block| {
                let mut offset = 0usize;
                block.tensors.iter().map(|t| {
                    let raw = raw_byte_size(t);
                    let layout = StagingLayout { offset, raw_bytes: raw };
                    offset += raw;
                    layout
                }).collect()
            })
            .collect();

        // GPU layout: pack every tensor's BF16 final region sequentially in
        // gpu_buf, starting at offset 0.  Then any FP8 tensors get a transient
        // raw landing zone packed AFTER the final region.  BF16/F32 tensors
        // DMA directly into their final region (raw_offset == final_offset).
        let gpu_layouts: Vec<Vec<GpuTensorLayout>> = blocks
            .iter()
            .map(|block| {
                let final_total: usize = block.tensors.iter().map(|t| t.numel * 2).sum();
                let mut final_off = 0usize;
                let mut raw_off = final_total; // FP8 raw region starts here
                let mut out = Vec::with_capacity(block.tensors.len());
                for t in &block.tensors {
                    let bf16_bytes = t.numel * 2;
                    let (this_raw_off, this_raw_bytes) = match t.src_dtype {
                        SourceDtype::BF16 => (final_off, bf16_bytes),
                        SourceDtype::F32 => (final_off, t.numel * 4),
                        SourceDtype::F8E4M3 { .. } => {
                            let off = raw_off;
                            raw_off += t.numel; // 1 byte per FP8 element
                            (off, t.numel)
                        }
                    };
                    out.push(GpuTensorLayout {
                        final_offset: final_off,
                        final_numel: t.numel,
                        raw_offset: this_raw_off,
                        raw_bytes: this_raw_bytes,
                        src_dtype: t.src_dtype,
                    });
                    final_off += bf16_bytes;
                }
                out
            })
            .collect();

        let mmaps = Arc::new(mmaps);

        let max_staging_bytes = blocks.iter().map(|b| b.staging_bytes).max().unwrap_or(1);
        let max_gpu_bytes = blocks.iter().map(|b| b.gpu_bytes).max().unwrap_or(1);

        let staging_a = PinnedHostBuffer::<u8>::with_capacity_elems(max_staging_bytes, PinnedAllocFlags::DEFAULT)?;
        let staging_b = PinnedHostBuffer::<u8>::with_capacity_elems(max_staging_bytes, PinnedAllocFlags::DEFAULT)?;

        // Pre-allocate two GPU staging buffers, one per slot. After load(), no
        // per-block device.alloc happens in the hot path.
        let gpu_buf_a: CudaSlice<u8> = unsafe { device.alloc::<u8>(max_gpu_bytes)? };
        let gpu_buf_b: CudaSlice<u8> = unsafe { device.alloc::<u8>(max_gpu_bytes)? };

        eprintln!("[FlameSwap] {} blocks, max staging {:.1}MB, max gpu {:.1}MB, pinned total {:.1}MB, GPU staging total {:.1}MB",
            num_blocks,
            max_staging_bytes as f64 / 1e6,
            max_gpu_bytes as f64 / 1e6,
            (max_staging_bytes * 2) as f64 / 1e6,
            (max_gpu_bytes * 2) as f64 / 1e6);

        let state = Arc::new((
            Mutex::new(StagingState { request: None, complete: None, shutdown: false }),
            Condvar::new(),
        ));

        let ptrs = [
            SendPtr(staging_a.as_ptr() as *mut u8),
            SendPtr(staging_b.as_ptr() as *mut u8),
        ];
        let thread_mmaps = Arc::clone(&mmaps);
        let thread_blocks: Vec<Vec<TensorMeta>> = blocks.iter().map(|b| b.tensors.clone()).collect();
        let thread_layouts = layouts.clone();
        let thread_state = Arc::clone(&state);

        let stage_thread = thread::spawn(move || {
            staging_thread_main(
                thread_mmaps, thread_blocks, thread_layouts,
                [ptrs[0].0, ptrs[1].0], thread_state,
            );
        });

        let transfer = Stream::new()?;
        let ready_a = Event::new()?;
        let ready_b = Event::new()?;
        let done_a = Event::new()?;
        let done_b = Event::new()?;

        Ok(Self {
            mmaps,
            blocks,
            layouts,
            gpu_layouts,
            staging: [staging_a, staging_b],
            gpu_buf: [gpu_buf_a, gpu_buf_b],
            slot_state: [SlotState::Idle, SlotState::Idle],
            max_staging_bytes,
            max_gpu_bytes,
            transfer,
            ready_event: [ready_a, ready_b],
            done_event: [done_a, done_b],
            pending: None,
            next_slot: Slot::A,
            state,
            stage_thread: Some(stage_thread),
            device: Arc::clone(device),
        })
    }

    pub fn num_blocks(&self) -> usize { self.blocks.len() }

    /// Submit a staging request for `idx` and return immediately.  No DMA,
    /// no kernel launches.  Slot reuse hazards are guarded by `done_event`:
    /// if the chosen slot is still flagged InCompute from the previous
    /// round, we record the done_event on the default stream (capturing all
    /// caller-issued kernels submitted up to now) and tell the transfer
    /// stream to wait on it before any subsequent DMA fires.
    pub fn prefetch(&mut self, idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        assert!(idx < self.blocks.len(), "block index {idx} out of range");
        assert!(
            self.pending.is_none(),
            "prefetch({idx}) called before await_block consumed the previous prefetch"
        );

        let slot = self.next_slot;
        let prior = self.slot_state[slot.idx()];
        assert!(
            prior == SlotState::Idle || prior == SlotState::InCompute,
            "slot {slot:?} not reusable: state = {prior:?}",
        );

        if prior == SlotState::InCompute {
            // Snapshot the default stream's progress and gate the transfer
            // stream behind it so the upcoming H2D + dequant cannot stomp on
            // memory the caller is still reading from.
            self.done_event[slot.idx()].record_default()?;
            self.transfer.wait_event(&self.done_event[slot.idx()])?;
        }
        self.slot_state[slot.idx()] = SlotState::StagingRequested;

        self.request_staging(idx, slot);

        self.pending = Some(PendingRequest { block_idx: idx, slot });
        self.next_slot = slot.flip();
        Ok(())
    }

    /// Wait for the staging copy to land in the pinned buffer, then issue
    /// the H2D copies + FP8 dequant kernels on the transfer stream, gate the
    /// default stream behind a ready_event, and return non-owning Tensor
    /// views into gpu_buf[slot].  No host-side `cudaStreamSynchronize` is
    /// performed — the caller's subsequent default-stream kernels (and any
    /// to_vec/D2H copies) will automatically wait on ready_event.
    pub fn await_block(&mut self, idx: usize) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        let req = self.pending.take()
            .ok_or("await_block called without a prior prefetch")?;
        assert_eq!(req.block_idx, idx);
        let slot = req.slot;
        debug_assert_eq!(self.slot_state[slot.idx()], SlotState::StagingRequested);

        self.wait_staging(idx, slot)?;
        self.slot_state[slot.idx()] = SlotState::Staged;

        let block = &self.blocks[idx];
        let layout = &self.layouts[idx];
        let gpu_layout = &self.gpu_layouts[idx];

        self.slot_state[slot.idx()] = SlotState::Transferring;
        let gpu_base = *self.gpu_buf[slot.idx()].device_ptr();
        let host_base = self.staging[slot.idx()].as_ptr();
        let transfer_raw = self.transfer.as_raw();
        for ((t_meta, sl), gl) in block.tensors.iter().zip(layout.iter()).zip(gpu_layout.iter()) {
            debug_assert_eq!(sl.raw_bytes, gl.raw_bytes);
            unsafe {
                let dst = (gpu_base + gl.raw_offset as u64) as *mut c_void;
                let src = host_base.add(sl.offset) as *const c_void;
                ffi::flame_cuda_memcpy_async(dst, src, gl.raw_bytes, 1, transfer_raw);
            }
            if let SourceDtype::F8E4M3 { scale } = gl.src_dtype {
                let raw_ptr = (gpu_base + gl.raw_offset as u64) as *const c_void;
                let final_ptr = (gpu_base + gl.final_offset as u64) as *mut c_void;
                let ret = unsafe {
                    ffi::flame_fp8_to_bf16(
                        raw_ptr,
                        final_ptr,
                        scale,
                        gl.final_numel,
                        transfer_raw,
                    )
                };
                if ret != 0 {
                    return Err(format!("flame_fp8_to_bf16 failed for {} ({})", t_meta.name, ret).into());
                }
            }
        }

        // Publish "data ready" so the default stream waits before reading.
        self.ready_event[slot.idx()].record(&self.transfer)?;
        ffi::default_stream_wait_event(&self.ready_event[slot.idx()])?;
        self.slot_state[slot.idx()] = SlotState::Ready;

        // Build non-owning Tensor views over the gpu_buf region.
        let mut weights = HashMap::with_capacity(block.tensors.len());
        for (t_meta, gl) in block.tensors.iter().zip(gpu_layout.iter()) {
            let shape = Shape::new(t_meta.shape.clone());
            let ptr = (gpu_base + gl.final_offset as u64) as *mut u16;
            let tensor = unsafe {
                Tensor::view_from_buffer(ptr, shape, Arc::clone(&self.device))
            };
            weights.insert(t_meta.name.clone(), tensor);
        }
        self.slot_state[slot.idx()] = SlotState::InCompute;
        Ok(weights)
    }

    pub fn load_block(&mut self, idx: usize) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        self.prefetch(idx)?;
        self.await_block(idx)
    }

    pub fn pinned_bytes(&self) -> usize {
        self.max_staging_bytes * 2
    }

    fn request_staging(&self, block_idx: usize, slot: Slot) {
        let (lock, cvar) = &*self.state;
        let mut guard = lock.lock().unwrap();
        guard.request = Some((block_idx, slot));
        cvar.notify_one();
    }

    fn wait_staging(&self, block_idx: usize, slot: Slot) -> Result<(), Box<dyn std::error::Error>> {
        let (lock, cvar) = &*self.state;
        let mut guard = lock.lock().unwrap();
        loop {
            if let Some((bidx, bslot)) = guard.complete {
                if bidx == block_idx && bslot == slot {
                    guard.complete = None;
                    return Ok(());
                }
            }
            guard = cvar.wait(guard).unwrap();
        }
    }
}

impl Drop for FlameSwap {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.state;
            let mut guard = lock.lock().unwrap();
            guard.shutdown = true;
            cvar.notify_one();
        }
        if let Some(handle) = self.stage_thread.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// CPU staging thread — multi-threaded dequant from mmap
// ---------------------------------------------------------------------------

fn staging_thread_main(
    mmaps: Arc<Vec<Mmap>>,
    block_tensors: Vec<Vec<TensorMeta>>,
    layouts: Vec<Vec<StagingLayout>>,
    staging_ptrs: [*mut u8; 2],
    state: Arc<(Mutex<StagingState>, Condvar)>,
) {
    let (lock, cvar) = &*state;
    loop {
        let mut guard = lock.lock().unwrap();
        while guard.request.is_none() && !guard.shutdown {
            guard = cvar.wait(guard).unwrap();
        }
        if guard.shutdown { return; }
        let (block_idx, slot) = guard.request.take().unwrap();
        drop(guard);

        let dst_base = staging_ptrs[slot.idx()];
        let tensors = &block_tensors[block_idx];
        let layout = &layouts[block_idx];

        // Pure raw memcpy from mmap into pinned staging — no CPU dequant.
        // FP8 → BF16 conversion happens on GPU after the H2D transfer.
        for (t, sl) in tensors.iter().zip(layout.iter()) {
            let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + sl.raw_bytes];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    dst_base.add(sl.offset),
                    sl.raw_bytes,
                );
            }
        }

        let mut guard = lock.lock().unwrap();
        guard.complete = Some((block_idx, slot));
        cvar.notify_one();
    }
}

// ---------------------------------------------------------------------------
// Safetensors header parsing
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct HeaderEntry {
    name: String,
    shape: Vec<usize>,
    data_offset: usize,
    src_dtype: SourceDtype,
}

fn parse_safetensors_header(mmap: &[u8]) -> Result<Vec<HeaderEntry>, Box<dyn std::error::Error>> {
    if mmap.len() < 8 { return Err("file too small for safetensors".into()); }
    let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
    if mmap.len() < 8 + header_len { return Err("truncated safetensors header".into()); }

    let header_str = std::str::from_utf8(&mmap[8..8 + header_len])?;
    let data_start = 8 + header_len;
    let mut entries = Vec::new();
    let mut pos = 0;
    let bytes = header_str.as_bytes();

    while pos < bytes.len() {
        let Some(key_start) = find_char(bytes, b'"', pos) else { break };
        let Some(key_end) = find_char(bytes, b'"', key_start + 1) else { break };
        let key = &header_str[key_start + 1..key_end];
        pos = key_end + 1;

        if key == "__metadata__" {
            if let Some(obj_start) = find_char(bytes, b'{', pos) {
                pos = skip_object(bytes, obj_start);
            }
            continue;
        }

        let Some(obj_start) = find_char(bytes, b'{', pos) else { break };
        let obj_end = skip_object(bytes, obj_start);
        let obj_str = &header_str[obj_start..obj_end];
        pos = obj_end;

        let dtype = extract_string_field(obj_str, "dtype").unwrap_or_default();
        let src_dtype = match dtype.as_str() {
            "BF16" => SourceDtype::BF16,
            "F32" => SourceDtype::F32,
            "F8_E4M3" => SourceDtype::F8E4M3 { scale: 1.0 },
            _ => continue,
        };

        let shape = extract_array_field(obj_str, "shape").unwrap_or_default();
        let offsets = extract_array_field(obj_str, "data_offsets").unwrap_or_default();
        if offsets.len() != 2 { continue; }

        entries.push(HeaderEntry {
            name: key.to_string(), shape,
            data_offset: data_start + offsets[0], src_dtype,
        });
    }
    Ok(entries)
}

fn find_char(bytes: &[u8], ch: u8, from: usize) -> Option<usize> {
    bytes[from..].iter().position(|&b| b == ch).map(|p| p + from)
}

fn skip_object(bytes: &[u8], pos: usize) -> usize {
    let mut depth = 0i32;
    let mut i = pos;
    let mut in_string = false;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_string => { i += 1; }
            b'"' => { in_string = !in_string; }
            b'{' if !in_string => { depth += 1; }
            b'}' if !in_string => { depth -= 1; if depth == 0 { return i + 1; } }
            _ => {}
        }
        i += 1;
    }
    bytes.len()
}

fn extract_string_field(obj: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\"", field);
    let idx = obj.find(&pattern)?;
    let rest = &obj[idx + pattern.len()..];
    let q1 = rest.find('"')? + 1;
    let q2 = rest[q1..].find('"')?;
    Some(rest[q1..q1 + q2].to_string())
}

fn extract_array_field(obj: &str, field: &str) -> Option<Vec<usize>> {
    let pattern = format!("\"{}\"", field);
    let idx = obj.find(&pattern)?;
    let rest = &obj[idx + pattern.len()..];
    let bracket_start = rest.find('[')? + 1;
    let bracket_end = rest[bracket_start..].find(']')?;
    let inner = &rest[bracket_start..bracket_start + bracket_end];
    Some(inner.split(',').filter_map(|s| s.trim().parse().ok()).collect())
}
