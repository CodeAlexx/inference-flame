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

/// Convert a single FP8 E4M3 byte to BF16 bits.
#[inline]
fn fp8_e4m3_to_bf16(bits: u8, scale: f32) -> u16 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;

    if exp == 0 && mant == 0 {
        return if sign == 1 { 0x8000 } else { 0 };
    }
    if exp == 0xF && mant == 0x7 {
        return 0x7FC0; // NaN
    }

    let (effective_exp, effective_mant) = if exp == 0 {
        (-6i32, mant as f32 / 8.0)
    } else {
        (exp as i32 - 7, 1.0 + mant as f32 / 8.0)
    };

    let magnitude = effective_mant * (2.0f32).powi(effective_exp);
    let f = if sign == 1 { -magnitude } else { magnitude } * scale;
    half::bf16::from_f32(f).to_bits()
}

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
    total_u16s: usize,
}

#[derive(Debug, Clone)]
struct StagingLayout {
    offset: usize,
    numel: usize,
}

/// Pre-computed layout for a tensor inside a pre-allocated GPU buffer.
///
/// Phase 1: every output tensor is BF16, so `final_offset` is byte-addressed
/// into `gpu_buf[slot]` and `numel * 2` bytes are written there.
#[derive(Debug, Clone)]
struct GpuTensorLayout {
    final_offset: usize, // byte offset into gpu_buf[slot]
    numel: usize,        // BF16 element count
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

struct PendingTensor {
    name: String,
    offset: usize,    // byte offset into gpu_buf[slot]
    numel: usize,     // BF16 elements
    shape: Vec<usize>,
}

struct Pending {
    block_idx: usize,
    slot: Slot,
    tensors: Vec<PendingTensor>,
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

struct SendPtr(*mut u16);
unsafe impl Send for SendPtr {}

// ---------------------------------------------------------------------------
// FlameSwap
// ---------------------------------------------------------------------------

pub struct FlameSwap {
    mmaps: Arc<Vec<Mmap>>,
    blocks: Vec<BlockMeta>,
    layouts: Vec<Vec<StagingLayout>>,
    gpu_layouts: Vec<Vec<GpuTensorLayout>>,
    staging: [PinnedHostBuffer<u16>; 2],
    gpu_buf: [CudaSlice<u8>; 2],
    slot_state: [SlotState; 2],
    max_block_bytes: usize,
    transfer: Stream,
    event: Event,
    pending: Option<Pending>,
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
            .map(|_| BlockMeta { tensors: Vec::new(), total_u16s: 0 })
            .collect();

        for idx in &block_indices {
            let tensors = all_tensors.remove(idx).unwrap();
            let total_u16s: usize = tensors.iter().map(|t| t.numel).sum();
            blocks[*idx] = BlockMeta { tensors, total_u16s };
        }

        let layouts: Vec<Vec<StagingLayout>> = blocks
            .iter()
            .map(|block| {
                let mut offset = 0usize;
                block.tensors.iter().map(|t| {
                    let layout = StagingLayout { offset, numel: t.numel };
                    offset += t.numel;
                    layout
                }).collect()
            })
            .collect();

        // Phase 1: every output tensor is BF16 (= 2 bytes per element).
        // Pack tensors sequentially in gpu_buf at byte offsets.
        let gpu_layouts: Vec<Vec<GpuTensorLayout>> = blocks
            .iter()
            .map(|block| {
                let mut offset_bytes = 0usize;
                block.tensors.iter().map(|t| {
                    let layout = GpuTensorLayout { final_offset: offset_bytes, numel: t.numel };
                    offset_bytes += t.numel * 2;
                    layout
                }).collect()
            })
            .collect();

        let mmaps = Arc::new(mmaps);

        let max_u16s = blocks.iter().map(|b| b.total_u16s).max().unwrap_or(1);
        let max_block_bytes = max_u16s * 2;
        let staging_a = PinnedHostBuffer::<u16>::with_capacity_elems(max_u16s, PinnedAllocFlags::DEFAULT)?;
        let staging_b = PinnedHostBuffer::<u16>::with_capacity_elems(max_u16s, PinnedAllocFlags::DEFAULT)?;

        // Pre-allocate two GPU staging buffers, one per slot. After load(), no
        // per-block device.alloc happens in the hot path.
        let gpu_buf_a: CudaSlice<u8> = unsafe { device.alloc::<u8>(max_block_bytes)? };
        let gpu_buf_b: CudaSlice<u8> = unsafe { device.alloc::<u8>(max_block_bytes)? };

        eprintln!("[FlameSwap] {} blocks, {:.1}MB max block, {:.1}MB pinned total, {:.1}MB GPU staging total",
            num_blocks, max_block_bytes as f64 / 1e6,
            (staging_a.len() + staging_b.len()) as f64 * 2.0 / 1e6,
            (max_block_bytes * 2) as f64 / 1e6);

        let state = Arc::new((
            Mutex::new(StagingState { request: None, complete: None, shutdown: false }),
            Condvar::new(),
        ));

        let ptrs = [
            SendPtr(staging_a.as_ptr() as *mut u16),
            SendPtr(staging_b.as_ptr() as *mut u16),
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
        let event = Event::new()?;

        Ok(Self {
            mmaps,
            blocks,
            layouts,
            gpu_layouts,
            staging: [staging_a, staging_b],
            gpu_buf: [gpu_buf_a, gpu_buf_b],
            slot_state: [SlotState::Idle, SlotState::Idle],
            max_block_bytes,
            transfer,
            event,
            pending: None,
            next_slot: Slot::A,
            state,
            stage_thread: Some(stage_thread),
            device: Arc::clone(device),
        })
    }

    pub fn num_blocks(&self) -> usize { self.blocks.len() }

    pub fn prefetch(&mut self, idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        assert!(idx < self.blocks.len(), "block index {idx} out of range");

        // Drain any leftover pending from a prior call.  In Phase 1 the DMA
        // is on the default stream so it is already serialized with the
        // caller's compute, but we still drop the slot record.
        if self.pending.is_some() {
            self.pending = None;
        }

        let slot = self.next_slot;

        // Slot reuse safety: in Phase 1 all DMA + compute is on the default
        // stream which serializes naturally.  Both Idle (first use) and
        // InCompute (caller has finished launching kernels that read this
        // slot) are valid starting states; subsequent default-stream DMA
        // will execute after those kernels complete in stream order.
        let prior = self.slot_state[slot.idx()];
        assert!(
            prior == SlotState::Idle || prior == SlotState::InCompute,
            "slot {slot:?} not reusable: state = {prior:?}",
        );
        self.slot_state[slot.idx()] = SlotState::Idle;

        self.slot_state[slot.idx()] = SlotState::StagingRequested;
        self.request_staging(idx, slot);
        self.wait_staging(idx, slot)?;
        self.slot_state[slot.idx()] = SlotState::Staged;

        let block = &self.blocks[idx];
        let layout = &self.layouts[idx];
        let gpu_layout = &self.gpu_layouts[idx];
        let mut pending_tensors = Vec::with_capacity(block.tensors.len());

        self.slot_state[slot.idx()] = SlotState::Transferring;
        let gpu_base = *self.gpu_buf[slot.idx()].device_ptr();
        let host_base = self.staging[slot.idx()].as_ptr();
        for ((t_meta, sl), gl) in block.tensors.iter().zip(layout.iter()).zip(gpu_layout.iter()) {
            debug_assert_eq!(sl.numel, gl.numel);
            unsafe {
                let dst = (gpu_base + gl.final_offset as u64) as *mut c_void;
                let src = host_base.add(sl.offset) as *const c_void;
                let bytes = gl.numel * std::mem::size_of::<u16>();
                ffi::flame_cuda_memcpy_async(dst, src, bytes, 1, std::ptr::null_mut());
            }
            pending_tensors.push(PendingTensor {
                name: t_meta.name.clone(),
                offset: gl.final_offset,
                numel: gl.numel,
                shape: t_meta.shape.clone(),
            });
        }
        self.slot_state[slot.idx()] = SlotState::Ready;

        self.pending = Some(Pending { block_idx: idx, slot, tensors: pending_tensors });
        self.next_slot = slot.flip();
        if idx + 1 < self.blocks.len() {
            self.request_staging(idx + 1, self.next_slot);
        }
        Ok(())
    }

    pub fn await_block(&mut self, idx: usize) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        let pending = self.pending.take()
            .ok_or("await_block called without a prior prefetch")?;
        assert_eq!(pending.block_idx, idx);
        let slot = pending.slot;
        debug_assert_eq!(self.slot_state[slot.idx()], SlotState::Ready);

        let gpu_base = *self.gpu_buf[slot.idx()].device_ptr();
        let mut weights = HashMap::with_capacity(pending.tensors.len());
        for pt in pending.tensors {
            let shape = Shape::new(pt.shape);
            let ptr = (gpu_base + pt.offset as u64) as *mut u16;
            let tensor = unsafe {
                Tensor::view_from_buffer(ptr, shape, Arc::clone(&self.device))
            };
            weights.insert(pt.name, tensor);
        }
        // Caller now owns non-owning views into gpu_buf[slot]; mark slot as
        // in compute so the next prefetch knows to wait for default-stream
        // serialization (Phase 1) or the done event (Phase 3).
        self.slot_state[slot.idx()] = SlotState::InCompute;
        Ok(weights)
    }

    pub fn load_block(&mut self, idx: usize) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        self.prefetch(idx)?;
        self.await_block(idx)
    }

    pub fn pinned_bytes(&self) -> usize {
        (self.staging[0].len() + self.staging[1].len()) * std::mem::size_of::<u16>()
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
    staging_ptrs: [*mut u16; 2],
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

        // Process each tensor — use thread::scope for FP8 parallelism
        for (t, sl) in tensors.iter().zip(layout.iter()) {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(dst_base.add(sl.offset), sl.numel)
            };
            match t.src_dtype {
                SourceDtype::F8E4M3 { scale } => {
                    let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + t.numel];
                    // Parallel FP8 dequant using 8 chunks
                    let num_threads = 8usize.min(t.numel / 1024 + 1);
                    // Multi-threaded FP8 dequant using chunks_mut
                    let chunk_size = (t.numel + num_threads - 1) / num_threads;
                    thread::scope(|s| {
                        for (i, dst_chunk) in dst.chunks_mut(chunk_size).enumerate() {
                            let start = i * chunk_size;
                            let src_chunk = &src[start..start + dst_chunk.len()];
                            s.spawn(move || {
                                for (d, &byte) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                                    *d = fp8_e4m3_to_bf16(byte, scale);
                                }
                            });
                        }
                    });
                }
                SourceDtype::F32 => {
                    let byte_len = t.numel * 4;
                    let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + byte_len];
                    let src_f32 = unsafe {
                        std::slice::from_raw_parts(src.as_ptr() as *const f32, t.numel)
                    };
                    for (d, &f) in dst.iter_mut().zip(src_f32.iter()) {
                        *d = half::bf16::from_f32(f).to_bits();
                    }
                }
                SourceDtype::BF16 => {
                    let byte_len = t.numel * 2;
                    let src = &mmaps[t.file_idx][t.file_offset..t.file_offset + byte_len];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(), dst.as_mut_ptr() as *mut u8, byte_len,
                        );
                    }
                }
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
