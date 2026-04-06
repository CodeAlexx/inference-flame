//! CPU-cached block swapper for FLAME inference.
//!
//! All block weights are pre-dequanted to BF16 in CPU RAM at init time.
//! At runtime, a background staging thread memcpys from CPU cache → pinned
//! buffer, then async DMA copies pinned → GPU. No per-element conversion
//! at runtime — just fast memcpy.
//!
//! Three-stage pipeline, all overlapped:
//!   CPU thread:  cache[N+1] → pinned[B]  |  cache[N+2] → pinned[A]
//!   DMA stream:  pinned[A] → GPU          |  pinned[B] → GPU
//!   GPU compute: block N                  |  block N+1
//!
//! Two pinned staging buffers (~2GB each). While GPU computes block N from
//! buffer A, the staging thread copies block N+1 from CPU cache into buffer B,
//! and DMA transfers it to GPU. Everything overlaps.
//!
//! Peak CPU RAM: ~33GB (all blocks pre-dequanted to BF16).
//! Peak pinned memory: ~4GB (two staging buffers).
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

/// Convert a single FP8 E4M3 byte to f32.
/// E4M3: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
#[inline]
fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }

    let (effective_exp, effective_mant) = if exp == 0 {
        (-6i32, mant as f32 / 8.0)
    } else {
        (exp as i32 - 7, 1.0 + mant as f32 / 8.0)
    };

    let magnitude = effective_mant * (2.0f32).powi(effective_exp);
    if sign == 1 { -magnitude } else { magnitude }
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

/// Per-tensor offset within the staging buffer.
#[derive(Debug, Clone)]
struct StagingLayout {
    offset: usize,
    numel: usize,
}

// ---------------------------------------------------------------------------
// Pending DMA
// ---------------------------------------------------------------------------

struct PendingTensor {
    name: String,
    gpu: CudaSlice<u16>,
    shape: Vec<usize>,
}

struct Pending {
    block_idx: usize,
    tensors: Vec<PendingTensor>,
}

// ---------------------------------------------------------------------------
// Double-buffer slot
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    A = 0,
    B = 1,
}

impl Slot {
    fn flip(self) -> Self {
        match self {
            Slot::A => Slot::B,
            Slot::B => Slot::A,
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------------------
// Staging thread communication
// ---------------------------------------------------------------------------

struct StagingState {
    request: Option<(usize, Slot)>,   // (block_idx, slot)
    complete: Option<(usize, Slot)>,  // (block_idx, slot)
    shutdown: bool,
}

/// Send-safe wrapper for raw pointer passed to staging thread.
struct SendPtr(*mut u16);
unsafe impl Send for SendPtr {}

// ---------------------------------------------------------------------------
// FlameSwap
// ---------------------------------------------------------------------------

pub struct FlameSwap {
    /// Pre-dequanted block data in CPU RAM. Each entry is a flat BF16 buffer
    /// matching the staging layout for that block. All FP8→BF16 conversion
    /// happens at init, never at runtime.
    cpu_blocks: Arc<Vec<Vec<u8>>>,

    /// Per-block metadata (tensor names, shapes, layouts).
    blocks: Vec<BlockMeta>,

    /// Per-block staging layouts: offset + numel per tensor.
    layouts: Vec<Vec<StagingLayout>>,

    /// Double-buffered pinned staging: [A, B].
    staging: [PinnedHostBuffer<u16>; 2],

    /// Transfer stream for async H2D.
    transfer: Stream,

    /// Event for stream ordering.
    event: Event,

    /// Currently in-flight DMA.
    pending: Option<Pending>,

    /// Which slot was last used for DMA.
    next_slot: Slot,

    /// Shared state with CPU staging thread.
    state: Arc<(Mutex<StagingState>, Condvar)>,

    /// CPU staging thread handle.
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
        // ----- 1. Parse headers, build block metadata -----------------------
        let mut all_tensors: HashMap<usize, Vec<TensorMeta>> = HashMap::new();

        // Open mmaps for header parsing + data reading
        let mmaps: Vec<Mmap> = paths
            .iter()
            .map(|p| {
                let f = std::fs::File::open(p)?;
                unsafe { Mmap::map(&f) }.map_err(Into::into)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

        for (file_idx, mmap) in mmaps.iter().enumerate() {
            let header_entries = parse_safetensors_header(mmap)?;

            // Build scale map for FP8 tensors
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

        // ----- 2. Precompute staging layouts --------------------------------
        let layouts: Vec<Vec<StagingLayout>> = blocks
            .iter()
            .map(|block| {
                let mut offset = 0usize;
                block
                    .tensors
                    .iter()
                    .map(|t| {
                        let layout = StagingLayout { offset, numel: t.numel };
                        offset += t.numel;
                        layout
                    })
                    .collect()
            })
            .collect();

        // ----- 3. Pre-dequant ALL blocks to BF16 in CPU RAM ----------------
        // This is a one-time cost at init. All FP8→BF16 conversion happens here.
        // At runtime, the staging thread just does fast memcpy.
        let t0 = std::time::Instant::now();
        eprintln!("[FlameSwap] Pre-caching {} blocks to CPU RAM (FP8→BF16 dequant)...", num_blocks);

        // Prepare per-block work items for parallel processing
        let block_work: Vec<(usize, &BlockMeta, &Vec<StagingLayout>)> = blocks
            .iter()
            .zip(layouts.iter())
            .enumerate()
            .map(|(i, (b, l))| (i, b, l))
            .collect();

        // Use std::thread::scope for parallel pre-dequant (up to 8 threads)
        let num_threads = 8usize.min(num_blocks);
        let chunks: Vec<Vec<(usize, &BlockMeta, &Vec<StagingLayout>)>> = {
            let chunk_size = (block_work.len() + num_threads - 1) / num_threads;
            block_work.chunks(chunk_size).map(|c| c.to_vec()).collect()
        };

        let mmaps_ref = &mmaps;
        let mut cpu_blocks: Vec<Vec<u8>> = vec![Vec::new(); num_blocks];

        thread::scope(|s| {
            let mut handles = Vec::new();
            for chunk in &chunks {
                let handle = s.spawn(move || {
                    let mut results: Vec<(usize, Vec<u8>)> = Vec::new();
                    for &(block_idx, block, layout) in chunk {
                        let byte_len = block.total_u16s * 2;
                        let mut buf = vec![0u8; byte_len];

                        for (t, sl) in block.tensors.iter().zip(layout.iter()) {
                            let dst_byte_offset = sl.offset * 2;
                            let dst_u16 = unsafe {
                                std::slice::from_raw_parts_mut(
                                    buf[dst_byte_offset..].as_mut_ptr() as *mut u16,
                                    sl.numel,
                                )
                            };

                            match t.src_dtype {
                                SourceDtype::F8E4M3 { scale } => {
                                    let src = &mmaps_ref[t.file_idx]
                                        [t.file_offset..t.file_offset + t.numel];
                                    for (d, &byte) in dst_u16.iter_mut().zip(src.iter()) {
                                        let f = fp8_e4m3_to_f32(byte) * scale;
                                        *d = half::bf16::from_f32(f).to_bits();
                                    }
                                }
                                SourceDtype::F32 => {
                                    let src_bytes = &mmaps_ref[t.file_idx]
                                        [t.file_offset..t.file_offset + t.numel * 4];
                                    let src_f32 = unsafe {
                                        std::slice::from_raw_parts(
                                            src_bytes.as_ptr() as *const f32,
                                            t.numel,
                                        )
                                    };
                                    for (d, &f) in dst_u16.iter_mut().zip(src_f32.iter()) {
                                        *d = half::bf16::from_f32(f).to_bits();
                                    }
                                }
                                SourceDtype::BF16 => {
                                    let src = &mmaps_ref[t.file_idx]
                                        [t.file_offset..t.file_offset + t.numel * 2];
                                    let dst_bytes = &mut buf[dst_byte_offset..dst_byte_offset + sl.numel * 2];
                                    dst_bytes.copy_from_slice(src);
                                }
                            }
                        }
                        results.push((block_idx, buf));
                    }
                    results
                });
                handles.push(handle);
            }

            for handle in handles {
                let results = handle.join().unwrap();
                for (idx, buf) in results {
                    cpu_blocks[idx] = buf;
                }
            }
        });

        let total_cached_mb: f64 = cpu_blocks.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
        eprintln!(
            "[FlameSwap] CPU cache ready: {} blocks, {:.1}MB total, {:.1}s (8 threads)",
            num_blocks, total_cached_mb, t0.elapsed().as_secs_f32()
        );

        // Drop mmaps — no longer needed, all data is in cpu_blocks
        drop(mmaps);

        let cpu_blocks = Arc::new(cpu_blocks);

        // ----- 4. Two pinned staging buffers --------------------------------
        let max_u16s = blocks.iter().map(|b| b.total_u16s).max().unwrap_or(1);
        let staging_a =
            PinnedHostBuffer::<u16>::with_capacity_elems(max_u16s, PinnedAllocFlags::DEFAULT)?;
        let staging_b =
            PinnedHostBuffer::<u16>::with_capacity_elems(max_u16s, PinnedAllocFlags::DEFAULT)?;

        // ----- 5. Spawn CPU staging thread ----------------------------------
        let state = Arc::new((
            Mutex::new(StagingState {
                request: None,
                complete: None,
                shutdown: false,
            }),
            Condvar::new(),
        ));

        let ptrs = [
            SendPtr(staging_a.as_ptr() as *mut u16),
            SendPtr(staging_b.as_ptr() as *mut u16),
        ];
        let thread_cpu_blocks = Arc::clone(&cpu_blocks);
        let thread_block_sizes: Vec<usize> = blocks.iter().map(|b| b.total_u16s).collect();
        let thread_state = Arc::clone(&state);

        let stage_thread = thread::spawn(move || {
            staging_thread_main(
                thread_cpu_blocks,
                thread_block_sizes,
                [ptrs[0].0, ptrs[1].0],
                thread_state,
            );
        });

        // ----- 6. Transfer stream + event -----------------------------------
        let transfer = Stream::new()?;
        let event = Event::new()?;

        Ok(Self {
            cpu_blocks,
            blocks,
            layouts,
            staging: [staging_a, staging_b],
            transfer,
            event,
            pending: None,
            next_slot: Slot::A,
            state,
            stage_thread: Some(stage_thread),
            device: Arc::clone(device),
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Begin async prefetch of block `idx` to GPU.
    ///
    /// Internally: requests CPU staging on the current slot (fast memcpy from
    /// pre-cached CPU data), waits for it, then starts async DMA to GPU.
    /// Immediately kicks off CPU staging for `idx+1` on the other slot.
    pub fn prefetch(&mut self, idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        assert!(idx < self.blocks.len(), "block index {idx} out of range");

        // If there's an un-consumed DMA, sync it so we don't leak
        if self.pending.is_some() {
            self.transfer.synchronize()?;
            self.pending = None;
        }

        let slot = self.next_slot;

        // Request staging for this block
        self.request_staging(idx, slot);
        // Wait for CPU staging to complete (fast — just a memcpy now)
        self.wait_staging(idx, slot)?;

        // DMA: staging[slot] → GPU (async on default stream)
        let block = &self.blocks[idx];
        let layout = &self.layouts[idx];
        let mut pending_tensors = Vec::with_capacity(block.tensors.len());

        for (t_meta, sl) in block.tensors.iter().zip(layout.iter()) {
            let gpu: CudaSlice<u16> = unsafe { self.device.alloc::<u16>(sl.numel)? };

            unsafe {
                let dst = *gpu.device_ptr() as *mut c_void;
                let src = self.staging[slot.idx()].as_ptr().add(sl.offset) as *const c_void;
                let bytes = sl.numel * std::mem::size_of::<u16>();
                ffi::flame_cuda_memcpy_async(dst, src, bytes, 1, std::ptr::null_mut());
            }

            pending_tensors.push(PendingTensor {
                name: t_meta.name.clone(),
                gpu,
                shape: t_meta.shape.clone(),
            });
        }

        self.pending = Some(Pending {
            block_idx: idx,
            tensors: pending_tensors,
        });

        // Flip slot and immediately start staging next block on the other slot
        self.next_slot = slot.flip();
        if idx + 1 < self.blocks.len() {
            self.request_staging(idx + 1, self.next_slot);
        }

        Ok(())
    }

    /// Wait for the prefetched block and return its weights as GPU tensors.
    pub fn await_block(
        &mut self,
        idx: usize,
    ) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        let pending = self
            .pending
            .take()
            .ok_or("await_block called without a prior prefetch")?;

        assert_eq!(
            pending.block_idx, idx,
            "await_block({idx}) but block {} was prefetched",
            pending.block_idx
        );

        // H2D runs on default (null) stream — naturally ordered with compute.
        // No explicit sync needed.

        let mut weights = HashMap::with_capacity(pending.tensors.len());
        for pt in pending.tensors {
            let shape = Shape::new(pt.shape);
            let tensor = Tensor::from_bf16_slice_gpu(pt.gpu, shape, Arc::clone(&self.device));
            weights.insert(pt.name, tensor);
        }

        Ok(weights)
    }

    /// Convenience: prefetch + await in one call.
    pub fn load_block(
        &mut self,
        idx: usize,
    ) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        self.prefetch(idx)?;
        self.await_block(idx)
    }

    /// Total pinned CPU memory used (bytes). Two staging buffers.
    pub fn pinned_bytes(&self) -> usize {
        (self.staging[0].len() + self.staging[1].len()) * std::mem::size_of::<u16>()
    }

    /// Total CPU RAM used by pre-cached blocks (bytes).
    pub fn cached_bytes(&self) -> usize {
        self.cpu_blocks.iter().map(|b| b.len()).sum()
    }

    // -- Internal staging coordination ------------------------------------

    fn request_staging(&self, block_idx: usize, slot: Slot) {
        let (lock, cvar) = &*self.state;
        let mut guard = lock.lock().unwrap();
        guard.request = Some((block_idx, slot));
        cvar.notify_one();
    }

    fn wait_staging(
        &self,
        block_idx: usize,
        slot: Slot,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
        // Signal shutdown
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
// CPU staging thread — fast memcpy from pre-cached BF16 data
// ---------------------------------------------------------------------------

fn staging_thread_main(
    cpu_blocks: Arc<Vec<Vec<u8>>>,
    block_sizes: Vec<usize>, // total_u16s per block
    staging_ptrs: [*mut u16; 2],
    state: Arc<(Mutex<StagingState>, Condvar)>,
) {
    let (lock, cvar) = &*state;
    loop {
        // Wait for a request
        let mut guard = lock.lock().unwrap();
        while guard.request.is_none() && !guard.shutdown {
            guard = cvar.wait(guard).unwrap();
        }
        if guard.shutdown {
            return;
        }
        let (block_idx, slot) = guard.request.take().unwrap();
        drop(guard); // release lock during copy

        // Fast memcpy: pre-cached BF16 → pinned staging buffer
        let src = &cpu_blocks[block_idx];
        let byte_len = block_sizes[block_idx] * 2; // u16 → bytes
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                staging_ptrs[slot.idx()] as *mut u8,
                byte_len,
            )
        };
        // src.len() should equal byte_len, but be safe
        let copy_len = src.len().min(byte_len);
        dst[..copy_len].copy_from_slice(&src[..copy_len]);

        // Signal completion
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
    if mmap.len() < 8 {
        return Err("file too small for safetensors".into());
    }

    let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
    if mmap.len() < 8 + header_len {
        return Err("truncated safetensors header".into());
    }

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
        if offsets.len() != 2 {
            continue;
        }

        entries.push(HeaderEntry {
            name: key.to_string(),
            shape,
            data_offset: data_start + offsets[0],
            src_dtype,
        });
    }

    Ok(entries)
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers
// ---------------------------------------------------------------------------

fn find_char(bytes: &[u8], ch: u8, from: usize) -> Option<usize> {
    bytes[from..].iter().position(|&b| b == ch).map(|p| p + from)
}

fn skip_object(bytes: &[u8], pos: usize) -> usize {
    let mut depth = 0i32;
    let mut i = pos;
    let mut in_string = false;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_string => {
                i += 1;
            }
            b'"' => {
                in_string = !in_string;
            }
            b'{' if !in_string => {
                depth += 1;
            }
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    bytes.len()
}

fn extract_string_field(obj: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\"", field);
    let idx = obj.find(&pattern)?;
    let after_key = idx + pattern.len();
    let rest = &obj[after_key..];
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
    let values: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Some(values)
}
