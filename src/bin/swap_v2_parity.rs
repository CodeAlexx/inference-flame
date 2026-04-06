//! Phase 2 parity test: load block tensors via FlameSwap and compare them
//! byte-for-byte against the raw safetensors file.
//!
//! Klein 4B is all BF16, so the FlameSwap path (raw memcpy in staging thread,
//! H2D into pre-allocated gpu_buf, view_from_buffer wrap) must produce
//! tensors that are bit-exact equal to the BF16 bytes on disk.

use std::collections::HashMap;
use std::time::Instant;

use flame_swap::FlameSwap;
use memmap2::Mmap;
use serde_json::Value;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors";

#[derive(Debug, Clone)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offset: usize, // absolute byte offset into the mmap
    bytes: usize,
}

fn parse_safetensors_header(mmap: &Mmap) -> Result<HashMap<String, TensorEntry>, Box<dyn std::error::Error>> {
    let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
    let header_str = std::str::from_utf8(&mmap[8..8 + header_len])?;
    let data_start = 8 + header_len;
    let v: Value = serde_json::from_str(header_str)?;
    let obj = v.as_object().ok_or("header is not an object")?;

    let mut out = HashMap::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().ok_or("missing dtype")?.to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let offsets = v["data_offsets"]
            .as_array()
            .ok_or("missing data_offsets")?;
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        out.insert(
            k.clone(),
            TensorEntry {
                dtype,
                shape,
                data_offset: data_start + start,
                bytes: end - start,
            },
        );
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    println!("--- FlameSwap v2 parity test (Klein 4B, BF16) ---");
    println!("Model: {MODEL_PATH}");

    // Direct mmap path so we can read the source-of-truth bytes.
    let file = std::fs::File::open(MODEL_PATH)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let header = parse_safetensors_header(&mmap)?;
    println!("Parsed header: {} tensors", header.len());

    // Init FlameSwap with the same block_fn as klein/swap_smoke.
    let num_double = 5usize;
    let mut swap = FlameSwap::load(
        &[MODEL_PATH],
        &device,
        |name| {
            if let Some(rest) = name.strip_prefix("double_blocks.") {
                rest.split('.').next()?.parse().ok()
            } else if let Some(rest) = name.strip_prefix("single_blocks.") {
                let idx: usize = rest.split('.').next()?.parse().ok()?;
                Some(num_double + idx)
            } else {
                None
            }
        },
    )?;

    let blocks_to_check: &[(usize, &str)] = &[
        (0, "double_blocks.0"),
        (5, "single_blocks.0"),
    ];

    let mut all_ok = true;
    for &(block_idx, label) in blocks_to_check {
        println!("\n--- block {block_idx} ({label}) ---");
        let t0 = Instant::now();
        swap.prefetch(block_idx)?;
        let weights = swap.await_block(block_idx)?;
        let elapsed = t0.elapsed().as_millis();
        println!("loaded {} tensors in {elapsed}ms", weights.len());

        let mut max_diff = 0u32;
        let mut total_elems = 0usize;
        let mut tensors_checked = 0usize;

        for (name, tensor) in &weights {
            let entry = header
                .get(name)
                .ok_or_else(|| format!("tensor {name} not found in header"))?;
            if entry.dtype != "BF16" {
                return Err(format!("Klein expected all BF16, but {name} is {}", entry.dtype).into());
            }
            let numel: usize = entry.shape.iter().product();
            // Source bytes from disk, interpreted as BF16 u16 values.
            let src_bytes = &mmap[entry.data_offset..entry.data_offset + entry.bytes];
            let src_u16: &[u16] = unsafe {
                std::slice::from_raw_parts(src_bytes.as_ptr() as *const u16, numel)
            };

            // GPU bytes pulled back to host as u16.
            let gpu_u16 = tensor.to_vec_bf16()?;
            if gpu_u16.len() != numel {
                return Err(format!(
                    "{name}: gpu numel {} != expected {numel}",
                    gpu_u16.len()
                ).into());
            }

            for i in 0..numel {
                let d = (src_u16[i] as i32 - gpu_u16[i] as i32).unsigned_abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            total_elems += numel;
            tensors_checked += 1;
        }

        if max_diff == 0 {
            println!(
                "block {block_idx}: ✓ exact match — {tensors_checked} tensors / {total_elems} elements"
            );
        } else {
            println!(
                "block {block_idx}: ✗ DIFFER — max_u16_diff={max_diff} across {tensors_checked} tensors"
            );
            all_ok = false;
        }
    }

    if all_ok {
        println!("\n--- PASS ---");
        Ok(())
    } else {
        Err("parity mismatch".into())
    }
}
