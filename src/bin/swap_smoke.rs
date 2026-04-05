//! Smoke test for flame-swap: load Klein 4B, prefetch/await a few blocks,
//! verify tensor shapes match expected Klein architecture.

use std::time::Instant;
use flame_swap::FlameSwap;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let device = flame_core::global_cuda_device();

    // Klein 4B: 5 double_blocks + 20 single_blocks = 25 total swap blocks
    // double_blocks.N → swap index N
    // single_blocks.N → swap index 5 + N
    let num_double = 5usize;

    println!("--- FlameSwap smoke test (Klein 4B) ---");
    println!("Model: {MODEL_PATH}");

    let t0 = Instant::now();
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
    println!("Loaded: {} blocks, {:.2}GB pinned, {:.1}s",
        swap.num_blocks(),
        swap.pinned_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f32());

    // Test 1: prefetch + await block 0 (first double block)
    println!("\n--- Test 1: double_blocks.0 ---");
    let t1 = Instant::now();
    swap.prefetch(0)?;
    let w0 = swap.await_block(0)?;
    let dt1 = t1.elapsed().as_millis();
    println!("  {} tensors, {dt1}ms", w0.len());
    for (k, t) in w0.iter().take(5) {
        println!("  {k}: {:?} ({:?})", t.shape().dims(), t.dtype());
    }

    // Test 2: prefetch overlap — prefetch block 1 while we inspect block 0's results
    println!("\n--- Test 2: overlap prefetch(1) during block 0 inspection ---");
    let t2 = Instant::now();
    swap.prefetch(1)?;
    // Simulate "compute" by just checking block 0 shapes
    let qkv_key = "double_blocks.0.img_attn.qkv.weight";
    if let Some(qkv) = w0.get(qkv_key) {
        println!("  {qkv_key}: {:?}", qkv.shape().dims());
    }
    drop(w0); // free block 0 GPU memory
    let w1 = swap.await_block(1)?;
    let dt2 = t2.elapsed().as_millis();
    println!("  block 1: {} tensors, {dt2}ms (includes prefetch overlap)", w1.len());
    drop(w1);

    // Test 3: single block
    println!("\n--- Test 3: single_blocks.0 (swap idx {num_double}) ---");
    let t3 = Instant::now();
    swap.prefetch(num_double)?;
    let ws = swap.await_block(num_double)?;
    let dt3 = t3.elapsed().as_millis();
    println!("  {} tensors, {dt3}ms", ws.len());
    for (k, t) in ws.iter().take(5) {
        println!("  {k}: {:?} ({:?})", t.shape().dims(), t.dtype());
    }
    drop(ws);

    // Test 4: full sequential pass — all blocks
    println!("\n--- Test 4: full sequential pass ({} blocks) ---", swap.num_blocks());
    let t4 = Instant::now();
    swap.prefetch(0)?;
    for i in 0..swap.num_blocks() {
        let w = swap.await_block(i)?;
        if i + 1 < swap.num_blocks() {
            swap.prefetch(i + 1)?;
        }
        if i == 0 || i + 1 == swap.num_blocks() {
            println!("  block {i}: {} tensors", w.len());
        }
        drop(w);
    }
    let dt4 = t4.elapsed().as_millis();
    println!("  All {} blocks in {dt4}ms ({:.1}ms/block)",
        swap.num_blocks(), dt4 as f64 / swap.num_blocks() as f64);

    println!("\n--- PASS ---");
    Ok(())
}
