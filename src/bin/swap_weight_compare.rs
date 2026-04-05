//! Compare block weights loaded via FlameSwap vs sync load_block_from_disk.
//! Finds the exact tensor that differs between both paths.

use flame_core::{global_cuda_device, DType, Result, Shape, Tensor};
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use std::collections::HashMap;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();
    let config = LTX2Config::default();

    println!("Loading model globals...");
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;

    // Load block 0 via sync path
    println!("\nLoading block 0 via sync (load_block_from_disk)...");
    let sync_block = model.load_block(0)?;

    // Now init FlameSwap and load block 0 via async path
    println!("Initializing FlameSwap...");
    model.init_swap()?;

    println!("Loading block 0 via FlameSwap...");
    let swap = model.swap.as_mut().unwrap();
    let key_prefix = &model.key_prefix;

    swap.prefetch(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let raw_weights = swap.await_block(0).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Strip prefix
    let swap_weights: HashMap<String, Tensor> = raw_weights.into_iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix(key_prefix).unwrap_or(&k).to_string();
            (stripped, v)
        })
        .collect();

    println!("\nSync block keys: (from load_block struct fields)");
    println!("Swap block keys: {} total", swap_weights.len());
    for k in swap_weights.keys().take(5) {
        println!("  {k}");
    }

    // Compare specific weights
    // The sync path builds LTX2TransformerBlock which pre-transposes weights.
    // The swap path returns raw weights that get pre-transposed in load_block_from_weights_static.
    // Let's build the block from swap weights too and compare.
    println!("\nBuilding block from swap weights...");
    let swap_block = LTX2StreamingModel::load_block_from_weights_static(
        &config, 0, swap_weights,
    )?;

    // Compare attn1 Q weight
    compare_tensors("attn1.to_q_weight", &sync_block.attn1.to_q_weight, &swap_block.attn1.to_q_weight)?;
    compare_tensors("attn1.to_k_weight", &sync_block.attn1.to_k_weight, &swap_block.attn1.to_k_weight)?;
    compare_tensors("attn1.to_v_weight", &sync_block.attn1.to_v_weight, &swap_block.attn1.to_v_weight)?;
    compare_tensors("attn1.to_q_bias", &sync_block.attn1.to_q_bias, &swap_block.attn1.to_q_bias)?;
    compare_tensors("attn1.norm_q_weight", &sync_block.attn1.norm_q_weight, &swap_block.attn1.norm_q_weight)?;
    compare_tensors("attn1.to_out_weight", &sync_block.attn1.to_out_weight, &swap_block.attn1.to_out_weight)?;

    compare_tensors("attn2.to_q_weight", &sync_block.attn2.to_q_weight, &swap_block.attn2.to_q_weight)?;

    compare_tensors("ff.gelu_proj_weight", &sync_block.ff.gelu_proj_weight, &swap_block.ff.gelu_proj_weight)?;
    compare_tensors("ff.out_weight", &sync_block.ff.out_weight, &swap_block.ff.out_weight)?;

    compare_tensors("scale_shift_table", &sync_block.scale_shift_table, &swap_block.scale_shift_table)?;

    if let (Some(ref a), Some(ref b)) = (&sync_block.attn1.to_gate_logits_weight, &swap_block.attn1.to_gate_logits_weight) {
        compare_tensors("attn1.gate_logits_weight", a, b)?;
    }

    if let (Some(ref a), Some(ref b)) = (&sync_block.prompt_scale_shift_table, &swap_block.prompt_scale_shift_table) {
        compare_tensors("prompt_scale_shift_table", a, b)?;
    }

    // Now run block 0 forward with both and compare output
    println!("\n=== Forward comparison ===");
    let inner_dim = config.inner_dim();
    let num_tokens = 270;

    // Create dummy inputs
    let x = Tensor::from_f32_to_bf16(
        vec![0.1f32; num_tokens * inner_dim],
        Shape::from_dims(&[1, num_tokens, inner_dim]),
        device.clone(),
    )?;
    let enc = Tensor::from_f32_to_bf16(
        vec![0.05f32; 1024 * inner_dim],
        Shape::from_dims(&[1, 1024, inner_dim]),
        device.clone(),
    )?;
    let temb = Tensor::from_f32_to_bf16(
        vec![0.01f32; num_tokens * 9 * inner_dim],
        Shape::from_dims(&[1, num_tokens, 9 * inner_dim]),
        device.clone(),
    )?;

    println!("Running sync block 0 forward...");
    let out_sync = sync_block.forward_video_only(&x, &enc, &temb, None, None, None)?;
    let sync_data = out_sync.to_vec()?;
    println!("  Output: mean={:.6} std={:.6}",
        sync_data.iter().sum::<f32>() / sync_data.len() as f32,
        {
            let m = sync_data.iter().sum::<f32>() / sync_data.len() as f32;
            (sync_data.iter().map(|v| (v - m).powi(2)).sum::<f32>() / sync_data.len() as f32).sqrt()
        });

    println!("Running swap block 0 forward...");
    let out_swap = swap_block.forward_video_only(&x, &enc, &temb, None, None, None)?;
    let swap_data = out_swap.to_vec()?;
    println!("  Output: mean={:.6} std={:.6}",
        swap_data.iter().sum::<f32>() / swap_data.len() as f32,
        {
            let m = swap_data.iter().sum::<f32>() / swap_data.len() as f32;
            (swap_data.iter().map(|v| (v - m).powi(2)).sum::<f32>() / swap_data.len() as f32).sqrt()
        });

    compare_tensors("block0_output", &out_sync, &out_swap)?;

    println!("\nDone.");
    Ok(())
}

fn compare_tensors(name: &str, a: &Tensor, b: &Tensor) -> anyhow::Result<()> {
    let a_shape = a.shape().dims().to_vec();
    let b_shape = b.shape().dims().to_vec();

    if a_shape != b_shape {
        println!("  {} SHAPE MISMATCH: {:?} vs {:?}", name, a_shape, b_shape);
        return Ok(());
    }

    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;

    let mut max_diff: f32 = 0.0;
    let mut max_idx = 0;
    let mut n_diff = 0usize;
    for (i, (&av, &bv)) in a_data.iter().zip(b_data.iter()).enumerate() {
        let d = (av - bv).abs();
        if d > 0.0001 { n_diff += 1; }
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }

    if max_diff < 0.001 {
        println!("  {} ✓ MATCH ({} elems, max_diff={:.6})", name, a_data.len(), max_diff);
    } else {
        println!("  {} ✗ DIFFER ({} elems, max_diff={:.4} at idx {}, {}/{} differ >0.0001)",
            name, a_data.len(), max_diff, max_idx, n_diff, a_data.len());
        println!("    sync[{}]={:.6} swap[{}]={:.6}", max_idx, a_data[max_idx], max_idx, b_data[max_idx]);
        // First 5 values
        println!("    sync[:5]={:?}", &a_data[..5]);
        println!("    swap[:5]={:?}", &b_data[..5]);
    }

    Ok(())
}
