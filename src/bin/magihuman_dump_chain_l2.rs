//! Dump our chain output at layer 2 (after layers 0,1,2) so we can diff
//! element-wise vs Python's reference and characterize the drift.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::magihuman_dit::{
    MMTransformerLayer, MlpAct, GELU7_LAYERS, MM_LAYERS,
};

const FIXTURE_INPUT: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/h_after_adapter_40tok.safetensors";
const WEIGHTS: &str = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";

struct Facilitator;
impl flame_diffusion::block_offload::BlockFacilitator for Facilitator {
    fn block_count(&self) -> usize { 40 }
    fn classify_key(&self, name: &str) -> Option<usize> {
        name.strip_prefix("block.layers.")?.split('.').next()?.parse().ok()
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();
    let input_fix = flame_core::serialization::load_file(Path::new(FIXTURE_INPUT), &device)?;
    let input = input_fix.get("input").unwrap().clone();
    let rope = input_fix.get("rope").unwrap().clone();
    let group_sizes = vec![24usize, 8, 8];
    let mut offloader = flame_diffusion::BlockOffloader::load(&[WEIGHTS], &Facilitator, device.clone())
        .map_err(|e| anyhow!("offloader: {e}"))?;
    let rope_b = rope.to_dtype(DType::F32)?;
    let mut h = input.to_dtype(DType::BF16)?;

    offloader.prefetch_block(0).map_err(|e| anyhow!("prefetch: {e}"))?;
    for i in 0..3 {
        let raw = offloader.await_block(i).map_err(|e| anyhow!("await {i}: {e}"))?;
        if i + 1 < 40 {
            offloader.prefetch_block(i + 1).map_err(|e| anyhow!("prefetch {}: {e}", i + 1))?;
        }
        let weights: std::collections::HashMap<String, _> = raw.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let prefix = format!("block.layers.{i}.");
        let _is_mm = MM_LAYERS.contains(&i);
        let act = if GELU7_LAYERS.contains(&i) { MlpAct::GELU7 } else { MlpAct::SwiGLU7 };
        let layer = MMTransformerLayer::load_with_layout(&weights, &prefix, act, true)?;
        h = layer.forward(&h, &rope_b, &group_sizes)?;
        flame_core::cuda_alloc_pool::clear_pool_cache();
    }

    let out_path = "output/parity_run_2026-05-03/our_chain_after_l2.safetensors";
    let h_f32 = h.to_dtype(DType::F32)?;
    let mut map = std::collections::HashMap::new();
    map.insert("our_h_after_3_layers".to_string(), h_f32);
    flame_core::serialization::save_file(&map, Path::new(out_path))?;
    println!("Saved {}", out_path);
    Ok(())
}
