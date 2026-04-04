//! Debug encoder: single token, dump intermediate hidden states.

use inference_flame::models::qwen3_encoder::Qwen3Encoder;
use flame_core::{global_cuda_device, DType, Tensor};
use std::collections::HashMap;
use std::time::Instant;

const ENCODER_DIR: &str = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218";

fn load_sharded(dir: &str, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut all = HashMap::new();
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| { let n = e.file_name().to_string_lossy().to_string(); n.starts_with("model-") && n.ends_with(".safetensors") })
        .map(|e| e.path()).collect();
    paths.sort();
    for p in &paths {
        all.extend(flame_core::serialization::load_file(p, device)?);
    }
    Ok(all)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("Loading Qwen3 8B...");
    let t0 = Instant::now();
    let weights = load_sharded(ENCODER_DIR, &device)?;
    let config = Qwen3Encoder::config_from_weights(&weights)?;
    let encoder = Qwen3Encoder::new(weights, config, device.clone());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Single token: 14990 ("hello"), no padding
    let token_ids = vec![14990i32];
    println!("\nEncoding single token [14990] ('hello')...");
    let result = encoder.encode(&token_ids)?;

    // Print first 5 values
    let result_f32 = result.to_dtype(DType::F32)?;
    let data = result_f32.to_vec()?;
    println!("  Output shape: {:?}", result.dims());
    println!("  Output[0,:5]: {:?}", &data[..5]);

    // Test 2 tokens (no padding): [14990, 1879] = "hello world"
    println!("\nEncoding 2 tokens [14990, 1879] ('hello world')...");
    let token_ids_2 = vec![14990i32, 1879];
    let result_2 = encoder.encode(&token_ids_2)?;
    let r2_f32 = result_2.to_dtype(DType::F32)?;
    let d2 = r2_f32.to_vec()?;
    println!("  Output shape: {:?}", result_2.dims());
    // First 4096 values = layer 8 output for token 0
    println!("  Token 0 layer8 [:5]: {:?}", &d2[..5]);
    println!("  (PyTorch ref:        [-1.2969, 4.0625, 1.2344, 1.1719, 6.0625])");
    // Token 1 starts at offset 12288 (3 stacked layers × 4096)
    println!("  Token 1 layer8 [:5]: {:?}", &d2[12288..12293]);
    println!("  (PyTorch ref:        [0.0078, -0.2354, -0.6563, -1.2813, 1.6172])");

    Ok(())
}
