//! Parity test: compare Rust ACE-Step outputs against Python reference tensors.
use flame_core::{serialization, CudaDevice, DType, Shape, Tensor};
use inference_flame::models::acestep_condition::AceStepConditionEncoder;
use inference_flame::models::acestep_dit::AceStepDiT;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

fn compare(name: &str, rust: &Tensor, py: &Tensor, atol: f32) {
    let r = rust.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let p = py.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_eq!(r.len(), p.len(), "{name}: length mismatch {} vs {}", r.len(), p.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for (a, b) in r.iter().zip(p.iter()) {
        let d = (a - b).abs();
        if d > max_diff { max_diff = d; }
        sum_diff += d as f64;
    }
    let mean_diff = sum_diff / r.len() as f64;
    let pass = max_diff < atol;
    eprintln!("[{}] {} max_diff={:.6} mean_diff={:.8} (atol={})",
        if pass { "PASS" } else { "FAIL" }, name, max_diff, mean_diff, atol);
    if !pass {
        // Print first few values for debugging
        eprintln!("  rust first 5: {:?}", &r[..5.min(r.len())]);
        eprintln!("  py   first 5: {:?}", &p[..5.min(p.len())]);
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = CudaDevice::new(0).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Load Python reference tensors
    let dit_ref = serialization::load_file(Path::new("/tmp/acestep_dit_parity_ref.safetensors"), &device)?;
    let gen_ref = serialization::load_file(Path::new("/tmp/acestep_gen_parity_ref.safetensors"), &device)?;

    // Load model
    let ckpt = Path::new("/home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo/model.safetensors");
    let all_weights = serialization::load_file(ckpt, &device)?;

    // Split weights
    let mut decoder_weights = HashMap::new();
    for (k, v) in &all_weights {
        if k.starts_with("decoder.") {
            decoder_weights.insert(k.clone(), v.clone());
        }
    }

    let mut dit = AceStepDiT::from_weights(decoder_weights)?;

    // Load condition ref
    let cond_ref = serialization::load_file(Path::new("/tmp/acestep_parity_ref.safetensors"), &device)?;
    let encoder_hs = cond_ref.get("encoder_hs").unwrap();
    let context = cond_ref.get("context_latents").unwrap();

    // --- Test 1: Single DiT step ---
    let noise = dit_ref.get("noise").unwrap();
    let py_pred = dit_ref.get("dit_pred").unwrap();
    let t = Tensor::from_vec(vec![0.875f32], Shape::from_dims(&[1]), device.clone())?
        .to_dtype(DType::BF16)?;

    eprintln!("\n=== DiT Single Step Parity ===");
    eprintln!("noise: {:?}", noise.shape().dims());
    eprintln!("encoder_hs: {:?}", encoder_hs.shape().dims());
    eprintln!("context: {:?}", context.shape().dims());

    let rust_pred = dit.forward(noise, &t, &t, encoder_hs, context)?;
    compare("dit_step", &rust_pred, py_pred, 0.5); // BF16 tolerance

    // --- Test 2: Check generation output shapes ---
    let py_gen = gen_ref.get("target_latents").unwrap();
    eprintln!("\n=== Generation Reference Stats ===");
    let py_v = py_gen.to_dtype(DType::F32)?.to_vec()?;
    let py_mean: f32 = py_v.iter().sum::<f32>() / py_v.len() as f32;
    let py_std: f32 = (py_v.iter().map(|x| (x - py_mean).powi(2)).sum::<f32>() / py_v.len() as f32).sqrt();
    eprintln!("py_gen: shape={:?} mean={:.6} std={:.6}", py_gen.shape().dims(), py_mean, py_std);

    eprintln!("\n=== All parity tests complete ===");
    Ok(())
}
