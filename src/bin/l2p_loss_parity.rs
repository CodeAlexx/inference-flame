//! T3 — L2P training-step loss parity (Rust side, mirrors t3_loss_parity.py).
//!
//! Reads SHARED inputs (clean, noise, cap_feats, sigma) from a safetensors
//! file produced by the Python script, runs ONE step of the train_l2p.rs
//! math (rectified-flow target + MSE), and writes the resulting tensors so
//! a diff against Python can flag any divergence in:
//!  - noisy = (1-sigma)*clean + sigma*noise
//!  - target = noise - clean
//!  - pred  = L2pDiT.forward(noisy, t=sigma, cap_feats)  (DiT applies sign)
//!  - loss  = mean((pred - target)^2) in F32
//!
//! Usage:
//!   FLAME_ALLOC_POOL=0 FLAME_AUTOGRAD_OFF=1 \
//!   l2p_loss_parity \
//!     --model /home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors \
//!     --shared /tmp/l2p_thorough_parity/t3_shared_inputs.safetensors \
//!     --out    /tmp/l2p_thorough_parity/t3_rust_outputs.safetensors

use cudarc::driver::CudaDevice;
use flame_core::serialization::{load_file, save_tensors, SerializationFormat};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;

use inference_flame::models::l2p::weight_loader::load_l2p_safetensors;
use inference_flame::models::l2p::L2pDiT;

struct Args {
    model_path: String,
    shared_path: String,
    out_path: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut shared_path = String::new();
    let mut out_path = String::new();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { i += 1; model_path = argv[i].clone(); }
            "--shared" => { i += 1; shared_path = argv[i].clone(); }
            "--out" => { i += 1; out_path = argv[i].clone(); }
            other => { eprintln!("Unknown arg: {other}"); std::process::exit(1); }
        }
        i += 1;
    }
    if model_path.is_empty() || shared_path.is_empty() || out_path.is_empty() {
        eprintln!("Usage: l2p_loss_parity --model <path> --shared <inputs.safetensors> --out <outputs.safetensors>");
        std::process::exit(1);
    }
    Args { model_path, shared_path, out_path }
}

fn main() -> Result<()> {
    if std::env::var("FLAME_AUTOGRAD_OFF").ok().as_deref() != Some("0") {
        flame_core::AutogradContext::set_enabled(false);
    }

    let args = parse_args();
    let device = CudaDevice::new(0).map_err(|e| Error::InvalidOperation(format!("CUDA init: {e:?}")))?;
    println!("[t3-rust] CUDA ready");

    // Model
    let translated = load_l2p_safetensors(std::path::Path::new(&args.model_path), &device)?;
    let mut model = L2pDiT::new_resident(translated, device.clone());
    println!("[t3-rust] model loaded");

    // Shared inputs
    let shared = load_file(&args.shared_path, &device)?;
    let clean = shared.get("clean")
        .ok_or_else(|| Error::InvalidInput("missing 'clean'".into()))?
        .to_dtype(DType::BF16)?;
    let noise = shared.get("noise")
        .ok_or_else(|| Error::InvalidInput("missing 'noise'".into()))?
        .to_dtype(DType::BF16)?;
    let cap_feats = shared.get("cap_feats")
        .ok_or_else(|| Error::InvalidInput("missing 'cap_feats'".into()))?
        .to_dtype(DType::BF16)?;
    let sigma_t = shared.get("sigma")
        .ok_or_else(|| Error::InvalidInput("missing 'sigma'".into()))?;
    let sigma_val = sigma_t.to_vec()?[0];
    println!("[t3-rust] sigma = {}", sigma_val);

    // Mirror train_l2p.rs step-0 math:
    //   noisy = (1 - sigma) * clean + sigma * noise
    //   target = noise - clean
    //   v_in = sigma  → L2pDiT.forward applies (1 - v_in) * 1000 internally
    let noisy = clean.mul_scalar(1.0 - sigma_val)?
        .add(&noise.mul_scalar(sigma_val)?)?;
    let target = noise.sub(&clean)?;
    let v_in = Tensor::from_vec(vec![sigma_val], Shape::from_dims(&[1]), device.clone())?
        .to_dtype(DType::BF16)?;

    println!("[t3-rust] noisy shape={:?} dtype={:?}", noisy.shape().dims(), noisy.dtype());
    println!("[t3-rust] target shape={:?} dtype={:?}", target.shape().dims(), target.dtype());

    // Forward (autograd OFF — pure inference)
    let pred = model.forward(&noisy, &v_in, &cap_feats)?;
    println!("[t3-rust] pred shape={:?} dtype={:?}", pred.shape().dims(), pred.dtype());

    if pred.shape().dims() != target.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "shape mismatch: pred={:?} target={:?}",
            pred.shape().dims(), target.shape().dims()
        )));
    }

    // Loss in F32
    let pred_f32 = pred.to_dtype(DType::F32)?;
    let target_f32 = target.to_dtype(DType::F32)?;
    let diff = pred_f32.sub(&target_f32)?;
    let loss = diff.mul(&diff)?.mean()?;
    let loss_val = loss.to_vec()?[0];
    println!("[t3-rust] loss = {:.6}", loss_val);

    // Save (all F32 to match Python)
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("noisy".into(), noisy.to_dtype(DType::F32)?);
    out.insert("target".into(), target.to_dtype(DType::F32)?);
    out.insert("pred".into(), pred_f32);
    out.insert("loss".into(), loss);

    if let Some(parent) = std::path::Path::new(&args.out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    save_tensors(&out, std::path::Path::new(&args.out_path), SerializationFormat::SafeTensors)?;
    println!("[t3-rust] wrote {} → {}", out.len(), args.out_path);

    Ok(())
}
