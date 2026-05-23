//! L2P parity capture — Rust forward + per-layer intermediate dump.
//!
//! Phase B of the 4-phase parity test. Runs ONE `L2pDiT::forward_with_capture`
//! call with a fixed input matching `python_capture.py`, dumps the capture
//! map to safetensors for diff vs the Python reference.
//!
//! Usage:
//!   cargo run --release --bin l2p_parity_capture -- \
//!       --model /home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors \
//!       --sample /home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512/10.safetensors \
//!       --sigma 0.5 \
//!       --out /tmp/l2p_parity/rust_capture_sigma0.5.safetensors
//!
//! Reads `pixel` (CHW float, [3,512,512]) and `cap_feats` ([1,N,2560]) from
//! the sample. Casts both to BF16, reshapes pixel to [1,3,512,512], and
//! invokes the DiT in BF16 mode (matches Python script).

use cudarc::driver::CudaDevice;
use flame_core::serialization::{load_file, save_tensors, SerializationFormat};
use flame_core::{DType, Error, Result, Tensor};
use std::collections::HashMap;

use inference_flame::models::l2p::weight_loader::load_l2p_safetensors;
use inference_flame::models::l2p::L2pDiT;

struct Args {
    model_path: String,
    sample_path: String,
    sigma: f32,
    out_path: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut sample_path = String::new();
    let mut sigma = 0.5_f32;
    let mut out_path = String::new();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { i += 1; model_path = argv[i].clone(); }
            "--sample" => { i += 1; sample_path = argv[i].clone(); }
            "--sigma" => { i += 1; sigma = argv[i].parse().expect("bad sigma"); }
            "--out" => { i += 1; out_path = argv[i].clone(); }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    if model_path.is_empty() || sample_path.is_empty() || out_path.is_empty() {
        eprintln!("Usage: l2p_parity_capture --model <path> --sample <path> --sigma <f32> --out <path>");
        std::process::exit(1);
    }
    Args { model_path, sample_path, sigma, out_path }
}

fn main() -> Result<()> {
    // Match the inference binary's autograd-off discipline.
    if std::env::var("FLAME_AUTOGRAD_OFF").ok().as_deref() != Some("0") {
        flame_core::AutogradContext::set_enabled(false);
    }

    let args = parse_args();
    let device = CudaDevice::new(0).map_err(|e| Error::InvalidOperation(format!("CUDA init: {e:?}")))?;
    println!("[rust-capture] CUDA device ready");

    // ----- load model -----
    println!("[rust-capture] loading L2P weights: {}", args.model_path);
    let translated = load_l2p_safetensors(std::path::Path::new(&args.model_path), &device)?;
    println!("[rust-capture] translated {} keys", translated.len());
    let mut model = L2pDiT::new_resident(translated, device.clone());

    // ----- fixed input -----
    println!("[rust-capture] loading sample: {}", args.sample_path);
    let sample = load_file(&args.sample_path, &device)?;
    let pixel = sample.get("pixel").ok_or_else(|| {
        Error::InvalidInput("sample missing 'pixel'".into())
    })?;
    let cap_feats = sample.get("cap_feats").ok_or_else(|| {
        Error::InvalidInput("sample missing 'cap_feats'".into())
    })?;
    println!("[rust-capture]   pixel shape: {:?} dtype: {:?}", pixel.shape().dims(), pixel.dtype());
    println!("[rust-capture]   cap_feats shape: {:?} dtype: {:?}", cap_feats.shape().dims(), cap_feats.dtype());

    // Reshape pixel to [1, 3, 512, 512] BF16
    let p_dims = pixel.shape().dims();
    let pixel_bf16 = pixel.to_dtype(DType::BF16)?;
    let x = if p_dims.len() == 3 {
        pixel_bf16.reshape(&[1, p_dims[0], p_dims[1], p_dims[2]])?
    } else {
        pixel_bf16
    };

    // cap_feats: ensure [B, S, 2560] BF16
    let cap = cap_feats.to_dtype(DType::BF16)?;
    let c_dims = cap.shape().dims();
    let cap_3d = if c_dims.len() == 2 {
        cap.reshape(&[1, c_dims[0], c_dims[1]])?
    } else {
        cap
    };

    println!("[rust-capture]   x (model input) shape: {:?} dtype: {:?}", x.shape().dims(), x.dtype());
    println!("[rust-capture]   cap_3d shape: {:?} dtype: {:?}", cap_3d.shape().dims(), cap_3d.dtype());

    // Timestep: model expects sigma in [0,1] per L2pDiT convention.
    let t_tensor = Tensor::from_vec_dtype(
        vec![args.sigma],
        flame_core::Shape::from_dims(&[1]),
        device.clone(),
        DType::BF16,
    )?;
    println!("[rust-capture] sigma input = {}  (DiT will compute t_scaled = (1 - sigma) * 1000 = {})",
        args.sigma, (1.0 - args.sigma) * 1000.0);

    // ----- forward with capture -----
    let mut capture: HashMap<String, Tensor> = HashMap::new();
    let _ = model.forward_with_capture(&x, &t_tensor, &cap_3d, &mut capture)?;
    println!("[rust-capture] captured {} tensors", capture.len());

    // ----- save -----
    if let Some(parent) = std::path::Path::new(&args.out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    // Save as F32 to match Python capture dtype, so diff doesn't double-quantize.
    let mut save_map: HashMap<String, Tensor> = HashMap::new();
    for (k, v) in capture.iter() {
        let f = if v.dtype() == DType::F32 { v.clone() } else { v.to_dtype(DType::F32)? };
        save_map.insert(k.clone(), f);
    }
    save_tensors(&save_map, std::path::Path::new(&args.out_path), SerializationFormat::SafeTensors)?;
    println!("[rust-capture] wrote {} tensors → {}", save_map.len(), args.out_path);

    // Print summary table.
    let mut keys: Vec<&String> = save_map.keys().collect();
    keys.sort();
    for k in keys {
        let t = &save_map[k];
        let v = t.to_vec()?;
        let mut amax = 0.0f32;
        let mut amean = 0.0f64;
        for x in &v {
            let a = x.abs();
            if a > amax { amax = a; }
            amean += a as f64;
        }
        amean /= v.len().max(1) as f64;
        println!("  {:<32}  shape={:?}  abs.mean={:.4e}  abs.max={:.4e}",
            k, t.shape().dims(), amean as f32, amax);
    }

    Ok(())
}
