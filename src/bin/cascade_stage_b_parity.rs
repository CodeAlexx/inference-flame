//! Stage B step-0 parity: compare Rust forward and intermediate-level outputs
//! against the Python reference dumps.

use std::collections::HashMap;
use std::path::PathBuf;

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::wuerstchen_unet::{WuerstchenUNet, WuerstchenUNetConfig};

const CKPT_ROOT: &str =
    "/home/alex/.cache/huggingface/hub/models--stabilityai--stable-cascade/snapshots/a89f66d459ae653e3b4d4f992a7c3789d0dc4d16";

fn stage_b_path() -> PathBuf { PathBuf::from(CKPT_ROOT).join("stage_b_bf16.safetensors") }
fn ref_path() -> PathBuf {
    PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/cascade_gen/stage_b_ref_step0.safetensors")
}
fn int_path() -> PathBuf {
    PathBuf::from("/home/alex/EriDiffusion/inference-flame/output/cascade_gen/stage_b_intermediates.safetensors")
}

fn cos(a: &Tensor, b: &Tensor) -> anyhow::Result<(f32, f32, f32)> {
    if a.shape().dims() != b.shape().dims() {
        eprintln!("    shape mismatch: rust {:?} vs ref {:?}", a.shape().dims(), b.shape().dims());
        return Ok((0.0, f32::INFINITY, f32::INFINITY));
    }
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut dot=0f64; let mut na=0f64; let mut nb=0f64; let mut mxd=0f32; let mut mxr=0f32;
    for (x,y) in av.iter().zip(bv.iter()) {
        let xd=*x as f64; let yd=*y as f64;
        dot+=xd*yd; na+=xd*xd; nb+=yd*yd;
        let d=(x-y).abs(); if d>mxd {mxd=d;}
        if y.abs()>mxr {mxr=y.abs();}
    }
    let c = if na>0.0 && nb>0.0 {(dot/(na.sqrt()*nb.sqrt())) as f32} else {0.0};
    let rel = if mxr>0.0 {mxd/mxr} else {mxd};
    Ok((c, mxd, rel))
}

fn stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
    let n = v.len() as f32;
    let m: f32 = v.iter().sum::<f32>() / n;
    let s = (v.iter().map(|x|(x-m).powi(2)).sum::<f32>() / n).sqrt();
    let mx = v.iter().cloned().fold(f32::MIN, f32::max);
    let mn = v.iter().cloned().fold(f32::MAX, f32::min);
    println!("  [{name}] {:?} mean={:.4} std={:.4} min={:.4} max={:.4}",
             t.shape().dims(), m, s, mn, mx);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::try_init().ok();
    let device = global_cuda_device();

    println!("=== Stage B step-0 parity ===");
    let ref_w = flame_core::serialization::load_file(ref_path(), &device)?;
    let x = ref_w.get("x").unwrap().clone();
    let effnet = ref_w.get("effnet").unwrap().clone();
    let text_pooled = ref_w.get("text_pooled").unwrap().squeeze(Some(1))?;
    let v_cond_ref = ref_w.get("v_cond").unwrap().clone();

    let int_map: Option<HashMap<String, Tensor>> = if int_path().exists() {
        Some(flame_core::serialization::load_file(int_path(), &device)?)
    } else { None };

    println!("Loading Stage B UNet...");
    let unet = WuerstchenUNet::load(
        stage_b_path().to_str().unwrap(),
        WuerstchenUNetConfig::stage_b(),
        &device,
    )?;
    println!("  loaded.");

    let mut dumps: HashMap<String, Tensor> = HashMap::new();
    let v_cond_rust = unet.forward_with_dumps(&x, 1.0, Some(&text_pooled), None, Some(&effnet), Some(&mut dumps))?;
    println!();
    stats("v_cond_rust", &v_cond_rust)?;
    let (c, md, mr) = cos(&v_cond_rust, &v_cond_ref)?;
    println!("Final v_cond: cos_sim={c:.6} max_abs_diff={md:.4e} max_rel={mr:.4e}");

    if let Some(int) = int_map {
        println!();
        println!("--- Per-stage comparison ---");
        let keys = [
            "emb",
            "down_0_after_ds", "down_0_out",
            "down_1_after_ds", "down_1_out",
            "down_2_after_ds", "down_2_out",
            "down_3_after_ds", "down_3_out",
            "up_0_before_us", "up_0_after_us",
            "up_1_before_us", "up_1_after_us",
            "up_2_before_us", "up_2_after_us",
            "up_3_before_us", "up_3_after_us",
            "out",
        ];
        for k in keys.iter() {
            let r_t = dumps.get(*k);
            let ref_t = int.get(*k);
            match (r_t, ref_t) {
                (Some(rt), Some(rf)) => {
                    let (c, md, mr) = cos(rt, rf)?;
                    let v = rt.to_dtype(DType::F32)?.to_vec_f32()?;
                    let n = v.len() as f32;
                    let mn: f32 = v.iter().sum::<f32>() / n;
                    let st = (v.iter().map(|x|(x-mn).powi(2)).sum::<f32>() / n).sqrt();
                    let rv = rf.to_dtype(DType::F32)?.to_vec_f32()?;
                    let rmn: f32 = rv.iter().sum::<f32>() / (rv.len() as f32);
                    let rst = (rv.iter().map(|x|(x-rmn).powi(2)).sum::<f32>() / (rv.len() as f32)).sqrt();
                    println!("  {k:20} cos={c:.6}  rel={mr:.3e}  rust_std={st:.3}  ref_std={rst:.3}");
                }
                (Some(_), None) => println!("  {k:20} (no ref)"),
                (None, Some(_)) => println!("  {k:20} (no rust dump)"),
                (None, None) => {}
            }
        }
    }

    println!();
    println!("=== Summary: cos_sim={c:.6} {}", if c >= 0.99 {"OK"} else {"FAIL"});
    let _ = device;
    Ok(())
}
