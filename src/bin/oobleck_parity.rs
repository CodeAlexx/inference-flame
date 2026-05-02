//! Parity test: Stable Audio Oobleck VAE decoder Rust port vs Python reference.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType};
use inference_flame::models::sa_audio_vae::OobleckDecoder;

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/oobleck_decode_smoke.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/stable_audio_oobleck_vae.safetensors";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let sum: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).abs()).sum();
    (sum / a.len() as f64) as f32
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("--- Loading parity fixture ---");
    let fix = flame_core::serialization::load_file(Path::new(FIXTURE), &device)
        .map_err(|e| anyhow!("load fixture: {e}"))?;
    let latent = fix.get("latent_input").ok_or_else(|| anyhow!("missing latent_input"))?.clone();
    let reference = fix.get("decoded_output").ok_or_else(|| anyhow!("missing decoded_output"))?.clone();
    let t_lat = fix.get("meta.t_lat").unwrap().to_vec_f32().unwrap()[0] as usize;
    let seed = fix.get("meta.seed").unwrap().to_vec_f32().unwrap()[0] as usize;
    println!("  latent shape: {:?} dtype: {:?}", latent.shape().dims(), latent.dtype());
    println!("  reference shape: {:?} dtype: {:?}", reference.shape().dims(), reference.dtype());
    println!("  t_lat={t_lat} seed={seed}");

    println!("\n--- Loading OobleckDecoder ---");
    let dec = OobleckDecoder::load_default(VAE_PATH, &device).map_err(|e| anyhow!("load: {e}"))?;

    println!("\n--- Decode (Rust) ---");
    let z_f32 = latent.to_dtype(DType::F32)?;

    // Bisect
    let bisect_path = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/oobleck_bisect.safetensors";
    if Path::new(bisect_path).exists() {
        let bisect = flame_core::serialization::load_file(Path::new(bisect_path), &device)
            .map_err(|e| anyhow!("load bisect: {e}"))?;
        let stages = dec.decode_with_dumps(&z_f32).map_err(|e| anyhow!("dumps: {e}"))?;
        println!("\n--- Bisect ---");
        for (name, t) in &stages {
            if let Some(ref_t) = bisect.get(name) {
                let ours = t.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ours: {e}"))?;
                let theirs = ref_t.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("theirs: {e}"))?;
                if ours.len() != theirs.len() {
                    println!("  {name:>40}: SHAPE MISMATCH ours={} theirs={}", ours.len(), theirs.len());
                    continue;
                }
                let mxa = max_abs(&ours, &theirs);
                let cos = cosine(&ours, &theirs);
                println!("  {name:>40}: max_abs={mxa:.4} cos={cos:.4}  shape={:?}", t.shape().dims());
            } else {
                println!("  {name:>40}: (no reference)");
            }
        }
    }

    // DecoderBlock 1 sub-bisect
    let db1_path = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/oobleck_db1_bisect.safetensors";
    if Path::new(db1_path).exists() {
        let db1 = flame_core::serialization::load_file(Path::new(db1_path), &device)
            .map_err(|e| anyhow!("load db1: {e}"))?;
        let db1_in = db1.get("db1_input").ok_or_else(|| anyhow!("missing db1_input"))?.clone();
        println!("\n--- DecoderBlock 1 sub-bisect ---");
        let names = ["db1_after_sub_0_SnakeBeta", "db1_after_sub_1_ConvTranspose1d", "db1_after_sub_2_ResidualUnit", "db1_after_sub_3_ResidualUnit", "db1_after_sub_4_ResidualUnit"];
        let mut x = db1_in.clone();
        for (i, name) in names.iter().enumerate() {
            x = dec.db1_sub(i, &x).map_err(|e| anyhow!("db1_sub {i}: {e}"))?;
            if let Some(ref_t) = db1.get(*name) {
                let ours = x.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ours: {e}"))?;
                let theirs = ref_t.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("theirs: {e}"))?;
                let mxa = max_abs(&ours, &theirs);
                let cos = cosine(&ours, &theirs);
                println!("  sub {i} ({name}): max_abs={mxa:.4} cos={cos:.4} shape={:?}", x.shape().dims());
                println!("    ours[..5]:   {:?}", &ours[..5]);
                println!("    theirs[..5]: {:?}", &theirs[..5]);
            }
        }
    }

    // ResidualUnit inner sub-bisect
    let resu_path = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/oobleck_resu_bisect.safetensors";
    if Path::new(resu_path).exists() {
        let resu = flame_core::serialization::load_file(Path::new(resu_path), &device)
            .map_err(|e| anyhow!("load resu: {e}"))?;
        let resu_in = resu.get("resu_input").ok_or_else(|| anyhow!("missing resu_input"))?.clone();
        println!("\n--- ResidualUnit inner bisect ---");
        let names = ["resu_after_inner_0_SnakeBeta", "resu_after_inner_1_Conv1d", "resu_after_inner_2_SnakeBeta", "resu_after_inner_3_Conv1d"];
        let mut x = resu_in.clone();
        for (i, name) in names.iter().enumerate() {
            x = dec.db1_resu_inner(i, &x).map_err(|e| anyhow!("inner {i}: {e}"))?;
            if let Some(ref_t) = resu.get(*name) {
                let ours = x.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ours: {e}"))?;
                let theirs = ref_t.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("theirs: {e}"))?;
                let mxa = max_abs(&ours, &theirs);
                let cos = cosine(&ours, &theirs);
                println!("  inner {i} ({name}): max_abs={mxa:.4} cos={cos:.4}");
            }
        }
    }

    let out = dec.decode(&z_f32).map_err(|e| anyhow!("decode: {e}"))?;
    println!("\n  output shape: {:?} dtype: {:?}", out.shape().dims(), out.dtype());

    let ref_dims = reference.shape().dims().to_vec();
    let out_dims = out.shape().dims().to_vec();
    if ref_dims != out_dims {
        return Err(anyhow!("shape mismatch: ref {ref_dims:?} vs out {out_dims:?}"));
    }

    let ref_vec = reference.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("ref to_vec: {e}"))?;
    let out_vec = out.to_dtype(DType::F32)?.to_vec_f32().map_err(|e| anyhow!("out to_vec: {e}"))?;

    let mxa = max_abs(&out_vec, &ref_vec);
    let mna = mean_abs(&out_vec, &ref_vec);
    let cos = cosine(&out_vec, &ref_vec);

    println!("\n--- Parity ---");
    println!("  max_abs : {mxa:.6}");
    println!("  mean_abs: {mna:.6}");
    println!("  cos     : {cos:.6}");
    println!("  out[0..5]: {:?}", &out_vec[..5]);
    println!("  ref[0..5]: {:?}", &ref_vec[..5]);
    let n = out_vec.len();
    println!("  out[mid]: {:?}", &out_vec[n / 2..n / 2 + 5]);
    println!("  ref[mid]: {:?}", &ref_vec[n / 2..n / 2 + 5]);

    if mxa > 0.05 {
        return Err(anyhow!("PARITY FAIL: max_abs {mxa} > 0.05"));
    }
    if cos < 0.999 {
        return Err(anyhow!("PARITY FAIL: cos {cos} < 0.999"));
    }
    println!("\n  PARITY OK ✓");
    Ok(())
}
