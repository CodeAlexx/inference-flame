//! `wan_smoke` — standalone smoke test for the Wan2.2 TI2V-5B components
//! that we CAN test without the DiT weights (which aren't local).
//!
//! Verifies:
//!   1. `Umt5Encoder::load` + `encode("a cat walking")` on
//!      `/home/alex/.serenity/models/text_encoders/umt5_xxl_fp16.safetensors`,
//!      producing `[1, 512, 4096]` BF16 with no NaN/Inf.
//!   2. `Wan22VaeDecoder::load` on
//!      `/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors`, then
//!      `decode(random_latent)` on a small latent → RGB video tensor,
//!      shape check + no NaN/Inf.
//!
//! DiT / end-to-end sampling need the TI2V-5B checkpoint, which isn't
//! on this box. This smoke test is the most we can run today.

use std::path::Path;
use std::time::Instant;

use flame_core::{global_cuda_device, DType, Shape, Tensor};

use inference_flame::models::wan::{Umt5Encoder, Wan22VaeDecoder};

const UMT5_PATH: &str = "/home/alex/.serenity/models/text_encoders/umt5_xxl_fp16.safetensors";
const TOKENIZER_PATH: &str = "/home/alex/.serenity/models/upscalers/umt5-xxl/tokenizer.json";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors";

fn tensor_stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let data = t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = data.len();
    let nan = data.iter().filter(|v| v.is_nan()).count();
    let inf = data.iter().filter(|v| v.is_infinite()).count();
    let mean: f32 = data.iter().copied().sum::<f32>() / (n as f32);
    let abs_mean: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / (n as f32);
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &data {
        if v.is_finite() {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
    }
    println!(
        "  [{name}] shape={:?}  n={n}  nan={nan}  inf={inf}  mean={mean:.4}  |mean|={abs_mean:.4}  range=[{lo:.4}, {hi:.4}]",
        t.shape().dims(),
    );
    Ok(())
}

fn conv3d_isolated_test(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> anyhow::Result<()> {
    use flame_core::conv3d_simple::Conv3d;
    use flame_core::serialization::load_file_filtered;
    println!("=== Isolated time_conv test ===");
    // Load just the time_conv weights for decoder.upsamples.0.upsamples.3
    let weights = load_file_filtered(
        Path::new("/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"),
        device,
        |k| k.starts_with("decoder.upsamples.0.upsamples.3.time_conv."),
    )?;
    println!("  loaded {} weight tensors", weights.len());
    let w = weights.get("decoder.upsamples.0.upsamples.3.time_conv.weight").unwrap();
    let b = weights.get("decoder.upsamples.0.upsamples.3.time_conv.bias").unwrap();
    println!("  weight shape: {:?}  dtype: {:?}", w.shape().dims(), w.dtype());
    println!("  bias   shape: {:?}  dtype: {:?}", b.shape().dims(), b.dtype());

    // Build a Conv3d with exactly the time_conv geometry.
    // CausalConv3d(dim=1024, dim*2=2048, kernel=(3,1,1), padding=(1,0,0))
    // After CausalConv3d's static zero-pad, the conv sees a 5-D tensor and
    // applies temporal padding=0, so Conv3d has padding=(0,0,0).
    let mut conv = Conv3d::new(
        1024, 2048,
        (3, 1, 1),
        Some((1, 1, 1)),
        Some((0, 0, 0)),
        None, None, true, device.clone(),
    )?;
    conv.weight = w.clone().to_dtype(DType::F32)?;
    conv.bias_tensor = Some(b.clone().to_dtype(DType::F32)?);

    // Verify loaded weight stats match what Python sees.
    let wv = conv.weight.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let wn = wv.len() as f32;
    let w_mean = wv.iter().sum::<f32>() / wn;
    let w_abs = wv.iter().map(|x| x.abs()).sum::<f32>() / wn;
    let w_var = wv.iter().map(|x| (x - w_mean) * (x - w_mean)).sum::<f32>() / wn;
    println!(
        "  weight stats:  numel={} mean={:.6} |mean|={:.6} std={:.6}",
        wv.len(), w_mean, w_abs, w_var.sqrt()
    );
    println!("  (Python:   numel=6291456 mean=0.000159 |mean|=0.005522 std=0.017099)");

    // Build input: [1, 1024, 3, 26, 44] all 0.2.
    let numel = 1 * 1024 * 3 * 26 * 44;
    let data = vec![0.2f32; numel];
    let x = Tensor::from_vec(
        data,
        Shape::from_dims(&[1, 1024, 3, 26, 44]),
        device.clone(),
    )?;
    // Manually zero-pad on the left along dim=2 by 2 frames (match
    // CausalConv3d with padding=(1,0,0) → time_pad=2).
    let pad_shape = [1, 1024, 2, 26, 44];
    let pad = Tensor::zeros_dtype(
        Shape::from_dims(&pad_shape),
        DType::F32,
        device.clone(),
    )?;
    let x_padded = Tensor::cat(&[&pad, &x], 2)?;
    println!("  padded shape: {:?}", x_padded.shape().dims());

    let y = conv.forward(&x_padded)?;

    // Sanity test with known kernel: all-ones weight/bias=0.
    // Expected output: each element = sum(kernel) * sum(input_slice) = 3 * 1024 * 0.2 = 614.4
    // (three temporal positions, 1024 input channels, times 0.2). Bias adds.
    let mut sanity = Conv3d::new(
        1024, 2048,
        (3, 1, 1),
        Some((1, 1, 1)),
        Some((0, 0, 0)),
        None, None, true, device.clone(),
    )?;
    let wn = 2048 * 1024 * 3;
    sanity.weight = Tensor::from_vec(
        vec![1.0f32; wn],
        Shape::from_dims(&[2048, 1024, 3, 1, 1]),
        device.clone(),
    )?;
    sanity.bias_tensor = Some(Tensor::zeros_dtype(
        Shape::from_dims(&[2048]),
        DType::F32,
        device.clone(),
    )?);
    let xs = Tensor::from_vec(
        vec![0.2f32; 1 * 1024 * 3 * 26 * 44],
        Shape::from_dims(&[1, 1024, 3, 26, 44]),
        device.clone(),
    )?;
    // Zero-pad temporally by 2 on the left.
    let xs_pad = Tensor::cat(&[
        &Tensor::zeros_dtype(Shape::from_dims(&[1, 1024, 2, 26, 44]), DType::F32, device.clone())?,
        &xs,
    ], 2)?;
    let ys = sanity.forward(&xs_pad)?;
    let ys_vec = ys.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let ys_abs_mean = ys_vec.iter().map(|v| v.abs()).sum::<f32>() / ys_vec.len() as f32;
    let ys_max = ys_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!();
    println!("  SANITY conv (all-ones kernel, all-0.2 input):");
    println!("    shape={:?}  |mean|={ys_abs_mean:.2}  max={ys_max:.2}",
             ys.shape().dims());
    println!("    expected max ≈ 3*1024*0.2 = 614.4 (last temporal position)");
    println!("    expected first temp = 1*1024*0.2 = 204.8 (two of three taps hit zero-pad)");
    println!();

    let y_vec = y.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = y_vec.len() as f32;
    let mean: f32 = y_vec.iter().sum::<f32>() / n;
    let abs_mean: f32 = y_vec.iter().map(|x| x.abs()).sum::<f32>() / n;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &y_vec { if v < lo { lo = v; } if v > hi { hi = v; } }
    println!(
        "  RUST OUTPUT: shape={:?}  mean={mean:.4}  |mean|={abs_mean:.4}  range=[{lo:.4}, {hi:.4}]",
        y.shape().dims()
    );
    println!("  PY REFERENCE: |mean|=0.1271 range=[-0.4199, 0.6211]");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    if std::env::var("WAN_SMOKE_CONV_TEST").is_ok() {
        conv3d_isolated_test(&device)?;
        return Ok(());
    }

    // -------------------------------------------------------------------
    // Stage 1: UMT5-XXL
    // -------------------------------------------------------------------
    println!("=== Stage 1: UMT5-XXL ===");
    let t0 = Instant::now();
    let mut t5 = Umt5Encoder::load(Path::new(UMT5_PATH), &device)?;
    println!("  UMT5 loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Tokenize — match Python reference prompt exactly.
    let prompt = "a calico cat walking slowly across a sunlit wooden floor, cinematic";
    let pos_ids: Vec<i32> = {
        use tokenizers::Tokenizer;
        let tok = Tokenizer::from_file(Path::new(TOKENIZER_PATH))
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        enc.get_ids().iter().map(|&u| u as i32).collect()
    };
    println!("  prompt=\"{prompt}\"  tokens={}", pos_ids.len());
    println!("  rust token ids (first 20): {:?}", pos_ids.iter().take(20).collect::<Vec<_>>());

    let t1 = Instant::now();
    let cond = t5.encode(&pos_ids)?;
    println!("  encode in {:.1}s", t1.elapsed().as_secs_f32());
    tensor_stats("cond", &cond)?;

    // Print first few values of first non-pad token so we can diff
    // against the Python reference.
    let cond_vec = cond.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let stride = 4096;
    println!("  rust cond token 0 first 5: {:.4} {:.4} {:.4} {:.4} {:.4}",
        cond_vec[0], cond_vec[1], cond_vec[2], cond_vec[3], cond_vec[4]);
    let mut tok0_abs = 0f32;
    for v in &cond_vec[..stride] { tok0_abs += v.abs(); }
    println!("  rust cond token 0 abs mean: {:.4}", tok0_abs / stride as f32);

    // Dump if requested.
    if let Ok(out_path) = std::env::var("WAN_SMOKE_COND_OUT") {
        let mut m = std::collections::HashMap::new();
        m.insert("cond".to_string(), cond.clone());
        let _ = flame_core::serialization::save_file(&m, &out_path);
        println!("  saved cond → {out_path}");
    }

    // Shape: [1, 512, 4096]
    let d = cond.shape().dims();
    assert_eq!(d.len(), 3, "cond rank");
    assert_eq!(d[0], 1);
    assert_eq!(d[1], 512);
    assert_eq!(d[2], 4096);

    drop(t5);
    drop(cond);
    println!();

    // -------------------------------------------------------------------
    // Stage 2: Wan2.2 VAE
    // -------------------------------------------------------------------
    println!("=== Stage 2: Wan2.2 VAE (decode random latent) ===");
    let t2 = Instant::now();
    let vae = Wan22VaeDecoder::load(Path::new(VAE_PATH), &device)?;
    println!("  VAE loaded in {:.1}s", t2.elapsed().as_secs_f32());

    // Very small latent: 704×416, 9 frames.
    //   T' = (9 - 1) / 4 + 1 = 3
    //   H' = 416 / 16 = 26
    //   W' = 704 / 16 = 44
    // Channels = 48.
    let lat_t = 3;
    let lat_h = 26;
    let lat_w = 44;
    let z_dim = 48;
    let numel = 1 * z_dim * lat_t * lat_h * lat_w;
    println!("  latent shape: [1, {z_dim}, {lat_t}, {lat_h}, {lat_w}]  numel={numel}");

    // If an external latent file is provided, use that (for parity tests).
    // Otherwise, generate deterministic pseudo-Gaussian noise.
    let latent = if let Ok(ext_path) = std::env::var("WAN_SMOKE_LATENT") {
        println!("  loading external latent from {ext_path}");
        let map = flame_core::serialization::load_file(
            Path::new(&ext_path),
            &device,
        )?;
        let t = map
            .get("latent")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing 'latent' key"))?;
        if t.dtype() != DType::BF16 { t.to_dtype(DType::BF16)? } else { t }
    } else {
        let latent_data: Vec<f32> = {
            use rand::prelude::*;
            let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
            let mut v = Vec::with_capacity(numel);
            for _ in 0..numel / 2 {
                let u1: f32 = rng.gen::<f32>().max(1e-10);
                let u2: f32 = rng.gen::<f32>();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                v.push(r * theta.cos());
                v.push(r * theta.sin());
            }
            if numel % 2 == 1 {
                v.push(0.0);
            }
            v
        };
        Tensor::from_vec(
            latent_data,
            Shape::from_dims(&[1, z_dim, lat_t, lat_h, lat_w]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
    };

    let t3 = Instant::now();
    let rgb = vae.decode(&latent)?;
    println!("  decode in {:.1}s", t3.elapsed().as_secs_f32());
    tensor_stats("rgb", &rgb)?;

    // Dump frame 0 as PNG if WAN_SMOKE_FRAME=path is set.
    if let Ok(out_path) = std::env::var("WAN_SMOKE_FRAME") {
        let f32_data = rgb.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let dims = rgb.shape().dims().to_vec();
        let h_out = dims[3];
        let w_out = dims[4];
        let mut buf = vec![0u8; 3 * h_out * w_out];
        for c in 0..3 {
            for h in 0..h_out {
                for w in 0..w_out {
                    let src_idx = c * dims[2] * h_out * w_out + h * w_out + w;
                    let dst_idx = (h * w_out + w) * 3 + c;
                    let v = f32_data[src_idx].clamp(-1.0, 1.0);
                    buf[dst_idx] = (((v + 1.0) * 0.5 * 255.0).round() as i32).clamp(0, 255) as u8;
                }
            }
        }
        let img = image::RgbImage::from_raw(w_out as u32, h_out as u32, buf).unwrap();
        img.save(&out_path).ok();
        println!("  saved frame 0 → {out_path}");
    }

    // Expected output: [1, 3, (T'-0)*4, H'*32, W'*32]
    //   Wait: Wan 2.2 decoder upsamples temporal by 4 (groups 0 and 1 both
    //   do upsample3d ×2), spatial by 8 (groups 0, 1, 2 do upsample2d ×2),
    //   then unpatchify(2) multiplies spatial by 2 more → 16×.
    // So RGB shape: [1, 3, lat_t*4, lat_h*16, lat_w*16].
    let d = rgb.shape().dims();
    println!("  expected: [1, 3, {}, {}, {}]", lat_t * 4, lat_h * 16, lat_w * 16);
    assert_eq!(d.len(), 5);
    assert_eq!(d[0], 1);
    assert_eq!(d[1], 3);
    assert_eq!(d[2], lat_t * 4);
    assert_eq!(d[3], lat_h * 16);
    assert_eq!(d[4], lat_w * 16);

    println!();
    println!("SMOKE PASS  (UMT5 + Wan2.2 VAE standalone).");
    Ok(())
}
