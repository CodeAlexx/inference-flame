//! Op-level microbenchmark — flame-core BF16 GPU.
//!
//! Times every primitive at the exact tensor shapes our models hit during
//! 1024² inference. No model loads, no I/O. 20 iters per op, first 5
//! discarded, median reported in ms.
//!
//! Run with:
//!   LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib \
//!     ./target/release/bench_flame

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;
use flame_core::{DType, Result, Shape, Tensor};

const WARMUP: usize = 5;
const ITERS: usize = 15;

fn dev_sync(dev: &Arc<CudaDevice>) {
    let _ = dev.synchronize();
}

fn bench<F: FnMut() -> Result<()>>(name: &str, dev: &Arc<CudaDevice>, mut f: F) {
    for _ in 0..WARMUP {
        if let Err(e) = f() {
            eprintln!("{name}: warmup error {e:?}");
            return;
        }
    }
    dev_sync(dev);

    let mut times: Vec<f64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        dev_sync(dev);
        let t0 = Instant::now();
        if let Err(e) = f() {
            eprintln!("{name}: iter error {e:?}");
            return;
        }
        dev_sync(dev);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    println!("{:32}  {:8.2} ms", name, median);
}

fn rb(shape: &[usize], dev: &Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, dev.clone())?;
    if t.dtype() != DType::BF16 {
        t.to_dtype(DType::BF16)
    } else {
        Ok(t)
    }
}

fn main() -> Result<()> {
    flame_core::AutogradContext::set_enabled(false);

    let dev = CudaDevice::new(0)
        .map_err(|e| flame_core::Error::InvalidOperation(format!("CUDA: {e:?}")))?;

    println!("# flame-core, BF16 on CUDA device 0");
    println!("# {WARMUP} warmup + {ITERS} iters, median ms");
    println!();

    // ── 1. GEMM (2D) ────────────────────────────────────────────────────────
    println!("# GEMM (2D)");
    {
        let a = rb(&[4096, 3072], &dev)?;
        let b = rb(&[3072, 3072], &dev)?;
        bench("gemm_small", &dev, || {
            let _ = a.matmul(&b)?;
            Ok(())
        });
    }
    {
        let a = rb(&[4096, 3840], &dev)?;
        let b = rb(&[3840, 3840], &dev)?;
        bench("gemm_medium", &dev, || {
            let _ = a.matmul(&b)?;
            Ok(())
        });
    }
    {
        let a = rb(&[4096, 3840], &dev)?;
        let b = rb(&[3840, 15360], &dev)?;
        bench("gemm_large", &dev, || {
            let _ = a.matmul(&b)?;
            Ok(())
        });
    }
    {
        let a = rb(&[4608, 3072], &dev)?;
        let b = rb(&[3072, 9216], &dev)?;
        bench("gemm_flux_qkv", &dev, || {
            let _ = a.matmul(&b)?;
            Ok(())
        });
    }
    {
        let a = rb(&[4608, 3072], &dev)?;
        let b = rb(&[3072, 12288], &dev)?;
        bench("gemm_flux_mlp", &dev, || {
            let _ = a.matmul(&b)?;
            Ok(())
        });
    }
    println!();

    // ── 1b. GEMM via fused_linear3d (the model's actual hot path) ──────────
    println!("# GEMM via fused_linear3d_native (PyTorch [Cout,Cin] weight)");
    {
        let x = rb(&[1, 4608, 3072], &dev)?;
        let w = rb(&[9216, 3072], &dev)?;
        let bias = rb(&[9216], &dev)?;
        bench("linear3d_flux_qkv", &dev, || {
            let _ = flame_core::ops::fused_inference::fused_linear3d_native(
                &x,
                &w,
                Some(&bias),
            )?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 4608, 3072], &dev)?;
        let w = rb(&[12288, 3072], &dev)?;
        let bias = rb(&[12288], &dev)?;
        bench("linear3d_flux_mlp_up", &dev, || {
            let _ = flame_core::ops::fused_inference::fused_linear3d_native(
                &x,
                &w,
                Some(&bias),
            )?;
            Ok(())
        });
    }
    {
        // Single block linear1: 3072 → 21504 (the worst-case in FLUX profile)
        let x = rb(&[1, 4608, 3072], &dev)?;
        let w = rb(&[21504, 3072], &dev)?;
        let bias = rb(&[21504], &dev)?;
        bench("linear3d_flux_single1", &dev, || {
            let _ = flame_core::ops::fused_inference::fused_linear3d_native(
                &x,
                &w,
                Some(&bias),
            )?;
            Ok(())
        });
    }
    {
        // Single block linear2: 15360 → 3072
        let x = rb(&[1, 4608, 15360], &dev)?;
        let w = rb(&[3072, 15360], &dev)?;
        let bias = rb(&[3072], &dev)?;
        bench("linear3d_flux_single2", &dev, || {
            let _ = flame_core::ops::fused_inference::fused_linear3d_native(
                &x,
                &w,
                Some(&bias),
            )?;
            Ok(())
        });
    }
    println!();

    // ── 2. SDPA ─────────────────────────────────────────────────────────────
    println!("# SDPA");
    {
        let q = rb(&[1, 30, 4096, 128], &dev)?;
        let k = rb(&[1, 30, 4096, 128], &dev)?;
        let v = rb(&[1, 30, 4096, 128], &dev)?;
        bench("sdpa_zimage", &dev, || {
            let _ = flame_core::attention::sdpa(&q, &k, &v, None)?;
            Ok(())
        });
    }
    {
        let q = rb(&[1, 24, 4608, 128], &dev)?;
        let k = rb(&[1, 24, 4608, 128], &dev)?;
        let v = rb(&[1, 24, 4608, 128], &dev)?;
        bench("sdpa_flux", &dev, || {
            let _ = flame_core::attention::sdpa(&q, &k, &v, None)?;
            Ok(())
        });
    }
    {
        let q = rb(&[1, 24, 1536, 128], &dev)?;
        let k = rb(&[1, 24, 1536, 128], &dev)?;
        let v = rb(&[1, 24, 1536, 128], &dev)?;
        bench("sdpa_klein", &dev, || {
            let _ = flame_core::attention::sdpa(&q, &k, &v, None)?;
            Ok(())
        });
    }
    {
        let q = rb(&[1, 32, 512, 64], &dev)?;
        let k = rb(&[1, 32, 512, 64], &dev)?;
        let v = rb(&[1, 32, 512, 64], &dev)?;
        bench("sdpa_t5", &dev, || {
            let _ = flame_core::attention::sdpa(&q, &k, &v, None)?;
            Ok(())
        });
    }
    println!();

    // ── 3. RMSNorm ──────────────────────────────────────────────────────────
    println!("# RMSNorm");
    {
        let x = rb(&[512, 4096], &dev)?;
        let w = rb(&[4096], &dev)?;
        bench("rmsnorm_small", &dev, || {
            let _ = flame_core::cuda_ops_bf16::rms_norm_bf16(&x, Some(&w), 1e-6)?;
            Ok(())
        });
    }
    {
        let x = rb(&[4096, 3840], &dev)?;
        let w = rb(&[3840], &dev)?;
        bench("rmsnorm_medium", &dev, || {
            let _ = flame_core::cuda_ops_bf16::rms_norm_bf16(&x, Some(&w), 1e-6)?;
            Ok(())
        });
    }
    {
        let x = rb(&[4608, 3072], &dev)?;
        let w = rb(&[3072], &dev)?;
        bench("rmsnorm_large", &dev, || {
            let _ = flame_core::cuda_ops_bf16::rms_norm_bf16(&x, Some(&w), 1e-6)?;
            Ok(())
        });
    }
    println!();

    // ── 4. LayerNorm ────────────────────────────────────────────────────────
    println!("# LayerNorm");
    {
        let x = rb(&[512, 4096], &dev)?;
        bench("layernorm_small", &dev, || {
            let _ = flame_core::cuda_ops_bf16::layer_norm_bf16(&x, None, None, 1e-6)?;
            Ok(())
        });
    }
    {
        let x = rb(&[4608, 3072], &dev)?;
        bench("layernorm_flux", &dev, || {
            let _ = flame_core::cuda_ops_bf16::layer_norm_bf16(&x, None, None, 1e-6)?;
            Ok(())
        });
    }
    println!();

    // ── 5. Elementwise ──────────────────────────────────────────────────────
    println!("# Elementwise");
    {
        let x = rb(&[1, 4096, 15360], &dev)?;
        bench("silu", &dev, || {
            let _ = x.silu()?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 4608, 12288], &dev)?;
        bench("gelu", &dev, || {
            let _ = x.gelu()?;
            Ok(())
        });
    }
    {
        let a = rb(&[1, 4096, 3840], &dev)?;
        let b = rb(&[1, 4096, 3840], &dev)?;
        bench("mul", &dev, || {
            let _ = a.mul(&b)?;
            Ok(())
        });
        bench("add", &dev, || {
            let _ = a.add(&b)?;
            Ok(())
        });
        bench("mul_scalar", &dev, || {
            let _ = a.mul_scalar(0.5)?;
            Ok(())
        });
    }
    println!();

    // ── 6. Permute / Transpose ─────────────────────────────────────────────
    println!("# Permute / Transpose");
    {
        let x = rb(&[1, 4096, 30, 128], &dev)?;
        bench("permute_0213", &dev, || {
            let _ = x.permute(&[0, 2, 1, 3])?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 3840, 4096], &dev)?;
        bench("permute_021", &dev, || {
            let _ = x.permute(&[0, 2, 1])?;
            Ok(())
        });
    }
    {
        let x = rb(&[3840, 3840], &dev)?;
        bench("transpose_2d", &dev, || {
            let _ = flame_core::bf16_elementwise::transpose2d_bf16(&x)?;
            Ok(())
        });
    }
    println!();

    // ── 7. Softmax ──────────────────────────────────────────────────────────
    println!("# Softmax");
    {
        let x = rb(&[30, 4096, 64], &dev)?;
        bench("softmax_small", &dev, || {
            let _ = x.softmax(-1)?;
            Ok(())
        });
    }
    {
        let x = rb(&[30, 4096, 4096], &dev)?;
        bench("softmax_large", &dev, || {
            let _ = x.softmax(-1)?;
            Ok(())
        });
    }
    println!();

    // ── 8. Cast ─────────────────────────────────────────────────────────────
    println!("# Cast");
    {
        let x = rb(&[1, 4096, 3840], &dev)?;
        bench("cast_bf16_f32", &dev, || {
            let _ = x.to_dtype(DType::F32)?;
            Ok(())
        });
    }
    {
        let x = Tensor::randn(
            Shape::from_dims(&[1, 4096, 3840]),
            0.0,
            1.0,
            dev.clone(),
        )?
        .to_dtype(DType::F32)?;
        bench("cast_f32_bf16", &dev, || {
            let _ = x.to_dtype(DType::BF16)?;
            Ok(())
        });
    }
    println!();

    // ── 9. GroupNorm 2D ─────────────────────────────────────────────────────
    // flame-core's group_norm_bf16 takes NHWC layout, while PyTorch is NCHW.
    // Bench it as the model would call it (after a permute), so we measure
    // the apples-to-apples cost. The permute is timed separately above.
    println!("# GroupNorm 2D (NHWC in flame-core)");
    {
        let x = rb(&[1, 128, 128, 512], &dev)?; // NHWC for [N=1,H=128,W=128,C=512]
        let w = rb(&[512], &dev)?;
        let b = rb(&[512], &dev)?;
        bench("groupnorm_vae_small", &dev, || {
            let _ = flame_core::cuda_ops_bf16::group_norm_bf16(
                &x, Some(&w), Some(&b), 32, 1e-6,
            )?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 1024, 1024, 128], &dev)?;
        let w = rb(&[128], &dev)?;
        let b = rb(&[128], &dev)?;
        bench("groupnorm_vae_large", &dev, || {
            let _ = flame_core::cuda_ops_bf16::group_norm_bf16(
                &x, Some(&w), Some(&b), 32, 1e-6,
            )?;
            Ok(())
        });
    }
    println!();

    // ── 10. Conv2d ──────────────────────────────────────────────────────────
    println!("# Conv2d (NCHW)");
    {
        let x = rb(&[1, 512, 128, 128], &dev)?;
        let w = rb(&[512, 512, 3, 3], &dev)?;
        let conv = flame_core::conv::Conv2d::new_with_bias_zeroed(
            512, 512, 3, 1, 1, dev.clone(), true,
        )?;
        // Replace the zeroed weight with random for fair timing.
        let _ = w; // already a tensor; conv.forward will use its own weights
        bench("conv2d_vae_mid", &dev, || {
            let _ = conv.forward(&x)?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 128, 1024, 1024], &dev)?;
        let conv = flame_core::conv::Conv2d::new_with_bias_zeroed(
            128, 128, 3, 1, 1, dev.clone(), true,
        )?;
        bench("conv2d_vae_up", &dev, || {
            let _ = conv.forward(&x)?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 512, 128, 128], &dev)?;
        let conv = flame_core::conv::Conv2d::new_with_bias_zeroed(
            512, 512, 1, 1, 0, dev.clone(), true,
        )?;
        bench("conv2d_vae_1x1", &dev, || {
            let _ = conv.forward(&x)?;
            Ok(())
        });
    }
    println!();

    // ── 12. Conv3d ──────────────────────────────────────────────────────────
    println!("# Conv3d (video VAE / video DiT)");
    {
        let x = rb(&[1, 128, 16, 64, 64], &dev)?;
        let w = rb(&[128, 128, 3, 3, 3], &dev)?;
        let bias = rb(&[128], &dev)?;
        let conv = flame_core::conv3d_bf16::Conv3dBF16::from_weights(
            w, Some(bias), (1, 1, 1), (1, 1, 1),
        );
        bench("conv3d_vae_temporal", &dev, || {
            let _ = conv.forward(&x)?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 512, 4, 32, 32], &dev)?;
        let w = rb(&[512, 512, 3, 3, 3], &dev)?;
        let bias = rb(&[512], &dev)?;
        let conv = flame_core::conv3d_bf16::Conv3dBF16::from_weights(
            w, Some(bias), (1, 1, 1), (1, 1, 1),
        );
        bench("conv3d_vae_small", &dev, || {
            let _ = conv.forward(&x)?;
            Ok(())
        });
    }
    {
        let x = rb(&[1, 128, 16, 64, 64], &dev)?;
        let w = rb(&[128, 128, 3, 3, 3], &dev)?;
        let bias = rb(&[128], &dev)?;
        let conv = flame_core::conv3d_bf16::Conv3dBF16::from_weights(
            w, Some(bias), (1, 1, 1), (2, 1, 1),
        );
        bench("conv3d_causal", &dev, || {
            let _ = conv.forward(&x)?;
            Ok(())
        });
    }
    println!();

    // ── 13. SDPA causal ─────────────────────────────────────────────────────
    println!("# SDPA causal (video temporal attention)");
    {
        let q = rb(&[1, 32, 768, 128], &dev)?;
        let k = rb(&[1, 32, 768, 128], &dev)?;
        let v = rb(&[1, 32, 768, 128], &dev)?;
        bench("sdpa_causal_ltx", &dev, || {
            let _ = flame_core::cuda_ops_bf16::sdpa_stream_bf16(
                &q, &k, &v, None, 768, true, None,
            )?;
            Ok(())
        });
    }
    {
        let q = rb(&[1, 40, 1024, 128], &dev)?;
        let k = rb(&[1, 40, 1024, 128], &dev)?;
        let v = rb(&[1, 40, 1024, 128], &dev)?;
        bench("sdpa_causal_wan", &dev, || {
            let _ = flame_core::cuda_ops_bf16::sdpa_stream_bf16(
                &q, &k, &v, None, 1024, true, None,
            )?;
            Ok(())
        });
    }
    println!();

    // ── 14. SDPA + additive bias ───────────────────────────────────────────
    println!("# SDPA + additive bias (T5 relative position)");
    {
        let q = rb(&[1, 32, 512, 64], &dev)?;
        let k = rb(&[1, 32, 512, 64], &dev)?;
        let v = rb(&[1, 32, 512, 64], &dev)?;
        let bias = rb(&[1, 32, 512, 512], &dev)?;
        bench("sdpa_bias_t5", &dev, || {
            let _ = flame_core::attention::sdpa_with_bias(
                &q, &k, &v, Some(&bias), Some(1.0),
            )?;
            Ok(())
        });
    }
    println!();

    // ── 15. 3D RoPE ─────────────────────────────────────────────────────────
    println!("# 3D RoPE  -- MISSING in flame-core (no rope_3d primitive)");
    println!("{:32}  {:>8}", "rope_3d_ltx", "MISSING");
    println!("{:32}  {:>8}", "rope_3d_wan", "MISSING");
    println!();

    // ── 16. 3D upsample ─────────────────────────────────────────────────────
    println!("# 3D upsample  -- MISSING in flame-core (only 2D upsample)");
    println!("{:32}  {:>8}", "upsample_3d_nearest", "MISSING");
    println!("{:32}  {:>8}", "upsample_3d_spatial", "MISSING");
    println!();

    // ── 17. Temporal blend ──────────────────────────────────────────────────
    println!("# Temporal blend  -- MISSING in flame-core (no overlap-add helper)");
    println!("{:32}  {:>8}", "temporal_blend", "MISSING");
    println!();

    // ── 18. 3D GroupNorm ────────────────────────────────────────────────────
    println!("# GroupNorm 3D  -- MISSING in flame-core (group_norm_bf16 is 4D only)");
    println!("{:32}  {:>8}", "groupnorm3d_small", "MISSING");
    println!("{:32}  {:>8}", "groupnorm3d_large", "MISSING");
    println!();

    Ok(())
}
