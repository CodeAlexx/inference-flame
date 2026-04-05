//! Dump RoPE values for comparison against Python reference.
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::ltx2_model::compute_rope_frequencies;

fn main() -> anyhow::Result<()> {
    let device = global_cuda_device();

    let lat_f = 2usize;
    let lat_h = 9usize;
    let lat_w = 15usize;
    let num_tokens = lat_f * lat_h * lat_w; // 270
    let vae_sf = [8usize, 32, 32];
    let causal_offset = 1usize;
    let frame_rate = 25.0f32;

    // Build coordinates (same as build_video_coords)
    let batch_size = 1;
    let mut coords_data = vec![0.0f32; batch_size * 3 * num_tokens * 2];
    for f in 0..lat_f {
        for h in 0..lat_h {
            for w in 0..lat_w {
                let token_idx = f * lat_h * lat_w + h * lat_w + w;
                let f_start = (f * vae_sf[0]) as f32;
                let f_end = ((f + 1) * vae_sf[0]) as f32;
                let h_start = (h * vae_sf[1]) as f32;
                let h_end = ((h + 1) * vae_sf[1]) as f32;
                let w_start = (w * vae_sf[2]) as f32;
                let w_end = ((w + 1) * vae_sf[2]) as f32;

                let vae_t = vae_sf[0] as f32;
                let f_start_c = (f_start + causal_offset as f32 - vae_t).max(0.0);
                let f_end_c = (f_end + causal_offset as f32 - vae_t).max(0.0);
                let f_start_s = f_start_c / frame_rate;
                let f_end_s = f_end_c / frame_rate;

                coords_data[0 * num_tokens * 2 + token_idx * 2] = f_start_s;
                coords_data[0 * num_tokens * 2 + token_idx * 2 + 1] = f_end_s;
                coords_data[1 * num_tokens * 2 + token_idx * 2] = h_start;
                coords_data[1 * num_tokens * 2 + token_idx * 2 + 1] = h_end;
                coords_data[2 * num_tokens * 2 + token_idx * 2] = w_start;
                coords_data[2 * num_tokens * 2 + token_idx * 2 + 1] = w_end;
            }
        }
    }

    let coords = Tensor::from_vec_dtype(
        coords_data,
        Shape::from_dims(&[1, 3, num_tokens, 2]),
        device.clone(),
        DType::F32,
    )?;

    let (cos, sin) = compute_rope_frequencies(
        &coords, 4096, &[20.0, 2048.0, 2048.0], 10000.0, 32,
    )?;

    println!("cos: {:?} {:?}", cos.shape().dims(), cos.dtype());
    println!("sin: {:?} {:?}", sin.shape().dims(), sin.dtype());

    let cos_data = cos.to_vec()?;
    let sin_data = sin.to_vec()?;

    // Token 0, Head 0 (first 64 values = head 0's RoPE)
    let hd = 64; // head_dim/2
    println!("\nToken 0, Head 0:");
    println!("  cos[:8] = {:?}", &cos_data[..8]);
    println!("  sin[:8] = {:?}", &sin_data[..8]);

    // Token 135 (f=1), Head 0
    let off = 135 * hd;
    println!("Token 135 (f=1), Head 0:");
    println!("  cos[:8] = {:?}", &cos_data[off..off+8]);
    println!("  sin[:8] = {:?}", &sin_data[off..off+8]);

    // Token 1 (w=1), Head 0
    let off = 1 * hd;
    println!("Token 1 (w=1), Head 0:");
    println!("  cos[:8] = {:?}", &cos_data[off..off+8]);
    println!("  sin[:8] = {:?}", &sin_data[off..off+8]);

    // Token 15 (h=1), Head 0
    let off = 15 * hd;
    println!("Token 15 (h=1), Head 0:");
    println!("  cos[:8] = {:?}", &cos_data[off..off+8]);
    println!("  sin[:8] = {:?}", &sin_data[off..off+8]);

    // Head 31, Token 0
    // cos shape is [B, H, N, D_head/2] = [1, 32, 270, 64]
    // Flatten: offset = head*N*hd + token*hd
    let off = 31 * num_tokens * hd + 0 * hd;
    println!("Token 0, Head 31:");
    println!("  cos[:8] = {:?}", &cos_data[off..off+8]);
    println!("  sin[:8] = {:?}", &sin_data[off..off+8]);

    // Count padding
    let mut n_pad = 0;
    for &v in &cos_data[..hd] {
        if (v - 1.0).abs() < 0.01 { n_pad += 1; } else { break; }
    }
    println!("\nPadding: {} leading 1s (of {} per head)", n_pad, hd);

    Ok(())
}
