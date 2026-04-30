//! Element-wise compare Wan UMT5 prompt embeds vs diffusers reference.
//!
//! Loads tests/pytorch_fixtures/helios/diffusers_prompt_embeds.safetensors,
//! runs our `Umt5Encoder::encode` on the same token ids, compares.

use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::wan::t5::Umt5Encoder;

const FIXTURE: &str = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/helios/diffusers_prompt_embeds.safetensors";
const UMT5_PATH: &str = "/home/alex/.cache/huggingface/hub/models--BestWishYsh--Helios-Distilled/snapshots/1999182614cb08d3bdcc46b9827504af2914b87b/text_encoder/model.safetensors";

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    println!("--- Loading diffusers fixture ---");
    let map = flame_core::serialization::load_file(Path::new(FIXTURE), &device)
        .map_err(|e| anyhow!("load fixture: {e}"))?;

    let token_ids = map.get("token_ids").ok_or_else(|| anyhow!("missing token_ids"))?.clone();
    let pe_padded = map.get("prompt_embeds").ok_or_else(|| anyhow!("missing prompt_embeds"))?.clone();
    let pe_raw = map.get("prompt_embeds_raw").ok_or_else(|| anyhow!("missing prompt_embeds_raw"))?.clone();
    let real_seq_len = map.get("meta.real_seq_len").unwrap().to_vec_f32().unwrap()[0] as usize;
    println!("  token_ids shape: {:?}", token_ids.shape().dims());
    println!("  prompt_embeds_padded shape: {:?}", pe_padded.shape().dims());
    println!("  real_seq_len: {}", real_seq_len);

    // Extract token ids as i32 vec (only the real tokens).
    let token_ids_i32: Vec<i32> = {
        let v = token_ids.to_vec_f32().map_err(|e| anyhow!("token_ids to_vec_f32: {e}"))?;
        v[..real_seq_len].iter().map(|&f| f as i32).collect()
    };
    println!("  real ids: {:?}", token_ids_i32);

    println!("\n--- Loading Wan UMT5 ---");
    let mut umt5 = Umt5Encoder::load(Path::new(UMT5_PATH), &device)
        .map_err(|e| anyhow!("Umt5Encoder::load: {e}"))?;

    // Diffusers reference raw_embed for token id 320:
    //   [-3.734, -6.0625, -0.106, -1.578, -0.394] magnitude 3.56
    // After layer 0:
    //   [56.75, -10.875, -3.8125, 0.555, 1.836] magnitude 8.19
    // Final encoder output:
    //   [0.00146, 0.0325, -0.0728, 0.00243, -0.00354] magnitude 0.049
    println!("\n--- Diffusers reference (from fixture) ---");
    let bisect_path = "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/helios/diffusers_umt5_bisect.safetensors";
    let bisect = flame_core::serialization::load_file(Path::new(bisect_path), &device)
        .map_err(|e| anyhow!("load bisect: {e}"))?;
    let diff_raw_embed = bisect.get("raw_embed").unwrap();
    let diff_layer0 = bisect.get("layer0_out").unwrap();
    let diff_raw_vec = diff_raw_embed.to_dtype(DType::F32)?.to_vec_f32()?;
    let diff_layer0_vec = diff_layer0.to_dtype(DType::F32)?.to_vec_f32()?;
    let _ = (diff_raw_vec, diff_layer0_vec);
    println!("  (see source for expected values)");

    println!("\n--- Encoding ---");
    let our_pe = umt5.encode(&token_ids_i32).map_err(|e| anyhow!("encode: {e}"))?;
    println!("  our_pe shape: {:?} dtype: {:?}", our_pe.shape().dims(), our_pe.dtype());

    // Compare to diffusers's PADDED prompt_embeds (same shape: (1, 512, 4096)).
    if our_pe.shape().dims() != pe_padded.shape().dims() {
        return Err(anyhow!(
            "shape mismatch: our {:?} vs diffusers {:?}",
            our_pe.shape().dims(),
            pe_padded.shape().dims()
        ));
    }
    let our_vec = our_pe.to_dtype(DType::F32)?.to_vec_f32()?;
    let diff_vec = pe_padded.to_dtype(DType::F32)?.to_vec_f32()?;

    let dims = our_pe.shape().dims().to_vec();
    let (b, s, d) = (dims[0], dims[1], dims[2]);

    // Per-position max/mean diff.
    println!("\n--- Per-position diff (real tokens only) ---");
    for pos in 0..real_seq_len {
        let off = pos * d;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for i in 0..d {
            let dv = (our_vec[off + i] - diff_vec[off + i]).abs();
            if dv > max_abs {
                max_abs = dv;
            }
            sum_abs += dv as f64;
        }
        let mean_abs = (sum_abs / d as f64) as f32;
        let our_mag = (0..d).map(|i| our_vec[off + i].abs()).sum::<f32>() / d as f32;
        let diff_mag = (0..d).map(|i| diff_vec[off + i].abs()).sum::<f32>() / d as f32;
        println!(
            "  pos {pos:>3}: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}  |our|={our_mag:.4e} |diff|={diff_mag:.4e}"
        );
    }

    // First 5 elements of pos 0
    println!("\n--- pos 0 first 5 elements ---");
    println!("  our:      {:?}", &our_vec[..5]);
    println!("  diff:     {:?}", &diff_vec[..5]);

    // Cosine similarity per position
    println!("\n--- Per-position cosine similarity (real tokens) ---");
    for pos in 0..real_seq_len {
        let off = pos * d;
        let mut dot = 0.0f64;
        let mut nn1 = 0.0f64;
        let mut nn2 = 0.0f64;
        for i in 0..d {
            let a = our_vec[off + i] as f64;
            let b = diff_vec[off + i] as f64;
            dot += a * b;
            nn1 += a * a;
            nn2 += b * b;
        }
        let cos = dot / (nn1.sqrt() * nn2.sqrt());
        println!("  pos {pos:>3}: cos={cos:.6}");
    }

    let _ = (b, s);
    let _ = pe_raw;
    Ok(())
}
