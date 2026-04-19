//! Smoke-test that `LTX2StreamingModel::add_lora` actually changes loaded
//! block weights.
//!
//! Procedure:
//! 1. Load globals.
//! 2. Load block 0 — capture `attn1.to_q_weight` as the baseline.
//! 3. Attach the distilled LoRA.
//! 4. Reload block 0 — capture the fused `attn1.to_q_weight`.
//! 5. Compute cos_sim between baseline and fused in F32. Must be < 1.0.
//!    Also assert they differ by a meaningful amount.
//!
//! This does NOT validate the numerical fusion formula — `lora_fusion_parity`
//! already does that. This just proves the LoRA actually reaches the block
//! weights through the code path.

use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::lora_loader::LoraWeights;
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors";
const LORA_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors";

fn cos_sim_and_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<(f64, f64)> {
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut abs_diff = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        abs_diff += (x - y).abs();
    }
    Ok((dot / (na.sqrt() * nb.sqrt()).max(1e-12), abs_diff / av.len() as f64))
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("=== LTX-2 LoRA wiring check ===");
    let device = global_cuda_device();
    let config = LTX2Config::default();

    println!("Loading globals (no LoRA)...");
    let mut model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;

    println!("Loading block 0 (no LoRA)...");
    let baseline_block = model.load_block(0)?;
    let baseline_q = baseline_block.attn1.to_q_weight.clone();
    let baseline_k = baseline_block.attn1.to_k_weight.clone();
    let baseline_ff = baseline_block.ff.gelu_proj_weight.clone();
    drop(baseline_block);

    println!("Loading and attaching LoRA: {}", LORA_PATH);
    let lora = LoraWeights::load(LORA_PATH, 1.0, &device)?;
    println!("  LoRA keys: {}, rank: {:?}", lora.len(), lora.rank());
    model.add_lora(lora);

    println!("Reloading block 0 (with LoRA)...");
    let fused_block = model.load_block(0)?;
    let fused_q = fused_block.attn1.to_q_weight.clone();
    let fused_k = fused_block.attn1.to_k_weight.clone();
    let fused_ff = fused_block.ff.gelu_proj_weight.clone();
    drop(fused_block);

    let (cos_q, mad_q) = cos_sim_and_diff(&baseline_q, &fused_q)?;
    let (cos_k, mad_k) = cos_sim_and_diff(&baseline_k, &fused_k)?;
    let (cos_ff, mad_ff) = cos_sim_and_diff(&baseline_ff, &fused_ff)?;
    println!();
    println!("attn1.to_q.weight:      cos_sim={:.6} mean|diff|={:.6e}", cos_q, mad_q);
    println!("attn1.to_k.weight:      cos_sim={:.6} mean|diff|={:.6e}", cos_k, mad_k);
    println!("ff.net.0.proj.weight:   cos_sim={:.6} mean|diff|={:.6e}", cos_ff, mad_ff);

    // Also check a global tensor (proj_in / patchify_proj)
    println!();
    println!("Global-fusion sanity: time_embed.linear_weight (post add_lora vs original)...");
    // We don't have the baseline for globals anymore since we mutated the
    // struct in place. Still useful: assert the post-fusion globals at least
    // contain finite values with expected magnitude.
    let tev = model.time_embed.linear_weight.to_dtype(DType::F32)?.to_vec_f32()?;
    let min = tev.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = tev.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_abs: f32 = tev.iter().map(|x| x.abs()).sum::<f32>() / tev.len() as f32;
    println!("  time_embed.linear_weight: min={:.4} max={:.4} mean|.|={:.4e}", min, max, mean_abs);
    assert!(min.is_finite() && max.is_finite(), "globals NaN/inf after fusion");

    let all_ok = cos_q < 1.0 && cos_k < 1.0 && cos_ff < 1.0
        && mad_q > 1e-6 && mad_k > 1e-6 && mad_ff > 1e-6;
    if all_ok {
        println!("\nPASS: LoRA fusion changes block weights as expected.");
        Ok(())
    } else {
        println!("\nFAIL: some tensors unchanged (cos=1.0 or mean|diff|=~0). LoRA wiring broken.");
        std::process::exit(1)
    }
}
