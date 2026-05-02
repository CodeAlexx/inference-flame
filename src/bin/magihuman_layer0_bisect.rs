//! Per-step parity for layer 0 of MagiHuman DiT on the 40-token chain.
//! Find the first sub-step where the video rows diverge from Python.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use flame_core::{global_cuda_device, DType, Tensor};
use inference_flame::models::magihuman_dit::{MMTransformerLayer, MlpAct, GELU7_LAYERS};

const ADAPTER_FIXTURE: &str =
    "/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/h_after_adapter_40tok.safetensors";
const WEIGHTS: &str =
    "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// Compare per modality on the leading axis.
/// `dim_layout` describes how rows split into V/A/T groups for this tensor.
fn compare(name: &str, ours: &Tensor, refr: &Tensor, group_dim_size: usize) {
    let our_v = ours.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap();
    let ref_v = refr.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap();
    if our_v.len() != ref_v.len() {
        println!("{name:25}  SHAPE MISMATCH: ours={} ref={}", our_v.len(), ref_v.len());
        return;
    }
    let total_rows = ours.shape().dims()[0];
    let stride = our_v.len() / total_rows;
    let v_end = group_dim_size; // 24
    let a_end = group_dim_size + 8; // 32
    let t_end = group_dim_size + 16; // 40
    let v_o = &our_v[0..v_end * stride];
    let v_r = &ref_v[0..v_end * stride];
    let a_o = &our_v[v_end * stride..a_end * stride];
    let a_r = &ref_v[v_end * stride..a_end * stride];
    let t_o = &our_v[a_end * stride..t_end * stride];
    let t_r = &ref_v[a_end * stride..t_end * stride];
    println!(
        "{name:25}  cos all={:.6} V={:.6} A={:.6} T={:.6}  max_abs ours={:.3} ref={:.3}",
        cosine(&our_v, &ref_v),
        cosine(v_o, v_r),
        cosine(a_o, a_r),
        cosine(t_o, t_r),
        max_abs(&our_v),
        max_abs(&ref_v),
    );
}

struct Facilitator;
impl flame_diffusion::block_offload::BlockFacilitator for Facilitator {
    fn block_count(&self) -> usize {
        40
    }
    fn classify_key(&self, name: &str) -> Option<usize> {
        name.strip_prefix("block.layers.")?.split('.').next()?.parse().ok()
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    // CLI: [layer_idx] [input_fixture] [input_key] [refs_fixture]
    // Defaults reproduce the original layer-0 case.
    let args: Vec<String> = std::env::args().collect();
    let layer_idx: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(0);
    let input_path = args.get(2).cloned().unwrap_or_else(|| ADAPTER_FIXTURE.to_string());
    let input_key = args.get(3).cloned().unwrap_or_else(|| "input".to_string());
    let refs_path = args.get(4).cloned().unwrap_or_else(|| {
        format!("/home/alex/EriDiffusion/inference-flame/tests/pytorch_fixtures/magihuman/layer{layer_idx}_intermediates_40tok.safetensors")
    });
    println!("layer={layer_idx} input={input_path}#{input_key} refs={refs_path}");

    let input_fix = flame_core::serialization::load_file(Path::new(&input_path), &device)?;
    let h = input_fix.get(&input_key).ok_or_else(|| anyhow!("missing key {input_key} in input"))?
        .to_dtype(DType::BF16)?;
    let rope_fix = flame_core::serialization::load_file(Path::new(ADAPTER_FIXTURE), &device)?;
    let rope = rope_fix.get("rope").unwrap().to_dtype(DType::F32)?;
    let group_sizes = vec![24usize, 8, 8];

    let refs = flame_core::serialization::load_file(Path::new(&refs_path), &device)?;

    let mut offloader =
        flame_diffusion::BlockOffloader::load(&[WEIGHTS], &Facilitator, device.clone())
            .map_err(|e| anyhow!("offloader: {e}"))?;
    for i in 0..=layer_idx {
        offloader.prefetch_block(i).map_err(|e| anyhow!("prefetch {i}: {e}"))?;
    }
    let raw = offloader.await_block(layer_idx).map_err(|e| anyhow!("await: {e}"))?;
    let weights: HashMap<String, _> = raw.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let act = if GELU7_LAYERS.contains(&layer_idx) { MlpAct::GELU7 } else { MlpAct::SwiGLU7 };
    let layer = MMTransformerLayer::load_with_layout(&weights, &format!("block.layers.{layer_idx}."), act, true)?;

    let mut intermediates: HashMap<String, Tensor> = HashMap::new();
    let _final = layer.forward_with_intermediates(&h, &rope, &group_sizes, &mut intermediates)?;

    // Dump Rust intermediates so they can be fed back into Python for cross-runtime checks.
    if let Ok(dump_to) = std::env::var("DUMP_RUST_INTERMEDIATES_TO") {
        let mut bytes_map = std::collections::BTreeMap::new();
        for (k, t) in &intermediates {
            bytes_map.insert(k.clone(), t.to_dtype(DType::F32)?);
        }
        // Use flame_core serialization
        flame_core::serialization::save_file(&bytes_map.into_iter().collect::<HashMap<_,_>>(), Path::new(&dump_to))?;
        println!("dumped Rust intermediates to {dump_to}");
    }

    // Print in reference order.
    let order = [
        "h_in",
        "after_pre_norm",
        "after_qkv",
        "after_q_norm",
        "after_k_norm",
        "after_rope_q",
        "after_rope_k",
        "after_sdpa",
        "after_gate",
        "after_attn_proj",
        "after_attn_residual",
        "after_mlp_pre_norm",
        "after_mlp_up",
        "after_mlp_act",
        "after_mlp_down",
        "after_layer0", // key in dump is hard-coded; rename below if needed
    ];

    for k in &order {
        let Some(ours) = intermediates.get(*k) else {
            println!("{k:25}  MISSING in Rust");
            continue;
        };
        let Some(refr) = refs.get(*k) else {
            println!("{k:25}  MISSING in refs");
            continue;
        };
        compare(k, ours, refr, 24);
    }

    Ok(())
}
