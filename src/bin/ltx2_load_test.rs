//! LTX-2 load test — verify key remapping against the actual 43GB checkpoint.
//!
//! Loads global params via LTX2StreamingModel::load_globals, then loads block 0.
//! Reports all key counts, shapes, and any errors.

use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use std::time::Instant;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    println!("============================================================");
    println!("LTX-2.3 Load Test — Key Remapping Verification");
    println!("============================================================");
    println!("Checkpoint: {}", MODEL_PATH);

    // Verify checkpoint exists
    let meta = std::fs::metadata(MODEL_PATH)
        .or_else(|_| {
            // Follow symlink
            let target = std::fs::read_link(MODEL_PATH)?;
            std::fs::metadata(&target)
        })?;
    println!("File size: {:.2} GB", meta.len() as f64 / 1e9);

    let config = LTX2Config::default();
    println!("\nConfig:");
    println!("  inner_dim: {} ({}h x {}d)", config.inner_dim(),
             config.num_attention_heads, config.attention_head_dim);
    println!("  num_layers: {}", config.num_layers);
    println!("  caption_channels: {}", config.caption_channels);

    // ---- Stage 1: Load globals ----
    println!("\n--- Stage 1: Load globals ---");
    let t0 = Instant::now();
    let model = match LTX2StreamingModel::load_globals(MODEL_PATH, &config) {
        Ok(m) => {
            println!("  SUCCESS in {:.1}s", t0.elapsed().as_secs_f32());
            m
        }
        Err(e) => {
            println!("  FAILED: {}", e);
            println!("\nDiagnostic: dumping first 20 global-level keys from checkpoint...");
            dump_global_keys(MODEL_PATH, 20);
            return Err(e.into());
        }
    };

    // Print global param shapes
    println!("\n  Global params loaded:");
    println!("    proj_in_weight: {:?}", model.proj_in_weight.shape().dims());
    println!("    proj_in_bias: {:?}", model.proj_in_bias.shape().dims());
    println!("    scale_shift_table: {:?}", model.scale_shift_table.shape().dims());
    println!("    proj_out_weight: {:?}", model.proj_out_weight.shape().dims());
    println!("    proj_out_bias: {:?}", model.proj_out_bias.shape().dims());
    println!("    connector: {}", if model.connector.is_some() { "YES" } else { "no" });
    println!("    caption_projection: {}", if model.caption_projection.is_some() { "YES" } else { "no" });
    println!("    prompt_adaln_single: {}", if model.prompt_adaln_single.is_some() { "YES" } else { "no" });
    println!("    time_embed.num_mod_params: {}", model.time_embed.num_mod_params);

    // ---- Stage 2: Load block 0 ----
    println!("\n--- Stage 2: Load block 0 ---");
    let t0 = Instant::now();
    match model.load_block(0) {
        Ok(block) => {
            let dt = t0.elapsed().as_secs_f32();
            println!("  SUCCESS in {:.2}s", dt);
            println!("\n  Block 0 params:");
            println!("    scale_shift_table: {:?}", block.scale_shift_table.shape().dims());
            println!("    attn1.to_q_weight: {:?}", block.attn1.to_q_weight.shape().dims());
            println!("    attn1.to_k_weight: {:?}", block.attn1.to_k_weight.shape().dims());
            println!("    attn1.to_v_weight: {:?}", block.attn1.to_v_weight.shape().dims());
            println!("    attn1.to_out_weight: {:?}", block.attn1.to_out_weight.shape().dims());
            println!("    attn1.norm_q_weight: {:?}", block.attn1.norm_q_weight.shape().dims());
            println!("    attn1.norm_k_weight: {:?}", block.attn1.norm_k_weight.shape().dims());
            println!("    attn1.to_gate_logits: {}",
                     if block.attn1.to_gate_logits_weight.is_some() { "YES" } else { "no" });
            if let Some(ref gl) = block.attn1.to_gate_logits_weight {
                println!("      weight: {:?}", gl.shape().dims());
            }
            println!("    attn2.to_q_weight: {:?}", block.attn2.to_q_weight.shape().dims());
            println!("    attn2.to_gate_logits: {}",
                     if block.attn2.to_gate_logits_weight.is_some() { "YES" } else { "no" });
            println!("    ff.gelu_proj_weight: {:?}", block.ff.gelu_proj_weight.shape().dims());
            println!("    ff.out_weight: {:?}", block.ff.out_weight.shape().dims());
            println!("    norm1_weight: {}", if block.norm1_weight.is_some() { "YES" } else { "None (expected)" });
            println!("    prompt_scale_shift_table: {}",
                     if block.prompt_scale_shift_table.is_some() { "YES" } else { "no" });
            if let Some(ref psst) = block.prompt_scale_shift_table {
                println!("      shape: {:?}", psst.shape().dims());
            }

            // Estimate block VRAM
            let estimate_mb = (
                // attn1: Q,K,V,Out weights + biases + norms + gate
                4096 * 4096 * 4 * 2  // 4 weight matrices, bf16
                + 4096 * 4 * 2       // 4 biases
                + 4096 * 2 * 2       // 2 norms
                + 32 * 4096 * 2      // gate weight
                // attn2: same
                + 4096 * 4096 * 4 * 2
                + 4096 * 4 * 2
                + 4096 * 2 * 2
                + 32 * 4096 * 2
                // ff
                + 16384 * 4096 * 2   // gelu proj
                + 4096 * 16384 * 2   // out proj
                + (16384 + 4096) * 2 // biases
                // tables
                + 9 * 4096 * 4       // scale_shift f32
                + 2 * 4096 * 4       // prompt_scale_shift f32
                + 5 * 4096 * 4       // a2v_ca f32
            ) as f64 / 1e6;
            println!("    ~estimated block VRAM: {:.0} MB", estimate_mb);

            drop(block);
            println!("  Block 0 dropped (GPU freed)");
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    println!("\n============================================================");
    println!("Load test complete.");
    println!("============================================================");

    Ok(())
}

fn dump_global_keys(path: &str, limit: usize) {
    use std::io::Read;
    let Ok(file) = std::fs::File::open(path) else { return };
    let Ok(mmap) = (unsafe { memmap2::Mmap::map(&file) }) else { return };
    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let Ok(metadata): std::result::Result<serde_json::Value, _> =
        serde_json::from_slice(&mmap[8..8 + header_size]) else { return };

    if let Some(obj) = metadata.as_object() {
        let mut global_keys: Vec<_> = obj.keys()
            .filter(|k| *k != "__metadata__"
                && !k.contains("transformer_blocks")
                && !k.contains("audio"))
            .collect();
        global_keys.sort();
        for (i, k) in global_keys.iter().take(limit).enumerate() {
            let shape = obj[*k].get("shape")
                .and_then(|v| v.as_array())
                .map(|a| format!("{:?}", a))
                .unwrap_or_default();
            println!("    [{}] {} {}", i, k, shape);
        }
        println!("    ... ({} total global non-audio keys)", global_keys.len());
    }
}
