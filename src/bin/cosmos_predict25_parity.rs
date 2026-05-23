//! `cosmos_predict25_parity` — Rust-side per-layer capture for
//! Cosmos-Predict2.5-2B parity bisect.
//!
//! Mirrors the layer probe set in
//! `inference-flame/ports/cosmos-predict25-2b/parity/cosmos_predict25_per_layer_capture.py`.
//! Reads the same noise + text embedding the Python ref used (loaded from
//! `captures/input_noise.safetensors` + the file path in
//! `COSMOS_TEXT_EMB_BF16`), walks the DiT forward via the public methods on
//! `CosmosPredict25Dit`, and dumps each intermediate to
//! `parity/rust_captures/<name>.safetensors` as an F32 CPU tensor.
//!
//! The comparison is done by `cosmos_predict25_compare_captures.py`.
//!
//! USAGE:
//! ```text
//! COSMOS_DIT_PATH=/path/to/cosmos_predict25_2b_dit.safetensors \
//! COSMOS_TEXT_EMB_BF16=/path/to/captures/text_emb.safetensors \
//! cargo run --release --bin cosmos_predict25_parity -- \
//!     [--num-frames 81] [--res-w 832] [--res-h 480] [--fps 16.0]
//! ```
//!
//! Output directory: hard-coded to
//! `inference-flame/ports/cosmos-predict25-2b/parity/rust_captures/`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context};
use cudarc::driver::CudaDevice;

use flame_core::serialization::{load_file, save_tensors, SerializationFormat};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::cosmos_predict25_dit::{
    build_cosmos_rope_freqs, CosmosPredict25Config, CosmosPredict25Dit,
};

const PROMPT: &str = "a cat in a sunlit garden, photorealistic";
const SEED: u64 = 42;
const SHIFT: f32 = 5.0;

fn capture_dir() -> PathBuf {
    PathBuf::from(
        "/home/alex/EriDiffusion/inference-flame/ports/cosmos-predict25-2b/parity/rust_captures",
    )
}

fn input_capture_dir() -> PathBuf {
    PathBuf::from(
        "/home/alex/EriDiffusion/inference-flame/ports/cosmos-predict25-2b/parity/captures",
    )
}

/// Save a single tensor as F32 CPU under `parity/rust_captures/<name>.safetensors`.
fn stash(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let f32 = t.to_dtype(DType::F32).context("cast to F32")?;
    let mut tensors = HashMap::new();
    tensors.insert(name.to_string(), f32);
    let path = capture_dir().join(format!("{name}.safetensors"));
    save_tensors(&tensors, &path, SerializationFormat::SafeTensors)
        .map_err(|e| anyhow!("save {name}: {e}"))?;
    let dims = t.shape().dims();
    println!("  [capture] {name}: shape={:?} dtype={:?}", dims, t.dtype());
    Ok(())
}

fn parse_args() -> (usize, usize, usize, f32) {
    // (num_frames, res_w, res_h, fps)
    let mut num_frames = 81usize;
    let mut res_w = 832usize;
    let mut res_h = 480usize;
    let mut fps = 16.0f32;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match a {
            "--num-frames" => {
                num_frames = v.parse().expect("usize");
                i += 2;
            }
            "--res-w" => {
                res_w = v.parse().expect("usize");
                i += 2;
            }
            "--res-h" => {
                res_h = v.parse().expect("usize");
                i += 2;
            }
            "--fps" => {
                fps = v.parse().expect("f32");
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    (num_frames, res_w, res_h, fps)
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::try_init();
    let (num_frames, res_w, res_h, fps) = parse_args();

    let cap_dir = capture_dir();
    std::fs::create_dir_all(&cap_dir).context("mkdir rust_captures")?;

    let device: Arc<CudaDevice> = global_cuda_device();

    let dit_path = std::env::var("COSMOS_DIT_PATH")
        .context("COSMOS_DIT_PATH not set (path to cosmos_predict25 DiT .safetensors)")?;
    let txt_emb_path = std::env::var("COSMOS_TEXT_EMB_BF16")
        .context("COSMOS_TEXT_EMB_BF16 not set (matching text embedding file)")?;

    let in_noise_path = input_capture_dir().join("input_noise.safetensors");
    if !in_noise_path.exists() {
        return Err(anyhow!(
            "{} not found. Run cosmos_predict25_per_layer_capture.py first.",
            in_noise_path.display()
        ));
    }

    println!("=== Cosmos-Predict2.5-2B Rust per-layer parity capture ===");
    println!("  prompt:    {PROMPT:?}");
    println!("  seed:      {SEED}");
    println!("  res:       {res_w}x{res_h}");
    println!("  num_frames:{num_frames}");
    println!("  fps:       {fps}");
    println!("  dit_path:  {dit_path}");
    println!("  txt_emb:   {txt_emb_path}");
    println!("  noise in:  {}", in_noise_path.display());
    println!("  cap dir:   {}", cap_dir.display());

    // Geometry.
    let h_lat = res_h / 8;
    let w_lat = res_w / 8;
    let t_lat = (num_frames - 1) / 4 + 1;
    let in_c = 16usize;
    println!("  lat geom:  [1, {in_c}, {t_lat}, {h_lat}, {w_lat}]");

    // Load model.
    println!("Loading DiT (production preset)...");
    let cfg = CosmosPredict25Config::cosmos_v2_2b_production();
    let dit = CosmosPredict25Dit::from_safetensors(&PathBuf::from(&dit_path), cfg, device.clone())
        .map_err(|e| anyhow!("DiT load: {e}"))?;
    println!("  loaded.");

    // Load noise (Python-generated) — keep BF16.
    let noise_map = load_file(&in_noise_path, &device)
        .map_err(|e| anyhow!("load noise: {e}"))?;
    let noise = noise_map
        .get("noise_bf16")
        .ok_or_else(|| anyhow!("input_noise.safetensors missing key 'noise_bf16'"))?
        .clone();
    let nd = noise.shape().dims();
    if nd != [1, in_c, t_lat, h_lat, w_lat] {
        return Err(anyhow!(
            "noise shape {:?} does not match expected [1, {in_c}, {t_lat}, {h_lat}, {w_lat}]",
            nd
        ));
    }

    // Load text emb.
    let txt_map = load_file(&PathBuf::from(&txt_emb_path), &device)
        .map_err(|e| anyhow!("load text emb: {e}"))?;
    // Accept several common key names.
    let crossattn_emb = ["prompt_emb_bf16", "prompt_emb", "text_emb", "crossattn_emb"]
        .iter()
        .find_map(|k| txt_map.get(*k).cloned())
        .ok_or_else(|| {
            anyhow!(
                "text emb file {txt_emb_path} missing key (looked for prompt_emb_bf16/prompt_emb/text_emb/crossattn_emb)"
            )
        })?;
    let crossattn_emb = if crossattn_emb.dtype() == DType::BF16 {
        crossattn_emb
    } else {
        crossattn_emb
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("text emb cast to BF16: {e}"))?
    };
    let txt_dims = crossattn_emb.shape().dims();
    println!("  text emb shape: {:?}", txt_dims);

    // -------------------------------------------------------------------------
    // Build inputs that the production DiT expects.
    // -------------------------------------------------------------------------
    // condition_video_input_mask = zeros [1, 1, T_lat, H_lat, W_lat] (image mode)
    let cond_vid_mask = Tensor::zeros_dtype(
        Shape::from_dims(&[1, 1, t_lat, h_lat, w_lat]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| anyhow!("cond vid mask: {e}"))?;

    // padding_mask = zeros [1, 1, H_lat, W_lat]
    let padding_mask = Tensor::zeros_dtype(
        Shape::from_dims(&[1, 1, h_lat, w_lat]),
        DType::BF16,
        device.clone(),
    )
    .map_err(|e| anyhow!("padding mask: {e}"))?;

    // Timesteps: sigma_0 * 1000 where sigma_0 = shift*0.999 / (1 + (shift-1)*0.999)
    let sigma_0 = (SHIFT * 0.999) / (1.0 + (SHIFT - 1.0) * 0.999);
    let t_value = sigma_0 * 1000.0;
    println!("  sigma_0={:.6} t_value={:.3}", sigma_0, t_value);
    let ts_vec = vec![t_value; t_lat];
    let timesteps_b_t = Tensor::from_vec(ts_vec, Shape::from_dims(&[1, t_lat]), device.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
        .map_err(|e| anyhow!("timesteps: {e}"))?;

    // -------------------------------------------------------------------------
    // Manual forward walk — capturing at each layer.
    //
    // Order mirrors the Python script's hook set and the
    // `cosmos_predict25_compare_captures.py` ORDERED_NAMES.
    // -------------------------------------------------------------------------

    // 1. LVG cat + padding-mask cat (production: lvg_wrapper=true → 16+1+1=18 ch)
    let x_after_lvg = Tensor::cat(&[&noise, &cond_vid_mask], 1)
        .and_then(|x| x.contiguous())
        .map_err(|e| anyhow!("LVG cat: {e}"))?;
    let mask_5d = padding_mask
        .reshape(&[1, 1, 1, h_lat, w_lat])
        .and_then(|x| x.broadcast_to(&Shape::from_dims(&[1, 1, t_lat, h_lat, w_lat])))
        .and_then(|x| x.contiguous())
        .map_err(|e| anyhow!("padding mask broadcast: {e}"))?;
    let x_with_mask = Tensor::cat(&[&x_after_lvg, &mask_5d], 1)
        .and_then(|x| x.contiguous())
        .map_err(|e| anyhow!("padding mask cat: {e}"))?;
    let dims = x_with_mask.shape().dims();
    println!("  post-cat x shape: {:?}", dims);

    // 2. Crossattn projection (100352 → 1024 with bias + GELU exact).
    let crossattn_proj_out = dit
        .apply_crossattn_proj(&crossattn_emb)
        .map_err(|e| anyhow!("crossattn_proj: {e}"))?;
    stash("crossattn_post_proj", &crossattn_proj_out)?;

    // 3. Patchify (no learnable params).
    let x_patches = dit
        .patchify(&x_with_mask)
        .map_err(|e| anyhow!("patchify: {e}"))?;
    let pd = x_patches.shape().dims();
    let (t_p, h_p, w_p) = (pd[1], pd[2], pd[3]);

    // 4. x_embedder Linear.
    let x_emb = dit
        .x_embedder(&x_patches)
        .map_err(|e| anyhow!("x_embedder: {e}"))?;
    stash("post_x_embedder", &x_emb)?;

    // 5. Timesteps + TimestepEmbedding (capture pre-norm via sinusoidal+lin path).
    let sinus = dit
        .sinusoidal_timesteps(&timesteps_b_t)
        .and_then(|t| t.to_dtype(DType::BF16))
        .map_err(|e| anyhow!("sinusoidal: {e}"))?;
    let (t_emb_pre, _adaln_pre) = dit
        .timestep_embedding(&sinus)
        .map_err(|e| anyhow!("timestep_embedding: {e}"))?;
    stash("t_emb_pre_norm", &t_emb_pre)?;

    // 6. prepare_timestep applies t_embedding_norm internally.
    let (t_emb_post, adaln_lora_opt) = dit
        .prepare_timestep(&timesteps_b_t)
        .map_err(|e| anyhow!("prepare_timestep: {e}"))?;
    stash("t_emb_post_norm", &t_emb_post)?;
    let adaln_lora = adaln_lora_opt
        .ok_or_else(|| anyhow!("V2_2B requires adaln_lora but got None"))?;

    // 7. extra_pos_emb (LearnablePosEmbAxis) — only if production preset
    //    has it enabled. The shipped checkpoint has NO `extra_pos_embedder.*`
    //    keys, so `cosmos_v2_2b_production()` sets the flag false; skip.
    let extra_pos_emb_opt = if dit.config.extra_per_block_abs_pos_emb {
        let pe = dit
            .learnable_pos_emb(1, t_p, h_p, w_p)
            .map_err(|e| anyhow!("learnable_pos_emb: {e}"))?;
        stash("extra_pos_emb", &pe)?;
        Some(pe)
    } else {
        None
    };

    // 8. 3D RoPE cos/sin — concat them as the comparison expects a single
    //    `rope_freqs` tensor (Python's pos_embedder.forward output).
    //    Production preset: rope_h/w_extrapolation_ratio=3.0, t=1.0,
    //    enable_fps_modulation=false.
    let head_dim = dit.config.head_dim();
    let (rope_cos, rope_sin) = build_cosmos_rope_freqs(
        head_dim,
        t_p,
        h_p,
        w_p,
        Some(fps),
        16.0, // base_fps
        dit.config.rope_h_extrapolation_ratio,
        dit.config.rope_w_extrapolation_ratio,
        dit.config.rope_t_extrapolation_ratio,
        dit.config.rope_enable_fps_modulation,
        &device,
    )
    .map_err(|e| anyhow!("rope_freqs: {e}"))?;
    // Reconstruct Python layout `[L, 1, 1, D]` as `cat([cos2, sin2], -1)`?
    // No — Python returns `cat([t,h,w]*2, -1)`. We have cos/sin halves. To
    // approximate the Python output, we save cos and sin separately and let
    // the comparator handle both.
    stash("rope_cos", &rope_cos)?;
    stash("rope_sin", &rope_sin)?;
    // Also save a "rope_freqs"-equivalent tensor: per token, the angle vector
    // is just acos(cos) (with appropriate sign from sin). Skipped — cos+sin
    // is sufficient.

    // ------------------------------------------------------------------------
    // Block-by-block forward. We capture at block 0 in detail, and at
    // block 13 + block 27 just their outputs. The block walk mirrors
    // `transformer_block` but with F32 residual stream.
    // ------------------------------------------------------------------------
    // Each block adds extra_per_block_pos_emb at the top, then sub-blocks.
    let mut x_residual = x_emb.clone();

    let num_blocks = dit.config.num_blocks;
    for block_idx in 0..num_blocks {
        // Match transformer_block's F32 residual pattern; add the per-block
        // pos emb ONLY if the production preset enables it (off for V2_2B).
        let x_f32_pre = if let Some(pe) = extra_pos_emb_opt.as_ref() {
            x_residual
                .add(pe)
                .and_then(|t| t.to_dtype(DType::F32))
                .map_err(|e| anyhow!("block {block_idx} pe-add: {e}"))?
        } else {
            x_residual
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("block {block_idx} cast: {e}"))?
        };

        // ----- Self-attention path -----
        let (sh_sa, sc_sa, ga_sa) = dit
            .adaln_modulation_chunk(&t_emb_post, &adaln_lora, block_idx, "self_attn")
            .map_err(|e| anyhow!("block {block_idx} adaln self: {e}"))?;
        let x_bf16 = x_f32_pre
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("block {block_idx} bf16-1: {e}"))?;
        let x_mod_sa = dit
            .apply_layer_norm_modulate(&x_bf16, &sh_sa, &sc_sa)
            .map_err(|e| anyhow!("block {block_idx} mod self: {e}"))?;
        if block_idx == 0 {
            stash("block_0_post_modulate_pre_self_attn", &x_mod_sa)?;
        }
        let sa_out = dit
            .self_attention(&x_mod_sa, &rope_cos, &rope_sin, block_idx)
            .map_err(|e| anyhow!("block {block_idx} self_attn: {e}"))?;
        if block_idx == 0 {
            stash("block_0_post_self_attn", &sa_out)?;
        }
        let sa_gated = dit
            .apply_gate(&sa_out, &ga_sa)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| anyhow!("block {block_idx} gate self: {e}"))?;
        let x_f32_after_sa = x_f32_pre
            .add(&sa_gated)
            .map_err(|e| anyhow!("block {block_idx} add self: {e}"))?;

        // ----- Cross-attention path -----
        let (sh_ca, sc_ca, ga_ca) = dit
            .adaln_modulation_chunk(&t_emb_post, &adaln_lora, block_idx, "cross_attn")
            .map_err(|e| anyhow!("block {block_idx} adaln cross: {e}"))?;
        let x_bf16 = x_f32_after_sa
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("block {block_idx} bf16-2: {e}"))?;
        let x_mod_ca = dit
            .apply_layer_norm_modulate(&x_bf16, &sh_ca, &sc_ca)
            .map_err(|e| anyhow!("block {block_idx} mod cross: {e}"))?;
        if block_idx == 0 {
            stash("block_0_post_modulate_pre_cross_attn", &x_mod_ca)?;
        }
        let ca_out = dit
            .cross_attention(&x_mod_ca, &crossattn_proj_out, None, block_idx)
            .map_err(|e| anyhow!("block {block_idx} cross_attn: {e}"))?;
        if block_idx == 0 {
            stash("block_0_post_cross_attn", &ca_out)?;
        }
        let ca_gated = dit
            .apply_gate(&ca_out, &ga_ca)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| anyhow!("block {block_idx} gate cross: {e}"))?;
        let x_f32_after_ca = x_f32_after_sa
            .add(&ca_gated)
            .map_err(|e| anyhow!("block {block_idx} add cross: {e}"))?;

        // ----- MLP path -----
        let (sh_m, sc_m, ga_m) = dit
            .adaln_modulation_chunk(&t_emb_post, &adaln_lora, block_idx, "mlp")
            .map_err(|e| anyhow!("block {block_idx} adaln mlp: {e}"))?;
        let x_bf16 = x_f32_after_ca
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("block {block_idx} bf16-3: {e}"))?;
        let x_mod_m = dit
            .apply_layer_norm_modulate(&x_bf16, &sh_m, &sc_m)
            .map_err(|e| anyhow!("block {block_idx} mod mlp: {e}"))?;
        if block_idx == 0 {
            stash("block_0_post_modulate_pre_mlp", &x_mod_m)?;
        }
        let mlp_out = dit
            .mlp(&x_mod_m, block_idx)
            .map_err(|e| anyhow!("block {block_idx} mlp: {e}"))?;
        if block_idx == 0 {
            stash("block_0_post_ffn", &mlp_out)?;
        }
        let mlp_gated = dit
            .apply_gate(&mlp_out, &ga_m)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| anyhow!("block {block_idx} gate mlp: {e}"))?;
        let x_f32_block = x_f32_after_ca
            .add(&mlp_gated)
            .map_err(|e| anyhow!("block {block_idx} add mlp: {e}"))?;

        let x_block = x_f32_block
            .to_dtype(DType::BF16)
            .map_err(|e| anyhow!("block {block_idx} bf16-final: {e}"))?;
        if block_idx == 0 {
            stash("block_0_output", &x_block)?;
        }
        if block_idx == 13 {
            stash("block_13_output", &x_block)?;
        }
        if block_idx == num_blocks - 1 {
            stash("block_27_output", &x_block)?;
        }
        x_residual = x_block;
    }

    // 9. FinalLayer
    stash("pre_final_layer", &x_residual)?;
    let x_final = dit
        .final_layer(&x_residual, &t_emb_post, &adaln_lora)
        .map_err(|e| anyhow!("final_layer: {e}"))?;
    stash("post_final_layer", &x_final)?;

    // 10. Unpatchify (no params) — matches Python `post_unpatchify`.
    let velocity = dit
        .unpatchify(&x_final)
        .map_err(|e| anyhow!("unpatchify: {e}"))?;
    stash("post_unpatchify", &velocity)?;

    println!("=== DONE ===");
    Ok(())
}
