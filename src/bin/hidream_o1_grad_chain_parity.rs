//! HiDream-O1 per-checkpoint gradient-chain parity vs PyTorch ai-toolkit.
//!
//! Mirrors `inference-flame/src/bin/l2p_grad_chain_parity.rs`. The L2P binary
//! settled the L2P body-cascade question; this binary settles the analogous
//! question for HiDream-O1 — specifically, whether the
//! `Op::RoPePrecomputed` shape-sniff fix actually landed and whether V LoRA-B
//! grad cos is ≈ 1.0 (the decisive test from
//! `project_hidream_o1_qkv_lora_grad_collapse_2026-05-20`).
//!
//! ## Strategy
//!
//! 1. Load shared inputs from a Python-produced reference dump
//!    (`/tmp/hidream_o1_train_step_ref.safetensors`) — `noisy`, `input_ids`,
//!    `position_ids`, `vinput_mask`, `token_types`, `timestep`,
//!    `target_velocity`.
//! 2. Build a LoRA stack across ALL decoder layers + 5 resident heads
//!    (`LORA_ADAPTERS = 257`, rank=32, alpha=32) — matches the Python ref
//!    setup in `tests/parity/hidream_o1_train_step_ref.py`.
//! 3. Load LoRA init from the Python ref's `init.<adapter>` keys via
//!    `parity_hidream_o1_train_step`-compatible loader.
//! 4. Arm BOTH probe stores (`trap::arm_probes()` for the older soul.md
//!    flow AND `trap::arm_for_env()` for the new L2P-style block trap driven
//!    by `HIDREAM_BLOCK_PROBE_LAYER`).
//! 5. Forward → MSE velocity loss → backward.
//! 6. Dump per-Parameter grads + per-checkpoint activation grads to
//!    safetensors with `act_grad.block.<name>` keys.
//!
//! ## Required env
//!
//!   LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib
//!   FLAME_ALLOC_POOL=0                        — mandatory (HiDream-O1
//!                                                 leaks pool memory)
//!   HIDREAM_BLOCK_PROBE_LAYER=35              — pick which layer to probe
//!                                                 (35 = last, cleanest signal)
//!
//! ## Companion files
//!
//!   inference-flame/src/models/hidream_o1/parity/grad_direction.py  — Python ref
//!   inference-flame/src/models/hidream_o1/parity/grad_diff.py       — cos diff
//!
//! ## Usage
//!
//!   LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib FLAME_ALLOC_POOL=0 \
//!     HIDREAM_BLOCK_PROBE_LAYER=35 \
//!     ./target/release/hidream_o1_grad_chain_parity \
//!       --ref /tmp/hidream_o1_train_step_ref.safetensors \
//!       --lora-ref /tmp/hidream_o1_lora_step_ref.safetensors \
//!       --model /home/alex/HiDream-O1-Image-Full-weights \
//!       --out /tmp/hidream_o1_grad_chain/grads_rust.safetensors

use flame_core::serialization::{load_file, save_tensors, SerializationFormat};
use flame_core::{
    autograd::AutogradContext, DType, Error, Result, Shape, Tensor, TensorId,
};
use inference_flame::models::hidream_o1::{
    default_target_suffixes, HiDreamO1Config, HiDreamO1WeightLoader, LoraRegistry, MRopePositions,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::ExitCode;

const LORA_RANK: usize = 32;
const LORA_ALPHA: f32 = 32.0;
/// 252 decoder linears (36 layers × 7) + 5 resident O1 heads = 257.
const LORA_ADAPTERS: usize = 257;

struct Args {
    /// HiDream-O1 weights dir (Full variant by default — matches
    /// parity_hidream_o1_train_step).
    model: PathBuf,
    /// Python training-step reference (`tests/parity/hidream_o1_train_step_ref.py
    /// --dump-layers --lora-step`).
    ref_path: PathBuf,
    /// Python LoRA reference dump (provides `init.<adapter>` tensors so
    /// Rust and PyTorch start from byte-identical LoRA params).
    lora_ref: PathBuf,
    /// Where to write the Rust grad dump.
    out: PathBuf,
    /// Disable training-time gradient checkpointing for bisect runs.
    no_checkpoint: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        model: PathBuf::from("/home/alex/HiDream-O1-Image-Full-weights"),
        ref_path: PathBuf::from("/tmp/hidream_o1_train_step_ref.safetensors"),
        lora_ref: PathBuf::from("/tmp/hidream_o1_lora_step_ref.safetensors"),
        out: PathBuf::from("/tmp/hidream_o1_grad_chain/grads_rust.safetensors"),
        no_checkpoint: false,
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                i += 1;
                a.model = PathBuf::from(&argv[i]);
            }
            "--ref" | "--ref-path" => {
                i += 1;
                a.ref_path = PathBuf::from(&argv[i]);
            }
            "--lora-ref" => {
                i += 1;
                a.lora_ref = PathBuf::from(&argv[i]);
            }
            "--out" => {
                i += 1;
                a.out = PathBuf::from(&argv[i]);
            }
            "--no-checkpoint" => {
                a.no_checkpoint = true;
            }
            "-h" | "--help" => {
                eprintln!(
                    "hidream_o1_grad_chain_parity — per-checkpoint grad parity\n\
                     \n\
                     Usage:\n  \
                     hidream_o1_grad_chain_parity \\\n    \
                       [--model <dir>] \\\n    \
                       [--ref <safetensors>] \\\n    \
                       [--lora-ref <safetensors>] \\\n    \
                       [--out <safetensors>] \\\n    \
                       [--no-checkpoint]\n\
                     \n\
                     Required env:\n  \
                       LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib\n  \
                       FLAME_ALLOC_POOL=0\n  \
                       HIDREAM_BLOCK_PROBE_LAYER=<N>     (0..36)\n"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    a
}

fn scalar1(t: &Tensor) -> Result<f32> {
    let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
    v.first()
        .copied()
        .ok_or_else(|| Error::InvalidInput("scalar tensor is empty".into()))
}

fn decode_position_ids(pos: &Tensor) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let dims = pos.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != 3 {
        return Err(Error::InvalidInput(format!(
            "position_ids: expected [3, S_total], got {dims:?}"
        )));
    }
    let s_total = dims[1];
    let flat = pos.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut t = Vec::with_capacity(s_total);
    let mut h = Vec::with_capacity(s_total);
    let mut w = Vec::with_capacity(s_total);
    for i in 0..s_total {
        t.push(flat[i] as u32);
        h.push(flat[s_total + i] as u32);
        w.push(flat[2 * s_total + i] as u32);
    }
    Ok((t, h, w))
}

fn gather_image_rows(x_pred: &Tensor, vinput_mask: &Tensor) -> Result<Tensor> {
    let xd = x_pred.shape().dims().to_vec();
    if xd.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "gather_image_rows: expected [B,S,C], got {xd:?}"
        )));
    }
    let (_b, s_total, _c) = (xd[0], xd[1], xd[2]);
    let host = vinput_mask.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut first: Option<usize> = None;
    let mut last: Option<usize> = None;
    for i in 0..s_total {
        if host[i] != 0.0 {
            first.get_or_insert(i);
            last = Some(i);
        }
    }
    let (first, last) = (
        first.ok_or_else(|| Error::InvalidInput("vinput_mask has no image slots".into()))?,
        last.unwrap(),
    );
    let len = last - first + 1;
    let count = host.iter().filter(|&&x| x != 0.0).count();
    if count != len {
        return Err(Error::InvalidInput(format!(
            "non-contiguous image slots not supported (count {count} != span {len})"
        )));
    }
    x_pred.narrow(1, first, len)
}

/// Load PyTorch ai-toolkit LoRA init values into the Rust registry, replacing
/// the random Kaiming init in-place. Same logic as
/// `parity_hidream_o1_train_step::load_lora_init`, but uses
/// `flame_core::serialization::load_file` so we don't depend on the
/// `safetensors` crate directly.
fn load_lora_init_from_map(
    lora: &LoraRegistry,
    lora_map: &HashMap<String, Tensor>,
) -> Result<()> {
    let named = lora.named_parameters();
    if named.len() != LORA_ADAPTERS * 2 {
        return Err(Error::InvalidInput(format!(
            "expected {} LoRA tensors, got {}",
            LORA_ADAPTERS * 2,
            named.len()
        )));
    }
    for (name, p) in named {
        let key = format!("init.{name}");
        let t = lora_map.get(&key).ok_or_else(|| {
            Error::InvalidInput(format!("lora ref missing key {key:?}"))
        })?;
        // load_file returns the tensor in its on-disk dtype (F32 per the
        // Python ref's `_clone_or_zero_grad` which casts to float). The
        // registry was built with F32 leaves.
        let t = if t.dtype() == DType::F32 {
            t.clone()
        } else {
            t.to_dtype(DType::F32)?
        };
        p.set_data(t.requires_grad_(true))?;
    }
    Ok(())
}

fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let args = parse_args();

    // Pre-flight: HiDream-O1 OOMs without pool=0.
    if std::env::var("FLAME_ALLOC_POOL").as_deref() != Ok("0") {
        eprintln!(
            "WARNING: FLAME_ALLOC_POOL is not set to 0. HiDream-O1 backward \
             may OOM without this (1 GB/sample leak)."
        );
    }
    if !args.ref_path.exists() {
        return Err(Error::InvalidInput(format!(
            "ref dump {} not found; run \
             EriDiffusion-v2/tests/parity/hidream_o1_train_step_ref.py \
             --dump-layers --lora-step first",
            args.ref_path.display()
        )));
    }
    if !args.lora_ref.exists() {
        return Err(Error::InvalidInput(format!(
            "lora ref dump {} not found; run \
             EriDiffusion-v2/tests/parity/hidream_o1_train_step_ref.py \
             --lora-step first",
            args.lora_ref.display()
        )));
    }
    if !args.model.exists() {
        return Err(Error::InvalidInput(format!(
            "model dir not found: {}",
            args.model.display()
        )));
    }

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();
    log::info!("[grad-chain] CUDA ready");

    // ── Phase 1: shared inputs from Python ref ───────────────────────────
    log::info!("[grad-chain] loading ref tensors from {}", args.ref_path.display());
    let ref_tensors = load_file(&args.ref_path, &device)?;
    let get = |k: &str| -> Result<&Tensor> {
        ref_tensors
            .get(k)
            .ok_or_else(|| Error::InvalidInput(format!("ref missing key {k:?}")))
    };

    let noisy = get("noisy")?.to_dtype(DType::BF16)?;
    let input_ids = get("input_ids")?.to_dtype(DType::I32)?;
    let position_ids = get("position_ids")?.clone();
    let vinput_mask = get("vinput_mask")?.to_dtype(DType::BF16)?;
    let token_types = get("token_types")?.to_dtype(DType::BF16)?;
    let timestep = get("timestep")?.to_dtype(DType::BF16)?;
    let t_scalar = scalar1(get("t_scalar")?)?;
    let target_velocity = get("target_velocity")?.to_dtype(DType::F32)?;

    log::info!(
        "[grad-chain] shapes: noisy={:?} ids={:?} pos={:?} vmask={:?} ts={:?} t={}",
        noisy.shape().dims(),
        input_ids.shape().dims(),
        position_ids.shape().dims(),
        vinput_mask.shape().dims(),
        timestep.shape().dims(),
        t_scalar,
    );

    let (t_pos, h_pos, w_pos) = decode_position_ids(&position_ids)?;
    let pos_view = MRopePositions {
        t: &t_pos,
        h: &h_pos,
        w: &w_pos,
    };

    // ── Phase 2: model ───────────────────────────────────────────────────
    let cfg = HiDreamO1Config::dev_8b();
    log::info!("[grad-chain] loading model from {}", args.model.display());
    let loader = HiDreamO1WeightLoader::from_dir(&args.model).map_err(|e| {
        Error::InvalidInput(format!("HiDreamO1WeightLoader: {e}"))
    })?;
    let mut model = loader
        .load_model(&cfg, &device)
        .map_err(|e| Error::InvalidInput(format!("load_model: {e}")))?;

    // ── Phase 3: LoRA stack matching the Python ref exactly ──────────────
    let lora_map = load_file(&args.lora_ref, &device)?;
    let lora = LoraRegistry::new_with_dtype_and_resident(
        &cfg,
        LORA_RANK,
        LORA_ALPHA,
        default_target_suffixes(),
        0,
        &device,
        DType::F32,
        true, // include resident O1 head adapters
    )?;
    load_lora_init_from_map(&lora, &lora_map)?;
    log::info!(
        "[grad-chain] LoRA attached: {} adapters, rank={}, alpha={}",
        lora.len(),
        lora.rank,
        lora.alpha
    );

    // Optionally disable training-time gradient checkpointing. Default is
    // enabled by the trainer; turning it off makes the block_trap probe IDs
    // live on the single (only) tape, which simplifies the additive retain.
    if args.no_checkpoint {
        std::env::set_var("HIDREAM_O1_DISABLE_TRAIN_CHECKPOINT", "1");
        log::info!("[grad-chain] gradient checkpointing DISABLED");
    }

    // ── Phase 4: arm BOTH trap stores ───────────────────────────────────
    // Old soul.md probes (at `HIDREAM_BWD_PROBE_LAYER`) for backwards-
    // compatible diagnostics, and new L2P-style block probes (driven by
    // `HIDREAM_BLOCK_PROBE_LAYER`) for this binary's per-checkpoint diff.
    inference_flame::models::hidream_o1::trap::arm_probes();
    inference_flame::models::hidream_o1::trap::arm_for_env();
    if std::env::var("HIDREAM_BLOCK_PROBE_LAYER").is_err() {
        eprintln!(
            "[grad-chain] HIDREAM_BLOCK_PROBE_LAYER not set — block-probe \
             diagnostics will be empty. Set e.g. \
             HIDREAM_BLOCK_PROBE_LAYER=35 to probe the last decoder layer."
        );
    }

    // ── Phase 5: forward ─────────────────────────────────────────────────
    log::info!("[grad-chain] running forward (autograd enabled) ...");
    let start = std::time::Instant::now();
    let x_pred_full = model.forward_lora(
        &input_ids,
        &timestep,
        &noisy,
        &pos_view,
        &vinput_mask,
        &token_types,
        None,
        Some(&lora),
    )?;
    log::info!(
        "[grad-chain] forward done in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    let x_pred_rows = gather_image_rows(&x_pred_full, &vinput_mask)?;
    let sigma = t_scalar.max(1.0e-3);
    let pred_velocity = noisy
        .to_dtype(DType::F32)?
        .sub(&x_pred_rows.to_dtype(DType::F32)?)?
        .mul_scalar(1.0 / sigma)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?;
    let loss = pred_velocity.sub(&target_velocity)?.square()?.mean()?;
    let loss_val = scalar1(&loss)?;
    log::info!("[grad-chain] loss = {loss_val:.9}");

    // ── Phase 6: register intermediates for grad retention ──────────────
    // Walk both probe stores. With training-time gradient checkpointing on,
    // the OUTER tape sees only block-boundary I/O; intra-block IDs live on
    // the sub-tape. The decoder's `record_block()` already routes those into
    // the additive retain set during recompute.
    let mut retain_ids: HashSet<TensorId> = HashSet::new();
    retain_ids.insert(pred_velocity.id());
    retain_ids.insert(x_pred_rows.id());
    retain_ids.insert(x_pred_full.id());
    AutogradContext::retain_intermediate_grads(retain_ids);

    // ── Phase 7: backward ────────────────────────────────────────────────
    log::info!("[grad-chain] running backward ...");
    let grads = loss.backward()?;
    let intermediate_grads = AutogradContext::take_retained_intermediate_grads();
    let block_probes = inference_flame::models::hidream_o1::trap::take_block_probes();
    let trap_probes = inference_flame::models::hidream_o1::trap::take_probes()
        .unwrap_or_default();
    log::info!(
        "[grad-chain] backward done: {} retained grads | {} block probes | {} trap probes",
        intermediate_grads.len(),
        block_probes.len(),
        trap_probes.len(),
    );

    // ── Phase 8: dump grads ──────────────────────────────────────────────
    let mut out_map: HashMap<String, Tensor> = HashMap::new();

    // 8a) LoRA parameter grads. Naming matches the Python LoRA ref's
    //     `grad_pre.<adapter_name>` keys so the diff script can compare
    //     directly (we use plain `<adapter>.grad` here for clarity).
    let named = lora.named_parameters();
    let mut lora_missing: Vec<String> = Vec::new();
    let mut lora_zero = 0usize;
    for (name, p) in &named {
        match grads.get(p.id()) {
            Some(g) => {
                let g_f32 = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                let sq = g_f32.mul(&g_f32)?.sum()?.to_vec()?[0];
                if sq <= 0.0 {
                    lora_zero += 1;
                }
                out_map.insert(format!("{name}.grad"), g_f32);
            }
            None => lora_missing.push(name.clone()),
        }
    }
    if !lora_missing.is_empty() {
        eprintln!(
            "[grad-chain] WARNING: {}/{} LoRA grads missing: {:?}",
            lora_missing.len(),
            named.len(),
            &lora_missing[..lora_missing.len().min(5)]
        );
    }
    if lora_zero > 0 {
        eprintln!(
            "[grad-chain] WARNING: {}/{} LoRA grads exactly zero",
            lora_zero,
            named.len()
        );
    }

    // 8b) block-probe activation grads (the L2P-style per-checkpoint dump).
    let mut probe_missing: Vec<String> = Vec::new();
    let probe_layer_str = std::env::var("HIDREAM_BLOCK_PROBE_LAYER")
        .unwrap_or_else(|_| "unknown".to_string());
    for (name, id) in &block_probes {
        match intermediate_grads.get(id) {
            Some(g) => {
                let g_f32 = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                out_map.insert(format!("act_grad.block.{name}"), g_f32);
            }
            None => probe_missing.push(name.clone()),
        }
    }
    log::info!(
        "[grad-chain] block-probe layer {}: dumped {} grads, {} missing",
        probe_layer_str,
        block_probes.len() - probe_missing.len(),
        probe_missing.len(),
    );
    if !probe_missing.is_empty() {
        eprintln!(
            "[grad-chain] block probes missing grads: {:?}",
            probe_missing
        );
    }

    // 8c) loss + sanity scalars.
    out_map.insert(
        "_loss".to_string(),
        Tensor::from_vec(vec![loss_val], Shape::from_dims(&[1]), device.clone())?,
    );

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error::InvalidInput(format!("mkdir {}: {e}", parent.display()))
        })?;
    }
    save_tensors(
        &out_map,
        &args.out,
        SerializationFormat::SafeTensors,
    )?;
    log::info!(
        "[grad-chain] wrote {} tensors → {}",
        out_map.len(),
        args.out.display()
    );

    // ── Phase 9: compact per-grad summary ───────────────────────────────
    let mut keys: Vec<&String> = out_map.keys().filter(|k| !k.starts_with('_')).collect();
    keys.sort();
    println!();
    println!("=== Rust grad summary (top probe + LoRA-B grads) ===");
    for k in &keys {
        if !k.starts_with("act_grad.block.") && !k.ends_with(".lora_B.grad") {
            continue;
        }
        let t = &out_map[k.as_str()];
        let v = t.to_vec()?;
        let mut amax = 0.0_f32;
        let mut sumsq = 0.0_f64;
        for x in &v {
            let a = x.abs();
            if a > amax {
                amax = a;
            }
            sumsq += (*x as f64) * (*x as f64);
        }
        let rms = (sumsq / v.len().max(1) as f64).sqrt() as f32;
        println!(
            "  {:<55}  shape={:<24?}  rms={:.3e}  abs.max={:.3e}",
            k,
            t.shape().dims(),
            rms,
            amax
        );
    }
    println!();
    println!("V LoRA-B layer {}: see hidream_o1_grad_chain/grads_rust.safetensors", probe_layer_str);
    println!("Run grad_diff.py for the per-checkpoint cos table vs PyTorch ref.");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("[grad-chain] FATAL: {e:#}");
            ExitCode::from(2)
        }
    }
}
