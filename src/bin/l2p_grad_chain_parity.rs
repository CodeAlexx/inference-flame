//! T3 follow-up — L2P gradient-direction parity vs PyTorch diffsynth.
//!
//! Task #46 from `HANDOFF_2026-05-22_L2P_GRAD_FLOW.md`. The load-bearing
//! question: does our backward pass produce LoRA gradients in the SAME
//! DIRECTION as PyTorch on the same input? Forward parity (cos≥0.999) is
//! settled; this binary settles backward parity.
//!
//! Strategy:
//!   - Reuse `/tmp/l2p_thorough_parity/t3_shared_inputs.safetensors`
//!     ({clean, noise, cap_feats, sigma}) — or fall back to building one
//!     from `cache/boxjana_l2p_512/10.safetensors`.
//!   - Build LoRA ONLY on `layers.0` (8 modules × 2 params = 16 Parameters)
//!     to isolate the test from layer-cascade noise. Match `train_l2p.rs`
//!     init exactly: F32 dtype, `1/sqrt(3·in_features)` Kaiming-uniform-eq
//!     std, zero up; seed = base + idx so each module is distinct.
//!   - Forward + F32 MSE(pred, target) + `loss.backward()`.
//!   - Dump per-Parameter grads to safetensors with PEFT save_key naming.
//!
//! Companion Python reference: `src/models/l2p/parity/t3_grad_direction.py`.
//! Diff harness:               `src/models/l2p/parity/t3_grad_diff.py`.
//!
//! Required env:
//!   LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib
//!   FLAME_ALLOC_POOL=0
//!   FLAME_ASSERT_GRAD_FLOW=1
//!   FLAME_NO_CUDNN_SDPA_BWD=1
//!
//! Usage:
//!   l2p_grad_chain_parity \
//!     --model /home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors \
//!     --shared /tmp/l2p_thorough_parity/t3_shared_inputs.safetensors \
//!     --out   /tmp/l2p_thorough_parity/grads_rust.safetensors

use cudarc::driver::CudaDevice;
use flame_core::autograd::AutogradContext;
use flame_core::parameter::Parameter;
use flame_core::serialization::{load_file, save_tensors, SerializationFormat};
use flame_core::{DType, Error, Result, Shape, Tensor, TensorId};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference_flame::lora::{LoraStack, Slot, TrainEntry};
use inference_flame::models::l2p::{weight_loader::translate_l2p_keys, L2pDiT};

// Mirror train_l2p.rs constants.
const DIM: usize = 3840;
const QKV_OUT: usize = 3 * DIM;
const MLP_HIDDEN: usize = 10240;
const T_EMB_DIM: usize = 256;
const ADALN_OUT: usize = 4 * DIM;
const LORA_RANK: usize = 16;
const LORA_ALPHA: f32 = 16.0;
const LORA_SEED_BASE: u64 = 42;
const SIGMA: f32 = 0.5;

struct Args {
    model_path: String,
    shared_path: String,
    out_path: String,
    sample_path: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_path = String::from(
        "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors",
    );
    let mut shared_path =
        String::from("/tmp/l2p_thorough_parity/t3_shared_inputs.safetensors");
    let mut out_path =
        String::from("/tmp/l2p_thorough_parity/grads_rust.safetensors");
    let mut sample_path = String::from(
        "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512/10.safetensors",
    );
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                i += 1;
                model_path = argv[i].clone();
            }
            "--shared" => {
                i += 1;
                shared_path = argv[i].clone();
            }
            "--out" => {
                i += 1;
                out_path = argv[i].clone();
            }
            "--sample" => {
                i += 1;
                sample_path = argv[i].clone();
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    Args { model_path, shared_path, out_path, sample_path }
}

/// One LoRA target on `layers.0`. Matches the row of the train_l2p table.
#[derive(Clone)]
struct LoraTarget {
    weight_key: String,
    save_key: String,
    in_dim: usize,
    out_dim: usize,
    slot: Slot,
}

fn layer0_targets() -> Vec<LoraTarget> {
    let prefix = "layers.0";
    vec![
        LoraTarget {
            weight_key: format!("{prefix}.attention.qkv.weight"),
            save_key: format!("{prefix}.attention.to_q"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::RowRange { start: 0, len: DIM },
        },
        LoraTarget {
            weight_key: format!("{prefix}.attention.qkv.weight"),
            save_key: format!("{prefix}.attention.to_k"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::RowRange { start: DIM, len: DIM },
        },
        LoraTarget {
            weight_key: format!("{prefix}.attention.qkv.weight"),
            save_key: format!("{prefix}.attention.to_v"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::RowRange { start: 2 * DIM, len: DIM },
        },
        LoraTarget {
            weight_key: format!("{prefix}.attention.out.weight"),
            save_key: format!("{prefix}.attention.to_out.0"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::Full,
        },
        LoraTarget {
            weight_key: format!("{prefix}.feed_forward.w1.weight"),
            save_key: format!("{prefix}.feed_forward.w1"),
            in_dim: DIM,
            out_dim: MLP_HIDDEN,
            slot: Slot::Full,
        },
        LoraTarget {
            weight_key: format!("{prefix}.feed_forward.w2.weight"),
            save_key: format!("{prefix}.feed_forward.w2"),
            in_dim: MLP_HIDDEN,
            out_dim: DIM,
            slot: Slot::Full,
        },
        LoraTarget {
            weight_key: format!("{prefix}.feed_forward.w3.weight"),
            save_key: format!("{prefix}.feed_forward.w3"),
            in_dim: DIM,
            out_dim: MLP_HIDDEN,
            slot: Slot::Full,
        },
        LoraTarget {
            weight_key: format!("{prefix}.adaLN_modulation.0.weight"),
            save_key: format!("{prefix}.adaLN_modulation.0"),
            in_dim: T_EMB_DIM,
            out_dim: ADALN_OUT,
            slot: Slot::Full,
        },
    ]
}

/// Match train_l2p.rs::make_lora_pair exactly. F32 params, std = 1/sqrt(3·in_dim).
fn make_lora_pair(
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    device: &Arc<flame_core::CudaDevice>,
    seed: u64,
) -> Result<(Parameter, Parameter)> {
    let down_std = 1.0_f32 / ((in_dim as f32) * 3.0).sqrt();
    let down = Tensor::randn_seeded(
        Shape::from_dims(&[in_dim, rank]),
        0.0,
        down_std,
        seed,
        device.clone(),
    )?
    .to_dtype(DType::F32)?
    .requires_grad_(true);
    let up = Tensor::zeros_dtype(
        Shape::from_dims(&[rank, out_dim]),
        DType::F32,
        device.clone(),
    )?
    .requires_grad_(true);
    Ok((Parameter::new(down), Parameter::new(up)))
}

fn ensure_shared_inputs(
    shared_path: &Path,
    sample_path: &Path,
    device: &Arc<flame_core::CudaDevice>,
) -> Result<()> {
    if shared_path.exists() {
        println!("[grad-rust] reusing shared inputs: {}", shared_path.display());
        return Ok(());
    }
    println!(
        "[grad-rust] building shared inputs from {} → {}",
        sample_path.display(),
        shared_path.display()
    );
    if let Some(parent) = shared_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error::InvalidOperation(format!("mkdir {}: {e}", parent.display()))
        })?;
    }
    let sample = load_file(
        sample_path.to_str().ok_or_else(|| {
            Error::InvalidInput("sample_path not UTF-8".into())
        })?,
        device,
    )?;
    let pixel = sample
        .get("pixel")
        .ok_or_else(|| Error::InvalidInput("sample missing 'pixel'".into()))?;
    let d = pixel.shape().dims().to_vec();
    if d.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "pixel shape {:?} != [3, H, W]",
            d
        )));
    }
    let clean = pixel
        .to_dtype(DType::BF16)?
        .reshape(&[1, d[0], d[1], d[2]])?;
    // Seed=42 noise matching `train_l2p` default (Box-Muller via flame
    // `randn_seeded`). Reuse seed=42 for reproducibility.
    let noise = Tensor::randn_seeded(
        clean.shape().clone(),
        0.0,
        1.0,
        42,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let cap_feats = sample
        .get("cap_feats")
        .ok_or_else(|| Error::InvalidInput("sample missing 'cap_feats'".into()))?
        .to_dtype(DType::BF16)?;
    // Ensure cap is [1, S, 2560].
    let cap_dims = cap_feats.shape().dims().to_vec();
    let cap = if cap_dims.len() == 2 {
        cap_feats.reshape(&[1, cap_dims[0], cap_dims[1]])?
    } else {
        cap_feats
    };
    let sigma_t = Tensor::from_vec(
        vec![SIGMA],
        Shape::from_dims(&[1]),
        device.clone(),
    )?;
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("clean".into(), clean);
    out.insert("noise".into(), noise);
    out.insert("cap_feats".into(), cap);
    out.insert("sigma".into(), sigma_t);
    save_tensors(&out, shared_path, SerializationFormat::SafeTensors)?;
    println!("[grad-rust] wrote shared inputs ({} keys)", out.len());
    Ok(())
}

fn main() -> Result<()> {
    // Diagnostic pre-flight (mirror train_l2p).
    if std::env::var("FLAME_ALLOC_POOL").as_deref() != Ok("0") {
        eprintln!(
            "WARNING: FLAME_ALLOC_POOL is not set to 0. L2P forward may OOM \
             without this. Recommend `FLAME_ALLOC_POOL=0`."
        );
    }
    if std::env::var("FLAME_AUTOGRAD_OFF").as_deref() == Ok("1") {
        eprintln!("FATAL: FLAME_AUTOGRAD_OFF=1 disables training. Unset.");
        std::process::exit(2);
    }

    let args = parse_args();
    let device = CudaDevice::new(0)
        .map_err(|e| Error::InvalidOperation(format!("CUDA init: {e:?}")))?;
    println!("[grad-rust] CUDA ready");

    flame_core::config::set_default_dtype(DType::BF16);

    // ── Phase 1: shared inputs ────────────────────────────────────────
    let shared_pb = PathBuf::from(&args.shared_path);
    let sample_pb = PathBuf::from(&args.sample_path);
    ensure_shared_inputs(&shared_pb, &sample_pb, &device)?;

    let shared = load_file(&args.shared_path, &device)?;
    let mut clean = shared
        .get("clean")
        .ok_or_else(|| Error::InvalidInput("missing 'clean'".into()))?
        .to_dtype(DType::BF16)?;
    let mut noise = shared
        .get("noise")
        .ok_or_else(|| Error::InvalidInput("missing 'noise'".into()))?
        .to_dtype(DType::BF16)?;
    let cap_feats = shared
        .get("cap_feats")
        .ok_or_else(|| Error::InvalidInput("missing 'cap_feats'".into()))?
        .to_dtype(DType::BF16)?;
    // BISECT 2026-05-23: optionally crop to 256² to fit no-GC parity in memory.
    let half_res = std::env::var("L2P_PARITY_HALF_RES").as_deref() == Ok("1");
    if half_res {
        let d = clean.shape().dims().to_vec();
        let new_h = d[2] / 2;
        let new_w = d[3] / 2;
        clean = clean.narrow(2, 0, new_h)?.narrow(3, 0, new_w)?.contiguous()?;
        noise = noise.narrow(2, 0, new_h)?.narrow(3, 0, new_w)?.contiguous()?;
        eprintln!("[grad-rust] HALF_RES bisect: clean={:?} noise={:?}", clean.shape().dims(), noise.shape().dims());
    }
    let sigma_val = shared
        .get("sigma")
        .ok_or_else(|| Error::InvalidInput("missing 'sigma'".into()))?
        .to_vec()?[0];
    println!(
        "[grad-rust] inputs: clean={:?} noise={:?} cap={:?} sigma={}",
        clean.shape().dims(),
        noise.shape().dims(),
        cap_feats.shape().dims(),
        sigma_val,
    );
    if (sigma_val - SIGMA).abs() > 1e-6 {
        eprintln!(
            "WARNING: shared sigma {sigma_val} != pinned {SIGMA}; using shared value.",
        );
    }

    // ── Phase 2: model ────────────────────────────────────────────────
    println!("[grad-rust] loading L2P model: {}", args.model_path);
    let raw = load_file(&args.model_path, &device)?;
    let translated = translate_l2p_keys(raw)?;
    let mut model = L2pDiT::new_resident(translated, device.clone());
    // BISECT 2026-05-23: try with grad_checkpoint DISABLED to test if the
    // checkpoint-recompute path is the cascade source. Set env
    // L2P_PARITY_NO_GC=1 to force-disable.
    let force_no_gc = std::env::var("L2P_PARITY_NO_GC").as_deref() == Ok("1");
    if force_no_gc {
        model.set_grad_checkpoint(false);
        eprintln!("[grad-rust] grad_checkpoint DISABLED for bisect");
    } else {
        // Required at 512² on 24 GB to fit the backward activation tape.
        // Adds ~30% compute overhead but is the production training-time
        // configuration (matches `--grad-checkpoint` flag on train_l2p).
        model.set_grad_checkpoint(true);
    }
    println!("[grad-rust] model loaded (grad checkpoint enabled)");

    // ── Phase 3: LoRA stack on layers.0 only ──────────────────────────
    let targets = layer0_targets();
    let n_targets = targets.len();
    let scale = LORA_ALPHA / LORA_RANK as f32;
    let mut train_map: HashMap<String, Vec<TrainEntry>> = HashMap::new();
    let mut params: Vec<Parameter> = Vec::new();
    let mut named: Vec<(String, Parameter)> = Vec::new();
    for (idx, target) in targets.into_iter().enumerate() {
        let (down, up) = make_lora_pair(
            target.in_dim,
            target.out_dim,
            LORA_RANK,
            &device,
            LORA_SEED_BASE + idx as u64,
        )?;
        params.push(down.clone());
        params.push(up.clone());
        named.push((target.save_key.clone(), down.clone()));
        train_map
            .entry(target.weight_key.clone())
            .or_default()
            .push(TrainEntry {
                slot: target.slot,
                down: down.clone(),
                up: up.clone(),
                scale,
            });
        // Also push up by a separate (matching) name suffix so the dump can
        // distinguish A vs B. We use a 2-tuple list of (save_key + ".lora_X").
    }
    // Rebuild named with explicit A/B distinction.
    named.clear();
    let mut idx = 0usize;
    for (weight_key, entries) in &train_map {
        let _ = weight_key;
        for e in entries {
            let _ = e;
        }
        idx += entries.len();
    }
    let _ = idx;
    // Walk train_map deterministically by reconstructing target order.
    let walk = layer0_targets();
    let mut grad_keys: Vec<(String, Parameter, Parameter)> = Vec::with_capacity(walk.len());
    for target in &walk {
        // Find the entry in train_map matching slot.
        let entries = train_map
            .get(&target.weight_key)
            .expect("weight_key was inserted");
        // Match by save_key index ordering — entries are pushed in target order.
        let pos_in_block = walk
            .iter()
            .filter(|t| t.weight_key == target.weight_key)
            .position(|t| t.save_key == target.save_key)
            .expect("save_key in target list");
        let e = &entries[pos_in_block];
        grad_keys.push((target.save_key.clone(), e.down.clone(), e.up.clone()));
    }
    println!(
        "[grad-rust] LoRA: {} targets on layers.0 ({} Parameters)",
        n_targets,
        params.len()
    );
    // Dump LoRA init weights so the Python reference can load identical
    // values. flame_core::randn_seeded uses Box-Muller; PyTorch uses
    // MT19937 — same seed gives different bits. Without identical A
    // matrices the LoRA branch output differs, and lora_B grads (which
    // depend on A and the upstream chain) would diverge for a benign
    // reason. We dump A=[in,rank], B=[rank,out] native shapes.
    {
        let mut init_map: HashMap<String, Tensor> = HashMap::new();
        for (save_key, down, up) in &grad_keys {
            let d = down.tensor()?;
            let u = up.tensor()?;
            init_map.insert(
                format!("{save_key}.lora_A_init"),
                if d.dtype() == DType::F32 { d } else { d.to_dtype(DType::F32)? },
            );
            init_map.insert(
                format!("{save_key}.lora_B_init"),
                if u.dtype() == DType::F32 { u } else { u.to_dtype(DType::F32)? },
            );
        }
        let init_path = "/tmp/l2p_thorough_parity/lora_init_rust.safetensors";
        if let Some(parent) = Path::new(init_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        save_tensors(
            &init_map,
            Path::new(init_path),
            SerializationFormat::SafeTensors,
        )?;
        println!(
            "[grad-rust] wrote LoRA init ({} tensors) → {}",
            init_map.len(),
            init_path
        );
    }

    let stack = Arc::new(LoraStack::new_training(train_map));
    model.set_lora(stack);

    // ── Phase 4: forward ─────────────────────────────────────────────
    let noisy = clean
        .mul_scalar(1.0 - sigma_val)?
        .add(&noise.mul_scalar(sigma_val)?)?;
    let target = noise.sub(&clean)?;
    let v_in = Tensor::from_vec(
        vec![sigma_val],
        Shape::from_dims(&[1]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    println!("[grad-rust] running forward (with capture)...");
    // Arm L2P intra-block probes for layer specified by L2P_BLOCK_PROBE_LAYER.
    inference_flame::models::l2p::block_trap::arm_for_env();
    let mut capture: HashMap<String, Tensor> = HashMap::new();
    let pred = model.forward_with_capture(&noisy, &v_in, &cap_feats, &mut capture)?;
    let block_probes = inference_flame::models::l2p::block_trap::take();
    if !block_probes.is_empty() {
        eprintln!("[grad-rust] block probes captured: {} ({:?})", block_probes.len(), {
            let mut ks: Vec<_> = block_probes.keys().collect();
            ks.sort();
            ks
        });
    }
    println!(
        "[grad-rust] pred {:?} target {:?}  captured {} intermediates",
        pred.shape().dims(),
        target.shape().dims(),
        capture.len(),
    );
    if pred.shape().dims() != target.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "shape mismatch: pred={:?} target={:?}",
            pred.shape().dims(),
            target.shape().dims()
        )));
    }

    // Register all captured intermediates for grad retention. Tensor::clone
    // (via cap! macro's contiguous() = self.clone() fast path for already-
    // contiguous tensors) preserves TensorId, so these IDs are the same
    // nodes the autograd tape recorded during forward.
    let mut retain_ids: HashSet<TensorId> = HashSet::new();
    let mut name_for_id: HashMap<TensorId, String> = HashMap::new();
    for (name, t) in &capture {
        retain_ids.insert(t.id());
        name_for_id.insert(t.id(), name.clone());
    }
    // Also retain block-probe IDs.
    for (name, id) in &block_probes {
        retain_ids.insert(*id);
        name_for_id.insert(*id, format!("block.{name}"));
    }
    println!("[grad-rust] retaining grads for {} intermediate tensor IDs", retain_ids.len());
    AutogradContext::retain_intermediate_grads(retain_ids);

    // Optional: dump SDPA saved Q/K/V + fwd output for the target layer's
    // attention call. Captured BEFORE backward so the snapshot reflects the
    // exact tensors the backward dispatch will read. Replay-test for the
    // L2P body cascade hunt: load this dump into
    // `flame-core/tests/sdpa_l2p_replay.rs`, run the kernel in isolation,
    // and decide kernel-bug vs autograd-context contamination.
    //
    // Trigger:   L2P_DUMP_SDPA_INPUTS=1
    // Layer:     L2P_BLOCK_PROBE_LAYER (reuses existing probe gate)
    // Probe key: "e0.sdpa_out" (output of `sdpa(&q, &k, &v, None)` in
    //             joint_attention)
    let dump_sdpa_inputs = std::env::var("L2P_DUMP_SDPA_INPUTS").as_deref() == Ok("1");
    // 2026-05-29: flame-core removed `FlashAttentionSavedSnapshot` +
    // `lookup_flash_attention_saved`. Local stub keeps this default-off SDPA
    // dump probe compiling at HEAD; it never runs (lookup yields None). The
    // grad-direction parity in Phase 5 below is unaffected.
    #[allow(dead_code)]
    struct FlashAttnSnapStub {
        query: Tensor, key: Tensor, value: Tensor,
        output: Option<Tensor>, stats: Option<Tensor>,
        scale: f32, causal: bool, padding_lens: Option<(usize, usize)>,
    }
    let mut sdpa_snapshot: Option<(FlashAttnSnapStub, TensorId)> = None;
    let mut sdpa_out_fwd_clone: Option<Tensor> = None;
    if dump_sdpa_inputs {
        if let Some(&e0_id) = block_probes.get("e0.sdpa_out") {
            match Option::<FlashAttnSnapStub>::None {
                Some(snap) => {
                    eprintln!(
                        "[grad-rust] SDPA snapshot for e0.sdpa_out (id={:?}): q={:?} k={:?} v={:?} pad={:?} scale={:.6} causal={}",
                        e0_id,
                        snap.query.shape().dims(),
                        snap.key.shape().dims(),
                        snap.value.shape().dims(),
                        snap.padding_lens,
                        snap.scale,
                        snap.causal,
                    );
                    sdpa_snapshot = Some((snap, e0_id));
                    // Also stash the forward output of e0.sdpa_out from the
                    // capture map (when present — note that block_probes is
                    // populated by `aprobe!` in dit.rs, but it's just the id;
                    // we don't have the tensor handle here unless someone
                    // also added to `capture`). Best-effort: scan capture.
                    for (name, t) in &capture {
                        if t.id() == e0_id {
                            sdpa_out_fwd_clone = Some(t.clone());
                            eprintln!("[grad-rust]   also captured fwd output via capture[{name}]");
                            break;
                        }
                    }
                }
                None => {
                    eprintln!(
                        "[grad-rust] WARNING: L2P_DUMP_SDPA_INPUTS=1 set but no Op::FlashAttention entry found for e0.sdpa_out id={:?}",
                        e0_id,
                    );
                }
            }
        } else {
            eprintln!(
                "[grad-rust] WARNING: L2P_DUMP_SDPA_INPUTS=1 set but no 'e0.sdpa_out' in block_probes. Is L2P_BLOCK_PROBE_LAYER set?"
            );
        }
    }

    // ── Phase 5: loss + backward ─────────────────────────────────────
    let pred_f32 = pred.to_dtype(DType::F32)?;
    let target_f32 = target.to_dtype(DType::F32)?;
    let diff = pred_f32.sub(&target_f32)?;
    let loss = diff.mul(&diff)?.mean()?;
    let loss_val = loss.to_vec()?[0];
    println!("[grad-rust] loss = {:.6}", loss_val);

    println!("[grad-rust] running backward...");
    let grads = loss.backward()?;
    let intermediate_grads = AutogradContext::take_retained_intermediate_grads();
    println!(
        "[grad-rust] backward done, captured {} intermediate grads",
        intermediate_grads.len()
    );

    // ── Phase 6: dump grads ──────────────────────────────────────────
    let mut out_map: HashMap<String, Tensor> = HashMap::new();
    let mut missing: Vec<String> = Vec::new();
    let mut zero_count = 0usize;
    for (save_key, down, up) in &grad_keys {
        let a_name = format!("{save_key}.lora_A.grad");
        let b_name = format!("{save_key}.lora_B.grad");
        match grads.get(down.id()) {
            Some(g) => {
                let g_f32 = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                let s = g_f32.mul(&g_f32)?.sum()?.to_vec()?[0];
                if s <= 0.0 {
                    zero_count += 1;
                }
                out_map.insert(a_name, g_f32);
            }
            None => missing.push(a_name),
        }
        match grads.get(up.id()) {
            Some(g) => {
                let g_f32 = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                let s = g_f32.mul(&g_f32)?.sum()?.to_vec()?[0];
                if s <= 0.0 {
                    zero_count += 1;
                }
                out_map.insert(b_name, g_f32);
            }
            None => missing.push(b_name),
        }
    }
    if !missing.is_empty() {
        eprintln!(
            "[grad-rust] WARNING: {} grad tensors missing: {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        );
    }
    if zero_count > 0 {
        eprintln!(
            "[grad-rust] WARNING: {}/{} grads are exactly zero",
            zero_count,
            out_map.len()
        );
    }

    // Dump intermediate activation gradients with `act_grad.<name>` keys so
    // the Python diff harness can match them against `register_full_backward_hook`
    // outputs on the diffsynth side.
    let mut intermediate_missing: Vec<String> = Vec::new();
    for (name, t) in &capture {
        let id = t.id();
        match intermediate_grads.get(&id) {
            Some(g) => {
                let g_f32 = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                out_map.insert(format!("act_grad.{name}"), g_f32);
            }
            None => intermediate_missing.push(name.clone()),
        }
    }
    if !intermediate_missing.is_empty() {
        eprintln!(
            "[grad-rust] WARNING: {} intermediate grads missing (no grad reached these tensors): {:?}",
            intermediate_missing.len(),
            &intermediate_missing[..intermediate_missing.len().min(8)]
        );
    }
    println!(
        "[grad-rust] dumped {} intermediate activation grads (out of {} captured points)",
        capture.len() - intermediate_missing.len(),
        capture.len(),
    );

    // Also dump block-probe grads with `act_grad.block.<name>` keys.
    let mut block_probe_missing: Vec<String> = Vec::new();
    for (probe_name, id) in &block_probes {
        match intermediate_grads.get(id) {
            Some(g) => {
                let g_f32 = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                out_map.insert(format!("act_grad.block.{probe_name}"), g_f32);
            }
            None => block_probe_missing.push(probe_name.clone()),
        }
    }
    if !block_probes.is_empty() {
        println!(
            "[grad-rust] dumped {} block-probe grads (missing {})",
            block_probes.len() - block_probe_missing.len(),
            block_probe_missing.len(),
        );
        if !block_probe_missing.is_empty() {
            eprintln!("[grad-rust] block probes missing grads: {:?}", block_probe_missing);
        }
    }

    // Dump the SDPA backward inputs as a sibling safetensors file. Written
    // to a fixed path so the flame-core replay test knows where to find it:
    //   /tmp/l2p_thorough_parity/sdpa_inputs_layer<N>.safetensors
    if let Some((snap, e0_id)) = sdpa_snapshot {
        let layer_str = std::env::var("L2P_BLOCK_PROBE_LAYER").unwrap_or_else(|_| "X".into());
        let dump_path = format!(
            "/tmp/l2p_thorough_parity/sdpa_inputs_layer{}.safetensors",
            layer_str,
        );
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // All tensors saved as F32 for portability + numerical-comparison
        // ground truth in the replay test. Caller can re-cast to BF16 to
        // match the production path.
        let to_f32 = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() == DType::F32 { Ok(t.clone()) } else { t.to_dtype(DType::F32) }
        };
        map.insert("saved_q".into(), to_f32(&snap.query)?);
        map.insert("saved_k".into(), to_f32(&snap.key)?);
        map.insert("saved_v".into(), to_f32(&snap.value)?);
        if let Some(o) = snap.output.as_ref() {
            map.insert("saved_output".into(), to_f32(o)?);
        }
        if let Some(s) = snap.stats.as_ref() {
            map.insert("saved_stats".into(), to_f32(s)?);
        }
        // Also dump the captured fwd output tensor handle if available.
        if let Some(fwd) = sdpa_out_fwd_clone.as_ref() {
            map.insert("output_fwd_from_capture".into(), to_f32(fwd)?);
        }
        // The output_grad flowing INTO SDPA backward is the intermediate
        // grad of e0.sdpa_out, captured by retain_intermediate_grads.
        if let Some(g) = intermediate_grads.get(&e0_id) {
            map.insert("output_grad".into(), to_f32(g)?);
        } else {
            eprintln!("[grad-rust] WARNING: no intermediate grad captured for e0.sdpa_out id={:?}", e0_id);
        }
        // Metadata (scale, causal, padding_lens) as singleton tensors.
        let device_m = snap.query.device().clone();
        map.insert(
            "_meta_scale".into(),
            Tensor::from_vec(vec![snap.scale], Shape::from_dims(&[1]), device_m.clone())?,
        );
        map.insert(
            "_meta_causal".into(),
            Tensor::from_vec(vec![if snap.causal { 1.0 } else { 0.0 }], Shape::from_dims(&[1]), device_m.clone())?,
        );
        if let Some((rq, rk)) = snap.padding_lens {
            map.insert(
                "_meta_padding_lens".into(),
                Tensor::from_vec(vec![rq as f32, rk as f32], Shape::from_dims(&[2]), device_m.clone())?,
            );
        }
        if let Some(parent) = Path::new(&dump_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        save_tensors(&map, Path::new(&dump_path), SerializationFormat::SafeTensors)?;
        println!("[grad-rust] wrote SDPA replay dump ({} tensors) → {}", map.len(), dump_path);
        // Echo the shapes so the test author can sanity-check.
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();
        for k in keys {
            let t = &map[k];
            println!(
                "  {:<35}  shape={:?}  dtype={:?}",
                k,
                t.shape().dims(),
                t.dtype()
            );
        }
    }

    if let Some(parent) = Path::new(&args.out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    // Also stash the loss value for sanity.
    let loss_t = Tensor::from_vec(
        vec![loss_val],
        Shape::from_dims(&[1]),
        device.clone(),
    )?;
    out_map.insert("_loss".into(), loss_t);
    save_tensors(
        &out_map,
        Path::new(&args.out_path),
        SerializationFormat::SafeTensors,
    )?;
    println!(
        "[grad-rust] wrote {} grad tensors → {}",
        out_map.len(),
        args.out_path
    );

    // ── Brief per-grad summary ──────────────────────────────────────
    let mut keys: Vec<&String> = out_map.keys().filter(|k| !k.starts_with('_')).collect();
    keys.sort();
    for k in keys {
        let t = &out_map[k];
        let v = t.to_vec()?;
        let mut amax = 0.0_f32;
        let mut amean = 0.0_f64;
        for x in &v {
            let a = x.abs();
            if a > amax {
                amax = a;
            }
            amean += a as f64;
        }
        amean /= v.len().max(1) as f64;
        println!(
            "  {:<50}  shape={:?}  abs.mean={:.3e}  abs.max={:.3e}",
            k,
            t.shape().dims(),
            amean as f32,
            amax
        );
    }

    // Touch unused constant so the compiler can't elide.
    let _ = QKV_OUT;
    Ok(())
}
