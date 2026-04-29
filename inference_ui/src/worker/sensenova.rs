//! SenseNova-U1 8B-MoT — text-to-image worker.
//!
//! Mirrors the structure of `worker/chroma.rs` and the underlying generation
//! pipeline of `inference-flame/src/bin/sensenova_u1_gen.rs::run_t2i`. Only
//! the non-think T2I path is wired here — chat / VQA / think-mode / it2i are
//! deliberately out of scope for this UI worker (covered by the standalone
//! CLI bins).
//!
//! ## Architecture quick-recap
//!
//! - **Single weights dir** at `/home/alex/.serenity/models/sensenova_u1`
//!   (Qwen3-8B backbone in MoT mode, ~32.7 GB BF16, sharded in 8 safetensors).
//! - The model is loaded **once** on first call and kept resident across jobs.
//!   Block weights live in pinned-host RAM via `BlockOffloader`; only the
//!   active block(s) sit in VRAM at any given moment, so peak GPU memory is
//!   ~11 GB regardless of model size.
//! - Tokenizer is constructed in-process from `vocab.json` + `merges.txt` +
//!   `added_tokens.json` (no unified `tokenizer.json` ships with the weights).
//!
//! ## VRAM budget on a 24 GB card
//!
//! - Resident loader pinned host:           ~32 GB host RAM (not VRAM)
//! - Block offloader steady-state on GPU:   ~11 GB
//! - Activations (1024² latent + workspace): ~2-3 GB
//!
//! ## Per-gen cost target (1024², 50 steps)
//!
//! - First job ever: ~80 s warm load + ~130 s denoise = ~3.5 min
//! - Subsequent jobs (resident model): ~130 s denoise (50 × 2.6 s/step)
//!
//! Matches the standalone CLI bin's measured baseline.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::CudaDevice;
use egui::ColorImage;
use rand::{Rng, SeedableRng};

use flame_core::{DType, Shape, Tensor};

use inference_flame::models::sensenova_u1::{KvCache, SenseNovaU1, TimeOrScale};

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use super::{GenerateJob, UiMsg, WorkerEvent};

// ===========================================================================
// Weight + tokenizer paths.
// ===========================================================================

/// Default weights directory. The Base ComboBox surfaces a placeholder filename
/// (`sensenova-u1.safetensors`); paths::resolve_image_model_path returns
/// `None` for SenseNova so this const is always used.
const SENSENOVA_WEIGHTS_DIR: &str = "/home/alex/.serenity/models/sensenova_u1";

// ===========================================================================
// Generation defaults (match `bin/sensenova_u1_gen.rs::Args::defaults()`).
// ===========================================================================

const DEFAULT_STEPS: u32 = 50;
const DEFAULT_CFG: f32 = 4.0;
const DEFAULT_SEED: u64 = 42;

/// Reference timestep_shift from the python `t2i_generate` defaults.
const TIMESTEP_SHIFT: f32 = 3.0;

/// System message embedded in the cond branch (mirrors the CLI bin verbatim).
const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands and executes ",
    "user intent.\n\nYou support two modes:\n\n",
    "1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. ",
    "Put all reasoning inside the block using plain text. DO NOT include any image tags. ",
    "Keep it reasonable and directly useful for producing the final image.\n\n",
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n",
    "Task Types:\n\nA. Text-to-Image Generation:\n",
    "- Generate a high-quality image based on the user's description.\n",
    "- Ensure visual clarity, semantic consistency, and completeness.\n",
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n",
    "B. Image Editing:\n",
    "- Use the provided image(s) as input or reference for modification or transformation.\n",
    "- The result can be an edited image or a new image based on the reference(s).\n",
    "- Preserve all unspecified attributes unless explicitly changed.\n\n",
    "General Rules:\n",
    "- For any visible text in the image, follow the language specified for the rendered text in ",
    "the user's description, not the language of the prompt. If no language is specified, use the ",
    "user's input language."
);

// ===========================================================================
// State
// ===========================================================================

/// Worker-local SenseNova-U1 resources. Both the tokenizer and the model are
/// kept resident across jobs once initialised — the model load is ~80 s on
/// first call (BlockOffloader populates pinned host RAM) and re-doing that
/// per job would be unusable in a UI.
pub struct SenseNovaState {
    pub device: Arc<CudaDevice>,
    /// Lazily built on first call. Cheap (~ms) — kept here so we don't hit the
    /// disk per job.
    tokenizer: Option<Tokenizer>,
    /// Lazily built on first call. ~80 s load with BlockOffloader. After that,
    /// `forward_und` / `forward_gen` stream blocks from pinned host RAM.
    model: Option<SenseNovaU1>,
}

impl SenseNovaState {
    pub fn new() -> Result<Self, String> {
        let device =
            CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e:?}"))?;
        Ok(Self {
            device,
            tokenizer: None,
            model: None,
        })
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

pub fn run(
    job: &GenerateJob,
    state: &mut SenseNovaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) {
    let steps: u32 = if job.steps == 0 { DEFAULT_STEPS } else { job.steps };
    let cfg: f32 = if job.cfg <= 0.0 { DEFAULT_CFG } else { job.cfg };

    let _ = ev_tx.send(WorkerEvent::Started {
        id: job.id,
        job: job.clone(),
        total_steps: steps,
    });
    ctx.request_repaint();

    match run_inner(job, steps, cfg, state, ui_rx, ev_tx, ctx, pending) {
        Ok(image) => {
            let _ = ev_tx.send(WorkerEvent::Done {
                id: job.id,
                image,
                prompt: job.prompt.clone(),
            });
            ctx.request_repaint();
        }
        Err(RunError::Cancelled) => {
            let _ = ev_tx.send(WorkerEvent::Failed {
                id: job.id,
                error: "cancelled".into(),
            });
            ctx.request_repaint();
        }
        Err(RunError::Other(msg)) => {
            log::warn!("SenseNova-U1 job {} failed: {msg}", job.id);
            let _ = ev_tx.send(WorkerEvent::Failed {
                id: job.id,
                error: msg,
            });
            ctx.request_repaint();
        }
    }
}

// ===========================================================================
// Internals
// ===========================================================================

enum RunError {
    Cancelled,
    Other(String),
}

impl<E: std::fmt::Debug> From<E> for RunError {
    fn from(e: E) -> Self {
        RunError::Other(format!("{e:?}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    job: &GenerateJob,
    steps: u32,
    cfg_scale: f32,
    state: &mut SenseNovaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    // Final pool-flush wrapper, same pattern as worker/chroma.rs: guarantee
    // we trim CUDA mempool on every exit path so the next job sees a clean
    // pool. The block offloader keeps pinned host RAM resident across jobs;
    // we only flush GPU-side per-step temporaries.
    let result = (|| -> Result<ColorImage, RunError> {
        run_inner_body(job, steps, cfg_scale, state, ui_rx, ev_tx, ctx, pending)
    })();
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::device::trim_cuda_mempool(0);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_inner_body(
    job: &GenerateJob,
    steps: u32,
    cfg_scale: f32,
    state: &mut SenseNovaState,
    ui_rx: &Receiver<UiMsg>,
    ev_tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    pending: &mut VecDeque<UiMsg>,
) -> Result<ColorImage, RunError> {
    let device = state.device.clone();

    // Resolve weights dir. `job.path` is the Base ComboBox override; for
    // SenseNova the placeholder filename is informational and the on-disk
    // layout is a directory, so the override is ignored — we always use
    // `SENSENOVA_WEIGHTS_DIR`. (Mirrors the CLI bin's --weights default.)
    let weights_dir = PathBuf::from(SENSENOVA_WEIGHTS_DIR);
    if !weights_dir.is_dir() {
        return Err(RunError::Other(format!(
            "SenseNova-U1 weights dir not found at {}",
            weights_dir.display()
        )));
    }

    // -------- 1. Tokenizer + model lazy-load --------
    if state.tokenizer.is_none() {
        log::info!("SenseNova-U1: building tokenizer from {}", weights_dir.display());
        let t0 = Instant::now();
        state.tokenizer = Some(
            build_tokenizer(&weights_dir)
                .map_err(|e| RunError::Other(format!("tokenizer build: {e}")))?,
        );
        log::info!("SenseNova-U1: tokenizer ready in {:.2}s", t0.elapsed().as_secs_f32());
    }
    if state.model.is_none() {
        log::info!("SenseNova-U1: loading model from {}", weights_dir.display());
        let t0 = Instant::now();
        state.model = Some(
            SenseNovaU1::load(&weights_dir, &device)
                .map_err(|e| RunError::Other(format!("SenseNovaU1::load: {e:?}")))?,
        );
        log::info!("SenseNova-U1: model ready in {:.2}s", t0.elapsed().as_secs_f32());
    }
    let tok = state.tokenizer.as_ref().expect("tokenizer just initialised");
    let model = state.model.as_mut().expect("model just initialised");

    drain_pending(ui_rx, pending)?;

    // -------- 2. Geometry validation --------
    let cfg = model.config().clone();
    let merge = cfg.downsample_ratio.recip().round() as usize; // 2
    let patch = cfg.patch_size; // 16
    let p_eff = patch * merge; // 32

    let height = job.height as usize;
    let width = job.width as usize;
    if height == 0 || width == 0 {
        return Err(RunError::Other("zero-sized output".into()));
    }
    if width % p_eff != 0 || height % p_eff != 0 {
        return Err(RunError::Other(format!(
            "SenseNova-U1 requires width × height to be divisible by {p_eff}; got {width}×{height}"
        )));
    }

    let grid_h = height / patch;
    let grid_w = width / patch;
    let token_h = grid_h / merge;
    let token_w = grid_w / merge;
    let l_tokens = token_h * token_w;
    let b: usize = 1;

    // -------- 3. Tokenise cond + uncond (non-think branch only) --------
    // Non-think variant: append the literal `<think>\n\n</think>\n\n<img>`
    // continuation so the assistant emits an empty think block followed by
    // the <img> sentinel. Mirrors `bin/sensenova_u1_gen.rs::run_t2i` when
    // think_mode=false.
    let cond_query = build_t2i_query(
        SYSTEM_MESSAGE_FOR_GEN,
        &job.prompt,
        "<think>\n\n</think>\n\n<img>",
    );
    let uncond_query = build_t2i_query("", "", "<img>");
    let cond_ids =
        encode_query(tok, &cond_query).map_err(|e| RunError::Other(format!("cond encode: {e}")))?;
    let uncond_ids = encode_query(tok, &uncond_query)
        .map_err(|e| RunError::Other(format!("uncond encode: {e}")))?;
    log::info!(
        "SenseNova-U1: cond tokens={} uncond tokens={}",
        cond_ids.len(),
        uncond_ids.len()
    );

    // -------- 4. Prefix forwards --------
    let t0 = Instant::now();
    let (cond_cache, _cond_last_hidden) = model
        .forward_und(&cond_ids)
        .map_err(|e| RunError::Other(format!("forward_und cond: {e:?}")))?;
    let (uncond_cache, _) = model
        .forward_und(&uncond_ids)
        .map_err(|e| RunError::Other(format!("forward_und uncond: {e:?}")))?;
    log::info!(
        "SenseNova-U1: prefix forwards done in {:.2}s",
        t0.elapsed().as_secs_f32()
    );

    drain_pending(ui_rx, pending)?;

    // -------- 5. Build noise image + patchify constants --------
    let seed_u64: u64 = if job.seed < 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(DEFAULT_SEED)
    } else if job.seed == 0 {
        // Treat 0 as "use the default" — the bin's default is 42 and matches
        // upstream python.
        DEFAULT_SEED
    } else {
        job.seed as u64
    };

    let noise_scale = model.compute_noise_scale(grid_h, grid_w);
    let mut img = make_noise_image(seed_u64, &[b, 3, height, width], &device)
        .map_err(|e| RunError::Other(format!("noise image: {e}")))?;
    img = img
        .mul_scalar(noise_scale)
        .map_err(|e| RunError::Other(format!("noise scale: {e:?}")))?;

    log::info!(
        "SenseNova-U1: grid={}x{}  tokens={}x{} (L={})  noise_scale={:.4}",
        grid_h,
        grid_w,
        token_h,
        token_w,
        l_tokens,
        noise_scale
    );

    // -------- 6. Build timestep grid --------
    let mut t_uniform: Vec<f32> = (0..=steps as usize)
        .map(|i| i as f32 / steps as f32)
        .collect();
    t_uniform = model.apply_time_schedule(&t_uniform, l_tokens, TIMESTEP_SHIFT);

    drain_pending(ui_rx, pending)?;

    // -------- 7. CFG denoise loop (2 forwards/step via forward_gen) --------
    let s_norm = noise_scale / cfg.noise_scale_max_value;
    let t_start = Instant::now();

    for step in 0..steps as usize {
        drain_pending(ui_rx, pending)?;

        let t = t_uniform[step];
        let t_next = t_uniform[step + 1];
        let step_t0 = Instant::now();

        // Scoped block: every per-step temporary drops at `}` BEFORE the next
        // iteration's allocations. Mirrors run_t2i in the CLI bin and the
        // chroma worker — prevents pool growth across steps.
        let next_img = {
            // z (target dim) and pixel_values (input to gen embedder)
            let z = patchify(&img, p_eff, false)?; // [B, L, 3072]
            let pixel_values = patchify(&img, patch, true)?; // [B, grid_h*grid_w, 768]
            let pixel_flat = pixel_values
                .reshape(&[b * grid_h * grid_w, 3 * patch * patch])
                .map_err(|e| RunError::Other(format!("pixel reshape: {e:?}")))?;

            // gen embedder + timestep / noise_scale embedding
            let mut image_embeds = model
                .extract_feature_gen(&pixel_flat, grid_h, grid_w)
                .map_err(|e| RunError::Other(format!("extract_feature_gen: {e:?}")))?;

            let t_vec_data = vec![t; b * l_tokens];
            let t_tensor = Tensor::from_vec(
                t_vec_data,
                Shape::from_dims(&[b * l_tokens]),
                device.clone(),
            )
            .map_err(|e| RunError::Other(format!("t_vec alloc: {e:?}")))?
            .to_dtype(DType::BF16)
            .map_err(|e| RunError::Other(format!("t_vec bf16: {e:?}")))?;
            let t_emb = model
                .time_or_scale_embed(&t_tensor, TimeOrScale::Timestep)
                .map_err(|e| RunError::Other(format!("time embed: {e:?}")))?
                .reshape(&[b, l_tokens, cfg.hidden_size])
                .map_err(|e| RunError::Other(format!("time embed reshape: {e:?}")))?;
            let mut additive = t_emb;
            if cfg.add_noise_scale_embedding {
                let s_tensor = Tensor::from_vec(
                    vec![s_norm; b * l_tokens],
                    Shape::from_dims(&[b * l_tokens]),
                    device.clone(),
                )
                .map_err(|e| RunError::Other(format!("s_vec alloc: {e:?}")))?
                .to_dtype(DType::BF16)
                .map_err(|e| RunError::Other(format!("s_vec bf16: {e:?}")))?;
                let s_emb = model
                    .time_or_scale_embed(&s_tensor, TimeOrScale::NoiseScale)
                    .map_err(|e| RunError::Other(format!("noise embed: {e:?}")))?
                    .reshape(&[b, l_tokens, cfg.hidden_size])
                    .map_err(|e| RunError::Other(format!("noise embed reshape: {e:?}")))?;
                additive = additive
                    .add(&s_emb)
                    .map_err(|e| RunError::Other(format!("additive add: {e:?}")))?;
            }
            image_embeds = image_embeds
                .add(&additive)
                .map_err(|e| RunError::Other(format!("image_embeds + additive: {e:?}")))?;

            // CFG cond + uncond passes through forward_gen.
            let h_cond = forward_gen_for(
                model,
                &image_embeds,
                cond_cache.next_t_index,
                token_h,
                token_w,
                &cond_cache,
            )
            .map_err(|e| RunError::Other(format!("forward_gen cond step {step}: {e}")))?;
            let h_uncond = forward_gen_for(
                model,
                &image_embeds,
                uncond_cache.next_t_index,
                token_h,
                token_w,
                &uncond_cache,
            )
            .map_err(|e| RunError::Other(format!("forward_gen uncond step {step}: {e}")))?;
            let x_cond = model
                .fm_head_forward(&h_cond)
                .map_err(|e| RunError::Other(format!("fm_head cond: {e:?}")))?;
            let x_uncond = model
                .fm_head_forward(&h_uncond)
                .map_err(|e| RunError::Other(format!("fm_head uncond: {e:?}")))?;
            let denom = (1.0 - t).max(cfg.t_eps);
            let inv_denom = 1.0 / denom;
            let v_cond = x_cond
                .sub(&z)
                .map_err(|e| RunError::Other(format!("v_cond sub: {e:?}")))?
                .mul_scalar(inv_denom)
                .map_err(|e| RunError::Other(format!("v_cond scale: {e:?}")))?;
            let v_uncond = x_uncond
                .sub(&z)
                .map_err(|e| RunError::Other(format!("v_uncond sub: {e:?}")))?
                .mul_scalar(inv_denom)
                .map_err(|e| RunError::Other(format!("v_uncond scale: {e:?}")))?;

            // CFG combine (cfg_norm='none' branch — matches the bin default).
            let v_diff = v_cond
                .sub(&v_uncond)
                .map_err(|e| RunError::Other(format!("cfg sub: {e:?}")))?;
            let v = v_uncond
                .add(
                    &v_diff
                        .mul_scalar(cfg_scale)
                        .map_err(|e| RunError::Other(format!("cfg scale: {e:?}")))?,
                )
                .map_err(|e| RunError::Other(format!("cfg combine: {e:?}")))?;

            // Euler step on z, then unpatchify back to image space.
            let z_next = z
                .add(
                    &v.mul_scalar(t_next - t)
                        .map_err(|e| RunError::Other(format!("euler mul: {e:?}")))?,
                )
                .map_err(|e| RunError::Other(format!("euler add: {e:?}")))?;
            unpatchify(&z_next, p_eff, height, width)?
        };
        img = next_img;

        let step1 = (step + 1) as u32;
        let elapsed = t_start.elapsed().as_secs_f32();
        let per_step = elapsed / step1 as f32;
        let eta_secs = ((steps - step1) as f32 * per_step).round().max(0.0) as u32;
        let _ = ev_tx.send(WorkerEvent::Progress {
            id: job.id,
            step: step1,
            total: steps,
            eta_secs,
        });
        ctx.request_repaint();
        log::debug!(
            "SenseNova-U1: step {:>3}/{} t={:.4}->{:.4} {:.2}s",
            step1,
            steps,
            t,
            t_next,
            step_t0.elapsed().as_secs_f32()
        );
    }

    // -------- 8. Denorm + convert to ColorImage --------
    let final_img = img
        .mul_scalar(0.5)
        .map_err(|e| RunError::Other(format!("denorm mul: {e:?}")))?
        .add_scalar(0.5)
        .map_err(|e| RunError::Other(format!("denorm add: {e:?}")))?;

    image_to_color_image(&final_img).map_err(|e| RunError::Other(format!("to ColorImage: {e}")))
}

/// Thin `forward_gen` wrapper because the public signature takes
/// `attn_mask: Option<&Tensor>` and we always pass `None` (no padding in our
/// prefix; full attention with the implicit causal-cross-prefix from the
/// cached prefix tokens). Mirrors `forward_gen_for` in the CLI bin.
fn forward_gen_for(
    model: &mut SenseNovaU1,
    image_embeds: &Tensor,
    text_len: usize,
    token_h: usize,
    token_w: usize,
    cache: &KvCache,
) -> Result<Tensor, String> {
    model
        .forward_gen(image_embeds, text_len, token_h, token_w, cache, None)
        .map_err(|e| format!("forward_gen: {e:?}"))
}

// ===========================================================================
// Helpers — mirror the CLI bin verbatim. Citations to the bin in each fn.
// ===========================================================================

/// Build a Qwen3-style ByteLevel-BPE tokenizer from `vocab.json` + `merges.txt`,
/// then add the 293 special tokens from `added_tokens.json` in ID order so the
/// auto-assigned IDs match the reference (151643..151935, contiguous).
///
/// Mirrors `bin/sensenova_u1_gen.rs::build_tokenizer`.
fn build_tokenizer(weights_dir: &Path) -> Result<Tokenizer, String> {
    let vocab = weights_dir.join("vocab.json");
    let merges = weights_dir.join("merges.txt");
    let added = weights_dir.join("added_tokens.json");

    let bpe = BPE::from_file(
        vocab.to_str().ok_or("vocab path not utf-8")?,
        merges.to_str().ok_or("merges path not utf-8")?,
    )
    .build()
    .map_err(|e| format!("BPE::build failed: {e}"))?;

    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevel::default()));

    let raw = std::fs::read_to_string(&added)
        .map_err(|e| format!("read {}: {e}", added.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).map_err(|e| format!("added_tokens.json: {e}"))?;
    let mut entries: Vec<(String, u64)> = map
        .into_iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    entries.sort_by_key(|(_, id)| *id);

    let base_size = tok.get_vocab_size(false) as u64;
    if let Some((_, first_id)) = entries.first() {
        if *first_id != base_size {
            return Err(format!(
                "added_tokens.json starts at id {first_id} but base vocab size is {base_size}; \
                 cannot align IDs via append-only AddedToken API"
            ));
        }
    }

    let added_tokens: Vec<AddedToken> = entries
        .into_iter()
        .map(|(content, _)| AddedToken::from(content, true))
        .collect();
    tok.add_special_tokens(&added_tokens);

    let im_start = tok
        .token_to_id("<|im_start|>")
        .ok_or("<|im_start|> not in tokenizer")?;
    if im_start != 151644 {
        return Err(format!(
            "<|im_start|> mapped to {im_start}, expected 151644"
        ));
    }
    Ok(tok)
}

/// Build the T2I chat-template prompt. Mirrors `_build_t2i_query` in the
/// reference python (`modeling_neo_chat.py:431`) and `bin/sensenova_u1_gen.rs`.
fn build_t2i_query(system: &str, user: &str, append: &str) -> String {
    let mut q = String::new();
    if !system.is_empty() {
        q.push_str("<|im_start|>system\n");
        q.push_str(system);
        q.push_str("<|im_end|>\n");
    }
    q.push_str("<|im_start|>user\n");
    q.push_str(user);
    q.push_str("<|im_end|>\n");
    q.push_str("<|im_start|>assistant\n");
    q.push_str(append);
    q
}

fn encode_query(tok: &Tokenizer, query: &str) -> Result<Vec<i32>, String> {
    let enc = tok
        .encode(query, false)
        .map_err(|e| format!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

/// Box-Muller seeded Gaussian noise → BF16 Tensor of shape `[B, 3, H, W]`.
/// Mirrors `bin/sensenova_u1_gen.rs::make_noise_image`.
fn make_noise_image(
    seed: u64,
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> Result<Tensor, String> {
    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let t = Tensor::from_vec(data, Shape::from_dims(shape), device.clone())
        .map_err(|e| format!("noise from_vec: {e:?}"))?;
    t.to_dtype(DType::BF16)
        .map_err(|e| format!("noise bf16: {e:?}"))
}

/// `patchify(images=[B, 3, H, W], p, channel_first)` → `[B, h*w, p*p*3]`.
/// Mirrors `bin/sensenova_u1_gen.rs::patchify`. See that file for the full
/// einsum reference.
fn patchify(images: &Tensor, p: usize, channel_first: bool) -> Result<Tensor, RunError> {
    let dims = images.shape().dims().to_vec();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(RunError::Other(format!(
            "patchify expects [B, 3, H, W], got {dims:?}"
        )));
    }
    let (b, h, w) = (dims[0], dims[2], dims[3]);
    if h % p != 0 || w % p != 0 {
        return Err(RunError::Other(format!(
            "patchify: H={h} W={w} not divisible by p={p}"
        )));
    }
    let gh = h / p;
    let gw = w / p;
    let x = images
        .reshape(&[b, 3, gh, p, gw, p])
        .map_err(|e| RunError::Other(format!("patchify reshape: {e:?}")))?;
    let x = if channel_first {
        x.permute(&[0, 2, 4, 1, 3, 5])
    } else {
        x.permute(&[0, 2, 4, 3, 5, 1])
    }
    .map_err(|e| RunError::Other(format!("patchify permute: {e:?}")))?;
    x.reshape(&[b, gh * gw, p * p * 3])
        .map_err(|e| RunError::Other(format!("patchify final reshape: {e:?}")))
}

/// `unpatchify(x=[B, L, p*p*3], p, h, w)` → `[B, 3, h, w]`. The inner axis
/// flatten order is (kH, kW, C) (channel_first=false during patchify). Mirrors
/// `bin/sensenova_u1_gen.rs::unpatchify`.
fn unpatchify(x: &Tensor, p: usize, h: usize, w: usize) -> Result<Tensor, RunError> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(RunError::Other(format!(
            "unpatchify expects [B, L, D], got {dims:?}"
        )));
    }
    let b = dims[0];
    let gh = h / p;
    let gw = w / p;
    if gh * gw != dims[1] {
        return Err(RunError::Other(format!(
            "unpatchify: L={} != gh*gw={}*{}",
            dims[1], gh, gw
        )));
    }
    let x = x
        .reshape(&[b, gh, gw, p, p, 3])
        .map_err(|e| RunError::Other(format!("unpatchify reshape: {e:?}")))?;
    let x = x
        .permute(&[0, 5, 1, 3, 2, 4])
        .map_err(|e| RunError::Other(format!("unpatchify permute: {e:?}")))?;
    x.reshape(&[b, 3, gh * p, gw * p])
        .map_err(|e| RunError::Other(format!("unpatchify final reshape: {e:?}")))
}

// ===========================================================================
// Cancel + image-conversion helpers
// ===========================================================================

fn drain_pending(
    ui_rx: &Receiver<UiMsg>,
    pending: &mut VecDeque<UiMsg>,
) -> Result<(), RunError> {
    while let Ok(msg) = ui_rx.try_recv() {
        match msg {
            UiMsg::Cancel => return Err(RunError::Cancelled),
            UiMsg::Shutdown => {
                pending.push_back(UiMsg::Shutdown);
                return Err(RunError::Cancelled);
            }
            other => pending.push_back(other),
        }
    }
    Ok(())
}

/// Convert SenseNova's final `[1, 3, H, W]` BF16 image (already denormed into
/// [0, 1]) into an `egui::ColorImage` for display.
fn image_to_color_image(img: &Tensor) -> Result<ColorImage, String> {
    let dims = img.shape().dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(format!(
            "image_to_color_image expects [1, 3, H, W], got {dims:?}"
        ));
    }
    let h = dims[2];
    let w = dims[3];
    let f32_t = img
        .to_dtype(DType::F32)
        .map_err(|e| format!("img f32: {e:?}"))?;
    let host = f32_t.to_vec().map_err(|e| format!("img to_vec: {e:?}"))?;

    let plane = h * w;
    let mut pixels = Vec::with_capacity(plane);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = (host[i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (host[plane + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (host[2 * plane + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    Ok(ColorImage {
        size: [w, h],
        pixels,
    })
}
