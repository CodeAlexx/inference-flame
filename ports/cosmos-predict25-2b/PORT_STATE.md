# PORT_STATE: cosmos-predict25-2b

Last updated: 2026-05-21 by port-bugfix (chunk 1 skeptic-blockers resolved)

## Phase
build-complete (steps 1-11 done; step 12 smoke is GPU-gated)

## Status
PORT_SPEC.md and BUILD_PLAN.md written. Op map confirms **no flame-core
changes blocking** — all primitives exist (`fused_rms_norm`,
`fused_linear3d_native`, `rope_halfsplit_bf16`, `sdpa`, Wan21 VAE).
RoPE strategy validated: head_dim=128 → axis split (t=44, h=42, w=42)
identical to Wan numbers; Cosmos uses GPT-NeoX half-split rotation (TE
`apply_rotary_pos_emb(..., fused=True)`), matches our `rope_halfsplit_bf16`.

12-step build order written. Memory plan: 480p plain, 720p with
BlockOffloader. Text encoder Phase A (cached embeddings) ships in this
port; live Cosmos-Reason1-7B encode deferred to port-extension.

GPU smoke (step 12) deferred until HiDream trainer releases the device.
Steps 1-11 are code-only and can proceed now.

## Checklist
- [x] Intake (PORT_SPEC.md)
- [x] Plan (BUILD_PLAN.md)
- [x] Build step 1 — Skeleton + state-dict tooling
- [x] Build step 2 — Embeddings (Timesteps, TimestepEmbedding, LearnablePosEmbAxis)
- [x] Build step 3 — 3-axis RoPE freq builder (8/8 unit tests pass)
- [x] Build step 4 — Attention (self + I2V cross dual-K/V, latter gated for fine-tunes)
- [x] Build step 5 — GPT2FeedForward FFN (tanh-approx GELU; exact-erf parity ceiling ~0.02%)
- [x] Build step 6 — transformer_block (modulate-attn-cross-ffn, extra_per_block_pos_emb inside block per Python :1267-1268)
- [x] Build step 7 — Full MiniTrainDIT.forward (patchify, padding-mask cat+contig, RoPE+pos-emb once, 28-block loop with debug magnitude probe, FinalLayer 2-chunk, unpatchify)
- [x] Build step 8 — Rectified-flow sampler primitives + **full FlowUniPCMultistepScheduler bh2 multistep** (cosmos_unipc.rs, parity bit-exact vs Python at step 0; multistep corrector-path parity DEFERRED until pre-parity cleanup)
- [ ] Build step 7 — Full MiniTrainDIT forward
- [ ] Build step 8 — Rectified-flow sampler
- [ ] Build step 9 — Cosmos-Reason1-7B encoder wiring
- [ ] Build step 10 — Wan VAE wiring + key remap
- [ ] Build step 11a/b/c — three inference binaries
- [ ] Skeptic (after each chunk)
- [ ] Bugfix
- [ ] Parity
- [ ] Smoke (GPU-gated — wait for HiDream trainer)

## Chunk 1 deliverables
- `inference-flame/src/models/cosmos_predict25_dit.rs` (862 LOC) — config + skeleton + Timesteps + TimestepEmbedding + LearnablePosEmbAxis + build_cosmos_rope_freqs + 8 unit tests
- `parity/convert_dit_pt_to_safetensors.py` — `.pt` → `.safetensors` w/ EMA prefix stripping
- `parity/dump_state_dict_keys.py` — key/shape/dtype dumper
- mod.rs export added

## Chunk 1 builder findings (spec corrections folded into BUILD_PLAN.md)
1. Weight prefix for V2_2B learnable axis pos emb is `extra_pos_embedder.*`, NOT `pos_embedder.*` (which is the rope3d module's computed buffers). When `extra_per_block_abs_pos_emb=True` two modules coexist.
2. `t_embedder.1.linear_{1,2}.bias` is ABSENT when `use_adaln_lora=True` (PyTorch `bias=not use_adaln_lora`). V2_2B has use_adaln_lora=True → no biases.
3. Cosmos `Timesteps` returns cos-first sin-second, NOT the sin-first convention seen elsewhere. Documented in code comment.
4. `rope_t_extrapolation_ratio` defaults to 1.0 for V2_2B (inherited from `MiniTrainDIT.__init__`); 14B and 7B _MININET variants override differently.

## Anima discovery (2026-05-21, mid-build)
`inference-flame/src/models/anima.rs` (1252 LOC) is a complete inference port
of Cosmos Predict2 (image-only, T=1). Same MiniTrainDIT defaults: 28 blocks,
2048 hidden, 16 heads, head_dim=128, MLP 8192, AdaLN-LoRA(256→6144), QK
RMSNorm(128), patchify(2,2,1), in_ch=17 (16 latent + padding mask),
crossattn ctx 1024-dim, 3D RoPE with cossin builder. Forward already takes
[B, T, H, W, C] — supports video shape; the anima_infer bin just runs T=1.

**User direction**: cosmos stays an independent file. Copy patterns from
anima.rs as a read-only template for chunks 2+; do not import, refactor,
or extend anima. Cosmos differences to handle:
- No internal LLM Adapter (cosmos uses external Cosmos-Reason1-7B → 1024d)
- I2VCrossAttention dual-K/V branch (anima cross-attn is single K/V)
- FPS modulation in 3D RoPE (already shipped in chunk 1's `build_cosmos_rope_freqs`)
- `extra_per_block_abs_pos_emb=True` (LearnablePosEmbAxis at every block, not just input — verify anima's actual behavior)
- Video T>1 with multi-frame conditioning (i2v, v2v)
- Wan 2.1 VAE (not Qwen Image VAE)
- Rectified-flow scheduler from Cosmos source (anima may use a different sigma schedule)

See memory: [[project_cosmos_predict25_independent_port]].

## Chunk 1 skeptic-bait (for port-skeptic agent)
1. `extra_pos_embedder` vs `pos_embedder` prefix choice gated on `extra_per_block_abs_pos_emb` — confirm with `dump_state_dict_keys.py` against a real checkpoint.
2. `sinusoidal_timesteps` per-element exponent calculation — bit-equivalent to Python's vectorised form?
3. RoPE row-write order `[t, h, w]` — easy to flip; cos≈0.99 silent-failure trap.
4. NTK ratio `dim_axis as f64 / (dim_axis as f64 - 2.0)` — must stay float division (matches Py3); if integer-fixed it drifts.
5. `convert_dit_pt_to_safetensors.py` checkpoint-shape heuristic — prefers `"ema"` over `"model"`; falls back via loop; will abort on truly exotic layouts.

## Skeptic chunk-1 triage (2026-05-21)
See `SKEPTIC_FINDINGS_2026-05-21.md` for full evidence. Items 1-4 of builder skeptic-bait above were REFUTED with concrete evidence; item 5 was CONFIRMED. Skeptic found additional issues:

| # | Finding | Severity | Fix? |
|---|---------|----------|------|
| F1 | `learnable_pos_emb` missing L2-norm output scaling (Python `:850-852`) | BLOCKER | yes |
| F2 | Missing `t_embedding_norm` RMSNorm post-timestep (Python `:1754`; anima `:280`) | BLOCKER | yes |
| F5 | Convert script doesn't strip `net.` prefix (anima checkpoints use it) | BLOCKER | yes |
| F3 | `axis_lens` returns pre-patch dims (240) not post-patch (120) | FRAGILE | yes |
| F9 | `enable_fps_modulation=false + t>1 + fps=None` rejected, should compute integer positions | FRAGILE | yes |
| F10 | `Timesteps` API takes 1D, returns rank-2; Python takes 2D `[B,T]` returns rank-3 `[B,T,D]` | FRAGILE | yes |
| F4 | RoPE tests are tautological (exercise `build_cpu_layout`, not the real function) | FRAGILE | defer to parity phase |
| F6 | head_dim validation has confused boolean precedence (V2_2B passes by accident) | STYLE | yes (1-line) |
| F7 | `broadcast_to + add` non-contig chain unverified | FRAGILE | defer (GPU smoke will catch) |
| F8 | `unwrap_or(v)` swallows dtype cast failures | STYLE | yes (1-line) |
| F11 | `axis_lens` is dead code | STYLE | merge with F3 fix |

Bugfix touchpoints: cosmos_predict25_dit.rs (F1, F2, F3, F6, F8, F9, F10, F11), convert_dit_pt_to_safetensors.py (F5).

## Smoke result 2026-05-21 (overnight, GPU-free)
End-to-end pipeline runs but **output is colored noise** at both 1-step and 35-step. Per-component diagnostics:

| Component | Diagnostic | Status |
|---|---|---|
| Text encoder | output mean=0, rms=1, |e|=50 | ✅ correct |
| Cross-attn | |cond-uncond|=0.39 (20% of |cond_v|) | ✅ wired |
| DiT velocity | |cond_v|=1.5-2.5, |uncond_v|=1.1-2.0 | ✅ reasonable |
| x0_pred = x - σ*v | rms=1.1-1.6 throughout denoise | ✅ Wan-VAE-sane |
| Trajectory | x grows from 4.6 → 6 (Euler+cfg=1), 4.6 → 12 (UniPC+cfg=7) | ❌ wrong direction |
| VAE decode | pixels [-1, 1], rms=0.39 | ✅ in range |
| Visual output | colored noise (cat ≈ forest pixel stats) | ❌ wrong |

**Smoking gun**: x0_pred is sane but x trajectory diverges. Model predicts reasonable clean targets per-step but integration doesn't converge. Subtle layer-level numerical drift, requires per-layer Python parity capture to localize.

## Fixes applied during smoke
- F32 residual stream in `transformer_block` (anima oracle pattern) — values still hit ~33k pre-FinalLayer; FinalLayer's LN brings them back to O(1) velocity output, so this didn't fix the bug but eliminated a potential BF16-overflow risk.
- F6 head_dim validation: was `head_dim % 6 != 0` (too strict — rejects 128); fixed to `even && >= 6`.
- Aggressive `.contiguous()` after LVG mask cat + parent padding-mask cat — did not change output.

## Known correct (component-level)
- Production checkpoint loads cleanly: 569/689 tensors (rest are TE `_extra_state` + `accum_*` training scalars + rope3d computed buffers).
- Path defaults: `DEFAULT_COSMOS_DIR = /home/alex/.cosmos-predict25` (matches downloaded staging).
- All three weights staged: DiT 4.1 GB, Cosmos-Reason1-7B 16 GB, Wan VAE 243 MB.
- Tokenizer: `tokenizer.json` from Cosmos-Reason1-7B with qwen chat template applied in Rust.

## Next action
1. Generate Python parity capture: `inference-flame/ports/cosmos-predict25-2b/parity/cosmos_predict25_per_layer_capture.py`. Run Cosmos via diffusers/upstream-repo with the SAME prompt + seed, capture intermediate activations (post-x_embedder, post-block-0, post-block-13, post-block-27, post-FinalLayer, pre-VAE). Save to safetensors.
2. Add a Rust parity binary that runs the same prompt + seed, captures intermediates at the same points, and compares cos-distance.
3. Bisect: find the FIRST layer where cos-distance drops below 0.999. That's the bug.

Most likely culprits (in order):
- (a) RoPE position math — head_dim split correct (44,42,42) but rotation order/sign might be wrong vs TE's `apply_rotary_pos_emb(fused=True)`.
- (b) Patchify/unpatchify einops dim order — verified vs Python source but subtle stride bugs possible.
- (c) adaLN-LoRA modulation summation — chunk-2/3 builders verified math vs Python, but the production checkpoint has 3 separate adaLN per block (not shared 9-chunk) which chunk-6 confirmed already-correct in chunk-2 code. Could still be wrong on chunk order (shift, scale, gate vs scale, shift, gate).
- (d) Cross-attn dual-K/V branch reused `q` after RMSNorm — chunk-2 skeptic verified correct, but verify k_proj/v_proj input dim is 1024 (post-crossattn_proj) not 100352.

Step 12 (smoke) marked **partial**: pipeline runs end-to-end, all components numerically sane, but the integrated denoise produces wrong outputs. Full convergence requires layer-level parity work (~1 day).

When HiDream trainer releases GPU + weights are staged:
1. Run `parity/convert_dit_pt_to_safetensors.py`, `parity/convert_wan21_vae_pth_to_safetensors.py`, `parity/cosmos_reason1_encode_ref.py`, `parity/wan21_vae_encode_decode_ref.py`
2. Run cosmos parity tests with weight paths set
3. Address chunk-2 pre-parity cleanup items (task #16)
4. Wire BlockOffloader into DiT for 720p path
5. Add multistep UniPC parity fixture (corrector path)
6. Run `/port-smoke` per the skill — bisected (1 step → 5 → 50 → full) at 480p first

## Chunk 5 deliverables
- `inference-flame/src/bin/cosmos_predict25_common.rs` — ~1000 LOC shared module (CLI parsing, 5-stage orchestrator)
- `inference-flame/src/bin/cosmos_predict25_{t2v,i2v,v2v}_infer.rs` — thin wrappers (Mode-dispatching)
- 3 `[[bin]]` entries added to `inference-flame/Cargo.toml`
- All binaries compile clean; `--help` works; bad-path test exits cleanly

## Open known-issues post-chunk-5 (address before /port-smoke)
- **BlockOffloader unwired in DiT** — 720p will OOM on 24 GB until wired. Currently warns and falls through. 480p path is fine.
- **Cosmos uses non-canonical CFG**: `cond + g*(cond-uncond)` not the usual `uncond + g*(cond-uncond)`. Functionally equivalent but bias-shifted; document for users.
- **VAE input range**: built `[-1, 1]` per Wan21Vae docstring; verify Python pipeline normalization matches.
- **timesteps c_noise scaling**: rectified-flow passes `sigma * 1000` raw; EDM rescale is skipped. Confirm at parity time.
- **fps default 16**: matches `Fps-16` post-trained checkpoint name; if user picks fps=24 the RoPE positions extrapolate.
- **Python default num_frames is 77, not 81** — both satisfy `(N-1) % 4 == 0`; we use 81. Trivial to change if needed.
- **UniPC corrector parity untested** — only step-0 fixture exists; multistep fixture needed before smoke quality claim.

## Chunk 4 spec corrections (load-bearing — must be propagated to chunks 5+)

**The biggest finding of the port so far.** The chunk-1/2/3 PORT_SPEC and BUILD_PLAN had the text encoder strategy wrong:

1. **Cosmos-Reason1-7B emits 100352-dim embeddings**, not last-hidden-state 3584-dim. The strategy is `EmbeddingConcatStrategy::FullConcat` — 28 layers × 3584 hidden_size = **100352**, with each per-layer output **mean-normalized first**. Evidence: `text_encoder.py` + `EmbeddingConcatStrategy.FULL_CONCAT`.

2. **The DiT has a `crossattn_proj = Linear(100352, 1024, bias=True) + GELU(exact-erf)`** that runs BEFORE the block loop. Python `minimal_v4_dit.py:1565-1569` declares it; `:1738-1739` applies it.

3. **`use_crossattn_projection=True` is a production override**, NOT base. The base `COSMOS_V2_2B_NET` doesn't enable it; the override at `configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:146-149` does. The shipped checkpoint expects the override.

4. **Use `CosmosPredict25Config::cosmos_v2_2b_production()` preset** for production wiring (sets `use_crossattn_projection=true, crossattn_proj_in_channels=100352, crossattn_emb_channels=1024`).

5. **The `crossattn_proj` GELU is exact-erf** (bare `nn.GELU()` at Python `:1568`), same as MLP. Agent used `Tensor::gelu_exact()`. Correct.

## Chunk 4 deliverables
- `inference-flame/src/models/cosmos_reason1.rs` — 440 LOC wrapper over Qwen25VLEncoder; chat template + tokenize+pad + per-layer mean-normalize + `EmbeddingConcatStrategy::{FullConcat, MeanPooling}`. 8 tests pass (6 cpu + 2 GPU-gated stubs).
- `cosmos_predict25_dit.rs` — added `use_crossattn_projection`, `crossattn_proj_in_channels`, `apply_crossattn_proj()` method, `cosmos_v2_2b_production()` preset, 2 new tests. 24/24 cosmos tests pass.
- `parity/dump_cosmos_reason1_keys.py`, `parity/cosmos_reason1_encode_ref.py`, `parity/convert_wan21_vae_pth_to_safetensors.py`, `parity/wan21_vae_encode_decode_ref.py` — all GPU-streamed, gated on HF auth + GPU-free.

## To enable GPU parity (deferred — needs weights + GPU)
1. `huggingface-cli login` for gated repos.
2. Download Cosmos-Reason1-7B (15 GB).
3. Download Cosmos-Predict2.5-2B/tokenizer.pth (508 MB) + base/post-trained/*.pt (4.1 GB).
4. Run the parity scripts.
5. Set `COSMOS_REASON1_PATH` + `WAN21_VAE_COSMOS_SAFETENSORS` env vars and re-run cosmos_predict25 tests.

## Open known-issues post-chunk-3 (defer to pre-parity cleanup)
- UniPC multistep parity fixture only covers step 0 — the bh2 corrector
  path (active from step 1 onward) is not yet parity-tested. Add a
  multistep fixture (3-5 steps) before smoke.
- Anima BF16↔F32 sub-block divergence still unobserved at tiny scale
  (L∞=2.56 after 2 blocks). Check at 28-block real scale during parity.
- F1 RoPE 3-axis fused kernel still a "nice-to-have" optimization
  (BUILD_PLAN R8) — not blocking; cosmos builds cos/sin on CPU once
  per shape.

## Chunk 3 spec corrections (folded into BUILD_PLAN.md)
1. Padding mask shape is `[B, 1, H, W]` (broadcast to T inside model), NOT `[B, 1, T, H, W]`. Python `video2world.py:431` passes zeros; model unsqueezes+repeats across T.
2. **Patchify vs unpatchify are asymmetric** in Python source. Patchify einops: `b c (t r) (h m) (w n) -> b t h w (c r m n)`. Unpatchify: `b t h w (p1 p2 t' c) -> b c (t t') (h p1) (w p2)`. The FinalLayer's output Linear was trained to bridge the two. **Patchify→Unpatchify is NOT bit-identical** without the model in between; round-trip test passes only on shape.
3. Rectified-flow `shift=5.0` default for V2_2B (Python `text2world_model_rectified_flow.py:502`). CFG default 7.0.
4. **UniPC multistep corrector is what Python actually uses** for inference (`FlowUniPCMultistepScheduler`), not pure Euler. Chunk-3 shipped FlowMatch + Euler + CFG only. UniPC deferred — may cause visible quality gap at smoke.
5. FinalLayer adaLN-LoRA uses 2-chunk (shift, scale, NO gate) and the LoRA features are reused — only `adaln_lora[:, :, :2*hidden]` slice. Python `:1097-1126`.
6. Magnitude probe at L∞=2.56 after 2 tiny blocks; well below anima's 200+ threshold — chunk-2 skeptic's BF16/F32 divergence concern not yet observed at tiny scale. Re-check at 28-block full scale during parity.

## Chunk 3 skeptic-bait (for next /port-skeptic)
1. Patchify permute `[0,2,4,6,1,3,5,7]` (B, T_p, H_p, W_p, C, r, m, n) — stride-layout assumptions in flame-core's reshape may produce subtly wrong layouts. Parity will catch.
2. Unpatchify permute `[0,7,1,6,2,4,3,5]` — compounded by patchify/unpatchify Python asymmetry.
3. Padding-mask cat: `mask_5d` is stride-0 broadcast view; `Tensor::cat` may mis-read the stride-0 input even with `.contiguous()` after. Test: pass non-zero padding mask, verify it propagates to all T positions.
4. `default_owned` lifetime — Rust binding inside if-block with ref in match arm. Future edit risk.
5. `Tensor::max_all` routes through GpuOps F32 path (legacy). Debug-only, but flagged.
6. Sigma `linspace` endpoint — Python `numpy.linspace(sigma_max, sigma_min, n+1)[:-1]` gives exactly n values; f64 reimplementation uses `denom=n` over `n+1` points. Should match numpy but worth verifying.

## Chunk 2 spec corrections discovered during build (folded into BUILD_PLAN.md)
1. **GELU is exact-erf** (Python `nn.GELU()` line 240 — bare, no `approximate=`). flame-core only has tanh-approx. Per-block ~0.02% magnitude ceiling unless we add exact-erf to flame-core. Documented in `mlp` docstring.
2. **V2_2B base has NO I2V dual-K/V cross-attn.** `extra_image_context_dim=None` in config. Image conditioning flows via VAE-encoded conditional latent frames + padding-mask channel, NOT cross-attn dual K/V. The dual-K/V code path is implemented and gated on `config.extra_image_context_dim.is_some()` for future fine-tuned variants.
3. **`extra_per_block_pos_emb` is added inside `Block.forward`** (Python `:1267-1268`), not in `MiniTrainDIT.forward`. Chunk 7's full-forward needs to compute it once via `learnable_pos_emb` and pass the SAME tensor as `extra_per_block_pos_emb=Some(&t)` to every block call.
4. **Weight key names confirmed**: `q_proj`, `k_proj`, `v_proj`, `output_proj`, `layer1`, `layer2` (Python sources lines 453-462, 241-242). Earlier BUILD_PLAN mapping table had inconsistent placeholders.
5. **I2VCrossAttention.k_img input dim is 2048 not 1024**. The image context is pre-projected from `extra_image_context_dim → model_channels=2048` by `img_context_proj` Sequential outside the block (Python `:1558-1567`).
6. **adaLN chunk order is `(shift, scale, gate)` × 3 sub-blocks**, and `adaln_lora` is added to all three sub-block modulations identically (same tensor shared across self/cross/ffn). Python `:1272-1274`.

## Chunk 2 skeptic-clean (2026-05-21)
See `SKEPTIC_FINDINGS_2026-05-21_chunk2.md`. **0 BLOCKERs**, 3 FRAGILE, 5 STYLE, 1 FLAME-CORE, 4 DISAGREE (all builder skeptic-bait items refuted as not-bugs). Deferred items, to revisit before `/port-parity`:

| # | Issue | Severity | Defer-until |
|---|-------|----------|-------------|
| ~~F1~~ | ~~GELU tanh-approx vs Python exact-erf~~ | RESOLVED 2026-05-21 | flame-core `Tensor::gelu_exact()` added; bit-exact vs PyTorch CUDA fixture; cosmos `mlp` switched to `gelu_exact()` |
| F2 | `apply_gate` rank-check missing | STYLE | cleanup pass |
| F3 | `apply_gate` broadcast+mul stride-0 path unverified (chunk-1 F7 hazard) | FRAGILE | parity will catch |
| F6 | tests use `head_dim=6` corner case | STYLE | parity at head_dim=128 catches |
| F7 | I2V test threshold 1e-4 too loose | FRAGILE | tighten before parity |
| F8 | modulation test pins lora=0; lora-add line not independently load-bearing | FRAGILE | tighten before parity |
| F9 | brittle `max_diff == 0.0` for repeat GEMMs | STYLE | cleanup pass |
| F11 | `extra_per_block_pos_emb` add lacks shape guard | STYLE | cleanup pass |
| F12 | adaln_modulation_chunk creates 3 narrow-views + 3 materialize copies per block | STYLE | perf pass |

**Anima oracle BF16/F32 callout**: Anima explicitly BF16↔F32 casts between sub-blocks because hidden values reach 200+. Cosmos chunk 2 stays BF16 through the residual stream. May surface as parity drift at chunk 7. Recommended: add magnitude probe (`assert |x| < 200`) after each block during smoke. **Watch when chunk 7 ships.**

## Chunk 2 skeptic-bait (REFUTED by chunk-2 skeptic)
1. `linear_no_bias` always reshapes to `[1, n, cin]` (B=1 collapse). Math correct but if `fused_linear3d_native` has any batch-dependent behavior beyond `Cout/Cin`, result could differ from anima's `matmul(x_2d, w.T)` pattern.
2. Cross-attn does `rms_norm_per_head_bnhd → permute` while self-attn does `permute → rms_norm_per_head_bhnd`. Both should be equivalent (last-dim norm) but stride-sensitive kernels may behave differently.
3. `apply_gate` via `broadcast_to + mul` produces stride-0 view on H,W. F7 chunk-1 hazard still unverified.
4. `modulate_pre_fused_bf16` batch indexing — passed `[B*T, H*W, D]` for x and `[B*T, D]` for shift/scale assuming the kernel reads `batch_idx = row / seq_len` with `seq_len = H*W`. Verify against `bf16_ops.rs:1383`.
5. `unflatten_thw` after attention assumes token row-order `(t, h, w)` row-major matches `build_cosmos_rope_freqs` output. Any silent permute breaks cos/sin alignment.
6. `output_proj` weight key in cross-attn — assumed `cross_attn.output_proj.weight` (since `I2VCrossAttention extends Attention`). HF checkpoint variants may differ.

## Major reuse discovered during plan
- `inference-flame/src/models/qwen25vl_encoder.rs` (749 LOC) — already the
  exact text encoder Cosmos-Reason1-7B needs (Qwen2.5-VL-7B-Instruct, same
  architecture). Originally for Kandinsky-5.
- `inference-flame/src/vae/wan21_vae.rs` + `wan21_encoder.rs` — Wan 2.1 VAE
  is what Cosmos calls "tokenizer", already shipped (wan22 port).
- `Wan22Dit::sinusoidal_embedding` (`wan22_dit.rs:334`) — timestep pattern.
- `inference_flame::mux::write_mp4_video_only` — mp4 output.
- No flame-core changes blocking.

## Open issues
1. VRAM budget on 24 GB vs upstream 32.54 GB — decide in plan phase.
2. Cosmos-Reason1-7B text encoder: live forward (15 GB BF16) vs frozen
   embedding cache — affects offload sequencing.
3. `.pt` → `.safetensors` conversion script needed at build kickoff.
4. Variant choice: `base/pre-trained` vs `base/post-trained` vs `base/distilled`.
5. `tokenizer.pth` (Wan2.1 VAE) key naming — verify matches our existing
   `Wan21VaeDecoder` / `Wan21VaeEncoder` key expectations.
6. Does `extra_image_context_dim` get set in COSMOS_V2_2B_NET? Determines
   whether I2VCrossAttention's image-K/V branch is active.

## Recent handoffs
- None yet (port just opened).
