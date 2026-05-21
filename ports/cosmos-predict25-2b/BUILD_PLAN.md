# BUILD_PLAN: cosmos-predict25-2b

Last updated: 2026-05-20 by port-plan

## Op map

| Model op | flame-core / inference-flame target | autograd? | parity-risk | notes |
|----------|--------------------------------------|-----------|-------------|-------|
| `RMSNorm` (eps=1e-6, gain only) | `flame_core::ops::fused_inference::fused_rms_norm` | N/A (inf-only) | low | reuse Wan/Klein pattern |
| `LayerNorm` (no affine, eps=1e-6) | `flame_core::layer_norm::layer_norm` with `weight=ones, bias=zeros`; or `cuda_ops_bf16::layer_norm_bf16` | N/A | low | only at FinalLayer; pre-build the unit gamma/beta once |
| Q/K head-dim RMSNorm | `fused_rms_norm` over last dim with shape `[..., head_dim]` | N/A | low | std pattern |
| `Linear(bias=False)` | `ops::fused_inference::fused_linear3d_native` (cuBLASLt BF16) | N/A | low | weight is `[Cout, Cin]` row-major, matches PyTorch native; this is what every FLUX/Chroma/Klein block uses |
| `SiLU` | `Tensor::silu` | N/A | low | used in TimestepEmbedding and adaLN-LoRA |
| `GELU` (GPT2 FFN) | `Tensor::gelu` (`tanh` approximation if GPT2-style) | N/A | low | confirm exact GELU variant during build (`GPT2FeedForward` source) |
| `Sinusoidal Timesteps` | reuse `Wan22Dit::sinusoidal_embedding` pattern at `wan22_dit.rs:334` | N/A | low | Cosmos uses standard sin/cos timestep (`Timesteps` class), same formula |
| `FourierFeatures` | small new module: `cos(2π·band·x)` + `sin(2π·band·x)` per channel | N/A | low | check call sites — may be unused for the 2B variant |
| `PatchEmbed` (Linear over `[C * P_s² * P_t]` flattened patch) | `Tensor::unfold` + 2D unfold + `fused_linear3d_native` | N/A | medium | spatial=2, temporal=1; mirror Klein-style patchify |
| `LearnablePosEmbAxis` (per-axis additive abs pos emb T/H/W) | new module — `pos_emb_t[t] + pos_emb_h[h] + pos_emb_w[w]` broadcast-add per token; **added at every block** because `extra_per_block_abs_pos_emb=True` | N/A | medium | learnable buffers, simple param load + broadcast |
| `VideoRopePosition3DEmb` (3-axis RoPE freq table) | new freq builder; head_dim=128 → axis_split=(t=44, h=42, w=42), each half-rotated | N/A | high | see "RoPE strategy" below |
| 3D RoPE apply (half-split / GPT-NeoX layout via TE `fused=True`) | `flame_core::bf16_ops::rope_halfsplit_bf16` | N/A | high | the freq tensor is concatenated `[t,h,w] * 2` per Cosmos source → exact half-split format flame-core expects |
| `SDPA` (self-attn, mask-less) | `flame_core::attention::sdpa(q, k, v, None)` → cuDNN BF16 forward | N/A | low | head_dim=128, supported |
| `SDPA` (cross-attn text K/V) | `sdpa(q, k_text, v_text, None)` | N/A | low | first of two cross-attn calls |
| `SDPA` (cross-attn image K/V) | `sdpa(q, k_img, v_img, None)` | N/A | low | second call; **outputs SUMMED** then projected (`result + result_img`) — not concat |
| `adaLN-LoRA` (Linear→SiLU→Linear → 3 chunks of `hidden`) | composition: `Linear(emb, hidden_pre)` → reused; `Linear(hidden_pre, 256, bias=False)` → `SiLU` → `Linear(256, 3*hidden, bias=False)` → split | N/A | medium | rank=256, applied once per block on the conditioning embedding `emb_B_T_D` |
| `(1+scale)*x + shift` modulate | `ops::fused_inference::fused_rms_norm_modulate` where applicable; otherwise hand-written `x * (1+scale) + shift` | N/A | low | flame-core has the fused version for the common case |
| Gate residual `x = x + gate * sub(x)` | `Tensor::add` + `Tensor::mul` broadcast | N/A | low | trivial |
| `concat_padding_mask=True` (mask channel concat into latent) | `Tensor::cat` along channel axis + `.contiguous()` | N/A | medium | **must `.contiguous()` after cat** (CONTEXT.md trap) |
| Unpatchify (FinalLayer) | reuse Klein/Z-Image `unpatchify` (Linear → reshape → permute → reshape) | N/A | low | spatial=2, temporal=1, out_c=16 |
| Rectified-flow Euler step | new sampler module `cosmos_rf_sampler.rs` (or reuse Klein/Z-Image's flow-match sampler) | N/A | medium | scheduler at `schedulers/rectified_flow.py` is short (172 lines); port verbatim |
| Wan 2.1 VAE encode (i2v / v2v conditioning) | `inference_flame::vae::wan21_encoder::Wan21VaeEncoder` (already shipped) | N/A | low | reuse |
| Wan 2.1 VAE decode (final pixel output) | `inference_flame::vae::wan21_vae::Wan21VaeDecoder::decode` (already shipped) | N/A | low | reuse |
| Cosmos-Reason1-7B text encode (Qwen2.5-VL) | `inference_flame::models::qwen25vl_encoder::Qwen25VLEncoder` (already shipped, 749 LOC, Kandinsky-5 path) | N/A | medium | Cosmos-Reason1-7B = Qwen2.5-VL-7B-Instruct fine-tune → same architecture, same encoder. Verify weight keys map 1:1. |
| MP4 mux | `inference_flame::mux::write_mp4_video_only` (already shipped) | N/A | low | reuse |

## Weight mapping (HF state_dict → flame layer names)

Source: PyTorch `.pt` checkpoint at `nvidia/Cosmos-Predict2.5-2B/base/<variant>/<uuid>_ema_bf16.pt`.

**Step 0 of build**: write `parity/convert_dit_pt_to_safetensors.py` — loads `.pt`, drops any non-EMA buffers, writes `cosmos_predict25_2b_dit.safetensors`. Rust loads safetensors only (pure-Rust rule).

PyTorch model attribute paths (from `MiniTrainDIT.__init__`, file `networks/minimal_v4_dit.py:1446`):

```
x_embedder.proj.weight                          → patch_embed.proj.weight     [hidden, in_c*p_s²*p_t]
t_embedder.0.<no params>                        # Timesteps (sinusoidal)
t_embedder.1.linear_1.weight                    → time_embed.fc1.weight       [hidden, hidden]
t_embedder.1.linear_1.bias                      → time_embed.fc1.bias         [hidden]    ⚠ ABSENT when use_adaln_lora=True (PyTorch: `bias=not use_adaln_lora`). V2_2B has use_adaln_lora=True → no bias.
t_embedder.1.linear_2.weight                    → time_embed.fc2.weight       [hidden, hidden]
t_embedder.1.linear_2.bias                      → time_embed.fc2.bias         [hidden]    ⚠ same condition — absent for V2_2B
t_embedder.1.adaln_lora.0.weight                → time_embed.adaln_lora.0.weight   [adaln_dim, hidden]  (only if use_adaln_lora)
t_embedder.1.adaln_lora.1.weight                → time_embed.adaln_lora.1.weight   [3*hidden, adaln_dim]
t_embedding_norm.weight                         → time_norm.weight             [hidden]
extra_pos_embedder.pos_emb_t                    → pos_emb.t                    [len_t, hidden]   (when extra_per_block_abs_pos_emb=True, which V2_2B has)
extra_pos_embedder.pos_emb_h                    → pos_emb.h                    [len_h, hidden]
extra_pos_embedder.pos_emb_w                    → pos_emb.w                    [len_w, hidden]
pos_embedder.seq                                → (not loaded — buffer recomputed at init)
pos_embedder.dim_spatial_range                  → (not loaded — recomputed at init)
pos_embedder.dim_temporal_range                 → (not loaded — recomputed at init)
                                                  # NOTE: For V2_2B with extra_per_block_abs_pos_emb=True:
                                                  #   self.pos_embedder       = VideoRopePosition3DEmb (rope3d, no learnable params)
                                                  #   self.extra_pos_embedder = LearnablePosEmbAxis    (learnable pos_emb_{t,h,w})
                                                  # confirm via parity/dump_state_dict_keys.py at intake load.
blocks.<i>.adaLN_modulation.{0,1}.weight        → blocks[i].mod_lora.{0,1}.weight
blocks.<i>.adaLN_modulation.linear.weight       → blocks[i].mod_proj.weight    [3*hidden, hidden_or_adaln_dim]
blocks.<i>.self_attn.q.weight                   → blocks[i].self_attn.q.weight
blocks.<i>.self_attn.k.weight                   → blocks[i].self_attn.k.weight
blocks.<i>.self_attn.v.weight                   → blocks[i].self_attn.v.weight
blocks.<i>.self_attn.q_norm.weight              → blocks[i].self_attn.q_norm.weight  [head_dim]
blocks.<i>.self_attn.k_norm.weight              → blocks[i].self_attn.k_norm.weight  [head_dim]
blocks.<i>.self_attn.output_proj.weight         → blocks[i].self_attn.out_proj.weight
blocks.<i>.cross_attn.q.weight                  → blocks[i].cross_attn.q.weight
blocks.<i>.cross_attn.k.weight                  → blocks[i].cross_attn.k.weight      (text K)
blocks.<i>.cross_attn.v.weight                  → blocks[i].cross_attn.v.weight      (text V)
blocks.<i>.cross_attn.q_norm.weight             → blocks[i].cross_attn.q_norm.weight
blocks.<i>.cross_attn.k_norm.weight             → blocks[i].cross_attn.k_norm.weight
blocks.<i>.cross_attn.k_img.weight              → blocks[i].cross_attn.k_img.weight  (image K, dual branch)
blocks.<i>.cross_attn.v_img.weight              → blocks[i].cross_attn.v_img.weight
blocks.<i>.cross_attn.k_img_norm.weight         → blocks[i].cross_attn.k_img_norm.weight
blocks.<i>.cross_attn.output_proj.weight        → blocks[i].cross_attn.out_proj.weight
blocks.<i>.mlp.fc1.weight                       → blocks[i].mlp.fc1.weight
blocks.<i>.mlp.fc2.weight                       → blocks[i].mlp.fc2.weight
final_layer.adaln_lora.0.weight                 → final.adaln_lora.0.weight
final_layer.adaln_lora.1.weight                 → final.adaln_lora.1.weight
final_layer.proj.weight                         → final.proj.weight                  [p_s²*p_t*out_c, hidden]
crossattn_proj.0.weight                         → crossattn_proj.0.weight            [crossattn_emb_channels, crossattn_proj_in_channels] = [1024, 100352]  ⚠ Present iff `use_crossattn_projection=True`. Production checkpoint (Cosmos-Reason1-7B FULL_CONCAT) sets this; base `COSMOS_V2_2B_NET` does NOT. Python `:1565-1569`.
crossattn_proj.0.bias                           → crossattn_proj.0.bias              [crossattn_emb_channels] = [1024]                                       ⚠ Same gate as above. GELU at index 1 has no learnable params (no key).
```

**Note on `crossattn_proj` index numbering**: Python `nn.Sequential(Linear, GELU)` numbers children 0, 1. `Linear` is at index 0 (→ `crossattn_proj.0.weight`, `crossattn_proj.0.bias`); `nn.GELU()` is at index 1 with no params. The `nn.GELU()` is the **exact-erf** form (Python defaults `approximate='none'`), not tanh-approx — flame-core's `Tensor::gelu_exact()` is used. Confirmed against `minimal_v4_dit.py:1568`.

**Note**: confirm exact attribute names at intake load — write `parity/dump_state_dict_keys.py` at build kickoff. The above is the expected pattern; PyTorch source uses these but EMA wrappers can prepend `module.` or `ema.`.

Fused QKV: Cosmos uses **split Q/K/V** (`self.q`, `self.k`, `self.v` Linears). Our `fused_linear3d_native` works fine on split — no conversion needed. (Klein-style fused-QKV would be an optimization, deferred.)

## flame-core changes needed

**None blocking.** All required ops exist:
- 3-axis RoPE freq builder — implemented at **port** level, not flame-core (no kernel work needed, just compose existing `rope_halfsplit_bf16`)
- adaLN-LoRA — pure composition of existing ops
- Dual-K/V cross-attention — two SDPA calls + add — no new kernel
- `tokenizer.pth` (Wan VAE) load — `Wan21VaeDecoder` / `Wan21VaeEncoder` already shipped

**Nice-to-have, deferred:**
- `fused_axis_rope_3d_bf16` kernel — would replace the 3-axis freq build + halfsplit RoPE with a single launch. Performance optimization, not correctness. Defer until smoke is clean.

## Build order

1. **Port skeleton + state-dict dumper** (port-build kickoff)
   - `src/models/cosmos_predict25_dit.rs` empty module with config struct `CosmosPredict25Config`
   - `parity/convert_dit_pt_to_safetensors.py` — converts `.pt` → safetensors offline
   - `parity/dump_state_dict_keys.py` — prints every key + shape to confirm weight names
   - Smoke: `cargo check`

2. **Embeddings** (timestep + pos)
   - Sinusoidal `Timesteps` (copy `Wan22Dit::sinusoidal_embedding`)
   - `TimestepEmbedding` with adaLN-LoRA branch
   - `LearnablePosEmbAxis` (T/H/W parameter buffers + broadcast-add)
   - Smoke: load weights, forward t=0.5 → check output shape

3. **3-axis RoPE freq builder**
   - `build_cosmos_rope_freqs(head_dim=128, T, H, W, fps, base_fps=24)` → returns `(cos, sin)` shape `[T*H*W, head_dim]`
   - Axis split: `dim_h = head_dim // 6 * 2 = 42`, `dim_w = 42`, `dim_t = head_dim - 2*dim_h = 44`
   - Build per-axis freqs (NTK-aware), outer-product with positions, concat `[t,h,w] * 2` along last dim → exactly the half-split layout `rope_halfsplit_bf16` expects
   - Test: cosmos `VideoRopePosition3DEmb.generate_embeddings` Python ref → Rust output, cos ≥ 0.999

4. **Attention modules**
   - `Attention` (self-attn): QKV split projections, Q/K RMSNorm, `rope_halfsplit_bf16` on Q & K, `sdpa`, out_proj
   - `I2VCrossAttention`: super's text-K/V/Q, plus `k_img`/`v_img` projections + `k_img_norm`, two SDPA calls, sum outputs, out_proj
   - Test: per-block parity at attention output, cos ≥ 0.999

5. **FFN**
   - `GPT2FeedForward` (Linear → GELU → Linear, no bias)
   - Verify GELU variant (tanh approx vs exact erf) by reading PyTorch source

6. **Block**
   - adaLN-LoRA modulation generator from `emb_B_T_D`
   - 6-modulator pattern: (shift, scale, gate) × {self-attn, cross-attn, ffn}? — actually 3 chunks per sub-block × 3 sub-blocks = 9 chunks. Confirm against `Block.forward` source.
   - At every block: add `LearnablePosEmbAxis` broadcast to x
   - Sub-block residuals
   - Parity: full block output, cos ≥ 0.999

7. **Full model (`MiniTrainDIT.forward`)**
   - Patch embed
   - Concat padding mask channel (with `.contiguous()`)
   - Loop over 28 blocks
   - Final layer (LayerNorm + adaLN-LoRA → Linear → unpatchify)
   - Parity: full forward at one timestep, cos ≥ 0.999

8. **Rectified-flow sampler**
   - Port `schedulers/rectified_flow.py` verbatim (172 lines)
   - Euler step, CFG split (text + image cond branches), num_steps configurable
   - Smoke: 5-step sample with random latent + zero text emb → finite, sane magnitude

9. **Text encoder wiring (Cosmos-Reason1-7B / Qwen2.5-VL)**
   - **Reuse**: `inference-flame/src/models/qwen25vl_encoder.rs` (749 LOC, Qwen2.5-VL-7B-Instruct text-only path, originally shipped for Kandinsky-5). Cosmos-Reason1-7B is a fine-tune of the same base — identical architecture, same encoder code.
   - Verify weight keys map 1:1 by dumping Cosmos-Reason1-7B vs Qwen2.5-VL-7B-Instruct safetensors keys side-by-side (`parity/dump_reason1_keys.py`).
   - **Chat template**: Qwen2.5-VL uses a specific chat-template format. Check Kandinsky-5 binary for existing Rust-side template application; if not present, port it using the `tokenizers` crate's chat-template support (or hand-roll the prefix/suffix tokens — the template is a fixed format).
   - Build `cosmos_reason1_text_encode(prompt: &str) → Tensor` helper at `src/models/cosmos_reason1.rs` (thin wrapper that loads weights, applies chat template, runs `Qwen25VLEncoder::forward`, returns hidden states + pad mask).
   - Parity script: `parity/cosmos_reason1_text_emb_ref.py` — runs Cosmos-Reason1-7B via HF transformers, dumps hidden states; Rust matches with cos ≥ 0.999.

10. **Wan VAE wiring**
    - Confirm `tokenizer.pth` key layout matches `Wan21VaeDecoder` / `Wan21VaeEncoder` — if rename needed, do it at load time in the binary (don't fork the VAE module)
    - Smoke: encode a single-frame image → decode back, cos ≥ 0.999 vs Python ref

11. **Three end-to-end binaries** (user direction: separate bins, not unified)
    - `src/bin/cosmos_predict25_t2v_infer.rs` — text-only conditioning
      - CLI: `--prompt`, `--num-frames`, `--num-steps`, `--cfg`, `--seed`, `--variant {pre-trained|post-trained|distilled}` (default `post-trained`), `--resolution {480p|720p}` (default `480p`)
      - Stage sequence: **load text encoder** (Cosmos-Reason1-7B) → encode prompt → **drop text encoder** → load DiT → denoise loop → **drop DiT** → load Wan21VaeDecoder → decode latent → **drop VAE** → mp4 mux
    - `src/bin/cosmos_predict25_i2v_infer.rs` — image conditioning
      - CLI adds `--input-image <path>`
      - Stage sequence: load text encoder → encode prompt → drop text encoder → **load Wan21VaeEncoder** → encode image to 1 latent conditioning frame → **drop encoder** → load DiT (passes image_context to I2VCrossAttention dual-K/V branch) → denoise → drop DiT → load Wan21VaeDecoder → decode → drop → mp4
    - `src/bin/cosmos_predict25_v2v_infer.rs` — video conditioning
      - CLI adds `--input-video <path>`
      - Stage sequence: same as i2v but encoder produces `num_latent_conditional_frames > 1` (multiple frames)
    - Common helpers in `src/bin/cosmos_predict25_common.rs` (config parsing, stage sequencer, BlockOffloader setup at 720p)

12. **Smoke** (`/port-smoke` skill)
    - **GPU-gated, deferred until HiDream trainer releases the GPU.**
    - Bisect: 1-step → 5-step → 50-step → full
    - Resolution: 480p first (832×480), then 720p (1280×704) only if 480p fits
    - Hard caps: 78°C, 20 min per [[feedback_gpu_thermal_budget]]

## Risk register

### High
- **R1 — VRAM at 720p.** Upstream lists 32.54 GB. We have 24 GB. Mitigation: 480p first, BlockOffloader for DiT if needed, drop num_frames if needed. **No code workaround needed at build phase** — set BlockOffloader path up front so it's available; choose at runtime via `--resolution`.
- **R2 — Cosmos-Reason1-7B text encoder on 24 GB**. Qwen2.5-VL-7B in BF16 = ~14 GB; activations push that to ~16-17 GB during forward. With DiT (~4 GB weights + activations) and VAE (~1 GB), the three stages cannot coexist on 24 GB. Mitigation: strict load-encode-drop sequencing in each binary; pinned-memory transfer of text embeddings between stages. This is identical to the wan22 / Helios T2V stage discipline. ([[project_helios_port_state]] precedent.) **No cached-embedding fallback** per user direction — pure-Rust live encode is the spec.
- **R3 — RoPE rotation pattern mismatch.** Cosmos uses TE's `apply_rotary_pos_emb(..., fused=True)` which is **half-split** (GPT-NeoX layout). Our `rope_halfsplit_bf16` matches. **Must NOT route to interleaved `rope_fused_bf16`** — that's the HiDream-O1 / Z-Image trap ([[project_hidream_o1_qkv_lora_grad_collapse_2026-05-20]], [[feedback_rope_fused_autograd]]). Per-axis-freq layout (`[t,h,w]*2`) confirmed half-split-friendly by reading `VideoRopePosition3DEmb.generate_embeddings:785-793`.

### Medium
- **R4 — `tokenizer.pth` Wan VAE key naming.** If Cosmos renamed keys vs upstream Wan 2.1, we'd need a load-time remapper. Mitigation: dump keys in step 10; remap in binary not in VAE module.
- **R5 — `concat_padding_mask` channel join non-contig hazard.** `Tensor::cat` doesn't guarantee contiguous, downstream Linear may silently miscompute. Mitigation: always `.contiguous()` after cat (CONTEXT.md trap).
- **R6 — adaLN-LoRA chunk layout.** 3-chunk split (shift, scale, gate) — confirm chunk order against PyTorch source at block 6. Wrong order = silently wrong output, hard to detect from L2 norms.
- **R7 — `extra_per_block_abs_pos_emb=True`**: learnable pos emb added every block. Easy to forget after block 0. Mitigation: make this part of the Block forward, not the model forward.
- **R8 — GPT2FeedForward GELU variant.** Tanh approx vs exact erf is a one-line difference but ~1% magnitude error. Read source at build step 5.
- **R9 — Three weight variants** (`pre-trained` / `post-trained` / `distilled`). Distilled has different sampler defaults (fewer steps, no CFG). Default to `post-trained` per upstream README; expose `--variant` flag.

### Low
- **R10 — BF16 cos/sin precision floor.** [[project_bf16_rope_pattern_audit_2026-05-19]]: many ports cast cos/sin to BF16 at build, losing precision. Cosmos's RoPE freqs are simple sinusoidal — keep cos/sin in F32 until just before the kernel call, then cast.
- **R11 — `.pt` checkpoint EMA wrapper.** EMA keys may be prefixed with `ema.` or `module.` — strip in the conversion script.

## Memory plan

- **Inference 480p (832×480)**: target plain mode (no offload). Math: DiT 2B BF16 = 4 GB activations ≈ 6-8 GB at 480p × 24 latent frames → ~12 GB total. Should fit on 24 GB after VAE/text encoder are dropped between stages.
- **Inference 720p (1280×704)**: needs BlockOffloader. Activations at 720p ≈ 18-20 GB → 4 GB margin. Stream-load blocks (Klein/ERNIE precedent).
- **Text encoder**: load → encode → drop, even in the live path. Pinned-memory cache between stages.
- **Per-stage handoff**: explicit `drop(text_encoder)` + `Wan21VaeEncoder::drop` + `Wan21VaeDecoder::drop` between phases.
- No training in this port — no checkpointing / grad-checkpointing needed.

## Parity plan

- **Reference**: `cosmos_predict2/_src/predict2/inference/video2world.py` (orchestrator) + `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` (forward).
- **Capture points** (per-layer GPU streaming Python script `parity/per_layer_capture.py`, generated at build kickoff):
  1. patch_embed output `[B, T*H*W, hidden]`
  2. timestep embedding output (before + after adaLN-LoRA) `[B, hidden]` and `[B, 3*hidden]`
  3. learnable pos emb per axis: `pos_emb_t`, `pos_emb_h`, `pos_emb_w` slices
  4. block 0 self-attn output `[B, S, hidden]`
  5. block 0 cross-attn output (with and without `img_context`)
  6. block 0 ffn output
  7. block 0 final output
  8. blocks 13, 27 final output (mid and last)
  9. final_layer output (pre-unpatchify) `[B, S, p²*p_t*out_c]`
  10. unpatchify output `[B, out_c, T, H, W]`
- **Generation method**: per-layer GPU streaming Python, **NOT** full-model CPU (CONTEXT.md rule, [[feedback_pytorch_cpu_vs_cuda_bf16]]).
- **Bar**: cos ≥ 0.999 per capture point. cos=0.99 with high max_abs = cat-not-contig or RoPE-layout mismatch.
- **Harness**: `flame_core::parity::ParityHarness` ([[project_flame_diagnostics_parity_2026-05-09]]).

## Text encoder strategy (locked: pure-Rust live encode)

Cosmos-Reason1-7B is Qwen2.5-VL-7B-Instruct fine-tuned for physical-AI prompts. Cross-attention uses its hidden states. **Pure-Rust live encode** per CONTEXT.md runtime rule and user direction.

**The Rust encoder is already shipped.** `inference-flame/src/models/qwen25vl_encoder.rs` (749 LOC) implements the Qwen2.5-VL-7B-Instruct text-only forward path. Originally built for Kandinsky-5 inference. Cosmos-Reason1-7B has the **identical architecture** (only weights differ), so the encoder code is reusable verbatim.

**What's left for step 9:**
1. Verify Cosmos-Reason1-7B safetensors weight keys map 1:1 to the encoder's HashMap expectations (likely identical — both follow HF Qwen2.5-VL conventions). One Python dump-script confirms.
2. Apply the Qwen chat template in Rust before tokenization (Cosmos source: `text_encoder.py:163` calls `tokenizer.apply_chat_template`). The template is a fixed string format with `<|im_start|>` / `<|im_end|>` markers — port directly using the `tokenizers` crate.
3. Build `cosmos_reason1_text_encode(&str) → Tensor` wrapper at `src/models/cosmos_reason1.rs` (thin layer over `Qwen25VLEncoder`).
4. Parity ref `parity/cosmos_reason1_text_emb_ref.py` — run Cosmos-Reason1-7B via HF transformers, dump hidden states; Rust matches cos ≥ 0.999.

Python is **only** used for parity reference generation in `parity/` — runtime is pure Rust ([[feedback_python_is_dev_tool]]).

## Decisions locked (2026-05-20)

- **Variant**: default `post-trained`. Loader accepts any of `{pre-trained, post-trained, distilled}` via `--variant` flag (since adding it costs ~5 lines).
- **Text encoder**: pure-Rust live encode using existing `qwen25vl_encoder.rs`. No cached-embedding fallback.
- **Resolution**: 480p first as smoke target. 720p code path written from day one (BlockOffloader-backed), gated on `--resolution 720p` at runtime.
- **Binary structure**: three separate binaries — `cosmos_predict25_t2v_infer`, `_i2v_infer`, `_v2v_infer`. Shared helpers in `cosmos_predict25_common.rs`.
