# PORT_SPEC: cosmos-predict25-2b

Last updated: 2026-05-20 by port-intake

## Source

- HF repo: https://huggingface.co/nvidia/Cosmos-Predict2.5-2B (gated)
- Reference repo: https://github.com/nvidia-cosmos/cosmos-predict2.5 (cloned at `/home/alex/refs/cosmos-predict2.5/`)
- Paper: arXiv 2511.00062 — "World Simulation with Video Foundation Models for Physical AI"
- License: NVIDIA Open Model License (commercial use OK; safety-guardrail bypass terminates rights)

## Architecture

- Type: **MiniTrainDIT** (single-stream DiT with cross-attention to text encoder)
- Variant: COSMOS_V2_2B_NET (`cosmos_predict2/_src/predict2/configs/text2world/defaults/net.py:79`)
- Params: 2,059,174,912 (~2.06B)
- Precision: BF16 (FP16/F32 unsupported per model card)
- Block count: 28
- Hidden dim / heads / head_dim: 2048 / 16 / 128
- Patch: spatial=2, temporal=1
- In/out channels: 16 / 16 (latent space)
- Max spatial grid: 240×240 patches (i.e. 480×480 patch coords, full 720p frame 1280×704)
- Max frames (latent): 128

### Module breakdown (`networks/minimal_v4_dit.py`)
- `RMSNorm` (eps=1e-6) — used for `t_embedding_norm`, plus TE's RMSNorm for Q/K norm
- `Timesteps` + `TimestepEmbedding` — sinusoidal then MLP; if `use_adaln_lora=True`, second linear is rank-256 → 3*hidden
- `FourierFeatures` — for additional conditioning
- `PatchEmbed` — `Linear(in_c * patch_s² * patch_t, hidden, bias=False)`
- `VideoRopePosition3DEmb` — 3D RoPE applied per axis (T/H/W); GPT-NeoX half-split layout (`apply_rotary_pos_emb(..., fused=True)` from transformer-engine)
- `LearnablePosEmbAxis` — per-axis additive learnable absolute positional embedding (T/H/W); `extra_per_block_abs_pos_emb=True` so this gets added every block
- `Block` × 28:
  - Self-`Attention`: x_dim=2048, ctx_dim=2048, head_dim=128, with Q/K RMSNorm (head-dim, eps=1e-6), 3D RoPE applied to Q and K
  - `I2VCrossAttention`: queries from x, keys/values from text-encoder embeddings (text_context) **and** optional image latent (`img_latent_dim=1024`, second K/V branch with its own `k_img_norm`); no RoPE on cross-attention
  - `GPT2FeedForward` (GELU); FFN expansion derived from `ffn_mult` in net config
  - adaLN-LoRA modulation: 3-chunk (shift, scale, gate) per sub-block; LoRA dim=256
- `FinalLayer`: `LayerNorm(elementwise_affine=False)` + adaLN-LoRA(2-chunk) + `Linear(hidden, patch_s² * patch_t * out_c, bias=False)`

## Weights

- File: `base/pre-trained/<uuid>_ema_bf16.pt` — 4.12 GB
  - `d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt` for pre-trained
  - `81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt` for post-trained (4.12 GB)
  - Three variants under `base/`: `pre-trained/`, `post-trained/`, `distilled/`
- Format: **PyTorch pickle (.pt)**, NOT safetensors → must convert offline (Python parity script under `parity/`)
- Sharding: none, single file
- Also at repo root: `tokenizer.pth` (508 MB) — this is the **Wan 2.1 VAE** (see Dependencies)

## Dependencies

- **Text encoder**: `nvidia/Cosmos-Reason1-7B` — a Qwen2.5-VL-7B-Instruct fine-tune used as a vision-language text encoder for physical-AI prompts. Approx 15 GB BF16 in safetensors. Cross-attention key/value source.
  - Token sequence is from the chat-template-applied Qwen tokenizer, padded with `tokenizer.pad_id`. See `text_encoder.py:163-170`.
  - Reuse opportunity: `inference-flame/src/models/` already has Qwen-family encoder work for qwenimage; check for module overlap before reimplementing.
- **VAE**: **Wan 2.1 VAE**, packaged inside `nvidia/Cosmos-Predict2.5-2B/tokenizer.pth` (508 MB)
  - 16 latent channels, 8× spatial compression, **4× temporal** compression (note: config name `cv8x8x8` is misleading — actual is 8×8×4, see `tokenizers/wan2pt1.py:878-883` and `:1034-1040`)
  - **Already ported**: `inference-flame/src/vae/wan21_vae.rs` (decoder) + `wan21_encoder.rs` — should drop-in
- **Scheduler**: **Rectified Flow** (flow matching) — `cosmos_predict2/_src/predict2/schedulers/rectified_flow.py`, 172 lines, similar pattern to Z-Image/Klein flow-matching sampling
- **Tokenizer (text)**: Qwen2.5-VL-7B-Instruct tokenizer (HF tokenizer.json, plus chat-template apply)
- **Guardrails** (optional, will skip): `nvidia/Cosmos-Guardrail1` — text + video safety; only enforced by upstream `inference.py` when `--disable_guardrails` not set. Skip for our port; this is a research/inference pipeline, not a hosted service.

## Forward-pass ops (vs flame-core)

| Op | flame-core path | Status |
|----|------|--------|
| RMSNorm | `flame_core::norm::rms_norm` | ✓ |
| LayerNorm | `flame_core::norm::layer_norm` | ✓ |
| Linear (bias=False) | `flame_core::nn::linear` | ✓ |
| SiLU / GELU | `flame_core::activation::{silu,gelu}` | ✓ |
| 3D RoPE (half-split / GPT-NeoX layout, fused TE-style) | `flame_core::rope::*` | likely ✓ via halfsplit cohort (see [[project_bf16_rope_pattern_audit_2026-05-19]]) — verify cos/sin precision; BF16 trap applies if cast at construction |
| Q/K RMSNorm (per-head) | rms_norm over last dim | ✓ |
| SDPA | `flame::sdpa` | ✓ |
| Sinusoidal timesteps | reuse Klein/Z-Image timestep encoder | ✓ |
| Fourier features | small module | port directly |
| Learnable additive abs pos emb (per-axis T/H/W) | new parameter buffers, simple broadcast-add | port directly |
| adaLN-LoRA (Linear→SiLU→Linear stack producing 3*hidden chunks) | composition of existing ops | port directly |
| Patch embed (Linear over flattened patches) | `unfold` + Linear | reuse Klein-style |
| Final-layer unpatchify | reuse Klein/Z-Image unpatchify | ✓ |

### Special / risk items
- **RoPE layout**: TE's `apply_rotary_pos_emb(..., fused=True)` is the GPT-NeoX **half-split** form (first-half / second-half) — NOT interleaved. Apply via flame-core's halfsplit RoPE; do NOT route to the interleaved kernel (HiDream-O1 trap, see [[project_hidream_o1_qkv_lora_grad_collapse_2026-05-20]]).
- **3D RoPE composition**: cos/sin tensors per (T, H, W) axis are concatenated on the head-dim (not summed). Verify the exact split (likely head_dim split into T/H/W thirds, each then half-split) — read `VideoRopePosition3DEmb.forward` carefully during build.
- **Cross-attention with dual K/V branch**: I2VCrossAttention has two K/V projections — one from text context, one from image latent — concatenated along seq. Mask shape must allow both.
- **adaLN-LoRA**: Two-layer Linear stack (`Linear(hidden, 256, bias=False)` → SiLU → `Linear(256, 3*hidden, bias=False)`) producing modulation params. Cheap.
- **`extra_per_block_abs_pos_emb=True`**: learnable abs pos emb is added **at every block**, not just once. Don't fold into the patch embed.
- **`concat_padding_mask=True`**: a padding-mask channel is concatenated into the latent at input.

## Parity reference

- Inference: `cosmos_predict2/_src/predict2/inference/video2world.py` + `cosmos_predict2/inference.py` (orchestrator). The `Video2WorldInference` class is the single end-to-end entry point.
- Per-layer GPU streaming Python ref under `inference-flame/ports/cosmos-predict25-2b/parity/` — write at port-build time; do NOT generate on CPU (BF16 CPU vs CUDA divergence trap).
- Bar: `cos ≥ 0.999` per layer with `flame_core::parity::ParityHarness`.

## Goal

- **Inference only.** Modes: text2video, image2video, video2video (all three unified in one model — input branch toggles `num_latent_conditional_frames`).
- Target dir: `/home/alex/EriDiffusion/inference-flame/`
  - DiT module: `src/models/cosmos_predict25_dit.rs`
  - Binary: `src/bin/cosmos_predict25_infer.rs`
  - Port docs: `ports/cosmos-predict25-2b/`
  - Reuse: `src/vae/wan21_vae.rs` (already there), `src/mux.rs` for mp4

## Open questions

1. **VRAM budget on 24 GB.** Upstream lists 32.54 GB at 720p. We have 24 GB. Options:
   - Drop to 480p (832×480) variant — still flagged as supported.
   - Apply BlockOffloader pattern (Klein/ERNIE/flux precedent — see [[project_edv2_flux_blockoffloader_2026-05-07]]).
   - Bisect by frame count first; max_frames=128 latent ≈ 509 pixel frames at 4× temporal. Smaller `num_output_frames` may fit.
   - Decide during port-plan.
2. **Cosmos-Reason1-7B encoder**: full Qwen2.5-VL forward needed, or a frozen-embedding cache path acceptable for inference? Upstream calls it live every prompt; running 7B VLM + 2B DiT + Wan2.1 VAE on 24 GB requires offload between stages.
3. **`.pt` → `.safetensors` conversion**: write a one-shot Python parity script `parity/convert_dit_pt_to_safetensors.py` at port-build kickoff; Rust loads safetensors only.
4. **Variant choice**: `base/pre-trained/` vs `base/post-trained/` vs `base/distilled/` — confirm which matches the README's reported quality numbers; default to `post-trained` for end-user inference unless distilled is the recommended consumer ckpt.
5. **`tokenizer.pth` format**: is it a torch state_dict that maps 1:1 onto our existing `Wan21Vae*` Rust modules, or does Cosmos rename keys? Verify before plan phase.
6. **`extra_image_context_dim`**: COSMOS_V2_2B_NET sets `extra_per_block_abs_pos_emb=True` but does it use the I2V image-latent K/V branch by default? `MiniTrainDIT.__init__` gates with `extra_image_context_dim is None`. The 2B is a unified t2v/i2v/v2v model so likely yes — confirm.

## References to existing memory

- BF16 RoPE precision trap: [[project_bf16_rope_pattern_audit_2026-05-19]]
- RoPE half-split vs interleaved hazard: [[project_hidream_o1_qkv_lora_grad_collapse_2026-05-20]]
- Wan VAE / video DiT precedent: [[project_helios_port_state]] (Helios is also a video DiT with Wan VAE), wan22 port at `inference-flame/ports/wan22/`
- BlockOffloader for >24GB DiTs: [[project_edv2_flux_blockoffloader_2026-05-07]]
- Tensor::cat non-contig hazard (matters for 3D RoPE concat across axes): see CONTEXT.md "Known traps"
