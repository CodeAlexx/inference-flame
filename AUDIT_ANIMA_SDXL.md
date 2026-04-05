# Inference-Flame Pipeline Audit: All Models
**Date**: 2026-04-03
**Verified against**: ComfyUI ground truth (`serenityflow/comfy/ldm/`)
**Also checked**: `serenity/inference/`, `serenity-inference/models/`

## SDXL — PASS

SDXL UNet (`src/models/sdxl_unet.rs`) and LDM VAE (`src/vae/ldm_decoder.rs`) match Python references.

| Component | Status | Notes |
|-----------|--------|-------|
| UNet config (320ch, mult=[1,2,4]) | OK | All params match |
| Transformer depth [0,0,2,2,10,10] / 10 / [0,0,0,2,2,2,10,10,10] | OK | |
| Timestep embed: sin(320) → Linear(320→1280,bias) → SiLU → Linear(1280→1280,bias) | OK | |
| Label embed: Linear(2816→1280,bias) → SiLU → Linear(1280→1280,bias) | OK | |
| ResBlock GroupNorm eps=1e-5 | OK | UNet uses PyTorch default 1e-5 (VAE separately uses 1e-6) |
| SpatialTransformer GroupNorm eps=1e-6 | OK | Matches diffusers Attention norm |
| BasicTransformerBlock LayerNorm eps=1e-5 | OK | Matches diffusers |
| Attention: to_q/k/v no bias, to_out has bias | OK | |
| GEGLU feed-forward | OK | |
| Linear proj_in/proj_out (use_linear_in_transformer=true) | OK | |
| Skip connection channel tracking | OK | Verified via test |
| Upsample sub-index (1 if no transformer, 2 if transformer) | OK | |
| VAE: scale=0.13025, shift=0.0, block_out=(128,256,512,512), 3 resnets, eps=1e-6 | OK | |

**SDXL is ready for end-to-end testing once a binary is wired up.**

---

## ANIMA (Cosmos Predict2) — 4 BUGS FOUND

Reference: ComfyUI `comfy/ldm/cosmos/position_embedding.py`, `comfy/ldm/cosmos/blocks.py`, `comfy/ldm/anima/model.py`

### BUG 1 — CRITICAL: 3D RoPE frequency dimension split is wrong

**File**: `src/models/anima.rs:882-884`

**Rust (WRONG)**:
```rust
let dims_t: usize = 16;
let dims_h: usize = 24;
let dims_w: usize = 24;
// 16 + 24 + 24 = 64 frequency bins
```

**Python (CORRECT)** — `position_embedding.py:81-83`:
```python
dim = head_dim           # 128 (full head_dim, NOT half)
dim_h = dim // 6 * 2     # 128 // 6 * 2 = 42
dim_w = dim_h             # 42
dim_t = dim - 2 * dim_h  # 128 - 84 = 44
```
Each axis uses `dim_x // 2` frequency bins:
- **t**: 44 / 2 = **22** bins
- **h**: 42 / 2 = **21** bins
- **w**: 42 / 2 = **21** bins
- Total: 22 + 21 + 21 = **64** ✓

**Fix**: Change to `dims_t=22, dims_h=21, dims_w=21`.

### BUG 1b — CRITICAL: RoPE frequency exponents use wrong normalization

**Rust (WRONG)** — `anima.rs:890-898`:
```rust
// For h: 1.0 / theta^(2*i / (2*dims_h))  with dims_h=24
// Exponents: [0/24, 1/24, 2/24, ..., 23/24]
freqs_h: (0..dims_h).map(|i| 1.0 / theta.powf(2.0 * (i as f32) / (2 * dims_h) as f32))
```

**Python (CORRECT)** — `position_embedding.py:87,131`:
```python
dim_spatial_range = arange(0, dim_h, 2)[:dim_h//2] / dim_h
# For dim_h=42: [0/42, 2/42, 4/42, ..., 40/42]  (21 values)
h_spatial_freqs = 1.0 / (h_theta ** dim_spatial_range)
```

The exponents should be `[0/42, 2/42, 4/42, ..., 40/42]` not `[0/24, 1/24, ..., 23/24]`.

**Fix**: For each axis, compute `exponent = arange(0, dim_x, 2)[:dim_x//2] / dim_x` where `dim_x` is the *full* dimension (42 or 44), NOT the number of frequency bins.

### BUG 2 — MEDIUM: NTK extrapolation scaling missing from 3D RoPE

**File**: `src/models/anima.rs:887` — uses plain `theta = 10000.0`

**Python (CORRECT)** — `position_embedding.py:96-129`:
```python
h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
# For 16ch model: h_extrapolation_ratio=4.0, dim_h=42
# h_ntk_factor = 4.0 ** (42/40) = 4.0 ** 1.05 ≈ 4.287
h_theta = 10000.0 * h_ntk_factor   # ≈ 42870
w_theta = 10000.0 * w_ntk_factor   # ≈ 42870
t_theta = 10000.0 * t_ntk_factor   # = 10000 (t_extrapolation_ratio=1.0)
```

**Fix**: Compute per-axis NTK-scaled theta. The `h/w_extrapolation_ratio` is 4.0 for 16ch Anima, 3.0 for 15ch. Temporal ratio is always 1.0.

### BUG 3 — HIGH: LLM Adapter cross-attention missing RoPE

**File**: `src/models/anima.rs:626` — comment says "No RoPE on cross-attention"

**Python (CORRECT)** — `anima/model.py:131-132`:
```python
# Cross-attention DOES get RoPE embeddings:
attn_out = self.cross_attn(normed, ..., context=context,
    position_embeddings=position_embeddings,           # Q gets target RoPE
    position_embeddings_context=position_embeddings_context)  # K gets source RoPE
```

And in `Attention.forward` (line 73-78):
```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    cos, sin = position_embeddings_context      # different positions for K!
    key_states = apply_rotary_pos_emb(key_states, cos, sin)
```

**Fix**: Apply 1D RoPE in LLM adapter cross-attention:
- Q uses RoPE from `position_ids = arange(target_seq_len)` 
- K uses RoPE from `position_ids_context = arange(source_seq_len)` (Qwen3 hidden states length)

### BUG 4 — HIGH: Timestep embedding linear_1 has bias (should not)

**File**: `src/models/anima.rs:214-218`

**Rust (WRONG)**: uses `linear_with_bias` for `net.t_embedder.1.linear_1`

**Python (CORRECT)** — `cosmos_predict2_modeling.py:703`:
```python
self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
# With use_adaln_lora=True → bias=False
```

**Fix**: Change to `linear_no_bias` for `net.t_embedder.1.linear_1`. The safetensors file won't contain a `.bias` key for this layer.

---

## THINGS THAT ARE CORRECT (verified against ComfyUI ground truth)

| Component | Status | Reference |
|-----------|--------|-----------|
| RoPE format: rotation matrix [S, D/2, 2, 2] with interleaved pairs | OK | `blocks.py:162-163` |
| RoPE only on self-attention (backbone), NOT cross-attention (backbone) | OK | `blocks.py:159` |
| AdaLN-LoRA: SiLU → Linear(D→256) → Linear(256→3D) + base_adaln, then chunk(3) | OK | `blocks.py:688` |
| Self-attn: Q/K/V no bias, QK RMSNorm(eps=1e-6) | OK | `blocks.py` |
| Cross-attn: Q(2048), K/V(1024→2048), no bias | OK | |
| MLP: Linear(2048→8192, no bias) → GELU → Linear(8192→2048, no bias) | OK | |
| Final layer: adaln(2 outputs) → LayerNorm(no affine, eps=1e-6) → Linear(2048→64) | OK | |
| Patchify: 17ch (16+mask) × 2×2 = 68 → Linear(68→2048, no bias) | OK | |
| LLM Adapter: 6 blocks, dim=1024, 16 heads, head_dim=64, 1D RoPE | OK | `anima/model.py` |
| LLM Adapter self-attn: 1D RoPE on Q and K | OK | |
| LLM Adapter MLP: Linear(1024→4096, bias) → GELU → Linear(4096→1024, bias) | OK | `nn.Linear` default `bias=True` |
| LLM Adapter embed: Embedding(32128, 1024) | OK | T5 vocab |

---

## Impact on Klein

**None.** Klein uses a completely different RoPE implementation (2D packed positional IDs with standard axes_dims=(16,56,56) and theta=10000). Klein's RoPE was already validated and is working. The Anima 3D rotation-matrix RoPE bugs are isolated to `anima.rs`.

## Anima Priority Order for Fixes

1. **BUG 1 + 1b** (CRITICAL) — RoPE dim split + frequency exponents. Without this, every attention output is wrong.
2. **BUG 2** (MEDIUM) — NTK scaling. Affects high-res extrapolation quality.
3. **BUG 3** (HIGH) — LLM adapter cross-attn RoPE. Text conditioning won't align properly.
4. **BUG 4** (HIGH) — Timestep bias. Will crash on weight load if tested end-to-end.

---

## SD3 MMDiT — PASS

**File**: `src/models/sd3_mmdit.rs` (723 lines)
**Verified against**: ComfyUI `comfy/ldm/common_dit.py`, `serenity-inference/models/sd3_dit.py`

Every component matches the Python ground truth:

| Component | Status | Notes |
|-----------|--------|-------|
| Config auto-detect (hidden_size, depth=num_heads, patch_size, context_dim, pooled_dim) | OK | `depth = hidden_size / 64` convention correct |
| Timestep embed: sinusoidal(256) → Linear(256→H,bias) → SiLU → Linear(H→H,bias) | OK | |
| Pooled embed: Linear(pooled_dim→H,bias) → SiLU → Linear(H→H,bias) | OK | |
| Conditioning: c = t_emb + y_emb | OK | |
| Context embedder: Linear(context_dim→H, bias) | OK | |
| Patch embed: Conv2d(in_ch, H, k=P, s=P, bias=True) | OK | |
| Position embed: learned, centered crop `top=(max-ph)/2` | OK | Matches `cropped_pos_embed` exactly |
| Joint attention: fused QKV(3*H,bias), cat(ctx,x) on seq dim → SDPA → split | OK | |
| QK norm: LayerNorm(affine=True, eps=1e-6, head_dim=64) | OK | |
| adaLN: SiLU(c) → Linear → chunk(6) for non-last, chunk(2) for pre_only last | OK | |
| X stream: always 6 mods | OK | |
| Post-attn: proj(bias) → gate*proj + residual | OK | |
| MLP: Linear(H→4H,bias) → GELU(tanh) → Linear(4H→H,bias) | OK | |
| Final layer: LN(no affine,1e-6) → adaLN(2 mods) → Linear(H→P²C,bias) | OK | |
| Unpatchify: [B,N,P²C] → permute "nhwpqc→nchpwq" → [B,C,H,W] | OK | |
| No RoPE (learned pos embed) | OK | |
| LayerNorm eps=1e-6 throughout | OK | |

**SD3 is ready for end-to-end testing once a binary is wired up.**

---

## FLUX 1 (Dev/Schnell) — NOT IMPLEMENTED, REFERENCES COPIED IN

No Flux 1 Rust model exists in inference-flame. Only Klein (Flux 2) is implemented.
**Reference files copied into repo for implementation:**
- `reference_flux1dev.py` — Complete end-to-end pipeline (text encode → denoise → VAE → PNG)
- `reference_flux1_dit.py` — Full DiT architecture (534 lines, key-exact for BFL .safetensors)

### Flux 1 Architecture (from reference_flux1_dit.py)

```
FluxConfig:
  num_double_blocks: 19
  num_single_blocks: 38
  inner_dim: 3072
  num_heads: 24
  head_dim: 128
  in_channels: 64
  joint_attention_dim: 4096  (T5-XXL)
  mlp_ratio: 4.0             (GELU, not SwiGLU)
  timestep_dim: 256
  has_guidance: true          (Dev=true, Schnell=false)
  vector_dim: 768             (CLIP-L pooled)
  axes_dims_rope: (16, 56, 56)
  rope_theta: 10000.0
```

### Key Weight Names (all have bias)
```
img_in.weight/bias                          [3072, 64]
txt_in.weight/bias                          [3072, 4096]
time_in.in_layer.weight/bias                [3072, 256]
time_in.out_layer.weight/bias               [3072, 3072]
guidance_in.in_layer.weight/bias            (Dev only)
vector_in.in_layer.weight/bias              [3072, 768]

double_blocks.{i}.img_mod.lin.weight/bias   [18432, 3072] (6 mods)
double_blocks.{i}.img_attn.qkv.weight/bias  [9216, 3072]
double_blocks.{i}.img_attn.proj.weight/bias  [3072, 3072]
double_blocks.{i}.img_attn.norm.query_norm.scale  [128]
double_blocks.{i}.img_attn.norm.key_norm.scale    [128]
double_blocks.{i}.img_mlp.0.weight/bias     [12288, 3072]  GELU(tanh)
double_blocks.{i}.img_mlp.2.weight/bias     [3072, 12288]
(txt_* mirrors img_*)

single_blocks.{i}.modulation.lin.weight/bias  [9216, 3072] (3 mods)
single_blocks.{i}.linear1.weight/bias   [21504, 3072]  (QKV+GELU_up fused: 3*3072+4*3072)
single_blocks.{i}.linear2.weight/bias   [3072, 15360]  (proj+GELU_down fused: 3072+4*3072)
single_blocks.{i}.norm.query_norm.scale  [128]
single_blocks.{i}.norm.key_norm.scale    [128]

final_layer.adaLN_modulation.1.weight/bias  [6144, 3072]
final_layer.linear.weight/bias               [64, 3072]
```

### Implementation Notes
- **RoPE**: Rotation matrix format [S, D/2, 2, 2] (ComfyUI `math.py:27-28`), same as Cosmos/Anima. Also works as complex-number multiply (serenity-inference reference). Both equivalent.
- **RoPE applied AFTER concat**: txt+img are concatenated, THEN RoPE applied to the joint sequence. Position IDs: txt_ids are zeros `[N_txt, 3]`, img_ids are `(batch_idx, y_patch, x_patch)` per 2x2 packed patch.
- **Timestep**: sigma values [0,1] scaled by `time_factor=1000` before sinusoidal embedding.
- **Single block fused ops**: `linear1` fuses QKV + GELU up-projection. `linear2` fuses attn proj + GELU down-projection. Must split correctly.
- **QK norm**: RMSNorm (param name `.scale`), NOT LayerNorm. Differs from SD3 which uses LayerNorm.
- **Modulation pre-norm**: LayerNorm(no affine, eps=1e-6) → `(1+scale)*normed + shift`
- **VAE**: Use `LdmVAEDecoder` with `in_channels=16, scale=0.3611, shift=0.1159`
- **Sigma schedule**: Dynamic mu-shift based on packed sequence length. See `reference_flux1dev.py:70-106`.
- **No CFG**: Dev uses guidance embedding (scalar injected as timestep-like MLP), not classifier-free guidance.

### Key differences from Klein (Flux 2)

| Feature | Flux 1 (Dev/Schnell) | Klein (Flux 2) |
|---------|---------------------|----------------|
| Biases | Everywhere | None |
| Modulation | Per-block (6 double / 3 single) | Shared (3 at model level) |
| MLP | GELU(tanh) 4x | SwiGLU 6x |
| Guidance embed | Dev=yes, Schnell=no | Never |
| Vector in (CLIP pooled) | Yes (768d) | No |
| in_channels | 64 | 128 |
| VAE latent channels | 16 | 32 |
| RoPE theta | 10000 | 2000 |
| RoPE axes | (16,56,56) | (32,32,32,32) |
| QK norm param name | `.scale` (RMSNorm) | `.weight` (RMSNorm) |
| Blocks (full) | 19 double + 38 single | 5+20 (4B) / 8+24 (9B) |
| Model size | 12B | 4B / 9B |

---

## FULL AUDIT SUMMARY

| Model | Status | Bugs |
|-------|--------|------|
| **SDXL UNet + VAE** | PASS | 0 |
| **SD3 MMDiT** | PASS | 0 |
| **Anima (Cosmos Predict2)** | **4 BUGS** | 2 critical (RoPE), 2 high (adapter RoPE, timestep bias) |
| **Flux 1** | GAP | Not implemented. References copied: `reference_flux1_dit.py`, `reference_flux1dev.py` |
| **Klein (Flux 2)** | SKIP | Already validated and working — not audited again |
| **Z-Image** | NOT AUDITED | Out of scope for this audit |
| **LTX-2** | NOT AUDITED | Out of scope for this audit |
