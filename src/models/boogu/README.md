# boogu — Boogu-Image-0.1-Base (pure-Rust inference)

Pure-Rust flame-core port of **Boogu-Image-0.1-Base**, a Flux/Qwen-Image-class
flow-matching T2I model (Lumina2/OmniGen2 DiT lineage, 38.5 GB bf16). Faithful
re-implementation of the verified pure-Mojo port, gated against the Python torch
oracle. **Inference only** (T2I + prompt rewriter).

## Architecture
- **DiT** (`transformer.rs`): 8 double-stream + 32 single-stream blocks + 2 context-
  refiner + 2 noise-refiner. hidden 3360, 28 q / 7 kv heads (GQA 4:1), head_dim 120,
  3-axis interleaved RoPE, RMSNormZero modulation, SwiGLU. Resident on 24 GB.
- **Text encoder** (`encoder.rs`): Qwen3-VL-8B language tower (reuses `qwen3_encoder`),
  `model.language_model.*`→`model.*` remap, `hidden_states[-1]`.
- **VAE**: stock FLUX.1-dev 16ch (`vae::ldm_decoder`), tiled decode at 1024 (`decode_1024.rs`).
- **Scheduler** (`sampling/boogu_sampling.rs`): FlowMatchEuler + static v1 time-shift (mu 1.15).
- **Rewriter** (`generate.rs`): Qwen3-VL KV-cached autoregressive decode + sampler.

## Run
```
boogu_encode "<prompt>"                       # → output/boogu_embeddings.safetensors
boogu_infer --size 1024 --steps 20 --seed 42  # → output/boogu_rust.png
boogu_rewrite "<short idea>"                   # expand a rough idea → full instruction
```

## Verification (vs torch oracle)
- Full DiT velocity: cos **0.9999189**.
- End-to-end pipeline: latent cos **0.9965**, image **PSNR 31.1 dB**.
- Rewriter (greedy): **100% token match** vs `transformers .generate`.

## Speed
- **1024² denoise: 2.24 s/step → 7.6× faster than the Mojo port** (17.06 s/step); peak ~20.9 GB.
- head_dim 120 hits flame's fused cuDNN SDPA by zero-padding to 128 with a pre-scaled Q
  (keeps the softmax scale exactly 1/√120) — ~12× the manual FP32 path.

flame-core changes required: **none** (all ops map to existing primitives).
Detailed state in `PORT_STATE.md` (local-only per repo docs policy).
