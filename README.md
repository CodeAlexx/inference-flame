# inference-flame

Pure Rust diffusion model inference using [flame-core](https://github.com/CodeAlexx/Flame). No Python, no diffusers, no ONNX.

| Klein 9B | Z-Image | Anima 2B | SDXL |
|---|---|---|---|
| ![Klein 9B Sample](docs/klein9b_sample.png) | ![Z-Image Sample](docs/zimage_sample.png) | ![Anima Sample](docs/anima_sample.png) | ![SDXL Sample](docs/sdxl_sample.png) |
| *1024x1024, 50 steps, CFG 4.0* | *1024x1024, 8 steps, turbo* | *1024x1024, 30 steps, CFG 4.5* | *1024x1024, 30 steps, CFG 7.5* |

## Performance

**Klein 9B Base (1024x1024, 50 steps, CFG 4.0, 3090 Ti 24GB)**

| Stage | PyTorch (block offload) | Flame (all on GPU) |
|---|---|---|
| Text Encode (Qwen3 8B) | 62s | 20s |
| Model Load | 62s | 19s |
| Denoise (50 steps) | 193s (3.86s/step) | 247s (4.94s/step) |
| VAE Decode | 1.9s | 7.7s |
| **Total** | **322s** | **295s** |

Denoise is **10% faster per-step** than PyTorch. Fits entirely on a single 24GB GPU without block offloading.

## Supported Models

| Model | Architecture | Status |
|---|---|---|
| Klein 4B | Flux 2 DiT (5+20 blocks) | Working |
| Klein 9B | Flux 2 DiT (8+24 blocks) | Working |
| Z-Image | NextDiT (6.15B) | Working |
| SD3.5 | MMDiT | Built, needs TE |
| SDXL | LDM UNet | Working |
| LTX-2 | Video DiT | Ported, needs testing |
| Anima 2B | Cosmos Predict2 DiT | Working |

## Pipeline

Matches [BFL's official reference](https://github.com/black-forest-labs/flux2):

1. **Qwen3 text encoder** (4B or 8B) with half-split RoPE, pad-aware causal mask
2. **Direct velocity Euler sampler** with dynamic mu schedule
3. **Flux2 VAE decoder** with inverse BatchNorm latent denormalization

## Build & Run

```bash
# Build flame-core first
cd flame-core && cargo build --release --lib

# Build inference binaries
cd inference-flame && cargo build --release

# Run Klein 4B (needs ~13GB VRAM)
LD_LIBRARY_PATH=/path/to/cudnn/lib \
  target/release/klein_infer "your prompt here"

# Run Klein 9B (needs ~24GB VRAM, auto-falls back to block offloading)
LD_LIBRARY_PATH=/path/to/cudnn/lib \
  target/release/klein9b_infer "your prompt here"

# Run Z-Image (needs pre-computed text embeddings)
python3 tools/zimage_encode.py --prompt "your prompt" --output embeddings.safetensors
LD_LIBRARY_PATH=/path/to/cudnn/lib \
  target/release/zimage_infer \
    --model /path/to/z_image_turbo_bf16.safetensors \
    --vae /path/to/vae/diffusion_pytorch_model.safetensors \
    --embeddings embeddings.safetensors \
    --output output/zimage_output.png
```

## Checkpoints

| Model | Path | Size |
|---|---|---|
| Klein 4B | `flux-2-klein-base-4b.safetensors` | 7.3GB |
| Klein 9B | `flux-2-klein-base-9b.safetensors` | 17GB |
| Qwen3 4B TE | `qwen_3_4b.safetensors` | 7.5GB |
| Qwen3 8B TE | HuggingFace cache (5 shards) | 15GB |
| Flux2 VAE | `flux2-vae.safetensors` | 321MB |
| Z-Image Turbo | `z_image_turbo_bf16.safetensors` | 12.3GB |
| Z-Image VAE | `zimage_base/vae/diffusion_pytorch_model.safetensors` | 168MB |

## Requirements

- NVIDIA GPU with CUDA 12+ (tested on 3090 Ti)
- cuDNN 9.x
- [flame-core](https://github.com/CodeAlexx/Flame) (linked via Cargo path dependency)

## License

MIT
