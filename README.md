# inference-flame

Pure Rust diffusion model inference using [flame-core](https://github.com/CodeAlexx/Flame). No Python, no diffusers, no ONNX.

| Klein 9B | Z-Image | Anima 2B |
|---|---|---|
| ![Klein 9B](docs/klein9b_sample.png) | ![Z-Image](docs/zimage_sample.png) | ![Anima 2B](docs/anima_sample.png) |
| *50 steps, CFG 4.0* | *8 steps, turbo* | *30 steps, CFG 4.5* |

| SDXL | Chroma 8.9B | QwenImage-2512 |
|---|---|---|
| ![SDXL](docs/sdxl_sample.png) | ![Chroma](docs/chroma_sample.png) | ![QwenImage-2512](docs/qwenimage_sample.png) |
| *30 steps, CFG 7.5* | *40 steps, CFG 4.0* | *30 steps, CFG 4.0, 1024²* |

| SD3.5 Medium | ERNIE-Image 8B | LTX-2.3 Video |
|---|---|---|
| ![SD3.5 Medium](docs/sd3_medium_sample.png) | ![ERNIE-Image](docs/ernie_image_sample.png) | ![LTX-2.3](docs/ltx2_sample.png) |
| *28 steps, CFG 4.5, 1024²* | *28 steps, CFG 4.0, 1024²* | *10s video + audio — [sample.mp4](docs/ltx2_sample.mp4)* |

https://github.com/CodeAlexx/inference-flame/raw/master/docs/ltx2_sample.mp4

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
| Chroma 8.9B | FLUX-schnell DiT + distilled guidance (19+38 blocks) | Working — real CFG, 1024²/40 on a 24 GB card via BlockOffloader |
| SD3.5 Medium | MMDiT (24 blocks, dual attention) | Working — full prompt-to-PNG, CLIP-L + CLIP-G + T5-XXL, 1024² resident |
| SD3.5 Large | MMDiT (38 blocks) | Built, needs full pipeline |
| SDXL | LDM UNet | Working |
| QwenImage-2512 | 60-layer DiT + 3D VAE (Qwen2.5-VL-7B text encoder) | Working — 1024²/30, true CFG with norm rescale, 3-axis RoPE, BlockOffloader |
| ERNIE-Image 8B | 36-layer single-stream DiT (Mistral-3 3B text encoder) | Working — 1024²/28, sequential CFG, fused RoPE kernel, ~98s on 3090 Ti |
| LTX-2.3 | Video DiT + 3D VAE + BigVGAN vocoder | **World's first pure-Rust video pipeline.** Video working end-to-end (prompt → MP4). Audio path runs but still has artifacts — needs more work. |
| Motif-Video 2B | 12 dual + 24 single DiT (T5Gemma2 text encoder, Wan 2.1 VAE) | Working end-to-end (prompt → MP4). 1280×720×49 @ 24fps, APG with norm-threshold clipping + momentum EMA matching reference. VAE decode currently via Python bridge (Rust `Wan21VaeDecoder` uses different safetensors key layout than diffusers-style motif checkpoint — `MOTIF_HANDOFF.md` has the diff). |
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
