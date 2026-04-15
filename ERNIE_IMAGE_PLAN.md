# ERNIE-Image Implementation Plan

## Architecture (from config.json)
- **Transformer**: 36 blocks, hidden=4096, heads=32, head_dim=128, FFN=12288, in/out=128ch, patch_size=1
- **Text encoder**: Mistral-3 (hidden=3072, 26 layers, 32 heads, 8 KV heads GQA)
- **VAE**: AutoencoderKLFlux2 (same as Klein/FLUX) — latent_channels=32, patch=[2,2] → 128ch effective
- **Scheduler**: FlowMatchEulerDiscreteScheduler, shift=3.0 (same as Klein)
- **RoPE**: theta=256, axes_dim=[32,48,48], 3-axis (text_offset, height, width)

## Key differences from Klein
1. Text encoder is Mistral-3 (not Qwen3)
2. Single-stream attention with shared AdaLN (all blocks share same modulation from timestep)
3. 36 blocks vs Klein's 25 (4B) or 57 (9B)
4. hidden=4096 (between Klein 4B's 3072 and 9B's 4608)
5. RoPE theta=256 (not 10000)
6. No guidance embedding

## Files to create/modify in inference-flame
1. `src/models/ernie_image.rs` — rewrite with correct dims (existing has wrong config)
2. `src/models/mistral_encoder.rs` — check if already exists, port Mistral-3 text encoder
3. VAE: reuse KleinVaeEncoder/Decoder (same FLUX-2 VAE)

## Files to create in flame-diffusion
1. `ernie-trainer/` — new trainer crate following klein-trainer pattern
2. `ernie-trainer/src/facilitator.rs` — BlockFacilitator for `layers.{i}.*`
3. `ernie-trainer/src/model.rs` — training model with BlockOffloader
4. `ernie-trainer/src/bin/prepare_dataset.rs` — cache latents + text embeddings
5. `ernie-trainer/src/bin/ernie_lora_train.rs` — training binary

## Weights location
- `/home/alex/models/ERNIE-Image/transformer/` — 2 shards, 16GB total
- `/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors`
- `/home/alex/models/ERNIE-Image/text_encoder/` — Mistral-3
- `/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json`

## Training approach
- Same as Klein: flow matching, target = noise - latents
- VAE encode with BN normalization (same pattern as Klein)
- Text: Mistral-3 encode, use hidden_states[-2] (second-to-last layer)
- Block offloading via BlockOffloader for 24GB cards
