# LTX-2 Audio Generation Audit

## Checkpoint Audio Weights

**2,896 audio-related keys** (13.16 GB) in the distilled checkpoint:
- `audio_vae.encoder`: 44 keys
- `audio_vae.decoder`: 56 keys  
- `audio_vae.per_channel_statistics`: 2 keys
- `model.diffusion_model.*audio*`: 2,792 keys (dual-stream transformer)
- `text_embedding_projection.audio_aggregate_embed`: 2 keys

## Architecture

- **Dual-stream transformer**: Video (4096-dim) + Audio (2048-dim) with cross-attention
- **Audio latent**: 128 channels, 4x scale factor, 16kHz sample rate, 160 hop length
- **48 blocks** each have: audio self-attn, audio cross-attn, audio↔video cross-attn, audio FFN
- **RoPE**: 1D for audio (vs 3D for video)

## What FLAME Loads vs Skips

| Component | Loaded | Skipped |
|-----------|--------|---------|
| Video stream globals | ✅ | |
| Video transformer blocks | ✅ | |
| Audio stream globals | | ❌ |
| Audio transformer blocks | | ❌ |
| Audio-to-Video cross-attn | ✅ (weights loaded, not used in video-only) | |
| Video-to-Audio cross-attn | | ❌ |
| Audio aggregate embed | | ❌ |
| Audio VAE | | ❌ |
| Vocoder | | ❌ |

Filter in `load_globals`: `!k.contains("audio")`
Filter in FlameSwap: `if k.contains("audio") && !k.contains("audio_to_video") { return None; }`

## Audio Pipeline (Official Python)

```
Waveform → AudioProcessor (STFT+Mel) → AudioEncoder → Audio Latent
                                                         ↓
                            Dual-stream denoise (48 blocks, audio+video)
                                                         ↓
Audio Latent → AudioDecoder → Mel Spectrogram → Vocoder (BigVGAN) → Waveform (24kHz)
```

## Files Available Locally

- ✅ Audio VAE: `/home/alex/.serenity/models/checkpoints/ltx2-diffusers/audio_vae/` (102 MB)
- ✅ Vocoder: `/home/alex/.serenity/models/checkpoints/ltx2-diffusers/vocoder/` (111 MB)
- ✅ All audio weights embedded in main checkpoint (13.16 GB)
- ✅ Official Python pipeline: `ltx-pipelines/a2vid_two_stage.py`

## What's Needed for Audio in FLAME

1. Load audio globals (time_embed, proj_in/out, scale_shift_table)
2. Load audio block weights in parallel with video blocks (+13GB)
3. Implement audio VAE encoder/decoder (102 MB model)
4. Implement BigVGAN vocoder (111 MB model)
5. Wire dual-stream forward: both video AND audio through all 48 blocks
6. Audio conditioning: encode input audio or generate from text
