#!/usr/bin/env python
"""HunyuanVideo 1.5 — Stage 3 (VAE decode -> MP4).

Loads the latent tensor saved by Stage 2 (Rust denoise), decodes via
HunyuanVideo VAE, and saves video as MP4.

Usage:
    python hunyuan15_decode.py /path/to/latents.safetensors /path/to/output.mp4
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAE_PATH = "/home/alex/.serenity/models/vaes/hunyuan_video_vae_bf16.safetensors"
VAE_LATENT_CHANNELS = 16
SCALING_FACTOR = 0.476986
SAMPLE_FPS = 24


def save_video_mp4(frames: np.ndarray, path: str, fps: int = 24):
    """Save [F, H, W, 3] uint8 array as MP4 using ffmpeg."""
    import subprocess

    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate(input=frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")


def main() -> int:
    latents_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/alex/serenity/output/hunyuan15_latents.safetensors"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else (
        "/home/alex/serenity/output/hunyuan15_output.mp4"
    )

    device = "cuda"

    print("=== HunyuanVideo 1.5 — Stage 3 (VAE decode) ===")
    print(f"Latents: {latents_path}")
    print(f"Output:  {out_path}")
    print()

    # ------------------------------------------------------------------
    # Load latent
    # ------------------------------------------------------------------
    tensors = load_file(latents_path, device=device)
    latent = tensors["latent"]  # [B, C, T, H, W]
    print(f"  latent: {tuple(latent.shape)} {latent.dtype}")

    # HunyuanVideo 1.5 transformer outputs 32 channels, but the VAE
    # only has 16 latent channels.  Take the first 16 if needed.
    if latent.shape[1] > VAE_LATENT_CHANNELS:
        print(f"  Trimming latent channels {latent.shape[1]} -> {VAE_LATENT_CHANNELS}")
        latent = latent[:, :VAE_LATENT_CHANNELS]

    # Undo the scaling applied during encoding / denoising
    latent = latent / SCALING_FACTOR

    # ------------------------------------------------------------------
    # Load VAE  (diffusers AutoencoderKLHunyuanVideo)
    # ------------------------------------------------------------------
    print("--- Loading HunyuanVideo VAE ---")
    t0 = time.time()

    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
        AutoencoderKLHunyuanVideo,
    )

    vae = AutoencoderKLHunyuanVideo(
        in_channels=3,
        out_channels=3,
        latent_channels=VAE_LATENT_CHANNELS,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        spatial_compression_ratio=8,
        temporal_compression_ratio=4,
    )

    state_dict = load_file(VAE_PATH, device=device)
    vae.load_state_dict(state_dict)
    vae = vae.eval().requires_grad_(False).to(device)
    print(f"  VAE loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    print("--- VAE decode ---")
    t0 = time.time()

    # The VAE's post_quant_conv + decoder expect float precision for
    # large temporal sequences.  Use autocast for memory efficiency.
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # latent goes through post_quant_conv then decoder
        decoded = vae.decode(latent, return_dict=False)[0]  # [B, C, T, H, W]

    video = decoded[0]  # [C, T, H, W]
    print(f"  Decoded: {tuple(video.shape)} in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Convert to uint8 and save
    # ------------------------------------------------------------------
    # VAE output is in [-1, 1] range
    video = video.float().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    # [C, T, H, W] -> [T, H, W, C]
    frames = video.permute(1, 2, 3, 0).cpu().numpy()
    print(f"  Frames: {frames.shape} (T, H, W, C)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_video_mp4(frames, out_path, fps=SAMPLE_FPS)

    print()
    print("============================================================")
    print(f"VIDEO SAVED: {out_path}")
    print(f"  {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}, {SAMPLE_FPS} fps")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
