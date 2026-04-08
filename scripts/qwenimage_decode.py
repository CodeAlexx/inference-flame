#!/usr/bin/env python
"""Qwen-Image-2512 — Stage 3 (VAE decode → PNG).

Loads the packed latents saved by Stage 2 (Rust `qwenimage_gen`), unpacks
them back to [B, C, F, H, W] geometry, applies the per-channel latent
normalization, and decodes via `AutoencoderKLQwenImage` from diffusers.
Saves a PNG. Only the VAE is on GPU.

Usage:
    python qwenimage_decode.py /path/to/latents.safetensors /path/to/out.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import AutoencoderKLQwenImage

REPO_SNAP = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/snapshots/"
    "25468b98e3276ca6700de15c6628e51b7de54a26"
)
VAE_SCALE_FACTOR = 8  # 2 ** len(temperal_downsample) = 2^3


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Diffusers `_unpack_latents` — reverse of `_pack_latents`.

    Input:  [B, (H/16)*(W/16), 64]   (H, W in pixels)
    Output: [B, 16, 1, H/8, W/8]
    """
    batch_size, num_patches, channels = latents.shape
    # Follow pipeline_qwenimage.py:326-339
    h = 2 * (int(height) // (VAE_SCALE_FACTOR * 2))
    w = 2 * (int(width) // (VAE_SCALE_FACTOR * 2))

    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, h, w)
    return latents


def main() -> int:
    latents_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/alex/serenity/output/qwenimage_latents.safetensors"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else (
        "/home/alex/serenity/output/qwenimage.png"
    )

    device = "cuda"

    print("=== Qwen-Image-2512 — Stage 3 (VAE decode) ===")
    print(f"Latents: {latents_path}")
    print(f"Output:  {out_path}")
    print()

    # ------------------------------------------------------------------
    # Load packed latents + size metadata
    # ------------------------------------------------------------------
    tensors = load_file(latents_path, device=device)
    packed = tensors["packed_latent"]  # [1, seq, 64] BF16
    height = int(tensors["height"].float().item())
    width = int(tensors["width"].float().item())
    print(f"  packed: {tuple(packed.shape)}  size: {width}x{height}")

    latents = unpack_latents(packed, height, width).to(dtype=torch.bfloat16)
    print(f"  unpacked: {tuple(latents.shape)}")

    # ------------------------------------------------------------------
    # Latent denormalization (pipeline_qwenimage.py:741-749)
    # ------------------------------------------------------------------
    snap = Path(REPO_SNAP)
    vae = AutoencoderKLQwenImage.from_pretrained(
        snap / "vae",
        dtype=torch.bfloat16,
    ).to(device)
    vae.eval()

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    # ------------------------------------------------------------------
    # VAE decode
    # ------------------------------------------------------------------
    print("--- VAE decode ---")
    t0 = time.time()
    with torch.no_grad():
        out = vae.decode(latents, return_dict=False)[0]
    # Shape: [B, 3, F=1, H, W] — slice the single frame
    image = out[:, :, 0]
    print(f"  VAE decoded: {tuple(image.shape)} in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Denormalize to uint8
    # ------------------------------------------------------------------
    image = image.float().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    # [B, 3, H, W] → [H, W, 3] (first in batch)
    img_np = image[0].permute(1, 2, 0).cpu().numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(img_np).save(out_path)
    print()
    print("============================================================")
    print(f"IMAGE SAVED: {out_path}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
