#!/usr/bin/env python
"""LTX-2.3 I2V — encode reference image + text embeddings.

Encodes a reference image via the LTX-2.3 VAE (single-frame encode),
loads pre-cached Gemma text embeddings, and saves everything for the
Rust I2V binary.

Usage:
    python ltx2_i2v_encode.py <image_path> [text_embeddings.safetensors] [output.safetensors]

Output tensors (BF16):
    text_hidden:   [1, L, 4096]   — Gemma text embeddings (copied from cached)
    image_latent:  [1, 128, 1, H_lat, W_lat] — VAE-encoded reference image

Requires:
    - LTX-2.3 VAE weights at VAE_PATH
    - Pre-cached text embeddings (from the T2V encode pipeline)
    - diffusers (for VAE loading)
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file

# Paths
VAE_PATH = "/home/alex/.serenity/models/checkpoints/ltx2-diffusers/vae"
DEFAULT_TEXT_EMBEDS = (
    "/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors"
)
DEFAULT_OUTPUT = (
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_i2v_embeds.safetensors"
)

# LTX-2.3 latent geometry
SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 8
LATENT_CHANNELS = 128

# Target resolution (must match the Rust binary's WIDTH/HEIGHT)
TARGET_W = 480
TARGET_H = 288


def load_vae(vae_path: str, device: str, dtype: torch.dtype):
    """Load LTX-2 VAE from diffusers directory format."""
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        vae_path, torch_dtype=dtype
    )
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    return vae


def encode_image(
    vae,
    image_path: str,
    target_h: int,
    target_w: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a single image into LTX-2 latent space.

    Returns: [1, 128, 1, H_lat, W_lat] tensor in BF16.
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    print(f"  Original size: {img.size}")

    # Convert to tensor [C, H, W] in [0, 1]
    import numpy as np

    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # Resize to target resolution
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    )  # [1, 3, H, W]

    # Normalize to [-1, 1] (standard VAE input range)
    img_tensor = (img_tensor * 2.0) - 1.0
    img_tensor = img_tensor.to(device=device, dtype=dtype)

    # Add temporal dimension: [1, 3, H, W] → [1, 3, 1, H, W]
    video_input = img_tensor.unsqueeze(2)
    print(f"  VAE input: {tuple(video_input.shape)} {video_input.dtype}")

    # Encode
    t0 = time.time()
    with torch.no_grad():
        encoder_output = vae.encode(video_input, causal=True)
        latent = encoder_output.latent_dist.sample()
    print(f"  VAE encoded in {time.time() - t0:.1f}s")
    print(f"  Latent shape: {tuple(latent.shape)} {latent.dtype}")

    # LTX-2 normalization is identity (mean=0, std=1, scale=1)
    # No normalization needed.

    return latent.to(dtype=torch.bfloat16)


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    text_embeds_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEXT_EMBEDS
    output_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT

    if image_path is None:
        print("Usage: python ltx2_i2v_encode.py <image_path> [text_embeds.safetensors] [output.safetensors]")
        print()
        print("  image_path:    Reference image for I2V")
        print(f"  text_embeds:   Pre-cached Gemma embeddings (default: {DEFAULT_TEXT_EMBEDS})")
        print(f"  output:        Output path (default: {DEFAULT_OUTPUT})")
        return 1

    device = "cuda"
    dtype = torch.bfloat16

    print("=== LTX-2.3 I2V Encode (image + text) ===")
    print(f"  Image:       {image_path}")
    print(f"  Text embeds: {text_embeds_path}")
    print(f"  Output:      {output_path}")
    print(f"  Target:      {TARGET_W}x{TARGET_H}")
    print()

    # 1. Load pre-cached text embeddings
    print("--- Loading text embeddings ---")
    text_data = load_file(text_embeds_path, device="cpu")
    text_hidden = text_data["text_hidden"]
    print(f"  text_hidden: {tuple(text_hidden.shape)} {text_hidden.dtype}")

    # 2. Load VAE and encode image
    print("\n--- Loading LTX-2.3 VAE ---")
    t0 = time.time()
    vae = load_vae(VAE_PATH, device, dtype)
    print(f"  VAE loaded in {time.time() - t0:.1f}s")

    print("\n--- Encoding reference image ---")
    image_latent = encode_image(vae, image_path, TARGET_H, TARGET_W, device, dtype)

    del vae
    torch.cuda.empty_cache()

    # 3. Save
    print("\n--- Saving ---")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tensors = {
        "text_hidden": text_hidden.to(dtype=torch.bfloat16).contiguous().cpu(),
        "image_latent": image_latent.contiguous().cpu(),
    }
    save_file(tensors, output_path)
    print(f"  Saved to {output_path}")
    for k, v in tensors.items():
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")

    print(f"\nNext: ./target/release/ltx2_i2v_gen {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
