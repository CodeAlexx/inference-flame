#!/usr/bin/env python
"""QwenImage VAE decode reference dumper.

Takes a raw latent produced by `AutoencoderKLQwenImage.encode().latent_dist.sample()`
(what `/home/alex/datasets/boxjana_cached/*.safetensors` contains under key
`latent`) and saves the Python VAE's decode output to
`/home/alex/EriDiffusion/inference-flame/output/qwenimage_vae_decode_ref.safetensors`
for bit-level comparison against the Rust decoder.

Usage:
    python scripts/qwenimage_vae_decode_ref.py [<cached_latent.safetensors>]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import load_file, save_file

VAE_PATH = Path(
    "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors"
)
VAE_DIR = VAE_PATH.parent
CACHE = Path("/home/alex/datasets/boxjana_cached/10.safetensors")
OUT = Path("/home/alex/EriDiffusion/inference-flame/output/qwenimage_vae_decode_ref.safetensors")


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE
    print(f"[ref] loading latent from {src}")
    blob = load_file(src, device="cuda")
    # The musubi cache stores *normalized* latents — produced by
    # `encode_pixels_to_latents` which applies `(raw - mean) * inv_std`.
    # Shape is [1, 16, H/8, W/8] (T squeezed out).
    norm = blob["latent"].to(torch.bfloat16)
    assert norm.ndim == 4 and norm.shape[1] == 16, f"unexpected shape {tuple(norm.shape)}"
    norm5 = norm.unsqueeze(2)  # [1, 16, 1, H/8, W/8]

    print(f"[ref] loading VAE from {VAE_DIR}")
    vae = AutoencoderKLQwenImage.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16).to("cuda")
    vae.eval()

    # Unnormalize: diffusers VAE.decode wants raw latents (pre-normalization).
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, vae.config.z_dim, 1, 1, 1)
    mean = mean.to(norm5.device, norm5.dtype)
    inv_std = inv_std.to(norm5.device, norm5.dtype)
    raw5 = norm5 / inv_std + mean  # == norm5 * std + mean

    with torch.no_grad():
        dec = vae.decode(raw5, return_dict=False)[0]  # [B, 3, F=1, H, W]
    print(f"[ref] decoded: {tuple(dec.shape)} {dec.dtype}")

    # Save the *normalized* latent (what the Rust decoder expects) plus
    # the Python reference RGB for bit-level comparison.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"normalized_latent": norm5.contiguous(), "rgb": dec.contiguous()}, OUT)
    print(f"[ref] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
