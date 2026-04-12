#!/usr/bin/env python
"""LDM VAE encoder parity test — Python reference for comparison with Rust.

Usage:
    python ldm_vae_encode_parity.py [vae_model_id_or_path] [latent_channels]

Defaults to SDXL VAE (4 latent channels). Loads the same deterministic input
that the Rust binary generates, or creates it if not found.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from diffusers import AutoencoderKL

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
DEFAULT_VAE = "stabilityai/sdxl-vae"


def stats(name: str, t: torch.Tensor):
    t_f = t.float()
    print(f"  {name:<28} shape={tuple(t.shape)} "
          f"mean={t_f.mean().item():+.6f} std={t_f.std().item():.6f} "
          f"min={t_f.min().item():.6f} max={t_f.max().item():.6f}")


def main():
    vae_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAE
    latent_ch = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print("=== LDM VAE Encoder Parity Test (Python) ===")
    print(f"  VAE: {vae_id}")
    print(f"  Latent channels: {latent_ch}")
    print()

    device = "cuda"
    dtype = torch.bfloat16

    # Try to load the Rust-generated input for exact comparison
    input_path = os.path.join(OUTPUT_DIR, "ldm_vae_encode_input.safetensors")
    if os.path.exists(input_path):
        loaded = load_file(input_path)
        image = loaded["image"].to(device=device, dtype=dtype).unsqueeze(0) \
            if loaded["image"].dim() == 3 else loaded["image"].to(device=device, dtype=dtype)
        # The Rust binary saves [1, 3, 512, 512] as F32
        if image.dim() == 4 and image.shape[0] == 1:
            pass  # already [1, 3, H, W]
        print(f"  Loaded input from {input_path}")
    else:
        # Generate same deterministic input as Rust
        h, w = 512, 512
        n = 3 * h * w
        pixel_data = torch.linspace(-1.0, 1.0, n, dtype=torch.float32)
        # Match Rust: (i / n) * 2 - 1 for i in 0..n
        pixel_data = torch.tensor(
            [(i / n) * 2.0 - 1.0 for i in range(n)], dtype=torch.float32
        )
        image = pixel_data.reshape(1, 3, h, w).to(device=device, dtype=dtype)
        print("  Generated deterministic input (no Rust input file found)")

    stats("input image", image)

    # Load VAE
    print(f"  Loading VAE: {vae_id}")
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype).to(device).eval()
    print(f"  VAE loaded")

    # Encode
    with torch.no_grad():
        # Full encoder forward
        h = vae.encoder(image)

        # quant_conv if present
        if hasattr(vae, "quant_conv") and vae.quant_conv is not None:
            h = vae.quant_conv(h)
            print("  quant_conv applied")

        stats("encoder output (mean+logvar)", h)

        # Take mean (first latent_ch channels)
        latent = h[:, :latent_ch]
        stats("output latent (mean only)", latent)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ldm_vae_encode_python.safetensors")
    save_file({"latent": latent.float().cpu()}, out_path)
    print(f"  Saved output to {out_path}")

    # Compare with Rust output if available
    rust_path = os.path.join(OUTPUT_DIR, "ldm_vae_encode_rust.safetensors")
    if os.path.exists(rust_path):
        rust_data = load_file(rust_path)
        rust_latent = rust_data["latent"].to(device=device, dtype=torch.float32)
        py_latent = latent.float()

        diff = (rust_latent - py_latent).abs()
        print(f"\n  === Comparison ===")
        print(f"  Max abs diff:  {diff.max().item():.6f}")
        print(f"  Mean abs diff: {diff.mean().item():.6f}")
        print(f"  Rust first 8:  {rust_latent.flatten()[:8].tolist()}")
        print(f"  Python first 8: {py_latent.flatten()[:8].tolist()}")

        if diff.max().item() < 0.05:
            print("  PASS: Rust and Python outputs match within tolerance")
        else:
            print("  WARNING: Large difference detected — investigate")
    else:
        print(f"\n  No Rust output found at {rust_path} — run the Rust binary first")

    print("\nDone.")


if __name__ == "__main__":
    main()
