#!/usr/bin/env python
"""Wan2.2 VAE encoder parity test — Python reference.

Loads the same random input that the Rust parity binary generated,
encodes it with the Python Wan2.2 VAE, and compares the output.

Usage:
    # First run the Rust binary to generate the input:
    #   cargo run --release --bin wan22_vae_encode_parity
    # Then run this:
    python scripts/wan22_vae_encode_parity.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.cuda.amp as amp
import numpy as np
from safetensors.torch import save_file, load_file

# Paths
VAE_PTH = os.environ.get(
    "WAN_VAE_PTH",
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
)
OUTPUT_DIR = "/home/alex/EriDiffusion/inference-flame/output"
INPUT_PATH = f"{OUTPUT_DIR}/wan22_vae_encode_input.safetensors"
RUST_OUTPUT_PATH = f"{OUTPUT_DIR}/wan22_vae_encode_output.safetensors"
PY_OUTPUT_PATH = f"{OUTPUT_DIR}/wan22_vae_encode_output_python.safetensors"

# Add Wan2.2 to path
sys.path.insert(0, "/home/alex/Wan2.2")
from wan.modules.vae2_2 import Wan2_2_VAE


def stats(name: str, t: torch.Tensor):
    t_f = t.float()
    print(
        f"  [{name}] shape={tuple(t.shape)} "
        f"mean={t_f.mean():.4f} |mean|={t_f.abs().mean():.4f} "
        f"range=[{t_f.min():.4f}, {t_f.max():.4f}] "
        f"nan={t_f.isnan().sum().item()} inf={t_f.isinf().sum().item()}"
    )


def main() -> int:
    device = "cuda"
    dtype = torch.bfloat16

    print("=== Wan2.2 VAE Encoder Parity Test (Python) ===\n")

    # --- Load input from Rust ---
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("Run the Rust binary first: cargo run --release --bin wan22_vae_encode_parity")
        return 1

    loaded = load_file(INPUT_PATH, device="cpu")
    video = loaded["video"].to(device=device, dtype=dtype)
    stats("input video", video)
    print()

    # --- Load VAE ---
    print("--- Loading Wan2.2 VAE ---")
    t0 = time.time()
    vae = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=VAE_PTH,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=dtype,
        device=device,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    # --- Encode ---
    print("--- Encoding ---")
    t1 = time.time()
    with torch.no_grad(), amp.autocast(dtype=dtype):
        # Use the model's encode method which handles patchify + normalize
        latent = vae.model.encode(video, vae.scale).float()
    print(f"  Encode: {time.time() - t1:.1f}s")
    stats("output latent", latent)

    # Save Python output
    save_file({"latent": latent.cpu()}, PY_OUTPUT_PATH)
    print(f"  Saved to {PY_OUTPUT_PATH}\n")

    # --- Compare with Rust if available ---
    if os.path.exists(RUST_OUTPUT_PATH):
        print("--- Comparing Rust vs Python ---")
        rust_data = load_file(RUST_OUTPUT_PATH, device="cpu")
        rust_latent = rust_data["latent"]
        py_latent = latent.cpu().float()

        stats("rust latent", rust_latent)
        stats("python latent", py_latent)

        diff = (rust_latent - py_latent).abs()
        print(f"\n  Absolute diff: mean={diff.mean():.6f} max={diff.max():.6f}")
        print(f"  Relative diff: mean={(diff / (py_latent.abs() + 1e-8)).mean():.6f}")

        # Cosine similarity per-sample
        rust_flat = rust_latent.reshape(1, -1)
        py_flat = py_latent.reshape(1, -1)
        cos_sim = torch.nn.functional.cosine_similarity(rust_flat, py_flat)
        print(f"  Cosine similarity: {cos_sim.item():.6f}")

        if diff.max() < 0.1:
            print("\n  PASS: Outputs are close (max diff < 0.1)")
        elif diff.max() < 1.0:
            print("\n  WARN: Outputs have moderate differences (max diff < 1.0)")
        else:
            print("\n  FAIL: Large differences detected")
    else:
        print(f"Rust output not found at {RUST_OUTPUT_PATH}")
        print("Run Rust binary first for comparison.")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
