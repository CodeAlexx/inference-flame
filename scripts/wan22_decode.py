#!/usr/bin/env python
"""Wan2.2-T2V-A14B — Stage 3 (VAE decode → MP4).

Loads the latent tensor saved by Stage 2 (Rust denoise), unpacks from the
patched layout back to [C, F, H, W], decodes via Wan2.1 VAE, and saves video.

Usage:
    python wan22_decode.py /path/to/latents.safetensors /path/to/output.mp4
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

# Add Wan2.2 source to path — load module directly to avoid librosa dependency
WAN_ROOT = "/home/alex/Wan2.2"
sys.path.insert(0, WAN_ROOT)

import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_vae_mod = _load_module("wan.modules.vae2_1", f"{WAN_ROOT}/wan/modules/vae2_1.py")
Wan2_1_VAE = _vae_mod.Wan2_1_VAE

VAE_PATH = "/home/alex/.serenity/models/upscalers/Wan2.1_VAE.safetensors"
SAMPLE_FPS = 16


def save_video_mp4(frames: np.ndarray, path: str, fps: int = 16):
    """Save [F, H, W, 3] uint8 array as MP4 using ffmpeg."""
    import subprocess
    import tempfile

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
        "/home/alex/serenity/output/wan22_latents.safetensors"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else (
        "/home/alex/serenity/output/wan22_output.mp4"
    )

    device = "cuda"

    print("=== Wan2.2-T2V-A14B — Stage 3 (VAE decode) ===")
    print(f"Latents: {latents_path}")
    print(f"Output:  {out_path}")
    print()

    # ------------------------------------------------------------------
    # Load latent
    # ------------------------------------------------------------------
    tensors = load_file(latents_path, device=device)
    latent = tensors["latent"]  # [C, F, H, W] BF16
    print(f"  latent: {tuple(latent.shape)} {latent.dtype}")

    # ------------------------------------------------------------------
    # Load VAE
    # ------------------------------------------------------------------
    print("--- Loading Wan2.1 VAE ---")
    t0 = time.time()

    # Build VAE model on meta device, then load safetensors weights
    WanVAE_ = _vae_mod.WanVAE_
    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )
    with torch.device("meta"):
        vae_model = WanVAE_(**cfg)

    state_dict = load_file(VAE_PATH, device=device)
    vae_model.load_state_dict(state_dict, assign=True)
    vae_model = vae_model.eval().requires_grad_(False).to(device)

    # Build the Wan2_1_VAE wrapper manually (avoids torch.load)
    vae = type("VAEWrapper", (), {})()
    vae.model = vae_model
    vae.dtype = torch.float
    vae.device = device
    mean = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]
    std = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]
    vae.mean = torch.tensor(mean, dtype=torch.float, device=device)
    vae.std = torch.tensor(std, dtype=torch.float, device=device)
    vae.scale = [vae.mean, 1.0 / vae.std]

    def decode_fn(zs):
        with torch.amp.autocast("cuda", dtype=torch.float):
            return [
                vae.model.decode(u.unsqueeze(0), vae.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
    vae.decode = decode_fn
    print(f"  VAE loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    print("--- VAE decode ---")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        video = vae.decode([latent])  # returns list of [C, F, H, W] tensors
    video = video[0]  # [C, F, H, W] float
    print(f"  Decoded: {tuple(video.shape)} in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Convert to uint8 and save
    # ------------------------------------------------------------------
    # video is in [-1, 1] range
    video = video.float().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    # [C, F, H, W] → [F, H, W, C]
    frames = video.permute(1, 2, 3, 0).cpu().numpy()
    print(f"  Frames: {frames.shape} (F, H, W, C)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_video_mp4(frames, out_path, fps=SAMPLE_FPS)

    print()
    print("============================================================")
    print(f"VIDEO SAVED: {out_path}")
    print(f"  {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}, {SAMPLE_FPS} fps")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
