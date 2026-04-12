#!/usr/bin/env python
"""LTX-2.3 Video VAE Encoder parity test — standalone (no diffusers).

Loads encoder weights from the full checkpoint, builds a forward pass
mirroring the Rust encoder, and compares outputs.

Usage:
    python scripts/ltx2_vae_encode_parity.py
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

CHECKPOINT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors"
OUTPUT_DIR = "/home/alex/EriDiffusion/inference-flame/output"
INPUT_PATH = f"{OUTPUT_DIR}/ltx2_vae_encode_input.safetensors"
RUST_OUTPUT = f"{OUTPUT_DIR}/ltx2_vae_encode_output.safetensors"
PY_OUTPUT = f"{OUTPUT_DIR}/ltx2_vae_encode_python.safetensors"
EPS = 1e-6

def stats(name, t):
    f = t.float()
    print(f"  {name:<28} shape={tuple(t.shape)} mean={f.mean():.6f} std={f.std():.6f} min={f.min():.6f} max={f.max():.6f}")


def pixel_norm(x, eps=1e-6):
    return x / (x.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()


def space_to_depth(x, stride):
    b, c, f, h, w = x.shape
    st, sh, sw = stride
    x = x.reshape(b, c, f//st, st, h//sh, sh, w//sw, sw)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(b, c*st*sh*sw, f//st, h//sh, w//sw)


def patchify(x):
    """[B,3,T,H,W] -> [B,48,T,H/4,W/4]"""
    b, c, t, h, w = x.shape
    x = x.reshape(b, c, t, h//4, 4, w//4, 4)
    x = x.permute(0, 1, 4, 6, 2, 3, 5)
    return x.reshape(b, c*16, t, h//4, w//4)


class CausalConv3d(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        kt, kh, kw = weight.shape[2:]
        self.time_pad = kt - 1
        self.spatial_pad = (kh // 2, kw // 2)

    def forward(self, x):
        if self.time_pad > 0:
            first = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([first, x], dim=2)
        return F.conv3d(x, self.weight, self.bias,
                        padding=(0, self.spatial_pad[0], self.spatial_pad[1]))


def load_conv(w, prefix):
    weight = w[f"{prefix}.weight"]
    bias = w.get(f"{prefix}.bias")
    return CausalConv3d(weight, bias)


def resnet_forward(x, conv1, conv2, shortcut=None):
    h = pixel_norm(x)
    h = F.silu(h)
    h = conv1(h)
    h = pixel_norm(h)
    h = F.silu(h)
    h = conv2(h)
    if shortcut is not None:
        x = shortcut(x)
    return x + h


def downsample_forward(x, stride, conv, group_size):
    st, sh, sw = stride
    if st > 1:
        first = x[:, :, :1].expand(-1, -1, st-1, -1, -1)
        x_pad = torch.cat([first, x], dim=2)
    else:
        x_pad = x

    # Residual
    res = space_to_depth(x_pad, stride)
    b, c, f, h, w_dim = res.shape
    groups = c // group_size
    res = res.reshape(b, groups, group_size, f, h, w_dim).mean(dim=2)

    # Main
    main = conv(x_pad)
    main = space_to_depth(main, stride)

    return main + res


def main():
    device = "cuda"
    dtype = torch.bfloat16

    print("=== LTX-2.3 Video VAE Encoder Parity (Python) ===\n")

    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}\nRun Rust binary first.")
        return 1

    data = load_file(INPUT_PATH, device="cpu")
    video = data["input"].to(device=device, dtype=dtype)
    stats("input", video)

    # Load weights
    print("\n--- Loading weights ---")
    t0 = time.time()
    all_w = load_file(CHECKPOINT, device="cpu")
    # Strip vae. prefix, keep encoder + stats keys
    w = {}
    for k, v in all_w.items():
        for pfx in ["vae.", "video_vae.", ""]:
            if k.startswith(pfx) and len(pfx) <= len(k):
                stripped = k[len(pfx):]
                if stripped.startswith("encoder.") or stripped.startswith("per_channel_statistics."):
                    w[stripped] = v.to(device=device, dtype=dtype)
    print(f"  {len(w)} encoder keys in {time.time()-t0:.1f}s")

    # Block schedule from Rust: 9 blocks, alternating mid/down
    # Block 0: mid  128->128, 4 resnets  (down_blocks.0)
    # Block 1: down stride(1,2,2) 128->256 (down_blocks.1)
    # Block 2: mid  256->256, 6 resnets  (down_blocks.2)
    # Block 3: down stride(2,1,1) 256->512 (down_blocks.3)
    # Block 4: mid  512->512, 4 resnets  (down_blocks.4)
    # Block 5: down stride(2,2,2) 512->1024 (down_blocks.5)
    # Block 6: mid  1024->1024, 2 resnets (down_blocks.6)
    # Block 7: down stride(2,2,2) 1024->1024 (down_blocks.7)
    # Block 8: mid  1024->1024, 2 resnets (down_blocks.8)

    BLOCKS = [
        ("mid",  0, 128,  128,  4),
        ("down", 1, (1,2,2), 128, 256, 64),   # conv_out_ch=64, group_size=128*4/256=2
        ("mid",  2, 256,  256,  6),
        ("down", 3, (2,1,1), 256, 512, 256),   # conv_out_ch=256, group_size=256*2/512=1
        ("mid",  4, 512,  512,  4),
        ("down", 5, (2,2,2), 512, 1024, 128),  # conv_out_ch=128, group_size=512*8/1024=4
        ("mid",  6, 1024, 1024, 2),
        ("down", 7, (2,2,2), 1024, 1024, 128), # conv_out_ch=128, group_size=1024*8/1024=8
        ("mid",  8, 1024, 1024, 2),
    ]

    # Forward pass
    print("\n--- Encoding ---")
    t1 = time.time()
    with torch.no_grad():
        x = patchify(video)
        stats("after patchify", x)

        # conv_in
        conv_in = load_conv(w, "encoder.conv_in.conv")
        x = conv_in(x)
        stats("after conv_in", x)

        # Blocks
        for spec in BLOCKS:
            if spec[0] == "mid":
                _, bi, in_ch, out_ch, n_res = spec
                for j in range(n_res):
                    pfx = f"encoder.down_blocks.{bi}.res_blocks.{j}"
                    c1 = load_conv(w, f"{pfx}.conv1.conv")
                    c2 = load_conv(w, f"{pfx}.conv2.conv")
                    sc_key = f"{pfx}.conv_shortcut.conv.weight"
                    sc = load_conv(w, f"{pfx}.conv_shortcut.conv") if sc_key in w else None
                    x = resnet_forward(x, c1, c2, sc)
                stats(f"after block {bi} (mid {in_ch}x{n_res})", x)
            else:
                _, bi, stride, in_ch, out_ch, conv_out_ch = spec
                conv = load_conv(w, f"encoder.down_blocks.{bi}.conv.conv")
                group_size = (in_ch * stride[0]*stride[1]*stride[2]) // out_ch
                x = downsample_forward(x, stride, conv, group_size)
                stats(f"after block {bi} (down {stride})", x)

        # conv_out
        x = pixel_norm(x, eps=1e-8)
        x = F.silu(x)
        conv_out = load_conv(w, "encoder.conv_out.conv")
        x = conv_out(x)
        stats("after conv_out (129ch)", x)

        # 129 -> 256 expansion, take first 128
        last = x[:, -1:].expand(-1, 127, -1, -1, -1)
        x = torch.cat([x, last], dim=1)
        mu = x[:, :128]

        # Per-channel normalize
        mean = w["per_channel_statistics.mean-of-means"].reshape(1, 128, 1, 1, 1)
        std = w["per_channel_statistics.std-of-means"].reshape(1, 128, 1, 1, 1)
        latent = (mu - mean) / std

    print(f"  Encode: {time.time()-t1:.2f}s")
    stats("latent (normalized)", latent)

    # Save
    save_file({"latent": latent.float().cpu()}, PY_OUTPUT)
    print(f"  Saved to {PY_OUTPUT}")

    # Compare
    if os.path.exists(RUST_OUTPUT):
        print("\n--- Comparing ---")
        rust = load_file(RUST_OUTPUT)
        rust_lat = rust["latent"]
        py_lat = latent.float().cpu()

        diff = (rust_lat - py_lat).abs()
        cos = F.cosine_similarity(rust_lat.flatten().unsqueeze(0),
                                   py_lat.flatten().unsqueeze(0))
        print(f"  Max diff:   {diff.max():.6f}")
        print(f"  Mean diff:  {diff.mean():.6f}")
        print(f"  Cosine sim: {cos.item():.6f}")
        print(f"  Rust first 5:  {rust_lat.flatten()[:5].tolist()}")
        print(f"  Python first 5: {py_lat.flatten()[:5].tolist()}")

        if diff.max() < 0.5:
            print("  PASS")
        elif diff.max() < 2.0:
            print("  WARN")
        else:
            print("  FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
