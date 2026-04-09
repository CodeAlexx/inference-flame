#!/usr/bin/env python
"""Wan2.1-VACE-14B — Stage 1 (text encode + VACE context encode).

Encodes prompt via UMT5-XXL, encodes source frames/masks via VAE for VACE
conditioning, and saves everything for Rust Stage 2.

For animate-anything (image → video):
    python wan22_vace_encode.py "prompt" "" --image ref.png --out embeds.safetensors

For text-to-video (no source):
    python wan22_vace_encode.py "prompt" "" --out embeds.safetensors

Output tensors:
    cond:          [1, L_cond, 4096] BF16
    uncond:        [1, L_uncond, 4096] BF16
    vace_context:  [96, F_lat, H_lat, W_lat] BF16  (z + mask concatenated)
    target_h:      [1] F32
    target_w:      [1] F32
    frame_num:     [1] F32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, T5EncoderModel, T5Config

# Paths
UMT5_WEIGHTS = "/home/alex/.serenity/models/text_encoders/umt5_xxl_fp16.safetensors"
UMT5_TOKENIZER = "google/umt5-xxl"
VAE_WEIGHTS = "/home/alex/.serenity/models/upscalers/Wan2.1_VAE.safetensors"
TEXT_LEN = 512

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
WanVAE_ = _vae_mod.WanVAE_

DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


def build_vae():
    """Build Wan2.1 VAE from safetensors."""
    cfg = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
               num_res_blocks=2, attn_scales=[],
               temperal_downsample=[False, True, True], dropout=0.0)
    with torch.device("meta"):
        model = WanVAE_(**cfg)
    state = load_file(VAE_WEIGHTS, device="cuda")
    model.load_state_dict(state, assign=True)
    model = model.eval().requires_grad_(False).to("cuda")

    mean_vals = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    std_vals = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    mean = torch.tensor(mean_vals, dtype=torch.float, device="cuda")
    std = torch.tensor(std_vals, dtype=torch.float, device="cuda")
    scale = [mean, 1.0 / std]
    return model, scale


def encode_masks(mask, vae_stride=VAE_STRIDE):
    """Encode mask to VACE format: [1, F, H, W] → [64, F_lat, H_lat, W_lat].

    Reshapes spatial dims into channels (8×8=64) then temporally downsamples.
    """
    c, depth, height, width = mask.shape
    new_depth = int((depth + 3) // vae_stride[0])
    height = 2 * (int(height) // (vae_stride[1] * 2))
    width = 2 * (int(width) // (vae_stride[2] * 2))

    mask = mask[0, :, :, :]
    mask = mask.view(depth, height, vae_stride[1], width, vae_stride[2])
    mask = mask.permute(2, 4, 0, 1, 3)  # [8, 8, depth, height, width]
    mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height, width)  # [64, depth, H, W]

    mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width),
                         mode='nearest-exact').squeeze(0)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="a cat sitting on a windowsill, looking outside")
    parser.add_argument("negative", nargs="?", default=DEFAULT_NEG)
    parser.add_argument("--image", type=str, default=None, help="Reference image for animate-anything")
    parser.add_argument("--out", type=str, default="/home/alex/serenity/output/wan22_vace_embeds.safetensors")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=272)
    parser.add_argument("--frames", type=int, default=81)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    # Latent sizes
    lat_h = (args.height // VAE_STRIDE[1] // PATCH_SIZE[1]) * PATCH_SIZE[1]
    lat_w = (args.width // VAE_STRIDE[2] // PATCH_SIZE[2]) * PATCH_SIZE[2]
    h = lat_h * VAE_STRIDE[1]
    w = lat_w * VAE_STRIDE[2]
    f_lat = (args.frames - 1) // VAE_STRIDE[0] + 1

    print("=== Wan2.1-VACE-14B — Stage 1 (text + VACE encode) ===")
    print(f"Prompt: {args.prompt!r}")
    print(f"Image:  {args.image}")
    print(f"Size:   {w}x{h}, frames={args.frames}, f_lat={f_lat}")
    print()

    # ------------------------------------------------------------------
    # 1. Encode text
    # ------------------------------------------------------------------
    print("--- Loading UMT5-XXL ---")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(UMT5_TOKENIZER)
    config = T5Config.from_pretrained(UMT5_TOKENIZER)
    model = T5EncoderModel(config).to(dtype=dtype)
    state_dict = load_file(UMT5_WEIGHTS, device="cpu")
    state_dict.pop("spiece_model", None)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    def encode_text(text):
        tokens = tokenizer([text], max_length=TEXT_LEN, padding=True,
                           truncation=True, return_tensors="pt",
                           add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        seq_len = tokens.attention_mask.sum(dim=1).item()
        return out.last_hidden_state[:, :seq_len, :].to(dtype=dtype)

    cond = encode_text(args.prompt)
    uncond = encode_text(args.negative)
    print(f"  cond: {tuple(cond.shape)}, uncond: {tuple(uncond.shape)}")
    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Encode VACE context (source frames + masks)
    # ------------------------------------------------------------------
    print("\n--- Loading VAE ---")
    vae_model, vae_scale = build_vae()
    print("  VAE ready")

    if args.image is not None:
        # Animate-anything: first frame is reference image, rest are zeros
        # Mask: first frame = 0 (known), rest = 1 (generate)
        print(f"--- Encoding reference image: {args.image} ---")
        img = Image.open(args.image).convert("RGB")
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5
        img_resized = F.interpolate(img_t.unsqueeze(0), size=(h, w),
                                     mode="bicubic", align_corners=False).squeeze(0)

        # Source video: first frame = image, rest = zeros
        src_video = torch.cat([
            img_resized.unsqueeze(1),
            torch.zeros(3, args.frames - 1, h, w),
        ], dim=1).to(device)  # [3, F, H, W]

        # Mask: first frame = 0 (keep), rest = 1 (generate)
        mask = torch.ones(1, args.frames, h, w, device=device)
        mask[:, 0] = 0.0
    else:
        # Pure T2V: all frames zeros, all masks ones (generate everything)
        src_video = torch.zeros(3, args.frames, h, w, device=device)
        mask = torch.ones(1, args.frames, h, w, device=device)

    # VACE encode: inactive + reactive + mask
    print("--- VACE encoding ---")
    t0 = time.time()

    # Apply mask: inactive = src * (1-mask), reactive = src * mask
    mask_binary = torch.where(mask > 0.5, 1.0, 0.0)
    inactive = src_video * (1 - mask_binary) + 0 * mask_binary
    reactive = src_video * mask_binary + 0 * (1 - mask_binary)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float):
        inactive_lat = vae_model.encode(inactive.unsqueeze(0), vae_scale).float().squeeze(0)
        reactive_lat = vae_model.encode(reactive.unsqueeze(0), vae_scale).float().squeeze(0)
    z = torch.cat([inactive_lat, reactive_lat], dim=0)  # [32, F_lat, H_lat, W_lat]
    print(f"  z (inactive+reactive): {tuple(z.shape)} in {time.time() - t0:.1f}s")

    # Encode mask
    m = encode_masks(mask, VAE_STRIDE)  # [64, F_lat, H_lat, W_lat]
    print(f"  m (mask): {tuple(m.shape)}")

    # Concatenate z + m → vace_context
    vace_ctx = torch.cat([z, m], dim=0)  # [96, F_lat, H_lat, W_lat]
    print(f"  vace_context: {tuple(vace_ctx.shape)}")

    del vae_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tensors = {
        "cond": cond.contiguous().cpu(),
        "uncond": uncond.contiguous().cpu(),
        "vace_context": vace_ctx.to(dtype=dtype).contiguous().cpu(),
        "target_h": torch.tensor([h], dtype=torch.float32),
        "target_w": torch.tensor([w], dtype=torch.float32),
        "frame_num": torch.tensor([args.frames], dtype=torch.float32),
    }
    save_file(tensors, args.out)
    print(f"\nSAVED: {args.out}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
