#!/usr/bin/env python
"""Wan2.2-I2V-A14B — Stage 1 (UMT5-XXL text encode + VAE image encode).

Encodes prompt via UMT5-XXL, encodes reference image via Wan2.1 VAE,
builds the I2V mask, and saves everything for Rust Stage 2.

Usage:
    python wan22_i2v_encode.py "prompt" "negative" image.png output.safetensors [W] [H] [F]

Output tensors (BF16 unless noted):
    cond:          [1, L_cond, 4096]
    uncond:        [1, L_uncond, 4096]
    y:             [20, F_lat, H_lat, W_lat]  (4 mask + 16 VAE channels)
    target_h:      [1] int64
    target_w:      [1] int64
"""
from __future__ import annotations

import os
import sys
import time
import math

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

# Wan's default Chinese negative prompt
DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


def build_i2v_mask(frame_num: int, lat_h: int, lat_w: int, device: str) -> torch.Tensor:
    """Build the I2V conditioning mask [4, F_lat, H, W].

    First frame = 1.0 (known), rest = 0.0 (to generate).
    Reshaped to match VAE temporal stride.
    """
    f_lat = (frame_num - 1) // VAE_STRIDE[0] + 1
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    # Pad first frame to 4 copies, then reshape to VAE temporal layout
    msk = torch.cat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]  # [4, F_lat, H, W]
    return msk


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "astronaut running on mars"
    negative = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NEG
    image_path = sys.argv[3] if len(sys.argv) > 3 else "/home/alex/serenity/output/chroma_smoke.png"
    out_path = sys.argv[4] if len(sys.argv) > 4 else "/home/alex/serenity/output/wan22_i2v_embeds.safetensors"
    target_w = int(sys.argv[5]) if len(sys.argv) > 5 else 480
    target_h = int(sys.argv[6]) if len(sys.argv) > 6 else 272
    frame_num = int(sys.argv[7]) if len(sys.argv) > 7 else 81

    device = "cuda"
    dtype = torch.bfloat16

    print("=== Wan2.2-I2V-A14B — Stage 1 (text + image encode) ===")
    print(f"Prompt:   {prompt!r}")
    print(f"Image:    {image_path}")
    print(f"Size:     {target_w}x{target_h}, frames={frame_num}")
    print(f"Output:   {out_path}")
    print()

    # Compute latent sizes (must be divisible by patch_size * vae_stride)
    lat_h = target_h // VAE_STRIDE[1]  # /8
    lat_w = target_w // VAE_STRIDE[2]  # /8
    # Ensure divisible by patch_size
    lat_h = (lat_h // PATCH_SIZE[1]) * PATCH_SIZE[1]
    lat_w = (lat_w // PATCH_SIZE[2]) * PATCH_SIZE[2]
    h = lat_h * VAE_STRIDE[1]
    w = lat_w * VAE_STRIDE[2]
    f_lat = (frame_num - 1) // VAE_STRIDE[0] + 1
    print(f"  Target pixels: {w}x{h}, lat: {lat_w}x{lat_h}, f_lat={f_lat}")

    # ------------------------------------------------------------------
    # 1. Encode text
    # ------------------------------------------------------------------
    print("\n--- Loading UMT5-XXL ---")
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
            out = model(input_ids=tokens.input_ids,
                        attention_mask=tokens.attention_mask)
        seq_len = tokens.attention_mask.sum(dim=1).item()
        return out.last_hidden_state[:, :seq_len, :].to(dtype=dtype)

    print("--- Encoding text ---")
    cond = encode_text(prompt)
    uncond = encode_text(negative)
    print(f"  cond: {tuple(cond.shape)}, uncond: {tuple(uncond.shape)}")

    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Encode reference image via VAE
    # ------------------------------------------------------------------
    print("\n--- Loading Wan2.1 VAE ---")
    t0 = time.time()
    vae_cfg = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
                   num_res_blocks=2, attn_scales=[],
                   temperal_downsample=[False, True, True], dropout=0.0)
    with torch.device("meta"):
        vae_model = WanVAE_(**vae_cfg)
    vae_state = load_file(VAE_WEIGHTS, device=device)
    vae_model.load_state_dict(vae_state, assign=True)
    vae_model = vae_model.eval().requires_grad_(False).to(device)

    # Build normalization
    mean_vals = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    std_vals = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    vae_mean = torch.tensor(mean_vals, dtype=torch.float, device=device)
    vae_std = torch.tensor(std_vals, dtype=torch.float, device=device)
    scale = [vae_mean, 1.0 / vae_std]
    print(f"  VAE loaded in {time.time() - t0:.1f}s")

    # Load and preprocess image
    print("--- Encoding reference image ---")
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # normalize to [-1, 1]
    img_tensor = img_tensor.to(device)

    # Resize to target resolution
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze(0)  # [3, H, W]

    # Build video input: first frame = image, rest = zeros
    video_input = torch.cat([
        img_resized.unsqueeze(1),  # [3, 1, H, W]
        torch.zeros(3, frame_num - 1, h, w, device=device),
    ], dim=1)  # [3, F, H, W]

    # VAE encode
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float):
        y_encoded = vae_model.encode(video_input.unsqueeze(0), scale).float().squeeze(0)
    # y_encoded: [16, F_lat, H_lat, W_lat]
    print(f"  VAE encoded: {tuple(y_encoded.shape)} in {time.time() - t0:.1f}s")

    # Build mask
    msk = build_i2v_mask(frame_num, lat_h, lat_w, device)
    print(f"  mask: {tuple(msk.shape)}")

    # Concat mask + encoded image latents
    y = torch.cat([msk, y_encoded], dim=0)  # [20, F_lat, H_lat, W_lat]
    print(f"  y (mask+latent): {tuple(y.shape)}")

    del vae_model, vae_state
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tensors = {
        "cond": cond.contiguous().cpu(),
        "uncond": uncond.contiguous().cpu(),
        "y": y.to(dtype=dtype).contiguous().cpu(),
        "target_h": torch.tensor([h], dtype=torch.float32),
        "target_w": torch.tensor([w], dtype=torch.float32),
        "frame_num": torch.tensor([frame_num], dtype=torch.float32),
    }
    save_file(tensors, out_path)
    print(f"\nEMBEDDINGS SAVED: {out_path}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print(f"\nNext: ./target/release/wan22_i2v_gen {out_path} <latents.safetensors>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
