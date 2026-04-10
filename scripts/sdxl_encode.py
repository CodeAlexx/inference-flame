#!/usr/bin/env python
"""SDXL — encode text embeddings (CLIP-L + CLIP-G).

Saves cached embeddings for the Rust sdxl_infer binary.

Usage:
    python sdxl_encode.py "prompt" "negative" /path/to/embeds.safetensors
"""
from __future__ import annotations
import os, sys
import torch
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

CLIP_L = "openai/clip-vit-large-patch14"
CLIP_G = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a photograph of an astronaut riding a horse"
    negative = sys.argv[2] if len(sys.argv) > 2 else ""
    out = sys.argv[3] if len(sys.argv) > 3 else "/home/alex/serenity/output/sdxl_embeddings.safetensors"

    device = "cuda"
    dtype = torch.float16

    print(f"=== SDXL Encode: {prompt!r} ===")

    # CLIP-L (768-dim)
    print("--- CLIP-L ---")
    tok_l = CLIPTokenizer.from_pretrained(CLIP_L)
    clip_l = CLIPTextModel.from_pretrained(CLIP_L, torch_dtype=dtype).to(device).eval()

    def encode_l(text):
        tokens = tok_l([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_l(**tokens)
        return out.last_hidden_state, out.pooler_output

    cond_l_hs, cond_l_pool = encode_l(prompt)
    uncond_l_hs, uncond_l_pool = encode_l(negative)
    del clip_l; torch.cuda.empty_cache()

    # CLIP-G (1280-dim)
    print("--- CLIP-G ---")
    tok_g = CLIPTokenizer.from_pretrained(CLIP_G)
    clip_g = CLIPTextModelWithProjection.from_pretrained(CLIP_G, torch_dtype=dtype).to(device).eval()

    def encode_g(text):
        tokens = tok_g([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_g(**tokens)
        return out.last_hidden_state, out.text_embeds

    cond_g_hs, cond_g_pool = encode_g(prompt)
    uncond_g_hs, uncond_g_pool = encode_g(negative)
    del clip_g; torch.cuda.empty_cache()

    # SDXL format:
    # context = cat(clip_l_hidden[77,768], clip_g_hidden[77,1280]) → [1, 77, 2048]
    # y = cat(clip_l_pooled[768], clip_g_pooled[1280], zeros[768]) → [1, 2816]
    # Pad CLIP-L hidden from 768 to match: actually SDXL concats along hidden dim
    # CLIP-L: [1,77,768], CLIP-G: [1,77,1280] → pad CLIP-L to 1280 then cat?
    # No — SDXL cross-attn expects [1, 77, 2048]: just cat along last dim
    context = torch.cat([cond_l_hs, cond_g_hs], dim=-1)  # [1, 77, 2048]
    context_uncond = torch.cat([uncond_l_hs, uncond_g_hs], dim=-1)

    # Pooled: [768] + [1280] + [768 zeros] = [2816]
    zeros_pad = torch.zeros(1, 768, device=device, dtype=dtype)
    y = torch.cat([cond_l_pool, cond_g_pool, zeros_pad], dim=-1)
    y_uncond = torch.cat([uncond_l_pool, uncond_g_pool, zeros_pad], dim=-1)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tensors = {
        "context": context.to(torch.bfloat16).cpu(),
        "context_uncond": context_uncond.to(torch.bfloat16).cpu(),
        "y": y.to(torch.bfloat16).cpu(),
        "y_uncond": y_uncond.to(torch.bfloat16).cpu(),
    }
    save_file(tensors, out)
    print(f"\nSAVED: {out}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

if __name__ == "__main__":
    main()
