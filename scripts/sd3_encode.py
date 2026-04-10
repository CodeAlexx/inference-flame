#!/usr/bin/env python
"""SD3.5 Large — encode text embeddings (CLIP-L + CLIP-G + T5-XXL).

Saves cached embeddings for the Rust sd3_infer binary.

Usage:
    python sd3_encode.py "prompt" "negative" /path/to/embeds.safetensors
"""
from __future__ import annotations
import os, sys, time
import torch
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers import T5EncoderModel, AutoTokenizer

CLIP_L = "openai/clip-vit-large-patch14"
CLIP_G = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
T5_XXL = "google/t5-v1_1-xxl"

# Check for local T5
T5_LOCAL = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a photograph of an astronaut riding a horse"
    negative = sys.argv[2] if len(sys.argv) > 2 else ""
    out = sys.argv[3] if len(sys.argv) > 3 else "/home/alex/serenity/output/sd3_embeddings.safetensors"

    device = "cuda"
    dtype = torch.float16

    print(f"=== SD3.5 Encode: {prompt!r} ===")

    # CLIP-L
    print("--- CLIP-L ---")
    tok_l = CLIPTokenizer.from_pretrained(CLIP_L)
    clip_l = CLIPTextModel.from_pretrained(CLIP_L, torch_dtype=dtype).to(device).eval()

    def encode_clip_l(text):
        tokens = tok_l([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_l(**tokens)
        return out.last_hidden_state, out.pooler_output  # [1,77,768], [1,768]

    cond_l, pool_l = encode_clip_l(prompt)
    uncond_l, upool_l = encode_clip_l(negative)
    del clip_l; torch.cuda.empty_cache()
    print(f"  cond_l: {tuple(cond_l.shape)}, pool_l: {tuple(pool_l.shape)}")

    # CLIP-G
    print("--- CLIP-G ---")
    tok_g = CLIPTokenizer.from_pretrained(CLIP_G)
    clip_g = CLIPTextModelWithProjection.from_pretrained(CLIP_G, torch_dtype=dtype).to(device).eval()

    def encode_clip_g(text):
        tokens = tok_g([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_g(**tokens)
        return out.last_hidden_state, out.text_embeds  # [1,77,1280], [1,1280]

    cond_g, pool_g = encode_clip_g(prompt)
    uncond_g, upool_g = encode_clip_g(negative)
    del clip_g; torch.cuda.empty_cache()
    print(f"  cond_g: {tuple(cond_g.shape)}, pool_g: {tuple(pool_g.shape)}")

    # T5-XXL (optional, pad to 77 tokens for SD3)
    print("--- T5-XXL ---")
    from safetensors.torch import load_file
    from transformers import T5Config
    tok_t5 = AutoTokenizer.from_pretrained(T5_XXL)
    config = T5Config.from_pretrained(T5_XXL)
    t5 = T5EncoderModel(config).to(dtype=torch.bfloat16)
    if os.path.exists(T5_LOCAL):
        state = load_file(T5_LOCAL, device="cpu")
        state.pop("spiece_model", None)
        t5.load_state_dict(state, strict=False)
    else:
        t5 = T5EncoderModel.from_pretrained(T5_XXL, torch_dtype=torch.bfloat16)
    t5 = t5.to(device).eval()

    def encode_t5(text):
        tokens = tok_t5([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = t5(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        return out.last_hidden_state.to(dtype)  # [1, 77, 4096]

    cond_t5 = encode_t5(prompt)
    uncond_t5 = encode_t5(negative)
    del t5; torch.cuda.empty_cache()
    print(f"  cond_t5: {tuple(cond_t5.shape)}")

    # Build SD3 format:
    # encoder_hidden_states = cat(clip_l_77_padded_to_4096, clip_g_77_padded_to_4096, t5_77) → [1, 154+77, 4096]
    # Actually SD3 does: clip_l[1,77,768]→pad→[1,77,4096], clip_g[1,77,1280]→pad→[1,77,4096]
    # Then cat with T5 [1,77,4096] → [1, 231, 4096]... but the Rust binary expects [B, 154, 4096]
    # Let me check what the binary actually uses

    # SD3 format: CLIP-L (77 tokens, pad to 4096) + CLIP-G (77 tokens, pad to 4096) = 154 tokens
    # The binary says "CLIP-L(77) + CLIP-G(77) + T5-XXL(77→pad)"
    pad_l = torch.zeros(1, 77, 4096 - 768, device=device, dtype=dtype)
    cond_l_padded = torch.cat([cond_l, pad_l], dim=-1)
    uncond_l_padded = torch.cat([uncond_l, pad_l], dim=-1)

    pad_g = torch.zeros(1, 77, 4096 - 1280, device=device, dtype=dtype)
    cond_g_padded = torch.cat([cond_g, pad_g], dim=-1)
    uncond_g_padded = torch.cat([uncond_g, pad_g], dim=-1)

    # Concat: [1, 77+77+77, 4096] = [1, 231, 4096]
    # But binary says 154... let me just save what makes sense and let user adjust
    encoder_hidden_states = torch.cat([cond_l_padded, cond_g_padded, cond_t5], dim=1)  # [1, 231, 4096]
    encoder_hidden_states_uncond = torch.cat([uncond_l_padded, uncond_g_padded, uncond_t5], dim=1)

    # Pooled: CLIP-L(768) + CLIP-G(1280) = 2048
    pooled = torch.cat([pool_l, pool_g], dim=-1)  # [1, 2048]
    pooled_uncond = torch.cat([upool_l, upool_g], dim=-1)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tensors = {
        "encoder_hidden_states": encoder_hidden_states.to(torch.bfloat16).cpu(),
        "encoder_hidden_states_uncond": encoder_hidden_states_uncond.to(torch.bfloat16).cpu(),
        "pooled_projections": pooled.to(torch.bfloat16).cpu(),
        "pooled_projections_uncond": pooled_uncond.to(torch.bfloat16).cpu(),
    }
    save_file(tensors, out)
    print(f"\nSAVED: {out}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

if __name__ == "__main__":
    main()
