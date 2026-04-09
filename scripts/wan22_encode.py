#!/usr/bin/env python
"""Wan2.2-T2V-A14B — Stage 1 (UMT5-XXL text encode).

Loads UMT5-XXL encoder via HuggingFace transformers, encodes prompt + negative
prompt, saves variable-length BF16 embeddings to safetensors.

Usage:
    python wan22_encode.py "your prompt" "negative prompt" /path/to/embeds.safetensors

Output tensors (BF16):
    cond:   [1, L_cond, 4096]
    uncond: [1, L_uncond, 4096]
"""
from __future__ import annotations

import os
import sys
import time

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, T5EncoderModel

# Paths
UMT5_WEIGHTS = "/home/alex/.serenity/models/text_encoders/umt5_xxl_fp16.safetensors"
UMT5_TOKENIZER = "google/umt5-xxl"
TEXT_LEN = 512

# Wan's default Chinese negative prompt
DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "A serene mountain landscape at sunset with golden light "
        "reflecting off a still lake, cinematic, highly detailed"
    )
    negative = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NEG
    out_path = sys.argv[3] if len(sys.argv) > 3 else (
        "/home/alex/serenity/output/wan22_embeds.safetensors"
    )

    device = "cuda"
    dtype = torch.bfloat16

    print("=== Wan2.2-T2V-A14B — Stage 1 (UMT5-XXL encode) ===")
    print(f"Prompt:   {prompt!r}")
    print(f"Negative: {negative[:80]!r}...")
    print(f"Output:   {out_path}")
    print()

    # ------------------------------------------------------------------
    # Load UMT5-XXL encoder from safetensors (HF format keys)
    # ------------------------------------------------------------------
    print("--- Loading UMT5-XXL tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(UMT5_TOKENIZER)

    print("--- Loading UMT5-XXL encoder ---")
    t0 = time.time()
    from transformers import T5Config
    config = T5Config.from_pretrained(UMT5_TOKENIZER)
    model = T5EncoderModel(config).to(dtype=dtype)
    state_dict = load_file(UMT5_WEIGHTS, device="cpu")
    state_dict.pop("spiece_model", None)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def encode(text: str) -> torch.Tensor:
        tokens = tokenizer(
            [text],
            max_length=TEXT_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
        hidden = out.last_hidden_state  # [1, L, 4096]
        # Trim to actual (non-padded) length
        seq_len = tokens.attention_mask.sum(dim=1).item()
        return hidden[:, :seq_len, :].to(dtype=dtype)

    print("--- Encoding cond ---")
    t0 = time.time()
    cond = encode(prompt)
    print(f"  cond: {tuple(cond.shape)} in {time.time() - t0:.1f}s")

    print("--- Encoding uncond ---")
    t0 = time.time()
    uncond = encode(negative)
    print(f"  uncond: {tuple(uncond.shape)} in {time.time() - t0:.1f}s")

    # Free VRAM
    del model
    torch.cuda.empty_cache()
    print("  Encoder evicted")
    print()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tensors = {
        "cond": cond.contiguous().cpu(),
        "uncond": uncond.contiguous().cpu(),
    }
    save_file(tensors, out_path)
    print(f"EMBEDDINGS SAVED: {out_path}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print()
    print(f"Next: ./target/release/wan22_t2v_gen {out_path} <latents.safetensors>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
