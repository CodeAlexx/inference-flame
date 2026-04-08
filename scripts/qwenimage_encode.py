#!/usr/bin/env python
"""Qwen-Image-2512 — Stage 1 (text encode).

Loads Qwen2.5-VL-7B from the Qwen-Image-2512 repo, applies the exact
diffusers prompt template, drops the first 34 tokens (system prompt),
extracts only the non-padded hidden states, and saves cond + uncond
embeddings to a safetensors file. Never loads the DiT or VAE.

Usage:
    python qwenimage_encode.py \\
        "a photograph of an astronaut riding a horse on mars" \\
        ""  \\
        /path/to/embeds.safetensors

The saved file contains two BF16 tensors (shapes match the DROPPED,
non-padded output — they will typically have different seq lengths):
    cond:   [1, L_cond,   3584]
    uncond: [1, L_uncond, 3584]

Stage 2 (Rust) loads these and runs the DiT denoise loop.
Stage 3 (Python) runs the VAE decode.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

# Diffusers pipeline constants (pipeline_qwenimage.py:175-177)
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects "
    "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1024

REPO = "Qwen/Qwen-Image-2512"
SNAP = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/snapshots/"
    "25468b98e3276ca6700de15c6628e51b7de54a26"
)


def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    """Return the non-padded hidden states for each sample (diffusers
    _extract_masked_hidden). For single-prompt input this returns one
    tensor of shape [valid_len, hidden]."""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror of `_get_qwen_prompt_embeds` → drop system-prompt tokens,
    return the [1, L, hidden] tensor for a single prompt."""
    txt = PROMPT_TEMPLATE_ENCODE.format(prompt)
    drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX

    txt_tokens = tokenizer(
        [txt],
        max_length=TOKENIZER_MAX_LENGTH + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
    hidden_states = out.hidden_states[-1]
    split = extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split = [e[drop_idx:] for e in split]  # drop the 34 system-prompt tokens

    # For a single prompt, return [1, L, H]
    emb = split[0].unsqueeze(0).to(dtype=dtype, device=device)
    return emb


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "a photograph of an astronaut riding a horse on mars, "
        "cinematic lighting, highly detailed"
    )
    negative = sys.argv[2] if len(sys.argv) > 2 else ""
    out_path = sys.argv[3] if len(sys.argv) > 3 else (
        "/home/alex/serenity/output/qwenimage_embeds.safetensors"
    )

    device = "cuda"
    dtype = torch.bfloat16

    print("=== Qwen-Image-2512 — Stage 1 (text encode) ===")
    print(f"Prompt:   {prompt!r}")
    print(f"Negative: {negative!r}")
    print(f"Output:   {out_path}")
    print()

    snap = Path(SNAP)
    if not snap.exists():
        print(f"[error] snapshot path does not exist: {snap}")
        print("       download the text_encoder shards first via:")
        print(f"       hf_hub_download('{REPO}', 'text_encoder/...')")
        return 2

    print("--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(snap / "tokenizer")

    print("--- Loading Qwen2.5-VL-7B text encoder ---")
    t0 = time.time()
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snap / "text_encoder",
        dtype=dtype,
        device_map=device,
    )
    text_encoder.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    print("--- Encoding cond ---")
    t0 = time.time()
    cond = encode_prompt(tokenizer, text_encoder, prompt, device, dtype)
    print(f"  cond:   {tuple(cond.shape)} in {time.time() - t0:.1f}s")

    print("--- Encoding uncond ---")
    t0 = time.time()
    uncond = encode_prompt(tokenizer, text_encoder, negative, device, dtype)
    print(f"  uncond: {tuple(uncond.shape)} in {time.time() - t0:.1f}s")

    # Free VRAM
    del text_encoder
    torch.cuda.empty_cache()
    print("  Text encoder evicted")
    print()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tensors = {
        "cond": cond.contiguous().cpu(),
        "uncond": uncond.contiguous().cpu(),
    }
    save_file(tensors, out_path)
    print(f"EMBEDDINGS SAVED: {out_path}")
    print(f"  cond:   {tuple(tensors['cond'].shape)}  {tensors['cond'].dtype}")
    print(f"  uncond: {tuple(tensors['uncond'].shape)}  {tensors['uncond'].dtype}")
    print()
    print(f"Next: ./target/release/qwenimage_gen {out_path} <latents.safetensors>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
