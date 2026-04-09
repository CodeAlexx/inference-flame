#!/usr/bin/env python
"""Wan2.2-T2V-A14B — Stage 1 (UMT5-XXL text encode).

Loads UMT5-XXL encoder, encodes prompt + negative prompt, saves variable-length
BF16 embeddings to safetensors. Never loads the DiT or VAE.

Usage:
    python wan22_encode.py "your prompt" "negative prompt" /path/to/embeds.safetensors

Output tensors (BF16):
    cond:          [1, L_cond, 4096]
    uncond:        [1, L_uncond, 4096]
    cond_seq_len:  [1]  (scalar — actual token count)
    uncond_seq_len: [1]  (scalar — actual token count)
"""
from __future__ import annotations

import os
import sys
import time

import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# UMT5-XXL encoder (Wan's custom T5 implementation)
# ---------------------------------------------------------------------------
# We load the Wan2.2 T5/tokenizer modules directly to avoid pulling in
# the full wan package (which requires librosa for speech2video).
WAN_ROOT = "/home/alex/Wan2.2"
sys.path.insert(0, WAN_ROOT)

import importlib.util

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_tok_mod = _load_module("wan.modules.tokenizers", f"{WAN_ROOT}/wan/modules/tokenizers.py")
_t5_mod = _load_module("wan.modules.t5", f"{WAN_ROOT}/wan/modules/t5.py")
umt5_xxl = _t5_mod.umt5_xxl
HuggingfaceTokenizer = _tok_mod.HuggingfaceTokenizer

# Paths
UMT5_CHECKPOINT = "/home/alex/.serenity/models/text_encoders/umt5_xxl_fp16.safetensors"
UMT5_TOKENIZER = "google/umt5-xxl"
TEXT_LEN = 512


def load_umt5_encoder(device: str, dtype: torch.dtype):
    """Load UMT5-XXL encoder-only model from safetensors checkpoint."""
    print("--- Loading UMT5-XXL encoder ---")
    t0 = time.time()

    # Build model on CPU
    model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=dtype,
        device="cpu",
    ).eval().requires_grad_(False)

    # Load weights from safetensors
    from safetensors.torch import load_file
    state_dict = load_file(UMT5_CHECKPOINT, device="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    # Tokenizer
    tokenizer = HuggingfaceTokenizer(
        name=UMT5_TOKENIZER, seq_len=TEXT_LEN, clean="whitespace"
    )

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def encode_prompt(
    model, tokenizer, prompt: str, device: str
) -> tuple[torch.Tensor, int]:
    """Encode a single prompt, return (hidden_states, seq_len)."""
    ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_len = mask.gt(0).sum(dim=1).long().item()

    with torch.no_grad():
        context = model(ids, mask)

    # Trim to actual sequence length
    emb = context[:seq_len].unsqueeze(0)  # [1, L, 4096]
    return emb, seq_len


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "A serene mountain landscape at sunset with golden light "
        "reflecting off a still lake, cinematic, highly detailed"
    )
    negative = sys.argv[2] if len(sys.argv) > 2 else (
        # Wan's default Chinese negative prompt
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    out_path = sys.argv[3] if len(sys.argv) > 3 else (
        "/home/alex/serenity/output/wan22_embeds.safetensors"
    )

    device = "cuda"
    dtype = torch.bfloat16

    print("=== Wan2.2-T2V-A14B — Stage 1 (UMT5-XXL encode) ===")
    print(f"Prompt:   {prompt!r}")
    print(f"Negative: {negative!r}")
    print(f"Output:   {out_path}")
    print()

    model, tokenizer = load_umt5_encoder(device, dtype)

    print("--- Encoding cond ---")
    t0 = time.time()
    cond, cond_len = encode_prompt(model, tokenizer, prompt, device)
    print(f"  cond: {tuple(cond.shape)}, seq_len={cond_len} in {time.time() - t0:.1f}s")

    print("--- Encoding uncond ---")
    t0 = time.time()
    uncond, uncond_len = encode_prompt(model, tokenizer, negative, device)
    print(f"  uncond: {tuple(uncond.shape)}, seq_len={uncond_len} in {time.time() - t0:.1f}s")

    # Free VRAM
    del model
    torch.cuda.empty_cache()
    print("  Encoder evicted")
    print()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tensors = {
        "cond": cond.contiguous().cpu(),
        "uncond": uncond.contiguous().cpu(),
        "cond_seq_len": torch.tensor([cond_len], dtype=torch.int64),
        "uncond_seq_len": torch.tensor([uncond_len], dtype=torch.int64),
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
