#!/usr/bin/env python3
"""Generate Z-Image text embeddings using Qwen3 text encoder.

Outputs a safetensors file with key 'cap_feats' [1, seq_len, 2560]
for consumption by the Rust zimage_infer binary.

Usage:
    python tools/zimage_encode.py \
        --encoder /home/alex/.serenity/models/zimage_base/text_encoder \
        --prompt "a cat sitting on a windowsill" \
        --output /home/alex/serenity/output/zimage_embeddings.safetensors
"""
from __future__ import annotations

import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode_prompt(
    model_path: str,
    tokenizer_path: str,
    prompt: str,
    max_length: int = 512,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode a prompt using Qwen3, extracting hidden_states[-2]."""
    print(f"[+] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"[+] Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Apply chat template (Z-Image uses enable_thinking=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )

    print(f"[+] Tokenizing ({len(text)} chars)...")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    print(f"    Token count: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # Extract second-to-last hidden state (Z-Image convention)
    hidden = outputs.hidden_states[-2]  # [B, seq, 2560]

    # Keep only non-padding tokens
    mask = inputs["attention_mask"].bool()
    cap_feats = hidden[mask].unsqueeze(0)  # [1, non_pad_len, 2560]

    print(f"    cap_feats shape: {cap_feats.shape} dtype: {cap_feats.dtype}")
    return cap_feats


def main():
    parser = argparse.ArgumentParser(description="Z-Image text embedding generator")
    parser.add_argument(
        "--encoder",
        default="/home/alex/.serenity/models/zimage_base/text_encoder",
        help="Path to Qwen3 text encoder",
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/alex/.serenity/models/zimage_base/tokenizer",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--prompt",
        default="a beautiful landscape with mountains and a river at sunset, photorealistic",
        help="Text prompt to encode",
    )
    parser.add_argument(
        "--output",
        default="/home/alex/serenity/output/zimage_embeddings.safetensors",
        help="Output safetensors path",
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    cap_feats = encode_prompt(
        args.encoder,
        args.tokenizer,
        args.prompt,
        max_length=args.max_length,
    )

    # Save as BF16
    cap_feats = cap_feats.to(torch.bfloat16).cpu()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"cap_feats": cap_feats}, str(output_path))
    print(f"\n[+] Saved embeddings to {output_path}")
    print(f"    Shape: {cap_feats.shape}")


if __name__ == "__main__":
    main()
