#!/usr/bin/env python
"""Dump intermediate tensors of the Qwen2.5-VL text encoder for
layer-by-layer parity against the Rust port.

Emits to `output/qwen25vl_layer_dump.safetensors`:
  token_ids        — [seq_len] i64 (full padded)
  embed_out        — [1, seq, 3584] bf16  (after model.embed_tokens)
  layer_{i}_out    — [1, seq, 3584] bf16  for i in {0, 1, 27}
  final_hidden     — [1, seq, 3584] bf16  (after model.norm)
  non_pad_mask     — [seq] bool
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

PROMPT = "a photograph of an astronaut riding a horse on mars"
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
SNAP = Path("/home/alex/.serenity/models/checkpoints/qwen-image-2512")
OUT = Path("/home/alex/EriDiffusion/inference-flame/output/qwen25vl_layer_dump.safetensors")
MAX_LEN = 1058  # 1024 + 34


def main() -> int:
    tok = AutoTokenizer.from_pretrained(SNAP / "tokenizer")
    wrapped = PROMPT_TEMPLATE.format(PROMPT)
    enc = tok(
        [wrapped],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    text = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        SNAP / "text_encoder",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    text.eval()

    # The actual language model — drop the visual tower.
    lm = text.model  # Qwen2_5_VLModel
    # Get at the text-only submodule. In HF this is `text.model.language_model`
    # or `text.model` depending on version.
    # For Qwen2_5_VLForConditionalGeneration: forward calls self.model which is Qwen2_5_VLModel
    # and produces hidden states internally. The easy path: run full forward with output_hidden_states.

    with torch.no_grad():
        out = text(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True,
        )
    # `out.hidden_states` is a tuple of length (num_layers + 1):
    #   [0] — embedding (pre-layer-0)
    #   [i] — output AFTER layer (i-1), i.e. [1] = after layer 0
    #   ...
    #   [num_layers] — after final layer (pre final-norm)
    # Then there's usually a final_hidden which is AFTER model.norm.
    # For causal LMs, `out.hidden_states[-1]` is typically the FINAL normed hidden.
    hs = out.hidden_states
    print(f"got {len(hs)} hidden-state tensors, shapes: {[tuple(h.shape) for h in hs[:3]]} ... {[tuple(h.shape) for h in hs[-2:]]}")

    emb = hs[0]
    layer0 = hs[1]
    layer1 = hs[2]
    final_hidden = hs[-1]
    layer27 = hs[-2] if len(hs) >= 29 else hs[-1]

    # flame-core's safetensors loader skips integer dtypes — store token
    # ids and attention mask as F32 so they survive the round-trip.
    tensors = {
        "token_ids": enc.input_ids[0].to(torch.float32).cpu(),
        "embed_out": emb.contiguous().cpu(),
        "layer_0_out": layer0.contiguous().cpu(),
        "layer_1_out": layer1.contiguous().cpu(),
        "layer_27_out": layer27.contiguous().cpu(),
        "final_hidden": final_hidden.contiguous().cpu(),
        "attention_mask": enc.attention_mask[0].to(torch.float32).cpu(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
