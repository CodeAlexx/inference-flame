#!/usr/bin/env python
"""HunyuanVideo 1.5 — Stage 1 (text encode + SingleTokenRefiner).

Loads the DiT checkpoint's text preprocessing layers (txt_in, byt5_in,
cond_type_embedding) along with Qwen2.5-VL text encoder, runs the
SingleTokenRefiner, and saves the refined text embeddings for Rust Stage 2.

Usage:
    python hunyuan15_encode.py "prompt" /path/to/embeds.safetensors [W] [H] [F]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

# Qwen2.5-VL for text encoding
QWEN_MODEL = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"  # has Qwen2.5-VL-7B text_encoder
DIT_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model_bf16.safetensors"

# HunyuanVideo uses a simple prompt template
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video.\n"
    "2. The camera angle and movement.\n"
    "3. The content and composition of each frame.\n"
    "4. Dynamics and changes between frames.\n"
    "5. The overall visual style and atmosphere.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="A cat sitting on a windowsill watching birds fly by outside")
    parser.add_argument("out", nargs="?", default="/home/alex/serenity/output/hunyuan15_embeds.safetensors")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=272)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument(
        "--negative",
        type=str,
        default="",
        help="Negative prompt for CFG. Empty string disables uncond emission.",
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    print("=== HunyuanVideo 1.5 — Stage 1 (text encode) ===")
    print(f"Prompt: {args.prompt!r}")
    print()

    # ------------------------------------------------------------------
    # 1. Encode text with Qwen2.5-VL (cond + optional uncond for CFG)
    # ------------------------------------------------------------------
    print("--- Loading Qwen2.5-VL tokenizer + encoder ---")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(f"{QWEN_MODEL}/tokenizer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{QWEN_MODEL}/text_encoder",
        dtype=dtype,
        device_map=device,
    )
    text_encoder.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    def _encode(prompt_str: str):
        formatted_ = PROMPT_TEMPLATE.format(prompt_str)
        tokens_ = tokenizer(
            [formatted_],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out_ = text_encoder(
                input_ids=tokens_.input_ids,
                attention_mask=tokens_.attention_mask,
                output_hidden_states=True,
            )
        return out_.hidden_states[-1], tokens_.attention_mask

    text_hidden, text_mask = _encode(args.prompt)
    print(f"  text_hidden (cond): {tuple(text_hidden.shape)}")

    use_uncond = bool(args.negative is not None)  # always emit uncond (default = "")
    if use_uncond:
        uncond_hidden, uncond_mask = _encode(args.negative)
        print(f"  text_hidden (uncond): {tuple(uncond_hidden.shape)}")

    del text_encoder
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Run SingleTokenRefiner (from DiT checkpoint)
    # ------------------------------------------------------------------
    print("\n--- Loading SingleTokenRefiner from DiT checkpoint ---")
    t0 = time.time()
    dit_state = load_file(DIT_CHECKPOINT, device=device)

    # Extract txt_in weights
    txt_in_keys = {k: v for k, v in dit_state.items() if k.startswith("txt_in.")}
    # Extract byt5_in weights (if needed)
    byt5_keys = {k: v for k, v in dit_state.items() if k.startswith("byt5_in.")}
    # cond_type_embedding
    cond_type_w = dit_state.get("cond_type_embedding.weight")  # [3, 2048]

    print(f"  {len(txt_in_keys)} txt_in keys, {len(byt5_keys)} byt5 keys")

    # SingleTokenRefiner forward:
    # 1. t_embedder(timestep=0) → timestep_aware
    # 2. c_embedder(mean(text)) → context_aware
    # 3. c = timestep_aware + context_aware
    # 4. x = input_embedder(text)
    # 5. x = individual_token_refiner(x, c, mask) — 2 self-attn blocks

    # timestep = 0 for encoding
    dim = 2048
    freq_dim = 256

    # Sinusoidal embedding for t=0
    half = freq_dim // 2
    t_emb = torch.zeros(1, freq_dim, device=device, dtype=dtype)
    # t=0 → cos(0)=1, sin(0)=0
    t_emb[0, :half] = 1.0

    # t_embedder: Linear(256, 2048) → SiLU → Linear(2048, 2048)
    t_w0 = txt_in_keys["txt_in.t_embedder.mlp.0.weight"]
    t_b0 = txt_in_keys["txt_in.t_embedder.mlp.0.bias"]
    t_w2 = txt_in_keys["txt_in.t_embedder.mlp.2.weight"]
    t_b2 = txt_in_keys["txt_in.t_embedder.mlp.2.bias"]
    timestep_aware = torch.nn.functional.silu(torch.nn.functional.linear(t_emb, t_w0, t_b0))
    timestep_aware = torch.nn.functional.linear(timestep_aware, t_w2, t_b2)  # [1, 2048]

    # c_embedder: TextProjection(3584, 2048)
    # context_aware = mean(text, weighted by mask)
    mask_float = text_mask.float().unsqueeze(-1)
    context_mean = (text_hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1)  # [1, 3584]
    c_w1 = txt_in_keys["txt_in.c_embedder.linear_1.weight"]
    c_b1 = txt_in_keys["txt_in.c_embedder.linear_1.bias"]
    c_w2 = txt_in_keys["txt_in.c_embedder.linear_2.weight"]
    c_b2 = txt_in_keys["txt_in.c_embedder.linear_2.bias"]
    context_aware = torch.nn.functional.silu(torch.nn.functional.linear(context_mean.to(dtype), c_w1, c_b1))
    context_aware = torch.nn.functional.linear(context_aware, c_w2, c_b2)  # [1, 2048]

    c = timestep_aware + context_aware  # [1, 2048]

    def _refine(text_hidden_in, text_mask_in):
        # c_embedder: TextProjection(3584, 2048)
        mask_float = text_mask_in.float().unsqueeze(-1)
        context_mean = (text_hidden_in * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware = torch.nn.functional.silu(
            torch.nn.functional.linear(context_mean.to(dtype), c_w1, c_b1)
        )
        context_aware = torch.nn.functional.linear(context_aware, c_w2, c_b2)  # [1, 2048]
        c_local = timestep_aware + context_aware

        # input_embedder
        ie_w = txt_in_keys["txt_in.input_embedder.weight"]
        ie_b = txt_in_keys["txt_in.input_embedder.bias"]
        x_local = torch.nn.functional.linear(text_hidden_in.to(dtype), ie_w, ie_b)

        for block_idx in range(2):
            bp = f"txt_in.individual_token_refiner.blocks.{block_idx}"
            ada_w = txt_in_keys[f"{bp}.adaLN_modulation.1.weight"]
            ada_b = txt_in_keys[f"{bp}.adaLN_modulation.1.bias"]
            gates = torch.nn.functional.linear(torch.nn.functional.silu(c_local), ada_w, ada_b)
            gate_msa, gate_mlp = gates.chunk(2, dim=1)

            n1_w = txt_in_keys[f"{bp}.norm1.weight"]
            n1_b = txt_in_keys[f"{bp}.norm1.bias"]
            norm_x = torch.nn.functional.layer_norm(
                x_local.float(), [dim], n1_w.float(), n1_b.float()
            ).to(dtype)

            qkv_w = txt_in_keys[f"{bp}.self_attn_qkv.weight"]
            qkv_b = txt_in_keys[f"{bp}.self_attn_qkv.bias"]
            qkv = torch.nn.functional.linear(norm_x, qkv_w, qkv_b)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(1, -1, 16, 128).transpose(1, 2)
            k = k.reshape(1, -1, 16, 128).transpose(1, 2)
            v = v.reshape(1, -1, 16, 128).transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(1, -1, dim)
            proj_w = txt_in_keys[f"{bp}.self_attn_proj.weight"]
            proj_b = txt_in_keys[f"{bp}.self_attn_proj.bias"]
            attn = torch.nn.functional.linear(attn, proj_w, proj_b)
            x_local = x_local + attn * gate_msa.unsqueeze(1)

            n2_w = txt_in_keys[f"{bp}.norm2.weight"]
            n2_b = txt_in_keys[f"{bp}.norm2.bias"]
            norm_x2 = torch.nn.functional.layer_norm(
                x_local.float(), [dim], n2_w.float(), n2_b.float()
            ).to(dtype)
            mlp_w1 = txt_in_keys[f"{bp}.mlp.fc1.weight"]
            mlp_b1 = txt_in_keys[f"{bp}.mlp.fc1.bias"]
            mlp_w2 = txt_in_keys[f"{bp}.mlp.fc2.weight"]
            mlp_b2 = txt_in_keys[f"{bp}.mlp.fc2.bias"]
            mlp_out = torch.nn.functional.silu(
                torch.nn.functional.linear(norm_x2, mlp_w1, mlp_b1)
            )
            mlp_out = torch.nn.functional.linear(mlp_out, mlp_w2, mlp_b2)
            x_local = x_local + mlp_out * gate_mlp.unsqueeze(1)

        if cond_type_w is not None:
            x_local = x_local + cond_type_w[0].unsqueeze(0).unsqueeze(0)
        return x_local

    refined_txt = _refine(text_hidden, text_mask)
    print(f"  Refined text (cond): {tuple(refined_txt.shape)}")
    if use_uncond:
        refined_uncond = _refine(uncond_hidden, uncond_mask)
        print(f"  Refined text (uncond): {tuple(refined_uncond.shape)}")
    print(f"  Done in {time.time() - t0:.1f}s")

    del dit_state
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tensors = {
        "txt_embeds": refined_txt.contiguous().cpu(),
        "txt_mask": text_mask.to(torch.float32).contiguous().cpu(),
        "target_h": torch.tensor([args.height], dtype=torch.float32),
        "target_w": torch.tensor([args.width], dtype=torch.float32),
        "frame_num": torch.tensor([args.frames], dtype=torch.float32),
    }
    save_file(tensors, args.out)
    print(f"\nSAVED: {args.out}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
