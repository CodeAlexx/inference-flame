#!/usr/bin/env python
"""Quantize a diffusers-style BF16 safetensors shard set to a single FP8 slab.

Produces the same slab layout the LTX-2 FP8 loader expects (see
`inference-flame/src/models/ltx2_model.rs::load_fp8_resident`):

  - 2D Linear `.weight` tensors whose name matches one of the downcast
    suffixes below are stored as `F8_E4M3` bytes with a companion
    `<name>_scale` F32 scalar (= max_abs / 448.0 per tensor).
  - Everything else (norms, biases, F32 scale_shift_tables, input
    embedders, etc.) is copied through unchanged.

Target models (what this script can quantize):
  - Qwen-Image / Qwen-Image-Edit-2511/2512:
      `attn.(to_q|to_k|to_v|to_out.0|add_q_proj|add_k_proj|add_v_proj|to_add_out).weight`
      `(img_mlp|txt_mlp).net.0.proj.weight`, `(img_mlp|txt_mlp).net.2.weight`
      `img_mod.1.weight`, `txt_mod.1.weight`
      `img_in.weight`, `txt_in.weight`, `time_text_embed.*linear_[12].weight`,
      `norm_out.linear.weight`, `proj_out.weight`
  - Klein 4B / 9B: FLUX2-style DiT — `transformer_blocks.*.` covers nearly
    everything; the norm/embedder carve-outs match.

Usage:
    python fp8_quantize.py --in /path/to/diffusers/transformer --out /path/to/slab.safetensors
    # or with an explicit shard list:
    python fp8_quantize.py --in shard0.safetensors:shard1.safetensors --out slab.safetensors

Any `--in` directory is scanned for `*.safetensors` (skipping
`diffusion_pytorch_model.safetensors.index.json`).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FP8_MAX = 448.0  # e4m3fn max representable
FP8_MAX_T = torch.tensor(FP8_MAX)

# Name suffixes that should be FP8-quantized.
# Covers two families of 2D Linear weight naming:
#   (1) Diffusers-style (Qwen-Image, SD3, standard HF transformers)
#   (2) FLUX-LDM-style (Klein 4B/9B, FLUX.1) — fused qkv, `double_blocks.*`,
#       `single_blocks.*`, `.linear1/.linear2` in single stream, etc.
FP8_SUFFIXES = (
    # --- Diffusers-style attention ---
    ".to_q.weight", ".to_k.weight", ".to_v.weight", ".to_out.0.weight",
    ".add_q_proj.weight", ".add_k_proj.weight", ".add_v_proj.weight",
    ".to_add_out.weight",
    # --- Diffusers-style MLPs ---
    ".img_mlp.net.0.proj.weight", ".img_mlp.net.2.weight",
    ".txt_mlp.net.0.proj.weight", ".txt_mlp.net.2.weight",
    ".ff.net.0.proj.weight", ".ff.net.2.weight",
    # --- Diffusers-style modulation ---
    ".img_mod.1.weight", ".txt_mod.1.weight",
    # --- FLUX-LDM double-stream (Klein / FLUX.1) ---
    ".img_attn.qkv.weight", ".img_attn.proj.weight",
    ".txt_attn.qkv.weight", ".txt_attn.proj.weight",
    ".img_mlp.0.weight", ".img_mlp.2.weight",
    ".txt_mlp.0.weight", ".txt_mlp.2.weight",
    ".img_mod.lin.weight", ".txt_mod.lin.weight",
    # --- FLUX-LDM single-stream ---
    ".linear1.weight", ".linear2.weight",
    ".modulation.lin.weight",
    # --- Global Linears that are large enough to matter ---
    "img_in.weight", "txt_in.weight",
    "time_in.in_layer.weight", "time_in.out_layer.weight",
    "vector_in.in_layer.weight", "vector_in.out_layer.weight",
    "guidance_in.in_layer.weight", "guidance_in.out_layer.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "norm_out.linear.weight", "proj_out.weight",
    # FLUX final layer
    "final_layer.linear.weight",
    "final_layer.adaLN_modulation.1.weight",
    "single_stream_modulation.lin.weight",
)


def is_fp8_target(name: str, tensor: torch.Tensor) -> bool:
    """True iff `name` is a 2D Linear weight that should be FP8-quantized."""
    if tensor.dim() != 2:
        return False
    if tensor.dtype != torch.bfloat16:
        # Only quantize BF16 sources; if it's already F32 or F16 we leave it
        # alone — the caller can pre-cast if they want.
        return False
    return any(name.endswith(s) or name == s.lstrip(".") for s in FP8_SUFFIXES)


def quantize_tensor(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """BF16 [..., ...] → (FP8 e4m3fn [..., ...], scale: float).

    scale = max_abs / 448  (covers the full E4M3 range).
    The decode path is  decoded = fp8_to_f32(byte) * scale, so the encode is
      byte = f32_to_fp8(x / scale)  (handled by tensor.to(torch.float8_e4m3fn)).
    """
    abs_max = t.detach().float().abs().max().item()
    if abs_max == 0.0:
        scale = 1.0
    else:
        scale = abs_max / FP8_MAX
    scaled = t.float() / scale
    # Clamp explicitly — `to(float8_e4m3fn)` saturates to ±448 but clamping
    # first avoids any undefined inf/nan path.
    scaled = scaled.clamp(-FP8_MAX, FP8_MAX)
    fp8 = scaled.to(torch.float8_e4m3fn)
    return fp8, float(scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True,
                    help="Directory of shards, single file, or colon-separated file list")
    ap.add_argument("--out", required=True, help="Output slab path (.safetensors)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Collect shards.
    if ":" in args.input:
        shards = [Path(p) for p in args.input.split(":")]
    elif os.path.isdir(args.input):
        shards = sorted(Path(p) for p in glob.glob(os.path.join(args.input, "*.safetensors")))
        # Filter out any index-like shards.
        shards = [s for s in shards if not s.name.endswith(".index.json")]
    else:
        shards = [Path(args.input)]

    if not shards:
        print(f"[error] no shards found under {args.input!r}")
        return 2

    print("=== fp8_quantize ===")
    for s in shards:
        sz = s.stat().st_size / 1e9
        print(f"  in: {s}  ({sz:.2f} GB)")
    print(f"  out: {args.out}")
    print()

    # Walk every shard and classify each tensor.
    t0 = time.time()
    out_tensors: dict[str, torch.Tensor] = {}
    n_fp8 = 0
    n_passthrough = 0
    bytes_fp8 = 0
    bytes_passthrough = 0

    for shard in shards:
        print(f"--- Loading {shard.name} ---")
        t_s = time.time()
        tensors = load_file(str(shard))
        print(f"  {len(tensors)} tensors in {time.time()-t_s:.1f}s")

        for name, t in tensors.items():
            if is_fp8_target(name, t):
                fp8, scale = quantize_tensor(t)
                out_tensors[name] = fp8.contiguous()
                out_tensors[f"{name}_scale"] = torch.tensor(scale, dtype=torch.float32)
                n_fp8 += 1
                bytes_fp8 += fp8.numel()  # 1 byte/elem
            else:
                out_tensors[name] = t.contiguous()
                n_passthrough += 1
                bytes_passthrough += t.numel() * t.element_size()

    print()
    print(f"FP8-quantized:  {n_fp8} tensors, {bytes_fp8/1e9:.2f} GB raw")
    print(f"Pass-through:   {n_passthrough} tensors, {bytes_passthrough/1e9:.2f} GB raw")
    print(f"Total slab size ≈ {(bytes_fp8 + bytes_passthrough)/1e9:.2f} GB")
    print(f"(was ≈ {(bytes_fp8*2 + bytes_passthrough)/1e9:.2f} GB pre-quantize)")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    print()

    if args.dry_run:
        print("[dry-run] not writing output")
        return 0

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"--- Writing {args.out} ---")
    t_w = time.time()
    save_file(out_tensors, args.out)
    print(f"  written in {time.time()-t_w:.1f}s")

    print()
    print(f"DONE. Slab: {args.out}")
    print(f"Size: {os.path.getsize(args.out)/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
