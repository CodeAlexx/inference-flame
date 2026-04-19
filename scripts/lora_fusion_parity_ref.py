#!/usr/bin/env python
"""Parity reference for LoRA fusion: loads the LTX-2 distilled LoRA and
emits the per-key `strength * (B @ A)` deltas for the Rust side to match
against. No model required — pure linear algebra on the LoRA file.

Emits: output/lora_fusion_ref_ltx2_distilled.safetensors
  keys: <base_key>  — float32, shape [out, in], equals strength * B @ A
        __meta__   — one f32 scalar tensor holding `strength`
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LORA = Path("/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors")
OUT = Path("/home/alex/EriDiffusion/inference-flame/output/lora_fusion_ref_ltx2_distilled.safetensors")
STRENGTH = 1.0
# Which keys to sample. Use a handful spread across the model so any
# systematic indexing bug shows up somewhere.
SAMPLE_BASE_KEYS = [
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.attn1.to_k.weight",
    "transformer_blocks.0.attn1.to_v.weight",
    "transformer_blocks.0.attn1.to_out.0.weight",
    "transformer_blocks.0.ff.net.0.proj.weight",
    "transformer_blocks.12.attn2.to_q.weight",
    "transformer_blocks.25.ff.net.2.weight",
]


def main() -> int:
    if not LORA.exists():
        print(f"ERROR: LoRA not found at {LORA}", file=sys.stderr)
        return 2

    # Collect A/B pairs from the file
    a_by_base: dict[str, torch.Tensor] = {}
    b_by_base: dict[str, torch.Tensor] = {}
    all_keys: list[str] = []

    with safe_open(LORA, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        for key in all_keys:
            if ".lora_A." in key or ".lora_a." in key:
                base = strip(key, ".lora_A.weight", ".lora_a.weight")
                if base:
                    a_by_base[base] = f.get_tensor(key)
            elif ".lora_B." in key or ".lora_b." in key:
                base = strip(key, ".lora_B.weight", ".lora_b.weight")
                if base:
                    b_by_base[base] = f.get_tensor(key)

    print(f"LoRA file: {LORA.name}")
    print(f"  total tensors: {len(all_keys)}")
    print(f"  lora_A pairs: {len(a_by_base)}")
    print(f"  lora_B pairs: {len(b_by_base)}")
    print(f"  A/B paired:   {sum(1 for k in a_by_base if k in b_by_base)}")

    # Sanity — sample keys must exist
    missing = [k for k in SAMPLE_BASE_KEYS if k not in a_by_base or k not in b_by_base]
    if missing:
        print(f"\nSome sample keys are not in this LoRA (fine — swapping in present ones):", file=sys.stderr)
        for k in missing:
            print(f"  missing: {k}", file=sys.stderr)
        sample_keys = [k for k in SAMPLE_BASE_KEYS if k not in missing]
        if len(sample_keys) < 3:
            # Pick first few A/B paired keys available
            paired = sorted(set(a_by_base.keys()) & set(b_by_base.keys()))
            sample_keys = paired[:7]
            print(f"Falling back to first 7 paired keys:", file=sys.stderr)
            for k in sample_keys:
                print(f"  {k}  A:{tuple(a_by_base[k].shape)}  B:{tuple(b_by_base[k].shape)}", file=sys.stderr)
    else:
        sample_keys = SAMPLE_BASE_KEYS

    # Also dump the first 3 paired keys regardless to catch the distilled
    # LoRA's own key convention if it doesn't use `transformer_blocks.*`.
    paired_all = sorted(set(a_by_base.keys()) & set(b_by_base.keys()))
    extra = [k for k in paired_all[:5] if k not in sample_keys]
    sample_keys = list(sample_keys) + extra
    print(f"\nFinal sample keys ({len(sample_keys)}):")
    for k in sample_keys:
        a = a_by_base[k]
        b = b_by_base[k]
        print(f"  {k}")
        print(f"    lora_A shape {tuple(a.shape)} dtype {a.dtype}")
        print(f"    lora_B shape {tuple(b.shape)} dtype {b.dtype}")

    # Compute delta per key — all math in float32
    out: dict[str, torch.Tensor] = {}
    for base in sample_keys:
        a = a_by_base[base].to(torch.float32)  # [rank, in]
        b = b_by_base[base].to(torch.float32)  # [out, rank]
        delta = b @ a  # [out, in]
        if abs(STRENGTH - 1.0) > 1e-6:
            delta = delta * STRENGTH
        out[base] = delta.contiguous()

    # Encode strength as a 1-element tensor so it round-trips through
    # safetensors alongside the deltas.
    out["__strength__"] = torch.tensor([STRENGTH], dtype=torch.float32)

    # flame-core's safetensors loader skips integer dtypes, so avoid those.
    # Floats are fine.

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT)
    print(f"\nwrote {OUT}")
    for k, v in out.items():
        print(f"  {k:<55} shape={tuple(v.shape)} dtype={v.dtype}")
    return 0


def strip(key: str, *suffixes: str) -> str | None:
    k = key
    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model."):]
    for s in suffixes:
        if k.endswith(s):
            return k[: -len(s)] + ".weight"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
