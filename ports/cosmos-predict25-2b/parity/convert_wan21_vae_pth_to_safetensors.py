#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert Cosmos-Predict2.5-2B's `tokenizer.pth` (Wan 2.1 VAE state_dict
pickled by PyTorch) into a safetensors file the Rust runtime can load.

The Cosmos repo loads it with `model.load_state_dict(ckpt, assign=True)`
on a `WanVAE_` module (tokenizers/wan2pt1.py:670) — no prefix mangling.
So the .pth is a flat state_dict that should map 1:1 onto our existing
`Wan21VaeDecoder` / `Wan21VaeEncoder` modules.

This script:
  1. Loads the .pth (CPU mmap; no GPU needed — this is just a format
     conversion, not numerics).
  2. Strips any `model.` / `module.` wrapper prefixes.
  3. Casts everything to BF16 (matching Wan 2.1's BF16 inference
     convention; the existing Rust loader casts on load anyway, but
     starting in BF16 saves disk).
  4. Writes `wan21_vae_cosmos.safetensors` next to this script.
  5. Dumps `wan21_vae_cosmos_keys.txt` listing every key/shape/dtype
     for cross-reference with our existing Wan VAE loader.

Usage:
    # Download `nvidia/Cosmos-Predict2.5-2B/tokenizer.pth` (~508 MB).
    huggingface-cli login
    huggingface-cli download nvidia/Cosmos-Predict2.5-2B tokenizer.pth
    python convert_wan21_vae_pth_to_safetensors.py
    # or:
    COSMOS_TOKENIZER_PTH=/path/to/tokenizer.pth \
        python convert_wan21_vae_pth_to_safetensors.py
"""

import os
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"ERROR: {e}. `pip install torch safetensors`")
    sys.exit(1)


WRAPPER_PREFIXES = ("model.", "module.", "vae.", "tokenizer.")


def find_pth() -> Path:
    if "COSMOS_TOKENIZER_PTH" in os.environ:
        p = Path(os.environ["COSMOS_TOKENIZER_PTH"])
        if not p.exists():
            print(f"ERROR: $COSMOS_TOKENIZER_PTH does not exist: {p}")
            sys.exit(1)
        return p
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download("nvidia/Cosmos-Predict2.5-2B", "tokenizer.pth")
        )
    except Exception as e:
        print(f"ERROR: HF download failed: {e}")
        sys.exit(1)


def strip_prefix(key: str) -> str:
    for p in WRAPPER_PREFIXES:
        while key.startswith(p):
            key = key[len(p) :]
    return key


def main() -> int:
    pth = find_pth()
    print(f"Source .pth: {pth} ({pth.stat().st_size / 1e6:.1f} MB)")

    # weights_only=True is the safer default; map_location='cpu' keeps
    # this script GPU-free.
    raw = torch.load(pth, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        print(f"ERROR: expected dict state_dict, got {type(raw)}")
        return 1

    print(f"Top-level state_dict has {len(raw)} entries")
    # Some checkpoints wrap state under "state_dict" / "model".
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
        print(f"  unwrapped 'state_dict' -> {len(raw)} entries")
    elif "model" in raw and isinstance(raw["model"], dict) and all(
        isinstance(v, torch.Tensor) for v in raw["model"].values()
    ):
        raw = raw["model"]
        print(f"  unwrapped 'model' -> {len(raw)} entries")

    if not all(isinstance(v, torch.Tensor) for v in raw.values()):
        bad = [k for k, v in raw.items() if not isinstance(v, torch.Tensor)][:5]
        print(f"ERROR: not all values are tensors. First bad keys: {bad}")
        return 1

    # Strip wrapper prefixes, cast to BF16.
    remapped: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        new_k = strip_prefix(k)
        if new_k in remapped:
            print(f"WARNING: duplicate key after prefix strip: {new_k}")
        # Wan 2.1 norm gains and biases should stay in F32 in the
        # safetensors file for headroom; the Rust loader will downcast
        # if it wants to. Everything else → BF16.
        if v.dtype == torch.float64:
            v = v.to(torch.float32)
        if v.dtype in (torch.float32,) and not (
            new_k.endswith(".gamma") or new_k.endswith(".bias") or "norm" in new_k
        ):
            v = v.to(torch.bfloat16)
        remapped[new_k] = v.contiguous()

    out_path = Path(__file__).parent / "wan21_vae_cosmos.safetensors"
    save_file(remapped, str(out_path))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    keys_path = Path(__file__).parent / "wan21_vae_cosmos_keys.txt"
    with keys_path.open("w") as f:
        f.write(f"# Source: {pth}\n# Total keys: {len(remapped)}\n")
        for k in sorted(remapped):
            t = remapped[k]
            f.write(f"{k}\t{tuple(t.shape)}\t{t.dtype}\n")
    print(f"Wrote {keys_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
