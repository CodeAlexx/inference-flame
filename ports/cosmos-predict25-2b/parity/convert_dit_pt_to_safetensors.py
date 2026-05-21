#!/usr/bin/env python3
"""
Convert a Cosmos-Predict2.5 `.pt` DiT checkpoint to `safetensors`.

Cosmos ships its DiT weights as a pickled `.pt` file (e.g.
`base/post-trained/<uuid>_ema_bf16.pt`). The Rust loader only reads
safetensors. This one-shot script normalises a checkpoint:

  - load the pickle on CPU
  - if the top-level is a dict-of-dicts ({"ema": ..., "model": ...}),
    pick the EMA branch in preference
  - strip any "module." / "ema." / "model." key prefixes
  - drop any *non-tensor* values (optimizer state, scalars, etc.)
  - cast tensors to BF16 unless already BF16
  - write `<input-stem>.safetensors` alongside the input

Usage:
    python3 convert_dit_pt_to_safetensors.py /path/to/x_ema_bf16.pt
    python3 convert_dit_pt_to_safetensors.py /path/to/x_ema_bf16.pt --out /tmp/dit.safetensors

This is a *development* tool. Runtime is pure Rust; this script is never
called from a binary.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file


_PREFIX_CANDIDATES = ("module.", "ema.", "model.", "_orig_mod.", "net.")


def _flatten_to_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Walk a loaded pickle until we have a flat str→Tensor map."""
    if isinstance(obj, torch.nn.Module):
        return dict(obj.state_dict())
    if isinstance(obj, dict):
        # Prefer EMA, then model, then state_dict, otherwise treat dict as-is.
        for key in ("ema", "ema_state_dict", "model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], (dict, torch.nn.Module)):
                return _flatten_to_state_dict(obj[key])
        # Heuristic: if every value is a Tensor we're already flat.
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)
        # Heuristic: if some values are dicts of tensors, merge with dotted keys.
        flat: Dict[str, torch.Tensor] = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                flat[k] = v
            elif isinstance(v, dict) and all(
                isinstance(x, torch.Tensor) for x in v.values()
            ):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
        if flat:
            return flat
    raise SystemExit(
        f"unrecognised checkpoint structure: top-level type {type(obj)!r}"
    )


def _strip_prefix(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIX_CANDIDATES:
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    return name


def convert(in_path: Path, out_path: Path) -> None:
    print(f"loading {in_path} (this may take a minute) …", flush=True)
    obj = torch.load(str(in_path), map_location="cpu", weights_only=False)
    sd_raw = _flatten_to_state_dict(obj)

    sd_out: Dict[str, torch.Tensor] = {}
    n_skipped_nontensor = 0
    n_cast = 0
    for k, v in sd_raw.items():
        if not isinstance(v, torch.Tensor):
            n_skipped_nontensor += 1
            continue
        nk = _strip_prefix(k)
        # Cast non-floating dtypes through float32 first to avoid bf16 surprises.
        if v.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            v = v.to(torch.float32)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)
            n_cast += 1
        # safetensors needs contiguous buffers
        if not v.is_contiguous():
            v = v.contiguous()
        sd_out[nk] = v

    print(
        f"  → {len(sd_out)} tensors, {n_cast} cast to bf16, "
        f"{n_skipped_nontensor} non-tensor entries dropped"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd_out, str(out_path))
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="path to *.pt checkpoint")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path; defaults to <input-stem>.safetensors next to input",
    )
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.is_file():
        raise SystemExit(f"input not found: {in_path}")

    out_path: Path = (
        args.out
        if args.out is not None
        else in_path.with_suffix(".safetensors")
    )

    convert(in_path, out_path)


if __name__ == "__main__":
    main()
