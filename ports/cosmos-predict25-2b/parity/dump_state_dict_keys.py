#!/usr/bin/env python3
"""
Dump every key in a `.pt` or `.safetensors` checkpoint with shape + dtype.

Used during port-build to confirm the exact attribute names produced by the
Cosmos `.pt` checkpoint (EMA wrappers can prepend `ema.` / `module.`, the
state-dict layout in BUILD_PLAN.md is the *expected* pattern but needs
verification).

Output: one line per tensor, sorted alphabetically:
    <key>    <dtype>    [d0, d1, ...]

Usage:
    python3 dump_state_dict_keys.py path/to/checkpoint.pt
    python3 dump_state_dict_keys.py path/to/checkpoint.safetensors
    python3 dump_state_dict_keys.py path/to/x.pt --output keys.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def _iter_safetensors(path: Path) -> Iterable[Tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)


def _iter_pt(path: Path) -> Iterable[Tuple[str, torch.Tensor]]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)

    def _walk(prefix: str, x):
        if isinstance(x, torch.Tensor):
            yield prefix, x
            return
        if isinstance(x, dict):
            for k, v in x.items():
                yield from _walk(f"{prefix}.{k}" if prefix else str(k), v)
            return
        # Anything else (numbers, optimizer state, etc.) we skip — but tag once.
        return

    yield from _walk("", obj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="write to file instead of stdout",
    )
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.is_file():
        raise SystemExit(f"input not found: {in_path}")

    if in_path.suffix == ".safetensors":
        it = _iter_safetensors(in_path)
    else:
        it = _iter_pt(in_path)

    rows = []
    for k, t in it:
        shape = list(t.shape)
        rows.append((k, str(t.dtype).replace("torch.", ""), shape))
    rows.sort(key=lambda r: r[0])

    out_lines = [
        f"{k}\t{dt}\t{shape}"
        for k, dt, shape in rows
    ]
    out_lines.append(f"# total tensors: {len(rows)}")
    body = "\n".join(out_lines) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
        print(f"wrote {args.output} ({len(rows)} tensors)", file=sys.stderr)
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
