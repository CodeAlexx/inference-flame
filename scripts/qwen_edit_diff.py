#!/usr/bin/env python
"""Diff Rust vs diffusers per-step Qwen-Edit-2511 dumps.

Locates the first step + key where Rust and Python disagree beyond a
tolerance. This is the point where `qwenimage_edit_gen` diverges from
the diffusers reference.

Usage:
    python qwen_edit_diff.py <dump_dir>

Expects:
    <dump_dir>/step_000.safetensors, step_001.safetensors, ...   (Rust)
    <dump_dir>/py/step_000.safetensors, step_001.safetensors, ... (diffusers)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    n = af.norm() * bf.norm()
    if n.item() < 1e-30:
        return 1.0
    return (af @ bf / n).item()


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "cos": cos_sim(a, b),
        "a_mean": float(a.float().mean().item()),
        "a_std": float(a.float().std().item()),
        "b_mean": float(b.float().mean().item()),
        "b_std": float(b.float().std().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=str)
    ap.add_argument("--cos-threshold", type=float, default=0.999)
    ap.add_argument("--max-abs-threshold", type=float, default=0.05)
    ap.add_argument("--keys", nargs="*",
                    default=["latents_in", "cond_pred_full", "uncond_pred_full",
                            "noise_pred_cfg", "latents_out"])
    args = ap.parse_args()

    dd = Path(args.dump_dir)
    py_dir = dd / "py"
    rust_files = sorted(dd.glob("step_*.safetensors"))
    py_files = sorted(py_dir.glob("step_*.safetensors"))
    n = min(len(rust_files), len(py_files))
    if n == 0:
        print(f"[error] no step_*.safetensors pairs found in {dd} and {py_dir}")
        return 2

    print(f"Comparing {n} steps (keys={args.keys})")
    print(f"Thresholds: cos ≥ {args.cos_threshold}, max_abs ≤ {args.max_abs_threshold}")
    print()

    hdr = f"{'step':>4}  {'key':25s}  {'cos':>10s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'a_mean':>10s}  {'b_mean':>10s}  {'flag'}"
    print(hdr)
    print("-" * len(hdr))

    first_divergence = None
    for i in range(n):
        rstep = load_file(rust_files[i])
        pstep = load_file(py_files[i])
        for key in args.keys:
            if key not in rstep or key not in pstep:
                continue
            s = stats(rstep[key], pstep[key])
            flag = ""
            if s["cos"] < args.cos_threshold or s["max_abs"] > args.max_abs_threshold:
                flag = "  <-- DIVERGE"
                if first_divergence is None:
                    first_divergence = (i, key, s)
            print(f"{i:>4}  {key:25s}  {s['cos']:10.6f}  {s['max_abs']:10.4e}  "
                  f"{s['mean_abs']:10.4e}  {s['a_mean']:10.4e}  {s['b_mean']:10.4e}{flag}")
        if first_divergence is not None and i >= first_divergence[0] + 2:
            # Print a few more steps after the first divergence then stop —
            # once divergence is found, subsequent steps are dominated by it.
            print()
            print(f"[stopping at step {i} — first divergence was at step {first_divergence[0]}, key {first_divergence[1]!r}]")
            break

    if first_divergence is None:
        print()
        print("✅ All steps match within tolerance.")
        return 0

    step, key, s = first_divergence
    print()
    print(f"=== FIRST DIVERGENCE ===")
    print(f"  step={step} key={key}")
    print(f"  cos={s['cos']:.6f}  max_abs={s['max_abs']:.4e}  mean_abs={s['mean_abs']:.4e}")
    print(f"  rust: mean={s['a_mean']:.4e} std={s['a_std']:.4e}")
    print(f"  py:   mean={s['b_mean']:.4e} std={s['b_std']:.4e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
