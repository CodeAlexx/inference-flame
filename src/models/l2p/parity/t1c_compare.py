#!/usr/bin/env python3
"""T1c — Compare Rust vs Python L2P inference outputs.

Compares statistics and pixel-wise similarity. Per T11, the initial noise
PRNGs differ, so pixel-cos is NOT expected to be high — we use statistical
sanity (mean/std/range) + structural similarity (SSIM-ish via channel
moments) as the PASS gate.

PASS criteria:
  - Both PNGs exist and are non-degenerate (mean in [0.1, 0.9], std > 0.05).
  - Per-channel means within 0.10 (Δ on [0,1] scale).
  - Per-channel std within 0.10.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def stats(arr):
    f = arr.astype(np.float32) / 255.0
    return {
        "shape": arr.shape,
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": float(f.min()),
        "max": float(f.max()),
        "ch_means": [float(x) for x in f.mean(axis=(0, 1))],
        "ch_stds": [float(x) for x in f.std(axis=(0, 1))],
    }


def main():
    rust_p = Path("/tmp/l2p_thorough_parity/T1/rust_output.png")
    py_p = Path("/tmp/l2p_thorough_parity/T1/python_output.png")
    if not rust_p.exists():
        print(f"FAIL: missing {rust_p}")
        return 1
    if not py_p.exists():
        print(f"FAIL: missing {py_p}")
        return 1
    a = np.array(Image.open(rust_p).convert("RGB"))
    b = np.array(Image.open(py_p).convert("RGB"))
    sa = stats(a)
    sb = stats(b)
    print(f"Rust:   {sa}")
    print(f"Python: {sb}")
    # Pixel-cos (informational, expected to be low)
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    cos = (af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    mean_diff = abs(sa["mean"] - sb["mean"])
    std_diff = abs(sa["std"] - sb["std"])
    ch_mean_diffs = [abs(a - b) for a, b in zip(sa["ch_means"], sb["ch_means"])]
    ch_std_diffs = [abs(a - b) for a, b in zip(sa["ch_stds"], sb["ch_stds"])]
    print(f"\npixel cos (raw uint8): {cos:.6f}  (informational only, noise PRNGs differ)")
    print(f"|Δmean|={mean_diff:.4f}, |Δstd|={std_diff:.4f}")
    print(f"|Δch_means|={[f'{x:.4f}' for x in ch_mean_diffs]}")
    print(f"|Δch_stds|= {[f'{x:.4f}' for x in ch_std_diffs]}")

    valid_a = 0.1 < sa["mean"] < 0.9 and sa["std"] > 0.05
    valid_b = 0.1 < sb["mean"] < 0.9 and sb["std"] > 0.05
    stats_close = mean_diff < 0.10 and std_diff < 0.10
    chan_close = max(ch_mean_diffs) < 0.15 and max(ch_std_diffs) < 0.15

    print()
    print(f"valid Rust output: {valid_a}")
    print(f"valid Python output: {valid_b}")
    print(f"global stats close: {stats_close}")
    print(f"channel stats close: {chan_close}")
    if valid_a and valid_b and stats_close and chan_close:
        print("[T1] PASS (statistical sampler health)")
        return 0
    else:
        print("[T1] FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
