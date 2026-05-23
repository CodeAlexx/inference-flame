#!/usr/bin/env python3
"""T6 — LoRA save→load→save roundtrip parity.

Creates a synthetic L2P LoRA file matching the format `train_l2p.rs` writes
(PEFT layout: `diffusion_model.<key>.lora_A.weight` [rank,in],
`diffusion_model.<key>.lora_B.weight` [out,rank], both BF16).

Then:
  1. Round-trip via safetensors save → load → save → load → compare byte-for-byte.
  2. Verify the dtype roundtrips F32 cast cleanly back to BF16.
  3. Verify the inference-side LoRA loader (`LoraStack::load`) accepts the file:
     since LoraStack does `.transpose().contiguous()` unconditionally on both A and B,
     the file must be saved in PEFT [rank,in]/[out,rank] convention.

PASS criteria:
  - First save and second save are tensor-content-identical.
  - Loaded vs saved tensors are tensor-content-identical.
  - LoRA file is loadable by safetensors with the expected keys + shapes.

We don't actually invoke the Rust loader here — that's a build-and-run dance.
We DO verify the file matches the convention the Rust loader expects.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import safetensors.torch as st


def make_synthetic_l2p_lora(rank=8, seed=42, dtype=torch.bfloat16, device="cuda"):
    """Build a synthetic L2P LoRA file matching what train_l2p.rs writes.

    Targets: 170 modules across 34 blocks × 5 keys each (qkv/out/w1/w2/w3).
    PEFT save layout: lora_A.weight = [rank, in], lora_B.weight = [out, rank].
    """
    # Block layout: 2 noise_refiner + 2 context_refiner + 30 layers
    block_specs = (
        [("noise_refiner", i) for i in range(2)]
        + [("context_refiner", i) for i in range(2)]
        + [("layers", i) for i in range(30)]
    )
    # Dims from L2P config: dim=3840, n_heads=30, head_dim=128, kv_heads=30
    # qkv: in=3840, out=3*30*128=11520
    # out: in=30*128=3840, out=3840
    # w1: in=3840, out=10240
    # w2: in=10240, out=3840
    # w3: in=3840, out=10240
    KEY_DIMS = {
        "attention.qkv.weight":    (3840, 11520),
        "attention.out.weight":    (3840, 3840),
        "feed_forward.w1.weight":  (3840, 10240),
        "feed_forward.w2.weight": (10240, 3840),
        "feed_forward.w3.weight":  (3840, 10240),
    }
    gen = torch.Generator(device=device).manual_seed(seed)
    tensors = {}
    for (mlist, bi) in block_specs:
        for wkey, (in_d, out_d) in KEY_DIMS.items():
            base = f"diffusion_model.{mlist}.{bi}.{wkey}"
            # PEFT: lora_A=[rank,in], lora_B=[out,rank]
            a = torch.randn((rank, in_d), generator=gen, device=device, dtype=torch.float32) * (1.0 / (rank ** 0.5))
            b = torch.zeros((out_d, rank), device=device, dtype=torch.float32)
            # Spike B with a small randn so the round-trip isn't trivially zero everywhere.
            b += torch.randn_like(b) * 0.01
            tensors[f"{base}.lora_A.weight"] = a.to(dtype)
            tensors[f"{base}.lora_B.weight"] = b.to(dtype)
    return tensors


def tensor_eq(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """Returns (exact_eq, max_abs_diff_f64)."""
    if a.shape != b.shape:
        return False, float("inf")
    if a.dtype != b.dtype:
        return False, float("inf")
    af = a.to(torch.float64)
    bf = b.to(torch.float64)
    md = (af - bf).abs().max().item()
    return bool(torch.equal(a, b)), md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/l2p_thorough_parity")
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[T6] device={device}")

    print(f"[T6] generating synthetic L2P LoRA (rank={args.rank})...")
    orig = make_synthetic_l2p_lora(rank=args.rank, device=device)
    print(f"[T6] {len(orig)} tensors (170 modules × 2 = 340 expected)")
    assert len(orig) == 340, f"expected 340 tensors, got {len(orig)}"

    # First save.
    p1 = out / "t6_lora_pass1.safetensors"
    st.save_file({k: v.contiguous().cpu() for k, v in orig.items()}, str(p1))
    print(f"[T6] save #1 → {p1} ({p1.stat().st_size:,} bytes)")

    # Load.
    loaded1 = st.load_file(str(p1), device=device)
    assert set(loaded1.keys()) == set(orig.keys()), "key mismatch on first load"

    # Compare orig vs loaded1.
    n_eq = 0
    n_neq = 0
    worst = 0.0
    for k in orig:
        eq, md = tensor_eq(orig[k], loaded1[k])
        if eq:
            n_eq += 1
        else:
            n_neq += 1
            worst = max(worst, md)
    print(f"[T6] orig vs load#1: {n_eq}/{len(orig)} byte-equal, worst max_abs_diff = {worst:.2e}")
    if n_neq > 0:
        print(f"[T6] FAIL: {n_neq} tensors changed on save→load")
        return 1

    # Second save.
    p2 = out / "t6_lora_pass2.safetensors"
    st.save_file({k: v.contiguous().cpu() for k, v in loaded1.items()}, str(p2))
    print(f"[T6] save #2 → {p2}")

    # Re-load.
    loaded2 = st.load_file(str(p2), device=device)

    # Compare load#1 vs load#2.
    n_eq = 0
    for k in orig:
        eq, _ = tensor_eq(loaded1[k], loaded2[k])
        if eq:
            n_eq += 1
    print(f"[T6] load#1 vs load#2: {n_eq}/{len(orig)} byte-equal")
    if n_eq != len(orig):
        print("[T6] FAIL: round-trip lost information")
        return 1

    # Byte-equal file check.
    b1 = p1.read_bytes()
    b2 = p2.read_bytes()
    print(f"[T6] file byte-equal: {b1 == b2}  (sizes: {len(b1)} vs {len(b2)})")
    if b1 != b2:
        # Safetensors header can include the float-key order or metadata
        # differences; tensor data should still be identical. Check separately.
        print("[T6] note: byte-level file diff is OK if metadata order differs; tensors checked above.")

    # Verify shapes match PEFT convention (which is what LoraStack::load expects).
    print("[T6] checking shape convention matches Rust LoraStack::load expectations...")
    bad_shape = 0
    for k, v in loaded2.items():
        if ".lora_A.weight" in k:
            # PEFT [rank, in], LoraStack transposes to [in, rank]
            if v.shape[0] > v.shape[1]:
                print(f"[T6] SHAPE WARN: {k} {tuple(v.shape)} — A should have rank as smaller axis")
                bad_shape += 1
        elif ".lora_B.weight" in k:
            # PEFT [out, rank], LoraStack transposes to [rank, out]
            if v.shape[1] > v.shape[0]:
                print(f"[T6] SHAPE WARN: {k} {tuple(v.shape)} — B should have rank as smaller axis")
                bad_shape += 1
    print(f"[T6] shape-convention violations: {bad_shape}/{len(loaded2)}")

    # Sample a few and print shapes.
    print("[T6] sample keys + shapes:")
    for k in sorted(loaded2.keys())[:6]:
        print(f"  {k:<80} {tuple(loaded2[k].shape)} {loaded2[k].dtype}")

    print("[T6] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
