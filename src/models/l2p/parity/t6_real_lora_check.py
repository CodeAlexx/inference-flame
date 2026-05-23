#!/usr/bin/env python3
"""T6b — verify a real trained L2P LoRA has the expected PEFT convention.

The Rust trainer (train_l2p.rs) was patched 2026-05-22 to transpose lora_A/B
before save. This test confirms the saved file has [rank, in] for A and
[out, rank] for B, matching what LoraStack::load expects.
"""
import sys
from pathlib import Path
import safetensors.torch as st

LORA_PATH = "/tmp/l2p_lora_step500_TRANSPOSED.safetensors"  # post-fix file

# Expected dims at L2P config (dim=3840, n_heads=30, head_dim=128):
EXPECTED = {
    "attention.qkv.weight":    (3840, 11520),    # in, out
    "attention.out.weight":    (3840, 3840),
    "feed_forward.w1.weight":  (3840, 10240),
    "feed_forward.w2.weight": (10240, 3840),
    "feed_forward.w3.weight":  (3840, 10240),
}


def main():
    p = Path(LORA_PATH)
    if not p.exists():
        print(f"[T6b] SKIP: {p} not found")
        return 0
    print(f"[T6b] loading {p}")
    t = st.load_file(str(p), device="cpu")
    print(f"[T6b] {len(t)} tensors")

    # Sanity: every tensor should be 2D and named with .lora_A.weight or .lora_B.weight
    bad = []
    rank_seen = set()
    for k, v in t.items():
        if v.ndim != 2:
            bad.append((k, v.shape, "not 2D"))
            continue
        if ".lora_A.weight" not in k and ".lora_B.weight" not in k:
            bad.append((k, v.shape, "wrong suffix"))
            continue
        # Find base key
        is_a = ".lora_A.weight" in k
        is_b = ".lora_B.weight" in k
        # Find which weight (qkv/out/w1/w2/w3)
        matched = None
        for wkey, (in_d, out_d) in EXPECTED.items():
            if wkey in k:
                matched = (wkey, in_d, out_d)
                break
        if matched is None:
            bad.append((k, v.shape, "unknown weight key"))
            continue
        wkey, in_d, out_d = matched
        if is_a:
            # PEFT: A=[rank, in]. So shape[1] == in_d and shape[0] == rank
            if v.shape[1] != in_d:
                bad.append((k, v.shape, f"A: in={v.shape[1]} != expected {in_d}"))
            rank_seen.add(v.shape[0])
        elif is_b:
            # PEFT: B=[out, rank]. So shape[0] == out_d and shape[1] == rank
            if v.shape[0] != out_d:
                bad.append((k, v.shape, f"B: out={v.shape[0]} != expected {out_d}"))
            rank_seen.add(v.shape[1])
    print(f"[T6b] ranks observed: {sorted(rank_seen)}")
    if bad:
        print(f"[T6b] FAIL: {len(bad)} shape/key issues")
        for b in bad[:10]:
            print(f"  {b}")
        return 1

    # Stats on lora_B nonzero — ratio
    nz_total = 0
    nb_total = 0
    for k, v in t.items():
        if ".lora_B.weight" in k:
            nb_total += 1
            if v.abs().max().item() > 0:
                nz_total += 1
    print(f"[T6b] lora_B nonzero modules: {nz_total}/{nb_total}")

    print(f"[T6b] PASS — {len(t)} tensors, all in PEFT convention, rank(s)={sorted(rank_seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
