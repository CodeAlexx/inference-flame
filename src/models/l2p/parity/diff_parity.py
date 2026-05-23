#!/usr/bin/env python3
"""Diff two parity-capture safetensors files (Python vs Rust).

Both files store F32 tensors keyed by the L2P capture name set.
Prints a table sorted by cosine similarity ascending (worst first).
Flags anything below cos=0.999 with a banner.
"""

import argparse
import os
import safetensors.torch as st
import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Compute cosine sim, max_abs, MSE in F64 over the FLATTENED tensors."""
    a = a.detach().to(torch.float64).reshape(-1)
    b = b.detach().to(torch.float64).reshape(-1)
    dot = (a * b).sum().item()
    na = (a * a).sum().sqrt().item()
    nb = (b * b).sum().sqrt().item()
    denom = na * nb
    cos = float(dot / denom) if denom > 0 else float("nan")
    diff = (a - b)
    max_abs = float(diff.abs().max().item())
    mse = float((diff * diff).mean().item())
    return cos, max_abs, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="python_capture .safetensors")
    ap.add_argument("--rust", required=True, help="rust_capture .safetensors")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()

    py = st.load_file(args.python, device="cpu")
    ru = st.load_file(args.rust, device="cpu")

    keys = sorted(set(py.keys()) | set(ru.keys()))
    print(f"[diff] python={args.python}")
    print(f"[diff] rust  ={args.rust}")
    print(f"[diff] threshold cos >= {args.threshold}")
    print()

    rows = []
    for k in keys:
        if k not in py:
            rows.append((k, None, None, None, "MISSING in python", False))
            continue
        if k not in ru:
            rows.append((k, None, None, None, "MISSING in rust", False))
            continue
        a = py[k]  # python
        b = ru[k]  # rust
        note = ""

        # Normalize shape mismatches that are due to differing layouts but
        # mathematically equivalent content.

        # 1. x_after_embedder: Python [1024, 3840] (no batch), Rust [1, 1024, 3840]
        if a.dim() == 2 and b.dim() == 3 and b.shape[0] == 1 and a.shape == b.shape[1:]:
            b = b.squeeze(0)

        # 2. cap_after_embedder:
        #    Python [224, 3840] (POST-pad: 204 cap rows + 20 cap_pad_token rows)
        #    Rust   [1, 204, 3840] (pre-pad — cap_embedder runs BEFORE pad_to_multiple)
        if k == "cap_after_embedder" and a.dim() == 2 and b.dim() == 3 and b.shape[0] == 1:
            # Compare the first cap_len rows of Python against all of Rust.
            cap_len_rust = b.shape[1]
            if cap_len_rust <= a.shape[0]:
                a_sub = a[:cap_len_rust]
                b_sub = b.squeeze(0)
                cos, ma, mse = cos_sim(a_sub, b_sub)
                note = f"compared first {cap_len_rust} rows of py (pre-pad subset)"
                passed = (cos >= args.threshold)
                rows.append((k, cos, ma, mse, note, passed))
                continue

        # 3. Token-axis 1248: re-order Rust [cap|img] → [img|cap].
        if a.shape == b.shape and a.dim() == 3 and a.shape[1] == 1248:
            # Python: [img (1024) then cap (224)]
            # Rust:   [cap (224 padded) then img (1024)]
            b_reord = torch.cat([b[:, 224:1248, :], b[:, 0:224, :]], dim=1)
            cos, ma, mse = cos_sim(a, b_reord)
            note = "reordered rust [cap|img]→[img|cap]"
            passed = (cos >= args.threshold)
            rows.append((k, cos, ma, mse, note, passed))
            continue

        if a.shape == b.shape:
            cos, ma, mse = cos_sim(a, b)
            passed = (cos >= args.threshold)
            rows.append((k, cos, ma, mse, note, passed))
            continue

        # shape mismatch we couldn't reconcile
        note = f"shape mismatch py={tuple(a.shape)} ru={tuple(b.shape)}"
        rows.append((k, float("nan"), float("nan"), float("nan"), note, False))

    # sort by cos ascending, with errors at the top
    def sort_key(r):
        if r[1] is None:
            return -2.0
        if isinstance(r[1], float) and (r[1] != r[1]):  # NaN
            return -1.0
        return r[1]

    rows_sorted = sorted(rows, key=sort_key)

    print(f"{'name':<32}  {'cos':>12}  {'max_abs':>12}  {'mse':>12}  status  note")
    print("-" * 110)
    for k, cos, ma, mse, note, passed in rows_sorted:
        if cos is None:
            print(f"{k:<32}  {'-':>12}  {'-':>12}  {'-':>12}  ERR     {note}")
            continue
        status = "OK   " if passed else "FAIL "
        cos_s = f"{cos:.6f}" if cos == cos else "NaN"
        ma_s = f"{ma:.4e}" if ma == ma else "NaN"
        mse_s = f"{mse:.4e}" if mse == mse else "NaN"
        print(f"{k:<32}  {cos_s:>12}  {ma_s:>12}  {mse_s:>12}  {status}  {note}")

    # Summary
    nfail = sum(1 for r in rows_sorted if r[1] is None or (r[1] == r[1] and r[1] < args.threshold))
    npass = len(rows_sorted) - nfail
    print()
    print(f"[summary] {npass}/{len(rows_sorted)} passed (cos >= {args.threshold})")
    # Print first-divergence: first key with non-OK status by capture order
    capture_order = [
        "t_emb", "x_after_embedder", "cap_after_embedder",
        "context_refiner_0_out", "context_refiner_1_out",
        "noise_refiner_0_out", "noise_refiner_1_out",
        "unified_initial",
    ] + [f"unified_after_layer_{i:02d}" for i in range(30)] + [
        "feat_map", "local_decoder_out"
    ]
    by_name = {k: (cos, ma, mse, note, passed) for k, cos, ma, mse, note, passed in rows_sorted if cos is not None}
    print()
    print("[forward order, first divergence]")
    first_bad = None
    for k in capture_order:
        if k not in by_name:
            continue
        cos, ma, mse, note, passed = by_name[k]
        if not passed and first_bad is None:
            first_bad = (k, cos, ma)
        print(f"  {k:<32}  cos={cos:.6f}  max_abs={ma:.3e}  {'OK' if passed else 'FAIL'}")
    if first_bad is not None:
        print(f"\n*** first divergence: {first_bad[0]}  cos={first_bad[1]:.6f}  max_abs={first_bad[2]:.3e}")
    else:
        print("\n*** all-clean across capture chain")


if __name__ == "__main__":
    main()
