#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dump every weight key + shape + dtype in `nvidia/Cosmos-Reason1-7B`,
side-by-side with our Rust-side expected keys for `Qwen2.5-VL-7B-Instruct`
(`Qwen25VLEncoder::expected_weight_keys`, 28 layers).

Goal: confirm Cosmos-Reason1-7B is byte-for-byte the same checkpoint
layout as the upstream Qwen2.5-VL-7B-Instruct. If it's a 1:1 match, we
can reuse `Qwen25VLEncoder::new` verbatim with the Cosmos weights. If
keys diverge (extra `vlm.` / `model.` wrapper, missing vision shards
when text-only, renamed projections, etc.), we'll add a load-time
remapper in `src/models/cosmos_reason1.rs`.

Usage:
    huggingface-cli login   # required: nvidia/Cosmos-Reason1-7B is gated
    python dump_cosmos_reason1_keys.py
    # or override the local path:
    COSMOS_REASON1_PATH=/path/to/local/snapshot python dump_cosmos_reason1_keys.py

Outputs:
    cosmos_reason1_keys.txt    — sorted `key shape dtype` lines
    cosmos_reason1_diff.txt    — lines present in Rust expected keys but
                                  missing from the checkpoint, and vice
                                  versa (excluding visual.*).
"""

import os
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. `pip install safetensors`")
    sys.exit(1)


def find_snapshot() -> Path:
    """Find the Cosmos-Reason1-7B snapshot directory.

    Order:
    1. `$COSMOS_REASON1_PATH` (explicit override)
    2. HF cache hit for `nvidia/Cosmos-Reason1-7B`
    """
    if "COSMOS_REASON1_PATH" in os.environ:
        p = Path(os.environ["COSMOS_REASON1_PATH"])
        if not p.exists():
            print(f"ERROR: $COSMOS_REASON1_PATH does not exist: {p}")
            sys.exit(1)
        return p
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                "nvidia/Cosmos-Reason1-7B",
                allow_patterns=["*.safetensors", "*.json"],
            )
        )
    except Exception as e:
        print(f"ERROR: HF download failed: {e}")
        print(
            "  Set COSMOS_REASON1_PATH to a local snapshot, or run:\n"
            "    huggingface-cli login && huggingface-cli download nvidia/Cosmos-Reason1-7B"
        )
        sys.exit(1)


def expected_qwen25vl_keys(num_layers: int = 28) -> set[str]:
    """Mirror of Rust `expected_weight_keys` for Qwen2.5-VL text-only path."""
    keys = {"model.embed_tokens.weight"}
    for i in range(num_layers):
        p = f"model.layers.{i}"
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]:
            keys.add(f"{p}.{suffix}")
    keys.add("model.norm.weight")
    return keys


def main() -> int:
    snap = find_snapshot()
    print(f"Snapshot: {snap}")

    shards = sorted(snap.glob("*.safetensors"))
    if not shards:
        # Sometimes the model is sharded inside a `text_encoder/` subdir
        # (e.g. flux-style snapshot layouts). Probe both.
        shards = sorted(snap.glob("**/*.safetensors"))
    if not shards:
        print(f"ERROR: no *.safetensors in {snap}")
        return 1

    all_keys: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                all_keys[k] = (tuple(t.shape), str(t.dtype))

    out_path = Path(__file__).parent / "cosmos_reason1_keys.txt"
    with out_path.open("w") as f:
        f.write(f"# Snapshot: {snap}\n")
        f.write(f"# Total keys: {len(all_keys)}\n")
        for k in sorted(all_keys):
            shape, dtype = all_keys[k]
            f.write(f"{k}\t{shape}\t{dtype}\n")
    print(f"Wrote {out_path} ({len(all_keys)} keys)")

    # Cross-reference against Rust expected keys.
    expected = expected_qwen25vl_keys()
    present_text_only = {k for k in all_keys if not k.startswith("visual.")}

    missing_in_ckpt = expected - set(all_keys)
    extra_in_ckpt = present_text_only - expected

    diff_path = Path(__file__).parent / "cosmos_reason1_diff.txt"
    with diff_path.open("w") as f:
        f.write("# Rust expected keys MISSING from Cosmos-Reason1-7B checkpoint:\n")
        for k in sorted(missing_in_ckpt):
            f.write(f"  - {k}\n")
        f.write(
            "\n# Checkpoint keys NOT in Rust expected list (text-only, "
            "excluding visual.*):\n"
        )
        for k in sorted(extra_in_ckpt):
            shape, dtype = all_keys[k]
            f.write(f"  + {k}\t{shape}\t{dtype}\n")
    print(f"Wrote {diff_path}")
    print(
        f"  missing from ckpt: {len(missing_in_ckpt)}, "
        f"extra in ckpt (text-only): {len(extra_in_ckpt)}"
    )
    if missing_in_ckpt:
        print("  WARNING: some Rust-expected keys are MISSING from the checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
