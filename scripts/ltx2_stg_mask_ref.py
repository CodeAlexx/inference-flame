#!/usr/bin/env python
"""Emit reference STG skip-layer masks by importing Lightricks's actual
`Transformer3DModel.create_skip_layer_mask` (transformer3d.py:173-188).
No reimplementation — we construct a minimal Transformer3DModel-like
stand-in that runs the exact mask-building loop from upstream.

Output: output/ltx2_stg_mask_ref.safetensors
  mask_large : [48, 3]  — num_layers=48, batch=1, num_conds=3, skip=[11,25,35,39]
  mask_small : [4,  3]  — num_layers=4,  batch=1, num_conds=3, skip=[1,3]  (debug)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

LTX_REPO = Path("/tmp/ltx-video")
sys.path.insert(0, str(LTX_REPO))

OUT = Path("/home/alex/EriDiffusion/inference-flame/output/ltx2_stg_mask_ref.safetensors")


def reference_create_skip_layer_mask(
    num_layers: int,
    batch_size: int,
    num_conds: int,
    ptb_index: int,
    skip_block_list: list[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact copy of the body of Transformer3DModel.create_skip_layer_mask
    (ltx_video/models/transformers/transformer3d.py:173-188), lifted as a
    free function so we don't need to instantiate the full model to verify
    the tiny mask-building math.  `skip_block_list is None or empty` is
    handled by the caller here.
    """
    mask = torch.ones((num_layers, batch_size * num_conds), device=device, dtype=dtype)
    for block_idx in skip_block_list:
        mask[block_idx, ptb_index::num_conds] = 0
    return mask


def main() -> int:
    # Canonical: 48 layers × batch 1 × 3 conds (uncond+cond+perturb).
    # Perturb slot = last one → ptb_index = num_conds - 1 = 2.
    skip_large = [11, 25, 35, 39]
    mask_large = reference_create_skip_layer_mask(
        num_layers=48,
        batch_size=1,
        num_conds=3,
        ptb_index=2,
        skip_block_list=skip_large,
    )

    # Debug: tiny case for eyeball verification.
    skip_small = [1, 3]
    mask_small = reference_create_skip_layer_mask(
        num_layers=4,
        batch_size=1,
        num_conds=3,
        ptb_index=2,
        skip_block_list=skip_small,
    )

    # Also verify against the real class to make sure we didn't drift.
    # Import and instantiate a skeleton Transformer3DModel with 4 blocks
    # via monkey-patching its __init__ to avoid loading actual weights.
    from ltx_video.models.transformers.transformer3d import Transformer3DModel

    # Build a placeholder that only has .transformer_blocks, .device, .dtype
    class Stub:
        def __init__(self, n):
            self.transformer_blocks = list(range(n))
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    # Bind the real method to our stub and call it to confirm equivalence.
    real_mask_fn = Transformer3DModel.create_skip_layer_mask
    stub_large = Stub(48)
    out_large = real_mask_fn(stub_large, batch_size=1, num_conds=3, ptb_index=2, skip_block_list=skip_large)
    assert torch.equal(out_large, mask_large), "large mask drift from Lightricks ref"

    stub_small = Stub(4)
    out_small = real_mask_fn(stub_small, batch_size=1, num_conds=3, ptb_index=2, skip_block_list=skip_small)
    assert torch.equal(out_small, mask_small), "small mask drift from Lightricks ref"

    print(f"  mask_large shape={tuple(mask_large.shape)}, zero rows = "
          f"{[i for i in range(mask_large.shape[0]) if (mask_large[i] == 0).any()]}")
    print(f"  mask_small shape={tuple(mask_small.shape)}, zero rows = "
          f"{[i for i in range(mask_small.shape[0]) if (mask_small[i] == 0).any()]}")
    print(f"  small mask:\n{mask_small}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "mask_large": mask_large.contiguous(),
            "mask_small": mask_small.contiguous(),
        },
        str(OUT),
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
