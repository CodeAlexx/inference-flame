#!/usr/bin/env python
"""Numerical reference for STG rescaling (pipeline_ltx_video.py:1251-1262):

  factor = std(noise_pred_text) / std(noise_pred_guided)
  factor = rescaling_scale * factor + (1 - rescaling_scale)
  noise_pred_out = noise_pred_guided * factor

Emits tiny BF16 tensors so the Rust impl can match the math.  std is
computed per-batch over ALL other dims (view(B, -1).std(dim=1)).
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path("/home/alex/EriDiffusion/inference-flame/output/ltx2_stg_rescale_ref.safetensors")

RESCALE = 0.7  # Lightricks default


def stg_rescale(
    pos: torch.Tensor, guided: torch.Tensor, rescaling_scale: float
) -> torch.Tensor:
    """Verbatim port of pipeline_ltx_video.py:1251-1262."""
    batch_size = pos.shape[0]
    noise_pred_text_std = pos.view(batch_size, -1).std(dim=1, keepdim=True)
    noise_pred_std = guided.view(batch_size, -1).std(dim=1, keepdim=True)

    factor = noise_pred_text_std / noise_pred_std
    factor = rescaling_scale * factor + (1 - rescaling_scale)
    # Lightricks reshapes to [B, 1, 1] and relies on broadcasting across
    # the remaining dims.  Do the same.
    factor = factor.view(batch_size, *[1] * (guided.ndim - 1))
    return guided * factor


def main() -> int:
    torch.manual_seed(0xC0FFEE)

    # Tiny case
    pos = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    guided = torch.randn(1, 4, 8, dtype=torch.bfloat16) * 1.3  # different std
    out_f32 = stg_rescale(pos.float(), guided.float(), RESCALE)
    out_bf16 = out_f32.to(torch.bfloat16)

    # Larger case
    pos_big = torch.randn(2, 8, 4, 8, 8, dtype=torch.bfloat16)
    guided_big = torch.randn(2, 8, 4, 8, 8, dtype=torch.bfloat16) * 0.7
    out_big_f32 = stg_rescale(pos_big.float(), guided_big.float(), RESCALE)
    out_big_bf16 = out_big_f32.to(torch.bfloat16)

    # Also export the "factor" for debugging.
    B = pos.shape[0]
    pos_std = pos.float().view(B, -1).std(dim=1)
    guided_std = guided.float().view(B, -1).std(dim=1)
    factor = RESCALE * (pos_std / guided_std) + (1 - RESCALE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "pos": pos.contiguous(),
            "guided": guided.contiguous(),
            "out_f32": out_f32.contiguous(),
            "out_bf16": out_bf16.contiguous(),
            "factor_f32": factor.contiguous(),
            "pos_big": pos_big.contiguous(),
            "guided_big": guided_big.contiguous(),
            "out_big_f32": out_big_f32.contiguous(),
            "out_big_bf16": out_big_bf16.contiguous(),
        },
        str(OUT),
    )
    print(f"  factor (small): {factor.tolist()}")
    print(f"  rescaling_scale: {RESCALE}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
