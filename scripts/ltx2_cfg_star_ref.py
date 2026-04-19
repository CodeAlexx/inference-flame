#!/usr/bin/env python
"""Numerical reference for CFG-star rescale math (pipeline_ltx_video.py:1227-1240).

  alpha = (<eps_text, eps_uncond> / ||eps_uncond||^2 + 1e-8)
  eps_uncond <- alpha * eps_uncond

Emits tiny BF16 tensors so the Rust impl can diff against them at machine
precision.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path("/home/alex/EriDiffusion/inference-flame/output/ltx2_cfg_star_ref.safetensors")


def cfg_star_rescale(
    eps_text: torch.Tensor, eps_uncond: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim port of pipeline_ltx_video.py:1227-1240.  Operates per-batch.

    Returns (rescaled_uncond, alpha) for inspection.
    """
    batch_size = eps_text.shape[0]
    positive_flat = eps_text.reshape(batch_size, -1)
    negative_flat = eps_uncond.reshape(batch_size, -1)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
    alpha = dot_product / squared_norm  # [B, 1]
    rescaled = alpha.view(batch_size, *[1] * (eps_uncond.ndim - 1)) * eps_uncond
    return rescaled, alpha


def main() -> int:
    torch.manual_seed(0x13377117)

    # Tiny tensors so we can diff element-wise at BF16 precision.
    # Shape [B=1, C=4, N=8] — a 1D sequence in 4 channels, batch 1.
    eps_text = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    eps_uncond = torch.randn(1, 4, 8, dtype=torch.bfloat16)

    # Compute in F32 (matches Lightricks — torch autocast context is f32
    # outside the transformer forward in pipeline_ltx_video.py).
    rescaled_f32, alpha_f32 = cfg_star_rescale(eps_text.float(), eps_uncond.float())
    rescaled_bf16 = rescaled_f32.to(torch.bfloat16)

    # For fuller coverage: a larger shape that stress-tests the reduction.
    eps_text_big = torch.randn(2, 8, 4, 8, 8, dtype=torch.bfloat16)
    eps_uncond_big = torch.randn(2, 8, 4, 8, 8, dtype=torch.bfloat16)
    rescaled_big_f32, alpha_big_f32 = cfg_star_rescale(
        eps_text_big.float(), eps_uncond_big.float()
    )
    rescaled_big_bf16 = rescaled_big_f32.to(torch.bfloat16)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "eps_text": eps_text.contiguous(),
            "eps_uncond": eps_uncond.contiguous(),
            "rescaled_bf16": rescaled_bf16.contiguous(),
            "rescaled_f32": rescaled_f32.contiguous(),
            "alpha_f32": alpha_f32.contiguous(),
            "eps_text_big": eps_text_big.contiguous(),
            "eps_uncond_big": eps_uncond_big.contiguous(),
            "rescaled_big_bf16": rescaled_big_bf16.contiguous(),
            "rescaled_big_f32": rescaled_big_f32.contiguous(),
            "alpha_big_f32": alpha_big_f32.contiguous(),
        },
        str(OUT),
    )
    print(f"  alpha small: {alpha_f32.flatten().tolist()}")
    print(f"  alpha big:   {alpha_big_f32.flatten().tolist()}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
