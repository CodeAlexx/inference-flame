#!/usr/bin/env python
"""AdaIN latent-normalize + tone-map parity reference.

Both functions are pure tensor math from Lightricks's
`ltx_video/pipelines/pipeline_ltx_video.py`:

- `adain_filter_latent` at :1790-1818. Per-channel mean+std match of a
  target tensor to a reference tensor, with a blend factor (default 1.0
  = fully replace target's stats with reference's).
- `tone_map_latents` at :1748-1787. Sigmoid-based amplitude compression
  applied to latent tensors before VAE decode. `compression=0.0` is
  identity; distilled configs use 0.6 on the second pass.

We import Lightricks's actual functions to avoid any reconstruction,
run them on tiny seeded BF16 tensors, and emit input/output pairs
for the Rust ports to match.
"""
from __future__ import annotations

import sys
from pathlib import Path

LTX_REPO = Path("/tmp/ltx-video")
sys.path.insert(0, str(LTX_REPO))

import torch
from safetensors.torch import save_file

from ltx_video.pipelines.pipeline_ltx_video import (  # noqa: E402
    adain_filter_latent,
    LTXVideoPipeline,
)

# tone_map_latents is a staticmethod on LTXVideoPipeline; call it unbound.
tone_map_latents = LTXVideoPipeline.tone_map_latents

OUT = Path("/home/alex/EriDiffusion/inference-flame/output/ltx2_adain_tone_map_ref.safetensors")


def main() -> int:
    out: dict[str, torch.Tensor] = {}

    # --- AdaIN ---
    torch.manual_seed(13)
    # Shape pattern: [B, C, F, H, W] — same 5D layout the pipeline uses.
    ref = torch.randn(1, 4, 3, 4, 4, dtype=torch.float32).to(torch.bfloat16)
    tgt = torch.randn(1, 4, 3, 4, 4, dtype=torch.float32).to(torch.bfloat16)

    out["adain_reference"] = ref.clone()
    out["adain_target"] = tgt.clone()
    # Default factor=1.0 — full replacement of target stats.
    out["adain_out_factor1p0"] = adain_filter_latent(latents=tgt, reference_latents=ref, factor=1.0).contiguous()
    out["adain_out_factor0p5"] = adain_filter_latent(latents=tgt, reference_latents=ref, factor=0.5).contiguous()
    out["adain_out_factor0p0"] = adain_filter_latent(latents=tgt, reference_latents=ref, factor=0.0).contiguous()

    # --- Tone map ---
    # Seeded, wider amplitude so the sigmoid curve actually bends.
    torch.manual_seed(17)
    x = torch.randn(1, 4, 2, 4, 4, dtype=torch.float32).to(torch.bfloat16) * 3.0
    out["tonemap_input"] = x.clone()
    out["tonemap_out_compress_0p0"] = tone_map_latents(latents=x, compression=0.0).contiguous()
    out["tonemap_out_compress_0p6"] = tone_map_latents(latents=x, compression=0.6).contiguous()
    out["tonemap_out_compress_1p0"] = tone_map_latents(latents=x, compression=1.0).contiguous()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT)
    print(f"wrote {OUT}")
    for k, v in out.items():
        flat = v.float().flatten()
        print(f"  {k:<28} shape={tuple(v.shape)} min={flat.min():.4f} max={flat.max():.4f} mean_abs={flat.abs().mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
