#!/usr/bin/env python
"""LatentUpsampler (LTX-2.3 spatial x2) parity reference.

Imports Lightricks's actual class from the cloned LTX-Video repo,
loads our on-disk 22B upscaler checkpoint, runs a deterministic
forward on a seeded latent, and dumps:
  config           — JSON metadata string from the safetensors header
  input_latent     — BF16 [B, 128, F, H, W] seeded random
  output_latent    — BF16 [B, 128, F, 2H, 2W] from LatentUpsampler

The Rust port should then load `input_latent`, feed it through the Rust
LatentUpsampler forward, and compare against `output_latent`. Pass
criterion: cos_sim ≥ 0.999, max_abs under BF16 noise.

No reconstructions — the class source is Lightricks's own.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

LTX_REPO = Path("/tmp/ltx-video")
sys.path.insert(0, str(LTX_REPO))

from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler  # noqa: E402

CKPT = Path(
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
)
OUT = Path("/home/alex/EriDiffusion/inference-flame/output/ltx2_latent_upsampler_ref.safetensors")

# Small deterministic input — exercises all layers without OOM during
# the Rust port's parity run.
SEED = 42
SHAPE = (1, 128, 2, 8, 8)  # [B, C, F, H, W] — F=2 frames, 8x8 spatial -> 16x16 out


def read_config(path: Path) -> dict:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode("utf-8"))
    meta = header.get("__metadata__") or {}
    cfg = meta.get("config")
    if cfg is None:
        raise RuntimeError(f"no config in __metadata__ of {path}")
    return json.loads(cfg)


def main() -> int:
    cfg = read_config(CKPT)
    print(f"config: {cfg}")

    # Instantiate LatentUpsampler from Lightricks's class.
    model = LatentUpsampler(
        in_channels=cfg["in_channels"],
        mid_channels=cfg["mid_channels"],
        num_blocks_per_stage=cfg["num_blocks_per_stage"],
        dims=cfg["dims"],
        spatial_upsample=cfg["spatial_upsample"],
        temporal_upsample=cfg["temporal_upsample"],
    )
    state = load_file(CKPT, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"  first missing: {missing[:3]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}")
    model = model.to(dtype=torch.bfloat16, device="cuda").eval()

    # Seeded input — same seed → same values so Rust can reproduce
    # via its own RNG using the same algorithm.
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(*SHAPE, generator=g, dtype=torch.float32).to(torch.bfloat16).cuda()

    with torch.no_grad():
        y = model(x)

    print(f"input  shape: {tuple(x.shape)} dtype={x.dtype}")
    print(f"output shape: {tuple(y.shape)} dtype={y.dtype}")
    print(f"input  range: [{x.float().min():.3f}, {x.float().max():.3f}]  mean|.|={x.float().abs().mean():.4f}")
    print(f"output range: [{y.float().min():.3f}, {y.float().max():.3f}]  mean|.|={y.float().abs().mean():.4f}")

    out = {
        "input_latent": x.contiguous().cpu(),
        "output_latent": y.contiguous().cpu(),
        # Dump the config as a safetensors-friendly tensor (list of utf-8 bytes
        # as int32) so the Rust side can verify it read the same one.
        "_config_utf8": torch.tensor(
            [b for b in json.dumps(cfg).encode("utf-8")], dtype=torch.int32
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT)
    print(f"\nwrote {OUT}")
    for k, v in out.items():
        print(f"  {k:<18} shape={tuple(v.shape)} dtype={v.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
