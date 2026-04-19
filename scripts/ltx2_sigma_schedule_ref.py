#!/usr/bin/env python
"""Generate LTX-2 timestep/sigma reference data by IMPORTING Lightricks's
actual code, not reconstructing it. No shortcuts, no fabricated formulas.

Sources (cloned shallow into /tmp/ltx-video):
  - ltx_video/schedulers/rf.py                              → schedule fns
  - configs/ltxv-13b-0.9.8-distilled.yaml                   → distilled timesteps
  - ltxv-13b-0.9.8-distilled.safetensors (`allowed_inference_steps`
    in __metadata__)                                       → authoritative list

Emits: output/ltx2_sigma_ref.safetensors
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

LTX_REPO = Path("/tmp/ltx-video")
sys.path.insert(0, str(LTX_REPO))

import yaml  # noqa: E402

# Pull the actual Lightricks code — anything the Rust side should match
# comes from here, not a reconstruction.
from ltx_video.schedulers.rf import (  # noqa: E402
    linear_quadratic_schedule,
    RectifiedFlowScheduler,
)

DISTILLED_CKPT = Path(
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
)
DISTILLED_YAML = LTX_REPO / "configs" / "ltxv-13b-0.9.8-distilled.yaml"
DEV_YAML = LTX_REPO / "configs" / "ltxv-13b-0.9.8-dev.yaml"
OUT = Path("/home/alex/EriDiffusion/inference-flame/output/ltx2_sigma_ref.safetensors")


def read_allowed_inference_steps(path: Path) -> list[float] | None:
    """Parse safetensors __metadata__ for `allowed_inference_steps`."""
    if not path.exists():
        print(f"  (skipping: {path} not on disk)")
        return None
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
    header = json.loads(header_json)
    meta = header.get("__metadata__") or {}
    raw = meta.get("allowed_inference_steps")
    if raw is None:
        print(f"  (no allowed_inference_steps in {path.name} — metadata keys: {list(meta.keys())})")
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = json.loads(raw.replace("(", "[").replace(")", "]"))
    return [float(x) for x in parsed]


def main() -> int:
    out: dict[str, torch.Tensor] = {}

    # --- 1. Authoritative list from safetensors metadata -----------------
    print("[1/4] Reading allowed_inference_steps from distilled checkpoint...")
    allowed = read_allowed_inference_steps(DISTILLED_CKPT)
    if allowed is not None:
        out["distilled_checkpoint_timesteps"] = torch.tensor(allowed, dtype=torch.float32)
        print(f"       len={len(allowed)}  values={allowed}")

    # --- 2. Distilled yaml first_pass / second_pass ---------------------
    print("[2/4] Reading distilled yaml first/second pass timesteps...")
    with DISTILLED_YAML.open() as f:
        dcfg = yaml.safe_load(f)
    fp = dcfg["first_pass"]["timesteps"]
    sp = dcfg["second_pass"]["timesteps"]
    print(f"       first_pass.timesteps  (len={len(fp)}): {fp}")
    print(f"       second_pass.timesteps (len={len(sp)}): {sp}")
    out["distilled_first_pass_timesteps"] = torch.tensor(fp, dtype=torch.float32)
    out["distilled_second_pass_timesteps"] = torch.tensor(sp, dtype=torch.float32)

    # --- 3. linear_quadratic_schedule (Lightricks's actual fn) ----------
    # Run their function at a few step counts the bins use.
    print("[3/4] Running Lightricks's linear_quadratic_schedule...")
    for n in (8, 20, 25, 30):
        sched = linear_quadratic_schedule(n, threshold_noise=0.025).tolist()
        out[f"linear_quadratic_{n}"] = torch.tensor(sched, dtype=torch.float32)
        print(f"       n={n}: first 4 = {sched[:4]}  last 4 = {sched[-4:]}")

    # --- 4. RectifiedFlowScheduler.set_timesteps full-pipeline shape ----
    # This is what pipeline_ltx_video.py actually calls. Do it in a mode
    # the dev yaml uses.
    print("[4/4] Running RectifiedFlowScheduler.set_timesteps...")

    # Dev-mode defaults (per configs/ltxv-13b-0.9.8-dev.yaml):
    #   sampler: LinearQuadratic, shifting: null, num_train_timesteps: 1000
    scheduler = RectifiedFlowScheduler(
        sampler="LinearQuadratic",
        num_train_timesteps=1000,
        shifting=None,
    )
    # set_timesteps computes sigmas; samples_shape is used only when
    # shifting!=None (SD3 or SimpleDiffusion), so default tensor works.
    # Same call pipeline_ltx_video.py makes. Picking 30 to match dev yaml.
    scheduler.set_timesteps(num_inference_steps=30, samples_shape=torch.Size([1, 128, 1, 8, 8]), device="cpu")
    dev_timesteps = scheduler.timesteps.cpu().float()
    dev_sigmas = scheduler.sigmas.cpu().float()
    print(f"       dev 30-step: timesteps.shape={tuple(dev_timesteps.shape)}  "
          f"first 4 ts={dev_timesteps[:4].tolist()}  last 4 ts={dev_timesteps[-4:].tolist()}")
    print(f"                     sigmas.shape={tuple(dev_sigmas.shape)}")
    out["rf_dev_30_timesteps"] = dev_timesteps.clone()
    out["rf_dev_30_sigmas"] = dev_sigmas.clone()

    # Uniform sampler path (for completeness) — some legacy binaries use it.
    uniform = RectifiedFlowScheduler(sampler="Uniform", num_train_timesteps=1000, shifting=None)
    uniform.set_timesteps(num_inference_steps=30, samples_shape=torch.Size([1, 128, 1, 8, 8]), device="cpu")
    out["rf_uniform_30_sigmas"] = uniform.sigmas.cpu().float()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT)
    print(f"\nwrote {OUT}")
    for k, v in out.items():
        print(f"  {k:<40} shape={tuple(v.shape)} dtype={v.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
