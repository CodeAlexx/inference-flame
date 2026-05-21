#!/usr/bin/env python3
"""
Generate a GPU-side parity fixture for the Cosmos UniPC scheduler.

Runs `FlowUniPCMultistepScheduler` with the Cosmos V2_2B defaults
(`num_train_timesteps=1000`, `solver_order=2`, `predict_x0=True`,
`solver_type='bh2'`, `shift=5.0`, `lower_order_final=True`,
`final_sigmas_type='zero'`) for `num_inference_steps=10`, executes ONE
`step` call with deterministic random inputs (torch.manual_seed(0),
`torch.randn` on CUDA), and writes the result to
`cosmos_unipc_step_ref.safetensors`.

The Rust parity test loads this fixture and asserts cos >= 0.9999 against
the Python `prev_sample`.

CRITICAL: must be GPU-generated. The per-port CONTEXT.md forbids CPU parity
refs (PyTorch CPU vs CUDA BF16 diverges per-layer at cos=0.5).

Usage:
    cd /home/alex/EriDiffusion/inference-flame/ports/cosmos-predict25-2b/parity
    python3 cosmos_unipc_step_ref.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# Load Cosmos's FlowUniPCMultistepScheduler from the source clone.
# We bypass `cosmos_predict2/__init__.py`'s CUDA-extra guard by loading
# `fm_solvers_unipc.py` directly via importlib — it only depends on
# `numpy`, `torch`, and `diffusers.schedulers`.
import importlib.util  # noqa: E402

SCHEDULER_FILE = Path(
    "/home/alex/refs/cosmos-predict2.5/cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py"
)
spec = importlib.util.spec_from_file_location("cosmos_unipc_ref", SCHEDULER_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load {SCHEDULER_FILE}")
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)
FlowUniPCMultistepScheduler = _module.FlowUniPCMultistepScheduler


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; this script must run on GPU.", file=sys.stderr)
        return 1
    device = torch.device("cuda:0")
    dtype = torch.float32  # F32 throughout — the Rust scheduler is dtype-generic.

    # Deterministic inputs.
    torch.manual_seed(0)
    # Use a small but non-trivial latent. Shape arbitrary; we just need
    # parity on a single `step` call. Use [1, 4, 4, 4] for the example
    # (4-D, like an image latent BCHW). Cosmos's actual latents are 5D
    # [B,C,T,H,W] but the scheduler is shape-agnostic — `step` doesn't
    # care.
    shape = (1, 4, 4, 4)
    sample = torch.randn(shape, device=device, dtype=dtype)
    model_output = torch.randn(shape, device=device, dtype=dtype)

    # Build the scheduler.
    sch = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        shift=1.0,  # __init__ default; set_timesteps overrides
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        disable_corrector=[],
        final_sigmas_type="zero",
    )
    sch.set_timesteps(num_inference_steps=10, device=device, shift=5.0)

    sigmas = sch.sigmas.to(dtype=torch.float32).cpu().contiguous()

    # Run one step (step_index advances from None → 0 internally on first call).
    # Need to pass a `timestep` arg matching sch.timesteps[0] — Python uses it
    # to initialize step_index via `_init_step_index`.
    ts0 = sch.timesteps[0].item() if hasattr(sch.timesteps[0], "item") else sch.timesteps[0]
    out = sch.step(model_output=model_output, timestep=int(ts0), sample=sample)
    prev_sample = out.prev_sample.contiguous()

    # Save the fixture.
    output_path = Path(__file__).parent / "cosmos_unipc_step_ref.safetensors"
    save_file(
        {
            "sample": sample.detach().cpu().contiguous(),
            "model_output": model_output.detach().cpu().contiguous(),
            "prev_sample": prev_sample.detach().cpu().contiguous(),
            "sigmas": sigmas,
        },
        str(output_path),
    )
    print(f"Wrote fixture to {output_path}")
    print(f"  shape: {tuple(prev_sample.shape)}")
    print(f"  sigmas[0]={sigmas[0].item():.6f} sigmas[-1]={sigmas[-1].item():.6f}")
    print(f"  num_inference_steps=10, shift=5.0, solver_order=2")
    print(f"  sample[0,0,0,:4]={sample[0,0,0,:4].cpu().tolist()}")
    print(f"  prev[0,0,0,:4]={prev_sample[0,0,0,:4].cpu().tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
