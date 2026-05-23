#!/usr/bin/env python3
"""T3-lite — Training step loss parity (Python vs Rust, NO LoRA).

The goal: given the same (noisy, target, sigma) triple and the same base
L2P weights, do Rust and Python compute the same loss?

This test isolates the training math (rectified-flow target + MSE loss)
from autograd correctness. Since LoRA-B starts at zero, step-0 forward is
purely the base model. If Python loss == Rust loss to BF16-floor precision,
then:
  - target formula matches Python (noise - clean, not clean - noise)
  - loss formula matches (MSE in F32)
  - the model's forward pred matches (already verified by T2 forward parity)

What it does NOT test:
  - LoRA injection arithmetic
  - autograd backward pass
  - optimizer step

Approach:
  1. Python: load reference DiT, set up SAME pixel/cap_feats/noise/sigma as
     Rust trainer would see at step 0, compute the loss, dump to safetensors.
  2. Rust: a NEW small binary `l2p_loss_parity` does the same math, but
     with shared inputs read from a safetensors file. Diff the loss.

This script only writes the PYTHON SIDE outputs. The Rust side requires
the binary to be built first.

Inputs are dumped to:
  /tmp/l2p_thorough_parity/t3_shared_inputs.safetensors
    - clean   : [1, 3, 512, 512] BF16
    - noise   : [1, 3, 512, 512] BF16
    - cap_feats: [1, S, 2560]   BF16
    - sigma   : [1]              F32

Python outputs:
  /tmp/l2p_thorough_parity/t3_python_outputs.safetensors
    - noisy   : [1, 3, 512, 512] F32
    - target  : [1, 3, 512, 512] F32
    - pred    : [1, 3, 512, 512] F32   (= -DiT(noisy, t=sigma))
    - loss    : []                F32
"""
import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

# Force torch SDPA path.
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import torch
import safetensors.torch as st


REPO_ROOT = Path(__file__).resolve().parents[1] / "reference"
WEIGHTS_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
SAMPLE_PATH = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512/10.safetensors"
OUT_DIR = Path("/tmp/l2p_thorough_parity")


def _load_zimage_dit_module():
    sys.path.insert(0, str(REPO_ROOT))

    diffsynth_pkg = types.ModuleType("diffsynth")
    diffsynth_pkg.__path__ = [str(REPO_ROOT / "diffsynth")]
    sys.modules["diffsynth"] = diffsynth_pkg
    core_pkg = types.ModuleType("diffsynth.core")
    core_pkg.__path__ = [str(REPO_ROOT / "diffsynth" / "core")]
    sys.modules["diffsynth.core"] = core_pkg
    models_pkg = types.ModuleType("diffsynth.models")
    models_pkg.__path__ = [str(REPO_ROOT / "diffsynth" / "models")]
    sys.modules["diffsynth.models"] = models_pkg

    def load_submod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    attn_path = REPO_ROOT / "diffsynth" / "core" / "attention" / "attention.py"
    grad_path = REPO_ROOT / "diffsynth" / "core" / "gradient" / "gradient_checkpoint.py"
    attn_pkg = types.ModuleType("diffsynth.core.attention")
    attn_pkg.__path__ = [str(attn_path.parent)]
    sys.modules["diffsynth.core.attention"] = attn_pkg
    attn_mod = load_submod("diffsynth.core.attention.attention", attn_path)
    attn_pkg.attention_forward = attn_mod.attention_forward
    grad_pkg = types.ModuleType("diffsynth.core.gradient")
    grad_pkg.__path__ = [str(grad_path.parent)]
    sys.modules["diffsynth.core.gradient"] = grad_pkg
    grad_mod = load_submod("diffsynth.core.gradient.gradient_checkpoint", grad_path)
    grad_pkg.gradient_checkpoint_forward = grad_mod.gradient_checkpoint_forward

    dit_path = REPO_ROOT / "diffsynth" / "models" / "z_image_dit_L2P.py"
    return load_submod("diffsynth.models.z_image_dit_L2P", dit_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    # 1. Load sample.
    print(f"[T3] loading sample {SAMPLE_PATH}")
    sample = st.load_file(SAMPLE_PATH, device="cuda")
    clean = sample["pixel"].to(torch.bfloat16)  # [3, 512, 512]
    cap_feats = sample["cap_feats"].to(torch.bfloat16)  # [1, S, 2560]
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)  # [1, 3, 512, 512]
    if cap_feats.ndim == 2:
        cap_feats = cap_feats.unsqueeze(0)

    # 2. Generate fixed noise via CUDA generator.
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(clean.shape, generator=g, device=device, dtype=torch.bfloat16)

    # 3. Compute noisy + target.
    sigma = float(args.sigma)
    noisy = (1.0 - sigma) * clean + sigma * noise
    target = noise - clean  # rectified-flow training target

    print(f"[T3] clean: {tuple(clean.shape)} {clean.dtype}")
    print(f"[T3] noise: {tuple(noise.shape)} {noise.dtype}")
    print(f"[T3] noisy: {tuple(noisy.shape)} {noisy.dtype}")
    print(f"[T3] target: {tuple(target.shape)} {target.dtype}")
    print(f"[T3] sigma: {sigma}")

    # 4. Dump shared inputs for Rust to read.
    shared = {
        "clean": clean.cpu().contiguous(),
        "noise": noise.cpu().contiguous(),
        "cap_feats": cap_feats.cpu().contiguous(),
        "sigma": torch.tensor([sigma], dtype=torch.float32),
    }
    st.save_file(shared, str(OUT_DIR / "t3_shared_inputs.safetensors"))
    print(f"[T3] wrote shared inputs to {OUT_DIR / 't3_shared_inputs.safetensors'}")

    # 5. Load Python reference DiT + run forward.
    print(f"[T3] loading reference module")
    dit_mod = _load_zimage_dit_module()
    ZImageDiT = dit_mod.ZImageDiT
    print(f"[T3] instantiating + loading weights")
    model = ZImageDiT()
    weights = st.load_file(WEIGHTS_PATH)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"[T3] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # The Python L2P forward expects:
    #   x: List[Tensor[C, F, H, W]] with F=1, t scaled by 1000 internally,
    #   cap_feats: List[Tensor[S, 2560]]
    x_4d = noisy.squeeze(0).unsqueeze(1)  # [3, 1, 512, 512]
    x_list = [x_4d]
    cap_list = [cap_feats.squeeze(0)]
    # `t` arrives in [0,1]; the DiT does `t * t_scale = t * 1000` internally.
    t_tensor = torch.tensor([sigma], device=device, dtype=torch.bfloat16)

    print(f"[T3] running forward...")
    with torch.no_grad():
        out_list, _ = model(x_list, t_tensor, cap_list)
    # out_list[0] is the DiT raw output (no sign flip). The Python pipeline
    # applies `* -1` in `model_fn_z_image` (see z_image_L2P.py:329).
    # Reference: model_fn_z_image returns `-dit_out`. Our pred should match.
    pred_dit = out_list[0]  # [3, 1, 512, 512]
    if pred_dit.ndim == 4:
        pred_dit = pred_dit.unsqueeze(0)  # add batch
    elif pred_dit.ndim == 5:
        pass  # already [B, 3, 1, 512, 512]
    # Squeeze frame axis.
    pred_dit = pred_dit.squeeze(2)  # [1, 3, 512, 512]
    pred = -pred_dit  # Python pipeline applies the negation in model_fn_z_image

    # 6. Compute loss.
    diff = pred.to(torch.float32) - target.to(torch.float32)
    loss = (diff * diff).mean()
    loss_val = loss.item()
    print(f"[T3-python] loss = {loss_val:.6f}")

    # 7. Save outputs.
    out = {
        "noisy": noisy.to(torch.float32).cpu().contiguous(),
        "target": target.to(torch.float32).cpu().contiguous(),
        "pred": pred.to(torch.float32).cpu().contiguous(),
        "loss": loss.to(torch.float32).reshape([]).cpu(),
    }
    st.save_file(out, str(OUT_DIR / "t3_python_outputs.safetensors"))
    print(f"[T3] wrote Python outputs to {OUT_DIR / 't3_python_outputs.safetensors'}")

    print()
    print(f"[T3-python] sigma={sigma} → loss={loss_val:.6f}")
    print("[T3-python] PASS — next step: build/run Rust loss-parity binary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
