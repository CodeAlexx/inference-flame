#!/usr/bin/env python3
"""T1b — Python denoise loop using diffsynth L2P DiT.

Runs the full 30-step Euler + CFG denoise pipeline with the cap_feats /
cap_feats_uncond pre-saved by t1a_encode.py. Saves a PNG.

Noise is generated with torch.randn(generator=cuda, seed) — this is
KNOWN to differ from Rust's Box-Muller (T11). So the resulting PNG will
NOT be pixel-equal to the Rust PNG; we only compare statistics + visual
plausibility.
"""
import argparse
import importlib.util
import os
import sys
import time
import types
from pathlib import Path

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import numpy as np
import torch
import safetensors.torch as st
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1] / "reference"
WEIGHTS = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
EMB = "/tmp/l2p_thorough_parity/T1/embeddings.safetensors"
OUT_PNG = "/tmp/l2p_thorough_parity/T1/python_output.png"


def _load_dit_module():
    sys.path.insert(0, str(REPO_ROOT))
    diffsynth_pkg = types.ModuleType("diffsynth")
    diffsynth_pkg.__path__ = [str(REPO_ROOT / "diffsynth")]
    sys.modules["diffsynth"] = diffsynth_pkg
    for sub in ("core", "models"):
        m = types.ModuleType(f"diffsynth.{sub}")
        m.__path__ = [str(REPO_ROOT / "diffsynth" / sub)]
        sys.modules[f"diffsynth.{sub}"] = m

    def load_submod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    attn_path = REPO_ROOT / "diffsynth" / "core" / "attention" / "attention.py"
    attn_pkg = types.ModuleType("diffsynth.core.attention")
    attn_pkg.__path__ = [str(attn_path.parent)]
    sys.modules["diffsynth.core.attention"] = attn_pkg
    attn_mod = load_submod("diffsynth.core.attention.attention", attn_path)
    attn_pkg.attention_forward = attn_mod.attention_forward

    grad_path = REPO_ROOT / "diffsynth" / "core" / "gradient" / "gradient_checkpoint.py"
    grad_pkg = types.ModuleType("diffsynth.core.gradient")
    grad_pkg.__path__ = [str(grad_path.parent)]
    sys.modules["diffsynth.core.gradient"] = grad_pkg
    grad_mod = load_submod("diffsynth.core.gradient.gradient_checkpoint", grad_path)
    grad_pkg.gradient_checkpoint_forward = grad_mod.gradient_checkpoint_forward

    dit_path = REPO_ROOT / "diffsynth" / "models" / "z_image_dit_L2P.py"
    return load_submod("diffsynth.models.z_image_dit_L2P", dit_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")

    print(f"[T1b] loading embeddings from {EMB}")
    emb = st.load_file(EMB, device="cuda")
    cap_feats = emb["cap_feats"].to(device=device, dtype=torch.bfloat16)
    cap_feats_uncond = emb["cap_feats_uncond"].to(device=device, dtype=torch.bfloat16)
    print(f"  cap_feats: {tuple(cap_feats.shape)}")
    print(f"  cap_feats_uncond: {tuple(cap_feats_uncond.shape)}")

    print(f"[T1b] loading reference DiT")
    dit_mod = _load_dit_module()
    model = dit_mod.ZImageDiT()
    sd = st.load_file(WEIGHTS)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    def flux_sigmas(n, shift=3.0):
        sigmas = torch.linspace(1, 0, n + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        return torch.cat([sigmas, torch.zeros(1)])

    sigmas = flux_sigmas(args.steps, 3.0).to(device, dtype=torch.bfloat16)
    print(f"[T1b] sigma range [{sigmas[0].item():.4f}, {sigmas[-1].item():.4f}]")

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randn(
        (1, 3, args.height, args.width),
        generator=g, device=device, dtype=torch.bfloat16,
    )
    print(f"[T1b] initial noise: abs.mean={x.abs().mean().item():.4f}")

    cap_list = [cap_feats.squeeze(0)]
    cap_list_uncond = [cap_feats_uncond.squeeze(0)]

    print(f"[T1b] denoise {args.steps} steps, cfg={args.cfg}")
    t0 = time.time()
    for i in range(args.steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        x_list = [x.squeeze(0).unsqueeze(1)]  # [3, 1, H, W]
        # Python pipeline convention (z_image_L2P.py:319):
        #   model_fn does `timestep = (1000 - timestep) / 1000` BEFORE the DiT,
        #   so the DiT receives (1 - sigma). The DiT internally multiplies by
        #   t_scale=1000 → final t embedding input is (1-sigma)*1000.
        # Rust convention (l2p_sampling.rs comment): l2p_euler_step passes
        # `sigma` directly; the DiT does `(1-sigma)*1000` internally. Both
        # land at the same t_embed input — but the inversion happens in
        # different places.
        t_in = (1.0 - sigma).reshape(1).to(dtype=torch.bfloat16)
        with torch.no_grad():
            pred_cond_list, _ = model(x_list, t_in, cap_list)
            pred_cond = pred_cond_list[0].squeeze(1).unsqueeze(0)
            pred_cond = -pred_cond  # pipeline sign flip
            if args.cfg > 1.0:
                pred_uncond_list, _ = model(x_list, t_in, cap_list_uncond)
                pred_uncond = pred_uncond_list[0].squeeze(1).unsqueeze(0)
                pred_uncond = -pred_uncond
                pred = pred_uncond + args.cfg * (pred_cond - pred_uncond)
            else:
                pred = pred_cond
            dsigma = (sigma_next - sigma).to(pred.dtype)
            x = x + dsigma * pred
        if (i + 1) % 5 == 0 or i == args.steps - 1:
            print(f"  step {i+1}/{args.steps}  x.abs.mean={x.abs().mean().item():.4f}")
    print(f"[T1b] denoise done in {time.time()-t0:.1f}s")

    img = x.squeeze(0).float().cpu().clamp(-1, 1)
    img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    img_np = img.permute(1, 2, 0).numpy()
    Image.fromarray(img_np).save(OUT_PNG)
    print(f"[T1b] saved {OUT_PNG}")
    print(f"  stats: shape={img_np.shape}, mean={img_np.astype(float).mean()/255.0:.4f}, std={img_np.astype(float).std()/255.0:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
