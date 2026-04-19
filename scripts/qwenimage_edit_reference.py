#!/usr/bin/env python
"""Qwen-Image-Edit-2511 diffusers-reference denoise loop for parity diff.

Loads the SAME cached embeds and noise that Rust's `qwenimage_edit_gen`
would use (when Rust was run with QWEN_DUMP_DIR set), runs the 2511
diffusers denoise loop inline, and writes per-step tensor dumps with
identical keys to Rust's dump so `qwen_edit_diff.py` can compare them.

Inputs (all in ${DUMP_DIR}):
    embeds.safetensors    # copied from Rust encode step (cond, uncond, image_latents, image_h, image_w)
    noise.safetensors     # dumped by Rust (noise_packed shape [1, seq, 64])

Outputs (written to ${DUMP_DIR}/py/):
    step_000.safetensors, step_001.safetensors, ...
      keys: latents_in, cond_pred_full, uncond_pred_full,
            noise_pred_cfg, latents_out, sigma_curr, sigma_next

Usage:
    python qwenimage_edit_reference.py <dump_dir> [--steps N] [--cfg F]

Assumes QwenImageEditPlusPipeline is importable (diffusers ≥ 0.37).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

SNAP = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/"
    "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
)


def calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    # exponential time-shift helper (matches diffusers calculate_shift)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=str)
    ap.add_argument("--embeds", type=str,
                    default="/home/alex/EriDiffusion/inference-flame/output/qwenimage_edit_embeds.safetensors",
                    help="Path to cached embeddings (from qwenimage_edit_encode.py)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--snap", type=str, default=SNAP)
    ap.add_argument("--disable-zero-cond-t", action="store_true",
                    help="Force transformer.zero_cond_t = False (diagnostic)")
    ap.add_argument("--with-mask", action="store_true",
                    help="Pass all-ones encoder_hidden_states_mask (default: None, matches Rust)")
    args = ap.parse_args()

    dd = Path(args.dump_dir)
    out_dir = dd / "py"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== Qwen-Image-Edit-2511 diffusers reference denoise ===")
    print(f"  dump_dir: {dd}")
    print(f"  out:      {out_dir}")
    print(f"  steps:    {args.steps}   cfg: {args.cfg}")
    print()

    # --- Load embeds from the Rust-compatible safetensors ---
    embeds_path = Path(args.embeds)
    if not embeds_path.exists():
        print(f"[error] missing {embeds_path}")
        return 2
    embeds = load_file(embeds_path)
    cond = embeds["cond"].to(device=device, dtype=dtype)
    uncond = embeds["uncond"].to(device=device, dtype=dtype)
    image_latents = embeds["image_latents"].to(device=device, dtype=dtype)
    image_h = int(embeds["image_h"].item())
    image_w = int(embeds["image_w"].item())
    print(f"  cond:          {tuple(cond.shape)}")
    print(f"  uncond:        {tuple(uncond.shape)}")
    print(f"  image_latents: {tuple(image_latents.shape)}")
    print(f"  image size:    {image_w}x{image_h}")

    # --- Load noise (same bytes Rust is using) ---
    noise_path = dd / "noise.safetensors"
    if not noise_path.exists():
        print(f"[error] missing {noise_path} — run Rust with QWEN_DUMP_DIR=<dir> first")
        return 2
    noise = load_file(noise_path)["noise_packed"].to(device=device, dtype=dtype)
    print(f"  noise_packed:  {tuple(noise.shape)}")

    # --- Transformer (CPU-offloaded for 24 GB budget) ---
    # The 20B bf16 transformer is ~40 GB; won't fit. Use accelerate sequential
    # dispatch: keep most weights on CPU (pinned), move per-layer to GPU during
    # forward. Slow but correct — we only need a few forwards for parity.
    print("\n--- Loading transformer (sequential CPU offload) ---")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        Path(args.snap) / "transformer",
        torch_dtype=dtype,
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "60GiB"},
    ).eval()
    print("  loaded")
    for p in transformer.parameters():
        p.requires_grad_(False)

    if args.disable_zero_cond_t:
        print(f"  [diag] forcing transformer.zero_cond_t = False (was {transformer.zero_cond_t})")
        transformer.zero_cond_t = False
        for block in transformer.transformer_blocks:
            block.zero_cond_t = False

    # --- Scheduler (match 2511's FlowMatchEulerDiscreteScheduler config) ---
    sched_path = Path(args.snap) / "scheduler" / "scheduler_config.json"
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(Path(args.snap) / "scheduler")
    print(f"  scheduler: base_shift={scheduler.config.base_shift} max_shift={scheduler.config.max_shift} "
          f"base_seq={scheduler.config.base_image_seq_len} max_seq={scheduler.config.max_image_seq_len} "
          f"shift_terminal={scheduler.config.shift_terminal}")

    # --- Geometry for RoPE (img_shapes) ---
    # target + single reference, both at same dims as the 2511 edit
    vae_scale_factor = 8
    patch = 2
    h_patched = image_h // vae_scale_factor // patch
    w_patched = image_w // vae_scale_factor // patch
    img_shapes = [
        [
            (1, h_patched, w_patched),
            (1, h_patched, w_patched),
        ]
    ]
    target_seq_len = h_patched * w_patched

    # --- Initial latents = noise ---
    latents = noise.clone()

    # --- Sigma schedule ---
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = (lambda: None, None)  # placeholder, reset below

    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, args.steps, device, sigmas=sigmas, mu=mu,
    )
    print(f"  mu={mu:.4f}  exp(mu)={math.exp(mu):.4f}  N={num_inference_steps}")
    print(f"  t[0]={timesteps[0].item():.4f}  t[-1]={timesteps[-1].item():.4f}")

    # --- Encoder-hidden-states masks ---
    # Default: pass None so the Attention path skips joint_attention_mask
    # (matches Rust which calls sdpa(q,k,v,None)). Override with
    # --with-mask to pass all-ones and see if that changes anything.
    if args.with_mask:
        cond_mask = torch.ones((cond.shape[0], cond.shape[1]), dtype=torch.bool, device=device)
        uncond_mask = torch.ones((uncond.shape[0], uncond.shape[1]), dtype=torch.bool, device=device)
    else:
        cond_mask = None
        uncond_mask = None

    # --- Denoise loop (inline from pipeline_qwenimage_edit_plus.__call__) ---
    print("\n--- Denoising ---")
    scheduler.set_begin_index(0)
    for i, t in enumerate(timesteps):
        sigma_curr = sigmas[i] if i < len(sigmas) else 0.0
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0

        latents_in_snap = latents.clone()

        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(dtype) / 1000.0

        with torch.no_grad():
            cond_pred_full = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states_mask=cond_mask,
                encoder_hidden_states=cond,
                img_shapes=img_shapes,
                attention_kwargs={},
                return_dict=False,
            )[0]
            uncond_pred_full = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states_mask=uncond_mask,
                encoder_hidden_states=uncond,
                img_shapes=img_shapes,
                attention_kwargs={},
                return_dict=False,
            )[0]

        # Slice target portion.
        cond_pred = cond_pred_full[:, :target_seq_len, :]
        uncond_pred = uncond_pred_full[:, :target_seq_len, :]

        comb_pred = uncond_pred + args.cfg * (cond_pred - uncond_pred)
        cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)

        prev_sample = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latents = prev_sample

        tmap = {
            "latents_in": latents_in_snap.contiguous().cpu(),
            "cond_pred_full": cond_pred_full.contiguous().cpu(),
            "uncond_pred_full": uncond_pred_full.contiguous().cpu(),
            "noise_pred_cfg": noise_pred.contiguous().cpu(),
            "latents_out": latents.contiguous().cpu(),
            "sigma_curr": torch.tensor([float(sigma_curr)], dtype=torch.float32),
            "sigma_next": torch.tensor([float(sigma_next)], dtype=torch.float32),
        }
        save_file(tmap, str(out_dir / f"step_{i:03d}.safetensors"))

        if i == 0 or (i + 1) % 10 == 0 or i + 1 == args.steps:
            print(f"  step {i + 1}/{args.steps}  t={t.item():.4f}  sigma={sigma_curr:.4f}")

    print(f"\nDONE. Dumps at {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
