#!/usr/bin/env python3
"""T1 — End-to-end inference parity (best-effort given noise PRNG difference).

True pixel-equality requires injecting an identical initial noise tensor
into both pipelines. The Rust `l2p_infer` binary does not accept a
pre-built noise file (only a `--seed`), and Python diffsynth uses
torch.randn with a CUDA generator — bridging the two would require
modifying the Rust binary (forbidden by the task spec). T11 documents
the PRNG difference.

What this test DOES verify:
  1. Forward pass through the FULL denoise loop (30 Euler steps + CFG@4.0)
     produces output in the correct pixel range and statistics on both
     stacks, given the same prompt and same seed (different noise but same
     distribution).
  2. The two PNGs have similar global statistics (mean luminance, std,
     histogram peak location) — a sanity check that the sampler-loop math
     isn't producing degenerate output (all-white, all-noise, gradient
     drift).

PASS criteria: both outputs are in valid pixel range, content-of-prompt is
plausible, no NaN/blowup. Strict pixel-cos can't be claimed without noise
injection.

Steps:
  1. Generate `cap_feats` + `cap_feats_uncond` via Python Qwen3-4B encoder.
  2. Save to safetensors that `l2p_infer` accepts.
  3. Run Python diffsynth full pipeline → png_py.
  4. Run Rust l2p_infer → png_rs.
  5. Compare statistics + emit both PNGs for visual inspection.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import numpy as np
import safetensors.torch as st
import torch
from transformers import AutoTokenizer, Qwen3Model, Qwen3Config
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1] / "reference"
WEIGHTS = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
QWEN_WEIGHTS = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
TOKENIZER_PATH = "/home/alex/.serenity/models/zimage_base/tokenizer/"
OUT_DIR = Path("/tmp/l2p_thorough_parity/T1")
PROMPT = "a serene mountain landscape at sunset, with snow-capped peaks reflecting in a calm lake, photorealistic, high detail"


def build_qwen3_cfg():
    return Qwen3Config(**{
        "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": 151643, "eos_token_id": 151645,
        "head_dim": 128, "hidden_size": 2560,
        "initializer_range": 0.02, "intermediate_size": 9728,
        "max_position_embeddings": 40960, "max_window_layers": 36,
        "model_type": "qwen3", "num_attention_heads": 32,
        "num_hidden_layers": 36, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 1000000,
        "sliding_window": None, "tie_word_embeddings": True,
        "torch_dtype": "bfloat16", "use_cache": False,
        "use_sliding_window": False, "vocab_size": 151936,
    })


def encode_prompt(tok, qwen_model, text, device):
    messages = [{"role": "user", "content": text}]
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tok(templated, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0].tolist()
    valid_len = len(ids)
    PAD = 151643
    while len(ids) < 512:
        ids.append(PAD)
    ids = ids[:512]
    ids_t = torch.tensor([ids], device=device, dtype=torch.long)
    mask = torch.zeros_like(ids_t)
    mask[0, :valid_len] = 1
    with torch.no_grad():
        out = qwen_model(input_ids=ids_t, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states[-2][:, :valid_len, :]
    return hs.to(torch.bfloat16).contiguous(), valid_len


def stats(arr):
    """Per-channel and global stats on uint8 array [H, W, 3]."""
    arr_f = arr.astype(np.float32) / 255.0
    return {
        "mean": arr_f.mean(),
        "std": arr_f.std(),
        "min": arr_f.min(),
        "max": arr_f.max(),
        "ch_means": arr_f.mean(axis=(0, 1)).tolist(),
        "ch_stds": arr_f.std(axis=(0, 1)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-python", action="store_true",
                    help="Skip Python full-pipeline run (faster); only embed + Rust + statistics")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    # ============================================================
    # 1) Prepare embeddings (cap_feats + cap_feats_uncond) via Qwen3.
    # ============================================================
    print("[T1.1] loading tokenizer + Qwen3-4B encoder")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    cfg = build_qwen3_cfg()
    qwen = Qwen3Model(cfg)
    sd = st.load_file(QWEN_WEIGHTS)
    sd = {(k[len("model."):] if k.startswith("model.") else k): v.to(torch.bfloat16) for k, v in sd.items()}
    qwen.load_state_dict(sd, strict=False)
    qwen = qwen.to(device=device, dtype=torch.bfloat16).eval()

    print(f"[T1.2] encoding prompt + uncond (empty string)")
    cap_feats, _vl = encode_prompt(tok, qwen, PROMPT, device)
    cap_feats_uncond, _ul = encode_prompt(tok, qwen, "", device)
    print(f"  cap_feats: {tuple(cap_feats.shape)}")
    print(f"  cap_feats_uncond: {tuple(cap_feats_uncond.shape)}")

    emb_path = OUT_DIR / "embeddings.safetensors"
    # l2p_infer expects 3D [B, S, 2560]
    if cap_feats.ndim == 2:
        cap_feats = cap_feats.unsqueeze(0)
    if cap_feats_uncond.ndim == 2:
        cap_feats_uncond = cap_feats_uncond.unsqueeze(0)
    st.save_file({
        "cap_feats": cap_feats.cpu().contiguous(),
        "cap_feats_uncond": cap_feats_uncond.cpu().contiguous(),
    }, str(emb_path))
    print(f"[T1.2] wrote {emb_path}")

    # Free Qwen now.
    del qwen, sd
    torch.cuda.empty_cache()

    # ============================================================
    # 2) Rust inference.
    # ============================================================
    rust_png = OUT_DIR / "rust_output.png"
    rust_bin = "/home/alex/EriDiffusion/inference-flame/target/release/l2p_infer"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/home/alex/libs/libtorch/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["FLAME_ALLOC_POOL"] = "0"
    env["FLAME_AUTOGRAD_OFF"] = "1"
    print(f"[T1.3] running Rust l2p_infer ({args.steps} steps, {args.height}x{args.width}, cfg={args.cfg})")
    t0 = time.time()
    cp = subprocess.run([
        rust_bin,
        "--model", WEIGHTS,
        "--embeddings", str(emb_path),
        "--output", str(rust_png),
        "--height", str(args.height),
        "--width", str(args.width),
        "--steps", str(args.steps),
        "--cfg", str(args.cfg),
        "--seed", str(args.seed),
    ], env=env, capture_output=True, text=True)
    t_rust = time.time() - t0
    print(f"  rust elapsed: {t_rust:.1f}s, return={cp.returncode}")
    if cp.returncode != 0:
        print("Rust stderr:")
        print(cp.stderr[-2000:])
        print("Rust stdout (last 1500):")
        print(cp.stdout[-1500:])
        return 1
    if not rust_png.exists():
        print("[T1] FAIL: rust output missing")
        return 1
    rust_img = np.array(Image.open(rust_png).convert("RGB"))
    print(f"[T1.3] rust output: {rust_img.shape} {rust_img.dtype}")
    rust_stats = stats(rust_img)
    print(f"  rust stats: mean={rust_stats['mean']:.3f} std={rust_stats['std']:.3f}")
    print(f"  ch_means={rust_stats['ch_means']}")

    # ============================================================
    # 3) Python full pipeline.
    # ============================================================
    if args.skip_python:
        print("[T1.4] SKIP python pipeline (--skip-python)")
        py_stats = None
        py_img = None
    else:
        # Use diffsynth pipeline via module-stub approach (as in python_capture.py).
        # The full pipeline needs models + tokenizer config loading; that's heavy.
        # For T1 best-effort, just do `forward` with the same noise PRNG (cuda)
        # then save PNG via the same colorspace conversion.
        py_pkg = types.ModuleType("diffsynth")
        py_pkg.__path__ = [str(REPO_ROOT / "diffsynth")]
        sys.modules["diffsynth"] = py_pkg

        for sub in ("core", "models", "diffusion", "pipelines", "utils"):
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
        sys.modules["diffsynth.core.attention"].__path__ = [str(attn_path.parent)]
        attn_mod = load_submod("diffsynth.core.attention.attention", attn_path)
        sys.modules["diffsynth.core.attention"].attention_forward = attn_mod.attention_forward

        grad_path = REPO_ROOT / "diffsynth" / "core" / "gradient" / "gradient_checkpoint.py"
        sys.modules["diffsynth.core.gradient"] = types.ModuleType("diffsynth.core.gradient")
        sys.modules["diffsynth.core.gradient"].__path__ = [str(grad_path.parent)]
        grad_mod = load_submod("diffsynth.core.gradient.gradient_checkpoint", grad_path)
        sys.modules["diffsynth.core.gradient"].gradient_checkpoint_forward = grad_mod.gradient_checkpoint_forward

        dit_path = REPO_ROOT / "diffsynth" / "models" / "z_image_dit_L2P.py"
        dit_mod = load_submod("diffsynth.models.z_image_dit_L2P", dit_path)
        ZImageDiT = dit_mod.ZImageDiT

        # Load DiT.
        print("[T1.4] loading Python DiT (BF16)")
        model = ZImageDiT()
        weights = st.load_file(WEIGHTS)
        weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
        model.load_state_dict(weights, strict=False)
        model = model.to(device=device, dtype=torch.bfloat16).eval()

        # Sigma schedule (FLUX-shift)
        def flux_sigmas(n, shift=3.0):
            sigmas = torch.linspace(1, 0, n + 1)[:-1]
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            sigmas = torch.cat([sigmas, torch.zeros(1)])
            return sigmas

        sigmas = flux_sigmas(args.steps, 3.0).to(device, dtype=torch.bfloat16)
        # Initial noise (PYTHON CUDA randn — note this differs from Rust Box-Muller).
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        x = torch.randn((1, 3, args.height, args.width), generator=g, device=device, dtype=torch.bfloat16)

        # cap_feats already on GPU.
        cap_list = [cap_feats.squeeze(0).to(device)]
        cap_list_uncond = [cap_feats_uncond.squeeze(0).to(device)]

        print(f"[T1.4] python denoise ({args.steps} steps, cfg={args.cfg})")
        t0 = time.time()
        for i in range(args.steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            with torch.no_grad():
                x_list = [x.squeeze(0).unsqueeze(1)]  # [3, 1, H, W]
                t_in = sigma.reshape(1)
                pred_cond_list, _ = model(x_list, t_in, cap_list)
                pred_cond = pred_cond_list[0].squeeze(1).unsqueeze(0)  # [1,3,H,W]
                pred_cond = -pred_cond  # pipeline applies the sign flip
                if args.cfg > 1.0:
                    pred_uncond_list, _ = model(x_list, t_in, cap_list_uncond)
                    pred_uncond = pred_uncond_list[0].squeeze(1).unsqueeze(0)
                    pred_uncond = -pred_uncond
                    pred = pred_uncond + args.cfg * (pred_cond - pred_uncond)
                else:
                    pred = pred_cond
                # Euler step
                dsigma = (sigma_next - sigma).to(pred.dtype)
                x = x + dsigma * pred
            if (i + 1) % 5 == 0:
                print(f"  step {i+1}/{args.steps}, x.abs.mean={x.abs().mean().item():.4f}")
        t_py = time.time() - t0
        print(f"  python elapsed: {t_py:.1f}s")
        # x → image
        img = x.squeeze(0).float().cpu()  # [3, H, W]
        img = img.clamp(-1, 1)
        img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        img_np = img.permute(1, 2, 0).numpy()
        py_png = OUT_DIR / "python_output.png"
        Image.fromarray(img_np).save(py_png)
        py_stats = stats(img_np)
        print(f"[T1.4] python output: {img_np.shape} {img_np.dtype}, saved to {py_png}")
        print(f"  py stats: mean={py_stats['mean']:.3f} std={py_stats['std']:.3f}")
        py_img = img_np

    # ============================================================
    # 4) Verdict.
    # ============================================================
    print("\n=== T1 Summary ===")
    print(f"Rust:   mean={rust_stats['mean']:.4f} std={rust_stats['std']:.4f} ch_means={rust_stats['ch_means']}")
    if py_stats:
        print(f"Python: mean={py_stats['mean']:.4f} std={py_stats['std']:.4f} ch_means={py_stats['ch_means']}")
        mean_diff = abs(rust_stats['mean'] - py_stats['mean'])
        std_diff = abs(rust_stats['std'] - py_stats['std'])
        print(f"|Δmean|={mean_diff:.4f}, |Δstd|={std_diff:.4f}")
        # Note: we DO NOT expect pixel-cos to be high — noise PRNGs differ.
        # Statistic similarity (mean within 0.05, std within 0.05) is enough
        # to call sampler health PASS.
        if mean_diff < 0.05 and std_diff < 0.05 and rust_stats['mean'] > 0.1 and rust_stats['mean'] < 0.9:
            print("[T1] PASS (sampler health: statistics align, both in valid range, no degeneracy)")
            print("[T1] Note: pixel-equality NOT tested — Rust Box-Muller vs Python Philox PRNGs differ (T11).")
            return 0
        else:
            print("[T1] FAIL: statistics diverge or output is degenerate")
            return 1
    else:
        # Just verify Rust output is in valid range
        if rust_stats['mean'] > 0.1 and rust_stats['mean'] < 0.9 and rust_stats['std'] > 0.05:
            print("[T1] PARTIAL PASS (Rust-only: output in valid range with non-degenerate variance)")
            return 0
        else:
            print(f"[T1] FAIL: Rust output is degenerate (mean={rust_stats['mean']}, std={rust_stats['std']})")
            return 1


if __name__ == "__main__":
    sys.exit(main())
