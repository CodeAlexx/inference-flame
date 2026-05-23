#!/usr/bin/env python3
"""T7 — Caption encoding parity (Qwen3-4B text encoder).

The Rust prepare_l2p pipeline produces cached `cap_feats` of shape [1, S, 2560]
by:
  1. Building manual chat template: `<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n`
  2. Tokenizing (no special tokens added) — gives 204 ids for our boxjana #10 sample
  3. Padding to 512 with PAD_TOKEN_ID=151643
  4. Running Qwen3-4B encode, extracting layer 34 (penultimate, 0-indexed)
  5. Slicing first 204 rows → cap_feats[1, 204, 2560]

The Python diffsynth path does:
  1. apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)
     — already proven to produce IDENTICAL token IDs to Rust manual template (T7-pre check)
  2. Pad to max_sequence_length (configurable; 512 in our case)
  3. Run text_encoder, take hidden_states[-2]  (= layer 35 in 1-indexed = layer 34 in 0-indexed)
  4. Mask out padded rows using attention mask

T7 compares Python's output of step 3 (sliced to first 204 rows) vs cached `cap_feats`.

PASS criteria: cos >= 0.999.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import safetensors.torch as st
from transformers import AutoTokenizer, Qwen3Model, Qwen3Config


WEIGHTS_PATH = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
TOKENIZER_PATH = "/home/alex/.serenity/models/zimage_base/tokenizer/"
SAMPLE_TXT = "/home/alex/boxjana_workspace/boxjana/10.txt"
CACHED_CAP_FEATS = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512/10.safetensors"
PAD_TOKEN_ID = 151643
PAD_LEN = 512
EXTRACT_LAYER_NEG = -2  # penultimate


def build_qwen3_config():
    return Qwen3Config(**{
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 40960,
        "max_window_layers": 36,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": False,
        "use_sliding_window": False,
        "vocab_size": 151936,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--out-dir", default="/tmp/l2p_thorough_parity/T7")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[T7] FAIL: CUDA required")
        return 1
    device = torch.device("cuda")

    # 1. Load tokenizer + apply chat template.
    print(f"[T7] loading tokenizer from {TOKENIZER_PATH}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    prompt = Path(SAMPLE_TXT).read_text().strip()
    messages = [{"role": "user", "content": prompt}]
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    # Match the Rust path: pad to PAD_LEN with PAD_TOKEN_ID.
    enc = tok(templated, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0].tolist()
    valid_len = min(len(ids), PAD_LEN)
    while len(ids) < PAD_LEN:
        ids.append(PAD_TOKEN_ID)
    ids = ids[:PAD_LEN]
    ids_t = torch.tensor([ids], device=device, dtype=torch.long)
    attn_mask = (ids_t != PAD_TOKEN_ID).long()
    # Force attention_mask to all-ones over the first valid_len positions
    # to match the Rust pipeline (which has no separate mask).
    attn_mask = torch.zeros_like(ids_t)
    attn_mask[0, :valid_len] = 1
    print(f"[T7] templated len={len(templated)} chars, valid_len={valid_len}, padded to {PAD_LEN}")

    # 2. Build + load Qwen3 model.
    print(f"[T7] loading Qwen3-4B weights from {WEIGHTS_PATH}")
    cfg = build_qwen3_config()
    model = Qwen3Model(cfg)
    weights = st.load_file(WEIGHTS_PATH)
    # The safetensors file may have a "model." prefix or be plain — try both.
    # Diffsynth's loader uses keys directly. We replicate that.
    sd = {}
    for k, v in weights.items():
        # Strip "model." prefix if present — common HF artifact.
        kk = k[len("model."):] if k.startswith("model.") else k
        sd[kk] = v
    # Cast to BF16
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[T7] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # 3. Run encoder, extract hidden_states[-2].
    print(f"[T7] running Qwen3 forward (BF16) with attention_mask...")
    with torch.no_grad():
        outputs = model(
            input_ids=ids_t,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
    hs = outputs.hidden_states
    print(f"[T7] hidden_states count = {len(hs)}; using [{EXTRACT_LAYER_NEG}]")
    py_cap_feats = hs[EXTRACT_LAYER_NEG][:, :valid_len, :].to(torch.float32).cpu()
    print(f"[T7] python cap_feats: shape={tuple(py_cap_feats.shape)}, dtype={py_cap_feats.dtype}")
    print(f"  abs.mean={py_cap_feats.abs().mean().item():.4e}, abs.max={py_cap_feats.abs().max().item():.4e}")

    # 4. Load cached Rust cap_feats.
    cache = st.load_file(CACHED_CAP_FEATS, device="cpu")
    ru_cap_feats = cache["cap_feats"].to(torch.float32)
    if ru_cap_feats.ndim == 2:
        ru_cap_feats = ru_cap_feats.unsqueeze(0)
    print(f"[T7] rust cap_feats: shape={tuple(ru_cap_feats.shape)}, dtype={ru_cap_feats.dtype}")
    print(f"  abs.mean={ru_cap_feats.abs().mean().item():.4e}, abs.max={ru_cap_feats.abs().max().item():.4e}")

    if py_cap_feats.shape != ru_cap_feats.shape:
        print(f"[T7] FAIL: shape mismatch py={py_cap_feats.shape} ru={ru_cap_feats.shape}")
        return 1

    # 5. Compare.
    a = py_cap_feats.flatten().double()
    b = ru_cap_feats.flatten().double()
    diff = (a - b)
    cos = (a * b).sum().item() / (a.norm().item() * b.norm().item())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    print(f"\n[T7] cos={cos:.6f}  max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}  mse={mse:.4e}")

    # Save for inspection.
    st.save_file({
        "py_cap_feats": py_cap_feats.contiguous(),
        "ru_cap_feats": ru_cap_feats.contiguous(),
    }, str(out_dir / "captures.safetensors"))
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"cos = {cos:.6f}\nmax_abs = {max_abs:.4e}\nmean_abs = {mean_abs:.4e}\nmse = {mse:.4e}\n")

    passed = cos >= args.threshold
    print(f"[T7] {'PASS' if passed else 'FAIL'} (threshold cos>={args.threshold})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
