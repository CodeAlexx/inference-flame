#!/usr/bin/env python3
"""T1a — Encode prompt + uncond into embeddings.safetensors (separate proc).

Runs Qwen3-4B in its own process so we can fully release GPU memory
before Rust inference and full-pipeline Python runs.
"""
import os
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"
import sys
from pathlib import Path
import torch
import safetensors.torch as st
from transformers import AutoTokenizer, Qwen3Model, Qwen3Config

QWEN_WEIGHTS = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
TOKENIZER_PATH = "/home/alex/.serenity/models/zimage_base/tokenizer/"
OUT = Path("/tmp/l2p_thorough_parity/T1/embeddings.safetensors")
PROMPT = "a serene mountain landscape at sunset, with snow-capped peaks reflecting in a calm lake, photorealistic, high detail"


def build_cfg():
    return Qwen3Config(**{
        "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": 151643, "eos_token_id": 151645,
        "head_dim": 128, "hidden_size": 2560, "initializer_range": 0.02,
        "intermediate_size": 9728, "max_position_embeddings": 40960,
        "max_window_layers": 36, "model_type": "qwen3",
        "num_attention_heads": 32, "num_hidden_layers": 36,
        "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
        "rope_scaling": None, "rope_theta": 1000000, "sliding_window": None,
        "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "use_cache": False, "use_sliding_window": False, "vocab_size": 151936,
    })


def encode(tok, model, text, device):
    messages = [{"role": "user", "content": text}]
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    enc = tok(templated, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0].tolist()
    valid_len = len(ids)
    while len(ids) < 512:
        ids.append(151643)
    ids = ids[:512]
    ids_t = torch.tensor([ids], device=device, dtype=torch.long)
    mask = torch.zeros_like(ids_t)
    mask[0, :valid_len] = 1
    with torch.no_grad():
        out = model(input_ids=ids_t, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states[-2][:, :valid_len, :].to(torch.bfloat16).contiguous()
    return hs


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print("[T1a] loading tokenizer + Qwen3-4B")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    cfg = build_cfg()
    model = Qwen3Model(cfg)
    sd = st.load_file(QWEN_WEIGHTS)
    sd = {(k[len("model."):] if k.startswith("model.") else k): v.to(torch.bfloat16) for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    print("[T1a] encoding prompt + empty uncond")
    cap = encode(tok, model, PROMPT, device)
    unc = encode(tok, model, "", device)
    if cap.ndim == 2:
        cap = cap.unsqueeze(0)
    if unc.ndim == 2:
        unc = unc.unsqueeze(0)
    st.save_file({
        "cap_feats": cap.cpu().contiguous(),
        "cap_feats_uncond": unc.cpu().contiguous(),
    }, str(OUT))
    print(f"[T1a] wrote {OUT}")
    print(f"  cap_feats: {tuple(cap.shape)}")
    print(f"  cap_feats_uncond: {tuple(unc.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
