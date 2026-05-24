#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate the Cosmos-Reason1-7B text-encoder parity fixture for the
Rust-side `CosmosReason1Encoder` test.

Mirrors `cosmos_predict2/_src/predict2/text_encoders/text_encoder.py`'s
`compute_text_embeddings_online` for a single prompt, on CUDA, in BF16.

GPU-streamed: loads the 7B Qwen2.5-VL model in BF16 (~14 GB), runs
forward with `output_hidden_states=True`, dumps the layer outputs +
the FullConcat aggregation. Writes the fixture to
`cosmos_reason1_encode_ref.safetensors` next to this script.

CRITICAL: per port CONTEXT.md, never CPU-generate parity refs (PyTorch
BF16 CPU vs CUDA diverge at cos=0.5 per layer). This script asserts
`device='cuda'` and falls back hard.

Usage:
    huggingface-cli login   # required: nvidia/Cosmos-Reason1-7B is gated
    python cosmos_reason1_encode_ref.py
    # or:
    COSMOS_REASON1_PATH=/path/to/local/snapshot python cosmos_reason1_encode_ref.py
"""

import json
import os
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen25VL
    except ImportError:
        _Qwen25VL = None
    from transformers import AutoModelForCausalLM
except ImportError as e:
    print(f"ERROR: {e}. `pip install transformers safetensors`")
    sys.exit(1)


# Mirror constants from Rust `cosmos_reason1.rs` and Python `text_encoder.py`.
COSMOS_PAD_TOKENS = 512
QWEN25VL_PAD_ID = 151643
COSMOS_SYSTEM_PROMPT = (
    "You are a helpful assistant who will provide prompts to an image generator."
)
TEST_PROMPT = "a cat in a sunlit garden, photorealistic"


def find_snapshot() -> Path:
    if "COSMOS_REASON1_PATH" in os.environ:
        p = Path(os.environ["COSMOS_REASON1_PATH"])
        if not p.exists():
            print(f"ERROR: $COSMOS_REASON1_PATH does not exist: {p}")
            sys.exit(1)
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("nvidia/Cosmos-Reason1-7B"))


def mean_normalize(t: torch.Tensor) -> torch.Tensor:
    """Mirror `TextEncoder.mean_normalize` (text_encoder.py:119-129).

    `t.std(...)` defaults to unbiased (N-1) — keep that, the Rust side
    matches.
    """
    return (t - t.mean(dim=-1, keepdim=True)) / (t.std(dim=-1, keepdim=True) + 1e-8)


def main() -> int:
    if not torch.cuda.is_available():
        print(
            "ERROR: no CUDA device. CPU BF16 parity refs are forbidden "
            "(see CONTEXT.md / feedback_pytorch_cpu_vs_cuda_bf16)."
        )
        return 1

    snap = find_snapshot()
    print(f"Snapshot: {snap}")

    # Load tokenizer (text-only path; AutoTokenizer skips the vision processor).
    tok = AutoTokenizer.from_pretrained(snap)
    print(f"Tokenizer: {type(tok).__name__}, pad_token_id={tok.pad_token_id}")
    if tok.pad_token_id != QWEN25VL_PAD_ID:
        print(
            f"WARNING: tokenizer pad_token_id={tok.pad_token_id} != "
            f"expected {QWEN25VL_PAD_ID}; Rust pads with {QWEN25VL_PAD_ID}"
        )

    # Apply Cosmos chat template (system + user, add_generation_prompt=False).
    messages = [
        {"role": "system", "content": COSMOS_SYSTEM_PROMPT},
        {"role": "user", "content": TEST_PROMPT},
    ]
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print("Templated prompt (first 300 chars):")
    print(repr(templated[:300]))

    # Tokenize with no extra specials (template already has them).
    ids = tok(templated, add_special_tokens=False).input_ids
    print(f"Raw token count: {len(ids)}")

    # Pad / truncate to 512 with pad_id.
    if len(ids) < COSMOS_PAD_TOKENS:
        ids = ids + [QWEN25VL_PAD_ID] * (COSMOS_PAD_TOKENS - len(ids))
    else:
        ids = ids[:COSMOS_PAD_TOKENS]
    assert len(ids) == COSMOS_PAD_TOKENS

    ids_tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
    print(f"Token tensor: {tuple(ids_tensor.shape)} {ids_tensor.dtype}")

    # Load the 7B model in BF16. We use AutoModelForCausalLM for the
    # **text-only** path so we don't need to instantiate the vision
    # encoder. The hidden-states layer of the LM is what Cosmos consumes
    # (text_encoder.py iterates `outputs.hidden_states[1:]` — skips the
    # input-embedding entry, then mean-normalizes each).
    print("Loading model in BF16 on CUDA (~14 GB)...")
    if _Qwen25VL is None:
        print("ERROR: transformers lacks Qwen2_5_VLForConditionalGeneration. Upgrade `transformers`.")
        return 1
    model = _Qwen25VL.from_pretrained(
        snap,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",  # match flame::sdpa numerically
    )
    model.eval()

    with torch.no_grad():
        # Qwen2.5-VL ForConditionalGeneration exposes a text-only LM through
        # `model.model` (or `model.language_model`). Run that path so we
        # don't need vision inputs.
        lm = getattr(model, "language_model", None) or getattr(model, "model", None) or model
        out = lm(
            input_ids=ids_tensor,
            attention_mask=None,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = out.hidden_states  # tuple of (num_layers+1) tensors

    # Skip hidden_states[0] (the input embedding); iterate the
    # `num_layers` post-block outputs.
    layer_outs = list(hidden_states[1:])
    print(f"Got {len(layer_outs)} per-layer hidden states; each shape={tuple(layer_outs[0].shape)}")

    # Mean-normalize each layer (in BF16, with F32 reductions matching
    # PyTorch internal accumulation).
    normalized = [mean_normalize(h) for h in layer_outs]

    # FULL_CONCAT aggregation: concat along last dim → [1, 512, 100352].
    full_concat = torch.cat(normalized, dim=-1).to(torch.bfloat16).contiguous()
    print(f"FullConcat output: {tuple(full_concat.shape)} {full_concat.dtype}")

    # MEAN_POOLING: stack + mean → [1, 512, 3584].
    mean_pool = torch.stack(normalized, dim=0).mean(dim=0).to(torch.bfloat16).contiguous()
    print(f"MeanPooling output: {tuple(mean_pool.shape)} {mean_pool.dtype}")

    # Save fixture. token_ids → I32 for Rust loadability; layer outputs
    # also serialized so layer-by-layer parity bisects can find drift.
    out_path = Path(__file__).parent / "cosmos_reason1_encode_ref.safetensors"
    fixture: dict[str, torch.Tensor] = {
        "token_ids": ids_tensor.to(torch.int32).contiguous(),
        "templated_text_bytes": torch.tensor(
            list(templated.encode("utf-8")), dtype=torch.uint8
        ),
        "full_concat_bf16": full_concat,
        "mean_pooling_bf16": mean_pool,
        # Alias for downstream parity scripts (per-layer DiT capture).
        "prompt_emb_bf16": full_concat.clone(),
    }
    for i, t in enumerate(normalized):
        fixture[f"normalized_layer_{i:02d}_bf16"] = t.to(torch.bfloat16).contiguous()
    for i, t in enumerate(layer_outs):
        fixture[f"raw_layer_{i:02d}_bf16"] = t.to(torch.bfloat16).contiguous()

    save_file(fixture, str(out_path))
    print(f"Wrote fixture {out_path} ({len(fixture)} tensors)")

    # Also dump a small JSON manifest so Rust can sanity-check shapes
    # without parsing safetensors headers manually.
    manifest = {
        "snapshot": str(snap),
        "prompt": TEST_PROMPT,
        "system_prompt": COSMOS_SYSTEM_PROMPT,
        "num_layers": len(layer_outs),
        "hidden_size": layer_outs[0].shape[-1],
        "pad_id": QWEN25VL_PAD_ID,
        "pad_tokens": COSMOS_PAD_TOKENS,
        "full_concat_dim": full_concat.shape[-1],
        "mean_pooling_dim": mean_pool.shape[-1],
    }
    (Path(__file__).parent / "cosmos_reason1_encode_ref.json").write_text(
        json.dumps(manifest, indent=2)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
