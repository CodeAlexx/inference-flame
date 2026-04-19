#!/usr/bin/env python
"""LTX-2 negative-prompt parity reference.

Generates the authoritative Lightricks-style negative-prompt features at
TWO stages of the pipeline:

  Gemma-3 12B  →  FeatureExtractor (concat+project)  →  Connector (2-layer)
                         ^^^^^                             ^^^^^
                     "precompute"                       "encode"

Rust `feature_extract_and_project` lives at the `precompute` stage — the
output goes straight into `LTX2StreamingModel.forward_*`, where the
connector is applied INSIDE the DiT forward (see
src/models/ltx2_model.rs::load_video_embeddings_connector). So a true
apples-to-apples parity check compares precompute(Python) to
feature_extract_and_project(Rust).

We also save the "encode" outputs because they are what
`cache_av_embeddings.py`-style callers persist, and future tests that
exercise the full encoder pipeline can use them.

Output: output/ltx2_neg_prompt_ref.safetensors with:

  # Raw (pre-connector) features — what Rust produces today
  precompute_v_neg             [1, 1024, 4096] bf16
  precompute_a_neg             [1, 1024, 2048] bf16
  precompute_v_empty           [1, 1024, 4096] bf16
  precompute_a_empty           [1, 1024, 2048] bf16

  # Post-connector features — for downstream checks
  encode_v_neg                 [1, S, 4096] bf16
  encode_a_neg                 [1, S, 2048] bf16
  encode_v_empty               [1, S, 4096] bf16
  encode_a_empty               [1, S, 2048] bf16

  # Metadata
  negative_string_bytes        [N] uint8 — exact UTF-8 of the default neg
  negative_string_len          [] int64  — byte length of the string

The exact negative string — copied verbatim from
/tmp/ltx-video/ltx_video/inference.py:351 — is:

    "worst quality, inconsistent motion, blurry, jittery, distorted"

Script is idempotent; re-run to regenerate.
"""
from __future__ import annotations

import sys
from pathlib import Path

# LTX-Desktop venv wires up `ltx_core` and `ltx_pipelines`
sys.path.insert(0, "/home/alex/LTX-Desktop/backend/.venv/lib/python3.12/site-packages")
sys.path.insert(0, "/home/alex/LTX-Desktop/backend")

import torch
from safetensors.torch import save_file

from ltx_pipelines.utils.model_ledger import ModelLedger
from ltx_core.text_encoders.gemma import encode_text

# Must match /tmp/ltx-video/ltx_video/inference.py:351-354 byte-for-byte.
DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"

CHECKPOINT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
# Dedup'd Gemma-3 dir: config/tokenizer/preprocessor from the standalone
# dump + symlinked shards from its `model/` subdir only. The original
# standalone layout has the SAME 5 shards at BOTH top level AND under
# `model/`; Lightricks's loader rglob's *.safetensors, doubling the load
# (2× 24GB = OOM on a 24GB card). Create this directory once with:
#   mkdir -p /tmp/gemma-3-12b-ltx-clean && cd $_
#   for f in added_tokens.json chat_template.json config.json \
#            generation_config.json preprocessor_config.json \
#            processor_config.json special_tokens_map.json \
#            tokenizer_config.json tokenizer.json tokenizer.model \
#            model.safetensors.index.json; do
#     ln -sfn /home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone/$f $f
#   done
#   for i in 1 2 3 4 5; do
#     ln -sfn /home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone/model/model-0000${i}-of-00005.safetensors \
#             model-0000${i}-of-00005.safetensors
#   done
GEMMA_PATH = "/tmp/gemma-3-12b-ltx-clean"

OUT_PATH = Path(
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_neg_prompt_ref.safetensors"
)


def _report(name: str, t: torch.Tensor) -> None:
    t_f = t.float()
    print(
        f"    {name}: shape={list(t.shape)} "
        f"mean={t_f.mean().item():+.6f} "
        f"std={t_f.std().item():.6f} "
        f"min={t_f.min().item():+.6f} max={t_f.max().item():+.6f} "
        f"frac_nonzero={(t_f != 0).float().mean().item():.4f}"
    )


def main() -> int:
    print("=== LTX-2 negative-prompt parity reference ===")
    print(f"default_negative  = {DEFAULT_NEGATIVE!r}")
    print(f"bytes             = {len(DEFAULT_NEGATIVE.encode('utf-8'))}")
    print(f"output            = {OUT_PATH}")
    print()

    # Gemma-3 12B bf16 = 24 GB. On a 24 GB card there isn't enough
    # headroom even for text encoding alone — loading OOMs at 22.7 GB.
    # Run on CPU instead; the parity target is the numeric tensor, not
    # the wall-clock.
    device = torch.device("cpu")
    print(f"Loading text encoder via ModelLedger on {device}...")
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=CHECKPOINT,
        gemma_root_path=GEMMA_PATH,
    )
    text_encoder = ledger.text_encoder()

    # precompute(): (video_feats, audio_feats, attention_mask) — this is
    # the apples-to-apples reference for our Rust feature_extractor path.
    print("Running precompute() for [default_negative, empty_string]...")
    with torch.no_grad():
        pc_neg = text_encoder.precompute(DEFAULT_NEGATIVE)
        pc_empty = text_encoder.precompute("")
    pc_v_neg, pc_a_neg, _pc_mask_neg = pc_neg
    pc_v_empty, pc_a_empty, _pc_mask_empty = pc_empty

    print("\nPRE-connector features (Rust feature_extract_and_project target):")
    _report("pc_v_neg", pc_v_neg)
    _report("pc_a_neg", pc_a_neg)
    _report("pc_v_empty", pc_v_empty)
    _report("pc_a_empty", pc_a_empty)

    # encode_text(): calls forward() which is precompute + connector.
    # Saved so the downstream cache-consumer tests have something to
    # cross-check against.
    print("\nRunning encode_text() (post-connector) for [default_negative, empty_string]...")
    with torch.no_grad():
        enc_results = encode_text(text_encoder, prompts=[DEFAULT_NEGATIVE, ""])
    (enc_v_neg, enc_a_neg) = enc_results[0]
    (enc_v_empty, enc_a_empty) = enc_results[1]

    print("POST-connector features:")
    _report("enc_v_neg", enc_v_neg)
    _report("enc_a_neg", enc_a_neg)
    _report("enc_v_empty", enc_v_empty)
    _report("enc_a_empty", enc_a_empty)

    # Embed the exact string + its length as tensors so Rust can read them
    # back without parsing comments / out-of-band metadata.
    neg_bytes = DEFAULT_NEGATIVE.encode("utf-8")
    tensors = {
        # Pre-connector (primary parity target)
        "precompute_v_neg": pc_v_neg.to(dtype=torch.bfloat16).contiguous().cpu(),
        "precompute_a_neg": pc_a_neg.to(dtype=torch.bfloat16).contiguous().cpu(),
        "precompute_v_empty": pc_v_empty.to(dtype=torch.bfloat16).contiguous().cpu(),
        "precompute_a_empty": pc_a_empty.to(dtype=torch.bfloat16).contiguous().cpu(),
        # Post-connector (downstream)
        "encode_v_neg": enc_v_neg.to(dtype=torch.bfloat16).contiguous().cpu(),
        "encode_a_neg": enc_a_neg.to(dtype=torch.bfloat16).contiguous().cpu(),
        "encode_v_empty": enc_v_empty.to(dtype=torch.bfloat16).contiguous().cpu(),
        "encode_a_empty": enc_a_empty.to(dtype=torch.bfloat16).contiguous().cpu(),
        # Metadata
        "negative_string_bytes": torch.tensor(list(neg_bytes), dtype=torch.uint8),
        "negative_string_len": torch.tensor(len(neg_bytes), dtype=torch.int64),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_PATH))
    print(f"\nwrote {OUT_PATH}")
    for k, v in tensors.items():
        print(f"  {k:<28} shape={tuple(v.shape)} dtype={v.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
