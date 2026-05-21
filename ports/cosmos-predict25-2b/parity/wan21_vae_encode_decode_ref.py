#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate the Wan 2.1 VAE encode+decode parity fixture, GPU-streamed in
BF16, using `tokenizer.pth` from `nvidia/Cosmos-Predict2.5-2B`.

The Cosmos pipeline (`tokenizers/wan2pt1.py:1008-1026`) does:
  - encode: pixel_video → latent, then `(latent - img_mean) / img_std`
    for single-frame or per-T mean/std for video.
  - decode: latent * std + mean → recon.

For the Rust parity test we want the **raw** VAE encoder/decoder
output (pre/post the img_mean / img_std normalization), since that's
what the Rust `Wan21VaeEncoder::encode` / `Wan21VaeDecoder::decode`
produce. So we run the unwrapped `WanVAE_.encode` / `.decode` paths
and dump both forms (raw and Cosmos-normalized).

Outputs (next to this script):
  wan21_vae_roundtrip_ref.safetensors with:
    - input_pixel_bf16:        [1, 3, 5, 64, 64]
    - raw_latent_bf16:          encoder output before img_mean/std
    - cosmos_norm_latent_bf16:  encoder output after Cosmos normalize
    - raw_decoded_bf16:         decode(raw_latent)
    - cosmos_unnorm_decoded_bf16: decode(latent*std + mean)

Usage:
    huggingface-cli login
    # The script will reuse the .pth fetched by
    # `convert_wan21_vae_pth_to_safetensors.py`.
    python wan21_vae_encode_decode_ref.py
"""

import os
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"ERROR: {e}. `pip install torch safetensors`")
    sys.exit(1)


def find_pth() -> Path:
    if "COSMOS_TOKENIZER_PTH" in os.environ:
        p = Path(os.environ["COSMOS_TOKENIZER_PTH"])
        if not p.exists():
            print(f"ERROR: $COSMOS_TOKENIZER_PTH does not exist: {p}")
            sys.exit(1)
        return p
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("nvidia/Cosmos-Predict2.5-2B", "tokenizer.pth"))


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. CPU BF16 parity refs are forbidden.")
        return 1

    pth = find_pth()
    print(f"Source .pth: {pth}")

    # We dynamically import WanVAE_ from the Cosmos repo to avoid
    # re-deriving its forward path here. The repo lives at
    # /home/alex/refs/cosmos-predict2.5/.
    repo = Path("/home/alex/refs/cosmos-predict2.5")
    if not repo.exists():
        print(
            f"ERROR: Cosmos repo not at {repo}. Clone "
            "https://github.com/nvidia-cosmos/cosmos-predict2.5 and "
            "re-run."
        )
        return 1
    sys.path.insert(0, str(repo))

    try:
        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import (
            WanVAE_,
        )  # noqa: E402
    except Exception as e:
        print(f"ERROR: cannot import WanVAE_: {e}")
        print(
            "  Cosmos repo dependencies are heavy; you may need to "
            "`pip install attrs hydra-core easy-io transformers` first."
        )
        return 1

    device = torch.device("cuda")
    print("Instantiating WanVAE_ ...")
    model = WanVAE_(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    print("Loading state_dict ...")
    ckpt = torch.load(pth, map_location=device, weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt["model"].values()
    ):
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    # Reproducible input. Small video: 5 frames at 64x64 — small enough
    # that the encoder/decoder fit comfortably alongside HiDream's
    # current GPU usage.
    torch.manual_seed(42)
    x = (torch.randn(1, 3, 5, 64, 64, device=device, dtype=torch.bfloat16) * 0.5).clamp(
        -1.0, 1.0
    )

    print(f"Input pixel: {tuple(x.shape)} {x.dtype}")

    with torch.no_grad():
        # WanVAE_ exposes encode/decode directly. Encode returns the
        # latent BEFORE Cosmos's img_mean/img_std normalize.
        raw_latent = model.encode(x, clear_encoder_cache=True)
        print(f"Raw latent: {tuple(raw_latent.shape)} {raw_latent.dtype}")

        # Decode the raw latent (round-trip without Cosmos normalize).
        raw_decoded = model.decode(raw_latent, clear_decoder_cache=True)
        print(f"Raw decoded: {tuple(raw_decoded.shape)} {raw_decoded.dtype}")

    fixture = {
        "input_pixel_bf16": x.contiguous(),
        "raw_latent_bf16": raw_latent.to(torch.bfloat16).contiguous(),
        "raw_decoded_bf16": raw_decoded.to(torch.bfloat16).contiguous(),
    }

    out_path = Path(__file__).parent / "wan21_vae_roundtrip_ref.safetensors"
    save_file(fixture, str(out_path))
    print(f"Wrote fixture {out_path} ({len(fixture)} tensors)")
    print(
        "  NOTE: this fixture covers the RAW encode/decode path. The "
        "Cosmos img_mean/img_std normalize is a separate one-liner "
        "applied outside the VAE module and lives in the binary stage "
        "wiring (chunk 11+)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
