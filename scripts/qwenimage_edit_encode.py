#!/usr/bin/env python
"""Qwen-Image-Edit — Stage 1 (text encode + VAE encode of reference image).

This is the EDIT variant. Differences from `qwenimage_encode.py` (T2I):

1. **Vision input to the text encoder**: Qwen2.5-VL is a multimodal model.
   For Edit, we feed both the text prompt AND the reference image through
   the processor → encoder. The output hidden states encode the user's
   instruction conditioned on the visual content.
2. **Different prompt template**: includes `<|vision_start|><|image_pad|><|vision_end|>`
   markers and a different system prompt. `start_idx = 64` (was 34 for T2I).
3. **Reference image VAE encoding**: the reference image is also encoded
   through the VAE and saved alongside the text embeddings. The Stage 2 Rust
   binary concatenates this with the noise latent at every denoise step.
4. **Two models on GPU sequentially**: text encoder → drop → VAE → drop. NEVER
   both at once — keeps each stage within the 24 GB budget.

Usage:
    python qwenimage_edit_encode.py \\
        /path/to/reference.png \\
        "make the cat blue" \\
        ""  \\
        /path/to/embeds.safetensors

Output safetensors keys:
    cond:          [1, L_cond, 3584]   BF16  — instruction-conditioned hidden states
    uncond:        [1, L_uncond, 3584] BF16  — same for the negative prompt
    image_latents: [1, packed_seq, 64] BF16  — VAE-encoded reference, packed for the DiT
    image_h:       [1] f32 — original reference height (for unpacking)
    image_w:       [1] f32 — original reference width
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor

# Diffusers pipeline_qwenimage_edit constants
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 64

REPO = "Qwen/Qwen-Image-Edit"
SNAP = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit/snapshots/"
    "ac7f9318f633fc4b5778c59367c8128225f1e3de"
)
VAE_SCALE_FACTOR = 8  # 2 ** len(temperal_downsample) = 2^3
LATENT_CHANNELS = 16  # z_dim


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Mirror of `calculate_dimensions` in pipeline_qwenimage_edit.py:155-162.

    Python:
        width  = math.sqrt(target_area * ratio)
        height = width / ratio
        width  = round(width  / 32) * 32
        height = round(height / 32) * 32

    Note: math.sqrt(target_area / ratio) is mathematically identical to
    math.sqrt(target_area * ratio) / ratio, so we can compute both
    independently. The critical step is the round-to-32, which the
    pipeline does BEFORE the additional round-to-16 in __call__().
    """
    import math
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = int(round(width / 32) * 32)
    height = int(round(height / 32) * 32)
    return width, height, ratio


def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    """Mirror of `_extract_masked_hidden` from the pipeline."""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def encode_prompt_with_image(
    processor,
    text_encoder,
    prompt: str,
    image: Image.Image,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the VLM with vision input and return the [1, L, 3584] hidden states
    for a single prompt + image, with the system-prompt tokens dropped."""
    txt = PROMPT_TEMPLATE_ENCODE.format(prompt)
    drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX

    model_inputs = processor(
        text=[txt],
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
    hidden_states = out.hidden_states[-1]
    split = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split = [e[drop_idx:] for e in split]

    emb = split[0].unsqueeze(0).to(dtype=dtype, device=device)
    return emb


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Mirror of `_pack_latents` in pipeline_qwenimage_edit.py:374-378.

    Input:  [B, 1, C, H, W]  (5D, single-frame video latent)
    Output: [B, (H/2)(W/2), C*4]
    """
    b, _f, c, h, w = latents.shape
    out = latents.view(b, c, h // 2, 2, w // 2, 2)
    out = out.permute(0, 2, 4, 1, 3, 5)
    out = out.reshape(b, (h // 2) * (w // 2), c * 4)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: qwenimage_edit_encode.py <ref_image> [prompt] [negative] [out.safetensors]")
        return 1

    ref_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "make this image more vibrant"
    negative = sys.argv[3] if len(sys.argv) > 3 else ""
    out_path = sys.argv[4] if len(sys.argv) > 4 else (
        "/home/alex/serenity/output/qwenimage_edit_embeds.safetensors"
    )

    device = "cuda"
    dtype = torch.bfloat16

    print("=== Qwen-Image-Edit — Stage 1 (text + VAE encode) ===")
    print(f"Reference: {ref_path}")
    print(f"Prompt:    {prompt!r}")
    print(f"Negative:  {negative!r}")
    print(f"Output:    {out_path}")
    print()

    # ------------------------------------------------------------------
    # Load + resize reference image
    # ------------------------------------------------------------------
    if not os.path.exists(ref_path):
        print(f"[error] reference image not found: {ref_path}")
        return 2

    ref_pil = Image.open(ref_path).convert("RGB")
    orig_w, orig_h = ref_pil.size
    print(f"--- Reference image: {orig_w}x{orig_h} ---")

    # Use the same target_area logic as the diffusers pipeline.
    target_area = 1024 * 1024
    aspect_ratio = orig_w / orig_h
    calc_w, calc_h, _ = calculate_dimensions(target_area, aspect_ratio)
    multiple_of = VAE_SCALE_FACTOR * 2  # = 16
    calc_w = (calc_w // multiple_of) * multiple_of
    calc_h = (calc_h // multiple_of) * multiple_of
    print(f"  Target latent input: {calc_w}x{calc_h}")

    image_processor = VaeImageProcessor(vae_scale_factor=VAE_SCALE_FACTOR * 2)
    ref_resized = image_processor.resize(ref_pil, calc_h, calc_w)
    ref_tensor = image_processor.preprocess(ref_resized, calc_h, calc_w)
    # `preprocess` returns `[B, C, H, W]`. The pipeline does `image.unsqueeze(2)` to
    # turn it into the 5D single-frame video shape `[B, C, 1, H, W]` for the 3D VAE.
    ref_tensor_5d = ref_tensor.unsqueeze(2).to(device=device, dtype=dtype)
    print(f"  Preprocessed: {tuple(ref_tensor_5d.shape)}")
    print()

    snap = Path(SNAP)
    if not snap.exists():
        print(f"[error] snapshot path does not exist: {snap}")
        print("       download configs first via:")
        print(f"       hf_hub_download('{REPO}', 'transformer/config.json')")
        return 2

    # ------------------------------------------------------------------
    # Stage 1a: text encoder (Qwen2.5-VL with vision tower)
    # ------------------------------------------------------------------
    print("--- Loading Qwen2.5-VL processor + text encoder ---")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(snap / "processor")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snap / "text_encoder",
        dtype=dtype,
        device_map=device,
    )
    text_encoder.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    print("--- Encoding cond (with vision) ---")
    t0 = time.time()
    cond = encode_prompt_with_image(
        processor, text_encoder, prompt, ref_resized, device, dtype
    )
    print(f"  cond:   {tuple(cond.shape)} in {time.time() - t0:.1f}s")

    print("--- Encoding uncond (with same vision input) ---")
    t0 = time.time()
    uncond = encode_prompt_with_image(
        processor, text_encoder, negative, ref_resized, device, dtype
    )
    print(f"  uncond: {tuple(uncond.shape)} in {time.time() - t0:.1f}s")

    # Drop the text encoder before loading the VAE.
    del text_encoder
    del processor
    torch.cuda.empty_cache()
    print("  Text encoder evicted")
    print()

    # ------------------------------------------------------------------
    # Stage 1b: VAE encode the reference image
    # ------------------------------------------------------------------
    print("--- Loading VAE ---")
    t0 = time.time()
    vae = AutoencoderKLQwenImage.from_pretrained(
        snap / "vae",
        dtype=dtype,
    ).to(device)
    vae.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("--- Encoding reference through VAE ---")
    t0 = time.time()
    with torch.no_grad():
        # Mirrors `_encode_vae_image` in pipeline_qwenimage_edit.py:398-419,
        # which calls `retrieve_latents(..., sample_mode='argmax')`. That helper
        # invokes `latent_dist.mode()`, and for `DiagonalGaussianDistribution`
        # `mode()` is defined as `return self.mean` — so `.mean` is the
        # deterministic, byte-identical equivalent (no noise injection).
        latents_dist = vae.encode(ref_tensor_5d).latent_dist
        image_latents = latents_dist.mode()

    # Per-channel normalization to match the diffusers pipeline.
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, LATENT_CHANNELS, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, LATENT_CHANNELS, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std
    print(f"  Raw VAE output: {tuple(image_latents.shape)} in {time.time() - t0:.1f}s")

    # Pack with the same _pack_latents the pipeline uses
    image_latents_packed = pack_latents(image_latents)
    print(f"  Packed: {tuple(image_latents_packed.shape)}")

    # Drop the VAE before saving (the safetensors save itself doesn't need GPU)
    del vae
    torch.cuda.empty_cache()
    print("  VAE evicted")
    print()

    # ------------------------------------------------------------------
    # Save everything Stage 2 needs
    # ------------------------------------------------------------------
    print("--- Saving embeddings ---")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tensors = {
        "cond": cond.contiguous().cpu(),
        "uncond": uncond.contiguous().cpu(),
        "image_latents": image_latents_packed.contiguous().cpu(),
        # Save the reference image dimensions and chosen target dims so the
        # Rust binary knows the latent geometry without re-decoding the image.
        "image_h": torch.tensor([float(calc_h)], dtype=torch.float32),
        "image_w": torch.tensor([float(calc_w)], dtype=torch.float32),
    }
    save_file(tensors, out_path)
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print()

    print("============================================================")
    print(f"EMBEDDINGS SAVED: {out_path}")
    print("============================================================")
    print()
    print(f"Next: ./target/release/qwenimage_edit_gen {out_path} <latents.safetensors>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
