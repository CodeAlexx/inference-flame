#!/usr/bin/env python
"""Emit reference tensors for LTX-Video's conditioning-mask preparation,
for the pragmatic latent-only subset we support in Rust.

Source mirrored: /tmp/ltx-video/ltx_video/pipelines/pipeline_ltx_video.py
    prepare_conditioning (:1383-1587)
    _get_latent_spatial_position (:1605-1650) — only the frame==0, centered,
        strip_latent_border=False subset we actually need
    _handle_non_first_conditioning_sequence (:1652-1726) — only
        prefix_latents_mode="drop" (n_frames > 1) and the single-frame grid-lerp
        case (our Rust pragmatic fallback; Lightricks would put this on the
        extra-token path, which we've deferred)

We work in LATENT SPACE: items are already-encoded latents. That matches the
Rust helper's `ConditioningItem.latent` contract and keeps the parity test
independent of the VAE.

Output: output/ltx2_conditioning_mask_ref.safetensors
  init_latents_input      : [1, 128, F=5, H=4, W=4]  — seeded noise, the target grid
  item0_latent            : [1, 128, f_l=1, 4, 4]   — frame=0, strength=1.0 keyframe
  item1_latent            : [1, 128, f_l=1, 4, 4]   — frame=8 (latent idx 1), strength=0.8
  item2_latent            : [1, 128, f_l=3, 4, 4]   — frame=16, multi-frame drop-prefix
  merged_latents          : [1, 128, 5, 4, 4]      — post-merge, before patchify
  conditioning_mask_5d    : [1, 1,   5, 4, 4]      — f32, strength at ref pos, 0 else
  conditioning_mask_packed: [1, 80]                 — patchified (patch_size=1) for transformer
  noise_for_add_noise     : [1, 128, 5, 4, 4]      — fixed noise used by add_image_cond_noise
  latents_after_addnoise  : [1, 128, 5, 4, 4]      — after add_image_cond_noise(t=0.5, scale=0.15)

Run:  python scripts/ltx2_conditioning_mask_ref.py
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path(
    "/home/alex/EriDiffusion/inference-flame/output/ltx2_conditioning_mask_ref.safetensors"
)

# Fixed latent grid params. Small so parity runs fast; dims match what the
# Rust side will build.
BATCH = 1
CHANNELS = 128
F_LAT = 5          # latent frames
H_LAT = 4          # latent height
W_LAT = 4          # latent width
NUM_PREFIX_LATENT_FRAMES = 2  # Lightricks default


def prepare_conditioning_latents(
    init_latents: torch.Tensor,
    items: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-building subset of `prepare_conditioning`
    (pipeline_ltx_video.py:1383-1587), working directly on pre-encoded latents.

    Supported item shapes (keyed off `frame_number` and `latent.shape[2]`):

    (A) frame_number == 0                    → spatial-position-frame-0 branch
        We emulate `_get_latent_spatial_position` (:1605-1650) with
        strip_latent_border=False, media_x/y=None (centered; since in this test
        the item fills the full grid h/w, centering is a no-op). Lines 1466-1487.

    (B) frame_number > 0, f_l > num_prefix_latent_frames
                                             → non-first-sequence "main body"
        Emulates `_handle_non_first_conditioning_sequence` :1683-1697 with
        prefix_latents_mode="drop" (our Rust default; Python default is
        "concat" which requires extra-token plumbing we've deferred).

    (C) frame_number > 0, f_l == 1            → single-frame mid-keyframe,
        PRAGMATIC FALLBACK (NOT Lightricks-default): we lerp directly into the
        grid at latent_frame = frame_number // 8. Python's `prepare_conditioning`
        would route this through the extra-tokens path (:1505-1541); we've
        deferred that. Documented in src/models/ltx2_conditioning.rs.

    Returns (merged_latents, conditioning_mask_5d).
    """
    merged = init_latents.clone()
    mask = torch.zeros(
        (merged.shape[0], 1, merged.shape[2], merged.shape[3], merged.shape[4]),
        dtype=torch.float32,
        device=merged.device,
    )

    for item in items:
        lat = item["latent"]
        fn = item["frame_number"]
        strength = item["strength"]
        b, c, f_l, h_l, w_l = lat.shape
        assert b == merged.shape[0] and c == merged.shape[1]
        assert h_l <= merged.shape[3] and w_l <= merged.shape[4]

        if fn == 0:
            # Frame-0, full-spatial, centered. With h_l == H_LAT and w_l == W_LAT
            # the centered placement is (0,0) and strip_latent_border=False leaves
            # the latent as-is — see pipeline_ltx_video.py:1466-1487.
            y0 = (merged.shape[3] - h_l) // 2
            x0 = (merged.shape[4] - w_l) // 2
            # lerp(init, media, strength) = init + strength*(media - init)
            merged[:, :, :f_l, y0 : y0 + h_l, x0 : x0 + w_l] = torch.lerp(
                merged[:, :, :f_l, y0 : y0 + h_l, x0 : x0 + w_l],
                lat,
                strength,
            )
            mask[:, :, :f_l, y0 : y0 + h_l, x0 : x0 + w_l] = strength
            continue

        # frame_number > 0
        assert fn % 8 == 0, f"media_frame_number must be multiple of 8, got {fn}"

        if f_l > NUM_PREFIX_LATENT_FRAMES:
            # Case (B): main-body insertion for a multi-frame clip, prefix dropped.
            # Mirrors pipeline_ltx_video.py:1687-1697 with prefix_latents_mode="drop"
            # (i.e. the prefix-handling branch at :1714-1716 sets latents=None and
            # we do NOT fall through to the extra-tokens path).
            f_l_start = fn // 8 + NUM_PREFIX_LATENT_FRAMES
            f_l_end = f_l_start + (f_l - NUM_PREFIX_LATENT_FRAMES)
            assert f_l_end <= merged.shape[2]
            merged[:, :, f_l_start:f_l_end] = torch.lerp(
                merged[:, :, f_l_start:f_l_end],
                lat[:, :, NUM_PREFIX_LATENT_FRAMES:],
                strength,
            )
            mask[:, :, f_l_start:f_l_end] = strength
        elif f_l == 1:
            # Case (C): single-frame mid-keyframe, grid-lerp pragmatic fallback.
            # NOT Lightricks-parity — see docstring.
            f_l_start = fn // 8
            assert f_l_start < merged.shape[2]
            merged[:, :, f_l_start : f_l_start + 1] = torch.lerp(
                merged[:, :, f_l_start : f_l_start + 1],
                lat,
                strength,
            )
            mask[:, :, f_l_start : f_l_start + 1] = strength
        else:
            raise ValueError(
                f"Unsupported conditioning item: frame_number={fn}, f_l={f_l}. "
                "Non-first-sequence with f_l in (1, NUM_PREFIX_LATENT_FRAMES] "
                "would need the extra-tokens path (deferred)."
            )

    return merged, mask


def add_image_cond_noise(
    init_latents: torch.Tensor,
    latents: torch.Tensor,
    conditioning_mask_5d: torch.Tensor,
    noise: torch.Tensor,
    t: float,
    noise_scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Closed-form port of `add_noise_to_image_conditioning_latents`
    (pipeline_ltx_video.py:596-620). `conditioning_mask_5d` is broadcast
    across the channel dim by unsqueeze-then-where.
    """
    # Broadcast mask from [B, 1, F, H, W] to latent shape for the where.
    broadcast_mask = conditioning_mask_5d.expand_as(latents)
    need_to_noise = broadcast_mask > (1.0 - eps)
    noised_latents = init_latents + noise_scale * noise * (t**2)
    return torch.where(need_to_noise, noised_latents, latents)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0xC0FFEE)

    # Target grid: seeded Gaussian noise.
    init_latents = torch.randn(BATCH, CHANNELS, F_LAT, H_LAT, W_LAT, dtype=torch.float32)

    # Item 0: frame 0, single latent-frame, strength 1.0 (hard I2V-style).
    item0 = torch.randn(BATCH, CHANNELS, 1, H_LAT, W_LAT, dtype=torch.float32)
    # Item 1: frame 8 → latent frame 1, single-frame mid-keyframe (case C), strength 0.8.
    item1 = torch.randn(BATCH, CHANNELS, 1, H_LAT, W_LAT, dtype=torch.float32)
    # Item 2: frame 16 → latent frame 2, MULTI-frame (f_l=3), case B with "drop" prefix,
    #         strength 1.0. With f_l=3 and NUM_PREFIX=2, only 1 frame gets inserted (body
    #         = latent[:, :, 2:3]) at f_l_start = 16//8 + 2 = 4.
    item2 = torch.randn(BATCH, CHANNELS, 3, H_LAT, W_LAT, dtype=torch.float32)

    items = [
        {"latent": item0, "frame_number": 0, "strength": 1.0},
        {"latent": item1, "frame_number": 8, "strength": 0.8},
        {"latent": item2, "frame_number": 16, "strength": 1.0},
    ]

    merged, mask5d = prepare_conditioning_latents(init_latents, items)

    # Patchify-style packing for transformer: [B, 1, F, H, W] -> [B, F*H*W]
    # (patch_size=1 for LTX-2.3, matches SymmetricPatchifier in inference.py:242).
    # clone() so safetensors doesn't detect shared storage with mask5d.
    mask_packed = mask5d.squeeze(1).reshape(BATCH, F_LAT * H_LAT * W_LAT).clone()

    # add_image_cond_noise: fixed noise, t=0.5, scale=0.15.
    noise = torch.randn_like(init_latents)
    latents_after = add_image_cond_noise(
        init_latents=init_latents,
        latents=merged,
        conditioning_mask_5d=mask5d,
        noise=noise,
        t=0.5,
        noise_scale=0.15,
    )

    tensors = {
        "init_latents_input": init_latents.contiguous(),
        "item0_latent": item0.contiguous(),
        "item1_latent": item1.contiguous(),
        "item2_latent": item2.contiguous(),
        "merged_latents": merged.contiguous(),
        "conditioning_mask_5d": mask5d.contiguous(),
        "conditioning_mask_packed": mask_packed.contiguous(),
        "noise_for_add_noise": noise.contiguous(),
        "latents_after_addnoise": latents_after.contiguous(),
    }
    # Params for the Rust parity bin so it doesn't hard-code constants.
    tensors["_param_t"] = torch.tensor([0.5], dtype=torch.float32)
    tensors["_param_noise_scale"] = torch.tensor([0.15], dtype=torch.float32)
    tensors["_param_strengths"] = torch.tensor([1.0, 0.8, 1.0], dtype=torch.float32)
    # float32 for the int params too — flame-core's safetensors loader skips
    # int32/int64 tensors (serialization.rs:471). Cast to usize at call site.
    tensors["_param_frame_numbers"] = torch.tensor([0, 8, 16], dtype=torch.float32)
    tensors["_param_f_lats"] = torch.tensor([1, 1, 3], dtype=torch.float32)

    save_file(tensors, str(OUT))
    print(f"wrote {OUT}")
    for k, v in tensors.items():
        print(f"  {k:32s} {tuple(v.shape)}  {v.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
