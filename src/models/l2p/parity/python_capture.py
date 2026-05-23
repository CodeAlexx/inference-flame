#!/usr/bin/env python3
"""L2P parity capture — Python reference forward + per-layer intermediate dump.

Phase A of the 4-phase parity test. Loads ZImageDiT, runs ONE forward pass
on a fixed input, captures named intermediates at the same boundaries used by
the Rust `forward_with_capture`, and writes them as F32 safetensors.

Constraints (per CONTEXT.md):
  - CUDA only; never CPU. PyTorch CPU vs CUDA BF16 diverges at cos=0.5/layer.
  - Capture stored F32 (so BF16 quant doesn't double-apply at compare time).
  - No flash-attn; force torch SDPA for determinism (env var below).

Usage:
  python3 python_capture.py --sigma 0.5 --out /tmp/l2p_parity/python_capture_sigma0.5.safetensors
  python3 python_capture.py --sigma 0.3 --out /tmp/l2p_parity/python_capture_sigma0.3.safetensors
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Force torch SDPA path (no flash-attn) for determinism.
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import torch
import safetensors.torch as st


REPO_ROOT = Path(__file__).resolve().parents[1] / "reference"
WEIGHTS_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
SAMPLE_PATH = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512/10.safetensors"


def _load_zimage_dit_module():
    """Bypass diffsynth package __init__ (which pulls in modelscope).

    Manually wire just the modules we need: core.attention and
    core.gradient, then load z_image_dit_L2P as a standalone module.
    """
    sys.path.insert(0, str(REPO_ROOT))
    # Pre-register the package skeleton so relative imports in z_image_dit_L2P
    # work (`from ..core.attention import attention_forward`).
    import types

    # Create stub package hierarchy
    diffsynth_pkg = types.ModuleType("diffsynth")
    diffsynth_pkg.__path__ = [str(REPO_ROOT / "diffsynth")]
    sys.modules["diffsynth"] = diffsynth_pkg

    core_pkg = types.ModuleType("diffsynth.core")
    core_pkg.__path__ = [str(REPO_ROOT / "diffsynth" / "core")]
    sys.modules["diffsynth.core"] = core_pkg

    models_pkg = types.ModuleType("diffsynth.models")
    models_pkg.__path__ = [str(REPO_ROOT / "diffsynth" / "models")]
    sys.modules["diffsynth.models"] = models_pkg

    # Load core.attention and core.gradient as real submodules.
    def load_submod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    attn_path = REPO_ROOT / "diffsynth" / "core" / "attention" / "attention.py"
    grad_path = REPO_ROOT / "diffsynth" / "core" / "gradient" / "gradient_checkpoint.py"

    attn_pkg = types.ModuleType("diffsynth.core.attention")
    attn_pkg.__path__ = [str(attn_path.parent)]
    sys.modules["diffsynth.core.attention"] = attn_pkg
    attn_mod = load_submod("diffsynth.core.attention.attention", attn_path)
    attn_pkg.attention_forward = attn_mod.attention_forward

    grad_pkg = types.ModuleType("diffsynth.core.gradient")
    grad_pkg.__path__ = [str(grad_path.parent)]
    sys.modules["diffsynth.core.gradient"] = grad_pkg
    grad_mod = load_submod("diffsynth.core.gradient.gradient_checkpoint", grad_path)
    grad_pkg.gradient_checkpoint_forward = grad_mod.gradient_checkpoint_forward

    # Now load the DiT model file.
    dit_path = REPO_ROOT / "diffsynth" / "models" / "z_image_dit_L2P.py"
    dit_mod = load_submod("diffsynth.models.z_image_dit_L2P", dit_path)
    return dit_mod


def install_capture_hooks(model, capture: dict):
    """Patch the model's forward to insert intermediate captures.

    We monkey-patch `forward` on the model to wrap the original. This is
    less invasive than swapping the whole forward — we lift the original,
    call the same sequence, but insert capture points.
    """
    # Patch the layers' forward to capture per-layer outputs.
    orig_block_forwards = {}

    def _make_block_hook(name, block):
        orig = block.forward

        def hooked_forward(*args, **kwargs):
            out = orig(*args, **kwargs)
            capture[name] = out.detach().to(torch.float32).contiguous().cpu()
            return out

        return hooked_forward

    for i, blk in enumerate(model.context_refiner):
        orig_block_forwards[("ctx", i)] = blk.forward
        blk.forward = _make_block_hook(f"context_refiner_{i}_out", blk)

    for i, blk in enumerate(model.noise_refiner):
        orig_block_forwards[("noise", i)] = blk.forward
        blk.forward = _make_block_hook(f"noise_refiner_{i}_out", blk)

    for i, blk in enumerate(model.layers):
        orig_block_forwards[("layer", i)] = blk.forward
        blk.forward = _make_block_hook(f"unified_after_layer_{i:02d}", blk)

    # Patch t_embedder forward to capture t_emb.
    orig_t = model.t_embedder.forward

    def t_hook(t):
        out = orig_t(t)
        capture["t_emb"] = out.detach().to(torch.float32).contiguous().cpu()
        return out

    model.t_embedder.forward = t_hook

    # Patch the x_embedder Linear and cap_embedder Sequential.
    x_emb_key = list(model.all_x_embedder.keys())[0]
    x_emb_module = model.all_x_embedder[x_emb_key]
    orig_x_emb_fwd = x_emb_module.forward

    def x_emb_hook(x):
        out = orig_x_emb_fwd(x)
        capture["x_after_embedder"] = out.detach().to(torch.float32).contiguous().cpu()
        return out

    x_emb_module.forward = x_emb_hook

    orig_cap_emb_fwd = model.cap_embedder.forward

    def cap_emb_hook(x):
        out = orig_cap_emb_fwd(x)
        capture["cap_after_embedder"] = out.detach().to(torch.float32).contiguous().cpu()
        return out

    model.cap_embedder.forward = cap_emb_hook

    # local_decoder hook for the pre-negation output.
    orig_ld_fwd = model.local_decoder.forward

    def ld_hook(*args, **kwargs):
        out = orig_ld_fwd(*args, **kwargs)
        capture["local_decoder_out"] = out.detach().to(torch.float32).contiguous().cpu()
        return out

    model.local_decoder.forward = ld_hook


def run_forward_with_capture(model, x_list, t_value, cap_list, capture):
    """Run the model forward, capturing unified_initial and feat_map by
    re-implementing the relevant slice of `forward` around the hooks already
    installed on the submodules.

    Why this nested approach: capturing unified_initial and feat_map requires
    intercepting AT specific points inside the forward that aren't natural
    module boundaries. We could re-trace the whole forward, but that risks
    drift from the Python reference. Instead, after the block-by-block
    captures fire, we reconstruct `unified_initial` and `feat_map` by hand
    from a separate model forward call setup.

    Simpler alternative: monkey-patch the `forward` method on the model
    class itself to add capture calls. That's what we do here.
    """
    # Patched forward — copy of the original with capture calls inserted.
    from torch.nn.utils.rnn import pad_sequence

    def patched_forward(self, x, t, cap_feats, patch_size=16, f_patch_size=1,
                        use_gradient_checkpointing=False,
                        use_gradient_checkpointing_offload=False):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)
        adaln_input = t

        (
            x_patches_flat_list,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        x_item_seqlens = [len(_) for _ in x_patches_flat_list]
        x_max_item_seqlen = max(x_item_seqlens)

        x_embed = torch.cat(x_patches_flat_list, dim=0)
        x_embed = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_embed)
        x_embed[torch.cat(x_inner_pad_mask)] = self.x_pad_token.to(
            dtype=x_embed.dtype, device=x_embed.device)
        x_embed = list(x_embed.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x_embed = pad_sequence(x_embed, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.noise_refiner:
            x_embed = layer(x=x_embed, attn_mask=x_attn_mask,
                            freqs_cis=x_freqs_cis, adaln_input=adaln_input)

        cap_item_seqlens = [len(_) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token.to(
            dtype=x_embed.dtype, device=x_embed.device)
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer(x=cap_feats, attn_mask=cap_attn_mask,
                              freqs_cis=cap_freqs_cis)

        # ---- unified ----
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len_i = cap_item_seqlens[i]
            unified.append(torch.cat([x_embed[i][:x_len], cap_feats[i][:cap_len_i]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len_i]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        # CAPTURE unified_initial
        capture["unified_initial"] = unified.detach().to(torch.float32).contiguous().cpu()

        for layer in self.layers:
            unified = layer(x=unified, attn_mask=unified_attn_mask,
                            freqs_cis=unified_freqs_cis, adaln_input=adaln_input)

        # ---- pixel decoding ----
        img_token_len = x_item_seqlens[0]
        img_features = unified[:, :img_token_len, :]

        F_ori, H_ori, W_ori = x_size[0]
        feat_H = H_ori // patch_size
        feat_W = W_ori // patch_size

        feat_map = img_features.view(bsz, feat_H, feat_W, self.dim).permute(0, 3, 1, 2)
        # CAPTURE feat_map (channel-first)
        capture["feat_map"] = feat_map.detach().to(torch.float32).contiguous().cpu()

        noisy_images = torch.stack(x, dim=0)
        if noisy_images.dim() == 5:
            noisy_images = noisy_images.squeeze(2)

        decoded_batch = self.local_decoder(noisy_images, feat_map)
        decoded_batch = decoded_batch.unsqueeze(2)
        x_final = list(decoded_batch.unbind(0))
        return x_final, {}

    import types as _types
    model.forward = _types.MethodType(patched_forward, model)
    with torch.no_grad():
        out, _ = model(x_list, t_value, cap_list)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, required=True,
                        help="Caller-side sigma in [0,1]. Inside Python, t*=t_scale=1000 directly. "
                             "For Rust parity at this exact value, Rust must apply (1-sigma)*1000 — "
                             "i.e. pass `1.0 - sigma_python` to the Rust binary.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--sample", type=str, default=SAMPLE_PATH)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (per CONTEXT.md: never CPU refs).")
    device = torch.device("cuda")

    print(f"[python-capture] loading reference module")
    dit_mod = _load_zimage_dit_module()
    ZImageDiT = dit_mod.ZImageDiT

    print(f"[python-capture] instantiating ZImageDiT")
    model = ZImageDiT()  # default L2P config

    print(f"[python-capture] loading weights from {args.weights}")
    weights = st.load_file(args.weights)
    # Mixed-precision file: cast everything to BF16 for consistent dtype.
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"[python-capture] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  first 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  first 5 unexpected: {unexpected[:5]}")

    model = model.to(device=device, dtype=torch.bfloat16).eval()

    print(f"[python-capture] loading fixed input from {args.sample}")
    sample = st.load_file(args.sample)
    pixel = sample["pixel"].to(device=device, dtype=torch.bfloat16)  # [3, 512, 512]
    cap_feats = sample["cap_feats"].to(device=device, dtype=torch.bfloat16)  # [1, N, 2560]

    # Python expects:
    #   x: List[Tensor[C, F, H, W]]  with F=1 for image
    #   cap_feats: List[Tensor[S, 2560]]
    x_4d = pixel.unsqueeze(1)  # [3, 1, 512, 512]
    x_list = [x_4d]
    cap_list = [cap_feats.squeeze(0)]  # [N, 2560]

    t_tensor = torch.tensor([args.sigma], device=device, dtype=torch.bfloat16)

    print(f"[python-capture] sigma={args.sigma}  t after t_scale={args.sigma * 1000.0}")

    capture = {}
    install_capture_hooks(model, capture)
    run_forward_with_capture(model, x_list, t_tensor, cap_list, capture)

    # Save capture (all F32 already).
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # safetensors wants contiguous tensors and same device set (cpu is fine).
    save_dict = {k: v.contiguous() for k, v in capture.items()}
    st.save_file(save_dict, args.out)
    print(f"[python-capture] wrote {len(save_dict)} tensors to {args.out}")
    for k in sorted(save_dict.keys()):
        t = save_dict[k]
        print(f"  {k}  shape={tuple(t.shape)}  dtype={t.dtype}  "
              f"abs.mean={t.abs().mean().item():.4e}  "
              f"abs.max={t.abs().max().item():.4e}")


if __name__ == "__main__":
    main()
