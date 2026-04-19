#!/usr/bin/env python
"""Per-substep dump of Qwen2.5-VL layer 0 for Rust↔Python parity.

Mirrors the keys produced by `Qwen25VLEncoder::layer0_substep_probe`:
    embed_out, normed_input, q_raw, k_raw, v_raw,
    q_heads, k_heads, v_heads, q_roped, k_roped,
    k_repeated, v_repeated, attn_sdpa, attn_merge, attn_o_out,
    after_attn, normed_post, gate_raw, up_raw, mlp_pre_down,
    mlp_out, layer_0_out, token_ids, attention_mask
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

PROMPT = "a photograph of an astronaut riding a horse on mars"
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
SNAP = Path("/home/alex/.serenity/models/checkpoints/qwen-image-2512")
OUT = Path("/home/alex/EriDiffusion/inference-flame/output/qwen25vl_layer0_substep.safetensors")
MAX_LEN = 1058


def main() -> int:
    tok = AutoTokenizer.from_pretrained(SNAP / "tokenizer")
    enc = tok(
        [PROMPT_TEMPLATE.format(PROMPT)],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    # Force eager attention so HF runs explicit matmul+softmax+matmul.
    # This lets our self-reproduced `attn_sdpa` match HF's internals
    # exactly — picking `sdpa` or `flash_attention_2` shifts the BF16
    # rounding in small but cosine-visible ways.
    model_full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        SNAP / "text_encoder",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model_full.eval()
    # transformers 4.57+: model_full.model is Qwen2_5_VLModel with .visual
    # and .language_model (Qwen2_5_VLTextModel). Older versions put layers on
    # model_full.model directly. Handle both.
    lm = getattr(model_full.model, "language_model", None) or model_full.model
    layer0 = lm.layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp

    dumps: dict[str, torch.Tensor] = {}

    # Monkey-patch eager_attention_forward AND apply_multimodal_rotary_pos_emb
    # for layer 0 so we can record pre-RoPE q/k, post-RoPE q/k, attn inputs,
    # and attn output.
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _m
    original_eager = _m.eager_attention_forward
    original_apply_rope = _m.apply_multimodal_rotary_pos_emb
    hf_trace: dict[str, torch.Tensor] = {}
    captured_eager = {"done": False}
    captured_rope = {"done": False}

    def traced_apply_rope(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        if not captured_rope["done"]:
            hf_trace["hf_q_pre_rope"] = q.detach().contiguous()
            hf_trace["hf_k_pre_rope"] = k.detach().contiguous()
            hf_trace["hf_cos_raw"] = cos.detach().to(torch.float32).contiguous()
            hf_trace["hf_sin_raw"] = sin.detach().to(torch.float32).contiguous()
        q_out, k_out = original_apply_rope(q, k, cos, sin, mrope_section, unsqueeze_dim=unsqueeze_dim)
        if not captured_rope["done"]:
            hf_trace["hf_q_roped_inner"] = q_out.detach().contiguous()
            hf_trace["hf_k_roped_inner"] = k_out.detach().contiguous()
            captured_rope["done"] = True
        return q_out, k_out

    def traced_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out = original_eager(module, query, key, value, attention_mask, scaling=scaling,
                             dropout=dropout, **kwargs)
        if not captured_eager["done"]:
            hf_trace["hf_q_roped"] = query.detach().contiguous()
            hf_trace["hf_k_roped"] = key.detach().contiguous()
            hf_trace["hf_v_heads"] = value.detach().contiguous()
            if attention_mask is not None:
                hf_trace["hf_attention_mask"] = attention_mask.detach().to(torch.float32).contiguous()
            hf_trace["hf_attn_out_bqhd"] = out[0].detach().contiguous()
            captured_eager["done"] = True
        return out

    _m.eager_attention_forward = traced_eager
    _m.apply_multimodal_rotary_pos_emb = traced_apply_rope

    def save(name: str):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            dumps[name] = t.detach().contiguous()
        return _hook

    # Input/output hooks on every submodule we care about.
    h_in_ln = layer0.input_layernorm.register_forward_hook(save("normed_input"))
    h_q = attn.q_proj.register_forward_hook(save("q_raw"))
    h_k = attn.k_proj.register_forward_hook(save("k_raw"))
    h_v = attn.v_proj.register_forward_hook(save("v_raw"))
    h_o = attn.o_proj.register_forward_hook(save("attn_o_out"))
    h_post_ln = layer0.post_attention_layernorm.register_forward_hook(save("normed_post"))
    h_gate = mlp.gate_proj.register_forward_hook(save("gate_raw"))
    h_up = mlp.up_proj.register_forward_hook(save("up_raw"))
    h_down = mlp.down_proj.register_forward_hook(save("mlp_out"))

    # Capture the attention output (post-sdpa, post-merge, pre-o_proj) by
    # hooking the input to o_proj.
    def _hook_o_input(_mod, inp):
        dumps["attn_merge"] = inp[0].detach().contiguous()
    h_o_in = attn.o_proj.register_forward_pre_hook(_hook_o_input)

    # Capture down_proj's input = silu(gate) * up.
    def _hook_down_input(_mod, inp):
        dumps["mlp_pre_down"] = inp[0].detach().contiguous()
    h_down_in = mlp.down_proj.register_forward_pre_hook(_hook_down_input)

    with torch.no_grad():
        out = model_full(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True,
        )

    for h in (h_in_ln, h_q, h_k, h_v, h_o, h_post_ln, h_gate, h_up, h_down, h_o_in, h_down_in):
        h.remove()

    _m.eager_attention_forward = original_eager
    _m.apply_multimodal_rotary_pos_emb = original_apply_rope
    print(f"[dump] hf_trace keys: {list(hf_trace.keys())}", flush=True)
    dumps.update(hf_trace)

    hs = out.hidden_states
    dumps["embed_out"] = hs[0].detach().contiguous()
    dumps["layer_0_out"] = hs[1].detach().contiguous()

    # `after_attn` = layer_0_out - mlp_out (the MLP residual sum)
    # but we can reconstruct it exactly because both are captured:
    # hs[1] = after_attn + mlp_out, so after_attn = hs[1] - mlp_out.
    # Even simpler — dump the residual branch directly by re-running
    # the attention once with hooks that compute it.
    dumps["after_attn"] = (dumps["layer_0_out"] - dumps["mlp_out"]).contiguous()

    # Derive q_heads, k_heads, v_heads from q_raw, k_raw, v_raw using
    # the same reshape the model internally uses (see HF modeling file).
    cfg = model_full.config
    # Qwen2_5_VLForConditionalGeneration.config has a sub-config for text.
    # Pull hidden_size / num_heads / num_kv_heads from the text config.
    text_cfg = getattr(cfg, "text_config", None) or cfg
    num_heads = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    hidden = text_cfg.hidden_size
    head_dim = getattr(text_cfg, "head_dim", None) or hidden // num_heads

    bsz, slen, _ = dumps["q_raw"].shape

    def reshape_heads(t: torch.Tensor, nh: int) -> torch.Tensor:
        return t.view(bsz, slen, nh, head_dim).transpose(1, 2).contiguous()

    dumps["q_heads"] = reshape_heads(dumps["q_raw"], num_heads)
    dumps["k_heads"] = reshape_heads(dumps["k_raw"], num_kv_heads)
    dumps["v_heads"] = reshape_heads(dumps["v_raw"], num_kv_heads)

    # q_roped / k_roped / k_repeated / v_repeated / attn_sdpa require
    # re-running the attention with the same rotary + mask. Easiest:
    # bypass HF and compute it ourselves using model cos/sin via forward
    # instrumentation. Instead, re-use the rotary_emb on `lm.rotary_emb`
    # and apply_multimodal_rotary_pos_emb to match the HF formula exactly.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    # Build position_ids the way Qwen2_5_VLModel.get_rope_index does for
    # text-only prompts: cumsum(attention_mask)-1, pad slots set to 1,
    # expanded across the 3 M-RoPE axes.
    attn_mask_t = enc.attention_mask.to("cuda")
    _pos_1d = attn_mask_t.long().cumsum(-1) - 1
    _pos_1d = _pos_1d.masked_fill(attn_mask_t == 0, 1)
    pos = _pos_1d.unsqueeze(0).expand(3, -1, -1).contiguous()
    # rotary_emb lives on lm.rotary_emb in modern transformers, or
    # attn.rotary_emb for older versions.
    rotary = getattr(lm, "rotary_emb", None) or attn.rotary_emb
    cos, sin = rotary(dumps["v_heads"], pos)  # [3, B, seq, head_dim]

    mrope_section = text_cfg.rope_scaling.get("mrope_section") if getattr(text_cfg, "rope_scaling", None) else None
    if mrope_section is None:
        mrope_section = [16, 24, 24]  # Qwen2.5-VL default

    q_roped, k_roped = apply_multimodal_rotary_pos_emb(
        dumps["q_heads"], dumps["k_heads"], cos, sin, mrope_section
    )
    dumps["q_roped"] = q_roped.contiguous()
    dumps["k_roped"] = k_roped.contiguous()

    n_rep = num_heads // num_kv_heads
    dumps["k_repeated"] = repeat_kv(dumps["k_roped"], n_rep).contiguous()
    dumps["v_repeated"] = repeat_kv(dumps["v_heads"], n_rep).contiguous()

    # sdpa against the HF causal+padding mask. Use the same scaled_dot_product
    # SDPA that HF uses internally (torch.nn.functional.scaled_dot_product_attention
    # with attn_mask=None, is_causal=True would work if no pad; with pad, need
    # explicit mask matching HF's _prepare_4d_causal_attention_mask).
    # For parity against our Rust build_causal_mask, build it the same way:
    real_len = int(enc.attention_mask[0].sum().item())
    mask = torch.full((slen, slen), -1e4, dtype=torch.bfloat16, device="cuda")
    idx_i = torch.arange(slen, device="cuda").unsqueeze(1)
    idx_j = torch.arange(slen, device="cuda").unsqueeze(0)
    mask[(idx_j <= idx_i) & (idx_j < real_len)] = 0.0
    mask = mask.view(1, 1, slen, slen)

    attn_sdpa = torch.nn.functional.scaled_dot_product_attention(
        dumps["q_roped"], dumps["k_repeated"], dumps["v_repeated"],
        attn_mask=mask, is_causal=False, dropout_p=0.0,
    )
    dumps["attn_sdpa"] = attn_sdpa.contiguous()

    # Save token_ids and attention_mask as f32 (flame skips integer dtypes).
    dumps["token_ids"] = enc.input_ids[0].to(torch.float32).cpu()
    dumps["attention_mask"] = enc.attention_mask[0].to(torch.float32).cpu()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in dumps.items()}, OUT)

    print(f"wrote {OUT}")
    for k, v in sorted(dumps.items()):
        print(f"  {k:<16} {tuple(v.shape)} {v.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
