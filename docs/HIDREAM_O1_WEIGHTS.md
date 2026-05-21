# HiDream-O1 — Weight Selection (Dev vs Full)

**TL;DR**

| Purpose | Checkpoint | Repo / Path |
|---|---|---|
| Inference (`hidream_o1_infer`) | **Dev** | `HiDream-O1-Image-Dev-weights` (≈33 GB) |
| Training (LoRA / full FT) | **Full** | `HiDream-ai/HiDream-O1-Image` on HF → `HiDream-O1-Image-Full-weights` (≈33 GB) |

**Do not train against Dev. Full strength LoRAs trained on Dev produce
mush-collapsed inference output.** This is documented behavior in
ai-toolkit's HiDream-O1 trainer node and confirmed empirically here on
2026-05-21 across both ai-toolkit and the EriDiffusion-v2 trainer.

---

## Why

Dev is step/CFG-distilled — its weights live on a tight manifold tuned for
low-step inference sampling. LoRA gradients computed against that manifold
update the model in directions that don't align with the inference
sampler's expectations, so when the LoRA is applied at full strength
(scale = α/rank with rank = α), the inference path catastrophically
diverges. Symptoms:

- 1024² dev sampler renders a foggy / over-smoothed blob with vague subject
  hints but no detail.
- Halving the LoRA-B scale at load time (`B × 0.5`) brings outputs back
  inside the safe manifold — but that's a band-aid, not a fix.

V67 Steampunk (a public LoRA that renders clean at full strength through
this inference path) was trained on **Full**. Two independent trainers
(ai-toolkit and ours, both on Dev) reproduce the failure on gigerver3 +
eri2 datasets at identical hyperparameters (rank=32, α=32, lr=1e-4,
adamw8bit, flowmatch shift, noise_scale=8). Switching to Full eliminates
the failure mode without changing anything else.

## Sources

- ai-toolkit / Saganaki22 HiDream_O1-ComfyUI: "Dev and Dev-2604 are
  intentionally not exposed in the training node because they are distilled
  models and may train unpredictably."
- RunComfy HiDream-O1 LoRA training notes: "Use the Full model for
  training; Dev is intentionally not exposed in the trainer because it is
  distilled and may train unpredictably."

## How

- **Inference**: nothing changes. `hidream_o1_infer` already points at the
  Dev directory and that's correct.
- **Training**: pass `--model-path /home/alex/HiDream-O1-Image-Full-weights`
  to `train_hidream_o1` (overrides the `DEFAULT_MODEL_PATH = ".../Dev-weights"`
  const in `crates/eridiffusion-cli/src/bin/train_hidream_o1.rs`).
- Resulting LoRA renders at `B × 1.0` through inference (Dev sampler) as
  intended — no need for an inference-side strength multiplier.

## File-system contract

```
/home/alex/HiDream-O1-Image-Dev-weights/   ← Dev,  inference
/home/alex/HiDream-O1-Image-Full-weights/  ← Full, training
```

Both directories use the same 8-shard `.safetensors` layout with the same
`config.json` / `chat_template.json` keys. Trainer and inference loader
share `inference_flame::models::hidream_o1::weight_loader`; the only
difference is the file contents.

## Related

- [HIDREAM_O1_KIJAI_PORT.md](HIDREAM_O1_KIJAI_PORT.md) — inference attention
  improvements ported from Kijai/ComfyUI.
